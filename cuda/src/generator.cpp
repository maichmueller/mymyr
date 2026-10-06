// ChunkGenerator: the host driver of the lifted successor kernels (include/mymyr/cuda/generator.hpp).

#include "mymyr/cuda/generator.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mymyr::cuda
{
void* Scratch::ensure(const ContextPtr& ctx, u64 bytes, cudaStream_t s)
{
    bytes = std::max<u64>(bytes, 64);
    if (m_buf.size() >= bytes && m_buf.stream() == s)
        return m_buf.data();
    DeviceBuffer nb(ctx, std::max(bytes, m_buf.size() + m_buf.size() / 2), s);
    if (m_last)
        nb.track(m_last);
    m_buf.reset(s);  // (on the current call's stream: its allocation stream may be gone)
    m_buf = std::move(nb);
    return m_buf.data();
}

void Scratch::track(std::shared_ptr<const LastUse> last)
{
    m_last = std::move(last);
    if (m_buf.data())
        m_buf.track(m_last);
}

namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

/// Goal flags of the rows are computed in chunks of at most this many bytes of views, derived bitsets and rows.
constexpr u64 k_goal_budget = u64{64} << 20;

/// Timed axiom launches between two folds of their events (each CUDA event holds host memory).
constexpr usize k_axiom_events = 256;

/// Parents per slice at least, for the CPU-fallback work of a chunk on the host's threads.
constexpr u64 k_host_slice = 64;

template<class T>
void upload_vec(const ContextPtr& ctx, Scratch& dst, const std::vector<T>& v, cudaStream_t s)
{
    void* d = dst.ensure(ctx, v.size() * sizeof(T), s);
    if (!v.empty())
        check(cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (chunk upload)");
}

/// Why the axioms cannot run on the device (empty: they can).
std::string axioms_reason(const plan::Compiled& C)
{
    for (const plan::Stratum& st : C.strata)
        for (const plan::Axiom& x : st.axioms)
        {
            const plan::Matcher& m = x.body;
            if (m.total > lifted::k_max_depth || m.steps.size() > lifted::k_max_depth)
                return "an axiom body of " + std::to_string(m.total) + " parameters (the device kernels take at most " +
                       std::to_string(lifted::k_max_depth) + ")";
            if (!m.npre.empty() || !m.nchecks.empty())
                return "numeric constraints in an axiom body";
        }
    return {};
}

/// Per axiom in plan_axiom order (strata, then their axioms): {the predicates of its stratum its body reads, as bits of
/// their index in the stratum's head list; the bit of its head} (lifted::Derived::reads). All bits when a stratum has
/// more than 64 head predicates.
std::vector<u64> axiom_reads(const plan::Compiled& C)
{
    constexpr u32 none = ViewLayout::k_none;
    const ViewLayout& V = C.view;
    std::vector<u32> pred_of_row(V.rows, none);
    for (u32 p = 0; p < V.unary_row.size(); ++p)
    {
        if (V.unary_row[p] != none && V.unary_row[p] < V.rows)
            pred_of_row[V.unary_row[p]] = p;
        for (const u32 first : {V.fwd_row[p], V.bwd_row[p]})
            if (first != none)
                for (u32 o = 0; o < V.num_objects && first + o < V.rows; ++o)
                    pred_of_row[first + o] = p;
    }
    std::vector<u64> out;
    for (const plan::Stratum& st : C.strata)
    {
        std::vector<u32> heads;
        for (const plan::Axiom& x : st.axioms)
            if (std::find(heads.begin(), heads.end(), x.head.pred) == heads.end())
                heads.push_back(x.head.pred);
        const bool fits = heads.size() <= 64;
        auto bit = [&](u32 p) -> u64
        {
            const auto it = std::find(heads.begin(), heads.end(), p);
            return it == heads.end() ? 0 : u64{1} << (it - heads.begin());
        };
        for (const plan::Axiom& x : st.axioms)
        {
            const plan::Matcher& m = x.body;
            u64 reads = 0;
            auto ref = [&](plan::TableRef r)
            {
                if (!r.is_view())
                    return;
                if (r.value() < pred_of_row.size() && pred_of_row[r.value()] != none)
                    reads |= bit(pred_of_row[r.value()]);
                else
                    reads = ~u64{0};  // a view row of no known predicate: it may read anything
            };
            for (const std::vector<plan::Check>* cs : {&m.pre_checks, &m.checks})
                for (const plan::Check& c : *cs)
                    if (c.pat.kind == plan::LitKind::Derived)
                        reads |= bit(c.pat.pred);
            for (const plan::Unary& u : m.unary)
                ref(u.ptr);
            for (const plan::Row& r : m.rows)
                ref(r.base);
            for (const plan::Edge& e : m.fc_out)
                ref(e.base);
            for (const plan::Row& r : m.fc_pre)
                ref(r.base);
            out.push_back(fits ? reads : ~u64{0});
            out.push_back(fits ? bit(x.head.pred) : ~u64{0});
        }
    }
    return out;
}
}  // namespace

std::string ChunkGenerator::unsupported(const Task& task)
{
    const plan::Compiled& C = task.compiled();
    // numeric states carry values after the atom words; the kernels and the device format read atoms only
    if (task.numeric_slots() > 0)
        return "numeric fluents (the device format has no numeric section yet)";
    if (C.ow > lifted::k_max_ow)
        return "object bitsets of " + std::to_string(C.ow) + " words (the device kernels take at most " +
               std::to_string(lifted::k_max_ow) + ", i.e. " + std::to_string(64 * lifted::k_max_ow) + " objects)";
    if (task.atoms().mode() == AtomMode::Frozen && task.words() > lifted::k_max_words)
        return "states of " + std::to_string(task.words()) + " words (the device kernels take at most " +
               std::to_string(lifted::k_max_words) + ")";
    return {};
}

namespace
{
/// Whether schema ps runs (with matcher m) in the deep kernels.
bool deep_matcher(const plan::Schema& ps, const plan::Matcher& m)
{
    return m.total > lifted::k_max_depth || m.steps.size() > lifted::k_max_depth || ps.arity > lifted::k_max_label;
}
}  // namespace

std::string ChunkGenerator::host_reason(const Task& task, u32 schema, bool witness)
{
    const plan::Compiled& C = task.compiled();
    const plan::Schema& ps = C.schemas.at(schema);
    const plan::Matcher& m = ps.pre[witness ? 0 : 1];
    if (m.total > lifted::k_deep_depth || m.steps.size() > lifted::k_deep_depth)
        return "a matcher of " + std::to_string(m.total) + " parameters (the device kernels take at most " +
               std::to_string(lifted::k_deep_depth) + ")";
    if (ps.arity > lifted::k_deep_depth)
        return "arity " + std::to_string(ps.arity) + " (the device labels take at most " + std::to_string(lifted::k_deep_depth) + ")";
    if (deep_matcher(ps, m))
    {
        const std::string deep = "a matcher deeper than " + std::to_string(lifted::k_max_depth) + " parameters ";
        if (C.ow > lifted::k_deep_ow)
            return deep + "over object bitsets of " + std::to_string(C.ow) + " words (the deep kernels take at most " +
                   std::to_string(lifted::k_deep_ow) + ")";
        if (!ps.ces.empty())
            return deep + "with conditional effects";
    }
    if (m.use_fc && C.ow > lifted::k_max_fc_ow)
        return "forward checking over object bitsets of " + std::to_string(C.ow) + " words (the device kernels take at most " +
               std::to_string(lifted::k_max_fc_ow) + ")";
    if (ps.ces.empty())
        return {};
    if (ps.bind_size > lifted::k_max_depth)
        return "conditional effects binding " + std::to_string(ps.bind_size) + " parameters (the device kernels take at most " +
               std::to_string(lifted::k_max_depth) + ")";
    for (const plan::CondEffect& ce : ps.ces)
    {
        if (ce.numeric)
            return "numeric conditional effects";
        if (ce.cond.steps.size() > lifted::k_ce_depth)
            return "a conditional effect with " + std::to_string(ce.cond.steps.size()) +
                   " forall parameters to bind (the device sub-plans take at most " + std::to_string(lifted::k_ce_depth) + ")";
        if (ce.cond.total > lifted::k_max_depth)
            return "a conditional effect binding " + std::to_string(ce.cond.total) + " parameters (the device kernels take at most " +
                   std::to_string(lifted::k_max_depth) + ")";
    }
    return {};
}

SchemaPlacement ChunkGenerator::place(const Task& task, bool witness)
{
    const plan::Compiled& C = task.compiled();
    const u32 S = static_cast<u32>(C.schemas.size());
    SchemaPlacement pl;
    pl.host.assign(S, 0);
    pl.ce.assign(S, 0);
    pl.deep.assign(S, 0);
    for (u32 s = 0; s < S; ++s)
    {
        const plan::Schema& ps = C.schemas[s];
        const plan::Matcher& m = ps.pre[witness ? 0 : 1];
        const bool ce = !ps.ces.empty();
        if (!host_reason(task, s, witness).empty())
        {
            pl.host[s] = 1;
            ++pl.host_count;
            pl.host_ce_count += ce ? 1 : 0;
            continue;
        }
        pl.ce[s] = ce ? 1 : 0;
        pl.ce_count += ce ? 1 : 0;
        pl.deep[s] = deep_matcher(ps, m) ? 1 : 0;
        pl.deep_count += pl.deep[s];
        if (!m.never)
            pl.device.push_back(s);
    }
    return pl;
}

ChunkGenerator::ChunkGenerator(ContextPtr ctx, TaskPtr task, cudaStream_t stream)
    : m_ctx(std::move(ctx)), m_task(std::move(task)), m_s(stream ? stream : m_ctx->stream()),
      m_flag(m_ctx ? m_ctx->lease_pinned(1, sizeof(u32)) : PinnedLease{})
{
    if (!m_ctx || !m_task)
        throw std::invalid_argument("mymyr: ChunkGenerator: null context or task");
    if (const std::string why = unsupported(*m_task); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run this task: " + why);
    DeviceGuard g(m_ctx->device());
    m_last = std::make_shared<LastUse>(m_ctx->device());
    for (Scratch* x : {&m_dev_derived, &m_goal_views, &m_goal_derived, &m_goal_rows, &m_views, &m_counts, &m_offsets, &m_scan,
                       &m_derived, &m_schema_sets, &m_put_mask, &m_deep_work, &m_missing, &m_h_seg, &m_h_rank, &m_h_bind, &m_h_words, &m_h_counts_seg,
                       &m_h_counts})
        x->track(m_last);
    const plan::Compiled& C = m_task->compiled();
    m_S = static_cast<u32>(C.schemas.size());
    for (const plan::Schema& ps : C.schemas)
        m_L = std::max(m_L, ps.arity);
    // where each schema runs, per witness setting (the device lists follow the export's matcher flags: build_lists)
    for (u32 wi = 0; wi < 2; ++wi)
        m_place[wi] = place(*m_task, wi == 0);
    // axioms: on the device unless a body matcher is beyond the kernels
    std::vector<u64> reads;
    if (m_task->has_axioms())
    {
        m_host_axioms_reason = axioms_reason(C);
        m_device_axioms = m_host_axioms_reason.empty();
        if (m_device_axioms)
        {
            reads = axiom_reads(C);
            // flat strata: no body reads a head predicate of its own stratum (lifted::Strata)
            u32 at = 0;
            for (const plan::Stratum& st : C.strata)
            {
                const auto n = static_cast<u32>(st.axioms.size());
                bool flat = true;
                for (u32 k = 0; k < n; ++k)
                    flat = flat && reads[2 * (u64{at} + k)] == 0;
                m_strata_begin.push_back(at);
                m_strata_count.push_back(n);
                m_strata_flat.push_back(flat ? 1 : 0);
                at += n;
            }
            m_axiom_reads = DeviceBuffer(m_ctx, reads.size() * sizeof(u64), m_s);
            check(cudaMemcpyAsync(m_axiom_reads.data(), reads.data(), reads.size() * sizeof(u64), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (axiom reads)");
        }
    }
    upload();
    refresh();
    m_view_words = u64{m_view.view_rows} * m_view.ow;  // the export's compact view (a function of the compiled plans)
    if (m_task->atoms().mode() == AtomMode::Lazy)
    {
        m_missing_words = (m_view.atom_total + 31) / 32;
        void* p = m_missing.ensure(m_ctx, (m_missing_words + 1) * sizeof(u32), m_s);
        check(cudaMemsetAsync(p, 0, (m_missing_words + 1) * sizeof(u32), m_s), "cudaMemsetAsync");
    }
    check(cudaStreamSynchronize(m_s), "cudaStreamSynchronize");  // the host vectors above may go
}

ChunkGenerator::~ChunkGenerator() = default;

void ChunkGenerator::upload()
{
    auto bundle = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*m_task, rl::k_device_arrays_version));
    m_exported_fluent = static_cast<u32>(bundle->scalar("plan_fluent_slots", 0));
    m_exported_derived = static_cast<u32>(bundle->scalar("plan_derived_slots", 0));
    m_dt = DeviceTask::upload(m_ctx, std::move(bundle));  // the previous upload is freed after this stream's work
    m_view = m_dt->acquire(m_s);
    ++m_uploads;
    build_lists();
}

void ChunkGenerator::build_lists()
{
    const rl::dev::TaskView t = rl::task_view(m_dt->bundle());
    std::vector<u32> sets;
    std::vector<u8> put(2 * u64{m_S}, 0);
    for (u32 wi = 0; wi < 2; ++wi)
    {
        const SchemaPlacement& pl = m_place[wi];
        m_sorts[wi] = false;
        // the device schemas without conditional effects write their successors in launch_put (write())
        for (const u32 s : pl.device)
            put[wi * u64{m_S} + s] = pl.ce[s] ? 0 : 1;
        for (u32 oi = 0; oi < 3; ++oi)
        {
            const auto order = static_cast<rl::dev::MatchOrder>(oi);
            std::vector<u32> lists[6];
            for (const u32 s : pl.device)
            {
                const u32 flags = rl::dev::schema_matcher_flags(t, s, wi == 0);
                const bool fc = rl::dev::device_fc(flags, order);
                lists[pl.deep[s] ? 4 + (fc ? 1 : 0) : (fc ? 2 : 0) + (pl.ce[s] ? 1 : 0)].push_back(s);
                // the deep kernels sort every segment in canonical order (their lanes find the bindings in any order)
                if (order != rl::dev::MatchOrder::Matcher &&
                    (pl.deep[s] || (t.schema[u64{s} * rl::dev::k_sc_count + rl::dev::k_sc_arity] > 1 && rl::dev::device_sorts(flags, fc))))
                    m_sorts[wi] = true;
            }
            Lists& L = m_lists[wi][oi];
            L.at = sets.size();
            L.fixed = static_cast<u32>(lists[0].size());
            L.fixed_ce = static_cast<u32>(lists[1].size());
            L.fc = static_cast<u32>(lists[2].size());
            L.fc_ce = static_cast<u32>(lists[3].size());
            L.deep = static_cast<u32>(lists[4].size());
            L.deep_fc = static_cast<u32>(lists[5].size());
            for (const auto& l : lists)
                sets.insert(sets.end(), l.begin(), l.end());
        }
    }
    upload_vec(m_ctx, m_schema_sets, sets, m_s);
    upload_vec(m_ctx, m_put_mask, put, m_s);
    // the axiom bodies close a set of derived atoms: any order (the kernels read k_mc_device_fc per body)
    m_axiom_fc = false;
    for (u32 a = 0; a < t.n_axioms; ++a)
        m_axiom_fc = m_axiom_fc || rl::dev::device_fc(rl::dev::axiom_matcher_flags(t, a), rl::dev::MatchOrder::Free);
}

void ChunkGenerator::set_stream(cudaStream_t s)
{
    s = s ? s : m_ctx->stream();
    if (s == m_s)
        return;
    DeviceGuard g(m_ctx->device());
    // the scratch (reallocated on the new stream as it is next needed) and the missing bitmap
    if (m_last->recorded())
        m_last->wait_on(s);
    else
        stream_wait(s, m_s);
    m_s = s;
    m_view = m_dt->acquire(m_s);
}

void ChunkGenerator::mark_last_use()
{
    m_last->record(m_s);
    if (m_s != m_ctx->stream())
        m_dt->release(m_s, m_last);  // (the task block's free waits for the mark, not for this stream)
}

bool ChunkGenerator::refresh()
{
    bool any = false;
    // the export interns the goal's atoms under lazy slots, so one refresh may call for another
    for (int i = 0; i < 4; ++i)
    {
        const AtomIndex& a = m_task->atoms();
        if (a.fluent_slots() == m_exported_fluent && a.derived_slots() == m_exported_derived)
            break;
        DeviceGuard g(m_ctx->device());
        upload();
        any = true;
    }
    return any;
}

void ChunkGenerator::host_work(const ChunkInput& in, bool witness, bool canonical)
{
    m_host_segment.clear();
    m_host_rank.clear();
    m_host_binding.clear();
    m_host_need.clear();
    m_host_exact.clear();
    m_host_words.clear();
    m_host_count_seg.clear();
    m_host_count.clear();
    m_host_row_words = 0;
    if (!needs_host(witness))
        return;
    if (!in.host_states && in.rows)
        throw std::logic_error("mymyr: ChunkGenerator: this task needs the parents on the host");
    const bool axioms = m_task->has_axioms();
    const bool host_derived = axioms && !m_device_axioms;
    const SchemaPlacement& pl = placement(witness);
    const u32 DW = std::max<u32>(1, m_task->workspace().successors().engine().derived_words());
    if (host_derived)
        m_host_derived.assign(static_cast<usize>(in.rows) * DW, 0);
    struct Row
    {
        u32 seg, rank, need, exact;
        usize words_at;
    };
    // the parents in slices on the host threads, each with its own workspace. Under frozen slots generation
    // interns nothing, so the slices are independent and their rows are concatenated in parent order (the rows of one
    // thread); lazy slots intern atoms in generation order: one thread.
    struct Part
    {
        std::vector<Row> rows;
        std::vector<u32> bind_rows, count_seg, count;
        std::vector<u64> words;
        double axiom_ms = 0;
    };
    const u32 n = m_task->atoms().mode() == AtomMode::Lazy
                      ? 1
                      : static_cast<u32>(std::clamp<u64>(in.rows / k_host_slice, 1, u64{8} * m_host_threads));
    const u32 T = std::min(n, m_host_threads);
    std::vector<Part> parts(n);
    auto work = [&](u32 t)
    {
        const auto [lo, hi] = ThreadPool::slice(in.rows, t, n);
        Part& part = parts[t];
        Successors& succ = m_task->workspace().successors();  // this thread's
        detail::Engine& e = succ.engine();
        std::vector<u64> tmp;
        for (u64 i = lo; i < hi; ++i)
        {
            const u64* row = in.host_states + i * in.host_stride;
            const u32 nw = bits::trimmed_size(row, in.words);
            // Successors::prepare, with the CPU's axiom time measured apart
            e.set_state(row, nw);
            e.set_numeric(nullptr);
            e.build_view();
            if (axioms)
            {
                const auto t0 = Clock::now();
                succ.axioms().evaluate();
                part.axiom_ms += ms_since(t0);
            }
            if (host_derived)
                std::copy_n(e.derived(), std::min(e.derived_words(), DW), m_host_derived.begin() + static_cast<std::ptrdiff_t>(i * DW));
            if (pl.host_count == 0)
                continue;
            u32 last_seg = ~u32{0}, rank = 0;
            succ.set_schema_filter(pl.host.data());
            try
            {
                succ.generate<false>(
                    [&](u32 s, const ObjectId* b, const Delta& d)
                    {
                        const u32 seg = static_cast<u32>(i) * m_S + s;
                        if (seg != last_seg)
                        {
                            if (last_seg != ~u32{0})
                            {
                                part.count_seg.push_back(last_seg);
                                part.count.push_back(rank);
                            }
                            last_seg = seg;
                            rank = 0;
                        }
                        u32 max_add = 0;
                        for (SlotId a : d.add)
                            max_add = std::max(max_add, bits::word_of(a.v) + 1);
                        const u32 n = apply_delta(row, nw, d, tmp);
                        part.rows.push_back({seg, rank++, std::max(nw, max_add), n, part.words.size()});
                        part.words.insert(part.words.end(), tmp.begin(), tmp.begin() + n);
                        const u32 arity = succ.arity(s);
                        for (u32 k = 0; k < m_L; ++k)
                            part.bind_rows.push_back(k < arity ? b[k].v : 0xFFFFFFFFu);
                        return true;
                    },
                    witness, canonical);
            }
            catch (...)
            {
                succ.set_schema_filter(nullptr);
                throw;
            }
            succ.set_schema_filter(nullptr);
            if (last_seg != ~u32{0})
            {
                part.count_seg.push_back(last_seg);
                part.count.push_back(rank);
            }
        }
    };
    if (T == 1)
        for (u32 t = 0; t < n; ++t)
            work(t);
    else
    {
        // the slices to the threads as they finish (the parents' successor counts vary)
        if (!m_pool)
            m_pool = std::make_unique<ThreadPool>(m_host_threads);
        std::atomic<u32> next{0};
        m_pool->run(
            [&](u32 t)
            {
                if (t < T)
                    for (u32 k; (k = next.fetch_add(1, std::memory_order_relaxed)) < n;)
                        work(k);
            });
    }
    // rows as arrays (in parent order: the slices in order); words at the widest successor's width
    u64 nrows = 0;
    for (const Part& part : parts)
    {
        nrows += part.rows.size();
        m_stats.host_axiom_ms += part.axiom_ms;
        for (const Row& r : part.rows)
            m_host_row_words = std::max(m_host_row_words, r.exact);
        m_host_count_seg.insert(m_host_count_seg.end(), part.count_seg.begin(), part.count_seg.end());
        m_host_count.insert(m_host_count.end(), part.count.begin(), part.count.end());
        m_host_binding.insert(m_host_binding.end(), part.bind_rows.begin(), part.bind_rows.end());
    }
    m_host_row_words = std::max<u32>(m_host_row_words, 1);
    m_host_words.assign(nrows * m_host_row_words, 0);
    u64 k = 0;
    for (const Part& part : parts)
        for (const Row& r : part.rows)
        {
            m_host_segment.push_back(r.seg);
            m_host_rank.push_back(r.rank);
            m_host_need.push_back(r.need);
            m_host_exact.push_back(r.exact);
            std::copy_n(part.words.begin() + static_cast<std::ptrdiff_t>(r.words_at), r.exact,
                        m_host_words.begin() + static_cast<std::ptrdiff_t>(k * m_host_row_words));
            ++k;
        }
    m_stats.host_rows += nrows;
}

bool ChunkGenerator::can_defer_host() const noexcept
{
    return m_task->atoms().mode() != AtomMode::Lazy && !(m_task->has_axioms() && !m_device_axioms);
}

void ChunkGenerator::begin_device(const ChunkInput& in, bool witness, bool canonical)
{
    DeviceGuard g(m_ctx->device());
    if (!can_defer_host())
        throw std::logic_error("mymyr: ChunkGenerator::begin_device: lazy slots or CPU axioms need the host work first (begin)");
    if (in.rows_dev && needs_host(witness))
        throw std::logic_error("mymyr: ChunkGenerator::begin_device: a device row count with CPU-fallback work (the host needs the rows)");
    m_witness = witness;
    m_canonical = canonical;
    m_deferred = in;
    m_host_pending = true;
    m_parents = lifted::Parents{in.states, in.stride, in.words, in.rows, nullptr, 0};
    m_parents.rows_dev = in.rows_dev;
}

void ChunkGenerator::host_work()
{
    if (!m_host_pending)
        throw std::logic_error("mymyr: ChunkGenerator::host_work: no chunk from begin_device()");
    host_work(m_deferred, m_witness, m_canonical);
    m_host_pending = false;
}

void ChunkGenerator::begin(const ChunkInput& in, bool witness, bool canonical)
{
    DeviceGuard g(m_ctx->device());
    if (in.rows_dev && needs_host(witness))
        throw std::logic_error("mymyr: ChunkGenerator::begin: a device row count with CPU-fallback work (the host needs the rows)");
    m_witness = witness;
    m_canonical = canonical;
    m_host_pending = false;
    host_work(in, witness, canonical);
    refresh();
    m_parents = lifted::Parents{in.states, in.stride, in.words, in.rows, nullptr, 0};
    m_parents.rows_dev = in.rows_dev;
    if (m_task->has_axioms() && !m_device_axioms)
    {
        const u32 DW = std::max<u32>(1, m_task->workspace().successors().engine().derived_words());
        upload_vec(m_ctx, m_derived, m_host_derived, m_s);
        m_parents.derived = static_cast<const u64*>(m_derived.data());
        m_parents.derived_words = DW;
    }
}

u32 ChunkGenerator::views_and_axioms(const lifted::Parents& p, u64* views, Scratch& derived, u64** derived_out)
{
    const lifted::Views v{views, m_view_words};
    for (;;)
    {
        check(lifted::launch_view(m_view, p, v, m_s), "launch_view");
        if (!m_device_axioms)
        {
            *derived_out = nullptr;
            return 0;
        }
        const u32 dw = std::max<u32>(1, bits::words_for(m_view.derived_slots));
        auto* der = static_cast<u64*>(derived.ensure(m_ctx, std::max<u64>(p.rows, 1) * dw * sizeof(u64), m_s));
        lifted::Derived d;
        d.data = der;
        d.words = dw;
        d.reads = static_cast<const u64*>(m_axiom_reads.data());
        if (m_missing_words)
        {
            d.missing_bits = static_cast<u32*>(m_missing.data());
            d.missing_flag = d.missing_bits + m_missing_words;
        }
        Event* ev = nullptr;
        if (m_timing)
        {
            if (m_axiom_event_next >= 2 * k_axiom_events)
                fold_axiom_events();
            while (m_axiom_events.size() < m_axiom_event_next + 2)
                m_axiom_events.emplace_back(true);
            ev = &m_axiom_events[m_axiom_event_next];
            m_axiom_event_next += 2;
            ev[0].record(m_s);
        }
        const lifted::Strata strata{m_strata_begin.data(), m_strata_count.data(), m_strata_flat.data(),
                                    static_cast<u32>(m_strata_flat.size())};
        check(lifted::launch_axioms(m_view, p, v, d, m_axiom_fc, strata, m_s), "launch_axioms");
        if (ev)
            ev[1].record(m_s);
        ++m_stats.axiom_chunks;
        *derived_out = der;
        // lazy slots: derived atoms without a slot are interned in canonical-id order, then views and axioms rerun
        if (!m_missing_words || !resolve_missing())
            return dw;
        ++m_stats.axiom_reruns;
    }
}

void ChunkGenerator::views()
{
    auto* views = static_cast<u64*>(m_views.ensure(m_ctx, static_cast<u64>(m_parents.rows) * m_view_words * sizeof(u64), m_s));
    if (!m_device_axioms || m_parents.rows == 0)
    {
        check(lifted::launch_view(m_view, m_parents, lifted::Views{views, m_view_words}, m_s), "launch_view");
        return;
    }
    lifted::Parents fluent = m_parents;
    fluent.derived = nullptr;
    fluent.derived_words = 0;
    u64* der = nullptr;
    const u32 dw = views_and_axioms(fluent, views, m_dev_derived, &der);
    m_parents.derived = der;
    m_parents.derived_words = dw;
}

void ChunkGenerator::goal_count(u32* out)
{
    check(lifted::launch_goal_count(m_view, m_parents, out, m_s), "launch_goal_count");
}

void ChunkGenerator::count()
{
    count_device();
    count_host();
}

void ChunkGenerator::count_device()
{
    const u64 n = static_cast<u64>(m_parents.rows) * m_S + 1;
    if (n > 0x7FFFFFFFull)
        throw std::length_error("mymyr: ChunkGenerator: more than 2^31 (state, schema) segments in one chunk");
    auto* counts = static_cast<u32*>(m_counts.ensure(m_ctx, n * sizeof(u32), m_s));
    m_offsets.ensure(m_ctx, n * sizeof(u32), m_s);  // (count_host's scan)
    m_scan.ensure(m_ctx, lifted::scan_temp_bytes(n), m_s);
    check(cudaMemsetAsync(counts, 0, n * sizeof(u32), m_s), "cudaMemsetAsync");
    const lifted::Views views{static_cast<u64*>(m_views.data()), m_view_words};
    // the counts do not depend on the matcher: its order is free unless the rows are in matcher order
    const rl::dev::MatchOrder order = m_canonical ? rl::dev::MatchOrder::Free : rl::dev::MatchOrder::Matcher;
    const Lists& L = m_lists[m_witness ? 0 : 1][static_cast<u32>(order)];
    const auto* sets = static_cast<const u32*>(m_schema_sets.data()) + L.at;
    const u32 w = m_witness ? 1u : 0u, c = m_canonical ? 1u : 0u;
    // conditional effects do not change the counts: the schemas with and without them are counted together
    const u32 shallow = L.fixed + L.fixed_ce + L.fc + L.fc_ce;
    u32* work = deep_work(L);
    const lifted::SchemaSet sets_of[4] = {{sets, L.fixed + L.fixed_ce, m_S, w, c, 0, 0},
                                          {sets + L.fixed + L.fixed_ce, L.fc + L.fc_ce, m_S, w, c, 1, 0},
                                          {sets + shallow, L.deep, m_S, w, c, 0, 1, work, m_deep_budget},
                                          {sets + shallow + L.deep, L.deep_fc, m_S, w, c, 1, 1, work, m_deep_budget}};
    for (const lifted::SchemaSet& set : sets_of)
        if (set.count)
            check(lifted::launch_count(m_view, m_parents, views, set, counts, m_s), "launch_count");
}

void ChunkGenerator::count_host()
{
    if (m_host_pending)
        throw std::logic_error("mymyr: ChunkGenerator::count_host: the chunk's host work has not run (host_work)");
    const u64 n = static_cast<u64>(m_parents.rows) * m_S + 1;
    auto* counts = static_cast<u32*>(m_counts.data());
    auto* offsets = static_cast<u32*>(m_offsets.data());
    const u64 tb = lifted::scan_temp_bytes(n);
    void* temp = m_scan.data();
    if (!m_host_count.empty())
    {
        upload_vec(m_ctx, m_h_counts_seg, m_host_count_seg, m_s);
        upload_vec(m_ctx, m_h_counts, m_host_count, m_s);
        check(lifted::launch_scatter_counts(static_cast<const u32*>(m_h_counts_seg.data()), static_cast<const u32*>(m_h_counts.data()),
                                            m_host_count.size(), counts, m_s),
              "launch_scatter_counts");
    }
    check(lifted::launch_scan(counts, offsets, n, temp, tb, m_s), "launch_scan");
}

void ChunkGenerator::write(const lifted::Labels& labels, u64* words, u32 out_words, u32* words_needed)
{
    const lifted::Views views{static_cast<u64*>(m_views.data()), m_view_words};
    // binding labels under witness pruning carry the witnesses: the CPU's (rl::dev::MatchOrder::Witness)
    const rl::dev::MatchOrder order = !m_canonical                   ? rl::dev::MatchOrder::Matcher
                                      : m_witness && labels.binding ? rl::dev::MatchOrder::Witness
                                                                    : rl::dev::MatchOrder::Free;
    const Lists& L = m_lists[m_witness ? 0 : 1][static_cast<u32>(order)];
    const auto* sets = static_cast<const u32*>(m_schema_sets.data()) + L.at;
    const u32 w = m_witness ? 1u : 0u, c = m_canonical ? 1u : 0u;
    const lifted::SchemaSet fixed{sets, L.fixed, m_S, w, c, 0, 0};
    const lifted::SchemaSet fixed_ce{sets + L.fixed, L.fixed_ce, m_S, w, c, 0, 0};
    const lifted::SchemaSet fc{sets + L.fixed + L.fixed_ce, L.fc, m_S, w, c, 1, 0};
    const lifted::SchemaSet fc_ce{sets + L.fixed + L.fixed_ce + L.fc, L.fc_ce, m_S, w, c, 1, 0};
    const u32 shallow = L.fixed + L.fixed_ce + L.fc + L.fc_ce;
    u32* work = deep_work(L);
    const lifted::SchemaSet deep{sets + shallow, L.deep, m_S, w, c, 0, 1, work, m_deep_budget};
    const lifted::SchemaSet deep_fc{sets + shallow + L.deep, L.deep_fc, m_S, w, c, 1, 1, work, m_deep_budget};
    lifted::SuccessorRows out;
    out.words = words;
    out.out_words = out_words;
    out.words_needed = words_needed;
    if (m_missing_words)
    {
        out.missing_bits = static_cast<u32*>(m_missing.data());
        out.missing_flag = out.missing_bits + m_missing_words;
    }
    // the lists without conditional effects label their rows, launch_put writes the successors one thread each
    // (a state with many successors does not bound the launch with its one thread's writes)
    lifted::SuccessorRows dout = out;
    dout.deferred = labels.binding && words && fixed.count + fc.count + deep.count + deep_fc.count ? 1 : 0;
    if (fixed.count)
        check(lifted::launch_write(m_view, m_parents, views, fixed, seg_offsets(), labels, dout, m_s), "launch_write");
    if (fixed_ce.count)
        check(lifted::launch_write_ce(m_view, m_parents, views, fixed_ce, seg_offsets(), labels, out, m_s), "launch_write_ce");
    if (fc.count)
        check(lifted::launch_write(m_view, m_parents, views, fc, seg_offsets(), labels, dout, m_s), "launch_write (fc)");
    if (fc_ce.count)
        check(lifted::launch_write_ce(m_view, m_parents, views, fc_ce, seg_offsets(), labels, out, m_s), "launch_write_ce (fc)");
    if (deep.count)
        check(lifted::launch_write(m_view, m_parents, views, deep, seg_offsets(), labels, dout, m_s), "launch_write (deep)");
    if (deep_fc.count)
        check(lifted::launch_write(m_view, m_parents, views, deep_fc, seg_offsets(), labels, dout, m_s), "launch_write (deep fc)");
    place_host(labels, words, out_words);
    // after the CPU-fallback rows, so that every row below the capacity is labeled when launch_put reads the labels
    // (a fallback row's schema is not deferred)
    if (dout.deferred)
        check(lifted::launch_put(m_view, m_parents, m_S, static_cast<const u8*>(m_put_mask.data()) + (m_witness ? 0 : u64{m_S}),
                                 seg_offsets(), labels, out, u64{m_parents.rows} * m_S, m_s),
              "launch_put");
}

void ChunkGenerator::place_host(const lifted::Labels& labels, u64* words, u32 out_words)
{
    const u64 n = m_host_segment.size();
    if (n == 0)
        return;
    // CPU-fallback rows: labels at the labels' width, words at out_words (rows that do not fit are zero, as in rl::expand)
    const u32 LW = labels.label_width;
    std::vector<u32> bind(n * LW, 0xFFFFFFFFu);
    for (u64 i = 0; i < n; ++i)
        std::copy_n(m_host_binding.begin() + static_cast<std::ptrdiff_t>(i * m_L), std::min(m_L, LW),
                    bind.begin() + static_cast<std::ptrdiff_t>(i * LW));
    upload_vec(m_ctx, m_h_seg, m_host_segment, m_s);
    upload_vec(m_ctx, m_h_rank, m_host_rank, m_s);
    upload_vec(m_ctx, m_h_bind, bind, m_s);
    lifted::HostRows h{static_cast<const u32*>(m_h_seg.data()), static_cast<const u32*>(m_h_rank.data()),
                       static_cast<const u32*>(m_h_bind.data()), nullptr, n};
    if (words && out_words)
    {
        m_staging.assign(n * out_words, 0);
        for (u64 i = 0; i < n; ++i)
            if (m_host_need[i] <= out_words)
                std::copy_n(m_host_words.begin() + static_cast<std::ptrdiff_t>(i * m_host_row_words), std::min(m_host_exact[i], out_words),
                            m_staging.begin() + static_cast<std::ptrdiff_t>(i * out_words));
        upload_vec(m_ctx, m_h_words, m_staging, m_s);
        h.words = static_cast<const u64*>(m_h_words.data());
    }
    // (pageable uploads are staged before cudaMemcpyAsync returns: the host vectors may be reused at once)
    check(lifted::launch_place_host(h, seg_offsets(), m_S, labels, words, out_words, m_s), "launch_place_host");
}

bool ChunkGenerator::resolve_missing()
{
    if (!m_missing_words)
        return false;
    DeviceGuard g(m_ctx->device());
    auto* bits_dev = static_cast<u32*>(m_missing.data());
    auto* flag = static_cast<u32*>(m_flag.data(0));
    check(cudaMemcpyAsync(flag, bits_dev + m_missing_words, sizeof(u32), cudaMemcpyDeviceToHost, m_s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(m_s), "cudaStreamSynchronize");
    if (*flag == 0)
        return false;
    std::vector<u32> bits(m_missing_words);
    check(cudaMemcpyAsync(bits.data(), bits_dev, m_missing_words * sizeof(u32), cudaMemcpyDeviceToHost, m_s), "cudaMemcpyAsync");
    check(cudaMemsetAsync(bits_dev, 0, (m_missing_words + 1) * sizeof(u32), m_s), "cudaMemsetAsync");
    check(cudaStreamSynchronize(m_s), "cudaStreamSynchronize");
    // canonical order: the slot numbering does not depend on which thread saw an atom first
    for (u64 w = 0; w < m_missing_words; ++w)
        for (u32 x = bits[w]; x; x &= x - 1)
            (void)m_task->atoms().intern(w * 32 + static_cast<u32>(std::countr_zero(x)));
    refresh();
    return true;
}

void ChunkGenerator::prepare(u32 rows)
{
    DeviceGuard g(m_ctx->device());
    refresh();
    const u64 r = std::max<u32>(rows, 1);
    (void)m_views.ensure(m_ctx, r * m_view_words * sizeof(u64), m_s);
    const u64 n = r * m_S + 1;
    (void)m_counts.ensure(m_ctx, n * sizeof(u32), m_s);
    (void)m_offsets.ensure(m_ctx, n * sizeof(u32), m_s);
    (void)m_scan.ensure(m_ctx, lifted::scan_temp_bytes(n), m_s);
    if (m_device_axioms && m_task->has_axioms())
        (void)m_dev_derived.ensure(m_ctx, r * std::max<u32>(1, bits::words_for(m_view.derived_slots)) * sizeof(u64), m_s);
    if (const u32 most = std::max(m_place[0].deep_count, m_place[1].deep_count); most > 0)
        (void)m_deep_work.ensure(m_ctx, lifted::deep_work_words(r * most) * sizeof(u32), m_s);
}

u32* ChunkGenerator::deep_work(const Lists& L)
{
    // (sized for the most deep schemas of any list, as prepare() does: a captured chunk keeps its address)
    const u32 most = std::max(m_place[0].deep_count, m_place[1].deep_count);
    if (L.deep + L.deep_fc == 0 || most == 0)
        return nullptr;
    const u64 rows = std::max<u32>(m_parents.rows, 1);
    return static_cast<u32*>(m_deep_work.ensure(m_ctx, lifted::deep_work_words(rows * most) * sizeof(u32), m_s));
}

u64 ChunkGenerator::capture_key() const noexcept
{
    u64 h = 0x9E3779B97F4A7C15ull ^ m_uploads ^ (u64{m_deep_budget} << 32);
    const void* const addresses[] = {m_views.data(),   m_counts.data(),      m_offsets.data(),     m_scan.data(),
                                     m_dev_derived.data(), m_missing.data(), m_schema_sets.data(), m_axiom_reads.data(),
                                     m_put_mask.data(), m_deep_work.data(), m_dt.get()};
    for (const void* p : addresses)
    {
        h ^= reinterpret_cast<std::uintptr_t>(p) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    return h;
}

u32 ChunkGenerator::host_words_needed(u32 out_words) const noexcept
{
    u32 w = 0;
    for (usize i = 0; i < m_host_need.size(); ++i)
        if (m_host_need[i] > out_words)
            w = std::max(w, m_host_exact[i]);
    return w;
}

void ChunkGenerator::goal_flags(const u64* rows, u64 stride, u32 words, u64 row_count, const u32* order, u64 n, u8* out)
{
    if (n == 0)
        return;
    DeviceGuard g(m_ctx->device());
    const bool derived = m_task->compiled().goal.uses_derived;
    if (derived && !m_device_axioms)
        throw std::logic_error("mymyr: ChunkGenerator::goal_flags: the axioms of this task run on the CPU (" +
                               m_host_axioms_reason + ")");
    if (words > lifted::k_max_words)
        throw std::invalid_argument("mymyr: ChunkGenerator::goal_flags: rows of " + std::to_string(words) + " words");
    const bool gather = order || stride != words;
    if (!derived && !gather)
    {
        check(lifted::launch_goal_rows(m_view, rows, words, n, nullptr, out, m_s), "launch_goal_rows");
        return;
    }
    const u64 dw_guess = std::max<u64>(1, bits::words_for(m_view.derived_slots));
    const u64 per = (words + (derived ? m_view_words + dw_guess : 0)) * sizeof(u64);
    const u64 chunk = std::max<u64>(1, std::min<u64>(k_goal_budget / std::max<u64>(per, 8), 0x7FFFFFFFull));
    for (u64 b = 0; b < n; b += chunk)
    {
        const u64 c = std::min(chunk, n - b);
        const u64* src = rows + b * stride;
        if (gather)
        {
            auto* dst = static_cast<u64*>(m_goal_rows.ensure(m_ctx, c * words * sizeof(u64), m_s));
            check(lifted::launch_gather_rows(order ? rows : rows + b * stride, stride, words, order ? row_count : c,
                                             order ? order + b : nullptr, c, dst, m_s),
                  "launch_gather_rows");
            src = dst;
        }
        if (!derived)
        {
            check(lifted::launch_goal_rows(m_view, src, words, c, nullptr, out + b, m_s), "launch_goal_rows");
            continue;
        }
        const lifted::Parents p{src, words, words, static_cast<u32>(c), nullptr, 0};
        auto* views = static_cast<u64*>(m_goal_views.ensure(m_ctx, c * m_view_words * sizeof(u64), m_s));
        u64* der = nullptr;
        const u32 dw = views_and_axioms(p, views, m_goal_derived, &der);
        check(lifted::launch_goal_rows_derived(m_view, src, words, c, der, dw, out + b, m_s), "launch_goal_rows_derived");
    }
}

void ChunkGenerator::fold_axiom_events()
{
    if (m_axiom_event_next == 0)
        return;
    DeviceGuard g(m_ctx->device());
    m_axiom_events[m_axiom_event_next - 1].synchronize();
    for (usize i = 0; i + 1 < m_axiom_event_next; i += 2)
        m_stats.device_axiom_ms += m_axiom_events[i + 1].elapsed_ms(m_axiom_events[i]);
    m_axiom_event_next = 0;
}

GeneratorStats ChunkGenerator::stats()
{
    fold_axiom_events();
    return m_stats;
}
}  // namespace mymyr::cuda
