// The device expand over a task suite (include/mymyr/cuda/suite_expand.hpp): one DeviceExpander per domain, each on the
// rows of its domain (in place when every domain's rows are one run of the batch, else gathered and scattered back).


#include "mymyr/cuda/suite_expand.hpp"

#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/suite_kernels.hpp"
#include "table_launch.hpp"
#include "expand_drop.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::cuda
{
namespace
{
template<class T>
T* scratch_of(Scratch& sc, const ContextPtr& ctx, u64 n, cudaStream_t s)
{
    return static_cast<T*>(sc.ensure(ctx, std::max<u64>(n, 1) * sizeof(T), s));
}
}  // namespace

struct SuiteExpander::Impl
{
    ContextPtr ctx;
    rl::TaskSuitePtr suite;
    std::vector<std::unique_ptr<DeviceExpander>> subs;  // per domain
    cudaStream_t s = nullptr;
    // the counted batch (several domains)
    struct Part
    {
        u32 domain = 0;
        u64 first = 0, rows = 0;  // runs: rows [first, first + rows) of the batch; gathered: positions of `index`
        u64 total = 0, base = 0;
        u32 words = 0;            // gathered: the part's rows (its table's width, at most the batch's)
        u64 rows_at = 0;          // gathered: its first row's word in the gathered rows
    };
    std::vector<Part> parts;  // in batch order of their first rows (runs), or by domain (gathered)
    bool runs = false;        // every domain's rows are one run of the batch
    bool counted = false, drop_noops = false;
    detail::ExpansionDrop drop;
    Scratch digest_offsets;
    rl::StateBatchView in{};
    u64 stride = 0, total = 0;
    // the suite on the device: each instance's domain and local id, and its fluent slots (uploaded once)
    DeviceBuffer dom_of, loc_of, limits;
    Scratch ctl, index, pos, order_temp, local, gathered, t_offsets, row_counts, batch_offsets, scan;
    PinnedBuffer host_ctl;

    [[nodiscard]] bool one() const noexcept { return subs.size() == 1; }

    /// The instances' fluent slots (rl::expand's validation).
    void upload_limits()
    {
        const u32 I = suite->size();
        std::vector<u32> lim(I);
        for (u32 g = 0; g < I; ++g)
            lim[g] = suite->task(g)->atoms().fluent_slots();
        if (limits.size() < u64{I} * sizeof(u32))
            limits = DeviceBuffer(ctx, u64{I} * sizeof(u32), s);
        check(cudaMemcpyAsync(limits.data(), lim.data(), u64{I} * sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the pageable source
    }

    /// The batch's domains on the device (suitek::launch_classify; validation: lifted::launch_check_multi), one
    /// synchronization: one run per domain present, or each domain's rows gathered (a stable sort by domain); then each
    /// domain's expander counts its part.
    u64 count_domains(const i32* ids, const rl::ExpandOptions& opt)
    {
        const u64 n = in.rows;
        const u32 D = suite->num_domains(), I = suite->size();
        const u64 words = suitek::k_classify_words + 2 * u64{D};
        auto* dl = scratch_of<i32>(local, ctx, n, s);
        auto* d = scratch_of<u32>(ctl, ctx, words, s);
        check(suitek::launch_classify(ids, n, static_cast<const u32*>(dom_of.data()), static_cast<const u32*>(loc_of.data()),
                                      I, D, dl, d, s),
              "launch_classify");
        if (opt.validate && n)
        {
            check(lifted::launch_check_multi(static_cast<const u32*>(limits.data()), I, ids, in.data, stride, in.words, n,
                                             d + suitek::k_bad_id, d + suitek::k_bad_row, s),
                  "launch_check_multi");
        }
        auto* h = static_cast<u32*>(host_ctl.data());
        check(cudaMemcpyAsync(h, d, words * sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        if (h[suitek::k_bad_id] != 0xFFFFFFFFu)
        {
            const u64 r = h[suitek::k_bad_id];
            i32 bad = 0;
            check(cudaMemcpy(&bad, ids + r, sizeof(i32), cudaMemcpyDeviceToHost), "cudaMemcpy");
            throw std::invalid_argument("mymyr: expand: task id " + std::to_string(bad) + " of row " + std::to_string(r) +
                                        " is outside the suite's " + std::to_string(I) + " instances");
        }
        if (h[suitek::k_bad_row] != 0xFFFFFFFFu)
            throw std::invalid_argument("mymyr: expand: state row " + std::to_string(h[suitek::k_bad_row]) +
                                        " sets atom slots its instance does not have (a state of another task?)");
        const u32* per = h + suitek::k_classify_words;
        const u32* first = per + D;
        u64 present = 0;
        for (u32 k = 0; k < D; ++k)
            present += per[k] ? 1 : 0;
        runs = !suite->numeric() && (n == 0 || h[suitek::k_changes] + 1 == present);
        parts.clear();
        const u64* g = nullptr;
        if (runs)
        {
            // the runs in batch order, the rows' local ids where they are
            for (u32 k = 0; k < D; ++k)
                if (per[k])
                    parts.push_back({k, first[k], per[k], 0, 0, 0, 0});
            std::sort(parts.begin(), parts.end(), [](const Part& a, const Part& b) { return a.first < b.first; });
        }
        else
        {
            // each domain's rows in batch order (a stable sort of the rows by domain), at its table's width
            auto* di = scratch_of<u32>(index, ctx, n, s);
            const u64 tb = lifted::order_temp_bytes(n);
            const auto* rid = reinterpret_cast<const u32*>(ids);
            const lifted::OrderKeys keys{static_cast<const u32*>(dom_of.data()), D};
            auto* dp = scratch_of<u32>(pos, ctx, n, s);
            check(lifted::launch_order(rid, I, keys, n, order_temp.ensure(ctx, tb, s), tb, di, dp, 1, s), "launch_order");
            detail::check_order(ctx, rid, I, keys, n, di, dp, 1, s);
            check(suitek::launch_local_ids(ids, di, n, static_cast<const u32*>(loc_of.data()), dl, s), "launch_local_ids");
            u64 at = 0, rows_at = 0;
            for (u32 k = 0; k < D; ++k)
                if (per[k])
                {
                    const u32 w = suite->table(k)->numeric() ? in.words : std::max<u32>(1, std::min(in.words, suite->table(k)->words()));
                    parts.push_back({k, at, per[k], 0, 0, w, rows_at});
                    at += per[k];
                    rows_at += u64{per[k]} * (w + suite->table(k)->numeric_words());
                }
            auto* gw = scratch_of<u64>(gathered, ctx, rows_at, s);
            for (const Part& p : parts)
                check(lifted::launch_gather_rows(in.data, stride, p.words + suite->table(p.domain)->numeric_words(), n, di + p.first, p.rows, gw + p.rows_at, s),
                      "launch_gather_rows");
            g = gw;
        }
        rl::ExpandOptions o = opt;
        o.validate = false;  // validated above, against the batch's rows
        total = 0;
        for (Part& p : parts)
        {
            const rl::StateBatchView v = runs ? rl::StateBatchView{in.data + p.first * stride, p.rows, in.words, stride, 0}
                                              : rl::StateBatchView{g + p.rows_at, p.rows, p.words, p.words + suite->table(p.domain)->numeric_words(),
                                                                   suite->table(p.domain)->numeric_words()};
            p.total = subs[p.domain]->count(v, dl + p.first, o);
            p.base = total;
            total += p.total;
        }
        return total;
    }

    void write_runs(rl::Expansion& out)
    {
        const u32 RW = out.words, L = out.label_width;
        u32 need = 0;
        for (const Part& p : parts)
        {
            rl::Expansion e;
            e.capacity = out.capacity > p.base ? std::min(out.capacity - p.base, p.total) : 0;
            e.words = out.words;
            e.label_width = L;
            e.succ = out.succ ? out.succ + p.base * RW : nullptr;
            e.parent = out.parent ? out.parent + p.base : nullptr;
            e.schema = out.schema ? out.schema + p.base : nullptr;
            e.binding = out.binding ? out.binding + p.base * L : nullptr;
            e.goal = out.goal ? out.goal + p.base : nullptr;
            e.succ_hash = out.succ_hash ? out.succ_hash + p.base : nullptr;
            e.noop = out.noop ? out.noop + p.base : nullptr;
            e.derived_words = out.derived_words;
            e.derived = out.derived ? out.derived + p.base * out.derived_words : nullptr;
            e.offsets = out.offsets ? out.offsets + p.first : nullptr;
            subs[p.domain]->write(e);
            need = std::max(need, e.words_needed);
            // the run's offsets and parents at batch positions (the next run rewrites its first offset, then shifts it)
            if (out.offsets)
                check(suitek::launch_add(out.offsets + p.first, p.rows + 1, static_cast<i32>(p.base), s), "launch_add");
            if (out.parent)
                check(suitek::launch_add(out.parent + p.base, e.capacity, static_cast<i32>(p.first), s), "launch_add");
        }
        out.total = total;
        out.words_needed = need;
        if (out.offsets && in.rows == 0)
            check(cudaMemsetAsync(out.offsets, 0, sizeof(i32), s), "cudaMemsetAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // as DeviceExpander::write: the outputs are done
    }

    void write_gathered(rl::Expansion& out)
    {
        const u64 n = in.rows;
        const u64 np = parts.size();
        // the batch's CSR offsets from the parts' counts, before any part writes
        auto* toff = scratch_of<i32>(t_offsets, ctx, n + np, s);
        auto* cnt = scratch_of<u32>(row_counts, ctx, n + 1, s);
        check(cudaMemsetAsync(cnt, 0, (n + 1) * sizeof(u32), s), "cudaMemsetAsync");
        const auto* di = static_cast<const u32*>(index.data());
        for (usize k = 0; k < np; ++k)
        {
            const Part& p = parts[k];
            subs[p.domain]->offsets(toff + p.first + k);
            check(lifted::launch_part_counts(di + p.first, toff + p.first + k, p.rows, cnt, s), "launch_part_counts");
        }
        auto* go = out.offsets ? out.offsets : scratch_of<i32>(batch_offsets, ctx, n + 1, s);
        const u64 stb = lifted::scan_temp_bytes(n + 1);
        check(lifted::launch_scan(cnt, reinterpret_cast<u32*>(go), n + 1, scan.ensure(ctx, stb, s), stb, s), "launch_scan");
        // each domain's rows straight at their batch positions, at the suite's widths; a part without axioms leaves its
        // derived rows to this zero fill
        if (out.derived && out.derived_words && std::min(total, out.capacity))
            check(cudaMemsetAsync(out.derived, 0, std::min(total, out.capacity) * out.derived_words * sizeof(u64), s),
                  "cudaMemsetAsync");
        u32 need = 0;
        for (usize k = 0; k < np; ++k)
        {
            const Part& p = parts[k];
            rl::Expansion e = out;
            e.offsets = nullptr;
            subs[p.domain]->write(e, DeviceExpander::RowMap{di + p.first, go});
            need = std::max(need, e.words_needed);
        }
        out.total = total;
        out.words_needed = need;
    }
};

SuiteExpander::SuiteExpander(ContextPtr ctx, rl::TaskSuitePtr suite, cudaStream_t stream) : m(std::make_unique<Impl>())
{
    if (!ctx || !suite)
        throw std::invalid_argument("mymyr: SuiteExpander: null context or suite");
    Impl& I = *m;
    I.ctx = std::move(ctx);
    I.suite = std::move(suite);
    I.s = stream ? stream : I.ctx->stream();
    const u32 D = I.suite->num_domains();
    for (u32 d = 0; d < D; ++d)
        I.subs.push_back(std::make_unique<DeviceExpander>(I.ctx, I.suite->table(d), I.s));
    if (D == 1)
        return;
    // the instances' domains and local ids, and their fluent slots
    DeviceGuard guard(I.ctx->device());
    std::vector<u32> dom(I.suite->size()), loc(I.suite->size());
    for (u32 g = 0; g < I.suite->size(); ++g)
    {
        dom[g] = I.suite->domain_of(g);
        loc[g] = I.suite->local_id(g);
    }
    I.dom_of = DeviceBuffer(I.ctx, dom.size() * sizeof(u32), I.s);
    I.loc_of = DeviceBuffer(I.ctx, loc.size() * sizeof(u32), I.s);
    check(cudaMemcpyAsync(I.dom_of.data(), dom.data(), dom.size() * sizeof(u32), cudaMemcpyHostToDevice, I.s), "cudaMemcpyAsync");
    check(cudaMemcpyAsync(I.loc_of.data(), loc.data(), loc.size() * sizeof(u32), cudaMemcpyHostToDevice, I.s), "cudaMemcpyAsync");
    I.upload_limits();  // synchronizes (the pageable sources above too)
    I.host_ctl = PinnedBuffer((suitek::k_classify_words + 2 * u64{D}) * sizeof(u32));
}

SuiteExpander::~SuiteExpander() = default;

const ContextPtr& SuiteExpander::context() const noexcept { return m->ctx; }
const rl::TaskSuitePtr& SuiteExpander::suite() const noexcept { return m->suite; }
DeviceExpander& SuiteExpander::domain(u32 d) const { return *m->subs.at(d); }
cudaStream_t SuiteExpander::stream() const noexcept { return m->s; }

void SuiteExpander::set_stream(cudaStream_t s)
{
    Impl& I = *m;
    s = s ? s : I.ctx->stream();
    if (s == I.s)
        return;
    for (auto& x : I.subs)
        x->set_stream(s);
    DeviceGuard guard(I.ctx->device());
    // the constants are used on s after the work before; the scratch is freed on the old stream (after its work) and
    // allocated again on s
    stream_wait(s, I.s);
    for (DeviceBuffer* b : {&I.dom_of, &I.loc_of, &I.limits})
        if (b->data())
            b->record_stream(s);
    for (Scratch* sc : {&I.ctl, &I.index, &I.pos, &I.order_temp, &I.local, &I.gathered, &I.t_offsets, &I.row_counts,
                        &I.batch_offsets, &I.scan})
        sc->reset();
    I.s = s;
    I.counted = false;  // the gathered rows are scratch
}

void SuiteExpander::set_chunk_rows(u64 rows) noexcept
{
    for (auto& x : m->subs)
        x->set_chunk_rows(rows);
}

void SuiteExpander::set_launch(BucketLaunch launch) noexcept
{
    for (auto& x : m->subs)
        x->set_launch(launch);
}

u64 SuiteExpander::count(rl::StateBatchView in, const i32* task_ids, const rl::ExpandOptions& opt)
{
    Impl& I = *m;
    if (I.one())
        return I.subs[0]->count(in, task_ids, opt);
    DeviceGuard guard(I.ctx->device());
    I.counted = false;
    if (in.rows && !in.data)
        throw std::invalid_argument("mymyr: expand: null state buffer");
    if (in.stride && in.stride < in.words + in.numeric_words)
        throw std::invalid_argument("mymyr: expand: row stride smaller than the row width");
    if (in.numeric_words != I.suite->numeric_words())
        throw std::invalid_argument("mymyr: device expand: input numeric width differs from the suite");
    if (in.rows > rl::k_max_rows)
        throw std::invalid_argument("mymyr: expand: more than 2^31 - 1 states in one batch");
    if (!task_ids && in.rows)
        throw std::invalid_argument("mymyr: expand: a batch over a suite of " + std::to_string(I.suite->size()) +
                                    " instances needs task ids");
    I.in = in;
    I.stride = in.stride ? in.stride : in.words + in.numeric_words;
    I.drop_noops = opt.drop_noops;
    rl::ExpandOptions full = opt;
    full.drop_noops = false;
    const u64 total = I.count_domains(task_ids, full);
    I.counted = true;
    return total;
}

void SuiteExpander::write(rl::Expansion& out)
{
    Impl& I = *m;
    if (I.one())
    {
        I.subs[0]->write(out);
        return;
    }
    if (!I.counted)
        throw std::logic_error("mymyr: SuiteExpander::write: count() first");
    DeviceGuard guard(I.ctx->device());
    if (out.binding && out.label_width < I.suite->label_width())
        throw std::invalid_argument("mymyr: expand: label width " + std::to_string(out.label_width) +
                                    " is below the largest schema arity " + std::to_string(I.suite->label_width()));
    if (out.capacity > rl::k_max_rows)
        throw std::invalid_argument("mymyr: expand: capacity above 2^31 - 1 rows");
    if (out.succ && out.words == 0 && out.capacity)
        throw std::invalid_argument("mymyr: expand: successor rows of zero words");
    if (out.goal && !out.succ)
        throw std::invalid_argument("mymyr: expand: goal flags need the successor rows");
    if (out.derived && !out.succ)
        throw std::invalid_argument("mymyr: expand: derived words need the successor rows");
    if (out.derived && out.derived_words < I.suite->derived_words())
        throw std::invalid_argument("mymyr: expand: derived rows of " + std::to_string(out.derived_words) +
                                    " words; the suite has instances of " + std::to_string(I.suite->derived_words()));
    if (out.numeric_words != I.suite->numeric_words())
        throw std::invalid_argument("mymyr: device expand: output numeric width differs from the suite");
    if (out.offsets && I.total > rl::k_max_rows)
        throw std::length_error("mymyr: expand: more than 2^31 - 1 successors in one batch");
    if (I.drop_noops && out.capacity && !out.index)
        throw std::invalid_argument("mymyr: expand: drop_noops requires index");
    rl::Expansion dest = out;
    if ((I.drop_noops || out.index) && !dest.offsets)
        dest.offsets = scratch_of<i32>(I.digest_offsets, I.ctx, I.in.rows + 1, I.s);
    rl::Expansion raw = I.drop_noops ? I.drop.prepare(dest, I.in.rows, I.ctx, I.s) : dest;
    if (I.runs)
        I.write_runs(raw);
    else
        I.write_gathered(raw);
    out.generated = I.total;
    out.total = raw.total;
    out.words_needed = raw.words_needed;
    if (I.drop_noops)
    {
        I.drop.compact(raw, dest, I.in.rows, I.ctx, I.s);
        i32 kept = 0;
        check(cudaMemcpyAsync(&kept, dest.offsets + I.in.rows, sizeof(i32), cudaMemcpyDeviceToHost, I.s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(I.s), "cudaStreamSynchronize");
        out.total = static_cast<u32>(kept);
    }
    else if (out.index)
    {
        check(launch_expansion_indices(detail::expansion_rows(dest), I.in.rows, I.s), "launch_expansion_indices");
        check(cudaStreamSynchronize(I.s), "cudaStreamSynchronize");
    }
}

void SuiteExpander::expand(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, const rl::ExpandOptions& options)
{
    count(in, task_ids, options);
    write(out);
}

std::string SuiteExpander::async_unsupported(u64 rows, const rl::ExpandOptions& options) const
{
    if (!m->one())
        return "a suite of several domains classifies its rows on the host";
    return m->subs[0]->async_unsupported(rows, options);
}

void SuiteExpander::expand_async(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, u32* status,
                                 const rl::ExpandOptions& options, const u32* rows_dev)
{
    if (!m->one())
        throw std::invalid_argument("mymyr: device expand: the expansion needs the host: " + async_unsupported(in.rows, options));
    m->counted = false;
    m->subs[0]->expand_async(in, task_ids, out, status, options, rows_dev);
}

void SuiteExpander::pad(const rl::Expansion& flat, u64 rows, rl::PaddedExpansion& out)
{
    m->subs[0]->pad(flat, rows, out);
}
}  // namespace mymyr::cuda
