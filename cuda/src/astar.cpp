// The device best-first searches (include/mymyr/cuda/astar.hpp, gbfs.hpp): the host driver of the batched A* and
// GBFS. Per step: pop a batch from the device bucket queue, gather the parents' rows, and per generator chunk:
// successors (ChunkGenerator), dedup (state_set), relaxation (astar_kernels.hpp), goal test and heuristic of the new
// states, and the sorted push of the chunk's entries into the queue. Every count of a step lives in the
// device control block (bfk::Ctl), so a step needs no host read: steps are captured into a CUDA graph and looped on the
// device; the host reads the control block once per loop. Where the host must take part (CPU-fallback schemas, derived
// goals, a heuristic that may meet states outside its grounding, lazy slots that met new atoms, a chunk past the
// capacities, a pop window too small) it launches the same kernels with reads between them.

#include "mymyr/cuda/astar.hpp"

#include "cost_program_build.hpp"

#include "mymyr/cuda/astar_kernels.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/gbfs.hpp"
#include "mymyr/cuda/state_set.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
double s_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

/// Row width of the device state rows: the fixed row-width buckets (as the device BrFS).
u32 bucket(u32 words)
{
    const u32 w = std::bit_ceil(std::max<u32>(words, 1));
    if (w > lifted::k_max_words)
        throw std::invalid_argument("mymyr: device best-first search: the states grew wider than " +
                                    std::to_string(lifted::k_max_words) + " words (the device kernels' limit); use the CPU search");
    return w;
}

/// A grow-only typed device array; ensure(n, keep) preserves the first `keep` elements.
template<class T>
class DevArray
{
public:
    T* ensure(const ContextPtr& ctx, u64 n, cudaStream_t s, u64 keep = 0)
    {
        if (n <= m_cap && m_buf.stream() == s)
            return data();
        const u64 cap = std::max<u64>({n, m_cap + m_cap / 2, 64});
        DeviceBuffer nb(ctx, cap * sizeof(T), s);
        if (keep)
            check(cudaMemcpyAsync(nb.data(), m_buf.data(), std::min(keep, m_cap) * sizeof(T), cudaMemcpyDeviceToDevice, s),
                  "cudaMemcpyAsync (grow)");
        m_buf = std::move(nb);  // the old buffer is freed after the copy (stream-ordered on s)
        m_cap = cap;
        return data();
    }
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(m_buf.data()); }
    [[nodiscard]] u64 capacity() const noexcept { return m_cap; }
    [[nodiscard]] u64 bytes() const noexcept { return m_cap * sizeof(T); }

private:
    DeviceBuffer m_buf;
    u64 m_cap = 0;
};

template<class T>
void to_device(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n)
        check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (H2D)");
}
template<class T>
void to_host(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n)
        check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (D2H)");
}

u64 quantize(u64 n) { return std::bit_ceil(std::max<u64>(n, 1024)); }
}  // namespace

std::string best_first_unsupported(const Task& task, const DeviceBestFirstOptions& o)
{
    if (std::string why = ChunkGenerator::unsupported(task); !why.empty())
        return why;
    const search::BestFirstOptions& so = o.search;
    if (so.control.goal.kind != search::GoalSpec::Kind::Task)
        return "goals other than the task's (GoalSpec::Kind::Task)";
    if (!so.control.blocked_states.empty())
        return "blocked states";
    if (so.control.observer)
        return "search observers";
    if (so.evaluator)
        return "a caller-owned evaluator (the device builds its heuristic from search.heuristic)";
    if (so.symmetry_pruning != SymmetryPruning::Off)
        return "symmetry pruning (search.symmetry_pruning; the device generates every applicable action)";
    if (o.batch == 0)
        return "a batch of 0 states";
    const heuristics::Kind k = so.heuristic.kind;
    if (k != heuristics::Kind::Blind)
    {
        DeviceHeuristicOptions ho = o.heuristic;
        ho.kind = k;
        ho.costs = so.heuristic.costs;
        if (std::string why = DeviceHeuristic::unsupported(task, ho); !why.empty())
            return why;
    }
    const heuristics::ActionCosts costs(task);
    if (!task.numeric_slots() && !costs.unit() && (!costs.state_independent() || !costs.integral()))
        return costs.state_independent() ? "action costs that are not integral" : "action costs that depend on the state";
    return {};
}

namespace
{
struct Driver
{
    ContextPtr ctx;
    TaskPtr task;
    const DeviceBestFirstOptions& o;
    bool greedy;
    cudaStream_t s = nullptr;
    search::BestFirstResult r;
    DeviceBestFirstStats st;
    std::unique_ptr<ChunkGenerator> gen;
    std::unique_ptr<DeviceHeuristic> heur;  // null: blind
    std::unique_ptr<heuristics::ActionCosts> costs;
    f64 g0 = 0;
    bool witness = false, canonical = true, host = false, host_goal = false, reopen = true;
    bool async_h = true;  // the heuristic without a synchronization (no CPU fallback can be needed)
    bool graphs = false, loops = false;
    u32 S = 0, L = 1;

    // stored states and the state set
    DevArray<u64> rows;
    u32 W = 1;
    DeviceBuffer table;
    u64 slots = 0;
    // search nodes
    DevArray<u32> g, h, parent, sidx, depth;
    DevArray<u8> flags;
    DevArray<u64> best;
    u64 node_cap = 0;
    // the open list: the arena and the run list
    DevArray<u64> open, run_key, run_begin, run_key2, run_begin2;
    DevArray<u32> run_count, run_count2;
    u64 runs_cap = 0;
    // action costs
    cuda::costs::Programs cost_programs;
    bfk::Costs dcosts;
    // the control block: on the device and its last read
    DeviceBuffer ctl;
    PinnedLease pin;
    bfk::Ctl* dctl = nullptr;
    bfk::Ctl* hc = nullptr;
    // the step: B parents in K chunks of C rows; a pop window of window_cap entries; cap candidates per chunk
    u32 B = 0, C = 0, K = 1, window_cap = 0;
    u64 cap = 0;
    DevArray<u32> wprefix, pos, batch_id, batch_g, batch_depth;
    DevArray<u64> wbegin, batch_rows;
    DevArray<u8> pop_flags;
    Scratch pop_temp;
    std::vector<u64> host_batch;
    // chunk scratch for `cap` candidates (grown only)
    Scratch l_parent, l_schema, l_binding, l_sort, cand, result, rank, rank_temp, c_id, c_gc, c_part, c_push, p_keys, p_entries,
        s_keys, s_entries, r_keys, r_counts, sort_temp, new_begin, fresh_rows, h_fresh, goal_u8, misc;
    std::vector<u64> host_rows;
    // captured steps
    GraphExec step_graph;
    std::vector<u64> graph_key;
    bool graph_looped = false;
    // the candidates of recent chunks (the capacity estimate)
    std::array<u64, 4> recent{};
    usize recent_at = 0;
    Clock::time_point t0, deadline;
    bool timed = false;
    u64 states_at_goal = 0;  // the states stored when the goal's parent was expanded (0: count)

    Driver(ContextPtr c, TaskPtr t, const DeviceBestFirstOptions& opt, bool gr) : ctx(std::move(c)), task(std::move(t)), o(opt), greedy(gr) {}

    [[nodiscard]] bfk::Nodes nodes() const
    {
        return {g.data(), h.data(), parent.data(), sidx.data(), depth.data(), flags.data(), best.data()};
    }
    [[nodiscard]] state_set::Table tab() const { return {static_cast<u64*>(table.data()), slots - 1}; }
    [[nodiscard]] state_set::Rows arena() const { return {rows.data(), W}; }
    [[nodiscard]] bfk::Runs runs_of() const
    {
        return {run_key.data(), run_begin.data(), run_count.data(), run_key2.data(), run_begin2.data(), run_count2.data(),
                static_cast<u32>(runs_cap)};
    }
    void sync() { check(cudaStreamSynchronize(s), "cudaStreamSynchronize"); }

    /// Reads the control block (one synchronization) and raises its errors.
    void read()
    {
        to_host(hc, dctl, 1, s);
        sync();
        if (hc->label_error)
            throw std::logic_error("mymyr: device best-first search: a sorted segment found no scratch (internal error)");
        if (hc->error & bfk::k_err_cost)
            throw std::domain_error("mymyr: device best-first search: an applicable action's cost is undefined, negative or "
                                    "2^31 or more");
        if (hc->error & bfk::k_err_overflow)
            throw std::overflow_error("mymyr: device best-first search: a g or f value does not fit 32 bits");
        if (hc->eval_flags[1])
            throw std::invalid_argument("mymyr: DeviceHeuristic: a state sets an atom slot the task has not assigned");
        if (hc->eval_flags[0])
            throw std::logic_error("mymyr: device best-first search: a state outside the heuristic's grounding (internal error)");
    }
    /// Writes the host's copy of the control block to the device.
    void write_ctl() { to_device(dctl, hc, 1, s); }
    void clear_abort()
    {
        hc->abort = 0;
        check(cudaMemsetAsync(&dctl->abort, 0, sizeof(u32), s), "cudaMemsetAsync");
    }

    void alloc_table(u64 n)
    {
        slots = n;
        table = DeviceBuffer(ctx, slots * sizeof(u64), s);
        check(cudaMemsetAsync(table.data(), 0, slots * sizeof(u64), s), "cudaMemsetAsync (state table)");
    }
    [[nodiscard]] u64 table_limit() const { return slots / 4 * 3; }
    /// Room in the table for `more` states beyond `count` (rehashed when it grows).
    void ensure_table(u64 count, u64 more)
    {
        if (count + more <= table_limit())
            return;
        u64 n = slots;
        while (count + more > n / 4)
            n *= 2;
        alloc_table(n);
        check(state_set::launch_rehash(tab(), arena(), count, s), "launch_rehash");
        ++st.rehashes;
    }
    /// Node arrays and rows for `n` states (the first `count` kept).
    void ensure_nodes(u64 count, u64 n)
    {
        rows.ensure(ctx, n * W, s, count * W);
        if (n <= node_cap)
            return;
        g.ensure(ctx, n, s, count);
        h.ensure(ctx, n, s, count);
        parent.ensure(ctx, n, s, count);
        sidx.ensure(ctx, n, s, count);
        depth.ensure(ctx, n, s, count);
        flags.ensure(ctx, n, s, count);
        best.ensure(ctx, n, s, count);
        node_cap = std::min({g.capacity(), h.capacity(), parent.capacity(), sidx.capacity(), depth.capacity(), flags.capacity(),
                             best.capacity()});
        check(bfk::launch_fill_u64(best.data() + count, node_cap - count, bfk::k_no_key, s), "launch_fill_u64");
    }
    [[nodiscard]] u64 state_limit() const { return std::min(node_cap, table_limit()); }
    /// Room for `more` candidates after the last read: states, arena entries, runs. What grows at least doubles (the
    /// captured steps embed the arrays: they are captured again when one moves).
    void ensure_room(u64 more)
    {
        const u64 count = hc->count;
        if (count + more > state_limit())
        {
            ensure_table(count, std::max(more, count));
            ensure_nodes(count, 2 * count + more);
        }
        if (rows.capacity() / W < node_cap)
            rows.ensure(ctx, node_cap * W, s, count * W);
        if (hc->open_size + more > open.capacity())
            open.ensure(ctx, 2 * hc->open_size + more, s, hc->open_size);
        const u64 live = u64{hc->runs};  // the list's runs up to `runs` (head may be past 0)
        if (live + more > runs_cap)
        {
            const u64 n = std::max<u64>(2 * live + more, 2 * runs_cap);
            run_key.ensure(ctx, n, s, live);
            run_begin.ensure(ctx, n, s, live);
            run_count.ensure(ctx, n, s, live);
            run_key2.ensure(ctx, n, s);
            run_begin2.ensure(ctx, n, s);
            run_count2.ensure(ctx, n, s);
            runs_cap = std::min({run_key.capacity(), run_begin.capacity(), run_count.capacity(), run_key2.capacity(),
                                 run_begin2.capacity(), run_count2.capacity()});
        }
    }
    [[nodiscard]] bfk::ChunkLimits limits(u64 cp) const
    {
        return {cp, state_limit(), open.capacity(), runs_cap};
    }

    /// The chunks' scratch for `n` candidates.
    void size_chunk(u64 n)
    {
        n = std::max<u64>(n, 1);
        l_parent.ensure(ctx, n * sizeof(u32), s);
        if (!dcosts.unit)
        {
            l_schema.ensure(ctx, n * sizeof(u32), s);
            l_binding.ensure(ctx, n * L * sizeof(u32), s);
        }
        if (sorts())
            l_sort.ensure(ctx, n * L * sizeof(u32), s);
        cand.ensure(ctx, n * W * sizeof(u64), s);
        result.ensure(ctx, n * sizeof(u32), s);
        rank.ensure(ctx, (n + 1) * sizeof(u32), s);
        rank_temp.ensure(ctx, state_set::rank_temp_bytes(n), s);
        c_id.ensure(ctx, n * sizeof(u32), s);
        c_gc.ensure(ctx, n * sizeof(u32), s);
        c_part.ensure(ctx, n, s);
        c_push.ensure(ctx, n, s);
        p_keys.ensure(ctx, n * sizeof(u64), s);
        p_entries.ensure(ctx, n * sizeof(u64), s);
        s_keys.ensure(ctx, n * sizeof(u64), s);
        s_entries.ensure(ctx, n * sizeof(u64), s);
        r_keys.ensure(ctx, n * sizeof(u64), s);
        r_counts.ensure(ctx, n * sizeof(u32), s);
        sort_temp.ensure(ctx, bfk::sort_temp_bytes(n), s);
        new_begin.ensure(ctx, n * sizeof(u64), s);
        fresh_rows.ensure(ctx, n * W * sizeof(u64), s);
        h_fresh.ensure(ctx, n * sizeof(u32), s);
        goal_u8.ensure(ctx, n, s);
        if (heur && async_h)
            heur->prepare(n, s);
    }
    [[nodiscard]] bool sorts() const { return canonical && gen->sorts(witness) && dcosts.unit; }
    /// The step's scratch (batch, pop window, the batch's rows).
    void size_step()
    {
        const u64 nb = std::max<u64>(B, window_cap);
        batch_id.ensure(ctx, nb, s);
        batch_g.ensure(ctx, nb, s);
        batch_depth.ensure(ctx, nb, s);
        batch_rows.ensure(ctx, u64{K} * C * W, s);
        wprefix.ensure(ctx, u64{window_cap} + 1, s);
        wbegin.ensure(ctx, window_cap, s);
        pop_flags.ensure(ctx, u64{window_cap} + 1, s);
        pos.ensure(ctx, u64{window_cap} + 1, s);
        pop_temp.ensure(ctx, bfk::pop_temp_bytes(window_cap), s);
    }

    /// Re-lays the rows out at `nw` words (lazy slots outgrew them) and rehashes the table; the batch rows follow.
    void widen(u32 nw)
    {
        DevArray<u64> grown;
        grown.ensure(ctx, std::max<u64>(rows.capacity() / W, u64{hc->count} + 1024) * nw, s);
        check(state_set::launch_relayout(rows.data(), W, grown.data(), nw, hc->count, s), "launch_relayout");
        rows = std::move(grown);
        W = nw;
        check(cudaMemsetAsync(table.data(), 0, slots * sizeof(u64), s), "cudaMemsetAsync (state table)");
        check(state_set::launch_rehash(tab(), arena(), hc->count, s), "launch_rehash");
        size_step();
        size_chunk(cap);
        gather_batch();
        ++st.widenings;
    }

    bool out_of_time()
    {
        if (o.search.control.cancel.requested())
        {
            r.status = search::SearchStatus::Cancelled;
            return true;
        }
        if (timed && Clock::now() >= deadline)
        {
            r.status = search::SearchStatus::OutOfTime;
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------------------------------------- the step
    [[nodiscard]] bfk::Pop pop_args() const
    {
        bfk::Pop p;
        p.runs = runs_of();
        p.entries = open.data();
        p.wprefix = wprefix.data();
        p.wbegin = wbegin.data();
        p.window_cap = window_cap;
        p.batch = B;
        p.max_expanded = o.search.control.budget.max_expanded;
        p.max_depth = o.search.control.budget.max_depth;
        p.mode = greedy ? bfk::k_gbfs : bfk::k_astar;
        p.single_bucket = o.single_bucket ? 1 : 0;
        return p;
    }
    void enqueue_pop_round()
    {
        const bfk::Pop p = pop_args();
        check(bfk::launch_pop_plan(p, dctl, s), "launch_pop_plan");
        check(bfk::launch_pop_flags(p, nodes(), dctl, pop_flags.data(), s), "launch_pop_flags");
        check(bfk::launch_pop_scan(pop_flags.data(), window_cap, pos.data(), pop_temp.data(), bfk::pop_temp_bytes(window_cap), s),
              "launch_pop_scan");
        check(bfk::launch_pop_take(p, nodes(), dctl, pop_flags.data(), pos.data(), batch_id.data(), batch_g.data(), batch_depth.data(), s),
              "launch_pop_take");
        check(bfk::launch_pop_finish(p, dctl, pos.data(), s), "launch_pop_finish");
    }
    void enqueue_gather()
    {
        check(bfk::launch_gather_rows(rows.data(), W, batch_id.data(), B, &dctl->n, batch_rows.data(), s), "launch_gather_rows");
    }
    /// The batch's rows again (after a widening), and their host copy when the CPU fallback needs it.
    void gather_batch()
    {
        enqueue_gather();
        if (host)
        {
            host_batch.resize(u64{hc->n} * W);
            to_host(host_batch.data(), batch_rows.data(), u64{hc->n} * W, s);
            sync();
        }
    }
    /// Chunk k's successor counts: its parents from the device count (ns = 0) or the host's ns.
    void chunk_gen(u32 k, u32 ns)
    {
        check(bfk::launch_chunk_begin(dctl, k, C, s), "launch_chunk_begin");
        const auto th = Clock::now();
        const u64 b0 = u64{k} * C;
        ChunkInput in{batch_rows.data() + b0 * W, W, W, ns ? ns : C, nullptr, W, static_cast<u32>(b0)};
        if (ns == 0)
            in.rows_dev = &dctl->rows;
        else if (host)
            in.host_states = host_batch.data() + b0 * W;
        gen->begin(in, witness, canonical);
        st.host_ms += ms_since(th);
        gen->views();
        gen->count();
    }
    [[nodiscard]] lifted::Labels labels_for(u32 k, u64 cp)
    {
        lifted::Labels lb;
        lb.parent = static_cast<u32*>(l_parent.data());
        lb.capacity = cp;
        lb.label_width = L;
        lb.parent_base = k * C;
        if (!dcosts.unit)
        {
            lb.schema = static_cast<u32*>(l_schema.data());
            lb.binding = static_cast<u32*>(l_binding.data());
        }
        u32* d_sort = sorts() ? static_cast<u32*>(l_sort.data()) : nullptr;
        lb.scratch = d_sort;
        lb.scratch_rows = d_sort ? cp : 0;
        lb.scratch_indexed = d_sort ? 1 : 0;
        lb.error = &dctl->label_error;
        lb.live = &dctl->live;
        return lb;
    }
    [[nodiscard]] bfk::Relax relax_args(u32 k) const
    {
        bfk::Relax rx;
        rx.table = tab();
        rx.arena = arena();
        rx.cand = static_cast<const u64*>(cand.data());
        rx.result = static_cast<const u32*>(result.data());
        rx.parent = static_cast<const u32*>(l_parent.data());
        rx.schema = dcosts.unit ? nullptr : static_cast<const u32*>(l_schema.data());
        rx.binding = dcosts.unit ? nullptr : static_cast<const u32*>(l_binding.data());
        rx.label_width = L;
        rx.seg_offsets = gen->seg_offsets();
        rx.num_schemas = S;
        rx.parent_base = k * C;
        rx.batch_id = batch_id.data();
        rx.batch_g = batch_g.data();
        rx.batch_depth = batch_depth.data();
        rx.costs = dcosts;
        rx.mode = greedy ? bfk::k_gbfs : bfk::k_astar;
        rx.reopen = reopen ? 1 : 0;
        rx.id = static_cast<u32*>(c_id.data());
        rx.gc = static_cast<u32*>(c_gc.data());
        rx.part = static_cast<u8*>(c_part.data());
        rx.push = static_cast<u8*>(c_push.data());
        return rx;
    }
    /// Dedup (ids in candidate order, the new states after the stored ones), the new states' rows, the relaxation.
    void chunk_commit(u32 k, u64 cp)
    {
        const state_set::Live live{&dctl->live};
        auto* d_cand = static_cast<u64*>(cand.data());
        auto* d_result = static_cast<u32*>(result.data());
        auto* d_rank = static_cast<u32*>(rank.data());
        check(state_set::launch_insert(tab(), arena(), {d_cand, W}, cp, d_result, s, live), "launch_insert");
        check(state_set::launch_rank(tab(), d_result, cp, d_rank, rank_temp.data(), state_set::rank_temp_bytes(cp), s, live),
              "launch_rank");
        state_set::Compact cm;
        cm.result = d_result;
        cm.rank = d_rank;
        cm.cand = d_cand;
        cm.parent = static_cast<const u32*>(l_parent.data());
        cm.seg_offsets = gen->seg_offsets();
        cm.num_schemas = S;
        cm.parent_base = k * C;
        cm.arena_tail = rows.data();
        cm.nodes_tail = nullptr;
        cm.base = 0;
        cm.words = W;
        cm.offset = &dctl->count;
        check(state_set::launch_compact(tab(), cm, cp, s, live), "launch_compact");
        check(state_set::launch_advance(d_rank, cp, &dctl->count, &dctl->fresh, s), "launch_advance");
        check(bfk::launch_fresh(nodes(), dctl, arena(), static_cast<u64*>(fresh_rows.data()), cp, s), "launch_fresh");
        const bfk::Relax rx = relax_args(k);
        check(bfk::launch_relax(rx, nodes(), dctl, cp, s), "launch_relax");
        check(bfk::launch_win(rx, nodes(), dctl, cp, s), "launch_win");
        check(bfk::launch_reset(rx, nodes(), dctl, cp, s), "launch_reset");
    }
    /// The goal test of the chunk's new states on the device.
    void chunk_goals(u64 cp)
    {
        auto* gu = static_cast<u8*>(goal_u8.data());
        check(lifted::launch_goal_rows(gen->view(), static_cast<const u64*>(fresh_rows.data()), W, cp, &dctl->fresh, gu, s),
              "launch_goal_rows");
        check(bfk::launch_set_goals(nodes(), dctl, gu, cp, s), "launch_set_goals");
    }
    /// The goal test on the host (goals with derived literals).
    void chunk_goals_host(u64 cp)
    {
        read();
        const u64 fresh = hc->fresh;
        std::vector<u8> hg(std::max<u64>(fresh, 1), 0);
        if (fresh)
        {
            host_rows.resize(fresh * W);
            to_host(host_rows.data(), static_cast<const u64*>(fresh_rows.data()), fresh * W, s);
            sync();
            const WorkspaceLease lease = task->workspace();
            Successors& succ = lease->successors();
            for (u64 i = 0; i < fresh; ++i)
            {
                const u64* w = host_rows.data() + i * W;
                hg[i] = succ.is_goal(StateView{w, bits::trimmed_size(w, W), nullptr, 0}) ? 1 : 0;
            }
            to_device(static_cast<u8*>(goal_u8.data()), hg.data(), fresh, s);
        }
        check(bfk::launch_set_goals(nodes(), dctl, static_cast<const u8*>(goal_u8.data()), cp, s), "launch_set_goals");
        sync();  // (hg is pageable)
    }
    /// GBFS's goal, the heuristic of the new states, their keys, the sorted push, A*'s goal, the append. host_driven: not
    /// captured (the heuristic refreshes its lazy slots' tables: the chunk may have interned atoms).
    void chunk_eval(u32 k, u64 cp, bool host_driven)
    {
        const bfk::Relax rx = relax_args(k);
        const auto* d_rank = static_cast<const u32*>(rank.data());
        if (greedy)
            check(bfk::launch_first_goal(rx, nodes(), dctl, cp, s), "launch_first_goal");
        check(bfk::launch_goal_gate(dctl, k, true, s), "launch_goal_gate");
        if (greedy)
            check(bfk::launch_goal_info(rx, d_rank, dctl, k, s), "launch_goal_info");
        auto* hf = static_cast<u32*>(h_fresh.data());
        if (heur)
        {
            const auto te = Clock::now();
            if (async_h && host_driven)
                heur->prepare(cp, s);
            if (async_h)
                heur->evaluate_async(static_cast<const u64*>(fresh_rows.data()), W, W, cp, &dctl->heur_n, hf, dctl->eval_flags, s);
            else
            {
                read();
                heur->evaluate(static_cast<const u64*>(fresh_rows.data()), W, W, hc->heur_n, hf, s);
            }
            st.heuristic_ms += ms_since(te);
        }
        check(bfk::launch_settle(nodes(), dctl, heur ? hf : nullptr, cp, s), "launch_settle");
        bfk::Push p;
        p.id = rx.id;
        p.gc = rx.gc;
        p.push = rx.push;
        p.mode = rx.mode;
        p.keys = static_cast<u64*>(p_keys.data());
        p.entries = static_cast<u64*>(p_entries.data());
        check(bfk::launch_push_keys(p, nodes(), dctl, cp, s), "launch_push_keys");
        check(bfk::launch_sort_runs(p.keys, p.entries, cp, static_cast<u64*>(s_keys.data()), static_cast<u64*>(s_entries.data()),
                                    static_cast<u64*>(r_keys.data()), static_cast<u32*>(r_counts.data()), &dctl->new_runs,
                                    sort_temp.data(), bfk::sort_temp_bytes(cp), s),
              "launch_sort_runs");
        if (!greedy)
        {
            check(bfk::launch_first_goal(rx, nodes(), dctl, cp, s), "launch_first_goal");
            check(bfk::launch_goal_gate(dctl, k, false, s), "launch_goal_gate");
            check(bfk::launch_goal_info(rx, d_rank, dctl, k, s), "launch_goal_info");
        }
        bfk::Append a;
        a.runs = runs_of();
        a.runs_key = static_cast<const u64*>(r_keys.data());
        a.runs_count = static_cast<const u32*>(r_counts.data());
        a.new_begin = static_cast<u64*>(new_begin.data());
        a.sorted_entries = static_cast<const u64*>(s_entries.data());
        a.arena = open.data();
        a.cands = cp;
        a.max_states = o.search.control.budget.max_states;
        a.mode = rx.mode;
        check(bfk::launch_append(a, dctl, s), "launch_append");
        check(bfk::launch_chunk_end(a, dctl, k, s), "launch_chunk_end");
    }
    /// Chunk k of a step without host work (graphs, device loops): everything sized by `cap`.
    void enqueue_chunk(u32 k)
    {
        chunk_gen(k, 0);
        check(bfk::launch_chunk_check(dctl, gen->seg_offsets() + u64{C} * S, k, limits(cap), s), "launch_chunk_check");
        gen->write(labels_for(k, cap), static_cast<u64*>(cand.data()), W);
        if (const u32* miss = gen->missing_flag())
            check(bfk::launch_chunk_missing(dctl, miss, k, s), "launch_chunk_missing");
        chunk_commit(k, cap);
        chunk_goals(cap);
        chunk_eval(k, cap, false);
    }
    void enqueue_step(unsigned long long handle)
    {
        enqueue_pop_round();
        enqueue_gather();
        for (u32 k = 0; k < K; ++k)
            enqueue_chunk(k);
        check(bfk::launch_step_end(dctl, pop_args(), limits(cap), K, C, handle, s), "launch_step_end");
    }

    /// Chunk k driven by the host: exact sizes (the candidates read after the counts), the CPU fallback, lazy slots,
    /// derived goals and the synchronous heuristic as needed.
    void host_chunk(u32 k)
    {
        const u32 ns = static_cast<u32>(std::min<u64>(hc->n - u64{k} * C, C));
        for (;;)  // redone after a width growth
        {
            chunk_gen(k, ns);
            u32* pm = reinterpret_cast<u32*>(pin.data(1));
            to_host(pm, gen->seg_offsets() + u64{ns} * S, 1, s);
            read();
            const u64 M = *pm;
            remember(M);
            const u64 cp = std::max<u64>(M, 1);
            size_chunk(cp);
            ensure_room(cp);
            check(bfk::launch_chunk_check(dctl, gen->seg_offsets() + u64{ns} * S, k, limits(cp), s), "launch_chunk_check");
            const lifted::Labels lb = labels_for(k, cp);
            gen->write(lb, static_cast<u64*>(cand.data()), W);
            bool widened = false;
            for (;;)
            {
                const auto tm = Clock::now();
                const bool missing = gen->resolve_missing();
                st.host_ms += ms_since(tm);
                if (!missing)
                    break;
                if (bucket(task->words()) > W)
                {
                    widen(bucket(task->words()));
                    widened = true;
                    break;
                }
                gen->write(lb, static_cast<u64*>(cand.data()), W);
            }
            if (widened)
                continue;  // redo the chunk at the new width (nothing of it was committed)
            chunk_commit(k, cp);
            if (host_goal)
                chunk_goals_host(cp);
            else
                chunk_goals(cp);
            chunk_eval(k, cp, true);
            read();
            return;
        }
    }
    /// A step driven by the host, or the rest of one: from its pop (rounds after the first when it continues a pop that
    /// ran out of window), or from chunk `from` after its pop and gather.
    void host_step(bool pop, u32 from)
    {
        ++st.host_steps;
        if (pop)
        {
            for (;;)
            {
                enqueue_pop_round();
                read();
                if (!(hc->abort & bfk::k_abort_pop))
                    break;
                clear_abort();
            }
            if (hc->status == bfk::k_running && hc->n > 0 && !hc->goal_pop)
                gather_batch();
        }
        if (hc->status == bfk::k_running && hc->n > 0 && !hc->goal_pop)
        {
            const u32 chunks = static_cast<u32>((u64{hc->n} + C - 1) / C);
            for (u32 k = from; k < chunks && hc->status == bfk::k_running; ++k)
                host_chunk(k);
        }
        check(bfk::launch_step_end(dctl, pop_args(), limits(cap), K, C, 0, s), "launch_step_end");
        read();
        remember_batch();
    }

    /// Records a chunk's candidates for the capacity estimate.
    void remember(u64 M) { recent[recent_at++ % recent.size()] = M; }
    /// Records the batch of the last step (host steps, and a device loop's last step).
    void remember_batch()
    {
        batch_prev = batch_last;
        batch_last = hc->n;
    }
    /// Whether the batches settled (device steps sized by estimate() then rarely outgrow their capture): the last step's
    /// batch was full or at most 1.5x the one before (plus a few). While the batches grow (a blind search's first layers
    /// grow 2-10x per step) the host drives the steps.
    [[nodiscard]] bool steady() const
    {
        return recent_at > 0 && (batch_last >= B || 2 * u64{batch_last} <= 3 * u64{batch_prev} + 64);
    }
    u32 batch_last = 0, batch_prev = 0;
    /// The capacity of the device steps' chunks: the most candidates of the recent chunks with a margin, quantized to
    /// powers of two (so that captured steps are reused).
    [[nodiscard]] u64 estimate() const
    {
        const u64 m = std::max<u64>(*std::ranges::max_element(recent), hc->next_M);
        return quantize(m + m / 4 + 256);
    }
    /// Steps replayed from a captured graph, looped on the device: one read per loop. An abort hands the step to the
    /// host.
    void device_round()
    {
        cap = estimate();
        size_chunk(cap);
        ensure_room(u64{K} * cap * 4);  // (a loop runs until the next step might not fit)
        gen->prepare(C);
        std::vector<u64> key = {cap,
                                W,
                                slots,
                                reinterpret_cast<u64>(table.data()),
                                reinterpret_cast<u64>(rows.data()),
                                node_cap,
                                reinterpret_cast<u64>(g.data()),
                                reinterpret_cast<u64>(h.data()),
                                reinterpret_cast<u64>(parent.data()),
                                reinterpret_cast<u64>(sidx.data()),
                                reinterpret_cast<u64>(depth.data()),
                                reinterpret_cast<u64>(flags.data()),
                                reinterpret_cast<u64>(best.data()),
                                open.capacity(),
                                reinterpret_cast<u64>(open.data()),
                                runs_cap,
                                reinterpret_cast<u64>(run_key.data()),
                                reinterpret_cast<u64>(run_begin.data()),
                                reinterpret_cast<u64>(run_count.data()),
                                reinterpret_cast<u64>(run_key2.data()),
                                reinterpret_cast<u64>(run_begin2.data()),
                                reinterpret_cast<u64>(run_count2.data()),
                                reinterpret_cast<u64>(batch_rows.data()),
                                reinterpret_cast<u64>(batch_id.data()),
                                reinterpret_cast<u64>(batch_g.data()),
                                reinterpret_cast<u64>(batch_depth.data()),
                                gen->capture_key()};
        for (const Scratch* x : {&l_parent, &l_schema, &l_binding, &l_sort, &cand, &result, &rank, &rank_temp, &c_id, &c_gc, &c_part,
                                 &c_push, &p_keys, &p_entries, &s_keys, &s_entries, &r_keys, &r_counts, &sort_temp, &new_begin,
                                 &fresh_rows, &h_fresh, &goal_u8})
            key.push_back(reinterpret_cast<u64>(x->data()));
        if (!step_graph || key != graph_key)
        {
            step_graph = GraphExec();
            if (loops)
                step_graph = GraphExec::capture_while(s, [&](unsigned long long handle) { enqueue_step(handle); });
            else
                step_graph = GraphExec::capture(s, [&] { enqueue_step(0); });
            graph_key = std::move(key);
            ++st.captures;
        }
        // the loop's steps and its chunk maximum
        hc->steps_left = loops ? std::max<u32>(1, o.loop_steps) : 1;
        hc->loop_max_M = 0;
        check(cudaMemcpyAsync(&dctl->steps_left, &hc->steps_left, sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        check(cudaMemcpyAsync(&dctl->loop_max_M, &hc->loop_max_M, sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        const u64 steps_before = hc->steps;
        step_graph.launch(s);
        if (loops)
            ++st.loops;
        read();
        remember(hc->loop_max_M);
        if (!hc->abort)
            remember_batch();
        st.graph_steps += hc->steps - steps_before;
        if (hc->abort)
        {
            ++st.aborts;
            const u32 a = hc->abort, k = hc->stop_chunk;
            clear_abort();
            if (a & bfk::k_abort_pop)
            {
                host_step(true, 0);  // (the pop's next rounds, then the step)
                return;
            }
            if (a & bfk::k_abort_missing)
            {
                (void)gen->resolve_missing();
                if (bucket(task->words()) > W)
                    widen(bucket(task->words()));
            }
            host_step(false, k);
        }
    }

    bool start(const State& s0);
    std::vector<Action> plan_to(u64 id, const State& s0);
    void solved(u64 id, const State& s0);
    /// The search ends at the goal candidate of Ctl::stop_chunk (after its parent's expansion, as the CPU's: GBFS its
    /// goal test on generation; A* a goal of the current f layer is the minimum and popped next): the chunk's later
    /// parents count as not expanded, their successors as not generated, their new states as not stored.
    void stop_at_goal(const State& s0);

    DeviceBestFirstResult run();
};

bool Driver::start(const State& s0)
{
    W = bucket(std::max(task->words(), s0.size_words()));
    ensure_nodes(0, 1024);
    std::vector<u64> row(W, 0);
    std::copy_n(s0.data(), s0.size_words(), row.begin());
    to_device(rows.data(), row.data(), W, s);
    hc->count = 1;
    alloc_table(o.expected_states ? std::bit_ceil(std::max<u64>(u64{1} << 16, 2 * o.expected_states + 2)) : u64{1} << 20);
    check(state_set::launch_rehash(tab(), arena(), 1, s), "launch_rehash");
    const u32 root[5] = {0, 0, bfk::k_none, 0, 0};  // g, h, parent, sidx, depth
    to_device(g.data(), root + 0, 1, s);
    to_device(parent.data(), root + 2, 1, s);
    to_device(sidx.data(), root + 3, 1, s);
    to_device(depth.data(), root + 4, 1, s);
    // h of the root
    const auto th = Clock::now();
    if (heur)
        heur->evaluate(rows.data(), W, W, 1, h.data(), s);
    else
        check(cudaMemsetAsync(h.data(), 0, sizeof(u32), s), "cudaMemsetAsync");
    st.heuristic_ms += ms_since(th);
    ++r.evaluations;
    u32 h0 = 0;
    to_host(&h0, h.data(), 1, s);
    sync();
    r.initial_h = h0 == bfk::k_inf ? std::numeric_limits<f64>::infinity() : static_cast<f64>(h0);
    if (h0 == bfk::k_inf)
    {
        ++r.dead_ends;
        r.status = search::SearchStatus::Unsolvable;
        return false;
    }
    const u8 f0 = bfk::k_open;
    to_device(flags.data(), &f0, 1, s);
    const u64 key = greedy ? u64{h0} << 32 : (u64{h0} << 32) | (u64{h0} + 1);
    const u64 entry = 0;  // g 0, id 0
    *hc = bfk::Ctl{};
    hc->open_size = 1;
    hc->runs = 1;
    hc->count = 1;
    hc->open_entries = 1;
    ensure_room(1024);
    to_device(open.data(), &entry, 1, s);
    to_device(run_key.data(), &key, 1, s);
    const u64 zero = 0;
    const u32 one = 1;
    to_device(run_begin.data(), &zero, 1, s);
    to_device(run_count.data(), &one, 1, s);
    write_ctl();
    sync();
    return true;
}

std::vector<Action> Driver::plan_to(u64 id, const State& s0)
{
    std::vector<std::pair<u64, u32>> chain;  // (state, successor index within its parent)
    for (u64 v = id;;)
    {
        u32 rec[2];
        to_host(rec, parent.data() + v, 1, s);
        to_host(rec + 1, sidx.data() + v, 1, s);
        sync();
        if (rec[0] == bfk::k_none)
            break;
        chain.emplace_back(v, rec[1]);
        v = rec[0];
    }
    std::reverse(chain.begin(), chain.end());
    std::vector<Action> plan;
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    std::vector<u64> prow(W, 0), crow(W), tmp;
    std::copy_n(s0.data(), std::min(s0.size_words(), W), prow.begin());
    for (const auto& [v, k] : chain)
    {
        to_host(crow.data(), rows.data() + v * W, W, s);
        sync();
        const u32 nw = bits::trimmed_size(prow.data(), W);
        succ.prepare(StateView{prow.data(), nw, nullptr, 0});
        u32 i = 0;
        bool found = false;
        succ.generate<true>(
            [&](u32 schema, const ObjectId* b, const Delta& d) -> bool
            {
                if (i++ != k)
                    return true;
                const u32 m = apply_delta(prow.data(), nw, d, tmp);
                tmp.resize(std::max<usize>(tmp.size(), W), 0);
                found = m <= W && std::equal(tmp.begin(), tmp.begin() + m, crow.begin()) &&
                        std::all_of(crow.begin() + m, crow.end(), [](u64 x) { return x == 0; });
                plan.emplace_back(SchemaId{schema}, std::vector<ObjectId>(b, b + succ.arity(schema)));
                return false;
            },
            witness, canonical);
        if (!found)
            throw std::logic_error("mymyr: device best-first search: the plan replay does not reproduce state " + std::to_string(v) +
                                   " (device and CPU successor orders differ)");
        prow = crow;
    }
    return plan;
}

void Driver::solved(u64 id, const State& s0)
{
    r.status = search::SearchStatus::Solved;
    r.plan = plan_to(id, s0);
    u32 gid = 0;
    to_host(&gid, g.data() + id, 1, s);
    std::vector<u64> row(W);
    to_host(row.data(), rows.data() + id * W, W, s);
    sync();
    r.cost = g0 + gid;
    r.goal_state = State(row.data(), W);
    if (greedy && !costs->unit())
    {
        const WorkspaceLease lease = task->workspace();
        r.cost = heuristics::plan_metric(lease->successors(), *costs, s0, g0, r.plan);
    }
}

void Driver::stop_at_goal(const State& s0)
{
    const u64 b0 = u64{hc->stop_chunk} * C;
    r.stats.expanded += hc->goal_parent - b0 + 1;
    r.stats.generated += hc->goal_end;
    if (!greedy)  // the chunk's new states were evaluated: those after the end are not
    {
        const u64 first = u64{hc->goal_base} + hc->goal_fresh_before;
        const u64 n = hc->goal_fresh - hc->goal_fresh_before;
        r.evaluations -= n;
        auto* d = static_cast<u32*>(misc.ensure(ctx, sizeof(u32), s));
        check(cudaMemsetAsync(d, 0, sizeof(u32), s), "cudaMemsetAsync");
        check(bfk::launch_count_dead(nodes(), first, n, d, s), "launch_count_dead");
        u32 dead_after = 0;
        to_host(&dead_after, d, 1, s);
        sync();
        r.dead_ends -= dead_after;
        r.stats.pruned -= dead_after;
    }
    states_at_goal = u64{hc->goal_base} + hc->goal_fresh_before;
    solved(hc->goal_id, s0);
}

DeviceBestFirstResult Driver::run()
{
    DeviceGuard guard(ctx->device());
    const auto t_setup = Clock::now();
    r.algorithm = greedy ? "gbfs_eager (device)" : "astar_eager (device)";
    r.store = "device";
    r.queue = "device buckets";
    s = ctx->stream();
    const search::BestFirstOptions& so = o.search;
    const search::Budget& b = so.control.budget;
    timed = std::isfinite(b.max_seconds);
    if (timed)
        deadline = t_setup + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(0.0, b.max_seconds)));
    witness = so.witness_pruning;
    canonical = so.canonical_order;
    reopen = so.reopen;
    costs = std::make_unique<heuristics::ActionCosts>(*task);
    g0 = costs->initial();
    const State s0 = so.start ? *so.start : task->initial_state();
    pin = ctx->lease_pinned(2, sizeof(bfk::Ctl));
    hc = static_cast<bfk::Ctl*>(pin.data(0));
    *hc = bfk::Ctl{};
    ctl = DeviceBuffer(ctx, sizeof(bfk::Ctl), s);
    dctl = static_cast<bfk::Ctl*>(ctl.data());
    // the device's action costs
    dcosts.unit = costs->unit() ? 1 : 0;
    if (!costs->unit())
    {
        const Task* t = task.get();
        const heuristics::ActionCosts* c = costs.get();
        cost_programs = cuda::costs::build(ctx, std::span(&t, 1), std::span(&c, 1), s);
        if (!cost_programs.why.empty() || !cost_programs.integral)
            throw std::invalid_argument("mymyr: the CUDA backend cannot run this search: " +
                                        (cost_programs.why.empty() ? std::string("action costs that are not integral") : cost_programs.why));
        dcosts.program = cost_programs.view;
    }
    // the heuristic: without a synchronization when the search starts at the initial state of the task whose
    // reachable states its grounding covers
    if (so.heuristic.kind != heuristics::Kind::Blind)
    {
        DeviceHeuristicOptions ho = o.heuristic;
        ho.kind = so.heuristic.kind;
        ho.costs = so.heuristic.costs;
        ho.budget = so.heuristic.budget;
        if (so.heuristic.relaxed)
            ho.relaxed = so.heuristic.relaxed;
        heur = std::make_unique<DeviceHeuristic>(ctx, task, ho);
    }
    async_h = !so.start || so.start->view() == task->initial_state().view();
    gen = std::make_unique<ChunkGenerator>(ctx, task, s);
    S = gen->num_schemas();
    L = gen->label_width();
    host = gen->needs_host(witness);
    host_goal = task->compiled().goal.uses_derived;
    {
        int l2 = 0;
        check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device()), "cudaDeviceGetAttribute");
        const u64 per = std::max<u64>(8, gen->view_words() * 8);
        const u64 chunk_cap = o.chunk_states ? o.chunk_states : std::max<u64>(8192, std::max<u64>(static_cast<u64>(l2) / 2, 1) / per);
        B = o.batch;
        C = static_cast<u32>(std::min<u64>(chunk_cap, B));
        K = (B + C - 1) / C;
        window_cap = static_cast<u32>(std::max<u64>(2 * u64{B}, 1024));
    }
    // steps without host work run as captured graphs, looped on the device without a time budget (a step of more than
    // k_max_graph_chunks chunks runs from the host)
    constexpr u32 k_max_graph_chunks = 16;
    graphs = o.graphs && GraphExec::enabled() && async_h && !host_goal && gen->capturable(witness) && K <= k_max_graph_chunks;
    loops = graphs && GraphExec::loops_enabled() && !timed;
    r.setup_seconds = s_since(t_setup);
    t0 = Clock::now();

    auto finish = [&]
    {
        sync();
        r.stats.states = states_at_goal ? states_at_goal : hc->count;
        r.stats.seconds = s_since(t0);
        if (r.status != search::SearchStatus::Solved)
        {
            r.plan.clear();
            r.cost = 0;
            r.goal_state.reset();
        }
        if (heur)
        {
            st.heuristic = heur->stats();
            st.heuristic.evaluations = r.evaluations;  // (the device loops' evaluations are counted here)
            r.heuristic.evaluations = st.heuristic.evaluations;
            r.heuristic.grounded = st.heuristic.evaluations - st.heuristic.fallbacks;
            r.heuristic.lifted = st.heuristic.fallbacks;
            r.heuristic.grounding_seconds = st.heuristic.grounding_seconds;
        }
        r.heuristic.dead_ends = r.dead_ends;
        st.uploads = gen ? gen->uploads() : 0;
        st.table_slots = slots;
        st.device_bytes = ctx->usage().used_high;
        r.store_bytes = rows.bytes() + g.bytes() * 5 + flags.bytes() + best.bytes() + open.bytes() + slots * 8;
        r.fluent_slots = task->atoms().fluent_slots();
        return DeviceBestFirstResult{std::move(r), st};
    };

    if (task->compiled().goal.unsatisfiable)
    {
        r.status = search::SearchStatus::Unsolvable;
        return finish();
    }
    if (task->is_goal(s0.view()))
    {
        W = bucket(std::max(task->words(), s0.size_words()));
        hc->count = 1;
        r.status = search::SearchStatus::Solved;
        r.cost = g0;
        r.goal_state = s0;
        return finish();
    }
    if (!start(s0))
        return finish();
    size_step();
    for (;;)
    {
        if (hc->head == hc->runs)
        {
            r.status = search::SearchStatus::Exhausted;
            break;
        }
        if (out_of_time())
            break;
        // the first steps run on the host (exact sizes) until the chunks' candidates settle
        if (graphs && steady())
            device_round();
        else
            host_step(true, 0);
        if (hc->status != bfk::k_running)
            break;
    }
    // the statistics the device counted (the root's evaluation and dead end were counted above)
    r.stats.expanded = hc->expanded;
    r.stats.generated = hc->generated;
    r.stats.pruned = hc->pruned;
    r.evaluations += hc->evaluations;
    r.dead_ends += hc->dead_ends;
    r.reopened = hc->reopened;
    st.popped = hc->popped;
    st.stale = hc->stale;
    st.steps = hc->steps;
    st.chunks = hc->chunks;
    st.max_batch = hc->max_batch;
    st.open_entries = hc->open_entries;
    switch (hc->status)
    {
        case bfk::k_solved:
        {
            u32 id = 0;
            to_host(&id, batch_id.data(), 1, s);
            sync();
            solved(id, s0);
            break;
        }
        case bfk::k_goal_found: stop_at_goal(s0); break;
        case bfk::k_exhausted: r.status = search::SearchStatus::Exhausted; break;
        case bfk::k_out_of_states:
        case bfk::k_out_of_expanded: r.status = search::SearchStatus::OutOfStates; break;
        default: break;  // (out of time or cancelled: set above)
    }
    return finish();
}
}  // namespace

namespace detail
{
DeviceBestFirstResult numeric_best_first(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options, bool greedy);

DeviceBestFirstResult best_first(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options, bool greedy)
{
    if (!ctx || !task)
        throw std::invalid_argument("mymyr: device best-first search: null context or task");
    if (const std::string why = best_first_unsupported(*task, options); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run this search: " + why);
    if (task->numeric_slots())
        return numeric_best_first(std::move(ctx), std::move(task), options, greedy);
    Driver d(std::move(ctx), std::move(task), options, greedy);
    return d.run();
}
}  // namespace detail

DeviceBestFirstResult astar(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options)
{
    return detail::best_first(std::move(ctx), std::move(task), options, false);
}
}  // namespace mymyr::cuda
