#include "mymyr/cuda/astar.hpp"

#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/numeric.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"
#include "mymyr/cuda/state_set.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace mymyr::cuda::detail
{
namespace
{
using Clock = std::chrono::steady_clock;
constexpr u32 none = ~u32{0};

u64 key(f64 v)
{
    const u64 b = std::bit_cast<u64>(v + 0.0);
    return b >> 63 ? ~b : b | (u64{1} << 63);
}
struct Entry
{
    u64 primary, secondary, sequence;
    u32 id;
    bool goal;
};
struct Later
{
    bool operator()(const Entry& a, const Entry& b) const
    {
        return std::tuple(a.primary, !a.goal, a.secondary, a.sequence) >
               std::tuple(b.primary, !b.goal, b.secondary, b.sequence);
    }
};
struct Node
{
    f64 g = 0, h = 0;
    u32 parent = none, depth = 0;
    Action action;
    bool open = false, closed = false, goal = false;
};

template<class T>
T* allocate(Scratch& scratch, const ContextPtr& ctx, u64 n, cudaStream_t s)
{
    return static_cast<T*>(scratch.ensure(ctx, std::max<u64>(n, 1) * sizeof(T), s));
}
template<class T>
void upload(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n) check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (numeric search upload)");
}
template<class T>
void download(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n) check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (numeric search download)");
}
}  // namespace

// Numeric priorities use the CPU's total order over doubles. The host heap expands one parent at a time to preserve
// eager search order; successor programs, metric evaluation, state equality and relaxation heuristics run on the device.
DeviceBestFirstResult numeric_best_first(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& o, bool greedy)
{
    DeviceGuard guard(ctx->device());
    const cudaStream_t s = ctx->stream();
    const auto setup = Clock::now();
    DeviceBestFirstResult out;
    auto& r = out.result;
    auto& st = out.device;
    r.algorithm = greedy ? "gbfs_eager" : "astar_eager";
    r.store = "device";
    r.queue = "heap";
    ChunkGenerator gen(ctx, task, s);
    heuristics::ActionCosts costs(*task);
    std::unique_ptr<DeviceHeuristic> heuristic;
    if (o.search.heuristic.kind != heuristics::Kind::Blind)
    {
        DeviceHeuristicOptions ho = o.heuristic;
        ho.kind = o.search.heuristic.kind;
        ho.costs = o.search.heuristic.costs;
        ho.budget = o.search.heuristic.budget;
        ho.relaxed = o.search.heuristic.relaxed;
        heuristic = std::make_unique<DeviceHeuristic>(ctx, task, ho);
    }
    const State start = o.search.start ? *o.search.start : task->initial_state();
    const f64 g0 = costs.initial(start.view());
    const u32 slots = task->numeric_slots(), L = gen.label_width();
    u32 W = std::max<u32>(1, std::max(task->words(), start.size_words()) + slots);
    if (W > lifted::k_max_words)
        throw std::invalid_argument("mymyr: device numeric search: state rows exceed the device width limit");
    std::vector<Node> nodes(1);
    nodes[0].g = g0;
    std::priority_queue<Entry, std::vector<Entry>, Later> open;
    u64 sequence = 0, arena_capacity = 0, table_capacity = 0;
    DeviceBuffer arena, table;
    Scratch candidates, labels_binding, labels_schema, labels_parent, flags, values, parent_value;
    Scratch result, rank, scan, ids, control;
    std::vector<u64> host_row(W);
    numeric::encode(*task, start.view(), host_row.data(), W);
    auto sync = [&] { check(cudaStreamSynchronize(s), "cudaStreamSynchronize (numeric search)"); };
    auto ensure = [&](u64 capacity, u32 width)
    {
        if (width > lifted::k_max_words)
            throw std::invalid_argument("mymyr: device numeric search: states grew beyond the device width limit");
        if (capacity > state_set::k_max_states)
            throw std::length_error("mymyr: device numeric search exceeds the state id limit");
        bool rehash = false;
        if (capacity > arena_capacity || width != W)
        {
            const u64 cap = std::max<u64>({capacity, arena_capacity + arena_capacity / 2, 64});
            DeviceBuffer next(ctx, cap * width * sizeof(u64), s);
            if (arena.data())
            {
                if (width == W)
                    check(cudaMemcpyAsync(next.data(), arena.data(), nodes.size() * W * sizeof(u64), cudaMemcpyDeviceToDevice, s),
                          "cudaMemcpyAsync (numeric search arena)");
                else
                {
                    auto view = gen.view();
                    view.numeric.storage = 0;
                    check(numeric::launch_convert(view, static_cast<const u64*>(arena.data()), W, W - slots,
                                                  static_cast<u64*>(next.data()), width, width - slots,
                                                  nodes.size(), true, s), "numeric::launch_convert");
                    ++st.widenings;
                    rehash = true;
                }
            }
            arena = std::move(next);
            arena_capacity = cap;
            W = width;
        }
        const u64 table_need = std::bit_ceil(std::max<u64>(1024, capacity * 2));
        if (table_need > table_capacity)
        {
            table_capacity = table_need;
            table = DeviceBuffer(ctx, table_capacity * sizeof(u64), s);
            rehash = true;
        }
        if (rehash)
        {
            check(cudaMemsetAsync(table.data(), 0, table_capacity * sizeof(u64), s), "cudaMemsetAsync (numeric search table)");
            if (arena.data() && st.open_entries)
                check(state_set::launch_rehash({static_cast<u64*>(table.data()), table_capacity - 1},
                                               {static_cast<const u64*>(arena.data()), W}, nodes.size(), s), "state_set::launch_rehash");
            ++st.rehashes;
        }
    };
    ensure(std::max<u64>(o.expected_states, 1), W);
    upload(static_cast<u64*>(arena.data()), host_row.data(), W, s);
    check(state_set::launch_rehash({static_cast<u64*>(table.data()), table_capacity - 1},
                                   {static_cast<const u64*>(arena.data()), W}, 1, s), "state_set::launch_rehash");
    auto load = [&](u32 id)
    {
        host_row.resize(W);
        download(host_row.data(), static_cast<const u64*>(arena.data()) + u64{id} * W, W, s);
        sync();
        return numeric::decode(*task, host_row.data(), W);
    };
    auto push = [&](u32 id)
    {
        const Node& n = nodes[id];
        open.push({key(greedy ? n.h : (n.g - g0) + n.h), key(greedy ? n.g - g0 : n.h), sequence++, id, !greedy && n.goal});
        ++st.open_entries;
    };
    auto solved = [&](u32 id)
    {
        r.status = search::SearchStatus::Solved;
        r.goal_state = load(id);
        r.cost = nodes[id].g;
        for (u32 n = id; nodes[n].parent != none; n = nodes[n].parent)
            r.plan.push_back(nodes[n].action);
        std::reverse(r.plan.begin(), r.plan.end());
        if (greedy && !costs.unit())
        {
            const WorkspaceLease lease = task->workspace();
            r.cost = heuristics::plan_metric(lease->successors(), costs, start, g0, r.plan);
        }
    };
    auto evaluate = [&](const u64* rows, u64 count, std::vector<f64>& h)
    {
        h.assign(count, 0);
        if (heuristic && count)
        {
            auto* d = allocate<f64>(values, ctx, count, s);
            heuristic->evaluate(rows, W, W - slots, count, d, s);
            download(h.data(), d, count, s);
            sync();
        }
    };
    const auto begin = Clock::now();
    r.setup_seconds = std::chrono::duration<double>(begin - setup).count();
    auto finish = [&]() -> DeviceBestFirstResult
    {
        sync();
        r.stats.states = nodes.size();
        r.stats.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        r.fluent_slots = task->atoms().fluent_slots();
        r.store_bytes = arena.size() + table.size() + nodes.size() * sizeof(Node) + open.size() * sizeof(Entry);
        st.steps = st.host_steps;
        st.max_batch = st.steps ? 1 : 0;
        st.uploads = gen.uploads();
        st.table_slots = table_capacity;
        st.device_bytes = ctx->usage().used_high;
        st.host_ms = gen.stats().host_axiom_ms;
        if (heuristic) st.heuristic = heuristic->stats();
        r.heuristic.evaluations = r.evaluations;
        r.heuristic.dead_ends = r.dead_ends;
        gen.mark_last_use();
        return std::move(out);
    };
    if (task->compiled().goal.unsatisfiable)
    {
        r.status = search::SearchStatus::Unsolvable;
        return finish();
    }
    if (task->is_goal(start.view()))
    {
        solved(0);
        return finish();
    }
    std::vector<f64> hs;
    evaluate(static_cast<const u64*>(arena.data()), 1, hs);
    ++r.evaluations;
    nodes[0].h = r.initial_h = hs[0];
    if (std::isinf(hs[0]))
    {
        ++r.dead_ends;
        r.status = search::SearchStatus::Unsolvable;
        return finish();
    }
    nodes[0].open = true;
    push(0);
    const auto& budget = o.search.control.budget;
    while (!open.empty())
    {
        const Entry e = open.top();
        if (!greedy && e.goal && nodes[e.id].open && key((nodes[e.id].g - g0) + nodes[e.id].h) == e.primary)
        {
            solved(e.id);
            break;
        }
        if (r.stats.expanded >= budget.max_expanded)
        {
            r.status = search::SearchStatus::OutOfStates;
            break;
        }
        if (std::chrono::duration<double>(Clock::now() - begin).count() >= budget.max_seconds)
        {
            r.status = search::SearchStatus::OutOfTime;
            break;
        }
        if (o.search.control.cancel.requested())
        {
            r.status = search::SearchStatus::Cancelled;
            break;
        }
        open.pop();
        ++st.popped;
        const u32 parent = e.id;
        if (!nodes[parent].open || (!greedy && key((nodes[parent].g - g0) + nodes[parent].h) != e.primary))
        {
            ++st.stale;
            continue;
        }
        nodes[parent].open = false;
        nodes[parent].closed = true;
        if (nodes[parent].depth >= budget.max_depth)
        {
            ++r.stats.pruned;
            continue;
        }
        ++r.stats.expanded;
        ++st.host_steps;
        ++st.chunks;
        if (gen.needs_host(o.search.witness_pruning)) (void)load(parent);
        ChunkInput input{static_cast<const u64*>(arena.data()) + u64{parent} * W, W, W, 1,
                         gen.needs_host(o.search.witness_pruning) ? host_row.data() : nullptr, W};
        gen.begin(input, o.search.witness_pruning, o.search.canonical_order);
        gen.views();
        gen.count();
        u32 T = 0;
        download(&T, gen.seg_offsets() + gen.num_schemas(), 1, s);
        sync();
        if (!T) continue;
        auto* b = allocate<u32>(labels_binding, ctx, u64{T} * L, s);
        auto* sc = allocate<u32>(labels_schema, ctx, T, s);
        auto* pa = allocate<u32>(labels_parent, ctx, T, s);
        auto* ctl = allocate<u32>(control, ctx, 3, s);
        lifted::Labels labels{b, sc, pa, T, L};
        labels.error = ctl + 1;
        u64* cand = nullptr;
        for (;;)
        {
            cand = allocate<u64>(candidates, ctx, u64{T} * W, s);
            check(cudaMemsetAsync(ctl, 0, 3 * sizeof(u32), s), "cudaMemsetAsync (numeric search control)");
            gen.write(labels, cand, W, ctl);
            const bool missing = gen.resolve_missing();
            u32 need = 0;
            download(&need, ctl, 1, s);
            sync();
            need = std::max(need, gen.host_words_needed(W));
            if (need > W - slots || missing)
            {
                ensure(nodes.size() + T, std::max(W, std::max(need + slots, task->words() + slots)));
                input.states = static_cast<const u64*>(arena.data()) + u64{parent} * W;
                input.words = W; input.stride = W; input.host_stride = W;
                if (gen.needs_host(o.search.witness_pruning)) { (void)load(parent); input.host_states = host_row.data(); }
                gen.begin(input, o.search.witness_pruning, o.search.canonical_order);
                gen.views(); gen.count();
                continue;
            }
            break;
        }
        std::vector<u32> schemas(T), bindings(u64{T} * L);
        std::vector<f64> gs(T);
        download(schemas.data(), sc, T, s);
        download(bindings.data(), b, u64{T} * L, s);
        auto* dgs = allocate<f64>(values, ctx, T, s);
        const f64 gp = nodes[parent].g;
        if (!gen.placement(o.search.witness_pruning).host_count)
        {
            auto* dp = allocate<f64>(parent_value, ctx, 1, s);
            upload(dp, &gp, 1, s);
            check(numeric::launch_costs(gen.view(), {static_cast<u32>(costs.kind()),
                task->compiled().num.metric.begin, task->compiled().num.metric.end}, gen.parents(), gen.parent_views(), labels, dp, T, dgs, ctl + 2, s),
                  "numeric::launch_costs");
            download(gs.data(), dgs, T, s);
            u32 error = 0;
            download(&error, ctl + 2, 1, s);
            sync();
            if (error) throw std::domain_error("mymyr: numeric search cost program rejected a labelled successor");
        }
        else
        {
            const State state = load(parent);
            const WorkspaceLease lease = task->workspace();
            Successors& succ = lease->successors();
            succ.prepare(state.view());
            u32 at = 0;
            succ.generate<false>([&](u32, const ObjectId*, const Delta& delta) { gs[at++] = costs.next(gp, delta); return true; },
                          o.search.witness_pruning, o.search.canonical_order);
            if (at != T) throw std::logic_error("mymyr: numeric search fallback transition count differs");
        }
        std::vector<u8> goal(T);
        auto* dg = allocate<u8>(flags, ctx, T, s);
        if (task->has_axioms() && !gen.device_axioms())
        {
            std::vector<u64> rows(u64{T} * W);
            download(rows.data(), cand, rows.size(), s); sync();
            for (u32 i = 0; i < T; ++i)
                goal[i] = task->is_goal(numeric::decode(*task, rows.data() + u64{i} * W, W).view());
        }
        else
        {
            gen.goal_flags(cand, W, W, T, nullptr, T, dg);
            download(goal.data(), dg, T, s); sync();
        }
        evaluate(cand, T, hs);
        ensure(nodes.size() + T, W);
        state_set::Table tab{static_cast<u64*>(table.data()), table_capacity - 1};
        state_set::Rows stored{static_cast<const u64*>(arena.data()), W}, rows{cand, W};
        auto* dr = allocate<u32>(result, ctx, T, s);
        auto* dk = allocate<u32>(rank, ctx, u64{T} + 1, s);
        check(state_set::launch_insert(tab, stored, rows, T, dr, s), "state_set::launch_insert");
        const u64 scan_bytes = state_set::rank_temp_bytes(T);
        check(state_set::launch_rank(tab, dr, T, dk, scan.ensure(ctx, scan_bytes, s), scan_bytes, s), "state_set::launch_rank");
        std::vector<u32> ranks(u64{T} + 1), child(T);
        download(ranks.data(), dk, u64{T} + 1, s); sync();
        const u32 base = static_cast<u32>(nodes.size());
        state_set::Compact compact;
        compact.result = dr; compact.rank = dk; compact.cand = cand;
        compact.arena_tail = static_cast<u64*>(arena.data()) + u64{base} * W;
        compact.base = base; compact.words = W;
        check(state_set::launch_compact(tab, compact, T, s), "state_set::launch_compact");
        auto* di = allocate<u32>(ids, ctx, T, s);
        check(state_set::launch_lookup(tab, stored, rows, T, di, s), "state_set::launch_lookup");
        download(child.data(), di, T, s); sync();
        nodes.resize(u64{base} + ranks[T]);
        bool full = false;
        u32 generated = T;
        for (u32 i = 0; i < T; ++i)
            if (ranks[i + 1] > ranks[i] && u64{base} + ranks[i + 1] > budget.max_states)
            {
                nodes.resize(u64{base} + ranks[i + 1]);
                generated = i + 1; full = true; break;
            }
        r.stats.generated += generated;
        if (full) { r.status = search::SearchStatus::OutOfStates; break; }
        for (u32 i = 0; i < T; ++i)
        {
            const u32 id = child[i];
            if (id == none) throw std::logic_error("mymyr: numeric search cannot resolve a successor");
            const bool fresh = ranks[i + 1] > ranks[i];
            Node& n = nodes[id];
            if (!fresh && (greedy || !(gs[i] < n.g && (n.open || (n.closed && o.search.reopen))))) continue;
            r.reopened += !fresh && n.closed;
            n.g = gs[i]; n.parent = parent; n.depth = nodes[parent].depth + 1;
            n.action.schema = SchemaId{schemas[i]};
            n.action.binding.clear();
            for (u32 j = 0; j < task->compiled().schemas[schemas[i]].arity; ++j)
                n.action.binding.push_back(ObjectId{bindings[u64{i} * L + j]});
            if (fresh)
            {
                n.goal = goal[i];
                if (greedy && n.goal) { solved(id); break; }
                ++r.evaluations;
                n.h = hs[i];
                if (std::isinf(n.h)) { ++r.dead_ends; ++r.stats.pruned; continue; }
            }
            n.open = true; n.closed = false;
            push(id);
        }
        if (r.status == search::SearchStatus::Solved) break;
    }
    return finish();
}
}  // namespace mymyr::cuda::detail
