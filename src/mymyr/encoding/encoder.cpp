#include "mymyr/encoding/encoder.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/task/workspace.hpp"

#include <cassert>
#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
constexpr usize k_cache_line = 64;
// Graphs per tile of the relation-major scan: it runs over the tile's graphs one relation at a time, so that the
// strided writes (G entries apart) fill whole cache lines.
constexpr u64 k_scan_tile = 128;
constexpr u64 k_none = std::numeric_limits<u64>::max();

[[nodiscard]] u64 index_limit(IndexType index)
{
    return static_cast<u64>(index == IndexType::I64 ? std::numeric_limits<i64>::max()
                                                    : std::numeric_limits<i32>::max());
}

[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }
}  // namespace

namespace detail
{
void check_index_range(u64 value, IndexType index, const char* what)
{
    const bool wide = index == IndexType::I64;
    if (value <= index_limit(index))
        return;
    throw std::overflow_error("mymyr: encoding: the batch needs " + std::to_string(value) + " " + what + ", more than " +
                              (wide ? "int64" : "int32") + " indices hold" + (wide ? "" : "; use index_dtype='int64'"));
}
}  // namespace detail

namespace
{
[[noreturn]] void unsupported(const char* option)
{
    throw std::invalid_argument(std::string("mymyr: encoding: ") + option + " is not supported by this encoder yet");
}

void refuse_unsupported(const FlatOptions& o)
{
    if (o.horizon.mode != HorizonMode::None)
        unsupported("horizon");
}

rl::ArrayShape shape1(u64 n) { return {static_cast<i64>(n)}; }
rl::ArrayShape shape2(u64 n, u64 m) { return {static_cast<i64>(n), static_cast<i64>(m)}; }

/// The state of one pool member: it counts a contiguous slice of the graphs and later fills the same slice, so that the
/// per-graph data of the slice stays with the member. Aligned so that members never share a cache line.
struct alignas(k_cache_line) Member
{
    // pass one
    std::vector<detail::GraphTotals> totals;  // the slice's graphs
    std::vector<u32> cell_counts;             // the slice's cells, graph after graph, in cell order (relation-major or
                                              // with predicate virtual nodes)
    std::vector<u32> action_starts;           // [slice graphs, starts_stride] the run starts of each graph's rows
    std::vector<u32> action_cursors;          // [num_schemas] the cursors of the merge of one action relation's runs
    std::vector<u64> tally;                   // [R] tuples per relation in the slice (relation-major)
    u64 entities = 0, tuples = 0, args = 0, targets = 0, history = 0;  // the slice's sizes
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none;  // the first row of each kind of invalid input
    u64 narrow_goal = k_none, bad_goal_bits = k_none;
    u64 narrow_subgoals = k_none, bad_subgoal_bits = k_none;
    u64 narrow_history = k_none, bad_history_bits = k_none, bad_history_count = k_none;
    u64 bad_slice = k_none, bad_action_order = k_none, bad_action_label = k_none;
    std::vector<u32> virtual_scratch;  // [predicates] the virtual ranks of one graph
    // between the passes
    u64 entity_start = 0, tuple_start = 0, arg_start = 0, target_start = 0, history_start = 0;  // where the slice
                                                                                                // starts in the batch
    std::vector<u64> start_tuples, start_args;  // [R] where the slice starts in each relation (relation-major)
    // pass two
    std::vector<u64> run_tuples, run_args;  // [R] the running positions
    std::vector<u32> tile_counts;           // [R, k_scan_tile] the counts of the tile being scanned
};
}  // namespace

struct EncodeScratch::Impl
{
    // The input count() saw, so that fill() can refuse another one.
    bool counted = false;
    EncodeInput counted_input;
    FlatSizes sizes;

    std::vector<u32> arity;          // [R] the arguments per tuple of each relation
    std::unique_ptr<u64[]> derived;  // [G, derived_stride] when the rows' derived words are evaluated here
    u64 derived_capacity = 0;
    u64 derived_stride = 0;
    std::vector<Member> members;
};

EncodeScratch::EncodeScratch() : m_impl(std::make_unique<Impl>()) {}
EncodeScratch::~EncodeScratch() = default;

namespace
{
using Scratch = EncodeScratch::Impl;

/// Whether two inputs read the same buffers with the same shape: every pointer, stride and width equal.
bool same_input(const EncodeInput& a, const EncodeInput& b)
{
    const auto same_layers = [](const LayerInput& x, const LayerInput& y)
    {
        return x.pos == y.pos && x.neg == y.neg && x.stride == y.stride && x.layers == y.layers && x.words == y.words &&
               x.derived_pos == y.derived_pos && x.derived_neg == y.derived_neg &&
               x.derived_stride == y.derived_stride && x.derived_words == y.derived_words;
    };
    return same_layers(a.subgoals, b.subgoals) && same_layers(a.history.masks, b.history.masks) &&
           a.history.dt == b.history.dt && a.history.count == b.history.count && a.states.data == b.states.data && a.states.rows == b.states.rows && a.states.words == b.states.words &&
           a.states.stride == b.states.stride && a.states.numeric_words == b.states.numeric_words &&
           a.task_ids == b.task_ids && a.derived == b.derived && a.derived_stride == b.derived_stride &&
           a.goal_pos == b.goal_pos && a.goal_neg == b.goal_neg && a.goal_stride == b.goal_stride &&
           a.goal_words == b.goal_words && a.goal_derived_pos == b.goal_derived_pos &&
           a.goal_derived_neg == b.goal_derived_neg && a.goal_derived_stride == b.goal_derived_stride &&
           a.goal_derived_words == b.goal_derived_words && a.actions == b.actions;
}

/// What the passes of one call share.
struct Pass
{
    const EncodeInput& in;
    Scratch& s;
    const EncodingContext& context;
    detail::ContextView view;
    u64 graphs;
    u32 members;
    u32 R;
    bool relation_major;
    bool evaluate;  // the derived words of the rows are evaluated here, into s.derived
    u64 starts_stride;  // the words of run starts of one graph: num_schemas + 1 with expansion rows, else 0
    // some row may have subgoal layers or history steps
    bool layered = in.subgoals.layers != 0 || in.history.masks.layers != 0 || in.history.count != nullptr;

    [[nodiscard]] u32 instance_of(u64 g) const noexcept { return in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0; }

    [[nodiscard]] const u64* derived_of(u64 g, u32 instance) const noexcept
    {
        if (!context.needs_derived(instance, in.goal_pos != nullptr, in.goal_derived_pos != nullptr,
                                   in.subgoals.layers != 0 && in.subgoals.derived_pos != nullptr))
            return nullptr;
        return in.derived ? in.derived + g * s.derived_stride : s.derived.get() + g * s.derived_stride;
    }

    /// What the lanes read for graph g of `instance`: the state row, its derived words (null if not needed) and the
    /// goal masks of the row (null: the instance's goal).
    /// Whether graph g has expansion rows.
    [[nodiscard]] bool has_actions(u64 g) const noexcept
    {
        return in.actions && in.actions->offsets[g + 1] > in.actions->offsets[g];
    }

    /// The valid history steps of graph g, and how many of the newest of them are encoded.
    [[nodiscard]] u32 history_valid(u64 g) const noexcept
    {
        return in.history.count ? static_cast<u32>(in.history.count[g]) : in.history.masks.layers;
    }
    [[nodiscard]] u32 history_used(u64 g) const noexcept
    {
        if (in.history.masks.layers == 0)
            return 0;
        const u32 valid = history_valid(g);
        const u32 k = context.schema().options().history_max_steps;
        return k != 0 && valid > k ? k : valid;
    }

    /// The cells of graph g of `v`'s instance.
    template<bool Layered>
    [[nodiscard]] detail::CellList cells_of(const detail::InstanceView& v, u64 g) const noexcept
    {
        if (!Layered)
            return detail::cell_list(v, in.goal_pos != nullptr, has_actions(g));
        return detail::cell_list(v, in.goal_pos != nullptr, has_actions(g), in.subgoals.layers != 0,
                                 history_used(g) != 0);
    }

    /// Gives `row` the subgoal layers and history steps of graph g, stored in `layers`, when it has any. Out of line:
    /// rows without layers are the hot ones.
    [[gnu::noinline]] void attach_layers(u64 g, detail::RowLayers& layers, detail::RowInput& row) const noexcept
    {
        const u32 used = history_used(g);
        if (in.subgoals.layers == 0 && used == 0)
            return;
        layers = {};
        if (const LayerInput& l = in.subgoals; l.layers != 0)
        {
            const u64 at = g * l.stride;
            const u64 derived_at = g * l.derived_stride;
            layers.subgoals = {l.pos + at,
                               l.neg + at,
                               l.derived_pos ? l.derived_pos + derived_at : nullptr,
                               l.derived_neg ? l.derived_neg + derived_at : nullptr,
                               l.layers,
                               l.words,
                               l.derived_words};
        }
        if (used != 0)
        {
            const LayerInput& l = in.history.masks;
            const u32 first = history_valid(g) - used;
            const u64 at = g * l.stride + u64{first} * l.words;
            const u64 derived_at = g * l.derived_stride + u64{first} * l.derived_words;
            layers.history.masks = {l.pos ? l.pos + at : nullptr,
                                    l.neg ? l.neg + at : nullptr,
                                    l.derived_pos ? l.derived_pos + derived_at : nullptr,
                                    l.derived_neg ? l.derived_neg + derived_at : nullptr,
                                    used,
                                    l.words,
                                    l.derived_words};
            layers.history.dt = in.history.dt + g * l.layers + first;
            layers.history.first = first;
        }
        row.layers = &layers;
    }

    /// The input of graph g; `layers` is the storage of its subgoal layers and history steps (the row refers to it iff it
    /// has any).
    template<bool Layered>
    [[nodiscard]] detail::RowInput row_input(u64 g, u32 instance, u32* starts, u32* cursors,
                                             detail::RowLayers& layers) const noexcept
    {
        detail::RowInput row{in.states.row(g), derived_of(g, instance)};
        if constexpr (Layered)
            attach_layers(g, layers, row);
        if (in.goal_pos)
        {
            row.goal_pos = in.goal_pos + g * in.goal_stride;
            row.goal_neg = in.goal_neg + g * in.goal_stride;
        }
        if (in.goal_derived_pos)
        {
            row.goal_derived_pos = in.goal_derived_pos + g * in.goal_derived_stride;
            row.goal_derived_neg = in.goal_derived_neg + g * in.goal_derived_stride;
        }
        if (in.actions)
        {
            const rl::Expansion& x = *in.actions;
            const u64 first = static_cast<u64>(x.offsets[g]);
            row.actions = {x.schema + first,
                           x.binding + first * x.label_width,
                           x.parent ? x.parent + first : nullptr,
                           x.label_width,
                           static_cast<u32>(x.offsets[g + 1] - x.offsets[g]),
                           g,
                           starts,
                           cursors};
        }
        return row;
    }

    /// Whether the expansion rows of graph g are a slice of the expansion: ascending offsets inside the rows written.
    [[nodiscard]] bool action_slice_valid(u64 g) const noexcept
    {
        const rl::Expansion& x = *in.actions;
        return x.offsets[g] >= 0 && x.offsets[g] <= x.offsets[g + 1] && static_cast<u64>(x.offsets[g + 1]) <= x.capacity;
    }
};

/// Whether the subgoal layers or history steps of graph g cannot be encoded for its instance (`v`): a history count
/// outside [0, H], or masks narrower than the instance. The first graph of each kind is recorded in the member.
[[gnu::noinline]] bool refuse_layers(const Pass& p, Member& m, const detail::InstanceView& v, u64 g)
{
    if (p.in.history.count && (p.in.history.count[g] < 0 ||
                               static_cast<u32>(p.in.history.count[g]) > p.in.history.masks.layers))
    {
        m.bad_history_count = std::min(m.bad_history_count, g);
        return true;
    }
    if (p.in.subgoals.layers != 0 && (p.in.subgoals.words < v.words ||
                                      (p.in.subgoals.derived_pos && p.in.subgoals.derived_words < v.derived_words)))
    {
        m.narrow_subgoals = std::min(m.narrow_subgoals, g);
        return true;
    }
    if (p.history_used(g) != 0 &&
        (p.in.history.masks.words < v.words ||
         (p.in.history.masks.derived_pos && p.in.history.masks.derived_words < v.derived_words)))
    {
        m.narrow_history = std::min(m.narrow_history, g);
        return true;
    }
    return false;
}

/// Pass one of a member: validates its graphs, evaluates their derived words, counts them, and (relation-major) tallies
/// the tuples per relation. An invalid graph is recorded by kind and counts as empty.
template<bool Layered>
void count_range(const Pass& p, u32 t)
{
    const auto [lo, hi] = ThreadPool::slice(p.graphs, t, p.members);
    const rl::TaskTable& table = p.context.table();
    const u32 instances = table.size();
    Member& m = p.s.members[t];
    m.totals.assign(hi - lo, detail::GraphTotals{});
    m.entities = m.tuples = m.args = m.targets = m.history = 0;
    m.bad_id = m.narrow = m.bad_bits = m.narrow_goal = m.bad_goal_bits = k_none;
    m.narrow_subgoals = m.bad_subgoal_bits = m.narrow_history = m.bad_history_bits = m.bad_history_count = k_none;
    m.bad_slice = m.bad_action_order = m.bad_action_label = k_none;
    m.virtual_scratch.resize(p.view.num_virtual);
    const bool keep_counts = p.relation_major || p.view.num_virtual != 0;
    if (p.starts_stride)
    {
        m.action_starts.resize((hi - lo) * p.starts_stride);
        m.action_cursors.resize(p.view.num_schemas);
    }
    if (p.relation_major)
        m.tally.assign(p.R, 0);
    m.cell_counts.clear();
    for (u64 g = lo; g < hi; ++g)
    {
        const u32 instance = p.instance_of(g);
        if (p.in.task_ids && (p.in.task_ids[g] < 0 || instance >= instances))
        {
            m.bad_id = std::min(m.bad_id, g);
            continue;
        }
        if (p.in.states.words < table.instance(instance).words)
        {
            m.narrow = std::min(m.narrow, g);
            continue;
        }
        if (p.in.actions && !p.action_slice_valid(g))
        {
            m.bad_slice = std::min(m.bad_slice, g);
            continue;
        }
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        if ((p.in.goal_pos && p.in.goal_words < v.words) ||
            (p.in.goal_derived_pos && p.in.goal_derived_words < v.derived_words))
        {
            m.narrow_goal = std::min(m.narrow_goal, g);
            continue;
        }
        if (Layered && refuse_layers(p, m, v, g))
            continue;
        detail::RowLayers layers;
        const detail::RowInput input = p.row_input<Layered>(
            g, instance, p.starts_stride ? m.action_starts.data() + (g - lo) * p.starts_stride : nullptr,
            m.action_cursors.data(), layers);
        const u64* row = input.atoms;
        const u64* derived = input.derived;
        if (derived && p.evaluate)
        {
            u64* dst = p.s.derived.get() + g * p.s.derived_stride;
            std::fill_n(dst, p.s.derived_stride, u64{0});
            if (!detail::has_foreign_atom_bits(v, row))  // the row is refused after the pass
            {
                const Task& task = *table.instance(instance).task;
                const WorkspaceLease workspace = task.workspace();
                auto& engine = workspace->engine();
                engine.set_state(row, v.words);
                engine.build_view();
                workspace->axioms().evaluate();
                std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), dst);
            }
        }
        const detail::CellList cells = detail::cell_list<Layered>(v, input);
        u32* cell_counts = nullptr;
        if (keep_counts)
        {
            const usize at = m.cell_counts.size();
            m.cell_counts.resize(at + cells.count);
            cell_counts = m.cell_counts.data() + at;
        }
        detail::GraphTotals& totals = m.totals[g - lo];
        totals = detail::count_graph<Layered>(v, input, cell_counts, m.virtual_scratch.data());
        if (totals.flags & (detail::k_graph_bad_row | detail::k_graph_bad_goal | detail::k_graph_bad_action_order |
                            detail::k_graph_bad_action_label | detail::k_graph_bad_layers | detail::k_graph_bad_history))
        {
            if (totals.flags & detail::k_graph_bad_row)
                m.bad_bits = std::min(m.bad_bits, g);
            else if (totals.flags & detail::k_graph_bad_goal)
                m.bad_goal_bits = std::min(m.bad_goal_bits, g);
            else if (totals.flags & detail::k_graph_bad_layers)
                m.bad_subgoal_bits = std::min(m.bad_subgoal_bits, g);
            else if (totals.flags & detail::k_graph_bad_history)
                m.bad_history_bits = std::min(m.bad_history_bits, g);
            else if (totals.flags & detail::k_graph_bad_action_label)
                m.bad_action_label = std::min(m.bad_action_label, g);
            else
                m.bad_action_order = std::min(m.bad_action_order, g);
            continue;
        }
        m.entities += totals.entities;
        m.tuples += totals.tuples;
        m.args += totals.args;
        m.targets += totals.targets;
        m.history += totals.history_steps;
        if (p.relation_major)
            for (u32 i = 0; i < cells.count; ++i)
                m.tally[cells.cells[i].relation] += cell_counts[i];
    }
}

/// Pass two of a member: the positions (relation-major) and the contents of its graphs.
template<class I, bool Layered>
void fill_range(const Pass& p, const FlatBuffers& b, u32 t)
{
    const auto [lo, hi] = ThreadPool::slice(p.graphs, t, p.members);
    const detail::GraphOut<I> out{p.graphs,
                                  static_cast<I*>(b.num_objects),
                                  static_cast<I*>(b.relation_counts),
                                  static_cast<I*>(b.relation_offsets),
                                  static_cast<I*>(b.relation_arg_offsets),
                                  static_cast<I*>(b.relation_args),
                                  b.entity_role,
                                  static_cast<I*>(b.entity_ref),
                                  b.goal_complete,
                                  static_cast<I*>(b.target_offsets),
                                  static_cast<I*>(b.target_entity),
                                  b.target_group,
                                  static_cast<I*>(b.history_offsets),
                                  static_cast<I*>(b.history_entity),
                                  b.history_dt};
    I* entity_offsets = static_cast<I*>(b.entity_offsets);
    Member& m = p.s.members[t];
    u64 entity = m.entity_start, tuple = m.tuple_start, arg = m.arg_start, target = m.target_start,
        history = m.history_start;
    u32* virtual_scratch = m.virtual_scratch.empty() ? nullptr : m.virtual_scratch.data();
    // fills graph g; `graph_counts` are its cell counts (null unless the pass kept them)
    auto fill_one = [&](u64 g, const u32* graph_counts)
    {
        const u32 instance = p.instance_of(g);
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        const detail::GraphTotals& totals = m.totals[g - lo];
        detail::RowLayers layers;
        const detail::RowInput row = p.row_input<Layered>(
            g, instance, p.starts_stride ? m.action_starts.data() + (g - lo) * p.starts_stride : nullptr,
            m.action_cursors.data(), layers);
        entity_offsets[g] = static_cast<I>(entity);
        b.task_ids[g] = p.in.task_ids ? p.in.task_ids[g] : 0;
        if (p.relation_major)
            detail::fill_graph_relation_major<I, Layered>(v, row, g, entity, totals, out, target, virtual_scratch,
                                                          graph_counts, history);
        else
            detail::fill_graph<I, Layered>(v, row, g, tuple, arg, entity, totals, out, target, virtual_scratch,
                                           graph_counts, history);
        entity += totals.entities;
        tuple += totals.tuples;
        arg += totals.args;
        target += totals.targets;
        if constexpr (Layered)
            history += totals.history_steps;
    };
    if (!p.relation_major)
    {
        const u32* graph_counts = m.cell_counts.empty() ? nullptr : m.cell_counts.data();
        for (u64 g = lo; g < hi; ++g)
        {
            fill_one(g, graph_counts);
            if (graph_counts)
                graph_counts += p.cells_of<Layered>(detail::instance_view(p.view, p.instance_of(g)), g).count;
        }
        return;
    }
    m.run_tuples = m.start_tuples;
    m.run_args = m.start_args;
    m.tile_counts.resize(p.R * k_scan_tile);
    const u32* cell_counts = m.cell_counts.data();
    for (u64 first = lo; first < hi; first += k_scan_tile)
    {
        const u64 width = std::min(k_scan_tile, hi - first);
        // the tile's counts, relation by relation (a relation without a cell in a graph has count 0)
        std::fill(m.tile_counts.begin(), m.tile_counts.end(), 0u);
        const u32* next = cell_counts;
        for (u64 j = 0; j < width; ++j)
        {
            const u32 instance = p.instance_of(first + j);
            const detail::CellList cells = p.cells_of<Layered>(detail::instance_view(p.view, instance), first + j);
            for (u32 i = 0; i < cells.count; ++i)
                m.tile_counts[cells.cells[i].relation * k_scan_tile + j] = next[i];
            next += cells.count;
        }
        // the exclusive scan in segment order r * G + g, written in runs of `width` per relation
        for (u32 r = 0; r < p.R; ++r)
        {
            I* offsets = out.relation_offsets + r * p.graphs + first;
            I* arg_offsets = out.relation_arg_offsets + r * p.graphs + first;
            const u32* counts = m.tile_counts.data() + r * k_scan_tile;
            const u64 arity = p.s.arity[r];
            u64 tuples = m.run_tuples[r];
            u64 args = m.run_args[r];
            if (m.tally[r] == 0)  // no graph of the slice has a tuple of the relation: the offsets do not advance
            {
                std::fill_n(offsets, width, static_cast<I>(tuples));
                std::fill_n(arg_offsets, width, static_cast<I>(args));
                continue;
            }
            for (u64 j = 0; j < width; ++j)
            {
                offsets[j] = static_cast<I>(tuples);
                arg_offsets[j] = static_cast<I>(args);
                tuples += counts[j];
                args += counts[j] * arity;
            }
            m.run_tuples[r] = tuples;
            m.run_args[r] = args;
        }
        const u32* graph_counts = cell_counts;
        for (u64 j = 0; j < width; ++j)
        {
            fill_one(first + j, graph_counts);
            graph_counts +=
                p.cells_of<Layered>(detail::instance_view(p.view, p.instance_of(first + j)), first + j).count;
        }
        cell_counts = next;
    }
}

/// The members that encode a batch of `graphs` graphs: all of the pool for a large batch, fewer for a small one, one
/// (the calling thread) for a batch under the grain.
[[nodiscard]] u32 members_for(u64 graphs, const ThreadPool& pool) noexcept
{
    const u64 wanted = std::max<u64>(1, (graphs + k_graphs_per_member - 1) / k_graphs_per_member);
    return static_cast<u32>(std::min<u64>(pool.size(), wanted));
}

/// Runs `f(t)` for the members [0, members) without allocating a closure: inline for one member, on the pool otherwise
/// (a pool member beyond `members` has nothing to do).
template<class F>
void run(ThreadPool& pool, u32 members, const F& f)
{
    if (members == 1)
    {
        f(0);
        return;
    }
    const std::function<void(u32)> job = [&f, members](u32 t)
    {
        if (t < members)
            f(t);
    };
    pool.run(job);
}
}  // namespace

std::shared_ptr<const FlatEncoder> FlatEncoder::create(rl::TaskTablePtr table, const FlatOptions& options)
{
    if (!table)
        invalid("an encoder needs a task table");
    auto encoder = std::shared_ptr<FlatEncoder>(new FlatEncoder());
    encoder->m_schema = EncodingSchema::create(*table, options);
    refuse_unsupported(options);
    encoder->m_context = EncodingContext::create(std::move(table), encoder->m_schema);
    return encoder;
}

std::shared_ptr<const FlatEncoder> FlatEncoder::create(TaskPtr task, const FlatOptions& options)
{
    return create(rl::TaskTable::single(std::move(task)), options);
}

void FlatEncoder::check_layer_inputs(const EncodeInput& in) const
{
    const auto check_layers = [](const LayerInput& l, const char* what, bool rows_share)
    {
        if (l.layers == 0)
            return;
        if (!l.pos || !l.neg)
            invalid(std::string(what) + " need both their positive and negative masks");
        if ((l.derived_pos == nullptr) != (l.derived_neg == nullptr))
            invalid(std::string(what) + " need both their positive and negative derived masks");
        if (l.stride < u64{l.layers} * l.words && !(rows_share && l.stride == 0))
            invalid(std::string("the row stride of the ") + what + " is smaller than their layers");
        if (l.derived_pos && l.derived_stride < u64{l.layers} * l.derived_words &&
            !(rows_share && l.derived_stride == 0))
            invalid(std::string("the derived row stride of the ") + what + " is smaller than their layers");
    };
    if (in.subgoals.layers != 0)
    {
        if (!m_schema->has_subgoals())
            invalid("subgoal layers are given, but the schema has no subgoal relations");
        if (in.subgoals.layers > m_schema->options().max_goal_level)
            invalid(std::to_string(in.subgoals.layers) + " subgoal layers are given, the schema has " +
                    std::to_string(m_schema->options().max_goal_level) + " levels");
        check_layers(in.subgoals, "subgoal layers", true);
    }
    if (in.history.masks.layers != 0)
    {
        if (!m_schema->has_history())
            invalid("history steps are given, but the schema has no history relations");
        if (!in.history.dt)
            invalid("the history steps need their dt");
        check_layers(in.history.masks, "history steps", false);
    }
}

FlatSizes FlatEncoder::count(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const
{
    const EncodingContext& context = *m_context;
    Scratch& s = scratch.impl();
    s.counted = false;
    const IndexType index = m_schema->options().index;
    const bool relation_major = m_schema->options().layout == Layout::Relation;
    const u64 graphs = in.states.rows;
    const u32 R = static_cast<u32>(m_schema->num_relations());
    const rl::StateBatchView& states = in.states;

    // 1. check what does not depend on a row
    if (states.numeric_words != 0)
        invalid("numeric state rows are not supported");
    if (graphs && !states.data)
        invalid("the state buffer is null");
    if (states.stride && states.stride < states.words)
        invalid("the state row stride is smaller than its width");
    if (in.derived && in.derived_stride < context.max_derived_words())
        invalid("the derived row stride is smaller than the derived words of an instance");
    if (in.goal_pos && !m_schema->has_goal_complete())
        invalid("goals are given, but the schema has no goal relations");
    if ((in.goal_pos == nullptr) != (in.goal_neg == nullptr))
        invalid("the goal needs both goal_pos and goal_neg");
    if ((in.goal_derived_pos == nullptr) != (in.goal_derived_neg == nullptr))
        invalid("the derived goal needs both goal_derived_pos and goal_derived_neg");
    if (in.goal_derived_pos && !in.goal_pos)
        invalid("the derived goal masks need the goal masks over the fluent slots");
    check_layer_inputs(in);
    if (const rl::Expansion* x = in.actions)
    {
        if (!m_schema->options().include_actions)
            invalid("actions are given, but the schema has no action relations");
        if (!x->offsets || !x->schema || !x->binding)
            invalid("the action expansion needs its offsets, schema and binding arrays");
        if (x->overflow())
            invalid("the action expansion overflowed: it has " + std::to_string(x->total) + " rows for a capacity of " +
                    std::to_string(x->capacity) + " and rows of " + std::to_string(x->words) + " words (" +
                    std::to_string(x->words_needed) + " needed)");
        if (x->label_width < context.table().label_width())
            invalid("the action expansion's label width " + std::to_string(x->label_width) +
                    " is below the table's largest action arity " + std::to_string(context.table().label_width()));
    }
    if (R && graphs > std::numeric_limits<u64>::max() / R - 1)
        throw std::overflow_error("mymyr: encoding: the batch has too many cells");
    const bool evaluate = !in.derived && context.max_derived_words() > 0;
    s.derived_stride = in.derived ? in.derived_stride : context.max_derived_words();
    if (evaluate && graphs * s.derived_stride > s.derived_capacity)
    {
        s.derived_capacity = graphs * s.derived_stride;
        s.derived = std::make_unique_for_overwrite<u64[]>(s.derived_capacity);
    }
    const u32 members = members_for(graphs, pool);
    s.members.resize(members);

    // 2. validate, evaluate and count, in parallel over the graphs
    const detail::ContextView view = context.view();
    const Pass pass{in, s, context, view, graphs, members, R, relation_major, evaluate,
                    in.actions ? u64{view.num_schemas} + 1 : 0};
    run(pool, members,
        [&pass](u32 t)
        {
            if (pass.layered)
                count_range<true>(pass, t);
            else
                count_range<false>(pass, t);
        });
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none, narrow_goal = k_none, bad_goal_bits = k_none;
    u64 bad_slice = k_none, bad_action_order = k_none, bad_action_label = k_none;
    u64 narrow_subgoals = k_none, bad_subgoal_bits = k_none, narrow_history = k_none, bad_history_bits = k_none,
        bad_history_count = k_none;
    for (const Member& m : s.members)
    {
        narrow_subgoals = std::min(narrow_subgoals, m.narrow_subgoals);
        bad_subgoal_bits = std::min(bad_subgoal_bits, m.bad_subgoal_bits);
        narrow_history = std::min(narrow_history, m.narrow_history);
        bad_history_bits = std::min(bad_history_bits, m.bad_history_bits);
        bad_history_count = std::min(bad_history_count, m.bad_history_count);
        bad_slice = std::min(bad_slice, m.bad_slice);
        bad_action_order = std::min(bad_action_order, m.bad_action_order);
        bad_action_label = std::min(bad_action_label, m.bad_action_label);
        bad_id = std::min(bad_id, m.bad_id);
        narrow = std::min(narrow, m.narrow);
        bad_bits = std::min(bad_bits, m.bad_bits);
        narrow_goal = std::min(narrow_goal, m.narrow_goal);
        bad_goal_bits = std::min(bad_goal_bits, m.bad_goal_bits);
    }
    const rl::TaskTable& table = this->table();
    if (bad_id != k_none)
        invalid("task id " + std::to_string(in.task_ids[bad_id]) + " of row " + std::to_string(bad_id) +
                " is outside the table's " + std::to_string(table.size()) + " instances");
    if (narrow != k_none)
        invalid("row " + std::to_string(narrow) + " has " + std::to_string(states.words) +
                " words, its instance needs " + std::to_string(table.instance(pass.instance_of(narrow)).words));
    if (narrow_goal != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_goal));
        invalid("the goal masks of row " + std::to_string(narrow_goal) + " are narrower than its instance needs (" +
                std::to_string(v.words) + " words, " + std::to_string(v.derived_words) + " derived words)");
    }
    if (narrow_subgoals != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_subgoals));
        invalid("the subgoal layers of row " + std::to_string(narrow_subgoals) +
                " are narrower than its instance needs (" + std::to_string(v.words) + " words, " +
                std::to_string(v.derived_words) + " derived words)");
    }
    if (narrow_history != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_history));
        invalid("the history steps of row " + std::to_string(narrow_history) +
                " are narrower than its instance needs (" + std::to_string(v.words) + " words, " +
                std::to_string(v.derived_words) + " derived words)");
    }
    if (bad_history_count != k_none)
        invalid("the history count of row " + std::to_string(bad_history_count) + " is " +
                std::to_string(in.history.count[bad_history_count]) + ", outside [0, " +
                std::to_string(in.history.masks.layers) + "]");
    if (bad_bits != k_none)
        invalid("row " + std::to_string(bad_bits) + " sets an atom bit its instance does not have");
    if (bad_subgoal_bits != k_none)
        invalid("the subgoal layers of row " + std::to_string(bad_subgoal_bits) +
                " set an atom bit its instance does not have");
    if (bad_history_bits != k_none)
        invalid("the history steps of row " + std::to_string(bad_history_bits) +
                " set an atom bit its instance does not have");
    if (bad_goal_bits != k_none)
        invalid("the goal of row " + std::to_string(bad_goal_bits) + " sets an atom bit its instance does not have");
    if (bad_slice != k_none)
        invalid("the offsets of the action expansion do not give state " + std::to_string(bad_slice) +
                " a slice of its rows");
    if (bad_action_label != k_none)
        invalid("the action rows of state " + std::to_string(bad_action_label) +
                " have a schema or an object its instance does not have, or belong to another state");
    if (bad_action_order != k_none)
        invalid("the action rows of state " + std::to_string(bad_action_order) +
                " are not in canonical order: strictly ascending in (schema, binding)");

    // 3. scan the members, in u64: where each starts in the batch and, relation-major, in every relation
    u64 entities = 0, tuples = 0, args = 0, targets = 0, history = 0;
    for (Member& m : s.members)
    {
        m.entity_start = entities;
        m.tuple_start = tuples;
        m.arg_start = args;
        m.target_start = targets;
        m.history_start = history;
        entities += m.entities;
        tuples += m.tuples;
        args += m.args;
        targets += m.targets;
        history += m.history;
    }
    if (relation_major)
    {
        for (Member& m : s.members)
        {
            m.start_tuples.resize(R);
            m.start_args.resize(R);
        }
        s.arity.resize(R);
        u64 tuple_position = 0, arg_position = 0;
        for (u32 r = 0; r < R; ++r)
        {
            s.arity[r] = m_schema->relation(r).arity;
            for (Member& m : s.members)
            {
                m.start_tuples[r] = tuple_position;
                m.start_args[r] = arg_position;
                tuple_position += m.tally[r];
                arg_position += m.tally[r] * s.arity[r];
            }
        }
    }

    FlatSizes sizes;
    sizes.entities = entities;
    sizes.tuples = tuples;
    sizes.args = args;
    sizes.targets = targets;
    sizes.history = history;
    const u64 limit = index_limit(index);
    if (entities > limit || tuples > limit || args > limit || targets > limit || history > limit ||
        graphs * R + 1 > limit || graphs + 1 > limit)
        sizes.overflow |= k_overflow_index;

    s.sizes = sizes;
    s.counted = true;
    s.counted_input = in;
    return sizes;
}

void FlatEncoder::fill(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
                       EncodeScratch& scratch) const
{
    Scratch& s = scratch.impl();
    if (!s.counted || !same_input(s.counted_input, in) || s.sizes.entities != sizes.entities ||
        s.sizes.tuples != sizes.tuples || s.sizes.args != sizes.args || s.sizes.targets != sizes.targets ||
        s.sizes.history != sizes.history)
        invalid("fill needs the scratch and the sizes of count() on the same input");
    if (sizes.overflow != 0)
        invalid("the batch does not fit: its sizes overflow");
    const u32 members = static_cast<u32>(s.members.size());
    if (pool.size() < members)
        invalid("fill needs the thread pool of count()");
    const EncodingSchema& schema = *m_schema;
    const FlatOptions& options = schema.options();
    const u64 graphs = in.states.rows;
    if (!out.task_ids || !out.num_objects || !out.entity_offsets || !out.relation_offsets || !out.relation_arg_offsets)
        invalid("a destination array of the batch is null");
    if (sizes.args > 0 && !out.relation_args)
        invalid("relation_args is null, the batch has tuple arguments");
    if ((out.relation_counts != nullptr) != options.emit_counts)
        invalid(options.emit_counts ? "relation_counts is null, the schema stores it"
                                    : "relation_counts must be null, the schema omits it");
    if ((out.entity_role != nullptr) != !schema.implicit_entity_arrays() ||
        (out.entity_ref != nullptr) != !schema.implicit_entity_arrays())
        invalid(schema.implicit_entity_arrays() ? "entity_role and entity_ref must be null, the schema omits them"
                                                : "entity_role and entity_ref are null, the schema stores them");
    if ((out.goal_complete != nullptr) != schema.has_goal_complete())
        invalid(schema.has_goal_complete() ? "goal_complete is null, the schema stores it"
                                           : "goal_complete must be null, the schema has no goal relations");
    if ((out.target_offsets != nullptr) != schema.has_targets() ||
        (!schema.has_targets() && (out.target_entity != nullptr || out.target_group != nullptr)) ||
        (sizes.targets > 0 && (!out.target_entity || !out.target_group)))
        invalid(schema.has_targets() ? "a target array is null, the schema stores them"
                                     : "the target arrays must be null, the schema has no targets");
    if ((out.history_offsets != nullptr) != schema.has_history() ||
        (!schema.has_history() && (out.history_entity != nullptr || out.history_dt != nullptr)) ||
        (sizes.history > 0 && (!out.history_entity || !out.history_dt)))
        invalid(schema.has_history() ? "a history array is null, the schema stores them"
                                     : "the history arrays must be null, the schema has no history");
    const bool evaluate = !in.derived && m_context->max_derived_words() > 0;
    const detail::ContextView view = m_context->view();
    const Pass pass{in,
                    s,
                    *m_context,
                    view,
                    graphs,
                    members,
                    static_cast<u32>(schema.num_relations()),
                    options.layout == Layout::Relation,
                    evaluate,
                    in.actions ? u64{view.num_schemas} + 1 : 0};
    run(pool, members,
        [&pass, &out, wide = options.index == IndexType::I64](u32 t)
        {
            if (pass.layered)
                wide ? fill_range<i64, true>(pass, out, t) : fill_range<i32, true>(pass, out, t);
            else
                wide ? fill_range<i64, false>(pass, out, t) : fill_range<i32, false>(pass, out, t);
        });
    const u64 cells = graphs * pass.R;
    auto close = [&]<class I>(I)
    {
        static_cast<I*>(out.entity_offsets)[graphs] = static_cast<I>(sizes.entities);
        if (out.target_offsets)
            static_cast<I*>(out.target_offsets)[graphs] = static_cast<I>(sizes.targets);
        if (out.history_offsets)
            static_cast<I*>(out.history_offsets)[graphs] = static_cast<I>(sizes.history);
        static_cast<I*>(out.relation_offsets)[cells] = static_cast<I>(sizes.tuples);
        static_cast<I*>(out.relation_arg_offsets)[cells] = static_cast<I>(sizes.args);
    };
    if (options.index == IndexType::I64)
        close(i64{});
    else
        close(i32{});
}

FlatSizes FlatEncoder::encode_into(const EncodeInput& in, const FlatCapacities& capacities, const FlatBuffers& out,
                                   ThreadPool& pool, EncodeScratch& scratch) const
{
    FlatSizes sizes = count(in, pool, scratch);
    if (sizes.entities > capacities.entities || sizes.tuples > capacities.tuples || sizes.args > capacities.args ||
        sizes.targets > capacities.targets || sizes.history > capacities.history)
        sizes.overflow |= k_overflow_capacity;
    if (!sizes.overflowed())
        fill(in, sizes, out, pool, scratch);
    return sizes;
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in) const
{
    ThreadPool pool(1);
    EncodeScratch scratch;
    return encode(in, pool, scratch);
}

namespace
{
/// The arrays of a bundle that allocate() built, in the order it adds them: task_ids, num_objects, entity_offsets,
/// [entity_role, entity_ref], [relation_counts], relation_offsets, relation_arg_offsets, relation_args,
/// [goal_complete], [target_offsets, target_entity, target_group], [history_offsets, history_entity, history_dt].
FlatBuffers buffers_of(rl::ArrayBundle& bundle, bool entity_arrays, bool counts, bool goal_complete, bool targets,
                       bool history)
{
    FlatBuffers out;
    const auto& arrays = bundle.arrays();
    usize next = 0;
    auto take = [&](const char* name)
    {
        assert(arrays[next].name == name);
        return bundle.mutable_data(arrays[next++]);
    };
    out.task_ids = static_cast<i32*>(take("task_ids"));
    out.num_objects = take("num_objects");
    out.entity_offsets = take("entity_offsets");
    if (entity_arrays)
    {
        out.entity_role = static_cast<u8*>(take("entity_role"));
        out.entity_ref = take("entity_ref");
    }
    if (counts)
        out.relation_counts = take("relation_counts");
    out.relation_offsets = take("relation_offsets");
    out.relation_arg_offsets = take("relation_arg_offsets");
    out.relation_args = take("relation_args");
    if (goal_complete)
        out.goal_complete = static_cast<u8*>(take("goal_complete"));
    if (targets)
    {
        out.target_offsets = take("target_offsets");
        out.target_entity = take("target_entity");
        out.target_group = static_cast<u8*>(take("target_group"));
    }
    if (history)
    {
        out.history_offsets = take("history_offsets");
        out.history_entity = take("history_entity");
        out.history_dt = static_cast<i32*>(take("history_dt"));
    }
    return out;
}
}  // namespace

rl::ArrayBundle FlatEncoder::allocate(u64 graphs, u64 entities, u64 args, u64 targets, u64 history) const
{
    const IndexType index = m_schema->options().index;
    const u64 R = m_schema->num_relations();
    const rl::DType itype = index == IndexType::I64 ? rl::DType::I64 : rl::DType::I32;
    rl::ArrayBundle::Builder builder;
    builder.add_uninitialized("task_ids", rl::DType::I32, shape1(graphs));
    builder.add_uninitialized("num_objects", itype, shape1(graphs));
    builder.add_uninitialized("entity_offsets", itype, shape1(graphs + 1));
    if (!m_schema->implicit_entity_arrays())
    {
        builder.add_uninitialized("entity_role", rl::DType::U8, shape1(entities));
        builder.add_uninitialized("entity_ref", itype, shape1(entities));
    }
    if (m_schema->emits_counts())
        builder.add_uninitialized("relation_counts", itype, shape2(graphs, R));
    builder.add_uninitialized("relation_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_arg_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_args", itype, shape1(args));
    if (m_schema->has_goal_complete())
        builder.add_uninitialized("goal_complete", rl::DType::U8, shape1(graphs));
    if (m_schema->has_targets())
    {
        builder.add_uninitialized("target_offsets", itype, shape1(graphs + 1));
        builder.add_uninitialized("target_entity", itype, shape1(targets));
        builder.add_uninitialized("target_group", rl::DType::U8, shape1(targets));
    }
    if (m_schema->has_history())
    {
        builder.add_uninitialized("history_offsets", itype, shape1(graphs + 1));
        builder.add_uninitialized("history_entity", itype, shape1(history));
        builder.add_uninitialized("history_dt", rl::DType::I32, shape1(history));
    }
    builder.scalar("version", 1);
    builder.scalar("num_graphs", static_cast<i64>(graphs));
    builder.scalar("num_relations", static_cast<i64>(R));
    builder.scalar("layout", m_schema->options().layout == Layout::Relation ? 1 : 0);
    builder.scalar("index_dtype", index == IndexType::I64 ? 1 : 0);
    builder.scalar("schema_fingerprint", std::bit_cast<i64>(m_schema->fingerprint()));
    builder.scalar("statics", 0);
    return builder.build();
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in, ThreadPool& pool,
                                                           EncodeScratch& scratch) const
{
    const FlatSizes sizes = count(in, pool, scratch);
    if (sizes.overflow & k_overflow_index)
    {
        const IndexType index = m_schema->options().index;
        const u64 graphs = in.states.rows;
        const u64 R = m_schema->num_relations();
        detail::check_index_range(sizes.entities, index, "entities");
        detail::check_index_range(sizes.tuples, index, "tuples");
        detail::check_index_range(sizes.args, index, "tuple arguments");
        detail::check_index_range(sizes.targets, index, "targets");
        detail::check_index_range(sizes.history, index, "history steps");
        detail::check_index_range(graphs * R + 1, index, "relation cells");
        detail::check_index_range(graphs + 1, index, "graphs");
    }
    auto bundle = std::make_shared<rl::ArrayBundle>(
        allocate(in.states.rows, sizes.entities, sizes.args, sizes.targets, sizes.history));
    fill(in, sizes,
         buffers_of(*bundle, !m_schema->implicit_entity_arrays(), m_schema->emits_counts(), m_schema->has_goal_complete(),
                    m_schema->has_targets(), m_schema->has_history()),
         pool, scratch);
    return bundle;
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode_capacity(const EncodeInput& in,
                                                                    const FlatCapacities& capacities, ThreadPool& pool,
                                                                    EncodeScratch& scratch, FlatSizes& sizes) const
{
    sizes = count(in, pool, scratch);
    if (sizes.entities > capacities.entities || sizes.tuples > capacities.tuples || sizes.args > capacities.args ||
        sizes.targets > capacities.targets || sizes.history > capacities.history)
        sizes.overflow |= k_overflow_capacity;
    const u64 graphs = in.states.rows;
    const bool fits = !sizes.overflowed();
    auto bundle = std::make_shared<rl::ArrayBundle>(
        allocate(graphs, capacities.entities, capacities.args, capacities.targets, capacities.history));
    const FlatBuffers out = buffers_of(*bundle, !m_schema->implicit_entity_arrays(), m_schema->emits_counts(),
                                       m_schema->has_goal_complete(), m_schema->has_targets(), m_schema->has_history());
    const usize index_bytes = m_schema->options().index == IndexType::I64 ? 8 : 4;
    // the entries a batch does not use are zero, and an overflowing batch is all zero
    auto zero = [](void* data, u64 from, u64 to, usize width)
    { std::memset(static_cast<std::byte*>(data) + from * width, 0, (to - from) * width); };
    if (fits)
    {
        fill(in, sizes, out, pool, scratch);
        zero(out.relation_args, sizes.args, capacities.args, index_bytes);
        if (out.entity_role)
        {
            zero(out.entity_role, sizes.entities, capacities.entities, 1);
            zero(out.entity_ref, sizes.entities, capacities.entities, index_bytes);
        }
        if (out.target_entity)
        {
            zero(out.target_entity, sizes.targets, capacities.targets, index_bytes);
            zero(out.target_group, sizes.targets, capacities.targets, 1);
        }
        if (out.history_entity)
        {
            zero(out.history_entity, sizes.history, capacities.history, index_bytes);
            zero(out.history_dt, sizes.history, capacities.history, 4);
        }
    }
    else
        for (const rl::ArrayInfo& info : bundle->arrays())
            std::memset(bundle->mutable_data(info), 0, info.elements() * rl::dtype_bytes(info.dtype));
    return bundle;
}
}  // namespace mymyr::encoding
