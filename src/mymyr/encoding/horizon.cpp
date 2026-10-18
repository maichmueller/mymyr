// The horizon batches of the relational encoder: the tuples of a graph that is a root state and the nodes below it. The
// cells come from the context (detail/horizon.hpp); this file counts and fills them. The word of a cell at a node (which
// of the cell's slots are tuples) is written once, so the counts and the contents agree.

#include "mymyr/encoding/encoder.hpp"

#include "encode_pass.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/encoding/detail/horizon.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <cassert>
#include <string>

namespace mymyr::encoding
{
namespace
{
using namespace detail;
using Scratch = EncodeScratch::Impl;

[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }

/// Everything the cells of one graph read.
struct Graph
{
    const InstanceView* v;
    GoalWords goal;
    RowInput row;  // the graph's goal and subgoal layers; atoms and derived of a node are given by the accessors
    const u64* nodes;
    u64 node_stride;
    const u64* derived;  // the first node's derived words (null: not read)
    u64 derived_stride;
    u32 count;  // the nodes
    const i32* src;
    const i32* dst;
    const i32* schema;
    const i32* binding;
    u32 label_width;
    u32 edges;
    const i32* schema_relation;  // [schemas] the relation of each action schema (-1: none)
    bool exclude;                // the root has no entity
};

const u64* atoms_of(const Graph& h, u32 node) { return h.nodes + u64{node} * h.node_stride; }
const u64* derived_of(const Graph& h, u32 node)
{
    return h.derived ? h.derived + u64{node} * h.derived_stride : nullptr;
}

struct NodeRange
{
    u32 first, last;
};

NodeRange scope_of(const Cell& c, u32 count)
{
    switch (horizon_scope(c.op))
    {
    case HorizonScope::Root: return {0, 1};
    case HorizonScope::Nodes: return {1, count};
    default: return {0, count};
    }
}

/// Word `w` of the slots [lo, hi) of the word `x` (the word of index w), the others cleared.
u64 clip_word(u64 x, u32 w, u32 lo, u32 hi)
{
    if (w == (lo >> 6))
        x &= ~u64{0} << (lo & 63u);
    if (w == ((hi - 1u) >> 6))
        x &= ~u64{0} >> (63u - ((hi - 1u) & 63u));
    return x;
}

RowInput node_input(const Graph& h, u32 node)
{
    RowInput row = h.row;
    row.atoms = atoms_of(h, node);
    row.derived = derived_of(h, node);
    return row;
}

/// Word `w` of the slots of cell `c` that are tuples at `node` (the cell is slot based: not static facts, not an edge
/// cell).
u64 cell_word(const Cell& c, const Graph& h, u32 node, u32 w)
{
    const u32 low = horizon_low(c.op);
    switch (horizon_kind(c.op))
    {
    case HorizonKind::Facts:
    {
        const u64* x = low == static_cast<u32>(CellOp::Derived) ? derived_of(h, node) : atoms_of(h, node);
        return x ? range_word(x, w, c.a, c.b) : 0;
    }
    case HorizonKind::Delta:
    {
        const bool derived = low == static_cast<u32>(CellOp::Derived);
        const u64* x = derived ? derived_of(h, node) : atoms_of(h, node);
        const u64* root = derived ? derived_of(h, 0) : atoms_of(h, 0);
        if (!x || !root)
            return 0;
        return clip_word(horizon_minus(c.op) ? root[w] & ~x[w] : x[w] & ~root[w], w, c.a, c.b);
    }
    case HorizonKind::Goal:
    case HorizonKind::Sat:
    {
        Cell flat = c;
        flat.op = low;
        return goal_word<true>(flat, h.goal, node_input(h, node), w);
    }
    case HorizonKind::Added:
    {
        Cell flat = c;
        flat.op = low;
        const u64 literals = goal_word<true>(flat, h.goal, h.row, w);
        if (!literals)
            return 0;
        const bool derived = goal_op_derived(low);
        const u64* x = derived ? derived_of(h, node) : atoms_of(h, node);
        const u64* root = derived ? derived_of(h, 0) : atoms_of(h, 0);
        if (!x || !root)
            return 0;
        // a positive literal is newly satisfied by an atom the node adds and newly falsified by one it deletes; a
        // negative literal the other way round
        const bool added = horizon_minus(c.op) == goal_op_negative(low);
        return literals & (added ? x[w] & ~root[w] : root[w] & ~x[w]);
    }
    default: return 0;
    }
}

bool edge_skipped(const Graph& h, const Cell& c, u32 e)
{
    if (horizon_kind(c.op) == HorizonKind::Action)
        return h.schema_relation[h.schema[e]] != static_cast<i32>(c.relation) || (h.exclude && h.dst[e] == 0);
    return h.exclude && (h.src[e] == 0 || h.dst[e] == 0);
}

/// The tuples of cell `c` in the graph.
u32 cell_count(const Cell& c, const Graph& h)
{
    const HorizonKind kind = horizon_kind(c.op);
    if (kind == HorizonKind::Action || kind == HorizonKind::Parent)
    {
        u32 n = 0;
        for (u32 e = 0; e < h.edges; ++e)
            n += edge_skipped(h, c, e) ? 0u : 1u;
        return n;
    }
    const auto [first, last] = scope_of(c, h.count);
    if (kind == HorizonKind::Facts && horizon_low(c.op) == static_cast<u32>(CellOp::Static))
        return (last - first) * (c.b - c.a);
    const u32 constants = h.goal.constants && (kind == HorizonKind::Goal || kind == HorizonKind::Sat) ? c.d - c.c : 0u;
    u32 n = 0;
    for (u32 node = first; node < last; ++node)
    {
        n += constants;
        if (c.a < c.b)
            for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
                n += popcount64(cell_word(c, h, node, w));
    }
    return n;
}

/// Calls out.put(node, objects) for the tuples of a slot based cell at `node`, in the order of the cell: the cell's
/// slots in ascending order, with the constant goal tuples merged in.
template<class Out>
void emit_node(const Cell& c, const Graph& h, u32 node, Out& out)
{
    const InstanceView& v = *h.v;
    const HorizonKind kind = horizon_kind(c.op);
    const u32 low = horizon_low(c.op);
    const bool derived = (kind == HorizonKind::Facts || kind == HorizonKind::Delta)
                             ? low == static_cast<u32>(CellOp::Derived)
                             : goal_op_derived(low);
    const i32* rows = derived ? v.derived_args : v.slot_args;
    const u32 stride = v.max_arity;
    u32 k = c.c;
    const u32 end = h.goal.constants && (kind == HorizonKind::Goal || kind == HorizonKind::Sat) ? c.d : c.c;
    if (c.a < c.b)
        for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
        {
            u64 bits = cell_word(c, h, node, w);
            while (bits)
            {
                const u32 slot = w * 64u + lowest_bit64(bits);
                bits &= bits - 1;
                const i32* objects = rows + u64{slot} * stride;
                while (k < end && tuple_before(v.goal_args + u64{k} * stride, objects, c.logical_arity))
                    out.put(node, v.goal_args + u64{k++} * stride);
                out.put(node, objects);
            }
        }
    while (k < end)
        out.put(node, v.goal_args + u64{k++} * stride);
}

/// Calls out.put(node, objects) / out.put_pair(src, dst) for the tuples of cell `c`, in order.
template<class Out>
void emit_cell(const Cell& c, const Graph& h, Out& out)
{
    const HorizonKind kind = horizon_kind(c.op);
    if (kind == HorizonKind::Action)
    {
        for (u32 e = 0; e < h.edges; ++e)
            if (!edge_skipped(h, c, e))
                out.put(static_cast<u32>(h.dst[e]), h.binding + u64{e} * h.label_width);
        return;
    }
    if (kind == HorizonKind::Parent)
    {
        for (u32 e = 0; e < h.edges; ++e)
            if (!edge_skipped(h, c, e))
                out.put_pair(static_cast<u32>(h.src[e]), static_cast<u32>(h.dst[e]));
        return;
    }
    const auto [first, last] = scope_of(c, h.count);
    const bool statics = kind == HorizonKind::Facts && horizon_low(c.op) == static_cast<u32>(CellOp::Static);
    for (u32 node = first; node < last; ++node)
    {
        if (statics)
            for (u32 row = c.a; row < c.b; ++row)
                out.put(node, h.v->static_args + u64{row} * h.v->max_arity);
        else
            emit_node(c, h, node, out);
    }
}

/// Whether a cell's tuples of a nullary predicate are over the auxiliary entity.
bool over_auxiliary(const Cell& c)
{
    const HorizonKind kind = horizon_kind(c.op);
    return c.logical_arity == 0 && kind != HorizonKind::Action && kind != HorizonKind::Parent;
}

/// Writes the entity ids of a cell's tuples.
template<class I>
struct Writer
{
    I* dst;
    u64 object_base;
    u64 auxiliary;
    u64 state_first;  // the entity of the first node that has one
    u32 skip;         // the nodes before it
    u64 predicate;    // the predicate entity of the cell (k_no_entity: none)
    bool lead;
    bool nullary;  // a nullary predicate: its only argument is the auxiliary entity
    u32 objects;

    u64 state(u32 node) const
    {
        assert(node >= skip);
        return state_first + node - skip;
    }
    void put(u32 node, const i32* row)
    {
        if (lead)
            *dst++ = static_cast<I>(state(node));
        if (predicate != k_no_entity)
            *dst++ = static_cast<I>(predicate);
        if (nullary)
            *dst++ = static_cast<I>(auxiliary);
        else
            for (u32 k = 0; k < objects; ++k)
                *dst++ = static_cast<I>(object_base + static_cast<u64>(row[k]));
    }
    void put_pair(u32 a, u32 b)
    {
        *dst++ = static_cast<I>(state(a));
        *dst++ = static_cast<I>(state(b));
    }
};

/// What the passes of one horizon call share.
struct HPass
{
    const EncodeInput& in;
    Scratch& s;
    const EncodingContext& context;
    ContextView view;
    u64 graphs;
    u32 members;
    u32 R;
    bool relation_major;
    bool evaluate;  // the derived words of the nodes are evaluated here, into s.derived
    bool edges_used;
    bool actions;
    bool exclude;
    bool include_root_target;
    const i32* schema_relation;

    [[nodiscard]] u32 instance_of(u64 g) const noexcept { return in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0; }
    [[nodiscard]] u64 first_node(u64 g) const noexcept { return static_cast<u64>(in.horizon.node_offsets[g]); }
    [[nodiscard]] u32 nodes_of(u64 g) const noexcept
    {
        return static_cast<u32>(in.horizon.node_offsets[g + 1] - in.horizon.node_offsets[g]);
    }
    [[nodiscard]] u64 first_edge(u64 g) const noexcept { return static_cast<u64>(in.horizon.edge_offsets[g]); }
    [[nodiscard]] u32 edges_of(u64 g) const noexcept
    {
        return static_cast<u32>(in.horizon.edge_offsets[g + 1] - in.horizon.edge_offsets[g]);
    }
    [[nodiscard]] CellList cells_of(const InstanceView& v) const noexcept
    {
        return in.goal_pos ? CellList{v.row_cells, v.num_row_cells} : CellList{v.cells, v.num_cells};
    }
    /// The state entities of a graph of `nodes` nodes, and its targets.
    [[nodiscard]] u32 states_of(u32 nodes) const noexcept { return nodes - (exclude ? 1u : 0u); }
    [[nodiscard]] u32 targets_of(u32 nodes) const noexcept { return nodes - (include_root_target ? 0u : 1u); }

    /// The graph's derived words, as the lanes read them (null if the instance does not read them).
    [[nodiscard]] const u64* derived_rows(u64 g, const InstanceView& v) const noexcept
    {
        if ((v.needs_derived & k_needs_state_derived) == 0)
            return nullptr;
        return in.derived ? in.derived + first_node(g) * s.derived_stride : s.derived.get() + first_node(g) * s.derived_stride;
    }

    [[nodiscard]] Graph graph(u64 g, const InstanceView& v, RowLayers& layers) const noexcept
    {
        RowInput row{in.horizon.nodes.row(first_node(g)), nullptr};
        if (const LayerInput& l = in.subgoals; l.layers != 0)
        {
            const u64 at = g * l.stride;
            const u64 derived_at = g * l.derived_stride;
            layers = {};
            layers.subgoals = {l.pos + at,
                               l.neg + at,
                               l.derived_pos ? l.derived_pos + derived_at : nullptr,
                               l.derived_neg ? l.derived_neg + derived_at : nullptr,
                               l.layers,
                               l.words,
                               l.derived_words};
            row.layers = &layers;
        }
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
        const HorizonInput& h = in.horizon;
        Graph out{};
        out.v = &v;
        out.goal = goal_words(v, row);
        out.row = row;
        out.nodes = row.atoms;
        out.node_stride = h.nodes.stride ? h.nodes.stride : h.nodes.words;
        out.derived = derived_rows(g, v);
        out.derived_stride = s.derived_stride;
        out.count = nodes_of(g);
        if (edges_used)
        {
            const u64 e = first_edge(g);
            out.edges = edges_of(g);
            out.src = h.edge_src ? h.edge_src + e : nullptr;
            out.dst = h.edge_dst ? h.edge_dst + e : nullptr;
            out.schema = h.edge_schema ? h.edge_schema + e : nullptr;
            out.binding = h.edge_binding ? h.edge_binding + e * h.label_width : nullptr;
            out.label_width = h.label_width;
        }
        out.schema_relation = schema_relation;
        out.exclude = exclude;
        return out;
    }

    /// Why edge `e` (an index into the edge arrays) of graph g is not valid for its instance `v`; null if it is.
    [[nodiscard]] const char* edge_fault(u64 e, u32 nodes, const InstanceView& v) const noexcept
    {
        const HorizonInput& h = in.horizon;
        if (h.edge_src[e] < 0 || static_cast<u32>(h.edge_src[e]) >= nodes || h.edge_dst[e] < 0 ||
            static_cast<u32>(h.edge_dst[e]) >= nodes)
            return "an endpoint outside the graph's nodes";
        if (!actions)
            return nullptr;
        const i32 schema = h.edge_schema[e];
        if (schema < 0 || static_cast<u32>(schema) >= v.num_schemas)
            return "a schema outside the table's";
        if (schema_relation[schema] < 0)
            return "a schema without an action relation";
        const i32* binding = h.edge_binding + e * h.label_width;
        for (u32 k = 0; k < v.schema_arity[schema]; ++k)
            if (binding[k] < 0 || static_cast<u32>(binding[k]) >= v.num_objects)
                return "an object its instance does not have";
        return nullptr;
    }
};

/// The first graph slice boundaries of the members: the graphs are split by their nodes.
void slice_graphs(const i32* node_offsets, u64 graphs, u32 members, std::vector<u64>& bounds)
{
    bounds.assign(members + 1, graphs);
    bounds[0] = 0;
    const u64 total = static_cast<u64>(node_offsets[graphs]);
    for (u32 t = 1; t < members; ++t)
        bounds[t] = static_cast<u64>(std::lower_bound(node_offsets, node_offsets + graphs + 1,
                                                      static_cast<i32>(total * t / members)) -
                                     node_offsets);
}

/// Pass one of a member: validates its graphs, evaluates the derived words of their nodes, counts them.
void count_range(const HPass& p, u32 t)
{
    const u64 lo = p.s.bounds[t], hi = p.s.bounds[t + 1];
    const rl::TaskTable& table = p.context.table();
    const u32 instances = table.size();
    const HorizonInput& hz = p.in.horizon;
    Member& m = p.s.members[t];
    m.totals.assign(hi - lo, GraphTotals{});
    m.entities = m.tuples = m.args = m.targets = m.history = 0;
    m.bad_id = m.narrow = m.bad_bits = m.narrow_goal = m.bad_goal_bits = k_none;
    m.narrow_subgoals = m.bad_subgoal_bits = m.bad_edge = k_none;
    m.virtual_scratch.resize(p.view.num_virtual);
    const bool keep_counts = p.relation_major || p.view.num_virtual != 0;
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
        if (hz.nodes.words < table.instance(instance).words)
        {
            m.narrow = std::min(m.narrow, g);
            continue;
        }
        const InstanceView v = instance_view(p.view, instance);
        if ((p.in.goal_pos && p.in.goal_words < v.words) ||
            (p.in.goal_derived_pos && p.in.goal_derived_words < v.derived_words))
        {
            m.narrow_goal = std::min(m.narrow_goal, g);
            continue;
        }
        if (p.in.subgoals.layers != 0 &&
            (p.in.subgoals.words < v.words || (p.in.subgoals.derived_pos && p.in.subgoals.derived_words < v.derived_words)))
        {
            m.narrow_subgoals = std::min(m.narrow_subgoals, g);
            continue;
        }
        const u32 nodes = p.nodes_of(g);
        if (p.edges_used)
        {
            bool bad = false;
            for (u64 e = p.first_edge(g); e < p.first_edge(g) + p.edges_of(g) && !bad; ++e)
                bad = p.edge_fault(e, nodes, v) != nullptr;
            if (bad)
            {
                m.bad_edge = std::min(m.bad_edge, g);
                continue;
            }
        }
        const u64 stride = hz.nodes.stride ? hz.nodes.stride : hz.nodes.words;
        const u64* first_row = hz.nodes.row(p.first_node(g));
        bool foreign = false;
        for (u32 node = 0; node < nodes && !foreign; ++node)
            foreign = has_foreign_atom_bits(v, first_row + u64{node} * stride);
        const bool reads_derived = (v.needs_derived & k_needs_state_derived) != 0;
        if (reads_derived && !foreign && p.in.derived)
            for (u32 node = 0; node < nodes && !foreign; ++node)
                foreign = has_foreign_derived_bits(v, p.in.derived + (p.first_node(g) + node) * p.s.derived_stride);
        if (foreign)
        {
            m.bad_bits = std::min(m.bad_bits, g);
            continue;
        }
        if (reads_derived && p.evaluate)
        {
            const Task& task = *table.instance(instance).task;
            const WorkspaceLease workspace = task.workspace();
            auto& engine = workspace->engine();
            for (u32 node = 0; node < nodes; ++node)
            {
                u64* dst = p.s.derived.get() + (p.first_node(g) + node) * p.s.derived_stride;
                std::fill_n(dst, p.s.derived_stride, u64{0});
                engine.set_state(first_row + u64{node} * stride, v.words);
                engine.build_view();
                workspace->axioms().evaluate();
                std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), dst);
            }
        }
        RowLayers layers;
        const Graph h = p.graph(g, v, layers);
        const RowInput& row = h.row;
        if (has_foreign_goal_bits(v, row))
        {
            m.bad_goal_bits = std::min(m.bad_goal_bits, g);
            continue;
        }
        if (row.layers && has_foreign_layer_bits(v, row.layers->subgoals))
        {
            m.bad_subgoal_bits = std::min(m.bad_subgoal_bits, g);
            continue;
        }
        const CellList list = p.cells_of(v);
        u32* cell_counts = nullptr;
        if (keep_counts)
        {
            const usize at = m.cell_counts.size();
            m.cell_counts.resize(at + list.count);
            cell_counts = m.cell_counts.data() + at;
        }
        GraphTotals totals = empty_totals();
        u32* used = m.virtual_scratch.data();
        for (u32 q = 0; q < p.view.num_virtual; ++q)
            used[q] = 0;
        for (u32 i = 0; i < list.count; ++i)
        {
            const Cell& c = list.cells[i];
            const u32 n = cell_count(c, h);
            if (cell_counts)
                cell_counts[i] = n;
            totals.tuples += n;
            totals.args += u64{n} * c.arity;
            if (n && over_auxiliary(c))
                totals.flags |= k_graph_auxiliary;
            if (n && p.view.num_virtual != 0 && p.view.relation_virtual[c.relation] != k_no_virtual)
                used[p.view.relation_virtual[c.relation]] = 1;
            if (p.relation_major)
                m.tally[c.relation] += n;
        }
        for (u32 q = 0; q < p.view.num_virtual; ++q)
            totals.virtuals += used[q];
        totals.targets = p.targets_of(nodes);
        totals.entities = v.num_objects + ((totals.flags & k_graph_auxiliary) ? 1u : 0u) + p.states_of(nodes) +
                          totals.virtuals;
        m.totals[g - lo] = totals;
        m.entities += totals.entities;
        m.tuples += totals.tuples;
        m.args += totals.args;
        m.targets += totals.targets;
    }
}

/// Pass two of a member: the positions (relation-major) and the contents of its graphs.
template<class I>
void fill_range(const HPass& p, const FlatBuffers& b, u32 t)
{
    const u64 lo = p.s.bounds[t], hi = p.s.bounds[t + 1];
    const GraphOut<I> out{p.graphs,
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
                          nullptr,
                          nullptr,
                          nullptr};
    I* entity_offsets = static_cast<I*>(b.entity_offsets);
    Member& m = p.s.members[t];
    u64 entity = m.entity_start, tuple = m.tuple_start, arg = m.arg_start, target = m.target_start;
    const u32 R = p.R;
    auto fill_one = [&](u64 g, const u32* graph_counts)
    {
        const u32 instance = p.instance_of(g);
        const InstanceView v = instance_view(p.view, instance);
        const GraphTotals& totals = m.totals[g - lo];
        RowLayers layers;
        const Graph h = p.graph(g, v, layers);
        entity_offsets[g] = static_cast<I>(entity);
        b.task_ids[g] = p.in.task_ids ? p.in.task_ids[g] : 0;
        // the entities: objects, the auxiliary entity, the state nodes, the predicate nodes
        const bool auxiliary = (totals.flags & k_graph_auxiliary) != 0;
        const u32 skip = p.exclude ? 1u : 0u;
        const u64 auxiliary_entity = entity + v.num_objects;
        const u64 state_first = auxiliary_entity + (auxiliary ? 1u : 0u);
        const u32 states = p.states_of(h.count);
        const u64 virtual_first = state_first + states;
        fill_graph_header<I>(v, h.row, g, target, out);
        fill_objects<I>(0, v.num_objects, entity, out);
        if (auxiliary)
            fill_auxiliary_entity<I>(v.num_objects, entity, out);
        if (out.entity_role)
            for (u32 k = 0; k < states; ++k)
            {
                out.entity_role[state_first + k] = k_role_state;
                out.entity_ref[state_first + k] = static_cast<I>(k + skip);
            }
        if (out.target_entity)
            for (u32 k = p.include_root_target ? 0u : 1u; k < h.count; ++k)
            {
                out.target_entity[target + k - (p.include_root_target ? 0u : 1u)] =
                    static_cast<I>(state_first + k - skip);
                out.target_group[target + k - (p.include_root_target ? 0u : 1u)] = k_target_group_state;
            }
        const CellList list = p.cells_of(v);
        // the virtual entities: the predicates that have a tuple, in the order of their names
        u32* rank = m.virtual_scratch.empty() ? nullptr : m.virtual_scratch.data();
        if (rank)
        {
            for (u32 q = 0; q < v.num_virtual; ++q)
                rank[q] = 0;
            for (u32 i = 0; i < list.count; ++i)
            {
                const u32 q = v.relation_virtual[list.cells[i].relation];
                if (q != k_no_virtual && graph_counts[i] != 0)
                    rank[q] = 1;
            }
            u32 used = 0;
            for (u32 q = 0; q < v.num_virtual; ++q)
            {
                const u32 flag = rank[q];
                rank[q] = flag ? used : k_no_virtual;
                used += flag;
            }
            const EntityBlocks blocks{0, 0, 0, 0, 0, virtual_first};
            fill_virtual_entities<I>(v, blocks, rank, out);
        }
        // the cells
        I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
        I* offsets = out.relation_offsets + g * R;
        I* arg_offsets = out.relation_arg_offsets + g * R;
        u64 tuples = tuple, args = arg;
        u32 r = 0;
        for (u32 i = 0; i < list.count; ++i)
        {
            const Cell& c = list.cells[i];
            if (p.relation_major)
            {
                if (counts)
                    for (; r < c.relation; ++r)
                        counts[r] = static_cast<I>(0);
            }
            else
                for (; r < c.relation; ++r)
                {
                    offsets[r] = static_cast<I>(tuples);
                    arg_offsets[r] = static_cast<I>(args);
                    if (counts)
                        counts[r] = static_cast<I>(0);
                }
            I* dst;
            if (p.relation_major)
                dst = out.relation_args + out.relation_arg_offsets[cell_segment(true, g, c.relation, p.graphs, R)];
            else
            {
                offsets[c.relation] = static_cast<I>(tuples);
                arg_offsets[c.relation] = static_cast<I>(args);
                dst = out.relation_args + args;
            }
            u64 predicate = k_no_entity;
            if (rank)
            {
                const u32 q = v.relation_virtual[c.relation];
                if (q != k_no_virtual && rank[q] != k_no_virtual)
                    predicate = virtual_first + rank[q];
            }
            const HorizonKind kind = horizon_kind(c.op);
            Writer<I> writer{dst,
                             entity,
                             auxiliary_entity,
                             state_first,
                             skip,
                             predicate,
                             horizon_lead(c.op),
                             over_auxiliary(c),
                             kind == HorizonKind::Parent ? 0u : c.logical_arity};
            emit_cell(c, h, writer);
            const u64 n = static_cast<u64>(writer.dst - dst) / c.arity;
            assert(!graph_counts || n == graph_counts[i]);
            if (counts)
                counts[c.relation] = static_cast<I>(n);
            tuples += n;
            args += n * c.arity;
            r = c.relation + 1;
        }
        if (!p.relation_major)
            for (; r < R; ++r)
            {
                offsets[r] = static_cast<I>(tuples);
                arg_offsets[r] = static_cast<I>(args);
                if (counts)
                    counts[r] = static_cast<I>(0);
            }
        else if (counts)
            for (; r < R; ++r)
                counts[r] = static_cast<I>(0);
        entity += totals.entities;
        tuple += totals.tuples;
        arg += totals.args;
        target += totals.targets;
    };
    const auto cells_of = [&](u64 g) { return p.cells_of(instance_view(p.view, p.instance_of(g))); };
    fill_member<I>(m, {p.graphs, p.R, p.relation_major, p.s.arity.data()}, lo, hi, out.relation_offsets,
                   out.relation_arg_offsets, fill_one, cells_of);
}

HPass make_pass(const EncodeInput& in, Scratch& s, const EncodingContext& context, u64 graphs, u32 members,
                bool evaluate)
{
    const FlatOptions& o = context.schema().options();
    return HPass{in,
                 s,
                 context,
                 context.view(),
                 graphs,
                 members,
                 static_cast<u32>(context.schema().num_relations()),
                 o.layout == Layout::Relation,
                 evaluate,
                 o.include_actions || o.horizon.parent_relation,
                 o.include_actions,
                 o.horizon.root == RootPolicy::Exclude,
                 o.horizon.root == RootPolicy::Include,
                 context.schema().schema_relations().data()};
}
}  // namespace

FlatSizes FlatEncoder::count_horizon(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const
{
    const EncodingContext& context = *m_context;
    Scratch& s = scratch.impl();
    const HorizonInput& h = in.horizon;
    const u64 graphs = h.graphs;
    const FlatOptions& options = m_schema->options();
    const rl::TaskTable& table = this->table();
    const u32 R = static_cast<u32>(m_schema->num_relations());

    // 1. check what does not depend on a graph
    if (in.states.rows != 0)
        invalid("a horizon schema encodes the nodes of in.horizon: in.states must be empty");
    if (in.actions)
        invalid("a horizon takes its actions as the edges of in.horizon: in.actions must be null");
    if (in.history.masks.layers != 0 || in.history.count || in.history.dt)
        invalid("a horizon has no history steps");
    if (h.nodes.numeric_words != 0)
        invalid("numeric state rows are not supported");
    if (h.nodes.stride && h.nodes.stride < h.nodes.words)
        invalid("the node row stride is smaller than its width");
    if (graphs && (!h.node_offsets || !h.edge_offsets))
        invalid("the horizon needs its node_offsets and edge_offsets");
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
    u64 nodes = 0, edges = 0;
    if (graphs)
    {
        if (h.node_offsets[0] != 0)
            invalid("node_offsets must start at 0, it starts at " + std::to_string(h.node_offsets[0]));
        if (h.edge_offsets[0] != 0)
            invalid("edge_offsets must start at 0, it starts at " + std::to_string(h.edge_offsets[0]));
        for (u64 g = 0; g < graphs; ++g)
        {
            if (h.node_offsets[g + 1] <= h.node_offsets[g])
                invalid("graph " + std::to_string(g) + " of the horizon has no nodes (node_offsets " +
                        std::to_string(h.node_offsets[g]) + ", " + std::to_string(h.node_offsets[g + 1]) +
                        "): a graph has at least its root");
            if (h.edge_offsets[g + 1] < h.edge_offsets[g])
                invalid("the edge_offsets of graph " + std::to_string(g) + " of the horizon decrease (" +
                        std::to_string(h.edge_offsets[g]) + ", " + std::to_string(h.edge_offsets[g + 1]) + ")");
        }
        nodes = static_cast<u64>(h.node_offsets[graphs]);
        edges = static_cast<u64>(h.edge_offsets[graphs]);
        if (nodes > h.nodes.rows)
            invalid("the horizon has " + std::to_string(nodes) + " nodes by its node_offsets, but the node batch has " +
                    std::to_string(h.nodes.rows) + " rows");
        if (!h.nodes.data)
            invalid("the node buffer is null");
    }
    const bool edges_used = options.include_actions || options.horizon.parent_relation;
    if (edges_used && edges)
    {
        if (!h.edge_src || !h.edge_dst)
            invalid("the horizon needs its edge_src and edge_dst");
        if (options.include_actions && (!h.edge_schema || !h.edge_binding))
            invalid("the horizon needs its edge_schema and edge_binding");
    }
    if (options.include_actions && graphs && h.label_width < table.label_width())
        invalid("the horizon's label width " + std::to_string(h.label_width) +
                " is below the table's largest action arity " + std::to_string(table.label_width()));
    if (R && graphs > std::numeric_limits<u64>::max() / R - 1)
        throw std::overflow_error("mymyr: encoding: the batch has too many cells");
    const bool evaluate = !in.derived && context.max_derived_words() > 0;
    s.derived_stride = in.derived ? in.derived_stride : context.max_derived_words();
    if (evaluate && nodes * s.derived_stride > s.derived_capacity)
    {
        s.derived_capacity = nodes * s.derived_stride;
        s.derived = std::make_unique_for_overwrite<u64[]>(s.derived_capacity);
    }
    const u32 members = members_for(nodes, pool);
    s.members.resize(members);
    s.bounds.clear();
    if (graphs)
        slice_graphs(h.node_offsets, graphs, members, s.bounds);
    else
        s.bounds.assign(members + 1, 0);

    // 2. validate, evaluate and count, in parallel over the graphs
    const HPass pass = make_pass(in, s, context, graphs, members, evaluate);
    run(pool, members, [&pass](u32 t) { count_range(pass, t); });
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none, narrow_goal = k_none, bad_goal_bits = k_none;
    u64 narrow_subgoals = k_none, bad_subgoal_bits = k_none, bad_edge = k_none;
    for (const Member& m : s.members)
    {
        bad_id = std::min(bad_id, m.bad_id);
        narrow = std::min(narrow, m.narrow);
        bad_bits = std::min(bad_bits, m.bad_bits);
        narrow_goal = std::min(narrow_goal, m.narrow_goal);
        bad_goal_bits = std::min(bad_goal_bits, m.bad_goal_bits);
        narrow_subgoals = std::min(narrow_subgoals, m.narrow_subgoals);
        bad_subgoal_bits = std::min(bad_subgoal_bits, m.bad_subgoal_bits);
        bad_edge = std::min(bad_edge, m.bad_edge);
    }
    auto of_graph = [&](u64 g) { return "graph " + std::to_string(g) + " of the horizon"; };
    if (bad_id != k_none)
        invalid("task id " + std::to_string(in.task_ids[bad_id]) + " of " + of_graph(bad_id) +
                " is outside the table's " + std::to_string(table.size()) + " instances");
    if (narrow != k_none)
        invalid("the nodes have " + std::to_string(h.nodes.words) + " words, the instance of " + of_graph(narrow) +
                " needs " + std::to_string(table.instance(pass.instance_of(narrow)).words));
    if (narrow_goal != k_none)
    {
        const InstanceView v = instance_view(pass.view, pass.instance_of(narrow_goal));
        invalid("the goal masks of " + of_graph(narrow_goal) + " are narrower than its instance needs (" +
                std::to_string(v.words) + " words, " + std::to_string(v.derived_words) + " derived words)");
    }
    if (narrow_subgoals != k_none)
    {
        const InstanceView v = instance_view(pass.view, pass.instance_of(narrow_subgoals));
        invalid("the subgoal layers of " + of_graph(narrow_subgoals) + " are narrower than its instance needs (" +
                std::to_string(v.words) + " words, " + std::to_string(v.derived_words) + " derived words)");
    }
    if (bad_edge != k_none)
    {
        const InstanceView v = instance_view(pass.view, pass.instance_of(bad_edge));
        for (u64 e = pass.first_edge(bad_edge); e < pass.first_edge(bad_edge) + pass.edges_of(bad_edge); ++e)
            if (const char* fault = pass.edge_fault(e, pass.nodes_of(bad_edge), v))
                invalid("edge " + std::to_string(e - pass.first_edge(bad_edge)) + " of " + of_graph(bad_edge) +
                        " has " + fault);
    }
    if (bad_bits != k_none)
    {
        const InstanceView v = instance_view(pass.view, pass.instance_of(bad_bits));
        for (u32 node = 0; node < pass.nodes_of(bad_bits); ++node)
            if (has_foreign_atom_bits(v, h.nodes.row(pass.first_node(bad_bits) + node)))
                invalid("node " + std::to_string(node) + " of " + of_graph(bad_bits) +
                        " sets an atom bit its instance does not have");
        invalid("a derived row of " + of_graph(bad_bits) + " sets an atom bit its instance does not have");
    }
    if (bad_subgoal_bits != k_none)
        invalid("the subgoal layers of " + of_graph(bad_subgoal_bits) + " set an atom bit its instance does not have");
    if (bad_goal_bits != k_none)
        invalid("the goal of " + of_graph(bad_goal_bits) + " sets an atom bit its instance does not have");

    // 3. scan the members
    const FlatSizes sizes = scan_members(s, *m_schema, graphs);
    s.sizes = sizes;
    s.counted = true;
    s.counted_input = in;
    return sizes;
}

void FlatEncoder::fill_horizon(const EncodeInput& in, const FlatSizes&, const FlatBuffers& out, ThreadPool& pool,
                               EncodeScratch& scratch) const
{
    Scratch& s = scratch.impl();
    const FlatOptions& options = m_schema->options();
    const u32 members = static_cast<u32>(s.members.size());
    const bool evaluate = !in.derived && m_context->max_derived_words() > 0;
    const HPass pass = make_pass(in, s, *m_context, in.horizon.graphs, members, evaluate);
    run(pool, members,
        [&pass, &out, wide = options.index == IndexType::I64](u32 t)
        { wide ? fill_range<i64>(pass, out, t) : fill_range<i32>(pass, out, t); });
}
}  // namespace mymyr::encoding
