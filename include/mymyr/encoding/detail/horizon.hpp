#pragma once

// The cells of a horizon batch. A horizon graph is a root state and the nodes below it, all of one instance; its
// relations hold facts of the nodes (anchored by a state entity), goal literals of the root, the satisfaction of the
// goal at the nodes, what a node adds to or deletes from its root, and the edges between the nodes. The cells of a
// horizon instance are `Cell`s (lanes.hpp) whose op is a horizon op: an op of at least k_horizon_op_base, which no
// flat code reads, built from
//   - a kind: what the tuples are (HorizonKind);
//   - a scope: the nodes whose tuples the cell lists (HorizonScope);
//   - lead: whether each tuple starts with the entity of its state node;
//   - minus: the deleting variant of a delta cell, the falsifying variant of an added-satisfaction cell;
//   - low: the flat op that says where the tuple's atoms come from: a CellOp for the cells of facts and delta, the goal
//     op (goal_op, lanes.hpp) for the cells of goal literals.
// Cell::a and b are the slot range (or static rows) as in the flat cells, c and d the rows of the constant goal tuples.
//
// The per-graph rules are written once for the host and for device kernels, like those of lanes.hpp, and this header is
// part of the same device-code subset: a graph is a HorizonGraph (POD views of the nodes, the edges and the graph's
// instance), its cells are counted by horizon_cell_count and filled by fill_horizon_cell, its entities laid out by
// horizon_entities and written by fill_horizon_entities, and positions come from the counts, exactly as for the flat
// cells. The one thing a graph needs between its count and its fill is the canonical form of its edges
// (canonical_horizon_edges): the action tuples are unique per (destination, action relation, objects) and ordered by
// them, the `_parent_` tuples unique per (source, destination) and ordered by them, whatever the order and multiplicity
// of the edges given. The canonical lists are written into `HorizonGraph::order`, a scratch of the caller's of
// `horizon_order_words(edges)` words per graph that must survive from the count to the fill.

#include "mymyr/encoding/detail/lanes.hpp"

namespace mymyr::encoding::detail
{
/// The role of a state entity and the group of the targets that are state entities (EntityRole::State and
/// TargetGroup::State in schema.hpp).
inline constexpr u8 k_role_state = 7;
inline constexpr u8 k_target_group_state = 4;

/// A graph whose edges are not valid: an endpoint outside the graph's nodes, a schema outside the table's, an object
/// outside the instance's.
inline constexpr u32 k_graph_bad_edge = 128u;

/// What the tuples of a horizon cell are.
enum class HorizonKind : u32
{
    Facts = 1,   // the atoms of a predicate at the nodes of the scope (low: Fluent, Derived or Static)
    Delta = 2,   // the atoms a node adds (deletes with `minus`) relative to the root (low: Fluent or Derived)
    Goal = 3,    // the plain goal literals of the root (low: a goal op of mode Plain)
    Sat = 4,     // the goal literals that hold / do not hold at the nodes of the scope (low: goal op, Satisfied / Unsatisfied)
    Added = 5,   // the goal literals a node newly satisfies (falsifies with `minus`): the literals whose atom the node adds or
                 // deletes (low: a goal op of mode Plain)
    Action = 6,  // one tuple per edge whose schema belongs to the relation: (destination, objects)
    Parent = 7,  // one tuple per edge: (source, destination)
};

/// Which nodes of a graph a cell lists: the root alone, the nodes below it, or all of them.
enum class HorizonScope : u32
{
    Root = 0,
    Nodes = 1,
    All = 2,
};

inline constexpr u32 k_horizon_op_base = 1u << 8;

MYMYR_HD constexpr u32 horizon_op(HorizonKind kind, HorizonScope scope, bool lead, bool minus, u32 low)
{
    return (static_cast<u32>(kind) << 8) | (static_cast<u32>(scope) << 12) | (lead ? 1u << 14 : 0u) |
           (minus ? 1u << 15 : 0u) | low;
}
MYMYR_HD constexpr bool is_horizon_op(u32 op) { return op >= k_horizon_op_base; }
MYMYR_HD constexpr HorizonKind horizon_kind(u32 op) { return static_cast<HorizonKind>((op >> 8) & 15u); }
MYMYR_HD constexpr HorizonScope horizon_scope(u32 op) { return static_cast<HorizonScope>((op >> 12) & 3u); }
MYMYR_HD constexpr bool horizon_lead(u32 op) { return ((op >> 14) & 1u) != 0; }
MYMYR_HD constexpr bool horizon_minus(u32 op) { return ((op >> 15) & 1u) != 0; }
MYMYR_HD constexpr u32 horizon_low(u32 op) { return op & 255u; }

/// Everything the cells of one horizon graph read. `order` is the graph's scratch for the canonical edge lists (null for
/// a graph without edges); it is written by canonical_horizon_edges.
struct HorizonGraph
{
    const InstanceView* v;
    GoalWords goal;
    RowInput row;  // the graph's goal and subgoal layers; atoms and derived of a node are given by horizon_node_input
    const u64* nodes;
    u64 node_stride;
    const u64* derived;  // the first node's derived words (null: not read)
    u64 derived_stride;
    u32 count;  // the nodes
    u32 edges;
    const i32* src;
    const i32* dst;
    const i32* schema;
    const i32* binding;
    u32 label_width;
    const i32* schema_relation;  // [schemas] the relation of each action schema (-1: none)
    u32* order;                  // horizon_order_words(edges) words
    bool use_actions;            // the schema has action relations: the edges' schemas and bindings are read
    bool use_parents;            // the schema has the parent relation
    bool exclude;                // the root has no entity
    bool root_target;            // the root is a target
};

/// The words of the scratch `HorizonGraph::order` of a graph with `edges` edges.
MYMYR_HD constexpr u64 horizon_order_words(u32 edges) { return 2u + 2u * u64{edges}; }

MYMYR_HD const u64* horizon_atoms(const HorizonGraph& h, u32 node) { return h.nodes + u64{node} * h.node_stride; }
MYMYR_HD const u64* horizon_derived(const HorizonGraph& h, u32 node)
{
    return h.derived ? h.derived + u64{node} * h.derived_stride : nullptr;
}

/// The row of the node as the flat lanes read a state: the graph's goals with the node's atoms and derived words.
MYMYR_HD RowInput horizon_node_input(const HorizonGraph& h, u32 node)
{
    RowInput row = h.row;
    row.atoms = horizon_atoms(h, node);
    row.derived = horizon_derived(h, node);
    return row;
}

/// The nodes [first, last) a cell lists.
struct HorizonNodes
{
    u32 first, last;
};

MYMYR_HD HorizonNodes horizon_nodes(const Cell& c, u32 count)
{
    switch (horizon_scope(c.op))
    {
        case HorizonScope::Root: return {0, 1};
        case HorizonScope::Nodes: return {1, count};
        default: return {0, count};
    }
}

/// Whether a cell's tuples of a nullary predicate are over the auxiliary entity.
MYMYR_HD bool horizon_over_auxiliary(const Cell& c)
{
    const HorizonKind kind = horizon_kind(c.op);
    return c.logical_arity == 0 && kind != HorizonKind::Action && kind != HorizonKind::Parent;
}

/// The state entities of a graph of `nodes` nodes, and its targets.
MYMYR_HD u32 horizon_states(u32 nodes, bool exclude) { return nodes - (exclude ? 1u : 0u); }
MYMYR_HD u32 horizon_targets(u32 nodes, bool root_target) { return nodes - (root_target ? 0u : 1u); }

// ---------------------------------------------------------------------------------------------------------------------
// The edges. The action tuples of a graph are the distinct (destination, relation, objects) of its edges, the parent
// tuples the distinct (source, destination); under the root policy `exclude` the root has no entity, so an edge into the
// root has no action tuple and an edge that touches it no parent tuple.

MYMYR_HD i32 horizon_edge_relation(const HorizonGraph& h, u32 e) { return h.schema_relation[h.schema[e]]; }

/// Whether edge a's action tuple is strictly before edge b's: ordered by destination, relation, then objects.
struct HorizonActionBefore
{
    const HorizonGraph* h;
    MYMYR_HD bool operator()(u32 a, u32 b) const
    {
        if (h->dst[a] != h->dst[b])
            return h->dst[a] < h->dst[b];
        const i32 ra = horizon_edge_relation(*h, a), rb = horizon_edge_relation(*h, b);
        if (ra != rb)
            return ra < rb;
        return tuple_before(h->binding + u64{a} * h->label_width, h->binding + u64{b} * h->label_width,
                            h->v->schema_arity[h->schema[a]]);
    }
};

/// Whether edge a's parent tuple is strictly before edge b's.
struct HorizonParentBefore
{
    const HorizonGraph* h;
    MYMYR_HD bool operator()(u32 a, u32 b) const
    {
        return h->src[a] != h->src[b] ? h->src[a] < h->src[b] : h->dst[a] < h->dst[b];
    }
};

/// Restores the max-heap of `idx[0, n)` below `root` (heap sort: no allocation, no recursion).
template<class Before>
MYMYR_HD void horizon_sift_down(u32* idx, u32 root, u32 n, const Before& before)
{
    for (;;)
    {
        u32 child = 2u * root + 1u;
        if (child >= n)
            return;
        if (child + 1u < n && before(idx[child], idx[child + 1u]))
            ++child;
        if (!before(idx[root], idx[child]))
            return;
        const u32 t = idx[root];
        idx[root] = idx[child];
        idx[child] = t;
        root = child;
    }
}

/// Sorts the n edge indices by `before` and drops the ones equal to their predecessor; returns what is left. Edges in
/// strictly ascending order (the usual case) are left as they are.
template<class Before>
MYMYR_HD u32 horizon_sort_unique(u32* idx, u32 n, const Before& before)
{
    bool ascending = true;
    for (u32 i = 1; i < n && ascending; ++i)
        ascending = before(idx[i - 1u], idx[i]);
    if (ascending)
        return n;
    for (u32 i = n / 2u; i-- > 0;)
        horizon_sift_down(idx, i, n, before);
    for (u32 end = n; end-- > 1;)
    {
        const u32 t = idx[0];
        idx[0] = idx[end];
        idx[end] = t;
        horizon_sift_down(idx, 0, end, before);
    }
    u32 m = 0;
    for (u32 i = 0; i < n; ++i)
        if (m == 0 || before(idx[m - 1u], idx[i]))
            idx[m++] = idx[i];
    return m;
}

/// The action edges (`order[2, 2 + order[0])`) and the parent edges (the `order[1]` entries after the `edges` of the
/// first list) of the graph, canonical: strictly ascending in the edges' action tuples / parent tuples. The edges' ends,
/// schemas and bindings must be valid.
MYMYR_HD void canonical_horizon_edges(const HorizonGraph& h)
{
    if (!h.order)
        return;
    h.order[0] = h.order[1] = 0;
    u32* actions = h.order + 2;
    u32* parents = actions + h.edges;
    if (h.use_actions)
    {
        u32 n = 0;
        for (u32 e = 0; e < h.edges; ++e)
            if (!(h.exclude && h.dst[e] == 0))
                actions[n++] = e;
        h.order[0] = horizon_sort_unique(actions, n, HorizonActionBefore{&h});
    }
    if (h.use_parents)
    {
        u32 n = 0;
        for (u32 e = 0; e < h.edges; ++e)
            if (!(h.exclude && (h.src[e] == 0 || h.dst[e] == 0)))
                parents[n++] = e;
        h.order[1] = horizon_sort_unique(parents, n, HorizonParentBefore{&h});
    }
}

/// The tuples of the action cell of `relation` in the graph.
MYMYR_HD u32 horizon_action_count(const HorizonGraph& h, u32 relation)
{
    u32 n = 0;
    if (h.order)
    {
        const u32* actions = h.order + 2;
        for (u32 i = 0; i < h.order[0]; ++i)
            n += static_cast<u32>(horizon_edge_relation(h, actions[i])) == relation ? 1u : 0u;
    }
    return n;
}

// ---------------------------------------------------------------------------------------------------------------------
// The slot based cells.

/// Word `w` of the slots [lo, hi) of the word `x` (the word of index w), the others cleared.
MYMYR_HD u64 horizon_clip_word(u64 x, u32 w, u32 lo, u32 hi)
{
    if (w == (lo >> 6))
        x &= ~u64{0} << (lo & 63u);
    if (w == ((hi - 1u) >> 6))
        x &= ~u64{0} >> (63u - ((hi - 1u) & 63u));
    return x;
}

/// Word `w` of the slots of cell `c` that are tuples at `node` (the cell is slot based: not static facts, not an edge
/// cell).
MYMYR_HD u64 horizon_cell_word(const Cell& c, const HorizonGraph& h, u32 node, u32 w)
{
    const u32 low = horizon_low(c.op);
    switch (horizon_kind(c.op))
    {
        case HorizonKind::Facts:
        {
            const u64* x = low == static_cast<u32>(CellOp::Derived) ? horizon_derived(h, node) : horizon_atoms(h, node);
            return x ? range_word(x, w, c.a, c.b) : 0;
        }
        case HorizonKind::Delta:
        {
            const bool derived = low == static_cast<u32>(CellOp::Derived);
            const u64* x = derived ? horizon_derived(h, node) : horizon_atoms(h, node);
            const u64* root = derived ? horizon_derived(h, 0) : horizon_atoms(h, 0);
            if (!x || !root)
                return 0;
            return horizon_clip_word(horizon_minus(c.op) ? root[w] & ~x[w] : x[w] & ~root[w], w, c.a, c.b);
        }
        case HorizonKind::Goal:
        case HorizonKind::Sat:
        {
            Cell flat = c;
            flat.op = low;
            return goal_word<true>(flat, h.goal, horizon_node_input(h, node), w);
        }
        case HorizonKind::Added:
        {
            const bool derived = goal_op_derived(low);
            const u64* x = derived ? horizon_derived(h, node) : horizon_atoms(h, node);
            const u64* root = derived ? horizon_derived(h, 0) : horizon_atoms(h, 0);
            if (!x || !root)
                return 0;
            // a positive literal is newly satisfied by an atom the node adds and newly falsified by one it deletes; a
            // negative literal the other way round. The atoms the node changes come first: most nodes change few.
            const bool added = horizon_minus(c.op) == goal_op_negative(low);
            const u64 changed = horizon_clip_word(added ? x[w] & ~root[w] : root[w] & ~x[w], w, c.a, c.b);
            if (!changed)
                return 0;
            Cell flat = c;
            flat.op = low;
            return changed & goal_word<true>(flat, h.goal, h.row, w);
        }
        default: return 0;
    }
}

/// Whether the cell has no tuple at any node: an added-satisfaction cell whose slots hold no literal of the goal.
MYMYR_HD bool horizon_cell_empty(const Cell& c, const HorizonGraph& h)
{
    if (horizon_kind(c.op) != HorizonKind::Added)
        return false;
    Cell flat = c;
    flat.op = horizon_low(c.op);
    for (u32 w = c.a >> 6; c.a < c.b && w <= ((c.b - 1u) >> 6); ++w)
        if (goal_word<true>(flat, h.goal, h.row, w) != 0)
            return false;
    return true;
}

/// The tuples of cell `c` in the graph.
MYMYR_HD u32 horizon_cell_count(const Cell& c, const HorizonGraph& h)
{
    const HorizonKind kind = horizon_kind(c.op);
    if (kind == HorizonKind::Action)
        return horizon_action_count(h, c.relation);
    if (kind == HorizonKind::Parent)
        return h.order ? h.order[1] : 0u;
    const HorizonNodes nodes = horizon_nodes(c, h.count);
    if (kind == HorizonKind::Facts && horizon_low(c.op) == static_cast<u32>(CellOp::Static))
        return (nodes.last - nodes.first) * (c.b - c.a);
    if (horizon_cell_empty(c, h))
        return 0;
    const u32 constants = h.goal.constants && (kind == HorizonKind::Goal || kind == HorizonKind::Sat) ? c.d - c.c : 0u;
    u32 n = 0;
    for (u32 node = nodes.first; node < nodes.last; ++node)
    {
        n += constants;
        if (c.a < c.b)
            for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
                n += popcount64(horizon_cell_word(c, h, node, w));
    }
    return n;
}

/// Calls out.put(node, objects) for the tuples of a slot based cell at `node`, in the order of the cell: the cell's
/// slots in ascending order, with the constant goal tuples merged in.
template<class Out>
MYMYR_HD void horizon_emit_node(const Cell& c, const HorizonGraph& h, u32 node, Out& out)
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
            u64 bits = horizon_cell_word(c, h, node, w);
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
MYMYR_HD void horizon_emit_cell(const Cell& c, const HorizonGraph& h, Out& out)
{
    const HorizonKind kind = horizon_kind(c.op);
    if (kind == HorizonKind::Action)
    {
        if (h.order)
        {
            const u32* actions = h.order + 2;
            for (u32 i = 0; i < h.order[0]; ++i)
            {
                const u32 e = actions[i];
                if (static_cast<u32>(horizon_edge_relation(h, e)) == c.relation)
                    out.put(static_cast<u32>(h.dst[e]), h.binding + u64{e} * h.label_width);
            }
        }
        return;
    }
    if (kind == HorizonKind::Parent)
    {
        if (h.order)
        {
            const u32* parents = h.order + 2 + h.edges;
            for (u32 i = 0; i < h.order[1]; ++i)
                out.put_pair(static_cast<u32>(h.src[parents[i]]), static_cast<u32>(h.dst[parents[i]]));
        }
        return;
    }
    const HorizonNodes nodes = horizon_nodes(c, h.count);
    const bool statics = kind == HorizonKind::Facts && horizon_low(c.op) == static_cast<u32>(CellOp::Static);
    if (!statics && horizon_cell_empty(c, h))
        return;
    for (u32 node = nodes.first; node < nodes.last; ++node)
    {
        if (statics)
            for (u32 row = c.a; row < c.b; ++row)
                out.put(node, h.v->static_args + u64{row} * h.v->max_arity);
        else
            horizon_emit_node(c, h, node, out);
    }
}

/// The sizes of one graph: its tuples, arguments, entities, targets and the predicates that have a tuple, with
/// `sink(i, cell, n)` called for the n tuples of each cell i of `list`. `used` is a scratch of v.num_virtual words (null
/// without predicate virtual nodes). The graph's canonical edges must be computed (canonical_horizon_edges).
template<class Sink>
MYMYR_HD GraphTotals horizon_count_graph(const InstanceView& v, const HorizonGraph& h, const CellList& list,
                                         u32* used, Sink& sink)
{
    GraphTotals totals = empty_totals();
    for (u32 q = 0; q < v.num_virtual; ++q)
        used[q] = 0;
    for (u32 i = 0; i < list.count; ++i)
    {
        const Cell& c = list.cells[i];
        const u32 n = horizon_cell_count(c, h);
        sink(i, c, n);
        totals.tuples += n;
        totals.args += u64{n} * c.arity;
        if (n && horizon_over_auxiliary(c))
            totals.flags |= k_graph_auxiliary;
        if (n && v.num_virtual != 0 && v.relation_virtual[c.relation] != k_no_virtual)
            used[v.relation_virtual[c.relation]] = 1;
    }
    for (u32 q = 0; q < v.num_virtual; ++q)
        totals.virtuals += used[q];
    totals.targets = horizon_targets(h.count, h.root_target);
    totals.entities = v.num_objects + ((totals.flags & k_graph_auxiliary) ? 1u : 0u) +
                      horizon_states(h.count, h.exclude) + totals.virtuals;
    return totals;
}

// ---------------------------------------------------------------------------------------------------------------------
// The entities. A horizon graph has, in this order, its objects, the auxiliary entity (iff some nullary tuple exists), a
// state entity per node (all but the root under `exclude`) and the predicate virtual nodes.

/// The first entity of each block of a graph (batch-global ids), and how many nodes have none.
struct HorizonEntities
{
    u64 objects;    // the first object
    u64 auxiliary;  // the auxiliary entity, if the graph has one
    u64 state;      // the entity of the first node that has one
    u64 virtuals;   // the first virtual entity
    u32 skip;       // the nodes before the first that has an entity
    u32 states;     // the state entities
    bool has_auxiliary;
};

MYMYR_HD HorizonEntities horizon_entities(const InstanceView& v, const HorizonGraph& h, const GraphTotals& totals,
                                          u64 entity_base)
{
    HorizonEntities e;
    e.objects = entity_base;
    e.auxiliary = entity_base + v.num_objects;
    e.has_auxiliary = (totals.flags & k_graph_auxiliary) != 0;
    e.state = e.auxiliary + (e.has_auxiliary ? 1u : 0u);
    e.skip = h.exclude ? 1u : 0u;
    e.states = horizon_states(h.count, h.exclude);
    e.virtuals = e.state + e.states;
    return e;
}

/// Writes the graph's header, its objects, the auxiliary entity and the state entities with their roles, node references
/// and targets (the virtual entities are written by fill_virtual_entities). `target_base` is the batch position of the
/// graph's first target.
template<class I>
MYMYR_HD void fill_horizon_entities(const InstanceView& v, const HorizonGraph& h, const HorizonEntities& e, u64 g,
                                    u64 target_base, const GraphOut<I>& out)
{
    fill_graph_header<I>(v, h.row, g, target_base, out);
    fill_objects<I>(0, v.num_objects, e.objects, out);
    if (e.has_auxiliary)
        fill_auxiliary_entity<I>(v.num_objects, e.objects, out);
    if (out.entity_role)
        for (u32 k = 0; k < e.states; ++k)
        {
            out.entity_role[e.state + k] = k_role_state;
            out.entity_ref[e.state + k] = static_cast<I>(k + e.skip);
        }
    if (out.target_entity)
    {
        const u32 first = h.root_target ? 0u : 1u;
        for (u32 k = first; k < h.count; ++k)
        {
            out.target_entity[target_base + k - first] = static_cast<I>(e.state + k - e.skip);
            out.target_group[target_base + k - first] = k_target_group_state;
        }
    }
}

/// The virtual rank of every predicate in the graph, as virtual_ranks (lanes.hpp) over `cell_counts` (one per cell of
/// `list`).
MYMYR_HD u32 horizon_virtual_ranks(const InstanceView& v, const CellList& list, const u32* cell_counts, u32* rank)
{
    for (u32 q = 0; q < v.num_virtual; ++q)
        rank[q] = 0;
    for (u32 i = 0; i < list.count; ++i)
    {
        const u32 q = v.relation_virtual[list.cells[i].relation];
        if (q != k_no_virtual && cell_counts[i] != 0)
            rank[q] = 1;
    }
    u32 used = 0;
    for (u32 q = 0; q < v.num_virtual; ++q)
    {
        const u32 flag = rank[q];
        rank[q] = flag ? used : k_no_virtual;
        used += flag;
    }
    return used;
}

/// Writes the entity ids of a cell's tuples, for the cell's emit.
template<class I>
struct HorizonWriter
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

    MYMYR_HD u64 state(u32 node) const { return state_first + node - skip; }
    MYMYR_HD void put(u32 node, const i32* row)
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
    MYMYR_HD void put_pair(u32 a, u32 b)
    {
        *dst++ = static_cast<I>(state(a));
        *dst++ = static_cast<I>(state(b));
    }
};

/// Writes the tuples of cell `c` to `dst` (the entity ids; `predicate` is the cell's virtual entity or k_no_entity) and
/// returns how many.
template<class I>
MYMYR_HD u32 fill_horizon_cell(const Cell& c, const HorizonGraph& h, const HorizonEntities& e, u64 predicate, I* dst)
{
    const HorizonKind kind = horizon_kind(c.op);
    HorizonWriter<I> writer{dst,
                            e.objects,
                            e.auxiliary,
                            e.state,
                            e.skip,
                            predicate,
                            horizon_lead(c.op),
                            horizon_over_auxiliary(c),
                            kind == HorizonKind::Parent ? 0u : c.logical_arity};
    horizon_emit_cell(c, h, writer);
    return static_cast<u32>(static_cast<u64>(writer.dst - dst) / c.arity);
}
}  // namespace mymyr::encoding::detail
