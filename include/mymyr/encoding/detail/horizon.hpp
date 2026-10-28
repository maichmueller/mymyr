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
// of the edges given. Edges that come ordered by (destination, schema, objects), as the horizon builder writes them,
// are canonicalized in one pass each; any other order is sorted (the action edges are grouped by destination by a
// counting pass and only the edges of one group are sorted; the parent pairs are bucketed by source after a counting
// pass by destination), so the cost is linear in the edges for graphs of a bounded degree either way. The canonical
// lists are written into `HorizonGraph::order`, a scratch of the caller's of `horizon_order_words(nodes, edges)` words
// per graph that must survive from the count to the fill.
//
// The relations `_sibling_` and `_cousin_` are read from the same canonical parent edges (the ones the root policy keeps,
// without the self-loops): a graph's kinship (horizon_kin_prepare, horizon_kin_fill_far) is built into
// `HorizonGraph::kin`, a scratch of `horizon_kin_words(nodes, edges)` words, and, for cousins, `HorizonGraph::far`, a
// scratch of as many words as horizon_kin_prepare returns; both must survive from the count to the fill like `order`.

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
    Sibling = 8, // the pairs of nodes that share a parent, both orders, ascending (kin_row)
    Cousin = 9,  // the pairs of nodes whose parents are siblings, both orders, ascending (kin_row)
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
    u32* order;                  // horizon_order_words(count, edges) words
    u32* kin;                    // horizon_kin_words(count, edges) words (null without sibling or cousin relations)
    u32* far;                    // the far lists of the cousins (null: gather cousin rows on demand)
    bool use_actions;            // the schema has action relations: the edges' schemas and bindings are read
    bool one_schema_per_relation;  // no two schemas share an action relation
    bool use_parents;            // the canonical parent edges are needed: the schema has a parent, sibling or cousin relation
    bool use_cousins;            // the schema has the cousin relation
    bool exclude;                // the root has no entity
    bool root_target;            // the root is a target
    const i32* parent_edges = nullptr;  // optional canonical pairs, graph-local edge indices
    u32 parent_pairs = 0;
};

/// The words of the scratch `HorizonGraph::order` of a graph with `nodes` nodes and `edges` edges: the number of action
/// edges and of parent pairs, the action edges, the destinations of the parent pairs, the offsets and fill counts of
/// the parent lists of the nodes, and the scratch of the sorts (a count per node and a list of the edges).
inline constexpr u64 k_order_words_per_graph = 3, k_order_words_per_edge = 3, k_order_words_per_node = 3;
MYMYR_HD constexpr u64 horizon_order_words(u32 nodes, u32 edges)
{
    return k_order_words_per_graph + k_order_words_per_edge * u64{edges} + k_order_words_per_node * u64{nodes};
}

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

/// Whether a cell's tuples of a nullary predicate are over the auxiliary entity: the cell is of a nullary predicate and
/// the context gives nullary tuples the auxiliary slot.
MYMYR_HD bool horizon_over_auxiliary(const InstanceView& v, const Cell& c)
{
    const HorizonKind kind = horizon_kind(c.op);
    return v.nullary_auxiliary && c.logical_arity == 0 && kind != HorizonKind::Action && kind != HorizonKind::Parent &&
           kind != HorizonKind::Sibling && kind != HorizonKind::Cousin;
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
/// strictly ascending order are left as they are; small runs are sorted by insertion, large ones by heap sort.
template<class Before>
MYMYR_HD u32 horizon_sort_unique(u32* idx, u32 n, const Before& before)
{
    bool ascending = true;
    for (u32 i = 1; i < n && ascending; ++i)
        ascending = before(idx[i - 1u], idx[i]);
    if (ascending)
        return n;
    if (n <= 16u)
    {
        for (u32 i = 1; i < n; ++i)
        {
            const u32 x = idx[i];
            u32 j = i;
            for (; j > 0 && before(x, idx[j - 1u]); --j)
                idx[j] = idx[j - 1u];
            idx[j] = x;
        }
    }
    else
    {
        for (u32 i = n / 2u; i-- > 0;)
            horizon_sift_down(idx, i, n, before);
        for (u32 end = n; end-- > 1;)
        {
            const u32 t = idx[0];
            idx[0] = idx[end];
            idx[end] = t;
            horizon_sift_down(idx, 0, end, before);
        }
    }
    u32 m = 0;
    for (u32 i = 0; i < n; ++i)
        if (m == 0 || before(idx[m - 1u], idx[i]))
            idx[m++] = idx[i];
    return m;
}

/// The edges `kept` accepts, sorted by `before` without repeats, written to `out`; returns their number. `kept.key` is the
/// leading part of `before` (a node below `nodes`): the edges are grouped by it first, in place if they come grouped
/// already (as the horizon builder writes the action edges) and otherwise by a stable counting pass over `counts[nodes]`,
/// and then every group is sorted on its own.
template<class Edges, class Before>
MYMYR_HD u32 horizon_canonical_list(u32* out, u32 edges, u32 nodes, u32* counts, const Edges& kept, const Before& before)
{
    bool grouped = true;
    u32 last = 0;
    for (u32 e = 0; e < edges && grouped; ++e)
        if (kept(e))
        {
            const u32 k = kept.key(e);
            grouped = k >= last;
            last = k;
        }
    u32 n = 0;
    if (grouped)
    {
        for (u32 e = 0; e < edges; ++e)
            if (kept(e))
                out[n++] = e;
    }
    else
    {
        for (u32 k = 0; k < nodes; ++k)
            counts[k] = 0;
        for (u32 e = 0; e < edges; ++e)
            if (kept(e))
                ++counts[kept.key(e)];
        for (u32 k = 0; k < nodes; ++k)
        {
            const u32 c = counts[k];
            counts[k] = n;
            n += c;
        }
        for (u32 e = 0; e < edges; ++e)
            if (kept(e))
                out[counts[kept.key(e)]++] = e;
    }
    u32 w = 0;
    for (u32 i = 0; i < n;)
    {
        const u32 k = kept.key(out[i]);
        u32 j = i + 1u;
        while (j < n && kept.key(out[j]) == k)
            ++j;
        const u32 m = horizon_sort_unique(out + i, j - i, before);
        if (w != i)
            for (u32 t = 0; t < m; ++t)
                out[w + t] = out[i + t];
        w += m;
        i = j;
    }
    return w;
}

/// The edges whose action tuples exist: under `exclude` those into the root do not, and they are grouped by destination.
struct HorizonActionEdges
{
    const HorizonGraph* h;
    MYMYR_HD bool operator()(u32 e) const { return !(h->exclude && h->dst[e] == 0); }
    MYMYR_HD u32 key(u32 e) const { return static_cast<u32>(h->dst[e]); }
};

/// The parent pairs of a graph, the distinct (source, destination) of its edges: for each source, in `dst` from
/// `begin[source]` on, its `fill[source]` destinations in ascending order. The pairs in the order of their sources are
/// the canonical parent tuples.
struct HorizonParents
{
    const u32* dst;
    const u32* begin;
    const u32* fill;
};

MYMYR_HD HorizonParents horizon_parents(const HorizonGraph& h)
{
    const u32* dst = h.order + 2u + h.edges;
    return {dst, dst + h.edges, dst + h.edges + (h.count + 1u)};
}

/// Visits canonical parent pairs in source/destination order, with the root policy applied.
template<class Sink>
MYMYR_HD void horizon_each_parent(const HorizonGraph& h, const Sink& sink)
{
    if (h.parent_edges)
    {
        for (u32 i = 0; i < h.parent_pairs; ++i)
        {
            const u32 e = static_cast<u32>(h.parent_edges[i]);
            if (!h.exclude || (h.src[e] != 0 && h.dst[e] != 0))
                sink(static_cast<u32>(h.src[e]), static_cast<u32>(h.dst[e]));
        }
    }
    else if (h.order)
    {
        const HorizonParents pairs = horizon_parents(h);
        for (u32 src = 0; src < h.count; ++src)
            for (u32 i = 0; i < pairs.fill[src]; ++i)
                sink(src, pairs.dst[pairs.begin[src] + i]);
    }
}

/// The parent pairs of the graph, written into its order scratch (horizon_parents); returns their number. Under `exclude`
/// the edges that touch the root are left out. The edges are taken in the order of their destinations, so that the
/// destinations of one source come out ascending and a repeated pair is the last one of its source: an edge list that is
/// ordered by destination (as the horizon builder writes it) is read as it is, any other is first ordered by destination
/// by a stable counting pass over `counts[nodes]` and `tmp[edges]`. The cost is linear in the edges and nodes.
MYMYR_HD u32 horizon_canonical_parents(const HorizonGraph& h, u32* counts, u32* tmp)
{
    const u32 nodes = h.count;
    u32* dst_of = h.order + 2u + h.edges;
    u32* begin = dst_of + h.edges;
    u32* fill = begin + (nodes + 1u);
    const auto kept = [&](u32 e) { return !(h.exclude && (h.src[e] == 0 || h.dst[e] == 0)); };
    for (u32 k = 0; k < nodes; ++k)
        fill[k] = 0;
    bool by_destination = true;
    i32 last = 0;
    for (u32 e = 0; e < h.edges; ++e)
        if (kept(e))
        {
            by_destination = by_destination && h.dst[e] >= last;
            last = h.dst[e];
            ++fill[h.src[e]];
        }
    u32 count = 0;
    const u32* order = nullptr;  // the kept edges by destination (null: the edges themselves)
    u32 listed = 0;
    if (!by_destination)
    {
        for (u32 k = 0; k < nodes; ++k)
            counts[k] = 0;
        for (u32 e = 0; e < h.edges; ++e)
            if (kept(e))
                ++counts[h.dst[e]];
        u32 at = 0;
        for (u32 k = 0; k < nodes; ++k)
        {
            const u32 c = counts[k];
            counts[k] = at;
            at += c;
        }
        for (u32 e = 0; e < h.edges; ++e)
            if (kept(e))
                tmp[counts[h.dst[e]]++] = e;
        order = tmp;
        listed = at;
    }
    u32 at = 0;
    for (u32 k = 0; k < nodes; ++k)
    {
        begin[k] = at;
        at += fill[k];
        fill[k] = 0;
    }
    begin[nodes] = at;
    // a repeated pair is written over the slot after its source's last one and not counted (without a branch: whether an
    // edge repeats its predecessor is data dependent)
    const auto add = [&](u32 e)
    {
        const u32 src = static_cast<u32>(h.src[e]), dst = static_cast<u32>(h.dst[e]);
        const u32 f = fill[src];
        u32* slot = dst_of + begin[src] + f;
        const u32 again = static_cast<u32>(f != 0 && slot[-1] == dst);
        *slot = dst;
        fill[src] = f + 1u - again;
        count += 1u - again;
    };
    if (order)
        for (u32 i = 0; i < listed; ++i)
            add(order[i]);
    else
        for (u32 e = 0; e < h.edges; ++e)
            if (kept(e))
                add(e);
    return count;
}

/// The action edges of the graph in the canonical order of each action relation (ascending destination, then objects,
/// without repeats), written to `out`; returns their number. Edges that come ordered by (destination, schema, objects)
/// are taken in one pass, repeated tuples dropped: with one schema per action relation the edges of each relation are
/// then in the canonical order. Any other order is sorted by horizon_canonical_list.
MYMYR_HD u32 horizon_canonical_actions(const HorizonGraph& h, u32* out, u32* counts)
{
    if (h.one_schema_per_relation)
    {
        u32 n = 0;
        u32 last = 0;
        bool ordered = true;
        for (u32 e = 0; e < h.edges && ordered; ++e)
        {
            const i32 dst = h.dst[e];
            if (h.exclude && dst == 0)
                continue;
            if (n != 0)
            {
                const i32 last_dst = h.dst[last];
                if (dst != last_dst)
                    ordered = dst > last_dst;
                else
                {
                    const i32 schema = h.schema[e], last_schema = h.schema[last];
                    if (schema != last_schema)
                        ordered = schema > last_schema;
                    else
                    {
                        const i32* a = h.binding + u64{last} * h.label_width;
                        const i32* b = h.binding + u64{e} * h.label_width;
                        const u32 arity = h.v->schema_arity[schema];
                        if (!tuple_before(a, b, arity))
                        {
                            ordered = !tuple_before(b, a, arity);  // equal: the same tuple again
                            continue;
                        }
                    }
                }
            }
            out[n++] = e;
            last = e;
        }
        if (ordered)
            return n;
    }
    return horizon_canonical_list(out, h.edges, h.count, counts, HorizonActionEdges{&h}, HorizonActionBefore{&h});
}

/// The action edges (`order[2, 2 + order[0])`) and the parent pairs (horizon_parents, `order[1]` of them) of the graph,
/// canonical: the action edges of each relation strictly ascending in their action tuples, the pairs strictly ascending.
/// The edges' ends, schemas and bindings must be valid.
MYMYR_HD void canonical_horizon_edges(const HorizonGraph& h)
{
    if (!h.order)
        return;
    h.order[0] = h.order[1] = 0;
    u32* actions = h.order + 2;
    u32* counts = actions + 2u * u64{h.edges} + (2u * u64{h.count} + 1u);
    if (h.use_actions)
        h.order[0] = horizon_canonical_actions(h, actions, counts);
    if (h.use_parents)
    {
        if (h.parent_edges)
        {
            if (!h.exclude)
                h.order[1] = h.parent_pairs;
            else
            {
                for (u32 i = 0; i < h.parent_pairs; ++i)
                {
                    const u32 e = static_cast<u32>(h.parent_edges[i]);
                    h.order[1] += static_cast<u32>(h.src[e] != 0 && h.dst[e] != 0);
                }
            }
        }
        else
            h.order[1] = horizon_canonical_parents(h, counts, counts + h.count);
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
// The kinship. The parents of a node x are the nodes with a canonical parent edge to x, self-loops left out (a node is
// not its own parent); under the root policy `exclude` the root has no such edge, so it is nobody's parent and nobody's
// child. Two nodes are siblings iff they differ and have a parent in common, and cousins iff they differ and a parent
// of the one and a parent of the other are different siblings. A relation lists each pair in both orders, by node
// and then by the other node, ascending.
//
// The data a node's relatives are read from, in the kin scratch:
//   - children(p) and parents(q): ascending, distinct (the children of p are the destinations of its canonical edges,
//     the parents of q the transpose);
//   - far(p), for cousins: the children of the siblings of p, ascending, distinct. The cousins of a node n are the
//     union of far(p) over the parents p of n, without n.
// A node with one parent has the list of that parent as its relatives; a node with several gets their union, which
// is gathered with a stamp per node and sorted in place.

/// The words of the kin scratch of a graph of `nodes` nodes and `edges` edges.
MYMYR_HD constexpr u64 horizon_kin_words(u32 nodes, u32 edges) { return 5u * u64{nodes} + 3u + 2u * u64{edges}; }

/// The arrays of the kin scratch: the offsets are `nodes + 1` words, `stamp` and `row` `nodes` words each, children
/// and parents one word per edge.
struct KinView
{
    u32 nodes;
    u32* child_offsets;
    u32* parent_offsets;
    u32* far_offsets;
    u32* stamp;
    u32* row;
    u32* children;
    u32* parents;
    u32* far;
};

MYMYR_HD KinView horizon_kin_view(const HorizonGraph& h)
{
    const u64 n = h.count;
    KinView k;
    k.nodes = h.count;
    k.child_offsets = h.kin;
    k.parent_offsets = h.kin + (n + 1u);
    k.far_offsets = h.kin + 2u * (n + 1u);
    k.stamp = h.kin + 3u * (n + 1u);
    k.row = k.stamp + n;
    k.children = k.row + n;
    k.parents = k.children + h.edges;
    k.far = h.far;
    return k;
}

/// Calls visit(x) for every node x of far(p) with repeats: the children of the siblings of p.
template<class Visit>
MYMYR_HD void kin_visit_far(const KinView& k, u32 p, Visit&& visit)
{
    for (u32 i = k.parent_offsets[p]; i < k.parent_offsets[p + 1u]; ++i)
    {
        const u32 g = k.parents[i];
        for (u32 j = k.child_offsets[g]; j < k.child_offsets[g + 1u]; ++j)
        {
            const u32 s = k.children[j];
            if (s == p)
                continue;
            for (u32 c = k.child_offsets[s]; c < k.child_offsets[s + 1u]; ++c)
                visit(k.children[c]);
        }
    }
}

struct HorizonValueBefore
{
    MYMYR_HD bool operator()(u32 a, u32 b) const { return a < b; }
};

MYMYR_HD void kin_clear_stamps(const KinView& k)
{
    for (u32 i = 0; i < k.nodes; ++i)
        k.stamp[i] = 0;
}

/// Builds the children and parents lists of the graph from its canonical parent edges (canonical_horizon_edges must
/// have run), and for a cousin relation the offsets of the far lists. Returns the entries of the far lists, which the
/// caller makes room for in `HorizonGraph::far` before calling horizon_kin_fill_far; they number at most the square of
/// the nodes and may not fit the offsets' 32 bits, which the caller must check before reading them.
MYMYR_HD u64 horizon_kin_prepare(const HorizonGraph& h)
{
    const KinView k = horizon_kin_view(h);
    const u32 n = k.nodes;
    for (u32 i = 0; i <= n; ++i)
        k.child_offsets[i] = k.parent_offsets[i] = k.far_offsets[i] = 0;
    u32 total = 0;
    horizon_each_parent(h, [&](u32 src, u32 dst)
    {
        if (src == dst)
            return;
        k.children[total++] = dst;
        ++k.child_offsets[src + 1u];
        ++k.parent_offsets[dst + 1u];
    });
    for (u32 i = 0; i < n; ++i)
    {
        k.child_offsets[i + 1u] += k.child_offsets[i];
        k.parent_offsets[i + 1u] += k.parent_offsets[i];
    }
    for (u32 i = 0; i < n; ++i)
        k.row[i] = k.parent_offsets[i];
    for (u32 p = 0; p < n; ++p)
        for (u32 j = k.child_offsets[p]; j < k.child_offsets[p + 1u]; ++j)
            k.parents[k.row[k.children[j]]++] = p;
    if (!h.use_cousins)
        return 0;
    kin_clear_stamps(k);
    u64 far_total = 0;
    for (u32 p = 0; p < n; ++p)
    {
        u32 count = 0;
        kin_visit_far(k, p,
                      [&](u32 x)
                      {
                          if (k.stamp[x] != p + 1u)
                          {
                              k.stamp[x] = p + 1u;
                              ++count;
                          }
                      });
        far_total += count;
        k.far_offsets[p + 1u] = static_cast<u32>(far_total);
    }
    return far_total;
}

/// Writes the far lists, after horizon_kin_prepare and with `HorizonGraph::far` holding the entries it returned.
MYMYR_HD void horizon_kin_fill_far(const HorizonGraph& h)
{
    const KinView k = horizon_kin_view(h);
    kin_clear_stamps(k);
    for (u32 p = 0; p < k.nodes; ++p)
    {
        u32* out = k.far + k.far_offsets[p];
        u32 count = 0;
        kin_visit_far(k, p,
                      [&](u32 x)
                      {
                          if (k.stamp[x] != p + 1u)
                          {
                              k.stamp[x] = p + 1u;
                              out[count++] = x;
                          }
                      });
        horizon_sort_unique(out, count, HorizonValueBefore{});
    }
}

/// The list the relatives of a node come from through its parent: the children of the parent for a sibling, its far
/// list for a cousin. The list of a sibling includes the node itself.
MYMYR_HD const u32* kin_parent_list(const KinView& k, bool cousin, u32 parent, u32& size)
{
    const u32* offsets = cousin ? k.far_offsets : k.child_offsets;
    size = offsets[parent + 1u] - offsets[parent];
    return (cousin ? k.far : k.children) + offsets[parent];
}

MYMYR_HD bool kin_contains(const u32* list, u32 size, u32 value)
{
    u32 lo = 0, hi = size;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2u;
        if (list[mid] < value)
            lo = mid + 1u;
        else
            hi = mid;
    }
    return lo < size && list[lo] == value;
}

/// The relatives of node `n` as an ascending list that may contain `n` (to be skipped): the list of its only parent
/// as it is, the union of its parents' lists gathered into the scratch otherwise (the gather marks with n + 1, so the
/// stamps are cleared before the first node of a pass).
MYMYR_HD const u32* kin_relatives(const KinView& k, bool cousin, u32 n, u32& size)
{
    const u32 first = k.parent_offsets[n], last = k.parent_offsets[n + 1u];
    if (last == first)
    {
        size = 0;
        return k.row;
    }
    if (last - first == 1u)
        return kin_parent_list(k, cousin, k.parents[first], size);
    u32 count = 0;
    for (u32 i = first; i < last; ++i)
    {
        u32 m;
        const u32* list = kin_parent_list(k, cousin, k.parents[i], m);
        for (u32 j = 0; j < m; ++j)
            if (k.stamp[list[j]] != n + 1u)
            {
                k.stamp[list[j]] = n + 1u;
                k.row[count++] = list[j];
            }
    }
    size = horizon_sort_unique(k.row, count, HorizonValueBefore{});
    return k.row;
}

/// Gathers one cousin row through recorded parent and child lists, using one stamp and row entry per node.
MYMYR_HD const u32* kin_cousin_row(const KinView& k, u32 n, u32& size)
{
    u32 count = 0;
    for (u32 i = k.parent_offsets[n]; i < k.parent_offsets[n + 1u]; ++i)
        kin_visit_far(k, k.parents[i], [&](u32 x)
        {
            if (x != n && k.stamp[x] != n + 1u)
            {
                k.stamp[x] = n + 1u;
                k.row[count++] = x;
            }
        });
    size = horizon_sort_unique(k.row, count, HorizonValueBefore{});
    return k.row;
}

/// The tuples of a sibling or cousin cell, saturated at the largest u32. Without stored far lists, cousins use
/// linear scratch by gathering each node's row on demand.
MYMYR_HD u32 horizon_kin_count(const HorizonGraph& h, bool cousin)
{
    const KinView k = horizon_kin_view(h);
    kin_clear_stamps(k);
    u64 total = 0;
    for (u32 n = 0; n < k.nodes; ++n)
    {
        if (cousin && !k.far)
        {
            u32 size;
            kin_cousin_row(k, n, size);
            total += size;
            continue;
        }
        const u32 first = k.parent_offsets[n], last = k.parent_offsets[n + 1u];
        if (last == first)
            continue;
        if (last - first == 1u)
        {
            u32 size;
            const u32* list = kin_parent_list(k, cousin, k.parents[first], size);
            total += size - (kin_contains(list, size, n) ? 1u : 0u);
            continue;
        }
        for (u32 i = first; i < last; ++i)
        {
            u32 m;
            const u32* list = kin_parent_list(k, cousin, k.parents[i], m);
            for (u32 j = 0; j < m; ++j)
                if (list[j] != n && k.stamp[list[j]] != n + 1u)
                {
                    k.stamp[list[j]] = n + 1u;
                    ++total;
                }
        }
    }
    return total > 0xffffffffull ? 0xffffffffu : static_cast<u32>(total);
}

/// Calls out.put_pair(n, relative) for the tuples of a sibling or cousin cell, in order.
template<class Out>
MYMYR_HD void horizon_kin_emit(const HorizonGraph& h, bool cousin, Out& out)
{
    const KinView k = horizon_kin_view(h);
    kin_clear_stamps(k);
    for (u32 n = 0; n < k.nodes; ++n)
    {
        u32 size;
        const u32* list = cousin && !k.far ? kin_cousin_row(k, n, size) : kin_relatives(k, cousin, n, size);
        for (u32 i = 0; i < size; ++i)
            if (list[i] != n)
                out.put_pair(n, list[i]);
    }
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

/// Word `w` of the goal tuples a goal cell takes from its slot range at a node: the cell's mask word (positive or
/// negative literals, fluent or derived), for the satisfied and unsatisfied variants masked by the node's state word: a
/// positive literal holds iff its bit is set, a negative one iff it is clear. A cell of a subgoal level (1 to 3) takes
/// the mask of the row's layer of that level (none: no tuples). The placement policy filters each level, the goal
/// included, against the other listing levels. The row is the graph's goals and layers with the node's atoms and
/// derived words; unlike the flat graphs, whose satisfaction masks of the levels are worked out once for their one state, every node has its own.
MYMYR_HD u64 horizon_goal_word(const Cell& c, const GoalWords& g, const RowInput& in, u32 w,
                               RepeatedLiterals policy)
{
    const bool derived = goal_op_derived(c.op);
    const bool negative = goal_op_negative(c.op);
    const GoalMode mode = goal_op_mode(c.op);
    const u32 level = goal_op_level(c.op);
    const u64* mask = derived ? (negative ? g.derived_neg : g.derived_pos) : (negative ? g.neg : g.pos);
    const MaskLayers layers = in.layers ? in.layers->subgoals : MaskLayers{};
    if (level > layers.layers)
        return 0;
    const u64* base = derived ? (negative ? layers.derived_neg : layers.derived_pos) : (negative ? layers.neg : layers.pos);
    const u32 stride = derived ? layers.derived_words : layers.words;
    const u64 m = horizon_clip_word(placed_level_word(policy, base, stride, level_count(layers), level, mask, w),
                                    w, c.a, c.b);
    if (mode == GoalMode::Plain || mode == GoalMode::PlainTarget)
        return m;
    const u64* x = derived ? in.derived : in.atoms;
    const u64 xw = x ? x[w] : u64{0};
    const bool wants_set = (mode == GoalMode::Satisfied) != negative;
    return m & (wants_set ? xw : ~xw);
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
            return horizon_goal_word(flat, h.goal, horizon_node_input(h, node), w, h.v->repeated_literals);
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
            return changed & horizon_goal_word(flat, h.goal, h.row, w, h.v->repeated_literals);
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
        if (horizon_goal_word(flat, h.goal, h.row, w, h.v->repeated_literals) != 0)
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
    if (kind == HorizonKind::Sibling || kind == HorizonKind::Cousin)
        return h.kin ? horizon_kin_count(h, kind == HorizonKind::Cousin) : 0u;
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
        horizon_each_parent(h, [&](u32 src, u32 dst) { out.put_pair(src, dst); });
        return;
    }
    if (kind == HorizonKind::Sibling || kind == HorizonKind::Cousin)
    {
        if (h.kin)
            horizon_kin_emit(h, kind == HorizonKind::Cousin, out);
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
        if (n && horizon_over_auxiliary(v, c))
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
    bool nullary;  // a nullary predicate over the auxiliary entity: it is its only argument
    u32 objects;
    u32 tuples;  // the tuples written

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
        ++tuples;
    }
    MYMYR_HD void put_pair(u32 a, u32 b)
    {
        *dst++ = static_cast<I>(state(a));
        *dst++ = static_cast<I>(state(b));
        ++tuples;
    }
};

/// Writes the tuples of cell `c` to `dst` (the entity ids; `predicate` is the cell's virtual entity or k_no_entity) and
/// returns how many. A tuple of a nullary predicate holds the auxiliary entity, or no argument when the context gives
/// nullary tuples none.
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
                            horizon_over_auxiliary(*h.v, c),
                            kind == HorizonKind::Parent || kind == HorizonKind::Sibling || kind == HorizonKind::Cousin
                                ? 0u
                                : c.logical_arity,
                            0u};
    horizon_emit_cell(c, h, writer);
    return writer.tuples;
}
}  // namespace mymyr::encoding::detail
