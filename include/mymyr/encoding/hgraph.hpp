#pragma once

// The HGraph view of a relation-major batch (encoder.hpp): the batch as one heterogeneous graph, without copying the
// tuples. It has a node type for every relation of the schema (its nodes are the relation's tuples, the atoms) and the
// node type `_symbol_`, whose nodes are the entities of the graphs. An atom node is joined to each entity that its tuple
// holds, by one edge type per argument position: the edge type (r, k, `_symbol_`) has an edge from tuple i of relation
// r to the entity at position k of that tuple, for every tuple of r in the batch, graph after graph.
//
// Numbering. The nodes of `_symbol_` are numbered by the batch's entity ids (batch-global: the entities of graph g are
// [entity_offsets[g], entity_offsets[g + 1]), objects first), so an edge's `_symbol_` end is the entity id stored in the
// batch. The nodes of relation r are numbered 0 .. n_r - 1 across the batch: tuple j of graph g is node
// ptr[g] + j with ptr = relation_ptr(r), and ptr[G] = n_r. The edges of the edge type (r, k, `_symbol_`) are numbered by
// their tuple, so the edge index of an edge type has two rows:
//   row 0  the tuple numbers 0 .. n_r - 1 (arange); written on request by edge_index_0
//   row 1  the entities at position k of the tuples: a strided view of the batch's relation_args (stride: the arity of
//          r), nothing is copied
// Relation-major packing is what makes row 1 one strided run: all tuples of a relation are contiguous across the graphs.
// Positions are those of the relation's slot roles (EncodingSchema::slot_roles): the leading slots (state, action,
// targets, history), the predicate slot, then the arguments. A nullary relation has one position, its auxiliary slot, so
// its edge type (r, 0, `_symbol_`) leads to the graph's auxiliary entity. A relation of arity 0 (a nullary state
// relation under Nullary::Arity0) has a node type and no edge type: its atoms are nodes without edges.
//
// The view works on the batches of the horizon, rooted-horizon and transition encoders as it does on those of the
// single-state encoder: their relations (the action, parent, sibling and cousin relations among them) and their state
// entities are node types, edge types and `_symbol_` nodes like any other.
//
// The view keeps the batch alive. It refuses a batch that is graph-major, one encoded with predicate virtual nodes (the
// node type already is the predicate) and one that stores its static tuples once per instance instead of in every graph.
// It reads the offsets of the batch only; a batch that violates the contract of encoder.hpp (see validate.hpp) gives
// views that violate it too.

#include "mymyr/encoding/schema.hpp"
#include "mymyr/rl/task_arrays.hpp"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace mymyr::encoding
{
/// A read-only row of indices inside a batch: `size` entries of the batch's index type, entry i at element i * stride
/// from `data`. It does not own the memory.
class IndexRow
{
public:
    IndexRow() = default;
    IndexRow(const void* data, u64 size, u64 stride, bool wide) noexcept
        : m_data(data), m_size(size), m_stride(stride), m_wide(wide)
    {
    }
    [[nodiscard]] const void* data() const noexcept { return m_data; }
    [[nodiscard]] u64 size() const noexcept { return m_size; }
    /// Elements (of the index type) from one entry to the next.
    [[nodiscard]] u64 stride() const noexcept { return m_stride; }
    /// Whether the entries are int64 (else int32).
    [[nodiscard]] bool wide() const noexcept { return m_wide; }
    [[nodiscard]] i64 operator[](u64 i) const noexcept
    {
        return m_wide ? static_cast<const i64*>(m_data)[i * m_stride] : static_cast<const i32*>(m_data)[i * m_stride];
    }

private:
    const void* m_data = nullptr;
    u64 m_size = 0;
    u64 m_stride = 1;
    bool m_wide = false;
};

/// An edge type (relation, slot, `_symbol_`): the relation and the position in its tuples.
struct HGraphEdgeType
{
    u32 relation = 0;
    u32 slot = 0;
    [[nodiscard]] friend bool operator==(const HGraphEdgeType&, const HGraphEdgeType&) noexcept = default;
};

/// The name of the node type of the entities.
inline constexpr std::string_view k_symbol_node_type = "_symbol_";

class HGraphView
{
public:
    /// The view of `batch`, which must have been encoded for `schema`. Throws std::invalid_argument ("mymyr: encoding:
    /// ...") for a graph-major batch, for a schema with predicate virtual nodes, for a batch with shared static
    /// storage, and for a batch that is not one of `schema` (its fingerprint or relation count differs) or lacks an
    /// array of the contract. Linear in the number of relations; does not read the tuples.
    [[nodiscard]] static HGraphView of(std::shared_ptr<const rl::ArrayBundle> batch,
                                       std::shared_ptr<const EncodingSchema> schema);

    [[nodiscard]] const std::shared_ptr<const rl::ArrayBundle>& batch() const noexcept { return m_batch; }
    [[nodiscard]] const std::shared_ptr<const EncodingSchema>& schema() const noexcept { return m_schema; }
    [[nodiscard]] u64 num_graphs() const noexcept { return m_graphs; }
    /// Whether the batch's indices are int64 (else int32).
    [[nodiscard]] bool wide() const noexcept { return m_wide; }

    // ---------------------------------------------------------------------------------------------- node types
    /// The node types: one per relation of the schema, in relation order, then `_symbol_`.
    [[nodiscard]] u32 num_node_types() const noexcept { return m_relations + 1; }
    /// The node type `_symbol_`.
    [[nodiscard]] u32 symbol_type() const noexcept { return m_relations; }
    /// The name of a node type: the relation's name, or `_symbol_`. Throws std::out_of_range for a type out of range.
    [[nodiscard]] std::string_view node_type_name(u32 type) const;
    /// The nodes of a type in the whole batch: the tuples of the relation (all graphs), or the entities.
    [[nodiscard]] u64 num_nodes(u32 type) const;
    /// The `_symbol_` nodes of each graph: entity_offsets itself, no copy. G + 1 entries.
    [[nodiscard]] IndexRow symbol_ptr() const;
    /// Writes the node ptr of relation `relation` into `out`, G + 1 entries of the index type: graph g's tuples are the
    /// nodes [ptr[g], ptr[g + 1]) of the type, numbered from 0 across the batch.
    void relation_ptr(u32 relation, void* out) const;

    // ---------------------------------------------------------------------------------------------- edge types
    /// The edge types: for every relation in relation order, one per position of its tuples.
    [[nodiscard]] u64 num_edge_types() const noexcept { return m_edge_begin.back(); }
    /// The edge type with this number. Throws std::out_of_range.
    [[nodiscard]] HGraphEdgeType edge_type(u64 index) const;
    /// The number of the edge type (relation, slot), or none if the relation has no such position.
    [[nodiscard]] std::optional<u64> find_edge_type(u32 relation, u32 slot) const noexcept;
    /// The edges of an edge type in the whole batch: the tuples of its relation.
    [[nodiscard]] u64 num_edges(u64 edge_type) const;
    /// Row 1 of an edge type: the entity at the position of each tuple, a strided view of the batch's relation_args.
    [[nodiscard]] IndexRow edge_index_1(u64 edge_type) const;
    /// Writes row 0 of an edge type into `out`: the tuple numbers 0 .. num_edges - 1, of the index type.
    void edge_index_0(u64 edge_type, void* out) const;

    // ---------------------------------------------------------------------------------------------- entity feature
    /// The entities of the batch (the `_symbol_` nodes).
    [[nodiscard]] u64 num_entities() const noexcept { return m_entities; }
    /// The EntityRole of each entity, as stored in the batch (num_entities() entries), or null if the batch's entity
    /// arrays are implicit; write_entity_roles then makes them.
    [[nodiscard]] const u8* stored_entity_roles() const noexcept { return m_entity_role; }
    /// Writes the role of each entity (num_entities() entries) into `out`.
    void write_entity_roles(u8* out) const;

private:
    HGraphView() = default;
    std::shared_ptr<const rl::ArrayBundle> m_batch;
    std::shared_ptr<const EncodingSchema> m_schema;
    const void* m_relation_offsets = nullptr;
    const void* m_relation_arg_offsets = nullptr;
    const void* m_relation_args = nullptr;
    const void* m_entity_offsets = nullptr;
    const void* m_num_objects = nullptr;
    const u8* m_entity_role = nullptr;
    std::vector<u64> m_edge_begin;  // the first edge type of each relation, then the number of edge types
    u64 m_graphs = 0;
    u64 m_entities = 0;
    u32 m_relations = 0;
    bool m_wide = false;
};
}  // namespace mymyr::encoding
