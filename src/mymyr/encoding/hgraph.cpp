#include "mymyr/encoding/hgraph.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("mymyr: encoding: " + what); }

i64 at(const void* data, bool wide, u64 i) noexcept
{
    return wide ? static_cast<const i64*>(data)[i] : static_cast<i64>(static_cast<const i32*>(data)[i]);
}

void put(void* data, bool wide, u64 i, i64 value) noexcept
{
    if (wide)
        static_cast<i64*>(data)[i] = value;
    else
        static_cast<i32*>(data)[i] = static_cast<i32>(value);
}

const rl::ArrayInfo& required(const rl::ArrayBundle& batch, const char* name)
{
    const rl::ArrayInfo* info = batch.find(name);
    if (!info)
        fail(std::string("the HGraph view needs the array ") + name + ", which the batch lacks");
    return *info;
}
}  // namespace

HGraphView HGraphView::of(std::shared_ptr<const rl::ArrayBundle> batch, std::shared_ptr<const EncodingSchema> schema)
{
    if (!batch || !schema)
        throw std::invalid_argument("mymyr: encoding: the HGraph view needs a batch and its schema");
    if (batch->scalar("layout") != 1)
        fail("the HGraph view needs a relation-major batch (layout='relation'), because it views the tuples of a "
             "relation as one strided run of arguments; this batch is graph-major");
    if (schema->options().predicate_virtual_nodes)
        fail("the HGraph view does not support predicate virtual nodes: every node type already is a predicate; encode "
             "without predicate_virtual_nodes");
    if (batch->scalar("statics", 0) != 0)
        fail("the HGraph view does not support shared static storage: it needs the static tuples of every graph in the "
             "batch; encode with the static tuples repeated per graph");
    if (batch->scalar("schema_fingerprint", 0) != std::bit_cast<i64>(schema->fingerprint()))
        fail("the batch was not encoded for this schema (the fingerprints differ)");
    const i64 graphs = batch->scalar("num_graphs");
    if (graphs < 0)
        fail("the batch has no num_graphs");
    const u64 R = schema->num_relations();
    if (batch->scalar("num_relations") != static_cast<i64>(R))
        fail("the batch has a different number of relations than the schema");

    HGraphView view;
    view.m_graphs = static_cast<u64>(graphs);
    view.m_relations = static_cast<u32>(R);
    view.m_wide = batch->scalar("index_dtype") == 1;
    const rl::ArrayInfo& relation_offsets = required(*batch, "relation_offsets");
    const rl::ArrayInfo& arg_offsets = required(*batch, "relation_arg_offsets");
    const rl::ArrayInfo& relation_args = required(*batch, "relation_args");
    const rl::ArrayInfo& entity_offsets = required(*batch, "entity_offsets");
    const rl::ArrayInfo& num_objects = required(*batch, "num_objects");
    const u64 cells = view.m_graphs * R;
    if (relation_offsets.elements() != cells + 1 || arg_offsets.elements() != cells + 1)
        fail("the offsets of the batch do not match its graphs and relations");
    if (entity_offsets.elements() != view.m_graphs + 1)
        fail("entity_offsets does not match the graphs of the batch");
    view.m_relation_offsets = batch->data(relation_offsets);
    view.m_relation_arg_offsets = batch->data(arg_offsets);
    view.m_relation_args = batch->data(relation_args);
    view.m_entity_offsets = batch->data(entity_offsets);
    view.m_num_objects = batch->data(num_objects);
    view.m_entities = static_cast<u64>(at(view.m_entity_offsets, view.m_wide, view.m_graphs));
    if (!schema->implicit_entity_arrays())
    {
        const rl::ArrayInfo& role = required(*batch, "entity_role");
        if (role.elements() < view.m_entities)
            fail("entity_role is shorter than the entities of the batch");
        view.m_entity_role = static_cast<const u8*>(batch->data(role));
    }
    view.m_edge_begin.reserve(R + 1);
    u64 edge_types = 0;
    for (const Relation& relation : schema->relations())
    {
        view.m_edge_begin.push_back(edge_types);
        edge_types += relation.arity;
    }
    view.m_edge_begin.push_back(edge_types);
    view.m_batch = std::move(batch);
    view.m_schema = std::move(schema);
    return view;
}

std::string_view HGraphView::node_type_name(u32 type) const
{
    if (type == m_relations)
        return k_symbol_node_type;
    return m_schema->relation(type).name;  // throws std::out_of_range
}

u64 HGraphView::num_nodes(u32 type) const
{
    if (type == m_relations)
        return m_entities;
    if (type > m_relations)
        throw std::out_of_range("mymyr: encoding: no such node type");
    const u64 first = static_cast<u64>(type) * m_graphs;
    return static_cast<u64>(at(m_relation_offsets, m_wide, first + m_graphs) - at(m_relation_offsets, m_wide, first));
}

IndexRow HGraphView::symbol_ptr() const { return {m_entity_offsets, m_graphs + 1, 1, m_wide}; }

void HGraphView::relation_ptr(u32 relation, void* out) const
{
    if (relation >= m_relations)
        throw std::out_of_range("mymyr: encoding: no such relation");
    const u64 first = static_cast<u64>(relation) * m_graphs;
    const i64 base = at(m_relation_offsets, m_wide, first);
    for (u64 g = 0; g <= m_graphs; ++g)
        put(out, m_wide, g, at(m_relation_offsets, m_wide, first + g) - base);
}

HGraphEdgeType HGraphView::edge_type(u64 index) const
{
    if (index >= num_edge_types())
        throw std::out_of_range("mymyr: encoding: no such edge type");
    // the relation whose edge types contain `index`: the last one that begins at or before it (relations of arity 0
    // begin where the next one does, so the search takes the last of equal beginnings)
    u32 low = 0, high = m_relations;
    while (high - low > 1)
    {
        const u32 mid = low + (high - low) / 2;
        (m_edge_begin[mid] <= index ? low : high) = mid;
    }
    return {low, static_cast<u32>(index - m_edge_begin[low])};
}

std::optional<u64> HGraphView::find_edge_type(u32 relation, u32 slot) const noexcept
{
    if (relation >= m_relations || slot >= m_edge_begin[relation + 1] - m_edge_begin[relation])
        return std::nullopt;
    return m_edge_begin[relation] + slot;
}

u64 HGraphView::num_edges(u64 edge_type) const { return num_nodes(this->edge_type(edge_type).relation); }

IndexRow HGraphView::edge_index_1(u64 edge_type) const
{
    const HGraphEdgeType type = this->edge_type(edge_type);
    const u64 first = static_cast<u64>(type.relation) * m_graphs;
    const u64 arity = m_schema->relation(type.relation).arity;
    const u64 start = static_cast<u64>(at(m_relation_arg_offsets, m_wide, first)) + type.slot;
    const void* data = m_wide ? static_cast<const void*>(static_cast<const i64*>(m_relation_args) + start)
                              : static_cast<const void*>(static_cast<const i32*>(m_relation_args) + start);
    return {data, num_nodes(type.relation), arity, m_wide};
}

void HGraphView::edge_index_0(u64 edge_type, void* out) const
{
    const u64 n = num_edges(edge_type);
    if (m_wide)
        for (u64 i = 0; i < n; ++i)
            static_cast<i64*>(out)[i] = static_cast<i64>(i);
    else
        for (u64 i = 0; i < n; ++i)
            static_cast<i32*>(out)[i] = static_cast<i32>(i);
}

void HGraphView::write_entity_roles(u8* out) const
{
    if (m_entity_role)
    {
        std::copy_n(m_entity_role, m_entities, out);
        return;
    }
    // implicit entity arrays: the objects, then the auxiliary entity
    for (u64 g = 0; g < m_graphs; ++g)
    {
        const u64 first = static_cast<u64>(at(m_entity_offsets, m_wide, g));
        const u64 last = static_cast<u64>(at(m_entity_offsets, m_wide, g + 1));
        const u64 objects = static_cast<u64>(at(m_num_objects, m_wide, g));
        for (u64 e = first; e < last; ++e)
            out[e] = static_cast<u8>(e - first < objects ? EntityRole::Object : EntityRole::Auxiliary);
    }
}
}  // namespace mymyr::encoding
