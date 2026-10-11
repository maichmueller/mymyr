#include "mymyr/encoding/context.hpp"

#include "mymyr/core/bitset.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
using detail::Cell;
using detail::CellOp;

bool is_state_relation(const Relation& relation)
{
    const bool state_lane = relation.lane == Lane::Fluent || relation.lane == Lane::Static || relation.lane == Lane::Derived;
    return state_lane && relation.sign == 0 && !relation.level && !relation.derivation && !relation.state_anchored &&
           relation.predicate >= 0;
}

std::span<const i32> ints(const rl::ArrayBundle& bundle, const char* name)
{
    const rl::ArrayInfo* info = bundle.find(name);
    return {static_cast<const i32*>(bundle.data(*info)), info->elements()};
}

/// What one instance contributes to the arrays.
struct InstanceRows
{
    std::vector<Cell> cells;
    std::vector<i32> slot_args, derived_args, static_args;  // rows of max_arity objects, -1 padded
    u64 num_objects = 0, words = 0, fluent_slots = 0, derived_words = 0, derived_slots = 0;
    u64 fingerprint = 0;
};

/// The objects of every slot of `pred` / `padded`, as rows of max_arity entries.
void append_slot_rows(std::vector<i32>& to, std::span<const i32> pred, std::span<const i32> padded, u32 width,
                      std::span<const i32> arity, u32 max_arity)
{
    for (u64 slot = 0; slot < pred.size(); ++slot)
    {
        const u32 logical = static_cast<u32>(arity[static_cast<u32>(pred[slot])]);
        for (u32 k = 0; k < max_arity; ++k)
            to.push_back(k < logical ? padded[slot * width + k] : -1);
    }
}

InstanceRows build_instance(const rl::TaskTable& table, u32 index, const std::vector<i32>& state_relation,
                            const EncodingSchema& schema, u32 max_arity)
{
    const rl::TaskTable::Instance& instance = table.instance(index);
    const Task& task = *instance.task;
    const rl::ArrayBundle meta = rl::atom_metadata(task);
    const auto atom_pred = ints(meta, "atom_pred");
    const auto atom_padded = ints(meta, "atom_args_padded");
    const auto derived_pred = ints(meta, "derived_pred");
    const auto derived_padded = ints(meta, "derived_args_padded");
    const auto fluent_offsets = ints(meta, "pred_slot_offsets");
    const auto derived_offsets = ints(meta, "derived_pred_slot_offsets");
    const auto static_pred = ints(meta, "static_pred");
    const auto static_padded = ints(meta, "static_args_padded");
    const auto arity = ints(meta, "pred_arity");
    const auto kind = ints(meta, "pred_kind");
    const u32 predicates = static_cast<u32>(arity.size());
    auto width_of = [](std::span<const i32> pred, std::span<const i32> padded)
    { return static_cast<u32>(pred.empty() ? 0 : padded.size() / pred.size()); };
    const u32 fluent_width = width_of(atom_pred, atom_padded);
    const u32 derived_width = width_of(derived_pred, derived_padded);
    const u32 static_width = width_of(static_pred, static_padded);

    InstanceRows out;
    out.num_objects = instance.num_objects;
    out.words = instance.words;
    out.fluent_slots = task.atoms().fluent_slots();
    out.derived_slots = task.atoms().derived_slots();
    out.derived_words = bits::words_for(out.derived_slots);
    out.fingerprint = task.encoding_fingerprint();
    u32 static_rows = 0;
    for (u32 ip = 0; ip < predicates; ++ip)
    {
        const u32 tp = instance.pred_map[ip];
        if (tp >= table.num_domain_predicates() || state_relation[tp] < 0)
            continue;
        const u32 relation_id = static_cast<u32>(state_relation[tp]);
        const Relation& relation = schema.relation(relation_id);
        const u32 logical = static_cast<u32>(arity[ip]);
        const Lane lane = kind[ip] == rl::k_pred_static ? Lane::Static : kind[ip] == rl::k_pred_fluent ? Lane::Fluent : Lane::Derived;
        if (relation.lane != lane || relation.logical_arity != logical || relation.arity != std::max(logical, 1u))
            throw std::logic_error("mymyr: encoding: the relation " + relation.name + " does not match its predicate");
        Cell cell{relation_id, 0, 0, 0, relation.arity, logical};
        if (lane == Lane::Fluent)
        {
            cell.op = static_cast<u32>(CellOp::Fluent);
            cell.a = static_cast<u32>(fluent_offsets[ip]);
            cell.b = static_cast<u32>(fluent_offsets[ip + 1]);
            detail::check_slot_range(atom_pred, atom_padded, fluent_width, cell.a, cell.b, ip, logical, relation.name);
        }
        else if (lane == Lane::Derived)
        {
            cell.op = static_cast<u32>(CellOp::Derived);
            cell.a = static_cast<u32>(derived_offsets[ip]);
            cell.b = static_cast<u32>(derived_offsets[ip + 1]);
            detail::check_slot_range(derived_pred, derived_padded, derived_width, cell.a, cell.b, ip, logical, relation.name);
        }
        else
        {
            std::vector<std::vector<i32>> rows;
            for (u64 s = 0; s < static_pred.size(); ++s)
                if (static_cast<u32>(static_pred[s]) == ip)
                    rows.emplace_back(static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width),
                                      static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width + logical));
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            cell.op = static_cast<u32>(CellOp::Static);
            cell.a = static_rows;
            for (const std::vector<i32>& row : rows)
            {
                for (u32 k = 0; k < max_arity; ++k)
                    out.static_args.push_back(k < logical ? row[k] : -1);
                ++static_rows;
            }
            cell.b = static_rows;
        }
        if (cell.a != cell.b)
            out.cells.push_back(cell);
    }
    std::sort(out.cells.begin(), out.cells.end(), [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
    append_slot_rows(out.slot_args, atom_pred, atom_padded, fluent_width, arity, max_arity);
    append_slot_rows(out.derived_args, derived_pred, derived_padded, derived_width, arity, max_arity);
    return out;
}
}  // namespace

void detail::check_slot_range(std::span<const i32> pred, std::span<const i32> padded, u32 width, u32 begin, u32 end,
                              u32 predicate, u32 arity, const std::string& name)
{
    auto not_a_range = [&]
    { throw std::logic_error("mymyr: encoding: the atom slots of predicate " + name + " are not one ascending range"); };
    for (u32 slot = begin; slot < end; ++slot)
    {
        if (static_cast<u32>(pred[slot]) != predicate)
            not_a_range();
        if (slot > begin &&
            !std::lexicographical_compare(padded.begin() + static_cast<std::ptrdiff_t>((slot - 1) * u64{width}),
                                          padded.begin() + static_cast<std::ptrdiff_t>((slot - 1) * u64{width} + arity),
                                          padded.begin() + static_cast<std::ptrdiff_t>(slot * u64{width}),
                                          padded.begin() + static_cast<std::ptrdiff_t>(slot * u64{width} + arity)))
            not_a_range();
    }
}

std::shared_ptr<const EncodingContext> EncodingContext::create(rl::TaskTablePtr table,
                                                              std::shared_ptr<const EncodingSchema> schema)
{
    if (!table || !schema)
        throw std::invalid_argument("mymyr: encoding: a context needs a table and a schema");
    // The schema of the table's own domain. Relation ids follow the sorted names, so they agree with the passed schema
    // when the fingerprints are equal, but a relation's predicate id follows the table's declaration order, which the
    // fingerprint ignores: predicates are mapped to relations through this schema only.
    const std::shared_ptr<const EncodingSchema> own = EncodingSchema::create(*table, schema->options());
    if (own->fingerprint() != schema->fingerprint())
        throw std::invalid_argument("mymyr: encoding: the schema is not the schema of the table's domain");
    // The relation of each domain predicate's state atoms, and the widest of them.
    std::vector<i32> state_relation(table->num_domain_predicates(), -1);
    u32 max_arity = 1;
    for (u32 id = 0; id < own->num_relations(); ++id)
    {
        const Relation& relation = own->relation(id);
        if (!is_state_relation(relation))
            continue;
        state_relation.at(static_cast<u32>(relation.predicate)) = static_cast<i32>(id);
        max_arity = std::max(max_arity, relation.logical_arity);
    }
    std::vector<InstanceRows> instances;
    instances.reserve(table->size());
    for (u32 i = 0; i < table->size(); ++i)
        instances.push_back(build_instance(*table, i, state_relation, *schema, max_arity));

    auto context = std::shared_ptr<EncodingContext>(new EncodingContext());
    context->m_max_arity = max_arity;
    context->m_needs_derived.assign(instances.size(), 0);
    std::vector<u64> inst;
    std::vector<u32> cells;
    std::vector<i32> slot_args, derived_args, static_args;
    for (u32 i = 0; i < instances.size(); ++i)
    {
        const InstanceRows& rows = instances[i];
        u64 columns[detail::k_instance_columns] = {};
        columns[detail::k_num_objects] = rows.num_objects;
        columns[detail::k_words] = rows.words;
        columns[detail::k_fluent_slots] = rows.fluent_slots;
        columns[detail::k_derived_words] = rows.derived_words;
        columns[detail::k_derived_slots] = rows.derived_slots;
        columns[detail::k_num_cells] = rows.cells.size();
        columns[detail::k_cell_offset] = cells.size() / 6;
        columns[detail::k_slot_args_offset] = slot_args.size() / max_arity;
        columns[detail::k_derived_args_offset] = derived_args.size() / max_arity;
        columns[detail::k_static_args_offset] = static_args.size() / max_arity;
        columns[detail::k_encoding_fingerprint] = rows.fingerprint;
        inst.insert(inst.end(), std::begin(columns), std::end(columns));
        for (const Cell& cell : rows.cells)
        {
            for (u32 value : {cell.relation, cell.op, cell.a, cell.b, cell.arity, cell.logical_arity})
                cells.push_back(value);
            if (cell.op == static_cast<u32>(CellOp::Derived))
            {
                context->m_needs_derived[i] = 1;
                context->m_max_derived_words = std::max(context->m_max_derived_words, static_cast<u32>(rows.derived_words));
            }
        }
        slot_args.insert(slot_args.end(), rows.slot_args.begin(), rows.slot_args.end());
        derived_args.insert(derived_args.end(), rows.derived_args.begin(), rows.derived_args.end());
        static_args.insert(static_args.end(), rows.static_args.begin(), rows.static_args.end());
    }
    auto shape = [&](const std::vector<i32>& rows) { return std::vector<i64>{static_cast<i64>(rows.size() / max_arity), max_arity}; };
    rl::ArrayBundle::Builder builder;
    builder.add("inst", rl::DType::U64, {static_cast<i64>(instances.size()), detail::k_instance_columns}, inst);
    builder.add("cell", rl::DType::U32, {static_cast<i64>(cells.size() / 6), 6}, cells);
    builder.add("slot_args", rl::DType::I32, shape(slot_args), slot_args);
    builder.add("derived_args", rl::DType::I32, shape(derived_args), derived_args);
    builder.add("static_args", rl::DType::I32, shape(static_args), static_args);
    context->m_arrays = builder.build();
    context->m_table = std::move(table);
    context->m_schema = std::move(schema);
    return context;
}

detail::ContextView EncodingContext::view(const std::byte* block) const noexcept
{
    const std::byte* base = block ? block : m_arrays.block();
    auto at = [&](const char* name) { return base + m_arrays.find(name)->offset; };
    detail::ContextView v;
    v.inst = reinterpret_cast<const u64*>(at("inst"));
    v.num_instances = m_table->size();
    v.cells = reinterpret_cast<const Cell*>(at("cell"));
    v.slot_args = reinterpret_cast<const i32*>(at("slot_args"));
    v.derived_args = reinterpret_cast<const i32*>(at("derived_args"));
    v.static_args = reinterpret_cast<const i32*>(at("static_args"));
    v.max_arity = m_max_arity;
    v.num_relations = m_schema->num_relations();
    return v;
}
}
