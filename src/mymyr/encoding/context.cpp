#include "mymyr/encoding/context.hpp"

#include "mymyr/core/bitset.hpp"

#include <algorithm>
#include <map>
#include <optional>
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

/// The goal relations of the schema: for every domain predicate, the relation of its goal literals of each sign
/// (positive, negative), as the plain, satisfied and unsatisfied variants (-1: the schema has none).
struct GoalRelations
{
    std::vector<i32> id;     // [predicate, sign, variant]
    std::vector<u8> target;  // [relation]: the plain relation carries a goal target slot

    [[nodiscard]] i32 at(u32 predicate, u32 sign, u32 variant) const { return id[(u64{predicate} * 2 + sign) * 3 + variant]; }
};

/// Bits of the per-instance flags that tell whether the derived atoms of a row are needed.
constexpr u8 k_needs_state = 1;          // the instance has a derived state cell
constexpr u8 k_needs_instance_goal = 2;  // its own goal has a derived literal that a satisfied / unsatisfied cell lists
constexpr u8 k_needs_row_goal = 4;       // it has a derived cell of satisfied / unsatisfied goal literals

/// What one instance contributes to the arrays.
struct InstanceRows
{
    std::vector<Cell> cells;
    std::vector<i32> slot_args, derived_args, static_args;  // rows of max_arity objects, -1 padded
    std::vector<u64> goal_words;                            // goal masks: pos, neg (words each), derived pos, neg
    std::vector<i32> goal_args;                             // the constant goal tuples, rows of max_arity objects
    u64 num_objects = 0, words = 0, fluent_slots = 0, derived_words = 0, derived_slots = 0;
    u64 fingerprint = 0;
    u64 goal_complete = 1;
    u8 needs = 0;
};

/// A goal literal whose truth no state changes: over a static predicate, or over an atom that has no slot.
struct ConstantLiteral
{
    std::vector<i32> objects;
    bool holds;
};

bool before(const std::vector<i32>& a, const std::vector<i32>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

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

/// The slot of the atom `objects` among the slots [begin, end) of `padded` (rows of `width` objects, ascending), if any.
std::optional<u32> find_slot(std::span<const i32> padded, u32 width, u32 begin, u32 end, const std::vector<i32>& objects)
{
    const u32 arity = static_cast<u32>(objects.size());
    u32 lo = begin, hi = end;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2;
        const i32* row = padded.data() + static_cast<u64>(mid) * width;
        if (std::lexicographical_compare(row, row + arity, objects.begin(), objects.end()))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < end && std::equal(objects.begin(), objects.end(), padded.data() + static_cast<u64>(lo) * width))
        return lo;
    return std::nullopt;
}

InstanceRows build_instance(const rl::TaskTable& table, u32 index, const std::vector<i32>& state_relation,
                            const GoalRelations& goal_relations, const EncodingSchema& schema, u32 max_arity)
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
    for (const Cell& cell : out.cells)
        if (cell.op == static_cast<u32>(CellOp::Derived))
            out.needs |= k_needs_state;

    // The goal: masks over the slots of the goal's atoms, and the literals whose truth is fixed as constant tuples.
    out.goal_complete = task.encoding_goal_complete() ? 1 : 0;
    const u32 W = static_cast<u32>(out.words), D = static_cast<u32>(out.derived_words);
    out.goal_words.assign(u64{2} * W + u64{2} * D, 0);
    u64* masks[2][2] = {{out.goal_words.data(), out.goal_words.data() + W},
                        {out.goal_words.data() + 2 * u64{W}, out.goal_words.data() + 2 * u64{W} + D}};  // [derived][sign]
    std::map<std::pair<u32, u32>, std::vector<ConstantLiteral>> constants;  // (predicate, sign)
    std::vector<std::vector<std::vector<i32>>> static_atoms;
    auto static_holds = [&](u32 ip, const std::vector<i32>& objects)
    {
        if (static_atoms.empty())
        {
            static_atoms.resize(predicates);
            for (u64 s = 0; s < static_pred.size(); ++s)
            {
                const u32 p = static_cast<u32>(static_pred[s]);
                static_atoms[p].emplace_back(static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width),
                                            static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width + arity[p]));
            }
            for (auto& rows : static_atoms)
            {
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            }
        }
        return std::binary_search(static_atoms[ip].begin(), static_atoms[ip].end(), objects);
    };
    const formalism::TaskData& data = task.data();
    for (const formalism::Literal& literal : task.encoding_goal())
    {
        const u32 ip = literal.pred.v;
        if (ip >= predicates || instance.pred_map[ip] >= table.num_domain_predicates())
            continue;
        const u32 tp = instance.pred_map[ip];
        const u32 sign = literal.positive ? 0 : 1;
        if (goal_relations.at(tp, sign, 0) < 0 && goal_relations.at(tp, sign, 1) < 0 && goal_relations.at(tp, sign, 2) < 0)
            continue;  // the schema has no goal relation of this predicate (nullary and equality can be left out)
        std::vector<i32> objects;
        for (formalism::Term term : data.terms_of(literal))
            objects.push_back(static_cast<i32>(formalism::term_object(term).v));
        const bool derived = kind[ip] == rl::k_pred_derived;
        bool holds = false, slotted = false;
        if (kind[ip] == rl::k_pred_static)
            holds = static_holds(ip, objects) == literal.positive;
        else
        {
            const auto slot = derived ? find_slot(derived_padded, derived_width, static_cast<u32>(derived_offsets[ip]),
                                                  static_cast<u32>(derived_offsets[ip + 1]), objects)
                                      : find_slot(atom_padded, fluent_width, static_cast<u32>(fluent_offsets[ip]),
                                                  static_cast<u32>(fluent_offsets[ip + 1]), objects);
            if (slot)
            {
                masks[derived ? 1 : 0][sign][*slot >> 6] |= u64{1} << (*slot & 63u);
                slotted = true;
                if (derived && (goal_relations.at(tp, sign, 1) >= 0 || goal_relations.at(tp, sign, 2) >= 0))
                    out.needs |= k_needs_instance_goal;
            }
            else
                holds = !literal.positive;  // an atom no state has is false
        }
        if (!slotted)
            constants[{ip, sign}].push_back({std::move(objects), holds});
    }
    for (auto& entry : constants)
    {
        auto& rows = entry.second;
        std::sort(rows.begin(), rows.end(),
                  [](const ConstantLiteral& x, const ConstantLiteral& y) { return before(x.objects, y.objects); });
        rows.erase(std::unique(rows.begin(), rows.end(),
                               [](const ConstantLiteral& x, const ConstantLiteral& y) { return x.objects == y.objects; }),
                   rows.end());
    }
    u32 goal_rows = 0;
    for (u32 ip = 0; ip < predicates; ++ip)
    {
        const u32 tp = instance.pred_map[ip];
        if (tp >= table.num_domain_predicates())
            continue;
        const bool derived = kind[ip] == rl::k_pred_derived, fixed = kind[ip] == rl::k_pred_static;
        const u32 logical = static_cast<u32>(arity[ip]);
        for (u32 sign = 0; sign < 2; ++sign)
        {
            const auto found = constants.find({ip, sign});
            for (u32 variant = 0; variant < 3; ++variant)
            {
                const i32 id = goal_relations.at(tp, sign, variant);
                if (id < 0)
                    continue;
                const Relation& relation = schema.relation(static_cast<u32>(id));
                Cell cell{static_cast<u32>(id), 0, 0, 0, relation.arity, logical};
                if (!fixed)
                {
                    cell.a = static_cast<u32>(derived ? derived_offsets[ip] : fluent_offsets[ip]);
                    cell.b = static_cast<u32>(derived ? derived_offsets[ip + 1] : fluent_offsets[ip + 1]);
                }
                cell.c = goal_rows;
                if (found != constants.end())
                    for (const ConstantLiteral& row : found->second)
                    {
                        if ((variant == 1 && !row.holds) || (variant == 2 && row.holds))
                            continue;
                        for (u32 k = 0; k < max_arity; ++k)
                            out.goal_args.push_back(k < logical ? row.objects[k] : -1);
                        ++goal_rows;
                    }
                cell.d = goal_rows;
                if (cell.a == cell.b && cell.c == cell.d)
                    continue;
                const detail::GoalMode mode = variant == 0 ? (goal_relations.target[static_cast<u32>(id)] ? detail::GoalMode::PlainTarget
                                                                                                  : detail::GoalMode::Plain)
                                      : variant == 1 ? detail::GoalMode::Satisfied
                                                     : detail::GoalMode::Unsatisfied;
                cell.op = detail::goal_op(derived, sign == 1, mode);
                if (derived && variant > 0 && cell.a != cell.b)
                    out.needs |= k_needs_row_goal;
                out.cells.push_back(cell);
            }
        }
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
    GoalRelations goal_relations;
    goal_relations.id.assign(u64{table->num_domain_predicates()} * 6, -1);
    goal_relations.target.assign(own->num_relations(), 0);
    for (u32 id = 0; id < own->num_relations(); ++id)
    {
        const Relation& relation = own->relation(id);
        const bool plain = relation.lane == Lane::Goal && !relation.derivation;
        const bool derivation =
            relation.lane == Lane::GoalDerivation &&
            (relation.derivation == GoalDerivation::Satisfied || relation.derivation == GoalDerivation::Unsatisfied);
        if (relation.level.value_or(1) != 0 || !(plain || derivation))
            continue;
        const u32 variant = plain ? 0 : relation.derivation == GoalDerivation::Satisfied ? 1 : 2;
        goal_relations.id[(u64{static_cast<u32>(relation.predicate)} * 2 + (relation.sign > 0 ? 0 : 1)) * 3 + variant] =
            static_cast<i32>(id);
        if (plain && own->slot_roles(id).front() == SlotRole::GoalTarget)
            goal_relations.target[id] = 1;
    }
    std::vector<InstanceRows> instances;
    instances.reserve(table->size());
    for (u32 i = 0; i < table->size(); ++i)
        instances.push_back(build_instance(*table, i, state_relation, goal_relations, *schema, max_arity));

    auto context = std::shared_ptr<EncodingContext>(new EncodingContext());
    context->m_max_arity = max_arity;
    context->m_needs_derived.assign(instances.size(), 0);
    std::vector<u64> inst;
    std::vector<u32> cells;
    std::vector<i32> slot_args, derived_args, static_args, goal_args;
    std::vector<u64> goal_words;
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
        columns[detail::k_cell_offset] = cells.size() / 8;
        columns[detail::k_slot_args_offset] = slot_args.size() / max_arity;
        columns[detail::k_derived_args_offset] = derived_args.size() / max_arity;
        columns[detail::k_static_args_offset] = static_args.size() / max_arity;
        columns[detail::k_encoding_fingerprint] = rows.fingerprint;
        columns[detail::k_goal_offset] = goal_words.size();
        columns[detail::k_goal_args_offset] = goal_args.size() / max_arity;
        columns[detail::k_goal_complete] = rows.goal_complete;
        inst.insert(inst.end(), std::begin(columns), std::end(columns));
        for (const Cell& cell : rows.cells)
        {
            for (u32 value : {cell.relation, cell.op, cell.a, cell.b, cell.arity, cell.logical_arity, cell.c, cell.d})
                cells.push_back(value);
        }
        context->m_needs_derived[i] = rows.needs;
        if (rows.needs != 0)
            context->m_max_derived_words = std::max(context->m_max_derived_words, static_cast<u32>(rows.derived_words));
        slot_args.insert(slot_args.end(), rows.slot_args.begin(), rows.slot_args.end());
        derived_args.insert(derived_args.end(), rows.derived_args.begin(), rows.derived_args.end());
        static_args.insert(static_args.end(), rows.static_args.begin(), rows.static_args.end());
        goal_words.insert(goal_words.end(), rows.goal_words.begin(), rows.goal_words.end());
        goal_args.insert(goal_args.end(), rows.goal_args.begin(), rows.goal_args.end());
    }
    auto shape = [&](const std::vector<i32>& rows) { return std::vector<i64>{static_cast<i64>(rows.size() / max_arity), max_arity}; };
    rl::ArrayBundle::Builder builder;
    builder.add("inst", rl::DType::U64, {static_cast<i64>(instances.size()), detail::k_instance_columns}, inst);
    builder.add("cell", rl::DType::U32, {static_cast<i64>(cells.size() / 8), 8}, cells);
    builder.add("slot_args", rl::DType::I32, shape(slot_args), slot_args);
    builder.add("derived_args", rl::DType::I32, shape(derived_args), derived_args);
    builder.add("static_args", rl::DType::I32, shape(static_args), static_args);
    builder.add("goal_words", rl::DType::U64, {static_cast<i64>(goal_words.size())}, goal_words);
    builder.add("goal_args", rl::DType::I32, shape(goal_args), goal_args);
    context->m_arrays = builder.build();
    context->m_table = std::move(table);
    context->m_schema = std::move(schema);
    return context;
}

detail::ContextView EncodingContext::view(const std::byte* block) const noexcept
{
    const std::byte* base = block ? block : m_arrays.block();
    // the arrays are in the order create() adds them: inst, cell, slot_args, derived_args, static_args, goal_words,
    // goal_args
    const auto& arrays = m_arrays.arrays();
    auto at = [&](usize k) { return base + arrays[k].offset; };
    detail::ContextView v;
    v.inst = reinterpret_cast<const u64*>(at(0));
    v.num_instances = m_table->size();
    v.cells = reinterpret_cast<const Cell*>(at(1));
    v.slot_args = reinterpret_cast<const i32*>(at(2));
    v.derived_args = reinterpret_cast<const i32*>(at(3));
    v.static_args = reinterpret_cast<const i32*>(at(4));
    v.goal_words = reinterpret_cast<const u64*>(at(5));
    v.goal_args = reinterpret_cast<const i32*>(at(6));
    v.max_arity = m_max_arity;
    v.num_relations = m_schema->num_relations();
    return v;
}
}
