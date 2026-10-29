#include "mymyr/encoding/literals.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
void check_arrays(const LiteralArrays& literals)
{
    if (literals.objects.size() != literals.predicates.size() * u64{literals.width})
        throw std::invalid_argument("mymyr: encoding: the objects of the literals must have " +
                                    std::to_string(literals.width) + " entries per literal");
}

/// Why literal i cannot be resolved (its kind is Invalid).
std::string invalid_reason(const EncodingContext& context, i64 instance, i64 predicate, std::span<const i32> objects)
{
    const rl::TaskTable& table = context.table();
    if (instance < 0 || instance >= table.size())
        return "its task id " + std::to_string(instance) + " is not in [0, " + std::to_string(table.size()) + ")";
    if (predicate < 0 || predicate >= table.num_domain_predicates())
        return "its predicate " + std::to_string(predicate) + " is not a domain predicate of the table (ids [0, " +
               std::to_string(table.num_domain_predicates()) + "))";
    const detail::ContextView v = context.view();
    const u32* r = v.literal_ranges + (static_cast<u64>(instance) * v.num_predicates + static_cast<u64>(predicate)) *
                                          detail::k_literal_columns;
    const std::string& name = table.predicate_names()[static_cast<u64>(predicate)];
    if (r[detail::k_literal_predicate_kind] == detail::k_literal_absent_predicate)
        return "its task has no predicate " + name;
    const u32 arity = r[detail::k_literal_arity];
    if (arity > objects.size())
        return "the predicate " + name + " takes " + std::to_string(arity) + " objects, the literals have room for " +
               std::to_string(objects.size());
    const u32 num_objects = table.instance(static_cast<u32>(instance)).num_objects;
    for (u32 k = 0; k < objects.size(); ++k)
    {
        if (k < arity && (objects[k] < 0 || static_cast<u32>(objects[k]) >= num_objects))
            return "its object " + std::to_string(objects[k]) + " is not in [0, " + std::to_string(num_objects) + ")";
        if (k >= arity && objects[k] != -1)
            return "the predicate " + name + " takes " + std::to_string(arity) + " objects, the literal has more";
    }
    return "it is out of range";
}
}  // namespace

std::string literal_text(const EncodingContext& context, i64 instance, i64 predicate, std::span<const i32> objects,
                         bool positive)
{
    const rl::TaskTable& table = context.table();
    const bool known_instance = instance >= 0 && instance < table.size();
    std::string atom = "(";
    atom += predicate >= 0 && predicate < static_cast<i64>(table.predicate_names().size())
                ? table.predicate_names()[static_cast<u64>(predicate)]
                : std::to_string(predicate);
    u32 arity = static_cast<u32>(objects.size());
    if (predicate >= 0 && predicate < table.num_domain_predicates())
        arity = std::min(arity, table.predicate_arities()[static_cast<u64>(predicate)]);
    for (u32 k = 0; k < arity; ++k)
    {
        atom += ' ';
        const i32 o = objects[k];
        if (known_instance)
        {
            const formalism::TaskData& data = table.instance(static_cast<u32>(instance)).task->data();
            if (o >= 0 && static_cast<u64>(o) < data.objects.size())
            {
                atom += data.str(data.objects[static_cast<u64>(o)].name);
                continue;
            }
        }
        atom += std::to_string(o);
    }
    atom += ')';
    return positive ? atom : "(not " + atom + ")";
}

void resolve_literals(const EncodingContext& context, const LiteralArrays& literals, std::span<const i32> instances,
                      std::span<LiteralKind> kinds, std::span<i32> slots)
{
    check_arrays(literals);
    const u64 n = literals.predicates.size();
    if ((!instances.empty() && instances.size() != n) || kinds.size() != n || slots.size() != n)
        throw std::invalid_argument("mymyr: encoding: the arrays of the literals differ in length");
    const detail::ContextView v = context.view();
    for (u64 i = 0; i < n; ++i)
    {
        const detail::LiteralSlot s =
            detail::resolve_literal(v, instances.empty() ? 0 : instances[i], literals.predicates[i],
                                    literals.objects.data() + i * literals.width, literals.width);
        kinds[i] = s.kind;
        slots[i] = s.kind == LiteralKind::Fluent || s.kind == LiteralKind::Derived ? static_cast<i32>(s.slot) : -1;
    }
}

LiteralMasks literal_masks(const EncodingContext& context, const LiteralArrays& literals,
                           const LiteralPlacement& placement, FixedLiterals fixed)
{
    check_arrays(literals);
    const u64 n = literals.predicates.size();
    if (placement.positive.size() != n || placement.rows.size() != n ||
        (!placement.layers.empty() && placement.layers.size() != n))
        throw std::invalid_argument("mymyr: encoding: the arrays of the literals differ in length");
    const rl::TaskTable& table = context.table();
    if (placement.instances.empty() ? placement.num_rows > 0 && table.size() != 1
                                    : placement.instances.size() != placement.num_rows)
        throw std::invalid_argument(placement.instances.empty()
                                        ? "mymyr: encoding: task_ids are required for a table of several instances"
                                        : "mymyr: encoding: task_ids must have one entry per row");
    for (const i32 instance : placement.instances)
        if (instance < 0 || static_cast<u32>(instance) >= table.size())
            throw std::invalid_argument("mymyr: encoding: task_ids must be in [0, " + std::to_string(table.size()) +
                                        ")");
    if (placement.num_layers == 0 && n > 0)
        throw std::invalid_argument("mymyr: encoding: literals need at least one layer");
    if (placement.instance_constants && placement.num_layers != 1)
        throw std::invalid_argument("mymyr: encoding: the instance's constant goal literals go to goals of one layer");

    LiteralMasks out;
    out.rows = placement.num_rows;
    out.layers = placement.num_layers;
    out.words = table.words();
    out.derived_words = table.derived_words();
    const u64 cells = u64{out.rows} * out.layers;
    out.pos.assign(cells * out.words, 0);
    out.neg.assign(cells * out.words, 0);
    out.derived_pos.assign(cells * out.derived_words, 0);
    out.derived_neg.assign(cells * out.derived_words, 0);
    out.unsatisfiable.assign(cells, 0);
    const detail::ContextView v = context.view();
    const u32 width = detail::k_constant_objects + v.max_arity;
    std::vector<std::vector<std::vector<i32>>> constants(out.rows);  // per row its constant literals
    if (placement.instance_constants)
        for (u32 row = 0; row < out.rows; ++row)
        {
            const detail::ConstantRows own =
                instance_goal_constants(context, placement.instances.empty() ? 0 : static_cast<u32>(placement.instances[row]));
            for (u32 k = 0; k < own.count; ++k)
                constants[row].emplace_back(own.row(k), own.row(k) + width);
        }
    for (u64 i = 0; i < n; ++i)
    {
        const i32 row = placement.rows[i];
        const i32 layer = placement.layers.empty() ? 0 : placement.layers[i];
        if (row < 0 || static_cast<u32>(row) >= out.rows)
            throw std::invalid_argument("mymyr: encoding: the row " + std::to_string(row) + " of literal " +
                                        std::to_string(i) + " is not in [0, " + std::to_string(out.rows) + ")");
        if (layer < 0 || static_cast<u32>(layer) >= out.layers)
            throw std::invalid_argument("mymyr: encoding: the layer " + std::to_string(layer) + " of literal " +
                                        std::to_string(i) + " is not in [0, " + std::to_string(out.layers) + ")");
        const i32 instance = placement.instances.empty() ? 0 : placement.instances[static_cast<u32>(row)];
        const i32 predicate = literals.predicates[i];
        const std::span<const i32> objects = literals.objects.subspan(i * literals.width, literals.width);
        const bool positive = placement.positive[i] != 0;
        const detail::LiteralSlot s = detail::resolve_literal(v, instance, predicate, objects.data(), literals.width);
        const u64 cell = u64{static_cast<u32>(row)} * out.layers + static_cast<u32>(layer);
        auto text = [&] { return literal_text(context, instance, predicate, objects, positive); };
        switch (s.kind)
        {
        case LiteralKind::Fluent:
        case LiteralKind::Derived:
        {
            const bool derived = s.kind == LiteralKind::Derived;
            std::vector<u64>& mask =
                derived ? (positive ? out.derived_pos : out.derived_neg) : (positive ? out.pos : out.neg);
            mask[cell * (derived ? out.derived_words : out.words) + (s.slot >> 6)] |= u64{1} << (s.slot & 63u);
            break;
        }
        case LiteralKind::Invalid:
            throw std::invalid_argument("mymyr: encoding: the literal " + text() + " (literal " + std::to_string(i) +
                                        ") is invalid: " + invalid_reason(context, instance, predicate, objects));
        default:
            if (fixed == FixedLiterals::Raise)
                throw std::invalid_argument(
                    "mymyr: encoding: the literal " + text() +
                    (s.kind == LiteralKind::Unreachable
                         ? " has no slot in its task: its atom holds in no state"
                         : " is over a static predicate: it has no slot"));
            if (!detail::fixed_literal_holds(s.kind, positive))
                out.unsatisfiable[cell] = 1;
            if (fixed == FixedLiterals::Encode)
            {
                std::vector<i32> literal(width, -1);
                literal[detail::k_constant_layer] = layer;
                literal[detail::k_constant_predicate] = predicate;
                literal[detail::k_constant_negative] = positive ? 0 : 1;
                for (u32 k = 0; k < literals.width && k < v.max_arity; ++k)
                    literal[detail::k_constant_objects + k] = objects[k];
                constants[static_cast<u32>(row)].push_back(std::move(literal));
            }
            break;
        }
    }
    if (fixed == FixedLiterals::Encode || placement.instance_constants)
    {
        for (auto& row : constants)
        {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            out.constant_capacity = std::max(out.constant_capacity, static_cast<u32>(row.size()));
        }
        out.constant_width = width;
        out.constants.assign(u64{out.rows} * out.constant_capacity * width, -1);
        for (u32 row = 0; row < out.rows; ++row)
            for (u32 k = 0; k < constants[row].size(); ++k)
                std::ranges::copy(constants[row][k],
                                  out.constants.begin() + static_cast<std::ptrdiff_t>((u64{row} * out.constant_capacity + k) * width));
    }
    return out;
}

detail::ConstantRows instance_goal_constants(const EncodingContext& context, u32 instance)
{
    const detail::ContextView v = context.view();
    if (instance >= v.num_instances)
        throw std::invalid_argument("mymyr: encoding: task id " + std::to_string(instance) + " is not in [0, " +
                                    std::to_string(v.num_instances) + ")");
    return detail::instance_view(v, instance).goal_constants;
}

std::string constant_fault_text(const EncodingContext& context, u32 instance, const detail::ConstantRows& rows,
                                u32 layers)
{
    const detail::ContextView v = context.view();
    u32 at = 0;
    const detail::ConstantFault fault = detail::constant_fault(detail::instance_view(v, instance), rows, layers, at);
    if (fault == detail::ConstantFault::None)
        return {};
    const i32* row = rows.row(at);
    const std::span<const i32> objects(row + detail::k_constant_objects, rows.width - detail::k_constant_objects);
    const std::string which = "constant literal " + std::to_string(at) + " ";
    switch (fault)
    {
    case detail::ConstantFault::Unordered:
        return which + "is not after the one before it: the literals of a row are strictly ascending (layer, "
                       "predicate, sign, objects)";
    case detail::ConstantFault::Layer:
        return which + "has the layer " + std::to_string(row[detail::k_constant_layer]) + ", outside [0, " +
               std::to_string(layers) + ")";
    case detail::ConstantFault::Sign:
        return which + "has the sign " + std::to_string(row[detail::k_constant_negative]) + ", not 0 or 1";
    case detail::ConstantFault::Invalid:
        return which + "is invalid: " +
               invalid_reason(context, instance, row[detail::k_constant_predicate], objects);
    default:
        return which + literal_text(context, instance, row[detail::k_constant_predicate], objects,
                                    row[detail::k_constant_negative] == 0) +
               " has a slot: the goal masks hold it";
    }
}
}  // namespace mymyr::encoding
