#pragma once
// Action identity and effect deltas.
//   - An action is the label (schema, binding objects): the schema's full parameter list, including the parameters
//     normalization introduced. Slot indices are never identities.
//   - A Delta lists the raw effect slots of one applicable action before any successor is materialized: apply as
//     "delete, then add". Lists may repeat slots, may delete atoms that are false and add atoms that are already true.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"

#include <algorithm>
#include <compare>
#include <span>
#include <vector>

namespace mymyr
{
/// Non-owning label, valid during the callback that received it.
struct ActionLabel
{
    SchemaId schema;
    std::span<const ObjectId> binding;
};

/// Owning label (plans, stored transitions, test oracles).
struct Action
{
    SchemaId schema;
    std::vector<ObjectId> binding;

    Action() = default;
    Action(SchemaId s, std::vector<ObjectId> b) : schema(s), binding(std::move(b)) {}
    explicit Action(const ActionLabel& l) : schema(l.schema), binding(l.binding.begin(), l.binding.end()) {}
    [[nodiscard]] ActionLabel label() const { return {schema, binding}; }

    friend bool operator==(const Action&, const Action&) = default;
    /// Canonical order: schema, then lexicographic binding.
    friend auto operator<=>(const Action& a, const Action& b)
    {
        if (auto c = a.schema <=> b.schema; c != 0)
            return c;
        return std::lexicographical_compare_three_way(a.binding.begin(), a.binding.end(), b.binding.begin(), b.binding.end());
    }
};

/// One numeric effect of an applicable action: numeric slot `slot` becomes op(old value, value); `value` was
/// evaluated on the parent state. The writes of one action apply in order to a copy of the parent's values.
struct NumericWrite
{
    u32 slot = 0;
    formalism::AssignOp op = formalism::AssignOp::Assign;
    f64 value = 0;
};

/// One total-cost effect of an applicable action: the metric value (g) becomes op(old, value).
struct AuxWrite
{
    formalism::AssignOp op = formalism::AssignOp::Increase;
    f64 value = 0;
};

[[nodiscard]] constexpr f64 apply_assign(formalism::AssignOp op, f64 old, f64 v) noexcept
{
    switch (op)
    {
        case formalism::AssignOp::Assign: return v;
        case formalism::AssignOp::Increase: return old + v;
        case formalism::AssignOp::Decrease: return old - v;
        case formalism::AssignOp::ScaleUp: return old * v;
        case formalism::AssignOp::ScaleDown: return old / v;
    }
    return v;
}

struct Delta
{
    std::span<const SlotId> add, del;
    /// Numeric tasks: the successor's numeric words (canonical, the task's numeric_words() words), valid during
    /// the callback. Null for classical tasks.
    const u64* num = nullptr;
    u32 nnum = 0;
    /// The numeric effects that produced `num` (empty when no numeric effect fired).
    std::span<const NumericWrite> writes{};
    /// The total-cost effects of the fired (conditional) effects, in order (tasks with total-cost). mimir's metric
    /// value of the successor is heuristics::ActionCosts::next(g, delta, successor).
    std::span<const AuxWrite> aux{};
};
}  // namespace mymyr
