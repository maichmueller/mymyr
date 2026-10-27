#pragma once
// Lifted conjunctions matched in the compiler's binding order, objects ascending. The iterative search intersects
// positive literal candidates in 32-object tiles and checks other literals when their last variable is bound.
// Still-free variables of a positive literal are existential filters, pruning candidates without changing order.
// The same search serves a host thread or a device warp; lane-parallel scans keep each tile's candidate set exact.
#include "mymyr/encoding/detail/literals.hpp"

namespace mymyr
{
class ThreadPool;
}
namespace mymyr::rl
{
struct MatchTemplates
{
    /// Literal resolution: instance object counts, tuple offsets, sorted tuple rows and predicate ranges.
    encoding::detail::ContextView literals{};
    u32 num_instances = 0, capacity = 0, max_vars = 0, max_arity = 0;
    const i32 *instance_offsets = nullptr, *num_vars = nullptr, *literal_offsets = nullptr, *order = nullptr;
    const i32 *predicate = nullptr, *arity = nullptr, *terms = nullptr;
    const u8 *polarity = nullptr, *distinct = nullptr;
};

/// Fixed output shapes: [rows, capacity], binding [rows, capacity, templates.max_vars], count [rows]. With first,
/// capacity is one and the template search stops at its first success. Otherwise slot j is instance-local template j.
/// template_id holds global packed template ids, -1 for failures/padding; binding holds variable-indexed object ids,
/// -1 for failures/padding. Count is the number found, including those beyond capacity. Status is cleared per call:
/// bit 0 output truncation, bit 1 device variable limit, bit 2 invalid instance. No input words are read for invalid ids.
struct MatchBatch
{
    const u64* states = nullptr;
    const i32* task_ids = nullptr;
    const u64* derived = nullptr;
    u64 rows = 0;
    u32 row_words = 0, derived_words = 0, capacity = 0;
    bool first = false;
    u8* found = nullptr;
    i32 *binding = nullptr, *template_id = nullptr, *count = nullptr;
    u64* status = nullptr;
};

namespace match_detail
{
namespace ed = encoding::detail;
struct Objects
{
    const i32* terms;
    const i32* binding;
    MYMYR_HD i32 operator[](u32 j) const { return terms[j] < 0 ? -(terms[j] + 1) : binding[terms[j]]; }
};

MYMYR_HD bool bit(const u64* words, u32 slot) { return (words[slot / 64] >> (slot % 64)) & 1; }

MYMYR_HD bool holds(const MatchTemplates& t, u32 g, u32 l, const i32* binding, const u64* state, const u64* derived)
{
    const Objects objects{t.terms + u64{l} * t.max_arity, binding};
    const auto slot = ed::resolve_literal(t.literals, g, t.predicate[l], objects, static_cast<u32>(t.arity[l]));
    bool truth = slot.kind == ed::LiteralKind::StaticTrue;
    if (slot.kind == ed::LiteralKind::Fluent)
        truth = bit(state, slot.slot);
    else if (slot.kind == ed::LiteralKind::Derived)
        truth = bit(derived, slot.slot);
    return slot.kind != ed::LiteralKind::Invalid && truth == (t.polarity[l] != 0);
}

template<class Group>
MYMYR_HD bool ready(const Group& grp, const MatchTemplates& t, u32 g, u32 k, const i32* binding,
                     const u64* state, const u64* derived, i32 changed)
{
    u32 failed = 0;
    for (u32 l = static_cast<u32>(t.literal_offsets[k]) + grp.lane(); l < static_cast<u32>(t.literal_offsets[k + 1]);
         l += grp.size())
    {
        bool bound = true, touched = changed < 0;
        for (i32 j = 0; j < t.arity[l]; ++j)
        {
            const i32 term = t.terms[u64{l} * t.max_arity + j];
            if (term >= 0 && binding[term] < 0)
                bound = false;
            touched |= term == changed;
        }
        if (bound && touched && !holds(t, g, l, binding, state, derived))
            failed = 1;
    }
    return grp.bit_or(failed) == 0;
}

template<class Group>
MYMYR_HD u32 candidates(const Group& grp, const MatchTemplates& t, u32 g, u32 k, i32 variable, u32 base,
                         u32 objects, const i32* binding, const u64* state, const u64* derived)
{
    u32 mask = objects - base >= 32 ? ~u32{0} : (u32{1} << (objects - base)) - 1;
    const u64* inst = t.literals.inst + u64{g} * ed::k_instance_columns;
    for (u32 l = t.literal_offsets[k]; l < static_cast<u32>(t.literal_offsets[k + 1]); ++l)
    {
        if (!t.polarity[l])
            continue;
        i32 position = -1;
        const i32* terms = t.terms + u64{l} * t.max_arity;
        for (i32 j = 0; j < t.arity[l]; ++j)
            if (terms[j] == variable)
                position = j;
        if (position < 0)
            continue;
        const u32* r = t.literals.literal_ranges + (u64{g} * t.literals.num_predicates + t.predicate[l]) * ed::k_literal_columns;
        const u32 kind = r[ed::k_literal_predicate_kind];
        const bool stat = kind == ed::k_literal_static_predicate, der = kind == ed::k_literal_derived_predicate;
        const i32* rows = stat ? t.literals.static_args + inst[ed::k_static_args_offset] * t.literals.max_arity
                             : der ? t.literals.derived_args + inst[ed::k_derived_args_offset] * t.literals.max_arity
                                   : t.literals.slot_args + inst[ed::k_slot_args_offset] * t.literals.max_arity;
        const u32 begin = r[stat ? ed::k_literal_static_begin : der ? ed::k_literal_derived_begin : ed::k_literal_fluent_begin];
        const u32 end = r[stat ? ed::k_literal_static_end : der ? ed::k_literal_derived_end : ed::k_literal_fluent_end];
        u32 lane_mask = 0;
        for (u32 n = begin + grp.lane(); n < end; n += grp.size())
        {
            const i32* row = rows + u64{n} * t.literals.max_arity;
            const u32 object = static_cast<u32>(row[position]);
            if (object < base || object - base >= 32 || (!stat && !bit(der ? derived : state, n)))
                continue;
            bool same = true;
            for (i32 j = 0; j < t.arity[l]; ++j)
            {
                const i32 term = terms[j];
                const i32 want = term == variable ? static_cast<i32>(object) : term < 0 ? -(term + 1) : binding[term];
                if (want >= 0)
                    same &= row[j] == want;
                if (term >= 0 && want < 0)
                    for (i32 before = 0; before < j; ++before)
                        if (terms[before] == term)
                            same &= row[before] == row[j];
            }
            if (same)
                lane_mask |= u32{1} << (object - base);
        }
        mask &= grp.bit_or(lane_mask);
        if (!mask)
            return 0;
    }
    if (t.distinct[k])
        for (u32 v = 0; v < static_cast<u32>(t.num_vars[k]); ++v)
            if (binding[v] >= 0 && static_cast<u32>(binding[v]) >= base && static_cast<u32>(binding[v]) - base < 32)
                mask &= ~(u32{1} << (static_cast<u32>(binding[v]) - base));
    return mask;
}

/// Scratch binding, base and mask each have num_vars[k] entries, private to the lane; no recursion or allocation.
template<class Group>
MYMYR_HD bool search(const Group& grp, const MatchTemplates& t, u32 g, u32 k, const u64* state, const u64* derived,
                      i32* binding, u32* base, u32* masks)
{
    const u32 vars = static_cast<u32>(t.num_vars[k]);
    for (u32 v = 0; v < vars; ++v)
        binding[v] = -1;
    if (!ready(grp, t, g, k, binding, state, derived, -1))
        return false;
    if (!vars)
        return true;
    const u32 objects = static_cast<u32>(t.literals.inst[u64{g} * ed::k_instance_columns + ed::k_num_objects]);
    i32 depth = 0;
    bool enter = true;
    while (depth >= 0)
    {
        const i32 variable = t.order[u64{k} * t.max_vars + depth];
        binding[variable] = -1;
        if (enter)
        {
            base[depth] = 0;
            masks[depth] = objects ? candidates(grp, t, g, k, variable, 0, objects, binding, state, derived) : 0;
            enter = false;
        }
        while (!masks[depth] && base[depth] < objects && objects - base[depth] > 32)
        {
            base[depth] += 32;
            masks[depth] = candidates(grp, t, g, k, variable, base[depth], objects, binding, state, derived);
        }
        if (!masks[depth])
        {
            --depth;
            continue;
        }
        const u32 pick = ed::lowest_bit64(masks[depth]);
        masks[depth] &= masks[depth] - 1;
        binding[variable] = static_cast<i32>(base[depth] + pick);
        if (!ready(grp, t, g, k, binding, state, derived, variable))
            continue;
        if (static_cast<u32>(depth + 1) == vars)
            return true;
        ++depth;
        enter = true;
    }
    return false;
}
}  // namespace match_detail

/// Host destination-passing form. Input widths must cover the suite's fluent/derived slots, and derived is required
/// when the templates use it. A pool partitions rows; no scratch or output is shared between workers.
void match_first(const MatchTemplates& templates, const MatchBatch& batch, ThreadPool* pool = nullptr);
}  // namespace mymyr::rl
