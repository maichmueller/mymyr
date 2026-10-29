#pragma once
// Lifted conjunctions matched in the compiler's binding order, objects ascending. The iterative search intersects
// positive literal candidates in 32-object tiles and checks other literals when their last variable is bound.
// Still-free variables of a positive literal are existential filters, pruning candidates without changing order.
// The same search serves a host thread or a device warp; lane-parallel scans keep each tile's candidate set exact.
// A scan reads only the indexed tuples that agree with the literal's leading constant and bound arguments (a binary
// search over the sorted tuples). Injective templates also get counting tests (domains_possible) that reject partial
// bindings without an injective completion; like the filters, they never change which binding is found first.
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
/// bit 0 output truncation, bit 1 device variable limit, bit 2 invalid instance, bit 3 search budget exhausted.
/// Exhausted templates have no result; other templates are still searched. No input words are read for invalid ids.
struct MatchBatch
{
    const u64* states = nullptr;
    const i32* task_ids = nullptr;
    const u64* derived = nullptr;
    u64 rows = 0;
    u32 row_words = 0, derived_words = 0, capacity = 0;
    bool first = false;
    /// Search budget per row and template: candidate tile scans (pruning included) and binding trials.
    u64 max_steps = 100000000;
    u8* found = nullptr;
    i32 *binding = nullptr, *template_id = nullptr, *count = nullptr;
    u64* status = nullptr;
};

namespace match_detail
{
namespace ed = encoding::detail;
struct Budget
{
    u64 remaining;
    bool exhausted = false;
    MYMYR_HD bool take()
    {
        if (remaining)
        {
            --remaining;
            return true;
        }
        exhausted = true;
        return false;
    }
};
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

/// The indexed tuples of literal l's predicate in instance g: rows [begin, end) of `rows`, true in every state when
/// static, else when bit n of the state (derived words when derived) is set.
struct Tuples
{
    const i32* rows;
    u32 begin, end;
    bool stat, der;
    MYMYR_HD bool holds(u32 n, const u64* state, const u64* derived) const
    {
        return stat || bit(der ? derived : state, n);
    }
};

MYMYR_HD Tuples tuples(const MatchTemplates& t, u32 g, u32 l)
{
    const u64* inst = t.literals.inst + u64{g} * ed::k_instance_columns;
    const u32* r = t.literals.literal_ranges + (u64{g} * t.literals.num_predicates + t.predicate[l]) * ed::k_literal_columns;
    const u32 kind = r[ed::k_literal_predicate_kind];
    const bool stat = kind == ed::k_literal_static_predicate, der = kind == ed::k_literal_derived_predicate;
    const i32* rows = stat ? t.literals.static_args + inst[ed::k_static_args_offset] * t.literals.max_arity
                      : der ? t.literals.derived_args + inst[ed::k_derived_args_offset] * t.literals.max_arity
                            : t.literals.slot_args + inst[ed::k_slot_args_offset] * t.literals.max_arity;
    const u32 begin = r[stat ? ed::k_literal_static_begin : der ? ed::k_literal_derived_begin : ed::k_literal_fluent_begin];
    const u32 end = r[stat ? ed::k_literal_static_end : der ? ed::k_literal_derived_end : ed::k_literal_fluent_end];
    return {rows, begin, end, stat, der};
}

MYMYR_HD i32 resolved(const MatchTemplates& t, u32 l, i32 j, const i32* binding)
{
    const i32 term = t.terms[u64{l} * t.max_arity + j];
    return term < 0 ? -(term + 1) : binding[term];
}

/// The first of rows [begin, end) whose objects 0 .. depth - 1 and object `depth` are not lexicographically less than
/// literal l's resolved arguments 0 .. depth - 1 and x (rows of one predicate are sorted lexicographically).
MYMYR_HD u32 lower_row(const MatchTemplates& t, u32 l, const i32* rows, u32 begin, u32 end, u32 depth, i64 x,
                        const i32* binding)
{
    while (begin < end)
    {
        const u32 mid = begin + (end - begin) / 2;
        const i32* row = rows + u64{mid} * t.literals.max_arity;
        i64 have = row[depth], want = x;
        for (u32 j = 0; j < depth; ++j)
            if (row[j] != resolved(t, l, static_cast<i32>(j), binding))
            {
                have = row[j];
                want = resolved(t, l, static_cast<i32>(j), binding);
                break;
            }
        if (have < want)
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

/// Narrows tup to the rows that share literal l's leading constant and bound arguments and, when `position` directly
/// follows them, have an object of base + [0, 32) at `position`: a binary search instead of a scan of every row. In
/// that case it returns true and sets `after` so that rows [tup.end, after) share the leading arguments and have larger
/// objects at `position`. Ranges of at most 8 rows per lane are scanned as they are, which is cheaper than searching.
MYMYR_HD bool narrow(const MatchTemplates& t, u32 l, i32 position, u32 base, const i32* binding, u32 lanes, Tuples& tup,
                     u32& after)
{
    if (tup.end - tup.begin <= 8 * lanes)
        return false;
    i32 known = 0;
    while (known < t.arity[l] && known != position && resolved(t, l, known, binding) >= 0)
        ++known;
    const u32 d = static_cast<u32>(known);
    u32 first = tup.begin, end = tup.end;
    if (known > 0)
    {
        const i64 last = resolved(t, l, known - 1, binding);
        first = lower_row(t, l, tup.rows, first, end, d - 1, last, binding);
        end = lower_row(t, l, tup.rows, first, end, d - 1, last + 1, binding);
    }
    if (known != position)
    {
        tup.begin = first;
        tup.end = end;
        return false;
    }
    tup.begin = lower_row(t, l, tup.rows, first, end, d, base, binding);
    tup.end = lower_row(t, l, tup.rows, tup.begin, end, d, i64{base} + 32, binding);
    after = end;
    return true;
}

/// Whether tuple `row` of literal l's predicate fits l with `object` at `position`: in the candidate scan of `variable`
/// (domain = false) every constant, bound and repeated argument agrees; in a projection (domain = true) its constant
/// and bound arguments agree and unbound variables are wildcards.
MYMYR_HD bool fits(const MatchTemplates& t, u32 l, const i32* row, i32 position, i32 variable, i32 object,
                   const i32* binding, bool domain)
{
    const i32* terms = t.terms + u64{l} * t.max_arity;
    bool same = true;
    for (i32 j = 0; j < t.arity[l]; ++j)
    {
        const i32 term = terms[j];
        if (domain)
        {
            const i32 want = term < 0 ? -(term + 1) : binding[term];
            if (j != position && want >= 0)
                same &= row[j] == want;
            continue;
        }
        const i32 want = term == variable ? object : term < 0 ? -(term + 1) : binding[term];
        if (want >= 0)
            same &= row[j] == want;
        if (term >= 0 && want < 0)
            for (i32 before = 0; before < j; ++before)
                if (terms[before] == term)
                    same &= row[before] == row[j];
    }
    return same;
}

/// The argument position of `variable` in literal l (its last occurrence), -1 when it does not occur.
MYMYR_HD i32 position_of(const MatchTemplates& t, u32 l, i32 variable)
{
    i32 position = -1;
    for (i32 j = 0; j < t.arity[l]; ++j)
        if (t.terms[u64{l} * t.max_arity + j] == variable)
            position = j;
    return position;
}

/// Candidate objects base + [0, 32) of `variable`: those satisfying every positive literal it occurs in. With
/// domain_literal >= 0, instead the projection of that literal's true tuples on domain_position, filtered only by its
/// constant and bound arguments. Under distinct, bound objects are removed.
template<class Group>
MYMYR_HD u32 candidates(const Group& grp, const MatchTemplates& t, u32 g, u32 k, i32 variable, u32 base, u32 objects,
                         const i32* binding, const u64* state, const u64* derived, i32 domain_literal = -1,
                         u32 domain_position = 0)
{
    u32 mask = objects - base >= 32 ? ~u32{0} : (u32{1} << (objects - base)) - 1;
    const bool domain = domain_literal >= 0;
    const u32 begin_literal = domain ? static_cast<u32>(domain_literal) : static_cast<u32>(t.literal_offsets[k]);
    const u32 end_literal = domain ? begin_literal + 1 : static_cast<u32>(t.literal_offsets[k + 1]);
    for (u32 l = begin_literal; l < end_literal; ++l)
    {
        if (!t.polarity[l])
            continue;
        const i32 position = domain ? static_cast<i32>(domain_position) : position_of(t, l, variable);
        if (position < 0)
            continue;
        Tuples tup = tuples(t, g, l);
        u32 after = 0;
        narrow(t, l, position, base, binding, grp.size(), tup, after);
        u32 lane_mask = 0;
        for (u32 n = tup.begin + grp.lane(); n < tup.end; n += grp.size())
        {
            const i32* row = tup.rows + u64{n} * t.literals.max_arity;
            const u32 object = static_cast<u32>(row[position]);
            if (object < base || object - base >= 32 || !tup.holds(n, state, derived))
                continue;
            if (fits(t, l, row, position, variable, static_cast<i32>(object), binding, domain))
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

/// Which of the 32 tiles after base + [0, 32) may hold candidates of `variable` (bit j: objects base + 32 * (j + 1) +
/// [0, 32)): those where every positive literal of the variable has a matching tuple. A literal whose leading
/// constant and bound arguments are followed by the variable reads only rows up to the first one past the tile (its
/// rows are sorted by the variable's object there), whose tile bounds the later ones from below.
template<class Group>
MYMYR_HD u32 later_tiles(const Group& grp, const MatchTemplates& t, u32 g, u32 k, i32 variable, u32 base, u32 objects,
                          const i32* binding, const u64* state, const u64* derived)
{
    const u32 count = objects - base > 32 ? (objects - base - 1) / 32 : 0;
    u32 later = count >= 32 ? ~u32{0} : (u32{1} << count) - 1;
    for (u32 l = static_cast<u32>(t.literal_offsets[k]); l < static_cast<u32>(t.literal_offsets[k + 1]) && later; ++l)
    {
        const i32 position = t.polarity[l] ? position_of(t, l, variable) : -1;
        if (position < 0)
            continue;
        Tuples tup = tuples(t, g, l);
        u32 after = 0;
        const bool tiled = narrow(t, l, position, base, binding, grp.size(), tup, after);
        u32 seen = 0;
        if (tiled)
        {
            if (tup.end < after)
            {
                const u32 next = static_cast<u32>(tup.rows[u64{tup.end} * t.literals.max_arity + position]) - base;
                seen = next < 33 * 32 ? ~u32{0} << (next / 32 - 1) : 0;
            }
        }
        else
        {
            for (u32 n = tup.begin + grp.lane(); n < tup.end; n += grp.size())
            {
                const i32* row = tup.rows + u64{n} * t.literals.max_arity;
                const u32 object = static_cast<u32>(row[position]);
                if (object < base + 32 || object - base >= 33 * 32 || !tup.holds(n, state, derived))
                    continue;
                if (fits(t, l, row, position, variable, static_cast<i32>(object), binding, false))
                    seen |= u32{1} << ((object - base) / 32 - 1);
            }
            seen = grp.bit_or(seen);
        }
        later &= seen;
    }
    return later;
}

/// Whether positive literal `other` has the predicate of `l`, an unbound variable at `position`, and at every other
/// position the object of `l` wherever `l` has a constant or bound argument (with `exact`, the same arguments).
MYMYR_HD bool projected(const MatchTemplates& t, u32 l, u32 other, i32 position, const i32* binding, bool exact)
{
    if (!t.polarity[other] || t.predicate[other] != t.predicate[l])
        return false;
    const i32 term = t.terms[u64{other} * t.max_arity + position];
    if (term < 0 || binding[term] >= 0)
        return false;
    for (i32 j = 0; j < t.arity[l]; ++j)
    {
        const i32 want = resolved(t, l, j, binding), have = resolved(t, other, j, binding);
        if (j != position && (exact || want >= 0) && want != have)
            return false;
    }
    return true;
}

/// The argument position of l other than `position` that holds l's only other unbound variable; -1 when there is none,
/// there are several or it is the variable at `position`.
MYMYR_HD i32 partner_position(const MatchTemplates& t, u32 l, i32 position, const i32* binding)
{
    const i32* terms = t.terms + u64{l} * t.max_arity;
    i32 q = -1;
    for (i32 j = 0; j < t.arity[l]; ++j)
    {
        if (j == position || terms[j] < 0 || binding[terms[j]] >= 0)
            continue;
        if (q >= 0)
            return -1;
        q = j;
    }
    return q >= 0 && terms[q] == terms[position] ? -1 : q;
}

/// Whether positive literal `other` has the predicate of `l`, distinct unbound variables at `position` and q, and the
/// objects of `l` at every other position (l's only unbound arguments are at `position` and q).
MYMYR_HD bool sibling(const MatchTemplates& t, u32 l, u32 other, i32 position, i32 q, const i32* binding)
{
    if (!t.polarity[other] || t.predicate[other] != t.predicate[l])
        return false;
    const i32* terms = t.terms + u64{other} * t.max_arity;
    if (terms[position] < 0 || binding[terms[position]] >= 0 || terms[q] < 0 || binding[terms[q]] >= 0 ||
        terms[q] == terms[position])
        return false;
    for (i32 j = 0; j < t.arity[l]; ++j)
        if (j != position && j != q && resolved(t, l, j, binding) != resolved(t, other, j, binding))
            return false;
    return true;
}

/// Objects c of base + [0, 32), bound to no variable, with at least `threshold` supporters: true tuples of l's predicate
/// with c at `position`, l's objects at its other positions but q, and at q an object other than c that no variable is
/// bound to (the tuples differ only at q, so their supporters are distinct objects). Lane counts are 8-bit fields that
/// saturate at 255 / lanes, so that their sum over the lanes fits; an object saturated in a lane is kept (more
/// supporters than counted only weaken the test).
template<class Group>
MYMYR_HD u32 supported(const Group& grp, const MatchTemplates& t, u32 g, u32 k, u32 l, i32 position, i32 q, u32 base,
                        u32 objects, u32 threshold, const i32* binding, const u64* state, const u64* derived)
{
    Tuples tup = tuples(t, g, l);
    u32 after = 0;
    narrow(t, l, position, base, binding, grp.size(), tup, after);
    const u32 vars = static_cast<u32>(t.num_vars[k]), cap = 255 / grp.size();
    u32 counts[8] = {}, saturated = 0;
    for (u32 n = tup.begin + grp.lane(); n < tup.end; n += grp.size())
    {
        const i32* row = tup.rows + u64{n} * t.literals.max_arity;
        const u32 object = static_cast<u32>(row[position]);
        if (object < base || object - base >= 32 || row[q] == row[position] || !tup.holds(n, state, derived))
            continue;
        bool same = true;
        for (i32 j = 0; j < t.arity[l]; ++j)
            if (j != position && j != q)
                same &= row[j] == resolved(t, l, j, binding);
        for (u32 v = 0; v < vars; ++v)
            same &= binding[v] != row[q];
        if (!same)
            continue;
        const u32 at = object - base, shift = at % 4 * 8;
        // A select over the eight words keeps the counters in registers on a device.
        for (u32 w = 0; w < 8; ++w)
            if (w == at / 4)
            {
                if (((counts[w] >> shift) & 255) < cap)
                    counts[w] += u32{1} << shift;
                else
                    saturated |= u32{1} << at;
            }
    }
    u32 mask = grp.bit_or(saturated);
    for (u32 w = 0; w < 8 && 4 * w < objects - base; ++w)
    {
        const u32 total = grp.sum(counts[w]);
        for (u32 b = 0; b < 4; ++b)
            if (((total >> (8 * b)) & 255) >= threshold)
                mask |= u32{1} << (4 * w + b);
    }
    for (u32 v = 0; v < vars; ++v)
        if (binding[v] >= 0 && static_cast<u32>(binding[v]) >= base && static_cast<u32>(binding[v]) - base < 32)
            mask &= ~(u32{1} << (static_cast<u32>(binding[v]) - base));
    return mask;
}

/// Hall test of an injective template's unbound variables by their supporters (see supported) in the projection of l on
/// `position`, where l's other unbound argument is at q: a variable v at `position` of s siblings of l (sibling) with
/// distinct variables at q needs a value with s supporters, since those variables bind distinct unbound objects. The
/// value sets of the variables are nested by s, so the test that, for every s, the variables needing at least s fit
/// the objects with at least s supporters covers all of Hall's conditions among them. `need` is scratch of num_vars[k]
/// entries.
template<class Group>
MYMYR_HD bool supporters_possible(const Group& grp, const MatchTemplates& t, u32 g, u32 k, u32 l, i32 position, i32 q,
                                  u32 objects, const i32* binding, const u64* state, const u64* derived, u32* need,
                                  Budget& budget)
{
    const u32 begin = static_cast<u32>(t.literal_offsets[k]), end = static_cast<u32>(t.literal_offsets[k + 1]);
    const u32 vars = static_cast<u32>(t.num_vars[k]);
    for (u32 v = 0; v < vars; ++v)
        need[v] = 0;
    u32 top = 0;
    for (u32 other = begin; other < end; ++other)
    {
        if (!sibling(t, l, other, position, q, binding))
            continue;
        const i32* terms = t.terms + u64{other} * t.max_arity;
        bool repeated = false;
        for (u32 before = begin; before < other && !repeated; ++before)
            repeated = sibling(t, l, before, position, q, binding) &&
                       t.terms[u64{before} * t.max_arity + position] == terms[position] &&
                       t.terms[u64{before} * t.max_arity + q] == terms[q];
        if (!repeated && ++need[terms[position]] > top)
            top = need[terms[position]];
    }
    for (u32 threshold = 1; threshold < top;)
    {
        u32 members = 0, next = top;
        for (u32 v = 0; v < vars; ++v)
            if (need[v] > threshold && need[v] < next)
                next = need[v];
        threshold = next;
        for (u32 v = 0; v < vars; ++v)
            members += need[v] >= threshold;
        u32 available = 0;
        for (u32 tile = 0; tile < objects && available < members; tile += 32)
        {
            if (!budget.take())
                return false;
            available += ed::popcount64(
                supported(grp, t, g, k, l, position, q, tile, objects, threshold, binding, state, derived));
        }
        if (available < members)
            return false;
    }
    return true;
}

/// Whether unbound variable v occurs at `position` of a positive literal that projected() relates to l.
MYMYR_HD bool member(const MatchTemplates& t, u32 k, u32 l, i32 position, u32 v, const i32* binding)
{
    for (u32 other = static_cast<u32>(t.literal_offsets[k]); other < static_cast<u32>(t.literal_offsets[k + 1]); ++other)
        if (t.terms[u64{other} * t.max_arity + position] == static_cast<i32>(v) &&
            projected(t, l, other, position, binding, false))
            return true;
    return false;
}

/// Projections that domains_possible examines together; their state is fixed scratch of the lane.
inline constexpr u32 k_projection_batch = 16;

/// The pigeonhole test of domains_possible for a batch of projections (id = literal * max_arity + position). The
/// tiles are the outer loop: per tile, each projection of the batch is scanned once and each variable that may still
/// be confined in one of them once. A projection leaves the batch when its objects cover its members and every variable
/// that may still be confined in it; a variable is no longer scanned once it lies outside every projection. Variables
/// are taken in blocks of 64 (bit masks), so wider templates repeat the tiles per block.
template<class Group>
MYMYR_HD bool batch_possible(const Group& grp, const MatchTemplates& t, u32 g, u32 k, u32 objects, const u32* id,
                              u32 count, const i32* binding, const u64* state, const u64* derived, Budget& budget)
{
    const u32 vars = static_cast<u32>(t.num_vars[k]);
    u32 available[k_projection_batch], confined[k_projection_batch], domain[k_projection_batch];
    u64 members[k_projection_batch], open[k_projection_batch];
    u32 settled = 0;
    u32 later = 0;
    for (u32 v = 0; v < vars; ++v)
        later += binding[v] < 0;
    for (u32 i = 0; i < count; ++i)
        confined[i] = 0;
    for (u32 block = 0; block < vars; block += 64)
    {
        const u32 width = vars - block < 64 ? vars - block : 64;
        u64 unbound = 0;
        for (u32 v = 0; v < width; ++v)
            if (binding[block + v] < 0)
                unbound |= u64{1} << v;
        later -= ed::popcount64(unbound);
        u32 active = 0;
        for (u32 i = 0; i < count; ++i)
        {
            if ((settled >> i) & 1)
                continue;
            const u32 l = id[i] / t.max_arity;
            const i32 position = static_cast<i32>(id[i] % t.max_arity);
            members[i] = 0;
            for (u64 rest = unbound; rest; rest &= rest - 1)
            {
                const u32 v = ed::lowest_bit64(rest);
                if (member(t, k, l, position, block + v, binding))
                    members[i] |= u64{1} << v;
            }
            open[i] = unbound & ~members[i];
            available[i] = 0;
            active |= u32{1} << i;
        }
        for (u32 tile = 0; tile < objects && active; tile += 32)
        {
            u64 scanned = 0;
            for (u32 rest = active; rest; rest &= rest - 1)
            {
                const u32 i = ed::lowest_bit64(rest);
                if (available[i] >= confined[i] + ed::popcount64(members[i]) + ed::popcount64(open[i]) + later)
                {
                    settled |= u32{1} << i;
                    active &= ~(u32{1} << i);
                    continue;
                }
                if (!budget.take())
                    return false;
                const u32 l = id[i] / t.max_arity, position = id[i] % t.max_arity;
                domain[i] = candidates(grp, t, g, k, t.terms[u64{l} * t.max_arity + position], tile, objects, binding,
                                       state, derived, static_cast<i32>(l), position);
                available[i] += ed::popcount64(domain[i]);
                scanned |= open[i];
            }
            for (; scanned; scanned &= scanned - 1)
            {
                const u32 v = ed::lowest_bit64(scanned);
                if (!budget.take())
                    return false;
                const u32 objects_of_v = candidates(grp, t, g, k, static_cast<i32>(block + v), tile, objects, binding,
                                                    state, derived);
                for (u32 rest = active; rest; rest &= rest - 1)
                {
                    const u32 i = ed::lowest_bit64(rest);
                    if (objects_of_v & ~domain[i])
                        open[i] &= ~(u64{1} << v);
                }
            }
        }
        // The projections still in the batch were scanned completely.
        for (u32 rest = active; rest; rest &= rest - 1)
        {
            const u32 i = ed::lowest_bit64(rest);
            confined[i] += ed::popcount64(members[i]) + ed::popcount64(open[i]);
            if (confined[i] > available[i])
                return false;
        }
    }
    return true;
}

/// Pigeonhole test of an injective template's unbound variables (false: no injective completion exists). The
/// projection of a positive literal on one argument position, filtered by its constant and bound arguments and without
/// bound objects, contains the candidates of every unbound variable at that position of a positive literal of the
/// predicate with at least those arguments (its members; unary predicates project types). Those members, and every
/// other unbound variable whose candidates lie inside the projection (another predicate or position over the same
/// objects), need distinct objects of it. Projections only drop constraints, so the test never rejects a satisfiable
/// binding. A projection with one other unbound argument also gets the supporter test (supporters_possible; `need` is
/// its scratch). Projections go to batch_possible in batches of k_projection_batch.
template<class Group>
MYMYR_HD bool domains_possible(const Group& grp, const MatchTemplates& t, u32 g, u32 k, u32 objects,
                                const i32* binding, const u64* state, const u64* derived, u32* need, Budget& budget)
{
    const u32 begin = static_cast<u32>(t.literal_offsets[k]), end = static_cast<u32>(t.literal_offsets[k + 1]);
    u32 id[k_projection_batch];
    u32 count = 0;
    for (u32 l = begin; l < end; ++l)
        for (i32 position = 0; position < t.arity[l]; ++position)
        {
            if (!projected(t, l, l, position, binding, true))
                continue;
            // An earlier literal with the same arguments has the same projection and members.
            bool seen = false;
            for (u32 before = begin; before < l && !seen; ++before)
                seen = projected(t, l, before, position, binding, true);
            if (seen)
                continue;
            const i32 q = partner_position(t, l, position, binding);
            if (q >= 0 && !supporters_possible(grp, t, g, k, l, position, q, objects, binding, state, derived, need, budget))
                return false;
            id[count++] = l * t.max_arity + static_cast<u32>(position);
            if (count == k_projection_batch)
            {
                if (!batch_possible(grp, t, g, k, objects, id, count, binding, state, derived, budget))
                    return false;
                count = 0;
            }
        }
    return !count || batch_possible(grp, t, g, k, objects, id, count, binding, state, derived, budget);
}

/// Scratch binding, base, masks and need each have num_vars[k] entries, private to the lane; no recursion or
/// allocation. Worst-case work is exponential. The budget counts candidate tile scans (those of the domain tests
/// included) and binding trials; a tile scan is linear in the template's literals and the indexed tuples it reads. All
/// lanes consume equal steps, and exhaustion returns false with budget.exhausted set. For injective templates, the
/// domain tests (domains_possible) run at the root and, from the first failure on (a rejected binding trial or a
/// variable without candidates), at every node with more than two unbound variables (candidate filtering decides the
/// last two directly); the nodes entered before that failure are tested when the search returns to them. A search
/// without failures thus pays for one test. PruneDomains = false compiles the tests out of searches of at most
/// two variables, and SkipTiles = false the jumps over tiles without candidates (a scanned tile is then followed by
/// the next one): both make a smaller search for short ones.
template<class Group, bool PruneDomains = true, bool SkipTiles = true>
MYMYR_HD bool search(const Group& grp, const MatchTemplates& t, u32 g, u32 k, const u64* state, const u64* derived,
                      i32* binding, u32* base, u32* masks, u32* need, Budget& budget)
{
    const u32 vars = static_cast<u32>(t.num_vars[k]);
    const u32 objects = static_cast<u32>(t.literals.inst[u64{g} * ed::k_instance_columns + ed::k_num_objects]);
    if (t.distinct[k] && vars > objects)
        return false;
    for (u32 v = 0; v < vars; ++v)
        binding[v] = -1;
    if (!ready(grp, t, g, k, binding, state, derived, -1))
        return false;
    if (!vars)
        return true;
    i32 depth = 0;
    bool enter = true;
    // Past the root, the domain tests start at the first failure; the nodes entered before it (depths below
    // `untested`) are tested when the search returns to them.
    bool testing = false;
    i32 untested = 0;
    while (depth >= 0)
    {
        const i32 variable = t.order[u64{k} * t.max_vars + depth];
        binding[variable] = -1;
        bool test = false;
        if constexpr (PruneDomains)
            test = t.distinct[k] && vars - static_cast<u32>(depth) > 2 &&
                   (enter ? testing || depth == 0 : depth < untested);
        if (!enter && depth < untested)
            untested = depth;
        if (test && !domains_possible(grp, t, g, k, objects, binding, state, derived, need, budget))
        {
            if (budget.exhausted)
                return false;
            --depth;
            enter = false;
            continue;
        }
        // After an exhausted tile the next one is scanned; a scanned tile without candidates is followed by the next
        // tile that later_tiles leaves (its scan is part of the step that found the tile empty).
        bool skip = enter;
        if (enter)
        {
            if (!budget.take())
                return false;
            base[depth] = 0;
            masks[depth] = objects ? candidates(grp, t, g, k, variable, 0, objects, binding, state, derived) : 0;
            enter = false;
        }
        while (!masks[depth] && objects - base[depth] > 32)
        {
            u32 next = 1;
            if (SkipTiles && skip)
            {
                const u32 later = later_tiles(grp, t, g, k, variable, base[depth], objects, binding, state, derived);
                next = later ? ed::lowest_bit64(later) + 1 : 33;
                if (objects - base[depth] <= 32 * next)
                    break;
            }
            if (!budget.take())
                return false;
            base[depth] += 32 * next;
            masks[depth] = candidates(grp, t, g, k, variable, base[depth], objects, binding, state, derived);
            skip = true;
        }
        if (!masks[depth])
        {
            if (!testing)
            {
                testing = true;
                untested = depth;
            }
            --depth;
            continue;
        }
        if (!budget.take())
            return false;
        const u32 pick = ed::lowest_bit64(masks[depth]);
        masks[depth] &= masks[depth] - 1;
        binding[variable] = static_cast<i32>(base[depth] + pick);
        if (!ready(grp, t, g, k, binding, state, derived, variable))
        {
            if (!testing)
            {
                testing = true;
                untested = depth + 1;
            }
            continue;
        }
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
