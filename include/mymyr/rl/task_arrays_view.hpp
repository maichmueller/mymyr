#pragma once
// The device task layout, version 2: a POD view of the flat arrays of
// rl::device_arrays(task, 2) plus the per-item functions that read them. The same code runs on the host (the reference
// and the tests) and on the device (MYMYR_HD): a view holds typed pointers into one block, which is either the host
// ArrayBundle or its uploaded copy (cuda::DeviceTask), so nothing in the layout is a pointer.
//
// This header is part of the device-code subset: it must compile as C++20 under nvcc, so it includes only
// core/types.hpp and uses no standard library beyond <cstdint>/<cstddef>. Keep it that way.
//
// Section "plan" (version 1) of the export, i.e. what version 2 adds to version 1 (all arrays are prefixed plan_):
//   plan_slot_of        u32 [atom_total + 1]  the slot table snapshot (cid -> slot; k_none: no slot; entry atom_total
//                                             is always k_none). Frozen mode: slot = cid (minus F_fluent if derived)
//   plan_rs             u64 [R]               rank*stride position tables (n entries each): the canonical layout's
//                                             positions first, then every static relation's positions
//   plan_static_words   u64 [SW]              static unary bitsets and row tables (TableRef static offsets)
//   plan_static_rel     u64 [P, 6]            per predicate: arity, key space size, rs offset, bits offset | k_none64,
//                                             hash table offset | k_none64, hash mask
//   plan_static_bits / plan_static_table u64  dense membership bits / open-addressing tables (key + 1) of the relations
//   plan_fluent_view_op / plan_derived_view_op u32 [F, 4]  per slot: row1, bit1, row2, bit2 (the view ops, in rows
//                                             of the device's view; a row is k_none when the view does not keep it)
//   plan_view_rows (scalar)                   rows of the per-state view (section plan 2, the compact view): the rows of
//                                             the CPU's ViewLayout that some exported matcher reads and some atom can
//                                             set, in order, then one zero row if a matcher reads a row no atom sets
//   plan_view_map       u32 [VR]              per ViewLayout row: its row in the device's view (k_none: no matcher reads
//                                             it). View TableRefs are ViewLayout rows: row o of reference r is the
//                                             view's row view_map[r + o] (table_row)
//   plan_pattern        u32 [NP, 4]           kind, predicate, first pattern var, var count;  plan_pattern_base u64 [NP]
//   plan_pattern_var    u32 [NV, 2]           parameter, offset of its position table in plan_rs
//   plan_check          u32 [NC, 2]           pattern, polarity
//   plan_matcher        u32 [NM, k_mc_count]  see MatcherCol: scalars plus (offset, count) into the tables below
//   plan_dom0 u64, plan_index u32             per-matcher static domains; per-matcher u32 lists (begin arrays, ...)
//   plan_unary u64 [NU, 2], plan_row u64 [NR, 3], plan_edge u64 [NE, 3], plan_step u32 [NS, 5]
//   plan_schema         u32 [S, k_sc_count]   see SchemaCol
//   plan_cond_effect    u32 [NCE, 5]          condition matcher, adds (offset, count), dels (offset, count)
//   plan_axiom          u32 [NA, 2]           head pattern, body matcher;  plan_stratum u32 [NST, 3] begin, count, recursive
// Offsets inside a matcher (step rows and checks, unary_begin, fc_*_begin) are relative to the matcher's own lists,
// exactly as in plan::Matcher; everything else is an absolute row of its table.

#include "mymyr/core/types.hpp"

namespace mymyr::rl::dev
{
inline constexpr u32 k_none = 0xFFFFFFFFu;
inline constexpr u64 k_none64 = ~u64{0};
inline constexpr u64 k_table_view = u64{1} << 63;  // plan::TableRef: bit 63 = view row, otherwise static word offset

/// Pattern kinds (plan::LitKind).
inline constexpr u32 k_lit_static = 0, k_lit_fluent = 1, k_lit_derived = 2;

/// Columns of plan_matcher. Pairs are (offset, count) into a table; offsets of u32 lists index plan_index.
enum MatcherCol : u32
{
    k_mc_total,
    k_mc_first_free,
    k_mc_flags,  // bit 0 never, bit 1 use_fc, bit 2 binds_in_order, bits 3-5 device_fc, fixed_in_order, witnesses
    k_mc_first_exist,
    k_mc_dom0, k_mc_dom0_n,                        // plan_dom0 (u64 words, total * OW)
    k_mc_unary_begin, k_mc_unary_begin_n,          // plan_index (total + 1)
    k_mc_unary, k_mc_unary_n,                      // plan_unary
    k_mc_pre_checks, k_mc_pre_checks_n,            // plan_check
    k_mc_checks, k_mc_checks_n,                    // plan_check
    k_mc_steps, k_mc_steps_n,                      // plan_step
    k_mc_rows, k_mc_rows_n,                        // plan_row
    k_mc_step_checks, k_mc_step_checks_n,          // plan_index
    k_mc_free_params, k_mc_free_params_n,          // plan_index
    k_mc_relevant, k_mc_relevant_n,                // plan_index (0/1 per parameter)
    k_mc_fc_out_begin, k_mc_fc_out_begin_n,        // plan_index
    k_mc_fc_out, k_mc_fc_out_n,                    // plan_edge
    k_mc_fc_pre, k_mc_fc_pre_n,                    // plan_row
    k_mc_fc_pre_to, k_mc_fc_pre_to_n,              // plan_index
    k_mc_fc_checks_begin, k_mc_fc_checks_begin_n,  // plan_index
    k_mc_fc_checks, k_mc_fc_checks_n,              // plan_index
    k_mc_check_vars_begin, k_mc_check_vars_begin_n,  // plan_index
    k_mc_check_vars, k_mc_check_vars_n,            // plan_index
    k_mc_count
};
inline constexpr u32 k_mc_never = 1, k_mc_use_fc = 2, k_mc_binds_in_order = 4;
/// The device's preferred matcher: fail-first forward checking where rl::device_search_costs puts it at less
/// than 1 / k_fc_margin of the fixed order's work (the export sets it for schema matchers and axiom bodies, never for
/// the conditions of conditional effects, whose sub-plans run in the fixed order, nor for object bitsets wider than
/// k_max_fc_ow words).
inline constexpr u32 k_mc_device_fc = 8;
/// The matcher's fixed order binds the free parameters in parameter order (binds_in_order is that and !use_fc): run in
/// the fixed order, its bindings come out in canonical (lexicographic) order.
inline constexpr u32 k_mc_fixed_in_order = 16;
/// Two or more witness-only parameters: under witness pruning the witness a search emits is the first one in its own
/// order, so the two matchers may label an effect-relevant binding with different witnesses.
inline constexpr u32 k_mc_witnesses = 32;

/// Deepest matcher (parameters) of the device kernels, and the widest object bitsets (words) of their forward checking
/// (a domain stack of (depth + 1) x depth x OW words per thread); cuda/lifted.hpp.
inline constexpr u32 k_max_matcher_depth = 16;
inline constexpr u32 k_max_fc_ow = 4;
/// Deepest matcher of the device's deep kernels (one domain per parameter and a trail; cuda/lifted.hpp).
inline constexpr u32 k_deep_matcher_depth = 32;

/// What fixes the order of a launch's bindings, which decides whether the device may pick its matcher.
enum class MatchOrder : u32
{
    Free,     // canonical order, no binding labels under witness pruning (counts, rows, BrFS)
    Witness,  // canonical order with binding labels (the witnesses must be the CPU's)
    Matcher,  // canonical order off: rows in the plan's matcher order
};
/// The device's matcher for a matcher with flags `flags` (plan_matcher k_mc_flags) in a launch of order `o`: the
/// plan's (k_mc_use_fc) where the order of its bindings is part of the result (MatchOrder::Matcher; Witness for
/// matchers with k_mc_witnesses), the device's preferred one (k_mc_device_fc) otherwise.
[[nodiscard]] MYMYR_HD constexpr bool device_fc(u32 flags, MatchOrder o)
{
    const bool plan = o == MatchOrder::Matcher || (o == MatchOrder::Witness && (flags & k_mc_witnesses) != 0);
    return (flags & (plan ? k_mc_use_fc : k_mc_device_fc)) != 0;
}
/// Whether the device sorts the matcher's bindings under canonical order (its fixed order does not bind in parameter
/// order, or it forward checks).
[[nodiscard]] MYMYR_HD constexpr bool device_sorts(u32 flags, bool fc) { return fc || (flags & k_mc_fixed_in_order) == 0; }

/// Columns of plan_schema.
enum SchemaCol : u32
{
    k_sc_arity,
    k_sc_bind_size,
    k_sc_pre0,  // matcher with witness pruning
    k_sc_pre1,  // matcher without
    k_sc_adds, k_sc_adds_n,          // plan_pattern
    k_sc_dels, k_sc_dels_n,          // plan_pattern
    k_sc_ces, k_sc_ces_n,            // plan_cond_effect
    k_sc_pre_lits, k_sc_pre_lits_n,  // plan_check
    k_sc_count
};

/// Columns of plan_static_rel.
enum StaticRelCol : u32
{
    k_sr_arity,
    k_sr_size,
    k_sr_rs,
    k_sr_bits,
    k_sr_table,
    k_sr_mask,
    k_sr_count
};

/// A read-only view of a version-2 export. Every pointer addresses the same block (host or device).
struct TaskView
{
    // scalars
    u32 num_objects = 0;
    u32 ow = 0;           // words per object bitset
    u32 state_words = 0;  // W of the export (init, goal masks)
    u32 num_preds = 0;
    u32 num_schemas = 0;
    u32 max_bind = 0;     // binding array size any schema, CE or axiom needs
    u32 view_rows = 0;
    u32 fluent_slots = 0, derived_slots = 0;  // slots covered by the snapshot
    u64 atom_total = 0;    // canonical space (fluent + derived)
    u64 fluent_total = 0;  // canonical fluent space
    u32 goal_check = 0, goal_check_n = 0;
    u32 goal_unsatisfiable = 0;
    // v1 arrays used by kernels
    const u64* init = nullptr;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    // section plan
    const u32* slot_of = nullptr;
    const u64* rs = nullptr;
    u64 n_rs = 0;
    const u64* static_words = nullptr;
    u64 n_static_words = 0;
    const u64* static_rel = nullptr;  // [num_preds, k_sr_count]
    const u64* static_bits = nullptr;
    u64 n_static_bits = 0;
    const u64* static_table = nullptr;
    u64 n_static_table = 0;
    const u32* fluent_view_op = nullptr;
    const u32* derived_view_op = nullptr;
    const u32* view_map = nullptr;  // [n_view_map]
    u32 n_view_map = 0;
    const u32* pattern = nullptr;  // [n_patterns, 4]
    const u64* pattern_base = nullptr;
    u32 n_patterns = 0;
    const u32* pattern_var = nullptr;  // [n_pattern_vars, 2]
    u32 n_pattern_vars = 0;
    const u32* check = nullptr;  // [n_checks, 2]
    u32 n_checks = 0;
    const u32* matcher = nullptr;  // [n_matchers, k_mc_count]
    u32 n_matchers = 0;
    const u64* dom0 = nullptr;
    u64 n_dom0 = 0;
    const u32* index = nullptr;
    u64 n_index = 0;
    const u64* unary = nullptr;  // [n_unary, 2]
    u32 n_unary = 0;
    const u64* row = nullptr;  // [n_rows, 3]
    u32 n_rows = 0;
    const u64* edge = nullptr;  // [n_edges, 3]
    u32 n_edges = 0;
    const u32* step = nullptr;  // [n_steps, 5]
    u32 n_steps = 0;
    const u32* schema = nullptr;  // [num_schemas, k_sc_count]
    const u32* cond_effect = nullptr;  // [n_cond_effects, 5]
    u32 n_cond_effects = 0;
    const u32* axiom = nullptr;  // [n_axioms, 2]
    u32 n_axioms = 0;
    const u32* stratum = nullptr;  // [n_strata, 3]
    u32 n_strata = 0;
};

/// The flags (k_mc_flags) of schema s's matcher with (witness) or without witness pruning, and of axiom a's body.
[[nodiscard]] MYMYR_HD u32 schema_matcher_flags(const TaskView& t, u32 s, bool witness)
{
    return t.matcher[u64{t.schema[u64{s} * k_sc_count + (witness ? k_sc_pre0 : k_sc_pre1)]} * k_mc_count + k_mc_flags];
}
[[nodiscard]] MYMYR_HD u32 axiom_matcher_flags(const TaskView& t, u32 a)
{
    return t.matcher[u64{t.axiom[u64{a} * 2 + 1]} * k_mc_count + k_mc_flags];
}

// ------------------------------------------------------------------------------------------------ primitives

[[nodiscard]] MYMYR_HD bool test_bit(const u64* w, u32 n, u64 bit)
{
    const u64 i = bit >> 6;
    return i < n && ((w[i] >> (bit & 63)) & 1) != 0;
}

/// plan::StaticRelation::hash_key.
[[nodiscard]] MYMYR_HD u64 static_hash(u64 k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return k;
}

/// Key of pattern p under a binding: the canonical id (fluent, derived) or the static relation's key.
[[nodiscard]] MYMYR_HD u64 pattern_key(const TaskView& v, u32 p, const u32* bind)
{
    const u32* pt = v.pattern + static_cast<u64>(p) * 4;
    u64 k = v.pattern_base[p];
    const u32* var = v.pattern_var + static_cast<u64>(pt[2]) * 2;
    for (u32 i = 0; i < pt[3]; ++i, var += 2)
        k += v.rs[var[1] + bind[var[0]]];
    return k;
}

/// Slot of canonical id k (k_none if it has none; any k >= atom_total maps to the sentinel entry).
[[nodiscard]] MYMYR_HD u32 slot_of(const TaskView& v, u64 k) { return v.slot_of[k < v.atom_total ? k : v.atom_total]; }

/// Membership of `key` in static predicate `pred` (plan::StaticRelation::contains).
[[nodiscard]] MYMYR_HD bool static_contains(const TaskView& v, u32 pred, u64 key)
{
    const u64* r = v.static_rel + static_cast<u64>(pred) * k_sr_count;
    if (key >= r[k_sr_size])
        return false;
    if (r[k_sr_bits] != k_none64)
        return test_bit(v.static_bits + r[k_sr_bits], static_cast<u32>((r[k_sr_size] + 63) >> 6), key);
    const u64* table = v.static_table + r[k_sr_table];
    const u64 mask = r[k_sr_mask];
    for (u64 j = static_hash(key) & mask;; j = (j + 1) & mask)
    {
        if (table[j] == key + 1)
            return true;
        if (table[j] == 0)
            return false;
    }
}

/// Truth of check c under a binding, compared with its polarity (Engine::holds). `derived` may be null when the
/// task has no derived predicates (a derived literal then reads as false).
[[nodiscard]] MYMYR_HD bool holds(const TaskView& v, u32 c, const u32* bind, const u64* state, u32 nw, const u64* derived,
                                  u32 ndw)
{
    const u32 p = v.check[static_cast<u64>(c) * 2];
    const bool pos = v.check[static_cast<u64>(c) * 2 + 1] != 0;
    const u32 kind = v.pattern[static_cast<u64>(p) * 4];
    const u64 k = pattern_key(v, p, bind);
    bool t;
    if (kind == k_lit_fluent)
        t = test_bit(state, nw, slot_of(v, k));
    else if (kind == k_lit_derived)
        t = derived != nullptr && test_bit(derived, ndw, slot_of(v, k));
    else
        t = static_contains(v, v.pattern[static_cast<u64>(p) * 4 + 1], k);
    return t == pos;
}

/// Applicability of a ground action (schema s, full binding) in a state: every precondition literal holds
/// (Successors::is_applicable). The binding must hold the schema's arity values, each < num_objects.
[[nodiscard]] MYMYR_HD bool is_applicable(const TaskView& v, u32 s, const u32* bind, const u64* state, u32 nw,
                                          const u64* derived, u32 ndw)
{
    const u32* sc = v.schema + static_cast<u64>(s) * k_sc_count;
    for (u32 i = 0; i < sc[k_sc_pre_lits_n]; ++i)
        if (!holds(v, sc[k_sc_pre_lits] + i, bind, state, nw, derived, ndw))
            return false;
    return true;
}

/// Status codes of apply_action.
inline constexpr u32 k_apply_ok = 0;
inline constexpr u32 k_apply_conditional = 1;  // the schema has conditional effects: CE sub-plans run on the device separately
inline constexpr u32 k_apply_no_slot = 2;      // an added atom has no slot yet (lazy slots: interning is host-side)
inline constexpr u32 k_apply_width = 3;        // an added atom lies beyond the output width

/// Successor words of `state` under an applicable action whose schema has no conditional effects: delete, then add
/// (Successors::apply). Writes out_words words (state words beyond nw read as zero).
[[nodiscard]] MYMYR_HD u32 apply_action(const TaskView& v, u32 s, const u32* bind, const u64* state, u32 nw, u64* out,
                                        u32 out_words)
{
    const u32* sc = v.schema + static_cast<u64>(s) * k_sc_count;
    if (sc[k_sc_ces_n] != 0)
        return k_apply_conditional;
    for (u32 i = 0; i < out_words; ++i)
        out[i] = i < nw ? state[i] : 0;
    for (u32 i = 0; i < sc[k_sc_dels_n]; ++i)
    {
        const u32 slot = slot_of(v, pattern_key(v, sc[k_sc_dels] + i, bind));
        if (slot != k_none && (slot >> 6) < out_words)
            out[slot >> 6] &= ~(u64{1} << (slot & 63));
    }
    for (u32 i = 0; i < sc[k_sc_adds_n]; ++i)
    {
        const u32 slot = slot_of(v, pattern_key(v, sc[k_sc_adds] + i, bind));
        if (slot == k_none)
            return k_apply_no_slot;
        if ((slot >> 6) >= out_words)
            return k_apply_width;
        out[slot >> 6] |= u64{1} << (slot & 63);
    }
    return k_apply_ok;
}

/// Goal test (Successors::goal_holds): derived goal literals need the state's derived bitset.
[[nodiscard]] MYMYR_HD bool goal_holds(const TaskView& v, const u64* state, u32 nw, const u64* derived, u32 ndw)
{
    if (v.goal_unsatisfiable)
        return false;
    for (u32 i = 0; i < v.goal_check_n; ++i)
        if (!holds(v, v.goal_check + i, nullptr, state, nw, derived, ndw))
            return false;
    return true;
}

// ------------------------------------------------------------------------------------------------ validation
// Structural checks of every table (bounds of all offsets and references). Items are numbered over the tables so a
// kernel can check them in parallel: [patterns | checks | matchers | schemas | cond effects | axioms | strata |
// fluent view ops | derived view ops | static relations]. Returns 0 or the number of the first failed rule (> 0).

[[nodiscard]] MYMYR_HD u64 validate_items(const TaskView& v)
{
    return u64{v.n_patterns} + v.n_checks + v.n_matchers + v.num_schemas + v.n_cond_effects + v.n_axioms + v.n_strata +
           v.fluent_slots + v.derived_slots + v.num_preds;
}

namespace detail
{
[[nodiscard]] MYMYR_HD bool in(u64 off, u64 n, u64 size) { return off <= size && n <= size - off; }

/// A table reference read at `rows` rows (1: a unary reference; num_objects: an object-indexed one): every view row it
/// reaches is a row of the view.
[[nodiscard]] MYMYR_HD bool table_ok(const TaskView& v, u64 ref, u64 rows)
{
    if (!(ref & k_table_view))
        return in(ref, rows * v.ow, v.n_static_words);
    const u64 r = ref & ~k_table_view;
    if (!in(r, rows, v.n_view_map))
        return false;
    for (u64 i = 0; i < rows; ++i)
        if (v.view_map[r + i] >= v.view_rows)
            return false;
    return true;
}

[[nodiscard]] MYMYR_HD u32 validate_pattern(const TaskView& v, u32 p)
{
    const u32* pt = v.pattern + static_cast<u64>(p) * 4;
    if (pt[0] > k_lit_derived || pt[1] >= v.num_preds)
        return 1;
    if (!in(pt[2], pt[3], v.n_pattern_vars))
        return 2;
    for (u32 i = 0; i < pt[3]; ++i)
    {
        const u32* var = v.pattern_var + (static_cast<u64>(pt[2]) + i) * 2;
        if (var[0] >= v.max_bind || !in(var[1], v.num_objects, v.n_rs))
            return 3;
    }
    if (pt[0] == k_lit_static && v.static_rel[static_cast<u64>(pt[1]) * k_sr_count + k_sr_arity] < pt[3])
        return 4;
    return 0;
}

[[nodiscard]] MYMYR_HD u32 validate_matcher(const TaskView& v, u32 m)
{
    const u32* x = v.matcher + static_cast<u64>(m) * k_mc_count;
    const u32 total = x[k_mc_total];
    if (x[k_mc_first_free] > total || total > v.max_bind)
        return 10;
    if (x[k_mc_dom0_n] != total * v.ow || !in(x[k_mc_dom0], x[k_mc_dom0_n], v.n_dom0))
        return 11;
    const u32 lists[] = {k_mc_unary_begin, k_mc_step_checks, k_mc_free_params, k_mc_relevant, k_mc_fc_out_begin,
                         k_mc_fc_pre_to, k_mc_fc_checks_begin, k_mc_fc_checks, k_mc_check_vars_begin, k_mc_check_vars};
    for (u32 c : lists)
        if (!in(x[c], x[c + 1], v.n_index))
            return 12;
    if (!in(x[k_mc_unary], x[k_mc_unary_n], v.n_unary) || !in(x[k_mc_pre_checks], x[k_mc_pre_checks_n], v.n_checks) ||
        !in(x[k_mc_checks], x[k_mc_checks_n], v.n_checks) || !in(x[k_mc_steps], x[k_mc_steps_n], v.n_steps) ||
        !in(x[k_mc_rows], x[k_mc_rows_n], v.n_rows) || !in(x[k_mc_fc_out], x[k_mc_fc_out_n], v.n_edges) ||
        !in(x[k_mc_fc_pre], x[k_mc_fc_pre_n], v.n_rows))
        return 13;
    const u32* ix = v.index;
    // unary_begin: total + 1 monotone entries within the matcher's unary list
    if (x[k_mc_unary_begin_n] != total + 1)
        return 14;
    for (u32 i = 0; i <= total; ++i)
    {
        const u32 b = ix[x[k_mc_unary_begin] + i];
        if (b > x[k_mc_unary_n] || (i > 0 && b < ix[x[k_mc_unary_begin] + i - 1]))
            return 15;
    }
    for (u32 i = 0; i < x[k_mc_unary_n]; ++i)
        if (!table_ok(v, v.unary[(static_cast<u64>(x[k_mc_unary]) + i) * 2], 1))
            return 16;
    // steps: parameters, row and check ranges relative to the matcher
    if (x[k_mc_first_exist] > x[k_mc_steps_n])
        return 17;
    for (u32 i = 0; i < x[k_mc_steps_n]; ++i)
    {
        const u32* st = v.step + (static_cast<u64>(x[k_mc_steps]) + i) * 5;
        if (st[0] >= total || st[1] > st[2] || st[2] > x[k_mc_rows_n] || st[3] > st[4] || st[4] > x[k_mc_step_checks_n])
            return 18;
    }
    for (u32 i = 0; i < x[k_mc_step_checks_n]; ++i)
        if (ix[x[k_mc_step_checks] + i] >= x[k_mc_checks_n])
            return 19;
    const u64 n = v.num_objects;
    for (u32 i = 0; i < x[k_mc_rows_n]; ++i)
    {
        const u64* r = v.row + (static_cast<u64>(x[k_mc_rows]) + i) * 3;
        if (!table_ok(v, r[0], n) || r[1] >= total)
            return 20;
    }
    // forward checking
    for (u32 i = 0; i < x[k_mc_free_params_n]; ++i)
        if (ix[x[k_mc_free_params] + i] >= total)
            return 21;
    for (u32 i = 0; i < x[k_mc_fc_out_n]; ++i)
    {
        const u64* e = v.edge + (static_cast<u64>(x[k_mc_fc_out]) + i) * 3;
        if (e[0] >= total || !table_ok(v, e[1], n))
            return 22;
    }
    if (x[k_mc_fc_out_begin_n] != 0)
    {
        if (x[k_mc_fc_out_begin_n] != total + 1)
            return 23;
        for (u32 i = 0; i <= total; ++i)
            if (ix[x[k_mc_fc_out_begin] + i] > x[k_mc_fc_out_n])
                return 23;
    }
    for (u32 i = 0; i < x[k_mc_fc_pre_n]; ++i)
    {
        const u64* r = v.row + (static_cast<u64>(x[k_mc_fc_pre]) + i) * 3;
        if (!table_ok(v, r[0], n) || r[1] >= total)
            return 24;
    }
    if (x[k_mc_fc_pre_to_n] != x[k_mc_fc_pre_n])
        return 25;
    for (u32 i = 0; i < x[k_mc_fc_pre_to_n]; ++i)
        if (ix[x[k_mc_fc_pre_to] + i] >= total)
            return 25;
    if (x[k_mc_fc_checks_begin_n] != 0)
    {
        if (x[k_mc_fc_checks_begin_n] != total + 1)
            return 26;
        for (u32 i = 0; i <= total; ++i)
            if (ix[x[k_mc_fc_checks_begin] + i] > x[k_mc_fc_checks_n])
                return 26;
    }
    for (u32 i = 0; i < x[k_mc_fc_checks_n]; ++i)
        if (ix[x[k_mc_fc_checks] + i] >= x[k_mc_checks_n])
            return 27;
    if (x[k_mc_check_vars_begin_n] != 0)
    {
        if (x[k_mc_check_vars_begin_n] != x[k_mc_checks_n] + 1)
            return 28;
        for (u32 i = 0; i <= x[k_mc_checks_n]; ++i)
            if (ix[x[k_mc_check_vars_begin] + i] > x[k_mc_check_vars_n])
                return 28;
    }
    for (u32 i = 0; i < x[k_mc_check_vars_n]; ++i)
        if (ix[x[k_mc_check_vars] + i] >= total)
            return 29;
    return 0;
}
}  // namespace detail

/// Checks item i (0 <= i < validate_items(v)); returns 0 or the number of the failed rule.
[[nodiscard]] MYMYR_HD u32 validate_item(const TaskView& v, u64 i)
{
    if (i < v.n_patterns)
        return detail::validate_pattern(v, static_cast<u32>(i));
    i -= v.n_patterns;
    if (i < v.n_checks)
    {
        const u32* c = v.check + i * 2;
        return c[0] < v.n_patterns && c[1] <= 1 ? 0 : 5;
    }
    i -= v.n_checks;
    if (i < v.n_matchers)
        return detail::validate_matcher(v, static_cast<u32>(i));
    i -= v.n_matchers;
    if (i < v.num_schemas)
    {
        const u32* s = v.schema + i * k_sc_count;
        if (s[k_sc_arity] > s[k_sc_bind_size] || s[k_sc_bind_size] > v.max_bind || s[k_sc_pre0] >= v.n_matchers ||
            s[k_sc_pre1] >= v.n_matchers)
            return 30;
        if (!detail::in(s[k_sc_adds], s[k_sc_adds_n], v.n_patterns) || !detail::in(s[k_sc_dels], s[k_sc_dels_n], v.n_patterns) ||
            !detail::in(s[k_sc_ces], s[k_sc_ces_n], v.n_cond_effects) ||
            !detail::in(s[k_sc_pre_lits], s[k_sc_pre_lits_n], v.n_checks))
            return 31;
        return 0;
    }
    i -= v.num_schemas;
    if (i < v.n_cond_effects)
    {
        const u32* c = v.cond_effect + i * 5;
        return c[0] < v.n_matchers && detail::in(c[1], c[2], v.n_patterns) && detail::in(c[3], c[4], v.n_patterns) ? 0 : 32;
    }
    i -= v.n_cond_effects;
    if (i < v.n_axioms)
    {
        const u32* a = v.axiom + i * 2;
        return a[0] < v.n_patterns && a[1] < v.n_matchers ? 0 : 33;
    }
    i -= v.n_axioms;
    if (i < v.n_strata)
    {
        const u32* s = v.stratum + i * 3;
        return detail::in(s[0], s[1], v.n_axioms) && s[2] <= 1 ? 0 : 34;
    }
    i -= v.n_strata;
    if (i < u64{v.fluent_slots} + v.derived_slots)
    {
        const u32* op = i < v.fluent_slots ? v.fluent_view_op + i * 4 : v.derived_view_op + (i - v.fluent_slots) * 4;
        const bool ok = (op[0] == k_none || (op[0] < v.view_rows && op[1] < v.num_objects)) &&
                        (op[2] == k_none || (op[2] < v.view_rows && op[3] < v.num_objects));
        return ok ? 0 : 35;
    }
    i -= u64{v.fluent_slots} + v.derived_slots;
    if (i < v.num_preds)
    {
        const u64* r = v.static_rel + i * k_sr_count;
        if (r[k_sr_arity] == 0 && r[k_sr_size] == 0)
            return 0;  // not a static predicate
        if (!detail::in(r[k_sr_rs], r[k_sr_arity] * v.num_objects, v.n_rs))
            return 36;
        if (r[k_sr_bits] != k_none64)
            return detail::in(r[k_sr_bits], (r[k_sr_size] + 63) >> 6, v.n_static_bits) ? 0 : 37;
        return r[k_sr_table] != k_none64 && detail::in(r[k_sr_table], r[k_sr_mask] + 1, v.n_static_table) ? 0 : 38;
    }
    return 39;
}
}  // namespace mymyr::rl::dev
