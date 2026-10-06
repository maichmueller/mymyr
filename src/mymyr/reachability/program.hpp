#pragma once
// Internal: the compiled rules of the relaxed reachability fixpoint and the data of one fixpoint's table.

#include "join.hpp"
#include "keymap.hpp"

#include "mymyr/reachability/relaxed_reachability.hpp"

#include <memory>
#include <vector>

namespace mymyr::reachability::detail
{
using reach::k_none;

/// The canonical id of an atom pattern under a binding: base + sum of rs[bind[var]].
struct KeyPattern
{
    u32 pred = 0;
    u64 base = 0;
    bool never = false;
    std::vector<reach::KeyVar> vars;
    std::vector<i32> terms;

    [[nodiscard]] u64 key(const u32* bind) const noexcept
    {
        u64 k = base;
        for (const reach::KeyVar& v : vars)
            k += v.rs[bind[v.var]];
        return k;
    }
};

struct Rule
{
    std::string name;       // provenance: schema name or "axiom:<head>"
    u32 schema = k_none;    // source schema (k_none for axioms)
    u32 axiom = k_none;     // source axiom
    u32 num_vars = 0;
    KeyPattern head;
    std::vector<reach::Lit> body;
    std::vector<KeyPattern> rel;           // relation literals, by slot
    std::vector<reach::Plan> plans;        // [0] unanchored, [1 + slot] anchored at slot, [1 + |rel|] the seed plan
    // Round 0 runs the unanchored plan, or, when a relation literal has arity >= 3 (a check the unanchored plan could
    // only test after enumerating its variables), the seed plan: anchored at that literal over the initial atoms, every
    // other literal reading New.
    u32 seed_slot = k_none;
};

struct Anchor
{
    u32 rule = 0;
    u32 slot = 0;
    u32 lit = 0;  // the anchor literal's index in Rule::body
    // Projection dedup: a variable of the anchor literal that no other literal and not the head mentions does not change
    // what the anchored plan derives, so a round runs the plan once per distinct tuple of the other anchor variables
    // (key_vars; packed with `id` into one word when that fits, else no dedup). Mimir's join projects such
    // variables away.
    bool dedup = false;
    u32 id = 0;
    std::vector<u32> key_vars;
    std::vector<u32> free_vars;  // the anchor-only variables (tested against their domain before the dedup)
};

struct Program
{
    u64 id = 0;  // process-unique (keys the per-thread fixpoint scratch)
    const Task* task = nullptr;
    Options options;
    std::unique_ptr<reach::StaticTables> st;
    u64 total = 0;           // canonical atoms
    u64 fluent_count = 0;
    std::vector<Rule> rules;
    std::vector<std::vector<Anchor>> anchors;  // per predicate
    std::vector<std::vector<Anchor>> seed_anchors;  // per predicate: rules seeded from its initial atoms (Rule::seed_slot)
    std::vector<u32> seed_rules;               // rules without relation literals (round 0 only)
    std::vector<u64> initial;                  // canonical ids of the initial fluent atoms
    bool static_goal = true;
    std::vector<u64> goal;                     // positive fluent and derived goal atoms, distinct (~0: outside the space)
    std::vector<u64> goal_bits;                // canonical bitset of `goal`
    std::vector<u32> relations;                // fluent and derived predicates with a canonical block
    u32 obj_bits = 1, id_bits = 1;             // bits per object id and for the anchor id in a dedup key (Anchor)
};

struct TableData
{
    reach::View view;            // the reached atoms (rows and canonical bitset)
    std::vector<std::vector<CanonicalAtom>> atoms;  // per predicate, derivation order
    std::vector<std::vector<u32>> tuples;           // per predicate, flat
    u64 fluent = 0, derived = 0;
    bool goal = false;
    u32 rounds = 0;
    // witnesses: per reached atom in derivation order (position), the rule (k_none: initial) and its binding
    bool witnesses = false;
    reach::KeyMap<u32> position;   // canonical id -> position
    std::vector<u32> wit_rule;
    std::vector<u64> wit_begin;    // positions + 1
    std::vector<u32> wit_bind;
};

/// Runs the fixpoint. forbidden: canonical ids never reached; stop_at_goal: return as soon as the goal holds.
std::unique_ptr<TableData> run_fixpoint(const Program& p, std::span<const CanonicalAtom> forbidden, bool stop_at_goal, bool record_witnesses,
                                        u64* bindings);
/// Whether the goal is reachable without `forbidden` (the fixpoint stops at the goal; no table is kept).
bool run_goal_query(const Program& p, std::span<const CanonicalAtom> forbidden);
}  // namespace mymyr::reachability::detail
