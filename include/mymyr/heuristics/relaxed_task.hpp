#pragma once
// The grounded delete relaxation behind h_max, h_add and h_FF.
//
// Grounding follows mimir's LiftedGrounder + RelaxedPlanningGraph (v0.16.3) so that the heuristic values equal
// mimir's:
//   1. Exploration: the delete-free task (every negative literal and every delete dropped, derived predicates made
//      fluent, each axiom an action adding its head) is explored from the initial state with mymyr's own lifted
//      successor generator, in naive rounds until no new atom appears (mimir's delete-free fixpoint). R is the set of
//      reached atoms.
//   2. Ground actions G: every binding of every schema that is applicable in the final relaxed state and statically
//      applicable in the unrelaxed task (static literals of both polarities hold, no atom required both true and
//      false); ground axioms likewise.
//   3. Operators: per ground action and per conditional effect instance one operator with precondition
//      pre(action) + fluent/derived literals of the effect condition and effect "add atoms / delete atoms". As in
//      mimir, the static literals of an effect condition are ignored and the instances of a universally quantified
//      effect are the product of each quantified parameter's objects that pass mimir's single-parameter static
//      consistency test (StaticConsistencyGraph::compute_vertices). Axioms become 0-cost operators.
//   4. Propositions: an atom true (fluent or derived) and, for fluent atoms of R used negatively somewhere, an atom
//      false. "p false" costs 0 in a state without p and is otherwise achieved by operators deleting p. Negated
//      derived atoms always cost 0 (mimir's "not y <- T"). Preconditions keep their multiplicity (h_add counts a
//      repeated atom twice, as mimir does).
//
// Everything is flat CSR, built once and shared read-only by every evaluator (one per thread). Construction respects a
// budget; beyond it build() returns nullptr and heuristics fall back to lifted evaluation.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/successor/action.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::heuristics
{
struct GroundingBudget
{
    u64 max_operators = 5'000'000;    // relaxed operators (one per ground action and effect instance, plus axioms)
    u64 max_bindings = 100'000'000;   // relaxed bindings enumerated by the exploration rounds
    double max_seconds = 60;          // wall time of the whole grounding
};

struct GroundingStats
{
    u64 reached_atoms = 0;   // |R|: fluent and derived atoms of the delete-free fixpoint
    u64 ground_actions = 0;  // |G|
    u64 ground_axioms = 0;
    u64 operators = 0;
    u64 propositions = 0;
    u64 bindings = 0;  // enumerated by the exploration
    u32 rounds = 0;
    double seconds = 0;
    bool over_budget = false;
    std::string reason;  // why the budget stopped the grounding
};

struct RelaxedTaskOptions
{
    /// Keep the operators with a positive precondition no operator can add and that is not reached (mimir's
    /// RPG keeps them: they never fire, but their positive effects count as achievers in the landmark generator's
    /// achiever index). The heuristics drop them.
    bool keep_unreachable_operators = false;
};

class RelaxedTask
{
public:
    static constexpr u32 k_none = ~u32{0};

    /// Grounds `task` within `budget`. Returns nullptr when the budget is exceeded (stats->reason says why).
    static std::shared_ptr<const RelaxedTask> build(const Task& task, const GroundingBudget& budget = {}, GroundingStats* stats = nullptr);
    static std::shared_ptr<const RelaxedTask> build(const Task& task, const GroundingBudget& budget, GroundingStats* stats,
                                                    const RelaxedTaskOptions& options);

    [[nodiscard]] const Task& task() const noexcept { return *m_task; }
    [[nodiscard]] const GroundingStats& stats() const noexcept { return m_stats; }

    // ------------------------------------------------------------------------------------ propositions
    [[nodiscard]] u32 num_props() const noexcept { return static_cast<u32>(m_prop_negative.size()); }
    [[nodiscard]] bool negative(u32 prop) const noexcept { return m_prop_negative[prop] != 0; }
    /// Propositions of a fluent or derived atom of the task by canonical id: first = "true" (k_none if the atom is not in
    /// R), second = "false" (k_none unless the atom is a fluent atom of R used negatively).
    [[nodiscard]] std::pair<u32, u32> fluent_props(CanonicalAtom cid) const noexcept;
    /// Canonical id of the atom of a proposition ("true" or "false"); ~0 for an atom outside the canonical layout.
    [[nodiscard]] CanonicalAtom prop_atom(u32 prop) const noexcept { return m_prop_atom[prop]; }

    // ------------------------------------------------------------------------------------ operators
    [[nodiscard]] u32 num_ops() const noexcept { return static_cast<u32>(m_npos.size()); }
    [[nodiscard]] std::span<const u32> pre(u32 op) const noexcept
    {
        return {m_pre.data() + m_pre_begin[op], m_pre_begin[op + 1] - m_pre_begin[op]};
    }
    [[nodiscard]] std::span<const u32> eff(u32 op) const noexcept
    {
        return {m_eff.data() + m_eff_begin[op], m_eff_begin[op + 1] - m_eff_begin[op]};
    }
    /// Precondition entries that are not "false" propositions (the counter of unsatisfied entries starts here).
    [[nodiscard]] u32 npos(u32 op) const noexcept { return m_npos[op]; }
    [[nodiscard]] bool is_axiom(u32 op) const noexcept { return m_ga[op] == k_none; }
    /// Ground action of an operator (k_none for axioms).
    [[nodiscard]] u32 ground_action(u32 op) const noexcept { return m_ga[op]; }
    /// Operators having `prop` in their precondition (with multiplicity).
    [[nodiscard]] std::span<const u32> pre_of(u32 prop) const noexcept
    {
        return {m_pre_of.data() + m_pre_of_begin[prop], m_pre_of_begin[prop + 1] - m_pre_of_begin[prop]};
    }
    /// Operators without "true" precondition entries.
    [[nodiscard]] std::span<const u32> zero_ops() const noexcept { return m_zero_ops; }

    // ------------------------------------------------------------------------------------ ground actions
    [[nodiscard]] u32 num_ground_actions() const noexcept { return static_cast<u32>(m_ga_schema.size()); }
    [[nodiscard]] ActionLabel ground_action_label(u32 ga) const noexcept;
    /// Id of ground action (schema, binding) or k_none.
    [[nodiscard]] u32 find_ground_action(const ActionLabel& a) const noexcept;
    /// Action cost (the task's total-cost semantics); real costs are available when every one is a non-negative
    /// integer below 2^31.
    [[nodiscard]] bool real_costs_available() const noexcept { return m_real_ok; }
    [[nodiscard]] u32 real_cost(u32 ga) const noexcept { return m_ga_cost[ga]; }

    // ------------------------------------------------------------------------------------ goal
    /// The task goal's propositions (distinct; static goal literals, negated derived atoms and negated atoms outside R
    /// are left out). goal_unreachable(): a positive goal atom has no proposition at all.
    [[nodiscard]] std::span<const u32> goal() const noexcept { return m_goal; }
    [[nodiscard]] bool goal_unreachable() const noexcept { return m_goal_unreachable; }

    struct Private
    {
    };
    RelaxedTask(Private, const Task& task) : m_task(&task) {}

private:
    friend class Grounder;

    const Task* m_task;
    GroundingStats m_stats;
    std::vector<u8> m_prop_negative;
    std::vector<CanonicalAtom> m_prop_atom;
    // canonical id of a task fluent atom -> (true prop, false prop): open addressing
    std::vector<u64> m_cid_keys;  // cid + 1, 0 = empty
    std::vector<u32> m_cid_pos, m_cid_neg;
    u64 m_cid_mask = 0;
    // operators
    std::vector<u32> m_pre_begin{0}, m_pre, m_eff_begin{0}, m_eff, m_npos, m_ga;
    std::vector<u32> m_pre_of_begin, m_pre_of, m_zero_ops;
    // ground actions
    std::vector<u32> m_ga_schema;
    std::vector<u32> m_ga_bind_begin;
    std::vector<ObjectId> m_ga_bind;
    std::vector<u32> m_ga_cost;
    bool m_real_ok = false;
    std::vector<u64> m_ga_keys;  // hash of (schema, binding) -> ga + 1 (open addressing)
    std::vector<u32> m_ga_slots;
    u64 m_ga_mask = 0;
    // goal
    std::vector<u32> m_goal;
    bool m_goal_unreachable = false;
};
}  // namespace mymyr::heuristics
