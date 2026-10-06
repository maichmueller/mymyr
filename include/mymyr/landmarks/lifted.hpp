#pragma once
// Lifted fact landmarks: necessary subgoals over partially ground atoms, straight from the schemas (Wichlacz, Höller &
// Hoffmann, "Landmark Heuristics for Lifted Classical Planning", IJCAI 2022, §3.1), with mimir 0.16.3's
// delete-relaxed reachability refinements.
//
//   auto g = mymyr::landmarks::lifted_fact_landmarks(*task);
//   for (const auto& l : g.lifted()) std::cout << mymyr::landmarks::format_lifted(*task, l) << "\n";
//
// Back-chaining from the positive fluent goal atoms: for a pattern P(u) (free positions allowed) the achievers are the
// (schema, conditional effect, add literal) triples unifying with it; for every fluent predicate Q that every achiever's
// positive fluent precondition mentions, the position-wise agreement of one chosen occurrence per achiever is a
// landmark pattern, with the member atoms its achievers' completions can produce. The reading is mimir's:
//   - a landmark with a member true initially is recorded, never expanded;
//   - achievers by unification (type-compatible objects, repeated variables agree);
//   - static filter (use_static_filter): static literals narrow the candidate objects of free parameters to a
//     fixpoint (PDDL "=" on object identity), a parameter with one candidate is bound, an achiever without a
//     consistent instance is dropped;
//   - intersection over achievers by predicate, in ascending predicate order, over up to
//     max_occurrence_combinations choice vectors; a pattern subsumed by an earlier more specific one (or, under
//     member-set identity, a member set containing an earlier one's) is dropped;
//   - members: completions over the candidates that are statically consistent, that some schema can add or that
//     hold initially, that are relaxed reachable (reachability_filter_members) and, when first_achievers_restricted
//     is set, reachable without the parent's members; a single member makes the landmark a fact;
//   - (only without first_achievers_restricted) the self-dependent-precondition rule: when no instance of the
//     pattern holds initially, a precondition every adder of which needs the pattern must use an initial atom;
//   - reachability narrowing of the free parameters: per literal, or jointly (the projection of the whole
//     precondition conjunction over the relaxed reachable atoms);
//   - first achievers (first_achievers_restricted): the achiever's pattern positions must agree with some member
//     and its preconditions must hold in R_{¬M}, the relaxed fixpoint that never reaches a member M (witness
//     shortcut first, one restricted fixpoint when needed); records are identified by their member sets;
//   - verify_pi_plus: every fact landmark that is neither a goal atom nor initially true must block the goal once
//     forbidden in the relaxed fixpoint (std::logic_error otherwise);
//   - complete_fact_landmarks: atoms (members of the disjunctive records, or every reachable non-initial atom)
//     whose removal blocks the relaxed goal become fact landmarks.
// Orderings: greedy-necessary between fact landmarks only (a partial landmark's parents are in LiftedLandmark::parents).
// The graph carries no achiever index. Disjunctive landmarks: the member sets of the partial records with at least two
// members (at most max_disjunctive_members; 0 = no bound), minus strict supersets of another set under member-set
// identity; sets holding a fact landmark are dropped.

#include "mymyr/landmarks/fact_landmark_graph.hpp"

namespace mymyr::reachability
{
class RelaxedReachability;
}

namespace mymyr::landmarks
{
enum class ReachabilityDisambiguation : u8
{
    Off,         // no narrowing beyond the static filter
    PerLiteral,  // each precondition literal narrows its own variables against the reachable atoms
    Joint,       // the projection of the whole precondition conjunction
};

enum class CompleteFactLandmarks : u8
{
    Off,
    Members,  // every member of every partial record
    All,      // every relaxed reachable atom not true initially (one restricted fixpoint each)
};

struct LiftedFactLandmarkOptions
{
    bool include_positive_goal_facts = true;
    bool compute_greedy_necessary_orderings = true;
    bool use_static_filter = true;
    usize max_occurrence_combinations = 64;
    usize max_disjunctive_members = 0;  // 0: no bound
    bool reachability_filter_members = true;
    ReachabilityDisambiguation reachability_disambiguation = ReachabilityDisambiguation::Joint;
    bool first_achievers_restricted = true;
    bool verify_pi_plus = true;
    CompleteFactLandmarks complete_fact_landmarks = CompleteFactLandmarks::Members;
};

/// Whether any option needs the relaxed reachability engine.
[[nodiscard]] bool needs_relaxed_reachability(const LiftedFactLandmarkOptions& options);

/// Lifted fact landmarks of `task`. Throws std::logic_error when verify_pi_plus finds a fact that is not a landmark,
/// or when an internal consistency check fails (an empty member set, first_achievers_restricted dropping every
/// achiever of a reachable landmark).
[[nodiscard]] FactLandmarkGraph lifted_fact_landmarks(const Task& task, const LiftedFactLandmarkOptions& options = {});

/// The same over an existing engine for the same task (built with the default options).
[[nodiscard]] FactLandmarkGraph lifted_fact_landmarks(const reachability::RelaxedReachability& engine,
                                                      const LiftedFactLandmarkOptions& options = {});

/// verify_pi_plus on any graph: throws std::logic_error naming the first fact landmark (not a positive goal atom, not initially
/// true) whose removal leaves the relaxed goal reachable. Does nothing when the relaxed goal is unreachable.
void verify_pi_plus_fact_landmarks(const Task& task, const FactLandmarkGraph& graph);
void verify_pi_plus_fact_landmarks(const reachability::RelaxedReachability& engine, const FactLandmarkGraph& graph);
}  // namespace mymyr::landmarks
