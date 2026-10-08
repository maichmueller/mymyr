#pragma once

#include "mymyr/search/best_first.hpp"
#include "mymyr/search/iw_family.hpp"

namespace mymyr::search
{
enum class AStarIwFeatures : u8
{
    Classical,
    Abstracted,
    BaseAbstracted,
};

struct AStarIwOptions
{
    SearchControl control;
    heuristics::Options heuristic;
    /// Optional caller-owned evaluator; create one per search thread.
    heuristics::Heuristic* evaluator = nullptr;
    BestFirstOptions::Store store = BestFirstOptions::Store::Auto;
    std::optional<State> start;
    bool witness_pruning = false;
    bool canonical_order = true;
    SymmetryPruning symmetry_pruning = SymmetryPruning::Off;  // as BestFirstOptions::symmetry_pruning
    u32 width = 1;  // classical: 1..5; abstracted modes: 1..3
    AStarIwFeatures features = AStarIwFeatures::Classical;
    LandmarkNovelty landmarks;
    bool preserve_goal_atoms = true;
    bool preserve_landmark_atoms = true;
    double weight = 1.0;  // finite and nonnegative
    bool allow_non_novel_root_goal = true;
    bool probe_novelty_before_heuristic = true;
};

struct AStarIwNoveltyStatistics
{
    u64 rejected = 0;       // generated successors failing novelty
    u64 stale = 0;          // queued states losing all their labels at their g
    u64 stale_g = 0;        // superseded or closed queue entries
    u64 probes = 0;         // read-only transition tests before heuristic evaluation
    u64 updates = 0;        // transition label updates (dead ends never update)
    u64 pop_tests = 0;      // stale-novelty tests on pop
    u64 table_bytes = 0;    // approximate allocated label/feature storage
};

struct AStarIwResult : BestFirstResult
{
    AStarIwNoveltyStatistics novelty;
};

/// Weighted A* with minimum-g novelty pruning, as in mimir. Priority is (g + weight*h, h, g, state id).
/// Requires unit-cost actions and no numeric fluents. Generation lowers tuple labels; a popped state must still
/// own a label at its g. Root successors are admitted without novelty when allow_non_novel_root_goal is set, but
/// only goals bypass the pop test. The initial state is always exempt. Novelty pruning does not imply optimality.
/// Budget, blocked states, goals, cancellation and observers follow best_first.hpp, with on_transition as well.
/// Invalid widths, weights or landmark groupings return Failed with a message; unsupported tasks throw
/// std::invalid_argument. The queue is a heap; the heuristic is evaluated per admitted successor, even on reopening.
[[nodiscard]] AStarIwResult astar_iw(const Task& task, const AStarIwOptions& options = {});
[[nodiscard]] AStarIwResult astar_iw(const Task& task, heuristics::Heuristic& heuristic, const AStarIwOptions& options = {});
}  // namespace mymyr::search
