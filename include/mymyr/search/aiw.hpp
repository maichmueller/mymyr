#pragma once
// Abstracted IW (AIW, BAIW) and abstracted LIW, matching mimir 0.16.3's
// AbstractedNoveltyPruningStrategyImpl run by one brfs::find_solution, and the Python aliases abstracted_iw and
// projective_iw in pymimir/wrapper_search_width.py.
//
//   mymyr::search::AbstractedIwOptions o;
//   o.width = 1;
//   mymyr::search::IwResult r = mymyr::search::abstracted_iw(*task, o);
//
// One breadth-first pass (not a ladder) whose novelty features are abstracted atoms. Per fluent atom p(o_1..o_n):
//   - a preserved atom (a positive fluent goal atom under preserve_goal_atoms, or a landmark atom of the coordinates
//     under preserve_landmark_atoms with a landmark graph) has the full feature p(o_1..o_n); if n <= 1 that is its
//     only feature;
//   - a nullary atom has the full feature p;
//   - otherwise one feature per position i: (p, i, o_i, signature of the other objects), where an object's signature
//     is its declared types ([count, types...], or [1, none] without types); BAIW (base_abstracted) drops the
//     signatures.
// Width 1: a transition is novel iff an added atom has an unseen feature (all of them are marked). Width 2 and 3
// (mimir's generate_tuples): the successor's atoms are groups of features; the tuples are the single features of added
// atoms, the pairs of distinct features from two distinct atoms at least one of which is added and, at width 3, the
// triples likewise; a transition is novel iff one of them is unseen, and all are marked. The start state marks all of
// its tuples. Root rule: every distinct successor of the start state other than itself enters the tree; one that is
// not novel is goal-tested and counted as expanded but not expanded, unless keep_depth_one_novel is set.
// With a landmark graph (abstracted LIW) there is one table per landmark rank, split as in LIW(k): flipped ranks take
// every tuple of the successor, kept ranks the tuples containing an added atom (search/liw.hpp).
//
// Statistics: IwResult::passes holds the one pass (arity = width). mimir reports this search with brfs statuses
// ("solved", "exhausted"). Budgets, blocked states, goal specs, observer events (with on_transition),
// SearchControl::coordination and layer orders are those of the IW family engine (src/mymyr/search/novelty_brfs.hpp).

#include "mymyr/search/iw.hpp"
#include "mymyr/search/iw_family.hpp"

namespace mymyr::search
{
struct AbstractedIwOptions
{
    SearchControl control;
    u32 width = 1;                      // 1, 2 or 3
    bool base_abstracted = false;       // BAIW: no type signatures
    bool preserve_goal_atoms = true;    // positive fluent goal atoms keep their full identity
    bool keep_depth_one_novel = false;  // expand the non-novel successors of the start state too
    /// Abstracted LIW: a landmark graph (and its grouping) splits the tables by landmark rank.
    LandmarkNovelty landmarks;
    bool preserve_landmark_atoms = true;  // with a graph: landmark atoms keep their full identity (mimir's default)
    LayerOrdering layers;
    bool witness_pruning = false;
    bool canonical_order = true;
    std::optional<State> start;
    /// As IwOptions::successor_order.
    std::function<void(StateView state, std::span<const Action> actions, std::vector<u32>& order)> successor_order;
};

/// Abstracted IW(width) / BAIW / abstracted LIW: one pass. Status Failed (with a message) for a width outside 1..3 or
/// an invalid grouping.
[[nodiscard]] IwResult abstracted_iw(const Task& task, const AbstractedIwOptions& options = {});

/// mimir's projective_iw alias: abstracted IW(1) with base_abstracted = !typed_projection and
/// preserve_goal_atoms = keep_goal_nonunary_atoms (width, base_abstracted and preserve_goal_atoms of the base options
/// are overridden).
struct ProjectiveIwOptions : AbstractedIwOptions
{
    bool typed_projection = false;
    bool keep_goal_nonunary_atoms = false;
};
[[nodiscard]] IwResult projective_iw(const Task& task, const ProjectiveIwOptions& options = {});
}  // namespace mymyr::search
