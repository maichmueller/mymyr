#pragma once
// LIW(k): iterated width over landmark-restricted novelty, matching mimir 0.16.3's
// iw::find_solution with Options::landmark_novelty_graph and LandmarkNoveltyPruningStrategyImpl.
//
//   auto graph = std::make_shared<const mymyr::landmarks::FactLandmarkGraph>(
//       mymyr::landmarks::FactLandmarkGraph::create({goal atoms...}));
//   mymyr::search::LiwOptions o;
//   o.max_arity = 1;
//   o.landmarks.graph = graph;
//   mymyr::search::IwResult r = mymyr::search::liw(*task, o);
//
// The ladder runs the width-0 pass (as in iw(), WidthZero) and then LIW(1), ..., LIW(max_arity), each from the start
// state with a fresh table, until a pass solves the task or stops on a budget; an exhausted ladder is
// SearchStatus::Exhausted (mimir's FAILED). There is no optimized IW(1) (mimir disables it with landmarks), so
// IwOptions::optimize_iw1 and IwOptions::tables are ignored. LIW(k) tracks pairs (landmark coordinate, free tuple of
// size <= k) (novelty/landmark_table.hpp): it prunes less than IW(k) and more than IW(k + 1), and still solves every
// task of width <= k. Grouping: one rank per fact landmark; with LandmarkNovelty::disjunctive the graph's
// disjunctive landmarks share a rank each (all_private: one rank per member; unshared_atoms: members taken out of
// the shared ranks). A null graph means no landmarks: every state has the coordinate BOT.
//
// Pass semantics, statistics, budgets, blocked states, goal specs, observer events (plus on_transition with Opened,
// Duplicate or Pruned) and SearchControl::coordination are those of the IW family engine (search/iw.hpp and
// src/mymyr/search/novelty_brfs.hpp). LiwOptions::layers selects an ordered or randomized layer order.

#include "mymyr/search/iw.hpp"
#include "mymyr/search/iw_family.hpp"

namespace mymyr::search
{
struct LiwOptions : IwOptions
{
    LandmarkNovelty landmarks;
};

/// The LIW ladder: the width-0 pass, then LIW(1..max_arity). Status Failed (with a message) for an arity above
/// novelty::k_max_arity or an invalid grouping.
[[nodiscard]] IwResult liw(const Task& task, const LiwOptions& options = {});
}  // namespace mymyr::search
