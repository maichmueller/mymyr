#pragma once
// Symmetry pruning of the applicable actions, as mimir's SymmetryPruning::WL1 mode of its lifted (KPKC) successor
// generator. Per state, the objects are partitioned into the colour classes of colour refinement (1-WL) on the state's
// object graph (datasets/object_graph.hpp: the state's atoms, the static atoms and the goal); then every action
// parameter is restricted to representatives: for a schema, let n(C) be the number of its parameters whose static
// domain meets class C; parameter i keeps, of each class C, the n(C) smallest objects of its static domain (by object
// id). An applicable action is kept iff every parameter is bound to a kept object.
//
// Guarantees: colour classes coarsen the automorphism orbits (two objects of one class need not be exchangeable), and
// even true orbits are only a per-object symmetry, so the pruned actions need not be symmetric to a kept one. A
// search with pruning on may therefore miss every plan of a solvable task (it reports it unsolvable), and A* may
// return a costlier plan. A plan it finds consists of real, applicable ground actions and is valid.
//
// Deviation from mimir: mimir's colour refinement never separates a vertex without neighbours from the other vertices
// of its colour (datasets/certificates.hpp), so an object that occurs in no atom or goal literal of arity >= 2 can
// share a class with objects that do; here every vertex is refined, and the classes are those of exact colour
// refinement. Where the two partitions agree, the kept actions are mimir's.

#include "mymyr/core/types.hpp"

#include <optional>
#include <string_view>

namespace mymyr
{
enum class SymmetryPruning : u8
{
    Off,
    Wl1,  // restrict action parameters to representatives of the colour refinement classes
};

/// "off" or "wl1".
[[nodiscard]] constexpr std::string_view to_string(SymmetryPruning p) noexcept { return p == SymmetryPruning::Wl1 ? "wl1" : "off"; }

/// The mode named "off" or "wl1" (case-sensitive), or nullopt.
[[nodiscard]] constexpr std::optional<SymmetryPruning> parse_symmetry_pruning(std::string_view name) noexcept
{
    if (name == "off")
        return SymmetryPruning::Off;
    if (name == "wl1")
        return SymmetryPruning::Wl1;
    return std::nullopt;
}
}  // namespace mymyr
