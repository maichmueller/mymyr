#pragma once
// Helpers shared by the IW family variants (aiw.cpp, liw.cpp, rollouts, portfolio).

#include "novelty_brfs.hpp"

#include "mymyr/novelty/landmark_table.hpp"
#include "mymyr/search/iw_family.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mymyr::search::detail
{
/// Slots of canonical fluent atoms (interned on first touch); atoms outside the fluent space are dropped.
inline std::vector<u32> fluent_slots_of(const Task& task, std::span<const CanonicalAtom> atoms)
{
    std::vector<u32> out;
    out.reserve(atoms.size());
    const AtomIndex& ix = task.atoms();
    for (CanonicalAtom c : atoms)
        if (c < ix.layout().fluent_count)
            out.push_back(ix.intern(c));
    return out;
}

/// The landmark coordinates of LIW options (mimir's make_grouping + LandmarkCoordinates). Throws
/// std::invalid_argument for all_private with unshared atoms (when disjunctive is set).
inline novelty::LandmarkCoordinates make_coordinates(const Task& task, const LandmarkNovelty& ln)
{
    if (!ln.graph)
        return novelty::LandmarkCoordinates();
    std::vector<std::vector<u32>> disjunctive;
    for (const auto& set : ln.graph->disjunctive())
        disjunctive.push_back(fluent_slots_of(task, set));
    return novelty::LandmarkCoordinates::make(fluent_slots_of(task, ln.graph->landmarks()), disjunctive, ln.disjunctive, ln.all_private,
                                              fluent_slots_of(task, ln.unshared_atoms));
}

/// Applies a layer ordering to an engine environment (`orderer` holds its state and must outlive the search);
/// returns an error message, or an empty string.
inline std::string apply_layers(Env& env, const LayerOrdering& lo, LayerOrderer& orderer)
{
    if (std::string e = check_layers(lo); !e.empty())
        return e;
    if (lo.kind == LayerOrdering::Kind::Queue)
        return {};
    if (env.coord)
        return "SearchControl::coordination is only supported with LayerOrdering::Kind::Queue: an ordered or truncated layer order cannot "
               "publish sound completed depths";
    orderer = LayerOrderer(env.task, lo);
    env.layers = &orderer;
    return {};
}

/// A result that could not run.
inline IwResult failed(std::string message)
{
    IwResult r;
    r.status = SearchStatus::Failed;
    r.message = std::move(message);
    return r;
}
}  // namespace mymyr::search::detail
