#pragma once
// Layer transition orderings for the width-1 pass of IW (IwOptions::transition_ordering): mimir 0.16.3's
// TransitionOrderingStrategy with requires_deferred_novelty (search/algorithms/strategies/transition_ordering_strategy.hpp,
// src/search/algorithms/brfs/transition_ordered_layer_impl.hpp). landmarks::LandmarkTransitionOrdering is one.
//
// With an ordering set, the width-1 pass runs layer by layer:
//   1. every state of the current layer (in admission order) is popped: goal test, expansion count, the optimized-IW(1)
//      skip mark and max_depth as in the queued pass; the transitions of the expanded ones are generated (the child is
//      materialized, nothing is admitted yet);
//   2. the ordering permutes the layer's transitions;
//   3. they are admitted in that order: the self-loop and duplicate rules of the optimized-IW(1) root, blocked states,
//      then the novelty test, which marks what it finds new. Novelty is decided at admission, so a preferred transition
//      wins a contested novel atom (and a contested successor state) over a later one.
// Arity-0 and arity > 1 passes stay queued (mimir's iw::find_solution with an ordering). Observer events: on_expand
// at the pop, on_generate / on_prune in admission order.

#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <span>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::search
{
/// One transition of a layer: the views are valid during TransitionOrdering::order only.
struct LayerTransition
{
    StateView parent;
    ActionLabel action;
    StateView child;
};

class TransitionOrdering
{
public:
    virtual ~TransitionOrdering() = default;
    /// Writes into `order` the indices of `layer` in admission order. `layer` is in generation order (parents in layer
    /// order, each parent's transitions in the successor generator's order); invalid and repeated indices are dropped,
    /// missing ones follow in generation order. Called once per layer, from the searching thread.
    virtual void order(const Task& task, std::span<const LayerTransition> layer, std::vector<u32>& order) const = 0;
};
}  // namespace mymyr::search
