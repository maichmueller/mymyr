#pragma once
// The order in which a breadth-first pass expands a layer: mimir's brfs::Options layer_ordering_strategy and
// max_next_layer_states (search/algorithms/brfs/ordered_layer.cpp, strategies/layer_ordering_strategy.cpp). Used by
// search::iw and search::brfs (single-threaded) and by the IW family variants (search/iw_family.hpp).
//
// With an ordered kind the search runs layer by layer: the next layer is collected in generation order and, before
// it is expanded, reordered by the kind (the start state's layer is never reordered). max_next_layer_states
// truncates: once the next layer holds that many entries, the current expansion and the rest of the current layer
// are dropped (neither expanded nor goal-tested).

#include "mymyr/core/types.hpp"

namespace mymyr::search
{
struct LayerOrdering
{
    enum class Kind : u8
    {
        Queue,       // the plain queued BrFS; the only kind that supports SearchControl::coordination
        InOrder,     // layer by layer in generation order (mimir's InOrderLayerOrderingStrategy)
        Reverse,     // every next layer reversed (mimir's ReverseOrderLayerOrderingStrategy)
        Randomized,  // every next layer shuffled (mimir's RandomizedLayerOrderingStrategy)
        GoalCount,   // every next layer stably sorted by satisfied goal literals (mimir's GoalCountLayerOrderingStrategy)
    };
    Kind kind = Kind::Queue;
    /// Randomized: the seed of a SplitMix64 (core/random.hpp; portable, unlike mimir's std::mt19937_64 with
    /// std::shuffle). The generator persists across the layers and the passes of one search.
    u64 seed = 0;
    /// Every ordered kind: once the next layer holds this many entries, the current expansion and the rest of the
    /// current layer are dropped, as in mimir.
    u32 max_next_layer_states = ~u32{0};
    /// GoalCount: states that satisfy more of the task's goal literals first (false: fewer first). The score counts
    /// the distinct fluent and derived goal literals that hold (positive ones true, negative ones false); numeric
    /// goal constraints are not counted. It is the task's goal even when the search has another goal (mimir's).
    bool prefer_more_satisfied_goals = true;
};
}  // namespace mymyr::search
