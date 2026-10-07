#pragma once
// The order in which a breadth-first pass expands a layer: mimir's brfs::Options layer_ordering_strategy,
// max_next_layer_states and its beam (beam_width, beam_novelty_mode, randomize_equal_score_ties;
// search/algorithms/brfs/ordered_layer.cpp and beam.cpp, strategies/layer_ordering_strategy.cpp). Used by search::iw,
// search::iw_pass, search::siw and search::brfs (single-threaded) and by the IW family variants (search/liw.hpp,
// search/aiw.hpp).
//
// With an ordered kind the search runs layer by layer: the next layer is collected in generation order and, before
// it is expanded, reordered by the kind (the start state's layer is never reordered). max_next_layer_states
// truncates: once the next layer holds that many entries, the current expansion and the rest of the current layer
// are dropped (neither expanded nor goal-tested).
//
// Beam (beam_width): the whole next layer is generated first; then it is ordered by the kind and only its first
// beam_width entries are kept (GoalCount: the beam_width best scores, equal scores in generation order or, with
// randomize_ties, in the order of random tie tokens; InOrder / Reverse / Randomized: the first beam_width entries of
// that order). The others are never expanded and never enter again: their states count as generated (mimir's
// generated_state_indices), not as part of the tree. The goal is tested when a kept state is popped. For a novelty
// pruned search the novelty mode decides which successors mark the novelty table:
//   - AllTested (mimir's ALL_TESTED): the ordinary novelty test, which marks the tuples of every successor it admits,
//     runs during generation, so successors the beam then drops still mark the table;
//   - SurvivorsOnly (mimir's SURVIVORS_ONLY): during generation the novelty test only reads the table; at the layer
//     boundary the kept entries are replayed in their order, each must add a tuple not already added by a better
//     ranked entry of the same layer, and only those mark the table. The replay can leave fewer than beam_width
//     entries. search::liw refuses it (as mimir: its landmark novelty table has no read-only test).
// search::brfs has no novelty table: there the two modes are the same (mimir's duplicate pruning).
// The beam selects after the whole next layer was generated, unlike max_next_layer_states, which cuts the
// generation; the two are mutually exclusive.

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
    /// Which successors of a beam search mark the novelty table (see above; mimir's BeamNoveltyMode).
    enum class BeamNovelty : u8
    {
        AllTested,      // every successor the novelty test admits, kept or not
        SurvivorsOnly,  // only the kept ones, replayed in rank order at the layer boundary
    };
    Kind kind = Kind::Queue;
    /// Randomized and randomize_ties: the seed of a SplitMix64 (core/random.hpp; portable, unlike mimir's
    /// std::mt19937_64 with std::shuffle). The generator persists across the layers and the passes of one search.
    u64 seed = 0;
    /// Every ordered kind: once the next layer holds this many entries, the current expansion and the rest of the
    /// current layer are dropped, as in mimir. Exclusive with beam_width.
    u32 max_next_layer_states = ~u32{0};
    /// GoalCount: states that satisfy more of the task's goal literals first (false: fewer first). The score counts
    /// the distinct fluent and derived goal literals that hold (positive ones true, negative ones false); numeric
    /// goal constraints are not counted. It is the task's goal even when the search has another goal (mimir's).
    bool prefer_more_satisfied_goals = true;
    /// Every ordered kind: keep only the first beam_width entries of each ordered next layer (~0: no beam).
    u32 beam_width = ~u32{0};
    /// With beam_width: which successors mark the novelty table.
    BeamNovelty beam_novelty = BeamNovelty::AllTested;
    /// GoalCount: equal scores are ordered by a random tie token per entry (drawn in generation order from the
    /// SplitMix64 of `seed`) instead of by generation order (mimir's randomize_equal_score_ties).
    bool randomize_ties = false;

    [[nodiscard]] bool beam() const noexcept { return beam_width != ~u32{0}; }
};
}  // namespace mymyr::search
