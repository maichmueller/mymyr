#pragma once
// The order in which a breadth-first pass expands a layer: the layer ordering strategy,
// max_next_layer_states and the beam (beam_width, the novelty mode, randomized equal-score ties, the parallel beam and
// the relaxed survivors-only beam,
// strategies/layer_ordering_strategy.cpp). Used by search::iw, search::iw_pass, search::siw and search::brfs (flat and
// chunked stores) and by the IW family variants (search/liw.hpp, search/aiw.hpp).
//
// With an ordered kind the search runs layer by layer: the next layer is collected in generation order and, before
// it is expanded, reordered by the kind (the start state's layer is never reordered). max_next_layer_states
// truncates: once the next layer holds that many entries, the current expansion and the rest of the current layer
// are dropped (neither expanded nor goal-tested).
//
// Beam (beam_width): the whole next layer is generated first; then it is ordered by the kind and only its first
// beam_width entries are kept (GoalCount: the beam_width best scores, equal scores in generation order or, with
// randomize_ties, in the order of random tie tokens; InOrder / Reverse / Randomized: the first beam_width entries of
// that order). The others are never expanded and never enter again: their states count as generated, not as part of the tree. The goal is tested when a kept state is popped. For a novelty
// pruned search the novelty mode decides which successors mark the novelty table:
//   - AllTested: the ordinary novelty test, which marks the tuples of every successor it admits,
//     runs during generation, so successors the beam then drops still mark the table;
//   - SurvivorsOnly: during generation the novelty test only reads the table; at the layer
//     boundary the kept entries are replayed in their order, each must add a tuple not already added by a better
//     ranked entry of the same layer, and only those mark the table. The replay can leave fewer than beam_width
//     entries. search::liw refuses it.
//   - RelaxedSurvivorsOnly: SurvivorsOnly with a cheaper selection. The
//     transitions of the layer, in generation order, are split into min(threads, ceil(n / beam_chunk)) contiguous
//     parts whose sizes differ by at most one (the longer ones first). Within each part the successors that pass the
//     read-only novelty test are ranked and the best beam_width kept, without the test for states seen before; the
//     parts' survivors are merged in rank order and taken until beam_width distinct states not seen before are
//     found; these are replayed as with SurvivorsOnly. Each part's top entries may be duplicates of each other or of
//     earlier states, so the layer can keep fewer than beam_width states even though more were available, and the
//     result depends on the number of parts, hence on the thread count of the search (for a fixed thread count it
//     is deterministic, also with randomize_ties). With one part (one thread, or at most beam_chunk transitions in
//     the layer) it is the same for every thread count. Needs Kind::GoalCount. With randomize_ties every transition
//     of the layer draws a tie token, in generation order.
// search::brfs has no novelty table: there AllTested and SurvivorsOnly are the same;
// RelaxedSurvivorsOnly ranks before it removes the states stored before, and stores only the states it looks at.
// The beam selects after the whole next layer was generated, unlike max_next_layer_states, which cuts the
// generation; the two are mutually exclusive.
//
// Threads (IwOptions::threads, AbstractedIwOptions::threads, BrfsOptions::threads): with threads > 1 a beam search
// generates the successors of a layer's states on several threads, together with what can be computed from them
// alone (the goal test of the states, the read-only part of the novelty test, the goal-count scores); the results
// are merged on the calling thread in the order of the serial search, so plans, counts and statistics equal the
// single-threaded search for every mode but RelaxedSurvivorsOnly. An observer, blocked states or a successor order
// hook run the search on the calling thread alone.

#include "mymyr/core/types.hpp"

#include <string>

namespace mymyr::search
{
struct LayerOrdering
{
    enum class Kind : u8
    {
        Queue,       // the plain queued BrFS; the only kind that supports SearchControl::coordination
        InOrder,     // layer by layer in generation order
        Reverse,     // every next layer reversed
        Randomized,  // every next layer shuffled
        GoalCount,   // every next layer stably sorted by satisfied goal literals
    };
    /// Which successors of a beam search mark the novelty table (see above).
    enum class BeamNovelty : u8
    {
        AllTested,             // every successor the novelty test admits, kept or not
        SurvivorsOnly,         // only the kept ones, replayed in rank order at the layer boundary
        RelaxedSurvivorsOnly,  // SurvivorsOnly with a per-part selection (see above; depends on the thread count)
    };
    Kind kind = Kind::Queue;
    /// Randomized and randomize_ties: the seed of a SplitMix64 (core/random.hpp; portable). The generator persists across the layers and the passes of one search.
    u64 seed = 0;
    /// Every ordered kind: once the next layer holds this many entries, the current expansion and the rest of the
    /// current layer are dropped. Exclusive with beam_width.
    u32 max_next_layer_states = ~u32{0};
    /// GoalCount: states that satisfy more of the task's goal literals first (false: fewer first). The score counts
    /// the distinct fluent and derived goal literals that hold (positive ones true, negative ones false); numeric
    /// goal constraints are not counted. It is the task's goal even when the search has another goal.
    bool prefer_more_satisfied_goals = true;
    /// Every ordered kind: keep only the first beam_width entries of each ordered next layer (~0: no beam).
    u32 beam_width = ~u32{0};
    /// With beam_width: which successors mark the novelty table.
    BeamNovelty beam_novelty = BeamNovelty::AllTested;
    /// GoalCount: equal scores are ordered by a random tie token per entry (drawn in generation order from the
    /// SplitMix64 of `seed`) instead of by generation order.
    bool randomize_ties = false;
    /// RelaxedSurvivorsOnly: the transitions per part of a layer's split.
    u32 beam_chunk = 1024;

    [[nodiscard]] bool beam() const noexcept { return beam_width != ~u32{0}; }
};

/// Checks a LayerOrdering against the rules above: an error message, or an empty string. Every search that takes a
/// LayerOrdering applies this check.
[[nodiscard]] inline std::string check_layers(const LayerOrdering& lo)
{
    if (lo.kind == LayerOrdering::Kind::Queue && lo.max_next_layer_states != ~u32{0})
        return "LayerOrdering::max_next_layer_states requires an ordered layer kind (every kind but Queue)";
    if (lo.max_next_layer_states == 0)
        return "LayerOrdering::max_next_layer_states must be positive";
    if (lo.beam_width == 0)
        return "LayerOrdering::beam_width must be positive";
    if (lo.beam() && lo.kind == LayerOrdering::Kind::Queue)
        return "LayerOrdering::beam_width requires an ordered layer kind (every kind but Queue)";
    if (lo.beam() && lo.max_next_layer_states != ~u32{0})
        return "LayerOrdering::beam_width and LayerOrdering::max_next_layer_states are mutually exclusive";
    if (!lo.beam() && lo.beam_novelty != LayerOrdering::BeamNovelty::AllTested)
        return "LayerOrdering::beam_novelty other than AllTested requires a beam (beam_width)";
    if (lo.randomize_ties && lo.kind != LayerOrdering::Kind::GoalCount)
        return "LayerOrdering::randomize_ties requires Kind::GoalCount (the only kind with scores)";
    if (lo.beam_novelty == LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly && lo.kind != LayerOrdering::Kind::GoalCount)
        return "LayerOrdering::beam_novelty RelaxedSurvivorsOnly requires Kind::GoalCount (it ranks by score)";
    if (lo.beam_chunk == 0)
        return "LayerOrdering::beam_chunk must be positive";
    return {};
}
}  // namespace mymyr::search
