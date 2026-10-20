#pragma once

// What the two parallel passes of the encoder share, for flat batches (encoder.cpp) and horizon batches (horizon.cpp):
// the state of one pool member, the scratch of a call, and the loop of the fill pass over a member's graphs in either
// layout. Not part of the interface.

#include "mymyr/encoding/detail/horizon.hpp"
#include "mymyr/encoding/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <functional>
#include <limits>
#include <vector>

namespace mymyr::encoding
{
namespace detail
{
inline constexpr usize k_cache_line = 64;
// Graphs per tile of the relation-major scan: it runs over the tile's graphs one relation at a time, so that the
// strided writes (G entries apart) fill whole cache lines.
inline constexpr u64 k_scan_tile = 128;
inline constexpr u64 k_none = std::numeric_limits<u64>::max();

/// The kin scratch of the graphs of a member's slice (horizon relations _sibling_ and _cousin_): per graph, the words
/// of horizon_kin_words followed by its far lists, from the count to the fill. Made when the schema needs it, behind a
/// pointer so that the members of the other schemas stay small.
struct KinStore
{
    std::vector<u32> words;
    std::vector<u64> at;    // [slice graphs] where the words of each graph start
    u64 overflow = k_none;  // the first graph with more far list entries or tuples than 32 bits count
};

/// The state of one pool member: it counts a contiguous slice of the graphs and later fills the same slice, so that the
/// per-graph data of the slice stays with the member. Aligned so that members never share a cache line.
struct alignas(k_cache_line) Member
{
    // pass one
    std::vector<GraphTotals> totals;  // the slice's graphs
    std::vector<u32> cell_counts;     // the slice's cells, graph after graph, in cell order (relation-major or
                                      // with predicate virtual nodes or layers)
    std::vector<u32> action_starts;   // [slice graphs, starts_stride] the run starts of each graph's rows
    std::vector<u32> action_cursors;  // [num_schemas] the cursors of the merge of one action relation's runs
    std::vector<u64> tally;           // [R] tuples per relation in the slice (relation-major)
    u64 entities = 0, tuples = 0, args = 0, targets = 0, history = 0;  // the slice's sizes
    u64 numeric = 0;  // the slice's numeric tuples (the entries of its numeric values)
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none;  // the first row of each kind of invalid input
    u64 narrow_goal = k_none, bad_goal_bits = k_none;
    u64 narrow_subgoals = k_none, bad_subgoal_bits = k_none;
    u64 narrow_history = k_none, bad_history_bits = k_none, bad_history_count = k_none;
    u64 bad_slice = k_none, bad_action_order = k_none, bad_action_label = k_none;
    u64 bad_edge = k_none;  // horizon: the first graph with an invalid edge
    std::vector<u64> effective;        // the subgoal levels of the row being counted or filled (effective_layers)
    std::vector<u32> virtual_scratch;  // [predicates] the virtual ranks of one graph
    // between the passes
    u64 entity_start = 0, tuple_start = 0, arg_start = 0, target_start = 0, history_start = 0;  // where the slice
                                                                                                // starts in the batch
    u64 numeric_start = 0;
    std::vector<u64> start_tuples, start_args;  // [R] where the slice starts in each relation (relation-major)
    // pass two
    std::vector<u64> run_tuples, run_args;  // [R] the running positions
    std::vector<u32> tile_counts;           // [R, k_scan_tile] the counts of the tile being scanned
    std::unique_ptr<KinStore> kin;          // horizon: the kinship of the slice's graphs, made when the schema needs it
};
}  // namespace detail

struct EncodeScratch::Impl
{
    // The input count() saw, so that fill() can refuse another one.
    bool counted = false;
    EncodeInput counted_input;
    FlatSizes sizes;
    u64 numeric_total = 0;  // the numeric tuples of the counted batch

    std::vector<u32> arity;          // [R] the arguments per tuple of each relation
    std::unique_ptr<std::byte[]> relation_scratch;  // the relation arrays of an encoding that does not store them
    u64 relation_scratch_bytes = 0;
    std::unique_ptr<u64[]> derived;  // [rows, derived_stride] when the rows' derived words are evaluated here
    u64 derived_capacity = 0;
    u64 derived_stride = 0;
    std::vector<u64> bounds;  // horizon: the first graph of each member, and the graphs (members + 1 entries)
    std::vector<u32> edge_order;  // horizon: the canonical edge lists of every graph (horizon_order_words per graph)
    std::vector<detail::Member> members;
};

namespace detail
{
/// The members that encode a batch of `graphs` graphs (a horizon batch: of nodes): all of the pool for a large batch,
/// fewer for a small one, one (the calling thread) for a batch under the grain.
[[nodiscard]] inline u32 members_for(u64 graphs, const ThreadPool& pool) noexcept
{
    const u64 wanted = std::max<u64>(1, (graphs + k_graphs_per_member - 1) / k_graphs_per_member);
    return static_cast<u32>(std::min<u64>(pool.size(), wanted));
}

/// Runs `f(t)` for the members [0, members) without allocating a closure: inline for one member, on the pool otherwise
/// (a pool member beyond `members` has nothing to do).
template<class F>
void run(ThreadPool& pool, u32 members, const F& f)
{
    if (members == 1)
    {
        f(0);
        return;
    }
    const std::function<void(u32)> job = [&f, members](u32 t)
    {
        if (t < members)
            f(t);
    };
    pool.run(job);
}

/// What the relation-major scan and the fills of a member's graphs need to know about the batch.
struct FillShape
{
    u64 graphs;
    u32 relations;
    bool relation_major;
    const u32* arity;  // [R] the arguments per tuple of each relation (relation-major)
};

/// The fill pass of the graphs [lo, hi) of member `m`. `cells_of(g)` is the CellList of graph g and `fill_one(g,
/// graph_counts)` fills it, where `graph_counts` are the graph's cell counts from pass one (null unless the pass kept
/// them). Graph-major, the graphs are filled in order. Relation-major, the offsets of every segment are scanned first,
/// a tile of graphs at a time, and the graphs are filled after the scan of their tile.
template<class I, class FillOne, class CellsOf>
void fill_member(Member& m, const FillShape& shape, u64 lo, u64 hi, I* relation_offsets, I* relation_arg_offsets,
                 const FillOne& fill_one, const CellsOf& cells_of)
{
    if (!shape.relation_major)
    {
        const u32* graph_counts = m.cell_counts.empty() ? nullptr : m.cell_counts.data();
        for (u64 g = lo; g < hi; ++g)
        {
            fill_one(g, graph_counts);
            if (graph_counts)
                graph_counts += cells_of(g).count;
        }
        return;
    }
    const u32 R = shape.relations;
    m.run_tuples = m.start_tuples;
    m.run_args = m.start_args;
    m.tile_counts.resize(R * k_scan_tile);
    const u32* cell_counts = m.cell_counts.data();
    for (u64 first = lo; first < hi; first += k_scan_tile)
    {
        const u64 width = std::min(k_scan_tile, hi - first);
        // the tile's counts, relation by relation (a relation without a cell in a graph has count 0)
        std::fill(m.tile_counts.begin(), m.tile_counts.end(), 0u);
        const u32* next = cell_counts;
        for (u64 j = 0; j < width; ++j)
        {
            const CellList cells = cells_of(first + j);
            for (u32 i = 0; i < cells.count; ++i)
                m.tile_counts[cells.cells[i].relation * k_scan_tile + j] = next[i];
            next += cells.count;
        }
        // the exclusive scan in segment order r * G + g, written in runs of `width` per relation
        for (u32 r = 0; r < R; ++r)
        {
            I* offsets = relation_offsets + r * shape.graphs + first;
            I* arg_offsets = relation_arg_offsets + r * shape.graphs + first;
            const u32* counts = m.tile_counts.data() + r * k_scan_tile;
            const u64 arity = shape.arity[r];
            u64 tuples = m.run_tuples[r];
            u64 args = m.run_args[r];
            if (m.tally[r] == 0)  // no graph of the slice has a tuple of the relation: the offsets do not advance
            {
                std::fill_n(offsets, width, static_cast<I>(tuples));
                std::fill_n(arg_offsets, width, static_cast<I>(args));
                continue;
            }
            for (u64 j = 0; j < width; ++j)
            {
                offsets[j] = static_cast<I>(tuples);
                arg_offsets[j] = static_cast<I>(args);
                tuples += counts[j];
                args += counts[j] * arity;
            }
            m.run_tuples[r] = tuples;
            m.run_args[r] = args;
        }
        const u32* graph_counts = cell_counts;
        for (u64 j = 0; j < width; ++j)
        {
            fill_one(first + j, graph_counts);
            graph_counts += cells_of(first + j).count;
        }
        cell_counts = next;
    }
}

/// Step three of count(): the members' sizes scanned in u64 (where each starts in the batch and, relation-major, in
/// every relation) and the batch's sizes, with k_overflow_index set when one does not fit the index type.
inline FlatSizes scan_members(EncodeScratch::Impl& s, const EncodingSchema& schema, u64 graphs)
{
    const u32 R = static_cast<u32>(schema.num_relations());
    u64 entities = 0, tuples = 0, args = 0, targets = 0, history = 0, numeric = 0;
    for (Member& m : s.members)
    {
        m.numeric_start = numeric;
        numeric += m.numeric;
        m.entity_start = entities;
        m.tuple_start = tuples;
        m.arg_start = args;
        m.target_start = targets;
        m.history_start = history;
        entities += m.entities;
        tuples += m.tuples;
        args += m.args;
        targets += m.targets;
        history += m.history;
    }
    if (schema.options().layout == Layout::Relation)
    {
        for (Member& m : s.members)
        {
            m.start_tuples.resize(R);
            m.start_args.resize(R);
        }
        s.arity.resize(R);
        u64 tuple_position = 0, arg_position = 0;
        for (u32 r = 0; r < R; ++r)
        {
            s.arity[r] = schema.relation(r).arity;
            for (Member& m : s.members)
            {
                m.start_tuples[r] = tuple_position;
                m.start_args[r] = arg_position;
                tuple_position += m.tally[r];
                arg_position += m.tally[r] * s.arity[r];
            }
        }
    }
    s.numeric_total = numeric;
    FlatSizes sizes;
    sizes.entities = entities;
    sizes.tuples = tuples;
    sizes.args = args;
    sizes.targets = targets;
    sizes.history = history;
    const u64 limit = static_cast<u64>(schema.options().index == IndexType::I64 ? std::numeric_limits<i64>::max()
                                                                                 : std::numeric_limits<i32>::max());
    if (entities > limit || tuples > limit || args > limit || targets > limit || history > limit ||
        graphs * R + 1 > limit || graphs + 1 > limit)
        sizes.overflow |= k_overflow_index;
    if (schema.emits_simple() && (entities + tuples > limit || 2 * args > limit))
        sizes.overflow |= k_overflow_index;
    return sizes;
}
}  // namespace detail
}  // namespace mymyr::encoding
