#pragma once
// Graph certificates for symmetry pruning, matching mimir's color-refinement and k-FWL algorithms (source: Grohe et
// al., https://arxiv.org/pdf/1907.09582):
//   - color refinement (1-WL): the stable colouring of the vertices, refined by the multiset of neighbour colours,
//     with mimir's work-list algorithm (signatures per vertex, classes split in the order of their signatures);
//   - k-FWL (k = 2, 3 or 4): the stable colouring of the vertex k-tuples, refined by the multisets of the k-tuples of
//     colours of the tuples that replace one position.
//
// A certificate is a 128-bit digest of canonical content. For colour refinement: the initial colours (palette
// sequences, in order), the decoding table of every refinement (old colour, sorted signature) -> new colour, and the
// histogram of the stable colours. For k-FWL: the palette, then per round (the initial colouring first) the classes
// in their canonical order, each as (old colour, signature, size). A k-FWL signature is a 128-bit hash of the
// multiset of the n colour k-tuples (two independent 64-bit sums of a 64-bit hash per k-tuple, so it needs no sort);
// a round renumbers the classes densely by (old colour, signature) and the refinement stops at the first round that
// splits no class. Isomorphic graphs get equal certificates. Weisfeiler-Leman is not complete, so non-isomorphic
// graphs may share one too; mimir's symmetry pruning uses nauty's canonical forms instead. mymyr does not link nauty,
// to avoid a new build dependency.
//
// The initial colour of a k-tuple (v_1, ..., v_k) is its ordered isomorphism type: the colours of v_1, ..., v_k and,
// for every pair of positions i < j, whether v_i = v_j and whether v_i and v_j are adjacent. It is invariant under
// isomorphisms of the ordered k-vertex subgraph and needs no canonical labelling, for every k.
//
// Deviations from mimir, which are bugs there:
//   - mimir's colour-refinement certificate compares the decoding table only (its identifying members list it
//     twice), neither the initial colours nor the class sizes: two single vertices of different colours share one;
//   - mimir's colour refinement only refines vertices with neighbours (its work list is built from the edges), so an
//     isolated vertex never leaves its initial class; here every vertex has a (possibly empty) signature;
//   - mimir's k-FWL gives a tuple the nauty canonical form of the subgraph induced by its distinct vertices,
//     built from an unordered map whose iteration order (which depends on the vertex ids) assigns the colours while the
//     edges use the insertion order. For k >= 3 colours and edges can be mismatched, so its certificate is not
//     invariant under relabelling and separates some isomorphic graphs; its certificate also omits the class sizes.
//     mymyr uses the ordered isomorphism type above, the standard initial colouring of k-FWL. Where mimir's
//     classes respect isomorphism, the two partitions of graphs into certificate classes agree.
//
// Cost of k-FWL on a graph with n vertices: n^k tuples of kfwl_bytes_per_tuple bytes each, and n^(k+1) colour
// k-tuples hashed per round (a handful of rounds on object graphs; a round of n^(k+1) = 2^30 takes several seconds on one
// core). KfwlLimits bounds both.

#include "mymyr/core/types.hpp"
#include "mymyr/datasets/object_graph.hpp"

#include <compare>
#include <vector>

namespace mymyr::datasets
{
struct Certificate
{
    u64 lo = 0, hi = 0;
    friend bool operator==(const Certificate&, const Certificate&) = default;
    friend auto operator<=>(const Certificate&, const Certificate&) = default;
};

struct CertificateHash
{
    [[nodiscard]] usize operator()(const Certificate& c) const noexcept { return static_cast<usize>(c.lo ^ (c.hi * 0x9E3779B97F4A7C15ULL)); }
};

/// Colour refinement: the certificate, and optionally the stable colour of every vertex (canonical: equal for
/// corresponding vertices of isomorphic graphs).
[[nodiscard]] Certificate color_refinement_certificate(const ObjectGraph& g, std::vector<u32>* stable_colors = nullptr);

/// Bytes kfwl_certificate allocates per vertex k-tuple: its colour (4) and its sort entry (two 64-bit signature
/// words, the old colour and the tuple index: 24).
inline constexpr u64 kfwl_bytes_per_tuple = 28;

/// Resource limits of kfwl_certificate: a graph above either limit is rejected with std::length_error before any
/// work.
struct KfwlLimits
{
    /// Most vertex k-tuples (n^k): the memory bound. The default, 2^26 tuples (1.75 GiB), allows n <= 8192 for k = 2,
    /// n <= 406 for k = 3 and n <= 90 for k = 4.
    u64 max_tuples = u64{1} << 26;
    /// Most colour k-tuples hashed per refinement round (n^(k+1)): the time bound. The default, 2^30, allows
    /// n <= 1024 for k = 2, n <= 181 for k = 3 and n <= 64 for k = 4, and keeps a round within seconds.
    u64 max_round_work = u64{1} << 30;
    friend bool operator==(const KfwlLimits&, const KfwlLimits&) = default;
};

/// k-FWL for k in {2, 3, 4}. Cost O(n^(k+1)) per round and n^k * kfwl_bytes_per_tuple bytes: meant for small graphs.
/// Throws std::invalid_argument for other k and std::length_error (naming n, k and the limit) for a graph beyond
/// `limits`.
[[nodiscard]] Certificate kfwl_certificate(const ObjectGraph& g, u32 k, const KfwlLimits& limits = {});
}  // namespace mymyr::datasets
