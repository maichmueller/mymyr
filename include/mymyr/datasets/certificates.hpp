#pragma once
// Graph certificates for symmetry pruning, matching mimir's color-refinement and k-FWL algorithms (source: Grohe et
// al., https://arxiv.org/pdf/1907.09582):
//   - color refinement (1-WL): the stable colouring of the vertices, refined by the multiset of neighbour colours,
//     with mimir's work-list algorithm (signatures per vertex, classes split in the order of their signatures);
//   - k-FWL (k = 2 or 3): the stable colouring of the vertex k-tuples, refined by the multisets of the k-tuples of
//     colours of the tuples that replace one position.
//
// A certificate is a 128-bit digest of the canonical content: the initial colours (palette sequences, in order), the
// decoding table of every refinement (old colour, sorted signature) -> new colour, and the histogram of the stable
// colours. Isomorphic graphs get equal certificates. Weisfeiler-Leman is not complete, so non-isomorphic graphs may
// share one too; mimir's symmetry pruning uses nauty's canonical forms instead. mymyr does not link nauty, to avoid
// a new build dependency.
//
// Deviations from mimir, which are bugs there:
//   - mimir's colour-refinement certificate compares the decoding table only (its identifying members list it
//     twice), neither the initial colours nor the class sizes: two single vertices of different colours share one;
//   - mimir's colour refinement only refines vertices with neighbours (its work list is built from the edges), so an
//     isolated vertex never leaves its initial class; here every vertex has a (possibly empty) signature;
//   - mimir's k-FWL gives a tuple the nauty canonical form of the subgraph induced by its distinct vertices,
//     built from an unordered map whose iteration order need not match the edges it adds (so colours and edges can be
//     mismatched); mymyr uses the ordered isomorphism type of the tuple (colours, equalities and adjacencies between
//     its positions), the standard initial colouring of k-FWL.

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

/// k-FWL for k in {2, 3}. Cost O(n^(k+1)) per round: meant for small graphs. Throws std::invalid_argument for other k.
[[nodiscard]] Certificate kfwl_certificate(const ObjectGraph& g, u32 k);
}  // namespace mymyr::datasets
