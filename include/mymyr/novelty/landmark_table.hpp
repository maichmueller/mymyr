#pragma once
// Landmark-restricted novelty for LIW(k), matching mimir 0.16.3's LandmarkCoordinates and LandmarkNoveltyTable.
//
// A feature is a pair (l, t): l a landmark coordinate of the state, t a free atom tuple of size <= k.
//   - Coordinates (LandmarkCoordinates): landmark atoms are grouped into ranks; a state's coordinates are the ranks of
//     the groups with a true member, or the single sentinel BOT when none holds. With one atom per group (the default)
//     rank and atom are in bijection; disjunctive landmarks share one rank among their members (an atom may carry
//     several ranks). For a transition s -> t, the ranks of t split into flipped (not a rank of s) and kept (a rank of
//     s); BOT flips on when s held a landmark and t holds none;
//   - state tuples: every subset of the state's atoms of size 0..k (the empty tuple included), none at all when the
//     state has fewer than k - 1 atoms (mimir's StateTupleIndexGenerator needs k entries among atoms + placeholder);
//   - transition tuples: every subset of the successor's atoms of size 1..k that contains an atom the transition added,
//     none when the successor has fewer than k - 1 atoms;
//   - the start state marks coordinates x state tuples; a transition marks flipped x state tuples of t and
//     kept x transition tuples; it is novel iff one of those pairs was unmarked.
// Every state admitted to a tree has all of its pairs marked, so a novel successor is never a duplicate.
//
// Layout: tuples are numbered by the combinadic rank of the padded tuple (placeholders 0 .. k-j-1, atom a as a + k),
// which does not depend on the atom count, so growing the table only appends. Dense: tuple-major rows of
// words_for(num_ranks) words (mimir's TupleMajorBitTable), while rows x row bytes fit LandmarkTableOptions;
// sparse (one way, like mimir): an open-addressing set of (tuple, rank) keys. For k <= 2, while it fits the same
// budget, the dense table is rank-major instead, so the tests become word operations over the state: k = 1 keeps per
// rank the set of marked atoms next to the tuple-major rows (a flipped rank tests the successor's atoms against its
// set, a kept rank tests the added atoms' rows); k = 2 keeps per rank and atom a the set of atoms b with {a, b}
// marked ({a} on the diagonal). A transition is novel iff one of its pairs is unmarked, and then all of them are
// marked (marking pairs that are all marked changes nothing), so the tests stop at the first unmarked pair. Layout
// never changes results.
//
// Atoms are fluent slots. Not thread-safe: one table per search.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/novelty/novelty_table.hpp"

#include <span>
#include <vector>

namespace mymyr::novelty
{
struct LandmarkTableOptions
{
    /// Largest dense table in bytes (mimir's default 256 MiB); above it the table is sparse. 0 forces sparse.
    u64 max_dense_bytes = u64{256} << 20;
};

class LandmarkCoordinates
{
public:
    /// No landmarks: every state's coordinate is BOT.
    LandmarkCoordinates() : LandmarkCoordinates(std::vector<std::vector<u32>>{}) {}
    /// One rank per group (groups of slots; duplicates within a group are ignored, empty groups kept as ranks that
    /// never hold).
    explicit LandmarkCoordinates(std::vector<std::vector<u32>> groups);

    /// mimir's grouping (LandmarkNoveltyPruningStrategyImpl::make_grouping and the LandmarkCoordinates
    /// constructors), over slots:
    ///   - without `disjunctive` (or without disjunctive sets): one rank per fact landmark;
    ///   - `all_private`: one rank per atom of fact landmarks and disjunctive members (unshared must be empty);
    ///   - otherwise one shared rank per disjunctive set (members minus `unshared`, in the given order) and one rank
    ///     per fact landmark or unshared member.
    /// Throws std::invalid_argument for all_private with unshared atoms.
    static LandmarkCoordinates make(std::vector<u32> fact_landmarks, const std::vector<std::vector<u32>>& disjunctive, bool use_disjunctive,
                                    bool all_private, const std::vector<u32>& unshared);

    [[nodiscard]] u32 num_ranks() const noexcept { return m_groups + 1; }
    [[nodiscard]] u32 bot() const noexcept { return m_groups; }
    [[nodiscard]] u32 mask_words() const noexcept { return bits::words_for(num_ranks()); }
    /// Landmark atoms (atoms that carry a rank), ascending.
    [[nodiscard]] std::span<const u32> atoms() const noexcept { return m_atoms; }
    [[nodiscard]] bool bijective() const noexcept { return m_bijective; }
    /// The ranks of a landmark atom (by its position in atoms()).
    [[nodiscard]] std::span<const u32> ranks_at(u32 position) const noexcept
    {
        return {m_flat.data() + m_offsets[position], m_offsets[position + 1] - m_offsets[position]};
    }

    /// The rank mask of a state without BOT (mask_words() words); returns whether some rank holds.
    bool mask(const u64* w, u32 n, u64* out) const;
    /// The coordinates of a state from its mask: the ranks set, or BOT.
    void collect(const u64* mask, bool any, std::vector<u32>& ranks) const;
    /// The coordinates of t split relative to s, from their masks.
    void split(const u64* pmask, bool pany, const u64* smask, bool sany, std::vector<u32>& flipped, std::vector<u32>& kept) const;
    /// The mask of a successor w (what mask(w, n, out) computes) from its parent's mask and the transition: `add`
    /// holds the atoms it adds (true in w, false in the parent), `del` the delete list (entries true in w are ignored).
    /// Returns whether some rank holds.
    bool child_mask(const u64* pmask, const u64* w, u32 n, std::span<const u32> add, std::span<const SlotId> del, u64* out) const;
    /// The split masks of child_mask's result: flipped and kept ranks, BOT included (mask_words() words each).
    void split_masks(const u64* pmask, bool pany, const u64* smask, bool sany, u64* flipped, u64* kept) const;

private:
    [[nodiscard]] u32 position_of(u32 atom) const noexcept { return atom < m_lookup.size() ? m_lookup[atom] : 0; }

    u32 m_groups = 0;
    bool m_bijective = true;
    std::vector<u32> m_atoms;                   // ascending
    std::vector<u32> m_offsets, m_flat;         // CSR: ranks of m_atoms[i]
    std::vector<u32> m_lookup;                  // slot -> position in m_atoms + 1 (0: not a landmark)
    std::vector<u32> m_member_offsets, m_members;  // CSR: the atoms of each rank
};

class LandmarkNoveltyTable
{
public:
    /// arity k in 1..k_max_arity (free coordinates); num_ranks = LandmarkCoordinates::num_ranks(); max_atoms: an
    /// upper bound of the slot count. Throws std::invalid_argument for a bad k and std::length_error if the sparse
    /// keys of k >= 4 cannot be ranked in 64 bits.
    LandmarkNoveltyTable(u32 arity, u32 num_ranks, u32 max_atoms, const LandmarkTableOptions& options = {});

    [[nodiscard]] u32 arity() const noexcept { return m_k; }
    [[nodiscard]] u32 capacity() const noexcept { return m_cap; }
    [[nodiscard]] bool dense() const noexcept { return m_fast != 0 || m_dense; }
    /// Slots below n become addressable.
    void reserve(u32 n);

    /// Marks ranks x state tuples of w; true if a pair was unmarked.
    bool mark_state(const u64* w, u32 n, std::span<const u32> ranks);
    /// Marks ranks x the tuples of the successor w that contain an atom false in the parent p; true if a pair was
    /// unmarked.
    bool mark_transition(const u64* p, u32 np, const u64* w, u32 n, std::span<const u32> ranks);
    /// Both at once for a transition p -> w: flipped x state tuples of w and kept x transition tuples, the rank sets as
    /// masks of words_for(num_ranks) words; `add` holds the atoms of w false in p. True if a pair was unmarked.
    bool mark_successor(const u64* p, u32 np, const u64* w, u32 n, std::span<const u32> add, const u64* flipped, const u64* kept);

    [[nodiscard]] u64 bytes() const noexcept;

private:
    // rank-major layouts (k <= 2)
    [[nodiscard]] u64 fast_bytes(u32 cap) const;
    void grow_fast(u32 cap);
    void leave_fast(u32 cap);
    void generic_reserve(u32 cap);
    void insert_generic(const u32* sorted, u32 j, u32 rank);
    [[nodiscard]] bool fast_state_novel(const u64* w, u32 n, u32 rank) const;
    void fast_state_mark(const u64* w, u32 n, u32 rank);
    [[nodiscard]] bool fast_transition_novel(const u64* w, u32 n, std::span<const u32> add, const u64* kept) const;
    void fast_transition_mark(const u64* w, u32 n, std::span<const u32> add, const u64* kept);
    bool fast_successor(const u64* w, u32 n, std::span<const u32> add, const u64* flipped, const u64* kept);

    void load_atoms(const u64* w, u32 n);
    void set_ranks(std::span<const u32> ranks);
    template<class Visit>
    void state_tuples(Visit&& visit);
    template<class Visit>
    void transition_tuples(const u64* p, u32 np, Visit&& visit);
    bool visit(const u32* sorted, u32 j);
    [[nodiscard]] u64 code(const u32* sorted, u32 j) const noexcept;
    void unrank(u64 r, u32* sorted, u32& j) const;
    [[nodiscard]] u64 rows_for(u32 cap) const;
    void to_sparse();
    bool sparse_insert(const u32* sorted, u32 j, u64 code, u32 rank);

    u32 m_k;
    u32 m_ranks;
    u32 m_rw;  // words per dense row
    u32 m_max_atoms;
    u32 m_cap = 0;
    u32 m_fast = 0;  // 1 or 2: the rank-major layout of that arity; 0: tuple-major or sparse
    u32 m_aw = 0;    // words per atom set of the rank-major layout (m_cap / 64)
    bool m_dense = true;  // tuple-major rows (m_rows) hold the marks (also under m_fast == 1)
    bool m_packed = false;  // sparse keys pack the atoms (k <= 3) instead of ranking them
    LandmarkTableOptions m_options;
    std::vector<std::vector<u64>> m_binom;  // m_binom[i][x] = C(x, i), i = 1..k, x < cap + k
    std::vector<u64> m_rows;                // dense
    TupleSet m_sparse;
    std::vector<u8> m_empty;                // rank-major: (rank, {}) marked
    std::vector<u64> m_bits;                // rank-major: k = 1 [rank][atom words]; k = 2 [rank][atom][atom words]
    // scratch
    std::vector<u32> m_atoms;
    std::vector<u32> m_rank_list;
    std::vector<u32> m_low;
    std::vector<u64> m_mask;
    bool m_novel = false;
};
}  // namespace mymyr::novelty
