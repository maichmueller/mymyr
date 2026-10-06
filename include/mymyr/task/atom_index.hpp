#pragma once
// The two-level atom index.
//
// Level 1, the canonical id: typed-dense mixed radix over per-position ranks,
//     cid(p, a_0..a_{k-1}) = offset[p] + sum_i rank[p][i][a_i] * stride[p][i].
//   The domain of position i of predicate p is the set of objects that can ever occur there: the static domains of the
//   parameters used at that position in add effects and axiom heads, plus constants and initial atoms. The id is
//   computed in O(arity) with no table, is identical across parses, processes and the GPU, and is the portable atom
//   key. Fluent predicates come first, derived ones after: cids [0, F_fluent) are fluent, [F_fluent, F_total) derived.
//
// Level 2, the slot: the bit position of an atom in a state (fluent) or in the per-state derived bitset (derived).
//   Fluent and derived atoms have separate slot spaces, so stored states never carry derived bits.
//   - Lazy mode: slots are assigned on first touch through `std::atomic<u32> slot_of[cid]` (a read is one load, a
//     first touch one CAS plus a fetch_add). The slot's (pred, args) record and view op are written before the slot is
//     published with a release store, so any thread that obtains the slot can read its record.
//   - Frozen mode: slot = cid (minus F_fluent for derived atoms); every record exists up front, nothing is mutated.
//
// The record arrays are ChunkedArrays: readers never see a reallocation while other threads intern.

#include "mymyr/core/chunked_array.hpp"
#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"

#include <atomic>
#include <memory>
#include <span>
#include <vector>

namespace mymyr
{
enum class AtomMode : u8
{
    Lazy,
    Frozen,
};

enum class AtomKind : u8
{
    Fluent = 0,
    Derived = 1,
};

/// Typed-dense canonical layout of all fluent and derived atoms.
struct CanonicalLayout
{
    /// Contribution of an object outside a position domain: any sum containing it is >= total.
    static constexpr u64 k_outside = u64{1} << 40;
    static constexpr u64 k_none = ~u64{0};
    /// Canonical spaces beyond this size need hashed interning (not implemented).
    static constexpr u64 k_max_total = u64{1} << 28;

    u32 num_objects = 0;
    u32 max_arity = 0;
    u64 fluent_count = 0;  // F_fluent
    u64 total = 0;         // F_fluent + F_derived
    std::vector<u32> arity;         // per predicate
    std::vector<u64> offset;        // per predicate: first cid (k_none for static predicates)
    std::vector<u64> size;          // per predicate: number of cids (0 for static predicates)
    std::vector<u64> pos_begin;     // per predicate: index of its first position in the per-position arrays
    std::vector<u64> rs;            // per position: num_objects entries, rank * stride or k_outside
    std::vector<u64> stride;        // per position
    std::vector<u32> domain_size;   // per position
    std::vector<u64> domain_begin;  // per position: start of its objects (in rank order) in `domain`
    std::vector<u32> domain;
    std::vector<u32> by_offset;     // fluent/derived predicates with size > 0, in increasing offset

    [[nodiscard]] bool has_predicate(u32 p) const { return offset[p] != k_none; }
    [[nodiscard]] AtomKind kind_of(CanonicalAtom c) const { return c < fluent_count ? AtomKind::Fluent : AtomKind::Derived; }
    /// rank * stride table of position i of predicate p (num_objects entries).
    [[nodiscard]] const u64* position_table(u32 p, u32 i) const
    {
        return rs.data() + (pos_begin[p] + i) * static_cast<u64>(num_objects);
    }

    /// Canonical id of p(args), or a value >= total if some argument lies outside its position domain.
    [[nodiscard]] CanonicalAtom encode(u32 p, const u32* args) const
    {
        u64 c = offset[p];
        for (u32 i = 0; i < arity[p]; ++i)
            c += position_table(p, i)[args[i]];
        return c;
    }
    /// Inverse of encode for c < total: returns the predicate and writes its arguments.
    u32 decode(CanonicalAtom c, u32* args) const;
};

/// Rows of the per-state view tables. Every fluent or derived predicate
/// of arity 1 owns one unary row; one of arity 2 owns n forward rows (row a0 holds the a1 with p(a0, a1)) and n
/// backward rows. A row is OW = words_for(n) words.
struct ViewLayout
{
    static constexpr u32 k_none = ~u32{0};
    u32 num_objects = 0;
    u32 ow = 0;    // words per row
    u32 rows = 0;  // total rows
    std::vector<u32> unary_row;  // per predicate
    std::vector<u32> fwd_row;    // per predicate: first forward row
    std::vector<u32> bwd_row;    // per predicate: first backward row
};

/// View op of an atom: the (row, bit) pairs it sets in the view.
struct ViewOp
{
    static constexpr u32 k_none = ~u32{0};
    u32 row1 = k_none, bit1 = 0;  // unary row (arity 1) or forward row (arity 2)
    u32 row2 = k_none, bit2 = 0;  // backward row (arity 2)
};

[[nodiscard]] ViewOp view_op_of(const ViewLayout& v, u32 pred, u32 arity, const u32* args);

class AtomIndex
{
public:
    static constexpr u32 k_empty = ~u32{0};
    static constexpr u32 k_pending = ~u32{0} - 1;

    AtomIndex() = default;
    AtomIndex(const AtomIndex&) = delete;
    AtomIndex& operator=(const AtomIndex&) = delete;

    /// Sets up the index. `layout` and `view` must outlive it. Not thread-safe (construction only), as is reset().
    void init(const CanonicalLayout* layout, const ViewLayout* view, AtomMode mode);
    void reset(AtomMode mode);

    [[nodiscard]] AtomMode mode() const noexcept { return m_mode; }
    [[nodiscard]] const CanonicalLayout& layout() const noexcept { return *m_layout; }

    /// Assigned fluent slots: a state has at most words_for(fluent_slots()) words.
    [[nodiscard]] u32 fluent_slots() const noexcept { return m_next[0].load(std::memory_order_acquire); }
    [[nodiscard]] u32 derived_slots() const noexcept { return m_next[1].load(std::memory_order_acquire); }
    /// Upper bounds (the canonical space sizes).
    [[nodiscard]] u32 max_fluent_slots() const noexcept { return static_cast<u32>(m_layout->fluent_count); }
    [[nodiscard]] u32 max_derived_slots() const noexcept
    {
        return static_cast<u32>(m_layout->total - m_layout->fluent_count);
    }

    /// Slot of canonical atom c (any value; c >= total means "outside the domains"), or k_empty if it has none. A
    /// pending or absent atom is false in every published state, so this never waits.
    [[nodiscard]] u32 find(CanonicalAtom c) const noexcept
    {
        const u32 v = m_slot_of[c < m_layout->total ? c : m_layout->total].load(std::memory_order_relaxed);
        return v < k_pending ? v : k_empty;
    }
    /// Slot of canonical atom c < total, assigning one on first touch. Thread-safe.
    u32 intern(CanonicalAtom c) const
    {
        const u32 v = m_slot_of[c < m_layout->total ? c : m_layout->total].load(std::memory_order_acquire);
        return v < k_pending ? v : intern_slow(c);
    }
    /// Raw table for hot paths: entries are slots, k_pending or k_empty. Entry `total` is a permanent k_empty.
    [[nodiscard]] const std::atomic<u32>* slot_table() const noexcept { return m_slot_of.get(); }

    /// Record of a slot: [pred, a_0, ..., a_{max_arity - 1}] (unused positions are zero).
    [[nodiscard]] const u32* record(AtomKind k, u32 slot) const noexcept { return m_records[idx(k)].get(slot); }
    [[nodiscard]] const ViewOp& view_op(AtomKind k, u32 slot) const noexcept { return *m_ops[idx(k)].get(slot); }
    [[nodiscard]] CanonicalAtom canonical(AtomKind k, u32 slot) const noexcept { return *m_cid[idx(k)].get(slot); }

    [[nodiscard]] PredicateId predicate(SlotId s) const noexcept { return PredicateId{record(AtomKind::Fluent, s.v)[0]}; }
    [[nodiscard]] std::span<const u32> arguments(SlotId s) const noexcept
    {
        const u32* r = record(AtomKind::Fluent, s.v);
        return {r + 1, m_layout->arity[r[0]]};
    }

    [[nodiscard]] u64 bytes() const noexcept;

private:
    static constexpr usize idx(AtomKind k) { return static_cast<usize>(k); }
    u32 intern_slow(CanonicalAtom c) const;
    void write_slot(AtomKind k, u32 slot, CanonicalAtom c) const;

    const CanonicalLayout* m_layout = nullptr;
    const ViewLayout* m_view = nullptr;
    AtomMode m_mode = AtomMode::Lazy;
    std::unique_ptr<std::atomic<u32>[]> m_slot_of;  // total + 1 entries
    // The only mutation of an otherwise immutable task: append-only slot assignment and records.
    mutable std::atomic<u32> m_next[2] = {0, 0};
    mutable ChunkedArray<u32> m_records[2];
    mutable ChunkedArray<ViewOp> m_ops[2];
    mutable ChunkedArray<CanonicalAtom> m_cid[2];
};
}  // namespace mymyr
