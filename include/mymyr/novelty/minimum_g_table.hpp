#pragma once

#include "mymyr/core/hash.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/novelty/novelty_table.hpp"
#include "mymyr/state/state.hpp"

#include <array>
#include <span>
#include <unordered_map>
#include <vector>

namespace mymyr::novelty
{
/// Minimum path-cost labels for fluent tuples, for best-first width pruning. Not thread-safe.
/// State queries enumerate sizes 0..arity; transitions enumerate sizes 1..arity containing an added atom.
/// As in mimir, neither query has tuples when the state has fewer than arity - 1 atoms.
/// Combinadic ranks are independent of atom capacity; tuples whose ranks overflow use their full atom identities.
/// Labels are exact doubles, so there is no limit on the number of distinct costs. Storage is sparse.
class MinimumGNoveltyTable
{
public:
    explicit MinimumGNoveltyTable(u32 arity);
    [[nodiscard]] u32 arity() const noexcept { return m_k; }
    bool test_and_update(StateView state, f64 g, u32 coordinate = 0);
    bool test_and_update(StateView parent, StateView state, f64 g, u32 coordinate = 0);
    [[nodiscard]] bool would_improve(StateView state, f64 g, u32 coordinate = 0) const;
    [[nodiscard]] bool would_improve(StateView parent, StateView state, f64 g, u32 coordinate = 0) const;
    [[nodiscard]] bool test_at_g(StateView state, f64 g, u32 coordinate = 0) const;
    [[nodiscard]] bool has_lowered_existing_label() const noexcept { return m_lowered; }
    [[nodiscard]] u64 bytes() const noexcept;

private:
    struct Key
    {
        u64 rank = 0;
        u32 coordinate = 0;
        bool packed = false;
        std::array<u32, k_max_arity> atoms{};
        friend bool operator==(const Key&, const Key&) = default;
    };
    struct KeyHash
    {
        usize operator()(const Key& key) const noexcept;
    };
    using Labels = std::unordered_map<Key, f64, KeyHash>;
    [[nodiscard]] static Key key(std::span<const u32> tuple, u32 coordinate);
    template<class Visit>
    bool tuples(StateView state, const StateView* parent, Visit&& visit) const;
    bool update(StateView state, const StateView* parent, f64 g, u32 coordinate);

    u32 m_k;
    std::array<Labels, k_max_arity + 1> m_labels;
    bool m_lowered = false;
};
}  // namespace mymyr::novelty
