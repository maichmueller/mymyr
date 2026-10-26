#pragma once

#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"

#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace mymyr
{
class Task;

/// Source fluent slot -> destination fluent slot; ~u32{0} denotes an atom absent from the destination.
struct SlotMapView
{
    const u32* data = nullptr;
    u32 size = 0;
};

/// A reusable name-based mapping between parses. Both tasks must outlive it.
/// Predicate/function names, arities and ordered object names identify atoms and numeric values.
/// Snapshots are immutable, remain valid as the source grows, and cover all source slots reserved before the call.
class SlotMap
{
public:
    SlotMap(const Task& source, const Task& target);
    static constexpr u32 missing = ~u32{0};
    static constexpr u32 checked_static = ~u32{0} - 1;
    [[nodiscard]] static std::shared_ptr<SlotMap> between(const Task& source, const Task& target);
    [[nodiscard]] std::shared_ptr<const std::vector<u32>> snapshot() const;
    [[nodiscard]] const Task& source() const noexcept { return m_source; }
    [[nodiscard]] const Task& target() const noexcept { return m_target; }
    [[nodiscard]] const std::vector<u32>& numeric_slots() const noexcept { return m_numeric; }
    [[nodiscard]] const std::vector<u64>& constant_words() const noexcept { return m_constant; }
    /// Unmatched source predicate signatures and object names, available without translating a row.
    [[nodiscard]] const std::vector<std::string>& unmatched() const noexcept { return m_unmatched; }
    /// Maps one row in O(source words + true atoms + numeric slots + target words).
    /// Missing true atoms and numeric values raise with their name and row unless drop is true.
    /// Unmatched target numeric slots retain their initial values. Returns whether every source value was preserved.
    bool translate_row(std::span<const u32> slots, StateView source, std::span<u64> atoms,
                       std::span<u64> numeric, bool drop = false, u64 row = 0) const;
    [[nodiscard]] State translate(StateView source, bool drop = false, bool* lossless = nullptr, u64 row = 0) const;

private:
    const Task& m_source;
    const Task& m_target;
    [[nodiscard]] u32 match_atom(u32 predicate, std::span<const u32> arguments) const;
    std::vector<u32> m_predicates, m_objects, m_numeric;
    std::vector<u64> m_constant;
    std::vector<std::string> m_unmatched;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<const std::vector<u32>> m_slots;
};

/// Clears destination, then maps each set source bit. Throws for a bit outside the map or destination slot space.
/// Numeric words are separate from these spans and retain their slot order within one normalized problem.
void map_fluent_words(SlotMapView map, std::span<const u64> source, std::span<u64> destination, u32 target_slots);
}  // namespace mymyr
