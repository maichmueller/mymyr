#pragma once

#include "mymyr/core/types.hpp"

#include <memory>
#include <mutex>
#include <span>
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

/// An incrementally extended mapping between slot spaces over one normalized problem. Both tasks must outlive it.
/// Snapshots are immutable, remain valid as the source grows, and cover all source slots reserved before the call.
class SlotMap
{
public:
    SlotMap(const Task& source, const Task& target);
    [[nodiscard]] std::shared_ptr<const std::vector<u32>> snapshot() const;

private:
    const Task& m_source;
    const Task& m_target;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<const std::vector<u32>> m_slots;
};

/// Clears destination, then maps each set source bit. Throws for a bit outside the map or destination slot space.
/// Numeric words are separate from these spans and retain their slot order within one normalized problem.
void map_fluent_words(SlotMapView map, std::span<const u64> source, std::span<u64> destination, u32 target_slots);
}  // namespace mymyr
