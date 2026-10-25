#include "mymyr/task/slot_map.hpp"

#include "mymyr/task/task.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace mymyr
{
SlotMap::SlotMap(const Task& source, const Task& target) : m_source(source), m_target(target)
{
    if (source.fingerprint() != target.fingerprint() || !target.atoms().frozen())
        throw std::invalid_argument("mymyr: a slot map needs one normalized problem and a frozen destination");
    if (source.numeric_words() != target.numeric_words() || source.numeric_storage() != target.numeric_storage() ||
        source.numeric_slots() != target.numeric_slots())
        throw std::invalid_argument("mymyr: slot map numeric layouts differ");
    for (u32 slot = 0; slot < source.numeric_slots(); ++slot)
        if (source.numeric_name(slot) != target.numeric_name(slot))
            throw std::invalid_argument("mymyr: slot map numeric slot order differs");
}

std::shared_ptr<const std::vector<u32>> SlotMap::snapshot() const
{
    std::lock_guard lock(m_mutex);
    const u32 size = m_source.atoms().fluent_slots();
    if (!m_slots || m_slots->size() < size)
    {
        auto slots = std::make_shared<std::vector<u32>>(m_slots ? *m_slots : std::vector<u32>{});
        slots->reserve(size);
        for (u32 slot = static_cast<u32>(slots->size()); slot < size; ++slot)
            slots->push_back(m_target.atoms().find(m_source.atoms().canonical(AtomKind::Fluent, slot)));
        m_slots = std::move(slots);
    }
    return m_slots;
}

void map_fluent_words(SlotMapView map, std::span<const u64> source, std::span<u64> destination, u32 target_slots)
{
    std::fill(destination.begin(), destination.end(), u64{0});
    for (usize word = 0; word < source.size(); ++word)
        for (u64 bits = source[word]; bits; bits &= bits - 1)
        {
            const u64 slot = word * 64 + std::countr_zero(bits);
            if (slot >= map.size || !map.data)
                throw std::invalid_argument("mymyr: encoding: source atom slot " + std::to_string(slot) +
                                            " is outside the slot map");
            const u32 mapped = map.data[slot];
            if (mapped >= target_slots || mapped / 64 >= destination.size())
                throw std::invalid_argument("mymyr: encoding: source atom slot " + std::to_string(slot) +
                                            " has no slot in the destination");
            destination[mapped / 64] |= u64{1} << (mapped % 64);
        }
}
}  // namespace mymyr
