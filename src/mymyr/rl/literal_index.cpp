#include "mymyr/encoding/literal_index.hpp"
#include <algorithm>
#include <stdexcept>

namespace mymyr::encoding::detail
{
void check_slot_range(std::span<const i32> pred, std::span<const i32> padded, u32 width, u32 begin, u32 end,
                              u32 predicate, u32 arity, const std::string& name)
{
    auto not_a_range = [&]
    { throw std::logic_error("mymyr: encoding: the atom slots of predicate " + name + " are not one ascending range"); };
    for (u32 slot = begin; slot < end; ++slot)
    {
        if (static_cast<u32>(pred[slot]) != predicate)
            not_a_range();
        if (slot > begin &&
            !std::lexicographical_compare(padded.begin() + static_cast<std::ptrdiff_t>((slot - 1) * u64{width}),
                                          padded.begin() + static_cast<std::ptrdiff_t>((slot - 1) * u64{width} + arity),
                                          padded.begin() + static_cast<std::ptrdiff_t>(slot * u64{width}),
                                          padded.begin() + static_cast<std::ptrdiff_t>(slot * u64{width} + arity)))
            not_a_range();
    }
}

}
