#pragma once
// The status names the reference tools print for a search status ("failed" for an exhausted ladder, "canceled",
// "error", ...), so that runs of both sides can be compared as text.

#include "mymyr/search/control.hpp"

namespace mymyr::parity
{
inline const char* reference_status_name(search::SearchStatus s) noexcept
{
    using search::SearchStatus;
    switch (s)
    {
        case SearchStatus::Solved: return "solved";
        case SearchStatus::Exhausted: return "failed";
        case SearchStatus::OutOfStates: return "out_of_states";
        case SearchStatus::OutOfTime: return "out_of_time";
        case SearchStatus::Cancelled: return "canceled";
        case SearchStatus::Failed: return "error";
        case SearchStatus::Unsolvable: return "unsolvable";
    }
    return "?";
}
}  // namespace mymyr::parity
