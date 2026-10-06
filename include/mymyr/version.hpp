#pragma once

#include <string_view>

namespace mymyr
{
/// Library version, set by CMake from the project version.
[[nodiscard]] std::string_view version() noexcept;
/// One-line build description (compiler, build type, options).
[[nodiscard]] std::string_view build_info() noexcept;
}  // namespace mymyr
