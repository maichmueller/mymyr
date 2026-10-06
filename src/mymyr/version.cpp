#include "mymyr/version.hpp"

#ifndef MYMYR_VERSION_STRING
#define MYMYR_VERSION_STRING "0.0.0"
#endif
#ifndef MYMYR_BUILD_TYPE
#define MYMYR_BUILD_TYPE "unknown"
#endif

namespace mymyr
{
std::string_view version() noexcept { return MYMYR_VERSION_STRING; }

std::string_view build_info() noexcept
{
#if defined(__clang__)
#define MYMYR_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#define MYMYR_COMPILER "gcc " __VERSION__
#else
#define MYMYR_COMPILER "unknown compiler"
#endif
    return "mymyr " MYMYR_VERSION_STRING " (" MYMYR_BUILD_TYPE ", " MYMYR_COMPILER ")";
}
}  // namespace mymyr
