cmake_minimum_required(VERSION 3.21)
project(fork_iw_variants CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# The IW-family variants run against the mimir fork 0.16.3, for parity and speed comparisons with mymyr_iw_variants:
# abstracted IW, LIW with exported landmarks, Rollout IW with portable randomized orderings, the per-rollout body of
# find_rollouts_parallel with a portable layer shuffle, and the atomic-goal portfolio. Links an install of the fork
# and its dependencies (built with the system GCC 11):
#   cmake -S bench/fork_iw_variants -B build/fork-iw-variants -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_PREFIX_PATH="<fork install>;<fork deps install>" \
#       && cmake --build build/fork-iw-variants
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(fork_iw_variants fork_iw_variants.cpp)
target_link_libraries(fork_iw_variants mimir::core)
set_target_properties(fork_iw_variants PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
