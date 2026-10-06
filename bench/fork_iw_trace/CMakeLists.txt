cmake_minimum_required(VERSION 3.21)
project(fork_iw_trace CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# Links the mimir fork 0.16.3 installed by tests/data/fork_golden/build_fork.sh.
# Build (with whatever compiler the fork and its dependencies were built with):
#   cmake -S bench/fork_iw_trace -B build/fork-iw-trace -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=<compiler> \
#         -DCMAKE_PREFIX_PATH="$WORK/install-fork;$WORK/install-fork-deps" && cmake --build build/fork-iw-trace
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(fork_iw_trace fork_iw_trace.cpp)
target_link_libraries(fork_iw_trace mimir::core)
set_target_properties(fork_iw_trace PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
