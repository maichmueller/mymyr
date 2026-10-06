cmake_minimum_required(VERSION 3.21)
project(fork_golden CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# Links the mimir fork installed by build_fork.sh ($WORK/install-fork) and its dependency installs.
# Build: tests/data/fork_golden/build.sh.
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(export_golden export_golden.cpp)
target_link_libraries(export_golden mimir::core)
set_target_properties(export_golden PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
