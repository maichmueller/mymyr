cmake_minimum_required(VERSION 3.21)
project(search_fork CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# Links the mimir fork installed by ../build_fork.sh ($WORK/install-fork) and its dependency installs.
# Build: run_numeric.py --build (into $WORK/build-search-fork, or --build-dir).
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(search_fork search_fork.cpp)
target_link_libraries(search_fork mimir::core)
set_target_properties(search_fork PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
