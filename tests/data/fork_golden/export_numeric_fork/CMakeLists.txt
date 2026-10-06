cmake_minimum_required(VERSION 3.21)
project(export_numeric_fork CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# Links the mimir fork installed by build_fork.sh ($WORK/install-fork) and its dependency installs.
# Build dir: $WORK/build-numeric-exporter (see ../build_fork.sh).
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(export_numeric_fork export_numeric_fork.cpp)
target_link_libraries(export_numeric_fork mimir::core)
set_target_properties(export_numeric_fork PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
