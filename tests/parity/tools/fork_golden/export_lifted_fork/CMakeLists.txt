cmake_minimum_required(VERSION 3.21)
project(export_lifted_fork CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Links the mimir fork installed by ../build_fork.sh ($WORK/install-fork) and its prebuilt deps.
find_package(mimir CONFIG REQUIRED COMPONENTS core)
add_executable(export_lifted_fork export_lifted_fork.cpp)
target_link_libraries(export_lifted_fork mimir::core)
set_target_properties(export_lifted_fork PROPERTIES BUILD_RPATH "${mimir_DIR}/../..")
