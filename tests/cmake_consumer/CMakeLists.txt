# A downstream C++ project that uses an installed mymyr (a `cmake --install` prefix or the installed wheel):
#
#   cmake -S tests/cmake_consumer -B build/consumer -Dmymyr_DIR=<prefix>/lib/cmake/mymyr
#   cmake --build build/consumer
#   build/consumer/consumer tests/data/pddl/logistics00/domain.pddl tests/cmake_consumer/data/problem.pddl
#
# With -DMYMYR_CONSUMER_FRONTEND=OFF the project uses the core component only and reads a normalized task file
# (tests/cmake_consumer/data/task.txt) instead of PDDL. ci/check_cmake_consumer.sh runs both.
cmake_minimum_required(VERSION 3.25)
project(mymyr_cmake_consumer LANGUAGES CXX)

option(MYMYR_CONSUMER_FRONTEND "Use the PDDL front end component" ON)
if(MYMYR_CONSUMER_FRONTEND)
    find_package(mymyr CONFIG REQUIRED COMPONENTS core frontend)
else()
    find_package(mymyr CONFIG REQUIRED COMPONENTS core)
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE mymyr::core)
if(MYMYR_CONSUMER_FRONTEND)
    target_link_libraries(consumer PRIVATE mymyr::frontend)
endif()
# A GCC outside the system prefix ships a newer libstdc++ than the loader finds by default.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(consumer PRIVATE -static-libstdc++ -static-libgcc)
endif()
