# libmymyr_cuda: the device runtime layer, the device task upload, device arenas and the smoke kernels; the lifted
# successor kernels, the device state set, the device BrFS and the device expand.
# Included from the top-level CMakeLists.txt when MYMYR_CUDA=ON, after enable_language(CUDA).
#
# Dialects:
#   - host code (*.cpp here, the tests, the Python bindings) is C++26 and built by the project's C++ compiler (GCC 16);
#     it includes <cuda_runtime_api.h>, a C header;
#   - device translation units (*.cu) are compiled by nvcc as C++23 (CUDA_STANDARD) with the project's GCC as nvcc's
#     host compiler (CMAKE_CUDA_HOST_COMPILER). They include only the device-code subset: core/types.hpp,
#     rl/task_arrays_view.hpp, cuda/kernels.hpp, cuda/lifted.hpp, cuda/state_set.hpp (and CUB / Thrust). Launchers
#     exchange PODs, raw pointers and cudaStream_t only, so no standard-library type crosses between the two compilers.

add_library(mymyr_cuda STATIC
    src/runtime.cpp
    src/arena.cpp
    src/device_task.cpp
    src/kernels.cu
    src/lifted.cu
    src/state_set.cu
    src/generator.cpp
    src/brfs.cpp
    src/expand.cpp
    src/multi_iw.cu
    src/multi_iw.cpp
    src/rollouts.cpp
)
add_library(mymyr::cuda ALIAS mymyr_cuda)
set_target_properties(mymyr_cuda PROPERTIES EXPORT_NAME cuda)
# the environment step on the device (cuda/env.hpp, cuda/env_kernels.hpp)
target_sources(mymyr_cuda PRIVATE src/env.cu src/env.cpp)
# novelty rewards, prefix masks and hindsight relabels on the device (cuda/rl_ops.hpp)
target_sources(mymyr_cuda PRIVATE src/rl_ops.cu)
# batched grounded heuristics, device A* and GBFS (cuda/heuristics.hpp, cuda/astar.hpp, cuda/gbfs.hpp)
target_sources(mymyr_cuda PRIVATE src/heuristics.cu src/h2.cu src/heuristics.cpp src/astar.cu src/astar.cpp src/gbfs.cpp src/numeric_best_first.cpp)
# axiom strata on the device
target_sources(mymyr_cuda PRIVATE src/axioms.cu)
# the kernel instantiations of the successor generator (conditional effects, wide rows, deep matchers, picks, count caches),
# generated: one translation unit per group of kernels (lifted_units.cmake), so that the compiles are small and parallel.
# The units are object libraries, so that those of the largest kernels can take a job pool of their own.
include(${CMAKE_CURRENT_LIST_DIR}/lifted_units.cmake)
foreach(units mymyr_cuda_lifted_units mymyr_cuda_lifted_heavy)
    if(units STREQUAL "mymyr_cuda_lifted_units")
        add_library(${units} OBJECT ${_mymyr_lifted_light_sources})
    else()
        add_library(${units} OBJECT ${_mymyr_lifted_heavy_sources})
    endif()
    target_include_directories(${units} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${units} PRIVATE mymyr::core mymyr::encoding CUDA::cudart_static
        $<BUILD_INTERFACE:mymyr_warnings>)
    target_compile_options(${units} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_compile_definitions(${units} PRIVATE MYMYR_HAS_CUDA=1)
    set_target_properties(${units} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY Static
        POSITION_INDEPENDENT_CODE ON)
    target_sources(mymyr_cuda PRIVATE $<TARGET_OBJECTS:${units}>)
endforeach()
# task tables on the device and the multi-instance launches of the kernels
target_sources(mymyr_cuda PRIVATE src/device_table.cpp src/table_launch.cpp src/lifted_multi.cu src/lifted_multi_write.cu src/lifted_multi_pick.cu)
# device state spaces, one task or the instances of a table in waves (cuda/state_space.hpp)
target_sources(mymyr_cuda PRIVATE src/state_space.cu src/state_space.cpp)
# IW over task tables (cuda/multi_iw.hpp: the searches grouped by instance)
target_sources(mymyr_cuda PRIVATE src/multi_iw_table.cpp)
# the relational encoder on the device: its context, scratch and kernels (cuda/encoding.hpp)
target_sources(mymyr_cuda PRIVATE src/encoding.cpp src/encoding.cu src/encoding_horizon.cu src/encoding_space_horizon.cu src/encoding_rows.cu src/encoding_stream.cpp src/encoding_collate.cpp src/encoding_collate.cu)
# task suites (several domains) on the device: the suite expand and the domain split of a batch
target_sources(mymyr_cuda PRIVATE src/suite.cu src/suite_expand.cpp)
# device action costs (cuda/cost_program.hpp: programs over static functions, the state spaces and A*)
target_sources(mymyr_cuda PRIVATE src/cost_program.cpp)
# Numeric programs use the CPU's operation order and rounding at every arithmetic instruction.
target_sources(mymyr_cuda PRIVATE src/numeric.cu src/numeric_pad.cu src/goal_program.cpp)
set_source_files_properties(src/numeric.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
# Most of what the encode kernels write is read by later kernels, not by the encoder: their global stores are
# cache-streaming (evict-first), which leaves the L2 to the rows and to the kernels around an encode.
set_source_files_properties(src/encoding.cu PROPERTIES COMPILE_OPTIONS "-Xptxas=-dscm=cs")
set_target_properties(mymyr_cuda PROPERTIES
    CUDA_STANDARD 23
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_RUNTIME_LIBRARY Static
    POSITION_INDEPENDENT_CODE ON)
# Device translation units need up to 3 GB in nvcc's front end and ptxas, and four kernels (conditional effects over
# 16 or more object-bitset words) up to 7 GB. With Ninja, -DCMAKE_JOB_POOLS=cuda_compile=<n> runs at most n of the
# library's compiles at a time while the rest of the build keeps its parallelism; cuda_compile_heavy=<m> additionally
# bounds the compiles of the four largest kernels (the units of mymyr_cuda_lifted_heavy), which otherwise share
# cuda_compile.
if(CMAKE_JOB_POOLS MATCHES "(^|;)cuda_compile=[0-9]+")
    set_target_properties(mymyr_cuda mymyr_cuda_lifted_units PROPERTIES JOB_POOL_COMPILE cuda_compile)
    set_target_properties(mymyr_cuda_lifted_heavy PROPERTIES JOB_POOL_COMPILE cuda_compile)
endif()
if(CMAKE_JOB_POOLS MATCHES "(^|;)cuda_compile_heavy=[0-9]+")
    set_target_properties(mymyr_cuda_lifted_heavy PROPERTIES JOB_POOL_COMPILE cuda_compile_heavy)
endif()
target_link_libraries(mymyr_cuda PUBLIC mymyr::core mymyr::encoding CUDA::cudart_static
    PRIVATE $<BUILD_INTERFACE:mymyr_warnings>)
target_compile_options(mymyr_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_definitions(mymyr_cuda PUBLIC MYMYR_HAS_CUDA=1)

# The gtests (tests/cuda/test_cuda.cpp) are declared in tests/cpp/CMakeLists.txt, where GTest is found.

# ------------------------------------------------------------------------------------------------ cost measurements
if(MYMYR_BENCH)
    add_executable(mymyr_device_costs_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_costs_bench.cpp)
    target_link_libraries(mymyr_device_costs_bench PRIVATE mymyr::cuda mymyr_warnings)
    target_compile_definitions(mymyr_device_costs_bench PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
    add_executable(mymyr_device_brfs_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_brfs_bench.cpp)
    target_link_libraries(mymyr_device_brfs_bench PRIVATE mymyr::cuda mymyr_warnings)
    add_executable(mymyr_device_iw_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_iw_bench.cpp)
    target_link_libraries(mymyr_device_iw_bench PRIVATE mymyr::cuda mymyr_warnings)
    # random-policy rollouts of device environments
    add_executable(mymyr_device_env_rollout_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_env_rollout_bench.cpp)
    target_link_libraries(mymyr_device_env_rollout_bench PRIVATE mymyr::cuda mymyr_warnings)
    # the batched device heuristics against the CPU heuristic, device A* and GBFS against the CPU searches
    add_executable(mymyr_device_heuristics_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_heuristics_bench.cpp)
    target_link_libraries(mymyr_device_heuristics_bench PRIVATE mymyr::cuda mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_device_heuristics_bench PRIVATE mymyr::frontend)
    endif()
    # device BrFS on tasks with conditional effects and axioms (text tasks, or PDDL through the front end)
    add_executable(mymyr_device_conditional_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_conditional_bench.cpp)
    target_link_libraries(mymyr_device_conditional_bench PRIVATE mymyr::cuda mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_device_conditional_bench PRIVATE mymyr::frontend)
        target_compile_definitions(mymyr_device_conditional_bench PRIVATE MYMYR_DEVICE_CONDITIONAL_FRONTEND=1)
    endif()
    # mixed-table env steps and expand against single instances, views memory, table construction and upload
    if(TARGET mymyr_frontend)
        add_executable(mymyr_table_bench ${PROJECT_SOURCE_DIR}/bench/cuda/table_bench.cpp)
        target_link_libraries(mymyr_table_bench PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings)
        target_compile_definitions(mymyr_table_bench PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
        # the device relational encoder on the rows of a task: captured replays, nodes of the graph, scratch
        add_executable(mymyr_device_encoding_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_encoding_bench.cpp)
        target_link_libraries(mymyr_device_encoding_bench PRIVATE mymyr::cuda mymyr::encoding mymyr::frontend mymyr_warnings)
        # device state spaces against the host (one large space; many small instances of a table)
        add_executable(mymyr_device_state_space_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_state_space_bench.cpp)
        target_link_libraries(mymyr_device_state_space_bench PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings)
        target_compile_definitions(mymyr_device_state_space_bench PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
        # suite batches (several domains) against one batch per domain, env steps and expand
        add_executable(mymyr_task_suite_bench ${PROJECT_SOURCE_DIR}/bench/cuda/task_suite_bench.cpp)
        target_link_libraries(mymyr_task_suite_bench PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings)
        target_compile_definitions(mymyr_task_suite_bench PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
    endif()
endif()

set_source_files_properties(src/numeric_best_first.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
