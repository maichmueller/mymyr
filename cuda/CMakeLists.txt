# libmymyr_cuda: the device runtime layer, the device task upload, device arenas and the smoke kernels; the lifted
# successor kernels, the device state set, the device BrFS and the device expand.
# Included from the top-level CMakeLists.txt when MYMYR_CUDA=ON, after enable_language(CUDA).
#
# Dialects:
#   - host code (*.cpp here, the tests, the Python bindings) is C++26 and built by the project's C++ compiler (GCC 16);
#     it includes <cuda_runtime_api.h>, a C header;
#   - device translation units (*.cu) are compiled by nvcc as C++20 (nvcc's newest dialect) with a host compiler nvcc
#     supports (CMAKE_CUDA_HOST_COMPILER; the top level picks g++-15..11 when the C++ compiler is GCC >= 16, whose
#     libstdc++ headers nvcc's front end cannot parse). They include only the device-code subset: core/types.hpp,
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
# the environment step on the device (cuda/env.hpp, cuda/env_kernels.hpp)
target_sources(mymyr_cuda PRIVATE src/env.cu src/env.cpp)
# novelty rewards, prefix masks and hindsight relabels on the device (cuda/rl_ops.hpp)
target_sources(mymyr_cuda PRIVATE src/rl_ops.cu)
# batched grounded heuristics, device A* and GBFS (cuda/heuristics.hpp, cuda/astar.hpp, cuda/gbfs.hpp)
target_sources(mymyr_cuda PRIVATE src/heuristics.cu src/heuristics.cpp src/astar.cu src/astar.cpp src/gbfs.cpp)
# rows of schemas with conditional effects (sub-plans at the leaf) and axiom strata on the device; the kernels share
# src/lifted_device.cuh with lifted.cu and are instantiated in units of their own (parallel builds)
target_sources(mymyr_cuda PRIVATE src/lifted_ce.cu src/lifted_ce16.cu src/lifted_ce64.cu src/axioms.cu src/lifted_deep.cu
                                   src/lifted_deep64.cu)
# task tables on the device and the multi-instance launches of the kernels (units of their own: parallel builds)
target_sources(mymyr_cuda PRIVATE src/device_table.cpp src/table_launch.cpp src/lifted_multi.cu src/lifted_multi_write.cu src/lifted_multi_pick.cu)
# device state spaces, one task or the instances of a table in waves (cuda/state_space.hpp)
target_sources(mymyr_cuda PRIVATE src/state_space.cu src/state_space.cpp)
# IW over task tables (cuda/multi_iw.hpp: the searches grouped by instance)
target_sources(mymyr_cuda PRIVATE src/multi_iw_table.cpp)
# task suites (several domains) on the device: the suite expand and the domain split of a batch
target_sources(mymyr_cuda PRIVATE src/suite.cu src/suite_expand.cpp)
# device action costs (cuda/cost_program.hpp: programs over static functions, the state spaces and A*)
target_sources(mymyr_cuda PRIVATE src/cost_program.cpp)
set_target_properties(mymyr_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_RUNTIME_LIBRARY Static
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mymyr_cuda PUBLIC mymyr::core CUDA::cudart_static)
target_compile_options(mymyr_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall,-Wextra --Werror=all-warnings>)
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
        target_compile_definitions(mymyr_device_heuristics_bench PRIVATE MYMYR_HAVE_FRONTEND)
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
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        # device state spaces against the host (one large space; many small instances of a table)
        add_executable(mymyr_device_state_space_bench ${PROJECT_SOURCE_DIR}/bench/cuda/device_state_space_bench.cpp)
        target_link_libraries(mymyr_device_state_space_bench PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings)
        target_compile_definitions(mymyr_device_state_space_bench PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        # suite batches (several domains) against one batch per domain, env steps and expand
        add_executable(mymyr_task_suite_bench ${PROJECT_SOURCE_DIR}/bench/cuda/task_suite_bench.cpp)
        target_link_libraries(mymyr_task_suite_bench PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings)
        target_compile_definitions(mymyr_task_suite_bench PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
    endif()
endif()
