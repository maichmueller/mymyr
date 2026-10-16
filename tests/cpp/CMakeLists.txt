# The root of the test data tree (tests/data), shared by every test target below instead of each one spelling out
# its own repository-relative path.
add_compile_definitions(MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
    MYMYR_PARITY_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/parity/data")

# googletest: prefer an installed package (e.g. dependencies/installs); otherwise fetch the pinned release.
find_package(GTest 1.12 CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(mymyr_tests
    core/test_bitset.cpp
    core/test_hash.cpp
    core/test_arena.cpp
    core/test_per_thread.cpp
    core/test_once.cpp
    core/test_block_cache.cpp
    core/test_version.cpp
    formalism/test_text_format.cpp
    task/test_task.cpp
    successor/test_successors.cpp
    successor/test_bindings.cpp
    state/test_stores.cpp
    search/test_brfs.cpp
    novelty/test_novelty_table.cpp
    search/test_iw.cpp
    rl/test_expand.cpp
    search/test_astar_iw.cpp
    search/test_best_first.cpp
    search/test_wastar_ehc_anytime.cpp
    heuristics/test_heuristics.cpp
    heuristics/test_landmark_heuristics.cpp
    search/test_plan_file.cpp
    search/test_beam.cpp
    search/test_beam_parallel.cpp
    # device task layout (device_arrays version 2), host side
    rl/test_task_arrays_v2.cpp
    rl/test_numeric_device_arrays.cpp
    # host environments and the counter-based RNG
    rl/test_env.cpp
    # novelty rewards, prefix masks, hindsight relabels
    rl/test_ops.cpp
    # IW family variants
    search/test_iw_variants.cpp
    search/test_rollouts.cpp
)
target_link_libraries(mymyr_tests PRIVATE mymyr::core mymyr_warnings GTest::gtest_main)
target_compile_definitions(mymyr_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
include(GoogleTest)
gtest_discover_tests(mymyr_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)

add_executable(mymyr_encoding_tests encoding/test_schema.cpp encoding/test_lanes.cpp
    encoding/test_validate.cpp)
target_link_libraries(mymyr_encoding_tests PRIVATE mymyr::encoding mymyr_warnings GTest::gtest_main)
gtest_discover_tests(mymyr_encoding_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600 LABELS encoding)

# The encoder over PDDL tasks (tests/data/pddl) through the front end.
if(TARGET mymyr_frontend)
    add_executable(mymyr_encoder_tests encoding/test_context.cpp encoding/test_encoder.cpp
        encoding/test_encoder_modes.cpp encoding/test_encoder_goals.cpp encoding/test_expand_derived.cpp)
    target_link_libraries(mymyr_encoder_tests PRIVATE mymyr::encoding mymyr::frontend mymyr_warnings GTest::gtest_main)
    gtest_discover_tests(mymyr_encoder_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800 LABELS encoding)
endif()

# Thread counts and thread creation; interposes pthread_create to simulate refused threads, so in an executable of
# its own.
add_executable(mymyr_thread_tests core/test_threads.cpp)
target_link_libraries(mymyr_thread_tests PRIVATE mymyr::core mymyr_warnings GTest::gtest_main ${CMAKE_DL_LIBS})
gtest_discover_tests(mymyr_thread_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 600)

# ------------------------------------------------------------------------------------------------ PDDL front end
if(TARGET mymyr_frontend)
    # The benchmark PDDL of the tests is vendored under tests/data (benchmarks/, ipc/, pddl/).
    add_executable(mymyr_frontend_tests frontend/test_frontend.cpp frontend/test_encoding_goal.cpp search/test_astar_iw_pddl.cpp
        search/test_wastar_ehc_anytime_pddl.cpp)
    target_link_libraries(mymyr_frontend_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_frontend_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_frontend_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ task tables
if(TARGET mymyr_frontend)
    # Task tables on the host: mixed-instance expand, environments and the CPU pool on the instance sets of
    # tests/cpp/rl/table_instance_sets.hpp; task suites (instances of several domains, rl/test_suite.cpp)
    add_executable(mymyr_table_tests rl/test_table.cpp rl/test_suite.cpp)
    target_link_libraries(mymyr_table_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_table_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_table_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ CUDA
if(TARGET mymyr_cuda)
    # Device memory, arenas and tail syncs, the device task upload round trip and the smoke kernels against the CPU
    # engine on the suite; the device BrFS and the device expand against the CPU; multi-search IW and device
    # rollouts against search::iw() and find_rollouts_parallel. Runs on GPU 0 only; skips when no device is visible.
    add_executable(mymyr_cuda_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_cuda.cpp
                                    ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_brfs.cpp
                                    ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_iw.cpp
                                    ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_deep_matchers.cpp)
    target_link_libraries(mymyr_cuda_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_cuda_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(mymyr_cuda_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    if(TARGET mymyr_frontend)
        add_executable(mymyr_device_numeric_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_numeric.cpp)
        target_link_libraries(mymyr_device_numeric_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_numeric_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_numeric_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    endif()
    # the device environment step against the host step (rl::HostEnv) on the suite, the device RNG
    add_executable(mymyr_device_env_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_env.cpp)
    target_link_libraries(mymyr_device_env_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_device_env_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(mymyr_device_env_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    # the device Philox against cuRAND's: needs the toolkit's cuRAND headers (a separate package in some distributions)
    find_path(MYMYR_CURAND_INCLUDE_DIR curand_philox4x32_x.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    if(MYMYR_CURAND_INCLUDE_DIR)
        add_executable(mymyr_device_curand_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_curand_philox.cpp
                                                 ${PROJECT_SOURCE_DIR}/tests/cuda/curand_philox.cu)
        target_link_libraries(mymyr_device_curand_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
        target_include_directories(mymyr_device_curand_tests PRIVATE ${MYMYR_CURAND_INCLUDE_DIR})
        set_target_properties(mymyr_device_curand_tests PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED ON
                                                                    CUDA_EXTENSIONS OFF)
        gtest_discover_tests(mymyr_device_curand_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    else()
        message(STATUS "mymyr: cuRAND headers not found; the Philox-against-cuRAND test is not built")
    endif()
    # the device kernels of novelty rewards, prefix masks and hindsight relabels against the host loops; the
    # count cache's recorded keys (sokoban's deep segments without a second search)
    add_executable(mymyr_device_rl_ops_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_rl_ops.cpp)
    target_link_libraries(mymyr_device_rl_ops_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_device_rl_ops_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(mymyr_device_rl_ops_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    # the batched device heuristics against heuristics::make_heuristic, device A* and GBFS against the CPU searches
    add_executable(mymyr_device_heuristics_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_heuristics.cpp)
    target_link_libraries(mymyr_device_heuristics_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_device_heuristics_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    if(TARGET mymyr_frontend)
        # PDDL tasks with real action costs
        target_sources(mymyr_device_heuristics_tests PRIVATE)
        target_link_libraries(mymyr_device_heuristics_tests PRIVATE mymyr::frontend)
        target_compile_definitions(mymyr_device_heuristics_tests PRIVATE MYMYR_DEVICE_HEURISTICS_FRONTEND)
    endif()
    gtest_discover_tests(mymyr_device_heuristics_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    # conditional effects and axiom strata on the device, on the suite's and further tasks with them (PDDL
    # through the front end; tasks whose PDDL is missing are skipped)
    if(TARGET mymyr_frontend)
        add_executable(mymyr_device_conditional_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_conditional.cpp)
        target_link_libraries(mymyr_device_conditional_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_conditional_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_conditional_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    endif()
    # the device expand and environments over task tables (mixed batches of several instances) against the host,
    # on the instance sets of tests/cpp/rl/table_instance_sets.hpp
    if(TARGET mymyr_frontend)
        add_executable(mymyr_device_table_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_table.cpp)
        target_link_libraries(mymyr_device_table_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_table_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_table_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the derived words of the device expand against rl::expand
        add_executable(mymyr_device_derived_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_derived.cpp)
        target_link_libraries(mymyr_device_derived_tests PRIVATE mymyr::cuda mymyr::encoding mymyr::frontend mymyr_warnings GTest::gtest_main)
        gtest_discover_tests(mymyr_device_derived_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the relational encoder's context and memory on the device (cuda/encoding.hpp) against the host
        add_executable(mymyr_device_encoding_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_encoding.cpp
                                                   ${PROJECT_SOURCE_DIR}/tests/cuda/context_probe.cu)
        target_link_libraries(mymyr_device_encoding_tests PRIVATE mymyr::cuda mymyr::encoding mymyr::frontend GTest::gtest_main)
        target_link_libraries(mymyr_device_encoding_tests PRIVATE mymyr_warnings)
        set_target_properties(mymyr_device_encoding_tests PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
        target_compile_definitions(mymyr_device_encoding_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_encoding_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the device env's calls captured into CUDA graphs and replayed, against eager calls and the host
        add_executable(mymyr_device_graphs_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_graphs.cpp)
        target_link_libraries(mymyr_device_graphs_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_graphs_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_graphs_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # device state spaces of one task and of tables (cuda/state_space.hpp) against the host
        add_executable(mymyr_device_state_space_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_state_space.cpp)
        target_link_libraries(mymyr_device_state_space_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_state_space_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_state_space_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 3600 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the device expand and environments over task suites (instances of several domains) against the host
        add_executable(mymyr_device_task_suite_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_task_suite.cpp)
        target_link_libraries(mymyr_device_task_suite_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_task_suite_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        gtest_discover_tests(mymyr_device_task_suite_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    endif()
endif()

# ------------------------------------------------------------------------------------------------ numerics and costs
target_sources(mymyr_tests PRIVATE task/test_numeric.cpp search/test_best_first_numeric.cpp)
if(TARGET mymyr_frontend)
    # numeric applicability rules, metric kinds and plan costs of tasks with numeric functions, from PDDL; the PDDL
    # semantics of union types, numeric expressions, undefined values, metrics and heuristic costs; ADL conditions
    # (quantifiers, implication, disjunction, nesting) in every position
    add_executable(mymyr_numeric_pddl_tests frontend/test_numeric_pddl.cpp frontend/test_pddl_semantics.cpp frontend/test_adl_conditions.cpp)
    target_link_libraries(mymyr_numeric_pddl_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_numeric_pddl_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_numeric_pddl_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
    # LM-cut and landmark count on hand-evaluated PDDL tasks
    add_executable(mymyr_landmark_heuristics_tests heuristics/test_landmark_heuristics_pddl.cpp)
    target_link_libraries(mymyr_landmark_heuristics_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    gtest_discover_tests(mymyr_landmark_heuristics_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ relaxed reachability and landmarks
add_executable(mymyr_landmarks_tests landmarks/test_landmarks.cpp)
target_link_libraries(mymyr_landmarks_tests PRIVATE mymyr::core mymyr_warnings GTest::gtest_main)
target_compile_definitions(mymyr_landmarks_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(mymyr_landmarks_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)

# ------------------------------------------------------------------------------------------------ datasets
if(TARGET mymyr_frontend)
    # state spaces on small instances (small_instances.inc), determinism across thread counts, options, costs,
    # numeric tasks, the pool, generalized state spaces, samplers; tuple graphs, knowledge bases
    add_executable(mymyr_datasets_tests datasets/test_datasets.cpp datasets/test_tuple_graphs.cpp
        datasets/test_knowledge_base.cpp)
    target_link_libraries(mymyr_datasets_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_datasets_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_datasets_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ parity
# Tests against recorded reference results, in the same scope (tests/parity/README.md).
include(${PROJECT_SOURCE_DIR}/tests/parity/CMakeLists.txt)
