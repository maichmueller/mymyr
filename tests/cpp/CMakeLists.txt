# The root of the test data tree (tests/data), shared by every test target below instead of each one spelling out
# its own repository-relative path.
add_compile_definitions(MYMYR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")

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
    search/test_best_first.cpp
    heuristics/test_heuristics.cpp
    search/test_plan_file.cpp
    search/test_layer_orders_fork.cpp
    search/test_beam.cpp
    search/test_beam_fork.cpp
    # device task layout (device_arrays version 2), host side
    rl/test_task_arrays_v2.cpp
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

# ------------------------------------------------------------------------------------------------ PDDL front end
if(TARGET mymyr_frontend)
    # Where the golden tests find their PDDL: MYMYR_WORK_DIR holds mimir-cs/ and mimir/, MYMYR_FORK_DATA_DIR the
    # fork's data/ (ipc/rovers[-ipc]/test/p30-hard.pddl). Neither has a default location; both can be overridden at
    # run time with the environment variables MYMYR_WORK and MYMYR_FORK_DATA. Tasks whose PDDL is missing are skipped.
    set(MYMYR_WORK_DIR "${PROJECT_SOURCE_DIR}/.work" CACHE PATH "Work directory with the benchmark PDDL (mimir-cs, mimir)")
    set(MYMYR_FORK_DATA_DIR "" CACHE PATH "The mimir fork's data directory (IPC tasks)")

    add_executable(mymyr_export_task frontend/export_task.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_export_task PRIVATE mymyr::frontend mymyr_warnings)
    target_compile_definitions(mymyr_export_task PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}" MYMYR_WORK_DIR="${MYMYR_WORK_DIR}" MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")

    add_executable(mymyr_frontend_tests frontend/golden.cpp frontend/test_frontend.cpp)
    target_link_libraries(mymyr_frontend_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_frontend_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_frontend_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ task tables
if(TARGET mymyr_frontend)
    # Task tables on the host: mixed-instance expand, environments and the CPU pool on the instance sets of
    # tests/cpp/rl/table_instance_sets.hpp (the fork's PDDL through the front end, plus tests/data/table_instances); task
    # suites (instances of several domains, rl/test_suite.cpp)
    add_executable(mymyr_table_tests rl/test_table.cpp rl/test_suite.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_table_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_table_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_table_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ heuristics
if(TARGET mymyr_frontend)
    # Heuristic values along the fork's golden walks (tests/data/expected, env MYMYR_GOLDEN_DIR).
    add_executable(mymyr_heuristics_golden_tests heuristics/test_heuristics_golden.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_heuristics_golden_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_heuristics_golden_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_heuristics_golden_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 3600)
endif()

# ------------------------------------------------------------------------------------------------ best-first search
if(TARGET mymyr_frontend)
    # A* plan costs equal the fork's golden optimal costs and 0.13.60's (tests/data/expected).
    add_executable(mymyr_search_golden_tests search/test_best_first_golden.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_search_golden_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_search_golden_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_search_golden_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 3600)
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
    # the device environment step against the host step (rl::HostEnv) on the suite, the device RNG
    add_executable(mymyr_device_env_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_env.cpp)
    target_link_libraries(mymyr_device_env_tests PRIVATE mymyr::cuda mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_device_env_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(mymyr_device_env_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
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
        target_sources(mymyr_device_heuristics_tests PRIVATE frontend/golden.cpp)
        target_link_libraries(mymyr_device_heuristics_tests PRIVATE mymyr::frontend)
        target_compile_definitions(mymyr_device_heuristics_tests PRIVATE MYMYR_DEVICE_HEURISTICS_FRONTEND
                                                           MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
                                                           MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
    endif()
    gtest_discover_tests(mymyr_device_heuristics_tests DISCOVERY_TIMEOUT 60
        PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    # conditional effects and axiom strata on the device, on the suite's and the fork's tasks with them (PDDL
    # through the front end; tasks whose PDDL is missing are skipped)
    if(TARGET mymyr_frontend)
        add_executable(mymyr_device_conditional_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_conditional.cpp frontend/golden.cpp)
        target_link_libraries(mymyr_device_conditional_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_conditional_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        gtest_discover_tests(mymyr_device_conditional_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    endif()
    # the device expand and environments over task tables (mixed batches of several instances) against the host,
    # on the instance sets of tests/cpp/rl/table_instance_sets.hpp
    if(TARGET mymyr_frontend)
        add_executable(mymyr_device_table_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_table.cpp frontend/golden.cpp)
        target_link_libraries(mymyr_device_table_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_table_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        gtest_discover_tests(mymyr_device_table_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the device env's calls captured into CUDA graphs and replayed, against eager calls and the host
        add_executable(mymyr_device_graphs_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_graphs.cpp frontend/golden.cpp)
        target_link_libraries(mymyr_device_graphs_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_graphs_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        gtest_discover_tests(mymyr_device_graphs_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # device state spaces of one task and of tables (cuda/state_space.hpp) against the host
        add_executable(mymyr_device_state_space_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_state_space.cpp frontend/golden.cpp)
        target_link_libraries(mymyr_device_state_space_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_state_space_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        gtest_discover_tests(mymyr_device_state_space_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 3600 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        # the device expand and environments over task suites (instances of several domains) against the host
        add_executable(mymyr_device_task_suite_tests ${PROJECT_SOURCE_DIR}/tests/cuda/test_device_task_suite.cpp frontend/golden.cpp)
        target_link_libraries(mymyr_device_task_suite_tests PRIVATE mymyr::cuda mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_device_task_suite_tests PRIVATE
            MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
            MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}")
        gtest_discover_tests(mymyr_device_task_suite_tests DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 1800 LABELS cuda ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    endif()
endif()

# ------------------------------------------------------------------------------------------------ numerics and costs
target_sources(mymyr_tests PRIVATE task/test_numeric.cpp search/test_best_first_numeric.cpp)
if(TARGET mymyr_frontend)
    # the fork's numeric applicability rules, metric kinds and plan costs of tasks with numeric functions, from PDDL
    add_executable(mymyr_numeric_pddl_tests frontend/test_numeric_pddl.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_numeric_pddl_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_numeric_pddl_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_numeric_pddl_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()

# ------------------------------------------------------------------------------------------------ relaxed reachability and landmarks
add_executable(mymyr_landmarks_tests landmarks/test_landmarks.cpp)
target_link_libraries(mymyr_landmarks_tests PRIVATE mymyr::core mymyr_warnings GTest::gtest_main)
target_compile_definitions(mymyr_landmarks_tests PRIVATE MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(mymyr_landmarks_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)

# ------------------------------------------------------------------------------------------------ IW family variants
if(TARGET mymyr_frontend)
    # AIW / LIW / Rollout IW / parallel rollouts / portfolio equal the fork's recorded runs
    # (tests/data/iw_variants/fork_golden.json).
    add_executable(mymyr_iw_variants_golden_tests search/test_iw_variants_golden.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_iw_variants_golden_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_iw_variants_golden_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_iw_variants_golden_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 3600)
endif()

# ------------------------------------------------------------------------------------------------ datasets
if(TARGET mymyr_frontend)
    # state spaces against the fork on its own test instances (fork_cases.inc), determinism across thread counts,
    # options, costs, numeric tasks, the pool, generalized state spaces, samplers, object graphs and certificates
    add_executable(mymyr_datasets_tests datasets/test_datasets.cpp frontend/golden.cpp)
    target_link_libraries(mymyr_datasets_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_datasets_tests PRIVATE
        MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MYMYR_WORK_DIR="${MYMYR_WORK_DIR}"
        MYMYR_FORK_DATA_DIR="${MYMYR_FORK_DATA_DIR}"
        $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)
    gtest_discover_tests(mymyr_datasets_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800)
endif()
