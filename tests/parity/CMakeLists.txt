# Included by tests/cpp/CMakeLists.txt, so that it shares the test scope (GTest, warnings, test data definitions).
# ------------------------------------------------------------------------------------------------ parity
# Tests against results recorded from a reference implementation (tests/parity/README.md). They read the recorded
# results from tests/parity/data and, where they need PDDL of the reference data, from MYMYR_PARITY_DATA (env; else
# the configured MYMYR_PARITY_DATA_DIR) and skip when it is absent. All carry the ctest label `parity`.
set(MYMYR_PARITY_DATA_DIR "" CACHE PATH "Directory with the reference data of the parity tests (see tests/parity/README.md)")
set(PARITY_DIR ${PROJECT_SOURCE_DIR}/tests/parity/cpp)
set(PARITY_DEFS MYMYR_SOURCE_DIR="${PROJECT_SOURCE_DIR}" MYMYR_PARITY_DATA_DIR="${MYMYR_PARITY_DATA_DIR}"
    $<$<BOOL:${MYMYR_SANITIZE}>:MYMYR_SANITIZED>)

# recorded results that need no PDDL
add_executable(mymyr_parity_tests ${PARITY_DIR}/test_iw_parity.cpp ${PARITY_DIR}/test_astar_iw_parity.cpp
    ${PARITY_DIR}/test_beam_parity.cpp ${PARITY_DIR}/test_layer_orders_parity.cpp ${PARITY_DIR}/test_best_first_numeric_parity.cpp)
target_include_directories(mymyr_parity_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/parity/common)
target_link_libraries(mymyr_parity_tests PRIVATE mymyr::core mymyr_warnings GTest::gtest_main)
target_compile_definitions(mymyr_parity_tests PRIVATE ${PARITY_DEFS})
gtest_discover_tests(mymyr_parity_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800 LABELS parity)

if(TARGET mymyr_frontend)
    add_executable(mymyr_export_task ${PARITY_DIR}/export_task.cpp ${PARITY_DIR}/golden.cpp)
    target_link_libraries(mymyr_export_task PRIVATE mymyr::frontend mymyr_warnings)
    target_compile_definitions(mymyr_export_task PRIVATE ${PARITY_DEFS})

    # one executable per area, each linking the shared golden comparison
    foreach(area frontend heuristics bindings best_first iw_variants)
        add_executable(mymyr_${area}_parity_tests ${PARITY_DIR}/test_${area}_parity.cpp ${PARITY_DIR}/golden.cpp)
        target_include_directories(mymyr_${area}_parity_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/parity/common)
        target_link_libraries(mymyr_${area}_parity_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
        target_compile_definitions(mymyr_${area}_parity_tests PRIVATE ${PARITY_DEFS})
        gtest_discover_tests(mymyr_${area}_parity_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 3600 LABELS parity)
    endforeach()

    add_executable(mymyr_datasets_parity_tests ${PARITY_DIR}/test_datasets_parity.cpp ${PARITY_DIR}/test_tuple_graphs_parity.cpp
        ${PARITY_DIR}/golden.cpp)
    target_link_libraries(mymyr_datasets_parity_tests PRIVATE mymyr::frontend mymyr_warnings GTest::gtest_main)
    target_compile_definitions(mymyr_datasets_parity_tests PRIVATE ${PARITY_DEFS})
    gtest_discover_tests(mymyr_datasets_parity_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 1800 LABELS parity)
endif()
