# Drivers that print the result lines the reference tools of the parity tests print, so the two sides can be compared
# run for run (see tests/parity/README.md). The reference tools themselves (fork_iw_trace, fork_iw_variants) are
# standalone projects in this directory.
include_directories(${PROJECT_SOURCE_DIR}/tests/parity/common)

add_executable(mymyr_brfs mymyr_brfs.cpp)
target_link_libraries(mymyr_brfs PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_brfs PRIVATE mymyr::frontend)
endif()
# IW / SIW / single passes; reads PDDL when the front end is built
add_executable(mymyr_iw mymyr_iw.cpp)
target_link_libraries(mymyr_iw PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_iw PRIVATE mymyr::frontend)
endif()
# heuristics and best-first search
add_executable(mymyr_heur mymyr_heur.cpp)
target_link_libraries(mymyr_heur PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_heur PRIVATE mymyr::frontend)
endif()
# relaxed reachability and landmarks
add_executable(mymyr_landmarks mymyr_landmarks.cpp)
target_link_libraries(mymyr_landmarks PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_landmarks PRIVATE mymyr::frontend)
endif()
# IW family variants: AIW / LIW / Rollout IW / parallel rollouts / portfolio
add_executable(mymyr_iw_variants mymyr_iw_variants.cpp)
target_link_libraries(mymyr_iw_variants PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_iw_variants PRIVATE mymyr::frontend)
endif()
# datasets: state spaces, generalized state spaces, samplers
add_executable(mymyr_datasets mymyr_datasets.cpp)
target_link_libraries(mymyr_datasets PRIVATE mymyr::core mymyr_warnings)
if(TARGET mymyr_frontend)
    target_link_libraries(mymyr_datasets PRIVATE mymyr::frontend)
endif()
