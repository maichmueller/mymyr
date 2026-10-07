cmake_minimum_required(VERSION 3.25)

# The package version lives in pyproject.toml. MYMYR_VERSION is the full PEP 440 string (it may carry
# a pre-release suffix); project() receives its numeric release part.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _mymyr_pyproject)
if(NOT _mymyr_pyproject MATCHES "\n *version *= *\"(([0-9]+\\.[0-9]+\\.[0-9]+)[^\"]*)\"")
    message(FATAL_ERROR "mymyr: no version = \"X.Y.Z...\" entry found in pyproject.toml")
endif()
set(MYMYR_VERSION "${CMAKE_MATCH_1}")
project(mymyr VERSION ${CMAKE_MATCH_2} LANGUAGES CXX)

# ------------------------------------------------------------------------------------------------ options
option(MYMYR_PYTHON "Build the Python extension (set by scikit-build-core)" OFF)
option(MYMYR_TESTS "Build the C++ unit tests" ON)
option(MYMYR_CUDA "Build the CUDA backend (libmymyr_cuda)" OFF)
option(MYMYR_FRONTEND "Build the loki PDDL front end (mymyr_frontend); needs the dependency superbuild" ON)
option(MYMYR_BENCH "Build the benchmark drivers (bench/cpp)" ON)
option(MYMYR_INSTALL_SDK "Install the C++ SDK: headers, static libraries and the mymyr CMake package" ON)
# Where the SDK lies inside the install prefix, empty or ending in '/': the wheel keeps it inside the mymyr package.
if(MYMYR_PYTHON)
    set(_mymyr_sdk_dir_default "mymyr/")
else()
    set(_mymyr_sdk_dir_default "")
endif()
set(MYMYR_SDK_DIR "${_mymyr_sdk_dir_default}" CACHE STRING "Directory of the C++ SDK inside the install prefix")
set(MYMYR_SANITIZE "" CACHE STRING "Sanitizers to enable: address;undefined or thread")
set(MYMYR_DEPS_PREFIX "${PROJECT_SOURCE_DIR}/dependencies/installs" CACHE PATH
    "Install prefix of the dependency superbuild (dependencies/CMakeLists.txt)")
option(MYMYR_DEPS_SANITIZED "The dependencies in MYMYR_DEPS_PREFIX are built with MYMYR_SANITIZE's ASan" OFF)
if(EXISTS "${MYMYR_DEPS_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${MYMYR_DEPS_PREFIX}")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mymyr_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mymyr_warnings INTERFACE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# A GCC outside the system prefix ships a newer libstdc++ than the loader finds by default. Test and tool
# executables get a build RPATH to it; the Python module links it statically.
set(MYMYR_LIBSTDCXX_DIR "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE _mymyr_libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_ABSOLUTE "${_mymyr_libstdcxx}")
        file(REAL_PATH "${_mymyr_libstdcxx}" _mymyr_libstdcxx)
        get_filename_component(_mymyr_libdir "${_mymyr_libstdcxx}" DIRECTORY)
        if(NOT _mymyr_libdir MATCHES "^/(usr/)?lib")
            set(MYMYR_LIBSTDCXX_DIR "${_mymyr_libdir}")
            list(APPEND CMAKE_BUILD_RPATH "${MYMYR_LIBSTDCXX_DIR}")
        endif()
    endif()
endif()

# Apple's linker warns when a static library appears twice on a link line, which CMake does for libraries that
# depend on each other.
if(APPLE)
    include(CheckLinkerFlag)
    check_linker_flag(CXX "LINKER:-no_warn_duplicate_libraries" MYMYR_HAS_NO_WARN_DUPLICATE_LIBRARIES)
    if(MYMYR_HAS_NO_WARN_DUPLICATE_LIBRARIES)
        add_link_options(LINKER:-no_warn_duplicate_libraries)
    endif()
endif()

if(MYMYR_SANITIZE)
    string(REPLACE ";" "," _mymyr_san "${MYMYR_SANITIZE}")
    add_compile_options(-fsanitize=${_mymyr_san} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${_mymyr_san})
endif()

# ------------------------------------------------------------------------------------------------ core library
add_library(mymyr_core STATIC
    src/mymyr/version.cpp
    src/mymyr/core/memory.cpp
    src/mymyr/formalism/task_data.cpp
    src/mymyr/formalism/text_format.cpp
    src/mymyr/task/atom_index.cpp
    src/mymyr/task/compile.cpp
    src/mymyr/task/task.cpp
    src/mymyr/task/workspace.cpp
    src/mymyr/successor/engine.cpp
    src/mymyr/successor/successors.cpp
    src/mymyr/successor/bindings.cpp
    src/mymyr/axioms/evaluator.cpp
    src/mymyr/search/brfs.cpp
    # IW family
    src/mymyr/novelty/novelty_table.cpp
    src/mymyr/search/iw.cpp
    src/mymyr/search/siw.cpp
    # task fingerprint, rl.expand, task arrays
    src/mymyr/formalism/fingerprint.cpp
    src/mymyr/rl/expand.cpp
    src/mymyr/rl/task_arrays.cpp
    src/mymyr/rl/device_search.cpp
    # task tables (many instances of one domain) and the CPU env pool
    src/mymyr/rl/task_table.cpp
    src/mymyr/rl/pool.cpp
    # task suites (instances of several domains)
    src/mymyr/rl/task_suite.cpp
    # environments (the host step, the reference of the device step)
    src/mymyr/rl/env.cpp
    # RL helpers: novelty rewards, prefix masks, hindsight relabels
    src/mymyr/rl/ops.cpp
    # heuristics and best-first search
    src/mymyr/heuristics/action_costs.cpp
    src/mymyr/heuristics/relaxed_task.cpp
    src/mymyr/heuristics/heuristics.cpp
    src/mymyr/heuristics/h2.cpp
    src/mymyr/heuristics/lifted_relaxation.cpp
    src/mymyr/search/best_first.cpp
    src/mymyr/search/astar_iw.cpp
    src/mymyr/novelty/minimum_g_table.cpp
    src/mymyr/search/astar.cpp
    src/mymyr/search/gbfs.cpp
    src/mymyr/search/beam.cpp
    src/mymyr/search/plan_file.cpp
    # relaxed reachability and landmarks
    src/mymyr/reachability/join.cpp
    src/mymyr/reachability/relaxed_reachability.cpp
    src/mymyr/landmarks/approximate.cpp
    src/mymyr/landmarks/lifted.cpp
    src/mymyr/search/iw_ordered.cpp
    src/mymyr/landmarks/transition_ordering.cpp
    # IW family variants
    src/mymyr/novelty/landmark_table.cpp
    src/mymyr/search/aiw.cpp
    src/mymyr/search/liw.cpp
    src/mymyr/search/rollout_iw.cpp
    src/mymyr/search/parallel_rollouts.cpp
    src/mymyr/search/portfolio.cpp
)
add_library(mymyr::core ALIAS mymyr_core)
set_target_properties(mymyr_core PROPERTIES EXPORT_NAME core)
target_include_directories(mymyr_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${MYMYR_SDK_DIR}include>)
target_compile_features(mymyr_core PUBLIC cxx_std_26)
target_compile_definitions(mymyr_core PRIVATE
    MYMYR_VERSION_STRING="${MYMYR_VERSION}"
    MYMYR_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_link_libraries(mymyr_core PRIVATE $<BUILD_INTERFACE:mymyr_warnings>)
find_package(Threads REQUIRED)
target_link_libraries(mymyr_core PUBLIC Threads::Threads)

# datasets: state spaces, generalized state spaces, samplers, object graphs, certificates
target_sources(mymyr_core PRIVATE
    src/mymyr/datasets/state_space.cpp
    src/mymyr/heuristics/perfect.cpp
    src/mymyr/datasets/generalized_state_space.cpp
    src/mymyr/datasets/sampler.cpp
    src/mymyr/datasets/object_graph.cpp
    src/mymyr/datasets/certificates.cpp
)

# ------------------------------------------------------------------------------------------------ PDDL front end
# loki (unpatched) + abseil + fmt from the dependency superbuild. Its own library so that the boost/hana/spirit
# compile cost stays out of the core.
if(MYMYR_FRONTEND)
    find_package(loki CONFIG QUIET COMPONENTS parsers)
    if(NOT loki_FOUND OR NOT TARGET loki::parsers)
        message(FATAL_ERROR
            "MYMYR_FRONTEND=ON needs loki, abseil, fmt and boost from the dependency superbuild, which were not found "
            "under MYMYR_DEPS_PREFIX='${MYMYR_DEPS_PREFIX}'. Build them with\n"
            "  cmake -S dependencies -B dependencies/build -G Ninja && cmake --build dependencies/build\n"
            "(set CC/CXX in the environment of both commands: the sub-builds inherit the compiler only from there), "
            "point -DMYMYR_DEPS_PREFIX=<prefix> at an existing install, or configure with -DMYMYR_FRONTEND=OFF.")
    endif()
    find_package(absl CONFIG REQUIRED)
    add_library(mymyr_frontend STATIC
        src/mymyr/frontend/domain.cpp
        src/mymyr/frontend/init_reader.cpp
        src/mymyr/frontend/translate.cpp
    )
    add_library(mymyr::frontend ALIAS mymyr_frontend)
    set_target_properties(mymyr_frontend PROPERTIES EXPORT_NAME frontend)
    # The parser dependencies are linked for the build tree only; the installed archive carries them (cmake/MymyrSdk.cmake).
    target_link_libraries(mymyr_frontend
        PUBLIC mymyr::core
        PRIVATE $<BUILD_INTERFACE:loki::parsers> $<BUILD_INTERFACE:absl::flat_hash_map>
                $<BUILD_INTERFACE:absl::flat_hash_set> $<BUILD_INTERFACE:absl::inlined_vector>
                $<BUILD_INTERFACE:absl::hash> $<BUILD_INTERFACE:absl::strings> $<BUILD_INTERFACE:mymyr_warnings>)
    target_compile_definitions(mymyr_frontend PUBLIC MYMYR_HAS_FRONTEND=1)
    # GCC 16 at -O3 reports -Wmaybe-uninitialized inside abseil's InlinedVector/raw_hash_set once they are inlined
    # into the front end (false positives; the headers are -isystem, the warning is issued after inlining).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(mymyr_frontend PRIVATE -Wno-maybe-uninitialized)
    endif()
    # GCC's -fsanitize=null / nonnull-attribute / returns-nonnull-attribute make the pinned abseil's constexpr
    # comparison of a function address with nullptr (hash_policy_traits.h) non-constant, which fails to compile.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND "undefined" IN_LIST MYMYR_SANITIZE)
        target_compile_options(mymyr_frontend PRIVATE -fno-sanitize=null,nonnull-attribute,returns-nonnull-attribute)
    endif()
    # Under ASan, abseil's hash tables change layout (generation counters) and poison free slots, so the front end
    # (which shares abseil and loki tables with the prebuilt libraries) can only be ASan-instrumented against an
    # ASan-instrumented dependency prefix (dependencies/CMakeLists.txt shows how to build one).
    if("address" IN_LIST MYMYR_SANITIZE AND NOT MYMYR_DEPS_SANITIZED)
        message(WARNING "MYMYR_SANITIZE contains address, but MYMYR_DEPS_SANITIZED is OFF: mymyr_frontend is compiled "
                        "without ASan (UBSan and the tests stay instrumented). For ASan on the front end, point "
                        "MYMYR_DEPS_PREFIX at an ASan build of the dependencies and set MYMYR_DEPS_SANITIZED=ON.")
        target_compile_options(mymyr_frontend PRIVATE -fno-sanitize=address)
    endif()
endif()

# ------------------------------------------------------------------------------------------------ CUDA backend
# libmymyr_cuda (cuda/CMakeLists.txt). Host code stays C++26 on the C++ compiler; nvcc compiles the device translation
# units as C++20 with a host compiler it supports. nvcc 13.0 rejects GCC > 15, and even with
# -allow-unsupported-compiler its front end fails on GCC 16's libstdc++ (__builtin_is_virtual_base_of in <memory>, which
# CUB pulls in), so with GCC >= 16 as the C++ compiler an older g++ is picked for nvcc unless CMAKE_CUDA_HOST_COMPILER
# is given. Pass the toolkit with -DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc (or CUDACXX); CMAKE_CUDA_ARCHITECTURES defaults
# to the visible GPU ("native").
if(MYMYR_CUDA)
    if(NOT CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX} AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        find_program(MYMYR_CUDA_HOST_CXX NAMES g++-15 g++-14 g++-13 g++-12 g++-11)
        if(NOT MYMYR_CUDA_HOST_CXX)
            message(FATAL_ERROR "MYMYR_CUDA=ON: nvcc cannot use GCC ${CMAKE_CXX_COMPILER_VERSION} as its host compiler and "
                                "no g++-15..11 was found; pass -DCMAKE_CUDA_HOST_COMPILER=<a g++ that nvcc supports>")
        endif()
        set(CMAKE_CUDA_HOST_COMPILER "${MYMYR_CUDA_HOST_CXX}" CACHE FILEPATH "Host compiler of nvcc" FORCE)
        message(STATUS "mymyr: nvcc host compiler ${CMAKE_CUDA_HOST_COMPILER} (the C++ compiler is GCC ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    # Executables are linked by the C++ compiler, which brings its own runtime. The CUDA language's implicit link
    # directories include nvcc's host compiler's (e.g. /usr/lib/gcc/x86_64-linux-gnu/12): left in, -lstdc++ would
    # resolve to that older libstdc++ and GCC 16 objects fail to link.
    if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
        list(FILTER CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES EXCLUDE REGEX "/gcc/|/gcc-[0-9]")
        list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES stdc++ m gcc_s gcc c)
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(cuda)
endif()

# ------------------------------------------------------------------------------------------------ install
if(MYMYR_INSTALL_SDK)
    include("${PROJECT_SOURCE_DIR}/cmake/MymyrSdk.cmake")
endif()

# ------------------------------------------------------------------------------------------------ tests
if(MYMYR_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ------------------------------------------------------------------------------------------------ benchmarks
if(MYMYR_BENCH)
    add_executable(mymyr_brfs bench/cpp/mymyr_brfs.cpp)
    target_link_libraries(mymyr_brfs PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_brfs PRIVATE mymyr::frontend)
    endif()
    # IW / SIW / single passes; reads PDDL when the front end is built
    add_executable(mymyr_iw bench/cpp/mymyr_iw.cpp)
    target_link_libraries(mymyr_iw PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_iw PRIVATE mymyr::frontend)
    endif()
    # heuristics and best-first search
    add_executable(mymyr_heur bench/cpp/mymyr_heur.cpp)
    target_link_libraries(mymyr_heur PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_heur PRIVATE mymyr::frontend)
    endif()
    # relaxed reachability and landmarks
    add_executable(mymyr_landmarks bench/cpp/mymyr_landmarks.cpp)
    target_link_libraries(mymyr_landmarks PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_landmarks PRIVATE mymyr::frontend)
    endif()
    # IW family variants: AIW / LIW / Rollout IW / parallel rollouts / portfolio
    add_executable(mymyr_iw_variants bench/cpp/mymyr_iw_variants.cpp)
    target_link_libraries(mymyr_iw_variants PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_iw_variants PRIVATE mymyr::frontend)
    endif()
    # CpuEnvPool env-steps/s over a task table (PDDL through the front end)
    if(TARGET mymyr_frontend)
        add_executable(mymyr_pool bench/cpp/mymyr_pool.cpp)
        target_link_libraries(mymyr_pool PRIVATE mymyr::frontend mymyr_warnings)
    endif()
    # datasets: state spaces, generalized state spaces, samplers
    add_executable(mymyr_datasets bench/cpp/mymyr_datasets.cpp)
    target_link_libraries(mymyr_datasets PRIVATE mymyr::core mymyr_warnings)
    if(TARGET mymyr_frontend)
        target_link_libraries(mymyr_datasets PRIVATE mymyr::frontend)
    endif()
endif()

# ------------------------------------------------------------------------------------------------ python
if(MYMYR_PYTHON)
    add_subdirectory(python)
endif()
