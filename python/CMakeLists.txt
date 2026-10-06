find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind 3.1 CONFIG REQUIRED)

# Free-threaded CPython is the only target. No shared NB_DOMAIN: downstream native modules use the versioned
# capsule C API instead.
nanobind_add_module(_core FREE_THREADED NB_STATIC
    src/module.cpp
    src/formalism_bindings.cpp
    src/ext_api.cpp
    src/arrays.cpp
    src/task_bindings.cpp
    src/rl_bindings.cpp
    src/rl_torch_bindings.cpp
    src/rl_jax_bindings.cpp
    src/rl_ops_bindings.cpp
    src/search_bindings.cpp
    src/landmarks_bindings.cpp
    src/datasets_bindings.cpp
)
target_link_libraries(_core PRIVATE mymyr::core mymyr_warnings)
# mymyr.Domain, when the loki front end is built (MYMYR_FRONTEND, the default).
if(TARGET mymyr_frontend)
    target_sources(_core PRIVATE src/frontend_bindings.cpp)
    target_link_libraries(_core PRIVATE mymyr::frontend)
endif()
# mymyr.cuda, when the CUDA backend is built (MYMYR_CUDA=ON; mymyr_cuda defines MYMYR_HAS_CUDA publicly).
if(TARGET mymyr_cuda)
    target_sources(_core PRIVATE src/cuda_bindings.cpp src/cuda_search_bindings.cpp src/cuda_heuristics_bindings.cpp)
    target_sources(_core PRIVATE src/cuda_datasets_bindings.cpp)
    target_link_libraries(_core PRIVATE mymyr::cuda)
endif()
# The XLA FFI targets of mymyr.rl.jax need the XLA FFI headers, which ship with jaxlib: found through the build's
# interpreter (no import of jax), or MYMYR_XLA_FFI_INCLUDE. Without them mymyr._core._rl_jax has no targets.
set(MYMYR_XLA_FFI_INCLUDE "" CACHE PATH "The XLA FFI headers (jaxlib's include directory); empty: found through jaxlib")
if(NOT MYMYR_XLA_FFI_INCLUDE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import importlib.util, os; s = importlib.util.find_spec('jaxlib'); print(os.path.join(os.path.dirname(s.origin), 'include') if s and s.origin else '')"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE _mymyr_xla_include ERROR_QUIET)
else()
    set(_mymyr_xla_include "${MYMYR_XLA_FFI_INCLUDE}")
endif()
if(_mymyr_xla_include AND EXISTS "${_mymyr_xla_include}/xla/ffi/api/ffi.h")
    message(STATUS "mymyr: XLA FFI headers in ${_mymyr_xla_include}")
    target_include_directories(_core SYSTEM PRIVATE "${_mymyr_xla_include}")
    target_compile_definitions(_core PRIVATE MYMYR_HAS_XLA_FFI=1)
else()
    message(STATUS "mymyr: no XLA FFI headers (jaxlib not found): mymyr.rl.jax has no FFI targets")
endif()
target_compile_definitions(_core PRIVATE MYMYR_VERSION_STRING="${MYMYR_VERSION}")
# Python.h goes before every other header: CPython 3.15's pyconfig.h sets feature-test macros (_POSIX_C_SOURCE,
# _XOPEN_SOURCE) that libstdc++'s headers define themselves, which warns when the standard headers come first.
target_compile_options(_core PRIVATE -include Python.h)
# The interpreter may already have loaded an older libstdc++ (conda ships GCC 15's). The capsule API is plain C, so
# no C++ object crosses the module boundary and a private static copy is safe (the usual wheel practice). Its symbols
# must stay hidden (--exclude-libs), or they interpose with the loaded libstdc++ and mix two versions' locale state.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE AND NOT MYMYR_SANITIZE)
    target_link_options(_core PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
endif()

# nanobind links macOS modules with a list of the CPython symbols that may stay undefined until load time; the list of
# nanobind 3.1 lacks this CPython 3.14 API, which the module calls (module.cpp) to enable deferred reference counting.
if(APPLE)
    target_link_options(_core PRIVATE LINKER:-U,_PyUnstable_Object_EnableDeferredRefcount)
endif()

install(TARGETS _core LIBRARY DESTINATION mymyr)

# Typing stubs: nanobind's stubgen on the freshly built module, recursive and with
# the private submodules (_formalism, _rl, _search, ...), installed as the stub package mymyr/_core/*.pyi next to the
# extension (py.typed ships with the Python sources). stubgen imports the module as mymyr._core from a staging
# directory that holds only the extension and mymyr/_typing.py (a namespace package), under `python -S`: an editable
# install's import hook would otherwise hand it the installed module. _typing.py holds the structural types the
# signatures name (typing.hpp); stubgen imports what they name, so without it the stubs would `import mymyr` only.
# Sanitizer builds skip it: their modules need the runtime preloaded.
option(MYMYR_PY_STUBS "Generate and install typing stubs for mymyr._core" ON)
if(MYMYR_PY_STUBS AND NOT MYMYR_SANITIZE)
    if(EXISTS ${NB_DIR}/stubgen.py)
        set(stubgen ${NB_DIR}/stubgen.py)
    else()
        set(stubgen ${NB_DIR}/src/stubgen.py)
    endif()
    set(stage ${CMAKE_CURRENT_BINARY_DIR}/stubgen)
    set(patterns ${CMAKE_CURRENT_SOURCE_DIR}/stubgen-patterns.txt)
    add_custom_command(
        OUTPUT ${stage}/mymyr/_core/__init__.pyi
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${stage}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${stage}/mymyr
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_core> ${CMAKE_CURRENT_SOURCE_DIR}/mymyr/_typing.py
                ${stage}/mymyr/
        COMMAND ${Python_EXECUTABLE} -S ${stubgen} -q -r -P -p ${patterns} -i ${stage} -m mymyr._core -O ${stage}/mymyr
        DEPENDS _core ${stubgen} ${patterns} ${CMAKE_CURRENT_SOURCE_DIR}/mymyr/_typing.py
        COMMENT "Generating typing stubs for mymyr._core"
        VERBATIM)
    add_custom_target(_core_stubs ALL DEPENDS ${stage}/mymyr/_core/__init__.pyi)
    install(DIRECTORY ${stage}/mymyr/_core DESTINATION mymyr FILES_MATCHING PATTERN "*.pyi")
endif()

# A downstream consumer of the C API (mymyr/ext.h), built exactly as a third-party module would be: C only, Python
# headers plus ext.h, no mymyr library, no nanobind. Shipped as mymyr._testing._ext_consumer for the ABI test.
option(MYMYR_PY_TEST_EXT "Build the C-API consumer test module" ON)
if(MYMYR_PY_TEST_EXT)
    enable_language(C)
    Python_add_library(_ext_consumer MODULE WITH_SOABI ${PROJECT_SOURCE_DIR}/tests/ext_consumer/ext_consumer.c)
    target_include_directories(_ext_consumer PRIVATE ${PROJECT_SOURCE_DIR}/include)
    install(TARGETS _ext_consumer LIBRARY DESTINATION mymyr/_testing)
endif()
