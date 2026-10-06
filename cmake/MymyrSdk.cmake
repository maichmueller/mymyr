# The C++ SDK: install rules and the `mymyr` CMake package.
#
#   <prefix>/<MYMYR_SDK_DIR>include/mymyr/...                  the public headers (cuda/ only with MYMYR_CUDA)
#   <prefix>/<MYMYR_SDK_DIR>lib/libmymyr_{core,frontend,cuda}.a
#   <prefix>/<MYMYR_SDK_DIR>lib/cmake/mymyr/mymyr{Config,ConfigVersion,Targets}.cmake
#
# The archives are self-contained: libmymyr_frontend.a includes the objects of its parser dependencies (loki, abseil,
# fmt; debug information stripped), so a consumer needs no other prefix. No public header includes a dependency's
# header. Included from the top-level CMakeLists.txt after the library targets are defined.
include(CMakePackageConfigHelpers)

set(_sdk_lib "${MYMYR_SDK_DIR}lib")
set(_sdk_cmake_dir "${_sdk_lib}/cmake/mymyr")

# The static archives of the imported targets (loki, abseil, fmt) behind the given ones, which are merged into the
# installed front end archive: the walk follows their link interfaces.
function(_mymyr_dependency_closure out_archives)
    set(todo ${ARGN})
    set(seen "")
    set(archives "")
    while(todo)
        list(POP_FRONT todo item)
        if(item MATCHES "^\\$<LINK_ONLY:(.*)>$")
            set(item "${CMAKE_MATCH_1}")
        endif()
        if(item IN_LIST seen OR item STREQUAL "Threads::Threads")
            continue()
        endif()
        list(APPEND seen "${item}")
        if(TARGET "${item}")
            get_target_property(type "${item}" TYPE)
            if(type STREQUAL "STATIC_LIBRARY")
                get_target_property(location "${item}" LOCATION)
                list(APPEND archives "${location}")
            endif()
            get_target_property(libs "${item}" INTERFACE_LINK_LIBRARIES)
            if(libs)
                list(APPEND todo ${libs})
            endif()
        endif()
    endwhile()
    set(${out_archives} "${archives}" PARENT_SCOPE)
endfunction()

set(_sdk_targets mymyr_core)
if(TARGET mymyr_frontend)
    list(APPEND _sdk_targets mymyr_frontend)
    _mymyr_dependency_closure(_frontend_archives
        loki::parsers absl::flat_hash_map absl::flat_hash_set absl::inlined_vector absl::hash)
    # abseil's time zone code uses CoreFoundation on Apple platforms
    target_link_libraries(mymyr_frontend INTERFACE
        "$<INSTALL_INTERFACE:$<$<PLATFORM_ID:Darwin>:-Wl,-framework,CoreFoundation>>")
endif()
if(TARGET mymyr_cuda)
    list(APPEND _sdk_targets mymyr_cuda)
endif()

install(TARGETS ${_sdk_targets} EXPORT mymyrTargets ARCHIVE DESTINATION "${_sdk_lib}")

set(_sdk_header_filter PATTERN "*.hpp" PATTERN "*.h")
if(NOT TARGET mymyr_cuda)
    set(_sdk_exclude PATTERN "cuda" EXCLUDE)
endif()
install(DIRECTORY include/mymyr DESTINATION "${MYMYR_SDK_DIR}include" FILES_MATCHING
    ${_sdk_exclude} ${_sdk_header_filter})

if(TARGET mymyr_frontend)
    # Merge the dependencies into the installed front end archive.
    if(APPLE)
        find_program(MYMYR_LIBTOOL NAMES libtool REQUIRED)
    endif()
    install(CODE "
        set(MYMYR_BUNDLE_OUT \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_sdk_lib}/$<TARGET_FILE_NAME:mymyr_frontend>\")
        set(MYMYR_BUNDLE_INPUTS \"${_frontend_archives}\")
        set(MYMYR_BUNDLE_SCRATCH \"${CMAKE_CURRENT_BINARY_DIR}/mymyr_bundle\")
        set(MYMYR_BUNDLE_AR \"${CMAKE_AR}\")
        set(MYMYR_BUNDLE_STRIP \"${CMAKE_STRIP}\")
        set(MYMYR_BUNDLE_LIBTOOL \"${MYMYR_LIBTOOL}\")
        include(\"${PROJECT_SOURCE_DIR}/cmake/BundleStaticLibraries.cmake\")
    ")
endif()

install(EXPORT mymyrTargets NAMESPACE mymyr:: DESTINATION "${_sdk_cmake_dir}")
configure_package_config_file(cmake/mymyrConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/mymyrConfig.cmake"
    INSTALL_DESTINATION "${_sdk_cmake_dir}")
# Before 1.0 a minor version may break the API, so a request is satisfied by the same major.minor only.
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/mymyrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mymyrConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/mymyrConfigVersion.cmake"
    DESTINATION "${_sdk_cmake_dir}")
