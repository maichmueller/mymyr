# mymyr dependency superbuild: pinned ExternalProjects installed into
# dependencies/installs, which the main build finds through CMAKE_PREFIX_PATH.
#
#   cmake -S dependencies -B dependencies/build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build dependencies/build
#
# The pins are the mimir fork's (its dependencies/ directory), with one deliberate difference: loki is
# built WITHOUT the fork's IndexedHashSet multi-level-chaining patch. mymyr chains at most one level
# (domain -> problem), where the patched and stock loki behave identically (same lookup result, indices
# continue the parent's space).
#
# A package that is already installed in the prefix is skipped, so re-running is cheap.
#
# The ExternalProjects do not forward the compiler or flags: set CC/CXX (and CFLAGS/CXXFLAGS/LDFLAGS) in the
# environment of both the configure and the build command, e.g. to pin a specific GCC, and for an ASan build of the
# dependencies (needed for ASan on mymyr's front end, see MYMYR_DEPS_SANITIZED in the top-level CMakeLists.txt):
#   env CC=.../gcc CXX=.../g++ CXXFLAGS="-fsanitize=address -fno-omit-frame-pointer" \
#       CFLAGS="-fsanitize=address -fno-omit-frame-pointer" LDFLAGS=-fsanitize=address \
#       cmake -S dependencies -B dependencies/build-asan -G Ninja -DMYMYR_DEPS_PREFIX=$PWD/dependencies/build-asan/installs
#   (same env) cmake --build dependencies/build-asan
cmake_minimum_required(VERSION 3.24)
project(mymyr_dependencies C CXX)  # C/CXX: installed package configs (absl -> Threads) need a language

include(ExternalProject)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type of the dependencies" FORCE)
endif()
set(MYMYR_DEPS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/installs" CACHE PATH "Install prefix of the dependencies")
set(CMAKE_INSTALL_PREFIX "${MYMYR_DEPS_PREFIX}" CACHE PATH "" FORCE)

# Archives of the pinned commits/tags (equivalent to the fork's git checkouts, but faster to fetch).
set(MYMYR_LOKI_COMMIT 02880ea1d4ce4ca11deca188050ea6cc83f88dbb)      # https://github.com/drexlerd/Loki.git
set(MYMYR_ABSEIL_COMMIT 5913446e0c0f8c1976627654956c3553e1519d2c)    # https://github.com/abseil/abseil-cpp.git
set(MYMYR_FMT_TAG 11.2.0)                                            # https://github.com/fmtlib/fmt
set(MYMYR_GTEST_TAG release-1.12.1)                                  # https://github.com/google/googletest
set(MYMYR_BOOST_VERSION 1.84.0)

set(_common_args
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${MYMYR_DEPS_PREFIX}
    -DCMAKE_PREFIX_PATH=${MYMYR_DEPS_PREFIX}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_STANDARD=20)
if(CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _common_args -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

set(_deps "")

# ------------------------------------------------------------------------------------------------ boost (headers)
# loki uses boost spirit x3 / hana / fusion headers only. We still run b2 because it installs the CMake package
# configuration (BoostConfig.cmake, boost_headers) that loki's find_package(Boost) needs; building the header-only
# boost_system stub keeps it to a header copy.
find_package(Boost ${MYMYR_BOOST_VERSION} EXACT QUIET CONFIG PATHS ${MYMYR_DEPS_PREFIX} NO_DEFAULT_PATH)
if(NOT Boost_FOUND)
    string(REPLACE "." "_" _boost_us ${MYMYR_BOOST_VERSION})
    ExternalProject_Add(boost
        URL https://archives.boost.io/release/${MYMYR_BOOST_VERSION}/source/boost_${_boost_us}.tar.gz
        URL_HASH SHA256=a5800f405508f5df8114558ca9855d2640a2de8f0445f051fa1c7c3383045724
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/boost
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ./bootstrap.sh --prefix=${MYMYR_DEPS_PREFIX} --with-libraries=system
        BUILD_COMMAND ""
        INSTALL_COMMAND ./b2 install -j8 -d0 link=static variant=release threading=multi cxxflags=-fPIC
                        --prefix=${MYMYR_DEPS_PREFIX})
    list(APPEND _deps boost)
else()
    message(STATUS "boost ${Boost_VERSION} already installed")
    add_custom_target(boost)
endif()

# ------------------------------------------------------------------------------------------------ abseil
find_package(absl QUIET CONFIG PATHS ${MYMYR_DEPS_PREFIX} NO_DEFAULT_PATH)
if(NOT absl_FOUND)
    ExternalProject_Add(abseil
        URL https://github.com/abseil/abseil-cpp/archive/${MYMYR_ABSEIL_COMMIT}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/abseil
        CMAKE_ARGS ${_common_args} -DABSL_PROPAGATE_CXX_STD=ON -DABSL_BUILD_TESTING=OFF -DBUILD_TESTING=OFF)
    list(APPEND _deps abseil)
else()
    message(STATUS "abseil already installed")
    add_custom_target(abseil)
endif()

# ------------------------------------------------------------------------------------------------ fmt
find_package(fmt QUIET CONFIG PATHS ${MYMYR_DEPS_PREFIX} NO_DEFAULT_PATH)
if(NOT fmt_FOUND)
    ExternalProject_Add(fmt
        URL https://github.com/fmtlib/fmt/archive/refs/tags/${MYMYR_FMT_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/fmt
        CMAKE_ARGS ${_common_args} -DFMT_TEST=OFF -DFMT_DOC=OFF)
    list(APPEND _deps fmt)
else()
    message(STATUS "fmt already installed")
    add_custom_target(fmt)
endif()

# ------------------------------------------------------------------------------------------------ googletest
find_package(GTest QUIET CONFIG PATHS ${MYMYR_DEPS_PREFIX} NO_DEFAULT_PATH)
if(NOT GTest_FOUND)
    ExternalProject_Add(googletest
        URL https://github.com/google/googletest/archive/refs/tags/${MYMYR_GTEST_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/googletest
        CMAKE_ARGS ${_common_args})
    list(APPEND _deps googletest)
else()
    message(STATUS "googletest already installed")
endif()

# ------------------------------------------------------------------------------------------------ loki (unpatched)
find_package(loki QUIET CONFIG PATHS ${MYMYR_DEPS_PREFIX} NO_DEFAULT_PATH)
if(NOT loki_FOUND)
    ExternalProject_Add(loki
        URL https://github.com/drexlerd/Loki/archive/${MYMYR_LOKI_COMMIT}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX ${CMAKE_BINARY_DIR}/loki
        DEPENDS boost abseil fmt
        CMAKE_ARGS ${_common_args} -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
        # Only the library: loki's CLI (exe/loki.cpp) needs argparse, which mymyr does not use.
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target parsers
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>)
    list(APPEND _deps loki)
else()
    message(STATUS "loki already installed")
endif()

message(STATUS "mymyr dependencies -> ${MYMYR_DEPS_PREFIX} (to build: ${_deps})")
