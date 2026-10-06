#!/usr/bin/env bash
# Out-of-source build and install of the mimir fork (maichmueller/mimir, tag v0.16.3, commit 8033459), a
# prerequisite for build.sh / export_golden (which link the installed library) and for export_all.py (which runs
# the fork's own executables indirectly through export_golden).
#
# Every path comes from an environment variable; none has a host-specific default:
#   WORK      work directory for clones and build trees (MYMYR_WORK, default <repo>/.work)
#   MIMIR     a clone of the fork, tag v0.16.3 / commit 8033459 (default $WORK/mimir-fork)
#   DEPS      the fork's dependency superbuild install prefix (default $WORK/install-fork-deps; built here unless
#             it already holds loki)
#   INSTALL   where the fork's library is installed (default $WORK/install-fork)
#   CC, CXX   the fork's compiler (CMake's defaults if unset; the fork builds against C++20)
#   FORK_EXECUTABLES  ON builds the fork's own exe/planner_*.cpp too (off by default: they stream
#             std::chrono durations with operator<<, which needs a C++20 standard library new enough for it)
#   JOBS      build parallelism (default 12); DEPS_JOBS for the dependency superbuild (default 6, since
#             boost/spirit translation units are memory-heavy); LTO_JOBS caps GCC's -flto=auto link partitions
#             (default 6)
#   STEPS     subset of "deps core exporters" to run (default: all); "lifted-exporter" and "numeric-exporter"
#             build one exporter each
#
# Build dirs: $WORK/build-fork-deps, $WORK/build-fork, $WORK/build-exporter (export_lifted_fork),
# $WORK/build-numeric-exporter (export_numeric_fork), used by make_golden.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../../.." && pwd -P)"

: "${WORK:=${MYMYR_WORK:-$REPO/.work}}"
: "${MIMIR:=$WORK/mimir-fork}"
: "${DEPS:=$WORK/install-fork-deps}"
: "${FORK_EXECUTABLES:=OFF}"
: "${INSTALL:=$WORK/install-fork}"
: "${JOBS:=12}"
: "${DEPS_JOBS:=6}"
: "${LTO_JOBS:=6}"
: "${STEPS:=deps core exporters}"
GEN=(-G Ninja)

COMPILERS=()
CXX_COMPILER=()  # the exporters are C++-only projects
if [[ -n "${CXX:-}" ]]; then
  COMPILERS=(-DCMAKE_C_COMPILER="${CC:-}" -DCMAKE_CXX_COMPILER="$CXX")
  CXX_COMPILER=(-DCMAKE_CXX_COMPILER="$CXX")
fi
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

echo "fork source: $MIMIR"
git -C "$MIMIR" rev-parse HEAD 2>/dev/null || cat "$MIMIR/.git/HEAD"
echo "compiler: $("${CXX:-c++}" --version | head -1)"

# 0) dependencies (the fork's own superbuild), only when DEPS does not hold them yet. Unix Makefiles share one
#    make jobserver, so -j DEPS_JOBS bounds all ExternalProject builds together; boost's b2 reads PARALLELISM.
#    nanobind is skipped (the fork's switch for pure C++ consumers): it needs Python >= 3.12 headers and only
#    serves pymimir.
if has_step deps && [[ ! -d "$DEPS/lib/cmake/loki" ]]; then
  cmake -S "$MIMIR/dependencies" -B "$WORK/build-fork-deps" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DDEPENDENCIES_BUILD_NANOBIND=OFF ${COMPILERS[@]+"${COMPILERS[@]}"}
  if [[ ! -d "$DEPS/lib/cmake/Boost-1.84.0" ]]; then  # boost alone first: it is the biggest build
    PARALLELISM=$DEPS_JOBS cmake --build "$WORK/build-fork-deps" --target boost -j "$DEPS_JOBS"
  fi
  PARALLELISM=$DEPS_JOBS cmake --build "$WORK/build-fork-deps" -j "$DEPS_JOBS"
fi

# 1) fork core (+ exe planners when FORK_EXECUTABLES=ON). Release => the fork's CMakeLists forces
#    "-O3 -Wall -DNDEBUG", and adds "-flto -ftime-trace -fPIC" (AppleClang) or "-flto=auto -fcoroutines -fPIC"
#    (GCC). Under GCC the link flags override -flto=auto (the last -flto wins) and a job pool runs at most two
#    links at a time. The installed library keeps an RPATH to shared dependencies in DEPS, if any
#    (CMAKE_INSTALL_RPATH_USE_LINK_PATH).
if has_step core; then
  EXTRA=()
  if "${CXX:-c++}" --version | grep -q 'Free Software Foundation'; then
    EXTRA=(-DCMAKE_SHARED_LINKER_FLAGS="-flto=$LTO_JOBS" -DCMAKE_EXE_LINKER_FLAGS="-flto=$LTO_JOBS"
           -DCMAKE_JOB_POOLS="link=2" -DCMAKE_JOB_POOL_LINK=link)
  fi
  cmake -S "$MIMIR" -B "$WORK/build-fork" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEPS" \
        -DBUILD_EXECUTABLES="$FORK_EXECUTABLES" -DMIMIR_VERSION_INFO=0.16.3 -DCMAKE_INSTALL_PREFIX="$INSTALL" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON ${COMPILERS[@]+"${COMPILERS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
  cmake --build "$WORK/build-fork" -j "$JOBS"
  cmake --install "$WORK/build-fork"
fi

# 2) the fork text exporters (link libmimir_core from the install; BUILD_RPATH to the install)
build_tool() {  # <source dir> <build dir>
  cmake -S "$1" -B "$2" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$INSTALL;$DEPS" \
        ${CXX_COMPILER[@]+"${CXX_COMPILER[@]}"}
  cmake --build "$2" -j "$JOBS"
}
if has_step exporters || has_step lifted-exporter; then
  build_tool "$HERE/export_lifted_fork" "$WORK/build-exporter"
fi
if has_step exporters || has_step numeric-exporter; then
  build_tool "$HERE/export_numeric_fork" "$WORK/build-numeric-exporter"
fi
