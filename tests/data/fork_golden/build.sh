#!/usr/bin/env bash
# Builds export_golden against the fork install made by build_fork.sh (same WORK, INSTALL, DEPS, JOBS variables).
# Binary: $WORK/build-fork-golden/export_golden.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../../.." && pwd -P)"
: "${WORK:=${MYMYR_WORK:-$REPO/.work}}"
: "${DEPS:=$WORK/install-fork-deps}"
: "${FORK_CXX:=}"
: "${INSTALL:=$WORK/install-fork}"
: "${JOBS:=12}"
COMPILERS=()
if [[ -n "$FORK_CXX" ]]; then
  export CXX="$FORK_CXX"
  COMPILERS=(-DCMAKE_CXX_COMPILER="$FORK_CXX")
fi
cmake -S "$HERE" -B "$WORK/build-fork-golden" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$INSTALL;$DEPS" ${COMPILERS[@]+"${COMPILERS[@]}"}
cmake --build "$WORK/build-fork-golden" -j "$JOBS"
