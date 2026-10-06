#!/usr/bin/env bash
# Builds and runs tests/cmake_consumer against an installed mymyr, once with the front end and once with the core only.
#
# usage: ci/check_cmake_consumer.sh MYMYR_CMAKE_DIR BUILD_DIR
#   MYMYR_CMAKE_DIR  the directory of mymyrConfig.cmake: <prefix>/lib/cmake/mymyr of `cmake --install`, or
#                    $(python -c "import mymyr; print(mymyr.get_cmake_dir())") of an installed wheel
#   BUILD_DIR        receives BUILD_DIR-frontend and BUILD_DIR-core
# The C++ compiler is the environment's (CXX), which must provide C++26 and the standard library the install was built with.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 MYMYR_CMAKE_DIR BUILD_DIR" >&2
    exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mymyr_dir=$(cd "$1" && pwd)
build=$2

ls "$mymyr_dir"/mymyrConfig.cmake > /dev/null

echo "== consumer with mymyr::core and mymyr::frontend"
cmake -S "$repo/tests/cmake_consumer" -B "$build-frontend" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -Dmymyr_DIR="$mymyr_dir" -DMYMYR_CONSUMER_FRONTEND=ON
cmake --build "$build-frontend"
"$build-frontend/consumer" "$repo/tests/data/pddl/logistics00/domain.pddl" "$repo/tests/cmake_consumer/data/problem.pddl"

echo "== consumer with mymyr::core"
cmake -S "$repo/tests/cmake_consumer" -B "$build-core" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -Dmymyr_DIR="$mymyr_dir" -DMYMYR_CONSUMER_FRONTEND=OFF
cmake --build "$build-core"
"$build-core/consumer" "$repo/tests/cmake_consumer/data/task.txt"
