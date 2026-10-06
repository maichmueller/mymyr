#!/usr/bin/env bash
# Prepare what a wheel build needs besides the Python build requirements: the compiler (Linux) and the
# dependency superbuild.
#
# usage: ci/wheel_toolchain.sh PREFIX
#
# PREFIX receives
#   gcc/    Linux only: GCC 16 with its glibc 2.34 sysroot (conda-forge), plus Ninja
#   deps/   the dependency superbuild (dependencies/CMakeLists.txt), the MYMYR_DEPS_PREFIX of the wheel build
# Running it again on a populated PREFIX does nothing, so PREFIX can be cached between builds.
#
# macOS builds use the Xcode compiler and MACOSX_DEPLOYMENT_TARGET (default 15.0) for the dependencies.
set -euo pipefail

MICROMAMBA_VERSION=2.9.0-0
MICROMAMBA_SHA256_LINUX_64=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3
GCC_SPEC="gxx_linux-64=16.1.*"

if [ $# -ne 1 ]; then
    echo "usage: $0 PREFIX" >&2
    exit 2
fi
mkdir -p "$1"
prefix=$(cd "$1" && pwd)
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
os=$(uname -s)

if [ "$os" = Linux ]; then
    # the wheel tests fetch their data with the VCS client, which the manylinux images lack
    command -v git > /dev/null || dnf install -y -q git-core
    cc=$prefix/gcc/bin/x86_64-conda-linux-gnu-gcc
    cxx=$prefix/gcc/bin/x86_64-conda-linux-gnu-g++
    if [ ! -x "$cxx" ]; then
        echo "installing GCC 16 from conda-forge into $prefix/gcc"
        mm=$prefix/micromamba
        curl -fsSL -o "$mm" \
            "https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64"
        echo "$MICROMAMBA_SHA256_LINUX_64  $mm" | sha256sum -c -
        chmod +x "$mm"
        MAMBA_ROOT_PREFIX=$prefix/mamba "$mm" create -y -q -p "$prefix/gcc" --override-channels -c conda-forge \
            "$GCC_SPEC" ninja
        rm -rf "$mm" "$prefix/mamba"
    fi
    export CC=$cc CXX=$cxx
    export PATH=$prefix/gcc/bin:$PATH
    cmake_extra=()
else
    export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-15.0}
    cmake_extra=(-DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET")
fi

# A changed pin in dependencies/CMakeLists.txt rebuilds the prefix.
pins=$repo/dependencies/CMakeLists.txt
key=$({ sha256sum "$pins" 2>/dev/null || shasum -a 256 "$pins"; } | cut -c1-16)
if [ -f "$prefix/deps/.built-$key" ]; then
    echo "dependencies already built in $prefix/deps"
    exit 0
fi
rm -rf "$prefix/deps" "$prefix/deps-build"

echo "building the dependencies with $(${CXX:-c++} --version | head -1)"
cmake -S "$repo/dependencies" -B "$prefix/deps-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMYMYR_DEPS_PREFIX="$prefix/deps" ${cmake_extra[@]+"${cmake_extra[@]}"}
cmake --build "$prefix/deps-build"
touch "$prefix/deps/.built-$key"
rm -rf "$prefix/deps-build"
echo "dependencies installed in $prefix/deps"
