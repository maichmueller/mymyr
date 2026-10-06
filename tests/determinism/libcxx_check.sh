#!/bin/bash
# The libc++ side of the Mac-vs-Linux determinism check, without a Mac.
#
# 1. Builds libc++/libc++abi (LLVM 22.1.8) with clang 22 into $WORK/libcxx-22 (flat layout), once.
# 2. Builds mymyr._core with clang 22 against that libc++, with _LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY: equal
#    elements of std::sort, std::nth_element and std::partial_sort come out in a random order on every run. No loki
#    front end (its dependencies are built against libstdc++), so the PDDL tests skip.
# 3. Runs tests/python/test_determinism.py five times (its hashes were written with GCC 16 and libstdc++), then the
#    whole Python suite, with that module.
#
#   WORK=<dir for libc++> CLANG=/usr/lib/llvm-22/bin OUT=<scratch dir> tests/determinism/libcxx_check.sh
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$ROOT/.work}
CLANG=${CLANG:-/usr/lib/llvm-22/bin}
OUT=${OUT:-$ROOT/build/libcxx-check}
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
V=22.1.8
LCX=$WORK/libcxx-22
mkdir -p "$OUT"
if [ ! -f "$LCX/lib/libc++.so" ]; then
  mkdir -p "$WORK/llvm-src" && cd "$WORK/llvm-src" || exit 1
  [ -f llvm-project-$V.src.tar.xz ] || curl -sSLO https://github.com/llvm/llvm-project/releases/download/llvmorg-$V/llvm-project-$V.src.tar.xz
  [ -d llvm-project-$V.src ] || tar -xf llvm-project-$V.src.tar.xz
  cmake -S llvm-project-$V.src/runtimes -B build-libcxx -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CLANG/clang" -DCMAKE_CXX_COMPILER="$CLANG/clang++" -DCMAKE_INSTALL_PREFIX="$LCX" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
    -DLIBCXX_INCLUDE_TESTS=OFF -DLIBCXXABI_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
    -DLIBCXXABI_USE_LLVM_UNWINDER=OFF > "$OUT/libcxx-cfg.log" 2>&1 || { echo "libc++ configure failed"; exit 1; }
  ninja -C build-libcxx -j 16 install-cxx install-cxxabi > "$OUT/libcxx-build.log" 2>&1 || { echo "libc++ build failed"; exit 1; }
fi
cd "$ROOT" || exit 1
FLAGS="-nostdinc++ -isystem $LCX/include/c++/v1 -D_LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY"
LINK="-nostdlib++ -L$LCX/lib -Wl,-rpath,$LCX/lib -lc++ -lc++abi"
cmake -S . -B build/libcxx-py -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CLANG/clang" \
  -DCMAKE_CXX_COMPILER="$CLANG/clang++" -DCMAKE_CXX_FLAGS="$FLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$LINK" \
  -DCMAKE_MODULE_LINKER_FLAGS="$LINK" -DCMAKE_EXE_LINKER_FLAGS="$LINK" -DMYMYR_FRONTEND=OFF -DMYMYR_PYTHON=ON \
  -DMYMYR_TESTS=OFF -DMYMYR_BENCH=OFF -DMYMYR_PY_STUBS=OFF -DMYMYR_PY_TEST_EXT=OFF -DPython_EXECUTABLE="$PYTHON" \
  > "$OUT/mymyr-cfg.log" 2>&1 || { echo "mymyr configure failed"; exit 1; }
cmake --build build/libcxx-py -j 16 > "$OUT/mymyr-build.log" 2>&1 || { echo "mymyr build failed"; exit 1; }
echo "build: $(grep -c 'warning:' "$OUT/mymyr-build.log") warnings"
# the package: the Python sources plus the libc++ module, ahead of the regular install on the path
rm -rf "$OUT/pkg" && mkdir -p "$OUT/pkg" && cp -r python/mymyr "$OUT/pkg/mymyr" && rm -f "$OUT"/pkg/mymyr/_core*.so
find build/libcxx-py -name "_core*.so" -exec cp {} "$OUT/pkg/mymyr/" \;
SITE=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
run() { env PYTHONPATH="$OUT/pkg:$SITE" "$PYTHON" -S -m pytest -q -p no:cacheprovider -W error "$@"; }
status=0
for i in 1 2 3 4 5; do
  run tests/python/test_determinism.py > "$OUT/determinism-$i.log" 2>&1 || status=1
  echo "determinism run $i: $(tail -1 "$OUT/determinism-$i.log")"
done
run tests/python > "$OUT/suite.log" 2>&1 || status=1
echo "suite: $(tail -1 "$OUT/suite.log")"
exit $status
