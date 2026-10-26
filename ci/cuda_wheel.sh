#!/usr/bin/env bash
# Build and check a CUDA wheel: manylinux_2_34 x86_64, one CPython ABI, SASS for the given GPU architectures, version
# <version>+cu13. It runs inside the manylinux image of the CPU wheels (tool.cibuildwheel.linux in pyproject.toml):
#   1. GCC 16 and the dependency superbuild (ci/wheel_toolchain.sh) and the CUDA compiler components
#      (ci/install_cuda_redist.py with ci/cuda_redist.json) go into the toolchain prefix, which can be cached;
#   2. a copy of the source tree gets the version <version>+cu13 and is built with jaxlib's XLA FFI headers
#      (ci/cuda_wheel_jaxlib_requirements.txt), so that the wheel has the JAX targets of mymyr.rl.jax and
#      mymyr.encoding.jax, and with the pinned build requirements (ci/cuda_wheel_build_requirements.txt) without build
#      isolation;
#   3. auditwheel repairs the wheel; the checks: the manylinux_2_34_x86_64 tag, extensions that need only glibc (the
#      CUDA runtime is linked statically and opens the driver, libcuda.so.1, at run time) and GPU code that is exactly
#      the requested architectures;
#   4. the wheel is installed into a fresh environment with the pinned test requirements
#      (ci/cuda_wheel_test_requirements.txt, torch 2.14 for CUDA 13.0 among them) and passes the CPU test suite with no
#      GPU visible, in a process that imports torch before mymyr.
#
# usage: ci/cuda_wheel.sh [--python ABI] [--archs LIST] [--out DIR] [--toolchain DIR] [--work DIR] [--jobs N]
#                         [--cuda-jobs N] [--stage STAGE] [--minutes N]
#   --python     CPython ABI from /opt/python: cp314t (default), cp315t, cp314 or cp315
#   --archs      CMAKE_CUDA_ARCHITECTURES (default "86-real;89-real;90-real"); N-real is SASS, N-virtual PTX, N both
#   --out        receives the wheel (default: wheelhouse/ in the source tree)
#   --toolchain  GCC, the dependencies and CUDA (default /opt/mymyr-toolchain)
#   --work       the source copy, build tree, test environment, caches and temporary files
#                (default /tmp/mymyr-cuda-wheel); about 7 GB per architecture
#   --jobs       parallel build jobs (default: the number of CPUs)
#   --cuda-jobs  parallel nvcc compiles (default 2), each up to 3 GB of memory; the four largest kernels (up to 7 GB)
#                compile one at a time, which can make one compile more than N
#   --stage      all (default): steps 1 to 4;
#                toolchain: step 1 only;
#                compile: steps 1 and 2, stopping the build cleanly --minutes after the start of the script if it has
#                not finished by then;
#                finish: the build in --work that a compile stage left (with the same --toolchain, unchanged, and
#                the same paths), to its end, then steps 3 and 4. Ninja resumes it: what compiled is not compiled
#                again
#   --minutes    the time limit of the compile stage (default 270)
#
# In a container, with the source tree at /src (see "Building a CUDA wheel" in docs/cuda.md):
#   podman run --rm -v "$PWD":/src -v ~/.cache/mymyr-cuda-toolchain:/opt/mymyr-toolchain \
#       -v /scratch/mymyr-cuda-wheel:/work <image> bash /src/ci/cuda_wheel.sh --work /work
set -euo pipefail

python_abi=cp314t
archs="86-real;89-real;90-real"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$repo/wheelhouse
toolchain=/opt/mymyr-toolchain
work=/tmp/mymyr-cuda-wheel
jobs=$(nproc)
cuda_jobs=2
stage=all
minutes=270

usage() {
    sed -n '/^# usage:/,/^# In a container/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --python) python_abi=$2 ;;
        --archs) archs=$2 ;;
        --out) out=$2 ;;
        --toolchain) toolchain=$2 ;;
        --work) work=$2 ;;
        --jobs) jobs=$2 ;;
        --cuda-jobs) cuda_jobs=$2 ;;
        --stage) stage=$2 ;;
        --minutes) minutes=$2 ;;
        *) usage ;;
    esac
    [ $# -ge 2 ] || usage
    shift 2
done
case "$python_abi" in
    cp3[0-9][0-9]t) python_dir=${python_abi%t}-$python_abi ;;
    cp3[0-9][0-9]) python_dir=$python_abi-$python_abi ;;
    *) echo "error: unknown CPython ABI '$python_abi'" >&2; exit 2 ;;
esac
case "$stage" in
    all | toolchain | compile | finish) ;;
    *) echo "error: unknown stage '$stage'" >&2; exit 2 ;;
esac
python=/opt/python/$python_dir/bin/python
if [ ! -x "$python" ]; then
    echo "error: no $python: run this in the manylinux_2_34 image named in pyproject.toml" >&2
    exit 1
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ && "$cuda_jobs" =~ ^[1-9][0-9]*$ && "$minutes" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --jobs, --cuda-jobs and --minutes must be positive" >&2
    exit 2
}

# the SASS and PTX that --archs asks for, named as cuobjdump lists them (sm_N), sorted and space-separated
sass=() ptx=()
IFS=';' read -r -a arch_list <<< "$archs"
for a in "${arch_list[@]}"; do
    case "$a" in
        *-real) sass+=("sm_${a%-real}") ;;
        *-virtual) ptx+=("sm_${a%-virtual}") ;;
        [0-9]*) sass+=("sm_$a"); ptx+=("sm_$a") ;;
        *) echo "error: --archs takes N, N-real or N-virtual entries, not '$a'" >&2; exit 2 ;;
    esac
done
[ ${#sass[@]} -gt 0 ] || { echo "error: --archs names no real architecture" >&2; exit 2; }
want_sass=$(printf '%s\n' "${sass[@]}" | sort -u | tr '\n' ' ')
want_ptx=$(printf '%s\n' ${ptx[@]+"${ptx[@]}"} | sed '/^$/d' | sort -u | tr '\n' ' ')

mkdir -p "$out" "$work"
out=$(cd "$out" && pwd)
work=$(cd "$work" && pwd)
# everything the build writes stays under --work: the container's own file system may be small
export TMPDIR=$work/tmp PIP_CACHE_DIR=$work/cache/pip XDG_CACHE_HOME=$work/cache
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR"
start=$SECONDS
step() { printf '\n== %s (%d min)\n' "$1" $(((SECONDS - start) / 60)); }

src=$work/src
build_venv=$work/venv-build
xla_include=$work/jaxlib/jaxlib/include
if [ "$stage" = finish ]; then
    # the source copy keeps the modification times the build tree was made with; a fresh copy would rebuild it all
    for path in "$src/pyproject.toml" "$build_venv/bin/python" "$xla_include/xla/ffi/api/ffi.h" "$work/build"; do
        [ -e "$path" ] || { echo "error: no $path: run the compile stage with this --work first" >&2; exit 1; }
    done
fi

step "toolchain in $toolchain"
bash "$repo/ci/wheel_toolchain.sh" "$toolchain"
"$python" -I "$repo/ci/install_cuda_redist.py" "$repo/ci/cuda_redist.json" "$toolchain/cuda"
[ "$stage" != toolchain ] || { step "done: toolchain in $toolchain"; exit 0; }
export CC=$toolchain/gcc/bin/x86_64-conda-linux-gnu-gcc CXX=$toolchain/gcc/bin/x86_64-conda-linux-gnu-g++
export PATH=$toolchain/gcc/bin:$toolchain/cuda/bin:$PATH
readelf=$toolchain/gcc/bin/x86_64-conda-linux-gnu-readelf
nvcc=$toolchain/cuda/bin/nvcc
"$nvcc" --version | tail -2

if [ "$stage" != finish ]; then
    step "source copy with the CUDA wheel version"
    rm -rf "$src" "$work/build" "$work/dist" "$work/repaired" "$work/jaxlib" "$build_venv"
    mkdir -p "$src"
    tar -C "$repo" -cf - --exclude=./.git --exclude=./build --exclude='./.venv*' --exclude=./.claude \
        --exclude=./.work --exclude=./wheelhouse --exclude=./dist --exclude='./.ci-*' --exclude=__pycache__ . |
        tar -C "$src" -xf -
    "$python" -I "$src/ci/set_rehearsal_version.py" --local cu13

    step "jaxlib for the XLA FFI headers, the build requirements"
    "$python" -m pip install --quiet --no-deps --require-hashes --only-binary=:all: --target "$work/jaxlib" \
        -r "$src/ci/cuda_wheel_jaxlib_requirements.txt"
    [ -f "$xla_include/xla/ffi/api/ffi.h" ] || { echo "error: no XLA FFI headers in $xla_include" >&2; exit 1; }
    "$python" -m venv "$build_venv"
    "$build_venv/bin/python" -m pip install --quiet --require-hashes --only-binary=:all: \
        -r "$src/ci/cuda_wheel_build_requirements.txt"
fi
version=$("$python" -I -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' \
    "$src/pyproject.toml")

step "build mymyr $version for $python_abi, CUDA architectures $archs"
rm -rf "$work/dist"
export CMAKE_BUILD_PARALLEL_LEVEL=$jobs
# the compile stage stops Ninja cleanly at the time limit (ci/run_until.py), and the finish stage resumes it
until=()
if [ "$stage" = compile ]; then
    until=("$python" -I "$repo/ci/run_until.py" "$(($(date +%s) + minutes * 60 - (SECONDS - start)))" --)
fi
status=0
${until[@]+"${until[@]}"} "$build_venv/bin/python" -m pip wheel -v --no-deps --no-build-isolation -w "$work/dist" \
    -C build-dir="$work/build/{wheel_tag}" \
    -C cmake.version=">=4.4" \
    -C cmake.define.MYMYR_DEPS_PREFIX="$toolchain/deps" \
    -C cmake.define.MYMYR_CUDA=ON \
    -C cmake.define.CMAKE_CUDA_COMPILER="$nvcc" \
    -C cmake.define.CMAKE_CUDA_ARCHITECTURES="$archs" \
    -C cmake.define.CMAKE_JOB_POOLS="cuda_compile=$cuda_jobs;cuda_compile_heavy=1" \
    -C cmake.define.MYMYR_XLA_FFI_INCLUDE="$xla_include" \
    "$src" || status=$?
if [ "$stage" = compile ] && [ $status -eq 75 ]; then
    step "done: build stopped at the time limit of $minutes min, resume it with --stage finish"
    exit 0
fi
[ $status -eq 0 ] || { echo "error: the build failed" >&2; exit $status; }
if [ "$stage" = compile ]; then
    step "done: build finished, package and test it with --stage finish"
    exit 0
fi
raw=("$work"/dist/mymyr-*.whl)
[ ${#raw[@]} -eq 1 ] || { echo "error: expected one wheel in $work/dist" >&2; exit 1; }

step "repair and check the wheel"
rm -rf "$work/repaired"
auditwheel repair --plat manylinux_2_34_x86_64 --exclude libcuda.so.1 --exclude libcudart.so.13 \
    -w "$work/repaired" "${raw[0]}"
repaired=("$work"/repaired/mymyr-"$version"-*.whl)
[ ${#repaired[@]} -eq 1 ] || { echo "error: expected one repaired wheel in $work/repaired" >&2; exit 1; }
wheel=${repaired[0]}
auditwheel show "$wheel"
case "$(basename "$wheel")" in
    mymyr-"$version"-*-"$python_abi"-manylinux_2_34_x86_64.whl) ;;
    *) echo "error: unexpected wheel name $(basename "$wheel")" >&2; exit 1 ;;
esac
check=$work/check
rm -rf "$check"
"$python" -I -m zipfile -e "$wheel" "$check"
status=0
while IFS= read -r -d '' so; do
    needed=$("$readelf" -d "$so" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | sort | tr '\n' ' ')
    echo "${so#"$check"/}: NEEDED $needed"
    for lib in $needed; do
        case "$lib" in
            libc.so.6 | libm.so.6 | ld-linux-x86-64.so.2 | libpthread.so.0 | libdl.so.2 | librt.so.1) ;;
            *) echo "error: ${so#"$check"/} needs $lib" >&2; status=1 ;;
        esac
    done
done < <(find "$check" -name '*.so' -print0)
core=("$check"/mymyr/_core*.so)
have_sass=$(cuobjdump --list-elf "${core[0]}" | grep -o 'sm_[0-9]*[a-z]*\.cubin' | sed 's/\.cubin$//' | sort -u | tr '\n' ' ')
have_ptx=$(cuobjdump --list-ptx "${core[0]}" | grep -o 'sm_[0-9]*[a-z]*\.ptx' | sed 's/\.ptx$//' | sort -u | tr '\n' ' ' || true)
echo "SASS: $have_sass PTX: ${have_ptx:-none}"
if [ "$have_sass" != "$want_sass" ] || [ "$have_ptx" != "$want_ptx" ]; then
    echo "error: the extension's GPU code is SASS '$have_sass' and PTX '$have_ptx'," \
         "expected SASS '$want_sass' and PTX '$want_ptx'" >&2
    status=1
fi
[ $status -eq 0 ] || exit 1
rm -rf "$check"
ls -l "$wheel"

step "test the wheel with the CPU suite, no GPU visible, torch imported first"
venv=$work/venv-test
rm -rf "$venv" "$work/consumer-"*
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --require-hashes --no-deps --only-binary=:all: \
    -r "$src/ci/cuda_wheel_test_requirements.txt"
"$venv/bin/python" -m pip install --quiet --force-reinstall --no-deps "$wheel"
bash "$src/tests/parity/fetch_data.sh" "$work/test-data"
mkdir -p "$work/test-run"
(
    cd "$work/test-run"
    export MYMYR_PARITY_DATA=$work/test-data CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
    "$venv/bin/python" "$src/ci/check_cuda_install.py" --version "$version" --archs "${want_sass//sm_/}" \
        --wheel "$wheel" --torch 2.14 --task "$src/tests/data/tasks/blocks__probBLOCKS-8-0.txt"
    "$venv/bin/python" "$src/ci/run_pytest.py" --runtimes none --import-first torch -W error -q -rs \
        -p no:cacheprovider "$src/tests/python" "$src/tests/parity/python"
    PATH=$venv/bin:$PATH bash "$src/ci/check_cmake_consumer.sh" \
        "$("$venv/bin/python" -c 'import mymyr; print(mymyr.get_cmake_dir())')" "$work/consumer"
)

cp "$wheel" "$out/"
step "done: $out/$(basename "$wheel")"
