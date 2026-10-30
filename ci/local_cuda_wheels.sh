#!/usr/bin/env bash
# Build the CUDA wheels of several CPython ABIs on one Linux host: ci/cuda_wheel.sh for each ABI, in the manylinux
# image of the release wheels (manylinux-x86_64-image under [tool.cibuildwheel.linux] in pyproject.toml), with
# podman, docker or apptainer. The wheels are those of the CUDA wheel workflow (version <version>+cu13, repaired by
# auditwheel, checked and passed through the CPU test suite) and go to ci/attach_cuda_wheels.sh, which tests them on a
# GPU and attaches them to the GitHub Release.
#
# usage: ci/local_cuda_wheels.sh --toolchain DIR --work DIR [--out DIR] [--python LIST] [--archs LIST] [--cpus N]
#                                [--memory-gb N] [--parallel N] [--jobs N] [--cuda-jobs N] [--engine ENGINE]
#                                [--image IMAGE] [--dry-run]
#   --toolchain  GCC 16, the dependencies and the CUDA compiler (about 1.7 GB): the first run installs them, later
#                runs reuse them
#   --work       one work directory per ABI, <work>/<abi> (about 1.2 GB per architecture while it builds, then
#                4.5 GB for the test environment), and the logs <work>/<abi>.log and <work>/toolchain.log
#   --out        receives the wheels (default: wheelhouse/ in the checkout)
#   --python     CPython ABIs, space-separated (default "cp314t cp315t")
#   --archs      CMAKE_CUDA_ARCHITECTURES (default "86-real;89-real;90-real")
#   --cpus       cores for all builds together (default: the online CPUs)
#   --memory-gb  memory in GB for all builds together (default: the available memory)
#   --parallel   ABIs that build at the same time (default: as many as keep --jobs at 8 or more each)
#   --jobs       build jobs of each ABI (default: from --cpus, --memory-gb and --parallel)
#   --cuda-jobs  parallel nvcc compiles of each ABI (default: --jobs - 1)
#   --engine     podman, docker or apptainer (default: the first of them that is installed)
#   --image      the image (default: the one in pyproject.toml); for apptainer also a .sif file
#   --dry-run    print the limits and the commands, run nothing
#
# Parallelism: an ABI builds with --jobs build jobs, at most --cuda-jobs of them nvcc compiles of up to 3 GB each, and
# one compile at a time of the four largest kernels, which takes up to 7 GB (cuda/lifted_units.cmake). A build of J
# jobs therefore stays within 7 + 3 (J - 1) GB plus 2 GB for Ninja, the linker and pip, and the default --jobs is the
# largest J within the memory and the cores of one ABI: --memory-gb and --cpus divided by --parallel. With podman and
# docker each container is limited to that share of the memory; --cpus is enforced only through --jobs. The ABIs
# build side by side as --parallel allows, each in a container of its own, after a first container has installed the
# toolchain.
#
# With rootful docker the files in --toolchain, --work and --out belong to root.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
toolchain=
work=
out=$repo/wheelhouse
abis="cp314t cp315t"
archs="86-real;89-real;90-real"
cpus=
memory_gb=
parallel=
jobs=
cuda_jobs=
engine=
image=
dry_run=0

# GB of memory: one nvcc compile, the largest kernels' compile, and Ninja, the linker and pip of one build
compile_gb=3
heavy_gb=7
reserve_gb=2

usage() {
    sed -n '/^# usage:/,/^#   --dry-run/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1; shift; continue ;;
        --toolchain) toolchain=${2:-} ;;
        --work) work=${2:-} ;;
        --out) out=${2:-} ;;
        --python) abis=${2:-} ;;
        --archs) archs=${2:-} ;;
        --cpus) cpus=${2:-} ;;
        --memory-gb) memory_gb=${2:-} ;;
        --parallel) parallel=${2:-} ;;
        --jobs) jobs=${2:-} ;;
        --cuda-jobs) cuda_jobs=${2:-} ;;
        --engine) engine=${2:-} ;;
        --image) image=${2:-} ;;
        *) usage ;;
    esac
    [ $# -ge 2 ] || usage
    shift 2
done
if [ -z "$toolchain" ] || [ -z "$work" ]; then
    usage
fi
read -r -a abi_list <<< "$abis"
[ ${#abi_list[@]} -gt 0 ] || { echo "error: --python names no ABI" >&2; exit 2; }
for abi in "${abi_list[@]}"; do
    [[ "$abi" =~ ^cp3[0-9][0-9]t?$ ]] || { echo "error: expected CPython ABIs such as cp314t, got '$abi'" >&2; exit 2; }
done
for value in "$cpus" "$memory_gb" "$parallel" "$jobs" "$cuda_jobs"; do
    [[ "$value" =~ ^([1-9][0-9]*)?$ ]] || {
        echo "error: --cpus, --memory-gb, --parallel, --jobs and --cuda-jobs must be positive integers" >&2
        exit 2
    }
done
if [ -z "$engine" ]; then
    for candidate in podman docker apptainer; do
        if command -v "$candidate" > /dev/null; then
            engine=$candidate
            break
        fi
    done
    [ -n "$engine" ] || { echo "error: no podman, docker or apptainer found; pass --engine" >&2; exit 1; }
fi
case "$engine" in
    podman | docker | apptainer) ;;
    *) echo "error: --engine takes podman, docker or apptainer, not '$engine'" >&2; exit 2 ;;
esac
if [ -z "$image" ]; then
    image=$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["tool"]["cibuildwheel"]["linux"]["manylinux-x86_64-image"])' \
        "$repo/pyproject.toml")
fi
if [ "$engine" = apptainer ] && [[ "$image" != *.sif ]]; then
    image=docker://$image
fi

if [ -z "$cpus" ]; then
    cpus=$(getconf _NPROCESSORS_ONLN)
fi
if [ -z "$memory_gb" ]; then
    [ -r /proc/meminfo ] || { echo "error: no /proc/meminfo to read the available memory from; pass --memory-gb" >&2; exit 2; }
    memory_gb=$(($(sed -n 's/^MemAvailable: *\([0-9]*\) kB$/\1/p' /proc/meminfo) / 1024 / 1024))
fi

# the jobs of one of P builds side by side: the largest J within its share of the cores and of the memory
jobs_of() {
    local share_gb=$((memory_gb / $1)) by_memory
    by_memory=$(((share_gb - heavy_gb - reserve_gb) / compile_gb + 1))
    [ $by_memory -le $((cpus / $1)) ] || by_memory=$((cpus / $1))
    echo $by_memory
}
if [ -z "$parallel" ]; then
    parallel=${#abi_list[@]}
    while [ "$parallel" -gt 1 ] && [ "$(jobs_of "$parallel")" -lt 8 ]; do
        parallel=$((parallel - 1))
    done
fi
[ "$parallel" -le ${#abi_list[@]} ] || parallel=${#abi_list[@]}
share_gb=$((memory_gb / parallel))
if [ -z "$jobs" ]; then
    jobs=$(jobs_of "$parallel")
    if [ "$jobs" -lt 1 ]; then
        echo "error: $memory_gb GB for $parallel builds at once leaves less than the" \
             "$((heavy_gb + reserve_gb + compile_gb)) GB that one build needs" >&2
        exit 1
    fi
fi
if [ -z "$cuda_jobs" ]; then
    cuda_jobs=$((jobs > 1 ? jobs - 1 : 1))
fi
# the toolchain builds alone, with every core that the memory allows
toolchain_jobs=$(((memory_gb - reserve_gb) / compile_gb))
[ "$toolchain_jobs" -le "$cpus" ] || toolchain_jobs=$cpus
[ "$toolchain_jobs" -ge 1 ] || toolchain_jobs=1

if [ $dry_run -eq 0 ]; then
    [ "$(uname -s)" = Linux ] || { echo "error: the CUDA wheels build on Linux x86_64 only" >&2; exit 1; }
    mkdir -p "$toolchain" "$work" "$out"
    toolchain=$(cd "$toolchain" && pwd)
    work=$(cd "$work" && pwd)
    out=$(cd "$out" && pwd)
fi

# the container command that runs ci/cuda_wheel.sh with a work directory, a memory limit in GB and the given arguments
container() {
    local dir=$1 gb=$2
    shift 2
    case "$engine" in
        podman | docker)
            command=("$engine" run --rm --memory "${gb}g" --volume "$repo:/src"
                --volume "$toolchain:/opt/mymyr-toolchain" --volume "$dir:/work" --volume "$out:/out" "$image") ;;
        apptainer)
            command=(apptainer exec --containall --bind "$repo:/src,$toolchain:/opt/mymyr-toolchain,$dir:/work,$out:/out"
                "$image") ;;
    esac
    command+=(bash /src/ci/cuda_wheel.sh --work /work "$@")
}
show() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    echo
}

echo "== $cpus cores and $memory_gb GB: $parallel of ${#abi_list[@]} ABIs at a time, each with --jobs $jobs" \
     "--cuda-jobs $cuda_jobs in $share_gb GB; the toolchain with --jobs $toolchain_jobs"
container "$work/toolchain" "$memory_gb" --stage toolchain --jobs "$toolchain_jobs"
toolchain_command=("${command[@]}")
if [ $dry_run -eq 1 ]; then
    show "${toolchain_command[@]}"
    for abi in "${abi_list[@]}"; do
        container "$work/$abi" "$share_gb" --python "$abi" --archs "$archs" --out /out --jobs "$jobs" --cuda-jobs "$cuda_jobs"
        show "${command[@]}"
    done
    exit 0
fi

elapsed() {
    local seconds=$(($(date +%s) - $1))
    printf '%dh%02dm' $((seconds / 3600)) $((seconds % 3600 / 60))
}

mkdir -p "$work/toolchain"
echo "== toolchain in $toolchain, log in $work/toolchain.log"
start=$(date +%s)
if ! "${toolchain_command[@]}" > "$work/toolchain.log" 2>&1; then
    tail -n 30 "$work/toolchain.log" >&2
    echo "error: the toolchain failed after $(elapsed "$start"); its log is $work/toolchain.log" >&2
    exit 1
fi
echo "== toolchain ready after $(elapsed "$start")"

# each ABI in a background container; at most --parallel run at once, and the oldest is waited for first
pids=() starts=()
reaped=0
failed=()
stop() {
    local pid
    for pid in ${pids[@]+"${pids[@]}"}; do
        kill "$pid" 2> /dev/null || true
    done
    exit 130
}
trap stop HUP INT TERM
reap() {
    local abi=${abi_list[$reaped]} began=${starts[$reaped]}
    if wait "${pids[$reaped]}"; then
        echo "== $abi: built after $(elapsed "$began")"
    else
        tail -n 30 "$work/$abi.log" >&2
        echo "error: $abi failed after $(elapsed "$began"); its log is $work/$abi.log" >&2
        failed+=("$abi")
    fi
    reaped=$((reaped + 1))
}
for abi in "${abi_list[@]}"; do
    [ $((${#pids[@]} - reaped)) -lt "$parallel" ] || reap
    mkdir -p "$work/$abi"
    container "$work/$abi" "$share_gb" --python "$abi" --archs "$archs" --out /out --jobs "$jobs" --cuda-jobs "$cuda_jobs"
    "${command[@]}" > "$work/$abi.log" 2>&1 &
    pids+=($!)
    starts+=("$(date +%s)")
    echo "== $abi: building, log in $work/$abi.log"
done
while [ $reaped -lt ${#pids[@]} ]; do
    reap
done
trap - HUP INT TERM
if [ ${#failed[@]} -gt 0 ]; then
    echo "error: no wheel for ${failed[*]}" >&2
    exit 1
fi
echo "== every wheel built after $(elapsed "$start") in $out:"
ls -l "$out"
