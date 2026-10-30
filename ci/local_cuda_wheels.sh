#!/usr/bin/env bash
# Build the CUDA wheels of several CPython ABIs on one Linux host: ci/cuda_wheel.sh for each ABI, in the manylinux
# image of the release wheels (manylinux-x86_64-image under [tool.cibuildwheel.linux] in pyproject.toml), with
# podman, docker or apptainer. The wheels are those of the CUDA wheel workflow (version <version>+cu13, repaired by
# auditwheel, checked and passed through the CPU test suite) and go to ci/attach_cuda_wheels.sh, which tests them on a
# GPU and attaches them to the GitHub Release.
#
# usage: ci/local_cuda_wheels.sh --toolchain DIR --work DIR [--out DIR] [--python LIST] [--archs LIST] [--cpus N]
#                                [--memory-gb N] [--parallel N] [--jobs N] [--cuda-jobs N] [--heavy-jobs N]
#                                [--nvcc-threads N] [--engine ENGINE] [--image IMAGE] [--dry-run]
#   --toolchain     GCC 16, the dependencies and the CUDA compiler (about 1.7 GB): the first run installs them,
#                   later runs reuse them
#   --work          one work directory per ABI, <work>/<abi> (about 1.2 GB per architecture while it builds, then
#                   4.5 GB for the test environment), and the logs <work>/<abi>.log and <work>/toolchain.log
#   --out           receives the wheels (default: wheelhouse/ in the checkout)
#   --python        CPython ABIs, space-separated (default "cp314t cp315t")
#   --archs         CMAKE_CUDA_ARCHITECTURES (default "86-real;89-real;90-real")
#   --cpus          cores for all builds together (default: the online CPUs)
#   --memory-gb     memory in GB for all builds together (default: the available memory)
#   --parallel      ABIs that build at the same time (default: as many as keep --jobs at 4 or more each)
#   --jobs          build jobs of each ABI (default: from --cpus, --memory-gb and --parallel)
#   --cuda-jobs     parallel nvcc compiles of each ABI besides those of the largest kernels (default: --jobs minus
#                   --heavy-jobs)
#   --heavy-jobs    parallel nvcc compiles of the largest kernels of each ABI (default: from the memory, at most 3)
#   --nvcc-threads  architectures that one nvcc compile builds at once (default: all of --archs)
#   --engine        podman, docker or apptainer (default: the first of them that is installed)
#   --image         the image (default: the one in pyproject.toml); for apptainer also a .sif file
#   --dry-run       print the limits and the commands, run nothing
#
# Parallelism: an nvcc compile builds T = --nvcc-threads architectures at once (nvcc --threads), and so takes up to T
# cores and the memory of T compiler processes, 1.25 GB each on average and up to 5 GB for the largest kernels (the
# units of mymyr_cuda_lifted_heavy in cuda/lifted_units.cmake, whose compiles run in a pool of --heavy-jobs). A build of
# J jobs, H of them heavy, thus takes up to J T cores and about 2 + T (5 H + 1.25 (J - H)) GB, 2 GB for Ninja, the
# linker and pip. The defaults give each ABI its share of --cpus and --memory-gb (divided by --parallel): the most
# jobs that the cores allow, with the most heavy compiles (up to 3) that leave room for them, and else fewer jobs and
# one heavy compile. With podman and docker each container is limited to that share of the memory; the cores are
# bounded only through the jobs. The ABIs build side by side as --parallel allows, each in a container of its own,
# after a first container has installed the toolchain.
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
heavy_jobs=
nvcc_threads=
engine=
image=
dry_run=0

# Memory of the compiler processes, measured over a build of the three release architectures: a compiler process
# (cicc or ptxas for one architecture of an nvcc compile, or GCC's cc1plus) takes 1.25 GB on average, in hundredths of
# a GB (a busy build's mean per process was at most 1.32 GB, 1.2 GB in 95 of 100 samples); one architecture of the
# largest kernels takes up to 5 GB (ptxas, 4.8 GB measured); Ninja, the linker and pip of one build 2 GB. Single
# compiles above the average (ptxas up to 3 GB in 95 of 100 samples) fit because most are below it, and the podman
# and docker memory limit stops a build that does not fit.
compile_cgb=125
heavy_cgb=500
reserve_gb=2
# the largest kernels are six units (cuda/lifted_units.cmake); at most half of them compile at once per build
max_heavy=3

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
        --heavy-jobs) heavy_jobs=${2:-} ;;
        --nvcc-threads) nvcc_threads=${2:-} ;;
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
for value in "$cpus" "$memory_gb" "$parallel" "$jobs" "$cuda_jobs" "$heavy_jobs" "$nvcc_threads"; do
    [[ "$value" =~ ^([1-9][0-9]*)?$ ]] || {
        echo "error: --cpus, --memory-gb, --parallel, --jobs, --cuda-jobs, --heavy-jobs and --nvcc-threads must be" \
             "positive integers" >&2
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
    # sed rather than tomllib: the host's python3 may predate 3.11
    image=$(sed -n '/^\[tool\.cibuildwheel\.linux\]$/,/^\[/s/^manylinux-x86_64-image *= *"\(.*\)"$/\1/p' \
        "$repo/pyproject.toml")
    [ -n "$image" ] || { echo "error: no manylinux-x86_64-image in $repo/pyproject.toml; pass --image" >&2; exit 1; }
fi
if [ "$engine" = apptainer ] && [[ "$image" != *.sif ]]; then
    image=docker://$image
fi

if [ -z "$cpus" ]; then
    cpus=$(getconf _NPROCESSORS_ONLN)
fi
if [ -z "$memory_gb" ]; then
    [ -r /proc/meminfo ] || {
        echo "error: no /proc/meminfo to read the available memory from; pass --memory-gb" >&2
        exit 2
    }
    memory_gb=$(($(sed -n 's/^MemAvailable: *\([0-9]*\) kB$/\1/p' /proc/meminfo) / 1024 / 1024))
fi

if [ -z "$nvcc_threads" ]; then
    IFS=';' read -r -a arch_list <<< "$archs"
    nvcc_threads=${#arch_list[@]}
    [ "$nvcc_threads" -ge 1 ] || nvcc_threads=1
fi

# The plan of one of P builds side by side, in plan_jobs and plan_heavy: the jobs J and heavy compiles H within its
# share of the cores (J nvcc compiles of T threads each) and of the memory (reserve + T (H heavy + (J - H) compile)).
# The most heavy compiles, up to max_heavy, that still leave J at the cores' limit, else one; plan_jobs is 0 when the
# memory is too small for one heavy compile.
plan() {
    local share_gb=$((memory_gb / $1)) by_cores=$((cpus / $1 / nvcc_threads)) h by_memory
    [ $by_cores -ge 1 ] || by_cores=1
    plan_jobs=0 plan_heavy=1
    for ((h = max_heavy; h >= 1; h--)); do
        # in units of 1/100 GB, as the per-compile memory is not a whole number of GB
        local rest=$((share_gb * 100 - reserve_gb * 100 - h * nvcc_threads * heavy_cgb))
        [ $rest -ge 0 ] || continue
        by_memory=$((h + rest / (nvcc_threads * compile_cgb)))
        [ $by_memory -le $by_cores ] || by_memory=$by_cores
        if [ $by_memory -ge $by_cores ] || [ $h -eq 1 ]; then
            plan_jobs=$by_memory plan_heavy=$h
            [ $plan_heavy -le $plan_jobs ] || plan_heavy=$plan_jobs
            return
        fi
    done
}
if [ -z "$parallel" ]; then
    parallel=${#abi_list[@]}
    while [ "$parallel" -gt 1 ]; do
        plan "$parallel"
        [ $plan_jobs -lt 4 ] || break
        parallel=$((parallel - 1))
    done
fi
[ "$parallel" -le ${#abi_list[@]} ] || parallel=${#abi_list[@]}
share_gb=$((memory_gb / parallel))
plan "$parallel"
if [ -z "$jobs" ]; then
    jobs=$plan_jobs
    if [ "$jobs" -lt 1 ]; then
        echo "error: $share_gb GB per build ($memory_gb GB for $parallel at a time) is less than the" \
             "$((reserve_gb + (nvcc_threads * heavy_cgb + 99) / 100)) GB that one build needs" >&2
        exit 1
    fi
fi
if [ -z "$heavy_jobs" ]; then
    heavy_jobs=$plan_heavy
    [ "$heavy_jobs" -le "$jobs" ] || heavy_jobs=$jobs
fi
if [ -z "$cuda_jobs" ]; then
    # the heavy compiles have a pool of their own, so the others leave them their slots
    cuda_jobs=$((jobs > heavy_jobs ? jobs - heavy_jobs : 1))
fi
# the toolchain builds alone, with every core that the memory allows
toolchain_jobs=$(((memory_gb - reserve_gb) * 100 / compile_cgb))
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
            command=(apptainer exec --containall
                --bind "$repo:/src,$toolchain:/opt/mymyr-toolchain,$dir:/work,$out:/out" "$image") ;;
    esac
    command+=(bash /src/ci/cuda_wheel.sh --work /work "$@")
}
# the container command that builds, checks and tests the wheel of one ABI
wheel_container() {
    container "$work/$1" "$share_gb" --python "$1" --archs "$archs" --out /out --jobs "$jobs" --cuda-jobs "$cuda_jobs" \
        --heavy-jobs "$heavy_jobs" --nvcc-threads "$nvcc_threads"
}
show() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    echo
}

echo "== $cpus cores and $memory_gb GB: $parallel of ${#abi_list[@]} ABIs at a time, each with --jobs $jobs" \
     "--cuda-jobs $cuda_jobs --heavy-jobs $heavy_jobs --nvcc-threads $nvcc_threads in $share_gb GB;" \
     "the toolchain with --jobs $toolchain_jobs"
container "$work/toolchain" "$memory_gb" --stage toolchain --jobs "$toolchain_jobs"
toolchain_command=("${command[@]}")
if [ $dry_run -eq 1 ]; then
    show "${toolchain_command[@]}"
    for abi in "${abi_list[@]}"; do
        wheel_container "$abi"
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
    wheel_container "$abi"
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
