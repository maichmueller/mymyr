#!/usr/bin/env bash
# Test the CUDA wheels of a release on GPU 0 and attach them to the release's GitHub Release.
#
# usage: ci/attach_cuda_wheels.sh [--dry-run] [--run RUN_ID | --dir DIR] --python PATH [--python PATH ...]
#                                 [--abis LIST] [--archs LIST] TAG
#   TAG        the release tag v<version>; the wheels must be mymyr-<version>+cu13-*.whl
#   --run      the workflow run whose cuda-wheel-* artifacts hold the wheels (default: the latest release workflow
#              run of TAG); a manual run of the CUDA wheel workflow has its own id. The run must have completed
#              successfully
#   --dir      test the wheels in DIR instead of downloading them, e.g. those of ci/cuda_wheel.sh
#   --python   an interpreter to test with; every wheel needs one of its ABI (cp314t: a free-threaded 3.14). Its
#              site packages are visible in the test environment, so JAX installed there takes part
#   --abis     the CPython ABIs there must be exactly one wheel for, space-separated (default "cp314t cp315t", the
#              CUDA wheel workflow's)
#   --archs    the GPU architectures every wheel must have SASS for, space-separated (default "86 89 90", the
#              CUDA wheel workflow's)
#   --dry-run  test the wheels and do not upload them
#
# The set of wheels must be complete before anything is tested: one per ABI in --abis and no others. Every wheel is
# installed into a fresh virtual environment, with the pinned test requirements (ci/cuda_wheel_test_requirements.txt:
# pytest, NumPy and torch 2.14 for CUDA 13.0) first and then the wheel itself with --force-reinstall, so that a mymyr
# of the same version in the interpreter's site packages cannot stand in for it. ci/check_cuda_install.py imports
# torch first and checks that mymyr is the wheel's (its files and version), its GPU code (SASS for every --archs and
# GPU 0), its JAX targets and that torch and mymyr exchange arrays on GPU 0; then the CUDA tests of this checkout run
# on GPU 0 in a process that imports torch before mymyr. The upload needs every wheel to pass, the checkout to be at
# TAG and the GitHub Release of TAG to exist; it runs gh release upload.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dry_run=0
run_id=
wheel_dir=
pythons=()
abis="cp314t cp315t"
archs="86 89 90"
tag=

usage() {
    sed -n '/^# usage:/,/^#   --dry-run/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1; shift; continue ;;
        --run) run_id=${2:-} ;;
        --dir) wheel_dir=${2:-} ;;
        --python) pythons+=("${2:-}") ;;
        --abis) abis=${2:-} ;;
        --archs) archs=${2:-} ;;
        -*) usage ;;
        *) [ -z "$tag" ] || usage; tag=$1; shift; continue ;;
    esac
    [ $# -ge 2 ] || usage
    shift 2
done
if [ -z "$tag" ] || [ ${#pythons[@]} -eq 0 ]; then
    usage
fi
if [ -n "$run_id" ] && [ -n "$wheel_dir" ]; then
    echo "error: --run and --dir exclude each other" >&2
    exit 2
fi
case "$tag" in
    v[0-9]*) version=${tag#v}+cu13 ;;
    *) echo "error: expected a tag v<version>, got '$tag'" >&2; exit 2 ;;
esac
read -r -a abi_list <<< "$abis"
[ ${#abi_list[@]} -gt 0 ] || { echo "error: --abis names no ABI" >&2; exit 2; }

if [ $dry_run -eq 0 ]; then
    (cd "$repo" && gh release view "$tag" --json tagName > /dev/null)
    head=$(git -C "$repo" rev-parse HEAD)
    tagged=$(git -C "$repo" rev-parse "$tag^{commit}")
    if [ "$head" != "$tagged" ]; then
        echo "error: the tests come from this checkout, which is at $head, not at $tag ($tagged)" >&2
        exit 1
    fi
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/mymyr-cuda-wheels.XXXXXX")
trap 'rm -rf "$work"' EXIT

if [ -z "$wheel_dir" ]; then
    if [ -z "$run_id" ]; then
        run_id=$(cd "$repo" && gh run list --workflow release.yml --branch "$tag" --limit 1 --json databaseId \
            --jq '.[0].databaseId')
        [ -n "$run_id" ] || { echo "error: no release workflow run for $tag" >&2; exit 1; }
    fi
    # a run that is still going or that failed may have the artifacts of some ABIs only
    state=$(cd "$repo" && gh run view "$run_id" --json status,conclusion --jq '"\(.status) \(.conclusion)"')
    if [ "$state" != "completed success" ]; then
        echo "error: run $run_id is '$state', not 'completed success'" >&2
        exit 1
    fi
    echo "== downloading the cuda-wheel-* artifacts of run $run_id"
    (cd "$repo" && gh run download "$run_id" --pattern 'cuda-wheel-*' --dir "$work/download")
    wheel_dir=$work/download
fi
wheels=()
while IFS= read -r wheel; do
    wheels+=("$wheel")
done < <(find "$wheel_dir" -name 'mymyr-*.whl' | sort)
# stops here without wheels: under set -u, bash before 4.4 (that of macOS is 3.2) refuses to expand an empty array
if [ ${#wheels[@]} -eq 0 ]; then
    echo "error: no mymyr-*.whl in $wheel_dir; nothing tested or attached" >&2
    exit 1
fi

# one wheel per expected ABI, each with the release's version and platform tag, and no others
status=0
for wheel in "${wheels[@]}"; do
    name=$(basename "$wheel")
    case "$name" in
        mymyr-"$version"-*-*-manylinux_2_34_x86_64.whl) ;;
        *) echo "error: $name is not mymyr-$version-<python>-<abi>-manylinux_2_34_x86_64.whl" >&2; status=1 ;;
    esac
    abi=$(echo "$name" | cut -d- -f4)
    case " $abis " in
        *" $abi "*) ;;
        *) echo "error: $name is for $abi, which is not in --abis '$abis'" >&2; status=1 ;;
    esac
done
for abi in "${abi_list[@]}"; do
    count=0
    for wheel in "${wheels[@]}"; do
        [ "$(basename "$wheel" | cut -d- -f4)" != "$abi" ] || count=$((count + 1))
    done
    [ $count -eq 1 ] || { echo "error: $count wheels for $abi in $wheel_dir, expected one" >&2; status=1; }
done
if [ $status -ne 0 ]; then
    echo "error: the wheels in $wheel_dir are not the complete set for ${abis// /, }; nothing tested or attached" >&2
    exit 1
fi

abi_of() {
    "$1" -c 'import sys, sysconfig; print(f"cp{sys.version_info[0]}{sys.version_info[1]}" + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""))'
}

for wheel in "${wheels[@]}"; do
    name=$(basename "$wheel")
    abi=$(echo "$name" | cut -d- -f4)
    echo
    echo "== $name ($(du -m "$wheel" | cut -f1) MB)"
    python=
    for candidate in "${pythons[@]}"; do
        if [ "$(abi_of "$candidate")" = "$abi" ]; then
            python=$candidate
            break
        fi
    done
    [ -n "$python" ] || { echo "error: no --python interpreter for $abi" >&2; exit 1; }
    venv=$work/venv-$abi
    "$python" -m venv --system-site-packages "$venv"
    "$venv/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes --no-deps \
        --only-binary :all: -r "$repo/ci/cuda_wheel_test_requirements.txt"
    "$venv/bin/python" -m pip install --quiet --disable-pip-version-check --force-reinstall --no-deps "$wheel"
    (
        cd "$work"
        export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
        "$venv/bin/python" "$repo/ci/check_cuda_install.py" --version "$version" --archs "$archs" --gpu \
            --wheel "$wheel" --torch 2.14 --task "$repo/tests/data/tasks/blocks__probBLOCKS-8-0.txt"
        "$venv/bin/python" -c 'import sys, torch, pytest; sys.exit(pytest.main(sys.argv[1:]))' \
            -W error -q -rs -p no:cacheprovider \
            "$repo"/tests/python/test_cuda*.py \
            "$repo/tests/python/test_encoding_cuda.py" \
            "$repo/tests/python/test_encoding_numeric_cuda.py" \
            "$repo/tests/python/test_rl_torch.py" \
            "$repo/tests/python/test_rl_jax.py" \
            "$repo/tests/python/test_encoding_jax.py"
    )
done

echo
if [ $dry_run -eq 1 ]; then
    echo "== every wheel passed; dry run, nothing uploaded"
    exit 0
fi
echo "== every wheel passed; attaching them to the GitHub Release $tag"
(cd "$repo" && gh release upload "$tag" "${wheels[@]}")
