#!/usr/bin/env bash
# Test the CUDA wheels of a release on GPU 0 and attach them to the release's GitHub Release.
#
# usage: ci/attach_cuda_wheels.sh [--dry-run] [--run RUN_ID | --dir DIR] --python PATH [--python PATH ...]
#                                 [--archs LIST] TAG
#   TAG        the release tag v<version>; the wheels must be mymyr-<version>+cu13-*.whl
#   --run      the workflow run whose cuda-wheel-* artifacts hold the wheels (default: the latest release workflow
#              run of TAG); a manual run of the CUDA wheel workflow has its own id
#   --dir      test the wheels in DIR instead of downloading them, e.g. those of ci/cuda_wheel.sh
#   --python   an interpreter to test with; every wheel needs one of its ABI (cp314t: a free-threaded 3.14). Its
#              site packages are visible in the test environment, so torch and JAX installed there take part
#   --archs    the GPU architectures every wheel must have SASS for, space-separated (default "89 90")
#   --dry-run  test the wheels and do not upload them
#
# Every wheel is installed into a fresh virtual environment and checked with ci/check_cuda_install.py (version, GPU
# code, JAX targets, SASS for GPU 0), then the CUDA tests run on GPU 0 from this checkout. The upload needs every
# wheel to pass, the checkout to be at TAG and the GitHub Release of TAG to exist; it runs gh release upload.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dry_run=0
run_id=
wheel_dir=
pythons=()
archs="89 90"
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
    echo "== downloading the cuda-wheel-* artifacts of run $run_id"
    (cd "$repo" && gh run download "$run_id" --pattern 'cuda-wheel-*' --dir "$work/download")
    wheel_dir=$work/download
fi
mapfile -t wheels < <(find "$wheel_dir" -name 'mymyr-*.whl' | sort)
[ ${#wheels[@]} -gt 0 ] || { echo "error: no mymyr wheels in $wheel_dir" >&2; exit 1; }

abi_of() {
    "$1" -c 'import sys, sysconfig; print(f"cp{sys.version_info[0]}{sys.version_info[1]}" + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""))'
}

for wheel in "${wheels[@]}"; do
    name=$(basename "$wheel")
    abi=$(echo "$name" | cut -d- -f4)
    echo
    echo "== $name ($(du -m "$wheel" | cut -f1) MB)"
    case "$name" in
        mymyr-"$version"-*-*-manylinux_2_34_x86_64.whl) ;;
        *) echo "error: expected mymyr-$version-<python>-<abi>-manylinux_2_34_x86_64.whl" >&2; exit 1 ;;
    esac
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
    "$venv/bin/python" -m pip install --quiet "$wheel" "pytest>=8" "nanobind>=3.1,<4"
    (
        cd "$work"
        export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
        "$venv/bin/python" "$repo/ci/check_cuda_install.py" --version "$version" --archs "$archs" --gpu
        "$venv/bin/python" -m pytest -W error -q -rs -p no:cacheprovider \
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
