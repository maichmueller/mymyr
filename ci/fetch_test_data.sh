#!/usr/bin/env bash
# Assemble the benchmark data the parity tests read: sparse single-commit checkouts of pinned
# revisions, plus the IPC tasks kept in tests/data/ipc (the fork does not publish them).
#
# usage: ci/fetch_test_data.sh DEST
#
# Layout under DEST (matching MYMYR_FORK_DATA, MYMYR_IPC and MYMYR_WORK of the test suites):
#   DEST/mimir-fork/data         fork benchmark data, with ipc/ from tests/data/ipc
#   DEST/work/mimir-cs/Benchmark upstream benchmarks (C# branch)
#   DEST/work/mimir/data         upstream benchmarks (main branch)
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

FORK_URL=https://github.com/maichmueller/mimir.git
FORK_SHA=81d771dd7b361591592261f30b24c83b7a9b1f53
UPSTREAM_URL=https://github.com/simon-stahlberg/mimir.git
UPSTREAM_CS_SHA=1674dfcdf01e35a9a98a62d74542eb78ce8a8ec9
UPSTREAM_SHA=c664ba695b7117055e3a7ade5e968a55a86003f9

CS_DOMAINS="
Benchmark/adl/airport-adl Benchmark/adl/caldera-split-opt18-adl Benchmark/adl/folding-opt23-adl
Benchmark/adl/miconic-simpleadl Benchmark/adl/openstacks-opt08-adl Benchmark/adl/pathways
Benchmark/adl/rubiks-cube-opt23-adl Benchmark/adl/rubiks-cube-sat23-adl
Benchmark/numeric/block-grouping Benchmark/numeric/counters Benchmark/numeric/delivery
Benchmark/numeric/drone Benchmark/numeric/expedition Benchmark/numeric/ext-plant-watering
Benchmark/numeric/farmland Benchmark/numeric/hydropower Benchmark/numeric/sailing Benchmark/numeric/tpp
Benchmark/strips/blocks Benchmark/strips/depot Benchmark/strips/driverlog Benchmark/strips/freecell
Benchmark/strips/gripper Benchmark/strips/logistics00 Benchmark/strips/miconic
Benchmark/strips/organic-synthesis-opt18-strips Benchmark/strips/organic-synthesis-sat18-strips
Benchmark/strips/parcprinter-opt11-strips Benchmark/strips/pegsol-08-strips Benchmark/strips/rovers
Benchmark/strips/snake-opt18-strips Benchmark/strips/sokoban-opt08-strips
Benchmark/strips/sokoban-opt11-strips Benchmark/strips/sokoban-sat08-strips
Benchmark/strips/transport-opt08-strips Benchmark/strips/visitall Benchmark/strips/zenotravel
"
UPSTREAM_DOMAINS="
data/barman data/fo-counters data/refuel data/refuel-adl data/sokoban data/tpp/numeric data/transport
data/woodworking data/zenotravel/numeric
"

if [ $# -ne 1 ]; then
    echo "usage: $0 DEST" >&2
    exit 2
fi
dest=$1
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)

# fetch URL SHA DIR PATHS...: DIR receives the files of PATHS at commit SHA, without git metadata.
fetch() {
    local url=$1 sha=$2 dir=$3
    shift 3
    if [ -f "$dir/.fetched-$sha" ]; then
        echo "$dir: already at $sha"
        return
    fi
    rm -rf "$dir"
    mkdir -p "$dir"
    cd "$dir"
    git init -q .
    git remote add origin "$url"
    git sparse-checkout set --cone "$@"
    echo "fetching $sha of ${url##*/} into $dir"
    git fetch -q --depth 1 --filter=blob:none origin "$sha"
    git checkout -q FETCH_HEAD
    rm -rf .git
    touch ".fetched-$sha"
    cd "$dest"
}

cd "$dest"
fetch "$FORK_URL" "$FORK_SHA" "$dest/mimir-fork" data
# shellcheck disable=SC2086
fetch "$UPSTREAM_URL" "$UPSTREAM_CS_SHA" "$dest/work/mimir-cs" $CS_DOMAINS
# shellcheck disable=SC2086
fetch "$UPSTREAM_URL" "$UPSTREAM_SHA" "$dest/work/mimir" $UPSTREAM_DOMAINS

rm -rf "$dest/mimir-fork/data/ipc"
cp -r "$repo/tests/data/ipc" "$dest/mimir-fork/data/ipc"

for d in mimir-fork/data work/mimir-cs/Benchmark work/mimir/data; do
    if [ -z "$(ls -A "$dest/$d" 2>/dev/null)" ]; then
        echo "error: $dest/$d is empty after the fetch" >&2
        exit 1
    fi
done
echo "test data ready in $dest"
