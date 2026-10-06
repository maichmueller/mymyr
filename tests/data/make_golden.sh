#!/usr/bin/env bash
# Regenerates tests/data/golden/ with the mimir fork's exporters (fork v0.16.3; built by build_fork.sh). The
# front-end tests (tests/cpp/frontend) compare mymyr's TaskData against these.
#
#   suite/<name>.txt          export_lifted_fork   on the 22-task BrFS suite (export_all.py's BFS_INSTANCES)
#   suite-numeric/<name>.txt  export_numeric_fork  on the same tasks (adds functions, action costs, metric)
#   numeric/<name>.txt        export_numeric_fork  on 20 numeric tasks
#   ipc/<domain>__<split>__<problem>.txt
#                             export_numeric_fork  on a sample of the fork's IPC-2023 tasks (data/ipc/<domain>[-ipc]/)
#
# The fork's export depends on heap addresses in typed domains (loki's pointer-keyed unordered sets; tests/cpp/
# frontend/golden.hpp). Exporters run with ASLR off (setarch -R) where possible, which makes a regeneration on the same
# machine reproduce the same files; the tests compare modulo that freedom anyway.
#
# Usage: tests/data/make_golden.sh [--work DIR] [--fork-data DIR] [--out DIR] [--timeout SECONDS]
#   --work       holds mimir-cs/, mimir/, build-exporter/, build-numeric-exporter/   (default: $MYMYR_WORK or <repo>/.work)
#   --fork-data  the fork's data/ directory with ipc/                                   (default: $MYMYR_FORK_DATA; the
#                ipc/ exports are skipped when unset)
#   --out        output directory                                                        (default: <repo>/tests/data/golden)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="${MYMYR_WORK:-$repo/.work}"
fork_data="${MYMYR_FORK_DATA:-}"
out="$repo/tests/data/golden"
limit=900
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work) work="$2"; shift 2 ;;
        --fork-data) fork_data="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --timeout) limit="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

lifted="$work/build-exporter/export_lifted_fork"
numeric="$work/build-numeric-exporter/export_numeric_fork"
for exe in "$lifted" "$numeric"; do
    [[ -x "$exe" ]] || { echo "missing exporter $exe (build the fork: tests/data/fork_golden/build_fork.sh)" >&2; exit 1; }
done
norand=()
if command -v setarch >/dev/null && setarch "$(uname -m)" -R true 2>/dev/null; then
    norand=(setarch "$(uname -m)" -R)
fi

failures=0
count=0
# export <exporter> <domain> <problem> <out file>
export_one() {
    local exe="$1" dom="$2" prob="$3" dst="$4"
    if [[ ! -f "$dom" || ! -f "$prob" ]]; then
        echo "  skip (PDDL missing): $prob"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    if timeout "$limit" "${norand[@]}" "$exe" "$dom" "$prob" "$dst.tmp" >/dev/null 2>"$dst.err"; then
        mv "$dst.tmp" "$dst"
        rm -f "$dst.err"
        count=$((count + 1))
    else
        echo "  FAILED: $exe $dom $prob ($(tail -c 300 "$dst.err" | tr '\n' ' '))"
        rm -f "$dst.tmp"
        failures=$((failures + 1))
    fi
}

# The domain file next to the problem.
domain_of() {
    local dir="$1" stem="$2"
    for cand in "domain_$stem.pddl" "domain-$stem.pddl" "domain.pddl"; do
        [[ -f "$dir/$cand" ]] && { echo "$dir/$cand"; return; }
    done
    echo "$dir/domain.pddl"
}

echo "suite -> $out/suite, $out/suite-numeric"
bench="$work/mimir-cs/Benchmark"
# tag problem [patched]: patched problems (adjusted for this exporter) live under tests/data/pddl/<name> instead
while read -r tag prob root; do
    [[ -z "$tag" ]] && continue
    dir="$bench/$tag"
    [[ "$root" == patched ]] && dir="$repo/tests/data/pddl/$(basename "$tag")"
    stem="${prob%.pddl}"
    name="$(basename "$tag")__$stem"
    dom="$(domain_of "$dir" "$stem")"
    export_one "$lifted" "$dom" "$dir/$prob" "$out/suite/$name.txt"
    export_one "$numeric" "$dom" "$dir/$prob" "$out/suite-numeric/$name.txt"
done <<'EOF'
strips/gripper prob05.pddl
strips/blocks probBLOCKS-8-0.pddl
strips/logistics00 probLOGISTICS-6-1.pddl
strips/miconic s7-4.pddl
strips/visitall visitall_x-6_y-3_r-100.pddl
strips/sokoban-opt08-strips p14.pddl
strips/depot p02.pddl
strips/driverlog p03.pddl
strips/rovers p02.pddl
strips/zenotravel p05.pddl
strips/transport-opt08-strips p23.pddl
strips/freecell p02.pddl
strips/snake-opt18-strips p05.pddl
strips/parcprinter-opt11-strips p03.pddl
strips/pegsol-08-strips p22.pddl
adl/miconic-simpleadl s10-2.pddl
adl/caldera-split-opt18-adl p04.pddl
adl/pathways p02.pddl
adl/folding-opt23-adl p01.pddl
adl/openstacks-opt08-adl p03.pddl
adl/philosophers p03-phil4.pddl patched
strips/organic-synthesis-opt18-strips p20.pddl
EOF

echo "numeric -> $out/numeric"
csn="$work/mimir-cs/Benchmark/numeric"
md="$work/mimir/data"
for d in block-grouping counters delivery drone expedition ext-plant-watering farmland hydropower sailing tpp; do
    export_one "$numeric" "$csn/$d/domain.pddl" "$csn/$d/pfile1.pddl" "$out/numeric/cs-$d.txt"
done
for d in fo-counters refuel refuel-adl tpp/numeric zenotravel/numeric woodworking barman transport; do
    export_one "$numeric" "$md/$d/domain.pddl" "$md/$d/test_problem.pddl" "$out/numeric/m-${d//\//-}.txt"
done
export_one "$numeric" "$csn/block-grouping/domain.pddl" "$repo/tests/data/numeric_tasks/pddl/block-grouping-conj.pddl" \
    "$out/numeric/cs-block-grouping-conj.txt"
export_one "$numeric" "$csn/delivery/domain.pddl" "$repo/tests/data/numeric_tasks/pddl/delivery-nometric.pddl" \
    "$out/numeric/cs-delivery-nometric.txt"

echo "ipc -> $out/ipc"
if [[ -z "$fork_data" ]]; then
    echo "  skip (set MYMYR_FORK_DATA or --fork-data)"
fi
for dir in "$fork_data"/ipc/*; do
    [[ -d "$dir" ]] || continue
    domain="$(basename "$dir")"
    domain="${domain%-ipc}"
    for sample in test/p01-easy test/p02-easy test/p03-easy test/p01-medium train/p01 train/p02; do
        split="${sample%%/*}"
        prob="${sample#*/}"
        export_one "$numeric" "$dir/$split/domain.pddl" "$dir/$split/$prob.pddl" "$out/ipc/${domain}__${split}__${prob}.txt"
    done
done

echo "wrote $count exports to $out ($failures failed)"
[[ $failures -eq 0 ]]
