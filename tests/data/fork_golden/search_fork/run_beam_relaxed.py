#!/usr/bin/env python3
"""The fork's relaxed SurvivorsOnly beam (relaxed_survivors_only_beam) on the beam suite of run_beam.py, written to
tests/data/beam/fork_beam_relaxed.json for tests/cpp/search/test_beam_fork.cpp.

    python3 tests/data/fork_golden/search_fork/run_beam_relaxed.py --build-dir DIR   # search_fork built by run_beam.py
    python3 tests/data/fork_golden/search_fork/run_beam_relaxed.py --only blocks

The fork has two relaxed selections. By default (lifted KPKC) it selects per expanded state: the state's action
schemas are split among the threads, each keeps its best beam_width successors, and the merged ones are interned
until the layer holds beam_width states, after which every further expanded state still adds one; a layer can thus
hold up to beam_width + (expanded states - 1) states. With iw1_atom_first_mode and the IW(1) add-effect precheck the
IW(1) pass selects per layer instead (parallel_beam_chunk_size parts of the layer's transitions), the selection
mymyr implements (search/layer_ordering.hpp, RelaxedSurvivorsOnly). Per task, goal-count order (more and fewer
satisfied goal literals first) and beam width 1, 4 and 32 this records IW(2) with --iw1-knobs atom_first and the
relaxed beam on 2, 4 and 8 threads (its arity-1 pass is the per-layer selection), and, as a control, the exact
SurvivorsOnly beam on one thread with and without --iw1-knobs atom_first: where both give the same arity-1 pass,
the knobs do not change that pass's search, and its relaxed pass is comparable.
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "tests" / "data" / "beam" / "fork_beam_relaxed.json"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import run_beam  # noqa: E402
import run_layer_orders  # noqa: E402

ORDERS = ["goal_count", "goal_count_fewer"]
WIDTHS = [1, 4, 32]
THREADS = [2, 4, 8]


def passes(exe, name, domain, problem, order, width, mode, threads, knobs, a):
    extra = ["--beam", str(width), "--beam-mode", mode, "--threads", str(threads)]
    if knobs:
        extra += ["--iw1-knobs", knobs]
    rec = run_layer_orders.run_one(exe, name, domain, problem, "iw", order, 2, None, a, extra=extra)
    if "killed" in rec or rec.get("status") not in ("solved", "failed", "exhausted"):
        return None
    return {"status": rec["status"], "passes": [[p["expanded"], p["generated_in_tree"]] for p in rec["passes"]]}


def run_one(exe, name, domain, problem, order, width, a):
    rec = {"task": name, "order": order, "beam": width,
           "survivors_only": passes(exe, name, domain, problem, order, width, "survivors_only", 1, "", a),
           "survivors_only_atom_first": passes(exe, name, domain, problem, order, width, "survivors_only", 1,
                                               "atom_first", a),
           "relaxed": {}}
    for t in THREADS:
        rec["relaxed"][str(t)] = passes(exe, name, domain, problem, order, width, "relaxed", t, "atom_first", a)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--build-dir", default=None, help="search_fork's build directory (default: <work>/build-search-fork)")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--max-states", type=int, default=1_000_000)
    ap.add_argument("--mem-gb", type=float, default=3)
    a = ap.parse_args()
    build = pathlib.Path(a.build_dir) if a.build_dir else pathlib.Path(a.work) / "build-search-fork"
    exe = build / "search_fork"
    todo = [(n, d, p, order, w) for n, (d, p) in sorted(run_beam.tasks().items()) if a.only in n for order in ORDERS
            for w in WIDTHS]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(lambda t: run_one(exe, *t, a), todo))
    doc = {"format": "mymyr-fork-beam-relaxed/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off",
                    "parallel_beam_chunk_size": 1024},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "runs": sorted(runs, key=lambda r: (r["task"], r["order"], r["beam"]))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(runs)} runs -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
