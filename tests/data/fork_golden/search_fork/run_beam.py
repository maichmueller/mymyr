#!/usr/bin/env python3
"""The fork's beam IW and beam BrFS on the BrFS suite (export_all.py's BFS_INSTANCES), written to
tests/data/beam/fork_beam.json for tests/cpp/search/test_beam_fork.cpp.

    python3 tests/data/fork_golden/search_fork/run_beam.py --build     # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_beam.py --only blocks

Per task: IW(2) (the ladder of arities 0, 1, 2) and BrFS with stop_if_goal, each with the goal-count ordering (more
and fewer satisfied goal literals first), beam widths 1, 4 and 32, and both beam novelty modes (ALL_TESTED,
SURVIVORS_ONLY). The tasks whose successor order differs between the fork and mymyr (test_search.py's ORDER_ONLY)
are left out. Paths as in ../export_all.py (MYMYR_WORK).
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "tests" / "data" / "beam" / "fork_beam.json"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import export_all  # noqa: E402
import run_layer_orders  # noqa: E402

LEAVE_OUT = ("philosophers", "pegsol", "organic-synthesis")
ORDERS = ["goal_count", "goal_count_fewer"]
WIDTHS = [1, 4, 32]
MODES = ["all_tested", "survivors_only"]
RUNS = [(algo, k, order, w, mode) for algo, k in (("iw", 2), ("brfs", 0)) for order in ORDERS for w in WIDTHS
        for mode in MODES]


def tasks():
    out = {}
    for ins in export_all.BFS_INSTANCES:
        name = ins["name"].replace("/", "__")
        if not name.startswith(LEAVE_OUT):
            out[name] = (ins["domain"], ins["problem"])
    return out


def run_one(exe, name, domain, problem, algo, k, order, width, mode, a):
    rec = run_layer_orders.run_one(exe, name, domain, problem, algo, order, k, None, a,
                                   extra=["--beam", str(width), "--beam-mode", mode])
    rec["beam"] = width
    rec["beam_mode"] = mode
    return rec


def key(r):
    return (r["task"], r["algo"], r["order"], r["beam"], r["beam_mode"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--build", action="store_true", help="configure and build search_fork first")
    ap.add_argument("--build-dir", default=None, help="default: <work>/build-search-fork")
    ap.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--max-states", type=int, default=1_000_000)
    ap.add_argument("--mem-gb", type=float, default=3)
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    build = pathlib.Path(a.build_dir) if a.build_dir else work / "build-search-fork"
    if a.build:
        run_layer_orders.build_search_fork(work, build, a.fork_cxx)
    exe = build / "search_fork"
    todo = [(n, d, p, *run) for n, (d, p) in sorted(tasks().items()) if a.only in n for run in RUNS]
    results = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    redo = {key({"task": t[0], "algo": t[3], "order": t[5], "beam": t[6], "beam_mode": t[7]}) for t in todo}
    keep = [r for r in results if key(r) not in redo]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        new = list(ex.map(lambda t: run_one(exe, *t, a), todo))
    for r in new:
        print(f"{r['task']:36s} {r['algo']:5s} {r['order']:17s} beam={r['beam']:<3d} {r['beam_mode']:15s} "
              f"{r.get('status', r.get('killed'))} len={r.get('plan_length')} exp={r.get('expanded')}", file=sys.stderr)
    runs = sorted(keep + new, key=key)
    doc = {"format": "mymyr-fork-beam/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(runs)} runs -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
