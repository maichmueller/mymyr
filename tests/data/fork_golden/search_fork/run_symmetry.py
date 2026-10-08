#!/usr/bin/env python3
"""The fork's WL1 symmetry pruning (search_fork --algo symmetry_states) on the tasks of the state-space parity suite
(tests/cpp/datasets/fork_cases.inc), written to tests/data/symmetry/fork_symmetry.json for
tests/cpp/successor/test_symmetry_fork.cpp and tests/python/test_symmetry.py.

    python3 tests/data/fork_golden/search_fork/run_symmetry.py --build      # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_symmetry.py --only gripper

Per task: the objects in the fork's order, and every state reachable without pruning with its fluent atoms (and fluent function values, "(f o1 ... ok)=v"), its
applicable actions under the KPKC generator's WL1 pruning, and the WL1 colour class of every object (atoms and
actions as indices into per-task name tables); then brfs (exhaustive, and stopping at a goal) and astar_eager with the
blind heuristic, both with WL1 pruning. The fork's data directory comes from MYMYR_FORK_DATA; the fork install from
MYMYR_WORK (as in run_layer_orders.py).
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "tests" / "data" / "symmetry" / "fork_symmetry.json"
CASES = REPO / "tests" / "cpp" / "datasets" / "fork_cases.inc"
sys.path.insert(0, str(HERE))
from run_layer_orders import build_search_fork, limit_memory  # noqa: E402


def tasks():
    """(directory, problem) of every case of fork_cases.inc, in its order."""
    return re.findall(r'^\s*\{"([^"]+)", "([^"]+)"', CASES.read_text(), re.M)


def run_one(exe, data, d, p, a):
    name = f"{d}/{p}"
    cmd = [str(exe), "--algo", "symmetry_states", "--domain", str(data / d / "domain.pddl"), "--problem", str(data / d / p),
           "--max-ms", str(a.seconds * 1000), "--max-states", str(a.max_states)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.seconds * 3 + 120, preexec_fn=limit_memory(a.mem_gb))
    except subprocess.TimeoutExpired:
        return {"task": name, "killed": "timeout"}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[7:])
            rec.pop("seconds", None)
            rec.pop("algo", None)
            return {"task": name, **rec}
    return {"task": name, "killed": "error: " + (r.stdout + r.stderr).strip()[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--data", default=os.environ.get("MYMYR_FORK_DATA", ""))
    ap.add_argument("--build", action="store_true", help="configure and build search_fork first")
    ap.add_argument("--build-dir", default=None, help="default: <work>/build-search-fork")
    ap.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--max-states", type=int, default=20_000)
    ap.add_argument("--mem-gb", type=float, default=3)
    a = ap.parse_args()
    if not a.data:
        sys.exit("run_symmetry.py: set MYMYR_FORK_DATA or --data to the fork's data directory")
    work = pathlib.Path(a.work)
    build = pathlib.Path(a.build_dir) if a.build_dir else work / "build-search-fork"
    if a.build:
        build_search_fork(work, build, a.fork_cxx)
    exe = build / "search_fork"
    todo = [(d, p) for d, p in tasks() if a.only in f"{d}/{p}"]
    results = json.loads(OUT.read_text())["tasks"] if OUT.exists() else []
    redo = {f"{d}/{p}" for d, p in todo}
    keep = [r for r in results if r["task"] not in redo]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        new = list(ex.map(lambda t: run_one(exe, pathlib.Path(a.data), *t, a), todo))
    for r in new:
        s = r.get("searches", {})
        print(f"{r['task']:44s} {r.get('killed', '')}states={r.get('num_states')} actions={r.get('num_actions')} "
              f"pruned={r.get('num_pruned_actions')} brfs={s.get('brfs_exhaustive', {}).get('expanded')} "
              f"astar={s.get('astar_blind', {}).get('plan_cost')}", file=sys.stderr)
    order = {f"{d}/{p}": i for i, (d, p) in enumerate(tasks())}
    runs = sorted(keep + new, key=lambda r: order.get(r["task"], len(order)))
    doc = {"format": "mymyr-fork-symmetry/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning WL1"},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "tasks": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{len(runs)} tasks -> {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
