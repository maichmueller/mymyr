#!/usr/bin/env python3
"""The fork's set-additive, h² and perfect heuristics on the walk states of the BrFS suite (export_all.py's
BFS_INSTANCES), and its A* with the perfect heuristic, written to tests/data/heuristics/fork_heuristics.json for
tests/cpp/heuristics/test_heuristics_golden.cpp.

    python3 tests/data/fork_golden/search_fork/run_heuristics.py --build     # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_heuristics.py --only blocks

The walks are those of the golden files (tests/data/expected: three seeded walks of 25 steps, README.md), regenerated
by search_fork --algo walk_h; every step carries the fluent atom count and set hash, so that a reader can check that it
replays the same states. A heuristic the fork cannot evaluate within the budget (time, memory, a state space of
--max-states states or more for perfect) is recorded under "killed". Paths as in ../export_all.py (MYMYR_WORK).
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import resource
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "tests" / "data" / "heuristics" / "fork_heuristics.json"
sys.path.insert(0, str(HERE.parent))
import export_all  # noqa: E402

HEURISTICS = ["setadd", "h2", "perfect"]


def tasks():
    return {ins["name"].replace("/", "__"): (ins["domain"], ins["problem"]) for ins in export_all.BFS_INSTANCES}


def limit_memory(gb):
    def f():
        b = int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return f


def run(exe, args, a):
    try:
        p = subprocess.run([str(exe)] + args, capture_output=True, text=True, timeout=a.seconds,
                           preexec_fn=limit_memory(a.mem_gb))
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r.pop("seconds", None)
            return r, None
    return None, "error: " + (p.stdout + p.stderr).strip()[-300:]


def run_task(exe, name, domain, problem, a):
    rec = {"walks": None, "h": {}, "killed": {}}
    common = ["--domain", str(domain), "--problem", str(problem), "--max-states", str(a.max_states)]
    for h in HEURISTICS:
        r, why = run(exe, ["--algo", "walk_h", "--h", h] + common, a)
        if r is None:
            rec["killed"][h] = why
            continue
        walks = [[{"atoms": s["atoms"], "hash": s["hash"]} for s in w["steps"]] for w in r["walks"]]
        if rec["walks"] is None:
            rec["walks"] = walks
        elif rec["walks"] != walks:
            rec["killed"][h] = "the walk differs between runs"
            continue
        rec["h"][h] = [[s["h"] for s in w["steps"]] for w in r["walks"]]
    if "perfect" in rec["h"]:
        r, why = run(exe, ["--algo", "astar_eager", "--h", "perfect", "--max-ms", str(a.seconds * 1000)] + common, a)
        if r is None:
            rec["killed"]["astar_perfect"] = why
        else:
            rec["astar_perfect"] = {k: r[k] for k in ("status", "plan_cost", "plan_length", "expanded", "generated")}
    print(f"{name:40s} {sorted(rec['h'])} killed={rec['killed']}", file=sys.stderr)
    return name, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--build", action="store_true", help="configure and build search_fork first")
    ap.add_argument("--build-dir", default=None, help="default: <work>/build-search-fork")
    ap.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--max-states", type=int, default=1_500_000)
    ap.add_argument("--mem-gb", type=float, default=6)
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    build = pathlib.Path(a.build_dir) if a.build_dir else work / "build-search-fork"
    if a.build:
        cfg = ["cmake", "-S", str(HERE), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
               f"-DCMAKE_PREFIX_PATH={work / 'install-fork'};{work / 'install-fork-deps'}"]
        if a.fork_cxx:
            cfg.append(f"-DCMAKE_CXX_COMPILER={a.fork_cxx}")
        subprocess.run(cfg, check=True)
        subprocess.run(["cmake", "--build", str(build), "-j", "8"], check=True)
    exe = build / "search_fork"
    todo = [(n, d, p) for n, (d, p) in sorted(tasks().items()) if a.only in n]
    results = json.loads(OUT.read_text())["tasks"] if OUT.exists() else {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, rec in ex.map(lambda t: run_task(exe, *t, a), todo):
            results[name] = rec
    doc = {"format": "mymyr-fork-heuristics/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off",
                    "heuristics": "SetAddHeuristic and H2Heuristic over a LiftedGrounder (unit costs), "
                                  "PerfectHeuristic over the search context"},
           "walks": {"num_walks": 3, "num_steps": 25, "seed_base": 1},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "tasks": dict(sorted(results.items()))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{len(results)} tasks -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
