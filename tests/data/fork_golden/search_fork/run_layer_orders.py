#!/usr/bin/env python3
"""The fork's IW and BrFS results under its layer ordering strategies on the BrFS suite (export_all.py's
BFS_INSTANCES), written to tests/data/layer_orders/fork_layer_orders.json for tests/cpp/search/test_layer_orders_fork.cpp.

    python3 tests/data/fork_golden/search_fork/run_layer_orders.py --build     # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_layer_orders.py --only blocks

Per task: IW(2) (the ladder of arities 0, 1, 2) and BrFS with stop_if_goal, each with the in-order, reverse and
goal-count orderings (more and fewer satisfied goal literals first), and with truncated next layers. Randomized
orderings are not compared (the fork shuffles with std::mt19937_64 and std::shuffle, mymyr with SplitMix64). The
tasks whose successor order differs between the fork and mymyr (test_search.py's ORDER_ONLY) and the largest ones
are left out. Paths as in ../export_all.py (MYMYR_WORK).
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
OUT = REPO / "tests" / "data" / "layer_orders" / "fork_layer_orders.json"
sys.path.insert(0, str(HERE.parent))
import export_all  # noqa: E402

LEAVE_OUT = ("philosophers", "pegsol", "organic-synthesis", "folding", "transport-opt08", "sokoban", "snake",
             "freecell", "visitall", "parcprinter", "caldera")
ORDERS = ["in_order", "reverse", "goal_count", "goal_count_fewer"]
RUNS = ([("iw", o, 2, None) for o in ORDERS] + [("iw", "reverse", 2, 7), ("iw", "goal_count", 2, 3)]
        + [("brfs", o, 0, None) for o in ORDERS] + [("brfs", "goal_count", 0, 5), ("brfs", "reverse", 0, 40)])


def tasks():
    out = {}
    for ins in export_all.BFS_INSTANCES:
        name = ins["name"].replace("/", "__")
        if not name.startswith(LEAVE_OUT):
            out[name] = (ins["domain"], ins["problem"])
    return out


def limit_memory(gb):
    def f():
        b = int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return f


def run_one(exe, name, domain, problem, algo, order, k, limit, a):
    cmd = [str(exe), "--algo", algo, "--order", order, "--k", str(k), "--domain", str(domain), "--problem",
           str(problem), "--max-ms", str(a.seconds * 1000), "--max-states", str(a.max_states)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    rec = {"task": name, "algo": algo, "order": order, "k": k, "limit": limit}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.seconds + 120,
                           preexec_fn=limit_memory(a.mem_gb))
    except subprocess.TimeoutExpired:
        rec["killed"] = "timeout"
        return rec
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r.pop("seconds", None)
            rec.update(r)
            return rec
    rec["killed"] = "error: " + (p.stdout + p.stderr).strip()[-300:]
    return rec


def key(r):
    return (r["task"], r["algo"], r["order"], r["k"], -1 if r["limit"] is None else r["limit"])


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
        cfg = ["cmake", "-S", str(HERE), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
               f"-DCMAKE_PREFIX_PATH={work / 'install-fork'};{work / 'install-fork-deps'}"]
        if a.fork_cxx:
            cfg.append(f"-DCMAKE_CXX_COMPILER={a.fork_cxx}")
        subprocess.run(cfg, check=True)
        subprocess.run(["cmake", "--build", str(build), "-j", "8"], check=True)
    exe = build / "search_fork"
    todo = [(n, d, p, *run) for n, (d, p) in sorted(tasks().items()) if a.only in n for run in RUNS]
    results = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    redo = {key({"task": t[0], "algo": t[3], "order": t[4], "k": t[5], "limit": t[6]}) for t in todo}
    keep = [r for r in results if key(r) not in redo]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        new = list(ex.map(lambda t: run_one(exe, *t, a), todo))
    for r in new:
        print(f"{r['task']:36s} {r['algo']:5s} {r['order']:17s} limit={r['limit']} {r.get('status', r.get('killed'))} "
              f"len={r.get('plan_length')} exp={r.get('expanded')}", file=sys.stderr)
    runs = sorted(keep + new, key=key)
    doc = {"format": "mymyr-fork-layer-orders/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(runs)} runs -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
