#!/usr/bin/env python3
"""The fork's k-FWL and nauty classes on the state-space suite (tests/cpp/datasets/fork_cases.inc: the fork's own
test instances, under MYMYR_FORK_DATA), written to tests/data/kfwl/fork_kfwl.json for
tests/cpp/datasets/test_kfwl.cpp and tests/python/test_kfwl.py.

    python3 tests/data/fork_golden/search_fork/run_kfwl.py --build      # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_kfwl.py --only gripper
    python3 tests/data/fork_golden/search_fork/run_kfwl.py --time gripper/p-2-0.pddl ...

Per task: the state space (remove_if_unsolvable = false, no symmetry pruning), and for every ceil(N / sample)-th
state its key, the number of vertices n of its object graph, its class among the sampled states by nauty canonical
form and by the fork's k-FWL certificate for k = 2, 3, 4 (k-FWL only on object graphs of at most --max-n2/3/4
vertices: the fork's 4-FWL needs 3.6 GB at n = 24), and the number of states
of the fork's symmetry-reduced state space (nauty). The format is described in tests/data/fork_golden/README.md
("k-FWL certificates").

--time prints, per given task (<dir>/<problem>), the fork's seconds per sampled state and peak RSS for each k as JSON
lines (each k in its own process, so that the peak RSS is that k's).
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
OUT = REPO / "tests" / "data" / "kfwl" / "fork_kfwl.json"
CASES = REPO / "tests" / "cpp" / "datasets" / "fork_cases.inc"
sys.path.insert(0, str(HERE))
import run_layer_orders  # noqa: E402


def suite():
    """(dir, problem) of every fork_cases.inc entry, in its order."""
    return re.findall(r'^\s*\{"([^"]+)", "([^"]+)"', CASES.read_text(), re.M)


def run(exe, data, task, extra, args):
    d, p = task
    command = [str(exe), "--algo", "kfwl", "--domain", str(data / d / "domain.pddl"), "--problem", str(data / d / p),
               "--max-states", str(args.max_states), "--sample", str(args.sample)] + extra
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds,
                                preexec_fn=run_layer_orders.limit_memory(args.mem_gb))
    except subprocess.TimeoutExpired:
        return {"killed": "timeout"}
    for line in result.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"killed": "error: " + (result.stdout + result.stderr).strip()[-500:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    parser.add_argument("--data", default=os.environ.get("MYMYR_FORK_DATA", ""))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--mem-gb", type=float, default=4)
    parser.add_argument("--max-n2", type=int, default=200)
    parser.add_argument("--max-n3", type=int, default=64)
    parser.add_argument("--max-n4", type=int, default=24)
    parser.add_argument("--time", nargs="*", help="timing mode on these <dir>/<problem> tasks")
    args = parser.parse_args()
    if not args.data:
        sys.exit("set MYMYR_FORK_DATA or --data to the fork's data directory")
    data = pathlib.Path(args.data)
    work = pathlib.Path(args.work)
    build = pathlib.Path(args.build_dir) if args.build_dir else work / "build-search-fork"
    if args.build:
        run_layer_orders.build_search_fork(work, build, args.fork_cxx)
    exe = build / "search_fork"
    caps = {2: args.max_n2, 3: args.max_n3, 4: args.max_n4}
    if args.time is not None:
        for t in args.time:
            task = tuple(t.split("/", 1))
            for k in (2, 3, 4):
                extra = []
                for j in (2, 3, 4):
                    extra += [f"--max-n{j}", str(caps[j] if j == k else 0)]
                r = run(exe, data, task, extra, args)
                keep = {key: r[key] for key in ("states", "n", f"k{k}_seconds", f"k{k}_state_seconds", f"k{k}_peak_rss_kb", "killed")
                        if key in r}
                print(json.dumps({"task": t, "k": k, **keep}))
        return
    todo = [t for t in suite() if args.only in f"{t[0]}/{t[1]}"]
    extra = [x for k in (2, 3, 4) for x in (f"--max-n{k}", str(caps[k]))]

    def golden(task):
        r = run(exe, data, task, extra, args)
        r.pop("algo", None)
        meta = {}
        for key in list(r):
            if key.endswith("_seconds") or key.endswith("_peak_rss_kb"):
                value = r.pop(key)
                if not key.endswith("_state_seconds"):
                    meta[key] = value
        if meta:
            r["meta"] = meta
        return r

    with cf.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(golden, todo))
    old = json.loads(OUT.read_text())["tasks"] if OUT.exists() else []
    redo = {f"{d}/{p}" for d, p in todo}
    tasks = [r for r in old if r["task"] not in redo]
    for (d, p), r in zip(todo, results):
        tasks.append({"task": f"{d}/{p}", **r})
        print(f"{d}/{p:40s} {r.get('killed') or str(r['states']) + ' states, ' + str(len(r['keys'])) + ' sampled'}", file=sys.stderr)
    order = {f"{d}/{p}": i for i, (d, p) in enumerate(suite())}
    tasks.sort(key=lambda r: order.get(r["task"], len(order)))
    document = {"format": "mymyr-fork-kfwl/1",
                "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
                "state_space": {"remove_if_unsolvable": False, "symmetry_pruning": False, "max_states": args.max_states},
                "sample": args.sample, "tasks": tasks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(f"{len(tasks)} tasks -> {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
