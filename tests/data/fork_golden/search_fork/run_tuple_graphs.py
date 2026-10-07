#!/usr/bin/env python3
"""The fork's tuple graphs on the state-space suite (tests/cpp/datasets/fork_cases.inc: the fork's own test instances,
under MYMYR_FORK_DATA), written to tests/data/tuple_graphs/fork_tuple_graphs.json for
tests/cpp/datasets/test_tuple_graphs_fork.cpp and tests/python/test_tuple_graphs.py.

    python3 tests/data/fork_golden/search_fork/run_tuple_graphs.py --build      # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_tuple_graphs.py --only gripper
    python3 tests/data/fork_golden/search_fork/run_tuple_graphs.py --time blocks_4/p02-easy.pddl ...

Per task: the state space (remove_if_unsolvable = false, no symmetry pruning) and the tuple graphs of width 0, and of
widths 1 and 2 with and without dominance pruning, of every vertex (every ceil(N / sample)-th vertex on spaces of more
than --sample vertices). The format is described in tests/data/tuple_graphs/README.md.

--time runs only the timing mode on the given tasks (<dir>/<problem>): the fork's TupleGraphImpl::create over every
vertex at widths 1 and 2 with dominance pruning, its seconds and peak RSS, printed as JSON lines.
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
OUT = REPO / "tests" / "data" / "tuple_graphs" / "fork_tuple_graphs.json"
CASES = REPO / "tests" / "cpp" / "datasets" / "fork_cases.inc"
sys.path.insert(0, str(HERE))
import run_layer_orders  # noqa: E402


def suite():
    """(dir, problem) of every fork_cases.inc entry, in its order."""
    return re.findall(r'^\s*\{"([^"]+)", "([^"]+)"', CASES.read_text(), re.M)


def run(exe, data, task, extra, args):
    d, p = task
    command = [str(exe), "--algo", "tuple_graphs", "--domain", str(data / d / "domain.pddl"), "--problem", str(data / d / p),
               "--max-states", str(args.max_states)] + extra
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds,
                            preexec_fn=run_layer_orders.limit_memory(args.mem_gb))
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
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--mem-gb", type=float, default=4)
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
    if args.time is not None:
        for t in args.time:
            task = tuple(t.split("/", 1))
            for width in (1, 2):
                r = run(exe, data, task, ["--time-width", str(width), "--time-pruning", "1"], args)
                print(json.dumps({"task": t, **r}))
        return
    todo = [t for t in suite() if args.only in f"{t[0]}/{t[1]}"]
    def golden(task):
        r = run(exe, data, task, ["--sample", str(args.sample)], args)
        if "killed" in r:  # the fork crashes at width 2 on states without fluent atoms (deadend)
            r = run(exe, data, task, ["--sample", str(args.sample), "--max-width", "1"], args)
            r["fork_failed_widths"] = [2]
        return r

    with cf.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(golden, todo))
    old = json.loads(OUT.read_text())["tasks"] if OUT.exists() else []
    redo = {f"{d}/{p}" for d, p in todo}
    tasks = [r for r in old if r["task"] not in redo]
    for (d, p), r in zip(todo, results):
        r.pop("algo", None)
        tasks.append({"task": f"{d}/{p}", **r})
        print(f"{d}/{p:40s} {r.get('killed') or str(r['states']) + ' states, ' + str(len(r['roots'])) + ' roots'}", file=sys.stderr)
    order = {f"{d}/{p}": i for i, (d, p) in enumerate(suite())}
    tasks.sort(key=lambda r: order.get(r["task"], len(order)))
    document = {"format": "mymyr-fork-tuple-graphs/1",
                "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
                "state_space": {"remove_if_unsolvable": False, "symmetry_pruning": False, "max_states": args.max_states},
                "sample": args.sample, "tasks": tasks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(f"{len(tasks)} tasks -> {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
