#!/usr/bin/env python3
"""The fork's best-first results on the numeric tasks of tests/data/numeric_tasks (the PDDL files those text tasks were
exported from, tests/data/make_golden.sh), written to tests/data/numeric_tasks/fork_best_first.json.

    python3 tests/data/fork_golden/search_fork/run_numeric.py --build        # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_numeric.py --only tpp     # a subset (merged into the file)

Every run is one fork process (A* eager / lazy with blind and h_max, GBFS eager / lazy with h_FF and h_add) with a
state budget, a time budget and a memory limit. Paths: --work (default $MYMYR_WORK or <repo>/.work) holds
install-fork, install-fork-deps (../build_fork.sh), mimir-cs/ and mimir/ (the benchmark clones of make_golden.sh).
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
OUT = REPO / "tests" / "data" / "numeric_tasks" / "fork_best_first.json"

RUNS = [("astar_eager", "blind"), ("astar_eager", "max"), ("astar_lazy", "blind"), ("astar_lazy", "max"),
        ("gbfs_eager", "ff"), ("gbfs_lazy", "ff"), ("gbfs_eager", "add")]


def tasks(work: pathlib.Path):
    csn = work / "mimir-cs" / "Benchmark" / "numeric"
    md = work / "mimir" / "data"
    out = {}
    for d in ["block-grouping", "counters", "delivery", "drone", "expedition", "ext-plant-watering", "farmland",
              "hydropower", "sailing", "tpp"]:
        out[f"cs-{d}"] = (csn / d / "domain.pddl", csn / d / "pfile1.pddl")
    for d in ["fo-counters", "refuel", "refuel-adl", "tpp/numeric", "zenotravel/numeric", "woodworking", "barman",
              "transport"]:
        out["m-" + d.replace("/", "-")] = (md / d / "domain.pddl", md / d / "test_problem.pddl")
    out["cs-block-grouping-conj"] = (csn / "block-grouping" / "domain.pddl",
                                     REPO / "tests/data/numeric_tasks/pddl/block-grouping-conj.pddl")
    out["cs-delivery-nometric"] = (csn / "delivery" / "domain.pddl",
                                   REPO / "tests/data/numeric_tasks/pddl/delivery-nometric.pddl")
    return out


def limit_memory(gb):
    def f():
        b = int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return f


def run_one(exe, name, domain, problem, algo, h, a):
    cmd = [str(exe), "--algo", algo, "--h", h, "--domain", str(domain), "--problem", str(problem),
           "--max-ms", str(a.seconds * 1000), "--max-states", str(a.max_states)]
    rec = {"task": name, "algo": algo, "h": h}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.seconds + 120,
                           preexec_fn=limit_memory(a.mem_gb))
    except subprocess.TimeoutExpired:
        rec["killed"] = "timeout"
        return rec
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[7:]))
            rec.pop("seconds", None)
            return rec
        if line.startswith("ERROR "):
            rec["killed"] = "error: " + json.loads(line[6:])
            return rec
    rec["killed"] = f"exit code {p.returncode}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--build", action="store_true", help="configure and build search_fork first")
    ap.add_argument("--build-dir", default=None, help="default: <work>/build-search-fork")
    ap.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--max-states", type=int, default=2_000_000)
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
    todo = [(n, d, p, algo, h) for n, (d, p) in sorted(tasks(work).items()) if a.only in n for algo, h in RUNS]
    results = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    keep = [r for r in results if not any(r["task"] == n and r["algo"] == algo and r["h"] == h
                                          for n, _, _, algo, h in todo)]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        new = list(ex.map(lambda t: run_one(exe, *t, a), todo))
    for r in new:
        print(f"{r['task']:26s} {r['algo']:12s} {r['h']:6s} {r.get('status', r.get('killed'))} "
              f"cost={r.get('plan_cost')} len={r.get('plan_length')} exp={r.get('expanded')}", file=sys.stderr)
    runs = sorted(keep + new, key=lambda r: (r["task"], r["algo"], r["h"]))
    doc = {"format": "mymyr-fork-best-first/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
           "budget": {"seconds": a.seconds, "max_states": a.max_states, "mem_gb": a.mem_gb},
           "runs": runs}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(runs)} runs -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
