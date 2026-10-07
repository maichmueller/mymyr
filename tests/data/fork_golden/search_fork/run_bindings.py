#!/usr/bin/env python3
"""The fork's binding generators along seeded walks (search_fork --algo walk_ground) on the BrFS suite (export_all.py's
BFS_INSTANCES) and the numeric tasks (run_numeric.py), written to tests/data/bindings/fork_bindings.json for
tests/cpp/successor/test_bindings_fork.cpp.

    python3 tests/data/fork_golden/search_fork/run_bindings.py --build     # build search_fork, then run
    python3 tests/data/fork_golden/search_fork/run_bindings.py --only gripper,cs-counters   # these tasks again

Per walk step: the fluent atom count and set hash of the state, the action taken, and the groundings (count, set hash
of the binding strings, ground literals per kind) of the goal literals as a ConjunctiveCondition and, per action schema,
of its precondition (ConjunctiveConditionSatisficingBindingGenerator) and of the action itself
(ActionSatisficingBindingGenerator). A task the fork cannot run within the budget is recorded under "killed". Paths as
in ../export_all.py (MYMYR_WORK).
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
OUT = REPO / "tests" / "data" / "bindings" / "fork_bindings.json"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import export_all  # noqa: E402
import run_numeric  # noqa: E402

WALKS, STEPS, SEED = 2, 15, 1
# organic-synthesis: schemas with very many parameters, beyond the fork's lifted generators in the budget
SKIP = {"organic-synthesis-opt18-strips__p20"}


def tasks(work):
    out = {ins["name"].replace("/", "__"): (ins["domain"], ins["problem"]) for ins in export_all.BFS_INSTANCES}
    out.update(run_numeric.tasks(work))
    return {k: v for k, v in out.items() if k not in SKIP}


def limit_memory(gb):
    def f():
        b = int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return f


def run_task(exe, name, domain, problem, a):
    cmd = [str(exe), "--algo", "walk_ground", "--domain", str(domain), "--problem", str(problem),
           "--walks", str(WALKS), "--steps", str(STEPS), "--seed", str(SEED)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.seconds, preexec_fn=limit_memory(a.mem_gb))
    except subprocess.TimeoutExpired:
        return name, {"killed": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r.pop("seconds", None)
            r.pop("algo", None)
            print(f"{name:40s} {len(r['schemas'])} schemas", file=sys.stderr)
            return name, r
    why = "error: " + (p.stdout + p.stderr).strip()[-300:]
    print(f"{name:40s} {why}", file=sys.stderr)
    return name, {"killed": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    ap.add_argument("--build", action="store_true", help="configure and build search_fork first")
    ap.add_argument("--build-dir", default=None, help="default: <work>/build-search-fork")
    ap.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--mem-gb", type=float, default=4)
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
    only = a.only.split(",")
    todo = [(n, d, p) for n, (d, p) in sorted(tasks(work).items()) if any(o in n for o in only)]
    results = json.loads(OUT.read_text())["tasks"] if OUT.exists() else {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, rec in ex.map(lambda t: run_task(exe, *t, a), todo):
            results[name] = rec
    doc = {"format": "mymyr-fork-bindings/1",
           "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off",
                    "generators": "ConjunctiveConditionSatisficingBindingGenerator (goal literals, schema "
                                  "preconditions), ActionSatisficingBindingGenerator; create_ground_conjunction_generator"},
           "walks": {"num_walks": WALKS, "num_steps": STEPS, "seed_base": SEED},
           "tasks": dict(sorted(results.items()))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{len(results)} tasks -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
