"""CPU encoder targets on small, static-heavy, wide and derived-predicate tasks.

Run with the CPU Python environment and --binary pointing at mymyr_encoding_bench. --compare adds an interleaved
second executable; alternating the order each round limits drift on shared hosts. JSON lines report minima and
medians, load averages, hashes and target results. --reference accepts {task: {flat_us: ..., actions_us: ...}} measured
elsewhere; absent references leave the corresponding speedups null. --raw preserves every round's measurements.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "blocks3": ("pddl/blocks", "smedium"),
    "blocks8": ("pddl/blocks", "probBLOCKS-8-0"),
    "logistics6": ("pddl/logistics00", "probLOGISTICS-6-1"),
    "rovers-p05m": ("ipc/rovers-ipc/test", "p05-medium"),
    "childsnack-p05m": ("ipc/childsnack-ipc/test", "p05-medium"),
    "satellite-p05m": ("ipc/satellite-ipc/test", "p05-medium"),
    "philosophers-p03": ("pddl/philosophers", "p03-phil4"),
    "blocks-p01e": ("ipc/blocksworld-ipc/test", "p01-easy"),
    "childsnack-p16e": ("ipc/childsnack-ipc/test", "p16-easy"),
    "rovers-p16e": ("ipc/rovers-ipc/test", "p16-easy"),
    "blocks-p05m": ("ipc/blocksworld-ipc/test", "p05-medium"),
    "sokoban-p16e": ("ipc/sokoban-ipc/test", "p16-easy"),
    "transport-p30m": ("ipc/transport-ipc/test", "p30-medium"),
    "miconic-p30m": ("ipc/miconic-ipc/test", "p30-medium"),
    "blocks-p20m": ("ipc/blocksworld-ipc/test", "p20-medium"),
}


def task_files(name, data):
    directory, problem = TASKS[name]
    for root in (ROOT / "tests/data", data):
        base = root / directory
        if (base / f"{problem}.pddl").is_file():
            return base / "domain.pddl", base / f"{problem}.pddl"
    raise FileNotFoundError(f"no task files for {name}; supply --data with a complete benchmark data directory")


def python_single(domain, problem):
    import mymyr
    from mymyr.encoding import FlatRelationEncoder

    task = mymyr.Task.from_pddl(domain, problem)
    encoder = FlatRelationEncoder(task, goal_derivations=())
    state = task.initial_state
    encoder.encode(state)
    samples = []
    for _ in range(15):
        start = time.perf_counter_ns()
        for _ in range(128):
            encoder.encode(state)
        samples.append((time.perf_counter_ns() - start) / 128000)
    return {"min": min(samples), "median": statistics.median(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "tests/data")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--states", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--skip-python", action="store_true")
    args = parser.parse_args()
    if args.states < 1 or args.rounds < 1:
        parser.error("states and rounds must be positive")
    references = json.loads(args.reference.read_text()) if args.reference else {}
    binaries = {"primary": args.binary.resolve()}
    if args.compare:
        binaries["comparison"] = args.compare.resolve()
    raw = args.raw.open("w") if args.raw else None
    try:
        for name in args.tasks:
            domain, problem = task_files(name, args.data)
            samples = {key: [] for key in binaries}
            loads = []
            servers = {}
            try:
                for label, binary in binaries.items():
                    servers[label] = subprocess.Popen(
                        [str(binary), "--targets-server", str(domain), str(problem), str(args.states), str(args.seed)],
                        text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                for round_index in range(args.rounds):
                    order = list(servers.items())
                    if round_index % 2:
                        order.reverse()
                    for label, server in order:
                        loads.append(os.getloadavg())
                        server.stdin.write("run\n")
                        server.stdin.flush()
                        line = server.stdout.readline()
                        if not line:
                            raise RuntimeError(f"{name}/{label}: {server.stderr.read()}")
                        record = json.loads(line)
                        samples[label].append(record)
                        if raw:
                            raw.write(json.dumps(dict(task=name, binary=label, round=round_index, load=loads[-1], **record)) + "\n")
                            raw.flush()
            finally:
                for server in servers.values():
                    if server.poll() is None:
                        try:
                            server.stdin.write("quit\n")
                            server.stdin.flush()
                        except BrokenPipeError:
                            pass
                    server.wait()
                    server.stdin.close()
                    server.stdout.close()
                    server.stderr.close()
            single = None if args.skip_python else python_single(domain, problem)
            for label, values in samples.items():
                metrics = {}
                for key in values[0]:
                    column = [row[key] for row in values]
                    if isinstance(column[0], (int, float)):
                        metrics[key] = {"min": min(column), "median": statistics.median(column)}
                    else:
                        if any(value != column[0] for value in column):
                            raise RuntimeError(f"inconsistent {key} for {name}/{label}")
                        metrics[key] = column[0]
                best = lambda key: metrics[key]["min"]
                reference = references.get(name, {})
                speedup = reference.get("flat_us", 0) / best("core_us") if reference.get("flat_us") else None
                action_speedup = reference.get("actions_us", 0) / best("actions_us") if reference.get("actions_us") else None
                targets = {
                    "core": best("core_us") <= best("core_limit_us"),
                    "actions": None if metrics["action_ns_per_int"] is None else best("action_ns_per_int") <= 1,
                    "derived_reuse": None if best("derived_words") == 0 else best("reused_derived_us") <= best("core_limit_us"),
                    "single_cpp": best("single_cpp_us") <= 0.5,
                    "single_python": None if single is None else single["min"] <= 3,
                    "threads": None if args.states < 4096 else best("eight_threads_speedup") >= 3.5,
                    "reference": None if speedup is None else speedup >= 20,
                    "reference_actions": None if action_speedup is None else action_speedup >= 20,
                    "relation": best("relation_over_graph") <= 1.2,
                    "hgraph": best("hgraph_with_encode_ratio") <= 1.1,
                    "simple": best("simple_over_core") <= 2.5,
                }
                print(json.dumps(dict(task=name, binary=label, rounds=args.rounds, load_min=min(x[0] for x in loads),
                                      load_max=max(x[0] for x in loads), metrics=metrics, single_python_us=single,
                                      reference_speedup=speedup, reference_actions_speedup=action_speedup,
                                      targets=targets)), flush=True)
            if len(samples) == 2 and samples["primary"][0]["hash"] != samples["comparison"][0]["hash"]:
                raise RuntimeError(f"executables produced different hashes for {name}")
    finally:
        if raw:
            raw.close()


if __name__ == "__main__":
    main()
