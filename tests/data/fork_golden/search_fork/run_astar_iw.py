#!/usr/bin/env python3
"""Generate AStarIW fork results on every BrFS suite instance.

Classical widths 1/2 and abstracted width 1 each use blind and h_max. Action-order differences and unsupported
cost models are retained in the JSON so comparison exclusions remain visible.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "tests/data/astar_iw/fork_astar_iw.json"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import export_all
import run_layer_orders

RUNS = [(h, k, f) for h in ("blind", "hmax") for k, f in ((1, "classical"), (2, "classical"), (1, "abstracted"))]


def run_one(exe, ins, h, width, features, args):
    record = {"task": ins["name"].replace("/", "__"), "algo": "astar_iw", "h": h, "width": width, "features": features}
    command = [str(exe), "--algo", "astar_iw", "--h", h, "--width", str(width), "--features", features,
               "--domain", str(ins["domain"]), "--problem", str(ins["problem"]),
               "--max-ms", str(args.seconds * 1000), "--max-states", str(args.max_states)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds + 120,
                                preexec_fn=run_layer_orders.limit_memory(args.mem_gb))
    except subprocess.TimeoutExpired:
        record["killed"] = "timeout"
        return record
    for line in result.stdout.splitlines():
        if line.startswith("RESULT "):
            data = json.loads(line[7:])
            data.pop("seconds", None)
            record.update(data)
            return record
    record["killed"] = "error: " + (result.stdout + result.stderr).strip()[-500:]
    return record


def key(record):
    return record["task"], record["h"], record["features"], record["width"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.environ.get("MYMYR_WORK", str(REPO / ".work")))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--fork-cxx", default=os.environ.get("FORK_CXX", ""))
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--mem-gb", type=float, default=3)
    args = parser.parse_args()
    work = pathlib.Path(args.work)
    build = pathlib.Path(args.build_dir) if args.build_dir else work / "build-search-fork"
    if args.build:
        run_layer_orders.build_search_fork(work, build, args.fork_cxx)
    todo = [(ins, *run) for ins in export_all.BFS_INSTANCES if args.only in ins["name"] for run in RUNS]
    old = json.loads(OUT.read_text())["runs"] if OUT.exists() else []
    redo = {(ins["name"].replace("/", "__"), h, f, k) for ins, h, k, f in todo}
    keep = [r for r in old if key(r) not in redo]
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda item: run_one(build / "search_fork", *item, args), todo))
    for r in results:
        print(f"{r['task']:40s} {r['h']:5s} {r['features']:10s} width={r['width']} "
              f"{r.get('status', r.get('killed'))} len={r.get('plan_length')} exp={r.get('expanded')}", file=sys.stderr)
    runs = sorted(keep + results, key=key)
    document = {"format": "mymyr-fork-astar-iw/1",
                "fork": {"version": "0.16.3", "successor_generator": "lifted KPKC, symmetry pruning off"},
                "budget": {"seconds": args.seconds, "max_states": args.max_states, "mem_gb": args.mem_gb}, "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"{len(runs)} runs -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
