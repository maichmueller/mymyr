"""The cost of calling the device encoder from Python.

For each task the script generates reachable rows from seeded random walks, tiles them to N rows on the device and times,
with interleaved rounds (every round runs every mode once), the ways of encoding them:

  - encode_exact: `encoder.encode(rows)`, which counts the batch, waits for its sizes once, allocates exact arrays and
    fills them.
  - encode_capacities: `encoder.encode(rows, capacities=...)`, which allocates the arrays at the capacities and never
    waits (the synchronize after it is the timing's).
  - op_eager: `torch.ops.mymyr.encode_flat` called eagerly, with the same capacities.
  - op_dynamo_eager: the op inside `torch.compile(fullgraph=True, backend="eager")`. This measures Dynamo's overhead;
    the eager backend executes the same opaque op and does not compile its implementation.
  - op_graph: the compiled measurement: the op captured in a CUDA graph and replayed (REPLAYS replays between events).
  - cpu_exact / cpu_single: the host encoder on the same rows, one call over all rows and one call of a single row.

All times are microseconds per encoded state (per call for cpu_single); the table prints the minimum and the median over
the rounds. The C++ device encoding bench (bench/cuda/device_encoding_bench.cpp) times the same kernels without Python;
its eager and graph numbers for the same N are the floor of these.

    env CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python bench/python/bench_encoding_device_calls.py [--tasks blocks-p01e,blocks-8]
        [--n 16384] [--rounds 30] [--json PATH]
"""

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

import mymyr.rl.torch as rt
from mymyr import encoding

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from bench_encoding_gpu import REPLAYS, load_task, make_encoder, padded_capacities, tile, walk_rows  # noqa: E402


def capacities_of(encoder, rows):
    batch = encoder.encode(rows)
    return padded_capacities(
        dict(graphs=batch.num_graphs, entities=int(batch.entity_offsets[-1]),
             args=int(batch.relation_arg_offsets[-1]))
    )


class Run:
    """One timed workload: `call()` returns its own time in microseconds, or None to be timed by the host clock."""

    def __init__(self, name, call, units, host=True):
        self.name, self.call, self.units, self.host, self.times = name, call, units, host, []

    def once(self):
        if self.host:
            start = time.perf_counter()
            self.call()
            return (time.perf_counter() - start) * 1e6 / self.units
        return self.call() / self.units


def bench_task(key, n, rounds, data):
    task, reason = load_task(key, data)
    if task is None:
        return dict(task=key, skipped=reason)
    encoder, reason = make_encoder(task)
    if encoder is None:
        return dict(task=key, skipped=reason)
    rows = tile(walk_rows(task, 4096, seed=1), n)
    device = torch.device("cuda", 0)
    drows = torch.from_numpy(rows.view(np.int64)).to(device)
    caps = capacities_of(encoder, rows)
    handle = rt.register(encoder)

    def op(states):
        return torch.ops.mymyr.encode_flat(handle, states, None, None, None, None, None, None, *([None] * 10),
                                           caps["graphs"], caps["entities"], caps["args"], 0, 0)

    torch._dynamo.reset()
    compiled = torch.compile(lambda states: op(states), fullgraph=True, backend="eager")
    stream = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        op(drows)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        op(drows)

    def sync(f):
        def run():
            f()
            torch.cuda.synchronize()

        return run

    def replays():
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1e3

    runs = [
        Run("encode_exact", sync(lambda: encoder.encode(drows)), n),
        Run("encode_capacities", sync(lambda: encoder.encode(drows, capacities=caps)), n),
        Run("op_eager", sync(lambda: op(drows)), n),
        Run("op_dynamo_eager", sync(lambda: compiled(drows)), n),
        Run("op_graph", replays, n * REPLAYS, host=False),
        Run("cpu_exact", lambda: encoder.encode(rows), n),
        Run("cpu_single", lambda: encoder.encode(rows[:1]), 1),
    ]
    for run in runs:  # warm up: compile, reserve, caches
        for _ in range(3):
            run.once()
    for _ in range(rounds):
        for run in runs:
            run.times.append(run.once())
    rt.unregister(handle)
    return dict(task=key, n=n, rounds=rounds, entities=caps["entities"] // n, tuples=int(encoder.encode(rows).relation_offsets[-1]) // n,
                us_per_state={r.name: dict(min=min(r.times), median=statistics.median(r.times)) for r in runs})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="blocks-p01e,blocks-8")
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--data", default=None, help="a directory of <domain>-ipc/test PDDL")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    results = [bench_task(key, args.n, args.rounds, args.data) for key in args.tasks.split(",")]
    for r in results:
        if "skipped" in r:
            print(f"{r['task']}: skipped ({r['skipped']})")
            continue
        print(f"{r['task']}  N={r['n']}  (entities {r['entities']}, tuples {r['tuples']} per state)")
        for name, t in r["us_per_state"].items():
            print(f"  {name:18s} min {t['min']:10.4f}  median {t['median']:10.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
