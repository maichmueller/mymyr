"""The cost of the relational encoder's XLA FFI target (mymyr.encoding.jax.encode) inside jitted JAX functions.

For each task the script generates reachable rows from seeded random walks, tiles them to N rows and times, with
interleaved rounds (every round runs every mode once), these ways of encoding them with the same capacities:

  - jax_cuda: a jitted function of one encode on the CUDA device, called from Python, until its outputs are ready;
  - jax_cuda_scan: a jitted lax.scan of STEPS encodes on the CUDA device (one call), per encoded state; with XLA's
    command buffers on (``--xla_gpu_enable_command_buffer=...CUSTOM_CALL,WHILE --xla_gpu_graph_min_graph_size=1``) the
    loop runs as a CUDA graph;
  - jax_cpu: the jitted encode on the CPU device (the host encoder with `--threads` threads);
  - cpu_capacities: `encoder.encode(rows, capacities=...)`, the host encoder's capacity form with the same threads.

All times are microseconds per encoded state; the table prints the minimum and the median over the rounds.
bench/python/bench_encoding_device_calls.py times the torch paths of the same kernels, and the C++ device encoding
bench (bench/cuda/device_encoding_bench.cpp) the kernels without Python.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python bench/python/bench_encoding_jax.py [--tasks blocks-p01e,blocks-8] [--n 16384]
        [--rounds 30] [--threads 1] [--json PATH]
"""

import argparse
import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import mymyr.encoding.jax as ej

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from bench_encoding_gpu import load_task, make_encoder, padded_capacities, tile, walk_rows  # noqa: E402

STEPS = 10  # encodes per lax.scan call


def capacities_of(encoder, rows):
    batch = encoder.encode(rows)
    return padded_capacities(
        dict(graphs=batch.num_graphs, entities=int(batch.entity_offsets[-1]), args=int(batch.relation_arg_offsets[-1]))
    )


def timed(f, units):
    def run():
        start = time.perf_counter()
        f()
        return (time.perf_counter() - start) * 1e6 / units

    return run


def bench_task(key, n, rounds, threads, data):
    task, reason = load_task(key, data)
    if task is None:
        return dict(task=key, skipped=reason)
    encoder, reason = make_encoder(task)
    if encoder is None:
        return dict(task=key, skipped=reason)
    rows = tile(walk_rows(task, 4096, seed=1), n)
    caps = capacities_of(encoder, rows)
    handle = ej.register(encoder)
    halves = rows.view(np.uint32)
    single = jax.jit(lambda r: ej.encode(handle, r, capacities=caps, threads=threads))

    def scan(steps):  # the rows of every step are an input of their own, so no encode is loop-invariant
        def body(carry, r):
            batch = ej.encode(handle, r, capacities=caps)
            return carry + batch.overflow, None

        return jax.lax.scan(body, jnp.uint32(0), steps)[0]

    modes = {}
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = None
    if gpu is not None:
        drows = jax.device_put(halves, gpu)
        steps = jax.device_put(np.stack([halves] * STEPS), gpu)
        scanned = jax.jit(scan)
        modes["jax_cuda"] = timed(lambda: jax.block_until_ready(single(drows)), n)
        modes["jax_cuda_scan"] = timed(lambda: jax.block_until_ready(scanned(steps)), n * STEPS)
    crows = jax.device_put(halves, jax.devices("cpu")[0])
    modes["jax_cpu"] = timed(lambda: jax.block_until_ready(single(crows)), n)
    modes["cpu_capacities"] = timed(lambda: encoder.encode(rows, capacities=caps, threads=threads), n)
    times = {name: [] for name in modes}
    for name, run in modes.items():  # warm up: compile, upload
        for _ in range(3):
            run()
    for _ in range(rounds):
        for name, run in modes.items():
            times[name].append(run())
    ej.unregister(handle)
    return dict(task=key, n=n, rounds=rounds, threads=threads, xla_flags=os.environ.get("XLA_FLAGS", ""),
                us_per_state={k: dict(min=min(v), median=statistics.median(v)) for k, v in times.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="blocks-p01e,blocks-8")
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data", default=None, help="a directory of <domain>-ipc/test PDDL")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    results = [bench_task(key, args.n, args.rounds, args.threads, args.data) for key in args.tasks.split(",")]
    for r in results:
        if "skipped" in r:
            print(f"{r['task']}: skipped ({r['skipped']})")
            continue
        print(f"{r['task']}  N={r['n']}  threads={r['threads']}")
        for name, t in r["us_per_state"].items():
            print(f"  {name:16s} min {t['min']:10.4f}  median {t['median']:10.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
