#!/usr/bin/env python3
"""mymyr.rl.CpuEnvPool env-steps/s from free-threaded Python over a mixed TaskTable (random policy,
autoreset), at 1 / 8 / 32 pool threads. Run with the free-threaded interpreter (the GIL stays off).

Drivers:
  step     one Python thread: pool.step() over all envs (the pool's threads split each batch; a lockstep batch)
  async    one Python thread, two halves of the envs in flight: recv one half, send it again, then the other (EnvPool's
           asynchronous use: the workers always have a queued send)
  threads  T Python threads, each owning N / T envs: send(its envs) -> recv_ticket, in a loop (a pool of T threads)

The received batches are dropped at once; their arrays go back to the pool (CpuEnvPool.recycle through the batch's
last reference) and later sends reuse them.

Prints one JSON line per (driver, threads) with env-steps/s, the speedup over 1 thread, and the 1-minute load average
before and after. The table is an instance set of tests/cpp/rl/table_instance_sets.hpp (the fork's PDDL plus tests/data/table_instances).

Usage: python bench/python/bench_table_pool.py [--set blocks] [--envs 16384] [--threads 1,8,32] [--steps 50]
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

import mymyr
from mymyr import rl

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORK = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GEN = ROOT / "tests/data/table_instances"
BW = FORK / "ipc/blocksworld-ipc/train"
MI = FORK / "ipc/miconic-ipc/test"
SETS = {
    "blocks": (BW / "domain.pddl", [BW / f"p{k}.pddl" for k in (13, 25, 37, 49, 61)]),
    "gripper": (FORK / "gripper/domain.pddl",
                [FORK / "gripper/test_problem4.pddl"] + [GEN / f"gripper/gripper-{k}.pddl" for k in (10, 20, 40, 80)]),
    "miconic": (MI / "domain.pddl",
                [MI / p for p in ("p01-easy.pddl", "p20-easy.pddl", "p01-medium.pddl", "p04-medium.pddl", "p16-medium.pddl")]),
    "logistics": (FORK / "logistics/domain.pddl",
                  [FORK / "logistics/test_problem.pddl"]
                  + [GEN / f"logistics/logistics-{k}.pddl" for k in ("c2-s3-p4-a1", "c3-s3-p8-a2", "c4-s4-p16-a3", "c6-s5-p30-a4")]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="blocks", choices=sorted(SETS))
ap.add_argument("--envs", type=int, default=16384)
ap.add_argument("--threads", default="1,8,32")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--max-steps", type=int, default=100)
ap.add_argument("--drivers", default="step,async,threads")
args = ap.parse_args()

if getattr(sys, "_is_gil_enabled", lambda: True)():
    print("warning: the GIL is enabled (run the free-threaded interpreter with PYTHON_GIL=0)", file=sys.stderr)

dom, probs = SETS[args.set]
tasks = [mymyr.Task.from_pddl(dom, p, atoms="frozen") for p in probs]
table = rl.TaskTable(tasks)
N = args.envs
ids = (np.arange(N) % len(tasks)).astype(np.int32)


def load() -> float:
    return os.getloadavg()[0]


def run_step(T: int) -> float:
    pool = rl.CpuEnvPool(table, N, threads=T, seed=1, max_steps=args.max_steps)
    pool.reset(task_ids=ids)
    for _ in range(args.warmup):
        pool.step()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        pool.step()
    return N * args.steps / (time.perf_counter() - t0)


def run_async(T: int) -> float:
    pool = rl.CpuEnvPool(table, N, threads=T, seed=1, max_steps=args.max_steps)
    pool.reset(task_ids=ids)
    halves = [np.arange(0, N // 2, dtype=np.int32), np.arange(N // 2, N, dtype=np.int32)]
    tickets = [pool.send(h) for h in halves]
    t0 = 0.0
    for t in range(args.warmup + args.steps):
        if t == args.warmup:
            t0 = time.perf_counter()
        for k in (0, 1):
            pool.recv_ticket(tickets[k])
            tickets[k] = pool.send(halves[k])
    rate = N * args.steps / (time.perf_counter() - t0)
    for k in (0, 1):
        pool.recv_ticket(tickets[k])
    return rate


def run_threads(T: int) -> float:
    pool = rl.CpuEnvPool(table, N, threads=T, seed=1, max_steps=args.max_steps)
    pool.reset(task_ids=ids)
    per = N // T
    barrier = threading.Barrier(T + 1)
    times = [0.0] * T

    def worker(w: int) -> None:
        envs = np.arange(w * per, (w + 1) * per, dtype=np.int32)
        for _ in range(args.warmup):
            pool.recv_ticket(pool.send(envs))
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            pool.recv_ticket(pool.send(envs))
        times[w] = time.perf_counter() - t0

    ths = [threading.Thread(target=worker, args=(w,)) for w in range(T)]
    for th in ths:
        th.start()
    barrier.wait()
    for th in ths:
        th.join()
    return per * T * args.steps / max(times)


info = {"set": args.set, "instances": len(tasks), "objects": table.num_objects, "words": table.instance_words,
        "envs": N, "steps": args.steps, "max_steps": args.max_steps, "cpus": os.cpu_count(),
        "gil": getattr(sys, "_is_gil_enabled", lambda: True)()}
for driver in args.drivers.split(","):
    base = None
    for T in (int(x) for x in args.threads.split(",")):
        before = load()
        rate = {"step": run_step, "async": run_async, "threads": run_threads}[driver](T)
        after = load()
        base = base or rate
        print(json.dumps({"tool": "bench_table_pool", "driver": driver, "threads": T, "env_steps_per_s": round(rate),
                          "speedup": round(rate / base, 2), "load_before": before, "load_after": after, **info}),
              flush=True)
