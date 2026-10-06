#!/usr/bin/env python3
"""RL-style multi-threaded workloads on the real bindings
from the mymyr_proto prototype to mymyr.Task / mymyr.rl). Run with the free-threaded interpreter.

Workloads (all threads share one task; native workspaces are per thread automatically):
  rollouts_local   Python random-walk episodes through the shared Task object; (s, a, s', done) into a local list
  rollouts_handle  the same through a per-thread task.local() handle
  bulk_expand      NumPy batches: rl.expand(batch of 256) -> NumPy picks one successor per parent -> next batch
  native_walks     the same random walks as one native call per thread (rl.random_walks)
  sample_encode    replay sampling: random batches of 256 States -> task.encode -> popcount
  pool_expand      one Python thread, one rl.expand call per batch of 256 x T states split over rl.ThreadPool(T)

Units are expanded states (bulk, pool) or steps (the rest). Prints one JSON line per (workload, threads, repeat),
with the 1-minute load average before and after each run.

Usage: python bench_rl.py [--domain D --problem P | --text T] [--threads 1,4,16,32] [--steps N] [--only a,b]
"""

import argparse
import json
import os
import pathlib
import random
import sys
import threading
import time

import numpy as np

import mymyr
from mymyr import rl

ROOT = pathlib.Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "tests/data/pddl/blocks"

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default=str(BLOCKS / "domain.pddl"))
ap.add_argument("--problem", default=str(BLOCKS / "probBLOCKS-8-0.pddl"))
ap.add_argument("--text", default="", help="a normalized text task instead of PDDL")
ap.add_argument("--atoms", default="frozen", choices=["frozen", "lazy", "auto"])
ap.add_argument("--threads", default="1,4,16,32")
ap.add_argument("--steps", type=int, default=200_000, help="per thread")
ap.add_argument("--episode", type=int, default=50)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--only", default="")
ap.add_argument("--out", default="", help="append the JSON lines to this file too")
a = ap.parse_args()


def current_cpu():
    """The CPU this thread runs on (Linux; -1 elsewhere): the task's memory is first touched on its NUMA node."""
    try:
        with open("/proc/thread-self/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[36])
    except (OSError, IndexError, ValueError):
        return -1


if a.text:
    task = mymyr.Task.from_text(a.text, atoms=a.atoms)
else:
    task = mymyr.Task.from_pddl(a.domain, a.problem, atoms=a.atoms)
load_cpu = current_cpu()
gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
out_file = open(a.out, "a") if a.out else None


def emit(record):
    line = json.dumps(record)
    print(line, flush=True)
    if out_file:
        out_file.write(line + "\n")
        out_file.flush()


def run(name, fn, T, rep):
    barrier = threading.Barrier(T + 1)
    out = [None] * T

    def body(i):
        task.local().successor_states(task.initial_state)  # warm up this thread's native workspace
        barrier.wait()
        out[i] = fn(i, T)

    ths = [threading.Thread(target=body, args=(i,)) for i in range(T)]
    load0 = os.getloadavg()[0]
    for th in ths:
        th.start()
    barrier.wait()
    t0 = time.perf_counter()
    for th in ths:
        th.join()
    dt = time.perf_counter() - t0
    units = sum(o[0] for o in out)
    emit({"workload": name, "threads": T, "repeat": rep, "gil": gil, "units": units, "seconds": round(dt, 4),
          "units_per_s": round(units / dt), "per_thread_per_s": round(units / dt / T),
          "load1_before": round(load0, 2), "load1_after": round(os.getloadavg()[0], 2)})


def rollouts(api, i):
    rng = random.Random(i)
    buf = []
    succ_of = api.successor_states
    init = api.initial_state
    s = init
    for step in range(1, a.steps + 1):
        succ = succ_of(s)
        if not succ:
            s = init
            continue
        j = rng.randrange(len(succ))
        s2 = succ[j]
        done = step % a.episode == 0
        buf.append((s, j, s2, done))
        s = init if done else s2
    return (a.steps, len(buf))


def rollouts_local(i, T):
    return rollouts(task, i)


def rollouts_handle(i, T):
    return rollouts(task.local(), i)


def bulk_expand(i, T):
    rng = np.random.default_rng(i)
    h = task.local()
    start = np.repeat(h.encode([h.initial_state]), a.batch, axis=0)
    batch = start
    expanded = 0
    replay = []
    while expanded < a.steps:
        exp = rl.expand(h, batch)
        expanded += batch.shape[0]
        off = exp.offsets
        counts = off[1:] - off[:-1]
        has = counts > 0
        if not has.any():
            batch = start
            continue
        # one uniformly random successor per parent (parents without successors are dropped)
        pick = off[:-1][has] + (rng.random(int(has.sum())) * counts[has]).astype(np.int64)
        batch = exp.succ[pick]
        replay.append(batch)
        if len(replay) > 50:
            replay.clear()
        if batch.shape[0] < a.batch // 2:
            batch = start
    return (expanded,)


def native_walks(i, T):
    rl.random_walks(task, a.steps, a.episode, i)
    return (a.steps,)


replay_pool = []


def sample_encode(i, T):
    rng = random.Random(i)
    h = task.local()
    n = total = 0
    while n < a.steps:
        arr = h.encode(rng.sample(replay_pool, 256))
        total += int(np.bitwise_count(arr).sum())
        n += 256
    return (n, total)


def pool_expand(T, rep):
    """One Python thread; each rl.expand call splits a batch of 256 x T states over a ThreadPool(T)."""
    pool = rl.ThreadPool(T)
    rng = np.random.default_rng(rep)
    B = a.batch * T
    start = np.repeat(task.encode([task.initial_state]), B, axis=0)
    batch = start
    rl.expand(task, batch, pool=pool)  # warm up the members' workspaces
    load0 = os.getloadavg()[0]
    expanded = 0
    t0 = time.perf_counter()
    while expanded < a.steps * T:
        exp = rl.expand(task, batch, pool=pool)
        expanded += batch.shape[0]
        off = exp.offsets
        counts = off[1:] - off[:-1]
        has = counts > 0
        pick = off[:-1][has] + (rng.random(int(has.sum())) * counts[has]).astype(np.int64)
        batch = exp.succ[pick]
        if batch.shape[0] < B // 2:
            batch = start
    dt = time.perf_counter() - t0
    emit({"workload": "pool_expand", "threads": T, "repeat": rep, "gil": gil, "units": expanded, "seconds": round(dt, 4),
          "units_per_s": round(expanded / dt), "per_thread_per_s": round(expanded / dt / T),
          "load1_before": round(load0, 2), "load1_after": round(os.getloadavg()[0], 2)})


emit({"task": a.text or a.problem, "atoms": task.atom_mode, "words": task.words, "python": sys.version.split()[0],
      "gil": gil, "numpy": np.__version__, "mymyr": mymyr.__version__, "cpus": os.cpu_count(),
      "load_cpu": load_cpu})
work = {"rollouts_local": rollouts_local, "rollouts_handle": rollouts_handle, "bulk_expand": bulk_expand,
        "native_walks": native_walks, "sample_encode": sample_encode, "pool_expand": None}
only = [x for x in a.only.split(",") if x]
for name, fn in work.items():
    if only and name not in only:
        continue
    if name == "sample_encode" and not replay_pool:
        s = task.initial_state
        rng = random.Random(0)
        for _ in range(20_000):
            succ = task.successor_states(s)
            s = rng.choice(succ) if succ else task.initial_state
            replay_pool.append(s)
    for T in map(int, a.threads.split(",")):
        for rep in range(a.repeat):
            if name == "pool_expand":
                pool_expand(T, rep)
            else:
                run(name, fn, T, rep)
