"""Python: a batch over a mymyr.rl.TaskSuite (instances of several domains) against the same rows as one
batch per domain, run one after another (the C++ side: build/.../mymyr_task_suite_bench).

The suites' domains are tables of five instances each (the instance sets of tests/python/test_rl_table.py, plus sokoban-ipc
as mymyr_task_suite_bench). Env i is in global instance i mod I (interleaved: every domain in turn), or the same multiset of
instances grouped by domain (grouped). The reference ("apart") is one batch per domain of exactly the suite batch's
rows of that domain (local ids in the same order), every domain's batch run one after another.

  - eager: BatchedEnv.step() (the random policy, the device fast path) of the suite env / of every domain's env;
  - graph: BatchedEnv.capture(K) replayed (the suite's graph / every domain's graph in turn);
  - jax: jax.jit of a lax.scan over T steps of mymyr.rl.jax.Env.step (the suite env / every domain's env in turn);
  - expand: mymyr.rl.expand of N device states (reached by --walk random steps) with their task ids, flat outputs
    (successors, parents, labels, offsets) in torch on the device (the suite / every domain's rows on its table).

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python bench/python/bench_task_suite.py [--suites gripper+blocks,miconic+blocks+gripper+sokoban]
        [--envs 16384,65536] [--modes eager,graph,jax,expand] [--layouts interleaved,grouped]

One JSON line per (suite, N, layout, mode, side): the median of --reps timed runs after a warmup; ratio lines give
t_suite / t_apart (gate: <= 1.3).
"""

import argparse
import json
import os
import pathlib
import statistics
import time

import numpy as np
import torch

import mymyr
from mymyr import rl
import mymyr.rl.torch as rt

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORK = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GEN = ROOT / "tests/data/table_instances"
BW = FORK / "ipc/blocksworld-ipc/train"
MI = FORK / "ipc/miconic-ipc/test"
SO = FORK / "ipc/sokoban-ipc"
SETS = {  # as tests/cpp/rl/table_instance_sets.hpp and mymyr_task_suite_bench
    "blocks": (BW / "domain.pddl", [BW / f"p{k}.pddl" for k in (13, 25, 37, 49, 61)]),
    "gripper": (FORK / "gripper/domain.pddl",
                [FORK / "gripper/test_problem4.pddl"] + [GEN / f"gripper/gripper-{k}.pddl" for k in (10, 20, 40, 80)]),
    "miconic": (MI / "domain.pddl",
                [MI / p for p in ("p01-easy.pddl", "p20-easy.pddl", "p01-medium.pddl", "p04-medium.pddl", "p16-medium.pddl")]),
    "sokoban": (SO / "train/domain.pddl",
                [SO / "train/p01.pddl", SO / "train/p36.pddl", SO / "train/p56.pddl", SO / "test/p22-easy.pddl",
                 SO / "test/p29-easy.pddl"]),
}
SUITES = ["gripper+blocks", "miconic+blocks+gripper+sokoban"]


def sync():
    torch.cuda.synchronize()


def timed(fn, reps, block=sync):
    out = []
    for _ in range(reps):
        block()
        t0 = time.perf_counter()
        fn()
        block()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def layout_ids(suite, n, layout):
    ids = (np.arange(n) % len(suite)).astype(np.int32)
    if layout == "grouped":
        ids = ids[np.argsort(np.asarray(suite.domain_of)[ids], kind="stable")]
    return ids


def split(suite, ids):
    """Per domain: (domain, the suite batch's rows of it, their local ids)."""
    dom = np.asarray(suite.domain_of)[ids]
    loc = np.asarray(suite.local_ids, np.int32)[ids]
    out = []
    for d in range(suite.num_domains):
        rows = np.flatnonzero(dom == d)
        if len(rows):
            out.append((d, rows, loc[rows]))
    return out


def emit(rec, t_suite, t_apart, unit_steps):
    for side, t in (("suite", t_suite), ("apart", t_apart)):
        print(json.dumps({**rec, "side": side, "us_per_step": t / unit_steps * 1e6}), flush=True)
    print(json.dumps({**rec, "side": "ratio", "ratio": t_suite / t_apart}), flush=True)


def bench_eager(suite, ids, parts, a, rec):
    dev = torch.device("cuda", 0)
    env = rt.BatchedEnv(suite, len(ids), task_ids=ids, device=dev, seed=1)
    apart = [rt.BatchedEnv(suite.tables[d], len(rows), task_ids=loc, device=dev, seed=1, first_env=int(rows[0]))
             for d, rows, loc in parts]  # fmt: skip
    for _ in range(a.warmup):
        env.step()
        for e in apart:
            e.step()

    def run_suite():
        for _ in range(a.steps):
            env.step()

    def run_apart():
        for _ in range(a.steps):
            for e in apart:
                e.step()

    ts, ta = timed(run_suite, a.reps), timed(run_apart, a.reps)
    env.check_errors()
    emit({**rec, "mode": "eager", "fast": env.core.fast}, ts, ta, a.steps)


def bench_graph(suite, ids, parts, a, rec):
    dev = torch.device("cuda", 0)
    K = a.graph_steps
    env = rt.BatchedEnv(suite, len(ids), task_ids=ids, device=dev, seed=1)
    apart = [rt.BatchedEnv(suite.tables[d], len(rows), task_ids=loc, device=dev, seed=1, first_env=int(rows[0]))
             for d, rows, loc in parts]  # fmt: skip
    for _ in range(a.warmup):
        env.step()
        for e in apart:
            e.step()
    g = env.capture(K)
    ga = [e.capture(K) for e in apart]
    g.replay()
    for x in ga:
        x.replay()
    R = max(1, a.steps // K)

    def run_suite():
        for _ in range(R):
            g.replay()

    def run_apart():
        for _ in range(R):
            for x in ga:
                x.replay()

    ts, ta = timed(run_suite, a.reps), timed(run_apart, a.reps)
    env.check_errors()
    emit({**rec, "mode": "graph", "graph_steps": K}, ts, ta, R * K)


def bench_jax(suite, ids, parts, a, rec):
    import jax
    import jax.numpy as jnp

    import mymyr.rl.jax as rj

    key = jax.random.key(1)

    def scan_of(env, n):
        @jax.jit
        def rollout(state):
            def body(carry, _):
                s, acc = carry
                s, obs, reward, terminated, truncated, info = env.step(key, s)
                return (s, acc + reward), None

            return jax.lax.scan(body, (state, jnp.zeros((n,), jnp.float32)), None, length=a.steps)[0]

        return rollout

    def runner(table, task_ids):
        env = rj.Env(table, device="cuda")
        roll = scan_of(env, len(task_ids))
        state, _ = env.reset(key, jnp.asarray(task_ids))
        state, _ = jax.block_until_ready(roll(state))  # compile and warm up
        box = {"s": state}

        def run():
            box["s"], acc = roll(box["s"])
            return acc

        return env, run

    env, rs = runner(suite, ids)
    apart = [runner(suite.tables[d], loc) for d, rows, loc in parts]
    accs = {}

    def run_suite():
        accs["s"] = rs()

    def run_apart():
        accs["a"] = [r() for _, r in apart]

    def block():
        jax.block_until_ready(accs)

    ts, ta = timed(run_suite, a.reps, block), timed(run_apart, a.reps, block)
    env.check_errors()
    emit({**rec, "mode": "jax", "fast": env.fast}, ts, ta, a.steps)


def bench_expand(suite, ids, parts, a, rec):
    dev = torch.device("cuda", 0)
    env = rt.BatchedEnv(suite, len(ids), task_ids=ids, device=dev, seed=1)
    for _ in range(a.walk):
        env.step()
    states = env.states.clone()
    tid = env.task_ids.clone()
    # the reference rows: each domain's rows of the walked batch, its local ids, its table's width
    dom = torch.as_tensor(suite.domain_of, dtype=torch.int32, device=dev)[tid.long()]
    loc_of = torch.as_tensor(suite.local_ids, dtype=torch.int32, device=dev)
    apart = []
    for d, t in enumerate(suite.tables):
        rows = torch.nonzero(dom == d).flatten()
        if len(rows):
            apart.append((t, states[rows][:, : t.words].contiguous(), loc_of[tid[rows].long()].contiguous()))
    out = {}

    def run_suite():
        for _ in range(a.expand_reps):
            out["s"] = rl.expand(suite, states, tid)

    def run_apart():
        for _ in range(a.expand_reps):
            out["a"] = [rl.expand(t, s, i) for t, s, i in apart]

    run_suite()
    run_apart()
    total = out["s"].total
    assert total == sum(x.total for x in out["a"])
    ts, ta = timed(run_suite, a.reps), timed(run_apart, a.reps)
    emit({**rec, "mode": "expand", "walk": a.walk, "successors": int(total)}, ts, ta, a.expand_reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--envs", default="16384,65536")
    ap.add_argument("--layouts", default="interleaved,grouped")
    ap.add_argument("--modes", default="eager,graph,jax,expand")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--graph-steps", type=int, default=10)
    ap.add_argument("--walk", type=int, default=20)
    ap.add_argument("--expand-reps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    modes = a.modes.split(",")
    for key in a.suites.split(","):
        tables = []
        for name in key.split("+"):
            dom, probs = SETS[name]
            tables.append(rl.TaskTable([mymyr.Task.from_pddl(dom, p, atoms="frozen") for p in probs]))
        suite = rl.TaskSuite(tables)
        for n in map(int, a.envs.split(",")):
            for layout in a.layouts.split(","):
                ids = layout_ids(suite, n, layout)
                parts = split(suite, ids)
                rec = {"suite": key, "envs": n, "layout": layout}
                for mode in modes:
                    {"eager": bench_eager, "graph": bench_graph, "jax": bench_jax, "expand": bench_expand}[mode](
                        suite, ids, parts, a, rec)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
