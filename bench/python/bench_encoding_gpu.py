"""Throughput of the relational encodings: the CPU encoder, the device environment step, and the device encoder.

Per task (the eight IPC tasks of the benchmark set) the script generates reachable rows from seeded random walks with
mymyr, then times a set of modes with interleaved rounds: every round runs every mode once, so slow phases of a shared
GPU or a loaded host hit all modes alike. It reports the minimum (the least disturbed run) and the median over the
rounds per mode.

  - cpu_*: FlatRelationEncoder.encode on the rows. `single` is one call per state; `batch256` and `batch_all` encode
    256 rows and all rows (tiled to at least --min-batch rows); `_t8` encodes with `threads=8` and `_pool8` with an
    `rl.ThreadPool(8)` of its own; `_cap` encodes into capacity-sized arrays (`capacities=`, sized to the batch); `_rel`
    is the relation-major layout. The encoder gives every thread at least 1024 graphs, so the default --min-batch
    lets all 8 threads work.
  - env_step_nN: the device environment step alone, a CUDA graph of 10 steps over N environments, replayed
    (microseconds per step and nanoseconds per environment). env_step_rand_nN is the same graph with a policy that only
    draws the random actions: the baseline of the step plus encoding. env_step_encode_nN adds the device encoder of the
    step's states to that policy.
  - device_encode_graph / device_encode_rel: the device encoder alone on the environment's states, synchronized.
    Skipped while mymyr.encoding has no DeviceEncoder.

Every record carries the bytes written per encoded state. The JSON output has the environment (GPU, driver, CUDA,
mymyr version and commit, CPU count, load), the targets as data, the recorded reference numbers, one
record per task and mode, and the result of every gate. The printed table lists each gate as pass, fail or n/a (no
mode that the gate needs ran); the device gates are soft: a failure on a contended GPU is to be reproduced in a rerun.

    env CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python bench/python/bench_encoding_gpu.py [--tasks T,...] [--rows 4096]
        [--rounds 20] [--modes cpu,env,device | mode names] [--json PATH] [--device 0] [--data DIR] [--quick]

Task PDDL is read from tests/data/ipc, or from DIR/<domain>-ipc/test with --data DIR; a task without data is skipped.
The task `blocks-8` (tests/data/tasks, no PDDL front end needed) is not part of the default set.
"""

import argparse
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

import mymyr
from mymyr import encoding, rl

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEPS = 10  # steps per captured graph
THREADS = 8
MIN_BATCH = 8192  # rows of the all-rows batches: 8 threads at the encoder's 1024 graphs per thread
SINGLE_CALLS = 256  # calls per timed run of the single-call mode
ENV_SIZES = (1024, 16384)

# (domain directory, problem) under <ipc>/<directory>/test; the tasks of the benchmark set
TASKS = {
    "blocks-p01e": ("blocksworld-ipc", "p01-easy"),
    "childsnack-p16e": ("childsnack-ipc", "p16-easy"),
    "rovers-p16e": ("rovers-ipc", "p16-easy"),
    "blocks-p05m": ("blocksworld-ipc", "p05-medium"),
    "sokoban-p16e": ("sokoban-ipc", "p16-easy"),
    "transport-p30m": ("transport-ipc", "p30-medium"),
    "miconic-p30m": ("miconic-ipc", "p30-medium"),
    "blocks-p20m": ("blocksworld-ipc", "p20-medium"),
}
TEXT_TASKS = {"blocks-8": "blocks__probBLOCKS-8-0"}  # tests/data/tasks/<name>.txt

# ------------------------------------------------------------------------------------------------ reference numbers
# Recorded timings of other encoders and prototypes on the same tasks, per task in the order of TASKS. They are the
# baselines the gates scale from and are reported next to the measured values.

# the device environment step: microseconds per step of a captured graph of 10 steps at N = 1024 and 16384
ENV_STEP_US = {
    "blocks-p01e": {1024: 37, 16384: 64},
    "childsnack-p16e": {1024: 40, 16384: 82},
    "rovers-p16e": {1024: 232, 16384: 370},
    "blocks-p05m": {1024: 76, 16384: 400},
    "sokoban-p16e": {1024: 108, 16384: 238},
    "transport-p30m": {1024: 510, 16384: 1755},
    "miconic-p30m": {1024: 101, 16384: 666},
    "blocks-p20m": {},  # no recorded step times
}
# a recorded CPU flat encoder: microseconds per state (single call, batch of 256, 8 threads)
RECORDED_CPU_US = {
    "blocks-p01e": (5.4, 1.0, 0.40),
    "childsnack-p16e": (8.2, 3.6, 0.85),
    "rovers-p16e": (10.4, 5.6, 1.29),
    "blocks-p05m": (11.9, 7.2, 1.69),
    "sokoban-p16e": (18.0, 13.3, 1.76),
    "transport-p30m": (54.6, 48.8, 6.74),
    "miconic-p30m": (69.2, 62.0, 8.66),
    "blocks-p20m": (20.6, 15.5, 3.45),
}
# a recorded reference flat encoder: microseconds per state (single call, batch of 256)
REFERENCE_FLAT_US = {
    "blocks-p01e": (32, 14),
    "childsnack-p16e": (55, 34),
    "rovers-p16e": (58, 36),
    "blocks-p05m": (70, 47),
    "sokoban-p16e": (97, 75),
    "transport-p30m": (140, 120),
    "miconic-p30m": (185, 171),
    "blocks-p20m": (113, 89),
}
# the CPU prototype with statics repeated, N = 16384: microseconds per state (1 thread, 8 threads)
PROTOTYPE_CPU_US = {
    "blocks-p01e": (0.141, 0.062),
    "childsnack-p16e": (0.497, 0.096),
    "rovers-p16e": (0.513, 0.163),
    "blocks-p05m": (0.708, 0.171),
    "sokoban-p16e": (0.617, 0.172),
    "transport-p30m": (2.108, 0.633),
    "miconic-p30m": (2.660, 0.567),
    "blocks-p20m": (2.032, 0.499),
}
# the GPU prototype, N = 16384: ns per graph of the full contract with statics repeated, the bytes per graph it writes,
# and the counts-only, COO and shared-statics variants
PROTOTYPE_GPU = {
    "blocks-p01e": dict(full_ns=9.6, bytes=646, counts_ns=5.0, counts_shared_ns=4.7, coo_ns=4.1, coo_shared_ns=3.4),
    "childsnack-p16e": dict(full_ns=25.5, bytes=1858, counts_ns=14.1, counts_shared_ns=8.7, coo_ns=19.1, coo_shared_ns=4.1),
    "rovers-p16e": dict(full_ns=48.9, bytes=3371, counts_ns=28.6, counts_shared_ns=13.9, coo_ns=39.8, coo_shared_ns=7.1),
    "blocks-p05m": dict(full_ns=20.2, bytes=1874, counts_ns=20.5, counts_shared_ns=19.5, coo_ns=18.9, coo_shared_ns=17.5),
    "sokoban-p16e": dict(full_ns=56.9, bytes=4312, counts_ns=50.5, counts_shared_ns=7.5, coo_ns=112.7, coo_shared_ns=14.9),
    "transport-p30m": dict(full_ns=58.7, bytes=12472, counts_ns=56.8, counts_shared_ns=15.3, coo_ns=90.8, coo_shared_ns=14.5),
    "miconic-p30m": dict(full_ns=81.5, bytes=17442, counts_ns=79.6, counts_shared_ns=16.8, coo_ns=128.2, coo_shared_ns=21.1),
    "blocks-p20m": dict(full_ns=39.5, bytes=4150, counts_ns=38.8, counts_shared_ns=37.0, coo_ns=39.1, coo_shared_ns=34.9),
}
# the prototype added to a captured step, N = 16384: ns per environment (the step alone, then the added time of the full
# contract with statics repeated, of COO, and of COO with shared statics)
PROTOTYPE_E2E_NS = {
    "blocks-p01e": dict(step=4.3, full=9.8, coo=4.2, coo_shared=3.3),
    "childsnack-p16e": dict(step=6.0, full=30.3, coo=22.4, coo_shared=4.0),
    "rovers-p16e": dict(step=21.5, full=56.7, coo=49.2, coo_shared=14.7),
    "blocks-p05m": dict(step=22.5, full=22.7, coo=24.4, coo_shared=22.9),
    "sokoban-p16e": dict(step=15.1, full=64.5, coo=117.5, coo_shared=18.3),
    "transport-p30m": dict(step=118, full=64.0, coo=96.0, coo_shared=19.8),
    "miconic-p30m": dict(step=43.6, full=86.9, coo=132.6, coo_shared=25.6),
}

# ------------------------------------------------------------------------------------------------ gates as data
# The targets: `gate` is the pass threshold the harness checks, `goal` what the encoder aims for. Device gates are soft.
TARGETS = [
    dict(id="T1", what="core lanes, one thread, batch of all rows", gate="<= 0.3 us + 1 ns per emitted int, per state",
         goal=None, soft=False),
    dict(id="T2", what="action lane", gate="<= 1 ns per int", goal=None, soft=False,
         needs="the action lane"),
    dict(id="T3", what="derived lanes", gate="T1 plus the axiom evaluation, or T1 alone with given derived words",
         goal=None, soft=False, needs="a task with derived predicates in the benchmark set"),
    dict(id="T-C2", what="single call from Python", gate="<= 3 us per call", goal=None, soft=False),
    dict(id="T-C1", what="8 threads against 1, batches of 4096 states or more", gate=">= 3.5x", goal=None,
         soft=False),
    dict(id="T6", what="against the recorded reference encoder, batch of 256", gate=">= 20x faster", goal=None, soft=False),
    dict(id="T7", what="relation-major against graph-major, one thread", gate="<= 1.2x", goal=None, soft=False),
    dict(id="ENV", what="device env step alone against the recorded step times",
         gate="not more than 20% slower (within_20_percent records the two-sided check)", goal=None,
         soft=True),
    dict(id="T-G1", what="device flat core, graph-major, N >= 4096",
         gate="<= 1.25 x prototype full ns per graph x (bytes written / prototype bytes)",
         goal="<= bytes / 150 GB/s + 2 ns", soft=True),
    dict(id="T-G2", what="device relation-major against graph-major", gate="<= 1.5x", goal="<= 1.25x", soft=True),
    dict(id="T-G3", what="encoding added to a captured env step, per environment",
         gate="<= 1.25 x prototype e2e full ns x (bytes written / prototype bytes)",
         goal="<= bytes / 150 GB/s + 3 ns", soft=True),
    dict(id="T-G4", what="launches and sync", gate="<= 4 kernels per width class, no host sync, no allocation after "
         "reserve, capturable", goal=None, soft=True, needs="the device encoder's reserve and a profiler"),
    dict(id="T-G5", what="device against the CPU encoder at N >= 4096", gate=">= 10x one thread and >= 3x 8 threads",
         goal=">= 20x / >= 5x", soft=True),
    dict(id="T-G6", what="scratch", gate="<= 16 B per graph + 4 B per cell + CUB temp", goal=None, soft=True,
         needs="the device encoder's bytes_needed"),
    dict(id="T-G7", what="HGraph view", gate="zero-copy, or <= 1.3x relation-major", goal="<= 1.1x", soft=True,
         needs="the HGraph view"),
    dict(id="T-G8", what="Simple on the device", gate="<= 2.5x the flat core", goal=None, soft=True,
         needs="the Simple encoding"),
    dict(id="T-G9", what="shared statics", gate="<= 1.25 x the shared-statics e2e ns", goal=None, soft=True,
         needs="shared statics"),
]

# ------------------------------------------------------------------------------------------------ tasks and rows


def ipc_roots(data=None):
    """The directories that hold the <domain>-ipc directories: `data` when given, else the vendored tasks."""
    return [pathlib.Path(data)] if data else [ROOT / "tests/data/ipc"]


def load_task(key, data=None):
    """(task, None), or (None, the reason the task cannot be loaded)."""
    if key in TEXT_TASKS:
        return mymyr.Task.from_text(str(ROOT / "tests/data/tasks" / f"{TEXT_TASKS[key]}.txt")), None
    if key not in TASKS:
        return None, f"unknown task (known: {', '.join([*TASKS, *TEXT_TASKS])})"
    if not hasattr(mymyr, "Domain"):
        return None, "mymyr was built without the PDDL front end"
    directory, problem = TASKS[key]
    for root in ipc_roots(data):
        base = root / directory / "test"
        if (base / "domain.pddl").is_file() and (base / f"{problem}.pddl").is_file():
            return mymyr.Task.from_pddl(base / "domain.pddl", base / f"{problem}.pddl"), None
    return None, f"no PDDL for {directory}/{problem} under " + ", ".join(str(r) for r in ipc_roots(data))


def walk_rows(task, count, seed, depth=64, patience=100):
    """Up to `count` distinct reachable states from seeded random walks that start at the initial state and take up
    to `depth` steps, as state rows [n, task.words] (u64). Stops early after `patience` walks that reach no new state.
    The rows are a function of the task, the count and the seed."""
    rng = random.Random(seed)
    seen, states, stale = set(), [], 0
    while len(states) < count and stale < patience:
        state, fresh = task.initial_state, False
        for _ in range(depth):
            if state not in seen:
                seen.add(state)
                states.append(state)
                fresh = True
                if len(states) >= count:
                    break
            successors = task.successor_states(state)
            if not successors:
                break
            state = rng.choice(successors)
        stale = 0 if fresh else stale + 1
    return np.ascontiguousarray(task.encode(states))


def tile(rows, n):
    """`rows` repeated cyclically to exactly n rows."""
    reps = -(-n // len(rows))
    return np.ascontiguousarray(np.tile(rows, (reps, 1))[:n])


# ------------------------------------------------------------------------------------------------ the encoder variants


def make_encoder(task, **options):
    """An encoder, or None with the reason when the installed mymyr does not support the options."""
    try:
        return encoding.FlatRelationEncoder(task, goal_derivations=(), **options), None
    except (ValueError, TypeError) as error:
        return None, str(error).splitlines()[0]


def batch_stats(batch):
    """(bytes written, ints written) of a batch: the arrays of its dictionary."""
    arrays = [a for a in batch.as_dict().values() if isinstance(a, np.ndarray)]
    return sum(a.nbytes for a in arrays), sum(a.size for a in arrays)


class Mode:
    """One timed workload. `run` does the work (and synchronizes when it uses the GPU); `units` is the number of
    encoded states (or steps) per run, `unit` what the time is per."""

    def __init__(self, name, group, run=None, units=1, unit="us_per_state", skip=None, **fields):
        self.name, self.group, self.run, self.units, self.unit, self.skip = name, group, run, units, unit, skip
        self.fields = fields
        self.times = []
        self.error = None

    def skipped(self, reason):
        self.skip = reason
        return self


def capacities_of(encoder, rows):
    """The capacity dict that fits the batch `rows` exactly."""
    batch = encoder.encode(rows)
    return dict(entities=int(batch.entity_offsets[-1]), tuples=int(batch.relation_offsets[-1]),
                args=int(batch.relation_arg_offsets[-1]))


def cpu_modes(rows, big, pool, encoders):
    """The CPU modes of one task. `encoders` = {layout: (encoder, reason)}; `pool` is an rl.ThreadPool."""
    modes = []
    batch256 = rows[:256] if len(rows) >= 256 else tile(rows, 256)
    for layout, suffix in (("graph", ""), ("relation", "_rel")):
        encoder, reason = encoders[layout]
        variants = [  # (name, rows, call)
            (f"cpu_batch256{suffix}", batch256, lambda e, d: e.encode(d)),
            (f"cpu_batch_all{suffix}", big, lambda e, d: e.encode(d)),
            (f"cpu_batch_all{suffix}_t8", big, lambda e, d: e.encode(d, threads=THREADS)),
        ]
        if layout == "graph":
            caps = capacities_of(encoder, big) if encoder is not None else None
            variants += [
                ("cpu_batch_all_pool8", big, lambda e, d: e.encode(d, pool=pool)),
                ("cpu_batch_all_cap", big, lambda e, d, c=caps: e.encode(d, capacities=c)),
            ]
        for name, data, call in variants:
            threads = THREADS if name.endswith(("_t8", "_pool8")) else 1
            mode = Mode(name, "cpu", units=len(data), layout=layout, threads=threads)
            if encoder is None:
                modes.append(mode.skipped(f"layout {layout} unavailable: {reason}"))
                continue
            mode.run = lambda e=encoder, d=data, c=call: c(e, d)
            modes.append(mode)
        if layout == "graph":
            singles = [np.ascontiguousarray(r[None]) for r in tile(rows, SINGLE_CALLS)]
            mode = Mode("cpu_single", "cpu", units=len(singles), layout=layout, threads=1)
            if encoder is None:
                modes.append(mode.skipped(f"layout {layout} unavailable: {reason}"))
            else:
                mode.run = lambda e=encoder, d=singles: [e.encode(r) for r in d]
                modes.append(mode)
    return modes


# ------------------------------------------------------------------------------------------------ the GPU side


class Gpu:
    """torch and the device, when usable."""

    def __init__(self, index):
        self.torch = None
        self.reason = None
        self.index = index
        try:
            import torch

            import mymyr.cuda as mc
            import mymyr.rl.torch as rt
        except ImportError as error:
            self.reason = f"no GPU modes ({error})"
            return
        if not torch.cuda.is_available() or not mc.available():
            self.reason = "no CUDA device"
            return
        self.torch, self.rt = torch, rt
        torch.cuda.set_device(index)
        self.device = torch.device("cuda", index)

    @property
    def ok(self):
        return self.torch is not None

    def sync(self):
        self.torch.cuda.synchronize(self.device)


def device_encoder_class():
    return getattr(encoding, "DeviceEncoder", None)


class DeviceRun:
    """A device encoder of one layout, reserved for `graphs` states, with its caller-owned output block and the
    acquisition that a captured graph holds until its last replay. The encoder is built over the CPU encoder of the
    layout (`DeviceEncoder(flat_encoder)`); `reserve(graphs)` sizes its scratch, `reserved_layout().bytes` is the output
    block that the caller allocates, `encode(states, output=...)` enqueues on the current stream, and `acquire()`
    returns the workspace to keep alive while captured work uses it."""

    def __init__(self, cls, task, gpu, graphs, layout):
        flat, reason = make_encoder(task, **({} if layout == "graph" else {"layout": layout}))
        if flat is None:
            raise ValueError(reason)
        self.encoder = cls(flat)
        self.encoder.reserve(graphs)
        self.output_bytes = int(self.encoder.reserved_layout().bytes)
        self.output = gpu.torch.empty(self.output_bytes, dtype=gpu.torch.uint8, device=gpu.device)
        self.hold = self.encoder.acquire()

    def encode(self, states):
        return self.encoder.encode(states, output=self.output)

    def sizes(self):
        """Scratch and output bytes of the reservation."""
        return dict(scratch_bytes=int(self.encoder.reserved_bytes()), output_bytes=self.output_bytes)


def release(graphs, holds):
    """Destroys the captured graphs, then the acquisitions they held: a graph retains its acquisition until its last
    replay, and an acquisition is released only after the work that used it."""
    graphs.clear()
    holds.clear()


def env_modes(task, gpu, sizes, want_encode):
    """The device modes of one task, the objects to release (captured graphs, acquisitions) and the largest
    environment. One environment per size N."""
    modes, graphs, holds = [], [], []
    env_last = None
    for n in sizes:
        names = (f"env_step_n{n}", f"env_step_rand_n{n}", f"env_step_encode_n{n}")
        try:
            env = gpu.rt.BatchedEnv(task, n, device=gpu.device, seed=1)
            for _ in range(5):
                env.step()
            gpu.sync()
            step = env.capture(STEPS)
            rand = env.capture(STEPS, lambda e: e.random_actions())
        except Exception as error:  # noqa: BLE001
            reason = f"device environment unavailable: {str(error).splitlines()[0][:160]}"
            modes += [Mode(m, "env", unit="us_per_step", skip=reason, N=n) for m in names]
            continue
        graphs += [env, step, rand]

        def replay(graph):
            graph.replay()
            gpu.sync()

        modes.append(Mode(names[0], "env", lambda g=step: replay(g), STEPS, "us_per_step", N=n, row_words=env.row_words))
        modes.append(Mode(names[1], "env", lambda g=rand: replay(g), STEPS, "us_per_step", N=n, row_words=env.row_words))
        cls = device_encoder_class()
        if cls is None:
            modes.append(Mode(names[2], "env", unit="us_per_step", N=n, skip="no device encoder"))
        elif want_encode:
            try:
                device = DeviceRun(cls, task, gpu, n, "graph")

                def policy(e, run=device):
                    run.encode(e.states)
                    return e.random_actions()

                encoded = env.capture(STEPS, policy)
                graphs.append(encoded)
                holds.append(device)
                modes.append(Mode(names[2], "env", lambda g=encoded: replay(g), STEPS, "us_per_step", N=n,
                                  **device.sizes()))
            except Exception as error:  # noqa: BLE001
                modes.append(Mode(names[2], "env", unit="us_per_step", N=n,
                                  skip=f"device encoder failed: {str(error).splitlines()[0][:160]}"))
        env_last = (env, n)
    return modes, (graphs, holds), env_last


def device_modes(task, gpu, env_n, want):
    """The device encoder alone on the states of the largest environment, graph-major and relation-major."""
    names = (("device_encode_graph", "graph"), ("device_encode_rel", "relation"))
    cls = device_encoder_class()
    if cls is None:
        return [Mode(n, "device", skip="no device encoder", layout=lay) for n, lay in names], []
    if not gpu.ok or env_n is None:
        return [Mode(n, "device", skip="no device environment to take states from", layout=lay) for n, lay in names], []
    env, n = env_n
    modes, holds = [], []
    for name, layout in names:
        try:
            device = DeviceRun(cls, task, gpu, n, layout)
            holds.append(device)

            def run(d=device):
                d.encode(env.states)
                gpu.sync()

            modes.append(Mode(name, "device", run, n, "us_per_state", layout=layout, N=n, **device.sizes()))
        except Exception as error:  # noqa: BLE001
            modes.append(Mode(name, "device", skip=f"device encoder failed: {str(error).splitlines()[0][:160]}",
                              layout=layout))
    return modes, holds


# ------------------------------------------------------------------------------------------------ timing


def time_rounds(modes, rounds):
    """Warm up each runnable mode, then run `rounds` interleaved rounds (the start mode rotates every round)."""
    live = [m for m in modes if m.run is not None and m.skip is None]
    for mode in list(live):
        try:
            for _ in range(2):
                mode.run()
        except Exception as error:  # noqa: BLE001
            mode.error = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
            live.remove(mode)
    for r in range(rounds):
        k = r % len(live) if live else 0
        for mode in live[k:] + live[:k]:
            if mode.error:
                continue
            t0 = time.perf_counter()
            try:
                mode.run()
            except Exception as error:  # noqa: BLE001
                mode.error = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
                continue
            mode.times.append((time.perf_counter() - t0) * 1e6 / mode.units)


def summarize(mode):
    """The record's timing fields."""
    out = {}
    if mode.times:
        out.update(rounds=len(mode.times), min=min(mode.times), median=statistics.median(mode.times))
        if mode.unit == "us_per_step" and "N" in mode.fields:
            out["min_ns_per_env"] = out["min"] * 1e3 / mode.fields["N"]
            out["median_ns_per_env"] = out["median"] * 1e3 / mode.fields["N"]
        if mode.unit == "us_per_state":
            out["min_ns_per_state"] = out["min"] * 1e3
    return out


# ------------------------------------------------------------------------------------------------ gates


class Index:
    """The measured records of one run, by task and mode."""

    def __init__(self, records):
        self.rec = {(r["task"], r["mode"]): r for r in records if r["status"] == "ok"}
        self.tasks = sorted({r["task"] for r in records}, key=lambda t: list(TASKS).index(t) if t in TASKS else 99)

    def get(self, task, mode, field="min"):
        r = self.rec.get((task, mode))
        return None if r is None else r.get(field)

    def bytes(self, task):
        for mode in ("cpu_batch_all", "cpu_batch256"):
            r = self.rec.get((task, mode))
            if r:
                return r.get("bytes_per_state")
        return None

    def largest_env(self, task, prefix):
        found = [(r["N"], r) for (t, m), r in self.rec.items() if t == task and m.startswith(prefix + "_n")]
        return max(found, key=lambda x: x[0])[1] if found else None


def verdict(ok):
    return "pass" if ok else "fail"


def gate_results(index):
    """{gate id: {task: dict(status, value, limit)}} for the gates that the measured modes allow."""
    out = {t["id"]: {} for t in TARGETS}
    for task in index.tasks:
        g = lambda m, f="min": index.get(task, m, f)  # noqa: E731
        b = index.bytes(task)
        ints = index.rec.get((task, "cpu_batch_all"), {}).get("ints_per_state")
        if g("cpu_batch_all") is not None and ints is not None:
            limit = 0.3 + 1e-3 * ints
            out["T1"][task] = dict(value=g("cpu_batch_all"), limit=limit, status=verdict(g("cpu_batch_all") <= limit))
        if g("cpu_single") is not None:
            out["T-C2"][task] = dict(value=g("cpu_single"), limit=3.0, status=verdict(g("cpu_single") <= 3.0))
        if g("cpu_batch_all") is not None and g("cpu_batch_all_t8") is not None:
            speedup = g("cpu_batch_all") / g("cpu_batch_all_t8")
            n = index.rec[(task, "cpu_batch_all")]["units"]
            if n >= 4096:
                out["T-C1"][task] = dict(value=speedup, limit=3.5, status=verdict(speedup >= 3.5))
        if g("cpu_batch256") is not None and task in REFERENCE_FLAT_US:
            limit = REFERENCE_FLAT_US[task][1] / 20
            out["T6"][task] = dict(value=g("cpu_batch256"), limit=limit, status=verdict(g("cpu_batch256") <= limit))
        if g("cpu_batch_all") is not None and g("cpu_batch_all_rel") is not None:
            ratio = g("cpu_batch_all_rel") / g("cpu_batch_all")
            out["T7"][task] = dict(value=ratio, limit=1.2, status=verdict(ratio <= 1.2))
        for n, base in ENV_STEP_US.get(task, {}).items():
            value = g(f"env_step_n{n}")
            if value is not None:
                ratio = value / base
                key = task if n == max(ENV_STEP_US[task]) else f"{task}@{n}"
                out["ENV"][key] = dict(value=value, limit=base, ratio=ratio, status=verdict(ratio <= 1.2),
                                       within_20_percent=abs(ratio - 1) <= 0.2,
                                       median_ratio=(g(f"env_step_n{n}", "median") or float("nan")) / base)
        proto = PROTOTYPE_GPU.get(task)
        graph = g("device_encode_graph")
        if graph is not None and proto and b:
            run = index.rec[(task, "device_encode_graph")]
            if run["units"] >= 4096:
                limit = 1.25 * proto["full_ns"] * b / proto["bytes"]
                out["T-G1"][task] = dict(value=graph * 1e3, limit=limit, status=verdict(graph * 1e3 <= limit),
                                         goal=b / 150 + 2)
                cpu1, cpu8 = g("cpu_batch_all"), g("cpu_batch_all_t8")
                if cpu1 is not None and cpu8 is not None:
                    s1, s8 = cpu1 / graph, cpu8 / graph
                    out["T-G5"][task] = dict(value=[s1, s8], limit=[10, 3], status=verdict(s1 >= 10 and s8 >= 3))
        rel = g("device_encode_rel")
        if graph is not None and rel is not None:
            out["T-G2"][task] = dict(value=rel / graph, limit=1.5, status=verdict(rel / graph <= 1.5))
        enc_run, rand_run = index.largest_env(task, "env_step_encode"), index.largest_env(task, "env_step_rand")
        e2e = PROTOTYPE_E2E_NS.get(task)
        if enc_run and rand_run and e2e and b and enc_run["N"] == rand_run["N"]:
            added = enc_run["min_ns_per_env"] - rand_run["min_ns_per_env"]
            limit = 1.25 * e2e["full"] * b / PROTOTYPE_GPU[task]["bytes"]
            out["T-G3"][task] = dict(value=added, limit=limit, status=verdict(added <= limit), goal=b / 150 + 3)
    return out


def gate_table(results):
    """(rows, text): one row per gate with its overall status."""
    needs = {t["id"]: t.get("needs") for t in TARGETS}
    rows = []
    for target in TARGETS:
        per_task = results[target["id"]]
        statuses = [r["status"] for r in per_task.values()]
        if not statuses:
            status, detail = "n/a", needs[target["id"]] or "the modes it needs were skipped"
            if needs[target["id"]]:
                detail = "needs " + detail
        else:
            status = "fail" if "fail" in statuses else "pass"
            detail = f"{statuses.count('pass')}/{len(statuses)} pass" + (
                "; " + ", ".join(t for t, r in per_task.items() if r["status"] == "fail") if "fail" in statuses else "")
            if status == "fail" and target["soft"]:
                detail += " (soft: rerun before it blocks)"
        rows.append(dict(id=target["id"], status=status, detail=detail, what=target["what"]))
    width = max(len(r["what"]) for r in rows)
    lines = [f"{r['id']:<6} {r['status']:<5} {r['what']:<{width}}  {r['detail']}" for r in rows]
    return rows, "\n".join(lines)


# ------------------------------------------------------------------------------------------------ environment


def shell(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_status(index):
    out = shell("nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits")
    if not out:
        return None
    util, used, total = (float(x) for x in out.split(","))
    return dict(utilization_percent=util, memory_used_mib=used, memory_total_mib=total)


def environment(gpu, args):
    env = dict(
        python=platform.python_version(), free_threaded=not getattr(sys, "_is_gil_enabled", lambda: True)(),
        numpy=np.__version__, mymyr_version=getattr(mymyr, "__version__", None),
        mymyr_commit=shell("git", "rev-parse", "--short", "HEAD"), host=platform.node(), cpu_count=os.cpu_count(),
        load_average_start=os.getloadavg(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_encoder=device_encoder_class() is not None, gpu=None)
    if gpu.ok:
        torch = gpu.torch
        props = torch.cuda.get_device_properties(gpu.device)
        env["gpu"] = dict(index=args.device, name=props.name, total_memory_mib=props.total_memory // 2**20,
                          cuda=torch.version.cuda, torch=torch.__version__,
                          driver=shell("nvidia-smi", "-i", str(args.device), "--query-gpu=driver_version",
                                       "--format=csv,noheader"), status_start=gpu_status(args.device))
    else:
        env["gpu_skipped"] = gpu.reason
    return env


# ------------------------------------------------------------------------------------------------ main

GROUPS = ("cpu", "env", "device")


def wanted(mode, selection):
    return selection is None or mode.name in selection or mode.group in selection


def run_task(key, args, gpu, pool):
    """The records of one task."""
    task, reason = load_task(key, args.data)
    base = dict(task=key)
    if task is None:
        return [dict(**base, mode="*", status="skipped", reason=reason)]
    rows = walk_rows(task, args.rows, args.seed)
    big = tile(rows, max(len(rows), args.min_batch))
    info = dict(**base, n_rows=len(rows), words=task.words, objects=len(task.objects))

    encoders = {"graph": make_encoder(task), "relation": make_encoder(task, layout="relation")}
    modes = cpu_modes(rows, big, pool, encoders)
    graphs, holds = [], []
    env_n = None
    selection = args.modes
    if gpu.ok:
        extra, (graphs, holds), env_n = env_modes(task, gpu, args.env_sizes, want_encode=True)
        modes += extra
    else:
        for n in args.env_sizes:
            modes += [Mode(f"env_step_n{n}", "env", unit="us_per_step", N=n, skip=gpu.reason),
                      Mode(f"env_step_rand_n{n}", "env", unit="us_per_step", N=n, skip=gpu.reason),
                      Mode(f"env_step_encode_n{n}", "env", unit="us_per_step", N=n,
                           skip=gpu.reason if device_encoder_class() else "no device encoder")]
    extra, more = device_modes(task, gpu, env_n, True)
    modes += extra
    holds += more
    modes = [m for m in modes if wanted(m, selection)]

    # the output of the CPU encoder per state, for the bytes column
    stats = {}
    for layout, (encoder, _) in encoders.items():
        if encoder is not None:
            batch = encoder.encode(rows[: min(len(rows), 256)])
            nbytes, nints = batch_stats(batch)
            stats[layout] = (nbytes / batch.num_graphs, nints / batch.num_graphs)
    time_rounds(modes, args.rounds)

    records = []
    for mode in modes:
        record = dict(info, mode=mode.name, group=mode.group, unit=mode.unit, units=mode.units, **mode.fields)
        layout = mode.fields.get("layout")
        if mode.skip is not None:
            record.update(status="skipped", reason=mode.skip)
        elif mode.error:
            record.update(status="error", reason=mode.error)
        else:
            record.update(status="ok", **summarize(mode))
            if mode.group in ("cpu", "device") and (layout or "graph") in stats:
                record["bytes_per_state"], record["ints_per_state"] = stats[layout or "graph"]
        records.append(record)
    if gpu.ok:
        gpu.sync()
    release(graphs, holds)
    if gpu.ok:
        gpu.sync()
        gpu.torch.cuda.empty_cache()
    return records


def baselines():
    return dict(
        env_step_us=ENV_STEP_US, previous_cpu_us={k: dict(zip(("single", "batch256", "threads8"), v))
                                                  for k, v in RECORDED_CPU_US.items()},
        reference_flat_us={k: dict(zip(("single", "batch256"), v)) for k, v in REFERENCE_FLAT_US.items()},
        prototype_cpu_us={k: dict(zip(("threads1", "threads8"), v)) for k, v in PROTOTYPE_CPU_US.items()},
        prototype_gpu_ns_per_graph=PROTOTYPE_GPU, prototype_e2e_ns_per_env=PROTOTYPE_E2E_NS)


def print_records(records):
    print(f"{'task':<16}{'mode':<26}{'unit':<12}{'min':>10}{'median':>10}  bytes/state")
    for r in records:
        if r["status"] == "ok":
            extra = f"{r['bytes_per_state']:.0f}" if "bytes_per_state" in r else ""
            print(f"{r['task']:<16}{r['mode']:<26}{r['unit']:<12}{r['min']:>10.3f}{r['median']:>10.3f}  {extra}")
        else:
            print(f"{r['task']:<16}{r['mode']:<26}{r['status']} ({r.get('reason', '')})")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks", help=f"comma-separated task keys (default: {','.join(TASKS)})")
    p.add_argument("--rows", type=int, default=4096, help="distinct rows per task (default 4096)")
    p.add_argument("--rounds", type=int, default=20, help="interleaved rounds (default 20)")
    p.add_argument("--modes", help="comma-separated mode names or groups (cpu, env, device); default all")
    p.add_argument("--json", metavar="PATH", help="write the result as JSON")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    p.add_argument("--data", metavar="DIR", help="directory with the <domain>-ipc task directories (default tests/data/ipc)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random walks")
    p.add_argument("--min-batch", type=int, default=MIN_BATCH, help="rows of the all-rows batches, tiled up to this")
    p.add_argument("--env-sizes", default=",".join(map(str, ENV_SIZES)), help="environment counts N")
    p.add_argument("--quick", action="store_true", help="a smoke configuration: one small task, few rows and rounds")
    args = p.parse_args(argv)
    args.env_sizes = tuple(int(x) for x in args.env_sizes.split(",") if x)
    args.tasks = args.tasks.split(",") if args.tasks else None
    args.modes = set(args.modes.split(",")) if args.modes else None
    if args.quick:
        args.tasks = args.tasks or ["blocks-8"]
        args.rows = min(args.rows, 128)
        args.rounds = min(args.rounds, 3)
        args.min_batch = min(args.min_batch, 256)
        args.env_sizes = tuple(n for n in args.env_sizes if n <= 1024) or (256,)
    args.tasks = args.tasks or list(TASKS)
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    gpu = Gpu(args.device)
    env = environment(gpu, args)
    records = []
    pool = rl.ThreadPool(THREADS)
    for key in args.tasks:
        records += run_task(key, args, gpu, pool)
    if gpu.ok:
        env["gpu"]["status_end"] = gpu_status(args.device)
    env["load_average_end"] = os.getloadavg()
    index = Index([r for r in records if r["mode"] != "*"])
    results = gate_results(index)
    rows, table = gate_table(results)
    print_records(records)
    print()
    print(table)
    if args.json:
        out = dict(environment=env, config={k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()},
                   targets=TARGETS, baselines=baselines(), records=records, gates=rows, gate_details=results)
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
