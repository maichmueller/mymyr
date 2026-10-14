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
    draws the random actions: the baseline of the step plus encoding. env_step_encode_nN adds the device encode of the
    step's states to that policy.
  - device_encode_graph / device_encode_rel: `FlatRelationEncoder.encode` on the device rows of the largest environment,
    graph-major and relation-major, with `capacities=` measured from the CPU encoding of those rows. One encode is
    captured into a CUDA graph on a side stream and 10 replays are timed between CUDA events (median over the
    repeats); device_encode_graph_eager is the same call timed as one eager call plus a synchronize, on the host clock.
    Every device run is built and warmed outside any capture, and only one is alive at a time. They are skipped with
    the reason "not supported on the device" while `encode` takes no device rows; on a build that has a device
    encode path the harness cannot drive, the run fails.

Every record carries the bytes written per encoded state, measured on the batch that mode times. The JSON output has
the environment (GPU, driver, CUDA, mymyr version and commit, CPU count, load), the targets as data, the recorded
reference numbers, one record per task and mode, and the result of every gate. Each gate lists the tasks it applies to
and prints pass, fail, error, "incomplete (k of n)" naming the tasks it could not evaluate (a skipped task or mode),
or n/a with the reason (no baseline, a batch under 4096 states, modes not selected or unavailable). The gate cpu-eight-threads
and the device gates are soft: a failure on a contended host or GPU blocks only after a rerun on a quiet host also fails.

    env CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python bench/python/bench_encoding_gpu.py [--tasks T,...] [--rows 4096]
        [--rounds 20] [--modes cpu,env,device | mode names] [--json PATH] [--device 0] [--data DIR] [--quick]

Task PDDL is read from tests/data/ipc, or from DIR/<domain>-ipc/test with --data DIR; a task without data is skipped
and its gates are incomplete. Of the eight tasks of the benchmark set only blocks-p01e is vendored under tests/data/ipc;
the other seven (p16, p30, ...) need --data to point at a full copy of the IPC learning-track domains.
The task `blocks-8` (tests/data/tasks, no PDDL front end needed) is not part of the default set.
"""

import argparse
import contextlib
import json
import math
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
REPLAYS = 10  # replays of a captured encode between one pair of timing events
CAPACITY_HEADROOM = 1.25  # the measured capacities of a device encode, for rows that change between encodes
PROBE_STATES = 8  # rows of the probe that tells whether `encode` takes device rows
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
# Timings of other encoders on the same tasks, per task in the order of TASKS: earlier and standalone encoders of the
# same output that are not part of mymyr, and recordings of the device environment step. They are the baselines the
# gates scale from and are reported next to the measured values.

# the device environment step as recorded on the benchmark GPU: microseconds per step of a captured graph of 10 steps
# at N = 1024 and 16384
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
# an earlier CPU flat encoder of the same output, not part of mymyr: microseconds per state (single call, batch of 256, 8 threads)
EARLIER_CPU_US = {
    "blocks-p01e": (5.4, 1.0, 0.40),
    "childsnack-p16e": (8.2, 3.6, 0.85),
    "rovers-p16e": (10.4, 5.6, 1.29),
    "blocks-p05m": (11.9, 7.2, 1.69),
    "sokoban-p16e": (18.0, 13.3, 1.76),
    "transport-p30m": (54.6, 48.8, 6.74),
    "miconic-p30m": (69.2, 62.0, 8.66),
    "blocks-p20m": (20.6, 15.5, 3.45),
}
# a reference flat encoder outside mymyr, driven from Python: microseconds per state (single call, batch of 256)
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
# a standalone CPU encoder of the same contract, not part of mymyr, with the static relations written for every graph,
# N = 16384: microseconds per state (1 thread, 8 threads)
STANDALONE_CPU_US = {
    "blocks-p01e": (0.141, 0.062),
    "childsnack-p16e": (0.497, 0.096),
    "rovers-p16e": (0.513, 0.163),
    "blocks-p05m": (0.708, 0.171),
    "sokoban-p16e": (0.617, 0.172),
    "transport-p30m": (2.108, 0.633),
    "miconic-p30m": (2.660, 0.567),
    "blocks-p20m": (2.032, 0.499),
}
# a standalone device encoder of the same contract, not part of mymyr, N = 16384: ns per graph of the full contract
# with the static relations written for every graph, the bytes per graph it writes, and the counts-only, COO and
# shared-statics variants
STANDALONE_DEVICE = {
    "blocks-p01e": dict(full_ns=9.6, bytes=646, counts_ns=5.0, counts_shared_ns=4.7, coo_ns=4.1, coo_shared_ns=3.4),
    "childsnack-p16e": dict(full_ns=25.5, bytes=1858, counts_ns=14.1, counts_shared_ns=8.7, coo_ns=19.1, coo_shared_ns=4.1),
    "rovers-p16e": dict(full_ns=48.9, bytes=3371, counts_ns=28.6, counts_shared_ns=13.9, coo_ns=39.8, coo_shared_ns=7.1),
    "blocks-p05m": dict(full_ns=20.2, bytes=1874, counts_ns=20.5, counts_shared_ns=19.5, coo_ns=18.9, coo_shared_ns=17.5),
    "sokoban-p16e": dict(full_ns=56.9, bytes=4312, counts_ns=50.5, counts_shared_ns=7.5, coo_ns=112.7, coo_shared_ns=14.9),
    "transport-p30m": dict(full_ns=58.7, bytes=12472, counts_ns=56.8, counts_shared_ns=15.3, coo_ns=90.8, coo_shared_ns=14.5),
    "miconic-p30m": dict(full_ns=81.5, bytes=17442, counts_ns=79.6, counts_shared_ns=16.8, coo_ns=128.2, coo_shared_ns=21.1),
    "blocks-p20m": dict(full_ns=39.5, bytes=4150, counts_ns=38.8, counts_shared_ns=37.0, coo_ns=39.1, coo_shared_ns=34.9),
}
# that standalone device encoder added to a captured step, N = 16384: ns per environment (the step alone, then the
# added time of the full contract with the static relations written for every graph, of COO, and of COO with shared
# statics)
STANDALONE_DEVICE_IN_STEP_NS = {
    "blocks-p01e": dict(step=4.3, full=9.8, coo=4.2, coo_shared=3.3),
    "childsnack-p16e": dict(step=6.0, full=30.3, coo=22.4, coo_shared=4.0),
    "rovers-p16e": dict(step=21.5, full=56.7, coo=49.2, coo_shared=14.7),
    "blocks-p05m": dict(step=22.5, full=22.7, coo=24.4, coo_shared=22.9),
    "sokoban-p16e": dict(step=15.1, full=64.5, coo=117.5, coo_shared=18.3),
    "transport-p30m": dict(step=118, full=64.0, coo=96.0, coo_shared=19.8),
    "miconic-p30m": dict(step=43.6, full=86.9, coo=132.6, coo_shared=25.6),
}

# ------------------------------------------------------------------------------------------------ gates as data
# The targets: `gate` is the pass threshold the harness checks, `goal` what the encoder aims for. A soft gate blocks only
# when a rerun on a quiet host fails as well.
TARGETS = [
    dict(id="cpu-core-one-thread", what="core lanes, one thread, batch of all rows", gate="<= 0.3 us + 1 ns per emitted int, per state",
         goal=None, soft=False),
    dict(id="cpu-action-lane", what="action lane", gate="<= 1 ns per int", goal=None, soft=False,
         needs="the action lane"),
    dict(id="cpu-derived-lanes", what="derived lanes", gate="cpu-core-one-thread plus the axiom evaluation, or cpu-core-one-thread alone with given derived words",
         goal=None, soft=False, needs="a task with derived predicates in the benchmark set"),
    dict(id="cpu-single-call", what="single call from Python", gate="<= 3 us per call", goal=None, soft=False),
    dict(id="cpu-eight-threads", what="8 threads against 1, batches of 4096 states or more", gate=">= 3.5x", goal=None,
         soft=True),
    dict(id="cpu-vs-reference-encoder", what="against the recorded reference encoder, batch of 256",
         gate=">= 20x faster", goal=None, soft=False),
    dict(id="cpu-relation-major", what="relation-major against graph-major, one thread", gate="<= 1.2x", goal=None, soft=False),
    dict(id="env-step", what="device env step alone against the recorded step times",
         gate="not more than 20% slower (a step more than 20% faster is printed as a warning)", goal=None,
         soft=True),
    dict(id="device-core", what="device flat core, graph-major, N >= 4096",
         gate="<= 1.25 x the standalone encoder's ns per graph x (bytes written / its bytes)",
         goal="<= bytes / 150 GB/s + 2 ns", soft=True),
    dict(id="device-relation-major", what="device relation-major against graph-major", gate="<= 1.5x", goal="<= 1.25x", soft=True),
    dict(id="device-in-step", what="encoding added to a captured env step, per environment",
         gate="<= 1.25 x the standalone encoder's added ns in a step x (bytes written / its bytes)",
         goal="<= bytes / 150 GB/s + 3 ns", soft=True),
    dict(id="device-launches", what="launches and sync", gate="<= 4 kernels per width class, no host sync, no allocation after "
         "reserve, capturable", goal=None, soft=True, needs="the device encoder's reserve and a profiler"),
    dict(id="device-vs-cpu", what="device against the CPU encoder at N >= 4096", gate=">= 10x one thread and >= 3x 8 threads",
         goal=">= 20x / >= 5x", soft=True),
    dict(id="device-scratch", what="scratch", gate="<= 16 B per graph + 4 B per cell + CUB temp", goal=None, soft=True,
         needs="the device encoder's bytes_needed"),
    dict(id="device-hgraph", what="HGraph view", gate="zero-copy, or <= 1.3x relation-major", goal="<= 1.1x", soft=True,
         needs="the HGraph view"),
    dict(id="device-simple", what="Simple on the device", gate="<= 2.5x the flat core", goal=None, soft=True,
         needs="the Simple encoding"),
    dict(id="device-shared-statics", what="shared statics", gate="<= 1.25 x the shared-statics e2e ns", goal=None, soft=True,
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


def describe(error):
    """One line for a failure: the type and the first line of the message."""
    message = str(error).splitlines()[0][:200] if str(error) else ""
    return f"{type(error).__name__}: {message}"


class Mode:
    """One timed workload.

    `run` does the work and synchronizes when it uses the GPU. The host clock times it, unless `reports_time`: then the
    run returns its own time in microseconds (as with CUDA events). `units` is the number of encoded
    states (or steps) per run, `unit` what the time is per. A mode with `open` builds what it times for itself: `open()`
    is a context manager that yields the run and releases everything on exit, so that one such mode is alive at a
    time. `stats` = (encoder, rows or a function that returns the rows) names the batch whose output size the record
    reports. `unavailable` marks a skip that is a property of the host or the build, not of the run."""

    def __init__(self, name, group, run=None, units=1, unit="us_per_state", skip=None, open=None, stats=None,
                 unavailable=False, reports_time=False, **fields):
        self.reports_time = reports_time
        self.name, self.group, self.run, self.units, self.unit, self.skip = name, group, run, units, unit, skip
        self.open, self.stats, self.unavailable = open, stats, unavailable
        self.fields = fields
        self.times = []
        self.error = None

    def skipped(self, reason, unavailable=False):
        self.skip, self.unavailable = reason, unavailable
        return self


def capacities_of(encoder, rows):
    """The capacity dict that fits the batch `rows` exactly."""
    batch = encoder.encode(rows)
    return dict(entities=int(batch.entity_offsets[-1]), tuples=int(batch.relation_offsets[-1]),
                args=int(batch.relation_arg_offsets[-1]))


def cpu_modes(rows, big, pool, encoders, wants):
    """The CPU modes of one task that `wants(name)` selects. `encoders` = {layout: (encoder, reason)}; `pool` is an
    rl.ThreadPool."""
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
            variants += [("cpu_batch_all_pool8", big, lambda e, d: e.encode(d, pool=pool))]
            if wants("cpu_batch_all_cap") and encoder is not None:
                caps = capacities_of(encoder, big)
                variants += [("cpu_batch_all_cap", big, lambda e, d, c=caps: e.encode(d, capacities=c))]
            elif wants("cpu_batch_all_cap"):
                variants += [("cpu_batch_all_cap", big, None)]
        for name, data, call in variants:
            if not wants(name):
                continue
            threads = THREADS if name.endswith(("_t8", "_pool8")) else 1
            mode = Mode(name, "cpu", units=len(data), layout=layout, threads=threads, stats=(encoder, data))
            if encoder is None:
                modes.append(mode.skipped(f"layout {layout} unavailable: {reason}", unavailable=True))
                continue
            mode.run = lambda e=encoder, d=data, c=call: c(e, d)
            modes.append(mode)
        if layout == "graph" and wants("cpu_single"):
            singles = [np.ascontiguousarray(r[None]) for r in tile(rows, SINGLE_CALLS)]
            mode = Mode("cpu_single", "cpu", units=len(singles), layout=layout, threads=1, stats=(encoder, batch256))
            if encoder is None:
                modes.append(mode.skipped(f"layout {layout} unavailable: {reason}", unavailable=True))
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

    def empty_cache(self):
        self.torch.cuda.empty_cache()


class DevicePathError(RuntimeError):
    """The build has a device encode path that the harness cannot drive."""


def device_path_exists(gpu):
    """Whether the build registers the device encode op (torch.ops.mymyr.encode_flat)."""
    try:
        gpu.torch.ops.mymyr.encode_flat
    except (AttributeError, RuntimeError):
        return False
    return True


def host_rows(states):
    """The state rows of a device tensor as a C-contiguous host array of u64 words."""
    rows = np.ascontiguousarray(states.cpu().numpy())
    return rows.view(np.uint64) if rows.dtype != np.uint64 else rows


def padded_capacities(capacities):
    """`capacities` with CAPACITY_HEADROOM on top, for rows that change between encodes."""
    return {k: math.ceil(v * CAPACITY_HEADROOM) for k, v in capacities.items()}


def probe_device_encode(gpu, encoder, states):
    """None when `encoder.encode` takes the device rows `states` (with measured `capacities=`); otherwise the reason it
    does not, starting with "not supported on the device". On a build that has a device encode path, a failed probe
    raises DevicePathError instead: the harness must drive that path, not skip it."""
    try:
        encoder.encode(states, capacities=capacities_of(encoder, host_rows(states)))
        gpu.sync()
    except Exception as error:  # noqa: BLE001
        if device_path_exists(gpu):
            raise DevicePathError(f"the device encode path exists but the harness cannot drive it: {describe(error)}") from error
        return f"not supported on the device ({describe(error)})"
    return None


class DeviceRun:
    """`FlatRelationEncoder.encode` on device rows with `capacities=` measured from the CPU encoding of those rows.
    The first encode runs at construction, outside any capture: whatever the encoder acquires for it stays with the
    encoder, so a later encode inside a capture takes nothing new. `close()` drops the encoder and the rows and waits for
    the device."""

    def __init__(self, gpu, encoder, states):
        self.gpu, self.encoder, self.states = gpu, encoder, states
        self.capacities = padded_capacities(capacities_of(encoder, host_rows(states)))
        self.encode()
        gpu.sync()

    def encode(self, states=None):
        return self.encoder.encode(self.states if states is None else states, capacities=self.capacities)

    def close(self):
        self.encoder = self.states = None
        self.gpu.sync()


def time_graph_replays(gpu, graph, replays):
    """Microseconds of `replays` back-to-back replays of a CUDA graph, between two timing events."""
    cuda = gpu.torch.cuda
    start, stop = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1e3


def replay_and_sync(gpu, graph):
    graph.replay()
    gpu.sync()


@contextlib.contextmanager
def encode_session(gpu, encoder, states, eager):
    """The run of one device encode of `states`: a CUDA graph of one encode captured on a side stream and replayed
    REPLAYS times per timed run (the time comes from CUDA events), or with `eager` one call and a synchronize."""
    run = DeviceRun(gpu, encoder, states)
    graph = output = None
    try:
        if eager:

            def timed():
                run.encode()
                gpu.sync()

            yield timed
        else:
            graph = gpu.torch.cuda.CUDAGraph()
            with gpu.torch.cuda.graph(graph, stream=gpu.torch.cuda.Stream()):
                output = run.encode()
            yield lambda: time_graph_replays(gpu, graph, REPLAYS)
    finally:
        graph = output = None
        run.close()


@contextlib.contextmanager
def encode_in_step_session(gpu, env, encoder):
    """The run of a captured environment step whose policy encodes the step's states."""
    run = DeviceRun(gpu, encoder, env.states)
    graph = None
    try:

        def policy(e):
            run.encode(e.states)
            return e.random_actions()

        graph = env.capture(STEPS, policy)
        yield lambda: replay_and_sync(gpu, graph)
    finally:
        graph = None
        run.close()


class Resources:
    """What the device modes keep alive: environments, their captured graphs and the modes that replay them.
    `release()` drops the modes' references first, then the captured graphs, then the environments, and waits for the
    device in between: a graph is destroyed before what it replays on, and nothing outlives the task."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.modes, self.graphs, self.envs = [], [], []

    def release(self):
        for mode in self.modes:
            mode.run = mode.open = mode.stats = None
        self.modes.clear()
        for owned in (self.graphs, self.envs):
            owned.clear()
            if self.gpu.ok:
                self.gpu.sync()
        if self.gpu.ok:
            self.gpu.empty_cache()


ENV_MODES = ("env_step_n{n}", "env_step_rand_n{n}", "env_step_encode_n{n}")
DEVICE_MODES = (("device_encode_graph", "graph", False), ("device_encode_rel", "relation", False),
                ("device_encode_graph_eager", "graph", True))


def device_modes(task, gpu, sizes, wants, encoders, resources):
    """The environment and device-encode modes of one task that `wants(name)` selects. Only what a selected mode needs
    is built. The device-encode modes take the largest environment's states. Environments, captured graphs and the
    modes that replay them are registered in `resources`."""
    names = {n: [pattern.format(n=n) for pattern in ENV_MODES] for n in sizes}
    selected_env = {n: [m for m in names[n] if wants(m)] for n in sizes}
    selected_device = [spec for spec in DEVICE_MODES if wants(spec[0])]
    largest = max(sizes) if sizes else None
    modes = []
    if not gpu.ok:
        for n in sizes:
            modes += [Mode(m, "env", unit="us_per_step", N=n).skipped(gpu.reason, True) for m in selected_env[n]]
        modes += [Mode(name, "device", layout=layout, timing="eager" if eager else "replay").skipped(gpu.reason, True)
                  for name, layout, eager in selected_device]
        return modes
    probe = {}

    def unsupported(env):
        """The reason the device encode is not available, or None; probed once per task."""
        if "reason" not in probe:
            encoder, reason = encoders["graph"]
            probe["reason"] = (f"layout graph unavailable: {reason}" if encoder is None
                               else probe_device_encode(gpu, encoder, env.states[:PROBE_STATES]))
        return probe["reason"]

    for n in sizes:
        need_device = n == largest and selected_device
        if not selected_env[n] and not need_device:
            continue
        try:
            env = gpu.rt.BatchedEnv(task, n, device=gpu.device, seed=1)
            for _ in range(5):
                env.step()
            gpu.sync()
            resources.envs.append(env)
            step = env.capture(STEPS) if names[n][0] in selected_env[n] else None
            rand = env.capture(STEPS, lambda e: e.random_actions()) if names[n][1] in selected_env[n] else None
        except Exception as error:  # noqa: BLE001
            reason = f"device environment unavailable: {str(error).splitlines()[0][:160]}"
            modes += [Mode(m, "env", unit="us_per_step", N=n, skip=reason) for m in selected_env[n]]
            if need_device:
                modes += [Mode(name, "device", layout=layout, timing="eager" if eager else "replay", skip=reason)
                          for name, layout, eager in selected_device]
            continue
        resources.graphs += [g for g in (step, rand) if g is not None]
        env_fields = dict(N=n, row_words=env.row_words)
        if step is not None:
            modes.append(Mode(names[n][0], "env", lambda g=step: replay_and_sync(gpu, g), STEPS, "us_per_step",
                              **env_fields))
        if rand is not None:
            modes.append(Mode(names[n][1], "env", lambda g=rand: replay_and_sync(gpu, g), STEPS, "us_per_step",
                              **env_fields))
        if names[n][2] in selected_env[n]:
            mode = Mode(names[n][2], "env", unit="us_per_step", units=STEPS, **env_fields)
            reason = unsupported(env)
            encoder = encoders["graph"][0]
            if reason is not None:
                mode.skipped(reason, unavailable=True)
            else:
                mode.stats = (encoder, lambda e=env: host_rows(e.states))
                mode.open = lambda e=env, enc=encoder: encode_in_step_session(gpu, e, enc)
            modes.append(mode)
        if need_device:
            for name, layout, eager in selected_device:
                fields = dict(layout=layout, N=n, timing="eager" if eager else "replay")
                mode = Mode(name, "device", units=n if eager else n * REPLAYS, reports_time=not eager, **fields)
                encoder, reason = encoders[layout]
                reason = f"layout {layout} unavailable: {reason}" if encoder is None else unsupported(env)
                if reason is not None:
                    mode.skipped(reason, unavailable=True)
                else:
                    mode.stats = (encoder, lambda e=env: host_rows(e.states))
                    mode.open = lambda e=env, enc=encoder, eg=eager: encode_session(gpu, enc, e.states, eg)
                modes.append(mode)
    resources.modes += modes
    return modes


# ------------------------------------------------------------------------------------------------ timing


def time_once(mode):
    """One timed run of `mode` (host clock, or the time the run reports); False after a failure."""
    t0 = time.perf_counter()
    try:
        reported = mode.run()
    except Exception as error:  # noqa: BLE001
        mode.error = describe(error)
        return False
    elapsed = reported if mode.reports_time else (time.perf_counter() - t0) * 1e6
    mode.times.append(elapsed / mode.units)
    return True


def warm_up(mode):
    try:
        for _ in range(2):
            mode.run()
    except Exception as error:  # noqa: BLE001
        mode.error = describe(error)


def time_rounds(modes, rounds):
    """Warm up each runnable mode, then run `rounds` interleaved rounds (the start mode rotates every round). The modes
    that build what they time (`open`) follow one after the other, each with its rounds, so that only one is alive."""
    live = [m for m in modes if m.run is not None and m.skip is None]
    for mode in live:
        warm_up(mode)
    live = [m for m in live if not m.error]
    for r in range(rounds):
        k = r % len(live) if live else 0
        for mode in live[k:] + live[:k]:
            if not mode.error:
                time_once(mode)
    for mode in modes:
        if mode.open is None or mode.skip is not None:
            continue
        try:
            with mode.open() as run:
                mode.run = run
                warm_up(mode)
                for _ in range(rounds if not mode.error else 0):
                    if not time_once(mode):
                        break
        except Exception as error:  # noqa: BLE001
            mode.error = describe(error)
        finally:
            mode.run = None


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
    """The records of one run, by task and mode. Tasks that could not be loaded appear in `task_skipped`."""

    def __init__(self, records):
        self.rec, self.task_skipped = {}, {}
        for r in records:
            if r["mode"] == "*":
                self.task_skipped[r["task"]] = r.get("reason")
            else:
                self.rec[(r["task"], r["mode"])] = r
        found = {r["task"] for r in records}
        self.tasks = sorted(found, key=lambda t: list(TASKS).index(t) if t in TASKS else 99)

    def get(self, task, mode, field="min"):
        r = self.rec.get((task, mode))
        return None if r is None or r["status"] != "ok" else r.get(field)

    def modes_of(self, task, prefix):
        """The records of `task` whose mode starts with `prefix`, any status."""
        return [r for (t, m), r in self.rec.items() if t == task and m.startswith(prefix)]


def outcome(status, reason=None, **fields):
    return dict(status=status, **({} if reason is None else dict(reason=reason)), **fields)


def verdict(ok):
    return "pass" if ok else "fail"


RANK = {"n/a": 0, "skipped": 1, "error": 2}


def unavailable(index, task, modes):
    """None when every mode of `task` was measured. Otherwise the outcome that stands for the worst missing one: an
    error, then a skip (missing data or a failed setup), then n/a (not selected, or unavailable on this host or build)."""
    worst = None
    for mode in modes:
        r = index.rec.get((task, mode))
        if r is None:
            if task in index.task_skipped:
                found = outcome("skipped", f"task skipped: {index.task_skipped[task]}")
            else:
                found = outcome("n/a", f"{mode} not selected")
        elif r["status"] == "ok":
            continue
        elif r["status"] == "error":
            found = outcome("error", f"{mode}: {r['reason']}")
        else:
            found = outcome("n/a" if r.get("unavailable") else "skipped", f"{mode}: {r['reason']}")
        if worst is None or RANK[found["status"]] > RANK[worst["status"]]:
            worst = found
    return worst


def gate_cpu_core(index, task):
    r = index.rec[(task, "cpu_batch_all")]
    limit = 0.3 + 1e-3 * r["ints_per_state"]
    return outcome(verdict(r["min"] <= limit), value=r["min"], limit=limit)


def gate_cpu_single_call(index, task):
    value = index.get(task, "cpu_single")
    return outcome(verdict(value <= 3.0), value=value, limit=3.0)


def gate_cpu_eight_threads(index, task):
    n = index.rec[(task, "cpu_batch_all")]["units"]
    if n < 4096:
        return outcome("n/a", f"a batch of {n} states, under 4096")
    speedup = index.get(task, "cpu_batch_all") / index.get(task, "cpu_batch_all_t8")
    return outcome(verdict(speedup >= 3.5), value=speedup, limit=3.5)


def gate_cpu_vs_reference(index, task):
    value = index.get(task, "cpu_batch256")
    limit = REFERENCE_FLAT_US[task][1] / 20
    return outcome(verdict(value <= limit), value=value, limit=limit)


def gate_cpu_relation_major(index, task):
    ratio = index.get(task, "cpu_batch_all_rel") / index.get(task, "cpu_batch_all")
    return outcome(verdict(ratio <= 1.2), value=ratio, limit=1.2)


def gate_env_step(index, task):
    """Every recorded size of the task that has a record: a step more than 20% slower fails, more than 20% faster is a
    warning (a much faster step can mean that the graph does less work)."""
    by_size, worst = {}, None
    for n, base in ENV_STEP_US[task].items():
        missing = unavailable(index, task, [f"env_step_n{n}"])
        if missing is not None:
            by_size[n] = missing
            continue
        value = index.get(task, f"env_step_n{n}")
        ratio = value / base
        by_size[n] = outcome(verdict(ratio <= 1.2), value=value, limit=base, ratio=ratio, faster=ratio < 0.8,
                             median_ratio=(index.get(task, f"env_step_n{n}", "median") or float("nan")) / base)
    statuses = {o["status"] for o in by_size.values()}
    for status in ("fail", "error", "skipped", "pass"):
        if status in statuses:
            shown = max((n for n, o in by_size.items() if o["status"] == status), default=None)
            return outcome(status, by_size[shown].get("reason"),
                           **{k: v for k, v in by_size[shown].items() if k not in ("status", "reason", "faster")},
                           faster=[n for n, o in by_size.items() if o.get("faster")], by_size=by_size)
    return outcome("n/a", next(iter(by_size.values()))["reason"], by_size=by_size)


def device_batch_too_small(index, task):
    """The n/a outcome when the device encode ran on under 4096 states, else None."""
    r = index.rec[(task, "device_encode_graph")]
    if r["N"] < 4096:
        return outcome("n/a", f"N = {r['N']}, under 4096")
    return None


def gate_device_core(index, task):
    small = device_batch_too_small(index, task)
    if small:
        return small
    r = index.rec[(task, "device_encode_graph")]
    proto = STANDALONE_DEVICE[task]
    b = r["bytes_per_state"]
    limit = 1.25 * proto["full_ns"] * b / proto["bytes"]
    return outcome(verdict(r["min"] * 1e3 <= limit), value=r["min"] * 1e3, limit=limit, goal=b / 150 + 2)


def gate_device_relation_major(index, task):
    ratio = index.get(task, "device_encode_rel") / index.get(task, "device_encode_graph")
    return outcome(verdict(ratio <= 1.5), value=ratio, limit=1.5)


def gate_device_vs_cpu(index, task):
    small = device_batch_too_small(index, task)
    if small:
        return small
    graph = index.get(task, "device_encode_graph")
    s1, s8 = index.get(task, "cpu_batch_all") / graph, index.get(task, "cpu_batch_all_t8") / graph
    return outcome(verdict(s1 >= 10 and s8 >= 3), value=[s1, s8], limit=[10, 3])


def gate_device_in_step(index, task):
    """The largest environment size with an encode-in-step record: the added time per environment of the encode."""
    sizes = [r["N"] for r in index.modes_of(task, "env_step_encode_n")]
    if not sizes:
        return unavailable(index, task, ["env_step_encode_n*"])
    n = max(sizes)
    missing = unavailable(index, task, [f"env_step_rand_n{n}", f"env_step_encode_n{n}"])
    if missing:
        return missing
    if n < 4096:
        return outcome("n/a", f"N = {n}, under 4096")
    enc, rand = index.rec[(task, f"env_step_encode_n{n}")], index.rec[(task, f"env_step_rand_n{n}")]
    added = enc["min_ns_per_env"] - rand["min_ns_per_env"]
    b = enc["bytes_per_state"]
    limit = 1.25 * STANDALONE_DEVICE_IN_STEP_NS[task]["full"] * b / STANDALONE_DEVICE[task]["bytes"]
    return outcome(verdict(added <= limit), value=added, limit=limit, goal=b / 150 + 3)


# per gate: (the reason it does not apply to a task or None, the modes it needs, its evaluation). The modes the gate
# derives itself (env-step, device-in-step) are named in the evaluation.
def _no_baseline(table):
    return lambda task: None if table.get(task) else f"no recorded baseline for {task}"


GATES = {
    "cpu-core-one-thread": (lambda task: None, ("cpu_batch_all",), gate_cpu_core),
    "cpu-single-call": (lambda task: None, ("cpu_single",), gate_cpu_single_call),
    "cpu-eight-threads": (lambda task: None, ("cpu_batch_all", "cpu_batch_all_t8"), gate_cpu_eight_threads),
    "cpu-vs-reference-encoder": (_no_baseline(REFERENCE_FLAT_US), ("cpu_batch256",), gate_cpu_vs_reference),
    "cpu-relation-major": (lambda task: None, ("cpu_batch_all", "cpu_batch_all_rel"), gate_cpu_relation_major),
    "env-step": (_no_baseline(ENV_STEP_US), None, gate_env_step),
    "device-core": (_no_baseline(STANDALONE_DEVICE), ("device_encode_graph",), gate_device_core),
    "device-relation-major": (lambda task: None, ("device_encode_graph", "device_encode_rel"), gate_device_relation_major),
    "device-in-step": (_no_baseline(STANDALONE_DEVICE_IN_STEP_NS), None, gate_device_in_step),
    "device-vs-cpu": (lambda task: None, ("device_encode_graph", "cpu_batch_all", "cpu_batch_all_t8"), gate_device_vs_cpu),
}


def gate_results(index, tasks):
    """{gate id: {task: outcome}} for the gates that have an evaluation. `tasks` are the tasks the run was asked to
    measure; every gate lists each of them. An outcome is pass or fail with its value and limit, error (a needed mode
    failed), skipped (a needed mode or the task was skipped), or n/a with the reason the gate does not apply."""
    out = {t["id"]: {} for t in TARGETS}
    for gate, (applies, modes, evaluate) in GATES.items():
        for task in tasks:
            reason = applies(task)
            if reason is not None:
                out[gate][task] = outcome("n/a", reason)
                continue
            missing = None if modes is None else unavailable(index, task, modes)
            out[gate][task] = missing or evaluate(index, task)
    return out


def gate_table(results):
    """(rows, text): one row per gate with its status: n/a (no task it applies to, with the reasons), pass, fail,
    error, or incomplete when a task it applies to was skipped or could not be evaluated (named, with k of n tasks
    evaluated). The result of a gate is never pass while a task it applies to is missing."""
    needs = {t["id"]: t.get("needs") for t in TARGETS}
    rows = []
    for target in TARGETS:
        per_task = results[target["id"]]
        applicable = {t: o for t, o in per_task.items() if o["status"] != "n/a"}
        row = dict(id=target["id"], what=target["what"])
        if not applicable:
            reasons = {}
            for task, o in per_task.items():
                reasons.setdefault(o["reason"], []).append(task)
            row.update(status="n/a", detail="; ".join(f"{reason} ({', '.join(tasks)})" for reason, tasks in reasons.items())
                       if reasons else "needs " + (needs[target["id"]] or "its modes"))
            if not reasons and needs[target["id"]] is None:
                row["detail"] = "no task was selected"
        else:
            measured = [t for t, o in applicable.items() if o["status"] in ("pass", "fail")]
            failed = [t for t in measured if applicable[t]["status"] == "fail"]
            errored = [t for t, o in applicable.items() if o["status"] == "error"]
            missing = [t for t, o in applicable.items() if o["status"] == "skipped"]
            status = "fail" if failed else "error" if errored else "incomplete" if missing else "pass"
            parts = [f"{len(measured) - len(failed)}/{len(measured)} pass"]
            if failed:
                parts.append("fail: " + ", ".join(failed))
            if errored:
                parts.append("error: " + ", ".join(f"{t} ({applicable[t]['reason']})" for t in errored))
            if missing:
                parts.append(f"incomplete ({len(measured)} of {len(applicable)}): missing "
                             + ", ".join(f"{t} ({applicable[t]['reason']})" for t in missing))
            if status in ("fail", "incomplete") and target["soft"] and failed:
                parts.append("soft: blocks only if a rerun on a quiet host fails as well")
            row.update(status=status, detail="; ".join(parts), measured=len(measured), applicable=len(applicable))
        if target["id"] == "env-step":
            faster = [f"{t}@{n} ({o['by_size'][n]['ratio']:.2f}x)" for t, o in per_task.items()
                      for n in o.get("faster", [])]
            if faster:
                row["warning"] = "faster than recorded by >20%: " + ", ".join(faster)
        rows.append(row)
    label = {r["id"]: (f"incomplete ({r['measured']} of {r['applicable']})" if r["status"] == "incomplete" else r["status"])
             for r in rows}
    width, status_width = max(len(r["what"]) for r in rows), max(len(v) for v in label.values())
    id_width = max(len(r["id"]) for r in rows)
    lines = [f"{r['id']:<{id_width}} {label[r['id']]:<{status_width}} {r['what']:<{width}}  {r['detail']}" for r in rows]
    lines += [f"{'warn':<{id_width}} env-step {r['warning']}" for r in rows if "warning" in r]
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
        device_encode_path=None, gpu=None)
    if gpu.ok:
        torch = gpu.torch
        env["device_encode_path"] = device_path_exists(gpu)
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


def group_of(name):
    return "cpu" if name.startswith("cpu") else "env" if name.startswith("env") else "device"


def wanted(name, selection):
    """Whether the mode `name` is selected by `--modes` (mode names and groups; None selects all)."""
    return selection is None or name in selection or group_of(name) in selection


def group_wanted(group, selection):
    return selection is None or group in selection or any(group_of(n) == group for n in selection)


def stats_of(mode, cache):
    """(bytes, ints) per state of the batch that `mode` timed, from its CPU encoding."""
    encoder, rows = mode.stats
    key = None
    if not callable(rows):
        key = (id(encoder), id(rows))
        if key in cache:
            return cache[key]
    batch = encoder.encode(rows() if callable(rows) else rows)
    nbytes, nints = batch_stats(batch)
    result = nbytes / batch.num_graphs, nints / batch.num_graphs
    if key is not None:
        cache[key] = result
    return result


def run_task(key, args, gpu, pool):
    """The records of one task."""
    task, reason = load_task(key, args.data)
    base = dict(task=key)
    if task is None:
        return [dict(**base, mode="*", status="skipped", reason=reason)]
    selection = args.modes

    def wants(name):
        return wanted(name, selection)

    info = dict(**base, words=task.words, objects=len(task.objects))
    encoders = {"graph": make_encoder(task), "relation": make_encoder(task, layout="relation")}
    modes = []
    if group_wanted("cpu", selection):
        rows = walk_rows(task, args.rows, args.seed)
        info["n_rows"] = len(rows)
        modes += cpu_modes(rows, tile(rows, max(len(rows), args.min_batch)), pool, encoders, wants)
    resources = Resources(gpu)
    modes += device_modes(task, gpu, args.env_sizes, wants, encoders, resources)
    time_rounds(modes, args.rounds)

    stats = {}
    records = []
    for mode in modes:
        record = dict(info, mode=mode.name, group=mode.group, unit=mode.unit, units=mode.units, **mode.fields)
        if mode.skip is not None:
            record.update(status="skipped", reason=mode.skip, **({"unavailable": True} if mode.unavailable else {}))
        elif mode.error:
            record.update(status="error", reason=mode.error)
        else:
            record.update(status="ok", **summarize(mode))
            if mode.stats is not None:
                record["bytes_per_state"], record["ints_per_state"] = stats_of(mode, stats)
        records.append(record)
    resources.release()
    return records


def baselines():
    return dict(
        env_step_us=ENV_STEP_US, earlier_cpu_us={k: dict(zip(("single", "batch256", "threads8"), v))
                                                  for k, v in EARLIER_CPU_US.items()},
        reference_flat_us={k: dict(zip(("single", "batch256"), v)) for k, v in REFERENCE_FLAT_US.items()},
        standalone_cpu_us={k: dict(zip(("threads1", "threads8"), v)) for k, v in STANDALONE_CPU_US.items()},
        standalone_device_ns_per_graph=STANDALONE_DEVICE, standalone_device_in_step_ns_per_env=STANDALONE_DEVICE_IN_STEP_NS)


def print_records(records):
    print(f"{'task':<16}{'mode':<28}{'unit':<12}{'min':>10}{'median':>10}  bytes/state")
    for r in records:
        if r["status"] == "ok":
            extra = f"{r['bytes_per_state']:.0f}" if "bytes_per_state" in r else ""
            print(f"{r['task']:<16}{r['mode']:<28}{r['unit']:<12}{r['min']:>10.3f}{r['median']:>10.3f}  {extra}")
        else:
            print(f"{r['task']:<16}{r['mode']:<28}{r['status']} ({r.get('reason', '')})")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks", help=f"comma-separated task keys (default: {','.join(TASKS)})")
    p.add_argument("--rows", type=int, default=4096, help="distinct rows per task (default 4096)")
    p.add_argument("--rounds", type=int, default=20, help="interleaved rounds, at least 1 (default 20)")
    p.add_argument("--modes", help="comma-separated mode names or groups (cpu, env, device); default all")
    p.add_argument("--json", metavar="PATH", help="write the result as JSON")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    p.add_argument("--data", metavar="DIR", help="directory with the <domain>-ipc task directories (default tests/data/ipc)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random walks")
    p.add_argument("--min-batch", type=int, default=MIN_BATCH, help="rows of the all-rows batches, tiled up to this")
    p.add_argument("--env-sizes", default=",".join(map(str, ENV_SIZES)), help="environment counts N")
    p.add_argument("--quick", action="store_true", help="a smoke configuration: one small task, few rows and rounds")
    args = p.parse_args(argv)
    if args.rounds < 1:
        p.error("--rounds must be at least 1")
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
    results = gate_results(Index(records), args.tasks)
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
