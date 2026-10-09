"""Datasets on the device through the Python layer (mymyr.cuda state spaces, mymyr.datasets with device=,
IW over task tables).

Checked against the CPU (mymyr.datasets and mymyr.search.iw):
- mymyr.datasets.state_space(task, device=...) equals the CPU space array by array (device arrays through torch, JAX
  and dlpack, and to_host()), for suite tasks and the fork's tasks with conditional effects, axioms, action costs and
  unsolvable states; statuses and outputs of generate_state_space;
- state_spaces over the table instance sets (one device pipeline; waves and tiny chunks) equal the CPU pool, the
  generalized state space from device results equals the CPU one, samplers over device spaces equal samplers over the
  host spaces;
- multi_iw and batched_iw1 over a TaskTable with task ids equal the per-instance device runs and mymyr.search.iw.

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_state_space.py
"""

import os
import pathlib

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import mymyr  # noqa: E402
import mymyr.search  # noqa: E402
from mymyr import datasets, rl  # noqa: E402
from conftest import ROOT, text_task, walk  # noqa: E402

mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
if not mc.available():
    pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)", allow_module_level=True)
torch = pytest.importorskip("torch")

FORK = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GEN = ROOT / "tests/data/table_instances"
BW = FORK / "ipc/blocksworld-ipc/train"
SETS = {  # as tests/cpp/rl/table_instance_sets.hpp (test_rl_table.py)
    "blocks": (BW / "domain.pddl", [BW / f"p{k}.pddl" for k in (13, 25, 37, 49, 61)]),
    "gripper": (FORK / "gripper/domain.pddl",
                [FORK / "gripper/test_problem4.pddl"] + [GEN / f"gripper/gripper-{k}.pddl" for k in (10, 20, 40, 80)]),
    "miconic-simpleadl": (FORK / "miconic-simpleadl/domain.pddl",
                          [FORK / "miconic-simpleadl/test_problem.pddl"]
                          + [GEN / f"miconic-simpleadl/simple-{k}.pddl" for k in ("f6-p4-q", "f12-p8-c", "f24-p16-q")]),
}
TEXT = ["depot__p02", "gripper__prob05", "miconic-simpleadl__s10-2", "openstacks-opt08-adl__p03", "philosophers__p03-phil4"]
FORK_TASKS = [("barman", "test_problem.pddl"), ("miconic-fulladl", "test_problem.pddl"), ("spanner", "p15-easy.pddl"),
              ("gripper", "p-2-0.pddl")]
KEYS = ["state_words", "forward_offsets", "forward_targets", "label_schemas", "label_bindings", "backward_offsets",
        "backward_sources", "backward_edges", "unit_goal_distances", "cost_goal_distances", "costs", "goal", "unsolvable",
        "alive"]


@pytest.fixture(scope="module")
def ctx():
    c = mc.Context(0, max_bytes=3 << 30)
    yield c
    c.synchronize()


def needs_frontend():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")


def fork_task(d, p):
    needs_frontend()
    dom, prob = FORK / d / "domain.pddl", FORK / d / p
    if not dom.is_file() or not prob.is_file():
        pytest.skip(f"fork data missing: {prob}")
    return mymyr.Task.from_pddl(dom, prob, atoms="frozen")


@pytest.fixture(scope="module")
def sets():
    needs_frontend()
    out = {}
    for name, (dom, probs) in SETS.items():
        if not dom.is_file() or not all(p.is_file() for p in probs):
            pytest.skip(f"the table instances of {name} are missing (set MYMYR_FORK_DATA)")
        out[name] = [mymyr.Task.from_pddl(dom, p, atoms="frozen") for p in probs]
    return out


def host_of(a):
    """A torch CUDA tensor (or a NumPy array) as NumPy; 64-bit words compared as uint64."""
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    return a.view(np.uint64) if a.dtype == np.int64 else a


def assert_same_arrays(dev, cpu):
    assert set(dev) == set(cpu)
    for k in dev:
        d, c = host_of(dev[k]), np.asarray(cpu[k])
        assert d.shape == c.shape, k
        assert d.dtype == c.dtype, k
        assert np.array_equal(d, c), k


def assert_same_space(d, c):
    """A DeviceStateSpace against the CPU StateSpace: properties, device arrays (torch) and to_host()."""
    assert (d.num_states, d.num_transitions, d.words, d.has_labels, d.label_width, d.unit_costs) == (
        c.num_states, c.num_transitions, c.words, c.has_labels, c.label_width, c.unit_costs)
    assert (d.num_goal_states, d.num_unsolvable_states, d.max_goal_distance, d.layers) == (
        c.num_goal_states, c.num_unsolvable_states, c.max_goal_distance, c.layers)
    assert d.initial_state_id == 0 and len(d) == c.num_states
    ca = c.arrays()
    assert_same_arrays(d.arrays("torch"), ca)
    h = d.to_host()
    assert isinstance(h, datasets.StateSpace)
    assert_same_arrays(h.arrays(), ca)
    if c.num_states:
        last = c.num_states - 1
        assert d.state(0) == c.state(0) and d.state(last) == c.state(last)
        assert d.state(last).task is d.task


def cpu_space(task, **kw):
    kw.setdefault("remove_if_unsolvable", False)
    return datasets.state_space(task, threads=4, **kw)


# ------------------------------------------------------------------------------------------------ single tasks

@pytest.mark.parametrize("name", TEXT)
def test_state_space_equals_cpu_text(ctx, name):
    task = text_task(name, atoms="frozen")
    c = cpu_space(task, max_states=60000)
    d = datasets.state_space(task, device=ctx, remove_if_unsolvable=False, max_states=60000)
    assert (c is None) == (d is None)
    if c is not None:
        assert isinstance(d, mc.DeviceStateSpace)
        assert_same_space(d, c)


@pytest.mark.parametrize("case", FORK_TASKS, ids=lambda c: c[0])
def test_state_space_equals_cpu_fork(ctx, case):
    task = fork_task(*case)
    for remove in (False, True):
        c = cpu_space(task, remove_if_unsolvable=remove)
        d = datasets.state_space(task, device=ctx, remove_if_unsolvable=remove)
        assert (c is None) == (d is None)
        if c is not None:
            assert_same_space(d, c)
    # a device ordinal takes the task's default context; labels off; tiny chunks
    c = cpu_space(task, labels=False)
    d = mc.state_space(task, device=0, remove_if_unsolvable=False, labels=False, chunk_states=7)
    assert_same_space(d, c)


def test_generation_outputs_and_statuses(ctx):
    task = fork_task("barman", "test_problem.pddl")
    c = cpu_space(task)
    for output in ("device", "host", "both"):
        r = mc.generate_state_space(task, ctx=ctx, output=output, remove_if_unsolvable=False)
        assert r and r.status == datasets.Status.OK and r.states == c.num_states and r.seconds >= 0
        assert (r.space is not None) == (output != "host") and (r.host is not None) == (output != "device")
        if r.host is not None:
            assert_same_arrays(r.host.arrays(), c.arrays())
            assert r.host.task is task
        assert r.stats["states"] == c.num_states and r.stats["waves"] == 1
    small = mc.generate_state_space(task, ctx=ctx, max_states=10)
    assert small.status == datasets.Status.OUT_OF_STATES and not small and small.space is None
    assert datasets.generate(task, max_states=10).status == datasets.Status.OUT_OF_STATES
    with pytest.raises(ValueError, match="output"):
        mc.generate_state_space(task, ctx=ctx, output="disk")
    with pytest.raises(ValueError, match="differs"):
        mc.state_space(task, ctx=ctx, device=1)


def test_numeric_state_spaces_equal_cpu(ctx):
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    expected = datasets.state_space(task, remove_if_unsolvable=False)
    actual = mc.state_space(task, ctx=ctx, remove_if_unsolvable=False)
    assert actual.numeric_words == task.numeric_words
    assert actual.row_words == actual.words + actual.numeric_words
    assert actual.state(0) == task.initial_state
    assert actual.num_states == expected.num_states
    for name, rows in expected.arrays().items():
        assert np.array_equal(actual.to_host().arrays()[name], rows)
    assert len(datasets.state_spaces([task], device=ctx)) == 1


def test_device_arrays_frameworks_and_streams(ctx):
    task = fork_task("barman", "test_problem.pddl")
    c = cpu_space(task)
    d = datasets.state_space(task, device=ctx, remove_if_unsolvable=False)
    ca = c.arrays()
    # dlpack (None): DLArrays, taken by torch zero-copy on a side stream
    raw = d.arrays()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        t = {k: torch.from_dlpack(v) for k, v in raw.items()}
        sums = {k: v.double().sum() for k, v in t.items() if v.dtype != torch.bool}
    s.synchronize()
    assert_same_arrays(t, ca)
    assert float(sums["costs"]) == float(np.sum(ca["costs"]))
    assert t["costs"].dtype == torch.float64 and t["cost_goal_distances"].dtype == torch.float64
    assert t["forward_targets"].device.type == "cuda"
    with pytest.raises(TypeError, match="NumPy"):
        d.arrays("numpy")
    # the arrays keep the space's memory alive
    keep = d.arrays("torch")["unit_goal_distances"]
    del d, raw, t
    ctx.synchronize()
    assert np.array_equal(keep.cpu().numpy(), ca["unit_goal_distances"])
    jax = pytest.importorskip("jax")
    if not any(x.platform == "gpu" for x in jax.devices()):
        pytest.skip("JAX sees no GPU (JAX_PLATFORMS=cuda,cpu)")
    d = datasets.state_space(task, device=ctx, remove_if_unsolvable=False)
    ja = d.arrays("jax")
    assert np.array_equal(np.asarray(ja["forward_targets"]), ca["forward_targets"])
    # 64-bit integers reach JAX as uint32 pairs (little-endian)
    assert np.array_equal(np.asarray(ja["forward_offsets"]).view(np.uint64), ca["forward_offsets"])


# ------------------------------------------------------------------------------------------------ tables

@pytest.mark.parametrize("name", list(SETS))
def test_state_spaces_equal_cpu_pool(ctx, sets, name):
    tasks = sets[name]
    table = rl.TaskTable(tasks)
    cpu = datasets.state_spaces(tasks, max_states=20000, remove_if_unsolvable=False)
    assert len(cpu) == len(tasks)
    runs = [
        datasets.state_spaces(table, device=ctx, max_states=20000, remove_if_unsolvable=False),
        datasets.state_spaces(tasks, device=0, max_states=20000, remove_if_unsolvable=False, wave_instances=2),
        mc.state_spaces(table, ctx=ctx, max_states=20000, remove_if_unsolvable=False, chunk_states=777, wave_states=3000),
    ]
    for dev in runs:
        assert len(dev) == len(tasks)
        for i, (d, c) in enumerate(zip(dev, cpu)):
            assert (d is None) == (c is None), i
            if c is not None:
                assert d.task is tasks[i]
                assert_same_space(d, c)
    results = mc.generate_state_spaces(table, ctx=ctx, output="host", max_states=20000, remove_if_unsolvable=False,
                                       wave_instances=2)
    assert results[0].stats["waves"] >= 2
    for r, c in zip(results, cpu):
        assert (r.host is None) == (c is None) and r.space is None
        if c is not None:
            assert_same_arrays(r.host.arrays(), c.arrays())
        else:
            assert r.status == datasets.Status.OUT_OF_STATES


def test_generalized_state_space_from_device(ctx, sets):
    tasks = [fork_task("gripper", p) for p in ("p-1-0.pddl", "p-2-0.pddl", "test_problem2.pddl", "test_problem4.pddl")]
    tasks.append(sets["gripper"][2])  # fails at max_states: skipped, as on the CPU
    cpu = datasets.generalized_state_space(tasks, remove_if_unsolvable=False, max_states=30000)
    dev = datasets.generalized_state_space(rl.TaskTable(tasks), device=ctx, remove_if_unsolvable=False, max_states=30000)
    assert len(dev.spaces) == len(cpu.spaces) == 4
    assert (dev.num_vertices, dev.num_edges) == (cpu.num_vertices, cpu.num_edges)
    for k, v in cpu.arrays().items():
        assert np.array_equal(dev.arrays()[k], v), k
    spaces = [s.to_host() for s in datasets.state_spaces(tasks, device=ctx, remove_if_unsolvable=False, max_states=30000)
              if s is not None]
    by_hand = datasets.GeneralizedStateSpace(datasets.sorted_by_size(spaces))
    assert np.array_equal(by_hand.arrays()["forward_targets"], cpu.arrays()["forward_targets"])


def test_samplers_over_device_spaces(ctx):
    task = fork_task("spanner", "p15-easy.pddl")
    c = cpu_space(task)
    d = datasets.state_space(task, device=ctx, remove_if_unsolvable=False)
    sd, sc = datasets.StateSpaceSampler(d, seed=5), datasets.StateSpaceSampler(c, seed=5)
    assert sd.space is d and sc.space is c
    assert (sd.num_states, sd.num_dead_end_states, sd.num_alive_states, sd.max_steps_to_goal) == (
        sc.num_states, sc.num_dead_end_states, sc.num_alive_states, sc.max_steps_to_goal)
    assert np.array_equal(sd.sample_states(500), sc.sample_states(500))
    assert np.array_equal(sd.sample_states_n_steps_from_goal(3, 100), sc.sample_states_n_steps_from_goal(3, 100))
    assert np.array_equal(sd.sample_dead_end_states(50), sc.sample_dead_end_states(50))
    assert sd.sample_state() == sc.sample_state()
    ids = sd.states_n_steps_from_goal(2)
    del sd
    assert np.array_equal(ids, sc.states_n_steps_from_goal(2))  # the view keeps the sampler's lists alive
    with pytest.raises(TypeError, match="StateSpace"):
        datasets.StateSpaceSampler(task)


# ------------------------------------------------------------------------------------------------ IW over tables

def starts_of(tasks, per):
    starts, ids = [], []
    for i, t in enumerate(tasks):
        w = walk(t, steps=per, seed=i, walks=1)[:per]
        starts += w
        ids += [i] * len(w)
    order = np.random.default_rng(3).permutation(len(starts))  # instances interleaved
    return [starts[k] for k in order], [ids[k] for k in order]


def iw_record(b, i):
    return (b.status[i], [str(a) for a in b.plan(i)], b.goal_state(i), b.cost(i),
            [(p.arity, p.status, p.expanded, p.generated) for p in b.passes(i)])


@pytest.mark.parametrize("name", ["gripper", "blocks"])
def test_multi_iw_over_tables(ctx, sets, name):
    tasks = sets[name][:3]
    table = rl.TaskTable(tasks)
    starts, ids = starts_of(tasks, 5)
    for arity in (1, 2):
        b = mc.multi_iw(table, starts, task_ids=ids, ctx=ctx, max_arity=arity, max_states=20000)
        assert len(b) == len(starts) and b.task_ids == ids
        for i, (s, k) in enumerate(zip(starts, ids)):
            one = mc.multi_iw(tasks[k], [s], ctx=ctx, max_arity=arity, max_states=20000)
            assert iw_record(b, i) == iw_record(one, 0)
            cpu = mymyr.search.iw(tasks[k], max_arity=arity, start=s, max_states=20000)
            assert b.status[i] == cpu.status and [str(a) for a in b.plan(i)] == [str(a) for a in cpu.plan]
            if b.solved[i]:
                assert b.goal_state(i).task is tasks[k] and b.plan(i)[0].task is tasks[k] if b.plan(i) else True
    # per-search goals: another start's atoms of the same instance
    goals = []
    for i, (s, k) in enumerate(zip(starts, ids)):
        other = next(t for t, kk in zip(starts[i + 1:] + starts[:i], ids[i + 1:] + ids[:i]) if kk == k)
        fresh = [a for a in other.atom_slots() if a not in set(s.atom_slots())]
        goals.append((fresh[:2], []))
    b = mc.multi_iw(table, starts, task_ids=np.asarray(ids, np.int32), ctx=ctx, goals=goals, max_arity=1)
    for i, (s, k) in enumerate(zip(starts, ids)):
        one = mc.multi_iw(tasks[k], [s], ctx=ctx, goals=[goals[i]], max_arity=1)
        assert iw_record(b, i) == iw_record(one, 0)


def test_batched_iw1_over_tables(ctx, sets):
    tasks = sets["gripper"][:3]
    table = rl.TaskTable(tasks)
    starts, ids = starts_of(tasks, 6)
    W = table.words
    rows = np.zeros((len(starts), W), np.uint64)
    for i, s in enumerate(starts):
        w = s.words
        rows[i, :len(w)] = w
    host = mc.batched_iw1(table, starts, task_ids=ids, ctx=ctx)
    dev_rows = torch.from_numpy(rows.view(np.int64)).cuda()
    dev_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    for tid in (dev_ids, ids):
        dev = mc.batched_iw1(table, dev_rows, task_ids=tid, ctx=ctx, stream=torch.cuda.current_stream())
        for i in range(len(starts)):
            assert iw_record(dev, i) == iw_record(host, i)
    for i, (s, k) in enumerate(zip(starts, ids)):
        one = mc.batched_iw1(tasks[k], [s], ctx=ctx)
        assert iw_record(host, i) == iw_record(one, 0)


def test_iw_over_tables_errors(ctx, sets):
    tasks = sets["gripper"][:2]
    table = rl.TaskTable(tasks)
    s0, s1 = tasks[0].initial_state, tasks[1].initial_state
    with pytest.raises(ValueError, match="task_ids"):
        mc.multi_iw(table, [s0, s1], ctx=ctx)
    with pytest.raises(ValueError, match="outside"):
        mc.multi_iw(table, [s0, s1], task_ids=[0, 2], ctx=ctx)
    with pytest.raises(ValueError, match="not a state of instance"):
        mc.multi_iw(table, [s0, s1], task_ids=[1, 0], ctx=ctx)
    with pytest.raises(ValueError, match="TaskTable"):
        mc.multi_iw(tasks[0], [s0], task_ids=[0], ctx=ctx)
    one = mc.multi_iw(rl.TaskTable(tasks[:1]), [s0], ctx=ctx)  # a table of one needs no ids
    assert one.task_ids == [0]
