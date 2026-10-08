"""mymyr.cuda: batched device heuristics, device A* and GBFS. Heuristic.evaluate against mymyr.search.Heuristic
(h_max, h_add, h²) and against Heuristic.reference (the h_FF/set-additive supporter rule) on walk states, from host states and from torch
CUDA rows read in place (uint32 values on the caller's stream); astar / gbfs at batch 1 against mymyr.search.astar /
gbfs (statistics and plan), larger batches against the optimal cost with replayed plans; refusals.

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_heuristics.py
"""

import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import mymyr  # noqa: E402
import mymyr.search  # noqa: E402
from conftest import ROOT, SMALL_TASKS, text_task, walk  # noqa: E402

mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
if not mc.available():
    pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)", allow_module_level=True)


@pytest.fixture(scope="module")
def ctx():
    c = mc.Context(0, max_bytes=2 << 30)
    yield c
    c.synchronize()


def replay(task, plan, start=None):
    s = task.initial_state if start is None else start
    for a in plan:
        assert task.is_applicable(s, a)
        s = task.apply(s, a)
    return s


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("kind", ["max", "add", "ff", "h2", "set_additive"])
def test_heuristic_equals_cpu(ctx, name, kind):
    task = text_task(name)
    states = walk(task, steps=25, seed=5, walks=3)
    h = mc.Heuristic(task, kind, ctx=ctx)
    values = h.evaluate(states)
    assert isinstance(values, np.ndarray) and values.dtype == np.float64 and values.shape == (len(states),)
    ref = [h.reference(s) for s in states]
    assert list(values) == ref
    if kind not in ("ff", "set_additive"):
        cpu = mymyr.search.Heuristic(task, kind)
        assert ref == [cpu(s) for s in states]
    assert h.kind == kind
    assert h.stats["evaluations"] >= len(states) and h.stats["operators"] > 0
    # the launch configuration changes no value
    for kw in ({"variant": "frontier"}, {"warp_groups": True}, {"threads": 64, "force_global": True}, {"max_blocks": 1}):
        assert list(mc.Heuristic(task, kind, ctx=ctx, **kw).evaluate(states)) == ref, kw


def test_heuristic_real_costs_and_one_state(ctx):
    task = text_task("depot__p02")
    s = walk(task, steps=10, seed=2, walks=1)[-1]
    for kind in ("max", "add", "h2", "set_additive"):
        h = mc.Heuristic(task, kind, ctx=ctx, costs="real")
        assert list(h.evaluate(s)) == [h.reference(s)]


@pytest.mark.parametrize("kind", ["ff", "h2", "set_additive"])
def test_heuristic_from_torch_rows(ctx, kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=30, seed=7, walks=4)
    W = max(task.words, max(s.num_words for s in states)) + 1  # padded rows
    rows = np.zeros((len(states), W), dtype=np.uint64)
    for i, s in enumerate(states):
        rows[i, : s.num_words] = s.words
    h = mc.Heuristic(task, kind, ctx=ctx)
    host = h.evaluate(states)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dev = torch.from_numpy(rows.view(np.int64)).cuda()
        out = h.evaluate(dev[:, : W - 1], stream=stream)
        assert isinstance(out, torch.Tensor) and out.is_cuda and out.dtype == torch.uint32 and out.shape == (len(states),)
        got = out.to(torch.int64).cpu().numpy()
    assert [float(v) if v != mc.Heuristic.DEAD_END else math.inf for v in got] == list(host)
    # the default stream of the context
    torch.cuda.synchronize()
    again = torch.from_dlpack(h.evaluate(dev)).to(torch.int64).cpu().numpy()
    assert list(again) == list(got)


def test_heuristic_dead_ends(ctx):
    task = text_task("philosophers__p03-phil4")
    states = walk(task, steps=40, seed=1, walks=6)
    h = mc.Heuristic(task, "max", ctx=ctx)
    cpu = mymyr.search.Heuristic(task, "max")
    assert list(h.evaluate(states)) == [cpu(s) for s in states]
    assert mc.Heuristic.DEAD_END == 0xFFFFFFFF


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "depot__p02"])
@pytest.mark.parametrize("h", ["blind", "max"])
def test_astar_batch_one_is_the_cpu_search(ctx, name, h):
    task = text_task(name)
    cpu = mymyr.search.astar(task, heuristic=h, max_expanded=20000)
    dev = mc.astar(task, heuristic=h, ctx=ctx, batch=1, max_expanded=20000)
    assert dev.status == cpu.status and dev.solved == cpu.solved
    assert (dev.stats.expanded, dev.stats.generated, dev.stats.states) == (
        cpu.stats.expanded,
        cpu.stats.generated,
        cpu.stats.states,
    )
    assert [str(a) for a in dev.plan] == [str(a) for a in cpu.plan] and dev.cost == cpu.cost
    assert dev.evaluations == cpu.evaluations and dev.initial_h == cpu.initial_h
    assert dev.algorithm == "astar_eager (device)"
    assert dev.device["steps"] > 0 and dev.device["popped"] >= dev.stats.expanded


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1"])
def test_astar_batches_find_optimal_plans(ctx, name):
    task = text_task(name)
    cpu = mymyr.search.astar(task, heuristic="max")
    assert cpu.solved
    for batch, single in ((64, True), (1000, True), (1000, False)):
        dev = mc.astar(task, heuristic="max", ctx=ctx, batch=batch, single_bucket=single)
        assert dev.solved and dev.cost == cpu.cost, (batch, single)
        assert replay(task, dev.plan) == dev.goal_state
        assert task.is_goal(dev.goal_state)
        assert dev.device["max_batch"] <= batch
        again = mc.astar(task, heuristic="max", ctx=ctx, batch=batch, single_bucket=single, chunk_states=97)
        assert [str(a) for a in again.plan] == [str(a) for a in dev.plan]
        assert again.stats.expanded == dev.stats.expanded and again.stats.generated == dev.stats.generated


def test_gbfs(ctx):
    task = text_task("depot__p02")
    cpu = mymyr.search.gbfs(task, heuristic="add")
    one = mc.gbfs(task, heuristic="add", ctx=ctx, batch=1)
    assert [str(a) for a in one.plan] == [str(a) for a in cpu.plan]
    assert (one.stats.expanded, one.stats.generated) == (cpu.stats.expanded, cpu.stats.generated)
    many = mc.gbfs(task, ctx=ctx, batch=100)  # h_FF
    assert many.solved and many.algorithm == "gbfs_eager (device)"
    assert replay(task, many.plan) == many.goal_state


def test_start_and_budgets(ctx):
    task = text_task("blocks__probBLOCKS-8-0")
    start = walk(task, steps=6, seed=4, walks=1)[-1]
    cpu = mymyr.search.astar(task, heuristic="max", start=start)
    dev = mc.astar(task, heuristic="max", ctx=ctx, start=start, batch=100)
    assert dev.cost == cpu.cost and replay(task, dev.plan, start) == dev.goal_state
    cut = mc.astar(task, heuristic="blind", ctx=ctx, batch=1, max_expanded=50)
    assert cut.status == mymyr.search.astar(task, heuristic="blind", max_expanded=50).status
    assert cut.stats.expanded == 50 and not cut.solved
    assert mc.astar(task, ctx=ctx, max_seconds=0).status == mymyr.search.Status.OUT_OF_TIME


def test_errors(ctx):
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    assert mc.Heuristic(numeric, "max", ctx=ctx).evaluate(numeric.initial_state).shape == (1,)
    assert mc.astar(numeric, ctx=ctx, max_expanded=10).status == mymyr.search.astar(numeric, heuristic="max", max_expanded=10).status
    task = text_task("depot__p02")
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError):
        mc.Heuristic(task, "goal_count", ctx=ctx)
    with pytest.raises(ValueError, match="heuristic"):
        mc.astar(task, heuristic="nope", ctx=ctx)
    with pytest.raises(ValueError, match="costs"):
        mc.Heuristic(task, "max", ctx=ctx, costs="free")
    with pytest.raises(ValueError, match="variant"):
        mc.Heuristic(task, "max", ctx=ctx, variant="fast")
    with pytest.raises(ValueError, match="batch"):
        mc.astar(task, ctx=ctx, batch=0)
    with pytest.raises(ValueError, match="another task"):
        mc.astar(task, ctx=ctx, start=other.initial_state)
    with pytest.raises(ValueError, match="another task"):
        mc.Heuristic(task, "max", ctx=ctx).reference(other.initial_state)
    with pytest.raises(ValueError):
        mc.Heuristic(task, "ff", ctx=ctx, max_operators=10)
    assert mc.Heuristic(task, "max", ctx=ctx).evaluate([]).shape == (0,)


@pytest.mark.parametrize("kind", ["h2", "set_additive"])
@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "cs-counters"])
def test_pair_heuristics_search_plans(ctx, kind, name):
    task = (
        mymyr.Task.from_text(ROOT / "tests/data/numeric_tasks" / f"{name}.txt")
        if name == "cs-counters"
        else text_task(name)
    )
    cpu = mymyr.search.astar(task, heuristic="h2")
    assert cpu.solved
    for batch in (1, 64):
        for search in (mc.astar, mc.gbfs):
            result = search(task, heuristic=kind, ctx=ctx, batch=batch, max_expanded=100000)
            assert result.solved and task.is_goal(replay(task, result.plan))
            if search is mc.astar and kind == "h2":
                assert result.cost == cpu.cost


def test_pair_heuristic_aliases_and_scratch_budget(ctx):
    task = text_task("depot__p02")
    expected = mc.Heuristic(task, "set_additive", ctx=ctx).evaluate(task.initial_state)
    for kind in ("hsa", "setadd"):
        h = mc.Heuristic(task, kind, ctx=ctx)
        assert h.kind == "set_additive" and np.array_equal(h.evaluate(task.initial_state), expected)
    base = mc.Heuristic(task, "h2", ctx=ctx)
    size = base.stats["group_bytes"]
    with pytest.raises(ValueError, match=r"propositions.*limit is"):
        mc.Heuristic(task, "h2", ctx=ctx, max_scratch_bytes=size - 1)
    one = mc.Heuristic(task, "h2", ctx=ctx, max_scratch_bytes=size)
    assert one.stats["blocks"] == 1 and not one.stats["shared"]
    assert np.array_equal(one.evaluate(task.initial_state), base.evaluate(task.initial_state))


def test_h2_proposition_limit_names_the_size_and_limit(ctx, tmp_path):
    n = 8192
    path = tmp_path / "initial-propositions.txt"
    path.write_text(
        f"O {n}\nP 1\nF 1 p\nSI 0\nFI {n}\n"
        + "".join(f"0 {i}\n" for i in range(n))
        + "G 0\nA 0\nX 0\n"
    )
    task = mymyr.Task.from_text(str(path))
    with pytest.raises(ValueError) as caught:
        mc.Heuristic(task, "h2", ctx=ctx)
    assert "8192" in str(caught.value) and "8191" in str(caught.value)
