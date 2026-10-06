"""mymyr.cuda: many IW searches at once on the device. multi_iw and batched_iw1 against mymyr.search.iw from the
same starts (status, plan, goal state, per-pass counts, cost), with per-search goals and budgets; batched_iw1 from
torch CUDA rows read in place; rollouts are deterministic and independent of the launch configuration (their equality
with the CPU rollouts is the C++ test's, tests/cuda/test_device_iw.cpp); numeric tasks and bad arguments raise.

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_iw.py
"""

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


def walk_goals(starts, atoms):
    """Per start: up to `atoms` fluent slots of another walk state that the start lacks (positive), none negative."""
    goals = []
    for i, s in enumerate(starts):
        t = starts[(7 * i + 3) % len(starts)]
        have = set(s.atom_slots())
        fresh = [a for a in t.atom_slots() if a not in have]
        goals.append((fresh[:: max(1, len(fresh) // atoms)][:atoms], []))
    return goals


def assert_equal(batch, i, cpu):
    assert batch.status[i] == cpu.status
    assert batch.solved[i] == cpu.solved
    plan = batch.plan(i)
    assert [str(a) for a in plan] == [str(a) for a in cpu.plan]
    assert batch.plan_length[i] == (len(cpu.plan) if cpu.solved else -1)
    assert batch.effective_width[i] == cpu.effective_width
    g = batch.goal_state(i)
    assert (g is None) == (cpu.goal_state is None)
    if g is not None:
        assert g == cpu.goal_state
    dev_passes, cpu_passes = batch.passes(i), cpu.passes
    assert [(p.arity, p.status, p.expanded, p.generated, p.generated_in_tree, p.skipped, p.placeholder) for p in dev_passes] == [
        (p.arity, p.status, p.expanded, p.generated, p.generated_in_tree, p.skipped, p.placeholder) for p in cpu_passes
    ]
    assert batch.expanded[i] == sum(p.expanded for p in cpu_passes)
    assert batch.generated[i] == sum(p.generated for p in cpu_passes)
    assert batch.cost(i) == cpu.cost


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("arity", [1, 2])
def test_multi_iw_equals_cpu_iw(ctx, name, arity):
    task = text_task(name)
    starts = walk(task, steps=12, seed=3, walks=2)
    for goals in (None, walk_goals(starts, 2)):
        batch = mc.multi_iw(task, starts, ctx=ctx, goals=goals, max_arity=arity, max_states=3000)
        assert len(batch) == len(starts)
        for i, s in enumerate(starts):
            kw = {} if goals is None else {"goal": [{"positive": list(goals[i][0]), "negative": []}]}
            cpu = mymyr.search.iw(task, max_arity=arity, start=s, max_states=3000, **kw)
            assert_equal(batch, i, cpu)
        assert batch.stats["candidates"] > 0
        assert batch.stats["loop_handoffs"] == 0  # (a hand-off needs a row over emit_budget: 2^24 tuples by default)


def test_multi_iw_launch_configuration_and_budgets(ctx):
    task = text_task("depot__p02")
    starts = walk(task, steps=15, seed=5, walks=2)
    goals = walk_goals(starts, 1)
    ref = mc.multi_iw(task, starts, ctx=ctx, goals=goals)
    for kw in ({"max_searches": 1}, {"max_searches": 5, "chunk_states": 3}, {"chunk_states": 1}):
        other = mc.multi_iw(task, starts, ctx=ctx, goals=goals, **kw)
        assert other.status == ref.status and other.plan_length == ref.plan_length
        assert other.expanded == ref.expanded and other.generated == ref.generated
        assert [other.plan(i) == ref.plan(i) for i in range(len(ref))] == [True] * len(ref)
    for budget in ({"max_expanded": 3}, {"max_states": 5}, {"max_depth": 1}, {"max_seconds": 0}):
        batch = mc.multi_iw(task, starts, ctx=ctx, **budget)
        for i, s in enumerate(starts):
            assert_equal(batch, i, mymyr.search.iw(task, max_arity=1, start=s, **budget))
    assert set(mc.multi_iw(task, starts, ctx=ctx, max_seconds=0).status) == {mymyr.search.Status.OUT_OF_TIME}


def test_batched_iw1_from_torch_rows(ctx):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    task = text_task("blocks__probBLOCKS-8-0")
    starts = walk(task, steps=20, seed=7, walks=3)
    goals = walk_goals(starts, 1)
    W = max(task.words, max(s.num_words for s in starts)) + 1  # padded rows
    rows = np.zeros((len(starts), W), dtype=np.uint64)
    for i, s in enumerate(starts):
        rows[i, : s.num_words] = s.words
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dev = torch.from_numpy(rows.view(np.int64)).cuda()
        batch = mc.batched_iw1(task, dev[:, : W - 1], ctx=ctx, stream=stream, goals=goals)
    host = mc.batched_iw1(task, starts, ctx=ctx, goals=goals)
    assert batch.status == host.status and batch.plan_length == host.plan_length
    assert [batch.goal_state(i) == host.goal_state(i) for i in range(len(host))] == [True] * len(host)
    assert sum(batch.solved) > 0
    for i, s in enumerate(starts):
        goal = [{"positive": list(goals[i][0]), "negative": []}]
        assert_equal(batch, i, mymyr.search.iw(task, max_arity=1, start=s, goal=goal))


def test_rollouts_are_deterministic(ctx):
    task = text_task("gripper__prob05")
    seeds = list(range(1, 200))
    ref = mc.rollouts(task, seeds, ctx=ctx, max_next_layer_states=4)
    again = mc.rollouts(task, seeds, ctx=ctx, max_next_layer_states=4, max_searches=13, chunk_states=7)
    assert again.status == ref.status and again.plan_length == ref.plan_length
    init = set(task.initial_state.atom_slots())
    for i in range(len(ref)):
        assert again.reached_atoms(i) == ref.reached_atoms(i)
        assert init <= set(ref.reached_slots(i))
        assert ref.reached_atoms(i) == sorted(ref.reached_atoms(i))
    # truncated layers differ between seeds
    assert len({tuple(ref.reached_slots(i)) for i in range(len(ref))}) > 1 or len({ref.expanded[i] for i in range(len(ref))}) > 1
    one = mc.rollouts(task, [seeds[17]], ctx=ctx, max_next_layer_states=4)
    assert one.reached_atoms(0) == ref.reached_atoms(17) and one.passes(0)[-1].expanded == ref.passes(17)[-1].expanded


def test_rollouts_with_a_goal_solve_and_replay(ctx):
    task = text_task("depot__p02")
    target = walk(task, steps=8, seed=11, walks=1)[-1]
    have = set(task.initial_state.atom_slots())
    goal = ([a for a in target.atom_slots() if a not in have][:2], [])
    batch = mc.rollouts(task, list(range(32)), ctx=ctx, goal=goal, max_next_layer_states=8)
    assert sum(batch.solved) > 0
    for i in range(len(batch)):
        if not batch.solved[i]:
            continue
        s = task.initial_state
        for a in batch.plan(i):
            s = task.apply(s, a)
        assert s == batch.goal_state(i)
        assert set(goal[0]) <= set(s.atom_slots())


def test_errors(ctx):
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    with pytest.raises(ValueError, match="numeric"):
        mc.multi_iw(numeric, [numeric.initial_state], ctx=ctx)
    task = text_task("depot__p02")
    s = task.initial_state
    with pytest.raises(ValueError, match="goals for"):
        mc.multi_iw(task, [s, s], ctx=ctx, goals=[([], [])])
    with pytest.raises(ValueError, match="width_zero"):
        mc.multi_iw(task, [s], ctx=ctx, width_zero="sometimes")
    with pytest.raises(ValueError, match="fluent slot"):
        mc.multi_iw(task, [s], ctx=ctx, goals=[([10**6], [])])
    with pytest.raises(ValueError):
        mc.multi_iw(task, [s], ctx=ctx, max_arity=3)
    with pytest.raises(ValueError, match="max_next_layer_states"):
        mc.rollouts(task, [1], ctx=ctx, max_next_layer_states=0)
    with pytest.raises(TypeError):
        mc.rollouts(task, [1], ctx=ctx, start=5)
    assert len(mc.multi_iw(task, [], ctx=ctx)) == 0
