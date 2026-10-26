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
    assert h.counters["evaluations"] >= len(states) and h.counters["operators"] > 0
    # the launch configuration changes no value
    for kw in ({"variant": "frontier"}, {"warp_groups": True}, {"block_threads": 64, "force_global": True}, {"max_blocks": 1}):
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
    assert dev.counters["steps"] > 0 and dev.counters["popped"] >= dev.stats.expanded


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
        assert dev.counters["max_batch"] <= batch
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
    for kind in ("lmcut", "lmcount"):
        with pytest.raises(ValueError, match=f"'{kind}'"):
            mc.Heuristic(task, kind, ctx=ctx)
        with pytest.raises(ValueError, match=f"'{kind}'"):
            mc.astar(task, heuristic=kind, ctx=ctx)
        with pytest.raises(ValueError, match=f"'{kind}'"):
            mc.gbfs(task, heuristic=kind, ctx=ctx)
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


NEGATIVE_COSTS = """(define (domain d) (:requirements :strips :negative-preconditions :action-costs)
 (:predicates (a) (goal)) (:functions (total-cost) (c))
 (:action on :parameters () :precondition (not (a)) :effect (and (a) (increase (total-cost) (c))))
 (:action off :parameters () :precondition (a) :effect (and (not (a)) (increase (total-cost) (c))))
 (:action finish :parameters () :precondition (a) :effect (and (goal) (increase (total-cost) 100))))"""
CONSTANT_NEGATIVE = """(define (domain d) (:requirements :strips :action-costs)
 (:predicates (goal)) (:functions (total-cost))
 (:action finish :parameters () :precondition () :effect (and (goal) (increase (total-cost) -1))))"""
FLUENT_NEGATIVE = """(define (domain d) (:requirements :numeric-fluents :action-costs)
 (:predicates (goal)) (:functions (total-cost) (x))
 (:action step :parameters () :precondition (< (x) 3) :effect (and (increase (x) 1) (increase (total-cost) (- 1 (x)))))
 (:action finish :parameters () :precondition (>= (x) 3) :effect (and (goal) (increase (total-cost) 10))))"""


def test_negative_costs_fail_as_on_the_cpu(ctx):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    failed = mymyr.search.Status.FAILED

    def task_of(domain, init):
        problem = f"(define (problem p) (:domain d) (:init {init}) (:goal (goal)) (:metric minimize (total-cost)))"
        return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))

    # a cost of -1 from a static function: known on the transition
    task = task_of(NEGATIVE_COSTS, "(= (total-cost) 0) (= (c) -1)")
    assert mymyr.search.astar(task, heuristic="blind").status == failed
    for batch in (1, 64):
        a = mc.astar(task, ctx=ctx, heuristic="blind", batch=batch)
        assert a.status == failed and a.plan == [] and a.goal_state is None
        assert a.message.startswith("mymyr astar_eager (device): A* needs non-negative action costs"), a.message
        g = mc.gbfs(task, ctx=ctx, heuristic="blind", batch=batch)
        assert g.status == failed and g.plan == [] and "the CPU GBFS accepts negative costs" in g.message, g.message
    assert mymyr.search.gbfs(task, heuristic="blind").solved
    # a constant negative cost: before the search starts, with the CPU's message
    task = task_of(CONSTANT_NEGATIVE, "(= (total-cost) 0)")
    cpu = mymyr.search.astar(task, heuristic="blind")
    assert cpu.status == failed and "negative action cost" in cpu.message
    for search in (mc.astar, mc.gbfs):
        r = search(task, ctx=ctx, heuristic="blind")
        assert (r.status, r.message) == (failed, cpu.message)
    # a cost that depends on a numeric fluent: the numeric device A* fails at the transition that lowers g, as the CPU
    task = task_of(FLUENT_NEGATIVE, "(= (total-cost) 0) (= (x) 0)")
    cpu = mymyr.search.astar(task, heuristic="blind")
    dev = mc.astar(task, ctx=ctx, heuristic="blind")
    assert cpu.status == failed and (dev.status, dev.message) == (failed, cpu.message)
    assert mc.gbfs(task, ctx=ctx, heuristic="blind").cost == mymyr.search.gbfs(task, heuristic="blind").cost


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


def test_pair_heuristic_names_and_scratch_budget(ctx):
    task = text_task("depot__p02")
    assert mc.Heuristic(task, "set_additive", ctx=ctx).kind == "set_additive"
    for kind in ("hsa", "setadd"):
        with pytest.raises(ValueError, match="unknown heuristic"):
            mc.Heuristic(task, kind, ctx=ctx)
    base = mc.Heuristic(task, "h2", ctx=ctx)
    size = base.counters["group_bytes"]
    with pytest.raises(ValueError, match=r"propositions.*limit is"):
        mc.Heuristic(task, "h2", ctx=ctx, max_scratch_bytes=size - 1)
    one = mc.Heuristic(task, "h2", ctx=ctx, max_scratch_bytes=size)
    assert one.counters["blocks"] == 1 and not one.counters["shared"]
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


@pytest.mark.parametrize("objects", [65, 80, 129, 192])
@pytest.mark.parametrize("heuristic", ["blind", "max"])
def test_wide_best_first_equals_cpu(ctx, tmp_path, objects, heuristic):
    from conftest import wide_task

    task = wide_task(tmp_path, objects)
    assert task.words > 64
    for device_search, cpu_search in ((mc.astar, mymyr.search.astar), (mc.gbfs, mymyr.search.gbfs)):
        cpu = cpu_search(task, heuristic=heuristic)
        dev = device_search(task, ctx=ctx, heuristic=heuristic, batch=1, chunk_states=1)
        assert dev.status == cpu.status and dev.plan == cpu.plan
        assert dev.goal_state == cpu.goal_state and dev.cost == cpu.cost
        assert (dev.stats.expanded, dev.stats.generated) == (cpu.stats.expanded, cpu.stats.generated)


@pytest.mark.parametrize("heuristic", ["blind", "max"])
def test_ipc_astar_equals_cpu(ctx, rovers_task, heuristic):
    task = rovers_task
    cpu = mymyr.search.astar(task, heuristic=heuristic, max_expanded=2)
    dev = mc.astar(task, ctx=ctx, heuristic=heuristic, batch=1, max_expanded=2)
    assert dev.status == cpu.status and dev.plan == cpu.plan
    assert (dev.stats.expanded, dev.stats.generated) == (cpu.stats.expanded, cpu.stats.generated)


def test_very_wide_astar_respects_byte_reservations(tmp_path):
    from conftest import wide_task

    task = wide_task(tmp_path, objects=2048)
    assert task.words == 65536
    budget = 512 << 20
    small = mc.Context(0, max_bytes=budget)
    cpu = mymyr.search.astar(task, heuristic="blind")
    dev = mc.astar(task, ctx=small, heuristic="blind")
    assert dev.status == cpu.status and dev.plan == cpu.plan
    assert dev.goal_state == cpu.goal_state and dev.cost == cpu.cost
    assert dev.counters["device_bytes"] <= budget
    small.synchronize()


def test_lazy_astar_crosses_wide_boundary(ctx, tmp_path):
    from conftest import wide_task

    task = wide_task(tmp_path, atoms="lazy", initial_atoms=4093, extra_effects=True)
    assert task.words == 64
    dev = mc.astar(task, ctx=ctx, heuristic="blind", batch=1)
    cpu = mymyr.search.astar(task, heuristic="blind")
    assert task.words > 64 and dev.counters["widenings"] > 0
    assert dev.status == cpu.status and dev.plan == cpu.plan
    assert dev.goal_state == cpu.goal_state and dev.cost == cpu.cost
    assert (dev.stats.expanded, dev.stats.generated) == (cpu.stats.expanded, cpu.stats.generated)


@pytest.mark.parametrize("batch", [1, 64])
def test_wide_ipc_best_first_equals_cpu(ctx, blocksworld_wide, batch):
    task = blocksworld_wide
    assert task.words > 64
    for device_search, cpu_search in ((mc.astar, mymyr.search.astar), (mc.gbfs, mymyr.search.gbfs)):
        cpu = cpu_search(task, heuristic="blind", max_expanded=300)
        dev = device_search(task, ctx=ctx, heuristic="blind", batch=batch, max_expanded=300)
        assert dev.status == cpu.status
        if batch == 1:
            assert dev.plan == cpu.plan and dev.goal_state == cpu.goal_state
            assert (dev.stats.expanded, dev.stats.generated) == (cpu.stats.expanded, cpu.stats.generated)
        elif device_search is mc.astar:
            assert dev.cost == cpu.cost


def test_wide_best_first_keeps_the_requested_batch(ctx, tmp_path):
    """Rows of 18,924 words: a step still pops `batch` states (the rows are gathered a chunk at a time). The initial
    state has 1100 successors; the goal follows the last of them."""
    n = 1100
    text = f"O {n}\nP 4\nF 1 picked\nF 0 free\nF 2 pad\nF 0 done\nSI 0\nFI 1\n1\nG 1\n3 1\n"
    text += "A 2\npick 1\nL 1\n1 1\nE 1\n0\nL 0\nF 3\n1 0\n0 1 0\n2 1 0 0\n"
    text += f"finish 0\nL 1\n0 1 {-n}\nE 1\n0\nL 0\nF 1\n3 1\n"
    path = tmp_path / "wide-fan.txt"
    path.write_text(text)
    task = mymyr.Task.from_text(path, atoms="dense")
    assert task.words > 16 << 10
    for device_search, cpu_search in ((mc.astar, mymyr.search.astar), (mc.gbfs, mymyr.search.gbfs)):
        cpu = cpu_search(task, heuristic="blind")
        dev = device_search(task, ctx=ctx, heuristic="blind", batch=512)
        assert dev.status == cpu.status and len(dev.plan) == len(cpu.plan) == 2
        assert dev.counters["max_batch"] == 512


def test_best_first_refuses_successors_beyond_memory(tmp_path):
    from conftest import fan_task, successor_refusal

    task = fan_task(tmp_path, 400)
    small = mc.Context(0, max_bytes=32 << 20)
    for device_search, name in ((mc.astar, r"device A\*"), (mc.gbfs, "device GBFS")):
        with pytest.raises(ValueError, match=successor_refusal(name, 160000, 32 << 20)):
            device_search(task, ctx=small, heuristic="blind", batch=1)
    small.synchronize()
