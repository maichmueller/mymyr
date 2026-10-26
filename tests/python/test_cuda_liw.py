"""Device landmark novelty equals CPU LIW, including launch grouping, budgets and seeded layer orders."""

import numpy as np
import pytest

import mymyr
import mymyr.search as search
from conftest import SMALL_TASKS, text_task, walk, wide_task
from test_cuda_iw import assert_equal, ctx, walk_goals  # noqa: F401

mc = pytest.importorskip("mymyr.cuda")
pytestmark = pytest.mark.skipif(not mc.available(), reason="no CUDA device")


def graphs_of(task):
    atoms = list(task.initial_state.atoms())[:4]
    return [
        search.FactLandmarkGraph(task, []),
        search.FactLandmarkGraph(task, atoms[:1], [atoms[:2], atoms[1:]]),
        search.approximate_fact_landmarks(task),
    ]


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("grouping", [{}, {"disjunctive": True}, {"disjunctive": True, "all_private": True}])
def test_liw_shared_and_per_search_graphs(ctx, name, grouping):
    task = text_task(name)
    starts = walk(task, steps=4, seed=5, walks=1)[:4]
    graphs = graphs_of(task)
    graphs = [graphs[i % len(graphs)] for i in range(len(starts))]
    goals = walk_goals(starts, 2)
    conditions = [task.ground_condition([task.atom(k) for k in positive]) for positive, _ in goals]
    for landmarks in (graphs, graphs[-1]):
        cpu = [search.liw(task, start=s, goal=conditions[i], landmarks=landmarks[i] if isinstance(landmarks, list) else landmarks,
                          max_width=1, max_states=80, **grouping) for i, s in enumerate(starts)]
        for options in ({}, {"max_searches": 1, "chunk_states": 1}, {"max_searches": 2, "chunk_states": 3}):
            batch = mc.liw(task, starts, ctx=ctx, landmarks=landmarks, goals=conditions, max_states=80, **grouping, **options)
            for i, expected in enumerate(cpu):
                assert_equal(batch, i, expected)


@pytest.mark.parametrize("budget", [{}, {"max_states": 1}, {"max_states": 8}, {"max_expanded": 2},
                                   {"max_depth": 1}, {"max_seconds": 0}])
@pytest.mark.parametrize("width_zero", ["expand_depth_one", "root_only"])
def test_liw_budgets_and_width_zero(ctx, budget, width_zero):
    task = text_task("blocks__probBLOCKS-8-0")
    graph = search.approximate_fact_landmarks(task)
    starts = [task.initial_state] * 3
    batch = mc.liw(task, starts, ctx=ctx, landmarks=graph, width_zero=width_zero, chunk_states=1, **budget)
    expected = search.liw(task, landmarks=graph, max_width=1, width_zero=width_zero, **budget)
    for i in range(len(starts)):
        assert_equal(batch, i, expected)


@pytest.mark.parametrize("private", [False, True])
def test_liw_unshared_overlapping_ranks_and_bot(ctx, tmp_path, private):
    task = wide_task(tmp_path)
    goals = list(task.successor_states(task.initial_state))
    atoms = [task.atom(slot) for state in goals for slot in state.atom_slots() if slot not in task.initial_state.atom_slots()]
    graph = search.FactLandmarkGraph(task, atoms[:1], [atoms, atoms[-1:]])
    grouping = {"disjunctive": True, "all_private": private}
    if not private:
        grouping["unshared_atoms"] = atoms[-1:]
    starts = [task.initial_state, *goals]
    batch = mc.liw(task, starts, ctx=ctx, landmarks=graph, chunk_states=1, max_searches=2, **grouping)
    for i, start in enumerate(starts):
        assert_equal(batch, i, search.liw(task, start=start, landmarks=graph, max_width=1, **grouping))


@pytest.mark.parametrize("lazy", [False, True])
def test_liw_wide_rows_and_many_ranks(ctx, tmp_path, lazy):
    task = wide_task(tmp_path, atoms="lazy" if lazy else "dense", initial_atoms=4093 if lazy else 80,
                     extra_effects=lazy)
    graph = search.FactLandmarkGraph(task, list(task.initial_state.atoms())[:75])
    starts = [task.initial_state] * 3
    batch = mc.liw(task, starts, ctx=ctx, landmarks=graph, chunk_states=1, table_bytes=1 << 20)
    expected = search.liw(task, landmarks=graph, max_width=1)
    for i in range(len(starts)):
        assert_equal(batch, i, expected)
    if not lazy:
        assert task.words > 64


@pytest.mark.parametrize("truncation", [None, 3])
def test_liw_rollout_seeds(ctx, truncation):
    task = text_task("blocks__probBLOCKS-8-0")
    graphs = graphs_of(task)
    seeds = [0, 1, 12345]
    for grouping in ({}, {"disjunctive": True}, {"disjunctive": True, "all_private": True}):
        cpu = [search.liw(task, landmarks=graphs[i], max_width=1, layer_order="randomized", seed=seed,
                          max_states=60, max_next_layer_states=truncation, **grouping) for i, seed in enumerate(seeds)]
        for options in ({}, {"max_searches": 1, "chunk_states": 1}, {"max_searches": 2, "chunk_states": 3}):
            batch = mc.rollouts(task, seeds, ctx=ctx, landmarks=graphs, max_states=60,
                                max_next_layer_states=truncation, **grouping, **options)
            for i, expected in enumerate(cpu):
                assert_equal(batch, i, expected)


def test_liw_task_table(ctx):
    import mymyr.rl as rl

    tasks = [text_task("blocks__probBLOCKS-8-0"), text_task("blocks__probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    ids = [1, 0, 1, 0]
    starts = [tasks[i].initial_state for i in ids]
    graphs = [search.approximate_fact_landmarks(tasks[i]) for i in ids]
    rows = np.zeros((len(ids), table.words), dtype=np.uint64)
    for i, task_id in enumerate(ids):
        encoded = np.asarray(tasks[task_id].encode([starts[i]]))
        rows[i, :encoded.shape[1]] = encoded[0]
    for rows in (starts, rows):
        batch = mc.liw(table, rows, ctx=ctx, task_ids=ids, landmarks=graphs, max_states=40, max_searches=1, chunk_states=1)
        assert batch.task_ids == ids
        for i, task_id in enumerate(ids):
            assert_equal(batch, i, search.liw(tasks[task_id], landmarks=graphs[i], max_width=1, max_states=40))


def test_liw_ipc_samples(ctx, rovers_task):
    task = rovers_task
    graph = search.approximate_fact_landmarks(task)
    expected = search.liw(task, landmarks=graph, max_width=1, max_states=40)
    batch = mc.liw(task, [task.initial_state] * 2, ctx=ctx, landmarks=graph, max_states=40, chunk_states=1)
    for i in range(2):
        assert_equal(batch, i, expected)


def test_liw_wide_ipc(ctx, blocksworld_wide):
    task = blocksworld_wide
    graph = search.FactLandmarkGraph(task, list(task.initial_state.atoms())[:3])
    batch = mc.liw(task, [task.initial_state] * 2, ctx=ctx, landmarks=graph, max_states=40, chunk_states=1)
    expected = search.liw(task, landmarks=graph, max_width=1, max_states=40)
    for i in range(2):
        assert_equal(batch, i, expected)


def test_liw_refusals(ctx, tmp_path):
    task = wide_task(tmp_path)
    graph = search.FactLandmarkGraph(task, [])
    with pytest.raises(ValueError, match="LIW arity 2.*CPU"):
        mc.liw(task, [task.initial_state], ctx=ctx, landmarks=graph, max_width=2)
    with pytest.raises(ValueError, match=r"LIW\(1\).*table_bytes.*raise table_bytes"):
        mc.liw(task, [task.initial_state], ctx=ctx, landmarks=graph, table_bytes=8)
    with pytest.raises(ValueError, match="one graph or one per search"):
        mc.liw(task, [task.initial_state] * 2, ctx=ctx, landmarks=[graph])
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="another task"):
        mc.liw(other, [other.initial_state], ctx=ctx, landmarks=graph)
    atoms = list(task.initial_state.atoms())
    graph = search.FactLandmarkGraph(task, [], [atoms])
    with pytest.raises(ValueError, match="all_private"):
        mc.liw(task, [task.initial_state], ctx=ctx, landmarks=graph, disjunctive=True,
               all_private=True, unshared_atoms=atoms)
    assert len(mc.liw(task, [], ctx=ctx, landmarks=graph)) == 0
    assert len(mc.rollouts(task, [], ctx=ctx, landmarks=graph)) == 0
