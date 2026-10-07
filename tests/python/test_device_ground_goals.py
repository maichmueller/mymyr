"""Ground conjunction goals have the same truth, plan and search counts on CPU and CUDA."""

import numpy as np
import pytest

import mymyr
from mymyr import search
from conftest import ROOT, text_task


def assert_result(batch, i, expected):
    assert batch.status[i] == expected.status
    assert [str(a) for a in batch.plan(i)] == [str(a) for a in expected.plan]
    assert batch.goal_state(i) == expected.goal_state
    assert batch.cost(i) == expected.cost
    assert [(p.arity, p.status, p.expanded, p.generated, p.generated_in_tree, p.skipped) for p in batch.passes(i)] == [
        (p.arity, p.status, p.expanded, p.generated, p.generated_in_tree, p.skipped) for p in expected.passes
    ]


@pytest.mark.parametrize("case", ["numeric", "arithmetic", "negative", "undefined", "initial", "f64", "derived", "not_derived"])
def test_ground_conjunction_searches(case):
    if case in {"derived", "not_derived"}:
        task = text_task("philosophers__p03-phil4", atoms="lazy")
        literal = task.goal_condition.literals[0]
        goal = task.ground_condition([literal if case == "derived" else task.literal(literal, positive=False)])
    else:
        name = "cs-hydropower" if case == "f64" else "cs-counters"
        task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks" / f"{name}.txt"), atoms="lazy")
        value = task.initial_state.numeric_values()[3] if case != "f64" else 0
        constraints = {
            "numeric": f"(> (value o3) {value + 1})",
            "arithmetic": f"(>= (+ (value o3) 0.125) {value + 1.125})",
            "negative": f"(<= (- (value o3)) {-value - 1})",
            "undefined": "(= (/ (value o3) 0) 0)",
            "initial": f"(= (value o3) {value})",
        }
        goal = task.goal_condition if case == "f64" else task.ground_condition(constraints=[constraints[case]])
    starts = [task.initial_state]
    labels = task.applicable_actions(starts[0])
    starts.append(task.apply(starts[0], labels[0]) if labels else starts[0])
    options = {"max_arity": 2, "max_states": 3000}
    expected = [search.iw(task, start=s, goal=goal, **options) for s in starts]
    assert expected[0].status == search.iw(task, goal=[goal], **options).status
    for result in expected:
        if result.solved:
            assert goal.holds(result.goal_state)
    try:
        from mymyr import cuda
    except ImportError:
        return
    if not cuda.available():
        return
    ctx = cuda.Context(0, max_bytes=1 << 30)
    batch = cuda.multi_iw(task, starts, goals=[goal, goal], ctx=ctx, max_searches=1, chunk_states=3, **options)
    for i, result in enumerate(expected):
        assert_result(batch, i, result)
    seeds = [7, 17]
    rollouts = cuda.rollouts(task, seeds, goal=goal, ctx=ctx, max_next_layer_states=3, **options)
    for i, seed in enumerate(seeds):
        assert_result(rollouts, i, search.iw(task, goal=goal, layer_order="randomized", seed=seed,
                                           max_next_layer_states=3, **options))
    from mymyr.rl import TaskTable

    table = TaskTable([task, task])
    ids = [1, 0]
    b = cuda.batched_iw1(table, starts, task_ids=ids, goals=[goal, goal], ctx=ctx, max_states=3000)
    references = [search.iw(task, start=s, goal=goal, max_arity=1, max_states=3000) for s in starts]
    for i, result in enumerate(references):
        assert_result(b, i, result)
    try:
        import torch
    except ImportError:
        return
    width = max(1, task.words, *(s.num_words for s in starts))
    rows = np.zeros((len(starts), width + task.numeric_words), np.uint64)
    for i, state in enumerate(starts):
        rows[i, :state.num_words] = state.words
        rows[i, width:] = state.numeric_words
    d = cuda.batched_iw1(table, torch.from_numpy(rows.view(np.int64)).cuda(), task_ids=ids,
                        goals=[goal, goal], ctx=ctx, max_states=3000)
    for i, result in enumerate(references):
        assert_result(d, i, result)
