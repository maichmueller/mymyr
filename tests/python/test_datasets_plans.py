"""mymyr.datasets.plan_rows and StateRows.from_plans: plans replayed into state rows and action labels."""

import importlib.util
import random

import numpy as np
import pytest

import mymyr
from mymyr import datasets, encoding, rl, search

from conftest import ROOT, text_task

IPC = ROOT / "tests/data/ipc"
PLANS = sorted(IPC.glob("*/train/*.plan"))
DOMAINS = sorted({p.parent.parent.name for p in PLANS})
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def ipc_task(plan, **options):
    options.setdefault("atoms", "dense")
    return mymyr.Task.from_pddl(plan.parent / "domain.pddl", plan.with_suffix(".pddl"), **options)


def domain_plans(domain):
    return [p for p in PLANS if p.parent.parent.name == domain]


def ipc_table(domain):
    plans = domain_plans(domain)
    table = rl.TaskTable.from_pddl(plans[0].parent / "domain.pddl", [p.with_suffix(".pddl") for p in plans],
                                   atoms="dense")
    return table, plans


def random_walk(task, steps, seed=0):
    rng = random.Random(seed)
    state, plan = task.initial_state, []
    for _ in range(steps):
        actions = task.applicable_actions(state)
        if not actions:
            break
        plan.append(actions[rng.randrange(len(actions))])
        state = plan[-1].apply(state)
    return plan


def check_against_expansions(source, pr, task_ids=None):
    """Every step's index is its label's row in rl.expand, whose successor is the next row; under drop_noops the
    step's row is kept unless it is a no-op, and its kept index is the step's index."""
    for i in range(pr.num_steps):
        row = pr.rows[i : i + 1]
        full = rl.expand(source, row, task_ids, noops=True)
        k = int(pr.index[i])
        assert full.total == pr.num_actions[i]
        assert int(full.schema[k]) == pr.schema[i]
        assert np.array_equal(full.binding[k], pr.binding[i])
        if i + 1 < len(pr.rows):
            assert np.array_equal(full.succ[k], pr.rows[i + 1])
        assert bool(full.noop[k]) == bool(pr.noop[i])
        kept = rl.expand(source, row, task_ids, drop_noops=True)
        hits = np.flatnonzero(np.asarray(kept.index) == k)
        dropped = bool(pr.noop[i]) and not bool(np.all(full.noop))
        assert len(hits) == (0 if dropped else 1)


@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
def test_ipc_plans_replay_to_the_goal(domain):
    for plan in domain_plans(domain):
        task = ipc_task(plan)
        text = plan.read_text()
        pr = datasets.plan_rows(task, text, goal_state=True)
        steps = search.parse_plan(task, text)
        assert pr.num_steps == len(steps) and pr.goal_reached and pr.goal_state
        assert pr.rows.dtype == np.uint64 and len(pr.rows) == len(steps) + 1
        assert np.array_equal(pr.rows[0], np.asarray(task.encode([task.initial_state]))[0])
        assert bool(rl.is_goal(task, pr.rows[-1:])[0])
        assert pr.schema.dtype == np.int32 and pr.binding.shape == (len(steps), task.label_width)
        assert [(int(s), tuple(int(o) for o in b if o >= 0)) for s, b in zip(pr.schema, pr.binding)] == [
            (a.schema, tuple(a.binding)) for a in steps]
        check_against_expansions(task, pr)
        # the plan's states, one by one
        state = task.initial_state
        for i, action in enumerate(steps):
            state = action.apply(state)
            assert np.array_equal(pr.rows[i + 1], np.asarray(task.encode([state]))[0])
        # text and Actions agree; without goal_state the reached state is left out
        same = datasets.plan_rows(task, steps)
        assert np.array_equal(same.rows, pr.rows[:-1]) and not same.goal_state and same.goal_reached
        for name in ("schema", "binding", "index", "noop"):
            assert np.array_equal(getattr(same, name), getattr(pr, name))
        assert np.array_equal(same.num_actions, pr.num_actions[:-1])


@needs_frontend
def test_pool_steps_follow_the_plans_of_a_table():
    table, plans = ipc_table("blocksworld-ipc")
    replays = [datasets.plan_rows(table, p.read_text(), task_id=i) for i, p in enumerate(plans)]
    for i, pr in enumerate(replays):
        assert pr.task_id == i and pr.goal_reached
        check_against_expansions(table, pr, [i])
        single = datasets.plan_rows(table.tasks[i], plans[i].read_text())
        assert np.array_equal(single.rows, pr.rows) and np.array_equal(single.index, pr.index)
    pool = rl.CpuEnvPool(table, num_envs=len(plans), sampler=None, autoreset=False)
    batch = pool.reset(task_ids=list(range(len(plans))))
    for step in range(max(pr.num_steps for pr in replays)):
        envs = [i for i, pr in enumerate(replays) if step < pr.num_steps]
        for e, env in enumerate(np.asarray(batch.env_ids).tolist()):
            if env in envs:
                pr = replays[env]
                assert np.array_equal(np.asarray(batch.states)[e, : pr.rows.shape[1]], pr.rows[step])
        batch = pool.step(envs, [int(replays[i].index[step]) for i in envs])
        for e, env in enumerate(envs):
            assert bool(batch.terminated[e]) == (step + 1 == replays[env].num_steps)


@needs_frontend
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_batched_env_steps_follow_a_plan():
    import torch

    from mymyr.rl.torch import BatchedEnv

    table, plans = ipc_table("rovers-ipc")
    replays = [datasets.plan_rows(table, p.read_text(), task_id=i) for i, p in enumerate(plans)]
    env = BatchedEnv(table, len(plans), task_ids=torch.arange(len(plans), dtype=torch.int32), sampler=None,
                     autoreset=False)
    env.reset()
    longest = max(pr.num_steps for pr in replays)
    for step in range(longest):
        for i, pr in enumerate(replays):
            if step < pr.num_steps:
                width = pr.rows.shape[1]
                assert np.array_equal(env.states[i, :width].numpy().view(np.uint64), pr.rows[step])
        # an environment past its plan's end sits at its goal, which absorbs any action
        action = torch.tensor([int(pr.index[min(step, pr.num_steps - 1)]) for pr in replays], dtype=torch.int64)
        env.step(action)
    assert env.at_goal.all()


def test_witness_pruning_locates_pruned_actions_at_their_witness():
    task = text_task("organic-synthesis-opt18-strips__p20")
    plan = random_walk(task, 2)
    full = datasets.plan_rows(task, plan, goal_state=True)
    pruned = datasets.plan_rows(task, plan, goal_state=True, witness_pruning=True)
    assert np.array_equal(full.rows, pruned.rows)
    assert np.array_equal(full.schema, pruned.schema) and np.array_equal(full.binding, pruned.binding)
    assert np.any(pruned.num_actions < full.num_actions) and np.any(pruned.index != full.index)
    for i in range(pruned.num_steps):
        exp = rl.expand(task, pruned.rows[i : i + 1], witness_pruning=True)
        k = int(pruned.index[i])
        assert exp.total == pruned.num_actions[i]
        assert int(exp.schema[k]) == pruned.schema[i] and np.array_equal(exp.succ[k], pruned.rows[i + 1])
    pool = rl.CpuEnvPool(task, num_envs=1, witness_pruning=True, autoreset=False)
    batch = pool.reset()
    for i in range(pruned.num_steps):
        assert np.array_equal(np.asarray(batch.states)[0], pruned.rows[i])
        batch = pool.step([0], [int(pruned.index[i])])
    assert np.array_equal(np.asarray(batch.states)[0], pruned.rows[-1])


def test_steps_that_do_not_apply_are_refused():
    task = text_task("gripper__prob05")
    plan = random_walk(task, 4)
    after = plan[0].apply(task.initial_state)
    gone = next(a for a in task.applicable_actions(task.initial_state) if not a.is_applicable(after))
    with pytest.raises(ValueError, match="plan step 2 .* is not applicable in its state"):
        datasets.plan_rows(task, [plan[0], gone])
    lines = [search.format_plan(task, [a]).splitlines()[0] for a in (plan[0], gone)]
    with pytest.raises(ValueError, match="line 2"):
        datasets.plan_rows(task, "\n".join(lines))
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="plan step 1 .* action of another task"):
        datasets.plan_rows(task, random_walk(other, 1))
    with pytest.raises(TypeError, match="plan step 1 is a tuple"):
        datasets.plan_rows(task, [(plan[0].schema, plan[0].binding)])


def test_sources_and_task_ids_are_checked():
    task = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="a Task source is instance 0"):
        datasets.plan_rows(task, [], task_id=1)
    with pytest.raises(IndexError, match="task_id 1"):
        datasets.plan_rows(rl.TaskSuite.group([task]), [], task_id=1)
    with pytest.raises(TypeError, match="plan_rows takes"):
        datasets.plan_rows("gripper", [])
    lazy = text_task("gripper__prob05", atoms="lazy")
    with pytest.raises(ValueError, match="lazy atom slots"):
        datasets.plan_rows(lazy, [])


def test_an_empty_plan_is_the_initial_state():
    task = text_task("gripper__prob05")
    initial = np.asarray(task.encode([task.initial_state]))
    pr = datasets.plan_rows(task, "; cost = 0 (unit cost)\n", goal_state=True)
    assert pr.num_steps == 0 and np.array_equal(pr.rows, initial) and not pr.goal_reached
    assert pr.num_actions.tolist() == [rl.expand(task, initial).total]
    assert datasets.plan_rows(task, []).rows.shape == (0, initial.shape[1])


def test_numeric_plans_keep_their_numeric_words():
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    result = search.astar(task)
    pr = datasets.plan_rows(task, result.plan, goal_state=True)
    assert pr.goal_reached and task.numeric_words > 0
    assert pr.rows.shape[1] == np.asarray(task.encode([task.initial_state])).shape[1]
    state = task.initial_state
    for i, action in enumerate(result.plan):
        state = action.apply(state)
        assert np.array_equal(pr.rows[i + 1], np.asarray(task.encode([state]))[0])
    check_against_expansions(task, pr)


def test_equivalent_actions_map_to_canonical_indices():
    task = text_task("gripper__prob05")
    plan = random_walk(task, 5, seed=3)
    pr = datasets.plan_rows(task, plan)
    # per step every applicable action, as Actions, labels and plan lines, with the plan's own action repeated
    equivalent, states = [], [task.initial_state]
    for action in plan:
        states.append(action.apply(states[-1]))
    for i, action in enumerate(plan):
        actions = task.applicable_actions(states[i])
        forms = [a if j % 3 == 0 else (a.schema, a.binding) if j % 3 == 1 else
                 search.format_plan(task, [a], start=states[i]).splitlines()[0] for j, a in enumerate(actions)]
        equivalent.append([action, *reversed(forms)])
    eq = datasets.plan_rows(task, plan, equivalent=equivalent, goal_state=True)
    assert eq.equivalent.dtype == np.int32 and eq.equivalent.shape == (len(plan), int(pr.num_actions.max()))
    for i in range(len(plan)):
        own = eq.equivalent[i][eq.equivalent[i] >= 0]
        assert own[0] == pr.index[i] and sorted(own.tolist()) == list(range(pr.num_actions[i]))
    hot = eq.equivalent_actions()
    assert hot.shape == (len(plan) + 1, int(eq.num_actions.max())) and not hot[-1].any()
    assert hot[:-1].sum(axis=1).tolist() == pr.num_actions.tolist()
    # the plan's own action alone
    alone = datasets.plan_rows(task, plan, equivalent=[[] for _ in plan])
    assert alone.equivalent.tolist() == [[int(k)] for k in pr.index]
    with pytest.raises(ValueError, match="no equivalent actions"):
        pr.equivalent_actions()
    with pytest.raises(ValueError, match="lists of equivalent actions for 5 steps"):
        datasets.plan_rows(task, plan, equivalent=[[]])
    elsewhere = [a for a in task.applicable_actions(states[1]) if not a.is_applicable(states[0])][0]
    with pytest.raises(ValueError, match="of plan step 1 is not applicable"):
        datasets.plan_rows(task, plan, equivalent=[[elsewhere], *[[] for _ in plan[1:]]])
    with pytest.raises(ValueError, match="of plan step 1"):
        datasets.plan_rows(task, plan, equivalent=[["(no-such-action)"], *[[] for _ in plan[1:]]])
    with pytest.raises(TypeError, match="of plan step 1"):
        datasets.plan_rows(task, plan, equivalent=[[3.5], *[[] for _ in plan[1:]]])


@needs_frontend
def test_plans_of_a_suite():
    plans = [domain_plans("ferry-ipc")[0], domain_plans("satellite-ipc")[1], domain_plans("ferry-ipc")[1]]
    suite = rl.TaskSuite.group([ipc_task(p) for p in plans])
    for i, plan in enumerate(plans):
        pr = datasets.plan_rows(suite, plan.read_text(), task_id=i, goal_state=True)
        assert pr.goal_reached
        check_against_expansions(suite, pr, [i])
        alone = datasets.plan_rows(ipc_task(plan), plan.read_text(), goal_state=True)
        assert np.array_equal(alone.index, pr.index)


@needs_frontend
def test_state_rows_from_plans_encode_with_their_labels():
    table, plans = ipc_table("blocksworld-ipc")
    replays = [datasets.plan_rows(table, p.read_text(), task_id=i, goal_state=i % 2 == 0) for i, p in enumerate(plans)]
    ds = datasets.StateRows.from_plans(replays)
    n = sum(len(pr.rows) for pr in replays)
    assert len(ds) == n and ds.rows.shape[1] == max(pr.rows.shape[1] for pr in replays)
    assert sorted(ds.labels) == ["action", "binding", "noop", "schema"]
    assert ds.task_ids.tolist() == [pr.task_id for pr in replays for _ in pr.rows]
    goal_rows = np.flatnonzero(ds.labels["action"] < 0)
    assert len(goal_rows) == sum(pr.goal_state for pr in replays)
    assert np.all(ds.labels["schema"][goal_rows] == -1) and np.all(ds.labels["binding"][goal_rows] == -1)
    assert ds.labels["action"].tolist() == [v for pr in replays for v in [*pr.index.tolist(), *[-1] * pr.goal_state]]

    encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
    collate = datasets.EncodingCollate(encoder, actions=True)
    picked = np.flatnonzero(ds.labels["action"] >= 0)
    # the row after a step is its successor, except after the last step of a plan without its goal state
    starts = np.cumsum([0, *[len(pr.rows) for pr in replays]])
    last = {int(starts[i + 1]) - 1 for i, pr in enumerate(replays) if not pr.goal_state}
    followed = np.array([r for r in picked if r not in last])
    loaded = collate(ds.take(picked))
    direct = encoder.encode(ds.rows[picked], ds.task_ids[picked], actions=loaded.actions)
    assert all(np.array_equal(loaded.graphs.as_dict()[k], direct.as_dict()[k]) for k in direct.as_dict())
    action = loaded.labels["action"]
    # each label is an action entity of its graph, and its expansion row leads to the plan's next state
    entities = encoding.action_entities(loaded.graphs)
    owner = np.searchsorted(np.asarray(loaded.graphs.entity_offsets), entities, side="right") - 1
    positions = encoding.action_indices(loaded.graphs, entities)
    for g, r in enumerate(picked):
        assert np.count_nonzero((owner == g) & (positions == action[g])) == 1
        if r in followed:
            succ = np.asarray(loaded.actions.succ)[int(loaded.actions.offsets[g]) + int(action[g])]
            width = ds.rows.shape[1]
            assert np.array_equal(np.pad(succ, (0, max(0, width - len(succ))))[:width], ds.rows[r + 1])


def test_state_rows_from_plans_with_equivalent_actions():
    task = text_task("gripper__prob05")
    plans = [random_walk(task, 4, seed=s) for s in range(3)]
    replays = [datasets.plan_rows(task, p, equivalent=[[] for _ in p], goal_state=s == 1) for s, p in enumerate(plans)]
    ds = datasets.StateRows.from_plans(replays)
    hot = ds.labels["equivalent_actions"]
    most = max(int(pr.num_actions.max()) for pr in replays)
    assert hot.dtype == bool and hot.shape == (len(ds), most)
    rows = np.flatnonzero(ds.labels["action"] >= 0)
    assert np.all(hot[rows, ds.labels["action"][rows]]) and hot.sum() == len(rows)
    with pytest.raises(ValueError, match="some plans have equivalent actions"):
        datasets.StateRows.from_plans([replays[0], datasets.plan_rows(task, plans[1])])
    with pytest.raises(ValueError, match="at least one plan"):
        datasets.StateRows.from_plans([])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_state_rows_from_plans_on_a_torch_device():
    import torch

    task = text_task("gripper__prob05")
    replays = [datasets.plan_rows(task, random_walk(task, 6, seed=s), goal_state=True) for s in range(2)]
    host = datasets.StateRows.from_plans(replays)
    ds = datasets.StateRows.from_plans(replays, device="cpu")
    assert isinstance(ds.rows, np.ndarray)  # CPU tensors are host arrays of the dataset
    assert np.array_equal(ds.rows, host.rows)
    assert all(np.array_equal(np.asarray(ds.labels[k]), host.labels[k]) for k in host.labels)
    if torch.cuda.is_available():
        on_device = datasets.StateRows.from_plans(replays, device="cuda:0")
        assert on_device.rows.is_cuda and on_device.labels["action"].is_cuda
        assert np.array_equal(on_device.rows.cpu().numpy().view(np.uint64), host.rows)
