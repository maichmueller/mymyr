"""Explicit labels follow indexed transitions without enumerating an action space."""

import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, SMALL_TASKS, text_task

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")


def compare_steps(index, label, action, **kwargs):
    exp = rl.expand(index.table, index.states, index._ids(), witness_pruning=False)
    schema = torch.full((index.num_envs,), -1, dtype=torch.int32)
    binding = torch.full((index.num_envs, label.label_width), -1, dtype=torch.int32)
    for i, a in enumerate(action.tolist()):
        if 0 <= a < int(exp.offsets[i + 1] - exp.offsets[i]):
            j = int(exp.offsets[i]) + a
            schema[i] = exp.schema[j]
            binding[i, :exp.label_width] = exp.binding[j]
    a = index.step(action, **kwargs)
    b = label.step((schema, binding), **kwargs)
    for name in a._fields:
        x, y = getattr(a, name), getattr(b, name)
        if name == "count":
            assert torch.equal((x > 0).to(torch.int32), y)
        elif isinstance(x, torch.Tensor):
            assert torch.equal(x, y), name
    for name in ("states", "steps", "draws", "task_ids", "at_goal", "_reset_counts"):
        assert torch.equal(getattr(index, name), getattr(label, name)), name
    return b


@pytest.mark.parametrize("name", SMALL_TASKS + ["numeric", "pegsol-08-strips__p22"])
@pytest.mark.parametrize("autoreset", [False, True])
def test_label_random_walk_equals_index(name, autoreset):
    task = (mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
            if name == "numeric" else text_task(name))
    kwargs = dict(autoreset=autoreset, max_steps=None if name == "pegsol-08-strips__p22" else 7,
                  seed=23, step_reward=-1.0,
                  goal_reward=8.0, dead_end_reward=-4.0, sampler=None)
    index = rt.BatchedEnv(task, 12, **kwargs)
    label = rt.BatchedEnv(task, 12, actions="label", threads=3, **kwargs)
    assert torch.equal(label.count, (index.count > 0).to(torch.int32))
    seen_dead = 0
    for t in range(40):
        action = index.random_actions()
        if t % 5 == 0:
            action[::4] = -1
        result = compare_steps(index, label, action, final=t % 2 == 0)
        seen_dead += int((result.terminated & ~result.goal).sum())
    if name == "pegsol-08-strips__p22":
        assert seen_dead > 0
    label.refresh()
    assert torch.equal(label.count, (index.count > 0).to(torch.int32))


@pytest.mark.parametrize("goals", [False, True])
@pytest.mark.parametrize("sampling", [False, True])
def test_label_suite_autoreset_and_goal_masks(goals, sampling):
    table = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")])
    kwargs = dict(goals=goals, max_steps=3, seed=5, sampler="uniform" if sampling else None)
    index = rt.BatchedEnv(table, 8, task_ids=torch.arange(8) % 2, **kwargs)
    label = rt.BatchedEnv(table, 8, actions="label", task_ids=torch.arange(8) % 2, **kwargs)
    if goals:
        for env in (index, label):
            env.goal_pos[::3].zero_()
            env.goal_neg[::3].zero_()
            env.refresh()
    for t in range(12):
        next_ids = None if sampling else (torch.arange(8, dtype=torch.int32) + t) % 2
        compare_steps(index, label, index.random_actions(), next_task_ids=next_ids)
    mask = torch.arange(8) % 2 == 0
    index.reset(mask)
    label.reset(mask)
    assert torch.equal(index.states, label.states)
    assert torch.equal(label.count, (index.count > 0).to(torch.int32))


def test_label_invalid_and_random_policy_errors():
    task = text_task("blocks__probBLOCKS-8-0")
    env = rt.BatchedEnv(task, 4, actions="label", autoreset=False, step_reward=-3.0)
    exp = rl.expand(task, env.states)
    schema = exp.schema[exp.offsets[:-1].to(torch.int64)].clone()
    binding = exp.binding[exp.offsets[:-1].to(torch.int64)].clone()
    schema[0], schema[1] = -1, task.num_schemas
    binding[2, 0], binding[3, 0] = -1, task.num_objects
    before = env.states.clone()
    result = env.step((schema, binding))
    assert result.invalid.all() and not result.terminated.any()
    assert torch.equal(env.states, before)
    assert (result.reward == -3.0).all() and (result.schema == -1).all()
    assert (result.binding == -1).all() and (env.count == 1).all()
    draws = env.draws.clone()
    with pytest.raises(ValueError, match="label"):
        env.step(None)
    with pytest.raises(ValueError, match="label"):
        env.random_actions()
    assert torch.equal(env.draws, draws)
    with pytest.raises(ValueError, match="label"):
        env.step(torch.zeros(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="actions"):
        rt.BatchedEnv(task, 1, actions="unknown")
    with pytest.raises(ValueError, match="label.*CPU"):
        rt.BatchedEnv(task, 1, actions="label", device="cuda")
    with pytest.raises(ValueError, match="label.*CPU"):
        rt.Env(task, actions="label", device=0)
    with pytest.raises(ValueError, match="label"):
        rt._ops.random_actions_impl(env.handle, env.draws, env.count, 0, None)


def test_label_array_validation_and_native_step():
    task = text_task("blocks__probBLOCKS-8-0")
    env = rt.BatchedEnv(task, 3, actions="label", autoreset=False)
    expected = rt.BatchedEnv(task, 3, actions="label", autoreset=False)
    exp = rl.expand(task, env.states)
    js = exp.offsets[:-1].to(torch.int64)
    schema, binding = exp.schema[js], exp.binding[js]
    for action in [(schema.to(torch.int64), binding), (schema, binding.to(torch.int64)),
                   (schema[:1], binding), (schema, binding[:1]), (schema, binding.flatten()),
                   (schema, binding[:, :0])]:
        with pytest.raises(ValueError, match="int32|shape|stride"):
            env.step(action)
    result = expected.step((schema, binding))
    out_schema = torch.empty_like(schema)
    out_binding = torch.empty_like(binding)
    env.core.step(env.states, steps=env.steps, action=(schema, binding), count=env.count,
                  schema=out_schema, binding=out_binding, at_goal=env.at_goal)
    assert torch.equal(env.states, expected.states)
    assert torch.equal(out_schema, result.schema) and torch.equal(out_binding, result.binding)
    with pytest.raises(ValueError, match="label"):
        env.core.step(env.states)
    with pytest.raises(ValueError, match="label"):
        env.core.step(env.states, action=(None, binding))


def test_inapplicable_label_with_valid_objects_does_not_move():
    task = text_task("blocks__probBLOCKS-8-0")
    env = rt.BatchedEnv(task, 2, actions="label", autoreset=False, step_reward=-2.0)
    exp = rl.expand(task, env.states)
    js = exp.offsets[:-1].to(torch.int64)
    schema, binding = exp.schema[js], exp.binding[js]
    env.step((schema, binding))
    assert not rl.is_applicable(task, env.states, schema, binding).any()
    before = env.states.clone()
    result = env.step((schema, binding))
    assert result.invalid.all() and (result.reward == -2.0).all()
    assert torch.equal(env.states, before)


@pytest.mark.needs_gil
def test_label_step_compiles_on_cpu():
    task = text_task("blocks__probBLOCKS-8-0")
    eager = rt.BatchedEnv(task, 4, actions="label", max_steps=3)
    compiled = rt.BatchedEnv(task, 4, actions="label", max_steps=3)

    @torch.compile(backend="aot_eager", fullgraph=True)
    def step(schema, binding):
        return compiled.step((schema, binding))

    for _ in range(6):
        exp = rl.expand(task, eager.states)
        js = exp.offsets[:-1].to(torch.int64)
        schema, binding = exp.schema[js], exp.binding[js]
        a, b = eager.step((schema, binding)), step(schema, binding)
        for name in a._fields:
            x, y = getattr(a, name), getattr(b, name)
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), name


@pytest.mark.parametrize("autoreset", [False, True])
@pytest.mark.parametrize("dead_end,terminal", [("no_successors", True), ("no_successors", False), ("none", True)])
def test_label_zero_arity_goal_and_dead_end(autoreset, dead_end, terminal):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = mymyr.Domain.from_string("""(define (domain finish) (:requirements :strips)
      (:predicates (ready) (done))
      (:action finish :parameters () :precondition (ready)
       :effect (and (not (ready)) (done))))""")
    tasks = [mymyr.Task(domain.instantiate_string(f"""(define (problem finish) (:domain finish)
      (:init (ready)) (:goal ({goal})))""")) for goal in ["done", "ready"]]
    table = rl.TaskTable(tasks)
    kwargs = dict(task_ids=torch.tensor([0, 1]), autoreset=autoreset, sampler=None,
                  dead_end=dead_end, dead_end_terminal=terminal, step_reward=-1.0)
    index = rt.BatchedEnv(table, 2, **kwargs)
    label = rt.BatchedEnv(table, 2, actions="label", **kwargs)
    for _ in range(3):
        result = compare_steps(index, label, torch.zeros(2, dtype=torch.int64))
        assert result.terminated.all() and result.goal.all()
    env = rt.BatchedEnv(tasks[1], 2, actions="label", autoreset=False, goals=True,
                        dead_end=dead_end, dead_end_terminal=terminal, step_reward=-1.0,
                        dead_end_reward=-7.0)
    env.goal_pos.fill_(-1)
    env.refresh()
    schema = torch.zeros(2, dtype=torch.int32)
    binding = torch.empty((2, 0), dtype=torch.int32)
    first = env.step((schema, binding))
    assert (first.schema == 0).all() and not first.invalid.any()
    assert (env.count == 0).all()
    second = env.step((schema, binding))
    assert (second.schema == -1).all() and not second.invalid.any()
    expected_term = dead_end == "no_successors" and terminal
    assert bool(second.terminated.all()) == expected_term
    assert (second.reward == (0.0 if expected_term else -8.0 if dead_end == "no_successors" else -1.0)).all()
