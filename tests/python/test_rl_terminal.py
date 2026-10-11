"""Solved resets and absorbing goal/dead-end dynamics across the RL entry points."""

import numpy as np
import pytest

import mymyr
from mymyr import rl


def task_from_strings(domain, problem):
    return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem), atoms="frozen")


@pytest.fixture(params=[("cpu", "auto"), ("cuda", "auto"), ("cuda", "general")])
def device_path(request):
    device, path = request.param
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available() or not pytest.importorskip("mymyr.cuda").available():
            pytest.skip("no visible CUDA device")
    return device, path


@pytest.fixture(params=["fluent", "derived", "numeric"])
def solved_task(request):
    if request.param == "numeric":
        return task_from_strings(
            """(define (domain solved) (:requirements :strips :numeric-fluents)
            (:predicates (p)) (:functions (level))
            (:action leave :parameters () :precondition (>= (level) 1)
             :effect (decrease (level) 1)))""",
            """(define (problem solved) (:domain solved) (:init (= (level) 1))
            (:goal (>= (level) 1)))""",
        )
    derived = "(:derived (done) (p object))" if request.param == "derived" else ""
    requirements = ":strips :derived-predicates" if derived else ":strips"
    goal = "done" if derived else "p object"
    return task_from_strings(
        f"""(define (domain solved) (:requirements {requirements}) (:constants object) (:predicates (p ?object) (done)) {derived}
        (:action leave :parameters (?object) :precondition (p ?object) :effect (not (p ?object))))""",
        f"""(define (problem solved) (:domain solved) (:init (p object)) (:goal ({goal})))""",
    )


def test_pool_reports_solved_reset_and_absorbs(solved_task):
    assert solved_task.initial_state.is_goal()
    assert rl.expand(solved_task, [solved_task.initial_state]).counts[0] > 0
    pool = rl.CpuEnvPool(solved_task, num_envs=2, threads=2, autoreset=False, step_reward=-1, goal_reward=5)
    reset = pool.reset()
    assert reset.goal.tolist() == [True, True]
    assert reset.terminated.tolist() == [True, True]
    assert not reset.stepped and reset.reward is None
    for actions in [np.zeros(2, np.int64), np.full(2, 99, np.int64), None]:
        step = pool.step(actions=actions)
        assert np.array_equal(step.states, reset.states)
        assert np.array_equal(step.final_states, reset.states)
        assert step.goal.all() and step.terminated.all()
        assert not step.truncated.any() and not step.invalid.any()
        assert not step.reward.any() and not step.steps.any()
        assert np.all(step.schema == -1) and np.all(step.binding == -1)
        assert np.array_equal(step.count, reset.count)


def test_pool_goal_arrival_pays_once():
    task = task_from_strings(
        """(define (domain arrival) (:requirements :strips) (:predicates (p) (done))
        (:action finish :parameters () :precondition (p) :effect (done)))""",
        """(define (problem arrival) (:domain arrival) (:init (p)) (:goal (done)))""",
    )
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, autoreset=False, step_reward=-1, goal_reward=5)
    assert not pool.reset().goal[0]
    first = pool.step(actions=np.zeros(1, np.int64))
    assert first.goal[0] and first.terminated[0] and first.reward[0] == 4
    second = pool.step(actions=np.zeros(1, np.int64))
    assert second.goal[0] and second.terminated[0] and second.reward[0] == 0
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.steps, second.steps)


def test_partial_pool_reset_uses_each_instances_goal():
    domain = mymyr.Domain.from_string(
        """(define (domain mixed) (:requirements :strips :numeric-fluents) (:predicates (p))
        (:functions (level)) (:action drain :parameters () :precondition (>= (level) 1)
         :effect (decrease (level) 1)))"""
    )
    tasks = [
        mymyr.Task(domain.instantiate_string(
            f"""(define (problem mixed-{level}) (:domain mixed) (:init (= (level) {level}))
            (:goal (>= (level) 1)))"""
        ), atoms="frozen")
        for level in [0, 1]
    ]
    pool = rl.CpuEnvPool(rl.TaskTable(tasks), num_envs=2, threads=2, autoreset=False)
    reset = pool.reset(task_ids=np.array([0, 1], np.int32))
    assert reset.goal.tolist() == [False, True]
    reset = pool.reset(env_ids=np.array([1], np.uint32), task_ids=np.array([0], np.int32))
    assert reset.goal.tolist() == [False] and reset.terminated.tolist() == [True]


def test_goal_query_requires_destination(solved_task):
    from mymyr._core import _rl_torch

    env = _rl_torch.Env(solved_task)
    with pytest.raises(TypeError, match="out"):
        env.goal(env.initial_states(), out=None)


def test_pool_relabelled_positive_negative_goal_masks():
    task = task_from_strings(
        """(define (domain masks) (:requirements :strips) (:predicates (p) (done))
        (:action leave :parameters () :precondition (p) :effect (and (not (p)) (done))))""",
        """(define (problem masks) (:domain masks) (:init (p)) (:goal (done)))""",
    )
    positive = np.asarray(task.encode([task.initial_state]))
    negative = np.asarray(rl.expand(task, [task.initial_state]).succ)
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, goals=True, autoreset=False, goal_reward=5)
    reset = pool.reset(goal_pos=positive, goal_neg=negative)
    assert reset.goal[0] and reset.terminated[0]
    step = pool.step(actions=np.zeros(1, np.int64))
    assert step.goal[0] and step.terminated[0] and step.reward[0] == 0
    assert np.array_equal(step.states, reset.states)


@pytest.mark.parametrize("dead_end,terminal,expected", [("no_successors", True, True), ("no_successors", False, False), ("none", True, False)])
def test_pool_initial_dead_end(dead_end, terminal, expected):
    task = task_from_strings(
        """(define (domain dead) (:requirements :strips) (:predicates (alive) (done))
        (:action finish :parameters () :precondition (alive) :effect (done)))""",
        """(define (problem dead) (:domain dead) (:init) (:goal (done)))""",
    )
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, autoreset=False, step_reward=-1,
                         dead_end=dead_end, dead_end_terminal=terminal, dead_end_reward=-5)
    reset = pool.reset()
    assert not reset.goal[0] and bool(reset.terminated[0]) == expected and reset.count[0] == 0
    step = pool.step()
    assert not step.goal[0] and bool(step.terminated[0]) == expected
    assert step.reward[0] == (0 if expected else -6 if dead_end == "no_successors" else -1)


def test_torchrl_solved_reset(solved_task, device_path):
    torch = pytest.importorskip("torch")
    device, path = device_path
    planning = pytest.importorskip("mymyr.rl.torch").PlanningEnv(solved_task, 2, device=device, path=path)
    try:
        reset = planning.reset()
        assert reset["terminated"].all() and reset["done"].all() and not reset["truncated"].any()
        reset["action"] = torch.zeros(2, dtype=torch.int64, device=device)
        step = planning.step(reset)["next"]
        assert torch.equal(reset["state"], step["state"])
        assert step["terminated"].all() and not step["reward"].any()
    finally:
        planning.close()


def test_mctx_solved_embedding_is_absorbing(solved_task, device_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mctx")
    device, path = device_path
    if device == "cuda":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("no visible JAX CUDA device")
    env = pytest.importorskip("mymyr.rl.jax").Env(solved_task, device=device, path=path, autoreset=False, step_reward=-1, goal_reward=5)
    state, _ = env.reset(None, 2)
    root, invalid = pytest.importorskip("mymyr.rl.jax").mctx_root(
        env, state, lambda params, observation: jnp.ones(2), num_actions=2
    )
    assert not np.asarray(root.value).any() and np.asarray(invalid).all()
    recurrent = pytest.importorskip("mymyr.rl.jax").mctx_recurrent_fn(
        env, lambda params, observation: jnp.ones(2), num_actions=2
    )
    result, reached = jax.jit(recurrent)(None, jax.random.key(0), jnp.zeros(2, jnp.int32), state)
    assert not np.asarray(result.reward).any() and not np.asarray(result.discount).any() and not np.asarray(result.value).any()
    for before, after in zip(state, reached):
        assert np.array_equal(before, after)


@pytest.mark.parametrize("dead_end,terminal", [("none", True), ("no_successors", False), ("no_successors", True)])
def test_mctx_dead_end_is_absorbing_even_with_nonterminal_env(dead_end, terminal):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mctx")
    rj = pytest.importorskip("mymyr.rl.jax")
    task = task_from_strings(
        """(define (domain dead-tree) (:requirements :strips) (:predicates (alive) (done))
        (:action finish :parameters () :precondition (alive) :effect (done)))""",
        """(define (problem dead-tree) (:domain dead-tree) (:init) (:goal (done)))""",
    )
    env = rj.Env(task, device="cpu", autoreset=False, dead_end=dead_end, dead_end_terminal=terminal, step_reward=-1)
    state, _ = env.reset(None, 2)
    recurrent = rj.mctx_recurrent_fn(env, lambda params, observation: jnp.ones(2), num_actions=2)
    result, reached = jax.jit(recurrent)(None, jax.random.key(0), jnp.zeros(2, jnp.int32), state)
    assert not np.asarray(result.reward).any() and not np.asarray(result.discount).any() and not np.asarray(result.value).any()
    for before, after in zip(state, reached):
        assert np.array_equal(before, after)
