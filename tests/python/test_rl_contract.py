"""Small planning tasks that exercise the environment and replay contract independently of benchmarks."""

import numpy as np
import pytest

import mymyr
from mymyr import rl

needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def toggle_task(*, init="", goal="(q)"):
    domain = mymyr.Domain.from_string("""
    (define (domain toggles) (:requirements :strips :negative-preconditions)
     (:predicates (p) (q))
     (:action up :parameters () :precondition (not (p)) :effect (p))
     (:action down :parameters () :precondition (p) :effect (not (p)))
     (:action finish :parameters () :precondition (p) :effect (q)))
    """)
    problem = f"(define (problem tiny) (:domain toggles) (:init {init}) (:goal {goal}))"
    return mymyr.Task(domain.instantiate_string(problem), atoms="frozen")


def jax_rl():
    pytest.importorskip("jax")
    from mymyr._core import _rl_jax

    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without XLA FFI headers")
    from mymyr.rl import jax as rj

    return rj


@needs_frontend
def test_expand_single_empty_state_matches_sequence():
    task = toggle_task()
    assert task.initial_state.num_words == 0
    single = rl.expand(task, task.initial_state, goal=True)
    sequence = rl.expand(task, [task.initial_state], goal=True)
    for field in ("succ", "schema", "binding", "goal", "offsets"):
        np.testing.assert_array_equal(getattr(single, field), getattr(sequence, field))


@needs_frontend
@pytest.mark.parametrize("autoreset", [False, True])
@pytest.mark.parametrize("final_state", [False, True])
def test_replay_reached_state_or_clear_error(autoreset, final_state):
    rj = jax_rl()
    import jax
    import jax.numpy as jnp

    task = toggle_task()
    env = rj.Env(task, device="cpu", max_steps=1, autoreset=autoreset, final_state=final_state)
    state, obs = env.reset(None, 1)
    action = jnp.zeros(1, jnp.int32)
    out = env.step(0, state, action)
    assert np.asarray(out[4]).all()
    build = jax.jit(lambda observation, step: rj.transition_item(observation, action, step))
    if autoreset and not final_state:
        with pytest.raises(ValueError, match="reached state"):
            build(obs, out)
    else:
        item = build(obs, out)
        expected = rl.expand(task, [task.initial_state]).succ[0]
        np.testing.assert_array_equal(rj.as_words(item["next_state"])[0], expected)


@needs_frontend
def test_goal_replay_copies_source_masks():
    rj = jax_rl()
    import jax.numpy as jnp

    task = toggle_task()
    env = rj.Env(task, device="cpu", goals=True, max_steps=1)
    state, obs = env.reset(None, 1)
    action = jnp.zeros(1, jnp.int32)
    item = rj.transition_item(obs, action, env.step(0, state, action))
    assert set(item) == set(rj.transition_spec(env, goals=True))
    np.testing.assert_array_equal(item["goal_pos"], obs.goal_pos)
    np.testing.assert_array_equal(item["goal_neg"], obs.goal_neg)


@needs_frontend
def test_torch_observations_survive_steps_and_resets():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    env = BatchedEnv(toggle_task(), 2, device="cpu", goals=True, max_steps=1)
    source = env.observation()
    before = tuple(tensor.clone() for tensor in source)
    first = env.step(torch.zeros(2, dtype=torch.int64))
    saved = tuple(tensor.clone() for tensor in first)
    assert not torch.equal(first.final_states, first.states)
    env.step(torch.zeros(2, dtype=torch.int64))
    env.reset()
    for actual, expected in zip(source, before):
        assert torch.equal(actual, expected)
    for actual, expected in zip(first, saved):
        assert torch.equal(actual, expected)
    assert env.observation(copy=False).states is env.states
    env.close()


def failure_choice_task():
    domain = mymyr.Domain.from_string("""
    (define (domain failure-choice) (:requirements :strips)
     (:predicates (alive) (halfway) (complete))
     (:action fail :parameters () :precondition (alive) :effect (not (alive)))
     (:action begin :parameters () :precondition (alive) :effect (and (not (alive)) (halfway)))
     (:action finish :parameters () :precondition (halfway) :effect (and (not (halfway)) (complete))))
    """)
    problem = "(define (problem tiny) (:domain failure-choice) (:init (alive)) (:goal (complete)))"
    return mymyr.Task(domain.instantiate_string(problem), atoms="frozen")


@needs_frontend
def test_default_return_prefers_success_over_early_failure():
    task = failure_choice_task()
    expansion = rl.expand(task, [task.initial_state])
    counts = rl.expand(task, expansion.succ).counts
    failure_slot = int(np.flatnonzero(counts == 0)[0])
    success_slot = int(np.flatnonzero(counts > 0)[0])
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1)
    pool.reset()
    failure = pool.step(actions=np.array([failure_slot], np.int32))
    pool.reset()
    first = pool.step(actions=np.array([success_slot], np.int32))
    last = pool.step(actions=np.zeros(1, np.int32))
    assert failure.terminated[0] and not failure.goal[0]
    assert last.terminated[0] and last.goal[0]
    assert float(failure.reward[0]) == 0.0
    assert float(first.reward[0] + last.reward[0]) == 1.0


def test_default_her_reward_matches_goal_reaching_objective():
    states = np.ones((1, 1, 1), np.uint64)
    done = np.ones((1, 1), np.bool_)
    result = rl.her_relabel(states, done, strategy="final", k=1)
    assert np.asarray(result.reward).item() == 1.0


@needs_frontend
def test_torchrl_instance_reset_does_not_reuse_observed_goals():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchrl")
    from mymyr.rl.torch import PlanningEnv

    tasks = [toggle_task(init="(p)", goal="(q)"), toggle_task(init="(q)", goal="(p)")]
    table = rl.TaskTable(tasks)
    env = PlanningEnv(table, 4, goals=True, task_ids=[0] * 4)
    source = env.reset()
    source["_reset"] = torch.tensor([[True], [True], [False], [False]])
    source["next_task_id"] = torch.ones(4, dtype=torch.int64)
    result = env.reset(source)
    assert result["task_id"].tolist() == [1, 1, 0, 0]
    assert not bool(result["terminated"].any())
    expected_pos, expected_neg = rl.goal_masks(table, np.array([1, 1, 0, 0], np.int32))
    np.testing.assert_array_equal(result["goal_pos"].numpy().view(np.uint64), expected_pos)
    np.testing.assert_array_equal(result["goal_neg"].numpy().view(np.uint64), expected_neg)
    env.close()


@needs_frontend
@pytest.mark.parametrize("via_tensordict", [False, True])
def test_torchrl_explicit_reset_goals_remain_supported(via_tensordict):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchrl")
    from mymyr.rl.torch import PlanningEnv

    env = PlanningEnv(toggle_task(), 2, goals=True)
    source = env.reset()
    positive = torch.zeros_like(source["goal_pos"])
    negative = torch.zeros_like(source["goal_neg"])
    if via_tensordict:
        source["next_goal_pos"], source["next_goal_neg"] = positive, negative
        result = env.reset(source)
    else:
        result = env.reset(goal_pos=positive, goal_neg=negative)
    assert bool(result["terminated"].all())
    assert not bool(result["goal_pos"].any())
    assert not bool(result["goal_neg"].any())
    env.close()
