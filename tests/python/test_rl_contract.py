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
    return mymyr.Task(domain.instantiate_string(problem))


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


ENCODED = ("encoding", "obs", "final_obs")  # the encoded observations of Observation / StepResult


def snapshot(result):
    """Owned copies of an Observation's or StepResult's tensors and of the arrays of its encoded observations."""
    import torch

    def keep(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.clone()
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in value.as_dict("numpy").items()}

    return {field: keep(getattr(result, field)) for field in result._fields}


def assert_unchanged(result, saved):
    import torch

    for field in result._fields:
        value, expected = getattr(result, field), saved[field]
        if expected is None or isinstance(expected, torch.Tensor):
            assert value is None if expected is None else torch.equal(value, expected), field
        else:
            actual = value.as_dict("numpy")
            assert actual.keys() == expected.keys(), field
            for key in actual:
                assert np.array_equal(actual[key], expected[key]), (field, key)


@needs_frontend
@pytest.mark.parametrize("encoded", [False, True], ids=["tensors", "encoded"])
def test_torch_observations_survive_steps_and_resets(encoded):
    torch = pytest.importorskip("torch")
    from mymyr import encoding
    from mymyr.rl.torch import BatchedEnv

    task = toggle_task()
    encoder = encoding.FlatRelationEncoder(rl.as_table(task)) if encoded else None
    env = BatchedEnv(task, 2, device="cpu", goals=True, max_steps=1, observation=encoder)
    source = env.observation()
    before = snapshot(source)
    first = env.step(torch.zeros(2, dtype=torch.int64))
    saved = snapshot(first)
    assert not torch.equal(first.final_states, first.states)
    # the encoded observations: None without an encoder, else the encoding of the observed and the reached rows
    assert all((getattr(r, f) is not None) == encoded for r in (source, first) for f in ENCODED if f in r._fields)
    env.step(torch.zeros(2, dtype=torch.int64))
    env.reset()
    assert_unchanged(source, before)
    assert_unchanged(first, saved)
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
    return mymyr.Task(domain.instantiate_string(problem))


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


@pytest.mark.needs_gil
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


@pytest.mark.needs_gil
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


def gripper_task():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    from conftest import ROOT

    d = ROOT / "tests/data/pddl/gripper"
    return mymyr.Task.from_pddl(d / "domain.pddl", d / "p01.pddl")


def test_environment_sizes_and_env_ids_are_checked_up_front():
    """num_envs and first_env are checked by the constructors (and JAX reset) with ValueError: the env ids
    first_env .. first_env + num_envs - 1 are 32-bit RNG streams."""
    task = gripper_task()
    rt = pytest.importorskip("mymyr.rl.torch")
    bad = [dict(num_envs=-1), dict(num_envs=2, first_env=-1), dict(num_envs=2, first_env=2**32 - 1),
           dict(num_envs=2**31)]
    for kw in bad:
        n, first = kw["num_envs"], kw.get("first_env", 0)
        with pytest.raises(ValueError, match="^mymyr: (num_envs|first_env)"):
            rt.BatchedEnv(task, n, first_env=first)
        with pytest.raises(ValueError, match="^mymyr: (num_envs|first_env)"):
            rl.CpuEnvPool(task, n, first_env=first)
    with pytest.raises(TypeError, match="^mymyr: num_envs and first_env must be ints"):
        rt.BatchedEnv(task, 2, first_env=1.5)
    last = rt.BatchedEnv(task, 2, first_env=2**32 - 2)  # the last two streams
    last.reset()
    last.step()
    assert rl.CpuEnvPool(task, 2, first_env=2**32 - 2).first_env == 2**32 - 2
    try:
        rj = jax_rl()
    except pytest.skip.Exception:
        return
    env = rj.Env(task, device="cpu")
    for kw in bad:
        with pytest.raises(ValueError, match="^mymyr: (num_envs|first_env)"):
            env.reset(None, kw["num_envs"], first_env=kw.get("first_env", 0))


def test_actions_must_be_integers_and_wide_ones_are_invalid():
    """A float action raises TypeError in every environment; an integer action outside the int32 range is flagged
    invalid (never wrapped into a valid index)."""
    task = gripper_task()
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    env = rt.BatchedEnv(task, 2)
    env.reset()
    for bad in (torch.tensor([0.7, 1.9]), torch.tensor([True, False]), [0.5, 1.0]):
        with pytest.raises(TypeError, match="^mymyr: actions must be integer"):
            env.step(bad)
    assert env.step(torch.tensor([2**32, 0])).invalid.tolist() == [True, False]
    pool = rl.CpuEnvPool(task, 2)
    with pytest.raises(TypeError, match="integers"):
        pool.step(actions=np.array([0.7, 1.9]))
    assert np.asarray(pool.step(actions=np.array([2**32, 0])).invalid).tolist() == [True, False]
    try:
        rj = jax_rl()
    except pytest.skip.Exception:
        return
    import jax
    import jax.numpy as jnp

    jenv = rj.Env(task, device="cpu")
    state, _ = jenv.reset(None, 2)
    with pytest.raises(TypeError, match="^mymyr: actions must be integer"):
        jenv.step(0, state, jnp.array([0.7, 1.9]))
    for wide in (np.array([2**32, 0], np.int64), np.array([2**31, 0], np.uint32)):
        assert np.asarray(jenv.step(0, state, wide)[5]["invalid"]).tolist() == [True, False]
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        traced = jax.jit(lambda s, a: jenv.step(0, s, a)[5]["invalid"])
        assert np.asarray(traced(state, jnp.array([2**32, 0], jnp.int64))).tolist() == [True, False]
    finally:
        jax.config.update("jax_enable_x64", old)


def test_the_parallel_environments_take_the_same_options():
    """BatchedEnv, CpuEnvPool and the JAX Env share canonical_order (the successor order actions index), first_env
    and the sampler default, and step the same successors under them."""
    from conftest import text_task

    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    rj = None
    try:
        rj = jax_rl()
    except pytest.skip.Exception:
        pass
    task = text_task("gripper__prob05")  # its generator order differs from the canonical one
    n, steps = 8, 12
    runs = {}
    for canonical in (True, False):
        env = rt.BatchedEnv(task, n, seed=3, first_env=7, max_steps=50, canonical_order=canonical)
        assert env.core.canonical_order is canonical
        pool = rl.CpuEnvPool(task, n, seed=3, first_env=7, max_steps=50, canonical_order=canonical, threads=2)
        env.reset()
        pool.reset()
        jenv = state = None
        if rj is not None:
            jenv = rj.Env(task, device="cpu", max_steps=50, canonical_order=canonical)
            state, _ = jenv.reset(None, n, first_env=7)
        rows = []
        for k in range(steps):
            # indices whose successors depend on the order: the last, a third and two thirds into the successors
            action = ((env.count - 1).clamp(min=0) if k % 3 == 0 else env.count * (k % 3) // 3).to(torch.int64)
            r = env.step(action)
            b = pool.step(actions=action.numpy())
            order = np.argsort(np.asarray(b.env_ids))
            got = r.states.numpy().view(np.uint64)
            assert np.array_equal(np.asarray(b.states)[order], got), (canonical, k)
            if jenv is not None:
                state = jenv.step(0, state, action.numpy().astype(np.int32))[0]
                assert np.array_equal(rj.as_words(state.states), got), (canonical, k)
            rows.append(got.copy())
        # the random policy: the env ids' streams from first_env on
        for k in range(3):
            got = env.step().states.numpy().view(np.uint64)
            b = pool.step()
            assert np.array_equal(np.asarray(b.states)[np.argsort(np.asarray(b.env_ids))], got), (canonical, k)
        runs[canonical] = rows
    assert any(not np.array_equal(a, b) for a, b in zip(runs[True], runs[False]))
    # one sampler default: "auto" (uniform without constructor task ids)
    table = rl.TaskTable([task, task])
    samplers = [rt.BatchedEnv(table, 2).sampler, rl.CpuEnvPool(table, 2).sampler]
    if rj is not None:
        samplers.append(rj.Env(table, device="cpu").sampler)
    assert all(isinstance(s, rl.InstanceSampler) and s.weights is None for s in samplers), samplers
    assert rt.BatchedEnv(table, 2, task_ids=[0, 1]).sampler is None
