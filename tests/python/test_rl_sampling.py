"""Instance curricula are deterministic, independent of batch layout and safe in compiled framework loops."""

import numpy as np
import pytest

import mymyr
from mymyr import rl


@pytest.fixture(scope="module")
def table():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = mymyr.Domain.from_string("""
    (define (domain toggles)
      (:requirements :strips :negative-preconditions)
      (:predicates (p) (q))
      (:action up :parameters () :precondition (not (p)) :effect (p))
      (:action down :parameters () :precondition (p) :effect (not (p))))
    """)
    tasks = [mymyr.Task(domain.instantiate_string(f"""
        (define (problem instance{index}) (:domain toggles) (:init {initial}) (:goal (q)))
    """)) for index, initial in enumerate(["", "(p)"])]
    return rl.TaskTable(tasks)


@pytest.mark.parametrize("weights", [[1], [-1, 2], [0, 0], [np.nan, 1], [np.inf, 1]])
def test_sampler_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        rl.InstanceSampler(2, weights)


def test_numpy_sampler_distribution_and_streams():
    streams = np.arange(20000, dtype=np.uint32)
    counts = np.zeros_like(streams)
    uniform = rl.InstanceSampler(4)
    draws = uniform.sample(37, streams, counts)
    assert draws.dtype == np.int32
    assert np.all(np.abs(np.bincount(draws) / len(draws) - 0.25) < 0.02)
    assert np.array_equal(draws, np.concatenate([
        uniform.sample(37, streams[:123], counts[:123]), uniform.sample(37, streams[123:], counts[123:])
    ]))
    assert not np.array_equal(draws, uniform.sample(37, streams, counts + 1))
    weighted = rl.InstanceSampler(4, [0, 1, 0, 3]).sample(37, streams, counts)
    assert set(weighted) == {1, 3}
    assert abs(np.mean(weighted == 3) - 0.75) < 0.02
    high_seed = uniform.sample(2**40 + 37, streams, counts)
    assert not np.array_equal(draws, high_seed)


@pytest.mark.needs_gil
@pytest.mark.torch_compile
def test_torch_numpy_sampling_and_compile():
    torch = pytest.importorskip("torch")
    for sampler in [rl.InstanceSampler(5), rl.InstanceSampler(5, [0, 1, 0, 3, 2])]:
        streams = np.arange(256, dtype=np.uint32)
        counters = streams % 7
        expected = sampler.sample(2**40 + 3, streams, counters)
        run = torch.compile(lambda ids, draws: sampler.sample(2**40 + 3, ids, draws), backend="eager", fullgraph=True)
        result = run(torch.from_numpy(streams.astype(np.int64)), torch.from_numpy(counters.astype(np.int64)))
        assert np.array_equal(result.numpy(), expected)


def test_batched_sampler_partial_reset_and_reseed(table):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    whole = rt.BatchedEnv(table, 32, seed=9, max_steps=1)
    first = rt.BatchedEnv(table, 11, seed=9, max_steps=1)
    second = rt.BatchedEnv(table, 21, seed=9, max_steps=1, first_env=11)
    assert torch.equal(whole.task_ids, torch.cat([first.task_ids, second.task_ids]))
    mask = torch.arange(32) % 3 == 0
    before_ids, before_counts = whole.task_ids.clone(), whole._reset_counts.clone()
    whole.reset(mask)
    first.reset(mask[:11])
    second.reset(mask[11:])
    assert torch.equal(whole.task_ids, torch.cat([first.task_ids, second.task_ids]))
    assert torch.equal(whole.task_ids[~mask], before_ids[~mask])
    assert torch.equal(whole._reset_counts, before_counts + mask.long())
    for _ in range(3):
        whole.step()
        first.step()
        second.step()
        assert torch.equal(whole.task_ids, torch.cat([first.task_ids, second.task_ids]))
    whole.set_seed(9)
    whole.reset()
    fresh = rt.BatchedEnv(table, 32, seed=9, max_steps=1)
    assert torch.equal(whole.task_ids, fresh.task_ids)
    expected = whole._reset_counts.clone()
    whole.reset(mask, task_ids=torch.ones(32, dtype=torch.int32))
    assert bool((whole.task_ids[mask] == 1).all())
    assert torch.equal(whole._reset_counts, expected + mask.long())


@pytest.mark.parametrize("weights", [None, [1, 3]])
@pytest.mark.parametrize("threads", [1, 3])
def test_pool_samples_the_batched_envs_instances(table, weights, threads):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    sampler = "auto" if weights is None else rl.InstanceSampler(2, weights)
    n = 12
    env = rt.BatchedEnv(table, n, seed=5, max_steps=2, sampler=sampler)
    pool = rl.CpuEnvPool(table, num_envs=n, threads=threads, seed=5, max_steps=2, sampler=sampler)
    assert (pool.sampler is not None) and pool.sampler.num_instances == 2

    def ids(batch):
        return np.asarray(batch.task_ids)[np.argsort(np.asarray(batch.env_ids))].tolist()

    env.reset()
    assert ids(pool.reset()) == env.task_ids.tolist()
    mask = np.arange(n) % 4 == 1
    env.reset(torch.from_numpy(mask))
    part = pool.reset(env_ids=np.flatnonzero(mask).astype(np.uint32))
    assert ids(part) == env.task_ids[torch.from_numpy(mask)].tolist()
    for _ in range(7):  # max_steps 2: the truncations autoreset into sampled instances
        env.step(torch.zeros(n, dtype=torch.int64))
        assert ids(pool.step(actions=np.zeros(n, np.int64))) == env.task_ids.tolist()
    # explicit ids override the draw and still advance the row's counter, as in BatchedEnv
    env.reset(torch.from_numpy(mask), task_ids=torch.ones(n, dtype=torch.int32))
    pool.reset(env_ids=np.flatnonzero(mask).astype(np.uint32), task_ids=np.ones(int(mask.sum()), np.int32))
    env.reset()
    assert ids(pool.reset()) == env.task_ids.tolist()
    fixed = rl.CpuEnvPool(table, num_envs=4, max_steps=1, sampler=None)
    assert fixed.sampler is None
    assert ids(fixed.reset()) == [0, 0, 0, 0]
    assert ids(fixed.reset(task_ids=np.array([1, 0, 1, 0], np.int32))) == [1, 0, 1, 0]
    assert ids(fixed.step(actions=np.zeros(4, np.int64))) == [1, 0, 1, 0]


@pytest.mark.parametrize("weights", [None, [1, 3]])
def test_pool_instances_are_the_numpy_samplers_draws(table, weights):
    """The pool's resets and autoresets draw InstanceSampler.sample(seed, env id, the env's reset counter); the
    constructor's start takes counter 0 and every reset or autoreset of an env advances its counter."""
    sampler = rl.InstanceSampler(2, weights)
    n = 16
    pool = rl.CpuEnvPool(table, num_envs=n, threads=3, seed=9, max_steps=1, sampler=sampler)
    env_ids = np.arange(n, dtype=np.uint32)

    def ids(batch):
        return np.asarray(batch.task_ids)[np.argsort(np.asarray(batch.env_ids))]

    def draws(k):
        return sampler.sample(9, env_ids, np.full(n, k, np.uint32))

    assert np.array_equal(ids(pool.reset()), draws(1))
    for k in range(2, 6):  # max_steps 1: every step autoresets each env into its next draw
        assert np.array_equal(ids(pool.step(actions=np.zeros(n, np.int64))), draws(k))


@pytest.mark.needs_gil
@pytest.mark.torch_compile
def test_batched_sampler_compiled_autoreset_and_fixed_ids(table):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    eager = rt.BatchedEnv(table, 16, seed=3, max_steps=1)
    compiled = rt.BatchedEnv(table, 16, seed=3, max_steps=1)
    run = torch.compile(lambda: compiled.step(), backend="eager", fullgraph=True)
    for _ in range(4):
        eager.step()
        run()
        assert torch.equal(compiled.task_ids, eager.task_ids)
        assert torch.equal(compiled._reset_counts, eager._reset_counts)
    fixed = rt.BatchedEnv(table, 16, task_ids=[1] * 16, max_steps=1)
    fixed.reset()
    fixed.step()
    assert fixed.task_ids.tolist() == [1] * 16
    weighted = rt.BatchedEnv(table, 16, sampler=rl.InstanceSampler(2, [0, 1]), max_steps=1)
    weighted.step()
    assert weighted.task_ids.tolist() == [1] * 16


@pytest.mark.needs_gil
def test_planning_sampler_rollout_and_external_curriculum(table):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchrl")
    rt = pytest.importorskip("mymyr.rl.torch")
    from tensordict import TensorDict

    env = rt.PlanningEnv(table, 16, seed=4, max_steps=1)
    rollout = env.rollout(8, break_when_any_done=False)
    assert torch.unique(rollout["task_id"]).tolist() == [0, 1]
    assert bool((rollout["task_id"][:, 1:] != rollout["task_id"][:, :-1]).any())
    mask = torch.arange(16) % 2 == 0
    previous = env.batched.task_ids.clone()
    out = env.reset(TensorDict({
        "_reset": mask.unsqueeze(-1), "task_id": torch.zeros(16, dtype=torch.int64),
        "next_task_id": torch.ones(16, dtype=torch.int64),
    }, batch_size=[16]))
    assert bool((env.batched.task_ids[mask] == 1).all())
    assert torch.equal(env.batched.task_ids[~mask], previous[~mask])
    assert bool((out["task_id"][mask] == 1).all())
    env.reset(task_ids=torch.zeros(16, dtype=torch.int32))
    assert env.batched.task_ids.tolist() == [0] * 16


@pytest.mark.parametrize("weights", [None, [0, 1, 0, 3, 2]])
def test_jax_sampling_jit_vmap_and_scan(weights):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    sampler = rl.InstanceSampler(5, weights)
    streams = np.arange(256, dtype=np.uint32)
    counts = streams % 7
    expected = sampler.sample(2**40 + 3, streams, counts)
    run = jax.jit(lambda ids, draws: sampler.sample(2**40 + 3, ids, draws))
    assert np.array_equal(np.asarray(run(jnp.asarray(streams), jnp.asarray(counts))), expected)
    mapped = jax.jit(jax.vmap(lambda ids, draws: sampler.sample(2**40 + 3, ids, draws)))
    assert np.array_equal(np.asarray(mapped(jnp.asarray(streams.reshape(4, 64)), jnp.asarray(counts.reshape(4, 64)))),
                          expected.reshape(4, 64))

    def advance(counters, _):
        return counters + 1, sampler.sample(3, jnp.asarray(streams), counters)

    _, values = jax.jit(lambda: jax.lax.scan(advance, jnp.asarray(counts), None, length=4))()
    assert np.array_equal(np.asarray(values), np.stack([sampler.sample(3, streams, counts + step) for step in range(4)]))


def test_jax_env_sampler_partial_reset_autoreset_and_reseed(table):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rj = pytest.importorskip("mymyr.rl.jax")
    env = rj.Env(table, max_steps=1)
    whole, _ = jax.jit(lambda key: env.reset(key, 32))(jax.random.key(9))
    first, _ = env.reset(9, 11)
    second, _ = env.reset(9, 21, first_env=11)
    assert np.array_equal(np.asarray(whole.task_id), np.concatenate([first.task_id, second.task_id]))
    mask = jnp.arange(32) % 3 == 0
    whole, _ = jax.jit(env.reset_where)(whole, mask)
    first, _ = env.reset_where(first, mask[:11])
    second, _ = env.reset_where(second, mask[11:])
    assert np.array_equal(np.asarray(whole.task_id), np.concatenate([first.task_id, second.task_id]))

    def advance(state, _):
        next_state, *_ = env.step(123, state)
        return next_state, next_state.task_id

    result, values = jax.jit(lambda state: jax.lax.scan(advance, state, None, length=4))(whole)
    assert np.array_equal(np.asarray(result.reset_counts), np.asarray(whole.reset_counts) + 4)
    assert set(np.asarray(values).flat) == {0, 1}
    restarted, _ = jax.jit(lambda state: env.reset_where(state, jnp.ones(32, jnp.bool_), key=9))(result)
    fresh, _ = env.reset(9, 32)
    assert np.array_equal(restarted.task_id, fresh.task_id)
    override, _ = jax.jit(lambda state: env.step(7, state, next_task_ids=jnp.ones(32, jnp.int32))[0:2])(result)
    assert np.asarray(override.task_id).tolist() == [1] * 32
    halves = jax.tree.map(lambda array: array.reshape(2, 16, *array.shape[1:]), whole)
    mapped, *_ = jax.jit(jax.vmap(lambda state: env.step(123, state)))(halves)
    expected, *_ = env.step(123, whole)
    assert np.array_equal(np.asarray(mapped.task_id).reshape(-1), expected.task_id)
    alternate_key, *_ = env.step(999, whole)
    assert np.array_equal(alternate_key.task_id, expected.task_id)


def test_environment_rejects_mismatched_sampler(table):
    rt = pytest.importorskip("mymyr.rl.torch")
    with pytest.raises(ValueError, match="instance count"):
        rt.BatchedEnv(table, 2, sampler=rl.InstanceSampler(3))
    with pytest.raises(ValueError, match="sampler must"):
        rt.BatchedEnv(table, 2, sampler="unknown")


def test_jax_weighted_and_disabled_sampling(table):
    pytest.importorskip("jax")
    rj = pytest.importorskip("mymyr.rl.jax")
    weighted = rj.Env(table, sampler=rl.InstanceSampler(2, [0, 1]), max_steps=1)
    state, _ = weighted.reset(3, 16)
    state, *_ = weighted.step(7, state)
    assert np.asarray(state.task_id).tolist() == [1] * 16
    fixed = rj.Env(table, sampler=None, max_steps=1)
    state, _ = fixed.reset(3, np.ones(16, np.int32))
    state, *_ = fixed.step(7, state)
    assert np.asarray(state.task_id).tolist() == [1] * 16
