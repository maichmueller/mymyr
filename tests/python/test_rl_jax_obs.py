"""mymyr.rl.jax: encoded observations of Env (``observation=``, ``Observation.encoding``, ``info["final_obs"]``)
against the host encoder on the same rows, bit for bit, under ``jax.jit`` and ``lax.scan``, on the CPU and, with a
visible GPU, on CUDA (and the two equal), with per-env goals.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_jax_obs.py
"""

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task
from test_rl_table import sets  # noqa: F401
from test_rl_torch_obs import CAPS, assert_same

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from mymyr._core import _rl_jax  # noqa: E402

if not _rl_jax.has_ffi():
    pytest.skip("mymyr was built without the XLA FFI headers", allow_module_level=True)

import mymyr.rl.jax as rj  # noqa: E402


def devices():
    out = ["cpu"]
    try:
        if jax.devices("gpu"):
            out.append("cuda")
    except RuntimeError:
        pass
    return out


def words(x):
    return np.ascontiguousarray(rj.as_words(np.asarray(x)))


def host(encoder, states, task_ids, goals=None):
    g = None if goals is None else tuple(words(m) for m in goals)
    return encoder.encode(words(states), np.ascontiguousarray(np.asarray(task_ids, np.int32)), goals=g)


def at(batch, k):
    """Step k of a batch stacked by lax.scan."""
    return jax.tree.map(lambda x: x[k], batch)


@pytest.fixture(scope="module")
def gripper(sets):  # noqa: F811
    return rl.TaskTable(sets["gripper"])


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_jitted_step_obs_are_the_host_encoding(gripper, dev, layout):
    encoder = encoding.FlatRelationEncoder(gripper, layout=layout)
    env = rj.Env(gripper, device=dev, max_steps=3, observation=encoder, obs_capacities=CAPS)
    key = jax.random.key(1)
    state, obs = env.reset(key, 16)
    assert_same(obs.encoding, host(encoder, obs.state, obs.task_id))
    step = jax.jit(env.step)
    resets = 0
    for t in range(8):
        before = state
        state, obs, reward, terminated, truncated, info = step(jax.random.key(t), state)
        assert_same(obs.encoding, host(encoder, obs.state, obs.task_id))
        assert_same(info["final_obs"], host(encoder, info["final_state"], before.task_id))
        resets += int(np.asarray(terminated | truncated).sum())
    assert resets > 0


@pytest.mark.parametrize("dev", devices())
def test_exact_obs_of_eager_calls(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rj.Env(gripper, device=dev, max_steps=3, observation=encoder)
    state, obs = env.reset(jax.random.key(2), 8)
    state, obs, *_ = env.step(jax.random.key(3), state)
    assert obs.encoding.sizes is None
    assert_same(obs.encoding, host(encoder, obs.state, obs.task_id))
    with pytest.raises(ValueError, match="capacities are required"):
        jax.jit(env.step)(jax.random.key(3), state)


@pytest.mark.parametrize("dev", devices())
def test_scanned_rollout_obs(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rj.Env(gripper, device=dev, max_steps=3, observation=encoder, obs_capacities=CAPS)
    state, _ = env.reset(jax.random.key(4), 12)
    K = 6

    def body(s, key):
        s2, obs, _, _, _, info = env.step(key, s)
        return s2, (obs.state, obs.task_id, obs.encoding, info["final_state"], s.task_id, info["final_obs"])

    _, (states, ids, obs, final, before, final_obs) = jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(
        jax.random.key(5), K)))(state)
    for k in range(K):
        assert_same(at(obs, k), host(encoder, states[k], ids[k]))
        assert_same(at(final_obs, k), host(encoder, final[k], before[k]))


def test_cpu_and_cuda_obs_are_equal(gripper):
    if "cuda" not in devices():
        pytest.skip("no visible CUDA device for JAX")
    encoder = encoding.FlatRelationEncoder(gripper)
    out = []
    for dev in ("cpu", "cuda"):
        env = rj.Env(gripper, device=dev, max_steps=3, observation=encoder, obs_capacities=CAPS)
        state, _ = env.reset(jax.random.key(6), 10)
        step = jax.jit(env.step)
        batches = []
        for t in range(4):
            state, obs, *_ = step(jax.random.key(t), state)
            batches.append({k: np.asarray(v) for k, v in obs.encoding.as_dict("numpy").items()})
        out.append(batches)
    for a, b in zip(*out):
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(a[key], b[key]), key  # the capacity forms, tails included


@pytest.mark.parametrize("dev", devices())
def test_per_env_goals_are_the_goal_relations(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rj.Env(gripper, device=dev, goals=True, max_steps=3, observation=encoder, obs_capacities=CAPS)
    probe = rj.Env(gripper, device=dev)
    ids = jnp.asarray(np.arange(8) % len(gripper), jnp.int32)
    ps, _ = probe.reset(jax.random.key(0), ids)
    for t in range(3):
        ps, *_ = probe.step(jax.random.key(10 + t), ps)
    gpos = ps.states[:, : 2 * env.words]
    gneg = jnp.zeros_like(gpos)
    state, obs = env.reset(jax.random.key(0), ids, goal_pos=gpos, goal_neg=gneg)
    assert_same(obs.encoding, host(encoder, obs.state, obs.task_id, goals=(obs.goal_pos, obs.goal_neg)))
    step = jax.jit(env.step)
    for t in range(4):
        before = state
        state, obs, _, _, _, info = step(jax.random.key(t), state)
        assert_same(obs.encoding, host(encoder, obs.state, obs.task_id, goals=(obs.goal_pos, obs.goal_neg)))
        assert_same(info["final_obs"], host(encoder, info["final_state"], before.task_id,
                                            goals=(before.goal_pos, before.goal_neg)))


def test_without_autoreset_the_final_obs_is_the_obs(gripper):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rj.Env(gripper, device="cpu", autoreset=False, observation=encoder, obs_capacities=CAPS)
    state, _ = env.reset(jax.random.key(1), 4)
    state, obs, _, _, _, info = env.step(jax.random.key(2), state)
    assert info["final_obs"] is obs.encoding


def test_refusals(gripper, sets):  # noqa: F811
    encoder = encoding.FlatRelationEncoder(gripper)
    with pytest.raises(ValueError, match="no action relations"):
        rj.Env(gripper, device="cpu", observation=encoder, obs_actions=True)
    with pytest.raises(ValueError, match="another table"):
        rj.Env(gripper, device="cpu", observation=encoding.FlatRelationEncoder(rl.TaskTable(sets["gripper"][:2])))
    with pytest.raises(ValueError, match="need an observation encoder"):
        rj.Env(gripper, device="cpu", obs_capacities=CAPS)
    with pytest.raises(TypeError, match="FlatRelationEncoder"):
        rj.Env(gripper, device="cpu", observation=encoding.FlatTransitionEncoder(gripper))


def test_without_an_encoder_there_is_no_encoding():
    task = text_task("gripper__prob05")
    env = rj.Env(task, device="cpu")
    state, obs = env.reset(jax.random.key(0), 2)
    assert obs.encoding is None
    _, obs, _, _, _, info = env.step(jax.random.key(0), state)
    assert obs.encoding is None and "final_obs" not in info


# ------------------------------------------------------------------------------------------------ live rows and suites


@pytest.fixture(scope="module")
def suite(sets):  # noqa: F811
    return rl.TaskSuite([rl.TaskTable(sets["gripper"]), rl.TaskTable(sets["blocks"])])


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_live_rows_are_the_first_rows(gripper, dev, layout):
    from mymyr.encoding import jax as ej

    encoder = encoding.FlatRelationEncoder(gripper, layout=layout)
    env = rj.Env(gripper, device=dev, max_steps=3)
    state, obs = env.reset(jax.random.key(3), 12)
    for t in range(3):
        state, obs, *_ = env.step(jax.random.key(t), state)
    encode = jax.jit(lambda rows, ids, live: ej.encode(encoder, rows, ids, capacities={"graphs": 12, **CAPS}, live=live))
    r = encoder.num_relations
    for live in (0, 1, 7, 12, 30):
        batch = encode(obs.state, obs.task_id, jnp.int32(live))
        k = min(live, 12)
        assert int(batch.device_graphs) == k and batch.num_graphs == k
        if k:
            assert_same(batch, host(encoder, obs.state[:k], obs.task_id[:k]))
        else:
            assert batch.sizes["entities"] == 0
        full = {name: np.asarray(a) for name, a in batch._arrays.items()}
        for name, start in (("task_ids", k), ("num_objects", k), ("entity_offsets", k + 1),
                            ("relation_offsets", k * r + 1), ("relation_arg_offsets", k * r + 1)):
            assert not full[name].reshape(-1)[start:].any(), (live, name)  # zero past the live graphs


def test_live_rows_on_cpu_and_cuda_are_equal(gripper):
    if "cuda" not in devices():
        pytest.skip("no visible CUDA device for JAX")
    from mymyr.encoding import jax as ej

    encoder = encoding.FlatRelationEncoder(gripper)
    env = rj.Env(gripper, device="cpu", max_steps=3)
    state, obs = env.reset(jax.random.key(3), 12)
    state, obs, *_ = env.step(jax.random.key(1), state)
    out = []
    for dev in ("cpu", "cuda"):
        rows, ids = jax.device_put((obs.state, obs.task_id), jax.devices(dev)[0])
        call = jax.jit(lambda r, i: ej.encode(encoder, r, i, capacities={"graphs": 12, **CAPS}, live=jnp.int32(5)))
        out.append([np.asarray(x) for x in jax.tree_util.tree_leaves(call(rows, ids))])
    for a, b in zip(*out, strict=True):
        assert np.array_equal(a, b)


def test_live_rows_need_capacities(gripper):
    from mymyr.encoding import jax as ej

    encoder = encoding.FlatRelationEncoder(gripper)
    rows = jnp.zeros((2, 2 * gripper.words), jnp.uint32)
    with pytest.raises(ValueError, match="need capacities"):
        ej.encode(encoder, rows, jnp.zeros(2, jnp.int32), live=1)


def suite_check(suite, encoders, obs, states, task_ids, goals=None):
    """A SuiteObservation against the host encodings of each domain's rows."""
    ids = np.asarray(task_ids).astype(np.int64)
    domain = np.asarray(suite.domain_of)[ids]
    local = np.asarray(suite.local_ids, np.int32)[ids]
    assert np.array_equal(np.asarray(obs.domain), domain)
    index = np.asarray(obs.index)
    for d, encoder in enumerate(encoders):
        rows = np.flatnonzero(domain == d)
        assert np.array_equal(index[rows], np.arange(len(rows)))
        assert obs.batches[d].num_graphs == len(rows)
        g = None if goals is None else (np.asarray(goals[0])[rows], np.asarray(goals[1])[rows])
        assert_same(obs.batches[d], host(encoder, np.asarray(states)[rows], local[rows], goals=g))


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("goals", [False, True])
def test_suite_obs_are_one_batch_per_domain(suite, dev, goals):
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rj.Env(suite, device=dev, goals=goals, max_steps=3, observation=encoders, obs_capacities=CAPS)
    state, obs = env.reset(jax.random.key(7), 20)

    def masks(o):
        return (o.goal_pos, o.goal_neg) if goals else None

    suite_check(suite, encoders, obs.encoding, obs.state, obs.task_id, masks(obs))
    step = jax.jit(env.step)
    for t in range(5):
        before = state
        state, obs, _, _, _, info = step(jax.random.key(t), state)
        assert isinstance(obs.encoding, rj.SuiteObservation)
        suite_check(suite, encoders, obs.encoding, obs.state, obs.task_id, masks(obs))
        suite_check(suite, encoders, info["final_obs"], info["final_state"], before.task_id,
                    (before.goal_pos, before.goal_neg) if goals else None)


def test_exact_suite_obs(suite):
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rj.Env(suite, device="cpu", max_steps=3, observation=encoders)
    state, obs = env.reset(jax.random.key(8), 10)
    state, obs, *_ = env.step(jax.random.key(1), state)
    suite_check(suite, encoders, obs.encoding, obs.state, obs.task_id)
    assert all(b.sizes is None for b in obs.encoding.batches)


def test_scanned_suite_obs(suite):
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rj.Env(suite, device=devices()[-1], max_steps=3, observation=encoders, obs_capacities=CAPS)
    state, _ = env.reset(jax.random.key(9), 12)
    K = 4

    def body(s, key):
        s2, obs, *_ = env.step(key, s)
        return s2, (obs.state, obs.task_id, obs.encoding)

    _, (states, ids, obs) = jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.key(2), K)))(state)
    for k in range(K):
        suite_check(suite, encoders, at(obs, k), states[k], ids[k])


def test_suite_refusals(suite, gripper):
    with pytest.raises(ValueError, match="2 encoders"):
        rj.Env(suite, device="cpu", observation=encoding.FlatRelationEncoder(gripper))
    encoders = [encoding.FlatRelationEncoder(t) for t in reversed(suite.tables)]
    with pytest.raises(ValueError, match="domain 0 of the suite"):
        rj.Env(suite, device="cpu", observation=encoders)
