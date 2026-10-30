"""Per-environment goals use the written instance's fluent slots across environment backends."""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, text_task


@pytest.fixture(params=["classical", "numeric"])
def suite(request):
    tasks = [text_task("blocks__probBLOCKS-8-0", atoms="dense"), text_task("gripper__prob05", atoms="dense")]
    if request.param == "numeric":
        tasks[1] = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    assert tasks[0].words != tasks[1].words
    return rl.TaskSuite.group(tasks)


def masks(suite, ids, mode):
    pos, neg = (np.array(a, copy=True) for a in rl.goal_masks(suite, ids))
    if mode == "kept":
        # A relabelled goal that fits both layouts stays valid across an instance change.
        pos[:] = 0
        neg[:] = 0
        common = min(task.num_atoms for task in suite.tasks)
        if common:
            pos[:, (common - 1) // 64] = np.uint64(1) << np.uint64((common - 1) % 64)
    elif mode == "stale":
        pos[:], neg[:] = rl.goal_masks(suite, 1 - ids)
        pos[1] = 0
        large = suite.tasks[0].num_atoms - 1
        pos[1, large // 64] = np.uint64(1) << np.uint64(large % 64)
    elif mode in ("positive", "negative"):
        bad = pos if mode == "positive" else neg
        slot = suite.tasks[1].num_atoms
        bad[1, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return pos, neg


def halves(a):
    return np.ascontiguousarray(a, dtype=np.uint64).view(np.uint32)


def words(a):
    return np.ascontiguousarray(np.asarray(a), dtype=np.uint32).view(np.uint64)


@pytest.mark.parametrize("mode", ["kept", "new", "stale", "positive", "negative"])
@pytest.mark.parametrize("tracked", [False, 8])
@pytest.mark.parametrize("backend", ["torch_cpu", "torch_cuda", "torch_general", "jax_cpu", "jax_cuda", "jax_general"])
def test_direct_instance_write(suite, backend, tracked, mode, request):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    cuda = backend.endswith(("cuda", "general"))
    if cuda:
        if not torch.cuda.is_available() or not pytest.importorskip("mymyr.cuda", exc_type=ImportError).available():
            pytest.skip("no visible CUDA device")
    cfg = dict(goals=True, sampler=None, autoreset=False, max_steps=3, track_visited=tracked)
    path = "general" if backend.endswith("general") else "auto"
    ids = np.array([0, 1], np.int32)
    pos, neg = masks(suite, ids, mode)
    invalid = mode in ("stale", "positive", "negative")
    device = "cuda" if cuda else "cpu"
    target = BatchedEnv(suite, 2, task_ids=ids, **cfg)
    request.addfinalizer(target.close)
    states = target.states.numpy().view(np.uint64).copy()
    as_tensor = lambda a, d=device: torch.from_numpy(np.ascontiguousarray(a).view(np.int64)).to(d)

    if backend.startswith("torch"):
        env = BatchedEnv(suite, 2, task_ids=1 - ids, device=device, path=path, **cfg)
        request.addfinalizer(env.close)
        if mode in ("kept", "stale"):
            env.goal_pos.copy_(as_tensor(pos))
            env.goal_neg.copy_(as_tensor(neg))
        env.states.copy_(as_tensor(states))
        env.task_ids.copy_(torch.from_numpy(ids).to(device))
        if mode not in ("kept", "stale"):
            env.goal_pos.copy_(as_tensor(pos))
            env.goal_neg.copy_(as_tensor(neg))
        if invalid:
            if cuda:
                env.refresh()
                with pytest.raises(ValueError, match="goal masks.*fluent slots"):
                    env.check_errors()
                env.check_errors()  # checking consumes the sticky error
            else:
                with pytest.raises(ValueError, match="goal masks.*fluent slots"):
                    env.refresh()
                with pytest.raises(ValueError, match="goal masks.*fluent slots"):
                    env.is_goal()
            pos, neg = masks(suite, ids, "new")
            env.goal_pos.copy_(as_tensor(pos))
            env.goal_neg.copy_(as_tensor(neg))
        env.refresh()
        if tracked:
            env.clear_visited()
        actual_goal = env.at_goal.cpu().numpy()
        step = lambda a: env.step(torch.from_numpy(a).to(device))
        convert = lambda x: x.cpu().numpy()
        state_fields = lambda: dict(states=convert(env.states).view(np.uint64), task_ids=convert(env.task_ids),
                                    steps=convert(env.steps), count=convert(env.count), at_goal=convert(env.at_goal),
                                    goal_pos=convert(env.goal_pos).view(np.uint64),
                                    goal_neg=convert(env.goal_neg).view(np.uint64))
    else:
        jax = pytest.importorskip("jax")
        from mymyr._core import _rl_jax
        if not _rl_jax.has_ffi():
            pytest.skip("built without XLA FFI")
        from mymyr.rl.jax import Env
        env = Env(suite, device=device, path=path, **cfg)
        state, _ = env.reset(0, 1 - ids)
        state = state._replace(states=jax.device_put(halves(states), env.device),
                               task_id=jax.device_put(ids, env.device),
                               goal_pos=jax.device_put(halves(pos), env.device),
                               goal_neg=jax.device_put(halves(neg), env.device))
        if invalid:
            if cuda:
                state = env.refresh(state)
                jax.block_until_ready(state)
                with pytest.raises(ValueError, match="goal masks.*fluent slots"):
                    env.check_errors()
                env.check_errors()
            else:
                with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="goal masks.*fluent slots"):
                    jax.block_until_ready(env.refresh(state))
            pos, neg = masks(suite, ids, "new")
            state = state._replace(goal_pos=jax.device_put(halves(pos), env.device),
                                   goal_neg=jax.device_put(halves(neg), env.device))
        state = env.refresh(state)
        if tracked:
            state = env.clear_visited(state)
        actual_goal = np.asarray(state.at_goal)

        def step(a):
            nonlocal state
            state, _, reward, terminated, truncated, info = env.step(0, state, a.astype(np.int32))
            return dict(reward=reward, terminated=terminated, truncated=truncated, **info)

        convert = np.asarray
        state_fields = lambda: dict(states=words(state.states), task_ids=np.asarray(state.task_id),
                                    steps=np.asarray(state.steps), count=np.asarray(state.count),
                                    at_goal=np.asarray(state.at_goal), goal_pos=words(state.goal_pos),
                                    goal_neg=words(state.goal_neg))

    try:
        target.goal_pos.copy_(as_tensor(pos, "cpu"))
        target.goal_neg.copy_(as_tensor(neg, "cpu"))
        target.refresh()
        np.testing.assert_array_equal(actual_goal, target.at_goal.numpy())
        pool = rl.CpuEnvPool(suite, 2, words=suite.words, threads=1, **cfg)
        reset = pool.reset(task_ids=ids, goal_pos=pos, goal_neg=neg)
        np.testing.assert_array_equal(actual_goal, np.asarray(reset.goal))
        for i in range(5):
            action = np.full(2, -1 if i == 1 else 0, np.int64)
            expected, got, pooled = target.step(torch.from_numpy(action)), step(action), pool.step(actions=action)
            for field in ("reward", "terminated", "truncated", "goal", "invalid", "revisited", "schema", "binding"):
                value = got[field] if isinstance(got, dict) else getattr(got, field)
                np.testing.assert_array_equal(convert(value), getattr(expected, field).numpy(), err_msg=field)
                np.testing.assert_array_equal(np.asarray(getattr(pooled, field)), getattr(expected, field).numpy())
            for field, value in state_fields().items():
                expected_value = getattr(target, field).numpy()
                if field in ("states", "goal_pos", "goal_neg"):
                    expected_value = expected_value.view(np.uint64)
                np.testing.assert_array_equal(value, expected_value, err_msg=field)
        env.check_errors()
    finally:
        if backend.startswith("torch"):
            env.close()
        target.close()


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("selection", [None, [True, True, False], [False, True, False], [True, False, False]])
def test_torch_rejected_reset_preserves_environment(suite, negative, padding, selection):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    ids = np.array([0, 1, 0], np.int32)
    env = BatchedEnv(suite, 3, task_ids=1 - ids, goals=True, sampler="uniform", max_steps=4, track_visited=8)
    try:
        env.step()
        before = [t.clone() for t in env._tensors()]
        pos, neg = masks(suite, ids, "new")
        slot = suite.tasks[1].words * 64 if padding else suite.tasks[1].num_atoms
        assert slot < 64 * suite.words
        (neg if negative else pos)[1, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
        mask = None if selection is None else torch.tensor(selection)
        kwargs = dict(task_ids=torch.from_numpy(ids), goal_pos=torch.from_numpy(pos.view(np.int64)),
                      goal_neg=torch.from_numpy(neg.view(np.int64)))
        if selection is None or selection[1]:
            with pytest.raises(ValueError, match="goal masks.*fluent slots"):
                env.reset(mask, **kwargs)
            for actual, expected in zip(env._tensors(), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            env.reset(mask, **kwargs)
            for actual, expected in zip(env._tensors(), before):
                if actual.shape[0] == env.num_envs:
                    torch.testing.assert_close(actual[~mask], expected[~mask], rtol=0, atol=0)
        env.step()
        env.check_errors()
    finally:
        env.close()


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_pool_rejects_goal_bits_outside_instance(suite, negative, padding):
    ids = np.array([0, 1], np.int32)
    pos, neg = masks(suite, ids, "new")
    slot = suite.tasks[1].words * 64 if padding else suite.tasks[1].num_atoms
    assert slot < 64 * suite.words
    bad = neg if negative else pos
    bad[1, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    pool = rl.CpuEnvPool(suite, 2, goals=True, sampler=None, threads=1)
    before = pool.reset(task_ids=1 - ids)
    with pytest.raises(ValueError, match="goal masks.*fluent slots"):
        pool.reset(task_ids=ids, goal_pos=pos, goal_neg=neg)
    after = pool.step(actions=np.full(2, -1, np.int64))
    np.testing.assert_array_equal(before.states, after.states)
    np.testing.assert_array_equal(before.task_ids, after.task_ids)


@pytest.mark.parametrize("device,path", [("cpu", "auto"), ("cuda", "auto"), ("cuda", "general")])
def test_jax_reset_validates_custom_goals(suite, device, path):
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("built without XLA FFI")
    if device == "cuda":
        if not pytest.importorskip("mymyr.cuda", exc_type=ImportError).available():
            pytest.skip("no visible CUDA device")
    from mymyr.rl.jax import Env

    env = Env(suite, device=device, path=path, goals=True, sampler=None)
    ids = np.array([0, 1], np.int32)
    pos, neg = (halves(a) for a in masks(suite, ids, "negative"))
    if device == "cpu":
        with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="goal masks.*fluent slots"):
            jax.block_until_ready(env.reset(0, ids, goal_pos=pos, goal_neg=neg))
    else:
        jax.block_until_ready(env.reset(0, ids, goal_pos=pos, goal_neg=neg))
        with pytest.raises(ValueError, match="goal masks.*fluent slots"):
            env.check_errors()
    state, _ = env.reset(0, ids)
    valid_pos, valid_neg = (halves(a) for a in masks(suite, ids, "new"))
    custom, _ = env.reset(0, ids, goal_pos=valid_pos, goal_neg=valid_neg)
    for field in ("states", "count", "at_goal", "reset_counts", "draws"):
        np.testing.assert_array_equal(getattr(custom, field), getattr(state, field))
    # Invalid assignments to rows outside the reset mask are ignored.
    state, _ = env.reset_where(state, np.array([True, False]), goal_pos=pos, goal_neg=neg)
    jax.block_until_ready(state)
    env.check_errors()
    before = [np.array(a, copy=True) for a in jax.tree.leaves(state)]
    if device == "cpu":
        with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="goal masks.*fluent slots"):
            jax.block_until_ready(env.reset_where(state, np.array([True, True]), goal_pos=pos, goal_neg=neg))
    else:
        jax.block_until_ready(env.reset_where(state, np.array([True, True]), goal_pos=pos, goal_neg=neg))
        with pytest.raises(ValueError, match="goal masks.*fluent slots"):
            env.check_errors()
    for actual, expected in zip(jax.tree.leaves(state), before):
        np.testing.assert_array_equal(actual, expected)
    jax.block_until_ready(env.step(0, state))
    env.check_errors()
