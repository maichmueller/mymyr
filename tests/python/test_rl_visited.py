"""Episode membership agrees with sets of States across host environment implementations."""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, text_task


def task_of(name):
    if name == "numeric":
        return mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    return text_task(name)


def state_hashes(states, tasks):
    """The visited-table keys of States: their rows hashed with their task's numeric words."""
    return np.concatenate([np.asarray(mymyr.hash_rows(task.encode([state]), numeric_words=task.numeric_words), np.uint64)
                           for state, task in zip(states, tasks)])


def decoded(table, batch):
    return rl.decode(table, np.asarray(batch.states), np.asarray(batch.task_ids))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("autoreset", [False, True])
@pytest.mark.parametrize("name", ["gripper__prob05", "blocks__probBLOCKS-8-0", "pegsol-08-strips__p22", "numeric", "suite"])
def test_rollouts_match_state_sets(name, threads, autoreset):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    tasks = [task_of("gripper__prob05"), task_of("numeric")] if name == "suite" else [task_of(name)]
    table = rl.TaskSuite.group(tasks) if name == "suite" else rl.TaskTable(tasks)
    n = 12
    cfg = dict(seed=71, threads=threads, autoreset=autoreset, max_steps=13, track_visited=True, sampler=None)
    env = BatchedEnv(table, n, **cfg)
    pool = rl.CpuEnvPool(table, n, **cfg)
    ids = np.arange(n, dtype=np.int32) % len(tasks)
    batch = pool.reset(task_ids=ids)
    env.reset(task_ids=torch.from_numpy(ids))
    history = [{s} for s in decoded(table, batch)]
    absorbing = np.zeros(n, dtype=bool)
    for step in range(40):
        previous = env.states.clone()
        count = env.count.clone()
        # Invalid actions exercise the stay semantics without changing the RNG stream.
        actions = np.full(n, -1, dtype=np.int64) if step % 7 == 0 else None
        a = env.step(None if actions is None else torch.from_numpy(actions))
        b = pool.step(actions=actions)
        for field in ("states", "task_ids", "reward", "terminated", "truncated", "goal", "invalid", "schema", "binding", "revisited"):
            left, right = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
            if field == "states":
                left = left.view(np.uint64)
            np.testing.assert_array_equal(left, right)
        reached = rl.decode(table, np.asarray(b.final_states), ids)
        observed = decoded(table, b)
        done = np.asarray(b.terminated) | np.asarray(b.truncated)
        for i, (state, next_state) in enumerate(zip(reached, observed)):
            initial_absorbed = bool(env.core.table.tasks[int(ids[i])].is_goal(
                rl.decode(table, np.asarray(previous[i:i+1]), ids[i:i+1])[0])) or int(count[i]) == 0
            expected = state in history[i] and not (not autoreset and (absorbing[i] or initial_absorbed))
            assert bool(b.revisited[i]) == expected
            if not absorbing[i]:
                history[i].add(state)
            if done[i] and autoreset:
                history[i] = {next_state}
            if done[i] and not autoreset:
                absorbing[i] = True
        if not autoreset and done.any():
            mask = np.asarray(done, dtype=bool).copy()
            b = pool.reset(env_ids=np.flatnonzero(mask).astype(np.int32))
            env.reset(torch.from_numpy(mask))
            for i, state in zip(np.asarray(b.env_ids), decoded(table, b)):
                history[int(i)] = {state}
                absorbing[int(i)] = False
        if step % 9 == 0:
            exp = rl.expand(table, env.states, env._ids(), framework="numpy", hashes=True)
            parent = np.asarray(exp.parent)
            successors = rl.decode(table, np.asarray(exp.succ), ids[parent])
            hashes = np.asarray(exp.succ_hash)
            np.testing.assert_array_equal(hashes, state_hashes(successors, [tasks[int(ids[p])] for p in parent]))
            got = rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), hashes, parent)
            np.testing.assert_array_equal(got, [s in history[int(p)] for s, p in zip(successors, parent)])
        assert not env.visited_status.any()
        assert pool.visited_status == 0
    assert env.visited_capacity == pool.visited_capacity == 32
    env.close()


@pytest.mark.parametrize("threads", [1, 4])
def test_overflow_only_misses_revisits(threads):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    cfg = dict(track_visited=2, threads=threads, autoreset=False, dead_end="none", seed=4)
    env = BatchedEnv(task, 16, **cfg)
    pool = rl.CpuEnvPool(task, 16, **cfg)
    seen = [{task.initial_state} for _ in range(16)]
    for _ in range(40):
        a, b = env.step(), pool.step()
        np.testing.assert_array_equal(a.revisited, b.revisited)
        for i, state in enumerate(task.decode(np.asarray(b.final_states))):
            if b.revisited[i]:
                assert state in seen[i]
            seen[i].add(state)
    assert env.visited_status.item() == pool.visited_status == 1
    assert (env.visited_count == 2).any()
    env.clear_visited()
    pool.clear_visited()
    assert torch.all(env.visited_count == 1)
    assert env.visited_status.item() == pool.visited_status == 1
    a, b = env.step(torch.full((16,), -1)), pool.step(actions=[-1] * 16)
    assert a.revisited.all() and np.asarray(b.revisited).all()


def test_masks_and_clear_after_state_writes():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    env = BatchedEnv(task, 6, track_visited=64, autoreset=False, seed=9)
    initial = env.states.clone()
    for _ in range(5):
        env.step()
    old = env.visited.clone()
    old_count = env.visited_count.clone()
    env.states[:3].copy_(initial[:3])
    env.refresh()
    assert torch.equal(env.visited, old) and torch.equal(env.visited_count, old_count)
    env.clear_visited(torch.tensor([True] * 3 + [False] * 3))
    assert torch.equal(env.visited[3:], old[3:])
    assert torch.all(env.visited_count[:3] == 1)
    exp = rl.expand(task, env.states, framework="numpy", hashes=True)
    parent = np.asarray(exp.parent)
    states = task.decode(np.asarray(exp.succ))
    hashes = np.asarray(exp.succ_hash)
    mask = rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), hashes, parent)
    expected = []
    for k, (state, p) in enumerate(zip(states, parent)):
        expected.append(state == task.initial_state if p < 3 else bool(rl.visited_mask(
            old.numpy(), old_count.numpy(), hashes[k : k + 1], np.array([p], np.int32))[0]))
    np.testing.assert_array_equal(mask, expected)
    thashes = torch.from_numpy(hashes.view(np.int64))
    np.testing.assert_array_equal(rl.visited_mask(env.visited, env.visited_count, thashes, torch.from_numpy(parent)), mask)
    subset = np.array([5, 0, 2], np.int32)
    picked = np.isin(parent, subset)
    mapped = np.array([int(np.flatnonzero(subset == p)[0]) for p in parent[picked]], np.int32)
    np.testing.assert_array_equal(rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), hashes[picked], mapped,
                                               rows=subset), mask[picked])
    np.testing.assert_array_equal(rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), np.zeros(2, np.uint64),
                                               np.array([-1, 0], np.int32), rows=np.array([-1], np.int32)), [False, False])


@pytest.mark.parametrize("capacity", [0, 1, 3, -1, 2**21, 2.5])
def test_invalid_capacity(capacity):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    for construct in (BatchedEnv, rl.CpuEnvPool):
        with pytest.raises((ValueError, TypeError)):
            construct(task, 2, track_visited=capacity)


@pytest.mark.parametrize("max_steps", [None, 524288])
def test_automatic_capacity_requires_bounded_episode(max_steps):
    pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    for construct in (BatchedEnv, rl.CpuEnvPool):
        with pytest.raises(ValueError, match="max_steps"):
            construct(task, 2, track_visited=True, max_steps=max_steps)


def test_label_step_updates_membership():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    opts = dict(track_visited=64, autoreset=True, max_steps=7, seed=12)
    indexed = BatchedEnv(task, 8, **opts)
    labelled = BatchedEnv(task, 8, actions="label", **opts)
    for step in range(25):
        exp = rl.expand(task, indexed.states, framework="torch")
        action = indexed.random_actions()
        take = exp.offsets[:-1].to(torch.int64) + action
        schema, binding = exp.schema[take].clone(), exp.binding[take].clone()
        if step % 5 == 0:
            schema.fill_(-1)
            action.fill_(-1)
        a, b = indexed.step(action), labelled.step((schema, binding))
        assert torch.equal(a.revisited, b.revisited)
        assert torch.equal(indexed.visited, labelled.visited)
        assert torch.equal(indexed.visited_count, labelled.visited_count)
        assert torch.equal(a.states, b.states)


def test_absorbed_goal_and_disabled_tracking():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    env = BatchedEnv(task, 2, goals=True, track_visited=8, autoreset=False)
    env.goal_pos.zero_()
    env.goal_neg.zero_()
    env.refresh()
    before = env.visited.clone()
    assert not env.step().revisited.any()
    assert torch.equal(env.visited, before)
    off = BatchedEnv(task, 2)
    assert off.visited is off.visited_count is off.visited_status is None
    assert not off.step().revisited.any()
    assert not np.asarray(rl.CpuEnvPool(task, 2).step().revisited).any()


def test_lookup_checks_indices_and_capacity():
    table = np.zeros((2, 8), np.uint64)
    count = np.zeros(2, np.int32)
    hashes = np.array([123], np.uint64)
    for parent, rows in [(np.array([2], np.int32), None), (np.array([0], np.int32), np.array([2], np.int32))]:
        with pytest.raises(ValueError, match="outside"):
            rl.visited_mask(table, count, hashes, parent, rows=rows)
    with pytest.raises(ValueError, match="capacity"):
        rl.visited_mask(table[:, :3].copy(), count, hashes, np.array([0], np.int32))

@pytest.mark.needs_gil
def test_torch_compile_keeps_visit_mutations():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("gripper__prob05")
    opts = dict(track_visited=32, max_steps=5, seed=3)
    eager, compiled = BatchedEnv(task, 8, **opts), BatchedEnv(task, 8, **opts)

    def loop(env):
        env.reset()
        first = env.step()
        second = env.step(torch.full((8,), -1, dtype=torch.int64))
        env.clear_visited()
        return first.revisited, second.revisited, env.visited_count.clone()

    run = torch.compile(loop, fullgraph=True, backend="aot_eager")
    for _ in range(2):
        a, b = loop(eager), run(compiled)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
        assert torch.equal(eager.visited, compiled.visited)
        assert torch.equal(eager.states, compiled.states)
    hashes = torch.from_numpy(state_hashes([task.initial_state] * 8, [task] * 8).view(np.int64))
    parents = torch.arange(8, dtype=torch.int32)
    mask = torch.compile(rl.visited_mask, fullgraph=True, backend="aot_eager")
    assert torch.equal(mask(eager.visited, eager.visited_count, hashes, parents),
                       rl.visited_mask(eager.visited, eager.visited_count, hashes, parents))


def test_pool_subsets_and_collected_batches():
    task = task_of("gripper__prob05")
    pool = rl.CpuEnvPool(task, 300, track_visited=8, threads=4, dead_end="none", sampler=None)
    pool.send(np.arange(0, 150, dtype=np.int32), actions=[-1] * 150)
    pool.send(np.arange(150, 300, dtype=np.int32), actions=[-1] * 150)
    batch = pool.recv(min_rows=300)
    assert np.asarray(batch.revisited).all()
    pool.send(env_ids=np.arange(1, 300, dtype=np.int32), actions=[-1] * 299)
    pool.clear_visited(env_ids=[0])
    assert np.asarray(pool.recv().revisited).all()
    pool.clear_visited(env_ids=[0, 149, 299])
    small = pool.step(env_ids=[299, 0, 149], actions=[-1] * 3)
    assert np.asarray(small.revisited).all()
    assert np.asarray(batch.revisited).all()


def test_autoreset_changes_domain_and_inserts_new_initial_hash():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    tasks = [task_of("gripper__prob05"), task_of("numeric")]
    table = rl.TaskSuite.group(tasks)
    cfg = dict(track_visited=32, max_steps=1, sampler=None, seed=2)
    env, pool = BatchedEnv(table, 2, **cfg), rl.CpuEnvPool(table, 2, **cfg)
    ids = np.array([0, 1], np.int32)
    env.reset(task_ids=torch.from_numpy(ids))
    pool.reset(task_ids=ids)
    for _ in range(4):
        ids = ids[::-1].copy()
        a, b = env.step(next_task_ids=torch.from_numpy(ids)), pool.step(next_task_ids=ids)
        np.testing.assert_array_equal(a.revisited, b.revisited)
        np.testing.assert_array_equal(a.states.numpy().view(np.uint64), b.states)
        assert torch.all(env.visited_count == 1)
        states = decoded(table, b)
        assert states == [tasks[int(i)].initial_state for i in ids]
        hashes = state_hashes(states, [tasks[int(i)] for i in ids])
        assert rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), hashes, np.arange(2, dtype=np.int32)).all()


def test_numeric_successor_hashes_find_visited_states():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    task = task_of("numeric")
    assert task.numeric_words
    env = BatchedEnv(task, 4, track_visited=64, autoreset=False, dead_end="none", seed=5, sampler=None)
    history = [{s} for s in task.decode(env.states.numpy().view(np.uint64))]
    for _ in range(6):
        env.step()
        for i, state in enumerate(task.decode(env.states.numpy().view(np.uint64))):
            history[i].add(state)
    exp = rl.expand(task, env.states, framework="numpy", hashes=True)
    parent = np.asarray(exp.parent)
    expected = [s in history[int(p)] for s, p in zip(task.decode(np.asarray(exp.succ)), parent)]
    assert any(expected)
    got = rl.visited_mask(env.visited.numpy(), env.visited_count.numpy(), np.asarray(exp.succ_hash), parent)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(exp.succ_hash, mymyr.hash_rows(np.asarray(exp.succ), numeric_words=exp.numeric_words))
    env.close()
