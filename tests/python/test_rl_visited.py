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
@pytest.mark.torch_compile
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


# ------------------------------------------------------------------------------------------------ device tables


def gpu_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("mymyr sees no CUDA device")
    return torch.device("cuda", 0)


def ipc_table(domain, problems=("p01",)):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/ipc" / f"{domain}-ipc" / "train"
    return rl.TaskTable([mymyr.Task.from_pddl(base / "domain.pddl", base / f"{p}.pddl", atoms="dense")
                         for p in problems])


def device_table(kind):
    """single: the single-instance kernels; multi: the multi-instance kernels; suite: a suite of two domains on the
    fast path; general: a suite with numeric fluents (the general path)."""
    if kind == "single":
        return rl.TaskTable([task_of("gripper__prob05")])
    if kind == "multi":
        return ipc_table("blocksworld", ("p01", "p02", "p13"))
    if kind == "suite":
        return rl.TaskSuite([ipc_table("rovers"), ipc_table("floortile", ("p01", "p02"))])
    return rl.TaskSuite.group([task_of("gripper__prob05"), task_of("numeric")])


def assert_same(cpu, gpu, *names):
    for name in names:
        left, right = getattr(cpu, name), getattr(gpu, name)
        assert torch_equal(left, right.cpu()), name


def torch_equal(a, b):
    import torch

    return a.shape == b.shape and bool(torch.equal(a, b))


@pytest.mark.parametrize("capacity", [True, 4])
@pytest.mark.parametrize("autoreset", [True, False])
@pytest.mark.parametrize("kind", ["single", "multi", "suite", "general"])
def test_device_rollouts_equal_host(kind, autoreset, capacity):
    """CPU and device BatchedEnvs keep the same tables word for word: steps of the random policy and invalid actions,
    autoresets or masked resets, overflow (H = 4, also after the status is zeroed), clear_visited; device
    visited_mask equals the host's."""
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    device = gpu_device()
    table = device_table(kind)
    n = 24
    cfg = dict(seed=71, autoreset=autoreset, max_steps=9, track_visited=capacity, sampler=None)
    cpu, gpu = BatchedEnv(table, n, **cfg), BatchedEnv(table, n, device=device, **cfg)
    assert gpu.capturable == (kind != "general")
    ids = torch.arange(n, dtype=torch.int32) % cpu.num_instances
    cpu.reset(task_ids=ids)
    gpu.reset(task_ids=ids.to(device))
    tables = ("states", "visited", "visited_count", "visited_status")
    assert_same(cpu, gpu, *tables)
    revisits = 0
    for step in range(30):
        action = torch.full((n,), -1, dtype=torch.int64) if step % 4 == 3 else None
        a = cpu.step(action)
        b = gpu.step(None if action is None else action.to(device))
        for name in ("states", "reward", "terminated", "truncated", "goal", "invalid", "count", "revisited"):
            assert torch_equal(getattr(a, name), getattr(b, name).cpu()), (step, name)
        assert_same(cpu, gpu, *tables)
        revisits += int(a.revisited.sum())
        if not autoreset and step % 2 == 1:
            done = a.terminated | a.truncated
            cpu.reset(done)
            gpu.reset(done.to(device))
            assert_same(cpu, gpu, *tables)
        if step == 12:
            mask = torch.arange(n) % 3 == 0
            cpu.clear_visited(mask)
            gpu.clear_visited(mask.to(device))
            assert_same(cpu, gpu, *tables)
        if step in (7, 20):
            # an acknowledged overflow: saturated rows set the bit again on their next step
            cpu.visited_status.zero_()
            gpu.visited_status.zero_()
        if step % 5 == 0:
            exp = rl.expand(table, cpu.states, cpu._ids(), framework="torch", hashes=True)
            want = rl.visited_mask(cpu.visited, cpu.visited_count, exp.succ_hash, exp.parent)
            got = rl.visited_mask(gpu.visited, gpu.visited_count, exp.succ_hash.to(device), exp.parent.to(device))
            assert torch_equal(want, got.cpu())
    assert revisits
    assert int(cpu.visited_status) == (1 if capacity == 4 else 0)
    gpu.check_errors()
    cpu.close()
    gpu.close()


def test_device_label_steps_equal_index_steps():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    device = gpu_device()
    task = task_of("gripper__prob05")
    opts = dict(track_visited=64, autoreset=True, max_steps=7, seed=12, device=device)
    indexed = BatchedEnv(task, 8, **opts)
    labelled = BatchedEnv(task, 8, actions="label", **opts)
    for step in range(25):
        exp = rl.expand(task, indexed.states, framework="torch", stream=torch.cuda.current_stream())
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
    indexed.close()
    labelled.close()


def target_reference(table, batch, states, task_ids, history, env_rows=None):
    """action_visited of a batch from sets of States: per target, whether its entity is an action whose successor (in
    the full canonical expansion of its graph's state) is in the history of the graph's environment."""
    torch = pytest.importorskip("torch")
    columns = batch.as_dict("numpy")
    sizes = batch.sizes
    targets = np.asarray(columns["target_entity"])
    live = len(targets) if sizes is None else int(sizes["targets"])
    role, ref = np.asarray(columns["entity_role"]), np.asarray(columns["entity_ref"])
    offsets = np.maximum.accumulate(np.asarray(columns["entity_offsets"]))
    host_states = states.cpu().numpy().view(np.uint64)
    ids = task_ids.cpu().numpy()
    multi = isinstance(table, rl.TaskSuite) or len(table.tasks) > 1
    full = rl.expand(table, host_states, ids if multi else None)
    successors = rl.decode(table, np.asarray(full.succ), ids[np.asarray(full.parent)]) if multi else \
        rl.decode(table, np.asarray(full.succ))
    want = np.zeros(len(batch.target_entity), bool)  # a device batch's targets at their capacity
    for t in range(live):
        e = int(targets[t])
        if role[e] != 4:
            continue
        g = int(np.searchsorted(offsets, e, side="right") - 1)
        row = g if env_rows is None else int(env_rows[g])
        want[t] = successors[int(full.offsets[row]) + int(ref[e])] in history[row]
    return torch.as_tensor(want)


def noop_ipc_table():
    return ipc_table("floortile")


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("capacity", [False, True])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_action_visited_matches_state_sets(dev, capacity, drop):
    """Observation.action_visited / StepResult.action_visited against sets of States per episode, aligned with the
    targets (goal targets are not actions: False), with and without dropped no-op actions."""
    torch = pytest.importorskip("torch")
    from mymyr import encoding
    from mymyr.rl.torch import BatchedEnv

    device = torch.device("cpu") if dev == "cpu" else gpu_device()
    table = noop_ipc_table()
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action", "goal"))
    caps = dict(entities=1 << 12, args=1 << 14, targets=1 << 12) if capacity else None
    n = 6
    env = BatchedEnv(table, n, device=device, seed=5, max_steps=6, dead_end="none", track_visited=True,
                     observation=encoder, obs_actions=True, obs_drop_noops=drop, obs_capacities=caps)
    observation = env.observation()
    history = [{s} for s in rl.decode(table, observation.states.cpu().numpy().view(np.uint64))]
    got = observation.action_visited
    assert got.dtype == torch.bool and got.device == device
    assert torch.equal(got.cpu(), target_reference(table, observation.encoding, observation.states,
                                                   observation.task_ids, history))
    hits = int(got.sum())
    for _ in range(12):
        step = env.step()
        reached = rl.decode(table, step.final_states.cpu().numpy().view(np.uint64))
        after = rl.decode(table, step.states.cpu().numpy().view(np.uint64))
        done = (step.terminated | step.truncated).cpu().numpy()
        for i in range(n):
            history[i].add(reached[i])
            if done[i]:
                history[i] = {after[i]}
        want = target_reference(table, step.obs, step.states, step.task_ids, history)
        assert torch.equal(step.action_visited.cpu(), want)
        hits += int(want.sum())
    assert hits
    env.close()
    # without targets, visited tables or action rows there is no action_visited
    plain = encoding.FlatRelationEncoder(table)
    for opts in (dict(observation=plain, obs_actions=True, track_visited=8),
                 dict(observation=encoder, track_visited=8), dict(observation=encoder, obs_actions=True)):
        other = BatchedEnv(table, 2, device=device, **opts)
        assert other.observation().action_visited is None and other.step().action_visited is None
        other.close()


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_suite_action_visited_per_domain_batch(dev):
    """Over a suite of two domains action_visited has one mask per domain batch; graph g of domain d's batch is the
    environment row of domain d with index g."""
    torch = pytest.importorskip("torch")
    from mymyr import encoding
    from mymyr.rl.torch import BatchedEnv

    device = torch.device("cpu") if dev == "cpu" else gpu_device()
    suite = rl.TaskSuite([ipc_table("rovers"), ipc_table("floortile")])
    encoder = encoding.SuiteEncoder(suite, target_sources=("action",))
    n = 10
    caps = dict(entities=1 << 13, args=1 << 15, targets=1 << 13) if dev == "cuda" else None
    env = BatchedEnv(suite, n, device=device, seed=4, max_steps=5, dead_end="none", track_visited=True,
                     observation=encoder, obs_actions=True, obs_capacities=caps)
    history = [{s} for s in rl.decode(suite, env.states.cpu().numpy().view(np.uint64), env.task_ids.cpu().numpy())]
    hits = 0
    for _ in range(8):
        before = env.task_ids.cpu().numpy()
        step = env.step()
        reached = rl.decode(suite, step.final_states.cpu().numpy().view(np.uint64), before)
        after = rl.decode(suite, step.states.cpu().numpy().view(np.uint64), step.task_ids.cpu().numpy())
        done = (step.terminated | step.truncated).cpu().numpy()
        for i in range(n):
            history[i].add(reached[i])
            if done[i]:
                history[i] = {after[i]}
        obs = step.obs
        assert len(step.action_visited) == len(obs.batches) == 2
        domain, index = obs.domain.cpu().numpy(), obs.index.cpu().numpy()
        for d, (batch, got) in enumerate(zip(obs.batches, step.action_visited)):
            rows = np.flatnonzero(domain == d)
            env_rows = np.full(n, -1)
            env_rows[index[rows]] = rows
            want = target_reference(suite, batch, step.states, step.task_ids, history, env_rows)
            assert torch.equal(got.cpu(), want)
            hits += int(want.sum())
    assert hits
    env.close()


def test_captured_rollout_replays_tables_and_action_visited():
    """A captured rollout's replays equal eager steps of a twin environment (tables, revisited, action_visited), and
    nothing changes between replays."""
    torch = pytest.importorskip("torch")
    from mymyr import encoding
    from mymyr.rl.torch import BatchedEnv

    device = gpu_device()
    table = noop_ipc_table()
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
    opts = dict(device=device, seed=3, max_steps=5, track_visited=True, observation=encoder, obs_actions=True,
                obs_drop_noops=True, obs_capacities=dict(entities=1 << 12, args=1 << 14, targets=1 << 12))
    eager, graph = BatchedEnv(table, 8, **opts), BatchedEnv(table, 8, **opts)
    rollout = graph.capture(3, final=True)
    names = ("states", "visited", "visited_count", "visited_status")
    hits = 0
    for _ in range(3):
        out = rollout.replay()
        for k in range(3):
            step = eager.step(final=True)
            assert torch.equal(out.revisited[k], step.revisited)
            assert torch.equal(out.action_visited[k], step.action_visited)
            hits += int(step.action_visited.sum())
        for name in names:
            assert torch.equal(getattr(graph, name), getattr(eager, name)), name
        kept = [getattr(graph, name).clone() for name in names] + [x.clone() for x in out.action_visited]
        torch.cuda.synchronize(device)
        now = [getattr(graph, name) for name in names] + list(out.action_visited)
        assert all(torch.equal(a, b) for a, b in zip(kept, now))
    assert hits
    eager.close()
    graph.close()


@pytest.mark.needs_gil
@pytest.mark.torch_compile
def test_compiled_device_steps_keep_tables():
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import BatchedEnv

    device = gpu_device()
    task = task_of("gripper__prob05")
    opts = dict(track_visited=32, max_steps=5, seed=3, device=device)
    eager, compiled = BatchedEnv(task, 8, **opts), BatchedEnv(task, 8, **opts)

    def loop(env):
        env.reset()
        first = env.step()
        second = env.step(torch.full((8,), -1, dtype=torch.int64, device=device))
        env.clear_visited(torch.arange(8, device=device) % 2 == 0)
        third = env.step()
        return first.revisited, second.revisited, third.revisited, env.visited_count.clone()

    run = torch.compile(loop, fullgraph=True, backend="aot_eager")
    for _ in range(2):
        a, b = loop(eager), run(compiled)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
        for name in ("states", "visited", "visited_count", "visited_status"):
            assert torch.equal(getattr(eager, name), getattr(compiled, name)), name
    hashes = torch.from_numpy(state_hashes([task.initial_state] * 8, [task] * 8).view(np.int64)).to(device)
    parents = torch.arange(8, dtype=torch.int32, device=device)
    mask = torch.compile(rl.visited_mask, fullgraph=True, backend="aot_eager")
    want = rl.visited_mask(eager.visited.cpu(), eager.visited_count.cpu(), hashes.cpu(), parents.cpu())
    assert torch.equal(mask(eager.visited, eager.visited_count, hashes, parents).cpu(), want)
    # the ops' fake implementations
    ops = torch.ops.mymyr
    rows = torch.tensor([7, -1, 0, 3], dtype=torch.int32, device=device)
    cases = [
        (ops.visited_mask.default, (eager.visited, eager.visited_count, hashes, parents, None)),
        (ops.visited_mask.default, (eager.visited, eager.visited_count, hashes, parents % 4, rows)),
        (ops.clear_visited.default, (eager.handle, eager.states, None, eager.visited, eager.visited_count,
                                     eager.visited_status, parents % 3 == 0)),
        (ops.step.default, (eager.handle, eager.states, None, eager.steps, eager.draws, eager.counts, eager.views,
                            eager.count, eager.at_goal, None, None, None, None, 0, True, None, None, eager.visited,
                            eager.visited_count, eager.visited_status, torch.zeros(8, dtype=torch.bool, device=device))),
        (ops.reset.default, (eager.handle, eager.states, None, eager.steps, eager.counts, eager.views, None, None,
                             eager.count, eager.at_goal, parents % 2 == 0, False, eager.visited, eager.visited_count,
                             eager.visited_status)),
    ]
    for op, args in cases:
        result = torch.library.opcheck(op, args)
        assert set(result.values()) == {"SUCCESS"}, (op, result)
    eager.close()
    compiled.close()


@pytest.mark.needs_gil
@pytest.mark.torch_compile
def test_compiled_observations_carry_action_visited():
    torch = pytest.importorskip("torch")
    from mymyr import encoding
    from mymyr.rl.torch import BatchedEnv

    device = gpu_device()
    table = noop_ipc_table()
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action", "goal"))
    opts = dict(device=device, seed=8, max_steps=4, track_visited=True, observation=encoder, obs_actions=True,
                obs_drop_noops=True, obs_capacities=dict(entities=1 << 12, args=1 << 14, targets=1 << 12))
    eager, compiled = BatchedEnv(table, 6, **opts), BatchedEnv(table, 6, **opts)
    run = torch.compile(lambda: compiled.step(final=False), fullgraph=True, backend="aot_eager")
    for _ in range(4):
        a, b = eager.step(final=False), run()
        assert torch.equal(a.revisited, b.revisited)
        assert torch.equal(a.action_visited, b.action_visited)
        assert torch.equal(eager.visited, compiled.visited)
    # the op's fake implementation, with and without a row map
    from mymyr.rl.torch._observation import _target_arrays

    rows = a.actions
    arrays = _target_arrays(a.obs, device)
    env_rows = torch.arange(6, dtype=torch.int32, device=device).flip(0)
    for mapping in (None, env_rows):
        args = (*arrays, rows.offsets, rows.index, rows.succ_hash, rows.status, eager.visited, eager.visited_count,
                mapping)
        result = torch.library.opcheck(torch.ops.mymyr.target_visited.default, args)
        assert set(result.values()) == {"SUCCESS"}, result
    eager.close()
    compiled.close()


def test_device_mask_outside_rows_reads_unvisited():
    torch = pytest.importorskip("torch")
    device = gpu_device()
    table = torch.zeros((2, 8), dtype=torch.int64, device=device)
    count = torch.zeros(2, dtype=torch.int32, device=device)
    hashes = torch.tensor([123, 5, 6], dtype=torch.int64, device=device)
    got = rl.visited_mask(table, count, hashes, torch.tensor([2, -1, 0], dtype=torch.int32, device=device))
    assert not got.any()
    got = rl.visited_mask(table, count, hashes, torch.tensor([0, 0, 1], dtype=torch.int32, device=device),
                          rows=torch.tensor([2, -3], dtype=torch.int32, device=device))
    assert not got.any()
