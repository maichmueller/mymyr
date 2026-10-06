"""Multi-instance batches over a mymyr.rl.TaskTable through the Python layer.

Instance sets of the fork's data (as tests/cpp/rl/table_instance_sets.hpp; each spans at least 3x in object count and two row-width
buckets): blocks, gripper, miconic, logistics and miconic-simpleadl (conditional effects, goal axioms). Checked:
- the table: domain predicate ids a function of the domain (two tables over other instances agree), problem-local
  predicates apart, the atom metadata and object offsets round-trip by name;
- rl.expand over a mixed batch (flat and padded, host, thread pool, device) equals the single-instance expansion of each
  row;
- the environments over a table with given actions equal the single-instance environments row by row (states, labels,
  rewards, termination, truncation, final states, autoreset);
- the random policy over a table is the same on the host env, the device env (fast and general, both bucket launches,
  batch splits), the JAX env (jit, scan, vmap; CPU and GPU) and the CPU pool (1 and 4 threads, split sends, several
  Python threads), with per-env goals and next_task_ids at autoreset.

GPU cases run only with a visible GPU (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_table.py
"""

import os
import pathlib
import threading

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, walk

FORK = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GEN = ROOT / "tests/data/table_instances"
BW = FORK / "ipc/blocksworld-ipc/train"
MI = FORK / "ipc/miconic-ipc/test"
SETS = {  # as tests/cpp/rl/table_instance_sets.hpp
    "blocks": (BW / "domain.pddl", [BW / f"p{k}.pddl" for k in (13, 25, 37, 49, 61)]),
    "gripper": (FORK / "gripper/domain.pddl",
                [FORK / "gripper/test_problem4.pddl"] + [GEN / f"gripper/gripper-{k}.pddl" for k in (10, 20, 40, 80)]),
    "miconic": (MI / "domain.pddl",
                [MI / p for p in ("p01-easy.pddl", "p20-easy.pddl", "p01-medium.pddl", "p04-medium.pddl", "p16-medium.pddl")]),
    "logistics": (FORK / "logistics/domain.pddl",
                  [FORK / "logistics/test_problem.pddl"]
                  + [GEN / f"logistics/logistics-{k}.pddl" for k in ("c2-s3-p4-a1", "c3-s3-p8-a2", "c4-s4-p16-a3", "c6-s5-p30-a4")]),
    "miconic-simpleadl": (FORK / "miconic-simpleadl/domain.pddl",
                          [FORK / "miconic-simpleadl/test_problem.pddl"]
                          + [GEN / f"miconic-simpleadl/simple-{k}.pddl"
                             for k in ("f6-p4-q", "f12-p8-c", "f24-p16-q", "f40-p50-q", "f60-p100-c")]),
}
MASK_SETS = ["blocks", "gripper", "miconic", "logistics"]  # goals without derived literals


@pytest.fixture(scope="module")
def sets():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    out = {}
    for name, (dom, probs) in SETS.items():
        if not dom.is_file() or not all(p.is_file() for p in probs):
            pytest.skip(f"the table instances of {name} are missing (set MYMYR_FORK_DATA)")
        out[name] = [mymyr.Task.from_pddl(dom, p, atoms="frozen") for p in probs]
    return out


def table_of(sets, name):
    return rl.TaskTable(sets[name])


def padded_rows(words, rows):
    out = np.zeros((rows.shape[0], words), np.uint64)
    out[:, : rows.shape[1]] = rows
    return out


def mixed_batch(table, tasks, per=5, seed=0):
    """Rows of random walks of every instance (table width), shuffled, with their task ids."""
    rows, ids = [], []
    for k, t in enumerate(tasks):
        st = walk(t, steps=per, seed=seed + k, walks=1)
        rows.append(padded_rows(table.words, np.asarray(t.encode(st))))
        ids += [k] * len(st)
    perm = np.random.default_rng(seed).permutation(len(ids))
    return np.concatenate(rows)[perm], np.asarray(ids, np.int32)[perm]


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


# ------------------------------------------------------------------------------------------------ the table


@pytest.mark.parametrize("name", list(SETS))
def test_table_metadata_round_trips_by_name(sets, name):
    tasks = sets[name]
    t = rl.TaskTable(tasks)
    assert len(t) == len(tasks) and t.words == max(x.max_words for x in tasks)
    assert t.num_objects == [x.num_objects for x in tasks] and max(t.num_objects) >= 3 * min(t.num_objects)
    assert len({(w > 2) + (w > 4) + (w > 16) for w in t.instance_words}) >= 2  # two row-width buckets
    m = t.atom_metadata()
    names = t.predicate_names
    off = m["atom_offsets"]
    for i, task in enumerate(tasks):
        objects = task.objects
        for j in range(task.num_atoms):
            atom = task.atom(j)
            row = off[i] + j
            assert names[m["atom_pred"][row]] == atom.predicate
            args = [a for a in m["atom_args"][row] if a >= 0]
            assert [objects[a] for a in args] == atom.objects
    assert m["object_offsets"].tolist() == [0] + np.cumsum(t.num_objects).tolist()
    ids = np.array([3, 0, 2, 2, 1], np.int32)
    assert rl.object_offsets(t, ids).tolist() == [0] + np.cumsum(np.asarray(t.num_objects)[ids]).tolist()
    # domain predicates: the same ids in a table over other instances, in another order
    other = rl.TaskTable([tasks[-1], tasks[1]])
    d = t.num_domain_predicates
    assert other.num_domain_predicates == d and other.predicate_names[:d] == names[:d]
    assert all(p == -1 for p in t.predicate_instances[:d])
    local = [(n, i) for n, i in zip(names[d:], t.predicate_instances[d:])]
    if name == "miconic-simpleadl":
        assert local and all(i >= 0 for _, i in local)  # goal axioms: problem-local, one id per instance
    else:
        assert not local
    assert repr(t).startswith("TaskTable(") and t.fingerprint == rl.TaskTable(tasks).fingerprint


def test_table_errors(sets):
    with pytest.raises(ValueError, match="domain"):
        rl.TaskTable([sets["blocks"][0], sets["gripper"][0]])
    with pytest.raises(ValueError):
        rl.TaskTable([])
    t = table_of(sets, "gripper")
    states, ids = mixed_batch(t, sets["gripper"], per=2)
    with pytest.raises(ValueError, match="needs task ids"):
        rl.expand(t, states)
    bad = ids.copy()
    bad[1] = 7
    with pytest.raises(ValueError, match="outside the table"):
        rl.expand(t, states, bad)
    with pytest.raises(ValueError, match="outside the table"):
        rl.is_goal(t, states, bad)


# ------------------------------------------------------------------------------------------------ expand


def check_rows(table, tasks, states, ids, x, K=None):
    off = np.asarray(x.offsets)
    for i in range(len(ids)):
        k = int(ids[i])
        xi = rl.expand(tasks[k], states[i : i + 1], goal=True)
        a, b = off[i], off[i + 1]
        assert b - a == xi.total, i
        assert np.array_equal(np.asarray(x.succ)[a:b], padded_rows(table.words, xi.succ)), i
        assert np.array_equal(np.asarray(x.schema)[a:b], xi.schema), i
        L = xi.label_width
        bind = np.asarray(x.binding)[a:b]
        assert np.array_equal(bind[:, :L], xi.binding) and (bind[:, L:] == -1).all(), i
        assert np.array_equal(np.asarray(x.goal)[a:b], xi.goal), i
        assert (np.asarray(x.parent)[a:b] == i).all()
    if K is not None:
        p = x.padded
        cnt = off[1:] - off[:-1]
        assert np.array_equal(np.asarray(p.count), cnt)
        for i in range(len(ids)):
            c = min(cnt[i], K)
            assert np.array_equal(np.asarray(p.succ)[i, :c], np.asarray(x.succ)[off[i] : off[i] + c])
            assert np.array_equal(np.asarray(p.schema)[i, :c], np.asarray(x.schema)[off[i] : off[i] + c])
            assert not np.asarray(p.mask)[i, c:].any()


@pytest.mark.parametrize("name", list(SETS))
def test_mixed_expand_equals_single(sets, name):
    tasks = sets[name]
    t = rl.TaskTable(tasks)
    states, ids = mixed_batch(t, tasks)
    x = rl.expand(t, states, ids, goal=True, K=8)
    check_rows(t, tasks, states, ids, x, K=8)
    assert x.num_objects == max(t.num_objects) and x.num_schemas == t.num_schemas
    # the actions are named by their rows' instances
    j = int(np.argmax(np.asarray(x.parent) == 0))
    k = int(ids[0])
    assert x.action(j) == rl.expand(tasks[k], states[:1]).action(0)
    # a thread pool, int64 ids and sequences of ids give the same arrays
    y = rl.expand(t, states, ids.astype(np.int64), goal=True, pool=rl.ThreadPool(3))
    z = rl.expand(t, states, ids.tolist(), goal=True)
    for f in ("succ", "parent", "schema", "binding", "goal", "offsets"):
        assert np.array_equal(getattr(x, f), getattr(y, f)) and np.array_equal(getattr(x, f), getattr(z, f)), f
    # batch splits
    h = len(ids) // 2
    a, b = rl.expand(t, states[:h], ids[:h], goal=True), rl.expand(t, states[h:], ids[h:], goal=True)
    assert np.array_equal(np.concatenate([a.succ, b.succ]), x.succ)
    assert np.array_equal(np.concatenate([a.schema, b.schema]), x.schema)
    # is_goal per instance
    flags = rl.is_goal(t, states, ids)
    for i in range(len(ids)):
        assert flags[i] == rl.is_goal(tasks[int(ids[i])], states[i : i + 1])[0]
    # expand_into over the table
    M = x.total
    succ = np.zeros((M, t.words), np.uint64)
    schema = np.zeros(M, np.int32)
    offsets = np.zeros(len(ids) + 1, np.int32)
    r = rl.expand_into(t, states, ids, succ=succ, schema=schema, offsets=offsets)
    assert r["total"] == M and np.array_equal(succ, x.succ) and np.array_equal(offsets, x.offsets)


@pytest.mark.parametrize("name", list(SETS))
def test_device_mixed_expand_equals_host(sets, name):
    dev = gpu_device()
    import torch

    tasks = sets[name]
    t = rl.TaskTable(tasks)
    states, ids = mixed_batch(t, tasks, per=6, seed=3)
    host = rl.expand(t, states, ids, goal=True, K=16)
    s = torch.from_numpy(states.view(np.int64)).to(dev)
    tid = torch.from_numpy(ids).to(dev)
    d = rl.expand(t, s, tid, goal=True, K=16)
    for f in ("succ", "parent", "schema", "binding", "goal", "offsets"):
        got = getattr(d, f).cpu().numpy()
        want = np.asarray(getattr(host, f))
        if f == "succ":
            got = got.view(np.uint64)
        assert np.array_equal(got, want), f
    for f in ("succ", "schema", "mask", "count"):
        got = getattr(d.padded, f).cpu().numpy()
        want = np.asarray(getattr(host.padded, f))
        assert np.array_equal(got.view(want.dtype) if f == "succ" else got, want), f
    j = int(np.argmax(np.asarray(host.parent) == 1))
    assert d.action(j) == host.action(j) and d.num_objects == host.num_objects
    with pytest.raises(TypeError, match="int32"):
        rl.expand(t, s, tid.to(torch.int64))
    bad = tid.clone()
    bad[0] = 99
    with pytest.raises(ValueError, match="outside the table"):
        rl.expand(t, s, bad)


# ------------------------------------------------------------------------------------------------ environments


def env_arrays(env):
    """The comparable state of a torch BatchedEnv as NumPy arrays."""
    return {
        "states": env.states.cpu().numpy().view(np.uint64).copy(),
        "task_ids": env.task_ids.cpu().numpy().copy(),
        "steps": env.steps.cpu().numpy().copy(),
        "count": env.count.cpu().numpy().copy(),
    }


def step_arrays(r):
    out = {f: getattr(r, f).cpu().numpy().copy() for f in r._fields}
    out["final_states"] = out["final_states"].view(np.uint64)
    return out


def given_actions(count, t, offset=0):
    """Deterministic actions from the counts (index count = invalid now and then)."""
    n = count.shape[0]
    return ((np.arange(n) + offset) * 7 + t * 3) % (count.astype(np.int64) + 1)


@pytest.mark.parametrize("name", list(SETS))
def test_mixed_env_equals_single_envs(sets, name):
    """A BatchedEnv over the table with given actions == one BatchedEnv per instance over the same rows."""
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    tasks = sets[name]
    t = rl.TaskTable(tasks)
    n = 40
    ids = np.array([(i * 3) % len(tasks) for i in range(n)], np.int32)
    cfg = dict(max_steps=5, goal_reward=2.0, dead_end_reward=-3.0, seed=1)
    mixed = rt.BatchedEnv(t, n, task_ids=ids, **cfg)
    singles = [rt.BatchedEnv(task, int((ids == k).sum()), **cfg) for k, task in enumerate(tasks)]
    rows = [np.flatnonzero(ids == k) for k in range(len(tasks))]
    for step in range(12):
        count = mixed.count.numpy()
        act = given_actions(count, step)
        r = step_arrays(mixed.step(torch.from_numpy(act), final=True))
        for k, env in enumerate(singles):
            if not len(rows[k]):
                continue
            q = step_arrays(env.step(torch.from_numpy(act[rows[k]]), final=True))
            W = env.words
            for f in ("reward", "terminated", "truncated", "goal", "invalid", "schema", "count"):
                assert np.array_equal(r[f][rows[k]], q[f]), (step, k, f)
            L = env.label_width
            assert np.array_equal(r["binding"][rows[k]][:, :L], q["binding"][:, :L]), (step, k)
            assert np.array_equal(r["final_states"][rows[k]][:, :W], q["final_states"]), (step, k)
            assert not r["final_states"][rows[k]][:, W:].any()
            assert np.array_equal(mixed.states.numpy()[rows[k]][:, :W], env.states.numpy()), (step, k)
            assert np.array_equal(mixed.task_ids.numpy()[rows[k]], np.full(len(rows[k]), k)), (step, k)
            assert np.array_equal(mixed.steps.numpy()[rows[k]], env.steps.numpy()), (step, k)


def host_rollout(t, ids, steps, goals=False, next_ids=None, seed=5, **cfg):
    """The reference: a host BatchedEnv over the table with the random policy; returns per-step arrays."""
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    n = len(ids)
    env = rt.BatchedEnv(t, n, task_ids=ids, goals=goals, seed=seed, **cfg)
    out = [env_arrays(env)]
    for s in range(steps):
        nxt = None if next_ids is None else torch.from_numpy(next_ids(s))
        r = step_arrays(env.step(next_task_ids=nxt, final=True))
        a = env_arrays(env)
        a.update(r)
        if goals:
            a["goal_pos"] = env.goal_pos.numpy().view(np.uint64).copy()
        out.append(a)
    return out


ROT = lambda n, k: (lambda s: ((np.arange(n) + s) % k).astype(np.int32))  # noqa: E731


def rollout_config(name):
    return dict(max_steps=6, goal_reward=1.0, dead_end="no_successors", dead_end_reward=-2.0)


@pytest.mark.parametrize("name", list(SETS))
def test_device_env_equals_host_env(sets, name):
    dev = gpu_device()
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    tasks = sets[name]
    t = rl.TaskTable(tasks)
    n, steps = 96, 14
    ids = (np.arange(n) * 5 % len(tasks)).astype(np.int32)
    goals = name in MASK_SETS
    ref = host_rollout(t, ids, steps, goals=goals, next_ids=ROT(n, len(tasks)), **rollout_config(name))
    fast_ok = rt.fast_unsupported(t) == ""
    paths = ["general"] + (["auto"] if fast_ok else [])
    for path in paths:
        for launch in ("widest", "per_bucket"):
            env = rt.BatchedEnv(t, n, task_ids=ids, goals=goals, seed=5, device=dev, path=path, **rollout_config(name))
            env.set_launch(launch)
            assert env.core.fast == (path == "auto")
            for s in range(steps):
                r = step_arrays(env.step(next_task_ids=torch.from_numpy(ROT(n, len(tasks))(s)).to(dev), final=True))
                a = env_arrays(env)
                for f, v in ref[s + 1].items():
                    got = r.get(f, a.get(f))
                    if f == "goal_pos":
                        got = env.goal_pos.cpu().numpy().view(np.uint64)
                    assert np.array_equal(got, v), (path, launch, s, f)
            env.check_errors()
    # a batch split (two envs, first_env = the second half's first env id) on a side stream
    h = n // 3
    side = torch.cuda.Stream(dev)
    with torch.cuda.stream(side):
        parts = [rt.BatchedEnv(t, b - a, task_ids=ids[a:b], goals=goals, seed=5, device=dev, first_env=a,
                               **rollout_config(name)) for a, b in ((0, h), (h, n))]
        for s in range(steps):
            nx = ROT(n, len(tasks))(s)
            for (a, b), env in zip(((0, h), (h, n)), parts):
                env.step(next_task_ids=torch.from_numpy(nx[a:b]).to(dev), final=True)
    side.synchronize()
    got = np.concatenate([env.states.cpu().numpy().view(np.uint64) for env in parts])
    assert np.array_equal(got, ref[-1]["states"])
    assert np.array_equal(np.concatenate([env.task_ids.cpu().numpy() for env in parts]), ref[-1]["task_ids"])


# ------------------------------------------------------------------------------------------------ JAX


def jax_devices():
    jax = pytest.importorskip("jax")
    out = [jax.devices("cpu")[0]]
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        try:
            import mymyr.cuda as mc

            if mc.available():
                out.append(gpus[0])
        except ImportError:
            pass
    return out


@pytest.mark.parametrize("name", ["gripper", "logistics", "miconic-simpleadl"])
def test_jax_env_equals_host_env(sets, name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from mymyr.rl import jax as rj

    tasks = sets[name]
    t = rl.TaskTable(tasks)
    n, steps = 48, 10
    ids = (np.arange(n) * 7 % len(tasks)).astype(np.int32)
    goals = name in MASK_SETS
    rot = ROT(n, len(tasks))
    ref = host_rollout(t, ids, steps, goals=goals, next_ids=rot, **rollout_config(name))
    nxt = jnp.stack([jnp.asarray(rot(s)) for s in range(steps)])
    for device in jax_devices():
        env = rj.Env(t, device=device, goals=goals, **rollout_config(name))
        state, obs = env.reset(None, jnp.asarray(ids))
        assert np.array_equal(rj.as_words(state.states), ref[0]["states"])
        assert np.array_equal(np.asarray(obs.task_id), ids)

        @jax.jit
        def roll(state):
            def body(st, nx):
                st, obs, reward, term, trunc, info = env.step(5, st, next_task_ids=nx)
                return st, (st.states, st.task_id, reward, term, trunc, info["final_state"], info["schema"], info["binding"])

            return jax.lax.scan(body, state, nxt)

        final, (st, tid, reward, term, trunc, fin, schema, binding) = roll(state)
        for s in range(steps):
            r = ref[s + 1]
            assert np.array_equal(rj.as_words(st[s]), r["states"]), (device, s)
            assert np.array_equal(np.asarray(tid[s]), r["task_ids"]), (device, s)
            assert np.array_equal(np.asarray(reward[s]), r["reward"]), (device, s)
            assert np.array_equal(np.asarray(term[s]), r["terminated"]), (device, s)
            assert np.array_equal(np.asarray(trunc[s]), r["truncated"]), (device, s)
            assert np.array_equal(rj.as_words(fin[s]), r["final_states"]), (device, s)
            assert np.array_equal(np.asarray(schema[s]), r["schema"]), (device, s)
            assert np.array_equal(np.asarray(binding[s]), r["binding"]), (device, s)
        if goals:
            assert np.array_equal(rj.as_words(final.goal_pos), ref[-1]["goal_pos"])
        # vmap: two halves as a batch dimension give the same first step
        half = jax.tree.map(lambda x: x.reshape((2, n // 2) + x.shape[1:]), state)
        s1 = jax.vmap(lambda st, nx: env.step(5, st, next_task_ids=nx)[0])(half, nxt[0].reshape(2, n // 2))
        assert np.array_equal(rj.as_words(s1.states).reshape(n, -1), ref[1]["states"]), device
        env.check_errors()


# ------------------------------------------------------------------------------------------------ the CPU pool


def pool_rollout(t, ids, steps, threads, split, goals, next_ids, seed=5, **cfg):
    n = len(ids)
    pool = rl.CpuEnvPool(t, n, threads=threads, goals=goals, seed=seed, **cfg)
    r0 = pool.reset(task_ids=ids)
    out = [{"states": np.array(r0.states), "task_ids": np.array(r0.task_ids), "count": np.array(r0.count)}]
    for s in range(steps):
        nx = next_ids(s)
        if split:
            cut = [0, n // 3, n // 2, n]
            for a, b in zip(cut, cut[1:]):
                pool.send(np.arange(a, b, dtype=np.int32), next_task_ids=nx[a:b])
            b = pool.recv(n)
        else:
            b = pool.step(next_task_ids=nx)
        assert np.array_equal(b.env_ids, np.arange(n))
        rec = {f: np.array(getattr(b, f)) for f in ("states", "task_ids", "count", "steps", "reward", "terminated",
                                                     "truncated", "goal", "invalid", "final_states", "schema", "binding")}
        if goals:
            rec["goal_pos"] = np.array(b.goal_pos)
        out.append(rec)
    assert pool.pending == 0
    return out


@pytest.mark.parametrize("name", list(SETS))
def test_pool_equals_host_env(sets, name):
    tasks = sets[name]
    t = rl.TaskTable(tasks)
    n, steps = 40, 10
    ids = (np.arange(n) * 3 % len(tasks)).astype(np.int32)
    goals = name in MASK_SETS
    rot = ROT(n, len(tasks))
    ref = host_rollout(t, ids, steps, goals=goals, next_ids=rot, **rollout_config(name))
    for threads, split in ((1, False), (4, True), (3, False)):
        got = pool_rollout(t, ids, steps, threads, split, goals, rot, **rollout_config(name))
        for s in range(steps + 1):
            for f, v in got[s].items():
                want = ref[s][f]
                assert np.array_equal(v, want), (threads, split, s, f)


def test_pool_from_python_threads(sets):
    """Several Python threads step disjoint envs of one pool at once (free-threaded Python: no lock of our own); each
    env's trajectory equals a sequential run."""
    tasks = sets["logistics"]
    t = rl.TaskTable(tasks)
    n, steps, T = 64, 12, 4
    ids = (np.arange(n) % len(tasks)).astype(np.int32)
    seq = pool_rollout(t, ids, steps, 2, False, False, lambda s: None, max_steps=4)
    pool = rl.CpuEnvPool(t, n, threads=4, seed=5, max_steps=4)
    pool.reset(task_ids=ids)
    per = n // T
    results = [None] * T
    errors = []

    def worker(w):
        try:
            envs = np.arange(w * per, (w + 1) * per, dtype=np.int32)
            last = None
            for _ in range(steps):
                last = pool.recv_ticket(pool.send(envs))
            results[w] = np.array(last.states)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(T)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    assert np.array_equal(np.concatenate(results), seq[-1]["states"])


def test_pool_batches_outlive_steps_and_pool(sets):
    """A received batch's arrays go back to the pool only once Python drops the batch and every array of it: kept
    arrays hold their values through later steps (which reuse dropped batches' arrays) and after the pool is gone."""
    t = table_of(sets, "blocks")
    n = 48
    pool = rl.CpuEnvPool(t, n, threads=3, seed=2, max_steps=5)
    pool.reset(task_ids=(np.arange(n) % len(t)).astype(np.int32))
    kept = pool.step()
    states, reward = kept.states, kept.reward
    want = (np.array(states), np.array(reward))
    del kept
    for _ in range(6):
        pool.step()  # dropped at once: their arrays are recycled
    assert np.array_equal(states, want[0]) and np.array_equal(reward, want[1])
    last = pool.step()
    del pool
    assert np.array_equal(states, want[0]) and last.rows == n and np.array(last.states).shape[0] == n
    del last


def test_pool_errors(sets):
    t = table_of(sets, "gripper")
    pool = rl.CpuEnvPool(t, 8, threads=2)
    with pytest.raises(ValueError):
        pool.send([1, 1])  # a repeated env id
    tk = pool.send([0, 1])
    with pytest.raises(ValueError):
        pool.send([1, 2])  # env 1 has a step in flight
    b = pool.recv_ticket(tk)
    assert b.rows == 2 and b.stepped and b.reward.dtype == np.float32
    with pytest.raises(ValueError):
        pool.recv()  # nothing pending
    with pytest.raises(ValueError, match="outside the table"):
        pool.reset(task_ids=np.full(8, 9, np.int32))
    r = pool.reset([2, 3], [1, 2])
    assert not r.stepped and r.reward is None and r.task_ids.tolist() == [1, 2]


# ------------------------------------------------------------------------------------------------ TorchRL over tables


def planning_table_env(table, ids, seed=0):
    """A PlanningEnv over a table for ParallelEnv workers (importable by reference; the table pickles)."""
    import mymyr.rl.torch as rt

    return rt.PlanningEnv(table, len(ids), task_ids=ids, seed=seed, max_steps=5, goal_reward=1.0, max_actions=4)


def last_successor(td):
    k = td["action_mask"].shape[-1]
    return td.set("action", (td["count"].clamp(max=k) - 1).clamp(min=0))


def test_torchrl_over_tables(sets):
    """TorchRL over a mixed table: the specs, resets that carry task_id (curriculum), SerialEnv == ParallelEnv (the
    table pickled into spawned workers), a collector, and rollouts equal to a BatchedEnv with the same actions."""
    pytest.importorskip("torchrl")
    import functools
    import warnings

    import torch
    import mymyr.rl.torch as rt
    from tensordict import TensorDict
    from torchrl.envs import ParallelEnv, SerialEnv
    from torchrl.envs.utils import check_env_specs

    t = table_of(sets, "gripper")
    n, I = 10, len(sets["gripper"])
    ids = torch.as_tensor(np.arange(n) % I, dtype=torch.int32)
    init = torch.as_tensor(np.array(t.initial_states()).view(np.int64))

    env = planning_table_env(t, ids)
    check_env_specs(env, seed=3)
    td = env.reset()
    assert torch.equal(td["task_id"], ids.to(torch.int64)) and torch.equal(td["state"], init[ids.long()])
    # a partial reset with task_id moves the reset rows to other instances; the others keep theirs
    mask = torch.arange(n) % 3 == 0
    new = (ids + 1) % I
    td = env.reset(TensorDict({"_reset": mask.unsqueeze(-1), "task_id": new.to(torch.int64)}, batch_size=[n]))
    want = torch.where(mask, new, ids).to(torch.int64)
    # (TorchRL fills the rows not reset from the input tensordict: the env's own task ids are the truth)
    assert torch.equal(env.batched.task_ids.long(), want)
    assert torch.equal(td["task_id"][mask], want[mask]) and torch.equal(td["state"][mask], init[want[mask]])

    # a rollout with a deterministic policy == BatchedEnv with the same actions (resets of finished rows keep the
    # rows' instances; the rollout's first reset keeps the current ones)
    env.reset(TensorDict({"task_id": ids.to(torch.int64)}, batch_size=[n]))
    env.set_seed(4)
    roll = env.rollout(8, last_successor, break_when_any_done=False)
    twin = rt.BatchedEnv(t, n, task_ids=ids, seed=4, max_steps=5, goal_reward=1.0, autoreset=True)
    for s in range(8):
        a = roll["action"][:, s]
        assert torch.equal(roll["state"][:, s], twin.states) and torch.equal(roll["task_id"][:, s], twin.task_ids.long())
        twin.step(a)
    assert torch.equal(roll["next", "task_id"][:, -1], ids.to(torch.int64))

    make = functools.partial(planning_table_env, t, ids, 0)
    serial = SerialEnv(2, make)
    par = ParallelEnv(2, make, mp_start_method="spawn")
    try:
        serial.set_seed(3)
        a = serial.rollout(7, last_successor, break_when_any_done=False)
        par.set_seed(3)
        b = par.rollout(7, last_successor, break_when_any_done=False)
        assert a.batch_size == (2, n, 7)
        for key in ["action", "state", "task_id", "count", ("next", "reward"), ("next", "done"), ("next", "state")]:
            assert torch.equal(a[key], b[key]), key
    finally:
        par.close()
        serial.close()

    with warnings.catch_warnings():  # torchrl 0.12 deprecates its own weight updaters at import
        warnings.simplefilter("ignore", DeprecationWarning)
        from torchrl.collectors import Collector
    collector = Collector(planning_table_env(t, ids, 1), last_successor, frames_per_batch=5 * n, total_frames=10 * n)
    batches = list(collector)
    collector.shutdown()
    assert len(batches) == 2
    for b in batches:
        assert b.numel() == 5 * n and torch.equal(b["task_id"][:, 0], ids.to(torch.int64))
        assert torch.all(b["next", "reward"] <= 1.0)


def test_per_env_goals_are_mask_tests(sets):
    """goals=True: the step's goal test is the per-env mask test (relabelled goals included)."""
    rt = pytest.importorskip("mymyr.rl.torch")
    tasks = sets["gripper"]
    t = rl.TaskTable(tasks)
    n = 24
    ids = (np.arange(n) % len(tasks)).astype(np.int32)
    env = rt.BatchedEnv(t, n, task_ids=ids, goals=True, seed=2, autoreset=False)
    gpos, gneg = rl.goal_masks(t, ids)
    assert np.array_equal(env.goal_pos.numpy().view(np.uint64), gpos)
    # relabel: the goal of every env becomes an atom of its current state (reached by any step that keeps it)
    import torch

    new_pos = np.zeros_like(gpos)
    new_pos[:, 0] = env.states.numpy().view(np.uint64)[:, 0] & np.uint64(1)
    env.reset(goal_pos=torch.from_numpy(new_pos.view(np.int64)), goal_neg=torch.zeros_like(env.goal_neg))
    for _ in range(4):
        r = env.step(final=True)
        want = rl.goal_test(r.final_states.numpy().view(np.uint64), new_pos, np.zeros_like(new_pos))
        assert np.array_equal(r.goal.numpy(), want)
