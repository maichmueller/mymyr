"""Cross-domain batches over a mymyr.rl.TaskSuite through the Python layer.

Suites of the table instance sets (test_rl_table.py): {gripper, blocks}, {miconic, blocks, gripper, logistics} and
{miconic-simpleadl, gripper, blocks} (conditional effects and goal axioms: the device's general path). Checked:
- the suite: create and group, global ids, each domain's own predicate ids in the atom metadata, pickling, errors;
- rl.expand / expand_into / is_goal over a suite (interleaved and domain-grouped batches; host, thread pool, device)
  equal each domain's table apart;
- the environments: a BatchedEnv over a suite with given actions equals one BatchedEnv per domain over the same rows;
  the random policy over a suite, with next_task_ids that move envs across domains at autoreset, is the same on the
  host env, the device env (fast and general path, interleaved and grouped rows), a captured rollout, the JAX env
  (lax.scan; CPU and GPU) and the CPU pool;
- TorchRL's check_env_specs over a suite (CPU and CUDA), mctx over a suite;
- IW and state spaces over a suite (mymyr.cuda.multi_iw / batched_iw1, mymyr.datasets.state_spaces) equal the
  per-instance results.

GPU cases run only with a visible GPU (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_task_suite.py
"""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import datasets, rl
from conftest import walk
from test_rl_table import (  # noqa: F401
    FORK,
    MASK_SETS,
    ROT,
    env_arrays,
    given_actions,
    gpu_device,
    host_rollout,
    jax_devices,
    padded_rows,
    pool_rollout,
    rollout_config,
    sets,
    step_arrays,
)

SUITES = {
    "gripper+blocks": ["gripper", "blocks"],
    "four": ["miconic", "blocks", "gripper", "logistics"],
    "adl": ["miconic-simpleadl", "gripper", "blocks"],
}
FAST = ["gripper+blocks", "four"]  # every domain on the device's fast path; mask goals


def suite_of(sets, key):
    return rl.TaskSuite([rl.TaskTable(sets[name]) for name in SUITES[key]])


def suite_batch(suite, per=4, seed=0, grouped=False):
    """Rows of random walks of every instance (suite width) with their global task ids: shuffled (domains
    interleaved) or grouped by domain (instances in global order)."""
    rows, ids = [], []
    for g in range(len(suite)):
        t = suite[g]
        st = walk(t, steps=per, seed=seed + g, walks=1)
        rows.append(padded_rows(suite.words, np.asarray(t.encode(st))))
        ids += [g] * len(st)
    states, ids = np.concatenate(rows), np.asarray(ids, np.int32)
    if not grouped:
        order = np.random.default_rng(seed).permutation(len(ids))
        states, ids = states[order], ids[order]
    return states, ids


def check_against_domains(suite, states, ids, x):
    """The expansion of a suite batch == each domain's rows expanded by its table apart (local ids, its width)."""
    off = np.asarray(x.offsets)
    succ, schema, binding, goal, parent = (np.asarray(getattr(x, f)) for f in ("succ", "schema", "binding", "goal", "parent"))
    dom = np.asarray(suite.domain_of)[ids]
    loc = np.asarray(suite.local_ids, np.int32)[ids]
    for d, t in enumerate(suite.tables):
        sel = np.flatnonzero(dom == d)
        if not len(sel):
            continue
        assert not states[sel, t.words:].any()
        y = rl.expand(t, np.ascontiguousarray(states[sel, : t.words]), loc[sel], goal=True)
        rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])
        assert len(rows) == y.total, d
        assert np.array_equal(off[sel + 1] - off[sel], np.diff(np.asarray(y.offsets))), d
        assert np.array_equal(succ[rows], padded_rows(suite.words, y.succ)), d
        assert np.array_equal(schema[rows], y.schema), d
        L = y.label_width
        assert np.array_equal(binding[rows][:, :L], y.binding) and (binding[rows][:, L:] == -1).all(), d
        assert np.array_equal(goal[rows], y.goal), d
        assert np.array_equal(parent[rows], sel[np.asarray(y.parent)]), d


# ------------------------------------------------------------------------------------------------ the suite


def test_suite_ids_metadata_and_pickling(sets):
    g, b = sets["gripper"], sets["blocks"]
    tg, tb = rl.TaskTable(g), rl.TaskTable(b)
    s = rl.TaskSuite([tg, tb])
    I = len(g) + len(b)
    assert len(s) == I and s.num_domains == 2 and s.tables == [tg, tb]
    assert s.domain_of == [0] * len(g) + [1] * len(b) and s.local_ids == list(range(len(g))) + list(range(len(b)))
    assert all(s[i] is t for i, t in enumerate(g + b)) and s.tasks == g + b
    assert s.global_id(1, 2) == len(g) + 2 and s.domain_names == [g[0].domain_name, b[0].domain_name]
    assert s.words == max(tg.words, tb.words) and s.label_width == max(tg.label_width, tb.label_width)
    assert s.num_schemas == max(tg.num_schemas, tb.num_schemas)
    assert s.schema_offsets == [0, tg.num_schemas, tg.num_schemas + tb.num_schemas]
    assert s.num_objects == tg.num_objects + tb.num_objects and s.max_objects == max(s.num_objects)
    assert s.instance_words == tg.instance_words + tb.instance_words and not s.numeric
    assert "gripper" in repr(s) and len({s.fingerprint, tg.fingerprint, tb.fingerprint}) == 3
    # the atom metadata: each domain's rows carry the predicate ids of the domain's own table
    m = s.atom_metadata()
    assert m["domain"].tolist() == s.domain_of and m["local_id"].tolist() == s.local_ids
    assert int(m["num_domains"]) == 2 and m["schema_offsets"].tolist() == s.schema_offsets
    po = m["pred_offsets"]
    for d, t in enumerate(s.tables):
        mt = t.atom_metadata()
        assert m["pred_arity"][po[d] : po[d + 1]].tolist() == mt["pred_arity"].tolist()
        for k in range(len(t)):
            gid = s.global_id(d, k)
            a, z = m["atom_offsets"][gid], m["atom_offsets"][gid + 1]
            ta, tz = mt["atom_offsets"][k], mt["atom_offsets"][k + 1]
            assert np.array_equal(m["atom_pred"][a:z], mt["atom_pred"][ta:tz])
            assert np.array_equal(m["atom_args"][a:z][:, : mt["atom_args"].shape[1]], mt["atom_args"][ta:tz])
    # goal masks and initial states in global order, each domain's rows padded to the suite's width
    gp, _ = s.goal_masks()
    init = s.initial_states()
    for d, t in enumerate(s.tables):
        tgp, _ = t.goal_masks()
        for k in range(len(t)):
            gid = s.global_id(d, k)
            assert np.array_equal(gp[gid], padded_rows(s.words, tgp[k : k + 1])[0])
            assert np.array_equal(init[gid], padded_rows(s.words, t.initial_states()[k : k + 1])[0])
    # group: tasks in any order, one table per domain in first-seen order; global id i = tasks[i]
    mixed = [b[0], g[1], b[2], g[0], b[1]]
    q = rl.TaskSuite.group(mixed)
    assert q.domain_of == [0, 1, 0, 1, 0] and q.local_ids == [0, 0, 1, 1, 2]
    assert [len(t) for t in q.tables] == [3, 2] and q.tables[0].tasks == [b[0], b[2], b[1]]
    assert all(q[i] is t for i, t in enumerate(mixed))
    for x in (s, q):
        y = pickle.loads(pickle.dumps(x))
        assert (y.domain_of, y.local_ids, y.fingerprint, len(y.tables)) == (x.domain_of, x.local_ids, x.fingerprint, 2)
        states, ids = suite_batch(x, per=2)
        a, c = rl.expand(x, states, ids), rl.expand(y, states, ids)
        assert np.array_equal(a.succ, c.succ) and np.array_equal(a.binding, c.binding)
    # one domain: the suite of its table
    one = rl.TaskSuite.group(g)
    assert one.num_domains == 1 and one.fingerprint == tg.fingerprint


def test_suite_errors(sets):
    tg = rl.TaskTable(sets["gripper"])
    with pytest.raises(ValueError, match="TaskSuite"):
        rl.TaskSuite([])
    with pytest.raises(ValueError, match="of one domain"):
        rl.TaskSuite([tg, rl.TaskTable(sets["gripper"][:2])])
    with pytest.raises(TypeError, match="TaskTables"):
        rl.TaskSuite([sets["gripper"][0]])
    with pytest.raises(TypeError, match="Tasks"):
        rl.TaskSuite.group([tg])
    with pytest.raises(ValueError):
        rl.TaskSuite.group([])
    with pytest.raises(TypeError, match="TaskSuite"):
        rl.expand(42, np.zeros((1, 1), np.uint64))
    s = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(s, per=1)
    with pytest.raises(ValueError, match="suite of 10 instances needs task ids"):
        rl.expand(s, states)
    with pytest.raises(ValueError, match="outside the suite's 10 instances"):
        rl.expand(s, states, np.full(len(ids), 10, np.int32))
    with pytest.raises(ValueError, match="one domain"):
        datasets.generalized_state_space(s)


# ------------------------------------------------------------------------------------------------ expand


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("key", list(SUITES))
def test_suite_expand_equals_domains_apart(sets, key, grouped):
    s = suite_of(sets, key)
    states, ids = suite_batch(s, grouped=grouped)
    x = rl.expand(s, states, ids, goal=True, K=8)
    check_against_domains(s, states, ids, x)
    assert x.table is s and x.num_objects == s.max_objects and x.num_schemas == s.num_schemas
    off = np.asarray(x.offsets)
    p = x.padded
    assert np.array_equal(np.asarray(p.count), np.diff(off))
    for i in range(len(ids)):
        c = min(off[i + 1] - off[i], 8)
        assert np.array_equal(np.asarray(p.succ)[i, :c], np.asarray(x.succ)[off[i] : off[i] + c])
        assert np.array_equal(np.asarray(p.binding)[i, :c], np.asarray(x.binding)[off[i] : off[i] + c])
    # the actions are named by their rows' instances
    for i in range(0, len(ids), 7):
        if off[i + 1] > off[i]:
            assert x.action(int(off[i])) == rl.expand(s[int(ids[i])], states[i : i + 1]).action(0), i
    # a thread pool, int64 ids and sequences of ids give the same arrays; so do batch splits
    y = rl.expand(s, states, ids.astype(np.int64), goal=True, pool=rl.ThreadPool(3))
    z = rl.expand(s, states, ids.tolist(), goal=True)
    for f in ("succ", "parent", "schema", "binding", "goal", "offsets"):
        assert np.array_equal(getattr(x, f), getattr(y, f)) and np.array_equal(getattr(x, f), getattr(z, f)), f
    h = len(ids) // 2
    a, b = rl.expand(s, states[:h], ids[:h]), rl.expand(s, states[h:], ids[h:])
    assert np.array_equal(np.concatenate([a.succ, b.succ]), x.succ)
    assert np.array_equal(np.concatenate([a.binding, b.binding]), x.binding)
    # is_goal per row; expand_into
    flags = rl.is_goal(s, states, ids)
    for i in range(len(ids)):
        assert flags[i] == rl.is_goal(s[int(ids[i])], states[i : i + 1])[0], i
    M = x.total
    succ = np.zeros((M, s.words), np.uint64)
    binding = np.zeros((M, s.label_width), np.int32)
    offsets = np.zeros(len(ids) + 1, np.int32)
    r = rl.expand_into(s, states, ids, succ=succ, binding=binding, offsets=offsets)
    assert r["total"] == M and np.array_equal(succ, x.succ) and np.array_equal(binding, x.binding)
    assert np.array_equal(offsets, x.offsets)
    # per-row helpers: domains, object offsets, per-env goal masks
    dom = rl.task_domains(s, ids)
    assert dom.dtype == np.int32 and dom.tolist() == [s.domain_of[i] for i in ids]
    assert rl.object_offsets(s, ids).tolist() == [0] + np.cumsum([s.num_objects[i] for i in ids]).tolist()
    if key in FAST:
        gp, gn = rl.goal_masks(s, ids)
        assert np.array_equal(gp, s.goal_masks()[0][ids])


def test_device_suite_expand_equals_host(sets):
    dev = gpu_device()
    import torch

    for key in SUITES:
        s = suite_of(sets, key)
        for grouped in (False, True):
            states, ids = suite_batch(s, per=6, seed=3, grouped=grouped)
            host = rl.expand(s, states, ids, goal=True, K=16)
            st = torch.from_numpy(states.view(np.int64)).to(dev)
            tid = torch.from_numpy(ids).to(dev)
            d = rl.expand(s, st, tid, goal=True, K=16)
            for f in ("succ", "parent", "schema", "binding", "goal", "offsets"):
                got = getattr(d, f).cpu().numpy()
                want = np.asarray(getattr(host, f))
                assert np.array_equal(got.view(np.uint64) if f == "succ" else got, want), (key, grouped, f)
            for f in ("succ", "schema", "binding", "mask", "count"):
                got = getattr(d.padded, f).cpu().numpy()
                want = np.asarray(getattr(host.padded, f))
                assert np.array_equal(got.view(want.dtype) if f == "succ" else got, want), (key, grouped, f)
            j = int(np.argmax(np.asarray(host.parent) == 1))
            assert d.action(j) == host.action(j) and d.table is s and d.num_schemas == s.num_schemas
            # destination passing into torch tensors
            M = host.total
            succ = torch.zeros((M, s.words), dtype=torch.int64, device=dev)
            offsets = torch.zeros(len(ids) + 1, dtype=torch.int32, device=dev)
            rl.expand_into(s, st, tid, succ=succ, offsets=offsets)
            assert np.array_equal(succ.cpu().numpy().view(np.uint64), host.succ), (key, grouped)
            assert np.array_equal(offsets.cpu().numpy(), host.offsets), (key, grouped)
        bad = tid.clone()
        bad[0] = len(s)
        with pytest.raises(ValueError, match="outside the suite"):
            rl.expand(s, st, bad)


# ------------------------------------------------------------------------------------------------ environments


@pytest.mark.parametrize("key", list(SUITES))
def test_suite_env_equals_domain_envs(sets, key):
    """A BatchedEnv over the suite with given actions == one BatchedEnv per domain table over the same rows."""
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    s = suite_of(sets, key)
    n = 48
    ids = (np.arange(n) * 5 % len(s)).astype(np.int32)
    dom = np.asarray(s.domain_of)[ids]
    loc = np.asarray(s.local_ids, np.int32)[ids]
    cfg = dict(max_steps=5, goal_reward=2.0, dead_end_reward=-3.0, seed=1)
    for dead_end in ("no_successors", "none"):
        mixed = rt.BatchedEnv(s, n, task_ids=ids, dead_end=dead_end, **cfg)
        rows = [np.flatnonzero(dom == d) for d in range(s.num_domains)]
        apart = [rt.BatchedEnv(t, len(r), task_ids=loc[r], dead_end=dead_end, **cfg) for t, r in zip(s.tables, rows)]
        assert mixed.core.row_words == s.row_words
        for step in range(12):
            act = given_actions(mixed.count.numpy(), step)
            r = step_arrays(mixed.step(torch.from_numpy(act), final=True))
            for d, env in enumerate(apart):
                sel = rows[d]
                q = step_arrays(env.step(torch.from_numpy(act[sel]), final=True))
                W, L = env.words, env.label_width
                for f in ("reward", "terminated", "truncated", "goal", "invalid", "schema", "count"):
                    assert np.array_equal(r[f][sel], q[f]), (dead_end, step, d, f)
                assert np.array_equal(r["binding"][sel][:, :L], q["binding"][:, :L]), (dead_end, step, d)
                assert np.array_equal(r["final_states"][sel][:, :W], q["final_states"]), (dead_end, step, d)
                assert not r["final_states"][sel][:, W:].any() and not mixed.states.numpy()[sel][:, W:].any()
                assert np.array_equal(mixed.states.numpy()[sel][:, :W], env.states.numpy()), (dead_end, step, d)
                glob = [s.global_id(d, int(k)) for k in env.task_ids.numpy()]
                assert np.array_equal(mixed.task_ids.numpy()[sel], glob), (dead_end, step, d)
                assert np.array_equal(mixed.steps.numpy()[sel], env.steps.numpy()), (dead_end, step, d)


def crossed_domains(suite, ref):
    dom = np.asarray(suite.domain_of)
    return sum(int((dom[a["task_ids"]] != dom[b["task_ids"]]).sum()) for a, b in zip(ref, ref[1:]))


def env_orders(suite, n):
    """Env task ids: domains interleaved, and grouped by domain."""
    inter = (np.arange(n) * 5 % len(suite)).astype(np.int32)
    grouped = inter[np.argsort(np.asarray(suite.domain_of)[inter], kind="stable")]
    return {"interleaved": inter, "grouped": grouped}


@pytest.mark.parametrize("key", list(SUITES))
def test_device_suite_env_equals_host(sets, key):
    dev = gpu_device()
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    s = suite_of(sets, key)
    n, steps = 96, 14
    goals = key in FAST
    assert (rt.fast_unsupported(s) == "") == (key in FAST)
    for order, ids in env_orders(s, n).items():
        rot = ROT(n, len(s))
        ref = host_rollout(s, ids, steps, goals=goals, next_ids=rot, **rollout_config(key))
        assert crossed_domains(s, ref) > 0
        for path in ["general"] + (["auto"] if key in FAST else []):
            env = rt.BatchedEnv(s, n, task_ids=ids, goals=goals, seed=5, device=dev, path=path, **rollout_config(key))
            assert env.core.fast == (path == "auto") and env.capturable == (path == "auto")
            for k in range(steps):
                r = step_arrays(env.step(next_task_ids=torch.from_numpy(rot(k)).to(dev), final=True))
                a = env_arrays(env)
                for f, v in ref[k + 1].items():
                    got = env.goal_pos.cpu().numpy().view(np.uint64) if f == "goal_pos" else r.get(f, a.get(f))
                    assert np.array_equal(got, v), (order, path, k, f)
            env.check_errors()
            # a refresh after writing states and task ids: the counts of the rows' instances
            env.task_ids.copy_(torch.from_numpy(np.roll(ids, 3)).to(dev))
            env.states.copy_(torch.from_numpy(s.initial_states()[np.roll(ids, 3)].view(np.int64)).to(dev))
            env.refresh()
            init_count = [int(c) for c in rl.expand(s, s.initial_states(), np.arange(len(s), dtype=np.int32)).counts]
            assert env.count.cpu().numpy().tolist() == [init_count[i] for i in np.roll(ids, 3)], (order, path)


def test_captured_suite_rollout_equals_host(sets):
    dev = gpu_device()
    rt = pytest.importorskip("mymyr.rl.torch")
    import torch

    for key in FAST:
        s = suite_of(sets, key)
        I, n, K, R = len(s), 96, 3, 4
        ids = env_orders(s, n)["interleaved"]
        cfg = rollout_config(key)
        ref = host_rollout(s, ids, K * R, goals=True, next_ids=lambda k: ROT(n, I)(k // K), **cfg)
        env = rt.BatchedEnv(s, n, task_ids=ids, goals=True, seed=5, device=dev, **cfg)
        assert env.capturable
        nxt = torch.zeros(n, dtype=torch.int32, device=dev)
        g = env.capture(K, next_task_ids=nxt, final=True)
        fields = ("reward", "terminated", "truncated", "goal", "invalid", "schema", "binding", "final_states", "count")
        for r in range(R):
            nxt.copy_(torch.from_numpy(ROT(n, I)(r)))
            out = g.replay()
            for k in range(K):
                got = step_arrays(rt.StepResult(*(getattr(out, f)[k] for f in fields)))
                for f, v in got.items():
                    assert np.array_equal(v, ref[r * K + k + 1][f]), (key, r, k, f)
            for f, v in env_arrays(env).items():
                assert np.array_equal(v, ref[(r + 1) * K][f]), (key, r, f)
        env.check_errors()


@pytest.mark.parametrize("key", ["gripper+blocks", "adl"])
def test_jax_suite_env_equals_host(sets, key):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from mymyr.rl import jax as rj

    s = suite_of(sets, key)
    n, steps = 48, 10
    ids = env_orders(s, n)["interleaved"]
    goals = key in FAST
    rot = ROT(n, len(s))
    ref = host_rollout(s, ids, steps, goals=goals, next_ids=rot, **rollout_config(key))
    nxt = jnp.stack([jnp.asarray(rot(k)) for k in range(steps)])
    for device in jax_devices():
        env = rj.Env(s, device=device, goals=goals, **rollout_config(key))
        assert env.num_instances == len(s) and env.table is s
        state, obs = env.reset(None, jnp.asarray(ids))
        assert np.array_equal(rj.as_words(state.states), ref[0]["states"])

        @jax.jit
        def roll(state):
            def body(st, nx):
                st, obs, reward, term, trunc, info = env.step(5, st, next_task_ids=nx)
                return st, (st.states, st.task_id, reward, term, trunc, info["final_state"], info["schema"], info["binding"])

            return jax.lax.scan(body, state, nxt)

        final, (st, tid, reward, term, trunc, fin, schema, binding) = roll(state)
        for k in range(steps):
            r = ref[k + 1]
            assert np.array_equal(rj.as_words(st[k]), r["states"]), (device, k)
            assert np.array_equal(np.asarray(tid[k]), r["task_ids"]), (device, k)
            assert np.array_equal(np.asarray(reward[k]), r["reward"]), (device, k)
            assert np.array_equal(np.asarray(term[k]), r["terminated"]), (device, k)
            assert np.array_equal(np.asarray(trunc[k]), r["truncated"]), (device, k)
            assert np.array_equal(rj.as_words(fin[k]), r["final_states"]), (device, k)
            assert np.array_equal(np.asarray(schema[k]), r["schema"]), (device, k)
            assert np.array_equal(np.asarray(binding[k]), r["binding"]), (device, k)
        if goals:
            assert np.array_equal(rj.as_words(final.goal_pos), ref[-1]["goal_pos"])
        # the expansion of the final states
        rows = rj.as_words(final.states)
        host = rl.expand(s, rows, np.asarray(final.task_id), goal=True, K=8)
        p = env.expand(final.states, 8, final.task_id)
        assert np.array_equal(np.asarray(p.count), np.asarray(host.padded.count)), device
        assert np.array_equal(np.asarray(p.schema), np.asarray(host.padded.schema)), device
        env.check_errors()


@pytest.mark.parametrize("key", list(SUITES))
def test_pool_suite_equals_host(sets, key):
    s = suite_of(sets, key)
    n, steps = 40, 10
    ids = env_orders(s, n)["interleaved"]
    goals = key in FAST
    rot = ROT(n, len(s))
    ref = host_rollout(s, ids, steps, goals=goals, next_ids=rot, **rollout_config(key))
    for threads, split in ((1, False), (4, True)):
        got = pool_rollout(s, ids, steps, threads, split, goals, rot, **rollout_config(key))
        for k in range(steps + 1):
            for f, v in got[k].items():
                assert np.array_equal(v, ref[k][f]), (threads, split, k, f)
    pool = rl.CpuEnvPool(s, 4)
    assert pool.table is s and pool.words == s.words
    with pytest.raises(ValueError, match="outside the suite"):
        pool.reset(task_ids=np.full(4, len(s), np.int32))


def test_torchrl_over_suites(sets):
    pytest.importorskip("torchrl")
    import torch
    import mymyr.rl.torch as rt
    from torchrl.envs.utils import check_env_specs

    s = suite_of(sets, "gripper+blocks")
    n = 12
    ids = torch.as_tensor(env_orders(s, n)["interleaved"])
    devices = ["cpu"]
    if torch.cuda.is_available():
        try:
            import mymyr.cuda as mc

            if mc.available():
                devices.append("cuda")
        except ImportError:
            pass
    for device in devices:
        env = rt.PlanningEnv(s, n, task_ids=ids, device=device, seed=0, max_steps=5, goal_reward=1.0, max_actions=8)
        check_env_specs(env, seed=3)
        td = env.reset()
        assert torch.equal(td["task_id"].cpu(), ids.to(torch.int64))
        init = torch.as_tensor(np.array(s.initial_states()).view(np.int64))
        assert torch.equal(td["state"].cpu(), init[ids.long()])
        roll = env.rollout(6, break_when_any_done=False)
        assert roll["action_mask"].shape == (n, 6, 8)


def test_mctx_over_suite(sets):
    pytest.importorskip("mctx")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import mctx

    from mymyr.rl import jax as rj

    s = suite_of(sets, "gripper+blocks")
    B, K = 12, 16
    ids = jnp.asarray(env_orders(s, B)["interleaved"])
    actions = []
    for device in jax_devices():
        env = rj.Env(s, device=device, autoreset=False)
        value = rj.goal_count_value(s)
        recurrent = rj.mctx_recurrent_fn(env, value, num_actions=K)
        state, _ = env.reset(None, ids)

        @jax.jit
        def search(state):
            root, invalid = rj.mctx_root(env, state, value, num_actions=K)
            return mctx.muzero_policy(None, jax.random.key(0), root, recurrent, num_simulations=12, invalid_actions=invalid)

        out = search(state)
        a = np.asarray(out.action)
        assert (a < np.minimum(np.asarray(state.count), K)).all()
        actions.append(a)
    assert all(np.array_equal(a, actions[0]) for a in actions)


# ------------------------------------------------------------------------------------------------ IW, datasets


def small_suite():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    paths = [FORK / "gripper/p-1-0.pddl", FORK / "blocks_4/p02-easy.pddl", FORK / "gripper/p-2-0.pddl",
             FORK / "blocks_4/test_problem.pddl"]
    if not all(p.is_file() for p in paths):
        pytest.skip("fork data missing")
    return rl.TaskSuite.group([mymyr.Task.from_pddl(p.parent / "domain.pddl", p, atoms="frozen") for p in paths])


def test_state_spaces_over_a_suite():
    s = small_suite()
    assert s.domain_of == [0, 1, 0, 1]
    want = [datasets.state_space(t, remove_if_unsolvable=False) for t in s.tasks]
    runs = [datasets.state_spaces(s, remove_if_unsolvable=False), datasets.state_spaces(s.tasks, threads=2, remove_if_unsolvable=False)]
    try:
        import mymyr.cuda as mc

        if mc.available():
            runs.append([d.to_host() for d in datasets.state_spaces(s, device=0, remove_if_unsolvable=False)])
            runs.append([d.to_host() for d in datasets.state_spaces(s.tasks, device=0, remove_if_unsolvable=False)])
            res = mc.generate_state_spaces(s, device=0, output="host", remove_if_unsolvable=False)
            runs.append([r.host for r in res])
    except ImportError:
        pass
    for spaces in runs:
        assert len(spaces) == len(s)
        for i, (a, b) in enumerate(zip(spaces, want)):
            assert a.task is s[i]
            for k, v in b.arrays().items():
                assert np.array_equal(a.arrays()[k], v), (i, k)


def iw_record(b, i):
    return (b.status[i], [str(a) for a in b.plan(i)], b.goal_state(i), b.cost(i),
            [(p.arity, p.status, p.expanded, p.generated) for p in b.passes(i)])


def test_iw_over_a_suite():
    dev = gpu_device()
    import torch

    import mymyr.cuda as mc

    s = small_suite()
    starts, ids = [], []
    for g in range(len(s)):
        w = walk(s[g], steps=3, seed=g, walks=1)[:3]
        starts += w
        ids += [g] * len(w)
    order = np.random.default_rng(1).permutation(len(starts))
    starts, ids = [starts[k] for k in order], [ids[k] for k in order]
    ctx = mc.Context(0, max_bytes=1 << 30)
    b = mc.multi_iw(s, starts, task_ids=ids, ctx=ctx, max_arity=2)
    assert isinstance(b, mc.SuiteIwBatch) and len(b) == len(starts) and b.task_ids == ids
    assert b.domains == [s.domain_of[g] for g in ids] and b.stats["passes"] > 0
    for i, (st, g) in enumerate(zip(starts, ids)):
        one = mc.multi_iw(s[g], [st], ctx=ctx, max_arity=2)
        assert iw_record(b, i) == iw_record(one, 0), i
    # batched IW(1): host starts, and device rows of the suite's width with device task ids
    rows = np.zeros((len(starts), s.words), np.uint64)
    for i, st in enumerate(starts):
        rows[i, : len(st.words)] = st.words
    host = mc.batched_iw1(s, starts, task_ids=ids, ctx=ctx)
    d = mc.batched_iw1(s, torch.from_numpy(rows.view(np.int64)).to(dev), task_ids=torch.tensor(ids, dtype=torch.int32, device=dev),
                       ctx=ctx)
    for i, (st, g) in enumerate(zip(starts, ids)):
        one = mc.batched_iw1(s[g], [st], ctx=ctx)
        assert iw_record(host, i) == iw_record(one, 0) == iw_record(d, i), i
    with pytest.raises(ValueError, match="task_ids"):
        mc.multi_iw(s, starts, ctx=ctx)
    with pytest.raises(ValueError, match="outside the suite"):
        mc.multi_iw(s, starts[:1], task_ids=[len(s)], ctx=ctx)
    ctx.synchronize()
