"""mymyr.rl.jax: the functional JAX environment against rl::HostEnv step by step (states, labels, rewards,
terminated / truncated, final states, autoreset) under jit, vmap and lax.scan, on the CPU and (with a visible GPU) on
CUDA; batch splits, env ids and concurrent calls; the mctx adapter; flashbax items; the jnp prefix masks, novelty
rewards and HER relabels against the native ones.

Without JAX (or a mymyr built without the XLA FFI headers) the module is skipped. The GPU cases run only when a GPU is
made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_jax.py

MYMYR_TEST_FULL_SUITE=1 runs the environment equality on all 22 suite tasks (the default: the small ones).
"""

import os
import threading

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, SMALL_TASKS, TASKS, text_task

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from mymyr._core import _rl_jax, _rl_torch  # noqa: E402

if not _rl_jax.has_ffi():
    pytest.skip("mymyr was built without the XLA FFI headers", allow_module_level=True)

import mymyr.rl.jax as rj  # noqa: E402

NUMERIC = ROOT / "tests/data/numeric_tasks"
ALL_TASKS = sorted(p.stem for p in TASKS.glob("*.txt"))
SUITE = ALL_TASKS if os.environ.get("MYMYR_TEST_FULL_SUITE") else SMALL_TASKS


def devices():
    out = ["cpu"]
    try:
        if jax.devices("gpu"):
            out.append("cuda")
    except RuntimeError:
        pass
    return out


def need_gpu():
    if "cuda" not in devices():
        pytest.skip("no visible CUDA device for JAX (set CUDA_VISIBLE_DEVICES and JAX_PLATFORMS)")
    import mymyr.cuda as mc

    if not mc.available():
        pytest.skip("mymyr sees no CUDA device")


# ------------------------------------------------------------------------------------------------ the reference


class HostRef:
    """rl::HostEnv (mymyr._core._rl_torch.Env on the CPU) on NumPy arrays: the reference of every step."""

    def __init__(self, task, n, seed, first_env=0, **cfg):
        self.env = _rl_torch.Env(task, seed=seed, **cfg)
        rw = self.env.row_words
        self.first_env = first_env
        self.states = np.tile(self.env.initial_states()[0], (n, 1))
        self.steps = np.zeros(n, np.int32)
        self.draws = np.zeros(n, np.uint64)
        self.count = np.full(n, self.env.initial_counts[0], np.int32)
        self.n, self.rw, self.L = n, rw, self.env.label_width

    def step(self, action=None):
        n = self.n
        out = dict(
            reward=np.zeros(n, np.float32),
            terminated=np.zeros(n, np.bool_),
            truncated=np.zeros(n, np.bool_),
            count=np.zeros(n, np.int32),
            final_states=np.zeros((n, self.rw), np.uint64),
            schema=np.zeros(n, np.int32),
            binding=np.zeros((n, self.L), np.int32),
            invalid=np.zeros(n, np.bool_),
            goal=np.zeros(n, np.bool_),
        )
        act = None if action is None else np.asarray(action, np.int64)
        self.env.step(self.states, steps=self.steps, draws=self.draws, action=act, first_env=self.first_env, **out)
        self.count = out["count"]
        return out


def u64(x):
    return rj.as_words(jax.device_get(x))


def check_equal(ref, out, state, obs, t):
    """One JAX step against the reference's step."""
    s2, o2, reward, terminated, truncated, info = out
    assert np.array_equal(u64(s2.states), ref.states), t
    assert np.array_equal(np.asarray(s2.steps), ref.steps), t
    draws = np.asarray(s2.draws).astype(np.uint64)
    assert np.array_equal(draws[:, 0] | (draws[:, 1] << np.uint64(32)), ref.draws), t
    assert np.array_equal(np.asarray(s2.count), ref.count), t
    assert np.array_equal(np.asarray(o2.count), ref.count), t
    r = ref.last
    assert np.array_equal(np.asarray(reward), r["reward"]), t
    assert np.array_equal(np.asarray(terminated), r["terminated"]), t
    assert np.array_equal(np.asarray(truncated), r["truncated"]), t
    assert np.array_equal(np.asarray(info["goal"]), r["goal"]), t
    assert np.array_equal(np.asarray(info["invalid"]), r["invalid"]), t
    assert np.array_equal(np.asarray(info["schema"]), r["schema"]), t
    assert np.array_equal(np.asarray(info["binding"]), r["binding"]), t
    assert np.array_equal(u64(info["final_state"]), r["final_states"]), t


CFG = dict(max_steps=12, step_reward=-1.0, goal_reward=2.5, dead_end_reward=-0.5)


def actions_for(t, count, gen):
    """The step's actions: the random policy (None) on most steps, given ones (some invalid) on every third."""
    if t % 3 != 2:
        return None
    a = (gen.random(count.shape[0]) * (count + 1).clip(min=1)).astype(np.int32)
    a[::7] = -1
    return a


def run_equal(task, device, n=64, steps=30, seed=5, **cfg):
    """A jitted JAX env equals HostEnv step by step; then lax.scan and vmap over the same trajectory."""
    env = rj.Env(task, device=device, **cfg)
    ref = HostRef(task, n, seed, **cfg)
    key = jax.random.key(seed)
    state, obs = env.reset(key, n)
    assert np.array_equal(u64(state.states), ref.states)
    assert np.array_equal(np.asarray(state.count), ref.count)
    step = jax.jit(env.step)
    gen = np.random.default_rng(3)
    start = state
    acts = []
    for t in range(steps):
        a = actions_for(t, ref.count, gen)
        acts.append(np.full(n, -2, np.int32) if a is None else a)
        ref.last = ref.step(a)
        out = step(key, state) if a is None else step(key, state, jnp.asarray(a))
        check_equal(ref, out, state, obs, t)
        state, obs = out[0], out[1]
    env.check_errors()
    return env, key, start, np.stack(acts), state


def scan_rollout(env, key, state, acts):
    """The same trajectory in one lax.scan (-2 rows: the random policy of that step)."""

    def body(s, a):
        rand = env.step(key, s)
        given = env.step(key, s, jnp.maximum(a, -1))
        pick = a[0] == -2
        out = jax.tree.map(lambda x, y: jnp.where(pick, x, y), rand, given)
        return out[0], (out[2], out[5]["final_state"])

    return jax.jit(lambda s, a: jax.lax.scan(body, s, a))(state, jnp.asarray(acts))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
@pytest.mark.parametrize("name", SUITE)
def test_jax_env_equals_host_env(name, atoms, device):
    if device == "cuda":
        need_gpu()
    task = text_task(name, atoms=atoms)
    env, key, start, acts, last = run_equal(task, device, **CFG)
    final, (rewards, finals) = scan_rollout(env, key, start, acts)
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(last)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # vmap: two halves as a batch dimension (per-row env ids keep each environment's stream)
    half = jax.tree.map(lambda x: x.reshape(2, x.shape[0] // 2, *x.shape[1:]), start)
    vstep = jax.jit(jax.vmap(env.step, in_axes=(None, 0)))
    whole = jax.jit(env.step)(key, start)
    split = vstep(key, half)
    for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(split)):
        assert np.array_equal(np.asarray(a), np.asarray(b).reshape(np.asarray(a).shape))
    if device == "cuda" and env.fast:
        env2 = rj.Env(task, device=device, path="general", **CFG)
        s2, _ = env2.reset(key, start.states.shape[0])
        g = jax.jit(env2.step)(key, s2)
        f = jax.jit(env.step)(key, env.reset(key, s2.states.shape[0])[0])
        assert np.array_equal(np.asarray(g[0].states), np.asarray(f[0].states))
        assert np.array_equal(np.asarray(g[2]), np.asarray(f[2]))


def test_autoreset_off_and_reset_where():
    task = text_task("blocks__probBLOCKS-8-0")
    for device in devices():
        env = rj.Env(task, device=device, autoreset=False, max_steps=4)
        ref = HostRef(task, 40, 2, autoreset=False, max_steps=4)
        key = jax.random.key(2)
        state, _ = env.reset(key, 40)
        step = jax.jit(env.step)
        for t in range(9):
            ref.last = ref.step()
            out = step(key, state)
            check_equal(ref, out, state, None, t)
            state = out[0]
            done = np.asarray(out[3]) | np.asarray(out[4])
            if done.any():
                ref.env.reset(ref.states, steps=ref.steps, count=ref.count, mask=done)
                state, obs = jax.jit(env.reset_where)(state, jnp.asarray(done))
                assert np.array_equal(u64(state.states), ref.states)
                assert np.array_equal(np.asarray(obs.count), ref.count)
        env.check_errors()


def test_batch_splits_env_ids_and_threads():
    task = text_task("gripper__prob05")
    for device in devices():
        env = rj.Env(task, device=device, max_steps=7)
        key = jax.random.key(11)
        n = 300

        @jax.jit
        def rollout(state):
            def body(s, _):
                s, obs, r, term, trunc, info = env.step(key, s)
                return s, (r, info["schema"])

            return jax.lax.scan(body, state, None, length=20)

        whole, _ = env.reset(key, n)
        a, _ = env.reset(key, 120)
        b, _ = env.reset(key, n - 120, first_env=120)
        rw, (rr, rs) = rollout(whole)
        ra, (ar, as_) = rollout(a)
        rb, (br, bs) = rollout(b)
        assert np.array_equal(np.asarray(rw.states), np.concatenate([np.asarray(ra.states), np.asarray(rb.states)]))
        assert np.array_equal(np.asarray(rr), np.concatenate([np.asarray(ar), np.asarray(br)], axis=1))
        assert np.array_equal(np.asarray(rs), np.concatenate([np.asarray(as_), np.asarray(bs)], axis=1))
        # concurrent calls from threads (one Env, the device's XLA streams) give the same results
        results = [None] * 4

        def work(i):
            results[i] = jax.device_get(rollout(whole)[1])

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for r in results:
            assert np.array_equal(r[0], np.asarray(rr)) and np.array_equal(r[1], np.asarray(rs))
        env.check_errors()


def test_keys_seed_the_counter_rng():
    task = text_task("blocks__probBLOCKS-8-0")
    env = rj.Env(task, device="cpu")
    assert np.array_equal(np.asarray(rj.key_seed(jax.random.key(7))), [7, 0])
    assert np.array_equal(np.asarray(rj.key_seed(jax.random.PRNGKey(7))), [7, 0])
    assert np.array_equal(np.asarray(rj.key_seed((1 << 40) + 3)), [3, 1 << 8])
    # per-step keys: each step's draws follow its key
    state, _ = env.reset(None, 16)
    k1, k2 = jax.random.key(1), jax.random.key(2)
    a = env.step(k1, state)
    b = env.step(k2, state)
    ref1, ref2 = HostRef(task, 16, 1), HostRef(task, 16, 2)
    ref1.step()
    ref2.step()
    assert np.array_equal(u64(a[0].states), ref1.states) and np.array_equal(u64(b[0].states), ref2.states)
    # the jnp Philox equals the native one
    c = np.arange(40, dtype=np.uint32).reshape(10, 4)
    want = np.asarray(_rl_torch.philox(c, 0x1234_5678_9ABC_DEF0), dtype=np.uint32)
    got = np.stack(rj.rng.philox(c[:, 0], c[:, 1], c[:, 2], c[:, 3], 0x9ABCDEF0, 0x12345678), axis=-1)
    assert np.array_equal(np.asarray(got), want)


def test_expansions_equal_rl_expand():
    task = text_task("logistics00__probLOGISTICS-6-1")
    rng = np.random.default_rng(1)
    for device in devices():
        env = rj.Env(task, device=device)
        state, _ = env.reset(None, 32)
        key = jax.random.key(3)
        for _ in range(int(rng.integers(3, 9))):
            state = env.step(key, state)[0]
        rows = u64(state.states)
        x = rl.expand(task, rows, words=env.row_words, goal=True)
        flat = env.expand_flat(state.states, int(x.total) + 5)
        assert np.array_equal(u64(flat.succ)[: x.total], x.succ)
        assert np.array_equal(np.asarray(flat.parent)[: x.total], x.parent)
        assert np.array_equal(np.asarray(flat.schema)[: x.total], x.schema)
        assert np.array_equal(np.asarray(flat.binding)[: x.total], x.binding)
        assert np.array_equal(np.asarray(flat.goal)[: x.total], x.goal)
        assert np.array_equal(np.asarray(flat.offsets), x.offsets)
        assert np.all(np.asarray(flat.parent)[x.total :] == -1)
        pad = env.expand(state.states, 16)
        p = x.pad(16)
        assert np.array_equal(np.asarray(pad.count), p.count)
        assert np.array_equal(np.asarray(pad.mask), p.mask)
        assert np.array_equal(np.asarray(pad.schema), p.schema)
        assert np.array_equal(np.asarray(pad.binding), p.binding)
        assert np.array_equal(u64(pad.succ).reshape(p.succ.shape), p.succ)
        # vmap of the padded expansion
        v = jax.vmap(lambda s: env.expand(s, 16))(state.states.reshape(4, 8, -1))
        assert np.array_equal(np.asarray(v.schema).reshape(32, 16), p.schema)


def test_numeric_tasks_on_the_cpu():
    task = mymyr.Task.from_text(str(NUMERIC / "cs-counters.txt"))
    assert task.numeric_slots > 0
    env = rj.Env(task, max_steps=10)
    assert env.device.platform == "cpu"
    ref = HostRef(task, 8, 4, max_steps=10)
    key = jax.random.key(4)
    state, _ = env.reset(key, 8)
    for t in range(12):
        ref.last = ref.step()
        out = jax.jit(env.step)(key, state)
        check_equal(ref, out, state, None, t)
        state = out[0]
    if "cuda" in devices():
        device = rj.Env(task, device="cuda", max_steps=10)
        state, _ = device.reset(key, 8)
        ref = HostRef(task, 8, 4, max_steps=10)
        for t in range(12):
            ref.last = ref.step()
            out = jax.jit(device.step)(key, state)
            check_equal(ref, out, state, None, t)
            state = out[0]


# ------------------------------------------------------------------------------------------------ mctx, flashbax


@pytest.mark.parametrize("policy", ["muzero", "gumbel"])
@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1"])
def test_mctx_on_successor_slots(name, policy):
    mctx = pytest.importorskip("mctx")
    task = text_task(name)
    B, K = 16, 32
    for device in devices():
        env = rj.Env(task, device=device, autoreset=False)
        value = rj.goal_count_value(task)
        recurrent = rj.mctx_recurrent_fn(env, value, num_actions=K)

        @jax.jit
        def search(state, key):
            root, invalid = rj.mctx_root(env, state, value, num_actions=K)
            if policy == "muzero":
                out = mctx.muzero_policy(None, key, root, recurrent, num_simulations=16, invalid_actions=invalid)
            else:
                out = mctx.gumbel_muzero_policy(
                    None, key, root, recurrent, num_simulations=16, invalid_actions=invalid, max_num_considered_actions=8
                )
            return out.action, out.search_tree.children_index

        # the adapter hands mctx no index: logits, values, rewards and discounts are floats, the embedding is data
        root, invalid = jax.jit(lambda s: rj.mctx_root(env, s, value, num_actions=K))(env.reset(None, B)[0])
        rec, _ = jax.jit(lambda s, a: recurrent(None, None, a, s))(env.reset(None, B)[0], jnp.zeros(B, jnp.int32))
        for x in (root.prior_logits, root.value, rec.prior_logits, rec.value, rec.reward, rec.discount):
            assert x.dtype == jnp.float32
        assert invalid.dtype == jnp.bool_ and invalid.shape == (B, K)
        state, obs = env.reset(None, B)
        step = jax.jit(env.step)
        for t in range(6):
            action, children = search(state, jax.random.key(t))
            again, _ = search(state, jax.random.key(t))
            assert np.array_equal(np.asarray(action), np.asarray(again))  # deterministic
            count = np.asarray(state.count)
            a = np.asarray(action)
            assert np.all((a >= 0) & ((a < count) | (count == 0))), (t, a, count)
            ch = np.asarray(children)  # mctx's node indices: -1 (unexpanded) or a node of the 16 simulations
            assert ch.min() >= -1 and ch.max() <= 16
            state, obs, reward, terminated, truncated, info = step(0, state, action)
            assert not np.any(np.asarray(info["invalid"]) & (count > 0))
            done = np.asarray(terminated)
            if done.any():
                state, obs = env.reset_where(state, jnp.asarray(done))
        env.check_errors()
    with pytest.raises(ValueError, match="autoreset"):
        rj.mctx_recurrent_fn(rj.Env(task, device="cpu"), rj.goal_count_value(task), num_actions=K)


def test_flashbax_item_buffer():
    fbx = pytest.importorskip("flashbax")
    task = text_task("blocks__probBLOCKS-8-0")
    for device in devices():
        env = rj.Env(task, device=device, max_steps=10, goal_reward=1.0)
        spec = rj.transition_spec(env)
        example = rj.transition_example(env)
        assert {k: (v.shape, v.dtype) for k, v in example.items()} == {k: (v.shape, v.dtype) for k, v in spec.items()}
        buf = fbx.make_item_buffer(max_length=4096, min_length=16, sample_batch_size=64, add_batches=True)
        key = jax.random.key(1)
        state, obs = env.reset(key, 32)

        @jax.jit
        def collect(state, obs, bstate):
            def body(c, _):
                state, obs, bstate = c
                out = env.step(key, state)
                bstate = buf.add(bstate, rj.transition_item(obs, None, out))
                return (out[0], out[1], bstate), None

            return jax.lax.scan(body, (state, obs, buf.init(example) if bstate is None else bstate), None, length=20)[0]

        state, obs, bstate = collect(state, obs, None)
        sample = jax.device_get(buf.sample(bstate, jax.random.key(2)).experience)
        for k, v in spec.items():
            assert sample[k].shape == (64, *v.shape) and sample[k].dtype == v.dtype
        rows = rj.as_words(sample["state"])
        nxt = rj.as_words(sample["next_state"])
        x = rl.expand(task, rows, words=env.row_words, goal=True)
        off = np.asarray(x.offsets)
        for i in range(64):
            s = int(sample["schema"][i])
            if s < 0:
                assert np.array_equal(nxt[i], rows[i])
                continue
            j = [
                j
                for j in range(off[i], off[i + 1])
                if x.schema[j] == s and np.array_equal(x.binding[j], sample["binding"][i])
            ]
            assert len(j) == 1 and np.array_equal(x.succ[j[0]], nxt[i])
            assert bool(sample["goal"][i]) == bool(x.goal[j[0]])
            assert float(sample["reward"][i]) == -1.0 + float(x.goal[j[0]])


# ------------------------------------------------------------------------------------------------ jnp helpers


def jwindow(task, n, steps, seed=3):
    from test_rl_ops import window

    return window(task, n=n, steps=steps, seed=seed)


def on(device):
    return jax.devices("gpu")[0] if device == "cuda" else jax.devices("cpu")[0]


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "sokoban-opt08-strips__p14"])
def test_jnp_prefix_masks_equal_native(name):
    from test_rl_ops import queries

    task = text_task(name)
    states, _ = jwindow(task, 24, 7)
    rows = states[-1]
    exp = rl.expand(task, rows)
    schema, prefix = queries(exp, 24)
    for device in devices():
        env = rj.Env(task, device=device)
        js = jax.device_put(jnp.asarray(rows.view(np.uint32)), on(device))
        flat = env.expand_flat(js, int(exp.total) + 8)
        pad = env.expand(js, 32)
        fm = jax.jit(rj.prefix_masks, static_argnums=(3, 4))
        for d in range(exp.label_width):
            want = rl.prefix_masks(exp, schema, prefix, d)
            q, p = jnp.asarray(schema), jnp.asarray(prefix)
            assert np.array_equal(np.asarray(fm(flat, q, p, d, task.num_objects)), want), (device, d)
            assert np.array_equal(np.asarray(fm(pad, q, p, d, task.num_objects)), want), (device, d)
            # rl.prefix_masks dispatches JAX expansions to the jnp version
            assert np.array_equal(np.asarray(rl.prefix_masks(pad, q, p, d, num_objects=task.num_objects)), want)
        want_s = rl.schema_masks(exp)
        assert np.array_equal(np.asarray(rj.schema_masks(flat, 24, task.num_schemas)), want_s)
        assert np.array_equal(np.asarray(rl.schema_masks(pad, num_schemas=task.num_schemas)), want_s)


def test_jnp_novelty_equals_native():
    task = text_task("gripper__prob05")
    states, _ = jwindow(task, 16, 20)
    for device in devices():
        seen = np.zeros((16, states.shape[-1]), np.uint64)
        jseen = jax.device_put(jnp.zeros((16, 2 * states.shape[-1]), jnp.uint32), on(device))
        upd = jax.jit(rj.novelty_update)
        for t in range(states.shape[0]):
            r, _ = rl.novelty_update(seen, states[t])
            jr, jseen = upd(jseen, jnp.asarray(states[t].view(np.uint32)))
            assert np.array_equal(np.asarray(jr), r), (device, t)
            assert np.array_equal(rj.as_words(jseen), seen)


@pytest.mark.parametrize("strategy", ["future", "final", "episode"])
def test_jnp_her_equals_native(strategy):
    task = text_task("sokoban-opt08-strips__p14")
    states, done = jwindow(task, 12, 24)
    atoms = np.full(states.shape[-1], 0x0F0F0F0F0F0F0F0F, np.uint64)
    for device in devices():
        for subset in (None, 2, 5):
            for extras in (False, True):
                kw = dict(strategy=strategy, k=3, subset=subset, step_reward=-0.5, goal_reward=4.0)
                if extras:
                    kw.update(goal_atoms=atoms, env_ids=np.arange(100, 112, dtype=np.uint32))
                want = rl.her_relabel(states, done, seed=0xABCDEF123, **kw)
                js = jax.device_put(jnp.asarray(states.view(np.uint32)), on(device))
                jd = jax.device_put(jnp.asarray(done), on(device))
                jkw = dict(kw)
                if extras:
                    jkw.update(goal_atoms=jnp.asarray(atoms.view(np.uint32)), env_ids=jnp.asarray(kw["env_ids"]))
                f = jax.jit(lambda s, d: rj.her_relabel(s, d, 0xABCDEF123, **jkw))
                got = f(js, jd)
                assert np.array_equal(rj.as_words(got.goal), want.goal), (device, subset, extras)
                for a, b in zip(got[1:], want[1:]):
                    assert np.array_equal(np.asarray(a), b), (device, subset, extras)
                # rl.her_relabel dispatches JAX arrays to the jnp version
                g2 = rl.her_relabel(js, jd, seed=0xABCDEF123, **jkw)
                assert np.array_equal(np.asarray(g2.source), want.source)


@pytest.mark.parametrize("name", ["gripper__prob05", "logistics00__probLOGISTICS-6-1"])
def test_indices_stay_in_range(name):
    """No index that mymyr.rl.jax computes reaches a gather or scatter out of range (checkify's index checks over the
    step, the expansions, the masks, HER, novelty, random actions and flashbax items), and the FFI outputs that are
    labels or counts stay in their ranges (-1 only as the documented "none"). The mctx adapter passes no index to mctx
    (prior logits, values, rewards, discounts are floats; the embedding is data): mctx's own indices are checked in
    test_mctx_on_successor_slots (checkify cannot check mctx's batched scatters)."""
    from jax.experimental import checkify

    def checked(f, *args):
        err, out = jax.jit(checkify.checkify(f, errors=checkify.index_checks))(*args)
        err.throw()
        return out

    # the check is live for the scatter mode the masks use
    with pytest.raises(checkify.JaxRuntimeError, match="out-of-bounds"):
        checked(lambda r: jnp.zeros((4, 3), jnp.uint8).at[r, 0].max(1, mode="promise_in_bounds"), jnp.array([4]))
    task = text_task(name)
    S, O = task.num_schemas, task.num_objects
    n = 24
    for device in devices():
        env = rj.Env(task, device=device, max_steps=5)
        key = jax.random.key(3)
        state, obs = env.reset(key, n)
        ids = np.asarray(state.env_id)
        reached, done = [], []
        for t in range(12):
            if t % 3 == 2:
                a = jnp.asarray((np.arange(n) % 7 - 1).astype(np.int32))  # -1 and slots at or past the count
                out = checked(lambda s, a: env.step(key, s, a), state, a)
            else:
                out = checked(lambda s: env.step(key, s), state)
            s2, o2, reward, terminated, truncated, info = out
            assert np.all(np.asarray(s2.count) >= 0) and np.array_equal(np.asarray(s2.env_id), ids)
            sch, bnd = np.asarray(info["schema"]), np.asarray(info["binding"])
            assert np.all((sch >= -1) & (sch < S)) and np.all((bnd >= -1) & (bnd < O))
            assert np.all((sch >= 0) | np.asarray(info["invalid"]) | (np.asarray(state.count) == 0))
            ra = np.asarray(checked(lambda s: env.random_actions(key, s, 3), s2))
            c = np.asarray(s2.count)
            assert np.all((ra >= 0) & ((ra < np.minimum(c, 3)) | (c == 0)))
            checked(lambda o, s: rj.transition_item(o, None, env.step(key, s)), o2, s2)
            reached.append(info["final_state"])
            done.append(terminated | truncated)
            state = s2
        checked(lambda s, m: env.reset_where(s, m), state, jnp.asarray(np.arange(n) % 2 == 0))
        pad = env.expand(state.states, 16)
        flat = env.expand_flat(state.states, 600)
        par = np.asarray(flat.parent)
        total = min(int(np.asarray(flat.offsets)[-1]), par.shape[0])
        assert np.all((par[:total] >= 0) & (par[:total] < n)) and np.all(par[total:] == -1)
        assert np.all(np.diff(np.asarray(flat.offsets)) >= 0) and np.asarray(flat.offsets)[0] == 0
        for e in (pad, flat):
            es, eb = np.asarray(e.schema), np.asarray(e.binding)
            assert np.all((es >= -1) & (es < S)) and np.all((eb >= -1) & (eb < O))
        m = np.asarray(pad.mask)
        assert np.all(np.asarray(pad.schema)[m] >= 0)
        qs = jnp.asarray(np.asarray(pad.schema)[:, 0].clip(0))
        qp = jnp.asarray(np.asarray(pad.binding)[:, 0])
        for d in range(pad.binding.shape[-1] + 1):
            for e in (pad, flat):
                checked(lambda e, s, p: rj.prefix_masks(e, s, p, d, O), e, qs, qp)
        for e in (pad, flat):
            checked(lambda e: rj.schema_masks(e, n, S), e)
        states, dones = jnp.stack(reached), jnp.stack(done)
        for strategy in ("future", "final", "episode"):
            for subset in (None, 2):
                h = checked(lambda s, d: rj.her_relabel(s, d, 7, strategy=strategy, k=3, subset=subset), states, dones)
                src = np.asarray(h.source)
                assert np.all((src >= 0) & (src < states.shape[0]))
        checked(lambda a, b: rj.novelty_update(a, b), jnp.zeros((n, states.shape[-1]), jnp.uint32), states[0])
        env.check_errors()


def test_random_actions_are_the_random_policy():
    task = text_task("gripper__prob05")
    for device in devices():
        env = rj.Env(task, device=device, max_steps=5)
        key = jax.random.key(4)
        state, _ = env.reset(key, 50, first_env=7)
        step = jax.jit(env.step)
        ra = jax.jit(env.random_actions, static_argnums=2)
        for _ in range(10):
            a = ra(key, state, 0)
            r1 = step(key, state)
            r2 = step(key, state, a)
            for x, y in zip(jax.tree.leaves(r1), jax.tree.leaves(r2)):
                assert np.array_equal(np.asarray(x), np.asarray(y))
            limited = np.asarray(ra(key, state, 2))
            assert np.all(limited < np.maximum(np.minimum(np.asarray(state.count), 2), 1))
            state = r1[0]
