"""mymyr.rl's RL helpers: prefix and schema masks against the expanded labels, width-1 novelty rewards against a
set-based reference, hindsight relabels (strategies, subsets, determinism, batch independence), on NumPy, and on torch
CPU / CUDA tensors (equal to NumPy) when torch is installed. The jnp versions are compared in test_rl_jax.py.

The CUDA cases run only when a GPU is made visible explicitly (conftest.py hides GPUs by default).
"""

import numpy as np
import pytest

from mymyr import rl
from mymyr._core import _rl_torch
from conftest import SMALL_TASKS, text_task

MASK_TASKS = SMALL_TASKS + ["sokoban-opt08-strips__p14", "freecell__p02"]


def window(task, n=24, steps=30, seed=3, max_steps=6):
    """Reached states [T, N, RW] (uint64) and done [T, N] of random-policy environments (rl::HostEnv)."""
    env = _rl_torch.Env(task, seed=seed, max_steps=max_steps)
    rw = env.row_words
    states = np.tile(env.initial_states()[0], (n, 1))
    steps_ = np.zeros(n, np.int32)
    draws = np.zeros(n, np.uint64)
    out_s, out_d = [], []
    for _ in range(steps):
        final = np.zeros((n, rw), np.uint64)
        term = np.zeros(n, np.bool_)
        trunc = np.zeros(n, np.bool_)
        env.step(states, steps=steps_, draws=draws, final_states=final, terminated=term, truncated=trunc)
        out_s.append(final)
        out_d.append(term | trunc)
    return np.stack(out_s), np.stack(out_d)


def queries(exp, n):
    """Each state's (i mod count)-th label as the taken action; every fifth prefix spoiled (no successor extends it)."""
    off = np.asarray(exp.offsets)
    schema = np.zeros(n, np.int32)
    prefix = np.full((n, exp.label_width), -1, np.int32)
    for i in range(n):
        c = off[i + 1] - off[i]
        if c == 0:
            continue
        j = off[i] + i % c
        schema[i] = exp.schema[j]
        prefix[i] = exp.binding[j]
        if i % 5 == 4:
            prefix[i, 0] = exp.num_objects
    return schema, prefix


def brute_masks(exp, schema, prefix, depth, n_obj):
    off = np.asarray(exp.offsets)
    out = np.zeros((len(schema), n_obj), np.bool_)
    for i in range(len(schema)):
        for j in range(off[i], off[i + 1]):
            b = exp.binding[j]
            if exp.schema[j] == schema[i] and np.array_equal(b[:depth], prefix[i, :depth]) and depth < len(b) and b[depth] >= 0:
                out[i, b[depth]] = True
    return out


@pytest.mark.parametrize("name", MASK_TASKS)
def test_prefix_and_schema_masks_equal_the_labels(name):
    task = text_task(name)
    states, _ = window(task, steps=7)
    rows = states[-1]
    exp = rl.expand(task, rows)
    n = rows.shape[0]
    schema, prefix = queries(exp, n)
    sm = rl.schema_masks(exp)
    off = np.asarray(exp.offsets)
    want = np.zeros((n, task.num_schemas), np.bool_)
    for i in range(n):
        want[i, exp.schema[off[i] : off[i + 1]]] = True
    assert np.array_equal(sm, want)
    pad = exp.pad(0)
    for depth in range(exp.label_width):
        m = rl.prefix_masks(exp, schema, prefix, depth)
        assert m.dtype == np.bool_ and m.shape == (n, task.num_objects)
        assert np.array_equal(m, brute_masks(exp, schema, prefix, depth, task.num_objects)), depth
        # the padded view gives the same masks
        assert np.array_equal(rl.prefix_masks(pad, schema, prefix, depth, num_objects=task.num_objects), m)
    assert np.array_equal(rl.schema_masks(pad, num_schemas=task.num_schemas), sm)


def test_mask_errors():
    task = text_task("blocks__probBLOCKS-8-0")
    exp = rl.expand(task, [task.initial_state])
    with pytest.raises(ValueError, match="num_objects"):
        rl.prefix_masks(exp.pad(4), np.zeros(1, np.int32), None, 0)
    with pytest.raises(ValueError, match="prefix"):
        rl.prefix_masks(exp, np.zeros(1, np.int32), np.zeros((1, 0), np.int32), 1)


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "sokoban-opt08-strips__p14"])
def test_novelty_update_counts_new_atoms(name):
    task = text_task(name)
    states, _ = window(task, n=16, steps=25)
    W = states.shape[-1]
    seen = np.zeros((16, W), np.uint64)
    ref = [set() for _ in range(16)]
    for t in range(states.shape[0]):
        r, out = rl.novelty_update(seen, states[t])
        assert out is seen and r.dtype == np.int32
        for i in range(16):
            atoms = {64 * w + b for w in range(W) for b in range(64) if (int(states[t, i, w]) >> b) & 1}
            assert r[i] == len(atoms - ref[i])
            ref[i] |= atoms
            assert all((int(seen[i, a // 64]) >> (a % 64)) & 1 for a in ref[i])
    with pytest.raises(ValueError, match="width 1"):
        rl.novelty_update(seen, states[0], width=2)


def check_her(states, done, h, strategy, k, subset, atoms=None):
    T, N, W = states.shape
    for t in range(T):
        for i in range(N):
            start, end = t, t
            while start > 0 and not done[start - 1, i]:
                start -= 1
            while end + 1 < T and not done[end, i]:
                end += 1
            lo = {"future": t, "final": end, "episode": start}[strategy]
            for j in range(k):
                src = int(h.source[t, i, j])
                assert lo <= src <= end
                cand = states[src, i] & (atoms if atoms is not None else np.uint64(0xFFFFFFFFFFFFFFFF))
                g = h.goal[t, i, j]
                assert np.all(g & ~cand == 0)
                pc = sum(bin(int(x)).count("1") for x in cand)
                size = sum(bin(int(x)).count("1") for x in g)
                assert size == (pc if subset is None else min(subset, pc))
                ok = bool(np.all(states[t, i] & g == g))
                assert bool(h.achieved[t, i, j]) == ok
                assert float(h.reward[t, i, j]) == (-1.0 + 3.0 if ok else -1.0)


@pytest.mark.parametrize("strategy", ["future", "final", "episode"])
@pytest.mark.parametrize("subset", [None, 1, 3])
def test_her_relabel(strategy, subset):
    task = text_task("gripper__prob05")
    states, done = window(task, n=10, steps=20)
    h = rl.her_relabel(states, done, strategy=strategy, k=3, subset=subset, seed=11, goal_reward=3.0)
    assert h.goal.shape == (20, 10, 3, states.shape[-1]) and h.goal.dtype == np.uint64
    check_her(states, done, h, strategy, 3, subset)
    atoms = np.full(states.shape[-1], 0x5555555555555555, np.uint64)
    ha = rl.her_relabel(states, done, strategy=strategy, k=3, subset=subset, seed=11, goal_reward=3.0, goal_atoms=atoms)
    check_her(states, done, ha, strategy, 3, subset, atoms)
    # deterministic; a column block with its env ids equals the whole batch's columns
    h2 = rl.her_relabel(states, done, strategy=strategy, k=3, subset=subset, seed=11, goal_reward=3.0)
    assert all(np.array_equal(a, b) for a, b in zip(h, h2))
    part = rl.her_relabel(
        np.ascontiguousarray(states[:, 4:9]), np.ascontiguousarray(done[:, 4:9]), strategy=strategy, k=3, subset=subset,
        seed=11, goal_reward=3.0, env_ids=np.arange(4, 9, dtype=np.uint32),
    )
    first = rl.her_relabel(
        np.ascontiguousarray(states[:, 4:9]), np.ascontiguousarray(done[:, 4:9]), strategy=strategy, k=3, subset=subset,
        seed=11, goal_reward=3.0, first_env=4,
    )
    for a, b, c in zip(h, part, first):
        assert np.array_equal(a[:, 4:9], b) and np.array_equal(b, c)


# ------------------------------------------------------------------------------------------------ torch


def torch_devices():
    torch = pytest.importorskip("torch")
    out = [torch.device("cpu")]
    if torch.cuda.is_available():
        try:
            import mymyr.cuda as mc

            if mc.available():
                out.append(torch.device("cuda", 0))
        except ImportError:
            pass
    return out


def test_torch_equals_numpy():
    torch = pytest.importorskip("torch")
    task = text_task("logistics00__probLOGISTICS-6-1")
    states, done = window(task, n=12, steps=16)
    exp = rl.expand(task, states[-1])
    schema, prefix = queries(exp, 12)
    want_masks = [rl.prefix_masks(exp, schema, prefix, d) for d in range(exp.label_width)]
    want_schema = rl.schema_masks(exp)
    seen = np.zeros((12, states.shape[-1]), np.uint64)
    want_r = [rl.novelty_update(seen, states[t])[0].copy() for t in range(16)]
    want_h = rl.her_relabel(states, done, k=2, subset=2, seed=4)
    for dev in torch_devices():
        s = torch.from_numpy(states.view(np.int64)).to(dev)
        rows = s[-1].contiguous()
        if dev.type == "cuda":
            import mymyr.cuda as mc

            x = rl.expand(task, rows)
            assert isinstance(x, mc.DeviceExpansion)
            kw = dict(num_objects=task.num_objects)
            skw = dict(num_schemas=task.num_schemas)
        else:
            x = rl.expand(task, rows)
            kw, skw = {}, {}
        q = torch.from_numpy(schema).to(dev)
        p = torch.from_numpy(prefix).to(dev)
        for d in range(exp.label_width):
            m = rl.prefix_masks(x, q, p, d, **kw)
            assert m.device == dev and m.dtype == torch.bool
            assert np.array_equal(m.cpu().numpy(), want_masks[d]), (dev, d)
        assert np.array_equal(rl.schema_masks(x, **skw).cpu().numpy(), want_schema)
        tseen = torch.zeros((12, states.shape[-1]), dtype=torch.int64, device=dev)
        for t in range(16):
            r, _ = rl.novelty_update(tseen, s[t])
            assert np.array_equal(r.cpu().numpy(), want_r[t]), (dev, t)
        h = rl.her_relabel(s, torch.from_numpy(done).to(dev), k=2, subset=2, seed=4)
        assert np.array_equal(h.goal.cpu().numpy().view(np.uint64), want_h.goal)
        for a, b in zip(h[1:], want_h[1:]):
            assert np.array_equal(a.cpu().numpy(), b), dev
