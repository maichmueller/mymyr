"""Zero-copy interop with torch and JAX on CPU: inputs are read in place through DLPack,
outputs come back in the caller's framework as views of one mymyr-owned block, exported through our own __dlpack__.

These tests skip without torch / JAX; the RL environment (.venv-rl: torch 2.14, jax 0.11.2) runs them."""

import gc

import numpy as np
import pytest

from mymyr import _core, rl
from conftest import text_task, walk


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def jax():
    """JAX with its CPU device as the default device (the interop under test is the host path; a GPU-visible run,
    JAX_PLATFORMS=cuda,cpu, keeps these arrays on the CPU)."""
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        yield jax


@pytest.fixture(scope="module")
def batch():
    task = text_task("logistics00__probLOGISTICS-6-1", atoms="lazy")
    states = walk(task, steps=12, seed=3, walks=3)
    ref = rl.expand(task, states, goal=True)
    return task, states, task.encode(states), ref


def same_expansion(exp, ref, to_numpy):
    assert exp.total == ref.total and not exp.overflow
    succ = to_numpy(exp.succ)
    assert np.array_equal(succ.view(np.uint64), ref.succ)
    for name in ("parent", "schema", "binding", "offsets", "goal"):
        assert np.array_equal(to_numpy(getattr(exp, name)), getattr(ref, name)), name


# ------------------------------------------------------------------------------------------------ torch


def test_torch_inputs_are_read_in_place(torch, batch):
    task, states, u64, ref = batch
    t = torch.from_numpy(u64.view(np.int64))  # torch words are int64 [N, W]
    ptr, rows, words, stride, fw, bits, signed = _core._import_info(t)
    assert (ptr, rows, words, stride, fw, bits, signed) == (t.data_ptr(), len(states), u64.shape[1], u64.shape[1], "torch", 64, True)
    wide = torch.zeros((len(states), 4 * u64.shape[1]), dtype=torch.int64)
    wide[:, : u64.shape[1]] = t
    strided = wide[::3, : u64.shape[1]]
    info = _core._import_info(strided)
    assert info[0] == wide.data_ptr() and info[3] == 3 * wide.shape[1]
    exp = rl.expand(task, strided, goal=True)
    same_expansion(exp, rl.expand(task, u64[::3].copy(), goal=True), lambda x: x.numpy())
    with pytest.raises(TypeError):
        rl.expand(task, wide[:, ::2])  # the words of a row are not contiguous
    with pytest.raises(TypeError):
        rl.expand(task, t.to(torch.float64))


def test_torch_outputs(torch, batch):
    task, states, u64, ref = batch
    exp = rl.expand(task, torch.from_numpy(u64.view(np.int64)), goal=True)
    assert exp.framework == "torch" and isinstance(exp.succ, torch.Tensor)
    assert (exp.succ.dtype, exp.parent.dtype, exp.schema.dtype, exp.binding.dtype, exp.offsets.dtype, exp.goal.dtype) == (
        torch.int64, torch.int32, torch.int32, torch.int32, torch.int32, torch.bool)
    same_expansion(exp, ref, lambda x: x.numpy())
    a, b = exp.succ, exp.succ
    assert a.data_ptr() == b.data_ptr() and a.data_ptr() % 64 == 0  # views of the block, no copy per access
    p = exp.pad(0)
    assert isinstance(p.succ, torch.Tensor) and p.succ.shape[:2] == (len(states), p.K) and p.mask.dtype == torch.bool
    assert torch.equal(p.count, torch.diff(exp.offsets))
    # NumPy in, torch out, and the other way around
    t = rl.expand(task, u64, framework="torch", goal=True)
    same_expansion(t, ref, lambda x: x.numpy())
    n = rl.expand(task, torch.from_numpy(u64.view(np.int64)), framework="numpy")
    assert n.succ.dtype == np.uint64 and np.array_equal(n.succ, ref.succ)
    # the block outlives the Expansion
    succ = rl.expand(task, states, framework="torch").succ
    gc.collect()
    assert np.array_equal(succ.numpy().view(np.uint64), ref.succ)


def test_torch_task_arrays_and_goals(torch, batch):
    task, states, u64, _ = batch
    words = task.encode(states, framework="torch")
    assert words.dtype == torch.int64 and task.decode(words) == states
    meta = task.atom_metadata(framework="torch")
    ref = task.atom_metadata()
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            assert isinstance(meta[k], torch.Tensor) and np.array_equal(meta[k].numpy(), v), k
        else:
            assert meta[k] == v
    d = task.device_arrays(framework="torch")
    assert d["init"].dtype == torch.int64 and task.state(d["init"]) == task.initial_state
    gpos, gneg = task.goal_masks(framework="torch")
    flags = torch.tensor([s.is_goal() for s in states])
    assert torch.equal(rl.goal_test(words, gpos, gneg), flags)
    assert torch.equal(rl.is_goal(task, words), flags)
    assert torch.equal(rl.goal_count(words, gpos, gneg) == 0, flags)
    envs = rl.goal_masks(task, torch.zeros(len(states), dtype=torch.int32))  # per-env rows, the table's width
    assert envs[0].shape == (len(states), task.max_words) and torch.equal(envs[0][:, : gpos.shape[0]], gpos.expand(len(states), -1))
    envs[0][1:] = 0
    envs[1][1:] = 0
    assert rl.goal_test(words, *envs)[1:].all()


def test_torch_destinations(torch, batch):
    task, states, u64, ref = batch
    M, W, L, N = ref.total, ref.words, ref.label_width, len(states)
    out = {
        "succ": torch.zeros((M, W), dtype=torch.int64),
        "parent": torch.zeros(M, dtype=torch.int32),
        "schema": torch.zeros(M, dtype=torch.int32),
        "binding": torch.zeros((M, L), dtype=torch.int32),
        "goal": torch.zeros(M, dtype=torch.bool),
        "offsets": torch.zeros(N + 1, dtype=torch.int32),
    }
    ptrs = {k: v.data_ptr() for k, v in out.items()}
    r = rl.expand_into(task, torch.from_numpy(u64.view(np.int64)), **out)
    assert r["total"] == M and not r["overflow"]
    assert {k: v.data_ptr() for k, v in out.items()} == ptrs
    for k, v in out.items():
        got = v.numpy()
        assert np.array_equal(got.view(np.uint64) if k == "succ" else got, getattr(ref, k)), k


def test_our_dlpack_into_torch(torch, blocks):
    exp = rl.expand(blocks, walk(blocks, steps=6), framework="dlpack")
    x = exp.succ
    t = torch.from_dlpack(x)
    assert t.data_ptr() == x.data_ptr and t.dtype == torch.uint64 and tuple(t.shape) == x.shape
    meta = blocks.atom_metadata(framework="dlpack")
    args = meta["atom_args"]  # inside a bundle: the offset is folded into the data pointer
    ta = torch.from_dlpack(args)
    assert ta.data_ptr() == args.data_ptr and ta.tolist() == blocks.atom_metadata()["atom_args"].tolist()


# ------------------------------------------------------------------------------------------------ JAX


def test_jax_inputs_are_read_in_place(jax, batch):
    import jax.numpy as jnp

    task, states, u64, ref = batch
    j = jnp.asarray(u64.view(np.uint32))  # JAX words are uint32 [N, 2W] (no x64)
    assert j.dtype == jnp.uint32 and j.shape == (len(states), 2 * u64.shape[1])
    ptr, rows, words, stride, fw, bits, signed = _core._import_info(j)
    assert (ptr, rows, words, stride, fw, bits, signed) == (j.unsafe_buffer_pointer(), len(states), u64.shape[1],
                                                            u64.shape[1], "jax", 32, False)
    exp = rl.expand(task, j, goal=True)
    assert exp.framework == "jax" and exp.succ.dtype == jnp.uint32 and exp.succ.shape == (ref.total, 2 * ref.words)
    assert (exp.parent.dtype, exp.binding.dtype, exp.goal.dtype) == (jnp.int32, jnp.int32, jnp.bool_)
    same_expansion(exp, ref, np.asarray)
    a, b = exp.succ, exp.succ
    assert a.unsafe_buffer_pointer() == b.unsafe_buffer_pointer() and a.unsafe_buffer_pointer() % 64 == 0
    p = exp.pad(0)
    assert p.succ.shape == (len(states), p.K, 2 * ref.words) and np.array_equal(np.asarray(p.count), np.diff(ref.offsets))
    # JAX in, NumPy / torch-free defaults out; NumPy in, JAX out
    n = rl.expand(task, j, framework="numpy")
    assert n.succ.dtype == np.uint64 and np.array_equal(n.succ, ref.succ)
    same_expansion(rl.expand(task, u64, framework="jax", goal=True), ref, np.asarray)


def test_jax_task_arrays_and_goals(jax, batch):
    import jax.numpy as jnp

    task, states, _, _ = batch
    words = task.encode(states, framework="jax")
    assert words.dtype == jnp.uint32 and task.decode(words) == states
    d = task.device_arrays(framework="jax")
    ref = task.device_arrays()
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            got = np.asarray(d[k])
            pairs = v.dtype == np.uint64 and got.dtype == np.uint32  # 64-bit arrays as uint32 pairs
            assert np.array_equal(got.view(np.uint64) if pairs else got, v), k
        else:
            assert d[k] == v, k
    assert task.state(d["init"]) == task.initial_state
    gpos, gneg = task.goal_masks(framework="jax")
    assert gpos.dtype == jnp.uint32
    flags = np.array([s.is_goal() for s in states])
    assert np.array_equal(np.asarray(rl.goal_test(words, gpos, gneg)), flags)
    assert np.array_equal(np.asarray(rl.is_goal(task, words)), flags)
    envs = rl.goal_masks(task, jnp.zeros(len(states), jnp.int32))  # per-env rows, the table's width (uint32 halves)
    assert envs[0].shape == (len(states), 2 * task.max_words) and envs[0].dtype == jnp.uint32
    # jitted code consumes the arrays as they come (a mask goal test in JAX agrees with ours)
    jit_test = jax.jit(lambda s, p, n: jnp.all((s & p) == p, axis=-1) & jnp.all((s & n) == 0, axis=-1))
    assert np.array_equal(np.asarray(jit_test(words, gpos, gneg)), flags)


def test_jax_arrays_are_not_destinations(jax, batch):
    import jax.numpy as jnp

    task, states, _, ref = batch
    with pytest.raises(TypeError, match="immutable"):
        rl.expand_into(task, states, succ=jnp.zeros((ref.total, 2 * ref.words), jnp.uint32))


def test_our_dlpack_into_jax(jax, blocks):
    import jax.dlpack

    exp = rl.expand(blocks, np.asarray(blocks.encode(walk(blocks, steps=6))).view(np.uint32), framework="dlpack")
    x = exp.succ
    assert x.dtype == "uint32"
    j = jax.dlpack.from_dlpack(x)
    assert j.unsafe_buffer_pointer() == x.data_ptr and j.shape == x.shape
    meta = blocks.atom_metadata(framework="dlpack")
    args = meta["atom_args"]
    ja = jax.dlpack.from_dlpack(args)
    assert ja.unsafe_buffer_pointer() == args.data_ptr and np.asarray(ja).tolist() == blocks.atom_metadata()["atom_args"].tolist()


def test_torch_and_jax_words_are_the_same_bytes(torch, jax, batch):
    import jax.numpy as jnp

    task, states, u64, ref = batch
    t = rl.expand(task, states, framework="torch").succ
    j = rl.expand(task, states, framework="jax").succ
    assert np.array_equal(t.numpy().view(np.uint32), np.asarray(j))
    back = rl.expand(task, jnp.asarray(t.numpy().view(np.uint32)))
    assert np.array_equal(np.asarray(back.succ).view(np.uint64), rl.expand(task, ref.succ).succ)
