"""mymyr.cuda: device task upload, smoke kernels vs the CPU engine, DLPack for device memory with stream semantics
(torch CUDA and JAX GPU), and host operations reading pinned / managed CUDA memory in place. See also test_cuda_brfs.py.

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX would take 75% of the GPU otherwise

import random  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import mymyr  # noqa: E402
from mymyr import _core, rl  # noqa: E402
from conftest import SMALL_TASKS, text_task, walk  # noqa: E402

mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
if not mc.available():
    pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)", allow_module_level=True)

SPIN = 200_000_000  # clock cycles (~0.1 s on an L4): long enough that an unsynchronized consumer would overtake


@pytest.fixture(scope="module")
def ctx():
    c = mc.Context(0, max_bytes=2 << 30)
    yield c
    c.synchronize()


def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    return torch


def jax_gpu():
    jax = pytest.importorskip("jax")
    try:
        return jax, jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("JAX has no CUDA backend here (JAX_PLATFORMS must include cuda)")


def words_of(states, W):
    out = np.zeros((len(states), W), dtype=np.uint64)
    for i, s in enumerate(states):
        out[i, : s.num_words] = s.words
    return out


def reference(task, seed=0):
    """Walk states, labels (applicable ones plus perturbed ones) and the CPU's answers, computed before the upload so
    that lazily interned atoms are in the export."""
    rng = random.Random(seed)
    states = walk(task, steps=15, seed=seed, walks=2)
    L = max(1, task.label_width)
    idx, schema, binding, applicable, succ = [], [], [], [], []
    for i, s in enumerate(states):
        acts = task.applicable_actions(s)[:24]
        labels = [(a.schema, list(a.binding)) for a in acts]
        # perturbed labels: another state's labels and random bindings (some inapplicable)
        other = states[rng.randrange(len(states))]
        labels += [(a.schema, list(a.binding)) for a in task.applicable_actions(other)[:4]]
        for _ in range(3):
            sc = rng.randrange(task.num_schemas)
            ar = task.formalism.schemas[sc].arity
            labels.append((sc, [rng.randrange(task.num_objects) for _ in range(ar)]))
        for sc, b in labels:
            ok = task.is_applicable(s, (sc, b))
            idx.append(i)
            schema.append(sc)
            binding.append(b + [0] * (L - len(b)))
            applicable.append(ok)
            succ.append(task.apply(s, (sc, b)) if ok else None)
    goal = [task.is_goal(s) for s in states]
    W = max([task.words] + [s.num_words for s in states] + [t.num_words for t in succ if t is not None])
    return dict(
        states=words_of(states, W),
        idx=np.array(idx, np.int32),
        schema=np.array(schema, np.int32),
        binding=np.array(binding, np.int32).reshape(len(idx), L),
        applicable=np.array(applicable),
        succ=succ,
        goal=np.array(goal),
        W=W,
    )


# ------------------------------------------------------------------------------------------------ upload and kernels


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_upload_round_trip_is_byte_identical(ctx, name):
    task = text_task(name)
    dt = mc.DeviceTask(task, ctx)
    assert dt.version == 2 and dt.bytes == dt.host_block().nbytes
    assert np.array_equal(dt.download(), dt.host_block())
    assert dt.validate() == (0, 0)
    assert dt.upload_seconds > 0 and dt.upload_device_ms >= 0


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
def test_smoke_kernels_agree_with_the_cpu(ctx, name, atoms):
    torch = torch_cuda()
    task = text_task(name, atoms=atoms)
    r = reference(task)
    dt = mc.DeviceTask(task, ctx)
    states = torch.from_numpy(r["states"].view(np.int64)).cuda()
    derived = torch.from_numpy(dt.host_derived(r["states"]).view(np.int64)).cuda() if task.has_axioms else None
    idx, schema, binding = (torch.from_numpy(r[k]).cuda() for k in ("idx", "schema", "binding"))

    ok = dt.applicable(states, idx, schema, binding, derived=derived)
    assert ok.device.type == "cuda" and ok.dtype == torch.bool
    assert np.array_equal(ok.cpu().numpy(), r["applicable"])

    succ, status = dt.apply(states, idx, schema, binding, derived=derived)
    succ, status = succ.cpu().numpy().view(np.uint64), status.cpu().numpy()
    has_ce = dt.arrays()["plan_has_conditional_effects"]
    checked = 0
    for i, t in enumerate(r["succ"]):
        if t is None:
            assert status[i] == -1
        elif status[i] == 1:  # conditional effects: the CPU applies them
            assert has_ce
        else:
            assert status[i] == 0
            expect = np.zeros(succ.shape[1], np.uint64)
            expect[: t.num_words] = t.words
            assert np.array_equal(succ[i], expect)
            checked += 1
    assert checked > 0

    goal = dt.goal(states, derived=derived)
    assert np.array_equal(goal.cpu().numpy(), r["goal"])


# ------------------------------------------------------------------------------------------------ DLPack, torch


def test_torch_zero_copy_both_directions(ctx, blocks):
    torch = torch_cuda()
    dt = mc.DeviceTask(blocks, ctx)
    # mymyr -> torch: the uploaded arrays are views of the device block
    host = blocks.device_arrays(framework="numpy")
    raw = dt.arrays()
    tarr = dt.arrays("torch")
    for k, v in host.items():
        if isinstance(v, int):
            assert tarr[k] == v
            continue
        t = tarr[k]
        assert t.device == torch.device("cuda", 0)
        if t.numel():  # torch reports 0 for empty tensors
            assert t.data_ptr() == raw[k].data_ptr
        assert dt.data_ptr <= raw[k].data_ptr and raw[k].data_ptr + raw[k].nbytes <= dt.data_ptr + dt.bytes
        assert np.array_equal(t.cpu().numpy().view(np.asarray(v).dtype), np.asarray(v)), k
    y = mc._impl._delayed_iota(1000, 5, 0, ctx)
    ty = torch.from_dlpack(y)
    assert ty.data_ptr() == y.data_ptr and ty.device.type == "cuda"
    # torch -> mymyr: read in place, row slices included
    r = reference(blocks)
    states = torch.from_numpy(r["states"].view(np.int64)).cuda()
    for view in (states, states[3:9], states[::2]):
        info = mc._impl._import_info(view, ctx)
        assert info[0] == view.data_ptr() and info[1] == view.shape[0] and info[4] == "torch"
        assert info[3] == view.stride(0)
    # outputs follow the input's framework, or the one named
    g = dt.goal(states)
    assert isinstance(g, torch.Tensor)
    gd = dt.goal(states, framework="dlpack")
    assert isinstance(gd, mymyr.DLArray) and gd.__dlpack_device__() == (2, 0)
    assert np.array_equal(torch.from_dlpack(gd).cpu().numpy(), r["goal"])
    ctx.synchronize()
    assert ctx.pending_imports == 0


def test_torch_stream_semantics(ctx):
    torch = torch_cuda()
    n = 1 << 16
    expect = n * 7 + n * (n - 1) // 2
    ts = torch.cuda.Stream()
    # mymyr produces on its stream after a long spin; torch consumes on its own stream: __dlpack__(stream=ts) orders
    # the consumer after the producer
    y = mc._impl._delayed_iota(n, 7, SPIN, ctx)
    with torch.cuda.stream(ts):
        t = torch.from_dlpack(y)
        s = t.sum()
    ts.synchronize()
    assert int(s) == expect
    # the same on torch's default stream, and on a stream passed to mymyr
    y = mc._impl._delayed_iota(n, 7, SPIN, ctx, stream=ts)
    assert int(torch.from_dlpack(y).sum()) == expect
    torch.cuda.synchronize()
    # control: without the handoff (stream=-1: no synchronization) the consumer overtakes the producer
    y = mc._impl._delayed_iota(n, 99, SPIN, ctx)
    with torch.cuda.stream(ts):
        t = torch.utils.dlpack.from_dlpack(y.__dlpack__(stream=-1))
        s = t.sum()
    ts.synchronize()
    assert int(s) != n * 99 + n * (n - 1) // 2
    ctx.synchronize()
    del t, y

    # torch produces on its stream (a spin, then the write); mymyr consumes on the arena's stream
    arena = mc.StateArena(ctx, words=4, capacity=8)
    with torch.cuda.stream(ts):
        x = torch.zeros((1000, 4), dtype=torch.int64, device="cuda")
        torch.cuda._sleep(SPIN)
        x.fill_(11)
        arena.append(x)
    arena.sync()
    arena.wait()
    assert arena.host_size == 1000 and arena.capacity >= 1000
    assert (arena.host_view() == 11).all()

    # lifetime: torch still reads (after a spin) when the producer drops its array; the memory must not be reused
    # before that read (the consumer's stream was recorded at the handoff)
    y = mc._impl._delayed_iota(n, 3, 0, ctx)
    with torch.cuda.stream(ts):
        t = torch.from_dlpack(y)
        torch.cuda._sleep(SPIN)
        s = t.sum()
    del t, y
    junk = [mc._impl._delayed_iota(n, 1 << 40, 0, ctx) for _ in range(4)]  # would reuse the freed block at once
    ts.synchronize()
    assert int(s) == n * 3 + n * (n - 1) // 2
    del junk

    # stream values
    y = mc._impl._delayed_iota(4, 0, 0, ctx)
    with pytest.raises(BufferError):
        y.__dlpack__(stream=0)
    with pytest.raises(TypeError):
        y.__dlpack__(stream=ts)
    for s in (None, 1, 2, -1, ts.cuda_stream):
        assert "dltensor" in repr(y.__dlpack__(stream=s))
    torch.cuda.synchronize()
    ctx.synchronize()


def test_task_block_outlives_its_owner_while_torch_reads(ctx, blocks):
    torch = torch_cuda()
    dt = mc.DeviceTask(blocks, ctx)
    ts = torch.cuda.Stream()
    expect = blocks.device_arrays(framework="numpy")["plan_matcher"].astype(np.int64).sum()
    with torch.cuda.stream(ts):
        m = dt.arrays("torch")["plan_matcher"]
        torch.cuda._sleep(SPIN)
        s = m.to(torch.int64).sum()
    del m, dt
    ts.synchronize()
    assert int(s) == expect


# ------------------------------------------------------------------------------------------------ host operations


@pytest.mark.parametrize("memory", ["pinned", "managed"])
def test_expand_reads_pinned_and_managed_memory_in_place(ctx, blocks, memory):
    r = reference(blocks)
    x = mc.to_memory(r["states"], memory, ctx)
    assert x.__dlpack_device__() == ({"pinned": 3, "managed": 13}[memory], 0)
    info = _core._import_info(x)
    assert info[0] == x.data_ptr and info[1:3] == (r["states"].shape[0], r["states"].shape[1])
    got = rl.expand(blocks, x)
    want = rl.expand(blocks, r["states"])
    assert np.array_equal(np.from_dlpack(got.succ), want.succ)
    assert np.array_equal(np.from_dlpack(got.offsets), want.offsets)
    assert np.array_equal(np.from_dlpack(rl.is_goal(blocks, x)), rl.is_goal(blocks, r["states"]))


def test_torch_pinned_tensors_are_read_in_place(blocks):
    torch = torch_cuda()
    r = reference(blocks)
    t = torch.from_numpy(r["states"].view(np.int64)).pin_memory()
    assert _core._import_info(t)[0] == t.data_ptr()
    assert torch.equal(rl.expand(blocks, t).succ, torch.from_numpy(rl.expand(blocks, r["states"]).succ.view(np.int64)))


def test_host_operations_reject_device_memory_clearly(ctx, blocks):
    # rl.expand / rl.expand_into take device arrays (test_cuda_brfs.py); the other host operations do not
    r = reference(blocks)
    x = mc.to_memory(r["states"], "device", ctx)
    with pytest.raises(TypeError, match="runs on the CPU"):
        rl.is_goal(blocks, x)
    torch = torch_cuda()
    with pytest.raises(TypeError, match="runs on the CPU"):
        rl.is_goal(blocks, torch.from_numpy(r["states"].view(np.int64)).cuda())
    with pytest.raises(TypeError, match="device destinations need device states"):
        rl.expand_into(blocks, r["states"], succ=torch.zeros((64, r["W"]), dtype=torch.int64, device="cuda"))
    with pytest.raises(TypeError, match="NumPy"):
        mc.to_memory(r["states"], "device", ctx, framework="numpy")


# ------------------------------------------------------------------------------------------------ JAX


def test_jax_gpu_zero_copy_and_kernels(ctx):
    jax, gpu = jax_gpu()
    import jax.numpy as jnp

    task = text_task("gripper__prob05", atoms="frozen")
    r = reference(task)
    dt = mc.DeviceTask(task, ctx)
    # mymyr -> JAX: word arrays as uint32 pairs, no copy
    raw, jarr = dt.arrays(), dt.arrays("jax")
    for k in ("init", "plan_matcher", "plan_schema"):
        assert list(jarr[k].devices())[0] == gpu and jarr[k].unsafe_buffer_pointer() == raw[k].data_ptr
    assert jarr["init"].dtype == jnp.uint32
    # 64-bit tables (not state words) arrive as uint32 pairs too: JAX without x64 would truncate them
    host = task.device_arrays(framework="numpy")
    for k in ("plan_rs", "plan_static_rel", "plan_row"):
        assert jarr[k].dtype == jnp.uint32 and np.array_equal(np.asarray(jarr[k]).view(np.uint64), host[k]), k
    # JAX -> mymyr: read in place; outputs are JAX arrays
    states = jax.device_put(jnp.asarray(r["states"].view(np.uint32)), gpu)
    info = mc._impl._import_info(states, ctx)
    assert info[0] == states.unsafe_buffer_pointer() and info[4] == "jax" and info[2] == r["W"]
    idx, schema, binding = (jax.device_put(jnp.asarray(r[k]), gpu) for k in ("idx", "schema", "binding"))
    ok = dt.applicable(states, idx, schema, binding)
    assert list(ok.devices())[0] == gpu and np.array_equal(np.asarray(ok), r["applicable"])
    succ, status = dt.apply(states, idx, schema, binding)
    assert succ.dtype == jnp.uint32 and succ.shape[1] % 2 == 0
    succ = np.asarray(succ).view(np.uint64)
    for i, t in enumerate(r["succ"]):
        if t is not None:
            expect = np.zeros(succ.shape[1], np.uint64)
            expect[: t.num_words] = t.words
            assert np.array_equal(succ[i], expect)
    assert np.array_equal(np.asarray(dt.goal(states)), r["goal"])
    # a StateArena slice as a JAX array
    arena = mc.StateArena(ctx, words=r["W"])
    arena.append(states)
    view = arena.device_view(2, 5, framework="jax")
    assert np.array_equal(np.asarray(view).view(np.uint64), r["states"][2:5])
    ctx.synchronize()


def test_jax_pinned_host_arrays_are_read_by_expand(ctx, blocks):
    jax, gpu = jax_gpu()
    import jax.numpy as jnp

    r = reference(blocks)
    pinned = jax.sharding.SingleDeviceSharding(gpu, memory_kind="pinned_host")
    x = jax.device_put(jnp.asarray(r["states"].view(np.uint32)), pinned)
    assert x.__dlpack_device__()[0] == 3
    assert _core._import_info(x)[0] == x.unsafe_buffer_pointer()
    got = rl.expand(blocks, x, framework="numpy")
    assert np.array_equal(got.succ, rl.expand(blocks, r["states"]).succ)
