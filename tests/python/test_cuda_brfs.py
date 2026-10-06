"""mymyr.cuda: the device BrFS against the CPU BrFS (counts, deterministic ids, plans), and rl.expand /
rl.expand_into on device inputs (torch CUDA, JAX GPU, DLArray) byte-equal to the CPU expansion, with the DLPack stream
semantics (inputs imported on the operation's stream, outputs handed off to the consumer's).

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_brfs.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX would take 75% of the GPU otherwise

import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import mymyr  # noqa: E402
import mymyr.search  # noqa: E402
from mymyr import rl  # noqa: E402
from conftest import SMALL_TASKS, text_task, walk  # noqa: E402

mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
if not mc.available():
    pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)", allow_module_level=True)

SPIN = 200_000_000  # clock cycles (~0.1 s on an L4): long enough that an unsynchronized consumer would overtake


@pytest.fixture(scope="module")
def ctx():
    c = mc.Context(0, max_bytes=4 << 30)
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


def host(x):
    """A device output as a NumPy array (state words as uint64)."""
    torch = pytest.importorskip("torch")
    if isinstance(x, np.ndarray):
        return x
    t = x if isinstance(x, torch.Tensor) else torch.from_dlpack(x)
    a = t.cpu().numpy()
    return a.view(np.uint64) if a.dtype == np.int64 else a


def assert_same_expansion(dev, cpu):
    """Equal outputs. The automatic capacity (the rows allocated) may differ: the CPU guesses it from a running
    branching estimate and reruns on overflow, the device counts first and allocates exactly the total."""
    assert (dev.total, dev.words, dev.words_needed, dev.label_width, len(dev)) == (
        cpu.total,
        cpu.words,
        cpu.words_needed,
        cpu.label_width,
        len(cpu),
    )
    for k in ("succ", "parent", "schema", "binding", "offsets", "goal"):
        a, b = getattr(dev, k), getattr(cpu, k)
        if b is None:
            assert a is None, k
            continue
        assert np.array_equal(host(a), b), k
    assert dev.overflow == cpu.overflow


def assert_same_padded(dev, cpu):
    assert dev.K == cpu.K and dev.overflow == cpu.overflow
    for k in ("index", "mask", "count", "succ", "schema", "binding", "goal"):
        a, b = getattr(dev, k), getattr(cpu, k)
        if b is None:
            assert a is None, k
            continue
        assert np.array_equal(host(a), b), k


# ------------------------------------------------------------------------------------------------ device BrFS


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
def test_device_brfs_ids_equal_the_cpu(ctx, name, atoms):
    task = text_task(name, atoms=atoms)
    r = mc.brfs(task, ctx=ctx, fingerprint=True)
    c = mymyr.search.brfs(task, threads=2, fingerprint=True)
    assert (r.states, r.expanded, r.generated, r.goal_states, r.layers, r.exhausted) == (
        c.states,
        c.expanded,
        c.generated,
        c.goal_states,
        c.layers,
        c.exhausted,
    )
    assert r.fingerprint == c.fingerprint and r.fingerprint != 0
    assert r.status == c.status and r.store == "device"
    # the chunk size changes nothing
    small = mc.brfs(text_task(name, atoms=atoms), ctx=ctx, fingerprint=True, chunk_states=97)
    assert (small.states, small.fingerprint) == (r.states, r.fingerprint)
    assert small.stats["chunks"] > r.stats["chunks"]
    # small layers run in device loops (frozen slots); timings=True drives every chunk from the host: the same ids
    if atoms == "frozen":
        assert r.stats["loops"] > 0 and r.stats["captures"] > 0
    timed = mc.brfs(text_task(name, atoms=atoms), ctx=ctx, fingerprint=True, timings=True)
    assert (timed.states, timed.layers, timed.fingerprint) == (r.states, r.layers, r.fingerprint)
    assert timed.stats["loops"] == 0


def test_device_brfs_state_space_and_plans(ctx):
    torch = torch_cuda()
    task = text_task("gripper__prob05", atoms="frozen")
    r = mc.brfs(task, ctx=ctx)
    words = r.state_words("torch")
    nodes = r.nodes("torch")
    assert words.device.type == "cuda" and tuple(words.shape) == (r.states, r.words)
    assert tuple(nodes.shape) == (r.states, 2) and int(nodes[0, 0]) == -1
    s0 = task.initial_state
    assert np.array_equal(host(words[0])[: s0.num_words], np.asarray(s0.words, np.uint64))
    # a node's parent comes first, and replaying its plan reaches its row
    i = r.states - 1
    parent = int(nodes[i, 0])
    assert 0 <= parent < i
    plan = r.plan_to(i)
    s = s0
    for a in plan:
        s = task.apply(s, a)
    assert np.array_equal(host(words[i])[: s.num_words], np.asarray(s.words, np.uint64))
    del words, nodes
    torch.cuda.synchronize()
    # stop_at_goal: the CPU's plan
    for name in ("depot__p02", "philosophers__p03-phil4", "miconic-simpleadl__s10-2"):
        task = text_task(name)
        g = mc.brfs(task, ctx=ctx, stop_at_goal=True)
        c = mymyr.search.brfs(task, stop_at_goal=True)
        assert g.solved and c.solved
        assert [a.label for a in g.plan] == [a.label for a in c.plan]
        assert g.states == c.states


# ------------------------------------------------------------------------------------------------ rl.expand on device


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
def test_expand_torch_cuda_is_byte_equal_to_the_cpu(ctx, name, atoms):
    torch = torch_cuda()
    task = text_task(name, atoms=atoms)
    states = walk(task, steps=12, seed=3, walks=3)
    W = max([task.words] + [s.num_words for s in states])
    x = words_of(states, W)
    t = torch.from_numpy(x.view(np.int64)).cuda()
    for kw in (
        dict(),
        dict(goal=True, K=0),
        dict(canonical=False),
        dict(witness=True, K=4),
        dict(capacity=17, goal=True),
        dict(words=1),
    ):
        dev = rl.expand(task, t, ctx=ctx, **kw)  # first: under lazy slots the device interns the new atoms
        cpu = rl.expand(task, x, **kw)
        if "capacity" in kw:
            assert dev.capacity == cpu.capacity == kw["capacity"]
        assert isinstance(dev, mc.DeviceExpansion) and dev.framework == "torch"
        assert dev.succ.device.type == "cuda" and dev.succ.dtype == torch.int64
        assert_same_expansion(dev, cpu)
        if "K" in kw:
            assert_same_padded(dev.padded, cpu.padded)
        assert_same_padded(dev.pad(3), cpu.pad(3))
        assert np.array_equal(host(dev.counts), cpu.counts)
        if cpu.total:
            j = min(cpu.total, cpu.capacity) - 1
            assert dev.action(j) == cpu.action(j)
            assert dev.actions(0) == cpu.actions(0)
    # row slices are read in place
    dev = rl.expand(task, t[1::2], ctx=ctx)
    assert_same_expansion(dev, rl.expand(task, x[1::2]))


def test_expand_into_torch_destinations(ctx):
    torch = torch_cuda()
    for name in ("gripper__prob05", "miconic-simpleadl__s10-2", "philosophers__p03-phil4"):
        task = text_task(name)
        states = walk(task, steps=10, seed=5)
        W = task.words
        x = words_of(states, W)
        t = torch.from_numpy(x.view(np.int64)).cuda()
        L = task.label_width
        cap = 64
        want = dict(
            succ=np.zeros((cap, W), np.uint64),
            parent=np.zeros(cap, np.int32),
            schema=np.zeros(cap, np.int32),
            binding=np.zeros((cap, L), np.int32),
            goal=np.zeros(cap, np.bool_),
            offsets=np.zeros(len(states) + 1, np.int32),
        )
        got = {
            k: torch.zeros(v.shape, dtype=torch.int64 if v.dtype == np.uint64 else getattr(torch, str(v.dtype).replace("bool_", "bool")), device="cuda")
            for k, v in want.items()
        }
        r_cpu = rl.expand_into(task, x, **want)
        r_dev = rl.expand_into(task, t, ctx=ctx, **got)
        assert r_dev == r_cpu
        for k, v in want.items():
            assert np.array_equal(host(got[k]), v), (name, k)
        # 32-bit successor words ([cap, 2W] int32) are the same bytes
        s32 = torch.zeros((cap, 2 * W), dtype=torch.int32, device="cuda")
        rl.expand_into(task, t, succ=s32, ctx=ctx)
        assert np.array_equal(s32.cpu().numpy().view(np.uint64), want["succ"])
    # destinations must be device arrays like the states
    with pytest.raises(TypeError, match="must be a CUDA device array"):
        rl.expand_into(task, t, succ=np.zeros((cap, W), np.uint64), ctx=ctx)
    with pytest.raises(TypeError, match="C-contiguous"):
        rl.expand_into(task, t, parent=torch.zeros((cap, 2), dtype=torch.int32, device="cuda")[:, 0], ctx=ctx)


def test_expand_validates_like_the_cpu(ctx):
    torch = torch_cuda()
    task = text_task("gripper__prob05", atoms="frozen")
    x = words_of([task.initial_state] * 2, task.words + 1)
    x[1, -1] = 1  # an unassigned slot
    with pytest.raises(ValueError, match="state row 1"):
        rl.expand(task, x)
    with pytest.raises(ValueError, match="state row 1"):
        rl.expand(task, torch.from_numpy(x.view(np.int64)).cuda(), ctx=ctx)
    assert rl.expand(task, torch.from_numpy(x.view(np.int64)).cuda(), ctx=ctx, validate=False).total > 0


def test_expand_stream_semantics(ctx):
    torch = torch_cuda()
    task = text_task("blocks__probBLOCKS-8-0", atoms="frozen")
    states = walk(task, steps=30, seed=1, walks=4)
    x = words_of(states, task.words)
    cpu = rl.expand(task, x)
    ts = torch.cuda.Stream()
    # producer: torch writes the states on its current stream after a spin; mymyr imports them with
    # __dlpack__(stream=<the context's stream>), which makes that stream wait for torch's
    src = torch.from_numpy(x.view(np.int64)).cuda()
    torch.cuda.synchronize()
    with torch.cuda.stream(ts):
        t = torch.zeros_like(src)
        torch.cuda._sleep(SPIN)
        t.copy_(src)
        dev = rl.expand(task, t, ctx=ctx)
    assert_same_expansion(dev, cpu)
    # stream=: the expansion runs on the given stream and its outputs are produced there
    with torch.cuda.stream(ts):
        t2 = torch.zeros_like(src)
        torch.cuda._sleep(SPIN)
        t2.copy_(src)
    dev = rl.expand(task, t2, ctx=ctx, stream=ts)
    assert dev.stream == ts.cuda_stream
    assert_same_expansion(dev, cpu)
    # the same from torch's default stream into a context stream
    t3 = torch.zeros_like(src)
    torch.cuda._sleep(SPIN)
    t3.copy_(src)
    assert_same_expansion(rl.expand(task, t3, ctx=ctx), cpu)
    # consumer: torch takes the outputs on another stream (the tensors are made at the property access, on the
    # current stream) and reads them after a spin while the expansion is dropped; the memory is not reused before
    # that read (the consumer's stream was recorded at the handoff)
    tc = torch.cuda.Stream()
    dev = rl.expand(task, src, ctx=ctx)
    with torch.cuda.stream(tc):
        succ = dev.succ
        torch.cuda._sleep(SPIN)
        total = succ.sum()
    del dev, succ
    junk = [rl.expand(task, src, ctx=ctx) for _ in range(4)]  # would reuse the freed block at once
    tc.synchronize()
    assert int(total) == int(cpu.succ.view(np.int64).sum())
    del junk
    torch.cuda.synchronize()
    ctx.synchronize()


def test_expand_dlarray_and_jax_inputs(ctx):
    task = text_task("logistics00__probLOGISTICS-6-1", atoms="frozen")
    states = walk(task, steps=15, seed=2)
    x = words_of(states, task.words)
    cpu = rl.expand(task, x, K=0)
    # a mymyr DLArray on the device: outputs are DLArrays unless a framework is named
    d = mc.to_memory(x, "device", ctx)
    dev = rl.expand(task, d, K=0)
    assert isinstance(dev.succ, mymyr.DLArray) and dev.succ.__dlpack_device__() == (2, 0)
    assert_same_expansion(dev, cpu)
    assert_same_padded(dev.padded, cpu.padded)
    # JAX GPU arrays: uint32 words in, JAX arrays out
    jax, gpu = jax_gpu()
    import jax.numpy as jnp

    j = jax.device_put(jnp.asarray(x.view(np.uint32)), gpu)
    dev = rl.expand(task, j, ctx=ctx, goal=True)
    assert list(dev.succ.devices())[0] == gpu and dev.succ.dtype == jnp.uint32
    want = rl.expand(task, x, goal=True)
    assert np.array_equal(np.asarray(dev.succ).view(np.uint64), want.succ)
    for k in ("parent", "schema", "binding", "offsets", "goal"):
        assert np.array_equal(np.asarray(getattr(dev, k)), getattr(want, k)), k
    ctx.synchronize()


def test_expand_from_many_threads(ctx):
    torch = torch_cuda()
    task = text_task("gripper__prob05", atoms="lazy")
    states = walk(task, steps=20, seed=9, walks=3)
    x = words_of(states, task.words)
    t = torch.from_numpy(x.view(np.int64)).cuda()
    want = rl.expand(task, x)
    errors = []

    def work():
        try:
            s = torch.cuda.Stream()
            for _ in range(5):
                dev = rl.expand(task, t, stream=s)  # the task's default context
                assert_same_expansion(dev, want)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    torch.cuda.synchronize()
