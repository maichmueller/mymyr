"""mymyr.cuda: the device BrFS against the CPU BrFS (counts, deterministic ids, plans), and rl.expand /
rl.expand_into on device inputs (torch CUDA, JAX GPU, DLArray) byte-equal to the CPU expansion, with the DLPack stream
semantics (inputs imported on the operation's stream, outputs handed off to the consumer's).

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_brfs.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX would take 75% of the GPU otherwise

import re  # noqa: E402
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
    assert dev.generated == cpu.generated
    assert dev.overflow == cpu.overflow


def assert_same_padded(dev, cpu):
    assert dev.k == cpu.k and dev.overflow == cpu.overflow
    for k in ("index", "mask", "count", "succ", "schema", "binding", "goal"):
        a, b = getattr(dev, k), getattr(cpu, k)
        if b is None:
            assert a is None, k
            continue
        assert np.array_equal(host(a), b), k


# ------------------------------------------------------------------------------------------------ device BrFS


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["dense", "reachable", "lazy"])
def test_device_brfs_ids_equal_the_cpu(ctx, name, atoms):
    task = text_task(name, atoms=atoms)
    r = mc.brfs(task, ctx=ctx, fingerprint=True, stop_at_goal=False)
    c = mymyr.search.brfs(task, threads=2, fingerprint=True, stop_at_goal=False)
    assert (r.stats.states, r.stats.expanded, r.stats.generated, r.num_goal_states, r.layers, r.exhausted) == (
        c.stats.states,
        c.stats.expanded,
        c.stats.generated,
        c.num_goal_states,
        c.layers,
        c.exhausted,
    )
    assert r.fingerprint == c.fingerprint and r.fingerprint != 0
    assert r.status == c.status and r.store == "device"
    # a whole-space search that expanded a goal state is solved, with the plan to the first one
    assert r.solved == c.solved and [a.label for a in r.plan] == [a.label for a in c.plan]
    # the chunk size changes nothing
    small = mc.brfs(text_task(name, atoms=atoms), ctx=ctx, fingerprint=True, chunk_states=97, stop_at_goal=False)
    assert (small.stats.states, small.fingerprint) == (r.stats.states, r.fingerprint)
    assert small.counters["chunks"] > r.counters["chunks"]
    # small layers run in device loops (fixed slots); timings=True drives every chunk from the host: the same ids
    if atoms != "lazy":
        assert r.counters["loops"] > 0 and r.counters["captures"] > 0
    timed = mc.brfs(text_task(name, atoms=atoms), ctx=ctx, fingerprint=True, timings=True, stop_at_goal=False)
    assert (timed.stats.states, timed.layers, timed.fingerprint) == (r.stats.states, r.layers, r.fingerprint)
    assert timed.counters["loops"] == 0


def test_device_brfs_state_space_and_plans(ctx):
    torch = torch_cuda()
    task = text_task("gripper__prob05", atoms="dense")
    r = mc.brfs(task, ctx=ctx)
    words = r.state_words("torch")
    nodes = r.nodes("torch")
    assert words.device.type == "cuda" and tuple(words.shape) == (r.stats.states, r.words)
    assert tuple(nodes.shape) == (r.stats.states, 2) and int(nodes[0, 0]) == -1
    s0 = task.initial_state
    assert np.array_equal(host(words[0])[: s0.num_words], np.asarray(s0.words, np.uint64))
    # a node's parent comes first, and replaying its plan reaches its row
    i = r.stats.states - 1
    parent = int(nodes[i, 0])
    assert 0 <= parent < i
    plan = r.plan_to(i)
    s = s0
    for a in plan:
        s = task.apply(s, a)
    assert np.array_equal(host(words[i])[: s.num_words], np.asarray(s.words, np.uint64))
    del words, nodes
    torch.cuda.synchronize()
    # stopping at the first goal: the CPU's plan
    for name in ("depot__p02", "philosophers__p03-phil4", "miconic-simpleadl__s10-2"):
        task = text_task(name)
        g = mc.brfs(task, ctx=ctx)
        c = mymyr.search.brfs(task)
        assert g.solved and c.solved
        assert [a.label for a in g.plan] == [a.label for a in c.plan]
        assert g.stats.states == c.stats.states


# ------------------------------------------------------------------------------------------------ rl.expand on device


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["dense", "reachable"])
def test_expand_torch_cuda_is_byte_equal_to_the_cpu(ctx, name, atoms):
    torch = torch_cuda()
    task = text_task(name, atoms=atoms)
    states = walk(task, steps=12, seed=3, walks=3)
    W = max([task.words] + [s.num_words for s in states])
    x = words_of(states, W)
    t = torch.from_numpy(x.view(np.int64)).cuda()
    for kw in (
        dict(),
        dict(goal=True, k=8),
        dict(canonical_order=False),
        dict(witness_pruning=True, k=4),
        dict(capacity=17, goal=True),
        dict(words=1),
    ):
        dev = rl.expand(task, t, ctx=ctx, **kw)
        cpu = rl.expand(task, x, **kw)
        if "capacity" in kw:
            assert dev.capacity == cpu.capacity == kw["capacity"]
        assert isinstance(dev, mc.DeviceExpansion) and dev.framework == "torch"
        assert dev.succ.device.type == "cuda" and dev.succ.dtype == torch.int64
        assert_same_expansion(dev, cpu)
        if "k" in kw:
            assert_same_padded(dev.padded, cpu.padded)
        assert_same_padded(dev.pad(3), cpu.pad(3))
        assert_same_padded(dev.pad(), cpu.pad())
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
    task = text_task("gripper__prob05", atoms="dense")
    assert task.num_atoms % 64
    x = words_of([task.initial_state] * 2, task.words + 1)
    x[1, task.num_atoms // 64] |= np.uint64(1) << np.uint64(task.num_atoms % 64)  # an unassigned slot
    with pytest.raises(ValueError, match="state row 1"):
        rl.expand(task, x)
    with pytest.raises(ValueError, match="state row 1"):
        rl.expand(task, torch.from_numpy(x.view(np.int64)).cuda(), ctx=ctx)
    assert rl.expand(task, torch.from_numpy(x.view(np.int64)).cuda(), ctx=ctx, validate=False).total > 0
    # padding: the CPU does not read it, the device rejects a row whose padding is not zero
    y = words_of([task.initial_state] * 2, task.words + 1)
    y[1, -1] = 1
    assert rl.expand(task, y).total == rl.expand(task, y[:, :-1]).total
    with pytest.raises(ValueError, match="state row 1"):
        rl.expand(task, torch.from_numpy(y.view(np.int64)).cuda(), ctx=ctx)


def test_expand_stream_semantics(ctx):
    torch = torch_cuda()
    task = text_task("blocks__probBLOCKS-8-0", atoms="dense")
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
    task = text_task("logistics00__probLOGISTICS-6-1", atoms="dense")
    states = walk(task, steps=15, seed=2)
    x = words_of(states, task.words)
    cpu = rl.expand(task, x, k=16)
    # a mymyr DLArray on the device: outputs are DLArrays unless a framework is named
    d = mc.to_memory(x, "device", ctx)
    dev = rl.expand(task, d, k=16)
    assert isinstance(dev.succ, mymyr.DLArray) and dev.succ.__dlpack_device__() == (2, 0)
    assert_same_expansion(dev, cpu)
    assert_same_padded(dev.padded, cpu.padded)
    # JAX GPU arrays: uint32 words in, JAX arrays out
    jax, gpu = jax_gpu()
    import jax.numpy as jnp

    j = jax.device_put(jnp.asarray(x.view(np.uint32)), gpu)
    for kw in (dict(), dict(words=1), dict(capacity=3)):
        dev = rl.expand(task, j, ctx=ctx, goal=True, **kw)
        assert list(dev.succ.devices())[0] == gpu and dev.succ.dtype == jnp.uint32
        want = rl.expand(task, x, goal=True, **kw)
        assert np.array_equal(np.asarray(dev.succ).view(np.uint64), want.succ)
        for k in ("parent", "schema", "binding", "offsets", "goal", "counts"):
            assert np.array_equal(np.asarray(getattr(dev, k)), getattr(want, k)), k
        assert dev.generated == want.generated and dev.overflow == want.overflow
        for width in (1, 16, None):
            got, expected = dev.pad(width), want.pad(width)
            assert got.k == expected.k and got.overflow == expected.overflow
            for field in ("index", "mask", "count", "schema", "binding", "goal"):
                assert np.array_equal(np.asarray(getattr(got, field)), getattr(expected, field)), field
            assert np.array_equal(np.asarray(got.succ).view(np.uint64), expected.succ)
    ctx.synchronize()


def test_expand_from_many_threads(ctx):
    torch = torch_cuda()
    task = text_task("gripper__prob05", atoms="reachable")
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


@pytest.mark.parametrize("objects", [65, 80, 129, 192])
def test_wide_brfs_equals_cpu(ctx, tmp_path, objects):
    from conftest import wide_task

    task = wide_task(tmp_path, objects)
    assert task.words > 64
    for stop in (False, True):
        cpu = mymyr.search.brfs(task, stop_at_goal=stop, fingerprint=True, threads=1)
        dev = mc.brfs(task, ctx=ctx, stop_at_goal=stop, fingerprint=True, chunk_states=1)
        assert (dev.stats.states, dev.stats.generated, dev.stats.expanded, dev.fingerprint, dev.num_goal_states) == (
            cpu.stats.states, cpu.stats.generated, cpu.stats.expanded, cpu.fingerprint, cpu.num_goal_states)
        assert dev.plan == cpu.plan and dev.status == cpu.status


def test_ipc_brfs_equals_cpu(ctx, rovers_task):
    task = rovers_task
    cpu = mymyr.search.brfs(task, max_states=32, fingerprint=True, threads=1)
    dev = mc.brfs(task, ctx=ctx, max_states=32, fingerprint=True, chunk_states=1)
    assert (dev.stats.states, dev.stats.generated, dev.stats.expanded, dev.fingerprint) == (
        cpu.stats.states, cpu.stats.generated, cpu.stats.expanded, cpu.fingerprint)
    assert dev.status == cpu.status and dev.plan == cpu.plan


def brfs_matches(dev, cpu):
    assert (dev.stats.states, dev.stats.generated, dev.stats.expanded, dev.fingerprint, dev.num_goal_states) == (
        cpu.stats.states, cpu.stats.generated, cpu.stats.expanded, cpu.fingerprint, cpu.num_goal_states)
    assert dev.status == cpu.status and dev.plan == cpu.plan


@pytest.mark.parametrize("stop", [False, True])
def test_wide_ipc_brfs_equals_cpu(ctx, blocksworld_wide, stop):
    task = blocksworld_wide
    assert task.words > 64
    cpu = mymyr.search.brfs(task, max_states=20000, stop_at_goal=stop, fingerprint=True, threads=1)
    brfs_matches(mc.brfs(task, ctx=ctx, max_states=20000, stop_at_goal=stop, fingerprint=True), cpu)


@pytest.mark.parametrize("kw", [{}, {"stop_at_goal": False}, {"max_states": 100_000}, {"max_states": 1000}])
def test_successor_windows_equal_cpu(tmp_path, kw):
    """A state with more successors than the context has room for at once: its successors a window at a time."""
    from conftest import fan_task

    task = fan_task(tmp_path, 400)
    kw = {"max_states": 160_001} | kw
    small = mc.Context(0, max_bytes=40 << 20)
    cpu = mymyr.search.brfs(task, fingerprint=True, threads=1, **kw)
    dev = mc.brfs(task, ctx=small, fingerprint=True, **kw)
    brfs_matches(dev, cpu)
    assert dev.counters["windows"] >= (2 if kw["max_states"] == 160_001 else 1)
    small.synchronize()


@pytest.mark.parametrize("drop, budget, windows", [(False, 16, 2), (True, 56, 1)])
def test_search_runs_on_past_a_windowed_state(tmp_path, drop, budget, windows):
    """The initial state's 90000 successors (45150 distinct) a window at a time in a small context, and the rest of
    the search after them in the same context (with drop: 180001 states and duplicates across parents)."""
    from conftest import symfan_task

    task = symfan_task(tmp_path, 300, drop)
    cpu = mymyr.search.brfs(task, stop_at_goal=False, fingerprint=True, threads=1)
    small = mc.Context(0, max_bytes=budget << 20)
    dev = mc.brfs(task, ctx=small, stop_at_goal=False, fingerprint=True)
    brfs_matches(dev, cpu)
    assert dev.stats.states == (180_001 if drop else 45_151)
    assert dev.counters["windows"] >= windows
    small.synchronize()


# A refusal the search sized (its memory left too little for a named need), not the last-resort refusal of an
# allocation that failed beyond its sizing ("a device memory allocation of N bytes failed during ...").
SIZED_BRFS_REFUSAL = (r"^mymyr: device BrFS: (?!a device memory allocation )"
                      r".*Raise the Context's max_bytes, or use the CPU BrFS\.$")


def run_budgets(task, budgets, cpu, **kw):
    """The outcome per budget: the device counters of a run equal to the CPU's, or None for a sized refusal. (Not
    monotone in the budget: the pool's free pieces can hold a window's buffers at one budget and not at a larger one.)"""
    outcomes = []
    for budget in budgets:
        small = mc.Context(0, max_bytes=budget)
        try:
            dev = mc.brfs(task, ctx=small, fingerprint=True, **kw)
            brfs_matches(dev, cpu)
            outcomes.append(dev.counters)
        except ValueError as e:
            assert re.match(SIZED_BRFS_REFUSAL, str(e)), str(e)
            outcomes.append(None)
        small.synchronize()
    return outcomes


def test_wide_fan_runs_in_twice_its_successor_bytes(tmp_path):
    """1100 successors of 151 KB rows (159 MiB) in a context of twice that: windows sized by what they allocate. In
    smaller contexts the search gives the CPU's result or refuses (the states after the fan have no successors: chunks
    without candidates need no room for new states)."""
    from conftest import wide_fan_task

    task = wide_fan_task(tmp_path)
    row = (task.words + 63) // 64 * 64 * 8
    cpu = mymyr.search.brfs(task, fingerprint=True, threads=1)
    largest = run_budgets(task, [1100 * row * eighths // 8 for eighths in (10, 13, 16)], cpu)[-1]
    assert largest is not None and largest["windows"] >= 1


def test_brfs_window_reaching_max_states_ends_the_search(tmp_path):
    """The wide fan with max_states=150 in 95-105 MiB: the arenas hold up to max_states, so the window held to their
    few spare rows ends the search with the CPU's result instead of refusing for want of more rows."""
    from conftest import wide_fan_task

    task = wide_fan_task(tmp_path)
    cpu = mymyr.search.brfs(task, max_states=150, fingerprint=True, threads=1)
    for mib in (95, 100, 105):
        small = mc.Context(0, max_bytes=mib << 20)
        brfs_matches(mc.brfs(task, ctx=small, max_states=150, fingerprint=True), cpu)
        small.synchronize()


def test_brfs_refuses_windows_held_to_a_few_spare_rows(tmp_path):
    """symfan(300) without drop just below the memory it needs: the fan's windows shrink to the arenas' few spare rows,
    which cannot grow; the search refuses at once instead of going a few successors per window."""
    import time

    from conftest import symfan_task

    task = symfan_task(tmp_path, 300, drop=False)
    cpu = mymyr.search.brfs(task, stop_at_goal=False, fingerprint=True, threads=1)
    window_refusals = 0
    for kib in (5580, 5730, 5890):
        small = mc.Context(0, max_bytes=kib << 10)
        t = time.perf_counter()
        try:
            brfs_matches(mc.brfs(task, ctx=small, stop_at_goal=False, fingerprint=True), cpu)
        except ValueError as e:
            assert re.match(SIZED_BRFS_REFUSAL, str(e)), str(e)
            window_refusals += "with a successor window's buffers" in str(e)
            assert time.perf_counter() - t < 3.0
        small.synchronize()
    assert window_refusals >= 1


def test_brfs_chunks_hold_the_parents_whose_views_fit(blocksworld_wide):
    """20000 states of 69 blocks in 64 MiB: a chunk holds the parents whose views fit an eighth of what the context can
    still allocate, not the 65536 parents of a large context."""
    cpu = mymyr.search.brfs(blocksworld_wide, fingerprint=True, threads=1, max_states=20000)
    small = mc.Context(0, max_bytes=64 << 20)
    brfs_matches(mc.brfs(blocksworld_wide, ctx=small, fingerprint=True, max_states=20000), cpu)
    small.synchronize()


def test_brfs_arenas_grow_geometrically_near_the_memory_limit():
    """Sokoban p05-medium (a CPU-fallback schema: the states' arena keeps a pinned mirror), 400000 states in 1.1 times
    the device memory they take in a large context: the arenas grow by a factor while it fits, not by each chunk's
    new states, and the mirror at least doubles."""
    from conftest import ROOT

    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    directory = ROOT / "tests/data/ipc/sokoban-ipc/test"
    task = mymyr.Task.from_pddl(directory / "domain.pddl", directory / "p05-medium.pddl")
    big = mc.Context(0)
    free = mc.brfs(task, ctx=big, max_states=400_000, fingerprint=True)
    big.synchronize()
    peak = big.usage()["used_high"]
    del big
    assert free.counters["host_schemas"] >= 1
    small = mc.Context(0, max_bytes=peak * 11 // 10)
    tight = mc.brfs(task, ctx=small, max_states=400_000, fingerprint=True)
    small.synchronize()
    assert (tight.stats.states, tight.fingerprint) == (free.stats.states, free.fingerprint)
    assert tight.counters["chunks"] > 2 * tight.counters["growths"], tight.counters
    assert tight.counters["mirror_growths"] <= 8, tight.counters


def test_brfs_under_tight_budgets_equals_cpu_or_refuses(tmp_path, blocksworld_wide):
    """Every budget gives the CPU's result or a ValueError that names the memory the search sized, never an allocator
    error or a failed allocation beyond that sizing: the setup, the windows of a state's successors, the chunks after
    them and the arenas' growth."""
    from conftest import symfan_task

    symfan = symfan_task(tmp_path, 300)
    cases = [(symfan, {"stop_at_goal": False}), (symfan, {"max_states": 2000}), (blocksworld_wide, {"max_states": 2000})]
    outcomes = []
    for task, kw in cases:
        cpu = mymyr.search.brfs(task, fingerprint=True, threads=1, **kw)
        outcomes += run_budgets(task, [budget << 20 for budget in (1, 4, 8, 16, 32, 48)], cpu, **kw)
    assert None in outcomes and any(x is not None for x in outcomes)

