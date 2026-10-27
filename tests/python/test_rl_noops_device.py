"""Device successor identities, stable no-op filtering and captured destination-passing expansion."""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, text_task, walk
from test_rl_noops import DOMAINS, pddl_task

torch = pytest.importorskip("torch")
mc = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
if not torch.cuda.is_available() or not mc.available():
    pytest.skip("requires a visible CUDA device", allow_module_level=True)


def array(x):
    result = x.cpu().numpy()
    return result.view(np.uint64) if result.dtype == np.int64 else result


def compare(table, rows, ids=None, *, derived=True):
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    device_ids = None if ids is None else torch.as_tensor(ids, device="cuda")
    for drop in (False, True):
        opts = dict(hashes=True, noops=True, drop_noops=drop, derived=derived, goal=True)
        want = rl.expand(table, rows, ids, **opts)
        got = rl.expand(table, source, device_ids, **opts)
        assert got.generated == want.generated and got.total == want.total and got.overflow == want.overflow
        for name in ("succ", "parent", "schema", "binding", "goal", "offsets", "succ_hash", "noop", "derived", "index"):
            expected = getattr(want, name)
            actual = getattr(got, name)
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(array(actual), expected, err_msg=name)
        np.testing.assert_array_equal(array(mymyr.hash_rows(got.succ, numeric_words=want.numeric_words,
                                                           stream=torch.cuda.current_stream())), want.succ_hash)
        if want.total:
            pad, ref = got.pad(), want.pad()
            for name in ("succ", "schema", "binding", "goal", "mask", "count", "index", "succ_hash", "noop", "canonical"):
                if getattr(ref, name) is None:
                    assert getattr(pad, name) is None, name
                else:
                    np.testing.assert_array_equal(array(getattr(pad, name)), getattr(ref, name), err_msg=name)
        cap = want.generated + 3
        hashes = torch.full((cap,), -1, dtype=torch.int64, device="cuda")
        noops = torch.ones(cap, dtype=torch.bool, device="cuda")
        indices = torch.full((cap,), -2, dtype=torch.int32, device="cuda")
        offsets = torch.empty(len(rows) + 1, dtype=torch.int32, device="cuda")
        status = torch.empty(1, dtype=torch.int32, device="cuda")
        result = rl.expand_into(table, source, device_ids, succ_hash=hashes, noop=noops, index=indices,
                               offsets=offsets, status=status, drop_noops=drop, stream=torch.cuda.current_stream())
        assert int(status.item()) == 0
        np.testing.assert_array_equal(array(offsets), want.offsets)
        np.testing.assert_array_equal(array(hashes[:want.total]), want.succ_hash)
        np.testing.assert_array_equal(array(noops[:want.total]), want.noop)
        full_indices = np.concatenate([np.arange(c, dtype=np.int32) for c in want.counts]) if len(rows) else []
        np.testing.assert_array_equal(array(indices[:want.total]), want.index if drop else full_indices)
        if "total" not in result:  # the asynchronous form clears the rows past the total
            assert not hashes[want.total:].any() and not noops[want.total:].any()
            assert torch.all(indices[want.total:] == -1)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("multi", [False, True])
def test_ipc_host_device_digests(domain, multi):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/ipc" / f"{domain}-ipc" / "train"
    first = mymyr.Task.from_pddl(base / "domain.pddl", base / "p01.pddl", atoms="dense")
    if multi:
        second = mymyr.Task.from_pddl(base / "domain.pddl", base / "p02.pddl", atoms="dense")
        table = rl.TaskTable([first, second])
        states = [walk(t, steps=4, walks=1) for t in (first, second)]
        rows = np.zeros((sum(map(len, states)), table.words), np.uint64)
        ids = np.concatenate([np.full(len(s), i, np.int32) for i, s in enumerate(states)])
        offset = 0
        for task, ss in zip((first, second), states):
            rows[offset:offset + len(ss), :task.words] = task.encode(ss)
            offset += len(ss)
        compare(table, rows, ids)
    else:
        compare(first, np.asarray(first.encode(walk(first, steps=6, walks=1))))


@pytest.mark.parametrize("name", ["miconic-simpleadl__s10-2", "openstacks-opt08-adl__p03", "philosophers__p03-phil4"])
def test_conditional_and_axiom_digests(name):
    task = text_task(name, atoms="dense")
    compare(task, np.asarray(task.encode(walk(task, steps=4, walks=1))))


@pytest.mark.parametrize("storage", ["i32", "f64"])
def test_numeric_device_digests(storage):
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"), numeric_storage=storage)
    compare(task, np.asarray(task.encode(walk(task, steps=5, walks=1))))


def test_suite_digests_and_global_order():
    a, b = text_task("gripper__prob05", atoms="dense"), text_task("blocks__probBLOCKS-8-0", atoms="dense")
    suite = rl.TaskSuite([rl.as_table(a), rl.as_table(b)])
    states = [walk(t, steps=5, walks=1) for t in (a, b)]
    rows = np.zeros((2 * min(map(len, states)), suite.words), np.uint64)
    ids = np.arange(len(rows), dtype=np.int32) % 2
    for i, task in enumerate((a, b)):
        rows[i::2, :task.words] = task.encode(states[i][:len(rows) // 2])
    compare(suite, rows, ids)


@pytest.mark.parametrize("objects", [7, 15, 31, 63, 64, 91])
def test_wide_label_only_digests_and_narrow_destinations(objects):
    n = objects
    task = pddl_task("""(define (domain bits) (:requirements :strips :conditional-effects)
      (:predicates (p ?x ?y) (q))
      (:action idle :parameters (?x ?y) :precondition (and) :effect (and))
      (:action set :parameters (?x ?y) :precondition (and) :effect (p ?x ?y))
      (:action conditional :parameters (?x ?y) :precondition (and) :effect (when (q) (p ?x ?y)))
      (:action clear :parameters () :precondition (q) :effect (not (q))))""",
      """(define (problem bits) (:domain bits) (:objects """ + " ".join(f"o{i}" for i in range(n)) +
      ") (:init (q)) (:goal (p o0 o0)))", atoms="dense")
    rows = np.vstack((task.encode([task.initial_state]), np.zeros((1, task.words), np.uint64)))
    compare(task, rows, derived=False)
    narrow = rl.expand(task, torch.as_tensor(rows.view(np.int64), device="cuda"), words=1, hashes=True, noops=True)
    full = rl.expand(task, rows, hashes=True, noops=True)
    np.testing.assert_array_equal(array(narrow.succ_hash), full.succ_hash)
    np.testing.assert_array_equal(array(narrow.noop), full.noop)


def test_fallback_schema_digests():
    task = pddl_task("""(define (domain many) (:requirements :strips)
      (:predicates (p)) (:action idle :parameters (?p0 ?p1 ?p2 ?p3 ?p4 ?p5 ?p6 ?p7 ?p8 ?p9 ?p10 ?p11 ?p12 ?p13 ?p14 ?p15 ?p16 ?p17 ?p18 ?p19 ?p20 ?p21 ?p22 ?p23 ?p24 ?p25 ?p26 ?p27 ?p28 ?p29 ?p30 ?p31 ?p32)
      :precondition (and) :effect (and))
      (:action set :parameters () :precondition (and) :effect (p)))""",
      """(define (problem many) (:domain many) (:objects x) (:init) (:goal (p)))""", atoms="dense")
    # the idle action has more parameters than the device kernels take: its successors are built on the host
    compare(task, np.asarray(task.encode([task.initial_state])))


@pytest.mark.parametrize("multi", [False, True])
def test_capture_replay_drops_noops_and_preserves_canonical_positions(multi):
    task = pddl_task(atoms="dense")
    table = rl.TaskTable([task, pddl_task(atoms="dense")]) if multi else task
    rows = np.asarray(task.encode([task.initial_state] * 3))
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    ids = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda") if multi else None
    cap = 16
    opts = dict(device="cuda", dtype=torch.int32)
    out = dict(succ_hash=torch.zeros(cap, dtype=torch.int64, device="cuda"), noop=torch.zeros(cap, dtype=torch.bool, device="cuda"),
               index=torch.full((cap,), -1, **opts), schema=torch.zeros(cap, **opts), parent=torch.zeros(cap, **opts),
               binding=torch.zeros((cap, 1), **opts), offsets=torch.zeros(4, **opts), status=torch.zeros(1, **opts))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def run():
        return rl.expand_into(table, source, ids, **out, drop_noops=True, stream=stream)
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    goal_state = next(state for _, state in task.successors(task.initial_state) if state.is_goal())
    effective = np.asarray(task.encode([goal_state] * 3))
    mixed = np.stack((rows[0], effective[0], np.zeros_like(rows[0])))
    for row in (rows, effective, mixed, np.zeros_like(rows), rows):
        source.copy_(torch.as_tensor(row.view(np.int64), device="cuda"))
        graph.replay()
        expected = rl.expand(table, row, None if ids is None else array(ids), hashes=True, drop_noops=True)
        assert int(out["status"].item()) == 0
        np.testing.assert_array_equal(array(out["offsets"]), expected.offsets)
        for name in ("succ_hash", "noop", "index", "schema", "parent", "binding"):
            np.testing.assert_array_equal(array(out[name][:expected.total]), getattr(expected, name))
        assert not out["succ_hash"][expected.total:].any()
        assert torch.all(out["index"][expected.total:] == -1)
    # Overflow is measured before the no-op drop, even when the kept expansion would fit.
    short = {name: (value[:6] if name not in ("status", "offsets") else value) for name, value in out.items()}
    rl.expand_into(table, source, ids, **short, drop_noops=True, stream=torch.cuda.current_stream())
    assert int(short["status"].item()) & 1
    # the offsets of an overflowing call count the successors before dropping: they size the next capacity
    generated = rl.expand(table, rows, None if ids is None else array(ids), hashes=True, drop_noops=True).generated
    assert int(short["offsets"][-1].item()) == generated > 6
    torch.cuda.synchronize()


@pytest.mark.parametrize("words", [0, 1, 4, 16, 65, 131])
@pytest.mark.parametrize("numeric_words", [0, 2])
def test_device_hash_rows_matches_host_and_captures(words, numeric_words):
    rng = np.random.default_rng(14)
    rows = rng.integers(0, 2**64 - 1, (7, words + numeric_words), dtype=np.uint64)
    if words:
        rows[:, words - 1] = 0
    expected = mymyr.hash_rows(rows, numeric_words=numeric_words)
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mymyr.hash_rows(source, numeric_words=numeric_words, stream=stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = mymyr.hash_rows(source, numeric_words=numeric_words, stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    np.testing.assert_array_equal(array(actual), expected)
    source.zero_()
    graph.replay()
    np.testing.assert_array_equal(array(actual), mymyr.hash_rows(np.zeros_like(rows), numeric_words=numeric_words))
    torch.cuda.synchronize()


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("rows", [False, True])
def test_consolidated_torch_ops_fullgraph_and_capture(padded, rows):
    from mymyr.rl.torch import register
    task = pddl_task(atoms="dense")
    handle = register(task)
    host = np.asarray(task.encode([task.initial_state] * 2))
    source = torch.as_tensor(host.view(np.int64), device="cuda")
    opts = dict(derived=rows, rows=rows, hashes=True, noops=True, drop_noops=True)
    def run(s):
        if padded:
            return torch.ops.mymyr.expand(handle, s, None, 4, capacity=16, **opts)
        return torch.ops.mymyr.expand_flat(handle, s, None, 16, **opts)
    eager = run(source)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    actual = compiled(source)
    assert len(actual) == 11
    for a, b in zip(actual, eager):
        assert torch.equal(a, b)
    assert not int(actual[-1].item())
    expected = rl.expand(task, host, hashes=True, drop_noops=True)
    if padded:
        np.testing.assert_array_equal(array(actual[7]), expected.pad(4).succ_hash)
        np.testing.assert_array_equal(array(actual[9]), expected.pad(4).canonical)
    else:
        np.testing.assert_array_equal(array(actual[7][:expected.total]), expected.succ_hash)
        np.testing.assert_array_equal(array(actual[9][:expected.total]), expected.index)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(source)
    torch.cuda.current_stream().wait_stream(stream)
    source.zero_()
    graph.replay()
    assert not captured[5].any() and not captured[7].any()
    torch.cuda.synchronize()


def test_all_noop_fallback_empty_parents_and_zero_capacity():
    task = pddl_task("""(define (domain idle) (:requirements :strips)
      (:predicates (p) (q)) (:action set :parameters () :precondition (and) :effect (p)))""",
      """(define (problem idle) (:domain idle) (:init) (:goal (q)))""", atoms="dense")
    # from the state with p set, the only successor is that state again
    done = next(state for _, state in task.successors(task.initial_state))
    rows = np.asarray(task.encode([done] * 3))
    compare(task, rows)
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    empty = rl.expand(task, source[:0], hashes=True, drop_noops=True)
    assert empty.total == empty.generated == 0 and empty.offsets.tolist() == [0]
    zero = rl.expand(task, source, capacity=0, hashes=True, drop_noops=True)
    # on a device overflow the offsets are those before dropping, so the total is the generated count
    assert zero.generated == zero.total == 3 and zero.overflow
    kept = rl.expand(task, source, hashes=True, drop_noops=True)
    assert kept.generated == kept.total == 3 and kept.noop.all() and not kept.index.any()


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("hashes,noops", [(True, False), (False, True), (True, True)])
def test_jax_cuda_matches_host(drop, hashes, noops):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("built without the XLA FFI headers")
    if not jax.devices("cuda"):
        pytest.skip("JAX sees no CUDA device")
    from mymyr.rl.jax import Env
    task = pddl_task(atoms="dense")
    rows = np.asarray(task.encode(walk(task, steps=4, walks=1)))
    expected = rl.expand(task, rows, hashes=hashes, noops=noops, drop_noops=drop)
    opts = dict(hashes=hashes, noops=noops, drop_noops=drop)
    env = Env(task, device="cuda")
    source = jax.device_put(rows.view(np.uint32), jax.devices("cuda")[0])
    actual = jax.jit(lambda s: env.expand_flat(s, expected.generated + 3, **opts))(source)
    m = expected.total
    assert int(actual.generated) == expected.generated
    for name in ("parent", "schema", "binding", "offsets"):
        got = np.asarray(getattr(actual, name))
        np.testing.assert_array_equal(got if name == "offsets" else got[:m], getattr(expected, name), err_msg=name)
    np.testing.assert_array_equal(np.asarray(actual.succ)[:m].copy().view(np.uint64), expected.succ)
    if hashes:
        np.testing.assert_array_equal(np.asarray(actual.succ_hash)[:m].copy().view(np.uint64).ravel(), expected.succ_hash)
        assert not np.asarray(actual.succ_hash)[m:].any()
    if noops or drop:
        np.testing.assert_array_equal(np.asarray(actual.noop)[:m], expected.noop)
        assert not np.asarray(actual.noop)[m:].any()
    if drop:
        np.testing.assert_array_equal(np.asarray(actual.index)[:m], expected.index)
        assert np.all(np.asarray(actual.index)[m:] == -1)
    padded = jax.jit(lambda s: env.expand(s, 5, **opts))(source)
    ref = expected.pad(5)
    assert int(padded.generated) == expected.generated
    for name in ("succ", "schema", "binding", "mask", "count", "noop", "canonical", "succ_hash"):
        if getattr(padded, name) is None:
            assert name in ("noop", "canonical", "succ_hash")
            continue
        got, want = np.asarray(getattr(padded, name)), getattr(ref, name)
        if name in ("succ", "succ_hash"):
            got = got.copy().view(np.uint64)
            if name == "succ_hash":
                got = got[..., 0]
        np.testing.assert_array_equal(got, want, err_msg=name)
    # vmap over a batch of state batches: each batch gets its own pre-drop count and rows
    batches = jnp.stack([source, source[::-1]])
    mapped = jax.jit(jax.vmap(lambda s: env.expand(s, 5, **opts)))(batches)
    flipped = rl.expand(task, np.ascontiguousarray(rows[::-1]), **opts)
    assert np.asarray(mapped.generated).tolist() == [expected.generated, flipped.generated]
    for b, ref_b in enumerate((ref, flipped.pad(5))):
        np.testing.assert_array_equal(np.asarray(mapped.count)[b], ref_b.count)
        np.testing.assert_array_equal(np.asarray(mapped.schema)[b], ref_b.schema)
        if hashes:
            np.testing.assert_array_equal(np.asarray(mapped.succ_hash)[b].copy().view(np.uint64)[..., 0], ref_b.succ_hash)
        if noops or drop:
            np.testing.assert_array_equal(np.asarray(mapped.noop)[b], ref_b.noop)
        if drop:
            np.testing.assert_array_equal(np.asarray(mapped.canonical)[b], ref_b.canonical)
    generic = rl.expand(task, source, **opts)
    if hashes:
        np.testing.assert_array_equal(np.asarray(generic.succ_hash).copy().view(np.uint64).ravel(), expected.succ_hash)
        np.testing.assert_array_equal(np.asarray(mymyr.hash_rows(generic.succ)).copy().view(np.uint64).ravel(),
                                      expected.succ_hash)


def test_device_hashes_key_the_visited_tables():
    from mymyr.rl.torch import BatchedEnv
    task = text_task("gripper__prob05")
    env = BatchedEnv(task, 4, max_steps=20, track_visited=True, autoreset=False)
    for _ in range(3):
        env.step()  # random policy
    host = rl.expand(task, env.states, framework="torch", hashes=True)
    device = rl.expand(task, env.states.cuda(), hashes=True)
    np.testing.assert_array_equal(array(device.succ_hash), array(host.succ_hash))
    want = rl.visited_mask(env.visited, env.visited_count, host.succ_hash, host.parent)
    got = rl.visited_mask(env.visited, env.visited_count, device.succ_hash.cpu(), device.parent.cpu())
    assert torch.equal(got, want) and want.any()  # stepping back reaches a visited state
    # every table holds the device hash of its environment's current state
    reached = mymyr.hash_rows(env.states.cuda(), stream=torch.cuda.current_stream()).cpu()
    for i in range(4):
        assert int(reached[i]) in env.visited[i].tolist()
    env.close()


def test_device_hashes_key_the_jax_visited_tables():
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("built without the XLA FFI headers")
    if not jax.devices("cuda"):
        pytest.skip("JAX sees no CUDA device")
    from mymyr.rl.jax import Env
    task = text_task("gripper__prob05", atoms="dense")
    cpu = jax.devices("cpu")[0]
    visits = Env(task, device="cpu", max_steps=20, track_visited=True)
    state, _ = visits.reset(0, 4)
    step = jax.jit(lambda key, s: visits.step(key, s))  # the random policy
    for key in range(3):
        state, *_ = step(key, state)
    gpu = Env(task, device="cuda")
    states = jax.device_put(np.asarray(state.states), jax.devices("cuda")[0])
    flat = gpu.expand_flat(states, 256, hashes=True)
    host = visits.expand_flat(state.states, 256, hashes=True)
    np.testing.assert_array_equal(np.asarray(flat.succ_hash), np.asarray(host.succ_hash))
    on_cpu = [jax.device_put(np.asarray(x), cpu) for x in (flat.succ_hash, flat.parent)]
    got = rl.visited_mask(state.visited, state.visited_count, *on_cpu)
    want = rl.visited_mask(state.visited, state.visited_count, host.succ_hash, host.parent)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert np.asarray(want).any()


def test_numeric_noop_uses_public_float_bits():
    task = pddl_task("""(define (domain values) (:requirements :strips :fluents)
      (:predicates (p)) (:functions (x))
      (:action keep :parameters () :precondition (and) :effect (assign (x) (x)))
      (:action zero :parameters () :precondition (and) :effect (assign (x) 0)))""",
      """(define (problem values) (:domain values) (:init (= (x) 0)) (:goal (and)))""",
      atoms="dense", numeric_storage="f64")
    rows = np.tile(task.encode([task.initial_state]), (3, 1))
    rows[:, -1] = np.array([0, 0x8000000000000000, 0x7ff8000000001234], np.uint64)
    compare(task, rows)


def test_captured_noop_filtering_respects_live_parent_count():
    task = pddl_task(atoms="dense")
    rows = np.asarray(task.encode([task.initial_state] * 3))
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    kw = dict(device="cuda", dtype=torch.int32)
    live = torch.tensor([3], **kw)
    out = dict(succ_hash=torch.empty(16, dtype=torch.int64, device="cuda"),
               noop=torch.empty(16, dtype=torch.bool, device="cuda"), index=torch.empty(16, **kw),
               offsets=torch.empty(4, **kw), status=torch.empty(1, **kw))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def run():
        rl.expand_into(task, source, **out, row_count=live, drop_noops=True, stream=stream)
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    for count in (3, 1, 0, 3):
        live.fill_(count)
        graph.replay()
        expected = rl.expand(task, rows[:count], hashes=True, drop_noops=True)
        offsets = np.pad(expected.offsets, (0, 3 - count), constant_values=expected.total)
        assert not int(out["status"].item())
        np.testing.assert_array_equal(array(out["offsets"]), offsets)
        np.testing.assert_array_equal(array(out["succ_hash"][:expected.total]), expected.succ_hash)
        np.testing.assert_array_equal(array(out["index"][:expected.total]), expected.index)
        assert not out["succ_hash"][expected.total:].any() and torch.all(out["index"][expected.total:] == -1)
    torch.cuda.synchronize()


@pytest.mark.parametrize("drop", [False, True])
def test_bounded_observation_actions_digest_without_successor_rows(drop):
    from mymyr import encoding
    from mymyr.encoding._device_actions import bounded_actions
    task = pddl_task(atoms="dense")
    encoder = encoding.FlatRelationEncoder(rl.TaskTable([task]))  # one table across the captured calls
    rows = np.asarray(task.encode([task.initial_state] * 2))
    source = torch.as_tensor(rows.view(np.int64), device="cuda")
    expected = rl.expand(task, rows, hashes=True, noops=True, drop_noops=drop)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bounded_actions(encoder, source, None, 16, hashes=True, noops=True, drop_noops=drop)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = bounded_actions(encoder, source, None, 16, hashes=True, noops=True, drop_noops=drop)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert actual.succ is None and not actual.status.any()
    np.testing.assert_array_equal(array(actual.succ_hash[:expected.total]), expected.succ_hash)
    np.testing.assert_array_equal(array(actual.noop[:expected.total]), expected.noop)
    if drop:
        np.testing.assert_array_equal(array(actual.index[:expected.total]), expected.index)
    source.zero_()
    graph.replay()
    assert not actual.offsets.any() and not actual.succ_hash.any()
    torch.cuda.synchronize()
