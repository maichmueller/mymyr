"""The relational encoder on the device: FlatRelationEncoder.encode on rows in CUDA memory equals the host encoder bit for
bit, waits for the device once without capacities and never with them, and torch.ops.mymyr.encode_flat gives the same
arrays under torch.compile and inside a captured CUDA graph.

The tests run only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_encoding_cuda.py
"""

import copy
import gc
import time
import warnings

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

PDDL = ROOT / "tests/data/pddl"
OPTION_SETS = {
    "default": {},
    "goal_targets_virtual": {
        "goal_derivations": ("plain", "satisfied", "unsatisfied"),
        "target_sources": ("goal",),
        "predicate_virtual_nodes": True,
    },
    "plain_goal_no_static": {"goal_derivations": ("plain",), "include_static": False, "nullary": False},
    "no_actions": {"include_actions": False, "entity_arrays": "auto", "goal_derivations": ()},
}


def gpu():
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("mymyr sees no CUDA device")
    return torch.device("cuda", 0)


def to_device(array, dev):
    """A NumPy array as a CUDA tensor (u64 words as int64)."""
    if array.dtype == np.uint64:
        array = array.view(np.int64)
    return torch.from_numpy(np.array(array, order="C")).to(dev)


def pddl_task(domain, problem, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(PDDL / domain / "domain.pddl", PDDL / domain / f"{problem}.pddl", **options)


class Case:
    """Rows of a task or table of instances: the state words, the instance of each row and, with derived atoms, the derived
    words of each row (the successors of the walked states, which rl.expand gives with theirs)."""

    def __init__(self, name, source, tasks, derived=False, steps=8):
        self.name, self.source, self.tasks = name, source, tasks
        table = rl.as_table(source) if isinstance(source, mymyr.Task) else source
        words = table.words
        rows, ids, dwords = [], [], []
        for index, task in enumerate(tasks):
            states = walk(task, steps=steps, walks=3, seed=index)
            raw = np.asarray(task.encode(states))
            padded = np.zeros((len(states), words), np.uint64)
            padded[:, : raw.shape[1]] = raw
            if derived:
                x = rl.expand(table, padded, np.full(len(states), index, np.int32) if len(tasks) > 1 else None, derived=True)
                padded = np.ascontiguousarray(np.asarray(x.succ))
                dwords.append(np.asarray(x.derived))
                ids += [index] * len(padded)
            else:
                ids += [index] * len(states)
            rows.append(padded)
        self.rows = np.ascontiguousarray(np.concatenate(rows))
        self.ids = np.asarray(ids, np.int32) if len(tasks) > 1 else None
        self.derived = np.ascontiguousarray(np.concatenate(dwords)) if derived else None

    @property
    def n(self):
        return len(self.rows)


@pytest.fixture(scope="module")
def cases():
    out = {}
    gripper = text_task("gripper__prob05")
    out["gripper"] = Case("gripper", gripper, [gripper])
    phil = text_task("philosophers__p03-phil4")
    out["philosophers"] = Case("philosophers", phil, [phil], derived=True)
    if hasattr(mymyr, "Domain"):
        blocks = [pddl_task("blocks", p, atoms="dense") for p in ("probBLOCKS-8-0", "smedium")]
        out["blocks_table"] = Case("blocks_table", rl.TaskTable(blocks), blocks)
        phils = [pddl_task("philosophers", p, atoms="dense") for p in ("p03-phil4", "p03-phil4-one-blocked")]
        out["philosophers_table"] = Case("philosophers_table", rl.TaskTable(phils), phils, derived=True)
    return out


def host_encode(encoder, case, goals=None, **kwargs):
    return encoder.encode(case.rows, case.ids, derived=case.derived, goals=goals, **kwargs)


def device_encode(encoder, case, dev, goals=None, **kwargs):
    ids = None if case.ids is None else to_device(case.ids, dev)
    derived = None if case.derived is None else to_device(case.derived, dev)
    if goals is not None:
        goals = tuple(to_device(g, dev) for g in goals)
    return encoder.encode(to_device(case.rows, dev), ids, derived=derived, goals=goals, **kwargs)


def same_batch(host, device):
    """The arrays and scalars of two batches: the same names, dtypes and bytes."""
    h, d = host.as_dict(), device.as_dict("numpy")
    assert h.keys() == d.keys()
    for key in h:
        a, b = np.asarray(h[key]), np.asarray(d[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert np.array_equal(a, b), key
    assert host.num_graphs == device.num_graphs and host.num_relations == device.num_relations
    assert host.implicit_entities == device.implicit_entities


def exact_capacities(batch, factor=1):
    sizes = {
        "entities": int(batch.entity_offsets[-1]),
        "graphs": batch.num_graphs,
        "args": int(batch.relation_arg_offsets[-1]),
        "targets": 0,
    }
    if hasattr(batch, "target_offsets"):
        sizes["targets"] = int(batch.target_offsets[-1])
    return {k: v * factor for k, v in sizes.items()}


def fluent_goals(case, seed=3):
    """Random goal masks over the fluent slots of every row: (goal_pos, goal_neg) [N, W]."""
    rng = np.random.default_rng(seed)
    pos = np.zeros(case.rows.shape, np.uint64)
    neg = np.zeros(case.rows.shape, np.uint64)
    table = rl.as_table(case.source) if isinstance(case.source, mymyr.Task) else case.source
    offsets = np.asarray(table.atom_metadata()["atom_offsets"])
    ids = np.zeros(case.n, np.int64) if case.ids is None else case.ids
    for row in range(case.n):
        slots = int(offsets[ids[row] + 1] - offsets[ids[row]])
        for slot in range(slots):
            draw = rng.random()
            if draw < 0.1:
                pos[row, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
            elif draw < 0.2:
                neg[row, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return pos, neg


# ------------------------------------------------------------------------------------------------ CPU == GPU


@pytest.mark.parametrize("options", OPTION_SETS.values(), ids=OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
def test_device_equals_host_bit_for_bit(cases, layout, index, options, n):
    dev = gpu()
    for case in cases.values():
        if n is not None:
            case = copy.copy(case)
            case.rows = case.rows[:n]
            case.ids = None if case.ids is None else case.ids[:n]
            case.derived = None if case.derived is None else case.derived[:n]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, **options)
        host = host_encode(encoder, case)
        batch = device_encode(encoder, case, dev)
        assert isinstance(batch, encoding.DeviceFlatBatch)
        assert batch.sizes is None and batch.device_sizes is None
        same_batch(host, batch)
        assert all(t.device == dev for key, t in batch.as_dict().items() if hasattr(t, "device"))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("options", ["default", "goal_targets_virtual"])
def test_row_goals_equal_host(cases, layout, options):
    dev = gpu()
    for name in ("gripper", "blocks_table"):
        if name not in cases:
            continue
        case = cases[name]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **OPTION_SETS[options])
        goals = fluent_goals(case)
        same_batch(host_encode(encoder, case, goals), device_encode(encoder, case, dev, goals))
        if case.ids is None:  # a goal row shared by all rows
            shared = tuple(g[:1] for g in goals)
            same_batch(host_encode(encoder, case, shared), device_encode(encoder, case, dev, shared))
        halves = tuple(np.ascontiguousarray(g).view(np.uint32) for g in goals)
        same_batch(host_encode(encoder, case, goals), encoder.encode(
            to_device(case.rows, dev), None if case.ids is None else to_device(case.ids, dev),
            goals=tuple(torch.from_numpy(h.view(np.int32)).to(dev) for h in halves)))


def test_derived_goal_masks_equal_host():
    dev = gpu()
    task = pddl_task("goals", "conjunctive")
    table = rl.as_table(task)
    states = walk(task, steps=6, walks=2)
    rows = np.ascontiguousarray(np.asarray(task.encode(states)))
    x = rl.expand(table, rows, derived=True)
    case = Case.__new__(Case)
    case.source, case.tasks, case.name = task, [task], "goals"
    case.rows, case.ids, case.derived = np.ascontiguousarray(np.asarray(x.succ)), None, np.asarray(x.derived)
    metadata = table.atom_metadata()
    derived_slots = len(metadata["derived_pred"])
    assert derived_slots > 0
    zeros = np.zeros(case.rows.shape, np.uint64)
    dpos = np.zeros((case.n, max(1, (derived_slots + 63) // 64)), np.uint64)
    dpos[:, 0] = 1
    dneg = np.zeros_like(dpos)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied", "unsatisfied"))
    goals = (zeros, zeros, dpos, dneg)
    same_batch(host_encode(encoder, case, goals), device_encode(encoder, case, dev, goals))


def test_the_instance_goals_equal_the_goal_masks_of_the_task(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    masks = rl.goal_masks(case.source)
    same_batch(host_encode(encoder, case, masks), device_encode(encoder, case, dev, masks))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", ["gripper", "blocks_table", "philosophers_table"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
def test_capacity_form_equals_host_exact_short_zero_and_ample(cases, layout, name, n):
    dev = gpu()
    if name not in cases:
        pytest.skip("built without the loki front end")
    case = cases[name]
    if n is not None:
        case = copy.copy(case)
        case.rows = case.rows[:n]
        case.ids = None if case.ids is None else case.ids[:n]
        case.derived = None if case.derived is None else case.derived[:n]
    encoder = encoding.FlatRelationEncoder(
        case.source, layout=layout, goal_derivations=("plain", "satisfied", "unsatisfied"), target_sources=("goal",),
        predicate_virtual_nodes=True,
    )
    exact = exact_capacities(host_encode(encoder, case))
    roomy = exact_capacities(host_encode(encoder, case), 2) if case.n else dict.fromkeys(exact, 10)
    short = dict(exact, args=max(0, exact["args"] - 1))
    for caps in (exact, roomy, short,
                 dict(exact, entities=max(0, exact["entities"] - 1)),
                 {"entities": 0, "graphs": case.rows.shape[0], "args": 0, "targets": 0}):
        host = host_encode(encoder, case, capacities=caps)
        batch = device_encode(encoder, case, dev, capacities=caps)
        same_batch(host, batch)
        got = batch.sizes
        assert got["invalid_input"] is False
        assert got == host.sizes
        assert batch.device_sizes.dtype == torch.int64 and batch.device_sizes.shape == (5,)
        assert batch.relation_args.shape[0] == caps["args"]
    assert host_encode(encoder, case, capacities=short).sizes["overflow"] == (exact["args"] > 0)


# ------------------------------------------------------------------------------------------------ inputs


def test_rows_of_other_producers_and_layouts_equal_host(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    host = host_encode(encoder, case)
    ids = None if case.ids is None else to_device(case.ids, dev)
    rows = to_device(case.rows, dev)

    # 32-bit halves, u64, a row stride wider than the row, a one-dimensional single row
    same_batch(host, encoder.encode(rows.view(torch.int32), ids))
    same_batch(host, encoder.encode(rows.view(torch.uint64), ids))
    padded = torch.zeros(rows.shape[0], rows.shape[1] + 3, dtype=torch.int64, device=dev)
    padded[:, : rows.shape[1]] = rows
    same_batch(host, encoder.encode(padded[:, : rows.shape[1]], ids))
    one = encoder.encode(rows[2], None if ids is None else ids[2:3])
    assert one.num_graphs == 1
    assert np.array_equal(one.relation_args.cpu().numpy(), encoder.encode(
        case.rows[2:3], None if case.ids is None else case.ids[2:3]).relation_args)

    class DLPackOnly:
        def __init__(self, tensor):
            self.t = tensor

        def __dlpack__(self, *args, **kwargs):
            return self.t.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self.t.__dlpack_device__()

    class InterfaceOnly:
        def __init__(self, tensor):
            self.__cuda_array_interface__ = tensor.__cuda_array_interface__
            self.t = tensor

    same_batch(host, encoder.encode(DLPackOnly(rows), None if ids is None else DLPackOnly(ids)))
    same_batch(host, encoder.encode(InterfaceOnly(rows), None if ids is None else InterfaceOnly(ids)))
    batch = encoder.encode(DLPackOnly(rows), None if ids is None else ids)
    assert isinstance(batch.relation_args, torch.Tensor) and batch.relation_args.device == dev


def test_the_arguments_are_checked_before_any_launch(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    ids = None if case.ids is None else to_device(case.ids, dev)
    prefix = "^mymyr: encoding: "
    with pytest.raises(ValueError, match=prefix + "actions are not encoded on the device"):
        encoder.encode(rows, ids, actions=rl.expand(rl.as_table(case.tasks[0]), case.rows[:1]))
    with pytest.raises(ValueError, match=prefix + "pool is a pool of host threads"):
        encoder.encode(rows, ids, pool=rl.ThreadPool(2))
    for framework in ("numpy", "jax", "dlpack"):
        with pytest.raises(ValueError, match=prefix + "device rows are encoded into torch tensors"):
            encoder.encode(rows, ids, framework=framework)
    encoder.encode(rows, ids, framework="torch", threads=4)  # threads have no effect on the device
    with pytest.raises(ValueError, match=prefix + "'task_ids' is on cpu: the rows of one call are all CUDA arrays"):
        encoder.encode(rows, None if ids is None else ids.cpu())
    with pytest.raises(ValueError, match=prefix + "'states' is on cpu: the rows of one call are all CUDA arrays"):
        encoder.encode(case.rows, ids)
    with pytest.raises(ValueError, match=prefix + "'states' must hold 64-bit state words"):
        encoder.encode(rows.to(torch.float64), ids)
    with pytest.raises(ValueError, match=prefix + "'task_ids' must be a contiguous int32 tensor"):
        encoder.encode(rows, to_device(np.zeros(case.n, np.int64), dev))
    with pytest.raises(ValueError, match=prefix + "'task_ids' must be a contiguous int32 tensor"):
        encoder.encode(rows, to_device(np.zeros(case.n + 1, np.int32), dev))
    with pytest.raises(ValueError, match=prefix + "'states' must have contiguous rows"):
        encoder.encode(rows.t().contiguous().t(), ids) if rows.shape[1] > 1 else (_ for _ in ()).throw(
            ValueError("mymyr: encoding: 'states' must have contiguous rows"))
    with pytest.raises(ValueError, match=prefix + "goals must be"):
        encoder.encode(rows, ids, goals=(rows,))
    with pytest.raises(ValueError, match=prefix + "capacities needs the keys"):
        encoder.encode(rows, ids, capacities={"entities": 1})
    with pytest.raises(ValueError, match=prefix + "capacities has the unknown key 'tuple'"):
        encoder.encode(rows, ids, capacities={"entities": 1, "graphs": rows.shape[0], "args": 1, "tuple": 1})
    with pytest.raises(ValueError, match=prefix + r"capacities\['args'\] must be at least 0"):
        encoder.encode(rows, ids, capacities={"entities": 1, "graphs": rows.shape[0], "args": -1})
    with pytest.raises(ValueError, match=prefix + "the schema has targets"):
        encoding.FlatRelationEncoder(case.source, target_sources=("goal",)).encode(
            rows, ids, capacities={"entities": 1, "graphs": rows.shape[0], "args": 1})
    if case.ids is not None:
        with pytest.raises(ValueError, match=prefix + "task_ids are required for a table of several instances"):
            encoder.encode(rows)
    if rows.shape[1] > 1:  # rows narrower than their instance
        with pytest.raises(ValueError, match=prefix + "a row is invalid"):
            encoder.encode(rows[:, :1], ids)
    with pytest.raises(ValueError, match=prefix + "a row is invalid"):  # a task id outside the table
        encoder.encode(torch.cat([rows[:1], rows[:1]]), to_device(np.full(2, 7, np.int32), dev))


def test_derived_words_are_evaluated_where_a_graph_reads_them(cases):
    dev = gpu()
    case = cases["philosophers"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    same_batch(host_encode(encoder, case), encoder.encode(rows))
    with pytest.raises(ValueError, match="^mymyr: encoding: 'derived' has"):
        encoder.encode(rows, derived=to_device(case.derived[:-1], dev))
    same_batch(host_encode(encoder, case), encoder.encode(rows, derived=to_device(case.derived, dev)))
    supplied = np.zeros_like(case.derived)
    same_batch(encoder.encode(case.rows, derived=supplied), encoder.encode(rows, derived=to_device(supplied, dev)))
    # a schema without derived relations or goal derived literals needs none
    plain = encoding.FlatRelationEncoder(case.source, include_derived=False, goal_derivations=())
    plain.encode(rows)


def test_invalid_rows_raise_without_capacities_and_are_flagged_with_them(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else None
    if case is None:
        pytest.skip("built without the loki front end")
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    bad = to_device(case.ids.copy(), dev)
    bad[3] = 5
    with pytest.raises(ValueError, match="^mymyr: encoding: a row is invalid"):
        encoder.encode(rows, bad)
    caps = exact_capacities(host_encode(encoder, case))
    batch = encoder.encode(rows, bad, capacities=caps)
    sizes = batch.sizes
    assert sizes["overflow"] and sizes["invalid_input"] and not sizes["capacity_overflow"]
    assert not batch.relation_args.any() and not batch.relation_offsets.any()
    assert encoder.encode(rows, to_device(case.ids, dev), capacities=caps).sizes["overflow"] is False


def test_capacity_overflow_does_not_set_the_index_overflow_flag(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, index_dtype="int32")
    batch = device_encode(encoder, case, dev, capacities={"entities": 1, "graphs": case.rows.shape[0], "args": 1})
    assert batch.sizes["capacity_overflow"] and not batch.sizes["index_overflow"]


def test_count_index_overflow_raises_before_exact_allocation(cases, monkeypatch):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, index_dtype="int32")
    native = encoding._device._device_encoder(encoder, 0)

    def count(self, stream, rows, allocate):
        return allocate(1 << 31, 0, 0, 0, 2)

    monkeypatch.setattr(type(native), "encode_exact", count)
    with pytest.raises(OverflowError, match="use index_dtype='int64'"):
        device_encode(encoder, case, dev)


def test_count_index_overflow_is_reported_in_capacity_sizes(cases, monkeypatch):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, index_dtype="int32")
    arrays = encoding._device._outputs

    def outputs(torch_, specs, device):
        out = arrays(torch_, specs, device)
        out["sizes"].copy_(torch.tensor([1 << 31, 0, 0, 0, 2], device=device))
        return out

    monkeypatch.setattr(encoding._device, "_outputs", outputs)
    native = encoding._device._device_encoder(encoder, 0)
    monkeypatch.setattr(type(native), "encode", lambda *a: None)
    batch = device_encode(encoder, case, dev, capacities={"graphs": case.n, "entities": 0, "args": 0})
    assert batch.sizes == {"entities": 1 << 31, "tuples": 0, "args": 0, "targets": 0, "overflow": True,
                           "capacity_overflow": False, "index_overflow": True, "invalid_input": False}


# ------------------------------------------------------------------------------------------------ waiting


def test_without_capacities_the_device_is_waited_for_once_and_with_capacities_never(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    ids = None if case.ids is None else to_device(case.ids, dev)
    rows = to_device(case.rows, dev)
    caps = exact_capacities(host_encode(encoder, case))
    encoder.encode(rows, ids)  # the first call reserves
    encoder.encode(rows, ids, capacities=caps)
    state = encoding._device.workspace_state

    before = state(encoder)
    for _ in range(3):
        encoder.encode(rows, ids)
    after = state(encoder)
    assert after["host_waits"] - before["host_waits"] == 3 and after["reservations"] == before["reservations"]

    before = after
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch calls the sync debug mode a prototype
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                encoder.encode(rows, ids, capacities=caps)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    after = state(encoder)
    assert after == before

    # the stream is kept busy: a call with capacities returns while it is, one without waits for it
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)
    done = torch.cuda.Event()
    encoder.encode(rows, ids, capacities=caps)
    done.record()
    assert not done.query()
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)
    start = time.perf_counter()
    encoder.encode(rows, ids)
    waited = time.perf_counter() - start
    assert waited > 0.1
    after = state(encoder)
    assert after["host_waits"] == before["host_waits"] + 1


def test_the_workspace_of_a_stream_is_reused_and_reserved_in_advance(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    stream = torch.cuda.Stream(dev)
    state = encoding._device.workspace_state
    assert state(encoder, stream=stream) == {"graphs": 0, "scratch_bytes": 0, "reservations": 0, "host_waits": 0}
    with torch.cuda.stream(stream):
        encoder.encode(rows[:5])
    first = state(encoder, stream=stream)
    assert first["reservations"] == 1 and first["graphs"] >= 5
    with torch.cuda.stream(stream):
        for n in (1, 5, 3, 7, case.n):
            encoder.encode(rows[:n])
    assert state(encoder, stream=stream)["reservations"] == 1
    encoder.reserve_device(100_000, stream=stream)
    reserved = state(encoder, stream=stream)
    assert reserved["reservations"] == 2 and reserved["graphs"] >= 100_000 and reserved["scratch_bytes"] > first["scratch_bytes"]
    with torch.cuda.stream(stream):
        encoder.encode(rows)
    assert state(encoder, stream=stream)["reservations"] == 2
    other = torch.cuda.Stream(dev)
    assert state(encoder, stream=other)["reservations"] == 0
    encoder.release_device(stream=stream)
    assert state(encoder, stream=stream)["graphs"] == 0
    with torch.cuda.stream(stream):
        same_batch(encoder.encode(case.rows), encoder.encode(rows))
    encoder.release_device()
    assert state(encoder, stream=stream)["graphs"] == 0
    with pytest.raises(ValueError, match="^mymyr: encoding: graphs must be an int"):
        encoder.reserve_device(-1)


@pytest.mark.parametrize("mode", ["exact", "capacities", "op"])
@pytest.mark.parametrize("wrong_stream", [False, True], ids=["current_stream", "default_stream_mutation"])
def test_streams_are_ordered_with_the_inputs_and_outputs(cases, monkeypatch, mode, wrong_stream):
    dev = gpu()
    if wrong_stream and torch.cuda.get_allocator_backend() != "native":
        pytest.skip("the stream mutation requires allocations completed before use on another stream")
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, goal_derivations=())
    first = case.rows[:1]
    successors = np.asarray(rl.expand(case.source, first).succ)
    candidates = np.concatenate((first, successors))
    counts = encoder.encode(candidates).relation_counts
    alternative = next((i for i in range(1, len(candidates))
                        if np.array_equal(counts[0], counts[i]) and
                        np.count_nonzero(candidates[0] ^ candidates[i]) == 1), None)
    assert alternative is not None
    # A single word changes while every cell keeps its count, so count and fill use the same extents in either order.
    before = np.repeat(candidates[:1], case.n, axis=0)
    after = np.repeat(candidates[alternative:alternative + 1], case.n, axis=0)
    host = encoder.encode(after)
    caps = exact_capacities(host)
    handle = rt.register(encoder)
    source = to_device(after, dev)
    rows = to_device(before, dev)
    stream = torch.cuda.Stream(dev)
    held_scratch = []
    allocate = encoding._device._scratch

    def scratch(*args):
        tensor = allocate(*args)
        held_scratch.append(tensor)
        return tensor

    monkeypatch.setattr(encoding._device, "_scratch", scratch)
    warm = []

    def warm_stream():
        warm.append(encoder.encode(rows))
        warm.append(encoder.encode(rows, capacities=caps))
        if mode == "op":
            warm.append(call_op(handle, rows, None, None, [None] * 4, caps))

    try:
        # Warm allocations and operator dispatch on both streams before delaying the write.
        warm_stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            warm_stream()
        torch.cuda.synchronize()
        warm.clear()
        if wrong_stream:
            monkeypatch.setattr(rt._ops, "_stream", lambda *a: 0)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            rows.copy_(source)
            if mode == "op":
                batch = encoding.DeviceFlatBatch.from_arrays(
                    encoder, call_op(handle, rows, None, None, [None] * 4, caps))
            else:
                batch = encoder.encode(rows, capacities=None if mode == "exact" else caps)
        torch.cuda.synchronize()
        if wrong_stream:
            with pytest.raises(AssertionError):
                same_batch(host, batch)
        else:
            same_batch(host, batch)
    finally:
        torch.cuda.synchronize()
        rt.unregister(handle)


# ------------------------------------------------------------------------------------------------ the batch


def test_a_device_batch_has_the_accessors_of_a_host_batch(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, emit_counts=False)
    host = host_encode(encoder, case)
    batch = device_encode(encoder, case, dev)
    assert batch.schema.schema_fingerprint == host.schema.schema_fingerprint
    assert batch.device == dev and batch.num_graphs == case.n
    assert batch.relation_args.dtype == torch.int32 and batch.task_ids.dtype == torch.int32
    with pytest.raises(AttributeError, match="no relation_counts"):
        batch.relation_counts
    with pytest.raises(AttributeError, match="no entity_role|has no"):
        batch.entity_role if batch.implicit_entities else batch.target_entity
    assert set(batch.as_dict()) == set(host.as_dict())
    assert batch.as_dict("torch")["relation_args"] is batch.relation_args
    with pytest.raises(ValueError, match="^mymyr: encoding: a device batch exports to"):
        batch.as_dict("jax")
    assert encoding.DeviceFlatBatch.relation_args.__doc__ == encoding.FlatBatch.relation_args.__doc__


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_names_and_descriptions_of_a_device_batch_equal_the_host_ones(cases, layout):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(
        case.source, layout=layout, target_sources=("goal",), predicate_virtual_nodes=True,
        goal_derivations=("plain", "satisfied"))
    host = host_encode(encoder, case)
    batch = device_encode(encoder, case, dev)
    capacity = device_encode(encoder, case, dev, capacities=exact_capacities(host))
    for g in (0, case.n // 2, case.n - 1):
        assert encoder.entity_names(batch, g) == encoder.entity_names(host, g)
        assert encoder.describe(batch, g) == encoder.describe(host, g)
        assert encoder.describe(capacity, g) == encoder.describe(host, g)
    over = device_encode(encoder, case, dev, capacities={"entities": 0, "graphs": case.rows.shape[0], "args": 0, "targets": 0})
    with pytest.raises(ValueError, match="overflowed its capacities"):
        encoder.describe(over, 0)
    with pytest.raises(IndexError):
        encoder.entity_names(batch, case.n)


def test_the_encoder_without_a_device_encoding_is_refused_clearly():
    dev = gpu()
    task = text_task("gripper__prob05")
    states = walk(task, steps=3, walks=1)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    rows = to_device(np.asarray(task.encode(states)), dev)
    with pytest.raises(ValueError, match="^mymyr: encoding: the device does not encode action targets"):
        encoder.encode(rows)
    with pytest.raises(ValueError, match="^mymyr: encoding: the device does not encode action targets"):
        encoder.reserve_device(8)


# ------------------------------------------------------------------------------------------------ the op

OP_ARGS = 8


def op_inputs(case, dev, goals=None):
    ids = None if case.ids is None else to_device(case.ids, dev)
    derived = None if case.derived is None else to_device(case.derived, dev)
    g = [None] * 4 if goals is None else [to_device(x, dev) for x in goals] + [None] * (4 - len(goals))
    return to_device(case.rows, dev), ids, derived, g


def call_op(handle, rows, ids, derived, g, caps):
    return torch.ops.mymyr.encode_flat(
        handle, rows, ids, derived, g[0], g[1], g[2], g[3], caps["graphs"], caps["entities"], caps["args"], caps["targets"]
    )


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
def test_the_op_equals_encode_with_capacities(cases, layout, n):
    dev = gpu()
    for name, options in (("blocks_table", "goal_targets_virtual"), ("philosophers_table", "default"),
                          ("gripper", "no_actions")):
        if name not in cases:
            continue
        case = cases[name]
        if n is not None:
            case = copy.copy(case)
            case.rows = case.rows[:n]
            case.ids = None if case.ids is None else case.ids[:n]
            case.derived = None if case.derived is None else case.derived[:n]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **OPTION_SETS[options])
        handle = rt.register(encoder)
        caps = exact_capacities(host_encode(encoder, case), 2)
        if not case.n:
            caps = dict.fromkeys(caps, 10)
        rows, ids, derived, g = op_inputs(case, dev)
        out = call_op(handle, rows, ids, derived, g, caps)
        assert len(out) == len(encoding.DeviceFlatBatch.array_names)
        batch = encoding.DeviceFlatBatch.from_arrays(encoder, out)
        same_batch(host_encode(encoder, case, capacities=caps), batch)
        assert batch.sizes["overflow"] is False
        rt.unregister(handle)


def test_the_op_takes_row_goals_and_derived_words(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    handle = rt.register(encoder)
    goals = fluent_goals(case)
    caps = exact_capacities(host_encode(encoder, case, goals), 2)
    out = call_op(handle, *op_inputs(case, dev, goals), caps)
    same_batch(host_encode(encoder, case, goals, capacities=caps), encoding.DeviceFlatBatch.from_arrays(encoder, out))
    with pytest.raises(ValueError, match="goals are"):
        torch.ops.mymyr.encode_flat(handle, to_device(case.rows, dev), None, None, to_device(goals[0], dev), None, None,
                                    None, 1, 1, 1, 1)
    rt.unregister(handle)


def test_the_fake_implementation_gives_the_shapes_of_the_capacities(cases):
    dev = gpu()
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    handle = rt.register(encoder)
    caps = {"graphs": len(case.rows), "entities": 70, "args": 500, "targets": 11}
    rows, ids, derived, g = op_inputs(case, dev)
    real = call_op(handle, rows, ids, derived, g, caps)
    with FakeTensorMode() as mode:
        fake = call_op(handle, mode.from_tensor(rows), None, None, [None] * 4, caps)
    assert len(fake) == len(real)
    for f, r in zip(fake, real, strict=True):
        assert f.shape == r.shape and f.dtype == r.dtype and f.device == r.device
    rt.unregister(handle)


@pytest.mark.needs_gil
def test_the_op_compiles_without_graph_breaks(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",), goal_derivations=("plain", "satisfied"))
    handle = rt.register(encoder)
    caps = exact_capacities(host_encode(encoder, case), 2)
    rows, ids, derived, g = op_inputs(case, dev)

    def fn(rows, ids):
        out = torch.ops.mymyr.encode_flat(
            handle, rows, ids, None, None, None, None, None, caps["graphs"], caps["entities"], caps["args"], caps["targets"]
        )
        return out[8] + 1, out[-1]

    torch._dynamo.reset()
    explained = torch._dynamo.explain(fn)(rows, ids)
    assert explained.graph_break_count == 0 and explained.graph_count == 1
    compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
    for _ in range(3):
        args, sizes = compiled(rows, ids)
        eager_args, eager_sizes = fn(rows, ids)
        assert torch.equal(args, eager_args) and torch.equal(sizes, eager_sizes)
    expected = host_encode(encoder, case, capacities=caps)
    assert np.array_equal((args - 1).cpu().numpy(), expected.relation_args)
    # new rows of another count are traced again and give the encoding of those rows
    n = rows.shape[0] // 2
    args, _ = compiled(rows[:n], None if ids is None else ids[:n])
    half = encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n], capacities=caps)
    assert np.array_equal((args - 1).cpu().numpy(), half.relation_args)
    rt.unregister(handle)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_a_captured_cuda_graph_replays_to_the_eager_result(cases, layout):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(
        case.source, layout=layout, target_sources=("goal",), goal_derivations=("plain", "satisfied"))
    handle = rt.register(encoder)
    caps = exact_capacities(host_encode(encoder, case), 2)
    rows, ids, derived, g = op_inputs(case, dev)
    eager = call_op(handle, rows, ids, derived, g, caps)
    stream = torch.cuda.Stream(dev)
    with torch.cuda.stream(stream):
        call_op(handle, rows, ids, derived, g, caps)
    stream.synchronize()
    reservations = encoding._device.workspace_state(encoder, stream=stream)["reservations"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call_op(handle, rows, ids, derived, g, caps)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(captured, eager, strict=True):
            assert torch.equal(a, b)
    # the graph reads the rows where they were: new rows written in place give the encoding of those rows
    other = np.ascontiguousarray(case.rows[::-1])
    rows.copy_(to_device(other, dev))
    if ids is not None:
        ids.copy_(to_device(np.ascontiguousarray(case.ids[::-1]), dev))
    graph.replay()
    torch.cuda.synchronize()
    expected = encoder.encode(other, None if case.ids is None else np.ascontiguousarray(case.ids[::-1]), capacities=caps)
    same_batch(expected, encoding.DeviceFlatBatch.from_arrays(encoder, captured))
    assert encoding._device.workspace_state(encoder, stream=stream)["reservations"] == reservations
    del graph
    rt.unregister(handle)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_an_encode_without_capacities_cannot_be_captured(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    stream = torch.cuda.Stream(dev)
    encoder.reserve_device(case.n, stream=stream)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="cannot be captured"):
        with torch.cuda.graph(graph, stream=stream):
            encoder.encode(rows)


def test_the_handle_keeps_the_encoder_alive(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, goal_derivations=("plain", "satisfied"))
    expected = host_encode(encoder, case, capacities=exact_capacities(host_encode(encoder, case)))
    caps = exact_capacities(host_encode(encoder, case))
    handle = rt.register(encoder)
    del encoder
    gc.collect()
    rows, ids, derived, g = op_inputs(case, dev)
    out = call_op(handle, rows, ids, derived, g, caps)
    same_batch(expected, encoding.DeviceFlatBatch.from_arrays(rt.lookup(handle), out))
    rt.unregister(handle)
    with pytest.raises(ValueError, match="no object registered"):
        call_op(handle, rows, ids, derived, g, caps)


def test_encoders_still_pickle_and_free_their_device_state(cases):
    import pickle
    import weakref

    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    device_encode(encoder, case, dev)
    clone = pickle.loads(pickle.dumps(encoder))
    same_batch(host_encode(clone, case), device_encode(clone, case, dev))
    ref = weakref.ref(encoder)
    del encoder
    gc.collect()
    assert ref() is None


def test_short_graph_outputs_are_refused_before_fill(cases, monkeypatch):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    caps = exact_capacities(host_encode(encoder, case))
    layout = encoding._device.layout_of(encoder)
    real = type(layout).specs

    def short(self, torch_, graphs, entities, args, targets):
        return real(self, torch_, graphs - 1, entities, args, targets)

    monkeypatch.setattr(type(layout), "specs", short)
    with pytest.raises(ValueError, match="^mymyr: encoding: the batch has more graphs than the output capacity"):
        encoder.encode(rows, capacities=caps)
    with pytest.raises(ValueError, match="^mymyr: encoding: the batch has more graphs than the output capacity"):
        encoder.encode(rows)


def test_graph_capacity_is_checked_before_device_encoding(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    caps["graphs"] = len(case.rows) - 1
    with pytest.raises(ValueError, match="more graphs than the output capacity"):
        device_encode(encoder, case, dev, capacities=caps)
    handle = rt.register(encoder)
    rows, ids, derived, goals = op_inputs(case, dev)
    try:
        with pytest.raises(ValueError, match="more graphs than the output capacity"):
            call_op(handle, rows, ids, derived, goals, caps)
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode() as mode, pytest.raises(ValueError, match="more graphs than the output capacity"):
            call_op(handle, mode.from_tensor(rows), None, None, [None] * 4, caps)
    finally:
        rt.unregister(handle)


def test_the_op_refuses_host_rows_with_a_value_error():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    handle = rt.register(encoder)
    rows = torch.from_numpy(np.asarray(task.encode([task.initial_state])).view(np.int64))
    try:
        with pytest.raises(ValueError, match="^mymyr: encoding: 'states' is on cpu"):
            call_op(handle, rows, None, None, [None] * 4, {"graphs": 1, "entities": 0, "args": 0, "targets": 0})
    finally:
        rt.unregister(handle)


def test_cuda_array_interface_stream_zero_is_refused():
    class Array:
        __cuda_array_interface__ = {"stream": 0}

    encoder = encoding.FlatRelationEncoder(text_task("gripper__prob05"))
    with pytest.raises(ValueError, match="CUDA array interface stream 0"):
        encoder.encode(Array())


@pytest.mark.parametrize("framework", ["Torch", "PYTORCH"])
def test_device_framework_names_follow_the_host_spelling(cases, framework):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    with pytest.raises(ValueError, match="framework"):
        encoder.encode(case.rows, framework=framework)
    with pytest.raises(ValueError, match="framework"):
        device_encode(encoder, case, dev, framework=framework)


@pytest.mark.parametrize("threads", [-5, 10**6, "2"])
def test_device_threads_are_validated_as_on_the_host(cases, threads):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    with pytest.raises((TypeError, ValueError)):
        encoder.encode(case.rows, threads=threads)
    with pytest.raises((TypeError, ValueError)):
        device_encode(encoder, case, dev, threads=threads)


def test_a_compiled_function_specializes_each_encoder_handle(cases):
    dev = gpu()
    case = cases["gripper"]
    encoders = [encoding.FlatRelationEncoder(case.source, emit_counts=counts) for counts in (False, True)]
    handles = [rt.register(encoder) for encoder in encoders]
    rows = to_device(case.rows, dev)
    caps = exact_capacities(encoders[0].encode(case.rows))

    def fn(handle, states):
        return call_op(handle, states, None, None, [None] * 4, caps)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    try:
        for handle in (*handles, handles[0]):
            eager = fn(handle, rows)
            actual = compiled(handle, rows)
            for a, b in zip(actual, eager, strict=True):
                assert torch.equal(a, b)
    finally:
        for handle in handles:
            rt.unregister(handle)


@pytest.mark.parametrize("mode", ["capacities", "op"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_captured_scratch_survives_larger_calls_and_a_second_capture(cases, monkeypatch, mode, layout):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, target_sources=("goal",))
    handle = rt.register(encoder)
    stream = torch.cuda.Stream(dev)
    scratch = encoding._device._scratch
    pointers = []

    def allocate(*args):
        tensor = scratch(*args)
        pointers.append((tensor.data_ptr(), tensor.numel()))
        return tensor

    monkeypatch.setattr(encoding._device, "_scratch", allocate)
    batches = []
    graphs = []
    held = []
    expected = []
    inputs = []

    def encode(rows, caps):
        if mode == "op":
            return encoding.DeviceFlatBatch.from_arrays(encoder, call_op(handle, rows, None, None, [None] * 4, caps))
        return encoder.encode(rows, capacities=caps)

    def pool_owns(graph, allocation):
        pointer, size = allocation
        return any(tuple(segment["segment_pool_id"]) == graph.pool() and
                   segment["address"] <= pointer and pointer + size <= segment["address"] + segment["total_size"]
                   for segment in torch.cuda.memory_snapshot())

    try:
        for n in (8, 4097):
            rows = np.ascontiguousarray(np.resize(case.rows, (n, case.rows.shape[1])))
            host = encoder.encode(rows)
            caps = exact_capacities(host, 2)
            expected.append(encoder.encode(rows, capacities=caps))
            device_rows = to_device(rows, dev)
            inputs.append(device_rows)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                encode(device_rows, caps)
            stream.synchronize()
            encoder.reserve_device(2 * n, stream=stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                batches.append(encode(device_rows, caps))
            graphs.append(graph)
            held.append(pointers[-1])
            assert all(pool_owns(g, p) for g, p in zip(graphs, held, strict=True))
        assert held[0][0] != held[1][0]
        encoder.release_device(stream=stream)
        assert all(pool_owns(g, p) for g, p in zip(graphs, held, strict=True))
        for graph, host, batch in zip(graphs, expected, batches, strict=True):
            graph.replay()
            torch.cuda.synchronize()
            same_batch(host, batch)
    finally:
        torch.cuda.synchronize()
        rt.unregister(handle)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("options", ["default", "goal_targets_virtual", "no_actions"])
def test_capacity_fill_overwrites_dirty_outputs_and_tails(cases, monkeypatch, layout, index, options):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, **OPTION_SETS[options])

    def outputs(torch_, specs, device):
        return {name: torch_.full(shape, 0x5a, dtype=dtype, device=device) for name, (shape, dtype) in specs.items()}

    monkeypatch.setattr(encoding._device, "_outputs", outputs)
    exact = host_encode(encoder, case)
    for caps in (exact_capacities(exact, 2), dict(exact_capacities(exact), args=0)):
        same_batch(host_encode(encoder, case, capacities=caps), device_encode(encoder, case, dev, capacities=caps))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("given", [False, True])
def test_device_row_stream_wraps_and_flushes_live_rows(cases, layout, index, given):
    dev = gpu()
    case = cases["philosophers_table"] if "philosophers_table" in cases else cases["philosophers"]
    n = min(33, case.n)
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, target_sources=("goal",),
                                           predicate_virtual_nodes=True)
    rows = to_device(case.rows[:n], dev)
    ids = None if case.ids is None else to_device(case.ids[:n], dev)
    derived = to_device(case.derived[:n], dev) if given else None
    host = encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n])
    caps = exact_capacities(host, factor=2)
    caps["graphs"] = n + 4
    ring = encoder.device_stream(n + 4, derived=given)
    for _ in range(3):
        ring.append(rows, ids, derived=derived)
        ring.append(rows[:5], None if ids is None else ids[:5], derived=None if derived is None else derived[:5])
        assert int(ring.device_overflow.item()) == 1
        assert len(ring) == n
        batch = ring.flush(capacities=caps)
        assert batch.device_graphs is not None
        same_batch(host, batch)
        assert len(ring) == 0
    empty = ring.flush(capacities=caps)
    same_batch(encoder.encode(case.rows[:0], None if case.ids is None else case.ids[:0]), empty)
    ring.clear()
    assert int(ring.device_overflow.item()) == 0


def test_device_row_stream_append_and_flush_are_capturable(cases):
    dev = gpu()
    case = cases["philosophers"]
    n = min(33, case.n)
    encoder = encoding.FlatRelationEncoder(case.source, target_sources=("goal",))
    host = encoder.encode(case.rows[:n])
    rows = to_device(case.rows[:n], dev)
    caps = exact_capacities(host, factor=2)
    caps["graphs"] = n + 7
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        ring = encoder.device_stream(n + 7)
        ring.append(rows)
        ring.flush(capacities=caps)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ring.append(rows)
            batch = ring.flush(capacities=caps)
        for _ in range(4):
            graph.replay()
        stream.synchronize()
        same_batch(host, batch)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_device_collation_of_rollout_slices_equals_one_encode(cases, layout, index):
    dev = gpu()
    case = cases["philosophers_table"] if "philosophers_table" in cases else cases["philosophers"]
    n = min(33, case.n)
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, target_sources=("goal",),
                                           predicate_virtual_nodes=True)
    rows = to_device(case.rows[:n], dev)
    ids = None if case.ids is None else to_device(case.ids[:n], dev)
    host = encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n])
    pieces = [(0, 3), (3, 3), (3, 11), (11, n)]
    batches = [encoder.encode(rows[a:b], None if ids is None else ids[a:b]) for a, b in pieces]
    same_batch(host, encoding.collate(batches))
    plan = encoding.DeviceCollator(batches)
    caps = exact_capacities(host)
    batch = plan.encode(capacities=caps)
    same_batch(host, batch)
    ample = exact_capacities(host, factor=2)
    host_ample = encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n], capacities=ample)
    same_batch(host_ample, plan.encode(capacities=ample))
    short = plan.encode(capacities={**caps, "args": 0})
    assert short.sizes["capacity_overflow"]
    assert not torch.count_nonzero(short.entity_offsets).item()
    assert not torch.count_nonzero(short.relation_offsets).item()
    same_batch(encoder.encode(case.rows[:0], None if case.ids is None else case.ids[:0]),
               encoding.collate([batches[1], batches[1]]))
    with pytest.raises(ValueError, match="^mymyr: encoding: collate received mixed host and device"):
        encoding.collate([host, batches[0]])
    bad = encoder.encode(rows, ids, capacities={**caps, "args": 0})
    invalid_plan = encoding.DeviceCollator([bad])
    with pytest.raises(ValueError, match="^mymyr: encoding: cannot collate an overflowed batch"):
        invalid_plan.encode()
    assert invalid_plan.encode(capacities=caps).sizes["invalid_input"]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        captured_plan = encoding.DeviceCollator(batches)
        captured_plan.encode(capacities=caps)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = captured_plan.encode(capacities=caps)
        for _ in range(3):
            graph.replay()
        stream.synchronize()
        same_batch(host, captured)


@pytest.mark.parametrize("mode", ["exact", "capacity"])
def test_evaluated_derived_words_follow_the_encoding_stream(mode):
    dev = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = mymyr.Domain.from_string(
        "(define (domain switching) (:requirements :strips :derived-predicates)"
        " (:predicates (flag ?x) (available ?x)) (:derived (available ?x) (flag ?x))"
        " (:action move :parameters (?a ?b) :precondition (flag ?a)"
        " :effect (and (not (flag ?a)) (flag ?b))))")
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain switching) (:objects a b) (:init (flag a)) (:goal (and)))"))
    old = np.asarray(task.encode([task.initial_state]))
    successors = np.asarray(rl.expand(task, old).succ)
    changed = next(row for row in successors if not np.array_equal(row, old[0]))[None, :]
    encoder = encoding.FlatRelationEncoder(task)
    before, host = encoder.encode(old), encoder.encode(changed)
    assert np.array_equal(before.relation_counts, host.relation_counts)
    assert not np.array_equal(before.relation_args, host.relation_args)
    original, rows = to_device(old, dev), to_device(changed, dev)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        destination = original.clone()
        encoder.encode(destination, capacities=exact_capacities(host))
        stream.synchronize()
        torch.cuda._sleep(10_000_000)
        destination.copy_(rows)
        result = encoder.encode(destination, **({"capacities": exact_capacities(host)} if mode == "capacity" else {}))
        stream.synchronize()
        same_batch(host, result)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_prepared_collation_keeps_ring_capacity_when_relations_grow(layout, index):
    dev = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = mymyr.Domain.from_string(
        "(define (domain switches) (:requirements :strips) (:predicates (left) (right))"
        " (:action left_on :parameters () :precondition (and) :effect (left))"
        " (:action right_on :parameters () :precondition (and) :effect (right)))")
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain switches) (:init) (:goal (and)))"))
    one = task.successor_states(task.initial_state)[0]
    two = max(task.successor_states(one), key=lambda state: sum(int(word).bit_count() for word in state.words))
    empty_rows = np.asarray(task.encode([task.initial_state, task.initial_state]))
    populated_rows = np.asarray(task.encode([two, one]))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, index_dtype=index, include_static=False,
                                           include_actions=False, goal_derivations=())
    empty_host, populated_host = encoder.encode(empty_rows), encoder.encode(populated_rows)
    assert empty_host.entity_ref.size == empty_host.relation_args.size == 0
    assert populated_host.entity_ref.size > 0 and populated_host.relation_args.size > 0
    caps = dict(graphs=3, entities=8, args=16, targets=0)
    rows, populated = to_device(empty_rows, dev), to_device(populated_rows, dev)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        ring = encoder.device_stream(3)
        ring.append(rows)
        ring.flush(capacities=caps)
        stream.synchronize()
        producer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(producer, stream=stream):
            ring.append(rows)
            source = ring.flush(capacities=caps)
        producer.replay()
        stream.synchronize()
        plan = encoding.DeviceCollator([source])
        # Check the backing pointers before any replay can populate the empty live arrays.
        prepared = plan._prepared[4][0]
        for name in ("entity_role", "entity_ref", "relation_args"):
            assert source._arrays[name].numel() > 0
            assert prepared[name].data_ptr() == source._arrays[name].data_ptr() != 0
        plan.encode(capacities=caps)
        stream.synchronize()
        consumer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(consumer, stream=stream):
            result = plan.encode(capacities=caps)
        for values, host_rows in ((populated, populated_rows), (rows, empty_rows), (populated, populated_rows)):
            if values is not rows:
                rows.copy_(values)
            else:
                rows.zero_()
            producer.replay()
            consumer.replay()
            stream.synchronize()
            same_batch(encoder.encode(host_rows, capacities=caps), result)
