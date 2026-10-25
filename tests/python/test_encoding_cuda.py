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
from mymyr.encoding._device import _POINTER_FIELDS
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
ALL_GOALS = ("plain", "satisfied", "unsatisfied")
LAYERED_OPTION_SETS = {
    "levels_targets": {"max_goal_level": 3, "goal_derivations": ALL_GOALS, "target_sources": ("goal", "subgoal")},
    "history": {"history": True},
    "history_newest_targets": {"history": True, "history_max_steps": 2, "target_sources": ("history",)},
    "levels_history_virtual": {
        "max_goal_level": 2,
        "history": True,
        "target_sources": ("goal", "subgoal", "history"),
        "predicate_virtual_nodes": True,
        "goal_derivations": ("plain", "satisfied"),
    },
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
    for key in ("subgoals", "history"):
        if kwargs.get(key) is not None:
            kwargs[key] = tuple(None if x is None else to_device(x, dev) for x in kwargs[key])
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
    if hasattr(batch, "history_offsets"):
        sizes["history"] = int(batch.history_offsets[-1])
    return {k: v * factor for k, v in sizes.items()}


def layered_rows(case, layers=3, steps=4, seed=11):
    """Random subgoal layers and history steps for the rows of ``case`` as the masks of ``encode``: (subgoals,
    history). A row uses a random number of its layers (the others are empty) and has a random number of valid steps;
    the literals are random over its slots (over its derived atoms for the derived masks)."""
    rng = np.random.default_rng(seed)
    table = rl.as_table(case.source) if isinstance(case.source, mymyr.Task) else case.source
    offsets = np.asarray(table.atom_metadata()["atom_offsets"])
    ids = np.zeros(case.n, np.int64) if case.ids is None else case.ids
    n, words = case.rows.shape
    dwords = None if case.derived is None else case.derived.shape[1]

    def masks(count):
        pos, neg = (np.zeros((n, count, words), np.uint64) for _ in range(2))
        dpos = dneg = None
        if dwords is not None:
            dpos, dneg = (np.zeros((n, count, dwords), np.uint64) for _ in range(2))
        return pos, neg, dpos, dneg

    def draw(pos, neg, dpos, dneg, row, k):
        slots = int(offsets[ids[row] + 1] - offsets[ids[row]])
        for slot in range(slots):
            r = rng.random()
            target = pos if r < 0.08 else neg if r < 0.16 else None
            if target is not None:
                target[row, k, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
        if dwords is not None:
            held = [w * 64 + b for w in range(dwords) for b in range(64) if (int(case.derived[row, w]) >> b) & 1]
            for slot in held:
                r = rng.random()
                target = dpos if r < 0.4 else dneg if r < 0.6 else None
                if target is not None:
                    target[row, k, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)

    used = rng.integers(0, layers + 1, n)
    valid = rng.integers(0, steps + 1, n).astype(np.int32)
    layer_masks, step_masks = masks(layers), masks(steps)
    for row in range(n):
        for k in range(int(used[row])):
            draw(*layer_masks, row, k)
        for k in range(steps):
            draw(*step_masks, row, k)
    dt = rng.integers(-20, 40, (n, steps)).astype(np.int32)
    keep = lambda m: tuple(x for x in m if x is not None)  # noqa: E731
    return keep(layer_masks), (*keep(step_masks)[:2], dt, valid, *keep(step_masks)[2:])


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
        assert batch.device_sizes.dtype == torch.int64 and batch.device_sizes.shape == (6,)
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


def sliced(case, n):
    case = copy.copy(case)
    case.rows = case.rows[:n]
    case.ids = None if case.ids is None else case.ids[:n]
    case.derived = None if case.derived is None else case.derived[:n]
    return case


def layered_kwargs(case, options, **kwargs):
    """The ``subgoals`` and ``history`` arguments (host arrays) an encoder with these options takes for the rows."""
    subgoals, history = layered_rows(case, **kwargs)
    out = {}
    if options.get("max_goal_level"):
        # the layers an encoder takes: at most as many as its levels
        out["subgoals"] = tuple(np.ascontiguousarray(x[:, : options["max_goal_level"]]) for x in subgoals)
    if options.get("history"):
        out["history"] = history
    return out


@pytest.mark.parametrize("options", LAYERED_OPTION_SETS.values(), ids=LAYERED_OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
@pytest.mark.parametrize("policy", ["every", "lowest", "highest"])
def test_subgoal_layers_and_history_equal_host_bit_for_bit(cases, layout, index, options, n, policy):
    dev = gpu()
    for case in cases.values():
        if n is not None:
            case = sliced(case, n)
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, repeated_literals=policy, **options)
        kwargs = layered_kwargs(case, options)
        host = host_encode(encoder, case, **kwargs)
        batch = device_encode(encoder, case, dev, **kwargs)
        assert isinstance(batch, encoding.DeviceFlatBatch)
        if options.get("history"):
            assert batch.history_dt.dtype == torch.int32 and batch.history_offsets.shape == (case.n + 1,)
        same_batch(host, batch)
        # fewer layers and steps than the arrays hold, and a history without counts: every step is valid
        if n is None and "history" in kwargs:
            pos, neg, dt, _count, *derived = kwargs["history"]
            every = {**kwargs, "history": (pos, neg, dt, None, *derived)}
            same_batch(host_encode(encoder, case, **every), device_encode(encoder, case, dev, **every))
        if n is None and "subgoals" in kwargs:
            one = {**kwargs, "subgoals": tuple(np.ascontiguousarray(x[:, :1]) for x in kwargs["subgoals"])}
            same_batch(host_encode(encoder, case, **one), device_encode(encoder, case, dev, **one))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("policy", ["every", "lowest", "highest"])
def test_subgoal_layers_and_history_with_row_goals_equal_host(cases, layout, policy):
    dev = gpu()
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    for name in ("gripper", "blocks_table", "philosophers_table"):
        if name not in cases:
            continue
        case = cases[name]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, repeated_literals=policy, **options)
        goals = fluent_goals(case)
        kwargs = layered_kwargs(case, options)
        same_batch(host_encode(encoder, case, goals, **kwargs), device_encode(encoder, case, dev, goals, **kwargs))
        if case.ids is None:  # layers that every row shares
            shared = {**kwargs, "subgoals": tuple(np.ascontiguousarray(x[:1]) for x in kwargs["subgoals"])}
            same_batch(host_encode(encoder, case, goals, **shared), device_encode(encoder, case, dev, goals, **shared))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", ["gripper", "blocks_table", "philosophers_table"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
def test_the_capacity_form_with_history_equals_host(cases, layout, name, n):
    dev = gpu()
    if name not in cases:
        pytest.skip("built without the loki front end")
    case = cases[name] if n is None else sliced(cases[name], n)
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **options)
    kwargs = layered_kwargs(case, options)
    exact = exact_capacities(host_encode(encoder, case, **kwargs))
    assert set(exact) == {"graphs", "entities", "args", "targets", "history"}
    roomy = exact_capacities(host_encode(encoder, case, **kwargs), 2) if case.n else dict.fromkeys(exact, 10)
    short_history = dict(exact, history=max(0, exact["history"] - 1))
    zero = {"entities": 0, "graphs": case.rows.shape[0], "args": 0, "targets": 0, "history": 0}
    for caps in (exact, roomy, short_history, dict(exact, targets=max(0, exact["targets"] - 1)), zero):
        host = host_encode(encoder, case, capacities=caps, **kwargs)
        batch = device_encode(encoder, case, dev, capacities=caps, **kwargs)
        same_batch(host, batch)
        assert batch.sizes == host.sizes and batch.sizes["invalid_input"] is False
        assert batch.device_sizes.dtype == torch.int64 and batch.device_sizes.shape == (6,)
        assert batch.history_entity.shape[0] == caps["history"]
    assert host_encode(encoder, case, capacities=short_history, **kwargs).sizes["overflow"] == (exact["history"] > 0)


def test_the_capacities_of_a_schema_with_history_name_the_history(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, history=True)
    caps = {"entities": 1, "graphs": case.n, "args": 1}
    with pytest.raises(ValueError, match="the schema has history: capacities needs the key 'history'"):
        device_encode(encoder, case, dev, capacities=caps, history=layered_rows(case)[1])
    plain = encoding.FlatRelationEncoder(case.source)
    batch = device_encode(plain, case, dev, capacities={**caps, "history": 0})
    assert "history" not in batch.sizes and batch.device_sizes.shape == (6,)


def test_layers_and_history_the_schema_does_not_have_are_refused(cases):
    dev = gpu()
    case = cases["gripper"]
    subgoals, history = layered_rows(case)
    plain = encoding.FlatRelationEncoder(case.source)
    with pytest.raises(ValueError, match="^mymyr: encoding: subgoal layers are given, but the schema has no subgoal"):
        device_encode(plain, case, dev, subgoals=subgoals)
    with pytest.raises(ValueError, match="^mymyr: encoding: history steps are given, but the schema has no history"):
        device_encode(plain, case, dev, history=history)
    levels = encoding.FlatRelationEncoder(case.source, max_goal_level=2)
    with pytest.raises(ValueError, match="3 subgoal layers are given, the schema has 2 levels"):
        device_encode(levels, case, dev, subgoals=subgoals)
    # an encoder with layers or history encodes the rows without them
    for options in ({"max_goal_level": 2}, {"history": True}):
        encoder = encoding.FlatRelationEncoder(case.source, **options)
        same_batch(host_encode(encoder, case), device_encode(encoder, case, dev))


def test_mask_pairs_with_different_row_strides_are_refused_before_any_launch():
    # the device reads both masks of a pair with one row stride: a negative mask with a shorter stride than the
    # positive one would be read past its end (only the checks run here, no kernel)
    gpu()
    from mymyr.encoding import _device
    n, layers, words = 8, 2, 1
    states = torch.zeros((n, words), dtype=torch.int64, device="cuda")
    pos = torch.zeros((2 * n, layers, words), dtype=torch.int64, device="cuda")[::2]
    neg = torch.zeros((n, layers, words), dtype=torch.int64, device="cuda")
    assert pos.stride(0) != neg.stride(0) and pos.shape == neg.shape
    dt = torch.zeros((n, layers), dtype=torch.int32, device="cuda")
    for subgoals, history in (((pos, neg), None), ((neg, pos), None), ((neg, neg, pos, neg), None),
                              (None, (pos, neg, dt)), (None, (neg, neg, dt, None, pos, neg))):
        with pytest.raises(ValueError, match="row stride of .* the two masks need the same one"):
            _device._Rows(torch, states, None, None, None, subgoals, history)
    goal_pos = torch.zeros((2 * n, words), dtype=torch.int64, device="cuda")[::2]
    goal_neg = torch.zeros((n, words), dtype=torch.int64, device="cuda")
    for goals in ((goal_pos, goal_neg), (goal_neg, goal_neg, goal_pos, goal_neg)):
        with pytest.raises(ValueError, match="row stride of .* the two masks need the same one"):
            _device._Rows(torch, states, None, None, goals)
    # equal strides are accepted, strided or not
    _device._Rows(torch, states, None, None, (goal_pos, goal_pos), (pos, pos), (pos, pos, dt))
    _device._Rows(torch, states, None, None, (goal_neg, goal_neg), (neg, neg), (neg, neg, dt))


def test_invalid_layers_and_history_are_flagged_or_raise(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    kwargs = layered_kwargs(case, options)
    caps = exact_capacities(host_encode(encoder, case, **kwargs))
    steps = kwargs["history"][2].shape[1]
    for count in (steps + 1, -1):
        bad_count = kwargs["history"][3].copy()
        bad_count[3] = count
        bad = {**kwargs, "history": (*kwargs["history"][:3], bad_count, *kwargs["history"][4:])}
        with pytest.raises(ValueError, match="^mymyr: encoding: a row is invalid"):
            device_encode(encoder, case, dev, **bad)
        sizes = device_encode(encoder, case, dev, capacities=caps, **bad).sizes
        assert sizes["overflow"] and sizes["invalid_input"] and not sizes["capacity_overflow"]
    # a subgoal bit past the slots of the instance
    pos = kwargs["subgoals"][0].copy()
    table = rl.as_table(case.source) if isinstance(case.source, mymyr.Task) else case.source
    offsets = np.asarray(table.atom_metadata()["atom_offsets"])
    ids = np.zeros(case.n, np.int64) if case.ids is None else case.ids
    row = next(r for r in range(case.n) if int(offsets[ids[r] + 1] - offsets[ids[r]]) < 64 * case.rows.shape[1])
    slots = int(offsets[ids[row] + 1] - offsets[ids[row]])
    pos[row, 0, slots // 64] |= np.uint64(1) << np.uint64(slots % 64)
    bad = {**kwargs, "subgoals": (pos, *kwargs["subgoals"][1:])}
    with pytest.raises(ValueError, match="set an atom bit its instance does not have"):
        host_encode(encoder, case, **bad)
    sizes = device_encode(encoder, case, dev, capacities=caps, **bad).sizes
    assert sizes["invalid_input"]
    assert device_encode(encoder, case, dev, capacities=caps, **kwargs).sizes["overflow"] is False


def test_the_arguments_are_checked_before_any_launch(cases):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    rows = to_device(case.rows, dev)
    ids = None if case.ids is None else to_device(case.ids, dev)
    prefix = "^mymyr: encoding: "
    with pytest.raises(ValueError, match=prefix + r"(?:the action expansion belongs to another table|'actions.schema' must be a CUDA array)"):
        encoder.encode(rows, ids, actions=rl.expand(rl.as_table(case.tasks[0]), case.rows[:1]))
    with pytest.raises(ValueError, match=prefix + r"'subgoals\[0\] \(pos\)' must be"):
        encoder.encode(rows, ids, subgoals=(rows, rows))
    with pytest.raises(ValueError, match=prefix + r"history must be \(pos, neg, dt\)"):
        encoder.encode(rows, ids, history=(rows, rows))
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
        return allocate(1 << 31, 0, 0, 0, 0, 0, 2, 0, 0, 0)

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
        out["sizes"].copy_(torch.tensor([1 << 31, 0, 0, 0, 0, 2], device=device))
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


def test_action_target_schemas_without_expansion_rows_are_encoded():
    dev = gpu()
    task = text_task("gripper__prob05")
    states = walk(task, steps=3, walks=1)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    rows = to_device(np.asarray(task.encode(states)), dev)
    batch = encoder.encode(rows)
    same_batch(encoder.encode(np.asarray(task.encode(states))), batch)
    assert len(batch.target_entity) == 0
    encoder.reserve_device(8)


@pytest.mark.parametrize("outputs", [("simple",), ("relations", "simple")])
def test_the_simple_graph_with_numeric_relations_is_refused_by_the_device_before_any_launch(outputs):
    dev = gpu()
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    encoder = encoding.FlatRelationEncoder(task, outputs=outputs)
    message = "^mymyr: encoding: the device does not encode Simple with numeric relations"
    rows = to_device(np.asarray(task.encode([task.initial_state])), dev)
    with pytest.raises(ValueError, match=message):
        encoder.encode(rows)
    with pytest.raises(ValueError, match=message):
        encoder.reserve_device(8)
    with pytest.raises(ValueError, match=message):
        encoder.device_stream(8)
    from torch._subclasses.fake_tensor import FakeTensorMode

    handle = rt.register(encoder)
    caps = dict.fromkeys(("graphs", "entities", "args", "targets"), 64)
    try:
        with pytest.raises(ValueError, match=message):
            call_op(handle, rows, None, None, [None] * 4, caps)
        with FakeTensorMode() as mode, pytest.raises(ValueError, match=message):
            call_op(handle, mode.from_tensor(rows), None, None, [None] * 4, caps)
    finally:
        rt.unregister(handle)
    # the native collator of an empty batch: nothing is read and no kernel is launched
    from mymyr._core._encoding import DeviceCollator as NativeCollator

    with pytest.raises(ValueError, match=message):
        NativeCollator([encoder], dev.index, 0, [[0] * _POINTER_FIELDS], [[0, 0, 0]])


# ------------------------------------------------------------------------------------------------ the op

OP_ARGS = 8


def op_inputs(case, dev, goals=None):
    ids = None if case.ids is None else to_device(case.ids, dev)
    derived = None if case.derived is None else to_device(case.derived, dev)
    g = [None] * 4 if goals is None else [to_device(x, dev) for x in goals] + [None] * (4 - len(goals))
    return to_device(case.rows, dev), ids, derived, g


def call_op(handle, rows, ids, derived, g, caps, subgoals=None, history=None):
    sg = [None] * 4 if subgoals is None else [*subgoals, *([None] * (4 - len(subgoals)))]
    hs = [None] * 6 if history is None else [*history, *([None] * (6 - len(history)))]
    return torch.ops.mymyr.encode_flat(
        handle, rows, ids, derived, g[0], g[1], g[2], g[3], *sg, *hs,
        caps["graphs"], caps["entities"], caps["args"], caps["targets"], caps.get("history", 0),
    )


def layered_operands(case, dev, options):
    """The subgoal and history operands of the op (device tensors, or None) for an encoder with these options."""
    kwargs = layered_kwargs(case, options)
    return tuple(None if key not in kwargs else tuple(to_device(x, dev) for x in kwargs[key])
                 for key in ("subgoals", "history"))


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
                                    None, *([None] * 10), 1, 1, 1, 1, 0)
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
        out = call_op(handle, rows, ids, None, [None] * 4, caps)
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

    def short(self, torch_, graphs, entities, args, targets, history=0, **sizes):
        return real(self, torch_, graphs - 1, entities, args, targets, history, **sizes)

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


@pytest.mark.needs_gil
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
@pytest.mark.parametrize("name", ["gripper", "blocks_table", "philosophers_table"])
def test_the_op_takes_subgoal_layers_and_history(cases, layout, name):
    dev = gpu()
    if name not in cases:
        pytest.skip("built without the loki front end")
    case = cases[name]
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **options)
    handle = rt.register(encoder)
    kwargs = layered_kwargs(case, options)
    caps = exact_capacities(host_encode(encoder, case, **kwargs), 2)
    rows, ids, derived, g = op_inputs(case, dev)
    subgoals, history = layered_operands(case, dev, options)
    out = call_op(handle, rows, ids, derived, g, caps, subgoals, history)
    assert len(out) == len(encoding.DeviceFlatBatch.array_names)
    batch = encoding.DeviceFlatBatch.from_arrays(encoder, out)
    same_batch(host_encode(encoder, case, capacities=caps, **kwargs), batch)
    assert batch.sizes["overflow"] is False and batch.sizes["history"] > 0 and batch.device_sizes.shape == (6,)
    for bad, text in (((subgoals[0], None), "come in pairs"), ):
        with pytest.raises(ValueError, match=text):
            call_op(handle, rows, ids, derived, g, caps, bad, history)
    with pytest.raises(ValueError, match="history need their masks"):
        call_op(handle, rows, ids, derived, g, caps, subgoals, (None, None, history[2], history[3]))
    rt.unregister(handle)


def test_the_fake_implementation_gives_the_history_arrays(cases):
    dev = gpu()
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = cases["gripper"]
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    handle = rt.register(encoder)
    caps = {"graphs": case.n, "entities": 400, "args": 3000, "targets": 60, "history": 25}
    rows, ids, derived, g = op_inputs(case, dev)
    subgoals, history = layered_operands(case, dev, options)
    real = call_op(handle, rows, ids, derived, g, caps, subgoals, history)
    with FakeTensorMode() as mode:
        fake = call_op(handle, mode.from_tensor(rows), None, None, [None] * 4, caps)
    assert len(fake) == len(real)
    for f, r in zip(fake, real, strict=True):
        assert f.shape == r.shape and f.dtype == r.dtype and f.device == r.device
    names = encoding.DeviceFlatBatch.array_names
    assert fake[names.index("history_entity")].shape == (25,) and fake[names.index("history_dt")].dtype == torch.int32
    rt.unregister(handle)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.needs_gil
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_the_op_with_layers_compiles_and_replays_in_a_cuda_graph(cases, layout):
    dev = gpu()
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **options)
    handle = rt.register(encoder)
    kwargs = layered_kwargs(case, options)
    caps = exact_capacities(host_encode(encoder, case, **kwargs), 2)
    rows, ids, derived, g = op_inputs(case, dev)
    subgoals, history = layered_operands(case, dev, options)

    def fn(rows, ids, subgoals, history):
        return call_op(handle, rows, ids, None, [None] * 4, caps, subgoals, history)

    eager = fn(rows, ids, subgoals, history)
    same_batch(host_encode(encoder, case, capacities=caps, **kwargs), encoding.DeviceFlatBatch.from_arrays(encoder, eager))
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
    for _ in range(2):
        for a, b in zip(compiled(rows, ids, subgoals, history), eager, strict=True):
            assert torch.equal(a, b)
    stream = torch.cuda.Stream(dev)
    with torch.cuda.stream(stream):
        fn(rows, ids, subgoals, history)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn(rows, ids, subgoals, history)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(captured, eager, strict=True):
            assert torch.equal(a, b)
    rt.unregister(handle)


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
        assert len(ring) == n
        batch = ring.flush(capacities=caps)
        assert batch.device_graphs is not None
        same_batch(host, batch)
        assert len(ring) == 0
    # an append that does not fit is dropped, and every flush until clear() reports it
    ring.append(rows, ids, derived=derived)
    ring.append(rows[:5], None if ids is None else ids[:5], derived=None if derived is None else derived[:5])
    assert int(ring.device_overflow.item()) == 1
    assert len(ring) == n
    for _ in range(2):
        dropped = ring.flush(capacities=caps)
        assert dropped.sizes["capacity_overflow"]
        assert not torch.count_nonzero(dropped.entity_offsets).item()
    ring.clear()
    assert int(ring.device_overflow.item()) == 0
    empty = ring.flush(capacities=caps)
    same_batch(encoder.encode(case.rows[:0], None if case.ids is None else case.ids[:0]), empty)
    ring.append(rows, ids, derived=derived)
    same_batch(host, ring.flush(capacities=caps))


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_device_row_stream_flush_zeroes_the_per_graph_tails_and_leaves_the_variable_tails(cases, layout, monkeypatch):
    dev = gpu()
    case = cases["philosophers"]
    n = min(33, case.n)
    encoder = encoding.FlatRelationEncoder(case.source, layout=layout, target_sources=("goal",),
                                           predicate_virtual_nodes=True)

    def outputs(torch_, specs, device):
        return {name: torch_.full(shape, 0x5a, dtype=dtype, device=device) for name, (shape, dtype) in specs.items()}

    monkeypatch.setattr(encoding._device, "_outputs", outputs)
    caps = exact_capacities(encoder.encode(case.rows[:n]), factor=2)
    caps["graphs"] = 64
    ring = encoder.device_stream(64)
    relations = encoder.schema.num_relations
    for count in (n, 5, 0):
        if count:
            ring.append(to_device(case.rows[:count], dev))
        batch = ring.flush(capacities=caps)
        host = encoder.encode(case.rows[:count])
        same_batch(host, batch)
        assert int(batch.device_graphs.item()) == count == batch.num_graphs
        sizes = batch.sizes
        per_graph = {"task_ids": count, "num_objects": count, "entity_offsets": count + 1, "goal_complete": count,
                     "target_offsets": count + 1, "relation_counts": count, "relation_offsets": count * relations + 1,
                     "relation_arg_offsets": count * relations + 1}
        for name, start in per_graph.items():
            if name in batch._arrays:
                assert not torch.count_nonzero(batch._arrays[name][start:]).item(), (count, name)
        # the flush writes nothing past the sizes of the variable arrays
        variable = {"entity_role": sizes["entities"], "entity_ref": sizes["entities"], "relation_args": sizes["args"],
                    "target_entity": sizes["targets"], "target_group": sizes["targets"]}
        for name, start in variable.items():
            if name in batch._arrays:
                assert bool((batch._arrays[name][start:] == 0x5a).all()), (count, name)


@pytest.mark.parametrize("option", [{"max_goal_level": 1}, {"history": True}])
def test_device_row_stream_refuses_subgoal_layers_and_history(cases, option):
    gpu()
    encoder = encoding.FlatRelationEncoder(cases["gripper"].source, **option)
    with pytest.raises(ValueError, match="^mymyr: encoding: a device stream carries no subgoal layers or history steps"):
        encoder.device_stream(8)


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
    other = encoding.FlatRelationEncoder(rl.TaskTable(list(case.tasks) * 2),
                                         layout=layout, index_dtype=index, target_sources=("goal",),
                                         predicate_virtual_nodes=True)
    assert other.table_fingerprint != encoder.table_fingerprint
    foreign = other.encode(rows[:3], torch.zeros(3, dtype=torch.int32, device=dev))
    with pytest.raises(ValueError, match="^mymyr: encoding: collate received batches of different tables"):
        encoding.DeviceCollator([batches[0], foreign])
    wide = {**exact_capacities(host, factor=2), "graphs": n + 9}
    tails = plan.encode(capacities=wide)
    same_batch(encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n], capacities=wide), tails)
    for name in ("task_ids", "num_objects", "entity_offsets", "relation_offsets", "relation_arg_offsets"):
        start = n + 1 if name == "entity_offsets" else n * encoder.schema.num_relations + 1 if "relation" in name else n
        assert not torch.count_nonzero(tails._arrays[name][start:]).item(), name
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
