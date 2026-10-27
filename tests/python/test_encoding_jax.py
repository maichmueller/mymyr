"""mymyr.encoding.jax: the XLA FFI target mymyr_encode_flat on the CPU and (with a visible GPU) on CUDA gives the host
encoder's capacity form bit for bit, in both layouts and index types, for batches of 0 to 4097 rows, mixed instances,
row goals and derived words, with exact, ample, short and zero capacities; jit equals eager, lax.scan and jax.vmap
work, command buffers give the same arrays, and a compiled function outlives every reference to its encoder.

Without JAX (or a mymyr built without the XLA FFI headers) the module is skipped. The CUDA cases run only when a GPU
is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_encoding_jax.py
"""

import contextlib
import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
import types

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from mymyr.encoding import _device
from conftest import ROOT, text_task, walk

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from mymyr._core import _encoding_jax, _rl_jax  # noqa: E402

if not _rl_jax.has_ffi():
    pytest.skip("mymyr was built without the XLA FFI headers", allow_module_level=True)

import mymyr.encoding.jax as ej  # noqa: E402

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


def platforms():
    out = ["cpu"]
    try:
        if jax.devices("gpu") and _encoding_jax.has_cuda():
            out.append("cuda")
    except RuntimeError:
        pass
    return out


PLATFORMS = platforms()


@pytest.fixture(autouse=True)
def release_encoders():
    """Unregisters the encoders a test registered and drops its compiled functions, which frees their device contexts
    (the registry keeps an encoder until it is unregistered)."""
    before = set(ej._registrations)
    yield
    for handle in set(ej._registrations) - before:
        ej.unregister(handle)
    jax.clear_caches()
    gc.collect()


def device(platform):
    return jax.devices("gpu")[0] if platform == "cuda" else jax.devices("cpu")[0]


def need_gpu():
    if "cuda" not in PLATFORMS:
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES) or a mymyr without the CUDA backend")


@contextlib.contextmanager
def x64(on):
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def halves(words):
    """u64 words as the uint32 halves JAX takes."""
    return None if words is None else np.ascontiguousarray(words).view(np.uint32)


def put(array, platform):
    return None if array is None else jax.device_put(array, device(platform))


def pddl_task(domain, problem, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(PDDL / domain / "domain.pddl", PDDL / domain / f"{problem}.pddl", **options)


class Case:
    """Rows of a task or table of instances: the state words, the instance of each row and, with derived atoms, the derived
    words of each row (the successors of walked states, which rl.expand gives with theirs)."""

    def __init__(self, source, tasks, derived=False, steps=8):
        self.source, self.tasks = source, tasks
        self.table = rl.as_table(source) if isinstance(source, mymyr.Task) else source
        rows, ids, dwords = [], [], []
        for index, task in enumerate(tasks):
            states = walk(task, steps=steps, walks=3, seed=index)
            raw = np.asarray(task.encode(states))
            padded = np.zeros((len(states), self.table.words), np.uint64)
            padded[:, : raw.shape[1]] = raw
            if derived:
                x = rl.expand(self.table, padded, np.full(len(states), index, np.int32) if len(tasks) > 1 else None,
                              derived=True)
                succ, derived_rows = np.asarray(x.succ), np.asarray(x.derived)
                padded = np.zeros((len(succ), self.table.words), np.uint64)  # as wide as the table's widest instance
                padded[:, : succ.shape[1]] = succ
                wide = np.zeros((len(succ), self.table.derived_words), np.uint64)
                wide[:, : derived_rows.shape[1]] = derived_rows
                dwords.append(wide)
            ids += [index] * len(padded)
            rows.append(padded)
        self.rows = np.ascontiguousarray(np.concatenate(rows))
        self.ids = np.asarray(ids, np.int32) if len(tasks) > 1 else None
        self.derived = np.ascontiguousarray(np.concatenate(dwords)) if derived else None

    @property
    def n(self):
        return len(self.rows)

    def resized(self, n):
        """The rows repeated or cut to n rows (the instances mixed as they come)."""
        out = Case.__new__(Case)
        out.source, out.tasks, out.table = self.source, self.tasks, self.table
        pick = np.arange(n) % self.n
        out.rows = np.ascontiguousarray(self.rows[pick])
        out.ids = None if self.ids is None else np.ascontiguousarray(self.ids[pick])
        out.derived = None if self.derived is None else np.ascontiguousarray(self.derived[pick])
        return out


@pytest.fixture(scope="module")
def cases():
    out = {}
    gripper = text_task("gripper__prob05")
    out["gripper"] = Case(gripper, [gripper])
    phil = text_task("philosophers__p03-phil4")
    out["philosophers"] = Case(phil, [phil], derived=True)
    if hasattr(mymyr, "Domain"):
        blocks = [pddl_task("blocks", p, atoms="dense") for p in ("probBLOCKS-8-0", "smedium")]
        out["blocks_table"] = Case(rl.TaskTable(blocks), blocks)
        phils = [pddl_task("philosophers", p, atoms="dense") for p in ("p03-phil4", "p03-phil4-one-blocked")]
        out["philosophers_table"] = Case(rl.TaskTable(phils), phils, derived=True)
    return out


def host_encode(encoder, case, goals=None, **kwargs):
    return encoder.encode(case.rows, case.ids, derived=case.derived, goals=goals, **kwargs)


def jax_inputs(case, platform, goals=None):
    g = None if goals is None else tuple(put(halves(m), platform) for m in goals)
    return put(halves(case.rows), platform), put(case.ids, platform), put(halves(case.derived), platform), g


def layered_inputs(platform, subgoals=None, history=None):
    """The ``subgoals`` and ``history`` arguments of ``ej.encode`` on ``platform``: the masks as uint32 halves, dt and
    count as they are."""

    def one(x):
        return None if x is None else put(halves(x) if x.dtype == np.uint64 else x, platform)

    return {key: None if value is None else tuple(one(x) for x in value)
            for key, value in (("subgoals", subgoals), ("history", history))}


def jax_encode(encoder, case, platform, goals=None, jit=True, subgoals=None, history=None, **kwargs):
    rows, ids, derived, g = jax_inputs(case, platform, goals)
    layers = layered_inputs(platform, subgoals, history)

    def f(rows, ids, derived, g, layers):
        return ej.encode(encoder, rows, ids, derived=derived, goals=g, **layers, **kwargs)

    return (jax.jit(f) if jit else f)(rows, ids, derived, g, layers)


def same_batch(host, batch):
    """The arrays and scalars of two batches: the same names, dtypes and bytes."""
    h, d = host.as_dict(), batch.as_dict("numpy")
    assert h.keys() == d.keys()
    for key in h:
        a, b = np.asarray(h[key]), np.asarray(d[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert np.array_equal(a, b), key
    assert host.num_graphs == batch.num_graphs and host.num_relations == batch.num_relations
    assert host.implicit_entities == batch.implicit_entities


def same_sizes(host, batch):
    got = batch.sizes
    assert got.pop("invalid_input") is False
    want = {k: v for k, v in host.sizes.items() if k != "invalid_input"}
    assert got == want


def exact_capacities(batch, factor=1):
    sizes = {
        "entities": int(batch.entity_offsets[-1]) if batch.num_graphs else 0,
        "graphs": batch.num_graphs,
        "args": int(batch.relation_arg_offsets[-1]) if batch.num_graphs else 0,
        "targets": 0,
    }
    if hasattr(batch, "target_offsets") and batch.num_graphs:
        sizes["targets"] = int(batch.target_offsets[-1])
    if hasattr(batch, "history_offsets"):
        sizes["history"] = int(batch.history_offsets[-1]) if batch.num_graphs else 0
    return {k: v * factor for k, v in sizes.items()}


def fluent_goals(case, seed=3):
    """Random goal masks over the fluent slots of every row: (goal_pos, goal_neg) [N, W]."""
    rng = np.random.default_rng(seed)
    pos = np.zeros(case.rows.shape, np.uint64)
    neg = np.zeros(case.rows.shape, np.uint64)
    offsets = np.asarray(case.table.atom_metadata()["atom_offsets"])
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


def layered_rows(case, layers=3, steps=4, seed=11):
    """Random subgoal layers and history steps for the rows of ``case`` as the masks of ``encode``: (subgoals,
    history). A row uses a random number of its layers (the others are empty) and has a random number of valid steps;
    the literals are random over its slots (over its derived atoms for the derived masks)."""
    rng = np.random.default_rng(seed)
    offsets = np.asarray(case.table.atom_metadata()["atom_offsets"])
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


# ------------------------------------------------------------------------------------------------ the targets


@pytest.mark.parametrize("options", OPTION_SETS.values(), ids=OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_the_targets_equal_the_capacity_form_bit_for_bit(cases, layout, index, options):
    with x64(index == "int64"):
        for case in cases.values():
            encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index, **options)
            caps = exact_capacities(host_encode(encoder, case))
            host = host_encode(encoder, case, capacities=caps)
            for platform in PLATFORMS:
                batch = jax_encode(encoder, case, platform, capacities=caps)
                assert isinstance(batch, ej.JaxFlatBatch)
                same_batch(host, batch)
                same_sizes(host, batch)
                assert all(a.devices() == {device(platform)} for a in batch.as_dict().values() if hasattr(a, "devices"))


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_batches_of_every_size_with_exact_ample_short_and_zero_capacities(cases, layout, index, n):
    with x64(index == "int64"):
        for name in ("gripper", "blocks_table", "philosophers_table"):
            if name not in cases:
                continue
            case = cases[name].resized(n)
            encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index,
                                                   **OPTION_SETS["goal_targets_virtual"])
            exact = exact_capacities(host_encode(encoder, case))
            choices = [exact, exact_capacities(host_encode(encoder, case), 2),
                       {"entities": 0, "graphs": n, "args": 0, "targets": 0}]
            if n:
                choices += [dict(exact, args=exact["args"] - 1), dict(exact, entities=exact["entities"] - 1)]
            for caps in choices:
                host = host_encode(encoder, case, capacities=caps)
                for platform in PLATFORMS:
                    batch = jax_encode(encoder, case, platform, capacities=caps)
                    same_batch(host, batch)
                    same_sizes(host, batch)
                    assert batch.relation_args.shape == (caps["args"],)
            if n:
                assert host_encode(encoder, case, capacities=choices[-1]).sizes["capacity_overflow"]


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("options", ["default", "goal_targets_virtual"])
def test_row_goals_equal_the_host(cases, layout, options):
    for name in ("gripper", "blocks_table"):
        if name not in cases:
            continue
        case = cases[name]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **OPTION_SETS[options])
        goals = fluent_goals(case)
        variants = [goals] + ([tuple(g[:1] for g in goals)] if case.ids is None else [])  # one row for all
        for g in variants:
            caps = exact_capacities(host_encode(encoder, case, g), 2)
            host = host_encode(encoder, case, g, capacities=caps)
            for platform in PLATFORMS:
                same_batch(host, jax_encode(encoder, case, platform, goals=g, capacities=caps))
        # 64-bit words are accepted as well as halves
        with x64(True):
            caps = exact_capacities(host_encode(encoder, case, goals))
            batch = ej.encode(encoder, jnp.asarray(case.rows), None if case.ids is None else jnp.asarray(case.ids),
                              goals=tuple(jnp.asarray(g) for g in goals), capacities=caps)
            same_batch(host_encode(encoder, case, goals, capacities=caps), batch)


def test_derived_words_and_derived_goal_masks_equal_the_host():
    task = pddl_task("goals", "conjunctive")
    table = rl.as_table(task)
    rows = np.ascontiguousarray(np.asarray(task.encode(walk(task, steps=6, walks=2))))
    x = rl.expand(table, rows, derived=True)
    case = Case.__new__(Case)
    case.source, case.tasks, case.table = task, [task], table
    case.rows, case.ids, case.derived = np.ascontiguousarray(np.asarray(x.succ)), None, np.asarray(x.derived)
    derived_slots = len(table.atom_metadata()["derived_pred"])
    assert derived_slots > 0
    zeros = np.zeros(case.rows.shape, np.uint64)
    dpos = np.zeros((case.n, max(1, (derived_slots + 63) // 64)), np.uint64)
    dpos[:, 0] = 1
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied", "unsatisfied"))
    goals = (zeros, zeros, dpos, np.zeros_like(dpos))
    for g in (None, goals):
        caps = exact_capacities(host_encode(encoder, case, g))
        host = host_encode(encoder, case, g, capacities=caps)
        for platform in PLATFORMS:
            same_batch(host, jax_encode(encoder, case, platform, goals=g, capacities=caps))


def test_jit_equals_eager_and_the_exact_path_equals_the_host(cases):
    for name, case in cases.items():
        encoder = encoding.FlatRelationEncoder(case.source, **OPTION_SETS["goal_targets_virtual"])
        exact = host_encode(encoder, case)
        caps = exact_capacities(exact, 2)
        for platform in PLATFORMS:
            jitted = jax_encode(encoder, case, platform, capacities=caps)
            eager = jax_encode(encoder, case, platform, capacities=caps, jit=False)
            same_batch(host_encode(encoder, case, capacities=caps), jitted)
            for a, b in zip(jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager), strict=True):
                assert np.array_equal(np.asarray(a), np.asarray(b)), name
            batch = jax_encode(encoder, case, platform, jit=False)  # without capacities: one wait, exact arrays
            same_batch(exact, batch)
            assert batch.sizes is None and batch.device_sizes is None and batch.overflow is None


def test_lax_scan_encodes_every_step(cases):
    case = cases["gripper"].resized(60)
    encoder = encoding.FlatRelationEncoder(case.source, layout="relation", target_sources=("goal",))
    steps = 5
    per_step = 12
    rows = halves(case.rows[: steps * per_step]).reshape(steps, per_step, -1)
    caps = exact_capacities(host_encode(encoder, case), 1)
    caps["graphs"] = per_step
    handle = ej.register(encoder)

    def body(carry, xs):
        batch = ej.encode(handle, xs, capacities=caps)
        return carry + batch.relation_args.sum(dtype=jnp.uint32), batch

    for platform in PLATFORMS:
        total, stacked = jax.jit(lambda r: jax.lax.scan(body, jnp.uint32(0), r))(put(rows, platform))
        assert stacked.relation_args.shape == (steps, caps["args"])
        for k in range(steps):
            step = case.resized(per_step)
            step.rows = case.rows[k * per_step:(k + 1) * per_step]
            host = host_encode(encoder, step, capacities=caps)
            for name in host.as_dict():
                if name in ej.JaxFlatBatch.array_names:
                    assert np.array_equal(np.asarray(getattr(stacked, name)[k]), np.asarray(getattr(host, name))), name
            assert ej._host_sizes(np.asarray(stacked.device_sizes[k]), False)["args"] == host.sizes["args"]
        sizes = stacked.sizes  # NumPy arrays over the stacked axis
        assert sizes["args"].shape == (steps,) and sizes["overflow"].shape == (steps,) and not sizes["overflow"].any()
        for k in range(steps):
            step = case.resized(per_step)
            step.rows = case.rows[k * per_step:(k + 1) * per_step]
            want = host_encode(encoder, step, capacities=caps).sizes
            for key in ("entities", "tuples", "args", "targets"):
                assert int(sizes[key][k]) == want[key], key


@pytest.mark.parametrize("name", ["gripper__prob05", "philosophers__p03-phil4"])
@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("platform", PLATFORMS)
def test_literal_mask_objects_are_pytrees_that_encode_under_jit(name, layered, platform):
    task = text_task(name)
    encoder = encoding.FlatRelationEncoder(task, max_goal_level=1)
    literals = [str(next(iter(task.initial_state.atoms())))]
    literals += [str(g) for g in task.encoding_goal if g.atom.kind == "derived"]
    masks = (encoding.subgoal_masks(task, [[literals], [[]]]) if layered else
             encoding.goal_masks(task, [literals, []]))
    masks.unsatisfiable[1] = True
    key = "subgoals" if layered else "goals"
    rows = np.asarray(task.encode([task.initial_state, task.initial_state]))
    expected = encoder.encode(rows, **{key: masks})
    caps = exact_capacities(expected)
    expected = encoder.encode(rows, capacities=caps, **{key: masks})
    leaves, tree = jax.tree_util.tree_flatten(masks)
    assert len(leaves) == len(masks) + 1
    converted = jax.tree_util.tree_unflatten(tree, [put(halves(a) if a.dtype == np.uint64 else a, platform)
                                                 for a in leaves])
    restored = jax.jit(lambda m: m)(converted)
    assert type(restored) is type(masks)
    np.testing.assert_array_equal(restored.unsatisfiable, masks.unsatisfiable)
    actual = jax.jit(lambda m: ej.encode(encoder, halves(rows), capacities=caps, **{key: m}))(converted)
    same_batch(expected, actual)


def test_vmap_encodes_each_batch_on_its_own(cases):
    case = cases["gripper"].resized(30)
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    caps["graphs"] = 10
    rows = halves(case.rows[:30]).reshape(3, 10, -1)
    for platform in PLATFORMS:
        batched = jax.jit(jax.vmap(lambda r: ej.encode(encoder, r, capacities=caps)))(put(rows, platform))
        assert batched.relation_args.shape == (3, caps["args"])
        sizes = batched.sizes
        assert sizes["args"].shape == (3,) and not sizes["capacity_overflow"].any() and not sizes["overflow"].any()
        for b in range(3):
            assert int(sizes["args"][b]) == ej.encode(encoder, put(rows[b], platform), capacities=caps).sizes["args"]
        for b in range(3):
            single = ej.encode(encoder, put(rows[b], platform), capacities=caps)
            for x, y in zip(jax.tree_util.tree_leaves(single), jax.tree_util.tree_leaves(batched), strict=True):
                assert np.array_equal(np.asarray(x), np.asarray(y)[b])


def test_the_cpu_result_does_not_depend_on_threads(cases):
    case = cases["gripper"].resized(4097)
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    one = jax_encode(encoder, case, "cpu", capacities=caps, threads=1)
    for threads in (0, 4):
        many = jax_encode(encoder, case, "cpu", capacities=caps, threads=threads)
        for x, y in zip(jax.tree_util.tree_leaves(one), jax.tree_util.tree_leaves(many), strict=True):
            assert np.array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------------------------------------------------ subgoal layers and history


@pytest.mark.parametrize("options", LAYERED_OPTION_SETS.values(), ids=LAYERED_OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("policy", [None, "every", "lowest", "highest"])
def test_subgoal_layers_and_history_equal_the_host_bit_for_bit(cases, layout, index, options, policy):
    with x64(index == "int64"):
        for case in cases.values():
            encoder = encoding.FlatRelationEncoder(case.source, layout=layout, index_dtype=index,
                **({"repeated_literals": policy} if policy is not None else {}), **options)
            kwargs = layered_kwargs(case, options)
            exact = host_encode(encoder, case, **kwargs)
            caps = exact_capacities(exact)
            assert ("history" in caps) == bool(options.get("history"))
            host = host_encode(encoder, case, capacities=caps, **kwargs)
            variants = [kwargs]
            # a history without counts (every step is valid) and fewer layers than the encoder has levels
            if "history" in kwargs:
                pos, neg, dt, _count, *derived = kwargs["history"]
                variants.append({**kwargs, "history": (pos, neg, dt, None, *derived)})
            if "subgoals" in kwargs:
                variants.append({**kwargs, "subgoals": tuple(np.ascontiguousarray(x[:, :1]) for x in kwargs["subgoals"])})
            for platform in PLATFORMS:
                batch = jax_encode(encoder, case, platform, capacities=caps, **kwargs)
                same_batch(host, batch)
                same_sizes(host, batch)
                if options.get("history"):
                    assert batch.history_dt.dtype == jnp.int32 and batch.history_entity.shape == (caps["history"],)
                    assert batch.history_offsets.shape == (case.n + 1,)
                # without capacities: one wait, the exact arrays
                same_batch(exact, jax_encode(encoder, case, platform, jit=False, **kwargs))
                for variant in variants[1:]:
                    want = host_encode(encoder, case, **variant)
                    roomy = exact_capacities(want, 2)
                    same_batch(host_encode(encoder, case, capacities=roomy, **variant),
                               jax_encode(encoder, case, platform, capacities=roomy, **variant))
                    same_batch(want, jax_encode(encoder, case, platform, jit=False, **variant))


@pytest.mark.parametrize("options", LAYERED_OPTION_SETS.values(), ids=LAYERED_OPTION_SETS.keys())
@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single_batches_with_layers_and_history_equal_the_host(cases, options, n):
    for case in cases.values():
        case = case.resized(n)
        encoder = encoding.FlatRelationEncoder(case.source, **options)
        kwargs = layered_kwargs(case, options)
        exact = host_encode(encoder, case, **kwargs)
        caps = exact_capacities(exact)
        host = host_encode(encoder, case, capacities=caps, **kwargs)
        for platform in PLATFORMS:
            same_batch(host, jax_encode(encoder, case, platform, capacities=caps, **kwargs))
            same_batch(exact, jax_encode(encoder, case, platform, jit=False, **kwargs))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("policy", [None, "every", "lowest", "highest"])
def test_subgoal_layers_and_history_with_row_goals_and_shared_layers_equal_the_host(cases, layout, policy):
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    for name in ("gripper", "blocks_table", "philosophers_table"):
        if name not in cases:
            continue
        case = cases[name]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout,
            **({"repeated_literals": policy} if policy is not None else {}), **options)
        goals = fluent_goals(case)
        kwargs = layered_kwargs(case, options)
        variants = [kwargs]
        if case.ids is None:  # layers that every row shares
            variants.append({**kwargs, "subgoals": tuple(np.ascontiguousarray(x[:1]) for x in kwargs["subgoals"])})
        for variant in variants:
            caps = exact_capacities(host_encode(encoder, case, goals, **variant), 2)
            host = host_encode(encoder, case, goals, capacities=caps, **variant)
            for platform in PLATFORMS:
                same_batch(host, jax_encode(encoder, case, platform, goals, capacities=caps, **variant))
        # 64-bit words are accepted as well as halves, and int64 dt and counts
        with x64(True):
            caps = exact_capacities(host_encode(encoder, case, goals, **kwargs))
            pos, neg, dt, count, *derived = kwargs["history"]
            history = tuple(jnp.asarray(x) for x in (pos, neg, dt.astype(np.int64), count.astype(np.int64), *derived))
            batch = ej.encode(encoder, jnp.asarray(case.rows), None if case.ids is None else jnp.asarray(case.ids),
                              derived=None if case.derived is None else jnp.asarray(case.derived),
                              goals=tuple(jnp.asarray(g) for g in goals),
                              subgoals=tuple(jnp.asarray(x) for x in kwargs["subgoals"]), history=history,
                              capacities=caps)
            same_batch(host_encode(encoder, case, goals, capacities=caps, **kwargs), batch)


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_the_capacity_form_with_history_equals_the_host(cases, layout):
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    for name in ("gripper", "blocks_table", "philosophers_table"):
        if name not in cases:
            continue
        case = cases[name]
        encoder = encoding.FlatRelationEncoder(case.source, layout=layout, **options)
        kwargs = layered_kwargs(case, options)
        exact = exact_capacities(host_encode(encoder, case, **kwargs))
        assert set(exact) == {"graphs", "entities", "args", "targets", "history"} and exact["history"] > 0
        short = dict(exact, history=exact["history"] - 1)
        zero = {"entities": 0, "graphs": case.n, "args": 0, "targets": 0, "history": 0}
        for caps in (exact, exact_capacities(host_encode(encoder, case, **kwargs), 2), short, zero):
            host = host_encode(encoder, case, capacities=caps, **kwargs)
            for platform in PLATFORMS:
                batch = jax_encode(encoder, case, platform, capacities=caps, **kwargs)
                same_batch(host, batch)
                same_sizes(host, batch)
                assert batch.history_entity.shape == (caps["history"],)
                assert batch.sizes["history"] == int(host_encode(encoder, case, **kwargs).history_offsets[-1])
        assert host_encode(encoder, case, capacities=short, **kwargs).sizes["capacity_overflow"]


def test_eager_calls_grow_the_held_history_capacity(cases):
    case = cases["gripper"].resized(8)
    options = LAYERED_OPTION_SETS["history"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    handle = ej.register(encoder)
    few = layered_kwargs(case, options, steps=1)
    many = layered_kwargs(case, options, steps=40, seed=5)
    rows = put(halves(case.rows), "cpu")
    for kwargs in (few, many, few):
        history = layered_inputs("cpu", history=kwargs["history"])["history"]
        batch = ej.encode(handle, rows, history=history)
        same_batch(host_encode(encoder, case, **kwargs), batch)
    held = ej._registrations[handle].held[8]
    assert len(held) == 4 and held[3] >= int(host_encode(encoder, case, **many).history_offsets[-1])


def test_lax_scan_and_vmap_carry_layers_and_history(cases):
    case = cases["gripper"].resized(30)
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    kwargs = layered_kwargs(case, options)
    caps = exact_capacities(host_encode(encoder, case, **kwargs))
    caps["graphs"] = 10

    def split(x):
        return np.ascontiguousarray(x.reshape(3, 10, *x.shape[1:]))

    def part(b):
        step = case.resized(10)
        step.rows = case.rows[10 * b:10 * (b + 1)]
        return step, {key: tuple(np.ascontiguousarray(x[10 * b:10 * (b + 1)]) for x in value)
                      for key, value in kwargs.items()}

    rows = split(halves(case.rows))
    stacked = {key: tuple(split(halves(x) if x.dtype == np.uint64 else x) for x in value) for key, value in kwargs.items()}
    for platform in PLATFORMS:
        args = (put(rows, platform), {key: tuple(put(x, platform) for x in value) for key, value in stacked.items()})

        def one(r, layers):
            return ej.encode(encoder, r, **layers, capacities=caps)

        mapped = jax.jit(jax.vmap(one))(*args)
        scanned = jax.jit(lambda r, layers: jax.lax.scan(lambda c, xs: (c, one(*xs)), 0, (r, layers))[1])(*args)
        for batched in (mapped, scanned):
            assert batched.history_entity.shape == (3, caps["history"])
            assert not batched.sizes["overflow"].any()
            for b in range(3):
                step, layers = part(b)
                host = host_encode(encoder, step, capacities=caps, **layers)
                for name, a in host.as_dict().items():
                    if name in ej.JaxFlatBatch.array_names:
                        assert np.array_equal(np.asarray(getattr(batched, name)[b]), np.asarray(a)), name
                assert int(batched.sizes["history"][b]) == host.sizes["history"]


def test_layers_and_history_the_schema_does_not_have_or_malformed_ones_are_refused(cases):
    case = cases["gripper"]
    subgoals, history = layered_rows(case)
    plain = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(plain, case))
    for platform in PLATFORMS:
        def refused(match, encoder=plain, capacities=caps, **kwargs):
            with pytest.raises(ValueError, match="^mymyr: encoding: " + match):
                jax_encode(encoder, case, platform, capacities=capacities, **kwargs)

        refused("subgoal layers are given, but the schema has no subgoal relations", subgoals=subgoals)
        refused("history steps are given, but the schema has no history relations", history=history)
        levels = encoding.FlatRelationEncoder(case.source, max_goal_level=2)
        refused("3 subgoal layers are given, the schema has 2 levels", levels, subgoals=subgoals)
        steps = encoding.FlatRelationEncoder(case.source, history=True)
        steps_caps = {**caps, "history": 64}
        refused("the schema has history: capacities needs the key 'history'", steps, history=history)
        refused("subgoals must be", levels, subgoals=subgoals[:1])
        refused("history must be", steps, steps_caps, history=history[:2])
        refused("'subgoals\\[1\\] \\(neg\\)' has another shape than 'subgoals\\[0\\] \\(pos\\)'", levels,
                subgoals=(subgoals[0][:, :2], subgoals[1][:, :1]))
        refused("'subgoals\\[0\\] \\(pos\\)' has 2 rows: one per state", levels,
                subgoals=(subgoals[0][:2, :2], subgoals[1][:2, :2]))
        refused("'history\\[0\\] \\(pos\\)' has 1 rows: one per state", steps, steps_caps,
                history=(history[0][:1], history[1][:1], history[2][:1]))
        refused("'history\\[2\\] \\(dt\\)' must be", steps, steps_caps, history=(history[0], history[1], history[2][:, :2]))
        refused("'history\\[3\\] \\(count\\)' must be", steps, steps_caps,
                history=(history[0], history[1], history[2], history[3][:3]))
        refused("'subgoals\\[0\\] \\(pos\\)' must be", levels, subgoals=(np.zeros((case.n, 2), np.uint64),) * 2)
        # an encoder with layers or history encodes the rows without them
        for encoder, c in ((levels, caps), (steps, steps_caps)):
            same_batch(host_encode(encoder, case, capacities=c), jax_encode(encoder, case, platform, capacities=c))
    with pytest.raises(ValueError, match=r"^mymyr: encoding: capacities are required under jax.jit"):
        jax_encode(steps, case, "cpu", history=history)


def test_invalid_history_counts_and_layer_bits_are_flagged_or_raise(cases):
    case = cases["blocks_table"] if "blocks_table" in cases else cases["gripper"]
    options = LAYERED_OPTION_SETS["levels_history_virtual"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    kwargs = layered_kwargs(case, options)
    caps = exact_capacities(host_encode(encoder, case, **kwargs))
    steps = kwargs["history"][2].shape[1]
    bad = []
    for count in (steps + 1, -1):
        bad_count = kwargs["history"][3].copy()
        bad_count[3] = count
        bad.append({**kwargs, "history": (*kwargs["history"][:3], bad_count, *kwargs["history"][4:])})
    # a subgoal bit past the slots of an instance
    pos = kwargs["subgoals"][0].copy()
    offsets = np.asarray(case.table.atom_metadata()["atom_offsets"])
    ids = np.zeros(case.n, np.int64) if case.ids is None else case.ids
    row = next(r for r in range(case.n) if int(offsets[ids[r] + 1] - offsets[ids[r]]) < 64 * case.rows.shape[1])
    slots = int(offsets[ids[row] + 1] - offsets[ids[row]])
    pos[row, 0, slots // 64] |= np.uint64(1) << np.uint64(slots % 64)
    bad.append({**kwargs, "subgoals": (pos, *kwargs["subgoals"][1:])})
    for platform in PLATFORMS:
        for b in bad:
            batch = jax_encode(encoder, case, platform, capacities=caps, **b)
            sizes = batch.sizes
            assert sizes["overflow"] and sizes["invalid_input"] and not sizes["capacity_overflow"]
            assert all(not np.asarray(a).any() for a in batch.as_dict().values() if hasattr(a, "shape"))
            with pytest.raises(ValueError, match=r"^mymyr: encoding: a row is invalid"):
                jax_encode(encoder, case, platform, jit=False, **b)
        assert jax_encode(encoder, case, platform, capacities=caps, **kwargs).sizes["overflow"] is False


@pytest.mark.parametrize("dtype", ["int64", "uint64", "uint32"])
@pytest.mark.parametrize("which", ["dt", "count"])
def test_history_integers_that_int32_cannot_hold_raise_or_flag_their_row(cases, which, dtype):
    # 2**32 + v narrows to v: a count or dt that wraps into the valid range is refused all the same
    case = cases["gripper"]
    options = LAYERED_OPTION_SETS["history"]
    encoder = encoding.FlatRelationEncoder(case.source, **options)
    pos, neg, dt, count = layered_kwargs(case, options)["history"]
    k = 2 if which == "dt" else 3
    fits = np.abs((dt, count)[k - 2]).astype(dtype)
    wide = fits.copy()
    at = (3, 0) if which == "dt" else 3
    wide[at] += 2**31 if dtype == "uint32" else 2**32
    with_fits = (pos, neg, *((fits, count) if k == 2 else (dt, fits)))
    with_wide = (pos, neg, *((wide, count) if k == 2 else (dt, wide)))
    # the forms of the history: with its count, and for dt also without (every step valid)
    forms = [with_wide] + ([with_wide[:3]] if k == 2 else [])
    narrow = (pos, neg, *(x.astype(np.int32) for x in with_fits[2:]))
    caps = exact_capacities(host_encode(encoder, case, history=narrow[:3]))  # every step valid: the most history
    message = rf"^mymyr: 'history\[{k}\] \({which}\)' has a value that int32 cannot hold"
    for form in forms:
        with pytest.raises(ValueError, match=message):
            host_encode(encoder, case, history=form)
    rows = halves(case.rows)

    def on(history, platform):
        return (*(put(halves(m), platform) for m in history[:2]), *(put(x, platform) for x in history[2:]))

    def f(rows, history):
        return ej.encode(encoder, rows, history=history, capacities=caps)

    with x64(dtype != "uint32"):
        for platform in PLATFORMS:
            for form in forms:
                device_form = on(form, platform)
                assert device_form[k].dtype == np.dtype(dtype)
                # concrete NumPy or JAX arrays raise, with and without capacities
                for given in (form, device_form):
                    for c in (None, caps):
                        with pytest.raises(ValueError, match=message):
                            ej.encode(encoder, put(rows, platform), history=given, capacities=c)
                # traced ones make the row invalid
                batch = jax.jit(f)(put(rows, platform), device_form)
                sizes = batch.sizes
                assert sizes["overflow"] and sizes["invalid_input"] and not sizes["capacity_overflow"]
                assert all(not np.asarray(a).any() for a in batch.as_dict().values() if hasattr(a, "shape"))
            # the same dtype within int32 encodes as int32 does
            host = host_encode(encoder, case, history=narrow, capacities=caps)
            same_batch(host, jax.jit(f)(put(rows, platform), on(with_fits, platform)))


@pytest.mark.parametrize("history", [False, True])
def test_an_empty_history_with_counts_equals_the_host_on_every_platform(cases, history):
    # H = 0: no count, zero counts and a count past the zero steps
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, history=history)
    n, words = case.rows.shape
    masks = np.zeros((n, 0, words), np.uint64)
    dt = np.zeros((n, 0), np.int32)
    past = np.zeros(n, np.int32)
    past[3] = 1
    caps = exact_capacities(host_encode(encoder, case))
    if history:
        caps["history"] = 64
    for count in (None, np.zeros(n, np.int32)):
        steps = (masks, masks, dt, count)
        host = host_encode(encoder, case, history=steps)
        for platform in PLATFORMS:
            same_batch(host, jax_encode(encoder, case, platform, jit=False, history=steps))
            same_batch(host_encode(encoder, case, history=steps, capacities=caps),
                       jax_encode(encoder, case, platform, capacities=caps, history=steps))
    steps = (masks, masks, dt, past)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the history count of row 3 is 1, outside \[0, 0\]"):
        host_encode(encoder, case, history=steps)
    for platform in PLATFORMS:
        sizes = jax_encode(encoder, case, platform, capacities=caps, history=steps).sizes
        assert sizes["overflow"] and sizes["invalid_input"] and not sizes["capacity_overflow"]
        with pytest.raises(ValueError, match=r"^mymyr: encoding: a row is invalid"):
            jax_encode(encoder, case, platform, jit=False, history=steps)


# ------------------------------------------------------------------------------------------------ errors and flags


def test_invalid_rows_are_flagged_with_capacities_and_raise_without(cases):
    case = cases["blocks_table"] if "blocks_table" in cases else None
    if case is None:
        pytest.skip("built without the loki front end")
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    bad = case.resized(case.n)
    bad.ids = bad.ids.copy()
    bad.ids[3] = 7
    for platform in PLATFORMS:
        batch = jax_encode(encoder, bad, platform, capacities=caps)
        sizes = batch.sizes
        assert sizes["overflow"] and sizes["invalid_input"]
        assert int(batch.overflow) & 4
        assert all(not np.asarray(a).any() for a in batch.as_dict().values() if hasattr(a, "shape"))
        with pytest.raises(ValueError, match=r"^mymyr: encoding: a row is invalid"):
            jax_encode(encoder, bad, platform, jit=False)


def test_unsupported_calls_raise_value_error_when_traced(cases):
    case = cases["gripper"]
    rows = halves(case.rows)
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    with pytest.raises(ValueError, match=r"^mymyr: encoding: capacities are required under jax.jit"):
        jax.jit(lambda r: ej.encode(encoder, r))(rows)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: index_dtype='int64' needs 64-bit JAX arrays"):
        ej.encode(encoding.FlatRelationEncoder(case.source, index_dtype="int64"), rows, capacities=caps)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: goals are given, but the schema has no goal relations"):
        plain = encoding.FlatRelationEncoder(case.source, goal_derivations=())
        ej.encode(plain, rows, goals=(rows, rows), capacities=caps)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: 'states' must hold"):
        ej.encode(encoder, np.zeros((2, 4), np.float32), capacities=caps)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the batch has 27 graphs, more than the capacity 3"):
        ej.encode(encoder, rows[:27], capacities=dict(caps, graphs=3))
    with pytest.raises(ValueError, match=r"^mymyr: encoding: capacities needs the keys"):
        ej.encode(encoder, rows, capacities={"args": 1})
    with pytest.raises(ValueError, match=r"^mymyr: encoding: threads must be an int"):
        ej.encode(encoder, rows, capacities=caps, threads=-1)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: no encoder is registered under handle -5"):
        ej.encode(-5, rows, capacities=caps)
    if "blocks_table" in cases:
        table = cases["blocks_table"]
        enc = encoding.FlatRelationEncoder(table.source)
        with pytest.raises(ValueError, match=r"^mymyr: encoding: task_ids are required for a table of several"):
            jax.jit(lambda r: ej.encode(enc, r, capacities=caps))(halves(table.rows))


def test_shared_statics_are_refused_before_tracing_on_every_platform(cases):
    case = cases["gripper"]
    rows = halves(case.rows)
    encoder = encoding.FlatRelationEncoder(case.source, statics="shared")
    caps = exact_capacities(host_encode(encoding.FlatRelationEncoder(case.source), case))
    message = r"^mymyr: encoding: the JAX FFI does not encode shared statics"
    with pytest.raises(ValueError, match=message):
        ej.register(encoder)
    assert ej.registered(encoder) is None
    with pytest.raises(ValueError, match=message):
        ej.encode(encoder, rows, capacities=caps)
    with pytest.raises(ValueError, match=message):
        jax.jit(lambda r: ej.encode(encoder, r, capacities=caps))(rows)
    assert ej.registered(encoder) is None


@pytest.mark.parametrize("lane", ["numeric", "numeric_table", "simple", "simple_numeric_table"])
def test_numeric_relations_and_the_simple_graph_are_refused_before_tracing_on_every_platform(cases, lane):
    if lane == "simple":
        task = cases["gripper"].source
        encoder = encoding.SimpleEncoder(task)
        message = r"^mymyr: encoding: the JAX FFI does not encode Simple: its fixed result layout has no node or edge arrays"
    else:
        # a numeric table is refused first, whatever the outputs
        task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
        outputs = ("simple",) if lane == "simple_numeric_table" else ("relations",)
        encoder = encoding.FlatRelationEncoder(task, include_numeric=lane == "numeric", outputs=outputs)
        message = r"^mymyr: encoding: the JAX FFI has no numeric output buffer"
    rows = halves(np.asarray(task.encode([task.initial_state])))
    caps = dict(graphs=1, entities=64, args=64, targets=0)
    with pytest.raises(ValueError, match=message):
        ej.register(encoder)
    assert ej.registered(encoder) is None
    with pytest.raises(ValueError, match=message):
        ej.encode(encoder, rows, capacities=caps)
    with pytest.raises(ValueError, match=message):
        jax.jit(lambda r: ej.encode(encoder, r, capacities=caps))(rows)
    assert ej.registered(encoder) is None


# ------------------------------------------------------------------------------------------------ batches and handles


def test_a_jax_batch_has_the_accessors_of_a_host_batch(cases):
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, emit_counts=False, entity_arrays="auto", include_actions=False,
                                           goal_derivations=())
    host = host_encode(encoder, case)
    caps = exact_capacities(host)
    batch = jax_encode(encoder, case, "cpu", capacities=caps)
    assert batch.num_graphs == host.num_graphs and batch.num_relations == host.num_relations
    assert batch.schema.num_relations == encoder.schema.num_relations
    assert batch.as_dict()["schema_fingerprint"] == host.as_dict()["schema_fingerprint"]
    assert batch.implicit_entities and host.implicit_entities
    for name in ("relation_counts", "entity_role", "entity_ref", "goal_complete", "target_offsets"):
        with pytest.raises(AttributeError, match=f"this batch has no {name}"):
            getattr(batch, name)
    assert batch.device_sizes.dtype == jnp.uint32 and batch.device_sizes.shape == (len(_device._SIZE_FIELDS), 2)
    assert int(batch.overflow) == 0
    assert set(batch.as_dict()) == set(host.as_dict())
    with pytest.raises(ValueError, match="exports to 'jax', 'dlpack' or 'numpy'"):
        batch.as_dict("torch")
    leaves, tree = jax.tree_util.tree_flatten(batch)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    same_batch(host_encode(encoder, case, capacities=caps), again)


def test_a_compiled_function_outlives_every_reference_to_its_encoder(cases):
    case = cases["gripper"]
    caps = exact_capacities(host_encode(encoding.FlatRelationEncoder(case.source), case))
    want = host_encode(encoding.FlatRelationEncoder(case.source), case, capacities=caps)
    for platform in PLATFORMS:
        encoder = encoding.FlatRelationEncoder(case.source)
        handle = ej.register(encoder)
        assert ej.register(encoder) == handle and ej.registered(encoder) == handle
        f = jax.jit(lambda r: ej.encode(handle, r, capacities=caps))
        rows = put(halves(case.rows), platform)
        same_batch(want, f(rows))
        del encoder
        gc.collect()
        same_batch(want, f(rows))  # the registry holds the encoder
        assert _encoding_jax.calls(handle)["cuda" if platform == "cuda" else "cpu"] == 2
        ej.unregister(handle)
        assert ej.registered(encoding.FlatRelationEncoder(case.source)) is None
        same_batch(want, f(rows))  # and so does the compiled function
        with pytest.raises(ValueError, match="no encoder is registered under handle"):
            f(rows[:5])  # a new trace looks the handle up


# ------------------------------------------------------------------------------------------------ CUDA


def test_the_cuda_target_runs_on_xla_memory_only(cases):
    """The CUDA target takes its scratch from the call's results: the scratch result holds the device encoder's
    scratch plus alignment slack, and the call runs without the registry's device workspaces."""
    need_gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    caps = exact_capacities(host_encode(encoder, case))
    lowered = jax.jit(lambda r: ej.encode(handle, r, capacities=caps)).lower(put(halves(case.rows), "cuda"))
    text = lowered.as_text()
    assert "mymyr_encode_flat" in text
    need = _encoding_jax.scratch_bytes(handle, case.n, 0, False)
    assert f"tensor<{need + 255}xui8>" in text
    # a CPU lowering of the same function carries no device scratch
    cpu_text = jax.jit(lambda r: ej.encode(handle, r, capacities=caps)).lower(put(halves(case.rows), "cpu")).as_text()
    assert "mymyr_encode_flat" in cpu_text and f"tensor<{need + 255}xui8>" not in cpu_text


COMMAND_BUFFER_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    import jax, jax.numpy as jnp
    sys.path.insert(0, {tests!r})
    from conftest import text_task, walk
    from mymyr import encoding
    from mymyr._core import _encoding_jax
    import mymyr.encoding.jax as ej

    task = text_task("gripper__prob05")
    rows = np.ascontiguousarray(np.asarray(task.encode(walk(task, steps=30, walks=4)))).view(np.uint32)[:120]
    out = []
    for layout in ("graph", "relation"):
        encoder = encoding.FlatRelationEncoder(task, layout=layout, target_sources=("goal",))
        handle = ej.register(encoder)
        host = encoder.encode(rows.view(np.uint64))
        caps = dict(graphs=40, entities=int(host.entity_offsets[-1]), args=int(host.relation_arg_offsets[-1]),
                    targets=int(host.target_offsets[-1]))
        gpu = jax.devices("gpu")[0]
        steps = jax.device_put(rows.reshape(3, 40, -1), gpu)

        def body(carry, xs):
            b = ej.encode(handle, xs, capacities=caps)
            return carry + jnp.sum(b.relation_args, dtype=jnp.uint32), b

        scan = jax.jit(lambda r: jax.lax.scan(body, jnp.uint32(0), r))
        single = jax.jit(lambda r: ej.encode(handle, r, capacities=caps))
        for _ in range(3):  # traced into command buffers, then replayed
            total, stacked = scan(steps)
            batch = single(steps[1])
        jax.block_until_ready((total, stacked, batch))
        for k in range(3):
            eager = ej.encode(handle, steps[k], capacities=caps)
            for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(stacked), strict=True):
                assert np.array_equal(np.asarray(a), np.asarray(b)[k]), layout
        for a, b in zip(jax.tree_util.tree_leaves(batch), jax.tree_util.tree_leaves(ej.encode(handle, steps[1], capacities=caps)), strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b)), layout
        out.append(_encoding_jax.calls(handle))
    print("calls", out)
    assert all(c["captured"] > 0 for c in out), out
    print("ok")
    """
)


def test_command_buffers_give_the_eager_arrays(cases):
    """With XLA's command buffers on for custom calls (and loops), the CUDA target is traced into CUDA graphs: its calls
    run under stream capture and the replays give the eager arrays."""
    need_gpu()
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE,CONDITIONAL"
                        " --xla_gpu_graph_min_graph_size=1").strip()
    script = COMMAND_BUFFER_SCRIPT.format(tests=str(ROOT / "tests/python"))
    run = subprocess.run([sys.executable, "-W", "error", "-c", script], env=env, capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
    assert run.stdout.strip().endswith("ok")


# ------------------------------------------------------------------------------------------------ sizes, devices, pools


def test_the_size_words_follow_the_native_size_fields():
    fields = _device._SIZE_FIELDS
    words = np.zeros((len(fields), 2), np.uint32)
    for k, name in enumerate(fields):
        words[k] = (100 + k, 0 if name != "targets" else 3)  # no size equals the flags; targets above 32 bits
    flags = fields.index("overflow")
    words[flags] = (_device._OVERFLOW_CAPACITY | _device._OVERFLOW_INPUT, 0)
    for history in (False, True):
        sizes = ej._host_sizes(words, history)
        # a schema without history has the keys of a host batch without it
        assert ("history" in sizes) == history
        for k, name in enumerate(fields):
            if name != "overflow" and name in sizes:
                assert sizes[name] == (100 + k) + ((3 << 32) if name == "targets" else 0), name
        assert sizes["overflow"] and sizes["capacity_overflow"] and sizes["invalid_input"]
        assert not sizes["index_overflow"]
    static = (("task_ids",), 1, 0, 0, 0, True, None, 0)
    batch = ej.JaxFlatBatch({"task_ids": jnp.zeros((2,), jnp.int32)}, jnp.asarray(words), static)
    assert int(batch.overflow) == _device._OVERFLOW_CAPACITY | _device._OVERFLOW_INPUT
    assert "history" not in batch.sizes
    static = (("task_ids", "history_offsets"), 1, 0, 0, 0, True, None, 0)
    batch = ej.JaxFlatBatch({"task_ids": jnp.zeros((2,), jnp.int32), "history_offsets": jnp.zeros((3,), jnp.int32)},
                            jnp.asarray(words), static)
    assert batch.sizes["history"] == 100 + fields.index("history")


def test_the_device_ordinal_is_the_local_hardware_id_even_when_it_is_zero():
    assert ej._ordinal(types.SimpleNamespace(local_hardware_id=0, id=5)) == 0
    assert ej._ordinal(types.SimpleNamespace(local_hardware_id=3, id=5)) == 3
    assert ej._ordinal(types.SimpleNamespace(id=5)) == 5
    assert ej._ordinal(types.SimpleNamespace(local_hardware_id=None, id=2)) == 2


def test_eager_calls_without_capacities_reuse_the_capacities_held_for_their_size(cases):
    case = cases["gripper"].resized(8)
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    want = host_encode(encoder, case)
    rows = put(halves(case.rows), "cpu")

    def calls():
        return _encoding_jax.calls(handle)["cpu"]

    same_batch(want, ej.encode(handle, rows))
    assert calls() == 2  # the first call counts the batch, then encodes it
    for k in range(3):
        same_batch(want, ej.encode(handle, rows))
        assert calls() == 3 + k  # one compiled call from then on
    # a batch larger than the held capacities is encoded again at capacities that hold it, and they stay
    reg = ej._registrations[handle]
    reg.held[8] = (0, 0, 0, 0)
    same_batch(want, ej.encode(handle, rows))
    assert calls() == 7
    assert all(h >= 64 for h in reg.held[8][:2])
    same_batch(want, ej.encode(handle, rows))
    assert calls() == 8


def test_the_flags_of_a_call_at_held_capacities_are_checked(cases):
    if "blocks_table" not in cases:
        pytest.skip("built without the loki front end")
    case = cases["blocks_table"].resized(8)
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    rows = put(halves(case.rows), "cpu")
    ej.encode(handle, rows, put(case.ids, "cpu"))  # the capacities are held now
    ids = case.ids.copy()
    ids[3] = 7
    with pytest.raises(ValueError, match=r"^mymyr: encoding: a row is invalid"):
        ej.encode(handle, rows, put(ids, "cpu"))


def test_the_cpu_pools_of_alternating_thread_counts_are_kept(cases):
    case = cases["gripper"].resized(64)
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    caps = exact_capacities(host_encode(encoder, case))
    rows = put(halves(case.rows), "cpu")
    want = host_encode(encoder, case, capacities=caps)
    for k in range(6):
        same_batch(want, ej.encode(handle, rows, capacities=caps, threads=1 + k % 2))
    assert _encoding_jax.calls(handle)["pools"] == 2


def test_a_batch_keeps_its_schema_after_the_encoder_is_unregistered(cases):
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    handle = ej.register(encoder)
    batch = ej.encode(handle, put(halves(case.rows), "cpu"), capacities=caps)
    schema = batch.schema
    ej.unregister(handle)
    del encoder
    gc.collect()
    assert batch.schema is schema  # held by the batch
    del batch
    jax.clear_caches()
    gc.collect()
    probe = object()
    assert sys.getrefcount(schema) == sys.getrefcount(probe)  # and by nothing else


def test_rows_made_on_xlas_stream_right_before_the_encode_are_complete(cases):
    need_gpu()
    case = cases["gripper"].resized(4097)
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    caps = exact_capacities(host_encode(encoder, case))
    want = host_encode(encoder, case, capacities=caps)
    base = put(halves(case.rows), "cuda")
    heavy = put(np.ones((1536, 1536), np.float32), "cuda")

    @jax.jit
    def f(rows, a):
        for _ in range(12):  # a long kernel chain: the rows depend on its (zero) result
            a = jnp.tanh(a @ a * 1e-3)
        lag = (jnp.sum(a) * 0.0).astype(jnp.uint32)
        return ej.encode(handle, rows + lag, capacities=caps)

    for _ in range(5):
        same_batch(want, f(base, heavy))


def test_an_encoder_whose_device_cannot_synchronize_keeps_its_generation(cases):
    """Destroying the last reference of an encoder waits for the device. CUDA refuses that wait while another thread
    captures a CUDA graph; the generation is then kept (not freed under work that may still read it) and counted."""
    need_gpu()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    handle = ej.register(encoder)
    _encoding_jax.prepare(handle, 0)  # the device slot exists, no compiled function holds the encoder
    x = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()
    capturing, release = threading.Event(), threading.Event()

    def capture():
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode="global"):
                (x + 1).sum()
                capturing.set()
                release.wait(60)
        except Exception:  # the capture was invalidated by the refused synchronization
            pass
        finally:
            capturing.set()

    before = _encoding_jax.leaked_devices()
    thread = threading.Thread(target=capture)
    thread.start()
    try:
        assert capturing.wait(60)
        time.sleep(0.2)
        ej.unregister(handle)  # the registry held the last reference
    finally:
        release.set()
        thread.join()
    assert _encoding_jax.leaked_devices() == before + 1


def test_a_schema_whose_kernels_need_more_shared_memory_than_a_block_has_is_refused_when_traced():
    need_gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    names = [f"p{k}" for k in range(4000)]
    domain = mymyr.Domain.from_string(
        "(define (domain many) (:requirements :strips) (:predicates " + " ".join(f"({n})" for n in names) + ")"
        " (:action clear :parameters () :precondition (and) :effect (and " + " ".join(f"(not ({n}))" for n in names)
        + ")))"
    )
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain many) (:init " + " ".join(f"({n})" for n in names) + ") (:goal (and)))"))
    encoder = encoding.FlatRelationEncoder(task, predicate_virtual_nodes=True)
    rows = put(halves(np.asarray(task.encode([task.initial_state]))), "cuda")
    caps = dict(graphs=1, entities=4096, args=4096, targets=0)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the schema has too many predicate virtual nodes"):
        jax.jit(lambda r: ej.encode(encoder, r, capacities=caps))(rows)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the schema has too many predicate virtual nodes"):
        ej.encode(encoder, rows, capacities=caps)


GATED = (
    "(define (domain gated) (:requirements :strips :derived-predicates) (:predicates (p ?x) (q ?x))"
    " (:derived (q ?x) (p ?x)) (:action set :parameters (?x) :precondition (and) :effect (p ?x)))"
)


def gated_task(objects):
    names = " ".join(f"o{k}" for k in range(objects))
    return mymyr.Task(mymyr.Domain.from_string(GATED).instantiate_string(
        f"(define (problem g{objects}) (:domain gated) (:objects {names}) (:init (p o0)) (:goal (and (q o1))))"))


def test_derived_words_are_required_only_where_the_device_cannot_evaluate_the_axioms():
    need_gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    # the device evaluates the axioms of small instances: the rows come without their derived words
    small = [gated_task(3), gated_task(5)]
    case = Case(rl.TaskTable(small), small, derived=True)
    encoder = encoding.FlatRelationEncoder(case.source)
    plain = Case.__new__(Case)
    plain.__dict__.update(case.__dict__, derived=None)
    caps = exact_capacities(host_encode(encoder, plain), 2)
    want = host_encode(encoder, plain, capacities=caps)
    same_batch(want, jax_encode(encoder, plain, "cuda", capacities=caps))
    # 2100 objects need 33 object words, which it does not evaluate: those rows need their derived words
    large = [gated_task(3), gated_task(2100)]
    case = Case(rl.TaskTable(large), large, derived=True, steps=2)
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case), 2)
    want = host_encode(encoder, case, capacities=caps)
    same_batch(want, jax_encode(encoder, case, "cuda", capacities=caps))
    plain = Case.__new__(Case)
    plain.__dict__.update(case.__dict__, derived=None)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the rows' derived words are required"):
        jax_encode(encoder, plain, "cuda", capacities=caps)


def test_derived_subgoal_masks_need_derived_words_only_where_the_device_cannot_evaluate_the_axioms():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    options = {"max_goal_level": 2, "history": True}
    for objects, evaluable in ((5, True), (2100, False)):
        tasks = [gated_task(3), gated_task(objects)]
        case = Case(rl.TaskTable(tasks), tasks, derived=True, steps=2)
        pick = np.concatenate([np.flatnonzero(case.ids == k)[:6] for k in range(2)])
        case.rows, case.ids, case.derived = (np.ascontiguousarray(x[pick]) for x in (case.rows, case.ids, case.derived))
        encoder = encoding.FlatRelationEncoder(case.source, **options)
        kwargs = layered_kwargs(case, options, layers=2)
        assert len(kwargs["subgoals"]) == 4 and len(kwargs["history"]) == 6
        caps = exact_capacities(host_encode(encoder, case, **kwargs), 2)
        want = host_encode(encoder, case, capacities=caps, **kwargs)
        plain = Case.__new__(Case)
        plain.__dict__.update(case.__dict__, derived=None)
        # a function gives the same arrays wherever it runs: the rows of an instance whose axioms the device cannot
        # evaluate need their derived words on every platform of a CUDA build
        refused = not evaluable and _encoding_jax.has_cuda()
        for platform in PLATFORMS:
            same_batch(want, jax_encode(encoder, case, platform, capacities=caps, **kwargs))
            if refused:
                with pytest.raises(ValueError, match=r"^mymyr: encoding: the rows' derived words are required"):
                    jax_encode(encoder, plain, platform, capacities=caps, **kwargs)
            else:  # the target evaluates the derived atoms that the layers read
                same_batch(want, jax_encode(encoder, plain, platform, capacities=caps, **kwargs))


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("count", [0, 2, 4])
def test_live_rows_with_subgoals_and_history_equal_host_capacity(platform, layout, count):
    task = text_task("gripper__prob05")
    words = np.asarray(task.encode([task.initial_state] * 4))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, max_goal_level=1, history=True,
                                           target_sources=("goal", "subgoal", "history"))
    layers = (words[:, None, :].copy(), np.zeros((4, 1, words.shape[1]), np.uint64))
    history = (*layers, np.full((4, 1), -1, np.int32), np.ones(4, np.int32))
    full = encoder.encode(words, subgoals=layers, history=history)
    caps = exact_capacities(full, 2)
    inputs = layered_inputs(platform, layers, history)
    batch = jax.jit(lambda rows, steps, live: ej.encode(encoder, rows, capacities=caps, live=live, **steps))(
        put(halves(words), platform), inputs, put(np.asarray(count, np.int32), platform))
    expected = encoder.encode(words[:count], subgoals=tuple(x[:count] for x in layers),
                              history=tuple(x[:count] for x in history), capacities=caps)
    same_batch(expected, batch)
    for name in ("entity_offsets", "history_offsets", "target_offsets"):
        assert not np.asarray(batch._arrays[name])[count + 1:].any(), name
