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

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
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


def platforms():
    out = ["cpu"]
    try:
        if jax.devices("gpu") and _encoding_jax.has_cuda():
            out.append("cuda")
    except RuntimeError:
        pass
    return out


PLATFORMS = platforms()


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
                padded = np.ascontiguousarray(np.asarray(x.succ))
                dwords.append(np.asarray(x.derived))
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
        out.rows = np.ascontiguousarray(self.rows[pick].reshape(n, self.rows.shape[1]))
        out.ids = None if self.ids is None else np.ascontiguousarray(self.ids[pick])
        out.derived = None if self.derived is None else np.ascontiguousarray(self.derived[pick].reshape(n, -1))
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


def jax_encode(encoder, case, platform, goals=None, jit=True, **kwargs):
    rows, ids, derived, g = jax_inputs(case, platform, goals)

    def f(rows, ids, derived, g):
        return ej.encode(encoder, rows, ids, derived=derived, goals=g, **kwargs)

    return (jax.jit(f) if jit else f)(rows, ids, derived, g)


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
    case = cases["gripper"]
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
            assert ej._host_sizes(np.asarray(stacked.device_sizes[k]))["args"] == host.sizes["args"]


def test_vmap_encodes_each_batch_on_its_own(cases):
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source)
    caps = exact_capacities(host_encode(encoder, case))
    caps["graphs"] = 10
    rows = halves(case.rows[:30]).reshape(3, 10, -1)
    for platform in PLATFORMS:
        batched = jax.jit(jax.vmap(lambda r: ej.encode(encoder, r, capacities=caps)))(put(rows, platform))
        assert batched.relation_args.shape == (3, caps["args"])
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
    phil = cases["philosophers"]
    enc = encoding.FlatRelationEncoder(phil.source)
    if _encoding_jax.has_cuda():
        with pytest.raises(ValueError, match=r"^mymyr: encoding: the rows' derived words are required"):
            jax.jit(lambda r: ej.encode(enc, r, capacities=caps))(halves(phil.rows))
        with pytest.raises(ValueError, match=r"^mymyr: encoding: the device does not encode action targets"):
            ej.encode(encoding.FlatRelationEncoder(case.source, target_sources=("action",)), rows, capacities=caps)


# ------------------------------------------------------------------------------------------------ batches and handles


def test_a_jax_batch_has_the_accessors_of_a_host_batch(cases):
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, emit_counts=False, entity_arrays="auto", include_actions=False,
                                           goal_derivations=())
    host = host_encode(encoder, case)
    caps = exact_capacities(host)
    batch = jax_encode(encoder, case, "cpu", capacities=caps)
    assert batch.num_graphs == host.num_graphs and batch.num_relations == host.num_relations
    assert batch.schema.fingerprint == encoder.schema.fingerprint
    assert batch.implicit_entities and host.implicit_entities
    for name in ("relation_counts", "entity_role", "entity_ref", "goal_complete", "target_offsets"):
        with pytest.raises(AttributeError, match=f"this batch has no {name}"):
            getattr(batch, name)
    assert batch.device_sizes.dtype == jnp.uint32 and batch.device_sizes.shape == (5, 2)
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
        ej.unregister(handle)
        same_batch(want, f(rows))  # and so does the compiled function
        with pytest.raises(ValueError, match="no encoder is registered under handle"):
            f(rows[:5])  # a new trace looks the handle up
        calls = _encoding_jax.calls(handle)
        assert calls["cuda" if platform == "cuda" else "cpu"] >= 1


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
    need = _encoding_jax.scratch_bytes(handle, case.n, 0)
    assert f"tensor<{need + 255}xui8>" in text


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
