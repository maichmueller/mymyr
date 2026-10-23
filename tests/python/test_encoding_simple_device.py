"""Device incidence graphs match the host arrays, including capacity tails and graph-local ordering."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task
from test_encoding_simple import BOTH, GOALS, TASK_NAMES
from test_encoding_cuda import (
    Case, call_op, device_encode, fluent_goals, gpu, host_encode, layered_rows,
    op_inputs, pddl_task, same_batch, to_device,
)


def make_case(name):
    if name == "mixed_blocks":
        tasks = [pddl_task("blocks", p) for p in ("smedium", "probBLOCKS-8-0")]
    elif name == "mixed_derived":
        tasks = [pddl_task("philosophers", p) for p in ("p03-phil4", "p03-phil4-one-blocked")]
    else:
        task = text_task(name)
        return Case(name, task, [task], derived="philosophers" in name, steps=4)
    return Case(name, rl.TaskTable(tasks), tasks, derived=name == "mixed_derived", steps=4)


def encoder_of(source, form, **options):
    if form == "simple":
        return encoding.SimpleEncoder(source, **options)
    return encoding.FlatRelationEncoder(source, outputs=BOTH, layout=form, **options)


def capacities(batch, factor=1):
    values = dict(graphs=batch.num_graphs, entities=int(batch.entity_offsets[-1]),
                  args=int(batch.edge_offsets[-1]) // 2, targets=0)
    if hasattr(batch, "target_offsets"):
        values["targets"] = int(batch.target_offsets[-1])
    if hasattr(batch, "history_offsets"):
        values["history"] = int(batch.history_offsets[-1])
    return {name: value * factor for name, value in values.items()}


OPTIONS = [
    dict(include_actions=False),
    dict(include_actions=False, goal_derivations=GOALS, target_sources=("goal",)),
    dict(nullary=False, equality=True, include_static=False, include_derived=False),
    dict(goal_derivations=GOALS, target_sources=("goal",)),
]


@pytest.mark.parametrize("name", [*TASK_NAMES, "mixed_blocks", "mixed_derived"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("options", OPTIONS)
def test_simple_device_arrays_equal_host(name, index, form, options):
    dev = gpu()
    case = make_case(name)
    encoder = encoder_of(case.source, form, index_dtype=index, **options)
    for n in (0, 1, case.n):
        host = encoder.encode(case.rows[:n], None if case.ids is None else case.ids[:n],
                              derived=None if case.derived is None else case.derived[:n])
        device = encoder.encode(to_device(case.rows[:n], dev),
                                None if case.ids is None else to_device(case.ids[:n], dev),
                                derived=None if case.derived is None else to_device(case.derived[:n], dev))
        same_batch(host, device)
        assert device.edge_index.dtype == (pytest.importorskip("torch").int32 if index == "int32" else
                                          pytest.importorskip("torch").int64)


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_simple_capacities_goals_layers_history_and_overflow(form, index):
    dev = gpu()
    case = make_case("mixed_derived")
    encoder = encoder_of(case.source, form, index_dtype=index, max_goal_level=3, history=True,
                         goal_derivations=GOALS, target_sources=("goal", "subgoal", "history"))
    subgoals, history = layered_rows(case)
    goals = (*fluent_goals(case), case.derived.copy(), np.zeros_like(case.derived))
    kwargs = dict(goals=goals, subgoals=subgoals, history=history)
    host = host_encode(encoder, case, **kwargs)
    same_batch(host, device_encode(encoder, case, dev, **kwargs))
    exact = capacities(host)
    for caps in (exact, capacities(host, 2), dict(exact, args=0), dict(exact, entities=0), dict(exact, history=0)):
        h = host_encode(encoder, case, capacities=caps, **kwargs)
        d = device_encode(encoder, case, dev, capacities=caps, **kwargs)
        same_batch(h, d)
        assert h.sizes == d.sizes
    # Evaluated derived words give the same graph as explicitly supplied words.
    h = encoder.encode(case.rows, case.ids, goals=goals, subgoals=subgoals, history=history)
    d = encoder.encode(to_device(case.rows, dev), to_device(case.ids, dev),
                       goals=tuple(to_device(x, dev) for x in goals),
                       subgoals=tuple(to_device(x, dev) for x in subgoals),
                       history=tuple(to_device(x, dev) for x in history))
    same_batch(h, d)


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_simple_op_compiles_and_captured_graph_replays(form, index):
    dev = gpu()
    torch = pytest.importorskip("torch")
    from mymyr.rl import torch as rt
    case = make_case("mixed_blocks")
    encoder = encoder_of(case.source, form, index_dtype=index, goal_derivations=GOALS, target_sources=("goal",))
    host = host_encode(encoder, case)
    caps = capacities(host, 2)
    args = op_inputs(case, dev)
    handle = rt.register(encoder)
    try:
        def run(rows):
            return call_op(handle, rows, *args[1:], caps)
        eager = run(args[0])
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        compiled_out = compiled(args[0])
        for a, b in zip(eager, compiled_out, strict=True):
            assert torch.equal(a, b)
        stream = torch.cuda.Stream(dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            run(args[0])
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                outputs = run(args[0])
            capture.replay()
        stream.synchronize()
        batch = encoding.DeviceFlatBatch.from_arrays(encoder, outputs)
        same_batch(host_encode(encoder, case, capacities=caps), batch)
        args[0].zero_()
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.default_stream(dev))
            capture.replay()
        stream.synchronize()
        changed = encoder.encode(np.zeros_like(case.rows), case.ids, capacities=caps)
        same_batch(changed, encoding.DeviceFlatBatch.from_arrays(encoder, outputs))
    finally:
        rt.unregister(handle)


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_simple_device_stream_and_collation(form, index):
    dev = gpu()
    case = make_case("mixed_blocks")
    encoder = encoder_of(case.source, form, index_dtype=index, goal_derivations=GOALS, target_sources=("goal",))
    host = host_encode(encoder, case)
    caps = capacities(host, 2)
    ring = encoder.device_stream(case.n + 3, device=dev)
    caps["graphs"] = case.n + 3
    split = case.n // 2
    ring.append(to_device(case.rows[:split], dev), to_device(case.ids[:split], dev))
    ring.append(to_device(case.rows[split:], dev), to_device(case.ids[split:], dev))
    batch = ring.flush(capacities=caps)
    # Compact exports trim capacity tails and preserve the two edge rows.
    same_batch(host, batch)
    assert len(ring) == 0
    ring.append(to_device(case.rows, dev), to_device(case.ids, dev))
    same_batch(host, ring.flush(capacities=caps))
    empty = encoder.encode(case.rows[:0], case.ids[:0])
    same_batch(empty, ring.flush(capacities=caps))
    parts = [encoder.encode(to_device(case.rows[a:b], dev), to_device(case.ids[a:b], dev))
             for a, b in ((0, 0), (0, split), (split, case.n))]
    same_batch(host, encoding.collate(parts))
    same_batch(empty, encoding.collate([parts[0], parts[0]]))
    same_batch(host_encode(encoder, case, capacities=caps), encoding.collate(parts, capacities=caps))


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_simple_constant_labels_are_stable_on_device(form, index):
    dev = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = mymyr.Domain.from_string(
        "(define (domain constants) (:requirements :strips) (:constants fixed)"
        " (:predicates (p ?x)) (:action set :parameters (?x) :precondition (and) :effect (p ?x)))")
    tasks = [mymyr.Task(domain.instantiate_string(
        f"(define (problem p{k}) (:domain constants) (:objects {' '.join('o'+str(i) for i in range(k))})"
        " (:init (p fixed)) (:goal (and)))")) for k in (1, 3)]
    case = Case("constants", rl.TaskTable(tasks), tasks)
    encoder = encoder_of(case.source, form, index_dtype=index, constant_labels=True)
    same_batch(host_encode(encoder, case), device_encode(encoder, case, dev))


@pytest.mark.parametrize("outputs", [("simple",), BOTH])
def test_simple_with_numeric_relations_is_refused(outputs):
    from mymyr.encoding._device import layout_of
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    encoder = encoding.FlatRelationEncoder(task, outputs=outputs)
    with pytest.raises(ValueError, match="^mymyr: encoding: the device does not encode Simple with numeric relations"):
        layout_of(encoder)


@pytest.mark.parametrize("name", ["cs-drone", "counters_f64"])
@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_simple_graphs_of_numeric_rows_without_numeric_relations(name, form, index):
    """The rows of a numeric table carry their numeric words after the atom words; without numeric relations
    (include_numeric=False) the Simple graph reads none of them and equals the host graph in every form."""
    dev = gpu()
    from test_encoding_numeric import build_cases
    from test_encoding_numeric_cuda import rows_of

    case = {c.name: c for c in build_cases()}.get(name)
    if case is None:
        pytest.skip("PDDL front end required")
    encoder = encoder_of(case.table, form, index_dtype=index, include_numeric=False)
    rows, ids = rows_of(case), np.asarray(case.ids, np.int32)
    assert rows.shape[1] > case.table.words
    host = encoder.encode(rows, ids)
    device_rows, device_ids = to_device(rows, dev), to_device(ids, dev)
    same_batch(host, encoder.encode(device_rows, device_ids))
    caps = capacities(host, 2)
    same_batch(encoder.encode(rows, ids, capacities=caps), encoder.encode(device_rows, device_ids, capacities=caps))
    n, split = len(rows), len(rows) // 2
    ring = encoder.device_stream(n + 3, device=dev)
    ring.append(device_rows[:split], device_ids[:split])
    ring.append(device_rows[split:], device_ids[split:])
    same_batch(host, ring.flush(capacities={**caps, "graphs": n + 3}))
    parts = [encoder.encode(device_rows[a:b], device_ids[a:b]) for a, b in ((0, split), (split, n))]
    same_batch(host, encoding.collate(parts))
    same_batch(encoder.encode(rows, ids, capacities=caps), encoding.collate(parts, capacities=caps))


@pytest.mark.parametrize("outputs", [("simple",), BOTH])
def test_simple_jax_refusal_explains_the_fixed_result_layout(outputs):
    pytest.importorskip("jax")
    from mymyr.encoding import jax as ej
    encoder = encoding.FlatRelationEncoder(text_task("gripper__prob05"), outputs=outputs)
    with pytest.raises(ValueError, match="JAX FFI does not encode Simple: its fixed result layout"):
        ej.register(encoder)
    from mymyr._core import _encoding_jax
    with pytest.raises(ValueError, match="JAX FFI does not encode Simple: its fixed result layout"):
        _encoding_jax.register(encoder)


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
def test_simple_refuses_device_action_rows(form):
    dev = gpu()
    task = text_task("gripper__prob05")
    rows = to_device(np.asarray(task.encode([task.initial_state])), dev)
    encoder = encoder_of(task, form)
    actions = rl.expand(rl.as_table(task), rows, framework="torch")
    with pytest.raises(ValueError, match="device action rows and horizons require relational output"):
        encoder.encode(rows, actions=actions)


@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
def test_simple_invalid_rows_zero_every_public_array(form):
    dev = gpu()
    torch = pytest.importorskip("torch")
    case = make_case("mixed_blocks")
    encoder = encoder_of(case.source, form)
    caps = capacities(host_encode(encoder, case))
    ids = to_device(case.ids, dev)
    ids[0] = 1000
    batch = encoder.encode(to_device(case.rows, dev), ids, capacities=caps)
    assert batch.sizes["invalid_input"]
    assert all(torch.count_nonzero(array).item() == 0 for array in batch._arrays.values())


@pytest.mark.parametrize("outputs", [("simple",), BOTH])
def test_simple_horizons_are_refused_before_any_launch(outputs):
    """The host encodes the Simple graph of a horizon; the device refuses it up front (layout, encode, transitions),
    not after allocating for it."""
    dev = gpu()
    torch = pytest.importorskip("torch")
    from mymyr.encoding import _device
    from test_encoding_device_actions import upload

    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode([task.initial_state]))
    message = "^mymyr: encoding: the device does not encode Simple horizons"
    horizon = encoding.FlatHorizonEncoder(task, horizon_mode="full", outputs=outputs)
    view = encoding.horizon(task, rows, depth=1)
    assert horizon.encode(view).num_graphs == 1  # the host encodes it
    with pytest.raises(ValueError, match=message):
        _device.layout_of(horizon)
    uploaded, words = upload(view, dev), to_device(rows, dev)
    transitions = encoding.FlatTransitionEncoder(task, outputs=outputs)
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_allocated(dev)
    with pytest.raises(ValueError, match=message):
        horizon.encode(uploaded)
    with pytest.raises(ValueError, match=message):
        transitions.encode(words)
    with pytest.raises(ValueError, match=message):
        transitions.encode(words, capacities=dict(graphs=1, entities=64, args=256, targets=64))
    assert torch.cuda.memory_allocated(dev) <= before


@pytest.mark.parametrize("name", ["philosophers__p03-phil4", "mixed_blocks"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("form", ["graph", "relation", "simple"])
@pytest.mark.parametrize("targets", [(), ("goal", "action")], ids=["actions", "action_targets"])
def test_simple_with_action_relations_and_targets_without_expansion_rows(name, index, form, targets):
    """Action relations and action targets without supplied expansions have no tuples and no targets; the Simple fill
    then runs over the wide scans and the relation-major cell entries that action relations take."""
    dev = gpu()
    case = make_case(name)
    encoder = encoder_of(case.source, form, index_dtype=index, include_actions=True, target_sources=targets,
                         goal_derivations=GOALS)
    host = host_encode(encoder, case)
    same_batch(host, device_encode(encoder, case, dev))
    caps = capacities(host, 2)
    same_batch(host_encode(encoder, case, capacities=caps), device_encode(encoder, case, dev, capacities=caps))
