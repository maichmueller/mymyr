"""The layouts, threads, capacities and omitted arrays of the relational encoder."""

import numpy as np
import pytest

from mymyr import encoding
from conftest import text_task, walk

TASKS = ["blocks__probBLOCKS-8-0", "gripper__prob05", "philosophers__p03-phil4"]
THREADS = [1, 2, 3, 8, 0]


def encoder_of(task, **options):
    return encoding.FlatRelationEncoder(task, **{"goal_derivations": (), **options})


def cells(batch):
    """{(graph, relation): tuples of instance-local entity ids}, read through the batch's layout (any option set)."""
    arrays = batch.as_dict()
    graphs, relations = batch.num_graphs, batch.num_relations
    relation_major = arrays["layout"] == 1
    offsets = arrays["relation_offsets"]
    arg_offsets = arrays["relation_arg_offsets"]
    arities = batch.schema.relation_arities
    base = arrays["entity_offsets"]
    out = {}
    for g in range(graphs):
        for r in range(relations):
            s = r * graphs + g if relation_major else g * relations + r
            count = int(offsets[s + 1] - offsets[s])
            if "relation_counts" in arrays:
                assert arrays["relation_counts"][g, r] == count
            if not count:
                continue
            args = arrays["relation_args"][arg_offsets[s] : arg_offsets[s + 1]] - base[g]
            out[g, r] = args.reshape(count, arities[r]).tolist()
    return out


def entities(batch):
    """(role, ref) lists per graph, explicit or implicit."""
    arrays = batch.as_dict()
    out = []
    for g in range(batch.num_graphs):
        lo, hi = int(arrays["entity_offsets"][g]), int(arrays["entity_offsets"][g + 1])
        if batch.implicit_entities:
            objects = int(arrays["num_objects"][g])
            out.append([(0, j) if j < objects else (8, -1) for j in range(hi - lo)])
        else:
            out.append(list(zip(arrays["entity_role"][lo:hi].tolist(), arrays["entity_ref"][lo:hi].tolist())))
    return out


def same_arrays(a, b):
    x, y = a.as_dict(), b.as_dict()
    assert x.keys() == y.keys()
    for key in x:
        assert np.array_equal(x[key], y[key]), key


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("options", [{}, {"index_dtype": "int64"}, {"emit_counts": False}])
def test_layouts_and_threads_give_the_same_graphs(name, options):
    task = text_task(name)
    states = walk(task, steps=40, walks=3)
    reference = encoder_of(task, **options).encode(states)
    expected_cells = cells(reference)
    relation = encoder_of(task, layout="relation", **options)
    assert relation.schema_fingerprint != encoder_of(task, **options).schema_fingerprint
    first = relation.encode(states)
    assert first.as_dict()["layout"] == 1
    assert cells(first) == expected_cells
    assert entities(first) == entities(reference)
    for encoder in (encoder_of(task, **options), relation):
        first = encoder.encode(states)
        for threads in THREADS:
            same_arrays(first, encoder.encode(states, threads=threads))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", TASKS)
def test_single_equals_batch_equals_slices(name, layout):
    task = text_task(name)
    states = walk(task, steps=25, walks=2)
    encoder = encoder_of(task, layout=layout)
    whole = encoder.encode(states, threads=2)
    whole_cells = cells(whole)
    relations = whole.num_relations
    for g, state in enumerate(states):
        single = cells(encoder.encode(state))
        assert single == {(0, r): t for (gg, r), t in whole_cells.items() if gg == g}
    for lo, hi in [(0, 1), (3, 9), (5, len(states)), (0, len(states))]:
        part = encoder.encode(states[lo:hi], threads=3)
        expected = {(g - lo, r): t for (g, r), t in whole_cells.items() if lo <= g < hi}
        assert cells(part) == expected
        assert entities(part) == entities(whole)[lo:hi]
    assert relations == len(encoder.relation_names)


def test_a_table_of_several_instances_under_threads():
    from mymyr import rl

    tasks = [text_task("blocks__probBLOCKS-8-0"), text_task("blocks__probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    states = [walk(t, steps=10, walks=1) for t in tasks]
    rows = np.concatenate([tasks[i].encode(s) for i, s in enumerate(states)])
    ids = np.array([0] * len(states[0]) + [1] * len(states[1]), dtype=np.int32)
    for layout in ("graph", "relation"):
        encoder = encoding.FlatRelationEncoder(table, goal_derivations=(), layout=layout)
        base = encoder.encode(rows, ids)
        for threads in THREADS:
            same_arrays(base, encoder.encode(rows, ids, threads=threads))


def test_omitted_counts_are_the_differences_of_the_offsets():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=2)
    for layout in ("graph", "relation"):
        full = encoder_of(task, layout=layout).encode(states)
        bare = encoder_of(task, layout=layout, emit_counts=False).encode(states)
        assert not hasattr(bare, "relation_counts") or "relation_counts" not in bare.as_dict()
        with pytest.raises(AttributeError):
            bare.relation_counts
        assert cells(bare) == cells(full)
        a, b = full.as_dict(), bare.as_dict()
        assert set(a) - set(b) == {"relation_counts"}
        assert not np.array_equal(a["schema_fingerprint"], b["schema_fingerprint"])


def test_implicit_entity_arrays_are_the_objects_then_the_auxiliary_entity():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=15, walks=2)
    options = {"include_actions": False}
    always = encoder_of(task, **options).encode(states)
    auto_encoder = encoder_of(task, entity_arrays="auto", **options)
    auto = auto_encoder.encode(states, threads=2)
    assert not always.implicit_entities and auto.implicit_entities
    assert "entity_role" not in auto.as_dict() and "entity_ref" not in auto.as_dict()
    with pytest.raises(AttributeError):
        auto.entity_role
    assert entities(auto) == entities(always)
    assert any(role == 8 for graph in entities(auto) for role, _ in graph)
    assert cells(auto) == cells(always)
    # actions are entities of their own once the schema has action relations: auto then stores the arrays
    with_actions = encoder_of(task, entity_arrays="auto").encode(states)
    assert not with_actions.implicit_entities
    assert "entity_role" in with_actions.as_dict()
    assert auto_encoder.options["entity_arrays"] == "auto"
    assert auto_encoder.options["emit_counts"] is True


def test_the_options_of_the_stored_arrays_are_validated_and_in_the_fingerprint():
    task = text_task("gripper__prob05")
    base = encoding.EncodingSchema(task, goal_derivations=(), include_actions=False)
    for change in ({"emit_counts": False}, {"entity_arrays": "auto"}):
        other = encoding.EncodingSchema(task, goal_derivations=(), include_actions=False, **change)
        assert other.schema_fingerprint != base.schema_fingerprint
    # an `auto` that cannot omit anything is the same encoding as `always`
    assert (
        encoding.EncodingSchema(task, goal_derivations=(), entity_arrays="auto").schema_fingerprint
        == encoding.EncodingSchema(task, goal_derivations=()).schema_fingerprint
    )
    with pytest.raises(ValueError, match="^mymyr: encoding: entity_arrays must be one of"):
        encoding.EncodingSchema(task, goal_derivations=(), entity_arrays="never")
    with pytest.raises(ValueError, match="^mymyr: encoding: layout must be one of"):
        encoder_of(task, layout="column")


def test_threads_are_validated():
    task = text_task("gripper__prob05")
    encoder = encoder_of(task)
    for bad in (-1, 10**6):
        with pytest.raises((ValueError, OverflowError)):
            encoder.encode(task.initial_state, threads=bad)
    with pytest.raises((TypeError, ValueError)):
        encoder.encode(task.initial_state, threads="2")


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_capacities_exact_roomy_short_and_zero(layout, index):
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=20, walks=2)
    encoder = encoder_of(task, layout=layout, index_dtype=index)
    expected = encoder.encode(states)
    assert expected.sizes is None
    exact = {
        "entities": int(expected.entity_offsets[-1]),
        "tuples": int(expected.relation_offsets[-1]),
        "args": int(expected.relation_arg_offsets[-1]),
    }
    for threads in (1, 4):
        batch = encoder.encode(states, capacities=exact, threads=threads)
        assert batch.sizes == {**exact, "overflow": False, "index_overflow": False}
        same_arrays(batch, expected)
        roomy = {"entities": exact["entities"] + 9, "tuples": exact["tuples"], "args": exact["args"] + 13}
        batch = encoder.encode(states, capacities=roomy, threads=threads)
        assert batch.sizes["overflow"] is False
        assert batch.relation_args.shape == (roomy["args"],)
        assert np.array_equal(batch.relation_args[: exact["args"]], expected.relation_args)
        assert not batch.relation_args[exact["args"] :].any()
        assert np.array_equal(batch.entity_role[: exact["entities"]], expected.entity_role)
        assert not batch.entity_role[exact["entities"] :].any() and not batch.entity_ref[exact["entities"] :].any()
        assert np.array_equal(batch.relation_offsets, expected.relation_offsets)
        for short in ({**exact, "entities": exact["entities"] - 1}, {**exact, "tuples": exact["tuples"] - 1},
                      {**exact, "args": exact["args"] - 1}, {"entities": 0, "tuples": 0, "args": 0}):
            batch = encoder.encode(states, capacities=short, threads=threads)
            assert batch.sizes["overflow"] is True and batch.sizes["index_overflow"] is False
            assert {k: batch.sizes[k] for k in exact} == exact  # the true sizes
            assert batch.relation_args.shape == (short["args"],)
            for key, value in batch.as_dict().items():
                if isinstance(value, np.ndarray):
                    assert not value.any(), key


def test_capacities_of_an_empty_batch_and_bad_capacities():
    task = text_task("gripper__prob05")
    encoder = encoder_of(task)
    empty = encoder.encode(np.zeros((0, task.words), np.uint64), capacities={"entities": 0, "tuples": 0, "args": 0})
    assert empty.num_graphs == 0 and empty.sizes["overflow"] is False
    state = task.initial_state
    for bad, error in (([1, 2, 3], TypeError), ({"entities": 1}, ValueError),
                       ({"entities": 1, "tuples": 1, "args": 1, "extra": 1}, ValueError),
                       ({"entities": -1, "tuples": 1, "args": 1}, (ValueError, OverflowError))):
        with pytest.raises(error):
            encoder.encode(state, capacities=bad)
