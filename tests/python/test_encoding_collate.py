"""Collated rollout slices equal one encoding, including empty slices and capacity tails."""
import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import BLOCKS, ROOT, text_task, walk


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("targets", [False, True])
@pytest.mark.parametrize("counts", [False, True])
def test_host_collation_equals_one_encoding(layout, index, targets, counts):
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    rows = np.asarray(task.encode(walk(task, steps=11, walks=2, seed=19)))
    encoder = encoding.FlatRelationEncoder(table, layout=layout, index_dtype=index, emit_counts=counts,
                                           target_sources=("goal",) if targets else (), predicate_virtual_nodes=targets)
    expected = encoder.encode(rows)
    slices = [encoder.encode(rows[:3]), encoder.encode(rows[:0]), encoder.encode(rows[3:11]), encoder.encode(rows[11:])]
    actual = encoding.collate(slices)
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, actual.as_dict()[key]), key
    assert actual.num_graphs == len(rows)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("virtual", [False, True])
def test_host_collation_of_shared_statics_equals_one_encoding(layout, index, virtual):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    tasks = [mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / f"{problem}.pddl")
             for problem in ("probBLOCKS-8-0", "smedium")]
    table = rl.TaskTable(tasks)
    encoded = [np.asarray(t.encode(walk(t, steps=7, walks=2, seed=23 + i))) for i, t in enumerate(tasks)]
    width = max(e.shape[1] for e in encoded)
    parts = [np.pad(e, ((0, 0), (0, width - e.shape[1]))) for e in encoded]
    # the instances interleaved, so that the slices hold one, the other or both of them
    rows = np.concatenate([parts[0][:4], parts[1][:6], parts[0][4:], parts[1][6:]])
    ids = np.repeat(np.array([0, 1, 0, 1], dtype=np.int32), [4, 6, len(parts[0]) - 4, len(parts[1]) - 6])
    common = dict(layout=layout, index_dtype=index, predicate_virtual_nodes=virtual, target_sources=("goal",))
    encoder = encoding.FlatRelationEncoder(table, statics="shared", **common)
    expected = encoder.encode(rows, ids)
    cuts = [0, 3, 3, 5, 9, len(rows)]
    slices = [encoder.encode(rows[a:b], ids[a:b]) for a, b in zip(cuts, cuts[1:])]
    assert [s.static_task_ids.tolist() for s in slices] == [[0], [], [0, 1], [1], [0, 1]]
    actual = encoding.collate(slices)
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, actual.as_dict()[key]), key
    repeated = encoding.FlatRelationEncoder(table, **common).encode(rows[:3], ids[:3])
    with pytest.raises(ValueError, match="^mymyr: encoding: collate received"):
        encoding.collate([slices[0], repeated])


@pytest.mark.parametrize("task_name", ["cs-drone", "gripper__prob05"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("layout", "outputs"), [("graph", ("relations",)), ("relation", ("relations",)),
                                                 ("graph", ("relations", "simple")), ("graph", ("simple",))])
def test_host_collation_of_numeric_values_and_simple_graphs_equals_one_encoding(task_name, dtype, layout, outputs):
    if task_name == "cs-drone":
        task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-drone.txt"))
    else:
        task = text_task(task_name)
    rows = np.asarray(task.encode(walk(task, steps=11, walks=2, seed=29)))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, numeric_dtype=dtype, outputs=outputs)
    expected = encoder.encode(rows)
    numeric = sum(lane == "numeric" for lane in encoder.relation_lanes)
    assert expected.as_dict().get("numeric_relations", 0) == (numeric if "relations" in outputs else 0)
    assert hasattr(expected, "numeric_values") == (task_name == "cs-drone" and "relations" in outputs)
    slices = [encoder.encode(rows[:3]), encoder.encode(rows[:0]), encoder.encode(rows[3:11]), encoder.encode(rows[11:])]
    actual = encoding.collate(slices)
    assert expected.as_dict().keys() == actual.as_dict().keys()
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, actual.as_dict()[key], equal_nan=np.asarray(value).dtype.kind == "f"), key


@pytest.mark.parametrize("option", [{"layout": "relation"}, {"index_dtype": "int64"}, {"include_static": False}])
def test_collation_refuses_incompatible_batches(option):
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode([task.initial_state]))
    first = encoding.FlatRelationEncoder(task).encode(rows)
    second = encoding.FlatRelationEncoder(task, **option).encode(rows)
    with pytest.raises(ValueError, match="^mymyr: encoding: collate received mixed"):
        encoding.collate([first, second])


def test_collation_refuses_batches_of_different_tables():
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode([task.initial_state]))
    one, two = rl.TaskTable([task]), rl.TaskTable([task, text_task("gripper__prob05")])
    first, second = encoding.FlatRelationEncoder(one), encoding.FlatRelationEncoder(two)
    assert first.schema_fingerprint == second.schema_fingerprint
    assert first.table_fingerprint == one.fingerprint != second.table_fingerprint == two.fingerprint
    a, b = first.encode(rows), second.encode(rows, np.array([1], dtype=np.int32))
    assert a.as_dict()["table_fingerprint"] % 2**64 == one.fingerprint
    with pytest.raises(ValueError, match="^mymyr: encoding: collate received batches of different tables"):
        encoding.collate([a, b])
    again = encoding.FlatRelationEncoder(two).encode(rows, np.array([0], dtype=np.int32))
    assert encoding.collate([b, again]).num_graphs == 2


def test_collation_requires_a_batch_to_establish_its_schema():
    with pytest.raises(ValueError, match="^mymyr: encoding: collate needs at least one batch"):
        encoding.collate([])


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_device_stream_capacity_validation_precedes_device_construction(capacity):
    pytest.importorskip("torch")
    encoder = encoding.FlatRelationEncoder(text_task("gripper__prob05"))
    with pytest.raises(ValueError, match="^mymyr: encoding: a device stream needs a positive integer capacity"):
        encoder.device_stream(capacity)
