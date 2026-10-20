"""Collated rollout slices equal one encoding, including empty slices and capacity tails."""
import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task, walk


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
