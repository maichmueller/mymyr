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


def test_collation_requires_a_batch_to_establish_its_schema():
    with pytest.raises(ValueError, match="^mymyr: encoding: collate needs at least one batch"):
        encoding.collate([])
