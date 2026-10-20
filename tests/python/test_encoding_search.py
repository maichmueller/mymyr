"""State spaces feed the encoders: the rows and edges of `StateSpace.arrays()` are the nodes and edges of a horizon, the
batch of horizons equals the horizon builder's for the same states, and the rows of a frontier encode directly."""

import numpy as np
import pytest

from mymyr import datasets, encoding, rl
from conftest import text_task
from test_encoding_pool import arrays_equal


@pytest.fixture(scope="module")
def task():
    return text_task("philosophers__p03-phil4")


@pytest.fixture(scope="module")
def space(task):
    result = datasets.generate(task, max_states=100_000, remove_if_unsolvable=False)
    assert result.space is not None
    return result.space


def neighbourhood_view(arrays, ids):
    """The depth-one horizons of the states `ids` from the arrays of a state space: the state's row and then the rows of
    its successors as nodes (one per transition), and one edge per transition with the transition's action label."""
    rows = arrays["state_words"]
    first, targets = arrays["forward_offsets"].astype(np.int64), arrays["forward_targets"].astype(np.int64)
    nodes, node_offsets, edge_offsets = [], [0], [0]
    src, dst, schema, binding = [], [], [], []
    for state in ids:
        edges = np.arange(first[state], first[state + 1])
        nodes += [rows[state], *rows[targets[edges]]]
        node_offsets.append(node_offsets[-1] + 1 + len(edges))
        edge_offsets.append(edge_offsets[-1] + len(edges))
        src += [0] * len(edges)
        dst += list(range(1, len(edges) + 1))
        schema.append(arrays["label_schemas"][edges])
        binding.append(arrays["label_bindings"][edges])
    return encoding.HorizonView(
        nodes=np.stack(nodes),
        node_offsets=np.array(node_offsets, np.int32),
        edge_offsets=np.array(edge_offsets, np.int32),
        edge_src=np.array(src, np.int32),
        edge_dst=np.array(dst, np.int32),
        edge_schema=np.concatenate(schema).astype(np.int32),
        edge_binding=np.concatenate(binding).astype(np.int32),
    )


@pytest.mark.parametrize("mode", ["full", "delta", "action"])
def test_the_arrays_of_a_state_space_are_a_horizon_and_encode_like_the_builders(task, space, mode):
    ids = list(range(0, space.num_states, max(1, space.num_states // 12)))
    arrays = space.arrays()
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode=mode, parent_relation=True)
    from_arrays = encoder.encode(neighbourhood_view(arrays, ids))

    rows = np.ascontiguousarray(arrays["state_words"][ids])
    built = encoder.encode(encoding.horizon(task, rows, depth=1, merge=False))
    arrays_equal(from_arrays, built)
    arrays_equal(from_arrays, encoder.encode(encoding.horizon_of_space(space, ids, 1, merge=False)))
    assert from_arrays.num_graphs == len(ids)


def test_merged_horizons_of_a_space_equal_the_builders(task, space):
    ids = [0, 5, 17]
    encoder = encoding.FlatRootedHorizonEncoder(task, sibling_relation=True)
    rows = np.ascontiguousarray(space.arrays()["state_words"][ids])
    arrays_equal(
        encoder.encode(encoding.horizon_of_space(space, ids, 2)), encoder.encode(encoding.horizon(task, rows, depth=2))
    )


def test_frontier_rows_encode_directly(task, space):
    arrays = space.arrays()
    first, targets = arrays["forward_offsets"].astype(np.int64), arrays["forward_targets"].astype(np.int64)
    frontier = np.unique(targets[first[0] : first[1]])  # the states one transition from the initial state
    rows = arrays["state_words"][frontier]
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    table = rl.as_table(task)
    expansion = rl.expand(table, rows)
    batch = encoder.encode(rows, actions=expansion)
    assert batch.num_graphs == len(frontier)
    arrays_equal(batch, encoder.encode([space.state(int(i)) for i in frontier], actions=expansion))
    # each frontier state's actions are the space's transitions out of it
    action_rows = encoding.action_rows(batch, expansion)
    graphs = np.searchsorted(np.asarray(batch.entity_offsets), encoding.action_entities(batch), side="right") - 1
    for g, row in zip(graphs.tolist(), action_rows.tolist()):
        state = int(frontier[g])
        successor = np.asarray(expansion.succ)[row]
        edges = np.arange(first[state], first[state + 1])
        assert any(np.array_equal(arrays["state_words"][targets[e]], successor) for e in edges)
