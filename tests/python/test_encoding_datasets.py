"""Datasets store rows and task ids and encode when a batch is loaded, through a collate function that takes an encoder:
encode-on-load equals encoding the stored rows directly, and nothing in a dataset file is an encoding."""

import zipfile

import numpy as np
import pytest

import mymyr
from mymyr import datasets, encoding, rl
from conftest import ROOT
from test_encoding_pool import arrays_equal
from test_rl_task_suite import suite_batch, suite_of
from test_rl_table import sets  # noqa: F401  (a fixture: the table instances of several domains)

GRIPPER = ROOT / "tests/data/pddl/gripper"
OPTIONS = dict(target_sources=("action", "goal"))
ENCODING_ARRAYS = {"entity_offsets", "entity_role", "entity_ref", "relation_args", "relation_offsets",
                   "relation_arg_offsets", "relation_counts", "target_entity", "goal_complete"}


@pytest.fixture(scope="module")
def knowledge_base():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    tasks = rl.TaskTable.from_pddl(GRIPPER / "domain.pddl", GRIPPER / "p*.pddl", atoms="dense")
    return datasets.KnowledgeBase(tasks, generalized=True, width=1)


def items_of(dataset, indices):
    return [dataset[int(i)] for i in indices]


def test_a_dataset_of_a_state_space_holds_the_rows_and_labels(knowledge_base):
    space = knowledge_base.state_spaces[1]
    dataset = datasets.StateRows.from_state_space(space, task_id=1)
    arrays = space.arrays()
    assert len(dataset) == space.num_states
    assert np.array_equal(dataset.rows, arrays["state_words"]) and (dataset.task_ids == 1).all()
    assert np.array_equal(dataset.labels["unit_goal_distances"], arrays["unit_goal_distances"])
    item = dataset[3]
    assert np.array_equal(item.row, arrays["state_words"][3]) and item.task_id == 1
    assert item.labels["unit_goal_distances"] == arrays["unit_goal_distances"][3]
    assert np.array_equal(dataset[-1].row, arrays["state_words"][-1])
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    subset = datasets.StateRows.from_state_space(space, [5, 2, 5], task_id=1, labels=("goal", "unsolvable"))
    assert np.array_equal(subset.rows, arrays["state_words"][[5, 2, 5]]) and set(subset.labels) == {"goal", "unsolvable"}
    assert np.array_equal(dataset.take([5, 2, 5]).rows, subset.rows)


def test_encoding_on_load_equals_encoding_the_stored_rows(knowledge_base):
    space = knowledge_base.state_spaces[1]
    task = knowledge_base.tasks[knowledge_base.task_indices[1]]
    dataset = datasets.StateRows.from_state_space(space, task_id=1)
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks, **OPTIONS)
    collate = datasets.EncodingCollate(encoder)
    chosen = [4, 0, 11, 11, 7]

    loaded = collate(items_of(dataset, chosen))
    direct = encoder.encode(dataset.rows[chosen], dataset.task_ids[chosen])
    arrays_equal(loaded.graphs, direct)
    assert np.array_equal(loaded.task_ids, dataset.task_ids[chosen]) and loaded.actions is None
    assert np.array_equal(loaded.labels["unit_goal_distances"], dataset.labels["unit_goal_distances"][chosen])
    # the same rows as States of the task
    arrays_equal(loaded.graphs, encoder.encode([space.state(i) for i in chosen], dataset.task_ids[chosen]))
    # a whole dataset taken at once is the same call
    arrays_equal(collate(dataset.take(chosen)).graphs, direct)


def test_actions_and_goals_are_encoded_when_asked(knowledge_base):
    space = knowledge_base.state_spaces[0]
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks, **OPTIONS)
    table = knowledge_base.tasks
    dataset = datasets.StateRows.from_state_space(space, task_id=0)
    chosen = [1, 3, 2]
    loaded = datasets.EncodingCollate(encoder, actions=True)(items_of(dataset, chosen))
    expansion = rl.expand(table, dataset.rows[chosen], dataset.task_ids[chosen])
    arrays_equal(loaded.graphs, encoder.encode(dataset.rows[chosen], dataset.task_ids[chosen], actions=expansion))
    assert loaded.actions.num_states == len(chosen)
    assert len(encoding.action_entities(loaded.graphs)) > 0
    assert np.array_equal(
        encoding.action_rows(loaded.graphs, loaded.actions),
        np.asarray(encoding.action_rows(loaded.graphs, expansion)),
    )

    goals = rl.goal_masks(table, dataset.task_ids)
    with_goals = datasets.StateRows(dataset.rows, dataset.task_ids, goals=tuple(np.asarray(g) for g in goals))
    assert with_goals[2].goal is not None
    arrays_equal(
        datasets.EncodingCollate(encoder)(items_of(with_goals, chosen)).graphs,
        encoder.encode(dataset.rows[chosen], dataset.task_ids[chosen], goals=(goals[0][chosen], goals[1][chosen])),
    )


def test_a_knowledge_base_gives_rows_of_all_its_spaces(knowledge_base):
    dataset = datasets.StateRows.from_knowledge_base(knowledge_base)
    spaces = knowledge_base.state_spaces
    assert len(dataset) == sum(s.num_states for s in spaces)
    expected_ids = np.concatenate(
        [np.full(s.num_states, t, np.int32) for s, t in zip(spaces, knowledge_base.task_indices)]
    )
    assert np.array_equal(dataset.task_ids, expected_ids)
    first = spaces[0].arrays()["state_words"]
    assert np.array_equal(dataset.rows[: len(first), : first.shape[1]], first)
    assert not dataset.rows[: len(first), first.shape[1] :].any()
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks)
    everything = datasets.EncodingCollate(encoder)(dataset).graphs
    arrays_equal(everything, encoder.encode(dataset.rows, dataset.task_ids))
    # each row as a State of its task encodes alike
    states = [s.state(i) for s in spaces for i in range(s.num_states)]
    arrays_equal(everything, encoder.encode(states, dataset.task_ids))


def test_dataset_files_hold_rows_and_task_ids_and_no_encodings(knowledge_base, tmp_path):
    path = tmp_path / "gripper.mymyr"
    datasets.save(knowledge_base, path)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    names = {name.rsplit("/", 1)[-1].removesuffix(".npy") for name in members}
    assert not (names & ENCODING_ARRAYS), names & ENCODING_ARRAYS
    assert any(name.endswith("state_words.npy") for name in members)
    assert "header.json" in members and not any("encod" in name.lower() for name in members)
    loaded = datasets.load(path)
    stored = datasets.StateRows.from_knowledge_base(knowledge_base)
    reloaded = datasets.StateRows.from_knowledge_base(loaded)
    assert np.array_equal(stored.rows, reloaded.rows) and np.array_equal(stored.task_ids, reloaded.task_ids)
    # encoding what was loaded gives the encoding of what was saved
    for kb in (knowledge_base, loaded):
        encoder = encoding.FlatRelationEncoder(kb.tasks)
        arrays_equal(datasets.EncodingCollate(encoder)(reloaded).graphs, encoder.encode(stored.rows, stored.task_ids))


def test_a_collate_function_pickles_with_its_encoder(knowledge_base):
    import pickle

    dataset = datasets.StateRows.from_knowledge_base(knowledge_base)
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks, **OPTIONS)
    collate = datasets.EncodingCollate(encoder, actions=True)
    loaded = pickle.loads(pickle.dumps(collate))
    items = items_of(dataset, [0, 5, len(dataset) - 1])
    arrays_equal(loaded(items).graphs, collate(items).graphs)


def test_suite_rows_are_encoded_into_a_batch_per_domain(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    dataset = datasets.StateRows(states, ids)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    loaded = datasets.EncodingCollate(encoder, actions=True)(items_of(dataset, [3, 0, 7, 1]))
    direct = encoder.encode(states[[3, 0, 7, 1]], ids[[3, 0, 7, 1]], actions=True)
    assert list(loaded.graphs) == list(direct) and loaded.actions is None
    for d in direct:
        arrays_equal(loaded.graphs[d], direct[d])
    assert np.array_equal(loaded.graphs.index, direct.index)


def test_dataset_and_collate_inputs_are_checked(knowledge_base):
    space = knowledge_base.state_spaces[0]
    with pytest.raises(ValueError, match="64-bit word array"):
        datasets.StateRows(np.zeros((3, 1), np.int32))
    with pytest.raises(ValueError, match="3 rows"):
        datasets.StateRows(np.zeros((3, 1), np.uint64), [0, 0])
    with pytest.raises(ValueError, match="label 'x' has 2 entries"):
        datasets.StateRows(np.zeros((3, 1), np.uint64), labels={"x": [1, 2]})
    with pytest.raises(ValueError, match="one row per state row"):
        datasets.StateRows(np.zeros((3, 1), np.uint64), goals=(np.zeros((2, 1), np.uint64),) * 2)
    with pytest.raises(TypeError, match="indexed by an integer"):
        datasets.StateRows.from_state_space(space)[1:3]
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks)
    with pytest.raises(ValueError, match="empty batch"):
        datasets.EncodingCollate(encoder)([])
    plain = datasets.StateRows.from_state_space(space)
    goal = (np.zeros(1, np.uint64), np.zeros(1, np.uint64))
    mixed = [plain[0], datasets.StateRow(plain[1].row, 0, plain[1].labels, goal)]
    with pytest.raises(ValueError, match="some rows of the batch have a goal"):
        datasets.EncodingCollate(encoder)(mixed)


def test_a_data_loader_encodes_batches_on_load(knowledge_base):
    torch = pytest.importorskip("torch")
    dataset = datasets.StateRows.from_knowledge_base(knowledge_base)
    encoder = encoding.FlatRelationEncoder(knowledge_base.tasks, **OPTIONS)
    collate = datasets.EncodingCollate(encoder, framework="torch")
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate)
    seen = 0
    for batch in loader:
        rows = np.arange(seen, seen + batch.graphs.num_graphs)
        expected = encoder.encode(dataset.rows[rows], dataset.task_ids[rows])
        assert isinstance(batch.graphs.relation_args, torch.Tensor)
        arrays_equal(batch.graphs, expected)
        seen += len(rows)
    assert seen == len(dataset)
