"""The Simple graph of the relational encoder (``SimpleEncoder``, ``outputs=('relations', 'simple')``): the atom-object
incidence graph of the tuples. Every batch is compared with a reference built from its relation arrays by the definition,
for states, subgoal layers, history, horizons and transitions, both layouts and index types; single rows, slices,
streams, threads and capacities, the labels across instances, constants, pickling, the refusals and a PyG smoke test
follow."""

import pickle
import random

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from horizon_support import expand_horizons
from test_encoding_layers import rows_of

SIMPLE_ARRAYS = ("node_offsets", "node_label", "edge_offsets", "edge_index", "edge_label")
RELATION_ARRAYS = ("relation_counts", "relation_offsets", "relation_arg_offsets", "relation_args")
GOALS = ("plain", "satisfied", "unsatisfied")
BOTH = ("relations", "simple")
TASK_NAMES = ["blocks__probBLOCKS-8-0", "gripper__prob05", "philosophers__p03-phil4", "openstacks-opt08-adl__p03"]
ROLES = 9


def needs_front_end():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")


def pddl_task(domain, problem):
    needs_front_end()
    directory = ROOT / "tests/data/pddl" / domain
    return mymyr.Task.from_pddl(directory / "domain.pddl", directory / f"{problem}.pddl")


def reference(batch):
    """The Simple arrays of a batch built from its relation arrays and entity roles by the definition."""
    schema = batch.schema
    graphs, relations = batch.num_graphs, batch.num_relations
    k = len(schema.simple_edge_label_names) // 2
    arity = schema.relation_arities
    constants = schema.simple_constants
    by_relation = batch.as_dict()["layout"] == 1
    entity_offsets = np.asarray(batch.entity_offsets, np.int64)
    role = np.asarray(batch.entity_role, np.int64)
    offsets = np.asarray(batch.relation_offsets, np.int64)
    arg_offsets = np.asarray(batch.relation_arg_offsets, np.int64)
    args = np.asarray(batch.relation_args, np.int64)
    node_offsets, labels, edge_offsets = [0], [], [0]
    source, target, edge_labels = [], [], []
    tuples_before = 0
    for g in range(graphs):
        first, last = entity_offsets[g], entity_offsets[g + 1]
        entity_labels = role[first:last].copy()
        for j in range(min(constants, len(entity_labels))):
            if entity_labels[j] == 0:
                entity_labels[j] = ROLES + relations + j
        labels.extend(entity_labels)
        forward = ([], [], [])
        reverse = ([], [], [])
        atoms = 0
        for r in range(relations):
            segment = r * graphs + g if by_relation else g * relations + r
            for t in range(int(offsets[segment + 1] - offsets[segment])):
                atom = int(last - first) + tuples_before + int(first) + atoms
                labels.append(ROLES + r)
                for position in range(arity[r]):
                    entity = int(args[arg_offsets[segment] + t * arity[r] + position]) + tuples_before
                    forward[0].append(entity)
                    forward[1].append(atom)
                    forward[2].append(position)
                    reverse[0].append(atom)
                    reverse[1].append(entity)
                    reverse[2].append(k + position)
                atoms += 1
        source += forward[0] + reverse[0]
        target += forward[1] + reverse[1]
        edge_labels += forward[2] + reverse[2]
        tuples_before += atoms
        node_offsets.append(int(last) + tuples_before)
        edge_offsets.append(len(source))
    return {
        "node_offsets": np.array(node_offsets),
        "node_label": np.array(labels, np.int64),
        "edge_offsets": np.array(edge_offsets),
        "edge_index": np.array([source, target], np.int64).reshape(2, -1),
        "edge_label": np.array(edge_labels, np.int64),
    }


def expect_simple_of(batch):
    """The Simple arrays of ``batch`` are the reference, and its counts follow the relation arrays."""
    expected = reference(batch)
    for name in SIMPLE_ARRAYS:
        assert np.array_equal(np.asarray(getattr(batch, name)), expected[name]), name
    nodes = int(batch.node_offsets[-1])
    edges = int(batch.edge_offsets[-1])
    assert nodes == int(batch.entity_offsets[-1]) + int(batch.relation_offsets[-1])
    assert edges == 2 * int(batch.relation_arg_offsets[-1])
    assert np.asarray(batch.edge_index).shape == (2, edges)
    assert np.asarray(batch.node_label).shape == (nodes,) and np.asarray(batch.edge_label).shape == (edges,)
    # per graph: V = entities + tuples, E = 2 * tuple arguments
    by_relation = batch.as_dict()["layout"] == 1
    graphs, relations = batch.num_graphs, batch.num_relations
    arities = batch.schema.relation_arities
    for g in range(graphs):
        tuples = arguments = 0
        for r in range(relations):
            s = r * graphs + g if by_relation else g * relations + r
            count = int(batch.relation_offsets[s + 1] - batch.relation_offsets[s])
            tuples += count
            arguments += count * arities[r]
        entities = int(batch.entity_offsets[g + 1] - batch.entity_offsets[g])
        assert int(batch.node_offsets[g + 1] - batch.node_offsets[g]) == entities + tuples
        assert int(batch.edge_offsets[g + 1] - batch.edge_offsets[g]) == 2 * arguments
    # the labels are in their vocabularies and the edge labels give the position
    k = len(batch.schema.simple_edge_label_names) // 2
    assert np.all(np.asarray(batch.node_label) < len(batch.schema.simple_label_names))
    assert np.all(np.asarray(batch.edge_label) < 2 * k)
    # an edge and its reverse
    index = np.asarray(batch.edge_index)
    label = np.asarray(batch.edge_label)
    for g in range(graphs):
        first, last = int(batch.edge_offsets[g]), int(batch.edge_offsets[g + 1])
        half = (last - first) // 2
        forward, backward = slice(first, first + half), slice(first + half, last)
        assert np.array_equal(index[0, forward], index[1, backward]) and np.array_equal(index[1, forward], index[0, backward])
        assert np.array_equal(label[forward] + k, label[backward])


def same_arrays(a, b, names=SIMPLE_ARRAYS):
    for name in names:
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


def both(task, **options):
    return encoding.FlatRelationEncoder(task, outputs=BOTH, **options)


# ------------------------------------------------------------------------------------------------ states
@pytest.mark.parametrize("name", TASK_NAMES)
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_the_graph_is_the_incidence_graph_of_the_tuples_in_both_layouts(name, index):
    task = text_task(name)
    states = walk(task, steps=10, walks=2)
    actions = rl.expand(rl.as_table(task), states)
    for options in (
        {"include_actions": False},
        {"goal_derivations": GOALS, "target_sources": ("goal",), "include_actions": False},
        {"goal_derivations": GOALS, "target_sources": ("goal", "action")},
        {"nullary": False, "equality": True, "include_static": False, "include_derived": False},
    ):
        given = actions if options.get("include_actions", True) else None
        batches = []
        for layout in ("graph", "relation"):
            encoder = both(task, index_dtype=index, layout=layout, **options)
            batch = encoder.encode(states, actions=given)
            expect_simple_of(batch)
            assert batch.schema.outputs == BOTH
            batches.append(batch)
        same_arrays(*batches)
        assert np.asarray(batches[0].node_offsets).dtype == np.dtype(index)
        assert np.asarray(batches[0].node_label).dtype == np.int32 and np.asarray(batches[0].edge_label).dtype == np.int32


def test_layers_and_history_are_nodes_and_edges_like_the_other_tuples():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=2)
    layers, history = rows_of(random.Random(3), states, 2, 3)
    batches = []
    for layout in ("graph", "relation"):
        encoder = both(
            task, layout=layout, max_goal_level=2, history=True, include_actions=False, goal_derivations=GOALS,
            target_sources=("goal", "subgoal", "history"))
        batch = encoder.encode(
            states, subgoals=encoding.subgoal_masks(encoder, layers), history=encoding.history_masks(encoder, history))
        expect_simple_of(batch)
        batches.append(batch)
    assert batches[0].history_offsets[-1] > 0
    sub = [i for i, n in enumerate(batches[0].schema.relation_names) if "sg]" in n]
    assert batches[0].relation_counts[:, sub].sum() > 0
    same_arrays(*batches)


def test_the_simple_encoder_is_the_simple_output_of_the_relation_encoder():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=2)
    options = {"goal_derivations": GOALS, "target_sources": ("goal", "action")}
    actions = rl.expand(rl.as_table(task), states)
    simple = encoding.SimpleEncoder(task, **options)
    full = both(task, **options)
    batch = simple.encode(states, actions=actions)
    other = full.encode(states, actions=actions)
    same_arrays(batch, other)
    same_arrays(batch, encoding.FlatRelationEncoder(task, outputs=("simple",), **options).encode(states, actions=actions))
    expect_simple_of(other)
    assert isinstance(simple, encoding.FlatRelationEncoder)
    assert simple.schema.outputs == ("simple",) and full.schema.outputs == BOTH
    assert simple.options["outputs"] == ("simple",) and encoding.FlatRelationEncoder(task).options["outputs"] == ("relations",)
    # the arrays the entities map back with are there, the relation arrays are not
    for name in ("task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref", *SIMPLE_ARRAYS):
        assert getattr(batch, name) is not None
        assert np.array_equal(np.asarray(getattr(batch, name)), np.asarray(getattr(other, name)))
    for name in RELATION_ARRAYS:
        with pytest.raises(AttributeError, match=f"this batch has no {name}"):
            getattr(batch, name)
    assert "relation_args" not in batch.as_dict() and "edge_index" in batch.as_dict()
    # a relation encoder's batch has no Simple arrays
    plain = encoding.FlatRelationEncoder(task, **options).encode(states, actions=actions)
    for name in SIMPLE_ARRAYS:
        with pytest.raises(AttributeError, match=f"this batch has no {name}"):
            getattr(plain, name)
    assert simple.schema_fingerprint != full.schema_fingerprint != encoding.FlatRelationEncoder(task, **options).schema_fingerprint


def test_the_empty_batch_and_single_rows():
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    for index in ("int32", "int64"):
        encoder = both(task, index_dtype=index, include_actions=False)
        empty = encoder.encode(np.zeros((0, table.words), np.uint64))
        expect_simple_of(empty)
        assert np.asarray(empty.node_offsets).tolist() == [0] and np.asarray(empty.edge_offsets).tolist() == [0]
        assert np.asarray(empty.node_label).shape == (0,) and np.asarray(empty.edge_index).shape == (2, 0)
        states = walk(task, steps=9, walks=2)
        batch = encoder.encode(states)
        node_offsets, edge_offsets = np.asarray(batch.node_offsets), np.asarray(batch.edge_offsets)
        for g, state in enumerate(states):
            single = encoder.encode(state)
            expect_simple_of(single)
            nodes, edges = slice(node_offsets[g], node_offsets[g + 1]), slice(edge_offsets[g], edge_offsets[g + 1])
            assert np.array_equal(np.asarray(single.node_label), np.asarray(batch.node_label)[nodes])
            assert np.array_equal(np.asarray(single.edge_label), np.asarray(batch.edge_label)[edges])
            assert np.array_equal(np.asarray(single.edge_index), np.asarray(batch.edge_index)[:, edges] - node_offsets[g])


def test_threads_give_the_same_batch():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=1400, walks=2)
    for outputs in (BOTH, ("simple",)):
        encoder = encoding.FlatRelationEncoder(task, outputs=outputs, include_actions=False)
        expected = encoder.encode(states)
        same_arrays(encoder.encode(states, threads=4), expected)
        assert encoder.encode(states, threads=3).edge_offsets[-1] == expected.edge_offsets[-1]


def test_a_table_of_several_instances():
    needs_front_end()
    directory = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(directory / "domain.pddl", directory / f"{p}.pddl") for p in ("smedium", "probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    rows, ids = [], []
    for i, task in enumerate(tasks):
        states = walk(task, steps=6, walks=2)
        rows.append(np.ascontiguousarray(task.encode(states)))
        ids += [i] * len(states)
    words = max(r.shape[1] for r in rows)
    padded = np.concatenate([np.pad(r, ((0, 0), (0, table.words - r.shape[1]))) for r in rows])
    for layout in ("graph", "relation"):
        encoder = both(table, layout=layout, include_actions=False)
        batch = encoder.encode(padded, ids)
        expect_simple_of(batch)
    assert words <= table.words


# ------------------------------------------------------------------------------------------------ call forms
def test_streams_flush_the_batch_encode_gives():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=14, walks=3)
    for outputs in (BOTH, ("simple",)):
        encoder = encoding.FlatRelationEncoder(task, outputs=outputs, include_actions=False)
        expected = encoder.encode(states)
        stream = encoder.stream()
        for start in range(0, len(states), 5):
            stream.append(states[start : start + 5])
        same_arrays(stream.flush(), expected)
        mutable = encoder.mutable_stream()
        mutable.append(list(range(len(states))), states)
        same_arrays(mutable.flush(), expected)


def test_capacities_hold_the_graph_in_their_first_entries():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=14, walks=2)
    for outputs in (BOTH, ("simple",)):
        encoder = encoding.FlatRelationEncoder(task, outputs=outputs, include_actions=False)
        expected = encoder.encode(states)
        entities, args = int(expected.entity_offsets[-1]), int(expected.edge_offsets[-1]) // 2
        padded = encoder.encode(states, capacities={"graphs": len(states), "entities": entities + 5, "args": args + 11})
        sizes = padded.sizes
        assert not sizes["overflow"] and sizes["args"] == args and sizes["entities"] == entities
        nodes, edges = int(expected.node_offsets[-1]), 2 * args
        assert sizes["entities"] + sizes["tuples"] == nodes
        label = np.asarray(padded.node_label)
        assert label.shape == (entities + 5 + args + 11,)
        assert np.array_equal(label[:nodes], np.asarray(expected.node_label)) and not label[nodes:].any()
        index = np.asarray(padded.edge_index)
        assert index.shape == (2, 2 * (args + 11))
        assert np.array_equal(index[:, :edges], np.asarray(expected.edge_index)) and not index[:, edges:].any()
        assert np.array_equal(np.asarray(padded.edge_label)[:edges], np.asarray(expected.edge_label))
        assert not np.asarray(padded.edge_label)[edges:].any()
        assert np.array_equal(np.asarray(padded.node_offsets), np.asarray(expected.node_offsets))
        # too small: the batch is all zero and says so
        small = encoder.encode(states, capacities={"graphs": len(states), "entities": entities, "args": args - 1})
        assert small.sizes["overflow"]
        for name in SIMPLE_ARRAYS:
            assert not np.asarray(getattr(small, name)).any(), name


# ------------------------------------------------------------------------------------------------ horizons
@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("root_policy", ["include", "exclude", "encode_only"])
def test_horizons_are_the_incidence_graph_of_their_tuples(mode, root_policy):
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    roots = np.ascontiguousarray(task.encode(walk(task, steps=2, walks=2)))
    view = expand_horizons(table, roots, depth=2, max_children=4)
    goals = ("plain", "satisfied", "added_satisfied", "added_unsatisfied") if mode == "delta" else ("plain", "satisfied")
    batches = []
    for layout in ("graph", "relation"):
        encoder = encoding.FlatHorizonEncoder(
            task, horizon_mode=mode, root_policy=root_policy, parent_relation=True, goal_derivations=goals,
            layout=layout, outputs=BOTH)
        batch = encoder.encode(view)
        expect_simple_of(batch)
        batches.append(batch)
    same_arrays(*batches)
    only = encoding.FlatHorizonEncoder(
        task, horizon_mode=mode, root_policy=root_policy, parent_relation=True, goal_derivations=goals, outputs=("simple",))
    same_arrays(only.encode(view), batches[0])


@pytest.mark.parametrize("cls", [encoding.FlatTransitionEncoder, encoding.FlatTransitionEffectsEncoder])
def test_transitions_are_the_incidence_graph_of_their_tuples(cls):
    task = text_task("gripper__prob05")
    states = walk(task, steps=6, walks=2)
    encoder = cls(task, goal_derivations=("plain", "satisfied"), outputs=BOTH, parent_relation=True)
    batch = encoder.encode(states)
    expect_simple_of(batch)
    same_arrays(cls(task, goal_derivations=("plain", "satisfied"), outputs=("simple",), parent_relation=True).encode(states), batch)
    assert int(batch.node_offsets[-1]) > int(batch.entity_offsets[-1])


# ------------------------------------------------------------------------------------------------ labels
def test_the_vocabulary_is_documented_in_the_schema():
    task = text_task("blocks__probBLOCKS-8-0")
    schema = encoding.EncodingSchema(task, outputs=BOTH)
    roles, relations = list(schema.entity_role_names), list(schema.relation_names)
    assert len(roles) == ROLES
    assert schema.simple_label_names == roles + relations
    k = max(schema.relation_arities)
    assert schema.simple_edge_label_names == [str(i) for i in range(k)] + [f"{i}^-1" for i in range(k)]
    assert schema.simple_constants == 0
    # the vocabulary does not depend on whether the schema stores the Simple graph
    assert encoding.EncodingSchema(task).simple_label_names == schema.simple_label_names
    assert encoding.EncodingSchema(task).schema_fingerprint != schema.schema_fingerprint
    assert encoding.EncodingSchema(task, outputs=("simple",)).schema_fingerprint != schema.schema_fingerprint
    # the names of a batch's nodes
    batch = encoding.SimpleEncoder(task).encode(task.initial_state)
    names = [schema.simple_label_names[i] for i in np.asarray(batch.node_label)]
    objects = int(batch.num_objects[0])
    assert names[:objects] == ["object"] * objects
    assert set(names[int(batch.entity_offsets[-1]) :]) <= set(relations)


@pytest.mark.parametrize("constant_labels", [False, True])
@pytest.mark.parametrize("domain, problems", [("blocks", ("smedium", "probBLOCKS-8-0")), ("gripper", ("p01", "p02"))])
def test_labels_are_the_same_in_every_instance_of_a_domain(domain, problems, constant_labels):
    needs_front_end()
    directory = ROOT / "tests/data/pddl" / domain
    tasks = [mymyr.Task.from_pddl(directory / "domain.pddl", directory / f"{p}.pddl") for p in problems]
    options = {"include_actions": False, "constant_labels": constant_labels}
    alone = [encoding.SimpleEncoder(task, **options) for task in tasks]
    table = rl.TaskTable(tasks)
    together = encoding.SimpleEncoder(table, **options)
    assert alone[0].schema.simple_label_names == alone[1].schema.simple_label_names == together.schema.simple_label_names
    assert alone[0].schema_fingerprint == alone[1].schema_fingerprint == together.schema_fingerprint
    assert (together.schema.simple_constants > 0) == (constant_labels and domain == "gripper")
    for i, task in enumerate(tasks):
        states = walk(task, steps=4, walks=1)
        words = np.ascontiguousarray(task.encode(states))
        rows = np.pad(words, ((0, 0), (0, table.words - words.shape[1])))
        one = alone[i].encode(states)
        shared = together.encode(rows, [i] * len(states))
        for name in ("node_label", "edge_label", "node_offsets", "edge_offsets", "edge_index"):
            assert np.array_equal(np.asarray(getattr(one, name)), np.asarray(getattr(shared, name))), name


def test_constants_have_labels_of_their_own_when_asked():
    task = pddl_task("gripper", "p01")
    states = walk(task, steps=8, walks=2)
    plain = encoding.SimpleEncoder(task)
    labelled = encoding.SimpleEncoder(task, constant_labels=True)
    constants = labelled.schema.simple_constants
    assert constants > 0 and plain.schema.simple_constants == 0
    assert labelled.schema_fingerprint != plain.schema_fingerprint
    names = labelled.schema.simple_label_names
    relations = labelled.num_relations
    assert len(names) == ROLES + relations + constants and names[:-constants] == plain.schema.simple_label_names
    assert names[ROLES + relations :] == list(task.objects[:constants])
    a, b = plain.encode(states), labelled.encode(states)
    for name in ("node_offsets", "edge_offsets", "edge_index", "edge_label"):
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))
    entity_offsets = np.asarray(b.entity_offsets)
    la, lb = np.asarray(a.node_label), np.asarray(b.node_label)
    offsets = np.asarray(b.node_offsets)
    changed = np.zeros(len(la), bool)
    for g in range(b.num_graphs):
        base = offsets[g]
        assert entity_offsets[g + 1] - entity_offsets[g] >= constants
        changed[base : base + constants] = True
        assert np.array_equal(lb[base : base + constants], ROLES + relations + np.arange(constants))
        assert not la[base : base + constants].any()
    assert np.array_equal(la[~changed], lb[~changed])
    expect_simple_of(both(task, constant_labels=True).encode(states))
    assert encoding.SimpleEncoder(task, constant_labels=False).schema_fingerprint == plain.schema_fingerprint
    assert pickle.loads(pickle.dumps(labelled)).options["constant_labels"] is True


# ------------------------------------------------------------------------------------------------ pickling and refusals
def test_encoders_pickle_with_their_outputs():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=6, walks=1)
    for encoder in (
        encoding.SimpleEncoder(task, include_actions=False, index_dtype="int64"),
        both(task, include_actions=False),
        encoding.FlatHorizonEncoder(task, horizon_mode="full", outputs=BOTH, goal_derivations=()),
    ):
        loaded = pickle.loads(pickle.dumps(encoder))
        assert type(loaded) is type(encoder) and loaded.options == encoder.options
        assert loaded.schema_fingerprint == encoder.schema_fingerprint
    encoder = encoding.SimpleEncoder(task, include_actions=False)
    loaded = pickle.loads(pickle.dumps(encoder))
    same_arrays(loaded.encode(states), encoder.encode(states))


def test_refusals():
    task = text_task("blocks__probBLOCKS-8-0")
    with pytest.raises(ValueError, match="predicate_virtual_nodes"):
        encoding.SimpleEncoder(task, predicate_virtual_nodes=True)
    with pytest.raises(ValueError, match="predicate_virtual_nodes"):
        encoding.FlatRelationEncoder(task, outputs=BOTH, predicate_virtual_nodes=True)
    with pytest.raises(ValueError, match="entity_arrays"):
        encoding.SimpleEncoder(task, entity_arrays="auto")
    with pytest.raises(ValueError, match="layout"):
        encoding.SimpleEncoder(task, layout="relation")
    with pytest.raises(ValueError, match="emit_counts"):
        encoding.SimpleEncoder(task, emit_counts=False)
    with pytest.raises(ValueError, match="constant_labels"):
        encoding.FlatRelationEncoder(task, constant_labels=True)
    with pytest.raises(ValueError, match="at least one"):
        encoding.FlatRelationEncoder(task, outputs=())
    with pytest.raises(ValueError, match=r"outputs must be one of 'relations', 'simple', got 'color'"):
        encoding.FlatRelationEncoder(task, outputs=("color",))
    with pytest.raises(TypeError, match="outputs"):
        encoding.SimpleEncoder(task, outputs=("simple",))
    with pytest.raises(TypeError):
        encoding.SimpleEncoder(task, horizon_mode="full")
    assert encoding.FlatRelationEncoder(task, outputs=BOTH, layout="relation", emit_counts=False).options["layout"] == "relation"


def test_the_name_helpers_and_the_hgraph_view_need_the_relation_arrays():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=3, walks=1)
    simple = encoding.SimpleEncoder(task, include_actions=False)
    batch = simple.encode(states)
    with pytest.raises(ValueError, match="holds only the Simple graph"):
        simple.entity_names(batch, 0)
    with pytest.raises(ValueError, match="holds only the Simple graph"):
        simple.describe(batch, 0)
    with pytest.raises(ValueError):
        encoding.hgraph(batch)
    # next to the relation arrays the names and the view work
    full = both(task, include_actions=False, layout="relation")
    mixed = full.encode(states)
    assert full.entity_names(mixed, 0)
    assert encoding.hgraph(mixed).num_graphs == len(states)


def test_numeric_tasks_are_refused():
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    with pytest.raises(ValueError, match="^mymyr: encoding: numeric"):
        encoding.SimpleEncoder(numeric)
    with pytest.raises(ValueError, match="^mymyr: encoding: numeric"):
        encoding.FlatRelationEncoder(numeric, outputs=BOTH)


def test_the_device_does_not_encode_simple():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    cuda = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend")
    if not cuda.available():
        pytest.skip("mymyr sees no CUDA device")
    task = text_task("gripper__prob05")
    rows = torch.from_numpy(np.ascontiguousarray(task.encode([task.initial_state])).view(np.int64)).cuda()
    for encoder in (encoding.SimpleEncoder(task), both(task)):
        with pytest.raises(ValueError, match="^mymyr: encoding: the device does not encode Simple"):
            encoder.encode(rows)


def test_pyg_accepts_a_batch_of_graphs():
    pytest.importorskip("torch")
    geometric = pytest.importorskip("torch_geometric")
    import torch

    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=6, walks=1)
    batch = encoding.SimpleEncoder(task, include_actions=False).encode(states, framework="torch")
    graphs = []
    node_offsets, edge_offsets = batch.node_offsets, batch.edge_offsets
    for g in range(batch.num_graphs):
        nodes = slice(int(node_offsets[g]), int(node_offsets[g + 1]))
        edges = slice(int(edge_offsets[g]), int(edge_offsets[g + 1]))
        graphs.append(
            geometric.data.Data(
                x=batch.node_label[nodes].long(),
                edge_index=batch.edge_index[:, edges].long() - int(node_offsets[g]),
                edge_attr=batch.edge_label[edges].long(),
            )
        )
    collated = geometric.data.Batch.from_data_list(graphs)
    assert collated.num_graphs == len(states)
    assert collated.x.shape == (int(node_offsets[-1]),) and collated.edge_index.shape == (2, int(edge_offsets[-1]))
    assert torch.equal(collated.edge_index, batch.edge_index.long())
    assert torch.equal(collated.x, batch.node_label.long())
    assert geometric.utils.is_undirected(collated.edge_index)
