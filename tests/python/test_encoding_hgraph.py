"""The HGraph view of a relation-major batch (``encoding.hgraph``): its node and edge types, the zero-copy rows (checked by
pointer identity), the reconstruction of the flat tuples from the edges for the encodings of states, horizons and
transitions, the auxiliary entity of the nullary relations, the lifetime of the view and its refusals."""

import gc
import pathlib
import threading

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from test_encoding_layers import rows_of

SYMBOL = "_symbol_"
TASK_NAMES = ["blocks__probBLOCKS-8-0", "gripper__prob05", "philosophers__p03-phil4", "openstacks-opt08-adl__p03"]
GOALS = ("plain", "satisfied", "unsatisfied")


def address(array):
    return array.__array_interface__["data"][0]


def edge_type(relation, position):
    return (relation, str(position), SYMBOL)


def view_tuples(view, schema_arities, relation, graph):
    """The tuples of ``relation`` in graph ``graph`` as an array [n, arity], read from the view's edges alone."""
    arity = schema_arities[relation]
    begin, end = (int(x) for x in np.asarray(view.ptr(relation))[graph : graph + 2])
    columns = [np.asarray(view.edge_index_1(edge_type(relation, k)))[begin:end] for k in range(arity)]
    return np.stack(columns, axis=1) if columns else np.zeros((end - begin, 0), columns[0].dtype if columns else np.int32)


def expect_view_is_the_batch(view, batch, reference, encoder):
    """The view of the relation-major ``batch`` has the tuples of the graph-major ``reference`` in every graph, its rows
    point into the batch, and its counts, ptrs and entity roles are those of the batch."""
    graphs, relations = batch.num_graphs, batch.num_relations
    arities = dict(zip(encoder.relation_names, encoder.relation_arities))
    assert view.num_graphs == graphs
    assert view.node_types == list(encoder.relation_names) + [SYMBOL]
    expected_types = [edge_type(name, k) for name, arity in arities.items() for k in range(arity)]
    assert view.edge_types == expected_types
    assert view.reverse_edge_types == [(SYMBOL, k, name) for name, k, _ in expected_types]
    ptr = np.asarray(view.ptr(SYMBOL))
    assert address(ptr) == address(batch.entity_offsets)
    assert np.array_equal(ptr, batch.entity_offsets)
    assert view.num_nodes(SYMBOL) == int(batch.entity_offsets[graphs])
    args_base, itemsize = address(batch.relation_args), batch.relation_args.itemsize
    for r, name in enumerate(encoder.relation_names):
        arity = arities[name]
        cells = np.asarray(reference.relation_counts)[:, r]
        total = int(cells.sum())
        assert view.num_nodes(name) == total
        assert np.array_equal(np.asarray(view.ptr(name)), np.concatenate([[0], np.cumsum(cells)]))
        first = int(batch.relation_arg_offsets[r * graphs])
        for k in range(arity):
            row0, row1 = view.edge_index(edge_type(name, k))
            assert view.num_edges(edge_type(name, k)) == total
            row0, row1 = np.asarray(row0), np.asarray(row1)
            assert np.array_equal(row0, np.arange(total))
            assert row1.shape == (total,) and row1.strides == (arity * itemsize,)
            assert row1.dtype == batch.relation_args.dtype == row0.dtype
            assert not row1.flags.writeable
            assert address(row1) == args_base + (first + k) * itemsize
            if total:
                assert np.shares_memory(row1, batch.relation_args) and not np.shares_memory(row0, batch.relation_args)
        for g in range(graphs):
            begin = int(reference.relation_arg_offsets[g * relations + r])
            count = int(cells[g])
            flat = np.asarray(reference.relation_args)[begin : begin + count * arity].reshape(count, arity)
            assert np.array_equal(view_tuples(view, arities, name, g), flat), (name, g)
    if not batch.implicit_entities:
        assert address(np.asarray(view.entity_role)) == address(batch.entity_role)
    roles = np.asarray(view.entity_role)
    assert roles.dtype == np.uint8 and roles.shape == (int(batch.entity_offsets[graphs]),)
    if not reference.implicit_entities:
        assert np.array_equal(roles, np.asarray(reference.entity_role))


def encode_both(task, states, task_ids=None, framework=None, **options):
    """The graph-major and the relation-major batch of the same states under the same options."""
    out = []
    for layout in ("graph", "relation"):
        encoder = encoding.FlatRelationEncoder(task, layout=layout, **options)
        out.append((encoder, encoder.encode(states, task_ids, framework=framework)))
    return out


# ------------------------------------------------------------------------------------------------ states
@pytest.mark.parametrize("name", TASK_NAMES)
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_the_edges_reconstruct_the_flat_tuples_of_a_state_batch(name, index):
    task = text_task(name)
    states = walk(task, steps=12, walks=2)
    table = rl.as_table(task)
    actions = rl.expand(table, states)
    for options in (
        {"include_actions": False},
        {"goal_derivations": GOALS, "target_sources": ("goal",), "include_actions": False},
        {"goal_derivations": GOALS, "target_sources": ("goal", "action")},
        {"nullary": False, "equality": True, "include_static": False, "include_derived": False},
    ):
        given = actions if options.get("include_actions", True) else None
        batches = []
        for layout in ("graph", "relation"):
            encoder = encoding.FlatRelationEncoder(task, index_dtype=index, layout=layout, **options)
            batches.append((encoder, encoder.encode(states, actions=given)))
        (_, reference), (encoder, batch) = batches
        expect_view_is_the_batch(encoding.hgraph(batch), batch, reference, encoder)


def test_the_edges_reconstruct_the_flat_tuples_with_layers_and_history():
    import random

    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=2)
    rng = random.Random(3)
    layers, history = rows_of(rng, states, 2, 3)
    batches = []
    for layout in ("graph", "relation"):
        encoder = encoding.FlatRelationEncoder(
            task, layout=layout, max_goal_level=2, history=True, include_actions=False, goal_derivations=GOALS,
            target_sources=("goal", "subgoal", "history"))
        batches.append((encoder, encoder.encode(
            states, subgoals=encoding.subgoal_masks(encoder, layers), history=encoding.history_masks(encoder, history))))
    (_, reference), (encoder, batch) = batches
    assert batch.relation_counts[:, [i for i, n in enumerate(encoder.relation_names) if "sg]" in n]].sum() > 0
    assert batch.history_offsets[-1] > 0
    expect_view_is_the_batch(encoding.hgraph(batch), batch, reference, encoder)


def test_implicit_entity_arrays_give_the_roles_of_objects_and_the_auxiliary_entity():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=6, walks=1)
    (_, reference), (encoder, batch) = encode_both(task, states, include_actions=False, entity_arrays="auto",
                                                  emit_counts=False)
    assert batch.implicit_entities
    view = encoding.hgraph(batch)
    roles = np.asarray(view.entity_role)
    assert roles.dtype == np.uint8 and roles.shape == (int(batch.entity_offsets[-1]),)
    names = batch.schema.entity_role_names
    for g in range(batch.num_graphs):
        begin, end = (int(x) for x in batch.entity_offsets[g : g + 2])
        objects = int(batch.num_objects[g])
        assert [names[r] for r in roles[begin : begin + objects]] == ["object"] * objects
        assert [names[r] for r in roles[begin + objects : end]] in ([], ["auxiliary"])
    # without stored roles the array is new; with them it is the batch's own
    explicit = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False).encode(states)
    assert address(np.asarray(encoding.hgraph(explicit).entity_role)) == address(explicit.entity_role)


def test_empty_and_single_row_batches():
    task = text_task("blocks__probBLOCKS-8-0")
    for index in ("int32", "int64"):
        for states in ([], [task.initial_state]):
            actions = rl.expand(rl.as_table(task), states) if states else None
            options = {"index_dtype": index, "goal_derivations": GOALS, "target_sources": ("goal", "action")}
            encoders = [encoding.FlatRelationEncoder(task, layout=layout, **options) for layout in ("graph", "relation")]
            if states:
                reference, batch = (e.encode(states, actions=actions) for e in encoders)
            else:
                reference, batch = (e.encode(np.zeros((0, rl.as_table(task).words), np.uint64)) for e in encoders)
            view = encoding.hgraph(batch)
            assert view.num_graphs == len(states)
            expect_view_is_the_batch(view, batch, reference, encoders[1])


# ------------------------------------------------------------------------------------------------ the nullary relations
def test_a_nullary_relation_has_an_edge_to_the_auxiliary_entity_exactly_where_it_has_a_tuple():
    task = mymyr.Task.from_pddl(ROOT / "tests/data/pddl/blocks/domain.pddl", ROOT / "tests/data/pddl/blocks/smedium.pddl")
    first = task.initial_state
    held = next(s for s in task.successor_states(first) if "handempty" not in {a.predicate.name for a in s.atoms()})
    encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False)
    batch = encoder.encode([first, held])
    view = encoding.hgraph(batch)
    assert encoder.relation_arities[encoder.relation_id("handempty")] == 1
    assert encoder.relation_logical_arities[encoder.relation_id("handempty")] == 0
    row1 = np.asarray(view.edge_index_1(edge_type("handempty", 0)))
    ptr = np.asarray(view.ptr("handempty"))
    assert ptr.tolist() == [0, 1, 1]  # the initial state has the atom, the state that holds a block does not
    roles = np.asarray(view.entity_role)
    assert [encoder.entity_role_names[roles[e]] for e in row1] == ["auxiliary"]
    assert int(row1[0]) == int(batch.entity_offsets[0] + batch.num_objects[0])
    # the second graph has no auxiliary entity: nothing it has is a nullary tuple
    second = roles[int(batch.entity_offsets[1]) : int(batch.entity_offsets[2])]
    assert len(second) == int(batch.num_objects[1]) and "auxiliary" not in {encoder.entity_role_names[r] for r in second}


def test_nullary_goal_literals_have_edges_to_the_auxiliary_entity():
    task = mymyr.Task.from_pddl(ROOT / "tests/data/pddl/blocks_flags/domain.pddl",
                                ROOT / "tests/data/pddl/blocks_flags/p01.pddl")
    encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False, goal_derivations=GOALS,
                                           target_sources=("goal",))
    states = walk(task, steps=8, walks=1)
    batch = encoder.encode(states)
    view = encoding.hgraph(batch)
    nullary = [n for n, a in zip(encoder.relation_names, encoder.relation_logical_arities) if a == 0]
    goals = [n for n in nullary if n.endswith("[g]") or "[g][" in n]
    assert goals, "the task has nullary goal literals"
    roles = np.asarray(view.entity_role)
    seen = {}
    for name in nullary:
        slots = encoder.relation_slot_roles[encoder.relation_id(name)]
        auxiliary = slots.index(encoder.slot_role_names.index("auxiliary_slot"))
        seen[name] = 0
        ptr = np.asarray(view.ptr(name))
        destinations = np.asarray(view.edge_index_1(edge_type(name, auxiliary)))
        for g in range(batch.num_graphs):
            for entity in destinations[ptr[g] : ptr[g + 1]]:
                seen[name] += 1
                assert encoder.entity_role_names[roles[entity]] == "auxiliary"
                assert int(entity) == int(batch.entity_offsets[g] + batch.num_objects[g])
    assert sum(seen.values()) > 0 and any(seen[n] > 0 for n in goals)


# ------------------------------------------------------------------------------------------------ horizons
def test_the_edges_reconstruct_the_flat_tuples_of_horizon_and_transition_batches():
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    roots = walk(task, steps=4, walks=2)
    horizon = encoding.horizon(table, roots, depth=2)
    cases = [
        (encoding.FlatHorizonEncoder, dict(horizon_mode="full", root_policy="include", goal_derivations=GOALS), horizon),
        (encoding.FlatHorizonEncoder, dict(horizon_mode="delta", root_policy="exclude",
                                          goal_derivations=("plain", "added_satisfied", "added_unsatisfied"),
                                          parent_relation=True), horizon),
        (encoding.FlatHorizonEncoder, dict(horizon_mode="action", root_policy="exclude"), horizon),
        (encoding.FlatRootedHorizonEncoder, dict(sibling_relation=True, cousin_relation=True), horizon),
        (encoding.FlatTransitionEncoder, {}, roots),
        (encoding.FlatTransitionEffectsEncoder, {}, roots),
    ]
    for cls, options, inputs in cases:
        for index in ("int32", "int64"):
            encoders = [cls(task, layout=layout, index_dtype=index, **options) for layout in ("graph", "relation")]
            reference, batch = (e.encode(inputs) for e in encoders)
            assert batch.num_graphs == len(roots)
            view = encoding.hgraph(batch)
            expect_view_is_the_batch(view, batch, reference, encoders[1])
    # the parent relation's edges join the state entities of a horizon
    encoder = encoding.FlatRootedHorizonEncoder(task, layout="relation")
    batch = encoder.encode(horizon)
    view = encoding.hgraph(batch)
    roles = np.asarray(view.entity_role)
    for k in (0, 1):
        destination = np.asarray(view.edge_index_1(edge_type("_parent_", k)))
        assert len(destination) and {encoder.entity_role_names[r] for r in roles[destination]} == {"state"}


# ------------------------------------------------------------------------------------------------ lifetime
def test_the_view_keeps_the_batch_alive():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=1)
    encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False)
    batch = encoder.encode(states)
    expected = {name: np.array(encoding.hgraph(batch).edge_index_1(edge_type(name, 0))) for name in ("on", "clear")}
    view = encoding.hgraph(batch)
    row = view.edge_index_1(edge_type("on", 0))  # an array of the view, not of the batch
    roles = view.entity_role
    del batch
    gc.collect()
    # the batch is gone but its arrays are in use: the view and the arrays it handed out keep them
    for name, values in expected.items():
        assert np.array_equal(view.edge_index_1(edge_type(name, 0)), values)
    assert np.array_equal(row, expected["on"])
    assert len(roles)
    del view
    gc.collect()
    assert np.array_equal(row, expected["on"])  # the arrays outlive the view
    assert len(roles)


def test_the_view_is_usable_from_many_threads():
    task = text_task("blocks__probBLOCKS-8-0")
    batch = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False).encode(walk(task, 20, walks=2))
    view = encoding.hgraph(batch)
    expected = {t: np.array(view.edge_index_1(t)) for t in view.edge_types}
    failures = []

    def work():
        try:
            for _ in range(50):
                for t, values in expected.items():
                    row0, row1 = view.edge_index(t)
                    assert np.array_equal(row1, values) and np.array_equal(row0, np.arange(len(values)))
                    assert view.num_edges(t) == len(values)
        except Exception as error:  # noqa: BLE001
            failures.append(error)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not failures


# ------------------------------------------------------------------------------------------------ frameworks
def test_row_1_is_a_strided_tensor_of_the_batch_in_torch():
    torch = pytest.importorskip("torch")
    task = text_task("blocks__probBLOCKS-8-0")
    batch = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False).encode(walk(task, 10, walks=1))
    view = encoding.hgraph(batch)
    for relation in ("on", "clear"):
        for k in range(batch.schema.relation_arities[batch.schema.relation_names.index(relation)]):
            tensor = view.edge_index_1(edge_type(relation, k), framework="torch")
            reference = np.asarray(view.edge_index_1(edge_type(relation, k)))
            assert isinstance(tensor, torch.Tensor) and tensor.shape == reference.shape
            assert tensor.data_ptr() == address(reference)
            assert tensor.tolist() == reference.tolist()
            assert view.edge_index_0(edge_type(relation, k), framework="torch").tolist() == list(range(len(reference)))
    assert view.ptr(SYMBOL, framework="torch").tolist() == batch.entity_offsets.tolist()


# ------------------------------------------------------------------------------------------------ the API and refusals
def test_edge_types_name_their_relation_and_position_and_reverse_edges_swap_the_rows():
    task = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=True)
    batch = encoder.encode(walk(task, 5, walks=1), actions=rl.expand(rl.as_table(task), walk(task, 5, walks=1)))
    view = encoding.hgraph(batch)
    assert encoding.HGraphView is type(view)
    assert view.batch.num_graphs == batch.num_graphs
    assert SYMBOL == view.node_types[-1] and view.node_types[:-1] == list(encoder.relation_names)
    forward = edge_type("stack", 2)
    reverse = (SYMBOL, "2", "stack")
    assert forward in view.edge_types and reverse in view.reverse_edge_types
    row0, row1 = view.edge_index(forward)
    back0, back1 = view.edge_index(reverse)
    assert np.array_equal(row0, back1) and np.array_equal(row1, back0)
    assert address(np.asarray(view.edge_index_0(reverse))) == address(np.asarray(view.edge_index_1(forward)))
    assert view.num_edges(reverse) == view.num_edges(forward)
    # the positions of a relation are those of its slot roles
    for name, roles in zip(encoder.relation_names, encoder.relation_slot_roles):
        assert [t for t in view.edge_types if t[0] == name] == [edge_type(name, k) for k in range(len(roles))]


def test_the_view_refuses_what_it_cannot_view():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, 4, walks=1)
    graph_major = encoding.FlatRelationEncoder(task, include_actions=False).encode(states)
    with pytest.raises(ValueError, match="^mymyr: encoding: the HGraph view needs a relation-major batch.*graph-major"):
        encoding.hgraph(graph_major)
    virtual = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False, predicate_virtual_nodes=True)
    with pytest.raises(ValueError, match="^mymyr: encoding: the HGraph view does not support predicate virtual nodes"):
        encoding.hgraph(virtual.encode(states))
    batch = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False).encode(states)
    with pytest.raises(TypeError):
        encoding.hgraph(None)
    with pytest.raises(TypeError):
        encoding.hgraph(batch.as_dict())
    view = encoding.hgraph(batch)
    with pytest.raises(KeyError, match="no node type nothing"):
        view.num_nodes("nothing")
    with pytest.raises(TypeError):
        view.num_nodes(3)
    for bad in (("on", "2", SYMBOL), ("on", "-1", SYMBOL), ("on", "x", SYMBOL), ("on", "0", "on"), (SYMBOL, "0", SYMBOL),
                ("nothing", "0", SYMBOL), ("on", "00x", SYMBOL), ("on", "", SYMBOL)):
        with pytest.raises(KeyError, match="no edge type"):
            view.edge_index(bad)
    for bad in ("on", ("on", "0"), ("on", 0, SYMBOL), ["on", "0", SYMBOL]):
        with pytest.raises(TypeError, match="an edge type is a tuple"):
            view.edge_index(bad)
    with pytest.raises(TypeError):
        view.edge_index(None)


def test_a_batch_that_overflowed_its_capacities_has_no_view():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, 6, walks=1)
    encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False)
    fits = encoder.encode(states, capacities={"graphs": 20, "entities": 400, "args": 4000})
    assert not fits.sizes["overflow"]
    view = encoding.hgraph(fits)
    reference = encoder.encode(states)
    assert view.num_nodes(SYMBOL) == int(reference.entity_offsets[-1])
    for name, arity in zip(encoder.relation_names, encoder.relation_arities):
        for k in range(arity):
            assert np.array_equal(view.edge_index_1(edge_type(name, k)), encoding.hgraph(reference).edge_index_1(edge_type(name, k)))
    overflowed = encoder.encode(states, capacities={"graphs": len(states), "entities": 3, "args": 3})
    assert overflowed.sizes["overflow"]
    with pytest.raises(ValueError, match="^mymyr: encoding: the batch overflowed its capacities"):
        encoding.hgraph(overflowed)


def test_a_device_batch_is_refused():
    with pytest.raises(TypeError, match="^mymyr: encoding: the HGraph view is of host batches"):
        encoding.hgraph(object.__new__(encoding.DeviceFlatBatch))
