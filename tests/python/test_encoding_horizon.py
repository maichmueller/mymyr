"""The Python API of the horizon encoder: FlatHorizonEncoder and HorizonView. The graphs are checked by name against
hand-computed horizons, a graph encoded alone against the same graph in a batch, the threads against each other, and the
refusals (streams, the device's host-only arrays, invalid views)."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import text_task, walk
from horizon_support import expand_horizons
from test_encoding import derived_words


def rows_of(task, states):
    return np.ascontiguousarray(task.encode(states))


def horizon_of(task, depth=2, max_children=None, roots=None):
    table = rl.as_table(task)
    states = roots if roots is not None else [task.initial_state]
    return table, expand_horizons(table, rows_of(task, states), depth=depth, max_children=max_children)


def encoder_of(task, **options):
    return encoding.FlatHorizonEncoder(task, **{"horizon_mode": "full", "goal_derivations": (), **options})


def graph_of(encoder, batch, graph):
    """The described graph with its position removed, comparable between a batch and a single-graph batch."""
    description = encoder.describe(batch, graph)
    return description.entities, description.relations


@pytest.fixture(scope="module")
def blocks():
    return text_task("blocks__probBLOCKS-8-0")


def test_the_classes_take_their_own_options(blocks):
    with pytest.raises(TypeError):
        encoding.FlatRelationEncoder(blocks, horizon_mode="full")
    with pytest.raises(TypeError):
        encoding.FlatHorizonEncoder(blocks)
    with pytest.raises(TypeError):
        encoding.FlatHorizonEncoder(blocks, horizon_mode="full", history=True)
    with pytest.raises(ValueError, match=r"^mymyr: encoding: a horizon encoder needs a horizon_mode \('full', 'delta' or 'action'\)"):
        encoding.FlatHorizonEncoder(blocks, horizon_mode=None)
    with pytest.raises(ValueError, match=r"horizon_mode must be one of 'full', 'delta', 'action', got 'none'"):
        encoding.FlatHorizonEncoder(blocks, horizon_mode="none")
    # the schema keeps None for no horizon
    with pytest.raises(ValueError, match="horizon_mode must be one of 'full', 'delta', 'action' or None"):
        encoding.EncodingSchema(blocks, horizon_mode="none")
    encoder = encoder_of(blocks)
    assert encoder.options["horizon_mode"] == "full" and encoder.options["root_policy"] == "include"
    assert encoding.EncodingSchema(blocks, horizon_mode="full", goal_derivations=()).schema_fingerprint == encoder.schema_fingerprint
    assert not hasattr(encoder, "stream") and not hasattr(encoder, "mutable_stream")


def test_a_view_has_the_documented_fields(blocks):
    _, view = horizon_of(blocks, depth=1)
    assert [name for name in view.__dataclass_fields__] == [
        "nodes", "node_offsets", "edge_offsets", "edge_src", "edge_dst", "edge_schema", "edge_binding", "task_ids",
        "truncated",
    ]
    assert view.nodes.shape[0] == int(view.node_offsets[-1]) and len(view.edge_src) == int(view.edge_offsets[-1])
    with pytest.raises(TypeError, match="nodes must be given"):
        encoding.HorizonView(None, view.node_offsets, view.edge_offsets)


def test_a_depth_one_horizon_by_name(blocks):
    table, view = horizon_of(blocks, depth=1)
    encoder = encoder_of(blocks, parent_relation=True)
    batch = encoder.encode(view)
    names = encoder.entity_names(batch, 0)
    objects = list(blocks.objects)
    children = int(view.node_offsets[1]) - 1
    assert names == objects + ["auxiliary"] + [f"target:{k}" for k in range(children + 1)]
    description = encoder.describe(batch, 0)
    assert description.relations["_parent_"] == tuple(("target:0", f"target:{k}") for k in range(1, children + 1))
    # the root has the initial state's atoms, every child differs from it by one action
    assert description.relations["handempty"][0] == ("target:0", "auxiliary")
    # the depth-one horizon of the initial blocks state: the root has no held block, a child that picks up or unstacks does
    held = {t[0] for t in description.relations["holding"]}
    pick = {f"target:{k + 1}" for k in range(children) if table.schema_names[int(view.edge_schema[k])] in ("pick-up", "unstack")}
    assert held == pick
    roles = [role for role, _ in description.entities]
    assert roles.count("state") == children + 1


def test_the_ref_of_a_state_entity_is_its_node_index(blocks):
    _, view = horizon_of(blocks, depth=2, max_children=3)
    encoder = encoder_of(blocks)
    batch = encoder.encode(view)
    roles = batch.schema.entity_role_names
    begin, end = (int(x) for x in batch.entity_offsets[:2])
    state_entities = [i for i in range(begin, end) if roles[int(batch.entity_role[i])] == "state"]
    assert [int(batch.entity_ref[i]) for i in state_entities] == list(range(int(view.node_offsets[1])))
    # relation `on` of node k holds the atoms of row node_offsets[0] + k of the view
    description = encoder.describe(batch, 0)
    names = {int(batch.entity_ref[i]): name for i, (_, name) in zip(state_entities, [d for d in description.entities if d[0] == "state"])}
    assert names == {k: f"target:{k}" for k in names}
    states = blocks.decode(np.asarray(view.nodes))
    for k in range(int(view.node_offsets[1])):
        on = {tuple(str(a).strip("()").split()[1:]) for a in states[k].atoms() if str(a).startswith("(on ")}
        assert on == {tuple(t[1:]) for t in description.relations["on"] if t[0] == f"target:{k}"}


@pytest.mark.parametrize("policy", ["include", "encode_only", "exclude"])
def test_root_policies_name_their_state_entities(blocks, policy):
    _, view = horizon_of(blocks, depth=1, max_children=2)
    encoder = encoder_of(blocks, root_policy=policy)
    batch = encoder.encode(view)
    states = [name for role, name in encoder.describe(batch, 0).entities if role == "state"]
    if policy == "exclude":
        assert states == ["target:1", "target:2"]
    else:
        assert states == ["target:0", "target:1", "target:2"]
    assert encoder.entity_names(batch, 0)[-len(states):] == states


@pytest.mark.parametrize("mode, derivations", [("full", ()), ("delta", ("added_satisfied", "added_unsatisfied")), ("action", ())])
def test_a_graph_encoded_alone_equals_its_graph_in_a_batch(mode, derivations):
    tasks = [text_task("blocks__probBLOCKS-8-0"), text_task("gripper__prob05")]
    task = tasks[0]
    roots = walk(task, steps=5, walks=3)[:5]
    _, view = horizon_of(task, depth=2, max_children=3, roots=roots)
    encoder = encoder_of(task, horizon_mode=mode, goal_derivations=derivations, parent_relation=True, root_policy="encode_only")
    batch = encoder.encode(view)
    assert batch.num_graphs == len(roots)
    for g in range(len(roots)):
        nodes = slice(int(view.node_offsets[g]), int(view.node_offsets[g + 1]))
        edges = slice(int(view.edge_offsets[g]), int(view.edge_offsets[g + 1]))
        single_view = encoding.HorizonView(
            np.asarray(view.nodes)[nodes], np.array([0, nodes.stop - nodes.start], np.int32),
            np.array([0, edges.stop - edges.start], np.int32), np.asarray(view.edge_src)[edges],
            np.asarray(view.edge_dst)[edges], np.asarray(view.edge_schema)[edges], np.asarray(view.edge_binding)[edges],
            np.asarray(view.task_ids)[g : g + 1])
        single = encoder.encode(single_view)
        assert graph_of(encoder, single, 0) == graph_of(encoder, batch, g)
        assert np.array_equal(single.relation_counts[0], batch.relation_counts[g])


def test_threads_and_pools_give_the_same_arrays(blocks):
    roots = rows_of(blocks, walk(blocks, steps=6, walks=40))
    table = rl.as_table(blocks)
    view = expand_horizons(table, roots, depth=1)
    encoder = encoder_of(blocks, horizon_mode="delta", goal_derivations=("added_satisfied",), parent_relation=True)
    reference = encoder.encode(view).as_dict()
    for threads in (2, 4, 0):
        other = encoder.encode(view, threads=threads).as_dict()
        assert other.keys() == reference.keys()
        for key, value in reference.items():
            assert np.array_equal(np.asarray(value), np.asarray(other[key])), (threads, key)
    pool = rl.ThreadPool(3)
    other = encoder.encode(view, pool=pool).as_dict()
    for key, value in reference.items():
        assert np.array_equal(np.asarray(value), np.asarray(other[key])), key
    with ThreadPoolExecutor(4) as executor:
        batches = list(executor.map(lambda _: encoder.encode(view, threads=2).as_dict(), range(4)))
    for other in batches:
        for key, value in reference.items():
            assert np.array_equal(np.asarray(value), np.asarray(other[key])), key


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_layouts_agree_by_name(blocks, layout):
    roots = walk(blocks, steps=4, walks=2)
    _, view = horizon_of(blocks, depth=2, max_children=2, roots=roots)
    reference = encoder_of(blocks, parent_relation=True)
    other = encoder_of(blocks, parent_relation=True, layout=layout, index_dtype="int64")
    a, b = reference.encode(view), other.encode(view)
    for g in range(len(roots)):
        assert graph_of(reference, a, g) == graph_of(other, b, g)


def test_a_table_of_several_instances_takes_task_ids():
    tasks = [text_task("blocks__probBLOCKS-8-0"), text_task("gripper__prob05")]
    first = text_task("blocks__probBLOCKS-8-0")
    table = rl.TaskTable([first, tasks[0]])
    roots = table.initial_states([0, 1])
    view = expand_horizons(table, np.asarray(roots), [0, 1], depth=1, max_children=2)
    encoder = encoding.FlatHorizonEncoder(table, horizon_mode="action", goal_derivations=())
    batch = encoder.encode(view)
    assert batch.task_ids.tolist() == [0, 1]
    bare = SimpleNamespace(**{**view.__dict__, "task_ids": None})
    with pytest.raises(ValueError, match="task_ids are required for a multi-instance table"):
        encoder.encode(bare)


def test_nodes_may_be_a_list_of_states(blocks):
    states = walk(blocks, steps=3, walks=1)
    _, view = horizon_of(blocks, depth=1, roots=states[:1])
    encoder = encoder_of(blocks, parent_relation=True)
    as_array = encoder.encode(view)
    count = int(view.node_offsets[-1])
    successors = [blocks.successor_states(states[0])]
    nodes = [states[0]] + [s for s in successors[0]][: count - 1]
    as_states = encoder.encode(SimpleNamespace(**{**view.__dict__, "nodes": nodes}))
    assert graph_of(encoder, as_states, 0) == graph_of(encoder, as_array, 0)
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="node 0 of graph 0 is not a state of instance 0"):
        encoder.encode(SimpleNamespace(**{**view.__dict__, "nodes": [other.initial_state] * count}))


def test_the_capacity_form(blocks):
    _, view = horizon_of(blocks, depth=1, max_children=3)
    encoder = encoder_of(blocks)
    exact = encoder.encode(view, capacities={"entities": 10_000, "tuples": 10_000, "args": 100_000, "targets": 1000})
    sizes = exact.sizes
    assert not sizes["overflow"] and sizes["targets"] == 4
    plain = encoder.encode(view)
    assert plain.sizes is None
    assert np.array_equal(np.asarray(plain.relation_args), np.asarray(exact.relation_args)[: sizes["args"]])
    small = encoder.encode(view, capacities={"entities": 3, "tuples": 3, "args": 3, "targets": 1})
    assert small.sizes["overflow"] and small.sizes["capacity_overflow"]
    with pytest.raises(ValueError, match="needs the key 'targets'"):
        encoder.encode(view, capacities={"entities": 1, "tuples": 1, "args": 1})


def test_a_per_graph_goal_replaces_the_instances(blocks):
    roots = walk(blocks, steps=3, walks=2)
    table, view = horizon_of(blocks, depth=1, max_children=2, roots=roots)
    encoder = encoder_of(blocks, goal_derivations=("plain", "satisfied"))
    own = encoder.encode(view)
    goal = rl.goal_masks(table, [0, 0]) if hasattr(rl, "goal_masks") else None
    if goal is None:
        pytest.skip("no goal masks")
    shared = tuple(np.ascontiguousarray(g[:1]) for g in goal)
    batch = encoder.encode(view, goals=shared)
    assert batch.num_graphs == own.num_graphs == len(roots)
    with pytest.raises(ValueError, match="goals\\[0\\] \\(goal_pos\\) has 5 rows"):
        encoder.encode(view, goals=tuple(np.zeros((5, g.shape[1]), np.uint64) for g in shared))


def test_pickling_keeps_the_class_and_the_options(blocks):
    encoder = encoder_of(blocks, horizon_mode="delta", goal_derivations=("added_satisfied",), root_policy="exclude",
                         parent_relation=True)
    clone = pickle.loads(pickle.dumps(encoder))
    assert type(clone) is encoding.FlatHorizonEncoder
    assert clone.options == encoder.options and clone.schema_fingerprint == encoder.schema_fingerprint
    plain = encoding.FlatRelationEncoder(blocks)
    assert type(pickle.loads(pickle.dumps(plain))) is encoding.FlatRelationEncoder


def test_a_flat_encoder_does_not_take_a_view(blocks):
    _, view = horizon_of(blocks, depth=1)
    with pytest.raises((TypeError, ValueError)):
        encoding.FlatRelationEncoder(blocks).encode(view)


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda v: {"node_offsets": np.array([1, 5], np.int32)}, "node_offsets must start at 0"),
        (lambda v: {"node_offsets": np.array([0, 0], np.int32)}, "graph 0 of the horizon has no nodes"),
        (lambda v: {"edge_dst": np.full_like(np.asarray(v.edge_dst), 999)}, "edge 0 of graph 0 of the horizon has an endpoint"),
        (lambda v: {"edge_schema": np.full_like(np.asarray(v.edge_schema), 999)}, "edge 0 of graph 0 .* a schema outside"),
        (lambda v: {"edge_offsets": np.array([0, 1, 2], np.int32)}, "edge_offsets must have"),
        (lambda v: {"task_ids": np.array([5], np.int32)}, "task id 5 of graph 0 of the horizon"),
        (lambda v: {"edge_src": np.asarray(v.edge_src)[:-1]}, "edge_src has"),
    ],
)
def test_invalid_views_raise_with_the_graph(blocks, change, match):
    _, view = horizon_of(blocks, depth=1, max_children=2)
    encoder = encoder_of(blocks, parent_relation=True)
    bad = SimpleNamespace(**{**view.__dict__, **change(view)})
    with pytest.raises(ValueError, match=f"^mymyr: encoding: .*{match}|{match}"):
        encoder.encode(bad)


def test_a_view_without_a_field_is_an_attribute_error(blocks):
    _, view = horizon_of(blocks, depth=1)
    bare = SimpleNamespace(**{k: v for k, v in view.__dict__.items() if k != "edge_dst"})
    with pytest.raises(AttributeError, match="the horizon view has no attribute 'edge_dst'"):
        encoder_of(blocks).encode(bare)


def with_edges(view, edges):
    """The view with the edges (source, destination, schema, binding row) in the given order."""
    width = np.asarray(view.edge_binding).shape[1]
    return encoding.HorizonView(
        nodes=view.nodes,
        node_offsets=view.node_offsets,
        edge_offsets=np.array([0, len(edges)], np.int32),
        edge_src=np.array([e[0] for e in edges], np.int32),
        edge_dst=np.array([e[1] for e in edges], np.int32),
        edge_schema=np.array([e[2] for e in edges], np.int32),
        edge_binding=np.array([e[3] for e in edges], np.int32).reshape(len(edges), width),
        task_ids=view.task_ids,
    )


@pytest.mark.parametrize("policy", ["include", "encode_only", "exclude"])
@pytest.mark.parametrize("mode", ["full", "delta", "action"])
def test_the_edge_tuples_are_distinct_and_ordered_whatever_the_edges(blocks, mode, policy):
    table, view = horizon_of(blocks, depth=1, max_children=3)
    # nodes 0 (the root) to 3; the edges below are the three of the view, then a repeated edge, an edge into the root,
    # a self-loop, a second edge into node 3 with the action of the first, and an edge between two children
    pick, unstack = list(table.schema_names).index("pick-up"), list(table.schema_names).index("unstack")
    edges = [
        (0, 1, pick, [4, -1]), (0, 2, pick, [5, -1]), (0, 3, unstack, [6, 0]),
        (0, 1, pick, [4, -1]),
        (2, 0, pick, [5, -1]),
        (1, 1, unstack, [6, 0]),
        (1, 3, unstack, [6, 0]),
        (3, 2, pick, [4, -1]),
    ]
    assert [e[:3] for e in edges[:3]] == [tuple(int(x) for x in t) for t in zip(view.edge_src, view.edge_dst, view.edge_schema)]
    encoder = encoder_of(blocks, horizon_mode=mode, root_policy=policy, parent_relation=True)
    batch = encoder.encode(with_edges(view, edges))
    rng = np.random.default_rng(0)
    for order in (edges[::-1], [edges[i] for i in rng.permutation(len(edges))]):
        other = encoder.encode(with_edges(view, order))
        for key, value in batch.as_dict().items():
            assert np.array_equal(value, other.as_dict()[key]), key
    # by hand: an action tuple per distinct (destination, objects), by destination and then objects, and none for an edge
    # into the root that is not a node; a parent tuple per distinct (source, destination) in ascending order, and none
    # for an edge that touches the root when the root is not a node
    relations = encoder.describe(batch, 0).relations
    rooted = policy != "exclude"
    pick_tuples = [("target:0", "o5")] * rooted + [("target:1", "o4"), ("target:2", "o4"), ("target:2", "o5")]
    parents = [(0, 1), (0, 2), (0, 3)] * rooted + [(1, 1), (1, 3)] + [(2, 0)] * rooted + [(3, 2)]
    assert relations["pick-up"] == tuple(pick_tuples)
    assert relations["unstack"] == (("target:1", "o6", "o0"), ("target:3", "o6", "o0"))
    assert relations["_parent_"] == tuple((f"target:{s}", f"target:{d}") for s, d in parents)
    assert not relations.get("put-down") and not relations.get("stack")


def test_given_derived_words_equal_the_evaluated_ones_of_a_horizon():
    task = text_task("philosophers__p03-phil4")
    table = rl.as_table(task)
    states = walk(task, steps=3, walks=1)
    view = expand_horizons(table, rows_of(task, states[:1]), depth=1, max_children=4)
    nodes = task.decode(np.asarray(view.nodes))
    words = derived_words(task, nodes)
    encoder = encoder_of(task, parent_relation=True)
    expected = encoder.encode(view)
    given = encoder.encode(view, derived=words)
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, given.as_dict()[key]), key
    padded = np.concatenate([words, np.ones((len(nodes), 2), np.uint64)], axis=1)
    assert np.array_equal(encoder.encode(view, derived=padded).relation_args, expected.relation_args)
    # the words are used: without derived atoms there are no derived tuples
    derived = [name for name, lane in zip(encoder.relation_names, encoder.relation_lanes) if lane == "derived"]
    blank = encoder.encode(view, derived=np.zeros_like(words))
    assert any(encoder.describe(expected, 0).relations.get(name) for name in derived)
    assert not any(encoder.describe(blank, 0).relations.get(name) for name in derived)
