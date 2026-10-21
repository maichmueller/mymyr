"""The horizon builder (`encoding.horizon`, `encoding.horizon_of_space`), the rooted horizon encoder and the transition
encoders. The builder is checked against an independent breadth-first reference written here, its determinism across
thread counts and processes, its bounds; the transition encoders against the horizon encoder on the depth-one view of
the same states; the sibling and cousin relations against a hand-computed dag; the refusals."""

import hashlib
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import mymyr
from mymyr import datasets, encoding, rl
from conftest import TASKS, text_task, walk
from python_children import run_code
from test_encoding import derived_words

ROOT = pathlib.Path(__file__).resolve().parents[2]

VIEW_FIELDS = ("nodes", "node_offsets", "edge_offsets", "edge_src", "edge_dst", "edge_schema", "edge_binding", "task_ids")
ADDED = ("plain", "added_satisfied", "added_unsatisfied")


def rows_of(task, states):
    return np.ascontiguousarray(task.encode(states))


def reference_horizons(table, roots, depth, merge=True):
    """The horizons by the documented rules, with a dict per graph: breadth first from the root, each node's successors
    in the order of rl.expand, a state that was reached before is an edge to its node."""
    nodes, edges, node_offsets, edge_offsets = [], [], [0], [0]
    for root in roots:
        rows = [root]
        graph_edges = []
        index = {root.tobytes(): 0}
        level = [0]
        for _ in range(depth):
            following = []
            for node in level:
                expansion = rl.expand(table, rows[node][None, :])
                succ = np.asarray(expansion.succ)
                for r in range(int(expansion.offsets[1])):
                    key = succ[r].tobytes()
                    if merge and key in index:
                        target = index[key]
                    else:
                        target = len(rows)
                        rows.append(succ[r].copy())
                        if merge:
                            index[key] = target
                        following.append(target)
                    graph_edges.append((node, target, int(expansion.schema[r]), np.asarray(expansion.binding[r])))
            level = following
        # the edges are ordered by destination, then schema, then objects; equal ones keep their order of discovery
        graph_edges.sort(key=lambda e: (e[1], e[2], tuple(e[3])))
        nodes += rows
        edges += graph_edges
        node_offsets.append(len(nodes))
        edge_offsets.append(len(edges))
    width = table.label_width
    return SimpleNamespace(
        nodes=np.stack(nodes), node_offsets=np.array(node_offsets, np.int32), edge_offsets=np.array(edge_offsets, np.int32),
        edge_src=np.array([e[0] for e in edges], np.int32), edge_dst=np.array([e[1] for e in edges], np.int32),
        edge_schema=np.array([e[2] for e in edges], np.int32),
        edge_binding=np.array([e[3] for e in edges], np.int32).reshape(len(edges), width))


def assert_same_view(actual, expected):
    for name in VIEW_FIELDS[:-1]:
        assert np.array_equal(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))), name


def digest(view):
    h = hashlib.sha256()
    for name in VIEW_FIELDS + ("truncated",):
        a = np.asarray(getattr(view, name))
        h.update(f"{name}{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def blocks():
    return text_task("blocks__probBLOCKS-8-0")


@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


@pytest.fixture(scope="module")
def philosophers():
    return text_task("philosophers__p03-phil4")


# ------------------------------------------------------------------------------------------------ the builder
@pytest.mark.parametrize("name, depth", [("blocks__probBLOCKS-8-0", 3), ("gripper__prob05", 2), ("logistics00__probLOGISTICS-6-1", 2)])
@pytest.mark.parametrize("merge", [True, False])
def test_the_builder_follows_the_documented_rules(name, depth, merge):
    task = text_task(name)
    table = rl.as_table(task)
    roots = rows_of(task, walk(task, steps=3, walks=2)[:3])
    view = encoding.horizon(task, roots, depth, merge=merge)
    assert_same_view(view, reference_horizons(table, roots, depth, merge))
    assert np.array_equal(np.asarray(view.task_ids), np.zeros(len(roots), np.int32))
    assert not np.asarray(view.truncated).any() and np.asarray(view.truncated).dtype == np.uint8
    # a view is a valid input of the horizon encoder, which encodes each graph
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode="full", parent_relation=True)
    assert encoder.encode(view).num_graphs == len(roots)


def test_merging_keeps_each_state_once_and_a_tree_does_not(blocks):
    roots = rows_of(blocks, [blocks.initial_state])
    merged = encoding.horizon(blocks, roots, 3)
    tree = encoding.horizon(blocks, roots, 3, merge=False)
    nodes = np.asarray(merged.nodes)
    assert len({row.tobytes() for row in nodes}) == len(nodes) < len(tree.nodes)
    # a node is expanded once, so a merged graph has fewer edges; a state reached again is an edge to its first node
    assert len(merged.edge_src) < len(tree.edge_src)
    dst, src = np.asarray(merged.edge_dst), np.asarray(merged.edge_src)
    assert (dst <= src).any() and (np.asarray(tree.edge_dst) > np.asarray(tree.edge_src)).all()
    assert int(merged.node_offsets[1]) == len(nodes) and int(merged.edge_offsets[1]) == len(dst)


def test_a_successor_equal_to_its_state_is_a_self_loop(gripper):
    roots = rows_of(gripper, [gripper.initial_state])
    view = encoding.horizon(gripper, roots, 1)
    src, dst = np.asarray(view.edge_src), np.asarray(view.edge_dst)
    assert [(0, 0)] == [(int(s), int(d)) for s, d in zip(src, dst) if s == d]
    tree = encoding.horizon(gripper, roots, 1, merge=False)
    assert len(tree.nodes) == len(view.nodes) + 1  # the self-loop's end is a node of its own in a tree


def test_depth_zero_is_the_roots_alone(blocks):
    roots = rows_of(blocks, walk(blocks, steps=2, walks=2)[:2])
    view = encoding.horizon(blocks, roots, 0)
    assert np.array_equal(np.asarray(view.nodes), roots) and list(np.asarray(view.node_offsets)) == [0, 1, 2]
    assert len(view.edge_src) == 0 and list(np.asarray(view.edge_offsets)) == [0, 0, 0]
    assert len(encoding.horizon(blocks, np.zeros((0, roots.shape[1]), np.uint64), 2).task_ids) == 0


def test_the_builder_accepts_states_and_task_ids_of_a_table(blocks, gripper):
    states = walk(blocks, steps=2, walks=2)[:2]
    assert_same_view(encoding.horizon(blocks, states, 2), encoding.horizon(blocks, rows_of(blocks, states), 2))
    table = rl.TaskTable([blocks, blocks])
    roots = rows_of(blocks, states)
    view = encoding.horizon(table, roots, 2, task_ids=[1, 0])
    assert list(np.asarray(view.task_ids)) == [1, 0]
    single = encoding.horizon(blocks, roots, 2)
    assert_same_view(view, single)
    with pytest.raises(ValueError, match="task_ids are required"):
        encoding.horizon(table, roots, 1)
    with pytest.raises(ValueError, match="task id"):
        encoding.horizon(table, roots, 1, task_ids=[0, 2])
    with pytest.raises(ValueError, match="not a state of instance 0"):
        encoding.horizon(blocks, [gripper.initial_state], 1)


def test_the_view_does_not_depend_on_the_threads_or_the_pool(blocks):
    roots = rows_of(blocks, walk(blocks, steps=4, walks=4))
    reference = digest(encoding.horizon(blocks, roots, 3))
    for threads in (0, 2, 3, 8):
        assert digest(encoding.horizon(blocks, roots, 3, threads=threads)) == reference, threads
    pool = rl.ThreadPool(4)
    assert digest(encoding.horizon(blocks, roots, 3, pool=pool)) == reference
    assert digest(encoding.horizon(blocks, roots, 3, merge=False, pool=pool)) == digest(encoding.horizon(blocks, roots, 3, merge=False))


def test_the_view_is_the_same_in_every_process(blocks):
    roots = rows_of(blocks, walk(blocks, steps=4, walks=4))
    mine = digest(encoding.horizon(blocks, roots, 3, threads=3))
    code = f"""
import hashlib, sys
import numpy as np
import mymyr
from mymyr import encoding
task = mymyr.Task.from_text({str(TASKS / 'blocks__probBLOCKS-8-0.txt')!r})
import random
rng = random.Random(0)
out = []
for _ in range(4):
    s = task.initial_state
    out.append(s)
    for _ in range(4):
        succ = task.successor_states(s)
        if not succ:
            break
        s = rng.choice(succ)
view = encoding.horizon(task, np.ascontiguousarray(task.encode(out)), 3, threads=int(sys.argv[1]))
h = hashlib.sha256()
for name in {VIEW_FIELDS + ('truncated',)!r}:
    a = np.asarray(getattr(view, name))
    h.update(f"{{name}}{{a.dtype}}{{a.shape}}".encode())
    h.update(a.tobytes())
print(h.hexdigest())
"""
    # the walks of the child are those of conftest.walk(task, steps=4, walks=4)? only when the seeds agree: compare
    # the child's two runs with each other and the parent's run on the same rows
    first = run_code(code, "1", capture_output=True).stdout.strip()
    second = run_code(code, "4", capture_output=True).stdout.strip()
    assert first == second and len(first) == 64
    states = []
    import random

    rng = random.Random(0)
    for _ in range(4):
        s = blocks.initial_state
        states.append(s)
        for _ in range(4):
            succ = blocks.successor_states(s)
            if not succ:
                break
            s = rng.choice(succ)
    assert digest(encoding.horizon(blocks, rows_of(blocks, states), 3)) == first
    assert mine == digest(encoding.horizon(blocks, roots, 3))


def test_max_nodes_refuses_or_truncates(blocks):
    roots = rows_of(blocks, [blocks.initial_state])
    full = encoding.horizon(blocks, roots, 3)
    count = len(full.nodes)
    assert encoding.horizon(blocks, roots, 3, max_nodes=count).truncated[0] == 0
    assert_same_view(encoding.horizon(blocks, roots, 3, max_nodes=None), full)
    with pytest.raises(ValueError, match=rf"graph 0 of the horizon would have more than max_nodes = {count - 1} nodes"):
        encoding.horizon(blocks, roots, 3, max_nodes=count - 1)
    cut = encoding.horizon(blocks, roots, 3, max_nodes=count - 1, truncate=True)
    assert list(np.asarray(cut.truncated)) == [1] and len(cut.nodes) == count - 1
    # the nodes that remain are the first ones, and every edge leads to one of them
    assert np.array_equal(np.asarray(cut.nodes), np.asarray(full.nodes)[: count - 1])
    assert int(np.asarray(cut.edge_dst).max()) < count - 1
    small = encoding.horizon(blocks, roots, 3, max_nodes=1, truncate=True)
    assert len(small.nodes) == 1 and len(small.edge_src) == 0 and list(np.asarray(small.truncated)) == [1]
    with pytest.raises(ValueError, match="max_nodes"):
        encoding.horizon(blocks, roots, 1, max_nodes=0)
    # the bound is per graph
    two = rows_of(blocks, walk(blocks, steps=3, walks=2)[:2])
    sizes = [len(encoding.horizon(blocks, two[g : g + 1], 2).nodes) for g in range(2)]
    assert sizes[0] != sizes[1]
    cap = min(sizes)
    both = encoding.horizon(blocks, two, 2, max_nodes=cap, truncate=True)
    assert list(np.asarray(both.truncated)) == [int(size > cap) for size in sizes]
    assert list(np.diff(np.asarray(both.node_offsets))) == [min(size, cap) for size in sizes]


def test_the_arguments_are_checked(blocks):
    roots = rows_of(blocks, [blocks.initial_state])
    with pytest.raises(ValueError, match="depth"):
        encoding.horizon(blocks, roots, -1)
    with pytest.raises(TypeError):
        encoding.horizon(blocks, roots, 1.5)
    with pytest.raises(ValueError, match="threads"):
        encoding.horizon(blocks, roots, 1, threads=-1)
    with pytest.raises((TypeError, ValueError)):
        encoding.horizon(blocks, roots[:, :0], 1)


@pytest.fixture(scope="module")
def space(philosophers):
    result = datasets.generate(philosophers, max_states=100_000, remove_if_unsolvable=False)
    assert result.space is not None
    return result.space


def test_a_state_space_gives_the_horizons_of_its_expansion(philosophers, space):
    ids = list(range(0, 60, 7))
    roots = rows_of(philosophers, [space.state(i) for i in ids])
    from_space = encoding.horizon_of_space(space, ids, 2)
    from_states = encoding.horizon(philosophers, roots, 2)
    # the space lists a state's transitions canonically: the same nodes, edges and action labels
    assert_same_view(from_space, from_states)
    encoder = encoding.FlatHorizonEncoder(space.task, horizon_mode="full", parent_relation=True)
    assert np.array_equal(np.asarray(encoder.encode(from_space).relation_args), np.asarray(encoder.encode(from_states).relation_args))
    assert encoding.horizon_of_space(space, [0], 0).nodes.shape[0] == 1
    everything = encoding.horizon_of_space(space, None, 0)
    assert len(everything.nodes) == space.num_states
    with pytest.raises(ValueError, match="not a state of the space"):
        encoding.horizon_of_space(space, [space.num_states], 1)
    assert_same_view(encoding.horizon_of_space(space, ids, 1, merge=False), encoding.horizon(philosophers, roots, 1, merge=False))
    cut = encoding.horizon_of_space(space, ids, 3, max_nodes=7, truncate=True)
    assert max(np.diff(np.asarray(cut.node_offsets))) == 7 and np.asarray(cut.truncated).any()
    with pytest.raises(ValueError, match="max_nodes = 7"):
        encoding.horizon_of_space(space, ids, 3, max_nodes=7)


def test_a_numeric_table_and_an_unlabelled_space_have_no_horizons(philosophers):
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    with pytest.raises(ValueError, match="^mymyr: encoding: the table is numeric"):
        encoding.horizon(numeric, [numeric.initial_state], 1)
    unlabelled = datasets.generate(philosophers, max_states=100_000, remove_if_unsolvable=False, labels=False).space
    assert unlabelled is not None
    with pytest.raises(ValueError, match="^mymyr: encoding: the state space has no transition labels"):
        encoding.horizon_of_space(unlabelled, [0], 1)


# ------------------------------------------------------------------------------------------------ classes
def test_the_classes_and_their_options(blocks):
    rooted = encoding.FlatRootedHorizonEncoder(blocks)
    options = rooted.options
    assert (options["horizon_mode"], options["root_policy"], options["parent_relation"]) == ("full", "encode_only", True)
    assert not options["sibling_relation"] and not options["cousin_relation"]
    assert isinstance(rooted, encoding.FlatHorizonEncoder)
    assert not hasattr(rooted, "stream")
    kin = encoding.FlatRootedHorizonEncoder(blocks, sibling_relation=True, cousin_relation=True)
    assert {"_parent_", "_sibling_", "_cousin_"} <= set(kin.relation_names)
    assert kin.relation_lanes[kin.relation_id("_sibling_")] == "sibling"
    assert kin.relation_lanes[kin.relation_id("_cousin_")] == "cousin"
    state_slot = kin.slot_role_names.index("state_slot")
    assert kin.relation_slot_roles[kin.relation_id("_sibling_")] == [state_slot, state_slot]
    assert kin.relation_slot_roles[kin.relation_id("_cousin_")] == [state_slot, state_slot]
    # a catalog without the options has neither, and the fingerprint changes only with them
    assert "_sibling_" not in rooted.relation_names and "_cousin_" not in rooted.relation_names
    assert rooted.schema_fingerprint != kin.schema_fingerprint
    plain = encoding.FlatHorizonEncoder(blocks, horizon_mode="full", root_policy="encode_only", parent_relation=True)
    assert plain.schema_fingerprint == rooted.schema_fingerprint
    transitions = encoding.FlatTransitionEncoder(blocks)
    effects = encoding.FlatTransitionEffectsEncoder(blocks)
    assert (transitions.options["horizon_mode"], transitions.options["root_policy"]) == ("full", "exclude")
    assert (effects.options["horizon_mode"], effects.options["root_policy"]) == ("delta", "exclude")
    assert not transitions.options["parent_relation"] and not hasattr(transitions, "stream")
    assert not isinstance(transitions, encoding.FlatHorizonEncoder)
    with pytest.raises(TypeError):
        encoding.FlatTransitionEncoder(blocks, horizon_mode="full")
    with pytest.raises(TypeError):
        encoding.FlatTransitionEffectsEncoder(blocks, sibling_relation=True)
    with pytest.raises(TypeError):
        encoding.FlatRootedHorizonEncoder(blocks, history=True)
    with pytest.raises(TypeError):
        encoding.FlatRelationEncoder(blocks, sibling_relation=True)
    with pytest.raises(ValueError, match="a horizon encoder needs a horizon_mode"):
        encoding.FlatRootedHorizonEncoder(blocks, horizon_mode=None)
    with pytest.raises(ValueError, match="a horizon encoder needs a horizon_mode"):
        encoding.FlatHorizonEncoder(blocks, horizon_mode=None)
    with pytest.raises(ValueError, match="sibling_relation needs a horizon mode"):
        encoding.EncodingSchema(blocks, sibling_relation=True)
    with pytest.raises(ValueError, match="cousin_relation needs a horizon mode"):
        encoding.EncodingSchema(blocks, cousin_relation=True)
    assert encoding.EncodingSchema(blocks, horizon_mode=None).schema_fingerprint == encoding.EncodingSchema(blocks).schema_fingerprint


@pytest.mark.parametrize("make", [
    lambda t: encoding.FlatRelationEncoder(t),
    lambda t: encoding.FlatHorizonEncoder(t, horizon_mode="delta", root_policy="exclude", sibling_relation=True),
    lambda t: encoding.FlatRootedHorizonEncoder(t, sibling_relation=True, cousin_relation=True, layout="relation"),
    lambda t: encoding.FlatTransitionEncoder(t, parent_relation=True, index_dtype="int64"),
    lambda t: encoding.FlatTransitionEffectsEncoder(t, root_policy="encode_only"),
])
def test_every_encoder_pickles_as_its_class(blocks, make):
    encoder = make(blocks)
    loaded = pickle.loads(pickle.dumps(encoder))
    assert type(loaded) is type(encoder) and loaded.options == encoder.options
    assert loaded.schema_fingerprint == encoder.schema_fingerprint


# ------------------------------------------------------------------------------------------------ kinship
def dag_view(table, task, edges, nodes=7):
    """A view of one graph of `nodes` copies of the initial state with the given edges (all of the first action schema)."""
    root = rows_of(task, [task.initial_state])
    width = table.label_width
    return encoding.HorizonView(
        nodes=np.repeat(root, nodes, axis=0), node_offsets=np.array([0, nodes], np.int32),
        edge_offsets=np.array([0, len(edges)], np.int32), edge_src=np.array([e[0] for e in edges], np.int32),
        edge_dst=np.array([e[1] for e in edges], np.int32), edge_schema=np.zeros(len(edges), np.int32),
        edge_binding=np.zeros((len(edges), width), np.int32), task_ids=np.zeros(1, np.int32))


# (edges, nodes, siblings, cousins, siblings without the root's edges, cousins without them), worked out by hand: the parents
# of a node are the sources of its edges (parallel edges once, self-loops left out), siblings share a parent, cousins have
# parents that are siblings; under root_policy exclude the root's edges are no parent edges
KIN_CASES = {
    # node 3 has two parents, the edge 0 -> 1 is given twice, 4 -> 0 leads back to the root
    "two parents": ([(0, 1), (0, 2), (0, 1), (1, 3), (2, 3), (1, 4), (2, 5), (3, 6), (4, 0)], 7,
                    [(1, 2), (2, 1), (3, 4), (3, 5), (4, 3), (5, 3)],
                    [(0, 6), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (6, 0)],
                    [(3, 4), (3, 5), (4, 3), (5, 3)], []),
    # node 4 has a self-loop (not a parent) and the parents 1 and 2
    "self-loop": ([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 5), (4, 4), (5, 2), (2, 6)], 7,
                  [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (4, 6), (6, 4)],
                  [(4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)],
                  [(4, 6), (6, 4)], []),
    # 3 and 4 have both parents 1 and 2
    "diamond": ([(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4)], 5,
                [(1, 2), (2, 1), (3, 4), (4, 3)], [(3, 4), (4, 3)], [(3, 4), (4, 3)], []),
}


@pytest.mark.parametrize("case", sorted(KIN_CASES))
@pytest.mark.parametrize("root_policy", ["include", "encode_only", "exclude"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_sibling_and_cousin_pairs_of_a_dag_by_hand(blocks, case, root_policy, layout):
    edges, nodes, siblings, cousins, siblings_wr, cousins_wr = KIN_CASES[case]
    table = rl.as_table(blocks)
    view = dag_view(table, blocks, edges, nodes)
    encoder = encoding.FlatHorizonEncoder(
        blocks, horizon_mode="full", root_policy=root_policy, parent_relation=True, sibling_relation=True,
        cousin_relation=True, include_actions=False, goal_derivations=(), layout=layout)
    description = encoder.describe(encoder.encode(view), 0)
    exclude = root_policy == "exclude"

    def named(pairs):
        return [(f"target:{i}", f"target:{j}") for i, j in pairs]

    assert list(description.relations.get("_sibling_", [])) == named(siblings_wr if exclude else siblings)
    assert list(description.relations.get("_cousin_", [])) == named(cousins_wr if exclude else cousins)
    # _parent_ has one tuple per distinct edge, self-loops included, ordered by (source, destination)
    parents = sorted({(s, d) for s, d in edges if not (exclude and 0 in (s, d))})
    assert list(description.relations["_parent_"]) == named(parents)


def test_kinship_of_a_tree_and_of_a_self_loop(blocks):
    table = rl.as_table(blocks)
    tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    encoder = encoding.FlatRootedHorizonEncoder(blocks, sibling_relation=True, cousin_relation=True, include_actions=False,
                                                goal_derivations=())
    relations = encoder.describe(encoder.encode(dag_view(table, blocks, tree, 6)), 0).relations
    assert list(relations["_sibling_"]) == [("target:1", "target:2"), ("target:2", "target:1"), ("target:3", "target:4"),
                                            ("target:4", "target:3")]
    assert sorted(relations["_cousin_"]) == sorted([("target:3", "target:5"), ("target:5", "target:3"),
                                                    ("target:4", "target:5"), ("target:5", "target:4")])
    # a self-loop does not make a node a parent: 0 -> 0, 0 -> 1 leaves node 1 without a sibling, and 1 -> 1 with 0 -> 1, 0 -> 2
    # leaves 1 and 2 siblings of each other only
    loop = encoder.describe(encoder.encode(dag_view(table, blocks, [(0, 0), (0, 1)], 2)), 0).relations
    assert "_sibling_" not in loop and "_cousin_" not in loop
    assert list(loop["_parent_"]) == [("target:0", "target:0"), ("target:0", "target:1")]
    both = encoder.describe(encoder.encode(dag_view(table, blocks, [(0, 1), (1, 1), (0, 2)], 3)), 0).relations
    assert list(both["_sibling_"]) == [("target:1", "target:2"), ("target:2", "target:1")]
    assert "_cousin_" not in both
    # no edges, no pairs
    assert "_sibling_" not in encoder.describe(encoder.encode(dag_view(table, blocks, [], 3)), 0).relations


# ------------------------------------------------------------------------------------------------ transitions
def same_batch(a, b):
    for name in a.as_dict():
        x, y = a.as_dict()[name], b.as_dict()[name]
        if isinstance(x, np.ndarray) or hasattr(x, "shape"):
            assert np.array_equal(np.asarray(x), np.asarray(y)), name
        else:
            assert x == y, name


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "philosophers__p03-phil4"])
@pytest.mark.parametrize("cls, mode", [(encoding.FlatTransitionEncoder, "full"), (encoding.FlatTransitionEffectsEncoder, "delta")])
@pytest.mark.parametrize("options", [
    {},
    {"layout": "relation", "index_dtype": "int64", "goal_derivations": ADDED, "include_actions": False},
    {"root_policy": "include", "parent_relation": True, "max_goal_level": 0},
    {"root_policy": "encode_only", "parent_relation": True, "nullary": False, "include_static": False},
])
def test_the_transition_encoders_equal_the_horizon_encoder_on_the_depth_one_view(name, cls, mode, options):
    task = text_task(name)
    states = walk(task, steps=4, walks=3)[:4]
    roots = rows_of(task, states)
    if mode == "full" and options.get("goal_derivations") == ADDED:  # the added derivations are the delta mode's
        options = {**options, "goal_derivations": ("plain", "satisfied", "unsatisfied")}
    transitions = cls(task, **options)
    horizon_options = {"root_policy": "exclude", **options}
    reference = encoding.FlatHorizonEncoder(task, horizon_mode=mode, **horizon_options)
    expected = reference.encode(encoding.horizon(task, roots, 1, merge=False))
    batch = transitions.encode(roots)
    assert batch.schema.schema_fingerprint == expected.schema.schema_fingerprint
    same_batch(batch, expected)
    # States, a given expansion and the threads make no difference
    same_batch(transitions.encode(states), expected)
    table = rl.as_table(task)
    same_batch(transitions.encode(roots, actions=rl.expand(table, roots)), expected)
    same_batch(transitions.encode(roots, threads=3), expected)
    same_batch(transitions.encode(roots, pool=rl.ThreadPool(2)), expected)


def test_a_transition_graph_alone_equals_the_same_graph_in_a_batch(philosophers):
    states = walk(philosophers, steps=5, walks=3)[:5]
    for cls in (encoding.FlatTransitionEncoder, encoding.FlatTransitionEffectsEncoder):
        encoder = cls(philosophers, goal_derivations=ADDED if cls is encoding.FlatTransitionEffectsEncoder else ("plain",))
        batch = encoder.encode(states)
        for g, state in enumerate(states):
            alone = encoder.encode([state])
            a, b = encoder.describe(batch, g), encoder.describe(alone, 0)
            assert (a.entities, a.relations) == (b.entities, b.relations)
        graph_major = encoder.describe(batch, 2)
        relation_major = cls(philosophers, layout="relation", goal_derivations=encoder.options["goal_derivations"])
        other = relation_major.describe(relation_major.encode(states), 2)
        assert (graph_major.entities, graph_major.relations) == (other.entities, other.relations)


def test_derived_atoms_and_goals_of_the_parents_are_used(philosophers):
    table = rl.as_table(philosophers)
    states = walk(philosophers, steps=3, walks=2)[:3]
    roots = rows_of(philosophers, states)
    encoder = encoding.FlatTransitionEncoder(philosophers)
    expected = encoder.encode(roots)
    # the parents' derived atoms given with an expansion that holds the successors' derived atoms: the same batch
    expansion = rl.expand(table, roots, derived=True)
    same_batch(encoder.encode(roots, derived=derived_words(philosophers, states), actions=expansion), expected)
    with pytest.raises(ValueError, match="derived needs successors"):
        encoder.encode(roots, derived=derived_words(philosophers, states), actions=rl.expand(table, roots))


def test_the_goals_of_the_parents_replace_the_instances_goal(blocks):
    table = rl.as_table(blocks)
    states = walk(blocks, steps=3, walks=2)[:3]
    roots = rows_of(blocks, states)
    # the goal of another state: its fluent atoms as the goal
    pos, neg = (np.asarray(m) for m in rl.goal_masks(table))
    goal = (np.repeat(roots[:1], len(roots), axis=0), np.zeros_like(np.repeat(pos, len(roots), axis=0)))
    for cls, mode in ((encoding.FlatTransitionEncoder, "full"), (encoding.FlatTransitionEffectsEncoder, "delta")):
        encoder = cls(blocks)
        reference = encoding.FlatHorizonEncoder(blocks, horizon_mode=mode, root_policy="exclude")
        expected = reference.encode(encoding.horizon(blocks, roots, 1, merge=False), goals=goal)
        same_batch(encoder.encode(roots, goals=goal), expected)
        shared = (goal[0][:1], goal[1][:1])
        same_batch(encoder.encode(roots, goals=shared), reference.encode(encoding.horizon(blocks, roots, 1, merge=False), goals=shared))
        assert not np.array_equal(np.asarray(encoder.encode(roots, goals=goal).relation_args),
                                  np.asarray(encoder.encode(roots).relation_args))


def is_delta(name):
    """Whether a relation holds what a node adds to or deletes from its root: `[+]p`, `[-]p`, `[+]p[g][sat+]`, ..."""
    return name.startswith(("[+]", "[-]")) and ("[g]" not in name or name.endswith(("[sat+]", "[sat-]")))


def test_a_self_loop_is_a_node_without_deltas(gripper):
    table = rl.as_table(gripper)
    roots = rows_of(gripper, [gripper.initial_state])
    expansion = rl.expand(table, roots)
    succ = np.asarray(expansion.succ)
    count = int(expansion.offsets[1])
    loops = [r for r in range(count) if (succ[r] == roots[0]).all()]
    assert loops and len(loops) < count
    effects = encoding.FlatTransitionEffectsEncoder(gripper, goal_derivations=ADDED, include_actions=False)
    relations = effects.describe(effects.encode(roots), 0).relations
    deltas = {n: ts for n, ts in relations.items() if is_delta(n)}
    assert deltas
    for row in loops:
        assert not [n for n, ts in deltas.items() if any(f"target:{row + 1}" in t for t in ts)]
    moving = [r for r in range(count) if r not in loops]
    assert all(any(f"target:{r + 1}" in t for ts in deltas.values() for t in ts) for r in moving)
    # the full mode lists the loop's node with the facts of its state
    full = encoding.FlatTransitionEncoder(gripper, include_actions=False)
    node = f"target:{loops[0] + 1}"
    assert any(node in t for ts in full.describe(full.encode(roots), 0).relations.values() for t in ts)
    # in a merged horizon the loop is an edge on the root, which has no delta
    delta = encoding.FlatHorizonEncoder(gripper, horizon_mode="delta", goal_derivations=ADDED, include_actions=False,
                                        parent_relation=True)
    merged = delta.describe(delta.encode(encoding.horizon(gripper, roots, 1)), 0).relations
    assert ("target:0", "target:0") in merged["_parent_"]
    assert not [n for n, ts in merged.items() if is_delta(n) and any("target:0" in t for t in ts)]


def test_the_transition_encoders_refuse_what_they_do_not_do(blocks):
    roots = rows_of(blocks, [blocks.initial_state])
    encoder = encoding.FlatTransitionEncoder(blocks)
    with pytest.raises(ValueError, match="task_ids"):
        encoding.FlatTransitionEncoder(rl.TaskTable([blocks, blocks])).encode(roots)
    other = rl.expand(rl.as_table(blocks), rows_of(blocks, walk(blocks, steps=2, walks=2)[:2]))
    with pytest.raises(ValueError, match="actions expand 2 states"):
        encoder.encode(roots, actions=other)
    with pytest.raises(AttributeError):
        encoder.stream()
    with pytest.raises(TypeError):
        encoder.encode(roots, nodes=roots)
    # capacities work as for the other encoders
    sizes = encoder.encode(roots).sizes
    assert sizes is None
    capacity = encoder.encode(roots, capacities={"graphs": 1, "entities": 4096, "args": 8192, "targets": 4096})
    assert not capacity.sizes["overflow"]
