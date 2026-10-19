"""Shared statics (``statics="shared"``) of the relational encoder through the Python API: the static relations are empty
in the graphs, the static tuples of each instance are stored once per batch with graph-local ids, and the blocks plus
the graphs rebuild the repeated batch exactly."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import BLOCKS, text_task, walk

TASKS = ["gripper__prob05", "blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "philosophers__p03-phil4"]
STATIC_ARRAYS = ("static_task_ids", "static_offsets", "static_arg_offsets", "static_args", "static_relations")
GOAL_ROLE, VIRTUAL_ROLE = 2, 1
PREDICATE_SLOT = 1
OPTION_SETS = {
    "default": {},
    "goals-targets": dict(goal_derivations=("plain", "satisfied", "unsatisfied"), target_sources=("goal",)),
    "virtual": dict(predicate_virtual_nodes=True, goal_derivations=("plain", "satisfied")),
    "everything": dict(
        goal_derivations=("plain", "satisfied", "unsatisfied"),
        target_sources=("goal", "action"),
        predicate_virtual_nodes=True,
        equality=True,
    ),
    "no-static": dict(include_static=False),
    "bare": dict(entity_arrays="auto", goal_derivations=(), include_actions=False, emit_counts=False),
}


def arrays(batch):
    """The arrays of a batch as int64 NumPy arrays by name."""
    return {k: np.asarray(v).astype(np.int64) for k, v in batch.as_dict().items() if np.ndim(v) > 0}


def repeated_form(batch):
    """The arrays of the repeated batch that the shared batch stands for: each graph's static cells from the block of its
    instance (ids shifted by the graph's first entity, a predicate key replaced by the graph's virtual node of that key),
    and the offsets and goal references that follow."""
    schema = batch.schema
    a = arrays(batch)
    graphs, relations = batch.num_graphs, batch.num_relations
    relation_major = schema.options["layout"] == "relation"
    arities, roles = schema.relation_arities, schema.relation_slot_roles
    statics = [int(r) for r in a["static_relations"]]
    entity_offsets, task_ids = a["entity_offsets"], a["task_ids"]
    cells = [[None] * relations for _ in range(graphs)]
    shared_tuples = np.zeros((graphs, relations), dtype=np.int64)
    static_tuples = np.zeros((graphs, relations), dtype=np.int64)
    for g in range(graphs):
        block = int(np.searchsorted(a["static_task_ids"], task_ids[g]))
        first, last = int(entity_offsets[g]), int(entity_offsets[g + 1])
        for r in range(relations):
            if r not in statics:
                s = r * graphs + g if relation_major else g * relations + r
                lo, hi = a["relation_arg_offsets"][s : s + 2]
                cells[g][r] = a["relation_args"][lo:hi]
                shared_tuples[g, r] = (hi - lo) // arities[r]
                continue
            s = block * len(statics) + statics.index(r)
            lo, hi = a["static_arg_offsets"][s : s + 2]
            rows = a["static_args"][lo:hi].reshape(-1, arities[r]).copy()
            static_tuples[g, r] = len(rows)
            for position, role in enumerate(roles[r]):
                if role == PREDICATE_SLOT:
                    nodes = {
                        int(a["entity_ref"][e]): e
                        for e in range(first, last)
                        if a["entity_role"][e] == VIRTUAL_ROLE
                    }
                    rows[:, position] = [nodes[int(key)] for key in rows[:, position]]
                else:
                    rows[:, position] += first
            cells[g][r] = rows.reshape(-1)
    if "entity_ref" in a:
        for g in range(graphs):
            for e in range(int(entity_offsets[g]), int(entity_offsets[g + 1])):
                if a["entity_role"][e] != GOAL_ROLE:
                    continue
                passed, r = 0, 0
                while r < relations and passed + shared_tuples[g, r] <= a["entity_ref"][e]:
                    passed += shared_tuples[g, r]
                    r += 1
                a["entity_ref"][e] += static_tuples[g, :r].sum()
    counts = np.zeros((graphs, relations), dtype=np.int64)
    segments = [(r, g) for r in range(relations) for g in range(graphs)] if relation_major else [
        (r, g) for g in range(graphs) for r in range(relations)
    ]
    arg_offsets, args = [0], []
    for r, g in segments:
        counts[g, r] = len(cells[g][r]) // arities[r]
        args.append(cells[g][r])
        arg_offsets.append(arg_offsets[-1] + len(cells[g][r]))
    tuple_offsets = np.concatenate([[0], np.cumsum([counts[g, r] for r, g in segments])])
    if "relation_counts" in a:
        a["relation_counts"] = counts
    a["relation_offsets"] = tuple_offsets
    a["relation_arg_offsets"] = np.asarray(arg_offsets)
    a["relation_args"] = np.concatenate(args) if args else np.zeros(0, dtype=np.int64)
    for name in STATIC_ARRAYS:
        del a[name]
    return a


def assert_rebuilds(shared_encoder, repeated_encoder, states, ids=None, **kwargs):
    shared = shared_encoder.encode(states, ids, **kwargs)
    repeated = repeated_encoder.encode(states, ids, **kwargs)
    assert shared.as_dict()["statics"] == 1 and repeated.as_dict()["statics"] == 0
    rebuilt, expected = repeated_form(shared), arrays(repeated)
    assert rebuilt.keys() == expected.keys()
    for key in expected:
        assert np.array_equal(rebuilt[key], expected[key]), key
    return shared


@pytest.mark.parametrize("options", OPTION_SETS.values(), ids=OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", TASKS)
def test_the_blocks_and_the_graphs_rebuild_the_repeated_batch(name, layout, index, options):
    task = text_task(name)
    states = walk(task, steps=12, walks=3)
    common = dict(layout=layout, index_dtype=index, **options)
    shared = encoding.FlatRelationEncoder(task, statics="shared", **common)
    repeated = encoding.FlatRelationEncoder(task, **common)
    kwargs = {"actions": rl.expand(rl.as_table(task), states)} if common.get("include_actions", True) else {}
    batch = assert_rebuilds(shared, repeated, states, **kwargs)
    d = batch.as_dict()
    # the static relations hold no tuple in a graph, every other relation is as in the repeated batch
    static = set(int(r) for r in d["static_relations"])
    counts = d.get("relation_counts")
    if counts is not None:
        assert not counts[:, sorted(static)].any() if static else True
    assert d["static_task_ids"].tolist() == [0]


def test_a_table_of_instances_keeps_one_block_each_in_instance_order():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    problems = ["probBLOCKS-8-0", "smedium", "probBLOCKS-8-0"]
    tasks = [mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / f"{problem}.pddl") for problem in problems]
    table = rl.TaskTable(tasks)
    states = [walk(t, steps=8, walks=1) for t in tasks]
    encoded = [np.asarray(tasks[i].encode(s)) for i, s in enumerate(states)]
    width = max(e.shape[1] for e in encoded)
    rows = np.concatenate([np.pad(e, ((0, 0), (0, width - e.shape[1]))) for e in encoded])
    ids = np.concatenate([np.full(len(s), i, dtype=np.int32) for i, s in enumerate(states)])
    order = np.random.default_rng(1).permutation(len(ids))
    for layout in ("graph", "relation"):
        common = dict(layout=layout, predicate_virtual_nodes=True, goal_derivations=("plain",), include_actions=False)
        shared = encoding.FlatRelationEncoder(table, statics="shared", **common)
        repeated = encoding.FlatRelationEncoder(table, **common)
        for permutation in (np.arange(len(ids)), order):
            batch = assert_rebuilds(shared, repeated, rows[permutation], ids[permutation])
            assert batch.static_task_ids.tolist() == [0, 1, 2]
        # a slice keeps the blocks its graphs need, unchanged
        whole = shared.encode(rows, ids)
        first_blocks = shared.encode(rows[:3], ids[:3])
        assert first_blocks.static_task_ids.tolist() == [0]
        part = shared.encode(rows[-4:], ids[-4:])
        assert part.static_task_ids.tolist() == [2]
        relations = len(whole.static_relations)
        for batch, block in ((first_blocks, 0), (part, 2)):
            for k in range(relations):
                lo, hi = (int(x) for x in batch.static_arg_offsets[k : k + 2])
                s = block * relations + k
                lo2, hi2 = (int(x) for x in whole.static_arg_offsets[s : s + 2])
                assert batch.static_args[lo:hi].tolist() == whole.static_args[lo2:hi2].tolist()


def test_a_single_state_and_an_empty_batch():
    task = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task, statics="shared")
    state = walk(task, steps=3, walks=1)[-1]
    single = encoder.encode(state)
    assert single.num_graphs == 1 and single.static_task_ids.tolist() == [0]
    empty = encoder.encode(np.zeros((0, task.words), np.uint64))
    assert empty.num_graphs == 0
    assert empty.static_task_ids.tolist() == [] and empty.static_offsets.tolist() == [0]
    assert len(empty.static_args) == 0


def test_the_pool_size_does_not_change_the_batch():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=1100, walks=5)
    assert len(states) > 5000
    for layout in ("graph", "relation"):
        encoder = encoding.FlatRelationEncoder(task, statics="shared", layout=layout, predicate_virtual_nodes=True)
        base = encoder.encode(states)
        for threads in (2, 3, 8, 0):
            other = encoder.encode(states, threads=threads)
            for key, value in base.as_dict().items():
                assert np.array_equal(value, other.as_dict()[key]), (layout, threads, key)


def test_the_option_and_what_it_changes_in_the_schema():
    task = text_task("blocks__probBLOCKS-8-0")
    repeat = encoding.FlatRelationEncoder(task)
    shared = encoding.FlatRelationEncoder(task, statics="shared")
    assert repeat.options["statics"] == "repeat" and shared.options["statics"] == "shared"
    assert repeat.schema_fingerprint == encoding.FlatRelationEncoder(task, statics="repeat").schema_fingerprint
    assert repeat.schema_fingerprint != shared.schema_fingerprint
    assert encoding.EncodingSchema(task, statics="shared").schema_fingerprint == shared.schema_fingerprint
    assert shared.relation_names == repeat.relation_names
    with pytest.raises(ValueError, match="^mymyr: encoding: statics must be one of 'repeat', 'shared'"):
        encoding.FlatRelationEncoder(task, statics="once")
    with pytest.raises(TypeError):
        encoding.FlatRelationEncoder(task, statics=True)
    states = walk(task, steps=3, walks=1)
    repeated = repeat.encode(states)
    assert not any(name in repeated.as_dict() for name in STATIC_ARRAYS)
    for name in STATIC_ARRAYS:
        with pytest.raises(AttributeError):
            getattr(repeated, name)
    batch = shared.encode(states)
    assert all(name in batch.as_dict() for name in STATIC_ARRAYS)
    assert batch.static_relations.tolist() == [i for i, lane in enumerate(shared.relation_lanes) if lane == "static"]
    # the static tuples of the block: ids local to the instance, ascending tuples
    arity = shared.relation_arities
    for k, relation in enumerate(batch.static_relations.tolist()):
        lo, hi = (int(x) for x in batch.static_arg_offsets[k : k + 2])
        tuples = batch.static_args[lo:hi].reshape(-1, arity[relation]).tolist()
        assert tuples == sorted(tuples)
        assert all(0 <= x <= task.num_objects for t in tuples for x in t)


def test_capacities_name_the_static_blocks():
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=10, walks=2)
    encoder = encoding.FlatRelationEncoder(task, statics="shared")
    exact = encoder.encode(states)
    probe = encoder.encode(
        states,
        capacities={"graphs": len(states), "entities": 10**6, "args": 10**6, "static_instances": 1, "static_args": 10**6},
    )
    size = probe.sizes
    assert size["static_instances"] == 1 and size["static_args"] == len(exact.static_args)
    assert size["static_tuples"] == int(exact.static_offsets[-1])
    assert not size["overflow"]
    caps = {
        "graphs": len(states) + 3,
        "entities": size["entities"] + 5,
        "args": size["args"] + 7,
        "static_instances": 3,
        "static_args": size["static_args"] + 11,
    }
    padded = encoder.encode(states, capacities=caps)
    d, e = padded.as_dict(), exact.as_dict()
    relations = len(exact.static_relations)
    assert d["static_task_ids"].tolist() == e["static_task_ids"].tolist() + [0, 0]
    assert d["static_args"][: len(e["static_args"])].tolist() == e["static_args"].tolist()
    assert not d["static_args"][len(e["static_args"]) :].any()
    assert len(d["static_offsets"]) == 3 * relations + 1
    assert (d["static_offsets"][relations:] == e["static_offsets"][-1]).all()
    assert (d["static_arg_offsets"][relations:] == e["static_arg_offsets"][-1]).all()
    assert padded.sizes["static_args"] == size["static_args"]
    # too small: nothing is encoded, the true sizes are reported
    short = encoder.encode(states, capacities={**caps, "static_args": size["static_args"] - 1})
    assert short.sizes["overflow"] and short.sizes["capacity_overflow"]
    assert short.sizes["static_args"] == size["static_args"]
    assert not any(np.asarray(v).any() for v in short.as_dict().values() if np.ndim(v) > 0)
    # the keys go with the option
    with pytest.raises(ValueError, match="static_instances"):
        encoder.encode(states, capacities={k: v for k, v in caps.items() if not k.startswith("static")})
    repeated = encoding.FlatRelationEncoder(task)
    with pytest.raises(ValueError, match="repeats its statics"):
        repeated.encode(states, capacities=caps)


def test_streams_flush_the_shared_form():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=15, walks=3)
    encoder = encoding.FlatRelationEncoder(task, statics="shared", layout="relation", predicate_virtual_nodes=True)
    expected = encoder.encode(states)
    stream = encoder.stream()
    stream.append(states[:7])
    stream.append(states[7:])
    flushed = stream.flush()
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, flushed.as_dict()[key]), key
    mutable = encoder.mutable_stream()
    mutable.append(list(range(len(states))), states)
    flushed = mutable.flush()
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, flushed.as_dict()[key]), key
    size = expected.sizes
    assert size is None
    probe = encoder.encode(
        states,
        capacities={"graphs": len(states), "entities": 10**5, "args": 10**5, "static_instances": 1, "static_args": 10**5},
    )
    caps = {
        "graphs": len(states),
        "entities": probe.sizes["entities"],
        "args": probe.sizes["args"],
        "static_instances": 1,
        "static_args": probe.sizes["static_args"],
    }
    stream.append(states)
    assert not stream.flush(capacities=caps).sizes["overflow"]
    stream.append(states)
    over = stream.flush(capacities={**caps, "static_args": 1})
    assert over.sizes["overflow"] and len(stream) == len(states)


@pytest.mark.parametrize("options", ["default", "virtual", "everything"])
def test_names_and_descriptions_are_those_of_the_repeated_batch(options):
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=6, walks=2)
    common = OPTION_SETS[options]
    shared = encoding.FlatRelationEncoder(task, statics="shared", **common)
    repeated = encoding.FlatRelationEncoder(task, **common)
    expansion = rl.expand(rl.as_table(task), states)
    a = shared.encode(states, actions=expansion)
    b = repeated.encode(states, actions=expansion)
    for g in range(len(states)):
        assert shared.entity_names(a, g) == repeated.entity_names(b, g)
        assert shared.describe(a, g) == repeated.describe(b, g)
    padded = shared.encode(
        states,
        actions=expansion,
        capacities={
            "graphs": len(states),
            "entities": 10**5,
            "args": 10**5,
            "targets": 10**5,
            "static_instances": 4,
            "static_args": 10**5,
        }
        if common.get("target_sources")
        else {"graphs": len(states), "entities": 10**5, "args": 10**5, "static_instances": 4, "static_args": 10**5},
    )
    for g in range(len(states)):
        assert shared.describe(padded, g) == repeated.describe(b, g)


def test_a_pickled_encoder_keeps_its_statics():
    task = text_task("gripper__prob05")
    states = walk(task, steps=8, walks=2)
    encoder = encoding.FlatRelationEncoder(task, statics="shared", layout="relation", index_dtype="int64")
    loaded = pickle.loads(pickle.dumps(encoder))
    assert loaded.options["statics"] == "shared"
    assert loaded.schema_fingerprint == encoder.schema_fingerprint
    a, b = encoder.encode(states).as_dict(), loaded.encode(states).as_dict()
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("framework", ["torch", "jax", "dlpack"])
def test_every_framework_exports_the_blocks(framework):
    if framework != "dlpack":
        pytest.importorskip(framework)
    task = text_task("gripper__prob05")
    states = walk(task, steps=5, walks=1)
    encoder = encoding.FlatRelationEncoder(task, statics="shared")
    batch = encoder.encode(states)
    numpy_arrays = batch.as_dict("numpy")
    exported = batch.as_dict(framework)
    assert exported["statics"] == 1
    for name in STATIC_ARRAYS:
        assert np.array_equal(np.from_dlpack(exported[name]), numpy_arrays[name]) or np.array_equal(
            np.asarray(exported[name]), numpy_arrays[name]
        ), name
