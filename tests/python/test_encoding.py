"""The Python API of the relational encoder: FlatRelationEncoder and FlatBatch. The graphs are checked against an
independent reconstruction from the state's atoms and against the C++ encoder's bytes, and the input and output interop
(State lists, u32 rows, zero-copy exports) and the error mapping are tested. Parity with mifrost is in
test_encoding_parity.py."""

import gc
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, SMALL_TASKS, text_task, walk

MODES = ["auto", "reachable", "dense"]
AUXILIARY = 8


def encoder_of(task, **options):
    return encoding.FlatRelationEncoder(task, **{"goal_derivations": (), **options})


def lane_relations(encoder, lanes=("fluent", "static", "derived")):
    return [i for i, lane in enumerate(encoder.relation_lanes) if lane in lanes]


def unpack(batch, graph):
    """{relation name: tuples of instance-local entity ids} of one graph, for the relations that have tuples."""
    output = {}
    relations = batch.num_relations
    names = batch.schema.relation_names
    arities = batch.schema.relation_arities
    base = int(batch.entity_offsets[graph])
    for relation, name in enumerate(names):
        count = int(batch.relation_counts[graph, relation])
        if not count:
            continue
        cell = graph * relations + relation
        begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        assert end - begin == count * arities[relation]
        assert batch.relation_offsets[cell + 1] - batch.relation_offsets[cell] == count
        arguments = batch.relation_args[begin:end] - base
        output[name] = [tuple(int(x) for x in row) for row in arguments.reshape(count, arities[relation])]
    return output


def reference(task, state, include_static=True, include_derived=True, nullary=True):
    """The tuples of a state by relation name, from its atoms; a nullary atom is the tuple over the auxiliary entity."""
    atoms = list(state.atoms())
    if include_derived:
        atoms += list(state.derived_atoms())
    if include_static:
        atoms += list(task.formalism.static_init)
    auxiliary = len(task.objects)
    output = {}
    for atom in atoms:
        name = atom.predicate.name
        if name == "=":
            continue
        arguments = tuple(obj.index for obj in atom.objects)
        if not arguments:
            if not nullary:
                continue
            arguments = (auxiliary,)
        output.setdefault(name, set()).add(arguments)
    return {name: sorted(values) for name, values in output.items()}


DOMAIN = """(define (domain features)
 (:requirements :strips :negative-preconditions :derived-predicates)
 (:predicates (fixed ?x) (flag ?x) (available ?x) (ready))
 (:derived (available ?x) (flag ?x))
 (:action enable :parameters (?x) :precondition (not (flag ?x)) :effect (flag ?x))
 (:action disable :parameters (?x) :precondition (flag ?x) :effect (not (flag ?x)))
 (:action switch :parameters () :precondition (and) :effect (ready)))"""


def feature_task(objects="a b", atoms="auto"):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string(DOMAIN)
    problem = f"""(define (problem p) (:domain features) (:objects {objects})
      (:init (fixed a) (flag a))
      (:goal (and (flag b) (not (flag a)) (fixed a) (not (fixed b))
                  (available b) (not (available a)) (ready))))"""
    return mymyr.Task(domain.instantiate_string(problem), atoms=atoms)


def feature_states(task):
    state = task.apply(task.initial_state, "(enable b)")
    state = task.apply(state, "(disable a)")
    return [task.initial_state, state, task.apply(state, "(switch)")]


def derived_words(task, states):
    """The derived atom words of each state: bit i is set iff the i-th derived slot's atom holds."""
    metadata = rl.as_table(task).atom_metadata()
    predicates = task.formalism.predicates
    slots = [
        (predicates[int(p)].name, [int(a) for a in args if a >= 0])
        for p, args in zip(metadata["derived_pred"], metadata["derived_args"])
    ]
    words = np.zeros((len(states), max(1, (len(slots) + 63) // 64)), np.uint64)
    for row, state in enumerate(states):
        for slot, (name, args) in enumerate(slots):
            if state.holds(task.atom((name, args))):
                words[row, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return words


# ------------------------------------------------------------------------------------------------ semantics
@pytest.mark.parametrize("atoms", MODES)
@pytest.mark.parametrize(
    "options", [{}, {"include_static": False}, {"include_derived": False}, {"nullary": False, "include_static": False}]
)
def test_graphs_equal_the_reconstruction_from_the_atoms(atoms, options):
    task = feature_task(atoms=atoms)
    states = feature_states(task)
    encoder = encoder_of(task, **options)
    batch = encoder.encode(states)
    assert batch.num_graphs == 3 and batch.num_relations == encoder.num_relations
    for graph, state in enumerate(states):
        assert unpack(batch, graph) == reference(
            task, state, options.get("include_static", True), options.get("include_derived", True),
            options.get("nullary", True))
    ready = encoder.relation_id("ready") if options.get("nullary", True) else None
    if ready is not None:
        assert encoder.relation_arities[ready] == 1
        assert batch.relation_counts[:, ready].tolist() == [0, 0, 1]


@pytest.mark.parametrize("atoms", MODES)
def test_entities_objects_first_then_the_auxiliary_entity_when_a_nullary_tuple_exists(atoms):
    task = feature_task(atoms=atoms)
    states = feature_states(task)
    batch = encoder_of(task).encode(states)
    objects = len(task.objects)
    assert batch.num_objects.tolist() == [objects] * 3
    # the initial state has a static nullary atom or none: the graphs with a nullary tuple have the auxiliary entity
    for graph in range(3):
        begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
        roles = batch.entity_role[begin:end].tolist()
        refs = batch.entity_ref[begin:end].tolist()
        has_nullary = any(
            tuple(t) == (objects,) for tuples in unpack(batch, graph).values() for t in tuples
        )
        assert roles == [0] * objects + ([AUXILIARY] if has_nullary else [])
        assert refs == list(range(objects)) + ([-1] if has_nullary else [])
    assert batch.schema.entity_role_names[AUXILIARY] == "auxiliary"


@pytest.mark.parametrize("atoms", MODES)
@pytest.mark.parametrize("name", SMALL_TASKS)
def test_walks_equal_the_reconstruction_from_the_atoms(name, atoms):
    task = text_task(name, atoms=atoms)
    states = walk(task, steps=3, walks=1)
    encoder = encoder_of(task)
    batch = encoder.encode(states)
    for graph, state in enumerate(states):
        assert unpack(batch, graph) == reference(task, state)


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_slot_layout_does_not_change_the_encoding(name):
    batches = []
    for atoms in MODES:
        task = text_task(name, atoms=atoms)
        batches.append(encoder_of(task).encode(walk(task, steps=6, seed=3)))
    first = batches[0].as_dict()
    for other in batches[1:]:
        for key, value in other.as_dict().items():
            assert np.array_equal(value, first[key]), key


@pytest.mark.parametrize("atoms", MODES)
def test_a_table_of_several_instances_numbers_entities_across_the_batch(atoms):
    first, second = feature_task(atoms=atoms), feature_task("a b c d", atoms=atoms)
    table = rl.TaskTable([first, second])
    states = [second.initial_state, first.initial_state, second.state(["(flag b)"])]
    ids = np.array([1, 0, 1], dtype=np.int32)
    encoder = encoder_of(table)
    batch = encoder.encode(states, task_ids=ids)
    assert batch.task_ids.tolist() == [1, 0, 1]
    assert batch.num_objects.tolist() == [4, 2, 4]
    for graph, (task_id, state) in enumerate(zip(ids, states)):
        assert unpack(batch, graph) == reference([first, second][task_id], state)
    # the graph of a row does not depend on the other rows
    for graph in range(3):
        single = encoder.encode([states[graph]], task_ids=ids[graph : graph + 1])
        assert unpack(single, 0) == unpack(batch, graph)


def test_empty_batches_and_states():
    task = text_task("gripper__prob05")
    encoder = encoder_of(task)
    empty = encoder.encode([])
    assert empty.num_graphs == 0 and empty.relation_counts.shape == (0, encoder.num_relations)
    assert empty.entity_offsets.tolist() == [0] and empty.relation_offsets.tolist() == [0]
    assert empty.relation_args.shape == (0,)
    none = encoder.encode(np.zeros((0, task.words), np.uint64))
    assert none.num_graphs == 0
    state = task.state([])
    one = encoder.encode(state)
    assert unpack(one, 0) == reference(task, state)


# ------------------------------------------------------------------------------------------------ interop
def test_state_lists_equal_rows():
    task = text_task("philosophers__p03-phil4")
    encoder = encoder_of(task)
    states = walk(task, steps=4, walks=2)
    rows = task.encode(states)
    from_states = encoder.encode(states)
    from_rows = encoder.encode(rows)
    assert from_states.schema.schema_fingerprint == from_rows.schema.schema_fingerprint
    for key, value in from_states.as_dict().items():
        assert np.array_equal(value, from_rows.as_dict()[key]), key
    single = encoder.encode(states[0])
    assert unpack(single, 0) == unpack(from_rows, 0)
    assert np.array_equal(encoder.encode(tuple(states)).relation_args, from_rows.relation_args)


@pytest.mark.parametrize("index", ["int32", "int64"])
def test_u32_rows_equal_u64_rows(index):
    task = text_task("philosophers__p03-phil4")
    encoder = encoder_of(task, index_dtype=index)
    rows = np.ascontiguousarray(task.encode(walk(task, steps=4, walks=2)))
    halves = rows.view(np.uint32)
    assert halves.shape == (len(rows), 2 * rows.shape[1])
    wide, narrow = encoder.encode(rows), encoder.encode(halves)
    for key, value in wide.as_dict().items():
        assert np.array_equal(value, narrow.as_dict()[key]), key
    assert wide.relation_args.dtype == (np.int32 if index == "int32" else np.int64)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_rows_of_other_frameworks_and_their_outputs(framework):
    module = pytest.importorskip(framework)
    task = text_task("gripper__prob05")
    encoder = encoder_of(task)
    rows = np.ascontiguousarray(task.encode(walk(task, steps=4, walks=1)))
    expected = encoder.encode(rows)
    if framework == "torch":
        source = module.from_numpy(rows.view(np.int64).copy())
    else:
        source = module.device_put(rows.view(np.uint32), module.devices("cpu")[0])
    batch = encoder.encode(source)
    exported = batch.as_dict()
    for key, value in expected.as_dict().items():
        assert np.array_equal(np.asarray(exported[key]), value), key
    assert type(batch.relation_args).__module__.split(".")[0] in {"torch", "jaxlib", "jax"}
    forced = encoder.encode(rows, framework=framework)
    assert type(forced.relation_args).__module__.split(".")[0] in {"torch", "jaxlib", "jax"}


def pointer(array):
    if hasattr(array, "data_ptr"):
        return array.data_ptr()
    if hasattr(array, "unsafe_buffer_pointer"):
        return array.unsafe_buffer_pointer()
    return array.__array_interface__["data"][0]


ARRAYS = ["task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref", "relation_counts",
          "relation_offsets", "relation_arg_offsets", "relation_args"]


def test_numpy_exports_share_the_memory_of_the_batch():
    task = text_task("gripper__prob05")
    batch = encoder_of(task).encode(walk(task, steps=3, walks=1))
    exported = batch.as_dict("numpy")
    for name in ARRAYS:
        array = getattr(batch, name)
        assert pointer(array) == pointer(exported[name]) == pointer(batch.as_dict()[name]), name
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        batch.relation_args[0] = 1
    assert exported["num_graphs"] == batch.num_graphs and exported["version"] == 1
    assert exported["schema_fingerprint"] % 2**64 == batch.schema.schema_fingerprint


@pytest.mark.parametrize("framework", ["torch", "jax", "dlpack"])
def test_framework_exports_share_the_memory_of_the_batch(framework):
    if framework != "dlpack":
        pytest.importorskip(framework)
    task = text_task("gripper__prob05")
    batch = encoder_of(task).encode(walk(task, steps=3, walks=1))
    exported = batch.as_dict(framework)
    for name in ARRAYS:
        value = exported[name]
        if framework == "dlpack":
            assert pointer(np.from_dlpack(value)) == pointer(getattr(batch, name)), name
        else:
            assert pointer(value) == pointer(getattr(batch, name)), name
            assert np.array_equal(np.asarray(value), getattr(batch, name)), name


def test_results_are_owned_and_independent_of_the_inputs():
    task = text_task("gripper__prob05")
    encoder = encoder_of(task)
    states = task.encode(walk(task, steps=3, walks=1))
    batch = encoder.encode(states)
    retained, expected = batch.relation_args, batch.relation_args.copy()
    schema = batch.schema
    states[:] = 0
    encoder.encode(states)
    del encoder, batch, states, task
    gc.collect()
    assert np.array_equal(retained, expected) and schema.num_relations > 0


def test_encoding_is_reentrant():
    task = text_task("philosophers__p03-phil4")
    encoder = encoder_of(task)
    states = walk(task, steps=5, walks=1)
    expected = encoder.encode(states).relation_args.copy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: encoder.encode(states).relation_args.copy(), range(64)))
    assert all(np.array_equal(value, expected) for value in results)


# ------------------------------------------------------------------------------------------------ derived words
def test_given_derived_words_equal_the_evaluated_ones():
    task = text_task("philosophers__p03-phil4")
    encoder = encoder_of(task)
    states = walk(task, steps=5, walks=2)
    rows = task.encode(states)
    expected = encoder.encode(rows)
    words = derived_words(task, states)
    given = encoder.encode(rows, derived=words)
    for key, value in expected.as_dict().items():
        assert np.array_equal(value, given.as_dict()[key]), key
    padded = np.concatenate([words, np.ones((len(states), 3), np.uint64)], axis=1)
    assert np.array_equal(encoder.encode(rows, derived=padded).relation_args, expected.relation_args)
    assert np.array_equal(encoder.encode(rows, derived=words.view(np.uint32)).relation_args, expected.relation_args)
    # the words are used: without derived atoms there are no derived tuples
    blank = encoder.encode(rows, derived=np.zeros_like(words))
    derived = [name for name, lane in zip(encoder.relation_names, encoder.relation_lanes) if lane == "derived"]
    assert any(unpack(expected, g).get(name) for g in range(len(states)) for name in derived)
    assert not any(unpack(blank, g).get(name) for g in range(len(states)) for name in derived)


# ------------------------------------------------------------------------------------------------ schema and API
def test_the_encoder_describes_its_schema():
    task = text_task("philosophers__p03-phil4")
    options = {"goal_derivations": (), "nullary": False, "include_static": False}
    encoder = encoding.FlatRelationEncoder(task, **options)
    schema = encoding.EncodingSchema(task, **options)
    for name in ("options", "num_relations", "relation_names", "relation_arities", "relation_logical_arities",
                 "relation_lanes", "relation_signs", "relation_levels", "relation_derivations", "relation_predicates",
                 "relation_schemas", "relation_slot_roles", "schema_relations", "slot_role_names", "entity_role_names",
                 "target_group_names", "schema_fingerprint"):
        assert getattr(encoder, name) == getattr(schema, name), name
    assert encoder.schema.schema_fingerprint == schema.schema_fingerprint
    assert encoder.relation_id(encoder.relation_names[3]) == 3
    with pytest.raises(KeyError):
        encoder.relation_id("no-such-relation")
    batch = encoder.encode(task.initial_state)
    assert batch.schema.relation_names == schema.relation_names
    assert encoder.table is not None
    assert encoding.FlatRelationEncoder(task, **options).schema_fingerprint == schema.schema_fingerprint


def test_the_default_goal_derivations_are_refused():
    task = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="^mymyr: encoding: goal_derivations is not supported"):
        encoding.FlatRelationEncoder(task)
    encoding.EncodingSchema(task)  # the catalog itself lists them


# ------------------------------------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    "options, match",
    [
        ({"goal_derivations": ("plain",)}, "goal_derivations is not supported"),
        ({"max_goal_level": 1, "target_sources": ("goal", "subgoal")}, "max_goal_level is not supported"),
        ({"target_sources": ("goal",)}, "targets is not supported"),
        ({"predicate_virtual_nodes": True}, "predicate_virtual_nodes is not supported"),
        ({"horizon_mode": "full"}, "horizon is not supported"),
        ({"layout": "relation"}, "layout='relation' is not supported"),
    ],
)
def test_options_the_encoder_does_not_encode_raise_value_error(options, match):
    with pytest.raises(ValueError, match=f"^mymyr: encoding: .*{match}"):
        encoding.FlatRelationEncoder(text_task("gripper__prob05"), **{"goal_derivations": (), **options})


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"max_goal_level": 4}, ValueError, "^mymyr: encoding: max_goal_level must be in"),
        ({"goal_derivations": ["plane"]}, ValueError, "goal_derivations"),
        ({"goal_derivations": "plain"}, TypeError, "collection"),
        ({"index_dtype": "int16"}, ValueError, "index_dtype"),
        ({"no_such_option": 1}, TypeError, "no_such_option"),
    ],
)
def test_invalid_options_raise_like_the_schema(options, error, match):
    task = text_task("gripper__prob05")
    with pytest.raises(error, match=match):
        encoder_of(task, **options)
    with pytest.raises(error, match=match):
        encoding.EncodingSchema(task, **{"goal_derivations": (), **options})


def test_numeric_tasks_and_bad_tables_are_refused():
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    with pytest.raises(ValueError, match="^mymyr: encoding: numeric"):
        encoder_of(numeric)
    with pytest.raises(TypeError):
        encoder_of("gripper")
    task = text_task("gripper__prob05")
    with pytest.raises(TypeError, match="TaskTable"):
        encoder_of(rl.TaskSuite([rl.TaskTable([task])]))


def test_lazy_atom_slots_are_refused():
    task = text_task("gripper__prob05", atoms="lazy")
    with pytest.raises(ValueError, match="lazy atom slots"):
        encoder_of(task)
    with pytest.raises(ValueError, match="lazy atom slots"):
        encoder_of(rl.TaskTable([task]))


@pytest.mark.parametrize("atoms", MODES)
def test_invalid_inputs_raise(atoms):
    first, second = feature_task(atoms=atoms), feature_task("a b c", atoms=atoms)
    encoder = encoder_of(first)
    with pytest.raises(ValueError, match="another task"):
        encoder.encode(second.initial_state)
    bad = first.encode([first.initial_state])
    assert first.num_atoms < 63
    bad[0, -1] |= np.uint64(1 << 63)
    with pytest.raises(ValueError, match="^mymyr: encoding: .*atom bit"):
        encoder.encode(bad)
    narrow = np.zeros((1, 0), np.uint64)
    with pytest.raises(ValueError):
        encoder.encode(narrow)
    padded = np.concatenate([first.encode([first.initial_state]), np.ones((1, 1), np.uint64)], 1)
    assert unpack(encoder.encode(padded), 0) == unpack(encoder.encode(first.initial_state), 0)  # padding is not read
    with pytest.raises(TypeError):
        encoder.encode("states")
    with pytest.raises(ValueError):
        encoder.encode(first.initial_state, framework="tensorflow")
    table_encoder = encoder_of(rl.TaskTable([first, second]))
    with pytest.raises(ValueError, match="^mymyr: encoding: task_ids are required"):
        table_encoder.encode([first.initial_state])
    with pytest.raises(ValueError, match="task id"):
        table_encoder.encode([first.initial_state], task_ids=[2])
    with pytest.raises(ValueError, match="another task"):
        table_encoder.encode([first.initial_state], task_ids=[1])
    rows = first.encode([first.initial_state] * 2)
    with pytest.raises(ValueError):
        encoder.encode(rows, derived=np.zeros((3, 1), np.uint64))
    with pytest.raises(TypeError):
        encoder.encode(rows, derived=np.zeros((2, 1), np.float32))


def test_a_derived_array_narrower_than_the_instance_needs_is_refused():
    task = text_task("philosophers__p03-phil4")
    encoder = encoder_of(task)
    rows = task.encode([task.initial_state])
    needed = derived_words(task, [task.initial_state]).shape[1]
    assert needed >= 1
    with pytest.raises(ValueError, match="^mymyr: encoding: the derived row stride"):
        encoder.encode(rows, derived=np.zeros((1, needed - 1), np.uint64))


# ------------------------------------------------------------------------------------------------ C++ bytes
# FNV-1a 64 over the scalars (name, value as 8 bytes), then each array's name, extents (8 bytes each) and bytes: the
# digest of the encoder's rl::ArrayBundle. tests/cpp/encoding/test_encoder.cpp asserts the same constants for the same
# encodings, so the arrays that Python exports are the bytes that the C++ encoder wrote.
GOLDEN_DIGESTS = {
    "gripper__prob05": 14883951552709315966,
    "philosophers__p03-phil4": 12253892925669182561,
}


def digest(batch):
    h = 0xCBF29CE484222325

    def feed(data):
        nonlocal h
        for byte in data:
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    for name, value in batch.as_dict("numpy").items():
        feed(name.encode())
        if isinstance(value, int):
            feed(int(value).to_bytes(8, "little", signed=True))
        else:
            for extent in value.shape:
                feed(int(extent).to_bytes(8, "little"))
            feed(np.ascontiguousarray(value).tobytes())
    return h


def first_successor_chain(task, length):
    rows = task.encode([task.initial_state])
    for _ in range(length - 1):
        expansion = rl.expand(task, rows[-1:])
        assert expansion.total > 0
        rows = np.concatenate([rows, np.asarray(expansion.succ)[:1]])
    return rows


@pytest.mark.parametrize("name", sorted(GOLDEN_DIGESTS))
def test_python_arrays_are_the_bytes_of_the_cpp_encoder(name):
    task = text_task(name)
    batch = encoder_of(task).encode(first_successor_chain(task, 6))
    assert digest(batch) == GOLDEN_DIGESTS[name]


# ------------------------------------------------------------------------------------------------ stubs
def test_generated_native_encoding_stubs_are_typed():
    directory = Path(importlib.util.find_spec("mymyr._core").origin).parent / "_core"
    path = directory / "_encoding.pyi"
    if not path.is_file():
        pytest.skip("no encoding stubs installed")
    source = path.read_text()
    assert "class FlatRelationEncoder" in source and "class FlatBatch" in source
    assert "states: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack" in source
    assert "table: mymyr._core._rl.TaskTable | mymyr._core.Task | mymyr._core.TaskHandle" in source
    assert "derived: mymyr._typing.SupportsDLPack" in source
    assert "def relation_counts(self) -> Any" in source
    assert "def as_dict(self, framework:" in source and "-> dict[str, Any]" in source
    assert ": object" not in source and "-> object" not in source
