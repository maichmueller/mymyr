"""Persistence of the encoding: a schema serializes as its options, catalog, fingerprint and version and is rebuilt over a
table and checked; an encoder pickles as its table and options (test_encoding_pickle.py); a batch is its arrays and
scalars, which round-trip through NumPy files and are checked against the schema when they are read back."""

import json
import pickle

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task, walk
from test_encoding_pickle import OPTION_SETS, TASKS
from test_encoding_pool import arrays_equal

SCALARS = {"version", "num_graphs", "num_relations", "layout", "index_dtype", "schema_fingerprint", "statics"}


# ------------------------------------------------------------------------------------------------ schemas


@pytest.mark.parametrize("options", OPTION_SETS, ids=["default", "all-lanes", "plain"])
@pytest.mark.parametrize("name", TASKS)
def test_a_schema_record_rebuilds_the_schema_over_a_table(name, options):
    task = text_task(name)
    schema = encoding.EncodingSchema(task, **options)
    record = encoding.schema_to_dict(schema)
    assert set(record) == {"version", "options", "fingerprint", "catalog"}
    assert record["version"] == schema.version == 1
    assert record["fingerprint"] == f"{schema.schema_fingerprint:016x}"
    assert record["catalog"]["names"] == schema.relation_names
    assert record["catalog"]["arities"] == schema.relation_arities
    assert record["catalog"]["slot_roles"] == schema.relation_slot_roles
    # it is plain data: a JSON round trip gives the same record, which rebuilds the same schema
    assert json.loads(json.dumps(record)) == record
    rebuilt = encoding.schema_from_dict(json.loads(json.dumps(record)), task)
    assert rebuilt.schema_fingerprint == schema.schema_fingerprint and rebuilt.options == schema.options
    assert encoding.schema_to_dict(rebuilt) == record


def test_a_record_of_a_horizon_schema_keeps_the_horizon_options():
    task = text_task("gripper__prob05")
    schema = encoding.EncodingSchema(task, horizon_mode="delta", root_policy="exclude", parent_relation=True,
                                     sibling_relation=True)
    record = json.loads(json.dumps(encoding.schema_to_dict(schema)))
    assert record["options"]["horizon_mode"] == "delta" and record["options"]["root_policy"] == "exclude"
    assert encoding.schema_from_dict(record, task).schema_fingerprint == schema.schema_fingerprint


def test_a_record_of_an_encoder_and_of_a_batch_is_the_schema_record():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    batch = encoder.encode([task.initial_state])
    assert encoding.schema_to_dict(encoder.schema) == encoding.schema_to_dict(batch.schema)
    assert batch.schema.schema_fingerprint == batch.as_dict()["schema_fingerprint"] % 2**64


def test_a_record_over_another_domain_says_how_the_catalogs_differ():
    record = encoding.schema_to_dict(encoding.EncodingSchema(text_task("gripper__prob05")))
    with pytest.raises(ValueError, match=r"^mymyr: encoding: the schema record's fingerprint is .* \(the record has \d+ relations, the schema over the table \d+\)"):
        encoding.schema_from_dict(record, text_task("blocks__probBLOCKS-8-0"))


def test_a_changed_record_is_refused():
    task = text_task("gripper__prob05")
    record = encoding.schema_to_dict(encoding.EncodingSchema(task))

    def altered(path, value):
        copy = json.loads(json.dumps(record))
        target = copy
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return copy

    cases = [
        (altered(["fingerprint"], f"{int(record['fingerprint'], 16) ^ 1:016x}"), "fingerprint is"),
        (altered(["version"], 2), "version 2 of the catalog rules"),
        (altered(["catalog", "arities", 3], 7), "relation 3 .* differs in its arities"),
        (altered(["catalog", "names"], record["catalog"]["names"][:-1]), "columns of the schema record's catalog differ"),
        (altered(["options", "layout"], "relation"), "fingerprint is"),
    ]
    for bad, message in cases:
        with pytest.raises(ValueError, match="mymyr: encoding: .*" + message):
            encoding.schema_from_dict(bad, task)
    # a catalog that disagrees is refused even if the fingerprint is right
    with pytest.raises(ValueError, match="differs in its names"):
        encoding.schema_from_dict(altered(["catalog", "names", 0], "zzz"), task)


@pytest.mark.parametrize(
    "record, message",
    [
        (None, "is a dict"),
        ({}, "no int 'version'"),
        ({"version": 1, "options": {}, "fingerprint": "zz", "catalog": {}}, "not a hexadecimal"),
        ({"version": 1, "options": {}, "fingerprint": "00", "catalog": {}}, "no list 'names'"),
        ({"version": True, "options": {}, "fingerprint": "00", "catalog": {}}, "no int 'version'"),
    ],
)
def test_malformed_records_are_refused(record, message):
    with pytest.raises(ValueError, match="mymyr: encoding: .*" + message):
        encoding.schema_from_dict(record, text_task("gripper__prob05"))


def test_an_encoder_pickle_carries_the_fingerprint_of_its_schema():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    function, (name, table, options, fingerprint) = encoder.__reduce__()
    assert (name, fingerprint) == ("FlatRelationEncoder", encoder.schema.schema_fingerprint)
    assert options == encoder.schema.options and table is encoder.table
    with pytest.raises(ValueError, match="schema fingerprint"):
        function(name, table, options, fingerprint ^ 1)
    loaded = pickle.loads(pickle.dumps(encoder))
    assert encoding.schema_to_dict(loaded.schema) == encoding.schema_to_dict(encoder.schema)


# ------------------------------------------------------------------------------------------------ batches

BATCH_OPTIONS = [
    {},
    dict(target_sources=("goal", "action"), predicate_virtual_nodes=True, max_goal_level=2),
    dict(layout="relation", index_dtype="int64", emit_counts=False, equality=True, nullary=False),
    dict(history=True, target_sources=("history", "goal"), entity_arrays="auto"),
]


BATCH_IDS = ["default", "levels", "relation-int64", "history"]


def batch_inputs(encoder, task, states):
    """The inputs of `encode` that the schema's options read: the expansion, and zero history steps and subgoal
    layers when the schema has them."""
    extra = {"actions": rl.expand(rl.as_table(task), states)}
    if encoder.schema.options["history"]:
        zeros = np.zeros((len(states), 2, encoder.table.words), np.uint64)
        extra["history"] = (zeros, zeros, np.zeros((len(states), 2), np.int32))
    if encoder.schema.options["max_goal_level"]:
        layers = np.zeros((1, 1, encoder.table.words), np.uint64)
        extra["subgoals"] = (layers, layers)
    return extra


@pytest.mark.parametrize("options", BATCH_OPTIONS, ids=BATCH_IDS)
@pytest.mark.parametrize("name", TASKS)
def test_a_batch_round_trips_through_its_arrays_and_a_numpy_file(name, options, tmp_path):
    task = text_task(name)
    encoder = encoding.FlatRelationEncoder(task, **options)
    states = walk(task, steps=6, walks=2)
    batch = encoder.encode(states, **batch_inputs(encoder, task, states))

    # the arrays as they are
    again = encoding.FlatBatch.from_arrays(batch.schema, batch.as_dict())
    arrays_equal(again, batch)
    assert again.num_graphs == batch.num_graphs and again.implicit_entities == batch.implicit_entities
    assert again.sizes is None
    # a copy that owns its memory
    for name_ in batch.as_dict():
        if isinstance(batch.as_dict()[name_], np.ndarray) and batch.as_dict()[name_].size:
            assert again.as_dict()[name_].__array_interface__["data"][0] != batch.as_dict()[name_].__array_interface__["data"][0]

    # through a file: the scalars come back as 0-d arrays
    path = tmp_path / "batch.npz"
    np.savez(path, **batch.as_dict())
    with np.load(path) as saved:
        loaded = encoding.FlatBatch.from_arrays(encoder.schema, saved)
    arrays_equal(loaded, batch)
    # and the encoder describes the loaded batch like the original
    assert encoder.entity_names(loaded, 0) == encoder.entity_names(batch, 0)
    assert str(encoder.describe(loaded, 1)) == str(encoder.describe(batch, 1))


def test_a_batch_with_capacities_round_trips_with_its_sizes(tmp_path):
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    states = walk(task, steps=4, walks=1)
    expansion = rl.expand(rl.as_table(task), states)
    capacities = dict(graphs=len(states), entities=400, args=4000, targets=200)
    batch = encoder.encode(states, actions=expansion, capacities=capacities)
    again = encoding.FlatBatch.from_arrays(encoder.schema, batch.as_dict(), sizes=batch.sizes)
    arrays_equal(again, batch)
    assert again.sizes == batch.sizes
    # sizes that disagree with the arrays are refused
    with pytest.raises(ValueError, match="sizes do not match"):
        encoding.FlatBatch.from_arrays(encoder.schema, batch.as_dict(), sizes=dict(batch.sizes, entities=1))
    # a batch that did not fit is all zero, and says so
    tight = encoder.encode(states, actions=expansion, capacities=dict(capacities, entities=5))
    assert tight.sizes["overflow"]
    arrays_equal(encoding.FlatBatch.from_arrays(encoder.schema, tight.as_dict(), sizes=tight.sizes), tight)
    broken = dict(tight.as_dict(), task_ids=np.ones_like(tight.task_ids))
    with pytest.raises(ValueError, match="overflowed, but 'task_ids' is not zero"):
        encoding.FlatBatch.from_arrays(encoder.schema, broken, sizes=tight.sizes)
    # without the sizes the batch is read as it is: the zeros past its contents are allowed
    assert encoding.FlatBatch.from_arrays(encoder.schema, batch.as_dict()).sizes is None


@pytest.mark.parametrize("options", BATCH_OPTIONS, ids=BATCH_IDS)
def test_an_overflowed_batch_is_checked_like_any_other_at_load(options):
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, **options)
    states = walk(task, steps=4, walks=1)
    capacities = dict(graphs=len(states), entities=3, args=3, targets=2, history=2)
    tight = encoder.encode(states, capacities=capacities, **batch_inputs(encoder, task, states))
    assert tight.sizes["capacity_overflow"]
    arrays = dict(tight.as_dict())

    def load(schema=encoder.schema, drop=(), **changes):
        record = {k: v for k, v in dict(arrays, **changes).items() if k not in drop}
        return encoding.FlatBatch.from_arrays(schema, record, sizes=tight.sizes)

    arrays_equal(load(), tight)
    with pytest.raises(ValueError, match="schema_fingerprint does not match"):
        load(schema_fingerprint=arrays["schema_fingerprint"] ^ 1)
    with pytest.raises(ValueError, match="unknown batch version 999"):
        load(version=999)
    with pytest.raises(ValueError, match="num_relations does not match the schema"):
        load(encoding.FlatRelationEncoder(task, **dict(options, include_static=False)).schema)
    with pytest.raises(ValueError, match="the array entity_offsets is missing"):
        load(drop=("entity_offsets",))
    with pytest.raises(ValueError, match="the array task_ids has the wrong shape"):
        load(task_ids=np.zeros(len(states) + 1, np.int32))
    wrong_width = np.int32 if np.asarray(arrays["relation_args"]).dtype == np.int64 else np.int64
    with pytest.raises(ValueError, match="the array relation_args has the wrong dtype"):
        load(relation_args=np.zeros(len(arrays["relation_args"]), wrong_width))
    # identity comes first: a record of another schema is refused as that, whatever it holds
    with pytest.raises(ValueError, match="schema_fingerprint does not match"):
        load(schema_fingerprint=arrays["schema_fingerprint"] ^ 1, task_ids=np.ones(len(states), np.int32))
    # a record that passes the checks but is not all zero contradicts its sizes
    with pytest.raises(ValueError, match="overflowed, but 'task_ids' is not zero"):
        load(task_ids=np.ones(len(states), np.int32))


def test_a_batch_of_another_schema_or_version_is_refused_at_load():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    batch = encoder.encode(walk(task, steps=3, walks=1))
    arrays = dict(batch.as_dict())
    other = encoding.FlatRelationEncoder(task, include_static=False).schema
    with pytest.raises(ValueError, match="num_relations does not match the schema"):
        encoding.FlatBatch.from_arrays(other, arrays)
    relation = encoding.FlatRelationEncoder(task, layout="relation").schema
    with pytest.raises(ValueError, match="layout"):
        encoding.FlatBatch.from_arrays(relation, arrays)
    with pytest.raises(ValueError, match="schema_fingerprint does not match"):
        encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, schema_fingerprint=arrays["schema_fingerprint"] ^ 1))
    with pytest.raises(ValueError, match="unknown batch version 2"):
        encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, version=2))


def test_corrupt_arrays_are_refused_at_load():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    batch = encoder.encode(walk(task, steps=3, walks=1))
    arrays = dict(batch.as_dict())
    load = lambda **changes: encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, **changes))  # noqa: E731
    load()
    args = np.array(arrays["relation_args"])
    args[0] = int(arrays["entity_offsets"][-1]) + 5
    with pytest.raises(ValueError, match="mymyr: encoding"):
        load(relation_args=args)
    offsets = np.array(arrays["relation_offsets"])
    offsets[1], offsets[2] = offsets[2], offsets[1] + 1
    with pytest.raises(ValueError, match="mymyr: encoding"):
        load(relation_offsets=offsets)
    with pytest.raises(ValueError, match="mymyr: encoding"):
        load(relation_args=np.array(arrays["relation_args"])[:-1])
    with pytest.raises(ValueError, match="mymyr: encoding"):
        load(entity_role=np.array(arrays["entity_role"])[:-2])
    wrong_type = np.array(arrays["relation_args"]).astype(np.int64)
    with pytest.raises(ValueError, match="mymyr: encoding"):
        load(relation_args=wrong_type)


def test_the_arrays_of_a_batch_are_checked_before_they_are_read():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    arrays = dict(encoder.encode(walk(task, steps=3, walks=1)).as_dict())
    with pytest.raises(ValueError, match="unknown batch array 'bogus'"):
        encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, bogus=np.zeros(1, np.int32)))
    missing = {k: v for k, v in arrays.items() if k != "layout"}
    with pytest.raises(ValueError, match="no 'layout'"):
        encoding.FlatBatch.from_arrays(encoder.schema, missing)
    with pytest.raises(ValueError, match="no arrays"):
        encoding.FlatBatch.from_arrays(encoder.schema, {k: v for k, v in arrays.items() if k in SCALARS})
    with pytest.raises(TypeError, match="mapping"):
        encoding.FlatBatch.from_arrays(encoder.schema, [1, 2])
    with pytest.raises(TypeError, match="int32, int64 or uint8"):
        encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, task_ids=np.zeros(2, np.float32)))
    with pytest.raises(TypeError, match="must be an integer"):
        encoding.FlatBatch.from_arrays(encoder.schema, dict(arrays, layout="graph"))
    with pytest.raises(TypeError):
        encoding.FlatBatch.from_arrays("not a schema", arrays)


def test_a_loaded_batch_exports_in_the_requested_framework():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    batch = encoder.encode(walk(task, steps=3, walks=1))
    loaded = encoding.FlatBatch.from_arrays(encoder.schema, batch.as_dict(), framework="dlpack")
    exported = loaded.as_dict()
    assert np.array_equal(np.from_dlpack(exported["relation_args"]), batch.relation_args)
    assert type(loaded.relation_args).__name__ != "ndarray"


@pytest.mark.parametrize("mode", ["full", "delta", "action"])
def test_horizon_batches_round_trip_too(mode):
    task = text_task("blocks__probBLOCKS-8-0")
    rows = np.ascontiguousarray(np.asarray(task.encode(walk(task, steps=2, walks=1))))
    encoder = encoding.FlatRootedHorizonEncoder(task, horizon_mode=mode, sibling_relation=True)
    batch = encoder.encode(encoding.horizon(task, rows, depth=2, max_nodes=40, truncate=True))
    arrays_equal(encoding.FlatBatch.from_arrays(encoder.schema, batch.as_dict()), batch)
    record = json.loads(json.dumps(encoding.schema_to_dict(encoder.schema)))
    arrays_equal(encoding.FlatBatch.from_arrays(encoding.schema_from_dict(record, task), batch.as_dict()), batch)
