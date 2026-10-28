"""Pickling of the relational encoder: it pickles as its table, its options and the fingerprint of its schema, and
loading rebuilds an encoder whose output equals the original's bit for bit (also in a spawned process) and checks the
fingerprint."""

import hashlib
import os
import pickle

import numpy as np
import pytest

import mymyr
from mymyr import _pickle, encoding, rl
from conftest import ROOT, text_task, walk
from python_children import run_code

OPTION_SETS = [
    {},
    dict(layout="relation", index_dtype="int64", goal_derivations=("plain", "satisfied", "unsatisfied"),
         target_sources=("goal", "action"), predicate_virtual_nodes=True, emit_counts=False, equality=True,
         nullary="omit", include_static=False),
    dict(entity_arrays="auto", goal_derivations=(), include_actions=False, include_derived=False),
    dict(statics="shared", predicate_virtual_nodes=True, target_sources=("goal",)),
]
TASKS = ["gripper__prob05", "blocks__probBLOCKS-8-0", "philosophers__p03-phil4"]
DIGEST = """
import hashlib, pickle, sys
import numpy as np
from mymyr import rl
encoder, states = pickle.load(open(sys.argv[1], "rb"))
batch = encoder.encode(states, actions=rl.expand(rl.as_table(encoder.table), states))
digest = hashlib.sha256()
for key, value in sorted(batch.as_dict().items()):
    digest.update(key.encode())
    digest.update(np.ascontiguousarray(value).tobytes() if isinstance(value, np.ndarray) else repr(value).encode())
print(digest.hexdigest())
"""


def digest(batch):
    out = hashlib.sha256()
    for key, value in sorted(batch.as_dict().items()):
        out.update(key.encode())
        out.update(np.ascontiguousarray(value).tobytes() if isinstance(value, np.ndarray) else repr(value).encode())
    return out.hexdigest()


def same(a, b):
    first, second = a.as_dict(), b.as_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(first[key], second[key]), key


@pytest.mark.parametrize("options", OPTION_SETS, ids=["default", "all-lanes", "plain", "shared"])
@pytest.mark.parametrize("name", TASKS)
def test_a_pickle_round_trip_gives_an_encoder_with_equal_output(name, options):
    task = text_task(name)
    encoder = encoding.FlatRelationEncoder(task, **options)
    loaded = pickle.loads(pickle.dumps(encoder))
    assert loaded is not encoder
    assert loaded.options == encoder.options
    assert loaded.schema_fingerprint == encoder.schema_fingerprint
    assert loaded.relation_names == encoder.relation_names
    states = walk(task, steps=6)
    table = rl.as_table(task)
    actions = rl.expand(table, states) if encoder.options["include_actions"] else None
    same(encoder.encode(states, actions=actions), loaded.encode(states, actions=actions))
    # the copy streams like the original
    stream = loaded.stream()
    stream.append(states, actions=actions)
    same(stream.flush(), encoder.encode(states, actions=actions))


def test_a_pickled_table_encoder_keeps_its_instances():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    blocks = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / f"{p}.pddl") for p in ("smedium", "probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table, target_sources=("goal",))
    loaded = pickle.loads(pickle.dumps(encoder))
    states = [tasks[1].initial_state, tasks[0].initial_state, tasks[1].initial_state]
    ids = np.array([1, 0, 1], dtype=np.int32)
    same(encoder.encode(states, ids), loaded.encode([loaded.table[i].initial_state for i in ids], ids))
    assert len(loaded.table) == 2


def test_the_pickle_stores_the_table_the_options_and_the_fingerprint():
    task = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task, layout="relation", target_sources=("goal",))
    restore, (name, table, options, fingerprint) = encoder.__reduce__()
    assert restore is _pickle._restore_flat_encoder
    assert name == "FlatRelationEncoder"
    assert table is task
    assert options == encoder.options and options["layout"] == "relation"
    assert fingerprint == encoder.schema_fingerprint


def test_a_pickled_encoder_encodes_alike_in_a_spawned_process(tmp_path):
    task = text_task("logistics00__probLOGISTICS-6-1")
    encoder = encoding.FlatRelationEncoder(task, layout="relation", target_sources=("goal", "action"),
                                           predicate_virtual_nodes=True, goal_derivations=("plain", "satisfied"))
    states = walk(task, steps=8)
    actions = rl.expand(rl.as_table(task), states)
    path = tmp_path / "encoder.pickle"
    path.write_bytes(pickle.dumps((encoder, states)))
    expected = digest(encoder.encode(states, actions=actions))
    for seed in ("0", "1234"):
        child = run_code(DIGEST, str(path), env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True)
        assert child.stdout.strip() == expected


class Tampered:
    """Pickles as a call of the encoder's reconstructor with the arguments `args`."""

    def __init__(self, args):
        self.args = args

    def __reduce__(self):
        return _pickle._restore_flat_encoder, self.args


@pytest.mark.parametrize("name", TASKS)
def test_a_fingerprint_mismatch_at_load_raises(name):
    task = text_task(name)
    encoder = encoding.FlatRelationEncoder(task)
    _, (name, table, options, fingerprint) = encoder.__reduce__()
    assert _pickle._restore_flat_encoder(name, table, options, fingerprint).schema_fingerprint == fingerprint
    for wrong in (fingerprint ^ 1, fingerprint ^ (1 << 63), 0):
        with pytest.raises(ValueError, match="mymyr: encoding: the pickled encoder's schema fingerprint"):
            pickle.loads(pickle.dumps(Tampered((name, table, options, wrong))))
    # options that shape the output but not the catalog (the layout) change the fingerprint too
    other = encoding.FlatRelationEncoder(task, layout="relation")
    assert other.schema_fingerprint != fingerprint
    with pytest.raises(ValueError, match="mymyr: encoding: the pickled encoder's schema fingerprint"):
        pickle.loads(pickle.dumps(Tampered((name, table, other.options, fingerprint))))


def test_a_pickle_with_invalid_options_raises():
    task = text_task("gripper__prob05")
    _, (name, table, options, fingerprint) = encoding.FlatRelationEncoder(task).__reduce__()
    with pytest.raises(ValueError, match="mymyr: encoding: layout must be one of"):
        pickle.loads(pickle.dumps(Tampered((name, table, {**options, "layout": "diagonal"}, fingerprint))))
