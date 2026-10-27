"""Flat encoding of compact lazy States through fixed slot spaces."""

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from test_encoding import derived_words, feature_states, feature_task
from test_encoding_layers import same


def translate_states(states, task):
    return [task.state([str(a) for a in s.atoms()], **(
        {"values": s.numeric_values()} if task.numeric_slots else {})) for s in states]


def pddl(domain, problem, atoms):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task.from_pddl(domain, problem, atoms=atoms)


@pytest.mark.parametrize("domain", ["blocksworld", "ferry", "sokoban", "spanner"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_ipc_states_equal_an_independent_parse(domain, layout):
    directory = ROOT / "tests/data/ipc" / f"{domain}-ipc/train"
    lazy = pddl(directory / "domain.pddl", directory / "p01.pddl", "lazy")
    fixed = pddl(directory / "domain.pddl", directory / "p01.pddl", "auto")
    states = walk(lazy, steps=12)
    encoder = encoding.FlatRelationEncoder(lazy, layout=layout)
    expected = encoding.FlatRelationEncoder(fixed, layout=layout).encode(translate_states(states, fixed))
    same(encoder.encode(states), expected)
    assert encoder.table.uid == lazy.frozen().uid
    same(encoder.encode(states[0]), encoder.encode([states[0]]))
    stream = encoder.stream()
    stream.append(states[:10])
    stream.append(states[10:])
    same(stream.flush(), expected)
    mutable = encoder.mutable_stream()
    mutable.append(list(range(len(states))), states)
    same(mutable.flush(), expected)


@pytest.mark.parametrize("mode", ["auto", "reachable", "dense"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_twin_modes_and_destination_masks(mode, layout):
    lazy = feature_task(atoms="lazy")
    states = feature_states(lazy)
    twin = lazy.frozen(mode)
    fixed_states = translate_states(states, twin)
    encoder = encoding.FlatRelationEncoder(twin, layout=layout, max_goal_level=2, history=True,
                                           target_sources=("goal", "subgoal", "history"))
    layers = encoding.subgoal_masks(encoder, [[["(flag b)", "(not (available a))"], ["(ready)"]]] * len(states))
    history = encoding.history_masks(encoder, [[(-1, ["(flag a)"])]] * len(states))
    pos = np.zeros((len(states), twin.words), np.uint64)
    neg = np.zeros_like(pos)
    slot = twin.atom("(flag b)").slot
    pos[:, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    options = dict(subgoals=layers, history=history, goals=(pos, neg), derived=derived_words(twin, fixed_states))
    same(encoder.encode(states, **options), encoder.encode(fixed_states, **options))
    same(encoder.encode(states, **{k: v for k, v in options.items() if k != "derived"}),
         encoder.encode(fixed_states, **{k: v for k, v in options.items() if k != "derived"}))
    actions = rl.expand(twin, fixed_states)
    same(encoder.encode(states, actions=actions, **options), encoder.encode(fixed_states, actions=actions, **options))


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_numeric_words_keep_their_values(layout):
    lazy = mymyr.Task.from_text(ROOT / "tests/data/numeric_tasks/m-refuel.txt", atoms="lazy")
    fixed = lazy.frozen()
    states = walk(lazy, steps=8)
    converted = [fixed.state([str(a) for a in s.atoms()], s.numeric_values()) for s in states]
    encoder = encoding.FlatRelationEncoder(lazy, layout=layout)
    same(encoder.encode(states), encoder.encode(converted))
    stream = encoder.stream()
    stream.append(states)
    same(stream.flush(), encoder.encode(converted))


def test_multi_instance_twins_and_mixed_slot_spaces():
    if hasattr(mymyr, "Domain"):
        directory = ROOT / "tests/data/pddl/gripper"
        lazy = [pddl(directory / "domain.pddl", directory / f"p{i}.pddl", "lazy") for i in ("01", "02")]
    else:
        lazy = [text_task("gripper__prob05", atoms="lazy") for _ in range(2)]
    twins = [t.frozen() for t in lazy]
    encoder = encoding.FlatRelationEncoder(rl.TaskTable(twins))
    walks = [walk(t, steps=8, walks=1, seed=i * 17) for i, t in enumerate(lazy)]
    states = [s for pair in zip(*walks) for s in pair]
    ids = [i for _ in walks[0] for i in range(2)]
    fixed = [twins[i].state([str(a) for a in s.atoms()]) for s, i in zip(states, ids)]
    same(encoder.encode(states, ids), encoder.encode(fixed, ids))
    states[0] = fixed[0]
    same(encoder.encode(states, ids), encoder.encode(fixed, ids))
    with pytest.raises(ValueError, match="task_ids"):
        encoder.encode(states)


def test_pickle_keeps_the_source_and_twin_link():
    lazy = text_task("gripper__prob05", atoms="lazy")
    states = walk(lazy, steps=8)
    encoder = encoding.FlatRelationEncoder(lazy)
    twin, restored, restored_states = pickle.loads(pickle.dumps((lazy.frozen(), encoder, states)))
    assert restored.table.uid == twin.uid
    same(restored.encode(restored_states), encoder.encode(states))
    simple, restored_states = pickle.loads(pickle.dumps((encoding.SimpleEncoder(lazy), states)))
    same(simple.encode(restored_states), encoding.SimpleEncoder(lazy).encode(states))
    for original in (encoder, encoding.SimpleEncoder(lazy)):
        restored = pickle.loads(pickle.dumps(original))
        restored_states = pickle.loads(pickle.dumps(states))
        expected = original.encode(states)
        same(restored.encode(restored_states), expected)
        same(original.encode(restored_states), expected)
        same(restored.encode(states), expected)


@pytest.mark.parametrize("mode", ["auto", "reachable", "dense"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_states_from_an_independent_lazy_parse(mode, layout):
    owner = text_task("blocks__probBLOCKS-8-0", atoms="lazy")
    source = text_task("blocks__probBLOCKS-8-0", atoms="lazy")
    twin = owner.frozen(mode)
    encoder = encoding.FlatRelationEncoder(twin, layout=layout)
    for seed in (17, 39):
        states = walk(source, steps=12, seed=seed)
        same(encoder.encode(states * 4, threads=2), encoder.encode(translate_states(states, twin) * 4))
        stream = encoder.stream()
        stream.append(states)
        same(stream.flush(), encoder.encode(translate_states(states, twin)))


@pytest.mark.parametrize("storage", ["auto", "f64"])
def test_numeric_states_from_an_independent_lazy_parse(storage):
    path = ROOT / "tests/data/numeric_tasks/m-refuel.txt"
    owner = mymyr.Task.from_text(path, atoms="lazy", numeric_storage=storage)
    source = mymyr.Task.from_text(path, atoms="lazy", numeric_storage=storage)
    states = walk(source, steps=8)
    encoder = encoding.FlatRelationEncoder(owner)
    same(encoder.encode(states), encoder.encode(translate_states(states, encoder.table)))


def test_independent_lazy_parse_with_different_numeric_storage_is_refused():
    path = ROOT / "tests/data/numeric_tasks/m-refuel.txt"
    owner = mymyr.Task.from_text(path, numeric_storage="i32")
    source = mymyr.Task.from_text(path, atoms="lazy", numeric_storage="f64")
    with pytest.raises(ValueError, match="not a state of instance"):
        encoding.FlatRelationEncoder(owner).encode(source.initial_state)


@pytest.mark.parametrize("handle", [False, True])
def test_lazy_schema_matches_the_encoder(handle):
    lazy = text_task("gripper__prob05", atoms="lazy")
    source = lazy.local() if handle else lazy
    schema = encoding.EncodingSchema(source)
    assert schema.schema_fingerprint == encoding.FlatRelationEncoder(source).schema.schema_fingerprint
    assert schema.schema_fingerprint == encoding.EncodingSchema(lazy.frozen()).schema_fingerprint


def test_threads_and_searching_source():
    lazy = text_task("blocks__probBLOCKS-8-0", atoms="lazy")
    encoder = encoding.FlatRelationEncoder(lazy)
    states = walk(lazy, steps=10)
    expected = encoder.encode(states * 120)
    same(encoder.encode(states * 120, threads=4), expected)
    active = threading.Event()
    stop = threading.Event()

    def search():
        local = lazy.local()
        active.set()
        while not stop.is_set():
            walk(local, steps=40, walks=1)

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(search)
        assert active.wait(10)
        try:
            for _ in range(6):
                restored, restored_task = pickle.loads(pickle.dumps((encoder, lazy)))
                same(restored.encode(restored_task.initial_state), encoder.encode(lazy.initial_state))
                same(encoder.encode(states * 120, threads=4), expected)
                gathered = walk(lazy, steps=8, walks=1)
                fixed = [encoder.table.state([str(a) for a in s.atoms()]) for s in gathered]
                same(encoder.encode(gathered), encoder.encode(fixed))
        finally:
            stop.set()
            pending.result()


def test_unrelated_source_is_refused():
    lazy = text_task("gripper__prob05", atoms="lazy")
    encoder = encoding.FlatRelationEncoder(lazy)
    other = text_task("blocks__probBLOCKS-8-0", atoms="lazy")
    with pytest.raises(ValueError, match="not a state of instance"):
        encoder.encode(other.initial_state)


def test_unreachable_atom_names_the_missing_atom():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain unreachable)
      (:requirements :strips :typing) (:types item)
      (:predicates (open ?x - item) (done ?x - item))
      (:action unlock :parameters (?x - item) :precondition (done ?x) :effect (open ?x))
      (:action finish :parameters (?x - item) :precondition (open ?x) :effect (done ?x)))""")
    lazy = mymyr.Task(domain.instantiate_string("""(define (problem p) (:domain unreachable)
      (:objects a b - item) (:init (open a)) (:goal (done a)))"""), atoms="lazy")
    twin = lazy.frozen("reachable")
    encoder = encoding.FlatRelationEncoder(twin)
    bad = lazy.state(["(done b)"])
    with pytest.raises(ValueError, match=r"\(done b\).*no slot"):
        encoder.encode(bad)
    same(encoder.encode(lazy.initial_state), encoder.encode(twin.initial_state))


def test_empty_lazy_slot_space():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain empty)
      (:requirements :strips :typing) (:types item) (:predicates (flag ?x - item))
      (:action set :parameters (?x - item) :precondition (and) :effect (flag ?x)))""")
    lazy = mymyr.Task(domain.instantiate_string("""(define (problem p) (:domain empty)
      (:objects """ + " ".join(f"o{i}" for i in range(130)) + """ - item) (:init) (:goal (and)))"""), atoms="lazy")
    encoder = encoding.FlatRelationEncoder(lazy)
    assert lazy.num_atoms == 0 and encoder.table.words > 1
    same(encoder.encode(lazy.initial_state), encoder.encode(encoder.table.initial_state))


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_large_external_instance(layout):
    directory = Path(os.environ.get("MYMYR_IPC_DATA", ROOT / "tests/data/ipc")) / "blocksworld-ipc/test"
    if not (directory / "p30-hard.pddl").exists():
        pytest.skip("external IPC instance is unavailable")
    lazy = pddl(directory / "domain.pddl", directory / "p30-hard.pddl", "lazy")
    fixed = pddl(directory / "domain.pddl", directory / "p30-hard.pddl", "auto")
    states = walk(lazy, steps=3, walks=1)
    same(encoding.FlatRelationEncoder(lazy, layout=layout).encode(states),
         encoding.FlatRelationEncoder(fixed, layout=layout).encode(translate_states(states, fixed)))
