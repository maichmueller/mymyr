"""The versioned C API (mymyr/ext.h) seen from a pure-C downstream module that links nothing of mymyr."""

import sys
import threading

import numpy as np
import pytest

import mymyr
from mymyr import _core
from conftest import text_task, walk

consumer = pytest.importorskip("mymyr._testing._ext_consumer")


def test_capsule_and_version():
    assert _core._C_API_VERSION == (1, 1)
    abi, minor, size = consumer.abi()
    assert (abi, minor) == _core._C_API_VERSION and size > 16
    assert consumer.version() == mymyr.__version__


def test_state_hash_matches_native_and_ignores_trailing_zero_words():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2**63, size=(64, 4), dtype=np.uint64)
    expected = mymyr.hash_rows(rows)
    for r, h in zip(rows, expected):
        assert consumer.hash_state(r) == int(h)
        padded = np.concatenate([r, np.zeros(3, np.uint64)])
        assert consumer.hash_state(padded) == int(h)
        assert consumer.equal_states(r, padded)
    assert not consumer.equal_states(rows[0], rows[1])


def test_rejects_non_state_buffers():
    with pytest.raises(TypeError):
        consumer.hash_state(np.zeros(4, np.float64))
    with pytest.raises(TypeError):
        consumer.hash_state(np.zeros((2, 2), np.uint64))
    with pytest.raises(TypeError):
        consumer.task_info(np.zeros(4, np.uint64))


# ------------------------------------------------------------------------------------------------ minor 1


def test_task_view(blocks):
    uid, fp, n, p, s, w, mw, lw, atoms, mode = consumer.task_info(blocks)
    assert (uid, fp, n, s, w, mw, lw, atoms) == (blocks.uid, blocks.fingerprint, blocks.num_objects, blocks.num_schemas,
                                                blocks.words, blocks.max_words, blocks.label_width, blocks.num_atoms)
    assert p == blocks.num_predicates and mode == 1
    st = blocks.initial_state
    assert consumer.task_info(st)[0] == consumer.task_info(blocks.local())[0] == blocks.uid
    assert consumer.state_task_uid(st) == blocks.uid
    assert consumer.state_task_uid(np.asarray(st.words)) == 0  # plain buffers carry no task
    assert consumer.hash_state(st) == int(mymyr.hash_rows(np.asarray(st.words)[None, :])[0])


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "openstacks-opt08-adl__p03", "miconic-simpleadl__s10-2"])
def test_successors_apply_goal_atoms(name):
    task = text_task(name, atoms="lazy")
    for s in walk(task, steps=8, seed=2):
        c = consumer.successors(task, s)
        py = task.successors(s)
        assert [(sch, b) for sch, b, _ in c] == [a.label for a, _ in py]
        assert [task.state(np.array(w, np.uint64)) for _, _, w in c] == [t for _, t in py]
        assert consumer.count_successors_nogil(task, s) == len(py)
        if c:
            assert consumer.successors(task, s, 1) == c[:1]  # the emit callback can stop the enumeration
        for (a, t), (sch, b, w) in zip(py, c):
            assert consumer.apply(task, s, sch, list(b)) == w
        assert consumer.is_goal(task, s) == task.is_goal(s)
        for slot in s.atom_slots():
            atom = task.atom(slot)
            assert consumer.atom_slot(task, atom.predicate_index, atom.object_indices) == slot
            assert consumer.atom_of_slot(task, slot) == (atom.predicate_index, tuple(atom.object_indices))
    s = task.initial_state
    acts = task.applicable_actions(s)
    inapplicable = [(i, b) for i in range(task.num_schemas) for b in [[0] * task.formalism.schemas[i].arity]
                    if (i, tuple(b)) not in {a.label for a in acts}]
    for sch, b in inapplicable:
        assert consumer.apply(task, s, sch, b) is None
    assert consumer.atom_of_slot(task, 1 << 30) is None


def test_store(blocks):
    states = walk(blocks, steps=20, seed=6)
    ids, inserted, lookups, size, first = consumer.store_roundtrip(blocks, states + states[:3])
    distinct = list(dict.fromkeys(states))
    assert size == len(distinct)
    assert ids == [distinct.index(s) for s in states + states[:3]] == lookups
    assert inserted[: len(states)] == [states.index(s) == i for i, s in enumerate(states)] and not any(inserted[len(states):])
    assert blocks.state(np.array(first, np.uint64)) == states[0]


def test_c_api_from_many_threads(blocks):
    states = walk(blocks, steps=30, seed=1)
    expected = [consumer.successors(blocks, s) for s in states]
    results, errors = [None] * 16, []

    def work(i):
        try:
            h = blocks.local()
            results[i] = [consumer.successors(h, s) for s in states]
            assert [consumer.count_successors_nogil(h, s) for s in states] == [len(e) for e in expected]
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(r == expected for r in results)


def test_gil_stays_disabled():
    assert not sys._is_gil_enabled()
