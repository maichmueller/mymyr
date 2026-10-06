"""States from explicit atoms and numeric values, derived atoms, the lazy applicable-action iterator, any_applicable
and is_dead_end."""

import threading

import numpy as np
import pytest

import mymyr
from conftest import ROOT, SMALL_TASKS, text_task, walk

NUMERIC = ROOT / "tests/data/numeric_tasks"


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_state_from_atoms_equals_the_state(name):
    task = text_task(name)
    for s in walk(task, steps=10, walks=1):
        assert task.state(s.atoms()) == s
        assert task.state({str(a) for a in s.atoms()}) == s  # any order
        assert task.state(list(s.atoms()) * 2) == s  # repeats


def test_state_atoms_are_validated():
    task = text_task("philosophers__p03-phil4")
    kinds = {p.name: p.kind for p in task.formalism.predicates}
    derived = next(a for a in task.initial_state.derived_atoms() if kinds[a.predicate] == "derived")
    with pytest.raises(ValueError, match="not a fluent atom"):
        task.state([str(derived)])
    static = task.formalism.static_init[0]
    with pytest.raises(ValueError, match="not a fluent atom"):
        task.state([(static.predicate.name, *[str(o) for o in static.objects])])
    with pytest.raises(KeyError):
        task.state(["(no-such-predicate)"])
    with pytest.raises(ValueError, match="values"):
        task.state(task.initial_state.atoms(), values=[1.0])


def test_derived_atoms_follow_from_the_axioms():
    task = text_task("openstacks-opt08-adl__p03")  # axioms
    for s in walk(task, steps=8, walks=2):
        d = s.derived_atoms()
        assert d == task.derived_atoms(s)
        assert all(s.holds(a) for a in d)
        assert task.state(s.atoms()).derived_atoms() == d
    assert text_task("gripper__prob05").initial_state.derived_atoms() == []


def numeric_task(name):
    return mymyr.Task.from_text(str(NUMERIC / f"{name}.txt"))


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower"])
def test_numeric_state_from_atoms_and_values(name):
    task = numeric_task(name)
    s = task.successor_states(task.initial_state)[-1]
    v = s.numeric_values()
    assert task.state(s.atoms(), values=v) == s
    assert task.state(s.atoms(), values=list(v)) == s
    assert task.state(s.atoms(), values=dict(zip(task.numeric_names, v.tolist()))) == s
    other = v.copy()
    other[0] += 1
    t = task.state(s.atoms(), values=other)
    assert t != s and np.array_equal(t.numeric_values(), other)
    with pytest.raises(ValueError, match="numeric slots"):
        task.state(s.atoms())
    with pytest.raises(ValueError, match="numeric values for"):
        task.state(s.atoms(), values=v[:-1])
    with pytest.raises(ValueError, match="no value for"):
        task.state(s.atoms(), values={task.numeric_names[0]: 1.0})
    with pytest.raises(ValueError, match="no numeric slot named"):
        task.state(s.atoms(), values={"(nope)": 1.0})
    with pytest.raises(ValueError, match="NaN"):
        task.state(s.atoms(), values=[float("nan")] * len(v))


def test_int32_slots_reject_fractions():
    task = numeric_task("cs-counters")
    assert task.numeric_storage == "i32"
    s = task.initial_state
    with pytest.raises(OverflowError):
        task.state(s.atoms(), values=[0.5] * task.numeric_slots)


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_iter_applicable_actions(name):
    task = text_task(name)
    for s in walk(task, steps=10, walks=1):
        it = task.iter_applicable_actions(s)
        assert isinstance(it, mymyr.ApplicableActions) and iter(it) is it
        assert list(it) == task.applicable_actions(s) == list(s.iter_applicable_actions())
        assert list(it) == []  # exhausted
        assert s.any_applicable() == task.any_applicable(s) == bool(task.applicable_actions(s))
        assert s.is_dead_end() == task.is_dead_end(s) == (not task.applicable_actions(s))


def test_iterator_survives_interleaved_calls_and_threads():
    task = text_task("logistics00__probLOGISTICS-6-1")
    s = task.initial_state
    expected = task.applicable_actions(s)
    it = task.iter_applicable_actions(s)
    got = [next(it)]
    for t in task.successor_states(task.successor_states(s)[0]):  # other states in this thread's workspace
        task.applicable_actions(t)
    out = []
    worker = threading.Thread(target=lambda: out.extend(it))  # finished on another thread
    worker.start()
    worker.join()
    assert got + out == expected


def test_dead_end():
    task = numeric_task("cs-counters")
    goal = mymyr.search.astar(task, heuristic="blind").goal_state
    assert task.any_applicable(goal) == (not task.is_dead_end(goal))
    blocks = text_task("blocks__probBLOCKS-8-0")
    empty = blocks.state([])  # no atoms: no block is clear, the hand is not empty
    assert empty.is_dead_end() and not empty.any_applicable()
    assert list(empty.iter_applicable_actions()) == []
