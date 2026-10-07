"""Binding generators: Task.bindings, Task.ground_conjunctions, applicable_actions(schema=, partial=), and the
conditions of Task.precondition / Task.goal_condition."""

import itertools
import sys
import threading

import pytest

import mymyr
from conftest import SMALL_TASKS, ROOT, text_task, walk

NUMERIC = ROOT / "tests/data/numeric_tasks"


def by_schema(actions):
    out = {}
    for a in actions:
        out.setdefault(a.schema, []).append(a)
    return out


@pytest.fixture(scope="module", params=SMALL_TASKS)
def task_states(request):
    task = text_task(request.param)
    return task, walk(task, steps=8, walks=2)


def test_schema_bindings_are_the_applicable_actions(task_states):
    task, states = task_states
    for s in states:
        actions = by_schema(task.applicable_actions(s))
        for k in range(task.num_schemas):
            expected = actions.get(k, [])
            got = list(task.bindings(k, s))
            assert sorted(got, key=lambda a: a.binding) == expected
            assert len(set(got)) == len(got)
            assert task.applicable_actions(s, schema=k) == expected


def test_partial_bindings_match_the_filtered_actions(task_states):
    task, states = task_states
    for s in states[:6]:
        actions = by_schema(task.applicable_actions(s))
        for k, acts in actions.items():
            for a in acts[:3]:
                n = a.arity
                subsets = [()] + [c for r in (1, 2) for c in itertools.combinations(range(n), r)] + [tuple(range(n))]
                for fixed in subsets:
                    expected = [b for b in acts if all(b.binding[i] == a.binding[i] for i in fixed)]
                    by_index = {i: a.binding[i] for i in fixed}
                    by_name = {task.precondition(k).variables[i]: a.objects[i] for i in fixed}
                    as_list = [a.objects[i] if i in fixed else None for i in range(n)]
                    assert task.applicable_actions(s, schema=k, partial=by_index) == expected
                    assert sorted(task.bindings(k, s, partial=by_name), key=lambda b: b.binding) == expected
                    assert sorted(task.bindings(k, s, partial=as_list), key=lambda b: b.binding) == expected


def test_limit_takes_a_prefix_of_the_order(task_states):
    task, states = task_states
    for s in states[:4]:
        for k in range(task.num_schemas):
            full = list(task.bindings(k, s))
            for limit in (0, 1, 2, 17, 40):
                assert list(task.bindings(k, s, limit=limit)) == full[:limit]
            assert list(task.bindings(k, s)) == full  # deterministic


def test_many_bindings_cross_the_chunk_boundaries():
    # The iterator fills chunks of 16, 32, ... bindings, each continuing after the last one of the previous chunk.
    task = text_task("gripper__prob05")
    s = task.initial_state
    for c in (task.precondition("pick"), "pick"):
        full = list(task.bindings(c, s))
        assert len(full) == len(set(full)) == 24
        assert list(task.bindings(c, s, limit=len(full) + 5)) == full
        for limit in (15, 16, 17, 23, 24):
            assert list(task.bindings(c, s, limit=limit)) == full[:limit]
    assert len(task.applicable_actions(s, schema="pick")) == 24


def test_precondition_agrees_with_is_applicable(task_states):
    task, states = task_states
    for s in states[:6]:
        applicable = set(task.applicable_actions(s))
        for k in range(task.num_schemas):
            c = task.precondition(k)
            assert c.arity == task.formalism.schemas[k].arity
            for b in task.bindings(c, s):
                assert all(isinstance(o, mymyr.formalism.Object) for o in b)
            for a in applicable:
                if a.schema == k:
                    got = list(task.bindings(c, s, partial=list(a.objects)))
                    assert [tuple(o.name for o in b) for b in got] == [tuple(a.objects)]


def test_goal_condition_agrees_with_is_goal():
    for name in SMALL_TASKS:
        task = text_task(name)
        g = task.goal_condition
        assert isinstance(g, mymyr.GroundCondition) and len(g.literals) > 0
        for s in walk(task, steps=6, walks=1):
            assert list(task.bindings(g, s)) == ([()] if task.is_goal(s) else [])


def test_ground_conjunctions_hold_in_the_state():
    task = text_task("philosophers__p03-phil4")  # derived predicates
    for s in walk(task, steps=6, walks=1):
        true = {str(a) for a in s.atoms()} | {str(a) for a in task.derived_atoms(s)}
        for k in range(task.num_schemas):
            items = list(task.ground_conjunctions(task.precondition(k), s))
            assert [b for b, *_ in items] == list(task.bindings(task.precondition(k), s))
            for _, static, fluent, derived in items:
                for lit in fluent + derived:
                    assert (str(lit.atom) in true) == lit.positive, lit
                for lit in static:
                    assert lit.atom.kind == "static"
                assert all(lit.atom.kind == "fluent" for lit in fluent)
                assert all(lit.atom.kind == "derived" for lit in derived)
            labelled = list(task.ground_conjunctions(k, s, limit=3))
            assert [a for a, *_ in labelled] == list(task.bindings(k, s, limit=3))


def test_ground_literal_values():
    task = text_task("blocks__probBLOCKS-8-0")
    s = task.initial_state
    a, static, fluent, derived = next(task.ground_conjunctions("unstack", s))
    assert isinstance(a, mymyr.Action) and derived == []
    lit = fluent[0]
    assert isinstance(lit, mymyr.GroundLiteral) and lit.positive
    assert lit == fluent[0] and hash(lit) == hash(fluent[0])
    assert str(lit) == str(lit.atom) and repr(lit).startswith("GroundLiteral(")


def test_numeric_schemas_follow_the_applicable_actions():
    for name in ("cs-counters", "cs-farmland", "cs-hydropower", "cs-delivery"):
        task = mymyr.Task.from_text(str(NUMERIC / f"{name}.txt"))
        for s in walk(task, steps=6, walks=2):
            actions = by_schema(task.applicable_actions(s))
            for k in range(task.num_schemas):
                assert task.applicable_actions(s, schema=k) == actions.get(k, [])


def test_conditions_compare_and_print():
    task = text_task("blocks__probBLOCKS-8-0")
    c = task.precondition("stack")
    assert c == task.precondition("stack") and hash(c) == hash(task.precondition("stack"))
    assert c != task.precondition("unstack") and c != task.goal_condition
    assert str(c).startswith("(?x0 ?x1) (and ") and repr(c) == "ConjunctiveCondition" + str(c)
    h = task.local()
    assert h.precondition("stack") == c


def test_invalid_arguments_raise():
    task = text_task("blocks__probBLOCKS-8-0")
    other = text_task("gripper__prob05")
    s = task.initial_state
    with pytest.raises(ValueError, match="partial= needs schema="):
        task.applicable_actions(s, partial={0: "o1"})
    with pytest.raises(ValueError, match="expected 2 entries"):
        task.bindings("stack", s, partial=["o1"])
    with pytest.raises(KeyError, match="no variable named"):
        task.bindings("stack", s, partial={"?nope": "o1"})
    with pytest.raises(IndexError, match="variable index"):
        task.bindings("stack", s, partial={5: "o1"})
    with pytest.raises(KeyError, match="no object named"):
        task.bindings("stack", s, partial={0: "nope"})
    with pytest.raises(TypeError):
        task.bindings("stack", s, partial="o1")
    with pytest.raises(KeyError, match="no schema named"):
        task.bindings("nope", s)
    with pytest.raises(IndexError):
        task.bindings(99, s)
    with pytest.raises(ValueError, match="another task"):
        task.bindings(other.precondition(0), s)
    with pytest.raises(ValueError, match="limit"):
        task.bindings("stack", s, limit=-1)


def test_a_fixed_object_outside_the_domain_gives_no_binding():
    task = text_task("gripper__prob05")
    s = task.initial_state
    pick = task.precondition("pick")
    # the first parameter of pick is a ball: fixing a room or a gripper there gives nothing, not an error
    rooms = {a.objects[1] for a in task.applicable_actions(s, schema="pick")}
    assert list(task.bindings(pick, s, partial={0: sorted(rooms)[0]})) == []


def test_objects_from_the_formalism_views():
    task = text_task("blocks__probBLOCKS-8-0")
    s = task.initial_state
    objects = task.formalism.objects
    a = task.applicable_actions(s, schema="unstack")[0]
    got = task.applicable_actions(s, schema="unstack", partial={0: objects[a.binding[0]]})
    assert a in got
    first = next(task.bindings(task.precondition("unstack"), s, partial={0: objects[a.binding[0]]}))
    assert first[0] == objects[a.binding[0]]


def test_concurrent_enumeration_over_one_task():
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=10, walks=2)
    schemas = range(task.num_schemas)
    expected = [[list(task.bindings(k, s)) for k in schemas] for s in states]
    conds = [task.precondition(k) for k in schemas]
    expected_c = [[list(task.bindings(c, s)) for c in conds] for s in states]
    errors = []

    def work(i):
        try:
            h = task.local() if i % 2 else task
            for _ in range(3):
                for j, s in enumerate(states):
                    for k in schemas:
                        assert list(h.bindings(k, s)) == expected[j][k]
                        assert list(task.bindings(conds[k], s, limit=7)) == expected_c[j][k][:7]
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


@pytest.mark.skipif(getattr(sys, "_is_gil_enabled", lambda: True)(), reason="needs a free-threaded interpreter")
def test_one_iterator_shared_by_threads():
    task = text_task("gripper__prob05")
    s = task.initial_state
    c = task.precondition("pick")
    expected = list(task.bindings(c, s))
    it = task.bindings(c, s)
    seen = [[] for _ in range(8)]

    def work(i):
        for b in it:
            seen[i].append(b)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    def names(bs):
        return sorted(tuple(o.name for o in b) for b in bs)

    got = [b for part in seen for b in part]
    assert names(got) == names(expected) and len(got) == len(expected)
