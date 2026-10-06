"""Coverage of the read-only formalism views: every declared field of every view class is reachable and consistent
on every exported suite task (plain lifted and numeric)."""

import gc
import pathlib

import pytest

from mymyr import formalism as fm

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIFTED = sorted((ROOT / "tests/data/tasks").glob("*.txt"))
NUMERIC = sorted((ROOT / "tests/data/numeric_tasks").glob("*.txt"))


def walk(obj, seen, counts):
    """Visit every field of every reachable view (depth-first), counting visits per class."""
    cls = type(obj)
    if cls in fm.VIEW_CLASSES:
        key = (cls.__name__, str(obj), getattr(obj, "index", None))
        if key in seen:
            return
        seen.add(key)
        counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
        repr(obj)
        for name in cls.fields:
            walk(getattr(obj, name), seen, counts)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            walk(x, seen, counts)


@pytest.mark.parametrize("path", LIFTED + NUMERIC, ids=lambda p: p.stem)
def test_every_field_is_reachable(path):
    t = fm.read_task_text(str(path))
    counts = {}
    walk(t, set(), counts)
    assert counts["NormalizedTask"] == 1
    assert counts.get("Schema", 0) == len(t.schemas) > 0
    assert counts.get("Object", 0) == len(t.objects)


def test_suite_present():
    assert len(LIFTED) >= 20 and len(NUMERIC) >= 10


def test_structure_openstacks():
    t = fm.read_task_text(str(ROOT / "tests/data/tasks/openstacks-opt08-adl__p03.txt"))
    assert len(t.objects) == 22
    assert len(t.axioms) == 2
    derived = [p for p in t.predicates if p.kind == "derived"]
    assert len(derived) == 2
    for a in t.axioms:
        pred, terms = a.head
        assert pred.kind == "derived" and len(terms) == pred.arity
        assert all(isinstance(x, fm.Variable) for x in terms)
    goal = t.goal
    assert len(goal) == len(goal.literals) > 0
    for lit in goal.literals:
        assert lit.is_ground and all(isinstance(x, fm.Object) for x in lit.terms)


def test_terms_resolve_to_scope():
    for path in LIFTED:
        t = fm.read_task_text(str(path))
        for s in t.schemas:
            names = [p.name for p in s.parameters]
            assert [p.position for p in s.parameters] == list(range(s.arity))
            assert s.original_arity <= s.arity
            for lit in s.precondition.literals:
                for x in lit.terms:
                    if isinstance(x, fm.Variable):
                        assert x.name == names[x.position]
            for ce in s.effects:
                extra = [p.name for p in ce.parameters]
                scope = names + extra
                for lit in ce.add_effects + ce.delete_effects + ce.condition.literals:
                    assert lit.predicate.kind in ("static", "fluent", "derived")
                    for x in lit.terms:
                        if isinstance(x, fm.Variable):
                            assert x.name == scope[x.position]
                for lit in ce.add_effects + ce.delete_effects:
                    assert lit.predicate.kind == "fluent"
                assert all(l.positive for l in ce.add_effects)
                assert not any(l.positive for l in ce.delete_effects)


def test_condition_partitions_by_kind():
    for path in LIFTED:
        t = fm.read_task_text(str(path))
        for s in t.schemas:
            c = s.precondition
            assert len(c.static_literals) + len(c.fluent_literals) + len(c.derived_literals) == len(c.literals)


def test_initial_state_atoms():
    for path in LIFTED:
        t = fm.read_task_text(str(path))
        for a in t.static_init:
            assert a.predicate.kind == "static" and len(a.objects) == a.predicate.arity
        for a in t.fluent_init:
            assert a.predicate.kind == "fluent" and len(a.objects) == a.predicate.arity


def test_numeric_views():
    seen_effect = seen_constraint = seen_metric = False
    for path in NUMERIC:
        t = fm.read_task_text(str(path))
        for v in t.static_values + t.fluent_values:
            assert isinstance(v.value, float) and len(v.objects) == v.function.arity
        if t.metric is not None:
            direction, expr = t.metric
            assert direction in ("minimize", "maximize")
            str(expr)
            seen_metric = True
        for s in t.schemas:
            for k in s.precondition.numeric_constraints:
                assert k.comparator in ("=", "!=", "<", "<=", ">", ">=")
                str(k.lhs), str(k.rhs), str(k)
                seen_constraint = True
            for ce in s.effects:
                for e in ce.numeric_effects:
                    assert e.operator in ("assign", "increase", "decrease", "scale-up", "scale-down")
                    assert e.function.kind == "fluent"
                    str(e)
                    seen_effect = True
    assert seen_effect and seen_constraint and seen_metric


def test_expression_tree():
    for path in NUMERIC:
        t = fm.read_task_text(str(path))
        for s in t.schemas:
            for k in s.precondition.numeric_constraints:
                stack = [k.lhs, k.rhs]
                while stack:
                    e = stack.pop()
                    if e.op == "number":
                        assert e.value is not None and e.function is None and not e.children
                    elif e.op == "function":
                        assert e.value is None and e.function is not None
                    else:
                        assert len(e.children) == (1 if e.op == "neg" else 2)
                        stack.extend(e.children)


def test_identity_and_hash():
    path = str(LIFTED[0])
    t = fm.read_task_text(path)
    a = t.predicates
    b = t.predicates
    assert a == b and [hash(x) for x in a] == [hash(x) for x in b]
    assert len(set(a)) == len(a)
    other = fm.read_task_text(path)
    assert other.predicates[0] != a[0]  # identity is per task instance


def test_views_keep_task_alive():
    t = fm.read_task_text(str(LIFTED[0]))
    schemas = t.schemas
    names = [s.name for s in schemas]
    del t
    gc.collect()
    assert [s.name for s in schemas] == names
    assert all(len(s.parameters) == s.arity for s in schemas)


def test_to_text_round_trip():
    for path in LIFTED[:5] + NUMERIC[:3]:
        t = fm.read_task_text(str(path))
        assert t.to_text().split() == path.read_text().split()


def test_concurrent_walks():
    """Views are immutable: many threads may walk one task at once (free-threaded build: truly parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    t = fm.read_task_text(str(LIFTED[0]))

    def job(_):
        counts = {}
        walk(t, set(), counts)
        return counts

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(job, range(32)))
    assert all(r == results[0] for r in results)
