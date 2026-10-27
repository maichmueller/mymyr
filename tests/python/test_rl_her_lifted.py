"""Lifted hindsight relabelling: the goal templates (mymyr.rl.lifted_goal_templates, GoalTemplates) and the binding
order of conditions they are matched in.

The templates are checked against the pure-Python reference in her_lifted_reference.py on IPC 2023 samples, the text
fixtures and large goals, and for their bounds, determinism, independence from other instances, the packed arrays,
the condition round trip and pickling.
"""

import math
import pickle
import random
import time

import numpy as np
import pytest

import mymyr
from conftest import ROOT, text_task, walk
from her_lifted_reference import below, philox4x32_10, subgoals, templates
from mymyr import rl
from mymyr._core import _rl_torch

IPC = ROOT / "tests/data/ipc"
DOMAINS = ["blocksworld", "childsnack", "ferry", "floortile", "miconic", "rovers", "satellite", "sokoban", "spanner",
           "transport"]
TEXT = ["blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1", "depot__p02",
        "miconic-simpleadl__s10-2", "openstacks-opt08-adl__p03", "philosophers__p03-phil4"]
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def ipc(domain, split, name):
    d = IPC / f"{domain}-ipc" / split
    return mymyr.Task.from_pddl(d / "domain.pddl", d / f"{name}.pddl")


def check_reference(t, i, task, **options):
    tuples, conds = templates(task, i, **options)
    assert t.goal_literals(i) == tuples
    assert t.conditions(i) == conds


def sizes(t, i):
    return [len(g) for g in t.goal_literals(i)]


# ------------------------------------------------------------------------------------------------ the RNG port


def test_reference_rng_matches_the_native_philox():
    counters = [(0, 0, 0, 2), (5, 1, 3, 2), (0xFFFFFFFF, 7, 11, 2), (99, 0, 2**31, 2)]
    for seed in (0, 7, 2**40 + 3):
        native = np.asarray(_rl_torch.philox(np.array(counters, dtype=np.uint32), seed))
        for c, row in zip(counters, native):
            assert philox4x32_10(c, seed & 0xFFFFFFFF, seed >> 32) == tuple(int(v) for v in row)


def test_reference_below_is_lemire():
    for x, n in [(0, 5), (2**64 - 1, 5), (2**63, 3), (123456789123456789, 1000)]:
        assert below(x, n) == (x * n) >> 64


# ------------------------------------------------------------------------------------------------ reference agreement


@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
def test_templates_equal_the_reference_on_ipc_samples(domain):
    picks = [("train", "p01"), ("train", "p02"), ("test", "p01-medium"), ("test", "p02-easy")]
    if domain != "satellite":
        picks.append(("test", "p05-medium") if (IPC / f"{domain}-ipc/test/p05-medium.pddl").exists() else ("test", "p03-easy"))
    tasks = [ipc(domain, s, n) for s, n in picks]
    table = rl.TaskTable(tasks)
    for seed in (0, 5):
        t = rl.lifted_goal_templates(table, seed=seed)
        assert t.num_instances == len(tasks)
        for i, task in enumerate(tasks):
            check_reference(t, i, task, seed=seed)


@needs_frontend
@pytest.mark.parametrize("distinct", [False, "equality"])
def test_templates_equal_the_reference_without_distinct_variables(distinct):
    tasks = [ipc("ferry", "test", "p05-medium"), ipc("sokoban", "test", "p05-medium")]
    t = rl.lifted_goal_templates(rl.TaskTable([tasks[0]]), distinct=distinct)
    check_reference(t, 0, tasks[0], distinct=distinct)
    for c in t.conditions(0):
        assert c.equalities == ()  # no IPC 2023 domain declares :equality
    t = rl.lifted_goal_templates(tasks[1], distinct=distinct, max_size=4, max_subgoals_per_size=3)
    check_reference(t, 0, tasks[1], distinct=distinct, max_size=4, max_subgoals_per_size=3)


@pytest.mark.parametrize("name", TEXT)
def test_templates_equal_the_reference_on_fixtures(name):
    task = text_task(name)
    for options in [{}, {"max_size": 3, "max_subgoals_per_size": 4, "seed": 9},
                    {"max_subsets_per_size": 2, "max_component_samples": 2, "max_products": 3, "seed": 1}]:
        t = rl.lifted_goal_templates(task, **options)
        check_reference(t, 0, task, **options)


@needs_frontend
def test_templates_of_the_pddl_fixtures():
    gripper = ROOT / "tests/data/pddl/gripper"
    counters = ROOT / "tests/data/pddl/counters"
    for d, p in [(gripper, "p01"), (gripper, "p02"), (counters, "p01"), (counters, "p02")]:
        task = mymyr.Task.from_pddl(d / "domain.pddl", d / f"{p}.pddl")
        t = rl.lifted_goal_templates(task)
        check_reference(t, 0, task)
    # counters: one nullary goal literal, one template without variables
    t = rl.lifted_goal_templates(mymyr.Task.from_pddl(counters / "domain.pddl", counters / "p01.pddl"))
    assert len(t) == 1 and t.max_vars == 0 and t.conditions(0)[0].arity == 0


@needs_frontend
@pytest.mark.parametrize("domain, name", [("floortile", "p10-hard"), ("satellite", "p10-hard"), ("satellite", "p30-hard")])
def test_large_goals_within_the_time_target(domain, name):
    # floortile p10-hard: two components of 425 literals; satellite: one sparse component of 352 / 583 literals
    task = ipc(domain, "test", name)
    rl.lifted_goal_templates(task)  # warm-up (the binding orders' compilation)
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        t = rl.lifted_goal_templates(task)
        best = min(best, time.perf_counter() - start)
    assert best < 2.0  # about 50 ms on a desktop core; the bound only catches an exponential search
    assert len(t) == 20 and sorted(sizes(t, 0), reverse=True) == sizes(t, 0)
    if domain == "floortile":
        check_reference(t, 0, task)


def goal_graph_task(images, modes, extra_singletons, seed):
    """A task whose goal graph has the shape of the large IPC rovers goals: `images` literals (img ?o ?m) over few
    modes (one large component of mode cliques joined through shared objectives) and isolated singleton literals."""
    rng = random.Random(seed)
    objectives = max(1, images // 2)
    pairs = sorted({(rng.randrange(objectives), rng.randrange(modes)) for _ in range(images)})
    rng.shuffle(pairs)
    domain = mymyr.Domain.from_string(
        "(define (domain goalgraph) (:requirements :strips :typing) (:types objective mode spot)"
        " (:predicates (img ?o - objective ?m - mode) (soil ?w - spot))"
        " (:action image :parameters (?o - objective ?m - mode) :precondition (and) :effect (img ?o ?m))"
        " (:action sample :parameters (?w - spot) :precondition (and) :effect (soil ?w)))"
    )
    objects = (" ".join(f"o{i}" for i in range(objectives)) + " - objective "
               + " ".join(f"m{i}" for i in range(modes)) + " - mode "
               + " ".join(f"w{i}" for i in range(extra_singletons)) + " - spot")
    goal = " ".join(f"(img o{o} m{m})" for o, m in pairs) + " " + " ".join(f"(soil w{i})" for i in range(extra_singletons))
    problem = f"(define (problem p) (:domain goalgraph) (:objects {objects}) (:init) (:goal (and {goal})))"
    return mymyr.Task(domain.instantiate_string(problem))


@needs_frontend
def test_dense_goal_graph_within_the_time_target():
    task = goal_graph_task(images=640, modes=3, extra_singletons=60, seed=0)
    rl.lifted_goal_templates(task)
    start = time.perf_counter()
    t = rl.lifted_goal_templates(task)
    assert time.perf_counter() - start < 2.0
    check_reference(t, 0, task)


def sparse_goal_task(shape, rng):
    """A task whose goal graph is a path, the line graph of a random tree, or a sparse random graph of unary, binary
    and ternary literals, with the goal literals in random order."""
    objects = {"path": 40, "tree": 40, "sparse": 80}[shape]
    count = rng.randint(10, {"path": 36, "tree": 36, "sparse": 60}[shape])
    literals = set()
    for _ in range(10 * count):
        if len(literals) == count:
            break
        if shape == "path":
            a = rng.randrange(objects - 1)
            literals.add(("b", (a, a + 1)))
        elif shape == "tree":
            a = rng.randrange(1, objects)
            literals.add(("b", (rng.randrange(a), a)))
        else:
            arity = rng.choice([1, 2, 2, 3])
            literals.add(({1: "u", 2: "b", 3: "t"}[arity], tuple(rng.randrange(objects) for _ in range(arity))))
    literals = sorted(literals)
    rng.shuffle(literals)
    domain = mymyr.Domain.from_string(
        "(define (domain sparsegoals) (:requirements :strips)"
        " (:predicates (u ?x) (b ?x ?y) (t ?x ?y ?z))"
        " (:action a1 :parameters (?x) :precondition (and) :effect (u ?x))"
        " (:action a2 :parameters (?x ?y) :precondition (and) :effect (b ?x ?y))"
        " (:action a3 :parameters (?x ?y ?z) :precondition (and) :effect (t ?x ?y ?z)))"
    )
    goal = " ".join(f"({p} {' '.join(f'o{o}' for o in args)})" for p, args in literals)
    names = " ".join(f"o{i}" for i in range(objects))
    problem = f"(define (problem p) (:domain sparsegoals) (:objects {names}) (:init) (:goal (and {goal})))"
    return mymyr.Task(domain.instantiate_string(problem))


@needs_frontend
def test_sparse_goal_graphs_equal_the_reference():
    # paths and trees reach deep levels of the subset search; each level's state must survive the deeper ones
    for case in range(120):
        rng = random.Random(case)
        task = sparse_goal_task(["path", "tree", "sparse"][case % 3], rng)
        options = {"max_size": rng.randint(4, 10), "max_subsets_per_size": rng.randint(1, 50), "seed": case}
        t = rl.lifted_goal_templates(task, **options)
        assert t.goal_literals(0) == subgoals(task, 0, **options), (case, options)


# ------------------------------------------------------------------------------------------------ bounds and order


@pytest.mark.parametrize("max_size, per_size", [(10, 2), (3, 1), (5, 4), (1, 3)])
def test_bounds_and_order(max_size, per_size):
    for name in TEXT:
        task = text_task(name)
        t = rl.lifted_goal_templates(task, max_size=max_size, max_subgoals_per_size=per_size)
        s = sizes(t, 0)
        assert s == sorted(s, reverse=True)
        assert all(1 <= k <= max_size for k in s)
        assert all(s.count(k) <= per_size for k in set(s))
        n = len(task.goal_condition.literals)
        for g in t.goal_literals(0):
            assert len(set(g)) == len(g) and all(0 <= i < n for i in g)
        assert s.count(1) >= 1  # a drawn component contributes its single literals


@pytest.mark.parametrize("name", ["logistics00__probLOGISTICS-6-1", "miconic-simpleadl__s10-2", "depot__p02"])
def test_subgoals_join_connected_subsets_of_distinct_components(name):
    task = text_task(name)
    lits = task.goal_condition.literals
    objs = [set(l.atom.object_indices) for l in lits]
    component = list(range(len(lits)))  # union-find over shared objects

    def find(i):
        while component[i] != i:
            i = component[i]
        return i

    for i in range(len(lits)):
        for j in range(i):
            if objs[i] & objs[j]:
                component[find(i)] = find(j)
    t = rl.lifted_goal_templates(task, max_subgoals_per_size=5)
    for g, c in zip(t.goal_literals(0), t.conditions(0)):
        parts = {}
        for i in g:
            parts.setdefault(find(i), []).append(i)
        assert [i for part in parts.values() for i in part] == list(g)  # components in order, each one contiguous
        for part in parts.values():
            assert part == sorted(part)
            seen, stack = {part[0]}, [part[0]]
            while stack:
                v = stack.pop()
                for u in part:
                    if u not in seen and objs[u] & objs[v]:
                        seen.add(u)
                        stack.append(u)
            assert seen == set(part)
        # one variable per distinct object, every pair distinct
        n = len(set().union(*(objs[i] for i in g)))
        assert c.arity == n and len(c.equalities) == n * (n - 1) // 2


# ------------------------------------------------------------------------------------------------ determinism


def test_deterministic_under_the_seed():
    tasks = [text_task(n) for n in ["gripper__prob05", "miconic-simpleadl__s10-2", "logistics00__probLOGISTICS-6-1"]]
    for task in tasks:
        a = rl.lifted_goal_templates(task, seed=3)
        b = rl.lifted_goal_templates(task, seed=3)
        assert a.goal_literals(0) == b.goal_literals(0)
        for k, v in a.arrays().items():
            np.testing.assert_array_equal(v, b.arrays()[k])
    # several components: the component draws depend on the seed
    task = text_task("miconic-simpleadl__s10-2")
    draws = {tuple(rl.lifted_goal_templates(task, seed=s).goal_literals(0)) for s in range(6)}
    assert len(draws) > 1


@needs_frontend
def test_independent_of_the_other_instances():
    a, b = ipc("miconic", "test", "p05-medium"), ipc("miconic", "test", "p04-medium")
    c, x = ipc("miconic", "test", "p16-medium"), ipc("miconic", "train", "p02")
    t1 = rl.lifted_goal_templates(rl.TaskTable([a, b, c]), seed=4)
    t2 = rl.lifted_goal_templates(rl.TaskTable([a, x, c]), seed=4)
    for i in (0, 2):
        assert t1.goal_literals(i) == t2.goal_literals(i)
        assert t1.conditions(i) == t2.conditions(i)
    # a Task is its table of one: instance 0
    assert rl.lifted_goal_templates(a, seed=4).goal_literals(0) == t1.goal_literals(0)
    # suites: instance ids are global
    g1 = ipc("ferry", "test", "p05-medium")
    suite = rl.TaskSuite.group([a, g1, c])
    t3 = rl.lifted_goal_templates(suite, seed=4)
    assert t3.goal_literals(0) == t1.goal_literals(0) and t3.goal_literals(2) == t1.goal_literals(2)
    check_reference(t3, 1, g1, seed=4)
    assert t3.conditions(g1) == t3.conditions(1) and t3.conditions(-1) == t3.conditions(2)


# ------------------------------------------------------------------------------------------------ packed arrays


@pytest.mark.parametrize("name", ["gripper__prob05", "philosophers__p03-phil4", "logistics00__probLOGISTICS-6-1"])
def test_packed_arrays(name):
    task = text_task(name)
    table = rl.TaskTable([task, task])
    t = rl.lifted_goal_templates(table, seed=2)
    a = t.arrays()
    I, K = t.num_instances, t.num_templates
    off, lo = a["instance_offsets"], a["literal_offsets"]
    assert off.shape == (I + 1,) and off[0] == 0 and off[-1] == K == len(t)
    assert lo.shape == (K + 1,) and lo[-1] == a["predicate"].shape[0]
    np.testing.assert_array_equal(np.diff(lo), a["template_size"])
    assert a["order"].shape == (K, t.max_vars) and a["terms"].shape == (lo[-1], t.max_arity)
    assert a["distinct"].tolist() == [1] * K
    assert t.nbytes > 0 and t.options["seed"] == 2 and t.options["distinct"] is True
    names, kinds = table.predicate_names, table.predicate_kinds
    lits = task.goal_condition.literals
    for i in range(I):
        conds = t.conditions(i)
        for k in range(off[i], off[i + 1]):
            c = conds[k - off[i]]
            assert a["num_vars"][k] == c.arity
            order = a["order"][k]
            assert tuple(order[: c.arity]) == c.binding_order(task) and (order[c.arity:] == -1).all()
            assert sorted(order[: c.arity].tolist()) == list(range(c.arity))
            for j, l in enumerate(range(lo[k], lo[k + 1])):
                lit = lits[a["goal_literal"][l]]
                assert names[a["predicate"][l]] == lit.atom.predicate.name
                code = {"static": 0, "fluent": 1, "derived": 2}
                assert code[kinds[a["predicate"][l]]] == a["kind"][l] == code[lit.atom.kind]
                assert bool(a["polarity"][l]) == lit.positive == c.literals[j].positive
                arity = a["arity"][l]
                assert (a["terms"][l, arity:] == -(2**31)).all()
                assert (a["terms"][l, :arity] >= 0).all()  # generated templates hold variables only


def test_binding_order_is_the_order_of_task_bindings():
    for name in ["gripper__prob05", "logistics00__probLOGISTICS-6-1", "depot__p02"]:
        task = text_task(name)
        t = rl.lifted_goal_templates(task, max_subgoals_per_size=3)
        for c in t.conditions(0)[:8]:
            order = c.binding_order(task)
            for s in walk(task, steps=15, walks=1)[::3]:
                found = [tuple(o.index for o in b) for b in task.bindings(c, s, limit=200)]
                keys = [tuple(b[v] for v in order) for b in found]
                assert keys == sorted(keys)
                first = list(task.bindings(c, s, limit=1))
                assert [tuple(o.index for o in b) for b in first] == found[:1]
    # with fixed variables, the order lists the free ones
    task = text_task("gripper__prob05")
    c = task.precondition(0)
    assert sorted(c.binding_order(task)) == list(range(c.arity))
    assert 0 not in c.binding_order(task, partial={0: 0})
    with pytest.raises(ValueError, match="another task"):
        c.binding_order(text_task("blocks__probBLOCKS-8-0"))


# ------------------------------------------------------------------------------------------------ conditions


def test_conditions_round_trip():
    tasks = [text_task("gripper__prob05"), text_task("gripper__prob05")]
    table = rl.TaskTable(tasks)
    t = rl.lifted_goal_templates(table, max_subgoals_per_size=3)
    back = rl.GoalTemplates.from_conditions(table, [t.conditions(i) for i in range(len(tasks))])
    assert back.options is None and len(back) == len(t)
    for i in range(len(tasks)):
        assert back.conditions(i) == t.conditions(i)
        assert back.goal_literals(i) == [()] * len(t.conditions(i))
    for k in ["instance_offsets", "template_size", "num_vars", "distinct", "literal_offsets", "order", "predicate",
              "polarity", "kind", "arity", "terms"]:
        np.testing.assert_array_equal(back.arrays()[k], t.arrays()[k])
    assert (back.arrays()["goal_literal"] == -1).all()


def test_from_conditions_takes_constants_and_mixed_distinctness():
    task = text_task("gripper__prob05")
    lits = task.goal_condition.literals
    ground = task.ground_condition([lits[0], lits[1]])
    plain, distinct = ground.lift(), ground.lift(True)
    t = rl.GoalTemplates.from_conditions(task, [[distinct, plain]])
    assert t.arrays()["distinct"].tolist() == [1, 0]
    assert t.conditions(0) == [distinct, plain]
    # an untyped precondition is a condition like any other
    pre = task.precondition(0)
    assert rl.GoalTemplates.from_conditions(task, [[pre]]).conditions(0) == [pre]


@needs_frontend
def test_from_conditions_rejects_typed_variables():
    task = ipc("satellite", "test", "p01-easy")
    with pytest.raises(ValueError, match="typed variables"):
        rl.GoalTemplates.from_conditions(task, [[task.precondition(0)]])


def test_from_conditions_rejects_invalid_input():
    task = text_task("gripper__prob05")
    other = text_task("blocks__probBLOCKS-8-0")
    c = task.ground_condition(list(task.goal_condition.literals[:2])).lift(True)
    with pytest.raises(ValueError, match="conditions for 2 instances"):
        rl.GoalTemplates.from_conditions(task, [[c], [c]])
    with pytest.raises(ValueError, match="another task"):
        rl.GoalTemplates.from_conditions(task, [[other.ground_condition([other.goal_condition.literals[0]]).lift()]])
    with pytest.raises(TypeError):
        rl.GoalTemplates.from_conditions(task, [["(at ball1 rooma)"]])


@needs_frontend
def test_numeric_goal_constraints_are_skipped():
    domain = mymyr.Domain.from_string(
        "(define (domain n) (:requirements :strips :typing :numeric-fluents) (:types item)"
        " (:predicates (ok ?x - item) (near ?x ?y - item)) (:functions (fuel))"
        " (:action fix :parameters (?x - item) :precondition (>= (fuel) 1)"
        "  :effect (and (ok ?x) (decrease (fuel) 1))))"
    )
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain n) (:objects a b c - item) (:init (= (fuel) 5) (near a b))"
        " (:goal (and (ok a) (ok b) (>= (fuel) 2) (ok c))))"
    ))
    assert len(task.goal_condition.numeric_constraints) == 1
    t = rl.lifted_goal_templates(task)
    check_reference(t, 0, task)
    for c in t.conditions(0):
        assert c.numeric_constraints == ()
    # (ok a), (ok b), (ok c) share no object: three components, joined by the component draws
    assert max(sizes(t, 0)) == 3
    assert all(i < 3 for g in t.goal_literals(0) for i in g)


@needs_frontend
def test_equality_mode_follows_the_domain_requirements():
    domain = mymyr.Domain.from_string(
        "(define (domain e) (:requirements :strips :typing :equality) (:types item)"
        " (:predicates (on ?x ?y - item)) (:action put :parameters (?x ?y - item) :precondition (and)"
        "  :effect (on ?x ?y)))"
    )
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain e) (:objects a b c - item) (:init) (:goal (and (on a b) (on b c))))"
    ))
    t = rl.lifted_goal_templates(task, distinct="equality")
    assert t.arrays()["distinct"].tolist() == [1] * len(t)
    check_reference(t, 0, task, distinct="equality")
    assert t.options["distinct"] == "equality"
    assert rl.lifted_goal_templates(task, distinct=False).arrays()["distinct"].tolist() == [0] * len(t)


# ------------------------------------------------------------------------------------------------ pickling and errors


def test_pickling():
    table = rl.TaskTable([text_task("gripper__prob05"), text_task("gripper__prob05")])
    for source in [table, text_task("logistics00__probLOGISTICS-6-1")]:
        t = rl.lifted_goal_templates(source, seed=6, distinct=False, max_size=4)
        back = pickle.loads(pickle.dumps(t))
        assert back.options == t.options and len(back) == len(t) and back.num_instances == t.num_instances
        for k, v in t.arrays().items():
            np.testing.assert_array_equal(back.arrays()[k], v)
        for i in range(t.num_instances):
            assert [str(c) for c in back.conditions(i)] == [str(c) for c in t.conditions(i)]
            assert back.goal_literals(i) == t.goal_literals(i)
    fixed = rl.GoalTemplates.from_conditions(table, [rl.lifted_goal_templates(table).conditions(0)[:2], []])
    back = pickle.loads(pickle.dumps(fixed))
    assert back.options is None and back.goal_literals(0) == [(), ()]


def test_invalid_options_and_lazy_tasks():
    task = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="max_size"):
        rl.lifted_goal_templates(task, max_size=0)
    with pytest.raises(ValueError, match="distinct"):
        rl.lifted_goal_templates(task, distinct="all")
    with pytest.raises(ValueError, match="max_component_samples"):
        rl.lifted_goal_templates(task, max_component_samples=2**31, max_size=4)
    with pytest.raises(ValueError, match="lazy atom slots"):
        rl.lifted_goal_templates(text_task("gripper__prob05", atoms="lazy"))
    t = rl.lifted_goal_templates(task)
    with pytest.raises(IndexError):
        t.conditions(1)
    with pytest.raises(ValueError, match="not an instance"):
        t.conditions(text_task("gripper__prob05"))
