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


# ------------------------------------------------------------------------------------------------ matching


def match_rows(tasks, states):
    """Rows and closed derived words, padded to the suite width, with a (task, state) pair per row."""
    source = rl.TaskSuite.group(tasks)
    rows = np.zeros((len(states), source.row_words), np.uint64)
    derived = np.zeros((len(states), source.derived_words), np.uint64)
    ids = np.array([g for g, _ in states], np.int32)
    for r, (g, state) in enumerate(states):
        task = tasks[g]
        encoded = np.asarray(task.encode([state]))[0]
        rows[r, :len(encoded)] = encoded
        for atom in task.derived_atoms(state):
            slot = atom.derived_slot
            derived[r, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return source, rows, ids, derived


def assert_match_oracle(tasks, states, templates, result, *, first=False):
    offsets = templates.arrays()["instance_offsets"]
    for row, (g, state) in enumerate(states):
        matches = []
        for local, condition in enumerate(templates.conditions(g)):
            bindings = list(tasks[g].bindings(condition, state, limit=1))
            at = int(offsets[g]) + local
            if bindings:
                matches.append((at, tuple(o.index for o in bindings[0])))
            if not first:
                assert bool(result.found[row, local]) == bool(bindings)
                assert result.template[row, local] == (at if bindings else -1)
                if bindings:
                    want = tuple(o.index for o in bindings[0])
                    assert tuple(result.binding[row, local, :len(want)]) == want
                    assert (result.binding[row, local, len(want):] == -1).all()
                else:
                    assert (result.binding[row, local] == -1).all()
        if first:
            assert bool(result.found[row]) == bool(matches)
            assert result.template[row] == (matches[0][0] if matches else -1)
            if matches:
                want = matches[0][1]
                assert tuple(result.binding[row, :len(want)]) == want
                assert (result.binding[row, len(want):] == -1).all()
            else:
                assert (result.binding[row] == -1).all()
        assert result.count[row] == (int(bool(matches)) if first else len(matches))
    assert int(result.status) == 0


@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("distinct", [True, False, "equality"])
def test_match_first_equals_bindings_on_ipc_walks(domain, distinct):
    tasks = [ipc(domain, "train", "p01"), ipc(domain, "train", "p02")]
    states = [(g, s) for g, task in enumerate(tasks) for s in walk(task, steps=12, seed=7, walks=1)]
    source, rows, ids, derived = match_rows(tasks, states)
    templates = rl.lifted_goal_templates(source, distinct=distinct)
    for first in (False, True):
        result = rl.match_first(rows, ids, templates, first=first, derived=derived)
        assert_match_oracle(tasks, states, templates, result, first=first)


@pytest.mark.parametrize("name", TEXT)
def test_match_first_equals_bindings_on_text_fixtures(name):
    task = text_task(name)
    states = [(0, s) for s in walk(task, steps=15, walks=1)]
    source, rows, ids, derived = match_rows([task], states)
    templates = rl.lifted_goal_templates(source)
    for first in (False, True):
        result = rl.match_first(rows, ids, templates, first=first, derived=derived)
        assert_match_oracle([task], states, templates, result, first=first)


def match_literal_case(atoms="dense"):
    from test_encoding_literals import goals_task

    task = goals_task("static_true", atoms=atoms)
    conditions = [
        task.condition(["?x", "?y"], ["(road ?x ?y)", "(not (lit ?y))"]),
        task.condition(["?x"], ["(road ?x ?x)"]),
        task.condition(["?x"], ["(not (lit ?x))", "(not (heavy ?x))"]),
        task.condition(["?x"], ["(road ?x a)", "(not (lit ?x))", "(heavy t1)"]),
        task.condition([], ["(not (heavy t2))"]),
        task.condition([], ["(heavy t2)"]),
        task.condition([], ["(done)"]),
        task.condition([], ["(not (done))"]),
        task.condition(["?unused"], ["(not (done))"]),
        task.condition(["?x", "?y"], ["(seen ?x)", "(seen ?y)"], equalities=[("?x", "?y", False)]),
        task.condition(["?x", "?y"], ["(seen ?x)", "(seen ?y)"]),
        task.condition([], ["(lit t1)"]),
        task.condition([], ["(not (lit t1))"]),
    ]
    states = [(0, s) for s in walk(task, steps=35, seed=12, walks=1)]
    source, rows, ids, derived = match_rows([task], states)
    templates = rl.GoalTemplates.from_conditions(source, [conditions])
    return [task], states, rows, ids, derived, templates


@needs_frontend
@pytest.mark.parametrize("atoms", ["reachable", "dense"])
def test_match_first_static_negative_derived_constants_repeated_and_unused_variables(atoms):
    tasks, states, rows, ids, derived, templates = match_literal_case(atoms)
    for first in (False, True):
        result = rl.match_first(rows, ids, templates, first=first, derived=derived)
        assert_match_oracle(tasks, states, templates, result, first=first)
    with pytest.raises(ValueError, match="derived"):
        rl.match_first(rows, ids, templates)
    with pytest.raises(ValueError, match="derived"):
        rl.match_first(rows, ids, templates, derived=derived[:, :0].copy())


def test_match_first_empty_padding_truncation_and_invalid_ids():
    tasks = [text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")]
    source, rows, ids, derived = match_rows(tasks, [(0, tasks[0].initial_state), (1, tasks[1].initial_state)])
    full = rl.lifted_goal_templates(source)
    templates = rl.GoalTemplates.from_conditions(source, [full.conditions(0), []])
    result = rl.match_first(rows, ids, templates)
    assert not result.found[1].any() and (result.binding[1] == -1).all()
    assert result.count[1] == 0 and (result.template[1] == -1).all()
    invalid = rl.match_first(rows, np.array([-1, 2], np.int32), templates)
    assert int(invalid.status) == 4 and not invalid.found.any() and not invalid.count.any()
    empty = rl.match_first(rows[:0], ids[:0], templates)
    assert empty.count.shape == (0,) and int(empty.status) == 0
    # A fixed true conjunction gives a match in every row, including a capacity of zero.
    lit = str(next(iter(tasks[0].initial_state.atoms())))
    hit = tasks[0].condition([], [lit])
    templates = rl.GoalTemplates.from_conditions(tasks[0], [[hit, hit]])
    row = np.asarray(tasks[0].encode([tasks[0].initial_state]))
    status = np.array(99, np.int64)
    out = rl.match_first(row, None, templates, capacity=1, status=status)
    assert out.count.tolist() == [2] and int(status) == 1 and out.found.tolist() == [[True]]
    out = rl.match_first(row, None, templates, capacity=0)
    assert out.found.shape == (1, 0) and out.count.tolist() == [2] and int(out.status) == 1
    out = rl.match_first(row, None, templates, first=True, status=status)
    assert out.count.tolist() == [1] and out.template.tolist() == [0] and int(status) == 0
    none = rl.GoalTemplates.from_conditions(tasks[0], [[]])
    out = rl.match_first(row, None, none, first=True)
    assert out.binding.shape == (1, 0) and out.template.tolist() == [-1] and not out.found.any()
    with pytest.raises(ValueError, match="narrower"):
        rl.match_first(rows[:, :0].copy(), ids, full)
    with pytest.raises(ValueError, match="task_ids"):
        rl.match_first(rows, None, full)
    for arg, value in (("task_ids", (0, 1)), ("derived", (0,)), ("status", ())):
        given = {"task_ids": ids, arg: value}
        with pytest.raises(TypeError, match=f"'{arg}' must be an array that speaks DLPack"):
            rl.match_first(rows, given.pop("task_ids"), full, **given)
    with pytest.raises(ValueError, match="capacity"):
        rl.match_first(row, None, templates, first=True, capacity=0)


def cuda_match_runtime():
    torch = pytest.importorskip("torch", exc_type=ImportError)
    cuda = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not torch.cuda.is_available() or not cuda.available():
        pytest.skip("no CUDA device")
    return torch


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("via_op", [False, True])
def test_match_first_device_suite_and_capture(first, via_op):
    torch = cuda_match_runtime()
    tasks = [text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")]
    states = [(g, s) for g, task in enumerate(tasks) for s in walk(task, steps=12, walks=1)]
    source, rows, ids, derived = match_rows(tasks, states)
    templates = rl.lifted_goal_templates(source)
    expected = rl.match_first(rows, ids, templates, first=first)
    with torch.cuda.device(0):
        device = templates.to("cuda:0")
        x = torch.tensor(rows.view(np.int64), device="cuda:0")
        task_ids = torch.tensor(ids, device="cuda:0")
        from mymyr.rl.torch import register, unregister
        handle = register(device) if via_op else None

        def run():
            if via_op:
                return rl.MatchResult(*torch.ops.mymyr.match_first(handle, x, task_ids, first))
            return rl.match_first(x, task_ids, device, first=first)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = run()
        stream.synchronize()
        for host, gpu in zip(expected, result):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        graph.replay()
        for host, gpu in zip(expected, captured):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())
        task_ids.fill_(-1)
        graph.replay()
        assert captured.status.item() == 4 and not captured.found.any().item()
        task_ids.copy_(torch.tensor(ids, device="cuda:0"))
        x.zero_()
        changed = rl.match_first(np.zeros_like(rows), ids, templates, first=first)
        graph.replay()
        for host, gpu in zip(changed, captured):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())
        assert device.nbytes == templates.nbytes
        if handle is not None:
            unregister(handle)


@needs_frontend
@pytest.mark.parametrize("first", [False, True])
def test_match_first_device_literal_kinds(first):
    torch = cuda_match_runtime()
    tasks, states, rows, ids, derived, templates = match_literal_case()
    expected = rl.match_first(rows, ids, templates, first=first, derived=derived)
    device = templates.upload(torch.cuda.current_stream(0))
    result = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"),
                            torch.tensor(ids, device="cuda:0"), device, first=first,
                            derived=torch.tensor(derived.view(np.int64), device="cuda:0"))
    for host, gpu in zip(expected, result):
        np.testing.assert_array_equal(host, gpu.cpu().numpy())
    small = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None, device,
                           derived=torch.tensor(derived.view(np.int64), device="cuda:0"), capacity=0)
    assert small.status.item() == 1
    np.testing.assert_array_equal(expected.count if not first else
                                  rl.match_first(rows, ids, templates, derived=derived).count,
                                  small.count.cpu().numpy())


@pytest.mark.needs_gil
@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("max_steps", [0, 1_000_000])
def test_match_first_torch_op_fake_and_compile(first, max_steps):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from mymyr.rl.torch._ops import match_first as op, register, unregister
    from torch._subclasses.fake_tensor import FakeTensorMode

    task = text_task("gripper__prob05")
    templates = rl.lifted_goal_templates(task)
    rows = torch.tensor(np.asarray(task.encode(walk(task, steps=3, walks=1))).view(np.int64))
    handle = register(templates)
    try:
        expected = rl.match_first(rows.numpy(), None, templates, first=first, max_steps=max_steps)
        eager = op(handle, rows, first=first, max_steps=max_steps)
        with FakeTensorMode() as mode:
            fake = op(handle, mode.from_tensor(rows), first=first, max_steps=max_steps)
            assert [(v.shape, v.dtype) for v in fake] == [(v.shape, v.dtype) for v in eager]
        compiled = torch.compile(lambda x: op(handle, x, first=first, max_steps=max_steps), backend="eager", fullgraph=True)(rows)
        for host, eager_tensor, compiled_tensor in zip(expected, eager, compiled):
            np.testing.assert_array_equal(host, eager_tensor.numpy())
            assert torch.equal(eager_tensor, compiled_tensor)
    finally:
        unregister(handle)


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
def test_match_first_problem_local_derived_goals(on_device):
    from test_encoding_literals import goals_task

    tasks = [goals_task("duplicate"), goals_task("static_true")]
    source, rows, ids, derived = match_rows(tasks, [(g, s) for g, task in enumerate(tasks)
                                                  for s in walk(task, steps=9, walks=1)])
    templates = rl.GoalTemplates.from_conditions(source, [
        [tasks[0].condition([], ["(axiom_0)"]), tasks[0].condition([], ["(not (axiom_0))"])],
        [tasks[1].condition([], ["(done)"]), tasks[1].condition([], ["(not (done))"])],
    ])
    expected = rl.match_first(rows, ids, templates, derived=derived)
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"),
                                 torch.tensor(ids, device="cuda:0"), templates.to("cuda:0"),
                                 derived=torch.tensor(derived.view(np.int64), device="cuda:0"))
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())
    else:
        states = [(g, s) for g, task in enumerate(tasks) for s in walk(task, steps=9, walks=1)]
        assert_match_oracle(tasks, states, templates, expected)


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
def test_match_first_candidates_span_multiple_object_tiles(on_device):
    domain = mymyr.Domain.from_string("""(define (domain tiles) (:requirements :strips)
      (:predicates (p ?x) (edge ?x ?y))
      (:action set :parameters (?x) :precondition (and) :effect (p ?x)))""")
    objects = " ".join(f"o{i:02}" for i in range(70))
    problem = f"""(define (problem tiles) (:domain tiles) (:objects {objects})
      (:init (p o40) (p o65) (edge o40 o65)) (:goal (and (p o40) (p o65))))"""
    task = mymyr.Task(domain.instantiate_string(problem))
    conditions = [task.condition(["?x", "?y"], ["(p ?x)", "(p ?y)", "(edge ?x ?y)"]),
                  task.condition(["?x", "?y"], ["(p ?x)", "(p ?y)"], equalities=[("?x", "?y", False)])]
    templates = rl.GoalTemplates.from_conditions(task, [conditions])
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates)
    assert_match_oracle([task], [(0, task.initial_state)], templates, expected)
    assert expected.found.all() and (expected.binding >= 32).all()
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None, templates.to("cuda:0"))
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


def test_match_first_thread_pool_matches_serial_with_status():
    task = text_task("gripper__prob05")
    templates = rl.lifted_goal_templates(task)
    rows = np.tile(np.asarray(task.encode(walk(task, steps=7, walks=1))), (32, 1))
    ids = np.zeros(len(rows), np.int32)
    ids[::5] = -1
    pool = rl.ThreadPool(4)
    for first, capacity in [(True, None), (False, 1), (False, 0)]:
        expected = rl.match_first(rows, ids, templates, first=first, capacity=capacity)
        actual = rl.match_first(rows, ids, templates, first=first, capacity=capacity, pool=pool)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        assert int(actual.status) & 4


@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("distinct", [True, False, "equality"])
def test_match_first_device_equals_host_on_ipc_walks(domain, distinct):
    torch = cuda_match_runtime()
    tasks = [ipc(domain, "train", "p01"), ipc(domain, "train", "p02")]
    states = [(g, s) for g, task in enumerate(tasks) for s in walk(task, steps=12, seed=7, walks=1)]
    source, rows, ids, derived = match_rows(tasks, states)
    templates = rl.lifted_goal_templates(source, distinct=distinct)
    uploaded = templates.to("cuda:0")
    x = torch.tensor(rows.view(np.int64), device="cuda:0")
    task_ids = torch.tensor(ids, device="cuda:0")
    d = torch.tensor(derived.view(np.int64), device="cuda:0")
    for first in (False, True):
        expected = rl.match_first(rows, ids, templates, first=first, derived=derived)
        actual = rl.match_first(x, task_ids, uploaded, first=first, derived=d)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


def test_match_first_device_variable_limit():
    torch = cuda_match_runtime()
    task = text_task("gripper__prob05")
    literal = str(next(iter(task.initial_state.atoms())))
    rows = np.asarray(task.encode([task.initial_state]))
    x = torch.tensor(rows.view(np.int64), device="cuda:0")
    for variables in (64, 65):
        condition = task.condition([f"?x{v}" for v in range(variables)], [literal])
        templates = rl.GoalTemplates.from_conditions(task, [[condition]])
        expected = rl.match_first(rows, None, templates, first=True)
        assert expected.found.all() and expected.binding.shape == (1, variables)
        if variables == 65:
            with pytest.raises(ValueError, match="64 variables"):
                templates.to("cuda:0")
        else:
            actual = rl.match_first(x, None, templates.to("cuda:0"), first=True)
            for host, gpu in zip(expected, actual):
                np.testing.assert_array_equal(host, gpu.cpu().numpy())


@pytest.mark.parametrize("on_device", [False, True])
def test_match_first_word_halves(on_device):
    task = text_task("gripper__prob05")
    templates = rl.lifted_goal_templates(task)
    rows = np.asarray(task.encode(walk(task, steps=5, walks=1)))
    expected = rl.match_first(rows, None, templates)
    halves = rows.view(np.uint32)
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(halves, device="cuda:0"), None, templates.to("cuda:0"))
        actual = [v.cpu().numpy() for v in actual]
    else:
        actual = rl.match_first(halves, None, templates)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_match_first_upload_accepts_stream_handles_and_stream_protocol():
    torch = cuda_match_runtime()
    task = text_task("gripper__prob05")
    templates = rl.lifted_goal_templates(task)
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates, first=True)
    x = torch.tensor(rows.view(np.int64), device="cuda:0")
    stream = torch.cuda.Stream(device=0)

    class StreamReference:
        cuda_stream = stream.cuda_stream

    for source in (stream, stream.cuda_stream, StreamReference()):
        uploaded = templates.upload(source)
        actual = rl.match_first(x, None, uploaded, first=True)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("typed", ["none", "static", "fluent", "binary", "binary_pairs"])
@pytest.mark.parametrize("first", [False, True])
def test_match_first_rejects_impossible_injective_domains(on_device, typed, first):
    limited_type = typed != "none"
    action = "(:action add :parameters (?x) :precondition (and) :effect (member ?x))" if typed == "fluent" else ""
    domain = mymyr.Domain.from_string(f"""(define (domain domains)
        (:requirements :adl :equality) (:predicates (ok) (member ?x) (pair ?x ?y))
        (:action set :parameters () :precondition (and) :effect (ok)) {action})""")
    objects = " ".join(f"o{i}" for i in range(70 if limited_type else 63))
    members = " ".join(f"(member o{i}) (pair o{i} o{i})" for i in range(63))
    if typed == "binary_pairs":
        members = " ".join(f"(pair o{i} o{j})" for i in range(63) for j in range(63))
    task = mymyr.Task(domain.instantiate_string(f"""(define (problem domains) (:domain domains)
        (:objects {objects}) (:init (ok) {members}) (:goal (ok)))"""))
    variables = [f"?v{i}" for i in range(64)]
    literals = ([f"(pair {a} {b})" for a, b in zip(variables[::2], variables[1::2])] if typed == "binary_pairs" else
                [f"(pair {v} {v})" for v in variables] if typed == "binary" else
                [f"(member {v})" for v in variables] if limited_type else ["(ok)"])
    budget = 512 if typed == "binary_pairs" else 3 if limited_type else 0
    condition = task.condition(variables, literals,
                               equalities=[(a, b, False) for i, a in enumerate(variables) for b in variables[i + 1:]])
    templates = rl.GoalTemplates.from_conditions(task, [[condition]])
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates, first=first, max_steps=budget)
    assert not expected.found.any() and expected.count.tolist() == [0]
    assert (expected.binding == -1).all() and (expected.template == -1).all()
    assert int(expected.status) == 0
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None,
                                templates.to("cuda:0"), first=first, max_steps=budget)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_match_first_search_budget_exhaustion_and_reset(on_device, first):
    domain = mymyr.Domain.from_string("""(define (domain budget)
        (:requirements :adl :equality) (:predicates (ok))
        (:action set :parameters () :precondition (and) :effect (ok)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem budget) (:domain budget)
        (:objects a b c d) (:init (ok)) (:goal (ok)))"""))
    variables = ["?a", "?b", "?c", "?d"]
    condition = task.condition(variables, ["(ok)"],
                               equalities=[(a, b, False) for i, a in enumerate(variables) for b in variables[i + 1:]])
    templates = rl.GoalTemplates.from_conditions(task, [[condition, condition]])
    rows = np.asarray(task.encode([task.initial_state] * 2))
    status = np.array(99, np.int64)
    expected = rl.match_first(rows, None, templates, first=first, max_steps=7, status=status, pool=rl.ThreadPool(2))
    assert int(expected.status) == 8 and not expected.found.any() and not expected.count.any()
    assert (expected.binding == -1).all() and (expected.template == -1).all()
    expected = rl.MatchResult(*(value.copy() for value in expected))
    complete = rl.match_first(rows, None, templates, first=first, max_steps=8, status=status)
    assert int(complete.status) == 0 and complete.found.all()
    assert complete.count.tolist() == ([1, 1] if first else [2, 2])
    unbounded = rl.match_first(rows, None, templates, first=first, max_steps=None)
    for want, have in zip(complete, unbounded):
        np.testing.assert_array_equal(want, have)
    if on_device:
        torch = cuda_match_runtime()
        from mymyr.rl.torch import register, unregister

        device = templates.to("cuda:0")
        x = torch.tensor(rows.view(np.int64), device="cuda:0")
        device_status = torch.tensor(99, device="cuda:0")
        handle = register(device)
        try:
            for budget, want in [(7, expected), (8, complete), (None, complete)]:
                actual = rl.match_first(x, None, device, first=first, max_steps=budget, status=device_status)
                via_op = torch.ops.mymyr.match_first(handle, x, first=first, max_steps=budget)
                for host, gpu, op in zip(want, actual, via_op):
                    np.testing.assert_array_equal(host, gpu.cpu().numpy())
                    assert torch.equal(gpu, op)
        finally:
            unregister(handle)


@pytest.mark.parametrize("max_steps", [-1, 2**63, 1.5])
def test_match_first_rejects_invalid_search_budget(max_steps):
    task = text_task("gripper__prob05")
    templates = rl.lifted_goal_templates(task)
    rows = np.asarray(task.encode([task.initial_state]))
    with pytest.raises((TypeError, ValueError)):
        rl.match_first(rows, None, templates, max_steps=max_steps)


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_match_first_exhausted_template_continues_and_combines_status(on_device, first):
    domain = mymyr.Domain.from_string("""(define (domain continued)
        (:requirements :adl) (:predicates (ok))
        (:action set :parameters () :precondition (and) :effect (ok)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem continued) (:domain continued)
        (:objects a b) (:init (ok)) (:goal (ok)))"""))
    expensive = task.condition(["?a", "?b"], ["(ok)"])
    ground = task.condition([], ["(ok)"])
    templates = rl.GoalTemplates.from_conditions(task, [[expensive, ground]])
    rows = np.asarray(task.encode([task.initial_state] * 2))
    ids = np.array([0, -1], np.int32)
    expected = rl.match_first(rows, ids, templates, first=first, capacity=1, max_steps=1)
    assert int(expected.status) == (12 if first else 13)
    assert expected.count.tolist() == [1, 0]
    assert expected.found.tolist() == ([True, False] if first else [[False], [False]])
    if first:
        assert expected.template.tolist() == [1, -1]
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"),
                                torch.tensor(ids, device="cuda:0"), templates.to("cuda:0"),
                                first=first, capacity=1, max_steps=1)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_match_first_shared_type_across_predicates_allows_object_reuse(on_device, first):
    domain = mymyr.Domain.from_string("""(define (domain shared)
        (:requirements :adl :equality) (:predicates (ok) (a ?x) (b ?x) (c ?x))
        (:action set :parameters () :precondition (and) :effect (ok)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem shared) (:domain shared)
        (:objects o0 o1 o2 o3 o4 o5 o6 o7 o8 o9)
        (:init (ok) (a o0) (a o1) (b o0) (b o1) (c o0) (c o1)) (:goal (ok)))"""))
    variables = ["?x", "?y", "?z"]
    literals = ["(a ?x)", "(b ?y)", "(c ?z)"]
    impossible = task.condition(variables, literals, equalities=[("?x", "?y", False), ("?x", "?z", False),
                                                                ("?y", "?z", False)])
    reusable = task.condition(variables, literals)
    templates = rl.GoalTemplates.from_conditions(task, [[impossible, reusable]])
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates, first=first, max_steps=6)
    assert int(expected.status) == 0 and expected.count.tolist() == [1]
    assert expected.found.tolist() == ([True] if first else [[False, True]])
    assert expected.template.tolist() == ([1] if first else [[-1, 1]])
    assert (expected.binding[0] if first else expected.binding[0, 1]).tolist() == [0, 0, 0]
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None,
                                templates.to("cuda:0"), first=first, max_steps=6)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_match_first_projection_filters_by_bound_arguments(on_device, first):
    # Nine distinct tiles of one colour, with eight tiles of each colour: the tiles of all colours together would
    # suffice, so only the projection filtered by the bound colour rejects the search without enumerating tiles.
    domain = mymyr.Domain.from_string("""(define (domain paint)
        (:requirements :adl :equality) (:predicates (ok) (painted ?t ?c))
        (:action set :parameters () :precondition (and) :effect (ok)))""")
    tiles = [f"t{i}" for i in range(16)]
    painted = " ".join(f"(painted {t} {'white' if i < 8 else 'black'})" for i, t in enumerate(tiles))
    task = mymyr.Task(domain.instantiate_string(f"""(define (problem paint) (:domain paint)
        (:objects white black {' '.join(tiles)}) (:init (ok) {painted}) (:goal (ok)))"""))
    nine = [f"?t{i}" for i in range(9)]
    variables = ["?c", *nine]
    same = task.condition(variables, [f"(painted {t} ?c)" for t in nine],
                          equalities=[(a, b, False) for i, a in enumerate(variables) for b in variables[i + 1:]])
    eight = task.condition(variables[:9], [f"(painted {t} ?c)" for t in nine[:8]],
                           equalities=[(a, b, False) for i, a in enumerate(variables[:9]) for b in variables[i + 1:9]])
    templates = rl.GoalTemplates.from_conditions(task, [[same, eight]])
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates, first=first, max_steps=1000)
    assert int(expected.status) == 0 and expected.count.tolist() == [1]
    assert expected.template.tolist() == ([1] if first else [[-1, 1]])
    canonical = next(iter(task.bindings(eight, task.initial_state, limit=1)))
    bound = expected.binding[0] if first else expected.binding[0, 1]
    assert bound.tolist() == [o.index for o in canonical] + [-1]
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None,
                                templates.to("cuda:0"), first=first, max_steps=1000)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("max_steps", [2, 3])
def test_match_first_pair_budget_with_small_and_large_template_tables(on_device, padding, max_steps):
    domain = mymyr.Domain.from_string("""(define (domain pairs)
        (:requirements :adl :equality) (:predicates (ok) (member ?x))
        (:action set :parameters () :precondition (and) :effect (ok)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem pairs) (:domain pairs)
        (:objects a b c) (:init (ok) (member a)) (:goal (ok)))"""))
    pair = task.condition(["?x", "?y"], ["(member ?x)", "(member ?y)"], equalities=[("?x", "?y", False)])
    conditions = [pair]
    if padding:
        conditions.append(task.condition(["?x", "?y", "?z"], ["(not (ok))"]))
    templates = rl.GoalTemplates.from_conditions(task, [conditions])
    rows = np.asarray(task.encode([task.initial_state]))
    expected = rl.match_first(rows, None, templates, capacity=1, max_steps=max_steps)
    assert int(expected.status) == (8 if max_steps == 2 else 0)
    assert not expected.found.any() and expected.count.tolist() == [0]
    assert (expected.binding == -1).all() and expected.template.tolist() == [[-1]]
    if on_device:
        torch = cuda_match_runtime()
        actual = rl.match_first(torch.tensor(rows.view(np.int64), device="cuda:0"), None,
                                templates.to("cuda:0"), capacity=1, max_steps=max_steps)
        for host, gpu in zip(expected, actual):
            np.testing.assert_array_equal(host, gpu.cpu().numpy())


# ------------------------------------------------------------------------------------------------ relabelling


def assert_lifted_parity(expected, actual):
    """Compare words as bit patterns across NumPy uint64 and torch int64 storage."""
    for name, host, tensor in zip(expected._fields, expected, actual):
        value = tensor.cpu().numpy()
        if host.dtype == np.uint64:
            value = value.view(np.uint64)
        np.testing.assert_array_equal(host, value, err_msg=name)


def lifted_window(task, *, n=3, steps=23, seed=13, scripted=False):
    from test_rl_ops import trajectory
    if scripted:
        env = _rl_torch.Env(task, seed=seed, max_steps=4)
        current = np.tile(env.initial_states()[0], (n, 1))
        step_count, draws = np.zeros(n, np.int32), np.zeros(n, np.uint64)
        prev, reached, done = [], [], []
        for q in range(steps):
            final = np.zeros_like(current)
            term, trunc = np.zeros(n, bool), np.zeros(n, bool)
            prev.append(current.copy())
            action = np.full(n, 2**31 - 1, np.int64) if q % 4 == 1 else None
            env.step(current, steps=step_count, draws=draws, action=action,
                     final_states=final, terminated=term, truncated=trunc)
            reached.append(final)
            done.append(term | trunc)
        prev, reached, done = np.stack(prev), np.stack(reached), np.stack(done)
        np.testing.assert_array_equal(prev[1::4], reached[1::4])
        assert done.any()
    else:
        prev, reached, done = trajectory(task, n=n, steps=steps, seed=seed, max_steps=7)
    previous = [task.decode(row) for row in prev]
    states = [task.decode(row) for row in reached]
    _, _, _, pd = match_rows([task], [(0, s) for row in previous for s in row])
    _, _, _, d = match_rows([task], [(0, s) for row in states for s in row])
    return prev, reached, done, previous, states, pd.reshape(steps, n, -1), d.reshape(steps, n, -1)


def assert_refine(tasks, previous, states, done, ids, templates, result, *, max_generated=None,
                  step_reward=-0.25, goal_reward=2.0):
    from her_lifted_reference import refine

    T, N = done.shape
    S = result.seg_valid.shape[1]
    W, D = result.goal_pos.shape[-1], result.goal_pos_derived.shape[-1]
    expected = [np.zeros_like(x) for x in result]
    for at in (0, 3, 4, 6, 7):
        expected[at].fill(-1)
    remaining = 2**63 if max_generated is None else max_generated
    overflow = False
    offsets = templates.arrays()["instance_offsets"]
    for i in range(N):
        emitted = []
        first = 0
        for last in range(T):
            if not done[last, i] and last + 1 < T:
                continue
            g = int(ids[i] if ids.ndim == 1 else ids[last, i])
            transitions = [(previous[t][i], states[t][i]) for t in range(first, last + 1)]
            for start, end, k, binding, goal in refine(tasks[g], transitions, templates.conditions(g)):
                emitted.append((start + first, end + first, g, int(offsets[g]) + k, binding, goal))
            first = last + 1
        allowed = min(remaining, len(emitted))
        remaining -= allowed
        overflow |= allowed > S
        keep = min(S, allowed)
        expected[12][i] = keep
        for j, (start, end, g, k, binding, goal) in enumerate(emitted[:keep]):
            expected[0][start:end+1, i] = j
            expected[1][start:end+1, i] = step_reward
            expected[1][end, i] = goal_reward
            expected[2][end, i] = True
            expected[3][i, j], expected[4][i, j] = start, end
            expected[5][i, j], expected[6][i, j] = True, k
            expected[7][i, j, :len(binding)] = binding
            for literals, width, positive, negative, slot_name in [
                (goal.fluent_literals, W, 8, 9, "slot"),
                (goal.derived_literals, D, 10, 11, "derived_slot"),
            ]:
                for literal in literals:
                    slot = getattr(literal.atom, slot_name)
                    which = positive if literal.positive else negative
                    mask = expected[which].view(np.uint64)
                    mask[i, j, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    expected[13][...] = int(overflow)
    for name, actual, want in zip(result._fields, result, expected):
        np.testing.assert_array_equal(actual, want, err_msg=name)


@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("distinct", [True, "equality"])
@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_equals_golden_refine_on_ipc_windows(domain, distinct, on_device):
    task = ipc(domain, "train", "p01")
    prev, reached, done, previous, states, pd, d = lifted_window(task)
    t = rl.lifted_goal_templates(task, distinct=distinct)
    result = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=23,
                                    derived=d, prev_derived=pd, step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(3, np.int32), t, result)
    if on_device:
        torch = cuda_match_runtime()
        x, p, flags, dd, pp = [torch.tensor(a, device="cuda:0") for a in
                               (reached.view(np.int64), prev.view(np.int64), done, d.view(np.int64), pd.view(np.int64))]
        actual = rl.her_relabel_lifted(x, flags, None, t.to("cuda:0"), prev_states=p, max_segments=23,
                                      derived=dd, prev_derived=pp, step_reward=-0.25, goal_reward=2.0)
        assert_lifted_parity(result, actual)


@pytest.mark.parametrize("name", TEXT)
def test_relabel_lifted_equals_golden_refine_on_fixtures(name):
    task = text_task(name)
    prev, reached, done, previous, states, pd, d = lifted_window(task)
    t = rl.lifted_goal_templates(task)
    result = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=23,
                                    derived=d, prev_derived=pd, step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(3, np.int32), t, result)


def toggle_window():
    task = text_task("gripper__prob05")
    # Ground goals make first achievement, loops, unmatched endpoints and generation order explicit.
    s0 = task.initial_state
    s1 = next(s for s in task.successor_states(s0) if s != s0)
    # Select a ground literal changed by the first action, without object-variable ambiguities.
    added = next(a for a in s1.atoms() if a not in s0.atoms())
    conditions = [task.condition([], [str(added)])]
    t = rl.GoalTemplates.from_conditions(task, [conditions])
    paths = [[s0, s1, s1, s0, s1, s0, s1, s0, s1],
             [s1, s0, s1, s0, s1, s0, s1, s0, s1]]
    previous = [[path[q] for path in paths] for q in range(8)]
    states = [[path[q+1] for path in paths] for q in range(8)]
    reached = np.stack([np.asarray(task.encode(row)) for row in states])
    prev = np.stack([np.asarray(task.encode(row)) for row in previous])
    done = np.zeros((8, 2), bool)
    done[3] = True
    return task, t, prev, reached, done, previous, states


@pytest.mark.parametrize("S", [0, 1, 2, 8])
@pytest.mark.parametrize("cap", [None, 0, 1, 3, 5, 20])
def test_relabel_lifted_scripted_loops_episode_order_and_caps(S, cap):
    task, t, prev, reached, done, previous, states = toggle_window()
    result = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=S,
                                    max_generated=cap, step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(2, np.int32), t, result, max_generated=cap)


def test_relabel_lifted_validation_budget_invalid_ids_and_padding():
    task, t, prev, reached, done, _, _ = toggle_window()
    kwargs = dict(prev_states=prev, max_segments=8)
    status = np.array(99, np.int64)
    out = rl.her_relabel_lifted(reached, done, None, t, status=status, **kwargs)
    assert out.status is status and int(status) == 0
    empty = rl.GoalTemplates.from_conditions(task, [[]])
    absent = rl.her_relabel_lifted(reached, done, None, empty, **kwargs)
    assert int(absent.status) == 0 and not absent.count.any() and (absent.segment == -1).all()
    for templates in (empty, t):
        out = rl.her_relabel_lifted(reached, done, np.full(2, -1, np.int32), templates, **kwargs)
        assert int(out.status) == 4 and not out.seg_valid.any() and not out.count.any()
        assert (out.segment == -1).all() and (out.seg_binding == -1).all() and not out.reward.any()
    # A variable-bearing query exhausts immediately with budget zero.
    full = rl.lifted_goal_templates(task)
    out = rl.her_relabel_lifted(reached, done, None, full, max_steps=0, **kwargs)
    assert int(out.status) == 8 and not out.count.any()
    for kw, value, error in [("max_segments", -1, "max_segments"), ("max_generated", -1, "max_generated"),
                              ("max_steps", -1, "max_steps"), ("words", 0, "widths"),
                              ("prev_states", None, "prev_states")]:
        with pytest.raises(ValueError, match=error):
            rl.her_relabel_lifted(reached, done, None, t, **{**kwargs, kw: value})
    with pytest.raises((TypeError, ValueError), match="done"):
        rl.her_relabel_lifted(reached, done.astype(np.int32), None, t, **kwargs)
    with pytest.raises((ValueError, TypeError), match="contiguous"):
        rl.her_relabel_lifted(reached[::2], done[::2], None, t, prev_states=prev[::2], max_segments=8)
    with pytest.raises(ValueError, match="prev_states"):
        rl.her_relabel_lifted(reached, done, None, t, prev_states=prev[:1], max_segments=8)


def test_relabel_lifted_empty_windows_and_half_words():
    task, t, prev, reached, done, _, _ = toggle_window()
    expected = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=8)
    halves = rl.her_relabel_lifted(reached.view(np.uint32), done, None, t, prev_states=prev.view(np.uint32), max_segments=8)
    for a, b in zip(expected, halves):
        np.testing.assert_array_equal(a, b)
    for T, N in [(0, 2), (8, 0), (0, 0)]:
        out = rl.her_relabel_lifted(reached[:T, :N].copy(), done[:T, :N].copy(), None, t,
                                   prev_states=prev[:T, :N].copy(), max_segments=2)
        assert not out.count.any() and not out.seg_valid.any() and int(out.status) == 0


def static_goal_task():
    domain = mymyr.Domain.from_string("""(define (domain fixed)
        (:requirements :strips) (:predicates (ok))
        (:action idle :parameters () :precondition (and) :effect (and)))""")
    return mymyr.Task(domain.instantiate_string("""(define (problem fixed) (:domain fixed)
        (:init (ok)) (:goal (ok)))"""))


@needs_frontend
@pytest.mark.parametrize("source_kind", ["task", "table", "suite"])
@pytest.mark.parametrize("framework", ["numpy", "torch", "cuda"])
@pytest.mark.parametrize("half_words", [False, True])
def test_relabel_lifted_static_only_default_goal_width(source_kind, framework, half_words):
    task = static_goal_task()
    assert task.words == 0
    table = rl.TaskTable([task])
    source = {"task": task, "table": table, "suite": rl.TaskSuite.group([task])}[source_kind]
    templates = rl.lifted_goal_templates(source)
    assert templates.words == table.words == 1
    states = np.zeros((2, 1, 2), np.uint64)
    done = np.zeros((2, 1), bool)
    torch = None if framework == "numpy" else (
        cuda_match_runtime() if framework == "cuda" else pytest.importorskip("torch"))

    def array(value):
        if value.dtype == np.uint64:
            value = value.view(np.uint32) if half_words else value
            if torch is not None:
                value = value.view(np.int32 if half_words else np.int64)
        return value if torch is None else torch.tensor(value, device="cuda:0" if framework == "cuda" else "cpu")

    x, flags = array(states), array(done)
    if framework == "cuda":
        templates = templates.to("cuda:0")
        assert templates.words == table.words
    for derived_width in (None, 0, 2):
        derived = None if derived_width is None else array(np.zeros((2, 1, derived_width), np.uint64))
        for words in (None, 2):
            out = rl.her_relabel_lifted(x, flags, None, templates, prev_states=x, max_segments=3,
                                       words=words, derived=derived, prev_derived=derived)
            assert out.goal_pos.shape == out.goal_neg.shape == (1, 3, 1 if words is None else words)
            assert out.goal_pos_derived.shape == out.goal_neg_derived.shape == (1, 3, derived_width or 0)
            assert int(out.status) == 0 and out.count.tolist() == [0]
            assert (out.segment == -1).all() and not out.seg_valid.any()
            assert all(not mask.any() for mask in (out.goal_pos, out.goal_neg, out.goal_pos_derived, out.goal_neg_derived))


@pytest.mark.parametrize("via_op", [False, True])
def test_relabel_lifted_torch_host_device_and_capture(via_op):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import register, unregister, her_relabel_lifted as op
    task, templates, prev, reached, done, _, _ = toggle_window()
    want = rl.her_relabel_lifted(reached, done, None, templates, prev_states=prev, max_segments=3, max_generated=4)
    devices = ["cpu"]
    try:
        import mymyr.cuda as mc
        if torch.cuda.is_available() and mc.available():
            devices.append("cuda:0")
    except ImportError:
        pass
    for device in devices:
        x = torch.tensor(reached.view(np.int64), device=device)
        p = torch.tensor(prev.view(np.int64), device=device)
        d = torch.tensor(done, device=device)
        ids = torch.zeros(2, dtype=torch.int32, device=device)
        t = templates if device == "cpu" else templates.to(device)
        handle = register(t)

        def run():
            if via_op:
                return rl.HerLiftedBatch(*op(handle, x, d, ids, p, 3, 4))
            return rl.her_relabel_lifted(x, d, ids, t, prev_states=p, max_segments=3, max_generated=4)

        try:
            out = run()
            assert_lifted_parity(want, out)
            if device == "cpu":
                continue
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            graph.replay()
            assert_lifted_parity(want, captured)
            ids.fill_(-1)
            graph.replay()
            assert captured.status.item() == 4 and not captured.count.any().item()
            ids.zero_()
            x.zero_()
            changed = rl.her_relabel_lifted(np.zeros_like(reached), done, None, templates, prev_states=prev,
                                           max_segments=3, max_generated=4)
            graph.replay()
            assert_lifted_parity(changed, captured)
            torch.cuda.synchronize()
        finally:
            unregister(handle)


def test_relabel_lifted_device_does_not_wait_for_its_stream():
    torch = cuda_match_runtime()
    task, templates, prev, reached, done, _, _ = toggle_window()
    t = templates.to("cuda:0")
    x = torch.tensor(reached.view(np.int64), device="cuda:0")
    p = torch.tensor(prev.view(np.int64), device="cuda:0")
    d = torch.tensor(done, device="cuda:0")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rl.her_relabel_lifted(x, d, None, t, prev_states=p, max_segments=8)
    stream.synchronize()
    event = torch.cuda.Event()
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(300_000_000)
            event.record()
            result = rl.her_relabel_lifted(x, d, None, t, prev_states=p, max_segments=8)
        pending = not event.query()
    finally:
        stream.synchronize()
    assert pending
    assert result.count.sum().item() > 0 and result.status.item() == 0


@pytest.mark.needs_gil
@pytest.mark.parametrize("static_only", [False, True])
def test_relabel_lifted_compile_and_opcheck(static_only):
    torch = pytest.importorskip("torch")
    from mymyr.rl.torch import register, unregister, her_relabel_lifted as op
    from mymyr.rl.torch._ops import _her_relabel_lifted_op
    if static_only:
        if not hasattr(mymyr, "Domain"):
            pytest.skip("built without the loki front end")
        t = rl.lifted_goal_templates(static_goal_task())
        prev = reached = np.zeros((1, 1, 1), np.uint64)
        done = np.zeros((1, 1), bool)
    else:
        task, t, prev, reached, done, _, _ = toggle_window()
    handle = register(t)
    x, p, d = torch.tensor(reached.view(np.int64)), torch.tensor(prev.view(np.int64)), torch.tensor(done)
    try:
        args = (handle, x, d, None, p, 3)
        torch.library.opcheck(_her_relabel_lifted_op, args)
        expected = op(*args)
        actual = torch.compile(lambda a, b, c: op(handle, a, b, None, c, 3), backend="eager", fullgraph=True)(x, d, p)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)
    finally:
        unregister(handle)


def suite_lifted_window():
    tasks = [text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")]
    rng = random.Random(27)
    T, N = 19, 2
    ids = np.empty((T, N), np.int32)
    done = np.zeros((T, N), bool)
    previous, states = [], []
    current = [None, None]
    for q in range(T):
        before, after = [], []
        for i in range(N):
            g = (q // 7 + i) % 2
            ids[q, i] = g
            if q % 7 == 0:
                current[i] = tasks[g].initial_state
            before.append(current[i])
            current[i] = rng.choice(tasks[g].successor_states(current[i]))
            after.append(current[i])
        previous.append(before)
        states.append(after)
        done[q] = q % 7 == 6
    source, rows, _, d = match_rows(tasks, [(int(ids[q, i]), states[q][i]) for q in range(T) for i in range(N)])
    _, prev, _, pd = match_rows(tasks, [(int(ids[q, i]), previous[q][i]) for q in range(T) for i in range(N)])
    return tasks, source, rows.reshape(T, N, -1), prev.reshape(T, N, -1), ids, done, previous, states


@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_suite_changes_instance_at_autoreset(on_device):
    tasks, source, rows, prev, ids, done, previous, states = suite_lifted_window()
    t = rl.lifted_goal_templates(source)
    expected = rl.her_relabel_lifted(rows, done, ids, t, prev_states=prev, max_segments=19,
                                    step_reward=-0.25, goal_reward=2.0, pool=rl.ThreadPool(2))
    assert_refine(tasks, previous, states, done, ids, t, expected)
    if on_device:
        torch = cuda_match_runtime()
        x, p, tid, d = [torch.tensor(a, device="cuda:0") for a in
                       (rows.view(np.int64), prev.view(np.int64), ids, done)]
        actual = rl.her_relabel_lifted(x, d, tid, t.to("cuda:0"), prev_states=p, max_segments=19,
                                      step_reward=-0.25, goal_reward=2.0)
        assert_lifted_parity(expected, actual)


@needs_frontend
@pytest.mark.parametrize("goal", ["(not (lit a))", "(done)", "(not (done))"])
@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_static_negative_derived_and_constant_goals(goal, on_device):
    from test_encoding_literals import goals_task
    task = goals_task("static_true", atoms="dense")
    # Alternate fluent and derived truth, while the static conjunct is satisfied throughout.
    path = [task.state(a) for a in [[], ["(lit a)", "(lit home)", "(flag)"], ["(lit a)"], [],
                                   ["(lit home)", "(flag)"], [], ["(lit a)"]]]
    previous = [[s] for s in path[:-1]]
    states = [[s] for s in path[1:]]
    _, prev, _, pd = match_rows([task], [(0, s) for s in path[:-1]])
    _, reached, _, d = match_rows([task], [(0, s) for s in path[1:]])
    prev, reached = prev[:, None, :], reached[:, None, :]
    pd, d = pd[:, None, :], d[:, None, :]
    done = np.zeros((len(states), 1), bool)
    t = rl.GoalTemplates.from_conditions(task, [[task.condition([], ["(heavy t1)", goal])]])
    expected = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, derived=d, prev_derived=pd,
                                    max_segments=8, step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(1, np.int32), t, expected)
    assert expected.count.sum() > 0
    if goal != "(not (lit a))":
        for missing in ("derived", "prev_derived"):
            kw = dict(derived=d, prev_derived=pd, prev_states=prev, max_segments=8)
            kw[missing] = None
            with pytest.raises(ValueError, match="derived"):
                rl.her_relabel_lifted(reached, done, None, t, **kw)
    if on_device:
        torch = cuda_match_runtime()
        x, p, dd, pp, flags = [torch.tensor(a, device="cuda:0") for a in
                              (reached.view(np.int64), prev.view(np.int64), d.view(np.int64), pd.view(np.int64), done)]
        actual = rl.her_relabel_lifted(x, flags, None, t.to("cuda:0"), prev_states=p, derived=dd, prev_derived=pp,
                                      max_segments=8, step_reward=-0.25, goal_reward=2.0)
        assert_lifted_parity(expected, actual)


@needs_frontend
@pytest.mark.parametrize("fixture", ["gripper", "counters"])
def test_relabel_lifted_pddl_fixtures(fixture):
    root = ROOT / "tests/data/pddl" / fixture
    task = mymyr.Task.from_pddl(root / "domain.pddl", root / "p01.pddl")
    prev, reached, done, previous, states, pd, d = lifted_window(task)
    t = rl.lifted_goal_templates(task)
    actual = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=23,
                                  step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(3, np.int32), t, actual)


@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_long_window(on_device):
    task, t, prev, reached, done, _, _ = toggle_window()
    # Long windows retain exact loop detection.
    base = np.concatenate([reached[:1, :1], prev[:1, :1]], axis=0)
    reached = np.tile(base, (513, 2, 1))[:1025].copy()
    prev = np.tile(base[::-1], (513, 2, 1))[:1025].copy()
    done = np.zeros((1025, 2), bool)
    expected = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=1)
    assert expected.seg_start.tolist() == [[1024], [1024]]
    assert expected.seg_end.tolist() == [[1024], [1024]]
    if on_device:
        torch = cuda_match_runtime()
        x, p, d = [torch.tensor(a, device="cuda:0") for a in (reached.view(np.int64), prev.view(np.int64), done)]
        actual = rl.her_relabel_lifted(x, d, None, t.to("cuda:0"), prev_states=p, max_segments=1)
        assert_lifted_parity(expected, actual)


@needs_frontend
@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_numeric_words_participate_in_loop_equality(on_device):
    root = ROOT / "tests/data/pddl/counters"
    task = mymyr.Task.from_pddl(root / "domain.pddl", root / "p01.pddl")
    t = rl.lifted_goal_templates(task)
    W = t.source.words
    row = np.asarray(task.encode([task.initial_state]))
    prev = np.tile(row, (5, 1, 1))
    reached = prev.copy()
    # Distinct numeric tails with the same fluent atoms, followed by a newly achieved nullary goal.
    prev[:, 0, W] = np.arange(5, dtype=np.uint64)
    reached[:, 0, W] = np.arange(1, 6, dtype=np.uint64)
    slot = task.literal("(done)").atom.slot
    reached[-1, 0, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    done = np.zeros((5, 1), bool)
    want = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=1)
    assert want.seg_start.tolist() == [[0]] and want.seg_end.tolist() == [[4]]
    assert want.achieved[:, 0].tolist() == [False] * 4 + [True]
    if on_device:
        torch = cuda_match_runtime()
        x, p, d = [torch.tensor(a, device="cuda:0") for a in (reached.view(np.int64), prev.view(np.int64), done)]
        actual = rl.her_relabel_lifted(x, d, None, t.to("cuda:0"), prev_states=p, max_segments=1)
        assert_lifted_parity(want, actual)


@needs_frontend
@pytest.mark.parametrize("framework", ["numpy", "torch", "cuda"])
def test_relabel_lifted_preserves_the_high_goal_bit(framework):
    names = [f"p{i:02d}" for i in range(64)]
    predicates = " ".join(f"({name})" for name in names)
    domain = mymyr.Domain.from_string(
        f"(define (domain flags) (:requirements :strips) (:predicates {predicates})"
        f" (:action mark :parameters () :precondition (and) :effect (and {predicates})))"
    )
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem p) (:domain flags) (:init) (:goal (p63)))"))
    assert task.literal("(p63)").atom.slot == 63
    t = rl.lifted_goal_templates(task)
    initial = task.initial_state
    final = task.successor_states(initial)[0]
    prev = np.asarray(task.encode([initial]))[None, :, :]
    reached = np.asarray(task.encode([final]))[None, :, :]
    done = np.zeros((1, 1), bool)
    expected = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, max_segments=1)
    assert expected.goal_pos.tolist() == [[[1 << 63]]]
    if framework == "numpy":
        return
    torch = cuda_match_runtime() if framework == "cuda" else pytest.importorskip("torch")
    device = "cuda:0" if framework == "cuda" else "cpu"
    x, p, d = [torch.tensor(a, device=device) for a in (reached.view(np.int64), prev.view(np.int64), done)]
    actual = rl.her_relabel_lifted(x, d, None, t.to(device) if framework == "cuda" else t,
                                  prev_states=p, max_segments=1)
    assert actual.goal_pos.item() == -(1 << 63)
    assert_lifted_parity(expected, actual)



@needs_frontend
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("on_device", [False, True])
def test_relabel_lifted_scripted_ipc_loops_and_resets(domain, on_device):
    torch = cuda_match_runtime() if on_device else None
    task = ipc(domain, "train", "p01")
    prev, reached, done, previous, states, pd, d = lifted_window(task, n=2, steps=13, scripted=True)
    t = rl.lifted_goal_templates(task)
    expected = rl.her_relabel_lifted(reached, done, None, t, prev_states=prev, derived=d, prev_derived=pd,
                                    max_segments=13, step_reward=-0.25, goal_reward=2.0)
    assert_refine([task], previous, states, done, np.zeros(2, np.int32), t, expected)
    if on_device:
        x, p, flags, dd, pp = [torch.tensor(a, device="cuda:0") for a in
                               (reached.view(np.int64), prev.view(np.int64), done, d.view(np.int64), pd.view(np.int64))]
        actual = rl.her_relabel_lifted(x, flags, None, t.to("cuda:0"), prev_states=p, derived=dd, prev_derived=pp,
                                      max_segments=13, step_reward=-0.25, goal_reward=2.0)
        assert_lifted_parity(expected, actual)
