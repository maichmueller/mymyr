"""Formulas: atoms, literals and conjunctive conditions as values (mymyr.formalism), their construction on a Task,
holds, lift and ground, pickling, and ground conditions as search goals.

The constructor tests follow the fork's Python tests of the same names on its own instances (its data/ directory: env
MYMYR_FORK_DATA; skipped when missing); holds is checked against the fork's recorded walks (tests/data/expected)."""

import json
import os
import pathlib
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

import mymyr
from mymyr import formalism as fm
from mymyr import search
from conftest import ROOT, SMALL_TASKS, text_task, walk

FORK_DATA = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GOLDEN = ROOT / "tests/data/expected"
WORK = pathlib.Path(os.environ.get("MYMYR_WORK", ROOT / ".work"))
IPC = pathlib.Path(os.environ["MYMYR_IPC"]) if "MYMYR_IPC" in os.environ else None
NUMERIC = ROOT / "tests/data/numeric_tasks"


def fork_task(domain, problem="test_problem.pddl"):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    d = FORK_DATA / domain
    if not (d / problem).is_file():
        pytest.skip(f"the fork's instance {domain}/{problem} is missing (set MYMYR_FORK_DATA)")
    return mymyr.Task.from_pddl(d / "domain.pddl", d / problem)


# ------------------------------------------------------------------------------------------------ the fork's tests


def test_new_atom_and_literal():
    task = fork_task("blocks_4")
    on = next(p for p in task.formalism.predicates if p.name == "on")
    atom = task.atom(on, "?v1", "?v2")
    assert isinstance(atom, mymyr.Atom) and not atom.is_ground
    assert str(atom) == "(on ?v1 ?v2)"
    assert [v.name for v in atom.terms] == ["v1", "v2"] and [v.position for v in atom.terms] == [0, 1]
    literal = task.literal(atom, positive=False)
    assert isinstance(literal, mymyr.Literal) and literal.atom == atom and not literal.positive
    assert str(literal) == "(not (on ?v1 ?v2))"
    assert task.literal("(not (on ?v1 ?v2))") == literal == task.literal("on", "?v1", "?v2", positive=False)
    assert task.atom("(on ?v1 ?v2)") == atom and hash(task.atom("(on ?v1 ?v2)")) == hash(atom)


def test_new_ground_atom_and_ground_literal():
    task = fork_task("blocks_4")
    on = next(p for p in task.formalism.predicates if p.name == "on")
    b1, b2 = (next(o for o in task.formalism.objects if o.name == n) for n in ("b1", "b2"))
    atom = task.atom(on, b1, b2)
    assert isinstance(atom, mymyr.GroundAtom) and atom.objects == (b1, b2) and atom.predicate == on
    assert str(atom) == "(on b1 b2)" and atom.kind == "fluent"
    assert task.atom("on", "b1", "b2") == task.atom("(on b1 b2)") == task.atom(("on", ["b1", "b2"])) == atom
    literal = task.literal(atom, positive=False)
    assert isinstance(literal, mymyr.GroundLiteral) and literal.atom == atom and not literal.positive
    assert str(literal) == "(not (on b1 b2))"


def test_new_ground_action():
    task = fork_task("blocks_4")
    b1, b2 = (next(o for o in task.formalism.objects if o.name == n) for n in ("b1", "b2"))
    action = task.action("stack", [b1, b2])
    assert not action.is_applicable(task.initial_state)
    assert str(action) == "(stack b1 b2)"


def test_new_conjunctive_condition():
    task = fork_task("blocks_4")
    literals = [task.literal("on", "?a", "?b"), task.literal("on", "?b", "?c")]
    c = task.condition(["?a", "?b", "?c"], literals)
    assert isinstance(c, mymyr.ConjunctiveCondition) and c.arity == 3
    assert [v.name for v in c.variables] == ["a", "b", "c"] and [v.position for v in c.variables] == [0, 1, 2]
    assert list(c.literals) == literals
    assert str(c) == "(?a ?b ?c) (and (on ?a ?b) (on ?b ?c))"
    # typed parameters, equalities and numeric constraints on a numeric task
    counters = text_task_numeric("cs-counters")
    k = counters.condition(["?x", "?y"], [], equalities=[("?x", "?y", False)], constraints=["(< (value ?x) (value ?y))"])
    assert k.variables[0].types == ()
    assert len(k.equalities) == 1 and k.equalities[0][2] is False and len(k.numeric_constraints) == 1
    assert str(k.numeric_constraints[0]) == "(< (value ?x) (value ?y))"
    assert len(k.ground(counters.initial_state)) == 6  # ordered pairs of distinct counters with distinct values
    typed = fork_task("gripper")
    kinds = [t.name for t in typed.formalism.types]
    if kinds:
        tc = typed.condition([("x", kinds[0]), ("y", kinds)], [])
        assert [t.name for t in tc.variables[0].types] == kinds[:1] and len(tc.variables[1].types) == len(kinds)
        assert len(tc.ground(typed.initial_state)) > 0
    with pytest.raises(ValueError, match="not a parameter"):
        task.condition(["?a"], ["(on ?a ?b)"])
    with pytest.raises(ValueError, match="takes 2"):
        task.condition(["?a"], ["(on ?a)"])
    with pytest.raises(KeyError):
        task.condition(["?a"], ["(no-such-predicate ?a)"])
    with pytest.raises(ValueError, match="constraint"):
        counters.condition(["?x"], [], constraints=["(< (value ?x) (no-such-function))"])


def text_task_numeric(name):
    return mymyr.Task.from_text(str(NUMERIC / f"{name}.txt"))


def test_literal_holds():
    task = fork_task("miconic-fulladl")
    s = task.initial_state
    actions = task.applicable_actions(s)
    assert actions
    for a in actions:
        pre = task.precondition(a.schema)
        (g,) = pre.ground(s, partial=list(a.objects))
        assert g.holds(s) and s.holds(g)
        for lit in g.literals:
            assert lit.holds(s)


def test_holds():
    task = fork_task("miconic-fulladl")
    assert not task.goal_condition.holds(task.initial_state)


def test_grounder():
    task = fork_task("gripper")
    s = task.initial_state
    pick = task.formalism.schemas[[x.name for x in task.formalism.schemas].index("pick")].precondition
    assert len(pick.ground(s)) == 4 and len(pick.ground(s, limit=1)) == 1
    assert pick.ground(s) == task.precondition("pick").ground(s)  # the same values from the task


def test_grounder_blacklist():
    """The fork drops the literals of blacklisted predicates from the groundings; here, the ground literals filter."""
    task = fork_task("gripper")
    pick = task.precondition("pick")
    groundings = pick.ground(task.initial_state)
    assert len(groundings) == 4
    for g in groundings:
        kept = [lit for lit in g.literals if lit.predicate.name != "ball"]
        assert len(kept) == len(g.literals) - 1
        assert task.ground_condition(kept).holds(task.initial_state)


def test_lift():
    task = fork_task("gripper", "test_problem2.pddl")
    goal = task.goal_condition
    lifted = goal.lift()
    assert lifted.arity == 3 and len(lifted.literals) == 2
    v = lifted.variables
    assert lifted.literals[0].terms == (v[0], v[1]) and lifted.literals[1].terms == (v[2], v[1])
    assert [x.name for x in v] == ["x0", "x1", "x2"]


def test_lift_inequalities():
    task = fork_task("hiking")
    goal = task.goal_condition
    with_ne = goal.lift(add_inequalities=True)
    without = goal.lift()
    assert with_ne.arity == without.arity == 2
    assert len(without.literals) == 1 and not without.equalities
    assert len(with_ne.literals) == 1 and [e[2] for e in with_ne.equalities] == [False]


def test_new_grounded_condition():
    task = fork_task("blocks_4")
    s = task.initial_state
    g = task.ground_condition(s.atoms())
    assert g.holds(s) and s.holds(g)
    assert not task.ground_condition([task.literal(a, positive=False) for a in s.atoms()][:1]).holds(s)


# ------------------------------------------------------------------------------------------------ values


def test_formalism_and_task_values_agree():
    for name in SMALL_TASKS:
        task = text_task(name)
        f = task.formalism
        assert f.goal == task.goal_condition and hash(f.goal) == hash(task.goal_condition)
        for k, schema in enumerate(f.schemas):
            assert schema.precondition == task.precondition(k)
        for a in f.static_init[:20]:
            assert a.kind == "static" and task.initial_state.holds(a)
        for a in f.fluent_init:
            assert a.slot is None and task.initial_state.holds(a)  # a NormalizedTask has no slots
        for ax in f.axioms:
            assert isinstance(ax.head, fm.Atom) and ax.head.predicate.kind == "derived"
            assert isinstance(ax.body, fm.ConjunctiveCondition) and ax.body.arity == len(ax.parameters)


def gripper():
    """gripper from PDDL (real object names)."""
    return golden_task(json.loads((GOLDEN / "gripper__prob05.json").read_text()))


def test_pickling():
    task = gripper()
    s = task.initial_state
    values = [
        task.atom("at", "?b", "rooma"),
        task.literal("at", "?b", "rooma", positive=False),
        task.atom("(at ball1 rooma)"),
        task.literal("(not (at ball1 rooma))"),
        task.condition(["?b", "?r"], ["(at ?b ?r)"], equalities=[("?b", "ball1", False)]),
        task.goal_condition,
        task.goal_condition.lift(True),
        task.precondition("pick"),
        task.precondition("pick").variables[0],
    ]
    for x in values:
        t2, y = pickle.loads(pickle.dumps((task, x)))
        assert type(y) is type(x) and str(y) == str(x) and repr(y) == repr(x)
        assert t2.fingerprint == task.fingerprint and y != x  # values of different task instances differ
        t3, z = pickle.loads(pickle.dumps((t2, y)))
        assert str(z) == str(x)
    g2 = pickle.loads(pickle.dumps((task, task.goal_condition, s)))
    assert g2[1].holds(g2[2]) == task.goal_condition.holds(s)
    with pytest.raises(TypeError, match="pickles with the Task"):
        pickle.dumps(task.formalism.goal)
    with pytest.raises(ValueError, match="another task"):
        task.goal_condition.holds(g2[2])


def test_numeric_constraints_round_trip_as_text():
    task = text_task_numeric("cs-counters")
    goal = task.goal_condition
    assert len(goal.numeric_constraints) == 3 and not goal.literals
    again = task.ground_condition(constraints=goal.numeric_constraints)
    assert again == goal
    third = task.ground_condition(constraints=["(<= (+ (value o0) 0.1) (value o1))"])
    assert str(third.numeric_constraints[0]) == "(<= (+ (value o0) 0.1) (value o1))"
    e = third.numeric_constraints[0].lhs
    assert e.op == "+" and [c.op for c in e.children] == ["function", "number"] and e.children[1].value == 0.1


# ------------------------------------------------------------------------------------------------ holds


def golden_cases():
    # philosophers: its PDDL declares no :derived-predicates requirement, which the front end insists on
    return sorted(f.stem for f in GOLDEN.glob("*.json") if not f.stem.startswith("philosophers")) if GOLDEN.is_dir() else []


def golden_task(g):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    tag, prob = g["source"]["tag"], g["source"]["problem"]
    dom = g["source"].get("domain_file", "domain.pddl")
    if tag.startswith("ipc/"):
        if IPC is None:
            pytest.skip("set MYMYR_IPC")
        d = IPC / tag[4:] / "test"
    else:
        d = WORK / "mimir-cs" / "Benchmark" / tag
    if not (d / prob).is_file():
        pytest.skip(f"PDDL of {g['task']} not found")
    return mymyr.Task.from_pddl(d / dom, d / prob)


@pytest.mark.parametrize("name", golden_cases())
def test_holds_agrees_with_the_recorded_walks(name):
    g = json.loads((GOLDEN / f"{name}.json").read_text())
    task = golden_task(g)
    goal = task.goal_condition
    rng = random.Random(0)
    checked = 0
    for w in g["walks"]["walks"]:
        if w.get("duplicate_action_strings"):
            continue  # two actions print alike: the recorded strings do not name the action taken
        s = task.initial_state
        for step in w["steps"]:
            fluent = step["fluent_atoms"]
            atoms = {str(a) for a in s.atoms()}
            if len(fluent.get("items", ())) == fluent["count"]:
                assert atoms == set(fluent["items"])
            assert goal.holds(s) == s.holds(goal) == s.is_goal() == (step["is_goal"] in (True, "True"))
            true = list(s.atoms()) + [a for a in s.derived_atoms() if a.kind == "derived"]
            assert all(a.holds(s) for a in true)
            if true:
                sample = rng.sample(true, min(4, len(true)))
                assert task.ground_condition(sample).holds(s)
                assert not task.ground_condition(sample + [task.literal(sample[0], positive=False)]).holds(s)
            checked += 1
            if step.get("taken") is None:
                break
            taken = [a for a in task.applicable_actions(s) if str(a) == step["taken"]]
            if len(taken) != 1:
                break  # the fork prints this action differently (parameters added by normalization)
            s = task.apply(s, taken[0])
    assert checked > 0


def test_negative_literals_hold_exactly_when_the_atom_is_false():
    for name in SMALL_TASKS:
        task = text_task(name)
        slots = range(task.num_atoms)
        for s in walk(task, steps=8, walks=1):
            on = set(s.atom_slots())
            for k in slots:
                a = task.atom(k)
                assert task.literal(a, positive=False).holds(s) == (k not in on)


# ------------------------------------------------------------------------------------------------ lift and ground


def test_lift_then_ground_recovers_the_condition():
    for name in SMALL_TASKS:
        task = text_task(name)
        rng = random.Random(1)
        for s in walk(task, steps=6, walks=1):
            atoms = list(s.atoms())
            g = task.ground_condition(rng.sample(atoms, min(3, len(atoms))))
            for add in (False, True):
                lifted = g.lift(add)
                groundings = lifted.ground(s)
                assert g in groundings
                assert all(x.holds(s) for x in groundings)
                # the binding of first appearance fixes the grounding
                objects = []
                for lit in g.literals:
                    for o in lit.objects:
                        if o not in objects:
                            objects.append(o)
                assert lifted.ground(s, partial=objects) == [g]


def test_ground_follows_the_bindings():
    for name in SMALL_TASKS[:4]:
        task = text_task(name)
        for s in walk(task, steps=4, walks=1):
            for k in range(task.num_schemas):
                pre = task.precondition(k)
                tuples = list(task.bindings(pre, s, limit=50))
                ground = pre.ground(s, limit=50)
                assert len(ground) == len(tuples)
                for b, g in zip(tuples, ground):
                    assert all(lit.holds(s) for lit in g.literals)
                    assert pre.ground(s, partial=list(b)) == [g]


# ------------------------------------------------------------------------------------------------ goals


def test_constructed_goals_give_the_plans_of_the_equivalent_forms():
    task = gripper()
    target = "(at ball1 roomb)"
    atom = task.atom(target)
    g = task.ground_condition([atom])
    for goal in ([[target]], [[atom]], g, [g], [[task.literal(atom)]]):
        a = search.astar(task, heuristic="blind", goal=goal)
        b = search.brfs(task, stop_at_goal=True, goal=goal)
        i = search.iw(task, max_arity=2, goal=goal)
        a0 = search.astar(task, heuristic="blind", goal=[[target]])
        assert a.plan == a0.plan and a.goal_state.holds(atom)
        assert b.solved and b.plan == search.brfs(task, stop_at_goal=True, goal=[[target]]).plan
        assert i.plan == search.iw(task, max_arity=2, goal=[[target]]).plan
    # the task's goal as a ground condition: the plans of the task's goal
    depot = text_task("depot__p02")
    goal = depot.goal_condition
    assert search.brfs(depot, stop_at_goal=True, goal=goal).plan == search.brfs(depot, stop_at_goal=True).plan
    assert search.astar(depot, heuristic="hmax", goal=goal).plan == search.astar(depot, heuristic="hmax").plan
    assert search.astar(depot, heuristic="hmax", goal=[goal.literals]).plan == search.astar(depot, heuristic="hmax").plan


def test_any_of_goals():
    task = gripper()
    near = task.ground_condition(["(carry ball1 left)"])
    far = task.ground_condition(["(at ball1 roomb)", "(at ball2 roomb)"])
    r = search.brfs(task, stop_at_goal=True, goal=[far, near])
    assert r.solved and len(r.plan) == len(search.brfs(task, stop_at_goal=True, goal=near).plan)
    with pytest.raises(TypeError, match="goal="):
        search.brfs(task, goal=["(at ball1 roomb)"])  # a flat list of atoms: ambiguous


def test_negative_derived_and_static_goal_literals():
    task = gripper()
    s = task.initial_state
    held = next(a for a in s.atoms() if a.predicate.name == "at-robby")
    neg = task.ground_condition([task.literal(held, positive=False)])
    for r in (search.brfs(task, stop_at_goal=True, goal=neg), search.astar(task, heuristic="blind", goal=neg),
              search.iw(task, max_arity=1, goal=neg)):
        assert len(r.plan) == 1
        assert not task.apply(s, r.plan[0]).holds(held)
    # a derived goal literal (philosophers: blocked) and its negation
    phil = text_task("philosophers__p03-phil4")
    derived = phil.goal_condition.literals[0]
    assert derived.atom.kind == "derived"
    one = phil.ground_condition([derived])
    r = search.iw(phil, max_arity=2, goal=one)
    assert r.status == search.Status.SOLVED and one.holds(r.goal_state)
    b = search.brfs(phil, stop_at_goal=True, goal=one)
    assert b.solved and len(b.plan) <= len(r.plan) and one.holds(b_states(phil, b.plan))
    initially = phil.ground_condition([phil.literal(derived, positive=False)])
    assert initially.holds(phil.initial_state)
    assert search.brfs(phil, stop_at_goal=True, goal=initially).plan == []
    # static literals: a true one is no constraint, a false one makes the goal impossible
    statics = task.formalism.static_init
    assert statics
    if statics:
        true_static = task.atom(str(statics[0]))
        g = task.ground_condition(["(at ball1 roomb)", true_static])
        assert search.brfs(task, stop_at_goal=True, goal=g).plan == search.brfs(task, stop_at_goal=True, goal=[["(at ball1 roomb)"]]).plan
        false_static = task.literal(true_static, positive=False)
        r = search.brfs(task, stop_at_goal=True, goal=task.ground_condition([false_static]))
        assert not r.solved


def b_states(task, plan):
    """The last state of a plan from the initial state."""
    s = task.initial_state
    for a in plan:
        s = task.apply(s, a)
    return s


def test_numeric_goal_constraint():
    task = text_task_numeric("cs-counters")
    v = task.initial_state.numeric_values()
    goal = task.ground_condition(constraints=[f"(> (value o3) {v[3] + 1})"])
    assert not goal.holds(task.initial_state)
    for r in (search.astar(task, heuristic="blind", goal=goal), search.gbfs(task, heuristic="blind", goal=goal)):
        assert r.status == search.Status.SOLVED and goal.holds(r.goal_state)
        assert r.goal_state.numeric_values()[3] > v[3] + 1
    b = search.brfs(task, stop_at_goal=True, goal=goal)
    assert b.solved and len(b.plan) == len(search.astar(task, heuristic="blind", goal=goal).plan)


def test_goal_of_another_task_is_refused():
    a = text_task("gripper__prob05")
    b = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="another task"):
        search.brfs(a, goal=b.goal_condition)


# ------------------------------------------------------------------------------------------------ threads


def test_construction_and_evaluation_from_eight_threads():
    task = text_task("philosophers__p03-phil4")  # derived atoms: the per-thread evaluation workspace
    states = walk(task, steps=10, walks=2)
    names = [str(a) for a in states[0].atoms()] + [str(a) for a in task.derived_atoms(states[0])][:10]

    def job(k):
        rng = random.Random(k)
        out = []
        for _ in range(30):
            picked = rng.sample(names, 3)
            g = task.ground_condition([task.literal(n, positive=rng.random() < 0.7) for n in picked])
            lifted = g.lift()
            out.append((str(g), tuple(g.holds(s) for s in states), tuple(len(lifted.ground(s, limit=5)) for s in states[:3])))
        return out

    with ThreadPoolExecutor(8) as ex:
        parallel = list(ex.map(job, range(8)))
    assert parallel == [job(k) for k in range(8)]
