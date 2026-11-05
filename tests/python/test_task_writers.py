"""The task writers (Task.to_text, Task.to_pddl, mymyr.write_pddl): every vendored task reads back as the same task.

A task read back from its PDDL is usually identical (equal ``to_text()``). Where normalization numbers things
differently the second time (a goal written back as a formula compiles into other derived helper predicates; a text
task's parameter types become PDDL types), the two tasks must have the same objects, predicates, functions, actions
and initial state, and agree on seeded random walks: the applicable actions, the atoms, the numeric values and the goal
test of every state, and the walk's cost under the metric.
"""

import pathlib

import pytest

import mymyr
from mymyr import search

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

DATA = pathlib.Path(__file__).resolve().parents[2] / "tests/data"


def _pddl_pairs():
    """(domain, problem) of every vendored PDDL problem: a directory's problems with its one domain file, else with the
    domain file named after the problem."""
    out = []
    for d in sorted({p.parent for p in DATA.rglob("*.pddl")}):
        files = sorted(d.glob("*.pddl"))
        domains = [f for f in files if "domain" in f.name.lower()]
        for p in (f for f in files if f not in domains):
            named = [dm for dm in domains if p.stem in dm.stem or dm.stem.replace("domain", "").strip("-_") == p.stem]
            if len(domains) == 1:
                out.append((domains[0], p))
            elif len(named) == 1:
                out.append((named[0], p))
            elif p.name == "conditional_problem.pddl":
                out.append((d / "conditional_domain.pddl", p))
            elif p.name == "test_problem.pddl" and (d / "domain.pddl").exists():
                out.append((d / "domain.pddl", p))
    return out


PDDL_PAIRS = _pddl_pairs()
TEXT_TASKS = sorted(p for p in DATA.rglob("*.txt") if p.parent.name in ("tasks", "tasks_iw", "numeric_tasks"))


def _id(path):
    return str(path.relative_to(DATA))


def _read_pddl(domain_text, problem_text):
    return mymyr.Task(mymyr.Domain.from_string(domain_text).instantiate_string(problem_text))


def _written(text, prefix="helper"):
    """The atom or predicate text with the domain's helpers named as the PDDL writer spells them (@domain-<k> ->
    helper-domain-<k>, unless that names a declared predicate); a task read back from PDDL declares them so."""
    return text.replace("(@domain-", f"({prefix}-domain-")


def _signature(t):
    f = t.formalism
    s = t.initial_state
    return {
        "objects": sorted(t.objects),
        "predicates": sorted((_written(f"({p.name})"), p.arity) for p in f.predicates if p.helper != "problem"),
        "functions": sorted((x.name, x.arity) for x in f.functions),
        "actions": sorted({x.name for x in f.schemas}),
        "atoms": sorted(_written(str(a)) for a in s.atoms()),
        "numeric": sorted(zip(t.numeric_names, [round(v, 9) for v in s.numeric_values()])),
        "goal": s.is_goal(),
    }


def _walk(t, seed, steps=150):
    """A seeded random walk: per state its applicable actions, atoms, numeric values and goal test; then the cost of
    the walk's actions under the task's metric."""
    out, plan = [], []
    s = t.initial_state
    x = seed
    for _ in range(steps):
        acts = sorted(t.applicable_actions(s), key=str)
        out.append(([str(a) for a in acts], sorted(str(a) for a in s.atoms()),
                    sorted(zip(t.numeric_names, [round(v, 9) for v in s.numeric_values()])), s.is_goal()))
        if not acts:
            break
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        a = acts[(x >> 33) % len(acts)]
        plan.append(a)
        s = t.apply(s, a)
    v = search.validate_plan(t, plan)
    return out, round(v.cost, 9)


def assert_same_task(a, b):
    if a.to_text() == b.to_text():
        return
    assert _signature(a) == _signature(b)
    for seed in (1, 2, 3):
        assert _written_walk(_walk(a, seed)) == _walk(b, seed), seed


def _written_walk(walk, prefix="helper", keep=lambda atom: True):
    steps, cost = walk
    return [(acts, sorted(_written(x, prefix) for x in atoms if keep(x)), values, goal)
            for acts, atoms, values, goal in steps], cost


@pytest.mark.parametrize("pair", PDDL_PAIRS, ids=lambda p: _id(p[1]))
def test_pddl_reads_back_as_the_same_task(pair):
    a = mymyr.Task.from_pddl(*pair)
    domain, problem = a.to_pddl()
    assert_same_task(a, _read_pddl(domain, problem))


@pytest.mark.parametrize("path", TEXT_TASKS, ids=_id)
def test_text_reads_back_identically_and_as_pddl(path, tmp_path):
    a = mymyr.Task.from_text(path)
    (tmp_path / "task.txt").write_text(a.to_text())
    assert mymyr.Task.from_text(tmp_path / "task.txt").to_text() == a.to_text()
    assert_same_task(a, _read_pddl(*a.to_pddl()))


def _domains():
    by = {}
    for d, p in PDDL_PAIRS:
        by.setdefault(d, []).append(p)
    return sorted((d, ps) for d, ps in by.items() if len(ps) > 1)


@pytest.mark.parametrize("group", _domains(), ids=lambda g: _id(g[0]))
def test_one_domain_serves_every_problem(group):
    domain, problems = group
    tasks = [mymyr.Task.from_pddl(domain, p) for p in problems]
    text, problem_texts = mymyr.write_pddl(tasks)
    assert len(problem_texts) == len(tasks)
    d = mymyr.Domain.from_string(text)
    for a, pt in zip(tasks, problem_texts):
        assert_same_task(a, mymyr.Task(d.instantiate_string(pt)))


def test_a_start_state_is_written_as_the_initial_state(blocks):
    s = blocks.initial_state
    for _ in range(5):
        s = blocks.apply(s, sorted(blocks.applicable_actions(s), key=str)[-1])
    b = _read_pddl(*blocks.to_pddl(start=s))
    assert sorted(map(str, b.initial_state.atoms())) == sorted(map(str, s.atoms()))
    assert b.to_pddl()[1] == blocks.to_pddl(start=s)[1]


def test_numeric_start_state():
    d = DATA / "pddl/counters"
    t = mymyr.Task.from_pddl(d / "domain.pddl", d / "p01.pddl")
    s = t.apply(t.initial_state, sorted(t.applicable_actions(t.initial_state), key=str)[0])
    b = _read_pddl(*t.to_pddl(start=s))
    assert dict(zip(b.numeric_names, b.initial_state.numeric_values())) == dict(
        zip(t.numeric_names, s.numeric_values()))


HELPERS = """(define (domain h) (:requirements :adl :derived-predicates)
 (:predicates (p ?x) (q ?x) {declared})
 (:action act :parameters (?x) :precondition (forall (?y) (or (p ?y) (q ?y))) :effect (p ?x)))"""
HELPERS_PROBLEM = "(define (problem h1) (:domain h) (:objects a b) (:init (q a)) (:goal (or (p a) (p b))))"


@pytest.mark.parametrize("declared", ["", "(helper-domain-0)", "(helper-domain-0) (helper_-domain-0 ?x)"])
def test_helpers_are_written_under_fresh_pddl_names(declared):
    a = _read_pddl(HELPERS.format(declared=declared), HELPERS_PROBLEM)
    assert [(p.name, p.helper) for p in a.formalism.predicates if p.helper] == [("@domain-0", "domain"), ("@problem-0", "problem")]
    domain, problem = a.to_pddl()
    written = "helper" + "_" * declared.count("helper") + "-domain-0"
    assert f"(:derived ({written})" in domain and "@" not in domain + problem
    b = _read_pddl(domain, problem)
    names = {p.name: p.helper for p in b.formalism.predicates}
    assert names[written] is None  # declared by the written domain
    assert [n for n, h in names.items() if h] == ["@problem-0"]  # the goal, written as a formula, compiles again
    declared_only = lambda atom: "@problem-" not in atom  # noqa: E731
    for seed in (1, 2):
        assert _written_walk(_walk(a, seed), written[: -len("-domain-0")], declared_only) == \
               _written_walk(_walk(b, seed), keep=declared_only)


def test_write_pddl_needs_tasks_of_one_domain(blocks):
    d = DATA / "pddl/counters"
    other = mymyr.Task.from_pddl(d / "domain.pddl", d / "p01.pddl")
    with pytest.raises(ValueError):
        mymyr.write_pddl([blocks, other])
    with pytest.raises(ValueError):
        mymyr.write_pddl([])
