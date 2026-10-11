"""Task.encoding_goal: the goal as written, expanded into signed ground literals (problems in tests/data/pddl/goals)."""

import pathlib
import pickle

import pytest

import mymyr
from mymyr import formalism as fm

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOALS = ROOT / "tests/data/pddl/goals"
BLOCKS = ROOT / "tests/data/pddl/blocks"
FEATURES = ROOT / "tests/data/pddl/features"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")

# problem: (literals, complete, unsatisfiable)
CASES = {
    "conjunctive": ({"(lit a)", "(not (ok t1))", "(at t2 b)", "(done)"}, True, False),
    "forall": ({"(lit home)", "(lit a)", "(lit b)"}, True, False),
    "forall_nested": ({"(at b1 home)", "(at b1 a)", "(at b1 b)"}, True, False),
    "forall_subtypes_constants": ({"(ok k0)", "(ok t1)", "(ok t2)", "(ok b1)"}, True, False),
    "forall_either": ({"(seen home)", "(seen a)", "(seen b)", "(seen b1)"}, True, False),
    "forall_empty_type": ({"(lit a)"}, True, False),
    "forall_negative": ({"(not (ok b1))"}, True, False),
    "forall_conjunctive_body": ({"(ok b1)", "(not (at b1 a))"}, True, False),
    "imply_true": ({"(ok t1)", "(at b1 home)", "(at b1 a)", "(at b1 b)"}, True, False),
    "imply_false": ({"(lit a)"}, True, False),
    "imply_equality": ({"(lit a)"}, True, False),
    "imply_quantified_antecedent": ({"(flag)", "(ok t2)"}, True, False),
    "imply_fluent": ({"(lit a)"}, False, False),
    "or": ({"(lit a)"}, False, False),
    "exists": ({"(lit a)"}, False, False),
    "forall_exists": ({"(lit a)"}, False, False),
    "negated_forall": ({"(lit a)"}, False, False),
    "not_or": ({"(lit a)", "(not (lit b))", "(not (ok t1))"}, True, False),
    "not_exists": ({"(lit a)", "(not (ok k0))", "(not (ok t1))", "(not (ok t2))", "(not (ok b1))"}, True, False),
    "not_imply": ({"(lit a)", "(not (ok t2))"}, True, False),
    "not_imply_false": ({"(lit a)"}, False, False),
    "not_not": ({"(lit a)", "(not (ok t1))"}, True, False),
    "not_and": ({"(lit a)"}, False, False),
    "contradiction": ({"(lit a)", "(not (lit a))", "(ok t1)"}, True, False),
    "forall_equality": ({"(not (= home a))", "(not (= a a))", "(not (= b a))"}, True, True),
    "static_false": ({"(heavy t2)", "(lit a)"}, True, True),
    "static_true": ({"(heavy t1)", "(not (heavy t2))", "(lit a)"}, True, False),
    "duplicate": ({"(lit home)", "(lit a)", "(lit b)", "(not (ok t1))"}, True, False),
}


@pytest.fixture(scope="module")
def domain():
    return mymyr.Domain.from_file(GOALS / "domain.pddl")


def task_of(domain, problem):
    return mymyr.Task(domain.instantiate(GOALS / f"{problem}.pddl"))


def literals(task):
    return [str(literal) for literal in task.encoding_goal]


def goal_key(task):
    """The encoding goal as (positive, predicate, object indices) triples, which survive a text round trip."""
    index = {o.name: i for i, o in enumerate(task.formalism.objects)}
    return [(x.positive, x.predicate.name, [index[o.name] for o in x.objects]) for x in task.encoding_goal]


@pytest.mark.parametrize("problem", sorted(CASES))
def test_expansion(domain, problem):
    expected, complete, unsatisfiable = CASES[problem]
    task = task_of(domain, problem)
    got = literals(task)
    assert set(got) == expected and len(got) == len(expected)
    assert task.encoding_goal_complete is complete
    assert task.encoding_goal_unsatisfiable is unsatisfiable
    assert all(isinstance(literal, fm.GroundLiteral) for literal in task.encoding_goal)


@pytest.mark.parametrize("problem", sorted(CASES))
def test_order_is_positive_first_then_predicate_then_objects(domain, problem):
    task = task_of(domain, problem)
    predicate_index = {p.name: i for i, p in enumerate(task.formalism.predicates)}
    object_index = {o.name: i for i, o in enumerate(task.formalism.objects)}
    keys = [
        (not literal.positive, predicate_index[literal.predicate.name], [object_index[o.name] for o in literal.objects])
        for literal in task.encoding_goal
    ]
    assert keys == sorted(keys)


@pytest.mark.parametrize("problem", sorted(CASES))
def test_normalized_task_carries_the_same_goal(domain, problem):
    normalized = domain.instantiate(GOALS / f"{problem}.pddl")
    task = mymyr.Task(normalized)
    assert [str(x) for x in normalized.encoding_goal] == literals(task)
    assert normalized.encoding_goal_complete is task.encoding_goal_complete
    assert normalized.encoding_goal_unsatisfiable is task.encoding_goal_unsatisfiable


def test_the_planners_goal_is_unchanged(domain):
    task = task_of(domain, "forall")
    assert len(task.goal_condition.literals) == 1
    assert task.goal_condition.literals[0].predicate.name.startswith("axiom_")
    # the planner still reaches the goal only through the derived predicate
    state = task.initial_state
    assert not task.is_goal(state)


def test_conjunctive_goals_equal_the_normalized_goal(domain):
    cases = [
        mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl"),
        task_of(domain, "conjunctive"),
        mymyr.Task.from_pddl(
            ROOT / "tests/data/pddl/gripper/domain.pddl", ROOT / "tests/data/pddl/gripper/p01.pddl"
        ),
    ]
    for task in cases:
        assert task.encoding_goal_complete
        assert not task.encoding_goal_unsatisfiable
        assert {str(x) for x in task.encoding_goal} == {str(x) for x in task.goal_condition.literals}


def test_numeric_constraints_are_not_expandable():
    domain = mymyr.Domain.from_string(
        "(define (domain n) (:requirements :strips :numeric-fluents) (:predicates (p)) (:functions (f))"
        " (:action inc :parameters () :precondition (and) :effect (and (p) (increase (f) 1))))"
    )
    task = mymyr.Task(
        domain.instantiate_string("(define (problem q) (:domain n) (:init (= (f) 0)) (:goal (and (p) (>= (f) 2))))")
    )
    assert literals(task) == ["(p)"]
    assert not task.encoding_goal_complete
    assert len(task.goal_condition.numeric_constraints) == 1


@pytest.mark.parametrize("problem", sorted(CASES))
def test_pickle_round_trip(domain, problem):
    task = task_of(domain, problem)
    # the task from the PDDL front end pickles as its source
    back = pickle.loads(pickle.dumps(task))
    assert literals(back) == literals(task)
    assert back.encoding_goal_complete is task.encoding_goal_complete
    assert back.encoding_goal_unsatisfiable is task.encoding_goal_unsatisfiable
    assert back.fingerprint == task.fingerprint


@pytest.mark.parametrize(
    "problem", ["forall", "or", "static_false", "conjunctive", "forall_nested", "not_or", "contradiction"]
)
def test_text_with_the_encoding_goal(domain, problem, tmp_path):
    normalized = domain.instantiate(GOALS / f"{problem}.pddl")
    path = tmp_path / "task.txt"
    path.write_text(normalized.to_text(encoding_goal=True))
    task = mymyr.Task.from_text(str(path))
    reference = mymyr.Task(normalized)
    assert goal_key(task) == goal_key(reference)
    assert task.encoding_goal_complete is reference.encoding_goal_complete
    assert task.encoding_goal_unsatisfiable is reference.encoding_goal_unsatisfiable
    # a task built from text pickles as text and keeps the goal
    back = pickle.loads(pickle.dumps(task))
    assert literals(back) == literals(task)
    assert back.fingerprint == task.fingerprint


@pytest.mark.parametrize(
    "problem, expected, complete",
    [
        ("conjunctive", 4, True),
        ("forall", 0, False),
        ("or", 1, False),
        ("static_false", 2, True),
    ],
)
def test_text_without_the_goal_as_written(domain, problem, expected, complete, tmp_path):
    path = tmp_path / "task.txt"
    path.write_text(domain.instantiate(GOALS / f"{problem}.pddl").to_text())
    task = mymyr.Task.from_text(str(path))
    assert len(task.encoding_goal) == expected
    assert task.encoding_goal_complete is complete



def test_text_with_numeric_constraints_is_incomplete(tmp_path):
    domain = mymyr.Domain.from_string(
        "(define (domain n) (:requirements :strips :numeric-fluents) (:predicates (p)) (:functions (f))"
        " (:action inc :parameters () :precondition (and) :effect (and (p) (increase (f) 1))))"
    )
    normalized = domain.instantiate_string(
        "(define (problem q) (:domain n) (:init (= (f) 0)) (:goal (and (p) (>= (f) 2))))"
    )
    path = tmp_path / "task.txt"
    path.write_text(normalized.to_text())
    task = mymyr.Task.from_text(str(path))
    assert [str(x) for x in task.encoding_goal] == ["(p)"]
    assert not task.encoding_goal_complete


def test_text_and_pddl_agree_on_a_negated_disjunction(domain, tmp_path):
    normalized = domain.instantiate(GOALS / "not_or.pddl")
    path = tmp_path / "task.txt"
    path.write_text(normalized.to_text())
    from_text = mymyr.Task.from_text(str(path))
    assert goal_key(from_text) == goal_key(mymyr.Task(normalized))
    assert from_text.encoding_goal_complete


@pytest.mark.parametrize("stem", ["p02", "p03"])
def test_expanded_goals_equal_their_written_out_twins(stem):
    domain = mymyr.Domain.from_file(FEATURES / "domain.pddl")
    written = mymyr.Task(domain.instantiate(FEATURES / f"{stem}.pddl"))
    twin = mymyr.Task(domain.instantiate(FEATURES / f"{stem}_expanded.pddl"))
    assert {str(x) for x in written.encoding_goal} == {str(x) for x in twin.goal_condition.literals}
    assert written.encoding_goal_complete and not written.encoding_goal_unsatisfiable


def test_a_false_static_literal_marks_the_problem_unsatisfiable():
    domain = mymyr.Domain.from_file(FEATURES / "domain.pddl")
    task = mymyr.Task(domain.instantiate(FEATURES / "p04.pddl"))
    assert "(fixed b)" in {str(x) for x in task.encoding_goal}
    assert task.encoding_goal_unsatisfiable
