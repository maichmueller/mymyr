"""Minimum-g width search, its heuristic adapters, controls and fork-compatible root rule."""
import concurrent.futures
import math

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status
from conftest import text_task

MODES = ["classical", "abstracted", "base_abstracted"]


@pytest.fixture
def gap(tmp_path):
    path = tmp_path / "task.txt"
    path.write_text("""O 0
P 4
F 0 seed
F 0 a
F 0 b
F 0 goal
SI 0
FI 1
0
G 1
3 1
A 4
add-a 0
L 1
0 1
E 1
0
L 0
F 1
1 1
add-b 0
L 1
0 1
E 1
0
L 0
F 1
2 1
finish 0
L 2
1 1
2 1
E 1
0
L 0
F 1
3 1
clear-seed 0
L 1
0 1
E 1
0
L 0
F 1
0 0
""")
    return mymyr.Task.from_text(str(path))


def same(a, b):
    assert a.status == b.status
    assert a.cost == b.cost
    assert [str(x) for x in a.plan] == [str(x) for x in b.plan]
    assert a.stats.expanded == b.stats.expanded
    assert a.stats.generated == b.stats.generated


@pytest.mark.parametrize("features", MODES)
@pytest.mark.parametrize("width", [1, 2])
def test_width_boundary_and_probe(gap, features, width):
    r = search.astar_iw(gap, features=features, width=width, heuristic="blind")
    assert isinstance(r, search.AStarIwResult)
    assert isinstance(r, search.BestFirstResult)
    assert r.status == (Status.SOLVED if width == 2 else Status.EXHAUSTED)
    assert r.novelty.updates > 0
    assert r.novelty.table_bytes > 0
    if r.solved:
        assert r.cost == len(r.plan) == 3
        assert gap.is_goal(r.goal_state)
    other = search.astar_iw(gap, features=features, width=width, heuristic="blind",
                            probe_novelty_before_heuristic=False)
    same(r, other)
    assert r.evaluations <= other.evaluations


@pytest.mark.parametrize("features", MODES)
def test_landmarks_and_preservation(gap, features):
    for preserve in [False, True]:
        r = search.astar_iw(gap, features=features, landmarks="approximate", preserve_landmark_atoms=preserve,
                            preserve_goal_atoms=preserve)
        assert r.solved and r.cost == 3
    assert search.astar_iw(gap, features=features, landmarks="lifted").solved


@pytest.mark.parametrize("weight", [-1, math.inf, math.nan])
def test_invalid_weight_returns_failed(gap, weight):
    r = search.astar_iw(gap, weight=weight)
    assert r.status == Status.FAILED and "weight" in r.message


@pytest.mark.parametrize("features,width", [("classical", 0), ("classical", 6), ("abstracted", 4), ("base_abstracted", 4)])
def test_invalid_width_returns_failed(gap, features, width):
    r = search.astar_iw(gap, features=features, width=width)
    assert r.status == Status.FAILED and "width" in r.message


def test_invalid_features(gap):
    with pytest.raises(ValueError, match="features"):
        search.astar_iw(gap, features="unknown")


@pytest.mark.parametrize("store", ["flat", "chunked", "compact"])
def test_stores_and_native_heuristic(gap, store):
    h = search.Heuristic(gap, "max")
    native = search.astar_iw(gap, width=2)
    shared = search.astar_iw(gap, width=2, heuristic=h, store=store)
    same(native, shared)
    assert shared.store == store and shared.queue == "heap"


def test_callable_and_batch_only_heuristics(gap):
    h = search.Heuristic(gap, "max")
    same(search.astar_iw(gap, width=2), search.astar_iw(gap, width=2, heuristic=lambda s: h(s)))

    class Batch:
        def evaluate_batch(self, states):
            return [h(s) for s in states]

    assert search.astar_iw(gap, width=2, heuristic=Batch()).solved
    assert search.astar_iw(gap, heuristic=lambda s: math.inf).status == Status.UNSOLVABLE
    with pytest.raises(RuntimeError, match="heuristic failure"):
        search.astar_iw(gap, heuristic=lambda s: (_ for _ in ()).throw(RuntimeError("heuristic failure")))


def test_root_goal_exemption(gap):
    goal = lambda s: "(seed)" not in {str(a) for a in s.atoms()}
    r = search.astar_iw(gap, goal=goal, heuristic="blind")
    assert r.solved and len(r.plan) == 1
    assert search.astar_iw(gap, goal=goal, heuristic="blind", allow_non_novel_root_goal=False).status == Status.EXHAUSTED


def test_budgets_cancel_observer_and_blocked_states(gap):
    for budget in [{"max_states": 1}, {"max_expanded": 1}]:
        assert search.astar_iw(gap, width=2, **budget).status == Status.OUT_OF_STATES
    assert search.astar_iw(gap, width=2, max_seconds=0).status == Status.OUT_OF_TIME
    assert search.astar_iw(gap, width=2, max_depth=1).status == Status.EXHAUSTED
    token = search.CancelToken()
    token.request()
    assert search.astar_iw(gap, width=2, cancel=token).status == Status.CANCELLED
    events = []

    class Observer:
        def on_start(self, state):
            events.append("start")

        def on_transition(self, *args):
            events.append("transition")

        def on_solution(self, *args):
            events.append("solution")

        def on_end(self, *args):
            events.append("end")

    result = search.astar_iw(gap, width=2, observer=Observer())
    assert result.solved and events[0] == "start" and events[-1] == "end"
    assert "transition" in events and "solution" in events
    blocked = search.astar_iw(gap, width=2, blocked_states=[result.goal_state])
    assert blocked.status == Status.EXHAUSTED
    initial_goal = search.astar_iw(gap, start=result.goal_state, heuristic=lambda s: math.inf)
    assert initial_goal.solved and not initial_goal.plan


def test_threads_and_shared_heuristic(gap):
    shared = search.Heuristic(gap, "max")
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: search.astar_iw(gap, width=2, features=MODES[i % 3], heuristic=shared), range(8)))
    assert all(r.solved and r.cost == 3 for r in results)


def test_binary_features_on_a_suite_task():
    task = text_task("blocks__probBLOCKS-8-0")
    for features in MODES:
        same(search.astar_iw(task, features=features, width=2),
             search.astar_iw(task, features=features, width=2, probe_novelty_before_heuristic=False))

@pytest.mark.parametrize("numeric", [False, True])
def test_refuses_non_unit_and_numeric_tasks(numeric):
    functions = "(fuel)" if numeric else "(total-cost) - number"
    change = "(decrease (fuel) 1)" if numeric else "(increase (total-cost) 2)"
    initial = "(= (fuel) 2)" if numeric else "(= (total-cost) 0)"
    requirement = ":fluents" if numeric else ":action-costs"
    domain = mymyr.Domain.from_string(
        f"(define (domain d) (:requirements :strips {requirement}) (:predicates (ready) (goal)) "
        f"(:functions {functions}) (:action finish :parameters () :precondition (ready) "
        f":effect (and (goal) {change})))")
    formalism = domain.instantiate_string(f"(define (problem p) (:domain d) (:init (ready) {initial}) (:goal (goal)))")
    task = mymyr.Task(formalism)
    with pytest.raises(ValueError, match="numeric fluents" if numeric else "unit cost"):
        search.astar_iw(task)

@pytest.mark.parametrize("conditional", [False, True])
def test_accepts_unit_cost_expressions(conditional):
    functions = "(total-cost) - number" if conditional else "(total-cost) - number (price) - number"
    cost = "(when (ready) (increase (total-cost) 1))" if conditional else "(increase (total-cost) (price))"
    requirement = ":conditional-effects" if conditional else ""
    domain = mymyr.Domain.from_string(
        f"(define (domain d) (:requirements :strips :action-costs {requirement}) (:predicates (ready) (goal)) "
        f"(:functions {functions}) (:action finish :parameters () :precondition (ready) :effect (and (goal) {cost})))")
    initial = "" if conditional else "(= (price) 1)"
    task = mymyr.Task(domain.instantiate_string(
        f"(define (problem p) (:domain d) (:init (ready) (= (total-cost) 0) {initial}) (:goal (goal)))"))
    assert search.astar_iw(task).cost == 1
