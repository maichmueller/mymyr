"""The relaxation heuristics h², set-additive and the perfect heuristic through mymyr.search: kind names and aliases,
Heuristic.perfect over a state space, heuristic='perfect' on A* and GBFS, goals passed by the search, and one
heuristic per thread over a shared task."""

import math
import threading

import pytest

import mymyr
from mymyr import datasets, search
from mymyr.search import Status
from conftest import text_task


@pytest.fixture(scope="module")
def depot():
    return text_task("depot__p02")


@pytest.fixture(scope="module")
def depot_space(depot):
    return datasets.state_space(depot, remove_if_unsolvable=False)


def test_kind_names_and_aliases(depot):
    s = depot.initial_state
    for kind, aliases in (("set_additive", ("hsa", "setadd")), ("h2", ())):
        h = search.Heuristic(depot, kind)
        assert h.kind == kind
        for a in aliases:
            assert search.Heuristic(depot, a).kind == kind
            assert search.Heuristic(depot, a)(s) == h(s)
    hmax, hadd, hff = (search.Heuristic(depot, k)(s) for k in ("max", "add", "ff"))
    sa, h2 = search.Heuristic(depot, "set_additive")(s), search.Heuristic(depot, "h2")(s)
    assert hmax <= h2 and hff <= sa <= hadd
    with pytest.raises(ValueError, match="perfect"):
        search.Heuristic(depot, "perfect")
    with pytest.raises(ValueError):
        search.Heuristic(depot, "h2", evaluation="lifted")
    with pytest.raises(ValueError):
        search.Heuristic(depot, "set_additive", evaluation="lifted")


def test_set_additive_relaxed_plan_and_preferred_actions(depot):
    h = search.Heuristic(depot, "set_additive")
    s = depot.initial_state
    plan = h.relaxed_plan(s)
    assert 0 < len(plan) <= h(s)
    preferred = {str(a) for a in h.preferred_actions(s)}
    assert preferred and preferred <= {str(a) for a in plan}
    r = search.gbfs(depot, heuristic="set_additive", preferred_operators=True, lazy=True)
    assert r.status == Status.SOLVED


def test_h2_is_admissible_and_at_least_hmax(depot, depot_space):
    h2, hmax = search.Heuristic(depot, "h2"), search.Heuristic(depot, "max")
    for i in range(0, depot_space.num_states, max(1, depot_space.num_states // 200)):
        s = depot_space.state(i)
        d = depot_space.unit_goal_distance(i)
        v = h2(s)
        assert hmax(s) <= v
        assert (math.isinf(v) and d < 0) or (d >= 0 and v <= d)


def test_perfect_is_the_goal_distance(depot, depot_space):
    hp = search.Heuristic.perfect(depot_space)
    assert hp.kind == "perfect"
    for i in range(0, depot_space.num_states, max(1, depot_space.num_states // 200)):
        d = depot_space.unit_goal_distance(i)
        assert hp(depot_space.state(i)) == (d if d >= 0 else math.inf)
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError):
        hp(other.initial_state)
    sym = datasets.state_space(depot, remove_if_unsolvable=False, symmetry_pruning=True,
                               certificate="color_refinement")
    with pytest.raises(ValueError):
        search.Heuristic.perfect(sym)


COST_DOMAIN = """(define (domain route)
  (:requirements :strips :action-costs)
  (:predicates (at-a) (at-b) (at-goal))
  (:functions (total-cost) - number)
  (:action direct :parameters () :precondition (at-a)
    :effect (and (not (at-a)) (at-goal) (increase (total-cost) 5)))
  (:action to-b :parameters () :precondition (at-a)
    :effect (and (not (at-a)) (at-b) (increase (total-cost) 1)))
  (:action b-to-goal :parameters () :precondition (at-b)
    :effect (and (not (at-b)) (at-goal) (increase (total-cost) 1))))
"""
COST_PROBLEM = """(define (problem route-1) (:domain route)
  (:init (at-a) (= (total-cost) 0))
  (:goal (at-goal))
  (:metric minimize (total-cost)))
"""


def test_perfect_with_real_costs(tmp_path):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    (tmp_path / "domain.pddl").write_text(COST_DOMAIN)
    (tmp_path / "problem.pddl").write_text(COST_PROBLEM)
    task = mymyr.Task.from_pddl(tmp_path / "domain.pddl", tmp_path / "problem.pddl")
    space = datasets.state_space(task, remove_if_unsolvable=False)
    s = task.initial_state
    assert search.Heuristic.perfect(space)(s) == 1
    assert search.Heuristic.perfect(space, costs="real")(s) == 2
    r = search.astar(task, heuristic=search.Heuristic.perfect(space, costs="real"), costs="real")
    assert r.status == Status.SOLVED and r.cost == 2
    # set-additive with real costs: the summed costs of its supporters
    assert search.Heuristic(task, "set_additive", costs="real")(s) == 2
    assert search.Heuristic(task, "h2", costs="real")(s) == 2


@pytest.mark.parametrize("algo", [search.astar, search.gbfs])
def test_heuristic_perfect_builds_the_state_space(depot, depot_space, algo):
    optimal = depot_space.unit_goal_distance(depot_space.initial_state_id)
    r = algo(depot, heuristic="perfect")
    assert r.status == Status.SOLVED and r.cost == optimal
    # the goal is popped, not expanded: one expansion per plan step
    assert r.stats.expanded == len(r.plan)
    with pytest.raises(ValueError, match="max_states"):
        algo(depot, heuristic="perfect", max_states=100)


def test_goals_of_the_search_reach_the_heuristic(depot):
    plan = search.astar(depot, heuristic="max").plan
    # an atom true after three steps but not initially: a goal of the search instead of the task's
    s = depot.initial_state
    for a in plan[:3]:
        s = depot.apply(s, a)
    later = sorted({str(a) for a in s.atoms()} - {str(a) for a in depot.initial_state.atoms()})
    assert later
    for kind in ("set_additive", "h2"):
        r = search.astar(depot, heuristic=kind, goal=[[later[0]]])
        assert r.status == Status.SOLVED and 0 < r.cost <= 3


def test_one_heuristic_per_thread_over_a_shared_task(depot, depot_space):
    states = [depot_space.state(i) for i in range(0, depot_space.num_states, max(1, depot_space.num_states // 100))]
    hp = search.Heuristic.perfect(depot_space)
    expected = {k: [search.Heuristic(depot, k)(s) for s in states] for k in ("set_additive", "h2")}
    expected["perfect"] = [hp(s) for s in states]
    errors = []

    def run(kind):
        try:
            h = search.Heuristic.perfect(depot_space) if kind == "perfect" else search.Heuristic(depot, kind)
            for _ in range(3):
                assert [h(s) for s in states] == expected[kind]
        except Exception as e:  # noqa: BLE001
            errors.append((kind, e))

    threads = [threading.Thread(target=run, args=(k,)) for k in ("set_additive", "h2", "perfect", "set_additive", "h2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_h2_proposition_limit_reports_the_size(tmp_path):
    n = 8192
    path = tmp_path / "initial-propositions.txt"
    path.write_text(
        f"O {n}\nP 1\nF 1 p\nSI 0\nFI {n}\n"
        + "".join(f"0 {i}\n" for i in range(n))
        + "G 0\nA 0\nX 0\n"
    )
    task = mymyr.Task.from_text(str(path))
    with pytest.raises(ValueError) as caught:
        search.Heuristic(task, "h2")
    assert "8192" in str(caught.value) and "8191" in str(caught.value)
