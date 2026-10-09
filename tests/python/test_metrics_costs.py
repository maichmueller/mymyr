"""Plan objectives and heuristic costs: the metrics mymyr refuses, undefined metric values, and heuristics whose action
costs follow the task's objective unless `costs=` overrides them."""

import math

import pytest

import mymyr
import mymyr.datasets
from mymyr import search

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

# two actions of 0.25 and 0.5 reach the goal for 0.75, one action reaches it for 1
DOMAIN = """
(define (domain d) (:requirements :strips :negative-preconditions :numeric-fluents :action-costs)
  (:predicates (half) (done))
  (:functions (x) (y) (total-cost))
  (:action direct :parameters () :precondition (not (done))
    :effect (and (done) (increase (total-cost) 1) (increase (x) 4)))
  (:action first :parameters () :precondition (not (half))
    :effect (and (half) (increase (total-cost) 0.25) (increase (x) 1)))
  (:action second :parameters () :precondition (and (half) (not (done)))
    :effect (and (done) (increase (total-cost) 0.5) (increase (x) 1)))
  (:action zero :parameters () :precondition (and) :effect (assign (y) 0)))
"""


def task(metric, init="(= (x) 0) (= (y) 1) (= (total-cost) 0)"):
    problem = f"(define (problem p) (:domain d) (:init {init}) (:goal (done)) {metric})"
    return mymyr.Task(mymyr.Domain.from_string(DOMAIN).instantiate_string(problem))


def plan_names(result):
    return [str(a) for a in result.plan]


@pytest.mark.parametrize("heuristic", ["blind", "max", "h2", "perfect"])
def test_astar_is_optimal_with_fractional_costs_by_default(heuristic):
    r = search.astar(task("(:metric minimize (total-cost))"), heuristic=heuristic)
    assert r.solved and math.isclose(r.cost, 0.75)
    assert plan_names(r) == ["(first)", "(second)"]


def test_heuristic_costs_follow_the_objective():
    t = task("(:metric minimize (total-cost))")
    s = t.initial_state
    for kind in ("max", "h2"):
        assert search.Heuristic(t, kind)(s) == pytest.approx(0.75)
        assert search.Heuristic(t, kind, costs="real")(s) == pytest.approx(0.75)
        assert search.Heuristic(t, kind, costs="unit")(s) == 1
    space = mymyr.datasets.state_space(t, remove_if_unsolvable=False)
    assert search.Heuristic.perfect(space)(s) == pytest.approx(0.75)
    assert search.Heuristic.perfect(space, costs="unit")(s) == 1
    # a metric over the fluents: the relaxation counts every action 0
    m = task("(:metric minimize (x))")
    assert search.Heuristic(m, "max")(m.initial_state) == 0
    r = search.astar(m, heuristic="max")
    assert r.solved and r.cost == 2 and plan_names(r) == ["(first)", "(second)"]


def test_the_metric_is_the_objective():
    # the metric names x: the total-cost effects do not count
    r = search.astar(task("(:metric minimize (x))"))
    assert r.solved and r.cost == 2
    r = search.astar(task("(:metric minimize (+ (x) 1))"))
    assert r.solved and r.cost == 3
    # without a metric, total-cost is the objective
    r = search.astar(task(""))
    assert r.solved and math.isclose(r.cost, 0.75)


@pytest.mark.parametrize(
    "metric, why",
    [
        ("(:metric maximize (x))", "(:metric maximize (x)) is not supported"),
        ("(:metric minimize (+ (total-cost) (x)))", "total-cost can only be the whole metric"),
    ],
)
def test_refused_metrics_name_the_metric(metric, why):
    t = task(metric)
    for r in (search.astar(t), search.gbfs(t), search.astar(t, heuristic="max")):
        assert r.status == search.Status.FAILED and why in r.message
    with pytest.raises(ValueError, match=why.replace("(", r"\(").replace(")", r"\)").replace("+", r"\+")):
        mymyr.datasets.state_space(t, remove_if_unsolvable=False)
    with pytest.raises(ValueError):
        search.Heuristic(t, "max")


def test_an_undefined_metric_is_an_error():
    start = task("(:metric minimize (y))", init="(= (x) 0) (= (total-cost) 0)")
    r = search.astar(start)
    assert r.status == search.Status.FAILED and "(:metric minimize (y)) is undefined in the start state" in r.message
    with pytest.raises(ValueError, match="is undefined in the start state"):
        mymyr.datasets.state_space(start, remove_if_unsolvable=False)
    # zero sets y to 0: the metric divides by zero in the states it reaches
    reached = task("(:metric minimize (/ (x) (y)))")
    r = search.astar(reached, heuristic="blind")
    assert r.status == search.Status.FAILED and "is undefined in a reached state" in r.message
    with pytest.raises(ValueError, match="is undefined in a reached state"):
        mymyr.datasets.state_space(reached, remove_if_unsolvable=False)


def test_invalid_costs_are_rejected():
    t = task("(:metric minimize (total-cost))")
    with pytest.raises(ValueError, match="costs must be 'auto', 'unit' or 'real'"):
        search.astar(t, costs="metric")
