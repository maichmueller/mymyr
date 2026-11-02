"""Search contracts on small PDDL tasks: open-list keys a bucket queue cannot hold, and the queue kinds' common order."""

import zlib

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from conftest import text_task


def pddl(domain, problem):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))


def labels(plan):
    return [str(a) for a in plan]


# ------------------------------------------------------------------------------------------------ open lists

# refund lowers total-cost by 1: GBFS's tie-breaking key g - g0 becomes -1.
REFUND = (
    """(define (domain d) (:requirements :strips :negative-preconditions :action-costs)
  (:predicates (a) (goal)) (:functions (total-cost))
  (:action refund :parameters () :precondition (not (a)) :effect (and (a) (decrease (total-cost) 1)))
  (:action finish :parameters () :precondition (a) :effect (and (goal) (increase (total-cost) 3))))""",
    "(define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal)) (:metric minimize (total-cost)))",
)


def outcome(r):
    return (r.status, labels(r.plan), r.cost, r.stats.expanded, r.stats.generated, r.stats.states, r.evaluations)


@pytest.mark.parametrize("lazy", [False, True])
def test_bucket_queue_rejects_a_negative_key(lazy):
    task = pddl(*REFUND)
    with pytest.raises(ValueError, match="queue='heap'") as e:
        search.gbfs(task, heuristic="blind", queue="bucket", lazy=lazy)
    assert "Queue::" not in str(e.value) and "BestFirstOptions" not in str(e.value)


@pytest.mark.parametrize("lazy", [False, True])
def test_auto_queue_moves_into_a_heap_at_a_negative_key(lazy):
    task = pddl(*REFUND)
    auto = search.gbfs(task, heuristic="blind", lazy=lazy)
    heap = search.gbfs(task, heuristic="blind", queue="heap", lazy=lazy)
    assert auto.status == Status.SOLVED and labels(auto.plan) == ["(refund)", "(finish)"]
    assert auto.queue == "heap"
    assert outcome(auto) == outcome(heap)


def fractional_late(state):
    """Integral values for most states, a fractional value for some: an auto bucket queue moves into a heap with many
    entries in it."""
    k = zlib.crc32(str(state).encode())
    return (k % 9) + (0.5 if k % 23 == 0 else 0.0)


@pytest.mark.parametrize(
    "run",
    [
        lambda t, **kw: search.astar(t, **kw),
        lambda t, **kw: search.astar(t, lazy=True, **kw),
        lambda t, **kw: search.gbfs(t, **kw),
        lambda t, **kw: search.gbfs(t, lazy=True, **kw),
    ],
    ids=["astar", "astar_lazy", "gbfs", "gbfs_lazy"],
)
def test_auto_queue_pops_in_the_heap_order_across_the_move(run):
    task = text_task("gripper__prob05")
    auto = run(task, heuristic=fractional_late, max_expanded=3000)
    heap = run(task, heuristic=fractional_late, queue="heap", max_expanded=3000)
    assert auto.queue == "heap"
    assert outcome(auto) == outcome(heap)
    with pytest.raises(ValueError, match="queue='heap'"):
        run(task, heuristic=fractional_late, queue="bucket", max_expanded=3000)
    # integral values stay in buckets, in the same order
    integral = lambda s: zlib.crc32(str(s).encode()) % 9  # noqa: E731
    auto = run(task, heuristic=integral, max_expanded=3000)
    heap = run(task, heuristic=integral, queue="heap", max_expanded=3000)
    bucket = run(task, heuristic=integral, queue="bucket", max_expanded=3000)
    assert auto.queue == "bucket" and bucket.queue == "bucket"
    assert outcome(auto) == outcome(heap) == outcome(bucket)
