"""Search contracts on small PDDL tasks: callable goals and heuristic dead ends, open-list keys a bucket queue cannot
hold, and the queue kinds' common order."""

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


# ------------------------------------------------------------------------------------------------ callable goals

# The task goal (goal) is unreachable (never is false) while a callable goal on (b) is one action away.
UNREACHABLE_TASK_GOAL = (
    """(define (domain d) (:requirements :strips)
  (:predicates (b) (never) (goal))
  (:action go-b :parameters () :precondition (and) :effect (b))
  (:action finish :parameters () :precondition (never) :effect (goal)))""",
    "(define (problem p) (:domain d) (:init) (:goal (goal)))",
)

# (b) holds only after go-trap, from where the task goal is unreachable: every state with (b) has h = +inf.
TRAP = (
    """(define (domain d) (:requirements :strips :negative-preconditions)
  (:predicates (a) (b) (trap) (goal))
  (:action go-trap :parameters () :precondition (not (trap)) :effect (and (trap) (b)))
  (:action go-a :parameters () :precondition (not (trap)) :effect (a))
  (:action finish :parameters () :precondition (and (a) (not (trap))) :effect (goal)))""",
    "(define (problem p) (:domain d) (:init) (:goal (goal)))",
)


def holds_b(state):
    return "(b)" in str(state)


CALLABLE_GOAL_SEARCHES = [
    ("astar", lambda t, **kw: search.astar(t, **kw)),
    ("astar_lazy", lambda t, **kw: search.astar(t, lazy=True, **kw)),
    ("gbfs", lambda t, **kw: search.gbfs(t, **kw)),
    ("gbfs_lazy", lambda t, **kw: search.gbfs(t, lazy=True, **kw)),
    ("ehc", lambda t, **kw: search.ehc(t, **kw)),
    ("beam", lambda t, **kw: search.beam(t, **kw)),
    ("anytime_astar", lambda t, **kw: search.anytime_astar(t, **kw)),
    ("astar_iw", lambda t, **kw: search.astar_iw(t, **kw)),
]


@pytest.mark.parametrize("heuristic", ["blind", "max", "ff", "lmcut"])
@pytest.mark.parametrize("name,run", CALLABLE_GOAL_SEARCHES, ids=[n for n, _ in CALLABLE_GOAL_SEARCHES])
def test_infinite_heuristic_values_do_not_prune_under_a_callable_goal(name, run, heuristic):
    # the heuristic estimates the task's goal: its infinite values prove nothing about the callable goal
    task = pddl(*UNREACHABLE_TASK_GOAL)
    r = run(task, heuristic=heuristic, goal=holds_b)
    assert r.status == Status.SOLVED, (name, heuristic, r.status)
    assert labels(r.plan) == ["(go-b)"]
    trap = pddl(*TRAP)
    r = run(trap, heuristic=heuristic, goal=holds_b)
    assert r.status == Status.SOLVED, (name, heuristic, r.status)
    assert holds_b(r.goal_state)
    # the task's goal keeps the dead-end proof
    r = run(task, heuristic=heuristic)
    assert r.status in ((Status.EXHAUSTED,) if heuristic == "blind" else (Status.UNSOLVABLE,)), (name, r.status)


@pytest.mark.parametrize("heuristic", ["max", "ff", "lmcut"])
def test_callable_goal_never_reports_unsolvable(heuristic):
    task = pddl(*UNREACHABLE_TASK_GOAL)
    for run in (search.astar, search.gbfs, search.beam, search.anytime_astar):
        r = run(task, heuristic=heuristic, goal=lambda s: False)
        assert r.status == Status.EXHAUSTED


@pytest.mark.parametrize("heuristic", ["max", "ff"])
def test_anytime_optimality_under_a_callable_goal_rests_on_exhaustion(heuristic):
    task = pddl(*TRAP)
    r = search.anytime_astar(task, heuristic=heuristic, goal=holds_b)
    assert r.status == Status.SOLVED
    assert labels(r.plan) == ["(go-trap)"]  # the cheapest plan; optimal only because the last iteration exhausted
    assert r.optimal
    assert r.iterations[-1].status == Status.EXHAUSTED


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


def test_auto_queue_takes_infinite_values_under_a_callable_goal():
    task = pddl(*TRAP)
    r = search.astar(task, heuristic="max", goal=holds_b)
    assert r.status == Status.SOLVED and r.queue == "heap"
    with pytest.raises(ValueError, match="queue='heap'"):
        search.astar(task, heuristic="max", goal=holds_b, queue="bucket")


# ------------------------------------------------------------------------------------------------ observer events


class EventLog(search.Observer):
    def __init__(self):
        self.expanded = 0
        self.generated = 0
        self.pruned = 0
        self.last_generate = None
        self.unpaired = 0

    def on_expand(self, id, state):
        self.expanded += 1

    def on_generate(self, parent, action, child, state, is_new):
        self.generated += 1
        self.last_generate = (parent, str(action), state)

    def on_prune(self, parent, action, state):
        self.pruned += 1
        if self.last_generate != (parent, str(action), state):
            self.unpaired += 1
        self.last_generate = None


OBSERVED_SEARCHES = [
    ("astar", lambda t, **kw: search.astar(t, heuristic="blind", **kw)),
    ("astar_lazy", lambda t, **kw: search.astar(t, heuristic="ff", lazy=True, **kw)),
    ("gbfs", lambda t, **kw: search.gbfs(t, heuristic="ff", **kw)),
    ("gbfs_lazy", lambda t, **kw: search.gbfs(t, heuristic="ff", lazy=True, **kw)),
    ("beam", lambda t, **kw: search.beam(t, **kw)),
    ("ehc", lambda t, **kw: search.ehc(t, **kw)),
    ("anytime_astar", lambda t, **kw: search.anytime_astar(t, **kw)),
    ("brfs", lambda t, **kw: search.brfs(t, **kw)),
    ("brfs_compact", lambda t, **kw: search.brfs(t, store="compact", **kw)),
    ("iw", lambda t, **kw: search.iw(t, **kw)),
    ("siw", lambda t, **kw: search.siw(t, **kw)),
    ("astar_iw", lambda t, **kw: search.astar_iw(t, **kw)),
    ("rollout_iw", lambda t, **kw: search.rollout_iw(t, max_rollouts=20, **kw)),
    ("liw", lambda t, **kw: search.liw(t, **kw)),
    ("abstracted_iw", lambda t, **kw: search.abstracted_iw(t, **kw)),
]


@pytest.mark.parametrize("name,run", OBSERVED_SEARCHES, ids=[n for n, _ in OBSERVED_SEARCHES])
def test_observer_events_match_the_statistics(name, run):
    task = text_task("gripper__prob05")
    s0 = task.initial_state
    blocked = task.successor_states(s0)[:2]
    budgets = [{"max_expanded": 500}, {"max_expanded": 500, "blocked_states": blocked}, {"max_states": 40},
               {"max_states": 40, "blocked_states": blocked}, {"max_expanded": 7, "blocked_states": blocked}]
    if name == "rollout_iw":  # budgets of its own (max_rollouts)
        budgets = [{}, {"blocked_states": blocked}]
    for kw in budgets:
        log = EventLog()
        r = run(task, observer=log, **kw)
        assert (log.expanded, log.generated) == (r.stats.expanded, r.stats.generated), (name, kw, r.status)
        assert log.unpaired == 0, (name, kw)
        if "blocked_states" in kw and name != "rollout_iw":
            assert log.pruned > 0, (name, kw)
