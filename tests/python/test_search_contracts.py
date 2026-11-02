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


# ------------------------------------------------------------------------------------------------ BrFS threads


def brfs_outcome(r):
    return (r.status, labels(r.plan), r.stats.expanded, r.stats.generated, r.stats.pruned, r.stats.states,
            r.num_goal_states, r.exhausted, r.solved)


BRFS_CASES = [
    ("visitall", "tests/data/pddl/visitall/domain.pddl", "tests/data/pddl/visitall/test_problem.pddl"),
    ("goals", "tests/data/pddl/goals/domain.pddl", "tests/data/pddl/goals/exists_single_witness.pddl"),
    ("gripper", None, "gripper__prob05"),
    ("blocks", None, "blocks__probBLOCKS-8-0"),
]


def brfs_task(domain, problem):
    from conftest import ROOT

    if domain is None:
        return text_task(problem)
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(ROOT / domain, ROOT / problem)


@pytest.mark.parametrize("name,domain,problem", BRFS_CASES, ids=[c[0] for c in BRFS_CASES])
def test_brfs_thread_count_does_not_change_the_result(name, domain, problem):
    task = brfs_task(domain, problem)
    s0 = task.initial_state
    blocked = task.successor_states(s0)[:1]
    budgets = [{}, {"max_states": 50}, {"max_states": 2}, {"max_states": 137}, {"max_states": 1000},
               {"max_expanded": 1}, {"max_expanded": 45}, {"max_expanded": 300},
               {"max_states": 400, "max_expanded": 100}, {"stop_at_goal": False, "max_states": 3000},
               {"stop_at_goal": False, "max_expanded": 500}, {"max_states": 300, "blocked_states": blocked},
               {"max_depth": 3, "stop_at_goal": False}]
    for kw in budgets:
        want = brfs_outcome(search.brfs(task, **kw))
        for threads in (2, 4, 8):
            got = brfs_outcome(search.brfs(task, threads=threads, **kw))
            assert got == want, (name, kw, threads)


def test_brfs_without_worker_observers_keeps_the_result():
    # an observer without make_worker runs the search on one thread: the result is the same as with workers
    task = text_task("gripper__prob05")

    class Plain(search.Observer):
        def on_expand(self, id, state):
            pass

    class Workers(Plain):
        def make_worker(self, k):
            return Plain()

    for kw in [{"max_states": 137}, {"max_expanded": 45}, {}]:
        plain = search.brfs(task, threads=4, observer=Plain(), **kw)
        workers = search.brfs(task, threads=4, observer=Workers(), **kw)
        assert plain.threads_used == 1 and workers.threads_used == 4
        assert brfs_outcome(plain) == brfs_outcome(workers) == brfs_outcome(search.brfs(task, **kw)), kw


class EventList(search.Observer):
    def __init__(self, sink=None):
        self.sink = [] if sink is None else sink

    def on_expand(self, id, state):
        self.sink.append(("expand", id, str(state)))

    def on_generate(self, parent, action, child, state, is_new):
        self.sink.append(("generate", parent, str(action), str(state), is_new))

    def on_prune(self, parent, action, state):
        self.sink.append(("prune", parent, str(action), str(state)))

    def make_worker(self, k):
        return EventList(self.sink)  # list.append is atomic: the workers share one list


def test_brfs_on_one_thread_of_the_concurrent_store_sends_the_serial_events():
    # what an observer without make_worker gets from brfs(threads > 1)
    threads = 1
    task = text_task("gripper__prob05")
    blocked = task.successor_states(task.initial_state)[:1]
    for kw in [{"max_states": 137}, {"max_expanded": 45}, {"max_states": 300, "blocked_states": blocked},
               {"stop_at_goal": False, "max_states": 500}]:
        serial = EventList()
        want = search.brfs(task, observer=serial, **kw)
        log = EventList()
        r = search.brfs(task, threads=threads, store="concurrent", observer=log, **kw)
        assert brfs_outcome(r) == brfs_outcome(want), kw
        assert sorted(log.sink) == sorted(serial.sink), kw
        assert sum(e[0] == "generate" for e in log.sink) == r.stats.generated
        assert sum(e[0] == "expand" for e in log.sink) == r.stats.expanded


# ------------------------------------------------------------------------------------------------ static goals

STATICALLY_FALSE_GOALS = ["static_false", "contradiction", "or_all_static_false", "forall_static_guard_unsatisfiable",
                          "not_imply_false", "forall_equality"]

UNSOLVABLE_SEARCHES = [
    ("brfs", lambda t: search.brfs(t)),
    ("brfs_threads", lambda t: search.brfs(t, threads=2)),
    ("brfs_compact", lambda t: search.brfs(t, store="compact")),
    ("astar", lambda t: search.astar(t, heuristic="blind")),
    ("gbfs", lambda t: search.gbfs(t, heuristic="ff")),
    ("ehc", lambda t: search.ehc(t)),
    ("beam", lambda t: search.beam(t)),
    ("iw", lambda t: search.iw(t)),
    ("siw", lambda t: search.siw(t)),
]


@pytest.mark.parametrize("problem", STATICALLY_FALSE_GOALS)
def test_statically_false_goals_are_unsolvable_at_once(problem):
    from conftest import ROOT

    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    goals = ROOT / "tests/data/pddl/goals"
    task = mymyr.Task.from_pddl(goals / "domain.pddl", goals / f"{problem}.pddl")
    for name, run in UNSOLVABLE_SEARCHES:
        r = run(task)
        assert r.status == Status.UNSOLVABLE, (name, r.status)
        assert r.stats.expanded == 0, name
    assert not any(task.is_goal(s) for s in [task.initial_state, *task.successor_states(task.initial_state)])


# ------------------------------------------------------------------------------------------------ derived goals

DERIVED_GOAL_DOMAIN = """(define (domain dg) (:requirements :strips :typing :derived-predicates :negative-preconditions)
  (:types obj)
  (:predicates (p) (q ?x - obj) (r ?x - obj) (s ?x - obj) (dq ?x - obj))
  (:derived (dq ?x - obj) (and (q ?x) (r ?x)))
  (:action start :parameters () :precondition (not (p)) :effect (p))
  (:action make-s :parameters (?x - obj) :precondition (p) :effect (s ?x))
  (:action make-q :parameters (?x - obj) :precondition (s ?x) :effect (q ?x))
  (:action make-r :parameters (?x - obj) :precondition (and (p) (q ?x)) :effect (r ?x)))"""


def derived_goal_problem(goal):
    return f"(define (problem dg1) (:domain dg) (:objects o1 o2 - obj) (:init) (:goal {goal}))"


@pytest.mark.parametrize(
    "heuristic,evaluation",
    [(h, "grounded") for h in ["goal_count", "max", "add", "ff", "lmcut", "h2"]] + [(h, "lifted") for h in ["max", "add", "ff"]],
)
def test_derived_atoms_of_goal_alternatives_are_estimated_like_the_task_goal(heuristic, evaluation):
    task = pddl(DERIVED_GOAL_DOMAIN, derived_goal_problem("(and (q o2))"))
    as_goal = pddl(DERIVED_GOAL_DOMAIN, derived_goal_problem("(dq o1)"))
    h = search.Heuristic(as_goal, heuristic, evaluation=evaluation)
    # every reachable state of task with its twin in as_goal (the same actions from the initial state)
    pairs = {str(task.initial_state): (task.initial_state, as_goal.initial_state)}
    frontier = list(pairs.values())
    while frontier:
        nxt = []
        for s, twin in frontier:
            twins = {str(a): u for a, u in as_goal.successors(twin)}
            for a, u in task.successors(s):
                if str(u) not in pairs:
                    pairs[str(u)] = (u, twins[str(a)])
                    nxt.append(pairs[str(u)])
        frontier = nxt
    checked = 0
    for s, twin in pairs.values():
        assert str(s) == str(twin)
        alternative = search.astar(task, start=s, heuristic=heuristic, evaluation=evaluation,
                                   goal=[["(dq o1)"]]).initial_h
        if not as_goal.is_goal(twin):
            assert alternative == h(twin), (str(s), heuristic)
            checked += 1
    assert checked > 5


# ------------------------------------------------------------------------------------------------ landmark count

# a3 makes (p) true through a conditional effect: the landmark (s c c) or (s o1 o1) of the goal is false in the goal
# state, so the goal state alone does not show that a path to it reached it.
LANDMARK_GOAL_TASK = (
    """(define (domain d) (:requirements :adl :derived-predicates :action-costs) (:constants c)
  (:predicates (p) (t) (q ?x) (r ?x) (s ?x ?y) (dq ?x)) (:functions (total-cost))
  (:derived (dq ?x0) (or (r ?x0) (and (p) (not (p)))))
  (:action a0 :parameters (?x0) :precondition (and (q ?x0) (p))
    :effect (and (q ?x0) (p) (s ?x0 ?x0) (when (q ?x0) (r ?x0)) (increase (total-cost) 1)))
  (:action a1 :parameters (?x0) :precondition (and (r ?x0) (not (dq ?x0)))
    :effect (and (p) (not (p)) (q ?x0) (increase (total-cost) 2)))
  (:action a2 :parameters (?x0 ?x1) :precondition (and (not (dq ?x0)) (not (r ?x1)))
    :effect (and (s ?x0 ?x1) (increase (total-cost) 2)))
  (:action a3 :parameters (?x0) :precondition (and (s ?x0 ?x0) (not (p)))
    :effect (and (q ?x0) (not (p)) (not (s ?x0 ?x0)) (when (s ?x0 ?x0) (p)) (increase (total-cost) 0))))""",
    """(define (problem pr) (:domain d) (:objects o1) (:init (q c) (q o1) (= (total-cost) 0))
  (:goal (and (p) (p))) (:metric minimize (total-cost)))""",
)


@pytest.mark.parametrize("costs", ["unit", "real"])
def test_landmark_count_is_zero_on_a_goal_state_alone(costs):
    task = pddl(*LANDMARK_GOAL_TASK)
    h = search.Heuristic(task, "lmcount", costs=costs)
    seen = {str(task.initial_state): task.initial_state}
    frontier = [task.initial_state]
    while frontier:
        frontier = [u for s in frontier for u in task.successor_states(s) if str(u) not in seen and not seen.update({str(u): u})]
    goals = [s for s in seen.values() if task.is_goal(s)]
    assert goals
    for s in goals:
        assert h(s) == 0, str(s)
