"""mymyr.search: the search family and heuristics from Python.

Checks invariants across algorithms and the control surface: budgets, cancellation, goals, blocked states, observers and
their errors, and many threads on one task.
"""

import threading

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from conftest import ROOT, text_task

GRIPPER = ROOT / "tests/data/benchmarks/classic/strips/gripper"


def passes(r):
    return [(p.width, p.expanded, p.generated, p.generated_in_tree) for p in r.passes]


def replay(task, plan):
    s = task.initial_state
    for a in plan:
        s = task.apply(s, a)
    return s


# ------------------------------------------------------------------------------------------------ cross-algorithm


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "depot__p02", "miconic-simpleadl__s10-2",
                                  "openstacks-opt08-adl__p03"])
def test_invariants_across_algorithms(name):
    task = text_task(name)
    blind = search.astar(task, heuristic="blind")
    hmax = search.astar(task, heuristic="max")
    lazy = search.astar(task, heuristic="max", lazy=True)
    bfs = search.brfs(task)
    assert blind.status == hmax.status == lazy.status == Status.SOLVED and bfs.solved
    # unit costs: optimal cost = the shortest plan = BrFS's plan
    assert blind.cost == hmax.cost == lazy.cost == len(bfs.plan) == len(blind.plan)
    for r in (blind, hmax, lazy, search.gbfs(task), search.gbfs(task, lazy=True), search.siw(task)):
        if r.status == Status.SOLVED:
            assert task.is_goal(replay(task, r.plan))
            assert r.goal_state == replay(task, r.plan)
    # the admissible heuristics bound the optimal cost from below
    for kind in ("blind", "max"):
        assert search.Heuristic(task, kind)(task.initial_state) <= blind.cost


BRFS_STORES = {
    "flat": dict(store="flat"),
    "chunked": dict(store="chunked"),
    "compact": dict(store="compact"),
    "concurrent": dict(store="concurrent"),
    "concurrent_4": dict(store="concurrent", threads=4),
    "in_order": dict(store="flat", layer_order="in_order"),
    "beam": dict(store="flat", layer_order="in_order", beam_width=1 << 20),
}


@pytest.mark.parametrize("store", sorted(BRFS_STORES))
def test_brfs_stops_at_the_first_goal_by_default(store):
    task = text_task("depot__p02")
    shortest = search.astar(task, heuristic="blind", costs="unit")
    r = search.brfs(task, **BRFS_STORES[store])
    assert r.status == Status.SOLVED and r.solved and not r.exhausted
    assert len(r.plan) == shortest.cost and task.is_goal(replay(task, r.plan))
    assert r.stats.states < search.brfs(task, stop_at_goal=False, **BRFS_STORES[store]).stats.states


@pytest.mark.parametrize("store", sorted(BRFS_STORES))
def test_a_whole_space_brfs_that_reached_a_goal_is_solved(store):
    task = text_task("depot__p02")
    first = search.brfs(task, **BRFS_STORES[store])
    r = search.brfs(task, stop_at_goal=False, **BRFS_STORES[store])
    assert r.status == Status.SOLVED and r.solved and r.exhausted and r.num_goal_states > 0
    # the plan to the first goal state expanded: the shortest one
    assert len(r.plan) == len(first.plan) and task.is_goal(replay(task, r.plan))


@pytest.mark.parametrize("name,kw", [("blocks__probBLOCKS-8-0", {}), ("blocks__probBLOCKS-8-0", dict(max_states=40)),
                                     ("openstacks-opt08-adl__p03", dict(max_width=1))])
def test_an_unsolved_siw_has_an_empty_plan_and_a_partial_plan(name, kw):
    task = text_task(name)
    r = search.siw(task, **kw)
    assert r.status != Status.SOLVED and not r.solved
    assert r.plan == [] and r.cost == 0
    solved = [sp for sp in r.subproblems if sp.status == Status.SOLVED]
    assert len(solved) == len(r.subproblems) - 1 and solved
    assert len(r.partial_plan) == sum(sp.plan_length for sp in solved) > 0
    # the partial plan reaches a state with as many unsatisfied goal literals as the failed subproblem started with
    s, goal = replay(task, r.partial_plan), task.goal_condition
    unsatisfied = sum(not s.holds(g) for g in [*goal.fluent_literals, *goal.derived_literals])
    assert unsatisfied == r.subproblems[-1].unsatisfied_at_start
    assert "partial_plan_length=" in repr(r)
    assert search.siw(task, max_width=2, goal=lambda s: True).partial_plan == []


def test_a_whole_space_brfs_without_a_goal_is_exhausted_with_an_empty_plan():
    task = text_task("depot__p02")
    r = search.brfs(task, stop_at_goal=False, goal=lambda s: False)
    assert r.status == Status.EXHAUSTED and r.exhausted and not r.solved and r.plan == [] and r.num_goal_states == 0


@pytest.mark.parametrize("store", sorted(BRFS_STORES))
def test_brfs_takes_the_common_start_blocked_and_budget_options(store):
    task = text_task("depot__p02")
    kw = BRFS_STORES[store]
    full = search.brfs(task, **kw)
    assert full.cost == len(full.plan) and full.goal_state is not None and task.is_goal(full.goal_state)
    assert search.validate_plan(task, full.plan).cost == full.cost
    # start: a shortest plan from that state
    after = task.apply(task.initial_state, full.plan[0])
    r = search.brfs(task, start=after, **kw)
    assert r.solved and len(r.plan) == len(full.plan) - 1 and search.validate_plan(task, r.plan, start=after).valid
    # blocked states are never stored and count as pruned; blocking every successor of the start leaves it alone
    succ = task.successor_states(task.initial_state)
    r = search.brfs(task, blocked_states=succ, **kw)
    assert r.status == Status.EXHAUSTED and r.stats.states == 1 and r.stats.expanded == 1
    assert r.stats.pruned == len(task.applicable_actions(task.initial_state))
    r = search.brfs(task, blocked_states=[after], **kw)
    assert r.solved and len(r.plan) >= len(full.plan) and r.stats.pruned > 0
    # max_depth: states deeper than it are stored, not expanded
    r = search.brfs(task, max_depth=2, **kw)
    assert r.status == Status.EXHAUSTED and r.layers <= 2
    assert search.brfs(task, max_depth=len(full.plan) + 1, **kw).solved  # the goal state is expanded
    # max_expanded stops it before the next expansion, max_states at the state that fills the store: exactly
    r = search.brfs(task, max_expanded=5, **kw)
    assert r.status == Status.OUT_OF_STATES and r.stats.expanded == 5 and r.plan == [] and r.goal_state is None
    for limit in (1, 2, 50, 1000):
        r = search.brfs(task, max_states=limit, stop_at_goal=False, **kw)
        assert r.status == Status.OUT_OF_STATES and r.stats.states == limit, (limit, r.stats)


def test_every_search_result_has_the_common_fields(blocks):
    results = [search.brfs(blocks), search.astar(blocks, heuristic="max"), search.gbfs(blocks),
               search.beam(blocks, beam_width=50), search.ehc(blocks), search.anytime_astar(blocks, heuristic="ff"),
               search.iw(blocks, max_width=2), search.siw(blocks), search.iw_pass(blocks, 2),
               search.liw(blocks, max_width=2), search.abstracted_iw(blocks, width=2), search.projective_iw(blocks),
               search.rollout_iw(blocks, seed=1), search.astar_iw(blocks, width=2),
               search.atomic_goal_portfolio(blocks), *search.find_rollouts_parallel(blocks, [1], max_width=2)]
    for r in results:
        name = type(r).__name__
        assert isinstance(r.status, Status) and r.solved == (r.status == Status.SOLVED), name
        assert isinstance(r.stats, search.Statistics) and isinstance(r.message, str), name
        assert all(isinstance(getattr(r.stats, f), int) for f in ("expanded", "generated", "states", "pruned",
                                                                     "memory_bytes")), name
        assert isinstance(r.stats.seconds, float) and r.stats.seconds >= 0, name
        if r.solved:
            v = search.validate_plan(blocks, r.plan)
            assert v.valid and r.cost == v.cost and r.goal_state.atoms() == v.state.atoms(), name
        else:
            assert r.plan == [] and r.cost == 0.0 and r.goal_state is None, name
    assert sum(r.solved for r in results) >= 5 and not all(r.solved for r in results)  # both kinds


# ------------------------------------------------------------------------------------------------ control surface


@pytest.fixture(scope="module")
def gripper():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(GRIPPER / "domain.pddl", GRIPPER / "prob05.pddl")  # from PDDL: real object names


def atom_strs(state):
    return {str(a) for a in state.atoms()}


def test_budgets(gripper):
    r = search.astar(gripper, heuristic="blind", max_expanded=100)
    assert r.status == Status.OUT_OF_STATES and r.stats.expanded <= 101
    r = search.astar(gripper, heuristic="blind", max_seconds=0.0)
    assert r.status == Status.OUT_OF_TIME
    r = search.iw(gripper, max_width=2, max_states=10)
    assert r.status == Status.OUT_OF_STATES


def test_cancel_from_another_thread(gripper):
    token = search.CancelToken()
    started = threading.Event()

    class Obs:
        def on_start(self, state):
            started.set()

    t = threading.Thread(target=lambda: (started.wait(10), token.request()))
    t.start()
    r = search.brfs(gripper)  # warm-up: not cancellable, just time
    r = search.astar(gripper, heuristic="blind", cancel=token, observer=Obs(), progress_interval=1000)
    t.join()
    assert token.requested
    assert r.status in (Status.CANCELLED, Status.SOLVED)  # a fast machine may finish first
    token2 = search.CancelToken()
    token2.request()
    assert search.iw(gripper, cancel=token2).status == Status.CANCELLED


def test_custom_and_any_of_goals(gripper):
    target = "(at ball1 roomb)"
    r = search.astar(gripper, heuristic="blind", goal=lambda s: target in atom_strs(s))
    assert r.status == Status.SOLVED and target in atom_strs(r.goal_state)
    assert len(r.plan) == 3  # pick, move, drop
    r2 = search.astar(gripper, heuristic="blind", goal=[[target]])
    assert r2.status == Status.SOLVED and r2.cost == 3
    # width 2: moving back to roomb with the ball makes no new single atom true
    r3 = search.iw(gripper, max_width=2, goal=[[target, "(not (at-robby rooma))"]])
    assert r3.status == Status.SOLVED and target in atom_strs(r3.goal_state)
    assert "(at-robby rooma)" not in atom_strs(r3.goal_state)


def test_blocked_states(gripper):
    s0 = gripper.initial_state
    succ = gripper.successor_states(s0)
    r = search.iw(gripper, max_width=1, blocked_states=succ)
    assert r.passes[-1].blocked > 0
    assert r.status == Status.EXHAUSTED and r.stats.expanded <= 2  # only the start state can be expanded
    r = search.astar(gripper, heuristic="blind", blocked_states=succ)
    assert r.status != Status.SOLVED and r.stats.expanded == 1


def test_start_state(gripper):
    s1 = gripper.successor_states(gripper.initial_state)[0]
    r = search.astar(gripper, heuristic="max", start=s1)
    assert r.status == Status.SOLVED
    s = s1
    for a in r.plan:
        s = gripper.apply(s, a)
    assert gripper.is_goal(s)


def test_observer_events(gripper):
    class Obs:
        def __init__(self):
            self.n = {"start": 0, "expand": 0, "generate": 0, "pass": 0, "solution": 0, "end": 0}
            self.status = None

        def on_start(self, state):
            self.n["start"] += 1

        def on_expand(self, i, state):
            self.n["expand"] += 1

        def on_generate(self, parent, action, child, state, is_new):
            self.n["generate"] += 1

        def on_pass(self, arity, stats):
            self.n["pass"] += 1

        def on_solution(self, plan, cost):
            self.n["solution"] += 1

        def on_end(self, status, stats):
            self.n["end"] += 1
            self.status = status

    o = Obs()
    r = search.astar(gripper, heuristic="max", observer=o)
    assert o.n["start"] == o.n["end"] == o.n["solution"] == 1 and o.status == Status.SOLVED
    assert o.n["expand"] == r.stats.expanded
    assert o.n["generate"] == r.stats.generated
    o = Obs()
    r = search.iw(gripper, max_width=2, observer=o)
    assert o.n["pass"] == len(r.passes)


def test_callback_errors_propagate(gripper):
    class Boom:
        def on_expand(self, i, state):
            raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        search.astar(gripper, heuristic="blind", observer=Boom())
    with pytest.raises(ZeroDivisionError):
        search.iw(gripper, goal=lambda s: 1 / 0)
    token = search.CancelToken()
    with pytest.raises(ZeroDivisionError):
        search.gbfs(gripper, goal=lambda s: 1 / 0, cancel=token)
    assert token.requested  # an error requests the caller's token

    class Stop:
        def on_progress(self, stats):
            return False

    assert search.astar(gripper, heuristic="blind", observer=Stop(), progress_interval=10).status == Status.CANCELLED


def test_argument_errors(gripper):
    other = text_task("depot__p02")
    with pytest.raises(ValueError):
        search.astar(gripper, heuristic="nope")
    with pytest.raises(ValueError):
        search.iw(gripper, blocked_states=[other.initial_state])
    with pytest.raises(ValueError):
        search.iw(gripper, width_zero="sideways")
    with pytest.raises(ValueError):
        search.astar(gripper, heuristic=search.Heuristic(other, "max"))
    with pytest.raises(TypeError):
        search.iw(gripper, cancel=object())


def test_heuristic_object(gripper):
    h = search.Heuristic(gripper, "ff")
    s = gripper.initial_state
    assert h(s) == h.evaluate(s) > 0
    plan = h.relaxed_plan(s)
    assert len(plan) == h(s)  # unit costs: h_FF is the relaxed plan's length
    pref = h.preferred_actions(s)
    assert pref and set(map(str, pref)) <= set(map(str, gripper.applicable_actions(s)))
    h2 = search.Heuristic(gripper, "add", share=h)
    assert h2(s) >= search.Heuristic(gripper, "max")(s)
    assert h.stats.evaluations >= 3 and h.kind == "ff"
    r = search.gbfs(gripper, heuristic=h)  # a Heuristic object as the search's evaluator
    assert r.status == Status.SOLVED


def test_many_threads_one_task(gripper):
    """Free-threaded: 16 threads search one task at once; every result equals the single-threaded one."""
    ref_iw = passes(search.iw(gripper, max_width=2))
    ref_astar = search.astar(gripper, heuristic="max", max_expanded=20_000).stats.expanded
    errors = []

    def work(i):
        try:
            for _ in range(2):
                task = gripper.local() if i % 2 else gripper
                assert passes(search.iw(task, max_width=2)) == ref_iw
                assert search.astar(task, heuristic="max", max_expanded=20_000).stats.expanded == ref_astar
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_reprs_and_types(gripper):
    r = search.iw(gripper, max_width=1)
    assert "IwResult(" in repr(r) and isinstance(r.passes[0], search.IwPass)
    assert str(Status.SOLVED) == "solved"
    assert isinstance(search.siw(gripper).subproblems[0], search.SiwSubproblem)
    b = search.beam(gripper, beam_width=50)
    assert b.algorithm == "beam" and isinstance(b.stats, search.Statistics)
    assert "BrfsResult(" in repr(search.brfs(gripper, max_states=1000))


NEGATIVE_CYCLE = (
    """(define (domain d) (:requirements :strips :negative-preconditions :action-costs)
 (:predicates (a) (goal)) (:functions (total-cost))
 (:action on :parameters () :precondition (not (a)) :effect (and (a) (decrease (total-cost) 1)))
 (:action off :parameters () :precondition (a) :effect (and (not (a)) (decrease (total-cost) 1)))
 (:action finish :parameters () :precondition (a) :effect (and (goal) (increase (total-cost) 100))))""",
    """(define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal)) (:metric minimize (total-cost)))""",
)


def test_astar_refuses_negative_action_costs_and_greedy_searches_terminate():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    task = mymyr.Task(mymyr.Domain.from_string(NEGATIVE_CYCLE[0]).instantiate_string(NEGATIVE_CYCLE[1]))
    # every lap of the on/off cycle lowers the cost: A* would reopen states forever
    for r in (search.astar(task, heuristic="blind"), search.astar(task, heuristic="blind", lazy=True),
              search.astar(task, heuristic="max", weight=2), search.anytime_astar(task, heuristic="blind")):
        assert r.status == Status.FAILED and r.plan == [] and r.cost == 0.0 and r.goal_state is None
        assert "non-negative action costs" in r.message and "from 0 to -1" in r.message, r.message
        assert r.stats.expanded <= 2
    # the greedy and layered searches close each state once and end with the plan they find
    for r in (search.gbfs(task, heuristic="blind"), search.ehc(task, heuristic="blind"),
              search.beam(task, heuristic="blind"), search.brfs(task), search.iw(task)):
        assert r.solved and r.cost == search.validate_plan(task, r.plan).cost
        assert r.cost == 100 - len(r.plan) + 1
