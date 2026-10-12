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
    return [(p.arity, p.expanded, p.generated, p.generated_in_tree) for p in r.passes]


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
    assert r.states < search.brfs(task, stop_at_goal=False, **BRFS_STORES[store]).states


@pytest.mark.parametrize("store", sorted(BRFS_STORES))
def test_a_whole_space_brfs_that_reached_a_goal_is_solved(store):
    task = text_task("depot__p02")
    first = search.brfs(task, **BRFS_STORES[store])
    r = search.brfs(task, stop_at_goal=False, **BRFS_STORES[store])
    assert r.status == Status.SOLVED and r.solved and r.exhausted and r.goal_states > 0
    # the plan to the first goal state expanded: the shortest one
    assert len(r.plan) == len(first.plan) and task.is_goal(replay(task, r.plan))


@pytest.mark.parametrize("name,kw", [("blocks__probBLOCKS-8-0", {}), ("blocks__probBLOCKS-8-0", dict(max_states=40)),
                                     ("openstacks-opt08-adl__p03", dict(max_arity=1))])
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
    assert search.siw(task, max_arity=2, goal=lambda s: True).partial_plan == []


def test_a_whole_space_brfs_without_a_goal_is_exhausted_with_an_empty_plan():
    task = text_task("depot__p02")
    r = search.brfs(task, stop_at_goal=False, goal=lambda s: False)
    assert r.status == Status.EXHAUSTED and r.exhausted and not r.solved and r.plan == [] and r.goal_states == 0


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
    r = search.iw(gripper, max_arity=2, max_states=10)
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
    r3 = search.iw(gripper, max_arity=2, goal=[[target, "(not (at-robby rooma))"]])
    assert r3.status == Status.SOLVED and target in atom_strs(r3.goal_state)
    assert "(at-robby rooma)" not in atom_strs(r3.goal_state)


def test_blocked_states(gripper):
    s0 = gripper.initial_state
    succ = gripper.successor_states(s0)
    r = search.iw(gripper, max_arity=1, blocked_states=succ)
    assert r.passes[-1].blocked > 0
    assert r.status == Status.EXHAUSTED and r.total.expanded <= 2  # only the start state can be expanded
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
    r = search.iw(gripper, max_arity=2, observer=o)
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
    ref_iw = passes(search.iw(gripper, max_arity=2))
    ref_astar = search.astar(gripper, heuristic="max", max_expanded=20_000).stats.expanded
    errors = []

    def work(i):
        try:
            for _ in range(2):
                task = gripper.local() if i % 2 else gripper
                assert passes(search.iw(task, max_arity=2)) == ref_iw
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
    r = search.iw(gripper, max_arity=1)
    assert "IwResult(" in repr(r) and isinstance(r.passes[0], search.IwPass)
    assert str(Status.SOLVED) == "solved"
    assert isinstance(search.siw(gripper).subproblems[0], search.SiwSubproblem)
    b = search.beam(gripper, width=50)
    assert b.algorithm == "beam" and isinstance(b.stats, search.Statistics)
    assert "BrfsResult(" in repr(search.brfs(gripper, max_states=1000))
