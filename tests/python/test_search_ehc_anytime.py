"""Weighted A*, enforced hill-climbing and the anytime search (restarting weighted A*) from Python.

Checks the suboptimality bound of weighted A* and its cost bound, enforced hill-climbing's helpful actions, fallback
and statistics, the anytime search's improving plans, the argument refusals, observers, pickling and threads.
"""

import math
import pickle
import threading

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from conftest import ROOT, text_task

# small unit-cost tasks (text tasks), and PDDL tasks with action costs
UNIT = ["logistics00__probLOGISTICS-6-1", "depot__p02", "miconic__s7-4", "zenotravel__p05", "rovers__p02",
        "driverlog__p03"]
COSTS = [ROOT / "tests/data/benchmarks/small" / d for d in ("transport", "woodworking")]


def replay(task, plan):
    s = task.initial_state
    for a in plan:
        assert a in task.applicable_actions(s)
        s = task.apply(s, a)
    return s


def check_plan(task, r):
    assert r.status == Status.SOLVED, r.message
    assert task.is_goal(replay(task, r.plan))
    assert r.goal_state == replay(task, r.plan)


def load(name):
    if isinstance(name, str):
        return text_task(name)
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(name / "domain.pddl", name / "test_problem.pddl")


@pytest.fixture(scope="module", params=UNIT + COSTS, ids=lambda x: x if isinstance(x, str) else x.name)
def optimal(request):
    task = load(request.param)
    r = search.astar(task, heuristic="lmcut")
    assert r.status == Status.SOLVED
    return task, r


# ------------------------------------------------------------------------------------------------ weighted A*


@pytest.mark.parametrize("lazy", [False, True])
def test_weighted_astar_stays_within_the_bound(optimal, lazy):
    task, opt = optimal
    for heuristic in ("lmcut", "max"):
        for w in (1.5, 2, 5):
            r = search.astar(task, heuristic=heuristic, weight=w, lazy=lazy)
            check_plan(task, r)
            assert opt.cost <= r.cost <= w * opt.cost + 1e-9, (heuristic, w)
            assert r.algorithm == ("astar_lazy" if lazy else "astar_eager")
    # weight 1 is A*
    a = search.astar(task, heuristic="max", lazy=lazy)
    one = search.astar(task, heuristic="max", lazy=lazy, weight=1)
    assert (one.cost, one.stats.expanded, one.stats.generated, one.plan) == (a.cost, a.stats.expanded,
                                                                            a.stats.generated, a.plan)


def test_weighted_astar_without_reopening_and_with_integral_weights():
    task = text_task("depot__p02")
    opt = search.astar(task, heuristic="lmcut").cost
    for queue in ("auto", "bucket", "heap"):
        r = search.astar(task, heuristic="max", weight=3, reopen=False, queue=queue)
        check_plan(task, r)
        assert r.cost <= 3 * opt
    assert search.astar(task, heuristic="ff", weight=2, queue="bucket").plan == \
        search.astar(task, heuristic="ff", weight=2, queue="heap").plan


def test_cost_bound():
    task = text_task("logistics00__probLOGISTICS-6-1")
    opt = search.astar(task, heuristic="lmcut")
    tight = search.astar(task, heuristic="lmcut", cost_bound=opt.cost)
    assert tight.status == Status.EXHAUSTED and tight.plan == []
    blind = search.astar(task, heuristic="blind", cost_bound=opt.cost)
    assert blind.status == Status.EXHAUSTED and blind.stats.expanded > tight.stats.expanded
    loose = search.astar(task, heuristic="lmcut", cost_bound=opt.cost + 1)
    check_plan(task, loose)
    assert loose.cost == opt.cost
    # an inadmissible heuristic prunes by g only, and finds a plan below the bound
    ff = search.astar(task, heuristic="ff", weight=2, cost_bound=opt.cost + 1)
    check_plan(task, ff)
    assert ff.cost <= opt.cost


def test_weight_and_bound_refusals():
    task = text_task("gripper__prob05")
    for w in (0.5, 0, -1, math.nan):
        with pytest.raises(ValueError, match="weight"):
            search.astar(task, weight=w)
    with pytest.raises(ValueError, match="finite"):
        search.astar(task, weight=math.inf)
    with pytest.raises(TypeError, match="weight"):
        search.astar(task, weight="2")
    with pytest.raises(ValueError, match="bucket"):
        search.astar(task, weight=1.5, queue="bucket")
    with pytest.raises(ValueError, match="cost_bound"):
        search.astar(task, cost_bound=math.nan)
    with pytest.raises(TypeError):
        search.gbfs(task, weight=2)


# ------------------------------------------------------------------------------------------------ enforced hill-climbing


@pytest.mark.parametrize("name", UNIT + COSTS, ids=lambda x: x if isinstance(x, str) else x.name)
def test_ehc_plans_replay(name):
    task = load(name)
    for helpful in ("auto", "first", "only", "off"):
        r = search.ehc(task, helpful_actions=helpful)
        check_plan(task, r)
        assert isinstance(r, search.EhcResult) and r.algorithm == "ehc"
        e = r.ehc
        assert e.lookaheads >= e.improvements + e.retries
        assert e.expanded <= r.stats.expanded and e.evaluations <= r.evaluations
        assert e.fallback == e.failed  # the default fallback runs when hill-climbing fails
        if helpful == "off":
            assert e.retries == 0
    r = search.ehc(task, heuristic="add", helpful_actions="off", fallback="none")
    assert r.status in (Status.SOLVED, Status.EXHAUSTED)
    if r.status == Status.SOLVED:
        check_plan(task, r)
    else:
        assert r.ehc.failed and not r.ehc.fallback and r.plan == []


def test_ehc_helpful_actions_need_preferred_operators():
    task = text_task("depot__p02")
    for helpful in ("first", "only"):
        r = search.ehc(task, heuristic="max", helpful_actions=helpful)
        assert r.status == Status.FAILED and "preferred operators" in r.message
    auto = search.ehc(task, heuristic="max")  # auto: every action
    check_plan(task, auto)
    assert auto.ehc.retries == 0
    with pytest.raises(ValueError, match="helpful_actions"):
        search.ehc(task, helpful_actions="sometimes")
    with pytest.raises(ValueError, match="fallback"):
        search.ehc(task, fallback="astar")


def test_ehc_with_path_dependent_and_python_heuristics():
    task = text_task("logistics00__probLOGISTICS-6-1")
    check_plan(task, search.ehc(task, heuristic="lmcount"))
    ff = search.Heuristic(task, "ff")

    class Ff:  # h_FF with its helpful actions, written in Python
        def __init__(self):
            self.state = None

        def evaluate(self, state):
            self.state = state
            return ff(state)

        def preferred_actions(self):
            return ff.preferred_actions(self.state)

    native = search.ehc(task)
    python = search.ehc(task, heuristic=Ff())
    check_plan(task, python)
    assert python.plan == native.plan
    assert python.ehc.lookaheads == native.ehc.lookaheads


def test_ehc_and_anytime_observers_count_every_event():
    task = text_task("depot__p02")

    class Counter(search.Observer):
        def __init__(self):
            self.expanded = self.generated = 0
            self.passes, self.solutions, self.ends = [], [], []

        def on_expand(self, id, state):
            self.expanded += 1

        def on_generate(self, parent, action, child, state, is_new):
            self.generated += 1

        def on_pass(self, index, stats):
            self.passes.append((index, stats.expanded))

        def on_solution(self, plan, cost):
            self.solutions.append(cost)

        def on_end(self, status, stats):
            self.ends.append(status)

    for kwargs in ({}, {"heuristic": "add", "helpful_actions": "off"}, {"helpful_actions": "only"}):
        c = Counter()
        r = search.ehc(task, observer=c, **kwargs)
        check_plan(task, r)
        assert (c.expanded, c.generated) == (r.stats.expanded, r.stats.generated)
        assert [i for i, _ in c.passes] == list(range(r.ehc.lookaheads))
        assert c.solutions == [r.cost] and c.ends == [Status.SOLVED]
    c = Counter()
    r = search.anytime_astar(task, heuristic="ff", observer=c)
    assert (c.expanded, c.generated) == (r.stats.expanded, r.stats.generated)
    assert c.solutions == [s.cost for s in r.solutions]
    assert [i for i, _ in c.passes] == list(range(len(r.iterations)))
    assert [e for _, e in c.passes] == [it.stats.expanded for it in r.iterations]
    assert c.ends == [Status.SOLVED]


# ------------------------------------------------------------------------------------------------ anytime


def test_anytime_improves_until_the_optimum(optimal):
    task, opt = optimal
    for heuristic in ("lmcut", "ff"):
        for lazy in (False, True):
            r = search.anytime_astar(task, heuristic=heuristic, lazy=lazy)
            check_plan(task, r)
            assert isinstance(r, search.AnytimeResult) and r.optimal
            assert r.cost == pytest.approx(opt.cost)
            costs = [s.cost for s in r.solutions]
            assert costs == sorted(costs, reverse=True) and len(set(costs)) == len(costs)
            assert costs[-1] == r.cost
            assert [s.weight for s in r.solutions] == sorted(s.weight for s in r.solutions)[::-1]
            assert all(it.cost_bound >= r.cost for it in r.iterations)
            assert r.iterations[0].cost_bound == math.inf and r.iterations[0].weight == 5
            assert sum(it.stats.expanded for it in r.iterations) == r.stats.expanded
            last = r.iterations[-1]
            assert last.status == Status.EXHAUSTED or (last.weight == 1 and not lazy)


def test_anytime_budgets_and_schedules():
    task = text_task("depot__p02")
    one = search.anytime_astar(task, heuristic="ff", weights=[2])
    assert all(it.weight == 2 for it in one.iterations)
    check_plan(task, one)
    capped = search.anytime_astar(task, heuristic="ff", max_expanded=one.solutions[0].expanded)
    assert capped.status == Status.SOLVED and capped.iterations[-1].status == Status.OUT_OF_STATES
    assert not capped.optimal
    nothing = search.anytime_astar(task, heuristic="ff", max_expanded=1)
    assert nothing.status == Status.OUT_OF_STATES and nothing.solutions == []
    bounded = search.anytime_astar(task, heuristic="lmcut", cost_bound=search.astar(task, heuristic="lmcut").cost)
    assert bounded.status == Status.EXHAUSTED and bounded.solutions == []
    for weights in ([], [1, 2], [0.5], [math.inf], [math.nan]):
        with pytest.raises(ValueError, match="weight"):
            search.anytime_astar(task, weights=weights)
    with pytest.raises(ValueError, match="bucket"):
        search.anytime_astar(task, weights=[1.5, 1], queue="bucket")


# ------------------------------------------------------------------------------------------------ results, threads


def test_results_pickle_and_print():
    task = text_task("depot__p02")
    e = search.ehc(task)
    a = search.anytime_astar(task, heuristic="ff")
    for r in (e, a):
        q = pickle.loads(pickle.dumps(r))
        assert type(q) is type(r) and q.cost == r.cost and q.status == r.status
        assert [str(x) for x in q.plan] == [str(x) for x in r.plan]  # q belongs to a copy of the task
    q = pickle.loads(pickle.dumps(e))
    assert repr(q.ehc) == repr(e.ehc) and q.ehc.lookaheads == e.ehc.lookaheads
    q = pickle.loads(pickle.dumps(a))
    assert [repr(s) for s in q.solutions] == [repr(s) for s in a.solutions]
    assert [repr(i) for i in q.iterations] == [repr(i) for i in a.iterations] and q.optimal == a.optimal
    assert repr(e).startswith("EhcResult(algorithm=ehc, status=solved")
    assert repr(a).startswith("AnytimeResult(algorithm=anytime_astar, status=solved")
    assert repr(search.astar(task)).startswith("BestFirstResult(algorithm=astar_eager")
    assert repr(a.solutions[0]).startswith("AnytimeSolution(cost=")
    assert repr(a.iterations[0]).startswith("AnytimeIteration(weight=5, cost_bound=inf")


def test_many_threads_one_task():
    task = text_task("logistics00__probLOGISTICS-6-1")
    runs = {
        "wastar": lambda: search.astar(task, heuristic="ff", weight=2),
        "ehc": lambda: search.ehc(task),
        "anytime": lambda: search.anytime_astar(task, heuristic="ff"),
    }
    ref = {k: f() for k, f in runs.items()}
    out, errors = [], []

    def work(k):
        try:
            r = runs[k]()
            out.append((k, r.plan, r.stats.expanded))
        except BaseException as e:  # re-raised below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in list(runs) * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(out) == 9
    for k, plan, expanded in out:
        assert plan == ref[k].plan and expanded == ref[k].stats.expanded


def test_heuristic_admissible():
    task = text_task("depot__p02")
    for kind in ("blind", "max", "h2", "lmcut"):
        assert search.Heuristic(task, kind).admissible, kind
    for kind in ("goal_count", "add", "ff", "set_additive", "lmcount"):
        assert not search.Heuristic(task, kind).admissible, kind
    costs = load(COSTS[0])
    assert search.Heuristic(costs, "max").admissible
    assert not search.Heuristic(costs, "max", costs="unit").admissible  # not the task's objective
