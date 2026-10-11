"""Heuristics written in Python for A*, GBFS and beam search: callables, objects with evaluate / evaluate_batch /
preferred_actions, dead ends, and errors raised by them."""

import math
import threading

import pytest

from mymyr import search
from mymyr.search import Status
from conftest import text_task


@pytest.fixture(scope="module")
def task():
    return text_task("logistics00__probLOGISTICS-6-1")


def same_search(a, b):
    assert a.status == b.status == Status.SOLVED
    assert a.cost == b.cost
    assert a.stats.expanded == b.stats.expanded
    assert a.stats.generated == b.stats.generated
    assert [str(x) for x in a.plan] == [str(x) for x in b.plan]


@pytest.mark.parametrize("lazy", [False, True])
def test_callable_searches_like_the_native_heuristic(task, lazy):
    h = search.Heuristic(task, "max")
    native = search.astar(task, heuristic="max", queue="heap", lazy=lazy)
    py = search.astar(task, heuristic=lambda s: h(s), lazy=lazy)
    same_search(native, py)
    assert py.queue == "heap"
    assert py.evaluations == native.evaluations
    assert py.heuristic.evaluations == py.evaluations


def test_object_with_evaluate(task):
    class GoalCount:
        def __init__(self, task):
            self.h = search.Heuristic(task, "goal_count")
            self.calls = 0

        def evaluate(self, state):
            self.calls += 1
            return self.h(state)

    obj = GoalCount(task)
    py = search.gbfs(task, heuristic=obj)
    native = search.gbfs(task, heuristic="goal_count", queue="heap")
    same_search(native, py)
    assert obj.calls == py.evaluations > 0
    for lazy in (False, True):
        r = search.astar(task, heuristic=obj, lazy=lazy, max_expanded=200_000)
        assert r.status == Status.SOLVED
    assert search.beam(task, heuristic=obj, width=100).status == Status.SOLVED


class Batched:
    """evaluate_batch over the native h_max: records the batch sizes."""

    def __init__(self, task, single=True):
        self.h = search.Heuristic(task, "max")
        self.sizes = []
        if single:
            self.evaluate = lambda s: self.h(s)

    def evaluate_batch(self, states):
        self.sizes.append(len(states))
        return [self.h(s) for s in states]


@pytest.mark.parametrize("single", [True, False])
def test_evaluate_batch(task, single):
    native = search.astar(task, heuristic="max", queue="heap")
    obj = Batched(task, single)
    py = search.astar(task, heuristic=obj)
    same_search(native, py)
    assert obj.sizes[0] == 1 or single  # the start state is evaluated alone
    assert sum(obj.sizes) == py.evaluations - single
    assert max(obj.sizes) > 1 and len(obj.sizes) < py.evaluations
    obj = Batched(task, single)
    g = search.gbfs(task, heuristic=obj)
    assert g.status == Status.SOLVED and obj.sizes
    obj = Batched(task, single)
    b = search.beam(task, heuristic=obj, width=50)
    assert b.status == Status.SOLVED and len(obj.sizes) <= b.layers
    # lazy searches evaluate one state at a time (through evaluate_batch when there is no evaluate)
    obj = Batched(task, single)
    r = search.astar(task, heuristic=obj, lazy=True)
    assert r.status == Status.SOLVED and r.cost == native.cost
    assert all(n == 1 for n in obj.sizes)


def test_batch_results_may_be_numpy_arrays(task):
    np = pytest.importorskip("numpy")
    h = search.Heuristic(task, "max")

    class Arr:
        def evaluate_batch(self, states):
            return np.array([h(s) for s in states], dtype=np.float32)

    assert search.astar(task, heuristic=Arr()).cost == search.astar(task, heuristic="max").cost


def test_preferred_actions(task):
    """A Python wrapper of h_FF with its preferred actions searches like the Heuristic object itself."""

    class Ff:
        def __init__(self):
            self.h = search.Heuristic(task, "ff")
            self.state = None

        def evaluate(self, state):
            self.state = state
            return self.h(state)

        def preferred_actions(self):
            return self.h.preferred_actions(self.state)

    for algo in (search.gbfs, search.astar):
        native = algo(task, heuristic=search.Heuristic(task, "ff"), lazy=True)
        py = algo(task, heuristic=Ff(), lazy=True)
        same_search(native, py)


def test_dead_ends(task):
    s0 = task.initial_state
    r = search.astar(task, heuristic=lambda s: 0.0 if s == s0 else math.inf)
    assert r.status in (Status.EXHAUSTED, Status.UNSOLVABLE) and r.dead_ends > 0
    r = search.gbfs(task, heuristic=lambda s: math.inf)
    assert r.status == Status.UNSOLVABLE  # a dead-end start state


def test_errors_propagate(task):
    with pytest.raises(ZeroDivisionError):
        search.astar(task, heuristic=lambda s: 1 / 0)
    calls = []

    def late(s):
        calls.append(1)
        if len(calls) == 50:
            raise KeyError("late")
        return 0

    token = search.CancelToken()
    with pytest.raises(KeyError, match="late"):
        search.gbfs(task, heuristic=late, cancel=token)
    assert token.requested and len(calls) == 50  # no Python call after the error

    class Short:
        def evaluate_batch(self, states):
            return [0.0]

    with pytest.raises(ValueError, match="evaluate_batch returned"):
        search.astar(task, heuristic=Short())
    with pytest.raises(ValueError, match="a Python heuristic returned nan"):
        search.astar(task, heuristic=lambda s: math.nan)
    with pytest.raises(ValueError, match=r"a Python heuristic returned -1.0; heuristic values are >= 0"):
        search.astar(task, heuristic=lambda s: -1.0)
    with pytest.raises(TypeError, match="a Python heuristic returned a str"):
        search.astar(task, heuristic=lambda s: "far")
    with pytest.raises(TypeError, match="a Python heuristic returned a NoneType"):
        search.astar(task, heuristic=lambda s: None)
    with pytest.raises(TypeError):
        search.astar(task, heuristic=object())

    class WrongTask:
        other = text_task("depot__p02")

        def evaluate(self, s):
            return 1

        def preferred_actions(self):
            return self.other.applicable_actions(self.other.initial_state)

    with pytest.raises(ValueError, match="another task"):
        search.gbfs(task, heuristic=WrongTask(), lazy=True)


def test_threads_share_one_callable(task):
    """Free-threaded: several searches call one Python heuristic at once."""
    h = search.Heuristic(task, "max")
    lock = threading.Lock()

    def hmax(s):
        with lock:  # a Heuristic object serializes its evaluations anyway
            return h(s)

    ref = search.astar(task, heuristic=hmax)
    out, errors = [], []

    def work():
        try:
            out.append(search.astar(task, heuristic=hmax).stats.expanded)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and out == [ref.stats.expanded] * 8
