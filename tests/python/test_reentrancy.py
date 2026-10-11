"""Re-entrancy: code that runs inside a search (an observer, a goal test, a heuristic written in Python) may enumerate
the task's actions, evaluate states, call a shared Heuristic and start other searches on the same task and thread
without changing the running search's result."""

import threading

import pytest

import mymyr
from mymyr import search
from mymyr.search import Observer, Status

from conftest import text_task

# (finish) needs (p a) and (p b): a corrupted enumeration returns plans that skip one of the puts.
DOMAIN = """(define (domain d) (:requirements :strips :typing)
(:types item) (:constants a b - item) (:predicates (p ?x - item) (done))
(:action put :parameters (?x - item) :precondition (and) :effect (p ?x))
(:action finish :parameters () :precondition (and (p a) (p b)) :effect (done)))"""
PROBLEM = "(define (problem p) (:domain d) (:init) (:goal (done)))"


@pytest.fixture(scope="module")
def two_puts(tmp_path_factory):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    d = tmp_path_factory.mktemp("two_puts")
    (d / "domain.pddl").write_text(DOMAIN)
    (d / "problem.pddl").write_text(PROBLEM)
    return mymyr.Task.from_pddl(d / "domain.pddl", d / "problem.pddl")


@pytest.fixture(scope="module", params=["two_puts", "philosophers__p03-phil4", "depot__p02"])
def task(request):
    if request.param == "two_puts":
        return request.getfixturevalue("two_puts")
    return text_task(request.param)  # philosophers: axioms and a derived goal


def meddle(task, state):
    """What a callback may do with the task of the running search: enumerate, test, apply, evaluate."""
    actions = state.applicable_actions()
    for a in actions[:4]:
        assert task.is_applicable(state, a)
        assert a.is_applicable(state)
        task.apply(state, a)
    assert len(task.successors(state)) == len(actions)
    assert state.any_applicable() == bool(actions)
    assert len(list(task.iter_applicable_actions(state))) == len(actions)
    state.derived_atoms()
    task.is_goal(state)


def nested_searches(task):
    budget = {"max_states": 200, "max_expanded": 100}
    for r in (search.brfs(task, max_states=200), search.iw(task, max_arity=1, **budget), search.astar(task, **budget)):
        if r.solved:
            assert replays(task, r.plan)


def replays(task, plan):
    s = task.initial_state
    for a in plan:
        if not task.is_applicable(s, a):
            return False
        s = task.apply(s, a)
    return task.is_goal(s)


class Meddler(Observer):
    """Meddles on the first `events` events; starts nested searches from the first `nested` ones."""

    def __init__(self, task, nested=2, workers=False, events=400):
        self.task = task
        self.nested = nested
        self.events = events
        self.workers = workers
        self.calls = 0
        self.lock = threading.Lock()

    def _event(self, state):
        with self.lock:
            self.calls += 1
            calls = self.calls
        if calls <= self.events:
            meddle(self.task, state)
        if calls <= self.nested:
            nested_searches(self.task)

    def on_expand(self, id, state):
        self._event(state)

    def on_generate(self, parent, action, child, state, is_new):
        self._event(state)

    def make_worker(self, k):
        return Meddler(self.task, nested=1) if self.workers else None


RUNS = {
    "iw": lambda t, o: search.iw(t, max_arity=2, observer=o),
    "siw": lambda t, o: search.siw(t, max_arity=2, observer=o),
    "brfs": lambda t, o: search.brfs(t, stop_at_goal=True, observer=o),
    "brfs_chunked": lambda t, o: search.brfs(t, store="chunked", stop_at_goal=True, observer=o),
    "brfs_compact": lambda t, o: search.brfs(t, store="compact", stop_at_goal=True, observer=o),
    "brfs_concurrent": lambda t, o: search.brfs(t, store="concurrent", stop_at_goal=True, observer=o),
    "astar": lambda t, o: search.astar(t, heuristic="max", observer=o),
    "astar_lazy": lambda t, o: search.astar(t, heuristic="max", lazy=True, observer=o),
    "gbfs": lambda t, o: search.gbfs(t, heuristic="ff", observer=o),
    "gbfs_lazy": lambda t, o: search.gbfs(t, heuristic="ff", lazy=True, observer=o),
    "beam": lambda t, o: search.beam(t, width=4, observer=o),
    "astar_iw": lambda t, o: search.astar_iw(t, observer=o),
    "rollout_iw": lambda t, o: search.rollout_iw(t, max_rollouts=200, observer=o),
}


def plan_of(r):
    return [str(a) for a in r.plan]


@pytest.mark.parametrize("name", sorted(RUNS))
def test_meddling_observers_do_not_change_the_search(task, name):
    run = RUNS[name]
    ref = run(task, None)
    obs = Meddler(task)
    got = run(task, obs)
    assert obs.calls > 0
    assert got.status == ref.status
    assert plan_of(got) == plan_of(ref)
    if got.solved:
        assert replays(task, got.plan)


@pytest.mark.parametrize("threads", [2, 4])
def test_meddling_worker_observers_do_not_change_brfs(task, threads):
    ref = search.brfs(task, stop_at_goal=True)
    obs = Meddler(task, workers=True)
    got = search.brfs(task, stop_at_goal=True, threads=threads, observer=obs)
    assert got.status == ref.status and len(got.plan) == len(ref.plan)
    if got.solved:
        assert replays(task, got.plan)


def test_nested_searches_two_levels_deep(two_puts):
    class Outer(Observer):
        def on_generate(self, parent, action, child, state, is_new):
            inner = search.iw(two_puts, max_arity=2, observer=Meddler(two_puts, nested=1))
            assert inner.solved and replays(two_puts, inner.plan)

    r = search.brfs(two_puts, stop_at_goal=True, observer=Outer())
    assert r.solved and replays(two_puts, r.plan) and len(r.plan) == 3


def test_goal_tests_and_python_heuristics_may_meddle(task):
    def goal(s):
        meddle(task, s)
        return task.is_goal(s)

    def h(s):
        meddle(task, s)
        return 0.0

    for ref, got in (
        (search.iw(task, max_arity=2), search.iw(task, max_arity=2, goal=goal)),
        (search.astar(task, heuristic="blind"), search.astar(task, heuristic=h)),
    ):
        assert got.status == ref.status and plan_of(got) == plan_of(ref)


def finishes(fn, seconds=120):
    """fn() on a thread, joined with a timeout: (finished, result or exception)."""
    out = []
    t = threading.Thread(target=lambda: out.append(_call(fn)), daemon=True)
    t.start()
    t.join(seconds)
    return (not t.is_alive()), (out[0] if out else None)


def _call(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return e


@pytest.mark.parametrize("kind", ["max", "ff", "goal_count", "perfect"])
def test_a_searchs_heuristic_object_is_callable_from_its_observer(two_puts, kind):
    if kind == "perfect":
        h = search.Heuristic.perfect(mymyr.datasets.state_space(two_puts, remove_if_unsolvable=False))
    else:
        h = search.Heuristic(two_puts, kind)
    values = []

    class Probe(Observer):
        def on_expand(self, id, state):
            values.append(h(state))
            h.preferred_actions(state)

    ref = search.astar(two_puts, heuristic=h)
    done, got = finishes(lambda: search.astar(two_puts, heuristic=h, observer=Probe(), max_seconds=60))
    assert done, "the search deadlocked on its heuristic object"
    assert not isinstance(got, Exception), got
    assert got.status == Status.SOLVED and plan_of(got) == plan_of(ref)
    assert values and h.stats.evaluations >= len(values)


def test_a_python_heuristic_may_call_the_heuristic_object_of_the_search(task):
    h = search.Heuristic(task, "max")
    ref = search.astar(task, heuristic=h)
    done, got = finishes(lambda: search.astar(task, heuristic=lambda s: h(s)))
    assert done and not isinstance(got, Exception), got
    assert got.status == ref.status and plan_of(got) == plan_of(ref)


def test_searches_on_other_threads_share_a_heuristic_object(task):
    h = search.Heuristic(task, "ff")
    ref = search.gbfs(task, heuristic=h)
    results, errors = [], []

    def work():
        try:
            for _ in range(3):
                results.append(plan_of(search.gbfs(task, heuristic=h)))
                h(task.initial_state)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [plan_of(ref)] * 12
