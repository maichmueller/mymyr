"""mymyr.search.Observer: event counts, errors, threads, and the control of brfs (max_seconds, cancel)."""

import threading

import pytest

import mymyr
from mymyr import search
from mymyr.search import Observer, Status

from conftest import text_task


@pytest.fixture(scope="module")
def task():
    return text_task("blocks__probBLOCKS-8-0")


@pytest.fixture(scope="module")
def small():
    return text_task("depot__p02")


class Counter(Observer):
    def __init__(self):
        self.started = self.expanded = self.generated = self.new = 0
        self.passes = []
        self.solutions = []
        self.ended = None
        self.threads = set()

    def on_start(self, state):
        self.started += 1

    def on_expand(self, id, state):
        self.expanded += 1
        self.threads.add(threading.get_ident())

    def on_generate(self, parent, action, child, state, is_new):
        self.generated += 1
        self.new += bool(is_new)

    def on_pass(self, arity, stats):
        self.passes.append((arity, stats.expanded, stats.generated, stats.states))

    def on_solution(self, plan, cost):
        self.solutions.append((plan, cost))

    def on_end(self, status, stats):
        self.ended = (status, stats.expanded, stats.generated)


def totals(r):
    if isinstance(r, search.BrfsResult):
        return r.expanded, r.generated
    s = r.stats if isinstance(r, search.BestFirstResult) else r.total
    return s.expanded, s.generated


RUNS = {
    "iw": lambda t, o: search.iw(t, max_arity=2, observer=o),
    "siw": lambda t, o: search.siw(t, max_arity=2, observer=o),
    "brfs": lambda t, o: search.brfs(t, stop_at_goal=True, observer=o),
    "brfs_exhaustive": lambda t, o: search.brfs(t, observer=o),
    "brfs_chunked": lambda t, o: search.brfs(t, store="chunked", stop_at_goal=True, observer=o),
    "brfs_compact": lambda t, o: search.brfs(t, store="compact", observer=o),
    "brfs_concurrent": lambda t, o: search.brfs(t, store="concurrent", stop_at_goal=True, observer=o),
    "astar": lambda t, o: search.astar(t, heuristic="max", observer=o),
    "astar_lazy": lambda t, o: search.astar(t, heuristic="max", lazy=True, observer=o),
    "gbfs": lambda t, o: search.gbfs(t, heuristic="ff", observer=o),
    "gbfs_lazy": lambda t, o: search.gbfs(t, heuristic="ff", lazy=True, observer=o),
}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_event_counts_equal_the_statistics(small, name):
    task = small
    o = Counter()
    r = RUNS[name](task, o)
    expanded, generated = totals(r)
    assert (o.expanded, o.generated) == (expanded, generated)
    assert o.started == 1 and o.ended == (r.status, expanded, generated)
    assert o.expanded > 0
    if r.status == Status.SOLVED:
        assert len(o.solutions) == 1 and o.solutions[0][0] == r.plan
    else:
        assert o.solutions == []
    plain = RUNS[name](task, None)
    assert (plain.status, totals(plain), plain.plan) == (r.status, (expanded, generated), r.plan)


@pytest.mark.parametrize("store", ["flat", "chunked", "compact", "concurrent"])
def test_brfs_layers_and_new_states(small, store):
    o = Counter()
    r = search.brfs(small, store=store, observer=o)
    assert r.status == Status.EXHAUSTED
    assert len(o.passes) == r.layers and [p[0] for p in o.passes] == list(range(r.layers))
    assert sum(p[1] for p in o.passes) == r.expanded and sum(p[2] for p in o.passes) == r.generated
    assert sum(p[3] for p in o.passes) == r.states - 1 == o.new


class Worker(Observer):
    def __init__(self):
        self.expanded = self.generated = self.new = 0
        self.threads = set()
        self.children = set()

    def on_expand(self, id, state):
        self.expanded += 1
        self.threads.add(threading.get_ident())

    def on_generate(self, parent, action, child, state, is_new):
        self.generated += 1
        self.new += bool(is_new)
        self.children.add(child)


class Root(Counter):
    def __init__(self):
        super().__init__()
        self.workers = []

    def make_worker(self, k):
        w = Worker()
        self.workers.append(w)
        return w


@pytest.mark.parametrize("threads", [2, 4])
def test_brfs_threads_send_hot_events_to_worker_observers(task, threads):
    o = Root()
    r = search.brfs(task, threads=threads, observer=o)
    assert r.threads == threads and len(o.workers) == threads
    assert sum(w.expanded for w in o.workers) == r.expanded
    assert sum(w.generated for w in o.workers) == r.generated
    assert sum(w.new for w in o.workers) == r.states - 1
    assert all(w.children == {None} for w in o.workers if w.generated)  # ids are assigned when a layer ends
    assert len(set().union(*(w.threads for w in o.workers))) > 1  # called from the worker threads
    assert o.expanded == 0 and o.started == 1 and o.ended[0] == Status.EXHAUSTED  # lifecycle on the root
    assert len(o.passes) == r.layers and sum(p[1] for p in o.passes) == r.expanded


def test_brfs_threads_without_make_worker_run_on_one_thread(task):
    o = Counter()
    r = search.brfs(task, threads=4, stop_at_goal=True, observer=o)
    assert r.threads == 1 and o.threads == {threading.get_ident()}
    assert (o.expanded, o.generated) == (r.expanded, r.generated)
    assert r.status == Status.SOLVED and o.solutions[0][0] == r.plan


def test_brfs_threads_return_a_shortest_plan(task):
    one = search.brfs(task, stop_at_goal=True)
    for threads in (2, 4):
        r = search.brfs(task, threads=threads, stop_at_goal=True)
        assert r.status == Status.SOLVED and len(r.plan) == len(one.plan)
        s = task.initial_state
        for a in r.plan:
            s = task.apply(s, a)
        assert task.is_goal(s)


class Boom(Observer):
    """Raises at the at-th expansion it sees."""

    def __init__(self, at=100):
        self.at = at
        self.seen = 0

    def on_expand(self, id, state):
        self.seen += 1
        if self.seen == self.at:
            raise KeyError("boom")


@pytest.mark.parametrize("name", ["iw", "brfs", "brfs_compact", "brfs_concurrent", "astar", "gbfs"])
def test_an_exception_in_an_event_is_reraised(task, name):
    with pytest.raises(KeyError, match="boom"):
        RUNS[name](task, Boom(at=5))


def test_an_exception_in_a_worker_observer_is_reraised(task):
    class Root(Observer):
        def make_worker(self, k):
            return Boom(at=1000)

    with pytest.raises(KeyError, match="boom"):
        search.brfs(task, threads=3, observer=Root())


def test_unoverridden_events_are_not_called(task):
    class OnlyEnd(Observer):
        def __init__(self):
            self.ends = 0

        def on_end(self, status, stats):
            self.ends += 1

    o = OnlyEnd()
    r = search.astar(task, heuristic="max", observer=o)
    assert o.ends == 1 and r.status == Status.SOLVED
    # the defaults are marked so that the searches skip them; a subclass's override is not
    assert getattr(Observer.on_expand, "_mymyr_default_event", False)
    assert not hasattr(OnlyEnd.on_end, "_mymyr_default_event")
    # an observer that overrides nothing is not installed: the same search
    a, b = search.astar(task, heuristic="max", observer=Observer()), search.astar(task, heuristic="max")
    assert (a.status, a.plan, a.stats.expanded, a.stats.generated) == (b.status, b.plan, b.stats.expanded,
                                                                      b.stats.generated)


def test_duck_typed_observers_still_work(small):
    class Plain:
        def __init__(self):
            self.n = 0

        def on_expand(self, id, state):
            self.n += 1

    o = Plain()
    r = search.brfs(small, observer=o)
    assert o.n == r.expanded


def test_on_progress_stops_the_search(task):
    class Stop(Observer):
        def on_progress(self, stats):
            return stats.expanded < 500

    for run in (lambda o: search.brfs(task, observer=o, progress_interval=100),
                lambda o: search.astar(task, heuristic="max", observer=o, progress_interval=100),
                lambda o: search.iw(task, max_arity=2, observer=o, progress_interval=100)):
        r = run(Stop())
        assert r.status == Status.CANCELLED
        assert totals(r)[0] <= 600


def test_brfs_budgets_and_cancel(task):
    assert search.brfs(task, max_seconds=0).status == Status.OUT_OF_TIME
    assert search.brfs(task, threads=3, max_seconds=0).status == Status.OUT_OF_TIME
    r = search.brfs(task, max_seconds=0.02)
    assert r.status == Status.OUT_OF_TIME and not r.exhausted and not r.solved
    token = search.CancelToken()
    token.request()
    for kw in ({}, {"store": "compact"}, {"threads": 2}):
        r = search.brfs(task, cancel=token, **kw)
        assert r.status == Status.CANCELLED and not r.exhausted
    assert search.brfs(task, max_states=1000).status == Status.OUT_OF_STATES
    assert search.brfs(task, stop_at_goal=True).status == Status.SOLVED
    assert search.brfs(text_task("depot__p02")).status == Status.EXHAUSTED


def test_cancel_from_another_thread(task):
    token = search.CancelToken()

    class Cancel(Observer):
        def on_expand(self, id, state):
            if id == 1000:
                threading.Thread(target=token.request).start()

    r = search.brfs(task, cancel=token, observer=Cancel())
    assert r.status == Status.CANCELLED
    assert isinstance(mymyr.search.Observer(), Observer)
