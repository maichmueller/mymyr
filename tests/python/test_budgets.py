"""Budgets inside an expansion: max_states bounds the stored states of every search and store (search/control.hpp:
the search stops at the state that fills the store, inside its parent's expansion), and the deadline and the
cancellation stop a search inside an expansion with many successors, at the first check of the expansion after they
become due."""

import threading
import time

import pytest

import mymyr
from mymyr import search

from mymyr.search import Observer, Status  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")

FANOUT = 4096  # successors of the initial state


def fanout_task(objects, extra=""):
    """The initial state has `objects` successors, each a dead end: objects + 1 states (plus `extra` actions)."""
    domain = mymyr.Domain.from_string(
        "(define (domain d) (:requirements :strips :typing) (:types item)"
        " (:predicates (p ?x - item) (free) (done))"
        " (:action put :parameters (?x - item) :precondition (free) :effect (and (p ?x) (not (free))))" + extra + ")"
    )
    names = " ".join(f"o{i}" for i in range(objects))
    problem = f"(define (problem p) (:domain d) (:objects {names} - item) (:init (free)) (:goal (p o{objects - 1})))"
    return mymyr.Task(domain.instantiate_string(problem), atoms="lazy")


@pytest.fixture(scope="module")
def fanout():
    return fanout_task(FANOUT)


BRFS = {
    "flat": dict(store="flat"),
    "chunked": dict(store="chunked"),
    "compact": dict(store="compact"),
    "concurrent": dict(store="concurrent"),
    "concurrent_4": dict(store="concurrent", threads=4),
    "beam_layer_step_4": dict(store="flat", threads=4, beam_width=8, layer_order="in_order"),
}


@pytest.mark.parametrize("name", sorted(BRFS))
@pytest.mark.parametrize("max_states", [1, 2, 3, 100])
def test_brfs_stores_at_most_max_states(fanout, name, max_states):
    r = search.brfs(fanout, max_states=max_states, **BRFS[name])
    assert r.status == Status.OUT_OF_STATES
    assert r.stats.states == max_states


BEST_FIRST = {
    "astar": lambda t, **kw: search.astar(t, heuristic="blind", **kw),
    "astar_lazy": lambda t, **kw: search.astar(t, lazy=True, heuristic="blind", **kw),
    "gbfs": lambda t, **kw: search.gbfs(t, heuristic="blind", **kw),
    "gbfs_lazy": lambda t, **kw: search.gbfs(t, lazy=True, heuristic="blind", **kw),
    "beam": lambda t, **kw: search.beam(t, heuristic="blind", **kw),
}


@pytest.mark.parametrize("name", sorted(BEST_FIRST))
@pytest.mark.parametrize("store", ["flat", "chunked", "compact"])
@pytest.mark.parametrize("max_states", [1, 2, 100])
def test_best_first_stores_at_most_max_states(fanout, name, store, max_states):
    r = BEST_FIRST[name](fanout, store=store, max_states=max_states)
    assert r.status == Status.OUT_OF_STATES
    assert r.stats.states == max_states


IW = {
    "iw": lambda t, **kw: search.iw(t, max_width=1, **kw),
    "iw2": lambda t, **kw: search.iw(t, max_width=2, optimize_iw1=False, **kw),
    "siw": lambda t, **kw: search.siw(t, max_width=1, **kw),
    "astar_iw": lambda t, **kw: search.astar_iw(t, heuristic="blind", **kw),
}


@pytest.mark.parametrize("name", sorted(IW))
@pytest.mark.parametrize("max_states", [1, 2, 100])
def test_iw_family_stops_at_max_states(fanout, name, max_states):
    r = IW[name](fanout, max_states=max_states)
    assert r.status == Status.OUT_OF_STATES


@pytest.mark.parametrize("store", ["flat", "chunked", "compact", "concurrent"])
def test_the_state_that_fills_the_store_stops_the_search(fanout, store):
    # FANOUT + 1 states: a budget of exactly the space stops at its last state, one more exhausts it
    r = search.brfs(fanout, store=store, max_states=FANOUT + 1, stop_at_goal=False)
    assert r.status == Status.OUT_OF_STATES and r.stats.states == FANOUT + 1
    r = search.brfs(fanout, store=store, max_states=FANOUT + 2, stop_at_goal=False)
    assert r.status == Status.SOLVED and r.exhausted and r.stats.states == FANOUT + 1


WIDE_TRANSITIONS = 2**24 + 256  # of the initial state's expansion
CHECK_TRANSITIONS = 1024  # the searches check the time and the cancellation every 1024 transitions of an expansion


# One expansion with WIDE_TRANSITIONS transitions (witness pruning off), almost all to one successor state: without
# checks inside the expansion a stop waits for the whole expansion.
@pytest.fixture(scope="module")
def wide():
    task = fanout_task(
        256,
        " (:action touch :parameters (?x ?y ?z - item) :precondition (free) :effect (and (done) (not (free))))",
    )
    r = search.brfs(task, witness_pruning=False, stop_at_goal=False)
    assert r.exhausted and r.stats.generated == WIDE_TRANSITIONS
    return task


WIDE = {
    "brfs": lambda t, **kw: search.brfs(t, witness_pruning=False, **kw),
    "brfs_compact": lambda t, **kw: search.brfs(t, store="compact", witness_pruning=False, **kw),
    "brfs_concurrent": lambda t, **kw: search.brfs(t, store="concurrent", threads=2, witness_pruning=False, **kw),
    "astar": lambda t, **kw: search.astar(t, heuristic="blind", **kw),
    "gbfs": lambda t, **kw: search.gbfs(t, heuristic="blind", **kw),
    "iw": lambda t, **kw: search.iw(t, max_width=1, **kw),
}


class StopOnFirstExpansion(Observer):
    """Calls `stop` when the first expansion begins: after the checks before it, so that only the checks inside the
    expansion can see the stop. The worker observers of a parallel search do the same."""

    def __init__(self, stop):
        self.stop = stop
        self.expansions = 0
        self.workers = []

    def on_expand(self, id, state):
        self.expansions += 1
        if self.expansions == 1:
            self.stop()

    def make_worker(self, k):
        worker = StopOnFirstExpansion(self.stop)
        self.workers.append(worker)
        return worker


def assert_stopped_inside_the_first_expansion(r, observer):
    assert sum(o.expansions for o in [observer, *observer.workers]) == r.stats.expanded == 1
    # at the first check after the stop became due, long before the expansion's end
    assert r.stats.generated <= CHECK_TRANSITIONS < WIDE_TRANSITIONS


@pytest.mark.parametrize("name", sorted(WIDE))
def test_the_deadline_stops_inside_an_expansion(wide, name):
    # sleeping inside on_expand passes the deadline in real time, however fast or busy the machine
    deadline = 0.5
    observer = StopOnFirstExpansion(lambda: time.sleep(deadline + 0.01))
    start = time.perf_counter()
    r = WIDE[name](wide, max_seconds=deadline, observer=observer)
    assert r.status == Status.OUT_OF_TIME
    assert_stopped_inside_the_first_expansion(r, observer)
    assert time.perf_counter() - start < 30


@pytest.mark.parametrize("name", sorted(WIDE))
def test_a_cancellation_stops_inside_an_expansion(wide, name):
    token = search.CancelToken()

    def request_from_another_thread():
        requester = threading.Thread(target=token.request)
        requester.start()
        requester.join()

    observer = StopOnFirstExpansion(request_from_another_thread)
    r = WIDE[name](wide, cancel=token, observer=observer)
    assert r.status == Status.CANCELLED
    assert_stopped_inside_the_first_expansion(r, observer)
