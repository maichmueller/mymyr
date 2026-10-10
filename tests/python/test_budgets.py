"""Budgets inside an expansion: max_states bounds the stored states of every search and store (search/control.hpp:
the search stops at the state that fills the store, inside its parent's expansion), and the deadline and the
cancellation stop a search inside an expansion with many successors."""

import threading
import time

import pytest

import mymyr
from mymyr import search

from mymyr.search import Status  # noqa: E402

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
    assert r.states == max_states


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
    "iw": lambda t, **kw: search.iw(t, max_arity=1, **kw),
    "iw2": lambda t, **kw: search.iw(t, max_arity=2, optimize_iw1=False, **kw),
    "siw": lambda t, **kw: search.siw(t, max_arity=1, **kw),
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
    assert r.status == Status.OUT_OF_STATES and r.states == FANOUT + 1
    r = search.brfs(fanout, store=store, max_states=FANOUT + 2, stop_at_goal=False)
    assert r.status == Status.SOLVED and r.exhausted and r.states == FANOUT + 1


# One expansion with 2^24 + 256 transitions (witness pruning off), almost all to one successor state: without checks
# inside the expansion a stop waits for the whole expansion.
@pytest.fixture(scope="module")
def wide():
    return fanout_task(
        256,
        " (:action touch :parameters (?x ?y ?z - item) :precondition (free) :effect (and (done) (not (free))))",
    )


@pytest.fixture(scope="module")
def expansion_seconds(wide):
    """The time of the whole search: one expansion of the initial state, then dead ends."""
    t0 = time.perf_counter()
    r = search.brfs(wide, witness_pruning=False, stop_at_goal=False)
    assert r.exhausted and r.generated == 2**24 + 256
    return time.perf_counter() - t0


WIDE = {
    "brfs": lambda t, **kw: search.brfs(t, witness_pruning=False, **kw),
    "brfs_compact": lambda t, **kw: search.brfs(t, store="compact", witness_pruning=False, **kw),
    "brfs_concurrent": lambda t, **kw: search.brfs(t, store="concurrent", threads=2, witness_pruning=False, **kw),
    "astar": lambda t, **kw: search.astar(t, heuristic="blind", **kw),
    "gbfs": lambda t, **kw: search.gbfs(t, heuristic="blind", **kw),
    "iw": lambda t, **kw: search.iw(t, max_arity=1, **kw),
}


@pytest.mark.parametrize("name", sorted(WIDE))
def test_the_deadline_stops_inside_an_expansion(wide, expansion_seconds, name):
    t0 = time.perf_counter()
    r = WIDE[name](wide, max_seconds=0.02)
    elapsed = time.perf_counter() - t0
    assert r.status == Status.OUT_OF_TIME
    assert elapsed < expansion_seconds / 3, (elapsed, expansion_seconds)


@pytest.mark.parametrize("name", sorted(WIDE))
def test_a_cancellation_stops_inside_an_expansion(wide, expansion_seconds, name):
    token = search.CancelToken()
    timer = threading.Timer(0.02, token.request)
    timer.start()
    t0 = time.perf_counter()
    try:
        r = WIDE[name](wide, cancel=token)
    finally:
        timer.cancel()
    elapsed = time.perf_counter() - t0
    assert r.status == Status.CANCELLED
    assert elapsed < expansion_seconds / 3, (elapsed, expansion_seconds)
