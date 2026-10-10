"""LM-cut and the landmark count through mymyr.search: kind names, admissibility of LM-cut, the path-dependent status
of the landmark count (initial_status, progress, merge, h(state, status), preferred actions), every best-first search
with both, refusals, and one heuristic per thread over a shared task."""

import math
import threading

import pytest

import mymyr
from mymyr import datasets, search
from mymyr.search import Status
from conftest import text_task

CHAIN_DOMAIN = """(define (domain chain) (:requirements :strips)
 (:predicates (y) (x) (g) (h))
 (:action get-y :parameters () :precondition (and) :effect (y))
 (:action lose-y :parameters () :precondition (y) :effect (not (y)))
 (:action get-x :parameters () :precondition (y) :effect (x))
 (:action lose-x :parameters () :precondition (x) :effect (not (x)))
 (:action make-g :parameters () :precondition (x) :effect (g))
 (:action lose-g :parameters () :precondition (g) :effect (not (g)))
 (:action get-h :parameters () :precondition (and) :effect (h)))"""
CHAIN_PROBLEM = "(define (problem p) (:domain chain) (:init) (:goal (and (g) (h))))"


@pytest.fixture(scope="module")
def depot():
    return text_task("depot__p02")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    d = tmp_path_factory.mktemp("chain")
    (d / "domain.pddl").write_text(CHAIN_DOMAIN)
    (d / "problem.pddl").write_text(CHAIN_PROBLEM)
    return mymyr.Task.from_pddl(d / "domain.pddl", d / "problem.pddl")


def action(task, state, name):
    (a,) = [a for a in task.applicable_actions(state) if str(a) == name]
    return a


def test_kind_names(depot):
    s = depot.initial_state
    for kind, aliases in (("lmcut", ("lm_cut",)), ("lmcount", ("lm_count", "landmark_count"))):
        h = search.Heuristic(depot, kind)
        assert h.kind == kind and h.grounded
        assert h.path_dependent == (kind == "lmcount")
        for a in aliases:
            assert search.Heuristic(depot, a)(s) == h(s)
        with pytest.raises(ValueError, match="lifted"):
            search.Heuristic(depot, kind, evaluation="lifted")
    assert search.Heuristic(depot, "lmcut").initial_status(s) == b""
    hmax = search.Heuristic(depot, "max")
    assert search.Heuristic(depot, "lmcut", share=hmax)(s) == search.Heuristic(depot, "lmcut")(s)


def test_lmcut_lies_between_hmax_and_hstar(depot):
    space = datasets.state_space(depot, remove_if_unsolvable=False)
    lmcut, hmax = search.Heuristic(depot, "lmcut"), search.Heuristic(depot, "max")
    for i in range(0, space.num_states, max(1, space.num_states // 200)):
        s = space.state(i)
        d = space.unit_goal_distance(i)
        assert hmax(s) <= lmcut(s) <= (math.inf if d < 0 else d)


def test_landmark_count_follows_the_path(chain):
    h = search.Heuristic(chain, "lmcount")
    assert h.path_dependent
    s = chain.initial_state
    status = h.initial_status(s)
    assert isinstance(status, bytes) and len(status) % 8 == 0 and len(status) > 0
    assert h(s, status) == h(s) == 4
    start = status
    for name in ("(get-y)", "(get-x)", "(lose-y)"):
        a = action(chain, s, name)
        t = chain.apply(s, a)
        status = h.progress(status, s, a, t)
        s = t
    # y was reached and is not needed again; the state alone does not know it
    assert h(s, status) == 2 and h(s) == 3
    assert h.evaluate(s, status=status) == 2
    # reached on another path that reached nothing: the landmarks both reached
    assert h.merge(status, start) == start
    # preferred: the applicable actions that add a needed landmark (g, h; not y)
    assert {str(a) for a in h.preferred_actions(s, status)} == {"(make-g)", "(get-h)"}
    with pytest.raises(ValueError, match="bytes"):
        h(s, b"\0")
    with pytest.raises(TypeError):
        h(s, "status")


@pytest.mark.parametrize("kind", ["lmcut", "lmcount"])
def test_every_best_first_search(depot, kind):
    optimal = search.astar(depot, heuristic="max").cost
    runs = [
        search.astar(depot, heuristic=kind),
        search.astar(depot, heuristic=kind, lazy=True),
        search.gbfs(depot, heuristic=kind),
        search.gbfs(depot, heuristic=kind, lazy=True),
        search.beam(depot, heuristic=kind),
    ]
    iw = search.astar_iw(depot, heuristic=kind)  # novelty pruning: incomplete
    assert iw.status in (Status.SOLVED, Status.EXHAUSTED), iw.message
    for r in runs + [iw] * (iw.status == Status.SOLVED):
        assert r.status == Status.SOLVED, r.message
        s = depot.initial_state
        for a in r.plan:
            s = depot.apply(s, a)
        assert depot.is_goal(s)
    if kind == "lmcut":
        assert runs[0].cost == runs[1].cost == optimal
    h = search.Heuristic(depot, kind)
    r = search.gbfs(depot, heuristic=h, lazy=True)
    assert r.status == Status.SOLVED


def test_refusals(chain):
    r = search.gbfs(chain, heuristic="lmcount", goal=[["(h)"]])
    assert r.status == Status.FAILED and "AnyOf" in r.message
    r = search.astar(chain, heuristic="lmcut", goal=[["(h)"]])
    assert r.status == Status.SOLVED and r.cost == 1


def test_one_heuristic_per_thread_over_a_shared_task(depot):
    states = [depot.initial_state]
    for a in search.astar(depot, heuristic="max").plan:
        states.append(depot.apply(states[-1], a))
    expected = {k: [search.Heuristic(depot, k)(s) for s in states] for k in ("lmcut", "lmcount")}
    errors = []

    def run(kind):
        try:
            h = search.Heuristic(depot, kind)
            for _ in range(3):
                assert [h(s) for s in states] == expected[kind]
        except Exception as e:  # noqa: BLE001
            errors.append((kind, e))

    threads = [threading.Thread(target=run, args=(k,)) for k in ("lmcut", "lmcount", "lmcut", "lmcount")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
