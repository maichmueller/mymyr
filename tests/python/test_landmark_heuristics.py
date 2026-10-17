"""LM-cut and the landmark count through mymyr.search: kind names, admissibility of LM-cut, the path-dependent status
of the landmark count (initial_status, progress, merge, h(state, status), preferred actions), every best-first search
with both, refusals, one heuristic per thread over a shared task, searches on several threads sharing one
landmark count object, and independence of the atom slots."""

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
    for kind, others in (("lmcut", ("lm_cut",)), ("lmcount", ("lm_count", "landmark_count"))):
        h = search.Heuristic(depot, kind)
        assert h.kind == kind and h.grounded
        assert h.path_dependent == (kind == "lmcount")
        for other in others:
            with pytest.raises(ValueError, match="unknown heuristic"):
                search.Heuristic(depot, other)
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
    # y was reached and is not needed again; alone, the state shows x, so every path to it reached y
    assert h(s, status) == 2 and h(s) == 2
    assert h.evaluate(s, status=status) == 2
    # reached on another path that reached nothing: the landmarks both reached
    assert h.merge(status, start) == start
    # preferred: the applicable actions that add a needed landmark (g, h; not y)
    assert {str(a) for a in h.preferred_actions(s, status)} == {"(make-g)", "(get-h)"}
    with pytest.raises(ValueError, match="bytes"):
        h(s, b"\0")
    with pytest.raises(TypeError):
        h(s, "status")
    # back in the initial state after reaching g: g and h are needed; alone, it starts a path
    for name in ("(make-g)", "(lose-x)", "(lose-g)"):
        a = action(chain, s, name)
        t = chain.apply(s, a)
        status = h.progress(status, s, a, t)
        s = t
    assert s == chain.initial_state
    assert h(s, status) == 2 and h(s) == 4


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



def test_searches_on_other_threads_share_a_landmark_count_object():
    task = text_task("philosophers__p03-phil4")  # derived landmark atoms: evaluated by the axioms
    h = search.Heuristic(task, "lmcount")
    status = h.initial_status(task.initial_state)
    ref = [str(a) for a in search.gbfs(task, heuristic=h, lazy=True).plan]
    results, errors = [], []

    def work():
        try:
            for _ in range(3):
                r = search.gbfs(task, heuristic=h, lazy=True)
                results.append([str(a) for a in r.plan])
                assert h.initial_status(task.initial_state) == status
                h(task.initial_state, status)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and ref and results == [ref] * 12


def test_values_statuses_and_searches_do_not_depend_on_atom_slots():
    """LM-cut, the landmark count with its statuses and preferred actions, and searches with them agree under every
    atom mode, lazy slots numbered by a randomized search included."""
    tasks = [text_task("depot__p02", atoms=m) for m in ("dense", "auto", "reachable", "lazy")]
    search.brfs(tasks[-1], max_states=3000, layer_order="randomized", seed=3, stop_at_goal=False)
    labels = [str(a) for a in search.astar(tasks[0], heuristic="max").plan]

    def trace(task):
        cut, count = search.Heuristic(task, "lmcut"), search.Heuristic(task, "lmcount")
        s = task.initial_state
        status = count.initial_status(s)
        out = []
        for name in labels + [None]:
            for a in task.applicable_actions(s):  # every successor of the path, with its progressed status
                t = task.apply(s, a)
                st = count.progress(status, s, a, t)
                out.append((str(a), cut(t), count(t, st), st, sorted(map(str, count.preferred_actions(t, st)))))
            out.append((None, cut(s), count(s, status), status, sorted(map(str, count.preferred_actions(s, status)))))
            if name is None:
                break
            a = next(x for x in task.applicable_actions(s) if str(x) == name)
            t = task.apply(s, a)
            status, s = count.progress(status, s, a, t), t
        return sorted(out, key=lambda x: (x[0] or "",) + x[1:])

    runs = (lambda t: search.astar(t, heuristic="lmcut"),
            lambda t: search.astar(t, heuristic="lmcount", lazy=True),
            lambda t: search.gbfs(t, heuristic="lmcount"),
            lambda t: search.gbfs(t, heuristic="lmcount", lazy=True))
    ref = trace(tasks[0])
    ref_runs = [run(tasks[0]) for run in runs]
    for task in tasks[1:]:
        assert trace(task) == ref, task.atom_mode
        for run, r in zip(runs, ref_runs):
            got = run(task)
            assert [str(a) for a in got.plan] == [str(a) for a in r.plan], task.atom_mode
            assert got.stats.expanded == r.stats.expanded and got.evaluations == r.evaluations, task.atom_mode

SHORTCUT_DOMAIN = """(define (domain d) (:requirements :strips)
 (:predicates (x) (y1) (y) (g))
 (:action get-x :parameters () :precondition (and) :effect (x))
 (:action get-y1 :parameters () :precondition (and) :effect (y1))
 (:action get-y :parameters () :precondition (y1) :effect (y))
 (:action from-x :parameters () :precondition (x) :effect (g))
 (:action from-y :parameters () :precondition (y) :effect (g)))"""


def test_possibly_before_first_achievers_are_sound(tmp_path):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    (tmp_path / "domain.pddl").write_text(SHORTCUT_DOMAIN)
    (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d) (:init) (:goal (g)))")
    task = mymyr.Task.from_pddl(tmp_path / "domain.pddl", tmp_path / "problem.pddl")

    def names(g):
        return {str(a) for a in g.landmarks}

    # the plan get-y1, get-y, from-y never makes x true: only h_max first achievers make it a landmark
    assert names(search.approximate_fact_landmarks(task)) == {"(g)", "(x)"}
    assert names(search.approximate_fact_landmarks(task, first_achievers="possibly_before")) == {"(g)"}
    pb = search.approximate_fact_landmarks(task, first_achievers=search.FirstAchievers.POSSIBLY_BEFORE)
    assert names(pb) == {"(g)"}
    with pytest.raises(ValueError, match="first_achievers"):
        search.approximate_fact_landmarks(task, first_achievers="cheapest")
