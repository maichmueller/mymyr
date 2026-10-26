"""FactLandmarkGraph as data: to_arrays / from_arrays, pickling (also into a spawned process and under lazy atom slots),
the validation of loaded arrays, and the achiever index rebuilt for given landmarks (achiever_index=True) against the
approximate generator's."""

import multiprocessing
import pickle

import numpy as np
import pytest

import mymyr
from mymyr import _pickle, search
from conftest import BLOCKS, ROOT, text_task, walk

IPC = ROOT / "tests/data/ipc"
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def load(name, **options):
    if name == "depot":
        return text_task("depot__p02", **options)
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    if name == "blocks":
        return mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl", **options)
    d = IPC / f"{name}-ipc/train"
    return mymyr.Task.from_pddl(d / "domain.pddl", d / "p02.pddl", **options)


TASKS = ["blocks", "childsnack", "rovers", "depot"]
APPROXIMATE = {
    "hmax": {},
    "possibly_before": {"first_achievers": "possibly_before"},
    "disjunctive": {"first_achievers": "possibly_before", "max_disjunctive_landmark_size": 4},
}


def graph(task, source):
    if source == "lifted":
        return search.lifted_fact_landmarks(task, max_disjunctive_members=4)
    return search.approximate_fact_landmarks(task, **APPROXIMATE[source])


def snapshot(g):
    """Every query of a graph, as strings (comparable across tasks with the same content)."""
    task = g.task
    s = lambda xs: [str(x) for x in xs]  # noqa: E731
    out = {
        "repr": repr(g),
        "len": len(g),
        "landmarks": s(g.landmarks),
        "disjunctive": [s(d) for d in g.disjunctive],
        "disjunctive_atoms": s(g.disjunctive_atoms),
        "orderings": [(str(a), str(b)) for a, b in g.orderings],
        "has_achiever_index": g.has_achiever_index,
        "lifted": [(str(x), x.predicate, x.objects, s(x.members), str(x.fact), x.is_fact, x.parents, x.initially_true)
                   for x in g.lifted],
    }
    atoms = [*g.landmarks, *g.disjunctive_atoms]
    out["is_landmark"] = [g.is_landmark(a) for a in atoms]
    out["predecessors"] = [s(g.predecessors(a)) for a in atoms]
    out["successors"] = [s(g.successors(a)) for a in atoms]
    states = walk(task, steps=10, walks=2)
    out["achieved"] = [s(g.achieved(x)) for x in states]
    out["unachieved"] = [s(g.unachieved(x)) for x in states]
    out["num_achieved"] = [(g.num_achieved(x), g.num_unachieved(x)) for x in states]
    if g.has_achiever_index:
        out["achievers"] = [s(g.achievers(a)) for a in atoms]
        out["first_achievers"] = [s(g.first_achievers(a)) for a in atoms]
        out["unique_achiever"] = [str(g.unique_achiever(a)) for a in atoms]
        actions = {str(x): x for a in atoms for x in g.achievers(a)}
        actions.update({str(x): x for x in task.applicable_actions(task.initial_state)})
        out["by_action"] = {
            k: (s(g.achieved_by(x)), s(g.first_achieved_by(x)), s(g.uniquely_achieved_by(x)), g.is_landmark_achiever(x),
                g.is_first_landmark_achiever(x), g.is_unique_landmark_achiever(x))
            for k, x in sorted(actions.items())}
    else:
        with pytest.raises(ValueError, match="no achiever index"):
            g.achievers(g.landmarks[0])
    return out


def same_arrays(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert np.array_equal(a[k], b[k]), k


# ------------------------------------------------------------------------------------------------ arrays and pickling

@pytest.mark.parametrize("source", [*APPROXIMATE, "lifted"])
@pytest.mark.parametrize("name", TASKS)
def test_pickle_round_trip_answers_every_query_alike(name, source):
    task = load(name)
    g = graph(task, source)
    assert g.has_achiever_index == (source != "lifted")
    assert (len(g.lifted) > 0) == (source == "lifted")
    task2, g2 = pickle.loads(pickle.dumps((task, g)))
    assert g2.task is task2
    assert snapshot(g2) == snapshot(g)
    same_arrays(g2.to_arrays(), g.to_arrays())
    # the graph alone brings its task along
    g3 = pickle.loads(pickle.dumps(g))
    assert g3.task.fingerprint == task.fingerprint
    assert snapshot(g3) == snapshot(g)
    # from_arrays on the same task gives the same graph
    g4 = search.FactLandmarkGraph.from_arrays(task, g.to_arrays())
    assert g4.task is task
    assert snapshot(g4) == snapshot(g)


def test_arrays_layout():
    task = load("depot")
    g = graph(task, "disjunctive")
    a = g.to_arrays()
    assert a["fingerprint"].dtype == np.uint64 and a["fingerprint"].shape == (1,)
    assert int(a["fingerprint"][0]) == task.fingerprint
    assert a["landmarks"].dtype == np.uint64 and a["landmarks"].shape == (len(g),)
    assert a["orderings"].shape == (len(g.orderings), 2)
    assert a["disjunctive_offsets"].shape == (len(g.disjunctive) + 1,)
    assert int(a["disjunctive_offsets"][-1]) == len(a["disjunctive_atoms"]) == sum(len(d) for d in g.disjunctive)
    n_actions = len(a["actions_schema"])
    assert a["actions_schema"].dtype == a["actions_binding"].dtype == np.int32
    assert a["actions_binding"].ndim == 2 and a["actions_binding"].shape[0] == n_actions
    assert a["achievers_offsets"].shape == (len(a["atoms"]) + 1,)
    assert a["achieved_offsets"].shape == a["uniquely_achieved_offsets"].shape == (n_actions + 1,)
    assert a["lifted_predicate"].shape == (0,)
    # without an index there are no index arrays; the lifted ones carry the lifted landmarks
    lifted = graph(task, "lifted").to_arrays()
    assert "actions_schema" not in lifted
    assert lifted["lifted_fact"].shape == lifted["lifted_initially_true"].shape == lifted["lifted_predicate"].shape
    assert (lifted["lifted_binding"] >= -1).all()
    # the arrays are fresh, writable copies
    first = int(a["landmarks"][0])
    a["landmarks"][0] = first + 1
    assert int(g.to_arrays()["landmarks"][0]) == first


def test_lazy_atom_slots():
    task = text_task("gripper__prob05", atoms="lazy")
    g = search.approximate_fact_landmarks(task, max_disjunctive_landmark_size=4)
    before = snapshot(g)
    blob = pickle.dumps((task, g))
    task2, g2 = pickle.loads(blob)
    assert snapshot(g2) == before
    # a fresh lazy task touches its atoms in another order; canonical ids do not depend on it
    fresh = text_task("gripper__prob05", atoms="lazy")
    walk(fresh, steps=15, seed=3)
    g3 = search.FactLandmarkGraph.from_arrays(fresh, g.to_arrays())
    assert snapshot(g3) == snapshot(search.approximate_fact_landmarks(fresh, max_disjunctive_landmark_size=4))
    # and the frozen twin of the same content reads the same arrays
    frozen = text_task("gripper__prob05")
    assert snapshot(search.FactLandmarkGraph.from_arrays(frozen, g.to_arrays())) == snapshot(
        search.approximate_fact_landmarks(frozen, max_disjunctive_landmark_size=4))


def _digest(blob):
    g = pickle.loads(blob)
    return snapshot(g), {k: v.tolist() for k, v in g.to_arrays().items()}


def test_spawned_process_receives_the_graph():
    task = load("depot")
    graphs = [graph(task, "disjunctive"), graph(task, "lifted")]
    blobs = [pickle.dumps(g) for g in graphs]
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        got = pool.map(_digest, blobs)
    assert got == [_digest(b) for b in blobs]


def test_restoring_onto_another_task_raises():
    g = graph(load("depot"), "hmax")
    other = text_task("gripper__prob05")
    with pytest.raises(ValueError, match="another task"):
        _pickle._restore_landmark_graph(other, g.task.fingerprint, g.to_arrays())
    with pytest.raises(ValueError, match="another task"):
        _pickle._restore_landmark_graph(g.task, g.task.fingerprint ^ 1, g.to_arrays())
    # another task's arrays under that task's own fingerprint fail validation
    with pytest.raises(ValueError, match="invalid landmark graph|landmark graph arrays"):
        _pickle._restore_landmark_graph(other, other.fingerprint, g.to_arrays())
    restored = _pickle._restore_landmark_graph(g.task, g.task.fingerprint, g.to_arrays())
    assert snapshot(restored) == snapshot(g)


@needs_frontend
@pytest.mark.parametrize("source", ["hmax", "lifted"])
def test_from_arrays_checks_task_fingerprint(source):
    task = load("childsnack")
    d = IPC / "childsnack-ipc/train"
    other = mymyr.Task.from_pddl(d / "domain.pddl", d / "p01.pddl")
    assert task.num_objects == other.num_objects
    assert task.fingerprint != other.fingerprint
    g = graph(task, source)
    arrays = g.to_arrays()
    with pytest.raises(ValueError, match="another task.*fingerprint mismatch"):
        search.FactLandmarkGraph.from_arrays(other, arrays)
    with pytest.raises(ValueError, match="another task.*fingerprint mismatch"):
        _pickle._restore_landmark_graph(other, other.fingerprint, arrays)
    same = load("childsnack")
    restored = search.FactLandmarkGraph.from_arrays(same, arrays)
    assert snapshot(restored) == snapshot(g)


def test_from_arrays_without_fingerprint():
    task = load("depot")
    g = graph(task, "disjunctive")
    arrays = g.to_arrays()
    del arrays["fingerprint"]
    assert snapshot(search.FactLandmarkGraph.from_arrays(task, arrays)) == snapshot(g)


@pytest.mark.parametrize("fingerprint,exc", [
    (np.array([], dtype=np.uint64), ValueError),
    (np.array([0, 0], dtype=np.uint64), ValueError),
    (np.array([[0]], dtype=np.uint64), TypeError),
    (np.array([0], dtype=np.int64), TypeError),
    (np.array([0], dtype=np.uint64), ValueError),
])
def test_from_arrays_rejects_invalid_fingerprint(fingerprint, exc):
    task = load("depot")
    assert task.fingerprint != 0
    arrays = dict(graph(task, "hmax").to_arrays(), fingerprint=fingerprint)
    with pytest.raises(exc, match="fingerprint"):
        search.FactLandmarkGraph.from_arrays(task, arrays)


def corrupt_cases(a):
    """(name, arrays, exception, message) for corruptions of a graph's arrays."""
    def with_(**changes):
        b = {k: v.copy() for k, v in a.items()}
        for k, v in changes.items():
            if v is None:
                del b[k]
            else:
                b[k] = v
        return b

    big = np.uint64(2**40)
    lm = a["landmarks"].copy()
    lm[0] = big
    offs = a["achievers_offsets"].copy()
    offs[1], offs[2] = offs[-1], 0  # decreasing (or not ending at the list's length)
    actions = a["actions_binding"].copy()
    actions[[0, 1]] = actions[[1, 0]]
    schema = a["actions_schema"].copy()
    schema[[0, 1]] = schema[[1, 0]]
    pad = a["actions_binding"].copy()
    pad[0, -1] = 0 if pad[0, -1] == -1 else -2
    achieved = a["achieved"].copy()
    achieved[0] = a["landmarks"][-1] if achieved[0] != a["landmarks"][-1] else a["landmarks"][0]
    return [
        ("atom outside", with_(landmarks=lm), ValueError, "canonical space"),
        ("unknown name", with_(extra=a["landmarks"]), ValueError, "unknown landmark graph array"),
        ("dtype", with_(landmarks=a["landmarks"].astype(np.int64)), TypeError, "uint64"),
        ("missing", with_(landmarks=None), ValueError, "missing"),
        ("partial index", with_(achievers=None), ValueError, "all together"),
        ("offsets", with_(achievers_offsets=offs), ValueError, "achievers"),
        ("offsets end", with_(disjunctive_offsets=np.array([0, 5], dtype=np.uint64)), ValueError, "disjunctive_offsets"),
        ("orderings shape", with_(orderings=np.zeros((1, 3), dtype=np.uint64)), ValueError, "orderings"),
        ("unsorted actions", with_(actions_binding=actions, actions_schema=schema), ValueError, "ascending"),
        ("binding pad", with_(actions_binding=pad), ValueError, "padded with -1"),
        ("schema", with_(actions_schema=np.full_like(a["actions_schema"], 999)), ValueError, "schema id"),
        ("per-action lists", with_(achieved=achieved), ValueError, "disagree"),
        ("not a mapping", [1, 2], TypeError, "mapping"),
    ]


@pytest.mark.parametrize("case", range(13))
def test_from_arrays_rejects_corrupt_arrays(case):
    task = load("depot")
    g = graph(task, "disjunctive")
    name, arrays, exc, msg = corrupt_cases(g.to_arrays())[case]
    with pytest.raises(exc, match=msg):
        search.FactLandmarkGraph.from_arrays(task, arrays)


def test_from_arrays_rejects_corrupt_lifted_arrays():
    task = load("depot")
    a = graph(task, "lifted").to_arrays()
    b = dict(a, lifted_members=a["lifted_members"][::-1].copy())
    with pytest.raises(ValueError, match="lifted landmark"):
        search.FactLandmarkGraph.from_arrays(task, b)
    b = dict(a, lifted_parents=np.full_like(a["lifted_parents"], 10**6))
    if len(b["lifted_parents"]):
        with pytest.raises(ValueError, match="parent"):
            search.FactLandmarkGraph.from_arrays(task, b)
    b = dict(a)
    del b["lifted_fact"]
    with pytest.raises(ValueError, match="all together"):
        search.FactLandmarkGraph.from_arrays(task, b)
    b = dict(a, lifted_binding=np.full_like(a["lifted_binding"], -5))
    with pytest.raises(ValueError, match="below -1"):
        search.FactLandmarkGraph.from_arrays(task, b)


@pytest.mark.parametrize("source", ["hmax", "lifted"])
@pytest.mark.parametrize("name", ["blocks", "depot"])
def test_liw_with_the_unpickled_graph(name, source):
    task = load(name)
    g = graph(task, source)
    task2, g2 = pickle.loads(pickle.dumps((task, g)))

    def key(r):
        return (str(r.status), [(p.width, p.expanded, p.generated) for p in r.passes], [str(a) for a in r.plan])

    assert key(search.liw(task2, max_width=1, landmarks=g2)) == key(search.liw(task, max_width=1, landmarks=g))


# ------------------------------------------------------------------------------------------------ achiever index rebuild

@pytest.mark.parametrize("source", list(APPROXIMATE))
@pytest.mark.parametrize("name", TASKS)
def test_rebuilt_index_equals_the_generators(name, source):
    task = load(name)
    g = graph(task, source)
    rule = APPROXIMATE[source].get("first_achievers", "hmax")
    plain = search.FactLandmarkGraph(task, g.landmarks, g.disjunctive, g.orderings)
    assert not plain.has_achiever_index
    r = search.FactLandmarkGraph(task, g.landmarks, g.disjunctive, g.orderings, achiever_index=True, first_achievers=rule)
    assert r.has_achiever_index
    a, b = r.to_arrays(), g.to_arrays()
    same = ["landmarks", "disjunctive_offsets", "disjunctive_atoms", "orderings", "actions_schema", "actions_binding",
            "atoms", "achievers_offsets", "achievers", "achieved_offsets", "achieved", "first_achieved_offsets",
            "first_achieved", "uniquely_achieved_offsets", "uniquely_achieved"]
    if rule == "hmax":
        same += ["first_achievers_offsets", "first_achievers"]
    for k in same:
        assert np.array_equal(a[k], b[k]), k
    for lm in g.landmarks:
        assert r.achievers(lm) == g.achievers(lm)
        assert r.first_achievers(lm) == g.first_achievers(lm)
        assert r.unique_achiever(lm) == g.unique_achiever(lm)
    if rule == "possibly_before":
        # the rebuild lists first achievers for every disjunctive member, the generator for those it expanded
        for m in g.disjunctive_atoms:
            assert set(map(str, r.first_achievers(m))) <= set(map(str, r.achievers(m)))


def test_rebuild_for_a_hand_made_subgoal_set():
    task = load("depot")
    g = graph(task, "hmax")
    subgoals = g.landmarks[:3]
    r = search.FactLandmarkGraph(task, subgoals, achiever_index=True)
    assert [str(x) for x in r.landmarks] == [str(x) for x in subgoals]
    for lm in subgoals:
        assert r.achievers(lm) == g.achievers(lm)
        assert r.first_achievers(lm) == g.first_achievers(lm)
    # per action: only the given landmarks
    for x in {str(x): x for lm in subgoals for x in g.achievers(lm)}.values():
        assert set(map(str, r.achieved_by(x))) == set(map(str, g.achieved_by(x))) & set(map(str, subgoals))
    # it pickles with its index
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.has_achiever_index and snapshot(r2) == snapshot(r)


def test_ground_action_budget():
    task = load("depot")
    n = len(graph(task, "hmax").to_arrays()["actions_schema"])
    with pytest.raises(RuntimeError, match="ground actions"):
        search.FactLandmarkGraph(task, [], achiever_index=True, max_ground_actions=n - 1)
    with pytest.raises(RuntimeError, match="ground actions"):
        search.approximate_fact_landmarks(task, max_ground_actions=n - 1)
    assert search.FactLandmarkGraph(task, [], achiever_index=True, max_ground_actions=n).has_achiever_index
    with pytest.raises(ValueError, match="first_achievers"):
        search.FactLandmarkGraph(task, [], achiever_index=True, first_achievers="cheapest")
    with pytest.raises(ValueError, match="no achiever index"):
        search.FactLandmarkGraph(task, [], max_ground_actions=n).achieved_by(task.applicable_actions(task.initial_state)[0])
