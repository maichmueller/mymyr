"""Persistence: StateSpace.from_arrays, pickling (datasets, search results, heuristics) and dataset files.

Round trips are exact: arrays bitwise equal, the same states, labels, goal distances, sampler draws, tuple graphs and
plans. Workers started with the spawn method receive state spaces and knowledge bases by pickle.
"""

import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import pickle
import threading
import zipfile

import numpy as np
import pytest

import mymyr
from mymyr import datasets, search
from mymyr.datasets import _files
from mymyr.rl import TaskTable

from conftest import ROOT, text_task

PDDL = ROOT / "tests/data/pddl"
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def pddl_task(d, p, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(PDDL / d / "domain.pddl", PDDL / d / f"{p}.pddl", **options)


def same_arrays(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), k


def _state(space):
    """What a state space pickles: (task, fingerprints, lazy slots, statistics, arrays with the state rows)."""
    from mymyr._core._datasets import _state_space_state

    return _state_space_state(space)


def same_space(s, t, *, statistics=True):
    """s and t are the same state space (t possibly of another task object of the same task)."""
    assert (t.num_states, t.num_transitions, t.words, t.numeric_words, t.has_labels) == (
        s.num_states, s.num_transitions, s.words, s.numeric_words, s.has_labels)
    assert t.task.fingerprint == s.task.fingerprint and t.task.encoding_fingerprint == s.task.encoding_fingerprint
    same_arrays(_state(s)[5], _state(t)[5])
    assert t.max_goal_distance == s.max_goal_distance and t.unit_costs == s.unit_costs
    for i in range(0, s.num_states, max(1, s.num_states // 7)):
        assert repr(t.state(i)) == repr(s.state(i))
        assert t.find(t.state(i)) == i and t.unit_goal_distance(i) == s.unit_goal_distance(i)
        assert t.cost_goal_distance(i) == s.cost_goal_distance(i)
    for e in range(0, s.num_transitions, max(1, s.num_transitions // 7)):
        assert (t.source(e), t.target(e), t.cost(e)) == (s.source(e), s.target(e), s.cost(e))
        if s.has_labels:
            assert repr(t.label(e)) == repr(s.label(e))
    a, b = datasets.StateSpaceSampler(s, seed=5), datasets.StateSpaceSampler(t, seed=5)
    assert np.array_equal(a.sample_states(300), b.sample_states(300))
    for n in range(s.max_goal_distance + 1):
        assert np.array_equal(a.sample_states_n_steps_from_goal(n, 20), b.sample_states_n_steps_from_goal(n, 20))
    if s.num_unsolvable_states:
        assert np.array_equal(a.sample_dead_end_states(20), b.sample_dead_end_states(20))
    if statistics:
        assert (t.threads, t.layers, t.search_seconds, t.post_seconds) == (s.threads, s.layers, s.search_seconds,
                                                                          s.post_seconds)


def same_graph(g, h):
    assert (g.root, g.width, g.dominance_pruning) == (h.root, h.width, h.dominance_pruning)
    same_arrays(g.arrays(), h.arrays())


# (dir, problem, generation options): unit and real costs, numeric fluents, derived atoms and unsolvable states
SPACES = [
    ("gripper", "p02", {}),
    ("counters", "p01", {}),
    ("counters", "p02", {"remove_if_unsolvable": False}),
    ("blocks_flags", "p01", {"labels": False}),
    ("philosophers", "p03-phil4-one-blocked", {"remove_if_unsolvable": False}),
]


# ------------------------------------------------------------------------------------------------ from_arrays
@pytest.mark.parametrize("case", SPACES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_from_arrays_round_trips(case):
    d, p, options = case
    task = pddl_task(d, p)
    s = datasets.state_space(task, threads=2, **options)
    t = datasets.StateSpace.from_arrays(task, s.arrays())
    assert t.task is task
    same_space(s, t, statistics=False)
    same_arrays(s.arrays(), t.arrays())
    assert (t.threads, t.layers, t.search_seconds, t.post_seconds) == (1, 0, 0.0, 0.0)
    for v in range(0, s.num_states, max(1, s.num_states // 5)):
        same_graph(datasets.tuple_graph(s, v, width=1), datasets.tuple_graph(t, v, width=1))
    # DLPack arrays on the CPU work the same
    same_arrays(datasets.StateSpace.from_arrays(task, s.arrays("dlpack")).arrays(), s.arrays())


def test_from_arrays_of_a_text_task():
    task = text_task("philosophers__p03-phil4", atoms="dense")
    s = datasets.state_space(task, threads=2, remove_if_unsolvable=False)
    t = datasets.StateSpace.from_arrays(task, s.arrays())
    same_space(s, t, statistics=False)
    same_space(s, pickle.loads(pickle.dumps(s)))


@needs_frontend
def test_from_arrays_validates():
    task = pddl_task("counters", "p01")
    a = datasets.state_space(task).arrays()

    def edit(name, f):
        b = dict(a)
        b[name] = f(a[name].copy())
        return b

    def flip(x):
        x[0] = not x[0]
        return x

    def bad_label(x):
        x[0, 0] = 999
        return x

    def nan(x):
        x[0] = np.nan
        return x

    def broken_reverse(x):
        x[0] ^= 1
        return x

    cases = [
        (edit("forward_targets", lambda x: x.astype(np.int64)), TypeError, "'forward_targets' must be a 1-d uint32"),
        ({**a, "extra": a["goal"]}, ValueError, "unknown state space array 'extra'"),
        ({k: v for k, v in a.items() if k != "forward_offsets"}, ValueError, "'forward_offsets' is missing"),
        (edit("goal", flip), ValueError, "do not match the unit goal distances"),
        (edit("state_words", lambda x: x[::-1].copy()), ValueError, "is not the task's initial state"),
        (edit("label_bindings", bad_label), ValueError, "binds an object the task does not have"),
        (edit("costs", nan), ValueError, "a transition cost is NaN"),
        (edit("backward_sources", broken_reverse), ValueError, "invalid state space arrays"),
        (edit("unit_goal_distances", lambda x: x[:-1]), ValueError, "not [num_states]"),
        ({k: v.tolist() for k, v in a.items()}, TypeError, "must be an array in host memory"),
    ]
    for arrays, error, message in cases:
        with pytest.raises(error, match=f"^mymyr: .*{message.replace('[', '.').replace(']', '.')}"):
            datasets.StateSpace.from_arrays(task, arrays)
    with pytest.raises(TypeError, match="^mymyr: .*mapping"):
        datasets.StateSpace.from_arrays(task, list(a.values()))
    with pytest.raises(TypeError, match="^mymyr: task must be"):
        datasets.StateSpace.from_arrays("task", a)
    with pytest.raises(ValueError, match="belong to another task"):  # rows of another problem
        datasets.StateSpace.from_arrays(pddl_task("counters", "p03-half"), a)
    with pytest.raises(ValueError, match="belong to another task"):  # another numeric layout
        datasets.StateSpace.from_arrays(pddl_task("counters", "p01", numeric_storage="f64"), a)


# ------------------------------------------------------------------------------------------------ pickling
@pytest.mark.parametrize("case", SPACES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_pickle_state_spaces(case):
    d, p, options = case
    task = pddl_task(d, p)
    s = datasets.state_space(task, threads=2, **options)
    t = pickle.loads(pickle.dumps(s))
    assert t.task is not task and t.task.fingerprint == task.fingerprint
    same_space(s, t)
    t2, back = pickle.loads(pickle.dumps((task, s)))
    assert back.task is t2  # one task object in one pickle
    r = datasets.generate(task, max_states=5)
    for x in (r, datasets.generate(task, **options)):
        y = pickle.loads(pickle.dumps(x))
        assert (y.status, y.states, y.seconds, bool(y)) == (x.status, x.states, x.seconds, bool(x))
        assert (y.space is None) == (x.space is None)
        if x.space is not None:
            same_space(x.space, y.space)


@needs_frontend
def test_pickle_tuple_graphs_generalized_spaces_and_knowledge_bases():
    g = PDDL / "gripper"
    table = TaskTable.from_pddl(g / "domain.pddl", [g / "p01.pddl", g / "p02.pddl", g / "test_problem.pddl"])
    kb = datasets.KnowledgeBase(table, generalized=True, width=1, threads=2)
    back = pickle.loads(pickle.dumps(kb))
    assert back.task_indices == kb.task_indices and back.width == kb.width and len(back.tasks) == len(kb.tasks)
    for s, t in zip(kb.state_spaces, back.state_spaces, strict=True):
        same_space(s, t)  # the generation statistics too: nothing is generated again
    for i in range(len(kb.state_spaces)):
        for x, y in zip(kb.tuple_graphs(i), back.tuple_graphs(i), strict=True):
            same_graph(x, y)
            assert y.space is back.state_spaces[i]
    same_arrays(kb.generalized_state_space.arrays(), back.generalized_state_space.arrays())
    assert back.state_spaces[0].task is back.tasks[back.task_indices[0]]
    plain = pickle.loads(pickle.dumps(datasets.KnowledgeBase(table, sort_by_size=False)))
    assert plain.task_indices == [0, 1, 2] and not plain.has_tuple_graphs and plain.generalized_state_space is None
    # generalized state spaces and tuple graphs on their own; one state space per pickle
    gss = pickle.loads(pickle.dumps(kb.generalized_state_space))
    same_arrays(gss.arrays(), kb.generalized_state_space.arrays())
    graphs = datasets.tuple_graphs(kb.state_spaces[2], width=2, dominance_pruning=False)
    copies = pickle.loads(pickle.dumps(graphs))
    assert all(c.space is copies[0].space for c in copies)
    for x, y in zip(graphs, copies, strict=True):
        same_graph(x, y)
        assert x == datasets.tuple_graph(kb.state_spaces[2], x.root, width=2, dominance_pruning=False)
    one = pickle.loads(pickle.dumps(graphs[3]))
    same_graph(graphs[3], one)


def _iw(t):
    return search.iw(t, max_arity=2)


SEARCHES = {
    "iw": _iw,
    "siw": lambda t: search.siw(t, max_arity=2),
    "astar": lambda t: search.astar(t, heuristic="lmcut"),
    "gbfs": lambda t: search.gbfs(t, heuristic="ff"),
    "beam": lambda t: search.beam(t, width=4),
    "brfs": search.brfs,
    "astar_iw": search.astar_iw,
    "rollout_iw": lambda t: search.rollout_iw(t, seed=3),
    "liw": search.liw,
    "iw_pass": lambda t: search.iw_pass(t, arity=1),
    "abstracted_iw": search.abstracted_iw,
    "projective_iw": search.projective_iw,
    "rollouts": lambda t: search.find_rollouts_parallel(t, [1, 2, 3], report_landing_states=True),
    "merged": lambda t: search.merge_landing_states(search.find_rollouts_parallel(t, [1, 2], report_landing_states=True)),
    "portfolio": search.atomic_goal_portfolio,
    "unsolved": lambda t: search.astar(t, heuristic="ff", max_expanded=1),
}


def public(x):
    """The public attributes of x, nested search records as dicts, other values by repr (states, actions, plans)."""
    if isinstance(x, (list, tuple)):
        return [public(v) for v in x]
    if type(x).__module__ == "mymyr._core._search" and not hasattr(x, "name"):
        out = {}
        for n in dir(x):
            if n.startswith("_") or n in ("arrays",):
                continue
            v = getattr(x, n)
            if not callable(v):
                out[n] = public(v)
        return out
    return repr(x)


@pytest.mark.parametrize("name", list(SEARCHES))
def test_pickle_search_results(name):
    task = pddl_task("gripper", "p02")
    r = SEARCHES[name](task)
    t2, back = pickle.loads(pickle.dumps((task, r)))
    assert type(back) is type(r) and public(back) == public(r)
    if hasattr(r, "plan"):
        assert [str(a) for a in back.plan] == [str(a) for a in r.plan]
        if r.plan:
            assert search.format_plan(t2, back.plan) == search.format_plan(task, r.plan)
    if getattr(r, "goal_state", None) is not None:
        assert t2.state(back.goal_state) == back.goal_state  # a state of the unpickled task
    if hasattr(r, "rollouts"):
        assert public(pickle.loads(pickle.dumps(r.rollouts[0]))) == public(r.rollouts[0])


@needs_frontend
def test_search_result_pickles_are_checked():
    task = pddl_task("gripper", "p01")
    r = search.iw(task, max_arity=2)
    assert r.plan
    restore, (kind, t, fingerprint, fields) = r.__reduce__()
    assert restore(kind, t, fingerprint, fields).plan == r.plan
    plan = fields[1]
    bad_schema = [(999, plan[0][1])] + plan[1:]
    bad_object = [(plan[0][0], (999,) * len(plan[0][1]))] + plan[1:]
    for f in (fields[:-1], fields + (0,), (fields[0], bad_schema) + fields[2:], (fields[0], bad_object) + fields[2:],
              (99,) + fields[1:]):
        with pytest.raises(ValueError, match="^mymyr: corrupt IW result pickle"):
            restore(kind, t, fingerprint, f)
    with pytest.raises(ValueError, match="^mymyr: unknown search result pickle kind"):
        restore("dfs", t, fingerprint, fields)
    with pytest.raises(ValueError, match="different task"):
        restore(kind, pddl_task("gripper", "p02"), fingerprint, fields)
    with pytest.raises(TypeError):
        restore(kind, "task", fingerprint, fields)


@needs_frontend
@pytest.mark.parametrize("kind", ["blind", "goal_count", "max", "add", "ff", "set_additive", "h2", "lmcut", "lmcount"])
def test_pickle_heuristics(kind):
    task = pddl_task("gripper", "p02")
    h = search.Heuristic(task, kind)
    space = datasets.state_space(task)
    t2, s2, back = pickle.loads(pickle.dumps((task, space, h)))
    assert back.kind == h.kind and s2.task is t2
    for i in range(space.num_states):
        assert back(s2.state(i)) == h(space.state(i))
    p = search.Heuristic.perfect(space, costs="unit")
    s2, q = pickle.loads(pickle.dumps((space, p)))
    assert q.kind == "perfect"
    for i in range(space.num_states):
        assert q(s2.state(i)) == p(space.state(i)) == space.unit_goal_distance(i)
    stats = pickle.loads(pickle.dumps(h.stats))
    assert public(stats) == public(h.stats)


@needs_frontend
def test_lazy_atom_slots_round_trip_exactly(tmp_path):
    g = PDDL / "gripper"
    task = mymyr.Task.from_pddl(g / "domain.pddl", g / "p02.pddl", atoms="lazy")
    s = datasets.state_space(task)
    assert "state_words" not in s.arrays()
    with pytest.raises(ValueError, match="^mymyr: a state space of a task with lazy atom slots is not rebuilt"):
        datasets.StateSpace.from_arrays(task, {**s.arrays(), "state_words": _state(s)[5]["state_words"]})
    back = pickle.loads(pickle.dumps(s))
    assert back.task.atom_mode == "lazy"
    same_space(s, back)
    datasets.save(s, tmp_path / "lazy.mymyr")
    same_space(s, datasets.load(tmp_path / "lazy.mymyr"))
    fresh = mymyr.Task.from_pddl(g / "domain.pddl", g / "p02.pddl", atoms="lazy")
    loaded = datasets.load(tmp_path / "lazy.mymyr", task=fresh)
    assert loaded.task is fresh and fresh.num_atoms == task.num_atoms
    same_space(s, loaded)
    # a task whose atoms got their slots in another order cannot hold these rows
    other = mymyr.Task.from_pddl(g / "domain.pddl", g / "p02.pddl", atoms="lazy")
    other.state([str(a) for a in reversed(s.state(s.num_states - 1).atoms())])
    slots = other.num_atoms
    with pytest.raises(ValueError, match="cannot number its lazy atom slots as the state space does"):
        datasets.load(tmp_path / "lazy.mymyr", task=other)
    assert other.num_atoms == slots  # refused before any slot is assigned
    # a task whose slots so far agree with the saved numbering takes the rest
    prefix = mymyr.Task.from_pddl(g / "domain.pddl", g / "p02.pddl", atoms="lazy")
    prefix.successor_states(prefix.initial_state)
    assert 0 < prefix.num_atoms < task.num_atoms
    same_space(s, datasets.load(tmp_path / "lazy.mymyr", task=prefix))
    assert prefix.num_atoms == task.num_atoms
    with pytest.raises(ValueError, match="saved under lazy atom slots; the task has frozen ones"):
        datasets.load(tmp_path / "lazy.mymyr", task=mymyr.Task.from_pddl(g / "domain.pddl", g / "p02.pddl"))


@needs_frontend
def test_device_state_spaces_pass_the_full_check():
    """to_host() of a device space builds it checking sizes and ranges only; its contents pass the full check."""
    cuda = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
    if not cuda.available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    for d, p, options in SPACES:
        task = pddl_task(d, p)
        host = datasets.state_space(task, device=0, **options).to_host()
        same_arrays(datasets.StateSpace.from_arrays(task, host.arrays()).arrays(), host.arrays())
        same_arrays(host.arrays(), datasets.state_space(task, **options).arrays())
        same_space(host, pickle.loads(pickle.dumps(host)))


@needs_frontend
def test_concurrent_pickling():
    task = pddl_task("philosophers", "p03-phil4-one-blocked")
    s = datasets.state_space(task, remove_if_unsolvable=False, threads=2)
    graphs = datasets.tuple_graphs(s, width=1)[:50]
    expected = pickle.dumps((s, graphs))
    errors = []

    def work():
        try:
            for _ in range(3):
                assert pickle.dumps((s, graphs)) == expected
                t, gs = pickle.loads(expected)
                assert all(g.space is t for g in gs)
                same_graph(gs[7], graphs[7])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors[0]


# ------------------------------------------------------------------------------------------------ files
@needs_frontend
def test_save_and_load_state_spaces(tmp_path):
    task = pddl_task("counters", "p01")
    s = datasets.state_space(task)
    path = tmp_path / "counters.mymyr"
    datasets.save(s, path)
    same_space(s, datasets.load(path))
    same_space(s, datasets.load(str(path), task=task))
    assert datasets.load(path, task=task).task is task
    with zipfile.ZipFile(path) as z:
        header = json.loads(z.read("header.json"))
        names = z.namelist()
    assert (header["format"], header["version"], header["kind"]) == ("mymyr.datasets", 1, "state_space")
    assert header["mymyr"] == mymyr.__version__ and header["tasks"][0]["fingerprint"] == f"{task.fingerprint:016x}"
    assert header["tasks"][0]["source"] == "pddl" and header["tasks"][0]["atom_slots"] is None
    assert all(n == "header.json" or n.endswith((".npy", ".pddl", ".txt")) for n in names)
    with np.load(path, allow_pickle=False) as z:  # an .npz: every array reads without pickle
        npy = [n for n in z.files if n.startswith("state_spaces/0/")]
        assert np.array_equal(z["state_spaces/0/forward_targets"], s.arrays()["forward_targets"]) and len(npy) >= 10
    # compressed, and file objects
    datasets.save(s, tmp_path / "small.mymyr", compress=True)
    same_space(s, datasets.load(tmp_path / "small.mymyr"))
    buf = io.BytesIO()
    datasets.save(s, buf)
    buf.seek(0)
    same_space(s, datasets.load(buf, task=task))
    # a text task
    tt = text_task("philosophers__p03-phil4", atoms="dense")
    ts = datasets.state_space(tt, remove_if_unsolvable=False)
    datasets.save(ts, tmp_path / "text.mymyr")
    assert datasets.load(tmp_path / "text.mymyr").task.fingerprint == tt.fingerprint
    same_space(ts, datasets.load(tmp_path / "text.mymyr", task=tt))


@needs_frontend
def test_save_and_load_knowledge_bases_and_tuple_graphs(tmp_path):
    g = PDDL / "gripper"
    table = TaskTable.from_pddl(g / "domain.pddl", [g / "p01.pddl", g / "p02.pddl"])
    kb = datasets.KnowledgeBase(table, generalized=True, width=1)
    datasets.save(kb, tmp_path / "kb.mymyr")
    for back in (datasets.load(tmp_path / "kb.mymyr"), datasets.load(tmp_path / "kb.mymyr", tasks=table)):
        assert back.task_indices == kb.task_indices
        for i, (s, t) in enumerate(zip(kb.state_spaces, back.state_spaces, strict=True)):
            same_space(s, t)
            for x, y in zip(kb.tuple_graphs(i), back.tuple_graphs(i), strict=True):
                same_graph(x, y)
        same_arrays(kb.generalized_state_space.arrays(), back.generalized_state_space.arrays())
    assert datasets.load(tmp_path / "kb.mymyr", tasks=list(table.tasks)).tasks[0] is table[0]
    with zipfile.ZipFile(tmp_path / "kb.mymyr") as z:
        assert sum(n.startswith("domains/") for n in z.namelist()) == 1  # one domain file for the table
    gss = kb.generalized_state_space
    datasets.save(gss, tmp_path / "gss.mymyr")
    same_arrays(datasets.load(tmp_path / "gss.mymyr").arrays(), gss.arrays())
    graphs = kb.tuple_graphs(1)
    datasets.save(graphs, tmp_path / "graphs.mymyr")
    loaded = datasets.load(tmp_path / "graphs.mymyr")
    assert len(loaded) == len(graphs) and all(x.space is loaded[0].space for x in loaded)
    for x, y in zip(graphs, loaded, strict=True):
        same_graph(x, y)
    datasets.save(graphs[2], tmp_path / "graph.mymyr")
    same_graph(graphs[2], datasets.load(tmp_path / "graph.mymyr"))
    with pytest.raises(TypeError, match="^mymyr: datasets.save takes .* a nonempty list of TupleGraphs"):
        datasets.save([], tmp_path / "none.mymyr")
    with pytest.raises(TypeError, match="^mymyr: datasets.save takes"):
        datasets.save(table, tmp_path / "table.mymyr")


def rewrite(path, out, edit_header=None, drop=None):
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(out, "w") as w:
        for n in z.namelist():
            if n == drop:
                continue
            data = z.read(n)
            if n == "header.json" and edit_header:
                h = json.loads(data)
                edit_header(h)
                data = json.dumps(h).encode()
            w.writestr(n, data)


@needs_frontend
def test_load_refuses_other_tasks_newer_versions_and_corrupt_files(tmp_path):
    task = pddl_task("counters", "p01")
    path = tmp_path / "s.mymyr"
    datasets.save(datasets.state_space(task), path)
    with pytest.raises(ValueError, match="^mymyr: task 0 given for .* is another task than the file's"):
        datasets.load(path, task=pddl_task("counters", "p03-half"))
    with pytest.raises(ValueError, match="^mymyr: .*encoding fingerprint mismatch"):
        datasets.load(path, task=pddl_task("counters", "p01", numeric_storage="f64"))
    with pytest.raises(ValueError, match="^mymyr: .* has 1 task"):
        datasets.load(path, tasks=[task, task])
    with pytest.raises(TypeError, match="^mymyr: pass task= or tasks="):
        datasets.load(path, task=task, tasks=[task])

    def version(h):
        h["version"] = _files.VERSION + 1

    def kind(h):
        h["kind"] = "replay_buffer"

    def fmt(h):
        h["format"] = "other"

    def fingerprint(h):
        h["tasks"][0]["fingerprint"] = "zz"

    cases = [(version, "dataset format version 2"), (kind, "holds a 'replay_buffer'"), (fmt, "not a mymyr dataset file"),
             (fingerprint, "corrupt dataset file")]
    for i, (edit, message) in enumerate(cases):
        rewrite(path, tmp_path / f"{i}.mymyr", edit_header=edit)
        with pytest.raises(ValueError, match=f"^mymyr: .*{message}"):
            datasets.load(tmp_path / f"{i}.mymyr")
    rewrite(path, tmp_path / "unknown.mymyr", edit_header=lambda h: h.update(encodings=[{"schema": 1}]))
    same_space(datasets.state_space(task), datasets.load(tmp_path / "unknown.mymyr"), statistics=False)
    rewrite(path, tmp_path / "member.mymyr", drop="state_spaces/0/forward_targets.npy")
    with pytest.raises(ValueError, match="^mymyr: the state space array 'forward_targets' is missing"):
        datasets.load(tmp_path / "member.mymyr")
    rewrite(path, tmp_path / "problem.mymyr", drop=json.loads(zipfile.ZipFile(path).read("header.json"))["tasks"][0]["problem"])
    with pytest.raises(ValueError, match="^mymyr: corrupt dataset file .*no member"):
        datasets.load(tmp_path / "problem.mymyr")
    rewrite(path, tmp_path / "header.mymyr", drop="header.json")
    with pytest.raises(ValueError, match="^mymyr: .* is not a mymyr dataset file"):
        datasets.load(tmp_path / "header.mymyr")
    (tmp_path / "text.mymyr").write_text("not a zip")
    with pytest.raises(ValueError, match="^mymyr: .* is not a mymyr dataset file .not a zip archive"):
        datasets.load(tmp_path / "text.mymyr")
    # arrays are read without pickle: an object array is refused
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(tmp_path / "object.mymyr", "w") as w:
        for n in z.namelist():
            if n == "state_spaces/0/goal.npy":
                b = io.BytesIO()
                np.save(b, np.array([object()] * 38, dtype=object), allow_pickle=True)
                w.writestr(n, b.getvalue())
            else:
                w.writestr(n, z.read(n))
    with pytest.raises(ValueError, match="^mymyr: corrupt dataset file"):
        datasets.load(tmp_path / "object.mymyr")


# ------------------------------------------------------------------------------------------------ workers
def digest(obj):
    """A digest of a state space's or a knowledge base's contents, computed in a worker process."""
    h = hashlib.sha256()
    spaces = obj.state_spaces if isinstance(obj, datasets.KnowledgeBase) else [obj]
    for s in spaces:
        for k, v in sorted(_state(s)[5].items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        h.update(datasets.StateSpaceSampler(s, seed=11).sample_states(100).tobytes())
        h.update(str([search.iw(s.task, max_arity=1).plan]).encode())
    if isinstance(obj, datasets.KnowledgeBase) and obj.has_tuple_graphs:
        for i in range(len(spaces)):
            for g in obj.tuple_graphs(i):
                for k, v in sorted(g.arrays().items()):
                    h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


@needs_frontend
def test_spawned_workers_receive_state_spaces_and_knowledge_bases():
    g = PDDL / "gripper"
    space = datasets.state_space(pddl_task("counters", "p01"))
    kb = datasets.KnowledgeBase(TaskTable.from_pddl(g / "domain.pddl", [g / "p01.pddl", g / "p02.pddl"]), width=1)
    objs = [space, kb]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        assert list(pool.map(digest, objs)) == [digest(o) for o in objs]
    with context.Pool(1) as pool:
        assert pool.map(digest, [kb]) == [digest(kb)]
