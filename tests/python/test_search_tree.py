"""Native IW trees agree with search events, replayable labels and task state rows."""

import gc
import pickle

import numpy as np
import pytest

import mymyr
from mymyr import search
from conftest import ROOT, text_task

ARRAYS = ("rows", "parent", "depth", "schema", "binding", "node_flags", "offsets", "t_schema",
          "t_binding", "t_row", "t_child", "t_outcome", "row_goal")


class Recorder:
    def __init__(self, start):
        self.start = start
        self.passes = []
        self.nodes = {0: (None, None, start)}
        self.transitions = []
        self.expanded = []

    def on_expand(self, node, state):
        self.expanded.append(node)
        assert self.nodes[node][2] == state

    def on_generate(self, parent, action, child, state, is_new):
        if is_new and child not in self.nodes:
            self.nodes[child] = (parent, action, state)

    def on_transition(self, parent, action, child, state, outcome):
        self.transitions.append((parent, action, child, state, outcome))

    def on_pass(self, width, stats):
        self.passes.append((self.nodes, self.transitions, self.expanded))
        self.nodes = {0: (None, None, self.start)}
        self.transitions = []
        self.expanded = []


def assert_tree_equal(a, b):
    assert (a.width, a.words, a.numeric_words, a.label_width) == (b.width, b.words, b.numeric_words, b.label_width)
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def assert_events(p, events):
    t = p.tree
    assert isinstance(t, search.SearchTree)
    nodes, transitions, expanded = events
    states, labels = t.states(), t.labels()
    assert t.num_nodes == len(nodes)
    assert t.num_transitions == len(transitions) == p.generated
    assert t.offsets.shape == (t.num_nodes + 1,)
    assert t.offsets[0] == 0 and t.offsets[-1] == t.num_transitions
    assert len(set(states)) == len(states)
    if t.task.atom_mode != "lazy":
        assert t.task.decode(t.rows) == states
    assert labels[0] is None and t.parent[0] == t.schema[0] == -1
    for i, (parent, action, state) in nodes.items():
        assert states[i] == state
        if i:
            assert t.parent[i] == parent and labels[i] == action
            assert t.task.apply(states[parent], labels[i]) == states[i]
            assert t.depth[i] == t.depth[parent] + 1
    grouped = [e for parent in range(t.num_nodes) for e in transitions if e[0] == parent]
    for i, (parent, action, child, state, outcome) in enumerate(grouped):
        assert t.offsets[parent] <= i < t.offsets[parent + 1]
        assert states[t.t_row[i]] == state
        assert t.t_child[i] == (-1 if child is None else child)
        assert t.t_schema[i] == action.schema
        binding = np.asarray(action.binding, dtype=np.int32)
        np.testing.assert_array_equal(t.t_binding[i, :len(binding)], binding)
        assert np.all(t.t_binding[i, len(binding):] == -1)
        assert t.t_outcome[i] == outcome.value
    assert np.count_nonzero(t.node_flags & 1) + np.count_nonzero(t.node_flags & 4) <= len(expanded)
    np.testing.assert_array_equal(t.row_goal, [t.task.is_goal(s) for s in states])
    np.testing.assert_array_equal((t.node_flags & 2) != 0, t.row_goal[:t.num_nodes] != 0)


@pytest.fixture(params=["blocks", "rovers", "axioms", "childsnack"])
def sample(request):
    if request.param == "childsnack":
        if not hasattr(mymyr, "Domain"):
            pytest.skip("built without the loki front end")
        base = ROOT / "tests/data/ipc/childsnack-ipc/test"
        return mymyr.Task.from_pddl(base / "domain.pddl", base / "p01-easy.pddl")
    return text_task({"blocks": "blocks__probBLOCKS-8-0", "rovers": "rovers__p02",
                      "axioms": "philosophers__p03-phil4"}[request.param])


@pytest.mark.parametrize("kind", ["iw1", "iw2", "pass", "liw", "abstracted", "projective"])
def test_iw_tree_matches_observer(sample, kind):
    obs = Recorder(sample.initial_state)
    kw = dict(tree="transitions", observer=obs, max_states=120, max_expanded=100)
    if kind.startswith("iw"):
        result = search.iw(sample, max_width=int(kind[-1]), **kw)
    elif kind == "pass":
        result = search.iw_pass(sample, 2, **kw)
    elif kind == "liw":
        result = search.liw(sample, max_width=1, landmarks=search.approximate_fact_landmarks(sample), **kw)
    elif kind == "abstracted":
        result = search.abstracted_iw(sample, width=2, **kw)
    else:
        result = search.projective_iw(sample, **kw)
    for p, events in zip(result.passes, obs.passes, strict=True):
        if p.placeholder:
            assert p.tree is None
        else:
            assert_events(p, events)
            assert p.tree.num_nodes == p.generated_in_tree + 1


@pytest.mark.parametrize("level", ["nodes", "transitions"])
def test_iw_tree_reconstructs_plan_and_owns_arrays(level):
    task = text_task("gripper__prob05")
    r = search.iw(task, max_width=2, tree=level, goal=[["(at-robby o1)"]])
    assert r.solved
    t = r.passes[-1].tree
    states, labels = t.states(), t.labels()
    goals = [i for i in range(t.num_nodes) if states[i] == r.goal_state]
    assert goals
    at = goals[0]
    plan = []
    while at:
        plan.append(labels[at])
        at = int(t.parent[at])
    assert list(reversed(plan)) == r.plan
    if level == "nodes":
        assert t.num_transitions == 0 and t.offsets.size == 0
        assert len(states) == t.num_nodes
    rows = t.rows
    same = t.rows
    assert np.shares_memory(rows, same) and not rows.flags.writeable
    with pytest.raises(ValueError):
        rows[0, 0] = 0
    expected = rows.copy()
    del t, r, task
    gc.collect()
    np.testing.assert_array_equal(rows, expected)


@pytest.mark.parametrize("level", ["nodes", "transitions"])
def test_iw_tree_pickle_with_lazy_slots_and_numeric_states(level):
    for name, options in [("blocks__probBLOCKS-8-0", dict(atoms="lazy")), ("cs-counters", {})]:
        if name == "cs-counters":
            task = mymyr.Task.from_text(ROOT / "tests/data/numeric_tasks/cs-counters.txt")
        else:
            task = text_task(name, **options)
        r = search.iw(task, max_width=2, tree=level, max_states=150)
        for original in [r, r.passes[-1], r.passes[-1].tree]:
            restored = pickle.loads(pickle.dumps(original))
            a = original.passes[-1].tree if hasattr(original, "passes") else original.tree if hasattr(original, "tree") else original
            b = restored.passes[-1].tree if hasattr(restored, "passes") else restored.tree if hasattr(restored, "tree") else restored
            assert_tree_equal(a, b)
            assert [set(map(str, s.atoms())) for s in a.states()] == [set(map(str, s.atoms())) for s in b.states()]
        if name == "cs-counters":
            assert r.passes[-1].tree.numeric_words == task.numeric_words > 0
            assert task.decode(r.passes[-1].tree.rows) == r.passes[-1].tree.states()


@pytest.mark.parametrize("level", ["nodes", "transitions"])
def test_iw_rollout_trees_are_identical_across_threads(level):
    task = text_task("rovers__p02")
    seeds = list(range(8))
    batches = [search.find_rollouts_parallel(task, seeds, threads=n, max_width=2, max_states=100,
                                            tree=level) for n in (1, 2, 8)]
    assert [r.threads_used for r in batches] == [1, 2, 8]
    for others in batches[1:]:
        for a, b in zip(batches[0], others, strict=True):
            for p, q in zip(a.passes, b.passes, strict=True):
                assert_tree_equal(p.tree, q.tree)
    restored = pickle.loads(pickle.dumps(batches[-1]))
    assert_tree_equal(restored[0].passes[-1].tree, batches[-1][0].passes[-1].tree)


@pytest.mark.parametrize("options", [dict(max_states=1), dict(max_states=10), dict(max_expanded=2), dict(max_depth=0),
                                     dict(width_zero="root_only")])
def test_iw_tree_keeps_partial_search_and_skipped_nodes(options):
    task = text_task("gripper__prob05")
    obs = Recorder(task.initial_state)
    r = search.iw_pass(task, 0, tree="transitions", observer=obs, **options)
    assert_events(r.passes[0], obs.passes[0])
    if "max_depth" in options or "width_zero" in options:
        assert np.count_nonzero(r.passes[0].tree.node_flags & 4)


def test_iw_tree_cancel_keeps_recorded_nodes():
    task = text_task("gripper__prob05")
    token = search.CancelToken()

    class Stop(Recorder):
        def on_expand(self, node, state):
            super().on_expand(node, state)
            if node == 1:
                token.request()

    obs = Stop(task.initial_state)
    r = search.iw(task, max_width=2, tree="transitions", observer=obs, cancel=token)
    assert r.status == search.Status.CANCELLED
    assert_events(r.passes[-1], obs.passes[-1])
    assert r.passes[-1].tree.num_nodes > 1


@pytest.mark.parametrize("fn", [search.iw, search.liw, search.abstracted_iw])
def test_iw_tree_beam_flags_and_serial_export(fn):
    task = text_task("rovers__p02")
    kw = dict(tree="transitions", layer_order="randomized", seed=42, beam_width=2, max_expanded=30)
    r = fn(task, threads=1, **kw)
    q = fn(task, threads=4, **kw)
    for p, pp in zip(r.passes, q.passes, strict=True):
        if not p.placeholder:
            t = p.tree
            assert_tree_equal(t, pp.tree)
            assert np.any(t.node_flags & 8)
            assert not np.any(((t.node_flags & 8) != 0) & ((t.node_flags & 1) != 0))


def test_iw_tree_custom_goal_calls_each_recorded_row_once():
    task = text_task("rovers__p02")
    calls = []

    def goal(state):
        calls.append(state)
        return False

    r = search.iw_pass(task, 1, goal=goal, tree="transitions", max_states=20)
    t = r.passes[0].tree
    assert len(calls) == len(t.row_goal) and len(set(calls)) == len(calls)
    assert not np.any(t.row_goal)
    assert len(t.row_goal) > t.num_nodes


def test_iw_tree_blocking_and_any_of_goal():
    task = text_task("gripper__prob05")
    blocked = [s for s in task.successor_states(task.initial_state) if s != task.initial_state][:1]
    obs = Recorder(task.initial_state)
    r = search.iw(task, max_width=1, tree="transitions", blocked_states=blocked, observer=obs)
    t = r.passes[-1].tree
    assert_events(r.passes[-1], obs.passes[-1])
    assert np.any(t.t_outcome == search.TransitionOutcome.BLOCKED.value)
    assert not any(s in blocked for s in t.states()[:t.num_nodes])
    any_goal = [[str(next(iter(task.initial_state.atoms())))]]
    r = search.iw_pass(task, 1, goal=any_goal, tree="transitions")
    assert r.solved and r.passes[0].tree.row_goal[0]


def test_siw_tree_pickle_and_subproblem_goal_flags():
    task = text_task("gripper__prob05")
    r = search.siw(task, max_width=2, tree="nodes")
    assert r.solved
    restored = pickle.loads(pickle.dumps(r))
    for p, q in zip(r.subproblems, restored.subproblems, strict=True):
        t = p.passes[-1].tree
        assert np.any(t.row_goal)
        assert_tree_equal(t, q.passes[-1].tree)
        pp = pickle.loads(pickle.dumps(p))
        assert_tree_equal(t, pp.passes[-1].tree)


@pytest.mark.parametrize("value,error", [(True, TypeError), (False, TypeError), (1, TypeError), ("all", ValueError),
                                         ("Nodes", ValueError)])
def test_iw_tree_record_level_is_strict(value, error):
    task = text_task("rovers__p02")
    for fn in (search.iw, search.siw, search.liw, search.abstracted_iw, search.projective_iw):
        with pytest.raises(error, match="tree"):
            fn(task, tree=value)
    with pytest.raises(error, match="tree"):
        search.iw_pass(task, 1, tree=value)
    with pytest.raises(error, match="tree"):
        search.find_rollouts_parallel(task, [0], tree=value)
    assert all(p.tree is None for p in search.iw(task, max_width=1).passes)


def test_iw_tree_pickle_rejects_corrupt_arrays_and_fingerprint():
    task = text_task("rovers__p02")
    t = search.iw_pass(task, 1, tree="transitions", max_states=20).passes[0].tree
    restore, args = t.__reduce__()
    for field, bad in [(5, np.full(t.num_nodes, 100, dtype=np.int32)),
                       (10, np.array([0, 100], dtype=np.int64)),
                       (13, np.full(t.num_transitions, -1, dtype=np.int32)),
                       (14, np.full(t.num_transitions, t.num_nodes + 1, dtype=np.int32)),
                       (15, np.full(t.num_transitions, 255, dtype=np.uint8)),
                       (16, np.full(len(t.row_goal), 2, dtype=np.uint8)),
                       (4, t.rows.astype(np.int32))]:
        fields = list(args[-1])
        fields[field] = bad
        with pytest.raises(ValueError, match="corrupt search tree"):
            restore(*args[:-1], tuple(fields))
    with pytest.raises(ValueError, match="fingerprint"):
        restore(args[0], args[1], args[2] ^ 1, args[3])


def test_iw_tree_with_layer_transition_ordering():
    task = text_task("rovers__p02")
    ordering = search.LandmarkTransitionOrdering(search.approximate_fact_landmarks(task))
    obs = Recorder(task.initial_state)
    r = search.iw_pass(task, 1, tree="transitions", transition_ordering=ordering, observer=obs, max_expanded=20)
    t = r.passes[0].tree
    assert t.num_transitions == r.passes[0].generated
    states = t.states()
    expected = sorted((p, a.schema, a.binding, -1 if c is None else c, str(s), o.value)
                      for p, a, c, s, o in obs.passes[0][1])
    actual = []
    for p in range(t.num_nodes):
        for i in range(t.offsets[p], t.offsets[p + 1]):
            binding = tuple(int(v) for v in t.t_binding[i] if v >= 0)
            actual.append((p, int(t.t_schema[i]), binding, int(t.t_child[i]), str(states[t.t_row[i]]), int(t.t_outcome[i])))
    assert sorted(actual) == expected
    q = search.iw_pass(task, 1, tree="nodes", transition_ordering=ordering, max_expanded=20)
    for name in ("parent", "depth", "schema", "binding", "node_flags"):
        np.testing.assert_array_equal(getattr(t, name), getattr(q.passes[0].tree, name))


def test_iw_rollout_tree_matches_worker_observer():
    task = text_task("rovers__p02")

    class Worker(Recorder):
        def on_expand(self, node, state):
            if node == 0 and self.expanded:
                self.on_pass(0, None)
            super().on_expand(node, state)

    class Workers:
        def __init__(self):
            self.workers = {}

        def make_worker(self, k):
            self.workers[k] = Worker(task.initial_state)
            return self.workers[k]

    obs = Workers()
    batch = search.find_rollouts_parallel(task, list(range(8)), max_width=2, threads=8,
                                           tree="transitions", observer=obs, max_states=60)
    assert batch.threads_used == 8
    for k, r in enumerate(batch):
        worker = obs.workers[k]
        worker.on_pass(0, None)
        for p, events in zip(r.passes, worker.passes, strict=True):
            assert_events(p, events)


@pytest.mark.parametrize("fn", [search.iw, search.abstracted_iw])
def test_iw_tree_relaxed_node_recording_uses_serial_beam(fn):
    task = text_task("rovers__p02")
    kw = dict(tree="nodes", layer_order="goal_count", beam_width=3,
              beam_novelty="relaxed_survivors_only", max_expanded=30)
    a, b = fn(task, threads=1, **kw), fn(task, threads=4, **kw)
    assert a.passes
    for p, q in zip(a.passes, b.passes, strict=True):
        assert_tree_equal(p.tree, q.tree)


@pytest.mark.parametrize("fn", [search.iw, search.abstracted_iw, search.liw])
def test_relaxed_beam_refuses_a_transition_tree_by_name(fn):
    task = text_task("rovers__p02")
    r = fn(task, tree="transitions", layer_order="goal_count", beam_width=3, beam_novelty="relaxed_survivors_only",
           max_expanded=30)
    assert r.status == search.Status.FAILED and "a transition tree (tree='transitions')" in r.message


def test_iw_tree_extra_goal_callback_errors_propagate():
    task = text_task("rovers__p02")
    calls = []

    def goal(state):
        calls.append(state)
        if len(calls) > 1:
            raise LookupError("recorded goal")
        return False

    with pytest.raises(LookupError, match="recorded goal"):
        search.iw_pass(task, 1, tree="transitions", max_states=2, goal=goal)
    assert len(calls) == 2


@pytest.mark.parametrize("level", ["nodes", "transitions"])
def test_iw_tree_empty_state_rows(tmp_path, level):
    path = tmp_path / "empty.txt"
    path.write_text("O 0\nP 0\nSI 0\nFI 0\nG 0\nA 0\n")
    task = mymyr.Task.from_text(path)
    result = search.iw_pass(task, 1, tree=level)
    assert result.solved
    t = result.passes[0].tree
    assert t.words == task.words == 0
    assert t.rows.shape == (1, 0) and t.binding.shape == (1, 0)
    assert t.states() == [task.initial_state]
    assert t.labels() == [None]
    np.testing.assert_array_equal(t.row_goal, [1])
    assert_tree_equal(t, pickle.loads(pickle.dumps(t)))
