"""mymyr.datasets: state spaces, the instance pool, generalized state spaces, samplers,
object graphs and certificates.

Counts against the fork use the fork's own test instances (its data/ directory: env MYMYR_FORK_DATA; skipped when
missing) with the fork's own state-space numbers. The torch and JAX tests skip
without those frameworks (run them in a venv with torch 2.14 / jax 0.11.2).
"""

import os
import pathlib
import threading

import numpy as np
import pytest

import mymyr
from mymyr import datasets, search

from conftest import TASKS, text_task

FORK_DATA = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def fork_task(d, p, **options):
    dom, prob = FORK_DATA / d / "domain.pddl", FORK_DATA / d / p
    if not dom.is_file() or not prob.is_file():
        pytest.skip(f"fork data missing: {prob}")
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    options.setdefault("atoms", "frozen")
    return mymyr.Task.from_pddl(dom, prob, **options)


@pytest.fixture(scope="module")
def depot():
    return text_task("depot__p02", atoms="frozen")  # 40320 states


@pytest.fixture(scope="module")
def depot_space(depot):
    s = datasets.state_space(depot, threads=2, remove_if_unsolvable=False)
    assert s is not None
    return s


# ------------------------------------------------------------------------------------------------ fork parity
# (dir, problem, states, transitions, goal, unsolvable, max goal distance, sum of finite cost goal distances)
FORK = [
    ("spanner", "p15-easy.pddl", 662, 1722, 40, 153, 14, None),
    ("barman", "test_problem.pddl", 406, 1298, 28, 70, 17, 5136.0),  # action costs
    ("miconic-fulladl", "test_problem.pddl", 87, 229, 6, 0, 8, None),  # derived predicates, conditional effects
    ("gripper", "p-2-0.pddl", 28, 104, 2, 0, 6, None),
]


@pytest.mark.parametrize("case", FORK, ids=lambda c: f"{c[0]}-{c[1]}")
def test_counts_match_the_fork(case):
    d, p, n, e, goal, unsolvable, max_d, cost_sum = case
    task = fork_task(d, p)
    for threads in (1, 3):
        r = datasets.generate(task, threads=threads, remove_if_unsolvable=False)
        assert r.status == datasets.Status.OK and r and r.states == n
        s = r.space
        assert (s.num_states, s.num_transitions, s.num_goal_states, s.num_unsolvable_states) == (n, e, goal, unsolvable)
        assert s.max_goal_distance == max_d
        a = s.arrays()
        if cost_sum is not None:
            c = a["cost_goal_distances"]
            assert c[np.isfinite(c)].sum() == cost_sum
            assert not s.unit_costs and a["costs"].dtype == np.float64 and a["costs"].shape == (e,)


# ------------------------------------------------------------------------------------------------ arrays
def test_arrays_are_consistent(depot, depot_space):
    s = depot_space
    a = s.arrays()
    n, e = s.num_states, s.num_transitions
    assert len(s) == n == 40320 and s.initial_state_id == 0 and not s.symmetry_reduced
    assert a["state_words"].shape == (n, s.row_words) and a["state_words"].dtype == np.uint64
    off, tgt = a["forward_offsets"], a["forward_targets"]
    assert off.shape == (n + 1,) and off[-1] == e and tgt.shape == (e,) and tgt.max() < n
    assert a["label_schemas"].shape == (e,) and a["label_bindings"].shape == (e, s.label_width)
    # the reverse CSR holds every forward edge once, per target, ascending
    boff, bsrc, bedge = a["backward_offsets"], a["backward_sources"], a["backward_edges"]
    assert boff[-1] == e and np.array_equal(np.sort(bedge), np.arange(e))
    src_of = np.repeat(np.arange(n), np.diff(off).astype(np.int64))
    assert np.array_equal(src_of[bedge], bsrc)
    assert np.array_equal(tgt[bedge], np.repeat(np.arange(n), np.diff(boff).astype(np.int64)))
    # V*: unit distances by an independent backward BFS in NumPy
    d = np.full(n, -1, np.int32)
    frontier = np.flatnonzero(a["goal"])
    d[frontier] = 0
    k = 0
    while frontier.size:
        k += 1
        pred = np.concatenate([bsrc[boff[v]:boff[v + 1]] for v in frontier])
        pred = np.unique(pred[d[pred] == -1])
        d[pred] = k
        frontier = pred
    assert np.array_equal(d, a["unit_goal_distances"])
    assert np.array_equal(a["cost_goal_distances"], np.where(d >= 0, d, np.inf))
    assert np.array_equal(a["unsolvable"], d < 0)
    assert np.array_equal(a["alive"], ~a["goal"] & ~a["unsolvable"])
    assert s.num_goal_states == a["goal"].sum() and s.goal_states() == list(np.flatnonzero(a["goal"]))
    # zero-copy, read-only views of one immutable result
    b = s.arrays()
    assert np.shares_memory(a["forward_targets"], b["forward_targets"])
    assert not a["forward_targets"].flags.writeable
    with pytest.raises(ValueError):
        a["goal"][0] = True
    # the same count as the BrFS without witness pruning
    assert search.brfs(depot, witness_pruning=False).states == n


def test_states_labels_and_transitions(depot, depot_space):
    s = depot_space
    for i in (0, 1, 17, 4000, s.num_states - 1):
        st = s.state(i)
        assert s.find(st) == i
        trans = s.transitions(i)
        # the task's canonical successors, in order
        succ = depot.successors(st)
        assert [str(a) for a, _ in trans] == [str(a) for a, _ in succ]
        assert [s.state(t) for _, t in trans] == [x for _, x in succ]
        off = s.arrays()["forward_offsets"]
        for k, (a, t) in enumerate(trans):
            edge = int(off[i]) + k
            assert str(s.label(edge)) == str(a) and s.source(edge) == i and s.target(edge) == t and s.cost(edge) == 1.0
    with pytest.raises(IndexError):
        s.state(s.num_states)
    with pytest.raises(IndexError):
        s.label(s.num_transitions)
    with pytest.raises(ValueError):
        s.find(text_task("gripper__prob05").initial_state)  # another task


def test_ids_and_arrays_independent_of_threads(depot, depot_space):
    ref = depot_space.arrays()
    for threads in (1, 5):
        a = datasets.state_space(depot, threads=threads, remove_if_unsolvable=False).arrays()
        assert a.keys() == ref.keys()
        for k in ref:
            assert np.array_equal(a[k], ref[k]), (threads, k)


# ------------------------------------------------------------------------------------------------ options
def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


DEADEND = """
(define (domain dead) (:requirements :strips)
 (:predicates (a) (b) (g))
 (:action ab :parameters () :precondition (a) :effect (and (not (a)) (b)))
 (:action bg :parameters () :precondition (and (b) (a)) :effect (g)))
"""


@needs_frontend
def test_options(tmp_path, depot):
    dom = write(tmp_path, "d.pddl", DEADEND)
    task = mymyr.Task.from_pddl(dom, write(tmp_path, "p.pddl", "(define (problem p) (:domain dead) (:init (a)) (:goal (g)))"))
    # the initial state cannot reach the goal: no space unless remove_if_unsolvable=False (the fork's default is True)
    r = datasets.generate(task)
    assert r.status == datasets.Status.UNSOLVABLE and r.space is None and not r
    assert datasets.state_space(task) is None
    s = datasets.state_space(task, remove_if_unsolvable=False)
    assert s.num_states == 2 and s.num_unsolvable_states == 2 and s.unit_goal_distance(0) == -1
    assert s.cost_goal_distance(0) == float("inf")
    # max_states: fails iff the space has max(max_states, 2) states or more
    assert datasets.generate(depot, max_states=40320).status == datasets.Status.OUT_OF_STATES
    assert datasets.generate(depot, max_states=40321, threads=2).status == datasets.Status.OK
    assert datasets.generate(depot, max_seconds=0.0).status == datasets.Status.TIMEOUT
    # without labels
    s = datasets.state_space(depot, labels=False)
    assert not s.has_labels and "label_schemas" not in s.arrays()
    with pytest.raises(ValueError):
        s.label(0)
    with pytest.raises(ValueError):
        datasets.generate(depot, symmetry_pruning=True, certificate="kfwl", k=5)
    with pytest.raises(ValueError):
        datasets.generate(depot, certificate="nauty")
    with pytest.raises(TypeError):
        datasets.generate("depot")
    assert str(datasets.Status.OUT_OF_STATES) == "out_of_states"


# ------------------------------------------------------------------------------------------------ pool
def test_pool_equals_single_generations():
    tasks = [text_task(n, atoms="frozen") for n in ("depot__p02", "philosophers__p03-phil4")] * 3
    rs = datasets.generate_many(tasks, threads=4, remove_if_unsolvable=False)
    assert [r.status for r in rs] == [datasets.Status.OK] * 6
    for t, r in zip(tasks, rs):
        assert r.space.task is t
        one = datasets.state_space(t, remove_if_unsolvable=False)
        assert np.array_equal(r.space.arrays()["forward_targets"], one.arrays()["forward_targets"])
    with pytest.raises(TypeError):
        datasets.generate_many([tasks[0], "x"])


def test_many_python_threads():
    task = text_task("philosophers__p03-phil4", atoms="frozen")
    out = [None] * 8

    def work(i):
        out[i] = datasets.state_space(task, threads=1 + i % 3, remove_if_unsolvable=False).arrays()["forward_targets"]

    th = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert all(np.array_equal(x, out[0]) for x in out)


# ------------------------------------------------------------------------------------------------ generalized
def test_generalized_state_space():
    tasks = [fork_task("gripper", p) for p in ("test_problem4.pddl", "p-2-0.pddl", "test_problem2.pddl", "p-1-0.pddl")]
    g = datasets.generalized_state_space(tasks, remove_if_unsolvable=False)
    sizes = [s.num_states for s in g.spaces]
    assert sizes == sorted(sizes) == [8, 28, 28, 256]
    assert g.spaces[1].task is tasks[1] and g.spaces[2].task is tasks[2]  # ties keep the input order
    assert not g.symmetry_reduced and g.num_vertices == sum(sizes)
    assert g.num_edges == sum(s.num_transitions for s in g.spaces)
    a = g.arrays()
    assert len(g.initial_vertices()) == 4 and a["initial"].sum() == 4
    for p, s in enumerate(g.spaces):
        vm = g.vertex_mapping(p)
        assert np.array_equal(a["vertex_problems"][vm], np.full(s.num_states, p))
        assert np.array_equal(a["vertex_problem_vertices"][vm], np.arange(s.num_states))
        assert np.array_equal(a["goal"][vm], s.arrays()["goal"])
    sym = datasets.generalized_state_space(tasks, remove_if_unsolvable=False, symmetry_pruning=True)
    assert sym.symmetry_reduced and len(sym.spaces) < 4 and sym.num_vertices < g.num_vertices
    with pytest.raises(IndexError):
        g.vertex_mapping(4)
    with pytest.raises(ValueError):
        datasets.GeneralizedStateSpace([g.spaces[0], datasets.state_space(fork_task("blocks_3", "test_problem.pddl"), remove_if_unsolvable=False)])


# ------------------------------------------------------------------------------------------------ samplers
def test_sampler():
    s = datasets.state_space(fork_task("spanner", "p15-easy.pddl"), remove_if_unsolvable=False)
    a, b = datasets.StateSpaceSampler(s, seed=7), datasets.StateSpaceSampler(s, seed=7)
    x = a.sample_states(1000)
    assert x.dtype == np.uint32 and np.array_equal(x, b.sample_states(1000))
    assert not np.array_equal(x, datasets.StateSpaceSampler(s, seed=8).sample_states(1000))
    a.set_seed(7)
    assert np.array_equal(a.sample_states(1000), x)
    assert (a.num_states, a.num_dead_end_states, a.num_alive_states) == (662, 153, 509)
    assert a.max_steps_to_goal == s.max_goal_distance == 14
    d = s.arrays()["unit_goal_distances"]
    for n in range(a.max_steps_to_goal + 1):
        ids = a.sample_states_n_steps_from_goal(n, 200)
        assert (d[ids] == n).all()
        support = a.states_n_steps_from_goal(n)
        assert np.array_equal(support, np.flatnonzero(d == n))
        assert d[a.sample_state_n_steps_from_goal(n)] == n
    assert s.arrays()["unsolvable"][a.sample_dead_end_states(500)].all()
    assert np.array_equal(a.dead_end_states(), np.flatnonzero(d < 0))
    with pytest.raises(ValueError):
        a.sample_state_n_steps_from_goal(15)
    # uniform: every state shows up
    assert np.unique(a.sample_states(40000)).size == s.num_states
    no_dead = datasets.StateSpaceSampler(datasets.state_space(fork_task("gripper", "p-2-0.pddl")))
    with pytest.raises(ValueError):
        no_dead.sample_dead_end_state()


# ------------------------------------------------------------------------------------------------ object graphs
def test_object_graphs_and_certificates():
    task = fork_task("gripper", "p-2-0.pddl")
    s = datasets.state_space(task)
    builder = datasets.ObjectGraphBuilder(task)
    cr, fwl = set(), set()
    for i in range(s.num_states):
        g = builder.build(s.state(i))
        a = g.arrays()
        assert g.num_objects == task.num_objects
        assert a["color"].shape == (g.num_vertices,) and a["offsets"][-1] == 2 * g.num_edges
        assert a["color"].max() < g.num_colors and g.palette(int(a["color"][0]))[0] == 0  # object colours first
        cr.add(g.color_refinement_certificate())
        fwl.add(g.kfwl_certificate(2))
        assert g.color_refinement_certificate() == datasets.object_graph(s.state(i)).color_refinement_certificate()
    assert len(cr) == len(fwl) == 12  # the fork's test: 12 isomorphism classes among the 28 states
    assert all(0 <= c < 2**128 for c in cr)
    assert len(builder.build(s.state(0)).stable_colors()) == builder.build(s.state(0)).num_vertices
    with pytest.raises(ValueError):
        builder.build(text_task("depot__p02").initial_state)
    with pytest.raises(ValueError):
        builder.build(s.state(0)).kfwl_certificate(5)
    sym = datasets.state_space(task, symmetry_pruning=True)
    assert sym.symmetry_reduced and sym.num_states == 12


# ------------------------------------------------------------------------------------------------ frameworks
def test_torch_arrays(depot_space):
    torch = pytest.importorskip("torch")
    s = datasets.state_space(fork_task("barman", "test_problem.pddl"), remove_if_unsolvable=False)
    for space in (depot_space, s):
        a, t = space.arrays(), space.arrays(framework="torch")
        assert t.keys() == a.keys()
        for k in a:
            assert isinstance(t[k], torch.Tensor)
            got = t[k].numpy() if k != "state_words" else t[k].numpy().view(np.uint64)
            assert np.array_equal(got, a[k]), k
    assert s.arrays(framework="torch")["costs"].dtype == torch.float64
    sm = datasets.StateSpaceSampler(s, seed=1)
    assert isinstance(sm.sample_states(10, framework="torch"), torch.Tensor)


def test_jax_arrays(depot_space):
    jax = pytest.importorskip("jax")
    a, j = depot_space.arrays(), depot_space.arrays(framework="jax")
    assert j.keys() == a.keys()
    for k in a:
        got = np.asarray(j[k])
        if a[k].dtype == np.uint64:  # JAX without x64: 64-bit arrays as uint32 pairs, low half first (arrays.hpp)
            assert got.dtype == np.uint32 and np.array_equal(got.view(np.uint64).reshape(a[k].shape), a[k]), k
        elif a[k].dtype == np.float64:  # JAX without x64 converts float64 to float32 (exact here: integers, inf)
            assert np.array_equal(got.astype(np.float64), a[k]), k
        else:
            assert np.array_equal(got, a[k]), k
    assert isinstance(j["forward_targets"], jax.Array)


def test_dlpack_arrays(depot_space):
    a = depot_space.arrays(framework="dlpack")
    assert all(hasattr(v, "__dlpack__") for v in a.values())
    assert np.array_equal(np.from_dlpack(a["forward_targets"]), depot_space.arrays()["forward_targets"])
    assert np.array_equal(np.from_dlpack(a["cost_goal_distances"]), depot_space.arrays()["cost_goal_distances"])


# ------------------------------------------------------------------------------------------------ stubs
def test_stubs_are_typed():
    import importlib.util

    d = pathlib.Path(importlib.util.find_spec("mymyr._core").origin).parent / "_core"
    if not (d / "_datasets.pyi").is_file():
        pytest.skip("no stubs installed")
    text = (d / "_datasets.pyi").read_text()
    assert "def generate(task: mymyr._core.Task | mymyr._core.TaskHandle, *, threads: int = 1" in text
    assert "max_states: int | None = None" in text and "max_seconds: float | None = None" in text
    assert "def space(self) -> StateSpace | None" in text
    assert "def transitions(self, id: int) -> list[tuple[mymyr._core.Action, int]]" in text
    assert "def arrays(self, framework: Literal['numpy', 'torch', 'jax', 'dlpack'] | None = None) -> dict[str, Any]" in text
    # no bare object parameters, no untyped list / dict / object results
    import ast

    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg not in ("self", "other") and arg.annotation is not None:
                    assert ast.unparse(arg.annotation) != "object", (node.name, arg.arg)
            if node.returns is not None and node.name != "__reduce__":
                assert ast.unparse(node.returns) not in ("object", "list", "dict", "tuple"), node.name
