"""Tuple graphs and knowledge bases (mymyr.datasets.TupleGraph, tuple_graphs, KnowledgeBase) on the vendored small
instances (tests/data/benchmarks/small)."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import datasets
from mymyr.rl import TaskTable

from conftest import ROOT

SMALL = ROOT / "tests/data/benchmarks/small"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def small_task(d, p):
    dom, prob = SMALL / d / "domain.pddl", SMALL / d / p
    return mymyr.Task.from_pddl(dom, prob, atoms="dense")


def space_of(task, **options):
    s = datasets.state_space(task, threads=1, remove_if_unsolvable=False, **options)
    assert s is not None
    return s


# ------------------------------------------------------------------------------------------------ the API
def same(g, h):
    """Equal tuple graphs of equal spaces (== compares graphs of one space object)."""
    a, b = g.arrays(), h.arrays()
    return (g.root, g.width, g.dominance_pruning) == (h.root, h.width, h.dominance_pruning) and a.keys() == b.keys() and all(
        np.array_equal(a[k], b[k]) for k in a
    )


@pytest.fixture(scope="module")
def gripper():
    return space_of(small_task("gripper", "p-2-0.pddl"))


def test_accessors_and_arrays(gripper):
    s = gripper
    g = datasets.tuple_graph(s, 0, width=1)
    assert (g.root, g.width, g.dominance_pruning, g.space.num_states) == (0, 1, True, s.num_states)
    assert len(g) == g.num_vertices and g.nbytes > 0
    assert g.vertices_at(0) == [0] and g.tuple(0) == [] and g.problem_vertices(0) == [0]
    assert g.problem_vertices_at(0) == [0]
    a = g.arrays()
    assert a["tuple_atoms"].dtype == np.uint32 and not a["tuple_atoms"].flags.writeable
    edges = 0
    for d in range(g.num_distances):
        assert g.vertices_at(d) == list(range(a["distance_offsets"][d], a["distance_offsets"][d + 1]))
        for v in g.vertices_at(d):
            assert g.distance(v) == d
            lo, hi = a["tuple_offsets"][v], a["tuple_offsets"][v + 1]
            assert g.tuple(v) == a["tuple_atoms"][lo:hi].tolist()
            assert [str(x) for x in g.atoms(v)] == [str(x) for x in (s.task.atom(i) for i in g.tuple(v))]
            lo, hi = a["problem_offsets"][v], a["problem_offsets"][v + 1]
            assert g.problem_vertices(v) == a["problem_vertices"][lo:hi].tolist()
            for x in g.problem_vertices(v):  # the tuple holds in its problem vertices
                assert all(s.state(x).holds(atom) for atom in g.atoms(v))
            assert all(g.distance(w) == d + 1 for w in g.successors(v))
            assert all(v in g.successors(u) for u in g.predecessors(v))
            edges += len(g.successors(v))
    assert edges == g.num_edges == a["successors"].size == a["predecessors"].size
    with pytest.raises(IndexError):
        g.tuple(g.num_vertices)
    with pytest.raises(IndexError):
        g.vertices_at(g.num_distances)
    with pytest.raises(IndexError):
        datasets.tuple_graph(s, s.num_states)
    with pytest.raises(ValueError, match="width"):
        datasets.tuple_graphs(s, width=6)


def test_width_zero_and_pruning(gripper):
    s = gripper
    for r in (0, 5):
        g = datasets.tuple_graph(s, r, width=0)
        successors = sorted({t for _, t in s.transitions(r)} - {r})
        assert g.num_distances == 2 and g.num_vertices == 1 + len(successors) and g.num_edges == len(successors)
        assert [g.problem_vertices(v)[0] for v in g.vertices_at(1)] == successors
    for r in range(0, s.num_states, 7):
        pruned, full = (datasets.tuple_graph(s, r, width=2, dominance_pruning=b) for b in (True, False))
        assert pruned.num_vertices <= full.num_vertices and len(pruned.vertices_at(0)) == 1
        assert [pruned.problem_vertices_at(d) for d in range(pruned.num_distances)] == [
            full.problem_vertices_at(d) for d in range(pruned.num_distances)
        ]


def test_independent_of_the_thread_count():
    for d, p in (("gripper", "p-2-0.pddl"), ("blocks_3", "test_problem.pddl")):
        s = datasets.state_space(small_task(d, p), threads=1)
        for width in (0, 1, 2):
            one = datasets.tuple_graphs(s, width=width, threads=1)
            for threads in (4, 8):
                assert datasets.tuple_graphs(s, width=width, threads=threads) == one
            assert datasets.tuple_graph(s, s.num_states - 1, width=width) == one[-1]


# ------------------------------------------------------------------------------------------------ knowledge bases
def gripper_table():
    return TaskTable.from_pddl(SMALL / "gripper" / "domain.pddl",
                               [SMALL / "gripper" / p for p in ("p-1-0.pddl", "p-2-0.pddl")], atoms="dense")


def test_knowledge_base_two_problems():
    """A knowledge base over gripper p-1-0 and p-2-0."""
    small_task("gripper", "p-2-0.pddl")
    kb = datasets.KnowledgeBase(gripper_table(), generalized=True)
    g = kb.generalized_state_space
    assert (g.num_vertices, g.num_edges) == (36, 128)
    assert len(g.goal_vertices()) == 4 and len(g.unsolvable_vertices()) == 0
    assert [s.num_states for s in g.spaces] == [s.num_states for s in kb.state_spaces] == [8, 28]
    assert not kb.has_tuple_graphs and kb.width is None
    with pytest.raises(ValueError, match="no tuple graphs"):
        kb.tuple_graphs(0)


@pytest.mark.parametrize("width,tg_vertices,tg_edges", [(1, 220, 184), (0, 128, 92)])
def test_knowledge_base_tuple_graphs(width, tg_vertices, tg_edges):
    """Knowledge base counts on gripper."""
    small_task("gripper", "p-2-0.pddl")
    kb = datasets.KnowledgeBase(gripper_table(), generalized=True, width=width)
    assert kb.width == width and len(kb) == 2
    graphs = [kb.tuple_graphs(i) for i in range(len(kb))]
    assert [len(x) for x in graphs] == [s.num_states for s in kb.state_spaces]
    assert sum(g.num_vertices for x in graphs for g in x) == tg_vertices
    assert sum(g.num_edges for x in graphs for g in x) == tg_edges
    assert kb.tuple_graph(1, 3) == graphs[1][3] == datasets.tuple_graph(kb.state_spaces[1], 3, width=width)


def test_knowledge_base_order_pickling_and_threads():
    small_task("gripper", "p-2-0.pddl")
    files = [SMALL / "gripper" / p for p in ("test_problem4.pddl", "p-2-0.pddl", "p-1-0.pddl")]
    table = TaskTable.from_pddl(SMALL / "gripper" / "domain.pddl", files, atoms="dense")
    kb = datasets.KnowledgeBase(table, width=1, threads=1)
    assert kb.task_indices == [2, 1, 0] and kb.generalized_state_space is None
    sizes = [s.num_states for s in kb.state_spaces]
    assert sizes == sorted(sizes)
    for s, i in zip(kb.state_spaces, kb.task_indices, strict=True):
        assert s.task.fingerprint == table[i].fingerprint
    unsorted = datasets.KnowledgeBase(list(table.tasks), sort_by_size=False, width=1, threads=4)
    assert unsorted.task_indices == [0, 1, 2]
    for i in range(3):
        assert all(same(g, h) for g, h in zip(unsorted.tuple_graphs(i), kb.tuple_graphs(2 - i), strict=True))
    back = pickle.loads(pickle.dumps(kb))
    assert back.task_indices == kb.task_indices and back.width == 1 and back.tasks[0].fingerprint == table[0].fingerprint
    for i in range(3):
        assert all(same(g, h) for g, h in zip(back.tuple_graphs(i), kb.tuple_graphs(i), strict=True))
    small = datasets.KnowledgeBase(table, max_states=30)  # failed generations are skipped
    assert small.task_indices == [2, 1]
    with pytest.raises(ValueError):
        datasets.KnowledgeBase(table, width=6)
