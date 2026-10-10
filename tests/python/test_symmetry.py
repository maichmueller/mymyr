"""Symmetry pruning (symmetry_pruning='wl1') from Python.

The fork's WL1-pruned applicable actions of every reachable state of the state-space suite's tasks
(tests/data/symmetry/fork_symmetry.json, recorded by tests/data/fork_golden/search_fork/run_symmetry.py) against
Task.applicable_actions(symmetry_pruning='wl1'), on the states whose colour classes equal the fork's (the fork never
separates an object without neighbours in the object graph from its class; successor/symmetry.hpp); then brfs and
A* with pruning against the fork's searches on the tasks where all classes agree, and valid plans throughout. The
fork's PDDL comes from MYMYR_FORK_DATA (skipped when missing).
"""

import collections
import json
import os
import pathlib

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from conftest import BLOCKS, ROOT

FORK = ROOT / "tests/data/symmetry/fork_symmetry.json"
FORK_DATA = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def fork_records():
    if not FORK.exists():
        return []
    return [r for r in json.loads(FORK.read_text())["tasks"] if "killed" not in r]


def partition(classes):
    """The partition of the objects that a class list induces (labels renamed in order of first occurrence)."""
    rename = {}
    return [rename.setdefault(c, len(rename)) for c in classes]


def state_key(task, state):
    items = [str(a) for a in state.atoms()]
    if task.numeric_slots:
        items += [f"{n}={v:.17g}" for n, v in zip(task.numeric_names, state.numeric_values()) if v == v]
    return " ".join(sorted(items))


def reaches_goal(task, plan):
    s = task.initial_state
    for a in plan:
        if not task.is_applicable(s, a):
            return False
        s = s.apply(a)
    return task.is_goal(s)


def load(rec):
    d, p = rec["task"].split("/", 1)
    dom, prob = FORK_DATA / d / "domain.pddl", FORK_DATA / d / p
    if not dom.exists() or not prob.exists():
        return None
    return mymyr.Task.from_pddl(dom, prob)


@pytest.mark.parametrize("rec", fork_records(), ids=lambda r: r["task"])
def test_matches_the_fork(rec):
    task = load(rec)
    if task is None:
        pytest.skip(f"{rec['task']} is missing (set MYMYR_FORK_DATA)")
    assert list(task.objects) == rec["objects"]
    atoms, actions = rec["atom_names"], rec["action_names"]
    fork = {}
    for st in rec["states"]:
        fork[" ".join(sorted(atoms[i] for i in st["atoms"]))] = st

    seen = {state_key(task, task.initial_state)}
    queue = collections.deque([task.initial_state])
    states = compared = refined = 0
    while queue:
        s = queue.popleft()
        st = fork[state_key(task, s)]
        states += 1
        kept = task.applicable_actions(s, symmetry_pruning="wl1")
        # the same actions through every enumeration API, all of them applicable
        assert [str(a) for a in s.applicable_actions(symmetry_pruning="wl1")] == [str(a) for a in kept]
        assert [str(a) for a in task.iter_applicable_actions(s, symmetry_pruning="wl1")] == [str(a) for a in kept]
        pairs = task.successors(s, symmetry_pruning="wl1")
        assert [str(a) for a, _ in pairs] == [str(a) for a in kept]
        assert [t for _, t in pairs] == task.successor_states(s, symmetry_pruning="wl1")
        everything = {str(a) for a in task.applicable_actions(s)}
        assert {str(a) for a in kept} <= everything
        n = task.num_objects
        mine = partition(mymyr.datasets.object_graph(s).stable_colors()[:n])
        if mine == partition(st["classes"]):
            compared += 1
            assert {str(a) for a in kept} == {actions[i] for i in st["actions"]}, rec["task"]
        else:
            refined += 1
        for t in task.successor_states(s):
            k = state_key(task, t)
            if k not in seen:
                seen.add(k)
                queue.append(t)
    assert states == rec["num_states"]

    ex = search.brfs(task, witness_pruning=False, symmetry_pruning="wl1", stop_at_goal=False)
    br = search.brfs(task, witness_pruning=False, symmetry_pruning="wl1")
    ar = search.astar(task, heuristic="blind", symmetry_pruning="wl1")
    if br.solved:
        assert reaches_goal(task, br.plan)
    if ar.status == Status.SOLVED:
        assert reaches_goal(task, ar.plan)
    if refined == 0:
        f = rec["searches"]
        assert ex.states == f["brfs_exhaustive"]["expanded"]
        assert br.solved == (f["brfs"]["status"] == "solved")
        if br.solved:
            assert len(br.plan) == f["brfs"]["plan_length"]
        assert (ar.status == Status.SOLVED) == (f["astar_blind"]["status"] == "solved")
        if ar.status == Status.SOLVED:
            assert ar.cost == f["astar_blind"]["plan_cost"]
    assert compared + refined == states


@pytest.fixture(scope="module")
def blocks_task():
    return mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl")


@pytest.mark.parametrize(
    "run",
    [
        lambda t, **kw: search.brfs(t, **kw),
        lambda t, **kw: search.astar(t, heuristic="blind", **kw),
        lambda t, **kw: search.astar(t, heuristic="ff", **kw),
        lambda t, **kw: search.gbfs(t, heuristic="ff", **kw),
        lambda t, **kw: search.beam(t, heuristic="ff", **kw),
        lambda t, **kw: search.astar_iw(t, **kw),
        lambda t, **kw: search.iw(t, **kw),
        lambda t, **kw: search.siw(t, **kw),
        lambda t, **kw: search.liw(t, **kw),
        lambda t, **kw: search.rollout_iw(t, **kw),
    ],
    ids=["brfs", "astar_blind", "astar_ff", "gbfs", "beam", "astar_iw", "iw", "siw", "liw", "rollout_iw"],
)
def test_searches_take_the_option(blocks_task, run):
    off = run(blocks_task)
    on = run(blocks_task, symmetry_pruning="wl1")
    for r in (off, on):
        if getattr(r, "solved", None) or getattr(r, "status", None) == Status.SOLVED:
            assert reaches_goal(blocks_task, r.plan)
    with pytest.raises(ValueError, match="symmetry_pruning"):
        run(blocks_task, symmetry_pruning="gi")


def test_pruning_shrinks_a_symmetric_space():
    gripper = FORK_DATA / "gripper"
    if not (gripper / "test_problem4.pddl").exists():
        pytest.skip("set MYMYR_FORK_DATA")
    task = mymyr.Task.from_pddl(gripper / "domain.pddl", gripper / "test_problem4.pddl")
    off = search.brfs(task, witness_pruning=False, stop_at_goal=False)
    on = search.brfs(task, witness_pruning=False, symmetry_pruning="wl1", stop_at_goal=False)
    assert on.states < off.states
    s = task.initial_state
    assert len(task.applicable_actions(s, symmetry_pruning="wl1")) < len(task.applicable_actions(s))


def test_bad_values_and_partial(blocks_task):
    s = blocks_task.initial_state
    for bad in ("WL1", "gi", "", 1):
        with pytest.raises(ValueError, match="symmetry_pruning must be 'off' or 'wl1'"):
            blocks_task.applicable_actions(s, symmetry_pruning=bad)
    schema = blocks_task.schemas[0]
    assert blocks_task.applicable_actions(s, schema=schema, symmetry_pruning="off") == blocks_task.applicable_actions(
        s, schema=schema
    )
    kept = blocks_task.applicable_actions(s, schema=schema, symmetry_pruning="wl1")
    assert {str(a) for a in kept} <= {str(a) for a in blocks_task.applicable_actions(s, symmetry_pruning="wl1")}
    with pytest.raises(ValueError, match="partial"):
        blocks_task.applicable_actions(s, schema=schema, partial={0: 0}, symmetry_pruning="wl1")
