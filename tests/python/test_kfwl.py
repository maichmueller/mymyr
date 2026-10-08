"""k-FWL certificates with k = 2, 3, 4 (ObjectGraph.kfwl_certificate, symmetry pruning with certificate="kfwl").

The partition of the sampled states of the state-space suite into certificate classes is compared with the fork's
kfwl::compute_certificate<K> through tests/data/kfwl/fork_kfwl.json (written by
tests/data/fork_golden/search_fork/run_kfwl.py; the format is in tests/data/fork_golden/README.md). Uses the fork's
own instances (env MYMYR_FORK_DATA; skipped when missing).
"""

import json
import os
import pathlib
import re

import pytest

import mymyr
from mymyr import datasets
from mymyr.rl import TaskTable

from conftest import ROOT

FORK_DATA = pathlib.Path(os.environ.get("MYMYR_FORK_DATA", "/nonexistent/mymyr-fork-data"))
GOLDEN = ROOT / "tests/data/kfwl/fork_kfwl.json"
SUITE = re.findall(r'^\s*\{"([^"]+)", "([^"]+)"', (ROOT / "tests/cpp/datasets/fork_cases.inc").read_text(), re.M)
CAPS = {2: 64, 3: 32, 4: 16}  # largest object graph certified per k here (the C++ test goes further)

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def fork_task(d, p):
    dom, prob = FORK_DATA / d / "domain.pddl", FORK_DATA / d / p
    if not prob.exists():
        pytest.skip(f"fork data missing: {prob}")
    return mymyr.Task.from_pddl(dom, prob, atoms="frozen")


def fnv(s):
    h = 0xCBF29CE484222325
    for b in s.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def state_key(state, numeric):
    key = "\n".join(sorted(str(a) for a in state.atoms()))
    if numeric:
        key += "".join("\n=%.17g" % v for v in state.numeric_values())
    return f"{fnv(key):016x}"


def classes(*labellings):
    """Canonical class ids (first occurrence) of the finest partition that puts equal labels of any labelling in one
    class (the join of the labellings' partitions)."""
    m = len(labellings[0])
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for labels in labellings:
        first = {}
        for i, label in enumerate(labels):
            j = first.setdefault(label, i)
            parent[find(i)] = find(j)
    ids = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(m)]


@pytest.fixture(scope="module")
def golden():
    return {r["task"]: r for r in json.loads(GOLDEN.read_text())["tasks"]}


@pytest.mark.parametrize("d,p", SUITE, ids=[f"{d}/{p}" for d, p in SUITE])
def test_partitions_match_the_fork(golden, d, p):
    """mymyr's k-FWL classes of the sampled states equal the fork's, with the classes of isomorphic states (nauty)
    joined: the fork's k-FWL is not invariant under relabelling for k >= 3 and splits some of them."""
    rec = golden[f"{d}/{p}"]
    task = fork_task(d, p)
    space = datasets.state_space(task, remove_if_unsolvable=False)
    assert space.num_states == rec["states"]
    numeric = task.numeric_slots > 0
    ids = {state_key(space.state(v), numeric): v for v in range(space.num_states)}
    graphs = [datasets.object_graph(space.state(ids[key])) for key in rec["keys"]]
    assert [g.num_vertices for g in graphs] == rec["n"]
    for k in (2, 3, 4):
        cap = min(CAPS[k], rec[f"k{k}_max_n"])
        sel = [i for i, g in enumerate(graphs) if g.num_vertices <= cap]
        mine = [graphs[i].kfwl_certificate(k) for i in sel]
        fork = [rec[f"k{k}"][i] for i in sel]
        nauty = [rec["nauty"][i] for i in sel]
        assert classes(mine) == classes(fork, nauty), f"k = {k}"
        assert classes(mine) == classes(mine, nauty), f"k = {k} separates isomorphic states"


def test_kfwl4_certificates_and_limits():
    task = fork_task("gripper", "p-2-0.pddl")
    space = datasets.state_space(task, remove_if_unsolvable=False)
    graphs = [datasets.object_graph(space.state(v)) for v in range(space.num_states)]
    certs = {k: {g.kfwl_certificate(k) for g in graphs} for k in (2, 3, 4)}
    assert len(certs[2]) == len(certs[3]) == len(certs[4]) == 12  # the fork's 12 isomorphism classes
    assert all(0 <= c < 2**128 for c in certs[4])
    g = graphs[0]
    assert g.num_vertices == 14
    assert g.kfwl_certificate(4) == g.kfwl_certificate(k=4, max_tuples=14**4, max_round_work=14**5)
    with pytest.raises(ValueError, match=r"4-FWL .* n = 14 .*max_tuples"):
        g.kfwl_certificate(4, max_tuples=14**4 - 1)
    with pytest.raises(ValueError, match=r"3-FWL .* n = 14 .*max_round_work"):
        g.kfwl_certificate(3, max_round_work=1000)
    for k in (1, 5):
        with pytest.raises(ValueError):
            g.kfwl_certificate(k)


def test_symmetry_pruning_with_k4():
    task = fork_task("gripper", "p-2-0.pddl")
    for threads in (1, 4, 8):
        sym = datasets.state_space(task, remove_if_unsolvable=False, symmetry_pruning=True, k=4, threads=threads)
        assert sym.symmetry_reduced and sym.num_states == 12  # the fork's symmetry-reduced space (nauty)
    tasks = [fork_task("gripper", p) for p in ("p-1-0.pddl", "p-2-0.pddl", "test_problem2.pddl")]
    gss = datasets.generalized_state_space(tasks, remove_if_unsolvable=False, symmetry_pruning=True, k=4)
    assert gss.symmetry_reduced and len(gss.spaces) == 2  # p-2-0 and test_problem2 are isomorphic
    table = TaskTable.from_pddl(FORK_DATA / "gripper" / "domain.pddl",
                                [FORK_DATA / "gripper" / p for p in ("p-1-0.pddl", "p-2-0.pddl")], atoms="frozen")
    kb = datasets.KnowledgeBase(table, generalized=True, symmetry_pruning=True, k=4, threads=2)
    assert [s.num_states for s in kb.state_spaces] == [6, 12]
    with pytest.raises(ValueError):
        datasets.state_space(task, symmetry_pruning=True, k=5)
