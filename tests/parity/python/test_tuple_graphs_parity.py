"""Parity: tuple graphs against the recorded reference graphs on the state-space suite
(tests/parity/data/tuple_graphs/fork_tuple_graphs.json, written by tests/parity/tools/fork_golden/search_fork/
run_tuple_graphs.py; the format is in tests/parity/tools/fork_golden/README.md)."""

import json

import pytest

import mymyr
from mymyr import datasets

from parity_support import DATA, FORK_DATA

GOLDEN = DATA / "tuple_graphs/fork_tuple_graphs.json"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def fork_task(d, p):
    dom, prob = FORK_DATA / d / "domain.pddl", FORK_DATA / d / p
    if not prob.exists():
        pytest.skip(f"reference data missing: {prob} (set MYMYR_PARITY_DATA)")
    return mymyr.Task.from_pddl(dom, prob, atoms="dense")


def space_of(task, **options):
    s = datasets.state_space(task, threads=1, remove_if_unsolvable=False, **options)
    assert s is not None
    return s


def fnv(s):
    h = 0xCBF29CE484222325
    for b in s.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def set_hash(items):
    return f"{sum(fnv(s) for s in items) & 0xFFFFFFFFFFFFFFFF:016x}"


def state_key(state, numeric):
    key = "\n".join(sorted(str(a) for a in state.atoms()))
    if numeric:
        key += "".join("\n=%.17g" % v for v in state.numeric_values())
    return f"{fnv(key):016x}"


def digest(g, keys):
    """Per distance the vertex, edge and problem-vertex counts, and the hashes of the vertex, edge and problem-vertex
    sets, with vertices named by distance and sorted atom names and states by their keys."""
    ids, vs, es, qs, n, m, p = {}, set(), set(), set(), [], [], []
    for d in range(g.num_distances):
        at = set()
        for v in g.vertices_at(d):
            ids[v] = f"{d}:" + "".join(sorted(str(a) for a in g.atoms(v)))
            vs.add(ids[v] + ":" + ",".join(sorted(keys[x] for x in g.problem_vertices(v))))
            at.add(ids[v])
        n.append(len(at))
        layer = {f"{d}:{keys[x]}" for x in g.problem_vertices_at(d)}
        p.append(len(layer))
        qs |= layer
    for d in range(1, g.num_distances):
        at = {f"{ids[u]}>{ids[v]}" for v in g.vertices_at(d) for u in g.predecessors(v)}
        m.append(len(at))
        es |= at
    return {"n": n, "m": m, "p": p, "v": set_hash(vs), "e": set_hash(es), "q": set_hash(qs)}


GOLDEN_TASKS = {r["task"]: r for r in json.loads(GOLDEN.read_text())["tasks"]}


@pytest.mark.parametrize("name", sorted(GOLDEN_TASKS))
def test_tuple_graphs_match_the_reference(name):
    record = GOLDEN_TASKS[name]
    d, p = name.split("/")
    task = fork_task(d, p)
    s = space_of(task)
    assert s.num_states == record["states"]
    keys = [state_key(s.state(i), task.numeric_slots > 0) for i in range(s.num_states)]
    assert len(set(keys)) == len(keys)
    index = {k: i for i, k in enumerate(keys)}
    roots = [index[k] for k in record["roots"]]
    for config, graphs in record["graphs"].items():
        width, pruning = int(config[1]), len(config) < 4 or config[3] == "1"
        for r, want in zip(roots, graphs, strict=True):
            assert digest(datasets.tuple_graph(s, r, width=width, dominance_pruning=pruning), keys) == want, (config, r)
