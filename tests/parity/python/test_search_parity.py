"""mymyr.search against the fork's golden data (tests/parity/data/expected, the C++ parity tests' fixtures), through the
Python API."""

import json

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from parity_support import DATA, IPC, WORK

GOLDEN = DATA / "expected"

# IW differs from the fork's per-pass counts only by successor order on these (canonical order vs the fork's
# KPKC order); rubiks-cube is slow in the fork's order.
ORDER_ONLY = ("philosophers", "pegsol", "satellite-ipc", "sokoban-ipc", "rovers-ipc", "floortile-ipc", "transport-ipc",
              "organic-synthesis")
SKIP = ORDER_ONLY + ("folding", "transport-opt08", "rubiks-cube")


def golden_cases():
    out = []
    if not GOLDEN.is_dir():
        return out
    for f in sorted(GOLDEN.glob("*.json")):
        name = f.stem
        if name.startswith(SKIP):
            continue
        out.append(name)
    return out


def load_golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def pddl_of(g):
    tag, prob = g["source"]["tag"], g["source"]["problem"]
    dom = g["source"].get("domain_file", "domain.pddl")
    if tag.startswith("ipc/"):
        if IPC is None:
            return None, None
        d = IPC / tag[4:] / "test"
    else:
        d = WORK / "mimir-cs" / "Benchmark" / tag
    return d / dom, d / prob


_TASKS = {}


def golden_task(name):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    if name not in _TASKS:
        g = load_golden(name)
        domain, problem = pddl_of(g)
        if domain is None:
            pytest.skip(f"PDDL of {name} not found (set MYMYR_PARITY_DATA)")
        if not domain.is_file() or not problem.is_file():
            pytest.skip(f"PDDL of {name} not found ({problem})")
        _TASKS[name] = mymyr.Task.from_pddl(domain, problem)
    return _TASKS[name]


def passes(r):
    return [(p.arity, p.expanded, p.generated, p.generated_in_tree) for p in r.passes]


def golden_passes(entry):
    return [(p["arity"], p["expanded"], p["generated"], p["generated_in_tree"]) for p in entry["passes"]]


def replay(task, plan):
    s = task.initial_state
    for a in plan:
        s = task.apply(s, a)
    return s


# ------------------------------------------------------------------------------------------------ golden replay


@pytest.mark.parametrize("name", golden_cases())
def test_golden_iw_per_pass(name):
    g = load_golden(name)
    task = golden_task(name)
    for entry in g["iw"]:
        if entry["status"] not in ("solved", "failed"):  # the fork stopped on a budget
            continue
        r = search.iw(task, max_arity=entry["k"], max_seconds=60)
        assert passes(r) == golden_passes(entry), (name, entry["k"])
        assert str(r.status) == ("solved" if entry["status"] == "solved" else "exhausted")
        if entry["status"] == "solved":
            assert len(r.plan) == entry["plan_length"]
            assert task.is_goal(replay(task, r.plan))


@pytest.mark.parametrize("name", [n for n in golden_cases()
                                  if load_golden(n)["brfs"].get("status") == "exhausted"
                                  and load_golden(n)["brfs"]["states"] <= 400_000])
def test_golden_brfs_counts(name):
    g = load_golden(name)["brfs"]
    r = search.brfs(golden_task(name), witness_pruning=False, stop_at_goal=False)
    assert r.exhausted and r.states == g["states"] and r.generated == g["generated"]


@pytest.mark.parametrize("name", [n for n in golden_cases()
                                  if load_golden(n)["astar"].get("status") == "solved"
                                  and load_golden(n)["astar"]["expanded"] <= 50_000])
def test_golden_astar_optimal_cost(name):
    g = load_golden(name)["astar"]
    task = golden_task(name)
    # the golden file's heuristic: blind with the task's action costs, else unit-cost h_max
    kw = dict(heuristic="blind", costs="real") if g["heuristic"] == "blind" else dict(heuristic="max", costs="unit")
    for lazy in (False, True):
        r = search.astar(task, lazy=lazy, **kw)
        assert r.status == Status.SOLVED
        assert r.cost == g["optimal_cost"], (name, lazy)
        assert task.is_goal(replay(task, r.plan))


def test_golden_heuristic_values_on_walk():
    """Initial-state h values of the Heuristic object equal the fork's (golden walks, step 0)."""
    name = "gripper__prob05"
    g = load_golden(name)["walks"]
    task = golden_task(name)
    names = g["heuristics"]
    step0 = g["walks"][0]["steps"][0]["h"]
    for kind, key in (("blind", "blind"), ("max", "hmax"), ("add", "hadd"), ("ff", "hff")):
        if key in step0:
            assert search.Heuristic(task, kind)(task.initial_state) == step0[key], (kind, names)
