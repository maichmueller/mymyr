"""Determinism across platforms and standard libraries.

Hashes of order-sensitive outputs on tasks checked into the repository: the normalized text tasks, and three PDDL
tasks through the loki front end (its translation order). Compared with tests/data/determinism.json, which was
written on Linux (GCC 16, libstdc++). The same test on macOS (Apple Clang, libc++) is the Mac-vs-Linux check: a
std::sort tie, a <random> distribution or a hash-table iteration that decides an order shows up there as a mismatch
in the named entry.

After an intended change of an order (a new tie-breaking rule, a new search default), regenerate the file and say why
in the commit:

    python tests/python/test_determinism.py --update
"""

import hashlib
import json
import platform
import sys

import numpy as np
import pytest

import mymyr
from mymyr import datasets, rl, search

from conftest import ROOT, SMALL_TASKS, text_task, walk

DATA = ROOT / "tests/data/determinism.json"
PDDL = {
    "blocks": ("tests/data/pddl/blocks", "probBLOCKS-8-0.pddl"),
    "logistics00": ("tests/data/pddl/logistics00", "probLOGISTICS-6-1.pddl"),
    "philosophers": ("tests/data/pddl/philosophers", "p03-phil4.pddl"),
}
CASES = [f"text:{n}" for n in SMALL_TASKS] + [f"pddl:{n}" for n in PDDL]
BUDGET = 20_000  # expansions (searches) and states (BrFS) per case
DATASETS_BUDGET = 100_000  # states of a state space (the larger tasks end as out_of_states)


def load(case):
    kind, name = case.split(":", 1)
    if kind == "text":
        return text_task(name)
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    d, p = PDDL[name]
    return mymyr.Task.from_pddl(ROOT / d / "domain.pddl", ROOT / d / p)


def arr(x):
    a = np.ascontiguousarray(np.asarray(x))
    return [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]


def plan(r):
    return [str(r.status), [str(a) for a in r.plan], repr(r.cost)]


def records(task):
    """The order-sensitive outputs of one task, by entry name."""
    s0 = task.initial_state
    states = walk(task, steps=16, seed=5, walks=4)
    exp = rl.expand(task, states)
    brfs = search.brfs(task, max_states=BUDGET, fingerprint=True)
    iw = search.iw(task, max_arity=2, max_expanded=BUDGET)
    siw = search.siw(task, max_arity=2, max_expanded=BUDGET)
    astar = search.astar(task, heuristic="max", max_expanded=BUDGET)
    gbfs = search.gbfs(task, heuristic="ff", max_expanded=BUDGET)
    lazy = search.gbfs(task, heuristic="ff", lazy=True, max_expanded=BUDGET)
    beam = search.beam(task, width=32, max_expanded=BUDGET)
    ss = datasets.generate(task, threads=2, max_states=DATASETS_BUDGET, remove_if_unsolvable=False)
    # every array but the state words (their slot layout depends on the task's atom mode and interning history)
    space = [] if ss.space is None else [
        [arr(v) for k, v in sorted(ss.space.arrays().items()) if k != "state_words"],
        arr(datasets.StateSpaceSampler(ss.space, seed=3).sample_states(64)),
    ]
    return {
        "task": [task.fingerprint, task.num_atoms, [str(a) for a in s0.atoms()], arr(task.encode([s0]))],
        "applicable": [str(a) for a in task.applicable_actions(s0)],
        "expand": [arr(exp.succ), arr(exp.offsets), arr(exp.schema), arr(exp.binding), arr(exp.parent)],
        "brfs": [brfs.states, brfs.expanded, brfs.generated, brfs.layers, brfs.fingerprint],
        "iw": plan(iw) + [[(p.arity, p.expanded, p.generated) for p in iw.passes]],
        "siw": plan(siw) + [siw.total.expanded],
        "astar": plan(astar) + [astar.stats.expanded, astar.stats.generated],
        "gbfs": plan(gbfs) + [gbfs.stats.expanded],
        "gbfs_lazy": plan(lazy) + [lazy.stats.expanded],
        "beam": plan(beam) + [beam.stats.expanded],
        "walks": rl.random_walks(task, 64, 16, seed=11),
        "datasets": [str(ss.status)] + space,
    }


def digests(task):
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16] for k, v in records(task).items()}


@pytest.fixture(scope="module")
def expected():
    if not DATA.is_file():
        pytest.skip(f"{DATA} missing: run `python tests/python/test_determinism.py --update`")
    return json.loads(DATA.read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_order_hashes(case, expected):
    if case not in expected:
        pytest.skip(f"{case} not in {DATA.name}")
    got = digests(load(case))
    differ = sorted(k for k in expected[case] if got.get(k) != expected[case][k])
    assert not differ, f"{case}: {differ} differ from {DATA.name} (written on {json.loads(DATA.read_text())['platform']})"


def main():
    if sys.argv[1:] != ["--update"]:
        raise SystemExit("usage: python tests/python/test_determinism.py --update")
    cases = {}
    for case in CASES:
        cases[case] = digests(load(case))
        print(case, cases[case])
    info = f"{platform.system()} {platform.machine()}, {mymyr.build_info()}"
    DATA.write_text(json.dumps({"platform": info, "cases": cases}, indent=1, sort_keys=True) + "\n")
    print("wrote", DATA)


if __name__ == "__main__":
    main()
