"""Free-threaded stress: 64 Python threads share one lazy-slot Task (atoms are interned while they run) and
drive the per-state API and rl.expand on it; every thread's results equal a single-threaded run on a separate Task.
Labels, goal flags and canonical hashes are slot-order independent, so the two tasks are comparable."""

import random
import sys
import threading

import numpy as np
import pytest

from mymyr import rl
from conftest import text_task

THREADS = 64
STEPS = 25


def trace(task, api, seed):
    """A seeded walk through `api` (the task or a handle): per step the labels, the successors' canonical hashes and
    the goal flag, plus an rl.expand of the states seen so far checked against the per-state results."""
    rng = random.Random(seed)
    s = api.initial_state
    out, seen = [], []
    for _ in range(STEPS):
        pairs = api.successors(s)
        labels = [a.label for a, _ in pairs]
        hashes = [t.canonical_hash() for _, t in pairs]
        assert labels == [a.label for a in api.applicable_actions(s)]
        assert [api.apply(s, a) for a, _ in pairs] == [t for _, t in pairs]
        out.append((labels, hashes, api.is_goal(s)))
        seen.append(s)
        if not pairs:
            s = api.initial_state
            continue
        s = pairs[rng.randrange(len(pairs))][1]
    return out, seen


def check_expand(task, seen, steps, kw):
    exp = rl.expand(task, seen, goal=True, **kw)
    off = exp.offsets.tolist()
    for i, (labels, hashes, _) in enumerate(steps[: len(seen)]):
        got = [(int(exp.schema[j]), tuple(int(o) for o in exp.binding[j] if o >= 0)) for j in range(off[i], off[i + 1])]
        assert got == labels
        assert [task.state(exp.succ[j]).canonical_hash() for j in range(off[i], off[i + 1])] == hashes
    assert np.array_equal(rl.is_goal(task, seen), [g for _, _, g in steps[: len(seen)]])


@pytest.mark.parametrize("name", ["openstacks-opt08-adl__p03", "logistics00__probLOGISTICS-6-1", "blocks__probBLOCKS-8-0"])
def test_sixty_four_threads_share_one_lazy_task(name):
    reference = text_task(name, atoms="lazy")
    expected = [trace(reference, reference, seed)[0] for seed in range(THREADS)]

    task = text_task(name, atoms="lazy")  # fresh: no slots assigned yet
    shared_pool = rl.ThreadPool(4)
    barrier = threading.Barrier(THREADS)
    results, errors = [None] * THREADS, []

    def work(i):
        try:
            api = task.local() if i % 2 else task
            barrier.wait()
            steps, seen = trace(task, api, i)
            results[i] = steps
            kw = ({}, {"pool": shared_pool}, {"threads": 2}, {"canonical": True})[i % 4]
            check_expand(task, seen, steps, kw)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert results == expected
    assert not sys._is_gil_enabled()


def test_threads_share_states_across_handles():
    """States made by one thread's handle are used by all others (value semantics across handles)."""
    task = text_task("gripper__prob05", atoms="lazy")
    rng = random.Random(0)
    states = [task.initial_state]
    for _ in range(40):
        succ = task.successor_states(states[-1])
        states.append(succ[rng.randrange(len(succ))])
    expected = [[a.label for a in task.applicable_actions(s)] for s in states]
    words = task.encode(states)
    barrier = threading.Barrier(THREADS)
    errors = []

    def work(i):
        try:
            h = task.local()
            barrier.wait()
            for _ in range(5):
                assert [[a.label for a in h.applicable_actions(s)] for s in states] == expected
                assert h.decode(words) == states
                assert set(h.successor_states(states[i % len(states)])) == set(task.successor_states(states[i % len(states)]))
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert not sys._is_gil_enabled()


def test_threads_share_one_python_heuristic():
    """A* and GBFS on 32 threads, all calling one Python heuristic (a callable, or an object with evaluate_batch) at
    once on one lazy-slot Task; every result equals the single-threaded one. A heuristic that raises stops only its
    own search."""
    from mymyr import search

    name = "logistics00__probLOGISTICS-6-1"

    def h(s):
        return 0.0 if s.is_goal() else 1.0

    class Batch:
        def evaluate_batch(self, states):
            return [h(s) for s in states]

    batch = Batch()

    def run(task, i):
        heur = (h, batch)[i % 2]
        if i % 4 < 2:
            r = search.astar(task, heuristic=heur, lazy=i % 8 >= 4)
        else:
            r = search.gbfs(task, heuristic=heur, lazy=i % 8 >= 4)
        return r.status, r.cost, r.stats.expanded, [str(a) for a in r.plan]

    reference = text_task(name, atoms="lazy")
    expected = [run(reference, i) for i in range(8)]
    task = text_task(name, atoms="lazy")
    threads_n = 32
    barrier = threading.Barrier(threads_n)
    results, errors = [None] * threads_n, []

    def boom(s):
        raise KeyError("boom")

    def work(i):
        try:
            api = task.local() if i % 3 == 0 else task
            barrier.wait()
            results[i] = run(api, i)
            with pytest.raises(KeyError, match="boom"):
                search.gbfs(api, heuristic=boom)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert results == [expected[i % 8] for i in range(threads_n)]
    assert not sys._is_gil_enabled()
