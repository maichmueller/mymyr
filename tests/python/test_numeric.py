"""Numeric fluents in Python: numeric slots of a task, State.numeric_values(), value semantics and pickling
of states with numeric values, and [W + NN] rows in mymyr.rl."""

import pickle

import numpy as np
import pytest

import mymyr
import mymyr.rl as rl
from conftest import ROOT

NUMERIC = ROOT / "tests/data/numeric_tasks"


def numeric_task(name, **options):
    return mymyr.Task.from_text(str(NUMERIC / f"{name}.txt"), **options)


@pytest.fixture(scope="module")
def counters():
    return numeric_task("cs-counters")


@pytest.fixture(scope="module")
def hydropower():
    return numeric_task("cs-hydropower")  # f64 slots


def test_numeric_slots(counters, hydropower):
    assert counters.numeric_slots == 4 and counters.numeric_storage == "i32" and counters.numeric_words == 2
    assert hydropower.numeric_storage == "f64" and hydropower.numeric_words == hydropower.numeric_slots
    names = counters.numeric_names
    assert len(names) == 4 and all(n.startswith("(") and n.endswith(")") for n in names)
    s = counters.initial_state
    v = s.numeric_values()
    assert v.dtype == np.float64 and v.shape == (4,)
    assert s.numeric_words.dtype == np.uint64 and s.numeric_words.shape == (2,)
    assert "=" in repr(s)  # values are shown


def test_states_differ_by_their_values(counters):
    s = counters.initial_state
    succ = counters.successor_states(s)
    same_atoms = [t for t in succ if np.array_equal(t.words, s.words)]
    assert same_atoms  # counters: increments change values only
    for t in same_atoms:
        assert t != s and hash(t) != hash(s)
        assert not np.array_equal(t.numeric_values(), s.numeric_values())
    assert len(set(succ)) == len(succ)


def test_pickle_keeps_the_values(counters, hydropower):
    for task in (counters, hydropower):
        s = task.initial_state
        t = task.successor_states(s)[-1]
        for x in (s, t):
            y = pickle.loads(pickle.dumps(x))  # a new task instance: compare content
            assert y.canonical_hash() == x.canonical_hash() and np.array_equal(y.words, x.words)
            assert np.array_equal(y.numeric_values(), x.numeric_values())
            task2, z = pickle.loads(pickle.dumps((task, x)))  # the same task in one pickle: equal states
            assert z.task is task2 and z == task2.decode(task2.encode([z]))[0]


def rows(task, states):
    W = max(max(s.num_words for s in states), task.words, 1)
    NN = task.numeric_words
    out = np.zeros((len(states), W + NN), dtype=np.uint64)
    for i, s in enumerate(states):
        out[i, : s.num_words] = s.words
        out[i, W:] = s.numeric_words
    return out, W


def test_rl_rows_carry_the_numeric_words(counters, hydropower):
    for task in (counters, hydropower):
        s0 = task.initial_state
        batch = [s0] + task.successor_states(s0)
        arr, W = rows(task, batch)
        NN = task.numeric_words
        for states in (arr, batch):
            exp = rl.expand(task, states, goal=True)
            assert exp.numeric_words == NN and exp.succ.shape[1] == exp.words + NN
            want = [t for s in batch for t in task.successor_states(s)]
            assert exp.total == len(want)
            for j, t in enumerate(want):
                row = exp.succ[j]
                assert np.array_equal(row[exp.words:], t.numeric_words)
                assert np.array_equal(row[: t.num_words], t.words) and not row[t.num_words: exp.words].any()
                assert bool(exp.goal[j]) == t.is_goal()
            pad = exp.pad()
            assert pad.succ.shape[2] == exp.words + NN
        goals = rl.is_goal(task, arr)
        assert [bool(g) for g in goals] == [s.is_goal() for s in batch]
        # destination passing: succ is [capacity, W + NN]
        cap = 256
        succ = np.zeros((cap, W + NN), dtype=np.uint64)
        r = rl.expand_into(task, arr, succ=succ)
        assert not r["overflow"] and np.array_equal(succ[: r["total"]], rl.expand(task, arr, words=W).succ)
        with pytest.raises(ValueError):
            rl.expand(task, np.zeros((1, NN), dtype=np.uint64))  # no atom words


def test_device_task_rejects_numeric_tasks(counters):
    # the device format has no numeric section yet: uploading a numeric task must fail, not drop the values
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    ctx = mc.Context(0, max_bytes=1 << 28)
    with pytest.raises(ValueError, match="numeric"):
        mc.DeviceTask(counters, ctx)
    # nor do the device kernels read the values: the device BrFS and the device expand refuse numeric tasks
    with pytest.raises(ValueError, match="numeric"):
        mc.brfs(counters, ctx=ctx)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    rows = torch.from_dlpack(counters.encode([counters.initial_state], framework="torch")).cuda()
    with pytest.raises(ValueError, match="numeric"):
        rl.expand(counters, rows)


def replay_to_goal(task, plan):
    s = task.initial_state
    for a in plan:
        s = task.apply(s, a)
    assert task.is_goal(s)
    return s


@pytest.mark.parametrize("h", ["blind", "goal_count", "max", "add", "ff"])
def test_best_first_search_on_numeric_tasks(counters, h):
    from mymyr import search

    optimal = search.astar(counters, heuristic="blind")
    assert optimal.status == search.Status.SOLVED and optimal.cost == len(optimal.plan)  # unit costs
    for r in (search.astar(counters, heuristic=h), search.gbfs(counters, heuristic=h),
              search.gbfs(counters, heuristic=h, lazy=True)):
        assert r.status == search.Status.SOLVED, (r.algorithm, h)
        goal = replay_to_goal(counters, r.plan)
        assert np.array_equal(goal.numeric_values(), r.goal_state.numeric_values())
        assert r.cost >= optimal.cost
    if h in ("blind", "max"):  # admissible
        assert search.astar(counters, heuristic=h).cost == optimal.cost


def test_python_heuristic_reads_the_values(counters):
    from mymyr import search

    seen = []

    def h(s):
        seen.append(s.numeric_values().sum())
        return 0.0

    r = search.astar(counters, heuristic=h)
    assert r.status == search.Status.SOLVED and len(set(seen)) > 1
