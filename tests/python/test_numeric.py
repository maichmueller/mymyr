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


def test_device_numeric_expand_and_search(counters):
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    torch = pytest.importorskip("torch")
    from mymyr import search

    ctx = mc.Context(0, max_bytes=1 << 28)
    uploaded = mc.DeviceTask(counters, ctx)
    assert uploaded.validate() == (0, 0)
    cpu_brfs = search.brfs(counters)
    gpu_brfs = mc.brfs(counters, ctx=ctx)
    assert gpu_brfs.states == cpu_brfs.states
    stored = torch.from_dlpack(gpu_brfs.state_words()).cpu().numpy()
    decoded = counters.decode(stored)
    assert len(decoded) == gpu_brfs.states and decoded[0] == counters.initial_state
    assert len(set(decoded)) == gpu_brfs.states
    starts = [counters.initial_state] + counters.successor_states(counters.initial_state)
    arr, _ = rows(counters, starts)
    device_rows = torch.from_numpy(arr).cuda()
    expected = rl.expand(counters, arr, goal=True)
    actual = rl.expand(counters, device_rows, goal=True, ctx=ctx)
    assert actual.numeric_words == counters.numeric_words
    assert np.array_equal(torch.from_dlpack(actual.succ).cpu().numpy(), expected.succ)
    assert np.array_equal(torch.from_dlpack(actual.goal).cpu().numpy(), expected.goal)
    padded = actual.pad()
    assert padded.succ.shape[-1] == expected.words + counters.numeric_words
    batch = mc.multi_iw(counters, starts, ctx=ctx)
    for i, start in enumerate(starts):
        want = search.iw(counters, start=start)
        assert batch.status[i] == want.status
        assert len(batch.plan(i)) == len(want.plan)


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


def test_numeric_device_arrays_have_programs(counters, hydropower):
    for task in (counters, hydropower):
        arrays = task.device_arrays()
        assert arrays["section_numeric"] == 1
        assert arrays["numeric_slots"] == task.numeric_slots
        assert arrays["num_code"].dtype == np.uint32 and arrays["num_code"].shape[1] == 4
        assert arrays["num_effects"].shape[1] == 8
        assert arrays["num_tables"].dtype == np.uint64
        assert arrays["num_initial"].dtype == np.float64
        assert np.array_equal(arrays["num_initial"], task.initial_state.numeric_values())
        assert not arrays["num_code"].flags.writeable


def test_classical_device_arrays_have_no_numeric_section():
    task = mymyr.Task.from_text(str(ROOT / "tests/data/tasks/gripper__prob05.txt"))
    arrays = task.device_arrays()
    assert "section_numeric" not in arrays and "num_code" not in arrays


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower", "m-refuel-adl"])
def test_numeric_cuda_labels_heuristics_and_table_starts(name):
    mc = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not mc.available():
        pytest.skip("no visible CUDA device")
    torch = pytest.importorskip("torch")
    from mymyr import search

    task = numeric_task(name, atoms="frozen")
    states = [task.initial_state]
    for i in range(8):
        actions = task.applicable_actions(states[-1])
        states.append(task.apply(states[-1], actions[i % len(actions)]) if actions else task.initial_state)
    packed, _ = rows(task, states)
    device_rows = torch.from_numpy(packed.view(np.int64)).cuda()
    ctx = mc.Context(0, max_bytes=2 << 30)
    dt = mc.DeviceTask(task, ctx)
    labels = []
    for s in states:
        labels.extend((a.schema, list(a.binding)) for a in task.applicable_actions(s)[:8])
    labels = labels[:48]
    assert labels
    idx = np.arange(len(labels), dtype=np.int32) % len(states)
    schema = np.array([a[0] for a in labels], dtype=np.int32)
    binding = np.zeros((len(labels), max(1, task.label_width)), dtype=np.int32)
    for i, (_, b) in enumerate(labels):
        binding[i, :len(b)] = b
    indices, schemas, bindings = (torch.from_numpy(x).cuda() for x in (idx, schema, binding))
    derived = torch.from_numpy(dt.host_derived(states).view(np.int64)).cuda() if task.has_axioms else None
    expected_ok = [task.is_applicable(states[i], label) for i, label in zip(idx, labels)]
    assert dt.applicable(device_rows, indices, schemas, bindings, derived=derived).cpu().tolist() == expected_ok
    successors, status = dt.apply(device_rows, indices, schemas, bindings, derived=derived)
    successors = successors.cpu().numpy().view(np.uint64)
    for j, (i, label, ok) in enumerate(zip(idx, labels, expected_ok)):
        assert status[j].item() == (0 if ok else -1)
        expected = rows(task, [task.apply(states[i], label)])[0][0] if ok else np.zeros(packed.shape[1], np.uint64)
        assert np.array_equal(successors[j], expected)
    assert dt.goal(device_rows, derived=derived).cpu().tolist() == [task.is_goal(s) for s in states]
    for kind in ("max", "add", "ff"):
        h = mc.Heuristic(task, kind, ctx=ctx)
        expected = [h.reference(s) for s in states]
        assert h.evaluate(states).tolist() == expected
        assert h.evaluate(device_rows).cpu().tolist() == expected
        if kind != "ff":
            cpu = search.Heuristic(task, kind)
            assert expected == [cpu(s) for s in states]
    table = rl.TaskTable([task, task])
    ids = [0, 1, 0, 1]
    starts = states[:4]
    host = mc.batched_iw1(table, starts, task_ids=ids, ctx=ctx, max_states=20000)
    device = mc.batched_iw1(table, torch.from_numpy(rows(task, starts)[0].view(np.int64)).cuda(), task_ids=ids, ctx=ctx, max_states=20000)
    for i, start in enumerate(starts):
        expected = search.iw(task, start=start, max_arity=1, max_states=20000)
        assert host.status[i] == device.status[i] == expected.status
        assert host.goal_state(i) == device.goal_state(i) == expected.goal_state
        assert host.cost(i) == device.cost(i) == expected.cost
    ctx.synchronize()
