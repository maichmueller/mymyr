"""The versioned C API (mymyr/ext.h) seen from a pure-C downstream module that links nothing of mymyr."""

import re
import sys
import threading

import numpy as np
import pytest

import mymyr
from mymyr import _core, search
from conftest import ROOT, text_task, walk

consumer = pytest.importorskip("mymyr._testing._ext_consumer")


def test_capsule_and_version():
    assert _core._C_API_VERSION == (1, 2)
    abi, minor, size = consumer.abi()
    assert (abi, minor) == _core._C_API_VERSION and size > 16
    assert consumer.version() == mymyr.__version__


def test_state_hash_matches_native_and_ignores_trailing_zero_words():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2**63, size=(64, 4), dtype=np.uint64)
    expected = mymyr.hash_rows(rows)
    for r, h in zip(rows, expected):
        assert consumer.hash_state(r) == int(h)
        padded = np.concatenate([r, np.zeros(3, np.uint64)])
        assert consumer.hash_state(padded) == int(h)
        assert consumer.equal_states(r, padded)
    assert not consumer.equal_states(rows[0], rows[1])


def test_rejects_non_state_buffers():
    with pytest.raises(TypeError):
        consumer.hash_state(np.zeros(4, np.float64))
    with pytest.raises(TypeError):
        consumer.hash_state(np.zeros((2, 2), np.uint64))
    with pytest.raises(TypeError):
        consumer.task_info(np.zeros(4, np.uint64))


# ------------------------------------------------------------------------------------------------ minor 1


def test_task_view(blocks):
    uid, fp, n, p, s, w, mw, lw, atoms, mode = consumer.task_info(blocks)
    assert (uid, fp, n, s, w, mw, lw, atoms) == (blocks.uid, blocks.fingerprint, blocks.num_objects, blocks.num_schemas,
                                                blocks.words, blocks.max_words, blocks.label_width, blocks.num_atoms)
    assert p == blocks.num_predicates and mode == 1
    st = blocks.initial_state
    assert consumer.task_info(st)[0] == consumer.task_info(blocks.local())[0] == blocks.uid
    assert consumer.state_task_uid(st) == blocks.uid
    assert consumer.state_task_uid(np.asarray(st.words)) == 0  # plain buffers carry no task
    assert consumer.hash_state(st) == int(mymyr.hash_rows(np.asarray(st.words)[None, :])[0])


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "openstacks-opt08-adl__p03", "miconic-simpleadl__s10-2"])
def test_successors_apply_goal_atoms(name):
    task = text_task(name, atoms="lazy")
    for s in walk(task, steps=8, seed=2):
        c = consumer.successors(task, s)
        py = task.successors(s)
        assert [(sch, b) for sch, b, _ in c] == [a.label for a, _ in py]
        assert [task.state(np.array(w, np.uint64)) for _, _, w in c] == [t for _, t in py]
        assert consumer.count_successors_nogil(task, s) == len(py)
        if c:
            assert consumer.successors(task, s, 1) == c[:1]  # the emit callback can stop the enumeration
        for (a, t), (sch, b, w) in zip(py, c):
            assert consumer.apply(task, s, sch, list(b)) == w
        assert consumer.is_goal(task, s) == task.is_goal(s)
        for slot in s.atom_slots():
            atom = task.atom(slot)
            assert consumer.atom_slot(task, atom.predicate_index, atom.object_indices) == slot
            assert consumer.atom_of_slot(task, slot) == (atom.predicate_index, tuple(atom.object_indices))
    s = task.initial_state
    acts = task.applicable_actions(s)
    inapplicable = [(i, b) for i in range(task.num_schemas) for b in [[0] * task.formalism.schemas[i].arity]
                    if (i, tuple(b)) not in {a.label for a in acts}]
    for sch, b in inapplicable:
        assert consumer.apply(task, s, sch, b) is None
    assert consumer.atom_of_slot(task, 1 << 30) is None


def test_store(blocks):
    states = walk(blocks, steps=20, seed=6)
    ids, inserted, lookups, size, first = consumer.store_roundtrip(blocks, states + states[:3])
    distinct = list(dict.fromkeys(states))
    assert size == len(distinct)
    assert ids == [distinct.index(s) for s in states + states[:3]] == lookups
    assert inserted[: len(states)] == [states.index(s) == i for i, s in enumerate(states)] and not any(inserted[len(states):])
    assert blocks.state(np.array(first, np.uint64)) == states[0]


def test_c_api_from_many_threads(blocks):
    states = walk(blocks, steps=30, seed=1)
    expected = [consumer.successors(blocks, s) for s in states]
    results, errors = [None] * 16, []

    def work(i):
        try:
            h = blocks.local()
            results[i] = [consumer.successors(h, s) for s in states]
            assert [consumer.count_successors_nogil(h, s) for s in states] == [len(e) for e in expected]
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(r == expected for r in results)


def test_gil_stays_disabled():
    assert not sys._is_gil_enabled()


# ------------------------------------------------------------------------------------------------ minor 2

NUMERIC = sorted((ROOT / "tests/data/numeric_tasks").glob("*.txt"))
NUMERIC_IDS = [p.stem for p in NUMERIC]
CAP = 50_000


def numeric_task(name, **options):
    return mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks" / f"{name}.txt"), **options)


def plan_cost(task, plan):
    return float(re.search(r"cost = ([0-9.e+-]+)", search.format_plan(task, plan)).group(1))


def initial_metric(task):
    return plan_cost(task, [])


def test_table_layout_is_append_only():
    last, end, first, size = consumer.table_layout()
    assert last == 16 + 8 * 16 and end == first == 152  # minor 1: 5 + 12 entries behind the 16-byte header
    assert size == 152 + 8 * 13 == consumer.abi()[2]


def test_minor2_gating_reads_the_table_size():
    try:
        assert consumer.pretend_size(0)
        assert not consumer.pretend_size(152)  # a provider of minor 1
        with pytest.raises(RuntimeError, match="minor 2"):
            consumer.numeric_info(numeric_task("cs-counters"))
        assert consumer.task_info(text_task("blocks__probBLOCKS-8-0"))  # minor 1 stays available
        assert consumer.pretend_size(consumer.abi()[2])
    finally:
        consumer.pretend_size(0)


@pytest.mark.parametrize("name", NUMERIC_IDS)
def test_numeric_task_view(name):
    task = numeric_task(name)
    slots, words, storage, metric = consumer.numeric_info(task)
    assert (slots, words) == (task.numeric_slots, task.numeric_words)
    assert storage == (1 if task.numeric_storage == "i32" else 0) or slots == 0
    names = consumer.numeric_names(task)
    assert [n for n, _ in names] == list(task.numeric_names)
    assert [i for _, i in names] == list(range(slots))
    assert consumer.numeric_slot_of(task, "(no-such-function)") == -1
    assert consumer.task_info(task)[0] == task.uid  # numeric tasks are accepted by task_from_py
    s0 = task.initial_state
    assert 0 <= metric <= 2 and consumer.metric_initial(task, s0) == initial_metric(task)


def test_numeric_slot_names_are_truncated_like_snprintf():
    task = numeric_task("cs-counters")
    full = task.numeric_names[0]
    assert consumer.numeric_name_truncated(task, 0, 512) == (full, len(full))
    assert consumer.numeric_name_truncated(task, 0, 4) == (full[:3], len(full))
    assert consumer.numeric_name_truncated(task, 0, 0)[1] == len(full)
    assert consumer.numeric_name_truncated(task, task.numeric_slots, 512) is None


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower", "cs-tpp", "cs-drone"])
def test_values_read_through_the_c_api_equal_the_state_values(name):
    task = numeric_task(name)
    for s in walk(task, steps=25, seed=3):
        assert consumer.numeric_values(task, s) == [float(v) for v in s.numeric_values()]
        assert consumer.state_task_uid(s) == task.uid
    with pytest.raises(ValueError):
        consumer.numeric_values(task, np.asarray(task.initial_state.words))  # a buffer has no numeric words


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower", "cs-tpp", "m-refuel", "m-zenotravel-numeric", "cs-delivery"])
def test_numeric_successors_apply_goal(name):
    task = numeric_task(name)
    for s in walk(task, steps=12, seed=5):
        g = 7.5
        c = consumer.successors_state(task, s, g)
        py = task.successors(s)
        assert [(sch, b) for sch, b, *_ in c] == [a.label for a, _ in py]
        for (_, t), (sch, b, words, numeric, metric) in zip(py, c):
            assert tuple(words) == tuple(np.asarray(t.words)) and numeric == tuple(int(w) for w in t.numeric_words)
            assert consumer.apply_state(task, s, g, sch, list(b)) == (words, numeric, metric)
            assert consumer.numeric_values(task, t) == [float(v) for v in t.numeric_values()]
        assert consumer.count_successors_state_nogil(task, s) == len(py)
        assert consumer.is_goal_state(task, s) == task.is_goal(s)
        if task.numeric_slots:  # the atom-words entries of minor 1 refuse a task whose states carry values
            assert consumer.minor1_on_numeric(task, s) == (-1, -1, -1, -1, -1)
    s = task.initial_state
    bad = [(i, [0] * task.formalism.schemas[i].arity) for i in range(task.num_schemas)]
    applicable = {a.label for a in task.applicable_actions(s)}
    for sch, b in bad:
        if (sch, tuple(b)) not in applicable:
            assert consumer.apply_state(task, s, 0.0, sch, b) is None


def test_the_metric_follows_the_task():
    unit, total, metric = numeric_task("cs-counters"), numeric_task("cs-tpp"), numeric_task("cs-delivery")
    assert [consumer.numeric_info(t)[3] for t in (unit, total, metric)] == [0, 1, 2]
    s = unit.initial_state
    assert {m for *_, m in consumer.successors_state(unit, s, 3.0)} == {4.0}  # unit costs: g + 1


def test_state_hash_and_equality_cover_the_values():
    task = numeric_task("cs-counters")
    s0 = task.initial_state
    succ = task.successor_states(s0)
    h0, h1, eq = consumer.state_hash_equal(s0, succ[0])
    assert not eq and h0 != h1
    assert consumer.state_hash_equal(s0, s0)[2]
    h, _, eq = consumer.state_hash_equal(succ[0], succ[0])
    assert eq and h == h1
    classical = text_task("blocks__probBLOCKS-8-0")
    c0 = classical.initial_state
    assert consumer.state_hash_equal(c0, c0)[0] == consumer.hash_state(c0)


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower", "cs-tpp"])
def test_numeric_store_keys_on_atoms_and_values(name):
    task = numeric_task(name)
    states = walk(task, steps=40, seed=4)
    ids, inserted, lookups, size, (first_words, first_numeric) = consumer.store_roundtrip_state(task, states + states[:5])
    distinct = list(dict.fromkeys(states))
    assert size == len(distinct)
    assert ids == [distinct.index(s) for s in states + states[:5]] == lookups
    assert inserted[: len(states)] == [states.index(s) == i for i, s in enumerate(states)] and not any(inserted[len(states):])
    assert tuple(first_words)[: len(np.asarray(states[0].words))] == tuple(np.asarray(states[0].words))
    assert first_numeric == tuple(int(w) for w in states[0].numeric_words)


@pytest.mark.parametrize("name", NUMERIC_IDS)
def test_brfs_through_the_c_api_matches_mymyr_brfs(name):
    """Whole-space (capped) breadth-first search written against the C API alone: same counts as search.brfs."""
    task = numeric_task(name)
    s0 = task.initial_state
    g0 = consumer.metric_initial(task, s0)
    ref = search.brfs(task, witness_pruning=False, max_states=CAP)
    states, expanded, generated, goals, layers, exhausted, solved, plan, _ = consumer.brfs(task, s0, g0, CAP, False)
    assert (states, expanded, generated, goals, layers, exhausted) == (
        ref.states, ref.expanded, ref.generated, ref.goal_states, ref.layers, ref.exhausted)
    assert not solved and plan == []


@pytest.mark.parametrize("name", NUMERIC_IDS)
def test_brfs_to_a_goal_through_the_c_api_matches_plan_and_cost(name):
    task = numeric_task(name)
    s0 = task.initial_state
    g0 = consumer.metric_initial(task, s0)
    ref = search.brfs(task, witness_pruning=False, max_states=250_000, stop_at_goal=True)
    states, expanded, generated, goals, layers, _, solved, plan, cost = consumer.brfs(task, s0, g0, 250_000, True)
    assert (states, expanded, generated, goals, layers, solved) == (
        ref.states, ref.expanded, ref.generated, ref.goal_states, ref.layers, ref.solved)
    if solved:
        assert plan == [(a.label[0], tuple(a.label[1])) for a in ref.plan]
        assert cost == plan_cost(task, ref.plan)
        s = s0
        for sch, b in plan:
            s = next(t for a, t in task.successors(s) if a.label == (sch, tuple(b)))
        assert task.is_goal(s) and consumer.is_goal_state(task, s)


def test_brfs_through_the_c_api_on_a_classical_task():
    task = text_task("gripper__prob05")
    s0 = task.initial_state
    ref = search.brfs(task, witness_pruning=False, max_states=CAP, stop_at_goal=True)
    states, expanded, generated, goals, layers, _, solved, plan, cost = consumer.brfs(task, s0, 0.0, CAP, True)
    assert (states, expanded, generated, solved) == (ref.states, ref.expanded, ref.generated, ref.solved)
    assert plan == [(a.label[0], tuple(a.label[1])) for a in ref.plan] and cost == len(plan)


def test_numeric_c_api_from_many_threads():
    task = numeric_task("cs-counters")
    s0 = task.initial_state
    g0 = consumer.metric_initial(task, s0)
    expected = consumer.brfs(task, s0, g0, CAP, False)
    states = walk(task, steps=30, seed=1)
    counts = [consumer.count_successors_state_nogil(task, s) for s in states]
    results, errors = [None] * 8, []

    def work(i):
        try:
            h = task.local()
            results[i] = (consumer.brfs(h, s0, g0, CAP, False),
                          [consumer.count_successors_state_nogil(h, s) for s in states],
                          [consumer.successors_state(h, s, 0.0) for s in states[:6]])
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reference = [consumer.successors_state(task, s, 0.0) for s in states[:6]]
    assert not errors and all(r == (expected, counts, reference) for r in results)
