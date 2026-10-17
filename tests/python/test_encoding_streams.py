"""The streams of the relational encoder through the Python API: a flush equals one encode of the rows in order, over
seeded random sequences of appends, updates and removals, with and without per-row goals, derived words and actions,
in both layouts and on one and several threads; the rules of the inputs, the errors, and the action rows of a flush."""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk

U64 = np.uint64
TASKS = ["gripper__prob05", "blocks__probBLOCKS-8-0", "philosophers__p03-phil4"]
# (goals, actions, derived words): goals are None, "pair" (fluent masks) or "quad" (with the derived masks)
VARIANTS = [
    (None, False, False),
    ("pair", False, False),
    (None, True, False),
    ("pair", True, True),
    ("quad", True, True),
]


def assert_same(a, b):
    first, second = a.as_dict(), b.as_dict()
    assert first.keys() == second.keys()
    for key, value in first.items():
        assert np.array_equal(value, second[key]), key


class Pool:
    """Candidate rows of a table (state words, instances, derived words, goals), and their expansions on demand."""

    def __init__(self, tasks, seed):
        self.tasks = tasks
        self.table = rl.TaskTable(tasks)
        self.rng = np.random.default_rng(seed)
        words, derived_words = self.table.words, self.table.derived_words
        rows, ids, derived = [], [], []
        for index, task in enumerate(tasks):
            states = walk(task, steps=8, seed=seed + index, walks=3)
            expansion = rl.expand(self.table, states, np.full(len(states), index, np.int32), derived=True,
                                   words=words)
            rows.append(expansion.succ)
            derived.append(expansion.derived)
            ids.append(np.full(len(expansion.succ), index, np.int32))
        self.rows = np.concatenate(rows)
        self.ids = np.concatenate(ids)
        self.derived = np.concatenate(derived)
        self.size = len(self.rows)
        valid = np.zeros((len(tasks), words), U64)
        derived_valid = np.zeros((len(tasks), derived_words), U64)
        for i in range(len(tasks)):
            valid[i] = np.bitwise_or.reduce(self.rows[self.ids == i], axis=0)
            if derived_words:
                derived_valid[i] = np.bitwise_or.reduce(self.derived[self.ids == i], axis=0)
        self.pos = self.rows & self.random(self.rows.shape)
        self.neg = valid[self.ids] & ~self.rows & self.random(self.rows.shape)
        self.derived_pos = derived_valid[self.ids] & self.random(self.derived.shape)
        self.derived_neg = derived_valid[self.ids] & ~self.derived_pos & self.random(self.derived.shape)

    def random(self, shape):
        return self.rng.integers(0, 2**64, shape, dtype=U64)

    def inputs(self, picks, variant, task_ids=True):
        """The keyword arguments of encode / append for rows `picks`, as the `variant` gives per-row inputs."""
        goals, actions, derived = variant
        picks = np.asarray(picks, dtype=np.int64)
        states, ids = self.rows[picks], self.ids[picks]
        out = {"states": states, "task_ids": ids}
        if derived:
            out["derived"] = self.derived[picks]
        if goals == "pair":
            out["goals"] = (self.pos[picks], self.neg[picks])
        elif goals == "quad":
            out["goals"] = (self.pos[picks], self.neg[picks], self.derived_pos[picks], self.derived_neg[picks])
        if actions:
            out["actions"] = rl.expand(self.table, states, ids)
        return out


def options_of(variant, layout):
    goals, actions, _ = variant
    targets = (("goal",) if goals else ()) + (("action",) if actions else ())
    return dict(goal_derivations=("plain", "satisfied", "unsatisfied") if goals else (), include_actions=actions,
                target_sources=targets, predicate_virtual_nodes=actions, layout=layout)


def pools():
    out = []
    for name in TASKS:
        out.append(pytest.param(name, id=name.split("__")[0]))
    return out


@pytest.fixture(scope="module")
def pool_of():
    cache = {}

    def make(name):
        if name not in cache:
            cache[name] = Pool([text_task(name)], seed=7)
        return cache[name]

    return make


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(str(x) for x in v))
@pytest.mark.parametrize("name", pools())
def test_a_stream_flush_equals_one_encode_over_random_appends(pool_of, name, variant, layout):
    pool = pool_of(name)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, layout))
    stream = encoder.stream()
    rng = random.Random(3)
    for round_ in range(5):
        picks = []
        for _ in range(rng.randint(1, 4)):
            chunk = [rng.randrange(pool.size) for _ in range(rng.choice([0, 1, 5, 17, 40]))]
            stream.append(**pool.inputs(chunk, variant))
            picks += chunk
        assert len(stream) == len(picks)
        threads = 1 if round_ % 2 else 3
        batch = stream.flush(threads=threads)
        assert len(stream) == 0
        whole = pool.inputs(picks, variant)
        assert_same(batch, encoder.encode(**whole))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(str(x) for x in v))
@pytest.mark.parametrize("name", pools())
def test_a_mutable_stream_flush_equals_one_encode_over_random_updates_and_removals(pool_of, name, variant, layout):
    pool = pool_of(name)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, layout))
    stream = encoder.mutable_stream()
    rng = random.Random(11)
    model = {}
    for step in range(60):
        op = rng.random()
        if op < 0.3 or not model:
            indices = []
            for _ in range(rng.choice([0, 1, 4, 9])):
                index = rng.randrange(-15, 80)
                if index not in model and index not in indices:
                    indices.append(index)
            picks = [rng.randrange(pool.size) for _ in indices]
            stream.append(indices, **pool.inputs(picks, variant))
            model.update(zip(indices, picks))
        elif op < 0.55:
            held = rng.sample(sorted(model), min(len(model), rng.choice([1, 3, 7])))
            picks = [rng.randrange(pool.size) for _ in held]
            stream.update(held, **pool.inputs(picks, variant))
            model.update(zip(held, picks))
        elif op < 0.8:
            held = rng.sample(sorted(model), min(len(model), rng.choice([1, 2, 6])))
            stream.remove(held)
            for index in held:
                del model[index]
        else:
            indices = sorted(model)
            assert stream.indices().tolist() == indices
            whole = pool.inputs([model[i] for i in indices], variant)
            batch = stream.flush(threads=1 + 2 * (step % 2))
            assert_same(batch, encoder.encode(**whole))
            assert len(stream) == len(model)
        assert len(stream) == len(model)
        assert all(index in stream for index in list(model)[:3])
    indices = sorted(model)
    assert_same(stream.flush(), encoder.encode(**pool.inputs([model[i] for i in indices], variant)))


def test_a_table_of_several_instances_streams_like_it_encodes():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    blocks = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / f"{p}.pddl") for p in ("smedium", "probBLOCKS-8-0")]
    pool = Pool(tasks, seed=2)
    assert set(pool.ids.tolist()) == {0, 1}
    variant = ("pair", True, True)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    stream = encoder.stream()
    mutable = encoder.mutable_stream()
    picks = list(range(0, pool.size, 3))
    for start in range(0, len(picks), 7):
        chunk = picks[start : start + 7]
        stream.append(**pool.inputs(chunk, variant))
        mutable.append([-1000 - p for p in chunk], **pool.inputs(chunk, variant))
    reference = encoder.encode(**pool.inputs(picks, variant))
    assert_same(stream.flush(), reference)
    assert_same(mutable.flush(), encoder.encode(**pool.inputs(picks[::-1], variant)))


def test_state_objects_stream_like_arrays():
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    states = walk(task, steps=7, walks=2)
    options = dict(goal_derivations=("plain", "satisfied"), target_sources=("goal", "action"), predicate_virtual_nodes=True)
    encoder = encoding.FlatRelationEncoder(task, **options)
    stream = encoder.stream()
    for chunk in (states[:4], states[4:5], states[5:]):
        stream.append(chunk, actions=rl.expand(table, chunk))
    assert_same(stream.flush(), encoder.encode(states, actions=rl.expand(table, states)))
    one = encoder.stream()
    one.append(states[3], actions=rl.expand(table, [states[3]]))
    assert_same(one.flush(), encoder.encode(states[3], actions=rl.expand(table, [states[3]])))
    mutable = encoder.mutable_stream()
    mutable.append([7, 2, 5], states[:3], actions=rl.expand(table, states[:3]))
    mutable.append(4, states[3], actions=rl.expand(table, [states[3]]))
    mutable.update([2], states[5:6], actions=rl.expand(table, states[5:6]))
    order = [states[5], states[3], states[2], states[0]]  # indices 2, 4, 5, 7
    mutable.remove(np.array([5], dtype=np.int32))
    assert mutable.indices().tolist() == [2, 4, 7]
    assert_same(mutable.flush(), encoder.encode([states[5], states[3], states[0]], actions=rl.expand(table, [states[5], states[3], states[0]])))
    assert order


def test_appended_arrays_may_be_reused_at_once():
    task = text_task("blocks__probBLOCKS-8-0")
    pool = Pool([task], seed=4)
    variant = ("pair", True, True)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    stream = encoder.stream()
    picks = list(range(0, 30))
    reference = encoder.encode(**pool.inputs(picks, variant))
    given = pool.inputs(picks, variant)
    owned = {k: (tuple(a.copy() for a in v) if isinstance(v, tuple) else v.copy()) for k, v in given.items() if k != "actions"}
    stream.append(actions=given["actions"], **owned)
    for value in owned.values():
        for array in value if isinstance(value, tuple) else (value,):
            array[...] = np.uint64(0xDEADBEEF) if array.dtype == U64 else 0
    assert_same(stream.flush(), reference)


def test_the_rows_of_a_stream_give_the_same_inputs_until_it_is_empty_again():
    pool = Pool([text_task("blocks__probBLOCKS-8-0")], seed=5)
    variant = ("pair", True, False)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    for stream in (encoder.stream(), encoder.mutable_stream()):
        counter = iter(range(1000))
        if isinstance(stream, encoding.FlatStream):
            def add(**kw):
                stream.append(**kw)
        else:
            def add(**kw):
                stream.append([next(counter) for _ in range(len(kw["states"]))], **kw)
        add(**pool.inputs([0, 1, 2], variant))
        for other in ((None, True, False), ("pair", False, False), (None, False, False)):
            with pytest.raises(ValueError, match="mymyr: encoding: the stream holds rows with"):
                add(**pool.inputs([3], other))
        assert len(stream) == 3
        stream.clear()
        assert len(stream) == 0
        add(**pool.inputs([3, 4], (None, False, False)))
        assert len(stream) == 2
        with pytest.raises(ValueError, match="the stream holds rows with"):
            add(**pool.inputs([3], variant))


def test_the_action_entities_of_a_flush_are_those_of_the_appended_states_in_graph_order():
    task = text_task("logistics00__probLOGISTICS-6-1")
    table = rl.as_table(task)
    states = walk(task, steps=6, walks=2)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",), goal_derivations=())
    stream = encoder.stream()
    chunks = [states[:3], states[3:4], states[4:]]
    for chunk in chunks:
        stream.append(chunk, actions=rl.expand(table, chunk))
    batch = stream.flush()
    graph = 0
    for chunk in chunks:
        for state in chunk:
            begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
            roles = batch.entity_role[begin:end]
            names = encoder.entity_names(batch, graph)
            assert sorted(n for n, r in zip(names, roles) if r == 4) == sorted(str(a) for a in state.applicable_actions())
            graph += 1


def test_invalid_rows_are_refused_at_the_append_and_change_nothing():
    pool = Pool([text_task("blocks__probBLOCKS-8-0")], seed=6)
    variant = ("pair", True, False)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    for stream in (encoder.stream(), encoder.mutable_stream()):
        mutable = isinstance(stream, encoding.FlatMutableStream)

        def add(picks, **changes):
            given = {**pool.inputs(picks, variant), **changes}
            return stream.append(list(range(len(picks))), **given) if mutable else stream.append(**given)

        add([0, 1])
        with pytest.raises(ValueError, match="mymyr: "):
            add([2, 3], task_ids=np.array([0, 5], np.int32))
        with pytest.raises(ValueError, match="mymyr: encoding: "):
            add([2, 3], states=pool.rows[[2, 3]][:, :0])
        with pytest.raises(ValueError, match="mymyr: encoding: "):
            add([2, 3], goals=(pool.pos[[2, 3]], pool.neg[[2]]))
        with pytest.raises(ValueError, match="mymyr: encoding: "):
            add([2, 3], actions=rl.expand(pool.table, pool.rows[[2]]))
        with pytest.raises(ValueError, match="mymyr: encoding: "):
            add([2, 3], derived=pool.derived[[2, 3]], goals=None)
        assert len(stream) == 2
        assert_same(stream.flush(), encoder.encode(**pool.inputs([0, 1], variant)))


def test_a_flush_that_the_encoder_refuses_keeps_the_rows():
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=())
    states = rl.expand(table, [task.initial_state]).succ
    spare = states.copy()
    slots = len(table.atom_metadata()["atom_pred"])
    assert slots < 64 * table.words
    spare[0, slots // 64] |= np.uint64(1) << np.uint64(slots % 64)  # a state bit the instance has no atom for
    stream = encoder.stream()
    stream.append(states[:2])
    stream.append(spare[:1])
    with pytest.raises(ValueError, match="mymyr: encoding: "):
        stream.flush()
    assert len(stream) == 3
    stream.clear()
    stream.append(states[:2])
    assert_same(stream.flush(), encoder.encode(states[:2]))
    # likewise a goal bit of no atom slot, in the mutable stream, where removing the row repairs the stream
    with_goals = encoding.FlatRelationEncoder(task)
    pos = np.zeros_like(states[:3])
    neg = np.zeros_like(states[:3])
    neg[2, slots // 64] = np.uint64(1) << np.uint64(slots % 64)
    mutable = with_goals.mutable_stream()
    mutable.append([4, 5, 6], states[:3], goals=(pos, neg))
    with pytest.raises(ValueError, match="mymyr: encoding: "):
        mutable.flush()
    assert len(mutable) == 3
    mutable.remove(6)
    assert_same(mutable.flush(), with_goals.encode(states[:2], goals=(pos[:2], neg[:2])))


def test_capacities_and_pools_of_a_flush():
    pool = Pool([text_task("gripper__prob05")], seed=8)
    variant = ("pair", True, False)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    picks = list(range(40))
    given = pool.inputs(picks, variant)
    exact = encoder.encode(**given)
    sizes = exact.as_dict()
    stream = encoder.stream()
    stream.append(**given)
    tight = {"entities": 1, "tuples": 1, "args": 1, "targets": 1}
    overflowed = stream.flush(capacities=tight)
    assert overflowed.sizes["overflow"] and overflowed.sizes["capacity_overflow"]
    assert len(stream) == 40
    roomy = {"entities": overflowed.sizes["entities"] + 9, "tuples": overflowed.sizes["tuples"] + 9,
             "args": overflowed.sizes["args"] + 9, "targets": overflowed.sizes["targets"] + 9}
    fitted = stream.flush(capacities=roomy, pool=rl.ThreadPool(2))
    assert not fitted.sizes["overflow"] and len(stream) == 0
    assert_same(fitted, encoder.encode(**given, capacities=roomy))
    assert sizes["num_graphs"] == 40
    mutable = encoder.mutable_stream()
    mutable.append(list(range(40)), **given)
    assert mutable.flush(capacities=tight).sizes["overflow"]
    assert len(mutable) == 40
    assert_same(mutable.flush(capacities=roomy), encoder.encode(**given, capacities=roomy))
    assert len(mutable) == 40
    with pytest.raises(ValueError, match="mymyr: encoding: capacities needs"):
        stream.flush(capacities={"entities": 1})


def test_a_stream_without_rows_flushes_to_an_empty_batch():
    pool = Pool([text_task("blocks__probBLOCKS-8-0")], seed=9)
    for variant in VARIANTS:
        encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
        for stream in (encoder.stream(), encoder.mutable_stream()):
            batch = stream.flush()
            assert batch.num_graphs == 0 and len(stream) == 0
            assert_same(batch, encoder.encode(**pool.inputs([], variant)))
    mutable = encoder.mutable_stream()
    assert mutable.indices().tolist() == []
    mutable.append([], **pool.inputs([], VARIANTS[-1]))
    mutable.remove([])
    assert len(mutable) == 0


def test_the_index_errors_of_a_mutable_stream_change_nothing():
    pool = Pool([text_task("blocks__probBLOCKS-8-0")], seed=10)
    variant = ("pair", False, False)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    stream = encoder.mutable_stream()
    stream.append([5, 3, 2**62], **pool.inputs([0, 1, 2], variant))
    rows = pool.inputs([3, 4, 5], variant)
    with pytest.raises(ValueError, match="in the stream already"):
        stream.append([3, 4, 6], **rows)
    with pytest.raises(ValueError, match="given twice"):
        stream.append([10, 11, 10], **rows)
    with pytest.raises(ValueError, match="2 indices for 3 rows"):
        stream.append([10, 11], **rows)
    with pytest.raises(KeyError, match="the index 4 is not in the stream"):
        stream.update([5, 4, 3], **rows)
    with pytest.raises(KeyError):
        stream.remove([3, 4])
    with pytest.raises(ValueError, match="given twice"):
        stream.update([5, 3, 5], **rows)
    with pytest.raises(ValueError, match="given twice"):
        stream.remove([5, 5])
    with pytest.raises(TypeError, match="indices"):
        stream.remove("5")
    with pytest.raises(TypeError, match="indices"):
        stream.remove(2.5)
    assert stream.indices().tolist() == [3, 5, 2**62]
    assert stream.indices().dtype == np.int64
    assert 5 in stream and 4 not in stream
    stream.update(np.array([5, 3], dtype=np.int32), **pool.inputs([6, 7], variant))
    stream.remove(2**62)
    assert_same(stream.flush(), encoder.encode(**pool.inputs([7, 6], variant)))


def test_concurrent_appends_and_flushes_lose_no_row():
    pool = Pool([text_task("gripper__prob05")], seed=12)
    variant = (None, True, False)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    stream = encoder.stream()
    mutable = encoder.mutable_stream()
    chunks = [list(range(i, i + 5)) for i in range(0, 40, 5)]

    def work(index):
        stream.append(**pool.inputs(chunks[index], variant))
        mutable.append([100 * index + k for k in range(5)], **pool.inputs(chunks[index], variant))
        return len(stream)

    with ThreadPoolExecutor(4) as executor:
        list(executor.map(work, range(len(chunks))))
    assert len(stream) == len(mutable) == 40
    batch = stream.flush()
    assert batch.num_graphs == 40
    assert_same(mutable.flush(), encoder.encode(**pool.inputs([p for c in chunks for p in c], variant)))


def test_the_framework_of_a_stream():
    pool = Pool([text_task("gripper__prob05")], seed=13)
    encoder = encoding.FlatRelationEncoder(pool.table, goal_derivations=())
    stream = encoder.stream(framework="numpy")
    stream.append(pool.rows[:3], pool.ids[:3])
    assert isinstance(stream.flush().relation_args, np.ndarray)
    with pytest.raises(ValueError):
        encoder.stream(framework="tensorflow")
    stream.append(pool.rows[:3], pool.ids[:3])
    with pytest.raises(ValueError):
        stream.flush(framework="tensorflow")
    assert stream.encoder is encoder
    assert encoder.mutable_stream().encoder is encoder


@pytest.mark.parametrize("mutable", [False, True], ids=["append-only", "mutable"])
@pytest.mark.parametrize("variant", [VARIANTS[1], VARIANTS[3]], ids=lambda v: "-".join(str(x) for x in v))
def test_a_flushed_batch_stays_valid_while_later_flushes_reuse_memory(mutable, variant):
    pool = Pool([text_task("blocks__probBLOCKS-8-0")], seed=17)
    encoder = encoding.FlatRelationEncoder(pool.table, **options_of(variant, "graph"))
    stream = encoder.mutable_stream() if mutable else encoder.stream()
    rng = random.Random(3)

    def add(picks):
        given = pool.inputs(picks, variant)
        stream.append(list(range(len(picks))), **given) if mutable else stream.append(**given)

    kept = []
    for round_ in range(12):
        picks = [rng.randrange(pool.size) for _ in range(rng.choice([1, 7, 40, 90]))]
        add(picks)
        batch = stream.flush(threads=1 + 2 * (round_ % 2))
        expected = encoder.encode(**pool.inputs(picks, variant))
        # every batch stays valid, and equal to its encode, after later flushes took and released memory
        kept.append((batch, expected, {k: np.array(v, copy=True) for k, v in batch.as_dict().items()}))
        if mutable:
            stream.clear()
        if round_ % 3 == 2:
            kept.pop(rng.randrange(len(kept)))
        for old, reference, snapshot in kept:
            assert_same(old, reference)
            for key, value in old.as_dict().items():
                assert np.array_equal(value, snapshot[key]), key
