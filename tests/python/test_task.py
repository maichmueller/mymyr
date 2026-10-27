"""mymyr.Task, TaskHandle, State, Action, Atom: construction, per-state API, labels (labels round-trip to
the applicable ground actions), value semantics, and pickling."""

import itertools
import pickle

import numpy as np
import pytest

import mymyr
from conftest import BLOCKS, ROOT, SMALL_TASKS, TASKS, gil_stays_off, text_task, walk

needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


# ------------------------------------------------------------------------------------------------ construction


@needs_frontend
def test_construction_paths_agree(blocks):
    dom = mymyr.Domain.from_file(BLOCKS / "domain.pddl")
    a = mymyr.Task(dom.instantiate(BLOCKS / "probBLOCKS-8-0.pddl"))
    b = mymyr.Task.from_pddl(dom, BLOCKS / "probBLOCKS-8-0.pddl")
    c = mymyr.Task(mymyr.Domain.from_string((BLOCKS / "domain.pddl").read_text()).instantiate_string(
        (BLOCKS / "probBLOCKS-8-0.pddl").read_text()))
    assert a.fingerprint == b.fingerprint == c.fingerprint == blocks.fingerprint
    assert len({a.uid, b.uid, c.uid, blocks.uid}) == 4  # uids are per instance
    assert blocks.domain_name == "blocks" and blocks.num_objects == 8 and blocks.num_schemas == 4
    assert blocks.objects == ["h", "g", "f", "e", "c", "b", "d", "a"]
    assert set(blocks.schemas) == {"pick-up", "put-down", "stack", "unstack"}
    assert isinstance(blocks.formalism, mymyr.formalism.NormalizedTask)
    assert blocks.atom_mode == "dense" and blocks.words == 2
    assert "blocks" in repr(blocks)


def test_options_and_text_tasks():
    lazy = text_task("gripper__prob05", atoms="lazy", matching="fc")
    dense = text_task("gripper__prob05", atoms="dense")
    assert lazy.atom_mode == "lazy" and dense.atom_mode == "dense"
    assert lazy.options["matching"] == "fc" and dense.options["atoms"] == "dense"
    assert lazy.fingerprint == dense.fingerprint  # the content, not the options
    assert text_task("gripper__prob05").options == {
        "atoms": "auto", "matching": "auto", "fc_free_params": 4, "numeric_storage": "auto", "numeric_quantum": 0.0,
        "numeric_tolerant": False, "reachability_budget": 1 << 25}
    for bad in ({"atoms": "sometimes"}, {"atoms": "frozen"}, {"numeric_storage": "f32"}, {"numeric_quantum": -1.0},
                {"numeric_quantum": float("nan")}, {"reachability_budget": -1}):
        with pytest.raises((ValueError, OverflowError)):
            text_task("gripper__prob05", **bad)
    numeric = mymyr.Task.from_text(str(sorted((ROOT / "tests/data/numeric_tasks").glob("*.txt"))[0]))
    assert numeric.numeric_slots > 0 and gripper_is_classical(dense)


def gripper_is_classical(task):
    return task.numeric_slots == 0 and task.numeric_words == 0 and task.numeric_storage is None


# ------------------------------------------------------------------------------------------------ per-state API


def test_initial_state_and_apply(blocks):
    s = blocks.initial_state
    assert isinstance(s, mymyr.State) and len(s) == 13
    assert s.words.dtype == np.uint64 and not s.words.flags.writeable
    acts = blocks.applicable_actions(s)
    assert [str(a) for a in acts] == ["(pick-up c)", "(pick-up b)", "(unstack d h)", "(unstack a g)"]
    t = blocks.apply(s, acts[0])
    assert t.holds("(holding c)") and not t.holds("(ontable c)") and not t.holds(("handempty",))
    assert blocks.apply(s, "(pick-up c)") == t == blocks.apply(s, ("pick-up", ["c"])) == acts[0].apply(s)
    assert s.apply(acts[0]) == t and t in {x for x in blocks.successor_states(s)}
    pairs = blocks.successors(s)
    assert [a for a, _ in pairs] == acts and [x for _, x in pairs] == blocks.successor_states(s)
    with pytest.raises(ValueError):
        blocks.apply(s, "(stack c b)")  # not applicable
    with pytest.raises(KeyError):
        blocks.action("(fly c)")
    with pytest.raises(ValueError):
        blocks.action("stack", ["c"])  # arity


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_labels_round_trip_to_the_applicable_ground_actions(name):
    """Every applicable action is a label (schema, binding) -> names -> the same action, and the labels are
    exactly the ground actions that are applicable (an exhaustive is_applicable sweep over all bindings)."""
    task = text_task(name)
    n = task.num_objects
    schema_arity = [task.formalism.schemas[i].arity for i in range(task.num_schemas)]
    groundings = sum(n ** k for k in schema_arity)
    for s in walk(task, steps=6, seed=1, walks=2):
        acts = task.applicable_actions(s)
        labels = [a.label for a in acts]
        assert labels == sorted(labels), "canonical order: schema, then binding"
        for a in acts:
            schema, binding = a.label
            assert task.action(schema, binding) == a
            assert task.action(a.name, a.objects) == a
            assert task.action(str(a)) == a
            assert pickle.loads(pickle.dumps(a)).label == a.label
            assert task.is_applicable(s, a)
        if groundings <= 200_000:
            brute = [(i, b) for i in range(task.num_schemas) for b in itertools.product(range(n), repeat=schema_arity[i])
                     if task.is_applicable(s, (i, b))]
            assert [(i, tuple(b)) for i, b in brute] == labels
        # successors agree with apply
        for a, t in task.successors(s):
            assert task.apply(s, a) == t


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_goal_equals_its_literals(name):
    """is_goal agrees with holds() on each goal literal (fluent, static and derived atoms)."""
    task = text_task(name)
    goal = task.formalism.goal.literals
    for s in walk(task, steps=15, seed=3, walks=3):
        expected = all(s.holds((lit.predicate.name, *[str(o) for o in lit.objects])) == lit.positive for lit in goal)
        assert s.is_goal() == task.is_goal(s) == expected


# ------------------------------------------------------------------------------------------------ values


def test_states_are_values(blocks):
    h = blocks.local()
    assert isinstance(h, mymyr.TaskHandle) and h.task is blocks
    s, t = blocks.initial_state, h.initial_state
    assert s == t and hash(s) == hash(t) and len({s, t}) == 1
    assert s.owner is blocks and t.owner is h and t.task is blocks
    other = mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl")
    assert other.initial_state != s  # task uid + content
    assert other.initial_state.canonical_hash() == s.canonical_hash()
    assert blocks.state(s.words) == s == blocks.state(s.atoms()) == blocks.state([str(x) for x in s.atoms()])
    assert blocks.decode(blocks.encode([s, t])) == [s, t]
    assert [a.slot for a in s.atoms()] == s.atom_slots()
    atom = blocks.atom("(on d h)")
    assert atom.kind == "fluent" and atom.slot in s.atom_slots() and s.holds(atom) and s.holds(atom.slot)
    assert blocks.atom(atom.slot) == atom and str(atom) == "(on d h)" and [str(o) for o in atom.objects] == ["d", "h"]
    with pytest.raises(ValueError):
        other.applicable_actions(s)  # a state of another task
    assert blocks.num_atoms % 64 and blocks.words == 2
    with pytest.raises(ValueError):
        blocks.state(np.array([0, 1 << 60], dtype=np.uint64))  # an unassigned slot
    row = blocks.encode(s)[0]
    assert blocks.state(np.array([*row, ~np.uint64(0)], dtype=np.uint64)) == s  # words past the task's are not read


def test_static_and_derived_atoms():
    task = text_task("philosophers__p03-phil4")
    kinds = {p.name: p.kind for p in task.formalism.predicates}
    derived = [n for n, k in kinds.items() if k == "derived"]
    assert derived
    s = task.initial_state
    static = task.formalism.static_init
    for a in static[:10]:
        assert s.holds((a.predicate.name, *[str(o) for o in a.objects]))


# ------------------------------------------------------------------------------------------------ pickling


def test_pickle_task_and_states(blocks):
    for task in (blocks, text_task("gripper__prob05", atoms="lazy"), text_task("openstacks-opt08-adl__p03")):
        states = walk(task, steps=10, seed=2)
        acts = task.applicable_actions(states[-1])
        t2, s2, a2, h2 = pickle.loads(pickle.dumps((task, states, acts, task.local())))
        assert t2.fingerprint == task.fingerprint and t2.uid != task.uid
        assert t2.options == task.options
        assert all(x.task is t2 for x in s2) and h2.task is t2
        assert [x.canonical_hash() for x in s2] == [x.canonical_hash() for x in states]
        # slot order (and so the printing order) may differ under lazy slots; the atoms are the same
        assert [sorted(map(str, x.atoms())) for x in s2] == [sorted(map(str, x.atoms())) for x in states]
        assert [a.label for a in a2] == [a.label for a in acts]
        assert t2.applicable_actions(s2[-1]) == a2


def test_pickled_state_into_a_lazy_task_uses_canonical_ids():
    lazy = text_task("logistics00__probLOGISTICS-6-1", atoms="lazy")
    states = walk(lazy, steps=25, seed=5)
    s2 = pickle.loads(pickle.dumps(states))
    assert [x.canonical_hash() for x in s2] == [x.canonical_hash() for x in states]


def test_fingerprint_mismatch_is_detected(blocks):
    from mymyr import _pickle

    s = blocks.initial_state
    reducer, args = s.__reduce__()
    assert reducer is _pickle._restore_state
    with pytest.raises(ValueError, match="fingerprint"):
        reducer(blocks, args[1] ^ 1, *args[2:])
    kind, a, b, ap, bp, opts, fp, enc = blocks.__reduce__()[1]
    with pytest.raises(ValueError, match="fingerprint"):
        _pickle._restore_task(kind, a, b.replace("(on a g)", "(on g a)"), ap, bp, opts, fp, enc)
    with pytest.raises(ValueError, match="encoding"):
        _pickle._restore_task(kind, a, b, ap, bp, opts, fp, (enc[0] ^ 1, enc[1]))


def test_pickles_keep_the_encoding():
    for atoms in ("auto", "reachable", "dense", "lazy"):
        task = text_task("logistics00__probLOGISTICS-6-1", atoms=atoms)
        states = walk(task, steps=25, seed=5)
        t2, s2 = pickle.loads(pickle.dumps((task, states)))
        assert t2.atom_mode == task.atom_mode and t2.encoding_fingerprint == task.encoding_fingerprint
        assert t2.num_atoms == task.num_atoms and t2.words == task.words
        assert np.array_equal(t2.encode(s2), task.encode(states))  # lazy pickles carry their slot table
        if atoms == "lazy":
            with pytest.raises(ValueError, match="lazy atom slots"):
                t2.decode(task.encode(states))
        else:
            assert t2.decode(task.encode(states)) == s2


def test_gil_stays_disabled():
    assert gil_stays_off()


@pytest.mark.parametrize("mode", ["auto", "reachable", "dense"])
def test_frozen_task_cache_and_pickle(mode):
    lazy = text_task("gripper__prob05", atoms="lazy", matching="fc", reachability_budget=1 << 24)
    frozen = lazy.frozen(mode)
    assert frozen.atom_mode != "lazy"
    assert frozen is lazy.frozen(mode)
    assert frozen is lazy.local().frozen(mode)
    assert frozen.frozen(mode) is frozen
    assert frozen.local().frozen(mode) is frozen
    assert frozen.fingerprint == lazy.fingerprint
    assert frozen.options == {**lazy.options, "atoms": mode}
    source, twin = pickle.loads(pickle.dumps((lazy, frozen)))
    assert source.frozen(mode) is twin
    assert twin.encoding_fingerprint == frozen.encoding_fingerprint
    with pytest.raises(ValueError, match="frozen atoms"):
        lazy.frozen("lazy")
    with pytest.raises(ValueError, match="frozen atoms"):
        lazy.frozen("other")


def test_frozen_cache_is_shared_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    lazy = text_task("gripper__prob05", atoms="lazy")
    with ThreadPoolExecutor(4) as pool:
        tasks = list(pool.map(lambda _: lazy.frozen(), range(16)))
    assert len({t.uid for t in tasks}) == 1
    assert all(t is tasks[0] for t in tasks)


def test_frozen_wrapper_can_be_collected_and_recreated():
    import gc
    import weakref

    lazy = text_task("gripper__prob05", atoms="lazy")
    twin = lazy.frozen()
    uid = twin.uid
    reference = weakref.ref(twin)
    del twin
    gc.collect()
    assert reference() is None
    twin = lazy.local().frozen()
    assert twin.uid == uid
    assert twin is lazy.frozen()
    source_reference = weakref.ref(lazy)
    del lazy
    gc.collect()
    assert source_reference() is not None
    del twin
    gc.collect()
    assert source_reference() is None


def test_frozen_cache_with_concurrent_collection():
    import gc
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lazy = text_task("gripper__prob05", atoms="lazy")
    barrier = threading.Barrier(9)
    stop = threading.Event()

    def freeze(worker):
        local = lazy.local()
        mode = ("auto", "reachable", "dense")[worker % 3]
        barrier.wait()
        for _ in range(100):
            twin = local.frozen(mode)
            assert twin is lazy.frozen(mode)
            del twin

    def collect():
        barrier.wait()
        while not stop.is_set():
            gc.collect()

    with ThreadPoolExecutor(9) as pool:
        collecting = pool.submit(collect)
        pending = [pool.submit(freeze, worker) for worker in range(8)]
        try:
            for result in pending:
                result.result()
        finally:
            stop.set()
            collecting.result()


def test_frozen_uses_the_source_reachability_budget():
    lazy = text_task("gripper__prob05", atoms="lazy", reachability_budget=1)
    assert lazy.frozen().atom_mode == "dense"
    with pytest.raises((ValueError, RuntimeError), match="reachability_budget"):
        lazy.frozen("reachable")
