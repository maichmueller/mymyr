"""Successor identities, bitwise no-op flags and stable filtering on host and JAX CPU."""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, text_task, walk

DOMAINS = ["blocksworld", "childsnack", "ferry", "floortile", "miconic", "rovers", "satellite", "sokoban", "spanner", "transport"]
DOMAIN = """(define (domain switches)
 (:requirements :strips :negative-preconditions)
 (:predicates (live) (p))
 (:action idle-a :parameters (?x) :precondition (live) :effect (and))
 (:action idle-b :parameters (?x) :precondition (live) :effect (and))
 (:action stop :parameters (?x) :precondition (and (live) (not (p))) :effect (not (live)))
 (:action set :parameters (?x) :precondition (and (live) (not (p))) :effect (p)))"""
PROBLEM = """(define (problem switches) (:domain switches) (:objects a) (:init (live)) (:goal (p)))"""


def pddl_task(domain=DOMAIN, problem=PROBLEM, **kwargs):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem), **kwargs)


def check_expansion(task, states):
    rows = np.asarray(task.encode(states))
    full = rl.expand(task, rows, hashes=True, noops=True, derived=True, goal=True)
    assert full.succ_hash.dtype == np.uint64 and full.noop.dtype == np.bool_
    assert full.index is None and full.generated == full.total
    np.testing.assert_array_equal(full.succ_hash, mymyr.hash_rows(full.succ, numeric_words=full.numeric_words))
    # Decode and re-encode to account for zero padding and newly interned lazy slots.
    padded = np.zeros((len(states), full.words + full.numeric_words), np.uint64)
    for i, state in enumerate(states):
        padded[i, :state.num_words] = state.words
        if full.numeric_words:
            padded[i, full.words:] = state.numeric_words
    np.testing.assert_array_equal(full.noop, np.all(full.succ == padded[full.parent], axis=1))
    expected = []
    indices = []
    for p, (begin, end) in enumerate(zip(full.offsets[:-1], full.offsets[1:])):
        keep = np.flatnonzero(~full.noop[begin:end])
        if not len(keep):
            keep = np.arange(end - begin)
        expected.extend(begin + keep)
        indices.extend(keep)
    expected = np.asarray(expected, np.int32)
    for kw in ({}, {"pool": rl.ThreadPool(3)}):
        kept = rl.expand(task, states, hashes=True, drop_noops=True, derived=True, goal=True, **kw)
        assert kept.generated == full.total and not kept.overflow
        np.testing.assert_array_equal(kept.index, indices)
        for name in ("succ", "parent", "schema", "binding", "goal", "derived", "succ_hash", "noop"):
            np.testing.assert_array_equal(getattr(kept, name), getattr(full, name)[expected])
        np.testing.assert_array_equal(kept.succ, full.succ[full.offsets[kept.parent] + kept.index])
        pad = kept.pad()
        assert not pad.overflow
        for p, count in enumerate(kept.counts):
            at = slice(kept.offsets[p], kept.offsets[p + 1])
            np.testing.assert_array_equal(pad.index[p, :count], kept.index[at])
            np.testing.assert_array_equal(pad.succ_hash[p, :count], kept.succ_hash[at])
            np.testing.assert_array_equal(pad.noop[p, :count], kept.noop[at])
        assert np.all(pad.index[~pad.mask] == -1)
        assert np.all(pad.succ_hash[~pad.mask] == 0) and not pad.noop[~pad.mask].any()
        h = np.zeros(full.total, np.uint64)
        n = np.zeros(full.total, np.bool_)
        index = np.full(full.total, -1, np.int32)
        offsets = np.zeros(len(states) + 1, np.int32)
        # The label-only path must digest the actual successor without storing rows.
        result = rl.expand_into(task, states, succ_hash=h, noop=n, index=index, offsets=offsets, drop_noops=True, **kw)
        assert result["generated"] == full.total and result["total"] == kept.total and not result["overflow"]
        np.testing.assert_array_equal(h[:kept.total], kept.succ_hash)
        np.testing.assert_array_equal(n[:kept.total], kept.noop)
        np.testing.assert_array_equal(index[:kept.total], kept.index)
        np.testing.assert_array_equal(offsets, kept.offsets)
    plain = rl.expand(task, states)
    assert plain.succ_hash is None and plain.noop is None and plain.index is None
    assert plain.pad().succ_hash is None and plain.pad().noop is None
    return full


@pytest.mark.parametrize("domain", DOMAINS)
def test_ipc_successor_digests_and_filtering(domain):
    base = ROOT / "tests/data/ipc" / f"{domain}-ipc" / "train"
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    task = mymyr.Task.from_pddl(base / "domain.pddl", base / "p01.pddl")
    check_expansion(task, walk(task, steps=12, walks=1, seed=9))


@pytest.mark.parametrize("name", ["openstacks-opt08-adl__p03", "philosophers__p03-phil4"])
def test_axiom_digests(name):
    task = text_task(name)
    check_expansion(task, walk(task, steps=4, walks=1))


@pytest.mark.parametrize("storage", ["i32", "f64"])
def test_numeric_digests(storage):
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"), numeric_storage=storage)
    check_expansion(task, walk(task, steps=8, walks=1))


def test_mixed_noops_all_noop_fallback_and_dead_end():
    task = pddl_task(atoms="dense")
    effective = next(s for _, s in task.successors(task.initial_state) if s.is_goal())
    dead = task.state(np.zeros(task.words, np.uint64))
    states = [task.initial_state, effective, dead]
    full = check_expansion(task, states)
    kept = rl.expand(task, states, drop_noops=True)
    assert kept.counts.tolist() == [2, 2, 0]
    assert kept.noop.tolist() == [False, False, True, True]
    # Enough room for the kept rows still overflows when the pre-drop rows do not fit.
    short = rl.expand(task, states, drop_noops=True, capacity=kept.total)
    assert short.generated == full.total > short.capacity and short.overflow and short.pad().overflow
    with pytest.raises(ValueError, match="requires index"):
        rl.expand_into(task, states, drop_noops=True, noop=np.zeros(full.total, bool))
    zero = rl.expand(task, states, drop_noops=True, capacity=0)
    assert zero.overflow and zero.generated == full.total
    empty = rl.expand(task, np.empty((0, task.words), np.uint64), drop_noops=True, hashes=True)
    assert empty.total == empty.generated == 0 and empty.offsets.tolist() == [0]


def test_lazy_slots_hash_and_new_atoms():
    domain = """(define (domain flags) (:requirements :strips)
      (:predicates (flag ?x)) (:action set :parameters (?x) :precondition (and) :effect (flag ?x)))"""
    problem = """(define (problem flags) (:domain flags)
      (:objects """ + " ".join(f"o{i}" for i in range(140)) + """ ) (:init) (:goal (and)))"""
    task = pddl_task(domain, problem, atoms="lazy")
    full = check_expansion(task, [task.initial_state])
    assert not full.noop.any() and full.words > 1
    short = rl.expand(task, [task.initial_state], hashes=True, noops=True, words=1)
    assert short.overflow and short.words_needed == full.words
    np.testing.assert_array_equal(short.succ_hash, full.succ_hash)


def test_numeric_noop_compares_signed_zero_and_nan_bits():
    domain = """(define (domain values) (:requirements :strips :fluents)
      (:predicates (p)) (:functions (x))
      (:action keep :parameters () :precondition (and) :effect (assign (x) (x)))
      (:action zero :parameters () :precondition (and) :effect (assign (x) 0)))"""
    problem = """(define (problem values) (:domain values) (:init (= (x) 0)) (:goal (and)))"""
    task = pddl_task(domain, problem, numeric_storage="f64")
    rows = np.tile(task.encode([task.initial_state]), (3, 1))
    rows[:, -1] = np.array([0, 0x8000000000000000, 0x7ff8000000001234], np.uint64)
    full = rl.expand(task, rows, hashes=True, noops=True)
    np.testing.assert_array_equal(full.noop, np.all(full.succ == rows[full.parent], axis=1))
    assert full.noop[:2].all() and not full.noop[2:].all()
    np.testing.assert_array_equal(full.succ_hash, mymyr.hash_rows(full.succ, numeric_words=task.numeric_words))


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("hashes,noops", [(True, False), (False, True), (True, True)])
def test_jax_cpu_matches_numpy(drop, hashes, noops):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("built without the XLA FFI headers")
    from mymyr.rl.jax import Env
    task = pddl_task(atoms="dense")
    rows = np.asarray(task.encode(walk(task, steps=4, walks=1)))
    expected = rl.expand(task, rows, hashes=hashes, noops=noops, drop_noops=drop)
    opts = dict(hashes=hashes, noops=noops, drop_noops=drop)
    env = Env(task, device="cpu")
    source = jnp.asarray(rows.view(np.uint32))
    actual = jax.jit(lambda s: env.expand_flat(s, expected.generated + 3, **opts))(source)
    m = expected.total
    assert int(actual.generated) == expected.generated
    for name in ("parent", "schema", "binding", "goal", "offsets"):
        want = getattr(expected, name) if name != "goal" else rl.expand(task, rows, goal=True, drop_noops=drop).goal
        got = np.asarray(getattr(actual, name))
        np.testing.assert_array_equal(got if name == "offsets" else got[:m], want)
    np.testing.assert_array_equal(np.asarray(actual.succ)[:m].view(np.uint64), expected.succ)
    if hashes:
        np.testing.assert_array_equal(np.asarray(actual.succ_hash)[:m].copy().view(np.uint64).ravel(), expected.succ_hash)
        assert not np.asarray(actual.succ_hash)[m:].any()
    if noops or drop:
        np.testing.assert_array_equal(np.asarray(actual.noop)[:m], expected.noop)
        assert not np.asarray(actual.noop)[m:].any()
    if drop:
        np.testing.assert_array_equal(np.asarray(actual.index)[:m], expected.index)
        assert np.all(np.asarray(actual.index)[m:] == -1)
    padded = jax.jit(lambda s: env.expand(s, 5, **opts))(source)
    ref = expected.pad(5)
    assert int(padded.generated) == expected.generated
    for name in ("succ", "schema", "binding", "mask", "count", "noop", "index", "succ_hash"):
        if name in ("noop", "index", "succ_hash") and getattr(padded, name) is None:
            continue
        got, want = np.asarray(getattr(padded, name)), getattr(ref, name)
        if name in ("succ", "succ_hash"):
            got = got.copy().view(np.uint64)
            if name == "succ_hash":
                got = got[..., 0]
        np.testing.assert_array_equal(got, want)
    limited = jax.jit(lambda s: env.expand_flat(s, 1, **opts))(source)
    assert int(limited.generated) > 1


def test_only_noop_action_keeps_all_rows():
    task = pddl_task("""(define (domain idle) (:requirements :strips)
        (:predicates (p)) (:action idle :parameters () :precondition (and) :effect (and)))""",
        """(define (problem idle) (:domain idle) (:init) (:goal (and)))""")
    exp = rl.expand(task, [task.initial_state] * 3, hashes=True, drop_noops=True)
    assert exp.generated == exp.total == 3
    assert exp.counts.tolist() == [1, 1, 1] and exp.noop.all() and not exp.index.any()


@pytest.mark.parametrize("canonical,witness", [(False, False), (True, True)])
def test_indices_follow_the_requested_generator_order(canonical, witness):
    task = text_task("rovers__p02")
    states = walk(task, steps=7, walks=1)
    opts = dict(canonical_order=canonical, witness_pruning=witness)
    full = rl.expand(task, states, noops=True, **opts)
    kept = rl.expand(task, states, drop_noops=True, **opts)
    np.testing.assert_array_equal(kept.succ, full.succ[full.offsets[kept.parent] + kept.index])
    np.testing.assert_array_equal(kept.schema, full.schema[full.offsets[kept.parent] + kept.index])
    ix = np.zeros(full.total, np.int32)
    result = rl.expand_into(task, states, index=ix, **opts)
    assert result["total"] == full.total
    np.testing.assert_array_equal(ix, np.concatenate([np.arange(c) for c in full.counts]))
    for pool in (None, rl.ThreadPool(3)):
        result = rl.expand_into(task, states, index=ix[:kept.total], drop_noops=True, pool=pool, **opts)
        assert result["overflow"] and result["generated"] == full.total


@pytest.mark.parametrize("name", ["cs-counters", "philosophers__p03-phil4"])
def test_jax_cpu_numeric_and_derived_digests(name):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("built without the XLA FFI headers")
    from mymyr.rl.jax import Env
    task = (mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
            if name == "cs-counters" else text_task(name))
    states = walk(task, steps=3, walks=1)
    expected = rl.expand(task, states, hashes=True, noops=True, drop_noops=True, derived=True, goal=True)
    env = Env(task, device="cpu")
    source = jnp.asarray(np.asarray(task.encode(states)).view(np.uint32))
    actual = jax.jit(lambda s: env.expand_flat(s, expected.generated + 2, hashes=True, noops=True,
                                             drop_noops=True, derived=True))(source)
    m = expected.total
    assert int(actual.generated) == expected.generated
    np.testing.assert_array_equal(np.asarray(actual.succ)[:m].copy().view(np.uint64), expected.succ)
    np.testing.assert_array_equal(np.asarray(actual.derived)[:m].copy().view(np.uint64), expected.derived)
    np.testing.assert_array_equal(np.asarray(actual.succ_hash)[:m].copy().view(np.uint64).ravel(), expected.succ_hash)
    np.testing.assert_array_equal(np.asarray(actual.noop)[:m], expected.noop)
    np.testing.assert_array_equal(np.asarray(actual.index)[:m], expected.index)
