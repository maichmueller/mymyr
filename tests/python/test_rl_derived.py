"""rl.expand(..., derived=True): the derived words of the successors (one axiom evaluation of each successor row).

Checked against the meaning of the words (a small domain with one derived predicate), against the expansion without
them (every other output is unchanged), across the call forms (capacity, threads, pools, expand_into, the padded view),
tasks without derived atoms (zero-width words), and the torch ops and JAX FFI variants.
"""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import SMALL_TASKS, text_task

DOMAIN = """(define (domain features)
 (:requirements :strips :negative-preconditions :derived-predicates)
 (:predicates (flag ?x) (available ?x))
 (:derived (available ?x) (flag ?x))
 (:action enable :parameters (?x) :precondition (not (flag ?x)) :effect (flag ?x))
 (:action disable :parameters (?x) :precondition (flag ?x) :effect (not (flag ?x))))"""

PROBLEM = """(define (problem p) (:domain features) (:objects a b c)
  (:init (flag a)) (:goal (available c)))"""


def feature_task():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task(mymyr.Domain.from_string(DOMAIN).instantiate_string(PROBLEM), atoms="dense")


def walk_rows(task, n=24, seed=0):
    rng = np.random.default_rng(seed)
    states = [task.initial_state]
    for _ in range(n):
        succ = task.successor_states(states[-1])
        if not succ:
            states.append(task.initial_state)
            continue
        states.append(succ[int(rng.integers(len(succ)))])
    return states


def test_words_are_the_successors_derived_atoms():
    task = feature_task()
    meta = task.atom_metadata()
    flag_slot = {}
    for slot in range(meta["num_atoms"]):
        pred = int(meta["atom_pred"][slot])
        if meta["pred_arity"][pred] == 1:
            flag_slot[int(meta["atom_args_padded"][slot][0])] = slot
    states = walk_rows(task)
    x = rl.expand(task, states, derived=True)
    assert x.derived.dtype == np.uint64 and x.derived.shape == (x.total, task.derived_words if hasattr(task, "derived_words") else 1)
    dpred = meta["derived_pred"]
    dargs = meta["derived_args_padded"]
    assert len(dpred) == 3
    for j in range(x.total):
        row = x.succ[j]
        for d in range(len(dpred)):
            obj = int(dargs[d][0])
            fluent = (int(row[flag_slot[obj] >> 6]) >> (flag_slot[obj] & 63)) & 1
            derived = (int(x.derived[j][d >> 6]) >> (d & 63)) & 1
            assert fluent == derived


@pytest.mark.parametrize("name", ["philosophers__p03-phil4", "openstacks-opt08-adl__p03", "blocks__probBLOCKS-8-0"])
def test_other_outputs_do_not_change(name):
    task = text_task(name)
    states = walk_rows(task)
    plain = rl.expand(task, states, goal=True)
    withd = rl.expand(task, states, goal=True, derived=True)
    assert plain.derived is None
    for field in ("succ", "parent", "schema", "binding", "offsets", "goal"):
        assert np.array_equal(getattr(plain, field), getattr(withd, field)), field
    assert withd.derived.shape[0] == withd.total


def test_zero_width_without_derived_atoms():
    task = text_task("blocks__probBLOCKS-8-0")
    x = rl.expand(task, task.initial_state, derived=True)
    assert x.derived.shape == (x.total, 0) and x.derived_words == 0


@pytest.mark.parametrize("name", ["philosophers__p03-phil4"])
def test_capacity_pool_and_threads(name):
    task = text_task(name)
    states = walk_rows(task)
    full = rl.expand(task, states, derived=True)
    assert full.derived.shape[1] > 0
    part = rl.expand(task, states, derived=True, capacity=full.total // 2)
    assert part.overflow and part.total == full.total
    assert np.array_equal(part.derived, full.derived[: full.total // 2])
    for kw in ({"threads": 4}, {"pool": rl.ThreadPool(3)}):
        other = rl.expand(task, states, derived=True, **kw)
        assert np.array_equal(other.derived, full.derived)
    # padded view
    p = full.pad(16)
    idx = p.index
    for i in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            want = full.derived[idx[i, k]] if idx[i, k] >= 0 else np.zeros_like(full.derived[0])
            assert np.array_equal(p.derived[i, k], want)
    assert rl.expand(task, states, K=8).padded.derived is None
    assert rl.expand(task, states, K=8, derived=True).padded.derived.shape[:2] == (len(states), 8)


def test_expand_into_destination():
    task = text_task("philosophers__p03-phil4")
    states = walk_rows(task)
    full = rl.expand(task, states, derived=True, goal=True)
    cap, D, W = full.total + 3, full.derived_words, full.words
    succ = np.zeros((cap, W), np.uint64)
    derived = np.full((cap, D), 7, np.uint64)
    r = rl.expand_into(task, states, succ=succ, derived=derived)
    assert r["total"] == full.total
    assert np.array_equal(derived[: full.total], full.derived)
    assert (derived[full.total :] == 7).all()  # rows past the total are not written
    d32 = np.zeros((cap, 2 * D), np.uint32)
    rl.expand_into(task, states, succ=succ, derived=d32)
    assert np.array_equal(d32[: full.total].view(np.uint64), full.derived)
    with pytest.raises(ValueError):
        rl.expand_into(task, states, derived=derived)  # needs the successor rows
    with pytest.raises(ValueError):
        rl.expand_into(task, states, succ=succ, derived=np.zeros((cap, D + 1 - 2), np.uint64) if D > 1 else np.zeros((cap, 0), np.uint64))


def test_suite_width_is_the_widest_domain():
    a, b = text_task("philosophers__p03-phil4"), text_task("blocks__probBLOCKS-8-0")
    suite = rl.TaskSuite.group([a, b])
    D = rl.TaskTable([a]).derived_words
    assert D > 0 and suite.derived_words == D
    ids = [0, 1, 0]
    x = rl.expand(suite, [a.initial_state, b.initial_state, a.initial_state], ids, derived=True)
    assert x.derived.shape == (x.total, D)
    alone = rl.expand(a, a.initial_state, derived=True)
    n = alone.total
    assert np.array_equal(x.derived[:n], alone.derived)
    nb = int(x.offsets[2] - x.offsets[1])
    assert not x.derived[n : n + nb].any()  # blocks has no derived atoms: zero words


@pytest.mark.needs_gil
def test_torch_ops_equal_rl_expand():
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    task = text_task("philosophers__p03-phil4", atoms="dense")
    env = rt.BatchedEnv(task, 8)
    for _ in range(4):
        env.step()
    h = rt.register(task)
    s = env.states.clone()
    x = rl.expand(task, s, words=s.shape[1], goal=True, derived=True)
    flat = torch.ops.mymyr.expand_flat_derived(h, s, None, None, False)
    assert torch.equal(flat[6], x.derived) and torch.equal(flat[0], x.succ)
    capped = torch.ops.mymyr.expand_flat_derived(h, s, None, x.total + 4, False)
    assert capped[6].shape == (x.total + 4, x.derived_words) and torch.equal(capped[6][: x.total], x.derived)
    assert (capped[6][x.total :] == 0).all()
    K = int((x.offsets[1:] - x.offsets[:-1]).max())
    padded = torch.ops.mymyr.expand_derived(h, s, None, K, False)
    ref = x.pad(K)
    assert torch.equal(padded[6], ref.derived) and torch.equal(padded[0], ref.succ)
    # the fake implementations give the real shapes
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_non_fake_inputs=True):
        fake = torch.ops.mymyr.expand_flat_derived(h, s, None, x.total + 4, False)
        assert fake[6].shape == capped[6].shape
        fake = torch.ops.mymyr.expand_derived(h, s, None, K, False)
        assert fake[6].shape == padded[6].shape


def test_jax_expansions_carry_the_derived_words():
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax

    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without the XLA FFI headers")
    import mymyr.rl.jax as rj

    task = text_task("philosophers__p03-phil4")
    env = rj.Env(task)
    state, _ = env.reset(None, 16)
    key = jax.random.key(1)
    for _ in range(4):
        state = env.step(key, state)[0]
    rows = np.asarray(state.states)
    rows64 = rows.view(np.uint64)
    x = rl.expand(task, rows64, words=env.row_words, goal=True, derived=True)
    flat = env.expand_flat(state.states, int(x.total) + 3, derived=True)
    d = np.asarray(flat.derived)
    assert d.shape == (x.total + 3, 2 * x.derived_words)
    assert np.array_equal(d[: x.total].view(np.uint64), x.derived)
    assert (d[x.total :] == 0).all()
    assert env.expand_flat(state.states, int(x.total) + 3).derived is None
    p = x.pad(16)
    pad = env.expand(state.states, 16, derived=True)
    assert np.array_equal(np.asarray(pad.derived).view(np.uint64).reshape(p.derived.shape), p.derived)
