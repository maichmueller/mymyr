"""Collapsed action parameters in the environments and encodings: the host, torch, JAX and the device list the same
(shrunk) action sets, masks and action relations.

The device parts need a GPU, e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_unused_parameters_rl.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="needs the PDDL front end")

# move binds ?from and ?to; ?by (a vehicle) and ?w (a weather) are mentioned by nothing but their types
DOMAIN = """(define (domain roads) (:requirements :typing)
  (:types place vehicle weather)
  (:predicates (at ?p - place) (road ?a - place ?b - place) (seen ?p - place))
  (:action move :parameters (?from - place ?by - vehicle ?to - place ?w - weather)
    :precondition (and (at ?from) (road ?from ?to)) :effect (and (at ?to) (seen ?to) (not (at ?from))))
  (:action look :parameters (?p - place ?by - vehicle) :precondition (at ?p) :effect (seen ?p)))"""
PROBLEM = """(define (problem p) (:domain roads)
  (:objects a b c d - place v1 v2 v3 - vehicle sun rain - weather)
  (:init (at a) (road a b) (road b c) (road c d) (road b a) (road a c))
  (:goal (seen d)))"""


def task(collapse=True):
    return mymyr.Task(mymyr.Domain.from_string(DOMAIN).instantiate_string(PROBLEM, collapse_unused_parameters=collapse),
                      atoms="dense")


def rows_of(t):
    states = [t.initial_state]
    for _ in range(3):
        states.append(t.apply(states[-1], t.applicable_actions(states[-1])[0]))
    return np.asarray(t.encode(states))


def test_host_action_sets_and_masks_shrink():
    collapsed, full = task(), task(False)
    rows_c, rows_f = rows_of(collapsed), rows_of(full)
    assert np.array_equal(rows_c, rows_f)  # the same states: the first action is the representative's
    xc, xf = rl.expand(collapsed, rows_c), rl.expand(full, rows_f)
    # per state: move has 3 * 2 = 6 copies, look 3
    for i in range(len(rows_c)):
        sc = np.asarray(xc.schema)[xc.offsets[i]:xc.offsets[i + 1]]
        sf = np.asarray(xf.schema)[xf.offsets[i]:xf.offsets[i + 1]]
        assert np.count_nonzero(sf == 0) == 6 * np.count_nonzero(sc == 0)
        assert np.count_nonzero(sf == 1) == 3 * np.count_nonzero(sc == 1)
    assert np.asarray(xc.binding).shape[1] == 2 and np.asarray(xf.binding).shape[1] == 4
    # the same successor sets
    assert {bytes(r) for r in np.asarray(xc.succ)} == {bytes(r) for r in np.asarray(xf.succ)}
    assert np.array_equal(rl.applicable_schemas(collapsed, rows_c), rl.applicable_schemas(full, rows_f))
    objects = rl.applicable_objects(collapsed, rows_c, schema=0, prefix=None, depth=0)
    assert np.array_equal(objects, rl.applicable_objects(full, rows_f, schema=0, prefix=None, depth=0))
    # ?to of move: depth 1 collapsed (after ?from), depth 2 in full (after ?from and ?by)
    n = len(rows_c)
    to = rl.applicable_objects(collapsed, rows_c, schema=0, prefix=np.zeros((n, 1), np.int32), depth=1)
    by = full.formalism.objects.index(next(o for o in full.formalism.objects if o.name == "v2"))
    prefix = np.tile(np.array([[0, by]], np.int32), (n, 1))
    assert np.array_equal(to, rl.applicable_objects(full, rows_f, schema=0, prefix=prefix, depth=2))
    assert to[0].sum() == 2  # from a: b and c


def test_jax_env_lists_the_host_actions():
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without the XLA FFI headers")
    import mymyr.rl.jax as rj
    devices = ["cpu"] + (["cuda"] if any(d.platform == "gpu" for d in jax.devices()) else [])
    t = task()
    for device in devices:
        env = rj.Env(t, device=device)
        state, _ = env.reset(None, 4)
        x = rl.expand(t, np.asarray(state.states).view(np.uint64), words=env.row_words, goal=True)
        flat = env.expand_flat(state.states, int(x.total) + 3)
        assert np.array_equal(np.asarray(flat.offsets), x.offsets)
        assert np.array_equal(np.asarray(flat.schema)[: x.total], x.schema)
        assert np.array_equal(np.asarray(flat.binding)[: x.total], x.binding)
        assert int(x.offsets[1]) == 4  # move a b, move a c, look a, ... with one action each


def gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device")
    pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    return torch, torch.device("cuda", 0)


def tensor(torch, a, device):
    a = np.asarray(a)
    return torch.from_numpy(a.view(np.int64).copy()).to(device)


@pytest.mark.parametrize("collapse", [True, False])
def test_device_actions_masks_and_encodings_equal_the_host(collapse):
    torch, device = gpu()
    t = task(collapse)
    rows = rows_of(t)
    words = tensor(torch, rows, device)
    host = rl.expand(t, rows)
    dev = rl.expand(t, words, framework="torch")
    assert np.array_equal(dev.offsets.cpu().numpy(), host.offsets)
    assert np.array_equal(dev.schema.cpu().numpy(), host.schema)
    assert np.array_equal(dev.binding.cpu().numpy(), host.binding)
    assert np.array_equal(rl.applicable_schemas(t, words).cpu().numpy(), rl.applicable_schemas(t, rows))
    assert np.array_equal(rl.applicable_objects(t, words, schema=0, prefix=None, depth=0).cpu().numpy(),
                          rl.applicable_objects(t, rows, schema=0, prefix=None, depth=0))
    encoder = encoding.FlatRelationEncoder(t, target_sources=("action",))
    expected = encoder.encode(rows, actions=host)
    actual = encoder.encode(words, actions=dev)
    e, a = expected.as_dict(), actual.as_dict("numpy")
    assert e.keys() == a.keys()
    for name in e:
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(e[name]), err_msg=name)
    # the action relation of move holds (entity, from, to) collapsed and (entity, from, by, to, w) in full
    move = encoder.relation_names.index("move")
    assert encoder.relation_arities[move] == (3 if collapse else 5)
