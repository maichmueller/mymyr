"""A table whose action schemas are all nullary (here: every parameter is unused and collapses) has action labels of
zero binding columns everywhere: rl.expand, the CPU env pool, the torch BatchedEnv on the host and the device (index
and label actions, captured steps), the JAX env, the action relations of the encoders, plan rows and state space
files. Host, device and JAX agree.

The device parts need a GPU, e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_nullary_actions.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import datasets, encoding, rl

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="needs the PDDL front end")

# press and finish mention their parameters only by type: both collapse to nullary schemas
DOMAIN = """(define (domain switches) (:requirements :typing :negative-preconditions)
  (:types light hand)
  (:predicates (on) (done))
  (:action press :parameters (?h - hand) :precondition (not (on)) :effect (on))
  (:action finish :parameters (?h - hand ?l - light) :precondition (and (on) (not (done)))
    :effect (and (done) (not (on)))))"""
PROBLEMS = ["(:objects l1 l2 - light h1 h2 - hand)", "(:objects l1 - light h1 h2 h3 - hand)"]


def tasks():
    domain = mymyr.Domain.from_string(DOMAIN)
    return [mymyr.Task(domain.instantiate_string(f"(define (problem p{i}) (:domain switches) {objects} (:init) "
                                                 "(:goal (done)))"), atoms="dense")
            for i, objects in enumerate(PROBLEMS)]


def walk(task):
    """The states of the one path: nothing, on, done."""
    states = [task.initial_state]
    for _ in range(2):
        states.append(task.apply(states[-1], task.applicable_actions(states[-1])[0]))
    return states


def rows_and_ids():
    ts = tasks()
    table = rl.TaskTable(ts)
    rows = np.ascontiguousarray(np.concatenate([np.asarray(t.encode(walk(t))) for t in ts]))  # one word each
    ids = np.repeat(np.arange(len(ts), dtype=np.int32), 3)
    return ts, table, rows, ids


def test_the_schemas_collapse_and_the_table_has_no_binding_columns():
    ts, table, rows, ids = rows_and_ids()
    assert [s.arity for s in ts[0].formalism.schemas] == [0, 0]
    assert table.label_width == 0 and rl.TaskSuite([table]).label_width == 0
    x = rl.expand(table, rows, ids, goal=True)
    assert np.asarray(x.binding).shape == (x.total, 0) and x.label_width == 0
    assert np.asarray(x.offsets).tolist() == [0, 1, 2, 3, 4, 5, 6]  # press, finish, press per instance
    names = [ts[0].formalism.schemas[s].name for s in np.asarray(x.schema)]
    assert names == ["press", "finish", "press"] * 2


def test_the_cpu_pool_steps_with_empty_bindings():
    _, table, _, _ = rows_and_ids()
    pool = rl.CpuEnvPool(table, num_envs=4, seed=3, max_steps=4)
    assert pool.label_width == 0
    pool.reset()
    for _ in range(3):
        step = pool.step()
        assert np.asarray(step.binding).shape == (4, 0) and np.asarray(step.schema).shape == (4,)


def compare(index, label):
    """One index step (the random policy) and the label step of the same actions: equal results and envs."""
    import torch

    exp = rl.expand(index.table, index.states, index._ids(), witness_pruning=False)
    action = index.random_actions()
    schema = torch.full((index.num_envs,), -1, dtype=torch.int32, device=index.states.device)
    for i, a in enumerate(action.tolist()):
        if 0 <= a < int(exp.offsets[i + 1] - exp.offsets[i]):
            schema[i] = int(exp.schema[int(exp.offsets[i]) + a])
    binding = torch.empty((index.num_envs, label.label_width), dtype=torch.int32, device=index.states.device)
    a = index.step(action, final=True)
    b = label.step((schema, binding), final=True)
    assert a.binding.shape == b.binding.shape == (index.num_envs, 0)
    for name in a._fields:
        x, y = getattr(a, name), getattr(b, name)
        if name == "count":
            assert torch.equal((x > 0).to(torch.int32), y)
        elif isinstance(x, torch.Tensor):
            assert torch.equal(x, y), name
    for name in ("states", "steps", "draws", "task_ids", "at_goal", "reset_counts"):
        assert torch.equal(getattr(index, name), getattr(label, name)), name
    return a


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_env_index_and_label_steps_agree(device):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    if device == "cuda":
        gpu()
    _, table, _, _ = rows_and_ids()
    kwargs = dict(seed=5, max_steps=5, device=torch.device(device, 0) if device == "cuda" else "cpu")
    index = rt.BatchedEnv(table, 6, **kwargs)
    label = rt.BatchedEnv(table, 6, actions="label", **kwargs)
    assert index.label_width == label.label_width == 0
    for _ in range(8):
        compare(index, label)


def test_device_random_steps_equal_the_host():
    torch, dev = gpu()
    rt = pytest.importorskip("mymyr.rl.torch")
    _, table, _, _ = rows_and_ids()
    host = rt.BatchedEnv(table, 6, seed=7, max_steps=4)
    device = rt.BatchedEnv(table, 6, seed=7, max_steps=4, device=dev)
    for _ in range(6):
        a, b = host.step(None, final=True), device.step(None, final=True)
        for name in a._fields:
            x, y = getattr(a, name), getattr(b, name)
            if isinstance(x, torch.Tensor):
                assert x.shape == y.shape and torch.equal(x, y.cpu()), name
        assert torch.equal(host.states, device.states.cpu())
    torch.cuda.synchronize()


def test_the_jax_env_has_no_binding_columns():
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax

    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without the XLA FFI headers")
    import mymyr.rl.jax as rj

    _, table, rows, ids = rows_and_ids()
    host = rl.expand(table, rows, ids, goal=True)
    devices = ["cpu"] + (["cuda"] if any(d.platform == "gpu" for d in jax.devices()) else [])
    for device in devices:
        env = rj.Env(table, device=device)
        assert env.label_width == 0
        state, _ = env.reset(jax.random.key(0), 4)
        state, _, _, _, _, info = env.step(jax.random.key(1), state)
        assert info["binding"].shape == (4, 0) and info["schema"].shape == (4,)
        put = jax.device_put(np.asarray(rows).view(np.uint32), jax.devices(device)[0])
        flat = env.expand_flat(put, host.total + 2, task_ids=jax.device_put(ids, jax.devices(device)[0]))
        assert np.asarray(flat.binding).shape == (host.total + 2, 0)
        assert np.array_equal(np.asarray(flat.offsets), np.asarray(host.offsets))
        assert np.array_equal(np.asarray(flat.schema)[: host.total], np.asarray(host.schema))
        padded = env.expand(put, 2, task_ids=jax.device_put(ids, jax.devices(device)[0]))
        assert np.asarray(padded.binding).shape == (len(rows), 2, 0)


def test_action_relations_capacity_bounds_plans_and_state_space_files(tmp_path):
    ts, table, rows, ids = rows_and_ids()
    x = rl.expand(table, rows, ids)
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
    batch = encoder.encode(rows, ids, actions=x)
    press = encoder.relation_names.index("press")
    assert encoder.relation_arities[press] == 1  # the action entity only
    assert int(batch.relation_counts[:, press].sum()) == 4 and int(batch.target_offsets[-1]) == 6
    bounds = encoder.capacity_bounds(table, len(rows), actions=True)
    assert bounds["entities"] >= int(batch.entity_offsets[-1]) and bounds["args"] >= int(batch.relation_arg_offsets[-1])
    replay = datasets.plan_rows(ts[0], ts[0].applicable_actions(ts[0].initial_state)[:1] +
                                ts[0].applicable_actions(walk(ts[0])[1])[:1])
    assert replay.binding.shape == (2, 0) and replay.schema.tolist() == [0, 1]
    labelled = datasets.StateRows.from_plans([replay])
    assert labelled.labels["binding"].shape == (2, 0) and labelled[0].labels["binding"].shape == (0,)
    space = datasets.state_space(ts[1])
    arrays = space.arrays()
    assert space.label_width == 0 and arrays["label_bindings"].shape == (space.num_transitions, 0)
    datasets.save(space, tmp_path / "space.npz")
    loaded = datasets.load(tmp_path / "space.npz", task=ts[1])
    again = loaded.arrays()
    assert again["label_bindings"].shape == (space.num_transitions, 0)
    assert np.array_equal(again["label_schemas"], arrays["label_schemas"])


def gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device")
    mc = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not mc.available():
        pytest.skip("no visible CUDA device")
    return torch, torch.device("cuda", 0)


def test_device_expansion_and_encodings_equal_the_host():
    torch, dev = gpu()
    _, table, rows, ids = rows_and_ids()
    host = rl.expand(table, rows, ids, goal=True)
    words = torch.from_numpy(np.asarray(rows).view(np.int64).copy()).to(dev)
    tids = torch.from_numpy(ids).to(dev)
    device = rl.expand(table, words, tids, goal=True, framework="torch")
    assert tuple(device.binding.shape) == (host.total, 0)
    assert np.array_equal(device.offsets.cpu().numpy(), np.asarray(host.offsets))
    assert np.array_equal(device.schema.cpu().numpy(), np.asarray(host.schema))
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
    expected = encoder.encode(rows, ids, actions=host).as_dict()
    actual = encoder.encode(words, tids, actions=device).as_dict("numpy")
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(expected[name]), err_msg=name)
    torch.cuda.synchronize()


def test_captured_device_steps_equal_eager():
    torch, dev = gpu()
    rt = pytest.importorskip("mymyr.rl.torch")
    _, table, _, _ = rows_and_ids()
    a = rt.BatchedEnv(table, 8, device=dev, seed=9, max_steps=4)
    b = rt.BatchedEnv(table, 8, device=dev, seed=9, max_steps=4)
    graph = a.capture(3, None, final=True)
    for _ in range(3):
        out = graph.replay()
        assert tuple(out.binding.shape[1:]) == (8, 0)
        for k in range(3):
            step = b.step(None, final=True)
            assert torch.equal(out.schema[k], step.schema) and torch.equal(out.reward[k], step.reward)
        assert torch.equal(a.states, b.states)
    a.check_errors()
    torch.cuda.synchronize()
