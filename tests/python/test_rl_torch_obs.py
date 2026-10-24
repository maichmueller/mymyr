"""mymyr.rl.torch: encoded observations of BatchedEnv (``observation=``, ``StepResult.obs`` / ``final_obs``,
``CapturedRollout`` observations) against the host encoder on the same rows, bit for bit, on the CPU and, with a
visible GPU, on the device (eager, captured and replayed), with per-env goals, opt-in actions and suites.

    env CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python -m pytest tests/python/test_rl_torch_obs.py
"""

import gc
from types import SimpleNamespace
import warnings
import weakref

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task
from test_rl_table import gpu_device, sets  # noqa: F401

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

CAPS = dict(entities=1 << 14, args=1 << 16)
VARIABLE = {"entity_role": "entities", "entity_ref": "entities", "relation_args": "args", "target_entity": "targets",
            "target_group": "targets", "numeric_values": "numeric"}
# the stored forms of a batch: the relation arrays, both, the Simple graph alone
OUTPUTS = {"relations": ("relations",), "both": ("relations", "simple"), "simple": ("simple",)}


def devices():
    return ["cpu", "cuda"]


def device_of(name):
    return torch.device("cpu") if name == "cpu" else gpu_device()


def arrays(batch):
    """A batch's scalars and arrays as NumPy, cut to its graphs and its sizes (capacity forms)."""
    d = batch.as_dict("numpy")
    sizes = batch.sizes
    g, r = d["num_graphs"], d["num_relations"]
    graphs = {"task_ids": g, "num_objects": g, "entity_offsets": g + 1, "relation_counts": g,
              "relation_offsets": g * r + 1, "relation_arg_offsets": g * r + 1, "goal_complete": g,
              "target_offsets": g + 1, "node_offsets": g + 1, "edge_offsets": g + 1}
    out = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            if key in graphs:
                value = value[: graphs[key]]
            if sizes is not None and key in VARIABLE:
                value = value[: sizes[VARIABLE[key]]]
            if sizes is not None and key == "node_label":
                value = value[: sizes["entities"] + sizes["tuples"]]
            if sizes is not None and key == "edge_label":
                value = value[: 2 * sizes["args"]]
            if sizes is not None and key == "edge_index":
                value = value[:, : 2 * sizes["args"]]
        out[key] = value
    return out


def assert_same(batch, expected):
    a, b = arrays(batch), arrays(expected)
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype, key
            assert np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def host(encoder, states, task_ids, goals=None, **kwargs):
    """The host encoding (exact) of rows given as tensors."""
    g = None if goals is None else tuple(np.ascontiguousarray(x.cpu().numpy()) for x in goals)
    return encoder.encode(np.ascontiguousarray(states.cpu().numpy()), np.ascontiguousarray(task_ids.cpu().numpy()),
                          goals=g, **kwargs)


@pytest.fixture(scope="module")
def gripper(sets):  # noqa: F811
    return rl.TaskTable(sets["gripper"])


@pytest.fixture(scope="module")
def suite(sets):  # noqa: F811
    return rl.TaskSuite([rl.TaskTable(sets["gripper"]), rl.TaskTable(sets["blocks"])])


# ------------------------------------------------------------------------------------------------ one table


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("capacities", [CAPS, None], ids=["capacities", "exact"])
def test_step_obs_are_the_host_encoding_of_the_result_rows(gripper, dev, layout, capacities):
    device = device_of(dev)
    encoder = encoding.FlatRelationEncoder(gripper, layout=layout)
    env = rt.BatchedEnv(gripper, 24, device=device, seed=3, max_steps=4, observation=encoder,
                        obs_capacities=capacities)
    assert env.observation_encoders == [encoder]
    resets = 0
    for _ in range(9):
        ids = env.task_ids.clone()
        r = env.step()
        assert_same(r.obs, host(encoder, r.states, r.task_ids))
        # the final observation: the reached states in the instances they were reached in
        assert_same(r.final_obs, host(encoder, r.final_states, ids))
        resets += int((r.terminated | r.truncated).sum())
        assert r.obs.device == device if dev == "cuda" else True
    assert resets > 0  # autoresets happened (they change the instance of a row)
    assert_same(env.observation().encoding, host(encoder, env.states, env.task_ids))


@pytest.mark.parametrize("dev", devices())
def test_without_autoreset_the_final_obs_is_the_obs(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 8, device=device_of(dev), seed=1, autoreset=False, observation=encoder,
                        obs_capacities=CAPS)
    r = env.step()
    assert r.final_obs is r.obs
    assert_same(r.obs, host(encoder, r.states, r.task_ids))
    r = env.step(final=False)
    assert r.final_obs is None and r.obs is not None


def test_without_an_encoder_there_are_no_obs(gripper):
    env = rt.BatchedEnv(gripper, 4, seed=1)
    r = env.step()
    assert r.obs is None and r.final_obs is None and env.observation().encoding is None


@pytest.mark.parametrize("dev", devices())
def test_per_env_goals_are_the_goal_relations(gripper, dev):
    device = device_of(dev)
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 12, device=device, seed=5, goals=True, max_steps=3, observation=encoder,
                        obs_capacities=CAPS)
    # goals: the states of a few random steps (a goal per env that is not its instance's goal)
    probe = rt.BatchedEnv(gripper, 12, device=device, seed=9, task_ids=env.task_ids.cpu())
    for _ in range(3):
        probe.step()
    goal_pos, goal_neg = probe.states[:, : env.words].clone(), torch.zeros_like(probe.states[:, : env.words])
    env.reset(task_ids=env.task_ids.clone(), goal_pos=goal_pos, goal_neg=goal_neg)
    obs = env.observation().encoding
    assert_same(obs, host(encoder, env.states, env.task_ids, goals=(env.goal_pos, env.goal_neg)))
    default = arrays(host(encoder, env.states, env.task_ids))
    assert not np.array_equal(arrays(obs)["relation_args"], default["relation_args"])
    for _ in range(5):
        ids, gp, gn = env.task_ids.clone(), env.goal_pos.clone(), env.goal_neg.clone()
        r = env.step()
        assert_same(r.obs, host(encoder, r.states, r.task_ids, goals=(r.goal_pos, r.goal_neg)))
        assert_same(r.final_obs, host(encoder, r.final_states, ids, goals=(gp, gn)))


@pytest.mark.parametrize("dev", devices())
def test_actions_are_off_by_default(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)  # a schema with action relations
    env = rt.BatchedEnv(gripper, 6, device=device_of(dev), seed=2, observation=encoder, obs_capacities=CAPS)
    r = env.step()
    lanes = np.asarray([lane == "action" for lane in encoder.relation_lanes])
    assert lanes.any()
    counts = arrays(r.obs)["relation_counts"]
    assert counts[:, lanes].sum() == 0
    assert_same(r.obs, host(encoder, r.states, r.task_ids))


def test_actions_opt_in_on_the_cpu(gripper):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 6, seed=2, observation=encoder, obs_actions=True)
    for _ in range(3):
        r = env.step()
        actions = rl.expand(gripper, r.states, r.task_ids)
        assert_same(r.obs, host(encoder, r.states, r.task_ids, actions=actions))
    lanes = np.asarray([lane == "action" for lane in encoder.relation_lanes])
    assert arrays(r.obs)["relation_counts"][:, lanes].sum() > 0




def test_refusals(gripper, sets):  # noqa: F811
    with pytest.raises(ValueError, match="action relations"):
        rt.BatchedEnv(gripper, 4, observation=encoding.FlatRelationEncoder(gripper, include_actions=False),
                      obs_actions=True)
    other = encoding.FlatRelationEncoder(rl.TaskTable(sets["gripper"][:2]))
    with pytest.raises(ValueError, match="another table"):
        rt.BatchedEnv(gripper, 4, observation=other)
    with pytest.raises(ValueError, match="need an observation encoder"):
        rt.BatchedEnv(gripper, 4, obs_capacities=CAPS)
    with pytest.raises(TypeError, match="FlatRelationEncoder"):
        rt.BatchedEnv(gripper, 4, observation=encoding.FlatTransitionEncoder(gripper))
    with pytest.raises(ValueError, match="2 encoders"):
        rt.BatchedEnv(rl.TaskSuite([gripper, rl.TaskTable(sets["blocks"])]), 4,
                      observation=encoding.FlatRelationEncoder(gripper))
    with pytest.raises(ValueError, match="at least the number"):
        rt.BatchedEnv(gripper, 4, observation=encoding.FlatRelationEncoder(gripper), obs_capacities={**CAPS, "graphs": 2})


def test_an_obs_that_does_not_fit_is_flagged(gripper):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 6, device=gpu_device(), seed=2, observation=encoder,
                        obs_capacities=dict(entities=4, args=4))
    r = env.step()
    assert r.obs.sizes["capacity_overflow"]


# ------------------------------------------------------------------------------------------------ captured rollouts


def test_observer_registrations_have_independent_ownership(gripper):
    from mymyr.rl.torch import _ops
    from mymyr.rl.torch._observation import Observer

    encoder = encoding.FlatRelationEncoder(gripper)
    external = _ops.register(encoder)
    assert _ops.register(encoder) == external
    # Registration and batch templates require no device allocation.
    env = SimpleNamespace(core=SimpleNamespace(table=gripper), num_envs=4, device=torch.device("cuda", 0))
    first = Observer(env, encoder, CAPS, False)
    second = Observer(env, encoder, CAPS, False)
    first_handle, second_handle = first.handles[0], second.handles[0]
    try:
        first.close()
        first.close()
        assert _ops.lookup(second_handle) is encoder
        assert _ops.lookup(external) is encoder
        with pytest.raises(ValueError, match="no object registered"):
            _ops.lookup(first_handle)
        second.close()
        reference = weakref.ref(encoder)
        del encoder, first, second
        gc.collect()
        assert reference() is _ops.lookup(external)
        assert _ops.register(reference()) == external
    finally:
        _ops.unregister(first_handle)
        _ops.unregister(second_handle)
        _ops.unregister(external)
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("external", [False, True], ids=["env_owned", "caller_registered"])
@pytest.mark.parametrize("close", [False, True], ids=["destroy", "close"])
@pytest.mark.needs_gil
def test_shared_encoder_survives_other_environment_close(gripper, compiled, external, close):
    from mymyr.rl.torch import _ops

    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    handle = _ops.register(encoder) if external else None
    first = rt.BatchedEnv(gripper, 4, device=device, observation=encoder, obs_capacities=CAPS)
    second = rt.BatchedEnv(gripper, 4, device=device, observation=encoder, obs_capacities=CAPS)
    torch._dynamo.reset()

    def step():
        return second.step(final=False).obs.relation_args

    call = torch.compile(step, fullgraph=True, backend="aot_eager") if compiled else step
    try:
        call()
        torch.cuda.synchronize(device)
        if close:
            first.close()
        else:
            del first
            gc.collect()
        call()
        torch.cuda.synchronize(device)
        assert_same(second.encode_observation(), host(encoder, second.states, second.task_ids))
        if external:
            assert _ops.lookup(handle) is encoder
    finally:
        torch.cuda.synchronize(device)
        second.close()
        if close:
            first.close()
        if external:
            assert _ops.lookup(handle) is encoder
            _ops.unregister(handle)
        torch._dynamo.reset()


@pytest.mark.parametrize("final", [False, True])
def test_captured_rollout_obs_equal_the_host_and_collate(gripper, final):
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 16, device=device, seed=4, max_steps=3, observation=encoder, obs_capacities=CAPS)
    K = 5
    rollout = env.capture(K, final=final)
    for _ in range(2):  # replays rewrite the same arrays
        ids = env.task_ids.clone()
        out = rollout.replay()
        assert len(out.obs) == K and (out.final_obs is None) == (not final)
        for k in range(K):
            assert_same(out.obs[k], host(encoder, out.states[k], out.task_ids[k]))
            if final:
                before = ids if k == 0 else out.task_ids[k - 1]
                assert_same(out.final_obs[k], host(encoder, out.final_states[k], before))
        # the K steps' observations, collated: one batch of the K * N rows in step order
        assert_same(encoding.collate(out.obs),
                    host(encoder, out.states.reshape(-1, out.states.shape[-1]), out.task_ids.reshape(-1)))


def test_captured_obs_equal_eager_obs(gripper):
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    eager = rt.BatchedEnv(gripper, 16, device=device, seed=6, max_steps=3, observation=encoder, obs_capacities=CAPS)
    captured = rt.BatchedEnv(gripper, 16, device=device, seed=6, max_steps=3, observation=encoder,
                             obs_capacities=CAPS)
    out = captured.capture(4).replay()
    for k in range(4):
        r = eager.step(final=False)
        assert_same(out.obs[k], r.obs)
        assert torch.equal(out.obs[k].device_sizes, r.obs.device_sizes)


@pytest.mark.parametrize("dev", devices())
def test_eager_step_obs_collate(gripper, dev):
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 10, device=device_of(dev), seed=3, max_steps=3, observation=encoder,
                        obs_capacities=CAPS)
    steps = [env.step(final=False) for _ in range(4)]
    states = torch.cat([r.states for r in steps])
    ids = torch.cat([r.task_ids for r in steps])
    assert_same(encoding.collate([r.obs for r in steps]), host(encoder, states, ids))


@pytest.mark.parametrize("outputs", OUTPUTS)
@pytest.mark.parametrize("over", ["table", "suite"])
def test_device_obs_with_capacities_never_wait(gripper, suite, over, outputs):
    device = gpu_device()
    table = gripper if over == "table" else suite
    encoders = [encoding.FlatRelationEncoder(t, outputs=OUTPUTS[outputs])
                for t in ([gripper] if over == "table" else suite.tables)]
    env = rt.BatchedEnv(table, 16, device=device, seed=3, max_steps=3, observation=encoders, obs_capacities=CAPS)
    env.step()  # the first calls make the per-stream state
    torch.cuda.synchronize(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch calls the sync debug mode a prototype
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                env.step()
                env.encode_observation()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)


def test_capture_needs_capacities(gripper):
    env = rt.BatchedEnv(gripper, 4, device=gpu_device(), observation=encoding.FlatRelationEncoder(gripper))
    with pytest.raises(ValueError, match="obs_capacities"):
        env.capture(2)


def test_obs_under_torch_cuda_graph(gripper):
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 8, device=device, seed=7, max_steps=3, observation=encoder, obs_capacities=CAPS)
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        env.encode_observation()  # acquires the encoder on the capture stream
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        r = env.step(final=False)
    for _ in range(3):
        graph.replay()
        assert_same(r.obs, host(encoder, r.states, r.task_ids))


@pytest.mark.needs_gil
def test_obs_under_torch_compile(gripper):
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    a = rt.BatchedEnv(gripper, 8, device=device, seed=7, max_steps=3, observation=encoder, obs_capacities=CAPS)
    b = rt.BatchedEnv(gripper, 8, device=device, seed=7, max_steps=3, observation=encoder, obs_capacities=CAPS)

    def step(env):
        r = env.step(final=False)
        return r.obs.relation_args, r.obs.device_sizes

    torch._dynamo.reset()
    probe = rt.BatchedEnv(gripper, 8, device=device, seed=7, max_steps=3, observation=encoder, obs_capacities=CAPS)
    assert torch._dynamo.explain(step)(probe).graph_break_count == 0
    compiled = torch.compile(step, fullgraph=True, backend="aot_eager")
    for _ in range(3):
        args, sizes = compiled(a)
        r = b.step(final=False)
        assert torch.equal(sizes, r.obs.device_sizes)
        assert torch.equal(args[: int(sizes[2])], r.obs.relation_args[: int(sizes[2])])


# ------------------------------------------------------------------------------------------------ suites


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("capacities", [CAPS, None], ids=["capacities", "exact"])
@pytest.mark.parametrize("invalid", [-1, "past_end", -(2**31), 2**31 - 1])
def test_suite_invalid_task_ids_are_reported_without_unsafe_gathers(suite, dev, capacities, invalid, monkeypatch):
    device = device_of(dev)
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rt.BatchedEnv(suite, 4, device=device, task_ids=[len(suite) - 1] * 4,
                        observation=encoders, obs_capacities=capacities)
    env.task_ids[0] = len(suite) if invalid == "past_end" else invalid
    original_getitem = torch.Tensor.__getitem__
    lookups = (env._observer._domain_of, env._observer._local_ids)

    def checked_getitem(tensor, index):
        if any(tensor is lookup for lookup in lookups):
            # Check before dispatch: an unsafe CUDA gather must never be submitted, even on a failing test.
            assert bool(((index >= 0) & (index < tensor.numel())).all())
        return original_getitem(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", checked_getitem)
    try:
        if dev == "cuda" and capacities is not None:
            obs = env.encode_observation()
            assert any(batch.sizes["invalid_input"] for batch in obs.batches)
        else:
            with pytest.raises(ValueError, match="task id|row is invalid"):
                env.encode_observation()
    finally:
        if dev == "cuda":
            torch.cuda.synchronize(device)
        env.close()


def suite_check(suite, encoders, obs, states, task_ids, goals=None):
    """A SuiteObservation against the host encodings of each domain's rows."""
    ids = task_ids.cpu().numpy().astype(np.int64)
    domain = np.asarray(suite.domain_of)[ids]
    local = np.asarray(suite.local_ids, np.int32)[ids]
    assert np.array_equal(obs.domain.cpu().numpy(), domain)
    index = obs.index.cpu().numpy()
    for d, encoder in enumerate(encoders):
        rows = np.flatnonzero(domain == d)
        assert np.array_equal(index[rows], np.arange(len(rows)))
        sel = torch.as_tensor(rows, device=states.device)
        g = None if goals is None else (goals[0][sel], goals[1][sel])
        assert obs.batches[d].num_graphs == len(rows)
        assert_same(obs.batches[d], host(encoder, states[sel], torch.as_tensor(local[rows]), goals=g))


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("capacities", [CAPS, None], ids=["capacities", "exact"])
@pytest.mark.parametrize("outputs", OUTPUTS)
def test_suite_obs_are_one_batch_per_domain(suite, dev, capacities, outputs):
    encoders = [encoding.FlatRelationEncoder(t, outputs=OUTPUTS[outputs]) for t in suite.tables]
    env = rt.BatchedEnv(suite, 20, device=device_of(dev), seed=2, max_steps=3, observation=encoders,
                        obs_capacities=capacities)
    for _ in range(6):
        ids = env.task_ids.clone()
        r = env.step()
        assert isinstance(r.obs, rt.SuiteObservation) and len(r.obs.batches) == 2
        suite_check(suite, encoders, r.obs, r.states, r.task_ids)
        suite_check(suite, encoders, r.final_obs, r.final_states, ids)


@pytest.mark.parametrize("dev", devices())
def test_suite_obs_with_goals(suite, dev):
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rt.BatchedEnv(suite, 12, device=device_of(dev), seed=4, goals=True, max_steps=3, observation=encoders,
                        obs_capacities=CAPS)
    for _ in range(4):
        r = env.step(final=False)
        suite_check(suite, encoders, r.obs, r.states, r.task_ids, goals=(r.goal_pos, r.goal_neg))


def test_captured_suite_rollout(suite):
    device = gpu_device()
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rt.BatchedEnv(suite, 16, device=device, seed=8, max_steps=3, observation=encoders, obs_capacities=CAPS)
    K = 4
    rollout = env.capture(K)
    for _ in range(2):
        out = rollout.replay()
        for k in range(K):
            suite_check(suite, encoders, out.obs[k], out.states[k], out.task_ids[k])
        states = out.states.reshape(-1, out.states.shape[-1])
        ids = out.task_ids.reshape(-1)
        domain = np.asarray(suite.domain_of)[ids.cpu().numpy()]
        local = np.asarray(suite.local_ids, np.int32)[ids.cpu().numpy()]
        for d, encoder in enumerate(encoders):
            rows = np.flatnonzero(domain == d)
            sel = torch.as_tensor(rows, device=device)
            assert_same(encoding.collate([o.batches[d] for o in out.obs]),
                        host(encoder, states[sel], torch.as_tensor(local[rows])))


@pytest.mark.parametrize(("layout", "outputs"), [("graph", "relations"), ("relation", "relations"), ("graph", "both"),
                                                 ("relation", "both"), ("graph", "simple")])
@pytest.mark.parametrize("targets", [False, True])
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_suite_overflow_zeroes_owned_buffers(suite, monkeypatch, layout, outputs, targets, captured):
    from mymyr.encoding import _device

    device = gpu_device()
    options = dict(target_sources=("goal",), index_dtype="int64") if targets else {}
    if targets and outputs != "simple":
        options["emit_counts"] = False
    encoders = [encoding.FlatRelationEncoder(t, layout=layout, outputs=OUTPUTS[outputs], **options)
                for t in suite.tables]
    env = rt.BatchedEnv(suite, 6, device=device, task_ids=[0, len(suite) - 1] * 3,
                        observation=encoders, obs_capacities=dict(entities=1, args=1, targets=1))
    allocate = _device._outputs

    def sentinel_outputs(*args, **kwargs):
        out = allocate(*args, **kwargs)
        for array in out.values():
            array.fill_(17)
        return out

    monkeypatch.setattr(_device, "_outputs", sentinel_outputs)
    try:
        rollout = env.capture(2, final=True) if captured else None
        for _ in range(2):
            if captured:
                result = rollout.replay()
                observations = (*result.obs, *result.final_obs)
            else:
                result = env.step()
                observations = (result.obs, result.final_obs)
            for observation in observations:
                for batch in observation.batches:
                    assert batch.sizes["capacity_overflow"]
                    for name, array in batch._arrays.items():
                        assert not torch.count_nonzero(array).item(), name
    finally:
        torch.cuda.synchronize(device)
        env.close()


@pytest.mark.parametrize("dev", devices())
def test_derived_atoms_of_the_observations(dev):
    task = text_task("philosophers__p03-phil4")  # axioms and a derived goal
    encoder = encoding.FlatRelationEncoder(task)
    env = rt.BatchedEnv(task, 6, device=device_of(dev), seed=2, max_steps=4, observation=encoder, obs_capacities=CAPS)
    for _ in range(4):
        r = env.step()
        assert_same(r.obs, host(encoder, r.states, r.task_ids))


# the numeric and the Simple observation forms: a numeric task's relations with its numeric values, the Simple graph
# of a classical task, and the Simple graph of a numeric task without numeric relations
FORMS = {"numeric": ("cs-counters", dict()),
         "simple": ("gripper__prob05", dict(outputs=("simple",))),
         "simple_of_numeric": ("cs-counters", dict(outputs=("simple",), include_numeric=False))}
NUMERIC_CAPS = dict(CAPS, numeric=1 << 12)


def form_task(form):
    name, options = FORMS[form]
    task = mymyr.Task.from_text(str(ROOT / f"tests/data/numeric_tasks/{name}.txt")) if name.startswith("cs-") \
        else text_task(name)
    return task, encoding.FlatRelationEncoder(task, **options)


@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("capacities", [NUMERIC_CAPS, None], ids=["capacities", "exact"])
@pytest.mark.parametrize("form", FORMS)
def test_numeric_and_simple_obs_are_the_host_encoding(dev, capacities, form):
    task, encoder = form_task(form)
    env = rt.BatchedEnv(task, 12, device=device_of(dev), seed=5, max_steps=5, observation=encoder,
                        obs_capacities=capacities)
    for _ in range(8):
        ids = env.task_ids.clone()
        r = env.step()
        assert_same(r.obs, host(encoder, r.states, r.task_ids))
        assert_same(r.final_obs, host(encoder, r.final_states, ids))
    if form == "numeric":
        assert r.obs.sizes["numeric"] > 0 if capacities else len(r.obs.numeric_values) > 0


@pytest.mark.needs_gil
@pytest.mark.parametrize("form", FORMS)
def test_numeric_and_simple_obs_under_torch_compile(form):
    device = gpu_device()
    task, encoder = form_task(form)
    a, b = (rt.BatchedEnv(task, 8, device=device, seed=7, max_steps=3, observation=encoder,
                          obs_capacities=NUMERIC_CAPS) for _ in range(2))
    name = "numeric_values" if form == "numeric" else "edge_index"

    def step(env):
        r = env.step(final=False)
        return getattr(r.obs, name), r.obs.device_sizes

    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph=True, backend="aot_eager")
    for _ in range(3):
        values, sizes = compiled(a)
        r = b.step(final=False)
        assert torch.equal(sizes, r.obs.device_sizes)
        assert torch.equal(values, getattr(r.obs, name))
        assert_same(r.obs, host(encoder, r.states, r.task_ids))


def test_one_table_without_the_front_end():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    env = rt.BatchedEnv(task, 5, seed=1, observation=encoder, obs_capacities=CAPS)
    r = env.step()
    assert_same(r.obs, host(encoder, r.states, r.task_ids))


@pytest.mark.parametrize("over", ["table", "suite"])
@pytest.mark.parametrize("capacity", [False, True])
def test_device_actions_are_the_expansion_of_the_observed_states(gripper, suite, over, capacity):
    device = gpu_device()
    table = gripper if over == "table" else suite
    encoders = [encoding.FlatRelationEncoder(t) for t in ([gripper] if over == "table" else suite.tables)]
    ids = None if over == "table" else [0, len(suite) - 1] * 3
    env = rt.BatchedEnv(table, 6, device=device, seed=4, max_steps=4, task_ids=ids, observation=encoders,
                        obs_actions=True, obs_capacities=CAPS if capacity else None)
    lanes = [np.asarray([lane == "action" for lane in e.relation_lanes]) for e in encoders]
    for _ in range(3):
        r = env.step()
        batches = [r.obs] if over == "table" else list(r.obs.batches)
        for k, batch in enumerate(batches):
            if over == "table":
                states, task_ids = r.states, r.task_ids
            else:
                rows = (r.obs.domain == k).nonzero().reshape(-1)
                states = r.states[rows]
                task_ids = torch.as_tensor(np.asarray(suite.local_ids, np.int32), device=device)[r.task_ids[rows].long()]
            t = encoders[k].table
            expected = host(encoders[k], states, task_ids,
                            actions=rl.expand(t, np.ascontiguousarray(states.cpu().numpy()),
                                              np.ascontiguousarray(task_ids.cpu().numpy())))
            assert_same(batch, expected)
            assert arrays(batch)["relation_counts"][:, lanes[k]].sum() > 0


@pytest.mark.parametrize("over", ["table", "suite"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_capacity_action_observations_capture_without_host_reads(gripper, suite, over, layout, monkeypatch):
    device = gpu_device()
    table = gripper if over == "table" else suite
    tables = [gripper] if over == "table" else suite.tables
    encoders = [encoding.FlatRelationEncoder(t, layout=layout, target_sources=("action", "goal")) for t in tables]
    caps = dict(entities=4096, args=16384, targets=4096)
    ids = [0, len(table) - 1] * 2
    env = rt.BatchedEnv(table, 4, device=device, task_ids=ids, max_steps=3, seed=7, goals=True,
                        observation=encoders, obs_actions=True, obs_capacities=caps)
    # Capture's warm-up acquires both the encoder contexts and the asynchronous expanders.
    env.step()

    def no_host(*args, **kwargs):
        raise AssertionError("capacity action observations must not read device values on the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "tolist"):
            patch.setattr(torch.Tensor, name, no_host)
        rollout = env.capture(3, final=True)
        rollout.replay()
    torch.cuda.synchronize(device)
    out = rollout.replay()
    for j in range(3):
        for final in (False, True):
            obs = out.final_obs[j] if final else out.obs[j]
            states = out.final_states[j] if final else out.states[j]
            ids_host = out.task_ids[j].cpu().numpy()
            domains = np.asarray(suite.domain_of)[ids_host] if over == "suite" else np.zeros(4, np.int32)
            local = np.asarray(suite.local_ids)[ids_host] if over == "suite" else ids_host
            batches = obs.batches if over == "suite" else [obs]
            for k, batch in enumerate(batches):
                selected = np.flatnonzero(domains == k)
                own = np.ascontiguousarray(states.cpu().numpy()[selected, :tables[k].words])
                own_ids = np.asarray(local[selected], np.int32)
                masks = tuple(np.ascontiguousarray(g.cpu().numpy()[selected]) for g in (env.goal_pos, env.goal_neg))
                expected = encoders[k].encode(own, own_ids, goals=masks, actions=rl.expand(tables[k], own, own_ids))
                assert_same(batch, expected)
                sizes = batch.sizes
                for name, size in VARIABLE.items():
                    if name in batch.as_dict():
                        assert not getattr(batch, name)[sizes.get(size, 0):].any()


@pytest.mark.parametrize("over", ["table", "suite"])
def test_action_observation_capacity_overflow_zeroes_all_outputs(gripper, suite, over):
    device = gpu_device()
    table = gripper if over == "table" else suite
    encoders = [encoding.FlatRelationEncoder(t, target_sources=("action",))
                for t in ([gripper] if over == "table" else suite.tables)]
    ids = None if over == "table" else [0, len(suite) - 1] * 2
    env = rt.BatchedEnv(table, 4, device=device, task_ids=ids, observation=encoders, obs_actions=True,
                        obs_capacities=dict(entities=0, args=0, targets=0))
    result = env.step()
    for obs in (result.obs, result.final_obs):
        for batch in (obs.batches if over == "suite" else [obs]):
            assert batch.sizes["capacity_overflow"] and not batch.sizes["invalid_input"]
            for value in batch.as_dict("numpy").values():
                if isinstance(value, np.ndarray):
                    assert not value.any()


def test_capacity_action_observations_return_before_the_device(gripper):
    from test_encoding_device_transitions import returns_before_the_device
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 4, device=device, observation=encoder, obs_actions=True, obs_capacities=CAPS)
    env.step()
    early, result = returns_before_the_device(lambda: env.step(final=False))
    assert early
    own = np.ascontiguousarray(result.states.cpu().numpy())
    ids = np.ascontiguousarray(result.task_ids.cpu().numpy())
    assert_same(result.obs, encoder.encode(own, ids, actions=rl.expand(gripper, own, ids)))


@pytest.mark.needs_gil
def test_capacity_action_observations_under_torch_compile(gripper):
    device = gpu_device()
    encoder = encoding.FlatRelationEncoder(gripper)
    env = rt.BatchedEnv(gripper, 4, device=device, observation=encoder, obs_actions=True, obs_capacities=CAPS)
    torch._dynamo.reset()
    try:
        call = torch.compile(lambda: env.step(final=False).obs.relation_args, fullgraph=True, backend="aot_eager")
        call()
        call()
        own = np.ascontiguousarray(env.states.cpu().numpy())
        ids = np.ascontiguousarray(env.task_ids.cpu().numpy())
        assert_same(env.encode_observation(), encoder.encode(own, ids, actions=rl.expand(gripper, own, ids)))
    finally:
        torch.cuda.synchronize(device)
        env.close()
        torch._dynamo.reset()


def test_captured_action_observations_follow_changing_suite_domains(suite):
    device = gpu_device()
    encoders = [encoding.FlatRelationEncoder(t) for t in suite.tables]
    env = rt.BatchedEnv(suite, 4, device=device, task_ids=[0] * 4, max_steps=1, observation=encoders,
                        obs_actions=True, obs_capacities=CAPS)
    next_ids = torch.zeros(4, dtype=torch.int32, device=device)
    rollout = env.capture(2, next_task_ids=next_ids, final=True)
    for instance in (len(suite) - 1, 0, len(suite) - 1):
        next_ids.fill_(instance)
        before = env.task_ids.clone()
        out = rollout.replay()
        for j in range(2):
            for obs, states, ids in ((out.obs[j], out.states[j], out.task_ids[j]),
                                     (out.final_obs[j], out.final_states[j], before if j == 0 else out.task_ids[j - 1])):
                global_ids = ids.cpu().numpy()
                domain = np.asarray(suite.domain_of)[global_ids]
                local = np.asarray(suite.local_ids, np.int32)[global_ids]
                for k, batch in enumerate(obs.batches):
                    selected = np.flatnonzero(domain == k)
                    own = np.ascontiguousarray(states.cpu().numpy()[selected, :suite.tables[k].words])
                    own_ids = np.ascontiguousarray(local[selected])
                    expected = encoders[k].encode(own, own_ids, actions=rl.expand(suite.tables[k], own, own_ids))
                    assert_same(batch, expected)
