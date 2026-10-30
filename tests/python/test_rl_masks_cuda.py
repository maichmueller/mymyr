"""Action masks and label-mode environments on a CUDA device, equal to the host's bit for bit.

- rl.applicable_schemas / applicable_objects / is_applicable on CUDA torch tensors == on host rows (random walks of
  the small tasks, tables, a suite, invalid labels, prefixes and bindings; tasks the kernels do not run take the host
  functions); the host errors for the same inputs; the torch ops (status bits instead of errors, fakes under
  torch.compile, CUDA graph capture);
- BatchedEnv(actions="label") on the device == on the CPU (random walks with invalid labels, autoreset with
  next_task_ids, final states, per-env goal masks, a suite), and BatchedEnv.capture of label steps (a policy of the
  mask ops) == the eager steps == the CPU env.

GPU only (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_masks_cuda.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, SMALL_TASKS, text_task, walk

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

DEVICE_TASKS = ["blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1", "depot__p02"]


def gpu():
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("mymyr sees no CUDA device")
    return torch.device("cuda", 0)


def numeric_task():
    return mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))


def arities(table):
    """Per row's domain: the schema arities ([S] per domain; a TaskSuite: per global instance id)."""
    if isinstance(table, rl.TaskSuite):
        return [[s.arity for s in table[g].formalism.schemas] for g in range(len(table))]
    task = table if isinstance(table, mymyr.Task) else table[0]
    return [s.arity for s in task.formalism.schemas]


def row_arities(table, ids, n):
    a = arities(table)
    if isinstance(table, rl.TaskSuite):
        return [a[int(ids[i])] for i in range(n)]
    return [a] * n


def compare(table, rows, ids=None, seed=3):
    """The three functions on device rows == on host rows; returns how many labels were applicable."""
    dev = gpu()
    host = np.ascontiguousarray(np.asarray(rows).view(np.int64)).copy()
    d_rows = torch.from_numpy(host).to(dev)
    d_ids = None if ids is None else torch.from_numpy(np.asarray(ids, np.int32)).to(dev)
    h_ids = None if ids is None else np.asarray(ids, np.int32)
    actual = rl.applicable_schemas(table, d_rows, d_ids)
    assert actual.device == dev and actual.dtype == torch.bool
    expected = np.asarray(rl.applicable_schemas(table, host, h_ids))
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
    n = host.shape[0]
    per_row = row_arities(table, h_ids, n)
    S = max(len(a) for a in per_row)
    width = max([max(a, default=0) for a in per_row] + [1])
    O = table.max_objects if not isinstance(table, mymyr.Task) else table.num_objects
    rng = np.random.default_rng(seed)
    exp = rl.expand(table, host, h_ids, witness_pruning=False)
    offsets, schemas, bindings = np.asarray(exp.offsets), np.asarray(exp.schema), np.asarray(exp.binding)
    applicable = 0
    for depth in range(width):
        for query in range(3):
            schema = rng.integers(-1, S + 1, n).astype(np.int32)
            for i in range(n):
                if 0 <= schema[i] < len(per_row[i]) and per_row[i][schema[i]] <= depth:
                    schema[i] = -1  # the host functions raise for a depth at or past the arity
            prefix = rng.integers(-1, O + 1, (n, width)).astype(np.int32)
            if query == 0:  # prefixes of applicable labels
                for i in range(n):
                    js = [j for j in range(offsets[i], offsets[i + 1]) if per_row[i][schemas[j]] > depth]
                    if js:
                        j = js[rng.integers(len(js))]
                        schema[i], prefix[i, :bindings.shape[1]] = schemas[j], bindings[j]
            pre = prefix if depth else None
            want = np.asarray(rl.applicable_objects(table, host, schema, pre, depth, h_ids))
            got = rl.applicable_objects(table, d_rows, torch.from_numpy(schema).to(dev),
                                        None if pre is None else torch.from_numpy(pre).to(dev), depth, d_ids)
            np.testing.assert_array_equal(got.cpu().numpy(), want, err_msg=f"depth {depth} query {query}")
    for query in range(3):
        schema = rng.integers(-1, S + 1, n).astype(np.int32)
        binding = rng.integers(-1, O + 1, (n, width)).astype(np.int32)
        if query < 2:
            for i in range(n):
                if offsets[i + 1] > offsets[i]:
                    j = rng.integers(offsets[i], offsets[i + 1])
                    schema[i], binding[i, :bindings.shape[1]] = schemas[j], bindings[j]
                    if query == 1 and per_row[i][schema[i]]:  # a valid label with one object changed
                        binding[i, rng.integers(per_row[i][schema[i]])] = rng.integers(O)
        want = np.asarray(rl.is_applicable(table, host, schema, binding, h_ids))
        got = rl.is_applicable(table, d_rows, torch.from_numpy(schema).to(dev), torch.from_numpy(binding).to(dev),
                               d_ids)
        np.testing.assert_array_equal(got.cpu().numpy(), want)
        applicable += int(want.sum())
        scalar = int(schema[0])
        np.testing.assert_array_equal(
            rl.is_applicable(table, d_rows, scalar, torch.from_numpy(binding).to(dev), d_ids).cpu().numpy(),
            np.asarray(rl.is_applicable(table, host, scalar, binding, h_ids)))
    # the torch ops: the same masks, status 0
    h = rt.register(table)
    mask, status = rt.applicable_schemas(h, d_rows, d_ids)
    assert int(status) == 0
    np.testing.assert_array_equal(mask.cpu().numpy(), expected)
    return applicable


@pytest.mark.parametrize("name", SMALL_TASKS + ["numeric"])
def test_device_masks_equal_host_on_walks(name):
    task = numeric_task() if name == "numeric" else text_task(name)
    states = walk(task, steps=6, walks=3)
    compare(task, task.encode(states))
    if name in DEVICE_TASKS:
        assert rt.fast_unsupported(task, actions="label") == ""
        dense = text_task(name, atoms="dense")
        compare(dense, dense.encode(walk(dense, steps=6, walks=3)))


def test_device_masks_tables_and_suites():
    gpu()
    blocks = rl.TaskTable([text_task("blocks__probBLOCKS-8-0")])
    suite = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0"),
                                text_task("logistics00__probLOGISTICS-6-1")])
    for table in (blocks, suite):
        env = rt.BatchedEnv(table, 24, task_ids=torch.arange(24) % len(table), seed=7, max_steps=5)
        for _ in range(4):
            env.step()
        ids = None if len(table) == 1 else env.task_ids.numpy()
        assert compare(table, env.states.numpy(), ids) > 0
    # a suite with an instance the kernels do not run: the host functions
    mixed = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("openstacks-opt08-adl__p03")])
    ids = np.array([0, 1, 1, 0], np.int32)
    compare(mixed, mixed.initial_states(ids), ids)


def test_device_masks_errors_and_status():
    dev = gpu()
    task = text_task("blocks__probBLOCKS-8-0")
    host = np.ascontiguousarray(np.asarray(task.encode(walk(task, steps=3, walks=1))).view(np.int64)).copy()
    rows = torch.from_numpy(host).to(dev)
    n = host.shape[0]
    h = rt.register(task)
    schema = next(i for i, s in enumerate(task.formalism.schemas) if s.arity >= 2)
    arity = task.formalism.schemas[schema].arity
    sch = torch.full((n,), schema, dtype=torch.int32, device=dev)
    # depth at the arity: the host's ValueError; status bit 8 in the op
    with pytest.raises(ValueError, match="depth must be below the schema arity"):
        rl.applicable_objects(task, rows, sch, torch.zeros((n, arity), dtype=torch.int32, device=dev), arity)
    _, status = rt.applicable_objects(h, rows, sch, torch.zeros((n, arity), dtype=torch.int32, device=dev), arity)
    assert int(status) == 8
    # narrow bindings
    narrow = torch.zeros((n, arity - 1), dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="binding width must be at least the schema arity"):
        rl.is_applicable(task, rows, sch, narrow)
    assert int(rt.is_applicable(h, rows, sch, narrow)[1]) == 16
    # task ids outside the table, atom bits past the instance's slots
    table = rl.TaskTable([task, task])
    bad_ids = torch.tensor([0, 2] + [1] * (n - 2), dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="task id 2 of row 1 is outside"):
        rl.applicable_schemas(table, rows, bad_ids)
    assert int(rt.applicable_schemas(rt.register(table), rows, bad_ids)[1]) == 1
    slots = task.num_atoms
    if slots % 64:
        bad = host.copy()
        bad[0, slots // 64] |= np.int64(1) << np.int64(63)
        d_bad = torch.from_numpy(bad).to(dev)
        with pytest.raises(ValueError, match="atom slots"):
            rl.applicable_schemas(task, d_bad)
        assert int(rt.applicable_schemas(h, d_bad)[1]) == 2
    # the scalar checks and the inputs' devices
    with pytest.raises(ValueError, match="num_objects"):
        rl.applicable_objects(task, rows, sch, None, 0, num_objects=0)
    with pytest.raises(ValueError, match="prefix"):
        rl.applicable_objects(task, rows, sch, None, 1)
    with pytest.raises(TypeError, match="CUDA"):
        rl.applicable_objects(task, rows, sch.cpu(), None, 0)
    with pytest.raises(TypeError, match="int32"):
        rl.is_applicable(task, rows, sch, torch.zeros((n, arity), dtype=torch.int64, device=dev))
    with pytest.raises(ValueError, match="shape"):
        rl.is_applicable(task, rows, sch[:1], torch.zeros((n, arity), dtype=torch.int32, device=dev))
    empty = rl.applicable_schemas(task, rows[:0])
    assert tuple(empty.shape) == (0, task.num_schemas) and empty.device == dev


@pytest.mark.needs_gil
def test_mask_ops_compile_and_capture():
    dev = gpu()
    task = text_task("gripper__prob05")
    host = np.ascontiguousarray(np.asarray(task.encode(walk(task, steps=8, walks=2))).view(np.int64)).copy()
    rows = torch.from_numpy(host).to(dev)
    n = rows.shape[0]
    h = rt.register(task)
    sch = torch.zeros(n, dtype=torch.int32, device=dev)
    binding = torch.zeros((n, task.label_width), dtype=torch.int32, device=dev)

    def masks(states):
        s, s_status = torch.ops.mymyr.applicable_schemas(h, states)
        o, o_status = torch.ops.mymyr.applicable_objects(h, states, sch, None, 0)
        a, a_status = torch.ops.mymyr.is_applicable(h, states, sch, binding)
        return s, o, a, s_status | o_status | a_status

    eager = masks(rows)
    assert int(eager[3]) == 0
    np.testing.assert_array_equal(eager[0].cpu().numpy(), np.asarray(rl.applicable_schemas(task, host)))
    compiled = torch.compile(masks, backend="aot_eager", fullgraph=True)(rows)
    for x, y in zip(eager, compiled):
        assert torch.equal(x, y)
    # CUDA graph: warm on a side stream, capture, then replay over new rows written in place
    static = rows.clone()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        masks(static)
    torch.cuda.current_stream(dev).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = masks(static)
    static.copy_(rows.flip(0))
    graph.replay()
    expected = masks(rows.flip(0))
    for x, y in zip(out, expected):
        assert torch.equal(x, y)
    # the CPU op: the host functions
    cpu_mask, cpu_status = rt.applicable_schemas(h, torch.from_numpy(host))
    assert int(cpu_status) == 0 and torch.equal(cpu_mask, eager[0].cpu())


# ------------------------------------------------------------------------------------------------ label-mode env


def labels_of(env, actions):
    """The labels of index actions in the states of a CPU env (-1 schema for out-of-range indices)."""
    exp = rl.expand(env.table, env.states, env._ids(), witness_pruning=False)
    schema = torch.full((env.num_envs,), -1, dtype=torch.int32)
    binding = torch.full((env.num_envs, env.label_width), -1, dtype=torch.int32)
    for i, a in enumerate(actions.tolist()):
        if 0 <= a < int(exp.offsets[i + 1] - exp.offsets[i]):
            j = int(exp.offsets[i]) + a
            schema[i] = exp.schema[j]
            binding[i, :exp.label_width] = exp.binding[j]
    return schema, binding


ENV_FIELDS = ("states", "task_ids", "steps", "draws", "count", "at_goal", "goal_pos", "goal_neg", "_reset_counts")


def expect_equal(host_env, dev_env, a, b, where):
    for f in a._fields:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y.cpu()), (where, f)
    for f in ENV_FIELDS:
        x, y = getattr(host_env, f), getattr(dev_env, f)
        assert (x is None and y is None) or torch.equal(x, y.cpu()), (where, f)


@pytest.mark.parametrize("name", DEVICE_TASKS + ["suite"])
@pytest.mark.parametrize("goals", [False, True])
def test_label_env_device_equals_host(name, goals):
    dev = gpu()
    if name == "suite":
        table = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")])
    else:
        table = text_task(name)
    n = 24
    ids = torch.arange(n, dtype=torch.int32) % (len(table) if name == "suite" else 1)
    kwargs = dict(actions="label", max_steps=6, seed=19, step_reward=-1.0, goal_reward=5.0, dead_end_reward=-3.0,
                  goals=goals, task_ids=ids, sampler="uniform" if name == "suite" else None)
    host_env = rt.BatchedEnv(table, n, **kwargs)
    dev_env = rt.BatchedEnv(table, n, device=dev, **kwargs)
    assert dev_env.capturable
    if goals:
        for env in (host_env, dev_env):
            env.goal_pos[::4].zero_()
            env.goal_neg[::4].zero_()
            env.refresh()
    # follows host_env's states and instances to translate index actions; per-env goals are not needed for that
    index = rt.BatchedEnv(table, n, **{k: v for k, v in kwargs.items() if k not in ("actions", "goals")})
    gen = torch.Generator().manual_seed(5)
    for t in range(30):
        index.states.copy_(host_env.states)
        index.task_ids.copy_(host_env.task_ids)
        index.refresh()
        actions = torch.randint(-1, 6, (n,), generator=gen)
        schema, binding = labels_of(index, actions)
        if t % 4 == 1:  # applicable objects in another schema, or objects outside the instance
            schema[::5] = (schema[::5] + 1) % (table.num_schemas + 1)
            binding[1::7, 0] = 1000
        nxt = None if name != "suite" or t % 3 else (torch.arange(n, dtype=torch.int32) + t) % len(table)
        final = t % 2 == 0
        a = host_env.step((schema, binding), next_task_ids=nxt, final=final)
        b = dev_env.step((schema.to(dev), binding.to(dev)), next_task_ids=None if nxt is None else nxt.to(dev),
                         final=final)
        expect_equal(host_env, dev_env, a, b, (name, t))
    dev_env.refresh()
    host_env.refresh()
    assert torch.equal(host_env.count, dev_env.count.cpu())
    dev_env.check_errors()


def test_label_env_device_refusals():
    dev = gpu()
    for name in ("openstacks-opt08-adl__p03", "miconic-simpleadl__s10-2"):
        task = text_task(name)
        why = rt.fast_unsupported(task, actions="label")
        assert why
        with pytest.raises(ValueError, match="label"):
            rt.BatchedEnv(task, 2, actions="label", device=dev)
    with pytest.raises(ValueError, match="label"):
        rt.BatchedEnv(numeric_task(), 2, actions="label", device=dev)
    env = rt.BatchedEnv(text_task("blocks__probBLOCKS-8-0"), 4, actions="label", device=dev)
    with pytest.raises(ValueError, match="label"):
        env.step(None)
    with pytest.raises(ValueError, match="label"):
        env.random_actions()
    with pytest.raises(ValueError, match="label"):
        env.step(torch.zeros(4, dtype=torch.int64, device=dev))
    with pytest.raises(ValueError, match="random policy"):
        env.capture(2)


def mask_policy(table):
    """A capturable policy of the mask ops: per row the applicable schema first in a rotation by the step counter,
    then objects depth by depth, the first applicable one in a rotation by the state."""
    h = rt.register(table)
    task = table if isinstance(table, mymyr.Task) else table[0]
    arity = torch.tensor([s.arity for s in task.formalism.schemas], dtype=torch.int32)
    S, L = len(arity), int(arity.max())
    O = task.num_objects
    on_device = {}  # the arities per device, copied outside the capture (the warm-up calls)

    def policy(env):
        dev = env.states.device
        if dev not in on_device:
            on_device[dev] = arity.to(dev)
        ar = on_device[dev]
        smask, _ = torch.ops.mymyr.applicable_schemas(h, env.states)
        rot = (torch.arange(S, device=dev)[None, :] + env.steps[:, None]) % S
        schema = torch.argmax(smask.to(torch.int32) * (S - rot), dim=1).to(torch.int32)
        binding = torch.zeros((env.num_envs, L), dtype=torch.int32, device=dev)
        orot = (torch.arange(O, device=dev)[None, :] + env.states[:, :1]) % O
        for d in range(L):
            sd = torch.where(ar[schema.long()] > d, schema, torch.full_like(schema, -1))
            omask, _ = torch.ops.mymyr.applicable_objects(h, env.states, sd, binding, d)
            binding[:, d] = torch.argmax(omask[:, :O].to(torch.int64) * (O - orot), dim=1).to(torch.int32)
        return schema, binding

    return policy


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05"])
def test_label_capture_equals_eager_and_host(name):
    dev = gpu()
    task = text_task(name)
    n, K = 64, 4
    kwargs = dict(actions="label", max_steps=7, seed=3, goal_reward=1.0, dead_end_reward=-2.0)
    a = rt.BatchedEnv(task, n, device=dev, **kwargs)
    b = rt.BatchedEnv(task, n, device=dev, **kwargs)
    c = rt.BatchedEnv(task, n, **kwargs)
    policy = mask_policy(task)
    g = a.capture(K, policy, final=True)
    moved = 0
    for r in range(5):
        if r == 3:
            mask = torch.arange(n, device=dev) % 3 == 1
            a.reset(mask)
            b.reset(mask)
            c.reset(mask.cpu())
        out = g.replay()
        for k in range(K):
            labels = policy(b)
            assert torch.equal(out.action_schema[k], labels[0]) and torch.equal(out.action_binding[k], labels[1])
            eager = b.step(labels, final=True)
            host = c.step((labels[0].cpu(), labels[1].cpu()), final=True)
            for f in ("reward", "terminated", "truncated", "goal", "invalid", "schema", "binding", "final_states",
                      "count", "states", "task_ids", "at_goal"):
                x = getattr(out, f)[k]
                assert torch.equal(x, getattr(eager, f)), (name, r, k, f)
                assert torch.equal(x.cpu(), getattr(host, f)), (name, r, k, f)
            moved += int((out.schema[k] >= 0).sum())
        for f in ENV_FIELDS:
            x, y, z = getattr(a, f), getattr(b, f), getattr(c, f)
            assert (x is None and z is None) or (torch.equal(x, y) and torch.equal(x.cpu(), z)), (name, r, f)
    assert moved > n
    a.check_errors()


def test_mask_ops_refuse_capturing_the_host_path():
    """A table the kernels do not run (a schema of 31 parameters) takes the host path, which a capture refuses."""
    dev = gpu()
    task = text_task("organic-synthesis-opt18-strips__p20")
    rows = torch.from_numpy(np.asarray(task.encode([task.initial_state])).view(np.int64).copy()).to(dev)
    h = rt.register(task)
    schema = torch.zeros(1, dtype=torch.int32, device=dev)
    binding = torch.zeros((1, task.label_width), dtype=torch.int32, device=dev)
    mask, status = rt.is_applicable(h, rows, schema, binding)  # eager: the host functions
    assert int(status) == 0
    assert torch.equal(mask.cpu(), torch.from_numpy(np.asarray(rl.is_applicable(task, rows.cpu().numpy(), 0,
                                                                                binding.cpu().numpy()))))
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="runs on the host .*parameters.*cannot be captured"):
        with torch.cuda.graph(graph):
            schema.add_(0)  # captured work before the refused call
            rt.is_applicable(h, rows, schema, binding)
