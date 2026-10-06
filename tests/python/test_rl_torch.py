"""mymyr.rl.torch: the environments against rl.expand (canonical order, labels, counts), rewards, termination,
truncation and autoreset, the counter-based RNG, the torch custom ops (fake implementations, torch.compile without
graph breaks), the TorchRL PlanningEnv (check_env_specs), and, with a visible GPU, the device environments equal to
the CPU's bit for bit.

The GPU tests run only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_torch.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, SMALL_TASKS, text_task

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

NUMERIC = ROOT / "tests/data/numeric_tasks"


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


@pytest.fixture(scope="module")
def blocks():
    return text_task("blocks__probBLOCKS-8-0", atoms="frozen")


def expand_rows(task, states):
    """rl.expand of the rows (as wide as the input) as (succ, schema, binding, offsets, goal) torch CPU tensors."""
    x = rl.expand(task, states.cpu(), words=states.shape[1] - task.numeric_words, goal=True)
    return x.succ, x.schema, x.binding, x.offsets.to(torch.int64), x.goal


def check_step(task, env, before, action, r):
    """One BatchedEnv step (autoreset off, no truncation) against rl.expand of the states before it."""
    succ, schema, binding, off, goal = expand_rows(task, before)
    L = binding.shape[1]
    count = off[1:] - off[:-1]
    after = env.states.cpu()
    for i in range(before.shape[0]):
        a, c = int(action[i]), int(count[i])
        if c == 0:  # stuck: no move, terminated
            assert torch.equal(after[i], before[i]) and bool(r.terminated[i]) and not bool(r.invalid[i])
            assert int(r.schema[i]) == -1
        elif a < 0 or a >= c:  # invalid: no move
            assert torch.equal(after[i], before[i]) and bool(r.invalid[i]) and not bool(r.terminated[i])
            assert int(r.schema[i]) == -1 and float(r.reward[i]) == env.core.step_reward
        else:
            j = int(off[i]) + a
            assert torch.equal(after[i], succ[j])
            assert int(r.schema[i]) == int(schema[j])
            assert torch.equal(r.binding[i, :L].cpu(), binding[j])
            assert bool(r.goal[i]) == bool(goal[j])
            assert not bool(r.invalid[i])


# ------------------------------------------------------------------------------------------------ CPU environments


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_actions_index_the_canonical_successor_order(name):
    task = text_task(name)
    n = 24
    env = rt.BatchedEnv(task, n, device="cpu", autoreset=False, seed=3)
    gen = torch.Generator().manual_seed(11)
    for t in range(12):
        before = env.states.clone()
        _, _, _, off, _ = expand_rows(task, before)
        count = (off[1:] - off[:-1]).to(torch.int32)
        assert torch.equal(env.count, count)
        action = (torch.rand(n, generator=gen) * count.clamp(min=1)).to(torch.int64)
        if t % 4 == 3:
            action[::5] = count[::5].to(torch.int64) + 2  # invalid
        r = env.step(action)
        check_step(task, env, before, action, r)
        assert torch.equal(r.count, env.count)


def test_rewards_truncation_and_autoreset(blocks):
    n = 32
    env = rt.BatchedEnv(blocks, n, max_steps=3, step_reward=-1.0, goal_reward=5.0, dead_end_reward=-2.0)
    init = torch.from_numpy(env.core.initial_states()[0].view(np.int64))
    assert torch.equal(env.states, init.expand(n, -1))
    assert torch.equal(env.count, torch.full((n,), env.core.initial_counts[0], dtype=torch.int32))
    for t in range(1, 7):
        before = env.states.clone()
        r = env.step(final=True)
        assert r.final_states.shape == (n, env.row_words)
        assert torch.all(r.reward == -1.0 + 5.0 * r.goal.float())  # blocks has no dead ends
        done = r.terminated | r.truncated
        assert torch.equal(r.terminated, r.goal)
        if t % 3 == 0:
            assert torch.all(done)  # the episode's third step truncates the rest
        assert torch.equal(r.truncated, ~r.terminated & (t % 3 == 0))
        # autoreset: finished envs restart, final_states keeps the reached state
        assert torch.equal(env.states[done], init.expand(int(done.sum()), -1))
        assert torch.all(env.steps[done] == 0)
        assert not torch.equal(r.final_states, before) or t == 0
    env.set_seed(9)
    assert torch.all(env.draws == 0)


def test_dead_ends_and_stuck_states():
    task = text_task("pegsol-08-strips__p22", atoms="frozen")
    n = 64
    env = rt.BatchedEnv(task, n, autoreset=False, step_reward=-1.0, dead_end_reward=-10.0)
    seen = 0
    for _ in range(40):
        r = env.step()
        dead = r.terminated & ~r.goal
        assert torch.all(r.reward[dead] == -11.0)
        assert torch.all(env.count[dead] == 0)
        seen += int(dead.sum())
    assert seen > 0


def test_random_policy_draws_from_the_counter_based_rng(blocks):
    n = 40
    a = rt.BatchedEnv(blocks, n, seed=7, max_steps=5)
    b = rt.BatchedEnv(blocks, n, seed=7, max_steps=5)
    for _ in range(15):
        action = rt.rng.successor_indices(7, torch.arange(n), b.draws, b.count)
        assert torch.all((action >= 0) & (action < b.count.clamp(min=1)))
        ra, rb = a.step(), b.step(action)
        assert torch.equal(a.states, b.states) and torch.equal(ra.reward, rb.reward)
        assert torch.equal(a.draws, b.draws)
    # a split batch: the second half runs as envs n/2.. of the whole
    c = rt.BatchedEnv(blocks, n // 2, seed=7, max_steps=5, first_env=n // 2)
    d = rt.BatchedEnv(blocks, n, seed=7, max_steps=5, threads=3)
    for _ in range(15):
        c.step()
        d.step()
    assert torch.equal(c.states, d.states[n // 2 :])


def test_philox_known_answers():
    # Random123 kat_vectors (philox4x32_10): counter, key -> output
    ctr = torch.tensor([[0, 0, 0, 0], [0xFFFFFFFF] * 4, [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344]])
    out = [rt.rng.philox(ctr[i : i + 1], k)[0].tolist() for i, k in enumerate([0, 0xFFFFFFFFFFFFFFFF, 0x299F31D0A4093822])]
    assert out[0] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    assert out[1] == [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]
    assert out[2] == [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]
    idx = rt.rng.successor_indices(1, torch.arange(20000), 0, 6)
    hist = torch.bincount(idx, minlength=6)
    assert idx.min() >= 0 and idx.max() < 6 and hist.min() > 3000
    assert int(rt.rng.successor_indices(1, 0, 0, 0)[0]) == -1


def test_refresh_after_writing_states(blocks):
    n = 16
    a = rt.BatchedEnv(blocks, n, seed=1)
    for _ in range(5):
        a.step()
    b = rt.BatchedEnv(blocks, n, seed=1)
    b.states.copy_(a.states)
    b.steps.copy_(a.steps)
    b.draws.copy_(a.draws)
    b.refresh()
    assert torch.equal(a.count, b.count)
    for _ in range(5):
        ra, rb = a.step(), b.step()
        assert torch.equal(a.states, b.states) and torch.equal(ra.reward, rb.reward)


def test_numeric_tasks_on_the_cpu():
    task = mymyr.Task.from_text(str(NUMERIC / "cs-counters.txt"))
    n = 8
    env = rt.BatchedEnv(task, n, autoreset=False, seed=2)
    assert env.core.numeric_words == task.numeric_words > 0
    gen = torch.Generator().manual_seed(5)
    for _ in range(6):
        before = env.states.clone()
        _, _, _, off, _ = expand_rows(task, before)
        count = off[1:] - off[:-1]
        action = (torch.rand(n, generator=gen) * count.clamp(min=1)).to(torch.int64)
        r = env.step(action)
        check_step(task, env, before, action, r)


def test_argument_errors(blocks):
    env = rt.BatchedEnv(blocks, 4)
    with pytest.raises(ValueError, match="int64"):
        rt._ops.step_impl(env.handle, env.states.to(torch.int32), None, env.steps, env.draws, env.counts, env.views,
                          env.count, None, None, None, None, 0, False)  # fmt: skip
    with pytest.raises(ValueError, match="shape"):
        rt._ops.step_impl(env.handle, env.states, None, env.steps, env.draws, env.counts, env.views, env.count,
                          None, None, torch.zeros(5, dtype=torch.int64), None, 0, False)  # fmt: skip
    with pytest.raises(ValueError):
        rt.Env(blocks, path="fast")
    with pytest.raises(TypeError, match="C-contiguous"):
        env.core.step(env.states.t().contiguous().t(), steps=env.steps, draws=env.draws)
    env.close()
    with pytest.raises(ValueError, match="handle"):
        env.step()


# ------------------------------------------------------------------------------------------------ ops and compile


def op_cases(env, task, dev):
    h = rt.register(task)
    s = env.states.clone()
    a = torch.zeros(env.num_envs, dtype=torch.int64, device=dev)
    m = torch.arange(env.num_envs, device=dev) % 2 == 0
    ids = env.task_ids if env.multi else None
    base = (env.handle, env.states, ids, env.steps, env.draws, env.counts, env.views, env.count, env.goal_pos, env.goal_neg)
    nxt = torch.zeros(env.num_envs, dtype=torch.int32, device=dev) if env.multi else None
    return [
        (torch.ops.mymyr.step.default, base + (None, nxt, 0, True)),
        (torch.ops.mymyr.step.default, base + (a, None, 0, False)),
        (torch.ops.mymyr.reset.default, (env.handle, env.states, ids, env.steps, env.counts, env.views, env.goal_pos,
                                         env.goal_neg, env.count, m, False)),
        (torch.ops.mymyr.refresh.default, (env.handle, env.states, ids, env.counts, env.views, env.count)),
        (torch.ops.mymyr.step_sync.default, base + (a, None, 0, True)),
        (torch.ops.mymyr.refresh_sync.default, (env.handle, env.states, ids, env.counts, env.views, env.count)),
        (torch.ops.mymyr.random_actions.default, (env.handle, env.draws, env.count, 0, 3)),
        (torch.ops.mymyr.expand.default, (h, s, ids, 8, False)),
        (torch.ops.mymyr.expand_flat.default, (h, s, ids, 64, False)),
        (torch.ops.mymyr.expand_flat.default, (h, s, ids, None, False)),
    ]


def check_ops(env, task, dev):
    for t in range(3):
        env.step()
    for op, args in op_cases(env, task, dev):
        result = torch.library.opcheck(op, args)
        assert set(result.values()) == {"SUCCESS"}, (op, result)
        # the fake implementation's shapes and dtypes equal the real ones
        real = op(*args)
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        with FakeTensorMode(shape_env=ShapeEnv()) as mode:
            fargs = tuple(mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in args)
            fake = op(*fargs)
        real = () if real is None else real
        fake = () if fake is None else fake
        assert len(real) == len(fake)
        for r, f in zip(real, fake):
            assert r.dtype == f.dtype and r.device == f.device and r.dim() == f.dim()
            for rs, fs in zip(r.shape, f.shape):
                if isinstance(fs, int):
                    assert rs == fs
    rt.unregister(rt.register(task))


def test_ops_fake_implementations_agree_with_real(blocks):
    check_ops(rt.BatchedEnv(blocks, 12, max_steps=4), blocks, torch.device("cpu"))


def test_expand_ops_equal_rl_expand(blocks):
    env = rt.BatchedEnv(blocks, 10)
    for _ in range(4):
        env.step()
    h = rt.register(blocks)
    s = env.states.clone()
    succ, schema, binding, off, goal = expand_rows(blocks, s)
    fsucc, fparent, fschema, fbinding, fgoal, foff = torch.ops.mymyr.expand_flat(h, s, None, None, False)
    assert torch.equal(fsucc, succ) and torch.equal(fschema, schema) and torch.equal(fbinding, binding)
    assert torch.equal(foff.to(torch.int64), off) and torch.equal(fgoal, goal)
    K = int((off[1:] - off[:-1]).max())
    psucc, pschema, pbinding, pgoal, pmask, pcount = torch.ops.mymyr.expand(h, s, None, K, False)
    assert torch.equal(pcount.to(torch.int64), off[1:] - off[:-1])
    assert torch.equal(psucc[pmask], succ) and torch.equal(pschema[pmask], schema)
    csucc, cparent, *_, coff = torch.ops.mymyr.expand_flat(h, s, None, 16, False)
    assert csucc.shape[0] == 16 and torch.equal(coff, foff) and torch.equal(csucc, succ[:16])


def rollout_fn(handle):
    def run(states, steps, draws, counts, views, count):
        total = torch.zeros((), device=states.device)
        for _ in range(5):
            r = torch.ops.mymyr.step(handle, states, None, steps, draws, counts, views, count, None, None, None, None,
                                     0, False)
            total = total + r[0].sum() + r[1].sum()
        return total

    return run


def check_compile(task, dev):
    a = rt.BatchedEnv(task, 32, device=dev, max_steps=4, seed=5)
    b = rt.BatchedEnv(task, 32, device=dev, max_steps=4, seed=5)
    fn = rollout_fn(a.handle)
    ex = torch._dynamo.explain(fn)(a.states, a.steps, a.draws, a.counts, a.views, a.count)
    assert ex.graph_break_count == 0 and ex.graph_count == 1
    a = rt.BatchedEnv(task, 32, device=dev, max_steps=4, seed=5)
    compiled = torch.compile(rollout_fn(a.handle), fullgraph=True, backend="aot_eager")
    for _ in range(3):
        x = compiled(a.states, a.steps, a.draws, a.counts, a.views, a.count)
        y = rollout_fn(b.handle)(b.states, b.steps, b.draws, b.counts, b.views, b.count)
        assert torch.equal(x, y) and torch.equal(a.states, b.states) and torch.equal(a.draws, b.draws)

    def env_loop(env):
        total = torch.zeros((), device=dev)
        for _ in range(3):
            total = total + env.step().reward.sum()
        return total

    torch._dynamo.reset()
    ex = torch._dynamo.explain(env_loop)(a)
    assert ex.graph_break_count == 0


def test_torch_compile_without_graph_breaks(blocks):
    check_compile(blocks, torch.device("cpu"))


def test_random_actions_compile_without_graph_breaks(blocks):
    """PlanningEnv.rand_action's op (mymyr::random_actions) in one graph with steps: no graph break, same choices."""
    a = rt.BatchedEnv(blocks, 16, seed=3, max_steps=4)
    b = rt.BatchedEnv(blocks, 16, seed=3, max_steps=4)

    def loop(env):
        total = torch.zeros((), dtype=torch.int64)
        for _ in range(3):
            act = env.random_actions(4)
            total = total + act.sum()
            env.step(act)
        return total

    torch._dynamo.reset()
    ex = torch._dynamo.explain(loop)(rt.BatchedEnv(blocks, 16, seed=3, max_steps=4))
    assert ex.graph_break_count == 0
    compiled = torch.compile(loop, fullgraph=True, backend="aot_eager")
    assert torch.equal(compiled(a), loop(b)) and torch.equal(a.states, b.states) and torch.equal(a.draws, b.draws)


# ------------------------------------------------------------------------------------------------ TorchRL


def check_planning_env(task, dev):
    pytest.importorskip("torchrl")
    from torchrl.envs.utils import check_env_specs

    for kwargs in [{}, {"max_actions": 8, "goals": True}]:
        env = rt.PlanningEnv(task, 16, device=dev, max_steps=4, goal_reward=1.0, **kwargs)
        check_env_specs(env, seed=3)
        td = env.rollout(12, break_when_any_done=False)
        assert td.batch_size == (16, 12)
        assert torch.all(td["action"] < td["count"].clamp(min=1))
        assert torch.all(td["next", "truncated"].sum(1) >= 2)  # max_steps 4 in 12 steps
        if "action_mask" in kwargs or kwargs:
            assert torch.equal(td["action_mask"].sum(-1), td["count"].clamp(max=8))
            assert td["goal_pos"].shape == (16, 12, env.batched.core.words)
        env.close()


def test_planning_env_specs(blocks):
    check_planning_env(blocks, "cpu")


def test_planning_env_partial_reset(blocks):
    pytest.importorskip("torchrl")
    from torchrl.envs.utils import step_mdp

    env = rt.PlanningEnv(blocks, 6, max_steps=0)
    td = env.reset()
    td["action"] = torch.zeros(6, dtype=torch.int64)
    td = step_mdp(env.step(td))
    moved = td["state"].clone()
    td["_reset"] = torch.tensor([True, False, True, False, False, False]).unsqueeze(-1)
    out = env.reset(td)
    init = torch.from_numpy(env.batched.core.initial_states()[0].view(np.int64))
    assert torch.equal(out["state"][0], init) and torch.equal(out["state"][2], init)
    assert torch.equal(out["state"][1], moved[1]) and torch.equal(env.batched.states[1], moved[1])
    assert torch.equal(out["count"][0], torch.tensor(env.batched.core.initial_counts[0]))
    assert not out["done"].any()
    # the environments' own state agrees
    assert torch.equal(env.batched.states, out["state"])


# ------------------------------------------------------------------------------------------------ device


def trajectories_equal(task, dev, n=256, steps=40, **cfg):
    """Device env == CPU env: random policy and given actions (with invalid ones), labels, final states."""
    gen = torch.Generator().manual_seed(17)
    cpu = rt.BatchedEnv(task, n, device="cpu", **cfg)
    gpu_env = rt.BatchedEnv(task, n, device=dev, **cfg)
    assert torch.equal(gpu_env.count.cpu(), cpu.count)
    for t in range(steps):
        action = None
        if t % 3 == 2:
            action = (torch.rand(n, generator=gen) * (cpu.count + 1).clamp(min=1)).to(torch.int64)
        a = cpu.step(action, final=True)
        b = gpu_env.step(None if action is None else action.to(dev), final=True)
        for field in a._fields:
            assert torch.equal(getattr(a, field), getattr(b, field).cpu()), (t, field)
        assert torch.equal(cpu.states, gpu_env.states.cpu()) and torch.equal(cpu.draws, gpu_env.draws.cpu())
        assert torch.equal(cpu.steps, gpu_env.steps.cpu())
    gpu_env.check_errors()
    return gpu_env


@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
@pytest.mark.parametrize("name", SMALL_TASKS)
def test_device_env_equals_cpu_env(name, atoms):
    dev = gpu()
    task = text_task(name, atoms=atoms)
    cfg = dict(max_steps=15, step_reward=-1.0, goal_reward=2.5, dead_end_reward=-0.5, seed=4)
    env = trajectories_equal(task, dev, **cfg)
    assert env.core.fast == (rt.fast_unsupported(task) == "")
    if env.core.fast:
        trajectories_equal(task, dev, path="general", **cfg)


def test_device_env_batches_streams_and_resets(blocks):
    dev = gpu()
    n = 300
    whole = rt.BatchedEnv(blocks, n, device=dev, seed=8, max_steps=6)
    half = rt.BatchedEnv(blocks, n - 120, device=dev, seed=8, max_steps=6, first_env=120)
    side = torch.cuda.Stream(dev)
    with torch.cuda.stream(side):
        other = rt.BatchedEnv(blocks, n, device=dev, seed=8, max_steps=6)
        for _ in range(20):
            other.step()
    for _ in range(20):
        whole.step()
        half.step()
    torch.cuda.synchronize()
    assert torch.equal(whole.states[120:], half.states) and torch.equal(whole.states, other.states)
    mask = torch.arange(n, device=dev) % 3 == 0
    whole.reset(mask)
    init = torch.from_numpy(whole.core.initial_states()[0].view(np.int64)).to(dev)
    assert torch.equal(whole.states[mask], init.expand(int(mask.sum()), -1)) and torch.all(whole.steps[mask] == 0)
    cpu = rt.BatchedEnv(blocks, n, seed=8, max_steps=6)
    for _ in range(20):
        cpu.step()
    cpu.reset(mask.cpu())
    assert torch.equal(whole.states.cpu(), cpu.states) and torch.equal(whole.count.cpu(), cpu.count)


def test_device_ops_and_compile(blocks):
    dev = gpu()
    check_ops(rt.BatchedEnv(blocks, 12, device=dev, max_steps=4), blocks, dev)
    check_compile(blocks, dev)
    env = rt.BatchedEnv(blocks, 10, device=dev)
    for _ in range(4):
        env.step()
    h = rt.register(blocks)
    s = env.states.clone()
    for args in [(s, None, 8, False), (s, None, 8, True)]:
        d = torch.ops.mymyr.expand(h, *args)
        c = torch.ops.mymyr.expand(h, args[0].cpu(), *args[1:])
        assert all(torch.equal(x.cpu(), y) for x, y in zip(d, c))
    for cap in [None, 16]:
        d = torch.ops.mymyr.expand_flat(h, s, None, cap, False)
        c = torch.ops.mymyr.expand_flat(h, s.cpu(), None, cap, False)
        rows = min(int(c[5][-1]), cap or 1 << 30)
        assert all(torch.equal(x.cpu()[:rows], y[:rows]) for x, y in zip(d[:5], c[:5])) and torch.equal(d[5].cpu(), c[5])


def test_device_planning_env(blocks):
    dev = gpu()
    check_planning_env(blocks, dev)
    # the same actions give the same TorchRL transitions on both devices
    envs = [rt.PlanningEnv(blocks, 8, device=d, max_steps=3, max_actions=6) for d in ("cpu", dev)]
    tds = [e.reset() for e in envs]
    gen = torch.Generator().manual_seed(2)
    for _ in range(10):
        a = (torch.rand(8, generator=gen) * tds[0]["count"].clamp(min=1, max=6)).to(torch.int64)
        outs = []
        for e, td in zip(envs, tds):
            td["action"] = a.to(e.device)
            outs.append(e.step_and_maybe_reset(td))
        for key in ["state", "count", "reward", "done", "terminated", "truncated", "action_mask"]:
            assert torch.equal(outs[0][0]["next", key], outs[1][0]["next", key].cpu()), key
        tds = [o[1] for o in outs]


def test_device_refusals():
    dev = gpu()
    numeric = mymyr.Task.from_text(str(NUMERIC / "cs-counters.txt"))
    with pytest.raises(ValueError, match="numeric"):
        rt.BatchedEnv(numeric, 4, device=dev)
    lazy = text_task("blocks__probBLOCKS-8-0", atoms="lazy")
    assert "lazy" in rt.fast_unsupported(lazy)
    with pytest.raises(ValueError, match="lazy"):
        rt.BatchedEnv(lazy, 4, device=dev, path="fast")
    env = rt.BatchedEnv(text_task("blocks__probBLOCKS-8-0", atoms="frozen"), 4, device=dev)
    with pytest.raises(ValueError, match="cuda"):  # a CPU action for a device env (BatchedEnv.step moves it)
        rt._ops.step_impl(env.handle, env.states, None, env.steps, env.draws, env.counts, env.views, env.count,
                          None, None, torch.zeros(4, dtype=torch.int64), None, 0, False)  # fmt: skip
    with pytest.raises(TypeError, match="cuda"):
        env.core.step(env.states.cpu())


# ------------------------------------------------------------------------------------------------ rand_action, ParallelEnv, collectors


def planning_devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        try:
            import mymyr.cuda as mc

            if mc.available():
                out.append(torch.device("cuda", 0))
        except ImportError:
            pass
    return out


@pytest.mark.parametrize("max_actions", [None, 3])
def test_rand_action_is_the_counter_random_policy(blocks, max_actions):
    pytest.importorskip("torchrl")
    from torchrl.envs.utils import step_mdp

    for dev in planning_devices():
        n = 24
        env = rt.PlanningEnv(blocks, n, device=dev, seed=6, max_actions=max_actions)
        twin = rt.BatchedEnv(blocks, n, device=dev, seed=6, autoreset=False)
        td = env.reset()
        for _ in range(10):
            count = twin.count.cpu()
            limit = count if max_actions is None else count.clamp(max=max_actions)
            want = rt.rng.successor_indices(6, torch.arange(n), twin.draws.cpu(), limit).clamp(min=0)
            td = env.rand_action(td)
            assert torch.equal(td["action"].cpu(), want)
            td = step_mdp(env.step(td))
            if max_actions is None:
                twin.step(None)  # the random policy's own step takes the same successors
            else:
                twin.step(want.to(twin.device))
            assert torch.equal(env.batched.states, twin.states) and torch.equal(env.batched.draws, twin.draws)
        # a rollout without a policy follows the same streams after set_seed
        env.set_seed(9)
        twin.set_seed(9)
        env.reset()
        twin.reset()
        roll = env.rollout(6, break_when_any_done=False)
        for t in range(6):
            count = twin.count.cpu()
            limit = count if max_actions is None else count.clamp(max=max_actions)
            want = rt.rng.successor_indices(9, torch.arange(n), twin.draws.cpu(), limit).clamp(min=0)
            assert torch.equal(roll["action"][:, t].cpu(), want)
            twin.step(want.to(twin.device))
        assert torch.equal(roll["next", "state"][:, -1], twin.states)
        env.close()


def last_successor(td):
    """A deterministic policy: the last of the first max_actions successors (0 without successors)."""
    k = td["action_mask"].shape[-1]
    return td.set("action", (td["count"].clamp(max=k) - 1).clamp(min=0))


def planning_env(n=8, seed=0, max_actions=None):
    """A PlanningEnv factory for ParallelEnv workers (importable by reference)."""
    task = mymyr.Task.from_text(str(ROOT / "tests/data/tasks/gripper__prob05.txt"))
    return rt.PlanningEnv(task, n, seed=seed, max_steps=5, goal_reward=1.0, max_actions=max_actions)


def test_serial_and_parallel_envs():
    pytest.importorskip("torchrl")
    import functools

    from torchrl.envs import ParallelEnv, SerialEnv

    make = functools.partial(planning_env, 8, 0, 4)
    serial = SerialEnv(2, make)
    serial.set_seed(3)
    a = serial.rollout(7, last_successor, break_when_any_done=False)
    assert a.batch_size == (2, 8, 7)
    assert torch.all(a["action"] < a["count"].clamp(min=1))
    # the same environments in worker processes give the same rollouts
    par = ParallelEnv(2, make, mp_start_method="spawn")
    try:
        par.set_seed(3)
        b = par.rollout(7, last_successor, break_when_any_done=False)
        for key in ["action", "state", "count", ("next", "reward"), ("next", "done"), ("next", "state")]:
            assert torch.equal(a[key], b[key]), key
    finally:
        par.close()
        serial.close()


def test_collector():
    pytest.importorskip("torchrl")
    import warnings

    with warnings.catch_warnings():  # torchrl 0.12 deprecates its own weight updaters at import
        warnings.simplefilter("ignore", DeprecationWarning)
        from torchrl.collectors import Collector

    env = planning_env(8, 1, 6)
    collector = Collector(env, None, frames_per_batch=64, total_frames=192)
    batches = list(collector)
    collector.shutdown()
    assert len(batches) == 3
    for b in batches:
        assert b.numel() == 64
        valid = b["action"] < b["count"]
        # invalid actions (the action spec's random policy may exceed the count) do not move
        assert torch.equal(b["next", "state"][~valid], b["state"][~valid])
        assert torch.all(b["next", "reward"] <= 1.0)
    # a deterministic policy collects the transitions of a plain rollout
    env2, env3 = planning_env(8, 1, 6), planning_env(8, 1, 6)
    collector = Collector(env2, last_successor, frames_per_batch=40, total_frames=40)
    got = next(iter(collector))
    collector.shutdown()
    want = env3.rollout(5, last_successor, break_when_any_done=False)
    for key in ["action", "state", ("next", "state"), ("next", "reward"), ("next", "done")]:
        assert torch.equal(got[key], want[key]), key
