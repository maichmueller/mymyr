"""Device environments over tasks with axioms, on the fast path: universal preconditions with implications and
quantified goals (compiled into axioms; tests/data/pddl/linking) and a recursive :derived predicate in preconditions of
both signs and in goals (tests/data/pddl/beacons), as tables, a table of one instance and a suite.

- the fast path runs them (``fast_unsupported`` is empty, the environment is capturable);
- a captured rollout of the random policy with visited tables and encoded observations equals the eager steps on the
  device, the general path, the CPU environment and the CPU pool, byte for byte: states, rewards, flags, counts, final
  states, revisited, the visited tables and the observation arrays; over a table of one instance also the
  observations' action rows (labels, successor hashes, no-op flags) and action_visited;
- torch.compile of a loop of steps runs as CUDA graphs and equals the eager steps;
- the JAX environment's jitted device steps equal its CPU steps.

GPU only (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_axioms_device.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT
from test_rl_table import gpu_device
from test_rl_torch_obs import CAPS, assert_same

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

DATA = ROOT / "tests/data/pddl"
CFG = dict(max_steps=6, goal_reward=1.0, dead_end_reward=-2.0)
NAMES = ["linking", "beacons", "linking-p03", "linking+beacons"]
STEP = ("states", "task_ids", "reward", "terminated", "truncated", "goal", "invalid", "schema", "binding", "count",
        "at_goal", "final_states", "revisited")
ROWS = ("parent", "schema", "binding", "offsets", "succ_hash", "noop", "status")


def fixture(name):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    d = DATA / name
    return [mymyr.Task.from_pddl(d / "domain.pddl", p, atoms="dense") for p in sorted(d.glob("p*.pddl"))]


def table_of(name):
    linking, beacons = fixture("linking"), fixture("beacons")
    return {"linking": rl.TaskTable(linking), "beacons": rl.TaskTable(beacons),
            "linking-p03": rl.TaskTable([linking[2]]),
            "linking+beacons": rl.TaskSuite([rl.TaskTable(linking), rl.TaskTable(beacons)])}[name]


def encoders(table):
    tables = list(table.tables) if isinstance(table, rl.TaskSuite) else [table]
    return [encoding.FlatRelationEncoder(t, target_sources=("action", "goal")) for t in tables]


def caps():
    return dict(CAPS, targets=1 << 12)


def per_domain(value):
    """An observation field as a tuple (one entry per domain batch over a suite)."""
    return value if isinstance(value, tuple) else (value,)


def expect_rows_equal(a, b, where):
    for x, y in zip(per_domain(a), per_domain(b), strict=True):
        for f in ROWS:
            assert torch.equal(getattr(x, f).cpu(), getattr(y, f).cpu()), (where, f)


def expect_obs_equal(a, b):
    for x, y in zip(per_domain(getattr(a, "batches", a)), per_domain(getattr(b, "batches", b)), strict=True):
        assert_same(x, y)


@pytest.mark.parametrize("name", NAMES)
def test_the_fast_path_runs_device_axioms(name):
    dev = gpu_device()
    table = table_of(name)
    assert rt.fast_unsupported(table) == ""
    env = rt.BatchedEnv(table, 8, device=dev, **CFG)
    assert env.capturable and env.core.capture_unsupported == ""
    assert env.views.shape[1] == env.core.cache_view_words > 0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.jax_on_cpu
def test_captured_rollout_equals_eager_general_cpu_and_pool(name):
    dev = gpu_device()
    table = table_of(name)
    n, K, R = 48, 4, 3
    common = dict(seed=7, **CFG)
    # action rows on the device need the asynchronous device expansion, which runs tables of one instance with axioms
    actions = name == "linking-p03"
    obs = dict(observation=encoders(table), obs_actions=actions, obs_capacities=caps(), track_visited=True)
    captured = rt.BatchedEnv(table, n, device=dev, **common, **obs)
    eager = rt.BatchedEnv(table, n, device=dev, **common, **obs)
    general = rt.BatchedEnv(table, n, device=dev, path="general", track_visited=True, **common)
    cpu = rt.BatchedEnv(table, n, **common, **obs)
    pool = rl.CpuEnvPool(table, n, threads=2, **common)
    pool.reset()
    assert captured.capturable and not general.capturable
    graph = captured.capture(K, final=True)
    goals = revisits = 0
    for r in range(R):
        out = graph.replay()
        for k in range(K):
            want = cpu.step(final=True)
            for env in (eager, general):
                got = env.step(final=True)
                for f in STEP:
                    assert torch.equal(getattr(got, f).cpu(), getattr(want, f)), (r, k, f, env.core.fast)
            b = pool.step()
            order = np.argsort(np.asarray(b.env_ids))
            assert np.array_equal(np.asarray(b.states)[order], want.states.numpy().view(np.uint64)), (r, k)
            assert np.array_equal(np.asarray(b.reward)[order], want.reward.numpy()), (r, k)
            assert np.array_equal(np.asarray(b.goal)[order], want.goal.numpy()), (r, k)
            for f in STEP:
                if f != "revisited":
                    assert torch.equal(getattr(out, f)[k].cpu(), getattr(want, f)), (r, k, f)
            assert torch.equal(out.revisited[k].cpu(), want.revisited), (r, k)
            expect_obs_equal(out.obs[k], want.obs)
            if actions:
                expect_rows_equal(out.obs_actions[k], want.actions, (r, k))
                for x, y in zip(per_domain(out.action_visited[k]), per_domain(want.action_visited), strict=True):
                    assert torch.equal(x.cpu(), y), (r, k)
            goals += int(want.goal.sum())
            revisits += int(want.revisited.sum())
        for env in (captured, eager, general):
            assert torch.equal(env.visited.cpu(), cpu.visited) and torch.equal(env.visited_count.cpu(), cpu.visited_count)
            assert torch.equal(env.views, captured.views) or env is general
    assert goals and revisits  # goals over derived atoms were reached, states revisited
    for env in (captured, eager, general):
        env.check_errors()


# torch's CUDA graph trees capture an empty graph when their manager starts (CUDAGraphTreeManager.__init__), which warns
@pytest.mark.needs_gil
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
@pytest.mark.torch_compile
def test_compiled_steps_equal_eager():
    """torch.compile of K steps over a table with axioms: one graph without breaks, run as CUDA graphs (no cudagraph
    skip), equal to the eager steps."""
    from torch._dynamo.utils import counters
    from test_rl_graphs import compile_mode, k_steps

    dev = gpu_device()
    table = table_of("linking")
    a = rt.BatchedEnv(table, 64, device=dev, seed=3, **CFG)
    b = rt.BatchedEnv(table, 64, device=dev, seed=3, **CFG)
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(k_steps(4), fullgraph=True, **compile_mode())
    for t in range(5):
        assert torch.equal(compiled(a), k_steps(4)(b)), t
        for f in ("states", "task_ids", "steps", "draws", "counts", "views", "count", "at_goal"):
            assert torch.equal(getattr(a, f), getattr(b, f)), (t, f)
    assert counters["inductor"]["cudagraph_skips"] == 0, dict(counters["inductor"])
    a.check_errors()


def test_jax_device_steps_equal_cpu_steps():
    gpu_device()
    jax = pytest.importorskip("jax")
    rj = pytest.importorskip("mymyr.rl.jax")
    try:
        if not jax.devices("gpu"):
            pytest.skip("JAX sees no GPU")
    except RuntimeError:
        pytest.skip("JAX sees no GPU")
    table = table_of("linking")
    dev, cpu = rj.Env(table, device="cuda", **CFG), rj.Env(table, device="cpu", **CFG)
    assert dev.fast
    step_dev, step_cpu = jax.jit(dev.step), jax.jit(cpu.step)
    key = jax.random.key(1)
    s_dev, _ = dev.reset(key, 16)
    s_cpu, _ = cpu.reset(key, 16)
    for k in range(10):
        s_dev, _, r_dev, t_dev, _, _ = step_dev(key, s_dev)
        s_cpu, _, r_cpu, t_cpu, _, _ = step_cpu(key, s_cpu)
        assert np.array_equal(rj.as_words(s_dev.states), rj.as_words(s_cpu.states)), k
        assert np.array_equal(np.asarray(r_dev), np.asarray(r_cpu)), k
        assert np.array_equal(np.asarray(t_dev), np.asarray(t_cpu)), k
    dev.check_errors()
