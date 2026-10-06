"""Environment steps as CUDA graphs and XLA command buffers, byte for byte equal to the eager steps.

- torch: BatchedEnv.capture (torch.cuda.CUDAGraph) of K steps, with the random policy, a policy, next_task_ids, final
  states and per-env goals, replayed on the table instances and single tasks == the eager steps == the host env; set_seed,
  resets and stream changes between replays; torch.cuda.graph around BatchedEnv calls (reset with a mask, random
  actions, steps); torch.compile(mode="reduce-overhead") of a K-step loop: no graph break, run as CUDA graphs, equal
  to eager; the general path: not capturable (capture() refuses, torch.compile keeps its ops out of the graphs), the
  expand ops tagged cudagraph_unsafe.
- JAX (a process per XLA_FLAGS setting): the jitted lax.scan of env.step (single tasks and table instances, fast and
  general path) and mctx's searches with XLA's command buffers (by default, the scan's body as a CUDA graph, the whole
  loop as one) == without them == the CPU env; the thunk dumps show the fast path's calls inside the command buffers
  and the general path's outside.

GPU only (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_graphs.py
"""

import numpy as np
import pytest

from mymyr import rl
from conftest import text_task
from test_rl_table import MASK_SETS, ROT, SETS, env_arrays, gpu_device, host_rollout, sets, step_arrays  # noqa: F401

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

FIELDS = ("reward", "terminated", "truncated", "goal", "invalid", "schema", "binding", "final_states", "count")
CFG = dict(max_steps=6, goal_reward=1.0, dead_end="no_successors", dead_end_reward=-2.0)


def stacked(results, field):
    return torch.stack([getattr(r, field) for r in results])


def expect_equal_envs(a, b, where):
    for f in ("states", "task_ids", "steps", "draws", "counts", "views", "count", "goal_pos", "goal_neg"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None and y is None) or torch.equal(x, y), (where, f)


def expect_equal_results(rollout, results, where):
    for f in FIELDS:
        assert torch.equal(getattr(rollout, f), stacked(results, f)), (where, f)


# ------------------------------------------------------------------------------------------------ torch: CUDAGraph


@pytest.mark.parametrize("name", MASK_SETS)
def test_captured_rollout_equals_host_on_tables(sets, name):
    """K random steps per graph with next_task_ids (rewritten between replays) and per-env goals == the host env."""
    dev = gpu_device()
    tasks = sets[name]
    t = rl.TaskTable(tasks)
    I, n, K, R = len(tasks), 96, 3, 5
    ids = (np.arange(n) * 5 % I).astype(np.int32)
    ref = host_rollout(t, ids, K * R, goals=True, next_ids=lambda s: ROT(n, I)(s // K), **CFG)
    for launch in ("widest", "per_bucket"):
        env = rt.BatchedEnv(t, n, task_ids=ids, goals=True, seed=5, device=dev, **CFG)
        env.set_launch(launch)
        assert env.capturable
        nxt = torch.zeros(n, dtype=torch.int32, device=dev)
        g = env.capture(K, next_task_ids=nxt, final=True)
        for r in range(R):
            nxt.copy_(torch.from_numpy(ROT(n, I)(r)))
            out = g.replay()
            for k in range(K):
                want = ref[r * K + k + 1]
                got = step_arrays(rt.StepResult(*(getattr(out, f)[k] for f in FIELDS)))
                for f, v in got.items():
                    assert np.array_equal(v, want[f]), (launch, r, k, f)
            a = env_arrays(env)
            for f, v in a.items():
                assert np.array_equal(v, ref[(r + 1) * K][f]), (launch, r, f)
            assert np.array_equal(env.goal_pos.cpu().numpy().view(np.uint64), ref[(r + 1) * K]["goal_pos"])
        env.check_errors()


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "miconic__s7-4"])
def test_captured_rollout_equals_eager(name):
    """A policy's graph (given actions, invalid ones too) and the random policy's, with set_seed, masked resets and a
    side stream between replays: byte for byte the eager steps."""
    dev = gpu_device()
    task = text_task(name, atoms="frozen")
    n, K = 200, 4

    def policy(e):  # a function of the state: indices in [0, count], count itself invalid
        return (e.states.sum(1) + e.steps) % (e.count.to(torch.int64) + 1)

    for pol in (None, policy, lambda e: e.random_actions(3)):
        a = rt.BatchedEnv(task, n, device=dev, seed=11, **CFG)
        b = rt.BatchedEnv(task, n, device=dev, seed=11, **CFG)
        g = a.capture(K, pol, final=True)
        side = torch.cuda.Stream(dev)
        for r in range(6):
            if r == 2:
                a.set_seed(4)
                b.set_seed(4)
            if r == 3:
                mask = torch.arange(n, device=dev) % 3 == 1
                a.reset(mask)
                b.reset(mask)
            if r == 4:  # a replay on another stream
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):
                    out = g.replay()
                torch.cuda.current_stream(dev).wait_stream(side)
            else:
                out = g.replay()
            acts, res = [], []
            for _ in range(K):
                act = None if pol is None else pol(b)
                acts.append(act)
                res.append(b.step(act, final=True))
            expect_equal_results(out, res, (name, r))
            if pol is not None:
                assert torch.equal(out.action, torch.stack(acts)), (name, r)
            expect_equal_envs(a, b, (name, r))
        a.check_errors()


def test_torch_cuda_graph_around_env_calls():
    """torch.cuda.graph around reset (a mask), random_actions and steps == the same calls eagerly."""
    dev = gpu_device()
    task = text_task("blocks__probBLOCKS-8-0", atoms="frozen")
    n = 128
    a = rt.BatchedEnv(task, n, device=dev, seed=2, autoreset=False, max_steps=5)
    b = rt.BatchedEnv(task, n, device=dev, seed=2, autoreset=False, max_steps=5)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        a.reset(done)
        r = a.step(a.random_actions(5))
        done.copy_(r.terminated | r.truncated)
    reward = r.reward
    done_b = torch.zeros(n, dtype=torch.bool, device=dev)
    for t in range(12):
        g.replay()
        b.reset(done_b)
        q = b.step(b.random_actions(5))
        done_b = q.terminated | q.truncated
        assert torch.equal(reward, q.reward) and torch.equal(done, done_b), t
        expect_equal_envs(a, b, t)


def test_capture_on_the_general_path_and_the_cpu_is_refused():
    dev = gpu_device()
    task = text_task("blocks__probBLOCKS-8-0", atoms="frozen")
    env = rt.BatchedEnv(task, 16, device=dev, path="general")
    assert not env.capturable and "general path" in env.core.capture_unsupported
    with pytest.raises(ValueError, match="general path"):
        env.capture(2)
    with pytest.raises(ValueError, match="CPU"):
        rt.BatchedEnv(task, 16).capture(2)
    for op in ("expand", "expand_flat", "step_sync", "refresh_sync"):
        assert torch.Tag.cudagraph_unsafe in getattr(torch.ops.mymyr, op).default.tags, op
    for op in ("reset", "refresh", "step", "random_actions"):
        assert torch.Tag.cudagraph_unsafe not in getattr(torch.ops.mymyr, op).default.tags, op


# ------------------------------------------------------------------------------------------------ TorchRL


@pytest.mark.parametrize("name", ["gripper", "blocks__probBLOCKS-8-0"])
def test_planning_env_graph_equals_eager(sets, name):
    """PlanningEnv steps replaying the captured step == eager steps (graph=False): rollouts with random actions and a
    policy, TorchRL's resets, a reset carrying task_id and goals, set_seed between rollouts; check_env_specs."""
    pytest.importorskip("torchrl")
    from tensordict import TensorDict
    from torchrl.envs.utils import check_env_specs

    dev = gpu_device()
    if name in SETS:
        t, I = rl.TaskTable(sets[name]), len(sets[name])
    else:
        t, I = text_task(name, atoms="frozen"), 1
    n = 64
    ids = torch.as_tensor(np.arange(n) * 5 % I, dtype=torch.int32)
    kw = dict(task_ids=ids, seed=3, max_steps=5, goal_reward=1.0, max_actions=6, goals=True, device=dev)
    envs = [rt.PlanningEnv(t, n, graph=g, **kw) for g in (True, False)]
    assert envs[0].graph and not envs[1].graph
    assert not rt.PlanningEnv(t, 4, task_ids=ids[:4], graph=True).graph  # the CPU
    check_env_specs(envs[0], seed=3)
    check_env_specs(envs[1], seed=3)

    def policy(td):
        return td.set("action", (td["state"].sum(-1) + td["count"]) % 7)  # invalid actions (>= count) included

    outs = []
    for env in envs:
        env.set_seed(9)
        env.reset(TensorDict({"task_id": ids.to(torch.int64)}, batch_size=[n]))
        a = env.rollout(9, break_when_any_done=False, auto_reset=False, tensordict=env.reset())
        mask = torch.arange(n, device=dev) % 4 == 1
        gp, gn = env.batched.goal_pos.clone(), env.batched.goal_neg.clone()
        gp[:, 0] ^= 1  # another goal for the reset rows
        new = TensorDict({"_reset": mask.unsqueeze(-1), "task_id": ((ids.to(dev) + 1) % I).to(torch.int64),
                          "goal_pos": gp, "goal_neg": gn}, batch_size=[n], device=dev)  # fmt: skip
        td = env.reset(new)
        env.set_seed(2)
        b = env.rollout(11, policy, break_when_any_done=False, auto_reset=False, tensordict=td)
        outs.append((a, b, env.batched.states.clone(), env.batched.draws.clone(), env.batched.goal_pos.clone()))
    for x, y in zip(*outs):
        assert (x == y).all(), name



def compile_mode():
    """reduce-overhead (inductor, needs triton), else the cudagraphs backend (CUDA graph trees without inductor)."""
    try:
        import triton  # noqa: F401

        return dict(mode="reduce-overhead")
    except ImportError:
        return dict(backend="cudagraphs")


def k_steps(K):
    def run(env):
        total = torch.zeros((), device=env.device)
        for _ in range(K):
            r = env.step(final=True)
            total = total + r.reward.sum() + r.final_states.sum() + r.binding.sum()
        return total

    return run


# torch's CUDA graph trees capture an empty graph when their manager starts (CUDAGraphTreeManager.__init__), which warns
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
@pytest.mark.parametrize("path", ["auto", "general"])
def test_reduce_overhead_rollout_equals_eager(path):
    """torch.compile(mode="reduce-overhead") of K steps: one graph, no break; on the fast path it runs as CUDA graphs
    (no cudagraph skip); on the general path the step_sync ops stay out of them; eager results either way. Without
    triton (the cudagraphs backend, which ignores the cudagraph_unsafe tag) the general path refuses the capture."""
    from torch._dynamo.utils import counters

    dev = gpu_device()
    task = text_task("blocks__probBLOCKS-8-0", atoms="frozen")
    a = rt.BatchedEnv(task, 256, device=dev, seed=7, path=path, **CFG)
    b = rt.BatchedEnv(task, 256, device=dev, seed=7, path=path, **CFG)
    torch._dynamo.reset()
    ex = torch._dynamo.explain(k_steps(4))(rt.BatchedEnv(task, 256, device=dev, seed=7, path=path, **CFG))
    assert ex.graph_break_count == 0 and ex.graph_count == 1
    torch._dynamo.reset()
    counters.clear()
    if path == "general" and "backend" in compile_mode():
        # the cudagraphs backend (no inductor without triton) ignores the cudagraph_unsafe tag and captures everything:
        # the general path's step refuses the capture
        fn = torch.compile(k_steps(4), fullgraph=True, **compile_mode())
        with pytest.raises(RuntimeError, match="general path"):
            for _ in range(3):  # the warm-up run, then the recording (the capture)
                fn(a)
        return
    compiled = torch.compile(k_steps(4), fullgraph=True, **compile_mode())
    for t in range(6):
        if t == 3:
            a.set_seed(13)
            b.set_seed(13)
        x = compiled(a)
        y = k_steps(4)(b)
        assert torch.equal(x, y), t
        expect_equal_envs(a, b, t)
    skips = counters["inductor"]["cudagraph_skips"]
    if path == "auto":
        assert skips == 0, dict(counters["inductor"])
    a.check_errors()


# ------------------------------------------------------------------------------------------------ JAX: command buffers

# A jitted lax.scan of env.step, run twice, on the CPU and the GPU: even steps take env.random_actions, odd steps a
# policy of the state (invalid actions included); next_task_ids over a table; all outputs and the final states saved.
SCAN = """
import sys
import jax, jax.numpy as jnp, numpy as np
import mymyr
from mymyr import rl
import mymyr.rl.jax as rj
tests, name, path, out = sys.argv[1:5]
sys.path.insert(0, tests)
from conftest import text_task
from test_rl_table import MASK_SETS, ROT, SETS

n, K = 96, 7
if name in SETS:
    dom, probs = SETS[name]
    table = rl.TaskTable([mymyr.Task.from_pddl(dom, p, atoms="frozen") for p in probs])
    I, goals = len(probs), name in MASK_SETS
else:
    table, I, goals = text_task(name, atoms="frozen"), 1, False
ids = (np.arange(n) * 5 % I).astype(np.int32)
nxt = jnp.stack([jnp.asarray(ROT(n, I)(s)) for s in range(K)])
res = {}
for device in (jax.devices("cpu")[0], jax.devices("gpu")[0]):
    env = rj.Env(table, device=device, goals=goals, path=path, max_steps=6, goal_reward=1.0, dead_end_reward=-2.0)
    state, _ = env.reset(None, jnp.asarray(ids))

    @jax.jit
    def roll(state):
        def body(st, xs):
            k, nx = xs
            policy = (st.states.sum(-1) + st.steps.astype(jnp.uint32)) % (st.count.astype(jnp.uint32) + 1)
            act = jnp.where(k % 2 == 0, env.random_actions(5, st), policy.astype(jnp.int32))
            st, obs, r, term, trunc, info = env.step(5, st, act, next_task_ids=nx if I > 1 else None)
            return st, (act, r, term, trunc, info["goal"], info["invalid"], info["schema"], info["binding"],
                        info["final_state"], st.states, st.task_id, st.steps, st.draws, st.count)
        return jax.lax.scan(body, state, (jnp.arange(K), nxt))

    final, ys = roll(state)
    final, zs = roll(final)
    env.check_errors()
    for k, x in enumerate(jax.tree_util.tree_leaves((final, ys, zs))):
        res[f"{device.platform}_{k:02d}"] = np.asarray(x)
    res[f"{device.platform}_fast"] = np.asarray(device.platform != "cpu" and env.fast)
np.savez(out, **res)
"""

XLA = {  # XLA_FLAGS: without command buffers, the default, the scan's body as a command buffer, the loop as one
    "off": "--xla_gpu_enable_command_buffer=",
    "default": "",
    "body": "--xla_gpu_graph_min_graph_size=1",
    "loop": "--xla_gpu_graph_min_graph_size=1 --xla_gpu_enable_command_buffer="
    "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL,DYNAMIC_SLICE_FUSION",
}


def custom_call_in_command_buffer(dump, fn):
    """Whether the thunk sequence of the jitted function ``fn`` runs a custom call inside a command buffer (a
    kCustomCall nested in a kCommandBuffer)."""
    import re

    found = False
    for f in dump.glob(f"*jit_{fn}.thunk_sequence_after_thunk_passes.txt"):
        stack = []
        for line in f.read_text().splitlines():
            m = re.match(r"^(\s*)\d+: (k\w+)", line)
            if not m:
                continue
            depth = len(m.group(1))
            stack = [d for d in stack if d < depth]
            if m.group(2) == "kCommandBuffer":
                stack.append(depth)
            elif m.group(2) == "kCustomCall" and stack:
                found = True
    return found


def run_xla_modes(tmp_path, script, fn, *args):
    """Runs ``script`` (argv: the tests directory, *args, the output .npz) in a process per XLA mode; checks that the
    jitted function ``fn`` runs its custom calls inside command buffers exactly where the device env is on the fast
    path and command buffers cover them (always with 'body' and 'loop', never with 'off', by size by default).
    Returns {mode: the saved arrays}."""
    import os
    import subprocess
    import sys

    from conftest import ROOT

    gpu_device()
    pytest.importorskip("jax")
    runs = {}
    for mode, flags in XLA.items():
        dump = tmp_path / mode
        env = dict(os.environ, XLA_FLAGS=f"{flags} --xla_dump_to={dump}", CUDA_VISIBLE_DEVICES="0",
                   JAX_PLATFORMS="cuda,cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")  # fmt: skip
        out = tmp_path / f"{mode}.npz"
        r = subprocess.run([sys.executable, "-c", script, str(ROOT / "tests/python"), *args, str(out)], env=env,
                           capture_output=True, text=True, timeout=900)  # fmt: skip
        assert r.returncode == 0, r.stderr[-4000:]
        runs[mode] = dict(np.load(out))
        fast = bool(runs[mode]["gpu_fast"])
        inside = custom_call_in_command_buffer(dump, fn)
        if mode != "default":  # (by default, XLA makes command buffers of bodies above a size)
            assert inside == (fast and mode != "off"), mode
        else:
            assert fast or not inside
    return runs


@pytest.mark.parametrize(
    "name, path",
    [("blocks__probBLOCKS-8-0", "auto"), ("gripper", "auto"), ("logistics", "auto"), ("gripper", "general")],
)
def test_jax_scan_with_command_buffers_equals_without(name, path, tmp_path):
    """The jitted scan (random and given actions, next_task_ids, goals, labels, final states), twice: the same bytes on
    the CPU and on the GPU without command buffers, by default, with the scan's body as a CUDA graph and with the whole
    loop as one; the fast path's step is inside the command buffers, the general path's (it synchronizes) outside."""
    if name in SETS and not all(p.is_file() for p in SETS[name][1]):
        pytest.skip(f"the table instances of {name} are missing (set MYMYR_FORK_DATA)")
    runs = run_xla_modes(tmp_path, SCAN, "roll", name, path)
    for mode, a in runs.items():
        assert bool(a["gpu_fast"]) == (path == "auto"), mode
    keys = sorted(k[4:] for k in runs["off"] if k.startswith("cpu_") and k != "cpu_fast")
    assert len(keys) == 10 + 2 * 14
    for mode, a in runs.items():
        for k in keys:
            if k not in ("06", "07"):  # counts, views: the device cache ([N, 0] on the CPU)
                assert np.array_equal(a[f"gpu_{k}"], a[f"cpu_{k}"]), (mode, k)
            assert np.array_equal(a[f"gpu_{k}"], runs["off"][f"gpu_{k}"]), (mode, k)


# A jitted mctx search (the env step inside mctx's simulation loop) on the GPU, 4 moves: actions, action weights, the
# trees' children, node values and embedded states saved.
MCTX = """
import sys
import jax, jax.numpy as jnp, numpy as np, mctx
import mymyr.rl.jax as rj
tests, name, policy, out = sys.argv[1:5]
sys.path.insert(0, tests)
from conftest import text_task

task = text_task(name)
B, K = 16, 32
env = rj.Env(task, device=jax.devices("gpu")[0], autoreset=False)
value = rj.goal_count_value(task)
recurrent = rj.mctx_recurrent_fn(env, value, num_actions=K)


@jax.jit
def search(state, key):
    root, invalid = rj.mctx_root(env, state, value, num_actions=K)
    if policy == "muzero":
        o = mctx.muzero_policy(None, key, root, recurrent, num_simulations=16, invalid_actions=invalid)
    else:
        o = mctx.gumbel_muzero_policy(None, key, root, recurrent, num_simulations=16, invalid_actions=invalid,
                                      max_num_considered_actions=8)
    t = o.search_tree
    return o.action, o.action_weights, t.children_index, t.node_values, t.embeddings.states


res = {}
state, _ = env.reset(None, B)
step = jax.jit(env.step)
for t in range(4):
    ys = search(state, jax.random.key(t))
    for k, x in enumerate(ys):
        res[f"gpu_{t}_{k}"] = np.asarray(x)
    state, obs, reward, terminated, truncated, info = step(0, state, ys[0])
    state, _ = env.reset_where(state, terminated)
env.check_errors()
res["gpu_fast"] = np.asarray(env.fast)
np.savez(out, **res)
"""


@pytest.mark.parametrize("policy", ["muzero", "gumbel"])
def test_mctx_with_command_buffers_equals_without(policy, tmp_path):
    """mctx's search with the env step as its dynamics: the same actions, weights and trees with and without command
    buffers (the step inside them)."""
    pytest.importorskip("mctx")
    runs = run_xla_modes(tmp_path, MCTX, "search", "blocks__probBLOCKS-8-0", policy)
    for mode, a in runs.items():
        assert a.keys() == runs["off"].keys() and len(a) == 4 * 5 + 1
        for k in a:
            assert np.array_equal(a[k], runs["off"][k]), (mode, k)
