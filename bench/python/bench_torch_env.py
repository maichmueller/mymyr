"""Environment steps per second of mymyr.rl.torch on the device.

  - batched: BatchedEnv.step() in a loop, the random policy of the counter-based RNG (lifted, canonical order, the
    device fast path; each step also returns rewards, flags and labels);
  - graph: BatchedEnv.capture(K) (a CUDA graph of K such steps, their outputs stacked), replayed steps / K times;
  - compiled: torch.compile(mode="reduce-overhead", fullgraph=True) of a loop of K steps returning the stacked rewards
    and flags (CUDA graph trees; the cudagraphs backend without triton), called steps / K times;
  - torchrl: PlanningEnv.rollout(T, break_when_any_done=False) with its random actions (TensorDict per step, a copy of
    the observation, TorchRL's resets of finished environments); each step replays the env's captured step graph;
  - torchrl_eager: the same with PlanningEnv(graph=False) (eager steps).

    env CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python bench/python/bench_torch_env.py [--envs 16384,65536] [--steps 200]
        [--modes batched,graph,compiled,torchrl,torchrl_eager] [--graph-steps K]

One JSON line per (task, N, mode): the median of --reps timed loops after a warmup (us_per_step: wall time between two
synchronizations), and host_us_per_step: the host's time to enqueue one step, from a burst of 40 steps issued after a
synchronization (median of 7; the launch queue does not fill in 40 steps, so it does not block the host).
"""

import argparse
import json
import pathlib
import statistics
import time

import torch

import mymyr
import mymyr.rl.torch as rt

ROOT = pathlib.Path(__file__).resolve().parents[2]
BURST = 40  # steps per host-time burst
TASKS = ["miconic__s7-4", "blocks__probBLOCKS-8-0", "gripper__prob05", "sokoban-opt08-strips__p14"]


def timed(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def issue_time(fn, reps):
    """The host's time to enqueue fn's work (median of reps, each after a synchronization)."""
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    return statistics.median(out)


def compile_options():
    try:
        import triton  # noqa: F401

        return dict(mode="reduce-overhead")
    except ImportError:
        return dict(backend="cudagraphs")


def k_steps(K):
    def run(env):
        rs = [env.step() for _ in range(K)]
        return torch.stack([r.reward for r in rs]), torch.stack([r.terminated | r.truncated for r in rs])

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--envs", default="16384,65536")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rollout-steps", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", default="batched,torchrl")
    ap.add_argument("--graph-steps", type=int, default=10)
    a = ap.parse_args()
    for name in a.tasks.split(","):
        task = mymyr.Task.from_text(str(ROOT / "tests/data/tasks" / f"{name}.txt"), atoms="frozen")
        for n in map(int, a.envs.split(",")):
            if "batched" in a.modes:
                env = rt.BatchedEnv(task, n, device="cuda", seed=1)
                for _ in range(10):
                    env.step()

                def run():
                    for _ in range(a.steps):
                        env.step()

                t = timed(run, a.reps)
                h = issue_time(lambda: [env.step() for _ in range(BURST)], 7) / BURST
                env.check_errors()
                print(json.dumps({"task": name, "envs": n, "mode": "batched", "fast": env.core.fast,
                                  "steps_per_s": n * a.steps / t, "us_per_step": t / a.steps * 1e6,
                                  "host_us_per_step": h * 1e6}), flush=True)  # fmt: skip
                env.close()
                del env
            K = a.graph_steps
            if "graph" in a.modes:
                env = rt.BatchedEnv(task, n, device="cuda", seed=1)
                for _ in range(10):
                    env.step()
                g = env.capture(K)
                g.replay()

                def replays():
                    for _ in range(a.steps // K):
                        g.replay()

                t = timed(replays, a.reps)
                h = issue_time(lambda: [g.replay() for _ in range(max(1, BURST // K))], 7) / (max(1, BURST // K) * K)
                env.check_errors()
                print(json.dumps({"task": name, "envs": n, "mode": "graph", "graph_steps": K,
                                  "steps_per_s": n * (a.steps // K * K) / t, "us_per_step": t / (a.steps // K * K) * 1e6,
                                  "host_us_per_step": h * 1e6}), flush=True)  # fmt: skip
                del g
                env.close()
                del env
            if "compiled" in a.modes:
                torch._dynamo.reset()
                env = rt.BatchedEnv(task, n, device="cuda", seed=1)
                for _ in range(10):
                    env.step()
                fn = torch.compile(k_steps(K), fullgraph=True, **compile_options())
                for _ in range(3):  # compile, record, replay
                    fn(env)

                def calls():
                    for _ in range(a.steps // K):
                        fn(env)

                t = timed(calls, a.reps)
                h = issue_time(lambda: [fn(env) for _ in range(max(1, BURST // K))], 7) / (max(1, BURST // K) * K)
                env.check_errors()
                print(json.dumps({"task": name, "envs": n, "mode": "compiled", "graph_steps": K,
                                  "options": compile_options(), "steps_per_s": n * (a.steps // K * K) / t,
                                  "us_per_step": t / (a.steps // K * K) * 1e6, "host_us_per_step": h * 1e6}),
                      flush=True)  # fmt: skip
                del fn
                env.close()
                del env
            for mode in ("torchrl", "torchrl_eager"):
                if mode not in a.modes.split(","):
                    continue
                penv = rt.PlanningEnv(task, n, device="cuda", seed=1, max_steps=100, graph=mode == "torchrl")
                penv.rollout(5, break_when_any_done=False)
                t = timed(lambda: penv.rollout(a.rollout_steps, break_when_any_done=False), a.reps)
                print(json.dumps({"task": name, "envs": n, "mode": mode, "graph": penv.graph,
                                  "steps_per_s": n * a.rollout_steps / t, "us_per_step": t / a.rollout_steps * 1e6}),
                      flush=True)  # fmt: skip
                penv.close()
                del penv
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
