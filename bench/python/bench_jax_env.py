"""Environment steps per second of the functional JAX environment inside jit, against the native flat
rollout (build/.../mymyr_device_env_rollout_bench: cuda::DeviceEnv in a C++ loop, the same config and outputs).

  - scan: ``jax.jit`` of a ``lax.scan`` over T steps of ``Env.step(key, state)`` (the random policy of the
    counter-based RNG; the device fast path runs in XLA FFI calls on XLA's stream). The scan carries the state and a
    reward sum; the step writes reward, terminated, truncated, goal, invalid and schema every step, and with --labels /
    --final also the bindings and final states (as mymyr_device_env_rollout_bench --labels --final);
  - loop: the jitted single step called from Python T times (the dispatch cost per step).

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python bench/python/bench_jax_env.py [--envs 16384,65536] [--steps 200] [--labels] [--final]

One JSON line per (task, N, mode): the median of --reps timed runs after a warmup (compilation excluded), with the
XLA_FLAGS in effect. Command buffers: the environment's FFI targets are command-buffer compatible, so XLA runs the
scan's body as a CUDA graph when its command buffers cover it (``--xla_gpu_graph_min_graph_size=1`` for a body as
small as this one: the step and the reward sum; ``--xla_gpu_enable_command_buffer=`` turns command buffers off).
"""

import argparse
import json
import os
import pathlib
import statistics
import time

import jax
import jax.numpy as jnp

import mymyr
import mymyr.rl.jax as rj

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS = ["miconic__s7-4", "blocks__probBLOCKS-8-0", "gripper__prob05", "sokoban-opt08-strips__p14"]


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--envs", default="16384,65536")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--modes", default="scan,loop")
    a = ap.parse_args()
    key = jax.random.key(1)
    for name in a.tasks.split(","):
        task = mymyr.Task.from_text(str(ROOT / "tests/data/tasks" / f"{name}.txt"), atoms="frozen")
        for n in map(int, a.envs.split(",")):
            env = rj.Env(task, device="cuda", max_steps=a.max_steps, labels=a.labels, final_state=a.final)
            common = {"task": name, "envs": n, "fast": env.fast, "labels": a.labels, "final": a.final,
                      "xla_flags": os.environ.get("XLA_FLAGS", "")}  # fmt: skip
            if "scan" in a.modes:

                @jax.jit
                def rollout(state):
                    def body(carry, _):
                        s, acc = carry
                        s, obs, reward, terminated, truncated, info = env.step(key, s)
                        return (s, acc + reward), None

                    zero = jnp.zeros(state.count.shape, jnp.float32)
                    return jax.lax.scan(body, (state, zero), None, length=a.steps)[0]

                state, _ = env.reset(key, n)
                state, _ = jax.block_until_ready(rollout(state))  # compile and warm up
                box = {"s": state}

                def run():
                    box["s"], acc = rollout(box["s"])
                    return acc

                t = timed(run, a.reps)
                env.check_errors()
                print(json.dumps({**common, "mode": "scan", "steps_per_s": n * a.steps / t,
                                  "us_per_step": t / a.steps * 1e6}), flush=True)  # fmt: skip
            if "loop" in a.modes:
                step = jax.jit(env.step)
                state, _ = env.reset(key, n)
                for _ in range(5):
                    state = step(key, state)[0]
                box = {"s": state}

                def run_loop():
                    s = box["s"]
                    for _ in range(a.steps):
                        s = step(key, s)[0]
                    box["s"] = s
                    return s.states

                t = timed(run_loop, a.reps)
                env.check_errors()
                print(json.dumps({**common, "mode": "loop", "steps_per_s": n * a.steps / t,
                                  "us_per_step": t / a.steps * 1e6}), flush=True)  # fmt: skip
            del env


if __name__ == "__main__":
    main()
