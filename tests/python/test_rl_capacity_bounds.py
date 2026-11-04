"""capacity_bounds as obs_capacities: random rollouts of the environments over the IPC 2023 train problems
(tests/data/ipc), with encoded observations, their action rows (no-op drop included) and per-environment goals, never
overflow the bounds, and every observation stays within them per relation and per field, on the CPU and on a device.

The device runs need a GPU (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_rl_capacity_bounds.py
"""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT
from test_rl_table import gpu_device

torch = pytest.importorskip("torch")
rt = pytest.importorskip("mymyr.rl.torch")

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the PDDL front end", allow_module_level=True)

IPC = ROOT / "tests" / "data" / "ipc"
DOMAINS = sorted(p.name for p in IPC.iterdir() if (p / "train" / "domain.pddl").exists())


def ipc_table(name, problems=2):
    train = IPC / name / "train"
    files = sorted(train.glob("p*.pddl"), key=lambda p: p.stat().st_size)[:problems]
    return rl.TaskTable([mymyr.Task.from_pddl(train / "domain.pddl", p, atoms="dense") for p in files])


def check_within(obs, bounds, names):
    sizes = obs.sizes
    assert not sizes["overflow"], sizes
    counts = obs.relation_counts
    counts = (counts.cpu().numpy() if hasattr(counts, "cpu") else np.asarray(counts))[: obs.num_graphs].sum(axis=0)
    for name, n in zip(names, counts.tolist()):
        assert n <= bounds.relation_tuples[name], (name, n, bounds.relation_tuples[name])
    for key in ("entities", "args", "targets"):
        if key in bounds:
            assert sizes[key] <= bounds[key], (key, sizes[key], bounds[key])
    return int(counts.sum())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("name", DOMAINS)
def test_rollouts_stay_within_the_bounds(name, device):
    dev = torch.device("cpu") if device == "cpu" else gpu_device()
    table = ipc_table(name)
    n, steps = 16, 12
    encoder = encoding.FlatRelationEncoder(table, target_sources=("goal", "action"))
    bounds = encoder.capacity_bounds(table, n, drop_noops=True)
    assert not bounds.unbounded
    env = rt.BatchedEnv(table, n, device=dev, seed=5, max_steps=8, goals=True, observation=encoder,
                        obs_capacities=bounds, obs_actions=True, obs_drop_noops=True,
                        task_ids=np.arange(n, dtype=np.int32) % len(table.tasks))
    tuples = 0
    for _ in range(steps):
        out = env.step()
        tuples += check_within(out.obs, bounds, encoder.relation_names)
        assert int(out.actions.status.cpu().max()) == 0
        assert int(out.actions.offsets[-1]) <= bounds["successors"]
    assert tuples > 0
    env.check_errors() if hasattr(env, "check_errors") else None
