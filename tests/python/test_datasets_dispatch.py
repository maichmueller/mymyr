"""mymyr.datasets entry points over many tasks and the device dispatch, on the CPU (every build).

- state_spaces over a TaskTable or a sequence of tasks equals the instance pool (generate_many) and state_space per task;
- generalized_state_space takes a TaskTable;
- device= on a CPU build raises ValueError (CUDA builds: tests/python/test_cuda_state_space.py); symmetry pruning stays on the
  CPU;
- the float64 arrays (costs, cost goal distances) are exported like the others (NumPy, dlpack, torch);
- a sampler's id views outlive the sampler.
"""

import numpy as np
import pytest

import mymyr._core
from mymyr import datasets, rl
from conftest import text_task

HAS_CUDA = hasattr(mymyr._core, "_cuda")


@pytest.fixture(scope="module")
def tasks():
    return [text_task("gripper__prob05", atoms="frozen") for _ in range(2)]


def test_state_spaces_over_tables_and_sequences(tasks):
    table = rl.TaskTable(tasks)
    pool = [r.space for r in datasets.generate_many(tasks, remove_if_unsolvable=False)]
    for spaces in (datasets.state_spaces(table, remove_if_unsolvable=False),
                   datasets.state_spaces(tasks, threads=2, remove_if_unsolvable=False)):
        assert len(spaces) == len(tasks)
        for s, p, t in zip(spaces, pool, tasks):
            assert s.task is t
            one = datasets.state_space(t, remove_if_unsolvable=False)
            for k, v in one.arrays().items():
                assert np.array_equal(s.arrays()[k], v) and np.array_equal(p.arrays()[k], v), k
    small = datasets.state_spaces(table, max_states=10)
    assert small == [None, None]
    g = datasets.generalized_state_space(table, remove_if_unsolvable=False)
    assert len(g.spaces) == 2


def test_device_dispatch_errors(tasks):
    with pytest.raises(ValueError, match="symmetry"):
        datasets.state_space(tasks[0], device=0, symmetry_pruning=True)
    if HAS_CUDA:
        pytest.skip("a CUDA build: the device path is tests/python/test_cuda_state_space.py")
    with pytest.raises(ValueError, match="CUDA"):
        datasets.state_space(tasks[0], device=0)
    with pytest.raises(ValueError, match="CUDA"):
        datasets.state_spaces(tasks, device=0)


def test_float64_exports():
    task = text_task("depot__p02", atoms="frozen")
    s = datasets.state_space(task, remove_if_unsolvable=False)
    a = s.arrays()
    d = a["cost_goal_distances"]
    assert d.dtype == np.float64 and d.shape == (s.num_states,) and not d.flags.writeable
    raw = s.arrays("dlpack")["cost_goal_distances"]
    assert np.array_equal(np.from_dlpack(raw), d)
    torch = pytest.importorskip("torch")
    t = s.arrays("torch")["cost_goal_distances"]
    assert t.dtype == torch.float64 and np.array_equal(t.numpy(), d)


def test_sampler_views_outlive_the_sampler():
    s = datasets.state_space(text_task("gripper__prob05", atoms="frozen"))
    sampler = datasets.StateSpaceSampler(s, seed=1)
    ids = sampler.states_n_steps_from_goal(2)
    want = np.array(ids)
    del sampler
    assert np.array_equal(ids, want) and len(ids) > 0
    with pytest.raises(TypeError):
        datasets.StateSpaceSampler(object())
