"""mymyr.cuda: conditional effects and axiom strata on the device. The device BrFS of tasks with conditional
effects (forall included), axioms and derived goals equals the CPU BrFS, with nothing of it on the CPU fallback (the
statistics say so). The byte equality of the expansion is the C++ test's (tests/cuda/test_device_conditional.cpp).

Runs only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_cuda_conditional.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest  # noqa: E402

import mymyr  # noqa: E402
import mymyr.search  # noqa: E402
from conftest import text_task  # noqa: E402

mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
if not mc.available():
    pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)", allow_module_level=True)

# (task, conditional effects, axioms)
TASKS = [
    ("miconic-simpleadl__s10-2", True, False),
    ("openstacks-opt08-adl__p03", False, True),
    ("philosophers__p03-phil4", False, True),  # derived goal
]


@pytest.fixture(scope="module")
def ctx():
    c = mc.Context(0, max_bytes=2 << 30)
    yield c
    c.synchronize()


@pytest.mark.parametrize("name,ce,axioms", TASKS)
@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
def test_device_brfs_runs_conditional_effects_and_axioms(ctx, name, ce, axioms, atoms):
    task = text_task(name, atoms=atoms)
    r = mc.brfs(task, ctx=ctx, fingerprint=True)
    c = mymyr.search.brfs(task, threads=2, fingerprint=True)
    assert (r.states, r.generated, r.goal_states, r.fingerprint) == (c.states, c.generated, c.goal_states, c.fingerprint)
    s = r.stats
    assert s["host_schemas"] == 0 and s["host_ce_schemas"] == 0
    assert (s["ce_schemas"] > 0) == ce
    assert s["device_axioms"] == (1 if axioms else 0)
    assert s["host_axiom_ms"] == 0.0
    assert s["axiom_ms"] >= 0.0 and s["axiom_reruns"] >= 0
    assert all(isinstance(v, (int, float)) for v in s.values())


def test_stop_at_goal_over_derived_goals(ctx):
    task = text_task("philosophers__p03-phil4")
    g = mc.brfs(task, ctx=ctx, stop_at_goal=True)
    c = mymyr.search.brfs(task, stop_at_goal=True)
    assert g.solved and c.solved
    assert [a.label for a in g.plan] == [a.label for a in c.plan]
    assert g.states == c.states
