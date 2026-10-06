"""Shared test setup. JAX and torch (when installed) stay on the CPU unless the caller sets CUDA_VISIBLE_DEVICES (and
JAX_PLATFORMS) explicitly: only test_cuda.py uses a GPU, and it skips otherwise."""

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "tests/data/pddl/blocks"
TASKS = ROOT / "tests/data/tasks"

# Suite tasks (normalized text format) with a mix of features, small enough for exhaustive label checks.
SMALL_TASKS = [
    "blocks__probBLOCKS-8-0",
    "gripper__prob05",
    "logistics00__probLOGISTICS-6-1",
    "depot__p02",
    "miconic-simpleadl__s10-2",  # conditional effects
    "openstacks-opt08-adl__p03",  # axioms
    "philosophers__p03-phil4",  # axioms, derived goal
]


def text_task(name, **options):
    import mymyr

    return mymyr.Task.from_text(str(TASKS / f"{name}.txt"), **options)


@pytest.fixture(scope="session")
def blocks():
    import mymyr

    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl")


def walk(task, steps=20, seed=0, walks=2):
    """States of seeded random walks (the initial state first)."""
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(walks):
        s = task.initial_state
        out.append(s)
        for _ in range(steps):
            succ = task.successor_states(s)
            if not succ:
                break
            s = rng.choice(succ)
            out.append(s)
    return out
