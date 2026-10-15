"""Shared test setup. JAX and torch (when installed) stay on the CPU unless the caller sets CUDA_VISIBLE_DEVICES (and
JAX_PLATFORMS) explicitly: only test_cuda.py uses a GPU, and it skips otherwise."""

import json
import os
import pathlib
import sys
import tempfile

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


# Tests that load an extension module which cannot run without the GIL (torchrl and tensordict import
# torch._dynamo, which loads triton; so do torch.compile and torch._dynamo themselves) are marked
# ``needs_gil``. Loading such a module enables the GIL for the rest of the process, and the free-threading
# tests assert that it stays disabled, so a marked test runs in a child pytest process: the tests of one
# module that were selected for this run share one child, and each reports the child's outcome.
# Every other test fails if it leaves the GIL enabled, so that a missing mark is found at the test that
# causes it rather than at a free-threading test later in the session. The child runs with the GIL enabled from
# the start (PYTHON_GIL=1, which its own worker processes inherit): it exists to load such modules, and a worker that
# loaded one under ``-W error`` would die on the interpreter's warning about enabling the GIL.

ISOLATED_REPORT = "MYMYR_TEST_ISOLATED_REPORT"
_isolated_outcomes: dict[str, dict] = {}
_isolated_groups: dict[str, list[str]] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_gil: loads a module that enables the GIL; runs in a child process (tests/python/conftest.py)"
    )


def pytest_collection_modifyitems(config, items):
    if ISOLATED_REPORT in os.environ:
        return
    for item in items:
        if item.get_closest_marker("needs_gil"):
            _isolated_groups.setdefault(str(item.path), []).append(item.nodeid)


def pytest_runtest_logreport(report):
    path = os.environ.get(ISOLATED_REPORT)
    if path is None:
        return
    entry = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
    if report.skipped:
        entry["skip"] = str(report.longrepr[2]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
    elif report.failed:
        entry["text"] = report.longreprtext
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_isolated(config, path):
    fd, report = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    from python_children import run_code

    args = ["-q", "-p", "no:cacheprovider", "--rootdir", str(config.rootpath)]
    for w in config.getoption("pythonwarnings") or []:
        args += ["-W", w]
    proc = run_code("import pytest; sys.exit(pytest.main(sys.argv[1:]))", *args, *_isolated_groups[path],
                    env={**os.environ, ISOLATED_REPORT: report, "PYTHON_GIL": "1"}, timeout=None, capture_output=True, check=False)
    try:
        for line in pathlib.Path(report).read_text().splitlines():
            entry = json.loads(line)
            old = _isolated_outcomes.get(entry["nodeid"])
            if old is None or old["outcome"] == "passed":
                _isolated_outcomes[entry["nodeid"]] = entry
    finally:
        os.unlink(report)
    for nodeid in _isolated_groups[path]:
        _isolated_outcomes.setdefault(nodeid, {"outcome": "failed", "text": proc.stdout[-4000:] + proc.stderr[-4000:]})


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if ISOLATED_REPORT in os.environ or not pyfuncitem.get_closest_marker("needs_gil"):
        return None
    if pyfuncitem.nodeid not in _isolated_outcomes:
        _run_isolated(pyfuncitem.config, str(pyfuncitem.path))
    entry = _isolated_outcomes[pyfuncitem.nodeid]
    if entry["outcome"] == "failed":
        pytest.fail(f"in the child process:\n{entry['text']}", pytrace=False)
    if entry["outcome"] == "skipped":
        pytest.skip(entry["skip"])
    return True


@pytest.fixture(autouse=True)
def _gil_stays_disabled():
    if ISOLATED_REPORT in os.environ or not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        yield
        return
    yield
    if sys._is_gil_enabled():
        pytest.fail("this test enabled the GIL (it loaded a module that needs it); mark it @pytest.mark.needs_gil")
