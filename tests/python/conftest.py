"""Shared test setup. JAX and torch (when installed) stay on the CPU unless the caller sets CUDA_VISIBLE_DEVICES (and
JAX_PLATFORMS) explicitly: only test_cuda.py uses a GPU, and it skips otherwise."""

import json
import os
import pathlib
import sys
import sysconfig
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


def python_lines(file_suffix, call):
    """The number of Python lines of the source file whose path ends in `file_suffix` that `call()` executes (line
    events of the tracer): a call whose count does not grow with its input runs no Python code of that file per item."""
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if not frame.f_code.co_filename.endswith(file_suffix):
            return None
        if event == "line":
            count += 1
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        call()
    finally:
        sys.settrace(previous)
    return count


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
# the markers of tests that run in a child process, one per test file and marker, and the child's environment
ISOLATED_MARKERS = {
    "needs_gil": {"PYTHON_GIL": "1"},
    # an initialized XLA GPU client polls its streams, which invalidates a torch CUDA graph capture held meanwhile
    "jax_on_cpu": {"JAX_PLATFORMS": "cpu"},
}
_isolated_outcomes: dict[str, dict] = {}
_isolated_groups: dict[tuple[str, str, str], list[str]] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_gil: loads a module that enables the GIL; runs in a child process (tests/python/conftest.py)"
    )
    config.addinivalue_line(
        "markers", "jax_on_cpu: holds a CUDA graph capture while it waits; runs in a child process whose JAX uses only "
        "the CPU; per_case=True isolates each case (tests/python/conftest.py)"
    )


def _isolated_marker(item):
    return next((name for name in ISOLATED_MARKERS if item.get_closest_marker(name)), None)


def _isolated_key(item, marker):
    per_case = item.get_closest_marker(marker).kwargs.get("per_case", False)
    return str(item.path), marker, item.nodeid if per_case else ""


def pytest_collection_modifyitems(config, items):
    if ISOLATED_REPORT in os.environ:
        return
    for item in items:
        if marker := _isolated_marker(item):
            _isolated_groups.setdefault(_isolated_key(item, marker), []).append(item.nodeid)


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


def _run_isolated(config, key):
    path, marker, _ = key
    fd, report = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    from python_children import run_code

    args = ["-q", "-p", "no:cacheprovider", "--rootdir", str(config.rootpath)]
    if config.inipath is not None:
        args += ["-c", str(config.inipath)]
    confcutdir = config.getoption("confcutdir")
    if confcutdir is None:
        confcutdir = config.inipath.parent if config.inipath is not None else config.rootpath
    args += ["--confcutdir", str(pathlib.Path(confcutdir).resolve()), "--import-mode", config.getoption("importmode")]
    for override in config.getoption("override_ini") or []:
        args += ["-o", override]
    for w in config.getoption("pythonwarnings") or []:
        args += ["-W", w]
    group = _isolated_groups[key]
    targets = [path + "::" + nodeid.split("::", 1)[1] for nodeid in group]
    proc = run_code("import pytest; sys.exit(pytest.main(sys.argv[1:]))", *args, *targets,
                    env={**os.environ, ISOLATED_REPORT: report, **ISOLATED_MARKERS[marker]}, timeout=None,
                    capture_output=True, check=False)
    try:
        for line in pathlib.Path(report).read_text().splitlines():
            entry = json.loads(line)
            old = _isolated_outcomes.get(entry["nodeid"])
            if old is None or old["outcome"] == "passed":
                _isolated_outcomes[entry["nodeid"]] = entry
    finally:
        os.unlink(report)
    for nodeid in group:
        _isolated_outcomes.setdefault(nodeid, {"outcome": "failed", "text": proc.stdout[-4000:] + proc.stderr[-4000:]})


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    marker = None if ISOLATED_REPORT in os.environ else _isolated_marker(pyfuncitem)
    if marker is None:
        return None
    if pyfuncitem.nodeid not in _isolated_outcomes:
        _run_isolated(pyfuncitem.config, _isolated_key(pyfuncitem, marker))
    entry = _isolated_outcomes[pyfuncitem.nodeid]
    if entry["outcome"] == "failed":
        pytest.fail(f"in the child process:\n{entry['text']}", pytrace=False)
    if entry["outcome"] == "skipped":
        pytest.skip(entry["skip"])
    return True


# A free-threaded interpreter. On the GIL build the threads of the concurrency tests take turns on the GIL.
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_stays_off():
    """Whether a free-threaded interpreter still runs without the GIL (always True on the GIL build)."""
    return not FREE_THREADED or not sys._is_gil_enabled()


@pytest.fixture(autouse=True)
def _gil_stays_disabled():
    if ISOLATED_REPORT in os.environ or not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        yield
        return
    yield
    if sys._is_gil_enabled():
        pytest.fail("this test enabled the GIL (it loaded a module that needs it); mark it @pytest.mark.needs_gil")


def wide_task(tmp_path, objects=80, atoms="dense", initial_atoms=1, extra_effects=False):
    """A four-state task whose two high-slot goals require IW(2) from the initial state."""
    import mymyr

    text = f"O {objects}\nP 1\nF 2 cell\nSI 0\nFI {initial_atoms}\n"
    text += "".join(f"0 {i // objects} {i % objects}\n" for i in range(initial_atoms))
    text += f"G 2\n0 1 {objects - 1} {objects - 1}\n0 1 {objects - 1} {objects - 2}\nA 3\n"
    for i in range(2):
        a, b = -objects, -objects + i
        text += f"set{i} 0\nL 1\n0 0 {a} {b}\nE 1\n0\nL 0\nF {2 if extra_effects else 1}\n0 1 {a} {b}\n"
        if extra_effects:
            text += f"0 1 {a + 1} {b}\n"
    text += "touch 2\nL 1\n0 1 0 1\nE 1\n0\nL 0\nF 1\n0 1 0 1\n"
    path = tmp_path / "wide.txt"
    path.write_text(text)
    return mymyr.Task.from_text(path, atoms=atoms)


@pytest.fixture(params=["p01-medium", "p05-medium"])
def rovers_task(request):
    import mymyr

    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    directory = ROOT / "tests/data/ipc/rovers-ipc/test"
    return mymyr.Task.from_pddl(directory / "domain.pddl", directory / f"{request.param}.pddl", atoms="dense")


@pytest.fixture(scope="session")
def blocksworld_wide():
    """Blocksworld with 69 blocks: dense rows of more than 64 words."""
    import mymyr

    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    directory = ROOT / "tests/data/ipc/blocksworld-ipc/test"
    return mymyr.Task.from_pddl(directory / "domain.pddl", directory / "p10-medium.pddl", atoms="dense")


def successor_refusal(search, successors, max_bytes):
    """The pattern of a device search's refusal of a state whose successors exceed the context's memory."""
    room = rf"\d+ bytes \(max_bytes={max_bytes}\)"
    return (rf"^mymyr: {search}: a state has {successors} successors, which need (\d+ bytes of device memory; the CUDA "
            rf"context can still allocate {room}|more device memory than the CUDA context can still allocate \({room}\))\. "
            r"Raise the Context's max_bytes, or use the CPU search\.$")


def fan_task(tmp_path, objects):
    """A task whose initial state has objects * objects successors, one per pair of objects, all dead ends; the goal
    is the last pair's."""
    import mymyr

    n = objects
    text = f"O {n}\nP 3\nF 1 left\nF 1 right\nF 0 free\nSI 0\nFI 1\n2\nG 2\n0 1 {n - 1}\n1 1 {n - 1}\n"
    text += "A 1\npick 2\nL 1\n2 1\nE 1\n0\nL 0\nF 3\n2 0\n0 1 0\n1 1 1\n"
    path = tmp_path / "fan.txt"
    path.write_text(text)
    return mymyr.Task.from_text(path)


def symfan_task(tmp_path, objects, drop=True):
    """pick(?x ?y) adds left(x) and left(y): the initial state has objects * objects successors, of which
    objects * (objects + 1) / 2 are distinct ((a, b) and (b, a) lie far apart in the successor order); drop(?x) turns
    left(x) into done(x), a second layer with duplicates across parents. Goal: done(o[-1]) and left(o[-2])."""
    import mymyr

    n = objects
    text = f"O {n}\nP 3\nF 1 left\nF 0 free\nF 1 done\nSI 0\nFI 1\n1\nG 2\n2 1 {n - 1}\n0 1 {n - 2}\n"
    text += f"A {2 if drop else 1}\npick 2\nL 1\n1 1\nE 1\n0\nL 0\nF 3\n1 0\n0 1 0\n0 1 1\n"
    if drop:
        text += "drop 1\nL 1\n0 1 0\nE 1\n0\nL 0\nF 2\n0 0 0\n2 1 0\n"
    path = tmp_path / f"symfan-{n}-{int(drop)}.txt"
    path.write_text(text)
    return mymyr.Task.from_text(path)


def wide_fan_task(tmp_path, objects=1100):
    """Rows of objects * objects / 64 words or more: the initial state has `objects` successors; the goal follows
    the last of them."""
    import mymyr

    n = objects
    text = f"O {n}\nP 4\nF 1 picked\nF 0 free\nF 2 pad\nF 0 done\nSI 0\nFI 1\n1\nG 1\n3 1\n"
    text += "A 2\npick 1\nL 1\n1 1\nE 1\n0\nL 0\nF 3\n1 0\n0 1 0\n2 1 0 0\n"
    text += f"finish 0\nL 1\n0 1 {-n}\nE 1\n0\nL 0\nF 1\n3 1\n"
    path = tmp_path / "wide-fan.txt"
    path.write_text(text)
    return mymyr.Task.from_text(path, atoms="dense")
