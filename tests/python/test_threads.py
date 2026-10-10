"""Thread counts: bounded with ValueError everywhere a caller's count becomes OS threads; a refused thread start is a
RuntimeError, never an abort or a hang."""

import subprocess
import sys
import textwrap

import pytest

from mymyr import datasets, rl, search

from conftest import TASKS, text_task

MANY = 10**6  # above the bound max(64, 4 x the hardware threads) on any machine


@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


CALLS = {
    "brfs": lambda t: search.brfs(t, threads=MANY),
    "iw": lambda t: search.iw(t, threads=MANY),
    "siw": lambda t: search.siw(t, threads=MANY),
    "liw": lambda t: search.liw(t, threads=MANY),
    "portfolio": lambda t: search.atomic_goal_portfolio(t, num_threads=MANY),
    "rollouts": lambda t: search.find_rollouts_parallel(t, [1, 2], num_threads=MANY),
    "generate": lambda t: datasets.generate(t, threads=MANY),
    "generate_many": lambda t: datasets.generate_many([t, t], threads=MANY),
    "tuple_graphs": lambda t: datasets.tuple_graphs(datasets.state_space(t), threads=MANY),
    "knowledge_base": lambda t: datasets.KnowledgeBase([t], threads=MANY),
    "thread_pool": lambda t: rl.ThreadPool(MANY),
    "expand": lambda t: rl.expand(t, [t.initial_state], threads=MANY),
    "env_pool": lambda t: rl.CpuEnvPool(t, 4, threads=MANY),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_too_many_threads_raise_value_error(gripper, name):
    with pytest.raises(ValueError, match=r"threads must be an int in \[0, \d+\], got 1000000"):
        CALLS[name](gripper)


# A child process with 128 MiB of spare address space: of 60 threads (8 MiB stacks), only some can start.
CHILD = textwrap.dedent(
    """
    import resource, sys
    import mymyr
    from mymyr import datasets, rl, search
    t = mymyr.Task.from_text(sys.argv[2])
    vm = int(next(x.split()[1] for x in open("/proc/self/status") if x.startswith("VmSize:"))) * 1024
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (vm + (128 << 20), hard))
    call = {
        "brfs": lambda: search.brfs(t, threads=60, stop_at_goal=True),
        "thread_pool": lambda: rl.ThreadPool(60),
        "generate": lambda: datasets.generate(t, threads=60),
        "env_pool": lambda: rl.CpuEnvPool(t, 64, threads=60),
    }[sys.argv[1]]
    try:
        call()
        print("returned")
    except RuntimeError as e:
        print("RuntimeError", e)
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    r = search.brfs(t, threads=4, stop_at_goal=True)
    print("after", r.status.name, len(r.plan))
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS and /proc are Linux")
@pytest.mark.parametrize("name", ["brfs", "thread_pool", "generate", "env_pool"])
def test_a_refused_thread_is_a_runtime_error(name):
    p = subprocess.run(
        [sys.executable, "-c", CHILD, name, str(TASKS / "gripper__prob05.txt")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    lines = p.stdout.splitlines()
    assert lines[0].startswith("RuntimeError mymyr: could not start worker thread"), p.stdout
    assert lines[1] == "after SOLVED 35", p.stdout


# A child process with 1.5 GiB of spare address space loads a table of 64 problems with the default thread count. Each
# worker thread reserves about 72 MiB (its stack and its malloc arena), so one thread per core does not fit on a
# machine with many cores.
LOADER = textwrap.dedent(
    """
    import resource, sys
    from mymyr.rl import TaskTable
    vm = int(next(x.split()[1] for x in open("/proc/self/status") if x.startswith("VmSize:"))) * 1024
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (vm + (3 << 29), hard))
    table = TaskTable.from_pddl(sys.argv[1], [sys.argv[2]] * 64)
    print("loaded", len(table))
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS and /proc are Linux")
def test_the_table_loader_default_fits_a_small_address_space():
    counters = TASKS.parent / "pddl" / "counters"
    p = subprocess.run(
        [sys.executable, "-c", LOADER, str(counters / "domain.pddl"), str(counters / "p01.pddl")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.splitlines() == ["loaded 64"], p.stdout
