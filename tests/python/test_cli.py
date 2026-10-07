"""The mymyr command line (mymyr.cli), run as ``python -m mymyr`` and as the installed ``mymyr`` script."""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

import mymyr
from mymyr import search

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDDL = ROOT / "tests/data/pddl"
EXPECTED = ROOT / "tests/data/expected"

CLASSICAL = {  # small tasks every search solves (the IW family with width 2)
    "gripper": (PDDL / "gripper/domain.pddl", PDDL / "gripper/test_problem.pddl"),
    "delivery": (PDDL / "delivery/domain.pddl", PDDL / "delivery/test_problem.pddl"),
    "visitall": (PDDL / "visitall/domain.pddl", PDDL / "visitall/test_problem.pddl"),
}
COUNTERS = (PDDL / "counters/domain.pddl", PDDL / "counters/p01.pddl")  # numeric, action costs
UNSOLVABLE = (PDDL / "counters/domain.pddl", PDDL / "counters/p02.pddl")
# tasks with an optimal cost in the goldens (tests/data/expected, the fork's A*)
GOLDEN = {
    "logistics00__probLOGISTICS-6-1": (PDDL / "logistics00/domain.pddl", PDDL / "logistics00/probLOGISTICS-6-1.pddl"),
    "philosophers__p03-phil4": (PDDL / "philosophers/domain.pddl", PDDL / "philosophers/p03-phil4.pddl"),
    "blocks__probBLOCKS-8-0": (PDDL / "blocks/domain.pddl", PDDL / "blocks/probBLOCKS-8-0.pddl"),
}
SEARCHES = ["astar", "gbfs", "brfs", "iw", "siw", "astar_iw"]


def script():
    exe = shutil.which("mymyr", path=str(pathlib.Path(sys.executable).parent))
    if exe is None:
        pytest.skip("the mymyr console script is not installed next to this interpreter")
    return [exe]


def run(*args, how="module"):
    cmd = [sys.executable, "-m", "mymyr"] if how == "module" else script()
    return subprocess.run([*cmd, *map(str, args)], capture_output=True, text=True, timeout=600)


def replay_cost(task, text):
    """The plan of IPC text, replayed with Task.apply: (plan, reaches the goal, cost line value)."""
    plan = search.parse_plan(task, text)
    s = task.initial_state
    for a in plan:
        assert a in task.applicable_actions(s)
        s = task.apply(s, a)
    cost = float(text.rstrip().splitlines()[-1].split("=", 1)[1].split("(")[0])
    return plan, task.is_goal(s), cost


def width_args(name):
    return ["--width", "2"] if name in ("iw", "siw", "astar_iw") else []


@pytest.mark.parametrize("how", ["module", "script"])
@pytest.mark.parametrize("search_name", SEARCHES)
@pytest.mark.parametrize("task_name", sorted(CLASSICAL))
def test_every_search_plans_the_classical_tasks(task_name, search_name, how):
    d, p = CLASSICAL[task_name]
    r = run("plan", d, p, "--search", search_name, *width_args(search_name), how=how)
    assert r.returncode == 0, r.stderr
    task = mymyr.Task.from_pddl(d, p)
    plan, goal, cost = replay_cost(task, r.stdout)
    assert goal and plan
    assert r.stderr == ""
    if search_name in ("astar", "brfs"):  # unit costs: A* with h_max and BrFS find the shortest plans
        ref = search.astar(task, heuristic="blind", costs="real")
        assert cost == ref.cost


@pytest.mark.parametrize("search_name", SEARCHES)
def test_every_search_on_a_numeric_task(search_name):
    d, p = COUNTERS
    task = mymyr.Task.from_pddl(d, p)
    r = run("plan", d, p, "--search", search_name)
    if search_name == "astar_iw":  # refuses numeric tasks: one clear line
        assert r.returncode == 2 and r.stdout == ""
        assert r.stderr.count("\n") == 1 and "numeric" in r.stderr and "Traceback" not in r.stderr
        return
    if search_name in ("iw", "siw"):  # supported; novelty over the atoms alone does not reach this goal
        assert r.returncode == 1 and "no plan" in r.stderr
        return
    assert r.returncode == 0, r.stderr
    plan, goal, cost = replay_cost(task, r.stdout)
    assert goal
    if search_name == "astar":  # h_max is admissible: the optimal metric value (uniform-cost search's)
        assert cost == search.astar(task, heuristic="blind", costs="real").cost == 6


@pytest.mark.parametrize("heuristic", ["max", "blind", "h2"])
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_astar_finds_the_optimal_cost_of_the_goldens(name, heuristic):
    golden = json.loads((EXPECTED / f"{name}.json").read_text())["astar"]
    if golden["heuristic"] != "hmax" or (heuristic != "max" and golden["expanded"] > 30_000):
        pytest.skip("the other heuristics on the small goldens only")
    d, p = GOLDEN[name]
    r = run("plan", d, p, "--heuristic", heuristic, "-v")
    assert r.returncode == 0, r.stderr
    _, goal, cost = replay_cost(mymyr.Task.from_pddl(d, p), r.stdout)
    assert goal and cost == golden["optimal_cost"]
    assert f"cost: {golden['optimal_cost']}" in r.stderr


def test_plan_file_reads_back(tmp_path):
    d, p = GOLDEN["logistics00__probLOGISTICS-6-1"]
    out = tmp_path / "plan.txt"
    r = run("plan", d, p, "-o", out, "--search", "gbfs", "-v")
    assert r.returncode == 0 and r.stdout == ""
    text = out.read_text()
    assert text.rstrip().splitlines()[-1].startswith("; cost = ")
    task = mymyr.Task.from_pddl(d, p)
    plan, goal, _ = replay_cost(task, text)
    assert goal
    assert search.format_plan(task, plan) == text
    for key in ("search: gbfs with heuristic ff", "status: solved", "expanded: ", "generated: ", "time: ", "cost: "):
        assert key in r.stderr


def test_exit_codes(tmp_path):
    d, p = GOLDEN["blocks__probBLOCKS-8-0"]
    # 1: proved unsolvable / exhausted
    for s in ("astar", "brfs", "gbfs"):
        r = run("plan", *UNSOLVABLE, "--search", s)
        assert r.returncode == 1 and r.stdout == "" and "no plan" in r.stderr, (s, r.stderr)
    # 3: a budget stopped the search
    r = run("plan", d, p, "--max-states", 100)
    assert r.returncode == 3 and "out_of_states" in r.stderr
    r = run("plan", d, p, "--search", "brfs", "--max-seconds", 0)
    assert r.returncode == 3 and "out_of_time" in r.stderr
    # 2: usage and parse errors, one line on stderr without a traceback
    bad = tmp_path / "bad.pddl"
    bad.write_text(d.read_text().replace("(:action pick-up", "(:durative-action pick-up"))
    cases = [
        (["plan", d], "PROBLEM"),
        (["plan", d, p, "--search", "dfs"], "invalid choice"),
        (["plan", d, p, "--search", "iw", "--heuristic", "ff"], "--heuristic applies to"),
        (["plan", d, p, "--search", "astar", "--width", "2"], "--width applies to"),
        (["plan", d, p, "--threads", "4"], "--threads applies to brfs"),
        (["plan", d, tmp_path / "missing.pddl"], "No such file or directory"),
        (["plan", bad, p], "durative actions are not supported"),
        (["info", bad], "durative actions are not supported"),
    ]
    for args, words in cases:
        r = run(*args)
        assert r.returncode == 2, (args, r.stderr)
        assert r.stderr.count("\n") == 1 and words in r.stderr and "Traceback" not in r.stderr, (args, r.stderr)
    r = run("plan", bad, p, "--debug")
    assert r.returncode == 2 and "Traceback" in r.stderr and "PddlError" in r.stderr
    # 0
    assert run("plan", d, p, "--search", "gbfs").returncode == 0


def test_brfs_threads():
    d, p = GOLDEN["blocks__probBLOCKS-8-0"]
    task = mymyr.Task.from_pddl(d, p)
    one, four = run("plan", d, p, "--search", "brfs"), run("plan", d, p, "--search", "brfs", "-j", "4")
    assert one.returncode == four.returncode == 0, four.stderr
    (plan1, goal1, _), (plan4, goal4, _) = replay_cost(task, one.stdout), replay_cost(task, four.stdout)
    assert goal1 and goal4 and len(plan1) == len(plan4) == 18


OR_DOMAIN = """(define (domain switches)
  (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :conditional-effects)
  (:types switch)
  (:predicates (on ?s - switch) (broken ?s - switch) (lit))
  (:action flip
    :parameters (?s - switch)
    :precondition (or (on ?s) (broken ?s))
    :effect (and (not (on ?s)) (when (broken ?s) (lit))))
  (:action press
    :parameters (?s - switch)
    :precondition (not (on ?s))
    :effect (on ?s)))
"""
OR_PROBLEM = """(define (problem two) (:domain switches)
  (:objects a b - switch)
  (:init (broken b))
  (:goal (lit)))
"""


def test_info_reports_or_splits_and_conditional_effects(tmp_path):
    (tmp_path / "d.pddl").write_text(OR_DOMAIN)
    (tmp_path / "p.pddl").write_text(OR_PROBLEM)
    r = run("info", tmp_path / "d.pddl", tmp_path / "p.pddl")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "domain: switches" in out and "problem: two" in out
    assert ":disjunctive-preconditions" in out and ":conditional-effects" in out
    assert "actions: 2 in the source, 3 schemas after normalization (1 split)" in out
    assert "  flip/1: 2 schemas" in out and "  press/1: 1 schema\n" in out
    assert "conditional effects: 2 (in 2 schemas)" in out
    assert "numeric: no" in out and "objects: 2" in out and "initial atoms: " in out
    # the domain alone
    r = run("info", tmp_path / "d.pddl")
    assert r.returncode == 0 and "actions: 2 in the source, 3 schemas" in r.stdout and "problem:" not in r.stdout


def test_info_reports_numeric_fluents():
    r = run("info", *COUNTERS)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "numeric: yes" in out and "value/1 fluent" in out and "total-cost/0 auxiliary" in out
    assert "initial function values: 5 (3 static, 2 fluent)" in out
    assert "metric: minimize" in out and "actions: 4 in the source, 4 schemas after normalization" in out


def test_help_and_version():
    for args in (["--help"], ["plan", "--help"], ["info", "--help"]):
        r = run(*args)
        assert r.returncode == 0 and r.stdout.startswith("usage: mymyr"), args
    plan_help = run("plan", "--help").stdout
    for word in ("--search", "--heuristic", "--width", "--threads", "--max-seconds", "--max-states", "--plan-file",
                 "--verbose", "--debug", "admissible", "Exit codes"):
        assert word in plan_help, word
    r = run("--version")
    assert r.returncode == 0 and r.stdout.strip() == f"mymyr {mymyr.__version__}"
    assert run("--version", how="script").stdout == r.stdout
    r = run()
    assert r.returncode == 2 and "usage: mymyr" in r.stderr


def test_main_in_process(capsys):
    from mymyr.cli import main

    d, p = CLASSICAL["delivery"]
    assert main(["plan", str(d), str(p), "--search", "gbfs"]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1].startswith("; cost = ")
