"""The mymyr command line (mymyr.cli), run as ``python -m mymyr`` and as the installed ``mymyr`` script."""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status
from python_children import run_code

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDDL = ROOT / "tests/data/pddl"

CLASSICAL = {  # small tasks every search solves (the IW family with width 2)
    "gripper": (PDDL / "gripper/domain.pddl", PDDL / "gripper/test_problem.pddl"),
    "delivery": (PDDL / "delivery/domain.pddl", PDDL / "delivery/test_problem.pddl"),
    "visitall": (PDDL / "visitall/domain.pddl", PDDL / "visitall/test_problem.pddl"),
}
COUNTERS = (PDDL / "counters/domain.pddl", PDDL / "counters/p01.pddl")  # numeric, action costs
UNSOLVABLE = (PDDL / "counters/domain.pddl", PDDL / "counters/p02.pddl")
# tasks with a known optimal cost
OPTIMAL = {
    "logistics00__probLOGISTICS-6-1": (PDDL / "logistics00/domain.pddl", PDDL / "logistics00/probLOGISTICS-6-1.pddl"),
    "philosophers__p03-phil4": (PDDL / "philosophers/domain.pddl", PDDL / "philosophers/p03-phil4.pddl"),
    "blocks__probBLOCKS-8-0": (PDDL / "blocks/domain.pddl", PDDL / "blocks/probBLOCKS-8-0.pddl"),
}
OPTIMAL_COST = {"logistics00__probLOGISTICS-6-1": 14, "philosophers__p03-phil4": 36, "blocks__probBLOCKS-8-0": 18}
# A* with h_max expands this many states on each (the larger ones are only run with h_max)
EXPANDED = {"logistics00__probLOGISTICS-6-1": 5802, "philosophers__p03-phil4": 28225, "blocks__probBLOCKS-8-0": 139798}
SEARCHES = ["astar", "gbfs", "ehc", "anytime", "brfs", "iw", "siw", "astar_iw"]


def script():
    exe = shutil.which("mymyr", path=str(pathlib.Path(sys.executable).parent))
    if exe is None:
        pytest.skip("the mymyr console script is not installed next to this interpreter")
    return [exe]


def run(*args, how="module"):
    if how == "module":
        code = "runpy.run_module('mymyr', run_name='__main__', alter_sys=True)"
        return run_code(code, *map(str, args), timeout=600, capture_output=True, check=False)
    return subprocess.run([*script(), *map(str, args)], capture_output=True, text=True, timeout=600)


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
        assert r.stderr.startswith("mymyr: error: AStarIW does not support")  # one prefix
        return
    if search_name in ("iw", "siw"):  # supported; novelty over the atoms alone does not reach this goal
        assert r.returncode == 1 and "no plan" in r.stderr
        return
    assert r.returncode == 0, r.stderr
    plan, goal, cost = replay_cost(task, r.stdout)
    assert goal
    if search_name in ("astar", "anytime"):  # the optimal metric value (uniform-cost search's)
        assert cost == search.astar(task, heuristic="blind", costs="real").cost == 6


@pytest.mark.parametrize("heuristic", ["max", "blind", "h2", "lmcut"])
@pytest.mark.parametrize("name", sorted(OPTIMAL))
def test_astar_finds_the_optimal_cost(name, heuristic):
    if heuristic != "max" and EXPANDED[name] > 30_000:
        pytest.skip("the other heuristics on the small tasks only")
    d, p = OPTIMAL[name]
    r = run("plan", d, p, "--heuristic", heuristic, "-v")
    assert r.returncode == 0, r.stderr
    _, goal, cost = replay_cost(mymyr.Task.from_pddl(d, p), r.stdout)
    assert goal and cost == OPTIMAL_COST[name]
    assert f"cost: {OPTIMAL_COST[name]}" in r.stderr


def test_plan_file_reads_back(tmp_path):
    d, p = OPTIMAL["logistics00__probLOGISTICS-6-1"]
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
    d, p = OPTIMAL["blocks__probBLOCKS-8-0"]
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
        (["plan", p, d], "this is a PDDL problem, where a domain belongs"),
        (["plan", d, p, "--search", "brfs", "--threads", "100000"], "threads must be an int in [0, "),
        (["plan", d, p, "--search", "gbfs", "--weight", "2"], "--weight applies to astar"),
        (["plan", d, p, "--weight", "0.5"], "--weight must be a finite number of at least 1"),
        (["plan", d, p, "--weight", "inf"], "--weight must be a finite number of at least 1"),
        (["plan", d, p, "--weights", "5,3"], "--weights applies to anytime"),
        (["plan", d, p, "--search", "anytime", "--weights", "1,2"], "--weights must not increase"),
        (["plan", d, p, "--search", "anytime", "--weights", "3,0.5"], "--weights must be finite numbers"),
        (["plan", d, p, "--search", "anytime", "--weights", "5,x"], "not a comma-separated list of numbers"),
    ]
    for args, words in cases:
        r = run(*args)
        assert r.returncode == 2, (args, r.stderr)
        assert r.stderr.count("\n") == 1 and words in r.stderr and "Traceback" not in r.stderr, (args, r.stderr)
        assert re.match(r"mymyr( plan| info)?: error: (?!mymyr)", r.stderr), (args, r.stderr)  # one prefix
    r = run("plan", bad, p, "--debug")
    assert r.returncode == 2 and "Traceback" in r.stderr and "PddlError" in r.stderr
    # 0
    assert run("plan", d, p, "--search", "gbfs").returncode == 0


@pytest.mark.parametrize("weight", ["1.5", "2", "5"])
def test_weighted_astar_stays_within_the_bound(weight):
    name = "logistics00__probLOGISTICS-6-1"
    d, p = OPTIMAL[name]
    r = run("plan", d, p, "--heuristic", "lmcut", "--weight", weight, "-v")
    assert r.returncode == 0, r.stderr
    _, goal, cost = replay_cost(mymyr.Task.from_pddl(d, p), r.stdout)
    assert goal and OPTIMAL_COST[name] <= cost <= float(weight) * OPTIMAL_COST[name]
    assert f"search: astar with heuristic lmcut, weight {float(weight):g}" in r.stderr


def test_ehc_and_anytime_statistics():
    name = "logistics00__probLOGISTICS-6-1"
    d, p = OPTIMAL[name]
    r = run("plan", d, p, "--search", "ehc", "-v")
    assert r.returncode == 0, r.stderr
    assert "search: ehc with heuristic ff" in r.stderr and "hill-climbing: " in r.stderr
    r = run("plan", d, p, "--search", "anytime", "--heuristic", "lmcut", "--weights", "3,1", "-v")
    assert r.returncode == 0, r.stderr
    _, goal, cost = replay_cost(mymyr.Task.from_pddl(d, p), r.stdout)
    assert goal and cost == OPTIMAL_COST[name]
    plans = [line for line in r.stderr.splitlines() if line.startswith("plan: cost ")]
    costs = [float(line.split()[2]) for line in plans]
    assert plans and costs == sorted(costs, reverse=True) and costs[-1] == cost
    assert "weight 3" in plans[0] and "optimal: yes" in r.stderr


def test_anytime_writes_its_best_plan_when_a_budget_stops_it():
    d, p = OPTIMAL["blocks__probBLOCKS-8-0"]
    task = mymyr.Task.from_pddl(d, p)
    # room for the first iteration (weighted A* with weight 5) only
    cap = search.astar(task, heuristic="ff", weight=5).stats.states + 1
    capped = search.anytime_astar(task, heuristic="ff", max_states=cap)
    assert capped.status == Status.SOLVED and capped.iterations[-1].status == Status.OUT_OF_STATES
    r = run("plan", d, p, "--search", "anytime", "--max-states", cap, "-v")
    assert r.returncode == 0, r.stderr
    _, goal, cost = replay_cost(task, r.stdout)
    assert goal and cost == capped.cost
    assert "status: solved" in r.stderr and "optimal: not proven" in r.stderr


@pytest.mark.parametrize("search_name", ["astar", "gbfs", "brfs", "siw"])
def test_plans_do_not_depend_on_the_atom_slots(search_name):
    d, p = OPTIMAL["logistics00__probLOGISTICS-6-1"]
    outs = [run("plan", d, p, "--search", search_name, "--atoms", atoms, "-v", *width_args(search_name))
            for atoms in ("lazy", "auto", "reachable", "dense")]
    assert all(r.returncode == 0 for r in outs), [r.stderr for r in outs]
    assert len({r.stdout for r in outs}) == 1
    counts = {re.sub(r"[0-9.]+ ?s\b|time[^,\n]*", "", r.stderr) for r in outs}
    assert len(counts) == 1, counts


def test_brfs_threads():
    d, p = OPTIMAL["blocks__probBLOCKS-8-0"]
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
