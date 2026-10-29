"""Lifted hindsight scans against pymimir-rl 0.2.6 with fixed injected templates.

The reference runs in an optional private interpreter, selected by MYMYR_LIFTED_HER_REFERENCE_PYTHON.
Without it the current interpreter is used; absence of the reference skips these tests. CI does not install it.
"""

import json
import os
from pathlib import Path
import random
import subprocess
import sys

DOMAIN = """(define (domain marking) (:requirements :strips :negative-preconditions)
 (:predicates (marked ?x) (free))
 (:action mark :parameters (?x) :precondition (free) :effect (and (marked ?x) (not (free))))
 (:action clear :parameters (?x) :precondition (marked ?x) :effect (and (not (marked ?x)) (free)))
 (:action wait :parameters () :precondition (and) :effect (and)))"""
PROBLEM = """(define (problem p) (:domain marking) (:objects a b) (:init (free)) (:goal (marked a)))"""


def _reference(payload):
    import importlib.metadata
    from types import SimpleNamespace
    import pymimir as mm
    from pymimir_rl.trajectory_refinements import LiftedHindsightTrajectoryRefiner

    assert importlib.metadata.version("pymimir-rl") == "0.2.6"
    problem = mm.Problem(mm.Domain(DOMAIN), PROBLEM)
    initial = problem.get_initial_state()

    def key(s):
        return tuple(sorted(str(a) for a in s.get_atoms(ignore_static=True, ignore_derived=True)))

    states = {key(initial): initial}
    stack = [initial]
    while stack:
        state = stack.pop()
        for action in state.generate_applicable_actions():
            reached = action.apply(state)
            if key(reached) not in states:
                states[key(reached)] = reached
                stack.append(reached)
    x = problem.new_variable("x", 0)
    pred = problem.get_domain().get_predicate("marked")
    literal = problem.new_literal(problem.new_atom(pred, [x]), True)
    condition = problem.new_conjunctive_condition([x], [literal])

    class Episode:
        def __init__(self, env, first, path):
            self.problem = problem
            self.env, self.first = env, first
            self.transitions = [SimpleNamespace(current_state=states[tuple(sorted(a))],
                                                successor_state=states[tuple(sorted(b))])
                                for a, b in zip(path, path[1:])]

        def __len__(self):
            return len(self.transitions)

        def __getitem__(self, index):
            return self.transitions[index]

        def clone_with_goal(self, start, end, goal):
            return [self.env, self.first + start, self.first + end, sorted(str(lit) for lit in goal)]

    trajectories = [Episode(env, first, path) for env, first, path in payload["episodes"]]
    refiner = object.__new__(LiftedHindsightTrajectoryRefiner)
    refiner.problems = [problem]
    refiner.max_generated = payload["cap"]
    refiner.problem_subgoals = {problem: [condition]}
    refiner.problem_blacklist = {problem: []}
    return refiner.refine(trajectories)


if __name__ == "__main__":
    print(json.dumps(_reference(json.load(sys.stdin))))
    raise SystemExit(0)

import numpy as np
import pytest
import mymyr
from mymyr import rl

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("seed", [None, 7, 19])
@pytest.mark.parametrize("cap", [1, 4, 100])
def test_lifted_segments_equal_pymimir_rl_refine(seed, cap):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    executable = os.environ.get("MYMYR_LIFTED_HER_REFERENCE_PYTHON", sys.executable)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_GIL": "1"}
    probe = subprocess.run([executable, "-c", "import pymimir_rl.trajectory_refinements"], env=env,
                           capture_output=True, text=True)
    if probe.returncode:
        pytest.skip("pymimir-rl reference is not importable")
    task = mymyr.Task(mymyr.Domain.from_string(DOMAIN).instantiate_string(PROBLEM))
    templates = rl.GoalTemplates.from_conditions(task, [[task.condition(["?x"], ["(marked ?x)"])]])
    T, N = 24, 2
    rng = random.Random(seed)
    paths = []
    for i in range(N):
        path = []
        for q in range(T + 1):
            if seed is None:
                label = [0, 1, 1, 0, 2, 0, 1, 0, 0][(q + i) % 9]
            else:
                label = rng.choice([0, label]) if q and label else rng.randrange(3)
            path.append(["(free)"] if label == 0 else [f"(marked {'a' if label == 1 else 'b'})"])
        paths.append(path)
    prev = np.stack([np.asarray(task.encode([task.state(paths[i][q]) for i in range(N)])) for q in range(T)])
    reached = np.stack([np.asarray(task.encode([task.state(paths[i][q+1]) for i in range(N)])) for q in range(T)])
    done = np.zeros((T, N), bool)
    done[7::8] = True
    episodes = [[i, first, paths[i][first:first+9]] for i in range(N) for first in range(0, T, 8)]
    run = subprocess.run([executable, str(Path(__file__).resolve())], input=json.dumps({"episodes": episodes, "cap": cap}),
                         env=env, capture_output=True, text=True, check=True)
    expected = json.loads(run.stdout)
    actual = rl.her_relabel_lifted(reached, done, None, templates, prev_states=prev,
                                   max_segments=T, max_generated=cap)
    records = []
    for i in range(N):
        for j in range(actual.count[i]):
            goal = task.state(actual.goal_pos[i, j]).atoms()
            records.append([i, int(actual.seg_start[i, j]), int(actual.seg_end[i, j]), sorted(map(str, goal))])
    assert records == expected
    assert int(actual.status) == 0
