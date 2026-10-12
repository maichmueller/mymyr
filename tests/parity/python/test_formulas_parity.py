"""Formulas: holds on the states of the fork's recorded walks (tests/parity/data/expected)."""

import json
import random

import pytest

import mymyr

from parity_support import DATA, IPC, WORK

GOLDEN = DATA / "expected"


def golden_cases():
    # philosophers: its PDDL declares no :derived-predicates requirement, which the front end insists on
    return sorted(f.stem for f in GOLDEN.glob("*.json") if not f.stem.startswith("philosophers")) if GOLDEN.is_dir() else []


def golden_task(g):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    tag, prob = g["source"]["tag"], g["source"]["problem"]
    dom = g["source"].get("domain_file", "domain.pddl")
    if tag.startswith("ipc/"):
        if IPC is None:
            pytest.skip("set MYMYR_PARITY_DATA")
        d = IPC / tag[4:] / "test"
    else:
        d = WORK / "mimir-cs" / "Benchmark" / tag
    if not (d / prob).is_file():
        pytest.skip(f"PDDL of {g['task']} not found")
    return mymyr.Task.from_pddl(d / dom, d / prob)


@pytest.mark.parametrize("name", golden_cases())
def test_holds_agrees_with_the_recorded_walks(name):
    g = json.loads((GOLDEN / f"{name}.json").read_text())
    task = golden_task(g)
    goal = task.goal_condition
    rng = random.Random(0)
    checked = 0
    for w in g["walks"]["walks"]:
        if w.get("duplicate_action_strings"):
            continue  # two actions print alike: the recorded strings do not name the action taken
        s = task.initial_state
        for step in w["steps"]:
            fluent = step["fluent_atoms"]
            atoms = {str(a) for a in s.atoms()}
            if len(fluent.get("items", ())) == fluent["count"]:
                assert atoms == set(fluent["items"])
            assert goal.holds(s) == s.holds(goal) == s.is_goal() == (step["is_goal"] in (True, "True"))
            true = list(s.atoms()) + [a for a in s.derived_atoms() if a.kind == "derived"]
            assert all(a.holds(s) for a in true)
            if true:
                sample = rng.sample(true, min(4, len(true)))
                assert task.ground_condition(sample).holds(s)
                assert not task.ground_condition(sample + [task.literal(sample[0], positive=False)]).holds(s)
            checked += 1
            if step.get("taken") is None:
                break
            taken = [a for a in task.applicable_actions(s) if str(a) == step["taken"]]
            if len(taken) != 1:
                break  # the fork prints this action differently (parameters added by normalization)
            s = task.apply(s, taken[0])
    assert checked > 0
