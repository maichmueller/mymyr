"""IPC plan text: search.format_plan writes a plan, search.parse_plan reads it back."""

import pytest

from mymyr import search
from conftest import text_task


@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


def test_round_trip(gripper):
    r = search.astar(gripper, heuristic="max")
    text = search.format_plan(gripper, r.plan)
    lines = text.splitlines()
    assert len(lines) == len(r.plan) + 1
    assert lines[:-1] == [str(a) for a in r.plan]
    assert lines[-1] == f"; cost = {int(r.cost)} (unit cost)"
    assert search.parse_plan(gripper, text) == r.plan
    assert search.parse_plan(gripper, "; comment\n\n" + text.upper()) == r.plan


def test_pddl_task_and_start_state(blocks):
    s1 = blocks.successor_states(blocks.initial_state)[0]
    r = search.gbfs(blocks, start=s1)
    text = search.format_plan(blocks, r.plan, start=s1)
    assert text.endswith(f"; cost = {int(r.cost)} (unit cost)\n")
    assert search.parse_plan(blocks, text, start=s1) == r.plan
    with pytest.raises(ValueError, match="not applicable"):
        search.format_plan(blocks, r.plan)  # not from the initial state


def test_errors(gripper):
    with pytest.raises(ValueError, match="line 2"):
        search.parse_plan(gripper, "; plan\n(no-such-action a b)\n")
    with pytest.raises(ValueError, match="line 1"):
        search.parse_plan(gripper, "pick ball1\n")
    other = text_task("depot__p02")
    with pytest.raises(ValueError, match="another task"):
        search.format_plan(gripper, other.applicable_actions(other.initial_state)[:1])
    with pytest.raises(TypeError):
        search.format_plan(gripper, ["(pick ball1 rooma left)"])
