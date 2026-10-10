"""A State or a row enters a task only as a state of that task: the RL batch entry points (rl.expand, rl.expand_into,
rl.is_goal) and the task's own (encode, decode, the state arguments) raise ValueError for a State of another task, a
State of another instance than its row's task id, a row that sets atom slots its task has not assigned, and a row
narrower than a frozen task's states. Rows of the task's width, or wider with zero padding, are its states."""

import numpy as np
import pytest

import mymyr
from mymyr import rl

from conftest import BLOCKS, ROOT

GRIPPER = ROOT / "tests/data/pddl/gripper"

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)


@pytest.fixture(scope="module")
def blocks():
    return mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl", atoms="frozen")


@pytest.fixture(scope="module")
def grippers():
    return [mymyr.Task.from_pddl(GRIPPER / "domain.pddl", GRIPPER / p) for p in ("p01.pddl", "p02.pddl", "test_problem.pddl")]


def test_rl_entry_points_refuse_a_state_of_another_task(blocks, grippers):
    other = grippers[0].initial_state
    with pytest.raises(ValueError, match=r"^mymyr: state 0 belongs to another task$"):
        rl.expand(blocks, [other])
    with pytest.raises(ValueError, match=r"^mymyr: the state belongs to another task$"):
        rl.expand(blocks, other)
    with pytest.raises(ValueError, match=r"^mymyr: state 1 belongs to another task$"):
        rl.is_goal(blocks, [blocks.initial_state, other])
    with pytest.raises(ValueError, match="belongs to another task"):
        rl.expand_into(blocks, [other], offsets=np.zeros(2, np.int32))
    # another instance of the same domain
    with pytest.raises(ValueError, match="belongs to another task"):
        rl.expand(grippers[0], [grippers[1].initial_state])
    assert list(rl.expand(blocks, [blocks.initial_state]).offsets) == [0, len(blocks.applicable_actions(blocks.initial_state))]


def test_a_table_checks_each_state_against_its_task_id(grippers):
    table = rl.TaskTable(grippers)
    states = [grippers[0].initial_state, grippers[1].initial_state]
    e = rl.expand(table, states, [0, 1])
    assert list(np.diff(e.offsets)) == [len(t.applicable_actions(t.initial_state)) for t in grippers[:2]]
    assert list(rl.is_goal(table, [table[2].initial_state], [2])) == [False]
    with pytest.raises(ValueError, match=r"^mymyr: state 0 is not a state of instance 1 of the table \(its task id\)$"):
        rl.expand(table, states, [1, 0])
    with pytest.raises(ValueError, match=r"^mymyr: state 0 is not a state of instance 2 of the table"):
        rl.is_goal(table, [grippers[0].initial_state], [2])
    with pytest.raises(ValueError, match="is not a state of instance 0"):
        rl.expand(rl.TaskSuite([table]), [grippers[2].initial_state], [0])
    with pytest.raises(ValueError, match="outside the table"):
        rl.expand(table, states, [0, 3])


def test_the_task_entry_points_refuse_a_state_of_another_task(blocks, grippers):
    with pytest.raises(ValueError, match=r"^mymyr: state 0 belongs to another task$"):
        blocks.encode([grippers[0].initial_state])
    with pytest.raises(ValueError, match=r"^mymyr: the state belongs to another task$"):
        blocks.encode(grippers[0].initial_state)
    assert np.array_equal(np.asarray(blocks.encode([blocks.initial_state])), np.asarray(blocks.encode(blocks.initial_state)))


def test_rows_narrower_than_a_frozen_task_raise(blocks):
    rows = np.asarray(blocks.encode([blocks.initial_state]))
    assert blocks.words == 2 and rows.shape == (1, 2)
    first = np.ascontiguousarray(rows[:, :1])
    with pytest.raises(ValueError, match=r"^mymyr: expand: state rows of 1 atom words; the states of instance 0 have 2"):
        rl.expand(blocks, first)
    with pytest.raises(ValueError, match=r"^mymyr: is_goal: state rows of 1 atom words"):
        rl.is_goal(blocks, first)
    with pytest.raises(ValueError, match=r"^mymyr: rows of 1 atom words; this task's states have 2 \(frozen atom slots\)$"):
        blocks.decode(first)
    # wider rows with zero padding are the same states
    wide = np.concatenate([rows, np.zeros((1, 1), np.uint64)], 1)
    assert np.array_equal(rl.expand(blocks, wide).offsets, rl.expand(blocks, rows).offsets)
    assert blocks.decode(wide) == [blocks.initial_state]


def test_rows_with_unassigned_atom_slots_raise(blocks):
    rows = np.asarray(blocks.encode([blocks.initial_state]))
    bad = np.concatenate([rows, np.ones((1, 1), np.uint64)], 1)
    with pytest.raises(ValueError, match=r"^mymyr: is_goal: state row 0 sets atom slots its instance has not assigned"):
        rl.is_goal(blocks, bad)
    with pytest.raises(ValueError, match=r"^mymyr: expand: state row 0 sets atom slots"):
        rl.expand(blocks, bad)
    with pytest.raises(ValueError, match=r"^mymyr: state 0 sets atom slots this task has not assigned"):
        blocks.decode(bad)
    with pytest.raises(ValueError, match=r"^mymyr: the state sets atom slots this task has not assigned"):
        blocks.applicable_actions(bad[0])
