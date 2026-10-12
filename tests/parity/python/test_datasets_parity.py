"""mymyr.datasets against the fork's state-space numbers on the fork's own test instances (env MYMYR_PARITY_DATA; skipped
when missing)."""

import numpy as np
import pytest

import mymyr
from mymyr import datasets

from parity_support import FORK_DATA


def fork_task(d, p, **options):
    dom, prob = FORK_DATA / d / "domain.pddl", FORK_DATA / d / p
    if not dom.is_file() or not prob.is_file():
        pytest.skip(f"reference data missing: {prob}")
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    options.setdefault("atoms", "dense")
    return mymyr.Task.from_pddl(dom, prob, **options)


# (dir, problem, states, transitions, goal, unsolvable, max goal distance, sum of finite cost goal distances)
FORK = [
    ("spanner", "p15-easy.pddl", 662, 1722, 40, 153, 14, None),
    ("barman", "test_problem.pddl", 406, 1298, 28, 70, 17, 5136.0),  # action costs
    ("miconic-fulladl", "test_problem.pddl", 87, 229, 6, 0, 8, None),  # derived predicates, conditional effects
    ("gripper", "p-2-0.pddl", 28, 104, 2, 0, 6, None),
]


@pytest.mark.parametrize("case", FORK, ids=lambda c: f"{c[0]}-{c[1]}")
def test_counts_match_the_fork(case):
    d, p, n, e, goal, unsolvable, max_d, cost_sum = case
    task = fork_task(d, p)
    for threads in (1, 3):
        r = datasets.generate(task, threads=threads, remove_if_unsolvable=False)
        assert r.status == datasets.Status.OK and r and r.states == n
        s = r.space
        assert (s.num_states, s.num_transitions, s.num_goal_states, s.num_unsolvable_states) == (n, e, goal, unsolvable)
        assert s.max_goal_distance == max_d
        a = s.arrays()
        if cost_sum is not None:
            c = a["cost_goal_distances"]
            assert c[np.isfinite(c)].sum() == cost_sum
            assert not s.unit_costs and a["costs"].dtype == np.float64 and a["costs"].shape == (e,)
