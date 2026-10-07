"""Generate per-task and generalized state spaces, then inspect graph certificates."""

from pathlib import Path

import mymyr
from mymyr import datasets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
domain = DATA / "domain.pddl"
tasks = [
    mymyr.Task.from_pddl(domain, DATA / "p01.pddl"),
    mymyr.Task.from_pddl(domain, DATA / "p02.pddl"),
]
space = datasets.state_space(tasks[0], remove_if_unsolvable=False)
generalized = datasets.generalized_state_space(tasks, remove_if_unsolvable=False)
graph = datasets.object_graph(space.state(space.initial_state_id))
sampler = datasets.StateSpaceSampler(space, seed=7)

print(space.num_states, space.num_transitions, generalized.num_vertices)
print(graph.color_refinement_certificate(), graph.kfwl_certificate(2))
print(sampler.sample_states(5))
