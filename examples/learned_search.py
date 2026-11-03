"""A learned heuristic in mymyr's searches: the model receives batches of state rows, encoded natively, and returns one
value per row as an array.

The model here is a stand-in for a trained network: a fixed linear function of each graph's tuple counts per relation
(2 per unsatisfied goal atom, -1 per carried ball). It is evaluated
  1. in A* on one instance, with batched expansions (many states per call), and
  2. by searches over several instances of different sizes on several threads, sharing the model: a SharedEvaluator
     coalesces their batches into one call, for GBFS and for the layers of breadth-first beams.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import mymyr
from mymyr import encoding, rl, search

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "tests/data/benchmarks/small/gripper/domain.pddl"
tasks = [mymyr.Task.from_pddl(DOMAIN, ROOT / f"tests/data/table_instances/gripper/gripper-{n}.pddl")
         for n in (4, 10, 20)]
table = rl.TaskTable(tasks)  # one encoder for all instances: the model generalizes across sizes


class RelationCountModel:
    """evaluate_batch(batch) -> one value per row of batch.rows; batch.encoded is the rows' FlatBatch."""

    def __init__(self, table):
        self.encoder = encoding.FlatRelationEncoder(table)
        names = self.encoder.schema.relation_names
        self.weights = np.zeros(len(names))
        self.weights[names.index("[+]at[g]")] = 2.0        # goal atoms
        self.weights[names.index("[+]at[g][sat]")] = -2.0  # ... that hold
        self.weights[names.index("carry")] = -1.0
        self.calls = 0
        self.states = 0

    def evaluate_batch(self, batch):
        self.calls += 1
        self.states += len(batch)
        graphs = batch.encoded
        counts = np.diff(np.asarray(graphs.relation_offsets)).reshape(len(batch), -1)
        return np.maximum(counts @ self.weights, 0.0)


# 1. A* with batched expansions: 32 expansions per round, their new successors evaluated in one call
model = RelationCountModel(table)
one = search.astar(tasks[0], heuristic=model, expansion_batch=32)
print(f"A*: {one.status}, cost {one.cost}, {one.stats.expanded} expanded, "
      f"{model.calls} model calls, {model.states / model.calls:.0f} states per call")
assert one.solved

# 2. searches on several instances and threads share one model: their batches meet in one call
model = RelationCountModel(table)
shared = search.SharedEvaluator(model, max_batch=4096, max_wait=0.002)
with ThreadPoolExecutor(len(tasks)) as pool:
    results = list(pool.map(lambda t: search.gbfs(t, heuristic=shared, expansion_batch=16), tasks))
for task, r in zip(tasks, results):
    print(f"GBFS on {len(task.objects)} objects: {r.status}, {len(r.plan)} actions")
print(f"shared GBFS: {shared.calls} calls, {shared.rows / shared.calls:.0f} states per call")
assert all(r.solved for r in results)

# the same model ranks the layers of breadth-first beams (lower first): one call per layer and search, coalesced
model = RelationCountModel(table)
shared = search.SharedEvaluator(model, max_wait=0.002)
with ThreadPoolExecutor(len(tasks)) as pool:
    beams = list(pool.map(lambda t: search.brfs(t, layer_order=shared, beam_width=32), tasks))
for task, r in zip(tasks, beams):
    print(f"beam on {len(task.objects)} objects: solved {r.solved}, {len(r.plan)} actions")
print(f"shared beam scores: {shared.calls} calls, {shared.rows / shared.calls:.0f} states per call")
assert all(r.solved for r in beams)
