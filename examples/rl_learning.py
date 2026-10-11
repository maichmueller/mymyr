"""Learn a tiny successor scorer from sparse rewards; evaluate on longer, held-out chains.

This NumPy REINFORCE smoke example uses native relations and applicable-action
readouts. Its schema/count features are deliberately simple, not a relational
neural network or a benchmark of general planning ability.
"""

from time import perf_counter

import numpy as np

import mymyr
from mymyr import encoding, rl


def chain_tasks(sizes):
    domain = mymyr.Domain.from_string("""
    (define (domain chain) (:requirements :strips)
     (:predicates (at ?cell) (next ?left ?right))
     (:action forward :parameters (?left ?right)
      :precondition (and (at ?left) (next ?left ?right))
      :effect (and (not (at ?left)) (at ?right)))
     (:action backward :parameters (?left ?right)
      :precondition (and (at ?left) (next ?right ?left))
      :effect (and (not (at ?left)) (at ?right))))
    """)
    tasks = []
    for size in sizes:
        objects = " ".join(f"cell{position}" for position in range(size))
        edges = " ".join(f"(next cell{position} cell{position + 1})" for position in range(size - 1))
        problem = f"""(define (problem chain-{size}) (:domain chain) (:objects {objects})
          (:init (at cell0) {edges}) (:goal (at cell{size - 1})))"""
        tasks.append(mymyr.Task(domain.instantiate_string(problem), atoms="frozen"))
    return tasks


def action_features(encoder, states, task_ids):
    expansion = rl.expand(encoder.table, states, task_ids=task_ids)
    if expansion.overflow:
        raise RuntimeError("this example requires every applicable action")
    observation = encoder.encode(states, task_ids=task_ids, actions=expansion)
    successors = encoder.encode(expansion.succ, task_ids=task_ids[expansion.parent])
    satisfied = np.array([source == "goal_satisfied" for source in encoder.relation_sources])
    unsatisfied = np.array([source == "goal_unsatisfied" for source in encoder.relation_sources])
    counts = successors.relation_counts
    goal_counts = np.column_stack((counts[:, satisfied].sum(1), counts[:, unsatisfied].sum(1)))
    goal_counts = goal_counts / np.maximum(goal_counts.sum(1, keepdims=True), 1)
    schemas = np.eye(encoder.table.num_schemas)[observation.action_schema]
    features = np.column_stack((schemas, goal_counts))
    arrays = (expansion.succ, observation.relation_args, observation.relation_counts,
              observation.relation_offsets, observation.relation_tuple_offsets, observation.object_offsets,
              observation.action_schema, observation.action_binding, observation.action_parent,
              observation.action_offsets, observation.action_tuple, successors.relation_args,
              successors.relation_counts, successors.relation_offsets, successors.relation_tuple_offsets,
              successors.object_offsets, features)
    return features, sum(array.nbytes for array in arrays)


def episode(pool, encoder, weights, rng, task_id, *, greedy=False):
    batch = pool.reset(task_ids=np.array([task_id], np.int32))
    rewards, gradients = [], []
    peak_bytes = 0
    while not batch.terminated[0]:
        features, storage = action_features(encoder, batch.states, batch.task_ids)
        peak_bytes = max(peak_bytes, storage)
        logits = features @ weights
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        slot = int(np.argmax(logits)) if greedy else int(rng.choice(len(logits), p=probabilities))
        gradients.append(features[slot] - probabilities @ features)
        batch = pool.step(actions=np.array([slot], np.int32))
        rewards.append(float(batch.reward[0]))
        if batch.terminated[0] or batch.truncated[0]:
            break
    return {
        "success": bool(batch.goal[0]), "length": len(rewards), "truncated": bool(batch.truncated[0]),
        "rewards": rewards, "gradients": gradients, "peak_array_bytes": peak_bytes,
    }


def evaluate(tasks, weights, seed, episodes=64, *, greedy=False):
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table)
    pool = rl.CpuEnvPool(table, num_envs=1, threads=1, max_steps=2 * max(task.num_objects for task in tasks),
                         autoreset=False)
    rng = np.random.default_rng(seed)
    results = [episode(pool, encoder, weights, rng, task_id, greedy=greedy)
               for task_id in range(len(tasks)) for _ in range(episodes)]
    solved_lengths = [result["length"] for result in results if result["success"]]
    return {
        "success_rate": np.mean([result["success"] for result in results]).item(),
        "mean_success_length": float(np.mean(solved_lengths)) if solved_lengths else None,
        "truncations": sum(result["truncated"] for result in results),
        "peak_counted_array_bytes": max(result["peak_array_bytes"] for result in results),
    }


def run(training_episodes=400, evaluation_episodes=64, seed=7):
    started = perf_counter()
    training_sizes, evaluation_sizes = (3, 4, 5), (8, 12)
    tasks = chain_tasks(training_sizes)
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table)
    pool = rl.CpuEnvPool(table, num_envs=1, threads=1, max_steps=2 * max(training_sizes), autoreset=False)
    sampler = rl.InstanceSampler(len(tasks))
    rng = np.random.default_rng(seed)
    weights = np.zeros(table.num_schemas + 2)
    visits = np.zeros(len(tasks), np.int32)
    baseline = 0.0
    for counter in range(training_episodes):
        task_id = int(sampler.sample(seed, np.array([0], np.uint32), np.array([counter], np.uint32))[0])
        visits[task_id] += 1
        result = episode(pool, encoder, weights, rng, task_id)
        future_return = 0.0
        update = np.zeros_like(weights)
        for reward, gradient in zip(reversed(result["rewards"]), reversed(result["gradients"])):
            future_return = reward + 0.95 * future_return
            update += (future_return - baseline) * gradient
        weights += 0.05 * update
        baseline += 0.05 * (future_return - baseline)
    held_out = chain_tasks(evaluation_sizes)
    report = {
        "training_objects": training_sizes, "held_out_objects": evaluation_sizes,
        "training_instance_visits": visits.tolist(),
        "random": evaluate(held_out, np.zeros_like(weights), seed + 1, evaluation_episodes),
        "learned_greedy": evaluate(held_out, weights, seed + 1, evaluation_episodes, greedy=True),
        "elapsed_seconds": perf_counter() - started,
    }
    return report


if __name__ == "__main__":
    print(run())
