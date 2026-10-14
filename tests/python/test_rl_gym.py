"""The Gymnasium environments (mymyr.rl.gym): Gymnasium's env checker, the RL contract against the CPU pool, seeding,
instance draws, spaces, registration and the optional dependency."""

import copy
import importlib.util
import pathlib
import pickle

import numpy as np
import pytest

import mymyr
from conftest import walk
from mymyr import rl
from python_children import run_code

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data"
HAS_GYM = importlib.util.find_spec("gymnasium") is not None
needs_gym = pytest.mark.skipif(not HAS_GYM, reason="needs gymnasium (the extra mymyr[gym])")
needs_frontend = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")

# vendored tasks: (domain file, problems) as TaskTable.from_pddl takes them
TASKS = {
    "blocks": ("pddl/blocks/domain.pddl", "pddl/blocks"),  # 2 instances
    "gripper": ("pddl/gripper/domain.pddl", "pddl/gripper"),  # 3 instances
    "logistics": ("pddl/logistics00/domain.pddl", "pddl/logistics00"),
    "philosophers": ("pddl/philosophers/domain.pddl", "pddl/philosophers"),  # axioms, a derived goal
    "miconic-simpleadl": ("benchmarks/small/miconic-simpleadl/domain.pddl", "benchmarks/small/miconic-simpleadl"),
    "deadend": ("benchmarks/small/deadend/domain.pddl", "benchmarks/small/deadend"),  # a dead end at the start
}
NUMERIC = ("pddl/counters/domain.pddl", "pddl/counters")
CLASSICAL = tuple(TASKS)
OBSERVATIONS = ("symbolic", "graph", "bits")
ACTIONS = ("index", "ground")


def paths(name):
    domain, problems = TASKS[name] if name in TASKS else NUMERIC
    return str(DATA / domain), str(DATA / problems)


_tables = {}


def table(name):
    if name not in _tables:
        _tables[name] = rl.TaskTable.from_pddl(*paths(name))
    return _tables[name]


def toggle_task(init="", goal="(q)"):
    domain = mymyr.Domain.from_string("""
    (define (domain toggles) (:requirements :strips :negative-preconditions)
     (:predicates (p) (q))
     (:action up :parameters () :precondition (not (p)) :effect (p))
     (:action down :parameters () :precondition (p) :effect (not (p)))
     (:action finish :parameters () :precondition (p) :effect (q)))
    """)
    problem = f"(define (problem tiny) (:domain toggles) (:init {init}) (:goal {goal}))"
    return mymyr.Task(domain.instantiate_string(problem))


def contract(**overrides):
    out = dict(max_steps=12, step_reward=-0.25, goal_reward=2.0, dead_end_reward=-3.0)
    out.update(overrides)
    return out


def reference_pool(t, **config):
    return rl.CpuEnvPool(t, 1, threads=1, autoreset=False, sampler=None, words=max(1, t.words), **config)


# ------------------------------------------------------------------------------------------------ the successor bound


@needs_frontend
@pytest.mark.parametrize("name", [*CLASSICAL, "counters"])
def test_successor_bounds_bound_every_reachable_state(name):
    t = table(name)
    bounds = rl.successor_bounds(t)
    assert len(bounds) == len(t)
    for task, bound in zip(t.tasks, bounds):
        space = mymyr.datasets.state_space(task, max_states=20000, remove_if_unsolvable=False)
        states = space.states() if space is not None else walk(task, steps=300, walks=5)
        assert max(len(task.applicable_actions(s)) for s in states) <= bound
        assert rl.successor_bounds(task) == [bound]


@needs_frontend
def test_import_without_gymnasium_names_the_extra():
    code = (
        "import sys\nsys.modules['gymnasium'] = None\n"
        "import mymyr, mymyr.rl\nassert 'gymnasium' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
        "try:\n    import mymyr.rl.gym\nexcept ImportError as e:\n    print(e)\nelse:\n    raise SystemExit('imported')\n"
    )
    out = run_code(code, capture_output=True).stdout
    assert "mymyr[gym]" in out and "gymnasium" in out


# ------------------------------------------------------------------------------------------------ Gymnasium's checker


@needs_gym
@needs_frontend
@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("name", CLASSICAL)
def test_check_env_passes(name, observation, action):
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from mymyr.rl import gym as mg

    env_id = mg.register_pddl(*paths(name), name=f"check-{name}-{observation}-{action}", observation=observation,
                              action=action, max_steps=20)
    env = gym.make(env_id)
    check_env(env.unwrapped)
    obs, info = env.reset(seed=3)
    for _ in range(25):
        if action == "index":
            step = env.step(env.action_space.sample(mask=info["action_mask"]))
        else:
            step = env.step(env.unwrapped.action_space.sample())
        obs, info = step[0], step[4]
        assert obs in env.observation_space
        if step[2] or step[3]:
            obs, info = env.reset()
    env.close()
    gym.registry.pop(env_id)


@needs_gym
@needs_frontend
@pytest.mark.parametrize("observation", ["symbolic", "bits"])
@pytest.mark.parametrize("action", ACTIONS)
def test_check_env_passes_numeric(observation, action):
    from gymnasium.utils.env_checker import check_env
    from mymyr.rl import gym as mg

    env = mg.Env(table("counters"), observation=observation, action=action, max_steps=20)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    if observation == "symbolic":
        assert obs.numeric == tuple(env.state.numeric_values())
    else:
        np.testing.assert_array_equal(obs["numeric"][: len(env.task.numeric_names)], env.state.numeric_values())
    with pytest.raises(ValueError, match="classical"):
        mg.Env(table("counters"), observation="graph")


# ------------------------------------------------------------------------------------------------ the RL contract


def drive(env, pool, steps, seed, rng):
    """Steps a gym Env and a reference CpuEnvPool (the numpy environment of the RL contract) with the same index
    actions, resetting the pool into the instances the env draws; compares every transition. Also steps a few times
    past terminations (absorbing states) before resetting. Returns counts of what the run covered."""
    counts = dict(steps=0, terminated=0, truncated=0, invalid=0, absorbing=0, solved_resets=0, terminal_resets=0)

    def same_state(rows, task_id):
        state = rl.decode(env.table, np.asarray(rows), [task_id])[0]
        assert env.state == state
        return state

    def reset(seed=None):
        obs, info = env.reset(seed=seed)
        batch = pool.reset(task_ids=np.array([info["task_id"]], np.int32))
        state = same_state(batch.states, info["task_id"])
        assert info["is_goal"] == bool(batch.goal[0]) == env.task.is_goal(state)
        assert info["terminated"] == bool(batch.terminated[0])
        counts["solved_resets"] += info["is_goal"]
        counts["terminal_resets"] += info["terminated"]
        if isinstance(obs, mymyr.rl.gym.SymbolicObservation):
            assert obs.atoms == frozenset(state.atoms())
        return info, batch, state, int(info["terminated"])  # a terminal reset state: the first step is absorbing

    info, batch, state, finished_for = reset(seed)
    task = env.task
    while counts["steps"] < steps:
        k = info["num_actions"]
        assert k == int(batch.count[0])
        action = int(rng.integers(-1, k + 2))  # some actions outside [0, k): invalid, no move
        successors = task.successor_states(state)
        obs, reward, terminated, truncated, info = env.step(action)
        batch = pool.step(actions=np.array([action], np.int64))
        assert isinstance(reward, float) and type(terminated) is bool and type(truncated) is bool
        assert reward == float(batch.reward[0])
        assert terminated == bool(batch.terminated[0]) and truncated == bool(batch.truncated[0])
        assert info["invalid"] == bool(batch.invalid[0])
        assert info["task_id"] == int(batch.task_ids[0]) and info["num_actions"] == int(batch.count[0])
        new_state = same_state(batch.states, info["task_id"])
        if finished_for:  # absorbing: no move, no reward, still terminated
            assert new_state == state and reward == 0.0 and terminated
            counts["absorbing"] += 1
        elif 0 <= action < len(successors):
            assert new_state == successors[action]  # independent of the pool: Task.successor_states
        else:  # no move: an invalid action, or a state without successors (stuck)
            assert new_state == state and (info["invalid"] or k == 0)
            counts["invalid"] += info["invalid"]
        if isinstance(obs, mymyr.rl.gym.SymbolicObservation):
            assert obs.atoms == frozenset(new_state.atoms())
        state = new_state
        counts["steps"] += 1
        counts["terminated"] += terminated and not finished_for
        counts["truncated"] += truncated
        if terminated and finished_for < int(rng.integers(0, 3)):
            finished_for += 1
            continue
        if terminated or truncated:
            info, batch, state, finished_for = reset()
            task = env.task
    return counts


@needs_gym
@needs_frontend
@pytest.mark.parametrize("name", [*CLASSICAL, "counters"])
def test_contract_equals_the_cpu_pool(name):
    from mymyr.rl import gym as mg

    t = table(name)
    config = contract(dead_end_terminal=name != "philosophers")
    env = mg.Env(t, **config)
    counts = drive(env, reference_pool(t, **config), 1000, seed=11, rng=np.random.default_rng(5))
    assert counts["steps"] == 1000 and counts["terminated"] + counts["truncated"] > 0 and counts["invalid"] > 0
    if name == "deadend":
        assert counts["terminated"] > 0 and counts["absorbing"] > 0
    env.close()


@needs_gym
@needs_frontend
def test_contract_solved_resets_and_dead_ends():
    from mymyr.rl import gym as mg

    tasks = [toggle_task(), toggle_task(init="(q)"), toggle_task(init="(p)", goal="(not (p))")]
    t = rl.TaskTable(tasks)
    config = contract(max_steps=5)
    env = mg.Env(t, **config)
    counts = drive(env, reference_pool(t, **config), 1000, seed=2, rng=np.random.default_rng(1))
    assert counts["solved_resets"] > 0 and counts["absorbing"] > 0 and counts["terminated"] > 0

    obs, info = env.reset(options={"task_id": 1})
    assert info["is_goal"] and obs.atoms == frozenset(tasks[1].initial_state.atoms())
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2 == obs and reward == 0.0 and terminated and not truncated and info["steps"] == 0


@needs_gym
@needs_frontend
@pytest.mark.parametrize("dead_end", ["no_successors", "none"])
def test_dead_end_start(dead_end):
    from mymyr.rl import gym as mg

    t = table("deadend")
    env = mg.Env(t, dead_end=dead_end, dead_end_reward=-1.0, max_steps=3)
    obs, info = env.reset(options={"task_id": 1})  # no applicable action at the start
    assert info["num_actions"] == 0 and not info["action_mask"].any()
    assert info["terminated"] == (dead_end == "no_successors")
    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2 == obs and reward == 0.0 and not truncated
    if dead_end == "none":  # stuck, not terminal: episodes end by truncation
        assert not terminated
    else:  # a terminal dead end at the reset: absorbing
        assert terminated and info["steps"] == 0


@needs_gym
@needs_frontend
@pytest.mark.parametrize("name", ["blocks", "gripper", "deadend", "counters"])
def test_vector_env_same_step_equals_the_cpu_pool(name):
    from mymyr.rl import gym as mg

    t = table(name)
    n, seed, config = 7, 23, contract()
    venv = mg.VectorEnv(t, n, autoreset_mode="same_step", threads=2, observation="bits", **config)
    pool = rl.CpuEnvPool(t, n, threads=3, seed=seed, sampler="auto", autoreset=True, words=t.words, **config)
    _, info = venv.reset(seed=seed)
    expected = rl.InstanceSampler(len(t)).sample(seed, np.arange(n), np.zeros(n)) if len(t) > 1 else np.zeros(n)
    np.testing.assert_array_equal(info["task_id"], expected)
    rng = np.random.default_rng(0)
    done_total = 0
    for _ in range(1000 // n + 1):
        actions = rng.integers(-1, info["num_actions"] + 1)
        obs, reward, terminated, truncated, info = venv.step(actions)
        batch = pool.step(actions=actions.astype(np.int64))
        np.testing.assert_array_equal(reward, batch.reward)
        np.testing.assert_array_equal(terminated, batch.terminated)
        np.testing.assert_array_equal(truncated, batch.truncated)
        np.testing.assert_array_equal(info["invalid"], batch.invalid)
        np.testing.assert_array_equal(info["task_id"], batch.task_ids)
        np.testing.assert_array_equal(info["num_actions"], batch.count)
        assert venv.states() == rl.decode(t, np.asarray(batch.states), batch.task_ids)
        done = np.flatnonzero(terminated | truncated)
        done_total += len(done)
        if len(done):
            np.testing.assert_array_equal(info["_final_obs"], terminated | truncated)
        for i in done:
            final = np.asarray(batch.final_states)[i]
            atoms = obs["atoms"] if isinstance(obs, dict) else obs
            final_obs = info["final_obs"][i]
            final_bits = final_obs["atoms"] if isinstance(final_obs, dict) else final_obs
            expect = np.unpackbits(final[: t.words].view(np.uint8), count=final_bits.shape[0], bitorder="little")
            np.testing.assert_array_equal(final_bits, expect)
            assert atoms.shape[0] == n
    assert done_total > 0


@needs_gym
@needs_frontend
@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("name", ["blocks", "gripper", "logistics"])
def test_vector_env_equals_sync_vector_env(name, observation):
    import gymnasium as gym
    from gymnasium.utils.env_checker import data_equivalence
    from mymyr.rl import gym as mg

    t = table(name)
    n, seed, config = 5, 7, contract(max_steps=9)
    venv = mg.VectorEnv(t, n, observation=observation, **config)
    sync = gym.vector.SyncVectorEnv(
        [lambda i=i: mg.Env(t, observation=observation, env_id=i, **config) for i in range(n)]
    )
    assert venv.single_observation_space == sync.single_observation_space
    assert venv.single_action_space == sync.single_action_space
    vobs, vinfo = venv.reset(seed=seed)
    sobs, sinfo = sync.reset(seed=[seed] * n)
    rng = np.random.default_rng(3)
    for _ in range(200):
        if observation == "graph":  # SyncVectorEnv stacks the Box arrays and tuples the Sequence ones
            synced = [{key: value[i] for key, value in sobs.items()} for i in range(n)]
            assert all(data_equivalence(a, b, exact=True) for a, b in zip(_split_graphs(vobs, n), synced))
        else:
            assert data_equivalence(vobs, sobs, exact=True)
        np.testing.assert_array_equal(vinfo["task_id"], sinfo["task_id"])
        np.testing.assert_array_equal(vinfo["action_mask"], sinfo["action_mask"])
        actions = rng.integers(0, vinfo["num_actions"] + 1)
        vobs, vr, vte, vtr, vinfo = venv.step(actions)
        sobs, sr, ste, str_, sinfo = sync.step(actions)
        np.testing.assert_array_equal(vr, sr)
        np.testing.assert_array_equal(vte, ste)
        np.testing.assert_array_equal(vtr, str_)
    venv.close()
    sync.close()


@needs_gym
@needs_frontend
@pytest.mark.parametrize("observation", ["symbolic", "bits"])
def test_async_vector_env_runs_the_envs_in_worker_processes(observation):
    import gymnasium as gym
    from mymyr.rl import gym as mg

    domain, problems = paths("gripper")
    fns = [lambda i=i: mg.Env(domain=domain, problems=problems, observation=observation, env_id=i, max_steps=4)
           for i in range(2)]
    workers = gym.vector.AsyncVectorEnv(fns, shared_memory=observation == "bits")  # symbolic: pickled observations
    local = gym.vector.SyncVectorEnv(fns)

    def same(a, b):
        if observation == "symbolic":  # atoms of restored copies of the tasks: compared by their text
            assert [sorted(map(str, o.atoms)) for o in a] == [sorted(map(str, o.atoms)) for o in b]
        else:
            np.testing.assert_array_equal(a, b)

    try:
        aobs, _ = workers.reset(seed=[3, 3])
        sobs, _ = local.reset(seed=[3, 3])
        for k in range(10):
            same(aobs, sobs)
            actions = np.array([k % 3, (k + 1) % 3])
            aobs, ar, ate, atr, _ = workers.step(actions)
            sobs, sr, ste, st, _ = local.step(actions)
            assert ar.tolist() == sr.tolist() and ate.tolist() == ste.tolist() and atr.tolist() == st.tolist()
    finally:
        workers.close()
        local.close()


def _split_graphs(obs, n):
    """The single-graph dicts of a batch of n graphs (entity ids made local)."""
    out = []
    R = (len(obs["relation_offsets"]) - 1) // n
    for g in range(n):
        lo_t = obs["relation_offsets"][g * R]
        lo_a, hi_a = obs["relation_arg_offsets"][g * R], obs["relation_arg_offsets"][(g + 1) * R]
        e0, e1 = obs["entity_offsets"][g], obs["entity_offsets"][g + 1]
        one = {
            "entity_offsets": np.array([0, e1 - e0], obs["entity_offsets"].dtype),
            "num_objects": obs["num_objects"][g : g + 1],
            "relation_arg_offsets": obs["relation_arg_offsets"][g * R : (g + 1) * R + 1] - lo_a,
            "relation_args": obs["relation_args"][lo_a:hi_a] - e0,
            "relation_offsets": obs["relation_offsets"][g * R : (g + 1) * R + 1] - lo_t,
            "task_ids": obs["task_ids"][g : g + 1],
        }
        out.append({k: one[k] for k in obs})
    return out


@needs_gym
@needs_frontend
def test_vector_env_next_step_and_disabled_autoreset():
    from mymyr.rl import gym as mg

    tasks = [toggle_task(), toggle_task(init="(q)")]
    t = rl.TaskTable(tasks)
    venv = mg.VectorEnv(t, 2, max_steps=4)
    obs, info = venv.reset(options={"task_ids": [0, 1]})
    assert info["is_goal"].tolist() == [False, True]
    _, reward, terminated, truncated, info = venv.step(np.array([0, 0]))
    assert terminated.tolist() == [False, True] and reward.tolist() == [0.0, 0.0]
    # next step: env 1 restarts (its action ignored): reward 0, no termination
    venv.set_next_task_ids([0, 0])
    _, reward, terminated, truncated, info = venv.step(np.array([0, 5]))
    assert not terminated.any() and info["task_id"].tolist() == [0, 0] and not info["invalid"][1]

    off = mg.VectorEnv(t, 2, max_steps=4, autoreset_mode="disabled")
    off.reset(options={"task_ids": [1, 0]})
    for _ in range(3):
        _, reward, terminated, _, info = off.step(np.array([0, 0]))
        assert terminated[0] and reward[0] == 0.0  # absorbing until reset
    obs, info = off.reset(options={"reset_mask": np.array([True, False]), "task_ids": [0, 0]})
    assert info["task_id"].tolist() == [0, 0] and info["steps"].tolist() == [0, 3]


@needs_gym
@needs_frontend
@pytest.mark.parametrize("mode", ["same_step", "next_step"])
def test_vector_env_next_task_ids_wait_for_each_envs_autoreset(mode):
    from mymyr.rl import gym as mg

    # instance 0: "up" then "finish" reach the goal; 1: solved at the start; 2: starts with (p)
    t = rl.TaskTable([toggle_task(), toggle_task(init="(q)"), toggle_task(init="(p)")])
    venv = mg.VectorEnv(t, 2, max_steps=10, autoreset_mode=mode, sampler=None)
    venv.reset(options={"task_ids": [1, 0]})
    venv.set_next_task_ids([2, 2])
    seen = []
    for actions in ([0, 0], [0, 1], [0, 0], [0, 0]):  # env 0 ends at the first step, env 1 at the second
        _, _, terminated, _, info = venv.step(np.array(actions))
        seen.append((info["task_id"].tolist(), terminated.tolist()))
    if mode == "same_step":  # the ending step starts the next episode
        assert [ids for ids, _ in seen[:2]] == [[2, 0], [2, 2]]
    else:  # the step after the ending one starts it
        assert [ids for ids, _ in seen[:3]] == [[1, 0], [2, 0], [2, 2]]
    assert seen[0][1][0] and seen[1][1][1]
    venv.set_next_task_ids([-1, 0])
    with pytest.raises(ValueError, match="next task ids"):
        venv.set_next_task_ids([3, 0])
    venv.set_next_task_ids(None)
    np.testing.assert_array_equal(venv._next_task_ids, [-1, -1])


# ------------------------------------------------------------------------------------------------ seeding and draws


@needs_gym
@needs_frontend
def test_seeding_and_instance_draws():
    from mymyr.rl import gym as mg

    t = table("gripper")
    env = mg.Env(t, env_id=4, max_steps=3)
    sampler = rl.InstanceSampler(len(t))
    draws = []
    env.reset(seed=99)
    draws.append(env.task_id)
    for _ in range(19):
        env.reset()
        draws.append(env.task_id)
    expected = sampler.sample(99, np.full(20, 4), np.arange(20))
    assert draws == expected.tolist()

    weighted = mg.Env(t, sampler=rl.InstanceSampler(3, weights=[0.0, 1.0, 3.0]))
    seen = set()
    weighted.reset(seed=1)
    for _ in range(60):
        seen.add(weighted.reset()[1]["task_id"])
    assert seen == {1, 2}

    fixed = mg.Env(t, sampler=None)
    assert {fixed.reset(seed=s)[1]["task_id"] for s in range(5)} == {0}
    fixed.fix_problem_index(2)
    assert {fixed.reset()[1]["task_id"] for _ in range(5)} == {2}
    assert fixed.reset(options={"task_id": 1})[1]["task_id"] == 1
    fixed.fix_problem_index(None)
    assert fixed.reset()[1]["task_id"] == 1  # without a sampler: the previous instance
    with pytest.raises(ValueError, match="outside"):
        fixed.reset(options={"task_id": 3})
    with pytest.raises(ValueError, match="unknown reset options"):
        fixed.reset(options={"goal": 1})


def _trajectory(env, seed, steps):
    out = []
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, info["num_actions"] + 1)))
        out.append((sorted(map(str, obs.atoms)), reward, terminated, truncated, info["task_id"]))
        if terminated or truncated:
            obs, info = env.reset()
    return out


@needs_gym
@needs_frontend
def test_same_seed_same_trajectory():
    from mymyr.rl import gym as mg

    env = mg.Env(table("blocks"), max_steps=7)
    assert _trajectory(env, 5, 300) == _trajectory(env, 5, 300)
    assert _trajectory(env, 5, 300) == _trajectory(mg.Env(table("blocks"), max_steps=7), 5, 300)


@needs_gym
@needs_frontend
def test_same_seed_same_trajectory_across_processes():
    domain, problems = paths("gripper")
    code = (
        "import hashlib, numpy as np\nfrom mymyr.rl import gym as mg\n"
        f"env = mg.Env(domain={domain!r}, problems={problems!r}, max_steps=6)\n"
        "obs, info = env.reset(seed=17)\nrng = np.random.default_rng(17)\nh = hashlib.sha256()\n"
        "for _ in range(500):\n"
        "    obs, r, te, tr, info = env.step(int(rng.integers(0, info['num_actions'] + 1)))\n"
        "    h.update(repr((sorted(map(str, obs.atoms)), r, te, tr, info['task_id'])).encode())\n"
        "    if te or tr:\n        obs, info = env.reset()\n"
        "print(h.hexdigest())\n"
    )
    import os

    digests = {
        run_code(code, capture_output=True, env={**os.environ, "PYTHONHASHSEED": str(s)}).stdout.strip()
        for s in (1, 2)
    }
    assert len(digests) == 1 and len(next(iter(digests))) == 64


# ------------------------------------------------------------------------------------------------ spaces and modes


@needs_gym
@needs_frontend
def test_symbolic_observation_fields_and_space():
    import gymnasium as gym
    from mymyr.rl import gym as mg

    t = table("philosophers")
    env = mg.Env(t, objects=True, goal=True, static=True, derived=True)
    obs, _ = env.reset(seed=0)
    task, state = env.task, env.state
    assert obs.atoms == frozenset(state.atoms()) | frozenset(state.derived_atoms())
    assert obs.objects == tuple(task.objects) and all(isinstance(o, str) for o in obs.objects)
    assert obs.goal == task.goal_condition
    assert {str(a) for a in obs.static} == {str(a) for a in task.formalism.static_init} and obs.static
    assert obs.numeric is None and hash(obs) == hash(obs._replace()) and obs == obs._replace()
    space = env.observation_space
    assert obs in space and space.sample() in space
    assert obs._replace(numeric=(1.0,)) not in space and obs._replace(static=None) not in space
    plain = mg.Env(t, objects=False, goal=False).reset(seed=0)[0]
    assert plain.objects is None and plain.goal is None and plain not in space
    copied = copy.deepcopy(space)
    assert copied == space and copied._observer is space._observer
    restored = pickle.loads(pickle.dumps(space))  # equal by content: the spaces of worker processes
    assert restored == space and hash(restored) == hash(space) and restored._observer is not space._observer
    assert sorted(map(str, pickle.loads(pickle.dumps(obs)).atoms)) == sorted(map(str, obs.atoms))
    batched = gym.vector.utils.batch_space(space, 3)
    assert isinstance(batched, gym.spaces.Tuple) and all(s == space for s in batched)


@needs_gym
@needs_frontend
def test_graph_and_bits_observations_match_their_sources():
    from mymyr import encoding
    from mymyr.rl import gym as mg

    t = table("blocks")
    graph = mg.Env(t, observation="graph")
    bits = mg.Env(t, observation="bits")
    encoder = encoding.FlatRelationEncoder(t, goal_derivations=(), include_actions=False, emit_counts=False,
                                           entity_arrays="auto")
    gobs, ginfo = graph.reset(seed=4)
    bobs, _ = bits.reset(seed=4)
    for _ in range(30):
        expected = encoder.encode([graph.state], [ginfo["task_id"]]).as_dict()
        assert set(gobs) == {k for k, v in expected.items() if isinstance(v, np.ndarray)}
        for key, value in gobs.items():
            np.testing.assert_array_equal(value, expected[key])
        width = bits.observation_space.shape[0]
        row = graph.task.encode([graph.state])[0]
        want = np.unpackbits(row.view(np.uint8), bitorder="little")[:width]
        np.testing.assert_array_equal(bobs, np.pad(want, (0, width - len(want))))
        action = int(np.flatnonzero(ginfo["action_mask"])[0])
        gobs, *_, ginfo = graph.step(action)
        bobs, *_ = bits.step(action)
    with pytest.raises(ValueError, match="one domain"):
        mg.Env(rl.TaskSuite.group([table("blocks")[0], table("gripper")[0]]), observation="graph")


@needs_gym
@needs_frontend
def test_index_and_ground_actions():
    from mymyr.rl import gym as mg

    t = table("blocks")
    index = mg.Env(t)
    ground = mg.Env(t, action="ground")
    _, info = index.reset(seed=8, options={"task_id": 0})  # blocks-8-0: 4 actions at the start
    ground.reset(seed=8, options={"task_id": 0})
    assert index.max_actions == max(rl.successor_bounds(t))
    actions = index.applicable_actions()
    assert actions == ground.applicable_actions() and info["action_mask"].sum() == len(actions)
    np.testing.assert_array_equal(index.action_masks(), info["action_mask"].astype(bool))
    second = actions[1]
    a = index.step(1)
    b = ground.step(second)
    assert a[0] == b[0] and a[1:4] == b[1:4]
    # ground actions as text, inapplicable ones, ones of another task
    obs, *_, info = ground.step(str(ground.applicable_actions()[0]))
    assert not info["invalid"]
    inapplicable = [a for a in ground.task.applicable_actions(ground.task.initial_state) if a not in
                    ground.applicable_actions()]
    if inapplicable:
        before = ground.state
        *_, info = ground.step(inapplicable[0])
        assert info["invalid"] and ground.state == before
    # an action of another instance is read by its names; names the instance lacks make no move
    small = t[1]
    action = small.applicable_actions(small.initial_state)[0]
    applicable = ground.applicable_actions()
    *_, info = ground.step(action)
    assert info["invalid"] == (ground.task.action(str(action)) not in applicable)
    other = table("gripper")[0]
    before = ground.state
    *_, info = ground.step(other.applicable_actions(other.initial_state)[0])
    assert info["invalid"] and ground.state == before
    *_, info = ground.step("(no-such-schema a)")
    assert info["invalid"]
    with pytest.raises(TypeError, match="ground action"):
        ground.step(3)
    assert ground.action_space.sample() in ground.action_space
    assert other.applicable_actions(other.initial_state)[0] not in ground.action_space
    # a cap below a state's successors puts the rest out of reach
    capped = mg.Env(t, max_actions=2)
    _, info = capped.reset(seed=8, options={"task_id": 0})
    assert info["num_actions"] > 2 and info["action_mask"].tolist() == [1, 1]
    *_, info = capped.step(3)
    assert info["invalid"]
    with pytest.raises(TypeError, match="int"):
        capped.step(1.5)


@needs_gym
@needs_frontend
def test_render_ansi():
    from mymyr.rl import gym as mg

    env = mg.Env(table("blocks"), render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    task = env.task
    assert text.startswith(f"instance {env.task_id} ({task.domain_name} {task.problem_name}), step 0")
    assert all(str(atom) in text for atom in env.state.atoms())
    assert all(str(lit) in text for lit in task.goal_condition.literals)
    assert mg.Env(table("blocks")).render() is None
    with pytest.raises(ValueError, match="render_mode"):
        mg.Env(table("blocks"), render_mode="rgb_array")


@needs_gym
@needs_frontend
def test_registration_and_make():
    import gymnasium as gym
    from mymyr.rl import gym as mg

    domain, problems = paths("gripper")
    env = gym.make("mymyr/PDDL-v0", domain=domain, problems=problems, max_steps=5)
    assert isinstance(env.unwrapped, mg.Env) and env.unwrapped.num_instances == 3
    env_id = mg.register_pddl(domain, problems, observation="bits", max_episode_steps=3)
    assert env_id == "mymyr/gripper-v0"
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.int8
    truncated = False
    for _ in range(3):
        *_, truncated, _ = env.step(0)
    assert truncated  # Gymnasium's TimeLimit
    venv = gym.make_vec(env_id, num_envs=4, vectorization_mode="vector_entry_point")
    assert isinstance(venv, mg.VectorEnv) and venv.num_envs == 4
    obs, info = venv.reset(seed=0)
    assert obs.shape[0] == 4
    venv.close()
    gym.registry.pop(env_id)
    assert mg.register_pddl(DATA / "pddl/blocks/domain.pddl", [DATA / "pddl/blocks/probBLOCKS-8-0.pddl"],
                            name="eight", version=2) == "mymyr/eight-v2"
    gym.registry.pop("mymyr/eight-v2")


@needs_gym
@needs_frontend
def test_constructor_arguments():
    from mymyr.rl import gym as mg

    t = table("blocks")
    with pytest.raises(ValueError, match="domain= and problems="):
        mg.Env()
    with pytest.raises(ValueError, match="either"):
        mg.Env(t, domain="x")
    with pytest.raises(ValueError, match="observation"):
        mg.Env(t, observation="pixels")
    with pytest.raises(ValueError, match="action"):
        mg.Env(t, action="literal")
    with pytest.raises(ValueError, match="index action mode"):
        mg.Env(t, action="ground", max_actions=4)
    with pytest.raises(ValueError, match="autoreset_mode"):
        mg.VectorEnv(t, 2, autoreset_mode="sometimes")
    with pytest.raises(RuntimeError, match="reset"):
        mg.Env(t).step(0)
    env = mg.Env(t.tasks[0])
    assert env.num_instances == 1 and env.reset(seed=0)[1]["task_id"] == 0
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
