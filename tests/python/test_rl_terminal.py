"""Solved resets and absorbing goal/dead-end dynamics across the RL entry points."""

import numpy as np
import pytest

import mymyr
from mymyr import rl

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def task_from_strings(domain, problem):
    return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))


@pytest.fixture(params=[("cpu", "auto"), ("cuda", "auto"), ("cuda", "general")])
def device_path(request):
    device, path = request.param
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available() or not pytest.importorskip("mymyr.cuda", exc_type=ImportError).available():
            pytest.skip("no visible CUDA device")
    return device, path


@pytest.fixture(params=["fluent", "derived", "numeric"])
def solved_task(request):
    if request.param == "numeric":
        return task_from_strings(
            """(define (domain solved) (:requirements :strips :numeric-fluents)
            (:predicates (p)) (:functions (level))
            (:action leave :parameters () :precondition (>= (level) 1)
             :effect (decrease (level) 1)))""",
            """(define (problem solved) (:domain solved) (:init (= (level) 1))
            (:goal (>= (level) 1)))""",
        )
    derived = "(:derived (done) (p object))" if request.param == "derived" else ""
    requirements = ":strips :derived-predicates" if derived else ":strips"
    goal = "done" if derived else "p object"
    return task_from_strings(
        f"""(define (domain solved) (:requirements {requirements}) (:constants object) (:predicates (p ?object) (done)) {derived}
        (:action leave :parameters (?object) :precondition (p ?object) :effect (not (p ?object))))""",
        f"""(define (problem solved) (:domain solved) (:init (p object)) (:goal ({goal})))""",
    )


def test_pool_reports_solved_reset_and_absorbs(solved_task):
    assert solved_task.initial_state.is_goal()
    assert rl.expand(solved_task, [solved_task.initial_state]).counts[0] > 0
    pool = rl.CpuEnvPool(solved_task, num_envs=2, threads=2, autoreset=False, step_reward=-1, goal_reward=5)
    reset = pool.reset()
    assert reset.goal.tolist() == [True, True]
    assert reset.terminated.tolist() == [True, True]
    assert not reset.stepped and reset.reward is None
    for actions in [np.zeros(2, np.int64), np.full(2, 99, np.int64), None]:
        step = pool.step(actions=actions)
        assert np.array_equal(step.states, reset.states)
        assert np.array_equal(step.final_states, reset.states)
        assert step.goal.all() and step.terminated.all()
        assert not step.truncated.any() and not step.invalid.any()
        assert not step.reward.any() and not step.steps.any()
        assert np.all(step.schema == -1) and np.all(step.binding == -1)
        assert np.array_equal(step.count, reset.count)


def test_pool_goal_arrival_pays_once():
    task = task_from_strings(
        """(define (domain arrival) (:requirements :strips) (:predicates (p) (done))
        (:action finish :parameters () :precondition (p) :effect (done)))""",
        """(define (problem arrival) (:domain arrival) (:init (p)) (:goal (done)))""",
    )
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, autoreset=False, step_reward=-1, goal_reward=5)
    assert not pool.reset().goal[0]
    first = pool.step(actions=np.zeros(1, np.int64))
    assert first.goal[0] and first.terminated[0] and first.reward[0] == 4
    second = pool.step(actions=np.zeros(1, np.int64))
    assert second.goal[0] and second.terminated[0] and second.reward[0] == 0
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.steps, second.steps)


def test_partial_pool_reset_uses_each_instances_goal():
    domain = mymyr.Domain.from_string(
        """(define (domain mixed) (:requirements :strips :numeric-fluents) (:predicates (p))
        (:functions (level)) (:action drain :parameters () :precondition (>= (level) 1)
         :effect (decrease (level) 1)))"""
    )
    tasks = [
        mymyr.Task(domain.instantiate_string(
            f"""(define (problem mixed-{level}) (:domain mixed) (:init (= (level) {level}))
            (:goal (>= (level) 1)))"""
        ))
        for level in [0, 1]
    ]
    pool = rl.CpuEnvPool(rl.TaskTable(tasks), num_envs=2, threads=2, autoreset=False)
    reset = pool.reset(task_ids=np.array([0, 1], np.int32))
    assert reset.goal.tolist() == [False, True]
    reset = pool.reset(env_ids=np.array([1], np.uint32), task_ids=np.array([0], np.int32))
    assert reset.goal.tolist() == [False] and reset.terminated.tolist() == [True]


def toggle_table():
    """Instance 0 starts unsolved (goal (q), reached by up then finish), instance 1 starts solved."""
    domain = mymyr.Domain.from_string(
        """(define (domain toggles) (:requirements :strips :negative-preconditions) (:predicates (p) (q))
        (:action up :parameters () :precondition (not (p)) :effect (p))
        (:action down :parameters () :precondition (p) :effect (not (p)))
        (:action finish :parameters () :precondition (p) :effect (q)))"""
    )
    inits = ["", "(q)"]
    return rl.TaskTable([
        mymyr.Task(domain.instantiate_string(f"(define (problem p{i}) (:domain toggles) (:init {init}) (:goal (q)))"))
        for i, init in enumerate(inits)
    ])


def test_batched_resets_report_solved_states(device_path):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    device, path = device_path
    env = rt.BatchedEnv(toggle_table(), 4, task_ids=[0, 1, 1, 0], device=device, path=path, autoreset=False)
    assert env.at_goal.tolist() == [False, True, True, False]
    assert env.observation().at_goal.tolist() == [False, True, True, False]
    mask = torch.tensor([True, True, False, False], device=device)
    env.reset(mask, task_ids=torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=device))
    assert env.at_goal.tolist() == [True, False, True, False]
    step = env.step(torch.zeros(4, dtype=torch.int64, device=device))
    assert step.terminated.tolist() == [True, False, True, False]
    assert step.reward.tolist() == [0.0, 0.0, 0.0, 0.0] and step.schema[0] == -1 and step.schema[2] == -1
    assert step.at_goal.tolist() == [True, False, True, False]
    env.states.copy_(env.states[torch.tensor([0, 0, 0, 0], device=device)])
    env.task_ids.fill_(1)
    env.refresh()
    assert env.at_goal.all()


def test_batched_resets_report_given_goals_that_hold(device_path):
    torch = pytest.importorskip("torch")
    rt = pytest.importorskip("mymyr.rl.torch")
    device, path = device_path
    table = toggle_table()
    env = rt.BatchedEnv(table, 2, task_ids=[0, 0], goals=True, device=device, path=path, autoreset=False)
    assert not env.at_goal.any()
    pos, neg = rl.goal_masks(table, np.array([0, 0], np.int32))
    pos[0], neg[0] = 0, 0  # the empty goal holds in every state
    as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int64)).to(device)
    env.reset(goal_pos=as_tensor(pos), goal_neg=as_tensor(neg))
    assert env.at_goal.tolist() == [True, False]
    step = env.step(torch.zeros(2, dtype=torch.int64, device=device))
    assert step.terminated.tolist() == [True, False] and step.goal.tolist() == [True, False]
    assert int(step.schema[0]) == -1 and int(step.schema[1]) >= 0


@pytest.mark.needs_gil
def test_torchrl_resets_report_given_goals_that_hold(device_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchrl")
    from mymyr.rl.torch import PlanningEnv

    device, path = device_path
    table = toggle_table()
    env = PlanningEnv(table, 2, goals=True, task_ids=[0, 0], device=device, path=path)
    try:
        source = env.reset()
        assert not bool(source["terminated"].any())
        pos, neg = rl.goal_masks(table, np.array([0, 0], np.int32))
        pos[1], neg[1] = 0, 0
        as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int64)).to(device)
        result = env.reset(goal_pos=as_tensor(pos), goal_neg=as_tensor(neg))
        assert result["terminated"].reshape(-1).tolist() == [False, True]
    finally:
        env.close()


def jax_env_module(device):
    jax = pytest.importorskip("jax")
    rj = pytest.importorskip("mymyr.rl.jax")
    if device == "cuda":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("no visible JAX CUDA device")
    return rj


def test_jax_resets_report_solved_states_and_given_goals(device_path):
    device, path = device_path
    rj = jax_env_module(device)
    jnp = pytest.importorskip("jax.numpy")
    table = toggle_table()
    env = rj.Env(table, device=device, path=path, goals=True, autoreset=False)
    state, obs = env.reset(None, jnp.asarray([0, 1, 1, 0], jnp.int32))
    assert np.asarray(obs.at_goal).tolist() == [False, True, True, False]
    state, obs = env.reset_where(state, jnp.asarray([True, False, False, True]), task_ids=jnp.asarray([1, 0, 0, 1], jnp.int32))
    assert np.asarray(obs.at_goal).tolist() == [True, True, True, True]
    pos, neg = rl.goal_masks(table, np.array([0, 0, 0, 0], np.int32))
    pos[2], neg[2] = 0, 0
    state, obs = env.reset(
        None, jnp.asarray([0, 0, 0, 0], jnp.int32), goal_pos=jnp.asarray(pos.view(np.uint32)),
        goal_neg=jnp.asarray(neg.view(np.uint32)),
    )
    assert np.asarray(obs.at_goal).tolist() == [False, False, True, False]
    state, obs, reward, terminated, truncated, info = env.step(0, state, jnp.zeros(4, jnp.int32))
    assert np.asarray(terminated).tolist() == [False, False, True, False]
    assert np.asarray(info["schema"]).tolist()[2] == -1 and not np.asarray(reward).any()
    assert np.asarray(obs.at_goal).tolist() == [False, False, True, False]


def test_goal_query_requires_destination(solved_task):
    from mymyr._core import _rl_torch

    env = _rl_torch.Env(solved_task)
    with pytest.raises(TypeError, match="out"):
        env.goal(env.initial_states(), out=None)


def test_pool_relabelled_positive_negative_goal_masks():
    task = task_from_strings(
        """(define (domain masks) (:requirements :strips) (:predicates (p) (done))
        (:action leave :parameters () :precondition (p) :effect (and (not (p)) (done))))""",
        """(define (problem masks) (:domain masks) (:init (p)) (:goal (done)))""",
    )
    positive = np.asarray(task.encode([task.initial_state]))
    negative = np.asarray(rl.expand(task, [task.initial_state]).succ)
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, goals=True, autoreset=False, goal_reward=5)
    reset = pool.reset(goal_pos=positive, goal_neg=negative)
    assert reset.goal[0] and reset.terminated[0]
    step = pool.step(actions=np.zeros(1, np.int64))
    assert step.goal[0] and step.terminated[0] and step.reward[0] == 0
    assert np.array_equal(step.states, reset.states)


@pytest.mark.parametrize("dead_end,terminal,expected", [("no_successors", True, True), ("no_successors", False, False), ("none", True, False)])
def test_pool_initial_dead_end(dead_end, terminal, expected):
    task = task_from_strings(
        """(define (domain dead) (:requirements :strips) (:predicates (alive) (done))
        (:action finish :parameters () :precondition (alive) :effect (done)))""",
        """(define (problem dead) (:domain dead) (:init) (:goal (done)))""",
    )
    pool = rl.CpuEnvPool(task, num_envs=1, threads=1, autoreset=False, step_reward=-1,
                         dead_end=dead_end, dead_end_terminal=terminal, dead_end_reward=-5)
    reset = pool.reset()
    assert not reset.goal[0] and bool(reset.terminated[0]) == expected and reset.count[0] == 0
    step = pool.step()
    assert not step.goal[0] and bool(step.terminated[0]) == expected
    assert step.reward[0] == (0 if expected else -6 if dead_end == "no_successors" else -1)


@pytest.mark.needs_gil
def test_torchrl_solved_reset(solved_task, device_path):
    torch = pytest.importorskip("torch")
    device, path = device_path
    planning = pytest.importorskip("mymyr.rl.torch").PlanningEnv(solved_task, 2, device=device, path=path)
    try:
        reset = planning.reset()
        assert reset["terminated"].all() and reset["done"].all() and not reset["truncated"].any()
        reset["action"] = torch.zeros(2, dtype=torch.int64, device=device)
        step = planning.step(reset)["next"]
        assert torch.equal(reset["state"], step["state"])
        assert step["terminated"].all() and not step["reward"].any()
    finally:
        planning.close()


def test_mctx_solved_embedding_is_absorbing(solved_task, device_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mctx")
    device, path = device_path
    if device == "cuda":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("no visible JAX CUDA device")
    env = pytest.importorskip("mymyr.rl.jax").Env(solved_task, device=device, path=path, autoreset=False, step_reward=-1, goal_reward=5)
    state, _ = env.reset(None, 2)
    root, invalid = pytest.importorskip("mymyr.rl.jax").mctx_root(
        env, state, lambda params, observation: jnp.ones(2), num_actions=2
    )
    assert not np.asarray(root.value).any() and np.asarray(invalid).all()
    recurrent = pytest.importorskip("mymyr.rl.jax").mctx_recurrent_fn(
        env, lambda params, observation: jnp.ones(2), num_actions=2
    )
    result, reached = jax.jit(recurrent)(None, jax.random.key(0), jnp.zeros(2, jnp.int32), state)
    assert not np.asarray(result.reward).any() and not np.asarray(result.discount).any() and not np.asarray(result.value).any()
    for before, after in zip(state, reached):
        assert np.array_equal(before, after)


@pytest.mark.parametrize("dead_end,terminal", [("none", True), ("no_successors", False), ("no_successors", True)])
def test_mctx_dead_end_is_absorbing_even_with_nonterminal_env(dead_end, terminal):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mctx")
    rj = pytest.importorskip("mymyr.rl.jax")
    task = task_from_strings(
        """(define (domain dead-tree) (:requirements :strips) (:predicates (alive) (done))
        (:action finish :parameters () :precondition (alive) :effect (done)))""",
        """(define (problem dead-tree) (:domain dead-tree) (:init) (:goal (done)))""",
    )
    env = rj.Env(task, device="cpu", autoreset=False, dead_end=dead_end, dead_end_terminal=terminal, step_reward=-1)
    state, _ = env.reset(None, 2)
    recurrent = rj.mctx_recurrent_fn(env, lambda params, observation: jnp.ones(2), num_actions=2)
    result, reached = jax.jit(recurrent)(None, jax.random.key(0), jnp.zeros(2, jnp.int32), state)
    assert not np.asarray(result.reward).any() and not np.asarray(result.discount).any() and not np.asarray(result.value).any()
    for before, after in zip(state, reached):
        assert np.array_equal(before, after)


# ------------------------------------------------------------------------------ device environments equal the host's

CORRIDOR = """(define (domain corridor) (:requirements :strips)
 (:predicates (at ?x) (link ?x ?y) (alive))
 (:action move :parameters (?x ?y) :precondition (and (alive) (at ?x) (link ?x ?y)) :effect (and (not (at ?x)) (at ?y)))
 (:action die :parameters (?x) :precondition (and (alive) (at ?x)) :effect (not (alive))))"""
# solved initially (with successors), the goal two moves away, an initial dead end, a longer corridor; dying is a
# dead end
CORRIDOR_PROBLEMS = [
    "(:objects a c) (:init (alive) (at c) (link c a)) (:goal (at c))",
    "(:objects a b c) (:init (alive) (at a) (link a b) (link b a) (link b c)) (:goal (at c))",
    "(:objects a b c) (:init (at a) (link a b) (link b c)) (:goal (at c))",
    "(:objects a b c d e) (:init (alive) (at a) (link a b) (link b c) (link c d) (link d e) (link d c)) (:goal (at e))",
]
# nullary predicates only: the fast path's cache has no view words
TOGGLES = """(define (domain toggles) (:requirements :strips :negative-preconditions) (:predicates (p) (q))
 (:action up :parameters () :precondition (not (p)) :effect (p))
 (:action down :parameters () :precondition (p) :effect (not (p)))
 (:action finish :parameters () :precondition (p) :effect (q)))"""
TOGGLES_PROBLEMS = ["(:init) (:goal (q))", "(:init (q)) (:goal (q))", "(:init (p)) (:goal (and (q) (not (p))))"]
# a derived goal and numeric goals: the general path only, without goal masks
RELAY = """(define (domain relay) (:requirements :strips :derived-predicates)
 (:predicates (at ?x) (link ?x ?y) (exit ?x) (out ?x) (alive))
 (:derived (out ?x) (and (at ?x) (exit ?x)))
 (:action move :parameters (?x ?y) :precondition (and (alive) (at ?x) (link ?x ?y)) :effect (and (not (at ?x)) (at ?y)))
 (:action die :parameters (?x) :precondition (and (alive) (at ?x)) :effect (not (alive))))"""
RELAY_PROBLEMS = [
    "(:objects a c) (:init (alive) (at c) (exit c) (link c a)) (:goal (out c))",
    "(:objects a b c) (:init (alive) (at a) (exit c) (link a b) (link b a) (link b c)) (:goal (out c))",
    "(:objects a b) (:init (at a) (exit b) (link a b)) (:goal (out b))",
]
LEVEL = """(define (domain level) (:requirements :strips :numeric-fluents) (:predicates (alive)) (:functions (level))
 (:action up :parameters () :precondition (and (alive) (<= (level) 3)) :effect (increase (level) 1))
 (:action down :parameters () :precondition (and (alive) (>= (level) 1)) :effect (decrease (level) 1))
 (:action die :parameters () :precondition (alive) :effect (not (alive))))"""
LEVEL_PROBLEMS = [
    "(:init (alive) (= (level) 3)) (:goal (>= (level) 3))",
    "(:init (alive) (= (level) 0)) (:goal (>= (level) 2))",
    "(:init (= (level) 1)) (:goal (>= (level) 3))",
]
FAST_KINDS = ("corridor", "toggles", "suite")
CONFIG = dict(max_steps=5, step_reward=-1.0, goal_reward=5.0, dead_end_reward=-2.0)
DEAD_ENDS = [("no_successors", True), ("no_successors", False), ("none", True)]


def table_of(domain, problems):
    parsed = mymyr.Domain.from_string(domain)
    name = domain.split("(domain ")[1].split(")")[0]
    return rl.TaskTable([
        mymyr.Task(parsed.instantiate_string(f"(define (problem {name}-{k}) (:domain {name}) {body})"))
        for k, body in enumerate(problems)
    ])


@pytest.fixture(scope="module")
def absorbing_tables():
    corridor, toggles = table_of(CORRIDOR, CORRIDOR_PROBLEMS), table_of(TOGGLES, TOGGLES_PROBLEMS)
    return {"corridor": corridor, "toggles": toggles, "suite": rl.TaskSuite([corridor, toggles]),
            "derived": table_of(RELAY, RELAY_PROBLEMS), "numeric": table_of(LEVEL, LEVEL_PROBLEMS)}


def weights_of(instances):
    """A curriculum with one zero weight."""
    return [0 if k == 1 else 1 + k % 3 for k in range(instances)]


def absorbing_rollout(env, steps, autoreset):
    """Per step: the step's results, the goal query of the reached rows and the environment's arrays. Given actions
    (some outside [0, count)) on even steps, the random policy on odd ones; without autoreset, the finished rows reset
    (into sampled instances) after every third step."""
    import torch
    from test_rl_table import env_arrays, step_arrays

    out = []
    for step in range(steps):
        action = None if step % 2 else torch.from_numpy((np.arange(env.num_envs) * 7 + step) % 4).to(env.device)
        record = {f"step_{k}": v for k, v in step_arrays(env.step(action, final=True)).items()}
        record["goal_query"] = env.is_goal().cpu().numpy()
        if not autoreset and step % 3 == 2:
            done = torch.from_numpy(record["step_terminated"] | record["step_truncated"]).to(env.device)
            record["reset_count"] = env.reset(done).cpu().numpy().copy()
        record.update(env_arrays(env))
        assert np.array_equal(record["at_goal"], env.is_goal().cpu().numpy())  # the flags are the states' goal tests
        record.update(draws=env.draws.cpu().numpy().copy(), reset_counts=env._reset_counts.cpu().numpy().copy())
        out.append(record)
    return out


@pytest.mark.parametrize("dead_end, terminal", DEAD_ENDS)
@pytest.mark.parametrize("autoreset", [False, True])
def test_device_absorbing_rollout_equals_host(absorbing_tables, dead_end, terminal, autoreset):
    """Solved and dead-end rows on the device, general and fast path (both bucket launches), tables and a suite, with
    and without per-env goals, derived and numeric goals on the general path: absorbing steps, solved resets, sampled
    partial resets and autoresets, and the goal query equal the host environment's."""
    from test_rl_table import gpu_device

    dev = gpu_device()
    rt = pytest.importorskip("mymyr.rl.torch")
    n, steps = 48, 12
    for kind, table in absorbing_tables.items():
        fast = kind in FAST_KINDS
        assert (rt.fast_unsupported(table) == "") == fast
        instances = len(table)
        sampler = rl.InstanceSampler(instances, weights_of(instances)) if autoreset else "uniform"
        for goals in (False, True) if fast else (False,):
            cfg = dict(CONFIG, dead_end=dead_end, dead_end_terminal=terminal, autoreset=autoreset, goals=goals, seed=7,
                       task_ids=(np.arange(n) % instances).astype(np.int32), sampler=sampler)  # fmt: skip
            ref = absorbing_rollout(rt.BatchedEnv(table, n, **cfg), steps, autoreset)
            absorbed = [r["step_terminated"] & (r["step_schema"] < 0) & (r["step_reward"] == 0) for r in ref]
            assert any(a.any() for a in absorbed), kind
            launches = [("general", "widest")] + ([("auto", "widest"), ("auto", "per_bucket")] if fast else [])
            for path, launch in launches:
                env = rt.BatchedEnv(table, n, device=dev, path=path, **cfg)
                env.set_launch(launch)
                assert env.core.fast == (path == "auto")
                got = absorbing_rollout(env, steps, autoreset)
                env.check_errors()
                for step, (a, b) in enumerate(zip(got, ref)):
                    assert a.keys() == b.keys()
                    for field, value in b.items():
                        assert np.array_equal(a[field], value), (kind, goals, path, launch, step, field)


@pytest.mark.jax_on_cpu
def test_captured_sampled_rollout_equals_host(absorbing_tables):
    """Autoresets into sampled instances inside a CUDA graph: replays equal the host environment's steps."""
    from test_rl_table import gpu_device, step_arrays

    dev = gpu_device()
    rt = pytest.importorskip("mymyr.rl.torch")
    n, K, R = 40, 4, 3
    for kind in FAST_KINDS:
        table, instances = absorbing_tables[kind], len(absorbing_tables[kind])
        sampler = rl.InstanceSampler(instances, weights_of(instances))
        cfg = dict(CONFIG, sampler=sampler, seed=11, goals=kind != "toggles")
        host = rt.BatchedEnv(table, n, **cfg)
        ref = [step_arrays(host.step(final=True)) for _ in range(K * R)]
        env = rt.BatchedEnv(table, n, device=dev, **cfg)
        assert env.capturable
        assert np.array_equal(env.task_ids.cpu().numpy(), rt.BatchedEnv(table, n, **cfg).task_ids.numpy())
        graph = env.capture(K, final=True)
        for r in range(R):
            out = graph.replay()
            for field in (f for f in rt.StepResult._fields if f not in ("obs", "final_obs")):
                got = getattr(out, field).cpu().numpy()
                for k in range(K):
                    want = ref[r * K + k][field]
                    assert np.array_equal(got[k].view(want.dtype), want), (kind, r, k, field)
        env.check_errors()
        assert np.array_equal(env._reset_counts.cpu().numpy(), host._reset_counts.numpy())
        assert np.array_equal(env.draws.cpu().numpy(), host.draws.numpy())


@pytest.mark.parametrize("dead_end, terminal", DEAD_ENDS)
@pytest.mark.parametrize("autoreset", [False, True])
def test_jax_device_absorbing_scan_equals_cpu(absorbing_tables, dead_end, terminal, autoreset):
    """A jitted scan over solved and dead-end rows (given and random actions, sampled resets) on the GPU, general and
    fast path (derived and numeric goals: the general path), equals the same scan on the CPU."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rj = pytest.importorskip("mymyr.rl.jax")
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("no visible JAX CUDA device")
    n, K = 40, 10
    for kind, table in absorbing_tables.items():
        instances = len(table)
        sampler = rl.InstanceSampler(instances, weights_of(instances))
        runs = {}
        fast = kind in FAST_KINDS
        for device, path in ((jax.devices("cpu")[0], "auto"), (gpu, "general")) + (((gpu, "auto"),) if fast else ()):
            env = rj.Env(table, device=device, path=path, goals=kind in ("corridor", "suite"), autoreset=autoreset,
                         sampler=sampler,
                         dead_end=dead_end, dead_end_terminal=terminal, **CONFIG)  # fmt: skip
            if device == gpu:
                assert env.fast == (path == "auto")
            state, _ = env.reset(jax.random.key(3), n)

            def body(st, k, env=env):
                policy = (st.states.sum(-1) + st.steps.astype(jnp.uint32)) % (st.count.astype(jnp.uint32) + 1)
                act = jnp.where(k % 2 == 0, policy.astype(jnp.int32), env.random_actions(5, st))
                st, _, reward, term, trunc, info = env.step(5, st, act)
                if not autoreset:
                    st, _ = env.reset_where(st, (term | trunc) & (k % 3 == 2))
                # the device cache (counts, views) has no CPU counterpart
                return st, (reward, term, trunc, info["goal"], info["invalid"], info["schema"], info["final_state"],
                            st._replace(counts=None, views=None))

            final, ys = jax.jit(lambda st, body=body: jax.lax.scan(body, st, jnp.arange(K)))(state)
            env.check_errors()
            runs[(device.platform, path)] = [np.asarray(x) for x in jax.tree_util.tree_leaves(ys)]
        cpu = runs.pop(("cpu", "auto"))
        for key, arrays in runs.items():
            assert len(arrays) == len(cpu)
            for k, (a, b) in enumerate(zip(arrays, cpu)):
                assert np.array_equal(a, b), (kind, key, k)
