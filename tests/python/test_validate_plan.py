"""Plan validation (search.validate_plan): every CPU search's plans validate with the cost the search reported, as
Actions and as IPC text; mutated plans fail at the action and for the reason an independent replay finds."""

import random

import pytest

import mymyr
from conftest import ROOT, SMALL_TASKS, text_task
from mymyr import search
from mymyr.search import PlanError, Status

# numeric tasks GBFS solves quickly; tpp, barman, transport, woodworking, zenotravel and delivery have metrics other
# than the plan length (delivery's is a fluent, which SIW's subplans carry on from one to the next)
NUMERIC = ["cs-counters", "cs-delivery", "cs-drone", "cs-tpp", "m-barman", "m-refuel-adl", "m-transport",
           "m-woodworking", "m-zenotravel-numeric"]
# small PDDL tasks the novelty searches solve (AStarIW among them)
PDDL = ["gripper", "delivery", "visitall"] if hasattr(mymyr, "Domain") else []
TASKS = SMALL_TASKS + [f"numeric:{n}" for n in NUMERIC] + [f"pddl:{n}" for n in PDDL]
B = dict(max_states=20_000, max_seconds=10)
SEARCHES = {
    "brfs": lambda t: search.brfs(t, **B),
    "brfs_threads": lambda t: search.brfs(t, threads=4, **B),
    "brfs_layer_beam": lambda t: search.brfs(t, layer_order="goal_count", beam_width=16, **B),
    "astar": lambda t: search.astar(t, heuristic="max", **B),
    "astar_lmcut": lambda t: search.astar(t, heuristic="lmcut", **B),
    "astar_lazy": lambda t: search.astar(t, heuristic="ff", lazy=True, **B),
    "weighted_astar": lambda t: search.astar(t, heuristic="ff", weight=3, **B),
    "gbfs": lambda t: search.gbfs(t, heuristic="ff", **B),
    "gbfs_lazy_lmcount": lambda t: search.gbfs(t, heuristic="lmcount", lazy=True, **B),
    "beam": lambda t: search.beam(t, beam_width=64, **B),
    "ehc": lambda t: search.ehc(t, **B),
    "anytime": lambda t: search.anytime_astar(t, heuristic="ff", **B),
    "iw": lambda t: search.iw(t, max_width=2, **B),
    "iw_layer_beam": lambda t: search.iw(t, max_width=2, layer_order="goal_count", beam_width=8, **B),
    "iw_pass": lambda t: search.iw_pass(t, 2, **B),
    "siw": lambda t: search.siw(t, max_width=2, **B),
    "liw": lambda t: search.liw(t, max_width=2, **B),
    "abstracted_iw": lambda t: search.abstracted_iw(t, width=2, **B),
    "projective_iw": lambda t: search.projective_iw(t, **B),
    "rollout_iw": lambda t: search.rollout_iw(t, seed=1, **B),
    "astar_iw": lambda t: search.astar_iw(t, width=2, **B),
    "rollouts": lambda t: search.find_rollouts_parallel(t, [1, 2], max_width=2, threads=2, **B),
    "portfolio": lambda t: search.atomic_goal_portfolio(t, **B),
}


def load(name):
    kind, _, rest = name.rpartition(":")
    if kind == "numeric":
        return mymyr.Task.from_text(ROOT / "tests/data/numeric_tasks" / f"{rest}.txt")
    if kind == "pddl":
        d = ROOT / "tests/data/pddl" / rest
        return mymyr.Task.from_pddl(d / "domain.pddl", d / "test_problem.pddl")
    return text_task(name)


@pytest.fixture(scope="module")
def tasks():
    return {name: load(name) for name in TASKS}


def atoms(state):
    return sorted(map(str, state.atoms()))


@pytest.mark.parametrize("search_name", sorted(SEARCHES))
def test_every_plan_validates_with_the_reported_cost(tasks, search_name):
    if search_name == "astar_iw" and not PDDL:
        pytest.skip("AStarIW(2) solves only the PDDL tasks of this set, which need the PDDL front end")
    plans = 0
    for name, task in tasks.items():
        try:
            out = SEARCHES[search_name](task)
        except ValueError:  # a task feature the search refuses (AStarIW: numeric tasks)
            continue
        for r in out if search_name == "rollouts" else [out]:
            if r.status != Status.SOLVED:
                assert r.plan == [] and r.cost == 0.0 and r.goal_state is None, name
                continue
            plans += 1
            v = search.validate_plan(task, r.plan)
            assert v.valid and bool(v) and v.error == PlanError.NONE and v.message == "", (name, v.message)
            assert v.step == len(r.plan) and v.cost == pytest.approx(r.cost, rel=1e-9, abs=1e-9), name
            assert atoms(v.state) == atoms(r.goal_state) and v.plan == r.plan, name
            text = search.validate_plan(task, search.format_plan(task, r.plan))
            assert text.valid and text.cost == v.cost and text.plan == r.plan, name
    assert plans > 0


def replay(task, steps):
    """The first action of `steps` (IPC text) that is not applicable in the state the ones before it reached, as
    (index, state); (len(steps), state) when the goal does not hold after the last; None for a plan."""
    s = task.initial_state
    for i, text in enumerate(steps):
        by_text = {str(a): a for a in task.applicable_actions(s)}
        if text not in by_text:
            return i, s
        s = task.apply(s, by_text[text])
    return None if task.is_goal(s) else (len(steps), s)


def mutations(task, plan, rng):
    """(kind, steps) mutations of a plan's IPC text."""
    steps = [str(a) for a in plan]
    n = len(steps)
    out = [("dropped", steps[:i] + steps[i + 1:]) for i in sorted({0, n // 2, n - 1})]
    out += [("swapped", steps[:i] + [steps[i + 1], steps[i]] + steps[i + 2:])
            for i in sorted({0, n // 2, max(0, n - 2)}) if i + 1 < n and steps[i] != steps[i + 1]]
    objects = task.objects
    for i in sorted({0, n // 2, n - 1}):
        a = plan[i]
        if not a.original_arity:
            continue
        k = rng.randrange(a.original_arity)
        args = list(a.objects[:a.original_arity])
        args[k] = rng.choice([o for o in objects if o != args[k]] or objects)
        out.append(("wrong argument", steps[:i] + ["(" + " ".join([a.name, *args]) + ")"] + steps[i + 1:]))
    goal = task.initial_state
    for a in plan:
        goal = task.apply(goal, a)
    extra = sorted(task.applicable_actions(goal), key=str)
    if extra:
        out.append(("extra action", steps + [str(extra[rng.randrange(len(extra))])]))
    out.append(("unknown action", steps[:1] + ["(no-such-action)"] + steps[1:]))
    if n:
        a = plan[0]
        out.append(("arity", ["(" + " ".join([a.name, *a.objects[:a.original_arity], objects[0]]) + ")"] + steps[1:]))
    return out


@pytest.mark.parametrize("name", TASKS)
def test_mutated_plans_fail_where_a_replay_fails(tasks, name):
    task = tasks[name]
    r = search.gbfs(task, heuristic="ff")
    assert r.solved
    rng = random.Random(name)
    checked = 0
    for kind, steps in mutations(task, r.plan, rng):
        v = search.validate_plan(task, "\n".join(steps))
        expected = replay(task, steps)
        what = (kind, steps, v.message)
        if expected is None:
            assert v.valid, what
            continue
        i, state = expected
        checked += 1
        assert not v.valid and not v and v.step == i and atoms(v.state) == atoms(state), what
        if i == len(steps):
            assert v.error == PlanError.GOAL_NOT_REACHED and v.action == "" and v.line == 0, what
            assert v.message.startswith("the goal does not hold"), what
            continue
        assert v.line == i + 1 and v.message.startswith(f"plan[{i}] "), what
        if kind == "unknown action" and i == 1:
            assert v.error == PlanError.UNKNOWN_ACTION, what
        elif kind == "arity" and i == 0:
            assert v.error == PlanError.WRONG_ARITY, what
        else:
            assert v.error in (PlanError.NOT_APPLICABLE, PlanError.WRONG_TYPE, PlanError.UNKNOWN_OBJECT), what
        if v.error == PlanError.NOT_APPLICABLE and v.condition.count("(") == 1:
            assert v.condition not in atoms(state), what  # an atom of the precondition that is false
    assert checked >= 3


def test_plan_errors_name_their_reason(blocks):
    plan = search.gbfs(blocks, heuristic="ff").plan
    text = search.format_plan(blocks, plan)
    v = search.validate_plan(blocks, "; a comment\n\n" + text)
    assert v.valid and v.cost == len(plan) and v.step == len(plan)
    v = search.validate_plan(blocks, "(pick-up a\n")
    assert (v.error, v.step, v.line) == (PlanError.SYNTAX, 0, 1)
    v = search.validate_plan(blocks, "(pick-up nothing)\n")
    assert v.error == PlanError.UNKNOWN_OBJECT and "nothing" in v.message
    v = search.validate_plan(blocks, str(plan[0]).upper())  # names are compared without regard to case
    assert v.step == 1 and v.error == PlanError.GOAL_NOT_REACHED and v.plan == plan[:1]
    v = search.validate_plan(blocks, [])
    assert v.error == PlanError.GOAL_NOT_REACHED and v.message.startswith("the goal does not hold in the start state")
    assert "is false" in v.message and v.condition and v.cost == 0.0


def test_a_start_state_and_plans_from_files(blocks, tmp_path):
    s = blocks.initial_state
    first = blocks.applicable_actions(s)[0]
    after = blocks.apply(s, first)
    rest = search.gbfs(blocks, start=after).plan
    assert search.validate_plan(blocks, rest, start=after).valid
    f = tmp_path / "plan.txt"
    f.write_text(search.format_plan(blocks, [first, *rest]))
    v = search.validate_plan(blocks, f)
    assert v.valid and v.plan == [first, *rest]
    assert search.validate_plan(blocks, str(f)).error == PlanError.SYNTAX  # a str is plan text, not a path
    with pytest.raises(FileNotFoundError):
        search.validate_plan(blocks, tmp_path / "missing.txt")
    with pytest.raises(TypeError):
        search.validate_plan(blocks, 3)
    other = mymyr.Task.from_pddl(ROOT / "tests/data/pddl/blocks/domain.pddl", ROOT / "tests/data/pddl/blocks/probBLOCKS-8-0.pddl")
    with pytest.raises(ValueError):
        search.validate_plan(other, [first])


def test_costs_follow_the_metric(tasks):
    for name in ("numeric:cs-tpp", "numeric:m-transport", "numeric:m-zenotravel-numeric"):
        task = tasks[name]
        r = search.gbfs(task, heuristic="ff")
        v = search.validate_plan(task, r.plan)
        assert v.valid and v.cost == pytest.approx(r.cost) and v.cost != len(r.plan), name
