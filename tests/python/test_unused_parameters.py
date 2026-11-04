"""Action parameters that nothing but their type mentions: collapsed by default (collapse_unused_parameters=True)."""

import pickle
import subprocess
import sys

import numpy as np
import pytest

import mymyr
from mymyr import rl, search

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="needs the PDDL front end")

# x has five parameters that no condition or effect mentions; y reaches the goal after it
FAN = ("(define (domain fan) (:requirements :adl) (:predicates (g) (h))"
       " (:action x :parameters (?a ?b ?c ?d ?e) :precondition (and) :effect (h))"
       " (:action y :parameters () :precondition (h) :effect (g)))")

LIGHTS = """(define (domain lights) (:requirements :typing)
  (:types switch room)
  (:predicates (on) (lit) (hot ?r - room))
  (:action press :parameters (?s - switch ?r - room) :precondition (on) :effect (lit))
  (:action heat :parameters (?s - switch ?r - room) :precondition (on) :effect (hot ?r)))"""
LIGHTS_PROBLEM = """(define (problem p) (:domain lights) (:objects s1 s2 - switch r1 r2 r3 - room)
  (:init (on)) (:goal (and (lit) (hot r3))))"""


def fan_problem(n):
    return ("(define (problem q) (:domain fan) (:objects " + " ".join(f"o{i}" for i in range(n)) +
            ") (:init) (:goal (g)))")


def fan(n, collapse=True):
    return mymyr.Task(mymyr.Domain.from_string(FAN).instantiate_string(fan_problem(n),
                                                                        collapse_unused_parameters=collapse))


def lights(collapse=True):
    return mymyr.Task(mymyr.Domain.from_string(LIGHTS).instantiate_string(LIGHTS_PROBLEM,
                                                                           collapse_unused_parameters=collapse))


def test_one_expansion_generates_the_collapsed_actions():
    collapsed = search.astar(fan(4), heuristic="blind")
    full = search.astar(fan(4, False), heuristic="blind")
    assert collapsed.status == full.status == search.Status.SOLVED
    assert [str(a) for a in collapsed.plan] == ["(x o0 o0 o0 o0 o0)", "(y)"]
    assert collapsed.stats.generated == 3  # x from the start, x and y from its successor
    assert full.stats.generated == 2 * 4 ** 5 + 1
    assert collapsed.cost == full.cost


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_twenty_objects_stay_small():
    # Without collapsing, one expansion of this task generates 3.2 million transitions (about 400 MB).
    code = ("import mymyr; from mymyr import search\n"
            f"d = mymyr.Domain.from_string({FAN!r})\n"
            f"t = mymyr.Task(d.instantiate_string({fan_problem(20)!r}))\n"
            "for f in (search.astar, search.gbfs, search.ehc):\n"
            "    r = f(t, heuristic='blind')\n"
            "    assert r.status == search.Status.SOLVED and r.stats.generated <= 3, (f, r.stats.generated)\n"
            "r = search.brfs(t)\n"
            "assert r.status == search.Status.SOLVED and r.stats.generated <= 3, r.stats.generated\n"
            "r = search.iw(t)\n"
            "assert r.status == search.Status.SOLVED and r.stats.generated <= 6, r.stats.generated\n"
            # the peak of this process (ru_maxrss would carry the parent's over exec)
            "print(next(int(l.split()[1]) for l in open('/proc/self/status') if l.startswith('VmHWM')) // 1024)\n")
    out = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout.split()[-1]) < 300  # MB: the interpreter and the module, no transition storage


def test_schema_views_and_written_actions():
    task = lights()
    press = task.formalism.schemas[0]
    assert press.arity == 0 and press.original_arity == 0 and press.written_arity == 2
    assert [(p.name, p.position, rep.name) for p, rep in press.collapsed_parameters] == [("s_0", 0, "s1"), ("r_0", 1, "r1")]
    heat = task.formalism.schemas[1]
    assert [p.position for p in heat.parameters] == [0]
    assert [p.position for p, _ in heat.collapsed_parameters] == [0]
    actions = task.applicable_actions(task.initial_state)
    assert [str(a) for a in actions] == ["(press s1 r1)", "(heat s1 r1)", "(heat s1 r2)", "(heat s1 r3)"]
    assert actions[0].binding == () and actions[2].objects == ["r2"]
    assert task.action("(heat s2 r2)") == actions[2]  # any switch: the same action
    with pytest.raises(ValueError, match="not of the type of parameter"):
        task.action("(heat r1 r2)")
    full = lights(False)
    assert len(full.applicable_actions(full.initial_state)) == 12
    assert full.formalism.schemas[0].collapsed_parameters == []


def test_plans_round_trip_and_refuse_the_wrong_type():
    task = lights()
    plan = search.parse_plan(task, "(press s2 r3)\n(heat s2 r3)\n")
    assert search.format_plan(task, plan).splitlines()[:2] == ["(press s1 r1)", "(heat s1 r3)"]
    assert search.validate_plan(task, "(press s1 r2)\n(heat s2 r3)\n")
    assert search.validate_plan(task, plan)
    bad = search.validate_plan(task, "(press r1 r1)\n(heat s1 r3)\n")
    assert bad.error == search.PlanError.WRONG_TYPE and bad.condition == "r1 - switch"
    with pytest.raises(ValueError, match="r1 - switch is not of its parameter's type"):
        search.parse_plan(task, "(press r1 r1)\n")


def test_empty_type_has_no_action():
    domain = mymyr.Domain.from_string(LIGHTS)
    problem = "(define (problem p) (:domain lights) (:objects r1 - room) (:init (on)) (:goal (lit)))"
    for collapse in (True, False):
        task = mymyr.Task(domain.instantiate_string(problem, collapse_unused_parameters=collapse))
        assert task.applicable_actions(task.initial_state) == []
        for heuristic in ("blind", "ff"):
            r = search.astar(task, heuristic=heuristic)
            assert r.status in (search.Status.EXHAUSTED, search.Status.UNSOLVABLE) and r.stats.generated == 0
    with pytest.raises(ValueError):
        search.parse_plan(mymyr.Task(domain.instantiate_string(problem)), "(press r1 r1)\n")


def test_rl_action_sets_shrink_consistently():
    for n, collapse, actions in ((3, True, 1), (3, False, 3 ** 5)):
        task = fan(n, collapse)
        table = rl.as_table(task)
        x = rl.expand(table, [task.initial_state])
        assert int(x.offsets[1]) == actions
        assert np.asarray(x.binding).shape[1] == (0 if collapse else 5)
        assert rl.applicable_schemas(task, [task.initial_state]).tolist() == [[True, False]]
        assert rl.is_applicable(task, [task.initial_state], 0, np.zeros((1, 0 if collapse else 5), np.int32)).tolist() == [True]
        pool = rl.CpuEnvPool(task, num_envs=2, seed=0, max_steps=4)
        pool.reset()
        step = pool.step()
        assert not step.terminated.any()


def test_the_setting_is_part_of_the_task_identity(tmp_path):
    task, full = lights(), lights(False)
    assert task.fingerprint != full.fingerprint
    for i, t in enumerate((task, full)):
        again = pickle.loads(pickle.dumps(t))
        assert again.fingerprint == t.fingerprint
        assert again.formalism.schemas[0].arity == t.formalism.schemas[0].arity
        path = tmp_path / f"t{i}.txt"
        path.write_text(t.to_text())
        read = mymyr.Task.from_text(str(path))
        assert read.to_text() == t.to_text()
        assert len(read.applicable_actions(read.initial_state)) == len(t.applicable_actions(t.initial_state))
    no_unused = "(define (domain b) (:predicates (p ?x)) (:action a :parameters (?x) :precondition (p ?x) :effect (not (p ?x))))"
    problem = "(define (problem q) (:domain b) (:objects o) (:init (p o)) (:goal (and)))"
    d = mymyr.Domain.from_string(no_unused)
    assert (mymyr.Task(d.instantiate_string(problem)).fingerprint
            != mymyr.Task(d.instantiate_string(problem, collapse_unused_parameters=False)).fingerprint)


def test_state_space_counts():
    from mymyr import datasets
    collapsed = datasets.state_space(fan(3), remove_if_unsolvable=False)
    full = datasets.state_space(fan(3, False), remove_if_unsolvable=False)
    assert collapsed.num_states == full.num_states == 3
    # x in every state, y in the two with (h)
    assert collapsed.num_transitions == 5 and full.num_transitions == 3 * 3 ** 5 + 2
