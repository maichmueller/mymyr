"""mymyr.PddlError: PDDL that mymyr cannot read names the file, the line, the action and the construct."""

import pathlib
import pickle

import pytest

import mymyr

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

DOMAIN = """(define (domain d)
  (:requirements :strips :typing{req})
  (:types block)
  (:predicates (on ?x ?y - block) (clear ?x - block))
  {extra}
  (:action move
    :parameters (?x ?y - block)
    :precondition (and (clear ?x) (clear ?y){pre})
    :effect {effect}))
"""

PROBLEM = """(define (problem p) (:domain d)
  (:objects a b - block)
  (:init (clear a) (clear b){init})
  (:goal (and (on a b){goal})){pextra})
"""

EFFECT = "(and (on ?x ?y) (not (clear ?y)))"


def domain(req="", extra="", pre="", effect=EFFECT):
    return DOMAIN.format(req=req, extra=extra, pre=pre, effect=effect)


def problem(init="", goal="", pextra=""):
    return PROBLEM.format(init=init, goal=goal, pextra=pextra)


DURATIVE = """(define (domain d)
  (:requirements :strips :typing :durative-actions)
  (:types block)
  (:predicates (on ?x ?y - block) (clear ?x - block))
  (:durative-action move
    :parameters (?x ?y - block)
    :duration (= ?duration 1)
    :condition (and (at start (clear ?x)) (at start (clear ?y)))
    :effect (and (at end (on ?x ?y)) (at end (not (clear ?y))))))
"""

PDDL_PLUS = """(define (domain d)
  (:requirements :strips :typing :numeric-fluents :time)
  (:types block)
  (:predicates (on ?x ?y - block) (clear ?x - block))
  (:functions (level))
  ({kind} {name}
    :parameters ()
    :precondition (clear a)
    :effect (increase (level) 1))
  (:action move
    :parameters (?x ?y - block)
    :precondition (and (clear ?x) (clear ?y))
    :effect (and (on ?x ?y) (not (clear ?y)))))
"""

OBJECT_FLUENTS = """(define (domain d)
  (:requirements :strips :typing :object-fluents)
  (:types block)
  (:predicates (on ?x ?y - block) (clear ?x - block))
  (:functions (below ?x - block) - block)
  (:action move
    :parameters (?x ?y - block)
    :precondition (and (clear ?x) (clear ?y))
    :effect (and (on ?x ?y) (not (clear ?y)))))
"""

# name: (domain text, problem text, file with the error, line, action, words the message contains)
CASES = {
    "durative_actions": (DURATIVE, problem(), "domain", 5, None, ["durative actions", "not supported", "move"]),
    "durative_without_requirement": (
        DURATIVE.replace(" :durative-actions", ""), problem(), "domain", 5, None, ["durative actions"]),
    "processes": (PDDL_PLUS.format(kind=":process", name="fill"), problem(), "domain", 6, None,
                  ["processes", "not supported", ":process fill"]),
    "events": (PDDL_PLUS.format(kind=":event", name="overflow"), problem(), "domain", 6, None,
               ["events", "not supported", ":event overflow"]),
    "preferences_requirement": (domain(req=" :preferences"), problem(), "domain", 2, None,
                                ["requirement :preferences", "not supported"]),
    "preference_in_precondition": (domain(req=" :preferences", pre=" (preference p0 (clear ?x))"), problem(),
                                   "domain", 8, "move", ["preferences", "not supported"]),
    "preference_in_goal": (domain(), problem(goal=" (preference p1 (clear a))"), "problem", 4, None,
                           ["preferences", "not supported"]),
    "constraints_requirement": (domain(req=" :constraints"), problem(), "domain", 2, None,
                                ["requirement :constraints", "not supported"]),
    "constraints_in_domain": (domain(extra="(:constraints (always (clear a)))"), problem(), "domain", 5, None,
                              ["trajectory constraints", ":constraints"]),
    "constraints_in_problem": (domain(), problem(pextra="\n  (:constraints (always (clear a)))"), "problem", 5, None,
                               ["trajectory constraints", ":constraints"]),
    "object_fluents": (OBJECT_FLUENTS, problem(), "domain", 5, None, ["object fluents", "below", "block"]),
    "timed_initial_literals_requirement": (domain(req=" :timed-initial-literals"), problem(), "domain", 2, None,
                                           ["requirement :timed-initial-literals", "not supported"]),
    "timed_initial_literal": (domain(), problem(init=" (at 10 (clear a))"), "problem", 3, None,
                              ["timed initial literals", "not supported"]),
    "non_deterministic_effect": (domain(req=" :non-deterministic", effect="(oneof (on ?x ?y) (clear ?y))"),
                                 problem(), "domain", 9, "move", ["non-deterministic", "oneof"]),
    "probabilistic_effect": (domain(req=" :probabilistic-effects", effect="(probabilistic 0.5 (on ?x ?y))"),
                             problem(), "domain", 9, "move", ["probabilistic effects"]),
    "unknown_requirement": (domain(req=" :teleportation"), problem(), "domain", 2, None,
                            ["unknown requirement :teleportation"]),
    "undefined_predicate": (domain(pre=" (holding ?x)"), problem(), "domain", 8, "move",
                            ["undefined predicate 'holding'"]),
    "undefined_type": (domain().replace("(?x ?y - block)", "(?x ?y - brick)"), problem(), "domain", 7, "move",
                       ["undefined type 'brick'"]),
    "undefined_constant": (domain(pre=" (clear k)"), problem(), "domain", 8, "move", ["undefined object 'k'"]),
    "undefined_function": (domain(req=" :numeric-fluents", pre=" (> (fuel) 1)"), problem(), "domain", 8, "move",
                           ["undefined function 'fuel'"]),
    "undefined_object_in_init": (domain(), problem(init=" (clear c)"), "problem", 3, None, ["undefined object 'c'"]),
    "undefined_object_in_goal": (domain(), problem(goal=" (clear c)"), "problem", 4, None, ["undefined object 'c'"]),
    "wrong_arity_in_domain": (domain(pre=" (clear ?x ?y)"), problem(), "domain", 8, "move",
                              ["wrong number of arguments", "'clear'", "2 given, 1 expected"]),
    "wrong_arity_in_init": (domain(), problem(init=" (on a)"), "problem", 3, None,
                            ["wrong number of arguments", "'on'"]),
    "syntax_error": (domain().replace("(:action move", "(:action move((("), problem(), "domain", 6, "move",
                     ["syntax error"]),
    "unbalanced_parentheses": (domain(), problem()[:-3], "problem", 5, None, ["syntax error", "expected ')'"]),
    "negative_initial_literal": (domain(), problem(init=" (not (clear a))"), "problem", 3, None,
                                 ["negative literals in the initial state"]),
    "domain_name_mismatch": (domain(), problem().replace("(:domain d)", "(:domain other)"), "problem", 1, None,
                             ["domain 'other'", "domain 'd'"]),
}


def write(tmp_path, name, d, p):
    dp, pp = tmp_path / f"{name}_domain.pddl", tmp_path / f"{name}_problem.pddl"
    dp.write_text(d)
    pp.write_text(p)
    return dp, pp


@pytest.mark.parametrize("name", sorted(CASES))
def test_unsupported_or_malformed_pddl_names_the_construct(tmp_path, name):
    d, p, where, line, action, words = CASES[name]
    dp, pp = write(tmp_path, name, d, p)
    with pytest.raises(mymyr.PddlError) as info:
        mymyr.Task.from_pddl(dp, pp)
    e = info.value
    assert isinstance(e, ValueError)
    path = dp if where == "domain" else pp
    assert e.path == str(path)
    assert e.line == line
    assert e.action == action
    for w in words:
        assert w in e.message, (w, e.message)
    text = str(e)
    assert "\n" not in text
    assert text.startswith(f"{path}:{line}: {e.message}")
    if action:
        assert text.endswith(f"(in action {action})")


def test_errors_of_the_domain_object_and_strings(tmp_path):
    dp, pp = write(tmp_path, "x", domain(pre=" (holding ?x)"), problem())
    with pytest.raises(mymyr.PddlError, match="undefined predicate 'holding'"):
        mymyr.Domain.from_file(dp)
    with pytest.raises(mymyr.PddlError) as info:
        mymyr.Domain.from_string(domain(pre=" (holding ?x)"))
    assert info.value.path is None and info.value.line == 8
    assert str(info.value) == "line 8: undefined predicate 'holding' (in action move)"
    good = mymyr.Domain.from_string(domain())
    with pytest.raises(mymyr.PddlError, match="undefined object 'c'") as info:
        good.instantiate_string(problem(goal=" (clear c)"))
    assert info.value.line == 4
    dp, pp = write(tmp_path, "y", domain(), problem(init=" (at 10 (clear a))"))
    with pytest.raises(mymyr.PddlError, match="timed initial literals"):
        mymyr.Domain.from_file(dp).instantiate(pp)
    with pytest.raises(mymyr.PddlError, match="timed initial literals"):
        mymyr.Domain.from_file(dp).instantiate(pp, fast_init=False)


def test_missing_files_raise_file_not_found(tmp_path):
    dp, pp = write(tmp_path, "ok", domain(), problem())
    with pytest.raises(FileNotFoundError) as info:
        mymyr.Task.from_pddl(tmp_path / "nonexistent.pddl", pp)
    assert info.value.filename == str(tmp_path / "nonexistent.pddl")
    with pytest.raises(FileNotFoundError):
        mymyr.Task.from_pddl(dp, tmp_path / "nonexistent.pddl")
    with pytest.raises(FileNotFoundError):
        mymyr.Domain.from_file(dp).instantiate(tmp_path / "nonexistent.pddl")
    assert mymyr.Task.from_pddl(dp, pp).num_schemas == 1


def test_pddl_error_pickles_and_formats():
    e = mymyr.PddlError("undefined type 'brick'", "/x/domain.pddl", 7, "move")
    assert str(e) == "/x/domain.pddl:7: undefined type 'brick' (in action move)"
    f = pickle.loads(pickle.dumps(e))
    assert (f.message, f.path, f.line, f.action, str(f)) == (e.message, e.path, e.line, e.action, str(e))
    assert str(mymyr.PddlError("bad")) == "bad"
    assert str(mymyr.PddlError("bad", "p.pddl")) == "p.pddl: bad"


def test_supported_pddl_is_unaffected():
    blocks = pathlib.Path(__file__).resolve().parents[2] / "tests/data/pddl/blocks"
    task = mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / "probBLOCKS-8-0.pddl")
    assert task.num_schemas == 4
