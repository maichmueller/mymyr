"""Parity: the flat relation encoder's goal relations against the recorded reference output
(tests/parity/data/encoding/mifrost_flat_goals.json)."""

import json

import pytest

import mymyr
from mymyr import encoding

from parity_support import DATA

MODES = ["auto", "reachable", "dense"]


def unpack(batch, graph):
    output = {}
    relations = len(batch.relation_names)
    for relation, name in enumerate(batch.relation_names):
        arity = int(batch.relation_arities[relation])
        cell = graph * relations + relation
        begin, end = batch.relation_offsets[cell : cell + 2]
        arguments = batch.relation_args[begin:end] - batch.object_offsets[graph]
        count = int(batch.relation_counts[graph, relation])
        assert end - begin == count * arity
        assert batch.relation_tuple_offsets[cell + 1] - batch.relation_tuple_offsets[cell] == count
        output[name] = [tuple(row) for row in arguments.reshape(count, arity)]
    return output


DOMAIN = """(define (domain features)
 (:requirements :strips :negative-preconditions :derived-predicates)
 (:predicates (fixed ?x) (flag ?x) (available ?x) (ready))
 (:derived (available ?x) (flag ?x))
 (:action enable :parameters (?x) :precondition (not (flag ?x)) :effect (flag ?x))
 (:action disable :parameters (?x) :precondition (flag ?x) :effect (not (flag ?x)))
 (:action switch :parameters () :precondition (and) :effect (ready)))"""


def feature_task(objects="a b", atoms="auto"):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string(DOMAIN)
    problem = f"""(define (problem p) (:domain features) (:objects {objects})
      (:init (fixed a) (flag a))
      (:goal (and (flag b) (not (flag a)) (fixed a) (not (fixed b))
                  (available b) (not (available a)) (ready))))"""
    return mymyr.Task(domain.instantiate_string(problem), atoms=atoms)


@pytest.mark.parametrize("atoms", MODES)
def test_flat_goal_relations_match_the_reference(atoms):
    fixture = json.loads((DATA / "encoding/mifrost_flat_goals.json").read_text())
    task = feature_task(atoms=atoms)
    goal = task.ground_condition(fixture["goals"])
    batch = encoding.FlatRelationEncoder(task).encode(task.initial_state, goals=goal)
    got = unpack(batch, 0)
    for name, expected in fixture["relations"].items():
        assert [list(values) for values in got[name]] == expected
    assert not got["[+]flag[g][sat]"] and not got["[-]flag[g][sat]"]
