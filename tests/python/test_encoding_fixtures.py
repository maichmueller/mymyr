"""The mifrost parity fixtures replay in mymyr: tasks load, action paths reproduce the recorded states, and the names
that the fixtures record exist in the task. Uses only mymyr and the JSON files (see
scripts/export_mifrost_encoding_fixture.py for the format)."""

import json
import re
from pathlib import Path

import pytest

import mymyr

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted((ROOT / "tests/data/encoding/mifrost").glob("f*.json"))
HELPER = re.compile(r"axiom_\d+$")
DECORATION = re.compile(r"^(?:\[[+-]\])?(.+?)(?:\[[a-z+\-]+\])*$")
GOAL_SOURCES = ("goal", "goal_derivation", "history")


def load(path):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    fixture = json.loads(path.read_text())
    task = mymyr.Task.from_pddl(ROOT / fixture["task"]["domain"], ROOT / fixture["task"]["problem"])
    return fixture, task


def replay(task, path):
    state = task.initial_state
    for name in path:
        state = task.apply(state, name)
    return state


def atoms(values):
    return sorted(str(a) for a in values)


def comparable_derived(names):
    """Derived atoms by name; the argument order of the normalizer's helper predicates (``axiom_<k>``) differs between
    mimir and mymyr, so those compare as multisets of sorted arguments."""
    output = []
    for name in names:
        predicate, *arguments = name.strip("()").split()
        output.append((predicate, tuple(sorted(arguments)) if HELPER.match(predicate) else tuple(arguments)))
    return sorted(output)


def base_predicate(relation):
    """The predicate or schema a catalog relation is named after (decoration such as ``[+]`` and ``[g][sat]`` removed)."""
    return DECORATION.match(relation).group(1)


@pytest.fixture(params=FIXTURES, ids=[p.stem[:3] for p in FIXTURES])
def case(request):
    return load(request.param)


def test_fixtures_are_present():
    assert [p.stem[:3] for p in FIXTURES] == [f"f{i:02d}" for i in range(1, 11)]


def test_provenance_and_structure(case):
    fixture, _ = case
    assert fixture["format"] == 1
    assert fixture["provenance"]["mifrost_version"] == "0.6.0"
    assert fixture["provenance"]["pymimir_version"] == "0.16.3"
    assert fixture["states"][0] == dict(id="s0", path=[], fluent=fixture["states"][0]["fluent"],
                                        derived=fixture["states"][0]["derived"])
    assert len(fixture["states"]) == 5
    assert len({tuple(s["fluent"]) + ("|",) + tuple(s["derived"]) for s in fixture["states"]}) >= 3
    pool = fixture["tuples"]
    state_ids = {s["id"] for s in fixture["states"]}
    for record in fixture["encodings"]:
        assert record["config"] in fixture["configs"] and record["state"] in state_ids
        outputs = record["outputs"]
        assert "error" not in outputs, outputs.get("error")
        for relation in outputs.get("relations", {}).values():
            assert relation["emit"] in pool and relation.get("sorted", relation["emit"]) in pool
            assert sorted(pool[relation["emit"]]) == pool[relation.get("sorted", relation["emit"])]
        if "entities" in outputs:
            total = sum(count for _, count in outputs["entities"]["runs"])
            assert total == sum(len(v) for v in outputs["entities"]["by_role"].values())
    for config in fixture["configs"].values():
        assert config["catalog"] in fixture["catalogs"]


def test_flat_catalogs_are_sorted_bytewise(case):
    fixture, _ = case
    for config in fixture["configs"].values():
        catalog = fixture["catalogs"][config["catalog"]]
        if "names" in catalog and config["encoder"] == "FlatRelationEncoder":
            names = catalog["names"]
            assert names == sorted(names, key=lambda n: n.encode())
            assert len(set(names)) == len(names)
            assert len(catalog["arities"]) == len(catalog["sources"]) == len(catalog["slots"]) == len(names)
            assert all(len(slots) == arity for slots, arity in zip(catalog["slots"], catalog["arities"]))


def test_states_replay_by_action_name(case):
    fixture, task = case
    for record in fixture["states"]:
        state = replay(task, record["path"])
        assert atoms(state.atoms()) == record["fluent"], record["id"]
        assert comparable_derived(atoms(state.derived_atoms())) == comparable_derived(record["derived"]), record["id"]


def test_objects_and_static_atoms(case):
    fixture, task = case
    assert sorted(task.objects) == fixture["task"]["objects"]
    assert atoms(task.formalism.static_init) == fixture["static_atoms"]


def test_catalog_names_exist_in_the_task(case):
    fixture, task = case
    predicates = {p.name for p in task.formalism.predicates}
    schemas = {s.name for s in task.formalism.schemas}
    for config in fixture["configs"].values():
        catalog = fixture["catalogs"][config["catalog"]]
        for name, source in zip(catalog.get("names", []), catalog.get("sources", [])):
            if name.startswith("_"):
                continue
            base = base_predicate(name)
            if source == "action":
                assert base in schemas, name
            else:
                assert base in predicates, name


def test_input_literals_and_actions_exist(case):
    fixture, task = case
    for record in fixture["encodings"]:
        inputs = record["inputs"]
        state = replay(task, next(s for s in fixture["states"] if s["id"] == record["state"])["path"])
        literals = list(inputs.get("goals", []))
        for layer in inputs.get("subgoal_layers", []):
            literals += layer
        for _, step in inputs.get("history_subgoals", []):
            literals += step
        if literals:
            task.ground_condition(literals)
        if "actions" in inputs:
            applicable = {str(a) for a in state.applicable_actions()}
            assert set(inputs["actions"]) <= applicable
        if "dag" in inputs:
            dag = inputs["dag"]
            for parent, child, action in dag["edges"]:
                assert dag["nodes"][child] == dag["nodes"][parent] + [action]
            for path in dag["nodes"]:
                replay(task, path)
        if "successor" in inputs:
            successor = inputs["successor"]
            replay(task, successor["path"])
            assert successor["path"][-1] == successor["action"]
