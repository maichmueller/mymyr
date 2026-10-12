"""The mifrost parity fixtures replay in mymyr: tasks load, action paths reproduce the recorded states, and the names
and arities that the fixtures record exist in the task. Uses only mymyr and the JSON files (see
scripts/export_mifrost_encoding_fixture.py for the format). The structural checks need only the JSON files; the others
need the PDDL front end and skip without it."""

import pytest

from encoding_parity import (
    FIXTURES, HELPER, ROOT, atoms, base_predicate, derived_in_fixture_order, flat_catalogs, hgraph_catalogs, load, read,
    replay, state_path,
)


@pytest.fixture(params=FIXTURES, ids=[p.stem[:3] for p in FIXTURES])
def recorded(request):
    """The fixture alone."""
    return read(request.param)


@pytest.fixture(params=FIXTURES, ids=[p.stem[:3] for p in FIXTURES])
def case(request):
    """The fixture and the task it names, loaded in mymyr."""
    return load(request.param)


def test_fixtures_are_present():
    assert [p.stem[:3] for p in FIXTURES] == [f"f{i:02d}" for i in range(1, 14)]


def test_provenance_and_structure(recorded):
    fixture = recorded
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


def test_paired_tasks_record_mifrost_on_the_problem_as_written(recorded):
    task = recorded["task"]
    if "mifrost_problem" not in task:
        assert "mifrost_on_problem" not in task
        return
    assert task["mifrost_problem"] != task["problem"]
    assert task["mifrost_on_problem"]
    assert all(outcome != "ok" for outcome in task["mifrost_on_problem"].values())


def test_flat_catalogs_are_sorted_bytewise(recorded):
    for config, catalog in flat_catalogs(recorded):
        if config["encoder"] != "FlatRelationEncoder":
            continue
        names = catalog["names"]
        assert names == sorted(names, key=lambda n: n.encode())
        assert len(set(names)) == len(names)
        assert len(catalog["arities"]) == len(catalog["sources"]) == len(catalog["slots"]) == len(names)
        assert all(len(slots) == arity for slots, arity in zip(catalog["slots"], catalog["arities"]))


def test_hgraph_catalogs_are_well_formed(recorded):
    for _, catalog in hgraph_catalogs(recorded):
        node_types = catalog["node_types"]
        assert node_types == sorted(node_types, key=lambda n: n.encode()) and len(set(node_types)) == len(node_types)
        assert "_symbol_" in node_types
        for edge in catalog["edge_types"]:
            assert edge["rel"].isdigit()
            assert {edge["src"], edge["dst"]} <= set(node_types)
            assert "_symbol_" in (edge["src"], edge["dst"]) and edge["src"] != edge["dst"]


def test_dag_and_successor_paths_agree_with_the_state(recorded):
    for record in recorded["encodings"]:
        path = state_path(recorded, record["state"])
        inputs = record["inputs"]
        if "dag" in inputs:
            dag = inputs["dag"]
            assert dag["nodes"][0] == path
            for parent, child, action in dag["edges"]:
                assert dag["nodes"][child] == dag["nodes"][parent] + [action]
        if "successor" in inputs:
            successor = inputs["successor"]
            assert successor["path"][:-1] == path
            assert successor["path"][-1] == successor["action"]


def test_input_literals_are_clean(recorded):
    for record in recorded["encodings"]:
        inputs = record["inputs"]
        groups = list(inputs.get("subgoal_layers", [])) + [step for _, step in inputs.get("history_subgoals", [])]
        groups += [inputs["goals"]] if "goals" in inputs else []
        for literals in groups:
            assert len(set(literals)) == len(literals), (record["config"], record["state"])
            positive = {l for l in literals if not l.startswith("(not ")}
            negative = {l[5:-1] for l in literals if l.startswith("(not ")}
            assert not positive & negative, (record["config"], record["state"])


def test_states_replay_by_action_name(case):
    fixture, task = case
    for record in fixture["states"]:
        state = replay(task, record["path"])
        assert atoms(state.atoms()) == record["fluent"], record["id"]
        assert derived_in_fixture_order(task, fixture, state) == record["derived"], record["id"]


def test_helper_atoms_exist_only_where_the_fixture_has_them(case):
    fixture, task = case
    for record in fixture["states"]:
        state = replay(task, record["path"])
        extra = [a for a in atoms(state.derived_atoms())
                 if HELPER.match(a.strip("()").split()[0]) and a.strip("()").split()[0]
                 not in fixture["task"]["helper_predicates"]]
        assert not extra or "mifrost_problem" in fixture["task"], extra


def test_objects_and_static_atoms(case):
    fixture, task = case
    assert sorted(task.objects) == fixture["task"]["objects"]
    assert atoms(task.formalism.static_init) == fixture["static_atoms"]


def test_catalog_names_exist_in_the_task(case):
    fixture, task = case
    predicates = {p.name for p in task.formalism.predicates}
    schemas = {s.name for s in task.formalism.schemas}
    for _, catalog in flat_catalogs(fixture):
        for name, source in zip(catalog["names"], catalog["sources"]):
            if name.startswith("_"):
                continue
            base = base_predicate(name)
            if source == "action":
                assert base in schemas, name
            else:
                assert base in predicates, name


def test_logical_arities_match_the_task(case):
    fixture, task = case
    predicates = {p.name: p.arity for p in task.formalism.predicates}
    schemas = {s.name: s.arity for s in task.formalism.schemas}
    for _, catalog in flat_catalogs(fixture):
        for name, source, arity in zip(catalog["names"], catalog["sources"], catalog["logical_arities"]):
            if name.startswith("_"):
                continue
            base = base_predicate(name)
            expected = schemas[base] if source == "action" else predicates[base]
            assert arity == expected, name


def test_action_arity_counts_existential_witnesses():
    path = ROOT / "tests/data/encoding/mifrost/f09_relay.json"
    fixture, task = load(path)
    schema = next(s for s in task.formalism.schemas if s.name == "step")
    assert schema.original_arity == 2 and schema.arity == 3
    for _, catalog in flat_catalogs(fixture):
        if "step" in catalog["names"]:
            assert catalog["logical_arities"][catalog["names"].index("step")] == 3
    names = [name for record in fixture["encodings"] for name in record["inputs"].get("actions", [])]
    assert any(name.startswith("(step ") and len(name.strip("()").split()) == 4 for name in names)


def test_hgraph_catalogs_match_the_task(case):
    """Edge positions of a predicate node are its argument positions; an action node has the target entity at
    position 0 and the action's objects (witnesses included) after it."""
    fixture, task = case
    positions = {p.name: max(p.arity, 1) for p in task.formalism.predicates}
    for schema in task.formalism.schemas:
        positions[schema.name] = max(positions.get(schema.name, 0), schema.arity + 1)
    for _, catalog in hgraph_catalogs(fixture):
        for node_type in catalog["node_types"]:
            if node_type != "_symbol_":
                assert base_predicate(node_type) in positions, node_type
        for edge in catalog["edge_types"]:
            node_type = edge["dst"] if edge["src"] == "_symbol_" else edge["src"]
            if node_type != "_symbol_":
                assert int(edge["rel"]) < positions[base_predicate(node_type)], edge


def test_input_literals_and_actions_exist(case):
    fixture, task = case
    for record in fixture["encodings"]:
        inputs = record["inputs"]
        path = state_path(fixture, record["state"])
        state = replay(task, path)
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
            for node in inputs["dag"]["nodes"]:
                replay(task, node)
        if "successor" in inputs:
            replay(task, inputs["successor"]["path"])
