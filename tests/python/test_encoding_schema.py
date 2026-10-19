"""EncodingSchema: the domain-scoped relation catalog of the relational encodings.

Catalogs, fingerprints, option handling and the helper predicates of quantified conditions.
"""

import os
import re

import pytest

import mymyr
from mymyr import rl
from mymyr.encoding import EncodingSchema

from conftest import ROOT, text_task
from python_children import run_code

DATA = ROOT / "tests/data"
PDDL = hasattr(mymyr, "Domain")
needs_pddl = pytest.mark.skipif(not PDDL, reason="built without the loki front end")

DOMAINS = {
    "blocks": (DATA / "pddl/blocks/domain.pddl", DATA / "pddl/blocks/probBLOCKS-8-0.pddl"),
    "childsnack": (DATA / "ipc/childsnack-ipc/test/domain.pddl", DATA / "ipc/childsnack-ipc/test/p01-easy.pddl"),
    "philosophers": (DATA / "pddl/philosophers/domain.pddl", DATA / "pddl/philosophers/p03-phil4.pddl"),
}


@pytest.fixture(scope="module")
def tasks():
    if not PDDL:
        pytest.skip("built without the loki front end")
    return {name: mymyr.Task.from_pddl(domain, problem) for name, (domain, problem) in DOMAINS.items()}


@needs_pddl
def test_include_static_false_drops_the_state_relation_of_every_static_predicate(tasks):
    default = EncodingSchema(tasks["childsnack"])
    schema = EncodingSchema(tasks["childsnack"], include_static=False)
    static = {n for n, lane in zip(default.relation_names, default.relation_lanes) if lane == "static"}
    assert len(static) == 13 and {"child", "waiting", "no_gluten_bread"} <= static
    assert set(default.relation_names) - set(schema.relation_names) == static
    assert not static & set(schema.relation_names)
    assert "static" not in schema.relation_lanes
    # their goal and history relations stay
    assert {"[+]child[hist]", "[-]waiting[hist]"} <= set(schema.relation_names)
    assert schema.schema_fingerprint != default.schema_fingerprint


@needs_pddl
def test_delta_horizon_has_no_relations_for_static_predicates(tasks):
    # the reference lists delta relations for static predicates too; they could never hold a tuple.
    schema = EncodingSchema(
        tasks["childsnack"], horizon_mode="delta", root_policy="exclude",
        goal_derivations=["added_satisfied", "added_unsatisfied"],
    )  # fmt: skip
    names = set(schema.relation_names)
    assert "[+]at[g][sat+]" in names and "[+]at" in names
    assert not [n for n in names if re.match(r"^\[[+-]\](child|waiting|place)\b.*(sat[+-]\]$)", n)]
    assert "[+]child" not in names and "child" in names


@needs_pddl
def test_helper_predicates_of_quantified_conditions_have_relations(tasks):
    schema = EncodingSchema(tasks["philosophers"])
    for name in ("axiom_0", "axiom_1", "[+]axiom_1[g]"):
        assert name in schema.relation_names
    lanes = dict(zip(schema.relation_names, schema.relation_lanes))
    assert lanes["axiom_0"] == lanes["axiom_1"] == lanes["blocked"] == "derived"
    assert dict(zip(schema.relation_names, schema.relation_arities))["axiom_1"] == 3


@needs_pddl
def test_helper_predicates_are_the_same_in_every_problem_of_a_domain():
    domain = DATA / "pddl/philosophers/domain.pddl"
    problems = [DATA / "pddl/philosophers/p03-phil4.pddl", DATA / "pddl/philosophers/p03-phil4-one-blocked.pddl"]
    schemas = [EncodingSchema(mymyr.Task.from_pddl(domain, p)) for p in problems]
    assert schemas[0].relation_names == schemas[1].relation_names
    assert schemas[0].schema_fingerprint == schemas[1].schema_fingerprint
    both = EncodingSchema(rl.TaskTable([mymyr.Task.from_pddl(domain, p) for p in problems]))
    assert both.schema_fingerprint == schemas[0].schema_fingerprint


@needs_pddl
def test_problem_local_helpers_of_a_forall_goal_are_not_part_of_the_schema():
    domain = DATA / "pddl/forall/domain.pddl"
    tasks = [mymyr.Task.from_pddl(domain, DATA / f"pddl/forall/{name}.pddl") for name in ("p01", "p02")]
    schemas = [EncodingSchema(task) for task in tasks]
    for task in tasks:
        table = rl.TaskTable([task])
        local = list(table.predicate_names)[table.num_domain_predicates :]
        assert local, "the forall goal compiles to a helper predicate of its problem"
    names = schemas[0].relation_names
    assert any(n.startswith("axiom_") for n in names)  # the domain's own helper, for the quantified precondition
    for task, schema in zip(tasks, schemas):
        table = rl.TaskTable([task])
        for local in list(table.predicate_names)[table.num_domain_predicates :]:
            assert not [n for n in schema.relation_names if local in n]
    assert schemas[0].relation_names == schemas[1].relation_names
    assert schemas[0].relation_slot_roles == schemas[1].relation_slot_roles
    assert schemas[0].schema_fingerprint == schemas[1].schema_fingerprint
    assert EncodingSchema(rl.TaskTable(tasks)).schema_fingerprint == schemas[0].schema_fingerprint


@needs_pddl
def test_helper_predicates_are_stable_with_quantified_actions_in_other_domains():
    small = DATA / "benchmarks/small/miconic-fulladl"
    domain = small / "domain.pddl"
    generated = DATA / "table_instances/miconic-simpleadl"
    problems = [small / "test_problem.pddl", generated / "simple-f3-p2-c.pddl",
                generated / "simple-f6-p4-q.pddl"]  # fmt: skip
    schemas = [EncodingSchema(mymyr.Task.from_pddl(domain, p)) for p in problems]
    assert any(n.startswith("axiom_") for n in schemas[0].relation_names)
    # the front end splits `stop` into one schema per disjunct and adds parameters for existential variables: one
    # relation per arity (1 to 3 parameters), holding all of them
    assert "stop" not in schemas[0].relation_names
    ids = [schemas[0].relation_id(f"stop#{k}") for k in (1, 2, 3)]
    assert [schemas[0].relation_arities[i] for i in ids] == [2, 3, 4]  # action slot + parameters
    assert all(schemas[0].schema_relations.count(i) > 1 for i in ids) and len(schemas[0].schema_relations) == 38
    assert [schemas[0].relation_schemas[i] for i in ids] == [6, 0, 1]
    for other in schemas[1:]:
        assert other.relation_names == schemas[0].relation_names
        assert other.schema_fingerprint == schemas[0].schema_fingerprint


def test_catalog_metadata_is_consistent():
    schema = EncodingSchema(text_task("philosophers__p03-phil4"), max_goal_level=1,
                            goal_derivations=["plain", "unsatisfied"], target_sources=["goal", "subgoal", "history"],
                            history=True, predicate_virtual_nodes=True)  # fmt: skip
    n = schema.num_relations
    columns = [
        schema.relation_names, schema.relation_arities, schema.relation_logical_arities, schema.relation_lanes,
        schema.relation_signs, schema.relation_levels, schema.relation_derivations, schema.relation_predicates,
        schema.relation_schemas, schema.relation_slot_roles,
    ]  # fmt: skip
    assert all(len(c) == n for c in columns)
    assert schema.relation_names == sorted(schema.relation_names, key=lambda s: s.encode())
    assert len(set(schema.relation_names)) == n
    for i, name in enumerate(schema.relation_names):
        assert schema.relation_id(name) == i
        roles = schema.relation_slot_roles[i]
        assert len(roles) == schema.relation_arities[i]
        assert roles[len(roles) - schema.relation_logical_arities[i] :] == [0] * schema.relation_logical_arities[i]
        assert (schema.relation_predicates[i] >= 0) != (schema.relation_schemas[i] >= 0)
        if schema.relation_lanes[i] == "action":
            assert schema.relation_signs[i] == 0 and roles[0] == 3
        if schema.relation_lanes[i] in ("goal", "history"):
            assert schema.relation_signs[i] == (1 if name.startswith("[+]") else -1)
    assert schema.relation_levels[schema.relation_id("[+]blocked[sg]")] == 1
    assert schema.relation_derivations[schema.relation_id("[-]blocked[sg][unsat]")] == "unsatisfied"
    assert schema.relation_derivations[schema.relation_id("[-]blocked[sg]")] is None
    # goal target slots, subgoal target slots, history target + history slots, and a predicate slot behind them
    assert schema.relation_slot_roles[schema.relation_id("[+]blocked[g]")][:2] == [4, 1]
    assert schema.relation_slot_roles[schema.relation_id("[+]blocked[sg]")][:2] == [5, 1]
    assert schema.relation_slot_roles[schema.relation_id("[+]blocked[hist]")][:3] == [6, 7, 1]
    assert schema.relation_slot_roles[schema.relation_id("[+]blocked[g][unsat]")][:1] == [1]
    assert schema.entity_role_names[:3] == ["object", "predicate_virtual", "goal"]
    assert schema.target_group_names == ["goal", "subgoal", "action", "history", "state"]
    with pytest.raises(KeyError, match="no relation"):
        schema.relation_id("nope")


def test_nullary_and_derived_predicates_follow_the_options():
    task = text_task("philosophers__p03-phil4")
    default = EncodingSchema(task)
    assert "axiom_0" in default.relation_names
    row = default.relation_id("axiom_0")
    assert default.relation_arities[row] == 1 and default.relation_logical_arities[row] == 0
    assert default.relation_slot_roles[row] == [8]  # the auxiliary slot: nullary atoms are tuples over the graph's one auxiliary entity
    assert default.slot_role_names[8] == "auxiliary_slot" and default.entity_role_names[8] == "auxiliary"
    assert default.relation_slot_roles[default.relation_id("[+]axiom_0[hist]")] == [7, 8]
    without_nullary = EncodingSchema(task, nullary=False)
    assert "axiom_0" not in without_nullary.relation_names and "[+]axiom_0[g]" not in without_nullary.relation_names
    assert without_nullary.entity_role_names == default.entity_role_names
    assert without_nullary.slot_role_names == default.slot_role_names
    assert 8 not in {role for roles in without_nullary.relation_slot_roles for role in roles}
    assert len(default.entity_role_names) == 9 and len(default.slot_role_names) == 9
    assert "=" not in default.relation_names and "[+]=[g]" not in default.relation_names
    with_equality = EncodingSchema(task, equality=True)
    assert with_equality.relation_arities[with_equality.relation_id("=")] == 2 and "[+]=[hist]" in with_equality.relation_names
    assert with_equality.schema_fingerprint != default.schema_fingerprint
    no_derived = EncodingSchema(task, include_derived=False)
    assert "blocked" not in no_derived.relation_names and "[+]blocked[g]" in no_derived.relation_names
    no_static = EncodingSchema(task, include_static=False)
    assert "queue" not in no_static.relation_names and "[+]queue[g]" in no_static.relation_names
    no_actions = EncodingSchema(task, include_actions=False)
    assert not [lane for lane in no_actions.relation_lanes if lane == "action"]


def test_options_round_trip_and_normalize():
    task = text_task("gripper__prob05")
    schema = EncodingSchema(task, goal_derivations=("satisfied", "plain", "satisfied"),
                            target_sources=["history", "goal"], history=True, history_max_steps=3)
    assert schema.options["goal_derivations"] == ("plain", "satisfied")
    assert schema.options["target_sources"] == ("goal", "history")
    assert schema.options["history"] is True and schema.options["history_max_steps"] == 3
    assert schema.options["horizon_mode"] is None and schema.options["root_policy"] is None
    again = EncodingSchema(task, **schema.options)
    assert again.schema_fingerprint == schema.schema_fingerprint and again.relation_names == schema.relation_names
    horizon = EncodingSchema(task, horizon_mode="full")
    assert horizon.options["root_policy"] == "include"
    assert EncodingSchema(rl.TaskTable([task]), horizon_mode="full").schema_fingerprint == horizon.schema_fingerprint


# ------------------------------------------------------------------------------------------------ fingerprint
PIN_DELTA, PIN_TARGETS, PIN_EXCLUDE = 0xF3C870FB1B1D5A8F, 0x2F5279FDFE082E1C, 0x069637BAFBF3C417
FINGERPRINT_VARIANTS = {
    "goal_derivations": {"goal_derivations": ["plain"]},
    "goal_derivations_unsat": {"goal_derivations": ["plain", "satisfied", "unsatisfied"]},
    "max_goal_level": {"max_goal_level": 1},
    "include_static": {"include_static": False},
    "include_derived": {"include_derived": False},
    "nullary": {"nullary": False},
    "equality": {"equality": True},
    "predicate_virtual_nodes": {"predicate_virtual_nodes": True},
    "include_actions": {"include_actions": False},
    "target_goal": {"target_sources": ["goal"]},
    "target_action": {"target_sources": ["action"]},
    "target_history": {"history": True, "target_sources": ["history"]},
    "history": {"history": True},
    "history_max_steps": {"history": True, "history_max_steps": 3},
    "target_subgoal": {"max_goal_level": 1, "target_sources": ["subgoal"]},
    "layout": {"layout": "relation"},
    "index_dtype": {"index_dtype": "int64"},
    "horizon_full": {"horizon_mode": "full"},
    "horizon_delta": {"horizon_mode": "delta", "goal_derivations": ["added_satisfied"]},
    "horizon_action": {"horizon_mode": "action"},
    "root_exclude": {"horizon_mode": "full", "root_policy": "exclude"},
    "root_encode_only": {"horizon_mode": "full", "root_policy": "encode_only"},
    "parent": {"horizon_mode": "full", "parent_relation": True},
}


def test_every_output_changing_option_changes_the_fingerprint():
    task = text_task("philosophers__p03-phil4")
    seen = {EncodingSchema(task).schema_fingerprint: "default"}
    for name, options in FINGERPRINT_VARIANTS.items():
        fingerprint = EncodingSchema(task, **options).schema_fingerprint
        assert fingerprint not in seen, f"{name} has the fingerprint of {seen.get(fingerprint)}"
        seen[fingerprint] = name


def test_fingerprint_ignores_the_order_of_options_and_their_spelling():
    task = text_task("philosophers__p03-phil4")
    first = EncodingSchema(task, goal_derivations=["plain", "unsatisfied", "satisfied"], target_sources=["goal", "action"],
                           max_goal_level=2, nullary=False)  # fmt: skip
    second = EncodingSchema(task, nullary=False, max_goal_level=2, target_sources=("action", "goal", "action"),
                            goal_derivations={"satisfied", "unsatisfied", "plain"})  # fmt: skip
    assert first.schema_fingerprint == second.schema_fingerprint
    # options that do not change the output keep the fingerprint
    assert EncodingSchema(task, root_policy=None).schema_fingerprint == EncodingSchema(task).schema_fingerprint
    blocks = text_task("blocks__probBLOCKS-8-0")
    assert (
        EncodingSchema(blocks, include_derived=False).schema_fingerprint == EncodingSchema(blocks).schema_fingerprint
    )  # blocks has no derived predicates


def test_fingerprint_values_are_pinned():
    # the fingerprints are part of the contract: the same on every platform, standard library and build
    philosophers = text_task("philosophers__p03-phil4")
    gripper = text_task("gripper__prob05")
    assert EncodingSchema(philosophers).schema_fingerprint == 0xA80E980B3FE26CF6
    delta = EncodingSchema(philosophers, horizon_mode="delta", goal_derivations=["added_satisfied"],
                           predicate_virtual_nodes=True)  # fmt: skip
    assert delta.schema_fingerprint == PIN_DELTA
    targets = EncodingSchema(gripper, max_goal_level=2, target_sources=["goal", "subgoal", "history", "action"],
                             history=True, layout="relation", index_dtype="int64")  # fmt: skip
    assert targets.schema_fingerprint == PIN_TARGETS
    exclude = EncodingSchema(gripper, horizon_mode="full", root_policy="exclude", parent_relation=True)
    assert exclude.schema_fingerprint == PIN_EXCLUDE


def test_fingerprint_is_stable_across_processes():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from conftest import text_task\n"
        "from mymyr.encoding import EncodingSchema\n"
        "t = text_task('philosophers__p03-phil4')\n"
        "print(EncodingSchema(t).schema_fingerprint, EncodingSchema(t, horizon_mode='delta', goal_derivations=['added_satisfied'],"
        " predicate_virtual_nodes=True, target_sources=[]).schema_fingerprint)\n" % str(ROOT / "tests/python")
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.add(run_code(code, env=env, capture_output=True).stdout)
    assert len(outputs) == 1
    task = text_task("philosophers__p03-phil4")
    assert outputs.pop().split()[0] == str(EncodingSchema(task).schema_fingerprint)


# ------------------------------------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"max_goal_level": 4}, ValueError, "^mymyr: encoding: max_goal_level must be in"),
        ({"max_goal_level": -1}, ValueError, "^mymyr: encoding: max_goal_level must be in"),
        ({"max_goal_level": 1.5}, TypeError, "int"),
        ({"target_sources": ["subgoal"]}, ValueError, "subgoal targets need max_goal_level"),
        ({"goal_derivations": ["added_satisfied"]}, ValueError, "delta horizon"),
        ({"goal_derivations": ["added_unsatisfied"], "horizon_mode": "full"}, ValueError, "delta horizon"),
        ({"horizon_mode": "full", "target_sources": ["goal"]}, ValueError, "horizon"),
        ({"horizon_mode": "action", "include_actions": False}, ValueError, "action"),
        ({"parent_relation": True}, ValueError, "parent_relation"),
        ({"root_policy": "exclude"}, ValueError, "root_policy 'exclude' needs a horizon mode"),
        ({"root_policy": "encode_only"}, ValueError, "root_policy 'encode_only' needs a horizon mode"),
        ({"target_sources": ["action"], "include_actions": False}, ValueError, "action targets need"),
        ({"target_sources": ["goal"], "goal_derivations": ["satisfied"]}, ValueError, "plain goal derivation"),
        ({"target_sources": ["subgoal"], "max_goal_level": 1, "goal_derivations": ["unsatisfied"]}, ValueError,
         "plain goal derivation"),
        ({"root_policy": "sideways", "horizon_mode": "full"}, ValueError, "root_policy"),
        ({"horizon_mode": "sideways"}, ValueError, "horizon_mode"),
        ({"layout": "row"}, ValueError, "layout"),
        ({"index_dtype": "int16"}, ValueError, "index_dtype"),
        ({"goal_derivations": ["plane"]}, ValueError, "goal_derivations"),
        ({"goal_derivations": "plain"}, TypeError, "collection"),
        ({"goal_derivations": [1]}, TypeError, "string"),
        ({"target_sources": ["states"]}, ValueError, "target_sources"),
        ({"layout": 1}, TypeError, "layout"),
        ({"no_such_option": 1}, TypeError, "no_such_option"),
    ],
)
def test_invalid_options_raise(options, error, match):
    with pytest.raises(error, match=match):
        EncodingSchema(text_task("gripper__prob05"), **options)


def test_root_policy_include_is_accepted_without_a_horizon():
    task = text_task("gripper__prob05")
    assert EncodingSchema(task, root_policy="include").schema_fingerprint == EncodingSchema(task).schema_fingerprint
    assert EncodingSchema(task, horizon_mode="full", root_policy="exclude").options["root_policy"] == "exclude"


def test_numeric_tasks_are_refused():
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    with pytest.raises(ValueError, match="^mymyr: encoding: numeric functions and constraints are not supported"):
        EncodingSchema(numeric)
    with pytest.raises(ValueError, match="numeric"):
        EncodingSchema(rl.TaskTable([numeric]))


def test_invalid_tables_raise():
    with pytest.raises(TypeError):
        EncodingSchema("gripper")
    with pytest.raises(TypeError):
        EncodingSchema(rl.TaskSuite([rl.TaskTable([text_task("gripper__prob05")])]))


@needs_pddl
def test_equality_relations_have_the_static_equality_atoms(tasks):
    # the table lists (= o o) among the static atoms of each instance, so the `=` relation has a tuple per object
    import numpy as np

    task = tasks["philosophers"]
    table = rl.TaskTable([task])
    metadata = table.atom_metadata()
    equality = table.predicate_names.index("=")
    atoms = np.asarray(metadata["static_args"])[np.asarray(metadata["static_pred"]) == equality]
    assert sorted(map(tuple, atoms[:, :2].tolist())) == [(o, o) for o in range(task.num_objects)]
    schema = EncodingSchema(task, equality=True)
    assert schema.relation_predicates[schema.relation_id("=")] == equality
    assert schema.relation_lanes[schema.relation_id("=")] == "static"
