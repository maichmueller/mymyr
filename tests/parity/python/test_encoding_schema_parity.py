"""Parity: the relation catalogs of EncodingSchema against the catalogs recorded from the reference flat and horizon
encoders (tests/parity/data/encoding/schema/catalogs.json); compared name for name, arity for arity and slot role for
slot role, with every difference listed."""

import json
import re

import pytest

import mymyr
from mymyr.encoding import EncodingSchema

from parity_support import DATA as PARITY_DATA, ROOT

DATA = ROOT / "tests/data"
CATALOGS = json.loads((PARITY_DATA / "encoding/schema/catalogs.json").read_text())
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


def native_options(entry):
    """The EncodingSchema keywords for a mifrost configuration of the catalog file."""
    kw = entry["kwargs"]
    out = {"equality": True}  # mifrost keeps the equality relations; native leaves them out by default
    if entry["kind"] == "flat":
        for key in ("goal_derivations", "max_goal_level", "target_sources", "include_static"):
            if key in kw:
                out[key] = kw[key]
        out["nullary"] = not kw.get("ignore_zero_arity_relations", True)
        out["predicate_virtual_nodes"] = bool(kw.get("use_predicate_virtual_nodes", False))
    else:
        out["horizon_mode"] = kw["transition_mode"]
        out["root_policy"] = kw["root_policy"]
        out["include_actions"] = not kw["ignore_actions"]
        # mifrost lists the nullary relations in every horizon catalog, whatever ignore_zero_arity_relations says.
        out["nullary"] = True
        out["parent_relation"] = bool(kw.get("enable_parent_relation", False))
        if "goal_derivations" in kw:
            out["goal_derivations"] = [name.split(".")[-1] for name in kw["goal_derivations"]]
        out["max_goal_level"] = kw.get("max_goal_level", 0)
        out["predicate_virtual_nodes"] = bool(kw.get("use_predicate_virtual_nodes", False))
    return out


def mifrost_catalog(entry):
    """(name, arity, slot roles) per relation, in mifrost's order."""
    offsets = entry["slot_role_offsets"]
    return [
        (name, arity, entry["slot_roles"][offsets[i] : offsets[i + 1]])
        for i, (name, arity) in enumerate(zip(entry["names"], entry["arities"]))
    ]


AUXILIARY_SLOT = 8


def native_catalog(schema):
    """(name, arity, slot roles) per relation, with the auxiliary slot of the nullary relations taken away: mifrost
    encodes a nullary atom as a tuple of its other slots only."""
    rows = []
    for name, arity, roles in zip(schema.relation_names, schema.relation_arities, schema.slot_roles):
        if roles and roles[-1] == AUXILIARY_SLOT:
            arity, roles = arity - 1, roles[:-1]
        rows.append((name, arity, roles))
    return rows


# mifrost source labels: 'state' covers the fluent, static and derived lanes.
SOURCE_OF_LANE = {
    "fluent": "state",
    "static": "state",
    "derived": "state",
    "goal": "goal",
    "goal_derivation": "goal_derivation",
    "goal_satisfaction": "goal_satisfaction",
    "history": "history",
    "action": "action",
    "parent": "parent",
}

# Entries compared with an explicitly listed difference.
SUPPORT_LITERALS = "flat_literals_relmo"  # mifrost's support_literals option is not ported
NO_STATIC = "flat_no_static"  # include_static=False drops the static relations from the catalog


@needs_pddl
@pytest.mark.parametrize("name", [n for n in CATALOGS if n not in (SUPPORT_LITERALS, NO_STATIC)])
def test_catalog_equals_mifrost(tasks, name):
    entry = CATALOGS[name]
    schema = EncodingSchema(tasks[entry["domain"]], **native_options(entry))
    assert native_catalog(schema) == mifrost_catalog(entry)
    nullary = [i for i, n in enumerate(schema.relation_logical_arities) if n == 0 and schema.relation_predicates[i] >= 0]
    assert bool(nullary) == native_options(entry)["nullary"]
    assert all(schema.slot_roles[i][-1] == AUXILIARY_SLOT for i in nullary)
    assert [SOURCE_OF_LANE[lane] for lane in schema.relation_lanes] == entry["sources"]
    assert schema.slot_role_names[:8] == [
        "argument_slot", "predicate_slot", "state_slot", "action_slot",
        "goal_target_slot", "subgoal_target_slot", "history_target_slot", "history_slot",
    ]  # fmt: skip
    assert len(schema.slot_role_names) == len(schema.entity_role_names) == 9


@needs_pddl
def test_support_literals_are_the_only_difference_to_the_relmo_catalog(tasks):
    entry = CATALOGS[SUPPORT_LITERALS]
    schema = EncodingSchema(tasks["blocks"], **native_options(entry))
    # support_literals adds `[+]p`, `[+]p[sat]` and `[+]p[unsat]` (and the `-` forms) to the goal lanes.
    literal = re.compile(r"^\[[+-]\][^\[]+(\[(sat|unsat)\])?$")
    dropped = [row for row in mifrost_catalog(entry) if literal.match(row[0])]
    assert len(dropped) == 2 * 3 * 5  # two signs, three names, five predicates (the types have no goal relations)
    assert native_catalog(schema) == [row for row in mifrost_catalog(entry) if row not in dropped]
    assert native_catalog(schema) == mifrost_catalog(CATALOGS["flat_relmo"])


@needs_pddl
def test_include_static_false_drops_the_static_relations(tasks):
    # mifrost keeps the (empty) static relations in the catalog; the native catalog leaves them out.
    entry = CATALOGS[NO_STATIC]
    schema = EncodingSchema(tasks["blocks"], **native_options(entry))
    expected = [row for row in mifrost_catalog(entry) if row[0] not in ("number", "object")]
    assert native_catalog(schema) == expected
    assert "object" not in schema.relation_names and "[+]object[hist]" in schema.relation_names
