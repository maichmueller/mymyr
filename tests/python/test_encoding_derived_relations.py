"""include_derived='domain': the helpers that normalization introduces for quantified and disjunctive conditions have
no relation at all (state, goal, subgoal, history, horizon), every other predicate keeps every relation the options ask
for, the derived predicates the domain declares included. The mark comes from the front end and survives the text
format; the batches of the remaining relations equal those of the full catalog."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, walk

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the PDDL front end", allow_module_level=True)

PDDL = ROOT / "tests" / "data" / "pddl"


def pddl_task(name, problem):
    return mymyr.Task.from_pddl(PDDL / name / "domain.pddl", PDDL / name / problem)


def philosophers():
    return pddl_task("philosophers", "p03-phil4.pddl")


def by_lane(schema, lane):
    return [n for n, l in zip(schema.relation_names, schema.relation_lanes) if l == lane]


def unpack(batch, graph):
    """{relation name: tuples of graph-local entity ids} of one graph, for the relations that have tuples."""
    out = {}
    names, arities, relations = batch.schema.relation_names, batch.schema.relation_arities, batch.num_relations
    base = int(batch.entity_offsets[graph])
    for relation, name in enumerate(names):
        count = int(batch.relation_counts[graph, relation])
        if count:
            cell = graph * relations + relation
            begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
            rows = (batch.relation_args[begin:end] - base).reshape(count, arities[relation])
            out[name] = [tuple(int(x) for x in row) for row in rows]
    return out


def test_domain_leaves_out_exactly_the_relations_of_the_compiled_helpers():
    task = philosophers()
    every = encoding.EncodingSchema(task)
    declared = encoding.EncodingSchema(task, include_derived="domain")
    compiled = {n for n, c in zip(every.relation_names, every.relation_compiled) if c}
    assert compiled == set(every.relation_names) - set(declared.relation_names)
    assert {n.strip("[+-]").split("[")[0] for n in compiled} == {"@domain-0", "@domain-1"}
    assert by_lane(every, "derived") == ["@domain-0", "@domain-1", "blocked", "blocked-trans"]
    assert by_lane(declared, "derived") == ["blocked", "blocked-trans"]
    assert not any(declared.relation_compiled)
    # the goal and history relations of the declared derived predicates stay
    assert {"[+]blocked[g]", "[+]blocked[hist]", "[-]blocked-trans[g][sat]"} <= set(declared.relation_names)
    # False drops the state relations of both kinds and keeps their goal and history relations
    none = encoding.EncodingSchema(task, include_derived=False)
    assert by_lane(none, "derived") == [] and "[+]@domain-0[g]" in none.relation_names


def test_a_domain_without_helpers_has_the_same_catalog():
    task = pddl_task("beacons", "p01.pddl")
    every = encoding.EncodingSchema(task)
    declared = encoding.EncodingSchema(task, include_derived="domain")
    assert by_lane(declared, "derived") == ["powered"] and not any(every.relation_compiled)
    assert declared.relation_names == every.relation_names
    assert declared.schema_fingerprint == every.schema_fingerprint


def test_a_domain_of_helpers_only_keeps_no_derived_relation():
    task = pddl_task("linking", "p01.pddl")
    every = encoding.EncodingSchema(task)
    declared = encoding.EncodingSchema(task, include_derived="domain")
    assert by_lane(every, "derived") == ["@domain-0", "@domain-1"]
    assert not [n for n in declared.relation_names if "@" in n]
    assert all(c == ("@domain-" in n) for n, c in zip(every.relation_names, every.relation_compiled))


def test_the_option_reads_back_and_rejects_other_values():
    task = philosophers()
    for value in (True, False, "domain"):
        assert encoding.FlatRelationEncoder(task, include_derived=value).options["include_derived"] == value
    for value in ("all", "none", 1):
        with pytest.raises(ValueError, match="include_derived must be True, False or 'domain'"):
            encoding.EncodingSchema(task, include_derived=value)


def test_batches_equal_the_full_catalog_on_the_kept_relations():
    task = philosophers()
    states = walk(task, steps=40, walks=6)
    full = encoding.FlatRelationEncoder(task).encode(states)
    kept = encoding.FlatRelationEncoder(task, include_derived="domain").encode(states)
    assert kept.num_graphs == full.num_graphs == len(states)
    # the objects come first in both; a graph whose only nullary tuples were helpers' loses the auxiliary entity
    assert (0 <= (full.entity_offsets - kept.entity_offsets)).all()
    names = set(kept.schema.relation_names)
    seen = {"blocked": 0, "@domain-0": 0}
    for graph in range(len(states)):
        a, b = unpack(full, graph), unpack(kept, graph)
        assert b == {n: t for n, t in a.items() if n in names}
        seen["blocked"] += len(b.get("blocked", []))
        seen["@domain-0"] += len(a.get("@domain-0", []))
    assert seen["blocked"] > 0 and seen["@domain-0"] > 0
    dropped = [i for i, c in enumerate(full.schema.relation_compiled) if c]
    assert int(kept.relation_counts.sum()) == int(full.relation_counts.sum()) - int(full.relation_counts[:, dropped].sum())


def test_the_mark_survives_the_text_format_and_pickling(tmp_path):
    task = philosophers()
    text = task.to_text()
    assert any(line.startswith("C ") and line.endswith(" @domain-0") for line in text.splitlines())
    path = tmp_path / "philosophers.txt"
    path.write_text(text)
    again = mymyr.Task.from_text(str(path))
    assert again.to_text() == text
    a = encoding.EncodingSchema(again, include_derived="domain")
    b = encoding.EncodingSchema(task, include_derived="domain")
    assert a.relation_names == b.relation_names and a.relation_compiled == b.relation_compiled
    encoder = encoding.FlatRelationEncoder(task, include_derived="domain")
    clone = pickle.loads(pickle.dumps(encoder))
    assert clone.relation_names == encoder.relation_names and clone.options["include_derived"] == "domain"


def test_a_table_refuses_instances_that_disagree_on_the_mark(tmp_path):
    task = philosophers()
    text = "\n".join(("D " + line[2:]) if line.startswith("C ") and len(line.split()) == 3 else line
                     for line in task.to_text().splitlines())
    path = tmp_path / "unmarked.txt"
    path.write_text(text + "\n")
    unmarked = mymyr.Task.from_text(str(path))
    assert not any(encoding.EncodingSchema(unmarked).relation_compiled)
    with pytest.raises(ValueError, match="derived, compiled"):
        rl.TaskTable([task, unmarked])


# option sets with the goal, subgoal and history relations on and off: (options, the relation names of a helper `h`
# besides its state relation `h`)
CONFIGS = {
    "goals and history": (dict(goal_derivations=("plain", "satisfied", "unsatisfied"), max_goal_level=1, history=True,
                               target_sources=("goal", "subgoal", "history")),
                          [f"[{s}]{{h}}{lv}{d}" for s in "+-" for lv in ("[g]", "[sg]") for d in ("", "[sat]", "[unsat]")]
                          + ["[+]{h}[hist]", "[-]{h}[hist]"]),
    "defaults": (dict(), ["[+]{h}[g]", "[+]{h}[g][sat]", "[-]{h}[g]", "[-]{h}[g][sat]", "[+]{h}[hist]", "[-]{h}[hist]"]),
    # the flat catalog lists the history relations whether or not the encoder fills them (history=False)
    "without goals": (dict(goal_derivations=()), ["[+]{h}[hist]", "[-]{h}[hist]"]),
    "horizon without goals": (dict(horizon_mode="full", goal_derivations=()), []),
    "delta horizon": (dict(horizon_mode="delta", goal_derivations=()), ["[+]{h}", "[-]{h}"]),
}


def relation_rows(schema):
    """{relation name: its description} over the schema's per-relation columns."""
    columns = (schema.relation_lanes, schema.relation_signs, schema.relation_levels, schema.relation_derivations,
               schema.relation_predicates, schema.relation_arities, schema.relation_slot_roles)
    return {name: tuple(c[r] for c in columns) for r, name in enumerate(schema.relation_names)}


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_domain_removes_every_relation_of_a_helper_and_nothing_else(config):
    """triage has one helper (@domain-0, a universal precondition): 'domain' removes its relations of every lane, those
    the options ask for, and keeps every other relation as True has it."""
    options, extras = CONFIGS[config]
    task = pddl_task("triage", "p01.pddl")
    every = encoding.EncodingSchema(task, **options)
    declared = encoding.EncodingSchema(task, include_derived="domain", **options)
    removed = {"@domain-0"} | {e.format(h="@domain-0") for e in extras}
    assert set(every.relation_names) - set(declared.relation_names) == removed
    assert set(declared.relation_names) <= set(every.relation_names)
    assert {n for n, c in zip(every.relation_names, every.relation_compiled) if c} == removed
    full, kept = relation_rows(every), relation_rows(declared)
    assert kept == {n: d for n, d in full.items() if n not in removed}
    # False drops the plain state relations of the derived predicates alone
    none = encoding.EncodingSchema(task, include_derived=False, **options)
    assert set(every.relation_names) - set(none.relation_names) == {"@domain-0"}


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_declared_derived_predicates_keep_every_relation(config):
    """beacons declares powered (:derived) and has no helper: its catalog under 'domain' is True's, the state, goal,
    subgoal and history relations of powered included."""
    options, extras = CONFIGS[config]
    task = pddl_task("beacons", "p01.pddl")
    every = encoding.EncodingSchema(task, **options)
    declared = encoding.EncodingSchema(task, include_derived="domain", **options)
    assert declared.relation_names == every.relation_names
    assert relation_rows(declared) == relation_rows(every)
    assert {"powered"} | {e.format(h="powered") for e in extras} <= set(declared.relation_names)


def test_declared_derived_goals_subgoals_and_history_are_encoded():
    """powered in the instance's goal, in per-row goals, subgoal layers and history steps fills the same tuples under
    'domain' as under True."""
    task = pddl_task("beacons", "p01.pddl")
    states = walk(task, steps=8, walks=2)
    options = CONFIGS["goals and history"][0]
    encoders = [encoding.FlatRelationEncoder(task, include_derived=v, **options) for v in (True, "domain")]
    for inputs in ({}, dict(goals=encoding.goal_masks(task, [["(powered n1)", "(not (powered n2))"]] * len(states)),
                            subgoals=encoding.subgoal_masks(task, [[["(powered n0)"]]] * len(states)),
                            history=encoding.history_masks(task, [[(-1, ["(powered n1)"]), (0, ["(lit n2)"])]] * len(states)))):
        a, b = (e.encode(states, **inputs) for e in encoders)
        assert np.array_equal(a.goal_complete, b.goal_complete)
        for graph in range(len(states)):
            assert unpack(a, graph) == unpack(b, graph)
        seen = {n for graph in range(len(states)) for n in unpack(b, graph)}
        want = {"[+]powered[g]"} | ({"[+]powered[sg]", "[-]powered[g]", "[+]powered[hist]"} if inputs else set())
        assert want <= seen, want - seen


def test_quantified_goals_become_predicates_of_the_problem():
    """A goal part that is not a literal becomes a derived predicate of the problem alone (an id past the domain's
    predicates), which no catalog lists: an instance's goal names no helper of the domain, and its goal tuples under
    'domain' equal True's."""
    tasks = [pddl_task("linking", p) for p in ("p01.pddl", "p02.pddl", "p03.pddl")]
    table = rl.TaskTable(tasks)
    names = encoding.EncodingSchema(table).relation_names
    for task, helper in zip(tasks, ("@problem-0", "@problem-1", "@problem-0")):
        assert helper in str(task.goal_condition)
        assert all(helper not in n for n in names)
        assert all("@" not in str(literal) for literal in task.encoding_goal)
        with pytest.raises(ValueError, match="of its problem alone"):
            encoding.goal_masks(task, [[f"({helper})"]])
    options = dict(goal_derivations=("plain", "satisfied", "unsatisfied"), target_sources=("goal",))
    full = encoding.FlatRelationEncoder(table, **options)
    kept = encoding.FlatRelationEncoder(table, include_derived="domain", **options)
    states = [s for task in tasks for s in walk(task, steps=10, walks=2)]
    ids = np.asarray([i for i, task in enumerate(tasks) for _ in walk(task, steps=10, walks=2)], np.int32)
    a, b = full.encode(states, ids), kept.encode(states, ids)
    assert np.array_equal(a.goal_complete, b.goal_complete)
    goal_lanes = {n for n, lane in zip(full.relation_names, full.relation_lanes) if lane in ("goal", "goal_derivation")}
    tuples = 0
    for graph in range(len(states)):
        x, y = unpack(a, graph), unpack(b, graph)
        assert {n: t for n, t in x.items() if n in goal_lanes} == {n: t for n, t in y.items() if n in goal_lanes}
        assert not [n for n in x if n in goal_lanes and "@" in n]
        tuples += sum(len(t) for n, t in y.items() if n in goal_lanes)
    assert tuples > 0


@pytest.mark.parametrize("name", ["triage", "philosophers"])
def test_capacity_bounds_hold_without_the_helpers(name):
    """The bounds under 'domain' are True's on the kept relations, and walks encoded with them stay within."""
    task = pddl_task("triage", "p01.pddl") if name == "triage" else philosophers()
    table = rl.TaskTable([task])
    states = walk(task, steps=10, walks=2)
    rows = np.stack([np.asarray(task.encode([s]))[0] for s in states])
    ids = np.zeros(len(rows), np.int32)
    options = dict(target_sources=("goal", "action"), goal_derivations=("plain", "satisfied", "unsatisfied"))
    full = encoding.FlatRelationEncoder(table, **options)
    kept = encoding.FlatRelationEncoder(table, include_derived="domain", **options)
    every, bounds = full.capacity_bounds(table, len(rows), drop_noops=True), kept.capacity_bounds(table, len(rows), drop_noops=True)
    assert not bounds.unbounded
    assert bounds.relation_tuples == {r: every.relation_tuples[r] for r in kept.relation_names}
    assert all(bounds[k] <= every[k] for k in bounds)
    actions = rl.expand(table, rows, ids)
    # philosophers' goal is over a derived predicate, which instance masks cannot hold
    for goals in (None, rl.goal_masks(table, ids)) if name == "triage" else (None,):
        exact = kept.encode(rows, ids, actions=actions, goals=goals)
        counts = np.asarray(exact.relation_counts).sum(axis=0)
        for r, n in zip(kept.relation_names, counts.tolist()):
            assert n <= bounds.relation_tuples[r], (r, n)
        caps = {k: v for k, v in bounds.items() if k != "successors"}
        fixed = kept.encode(rows, ids, actions=actions, goals=goals, capacities=caps)
        assert not fixed.sizes["overflow"]
        assert np.array_equal(np.asarray(fixed.relation_counts), np.asarray(exact.relation_counts))
