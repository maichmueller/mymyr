"""include_derived='domain': the derived predicates the domain declares keep their relations, the helpers that
normalization introduces for quantified and disjunctive conditions have none. The mark comes from the front end and
survives the text format; the batches of the remaining relations equal those of the full catalog."""

import pickle

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
    assert {n.strip("[+-]").split("[")[0] for n in compiled} == {"axiom_0", "axiom_1"}
    assert by_lane(every, "derived") == ["axiom_0", "axiom_1", "blocked", "blocked-trans"]
    assert by_lane(declared, "derived") == ["blocked", "blocked-trans"]
    assert not any(declared.relation_compiled)
    # the goal and history relations of the declared derived predicates stay
    assert {"[+]blocked[g]", "[+]blocked[hist]", "[-]blocked-trans[g][sat]"} <= set(declared.relation_names)
    # False drops the state relations of both kinds and keeps their goal and history relations
    none = encoding.EncodingSchema(task, include_derived=False)
    assert by_lane(none, "derived") == [] and "[+]axiom_0[g]" in none.relation_names


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
    assert by_lane(every, "derived") == ["axiom_0", "axiom_1"]
    assert not [n for n in declared.relation_names if "axiom_" in n]
    assert all(c == ("axiom_" in n) for n, c in zip(every.relation_names, every.relation_compiled))


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
    seen = {"blocked": 0, "axiom_0": 0}
    for graph in range(len(states)):
        a, b = unpack(full, graph), unpack(kept, graph)
        assert b == {n: t for n, t in a.items() if n in names}
        seen["blocked"] += len(b.get("blocked", []))
        seen["axiom_0"] += len(a.get("axiom_0", []))
    assert seen["blocked"] > 0 and seen["axiom_0"] > 0
    dropped = [i for i, c in enumerate(full.schema.relation_compiled) if c]
    assert int(kept.relation_counts.sum()) == int(full.relation_counts.sum()) - int(full.relation_counts[:, dropped].sum())


def test_the_mark_survives_the_text_format_and_pickling(tmp_path):
    task = philosophers()
    text = task.to_text()
    assert any(line.startswith("C ") and line.endswith(" axiom_0") for line in text.splitlines())
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
