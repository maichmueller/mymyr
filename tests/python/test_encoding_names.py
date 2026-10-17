"""Names of the entities and tuples of an encoded graph (FlatRelationEncoder.entity_names and describe): the naming rule of
each entity role against an independent reconstruction from the task, both layouts, the arrays an encoder may omit,
stability across processes, and the errors."""

import hashlib
import os

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from python_children import run_code

GOAL_ROLE, ACTION_ROLE, VIRTUAL_ROLE, AUXILIARY_ROLE = 2, 4, 1, 8
GOALS = ROOT / "tests/data/pddl/goals"
OPTIONS = dict(goal_derivations=("plain", "satisfied"), target_sources=("goal", "action"), predicate_virtual_nodes=True)


def encode(task, states, **options):
    encoder = encoding.FlatRelationEncoder(task, **{**OPTIONS, **options})
    return encoder, encoder.encode(states, actions=rl.expand(rl.as_table(task), states))


def roles_of(batch, graph):
    begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
    return batch.entity_role[begin:end].tolist(), batch.entity_ref[begin:end].tolist()


def atom_names(atom):
    return tuple(o.name for o in atom.objects)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1"])
def test_every_entity_role_is_named_by_its_rule(name, layout):
    task = text_task(name)
    states = walk(task, steps=5, walks=2)
    encoder, batch = encode(task, states, layout=layout)
    literals = {(x.positive, x.predicate.name, tuple(task.objects[o.index] for o in x.objects)) for x in task.encoding_goal}
    schemas = {s.name for s in task.formalism.schemas}
    for graph, state in enumerate(states):
        names = encoder.entity_names(batch, graph)
        roles, refs = roles_of(batch, graph)
        assert len(names) == len(roles)
        goal_names, virtual, action_names = [], [], []
        for entity, (role, ref, got) in enumerate(zip(roles, refs, names)):
            if role == 0:
                assert got == task.objects[ref]
            elif role == AUXILIARY_ROLE:
                assert got == "auxiliary" and ref == -1
            elif role == GOAL_ROLE:
                goal_names.append(got)
            elif role == VIRTUAL_ROLE:
                virtual.append((ref, got))
            elif role == ACTION_ROLE:
                action_names.append(got)
        assert sorted(goal_names) == sorted(
            f"[{'+' if positive else '-'}]({' '.join((predicate, *objects))})[g]" for positive, predicate, objects in literals)
        assert sorted(action_names) == sorted(str(a) for a in state.applicable_actions())
        assert all(n.split()[0].strip("(") in schemas for n in action_names)
        used = {a.predicate.name for a in [*state.atoms(), *state.derived_atoms(), *task.formalism.static_init]}
        used |= {x.predicate.name for x in task.encoding_goal}
        order = sorted(used, key=encoder.relation_id)
        assert virtual == [(encoder.relation_id(p), f"predicate:{p}") for p in order]


def test_the_auxiliary_entity_exists_in_the_graphs_with_a_nullary_tuple():
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=12, walks=2)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=())
    batch = encoder.encode(states)
    seen = set()
    for graph, state in enumerate(states):
        names = encoder.entity_names(batch, graph)
        nullary = any(a.predicate.name == "handempty" for a in state.atoms())
        seen.add(nullary)
        assert names[: len(task.objects)] == task.objects
        assert names[len(task.objects) :] == (["auxiliary"] if nullary else [])
    assert seen == {True, False}
    assert not batch.implicit_entities
    auto = encoding.FlatRelationEncoder(task, goal_derivations=(), entity_arrays="auto", include_actions=False)
    implicit = auto.encode(states)
    assert implicit.implicit_entities
    for graph in range(len(states)):
        assert auto.entity_names(implicit, graph) == encoder.entity_names(batch, graph)


def test_a_nullary_goal_literal_and_a_negative_one_are_named_after_their_tuple():
    task = text_task("blocks__probBLOCKS-8-0")
    table = rl.as_table(task)
    metadata = table.atom_metadata()
    predicates = task.formalism.predicates
    slot = {}
    for index, predicate in enumerate(metadata["atom_pred"]):
        arguments = tuple(task.objects[int(a)] for a in metadata["atom_args"][index] if a >= 0)
        slot[(predicates[int(predicate)].name, arguments)] = index
    pos = np.zeros((1, table.words), np.uint64)
    neg = np.zeros((1, table.words), np.uint64)
    for (predicate, arguments), sign in ((("handempty", ()), pos), (("clear", ("o3",)), neg)):
        bit = slot[(predicate, arguments)]
        sign[0, bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",), goal_derivations=("plain",))
    batch = encoder.encode([task.initial_state], goals=(pos, neg))
    names = encoder.entity_names(batch, 0)
    assert "[+](handempty)[g]" in names and "[-](clear o3)[g]" in names
    roles, _ = roles_of(batch, 0)
    assert sorted(n for n, r in zip(names, roles) if r == GOAL_ROLE) == ["[+](handempty)[g]", "[-](clear o3)[g]"]


def test_a_table_names_the_objects_of_each_rows_instance():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    blocks = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / f"{p}.pddl") for p in ("smedium", "probBLOCKS-8-0")]
    encoder = encoding.FlatRelationEncoder(rl.TaskTable(tasks), nullary=False, goal_derivations=())
    ids = np.array([1, 0, 1, 0], np.int32)
    batch = encoder.encode([tasks[i].initial_state for i in ids], ids)
    for graph, instance in enumerate(ids):
        assert encoder.entity_names(batch, graph) == tasks[instance].objects


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_describe_lists_the_entities_and_the_tuples_by_name(layout):
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=5, walks=1)
    encoder, batch = encode(task, states, layout=layout, predicate_virtual_nodes=False)
    for graph, state in enumerate(states):
        description = encoder.describe(batch, graph)
        names = encoder.entity_names(batch, graph)
        roles, _ = roles_of(batch, graph)
        assert description.graph == graph and description.task_id == 0
        assert description.entities == tuple((encoder.entity_role_names[r], n) for r, n in zip(roles, names))
        for predicate in ("on", "clear", "ontable", "holding"):
            expected = sorted(atom_names(a) for a in state.atoms() if a.predicate.name == predicate)
            assert sorted(description.relations.get(predicate, ())) == expected
        assert sorted(description.relations["handempty"] if "handempty" in description.relations else ()) == (
            [("auxiliary",)] if any(a.predicate.name == "handempty" for a in state.atoms()) else [])
        assert list(description.relations) == [n for n in encoder.relation_names if n in description.relations]
        actions = sorted(description.relations.get("stack", ())) + sorted(description.relations.get("unstack", ()))
        applicable = [a for a in state.applicable_actions() if a.name in ("stack", "unstack")]
        assert len(actions) == len(applicable)
        text = str(description)
        assert text.startswith(f"graph {graph} (instance 0): {len(names)} entities")
        assert all(n in text for n in names)


def test_the_names_do_not_depend_on_the_layout():
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=6, walks=2)
    first_encoder, first = encode(task, states, layout="graph")
    second_encoder, second = encode(task, states, layout="relation")
    for graph in range(len(states)):
        assert first_encoder.entity_names(first, graph) == second_encoder.entity_names(second, graph)
        assert first_encoder.describe(first, graph) == second_encoder.describe(second, graph)


def test_the_names_are_stable_across_runs_and_hash_seeds():
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=6, walks=2)
    encoder, batch = encode(task, states)
    text = "\n".join(str(encoder.describe(batch, g)) for g in range(len(states)))
    code = """
import hashlib, sys
import mymyr
from mymyr import encoding, rl
from conftest import text_task, walk
task = text_task("logistics00__probLOGISTICS-6-1")
states = walk(task, steps=6, walks=2)
encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied"), target_sources=("goal", "action"), predicate_virtual_nodes=True)
batch = encoder.encode(states, actions=rl.expand(rl.as_table(task), states))
text = "\\n".join(str(encoder.describe(batch, g)) for g in range(len(states)))
print(hashlib.sha256(text.encode()).hexdigest())
"""
    for seed in ("0", "99"):
        child = run_code(code, env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "tests/python")},
                         capture_output=True)
        assert child.stdout.strip() == hashlib.sha256(text.encode()).hexdigest()


def test_batches_of_the_capacity_form_are_named_and_overflowed_ones_refused():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    states = walk(task, steps=4, walks=1)
    exact = encoder.encode(states)
    sizes = exact.as_dict()
    roomy = {"entities": int(exact.entity_offsets[-1]) + 20, "tuples": 10_000, "args": 10_000, "targets": 100}
    padded = encoder.encode(states, capacities=roomy)
    assert not padded.sizes["overflow"]
    for graph in range(len(states)):
        assert encoder.entity_names(padded, graph) == encoder.entity_names(exact, graph)
    overflowed = encoder.encode(states, capacities={"entities": 1, "tuples": 1, "args": 1, "targets": 1})
    with pytest.raises(ValueError, match="mymyr: encoding: the batch overflowed"):
        encoder.entity_names(overflowed, 0)
    assert sizes["num_graphs"] == len(states)


def test_the_name_calls_check_their_arguments():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    states = walk(task, steps=2, walks=1)
    batch = encoder.encode(states)
    for bad in (-1, len(states), 10**6):
        with pytest.raises(IndexError, match="mymyr: encoding: graph"):
            encoder.entity_names(batch, bad)
        with pytest.raises(IndexError, match="mymyr: encoding: graph"):
            encoder.describe(batch, bad)
    other = encoding.FlatRelationEncoder(task, target_sources=("goal",), layout="relation")
    with pytest.raises(ValueError, match="mymyr: encoding: the batch was encoded with another schema"):
        other.entity_names(batch, 0)
    with pytest.raises(TypeError):
        encoder.entity_names("batch", 0)
    empty = encoder.encode([])
    with pytest.raises(IndexError):
        encoder.entity_names(empty, 0)


def test_the_goals_of_a_pddl_task_with_negative_literals_are_named():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    task = mymyr.Task.from_pddl(GOALS / "domain.pddl", GOALS / "forall_negative.pddl")
    states = [task.initial_state]
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",), goal_derivations=("plain",))
    batch = encoder.encode(states)
    names = encoder.entity_names(batch, 0)
    roles, _ = roles_of(batch, 0)
    goal = sorted(n for n, r in zip(names, roles) if r == GOAL_ROLE)
    expected = sorted(
        f"[{'+' if x.positive else '-'}]({' '.join([x.predicate.name] + [task.objects[o.index] for o in x.objects])})[g]"
        for x in task.encoding_goal)
    assert goal == expected
    assert any(n.startswith("[-]") for n in goal)
