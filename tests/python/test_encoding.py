"""Packed relational observations against independent semantic reconstruction."""

import gc
import importlib.util
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import SMALL_TASKS, text_task, walk


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


def atom_key(atom):
    return atom.predicate.name, tuple(obj.index for obj in atom.objects)


def reference(task, state, include_static=True, include_derived=True, goal=None):
    output = {}
    for predicate in task.formalism.predicates:
        if predicate.kind == "static" and not include_static:
            continue
        if predicate.kind == "derived" and not include_derived:
            continue
        for objects in itertools.product(range(len(task.objects)), repeat=predicate.arity):
            atom = task.atom((predicate.name, list(objects)))
            if state.holds(atom):
                name, args = atom_key(atom)
                output.setdefault(name, []).append(args)
    for literal in (task.goal_condition if goal is None else goal).literals:
        name, args = atom_key(literal.atom)
        polarity = "+" if literal.positive else "-"
        status = "sat" if literal.holds(state) else "unsat"
        output.setdefault(f"[{polarity}]{name}[g][{status}]", []).append(args)
    return {name: sorted(values) for name, values in output.items()}


DOMAIN = """(define (domain features)
 (:requirements :strips :negative-preconditions :derived-predicates)
 (:predicates (fixed ?x) (flag ?x) (available ?x) (ready))
 (:derived (available ?x) (flag ?x))
 (:action enable :parameters (?x) :precondition (not (flag ?x)) :effect (flag ?x))
 (:action disable :parameters (?x) :precondition (flag ?x) :effect (not (flag ?x)))
 (:action switch :parameters () :precondition (and) :effect (ready)))"""


def feature_task(objects="a b", atoms="frozen"):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string(DOMAIN)
    problem = f"""(define (problem p) (:domain features) (:objects {objects})
      (:init (fixed a) (flag a))
      (:goal (and (flag b) (not (flag a)) (fixed a) (not (fixed b))
                  (available b) (not (available a)) (ready))))"""
    return mymyr.Task(domain.instantiate_string(problem), atoms=atoms)


@pytest.mark.parametrize("atoms", ["frozen", "lazy"])
@pytest.mark.parametrize("include_static,include_derived", [(True, True), (False, False)])
def test_semantics_and_default_signed_derived_goals(atoms, include_static, include_derived):
    task = feature_task(atoms=atoms)
    states = [task.initial_state]
    state = task.apply(states[0], "(enable b)")
    state = task.apply(state, "(disable a)")
    states.append(task.apply(state, "(switch)"))
    encoder = encoding.FlatRelationEncoder(task, include_static=include_static, include_derived=include_derived)
    batch = encoder.encode(states)
    for graph, state in enumerate(states):
        got = {name: tuples for name, tuples in unpack(batch, graph).items() if tuples}
        assert got == reference(task, state, include_static, include_derived)
    ready = encoder.relation_names.index("ready")
    assert batch.relation_arities[ready] == 0
    assert batch.relation_counts[:, ready].tolist() == [0, 1]
    assert batch.version == 1 and batch.num_graphs == 2


def test_explicit_static_negative_and_empty_goals():
    task = feature_task()
    state = task.initial_state
    goal = task.ground_condition(["(fixed a)", "(not (fixed b))", "(fixed b)", "(not (fixed a))",
                                  "(flag a)", "(not (flag a))", "(available a)", "(not (available a))"])
    encoder = encoding.FlatRelationEncoder(task)
    batch = encoder.encode(state, goals=goal)
    assert {name: values for name, values in unpack(batch, 0).items() if values} == reference(task, state, goal=goal)
    empty = encoder.encode(state, goals=task.ground_condition([]))
    for relation, source in enumerate(encoder.relation_sources):
        if source.startswith("goal"):
            assert empty.relation_counts[0, relation] == 0


def test_mifrost_neutral_engine_semantic_fixture():
    fixture = json.loads((Path(__file__).parents[1] / "data/encoding/mifrost_flat_goals.json").read_text())
    task = feature_task()
    goal = task.ground_condition(fixture["goals"])
    batch = encoding.FlatRelationEncoder(task).encode(task.initial_state, goals=goal)
    got = unpack(batch, 0)
    for name, expected in fixture["relations"].items():
        assert [list(values) for values in got[name]] == expected
    assert not got["[+]flag[g][sat]"] and not got["[-]flag[g][sat]"]


def test_batch_global_objects_and_expansion_action_readouts():
    first, second = feature_task(), feature_task("a b c d")
    table = rl.TaskTable([first, second])
    states = [second.initial_state, first.initial_state, second.state(["(flag b)"])]
    ids = np.array([1, 0, 1], dtype=np.int32)
    expansion = rl.expand(table, states, task_ids=ids)
    encoder = encoding.FlatRelationEncoder(table)
    batch = encoder.encode(states, task_ids=ids, actions=expansion)
    assert batch.object_offsets.tolist() == [0, 4, 6, 10]
    assert batch.task_ids.tolist() == ids.tolist()
    assert np.array_equal(batch.action_offsets, expansion.offsets)
    assert np.array_equal(batch.action_schema, expansion.schema)
    assert np.array_equal(batch.action_binding, expansion.binding)
    assert np.array_equal(batch.action_parent, expansion.parent)
    assert len(np.unique(batch.action_tuple)) == expansion.total
    relations = len(encoder.relation_names)
    for action, tuple_id in enumerate(batch.action_tuple):
        cell = np.searchsorted(batch.relation_tuple_offsets, tuple_id, side="right") - 1
        graph, relation = divmod(cell, relations)
        assert graph == expansion.parent[action]
        assert batch.relation_schemas[relation] == expansion.schema[action]
        local = tuple_id - batch.relation_tuple_offsets[cell]
        arity = batch.relation_arities[relation]
        start = batch.relation_offsets[cell] + local * arity
        objects = batch.relation_args[start : start + arity] - batch.object_offsets[graph]
        assert objects.tolist() == expansion.binding[action, :arity].tolist()
    for graph, (task_id, state) in enumerate(zip(ids, states)):
        got = {name: values for name, values in unpack(batch, graph).items()
               if values and not name.endswith("[action]")}
        assert got == reference([first, second][task_id], state)


@pytest.mark.parametrize("name", SMALL_TASKS)
def test_walk_fluent_and_goal_reference(name):
    task = text_task(name, atoms="lazy")
    states = walk(task, steps=3, walks=1)
    encoder = encoding.FlatRelationEncoder(task, include_static=False, include_derived=False)
    batch = encoder.encode(states)
    for graph, state in enumerate(states):
        expected = {}
        for atom in state.atoms():
            name, args = atom_key(atom)
            expected.setdefault(name, []).append(args)
        for literal in task.goal_condition.literals:
            name, args = atom_key(literal.atom)
            expected.setdefault(f"[{'+' if literal.positive else '-'}]{name}[g][{'sat' if literal.holds(state) else 'unsat'}]", []).append(args)
        expected = {name: sorted(values) for name, values in expected.items()}
        assert {name: values for name, values in unpack(batch, graph).items() if values} == expected


def test_lazy_growth_and_slot_order_do_not_change_semantic_encoding():
    first, second = feature_task(atoms="lazy"), feature_task(atoms="lazy")
    enc_first, enc_second = encoding.FlatRelationEncoder(first), encoding.FlatRelationEncoder(second)
    first.state(["(flag b)"])
    second.state(["(ready)"])
    first_state, second_state = first.state(["(flag b)", "(ready)"]), second.state(["(flag b)", "(ready)"])
    first_batch, second_batch = enc_first.encode(first_state), enc_second.encode(second_state)
    assert first_batch.schema == second_batch.schema
    assert first_batch.object_names == second_batch.object_names
    assert first_batch.task_fingerprints == second_batch.task_fingerprints
    assert unpack(first_batch, 0) == unpack(second_batch, 0)
    for name in ["relation_args", "relation_counts", "relation_offsets", "relation_tuple_offsets"]:
        assert np.array_equal(getattr(first_batch, name), getattr(second_batch, name))


def test_problem_local_goal_predicates_stay_instance_scoped():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain local-goal)
      (:requirements :strips :disjunctive-preconditions) (:predicates (flag ?x))
      (:action enable :parameters (?x) :precondition (and) :effect (flag ?x)))""")
    tasks = [mymyr.Task(domain.instantiate_string(f"""(define (problem p) (:domain local-goal)
      (:objects a b c) (:init (flag a)) (:goal {goal}))"""), atoms="lazy")
             for goal in ["(or (flag a) (flag b))", "(or (flag b) (flag c))"]]
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table)
    batch = encoder.encode([task.initial_state for task in tasks], task_ids=[0, 1])
    locals_by_instance = {instance: [relation for relation, owner in enumerate(batch.relation_instances)
                                  if owner == instance and batch.relation_sources[relation] == "state"]
                          for instance in range(2)}
    assert all(locals_by_instance.values())
    for instance, relations in locals_by_instance.items():
        for relation in relations:
            assert f"[@instance:{instance}]" in batch.relation_names[relation]
            assert batch.relation_counts[1 - instance, relation] == 0
    assert batch.relation_counts[0, locals_by_instance[0]].sum() == 1
    assert batch.relation_counts[1, locals_by_instance[1]].sum() == 0
    other = encoding.FlatRelationEncoder(rl.TaskTable(list(reversed(tasks))))
    domain_relations = table.num_domain_predicates * 5
    assert encoder.relation_names[:domain_relations] == other.relation_names[:domain_relations]


def test_normalization_introduced_action_parameters_are_not_projected_away():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain witnesses) (:requirements :adl)
      (:predicates (flag ?x) (chosen ?x))
      (:action choose :parameters (?x) :precondition (exists (?witness) (flag ?witness))
         :effect (chosen ?x)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem p) (:domain witnesses)
      (:objects a b) (:init (flag a)) (:goal (chosen b)))"""), atoms="lazy")
    expansion = rl.expand(task, [task.initial_state])
    batch = encoding.FlatRelationEncoder(task).encode(task.initial_state, actions=expansion)
    assert batch.action_binding.shape == (2, 2)
    assert np.array_equal(batch.action_binding, expansion.binding)
    relation = batch.relation_names.index("choose[action]")
    assert batch.relation_arities[relation] == 2
    assert batch.relation_counts[0, relation] == 2


def test_empty_states_batches_and_no_actions():
    task = feature_task(atoms="lazy")
    encoder = encoding.FlatRelationEncoder(task)
    empty_state = task.state([])
    assert unpack(encoder.encode(empty_state), 0) == unpack(encoder.encode([empty_state]), 0)
    batch = encoder.encode([])
    assert batch.num_graphs == 0 and batch.relation_counts.shape == (0, len(encoder.relation_names))
    assert batch.object_offsets.tolist() == [0] and batch.relation_offsets.tolist() == [0]
    assert batch.action_binding.shape == (0, task.label_width)
    no_labels = encoder.encode(task.initial_state)
    assert no_labels.action_offsets.tolist() == [0, 0]


@pytest.mark.parametrize("with_actions", [False, True])
def test_action_readouts_reject_source_instance_mismatch(with_actions):
    first, second = feature_task(), feature_task("a b c")
    table = rl.TaskTable([first, second])
    first_state = first.initial_state
    if not with_actions:
        domain = mymyr.Domain.from_string("""(define (domain stuck) (:requirements :strips)
            (:predicates (ready) (done))
            (:action finish :parameters () :precondition (ready) :effect (done)))""")
        tasks = [mymyr.Task(domain.instantiate_string(f"""(define (problem p{index}) (:domain stuck)
            (:init) (:goal (done)))"""), atoms="frozen") for index in range(2)]
        first, second = tasks
        table = rl.TaskTable(tasks)
        first_state = first.initial_state
    expansion = rl.expand(table, [first_state], task_ids=np.array([0], np.int32))
    assert bool(expansion.total) == with_actions
    assert expansion.task_ids == [0]
    with pytest.raises(ValueError, match="source task ids"):
        encoding.FlatRelationEncoder(table).encode(
            [second.initial_state], task_ids=np.array([1], np.int32), actions=expansion
        )


def test_owned_results_and_readonly_numpy():
    task = feature_task()
    encoder = encoding.FlatRelationEncoder(task)
    states = task.encode([task.initial_state])
    batch = encoder.encode(states)
    retained = batch.relation_args
    expected = retained.copy()
    states[:] = 0
    encoder.encode(states)
    schema = batch.schema
    del task, encoder, batch, states
    gc.collect()
    assert np.array_equal(retained, expected) and schema["version"] == 1
    assert not retained.flags.writeable
    with pytest.raises(ValueError):
        retained[0] = 1


def test_concurrent_encoding_is_reentrant():
    task = feature_task(atoms="lazy")
    encoder = encoding.FlatRelationEncoder(task)
    state = task.initial_state
    expected = encoder.encode(state).relation_args.copy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: encoder.encode(state).relation_args.copy(), range(64)))
    assert all(np.array_equal(value, expected) for value in results)


@pytest.mark.parametrize("framework", ["numpy", "dlpack", "torch", "jax"])
def test_framework_exports(framework):
    if framework in {"torch", "jax"}:
        pytest.importorskip(framework)
    task = feature_task()
    batch = encoding.FlatRelationEncoder(task).encode(task.initial_state)
    values = batch.as_dict(framework=framework)
    if framework == "dlpack":
        got = np.from_dlpack(values["relation_counts"])
    else:
        got = np.asarray(values["relation_counts"])
    assert np.array_equal(got, batch.relation_counts)
    assert values["schema"] == batch.schema


def test_rejects_numeric_tasks():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain numeric) (:requirements :strips :numeric-fluents)
       (:predicates (ready)) (:functions (fuel))
       (:action consume :parameters () :precondition (ready) :effect (decrease (fuel) 1)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem n) (:domain numeric)
       (:init (ready) (= (fuel) 2)) (:goal (<= (fuel) 0)))"""))
    with pytest.raises(ValueError, match="numeric"):
        encoding.FlatRelationEncoder(task)


def test_rejects_bad_identity_ids_expansion_and_inputs():
    first, second = feature_task(), feature_task("a b c")
    encoder = encoding.FlatRelationEncoder(first)
    with pytest.raises(ValueError, match="another task"):
        encoder.encode(second.initial_state)
    with pytest.raises(ValueError, match="another task"):
        encoder.encode(first.initial_state, goals=second.goal_condition)
    with pytest.raises(ValueError, match="another task"):
        encoder.encode(first.initial_state, actions=rl.expand(second, [second.initial_state]))
    with pytest.raises(ValueError, match="different row counts"):
        encoder.encode([first.initial_state] * 2, actions=rl.expand(first, [first.initial_state]))
    with pytest.raises(ValueError, match="overflow"):
        encoder.encode(first.initial_state, actions=rl.expand(first, [first.initial_state], capacity=1))
    with pytest.raises(TypeError, match="Expansion"):
        encoder.encode(first.initial_state, actions=object())
    bad = first.encode([first.initial_state])
    bad[0, -1] |= np.uint64(1 << 63)
    with pytest.raises(ValueError, match="unassigned"):
        encoder.encode(bad)
    with pytest.raises(TypeError, match="GroundCondition"):
        encoder.encode(first.initial_state, goals="(flag a)")
    with pytest.raises(ValueError, match="numeric"):
        encoder.encode(first.initial_state, goals=first.ground_condition([], constraints=["(> 2 1)"]))
    with pytest.raises(ValueError, match="per state row"):
        encoder.encode([first.initial_state] * 2, goals=[first.goal_condition])
    table_encoder = encoding.FlatRelationEncoder(rl.TaskTable([first, second]))
    with pytest.raises(ValueError, match="task_ids"):
        table_encoder.encode([first.initial_state])
    with pytest.raises(ValueError, match="task id"):
        table_encoder.encode([first.initial_state], task_ids=[2])
    with pytest.raises(ValueError, match="another task"):
        table_encoder.encode([first.initial_state], task_ids=[1])
    with pytest.raises(TypeError, match="TaskTable"):
        encoding.FlatRelationEncoder(rl.TaskSuite([rl.TaskTable([first])]))


def test_generated_native_encoding_stubs_are_typed():
    directory = Path(importlib.util.find_spec("mymyr._core").origin).parent / "_core"
    path = directory / "_encoding.pyi"
    if not path.is_file():
        pytest.skip("no encoding stubs installed")
    source = path.read_text()
    assert "states: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack" in source
    assert "table: mymyr._core._rl.TaskTable | mymyr._core.Task | mymyr._core.TaskHandle" in source
    assert "actions: mymyr._core._rl.Expansion | None = None" in source
    assert "goals: mymyr._core._formalism.GroundCondition | Sequence[mymyr._core._formalism.GroundCondition | None]" in source
    assert "def relation_counts(self) -> Any" in source
    assert "-> dict[str, Any]" in source
    assert ": object" not in source and "-> object" not in source
