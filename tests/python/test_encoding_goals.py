"""The goal relations of the relational encoder through the Python API: the plain, satisfied and unsatisfied variants of
the instance's goal, goals given per row, goal entities and targets, and the arrays that belong to them. The tuples are
checked against a reconstruction from Task.encoding_goal and the atoms of the state."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk

GOALS = ROOT / "tests/data/pddl/goals"
GOAL_ROLE = 2
AUXILIARY_ROLE = 8
TEXT_TASKS = ["gripper__prob05", "blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "philosophers__p03-phil4"]
PDDL_PROBLEMS = ["conjunctive", "forall", "static_false", "forall_negative", "or_static_true", "forall_partly_expandable",
                 "imply_true", "static_true", "contradiction", "forall_nested"]


def goals_task(problem):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task.from_pddl(GOALS / "domain.pddl", GOALS / f"{problem}.pddl")


def cells(batch, graph):
    """{relation name: (tuples of instance-local entity ids, ids of the tuples in the batch)} of one graph."""
    relations = batch.num_relations
    graphs = batch.num_graphs
    names = batch.schema.relation_names
    arities = batch.schema.relation_arities
    relation_major = batch.schema.options["layout"] == "relation"
    base = int(batch.entity_offsets[graph])
    out = {}
    for relation, name in enumerate(names):
        cell = relation * graphs + graph if relation_major else graph * relations + relation
        count = int(batch.relation_counts[graph, relation])
        if not count:
            continue
        begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        arguments = (batch.relation_args[begin:end] - base).reshape(count, arities[relation])
        first = int(batch.relation_offsets[cell])
        out[name] = ([tuple(int(x) for x in row) for row in arguments], list(range(first, first + count)))
    return out


def goal_literals(task):
    return [(x.positive, x.predicate.name, tuple(o.index for o in x.objects)) for x in task.encoding_goal]


def true_atoms(task, state):
    atoms = list(state.atoms()) + list(state.derived_atoms()) + list(task.formalism.static_init)
    return {(a.predicate.name, tuple(o.index for o in a.objects)) for a in atoms}


def goal_reference(task, state, literals, schema_names):
    """{relation name: sorted tuples} of the goal relations for the goal `literals` in `state`; a nullary atom is the
    tuple over the auxiliary entity."""
    atoms = true_atoms(task, state)
    auxiliary = len(task.objects)
    out = {}
    for positive, name, objects in literals:
        holds = ((name, objects) in atoms) == positive
        base = f"[{'+' if positive else '-'}]{name}[g]"
        for relation in (base, base + ("[sat]" if holds else "[unsat]")):
            if relation in schema_names:
                out.setdefault(relation, set()).add(objects or (auxiliary,))
    return {name: sorted(tuples) for name, tuples in out.items()}


def goal_cells(batch, graph, targets):
    """The goal relations of a graph without the goal target slot, and the goal entities of its plain tuples."""
    out = {}
    entities = []
    base = int(batch.entity_offsets[graph])
    schema = batch.schema
    for name, (tuples, ids) in cells(batch, graph).items():
        relation = schema.relation_id(name)
        if schema.relation_lanes[relation] not in ("goal", "goal_derivation"):
            continue
        if targets and schema.relation_slot_roles[relation][0] == schema.slot_role_names.index("goal_target_slot"):
            for entity, tuple_id in zip((t[0] for t in tuples), ids):
                entities.append((entity, tuple_id))
            tuples = [t[1:] for t in tuples]
        out[name] = sorted(tuples) if not targets else tuples
    return out, entities, base


def check_goal_graphs(encoder, batch, states, per_state_literals, targets):
    names = set(batch.schema.relation_names)
    for graph, state in enumerate(states):
        expected = per_state_literals(graph, state, names)
        got, entities, base = goal_cells(batch, graph, targets)
        assert got == expected
        if targets:
            objects = int(batch.num_objects[graph])
            has_auxiliary = any(
                encoder.relation_logical_arities[r] == 0 and batch.relation_counts[graph, r]
                for r in range(batch.num_relations)
            )
            start = objects + int(has_auxiliary)
            assert [e for e, _ in entities] == list(range(start, start + len(entities)))
            role = batch.entity_role[base : int(batch.entity_offsets[graph + 1])]
            ref = batch.entity_ref[base : int(batch.entity_offsets[graph + 1])]
            assert [int(role[e]) for e, _ in entities] == [GOAL_ROLE] * len(entities)
            assert [int(ref[e]) for e, _ in entities] == [i for _, i in entities]
            begin, end = int(batch.target_offsets[graph]), int(batch.target_offsets[graph + 1])
            assert (batch.target_entity[begin:end] - base).tolist() == [e for e, _ in entities]
            assert batch.target_group[begin:end].tolist() == [0] * (end - begin)
            assert int(batch.entity_offsets[graph + 1]) - base == start + len(entities)


OPTIONS = [
    {},
    {"target_sources": ("goal",)},
    {"target_sources": ("goal",), "layout": "relation"},
    {"goal_derivations": ("plain",), "target_sources": ("goal",), "index_dtype": "int64"},
    {"goal_derivations": ("satisfied", "unsatisfied")},
    {"nullary": False, "target_sources": ("goal",)},
    {"include_static": False, "include_derived": False, "emit_counts": True},
]


@pytest.mark.parametrize("options", OPTIONS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()) or "default")
@pytest.mark.parametrize("name", TEXT_TASKS)
def test_the_goal_relations_equal_the_goal_literals_of_a_text_task(name, options):
    task = text_task(name)
    states = walk(task, steps=12, walks=2)
    encoder = encoding.FlatRelationEncoder(task, **options)
    batch = encoder.encode(states)
    literals = goal_literals(task)
    targets = bool(options.get("target_sources"))
    check_goal_graphs(
        encoder, batch, states, lambda g, s, names: goal_reference(task, s, literals, names), targets
    )
    assert batch.goal_complete.tolist() == [int(task.encoding_goal_complete)] * len(states)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("problem", PDDL_PROBLEMS)
def test_the_goal_relations_equal_the_goal_literals_of_a_pddl_goal(problem, layout):
    task = goals_task(problem)
    states = walk(task, steps=10, walks=2)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",), layout=layout)
    batch = encoder.encode(states)
    literals = goal_literals(task)
    check_goal_graphs(encoder, batch, states, lambda g, s, names: goal_reference(task, s, literals, names), True)
    assert batch.goal_complete.tolist() == [int(task.encoding_goal_complete)] * len(states)


def test_a_forall_goal_is_expanded_and_goal_complete_is_per_instance():
    forall, partly = goals_task("forall"), goals_task("forall_partly_expandable")
    table = rl.TaskTable([forall, partly])
    encoder = encoding.FlatRelationEncoder(table)
    states = [forall.initial_state, partly.initial_state]
    batch = encoder.encode(states, [0, 1])
    assert batch.goal_complete.tolist() == [1, 0]
    lit = encoder.relation_id("[+]lit[g]")
    assert batch.relation_counts[:, lit].tolist() == [3, 0]  # lit a, lit b and lit home


def test_the_constant_part_is_emitted_for_a_statically_false_goal():
    task = goals_task("static_false")
    assert task.encoding_goal_unsatisfiable
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied", "unsatisfied"))
    batch = encoder.encode([task.initial_state])
    heavy = encoder.relation_id
    assert batch.relation_counts[0, heavy("[+]heavy[g]")] == 1
    assert batch.relation_counts[0, heavy("[+]heavy[g][unsat]")] == 1
    assert batch.relation_counts[0, heavy("[+]heavy[g][sat]")] == 0


def test_the_auxiliary_entity_counts_the_nullary_goal_tuples():
    task = goals_task("conjunctive")  # the goal holds the nullary literal (done)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain",))
    batch = encoder.encode([task.initial_state])
    objects = len(task.objects)
    assert batch.entity_offsets.tolist() == [0, objects + 1]
    assert int(batch.entity_role[objects]) == AUXILIARY_ROLE
    done = encoder.relation_id("[+]done[g]")
    assert batch.relation_counts[0, done] == 1
    start = int(batch.relation_arg_offsets[done])
    assert int(batch.relation_args[start]) == objects


def fluent_masks(task, states, positive_rate=0.1, negative_rate=0.1, seed=0):
    metadata = rl.as_table(task).atom_metadata()
    slots = len(metadata["atom_pred"])
    words = max(1, (slots + 63) // 64)
    rng = np.random.default_rng(seed)
    pos = np.zeros((len(states), words), np.uint64)
    neg = np.zeros((len(states), words), np.uint64)
    for row in range(len(states)):
        for slot in range(slots):
            draw = rng.random()
            if draw < positive_rate:
                pos[row, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
            elif draw < positive_rate + negative_rate:
                neg[row, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return pos, neg, metadata


def masks_to_literals(task, metadata, pos, neg):
    predicates = task.formalism.predicates
    out = []
    for mask, positive in ((pos, True), (neg, False)):
        for slot in range(len(metadata["atom_pred"])):
            if (int(mask[slot // 64]) >> (slot % 64)) & 1:
                args = tuple(int(a) for a in metadata["atom_args"][slot] if a >= 0)
                out.append((positive, predicates[int(metadata["atom_pred"][slot])].name, args))
    return out


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_goals_of_the_rows_replace_the_instance_goal(layout):
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=3)
    pos, neg, metadata = fluent_masks(task, states, seed=4)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, target_sources=("goal",))
    batch = encoder.encode(states, goals=(pos, neg))
    check_goal_graphs(
        encoder,
        batch,
        states,
        lambda g, s, names: goal_reference(task, s, masks_to_literals(task, metadata, pos[g], neg[g]), names),
        True,
    )
    assert batch.goal_complete.tolist() == [1] * len(states)
    # a single shared row, as rl.goal_masks gives for a task, and u32 halves
    shared = encoder.encode(states, goals=(pos[:1], neg[:1]))
    literals = masks_to_literals(task, metadata, pos[0], neg[0])
    check_goal_graphs(encoder, shared, states, lambda g, s, names: goal_reference(task, s, literals, names), True)
    halves = encoder.encode(states, goals=(pos.view(np.uint32), neg.view(np.uint32)))
    assert halves.relation_args.tolist() == batch.relation_args.tolist()


@pytest.mark.parametrize("name", ["gripper__prob05", "blocks__probBLOCKS-8-0"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_the_goal_masks_of_the_task_give_the_default_output(name, layout):
    task = text_task(name)
    states = walk(task, steps=10, walks=3)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, target_sources=("goal",))
    default = encoder.encode(states)
    given = encoder.encode(states, goals=rl.goal_masks(task))
    assert default.as_dict().keys() == given.as_dict().keys()
    for key, value in default.as_dict().items():
        assert np.array_equal(value, given.as_dict()[key]), key


def test_goals_of_the_rows_over_derived_slots():
    task = goals_task("conjunctive")
    states = walk(task, steps=6, walks=2)
    table = rl.as_table(task)
    metadata = table.atom_metadata()
    derived = len(metadata["derived_pred"])
    assert derived > 0
    words = max(1, (len(metadata["atom_pred"]) + 63) // 64)
    zeros = np.zeros((len(states), words), np.uint64)
    dpos = np.zeros((len(states), max(1, (derived + 63) // 64)), np.uint64)
    dpos[:, 0] = 1
    dneg = np.zeros_like(dpos)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied", "unsatisfied"))
    batch = encoder.encode(states, goals=(zeros, zeros, dpos, dneg))
    name = task.formalism.predicates[int(metadata["derived_pred"][0])].name
    plain = encoder.relation_id(f"[+]{name}[g]")
    assert batch.relation_counts[:, plain].tolist() == [1] * len(states)
    sat = encoder.relation_id(f"[+]{name}[g][sat]")
    unsat = encoder.relation_id(f"[+]{name}[g][unsat]")
    holds = [int(state.holds(task.atom((name, [])))) for state in states]
    assert batch.relation_counts[:, sat].tolist() == holds
    assert batch.relation_counts[:, unsat].tolist() == [1 - h for h in holds]


def test_bad_goals_raise():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    states = walk(task, steps=3, walks=1)
    pos, neg, _ = fluent_masks(task, states)
    with pytest.raises(ValueError, match="^mymyr: encoding: goals must be"):
        encoder.encode(states, goals=(pos,))
    with pytest.raises(TypeError, match="^mymyr: encoding: goals must be"):
        encoder.encode(states, goals="no")
    with pytest.raises(ValueError, match="rows: one per state"):
        encoder.encode(states, goals=(pos[:2], neg[:2]))
    with pytest.raises(ValueError, match="another shape"):
        encoder.encode(states, goals=(pos, neg[:1]))
    with pytest.raises(ValueError, match="narrower than its instance"):
        encoder.encode(states, goals=(pos[:, :0], neg[:, :0]))
    wide = np.zeros((len(states), pos.shape[1]), np.uint64)
    wide[:, -1] = np.uint64(1) << np.uint64(63)
    with pytest.raises(ValueError, match="sets an atom bit its instance does not have"):
        encoder.encode(states, goals=(wide, neg))
    with pytest.raises(ValueError, match="no goal relations"):
        encoding.FlatRelationEncoder(task, goal_derivations=()).encode(states, goals=(pos, neg))


def test_the_arrays_exist_with_the_goal_relations_and_the_targets_only():
    task = text_task("gripper__prob05")
    state = task.initial_state
    plain = encoding.FlatRelationEncoder(task).encode(state)
    assert plain.goal_complete.tolist() == [int(task.encoding_goal_complete)]
    for name in ("target_offsets", "target_entity", "target_group"):
        with pytest.raises(AttributeError):
            getattr(plain, name)
    none = encoding.FlatRelationEncoder(task, goal_derivations=()).encode(state)
    with pytest.raises(AttributeError):
        none.goal_complete
    with_targets = encoding.FlatRelationEncoder(task, target_sources=("goal",)).encode(state)
    assert with_targets.target_offsets.tolist()[0] == 0
    assert "goal_complete" in with_targets.as_dict() and "target_group" in with_targets.as_dict()
    assert "goal_complete" not in none.as_dict()


def test_capacities_carry_the_targets():
    task = text_task("gripper__prob05")
    states = walk(task, steps=6, walks=2)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    exact = encoder.encode(states)
    sizes = {"entities": len(exact.entity_role), "tuples": int(exact.relation_offsets[-1]),
             "args": len(exact.relation_args), "targets": len(exact.target_entity)}
    with pytest.raises(ValueError, match="needs the key 'targets'"):
        encoder.encode(states, capacities={k: v for k, v in sizes.items() if k != "targets"})
    roomy = encoder.encode(states, capacities={**sizes, "targets": sizes["targets"] + 4})
    assert roomy.sizes["targets"] == sizes["targets"] and not roomy.sizes["overflow"]
    assert len(roomy.target_entity) == sizes["targets"] + 4
    assert not roomy.target_entity[sizes["targets"] :].any()
    assert roomy.target_entity[: sizes["targets"]].tolist() == exact.target_entity.tolist()
    short = encoder.encode(states, capacities={**sizes, "targets": sizes["targets"] - 1})
    assert short.sizes["capacity_overflow"] and not short.target_entity.any()


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_threads_give_identical_goal_batches(layout):
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=30, walks=3)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, target_sources=("goal",))
    many = states * (4000 // len(states) + 1)
    reference = encoder.encode(many)
    for threads in (2, 3):
        batch = encoder.encode(many, threads=threads)
        for key, value in reference.as_dict().items():
            assert np.array_equal(value, batch.as_dict()[key]), (threads, key)


def test_single_states_equal_the_batch():
    task = text_task("gripper__prob05")
    states = walk(task, steps=5, walks=2)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("goal",))
    batch = encoder.encode(states)
    for graph, state in enumerate(states):
        one = encoder.encode(state)
        got, entities, _ = goal_cells(one, 0, True)
        want, wanted_entities, _ = goal_cells(batch, graph, True)
        assert got == want
        base_one = int(one.relation_offsets[0])
        base_batch = int(batch.relation_offsets[graph * batch.num_relations])
        assert [(e, i - base_one) for e, i in entities] == [(e, i - base_batch) for e, i in wanted_entities]


def test_the_fingerprint_depends_on_the_targets():
    task = text_task("gripper__prob05")
    assert (
        encoding.EncodingSchema(task, target_sources=("goal",)).schema_fingerprint
        != encoding.EncodingSchema(task).schema_fingerprint
    )
