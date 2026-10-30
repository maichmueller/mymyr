"""The goal relations of the relational encoder through the Python API: the plain, satisfied and unsatisfied variants of
the instance's goal, goals given per row, goal entities and targets, and the arrays that belong to them. The tuples are
checked against a reconstruction from Task.encoding_goal and the atoms of the state."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk

GOALS = ROOT / "tests/data/pddl/goals"
WIDE = ROOT / "tests/data/pddl/wide_predicates"
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
    """{relation name: (tuples of instance-local entity ids, indices of the tuples among the graph's tuples in
    graph-major order)} of one graph."""
    relations = batch.num_relations
    graphs = batch.num_graphs
    names = batch.schema.relation_names
    arities = batch.schema.relation_arities
    relation_major = batch.schema.options["layout"] == "relation"
    base = int(batch.entity_offsets[graph])
    out = {}
    index = 0
    for relation, name in enumerate(names):
        cell = relation * graphs + graph if relation_major else graph * relations + relation
        count = int(batch.relation_counts[graph, relation])
        if not count:
            continue
        begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        arguments = (batch.relation_args[begin:end] - base).reshape(count, arities[relation])
        out[name] = ([tuple(int(x) for x in row) for row in arguments], list(range(index, index + count)))
        index += count
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


def goal_is_complete(task, options):
    """goal_complete of an instance: its goal could be expanded into literals and the options left out no literal whose
    truth a state can change (a nullary literal with nullary="omit"; a static literal is fixed)."""
    dropped = options.get("nullary", "auxiliary") == "omit" and any(
        not x.objects and x.predicate.kind != "static" for x in task.encoding_goal
    )
    return int(task.encoding_goal_complete and not dropped)


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
    {"nullary": "omit", "target_sources": ("goal",)},
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
    assert batch.goal_complete.tolist() == [goal_is_complete(task, options)] * len(states)


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


def test_goal_relations_wider_than_every_fluent_relation_are_encoded_whole():
    # wide_predicates: on/1 is the only fluent predicate, link/3 is static and trio/3 derived; a, b, c are objects 0, 1,
    # 2. The goal is (link a b c) (not (link b b b)) (trio a b c) (not (trio b c a)) (on b); in the initial state (on a)
    # holds and so do (trio a a a), the static link atoms and (not (trio b c a)).
    task = mymyr.Task.from_pddl(WIDE / "domain.pddl", WIDE / "p01.pddl") if hasattr(mymyr, "Domain") else pytest.skip(
        "built without the PDDL front end"
    )
    expected_goal = {
        "[+]link[g]": [(0, 1, 2)], "[+]link[g][sat]": [(0, 1, 2)],
        "[-]link[g]": [(1, 1, 1)], "[-]link[g][sat]": [(1, 1, 1)],
        "[+]trio[g]": [(0, 1, 2)], "[+]trio[g][unsat]": [(0, 1, 2)],
        "[-]trio[g]": [(1, 2, 0)], "[-]trio[g][sat]": [(1, 2, 0)],
        "[+]on[g]": [(1,)], "[+]on[g][unsat]": [(1,)],
    }
    row_goal = {"[+]on[g]": [(0,)], "[+]on[g][sat]": [(0,)], "[-]on[g]": [(2,)], "[-]on[g][sat]": [(2,)],
                "[+]trio[g]": [(0, 1, 0)], "[+]trio[g][unsat]": [(0, 1, 0)],
                "[-]trio[g]": [(1, 2, 0)], "[-]trio[g][sat]": [(1, 2, 0)]}
    metadata = rl.as_table(task).atom_metadata()
    on = {int(metadata["atom_args"][slot][0]): slot for slot in range(len(metadata["atom_pred"]))}
    derived = {
        tuple(int(x) for x in metadata["derived_args"][slot]): slot for slot in range(len(metadata["derived_pred"]))
    }
    words = max(1, (len(metadata["atom_pred"]) + 63) // 64)
    pos, neg = np.zeros((1, words), np.uint64), np.zeros((1, words), np.uint64)
    pos[0, on[0] // 64] |= np.uint64(1) << np.uint64(on[0] % 64)  # (on a)
    neg[0, on[2] // 64] |= np.uint64(1) << np.uint64(on[2] % 64)  # (not (on c))
    dwords = max(1, (len(metadata["derived_pred"]) + 63) // 64)
    dpos, dneg = np.zeros((1, dwords), np.uint64), np.zeros((1, dwords), np.uint64)
    slot = derived[(0, 1, 0)]
    dpos[0, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)  # (trio a b a)
    slot = derived[(1, 2, 0)]
    dneg[0, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)  # (not (trio b c a))
    for layout in ("graph", "relation"):
        for include_static in (True, False):
            for include_derived in (True, False):
                encoder = encoding.FlatRelationEncoder(
                    task, layout=layout, include_static=include_static, include_derived=include_derived,
                    goal_derivations=("plain", "satisfied", "unsatisfied"),
                )
                for goals, expected in ((None, expected_goal), ((pos, neg, dpos, dneg), row_goal)):
                    batch = encoder.encode([task.initial_state], goals=goals)
                    got = {n: sorted(t) for n, (t, _) in cells(batch, 0).items()}
                    state_cells = {n: t for n, t in got.items() if "[g]" not in n}
                    assert {n: t for n, t in got.items() if "[g]" in n} == expected
                    assert state_cells.get("on") == [(0,)]
                    assert state_cells.get("link") == ([(0, 1, 2), (2, 1, 0)] if include_static else None)
                    assert state_cells.get("trio") == ([(0, 0, 0)] if include_derived else None)
                    assert batch.goal_complete.tolist() == [1]


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
    sizes = {"entities": len(exact.entity_role), "graphs": exact.num_graphs,
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
        assert entities == wanted_entities


def test_the_fingerprint_depends_on_the_targets():
    task = text_task("gripper__prob05")
    assert (
        encoding.EncodingSchema(task, target_sources=("goal",)).schema_fingerprint
        != encoding.EncodingSchema(task).schema_fingerprint
    )


GOAL_PROBLEMS = sorted(p.stem for p in GOALS.glob("*.pddl") if p.stem != "domain")
ALL_DERIVATIONS = ("plain", "satisfied", "unsatisfied")


def same_but_goal_complete(a, b):
    first, second = a.as_dict(), b.as_dict()
    assert first.keys() == second.keys()
    for key, value in first.items():
        if key != "goal_complete":
            assert np.array_equal(value, second[key]), key


def slotted(literal):
    atom = literal.atom
    return (atom.slot if atom.kind == "fluent" else atom.derived_slot if atom.kind == "derived" else None) is not None


def ids_of(task, text):
    """(table predicate id, objects padded to the widest predicate) of an atom given by name."""
    table = rl.as_table(task)
    atom = task.literal(text).atom
    objects = list(atom.object_indices)
    width = max(table.predicate_arities)
    return table.predicate_names.index(atom.predicate.name), objects + [-1] * (width - len(objects))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("problem", GOAL_PROBLEMS)
def test_a_row_goal_of_the_instance_goal_literals_encodes_as_the_instance_goal(problem, layout):
    task = goals_task(problem)
    states = walk(task, steps=6, walks=2)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, goal_derivations=ALL_DERIVATIONS,
                                           target_sources=("goal",))
    default = encoder.encode(states)
    literals = [str(x) for x in task.encoding_goal]
    whole = encoding.goal_masks(task, [literals], fixed="encode")
    width = 3 + max(rl.as_table(task).predicate_arities)
    assert whole.constants.dtype == np.int32 and whole.constants.shape[::2] == (1, width)
    batch = encoder.encode(states, goals=whole)
    same_but_goal_complete(default, batch)
    assert batch.goal_complete.tolist() == [1] * len(states)
    # the slotted literals of the goal, with the instance's constant part
    own = [str(x) for x in task.encoding_goal if slotted(x)]
    kept = encoding.goal_masks(task, [own], fixed="raise", instance_constants=True)
    # the instance's goal has no constant part for a literal without goal relations (equality)
    assert {tuple(x) for x in kept.constants[0]} <= {tuple(x) for x in whole.constants[0]}
    same_but_goal_complete(default, encoder.encode(states, goals=kept))


ROW_GOALS = [["(heavy t1)", "(lit a)"], ["(not (heavy t2))", "(road a b)", "(heavy t2)", "(lit t1)", "(lit b)"], [],
             ["(not (lit t1))", "(road b a)"], ["(not (road a a))"]]


def triples(task, texts):
    out = []
    for text in texts:
        ground = task.literal(text)
        out.append((ground.positive, ground.atom.predicate.name, tuple(ground.atom.object_indices)))
    return out


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_the_constant_literals_of_rows_are_encoded_by_their_truth(layout):
    task = goals_task("static_true")
    states = walk(task, steps=8, walks=2)
    rows = [ROW_GOALS[g % len(ROW_GOALS)] for g in range(len(states))]
    masks = encoding.goal_masks(task, rows, fixed="encode")
    assert masks.constants.shape[:2] == (len(states), 4)  # the second row: four literals of fixed truth
    assert masks.unsatisfiable.tolist()[:len(ROW_GOALS)] == [False, True, False, False, False]
    encoder = encoding.FlatRelationEncoder(task, layout=layout, goal_derivations=ALL_DERIVATIONS,
                                           target_sources=("goal",))
    batch = encoder.encode(states, goals=masks)
    check_goal_graphs(encoder, batch, states,
                      lambda g, s, names: goal_reference(task, s, triples(task, rows[g]), names), True)
    assert batch.goal_complete.tolist() == [1] * len(states)
    for graph, state in enumerate(states[:len(ROW_GOALS)]):
        alone = encoder.encode([state], goals=encoding.goal_masks(task, [rows[graph]], fixed="encode"))
        assert goal_cells(alone, 0, True)[:2] == goal_cells(batch, graph, True)[:2]
    many = states * 200
    rows = [ROW_GOALS[g % len(ROW_GOALS)] for g in range(len(many))]
    masks = encoding.goal_masks(task, rows, fixed="encode")
    serial = encoder.encode(many, goals=masks)
    for threads in (2, 3):
        same_but_goal_complete(serial, encoder.encode(many, goals=masks, threads=threads))


def test_rows_of_several_instances_carry_the_constant_literals_of_their_own_instance():
    first, second = goals_task("static_true"), goals_task("static_false")
    table = rl.TaskTable([first, second])
    encoder = encoding.FlatRelationEncoder(table, goal_derivations=ALL_DERIVATIONS)
    states = [first.initial_state, second.initial_state, first.initial_state]
    ids = [0, 1, 0]
    rows = [["(lit b)"], ["(lit a)"], []]
    masks = encoding.goal_masks(table, rows, ids, fixed="encode", instance_constants=True)
    batch = encoder.encode(states, ids, goals=masks)
    names = set(batch.schema.relation_names)
    for graph, task in enumerate([first, second, first]):
        literals = [str(x) for x in task.encoding_goal if not slotted(x)] + rows[graph]
        assert goal_cells(batch, graph, False)[0] == goal_reference(task, states[graph], triples(task, literals), names)


def test_a_stream_of_rows_with_and_without_constant_literals_flushes_as_one_encode():
    task = goals_task("static_true")
    states = walk(task, steps=6, walks=2)
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=ALL_DERIVATIONS, target_sources=("goal",))
    rows = [ROW_GOALS[g % len(ROW_GOALS)] for g in range(len(states))]
    expected = encoder.encode(states, goals=encoding.goal_masks(task, rows, fixed="encode"))
    stream = encoder.stream()
    for begin in range(0, len(states), 3):
        stream.append(states[begin:begin + 3], goals=encoding.goal_masks(task, rows[begin:begin + 3], fixed="encode"))
    same_but_goal_complete(expected, stream.flush())
    mutable = encoder.mutable_stream()
    for index, state in enumerate(states):
        mutable.append([index], [state], goals=encoding.goal_masks(task, [ROW_GOALS[1]], fixed="encode"))
    for index, state in enumerate(states):
        mutable.update(index, state, goals=encoding.goal_masks(task, [rows[index]], fixed="encode"))
    same_but_goal_complete(expected, mutable.flush())
    mutable.remove(1)
    rest = [g for g in range(len(states)) if g != 1]
    alone = encoder.encode([states[g] for g in rest],
                           goals=encoding.goal_masks(task, [rows[g] for g in rest], fixed="encode"))
    same_but_goal_complete(alone, mutable.flush())


def constant_masks(task, constants):
    table = rl.as_table(task)
    zeros = np.zeros((1, table.words), np.uint64)
    return encoding.GoalMasks(zeros, zeros, None, None, np.zeros(1, np.uint8),
                              constants=np.asarray(constants, np.int32).reshape(1, len(constants), -1))


def test_invalid_constant_literals_are_refused():
    task = goals_task("static_true")
    encoder = encoding.FlatRelationEncoder(task)
    state = task.initial_state
    (heavy, t1), (_, t2), (lit, a) = ids_of(task, "(heavy t1)"), ids_of(task, "(heavy t2)"), ids_of(task, "(lit a)")
    good = [[0, heavy, 0, *t1], [0, heavy, 0, *t2]]
    encoder.encode([state], goals=constant_masks(task, good))
    cases = {
        "is not after the one before it": good[::-1],
        "has the layer 1, outside": [[1, heavy, 0, *t1]],
        "has the sign 2": [[0, heavy, 2, *t1]],
        "is invalid": [[0, heavy, 0, t1[0], 0, -1]],
        "has a slot": [[0, lit, 0, *a]],
    }
    for message, constants in cases.items():
        with pytest.raises(ValueError, match=f"^mymyr: encoding: the goal of row 0: constant literal [01] .*{message}"):
            encoder.encode([state], goals=constant_masks(task, constants))
    width = 3 + max(rl.as_table(task).predicate_arities)
    with pytest.raises(ValueError, match=f"have 2 columns, at least {width} are needed"):
        encoder.encode([state], goals=constant_masks(task, [[0, heavy]]))
    # the rows after a row's literals are padding: all -1
    end = [-1] * len(good[0])
    encoder.encode([state], goals=constant_masks(task, [good[0], end, end]))
    for constants in ([good[0], end, good[1]], [good[0], [-1, -1, -1, *t1]]):
        with pytest.raises(ValueError, match="constant literal 1 ends the literals .* not all -1"):
            encoder.encode([state], goals=constant_masks(task, constants))


def test_constant_literals_narrower_than_the_widest_predicate_are_refused_before_they_are_read():
    task = goals_task("static_true")
    table = rl.as_table(task)
    # a binary predicate whose objects lie past a row of three columns, with variants that test its truth
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied"))
    narrow = constant_masks(task, [[0, table.predicate_names.index("road"), 0]])
    message = f"have 3 columns, at least {3 + max(table.predicate_arities)} are needed"
    state = task.initial_state
    with pytest.raises(ValueError, match=message):
        encoder.encode([state], goals=narrow)
    with pytest.raises(ValueError, match=message):
        encoder.stream().append([state], goals=narrow)
    with pytest.raises(ValueError, match=message):
        encoder.mutable_stream().append([0], [state], goals=narrow)


def test_a_failing_constant_part_of_the_instance_makes_the_goals_that_keep_it_unsatisfiable():
    task = goals_task("static_false")  # (heavy t2) is static and false
    assert task.encoding_goal_unsatisfiable
    for fixed in ("decide", "encode", "raise"):
        kept = encoding.goal_masks(task, [[], ["(lit a)"]], fixed=fixed, instance_constants=True)
        assert kept.unsatisfiable.tolist() == [True, True]
        assert not encoding.goal_masks(task, [[], ["(lit a)"]], fixed=fixed).unsatisfiable.any()
    holds = encoding.goal_masks(goals_task("static_true"), [[]], instance_constants=True)
    assert len(holds.constants[0]) and not holds.unsatisfiable.any()
