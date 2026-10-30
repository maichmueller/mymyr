"""Literals and the masks they make: literal_slots (the kind and slot of each literal), goal_masks, subgoal_masks and
history_masks from lists of literals and literal_masks from arrays, with the literals of fixed truth decided or
refused, on single tasks and on tables of several instances, and the masks accepted by encode."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task

GOALS = ROOT / "tests/data/pddl/goals"
Kind = encoding.LiteralKind


def goals_task(problem, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task.from_pddl(GOALS / "domain.pddl", GOALS / f"{problem}.pddl", **options)


def predicate_id(source, name):
    table = rl.as_table(source)
    return table.predicate_names.index(name)


def ids(task, literal, width=3):
    """(table predicate id, objects padded to `width`) of an atom given by name."""
    atom = task.literal(literal).atom
    objects = list(atom.object_indices)
    return predicate_id(task, atom.predicate.name), objects + [-1] * (width - len(objects))


def reference_masks(task, rows, words, derived_words):
    """The masks of rows of fluent and derived literals, from the slots Task.literal gives."""
    pos = np.zeros((len(rows), words), dtype=np.uint64)
    neg = np.zeros_like(pos)
    derived_pos = np.zeros((len(rows), derived_words), dtype=np.uint64)
    derived_neg = np.zeros_like(derived_pos)
    for r, row in enumerate(rows):
        for literal in row:
            ground = task.literal(literal)
            atom = ground.atom
            derived = atom.kind == "derived"
            slot = atom.derived_slot if derived else atom.slot
            target = (derived_pos if ground.positive else derived_neg) if derived else (pos if ground.positive else neg)
            target[r, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return pos, neg, derived_pos, derived_neg


@pytest.mark.parametrize("atoms", ["reachable", "dense"])
def test_every_kind_resolves_by_its_predicate_and_objects(atoms):
    task = goals_task("static_true", atoms=atoms)
    cases = {
        "(lit a)": Kind.FLUENT,
        "(done)": Kind.DERIVED,
        "(heavy t1)": Kind.STATIC_TRUE,
        "(heavy t2)": Kind.STATIC_FALSE,
        "(road a b)": Kind.STATIC_TRUE,
        "(road a a)": Kind.STATIC_FALSE,
        "(= a a)": Kind.STATIC_TRUE,
        "(= a b)": Kind.STATIC_FALSE,
        "(lit t1)": Kind.UNREACHABLE,  # lit takes a place: no state has the atom
    }
    rows = [ids(task, literal) for literal in cases]
    kind, slot = encoding.literal_slots(task, [p for p, _ in rows], [o for _, o in rows])
    assert kind.dtype == np.uint8 and slot.dtype == np.int32
    assert [Kind(k) for k in kind] == list(cases.values())
    for (literal, expected), s in zip(cases.items(), slot.tolist()):
        atom = task.literal(literal).atom
        want = atom.slot if expected == Kind.FLUENT else atom.derived_slot if expected == Kind.DERIVED else -1
        assert s == want, literal


ROADS_DOMAIN = """(define (domain roads)
  (:predicates (at ?p) (road ?p ?q))
  (:action move :parameters (?p ?q) :precondition (and (at ?p) (road ?p ?q)) :effect (and (at ?q) (not (at ?p)))))"""
ROADS_PROBLEM = """(define (problem roads) (:domain roads) (:objects a b c)
  (:init (at a) (road a b) (road b a)) (:goal (at b)))"""


def test_an_atom_that_no_state_reaches_has_no_slot_in_the_reachable_layout_only(tmp_path):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    (tmp_path / "domain.pddl").write_text(ROADS_DOMAIN)
    (tmp_path / "problem.pddl").write_text(ROADS_PROBLEM)
    task, dense = (mymyr.Task.from_pddl(tmp_path / "domain.pddl", tmp_path / "problem.pddl", atoms=atoms)
                   for atoms in ("reachable", "dense"))
    assert task.literal("(at c)").atom.slot is None and dense.literal("(at c)").atom.slot is not None
    predicate, objects = ids(task, "(at c)", 1)
    assert encoding.literal_slots(task, [predicate], [objects])[0].tolist() == [Kind.UNREACHABLE]
    kind, slot = encoding.literal_slots(dense, [predicate], [objects])
    assert kind.tolist() == [Kind.FLUENT] and slot.tolist() == [dense.literal("(at c)").atom.slot]
    with pytest.raises(ValueError, match="no slot"):
        encoding.goal_masks(task, [["(at c)"]], fixed="raise")
    masks = encoding.goal_masks(task, [["(at c)", "(at b)"], ["(not (at c))"]])
    assert masks.unsatisfiable.tolist() == [True, False]
    assert masks.pos[0].any() and not masks.pos[1].any() and not masks.neg.any()
    slot = dense.literal("(at c)").atom.slot
    assert encoding.goal_masks(dense, [["(not (at c))"]]).neg[0, slot // 64] == np.uint64(1) << np.uint64(slot % 64)


def test_invalid_literals_are_a_kind_of_the_lookup_and_an_error_of_the_masks():
    task = goals_task("static_true")
    at = predicate_id(task, "at")
    objects = task.num_objects
    cases = [
        (at, [0, 1, -1], 0),  # valid
        (-1, [0, 1, -1], 0),
        (len(rl.as_table(task).predicate_names), [0, 1, -1], 0),
        (at, [-1, 1, -1], 0),
        (at, [0, objects, -1], 0),
        (at, [0, 1, 0], 0),  # padding that is not -1
        (at, [0, 1, -1], 1),  # a task id outside the table
    ]
    kind, slot = encoding.literal_slots(task, [c[0] for c in cases], [c[1] for c in cases], [c[2] for c in cases])
    assert kind[0] != Kind.INVALID
    assert set(kind[1:].tolist()) == {Kind.INVALID} and set(slot[1:].tolist()) == {-1}
    # a row of objects narrower than the predicate's arity
    assert encoding.literal_slots(task, [at], [[0]])[0].tolist() == [Kind.INVALID]
    for predicate, row, _ in cases[1:-1]:
        with pytest.raises(ValueError, match="invalid"):
            encoding.literal_masks(task, [predicate], [row], [True], [0], 1)
    with pytest.raises(ValueError, match="row"):
        encoding.literal_masks(task, [at], [[0, 1, -1]], [True], [1], 1)
    with pytest.raises(ValueError, match="layer"):
        encoding.literal_masks(task, [at], [[0, 1, -1]], [True], [0], 1, layers=[2], num_layers=2)
    with pytest.raises(ValueError, match="task_ids"):
        encoding.literal_masks(task, [at], [[0, 1, -1]], [True], [0], 1, task_ids=[1])
    with pytest.raises(ValueError, match="fixed"):
        encoding.goal_masks(task, [["(lit a)"]], fixed="keep")
    with pytest.raises(ValueError, match="variables"):
        encoding.goal_masks(task, [["(lit ?p)"]])
    with pytest.raises(KeyError, match="nowhere"):
        encoding.goal_masks(task, [["(lit nowhere)"]])
    with pytest.raises(TypeError):
        encoding.goal_masks(task, "(lit a)")
    with pytest.raises(TypeError):
        encoding.goal_masks(task, [["(lit a)"], "(lit b)"])
    with pytest.raises(TypeError):
        encoding.literal_slots(rl.as_table(task).predicate_names, [at], [[0, 1]])


def test_decide_leaves_out_what_holds_and_flags_what_fails_where_raise_refuses():
    task = goals_task("static_true")
    rows = [["(heavy t1)", "(not (heavy t2))", "(lit a)"], ["(heavy t2)", "(lit a)"], ["(not (heavy t1))"], []]
    masks = encoding.goal_masks(task, rows)
    assert isinstance(masks, encoding.GoalMasks)
    assert masks.unsatisfiable.dtype == np.bool_ and masks.unsatisfiable.tolist() == [False, True, True, False]
    table = rl.as_table(task)
    pos, neg, derived_pos, derived_neg = reference_masks(task, [["(lit a)"], ["(lit a)"], [], []], table.words,
                                                         table.derived_words)
    assert np.array_equal(masks.pos, pos) and np.array_equal(masks.neg, neg)
    assert np.array_equal(masks.derived_pos, derived_pos) and np.array_equal(masks.derived_neg, derived_neg)
    for literal in ("(heavy t1)", "(not (heavy t2))"):
        with pytest.raises(ValueError, match="static"):
            encoding.goal_masks(task, [[literal]], fixed="raise")
        with pytest.raises(ValueError, match="static"):
            encoding.subgoal_masks(task, [[[literal]]])
        with pytest.raises(ValueError, match="static"):
            encoding.history_masks(task, [[(0, [literal])]])
    layers = encoding.subgoal_masks(task, [[["(heavy t2)"], ["(heavy t1)", "(lit b)"]], [[]]], fixed="decide")
    assert isinstance(layers, encoding.LayerMasks)
    assert layers.unsatisfiable.tolist() == [[True, False], [False, False]]
    assert layers.pos.shape == (2, 2, table.words) and layers.pos[0, 1].any() and not layers.pos[0, 0].any()
    assert masks.constants is None and layers.constants is None


def test_encode_keeps_the_literals_of_fixed_truth_as_sorted_constant_literals():
    task = goals_task("static_true")
    width = 3 + max(rl.as_table(task).predicate_arities)
    rows = [["(lit a)", "(road b a)", "(not (heavy t2))", "(heavy t1)", "(lit t1)"], [], ["(heavy t2)", "(heavy t2)"]]
    masks = encoding.goal_masks(task, rows, fixed="encode")
    decided = encoding.goal_masks(task, rows)
    assert np.array_equal(masks.pos, decided.pos) and np.array_equal(masks.unsatisfiable, decided.unsatisfiable)
    assert masks.constants.dtype == np.int32 and masks.constants.shape == (3, 4, width)

    def literal(text, layer=0):
        ground = task.literal(text)
        predicate, objects = ids(task, str(ground.atom), width - 3)
        return [layer, predicate, int(not ground.positive), *objects]

    first = sorted(literal(x) for x in rows[0][1:])
    assert masks.constants[0].tolist() == first
    assert masks.constants[1].tolist() == [[-1] * width] * 4
    assert masks.constants[2].tolist() == [literal("(heavy t2)")] + [[-1] * width] * 3  # duplicates are merged
    assert not encoding.goal_masks(task, [[]], fixed="encode").constants.size
    # the instance's constant part: (heavy t1) and (not (heavy t2))
    own = encoding.goal_masks(task, [["(lit a)"], ["(heavy t1)"]], fixed="encode", instance_constants=True)
    part = sorted([literal("(heavy t1)"), literal("(not (heavy t2))")])
    assert own.constants.tolist() == [part, part]
    assert encoding.goal_masks(task, [[]], instance_constants=True).constants.tolist() == [part]
    layers = encoding.subgoal_masks(task, [[["(heavy t1)"], ["(lit a)", "(road a b)", "(heavy t1)"]]], fixed="encode")
    assert layers.constants.tolist() == [sorted([literal("(heavy t1)", 0), literal("(heavy t1)", 1),
                                                 literal("(road a b)", 1)])]
    predicates, objects = zip(*(ids(task, x, width - 3) for x in ("(heavy t1)", "(road a b)", "(lit a)")))
    arrays = encoding.literal_masks(task, predicates, objects, [True, False, True], [1, 0, 0], 2, fixed="encode")
    assert arrays.constants.tolist() == [[literal("(not (road a b))")], [literal("(heavy t1)")]]
    with pytest.raises(ValueError, match="instance_constants go with goals"):
        encoding.literal_masks(task, predicates, objects, [True] * 3, [0] * 3, 1, layers=[0] * 3,
                               instance_constants=True)


def test_derived_literals_set_the_derived_masks():
    task = text_task("philosophers__p03-phil4")
    derived = task.encoding_goal[0].atom
    assert derived.kind == "derived"
    fluent = next(iter(task.initial_state.atoms()))
    masks = encoding.goal_masks(task, [[str(derived), str(fluent)], [f"(not {derived})"]])
    table = rl.as_table(task)
    assert table.derived_words > 0 and len(masks) == 4
    expected = reference_masks(task, [[str(derived), str(fluent)], [f"(not {derived})"]], table.words,
                               table.derived_words)
    for got, want in zip(masks, expected):
        assert np.array_equal(got, want)
    assert masks.derived_pos[0].any() and masks.derived_neg[1].any() and not masks.unsatisfiable.any()


@pytest.mark.parametrize("helper", [encoding.goal_masks, encoding.subgoal_masks, encoding.history_masks])
def test_problem_local_derived_literals_have_slots_but_no_encoding_relation(helper):
    task = goals_task("duplicate")
    atom = task.literal("(axiom_0)").atom
    assert atom.kind == "derived" and atom.derived_slot is not None
    rows = [["(axiom_0)"]] if helper is encoding.goal_masks else (
        [[["(axiom_0)"]]] if helper is encoding.subgoal_masks else [[(0, ["(axiom_0)"])]])
    with pytest.raises(ValueError, match="derived predicate of its problem alone.*no relation"):
        helper(task, rows)


@pytest.mark.parametrize("name", ["gripper__prob05", "philosophers__p03-phil4"])
@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_mask_tuples_preserve_their_arrays_and_flags_when_pickled(name, layered, protocol):
    task = text_task(name)
    literal = str(next(iter(task.initial_state.atoms())))
    masks = (encoding.subgoal_masks(task, [[[literal]], []]) if layered else
             encoding.goal_masks(task, [[literal], []]))
    masks.unsatisfiable[0] = True
    assert isinstance(masks, tuple)
    assert masks.pos is masks[0] and masks.neg is masks[1] and masks[:] == tuple(masks)
    restored = pickle.loads(pickle.dumps(masks, protocol=protocol))
    assert type(restored) is type(masks)
    for got, want in zip((*restored, restored.unsatisfiable), (*masks, masks.unsatisfiable), strict=True):
        np.testing.assert_array_equal(got, want)
    assert restored.unsatisfiable.dtype == np.bool_ and restored.constants is None
    for name in ("pos", "neg", "derived_pos", "derived_neg", "unsatisfiable", "constants"):
        with pytest.raises(AttributeError):
            setattr(masks, name, None)
    static = next((str(a) for a in task.formalism.static_init), None)
    if static is not None:
        kept = (encoding.subgoal_masks(task, [[[literal, static]], []], fixed="encode") if layered else
                encoding.goal_masks(task, [[literal, static], []], fixed="encode"))
        restored = pickle.loads(pickle.dumps(kept, protocol=protocol))
        assert restored.constants.shape[:2] == (2, 1)
        np.testing.assert_array_equal(restored.constants, kept.constants)
        assert "constants (2, 1, " in repr(restored)


def test_goal_masks_equal_the_first_subgoal_layer_and_the_slots_of_task_literal():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, max_goal_level=1, goal_derivations=("plain",))
    atoms = [str(a) for a in task.initial_state.atoms()]
    rows = [atoms[:3], [f"(not {atoms[0]})", atoms[4]], []]
    goals = encoding.goal_masks(encoder, rows)
    layers = encoding.subgoal_masks(encoder, [[row] for row in rows])
    assert len(goals) == len(layers) == 2
    assert np.array_equal(goals.pos, layers.pos[:, 0]) and np.array_equal(goals.neg, layers.neg[:, 0])
    pos, neg, _, _ = reference_masks(task, rows, task.words, 0)
    assert np.array_equal(goals.pos, pos) and np.array_equal(goals.neg, neg)
    # the same masks from an encoder, its table and its task
    for source in (rl.as_table(task), task):
        other = encoding.goal_masks(source, rows)
        assert np.array_equal(other.pos, goals.pos) and np.array_equal(other.neg, goals.neg)


def test_literal_masks_are_the_goal_and_layer_masks_of_arrays():
    task = goals_task("static_true")
    # (atom, positive, row, layer)
    atoms = [("(lit a)", True, 0, 0), ("(ok t1)", False, 0, 1), ("(heavy t2)", True, 1, 0), ("(done)", True, 2, 1),
             ("(at t1 a)", True, 2, 1)]
    rows = [ids(task, atom) for atom, _, _, _ in atoms]
    predicates = [p for p, _ in rows]
    objects = [o for _, o in rows]
    positive = [sign for _, sign, _, _ in atoms]
    row_of = [r for _, _, r, _ in atoms]
    layer_of = [layer for _, _, _, layer in atoms]
    literals = [(atom if sign else f"(not {atom})", r, layer) for atom, sign, r, layer in atoms]
    goals = encoding.literal_masks(task, predicates, objects, positive, row_of, 3)
    named = encoding.goal_masks(task, [[t for t, r, _ in literals if r == row] for row in range(3)])
    for got, want in zip((*goals, goals.unsatisfiable), (*named, named.unsatisfiable)):
        assert np.array_equal(got, want)
    layered = encoding.literal_masks(task, predicates, objects, positive, row_of, 3, layers=layer_of)
    named = encoding.subgoal_masks(task, [[[t for t, r, layer in literals if r == row and layer == k] for k in range(2)]
                                          for row in range(3)], fixed="decide")
    assert isinstance(layered, encoding.LayerMasks) and layered.pos.shape[:2] == (3, 2)
    for got, want in zip((*layered, layered.unsatisfiable), (*named, named.unsatisfiable)):
        assert np.array_equal(got, want)
    with pytest.raises(ValueError, match="static"):
        encoding.literal_masks(task, predicates, objects, positive, row_of, 3, fixed="raise")
    empty = encoding.literal_masks(task, [], np.zeros((0, 2), dtype=np.int32), [], [], 2)
    assert empty.pos.shape == (2, task.words) and not empty.unsatisfiable.any()


def test_a_table_of_several_instances_resolves_each_row_in_its_own_instance():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    tasks = [goals_task("static_true"), goals_task("static_false")]
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table, goal_derivations=("plain", "satisfied"))
    # (road a b) holds in the first instance only
    rows = [["(road a b)", "(lit a)"], ["(road a b)", "(lit a)"], ["(lit b)"]]
    task_ids = np.array([0, 1, 1], dtype=np.int32)
    masks = encoding.goal_masks(encoder, rows, task_ids)
    expected = [s for s in (tasks[0].initial_state, tasks[1].initial_state)]
    assert masks.unsatisfiable.tolist() == [
        not tasks[0].literal("(road a b)").holds(expected[0]), not tasks[1].literal("(road a b)").holds(expected[1]),
        False]
    for row, (task_id, literals) in enumerate(zip(task_ids, rows)):
        task = tasks[task_id]
        pos, neg, _, _ = reference_masks(task, [[x for x in literals if x != "(road a b)"]], table.words,
                                         table.derived_words)
        assert np.array_equal(masks.pos[row], pos[0]) and np.array_equal(masks.neg[row], neg[0])
    with pytest.raises(ValueError, match="task_ids"):
        encoding.goal_masks(encoder, rows)
    with pytest.raises(ValueError, match="task_ids"):
        encoding.goal_masks(encoder, rows, [0, 2, 1])
    with pytest.raises(ValueError, match="task_ids"):
        encoding.goal_masks(encoder, rows, [0, 1])
    lit = predicate_id(table, "lit")
    kind, slot = encoding.literal_slots(table, [lit, lit], [[0], [0]], task_ids=[0, 1])
    assert kind.tolist() == [Kind.FLUENT, Kind.FLUENT]
    assert slot.tolist() == [t.literal(("lit", t.objects[0])).atom.slot for t in tasks]
    with pytest.raises(ValueError, match="task_ids"):
        encoding.literal_slots(table, [lit], [[0]])
    states = [tasks[i].initial_state for i in task_ids]
    batch = encoder.encode(states, task_ids, goals=masks)
    tuple_batch = encoder.encode(states, task_ids, goals=tuple(masks))
    for name, value in batch.as_dict().items():
        assert np.array_equal(value, tuple_batch.as_dict()[name]), name


def test_goal_masks_of_the_instance_goal_encode_as_the_instance_goal():
    task = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task, goal_derivations=("plain", "satisfied", "unsatisfied"))
    goal = [str(literal) for literal in task.encoding_goal]
    masks = encoding.goal_masks(encoder, [goal])
    assert not masks.unsatisfiable.any()
    state = task.initial_state
    own = encoder.encode([state])
    given = encoder.encode([state], goals=masks)
    for name, value in own.as_dict().items():
        assert np.array_equal(value, given.as_dict()[name]), name
    shared = encoder.encode([state, state], goals=encoding.goal_masks(encoder, [goal, goal]))
    assert shared.num_graphs == 2


def test_history_masks_build_their_steps_through_the_same_lookup():
    task = text_task("gripper__prob05")
    atoms = [str(a) for a in task.initial_state.atoms()]
    pos, neg, dt, count = encoding.history_masks(task, [[(-2, [atoms[0]]), (-1, [f"(not {atoms[1]})"])], []])
    assert pos.shape == neg.shape == (2, 2, task.words)
    assert dt.tolist() == [[-2, -1], [0, 0]] and count.tolist() == [2, 0]
    expected_pos, _, _, _ = reference_masks(task, [[atoms[0]]], task.words, 0)
    _, expected_neg, _, _ = reference_masks(task, [[f"(not {atoms[1]})"]], task.words, 0)
    assert np.array_equal(pos[0, 0], expected_pos[0]) and np.array_equal(neg[0, 1], expected_neg[0])
    assert not pos[1].any() and not neg[1].any()
