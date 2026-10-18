"""Subgoal layers and history steps of an encoding: the masks built from literals, the entities and tuples they add,
the level subtraction, the truncation to the newest steps, single rows against batches against streams, capacities,
names, options, pickling and the errors."""

import pickle
import random

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk

BLOCKS = "blocks__probBLOCKS-8-0"
SUBGOAL_ROLE, HISTORY_ROLE, HISTORY_TARGET_ROLE = 3, 5, 6


def literal(atom, positive=True):
    text = f"({' '.join([atom.predicate.name, *(o.name for o in atom.objects)])})"
    return text if positive else f"(not {text})"


def signed(atom, positive=True):
    return f"[{'+' if positive else '-'}]{literal(atom)}"


def rows_of(rng, states, levels, steps):
    """Random subgoal layers and histories for ``states`` as lists of literal strings."""
    every = [a for s in states for a in s.atoms()]
    layers = [[[literal(a, rng.random() < 0.7) for a in rng.sample(every, rng.randrange(0, 5))]
               for _ in range(rng.randrange(0, levels + 1))] for _ in states]
    history = [[(-(steps - i), [literal(a, rng.random() < 0.7) for a in rng.sample(every, rng.randrange(0, 4))])
                for i in range(rng.randrange(0, steps + 1))] for _ in states]
    return layers, history


def make(task, **options):
    options = {"max_goal_level": 2, "history": True, "goal_derivations": ("plain", "satisfied"),
               "target_sources": ("goal", "subgoal", "history"), **options}
    return encoding.FlatRelationEncoder(task, **options)


def same(a, b):
    first, second = a.as_dict(), b.as_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(first[key], second[key]), key


def names_of(encoder, batch, role):
    out = []
    for graph in range(batch.num_graphs):
        begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
        names = encoder.entity_names(batch, graph)
        out.append([n for n, r in zip(names, batch.entity_role[begin:end].tolist()) if r == role])
    return out


def test_a_literal_is_encoded_once_at_its_lowest_level():
    task = text_task(BLOCKS)
    state = task.initial_state
    atoms = list(state.atoms())
    goal = {(x.positive, x.predicate.name, tuple(task.objects[o.index] for o in x.objects)) for x in task.encoding_goal}
    free = [a for a in atoms if (True, a.predicate.name, tuple(o.name for o in a.objects)) not in goal]
    a, b, c = free[:3]
    encoder = make(task, history=False, target_sources=("goal", "subgoal"))
    layers = [[[literal(a), literal(b)], [literal(b), literal(c), literal(a, False)]]]
    batch = encoder.encode([state], subgoals=encoding.subgoal_masks(encoder, layers))
    first = {f"{signed(x)}[sg]" for x in (a, b)}
    second = {f"{signed(c)}[ssg]", f"[-]{literal(a)}[ssg]"}
    assert sorted(names_of(encoder, batch, SUBGOAL_ROLE)[0]) == sorted(first | second)


def test_a_goal_literal_in_a_layer_is_not_encoded_again():
    task = text_task(BLOCKS)
    state = task.initial_state
    goal_atom = next(x for x in task.encoding_goal if x.positive)
    name = literal(goal_atom)
    encoder = make(task, history=False, target_sources=("goal", "subgoal"))
    other = next(a for a in state.atoms() if literal(a) != name)
    layers = [[[name, literal(other)]]]
    batch = encoder.encode([state], subgoals=encoding.subgoal_masks(encoder, layers))
    assert names_of(encoder, batch, SUBGOAL_ROLE)[0] == [f"{signed(other)}[sg]"]


def test_only_the_newest_steps_of_a_history_are_encoded():
    task = text_task(BLOCKS)
    state = task.initial_state
    atoms = list(state.atoms())
    steps = [(-3, [literal(atoms[0])]), (-2, [literal(atoms[1])]), (-1, [literal(atoms[2], False)]),
             (0, [literal(atoms[3])])]
    encoder = make(task, history_max_steps=2)
    batch = encoder.encode([state], history=encoding.history_masks(encoder, [steps]))
    assert batch.history_offsets.tolist() == [0, 2]
    assert batch.history_dt.tolist() == [-1, 0]
    assert names_of(encoder, batch, HISTORY_ROLE) == [["history:-1#0", "history:0#1"]]
    assert [sorted(n) for n in names_of(encoder, batch, HISTORY_TARGET_ROLE)] == [
        sorted([f"history:-1#0:{signed(atoms[2], False)}", f"history:0#1:{signed(atoms[3])}"])]
    unlimited = make(task)
    full = unlimited.encode([state], history=encoding.history_masks(unlimited, [steps]))
    assert full.history_dt.tolist() == [-3, -2, -1, 0]


def test_history_tuples_hold_their_target_step_and_objects():
    task = text_task(BLOCKS)
    state = task.initial_state
    on = next(a for a in state.atoms() if a.predicate.name == "on")
    encoder = make(task)
    steps = [(-1, [literal(on)]), (0, [literal(on, False)])]
    batch = encoder.encode([state], history=encoding.history_masks(encoder, [steps]))
    relations = encoder.describe(batch, 0).relations
    objects = tuple(o.name for o in on.objects)
    assert relations["[+]on[hist]"] == ((f"history:-1#0:{signed(on)}", "history:-1#0", *objects),)
    assert relations["[-]on[hist]"] == ((f"history:0#1:{signed(on, False)}", "history:0#1", *objects),)


def test_a_row_without_layers_or_steps_adds_nothing():
    task = text_task(BLOCKS)
    states = walk(task, steps=3, walks=1)
    encoder = make(task)
    batch = encoder.encode(states, subgoals=encoding.subgoal_masks(encoder, [[] for _ in states]),
                           history=encoding.history_masks(encoder, [[] for _ in states]))
    assert batch.history_offsets.tolist() == [0] * (len(states) + 1)
    for graph in range(len(states)):
        assert not names_of(encoder, batch, HISTORY_ROLE)[graph]
        assert not names_of(encoder, batch, SUBGOAL_ROLE)[graph]


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_single_rows_batches_and_streams_agree(layout):
    task = text_task(BLOCKS)
    encoder = make(task, layout=layout)
    states = walk(task, steps=10, walks=3)
    rng = random.Random(7)
    layers, history = rows_of(rng, states, 2, 4)
    sg = encoding.subgoal_masks(encoder, layers)
    hist = encoding.history_masks(encoder, history)
    batch = encoder.encode(states, subgoals=sg, history=hist)
    for graph, state in enumerate(states):
        one = encoder.encode([state], subgoals=encoding.subgoal_masks(encoder, [layers[graph]]),
                             history=encoding.history_masks(encoder, [history[graph]]))
        assert encoder.describe(one, 0).relations == encoder.describe(batch, graph).relations
        assert encoder.describe(one, 0).entities == encoder.describe(batch, graph).entities
    stream = encoder.stream()
    mutable = encoder.mutable_stream()
    cuts = [0, 3, 4, 11, len(states)]
    for begin, end in zip(cuts, cuts[1:]):
        chunk = {"subgoals": tuple(x[begin:end] for x in sg), "history": tuple(x[begin:end] for x in hist)}
        stream.append(states[begin:end], **chunk)
        mutable.append(list(range(begin, end)), states[begin:end], **chunk)
    same(stream.flush(), batch)
    same(mutable.flush(), batch)


def test_streams_pad_rows_with_fewer_layers_and_steps():
    task = text_task(BLOCKS)
    encoder = make(task)
    states = walk(task, steps=4, walks=1)[:4]
    atoms = list(states[0].atoms())
    layers = [[[literal(atoms[0])]], [[literal(atoms[1])], [literal(atoms[2])]], [], [[literal(atoms[3])]]]
    history = [[(-1, [literal(atoms[0])])], [], [(-2, [literal(atoms[1])]), (-1, [literal(atoms[2])])], []]
    expected = encoder.encode(states, subgoals=encoding.subgoal_masks(encoder, layers),
                              history=encoding.history_masks(encoder, history))
    stream = encoder.stream()
    for i, state in enumerate(states):
        stream.append([state], subgoals=encoding.subgoal_masks(encoder, [layers[i]]),
                      history=encoding.history_masks(encoder, [history[i]]))
    same(stream.flush(), expected)


def test_capacities_take_a_history_count():
    task = text_task(BLOCKS)
    encoder = make(task)
    states = walk(task, steps=4, walks=1)
    layers, history = rows_of(random.Random(3), states, 2, 3)
    args = dict(subgoals=encoding.subgoal_masks(encoder, layers), history=encoding.history_masks(encoder, history))
    exact = encoder.encode(states, **args)
    roomy = dict(entities=4096, tuples=4096, args=16384, targets=1024, history=64)
    sizes = encoder.encode(states, capacities=roomy, **args).sizes
    assert sizes["history"] == len(exact.history_entity) > 0 and not sizes["overflow"]
    capacities = {k: sizes[k] for k in roomy}
    fitted = encoder.encode(states, capacities=capacities, **args)
    assert fitted.sizes["history"] == sizes["history"] and not fitted.sizes["overflow"]
    assert fitted.history_entity[: sizes["history"]].tolist() == exact.history_entity.tolist()
    short = encoder.encode(states, capacities={**capacities, "history": sizes["history"] - 1}, **args)
    assert short.sizes["overflow"]
    with pytest.raises(ValueError, match="history"):
        encoder.encode(states, capacities={k: v for k, v in capacities.items() if k != "history"}, **args)


def test_the_arrays_of_a_batch_without_history_have_no_history():
    task = text_task(BLOCKS)
    encoder = make(task, history=False, target_sources=("goal", "subgoal"))
    batch = encoder.encode(walk(task, steps=2, walks=1))
    for name in ("history_offsets", "history_entity", "history_dt"):
        with pytest.raises(AttributeError, match=name):
            getattr(batch, name)
        assert name not in batch.as_dict()


def test_target_groups_and_entity_roles_of_a_layered_batch():
    task = text_task(BLOCKS)
    encoder = make(task)
    state = task.initial_state
    atoms = list(state.atoms())
    batch = encoder.encode([state], subgoals=encoding.subgoal_masks(encoder, [[[literal(atoms[0])]]]),
                           history=encoding.history_masks(encoder, [[(-1, [literal(atoms[1]), literal(atoms[2])])]]))
    groups = batch.target_group.tolist()
    assert groups == sorted(groups)
    assert set(groups) == {0, 1, 3}
    assert groups.count(3) == 2
    roles = batch.entity_role.tolist()
    assert roles == sorted(roles, key=lambda r: [0, 8, 2, 3, 4, 5, 6, 1].index(r))


def test_subgoal_and_history_masks_have_the_shape_of_their_rows():
    task = text_task(BLOCKS)
    encoder = make(task)
    atoms = list(task.initial_state.atoms())
    pos, neg = encoding.subgoal_masks(encoder, [[[literal(atoms[0])], [literal(atoms[1], False)]], []])
    assert pos.shape == neg.shape == (2, 2, task.words) and pos.dtype == np.uint64
    assert not pos[1].any() and not pos[0, 1].any() and not neg[0, 0].any() and neg[0, 1].any()
    pos, neg, dt, count = encoding.history_masks(encoder, [[(-2, [literal(atoms[0])]), (-1, [])], []])
    assert pos.shape == neg.shape == (2, 2, task.words)
    assert dt.dtype == np.int32 and count.dtype == np.int32
    assert dt.tolist() == [[-2, -1], [0, 0]] and count.tolist() == [2, 0]


def test_literals_without_a_slot_are_refused():
    gripper = text_task("gripper__prob05")
    encoder = make(gripper, target_sources=("goal",))
    static = next(iter(gripper.formalism.static_init))
    with pytest.raises(ValueError, match="static"):
        encoding.subgoal_masks(encoder, [[[literal(static)]]])
    with pytest.raises(ValueError, match="static"):
        encoding.history_masks(encoder, [[(0, [literal(static)])]])
    with pytest.raises(ValueError, match="variables"):
        encoding.subgoal_masks(encoder, [[["(at ?b ?r)"]]])
    with pytest.raises(TypeError):
        encoding.history_masks(encoder, [["(at-robby rooma)"]])
    with pytest.raises(TypeError):
        encoding.subgoal_masks(encoder, "(at-robby rooma)")


def test_a_table_of_several_instances_needs_task_ids_for_its_masks():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    blocks = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / f"{p}.pddl") for p in ("smedium", "probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table, max_goal_level=1, history=True, goal_derivations=("plain",),
                                           target_sources=("goal", "subgoal"))
    with pytest.raises(ValueError, match="task_ids"):
        encoding.subgoal_masks(encoder, [[], []])
    with pytest.raises(ValueError, match="task_ids"):
        encoding.subgoal_masks(encoder, [[], []], task_ids=[0, 5])
    states = [t.initial_state for t in tasks]
    ids = np.array([0, 1], dtype=np.int32)
    layers = [[[literal(a) for a in s.atoms()]] for s in states]
    batch = encoder.encode(states, task_ids=ids, subgoals=encoding.subgoal_masks(encoder, layers, task_ids=ids))
    for graph, state in enumerate(states):
        goal = {(x.positive, x.predicate.name, tuple(tasks[graph].objects[o.index] for o in x.objects))
                for x in tasks[graph].encoding_goal}
        held = {(True, a.predicate.name, tuple(o.name for o in a.objects)) for a in state.atoms()}
        assert len(names_of(encoder, batch, SUBGOAL_ROLE)[graph]) == len(held - goal) > 0


def test_inputs_that_do_not_fit_the_encoder_are_refused():
    task = text_task(BLOCKS)
    state = task.initial_state
    atoms = list(state.atoms())
    full = make(task)
    sg = encoding.subgoal_masks(full, [[[literal(atoms[0])]]])
    hist = encoding.history_masks(full, [[(-1, [literal(atoms[0])])]])
    no_levels = make(task, max_goal_level=0, target_sources=("goal", "history"))
    with pytest.raises(ValueError, match="subgoal"):
        no_levels.encode([state], subgoals=sg)
    too_many = make(task, max_goal_level=1)
    with pytest.raises(ValueError, match="levels"):
        too_many.encode([state], subgoals=encoding.subgoal_masks(too_many, [[[], [literal(atoms[0])]]]))
    no_history = make(task, history=False, target_sources=("goal", "subgoal"))
    with pytest.raises(ValueError, match="history"):
        no_history.encode([state], history=hist)
    pos, neg, dt, count = hist
    with pytest.raises(ValueError, match="history count"):
        full.encode([state], history=(pos, neg, dt, np.array([5], dtype=np.int32)))
    with pytest.raises(ValueError, match="history count"):
        full.encode([state], history=(pos, neg, dt, np.array([-1], dtype=np.int32)))
    with pytest.raises(ValueError):
        full.encode([state, state], history=hist)
    wide = (np.ones_like(sg[0]) * np.uint64(0xFFFFFFFFFFFFFFFF), sg[1])
    with pytest.raises(ValueError, match="atom bit"):
        full.encode([state], subgoals=wide)


def test_the_history_options_are_validated_and_reported():
    task = text_task(BLOCKS)
    encoder = make(task, history_max_steps=3)
    assert encoder.options["history"] is True and encoder.options["history_max_steps"] == 3
    assert make(task, history=False, target_sources=("goal",)).options["history"] is False
    with pytest.raises(ValueError, match="history targets need history"):
        make(task, history=False, target_sources=("goal", "history"))
    with pytest.raises(ValueError, match="history_max_steps needs history"):
        make(task, history=False, history_max_steps=2, target_sources=("goal",))
    assert make(task).options["history_max_steps"] is None
    with pytest.raises(ValueError, match="history_max_steps"):
        make(task, history_max_steps=0)
    other = make(task, history_max_steps=4)
    assert other.schema_fingerprint != encoder.schema_fingerprint
    assert make(task).schema_fingerprint == make(task).schema_fingerprint
    plain = encoding.FlatRelationEncoder(task)
    assert plain.schema_fingerprint == encoding.FlatRelationEncoder(task, history=False).schema_fingerprint


def test_a_pickled_encoder_encodes_layers_like_the_original():
    task = text_task(BLOCKS)
    encoder = make(task, history_max_steps=3)
    loaded = pickle.loads(pickle.dumps(encoder))
    states = walk(task, steps=6, walks=2)
    layers, history = rows_of(random.Random(11), states, 2, 5)
    args = dict(subgoals=encoding.subgoal_masks(encoder, layers), history=encoding.history_masks(encoder, history))
    same(encoder.encode(states, **args), loaded.encode(states, **args))
    assert loaded.options == encoder.options
