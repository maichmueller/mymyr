"""Literals over the helper predicates that normalization introduces, given as a row's own goal, subgoal layers or
history steps: under include_derived='domain' the encoding has no relation for them, so every entry point refuses
them (the host raises ValueError naming the literal, the device and the JAX targets flag the row invalid) instead of
leaving part of the goal out; under include_derived=True they are encoded.

The device parts need a GPU, e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_encoding_helper_literals.py
"""

import re

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="needs the PDDL front end")

PHILOSOPHERS = ROOT / "tests/data/pddl/philosophers"
DECLARED = "(blocked philosopher-0)"  # a derived predicate the domain declares
REASON = "include_derived='domain' leaves the helpers out"
FLAGGED = "a row is invalid.*helper predicate that include_derived='domain' leaves out"  # device and JAX


def phil():
    """The task and an atom of a helper predicate that normalization introduced for a quantified condition of the
    domain, with a derived slot (found by its kind, whatever the helper's name)."""
    task = mymyr.Task.from_pddl(PHILOSOPHERS / "domain.pddl", PHILOSOPHERS / "p03-phil4.pddl")
    helpers = {p.name for p in task.formalism.predicates if p.helper == "domain"}
    atom = next(a for a in task.derived_atoms(task.initial_state)
                if a.predicate.name in helpers and a.derived_slot is not None)
    return task, atom


def refusal(where, literal):
    return re.escape(where + " lists the literal " + literal) + ".*" + re.escape(REASON)


def relation_count(encoder, batch, name):
    return int(batch.relation_counts[0, encoder.relation_names.index(name)])


def test_the_literal_is_a_compiled_helper_with_a_slot():
    task, atom = phil()
    name = atom.predicate.name
    assert atom.kind == "derived" and atom.derived_slot is not None
    schema = encoding.EncodingSchema(task)
    assert dict(zip(schema.relation_names, schema.relation_compiled))[name]
    assert name not in encoding.EncodingSchema(task, include_derived="domain").relation_names


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("builder", ["goal_masks", "subgoal_masks", "history_masks"])
def test_the_literal_helpers_of_an_encoder_refuse_it_under_domain(builder, negative):
    task, atom = phil()
    helper = str(atom)
    literal = f"(not {helper})" if negative else helper
    rows = {"goal_masks": [[DECLARED, literal]], "subgoal_masks": [[[DECLARED, literal]]],
            "history_masks": [[(0, [DECLARED, literal])]]}[builder]
    options = {"goal_masks": {}, "subgoal_masks": {"max_goal_level": 1}, "history_masks": {"history": True}}[builder]
    build = getattr(encoding, builder)
    with pytest.raises(ValueError, match=re.escape(f"the literal {literal} ") + ".*" + re.escape(REASON)):
        build(encoding.FlatRelationEncoder(task, include_derived="domain", **options), rows)
    masks = build(encoding.FlatRelationEncoder(task, **options), rows)
    derived = masks[4:] if builder == "history_masks" else masks[2:]  # history: (pos, neg, dt, count, dpos, dneg)
    assert np.asarray(derived[1 if negative else 0]).any()
    # the declared derived predicate keeps its relations
    build(encoding.FlatRelationEncoder(task, include_derived="domain", **options),
          {"goal_masks": [[DECLARED]], "subgoal_masks": [[[DECLARED]]], "history_masks": [[(0, [DECLARED])]]}[builder])


def test_literal_masks_from_arrays_refuse_it_under_domain():
    task, atom = phil()
    helper, name = str(atom), atom.predicate.name
    table = rl.as_table(task)
    predicates = np.array([table.predicate_names.index(name)], np.int32)
    objects = np.array([list(atom.object_indices)], np.int32)
    domain = encoding.FlatRelationEncoder(task, include_derived="domain")
    with pytest.raises(ValueError, match=re.escape(f"the literal {helper} ") + ".*" + re.escape(REASON)):
        encoding.literal_masks(domain, predicates, objects, [True], [0], 1)
    assert encoding.literal_masks(task, predicates, objects, [True], [0], 1).derived_pos.any()


@pytest.mark.parametrize("negative", [False, True])
def test_a_row_goal_over_a_helper_raises_under_domain_and_encodes_under_true(negative):
    task, atom = phil()
    helper, name = str(atom), atom.predicate.name
    literal = f"(not {helper})" if negative else helper
    goals = encoding.goal_masks(task, [[DECLARED, literal]] * 2)  # a Task source: every derived predicate
    states = [task.initial_state] * 2
    domain = encoding.FlatRelationEncoder(task, include_derived="domain")
    with pytest.raises(ValueError, match=refusal("the goal of row 0", literal)):
        domain.encode(states, goals=goals)
    # one goal shared by every row
    with pytest.raises(ValueError, match=refusal("the goal of row 0", literal)):
        domain.encode(states, goals=tuple(np.asarray(m)[:1] for m in goals))
    declared = domain.encode(states, goals=encoding.goal_masks(task, [[DECLARED]] * 2))
    assert relation_count(domain, declared, "[+]blocked[g]") == 1 and declared.goal_complete.tolist() == [1, 1]
    every = encoding.FlatRelationEncoder(task)
    batch = every.encode(states, goals=goals)
    assert relation_count(every, batch, f"[{'-' if negative else '+'}]{name}[g]") == 1


def test_subgoal_layers_and_history_steps_over_a_helper_raise_under_domain():
    task, atom = phil()
    helper, name = str(atom), atom.predicate.name
    states = [task.initial_state]
    layered = encoding.FlatRelationEncoder(task, include_derived="domain", max_goal_level=2)
    layers = encoding.subgoal_masks(task, [[[DECLARED], [helper]]])
    with pytest.raises(ValueError, match=refusal("subgoal layer 1 of row 0", helper)):
        layered.encode(states, subgoals=layers)
    steps = encoding.history_masks(task, [[(0, [DECLARED]), (1, [helper]), (2, [DECLARED])]])
    history = encoding.FlatRelationEncoder(task, include_derived="domain", history=True)
    with pytest.raises(ValueError, match=refusal("history step 1 of row 0", helper)):
        history.encode(states, history=steps)
    # a step that the encoding does not use (older than the newest history_max_steps) is not read
    newest = encoding.FlatRelationEncoder(task, include_derived="domain", history=True, history_max_steps=1)
    assert newest.encode(states, history=steps).history_offsets.tolist() == [0, 1]
    every = encoding.FlatRelationEncoder(task, max_goal_level=2, history=True)
    batch = every.encode(states, subgoals=layers, history=steps)
    assert relation_count(every, batch, f"[+]{name}[ssg]") == 1 and relation_count(every, batch, f"[+]{name}[hist]") == 1


def test_a_stream_refuses_the_rows_when_they_are_appended():
    task, atom = phil()
    helper = str(atom)
    goals = encoding.goal_masks(task, [[helper]])
    stream = encoding.FlatRelationEncoder(task, include_derived="domain").stream()
    with pytest.raises(ValueError, match=refusal("the goal of state 0", helper)):
        stream.append([task.initial_state], goals=goals)
    assert len(stream) == 0
    stream.append([task.initial_state], goals=encoding.goal_masks(task, [[DECLARED]]))
    assert stream.flush().num_graphs == 1


def test_a_horizon_goal_over_a_helper_raises_under_domain():
    from horizon_support import expand_horizons

    task, atom = phil()
    helper = str(atom)
    table = rl.as_table(task)
    view = expand_horizons(table, np.ascontiguousarray(task.encode([task.initial_state])), depth=1)
    goals = encoding.goal_masks(task, [[DECLARED, helper]])
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode="full", include_derived="domain")
    with pytest.raises(ValueError, match=refusal("the goal of graph 0 of the horizon", helper)):
        encoder.encode(view, goals=goals)
    encoder.encode(view, goals=encoding.goal_masks(task, [[DECLARED]]))


def test_jax_flags_the_row_invalid():
    jax = pytest.importorskip("jax")
    from mymyr._core import _rl_jax

    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without the XLA FFI headers")
    import mymyr.encoding.jax as ej

    task, atom = phil()
    helper = str(atom)
    rows = np.ascontiguousarray(task.encode([task.initial_state] * 2))
    domain = encoding.FlatRelationEncoder(task, include_derived="domain")
    good = encoding.goal_masks(task, [[DECLARED]] * 2)
    bad = encoding.goal_masks(task, [[DECLARED], [DECLARED, helper]])
    host = domain.encode(rows, goals=good)
    caps = dict(graphs=2, entities=int(host.entity_offsets[-1]), args=int(host.relation_arg_offsets[-1]), targets=0)
    platforms = ["cpu"] + (["cuda"] if any(d.platform == "gpu" for d in jax.devices()) else [])
    for platform in platforms:
        device = jax.devices(platform)[0]

        def put(a):
            return jax.device_put(np.ascontiguousarray(a).view(np.uint32), device)

        state = put(rows)
        with pytest.raises(ValueError, match=FLAGGED):
            ej.encode(domain, state, goals=tuple(put(m) for m in bad))
        flagged = jax.jit(lambda s, g: ej.encode(domain, s, goals=g, capacities=caps))(state, tuple(put(m) for m in bad))
        assert flagged.sizes["invalid_input"]
        fine = jax.jit(lambda s, g: ej.encode(domain, s, goals=g, capacities=caps))(state, tuple(put(m) for m in good))
        assert not fine.sizes["invalid_input"]


def gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device")
    mc = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not mc.available():
        pytest.skip("no visible CUDA device")
    return torch, torch.device("cuda", 0)


def test_the_device_encoder_flags_the_row_invalid():
    torch, dev = gpu()
    task, atom = phil()
    helper = str(atom)
    rows = np.ascontiguousarray(task.encode([task.initial_state] * 2))
    domain = encoding.FlatRelationEncoder(task, include_derived="domain", max_goal_level=1, history=True)

    def tensors(masks):
        return tuple(torch.from_numpy(np.ascontiguousarray(np.asarray(m)).view(np.int64)).to(dev) for m in masks)

    good = encoding.goal_masks(task, [[DECLARED]] * 2)
    host = domain.encode(rows, goals=good)
    caps = dict(graphs=2, entities=int(host.entity_offsets[-1]), args=int(host.relation_arg_offsets[-1]),
                targets=0, history=0)
    state = torch.from_numpy(rows.view(np.int64)).to(dev)
    for bad in ({"goals": tensors(encoding.goal_masks(task, [[DECLARED], [helper]]))},
                {"goals": tensors(good), "subgoals": tensors(encoding.subgoal_masks(task, [[[DECLARED]], [[helper]]]))}):
        with pytest.raises(ValueError, match=FLAGGED):
            domain.encode(state, **bad)
        assert domain.encode(state, capacities=caps, **bad).sizes["invalid_input"]
    assert not domain.encode(state, goals=tensors(good), capacities=caps).sizes["invalid_input"]
    torch.cuda.synchronize()


def test_device_literal_masks_flag_a_helper_literal_under_domain():
    torch, dev = gpu()
    task, atom = phil()
    name = atom.predicate.name
    table = rl.as_table(task)
    predicates = torch.tensor([table.predicate_names.index(name)], dtype=torch.int32, device=dev)
    objects = torch.tensor([list(atom.object_indices)], dtype=torch.int32, device=dev)
    positive = torch.ones(1, dtype=torch.bool, device=dev)
    rows = torch.zeros(1, dtype=torch.int32, device=dev)
    domain = encoding.FlatRelationEncoder(task, include_derived="domain")
    masks = encoding.literal_masks(domain, predicates, objects, positive, rows, 1)
    assert masks.status.item() == 4 and not masks.derived_pos.any()
    every = encoding.literal_masks(encoding.FlatRelationEncoder(task), predicates, objects, positive, rows, 1)
    assert every.status.item() == 0 and every.derived_pos.any()
    torch.cuda.synchronize()
