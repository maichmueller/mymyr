"""The numeric lane of the relational encoder: one relation `[num]f` per fluent function, one tuple per numeric slot of the
instance, and the value of every tuple in `numeric_values` (and in `node_value` with the Simple graph). Batches are compared
with the tuples and values the definition gives (the slots of the task and `rl.numeric_values`) for both layouts, index
types, value dtypes and storages, mixed instances and nullary functions, and the forms of the encoder agree."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, walk

COUNTERS = ROOT / "tests/data/pddl/counters"
NUMERIC = ROOT / "tests/data/numeric_tasks"
LAYOUTS = ("graph", "relation")
INDICES = ("int32", "int64")
DTYPES = ("float32", "float64")


def needs_front_end():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")


def counters(*problems, **options):
    needs_front_end()
    return rl.TaskTable.from_pddl(COUNTERS / "domain.pddl", [COUNTERS / f"{p}.pddl" for p in problems], **options)


class Case:
    """Rows of a table: the States of seeded walks of each instance, interleaved, and their instance ids."""

    def __init__(self, name, table, steps=12):
        self.name, self.table = name, table
        walks = [walk(task, steps=steps, walks=2, seed=i) for i, task in enumerate(table.tasks)]
        self.states, self.ids = [], []
        for k in range(max(len(w) for w in walks)):
            for i, w in enumerate(walks):
                if k < len(w):
                    self.states.append(w[k])
                    self.ids.append(i)
        self.ids = np.array(self.ids, np.int32)


def build_cases():
    cases = []
    try:
        cases.append(Case("counters_i32", counters("p01", "p02")))
        cases.append(Case("counters_f64", counters("p01", "p03-half", "p02")))
    except pytest.skip.Exception:
        pass
    for name in ("cs-drone", "m-refuel", "cs-expedition"):
        task = mymyr.Task.from_text(str(NUMERIC / f"{name}.txt"))
        cases.append(Case(name, rl.TaskTable([task])))
    return cases


@pytest.fixture(scope="module")
def cases():
    return {c.name: c for c in build_cases()}


CASE_NAMES = ["counters_i32", "counters_f64", "cs-drone", "m-refuel", "cs-expedition"]


def case_of(cases, name):
    if name not in cases:
        pytest.skip(f"{name} needs the loki front end")
    return cases[name]


def slot_function(name):
    """The function and the object names of a slot name such as '(value a)'."""
    parts = name.strip("()").split()
    return parts[0], parts[1:]


def expected(case, encoder, batch):
    """Per graph, the numeric tuples the definition gives: (relation id, local entity ids, value), in the order of the
    relation ids, with the values rounded to the dtype of the encoder."""
    dtype = np.float64 if encoder.options["numeric_dtype"] == "float64" else np.float32
    numeric = [r for r, lane in enumerate(encoder.relation_lanes) if lane == "numeric"]
    by_function = {encoder.relation_names[r]: r for r in numeric}
    num_objects = np.asarray(batch.num_objects)
    out = []
    for g, state in enumerate(case.states):
        task = case.table.tasks[int(case.ids[g])]
        values = state.numeric_values()
        objects = {name: i for i, name in enumerate(task.objects)}
        tuples = []
        for slot, name in enumerate(task.numeric_names):
            function, args = slot_function(name)
            relation = by_function.get(f"[num]{function}")
            if relation is None:
                continue
            local = [objects[a] for a in args] or [int(num_objects[g])]
            tuples.append((relation, local, dtype(values[slot])))
        out.append(sorted(tuples, key=lambda t: t[0]))  # stable: the slot order within a relation
    return out


def decode(batch, encoder):
    """The numeric tuples of every graph read from the batch arrays by the layout: (relation id, local entity ids, value)."""
    lanes = encoder.relation_lanes
    R, G = batch.num_relations, batch.num_graphs
    offsets = np.asarray(batch.relation_offsets)
    arg_offsets = np.asarray(batch.relation_arg_offsets)
    args = np.asarray(batch.relation_args)
    entity_offsets = np.asarray(batch.entity_offsets)
    values = np.asarray(batch.numeric_values)
    relation_major = encoder.options["layout"] == "relation"
    out = [[] for _ in range(G)]
    k = 0
    order = [(g, r) for r in range(R) for g in range(G)] if relation_major else [(g, r) for g in range(G) for r in range(R)]
    for g, r in order:
        if lanes[r] != "numeric":
            continue
        s = r * G + g if relation_major else g * R + r
        arity = encoder.relation_arities[r]
        for t in range(int(offsets[s + 1] - offsets[s])):
            first = int(arg_offsets[s]) + t * arity
            local = [int(a) - int(entity_offsets[g]) for a in args[first : first + arity]]
            out[g].append((r, local, values[k]))
            k += 1
    assert k == len(values)
    for g in range(G):
        out[g].sort(key=lambda t: t[0])
    return out


def same_values(got, want):
    assert len(got) == len(want)
    for g, (a, b) in enumerate(zip(got, want)):
        assert len(a) == len(b), g
        for (ra, la, va), (rb, lb, vb) in zip(a, b):
            assert ra == rb and la == lb, g
            assert (np.isnan(va) and np.isnan(vb)) or va == vb, (g, va, vb)


@pytest.mark.parametrize("name", CASE_NAMES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tuples_and_values_are_those_of_the_states(cases, name, layout, index, dtype):
    case = case_of(cases, name)
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, index_dtype=index, numeric_dtype=dtype, goal_derivations=())
    batch = encoder.encode(case.states, case.ids)
    assert batch.numeric_values.dtype == np.dtype(dtype)
    assert batch.as_dict()["numeric_dtype"] == (1 if dtype == "float64" else 0)
    want = expected(case, encoder, batch)
    assert any(want)
    same_values(decode(batch, encoder), want)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_values_equal_rl_numeric_values(cases, name):
    case = case_of(cases, name)
    encoder = encoding.FlatRelationEncoder(case.table, numeric_dtype="float64", goal_derivations=())
    batch = encoder.encode(case.states, case.ids)
    reference = rl.numeric_values(case.table, case.states, case.ids)  # [N, S]: slot order, NaN past an instance's slots
    got = decode(batch, encoder)
    for g, state in enumerate(case.states):
        task = case.table.tasks[int(case.ids[g])]
        by_name = {}
        for slot, slot_name in enumerate(task.numeric_names):
            by_name.setdefault(slot_function(slot_name)[0], []).append(reference[g, slot])
        per_relation = {}
        for relation, _, value in got[g]:
            per_relation.setdefault(encoder.relation_names[relation], []).append(value)
        for relation, values in per_relation.items():
            want = by_name[relation[len("[num]"):]]
            assert len(values) == len(want)
            assert all((np.isnan(a) and np.isnan(b)) or a == b for a, b in zip(values, want))


def test_an_undefined_value_is_nan(cases):
    needs_front_end()
    domain = mymyr.Domain.from_string(
        "(define (domain undef) (:requirements :strips :typing :numeric-fluents) (:types o) (:predicates (p))"
        " (:functions (u) (w ?x - o))"
        " (:action set-u :parameters () :precondition (and) :effect (assign (u) 1))"
        " (:action set-w :parameters (?x - o) :precondition (and) :effect (assign (w ?x) 4)))")
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem undefined) (:domain undef) (:objects a b - o) (:init (= (w a) 2)) (:goal (and (p))))"))
    table = rl.TaskTable([task])
    for dtype in DTYPES:
        encoder = encoding.FlatRelationEncoder(table, numeric_dtype=dtype, goal_derivations=())
        batch = encoder.encode([task.initial_state])
        values = np.asarray(batch.numeric_values)
        assert len(values) == 3 and int(np.isnan(values).sum()) == 2 and 2.0 in values
        assert encoder.relation_names[-2:] == ["[num]u", "[num]w"]
        # a nullary function is a tuple over the auxiliary entity
        u = encoder.relation_id("[num]u")
        assert batch.relation_counts[0, u] == 1
        first = int(np.asarray(batch.relation_arg_offsets)[u])
        assert int(np.asarray(batch.entity_role)[int(np.asarray(batch.relation_args)[first])]) == 8


MIX_DOMAIN = (
    "(define (domain mix) (:requirements :strips :typing :numeric-fluents) (:types a b)"
    " (:predicates (p ?x - a) (q)) (:functions (f ?x - a) (g ?y - b) (h))"
    " (:action inc-f :parameters (?x - a) :precondition (p ?x) :effect (and (q) (increase (f ?x) 1)))"
    " (:action inc-g :parameters (?y - b) :precondition (q) :effect (and (not (q)) (increase (g ?y) 2)))"
    " (:action inc-h :parameters () :precondition (q) :effect (and (not (q)) (increase (h) 0.5))))"
)
MIX_PROBLEMS = (
    "(define (problem both) (:domain mix) (:objects a2 a1 - a b1 b2 - b)"
    " (:init (p a1) (p a2) (= (f a1) 1) (= (f a2) 3) (= (g b1) 0) (= (g b2) 7) (= (h) 0)) (:goal (and (q))))",
    # no object of type b: no slot of g
    "(define (problem without-g) (:domain mix) (:objects a1 - a)"
    " (:init (p a1) (= (f a1) 0) (= (h) 1)) (:goal (and (q))))",
)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index", INDICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_an_instance_without_slots_of_a_function_has_no_tuples_of_its_relation(layout, index, dtype):
    needs_front_end()
    domain = mymyr.Domain.from_string(MIX_DOMAIN)
    case = Case("mix", rl.TaskTable([mymyr.Task(domain.instantiate_string(p)) for p in MIX_PROBLEMS]))
    assert [len(t.numeric_names) for t in case.table.tasks] == [5, 2]
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    assert [n for n in encoder.relation_names if n.startswith("[num]")] == ["[num]f", "[num]g", "[num]h"]
    batch = encoder.encode(case.states, case.ids)
    same_values(decode(batch, encoder), expected(case, encoder, batch))
    counts = np.asarray(batch.relation_counts)
    f, g, h = (encoder.relation_id(n) for n in ("[num]f", "[num]g", "[num]h"))
    second = case.ids == 1
    assert second.any() and (~second).any()
    assert (counts[second, f] == 1).all() and (counts[second, g] == 0).all() and (counts[:, h] == 1).all()
    assert (counts[~second, f] == 2).all() and (counts[~second, g] == 2).all()
    at = int(np.flatnonzero(second)[0])
    single = encoder.encode([case.states[at]], case.ids[at : at + 1])
    same_values(decode(single, encoder), [decode(batch, encoder)[at]])


@pytest.mark.parametrize("name", ["counters_i32", "cs-drone"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_forms_of_the_encoder_agree(cases, name, layout):
    case = case_of(cases, name)
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, numeric_dtype="float64", goal_derivations=())
    batch = encoder.encode(case.states, case.ids)
    reference = decode(batch, encoder)

    def same_arrays(other):
        a, b = batch.as_dict(), other.as_dict()
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True), key

    # threads
    same_arrays(encoder.encode(case.states, case.ids, threads=3))
    # single rows
    for g in range(0, len(case.states), 5):
        single = encoder.encode([case.states[g]], case.ids[g : g + 1])
        same_values(decode(single, encoder), [reference[g]])
    # the stream and the mutable stream
    stream = encoder.stream()
    stream.append(case.states[:7], case.ids[:7])
    stream.append(case.states[7:], case.ids[7:])
    same_arrays(stream.flush())
    mutable = encoder.mutable_stream()
    mutable.append(list(range(len(case.states))), case.states, case.ids)
    same_arrays(mutable.flush())
    # capacities pad numeric_values with zeros and need the key 'numeric'
    n = len(np.asarray(batch.numeric_values))
    capacities = {"graphs": len(case.states), "entities": int(batch.entity_offsets[-1]) + 3,
                  "args": len(np.asarray(batch.relation_args)) + 3, "numeric": n + 4}
    padded = encoder.encode(case.states, case.ids, capacities=capacities)
    values = np.asarray(padded.numeric_values)
    assert len(values) == n + 4 and np.array_equal(values[:n], batch.numeric_values, equal_nan=True) and not values[n:].any()
    assert padded.sizes["numeric"] == n and not padded.sizes["overflow"]
    with pytest.raises(ValueError, match="needs the key 'numeric'"):
        encoder.encode(case.states, case.ids, capacities={k: v for k, v in capacities.items() if k != "numeric"})
    tight = encoder.encode(case.states, case.ids, capacities={**capacities, "numeric": n - 1})
    assert tight.sizes["overflow"] and tight.sizes["capacity_overflow"] and tight.sizes["numeric"] == n
    assert not np.asarray(tight.numeric_values).any()
    flushed = encoder.stream()
    flushed.append(case.states, case.ids)
    again = flushed.flush(capacities={**capacities, "numeric": n - 1})
    assert again.sizes["overflow"] and again.sizes["numeric"] == n
    kept = mutable.flush(capacities=capacities)
    assert kept.sizes["numeric"] == n and not kept.sizes["overflow"]
    # the pickled encoder encodes the same bytes
    clone = pickle.loads(pickle.dumps(encoder))
    assert clone.options == encoder.options
    same_arrays(clone.encode(case.states, case.ids))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_simple_graph_gives_numeric_atom_nodes_their_values(cases, layout):
    case = case_of(cases, "cs-drone")
    both = encoding.FlatRelationEncoder(case.table, layout=layout, outputs=("relations", "simple"), goal_derivations=())
    batch = both.encode(case.states, case.ids)
    node_value = np.asarray(batch.node_value)
    assert node_value.dtype == np.float32 and len(node_value) == int(np.asarray(batch.node_offsets)[-1])
    want = expected(case, both, batch)
    node_offsets = np.asarray(batch.node_offsets)
    for g in range(batch.num_graphs):
        tail = node_value[node_offsets[g + 1] - len(want[g]) : node_offsets[g + 1]]  # the numeric atoms are the last nodes
        assert all((np.isnan(a) and np.isnan(float(b[2]))) or a == np.float32(b[2]) for a, b in zip(tail, want[g]))
        assert not node_value[node_offsets[g] : node_offsets[g + 1] - len(want[g])].any()
    if layout == "graph":
        simple = encoding.SimpleEncoder(case.table, goal_derivations=())
        only = simple.encode(case.states, case.ids)
        assert np.array_equal(np.asarray(only.node_value), node_value, equal_nan=True)
        assert not hasattr(only, "numeric_values")


def test_options_properties_and_the_catalog(cases):
    case = case_of(cases, "cs-drone")
    encoder = encoding.FlatRelationEncoder(case.table, goal_derivations=())
    assert encoder.options["include_numeric"] is True and encoder.options["numeric_dtype"] == "float32"
    names = [n for n, lane in zip(encoder.relation_names, encoder.relation_lanes) if lane == "numeric"]
    assert names == sorted(names) and all(n.startswith("[num]") for n in names) and names
    first = encoder.relation_lanes.index("numeric")
    assert all(lane == "numeric" for lane in encoder.relation_lanes[first:])
    for name in names:
        r = encoder.relation_id(name)
        assert encoder.relation_arities[r] == max(1, encoder.relation_logical_arities[r])
    # the option leaves them out
    plain = encoding.FlatRelationEncoder(case.table, include_numeric=False, goal_derivations=())
    assert "numeric" not in plain.relation_lanes and plain.num_relations == encoder.num_relations - len(names)
    batch = plain.encode(case.states, case.ids)
    assert not hasattr(batch, "numeric_values") and "numeric_dtype" not in batch.as_dict()
    assert plain.schema_fingerprint != encoder.schema_fingerprint
    wide = encoding.FlatRelationEncoder(case.table, numeric_dtype="float64", goal_derivations=())
    assert wide.schema_fingerprint != encoder.schema_fingerprint
    # a classical table has none, and the dtype is no part of its schema
    classical = mymyr.Task.from_text(str(ROOT / "tests/data/tasks/gripper__prob05.txt"))
    assert (encoding.EncodingSchema(classical, numeric_dtype="float64").schema_fingerprint
            == encoding.EncodingSchema(classical).schema_fingerprint)
    assert "numeric" not in encoding.EncodingSchema(classical).relation_lanes


def test_invalid_numeric_options_and_rows(cases):
    case = case_of(cases, "cs-drone")
    with pytest.raises(ValueError, match="numeric_dtype"):
        encoding.FlatRelationEncoder(case.table, numeric_dtype="float16")
    with pytest.raises(TypeError, match="numeric_dtype"):
        encoding.FlatRelationEncoder(case.table, numeric_dtype=32)
    encoder = encoding.FlatRelationEncoder(case.table, goal_derivations=())
    rows = np.asarray(case.table.initial_states())
    assert rows.shape[1] == case.table.words + case.table.numeric_words
    assert np.array_equal(np.asarray(encoder.encode(rows).numeric_values), np.asarray(encoder.encode(case.states[:1]).numeric_values), equal_nan=True)
    with pytest.raises(ValueError, match="rows \\[W \\+ NN\\]"):
        encoder.encode(rows[:, : case.table.words])


def test_horizon_encoders_refuse_numeric_domains(cases):
    case = case_of(cases, "cs-drone")
    for mode in ("full", "delta", "action"):
        with pytest.raises(ValueError, match="^mymyr: encoding: horizon encodings do not encode numeric functions"):
            encoding.FlatHorizonEncoder(case.table, horizon_mode=mode)
    with pytest.raises(ValueError, match="horizon encodings do not encode numeric functions"):
        encoding.EncodingSchema(case.table, horizon_mode="full")
    with pytest.raises(ValueError, match="numeric"):
        encoding.horizon(case.table.tasks[0], [case.table.tasks[0].initial_state], 1)
