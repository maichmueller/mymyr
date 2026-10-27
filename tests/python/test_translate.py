"""Moving state values between parses by predicate, function and object names."""

import pickle
import re

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, walk


def reorder_section(text, name):
    """Reverse declarations without changing the declarations themselves."""
    match = re.search(r"\(:" + name + r"\b", text, re.I)
    start = match.end()
    depth = 1
    end = start
    while depth:
        depth += (text[end] == "(") - (text[end] == ")")
        end += 1
    content = text[start:end - 1]
    if name in ("predicates", "functions"):
        entries = re.findall(r"\([^()]*\)(?:\s*-\s*number)?", content)
        replacement = "\n" + "\n".join(reversed(entries)) + "\n"
    else:
        tokens = content.split()
        groups, names = [], []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-":
                groups.append((names, tokens[i + 1]))
                names = []
                i += 2
            else:
                names.append(tokens[i])
                i += 1
        if names:
            groups.append((names, "object"))
        replacement = " " + " ".join(" ".join(reversed(names)) + " - " + kind for names, kind in reversed(groups)) + " "
    return text[:start] + replacement + text[end - 1:]


def parse(domain, problem, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem), **options)


def ipc_pair(tmp_path, atoms="auto"):
    path = ROOT / "tests/data/ipc/childsnack-ipc/test"
    domain = (path / "domain.pddl").read_text()
    problem = (path / "p05-medium.pddl").read_text()
    variant = reorder_section(reorder_section(domain, "predicates"), "types") + "\n; equivalent declarations\n"
    reordered = reorder_section(problem, "objects")
    (tmp_path / "domain.pddl").write_text(variant)
    (tmp_path / "problem.pddl").write_text(reordered)
    return parse(domain, problem, atoms=atoms), parse(variant, reordered, atoms=atoms)


def names(state):
    return {str(atom) for atom in state.atoms()}


@pytest.mark.parametrize("atoms", ["dense", "reachable", "lazy"])
def test_ipc_states_roundtrip_and_incremental_map(tmp_path, atoms):
    source, target = ipc_pair(tmp_path, atoms)
    mapping = mymyr.SlotMap(source, target)
    initial = mapping.translate(source.initial_state)
    assert initial.task is target
    assert names(initial) == names(source.initial_state)
    states = walk(source, steps=5, walks=1)
    translated = mapping.translate(states)
    assert [names(x) for x in translated] == [names(x) for x in states]
    assert target.translate(translated, source) == states
    assert source.local().translate(states[0], target.local()) == initial
    twin = target.frozen()
    assert source.translate(states[0], twin).task is twin
    with pytest.raises(ValueError, match="another source task"):
        mapping.translate(target.initial_state)


@pytest.mark.parametrize("framework", ["numpy", "torch", "jax"])
def test_ipc_rows_same_framework_and_pool(tmp_path, framework):
    if framework != "numpy":
        pytest.importorskip(framework)
    source, target = ipc_pair(tmp_path)
    states = walk(source, steps=4, walks=1)
    rows = source.encode(states, framework=framework)
    translated = source.translate(rows, target, pool=rl.ThreadPool(2))
    assert type(translated).__module__.split(".")[0] == type(rows).__module__.split(".")[0]
    assert [names(s) for s in target.decode(translated)] == [names(s) for s in states]
    roundtrip = target.translate(translated, source)
    assert np.array_equal(np.asarray(roundtrip), np.asarray(rows))


DOMAIN = """(define (domain sample) (:requirements :strips :typing)
(:types a b - object)
(:predicates (p ?x - object) (q ?x - object))
(:action set :parameters (?x - object) :precondition (and) :effect (and (p ?x) (q ?x))))"""
PROBLEM = """(define (problem sample) (:domain sample) (:objects one two - a)
(:init (p one)) (:goal (q two)))"""


def test_missing_raise_drop_names_and_rows():
    source = parse(DOMAIN, PROBLEM, atoms="dense")
    target = parse(DOMAIN.replace("(q ", "(r "), PROBLEM.replace("(q ", "(r "), atoms="dense")
    mapping = mymyr.SlotMap(source, target)
    assert "q/1" in mapping.unmatched
    states = [source.initial_state, source.state(["(p one)", "(q two)"])]
    with pytest.raises(ValueError, match=r"row 1: \(q two\)"):
        source.translate(states, target)
    with pytest.raises(ValueError, match=r"row 1: \(q two\)"):
        source.translate(source.encode(states), target, pool=rl.ThreadPool(2))
    translated, lossless = mapping.translate(states, missing="drop")
    assert lossless.dtype == np.bool_
    assert lossless.tolist() == [True, False]
    assert names(translated[1]) == {"(p one)"}
    rows, lossless = source.translate(source.encode(states), target, missing="drop")
    assert lossless.tolist() == [True, False]
    assert names(target.decode(rows)[1]) == {"(p one)"}
    with pytest.raises(ValueError, match="missing must"):
        mapping.translate(states, missing="ignore")
    bad = source.encode(states).copy()
    bad[0, -1] |= np.uint64(1 << 63)
    with pytest.raises(ValueError, match="unknown source slot"):
        mapping.translate(bad, missing="drop")


def test_missing_objects_and_arity_are_reported():
    source = parse(DOMAIN, PROBLEM, atoms="dense")
    target = parse(DOMAIN, PROBLEM.replace("two", "three"), atoms="dense")
    assert "two" in mymyr.SlotMap(source, target).unmatched
    with pytest.raises(ValueError, match=r"\(q two\)"):
        source.translate(source.state(["(q two)"]), target)
    different_arity = parse(DOMAIN.replace("(q ?x - object)", "(q ?x ?y - object)").replace("(q ?x)", "(q ?x ?x)"),
                            PROBLEM.replace("(q two)", "(q two two)"), atoms="dense")
    assert "q/1" in mymyr.SlotMap(source, different_arity).unmatched
    with pytest.raises(ValueError, match=r"\(q two\)"):
        source.translate(source.state(["(q two)"]), different_arity)


def test_fingerprints_follow_slot_meanings(tmp_path):
    source, target = ipc_pair(tmp_path, "dense")
    assert source.fingerprint != target.fingerprint
    assert source.encoding_fingerprint != target.encoding_fingerprint
    clone = pickle.loads(pickle.dumps(source))
    assert source.encoding_fingerprint == clone.encoding_fingerprint
    commented = parse(DOMAIN + "\n; comment", PROBLEM, atoms="dense")
    original = parse(DOMAIN, PROBLEM, atoms="dense")
    assert original.encoding_fingerprint == commented.encoding_fingerprint
    renamed_problem = parse(DOMAIN, PROBLEM.replace("(problem sample)", "(problem elsewhere)"), atoms="dense")
    extra_static = parse(DOMAIN.replace("(:predicates ", "(:predicates (fixed) "), PROBLEM, atoms="dense")
    assert original.encoding_fingerprint == renamed_problem.encoding_fingerprint == extra_static.encoding_fingerprint
    swapped = parse(DOMAIN.replace("(p ?x - object) (q ?x - object)", "(q ?x - object) (p ?x - object)"), PROBLEM, atoms="dense")
    assert original.encoding_fingerprint != swapped.encoding_fingerprint
    renamed = parse(DOMAIN, PROBLEM.replace("one", "other"), atoms="dense")
    assert original.encoding_fingerprint != renamed.encoding_fingerprint
    assert rl.TaskTable([original]).fingerprint != rl.TaskTable([swapped]).fingerprint


NUM_DOMAIN = """(define (domain numbers) (:requirements :strips :typing :numeric-fluents)
(:types item) (:predicates (ready)) (:functions (f ?x - item) (g ?x - item))
(:action change :parameters (?x - item) :precondition (ready)
:effect (and (increase (f ?x) 1) (increase (g ?x) 2))))"""
NUM_PROBLEM = """(define (problem n) (:domain numbers) (:objects a b - item)
(:init (ready) (= (f a) 3) (= (f b) 5) (= (g a) 7) (= (g b) 9)) (:goal (> (f a) 4)))"""


def numeric_pair():
    return (parse(NUM_DOMAIN, NUM_PROBLEM, numeric_storage="i32"),
            parse(reorder_section(NUM_DOMAIN, "functions"), reorder_section(NUM_PROBLEM, "objects"), numeric_storage="f64"))


def test_numeric_reorder_storage_and_loss():
    source, target = numeric_pair()
    states = [source.initial_state, source.successor_states(source.initial_state)[0]]
    translated = source.translate(states, target)
    for original, moved in zip(states, translated, strict=True):
        assert dict(zip(source.numeric_names, original.numeric_values(), strict=True)) == dict(
            zip(target.numeric_names, moved.numeric_values(), strict=True))
    assert target.translate(translated, source) == states
    rows = source.translate(source.encode(states), target)
    assert target.decode(rows) == translated
    with pytest.raises(ValueError, match="representable value"):
        target.translate(target.state([], values=[0.5] * target.numeric_slots), source)
    _, mask = target.translate(target.state([], values=[0.5] * target.numeric_slots), source, missing="drop")
    assert mask.tolist() == [False]


def test_tables_preserve_instance_ids_and_padding(tmp_path):
    source, target = ipc_pair(tmp_path)
    a, b = rl.TaskTable([source, source]), rl.TaskTable([target, target])
    ids = np.array([1, 0, 1], dtype=np.int32)
    rows = source.encode([source.initial_state] * 3)
    mapping = mymyr.SlotMap(a, b)
    expected = source.translate(rows, target)
    assert np.array_equal(a.translate(rows, ids, b), expected)
    assert np.array_equal(mapping.translate(rows, task_ids=ids), expected)
    assert np.array_equal(b.translate(expected, ids, a), rows)
    with pytest.raises(ValueError, match="same instance count"):
        a.translate(rows, ids, rl.TaskTable([target]))
    with pytest.raises(ValueError, match="task_ids"):
        mapping.translate(rows, task_ids=np.array([2, 0, 0], np.int32))


def gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not hasattr(mymyr, "cuda"):
        try:
            from mymyr import cuda
        except ImportError:
            pytest.skip("CUDA backend unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.cuda.set_per_process_memory_fraction(0.15, 0)
    return torch


@pytest.mark.parametrize("numeric", [False, True])
def test_device_rows_and_capture(tmp_path, numeric):
    torch = gpu()
    source, target = numeric_pair() if numeric else ipc_pair(tmp_path)
    states = [source.initial_state] * 4
    rows = torch.from_numpy(source.encode(states).view(np.int64).copy()).cuda()
    expected = source.translate(source.encode(states), target)
    mapping = mymyr.SlotMap(source, target)
    assert np.array_equal(mapping.translate(rows).cpu().numpy().view(np.uint64), expected)
    stream = torch.cuda.Stream()
    status = torch.empty(3, dtype=torch.int32, device="cuda")
    mapping.translate(rows, status=status, stream=stream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = source.translate(rows, target, status=status, stream=stream)
    graph.replay()
    torch.cuda.synchronize()
    assert status.tolist() == [0, -1, -1]
    assert np.array_equal(output.cpu().numpy().view(np.uint64), expected)
    graph.replay()
    torch.cuda.synchronize()
    assert np.array_equal(output.cpu().numpy().view(np.uint64), expected)
    graph.reset()
    del output
    stream.synchronize()
    a, b = rl.TaskTable([source, source]), rl.TaskTable([target, target])
    ids = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="cuda")
    assert np.array_equal(a.translate(rows, ids, b).cpu().numpy().view(np.uint64), expected)


def test_device_missing_and_status():
    torch = gpu()
    source = parse(DOMAIN, PROBLEM, atoms="dense")
    target = parse(DOMAIN.replace("(q ", "(r "), PROBLEM.replace("(q ", "(r "), atoms="dense")
    states = [source.initial_state, source.state(["(q two)"])]
    rows = torch.from_numpy(source.encode(states).view(np.int64).copy()).cuda()
    mapping = mymyr.SlotMap(source, target)
    with pytest.raises(ValueError, match=r"row 1: \(q two\)"):
        mapping.translate(rows)
    result, mask = mapping.translate(rows, missing="drop")
    assert mask.tolist() == [True, False]
    assert names(target.decode(result.cpu().numpy())[1]) == set()
    status = torch.empty(3, dtype=torch.int32, device="cuda")
    result = mapping.translate(rows, status=status)
    torch.cuda.synchronize()
    assert status[0].item() == 1 and status[1].item() == 1
    assert status[2].item() >= 0
    lazy = parse(DOMAIN, PROBLEM, atoms="lazy")
    before = lazy.num_atoms
    with pytest.raises(ValueError, match=r"fixed atom slots .*target\.frozen\(\)"):
        source.translate(rows, lazy)
    assert lazy.num_atoms == before
    host = source.translate(source.encode(states), lazy)
    assert np.array_equal(source.translate(rows, lazy.frozen()).cpu().numpy().view(np.uint64),
                          lazy.translate(host, lazy.frozen()))


def test_equal_layout_rows_are_identity():
    source = parse(DOMAIN, PROBLEM)
    target = parse(DOMAIN + "\n; another parse", PROBLEM)
    rows = source.encode([source.initial_state])
    assert source.translate(rows, target) is rows
    same, lossless = source.translate(rows, target, missing="drop")
    assert same is rows and lossless.tolist() == [True]


def test_device_jax_rows(tmp_path):
    gpu()
    jax = pytest.importorskip("jax")
    source, target = ipc_pair(tmp_path)
    host = source.encode([source.initial_state] * 3)
    rows = jax.device_put(host.view(np.uint32), jax.devices("gpu")[0])
    result = source.translate(rows, target)
    assert type(result).__module__.startswith(("jax", "jaxlib"))
    assert result.devices() == rows.devices()
    assert np.array_equal(np.asarray(result).view(np.uint64), source.translate(host, target))


def test_device_lazy_source_extends_map(tmp_path):
    torch = gpu()
    source, target = ipc_pair(tmp_path, "lazy")
    target = target.frozen()
    mapping = mymyr.SlotMap(source, target)
    initial = source.encode([source.initial_state])
    mapping.translate(torch.from_numpy(initial.view(np.int64).copy()).cuda())
    states = walk(source, steps=4, walks=1)
    host = source.encode(states)
    result = mapping.translate(torch.from_numpy(host.view(np.int64).copy()).cuda())
    assert np.array_equal(result.cpu().numpy().view(np.uint64), mapping.translate(host))


def test_missing_numeric_slots_raise_or_drop():
    source, _ = numeric_pair()
    target = parse(NUM_DOMAIN.replace("(g ", "(h "), NUM_PROBLEM.replace("(g ", "(h "))
    with pytest.raises(ValueError, match=r"row 0: \(g "):
        source.translate(source.initial_state, target)
    _, lossless = source.translate(source.encode([source.initial_state]), target, missing="drop")
    assert lossless.tolist() == [False]


def test_device_unrepresentable_numeric_status():
    torch = gpu()
    integer, floating = numeric_pair()
    host = floating.encode([floating.state([], values=[0.5] * floating.numeric_slots)])
    rows = torch.from_numpy(host.view(np.int64).copy()).cuda()
    with pytest.raises(ValueError, match="representable value"):
        floating.translate(rows, integer)
    status = torch.empty(3, dtype=torch.int32, device="cuda")
    result, mask = floating.translate(rows, integer, missing="drop", status=status)
    torch.cuda.synchronize()
    assert mask.tolist() == [False]
    assert status[0].item() == 4 and status[1].item() == 0
    expected, _ = floating.translate(host, integer, missing="drop")
    assert np.array_equal(result.cpu().numpy().view(np.uint64), expected)


def test_mixed_table_widths_ignore_padding(tmp_path):
    source, target = ipc_pair(tmp_path)
    path = ROOT / "tests/data/ipc/childsnack-ipc/test"
    domain = (path / "domain.pddl").read_text()
    problem = (path / "p01-easy.pddl").read_text()
    small_source = parse(domain, problem)
    small_target = parse(reorder_section(reorder_section(domain, "predicates"), "types"),
                         reorder_section(problem, "objects"))
    a, b = rl.TaskTable([small_source, source]), rl.TaskTable([small_target, target])
    ids = np.array([1, 0, 1], np.int32)
    rows = np.zeros((3, a.words), np.uint64)
    expected = np.zeros((3, b.words), np.uint64)
    for row, instance in enumerate(ids):
        src, dst = a[instance], b[instance]
        encoded = src.encode([src.initial_state])
        rows[row, :encoded.shape[1]] = encoded[0]
        rows[row, encoded.shape[1]:] = np.iinfo(np.uint64).max
        translated = src.translate(encoded, dst)
        expected[row, :translated.shape[1]] = translated[0]
    assert np.array_equal(a.translate(rows, ids, b), expected)


def test_lazy_targets_intern_only_translated_atoms(tmp_path):
    source, _ = ipc_pair(tmp_path, "dense")
    _, target = ipc_pair(tmp_path, "lazy")
    before = target.num_atoms
    states = walk(source, steps=4, walks=1)
    translated = mymyr.SlotMap(source, target).translate(states)
    assert [names(s) for s in translated] == [names(s) for s in states]
    assert target.num_atoms <= before + len(set().union(*(names(s) for s in states)))
    assert target.num_atoms < source.num_atoms
    rows = source.translate(source.encode(states), target)
    assert rows.shape == (len(states), target.words)
    assert np.array_equal(target.translate(rows, source), source.encode(states))


def invalid_and_missing_rows():
    """Rows 0 and 2 hold an atom the target lacks; row 2 also sets a bit past the source's slots."""
    source = parse(DOMAIN, PROBLEM, atoms="dense")
    target = parse(DOMAIN.replace("(q ", "(r "), PROBLEM.replace("(q ", "(r "), atoms="dense")
    rows = source.encode([source.state(["(q two)"]), source.initial_state, source.state(["(q two)"])])
    rows[2, -1] |= np.uint64(1 << 63)
    return source, target, rows, 64 * rows.shape[1] - 1


def test_invalid_slots_raise_before_missing_values():
    source, target, rows, slot = invalid_and_missing_rows()
    for missing in ("raise", "drop"):
        for pool in (None, rl.ThreadPool(2)):
            with pytest.raises(ValueError, match=f"row 2 sets an unknown source slot {slot}$"):
                source.translate(rows, target, missing=missing, pool=pool)


def test_device_invalid_slots_raise_like_the_host():
    torch = gpu()
    source, target, host, slot = invalid_and_missing_rows()
    rows = torch.from_numpy(host.view(np.int64).copy()).cuda()
    mapping = mymyr.SlotMap(source, target)
    for missing in ("raise", "drop"):
        with pytest.raises(ValueError, match=f"row 2 sets an unknown source slot {slot}$"):
            mapping.translate(rows, missing=missing)
    status = torch.empty(3, dtype=torch.int32, device="cuda")
    mapping.translate(rows, status=status)
    torch.cuda.synchronize()
    assert status.tolist() == [3, 2, slot]
    clean = torch.from_numpy(host[:2].view(np.int64).copy()).cuda()
    mapping.translate(clean, status=status)
    torch.cuda.synchronize()
    assert status.tolist() == [1, 0, status[2].item()] and status[2].item() >= 0
    a, b = rl.TaskTable([source, source]), rl.TaskTable([target, target])
    ids = torch.tensor([0, 1, 5], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=r"task_ids\[2\] is outside the source table"):
        a.translate(clean.repeat(2, 1)[:3], ids, b, missing="drop")


def test_invalid_task_ids_raise_before_invalid_slots_on_host_and_device():
    """Row 0 sets a bit past the source's slots and row 1 has a task id outside the table: both report the task id,
    since every task id is checked before any row."""
    source, target, host, _ = invalid_and_missing_rows()
    a, b = rl.TaskTable([source, source]), rl.TaskTable([target, target])
    rows = host[[2, 1]]
    message = r"task_ids\[1\] is outside the source table"
    for missing in ("raise", "drop"):
        with pytest.raises(ValueError, match=message):
            a.translate(rows, np.array([0, 5], dtype=np.int32), b, missing=missing)
    torch = gpu()
    device_rows = torch.from_numpy(rows.view(np.int64).copy()).cuda()
    ids = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    for missing in ("raise", "drop"):
        with pytest.raises(ValueError, match=message):
            a.translate(device_rows, ids, b, missing=missing)
    status = torch.empty(3, dtype=torch.int32, device="cuda")
    a.translate(device_rows, ids, b, status=status)
    torch.cuda.synchronize()
    assert status.tolist() == [3, 1, -1]
