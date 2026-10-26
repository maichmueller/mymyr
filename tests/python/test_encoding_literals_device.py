"""CUDA literals agree with host resolution and masks, including captured replay with changed inputs."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from test_encoding_literals import GOALS, goals_task, ids, predicate_id

mc = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
torch = pytest.importorskip("torch", exc_type=ImportError)
rt = pytest.importorskip("mymyr.rl.torch", exc_type=ImportError)


@pytest.fixture(autouse=True)
def visible_device():
    if not torch.cuda.is_available() or not mc.available():
        pytest.skip("no visible CUDA device")


def tensor(array):
    return torch.as_tensor(np.asarray(array), device="cuda:0")


@pytest.fixture(params=["reachable", "dense"])
def case(request, tmp_path):
    first = goals_task("static_true", atoms=request.param)
    problem = (GOALS / "static_false.pddl").read_text().replace("a b - place", "a b c - place")
    problem = problem.replace("(heavy t1)", "(heavy t2)")
    path = tmp_path / "problem.pddl"
    path.write_text(problem)
    tasks = [first, mymyr.Task.from_pddl(GOALS / "domain.pddl", path, atoms=request.param)]
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table, max_goal_level=2, include_static=False,
                                           include_derived=False, nullary=False)
    literals = ["(lit a)", "(lit b)", "(done)", "(heavy t1)", "(heavy t2)", "(lit t1)",
                "(road a b)", "(road a a)", "(= a a)", "(= a b)"]
    atoms = [ids(tasks[0], literal) for literal in literals]
    predicates = np.array([p for p, _ in atoms] * 2, np.int32)
    objects = np.array([o for _, o in atoms] * 2, np.int32)
    positive = np.array([True] * len(atoms) + [False] * len(atoms), np.bool_)
    rows = np.arange(len(predicates), dtype=np.int32) % 3
    task_ids = np.array([1, 0, 1], np.int32)
    layers = np.arange(len(predicates), dtype=np.int32) % 2
    return tasks, encoder, predicates, objects, positive, rows, task_ids, layers


def assert_masks(host, device):
    assert device.status.dtype == torch.int64 and device.status.shape == () and device.status.item() == 0
    assert device.unsatisfiable.dtype == torch.bool
    np.testing.assert_array_equal(host.unsatisfiable, device.unsatisfiable.cpu().numpy())
    assert len(host) == len(device)
    for expected, actual in zip(host, device, strict=True):
        np.testing.assert_array_equal(expected, actual.cpu().numpy().view(np.uint64))


def test_slots_resolve_every_kind_and_invalid_values(case):
    tasks, encoder, predicates, objects, _, rows, task_ids, _ = case
    instances = task_ids[rows]
    predicates = np.r_[predicates, np.array([-1, 100000, predicates[0], predicates[0]], np.int32)]
    objects = np.concatenate([objects, np.array([[-1, -1, -1], [0, -1, -1],
                                                [100000, -1, -1], [0, 0, 0]], np.int32)])
    instances = np.r_[instances, np.array([0, 0, 0, -1], np.int32)]
    expected = encoding.literal_slots(encoder, predicates, objects, instances)
    kinds, slots, status = encoding.literal_slots(encoder, tensor(predicates), tensor(objects), tensor(instances))
    assert kinds.dtype == torch.uint8 and slots.dtype == torch.int32
    for host, device in zip(expected, (kinds, slots), strict=True):
        np.testing.assert_array_equal(host, device.cpu().numpy())
    assert set(expected[0]) == {int(k) for k in (encoding.LiteralKind.FLUENT, encoding.LiteralKind.DERIVED,
                                               encoding.LiteralKind.STATIC_TRUE, encoding.LiteralKind.STATIC_FALSE,
                                               encoding.LiteralKind.UNREACHABLE, encoding.LiteralKind.INVALID)}
    assert status.item() == 4
    # A predicate with greater arity than the object width is invalid without reading beyond the input row.
    predicate = predicate_id(tasks[0], "road")
    kinds, slots, status = encoding.literal_slots(encoder, tensor(np.array([predicate], np.int32)),
                                                  tensor(np.zeros((1, 1), np.int32)), tensor(np.zeros(1, np.int32)))
    assert kinds.item() == encoding.LiteralKind.INVALID and slots.item() == -1 and status.item() == 4


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("halves", [False, True])
def test_goal_and_layer_masks_equal_host_with_duplicate_literals(case, layered, halves):
    _, encoder, p, o, signs, rows, task_ids, layers = case
    # Many threads target the same mask words and unsatisfiable cells.
    p, o, signs, rows, layers = (np.tile(x, (40, 1) if x.ndim == 2 else 40)
                                for x in (p, o, signs, rows, layers))
    kwargs = dict(layers=layers, num_layers=2) if layered else {}
    expected = encoding.literal_masks(encoder, p, o, signs, rows, 3, task_ids, **kwargs)
    device_kwargs = dict(layers=tensor(layers), num_layers=2) if layered else {}
    actual = encoding.literal_masks(encoder, tensor(p), tensor(o), tensor(signs), tensor(rows), 3,
                                    tensor(task_ids), word_dtype=torch.uint32 if halves else torch.int64,
                                    **device_kwargs)
    assert isinstance(actual, encoding.LayerMasks if layered else encoding.GoalMasks)
    assert actual.pos.dtype == (torch.uint32 if halves else torch.int64)
    assert_masks(expected, actual)
    assert actual.unsatisfiable.any().item()


def test_invalid_placements_and_unreferenced_task_ids_set_status(case):
    _, encoder, p, o, signs, rows, task_ids, layers = case
    for field in ("predicate", "object", "padding", "row", "layer", "instance"):
        pp, oo, rr, ll, tt = (x.copy() for x in (p, o, rows, layers, task_ids))
        if field == "predicate":
            pp[0] = -1
        elif field == "object":
            oo[0, 0] = 100000
        elif field == "padding":
            oo[0, -1] = 0
        elif field == "row":
            rr[0] = -1
        elif field == "layer":
            ll[0] = 2
        else:
            tt[0] = 2
        result = encoding.literal_masks(encoder, tensor(pp), tensor(oo), tensor(signs), tensor(rr), 3,
                                        tensor(tt), layers=tensor(ll), num_layers=2)
        assert result.status.item() == 4, field
    empty = encoding.literal_masks(encoder, tensor(p[:0]), tensor(o[:0]), tensor(signs[:0]), tensor(rows[:0]),
                                   3, tensor(np.array([0, -1, 1], np.int32)))
    assert empty.status.item() == 4 and not empty.pos.any().item()
    empty = encoding.literal_masks(encoder, tensor(p[:0]), tensor(o[:0]), tensor(signs[:0]), tensor(rows[:0]),
                                   0, layers=tensor(layers[:0]), num_layers=0)
    assert empty.pos.shape == (0, 0, encoder.table.words) and empty.status.item() == 0


@pytest.mark.parametrize("layered", [False, True])
def test_custom_ops_fake_compile_and_cuda_graph_replay(case, layered):
    _, encoder, p, o, signs, rows, task_ids, layers = case
    handle = rt.register(encoder)
    inputs = [tensor(x) for x in (p, o, signs, rows, task_ids, layers)]
    stream = torch.cuda.Stream(device=0)
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    try:
        def run(p, o, s, r, t, l):
            masks = torch.ops.mymyr.literal_masks(handle, p, o, s, r, 3, t, l if layered else None, 2 if layered else 1)
            slots = torch.ops.mymyr.literal_slots(handle, p, o, t[r.long()])
            return [*masks, *slots]

        eager = run(*inputs)
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            fake = run(*(mode.from_tensor(x) for x in inputs))
            for actual, reference in zip(fake, eager, strict=True):
                assert actual.shape == reference.shape and actual.dtype == reference.dtype
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        for actual, expected in zip(compiled(*inputs), eager, strict=True):
            assert torch.equal(actual, expected)
        with torch.cuda.stream(stream):
            run(*inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run(*inputs)
            graph.replay()
        stream.synchronize()
        for actual, expected in zip(captured, eager, strict=True):
            assert torch.equal(actual, expected)
        with torch.cuda.stream(stream):
            inputs[2].logical_not_()
            inputs[4].copy_(tensor(task_ids[::-1].copy()))
            inputs[5].copy_(1 - inputs[5])
            graph.replay()
        stream.synchronize()
        expected = encoding.literal_masks(encoder, p, o, ~signs, rows, 3, task_ids[::-1].copy(),
                                          **(dict(layers=1 - layers, num_layers=2) if layered else {}))
        result = (encoding.LayerMasks if layered else encoding.GoalMasks)(*captured[:6])
        assert_masks(expected, result)
        instances = task_ids[::-1][rows]
        for host, actual in zip(encoding.literal_slots(encoder, p, o, instances), captured[6:8], strict=True):
            np.testing.assert_array_equal(host, actual.cpu().numpy())
        with torch.cuda.stream(stream):
            inputs[0][0] = -1
            graph.replay()
        stream.synchronize()
        assert captured[5].item() == 4 and captured[-1].item() == 4
        with torch.cuda.stream(stream):
            inputs[0][0] = int(p[0])
            graph.replay()
        stream.synchronize()
        assert captured[5].item() == 0 and captured[-1].item() == 0
        assert_masks(expected, result)
    finally:
        stream.synchronize()
        del graph
        encoder.release_device(stream=stream)
        rt.unregister(handle)


def test_device_masks_encode_as_host_goals_and_layers(case):
    tasks, encoder, p, o, signs, rows, task_ids, layers = case
    states = np.concatenate([tasks[i].encode([tasks[i].initial_state]) for i in task_ids])
    goals = encoding.literal_masks(encoder, p, o, signs, rows, 3, task_ids)
    subgoals = encoding.literal_masks(encoder, p, o, signs, rows, 3, task_ids, layers=layers, num_layers=2)
    dg = encoding.literal_masks(encoder, tensor(p), tensor(o), tensor(signs), tensor(rows), 3, tensor(task_ids))
    ds = encoding.literal_masks(encoder, tensor(p), tensor(o), tensor(signs), tensor(rows), 3, tensor(task_ids),
                                layers=tensor(layers), num_layers=2)
    host = encoder.encode(states, task_ids, goals=goals, subgoals=subgoals)
    device = encoder.encode(tensor(states.view(np.int64)), tensor(task_ids), goals=dg, subgoals=ds)
    from test_encoding_cuda import same_batch
    same_batch(host, device)


def test_single_instance_defaults_nullary_empty_and_validation():
    task = goals_task("static_true")
    encoder = encoding.FlatRelationEncoder(task)
    p, o = ids(task, "(done)", 0)
    predicates = tensor(np.array([p], np.int32))
    objects = tensor(np.empty((1, 0), np.int32))
    positive, rows = tensor(np.ones(1, np.bool_)), tensor(np.zeros(1, np.int32))
    expected = encoding.literal_slots(encoder, [p], np.empty((1, 0), np.int32))
    kinds, slots, status = encoding.literal_slots(encoder, predicates, objects)
    assert kinds.item() == expected[0][0] and slots.item() == expected[1][0] and status.item() == 0
    assert_masks(encoding.literal_masks(encoder, [p], np.empty((1, 0), np.int32), [True], [0], 1),
                 encoding.literal_masks(encoder, predicates, objects, positive, rows, 1))
    for kwargs in (dict(fixed="raise"), dict(layers=rows), dict(num_layers=2), dict(num_rows=-1),
                   dict(word_dtype=torch.float32)):
        with pytest.raises(ValueError):
            encoding.literal_masks(encoder, predicates, objects, positive, rows, **(dict(num_rows=1) | kwargs))
    with pytest.raises(TypeError, match="int32"):
        encoding.literal_slots(encoder, predicates.long(), objects)
    with pytest.raises(TypeError, match="bool"):
        encoding.literal_masks(encoder, predicates, objects, positive.int(), rows, 1)
    handle = rt.register(encoder)
    try:
        with pytest.raises(ValueError, match="num_layers"):
            rt.literal_masks(handle, predicates, objects, positive, rows, 1, num_layers=2)
    finally:
        rt.unregister(handle)
    with pytest.raises(TypeError, match="encoder"):
        encoding.literal_slots(task, predicates, objects)
    with pytest.raises(TypeError, match="cuda"):
        encoding.literal_slots(encoder, predicates, objects.cpu())
    with pytest.raises(ValueError, match="contiguous"):
        encoding.literal_slots(encoder, predicates.expand(2), objects.expand(2, 0))


def test_fluent_words_without_derived_slots_include_the_high_bit():
    from conftest import text_task
    task = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task)
    atoms = [task.atom(slot) for slot in (0, 63, 64, 65)]
    assert {atom.slot for atom in atoms} >= {63, 64}
    p = np.array([predicate_id(task, atom.predicate.name) for atom in atoms], np.int32)
    width = max(len(atom.object_indices) for atom in atoms)
    o = np.array([list(atom.object_indices) + [-1] * (width - len(atom.object_indices)) for atom in atoms], np.int32)
    signs, rows = np.ones(len(p), np.bool_), np.zeros(len(p), np.int32)
    expected = encoding.literal_masks(encoder, p, o, signs, rows, 1)
    actual = encoding.literal_masks(encoder, tensor(p), tensor(o), tensor(signs), tensor(rows), 1)
    assert actual.derived_pos is None and actual.derived_neg is None
    assert_masks(expected, actual)
    handle = rt.register(encoder)
    try:
        inputs = (tensor(p), tensor(o), tensor(signs), tensor(rows))
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            fake = rt.literal_masks(handle, *(mode.from_tensor(x) for x in inputs), 1)
        assert fake[2].shape == (1, 0) and fake[3].shape == (1, 0)
        op = rt.literal_masks(handle, *inputs, 1)
        assert_masks(expected, encoding.GoalMasks(op[0], op[1], None, None, op[4], op[5]))
    finally:
        rt.unregister(handle)
