"""Action masks and full-label applicability equal the labels of an unpruned expansion."""

import concurrent.futures
import os
import pathlib
import sys
import threading
import time

import numpy as np
import pytest

import mymyr
from mymyr import rl
from conftest import ROOT, SMALL_TASKS, text_task, walk

DATA = ROOT / "tests/data"
PDDL = sorted(p for base in (DATA / "ipc", DATA / "pddl") for p in base.rglob("*.pddl") if p.name != "domain.pddl")


def pddl_task(path, **options):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    return mymyr.Task.from_pddl(path.parent / "domain.pddl", path, **options)


def check_equal(table, states, ids=None, pool=None, seed=17):
    exp = rl.expand(table, states, ids, witness_pruning=False)
    np.testing.assert_array_equal(rl.applicable_schemas(table, states, ids, pool=pool), rl.schema_masks(exp))
    n, width = exp.num_states, exp.label_width
    schemas = np.asarray(exp.schema)
    bindings = np.asarray(exp.binding)
    offsets = np.asarray(exp.offsets)
    rng = np.random.default_rng(seed)
    for schema in range(exp.num_schemas):
        chosen = np.full(n, schema, np.int32)
        arities = []
        for i in range(n):
            task = table if isinstance(table, mymyr.Task) else table[int(ids[i]) if ids is not None else 0]
            if schema < task.num_schemas:
                arities.append(task.formalism.schemas[schema].arity)
        for depth in range(min(arities, default=width)):
            for query in range(3):
                prefix = rng.integers(-1, exp.num_objects + 1, (n, width), dtype=np.int32)
                if query == 0:
                    for i in range(n):
                        js = np.flatnonzero(schemas[offsets[i]:offsets[i + 1]] == schema)
                        if len(js):
                            prefix[i] = bindings[offsets[i] + js[0]]
                elif query == 2 and depth:
                    prefix[:, depth - 1] = exp.num_objects
                pre = prefix if depth else None
                actual = rl.applicable_objects(table, states, chosen, pre, depth, ids, pool=pool)
                expected = rl.prefix_masks(exp, chosen, pre, depth)
                np.testing.assert_array_equal(actual, expected, err_msg=f"{schema}/{depth}/{query}")
        for query in range(3):
            binding = rng.integers(-1, exp.num_objects + 1, (n, width), dtype=np.int32)
            expected = np.zeros(n, bool)
            for i in range(n):
                task = table if isinstance(table, mymyr.Task) else table[int(ids[i]) if ids is not None else 0]
                if schema >= task.num_schemas:
                    continue
                arity = task.formalism.schemas[schema].arity
                js = np.flatnonzero(schemas[offsets[i]:offsets[i + 1]] == schema) + offsets[i]
                if query == 0 and len(js):
                    binding[i] = bindings[js[0]]
                    binding[i, arity:] = 123  # padding is ignored
                expected[i] = any(np.array_equal(binding[i, :arity], bindings[j, :arity]) for j in js)
            np.testing.assert_array_equal(rl.is_applicable(table, states, chosen, binding, ids, pool=pool), expected)
    return exp


@pytest.mark.parametrize("path", PDDL, ids=lambda p: str(p.relative_to(DATA)))
def test_masks_pddl_samples(path):
    task = pddl_task(path)
    check_equal(task, [task.initial_state])


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("matching", ["fixed", "fc"])
def test_masks_walks(name, matching):
    task = text_task(name, matching=matching)
    states = walk(task, steps=4, walks=1)
    check_equal(task, states)
    check_equal(task, task.encode(states))


@pytest.mark.parametrize("name", ["cs-counters", "m-refuel-adl", "cs-hydropower", "m-woodworking"])
def test_masks_numeric(name):
    task = mymyr.Task.from_text(str(DATA / "numeric_tasks" / f"{name}.txt"))
    check_equal(task, walk(task, steps=4, walks=1))


def test_masks_numeric_effect_rules():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    domain = """(define (domain values)
      (:requirements :typing :adl :numeric-fluents :action-costs)
      (:types item) (:predicates (done ?x - item))
      (:functions (f ?x - item) (divisor ?x - item) (total-cost))
      (:action change :parameters (?x ?y - item)
        :precondition (>= (f ?x) 0)
        :effect (and (done ?x) (scale-down (f ?x) (divisor ?y))))
      (:action conflict :parameters (?x ?y - item)
        :precondition (and)
        :effect (and (assign (f ?x) 1) (increase (f ?y) 2)))
      (:action cost :parameters (?x - item) :precondition (and)
        :effect (increase (total-cost) (/ 1 (divisor ?x)))))"""
    problem = """(define (problem values-p) (:domain values) (:objects a b c - item)
      (:init (= (f a) 1) (= (f b) 2) (= (divisor a) 0) (= (divisor b) 1) (= (divisor c) 2)
             (= (total-cost) 0))
      (:goal (done a)) (:metric minimize (total-cost)))"""
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    check_equal(task, [task.initial_state])


@pytest.mark.parametrize("shape", ["forall-free", "exists-two", "or-ands", "nested-forall-exists", "numeric-or", "constants"])
@pytest.mark.parametrize("context", ["pre", "derived"])
def test_masks_adl(shape, context):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    from test_adl_conditions import SHAPES, build

    domain, problem, _ = build(SHAPES[shape], context)
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    check_equal(task, walk(task, steps=4, walks=1))


def test_masks_tables_suites_pool():
    first = pddl_task(DATA / "pddl/gripper/p01.pddl")
    second = pddl_task(DATA / "pddl/gripper/p02.pddl")
    table = rl.TaskTable([first, second])
    ids = np.array([1, 0, 1, 0], np.int32)
    pool = rl.ThreadPool(3)
    check_equal(table, table.initial_states(ids), ids, pool)
    suite = rl.TaskSuite([table, rl.TaskTable([text_task(SMALL_TASKS[0])])])
    ids = np.array([2, 1, 0, 2], np.int32)
    check_equal(suite, suite.initial_states(ids), ids, pool)


def test_masks_concurrent_first_use():
    task = text_task("openstacks-opt08-adl__p03")
    states = walk(task, steps=3, walks=1)
    exp = rl.expand(task, states)
    expected = np.asarray(rl.schema_masks(exp))
    schema = np.zeros(len(states), np.int32)
    objects = np.asarray(rl.prefix_masks(exp, schema, None, 0))
    pool = rl.ThreadPool(3)
    barrier = threading.Barrier(8)

    def query(_):
        barrier.wait()
        np.testing.assert_array_equal(rl.applicable_schemas(task, states), expected)
        for _ in range(5):
            np.testing.assert_array_equal(rl.applicable_schemas(task, states, pool=pool), expected)
            np.testing.assert_array_equal(rl.applicable_objects(task, states, schema, None, 0), objects)
            assert not rl.is_applicable(task, states, -1, np.zeros((len(states), task.label_width), np.int32)).any()

    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        list(executor.map(query, range(8)))
    if sys.version_info >= (3, 13) and sysconfig_free_threaded():
        assert not sys._is_gil_enabled()


def sysconfig_free_threaded():
    import sysconfig
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_masks_framework_cpu(framework):
    lib = pytest.importorskip(framework)
    task = text_task(SMALL_TASKS[0])
    rows = np.asarray(task.encode([task.initial_state]))
    exp = rl.expand(task, rows)
    schema = np.zeros(1, np.int32)
    binding = np.zeros((1, task.label_width), np.int32)
    if framework == "torch":
        convert = lambda a: lib.from_numpy(a.copy())
        states = convert(rows.view(np.int64))
        as_numpy = lambda a: a.numpy()
    else:
        import jax.numpy as jnp
        convert = jnp.asarray
        states = convert(rows.view(np.uint32))
        as_numpy = np.asarray
    actual = rl.applicable_schemas(task, states)
    assert isinstance(actual, lib.Tensor if framework == "torch" else lib.Array)
    np.testing.assert_array_equal(as_numpy(actual), rl.schema_masks(exp))
    np.testing.assert_array_equal(as_numpy(rl.applicable_objects(task, states, convert(schema), None, 0)),
                                  rl.prefix_masks(exp, schema, None, 0))
    np.testing.assert_array_equal(as_numpy(rl.is_applicable(task, states, convert(schema), convert(binding))),
                                  rl.is_applicable(task, rows, schema, binding))


def test_masks_validation_and_mutation():
    task = text_task(SMALL_TASKS[0])
    state = task.initial_state
    rows = np.asarray(task.encode([state])).copy()
    exp = rl.expand(task, rows)
    np.testing.assert_array_equal(rl.applicable_schemas(task, state), rl.schema_masks(exp))
    np.testing.assert_array_equal(rl.applicable_objects(task, rows, 0, None, 0),
                                  rl.applicable_objects(task, rows, np.array([0], np.int32), None, 0))
    assert not rl.applicable_objects(task, rows, -1, None, 0).any()
    assert not rl.is_applicable(task, rows, -1, np.zeros((1, 0), np.int32)).any()
    with pytest.raises(ValueError, match="depth"):
        rl.applicable_objects(task, rows, 0, None, task.label_width)
    arity = task.formalism.schemas[0].arity
    with pytest.raises(ValueError, match="depth"):
        rl.applicable_objects(task, rows, np.array([0], np.int32), None, arity)
    schema = next(i for i, s in enumerate(task.formalism.schemas) if s.arity >= 2)
    with pytest.raises(ValueError, match="prefix"):
        rl.applicable_objects(task, rows, schema, None, 1)
    with pytest.raises(ValueError, match="num_objects"):
        rl.applicable_objects(task, rows, 0, None, 0, num_objects=0)
    with pytest.raises((ValueError, TypeError), match="schema"):
        rl.applicable_objects(task, rows, np.zeros(2, np.int32), None, 0)
    with pytest.raises(TypeError, match="dtype"):
        rl.is_applicable(task, rows, 0, np.zeros((1, 2), np.int64))
    for successor in task.successor_states(state)[:3] + [state]:
        rows[:] = task.encode([successor])
        np.testing.assert_array_equal(rl.applicable_schemas(task, rows), rl.schema_masks(rl.expand(task, rows)))
    empty = rows[:0]
    assert rl.applicable_schemas(task, empty).shape == (0, task.num_schemas)
    assert rl.is_applicable(task, empty, 0, np.zeros((0, task.label_width), np.int32)).shape == (0,)
    # Device tags can be rejected before any memory is imported or accessed.
    class DeviceRows:
        def __dlpack_device__(self):
            return (2, 0)
    with pytest.raises(ValueError, match="CPU; pass host rows"):
        rl.applicable_schemas(task, DeviceRows())
    with pytest.raises(ValueError, match="CPU; pass host rows"):
        rl.applicable_objects(task, rows, DeviceRows(), None, 0)
    with pytest.raises(ValueError, match="CPU; pass host rows"):
        rl.is_applicable(task, rows, 0, DeviceRows())


@pytest.mark.parametrize("pool_size", [0, 3])
@pytest.mark.parametrize("scalar_schema", [False, True])
def test_masks_reject_narrow_bindings(pool_size, scalar_schema):
    task = text_task(SMALL_TASKS[0])
    states = [task.initial_state] * 3
    schema = next(i for i, s in enumerate(task.formalism.schemas) if s.arity >= 2)
    arity = task.formalism.schemas[schema].arity
    selected = schema if scalar_schema else np.array([-1, schema, task.num_schemas], np.int32)
    pool = rl.ThreadPool(pool_size) if pool_size else None
    for width in (0, arity - 1):
        binding = np.zeros((len(states), width), np.int32)
        with pytest.raises(ValueError, match="binding.*arity"):
            rl.is_applicable(task, states, selected, binding, pool=pool)
    binding = np.zeros((len(states), arity), np.int32)
    np.testing.assert_array_equal(rl.is_applicable(task, states, selected, binding, pool=pool),
                                  rl.is_applicable(task, states, selected, binding))
    invalid = np.array([-1, task.num_schemas, task.num_schemas + 1], np.int32)
    assert not rl.is_applicable(task, states, invalid, binding[:, :0], pool=pool).any()


def test_masks_childsnack_timing(capsys):
    data = os.environ.get("MYMYR_IPC_DATA")
    if not data:
        pytest.skip("set MYMYR_IPC_DATA to the IPC data directory")
    directory = pathlib.Path(data) / "childsnack-ipc/test"
    if not all((directory / name).is_file() for name in ("domain.pddl", "p30-hard.pddl")):
        pytest.skip("childsnack p30-hard data is absent")
    task = pddl_task(directory / "p30-hard.pddl")
    state = task.initial_state
    timings = []
    for schema, definition in enumerate(task.formalism.schemas):
        witness = next(iter(task.bindings(schema, state, limit=1)), None)
        arity = definition.arity
        for depth in range(arity):
            prefix = np.zeros((1, depth), np.int32)
            if witness is not None:
                prefix[0] = witness.binding[:depth]
            start = time.perf_counter_ns()
            mask = rl.applicable_objects(task, state, schema, prefix, depth)
            cold = (time.perf_counter_ns() - start) / 1e6
            samples = []
            for _ in range(20):
                start = time.perf_counter_ns()
                rl.applicable_objects(task, state, schema, prefix, depth)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            timings.append((task.schemas[schema], depth, cold, float(np.median(samples))))
            expected = np.zeros(task.num_objects, bool)
            partial = {i: int(o) for i, o in enumerate(prefix[0])}
            # The binding iterator restricts before enumeration and stops at the first full-label witness.
            for obj in range(task.num_objects):
                partial[depth] = obj
                expected[obj] = next(iter(task.bindings(schema, state, partial=partial, limit=1)), None) is not None
            np.testing.assert_array_equal(np.asarray(mask)[0], expected)
    start = time.perf_counter_ns()
    actual = rl.applicable_schemas(task, state)
    schemas_ms = (time.perf_counter_ns() - start) / 1e6
    expected = [next(iter(task.bindings(i, state, limit=1)), None) is not None for i in range(task.num_schemas)]
    np.testing.assert_array_equal(actual[0], expected)
    with capsys.disabled():
        print("\nchildsnack p30-hard masks (ms):")
        for name, depth, cold, median in timings:
            print(f"  {name}[{depth}]: cold={cold:.4f}, warm_median={median:.4f}")
        print(f"  applicable_schemas: cold={schemas_ms:.4f}")
