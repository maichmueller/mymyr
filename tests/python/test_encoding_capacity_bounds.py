"""FlatRelationEncoder.capacity_bounds: static bounds that no batch exceeds. Over random walks of the IPC 2023 train
problems (tests/data/ipc) every batch of N rows, encoded exactly or with the bounds as its capacities, stays within the
bounds for N graphs per relation and per field: with the instances' goals and with goals of the rows, with expansion
rows and without, for two option sets. The environments take the bounds as obs_capacities (on the CPU here, on a
device in test_rl_capacity_bounds.py)."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, walk

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the PDDL front end", allow_module_level=True)

IPC = ROOT / "tests" / "data" / "ipc"
DOMAINS = sorted(p.name for p in IPC.iterdir() if (p / "train" / "domain.pddl").exists())
OPTIONS = {
    "targets": dict(target_sources=("goal", "action")),
    "virtual": dict(predicate_virtual_nodes=True, nullary="arity0", statics="shared", layout="relation",
                    goal_derivations=("plain", "satisfied", "unsatisfied")),
}


def ipc_table(name, problems=2):
    train = IPC / name / "train"
    files = sorted((p for p in train.glob("p*.pddl")), key=lambda p: p.stat().st_size)[:problems]
    return rl.TaskTable([mymyr.Task.from_pddl(train / "domain.pddl", p) for p in files])


def walk_rows(table, seed=0):
    """Rows of random walks of every instance, and their task ids."""
    rows, ids = [], []
    for i, task in enumerate(table.tasks):
        states = walk(task, steps=10, walks=2, seed=seed + i)
        rows += [np.asarray(task.encode([s]))[0] for s in states]
        ids += [i] * len(states)
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width), np.uint64)
    for k, r in enumerate(rows):
        out[k, : len(r)] = r
    return out, np.asarray(ids, np.int32)


def within(batch, bounds, schema):
    """Every relation and every field of `batch` is at most its bound."""
    counts = np.asarray(batch.relation_counts).sum(axis=0)
    for name, n in zip(schema.relation_names, counts.tolist()):
        assert n <= bounds.relation_tuples[name], (name, n, bounds.relation_tuples[name])
    sizes = batch.sizes if batch.sizes is not None else {
        "entities": int(batch.entity_offsets[-1]), "args": len(batch.relation_args),
        "targets": len(batch.target_entity) if "targets" in bounds else 0}
    for key in ("entities", "args", "targets"):
        if key in bounds:
            assert sizes[key] <= bounds[key], (key, sizes[key], bounds[key])
    return counts


@pytest.mark.parametrize("variant", sorted(OPTIONS))
@pytest.mark.parametrize("name", DOMAINS)
def test_walks_stay_within_the_bounds(name, variant):
    table = ipc_table(name)
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS[variant])
    rows, ids = walk_rows(table)
    n = len(rows)
    bounds = encoder.capacity_bounds(table, n, drop_noops=True)
    assert not bounds.unbounded and bounds.reason == ""
    plain = encoder.capacity_bounds(table, n, actions=False)
    expansion = rl.expand(table, rows, ids)
    assert int(expansion.total) <= bounds["successors"]
    goals = rl.goal_masks(table, ids)
    for actions, goal_rows, b in ((expansion, None, bounds), (None, None, plain), (expansion, goals, bounds),
                                  (None, goals, plain)):
        caps = {k: v for k, v in b.items() if k != "successors"}
        exact = encoder.encode(rows, ids, actions=actions, goals=goal_rows)
        within(exact, b, encoder.schema)
        fixed = encoder.encode(rows, ids, actions=actions, goals=goal_rows, capacities=caps)
        assert not fixed.sizes["overflow"]
        counts = within(fixed, b, encoder.schema)
        assert np.array_equal(counts, np.asarray(exact.relation_counts).sum(axis=0))
    static = [r for r, lane in zip(encoder.relation_names, encoder.relation_lanes) if lane == "static"]
    assert all(bounds.relation_exact[r] for r in static)
    # the others are upper bounds (a relation without tuples in any instance has the exact bound 0)
    assert all(lane in ("static", "numeric") for r, lane in zip(encoder.relation_names, encoder.relation_lanes)
               if bounds.relation_exact[r] and bounds.relation_tuples[r])


def test_static_relations_are_exact_and_scale_with_the_graphs():
    table = ipc_table("rovers-ipc", problems=1)
    encoder = encoding.FlatRelationEncoder(table)
    one, many = encoder.capacity_bounds(table, 1), encoder.capacity_bounds(table, 7)
    assert many["graphs"] == 7 and many["entities"] == 7 * one["entities"] and many["args"] == 7 * one["args"]
    assert all(many.relation_tuples[r] == 7 * one.relation_tuples[r] for r in encoder.relation_names)
    rows, ids = walk_rows(table)
    batch = encoder.encode(rows[:1], ids[:1])
    for r, lane, n in zip(encoder.relation_names, encoder.relation_lanes, np.asarray(batch.relation_counts)[0]):
        if lane in ("static", "numeric"):
            assert one.relation_exact[r] and one.relation_tuples[r] == n, r
    assert one.exact == {"graphs": True, "entities": False, "args": False}


def test_bounds_read_as_capacities_and_pickle():
    table = ipc_table("blocksworld-ipc")
    encoder = encoding.FlatRelationEncoder(table, target_sources=("goal",))
    bounds = encoder.capacity_bounds(table, 4)
    assert isinstance(bounds, encoding.CapacityBounds) and isinstance(bounds, dict)
    assert set(bounds) == {"graphs", "entities", "args", "targets"}
    assert list(bounds.relation_tuples) == list(encoder.relation_names)
    assert set(bounds.relation_exact) == set(encoder.relation_names)
    again = pickle.loads(pickle.dumps(bounds))
    assert again == bounds and again.relation_tuples == bounds.relation_tuples and again.exact == bounds.exact
    # successors only with the no-op drop (without it the entity capacity bounds the kept rows)
    assert "successors" not in bounds and "successors" in encoder.capacity_bounds(table, 4, drop_noops=True)
    assert "successors" not in encoder.capacity_bounds(table, 4, actions=False, drop_noops=True)


def test_a_suite_bounds_the_tables_of_the_encoders_domain():
    blocks, ferry = ipc_table("blocksworld-ipc"), ipc_table("ferry-ipc")
    suite = rl.TaskSuite([blocks, ferry])
    encoder = encoding.FlatRelationEncoder(ferry)
    assert encoder.capacity_bounds(suite, 3) == encoder.capacity_bounds(ferry, 3)
    other = encoding.FlatRelationEncoder(ipc_table("miconic-ipc"))
    with pytest.raises(ValueError, match="no table of this encoder's domain"):
        other.capacity_bounds(suite, 3)


def test_history_and_invalid_arguments():
    table = ipc_table("ferry-ipc", problems=1)
    every = encoding.FlatRelationEncoder(table, history=True, target_sources=("history",))
    with pytest.raises(ValueError, match="encodes every history step"):
        every.capacity_bounds(table, 2)
    three = encoding.FlatRelationEncoder(table, history=True, history_max_steps=3)
    a, b = three.capacity_bounds(table, 2), three.capacity_bounds(table, 2, history=2)
    assert a["history"] == 6 and b["history"] == 4 and every.capacity_bounds(table, 2, history=5)["history"] == 10
    hist = [r for r in three.relation_names if r.endswith("[hist]")]
    assert all(b.relation_tuples[r] * 3 == a.relation_tuples[r] * 2 for r in hist)
    for graphs in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="graphs"):
            three.capacity_bounds(table, graphs)
