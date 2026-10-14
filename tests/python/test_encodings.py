"""State encodings: a task's slot table, words and rows are a function of the task and its options (atoms='auto',
'reachable', 'dense'), equal across constructions, threads, processes and pickles; lazy slots stay inside their task
object. Rows over a table cost their own instance's width, and every numeric instance of a table stores its values
alike."""

import pickle

import numpy as np
import pytest

import mymyr
from mymyr import datasets, rl, search

from conftest import ROOT, text_task, walk
from python_children import run_code

GRIPPER = ROOT / "tests/data/pddl/gripper"
COUNTERS = ROOT / "tests/data/pddl/counters"


def needs_front_end():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")


def gripper_problem(path, balls):
    """A gripper problem with `balls` balls in rooma, all wanted in roomb."""
    names = " ".join(f"b{i}" for i in range(balls))
    init = " ".join(f"(ball b{i}) (at b{i} rooma)" for i in range(balls))
    goal = " ".join(f"(at b{i} roomb)" for i in range(balls))
    path.write_text(f"(define (problem g{balls}) (:domain gripper-strips) (:objects left right {names})\n"
                    f" (:init (room rooma) (room roomb) (gripper left) (gripper right) (free left) (free right)\n"
                    f"  (at-robby rooma) {init})\n (:goal (and {goal})))\n")
    return path


@pytest.fixture(scope="module")
def grippers(tmp_path_factory):
    """Problem files of 1, 30 and 100 balls (states of 1, 2 and 7 words)."""
    needs_front_end()
    d = tmp_path_factory.mktemp("gripper")
    return [gripper_problem(d / f"g{b:03d}.pddl", b) for b in (1, 30, 100)]


def encoding(task, states):
    return (task.atom_mode, task.words, task.num_atoms, task.encoding_fingerprint,
            [task.atom(s).__str__() for s in range(task.num_atoms)], np.asarray(task.encode(states)).tolist())


ENCODE_SCRIPT = """
import sys, mymyr
sys.path.insert(0, {tests!r})
from conftest import text_task, walk
for name in {names!r}:
    t = text_task(name)
    states = walk(t, steps=30, seed=7)
    print(name, t.atom_mode, t.words, t.encoding_fingerprint, mymyr.hash_rows(t.encode(states)).tolist())
"""


def test_two_processes_encode_alike():
    names = ["blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "openstacks-opt08-adl__p03"]
    script = ENCODE_SCRIPT.format(tests=str(ROOT / "tests/python"), names=names)
    outs = [run_code(script, capture_output=True).stdout for _ in range(2)]
    here = ""
    for name in names:
        t = text_task(name)
        states = walk(t, steps=30, seed=7)
        here += f"{name} {t.atom_mode} {t.words} {t.encoding_fingerprint} {mymyr.hash_rows(t.encode(states)).tolist()}\n"
    assert outs[0] == outs[1] == here


def test_constructions_and_pickles_encode_alike():
    for name in ("blocks__probBLOCKS-8-0", "philosophers__p03-phil4", "miconic-simpleadl__s10-2"):
        a, b = text_task(name), text_task(name)
        states = walk(a, steps=20, seed=3)
        enc = encoding(a, states)
        assert encoding(b, [b.state([str(x) for x in s.atoms()]) for s in states]) == enc
        a2, s2 = pickle.loads(pickle.dumps((a, states)))
        assert encoding(a2, s2) == enc
        assert a2.decode(a.encode(states)) == s2


@pytest.mark.parametrize("name", ["freecell__p02", "folding-opt23-adl__p01", "depot__p02", "philosophers__p03-phil4"])
def test_auto_mode_is_reachable_where_dense_is_wider(name):
    auto, dense, reach = text_task(name), text_task(name, atoms="dense"), text_task(name, atoms="reachable")
    info = auto.info
    assert info["reachability_complete"] and info["reachable_fluent"] == reach.num_atoms < dense.num_atoms
    assert dense.num_atoms == info["dense_fluent"] and dense.info["reachable_fluent"] == 0
    assert auto.atom_mode == ("dense" if dense.words <= reach.words else "reachable")
    assert auto.words == min(dense.words, reach.words)
    # the same states and searches in every mode
    states = walk(dense, steps=30, seed=4)
    for t in (auto, reach):
        mine = [t.state([str(x) for x in s.atoms()]) for s in states]
        assert [s.canonical_hash() for s in mine] == [s.canonical_hash() for s in states]
        assert [len(t.successor_states(s)) for s in mine] == [len(dense.successor_states(s)) for s in states]


def test_reachability_budget():
    name = "freecell__p02"
    assert text_task(name).atom_mode == "reachable"
    over = text_task(name, reachability_budget=10)
    assert over.atom_mode == "dense" and not over.info["reachability_complete"]
    assert over.options["reachability_budget"] == 10
    with pytest.raises(ValueError, match="reachability_budget"):
        text_task(name, atoms="reachable", reachability_budget=10)
    assert text_task(name, atoms="reachable", reachability_budget=0).info["reachability_complete"]


def test_tables_built_on_one_and_eight_threads_are_equal(grippers):
    domain = GRIPPER / "domain.pddl"
    files = [*grippers, GRIPPER / "p01.pddl", GRIPPER / "p02.pddl"]
    one = rl.TaskTable.from_pddl(domain, files, threads=1)
    eight = rl.TaskTable.from_pddl(domain, files, threads=8)
    assert [t.encoding_fingerprint for t in one.tasks] == [t.encoding_fingerprint for t in eight.tasks]
    assert list(one.instance_words) == list(eight.instance_words) and one.words == eight.words
    assert np.array_equal(one.initial_states(), eight.initial_states())
    ids = np.arange(len(files), dtype=np.int32)
    x, y = rl.expand(one, one.initial_states(), ids), rl.expand(eight, eight.initial_states(), ids)
    assert np.array_equal(x.succ, y.succ) and np.array_equal(x.binding, y.binding)


def test_rows_cost_their_own_instance_width(grippers):
    table = rl.TaskTable.from_pddl(GRIPPER / "domain.pddl", grippers)
    widths = list(table.instance_words)
    assert widths == [1, 2, 7] and table.words == 7
    small = table.tasks[0]
    states = walk(small, steps=6, seed=1)
    ids = np.zeros(len(states), np.int32)
    narrow = np.asarray(small.encode(states))
    base = rl.expand(table, narrow, ids)
    assert base.words == 1
    # rows as wide as the table: the padding is not read, the successors keep the batch's width
    wide = np.zeros((len(states), table.words), np.uint64)
    wide[:, :1] = narrow
    wide[:, 1:] = 12345
    again = rl.expand(table, wide, ids)
    assert again.words == 1 and np.array_equal(again.succ, base.succ)
    assert rl.is_goal(table, wide, ids).tolist() == [s.is_goal() for s in states]
    assert small.decode(wide) == states
    fixed = rl.expand(table, narrow, ids, words=table.words)
    assert fixed.words == table.words and np.array_equal(fixed.succ[:, :1], base.succ) and not fixed.succ[:, 1:].any()
    # a mixed batch is as wide as its widest instance
    mixed = table.initial_states(np.array([0, 1, 0], np.int32))
    assert mixed.shape == (3, 2)
    assert rl.expand(table, mixed, np.array([0, 1, 0], np.int32)).words == 2
    assert table.initial_states().shape == (3, 7)
    # the pool: each batch as wide as its envs' instances (and those they autoreset into; kept without a sampler)
    pool = rl.CpuEnvPool(table, 8, threads=2, seed=1, sampler=None)
    assert pool.words is None
    assert pool.reset(task_ids=np.zeros(8, np.int32)).words == 1
    b = pool.step()
    assert b.words == 1 and b.states.shape == (8, 1) and b.final_states.shape == (8, 1)
    assert pool.step(next_task_ids=np.full(8, 1, np.int32)).words == 2
    fixed_pool = rl.CpuEnvPool(table, 8, threads=2, seed=1, words=table.words)
    assert fixed_pool.words == 7 and fixed_pool.reset(task_ids=np.zeros(8, np.int32)).states.shape == (8, 7)
    # one env's trajectory does not depend on the width of its batch
    pool = rl.CpuEnvPool(table, 8, threads=2, seed=1)
    fixed_pool = rl.CpuEnvPool(table, 8, threads=2, seed=1, words=table.words)
    pool.reset(task_ids=np.zeros(8, np.int32))
    fixed_pool.reset(task_ids=np.zeros(8, np.int32))
    for _ in range(5):
        p, q = pool.step(), fixed_pool.step()
        assert np.array_equal(np.asarray(q.states)[:, : p.words], p.states) and not np.asarray(q.states)[:, p.words:].any()
        assert np.array_equal(p.reward, q.reward) and np.array_equal(p.schema, q.schema)


def test_one_numeric_storage_per_table():
    needs_front_end()
    domain = COUNTERS / "domain.pddl"
    files = [COUNTERS / "p01.pddl", COUNTERS / "p03-half.pddl", COUNTERS / "p02.pddl"]
    ints, half = mymyr.Task.from_pddl(domain, files[0]), mymyr.Task.from_pddl(domain, files[1])
    assert ints.numeric_storage == "i32" and half.numeric_storage == "f64"
    with pytest.raises(ValueError, match="one storage"):
        rl.TaskTable([ints, half])
    with pytest.raises(ValueError, match="int32"):
        mymyr.Task.from_pddl(domain, files[1], numeric_storage="i32")
    table = rl.TaskTable.from_pddl(domain, files)
    assert table.numeric_storage == "f64" and [t.numeric_storage for t in table.tasks] == ["f64"] * 3
    assert rl.TaskTable.from_pddl(domain, [files[0], files[2]]).numeric_storage == "i32"
    S = max(t.numeric_slots for t in table.tasks)
    want = np.full((3, S), np.nan)
    for i, t in enumerate(table.tasks):
        want[i, : t.numeric_slots] = t.initial_state.numeric_values()
    got = rl.numeric_values(table, table.initial_states(), np.arange(3, dtype=np.int32))
    assert got.dtype == np.float64 and np.array_equal(got, want, equal_nan=True)
    assert got[1, 0] == 0.5
    # env observations: the decoded values follow the per-state API along each env's moves
    pool = rl.CpuEnvPool(table, 6, threads=2, seed=3, sampler=None)  # autoresets keep the instances
    ids = np.array([0, 1, 2, 1, 0, 2], np.int32)
    obs = pool.reset(task_ids=ids)
    cur = [table.tasks[i].initial_state for i in ids]
    for _ in range(12):
        vals = rl.numeric_values(table, obs.states, obs.task_ids)
        for r in range(6):
            t = table.tasks[int(ids[r])]
            assert np.array_equal(vals[r, : t.numeric_slots], cur[r].numeric_values())
            assert np.isnan(vals[r, t.numeric_slots:]).all()
        obs = pool.step()
        for r in range(6):
            t = table.tasks[int(ids[r])]
            if obs.schema[r] >= 0:
                label = (int(obs.schema[r]), tuple(int(o) for o in obs.binding[r] if o >= 0))
                acts = [a for a in t.applicable_actions(cur[r]) if (a.label[0], tuple(a.label[1])) == label]
                cur[r] = t.apply(cur[r], acts[0])
            if obs.terminated[r] or obs.truncated[r]:
                cur[r] = t.initial_state


def test_lazy_slots_stay_inside_their_task():
    lazy = text_task("depot__p02", atoms="lazy")
    states = walk(lazy, steps=20, seed=2)
    rows = lazy.encode(states)
    with pytest.raises(ValueError, match="lazy atom slots"):
        lazy.decode(rows)
    with pytest.raises(ValueError, match="lazy atom slots"):
        lazy.state(np.asarray(rows)[0])
    with pytest.raises(ValueError, match="lazy atom slots"):
        rl.TaskTable([lazy])
    with pytest.raises(ValueError, match="lazy atom slots"):
        rl.expand(lazy, states)
    assert lazy.applicable_actions(states[-1])  # States carry their task
    arrays = datasets.generate(lazy, threads=2).space.arrays()
    assert "state_words" not in arrays and "forward_offsets" in arrays
    fixed = datasets.generate(text_task("depot__p02"), threads=2).space.arrays()
    assert "state_words" in fixed and np.array_equal(arrays["forward_targets"], fixed["forward_targets"])


def test_options_reach_the_table_loaders(grippers):
    domain = GRIPPER / "domain.pddl"
    table = rl.TaskTable.from_pddl(domain, grippers, atoms="dense", matching="fc", reachability_budget=0)
    assert all(t.atom_mode == "dense" and t.options["matching"] == "fc" for t in table.tasks)
    with pytest.raises(ValueError):
        rl.TaskTable.from_pddl(domain, grippers, atoms="lazy")
    with pytest.raises(ValueError):
        rl.TaskTable.from_pddl(domain, grippers, numeric_quantum=-0.5)


def lazy_after_a_search(name):
    """A lazy task whose slots a randomized search numbered (in an order unlike the canonical one)."""
    task = text_task(name, atoms="lazy")
    search.brfs(task, max_states=3000, layer_order="randomized", seed=3, stop_at_goal=False)
    return task


def test_heuristics_and_atom_listings_do_not_depend_on_lazy_slots():
    fixed, lazy = text_task("depot__p02"), lazy_after_a_search("depot__p02")
    kinds = ("ff", "add", "max", "set_additive", "h2", "goal_count", "lmcut", "lmcount")
    hf = {k: search.Heuristic(fixed, k) for k in kinds}
    hl = {k: search.Heuristic(lazy, k) for k in kinds}
    a, b = fixed.initial_state, lazy.initial_state
    for s in walk(fixed, steps=60, seed=4, walks=1)[1:] + [None]:
        assert [str(x) for x in a.atoms()] == [str(x) for x in b.atoms()]
        assert str(a) == str(b)
        for k in kinds:
            assert hf[k].evaluate(a) == hl[k].evaluate(b), k
        assert sorted(map(str, hf["ff"].relaxed_plan(a))) == sorted(map(str, hl["ff"].relaxed_plan(b)))
        if s is None:
            break
        label = next(str(x) for x in fixed.applicable_actions(a) if fixed.apply(a, x) == s)
        a, b = s, lazy.apply(b, next(x for x in lazy.applicable_actions(b) if str(x) == label))
    for run in (lambda t: search.gbfs(t, heuristic="ff", max_expanded=5000),
                lambda t: search.beam(t, width=32, max_expanded=5000)):
        rf, rl_ = run(fixed), run(lazy)
        assert [str(x) for x in rf.plan] == [str(x) for x in rl_.plan] and rf.stats.expanded == rl_.stats.expanded


def test_tuple_graph_atoms_do_not_depend_on_lazy_slots():
    fixed, lazy = text_task("gripper__prob05"), lazy_after_a_search("gripper__prob05")
    sf, sl = datasets.state_space(fixed), datasets.state_space(lazy)
    for root in range(0, sf.num_states, sf.num_states // 5):
        gf, gl = datasets.tuple_graph(sf, root, width=2), datasets.tuple_graph(sl, root, width=2)
        assert len(gf) == len(gl)
        for v in range(len(gf)):
            assert [str(x) for x in gf.atoms(v)] == [str(x) for x in gl.atoms(v)]


@pytest.mark.parametrize("name", ["logistics00__probLOGISTICS-6-1", "philosophers__p03-phil4", "depot__p02",
                                  "miconic-simpleadl__s10-2", "openstacks-opt08-adl__p03"])
@pytest.mark.parametrize("atoms", ["auto", "reachable", "dense"])
def test_fixed_slots_of_a_predicate_are_one_ascending_range(name, atoms):
    task = text_task(name, atoms=atoms)
    m = task.atom_metadata()
    P = m["num_predicates"]
    for kind, groups in (("atom", "pred_slot"), ("derived", "derived_pred_slot")):
        pred, args, off = m[kind + "_pred"], m[kind + "_args_padded"], m[groups + "_offsets"]
        count = pred.shape[0]
        assert np.array_equal(m[groups + "s"], np.arange(count))
        assert off.shape == (P + 1,) and off[0] == 0 and off[-1] == count
        for p in range(P):
            assert (pred[off[p]: off[p + 1]] == p).all()
            k = int(m["pred_arity"][p])
            tuples = [tuple(row[:k]) for row in args[off[p]: off[p + 1]].tolist()]
            assert all(a < b for a, b in zip(tuples, tuples[1:])), (kind, p)
    assert m["atom_pred"].shape[0] == task.num_atoms
    if task.has_axioms:
        assert m["derived_pred"].shape[0] > 0
