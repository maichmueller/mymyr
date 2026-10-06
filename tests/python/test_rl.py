"""mymyr.rl with NumPy: expand (flat CSR + padded view) against the per-state API, capacity and width overflow, input
encodings and zero-copy input, expand_into, pools, goal masks, atom metadata, device arrays, and our DLPack export."""

import numpy as np
import pytest

import mymyr
from mymyr import _core, rl
from conftest import SMALL_TASKS, text_task, walk


def words_of(states, W):
    out = np.zeros((len(states), W), dtype=np.uint64)
    for i, s in enumerate(states):
        out[i, : s.num_words] = s.words
    return out


def reference(task, states):
    """Per-state API: labels, successor states, parents, offsets."""
    labels, succ, parent, offsets = [], [], [], [0]
    for i, s in enumerate(states):
        for a, t in task.successors(s):
            labels.append(a.label)
            succ.append(t)
            parent.append(i)
        offsets.append(len(labels))
    return labels, succ, parent, offsets


@pytest.mark.parametrize("name", SMALL_TASKS)
@pytest.mark.parametrize("atoms", ["lazy", "frozen"])
def test_expand_equals_the_per_state_api(name, atoms):
    task = text_task(name, atoms=atoms)
    states = walk(task, steps=10, seed=4)
    labels, succ, parent, offsets = reference(task, states)
    W = task.words
    exp = rl.expand(task, words_of(states, W), goal=True)
    assert not exp.overflow and exp.total == len(labels) == len(exp)
    assert exp.num_states == len(states) and exp.label_width == task.label_width
    assert exp.offsets.tolist() == offsets and exp.parent.tolist() == parent
    assert exp.counts.tolist() == np.diff(offsets).tolist()
    got = [(int(s), tuple(int(o) for o in b if o >= 0)) for s, b in zip(exp.schema, exp.binding)]
    assert got == labels
    for j, t in enumerate(succ):
        assert task.state(exp.succ[j]) == t
        assert exp.goal[j] == t.is_goal()
        assert exp.action(j).label == labels[j]
    assert exp.actions(len(states) - 1) == task.applicable_actions(states[-1])
    # the same from a list of States, with a pool, and with a temporary pool
    for kw in ({}, {"pool": rl.ThreadPool(3)}, {"threads": 2}):
        again = rl.expand(task, states, goal=True, **kw)
        assert np.array_equal(again.offsets, exp.offsets) and np.array_equal(again.binding, exp.binding)
        assert all(task.state(again.succ[j]) == succ[j] for j in range(len(succ)))


def test_padded_view(blocks):
    states = walk(blocks, steps=15, seed=9)
    exp = rl.expand(blocks, states, K=0, goal=True)
    p = exp.padded
    counts = exp.counts
    assert p.K == 1 << int(np.ceil(np.log2(max(counts.max(), 1)))) and not p.overflow
    assert p.succ.shape == (len(states), p.K, exp.words) and p.binding.shape == (len(states), p.K, exp.label_width)
    assert np.array_equal(p.count, counts)
    assert np.array_equal(p.mask, np.arange(p.K)[None, :] < counts[:, None])
    for i in range(len(states)):
        for k in range(p.K):
            j = p.index[i, k]
            if p.mask[i, k]:
                assert j == exp.offsets[i] + k
                assert np.array_equal(p.succ[i, k], exp.succ[j]) and p.schema[i, k] == exp.schema[j]
                assert np.array_equal(p.binding[i, k], exp.binding[j]) and p.goal[i, k] == exp.goal[j]
            else:
                assert j == -1 and p.schema[i, k] == -1 and not p.succ[i, k].any() and (p.binding[i, k] == -1).all()
    small = exp.pad(2)
    assert small.overflow == (counts.max() > 2) and np.array_equal(small.count, counts)


def test_capacity_and_width_overflow():
    task = text_task("logistics00__probLOGISTICS-6-1", atoms="lazy")
    states = walk(task, steps=10, seed=1)
    full = rl.expand(task, states)
    cut = rl.expand(task, states, capacity=full.total // 3)
    assert cut.overflow and cut.total == full.total and len(cut) == full.total // 3
    assert np.array_equal(cut.offsets, full.offsets)
    assert np.array_equal(cut.succ, full.succ[: len(cut)])
    need = max(task.state(r).num_words for r in full.succ)
    if need > 1:
        narrow = rl.expand(task, states, words=need - 1)
        assert narrow.overflow and narrow.words_needed == need
    # the automatic sizing grows and reruns
    big = rl.expand(task, [task.initial_state] * 3000)
    assert not big.overflow and big.total == 3000 * full.counts[0]


def test_input_encodings_are_read_in_place(blocks):
    states = walk(blocks, steps=12, seed=3)
    u64 = blocks.encode(states)
    base = rl.expand(blocks, u64)
    ptr, rows, words, stride, fw, bits, signed = _core._import_info(u64)
    assert ptr == u64.ctypes.data and (rows, words, stride, fw, bits) == (len(states), 2, 2, "numpy", 64)
    for view in (u64.view(np.int64), u64.view(np.uint32), u64.view(np.int32)):
        info = _core._import_info(view)
        assert info[0] == u64.ctypes.data and info[2] == 2
        exp = rl.expand(blocks, view)
        assert exp.succ.dtype == view.dtype  # outputs follow the input's word encoding
        assert np.array_equal(exp.succ.view(np.uint64), base.succ) and np.array_equal(exp.offsets, base.offsets)
    wide = np.zeros((len(states), 8), dtype=np.uint64)
    wide[:, :2] = u64
    strided = wide[::2, :2]  # row stride 16 words
    info = _core._import_info(strided)
    assert info[0] == wide.ctypes.data and info[3] == 16
    exp = rl.expand(blocks, strided)
    ref = rl.expand(blocks, u64[::2].copy())
    assert np.array_equal(exp.succ, ref.succ) and np.array_equal(exp.offsets, ref.offsets)
    one = rl.expand(blocks, u64[0])  # a 1-d row is one state
    assert one.num_states == 1 and np.array_equal(one.succ, base.succ[: base.offsets[1]])
    assert rl.expand(blocks, states[0]).total == base.offsets[1]
    with pytest.raises(TypeError):
        rl.expand(blocks, u64.astype(np.float64))
    with pytest.raises(TypeError):
        rl.expand(blocks, u64.view(np.uint32)[:, :3])  # odd number of 32-bit columns
    with pytest.raises(ValueError):
        rl.expand(blocks, np.full((1, 3), 1 << 63, dtype=np.uint64))  # unassigned slots


def test_outputs_are_views_of_one_block(blocks):
    exp = rl.expand(blocks, walk(blocks, steps=5))
    a, b = exp.succ, exp.succ
    assert np.shares_memory(a, b) and a.ctypes.data == b.ctypes.data
    assert a.ctypes.data % 64 == 0 and exp.parent.ctypes.data % 64 == 0


def test_expand_into_writes_caller_arrays():
    task = text_task("gripper__prob05")
    states = walk(task, steps=8)
    ref = rl.expand(task, states, goal=True)
    M, W, L = ref.total, ref.words, ref.label_width
    succ = np.zeros((M, W), np.uint64)
    parent, schema, offsets = np.zeros(M, np.int32), np.zeros(M, np.int32), np.zeros(len(states) + 1, np.int32)
    binding, goal = np.zeros((M, L), np.int32), np.zeros(M, np.bool_)
    r = rl.expand_into(task, states, succ=succ, parent=parent, schema=schema, binding=binding, goal=goal, offsets=offsets)
    assert r == {"total": M, "words_needed": ref.words_needed, "overflow": False, "capacity": M}
    for got, want in ((succ, ref.succ), (parent, ref.parent), (schema, ref.schema), (binding, ref.binding),
                      (goal, ref.goal), (offsets, ref.offsets)):
        assert np.array_equal(got, want)
    half = np.zeros((M // 2, 2 * W), np.uint32)  # 32-bit destination, too small
    r = rl.expand_into(task, states, succ=half)
    assert r["overflow"] and r["total"] == M and np.array_equal(half.view(np.uint64), ref.succ[: M // 2])
    with pytest.raises(TypeError):
        rl.expand_into(task, states, parent=np.zeros(M, np.float32))


@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "miconic-simpleadl__s10-2"])
def test_goal_masks_and_tests(name):
    task = text_task(name)
    states = walk(task, steps=30, seed=7, walks=4)
    words = task.encode(states)
    gpos, gneg = task.goal_masks()
    assert gpos.dtype == np.uint64 and not gpos.flags.writeable
    flags = np.array([s.is_goal() for s in states])
    assert np.array_equal(rl.is_goal(task, words), flags)
    assert np.array_equal(rl.goal_test(words, gpos, gneg), flags)
    counts = rl.goal_count(words, gpos, gneg)
    assert np.array_equal(counts == 0, flags) and counts.dtype == np.int32
    per_env_pos, per_env_neg = rl.goal_masks(task, np.zeros(len(states), np.int32))  # one row per env
    assert per_env_pos.shape == (len(states), task.max_words) and per_env_pos.flags.writeable
    assert (per_env_pos[:, : gpos.shape[0]] == gpos).all() and not per_env_pos[:, gpos.shape[0]:].any()
    assert np.array_equal(rl.goal_test(words, per_env_pos, per_env_neg), flags)
    per_env_pos[0] = 0  # relabel env 0: an empty goal holds everywhere
    per_env_neg[0] = 0
    assert rl.goal_test(words, per_env_pos, per_env_neg)[0]


def test_derived_goals_refuse_masks():
    task = text_task("philosophers__p03-phil4")
    with pytest.raises(ValueError, match="derived"):
        task.goal_masks()
    states = walk(task, steps=10)
    assert rl.is_goal(task, states).tolist() == [s.is_goal() for s in states]


@pytest.mark.parametrize("atoms", ["lazy", "frozen"])
def test_atom_metadata(atoms):
    task = text_task("logistics00__probLOGISTICS-6-1", atoms=atoms)
    walk(task, steps=20)
    m = task.atom_metadata()
    F = m["num_atoms"]
    assert F == task.num_atoms and m["num_objects"] == task.num_objects and m["atom_mode"] == (atoms == "frozen")
    off, args = m["atom_args_offsets"], m["atom_args"]
    for slot in range(F):
        atom = task.atom(slot)
        assert m["atom_pred"][slot] == atom.predicate_index
        assert args[off[slot]: off[slot + 1]].tolist() == atom.object_indices
        k = len(atom.object_indices)
        assert m["atom_args_padded"][slot, :k].tolist() == atom.object_indices
        assert (m["atom_args_padded"][slot, k:] == -1).all()
    groups, goff = m["pred_slots"], m["pred_slot_offsets"]
    assert goff[-1] == F and sorted(groups.tolist()) == list(range(F))
    for p in range(m["num_predicates"]):
        assert (m["atom_pred"][groups[goff[p]: goff[p + 1]]] == p).all()
    assert (m["pred_kind"][m["atom_pred"]] == 1).all()  # fluent
    assert m["static_pred"].shape[0] == m["num_static_atoms"] == len(task.formalism.static_init)
    again = task.atom_metadata()
    assert np.shares_memory(again["atom_pred"], m["atom_pred"])  # a cached snapshot, exported without copies
    assert not m["atom_pred"].flags.writeable


def test_device_arrays(blocks):
    d = blocks.device_arrays(version=1)
    assert d["version"] == 1 and d["num_objects"] == 8 and d["label_width"] == 2
    init = blocks.initial_state
    assert blocks.state(d["init"]) == init
    gpos, gneg = blocks.goal_masks()
    assert np.array_equal(d["goal_pos"][: gpos.shape[0]], gpos) and not d["goal_neg"].any()
    for slot in range(d["num_atoms"]):
        p = d["atom_pred"][slot]
        cid = d["pred_offset"][p]
        for i, o in enumerate(d["atom_args_padded"][slot, : d["pred_arity"][p]]):
            q = d["pos_begin"][p] + i
            assert d["pos_rank"][q, o] >= 0
            cid += d["pos_rank"][q, o] * d["pos_stride"][q]
        assert cid == d["atom_cid"][slot]
    assert d["schema_arity"].tolist() == [blocks.formalism.schemas[i].arity for i in range(4)]
    # version 2 (the default) is a superset: every version-1 array and scalar unchanged, plus section "plan"
    v2 = blocks.device_arrays()
    assert v2["version"] == 2 and v2["section_core"] == 1 and v2["section_plan"] == 2
    for k, v in d.items():
        if k != "version":
            assert np.array_equal(np.asarray(v2[k]), np.asarray(v)), k
    plan = sorted(k for k in v2 if k not in d)
    assert "plan_matcher" in plan and "plan_schema" in plan and "section_plan" in plan
    assert v2["plan_matcher"].dtype == np.uint32 and v2["plan_matcher"].ndim == 2
    with pytest.raises(ValueError):
        blocks.device_arrays(version=3)


def test_random_walks_are_seeded(blocks):
    a = rl.random_walks(blocks, 2000, 50, 7)
    assert a == rl.random_walks(blocks, 2000, 50, 7) and a["steps"] == 2000 and a["successors"] > 2000
    assert a != rl.random_walks(blocks, 2000, 50, 8)


def test_our_dlpack_export(blocks):
    exp = rl.expand(blocks, walk(blocks, steps=5), framework="dlpack")
    x = exp.succ
    assert isinstance(x, mymyr.DLArray) and x.dtype == "uint64" and x.__dlpack_device__() == (1, 0)
    arr = np.from_dlpack(x)
    assert arr.ctypes.data == x.data_ptr and arr.shape == x.shape
    legacy = x.__dlpack__()
    versioned = x.__dlpack__(max_version=(1, 0))
    assert "dltensor" in repr(legacy) and "dltensor_versioned" in repr(versioned)
    del legacy, versioned  # unconsumed capsules free themselves
    copied = np.from_dlpack(x, copy=True)
    assert copied.ctypes.data != x.data_ptr and np.array_equal(copied, arr)
    with pytest.raises(BufferError):
        x.__dlpack__(stream=5)
    with pytest.raises(BufferError):
        x.__dlpack__(dl_device=(2, 0))
    meta = blocks.atom_metadata(framework="dlpack")
    assert meta["atom_pred"].readonly and np.from_dlpack(meta["atom_pred"]).tolist() == blocks.atom_metadata()["atom_pred"].tolist()
    # JAX-style words: uint32 [.., 2W] over the same bytes
    words = exp.succ  # uint64 DLArray
    j = rl.expand(blocks, np.from_dlpack(words).view(np.uint32), framework="dlpack").succ
    assert j.dtype == "uint32" and j.shape[1] == 2 * words.shape[1]
