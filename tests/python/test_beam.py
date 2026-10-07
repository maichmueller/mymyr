"""The beam over the layers of brfs and the IW family (beam_width, beam_novelty, randomize_ties): every search that
takes the layer keywords, the two novelty modes on a hand-made task, random ties and the refusals."""

import pytest

import mymyr
from mymyr import search
from mymyr.search import Status

from conftest import text_task


@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


@pytest.fixture(scope="module")
def blocks():
    return text_task("blocks__probBLOCKS-8-0")


def walk_task(path, edges, goal):
    """A graph walk over o0..o4 (move(x, y): at(x), adj(x, y) -> at(y), not at(x), visited(y)), from o0."""
    lines = ["O 5", "P 3", "S 2 adj", "F 1 at", "F 1 visited", f"SI {len(edges)}"]
    lines += [f"0 {x} {y}" for x, y in edges]
    lines += ["FI 2", "1 0", "2 0", f"G {len(goal)}"] + [f"2 1 {g}" for g in goal]
    lines += ["A 1", "move 2", "L 2", "1 1 0", "0 1 0 1", "E 1", "0", "L 0", "F 3", "1 1 1", "1 0 0", "2 1 1"]
    path.write_text("\n".join(lines) + "\n")
    return mymyr.Task.from_text(str(path))


def replays(task, plan):
    s = task.initial_state
    for a in plan:
        s = s.apply(a)
    return task.is_goal(s)


def passes(r):
    return [(p.arity, p.expanded, p.generated, p.generated_in_tree) for p in r.passes]


def beam(width, mode="all_tested", **kw):
    return dict(layer_order="goal_count", beam_width=width, beam_novelty=mode, **kw)


def test_novelty_modes_differ_on_a_dropped_successor(tmp_path):
    # o0 -> o1 -> o3 -> o2 and o0 -> o2; the goal prefers o1, width 1 drops the successor at o2. all_tested marked
    # its atoms, so o2 reached via o3 is not novel; survivors_only marked nothing for it.
    task = walk_task(tmp_path / "walk.txt", [(0, 1), (0, 2), (1, 3), (3, 2)], [1, 4])
    a = search.iw_pass(task, 1, **beam(1))
    s = search.iw_pass(task, 1, **beam(1, "survivors_only"))
    assert passes(a) == [(1, 3, 4, 2)]
    assert passes(s) == [(1, 4, 4, 3)]


@pytest.mark.parametrize("mode", ["all_tested", "survivors_only"])
def test_iw_iw_pass_and_siw_take_a_beam(blocks, mode):
    plain = search.iw(blocks, max_arity=2, layer_order="goal_count")
    wide = search.iw(blocks, max_arity=2, **beam(1 << 30))  # wider than every layer: all_tested changes nothing
    if mode == "all_tested":
        assert passes(wide) == passes(plain)
    r = search.iw(blocks, max_arity=2, **beam(4, mode))
    assert r.status in (Status.SOLVED, Status.EXHAUSTED)
    assert r.passes[-1].expanded != plain.passes[-1].expanded
    if r.solved:
        assert replays(blocks, r.plan)
    p = search.iw_pass(blocks, 2, **beam(4, mode))
    assert passes(p) == passes(r)[-1:] or r.passes[-1].arity != 2
    s = search.siw(blocks, max_arity=2, **beam(4, mode))
    assert s.status in (Status.SOLVED, Status.EXHAUSTED)
    if s.solved:
        assert replays(blocks, s.plan)


@pytest.mark.parametrize("mode", ["all_tested", "survivors_only"])
def test_brfs_takes_a_beam(gripper, mode):
    full = search.brfs(gripper, stop_at_goal=True, layer_order="goal_count")
    for store in ("flat", "chunked"):
        r = search.brfs(gripper, stop_at_goal=True, store=store, **beam(3, mode))
        assert r.solved and replays(gripper, r.plan)
        assert r.expanded < full.expanded
    # brfs has no novelty table: the modes are the same
    a = search.brfs(gripper, stop_at_goal=True, **beam(3))
    assert (r.expanded, r.generated, [str(x) for x in r.plan]) == (a.expanded, a.generated, [str(x) for x in a.plan])


def test_the_iw_family_takes_a_beam(blocks):
    r = search.liw(blocks, max_arity=1, **beam(4))
    assert [p.arity for p in r.passes] == [0, 1]
    for mode in ("all_tested", "survivors_only"):
        a = search.abstracted_iw(blocks, width=1, **beam(4, mode))
        assert a.passes[0].expanded > 0
        p = search.projective_iw(blocks, **beam(4, mode))
        assert p.passes[0].expanded > 0
    narrow = search.abstracted_iw(blocks, width=1, **beam(1))
    assert narrow.passes[0].generated_in_tree < search.abstracted_iw(blocks, width=1).passes[0].generated_in_tree


def test_random_ties_follow_the_seed(gripper):
    def key(**kw):
        r = search.iw(gripper, max_arity=2, **beam(3, randomize_ties=True, **kw))
        return r.status, [str(a) for a in r.plan], passes(r)

    assert key(seed=5) == key(seed=5)
    plain = search.iw(gripper, max_arity=2, **beam(3))
    assert len({str(key(seed=k)) for k in range(4)} | {str((plain.status, [str(a) for a in plain.plan], passes(plain)))}) > 1
    b = search.brfs(gripper, stop_at_goal=True, **beam(3, randomize_ties=True, seed=5))
    assert b.expanded == search.brfs(gripper, stop_at_goal=True, **beam(3, randomize_ties=True, seed=5)).expanded


def test_every_ordered_kind_takes_a_beam(gripper):
    for order in ("in_order", "reverse", "randomized"):
        r = search.brfs(gripper, stop_at_goal=True, layer_order=order, beam_width=2, seed=1)
        assert not r.solved or replays(gripper, r.plan)
        assert search.iw(gripper, layer_order=order, beam_width=2, seed=1).status != Status.FAILED


def test_invalid_beam_arguments(gripper):
    with pytest.raises(ValueError, match="beam_novelty must be"):
        search.iw(gripper, **beam(4, "some"))
    with pytest.raises(ValueError, match="beam_novelty must be"):
        search.brfs(gripper, **beam(4, "some"))
    with pytest.raises(ValueError, match="positive"):
        search.brfs(gripper, **beam(0))
    with pytest.raises(ValueError, match="ordered layer kind"):
        search.brfs(gripper, beam_width=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        search.brfs(gripper, max_next_layer_states=8, **beam(4))
    with pytest.raises(ValueError, match="requires a beam"):
        search.brfs(gripper, layer_order="goal_count", beam_novelty="survivors_only")
    with pytest.raises(ValueError, match="randomize_ties requires"):
        search.brfs(gripper, layer_order="reverse", beam_width=4, randomize_ties=True)
    with pytest.raises(ValueError, match="flat or chunked"):
        search.brfs(gripper, store="compact", **beam(4))
    # the IW searches report the error as FAILED
    for r in (search.iw(gripper, beam_width=4), search.iw_pass(gripper, 1, **beam(0)),
              search.siw(gripper, **beam(4, max_next_layer_states=3)),
              search.liw(gripper, **beam(4, "survivors_only")),
              search.abstracted_iw(gripper, beam_width=4),
              search.projective_iw(gripper, layer_order="reverse", randomize_ties=True)):
        assert r.status == Status.FAILED and r.message


def test_a_beam_refuses_a_transition_ordering(gripper):
    order = search.LandmarkTransitionOrdering(search.approximate_fact_landmarks(gripper))
    r = search.iw(gripper, max_arity=1, transition_ordering=order, **beam(4))
    assert r.status == Status.FAILED and "transition ordering" in r.message
