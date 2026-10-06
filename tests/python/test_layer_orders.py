"""Layer orderings of search.iw, siw, iw_pass and brfs (layer_order, seed, max_next_layer_states,
prefer_more_satisfied_goals): mimir's layer ordering strategies from Python."""

import pytest

from mymyr import search
from mymyr.search import Status

from conftest import text_task

ORDERS = ("in_order", "reverse", "randomized", "goal_count")


@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


@pytest.fixture(scope="module")
def blocks():
    return text_task("blocks__probBLOCKS-8-0")


def replays(task, plan):
    s = task.initial_state
    for a in plan:
        s = s.apply(a)
    return task.is_goal(s)


def iw_key(r):
    return r.status, [str(a) for a in r.plan], [(p.arity, p.expanded, p.generated) for p in r.passes]


def test_in_order_brfs_is_the_queued_brfs(gripper):
    q = search.brfs(gripper, witness_pruning=False, fingerprint=True)
    o = search.brfs(gripper, witness_pruning=False, fingerprint=True, layer_order="in_order")
    assert (o.states, o.expanded, o.generated, o.fingerprint) == (q.states, q.expanded, q.generated, q.fingerprint)


@pytest.mark.parametrize("order", ORDERS)
def test_ordered_brfs_finds_a_shallowest_goal(gripper, order):
    first = search.brfs(gripper, stop_at_goal=True)
    r = search.brfs(gripper, stop_at_goal=True, layer_order=order, seed=3)
    assert r.solved and len(r.plan) == len(first.plan)
    assert replays(gripper, r.plan)
    full = search.brfs(gripper, witness_pruning=False, layer_order=order, seed=3)
    assert full.states == search.brfs(gripper, witness_pruning=False).states  # the order changes ids, not states


def test_goal_count_order_and_direction(gripper):
    more = search.brfs(gripper, stop_at_goal=True, layer_order="goal_count")
    fewer = search.brfs(gripper, stop_at_goal=True, layer_order="goal_count", prefer_more_satisfied_goals=False)
    assert more.expanded <= fewer.expanded
    assert search.brfs(gripper, stop_at_goal=True, layer_order="reverse").expanded != search.brfs(
        gripper, stop_at_goal=True).expanded


def test_randomized_order_follows_the_seed(blocks):
    a = search.iw(blocks, max_arity=2, layer_order="randomized", seed=7)
    b = search.iw(blocks, max_arity=2, layer_order="randomized", seed=7)
    assert iw_key(a) == iw_key(b)
    keys = {str(iw_key(search.iw(blocks, max_arity=2, layer_order="randomized", seed=k))) for k in range(4)}
    assert len(keys) > 1


def test_truncated_layers(gripper):
    q = search.brfs(gripper, stop_at_goal=True)
    t = search.brfs(gripper, stop_at_goal=True, layer_order="goal_count", max_next_layer_states=4)
    assert t.expanded < q.expanded
    assert not t.solved or replays(gripper, t.plan)
    r = search.iw(gripper, max_arity=2, layer_order="goal_count", max_next_layer_states=3)
    full = search.iw(gripper, max_arity=2)
    assert sum(p.expanded for p in r.passes) <= sum(p.expanded for p in full.passes) or r.status != Status.SOLVED


def test_iw_siw_and_iw_pass_take_layer_orders(blocks):
    q = search.iw(blocks, max_arity=2)
    assert iw_key(search.iw(blocks, max_arity=2, layer_order="in_order")) == iw_key(q)
    for order in ORDERS:
        r = search.iw(blocks, max_arity=2, layer_order=order, seed=1)
        assert r.status in (Status.SOLVED, Status.EXHAUSTED)
        if r.solved:
            assert replays(blocks, r.plan)
        s = search.siw(blocks, max_arity=2, layer_order=order, seed=1)
        if s.solved:
            assert replays(blocks, s.plan)
        p = search.iw_pass(blocks, 1, layer_order=order, seed=1)
        assert [x.arity for x in p.passes] == [1]
    goal_count = search.iw(blocks, max_arity=2, layer_order="goal_count")
    assert goal_count.passes[-1].expanded != q.passes[-1].expanded or iw_key(goal_count) == iw_key(q)


def test_the_iw_family_takes_goal_count(blocks):
    r = search.liw(blocks, max_arity=1, layer_order="goal_count", prefer_more_satisfied_goals=False)
    assert [p.arity for p in r.passes] == [0, 1]
    a = search.abstracted_iw(blocks, width=1, layer_order="reverse")
    assert a.passes[0].expanded > 0


def test_invalid_layer_arguments(gripper):
    with pytest.raises(ValueError, match="layer_order must be"):
        search.brfs(gripper, layer_order="sideways")
    with pytest.raises(ValueError, match="positive"):
        search.brfs(gripper, layer_order="reverse", max_next_layer_states=0)
    with pytest.raises(ValueError, match="requires an ordered layer kind"):
        search.brfs(gripper, max_next_layer_states=5)
    r = search.iw(gripper, max_next_layer_states=5)  # the IW searches report the error as FAILED
    assert r.status == Status.FAILED and "max_next_layer_states" in r.message
    with pytest.raises(ValueError, match="flat or chunked"):
        search.brfs(gripper, layer_order="reverse", store="compact")
