"""mymyr.search: the IW family variants, landmarks and relaxed reachability from Python.

Replays the fork's recorded runs (tests/data/iw_variants/fork_golden.json, the C++ IwVariantGolden test's fixture: 508 rows over
the 24 configurations checked for parity, including the fork's name hashes of landing states and co-occurrence)
through the bindings; checks determinism of the randomized searches across runs and thread counts, the control surface
(budgets, cancellation, observers with on_transition and make_worker, callback errors, numeric tasks), the landmark and
reachability API against the searches and BrFS, and 8 Python threads running the variants at once on shared tasks.
"""

import json
import os
import pathlib
import sys
import threading

import pytest

import mymyr
from mymyr import search
from mymyr.search import ActionOrdering, Status, TransitionOutcome

from conftest import ROOT, TASKS, text_task

GOLDEN = ROOT / "tests/data/iw_variants/fork_golden.json"
WORK = pathlib.Path(os.environ.get("MYMYR_WORK", ROOT / ".work"))
FORK_DATA = pathlib.Path(os.environ["MYMYR_FORK_DATA"]) if "MYMYR_FORK_DATA" in os.environ else None
NUMERIC = ROOT / "tests/data/numeric_tasks"


# ------------------------------------------------------------------------------------------------ fork conventions

FORK_STATUS = {Status.SOLVED: "solved", Status.EXHAUSTED: "failed", Status.OUT_OF_STATES: "out_of_states",
               Status.OUT_OF_TIME: "out_of_time", Status.CANCELLED: "canceled", Status.FAILED: "error",
               Status.UNSOLVABLE: "unsolvable"}
ORDERING = {"in_order": ActionOrdering.IN_ORDER, "randomized": ActionOrdering.RANDOMIZED,
            "dgaf": ActionOrdering.DIRECT_GOAL_ACHIEVER_FIRST, "regression": ActionOrdering.GOAL_REGRESSION_RELEVANCE,
            "mixed": ActionOrdering.MIXED_REGRESSION_RANDOM}
NO_VALUE = 4294967295  # the fork's and the C++ API's "none" for u32 fields


def status_name(s, ladder):
    """The fork's status names per search kind (bench/cpp/mymyr_iw_variants.cpp, test_iw_variants_golden.cpp)."""
    if s == Status.EXHAUSTED:
        return "failed" if ladder else "exhausted"
    if s == Status.FAILED:
        return "failed"
    return FORK_STATUS[s]


def per_pass(passes, arity=None):
    return ";".join(f"{arity if arity is not None else p.arity}:{p.expanded}/{p.generated}/{p.generated_in_tree}"
                    for p in passes)


def fnv1a(text, h=0xCBF29CE484222325):
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def chain_hash(lines):
    h = 0xCBF29CE484222325
    for x in lines:
        h = fnv1a(x + "\n", h)
    return f"{h:x}"


def rollout_hashes(r):
    """The name hashes fork_iw_variants and mymyr_iw_variants print for a rollout (bench/cpp/mymyr_iw_variants.cpp rollout_json)."""
    out = {"reached_fluent_hash": chain_hash(sorted(str(a) for a in r.reached_fluent_atoms))}
    states = [" ".join(sorted(str(a) for a in s.atoms)) for s in r.landing_states]
    out["landing_states_hash"] = chain_hash(sorted(states))
    out["landing_by_atom_hash"] = chain_hash(sorted(f"{a} -> {states[i]}" for a, i in r.landing_state_by_atom.items()))
    lines = [f"{a}:" + "".join(" " + x for x in sorted(str(b) for b in row)) for a, row in r.co_occurrence.items()]
    out["co_occurrence_hash"] = chain_hash(sorted(lines))
    return out


def rollout_stats(s):
    return {k: getattr(s, k) for k in ("rollouts", "generated", "expanded", "feature_depth_improvements", "case1",
                                       "case2", "case3", "case4", "solved_propagations", "dead_ends",
                                       "depth_bound_prunings", "incumbent_bound_prunings", "max_rollout_depth",
                                       "tree_nodes")}


def parse_args(args):
    a = {"algo": None, "k": 1, "base": False, "preserve_goal": True, "lm_disjunctive": False, "lm_all_private": False,
         "ordering": "in_order", "seed": 0, "seeds": "", "workers": 4, "threads": 1, "orderings": "",
         "max_next_layer_states": None, "max_states": None}
    it = iter(args)
    for x in it:
        key = x[2:].replace("-", "_")
        if key in ("base", "lm_disjunctive", "lm_all_private"):
            a[key] = True
        elif key == "no_preserve_goal":
            a["preserve_goal"] = False
        elif key == "goal_atom":
            next(it)  # resolved in the row's goal_atom
        else:
            v = next(it)
            a[key] = int(v) if key in ("k", "seed", "workers", "threads", "max_next_layer_states", "max_states") else v
    return a


# ------------------------------------------------------------------------------------------------ golden replay

def golden_rows():
    if not GOLDEN.is_file():
        return {}
    rows = {}
    for r in json.loads(GOLDEN.read_text())["rows"]:
        rows.setdefault(r["task"], []).append(r)
    return rows


ROWS = golden_rows()
_TASKS = {}


def golden_task(name, src):
    if name in _TASKS:
        return _TASKS[name]
    if src["kind"] == "text":
        path = ROOT / src["path"]
        if not path.is_file():
            pytest.skip(f"{path} not found")
        task = mymyr.Task.from_text(str(path))
    else:
        if not hasattr(mymyr, "Domain"):
            pytest.skip("built without the loki front end")
        root = {"fork_data": FORK_DATA, "source": ROOT}.get(src.get("root"), WORK)
        if root is None:
            pytest.skip(f"PDDL of {name} not found (set MYMYR_FORK_DATA)")
        d, p = root / src["domain"], root / src["problem"]
        if not d.is_file() or not p.is_file():
            pytest.skip(f"PDDL of {name} not found ({p})")
        task = mymyr.Task.from_pddl(d, p)
    _TASKS[name] = task
    return task


def fluent_atoms(task, names):
    """The atoms among `names` that are fluent atoms of the task (the C++ test's Names.parse)."""
    out = []
    for x in names:
        try:
            a = task.atom(x)
        except (KeyError, ValueError):
            continue
        if a.kind == "fluent":
            out.append(a)
    return out


def run_row(task, row):
    a = parse_args(row["args"])
    e = row["expect"]
    what = f"{row['task']} {row['config']}"
    kw = {}
    if "goal_atom" in row:
        kw["goal"] = [[row["goal_atom"]]]
    lm = {}
    if "landmarks" in row:
        graph = search.FactLandmarkGraph(task, fluent_atoms(task, row["landmarks"]["facts"]),
                                         [fluent_atoms(task, s) for s in row["landmarks"]["sets"]])
        lm = dict(landmarks=graph, disjunctive=a["lm_disjunctive"], all_private=a["lm_all_private"])
    if a["algo"] in ("aiw", "liw"):
        if a["algo"] == "aiw":
            r = search.abstracted_iw(task, width=a["k"], base_abstracted=a["base"],
                                     preserve_goal_atoms=a["preserve_goal"], **lm, **kw)
            assert per_pass(r.passes, a["k"]) == e["per_pass"], what
        else:
            r = search.liw(task, max_arity=a["k"], **lm, **kw)
            assert per_pass(r.passes) == e["per_pass"], what
        assert status_name(r.status, a["algo"] == "liw") == e["status"], what
        if r.solved:
            assert len(r.plan) == e["plan_len"], what
    elif a["algo"] == "rollout_iw":
        r = search.rollout_iw(task, ordering=ORDERING[a["ordering"]], seed=a["seed"], max_states=a["max_states"], **kw)
        assert status_name(r.status, False) == e["status"], what
        assert rollout_stats(r.statistics) == e["rollout"], what
        assert r.root_solved == e["root_solved"], what
        if r.solved:
            assert len(r.plan) == e["plan_len"], what
    elif a["algo"] == "rollouts":
        seeds = [int(s) for s in a["seeds"].split(",")]
        b = search.find_rollouts_parallel(task, seeds, max_arity=a["k"], num_threads=4,
                                          max_next_layer_states=a["max_next_layer_states"], report_landing_states=True,
                                          report_co_occurrence=True, **kw)
        assert len(b) == len(e["rollouts"]), what
        for x, y in zip(b, e["rollouts"]):
            w = f"{what} seed {y['seed']}"
            assert x.seed == y["seed"], w
            assert status_name(x.status, True) == y["status"], w
            assert per_pass(x.passes) == y["per_pass"], w
            if x.solved:
                assert len(x.plan) == y["plan_len"], w
            assert x.num_states == y["num_states"], w
            assert len(x.reached_fluent_atoms) == y["reached_fluent"], w
            assert len(x.reached_derived_atoms) == y["reached_derived"], w
            assert len(x.landing_states) == y["landing_states"], w
            assert sum(s.direct_dead_end for s in x.landing_states) == y["landing_dead_ends"], w
            assert sum(len(row) for row in x.co_occurrence.values()) == y["co_occurrence_pairs"], w
            if row["source"]["kind"] == "pddl":  # the fork's names (text exports number the objects)
                h = rollout_hashes(x)
                assert h == {k: y[k] for k in h}, w
    elif a["algo"] == "portfolio":
        orderings = []
        for s in filter(None, a["orderings"].split(",")):
            kind, _, seed = s.partition(":")
            orderings.append((kind if kind != "dgaf" else "direct_goal_achiever_first", int(seed or 0)))
        r = search.atomic_goal_portfolio(task, num_rollout_workers=a["workers"], num_threads=a["threads"],
                                         rollout_orderings=orderings, **kw)
        assert status_name(r.status, True) == e["status"], what
        assert (r.plan_length if r.solved else 0) == e["plan_length"], what
        assert r.certified_optimal == e["certified_optimal"], what
        assert (NO_VALUE if r.winning_worker is None else r.winning_worker) == e["winning_worker"], what
        cs = "in_progress" if r.certifier_status is None else status_name(r.certifier_status, True)
        assert cs == e["certifier_status"], what
        assert (NO_VALUE if r.iw_completed_depth is None else r.iw_completed_depth) == e["iw_completed_depth"], what
        assert r.iw_lower_bound == e["iw_lower_bound"], what
        assert r.total_expansions == e["total_expansions"], what
        assert (r.certifier.total.expanded, r.certifier.total.generated) == (e["expanded"], e["generated"]), what
        assert [rollout_stats(s) for s in r.rollout_statistics] == e["rollout_statistics"], what
        assert [("in_progress" if s is None else status_name(s, False)) for s in r.rollout_statuses] \
            == e["rollout_statuses"], what
    else:
        raise AssertionError(f"{what}: unknown algo {a['algo']}")


@pytest.mark.parametrize("name", sorted(ROWS))
def test_golden_rows(name):
    """Every recorded fork run of this task, through the Python API (the C++ IwVariantGolden test's comparisons plus
    the name hashes of landing states and co-occurrence)."""
    rows = ROWS[name]
    task = golden_task(name, rows[0]["source"])
    for row in rows:
        run_row(task, row)


# ------------------------------------------------------------------------------------------------ helpers

@pytest.fixture(scope="module")
def gripper():
    return text_task("gripper__prob05")


def replay(task, plan, start=None):
    s = task.initial_state if start is None else start
    for a in plan:
        s = task.apply(s, a)
    return s


def rollout_key(r):
    """Everything a rollout reports, comparable across runs."""
    return (str(r.status), [(p.arity, p.expanded, p.generated, p.generated_in_tree) for p in r.passes],
            [str(a) for a in r.plan], r.num_states, [str(a) for a in r.reached_fluent_atoms],
            [str(a) for a in r.reached_derived_atoms],
            [(tuple(map(str, s.atoms)), s.direct_dead_end, tuple(sorted(map(str, s.state.atoms())))) for s in r.landing_states],
            {str(a): i for a, i in r.landing_state_by_atom.items()},
            {str(a): [str(b) for b in row] for a, row in r.co_occurrence.items()})


def portfolio_key(r):
    return (str(r.status), [str(a) for a in r.plan], r.cost, r.certified_optimal, r.winning_worker,
            r.iw_completed_depth, r.total_expansions, [rollout_stats(s) for s in r.rollout_statistics],
            r.rollout_statuses, r.rollout_rounds)


def iw_key(r):
    return (str(r.status), [(p.arity, p.expanded, p.generated, p.generated_in_tree) for p in r.passes],
            [str(a) for a in r.plan], r.cost)


# ------------------------------------------------------------------------------------------------ results and plans

@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "gripper__prob05", "miconic-simpleadl__s10-2",
                                  "openstacks-opt08-adl__p03", "philosophers__p03-phil4", "pathways__p02"])
def test_plans_replay(name):
    """Every solved variant's plan replays to a goal state from the start, the reported goal state."""
    task = text_task(name)
    runs = [search.liw(task, max_arity=1, landmarks="lifted", max_seconds=20),
            search.abstracted_iw(task, width=1), search.projective_iw(task, typed_projection=True),
            search.rollout_iw(task, ordering="dgaf", max_seconds=20),
            search.atomic_goal_portfolio(task, num_threads=1, max_seconds=20)]
    runs += list(search.find_rollouts_parallel(task, [1, 2, 3], max_arity=1, num_threads=2))
    for r in runs:
        if not r.solved:
            continue
        end = replay(task, r.plan)
        assert task.is_goal(end), (name, r)
        if hasattr(r, "goal_state"):
            assert r.goal_state == end


def test_liw_landmark_sources_agree(blocks):
    lifted = search.lifted_fact_landmarks(blocks)
    approx = search.approximate_fact_landmarks(blocks)
    assert iw_key(search.liw(blocks, max_arity=1, landmarks="lifted")) == iw_key(
        search.liw(blocks, max_arity=1, landmarks=lifted))
    assert iw_key(search.liw(blocks, max_arity=1, landmarks="approximate")) == iw_key(
        search.liw(blocks, max_arity=1, landmarks=approx))
    # no landmarks: every state's coordinate is BOT (a valid, weaker LIW)
    r = search.liw(blocks, max_arity=1)
    assert [p.arity for p in r.passes] == [0, 1]
    # abstracted LIW with a graph differs from abstracted IW without one
    assert search.abstracted_iw(blocks, landmarks=lifted).passes[0].expanded != search.abstracted_iw(
        blocks).passes[0].expanded


def test_projective_iw_is_abstracted_iw(blocks):
    for typed in (False, True):
        for keep in (False, True):
            assert iw_key(search.projective_iw(blocks, typed_projection=typed, keep_goal_nonunary_atoms=keep)) == \
                iw_key(search.abstracted_iw(blocks, width=1, base_abstracted=not typed, preserve_goal_atoms=keep))


def test_layer_orders(blocks):
    q = search.abstracted_iw(blocks, width=2)
    o = search.abstracted_iw(blocks, width=2, layer_order="in_order")
    assert q.passes[0].expanded == o.passes[0].expanded  # the same tree, layer by layer
    r1 = search.abstracted_iw(blocks, width=2, layer_order="randomized", seed=5)
    r2 = search.abstracted_iw(blocks, width=2, layer_order="randomized", seed=5)
    assert iw_key(r1) == iw_key(r2)
    t = search.liw(blocks, max_arity=2, layer_order="in_order", max_next_layer_states=4)
    assert all(p.expanded <= q.passes[0].expanded for p in t.passes)


def test_rollout_iw_orderings(blocks):
    goal = [["(on h c)"]]
    for name, kind in (("in_order", ActionOrdering.IN_ORDER), ("randomized", ActionOrdering.RANDOMIZED),
                       ("direct_goal_achiever_first", ActionOrdering.DIRECT_GOAL_ACHIEVER_FIRST),
                       ("regression", ActionOrdering.GOAL_REGRESSION_RELEVANCE),
                       ("mixed", ActionOrdering.MIXED_REGRESSION_RANDOM)):
        a = search.rollout_iw(blocks, ordering=kind, seed=3, goal=goal)
        b = search.rollout_iw(blocks, ordering=name, seed=3, goal=goal)
        assert rollout_stats(a.statistics) == rollout_stats(b.statistics) and a.plan == b.plan, name
        assert not a.solved or "(on h c)" in {str(x) for x in a.goal_state.atoms()}
        assert str(kind) in ("in_order", "randomized", "direct_goal_achiever_first", "goal_regression_relevance",
                             "mixed_regression_random")
    # seeds matter for the randomized orderings only
    s = [rollout_stats(search.rollout_iw(blocks, ordering="randomized", seed=k).statistics) for k in range(4)]
    assert len({json.dumps(x, sort_keys=True) for x in s}) > 1
    assert rollout_stats(search.rollout_iw(blocks, seed=1).statistics) == rollout_stats(
        search.rollout_iw(blocks, seed=2).statistics)
    # incumbent bound and max_rollouts
    r = search.rollout_iw(blocks, incumbent_bound=3)
    assert r.statistics.incumbent_bound_prunings > 0
    assert search.rollout_iw(blocks, max_rollouts=5).status == Status.FAILED


# ------------------------------------------------------------------------------------------------ determinism

@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "rovers__p02",
                                  "openstacks-opt08-adl__p03"])
def test_rollouts_identical_across_threads_and_runs(name):
    task = text_task(name)
    seeds = list(range(1, 17))
    kw = dict(max_arity=1, report_landing_states=True, report_co_occurrence=True, max_expanded=20_000)
    ref = [rollout_key(r) for r in search.find_rollouts_parallel(task, seeds, num_threads=1, **kw)]
    for threads in (3, 8, 16):
        b = search.find_rollouts_parallel(task, seeds, num_threads=threads, **kw)
        assert b.threads_used == min(threads, len(seeds))
        assert [rollout_key(r) for r in b] == ref, threads
    fresh = text_task(name)  # a fresh task: lazy slots assigned concurrently, canonical ids unchanged
    assert [rollout_key(r) for r in search.find_rollouts_parallel(fresh, seeds, num_threads=8, **kw)] == ref
    # a subset of seeds gives the same rollouts
    sub = search.find_rollouts_parallel(task, seeds[3:7], num_threads=2, **kw)
    assert [rollout_key(r) for r in sub] == ref[3:7]
    # intersect and merge are functions of the batch
    b = search.find_rollouts_parallel(task, seeds, num_threads=8, **kw)
    rows = search.intersect_co_occurrence(b)
    assert {str(k): sorted(map(str, v)) for k, v in rows.items()} == \
        {str(k): sorted(map(str, v)) for k, v in search.intersect_co_occurrence(list(b)).items()}
    first = b[0].co_occurrence
    for atom, row in rows.items():
        assert set(row) <= set(first[atom])
    merged = search.merge_landing_states(b)
    assert len(merged.by_rollout) == len(b) and len(set(merged.states)) == len(merged.states)
    for r, idx in zip(b, merged.by_rollout):
        assert {merged.states[i] for i in idx} == {s.state for s in r.landing_states}


def test_portfolio_serial_reproducible(blocks):
    goal = [["(on h c)"]]
    a = search.atomic_goal_portfolio(blocks, goal=goal, num_threads=1, base_seed=7)
    b = search.atomic_goal_portfolio(blocks, goal=goal, num_threads=1, base_seed=7)
    assert portfolio_key(a) == portfolio_key(b)
    assert a.solved and a.certified_optimal and a.winning_worker == 0 and a.threads_used == 1
    # threaded: timing decides the winner, but every plan is valid and as short as the certified one
    c = search.atomic_goal_portfolio(blocks, goal=goal, num_threads=4)
    assert c.solved and len(c.plan) == len(a.plan)
    assert "(on h c)" in {str(x) for x in replay(blocks, c.plan).atoms()}
    # the full goal, no plan within width 1: the certifier exhausts its space
    d = search.atomic_goal_portfolio(blocks, num_threads=1, num_rollout_workers=2,
                                     rollout_orderings=[ActionOrdering.IN_ORDER, ("randomized", 3)])
    assert d.status == Status.EXHAUSTED and d.plan_length is None and d.winning_worker is None
    assert len(d.rollout_statistics) == 2 and d.certifier_status == Status.EXHAUSTED


# ------------------------------------------------------------------------------------------------ control surface

def test_budgets_and_cancel(blocks):
    token = search.CancelToken()
    token.request()
    for fn in (lambda **k: search.liw(blocks, max_arity=2, **k), lambda **k: search.abstracted_iw(blocks, width=2, **k),
               lambda **k: search.rollout_iw(blocks, **k)):
        assert fn(cancel=token).status == Status.CANCELLED
        assert fn(max_seconds=0.0).status == Status.OUT_OF_TIME
    assert search.liw(blocks, max_arity=2, max_states=10).status == Status.OUT_OF_STATES
    assert search.rollout_iw(blocks, max_states=10).status == Status.OUT_OF_STATES
    b = search.find_rollouts_parallel(blocks, [1, 2], max_arity=2, cancel=token, num_threads=2)
    assert all(r.status == Status.CANCELLED for r in b)
    p = search.atomic_goal_portfolio(blocks, num_threads=1, max_expanded=10)
    assert p.status != Status.SOLVED and p.total_expansions <= 12


def test_cancel_from_another_thread(blocks):
    token = search.CancelToken()
    started = threading.Event()

    class Obs:
        def on_start(self, state):
            started.set()

    t = threading.Thread(target=lambda: (started.wait(10), token.request()))
    t.start()
    r = search.liw(blocks, max_arity=3, cancel=token, observer=Obs(), progress_interval=100)
    t.join()
    assert token.requested and r.status in (Status.CANCELLED, Status.SOLVED, Status.EXHAUSTED)


def test_observer_transition_events(blocks):
    class Obs:
        def __init__(self):
            self.expand = 0
            self.outcomes = {}
            self.passes = 0

        def on_expand(self, i, state):
            self.expand += 1

        def on_transition(self, parent, action, child, state, outcome):
            assert isinstance(outcome, TransitionOutcome)
            assert isinstance(state, mymyr.State) and isinstance(parent, int)
            if outcome == TransitionOutcome.OPENED:  # a new node: its id
                assert isinstance(child, int)
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

        def on_pass(self, arity, stats):
            self.passes += 1

    o = Obs()
    r = search.abstracted_iw(blocks, width=1, observer=o)
    assert o.expand == r.total.expanded and o.passes == 1
    assert sum(o.outcomes.values()) == r.total.generated
    assert o.outcomes[TransitionOutcome.OPENED] == r.passes[0].generated_in_tree
    o = Obs()
    r = search.rollout_iw(blocks, ordering="dgaf", goal=[["(on h c)"]], observer=o)
    assert r.solved and o.outcomes.get(TransitionOutcome.GOAL) == 1
    assert sum(o.outcomes.values()) == r.statistics.generated


def test_make_worker_protocol(blocks):
    """An observer with make_worker(k) runs the batch on several threads, each worker's hot events on its own object;
    without it the batch runs on the calling thread alone."""
    class Worker:
        def __init__(self):
            self.expand = 0
            self.threads = set()

        def on_expand(self, i, state):
            self.expand += 1
            self.threads.add(threading.get_ident())

    class Root:
        def __init__(self):
            self.workers = {}
            self.expand = 0
            self.passes = 0
            self.ended = 0

        def make_worker(self, k):
            self.workers[k] = Worker()
            return self.workers[k]

        def on_expand(self, i, state):
            self.expand += 1

        def on_pass(self, arity, stats):
            self.passes += 1

        def on_end(self, status, stats):
            self.ended += 1

    seeds = list(range(8))
    root = Root()
    b = search.find_rollouts_parallel(blocks, seeds, max_arity=1, num_threads=4, observer=root)
    assert b.threads_used == 4 and sorted(root.workers) == seeds and root.expand == 0 and root.ended == 1
    for k, r in enumerate(b):
        assert root.workers[k].expand == sum(p.expanded for p in r.passes)
    assert root.passes == sum(len(r.passes) for r in b)
    assert len(set().union(*(w.threads for w in root.workers.values()))) > 1 or os.cpu_count() == 1

    class Plain:
        def __init__(self):
            self.expand = 0

        def on_expand(self, i, state):
            self.expand += 1

    p = Plain()
    b2 = search.find_rollouts_parallel(blocks, seeds, max_arity=1, num_threads=4, observer=p)
    assert b2.threads_used == 1  # no make_worker: the calling thread alone
    assert p.expand == sum(pp.expanded for r in b2 for pp in r.passes)
    assert [rollout_key(r) for r in b2] == [rollout_key(r) for r in b]

    # a goal callable with make_worker(k); a plain one runs serially
    target = "(on h c)"

    class Goal:
        def __call__(self, s):
            return s.holds(target)

        def make_worker(self, k):
            return lambda s: s.holds(target)

    g = search.find_rollouts_parallel(blocks, seeds, max_arity=1, num_threads=4, goal=Goal())
    assert g.threads_used == 4 and any(r.solved for r in g)
    serial = search.find_rollouts_parallel(blocks, seeds, max_arity=1, num_threads=4, goal=lambda s: s.holds(target))
    assert serial.threads_used == 1
    assert [rollout_key(r) for r in g] == [rollout_key(r) for r in serial]
    # the portfolio's workers: 0 is the certifier
    root = Root()
    r = search.atomic_goal_portfolio(blocks, goal=Goal(), num_threads=1, num_rollout_workers=3, observer=root)
    assert sorted(root.workers) == [0, 1, 2, 3] and r.solved


def test_callback_errors_propagate(blocks):
    class Boom:
        def on_expand(self, i, state):
            raise KeyError("boom")

    for fn in (lambda **k: search.liw(blocks, **k), lambda **k: search.abstracted_iw(blocks, **k),
               lambda **k: search.rollout_iw(blocks, **k),
               lambda **k: search.find_rollouts_parallel(blocks, [1, 2], max_arity=1, **k),
               lambda **k: search.atomic_goal_portfolio(blocks, num_threads=1, **k)):
        with pytest.raises(KeyError, match="boom"):
            fn(observer=Boom())
        with pytest.raises(ZeroDivisionError):
            fn(goal=lambda s: 1 / 0)

    class BadWorker:
        def make_worker(self, k):
            raise RuntimeError("no worker")

    with pytest.raises(RuntimeError, match="no worker"):
        search.find_rollouts_parallel(blocks, [1, 2], max_arity=1, num_threads=2, observer=BadWorker())


def test_numeric_tasks_refused():
    path = NUMERIC / "cs-counters.txt"
    if not path.is_file():
        pytest.skip("numeric tasks not found")
    task = mymyr.Task.from_text(str(path))
    for fn in (lambda: search.liw(task), lambda: search.liw(task, landmarks="lifted"), lambda: search.abstracted_iw(task),
               lambda: search.projective_iw(task), lambda: search.rollout_iw(task),
               lambda: search.find_rollouts_parallel(task, [1]), lambda: search.atomic_goal_portfolio(task)):
        with pytest.raises(ValueError, match="numeric"):
            fn()
    assert search.iw(task, max_arity=1).status in (Status.SOLVED, Status.EXHAUSTED)  # IW supports them


def test_argument_errors(blocks, gripper):
    g = search.approximate_fact_landmarks(gripper)
    with pytest.raises(ValueError):
        search.liw(blocks, landmarks=g)  # another task's graph
    with pytest.raises(ValueError):
        search.liw(blocks, landmarks="exact")
    with pytest.raises(TypeError):
        search.liw(blocks, landmarks=5)
    with pytest.raises(ValueError):
        search.abstracted_iw(blocks, layer_order="sideways")
    with pytest.raises(ValueError):
        search.rollout_iw(blocks, ordering="best")
    with pytest.raises(ValueError):
        search.abstracted_iw(blocks, start=gripper.initial_state)
    with pytest.raises(ValueError):
        search.iw(blocks, transition_ordering=search.LandmarkTransitionOrdering(g))
    with pytest.raises(ValueError):
        search.atomic_goal_portfolio(blocks, rollout_orderings=[("dgaf", 1, 2)])
    # the C++'s own option checks: a FAILED result with a message
    g2 = search.FactLandmarkGraph(blocks, [], [["(on c a)", "(on b a)"]])
    r = search.liw(blocks, landmarks=g2, disjunctive=True, all_private=True, unshared_atoms=["(on a b)"])
    assert r.status == Status.FAILED and "all_private" in r.message
    r = search.abstracted_iw(blocks, max_next_layer_states=3)
    assert r.status == Status.FAILED and "max_next_layer_states" in r.message
    b = search.find_rollouts_parallel(blocks, [1], max_arity=1)
    b2 = search.find_rollouts_parallel(gripper, [1], max_arity=1)
    with pytest.raises(ValueError):
        search.intersect_co_occurrence([b[0], b2[0]])
    with pytest.raises(IndexError):
        b[1]
    assert search.abstracted_iw(blocks, width=4).status == Status.FAILED  # the C++'s own check (a message)


# ------------------------------------------------------------------------------------------------ landmarks, reachability

def test_landmark_graphs(blocks, gripper):
    for task in (blocks, gripper):
        a = search.approximate_fact_landmarks(task, max_disjunctive_landmark_size=4)
        lifted = search.lifted_fact_landmarks(task)
        assert a.has_achiever_index and not lifted.has_achiever_index and lifted.lifted and not a.lifted
        s0 = task.initial_state
        for g in (a, lifted):
            assert len(g) == len(g.landmarks) and g.num_unachieved(s0) + g.num_achieved(s0) == len(g)
            assert set(g.achieved(s0)) == {x for x in g.landmarks if s0.holds(x)}
            for before, after in g.orderings:
                assert before in g.predecessors(after) and after in g.successors(before)
                assert g.is_landmark(before) and g.is_landmark(after)
        search.verify_pi_plus_fact_landmarks(lifted)
        for x in a.landmarks:
            ach = a.achievers(x)
            assert set(map(str, a.first_achievers(x))) <= set(map(str, ach))
            u = a.unique_achiever(x)
            assert (u is None) == (len(ach) != 1)
            for act in ach:
                assert x in a.achieved_by(act) and a.is_landmark_achiever(act)
        with pytest.raises(ValueError):
            lifted.achievers(lifted.landmarks[0])
        for lm in lifted.lifted:
            assert str(lm).startswith(lm.predicate + "(") and (lm.fact is not None) == lm.is_fact
            assert all(isinstance(m, mymyr.Atom) for m in lm.members)
    # a graph from given atoms, normalized as the fork does
    g = search.FactLandmarkGraph(blocks, ["(on a g)", "(on a g)", "(clear f)"], [["(on a g)", "(on b a)"],
                                                                                   ["(on c a)", "(on b a)"]],
                                 [("(clear f)", "(on a g)")])
    assert [str(x) for x in g.landmarks] == ["(on a g)", "(clear f)"]
    assert len(g.disjunctive) == 1 and g.orderings[0][0] == blocks.atom("(clear f)")
    assert "FactLandmarkGraph(landmarks=2" in repr(g)


def test_relaxed_reachability(blocks, gripper):
    for task in (blocks, gripper, text_task("openstacks-opt08-adl__p03")):
        rr = search.RelaxedReachability(task)
        assert rr.goal_reachable and rr.table.goal_reachable
        # every atom of every reachable state is relaxed reachable
        seen = set()
        for s in task.successor_states(task.initial_state) + [task.initial_state]:
            for t in task.successor_states(s):
                seen.update(t.atoms())
        assert all(rr.is_reachable(a) for a in seen)
        preds = {a.predicate for a in seen}
        n = sum(len(rr.table.atoms(p)) for p in preds)
        assert n <= rr.table.num_fluent_atoms + rr.table.num_derived_atoms
        assert rr.statistics.fluent_atoms == rr.table.num_fluent_atoms
        # the lifted landmarks block the relaxed goal once forbidden (pi+), unless initially true
        g = search.lifted_fact_landmarks(task, reachability=rr)
        s0 = task.initial_state
        for x in g.landmarks:
            if not s0.holds(x):
                assert not rr.goal_reachable_without([x]), x
                assert not rr.restricted([x]).goal_reachable
        q = rr.table.witness_query([])
        some = next(iter(seen))
        assert q.avoids(some) == search.WitnessVerdict.REACHABLE_WITHOUT and q.memoized >= 1
        assert not search.RelaxedReachability(task, record_witnesses=False).table.has_witnesses
    rr = search.RelaxedReachability(blocks)
    on = rr.table.project(2, [("on", [0, 1])])
    assert len(on) == 2 and set(on[0]) <= {o for o in "abcdefgh"}
    assert rr.table.project(1, [("on", [0, "a"])], disequalities=[(0, "b")])[0].count("b") == 0


def test_transition_ordering(blocks):
    g = search.approximate_fact_landmarks(blocks)
    order = search.LandmarkTransitionOrdering(g)
    assert order.prefer_new_landmarks and order.graph.landmarks == g.landmarks
    r = search.iw(blocks, max_arity=1, transition_ordering=order)
    plain = search.iw(blocks, max_arity=1)
    assert r.status == plain.status and [p.arity for p in r.passes] == [p.arity for p in plain.passes]
    if r.solved:
        assert blocks.is_goal(replay(blocks, r.plan))
    s0 = blocks.initial_state
    a = blocks.applicable_actions(s0)[0]
    sc = order.score(s0, a, blocks.apply(s0, a))
    assert sc.num_new_landmarks >= 0 and not order.prefer(sc, sc)
    with pytest.raises(ValueError):
        search.LandmarkTransitionOrdering(search.lifted_fact_landmarks(blocks))  # no achiever index
    s = search.siw(blocks, max_arity=1, transition_ordering=order)
    assert s.status in (Status.SOLVED, Status.EXHAUSTED)


# ------------------------------------------------------------------------------------------------ free threading

def test_eight_threads_share_tasks():
    """Gate 2: 8 Python threads run the variants at once on shared tasks; every result equals the single-threaded one.
    The interpreter is free-threaded (the GIL stays disabled)."""
    assert not sys._is_gil_enabled()
    tasks = [text_task("gripper__prob05"), text_task("logistics00__probLOGISTICS-6-1"), text_task("driverlog__p03"),
             text_task("openstacks-opt08-adl__p03")]
    graphs = [search.lifted_fact_landmarks(t) for t in tasks]

    def run(i, task, graph, local):
        t = task.local() if local else task
        if i == 0:
            return iw_key(search.liw(t, max_arity=3, landmarks=graph, max_states=300_000))  # 0.06-0.3 s each
        if i == 1:
            return iw_key(search.abstracted_iw(t, width=3, max_states=300_000))
        if i == 2:
            return iw_key(search.projective_iw(t, layer_order="randomized", seed=11))
        if i == 3:
            r = search.rollout_iw(t, ordering="mixed", seed=5, max_states=20_000)
            return (str(r.status), rollout_stats(r.statistics), [str(a) for a in r.plan])
        if i == 4:
            b = search.find_rollouts_parallel(t, list(range(32)), max_arity=2, num_threads=2,
                                              report_landing_states=True, report_co_occurrence=True)
            return [rollout_key(r) for r in b]
        return portfolio_key(search.atomic_goal_portfolio(t, num_threads=1, num_rollout_workers=2,
                                                          max_expanded=20_000))

    jobs = [(i, k) for i in range(6) for k in range(len(tasks))]
    ref = {(i, k): run(i, tasks[k], graphs[k], False) for i, k in jobs}
    errors, results = [], {}
    barrier = threading.Barrier(8)

    def work(w):
        try:
            barrier.wait()
            for rep in range(2):
                for n, (i, k) in enumerate(jobs):
                    if (n + w + rep) % 2 == 0:
                        results[(w, rep, i, k)] = run(i, tasks[k], graphs[k], (w + rep) % 2 == 1)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(results) == 8 * len(jobs)
    for (w, rep, i, k), v in results.items():
        assert v == ref[(i, k)], (w, rep, i, k)
    assert not sys._is_gil_enabled()


def test_reprs_and_types(blocks):
    r = search.rollout_iw(blocks)
    assert "RolloutIwResult(" in repr(r) and isinstance(r.statistics, search.RolloutIwStatistics)
    assert isinstance(r.total, search.Statistics)
    b = search.find_rollouts_parallel(blocks, [1, 2], max_arity=1, report_landing_states=True)
    assert "ParallelRolloutsResult(" in repr(b) and isinstance(b[0], search.RolloutResult)
    assert isinstance(b.rollouts[1].search, search.IwResult) and b[-1].seed == 2
    assert isinstance(b[0].landing_states[0], search.LandingState) and "LandingState(" in repr(b[0].landing_states[0])
    p = search.atomic_goal_portfolio(blocks, num_threads=1, goal=[["(on h c)"]])
    assert "PortfolioResult(" in repr(p) and isinstance(p.certifier, search.IwResult)
    rr = search.RelaxedReachability(blocks)
    assert "RelaxedReachability(" in repr(rr) and isinstance(rr.statistics, search.ReachabilityStatistics)
    assert isinstance(search.merge_landing_states(b), search.MergedLandingStates)
    assert isinstance(search.lifted_fact_landmarks(blocks).lifted[0], search.LiftedLandmark)
