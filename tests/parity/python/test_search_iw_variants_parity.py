"""Parity: the IW family variants against the recorded reference runs.

Replays tests/parity/data/iw_variants/fork_golden.json (508 rows over the 24 configurations checked, including the
name hashes of landing states and co-occurrence) through the Python bindings."""

import json

import pytest

import mymyr
from mymyr import search
from mymyr.search import ActionOrdering, Status

from parity_support import DATA, FORK_DATA, ROOT, WORK

GOLDEN = DATA / "iw_variants/fork_golden.json"

def rollout_stats(s):
    return {k: getattr(s, k) for k in ("rollouts", "generated", "expanded", "feature_depth_improvements", "case1",
                                       "case2", "case3", "case4", "solved_propagations", "dead_ends",
                                       "depth_bound_prunings", "incumbent_bound_prunings", "max_rollout_depth",
                                       "tree_nodes")}


# ------------------------------------------------------------------------------------------------ fork conventions

FORK_STATUS = {Status.SOLVED: "solved", Status.EXHAUSTED: "failed", Status.OUT_OF_STATES: "out_of_states",
               Status.OUT_OF_TIME: "out_of_time", Status.CANCELLED: "canceled", Status.FAILED: "error",
               Status.UNSOLVABLE: "unsolvable"}
ORDERING = {"in_order": ActionOrdering.IN_ORDER, "randomized": ActionOrdering.RANDOMIZED,
            "dgaf": ActionOrdering.DIRECT_GOAL_ACHIEVER_FIRST, "regression": ActionOrdering.GOAL_REGRESSION_RELEVANCE,
            "mixed": ActionOrdering.MIXED_REGRESSION_RANDOM}
NO_VALUE = 4294967295  # the fork's and the C++ API's "none" for u32 fields


def status_name(s, ladder):
    """The fork's status names per search kind (bench/parity/mymyr_iw_variants.cpp, test_iw_variants_golden.cpp)."""
    if s == Status.EXHAUSTED:
        return "failed" if ladder else "exhausted"
    if s == Status.FAILED:
        return "failed"
    return FORK_STATUS[s]


def per_pass(passes, arity=None):
    return ";".join(f"{arity if arity is not None else p.width}:{p.expanded}/{p.generated}/{p.generated_in_tree}"
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
    """The name hashes fork_iw_variants and mymyr_iw_variants print for a rollout (bench/parity/mymyr_iw_variants.cpp rollout_json)."""
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
            pytest.skip(f"PDDL of {name} not found (set MYMYR_PARITY_DATA)")
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
            r = search.liw(task, max_width=a["k"], **lm, **kw)
            assert per_pass(r.passes) == e["per_pass"], what
        assert status_name(r.status, a["algo"] == "liw") == e["status"], what
        if r.solved:
            assert len(r.plan) == e["plan_len"], what
    elif a["algo"] == "rollout_iw":
        r = search.rollout_iw(task, ordering=ORDERING[a["ordering"]], seed=a["seed"], max_states=a["max_states"], **kw)
        assert status_name(r.status, False) == e["status"], what
        assert rollout_stats(r.rollout) == e["rollout"], what
        assert r.root_solved == e["root_solved"], what
        if r.solved:
            assert len(r.plan) == e["plan_len"], what
    elif a["algo"] == "rollouts":
        seeds = [int(s) for s in a["seeds"].split(",")]
        b = search.find_rollouts_parallel(task, seeds, max_width=a["k"], threads=4,
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
        r = search.atomic_goal_portfolio(task, num_rollout_workers=a["workers"], threads=a["threads"],
                                         rollout_orderings=orderings, **kw)
        assert status_name(r.status, True) == e["status"], what
        assert (r.plan_length if r.solved else 0) == e["plan_length"], what
        assert r.certified_optimal == e["certified_optimal"], what
        assert (NO_VALUE if r.winning_worker is None else r.winning_worker) == e["winning_worker"], what
        cs = "in_progress" if r.certifier_status is None else status_name(r.certifier_status, True)
        assert cs == e["certifier_status"], what
        assert (NO_VALUE if r.iw_completed_depth is None else r.iw_completed_depth) == e["iw_completed_depth"], what
        assert r.iw_lower_bound == e["iw_lower_bound"], what
        assert r.stats.expanded == e["total_expansions"], what
        assert (r.certifier.stats.expanded, r.certifier.stats.generated) == (e["expanded"], e["generated"]), what
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
