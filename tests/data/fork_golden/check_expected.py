#!/usr/bin/env python3
"""Validate tests/data/expected/*.json (format: README.md).

  python3 tests/data/fork_golden/check_expected.py [--dir tests/data/expected] [-v]

Structural checks (errors): size <= 2 MB; format tag; every set sorted, its count and FNV-1a-64 sum hash consistent with
its items; walk replay (the taken action is the splitmix64 choice among the sorted applicable actions; the last step
is a dead end or the step limit); goal flag against goal_count; IW pass arities and totals; BrFS states against
expanded; the name tables.
Sanity checks (warnings): h_max <= h_ff and h_max <= h_add, blind = 0, h = 0 on goal states (h_ff may exceed h_add:
the fork's relaxed plans need not follow h_add's best supporters).
Exit status 1 on any error.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FORMAT = "mymyr-fork-golden/1"
MAX_BYTES = 2_000_000
M64 = (1 << 64) - 1


def fnv1a64(s):
    h = 0xCBF29CE484222325
    for c in s.encode():
        h = ((h ^ c) * 0x100000001B3) & M64
    return h


def set_hash(items):
    return f"{sum(fnv1a64(x) for x in items) & M64:016x}"


class SplitMix64:
    def __init__(self, seed):
        self.x = seed & M64

    def next(self):
        self.x = (self.x + 0x9E3779B97F4A7C15) & M64
        z = self.x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


class Report:
    def __init__(self, verbose):
        self.errors, self.warnings, self.verbose = [], [], verbose

    def err(self, task, msg):
        self.errors.append(f"{task}: {msg}")

    def warn(self, task, msg):
        self.warnings.append(f"{task}: {msg}")


def check_set(r, task, where, s):
    items = s.get("items")
    if not isinstance(s.get("count"), int) or not re.fullmatch(r"[0-9a-f]{16}", s.get("hash", "")):
        r.err(task, f"{where}: bad count/hash")
        return None
    if items is None:
        return None
    if len(items) != s["count"]:
        r.err(task, f"{where}: count {s['count']} != {len(items)} items")
    if items != sorted(items):
        r.err(task, f"{where}: items not sorted")
    if set_hash(items) != s["hash"]:
        r.err(task, f"{where}: hash mismatch")
    return items


def check_walks(r, task, w, names):
    if w is None:
        return
    heur = set(w.get("heuristics", []))
    for wi, walk in enumerate(w["walks"]):
        rng = SplitMix64(walk["seed"])
        steps = walk["steps"]
        if not 1 <= len(steps) <= w["num_steps"] + 1:
            r.err(task, f"walk {wi}: {len(steps)} steps")
        for si, st in enumerate(steps):
            where = f"walk {wi} step {si}"
            check_set(r, task, where + " fluent", st["fluent_atoms"])
            check_set(r, task, where + " derived", st["derived_atoms"])
            acts = check_set(r, task, where + " applicable", st["applicable"])
            last = si == len(steps) - 1
            n = st["applicable"]["count"]
            if last:
                if st["taken"] is not None:
                    r.err(task, f"{where}: last step has a taken action")
                if walk["dead_end"] != (n == 0) or (n and len(steps) != w["num_steps"] + 1):
                    r.err(task, f"{where}: dead_end flag / step count inconsistent")
            else:
                if st["taken"] is None or n == 0:
                    r.err(task, f"{where}: missing taken action")
                idx = rng.next() % n if n else 0
                if acts is not None and n and acts[idx] != st["taken"]:
                    r.err(task, f"{where}: taken {st['taken']} != RNG choice {acts[idx]}")
                if not re.fullmatch(r"\([^\s()]+( [^\s()]+)*\)", st["taken"] or ""):
                    r.err(task, f"{where}: malformed action string {st['taken']!r}")
            h = st["h"]
            if h.get("blind") != 0:
                r.warn(task, f"{where}: blind = {h.get('blind')}")
            if st["is_goal"] != (h["goal_count"] == 0) and names and names["num_goal_literals"]:
                # goal_count counts fluent and derived goal literals; a static goal literal can still fail
                r.warn(task, f"{where}: is_goal={st['is_goal']} but goal_count={h['goal_count']}")
            for key in ("hmax", "hadd", "hff"):
                obs = h.get(key + "_observed")
                if obs is not None and (len(obs) < 2 or h[key] not in obs):
                    r.err(task, f"{where}: {key}={h[key]} with observed values {obs}")
            if {"hmax", "hadd", "hff"} <= heur:
                vals = [h["hmax"], h["hff"], h["hadd"]]
                if any(v is None for v in vals) and not all(v is None for v in vals):
                    r.err(task, f"{where}: inconsistent dead-end h values {vals}")
                elif vals[0] is not None and not vals[0] <= min(vals[1], vals[2]):
                    r.warn(task, f"{where}: h_max above h_ff or h_add: {vals}")
                if st["is_goal"] and any(v != 0 for v in vals):
                    r.warn(task, f"{where}: goal state with h {vals}")


def check_file(r, path):
    task = os.path.basename(path)[:-5]
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        r.err(task, f"file is {size} bytes (> 2 MB)")
    d = json.load(open(path))
    if d.get("format") != FORMAT:
        r.err(task, f"format {d.get('format')!r}")
    names = d.get("names")
    if names:
        for p in names["predicates"]:
            if not (len(p) == 3 and p[0] in "SFD" and isinstance(p[1], int)):
                r.err(task, f"bad predicate entry {p}")
        for s in names["schemas"]:
            if s["original_arity"] > s["arity"]:
                r.err(task, f"schema {s['name']}: original arity > arity")
    b = d.get("brfs")
    if b:
        if (b["status"] == "exhausted") != (b["states"] is not None) or (b["states"] is not None and b["states"] != b["expanded"]):
            r.err(task, f"brfs states/status inconsistent: {b}")
    for iw in d.get("iw", []):
        arities = [p["arity"] for p in iw["passes"]]
        if arities != list(range(len(arities))) or len(arities) > iw["k"] + 1:
            r.err(task, f"iw{iw['k']}: pass arities {arities}")
        if sum(p["expanded"] for p in iw["passes"]) != iw["expanded"] or sum(p["generated"] for p in iw["passes"]) != iw["generated"]:
            r.err(task, f"iw{iw['k']}: pass sums != totals")
        if (iw["status"] == "solved") != (iw["plan_length"] is not None):
            r.err(task, f"iw{iw['k']}: plan / status inconsistent")
    a = d.get("astar")
    if a and (a["status"] == "solved") != (a["optimal_cost"] is not None):
        r.err(task, f"astar: cost / status inconsistent")
    check_walks(r, task, d.get("walks"), names)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "tests", "data", "expected"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    r = Report(a.verbose)
    docs = {}
    for path in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        try:
            docs[os.path.basename(path)[:-5]] = check_file(r, path)
        except (KeyError, TypeError, ValueError) as e:
            r.err(os.path.basename(path), f"unreadable: {e!r}")
    print(f"{len(docs)} files checked")
    for kind, xs in (("ERROR", r.errors), ("warning", r.warnings)):
        for x in xs[: None if a.verbose or kind != "warning" else 20]:
            print(f"{kind}: {x}")
        if kind == "warning" and len(xs) > 20 and not a.verbose:
            print(f"... {len(xs) - 20} more warnings (-v)")
    print(f"errors={len(r.errors)} warnings={len(r.warnings)}")
    sys.exit(1 if r.errors else 0)


if __name__ == "__main__":
    main()
