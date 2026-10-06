#!/usr/bin/env python3
"""Export the fork golden data for every suite task into tests/data/expected/<domain>__<problem>.json.

One export_golden process per task (at most --jobs at a time). Each process prints one "PHASE <name> <json>" line per
finished phase; the driver keeps every finished phase. A process is killed when it exceeds --timeout seconds or --rss-mb
of resident memory; the phase it was in is recorded as killed (with the reason) and a new process continues with the
remaining phases. Timings go to the "meta" section, which check_expected.py ignores.

  MYMYR_WORK=<work dir> python3 tests/data/fork_golden/export_all.py [--jobs 6] [--only SUBSTR] [--timeout 1800]

MYMYR_WORK must hold clones of the fork (mimir-cs/Benchmark for the PDDL instances, mimir for export_golden's build).
MYMYR_IPC, if set, additionally resolves the IPC-tagged IW-suite tasks against an IPC data directory
(<MYMYR_IPC>/<domain>/test/{domain,problem}.pddl); without it those tasks are reported missing and skipped.

Tasks: the 22-task BrFS suite below and the IW suite (tests/data/tasks_iw), deduplicated by PDDL file.
See README.md for the format and the conventions.
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

MAC = platform.system() == "Darwin"
WORK = os.environ.get("MYMYR_WORK", os.path.join(REPO, ".work"))  # gitignored clones
BENCH = os.path.join(WORK, "mimir-cs", "Benchmark")
IPC = os.environ.get("MYMYR_IPC", "")
EXE = os.path.join(WORK, "build-fork-golden", "export_golden")
EXPECTED = os.path.join(REPO, "tests", "data", "expected")
PHASES = ["task", "walks", "iw1", "iw2", "brfs", "heur", "astar"]
FORMAT = "mymyr-fork-golden/1"
MAX_BYTES = 1_950_000  # every file stays below 2 MB


def inst(tag, prob, root=BENCH):
    """{"name", "tag", "domain", "problem"} for one BENCH-relative (or root-relative) instance."""
    d = os.path.join(root, tag.split("/", 1)[1] if tag.startswith("data/") else tag)
    stem = os.path.splitext(prob)[0]
    dom = None
    for cand in (f"domain_{stem}.pddl", f"domain-{stem}.pddl", "domain.pddl"):
        if os.path.exists(os.path.join(d, cand)):
            dom = os.path.join(d, cand)
            break
    return {"name": f"{tag.split('/')[-1]}/{stem}", "tag": tag, "domain": dom or os.path.join(d, "domain.pddl"),
            "problem": os.path.join(d, prob)}


# The BrFS suite: one lifted-successor-generator instance per domain, picked for a range of branching factors and
# object counts. "adl/philosophers" is patched (its requirements line adds :derived-predicates etc. so the C++
# parser accepts it) and shipped in the repository instead of the fork's data.
BFS_INSTANCES = [
    inst("strips/gripper", "prob05.pddl"),
    inst("strips/blocks", "probBLOCKS-8-0.pddl"),
    inst("strips/logistics00", "probLOGISTICS-6-1.pddl"),
    inst("strips/miconic", "s7-4.pddl"),
    inst("strips/visitall", "visitall_x-6_y-3_r-100.pddl"),
    inst("strips/sokoban-opt08-strips", "p14.pddl"),
    inst("strips/depot", "p02.pddl"),
    inst("strips/driverlog", "p03.pddl"),
    inst("strips/rovers", "p02.pddl"),
    inst("strips/zenotravel", "p05.pddl"),
    inst("strips/transport-opt08-strips", "p23.pddl"),
    inst("strips/freecell", "p02.pddl"),
    inst("strips/snake-opt18-strips", "p05.pddl"),
    inst("strips/parcprinter-opt11-strips", "p03.pddl"),
    inst("strips/pegsol-08-strips", "p22.pddl"),
    inst("adl/miconic-simpleadl", "s10-2.pddl"),
    inst("adl/caldera-split-opt18-adl", "p04.pddl"),
    inst("adl/pathways", "p02.pddl"),
    inst("adl/folding-opt23-adl", "p01.pddl"),
    inst("adl/openstacks-opt08-adl", "p03.pddl"),
    inst("adl/philosophers", "p03-phil4.pddl", root=os.path.join(REPO, "tests", "data", "pddl")),
    # extreme lifted case (large action schemas); the C++ lifted generator is expected to time out
    inst("strips/organic-synthesis-opt18-strips", "p20.pddl"),
]


def iw_paths(tag, prob):
    if tag.startswith("ipc/"):
        d = os.path.join(IPC, tag[4:], "test")
        return os.path.join(d, "domain.pddl"), os.path.join(d, prob)
    d = os.path.join(BENCH, tag)
    stem = os.path.splitext(prob)[0]
    for cand in (f"domain_{stem}.pddl", f"domain-{stem}.pddl", "domain.pddl"):
        if os.path.exists(os.path.join(d, cand)):
            return os.path.join(d, cand), os.path.join(d, prob)
    return os.path.join(d, "domain.pddl"), os.path.join(d, prob)


def suite_tasks():
    """[(name, domain, problem, sets, tag, problem file)] deduplicated by the PDDL pair."""
    by_pddl = {}
    for ins in BFS_INSTANCES:
        name = ins["name"].replace("/", "__")
        key = (os.path.realpath(ins["domain"]), os.path.realpath(ins["problem"]))
        by_pddl[key] = {"name": name, "domain": ins["domain"], "problem": ins["problem"], "sets": ["brfs_suite"],
                        "tag": ins["tag"], "problem_file": os.path.basename(ins["problem"])}
    for f in sorted(os.listdir(os.path.join(REPO, "tests", "data", "tasks_iw"))):
        kind, dom, prob = f[:-4].split("__")
        d, p = iw_paths(f"{kind}/{dom}", prob + ".pddl")
        key = (os.path.realpath(d), os.path.realpath(p))
        if key in by_pddl:
            by_pddl[key]["sets"].append("iw_suite")
        else:
            by_pddl[key] = {"name": f"{dom}__{prob}", "domain": d, "problem": p, "sets": ["iw_suite"],
                            "tag": f"{kind}/{dom}", "problem_file": prob + ".pddl"}
    names = [t["name"] for t in by_pddl.values()]
    assert len(names) == len(set(names)), "task names collide"
    return sorted(by_pddl.values(), key=lambda t: t["name"])


def rss_mb(pid):
    try:
        if MAC:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            return int(out or 0) / 1024
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


class Runner:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.live = {}  # pid -> Popen

    def total_rss(self):
        with self.lock:
            pids = list(self.live)
        return sum(rss_mb(p) for p in pids)

    def run_phases(self, task, phases, deadline, extra=()):
        """Runs one process; returns (finished {phase: json}, killed (phase, reason) or None, peak MB)."""
        cmd = [EXE, "--domain", task["domain"], "--problem", task["problem"], "--phases", ",".join(phases)] + list(extra) + self.a.exe_args
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        with self.lock:
            self.live[p.pid] = p
        finished, lines, err = {}, [], None
        reader = threading.Thread(target=lambda: lines.extend(iter(p.stdout.readline, "")), daemon=True)
        reader.start()
        killed, peak = None, 0.0
        while p.poll() is None:
            peak = max(peak, rss_mb(p.pid))
            reason = "timeout" if time.time() > deadline else ("rss" if peak > self.a.rss_mb else None)
            if reason:
                p.kill()
                p.wait()
                killed = reason
                break
            time.sleep(0.2)
        reader.join(timeout=10)
        stderr = p.stderr.read() if p.stderr else ""
        with self.lock:
            self.live.pop(p.pid, None)
        for line in lines:
            if line.startswith("PHASE "):
                _, name, js = line.rstrip("\n").split(" ", 2)
                finished[name] = json.loads(js)
            elif line.startswith("ERROR "):
                err = json.loads(line[6:])
        if killed is None and p.returncode not in (0, None) and err is None:
            err = f"exit {p.returncode}: {stderr.strip()[-300:]}"
        pending = [ph for ph in phases if ph not in finished]
        kill = (pending[0], killed) if killed and pending else None
        if err and pending and not kill:
            kill = (pending[0], "error: " + err)
        return finished, kill, peak

    def export(self, task):
        while self.total_rss() > self.a.max_total_rss_mb:  # memory gate before starting another process
            time.sleep(1.0)
        t0 = time.time()
        deadline = t0 + self.a.timeout
        todo = [ph for ph in PHASES if ph in self.a.phases]
        done, killed, peak = {}, {}, 0.0
        while todo:
            # A* uses h_max on unit-cost tasks; when grounding h_max already failed, it falls back to blind.
            extra = ["--astar-h", "blind"] if "heur" in killed else []
            fin, kill, pk = self.run_phases(task, todo, deadline if not killed else time.time() + self.a.retry_timeout, extra)
            peak = max(peak, pk)
            done.update(fin)
            todo = [ph for ph in todo if ph not in fin]
            if kill:
                killed[kill[0]] = kill[1]
                todo = [ph for ph in todo if ph != kill[0] and not (kill[0] == "walks" and ph == "heur")]
            elif todo:  # process ended without finishing (should not happen)
                for ph in todo:
                    killed[ph] = "not run"
                break
        # h_FF breaks ties among achievers in an order that depends on heap addresses, so it can change from one fork
        # process to the next: evaluate h in further independent processes and keep every value observed. A task whose
        # values vary gets --heur-runs-unstable runs in total.
        heur_repeats = []
        if "heur" in done:
            target = self.a.heur_runs
            while len(heur_repeats) + 1 < target:
                fin, _, pk = self.run_phases(task, ["heur"], time.time() + self.a.retry_timeout)
                peak = max(peak, pk)
                if "heur" not in fin:
                    break
                heur_repeats.append(fin["heur"])
                if fin["heur"]["walks"] != done["heur"]["walks"]:
                    target = max(target, self.a.heur_runs_unstable)
        return assemble(task, done, killed, peak, time.time() - t0, self.a, heur_repeats)


def assemble(task, done, killed, peak_mb, wall_s, a, heur_repeats=()):
    meta = {"wall_s": round(wall_s, 3), "peak_rss_mb": round(peak_mb, 1)}
    out = {"format": FORMAT, "task": task["name"], "sets": task["sets"],
           "source": {"tag": task["tag"], "problem": task["problem_file"],
                      "domain_file": os.path.basename(task["domain"])},
           "fork": fork_info(), "names": None, "brfs": None, "iw": [], "walks": None, "astar": None,
           "killed": killed, "meta": meta}
    if "task" in done:
        out["names"] = done["task"]
    for ph in ("brfs", "astar"):
        if ph in done:
            meta[ph] = done[ph].pop("meta", None)
            out[ph] = done[ph]
    for k in (1, 2):
        ph = f"iw{k}"
        if ph in done:
            meta[ph] = done[ph].pop("meta", None)
            out["iw"].append(done[ph])
    if "walks" in done:
        w = done["walks"]
        meta["walks"] = w.pop("meta", None)
        if "heur" in done:
            meta["heur"] = done["heur"].pop("meta", None)
            unstable = {"hmax": 0, "hadd": 0, "hff": 0}
            shape = [len(walk["steps"]) for walk in w["walks"]]
            heur_repeats = [r for r in heur_repeats if [len(x) for x in r["walks"]] == shape]  # same walks only
            for wi, (walk, hs) in enumerate(zip(w["walks"], done["heur"]["walks"])):
                for si, (step, h) in enumerate(zip(walk["steps"], hs)):
                    step["h"].update(h)
                    for key in unstable:
                        seen = {h[key]} | {r["walks"][wi][si][key] for r in heur_repeats}
                        if len(seen) > 1:  # null (infinity) sorts last
                            step["h"][key + "_observed"] = sorted(seen, key=lambda v: (v is None, v or 0))
                            unstable[key] += 1
            w["heur_runs"] = 1 + len(heur_repeats)
            w["h_unstable_steps"] = unstable
        w["heuristics"] = ["blind", "goal_count"] + (["hmax", "hadd", "hff"] if "heur" in done else [])
        out["walks"] = w
    fit_size(out)
    return out


FORK = None


def fork_info():
    global FORK
    if FORK is None:
        FORK = {"version": "0.16.3", "commit": "8033459612baa36745e225d2cebd1e99d2db4b04",
                "successor_generator": "lifted KPKC, symmetry pruning off"}
    return FORK


def dumps(out):
    """Top level one key per line; every walk step on its own line (compact JSON inside)."""
    parts = []
    for k, v in out.items():
        if k == "walks" and v is not None:
            inner = []
            for kk, vv in v.items():
                if kk == "walks":
                    ws = []
                    for walk in vv:
                        head = ",".join(f"{json.dumps(x)}:{json.dumps(y, separators=(',', ':'))}"
                                        for x, y in walk.items() if x != "steps")
                        steps = ",\n".join("    " + json.dumps(s, separators=(",", ":")) for s in walk["steps"])
                        ws.append("   {" + head + ',"steps":[\n' + steps + "]}")
                    inner.append('  "walks":[\n' + ",\n".join(ws) + "]")
                else:
                    inner.append(f"  {json.dumps(kk)}:{json.dumps(vv, separators=(',', ':'))}")
            parts.append(f' {json.dumps(k)}:{{\n' + ",\n".join(inner) + "}")
        else:
            parts.append(f" {json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def fit_size(out):
    """Drops the item lists of the largest sets (applicable actions first, then atoms) until the file fits;
    the count and the hash of every set always stay."""
    dropped = {"applicable": 0, "atoms": 0}
    out["walks_items_dropped"] = dropped
    if not out.get("walks"):
        return
    size = len(dumps(out).encode())
    if size <= MAX_BYTES:
        return
    for keys, label in ((("applicable",), "applicable"), (("fluent_atoms", "derived_atoms"), "atoms")):
        sets = [(len(json.dumps(st[k]["items"])), wi, si, k)
                for wi, walk in enumerate(out["walks"]["walks"]) for si, st in enumerate(walk["steps"])
                for k in keys if "items" in st[k]]
        # Largest first; ties by position, so the result is deterministic.
        for nbytes, wi, si, k in sorted(sets, key=lambda x: (-x[0], x[1], x[2], x[3])):
            del out["walks"]["walks"][wi]["steps"][si][k]["items"]
            dropped[label] += 1
            size -= nbytes + len(',"items":')
            if size <= MAX_BYTES:
                break
        size = len(dumps(out).encode())
        if size <= MAX_BYTES:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated substrings of task names")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per task (all phases)")
    ap.add_argument("--retry-timeout", type=float, default=900, help="seconds for a process that resumes after a kill")
    ap.add_argument("--rss-mb", type=float, default=4000, help="per-process resident memory limit")
    ap.add_argument("--heur-runs", type=int, default=5, help="independent processes that evaluate h on the walks")
    ap.add_argument("--heur-runs-unstable", type=int, default=50, help="processes when the h values vary")
    ap.add_argument("--max-total-rss-mb", type=float, default=12000, help="do not start a task above this total")
    ap.add_argument("--out", default=EXPECTED)
    ap.add_argument("--list", action="store_true", help="only list the tasks and their PDDL")
    ap.add_argument("exe_args", nargs="*", help="extra export_golden arguments (after --)")
    a = ap.parse_args()
    a.phases = a.phases.split(",")
    tasks = suite_tasks()
    if a.only:
        subs = a.only.split(",")
        tasks = [t for t in tasks if any(s in t["name"] for s in subs)]
    missing = [t for t in tasks if not (os.path.exists(t["domain"]) and os.path.exists(t["problem"]))]
    for t in missing:
        print(f"MISSING PDDL {t['name']}: {t['domain']} {t['problem']}", flush=True)
    tasks = [t for t in tasks if t not in missing]
    if a.list:
        for t in tasks:
            print(t["name"], ",".join(t["sets"]), t["domain"], t["problem"])
        return
    os.makedirs(a.out, exist_ok=True)
    runner = Runner(a)
    t_start = time.time()

    def one(task):
        out = runner.export(task)
        text = dumps(out)
        with open(os.path.join(a.out, task["name"] + ".json"), "w") as fh:
            fh.write(text)
        b = out.get("brfs") or {}
        iw = {r["k"]: r for r in out["iw"]}
        print(f"[{time.time() - t_start:7.1f}s] {task['name']:48} brfs={b.get('status')}/{b.get('states')} "
              f"iw1={iw.get(1, {}).get('status')} iw2={iw.get(2, {}).get('status')} "
              f"astar={(out.get('astar') or {}).get('optimal_cost')} killed={out['killed'] or '-'} "
              f"dropped={out['walks_items_dropped']} size={len(text) / 1e6:.2f}MB peak={out['meta']['peak_rss_mb']:.0f}MB",
              flush=True)
        return out

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(one, tasks))
    print(f"\n{len(results)} tasks written to {a.out}; missing PDDL: {len(missing)}")


if __name__ == "__main__":
    main()
