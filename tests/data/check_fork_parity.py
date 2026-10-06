#!/usr/bin/env python3
"""Broad parity check of mymyr's PDDL front end against the mimir fork, over whole IPC directories.

For every problem of the fork's IPC-2023 data (data/ipc/<domain>[-ipc]/<split>/*.pddl):
  1. the fork's export_numeric_fork writes the normalized task (ASLR off, cached in <out>/fork/);
  2. mymyr_export_task instantiates it (fast :init path) and writes both tasks in the canonical form of the golden
     tests (tests/cpp/frontend/golden.hpp), which must be equal;
  3. for problems up to --full-max-bytes, mymyr's full loki path must give the same task as the fast path.
Prints one line per task and a summary; exits 1 on any difference.

  tests/data/check_fork_parity.py --exporter .work/build-numeric-exporter/export_numeric_fork \\
      --mymyr build/dev/tests/cpp/mymyr_export_task --ipc <fork checkout>/data/ipc --out /tmp/parity [--splits test]
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import shutil
import subprocess
import sys
import time


def tasks(ipc, splits):
    for dom in sorted(os.listdir(ipc)):
        for split in splits:
            d = os.path.join(ipc, dom, split)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".pddl") and not f.startswith("domain"):
                    name = f"{dom.removesuffix('-ipc')}__{split}__{f[:-5]}"
                    yield name, os.path.join(d, "domain.pddl"), os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exporter", required=True)
    ap.add_argument("--mymyr", required=True)
    ap.add_argument("--ipc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--full-max-bytes", type=int, default=300_000)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "fork"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "canon"), exist_ok=True)
    norand = []
    if shutil.which("setarch") and subprocess.run(["setarch", platform.machine(), "-R", "true"]).returncode == 0:
        norand = ["setarch", platform.machine(), "-R"]
    todo = [t for t in tasks(a.ipc, a.splits.split(",")) if a.only in t[0]]

    def fork(t):
        name, dom, prob = t
        dst = os.path.join(a.out, "fork", name + ".txt")
        if os.path.exists(dst):
            return name, None
        t0 = time.time()
        try:
            r = subprocess.run(norand + [a.exporter, dom, prob, dst + ".tmp"], capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            return name, "fork timeout"
        if r.returncode != 0:
            return name, "fork failed: " + r.stderr.strip()[-200:]
        os.replace(dst + ".tmp", dst)
        with open(os.path.join(a.out, "fork", name + ".time"), "w") as f:
            f.write(f"{time.time() - t0:.3f}\n")
        return name, None

    fork_errors = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, err in ex.map(fork, todo):
            if err:
                fork_errors[name] = err

    ok = full_ok = full_checked = 0
    failures = []
    for name, dom, prob in todo:
        if name in fork_errors:
            print(f"{name:45} SKIP ({fork_errors[name]})", flush=True)
            continue
        golden = os.path.join(a.out, "fork", name + ".txt")
        canon = os.path.join(a.out, "canon", name)
        r = subprocess.run([a.mymyr, dom, prob, "--golden", golden, canon], capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(name)
            print(f"{name:45} MYMYR ERROR {r.stderr.strip()[-300:]}", flush=True)
            continue
        info = json.loads(r.stdout.strip().splitlines()[-1])
        same = open(canon + ".ours").read().split() == open(canon + ".golden").read().split()
        ok += same
        ft = os.path.join(a.out, "fork", name + ".time")
        fork_s = float(open(ft).read()) if os.path.exists(ft) else float("nan")
        line = f"{name:45} {'EQUAL' if same else 'DIFF '} mymyr {info['instantiate_s']:.3f}s fork-export {fork_s:.2f}s atoms {info['static_init'] + info['fluent_init']}"
        if not same:
            failures.append(name)
        if os.path.getsize(prob) <= a.full_max_bytes:
            full_checked += 1
            rf = subprocess.run([a.mymyr, dom, prob, canon + ".full.txt", "--full-init"], capture_output=True, text=True)
            rq = subprocess.run([a.mymyr, dom, prob, canon + ".fast.txt"], capture_output=True, text=True)
            fsame = rf.returncode == 0 and rq.returncode == 0 and open(canon + ".full.txt").read() == open(canon + ".fast.txt").read()
            full_ok += fsame
            line += " fast==full" if fsame else " FAST!=FULL"
            if not fsame:
                failures.append(name + " (fast vs full)")
        print(line, flush=True)
    n = len(todo) - len(fork_errors)
    print(f"\nparity: {ok}/{n} equal to the fork (canonical); fast==full {full_ok}/{full_checked}; "
          f"fork export failures {len(fork_errors)}; differences {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
