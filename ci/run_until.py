#!/usr/bin/env python3
"""Run a build command until a deadline, then stop its Ninja cleanly.

usage: run_until.py DEADLINE -- COMMAND [ARG...]

DEADLINE is a Unix time in seconds. If COMMAND still runs at DEADLINE, every Ninja process among its descendants gets
SIGINT, again every 10 seconds while COMMAND runs (the build tool may start Ninja later). An interrupted Ninja stops
its compiles, removes their partial outputs and keeps the log of the edges it finished, so that the same build, run
again, resumes after them. Only descendants are signalled, so other builds on the machine are left alone. The script
then waits for COMMAND and for the interrupted Ninja processes, which can outlive the build tool that started them.

COMMAND starts with SIGINT at its default action even when this script runs with SIGINT ignored, as a background job
of a non-interactive shell does: an ignored signal stays ignored across exec, so Ninja would ignore the interrupt, and
a shell cannot trap a signal that was ignored when it started.

The descendants are found through /proc, so the script runs on Linux only, as the CUDA wheel build that uses it
does; elsewhere it refuses before starting COMMAND. Should the script itself fail, it still waits for COMMAND.

Exit status: COMMAND's (0 if it succeeded) when no Ninja had to be interrupted; 75 when the build was stopped; 2 for
a wrong command line or a system without /proc.
"""

import os
import signal
import subprocess
import sys
import time

STOPPED = 75
RESEND_SECONDS = 10.0
PROC = "/proc"


def process_name(pid: int) -> str | None:
    try:
        with open(f"{PROC}/{pid}/comm") as stream:
            return stream.read().strip()
    except OSError:
        return None


def descendants(root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir(PROC):
        if not entry.isdigit():
            continue
        try:
            with open(f"{PROC}/{entry}/stat") as stream:
                stat = stream.read()
        except OSError:
            continue
        # pid (comm) state ppid ...; comm may hold spaces and parentheses
        children.setdefault(int(stat.rpartition(")")[2].split()[1]), []).append(int(entry))
    out, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    deadline = float(argv[1])
    if not os.path.isdir(f"{PROC}/self"):
        print(f"run_until: finds the build's Ninja processes through {PROC}, which this system does not have",
              file=sys.stderr)
        return 2
    previous = signal.signal(signal.SIGINT, signal.SIG_DFL)  # inherited by COMMAND, whatever this process ignores
    try:
        process = subprocess.Popen(argv[3:])
    finally:
        signal.signal(signal.SIGINT, previous)
    stopping = False
    interrupted: set[int] = set()
    with process:  # waits for COMMAND also when this loop raises
        while True:
            try:
                status = process.wait(timeout=RESEND_SECONDS if stopping else max(0.0, deadline - time.time()))
                break
            except subprocess.TimeoutExpired:
                if not stopping:
                    print("run_until: stopping the build at the deadline", file=sys.stderr, flush=True)
                stopping = True
                for pid in descendants(process.pid):
                    if process_name(pid) == "ninja":
                        try:
                            os.kill(pid, signal.SIGINT)
                            interrupted.add(pid)
                        except ProcessLookupError:
                            pass
    while any(process_name(pid) == "ninja" for pid in interrupted):
        time.sleep(0.5)
    if interrupted:
        return STOPPED
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
