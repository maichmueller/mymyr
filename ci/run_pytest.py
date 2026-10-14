#!/usr/bin/env python3
"""Run pytest and fail when fewer tests pass than the job's environment can run (ci/expected_counts.json).

usage: run_pytest.py --runtimes {none,rl} [--platform {linux,macos}] [pytest arguments...]

Tests that need the benchmark data skip when it is missing, so a run with a misconfigured data
directory can finish green with most of the suite skipped. This wrapper turns that into a failure.

The floors are measured with the full test data. The tests that need torch, JAX, gymnasium and their RL libraries skip
without them, so the floor depends on what the job installs:
  --runtimes none  no torch, JAX or gymnasium (pytest_min_passed_plain): the Linux 3.15t job, the macOS job, the wheel tests;
  --runtimes rl    ci/rl-requirements.txt installed (pytest_min_passed_rl): the Linux 3.14t job.
Five tests run only on Linux (pytest_linux_only_tests): the ones in test_threads.py that limit the address space
with RLIMIT_AS and read /proc (4 + 1 cases). The floor on macOS is lower by that many; the platform defaults to
the one running the check, as in check_ctest_log.py.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest


class Counter:
    def __init__(self):
        self.passed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtimes", choices=("none", "rl"), required=True,
                        help="the RL frameworks the job has installed")
    parser.add_argument("--platform", choices=("linux", "macos"),
                        default="macos" if sys.platform == "darwin" else "linux",
                        help="platform the tests run on (default: this one)")
    args, pytest_args = parser.parse_known_args(argv[1:])
    expected = json.loads(Path(__file__).with_name("expected_counts.json").read_text())
    minimum = expected["pytest_min_passed_rl" if args.runtimes == "rl" else "pytest_min_passed_plain"]
    if args.platform == "macos":
        minimum -= expected["pytest_linux_only_tests"]
    counter = Counter()
    code = int(pytest.main(pytest_args, plugins=[counter]))
    print(f"run_pytest: {counter.passed} passed, at least {minimum} expected")
    if code == 0 and counter.passed < minimum:
        print(f"error: only {counter.passed} tests passed (expected at least {minimum}); "
              "check that MYMYR_PARITY_DATA points at the parity test data", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
