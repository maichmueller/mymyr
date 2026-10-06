#!/usr/bin/env python3
"""Run pytest and fail when fewer than pytest_min_passed tests pass (ci/expected_counts.json).

usage: run_pytest.py [pytest arguments...]

Tests that need the benchmark data skip when it is missing, so a run with a misconfigured data
directory can finish green with most of the suite skipped. This wrapper turns that into a failure.
"""

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


def main() -> int:
    minimum = json.loads(Path(__file__).with_name("expected_counts.json").read_text())["pytest_min_passed"]
    counter = Counter()
    code = int(pytest.main(sys.argv[1:], plugins=[counter]))
    print(f"run_pytest: {counter.passed} passed, at least {minimum} expected")
    if code == 0 and counter.passed < minimum:
        print(f"error: only {counter.passed} tests passed (expected at least {minimum}); "
              "check that MYMYR_FORK_DATA, MYMYR_IPC and MYMYR_WORK point at the test data", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
