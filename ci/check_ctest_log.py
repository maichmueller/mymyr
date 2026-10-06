#!/usr/bin/env python3
"""Check the output of a ctest run: no failures, at least ctest_min_total tests, at most ctest_max_skipped
skipped (ci/expected_counts.json).

usage: check_ctest_log.py LOG

Tests that need the benchmark data skip when it is missing, so a run with a misconfigured data
directory exits 0 with most of the suite skipped. This check turns that into a failure. The counts
are also written to the GitHub step summary when available.
"""

import json
import os
import re
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    text = open(argv[1], errors="replace").read()
    expected = json.loads(Path(__file__).with_name("expected_counts.json").read_text())
    min_total, max_skipped = expected["ctest_min_total"], expected["ctest_max_skipped"]
    # CMake 4 leaves out ", 0 tests failed" when none failed: "100% tests passed out of 1205"
    m = re.search(r"(\d+)% tests passed(?:, (\d+) tests failed)? out of (\d+)", text)
    if m is None:
        print("error: no ctest summary found in the log", file=sys.stderr)
        return 1
    failed, total = int(m.group(2) or 0), int(m.group(3))
    skipped = len(re.findall(r"^\s+\d+ - .*\(Skipped\)\s*$", text, re.M))
    line = f"ctest: {total} tests, {failed} failed, {skipped} skipped"
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"{line}\n")
    errors = []
    if failed:
        errors.append(f"{failed} tests failed")
    if total < min_total:
        errors.append(f"only {total} tests were registered (expected at least {min_total})")
    if skipped > max_skipped:
        errors.append(f"{skipped} tests skipped (expected at most {max_skipped}); "
                      "check that MYMYR_FORK_DATA and MYMYR_WORK point at the test data")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
