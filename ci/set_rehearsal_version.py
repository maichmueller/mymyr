#!/usr/bin/env python3
"""Set the package version of a TestPyPI rehearsal or of a build with a local version label.

usage: set_rehearsal_version.py RUN_ID RUN_ATTEMPT
       set_rehearsal_version.py --version VERSION
       set_rehearsal_version.py --local LABEL

A rehearsal version is BASE.devN, where N is RUN_ID * 1000000 + RUN_ATTEMPT.
An explicit version must have the same base as pyproject.toml.
A local label gives BASE+LABEL (e.g. --local cu13 for the CUDA wheels); a label is lowercase letters and digits,
in dot-separated parts.
"""

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    path = ROOT / "pyproject.toml"
    text = path.read_text()
    base = tomllib.loads(text)["project"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", base):
        print(f"error: expected a final base version, got {base!r}", file=sys.stderr)
        return 1
    rehearsal = True
    if argv[1] == "--version":
        version = argv[2]
        if not re.fullmatch(re.escape(base) + r"\.dev[1-9][0-9]*", version):
            print(f"error: expected {base}.devN, got {version!r}", file=sys.stderr)
            return 1
    elif argv[1] == "--local":
        if not re.fullmatch(r"[a-z0-9]+(\.[a-z0-9]+)*", argv[2]):
            print(f"error: expected a local label of lowercase letters and digits, got {argv[2]!r}", file=sys.stderr)
            return 1
        version = f"{base}+{argv[2]}"
        rehearsal = False
    else:
        if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in argv[1:]):
            print("error: run id and attempt must be positive integers", file=sys.stderr)
            return 1
        run_id, attempt = map(int, argv[1:])
        if attempt >= 1_000_000:
            print("error: run attempt must be less than 1000000", file=sys.stderr)
            return 1
        version = f"{base}.dev{run_id * 1_000_000 + attempt}"

    updated, count = re.subn(r'(?m)^version\s*=\s*"' + re.escape(base) + r'"[ \t]*$',
                             f'version = "{version}"', text, count=1)
    if count != 1 or tomllib.loads(updated)["project"]["version"] != version:
        print("error: could not update the project version", file=sys.stderr)
        return 1
    path.write_text(updated)
    print(f"{'rehearsal' if rehearsal else 'package'} version {version}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"version={version}\nprerelease={'true' if rehearsal else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
