#!/usr/bin/env python3
"""Fails when a file outside the parity area names one of the reference implementations.

usage: ci/check_no_mimir.py [ROOT]

Every file tracked by git under ROOT (default: the repository this script lives in; without git, every file below it)
is searched, case-insensitively, for the names of the reference implementations. Allowed places:
  - tests/parity/ and bench/parity/, where the parity tests, their data, tools and drivers live;
  - LICENSE;
  - the acknowledgement and the license lines of README.md (the lines that link the project or state its terms);
  - this script (other files may name it).
"""

import re
import subprocess
import sys
from pathlib import Path

NAMES = re.compile("mimir|pymimir|mifrost", re.IGNORECASE)
ALLOWED_PREFIXES = ("tests/parity/", "bench/parity/")
ALLOWED_FILES = {"LICENSE", "ci/check_no_mimir.py"}
SELF_NAME = "check_no_" + "mimir"  # other files may name this script
# the README lines that may name it: the acknowledgement link and the license provenance
README_ALLOWED = (re.compile(r"\]\(https://github\.com/maichmueller/mimir\)"), re.compile(r"terms of mimir"))
SKIPPED_DIRS = {".git", "build", ".work", ".venv", ".venv-rl", "__pycache__", ".claude"}


def tracked_files(root: Path) -> list[str]:
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             check=True).stdout.strip()
        if Path(top).resolve() != root.resolve():
            raise subprocess.CalledProcessError(1, "git")
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        return [p for p in out.decode().split("\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [str(p.relative_to(root)) for p in sorted(root.rglob("*"))
                if p.is_file() and not SKIPPED_DIRS & set(p.relative_to(root).parts)]


def violations(root: Path) -> list[str]:
    found = []
    for rel in tracked_files(root):
        if rel in ALLOWED_FILES or rel.startswith(ALLOWED_PREFIXES):
            continue
        path = root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:4096]:  # binary
            if NAMES.search(rel):
                found.append(f"{rel}: the file name")
            continue
        if NAMES.search(rel):
            found.append(f"{rel}: the file name")
        for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if not NAMES.search(line.replace(SELF_NAME, "")):
                continue
            if rel == "README.md" and any(p.search(line) for p in README_ALLOWED):
                continue
            found.append(f"{rel}:{number}: {line.strip()[:120]}")
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    found = violations(root)
    for v in found:
        print(v)
    if found:
        print(f"error: {len(found)} references to a reference implementation outside the parity area", file=sys.stderr)
        return 1
    print("check_no_mimir: no references outside the parity area")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
