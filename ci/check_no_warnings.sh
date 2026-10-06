#!/usr/bin/env bash
# Fail when a compiler log contains warnings.
#
# usage: ci/check_no_warnings.sh LOG [--report-only]
#
# Prints every warning line; with --report-only the exit status stays 0. Counts the diagnostics of the
# compilers and the linker (GCC, Clang and ld: "file:line:col: warning:", nvcc: "file(line): warning #N"),
# not the notices of other tools in the same log (uv: "warning: Failed to hardlink files").
set -euo pipefail
pattern=': warning:|\): warning #'

if [ $# -lt 1 ]; then
    echo "usage: $0 LOG [--report-only]" >&2
    exit 2
fi
log=$1
count=$(grep -c -E "$pattern" "$log" || true)
if [ "$count" -eq 0 ]; then
    echo "compiler warnings: 0"
    exit 0
fi
echo "compiler warnings: $count"
grep -E "$pattern" "$log" | sort | uniq -c | sort -rn | head -50
if [ "${2:-}" = "--report-only" ]; then
    exit 0
fi
echo "error: the build must be free of warnings" >&2
exit 1
