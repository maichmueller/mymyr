"""Shared setup of the parity tests: where the recorded reference results and the reference data live."""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = ROOT / "tests/parity/data"

# The reference checkouts (env MYMYR_PARITY_DATA): mimir-fork/data (the fork's data directory with ipc/), mimir-cs/
# (classical and numeric benchmarks) and mimir/ (upstream benchmarks). Tests that need them skip when it is unset.
_PARITY = os.environ.get("MYMYR_PARITY_DATA")
PARITY_DATA = pathlib.Path(_PARITY) if _PARITY else None
FORK_DATA = PARITY_DATA / "mimir-fork" / "data" if PARITY_DATA else pathlib.Path("/nonexistent/mymyr-parity-data")
WORK = PARITY_DATA if PARITY_DATA else pathlib.Path("/nonexistent/mymyr-parity-data")
IPC = FORK_DATA / "ipc" if PARITY_DATA else None
