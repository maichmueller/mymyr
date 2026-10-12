"""Parity: the front end's text export against the recorded reference export."""

import pytest

import mymyr
from mymyr import formalism as fm

from parity_support import DATA, ROOT

BLOCKS = ROOT / "tests/data/pddl/blocks"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def test_untyped_task_equals_the_reference_export():
    # blocks is untyped, so the reference export is deterministic and must match token for token. The reference's lifted
    # exporter always writes an axiom section ("X 0"), ours only when needed; both sides go through read_task_text.
    task = mymyr.Domain.from_file(BLOCKS / "domain.pddl").instantiate(BLOCKS / "probBLOCKS-8-0.pddl")
    golden = fm.read_task_text(str(DATA / "golden/suite/blocks__probBLOCKS-8-0.txt"))
    assert task.to_text().split() == golden.to_text().split()
