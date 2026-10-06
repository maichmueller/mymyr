"""mymyr.Domain: the loki front end from Python (parse and normalize once, instantiate many, thread-safe)."""

import pathlib
import sys
import threading

import pytest

import mymyr
from mymyr import formalism as fm

ROOT = pathlib.Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "tests/data/pddl/blocks"
PHILOSOPHERS = ROOT / "tests/data/pddl/philosophers"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def test_instantiate_returns_normalized_task():
    dom = mymyr.Domain.from_file(PHILOSOPHERS / "domain.pddl")
    task = dom.instantiate(PHILOSOPHERS / "p03-phil4.pddl")
    assert isinstance(task, fm.NormalizedTask)
    assert task.domain_name == dom.name
    assert len(task.objects) > 0 and len(task.axioms) > 0
    assert repr(dom) == f"Domain({dom.name})"


def test_untyped_task_equals_the_fork_export():
    # blocks is untyped, so the fork's export is deterministic and must match token for token. The golden is
    # read and rewritten first: the fork's lifted exporter always writes an axiom section ("X 0"), ours only when needed.
    task = mymyr.Domain.from_file(BLOCKS / "domain.pddl").instantiate(BLOCKS / "probBLOCKS-8-0.pddl")
    golden = fm.read_task_text(str(ROOT / "tests/data/golden/suite/blocks__probBLOCKS-8-0.txt"))
    assert task.to_text().split() == golden.to_text().split()


def test_fast_init_equals_full_path():
    for d in (BLOCKS, PHILOSOPHERS):
        dom = mymyr.Domain.from_file(d / "domain.pddl")
        problem = next(p for p in sorted(d.glob("*.pddl")) if p.name != "domain.pddl")
        assert dom.instantiate(problem, fast_init=True).to_text() == dom.instantiate(problem, fast_init=False).to_text()


def test_from_string_equals_from_file():
    dom_file = mymyr.Domain.from_file(BLOCKS / "domain.pddl")
    dom_text = mymyr.Domain.from_string((BLOCKS / "domain.pddl").read_text())
    problem = (BLOCKS / "probBLOCKS-8-0.pddl").read_text()
    assert dom_text.instantiate_string(problem).to_text() == dom_file.instantiate_string(problem).to_text()


def test_concurrent_instantiation_on_one_domain():
    dom = mymyr.Domain.from_file(PHILOSOPHERS / "domain.pddl")
    problem = PHILOSOPHERS / "p03-phil4.pddl"
    expected = dom.instantiate(problem).to_text()
    results, errors = [], []

    def work(k):
        try:
            for i in range(4):
                results.append(dom.instantiate(problem, fast_init=(i + k) % 2 == 0).to_text())
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 64 and all(r == expected for r in results)
    assert not sys._is_gil_enabled()


def test_errors_surface_as_exceptions():
    dom = mymyr.Domain.from_file(BLOCKS / "domain.pddl")
    with pytest.raises(Exception):
        dom.instantiate(BLOCKS / "does-not-exist.pddl")
    with pytest.raises(Exception):
        dom.instantiate_string("(define (problem broken) (:domain blocks) (:objects a - nosuchtype))")
