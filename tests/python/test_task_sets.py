"""Task sets: mymyr.rl.TaskTable.from_pddl, the table of a domain's problem files (a list, a directory or a glob), and
the APIs that take a table of one domain (generalized state spaces, knowledge bases, task suites)."""

import pickle
import shutil

import pytest

import mymyr
from mymyr import datasets
from mymyr.rl import TaskSuite, TaskTable

from conftest import ROOT

COUNTERS = ROOT / "tests/data/pddl/counters"

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def fingerprints(table):
    return [t.fingerprint for t in table.tasks]


def instantiated(domain, problems, **options):
    """The tasks Domain.instantiate gives for these problem files."""
    d = mymyr.Domain.from_file(domain)
    return [mymyr.Task(d.instantiate(p), **options).fingerprint for p in problems]


@pytest.fixture
def problems_dir(tmp_path):
    """A directory with the domain (twice: a copy under another name, after a comment), two problems, a problem in a
    subdirectory and a file that is not PDDL."""
    shutil.copy(COUNTERS / "domain.pddl", tmp_path / "domain.pddl")
    (tmp_path / "other-domain.pddl").write_text("; a copy\n  " + (COUNTERS / "domain.pddl").read_text().upper())
    shutil.copy(COUNTERS / "p02.pddl", tmp_path / "b.pddl")
    shutil.copy(COUNTERS / "p01.pddl", tmp_path / "a.pddl")
    (tmp_path / "sub").mkdir()
    shutil.copy(COUNTERS / "p01.pddl", tmp_path / "sub" / "c.pddl")
    (tmp_path / "notes.txt").write_text("not a problem\n")
    return tmp_path


def test_files_keep_their_order():
    files = [COUNTERS / "p02.pddl", COUNTERS / "p01.pddl"]
    table = TaskTable.from_pddl(COUNTERS / "domain.pddl", files)
    assert len(table) == 2
    assert fingerprints(table) == instantiated(COUNTERS / "domain.pddl", files)
    assert [t.problem_name for t in table.tasks] == [mymyr.Task.from_pddl(COUNTERS / "domain.pddl", f).problem_name for f in files]
    # one file, str paths, a parsed Domain
    one = TaskTable.from_pddl(str(COUNTERS / "domain.pddl"), str(COUNTERS / "p01.pddl"))
    assert fingerprints(one) == fingerprints(table)[1:]
    domain = mymyr.Domain.from_file(COUNTERS / "domain.pddl")
    assert fingerprints(TaskTable.from_pddl(domain, files)) == fingerprints(table)


def test_a_directory_gives_its_problem_files_sorted(problems_dir):
    table = TaskTable.from_pddl(problems_dir / "domain.pddl", problems_dir)
    # a.pddl, b.pddl: not the domain, not the text file, not the subdirectory
    assert fingerprints(table) == instantiated(problems_dir / "domain.pddl", [problems_dir / "a.pddl", problems_dir / "b.pddl"])
    # the domain from elsewhere: the same problems
    assert fingerprints(TaskTable.from_pddl(COUNTERS / "domain.pddl", problems_dir)) == fingerprints(table)


def test_a_glob_gives_its_matches_sorted(problems_dir):
    flat = TaskTable.from_pddl(problems_dir / "domain.pddl", str(problems_dir / "*.pddl"))
    files = [problems_dir / "a.pddl", problems_dir / "b.pddl"]  # domain files are skipped
    assert fingerprints(flat) == instantiated(problems_dir / "domain.pddl", files)
    deep = TaskTable.from_pddl(problems_dir / "domain.pddl", str(problems_dir / "**" / "*.pddl"))
    assert fingerprints(deep) == instantiated(problems_dir / "domain.pddl", [*files, problems_dir / "sub" / "c.pddl"])
    assert fingerprints(TaskTable.from_pddl(problems_dir / "domain.pddl", str(problems_dir / "?.pddl"))) == fingerprints(flat)


def test_options_and_threads(problems_dir):
    files = sorted(problems_dir.glob("**/[abc].pddl"))
    one = TaskTable.from_pddl(problems_dir / "domain.pddl", files, atoms="frozen", threads=1)
    many = TaskTable.from_pddl(problems_dir / "domain.pddl", files, atoms="frozen", threads=4)
    assert fingerprints(one) == fingerprints(many) == instantiated(problems_dir / "domain.pddl", files, atoms="frozen")
    assert all(t.atom_mode == "frozen" for t in many.tasks)


def test_pickling_and_indexing(problems_dir):
    table = TaskTable.from_pddl(problems_dir / "domain.pddl", problems_dir)
    back = pickle.loads(pickle.dumps(table))
    assert len(back) == len(table) == 2
    assert fingerprints(back) == fingerprints(table)
    assert table[0].fingerprint == table[-2].fingerprint == fingerprints(table)[0]
    with pytest.raises(IndexError):
        table[2]


def test_errors(problems_dir, tmp_path_factory):
    empty = tmp_path_factory.mktemp("empty")
    with pytest.raises(ValueError, match="no problem files"):
        TaskTable.from_pddl(problems_dir / "domain.pddl", empty)
    with pytest.raises(ValueError, match="no problem files match"):
        TaskTable.from_pddl(problems_dir / "domain.pddl", str(problems_dir / "*.nothing"))
    with pytest.raises(RuntimeError):
        TaskTable.from_pddl(problems_dir / "domain.pddl", [problems_dir / "missing.pddl"])
    with pytest.raises(TypeError):
        TaskTable.from_pddl(problems_dir / "domain.pddl", 3)


def test_the_apis_of_one_domain_take_it(problems_dir):
    table = TaskTable.from_pddl(problems_dir / "domain.pddl", problems_dir, atoms="frozen")
    tasks = list(table.tasks)
    g = datasets.generalized_state_space(table, max_states=2000, threads=1)
    assert g.num_vertices == datasets.generalized_state_space(tasks, max_states=2000, threads=1).num_vertices
    kb = datasets.KnowledgeBase(table, max_states=2000, threads=1)
    assert kb.tasks is table
    assert len(TaskSuite([table])) == len(table)
