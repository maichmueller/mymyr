"""Arguments are checked where they enter the library: an int or float out of range raises ValueError naming the
argument and its range, a non-number raises TypeError naming the argument, and None keeps its meaning where a
parameter takes it."""

import re

import numpy as np
import pytest

import mymyr
from mymyr import datasets, rl, search

from conftest import BLOCKS, TASKS, text_task

SIGNATURE = re.compile(r"^(\w+)\((.*)\)( -> .*)?$")


def parameters(doc):
    """(names, required names, int names) of the signature line of a binding's docstring, or None."""
    m = SIGNATURE.match(doc.splitlines()[0]) if doc else None
    if not m:
        return None
    depth, cur, parts = 0, "", []
    for ch in m.group(2):
        depth += ch in "[(" or -(ch in "])")
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    parts = [p for p in parts if p and p not in ("*", "/")]
    names = [p.split(":")[0].strip() for p in parts]
    required = [p.split(":")[0].strip() for p in parts if ":" in p and "=" not in p]
    ints = [p.split(":")[0].strip() for p in parts if re.match(r"^\w+: int( \| None)?( = .*)?$", p)]
    return names, required, ints


def int_cases():
    """(function, parameter) for every int parameter of the public functions whose one required argument is a task."""
    out = []
    for mod in (search, datasets, rl):
        for name in sorted(dir(mod)):
            f = getattr(mod, name)
            if name.startswith("_") or not callable(f) or isinstance(f, type):
                continue
            sig = parameters(f.__doc__ or "")
            if sig and sig[1] in (["task"], ["table"]):
                out += [(f"{mod.__name__.rsplit('.', 1)[-1]}.{name}", p) for p in sig[2]]
    return out


CASES = int_cases()


@pytest.fixture(scope="module")
def task():
    return text_task("blocks__probBLOCKS-8-0")


def test_the_sweep_covers_the_searches_and_datasets():
    names = {f for f, _ in CASES}
    assert {"search.astar", "search.iw", "search.brfs", "search.beam", "search.siw", "datasets.generate"} <= names
    assert len(CASES) >= 100


@pytest.mark.parametrize(("function", "param"), CASES, ids=[f"{f}-{p}" for f, p in CASES])
def test_a_negative_int_names_the_argument(task, function, param):
    mod, name = function.split(".")
    f = getattr({"search": search, "datasets": datasets, "rl": rl}[mod], name)
    with pytest.raises(ValueError, match=rf"^mymyr: {param} must be (None or )?an int (>= \d+|in \[\d+, \d+\]), got -1$"):
        f(task, **{param: -1})


def test_budgets_name_the_argument_and_its_range(task):
    for kw, bound in (("max_states", ">= 0"), ("max_expanded", ">= 0"), ("max_depth", r"in \[0, 4294967295\]")):
        with pytest.raises(ValueError, match=rf"^mymyr: {kw} must be None or an int {bound}, got -5$"):
            search.astar(task, **{kw: -5})
    with pytest.raises(ValueError, match=r"^mymyr: max_seconds must be None or a number >= 0.0, got -1.0$"):
        search.astar(task, max_seconds=-1.0)
    with pytest.raises(ValueError, match=r"^mymyr: max_seconds must be None or a number >= 0.0, got nan$"):
        search.iw(task, max_seconds=float("nan"))
    with pytest.raises(ValueError, match=r"^mymyr: max_states must be None or an int >= 0, got -1$"):
        datasets.state_space(task, max_states=-1)
    # beyond the 64-bit limit the message names it
    with pytest.raises(ValueError, match=r"^mymyr: max_states must be None or an int in \[0, 18446744073709551615\], "
                                         r"got 18446744073709551616$"):
        search.astar(task, max_states=2**64)
    with pytest.raises(ValueError, match=r"^mymyr: max_depth must be None or an int in \[0, 4294967295\], got 4294967296$"):
        search.astar(task, max_depth=2**32)


def test_non_numbers_raise_type_error_naming_the_argument(task):
    with pytest.raises(TypeError, match=r"^mymyr: max_states must be an int, not float$"):
        search.astar(task, max_states=1.5)
    with pytest.raises(TypeError, match=r"^mymyr: max_arity must be an int, not str$"):
        search.iw(task, max_arity="2")
    with pytest.raises(TypeError, match=r"^mymyr: max_seconds must be a number, not str$"):
        search.astar(task, max_seconds="1")
    with pytest.raises(TypeError, match=r"^mymyr: seeds must be a sequence of ints$"):
        search.find_rollouts_parallel(task, 3)


def test_ints_of_any_index_type_and_none_are_accepted(task):
    r = search.astar(task, max_states=np.int64(50), max_depth=None, max_seconds=np.float32(60))
    assert r.status == search.Status.OUT_OF_STATES
    assert search.astar(task, max_states=50, max_seconds=60).status == search.Status.OUT_OF_STATES
    assert search.iw(task, max_arity=True).status in (search.Status.SOLVED, search.Status.EXHAUSTED)


def test_beam_width_is_positive(task):
    with pytest.raises(ValueError, match=r"^mymyr: width must be an int in \[1, 4294967295\], got 0$"):
        search.beam(task, width=0)
    with pytest.raises(ValueError, match=r"^mymyr: standard_weight must be an int in \[1, 4294967295\], got 0$"):
        search.gbfs(task, standard_weight=0)
    assert search.beam(task, width=1).status in (search.Status.SOLVED, search.Status.EXHAUSTED)


def test_expand_capacity_is_at_most_the_row_limit(task):
    s = [task.initial_state]
    with pytest.raises(ValueError, match=r"^mymyr: capacity must be None or an int in \[0, 2147483647\], got 2305843009213693952$"):
        rl.expand(task, s, capacity=2**61)
    with pytest.raises(ValueError, match=r"^mymyr: capacity must be None or an int in \[0, 2147483647\], got 2147483648$"):
        rl.expand(task, s, capacity=2**31)
    with pytest.raises(ValueError, match=r"^mymyr: K must be None or an int in \[0, 4294967295\], got -1$"):
        rl.expand(task, s, K=-1)
    e = rl.expand(task, s, capacity=1, K=0)
    assert e.capacity == 1 and e.overflow == (len(task.applicable_actions(task.initial_state)) > 1)


def test_lists_and_constructors_name_the_argument(task):
    with pytest.raises(ValueError, match=r"^mymyr: seeds must be an int >= 0, got -2$"):
        search.find_rollouts_parallel(task, [1, -2])
    with pytest.raises(ValueError, match=r"^mymyr: threads must be an int in \[0, \d+\], got -1$"):
        rl.ThreadPool(-1)
    with pytest.raises(ValueError, match=r"^mymyr: num_envs must be an int in \[1, 4294967295\], got 0$"):
        rl.CpuEnvPool(task, 0)
    with pytest.raises(ValueError, match=r"^mymyr: max_steps must be an int in \[0, 4294967295\], got -1$"):
        rl.CpuEnvPool(task, 2, max_steps=-1)
    with pytest.raises(ValueError, match=r"^mymyr: steps must be an int >= 0, got -1$"):
        rl.random_walks(task, -1)
    with pytest.raises(ValueError, match=r"^mymyr: k must be an int in \[2, 4\], got 5$"):
        datasets.object_graph(task.initial_state).kfwl_certificate(5)
    space = datasets.state_space(text_task("gripper__prob05"))
    with pytest.raises(ValueError, match=r"^mymyr: seed must be an int >= 0, got -1$"):
        datasets.StateSpaceSampler(space, seed=-1)
    with pytest.raises(ValueError, match=r"^mymyr: n must be an int in \[0, 2147483647\], got -1$"):
        datasets.StateSpaceSampler(space).states_n_steps_from_goal(-1)
    with pytest.raises(ValueError, match=r"^mymyr: k must be an int in \[0, 4294967295\], got -1$"):
        datasets.KnowledgeBase([task], k=-1)
    with pytest.raises(ValueError, match=r"^mymyr: fc_free_params must be an int in \[0, 4294967295\], got -1$"):
        mymyr.Task.from_text(str(TASKS / "blocks__probBLOCKS-8-0.txt"), fc_free_params=-1)


def test_atom_text_needs_balanced_parentheses(blocks):
    with pytest.raises(ValueError, match=r"^mymyr: unbalanced parentheses in '\(on a b'$"):
        blocks.ground_condition(["(on a b"])
    with pytest.raises(ValueError, match=r"unbalanced parentheses in 'on a b\)'"):
        blocks.ground_condition(["on a b)"])
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        blocks.ground_condition(["(not (on a b)"])
    assert str(blocks.ground_condition(["(on a b)"])) == str(blocks.ground_condition(["on a b"]))


@pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")
def test_a_swapped_domain_and_problem_name_the_file_kinds(tmp_path):
    d, p = BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl"
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL problem, where a domain belongs") as info:
        mymyr.Task.from_pddl(p, d)
    assert info.value.path == str(p)
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL problem, where a domain belongs"):
        mymyr.Domain.from_file(p)
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL problem, where a domain belongs"):
        mymyr.Domain.from_string(p.read_text())
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL domain, where a problem belongs") as info:
        mymyr.Domain.from_file(d).instantiate(d)
    assert info.value.path == str(d)
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL domain, where a problem belongs"):
        mymyr.Domain.from_file(d).instantiate_string(d.read_text())
    with pytest.raises(mymyr.PddlError, match=r"this is a PDDL problem, where a domain belongs"):
        rl.TaskTable.from_pddl(p, [d])
    # comments and blank lines before the header
    commented = tmp_path / "domain.pddl"
    commented.write_text("; blocks\n\n  ;; four operators\n" + d.read_text())
    assert mymyr.Task.from_pddl(commented, p).num_schemas == mymyr.Task.from_pddl(d, p).num_schemas


def test_the_cuda_module_names_the_build_option():
    try:
        import mymyr.cuda  # noqa: F401
    except ImportError as e:
        assert 'CMAKE_ARGS="-DMYMYR_CUDA=ON' in str(e)
    else:
        assert 'CMAKE_ARGS="-DMYMYR_CUDA=ON"' in mymyr.cuda.__doc__
