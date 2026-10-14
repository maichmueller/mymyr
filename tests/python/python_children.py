"""Run child Python programs on the module under test.

A child gets the parent's interpreter flags and ``sys.path`` and checks that it imports the same ``mymyr`` package and
``mymyr._core`` extension files as the parent, so a test exercises the tested build and not whichever one the
environment's site processing (an editable install, ``PYTHONPATH`` under ``-I`` / ``-E``) would select.
"""

import os
import subprocess
import sys

import mymyr
import mymyr._core


def requires_frontend(source):
    return ".from_pddl(" in source or "mymyr.Domain" in source


def _prelude():
    expected = [os.path.realpath(mymyr.__file__), os.path.realpath(mymyr._core.__file__)]
    return (
        f"import os, runpy, sys; sys.path[:] = {[os.path.abspath(path) for path in sys.path]!r}; "
        f"import mymyr, mymyr._core; found = [os.path.realpath(mymyr.__file__), os.path.realpath(mymyr._core.__file__)]; "
        f"assert found == {expected!r}, found\n"
    )


def _run(code, args, *, cwd, env, timeout, capture_output, check):
    return subprocess.run(
        [sys.executable, *subprocess._args_from_interpreter_flags(), "-W", "error", "-c", _prelude() + code, *args],
        cwd=cwd, env=env, check=check, timeout=timeout, capture_output=capture_output, text=True,
    )  # fmt: skip


def run_code(code, *args, cwd=None, env=None, timeout=120, capture_output=False, check=True):
    """Runs the source ``code`` as ``python -c code *args`` (``sys.argv == ['-c', *args]``); its first line is the
    prelude, so line numbers in a traceback are those of ``code`` plus one."""
    return _run(code, args, cwd=cwd, env=env, timeout=timeout, capture_output=capture_output, check=check)


def run_python(script, *args, cwd=None, env=None, timeout=120, capture_output=False, check=True):
    """Runs the file ``script`` as ``python script *args``."""
    code = f"sys.argv = [{os.path.abspath(script)!r}, *sys.argv[1:]]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
    return _run(code, args, cwd=cwd, env=env, timeout=timeout, capture_output=capture_output, check=check)
