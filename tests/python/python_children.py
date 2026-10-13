"""Run example programs with the interpreter and import paths of the module under test."""

import os
import pathlib
import subprocess
import sys

import mymyr
import mymyr._core


def requires_frontend(source):
    return ".from_pddl(" in source or "mymyr.Domain" in source


def module_bootstrap():
    paths = [os.path.abspath(path) for path in sys.path]
    expected = [os.path.realpath(mymyr.__file__), os.path.realpath(mymyr._core.__file__)]
    package_paths = list(dict.fromkeys([*mymyr.__path__, str(pathlib.Path(mymyr._core.__file__).parent)]))
    # Explicitly restore the path after startup: -I and -E ignore PYTHONPATH, and editable site hooks can otherwise
    # select a different build. Check both the Python package and the extension before executing the program.
    return (
        "import importlib.util, os, runpy, sys\n"
        f"sys.path[:] = {paths!r}\n"
        f"spec = importlib.util.spec_from_file_location('mymyr', {mymyr.__file__!r}, "
        f"submodule_search_locations={package_paths!r})\n"
        "package = importlib.util.module_from_spec(spec)\n"
        "sys.modules['mymyr'] = package\n"
        f"core_spec = importlib.util.spec_from_file_location('mymyr._core', {mymyr._core.__file__!r})\n"
        "core = importlib.util.module_from_spec(core_spec)\n"
        "sys.modules['mymyr._core'] = core\n"
        "core_spec.loader.exec_module(core)\n"
        "package._core = core\n"
        "spec.loader.exec_module(package)\n"
        "import mymyr, mymyr._core\n"
        f"assert [os.path.realpath(mymyr.__file__), os.path.realpath(mymyr._core.__file__)] == {expected!r}\n"
    )


def run_python(script, *, cwd, timeout, capture_output=False):
    bootstrap = module_bootstrap() + (
        f"sys.argv = [{str(pathlib.Path(script).resolve())!r}]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    return subprocess.run(
        [sys.executable, *subprocess._args_from_interpreter_flags(), "-W", "error", "-c", bootstrap],
        cwd=cwd, check=True, timeout=timeout, capture_output=capture_output, text=True,
    )
