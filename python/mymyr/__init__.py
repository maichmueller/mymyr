"""mymyr: a data-oriented, lifted-first planning core with free-threaded Python bindings.

    task = mymyr.Task.from_pddl("domain.pddl", "p01.pddl")   # immutable, shared by threads
    s = task.initial_state                                     # a value: hashable, picklable, thread-safe
    for a in task.applicable_actions(s):                       # Action = label (schema, binding), canonical order
        t = task.apply(s, a)
    for a in task.bindings("stack", s, partial={"x": "a"}):  # one schema's actions with parameters fixed
        ...
    s2 = task.state(["(on a b)", "(clear a)"])                 # a state from its atoms (values= for numeric tasks)
    h = task.local()                                           # per-thread handle for hot Python loops
    exp = mymyr.rl.expand(task, states)                        # batched: flat CSR + padded view, zero-copy arrays
    r = mymyr.search.iw(task, max_arity=2)                     # searches: iw, siw, brfs, astar, gbfs, beam
    space = mymyr.datasets.state_space(task)                   # state spaces, samplers, object graphs
"""

import os as _os

from mymyr import _core, formalism
from mymyr._core import (
    Action,
    ApplicableActions,
    Atom,
    Bindings,
    ConjunctiveCondition,
    DLArray,
    GroundConjunctions,
    GroundLiteral,
    State,
    Task,
    TaskHandle,
    __version__,
    build_info,
    free_threaded_build,
    hash_rows,
)
from mymyr import datasets, rl, search

__all__ = [
    "Action",
    "ApplicableActions",
    "Atom",
    "Bindings",
    "ConjunctiveCondition",
    "DLArray",
    "GroundConjunctions",
    "GroundLiteral",
    "State",
    "Task",
    "TaskHandle",
    "__version__",
    "build_info",
    "datasets",
    "formalism",
    "free_threaded_build",
    "get_cmake_dir",
    "get_include",
    "hash_rows",
    "rl",
    "search",
]

if hasattr(_core, "Domain"):  # built with the loki front end (MYMYR_FRONTEND, the default)
    from mymyr._core import Domain

    __all__.append("Domain")


_PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))


def get_include() -> str:
    """The include directory of the C++ headers (``mymyr/*.hpp``, and ``mymyr/ext.h`` of the plain-C API).

    The C++ package is a source-level interface: consumers compile with C++26 (GCC 16, or clang 22 with libc++) and
    the C++ standard library the wheel was built with (libstdc++ on Linux, libc++ on macOS), and are rebuilt when
    mymyr changes. The plain-C API in ``mymyr/ext.h`` is the stable binary interface. The directory exists in an
    installed wheel, not in an editable or in-tree install.
    """
    return _os.path.join(_PACKAGE_DIR, "include")


def get_cmake_dir() -> str:
    """The directory of ``mymyrConfig.cmake``: pass it as ``-Dmymyr_DIR=...`` to CMake and use
    ``find_package(mymyr CONFIG REQUIRED COMPONENTS core frontend)`` for the targets ``mymyr::core`` and
    ``mymyr::frontend`` (and ``mymyr::cuda`` in a CUDA wheel). The libraries are static and carry their
    dependencies, so nothing else needs to be installed.

    Like :func:`get_include`, this is a source-level interface that needs C++26; it exists in an installed wheel only.
    """
    return _os.path.join(_PACKAGE_DIR, "lib", "cmake", "mymyr")
