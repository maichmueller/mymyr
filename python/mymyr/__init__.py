"""mymyr: a data-oriented, lifted-first planning core with free-threaded Python bindings.

    task = mymyr.Task.from_pddl("domain.pddl", "p01.pddl")   # immutable, shared by threads
    s = task.initial_state                                     # a value: hashable, picklable, thread-safe
    for a in task.applicable_actions(s):                       # Action = label (schema, binding), canonical order
        t = task.apply(s, a)
    s2 = task.state(["(on a b)", "(clear a)"])                 # a state from its atoms (values= for numeric tasks)
    h = task.local()                                           # per-thread handle for hot Python loops
    exp = mymyr.rl.expand(task, states)                        # batched: flat CSR + padded view, zero-copy arrays
    r = mymyr.search.iw(task, max_arity=2)                     # searches: iw, siw, brfs, astar, gbfs, beam
    space = mymyr.datasets.state_space(task)                   # state spaces, samplers, object graphs
"""

from mymyr import _core, formalism
from mymyr._core import (
    Action,
    ApplicableActions,
    Atom,
    DLArray,
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
    "DLArray",
    "State",
    "Task",
    "TaskHandle",
    "__version__",
    "build_info",
    "datasets",
    "formalism",
    "free_threaded_build",
    "hash_rows",
    "rl",
    "search",
]

if hasattr(_core, "Domain"):  # built with the loki front end (MYMYR_FRONTEND, the default)
    from mymyr._core import Domain

    __all__.append("Domain")
