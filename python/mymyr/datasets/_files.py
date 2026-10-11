"""Dataset files: mymyr.datasets.save and load.

A file is a zip archive that numpy.load also opens (an .npz file): the member ``header.json`` describes the objects,
every array is an ``.npy`` member, and the tasks' PDDL (or normalized text) are text members. Nothing in a file is a
pickle, so loading an untrusted file reads data only: JSON, text and arrays read with ``allow_pickle=False``, then
validated by the constructors of the objects (StateSpace.from_arrays and the like).

Header (format "mymyr.datasets", version 1)::

    {"format": "mymyr.datasets", "version": 1, "mymyr": "<version that wrote it>",
     "kind": "state_space" | "generalized_state_space" | "knowledge_base" | "tuple_graph" | "tuple_graphs",
     "tasks": [{"source": "pddl", "domain": <member>, "problem": <member>, "domain_path": str, "problem_path": str,
                "options": {<Task keyword arguments>}, "fingerprint": <16 hex digits>,
                "encoding_fingerprint": <16 hex digits>, "atom_slots": <member> | null}
               | {"source": "text", "text": <member>, ...the same...}],
     "state_spaces": [{"task": <index into tasks>, "arrays": <member prefix>,
                       "statistics": {"threads": int, "layers": int, "search_seconds": float, "post_seconds": float}}],
     "tuple_graphs": [{"state_space": <index into state_spaces>, "arrays": <member prefix>}],
     "knowledge_base": {"options": {<KnowledgeBase keyword arguments>}, "task_indices": [int],
                        "tuple_graphs": [<index into tuple_graphs> per state space] | null}}

An object's arrays are the members ``<prefix><name>.npy``: a state space's are those of StateSpace.arrays() with
state_words always present; a group of tuple graphs (of one state space) is packed into ``meta`` [G, 3] (root, width,
dominance_pruning), ``sizes`` [G, 11] (each graph's array lengths) and the eleven arrays of TupleGraph.arrays(), each
concatenated over the graphs. ``atom_slots`` is the lazy slot table of a task with lazy atom slots (uint64: the numbers
of fluent and derived slots, then their canonical ids in slot order).

Compatibility: a reader takes every version up to its own, refuses a newer one and refuses a kind it does not know.
Sections and fields a reader does not know are ignored, so objects that later versions attach to a dataset (for
example the encoding schemas and encoders of mymyr.encoding) are added as new sections or kinds without changing the
version of the sections above.
"""

from __future__ import annotations

import json
import os
import secrets
import zipfile
from collections.abc import Sequence
from typing import Any, BinaryIO

import numpy as np

from mymyr import _core
from mymyr._core import Task, TaskHandle
from mymyr._core._datasets import (
    GeneralizedStateSpace,
    KnowledgeBase,
    StateSpace,
    TupleGraph,
    _knowledge_base_state,
    _pack_tuple_graphs,
    _restore_knowledge_base,
    _restore_state_space,
    _state_space_state,
    _unpack_tuple_graphs,
)
from mymyr._core._rl import TaskTable

FORMAT = "mymyr.datasets"
VERSION = 1
HEADER = "header.json"
KINDS = ("state_space", "generalized_state_space", "knowledge_base", "tuple_graph", "tuple_graphs")

# the Task keyword arguments, in the order of a task pickle's options; the KnowledgeBase ones after its tasks
_TASK_OPTIONS = ("atoms", "matching", "fc_free_params", "numeric_storage", "numeric_quantum", "numeric_tolerant",
                 "reachability_budget")
_KB_OPTIONS = ("threads", "max_states", "max_seconds", "remove_if_unsolvable", "labels", "sort_by_size", "generalized",
               "width", "dominance_pruning")

Dataset = StateSpace | GeneralizedStateSpace | KnowledgeBase | TupleGraph | list[TupleGraph]


def _hex(x: int) -> str:
    return f"{x:016x}"


def _unhex(s: Any, what: str) -> int:
    if not isinstance(s, str) or len(s) != 16:
        raise ValueError(f"mymyr: corrupt dataset file ({what} is not 16 hex digits)")
    return int(s, 16)


class _Writer:
    """The members of a file being written: the header's tasks and state spaces, deduplicated by object."""

    def __init__(self, zf: zipfile.ZipFile):
        self.zf = zf
        self.tasks: list[dict[str, Any]] = []
        self.task_ids: dict[int, int] = {}  # id(Task) -> index
        self.domains: dict[str, str] = {}  # domain text -> member
        self.spaces: list[dict[str, Any]] = []
        self.space_ids: dict[int, int] = {}  # id(StateSpace) -> index
        self.tuple_graphs: list[dict[str, Any]] = []
        self.keep: list[object] = []  # keeps the objects behind the ids alive while writing

    def text(self, name: str, text: str) -> str:
        self.zf.writestr(name, text.encode("utf-8"))
        return name

    def array(self, name: str, a: Any) -> str:
        with self.zf.open(name, "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asarray(a), allow_pickle=False)
        return name

    def task(self, task: Task) -> int:
        if id(task) in self.task_ids:
            return self.task_ids[id(task)]
        self.keep.append(task)
        _, (kind, a, b, a_path, b_path, options, fingerprint, (encoding_fingerprint, lazy)) = task.__reduce__()
        t = len(self.tasks)
        entry: dict[str, Any] = {"source": kind}
        if kind == "pddl":
            if a not in self.domains:
                self.domains[a] = self.text(f"domains/{len(self.domains)}.pddl", a)
            entry |= {"domain": self.domains[a], "problem": self.text(f"tasks/{t}/problem.pddl", b),
                      "domain_path": a_path, "problem_path": b_path}
        else:
            entry["text"] = self.text(f"tasks/{t}/task.txt", a)
        entry["options"] = dict(zip(_TASK_OPTIONS, options, strict=True))
        entry["fingerprint"] = _hex(fingerprint)
        entry["encoding_fingerprint"] = _hex(encoding_fingerprint)
        entry["atom_slots"] = self.array(f"tasks/{t}/atom_slots.npy", np.frombuffer(lazy, dtype="<u8")) if lazy else None
        self.tasks.append(entry)
        self.task_ids[id(task)] = t
        return t

    def space(self, space: StateSpace) -> int:
        if not isinstance(space, StateSpace):
            raise TypeError(f"mymyr: expected a mymyr.datasets.StateSpace, got {type(space).__name__}")
        if id(space) in self.space_ids:
            return self.space_ids[id(space)]
        self.keep.append(space)
        task, _, _, _, (threads, layers, search_seconds, post_seconds), arrays = _state_space_state(space)
        i = len(self.spaces)
        prefix = f"state_spaces/{i}/"
        for name, a in arrays.items():
            self.array(f"{prefix}{name}.npy", a)
        self.spaces.append({"task": self.task(task), "arrays": prefix,
                            "statistics": {"threads": threads, "layers": layers, "search_seconds": search_seconds,
                                           "post_seconds": post_seconds}})
        self.space_ids[id(space)] = i
        return i

    def graphs(self, graphs: Sequence[TupleGraph]) -> int:
        """A group of tuple graphs of one state space."""
        for g in graphs:
            if not isinstance(g, TupleGraph):
                raise TypeError(f"mymyr: expected mymyr.datasets.TupleGraphs, got {type(g).__name__}")
        if not graphs:
            raise ValueError("mymyr: no tuple graphs to save")
        packed = _pack_tuple_graphs(graphs)
        return self.packed(graphs[0].space, packed)

    def packed(self, space: StateSpace, packed: dict[str, Any]) -> int:
        j = len(self.tuple_graphs)
        prefix = f"tuple_graphs/{j}/"
        for name, a in packed.items():
            self.array(f"{prefix}{name}.npy", a)
        self.tuple_graphs.append({"state_space": self.space(space), "arrays": prefix})
        return j


def _write(zf: zipfile.ZipFile, obj: Dataset) -> None:
    w = _Writer(zf)
    header: dict[str, Any] = {"format": FORMAT, "version": VERSION, "mymyr": _core.__version__}
    if isinstance(obj, StateSpace):
        header["kind"] = "state_space"
        w.space(obj)
    elif isinstance(obj, GeneralizedStateSpace):
        header["kind"] = "generalized_state_space"
        for s in obj.spaces:
            w.space(s)
    elif isinstance(obj, KnowledgeBase):
        header["kind"] = "knowledge_base"
        table, options, spaces, task_indices, packs = _knowledge_base_state(obj)
        for t in table.tasks:
            w.task(t if isinstance(t, Task) else t.task)
        for s in spaces:
            w.space(s)
        groups = None if packs is None else [w.packed(s, p) for s, p in zip(spaces, packs, strict=True)]
        header["knowledge_base"] = {"options": dict(zip(_KB_OPTIONS, options, strict=True)),
                                    "task_indices": list(task_indices), "tuple_graphs": groups}
    elif isinstance(obj, TupleGraph):
        header["kind"] = "tuple_graph"
        w.graphs([obj])
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)) and obj and all(
            isinstance(g, TupleGraph) for g in obj):
        header["kind"] = "tuple_graphs"
        w.graphs(list(obj))
    else:
        raise TypeError("mymyr: datasets.save takes a StateSpace, GeneralizedStateSpace, KnowledgeBase, TupleGraph or "
                        f"a nonempty list of TupleGraphs of one state space, not {type(obj).__name__} (a "
                        "mymyr.cuda.DeviceStateSpace saves as its to_host())")
    header["tasks"] = w.tasks
    header["state_spaces"] = w.spaces
    header["tuple_graphs"] = w.tuple_graphs
    zf.writestr(HEADER, json.dumps(header, indent=1).encode("utf-8"))


def save(obj: Dataset, path: str | os.PathLike[str] | BinaryIO, *, compress: bool = False) -> None:
    """Saves a dataset to a file (a path, or a binary file object open for writing): a StateSpace, a
    GeneralizedStateSpace, a KnowledgeBase, a TupleGraph or a list of TupleGraphs of one state space, together with
    their tasks (their PDDL, or the normalized text of a task made from one) and the tasks' fingerprints.

    The file is a zip archive of a JSON header and NumPy .npy arrays (format "mymyr.datasets", version 1), which
    numpy.load also opens; it contains no pickle. compress: deflate the members (smaller, slower). A path is written
    to a new file next to it first and then renamed, so an interrupted save leaves no partial file. Read it back with
    mymyr.datasets.load.
    """
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    if hasattr(path, "write"):
        with zipfile.ZipFile(path, "w", compression=method, allowZip64=True) as zf:
            _write(zf, obj)
        return
    target = os.fspath(path)
    # a new file next to the target (permissions from the umask, as open() gives them), renamed when complete
    head, tail = os.path.split(os.path.abspath(target))
    tmp = os.path.join(head, f".{tail}.{secrets.token_hex(8)}.tmp")
    try:
        with open(tmp, "xb") as f, zipfile.ZipFile(f, "w", compression=method, allowZip64=True) as zf:
            _write(zf, obj)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class _Reader:
    def __init__(self, zf: zipfile.ZipFile, name: str):
        self.zf = zf
        self.name = name
        try:
            header = json.loads(zf.read(HEADER).decode("utf-8"))
        except KeyError:
            raise ValueError(f"mymyr: {name} is not a mymyr dataset file (no {HEADER})") from None
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:
            raise ValueError(f"mymyr: corrupt dataset file {name} ({HEADER}: {e})") from None
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            raise ValueError(f"mymyr: {name} is not a mymyr dataset file (format is not {FORMAT!r})")
        version = header.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"mymyr: corrupt dataset file {name} (version {version!r})")
        if version > VERSION:
            raise ValueError(f"mymyr: {name} has dataset format version {version}, written by mymyr "
                             f"{header.get('mymyr')}; this mymyr ({_core.__version__}) reads versions up to {VERSION}")
        if header.get("kind") not in KINDS:
            raise ValueError(f"mymyr: {name} holds a {header.get('kind')!r}, which this mymyr ({_core.__version__}) "
                             "cannot load")
        self.header = header

    def text(self, member: Any) -> str:
        if not isinstance(member, str):
            raise ValueError(f"mymyr: corrupt dataset file {self.name} (a member name is not a string)")
        try:
            return self.zf.read(member).decode("utf-8")
        except KeyError:
            raise ValueError(f"mymyr: corrupt dataset file {self.name} (no member {member})") from None
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:
            raise ValueError(f"mymyr: corrupt dataset file {self.name} ({member}: {e})") from None

    def array(self, member: str) -> np.ndarray:
        try:
            with self.zf.open(member) as f:
                a = np.lib.format.read_array(f, allow_pickle=False)
        except KeyError:
            raise ValueError(f"mymyr: corrupt dataset file {self.name} (no member {member})") from None
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:  # not an .npy, or an object array
            raise ValueError(f"mymyr: corrupt dataset file {self.name} ({member}: {e})") from None
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return a

    def arrays(self, prefix: Any) -> dict[str, np.ndarray]:
        if not isinstance(prefix, str):
            raise ValueError(f"mymyr: corrupt dataset file {self.name} (an array prefix is not a string)")
        names = [m for m in self.zf.namelist() if m.startswith(prefix) and m.endswith(".npy") and "/" not in m[len(prefix):]]
        return {m[len(prefix):-4]: self.array(m) for m in names}

    def slots(self, entry: dict[str, Any]) -> np.ndarray | None:
        member = entry.get("atom_slots")
        return None if member is None else self.array(member)

    def task(self, entry: dict[str, Any]) -> Task:
        """The task of a header entry, built from its source and checked against its fingerprints."""
        options = tuple(entry["options"][k] for k in _TASK_OPTIONS)
        lazy = self.slots(entry)
        encoding = (_unhex(entry["encoding_fingerprint"], "an encoding fingerprint"),
                    b"" if lazy is None else np.ascontiguousarray(lazy, dtype="<u8").tobytes())
        fingerprint = _unhex(entry["fingerprint"], "a fingerprint")
        if entry["source"] == "pddl":
            return _core._restore_task("pddl", self.text(entry["domain"]), self.text(entry["problem"]),
                                       entry.get("domain_path", ""), entry.get("problem_path", ""), options, fingerprint,
                                       encoding)
        if entry["source"] == "text":
            return _core._restore_task("text", self.text(entry["text"]), "", "", "", options, fingerprint, encoding)
        raise ValueError(f"mymyr: corrupt dataset file {self.name} (task source {entry['source']!r})")


def _describe(entry: dict[str, Any]) -> str:
    if entry.get("source") == "pddl":
        return f"{entry.get('domain_path') or '<domain>'} / {entry.get('problem_path') or '<problem>'}"
    return "a task in the normalized text format"


def _given_tasks(r: _Reader, given: Sequence[Task | TaskHandle] | TaskTable) -> list[Task | TaskHandle]:
    entries = r.header["tasks"]
    tasks = list(given.tasks) if isinstance(given, TaskTable) else list(given)
    if len(tasks) != len(entries):
        raise ValueError(f"mymyr: {r.name} has {len(entries)} task(s); {len(tasks)} were given")
    for i, (t, e) in enumerate(zip(tasks, entries, strict=True)):
        if not isinstance(t, (Task, TaskHandle)):
            raise TypeError(f"mymyr: tasks must be Tasks or TaskHandles, not {type(t).__name__}")
        if t.fingerprint != _unhex(e["fingerprint"], "a fingerprint"):
            raise ValueError(f"mymyr: task {i} given for {r.name} is another task than the file's ({_describe(e)}): "
                             "content fingerprint mismatch")
    return tasks


def _read(r: _Reader, tasks: Sequence[Task | TaskHandle] | TaskTable | None) -> Dataset:
    h = r.header
    entries = h["tasks"]
    ts = [r.task(e) for e in entries] if tasks is None else _given_tasks(r, tasks)

    def space(i: int) -> StateSpace:
        s = h["state_spaces"][i]
        t = s["task"]
        st = s["statistics"]
        stats = (st["threads"], st["layers"], st["search_seconds"], st["post_seconds"])
        e = entries[t]
        return _restore_state_space(ts[t], _unhex(e["fingerprint"], "a fingerprint"),
                                    _unhex(e["encoding_fingerprint"], "an encoding fingerprint"), r.slots(e), stats,
                                    r.arrays(s["arrays"]))

    kind = h["kind"]
    if kind == "state_space":
        return space(0)
    if kind == "generalized_state_space":
        return GeneralizedStateSpace([space(i) for i in range(len(h["state_spaces"]))])
    if kind in ("tuple_graph", "tuple_graphs"):
        g = h["tuple_graphs"][0]
        graphs = _unpack_tuple_graphs(space(g["state_space"]), r.arrays(g["arrays"]))
        return graphs[0] if kind == "tuple_graph" else graphs
    # knowledge_base
    kb = h["knowledge_base"]
    table = tasks if isinstance(tasks, TaskTable) else TaskTable(ts)
    spaces = [space(i) for i in range(len(h["state_spaces"]))]
    groups = kb["tuple_graphs"]
    packs = None if groups is None else [r.arrays(h["tuple_graphs"][j]["arrays"]) for j in groups]
    options = tuple(kb["options"][k] for k in _KB_OPTIONS)
    return _restore_knowledge_base(table, options, spaces, kb["task_indices"], packs)


def load(path: str | os.PathLike[str] | BinaryIO, *, task: Task | TaskHandle | None = None,
         tasks: Sequence[Task | TaskHandle] | TaskTable | None = None) -> Dataset:
    """Loads a dataset saved by mymyr.datasets.save: the StateSpace, GeneralizedStateSpace, KnowledgeBase, TupleGraph
    or list of TupleGraphs it holds.

    Without task / tasks, the tasks are built from the file (its PDDL needs the PDDL front end, mymyr.Domain) and
    checked against the saved fingerprints. task= (a file of one task) or tasks= (the file's tasks in order: a state
    space's, a generalized state space's in the order of its spaces, a knowledge base's TaskTable) takes existing tasks
    instead, so that the states of the dataset are states of those Task objects; each must be the file's task (content
    fingerprint) with the same state encoding (encoding fingerprint), else ValueError. A task with lazy atom slots
    takes the saved numbering: its slots must agree with it so far (ValueError otherwise).

    Every array is validated (as StateSpace.from_arrays and TupleGraph.from_arrays do) and the file holds no pickle,
    so loading reads data only. ValueError for a file that is not a mymyr dataset, a newer format version or an
    unknown kind.
    """
    if task is not None:
        if tasks is not None:
            raise TypeError("mymyr: pass task= or tasks=, not both")
        tasks = [task]
    if tasks is not None and not isinstance(tasks, (TaskTable, Sequence)):
        raise TypeError("mymyr: tasks must be a sequence of Tasks or a TaskTable")
    name = repr(os.fspath(path)) if not hasattr(path, "read") else "the file"
    try:
        zf = zipfile.ZipFile(path if hasattr(path, "read") else os.fspath(path), "r")
    except zipfile.BadZipFile:
        raise ValueError(f"mymyr: {name} is not a mymyr dataset file (not a zip archive)") from None
    with zf:
        r = _Reader(zf, name)
        try:
            return _read(r, tasks)
        except (KeyError, TypeError, IndexError) as e:
            if isinstance(e, TypeError) and str(e).startswith("mymyr:"):
                raise
            raise ValueError(f"mymyr: corrupt dataset file {name} ({type(e).__name__}: {e})") from None

