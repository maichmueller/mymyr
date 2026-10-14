"""The parts the single and the vector Gymnasium environment share: problems, instance draws, spaces and the
observation builders."""

import copy
from typing import Any, NamedTuple, Optional, Union

import gymnasium
import numpy as np
from gymnasium import spaces

from mymyr._core import Action, State, Task, TaskHandle
from mymyr._core._formalism import GroundAtom, GroundCondition
from mymyr._core._rl import TaskSuite, TaskTable, successor_bounds
from mymyr.rl import decode
from mymyr.rl._sampling import InstanceSampler, _instance_sampler

OBSERVATION_MODES = ("symbolic", "graph", "bits")
ACTION_MODES = ("index", "ground")
_INT32_MAX = np.iinfo(np.int32).max


class Problems:
    """The instances of an environment: a Task, TaskHandle, TaskTable or TaskSuite, or a PDDL domain and problems
    (``TaskTable.from_pddl``), with the instances' Tasks in table order."""

    def __init__(self, table: Any, domain: Any, problems: Any, task_options: Optional[dict]) -> None:
        if table is None:
            if domain is None or problems is None:
                raise ValueError("mymyr: pass a task, a TaskTable or a TaskSuite, or both domain= and problems=")
            table = TaskTable.from_pddl(str(domain), _problem_arg(problems), **(task_options or {}))
        elif domain is not None or problems is not None or task_options:
            raise ValueError("mymyr: pass either a task, table or suite, or domain= and problems= (with task_options=)")
        if isinstance(table, TaskHandle):
            table = table.task
        if isinstance(table, Task):
            self.tasks: list[Task] = [table]
            self.words = int(table.words)
            self.numeric_words = int(table.numeric_words)
        elif isinstance(table, (TaskTable, TaskSuite)):
            self.tasks = list(table.tasks)
            self.words = int(table.words)
            self.numeric_words = int(table.numeric_words)
        else:
            raise TypeError(f"mymyr: expected a Task, a TaskTable or a TaskSuite, not {type(table).__name__}")
        self.table = table
        self.uids = tuple(int(task.uid) for task in self.tasks)
        self.numeric = self.numeric_words > 0

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def row_words(self) -> int:
        return self.words + self.numeric_words

    def check_task_id(self, value: Any, name: str = "task_id") -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"mymyr: {name} must be an int, not {type(value).__name__}")
        if not 0 <= int(value) < len(self.tasks):
            raise ValueError(f"mymyr: {name} {int(value)} is outside the {len(self.tasks)} instances")
        return int(value)

    def domain_table(self) -> Union[Task, TaskTable]:
        """The table of the one domain (the encoder's input)."""
        if isinstance(self.table, TaskSuite):
            if self.table.num_domains != 1:
                raise ValueError("mymyr: graph observations need instances of one domain (a TaskTable), not a suite of "
                                 f"{self.table.num_domains} domains")
            return self.table.tables[0]
        return self.table

    def max_actions(self) -> int:
        """The default action count of the index action mode: the largest successor bound of an instance."""
        return max(1, max(successor_bounds(self.table)))

    def decode(self, rows: np.ndarray, task_ids: np.ndarray) -> list[State]:
        """The States of rows [N, row_words] of instances task_ids [N]."""
        if isinstance(self.table, Task):
            return self.table.decode(rows) if len(rows) else []
        return decode(self.table, rows, task_ids)


def _problem_arg(problems: Any) -> Any:
    if isinstance(problems, (str, bytes)) or hasattr(problems, "__fspath__"):
        return str(problems)
    return [str(p) for p in problems]


class Draws:
    """The instance of each env's episodes: episode k of env e (k = 0, 1, ... from its last seeding) starts in the
    sampler's draw for (seed, env id, k), the draw of a CpuEnvPool or torch BatchedEnv env with that seed and env id
    whose resets and autoresets are counted the same way; or in the fixed problem index; or, without a sampler, in
    its previous instance (instance 0 at first)."""

    def __init__(self, problems: Problems, sampler: Any, env_ids: np.ndarray) -> None:
        self.sampler: Optional[InstanceSampler] = _instance_sampler(sampler, len(problems), True)
        self.problems = problems
        self.env_ids = np.asarray(env_ids, np.uint64)
        n = len(self.env_ids)
        self.seeds = np.zeros(n, np.uint64)
        self.counters = np.zeros(n, np.uint64)
        self.current = np.zeros(n, np.int32)
        self.seeded = False
        self.fixed: Optional[int] = None

    def seed(self, envs: np.ndarray, seeds: Any) -> None:
        """Restarts the draws of envs at episode 0 of their seeds (one int or one per env)."""
        values = np.asarray(seeds, dtype=object)
        values = np.broadcast_to(values, envs.shape)
        self.seeds[envs] = np.array([int(v) & 0xFFFFFFFFFFFFFFFF for v in values.ravel()], np.uint64).reshape(envs.shape)
        self.counters[envs] = 0
        self.seeded = True

    def peek(self, envs: np.ndarray) -> np.ndarray:
        """The instances the next episodes of envs start in (without counting them)."""
        if self.fixed is not None:
            return np.full(len(envs), self.fixed, np.int32)
        if self.sampler is None:
            return self.current[envs].copy()
        seeds = self.seeds[envs]
        halves = np.stack([seeds & np.uint64(0xFFFFFFFF), seeds >> np.uint64(32)], axis=-1)
        return np.asarray(self.sampler.sample(halves, self.env_ids[envs], self.counters[envs]), np.int32)

    def start(self, envs: np.ndarray, ids: np.ndarray) -> None:
        """Counts the episodes of envs that start in instances ids."""
        self.counters[envs] += np.uint64(1)
        self.current[envs] = ids


def check_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"mymyr: a seed must be an int, not {type(seed).__name__}")
    if int(seed) < 0:
        raise ValueError("mymyr: a seed must be nonnegative")
    return int(seed)


# ------------------------------------------------------------------------------------------------ symbolic observations


class SymbolicObservation(NamedTuple):
    """A state as ground atoms (immutable, hashable values; the fields not requested are None).

    - ``atoms``: frozenset of the true fluent GroundAtoms (and, with ``derived=True``, the derived ones);
    - ``objects``: tuple of the instance's object names in object order (``objects=True``; ``task.objects``, their
      types in ``task.formalism.objects``);
    - ``goal``: the instance's goal as a GroundCondition (``goal=True``);
    - ``static``: frozenset of the instance's static GroundAtoms, the types included (``static=True``);
    - ``numeric``: tuple of the numeric fluents' values in ``task.numeric_names`` order (``numeric=True``).

    Atoms compare equal when they are atoms of the same Task object. An observation pickles with its Task (a copy
    in the receiving process) and is its own (deep) copy.
    """

    atoms: frozenset
    objects: Optional[tuple] = None
    goal: Optional[GroundCondition] = None
    static: Optional[frozenset] = None
    numeric: Optional[tuple] = None

    def __copy__(self) -> "SymbolicObservation":
        return self

    def __deepcopy__(self, memo: dict) -> "SymbolicObservation":
        return self


class SymbolicObserver:
    """Builds SymbolicObservations of States; the per-instance parts (objects, goal, static atoms) once."""

    def __init__(self, problems: Problems, *, objects: bool, goal: bool, static: bool, derived: bool,
                 numeric: Optional[bool]) -> None:
        self.problems = problems
        self.objects, self.goal, self.static, self.derived = bool(objects), bool(goal), bool(static), bool(derived)
        self.numeric = problems.numeric if numeric is None else bool(numeric)
        self._parts: dict[int, tuple] = {}

    def parts(self, t: int) -> tuple:
        parts = self._parts.get(t)
        if parts is None:
            task = self.problems.tasks[t]
            parts = (
                tuple(task.objects) if self.objects else None,
                task.goal_condition if self.goal else None,
                # the Task's own atoms (equal to the formalism's), which pickle with the Task
                frozenset(task.atom(a) for a in task.formalism.static_init) if self.static else None,
            )
            self._parts[t] = parts
        return parts

    def observe(self, state: State, t: int) -> SymbolicObservation:
        atoms = state.atoms()
        if self.derived:
            atoms += state.derived_atoms()
        objects, goal, static = self.parts(t)
        numeric = tuple(state.numeric_values()) if self.numeric else None
        return SymbolicObservation(frozenset(atoms), objects, goal, static, numeric)


class SymbolicSpace(gymnasium.Space):
    """The SymbolicObservations of an environment's instances: a custom space, since its elements are sets of
    mymyr GroundAtoms (no standard space holds arbitrary hashable values, and the atoms of an instance are known only
    as states reach them). ``contains`` checks the structure: a SymbolicObservation with exactly the requested fields,
    atoms that are GroundAtoms of the domain's fluent (and, with ``derived``, derived) predicates. ``sample`` returns
    the observation of a uniformly drawn instance's initial state. Not flattenable: use the ``bits`` or ``graph``
    observations for array encodings."""

    def __init__(self, observer: SymbolicObserver, seed: Optional[int] = None) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self._observer = observer
        tasks = observer.problems.tasks
        kinds = {"fluent", "derived"} if observer.derived else {"fluent"}
        self._predicates = frozenset(
            p.name for task in tasks for p in task.formalism.predicates if p.kind in kinds
        )
        self._fields = (True, observer.objects, observer.goal, observer.static, observer.numeric)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def sample(self, mask: Any = None, probability: Any = None) -> SymbolicObservation:
        if mask is not None or probability is not None:
            raise ValueError("mymyr: a SymbolicSpace samples without masks")
        tasks = self._observer.problems.tasks
        t = int(self.np_random.integers(len(tasks)))
        return self._observer.observe(tasks[t].initial_state, t)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, SymbolicObservation):
            return False
        if any((value is not None) != wanted for value, wanted in zip(x, self._fields)):
            return False
        if not isinstance(x.atoms, frozenset):
            return False
        for atom in x.atoms:
            if not isinstance(atom, GroundAtom) or atom.predicate.name not in self._predicates:
                return False
        if x.objects is not None and not (isinstance(x.objects, tuple) and all(isinstance(o, str) for o in x.objects)):
            return False
        if x.goal is not None and not isinstance(x.goal, GroundCondition):
            return False
        if x.static is not None and not (
            isinstance(x.static, frozenset) and all(isinstance(a, GroundAtom) and a.kind == "static" for a in x.static)
        ):
            return False
        if x.numeric is not None and not (isinstance(x.numeric, tuple) and all(isinstance(v, float) for v in x.numeric)):
            return False
        return True

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SymbolicSpace)
            and other._observer.problems.uids == self._observer.problems.uids
            and other._fields == self._fields
        )

    def __hash__(self) -> int:
        return hash((self._observer.problems.uids, self._fields))

    def __deepcopy__(self, memo: dict) -> "SymbolicSpace":
        # the copies (e.g. gymnasium's batch_space) share the instances; only the random generator is copied
        out = copy.copy(self)
        out._np_random = copy.deepcopy(self._np_random, memo)
        return out

    def __repr__(self) -> str:
        names = ("atoms", "objects", "goal", "static", "numeric")
        fields = ", ".join(n for n, f in zip(names, self._fields) if f)
        return f"SymbolicSpace({len(self._observer.problems)} instances; {fields})"


# ------------------------------------------------------------------------------------------------ ground actions


class GroundActionSpace(gymnasium.Space):
    """The ground actions of an environment's current instance (pddlgym's literal action space): its elements are
    mymyr Actions of that instance's Task, applicable or not. ``sample`` draws uniformly among the applicable actions
    of the current state; without one, it returns an inapplicable action of the instance (the first applicable
    action of its initial state, else a binding of its schema of the fewest parameters to object 0), which makes no
    move. Bound to one environment (the vector environment's spaces to one env each); equal to the spaces of every
    environment over the same instances."""

    def __init__(self, problems: Problems, current: Any, seed: Optional[int] = None) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self._problems = problems
        self._current = current  # () -> (instance, its applicable actions in the current state)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def sample(self, mask: Any = None, probability: Any = None) -> Action:
        if mask is not None or probability is not None:
            raise ValueError("mymyr: a GroundActionSpace samples without masks")
        t, actions = self._current()
        if actions:
            return actions[int(self.np_random.integers(len(actions)))]
        task = self._problems.tasks[t]
        initial = task.applicable_actions(task.initial_state)
        if initial:
            return initial[0]
        if task.num_objects or not any(s.arity for s in task.formalism.schemas):
            schema = min(range(task.num_schemas), key=lambda i: task.formalism.schemas[i].arity)
            return task.action(schema, [0] * task.formalism.schemas[schema].arity)
        raise ValueError("mymyr: the instance has no ground action to sample (no objects for its schemas)")

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Action):
            return False
        t, _ = self._current()
        return int(x.task.uid) == self._problems.uids[t]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GroundActionSpace) and other._problems.uids == self._problems.uids

    def __hash__(self) -> int:
        return hash(("GroundActionSpace", self._problems.uids))

    def __deepcopy__(self, memo: dict) -> "GroundActionSpace":
        out = copy.copy(self)
        out._np_random = copy.deepcopy(self._np_random, memo)
        return out

    def __repr__(self) -> str:
        return f"GroundActionSpace({len(self._problems)} instances)"


def ground_index(problems: Problems, t: int, actions: list[Action], action: Any, max_actions: int) -> int:
    """The index of a ground action among the applicable actions of a state of instance t; -1 (no move) when it is
    not one of them or its index is max_actions or more. An Action of another task (another instance of the
    domain), a text '(schema o1 ...)' and a (schema, objects) pair are read by their names in instance t, as
    pddlgym's literals are; names that instance does not have make no move either."""
    task = problems.tasks[t]
    if isinstance(action, Action) and int(action.task.uid) != problems.uids[t]:
        action = str(action)
    if not isinstance(action, Action):
        if not isinstance(action, (str, tuple)):
            raise TypeError(f"mymyr: a ground action is an Action, a str or a (schema, objects) tuple, not "
                            f"{type(action).__name__}")
        try:
            action = task.action(action)
        except (KeyError, ValueError):
            return -1
    try:
        index = actions.index(action)
    except ValueError:
        return -1
    return index if index < max_actions else -1


# ------------------------------------------------------------------------------------------------ array observations


class BitsObserver:
    """The fluent atom slots of rows as 0/1 int8 [n] (n: the most slots of an instance; an instance's slots past its
    own are 0), and for numeric tasks also the numeric values (float64, 0 past an instance's slots)."""

    def __init__(self, problems: Problems) -> None:
        self.problems = problems
        self.num_bits = max(1, max(int(task.num_atoms) for task in problems.tasks))
        self.num_values = max(len(task.numeric_names) for task in problems.tasks) if problems.numeric else 0

    def space(self, n: Optional[int] = None) -> spaces.Space:
        shape = (self.num_bits,) if n is None else (n, self.num_bits)
        bits = spaces.MultiBinary(shape)
        if not self.problems.numeric:
            return bits
        big = np.finfo(np.float64).max
        vshape = (self.num_values,) if n is None else (n, self.num_values)
        return spaces.Dict({"atoms": bits, "numeric": spaces.Box(-big, big, vshape, np.float64)})

    def observe(self, rows: np.ndarray, task_ids: np.ndarray) -> Any:
        words = rows[:, : self.problems.words]
        bits = np.unpackbits(words.view(np.uint8), axis=1, count=self.num_bits, bitorder="little").view(np.int8)
        if words.shape[1] * 64 < self.num_bits:
            bits = np.pad(bits, ((0, 0), (0, self.num_bits - bits.shape[1])))
        if not self.problems.numeric:
            return bits
        from mymyr.rl import numeric_values

        values = np.nan_to_num(np.asarray(numeric_values(self.problems.table, rows, task_ids)), nan=0.0)
        if values.shape[1] < self.num_values:
            values = np.pad(values, ((0, 0), (0, self.num_values - values.shape[1])))
        return {"atoms": bits, "numeric": np.ascontiguousarray(values[:, : self.num_values])}


_GRAPH_DEFAULTS = {"goal_derivations": (), "include_actions": False, "emit_counts": False, "entity_arrays": "auto"}
# per-graph arrays of the encoder's batch and their sizes per graph (n graphs: n * size, plus one for the offsets)
_GRAPH_FIXED = {"task_ids": 1, "num_objects": 1, "entity_offsets": 1, "relation_offsets": "R", "relation_arg_offsets": "R"}
_GRAPH_VARIABLE = ("entity_role", "entity_ref", "relation_args")


class GraphObserver:
    """Relational graphs of rows through mymyr.encoding: the arrays of the encoder's FlatBatch as a dict (one graph
    per row; the scalars of ``as_dict`` are left out)."""

    def __init__(self, problems: Problems, options: Optional[dict]) -> None:
        from mymyr.encoding import FlatRelationEncoder

        if problems.numeric:
            raise ValueError("mymyr: graph observations need a classical task (the encoder refuses numeric fluents)")
        self.problems = problems
        self.encoder = FlatRelationEncoder(problems.domain_table(), **{**_GRAPH_DEFAULTS, **(options or {})})
        self.num_relations = int(self.encoder.num_relations)
        probe = self.encoder.encode([problems.tasks[0].initial_state], [0])
        arrays = {k: v for k, v in probe.as_dict().items() if isinstance(v, np.ndarray)}
        self.keys = tuple(arrays)
        self.dtypes = {k: v.dtype for k, v in arrays.items()}
        max_objects = max(int(task.num_objects) for task in problems.tasks)
        self.max_entities = max_objects + 1  # the auxiliary entity of the nullary atoms

    def _shape(self, key: str, n: int) -> tuple:
        if key == "relation_counts":
            return (n, self.num_relations)
        size = _GRAPH_FIXED[key]
        return (n * (self.num_relations if size == "R" else 1) + (1 if key.endswith("offsets") else 0),)

    def space(self, n: int = 1) -> spaces.Dict:
        out: dict[str, spaces.Space] = {}
        for key in self.keys:
            dtype = self.dtypes[key]
            high = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
            if key in _GRAPH_VARIABLE:
                # Discrete elements: the stacked sequence's items are NumPy integer scalars, which Discrete takes
                if key == "relation_args":  # batch-global entity ids
                    item = spaces.Discrete(n * self.max_entities)
                elif key == "entity_ref":  # instance-local object ids, -1 for the auxiliary entity
                    item = spaces.Discrete(self.max_entities, start=-1)
                else:
                    item = spaces.Discrete(int(high) + 1)
                out[key] = spaces.Sequence(item, stack=True)
            elif key == "task_ids":
                # bounded by at least 1 so that a table of one instance does not give a degenerate Box
                out[key] = spaces.Box(0, max(1, len(self.problems) - 1), (n,), dtype)
            elif key == "num_objects":
                out[key] = spaces.Box(0, max(1, self.max_entities - 1), (n,), dtype)
            else:
                out[key] = spaces.Box(0, min(high, _INT32_MAX), self._shape(key, n), dtype)
        return spaces.Dict(out)

    def observe(self, rows: np.ndarray, task_ids: np.ndarray) -> dict:
        batch = self.encoder.encode(rows[:, : self.problems.words], task_ids)
        arrays = batch.as_dict()
        return {key: arrays[key] for key in self.keys}


def make_observer(problems: Problems, mode: str, symbolic: dict, graph_options: Optional[dict]) -> Any:
    if mode == "symbolic":
        return SymbolicObserver(problems, **symbolic)
    if mode == "bits":
        return BitsObserver(problems)
    if mode == "graph":
        return GraphObserver(problems, graph_options)
    raise ValueError(f"mymyr: observation must be one of {OBSERVATION_MODES}, not {mode!r}")


def check_action_mode(mode: str) -> str:
    if mode not in ACTION_MODES:
        raise ValueError(f"mymyr: action must be one of {ACTION_MODES}, not {mode!r}")
    return mode


def env_config(max_steps: int, step_reward: float, goal_reward: float, dead_end: str, dead_end_reward: float,
               dead_end_terminal: bool) -> dict:
    """The CpuEnvPool keywords of the RL contract (docs/rl.md)."""
    return dict(max_steps=max_steps, step_reward=step_reward, goal_reward=goal_reward, dead_end=dead_end,
                dead_end_reward=dead_end_reward, dead_end_terminal=dead_end_terminal)


def render_text(problems: Problems, t: int, state: State, steps: int, count: int) -> str:
    """The ansi rendering of a state: the instance, the true atoms and the goal's literals ([x]: holds)."""
    task = problems.tasks[t]
    atoms = sorted(str(atom) for atom in state.atoms())
    lines = [f"instance {t} ({task.domain_name} {task.problem_name}), step {steps}, {count} applicable actions",
             "state: " + (" ".join(atoms) if atoms else "(no atoms)")]
    if task.numeric_slots:
        lines.append("values: " + " ".join(f"{name}={value:g}" for name, value in
                                           zip(task.numeric_names, state.numeric_values())))
    goal = task.goal_condition
    parts = [("[x] " if state.holds(lit) else "[ ] ") + str(lit) for lit in goal.literals]
    parts += [str(c) for c in goal.numeric_constraints]
    lines.append("goal: " + ("; ".join(parts) if parts else "(empty)") + (" (reached)" if goal.holds(state) else ""))
    return "\n".join(lines) + "\n"


def as_int_ids(values: Any, n: int, problems: Problems, name: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"mymyr: {name} must be {n} ints")
    if ((ids < 0) | (ids >= len(problems))).any():
        raise ValueError(f"mymyr: {name} has an instance outside the {len(problems)} instances")
    return ids.astype(np.int32)


__all__ = ["GroundActionSpace", "SymbolicObservation", "SymbolicSpace"]
