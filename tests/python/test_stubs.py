"""The typing stubs of mymyr._core (python/CMakeLists.txt, MYMYR_PY_STUBS).

The build runs nanobind's stubgen on the new module and installs mymyr/_core/*.pyi next to the extension. These tests
check that the installed stubs parse, carry the typed signatures of mymyr.search, and match a fresh stubgen run on the
loaded module (so a stale stub package fails).
"""

import ast
import filecmp
import importlib.util
import pathlib
import subprocess
import sys

import pytest

import mymyr  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUBMODULES = {"__init__.pyi", "_datasets.pyi", "_formalism.pyi", "_rl.pyi", "_rl_jax.pyi", "_rl_ops.pyi", "_rl_torch.pyi",
              "_search.pyi"}


def installed_stubs() -> pathlib.Path:
    d = pathlib.Path(importlib.util.find_spec("mymyr._core").origin).parent / "_core"
    if not (d / "__init__.pyi").is_file():
        pytest.skip("no stubs installed (MYMYR_PY_STUBS=OFF or a sanitizer build)")
    return d


def test_stubs_parse_and_are_typed():
    d = installed_stubs()
    assert SUBMODULES <= {p.name for p in d.glob("*.pyi")}
    for p in d.glob("*.pyi"):
        ast.parse(p.read_text(), filename=str(p))
    search = (d / "_search.pyi").read_text()
    assert "def iw(task: mymyr._core.Task | mymyr._core.TaskHandle, *, max_arity: int = 2," in search
    assert "max_seconds: float | None = None" in search
    assert "cancel: CancelToken | None = None" in search
    assert (
        "heuristic: str | Heuristic | Callable[[mymyr._core.State], float] | mymyr._typing.HeuristicObject = 'max'"
        in search
    )
    assert "def perfect(space: mymyr._core._datasets.StateSpace, *, costs: str = 'unit') -> Heuristic:" in search
    assert "def plan(self) -> list[mymyr._core.Action]" in search
    assert "def goal_state(self) -> mymyr._core.State | None" in search
    # the IW family variants, landmarks and relaxed reachability (search_bindings.cpp, landmarks_bindings.cpp)
    task = "task: mymyr._core.Task | mymyr._core.TaskHandle"
    landmarks = "landmarks: FactLandmarkGraph | Literal['approximate', 'lifted'] | None = None"
    assert f"def liw({task}, *, max_arity: int = 2, {landmarks}, disjunctive: bool = False," in search
    assert f"def abstracted_iw({task}, *, width: int = 1, base_abstracted: bool = False," in search
    assert f"def astar_iw({task}, *, heuristic:" in search
    assert "features: str = 'classical'" in search
    assert "allow_non_novel_root_goal: bool = True" in search
    assert "probe_novelty_before_heuristic: bool = True" in search
    assert "class AStarIwResult(BestFirstResult):" in search
    assert f"def projective_iw({task}, *, typed_projection: bool = False," in search
    assert "layer_order: Literal['queue', 'in_order', 'reverse', 'randomized', 'goal_count'] = 'queue'" in search
    assert "max_next_layer_states: int | None = None, prefer_more_satisfied_goals: bool = True" in search
    beam = ("beam_width: int | None = None, beam_novelty: Literal['all_tested', 'survivors_only'] = 'all_tested', "
            "randomize_ties: bool = False")
    assert search.count(beam) == 7  # iw, iw_pass, siw, brfs, liw, abstracted_iw, projective_iw
    # iw, iw_pass, siw, brfs, astar, astar_iw, gbfs, beam, liw, abstracted_iw, projective_iw, rollout_iw,
    # find_rollouts_parallel, atomic_goal_portfolio
    assert search.count("symmetry_pruning: Literal['off', 'wl1'] = 'off'") == 14
    assert f"def rollout_iw({task}, *, ordering: ActionOrdering | str = ActionOrdering.IN_ORDER, seed: int = 0," in search
    assert f"def find_rollouts_parallel({task}, seeds: Sequence[int], *, max_arity: int = 2, num_threads: int = 0," in search
    rollouts = "results: ParallelRolloutsResult | Sequence[RolloutResult]"
    assert f"def intersect_co_occurrence({rollouts}) -> dict[mymyr._core._formalism.GroundAtom, list[mymyr._core._formalism.GroundAtom]]" in search
    assert f"def merge_landing_states({rollouts}) -> MergedLandingStates" in search
    assert "rollout_orderings: Sequence[ActionOrdering | str | tuple[ActionOrdering | str, int]] | None = None" in search
    assert "def co_occurrence(self) -> dict[mymyr._core._formalism.GroundAtom, list[mymyr._core._formalism.GroundAtom]]" in search
    assert "def landing_state_by_atom(self) -> dict[mymyr._core._formalism.GroundAtom, int]" in search
    assert "def rollout_statuses(self) -> list[Status | None]" in search
    assert "def winning_worker(self) -> int | None" in search
    assert f"def lifted_fact_landmarks({task}, *, reachability: RelaxedReachability | None = None," in search
    assert ") -> WitnessQuery: ..." in search
    for name, result in (("astar_iw", "AStarIwResult"), ("liw", "IwResult"), ("abstracted_iw", "IwResult"), ("projective_iw", "IwResult"),
                         ("rollout_iw", "RolloutIwResult"), ("find_rollouts_parallel", "ParallelRolloutsResult"),
                         ("atomic_goal_portfolio", "PortfolioResult"), ("approximate_fact_landmarks", "FactLandmarkGraph"),
                         ("lifted_fact_landmarks", "FactLandmarkGraph")):
        sig = next(line for line in search.splitlines() if line.startswith(f"def {name}("))
        assert sig.endswith(f") -> {result}:"), sig
    # every parameter and result typed: observers are mymyr.search.Observer (any object with some of its methods works)
    assert "observer: mymyr.search.Observer | None = None" in search
    brfs = next(line for line in search.splitlines() if line.startswith("def brfs("))
    for arg in ("max_seconds: float | None = None", "cancel: CancelToken | None = None",
                "observer: mymyr.search.Observer | None = None", "progress_interval: int | None = None"):
        assert arg in brfs, arg
    assert ": object" not in search.replace("def __eq__(self, other: object, /)", "")
    assert "-> object" not in search and "-> list:" not in search and "-> dict:" not in search
    core = (d / "__init__.pyi").read_text()
    assert "__version__: str" in core
    assert "_C_API: types.CapsuleType" in core
    assert "def __eq__(self, other: object, /) -> bool" in core
    assert "def __eq__(self, arg" not in core and "def __reduce__(self) -> object" not in core
    # the task API (typing.hpp, py_task.hpp): states as States or DLPack word arrays, typed results
    assert "import mymyr._typing" in core
    partial = ("partial: dict[int | str | _formalism.Variable, _formalism.Object | str | int | None] | "
               "Sequence[_formalism.Object | str | int | None] | None = None")
    symmetry = "symmetry_pruning: Literal['off', 'wl1'] = 'off'"
    assert (f"def applicable_actions(self, state: State | mymyr._typing.SupportsDLPack, *, schema: str | int | None = None, "
            f"{partial}, {symmetry}) -> list[Action]") in core
    # binding generators (task_bindings.cpp)
    target = ("target: str | int | _formalism.ConjunctiveCondition | _formalism.GroundCondition, "
              "state: State | mymyr._typing.SupportsDLPack")
    assert f"def bindings(self, {target}, {partial}, limit: int | None = None) -> Bindings" in core
    assert f"def ground_conjunctions(self, {target}, {partial}, limit: int | None = None) -> GroundConjunctions" in core
    assert "def __next__(self) -> Action | tuple[_formalism.Object, ...]" in core
    assert "def precondition(self, schema: str | int) -> _formalism.ConjunctiveCondition" in core
    assert "def goal_condition(self) -> _formalism.GroundCondition" in core
    assert f"def successors(self, state: State | mymyr._typing.SupportsDLPack, *, {symmetry}) -> list[tuple[Action, State]]" in core
    assert f"def successor_states(self, *, {symmetry}) -> list[State]" in core
    # formulas (formula_bindings.cpp): constructors on the task, values in mymyr._core._formalism
    head = ("predicate: str | int | _formalism.Predicate | _formalism.GroundAtom | _formalism.Atom | "
            "_formalism.GroundLiteral | _formalism.Literal | tuple[str | int | Sequence[str | int], ...], *terms")
    assert f"def atom(self, {head}) -> _formalism.GroundAtom | _formalism.Atom" in core
    assert f"def literal(self, {head}, positive: bool = True) -> _formalism.GroundLiteral | _formalism.Literal" in core
    assert ") -> _formalism.ConjunctiveCondition:" in core and "constraints: Iterable[str | _formalism.NumericConstraint] = ()" in core
    assert ("def holds(self, formula: _formalism.GroundAtom | _formalism.GroundLiteral | _formalism.GroundCondition | str | int | "
            "tuple[str | int | Sequence[str | int], ...]) -> bool") in core
    fm = (d / "_formalism.pyi").read_text()
    assert "def holds(self, state: mymyr._core.State) -> bool" in fm
    assert "def lift(self, add_inequalities: bool = False) -> ConjunctiveCondition" in fm
    assert "def ground(self, state: mymyr._core.State, limit: int | None = None, partial: " in fm
    assert ") -> list[GroundCondition]:" in fm
    assert "framework: Literal['numpy', 'torch', 'jax', 'dlpack'] | None = None) -> Any" in core
    assert "def owner(self) -> Task | TaskHandle" in core
    assert "state: object" not in core and "action: object" not in core and "-> list:" not in core
    # CUDA streams (typing.hpp StreamArg); contexts are typing.Any here, since CPU builds have no mymyr.cuda
    rl = (d / "_rl.pyi").read_text()
    assert "stream: int | mymyr._typing.SupportsCudaStream | None = None, ctx: Any | None = None" in rl
    # the environments of mymyr.rl.torch
    env = (d / "_rl_torch.pyi").read_text()
    table = "table: mymyr._core._rl.TaskSuite | mymyr._core._rl.TaskTable | mymyr._core.Task | mymyr._core.TaskHandle"
    assert f"def __init__(self, {table}, *, device: int | None = None," in env
    assert "path: Literal['auto', 'fast', 'general'] = 'auto'" in env
    ids = "task_ids: mymyr._typing.SupportsDLPack | None = None"
    assert f"def step(self, states: mymyr._typing.SupportsDLPack, {ids}, steps: mymyr._typing.SupportsDLPack | None = None," in env
    assert "dead_end: Literal['no_successors', 'none'] = 'no_successors'" in env
    assert "def _step_ptrs(self, rows: int, ptrs: Sequence[int], label_width: int, first_env: int = 0," in env
    assert "def successor_indices(seed: int, envs: mymyr._typing.SupportsDLPack" in env
    assert ": object" not in env and "-> object" not in env
    # the XLA FFI environments, the RL helpers and the random policy's actions
    assert "def random_actions(self, draws: mymyr._typing.SupportsDLPack, count: mymyr._typing.SupportsDLPack," in env
    jx = (d / "_rl_jax.pyi").read_text()
    assert "witness: bool = False, path: Literal['auto', 'fast', 'general'] = 'auto') -> None" in jx
    assert "def ffi_targets() -> list[tuple[str, str, dict[str, types.CapsuleType], bool]]" in jx
    assert "def initial_states(self) -> list[int]" in jx and "def initial_counts(self) -> list[int]" in jx
    # task tables, per-row task ids, the CPU pool
    assert "class TaskTable:" in rl and "def __init__(self, tasks: Sequence[mymyr._core.Task | mymyr._core.TaskHandle]) -> None" in rl
    rows = "states: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack"
    tids = "task_ids: mymyr._typing.SupportsDLPack | Sequence[int] | None = None"
    assert f"def expand(table: TaskSuite | TaskTable | mymyr._core.Task | mymyr._core.TaskHandle, {rows}, {tids}, *," in rl
    # task sets: a table of a domain's problem files (front-end builds)
    if "def from_pddl(" in rl:
        assert ("def from_pddl(domain: mymyr._core.Domain | str | os.PathLike, problems: str | os.PathLike | "
                "Sequence[str | os.PathLike], *, atoms: str = 'auto',") in rl
        assert "pilot_expansions: int = 1024, threads: int = 0) -> TaskTable:" in rl
    # knowledge bases and tuple graphs
    ds = (d / "_datasets.pyi").read_text()
    assert ("def tuple_graphs(space: StateSpace, *, width: int = 0, dominance_pruning: bool = True, threads: int = 0) -> "
            "list[TupleGraph]:") in ds
    assert "def tuple_graph(space: StateSpace, vertex: int, *, width: int = 0, dominance_pruning: bool = True) -> TupleGraph:" in ds
    assert "def atoms(self, vertex: int) -> list[mymyr._core._formalism.GroundAtom]:" in ds
    assert "def arrays(self, framework: Literal['numpy', 'torch', 'jax', 'dlpack'] | None = None) -> dict[str, Any]:" in ds
    kb = ("def __init__(self, tasks: mymyr._core._rl.TaskTable | Sequence[mymyr._core.Task | mymyr._core.TaskHandle], *, "
          "threads: int = 0, max_states: int | None = None,")
    assert kb in ds
    assert "generalized: bool = False, width: int | None = None, dominance_pruning: bool = True) -> None:" in ds
    assert "def generalized_state_space(self) -> GeneralizedStateSpace | None:" in ds
    assert "def tuple_graphs(self, space: int) -> list[TupleGraph]:" in ds
    assert "def __getstate__(self) -> tuple[mymyr._core._rl.TaskTable, int," in ds
    # task suites (several domains), taken wherever a table is
    assert "class TaskSuite:" in rl and "def __init__(self, tables: Sequence[TaskTable]) -> None" in rl
    assert "def group(tasks: Sequence[mymyr._core.Task | mymyr._core.TaskHandle]) -> TaskSuite" in rl
    assert "def domain_of(self) -> list[int]" in rl and "def global_id(self, domain: int, local: int) -> int" in rl
    assert "def __getstate__(self) -> tuple[list[mymyr._core.Task]]" in rl
    suite = "table: TaskSuite | TaskTable | mymyr._core.Task | mymyr._core.TaskHandle"
    assert f"def __init__(self, {suite}, num_envs: int, *, threads: int = 0," in rl
    assert f"def fast_unsupported({table}, *, canonical: bool = True," in env
    assert "def recv(self, min_rows: int = 0) -> PoolBatch" in rl and "class CpuEnvPool:" in rl
    assert f"def __init__(self, {table}, *, device: int | None = None," in jx
    assert ": object" not in rl and "-> object" not in rl and "-> list:" not in rl and "-> dict:" not in rl
    ops = (d / "_rl_ops.pyi").read_text()
    assert "strategy: Literal['future', 'final', 'episode'] = 'future'" in ops
    assert "prefix: mymyr._typing.SupportsDLPack | None, depth: int, mask: mymyr._typing.SupportsDLPack" in ops
    for text in (jx, ops):
        assert ": object" not in text and "-> object" not in text
        assert "-> list:" not in text and "-> dict:" not in text and "-> tuple:" not in text


def test_cuda_stubs_are_typed():
    d = installed_stubs()
    if not (d / "_cuda.pyi").is_file():
        pytest.skip("a CPU build: no mymyr._core._cuda")
    cuda = (d / "_cuda.pyi").read_text()
    stream = "stream: int | mymyr._typing.SupportsCudaStream | None = None"
    assert "def brfs(task: mymyr._core.Task | mymyr._core.TaskHandle, *, ctx: Context | None = None," in cuda
    assert f"def validate(self, {stream}) -> tuple[int, int]" in cuda
    assert "memory: Literal['device', 'managed', 'pinned'] = 'device', ctx: Context | None = None" in cuda
    assert "def host_derived(self, states: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack)" in cuda
    assert "def padded(self) -> DevicePaddedExpansion | None" in cuda
    assert "def actions(self, state: int) -> list[mymyr._core.Action]" in cuda
    assert "def status(self) -> mymyr._core._search.Status" in cuda
    assert "def stats(self) -> dict[str, float | int]" in cuda
    # multi-search IW, rollouts, batched IW(1)
    goal = "mymyr._core._formalism.GroundCondition | tuple[Sequence[int], Sequence[int]]"
    goals = f"goals: Sequence[{goal}] | None = None"
    assert f"starts: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack, *, ctx: Context | None = None, {goals}" in cuda
    assert "def rollouts(task: mymyr._core.Task | mymyr._core.TaskHandle, seeds: Sequence[int], *, ctx: Context | None = None" in cuda
    assert f"start: mymyr._core.State | None = None, goal: {goal} | None = None" in cuda
    assert f"ctx: Context | None = None, {stream}, {goals}, exact: bool = True" in cuda
    assert "-> IwBatch:" in cuda
    assert "def status(self) -> list[mymyr._core._search.Status]" in cuda
    assert "def passes(self, i: int) -> list[mymyr._core._search.IwPass]" in cuda
    assert "def goal_state(self, i: int) -> mymyr._core.State | None" in cuda
    # device heuristics, A*, GBFS
    assert "kind: Literal['max', 'add', 'ff', 'h2', 'set_additive'] = 'ff', *, costs: Literal['unit', 'real'] = 'unit', ctx: Context | None = None" in cuda
    assert f"def evaluate(self, states: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack, *, {stream}) -> Any" in cuda
    assert "def reference(self, state: mymyr._core.State) -> float" in cuda
    assert "max_scratch_bytes: int | None = None" in cuda
    assert "heuristic: Literal['blind', 'max', 'add', 'ff', 'h2', 'set_additive'] = 'max', costs: Literal['unit', 'real'] = 'unit'" in cuda
    assert "variant: Literal['auto', 'sweep', 'frontier'] = 'auto') -> DeviceSearchResult" in cuda
    assert "def stats(self) -> mymyr._core._search.Statistics" in cuda
    assert "def device(self) -> dict[str, float | int | bool | str]" in cuda
    # device state spaces, IW over task tables, samplers over device spaces
    table = "mymyr._core._rl.TaskTable | mymyr._core.Task | mymyr._core.TaskHandle"
    ids = "task_ids: mymyr._typing.SupportsDLPack | Sequence[int] | None = None"
    dss = f"ctx: Context | None = None, device: int | None = None, {stream}, threads: int = 0"
    assert f"def state_space(task: mymyr._core.Task | mymyr._core.TaskHandle, *, {dss}," in cuda
    assert ") -> DeviceStateSpace | None:" in cuda
    assert f"def state_spaces(table: {table}, *, {dss}," in cuda
    assert "wave_instances: int | None = None) -> list[DeviceStateSpace | None]:" in cuda
    assert "output: Literal['device', 'host', 'both'] = 'device') -> DeviceGenerationResult:" in cuda
    assert "wave_instances: int | None = None) -> list[DeviceGenerationResult]:" in cuda
    assert "def to_host(self) -> mymyr._core._datasets.StateSpace" in cuda
    assert "def host(self) -> mymyr._core._datasets.StateSpace | None" in cuda
    assert f"def multi_iw(task: {table}, starts: mymyr._core.State | Sequence[mymyr._core.State] | mymyr._typing.SupportsDLPack, *," in cuda
    assert f"def batched_iw1(task: {table}," in cuda and "def task_ids(self) -> list[int]" in cuda
    assert f"chunk_states: int | None = None, {ids}) -> IwBatch:" in cuda
    ds = (d / "_datasets.pyi").read_text()
    assert "def __init__(self, space: StateSpace | mymyr._core._cuda.DeviceStateSpace, seed: int = 0) -> None" in ds
    assert ": object" not in cuda and "-> object" not in cuda
    assert "-> list:" not in cuda and "-> dict:" not in cuda and "-> tuple:" not in cuda


def test_stubs_are_current(tmp_path):
    d = installed_stubs()
    pytest.importorskip("nanobind")
    cmd = [sys.executable, "-m", "nanobind.stubgen", "-q", "-r", "-P", "-p", str(ROOT / "python" / "stubgen-patterns.txt"),
           "-m", "mymyr._core", "-O", str(tmp_path)]
    subprocess.run(cmd, check=True, timeout=120)
    fresh = tmp_path / "_core"
    names = sorted(p.name for p in fresh.glob("*.pyi"))
    assert names == sorted(p.name for p in d.glob("*.pyi"))
    match, mismatch, errors = filecmp.cmpfiles(fresh, d, names, shallow=False)
    assert not mismatch and not errors, f"stale stubs: {mismatch + errors}; rebuild the package"
