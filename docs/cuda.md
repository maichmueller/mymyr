# CUDA

The optional CUDA backend needs an NVIDIA GPU at runtime. The wheels on PyPI contain the CPU backend; the GitHub
Releases after 0.1.0 carry CUDA wheels for Linux ([Installing a CUDA wheel](#installing-a-cuda-wheel)). For a source
build, set `MYMYR_CUDA=ON` and `CMAKE_CUDA_COMPILER` during the CMake configure or Python package build. The device
searches and generators take the options and return the result fields of the CPU ones
([API conventions](conventions.md)); a device result adds `counters`, a dict of device counts and times.

## Installing a CUDA wheel

The [GitHub Releases](https://github.com/maichmueller/mymyr/releases) after 0.1.0 carry CUDA wheels next to the CPU
wheels (v0.1.0 has CPU wheels only), with the local version `<version>+cu13`, e.g.
`mymyr-<version>+cu13-cp314-cp314t-manylinux_2_34_x86_64.whl`:

- Linux x86_64 with glibc 2.34 or newer, free-threaded CPython 3.14 and 3.15 (`cp314t`, `cp315t`);
- SASS for compute capabilities 8.6 (Ampere: A10, A40, RTX 30 series), 8.9 (Ada: L4, L40S, RTX 40 series) and 9.0
  (Hopper: H100, H200) and no PTX, so other GPUs, among them the A100 (8.0) and Blackwell GPUs, need a wheel built for
  them ([Building a CUDA wheel](#building-a-cuda-wheel));
- an NVIDIA driver 580 or newer ([Driver](#driver)); the CUDA runtime is linked into the extension, so no CUDA
  toolkit is needed;
- the XLA FFI targets of `mymyr.rl.jax` and `mymyr.encoding.jax`; JAX and torch are installed separately. The wheels
  are tested with torch 2.14 for CUDA 13.0 loaded first in the same process;
- about 1.3 GB per wheel: the device code of the three architectures is in the extension and again in the C++ SDK's
  static library.

PyPI has no `+cu13` files: ask for the version and add the release's assets as a source of packages, with
`<version>` the release's version,

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python "mymyr==<version>+cu13" \
    --find-links https://github.com/maichmueller/mymyr/releases/expanded_assets/v<version>
```

or install a wheel by its URL, with the `+` written as `%2B`:

```bash
uv pip install --python .venv/bin/python \
    https://github.com/maichmueller/mymyr/releases/download/v<version>/mymyr-<version>%2Bcu13-cp314-cp314t-manylinux_2_34_x86_64.whl
```

`pip install` takes the same arguments. `mymyr.build_info()` names the toolkit and the GPU code of the build, e.g.
`mymyr <version>+cu13 (Release, gcc 16.1.0, CUDA 13.4.92, sm_86 sm_89 sm_90)`.

## Building

The device code is C++23, compiled by nvcc with the project's GCC 16 as its host compiler (the host code stays C++26).
The configure step refuses an older setup with a message:

- CUDA Toolkit 13.4 or newer (nvcc 13.4 is the first to accept GCC 16 and `-std=c++23` in both compilation passes),
  with the CUDA runtime and CCCL headers;
- CMake 4.4 or newer (the first to map CUDA C++23 to nvcc's flag; `pip install "cmake>=4.4"` provides it).

nvcc's host compiler is the C++ compiler when that is GCC; `CMAKE_CUDA_HOST_COMPILER` or the `CUDAHOSTCXX` environment
variable choose another. Its standard library must be the one the C++ side uses, because both link into one binary.
`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (SASS for the visible GPU). For a build that runs on several GPU
generations, list real architectures, and a virtual one only for the newest, e.g.
`-DCMAKE_CUDA_ARCHITECTURES="80-real;86-real;89-real;90-real;100-real;120"`: `120` adds SASS and PTX for sm_120.

The test that checks the device Philox generator against cuRAND's is built when the toolkit provides cuRAND's headers
(`libcurand-dev-13-4` on Debian and Ubuntu); without them it is not built, and the library does not need them.

## Building a CUDA wheel

`ci/cuda_wheel.sh` builds the release's CUDA wheels, and builds a wheel for other GPUs or Python ABIs the same way. It
runs in the manylinux image of the release wheels (`manylinux-x86_64-image` under `[tool.cibuildwheel.linux]` in
`pyproject.toml`): it installs GCC 16, the dependencies and the CUDA 13.4 compiler (checksums in `ci/cuda_redist.json`)
into a toolchain directory that later builds reuse, builds the wheel with the version `<version>+cu13` and pinned build
requirements, repairs it with auditwheel, checks that it needs only glibc and holds exactly the requested GPU code, and
runs the CPU test suite on it with torch loaded first. It needs network access; its work directory takes about 1.2 GB
per architecture for the build tree, which it removes before the test environment (about 4.5 GB) is installed, and the
toolchain directory about 1.7 GB. One architecture compiled in 50 minutes with 8 build jobs and 4 parallel nvcc
compiles. Nearly all of that time is per architecture (cicc and ptxas; the front end that the architectures share is a
few percent), so the three architectures of the release wheels take three times as long. `--stage compile --minutes N`
stops the build cleanly N minutes after the script started (`--deadline T` at the Unix time T instead), `--stage resume` on the same work directory and toolchain
resumes it and stops it the same way, and `--stage finish` resumes it to its end and then packages and tests the wheel;
the release workflow splits each wheel across up to three jobs this way.

From the repository root, with podman (`docker run` takes the same arguments):

```bash
image=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["tool"]["cibuildwheel"]["linux"]["manylinux-x86_64-image"])')
mkdir -p ~/.cache/mymyr-cuda-toolchain /scratch/mymyr-cuda-wheel
podman run --rm -v "$PWD":/src -v ~/.cache/mymyr-cuda-toolchain:/opt/mymyr-toolchain \
    -v /scratch/mymyr-cuda-wheel:/work "$image" \
    bash /src/ci/cuda_wheel.sh --python cp314t --archs "80-real" --work /work --out /work/wheelhouse
```

With apptainer, convert the image once (set `APPTAINER_TMPDIR` to a local disk if unpacking fails on a network file
system), then run the script in it:

```bash
apptainer build manylinux_2_34.sif "docker://$image"
apptainer exec --containall --bind "$PWD":/src --bind ~/.cache/mymyr-cuda-toolchain:/opt/mymyr-toolchain \
    --bind /scratch/mymyr-cuda-wheel:/work manylinux_2_34.sif \
    bash /src/ci/cuda_wheel.sh --python cp314t --archs "80-real" --work /work --out /work/wheelhouse
```

`--archs` takes `CMAKE_CUDA_ARCHITECTURES` (SASS for the GPUs you run on: `80-real` for A100, `90-real` for H100,
`100-real` for B200), `--python` a CPython ABI of the image (`cp314t`, `cp315t`, `cp314`, `cp315`), and
`--jobs` and `--cuda-jobs` bound the parallel compiles (nvcc needs up to 3 GB per compile, 7 GB for the largest
kernels); `ci/cuda_wheel.sh --help` lists the options. The wheel lands in `--out`. To test it on the GPU, run
`ci/attach_cuda_wheels.sh --dry-run --dir /scratch/mymyr-cuda-wheel/wheelhouse --python <python3.14t> --abis cp314t --archs 80 v<version>`
from a checkout of the same version on a GPU node: it installs the pinned test requirements (torch 2.14 for CUDA 13.0
among them) and the wheel into a fresh environment that also sees the interpreter's JAX, checks that the
environment's mymyr is the wheel's and exchanges arrays with torch on GPU 0, and runs the CUDA tests on GPU 0.

## Driver

Device code built with a CUDA 13.x toolkit runs on NVIDIA drivers 580 or newer through CUDA minor-version
compatibility, as SASS for the GPU's own architecture. PTX is compiled by the driver at load time and needs a driver
that knows the toolkit's PTX version (newer than the minimum for SASS), so build SASS for every GPU generation you
run on and keep PTX only as the forward-compatible fallback for GPUs newer than the build. A GPU that has neither
matching SASS nor loadable PTX fails at the first device call.

Device operations include task upload and smoke kernels, BrFS, multi-IW and rollouts; batched grounded h-max, h-add,
h-FF, h² and set-additive; A* and GBFS; state-space generation; and the device path for `mymyr.rl.expand`. Device arrays can exchange
data with JAX and torch through DLPack. `mymyr.cuda.Context` owns a stream-ordered memory pool; set `max_bytes=` to
cap it. Arrays and device tasks keep their context alive.

Collecting an array while its allocation stream or a recorded DLPack consumer stream is capturing retains its
storage until that graph, its executable clones and their outstanding launches are released. This also applies
to warm-up outputs collected during capture on a torch side stream. Reclamation runs on a host worker, so a live
graph can keep the context's `usage()["used"]` elevated even after `synchronize()`. Release the graph and its
executables to return that storage to the pool; subsequent context calls observe completed reclamation.

The encoding helpers `literal_slots` and `literal_masks` resolve CUDA torch arrays of table predicate ids and
instance-local object ids into slots and goal or subgoal masks. They support both frozen atom layouts and CUDA graph
capture after warming the encoder on the capture stream; invalid inputs report value 4 in a device status scalar.
See [Encoding on the device](encoding.md#encoding-on-the-device) for shapes and the torch custom ops.

CUDA upload, expansion, BrFS, multi-IW, rollouts, state spaces, RL environments and A*/GBFS support numeric fluent
values, preconditions, effects, goals and metric costs. Public state rows preserve CPU I32/F64 encoding; internal
numeric rows hold one canonical double per fluent slot. TaskTable/TaskSuite numeric batches use per-instance kernels
with the shared atom/numeric row boundary. Classical tasks retain their atom-only layout and device kernels.

The device heuristic kinds are `"max"`, `"add"`, `"ff"`, `"h2"` and `"set_additive"`; `"blind"` is also accepted by device A* and GBFS.
Relaxation ignores numeric values and constraints as on the CPU, and the heuristic costs are the CPU heuristics'
(`costs="auto"` by default: the task's objective); real costs must stay below 2^31 units of their cost scale. Numeric
searches support fractional and state-dependent action costs. Cost-only classical searches require integral,
state-independent costs. Numeric A*/GBFS use
a host double-priority heap for CPU eager ordering, with device successor, cost and heuristic evaluation. Weighted A*
(`BestFirstOptions::weight` other than 1), a cost bound, enforced hill-climbing and the anytime search run on the CPU
only: the device searches refuse a weight or a bound with `std::invalid_argument`.

Multi-IW, batched IW(1) and rollouts accept `(positive_slots, negative_slots)` tuples or `GroundCondition` goals with
fluent and derived literals and numeric constraints. Multi-IW/batched IW lists select one conjunction per start;
rollouts take one conjunction shared by their seeds. Derived goals use the device axiom closure or the CPU axiom
fallback. Numeric goal expressions use at most 64 stack values. Callable goals and any-of alternatives within one
search require CPU search; device A*/GBFS use the task goal. See [Formula values](formulas.md) for goal construction.
The bindings reject goals simplified as impossible because of static literals or unreachable positive atoms.

Numeric searches and environments do not capture CUDA graphs, and numeric environments use the general path rather
than the environment fast path, because these paths synchronize for counts, lazy atoms and overflow checks.
Expansion and environments run states of any width on the device, and tasks of up to 2048 objects (object bitsets of
up to 32 words). Successor rows wider than 64 words are built 64 words at a time; BrFS, IW, A*, state spaces and the
numeric kernels hold whole rows and take states of at most 64 words. A `DeviceExpander` expands an instance the
kernels cannot run (more than 2048 objects, numeric rows wider than 64 words) on the host: its rows are copied to the
host, expanded on the host's threads and the successors copied back, which synchronizes. In a table or suite only
those instances take the host and the others stay on the device, so the environments' general path serves every
table and suite; the fast path and CUDA graph capture need every instance on the device. Unsupported schema/axiom
plans use the documented CPU fallback.
The derived predicates `axiom_<k>` compiled from quantified and disjunctive conditions (see
[Parsing and tasks](parsing-and-tasks.md)) are axioms like any other: the device evaluates their rules, unless a rule
has a numeric comparison or more parameters than the device kernels take; then the task's axioms run on the CPU
(`counters["device_axioms"] == 0` of a device BrFS result) and the results are the same.
The device evaluates numeric conditions and effects only in tasks with numeric state values: a task whose numeric
conditions or effects read only static functions and functions that never get a value (no initial value, no `assign`
effect) is refused with `ValueError`; run it on the CPU.

This BrFS example runs the repository's classical blocks fixture on device 0:

```python skip-if-no-cuda
from pathlib import Path

import mymyr
from mymyr import cuda

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
result = cuda.brfs(task)
print(result.status, result.solved, len(result.plan), result.cost, result.stats)
print(result.counters)  # per-phase device times and counters
```

Set `CUDA_VISIBLE_DEVICES=0` to select the device exposed to the process. `cuda.available()` reports whether the build
can see a CUDA device. `mymyr.cuda` cannot be imported from a CPU-only build.

`"max"`, `"add"` and `"h2"` equal the CPU heuristics, including dead ends. `"ff"` and `"set_additive"` choose
equally cheap supporters by the lowest BFS level through tight h-max achievers, then the smallest operator id.
`Heuristic.reference(state)` implements that rule on the CPU. Set-additive keeps axiom supporters and unions all
their preconditions' achiever sets; each action contributes one `(operator, proposition)` member, weighted by its
cost, and axioms contribute no member. An action supporting several propositions can contribute several members.
The CPU heuristic picks supporters by its queue order, so its value can differ when a tie decides the result.
On the 21 fitting heuristic-suite tasks, four seeded walks of up to 15 steps in each fixed-slot and lazy configuration
produce 2630 unit-cost samples: 968 (36.8%) differ from the CPU set-additive value, with zero mismatches against
`reference()`. The remaining organic-synthesis task exceeds the grounding budget.

h² accepts at most **8191 propositions**, matching the CPU limit. Every resident state group has two triangular
global cost tables plus proposition and operator scratch: `4 * P * (P + 1) + 4 * P + 4 * O` bytes, rounded up to
16 bytes (at least 16), for `P` propositions and `O` operators. The global scratch budget defaults to **512 MiB**;
fewer blocks evaluate the batch when necessary, reusing their tables between rows. This bounds quadratic scratch
independently of batch size and leaves memory for uploaded tasks and search states. If one state cannot fit the
budget, or the proposition count exceeds 8191, construction raises `ValueError` naming the size and limit.
Set `max_scratch_bytes=` on `cuda.Heuristic` to choose a smaller budget; C++ searches use
`DeviceBestFirstOptions::heuristic.max_scratch_bytes`. h² always uses a block per state and global Jacobi sweeps.
States outside the grounding raise an error for h² and set-additive, as on the CPU, because they have no lifted
evaluation.

```python skip-if-no-cuda
from pathlib import Path

import mymyr
from mymyr import cuda

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
ctx = cuda.Context(0, max_bytes=1 << 30)
for kind in ("h2", "set_additive"):
    heuristic = cuda.Heuristic(task, kind, ctx=ctx, max_scratch_bytes=64 << 20)
    values = heuristic.evaluate([task.initial_state])
    assert values[0] == heuristic.reference(task.initial_state)
result = cuda.astar(task, heuristic="h2", ctx=ctx, batch=64)
assert result.solved
state = task.initial_state
for action in result.plan:
    state = task.apply(state, action)
assert task.is_goal(state)
ctx.synchronize()
```

## Encoded observations and resident rows

[Device encoding](encoding.md#encoding-on-the-device) reads CUDA state rows directly and returns torch arrays:
relations, Simple graphs, numeric values, actions, horizons and shared static blocks, subject to the
[support table](encoding.md#what-the-device-encodes). Exact calls download output counts. Capacity calls retain
counts and status on the device and support CUDA graph capture after warm-up. `batch.nbytes` and
`batch.nbytes_by_array` report capacity payload bytes from host metadata without synchronizing.

Choose the path by where rows arrive:

- Already batched CUDA rows: `encoder.encode(rows, task_ids, capacities=...)` or the
  [torch custom op](encoding.md#the-torch-op) inside `torch.compile`.
- Rows accumulated over time: a [device row ring](encoding.md#device-row-streams-and-collation); prepare a
  `DeviceCollator` for repeated or captured collation.
- Several domains or resident training data: [`SuiteEncoder` and `EncodingCollate`](datasets.md#training-data-encoded-on-load)
  group global task ids on the device, returning a batch and local ids per domain. Capacity calls keep the grouping
  resident, including empty domains.
- Policy observations: [`BatchedEnv`](rl.md#encoded-observations) encodes states after autoreset and, on request,
  final states before reset. Capacity action observations can be captured with the step when expansion is asynchronous.
- Jitted JAX code: the [XLA FFI encoder](encoding.md#encoding-inside-jax) uses XLA's stream and fixed capacity shapes;
  its supported forms are narrower than the torch path.

Torch operations use the current stream. Order cross-stream producers and consumers with events or `wait_stream`.
Create rings, collators and encoder contexts before capture; keep the required owners through completion of the last
replay. See the [capture contract](encoding.md#streams-and-cuda-graphs) for lifetimes and operations allowed during
capture, and [encoded observations](rl.md#encoded-observations) for action expansion requirements.
