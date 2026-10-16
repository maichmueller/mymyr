# mymyr

[![CI](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml)
[![Wheels](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/mymyr)](https://pypi.org/project/mymyr/)

A data-oriented, lifted-first rewrite of the [mimir](https://github.com/maichmueller/mimir) classical-planning
core, with free-threaded Python bindings and an optional CUDA backend.

- Flat POD data with no global repositories: states are values, and containers are local and droppable.
- Python bindings for CPython 3.14 and newer. Free-threaded CPython (3.14t, 3.15t) is the main target: there
  Python callbacks (observers, heuristics, goal tests) of parallel searches and concurrent Python callers run in
  parallel. The bindings also build and run on the GIL build of CPython, where these Python parts take turns on
  the GIL; the native work (searches, state spaces, environment pools, encoders) releases it either way.
- A CUDA backend with classical and numeric tasks, and RL/JAX/PyTorch/Gymnasium interop are core features.

## Installing from PyPI

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python mymyr
```

The extra `mymyr[gym]` adds Gymnasium for the environments of `mymyr.rl.gym` ([Gymnasium](docs/rl.md#gymnasium)).

Releases carry CPU wheels for free-threaded CPython 3.14t and 3.15t on Linux (x86_64, glibc 2.34 or newer) and
macOS (arm64). The CUDA backend is not part of them; build from source with `-DMYMYR_CUDA=ON` for it. The badges above
show the state of the continuous integration (Linux build, tests and sanitizers; CUDA compile check; macOS) and of the
wheel builds.

## Requirements

- A C++26 compiler (GCC >= 16)
- CMake >= 3.25 and Ninja
- Optional: CUDA 13.4 or newer (`-DMYMYR_CUDA=ON`; its device code is C++23, with GCC 16 as nvcc's host compiler) and CMake >= 4.4 (the first to map CUDA C++23 to nvcc), on a driver >= 580
- Optional: CPython 3.14 or newer for the Python bindings (free-threaded 3.14t or 3.15t recommended)

## Building the dependencies

The PDDL front end (`-DMYMYR_FRONTEND=ON`, the default) needs loki, abseil, fmt and boost, pinned and built by the
dependency superbuild:

```bash
cmake -S dependencies -B dependencies/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build dependencies/build
```

This installs into `dependencies/installs`, which the main build picks up automatically. Set `CC`/`CXX` in the
environment of both commands to choose a compiler; point `-DMYMYR_DEPS_PREFIX=<prefix>` at an existing install to
reuse one built elsewhere, or configure the main build with `-DMYMYR_FRONTEND=OFF` to skip the front end (and this
step) entirely.

## Building and testing the library

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Add `-DMYMYR_CUDA=ON` (and, if needed, `-DCMAKE_CUDA_COMPILER=<path to nvcc>`) for the CUDA backend and its tests; it
needs CUDA 13.4 or newer and CMake 4.4 or newer, and the binaries run on NVIDIA drivers 580 or newer
(`-DCMAKE_CUDA_ARCHITECTURES` defaults to the visible GPU; see [CUDA](docs/cuda.md)).
Sanitizers: `-DMYMYR_SANITIZE="address;undefined"` or `-DMYMYR_SANITIZE=thread`.

The tests that compare mymyr with recorded reference results live in `tests/parity` (ctest label `parity`, pytest
marker `parity`); they skip when the reference data is not available. See [tests/parity/README.md](tests/parity/README.md).

## Using mymyr from C++

`cmake --install build --prefix <prefix>` installs the headers, the static libraries and a CMake package; the wheel
carries the same under `mymyr/include` and `mymyr/lib`. A consumer needs only that install and a C++26 compiler (GCC 16,
or clang 22 with libc++): the front end's parser dependencies are part of `libmymyr_frontend.a`.

```cmake
find_package(mymyr CONFIG REQUIRED COMPONENTS core frontend)   # components: core, encoding, frontend, cuda
target_link_libraries(app PRIVATE mymyr::core mymyr::frontend)
```

Pass `-Dmymyr_DIR=<prefix>/lib/cmake/mymyr`, or for the wheel `-Dmymyr_DIR=$(python -c "import mymyr; print(mymyr.get_cmake_dir())")`
(`mymyr.get_include()` is the header directory). The C++ package is a source-level interface: rebuild consumers when
mymyr changes, and use the C++ standard library the install was built with. The plain-C API (`mymyr/ext.h`) is the
stable binary interface. `tests/cmake_consumer/` is a complete example.

## Python

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest
```

(or the equivalent with plain `python -m venv` and `pip`, given a CPython 3.14 or newer interpreter, free-threaded
preferably). Pass CMake options through `CMAKE_ARGS`, e.g. `CMAKE_ARGS="-DMYMYR_CUDA=ON" uv pip install ...` for the
CUDA backend, or `-DMYMYR_DEPS_PREFIX=<prefix>` to reuse an existing dependency install.

### Example

```python
import mymyr
import mymyr.rl as rl
from mymyr import search

task = mymyr.Task.from_pddl("domain.pddl", "problem.pddl")
plan = search.astar(task, heuristic="max")
print(plan.status, len(plan.plan), plan.cost)

pool = rl.CpuEnvPool(task, num_envs=4, seed=0)
batch = pool.reset()
batch = pool.step()
print(batch.states.shape, batch.reward, batch.terminated)
```

User guides and tested scripts are in [docs/](docs/README.md) and [examples/](examples/); the option names, defaults
and result fields every module shares are in [API conventions](docs/conventions.md).

`mymyr.encoding.FlatRelationEncoder(task_or_table)` encodes states into relational graphs: the
objects are the entities, and the fluent, derived, static and goal atoms are the tuples of the relations of an
`EncodingSchema`. The batches are flat integer arrays that NumPy, Torch, JAX and DLPack import without a copy, with no
PyG dependency. See [native encoding](docs/encoding.md) for the array contract and the options.

## License

mymyr is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version
(SPDX: `GPL-3.0-or-later`; the full text is in [LICENSE](LICENSE)). These are the terms of mimir (Copyright (C) 2023 Dominik Drexler and Simon Stahlberg), from which mymyr is derived.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
