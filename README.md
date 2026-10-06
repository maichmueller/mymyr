# mymyr

[![CI](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml)
[![Wheels](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/mymyr)](https://pypi.org/project/mymyr/)

A data-oriented, lifted-first rewrite of the [mimir](https://github.com/maichmueller/mimir) classical-planning
core, with free-threaded Python bindings and an optional CUDA backend.

- Flat POD data with no global repositories: states are values, and containers are local and droppable.
- Free-threaded CPython (3.14t and 3.15t) is the only Python target.
- A CUDA backend and RL/JAX/PyTorch interop are core features.

## Installing from PyPI

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python mymyr
```

Releases carry CPU wheels for free-threaded CPython 3.14t and 3.15t on Linux (x86_64, glibc 2.34 or newer) and
macOS (arm64). The CUDA backend is not part of them; build from source with `-DMYMYR_CUDA=ON` for it. The badges above
show the state of the continuous integration (Linux build, tests and sanitizers; CUDA compile check; macOS) and of the
wheel builds.

## Requirements

- A C++26 compiler (GCC >= 16)
- CMake >= 3.25 and Ninja
- Optional: CUDA 13 (`-DMYMYR_CUDA=ON`)
- Optional: CPython 3.14t or 3.15t, free-threaded, for the Python bindings

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

Add `-DMYMYR_CUDA=ON` (and, if needed, `-DCMAKE_CUDA_COMPILER=<path to nvcc>`) for the CUDA backend and its tests.
Sanitizers: `-DMYMYR_SANITIZE="address;undefined"` or `-DMYMYR_SANITIZE=thread`.

A handful of tests compare mymyr against the mimir fork's own PDDL instances and recorded behavior
(`github.com/maichmueller/mimir`, its `data/` directory). Point `MYMYR_FORK_DATA` (and, for the IPC-2023 tasks,
`MYMYR_IPC`) at a checkout of that data to run them; without it, they skip rather than fail.

## Python

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest
```

(or the equivalent with plain `python -m venv` and `pip`, given a free-threaded 3.14t or 3.15t interpreter). Pass CMake
options through `CMAKE_ARGS`, e.g. `CMAKE_ARGS="-DMYMYR_CUDA=ON" uv pip install ...` for the CUDA backend, or
`-DMYMYR_DEPS_PREFIX=<prefix>` to reuse an existing dependency install.

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

## License

mymyr is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version
(SPDX: `GPL-3.0-or-later`; the full text is in [LICENSE](LICENSE)). These are the terms of mimir
(Copyright (C) 2023 Dominik Drexler and Simon Stahlberg), from which mymyr is ported and derived.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
