# mymyr

[![CI](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/ci.yml)
[![Wheels](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml/badge.svg)](https://github.com/maichmueller/mymyr/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/mymyr)](https://pypi.org/project/mymyr/)

A data-oriented, lifted-first rewrite of the [mimir](https://github.com/maichmueller/mimir) classical-planning
core, with free-threaded Python bindings and an optional CUDA backend.

mymyr parses PDDL, searches for plans, builds state spaces and provides batched reinforcement-learning environments
and relational state encodings for NumPy, JAX and PyTorch. Native work releases the GIL. Free-threaded Python also
lets Python callbacks and concurrent Python callers run in parallel; on a GIL interpreter, Python callbacks take
turns on the GIL.

## Install from PyPI

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python mymyr
```

CPU wheels support CPython 3.14 and 3.15, both GIL (`3.14`, `3.15`) and free-threaded (`3.14t`, `3.15t`), on Linux
x86_64 with glibc 2.34 or newer and macOS arm64 with macOS 15 or newer. Change `--python` to select an interpreter.
Wheels include the PDDL parser and C++ SDK. For Gymnasium environments, install `mymyr[gym]`; install JAX or PyTorch
separately when using those integrations.

Releases after 0.1.0 attach CUDA wheels for free-threaded CPython 3.14 and 3.15 on Linux x86_64 to their
[GitHub Release](https://github.com/maichmueller/mymyr/releases), with the version `<version>+cu13` (v0.1.0 has CPU
wheels only). They need an NVIDIA driver 580 or newer and carry GPU code for compute capabilities 8.6, 8.9 and 9.0.
With `<version>` the release's version:

```bash
uv pip install --python .venv/bin/python "mymyr==<version>+cu13" \
    --find-links https://github.com/maichmueller/mymyr/releases/expanded_assets/v<version>
```

See [Installing a CUDA wheel](docs/cuda.md#installing-a-cuda-wheel) for other GPUs and source builds.

Pushing a `v*` tag that matches the package version publishes to TestPyPI and PyPI; a manual release workflow run is
a TestPyPI rehearsal with a unique `.devN` version. The CUDA wheels of a release are built on a large Linux host
with `ci/local_cuda_wheels.sh` (a manual run of the CUDA wheel workflow builds them where there is none), and
`ci/attach_cuda_wheels.sh` tests them on a GPU and attaches them to the GitHub Release.

## Try it

Run this with `.venv/bin/python`; it needs no task files:

```python
import mymyr
from mymyr import rl, search

domain = mymyr.Domain.from_string("""
(define (domain switch)
  (:requirements :strips)
  (:predicates (ready) (done))
  (:action finish
    :parameters ()
    :precondition (ready)
    :effect (and (not (ready)) (done))))
""")
problem = domain.instantiate_string("""
(define (problem example) (:domain switch)
  (:init (ready))
  (:goal (done)))
""")
task = mymyr.Task(problem)
result = search.astar(task, heuristic="max")
assert result.solved and len(result.plan) == 1 and result.cost == 1
print(search.format_plan(task, result.plan))

pool = rl.CpuEnvPool(task, num_envs=4, seed=0)
pool.reset()
batch = pool.step()
assert batch.terminated.all()
print(batch.states.shape, batch.reward, batch.terminated)
```

For your own task files, use `mymyr.Task.from_pddl("domain.pddl", "problem.pddl")`.

## Build from source

Source builds need CPython 3.14 or newer, a C++26 compiler, CMake 3.25 or newer and Ninja. The commands below run
from a source checkout on Linux x86_64 with uv, Git, curl and sha256sum on `PATH`. The toolchain script downloads
GCC 16.1 from conda-forge using micromamba and builds the pinned dependencies (loki, Abseil, fmt, Boost
headers and googletest for the C++ tests) under `build/toolchain`; no system compiler installation is needed.

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python "cmake>=3.25" ninja
export PATH="$PWD/.venv/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=8
bash ci/wheel_toolchain.sh "$PWD/build/toolchain"
export CC="$PWD/build/toolchain/gcc/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$PWD/build/toolchain/gcc/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_ARGS="-DMYMYR_DEPS_PREFIX=$PWD/build/toolchain/deps"
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -W error tests/python
```

An existing GCC 16 build also works: set `CC` and `CXX` to its `gcc` and `g++`, build the
[dependencies](dependencies/CMakeLists.txt) with that compiler, and set `MYMYR_DEPS_PREFIX` through `CMAKE_ARGS` to
their install prefix. `-DMYMYR_FRONTEND=OFF` skips the parser and its dependencies. On macOS use a C++26-capable
Apple Clang/libc++ toolchain; see the [C++ package guide](docs/cpp-package.md) for native builds and SDK consumers.

For CUDA, follow the existing-GCC route above with a system or source-built GCC 16 and dependencies built with
that compiler. CI uses a system GCC 16 as nvcc's host compiler. With `CC` and `CXX` selecting that compiler and
`CMAKE_ARGS` pointing to its dependency prefix, put CUDA 13.4 or newer's `nvcc` on `PATH`, then rebuild with CMake
4.4 or newer:

```bash
uv pip install --python .venv/bin/python "cmake>=4.4"
CMAKE_ARGS="$CMAKE_ARGS -DMYMYR_CUDA=ON -DCMAKE_CUDA_COMPILER=$(command -v nvcc) -DCMAKE_CUDA_HOST_COMPILER=$CXX" \
  uv pip install --python .venv/bin/python --reinstall-package mymyr -e ".[test]"
```

The default CUDA architecture is the visible GPU. NVIDIA drivers 580 or newer can run matching SASS; PTX needs a
driver that understands the toolkit's PTX version. See [CUDA](docs/cuda.md) for architecture selection and backend
limits. Reference-comparison tests need the separate data described in [tests/parity/README.md](tests/parity/README.md).

## Documentation

Start with the [user guides](docs/README.md), [API conventions](docs/conventions.md) and runnable
[examples](examples/). See [parsing and task options](docs/parsing-and-tasks.md), [search](docs/search.md),
[RL and Gymnasium](docs/rl.md), [relational encodings](docs/encoding.md), [the CLI](docs/cli.md),
[the C API](docs/c-api.md) and [the C++ SDK](docs/cpp-package.md).

## License

mymyr is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version
(SPDX: `GPL-3.0-or-later`; the full text is in [LICENSE](LICENSE)). These are the terms of mimir (Copyright (C) 2023 Dominik Drexler and Simon Stahlberg), from which mymyr is derived.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
