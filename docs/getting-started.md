# Getting started

mymyr is a lifted-first planning library with Python bindings. Python wheels target free-threaded CPython 3.14t and
3.15t on Linux x86_64 (glibc 2.34 or newer) and macOS arm64. The published wheels contain the CPU backend. A CUDA
backend is built from source.

## Install a wheel

Create a free-threaded environment and install from PyPI:

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python mymyr
```

Use a 3.15t interpreter instead by changing the Python version passed to `uv venv`.

## Build from source

Building the PDDL-enabled package needs a C++26 compiler (GCC 16 or newer), CMake 3.25 or newer, Ninja, and a free-
threaded CPython 3.14t or 3.15t interpreter. The dependency superbuild installs loki, abseil, fmt and Boost headers;
the main build links them into the front-end archive.

```bash
cmake -S dependencies -B dependencies/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build dependencies/build
uv venv .venv --python 3.14t
CMAKE_ARGS="-DMYMYR_DEPS_PREFIX=$PWD/dependencies/installs" uv pip install --python .venv/bin/python -e .
```

To build the optional CUDA backend, add `-DMYMYR_CUDA=ON` and `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` to `CMAKE_ARGS`.
See the [CUDA page](cuda.md) for its current limits.

## Parse a task and search it

This example uses the small numeric counters task included in the repository. Run it from the repository root; replace
the paths with your own PDDL files to plan another task.

```python
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
result = search.astar(task, heuristic="max")

assert result.status == search.Status.SOLVED
print(search.format_plan(task, result.plan))
```

`Task.from_pddl` is available in builds with the PDDL front end. `Task.from_text` reads mymyr's normalized task format
when a parser is not needed. The next pages show task inspection, search options, heuristics and batched interfaces.
