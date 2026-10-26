# Getting started

mymyr is a lifted-first planning library with Python bindings. Python wheels support GIL and free-threaded CPython
3.14 and 3.15 on Linux x86_64 (glibc 2.34 or newer) and macOS arm64 (macOS 15 or newer). The wheels on PyPI contain
the CPU backend. The CUDA backend comes in wheels for free-threaded CPython on Linux, attached to the GitHub Releases
after 0.1.0, or is built from source.

## Install a wheel

Create a free-threaded environment and install from PyPI:

```bash
uv venv .venv --python 3.14t
uv pip install --python .venv/bin/python mymyr
```

Change the Python version passed to `uv venv` to `3.15t`, or to `3.14` or `3.15` for a GIL interpreter.

For the CUDA backend, install a release's CUDA wheel (version `<version>+cu13`, free-threaded CPython 3.14 and 3.15
on Linux x86_64, an NVIDIA driver 580 or newer) from the release's assets instead, with `<version>` the release's
version; releases after 0.1.0 have them:

```bash
uv pip install --python .venv/bin/python "mymyr==<version>+cu13" \
    --find-links https://github.com/maichmueller/mymyr/releases/expanded_assets/v<version>
```

[Installing a CUDA wheel](cuda.md#installing-a-cuda-wheel) lists the GPUs it supports and how to build one for others.

## Build from source

Building the PDDL-enabled package needs a C++26 compiler (GCC 16 or newer), CMake 3.25 or newer, Ninja, and CPython
3.14 or newer. A free-threaded interpreter (3.14t, 3.15t) runs the Python callbacks of parallel searches in parallel;
the GIL build works too, with those callbacks taking turns on the GIL. The dependency superbuild installs loki, abseil,
fmt and Boost headers; the main build links them into the front-end archive.

```bash
cmake -S dependencies -B dependencies/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build dependencies/build
uv venv .venv --python 3.14t
CMAKE_ARGS="-DMYMYR_DEPS_PREFIX=$PWD/dependencies/installs" uv pip install --python .venv/bin/python -e .
```

To build the optional CUDA backend, add `-DMYMYR_CUDA=ON` and `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` (CUDA 13.4 or
newer, with CMake 4.4 or newer) to `CMAKE_ARGS`. See the [CUDA page](cuda.md) for its current limits.

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
when a parser is not needed. The next pages show task inspection, search options, heuristics and batched interfaces;
every module names its options and result fields the same way ([API conventions](conventions.md)).
`search.validate_plan(task, result.plan)` checks a plan and computes its cost, and `task.to_pddl()` writes the
normalized task back as PDDL.

The same search runs from the command line: `mymyr plan tests/data/pddl/counters/domain.pddl
tests/data/pddl/counters/p01.pddl` prints the plan (see [Command line](cli.md)).
