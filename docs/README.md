# mymyr documentation

These pages describe the Python package, optional CUDA backend, plain-C API and C++ package.

- [Getting started](getting-started.md): install a wheel or build from source, parse a task and run a search.
- [Parsing and tasks](parsing-and-tasks.md): PDDL, task and state values, formalism views, actions and binding generators.
- [Formula values](formulas.md): atoms, literals, conditions, state checks, pickling and search goals.
- [Search](search.md): breadth-first search, the IW family, best-first search, heuristics, goals, budgets and observers.
- [Heuristics](heuristics.md): relaxation heuristics, h², set-additive and perfect heuristics.
- [Datasets](datasets.md): state spaces, generalized state spaces, samplers and certificates.
- [Reinforcement learning](rl.md): batched expansion, CPU environments and JAX/PyTorch interop.
- [CUDA](cuda.md): device requirements, supported operations and current task limits.
- [C API](c-api.md): the versioned `mymyr/ext.h` function table and a C consumer.
- [C++ package](cpp-package.md): install layout, CMake targets and a downstream consumer.

The executable Python scripts are in [`examples/`](../examples/). They use PDDL fixtures from `tests/data/` so they can
run from the repository root.

## Running the documentation examples

`tests/python/test_docs.py` extracts each fenced `python` block and runs a page's blocks in order in one namespace. A
block marked `python skip-if-no-cuda` needs a CUDA-enabled mymyr build and a visible CUDA device. These blocks are
skipped in CPU-only runs. Keep snippets short and use repository fixtures when a task file is needed.

`tests/python/test_examples.py` runs every `examples/*.py` script in a subprocess. Its optional framework/device
requirements are declared in the test's small example table so CPU-only CI can skip them explicitly.
