# C API

`include/mymyr/ext.h` is a plain-C header for Python extension modules. A consumer includes the header and imports
`mymyr._core._C_API` at runtime; it does not link `libmymyr` or share nanobind internals. The function table has ABI
version 1 and minor version 2 in this revision.

The table is append-only. Consumers built against a newer header check `MYMYR_API_HAS(api, field)` before using newer
entries; an incompatible ABI version fails during `mymyr_import_api()`. Views are borrowed from the Python objects
they describe, so keep those objects alive while using their pointers. Calls over a task can run without the Python
thread state attached; stores are single-threaded and should be local to a worker.

Minor 0 provides version, state hashing/equality and state views. Minor 1 adds tasks, atom-word successors and apply,
goal tests, atom lookup and stores. Minor 2 adds numeric slot/value access, metric values, and state-aware successors,
apply, goal tests, hashing/equality and stores. The minor-1 atom-word functions reject numeric tasks; use their
`*_state` counterparts for numeric values. The metric passed to a successor call is the path cost `g`, and the
callback receives the updated metric.

The checked-in [C consumer](../tests/ext_consumer/ext_consumer.c) begins with “A downstream extension written against
`mymyr/ext.h` only”: it imports the table, calls it from an extension module, and tests the numeric-aware BrFS path.
The [Python driver](../tests/python/test_ext_api.py) builds and exercises that consumer, including minor-1 and minor-2
table access.

Run the consumer tests with the Python suite:

```bash
.venv/bin/python -m pytest -W error tests/python/test_ext_api.py
```
