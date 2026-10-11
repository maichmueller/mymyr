# Native relational encoding

`mymyr.encoding` produces classical planning observations as packed arrays. It runs in C++, consumes native state
words directly, and needs neither mifrost nor a model framework. It is a first native flat contract informed by
mifrost's relational semantics, not a replacement for every mifrost encoder family.

```python
import numpy as np
import mymyr
from mymyr import encoding, rl

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt", atoms="lazy")
states = [task.initial_state]
expansion = rl.expand(task, states)
encoder = encoding.FlatRelationEncoder(task)
batch = encoder.encode(states, actions=expansion)
assert batch.num_graphs == 1
assert np.array_equal(batch.action_schema, expansion.schema)
assert np.array_equal(batch.action_parent, expansion.parent)
assert len(batch.action_tuple) == expansion.total
assert not batch.relation_args.flags.writeable
```

## Schema and tuple layout

The immutable schema is exposed through `encoder.schema` and retained in `batch.schema`. Its version is 1.
`relation_names`, `relation_arities` and `relation_sources` describe a fixed vocabulary for that encoder's table.
Every table predicate has five lanes: its positive state facts, positive satisfied/unsatisfied goals, and negative
satisfied/unsatisfied goals. A signed goal emits exactly one status tuple, including negative, static and derived
literals. State names use the predicate name; goal names use the mifrost convention, e.g. `[+]on[g][sat]` and
`[-]clear[g][unsat]`. Action relation names end in `[action]`.

`relation_predicates`, `relation_schemas`, `relation_kinds`, `relation_instances` and `relation_polarities` retain the
structured identity alongside the names. Predicate kinds are static=0, fluent=1, derived=2; absent IDs/kinds/polarity
are -1. Domain predicate IDs are the table's stable prefix. Problem-local predicates remain separate, with an
instance-qualified name and explicit `relation_instances`; they are not a shared cross-problem learned vocabulary.
Action schema IDs retain normalization-introduced parameters and distinct schemas even when their source names agree.
Names are descriptive labels, not unique planning identities; use the structured IDs for model/readout keys.

For N graphs and R relations, all indices/counts are int32:

| Field | Shape | Meaning |
|---|---|---|
| `relation_counts` | `[N,R]` | Tuple counts, including nullary facts/goals/actions |
| `relation_offsets` | `[N*R+1]` | CSR argument offsets in graph-major, then relation-major order |
| `relation_tuple_offsets` | `[N*R+1]` | CSR tuple offsets, including tuples with no arguments |
| `relation_args` | `[A]` | Packed batch-global object IDs |
| `object_offsets` | `[N+1]` | Exclusive object counts of each graph |
| `task_ids` | `[N]` | Original instance ID of each input row |

Cell `graph * R + relation` has `relation_counts[graph,relation]` tuples of that relation's arity, encoded in
`relation_args[relation_offsets[cell]:relation_offsets[cell+1]]`. Tuples are sorted lexicographically by object
binding within each cell, so lazy slot allocation order never becomes model input. Zero-arity predicates are not
dropped: they carry a count even though their argument segment is empty. Counts/offsets beyond int32 capacity fail
explicitly rather than wrapping or truncating.
Repeated initial facts have set semantics, while repeated caller action labels remain distinct readout rows.

`batch.object_names` and `batch.task_fingerprints` retain portable name/content metadata. Batch-global object IDs
are not global planning identities: subtract `object_offsets[graph]` to recover the instance-local object ID.
The result owns its schema, metadata and packed arrays and remains valid after the encoder, task or inputs are gone.

## Multi-instance batches and goals

Construct an encoder from one `Task`/`TaskHandle`, or a one-domain `rl.TaskTable`. A multi-instance table requires an
aligned `task_ids` vector. Native State and GroundCondition values are checked against their row's task ownership;
raw word-array callers remain responsible for supplying the matching task/slot encoding. Invalid or unassigned bits
are rejected. The encoder reads the task's current atom index directly rather than retaining stale lazy metadata.

Default state facts include static and axiom-derived atoms; `include_static=False` or `include_derived=False` chooses
an explicit projection. Derived goal satisfaction is evaluated correctly even when derived state facts are omitted.
Each encode uses its own WorkspaceLease. Shared encoders support concurrent calls; the caller must not concurrently
overwrite input array rows while a call reads them.

`goals=None` uses each instance's default normalized goal. For a single-task encoder, `goals=task.ground_condition(...)`
overrides the goal for every row. A list of native GroundConditions/None supplies one per row, including table batches;
None entries use that instance's default. An empty native GroundCondition is an explicit empty goal. This accepts
conjunctive classical literals, not arbitrary Python goal strings or numeric constraints.

## Action readouts

`actions=expansion` reuses an existing, non-overflowing `rl.Expansion` over the same table, input row order and
`task_ids`. The encoder does not enumerate actions or generate successors. The following arrays preserve every
original expansion action and its ordering:

The expansion's owned `task_ids` list records every source row, including rows without actions. The encoder checks
those IDs against its input. Matching instance IDs do not prove that two raw rows describe the same state;
the caller must still supply the source states in their original order.

| Field | Shape | Meaning |
|---|---|---|
| `action_offsets` | `[N+1]` | Original action CSR offsets |
| `action_schema` | `[M]` | Original schema IDs |
| `action_binding` | `[M,L]` | Instance-local object IDs, -1 after the schema arity |
| `action_parent` | `[M]` | Original parent graph index |
| `action_tuple` | `[M]` | Each original action's unique packed tuple index |

Canonical tuple sorting does not reorder action readouts. Index `relation_tuple_offsets` to locate an action's
graph/relation cell; then its local tuple number and arity locate its arguments. Omitting actions yields empty label
arrays and zero action counts, not an implicit applicability query. Capacity-truncated expansions are rejected.

## Frameworks and current limits

The output framework follows the input arrays, or uses NumPy for State values/lists. `framework="numpy"`, `"torch"`,
`"jax"` or `"dlpack"` overrides it. `batch.as_dict(framework=...)` exports the same owned storage using existing
mymyr DLPack support; framework packages are imported only when requested. NumPy outputs are read-only. Do not
mutate the shared buffers through frameworks whose tensor APIs cannot enforce that flag. This ragged CPU encoder
is not a JAX-jitted, CUDA or CUDA-graph-capturable operation; GPU inputs are not silently copied back to the CPU.

Numeric functions/constraints, including tasks declaring action-cost functions, are explicitly unsupported. Their
values must not disappear into an apparently complete Boolean observation. TaskSuite/cross-domain batches, history,
subgoal layers, lookahead DAGs, custom graph projections, automatic PyG/jraph conversion and WL features are not in
this version. See [the integration assessment](native-encoding-design.md) for the staged extensions and source strategy.

The C++ API is in `mymyr/encoding/flat.hpp`, linked with `mymyr::encoding`. It accepts caller-owned StateBatchView,
TaskTable instance IDs, ActionBatchView label columns and optional per-row GroundConditions. Output storage uses
the existing immutable ArrayBundle. No planning-backend ABI/type registry is imported.
