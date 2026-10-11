# Native relational encoding

`mymyr.encoding` produces classical planning observations as packed arrays. It runs in C++, consumes native state
words directly, and needs neither mifrost nor a model framework. It is a first native flat contract informed by
mifrost's relational semantics, not a replacement for every mifrost encoder family.

```python
import numpy as np
import mymyr
from mymyr import encoding, rl

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
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
binding within each cell, so the atom slot layout (`atoms='auto'`, `'reachable'` or `'dense'`) never becomes model
input. Zero-arity predicates are not dropped: they carry a count even though their argument segment is empty.
Counts/offsets beyond int32 capacity fail explicitly rather than wrapping or truncating.
Repeated initial facts have set semantics, while repeated caller action labels remain distinct readout rows.

`batch.object_names` and `batch.task_fingerprints` retain portable name/content metadata. Batch-global object IDs
are not global planning identities: subtract `object_offsets[graph]` to recover the instance-local object ID.
The result owns its schema, metadata and packed arrays and remains valid after the encoder, task or inputs are gone.

## Multi-instance batches and goals

Construct an encoder from one `Task`/`TaskHandle`, or a one-domain `rl.TaskTable`. A multi-instance table requires an
aligned `task_ids` vector. Native State and GroundCondition values are checked against their row's task ownership;
raw word-array callers remain responsible for supplying the matching task/slot encoding. Invalid or unassigned bits
are rejected. A row's words past its instance's own are padding and are not read; a row narrower than its instance's
states raises `ValueError`. Tasks with lazy atom slots are refused, since their rows have no fixed layout
([State encodings](parsing-and-tasks.md#state-encodings)).

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
`task_ids`. The encoder does not enumerate actions or generate successors. The expansion's `task_ids` list records
the instance of every source row, including rows without actions, and the encoder checks it against its input. Matching
instance IDs do not prove that two raw rows describe the same state: the caller supplies the source states in their
original order. The following arrays preserve every original expansion action and its ordering:

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
this version.

The C++ API is in `mymyr/encoding/flat.hpp`, linked with `mymyr::encoding`. It accepts caller-owned StateBatchView,
TaskTable instance IDs, ActionBatchView label columns and optional per-row GroundConditions. Output storage uses
the existing immutable ArrayBundle. No planning-backend ABI/type registry is imported.

## Encoding schema

`mymyr.encoding.EncodingSchema(table_or_task, **options)` (C++: `mymyr/encoding/schema.hpp`) is the relation catalog of
an encoding: which relations a batch holds, with their arities, lanes and slot roles. It is a function of the domain
and the options, never of the instances of a table, so tables over one domain give the same relation ids to the same
relations.

- **Names and order.** Relations are named `[+]on[g][sat]`: a sign for goal and history relations, the predicate or
  action schema name, `[hist]`, a goal level (`[g]`, `[sg]`, `[ssg]`, `[sssg]`) and a derivation (`[sat]`, `[unsat]`,
  `[sat+]`, `[sat-]`). State relations carry the plain name. The catalog is sorted by name, bytewise; a relation's id
  is its position. `relation_lanes`, `relation_signs`, `relation_levels`, `relation_derivations`,
  `relation_predicates` and `relation_schemas` describe each relation, `slot_roles` each position of its tuples.
- **What is listed.** Every domain predicate and action schema, the helper predicates of quantified conditions
  (`axiom_<k>`) included; the predicates only a problem's axioms define are not. Goal relations exist for every
  predicate except the types `object` and `number`. The equality predicate `=` is left out unless `equality=True`;
  the state relations of the static predicates (the types included) and of the derived predicates can be dropped
  (`include_static`, `include_derived`); their goal and history relations stay. Nullary predicates (`nullary=True`,
  the default) have relations of arity 1: their atoms are tuples over one auxiliary entity per graph (slot role
  `auxiliary_slot`, entity role `auxiliary`); with `nullary=False` they have no relations. `slot_role_names` and
  `entity_role_names` always list all nine roles.
  An action relation holds the schema's full parameter list (the written parameters and the ones normalization adds
  for existential variables). Schemas of one name and arity (a disjunctive precondition splits an action) share one
  relation; a name with several arities has one relation `name#arity` per arity.
- **Invalid combinations** raise `ValueError`: `max_goal_level` outside 0 to 3; subgoal targets without
  `max_goal_level >= 1`; action targets without `include_actions`; goal or subgoal targets without the `plain` goal
  derivation (only the plain goal relations carry the target slot); added derivations without the delta horizon mode;
  targets together with a horizon mode; the action horizon mode without `include_actions`; `parent_relation` or a
  `root_policy` other than `include` without a horizon mode.
- **Fingerprint.** `schema_fingerprint` hashes the catalog (names, lanes, arities, slot roles) and the options that
  shape the output but not the catalog (target sources, layout, index type, horizon mode and root policy). It does not
  depend on declaration order, on predicate ids, on the process or on the build.
