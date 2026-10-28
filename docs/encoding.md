# Native relational encoding

`mymyr.encoding` encodes states of classical and numeric tasks into relational graphs: one graph per state, with the
task's objects as entities and, for every relation of an `EncodingSchema`, the tuples of the state's atoms (and, for
numeric tasks, of its numeric function terms with their values). It runs in C++, reads native state words directly, and
needs no model framework. The result is a batch of flat arrays that every framework imports without a copy. Options
follow the [API conventions](conventions.md).

A graph holds only what its state, its instance and the inputs of the call have: an entity per object of the instance,
a tuple per atom, the auxiliary entity and a predicate's virtual node only when a tuple of the graph uses them, and a
goal, action or history entity only for a goal literal, an action or a step that the graph has. No entity or tuple
stands for something absent, and no node is reserved for a catalog entry that the state does not fill, so the graphs
of a small instance and of a large one differ only in how much they hold. A model that is
trained on the graphs of small instances reads those of larger ones of the same domain unchanged: the relation ids are
a function of the domain and the options alone.

```python
import numpy as np
import mymyr
from mymyr import encoding

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
encoder = encoding.FlatRelationEncoder(task)
state = task.initial_state
batch = encoder.encode([state])
assert batch.num_graphs == 1

at = encoder.relation_id("at")
assert batch.relation_counts[0, at] == sum(str(atom).startswith("(at ") for atom in state.atoms())
assert not batch.relation_args.flags.writeable
assert np.array_equal(encoder.encode(state).relation_args, batch.relation_args)
```

The encoder is used by the other modules as well:

- [Reinforcement learning](rl.md#encoding-the-pools-states): the states of a CPU environment pool, mapping a chosen
  action entity back to the expansion row and to the environment's action, reusing a batch's memory with `into=`, and
  [the rows of a suite](rl.md#several-domains) as one batch per domain.
- [Datasets that encode on load](datasets.md#training-data-encoded-on-load) with a collate function, and the storage of
  [encoders, schemas and batches](datasets.md#encoders-schemas-and-batches).
- [State spaces and search frontiers](search.md#encoding-the-states-of-a-search) as encoder input, and horizons built
  from a state space's arrays.

## Encoders and options

Every encoder is built from a `Task`, a `TaskHandle` or a one-domain `rl.TaskTable` and keyword options:

| Class | Encodes | Options |
|---|---|---|
| `FlatRelationEncoder` | one graph per state | all but the horizon options |
| `SimpleEncoder` | one [Simple graph](#the-simple-graph) per state | those of `FlatRelationEncoder` but `outputs` (fixed to `('simple',)`) |
| `FlatHorizonEncoder` | one graph per [horizon](#horizons-and-transitions) | all but `history` and `history_max_steps`; `horizon_mode` is required |
| `FlatRootedHorizonEncoder` | a horizon seen from its root | those of `FlatHorizonEncoder`, with other defaults |
| `FlatTransitionEncoder`, `FlatTransitionEffectsEncoder` | the [transitions](#transitions) of states | all but the history and the horizon mode (fixed), `sibling_relation` and `cousin_relation` |
| `EncodingSchema` | nothing: the [relation catalog](#the-relation-catalog) | all; none is required |

| Option | Values | Default | What it changes |
|---|---|---|---|
| `goal_derivations` | a collection of `'plain'`, `'satisfied'`, `'unsatisfied'`, `'added_satisfied'`, `'added_unsatisfied'` | `('plain', 'satisfied')` | which [goal relations](#goals) a predicate has |
| `max_goal_level` | 0 to 3 | 0 | the [subgoal](#subgoals) levels beyond the goal |
| `repeated_literals` | `"every"`, `"lowest"`, `"highest"` | `"every"` | placement of literals listed at several goal levels |
| `include_static` | bool | `True` | the state relations of the static predicates (the types included) |
| `include_derived` | bool | `True` | the state relations of the derived predicates |
| `include_numeric` | bool | `True` | the [numeric](#numeric-functions) relations and `numeric_values` |
| `nullary` | `'auxiliary'`, `'arity0'`, `'omit'` | `'auxiliary'` | the relations of nullary predicates: tuples over the [auxiliary entity](#lanes), tuples without an argument, or none |
| `equality` | bool | `False` | the relations of the equality predicate `=` |
| `predicate_virtual_nodes` | bool | `False` | a [virtual node](#predicate-virtual-nodes) per predicate in its tuples |
| `include_actions` | bool | `True` | one relation per action schema, filled from `actions` ([Actions](#actions)) |
| `target_sources` | a collection of `'goal'`, `'subgoal'`, `'action'`, `'history'` | `()` | which entities the batch lists as targets |
| `history` | bool | `False` | the [history](#history) relations |
| `history_max_steps` | `None` or an int of at least 1 | `None` (all steps) | how many of a row's newest history steps are encoded |
| `layout` | `'graph'`, `'relation'` | `'graph'` | the order of the cells ([Layouts](#layouts-and-omitted-arrays)) |
| `index_dtype` | `'int32'`, `'int64'` | `'int32'` | the dtype of the index arrays |
| `numeric_dtype` | `'float32'`, `'float64'` | `'float32'` | the dtype of `numeric_values` |
| `emit_counts` | bool | `True` | whether the batch has `relation_counts` |
| `entity_arrays` | `'always'`, `'auto'` | `'always'` | `'auto'` omits `entity_role` and `entity_ref` when they carry no information |
| `statics` | `'repeat'`, `'shared'` | `'repeat'` | `'shared'` stores the static tuples once per instance ([Shared statics](#shared-statics)) |
| `outputs` | a collection of `'relations'`, `'simple'` | `('relations',)` | the relation arrays, the [Simple graph](#the-simple-graph) or both |
| `constant_labels` | bool | `False` | a Simple node label of its own for each constant of the domain |
| `horizon_mode` | `'full'`, `'delta'`, `'action'`, `None` | per encoder | the [horizon](#horizons-and-transitions) variant; `None` is no horizon |
| `root_policy` | `None`, `'include'`, `'encode_only'`, `'exclude'` | `None` without a horizon; `'include'` with one | whether a horizon's root is a state entity and a target |
| `parent_relation` | bool | `False` | the `_parent_` relation of a horizon's edges |
| `sibling_relation`, `cousin_relation` | bool | `False` | the [kinship](#rooted-horizons) relations of a horizon's nodes |

`FlatRootedHorizonEncoder` defaults to `horizon_mode='full'`, `root_policy='encode_only'` and `parent_relation=True`;
the transition encoders to `root_policy='exclude'`. For `EncodingSchema` without a horizon, `root_policy` reads `None`
(`None` and `'include'` are accepted, and `'encode_only'` and `'exclude'` raise `ValueError`). A value outside its choices
raises `ValueError` naming the allowed ones, an option a class does not take `TypeError`
(`FlatRelationEncoder(task, horizon_mode='full')`), and a `FlatHorizonEncoder` with `horizon_mode=None` raises
`ValueError`: a horizon encoder has no mode without a horizon.

**Invalid combinations** raise `ValueError` when the schema is made:

- subgoal targets without `max_goal_level >= 1`; action targets without `include_actions`; goal or subgoal targets
  without the `plain` goal derivation (only the plain goal relations carry the target slot); history targets or
  `history_max_steps` without `history`;
- added derivations without the delta horizon mode; targets together with a horizon mode (the state nodes of a horizon
  are its targets); `history` together with a horizon mode; the action horizon mode without `include_actions`;
  `parent_relation`, `sibling_relation`, `cousin_relation` or a `root_policy` other than `include` without a horizon
  mode; a horizon mode over a domain with fluent numeric functions, whatever `include_numeric` says;
- `statics='shared'` together with a horizon mode or with the `'simple'` output; no `outputs`; `constant_labels`
  without the `'simple'` output, or for a task without recorded constants; the `'simple'` output with
  `predicate_virtual_nodes` or `entity_arrays='auto'`; and, without the `'relations'` output, a `layout` other than
  `'graph'` or `emit_counts=False`.

An encoder has the properties of its schema (`schema`, `options`, `relation_names`, `relation_id(name)`,
`relation_lanes`, ...), its `table` and the `table_fingerprint` of that table.

### The relation catalog

`mymyr.encoding.EncodingSchema(table_or_task, **options)` is the relation catalog of an encoding: which relations a
batch holds, with their arities, lanes and slot roles. It is a function of the domain and the options, never of the
instances of a table, so tables over one domain give the same relation ids to the same relations.

- **Names and order.** Relations are named `[+]on[g][sat]`: a sign for goal and history relations, the predicate or
  action schema name, `[hist]`, a goal level (`[g]`, `[sg]`, `[ssg]`, `[sssg]`) and a derivation (`[sat]`, `[unsat]`,
  `[sat+]`, `[sat-]`). State relations carry the plain name, numeric relations the prefix `[num]`. The catalog is
  sorted by name, bytewise, with the numeric relations last; a relation's id is its position. `relation_lanes`,
  `relation_signs`, `relation_levels`, `relation_derivations`, `relation_predicates` and `relation_schemas` describe
  each relation, `relation_arities` the encoded arity of its tuples and `relation_slot_roles` each position of them.
- **What is listed.** Every domain predicate and action schema, the helper predicates of quantified conditions
  (`axiom_<k>`) included; the predicates only a problem's axioms define are not. Goal relations exist for every
  predicate except the types `object` and `number`. The equality predicate `=` is left out unless `equality=True`; the
  state relations of the static predicates (the types included) and of the derived predicates can be dropped
  (`include_static`, `include_derived`); their goal and history relations stay. Nullary predicates have relations of
  arity 1 (slot role `auxiliary_slot`); with `nullary='arity0'` they have no argument slot, only the leading slots of
  their relation (a state relation without predicate virtual nodes has arity 0); with `nullary='omit'` they have none.
  `slot_role_names` and `entity_role_names` always list all nine roles. An action relation holds the schema's full
  parameter list (the written parameters and the ones normalization adds for existential variables). Schemas of one name
  and arity (a disjunctive precondition splits an action) share one relation; a name with several arities has one
  relation `name#arity` per arity. The catalog lists only relations a graph can have tuples in.
- **Fingerprint.** `schema_fingerprint` hashes the catalog (names, lanes, arities, slot roles) and the options that
  shape the output but not the catalog (target sources, layout, index type, `numeric_dtype` where values are stored,
  history and `history_max_steps`, horizon mode and root policy), `repeated_literals` (always with subgoal relations,
  otherwise only for a nondefault value), `statics='shared'` (the repeated form adds nothing to
  the hash), the outputs and, with `constant_labels`, the names of the constants, and which arrays are omitted:
  `relation_counts` (`emit_counts=False`) and `entity_role` / `entity_ref` (implicit entity arrays); an encoding that
  stores every array has one fingerprint however it was asked for. It does not depend on declaration order, on
  predicate ids, on the process or on the build.

## Encoding states

`encoder.encode(states, task_ids=None, *, derived=None, goals=None, actions=None, subgoals=None, history=None,
framework=None, threads=1, pool=None, capacities=None, into=None)` returns a `FlatBatch`. It is reentrant: one encoder
serves any number of concurrent calls.

- **`states`:**
  - an array of state words `[N, W]` of any DLPack framework: u64 words, or u32 halves `[N, 2W]` that are reinterpreted
    as the little-endian halves of the words (the layout JAX uses);
  - or a `State` or a list of `State`s, which are packed first (the output is then NumPy).

  The rows of a numeric table are `[N, W + NN]`: the atom words, then the table's `numeric_words` numeric words
  ([Numeric functions](#numeric-functions)); `State`s of a numeric task are packed that way. A row may be wider than
  its instance needs: the words past the instance's own are padding and are not read. A row narrower than its
  instance, or one that sets an atom bit its instance does not have, raises `ValueError`.
  `FlatRelationEncoder(lazy_task)` and `SimpleEncoder(lazy_task)` use `lazy_task.frozen()` as their table source.
  They accept the lazy task's `State`s and translate their set atom bits into one scratch row per worker. A flat
  encoder over a table of frozen twins also accepts those lazy States with the corresponding `task_ids`. Raw arrays,
  `derived`, `goals`, `subgoals`, `history` and `actions` use the twin's slots and table. Literal mask helpers called
  with the encoder use these slots too. The batches have the twin table's fingerprint, equal to an independent parse
  with the same frozen options. `encoder.table` exposes the twin; pickling the encoder preserves the lazy source and
  rebuilds its twin. An atom absent from a reachable twin raises `ValueError` naming the atom. Lazy States retain
  their compact storage; only the scratch row has the twin's width. Horizon and transition encoders require fixed rows.
- **`task_ids`:** the instance of each row, an int32 array or a sequence of ints. It is required for a table of several
  instances. A `State` must be a state of the task of its row (the table's instance named by its task id): a `State`
  of another `Task` object is accepted when that task has the same `fingerprint()` and `encoding_fingerprint()` (for
  example a second parse of the same file), and refused with a `ValueError` naming the row and the instance otherwise.
- **`derived`**, **`goals`**, **`actions`**, **`subgoals`** and **`history`** are the inputs of the
  [derived atoms](#derived-atoms), [goals](#goals), [actions](#actions), [subgoals](#subgoals) and
  [history](#history) of the rows; each applies to every row of the call.
- **`framework`:** `"numpy"`, `"torch"`, `"jax"` or `"dlpack"`; by default the framework of `states` (NumPy for
  `State`s).
- **`threads`:** the threads that encode the batch (default 1; 0 means one per hardware thread; at most 64 or four per
  hardware thread, whichever is more). The result does not depend on it: a batch is byte-identical for every thread
  count. Threads pay off for large batches only: waking a thread costs as much as encoding hundreds of graphs, so every
  thread gets at least 1024 graphs, and a batch of up to 1024 graphs (a single state included) is encoded on the calling
  thread whatever the count. The encoder keeps one pool of worker threads, rebuilt when the count changes (calls of one
  encoder with different counts, from several threads, rebuild it back and forth: give each such caller its own `pool`).
  The pool belongs to the process that made it: after `os.fork()` the child makes its own.
- **`pool`:** a `mymyr.rl.ThreadPool` that you hold, like `rl.expand`'s: it is used instead of `threads` and shared by
  the calls that name it (one batch runs on it at a time).
- **`capacities`** and **`into`:** fixed array sizes and the reuse of a batch's memory
  ([Capacities and buffer reuse](#capacities-and-buffer-reuse)).

Rows in CUDA memory are encoded on the GPU ([Encoding on the device](#encoding-on-the-device)). Every invalid input (a
task id out of range, a missing `task_ids`, a `State` of another task, a malformed `derived` array, rows of a numeric
table without their numeric words) raises `ValueError`; a batch that needs more indices than the index type holds
raises `OverflowError` (`index_dtype='int64'` lifts the limit).

## The batch

For `G` graphs and `R = schema.num_relations` relations, with `I` the index type (`int32` by default, `int64` with
`index_dtype='int64'`); entity ids are batch-global, and graph `g`'s first entity is `entity_offsets[g]`:

| Array | dtype, shape | Meaning |
|---|---|---|
| `task_ids` | int32 `[G]` | the instance of each row |
| `num_objects` | I `[G]` | the objects of the instance (the auxiliary entity is not counted) |
| `entity_offsets` | I `[G + 1]` | graph `g`'s entities are `[entity_offsets[g], entity_offsets[g + 1])` |
| `entity_role` | uint8 `[E]` | the role of each entity ([below](#entities)); omitted when implicit |
| `entity_ref` | I `[E]` | what each entity stands for ([below](#entities)); omitted like `entity_role` |
| `relation_counts` | I `[G, R]` | the tuples of cell `(g, r)`, in both layouts; omitted with `emit_counts=False` |
| `relation_offsets` | I `[G * R + 1]` | segment `s` holds the tuples `[offsets[s], offsets[s + 1])`; cell `(g, r)` is `s = g * R + r`, or `r * G + g` in the relation layout |
| `relation_arg_offsets` | I `[G * R + 1]` | the same offsets into `relation_args` |
| `relation_args` | I `[A]` | the entities of the tuples, tuple after tuple, `relation_arities[r]` per tuple |
| `goal_complete` | uint8 `[G]` | 1 iff the goal relations hold the whole goal; present iff the schema has goal relations |
| `target_offsets` | I `[G + 1]` | graph `g`'s targets are `[target_offsets[g], target_offsets[g + 1])`; present iff the schema has targets |
| `target_entity` | I `[T]` | the entity of each target, graph after graph, in entity order |
| `target_group` | uint8 `[T]` | the group of each target, an index into `encoder.target_group_names`: 0 goal, 1 subgoal, 2 action, 3 history, 4 state |
| `history_offsets` | I `[G + 1]` | graph `g`'s history steps are `[history_offsets[g], history_offsets[g + 1])`; present iff the schema has history |
| `history_entity` | I `[S]` | the entity of each history step, graph after graph, oldest first |
| `history_dt` | int32 `[S]` | the `dt` of each history step |
| `numeric_values` | float32 or float64 `[T_num]` | the value of each numeric tuple ([Numeric functions](#numeric-functions)) |
| `static_task_ids`, `static_offsets`, `static_arg_offsets`, `static_args`, `static_relations` | | the shared static tuples; present iff `statics='shared'` ([Shared statics](#shared-statics)) |

With `outputs` holding `'simple'`, a batch also has the [Simple graph](#the-simple-graph)'s arrays, and without
`'relations'` it has none of the `relation_*` arrays.

The scalars `version`, `num_graphs`, `num_relations`, `layout` (0: graph, 1: relation), `index_dtype` (0: int32,
1: int64), `schema_fingerprint` (the schema's fingerprint as a signed 64-bit value), `table_fingerprint` (the table's
`TaskTable.fingerprint`, likewise: the task ids of two batches name the same instances iff the values agree),
`statics` (0: the static tuples are repeated in every graph, 1: [shared](#shared-statics)), with `numeric_values`
`numeric_dtype` (0: float32, 1: float64) and `numeric_relations` (the number of numeric relations, the last ones of the
catalog), and with the Simple graph `simple_node_labels` and `simple_edge_labels` come with `batch.as_dict()`.

The tuples of a cell are the atoms of its predicate in the lexicographic order of their objects, so the atom slot
layout of a task (`atoms='auto'`, `'reachable'`, `'dense'`) never reaches a model. A static tuple appears once per
graph (with `statics='shared'`, once per instance of the batch). In the graph layout, encoding a slice of a batch gives
the slice of the batch's encoding with the offsets rebased (`entity_ref` needs no rebase); in the relation layout a
slice's cells lie in `R` separate runs, so it is encoded, not cut.

`batch.as_dict(framework)` exports the scalars and arrays without a copy; `batch.<array name>` exports one array in the
batch's framework. NumPy arrays are read-only; do not mutate the shared buffers through frameworks whose tensor APIs
cannot enforce that flag. The batch owns its memory and stays valid after the encoder, the task and the inputs are
gone. `batch.schema` is its `EncodingSchema`, and `batch.sizes` is `None` unless the batch was encoded with
[capacities](#capacities-and-buffer-reuse).

The metadata helpers in `mymyr.encoding` work in the batch's NumPy, torch or JAX framework, including batches from
streams and suites. Passing a `SuiteBatch` returns a dict of results by domain; entity and graph ids are local to
each domain's batch. They compute arrays on request and add nothing to `as_dict()` or `nbytes`.

| Function | Shape, dtype | Meaning |
|---|---|---|
| `entity_counts(batch)` | `[G]`, I | entities per graph |
| `tuple_counts(batch)` | `[G]`, I | tuples per graph, in either layout, including with `emit_counts=False`; shared static blocks are separate |
| `entity_graph(batch)` | `[E]`, I | the graph of each entity, for pooling |
| `entity_roles(batch)` | `[E]`, uint8 | stored roles, or the implicit object and auxiliary roles |
| `object_mask(batch)` | `[E]`, bool | whether each entity is an object |
| `object_entities(batch)` | `[sum(num_objects)]`, I | batch-global object entity ids, graph after graph |

```python
from mymyr.encoding import entity_counts, entity_graph, object_entities, object_mask

sizes = entity_counts(batch)
pooling_index = entity_graph(batch)
objects = object_entities(batch)  # graph g contributes num_objects[g] ids
is_object = object_mask(batch)
```

The first five helpers support `jax.jit` and CUDA graph capture without reading device values on the host. Their
shapes include capacity: per-graph entries past the live graph count are zero; unused entity entries point to
the last live graph (live count − 1, or -1 with no live graphs), with role 0 and object mask False. `E` is the stored entity
extent, or the scalar entity capacity retained by a device/JAX batch with implicit arrays. Host batches with
implicit arrays have no entity capacity storage, so their per-entity results cover only the live entities.
When wrapping the raw torch op's outputs with `DeviceFlatBatch.from_arrays`, supply `entity_capacity=` if entity
arrays are implicit and per-entity helpers will be used. `object_entities` has a data-dependent result length:
it cannot run under `jax.jit` or CUDA graph capture, and selecting objects on CUDA waits to allocate the result.
Use `object_mask` inside compiled or captured computations.

Per-graph object sizes are already `batch.num_objects`; the source of each relation is `batch.schema.relation_lanes`
([Lanes](#lanes)).

### Observation bytes

`batch.nbytes` is the total payload bytes; `batch.nbytes_by_array` is a fresh dictionary of array names to bytes.
They use shape and dtype metadata, without exporting tensors, allocating device memory or waiting for the device.
The API is the same for CPU, torch and JAX batches, including relations, Simple, numeric values, actions, horizons,
shared statics and capacity outputs. Capacity reports include unused entries; JAX scan/vmap reports include stacked
axes. Device counts are included under `device_sizes`, `device_static_sizes` and `device_graphs` when exposed.
Scalars, alignment, allocator overhead, inputs, schemas and scratch are excluded.

An HGraph view reports its backing batch, including capacity; arrays produced on request by `ptr` or `edge_index`
are separate allocations. Suite batches and observations include their domain batches and row mappings, with names
such as `batches.0.relation_args`. `datasets.EncodedRows` includes `graphs.*`, task ids and labels. Named aliases are
counted separately; retained input action expansions are excluded. These are payload sizes, not allocator or process
memory measurements.

```python
sizes = batch.nbytes_by_array
assert sizes == {name: array.nbytes for name, array in batch.as_dict().items() if isinstance(array, np.ndarray)}
assert batch.nbytes == sum(sizes.values())
print(batch.nbytes, batch.nbytes / batch.num_graphs, sizes)  # total, bytes per state, breakdown
```

### Entities

A graph's entities come in blocks, each present only when the options and the inputs give it entities:

| Block | Role (`entity_role`) | `entity_ref` |
|---|---|---|
| the objects of the instance, in the task's object order | 0 `object` | the instance-local object id: `task.objects[entity_ref]` is its name |
| the auxiliary entity, when the graph has a tuple over it (a nullary tuple, `nullary='auxiliary'`) | 8 `auxiliary` | -1 |
| the goal entities | 2 `goal` | the tuple's index among the graph's tuples in graph-major order |
| the subgoal entities | 3 `subgoal` | the same as a goal entity |
| the action entities | 4 `action` | the action's row among its state's rows of `actions` |
| the history step entities | 5 `history` | the step's index among the row's valid steps |
| the history target entities | 6 `history_target` | the tuple's index among the graph's tuples in graph-major order |
| the state entities of a horizon | 7 `state` | the graph-local node index |
| the predicate virtual nodes | 1 `predicate_virtual` | the id of the predicate's key relation |

Every `entity_ref` is graph-local or a property of the catalog: it is the same in either layout, whatever the graph's
place in the batch, so slices, streams and collation need no rebase. An object's instance-local id is also
`entity - entity_offsets[g]`.

The **graph-major tuple index** of a goal, subgoal or history target entity counts the tuples of its graph cell after
cell in relation order, so the graph's first tuple has index 0 in either layout. In the graph layout the tuple is
`relation_offsets[g * R] + entity_ref`; in the relation layout, subtract the counts `relation_counts[g, 0]`,
`relation_counts[g, 1]`, ... from `entity_ref` until the rest `j` falls into a cell `(g, r)`, and the tuple is
`relation_offsets[r * G + g] + j`. Such an entity is the first argument of its tuple.

### Layouts and omitted arrays

`layout='graph'` (the default) stores the tuples graph by graph; `layout='relation'` stores them relation by relation
across the graphs, so one relation's tuples of the whole batch are contiguous (a good fit for per-relation message
passing). Both hold the same tuples in the same order within a cell; only the order of the cells differs: a permutation
of the segments. `relation_counts`, the entity arrays (`entity_ref` included) and the target arrays are equal in both.

`emit_counts=False` drops `relation_counts`; the count of a cell is the difference of two consecutive
`relation_offsets`. `entity_arrays='auto'` drops `entity_role` and `entity_ref` when they carry no information: every
entity is an object (role 0, `entity_ref` the instance-local id) or the auxiliary entity (role 8, ref -1), which holds
without `target_sources`, `predicate_virtual_nodes`, `include_actions`, `history` and a horizon mode
(`schema.implicit_entities` and `encoder.implicit_entities` tell, `batch.implicit_entities` is the batch's flag).
Omitted arrays raise `AttributeError` on the batch and are absent from `as_dict()`.

## Lanes

A relation's lane (`schema.relation_lanes`) says what fills its tuples:

- **fluent**, **derived** and **static**: the atoms of the row, the [derived atoms](#derived-atoms) (given or
  evaluated) and the atoms of the instance, each atom in the relation of its predicate, with its arguments as the
  tuple. The types are static unary predicates, so `object` has one tuple per object. The `=` predicate has relations
  only with `equality=True`.
- the **nullary** atoms, as `nullary` chooses. With `'auxiliary'` (the default) all nullary atoms of a graph share one
  auxiliary entity, and each one is the unary tuple over it; the graph has the auxiliary entity exactly when it has
  such a tuple. With `'arity0'` a nullary atom is a tuple without an argument: its cell counts it in `relation_counts`
  and has no entry in `relation_args`, and no graph has an auxiliary entity. With `'omit'` there are no nullary
  relations and no auxiliary entity. The nullary literals of goals, subgoal layers and history steps follow the same
  choice: tuples over the auxiliary entity, or tuples of their leading slots alone (the goal entity of a goal tuple
  that has one, the step entity of a history tuple, the predicate's virtual node).
- **goal**, **goal_derivation** and **goal_satisfaction**: the [goal](#goals) and [subgoal](#subgoals) literals;
- **action**: one tuple per action of the state ([Actions](#actions));
- **history**: the literals of the row's [history](#history) steps;
- **numeric**: one tuple per numeric slot of the instance ([Numeric functions](#numeric-functions));
- **parent**, **sibling** and **cousin**: the edges of a horizon and the kinship of its nodes
  ([Horizons and transitions](#horizons-and-transitions)).

A call without `subgoals`, `history` or `actions` has no tuples in their relations.

```python
blocks = mymyr.Task.from_text("tests/data/tasks/blocks__probBLOCKS-8-0.txt")  # handempty is nullary
choices = {v: encoding.FlatRelationEncoder(blocks, nullary=v) for v in ("auxiliary", "arity0", "omit")}
encoded = {v: e.encode(blocks.initial_state) for v, e in choices.items()}
assert choices["auxiliary"].describe(encoded["auxiliary"], 0).relations["handempty"] == (("auxiliary",),)
assert choices["arity0"].describe(encoded["arity0"], 0).relations["handempty"] == ((),)
assert "handempty" not in choices["omit"].relation_names

# an arity-0 cell counts its tuples and has no arguments; no graph has the auxiliary entity
handempty = choices["arity0"].relation_id("handempty")
assert choices["arity0"].relation_arities[handempty] == 0 and encoded["arity0"].relation_counts[0, handempty] == 1
assert int(encoded["arity0"].entity_offsets[-1]) == int(encoded["auxiliary"].entity_offsets[-1]) - 1 == 8
```

### Goals

The goal of an instance is `Task.encoding_goal` read as a set of ground literals (a conjunction, with `forall`
expanded where the front end can). `goal_complete` is 0 for an instance whose goal had a conjunct it could not expand
and for one with a literal that the options leave out although a state can change its truth (a nullary literal under
`nullary='omit'`); a left-out literal of a static predicate has a fixed truth and leaves the goal complete.
A literal is placed in the relations of its predicate, `[+]` for a positive and `[-]` for a negative one. With the
default `goal_derivations=("plain", "satisfied")` the relations are, per predicate and sign:

- the plain relation `[+]on[g]` holds the tuple of every goal literal;
- `[+]on[g][sat]` holds the literals that hold in the state (a positive literal whose atom is true, a negative one
  whose atom is false); `goal_derivations` may add `"unsatisfied"`, `[+]on[g][unsat]`, the literals that do not.

The literals of fluent and derived predicates are evaluated against the state words with the goal's masks; those of
static predicates and of atoms without a slot (not reachable by any state) have a fixed truth and form the constant part
of the goal. A statically false literal is still emitted: in its plain relation, and in `[unsat]` for a positive
literal over a false atom or `[sat]` for a negative one. Goal literals whose predicate has no goal relation (equality
without `equality=True`, nullary atoms with `nullary='omit'`, the types) are not encoded. A cell lists its tuples in the
lexicographic order of their arguments, constant part and masked literals merged.

With `target_sources=("goal",)` every plain goal tuple gets a **goal entity** (after the objects and the auxiliary
entity), numbered in the order of the plain goal relations (relation id, then tuple), so that the id is
`entity_offsets[g] + num_objects[g] + has_auxiliary + k` for the `k`-th plain goal tuple of graph `g` (`has_auxiliary`
is 0 under `nullary='arity0'`). The entity is the first argument of its tuple, its `entity_ref` the tuple's [graph-major
index](#entities), and the batch lists the goal entities as targets of group 0.

**`goals=`** gives every row a goal of its own instead of its instance's: `(goal_pos, goal_neg)` or `(goal_pos,
goal_neg, goal_derived_pos, goal_derived_neg)`, masks over the fluent (derived) atom slots as `rl.goal_masks` returns
them. The arrays are `[N, W]` (`[N, D]` for the derived pair), u64 words or u32 halves, or `[1, W]` for one goal that
every row shares. These masks define the row's whole goal, with no constant part, and its `goal_complete` is 1.
With subgoal layers, `repeated_literals` decides which [levels](#subgoals) hold these literals. The masks of a row may be wider than its instance needs (the extra words are padding); a mask
that is narrower, or one that sets an atom bit its instance does not have, raises `ValueError`. Without the derived
pair a row's derived goal part is empty. The masks of `rl.goal_masks` (`table.goal_masks()`) are the instance's goal
over fluent slots only: they give no derived masks, and a goal with such a part keeps only its fluent literals. A call
with these masks gives the same batch as a call without `goals` only for an instance whose goal is complete and has no
literal over a static predicate, over a derived predicate or over an atom without a slot (the constant part, the
derived part), so that the masks are the whole goal.

**Goals from literals.** `mymyr.encoding.goal_masks(source, goals, task_ids=None, fixed="decide")` builds the goals of
rows from lists of literals (anything `Task.literal` takes, read in the task of the row's instance) as a `GoalMasks`:
`pos`, `neg` `[N, W]`, `derived_pos`, `derived_neg` `[N, D]` (`None` for a table without derived predicates) and
`unsatisfiable` bool `[N]`; it is the sequence `(pos, neg[, derived_pos, derived_neg])` that `goals=` takes. The source
is an encoder, a `TaskTable` or a `Task`. A literal over a static predicate, or over a fluent or derived atom without a
slot, has a fixed truth and no bit in the masks: `fixed="decide"` leaves it out when it holds and marks its row
unsatisfiable when it fails, `fixed="raise"` raises `ValueError`. Such a row goal differs from the instance's goal in
its constant part: the instance's goal lists its static literals as constant tuples, a row goal has none (its relations
list the slotted literals only), and both have `goal_complete` 1. `literal_masks(source, predicates, objects, positive,
rows, num_rows, task_ids=None)` builds the same from arrays: table predicate ids `[K]` (indices into
`table.predicate_names`), instance-local object ids `[K, k]` padded with -1, the sign and the row of each literal, and
the instance of each row. `literal_slots(source, predicates, objects, task_ids=None)` is the lookup itself: the
`LiteralKind` of each literal (`FLUENT`, `DERIVED`, `STATIC_TRUE`, `STATIC_FALSE`, `UNREACHABLE` for an atom without a
slot, `INVALID` for ids out of range) and its slot (-1 for the kinds without one).

```python
from mymyr import rl

goal_task = mymyr.Task.from_pddl("tests/data/pddl/goals/domain.pddl", "tests/data/pddl/goals/static_true.pddl")
row_goals = encoding.goal_masks(goal_task, [["(lit a)", "(heavy t1)"], ["(lit b)", "(heavy t2)"]])
assert row_goals.unsatisfiable.tolist() == [False, True]  # (heavy t1) holds, (heavy t2) fails: both are static
lit = rl.as_table(goal_task).predicate_names.index("lit")  # a table predicate id
a = goal_task.atom("(lit a)")
kind, slot = encoding.literal_slots(goal_task, [lit], [list(a.object_indices)])
assert kind.tolist() == [encoding.LiteralKind.FLUENT] and slot.tolist() == [a.slot]
```

### Actions

**`actions=`** is the `rl.Expansion` of the same states over the same table (`rl.expand(table, states, task_ids)`), on
the host: the encoder reads its action rows, the labels of the successors, instead of recomputing the applicable
actions. The expansion must hold the states of the call in order (`actions.num_states` equals the rows), with every
row's actions in the canonical order that `rl.expand` writes (ascending action schema, then ascending binding), and
must not have overflowed. It must be over the encoder's own table, or over a table whose instance for each row has the
same `fingerprint()` and `encoding_fingerprint()`; any other expansion, a label narrower than the table's, or a schema
or object the instance lacks raises `ValueError`. `actions` needs the action relations (`include_actions=True`, the
default). An expansion that `rl.expand` pruned (a witness expansion, which keeps the rows of a subset of the actions)
encodes exactly the rows it has.

Every distinct action of the state is one tuple of the action relation of its schema `name`, and the tuple is always
`(action entity, o_1, ..., o_k)`: the **action entity** (after the goal and subgoal entities) followed by the binding
`(o_1, ..., o_k)`. A schema with no parameters has the unary tuple `(action entity)` and takes no auxiliary entity. A
schema that the front end split in several (a disjunctive precondition) shares one relation, and an action that two of
its schemas both give is one tuple: the relation lists each binding once. There is no limit on the schemas of a
relation.

The action entities exist whenever the batch has action tuples, with or without `target_sources=("action",)`. They are
numbered in the order of the action relations (relation id, then tuple). The entity's `entity_ref` is the row of the
action among the rows of its own state in the expansion (the first one for a binding listed by several schemas). The
row of the expansion is `actions.offsets[g] + entity_ref`, so `actions.action(row)` is the action and
`actions.succ[row]` its successor: a policy that picks an action entity picks that row (`encoding.action_rows`,
`encoding.action_indices` and `encoding.action_entities` do the mapping; [Action entities](rl.md#action-entities)).
With `target_sources=("action",)` the batch also lists the action entities as targets of group 2, after the goal and
subgoal targets of their graph; without it they are entities and tuple arguments only. A call without `actions` has no
action tuples and no action entities.

Device rows accept a CUDA expansion as `actions=` or `(schema, binding, offsets)` device tensors, including
`expand_flat`'s six outputs. Split-schema runs use the same k-way merge as the host, and action targets, graph-local
references, both layouts and both index widths have identical output. The torch `encode_flat` op accepts
`action_schema`, `action_binding`, `action_offsets` and optional `action_parent` keyword tensors. The device takes
action rows only for the relation arrays of a classical table, and the JAX FFI refuses them because its operand list
has no expansion labels or offsets. The action mapping helpers return host NumPy indices;
reading CUDA offsets or chosen entities through them copies those arrays to the host.

### Predicate virtual nodes

With `predicate_virtual_nodes=True` a graph has one **predicate virtual node** (the last entities of the graph) for each
predicate that one of its tuples has a relation of: the relations of a predicate are its state relation, its goal
relations (`[+]on[g]`, `[+]on[g][sat]`, ...) and its history relations. A tuple of such a relation has the node as an
extra argument right before the predicate's own arguments (after the goal entity of a goal tuple that has one); action
tuples have no predicate slot. A node exists only if some tuple uses it, so a predicate without tuples in the graph has
no node, and the nodes are numbered by the name of their predicate (ascending, bytewise), whatever relations the schema
keeps. The node's `entity_ref` is the id of the predicate's key relation: its plain state relation, or for a predicate
without one (a static or derived predicate whose state relation is dropped) its first relation in the catalog (C++:
`EncodingSchema::virtual_key`). The nullary atoms of a predicate are the tuple over its node and the auxiliary entity,
or over its node alone with `nullary='arity0'`.

### Derived atoms

The derived atoms of a row (the atoms that the task's axioms define) fill the derived state relations and decide which
derived goal and subgoal literals hold (`[sat]`, `[unsat]`). By default the encoder evaluates them, and only for rows
whose instance has derived relations in the schema. **`derived=`** gives them instead: the derived atom words of each
row, `[N, D]` u64 (or u32 halves), as `rl.expand(..., derived=True)` computes them for the successors ([Derived
words](rl.md#derived-words)). The given words take precedence over the evaluation.

### Subgoals

With `max_goal_level = L` the schema has, for every predicate and sign, goal relations of the levels 1 to `L` besides
the goal's: `[+]on[sg]`, `[+]on[ssg]`, `[+]on[sssg]` and their `[sat]` and `[unsat]` variants, as `goal_derivations`
says. **`subgoals=`** gives the layers of every row, `(pos, neg)` or `(pos, neg, derived_pos, derived_neg)`: masks over
the fluent (derived) atom slots, `[N, L', W]` (`[N, L', D]` for the derived pair) with `L'` at most `max_goal_level`, or
`[1, L', W]` for layers that every row shares; u64 words or u32 halves. It needs `max_goal_level >= 1` and goal
relations. Layer `l` of a row (`l = 0, 1, ...`) fills the relations of level `l + 1`; a row may have fewer layers than
the schema has levels (the missing ones are empty). Masks narrower than the instance, or with a bit the instance does
not have, raise `ValueError`, as for `goals`. `mymyr.encoding.subgoal_masks(source, layers, task_ids=None,
fixed="raise")` builds them from lists of literals (per row, per layer) as a `LayerMasks`: `pos`, `neg` `[N, L', W]`,
the derived pair and `unsatisfiable` bool `[N, L']`, the sequence `subgoals=` takes. A subgoal of fixed truth (static,
or an atom without a slot) raises `ValueError` by default; `fixed="decide"` leaves it out or marks its layer
unsatisfiable, as `goal_masks` does for a row. `literal_masks(..., layers=, num_layers=)` builds the layers from
arrays.

`repeated_literals` decides where a literal listed at several levels is encoded. The goal is level 0, below all
subgoal layers; the sign is part of the literal, so positive and negative literals are placed independently.

| Value | Placement | Literal in the goal and layers 0 and 2 |
|---|---|---|
| `"every"` (default) | each level holds exactly its input; the goal holds its whole goal | levels 0, 1 and 3 |
| `"lowest"` | once, at the lowest level that lists it | level 0 |
| `"highest"` | once, at the highest level that lists it; lower levels lose it | level 3 |

For a literal in layers 0 and 2 only, `"every"` puts it at levels 1 and 3, `"lowest"` at level 1, and `"highest"`
at level 3. The policy applies to fluent and derived literals, using a row's own `goals` when given. The instance's
constant goal literals have no slots in a layer and are unaffected. `goal_complete` is unchanged by placement:
a goal literal moved to a subgoal level is still represented in the graph. With no subgoal layers the option has
no effect. History steps are independent of goal levels.

Every cell is a set of strictly ascending tuples: a literal listed twice in one layer is one tuple. The `[sat]` and
`[unsat]` variants follow the plain placement and contain the level's literals that hold, or do not hold, in the row's
state. Goal and subgoal target entity counts follow the number of plain tuples at their levels.

With `target_sources` including `"subgoal"` every tuple of a plain subgoal relation gets a **subgoal entity** (after the
goal entities), numbered in the order of the plain subgoal relations (relation id, then tuple), the first argument of
its tuple, with its [graph-major index](#entities) as `entity_ref`, exactly as a goal entity. The batch lists them as
targets of group 1, after the goal targets of their graph.

### History

With `history=True` the history relations `[+]on[hist]` and `[-]on[hist]` of the schema hold the literals of the steps
of a row's history. **`history=`** gives it: `(pos, neg, dt)`, `(pos, neg, dt, count)` or `(pos, neg, dt, count,
derived_pos, derived_neg)`: masks `[N, H, W]` (`[N, H, D]`) of the steps of a row, oldest first, `dt` an integer array
`[N, H]` (the time of each step relative to the state, as the caller defines it; the encoder only stores it) and
`count` an integer array `[N]` with the valid steps of each row, a prefix of its `H` (`None`: all `H`). Steps past a
row's count are padding and never read. `mymyr.encoding.history_masks(source, steps, task_ids=None)` builds the
arrays from lists of `(dt, literals)`; a literal of fixed truth says nothing about a step and raises `ValueError`.

Of a row's valid steps only the **newest** `history_max_steps` are encoded (`None`, the default: all of them). Each
encoded step is one **history step entity** (after the action entities), in step order, oldest first; the batch lists
them in `history_offsets` `[G + 1]`, `history_entity` `[S]` (the entity of each step) and `history_dt` int32 `[S]` (its
`dt`). The `entity_ref` of a step entity is the step's index among the row's valid steps (not among the encoded ones:
step `entity_ref` of the row's `history` is `history[2][g, entity_ref]` for its `dt` and `history[0][g, entity_ref]`
for its positive mask, whatever `history_max_steps` is).

A history tuple is `(history step, [predicate node,] o_1, ..., o_k)`, and with `target_sources` including `"history"`
it starts with a **history target entity** (after the history steps) of its own: `(history target, history step,
...)`. The tuples of a cell are in the order of the steps, then of the atom slots (which is the lexicographic order of
the objects). History target entities are numbered in the order of the history relations (relation id, then tuple) and
their `entity_ref` is the tuple's [graph-major index](#entities). That is the order in which the tuples are filled, not
the order of the steps, so the target of a literal is found through its tuple. The batch lists them as targets of
group 3, after the action targets of their graph.

```python
layered = encoding.FlatRelationEncoder(task, max_goal_level=1, history=True, target_sources=("subgoal", "history"))
subgoals = encoding.subgoal_masks(layered, [[["(carry o2 o14)", "(not (at-robby o0))", "(at o2 o1)"]]])
history = encoding.history_masks(layered, [[(-1, ["(at-robby o1)"]), (0, ["(at-robby o0)"])]])
layered_batch = layered.encode([state], subgoals=subgoals, history=history)

names = layered.entity_names(layered_batch, 0)
assert [n for n in names if n.endswith("[sg]")] == ["[+](at o2 o1)[sg]", "[+](carry o2 o14)[sg]", "[-](at-robby o0)[sg]"]
assert names[-4:] == ["history:-1#0", "history:0#1", "history:-1#0:[+](at-robby o1)", "history:0#1:[+](at-robby o0)"]
assert layered_batch.history_dt.tolist() == [-1, 0] and layered_batch.target_group.tolist() == [1, 1, 1, 3, 3]

lowest = encoding.FlatRelationEncoder(task, max_goal_level=1, repeated_literals="lowest", target_sources=("subgoal",))
lowest_batch = lowest.encode([state], subgoals=encoding.subgoal_masks(lowest, [[["(carry o2 o14)", "(at o2 o1)"]]]))
assert [n for n in lowest.entity_names(lowest_batch, 0) if n.endswith("[sg]")] == ["[+](carry o2 o14)[sg]"]
```

`(at o2 o1)` is also a literal of the task's goal. Under the default `repeated_literals="every"` it is in the goal
relations and in level 1, so it has a subgoal entity; under `"lowest"` it stays in the goal relations alone.

## Horizons and transitions

Goal and subgoal levels use `repeated_literals` as above, with satisfaction evaluated at each horizon node.

`FlatHorizonEncoder(table_or_task, *, horizon_mode, <options>)` encodes **horizons**: one graph per root state, holding
the root and the nodes below it (states reached from the root by actions) as state entities, so that a model sees a
root and its candidate successors in one graph. `horizon_mode` is required (`'full'`, `'delta'` or `'action'`), and
`root_policy`, `parent_relation`, `sibling_relation` and `cousin_relation` are its own options. It shares the batch
format, `schema`, `table`, `entity_names`, `describe` and pickling of the flat encoder, and a `FlatBatch` of a horizon
encoder has `batch.schema.options['horizon_mode']` set. `encoder.encode(view, *, derived=None, goals=None,
subgoals=None, framework=None, threads=1, pool=None, capacities=None)` returns a `FlatBatch` of `G` graphs, one per
root.

**The view.** The nodes and edges are given as a `mymyr.encoding.HorizonView` (or any object with its attributes),
the rows of one batch of states plus offsets and edges, as `rl.expand` produces them:

| Attribute | Meaning |
|---|---|
| `nodes` | state words `[M, W]` of any DLPack framework (u64, or u32 halves), or a list of `State`s |
| `node_offsets` | int32 `[G + 1]`, from 0, strictly ascending: graph `g`'s nodes are the rows `[node_offsets[g], node_offsets[g + 1])`; the first is its root, and **node `k` of graph `g` is row `node_offsets[g] + k`** |
| `edge_offsets` | int32 `[G + 1]`, from 0, ascending: the edges of graph `g` |
| `edge_src`, `edge_dst` | int32 `[E]`: graph-local node indices of an edge's ends |
| `edge_schema` | int32 `[E]`: the action schema of an edge |
| `edge_binding` | int32 `[E, L]`: the objects of an edge's ground action (`L` at least the table's `label_width`; only the schema's arity entries of a row are read) |
| `task_ids` | int32 `[G]`: the instance of each graph (required for a table of several instances) |
| `truncated` | uint8 `[G]`: set where `encoding.horizon` stopped a graph at `max_nodes`; `None` when not known |
| `parent_offsets`, `parent_edges` | optional int32 `[G + 1]` and `[P]`: graph-local edge indices representing the distinct parent pairs, sorted by `(src, dst)` within each graph |

The builders prepare the parent arrays so the encoder can reuse their canonical pairs. Supply both arrays together,
with one edge per distinct pair and offsets from zero, or omit both to canonicalize arbitrary edges during encoding.
Prepared pairs must describe the complete pair set of the unchanged edge arrays; omit them when editing edges.

**The edges.** The graph is a dag: a node may have several parents, an edge may lead back to the root or to its own
source, and the edges may come in any order and more than once. The **action tuples** of a relation are the distinct
`(dst node, objects)` of the edges whose schema belongs to it, ordered by destination node and then objects, and the
**`_parent_` tuples** are the distinct `(src, dst)` pairs in ascending order; so reordering or repeating the edges does
not change the batch, a node that several edges reach with the same action has one action tuple, and a node with
several parents has one `_parent_` tuple for each. (The tuples of every other relation are in the canonical order of
the flat encoder.) `rl.expand` rows are a valid edge list: the rows of a state's successors, `schema`, `binding`, with
`edge_src` the state's node and `edge_dst` the node you made of the successor. The edge arrays a schema does not read
(all of them without action and parent relations, `edge_schema` and `edge_binding` without action relations) and the
arrays of a batch without any edge may be `None`. `derived` is per node row `[M, D]`; `goals` and `subgoals` are per
graph (`[G, ...]`, or one row for all graphs) and replace the instance's goal and layers for that graph. Invalid views
raise `ValueError` naming the graph (`graph 3 of the horizon has no nodes`, `edge 2 of graph 0 of the horizon has an
endpoint outside the graph's nodes`, `node 1 of graph 2 sets an atom bit its instance does not have`).

**State entities.** The graph has an entity of role 7 (`state`) per node, after the objects and the auxiliary entity
and before the predicate virtual nodes. Its `entity_ref` is the **graph-local node index** `k`, so the node is row
`horizon.node_offsets[g] + entity_ref` of the view's `nodes`, whatever the layout or the graph's place in the batch.
Names are `target:k` (`encoder.entity_names`). The nodes that are candidates for a model's choice (`target_group` 4)
are all state entities under `root_policy='include'` and all but the root otherwise; the batch lists them in
`target_entity`, so the schema has targets (the capacity form needs `targets`).

**Root policies.** `include` (the default) makes the root node 0 a state entity and a target candidate. `encode_only`
gives it a state entity but does not list it as a target. `exclude` has no state entity for the root: the graph's
entities are the nodes 1 and up, and the root's relations are the plain ones (no state slot): the root's atoms are
tuples without a state argument. An edge into the root gives no action tuple (the action tuples start with the state of
their destination) and an edge that touches the root no `_parent_` tuple; an edge out of the root keeps its action
tuple, which names the node it leads to. The root has no entity at all; its relations say what it holds.

**Modes.** Every relation of a node has the node's state entity as its first argument (the slot role `state`):

- `full`: the atoms of every node, `on(node, a, b)`, the root included under `include` and `encode_only`; under
  `exclude` the root's atoms are the plain relations `on(a, b)` and the nodes' `on[state](node, a, b)`. The static
  atoms are the root's only (the same for every node): `object(target:0, a)`, or plain under `exclude`. The goal
  relations hold the goal's tuples once, at the root, and the satisfaction relations `[g][sat]`, `[g][unsat]` the
  satisfaction of the goal literals at every node.
- `delta`: the facts the node adds to or deletes from its **root** (not its parent): `[+]on(node, a, b)` for an atom
  that holds in the node and not in the root, `[-]on(node, a, b)` for one that holds in the root and not in the node.
  The root's own atoms are in the plain relations (with the root's state entity unless it is excluded). With
  `goal_derivations` `added_satisfied` and `added_unsatisfied`, the relations `[+]on[g][sat+]` and `[sat-]` hold the
  goal literals that the node makes true and those it makes false, relative to the root. A node equal to its root (a
  self-loop's end) has no delta tuples.
- `action`: the root's state and, for every edge, the action that leads to its target node: the action relation of the
  schema holds `(dst node, o_1, ..., o_k)`. The nodes carry no facts of their own.

The catalog lists only relations a graph can have tuples in: a static predicate has no delta relation and no relation
of the nodes (its atoms are the root's), and the added-satisfaction relations exist per goal level. The atoms and goal
literals of nullary predicates are tuples over the graph's auxiliary entity, in front of which stands the state entity
(`handempty(node, auxiliary)`); the graph has the auxiliary entity exactly when a nullary tuple exists. With
`nullary='arity0'` the state entity is their only argument (`handempty(node)`), and the root's nullary atoms under
`exclude` are tuples without an argument.

Streams do not encode horizons (a horizon is a batch of graphs with offsets, not a row): the horizon and transition
encoders have no `stream`, and C++ `FlatStream(encoder)` refuses them with `std::invalid_argument`. Shared statics are
refused with a horizon mode (the static tuples of a horizon graph are anchored by its root's state node and so are not
the instance's), and so are fluent numeric functions. The device encodes horizons of any depth and transitions
([Horizons on the device](#horizons-on-the-device)).

```python
from mymyr import rl

table = rl.as_table(task)
roots = np.ascontiguousarray(task.encode([task.initial_state]))
exp = rl.expand(table, roots)
n = int(exp.offsets[1])                                # the root's successors
view = encoding.HorizonView(
    nodes=np.concatenate([roots, np.asarray(exp.succ)]),            # node 0 is the root, nodes 1..n its successors
    node_offsets=np.array([0, n + 1], np.int32), edge_offsets=np.array([0, n], np.int32),
    edge_src=np.zeros(n, np.int32), edge_dst=np.arange(1, n + 1, dtype=np.int32),
    edge_schema=np.asarray(exp.schema), edge_binding=np.asarray(exp.binding),
)
horizon = encoding.FlatHorizonEncoder(task, horizon_mode="full", parent_relation=True)
horizon_batch = horizon.encode(view)
assert horizon.entity_names(horizon_batch, 0)[-n - 1:] == [f"target:{k}" for k in range(n + 1)]
```

### Building horizons

`encoding.horizon(table_or_task, roots, depth=1, *, task_ids=None, max_nodes=None, truncate=False, merge=True,
threads=1, pool=None)` builds the `HorizonView` of states by expanding them in C++ (`rl.expand`), so you need not write
the loop of the example above. `roots` are states (`State`s or an array `[N, W]`); graph `g` is the horizon of
`roots[g]`, an instance `task_ids[g]` of the table. `encoding.horizon_of_space(space, roots=None, depth=1, *,
max_nodes=None, truncate=False, merge=True)` does the same over a `datasets.StateSpace` generated with its transition
labels (`roots`: state ids, `None` for every state): the nodes and edges follow the space's recorded transitions, and
`FlatHorizonEncoder(space.task, ...)` encodes the view.

- **Order.** A graph is built breadth first. The root is node 0; then, level by level, the successors of the nodes of
  the level before, in node order, each node's successors in the canonical order of `rl.expand` (by action schema,
  then by objects). The nodes of the last level (`depth`) are not expanded; `depth=0` gives root-only graphs. Edge `e`
  of a graph goes from a node to one of its successors: one edge per (node, successor row), so two actions that lead to
  the same state are two edges. The edges are ordered by destination node, then by action schema, then by objects;
  edges that agree keep the order of their source nodes and of `rl.expand`. In a tree (`merge=False`) that is the
  order the nodes are numbered in.
- **Dedupe.** With `merge=True` a state is one node per graph: the first time it is reached it gets the next node
  number, and every later successor row with the same state words is an edge to that node. A state with several
  predecessors is one node with several incoming edges, a cycle is an edge to an earlier node, and a successor equal to
  its state is a self-loop. Graphs do not share nodes. With `merge=False` every successor row is a node of its own and
  the graph is a tree (the horizon of the example above).
- **Determinism.** The view does not depend on `threads`, on `pool` or on the process: expansion is deterministic and
  merging and numbering follow the fixed order above. `threads=0` uses one per hardware thread; `pool` is a
  `mymyr.rl.ThreadPool`.
- **Memory.** All the graphs of a call are built at once and held in memory. While they are built every node takes
  `8 * W + 8` bytes (`W` state words, its graph and number) and, when merging, `8` more for its hash and `8` to `16`
  for the hash table; every edge takes `4 * (4 + L)` bytes (`L` action label columns). The view (`8 * W * nodes + 4 *
  (3 + L) * edges + 4 * (P + G + 1)` bytes, with `P` distinct parent pairs and `G` graphs) is then written from them,
  so the peak is about the sum of the two; expansion runs in chunks of 4096 states, so nothing else grows with the
  number of nodes. The nodes of a graph are the states within `depth`
  actions of its root, which grows exponentially with `depth`, so bound it: `max_nodes` is the most nodes of a graph
  (the root included; `None` is unbounded). A graph that would have more raises `ValueError` ("mymyr: encoding: graph 3
  of the horizon would have more than `max_nodes = 100` nodes"), or, with `truncate=True`, stops growing: a successor
  that is not yet a node is left out together with its edge (successors that are nodes already still get their edges),
  and `view.truncated[g]` is 1. A result of more than 2^31 - 1 nodes or edges raises `ValueError`.

### Rooted horizons

`FlatRootedHorizonEncoder(table_or_task, *, horizon_mode='full', root_policy='encode_only', parent_relation=True,
sibling_relation=False, cousin_relation=False, <options>)` is a `FlatHorizonEncoder` (same `encode`, same view) with
the defaults of a graph seen from its root: the root is the state entity `target:0`, encoded but not a target
candidate, and `_parent_` holds the edges. `FlatHorizonEncoder` takes `sibling_relation` and `cousin_relation` too, as
does `EncodingSchema`; each needs a horizon mode.

The relations `_sibling_` and `_cousin_` (slot roles `state`, `state`) relate the nodes of a graph through its
`_parent_` edges, as sets over the merged nodes. The parents of a node `x` are the sources of the `_parent_` edges into
`x`: parallel edges count once, every distinct parent counts, a self-loop does not (a node is never its own parent),
and under `root_policy='exclude'` the root has no entity, so its edges are no parent edges: it is nobody's parent and
nobody's child.

- `_sibling_(i, j)` holds iff `i != j` and `i` and `j` have a parent in common.
- `_cousin_(i, j)` holds iff `i != j` and some parent `p` of `i` and some parent `q` of `j` are different siblings
  (`p != q`, `_sibling_(p, q)`).

Both relations list each pair in both orders, in ascending `(i, j)` order of the nodes' indices. A tree has the usual
meaning: siblings share a parent, cousins have parents that are siblings; edges that give a node a second parent make
more siblings and cousins (a node is related through each of its parents). The pair lists grow with the square of a
node's parents' fan-out: a dense dag (a wide horizon of a task with many applicable actions) can give `_sibling_` and
`_cousin_` far more tuples than it has edges, and the encoding time grows with them. An example, the dag with the
edges `0->1, 0->2, 0->1 (again), 1->3, 2->3, 1->4, 2->5, 3->6, 4->0`:

| | parents | children |
|---|---|---|
| 0 | 4 | 1, 2 |
| 1 | 0 | 3, 4 |
| 2 | 0 | 3, 5 |
| 3 | 1, 2 | 6 |
| 4 | 1 | 0 |
| 5 | 2 | |
| 6 | 3 | |

`_sibling_` is `(1,2) (2,1) (3,4) (3,5) (4,3) (5,3)`: 1 and 2 share the parent 0; 3 has the parents 1 and 2, so it is a
sibling of 1's other child 4 and of 2's other child 5, and 4 and 5 (parents 1 and 2) are not siblings. `_cousin_` is
`(0,6) (3,4) (3,5) (4,3) (4,5) (5,3) (5,4) (6,0)`: the siblings 1 and 2 make a child of 1 and a different child of 2
cousins (children of 1: 3 and 4; of 2: 3 and 5; so `(3,5)`, `(4,3)`, `(4,5)` and their reverses), and the siblings 3
and 4 make their children 6 and 0 cousins; 5 has no children. With `root_policy='exclude'` the edges `0->1`, `0->2` and
`4->0` are not parent edges: `_sibling_` is `(3,4) (3,5) (4,3) (5,3)` and `_cousin_` is empty. A self-loop does not
make relatives: with the edge `4->4` added, nothing above changes.

### Transitions

`FlatTransitionEncoder(table_or_task, *, root_policy='exclude', parent_relation=False, <options>)` and
`FlatTransitionEffectsEncoder(...)` encode the **transitions** of states: the graph of a state is its horizon of depth
one, with one node per successor (no merging: a successor that equals its state is a node of its own) and one edge per
successor, in the order of `rl.expand`. `FlatTransitionEncoder` is the `full` mode, each successor with its facts;
`FlatTransitionEffectsEncoder` the `delta` mode, each successor with what it adds to and deletes from its state (a
self-loop, or any transition that changes nothing, has no tuples). By default the state is not encoded as an entity
(`root_policy='exclude'`: its facts are the plain relations). The goal and action options are those of the horizon
encoder; the horizon mode is fixed.

`encoder.encode(states, task_ids=None, *, derived=None, goals=None, actions=None, subgoals=None, framework=None,
threads=1, pool=None, capacities=None)` takes the parent states: the expansion is made here (on the pool), or given as
`actions=` (the `rl.expand` of the same states; `derived` then needs an expansion made with `derived=True`). The batch
equals what `FlatHorizonEncoder` with the same options (`horizon_mode='full'` or `'delta'`) encodes of
`encoding.horizon(table, states, depth=1, merge=False)`, bit for bit.

```python
roots = np.ascontiguousarray(task.encode([task.initial_state]))
view = encoding.horizon(task, roots, depth=2, max_nodes=10_000)        # merged by state, bounded
assert view.truncated[0] == 0 and view.node_offsets[-1] == len(view.nodes)

rooted = encoding.FlatRootedHorizonEncoder(task, sibling_relation=True, cousin_relation=True)
rooted_batch = rooted.encode(view)
assert {"_parent_", "_sibling_", "_cousin_"} <= set(rooted.describe(rooted_batch, 0).relations)

transitions = encoding.FlatTransitionEncoder(task)
tree = encoding.horizon(task, roots, depth=1, merge=False)
same = encoding.FlatHorizonEncoder(task, horizon_mode="full", root_policy="exclude").encode(tree)
assert np.array_equal(transitions.encode([task.initial_state]).relation_args, same.relation_args)
```

### Horizons on the device

A `HorizonView` whose arrays are on a CUDA device encodes on the device into a torch `DeviceFlatBatch`, equal to the
host batch: full, delta and action horizons of any depth, every root policy, parent relations and rooted kinship,
both layouts and both index types. The device does not read prepared parent pairs (`parent_offsets`, `parent_edges`):
it canonicalizes the edges itself, and it validates the offsets and the graph-local endpoints, task ids, labels and
node bits before it fills anything (an invalid view raises `ValueError` without capacities and sets `invalid_input`
with them). Equal node rows of an uploaded view stay distinct nodes. Cousin relations gather each node's cousins on
demand, in scratch linear in the graph's nodes and edges.

There are three ways to a device view. Upload a host view (its arrays as CUDA tensors).
`encoding.horizon_of_space(device_space, roots, depth=k, *, max_nodes=None, truncate=False)` builds the view of a
`cuda.DeviceStateSpace` generated with its transition labels by traversing its recorded state ids on the current torch
stream: the nodes and edges are those of `horizon_of_space` over `device_space.to_host()`, in the same order. `roots`
are host state ids, CUDA int32 state ids or `None` (every state); the roots are traversed in parallel, each one breadth
first. It needs `merge=True` (the recorded ids are the merged states) and refuses numeric spaces; it reserves `G *
min(max_nodes or N, N)` node rows and `G * E` edge rows for `G` roots, `N` states and `E` transitions, and the offsets
delimit the rows it uses. A device root outside the space, or a graph past `max_nodes` without `truncate`, gives invalid
offsets, which the encoder refuses. `encoding.horizon(table, cuda_rows, depth=1)` builds the compact view on the device
from an expansion, with `merge` as on the host; it needs `depth=1` (deeper views come from a state space or an upload),
and `max_nodes`, `truncate` and `pool` raise `ValueError`. That builder runs every graph and successor in parallel
(merging by a hash table over each graph's rows): its time grows with the successors, not with the square of a graph's
fanout.

The transition encoders expand CUDA parent rows and encode them on the current stream; `actions=` a device expansion
reuses its successor rows. With `capacities` the expansion is bounded by the entity capacity (every successor is a
node with a state entity), and nothing waits for the device unless the table's expansion needs the host (instances
expanded apart, CPU-fallback schemas or axioms, lazy atom slots, batches beyond one chunk of views; `expand_into` in
[rl.md](rl.md)): an expansion past the capacity makes the batch report `capacity_overflow`, an invalid parent row
`invalid_input`. That expansion is sized by `capacities["entities"]`, not by the successors the rows have: the
successor rows, the builder's nodes and scratch and the encoder's horizon scratch each hold that many rows (about
`16 + 8 * W` bytes a row, `8 * D` more with derived words), and the expansion zeroes and goal-checks all of them, so a
generous entity capacity costs memory and time in proportion; an expansion of your own as `actions=` bounds it by its
own capacity. The automatic expansion does not run during CUDA graph capture: capture an encode with `actions=` an
expansion that an encode used before the capture (that warm-up imports its columns), or with its columns as torch
tensors; other forms raise `ValueError` before anything is enqueued.

The device refuses horizons and transitions of a numeric table (`"... do not encode numeric state rows"`) and the
Simple graph of a horizon (`"the device does not encode Simple horizons"`). `BatchedEnv.encode_transitions` encodes the
rows of an environment ([rl.md](rl.md)). The [torch op](#the-torch-op) takes node and edge offsets, endpoints and
labels, or parent rows with `action_succ` and expansion labels to build the transitions; the JAX FFI refuses horizons
and transitions because it has no node-offset or edge operands.

## The HGraph view

`encoding.hgraph(batch)` views a **relation-major** batch (`layout='relation'`) as one heterogeneous graph, the form
that relational graph networks take: a node type for every relation of the schema, whose nodes are the relation's
tuples (the atoms), and the node type `_symbol_`, whose nodes are the entities of the graphs. The view and every
array it hands out keep the batch alive and copy no tuple; creating a device view waits once for its metadata.

- **Node types**: `view.node_types` lists the names of the schema's relations in relation order, then `_symbol_`.
  `view.num_nodes(node_type)` counts the nodes in the whole batch (the relation's tuples, or the entities) and
  `view.ptr(node_type)` is the first node of each graph and the end, `[G + 1]`. The atoms of a relation are numbered
  from 0 across the batch, graph after graph, in the order of the batch's tuples (`ptr` is a new array of `G + 1`
  entries). The `_symbol_` nodes are numbered by the batch's entity ids, so `ptr('_symbol_')` is `batch.entity_offsets`
  itself and an edge's entity needs no renumbering. Every entity is a `_symbol_` node: the objects, the auxiliary
  entity and the goal, subgoal, action, history and state entities. The node feature `view.entity_role` is the
  `entity_role` of the batch (made from `num_objects` when the batch's entity arrays are implicit).
- **Edge types**: there is one edge type `(relation, position, '_symbol_')` for every relation and position of its
  tuples (`view.edge_types`; the position is the slot's number as text, the slots of `relation_slot_roles`). Its edge
  `i` leads from tuple `i` of the relation to the entity at that position of the tuple. `view.reverse_edge_types` are
  the same edges the other way round, `('_symbol_', position, relation)`. A nullary relation has one position, its
  auxiliary slot, so its edge leads to the graph's auxiliary entity, which a graph has exactly when it has a nullary
  tuple. A relation of arity 0 (a nullary state relation with `nullary='arity0'`) is a node type without an edge type:
  its atoms are nodes that no edge touches.
- **Edge index**: `view.edge_index(edge_type)` is the pair `(edge_index_0, edge_index_1)` of the edge type (also
  `view.edge_index_0(...)` and `view.edge_index_1(...)`; `view.num_edges(edge_type)` is the length). For a forward
  type, row 0 is the tuple numbers `0 .. n - 1`, made when asked for, and row 1 is the entity at the position of each
  tuple: a **strided view** of `batch.relation_args` (its stride is the relation's arity, read-only on the host), not a
  copy. The reverse type has the rows swapped. A framework that cannot import a strided DLPack tensor copies row 1 on
  import; NumPy and torch share the memory.
- **Other batches**: the view works on the batches of the horizon, rooted-horizon and transition encoders as on those
  of `FlatRelationEncoder`; their action, parent, sibling and cousin relations are node and edge types like the others,
  and their state entities are `_symbol_` nodes.
- **On the device**: of a relation-major `DeviceFlatBatch`, `hgraph` gives a `DeviceHGraphView` with the same types,
  methods and arrays as torch tensors on the batch's device, equal to the host view of the same rows. Row 1,
  `ptr('_symbol_')` and the stored `entity_role` are the batch's own device memory (a strided slice of `relation_args`:
  no copy, no kernel); row 0, a relation's `ptr` and implicit entity roles are made on request by torch on the current
  stream. Making the view reads the graph count, sizes and relations' offsets on the host in one transfer, so it waits
  for the device exactly once; create it outside CUDA graph capture. Its methods never wait.
  `framework='numpy'` copies to the host. Writing into a row writes into the batch. The view keeps the counts it read:
  a batch whose arrays are rewritten in place (a CUDA graph replay) needs a new view.
- **Refusals**: `ValueError` for a graph-major batch (the strided run needs the relation's tuples to be contiguous
  across the graphs), for `predicate_virtual_nodes` (the node type already is the predicate), for shared statics (the
  view needs the static tuples of every graph) and for a batch that overflowed its capacities; `TypeError` for a
  `JaxFlatBatch` (a JAX array has no strided views: every slice is a new array).

```python
relation_encoder = encoding.FlatRelationEncoder(task, layout="relation", include_actions=False)
relation_batch = relation_encoder.encode([task.initial_state])
view = encoding.hgraph(relation_batch)

at = ("at", "1", "_symbol_")              # (relation, position, '_symbol_'): the room of an `at` atom
assert at in view.edge_types and ("_symbol_", "1", "at") in view.reverse_edge_types
atoms, entities = view.edge_index(at)
assert np.shares_memory(entities, relation_batch.relation_args) and entities.strides == (2 * entities.itemsize,)
assert len(atoms) == view.num_edges(at) == view.num_nodes("at")
rooms = set(view.edge_index(("room", "0", "_symbol_"))[1].tolist())
assert set(entities.tolist()) <= rooms
assert view.ptr("_symbol_").tolist() == relation_batch.entity_offsets.tolist()
```

## The Simple graph

`SimpleEncoder(table_or_task, *, <options>, constant_labels=False)` encodes states into an ordinary homogeneous graph
with integer node and edge labels, the form that standard graph neural networks consume (label embeddings with GIN-E,
GATv2, PNA or a plain GCN). It is lossless with respect to the relational encoding: the graph is built from the same
tuples, nothing is merged or renumbered, and a graph holds only its own entities and atoms. `SimpleEncoder` takes the
options of `FlatRelationEncoder` and is one (`outputs=('simple',)`); `encode`, the streams and the capacity form are
the same. Any encoder can store the Simple graph next to the relation arrays with `outputs=('relations', 'simple')`
(`'relations'` alone is the default), and the horizon, rooted horizon and transition encoders take `outputs` as well.

```python
simple = encoding.SimpleEncoder(task, include_actions=False)
simple_batch = simple.encode([state])
full = encoding.FlatRelationEncoder(task, include_actions=False, outputs=("relations", "simple")).encode([state])

# one node per entity, then one per tuple; two edges per tuple position
assert simple_batch.node_offsets[-1] == full.entity_offsets[-1] + full.relation_offsets[-1]
assert simple_batch.edge_offsets[-1] == 2 * full.relation_arg_offsets[-1]
labels = simple.schema.simple_label_names
objects = int(simple_batch.num_objects[0])
assert [labels[i] for i in simple_batch.node_label[:objects]] == ["object"] * objects
assert {labels[i] for i in simple_batch.node_label[objects:]} <= set(simple.relation_names)

# the first `at` atom is joined to its two objects, by the edges 0 and 1 into its node
at = simple.relation_id("at")
atom = int(np.flatnonzero(simple_batch.node_label == len(simple.entity_role_names) + at)[0])
into = simple_batch.edge_index[1] == atom
assert simple_batch.edge_label[into].tolist() == [0, 1]
first = int(full.relation_arg_offsets[at])
assert simple_batch.edge_index[0][into].tolist() == full.relation_args[first : first + 2].tolist()
assert not hasattr(simple_batch, "relation_args") and hasattr(full, "relation_args")
```

**Nodes.** Graph `g` has one node per entity in entity order, so the node of entity `j` is `node_offsets[g] + j`,
followed by one node per tuple in graph-major tuple order (relation id, then tuple order). So `node_offsets[g + 1] -
node_offsets[g]` is the graph's entities plus its tuples. The label of an entity node is its role (`entity_role`: 0 to
8; role 1 is unused without `predicate_virtual_nodes`, which Simple refuses); the label of a tuple's node is
`9 + the relation id`. With `constant_labels=True` an object that is a constant of the domain has the label `9 + R + c`
instead of 0, `c` the constant's index in the domain's constant list (`schema.simple_constants` of them), so that the
label means the same object in every problem of the domain. `schema.simple_label_names` lists the role names, then the
relation names, then the constant names; its length is the vocabulary size of `node_label`.

**Edges.** For every tuple `t` (its node `a`) and every position `k` of its relation (the encoded arity, so a nullary
atom has one position, its auxiliary entity, and none with `nullary='arity0'`, where its node has no edges), the entity
`e` at that position gives two edges: `e -> a` with the label `k` and `a -> e` with the label `K + k`, `K` the largest
encoded arity of a relation. A graph's edges are all its forward edges in (tuple, position) order, then all its reverse
edges in the same order, so `edge_label % K` is the position (`relation_slot_roles` interprets it) and
`edge_offsets[g + 1] - edge_offsets[g]` is twice the graph's tuple arguments. `schema.simple_edge_label_names` lists
`'0' .. 'K-1'` and `'0^-1' .. '(K-1)^-1'`. Static, type, goal (with their derivation), subgoal and history atoms are
tuple nodes like any other, and target, action, history and state entities are entity nodes joined through their
slots, so a goal literal's status is in its label, for example `[+]on[g][unsat]`.

| Array | dtype, shape | Meaning |
|---|---|---|
| `node_offsets` | I `[G + 1]` | graph `g`'s nodes are `[node_offsets[g], node_offsets[g + 1])`: `entity_offsets[g] +` the tuples of the graphs before it |
| `node_label` | int32 `[V]` | the label of each node |
| `node_value` | float32 `[V]` | with numeric relations: the value of each numeric atom's node (NaN for a term without a value), 0 for every other node |
| `edge_offsets` | I `[G + 1]` | graph `g`'s edges are `[edge_offsets[g], edge_offsets[g + 1])`: twice the tuple arguments before it |
| `edge_index` | I `[2, E]` | row 0 the source, row 1 the target node of each edge (batch-global node ids) |
| `edge_label` | int32 `[E]` | the label of each edge |

A Simple-only batch has `task_ids`, `num_objects`, `entity_offsets`, `entity_role` and `entity_ref` (to map a node
back to its entity), `goal_complete` and the target and history arrays when its schema has them, the scalars of every
batch, and `simple_node_labels` and `simple_edge_labels` (the vocabulary sizes), and none of the `relation_*` arrays;
with `outputs=('relations', 'simple')` it has them as well. The Simple arrays are the same in both layouts, and
encoding rows alone, in a batch, in a stream or with capacities gives the same graphs: a graph encoded alone is the
slice of the batch's arrays with the node ids shifted by `node_offsets[g]`. With
capacities the Simple arrays have `entities + args` nodes and `2 * args` edges; the first `sizes['entities'] +
sizes['tuples']` nodes and `2 * sizes['args']` edges hold the graphs and the rest is zero. `edge_index` is then
`[2, 2 * args]`.

**Refusals.** `predicate_virtual_nodes` (a tuple's node already names its predicate), `entity_arrays='auto'` (the
graph keeps the entity arrays) and `statics='shared'` (the graph holds the static tuples among its own) raise
`ValueError` with Simple, and so do `layout='relation'` and `emit_counts=False` without the `'relations'` output, since
the batch has no relation arrays for them to shape. `constant_labels` needs the `'simple'` output and the constants of
the domain: a task read from mymyr's text format does not record them, so a table with such a task raises `ValueError`
with `constant_labels=True` (a PDDL domain without constants has none to label). `entity_names` and `describe` read the
relation arrays and raise `ValueError` for a Simple-only batch, and `hgraph` views relation arrays.

Rows in CUDA memory give the same Simple arrays ([Encoding on the device](#encoding-on-the-device)), for both index
types, in exact and capacity calls, the torch op, device row streams and collation; the fill kernels write the Simple
graph from the tuples they fill, and a Simple-only batch fills them into private storage that it does not keep. The
device refuses the Simple graph with numeric relations (`node_value`), and the JAX FFI refuses the Simple graph: its
fixed result layout has no node or edge arrays.

```python needs-torch_geometric
import torch_geometric

states = [task.initial_state] + task.successor_states(task.initial_state)[:2]
torch_batch = encoding.SimpleEncoder(task, include_actions=False).encode(states, framework="torch")

graphs = []
for g in range(torch_batch.num_graphs):
    nodes = slice(int(torch_batch.node_offsets[g]), int(torch_batch.node_offsets[g + 1]))
    edges = slice(int(torch_batch.edge_offsets[g]), int(torch_batch.edge_offsets[g + 1]))
    graphs.append(torch_geometric.data.Data(
        x=torch_batch.node_label[nodes].long(),
        edge_index=torch_batch.edge_index[:, edges].long() - int(torch_batch.node_offsets[g]),
        edge_attr=torch_batch.edge_label[edges].long()))
collated = torch_geometric.data.Batch.from_data_list(graphs)
assert collated.num_graphs == 3 and collated.x.shape == (int(torch_batch.node_offsets[-1]),)
```

## Numeric functions

A numeric task is encoded like any other: its `State`s are packed, and its rows `[N, W + NN]`
(`table.initial_states()`, `rl.expand`, `rl.CpuEnvPool`) are read as they are. For every **fluent** numeric function
`f` of the domain (a function that effects change and that states give values to; `total-cost` is no part of a state,
and a static function never changes) the schema has a relation `[num]f` (lane `numeric`). The numeric relations follow
all other relations of the catalog, sorted by name among themselves, so a schema without numeric functions has the
relation ids it would have without the lane. It has one tuple for each numeric slot of `f` in the instance, over the
objects of the ground term `(f o1 ... ok)`, or over the auxiliary entity for a nullary function, in the lexicographic
order of the objects. With `nullary='arity0'` the tuple of a nullary function has no argument; `nullary='omit'` leaves
out the nullary predicates, not the nullary functions. The tuples are the same in every graph of an instance: a state
changes their values, not their number.

`numeric_values` holds the value of every numeric tuple of the batch, in the order in which the numeric tuples appear
in `relation_args` (graph after graph in the graph layout, relation after relation in the relation layout), as
`float32` (`numeric_dtype='float32'`, the default) or `float64`; a function term without a value is NaN. It is present
iff the schema has numeric relations and the `'relations'` output; with the [Simple graph](#the-simple-graph) the
values are in `node_value`. Values stored as `int32` pairs (`task.numeric_storage == 'i32'`) are decoded exactly, so
the values are those of `state.numeric_values()` and `rl.numeric_values`. A table has one set of functions: all its
instances declare the same ones, and an instance without slots of `f` has no tuples in its relation.

```python
counters = mymyr.Task.from_text("tests/data/numeric_tasks/cs-counters.txt")  # four counters, value(o0..o3)
numeric_encoder = encoding.FlatRelationEncoder(counters)
first = counters.initial_state
second = counters.successor_states(first)[0]
numeric_batch = numeric_encoder.encode([first, second])

value = numeric_encoder.relation_id("[num]value")
assert numeric_encoder.relation_lanes[value] == "numeric" and numeric_batch.relation_counts[0, value] == 4
assert np.array_equal(numeric_batch.numeric_values, np.concatenate([first.numeric_values(), second.numeric_values()]))
assert numeric_batch.numeric_values.dtype == np.float32

# the tuple of a slot is the term's object, the value comes with it
args = np.asarray(numeric_batch.relation_args)[numeric_batch.relation_arg_offsets[value]:][:4]
assert [counters.objects[a - numeric_batch.entity_offsets[0]] for a in args] == ["o0", "o1", "o2", "o3"]

# float64 values, and nullary functions: tuples over the auxiliary entity, or without one under nullary='arity0'
drone = mymyr.Task.from_text("tests/data/numeric_tasks/cs-drone.txt")
wide = encoding.FlatRelationEncoder(drone, numeric_dtype="float64")
assert wide.encode([drone.initial_state]).numeric_values.dtype == np.float64
assert wide.relation_names.count("[num]battery-level") == 1
battery = wide.relation_id("[num]battery-level")
flat = encoding.FlatRelationEncoder(drone, numeric_dtype="float64", nullary="arity0")
assert wide.relation_arities[battery] == 1 and flat.relation_arities[flat.relation_id("[num]battery-level")] == 0
flat_batch, wide_batch = flat.encode([drone.initial_state]), wide.encode([drone.initial_state])
assert np.array_equal(flat_batch.numeric_values, wide_batch.numeric_values)
assert int(flat_batch.entity_offsets[-1]) == int(wide_batch.entity_offsets[-1]) - 1
```

`include_numeric=False` leaves the numeric relations and `numeric_values` out (the rows of a numeric table still carry
their numeric words). The numeric lane is the one lane that reads the numeric part of a row: goals, subgoals, history
and actions are over atoms, and a numeric constraint of the goal is no goal literal (`goal_complete` is 0 for a goal
with one). The numeric relations are never static, so `statics='shared'` leaves them in the graphs. Horizon encoders
refuse a domain with fluent numeric functions.

The device ([Encoding on the device](#encoding-on-the-device)) encodes numeric rows into the same tuples, order and
values as the CPU, for both layouts, both index types and both numeric dtypes, in exact and capacity calls, the torch
op, device streams and collation; the conversion of a value to `float32` rounds to nearest, ties to even, on both
processors. Capacity calls of a schema that stores `numeric_values` take the capacity `numeric`. The JAX FFI refuses
numeric tables, with or without `include_numeric`: its result layout has no numeric buffer.

## Shared statics

The static tuples of an instance are the same in every state, yet by default every graph holds its own copy: a batch of
a thousand states of one gripper task repeats the `ball`, `room` and `gripper` tuples a thousand times.
`statics='shared'` stores them once per instance of the batch instead. The static relations (the relations of lane
`static`, as `schema.relation_lanes` tells) are then empty in every graph (`relation_counts` is 0 in their cells), and
five more arrays hold the tuples. With `S` static relations (`R_s = S`) and `I_b` distinct instances in the batch:

| Array | dtype, shape | Meaning |
|---|---|---|
| `static_task_ids` | int32 `[I_b]` | the instances of the batch, ascending |
| `static_relations` | I `[R_s]` | the relation ids of the static relations, ascending |
| `static_offsets` | I `[I_b * R_s + 1]` | the tuples of the `k`-th static relation in the block of the `b`-th instance are the segment `b * R_s + k` |
| `static_arg_offsets` | I `[I_b * R_s + 1]` | the same offsets into `static_args` |
| `static_args` | I `[A_s]` | the entities of the static tuples, tuple after tuple, `relation_arities[r]` per tuple |

The blocks come instance after instance in ascending order of the instance, relation after relation in the order of
`static_relations`; the tuples of a cell are in the order they have in the repeated form, so the arrays are
deterministic and do not depend on the threads, the pool or the order of the rows (a batch that permutes its rows gives
the same blocks). The blocks of a batch are those of its distinct instances: a slice of the rows has the blocks of
the instances it holds, so `block(task_ids[g])` is `searchsorted(static_task_ids, task_ids[g])`.

The entities in `static_args` are **graph-local**, because the block belongs to the instance and not to a graph: an
object is its instance-local id (`task.objects[id]`), the auxiliary entity of a nullary tuple is `num_objects` (a
nullary tuple has no argument with `nullary='arity0'`), and the predicate slot of a predicate relation (with
`predicate_virtual_nodes`) holds the predicate's key, the `entity_ref` of its virtual node. A model that wants the
repeated form adds `entity_offsets[g]` to the object and auxiliary ids of every graph `g` that uses the block of
`task_ids[g]`, and looks the key up among the virtual nodes of `g`. A graph's entities are the same as in the repeated
form: an object, auxiliary or virtual node that only a static tuple uses still exists in the graph. The `entity_ref` of
a goal, subgoal or history target entity is its tuple index among the graph's own tuples, which in the shared form does
not count the static tuples. Subgoal layers and history steps need nothing of the static blocks: their relations are not
static.

```python
shared = encoding.FlatRelationEncoder(task, statics="shared")
repeated = encoding.FlatRelationEncoder(task)
states = [task.initial_state, task.successor_states(task.initial_state)[0]]
small, whole = shared.encode(states).as_dict(), repeated.encode(states).as_dict()

arities = shared.schema.relation_arities
k = 0                                               # the first static relation
relation = int(small["static_relations"][k])
for g in range(small["num_graphs"]):
    block = int(np.searchsorted(small["static_task_ids"], small["task_ids"][g]))
    lo, hi = small["static_arg_offsets"][block * len(small["static_relations"]) + k :][:2]
    local = small["static_args"][lo:hi].reshape(-1, arities[relation])
    rebuilt = local + small["entity_offsets"][g]    # no predicate slots without predicate_virtual_nodes
    cell = g * small["num_relations"] + relation
    lo, hi = whole["relation_arg_offsets"][cell : cell + 2]
    assert np.array_equal(rebuilt, whole["relation_args"][lo:hi].reshape(-1, arities[relation]))
    assert small["relation_counts"][g, relation] == 0   # the graph's own cell is empty
```

The rest of the batch follows the same rules as a repeated one:

- **Layouts and index types** are those of the repeated form; the static arrays are in the batch's index type.
- **Slices.** Encoding a slice of a shared batch gives the slice of its graphs, rebased as in the repeated form, with
  the blocks of the instances the slice holds. The block of an instance is the same in the slice and in the batch.
- **Capacities** take the keys `static_instances` and `static_args`
  ([Capacities and buffer reuse](#capacities-and-buffer-reuse)).
- **Streams** flush the same batch as `encode` over the rows; [collation](#collation) of host batches gives the batch
  of one encode of all their rows, blocks included.
- **Names.** `entity_names` and `describe` give the same names and tuples as for the repeated batch; the static tuples
  of a graph are listed in their relations, in the order of the repeated form.
- **Validation.** The C++ `validate` also checks the blocks: the instances, the offsets, the entities against the
  instance's objects and the sort order of the tuples.
- **Refusals.** Horizons, the Simple graph and the HGraph view refuse shared statics (their graphs hold the static
  tuples among their own), and so does the JAX FFI. The device encodes them
  ([What the device encodes](#what-the-device-encodes)).

## Capacities and buffer reuse

**`capacities=None`** (the default) sizes every array to the batch. A dict gives the capacity of the variable-size
arrays instead, so that batches of different sizes share array shapes (for example under `jax.jit` or a fixed-shape
model):

| Key | Required | Bounds |
|---|---|---|
| `graphs` | always | the number of input rows |
| `entities` | always | `entity_role`, `entity_ref` (the Simple graph: `entities + args` nodes) |
| `args` | always | `relation_args` (the Simple graph: `2 * args` edges) |
| `targets` | iff the schema has targets | `target_entity`, `target_group` |
| `history` | iff the schema has history | `history_entity`, `history_dt` |
| `numeric` | iff the batch stores `numeric_values` | `numeric_values` |
| `static_instances`, `static_args` | iff `statics='shared'` (refused otherwise) | the static blocks (`I_b`) and `static_args` (`A_s`) |

The offset arrays keep their sizes (`G + 1`, `G * R + 1`, `G` the rows of the call). Exceeding the graph capacity
raises before encoding; tuple counts have no separate capacity because every tuple has at least one argument. The batch
fills the first entries of each array and the rest is zero (`static_offsets` and `static_arg_offsets` repeat their last
value). `batch.sizes` holds the true sizes: `entities`, `tuples`, `args`, `targets`, with history `history`, with
numeric values `numeric`, with shared statics `static_instances`, `static_tuples` and `static_args`. A batch that does
not fit (a capacity too small, or an index type too narrow) leaves every array zero and sets `sizes['overflow']`;
`sizes['capacity_overflow']` and `sizes['index_overflow']` tell which. Nothing raises.

```python
fixed_encoder = encoding.FlatRelationEncoder(task)
capacities = dict(graphs=8, entities=64, args=512)
one = fixed_encoder.encode([state], capacities=capacities)
three = fixed_encoder.encode([state] * 3, capacities=capacities)
assert one.relation_args.shape == three.relation_args.shape == (512,)
assert three.entity_offsets.shape == (4,) and three.sizes["entities"] == 3 * one.sizes["entities"]
exact = fixed_encoder.encode([state])
assert np.array_equal(one.relation_args[: one.sizes["args"]], exact.relation_args) and not one.sizes["overflow"]

tight = fixed_encoder.encode([state] * 3, capacities=dict(graphs=8, entities=16, args=512))
assert tight.sizes["capacity_overflow"] and not tight.relation_args.any()
```

**`into=`** writes the encoding into an earlier batch instead of new memory: a `FlatBatch` that a call with the same
capacities made for the same number of rows of this schema and table. Every array of the batch is rewritten as a fresh
encode would make it (the Simple graph, the numeric values and the static blocks of the call's instances included), and
the call returns the batch with its `sizes` updated. It needs capacities and host rows, and raises `ValueError` before
writing if the batch does not fit the call. Reuse aliases: every array and tensor exported from the batch before the
call shares the memory that the call overwrites, and nothing waits for readers or copies on write, so use it once the
batch's previous contents are no longer read ([Reusing a batch's memory](rl.md#reusing-a-batchs-memory) shows it on
an environment pool).

**Memory of batches.** A batch is one block of memory. Fresh memory costs a kernel page fault per 4 KiB page, which
for a batch of tens of megabytes is several times the work of filling it, so released blocks are kept and handed out
again with their pages still mapped. The same cache serves the other batch outputs (`mymyr.rl.expand`, table batches).
It is bounded:

| Block size | Served by | Bound |
|---|---|---|
| up to 128 KiB | the system allocator | not cached by mymyr |
| up to 4 MiB | a cache private to the thread that releases the block | 4 blocks per thread |
| above 4 MiB | a cache shared by all threads | 8 blocks and 512 MiB in all; a block above 256 MiB is never kept |

A block returns to the cache of the thread that drops its last reference, whichever thread made it. Blocks are
anonymous memory mappings, so dropping one gives its memory back to the operating system. A shared block that no
acquire takes within the 256 acquires of blocks above 128 KiB that follow its release is dropped by itself, so a one-off
huge batch does not stay resident. `mymyr.block_cache_bytes()` reports what the calling thread's cache and the shared
cache hold, and `mymyr.trim_block_cache()` frees all of it; other threads' private caches are freed when those threads
end. Arrays that a batch still owns keep their memory either way.

```python
big = fixed_encoder.encode([state] * 20000)  # a batch of several megabytes
del big  # its block goes back to the cache
assert mymyr.block_cache_bytes() > 0
mymyr.trim_block_cache()  # and back to the system
assert mymyr.block_cache_bytes() == 0
```

## Streams and collation

Rows that arrive one by one (a search expanding states, an actor stepping several environments) are encoded in one call
by collecting them in a stream. `encoder.stream(*, framework=None)` returns an append-only `FlatStream`,
`encoder.mutable_stream(*, framework=None)` a `FlatMutableStream` whose rows have indices you choose. A stream copies
the rows it is given, so the caller's arrays can be reused at once, and `flush` encodes all rows it holds in a single
`encode` call: the batch equals `encoder.encode(rows, ...)` over the same rows bit for bit, whatever the way the rows
arrived.

```python
stream_encoder = encoding.FlatRelationEncoder(task, target_sources=("goal", "action"))
table = rl.as_table(task)
states = [task.initial_state]
for _ in range(3):
    states.append(task.successor_states(states[-1])[0])

stream = stream_encoder.stream()
stream.append(states[:2], actions=rl.expand(table, states[:2]))
stream.append(states[2:], actions=rl.expand(table, states[2:]))
assert len(stream) == 4
flushed = stream.flush()                     # the stream is empty again
assert len(stream) == 0 and flushed.num_graphs == 4
expected = stream_encoder.encode(states, actions=rl.expand(table, states))
assert np.array_equal(flushed.relation_args, expected.relation_args)
assert np.array_equal(flushed.entity_ref, expected.entity_ref)

mutable = stream_encoder.mutable_stream()
mutable.append([7, 3], states[:2], actions=rl.expand(table, states[:2]))
mutable.update(3, states[2], actions=rl.expand(table, states[2:3]))
mutable.remove(7)
assert list(mutable.indices()) == [3] and 3 in mutable
assert mutable.flush().num_graphs == 1       # the rows are kept
assert len(mutable) == 1
```

- **`FlatStream.append(states, task_ids=None, *, derived=None, goals=None, actions=None, subgoals=None,
  history=None)`** takes the arguments of `encode` and adds the rows. `flush(*, framework=None, threads=1, pool=None,
  capacities=None)` takes the remaining arguments of `encode`, encodes the rows in the order of the appends and
  **empties** the stream. A flush that raises, or whose batch overflowed its `capacities`, keeps the rows. `clear()`
  drops the rows, `len(stream)` counts them, `stream.encoder` is the encoder.
- **`FlatMutableStream`** stores each row under an index: `append(indices, states, task_ids=None, *, ...)` adds rows
  under new indices, `update(indices, states, ...)` replaces the rows under held ones (with all their inputs), and
  `remove(indices)` drops them. An index is an int, an integer array or a sequence of ints; any 64-bit integer,
  negative ones included. `flush` encodes the rows in **ascending order of their indices** and **keeps** them, so
  graph `j` of the batch is the row of `indices()[j]`; `index in stream` tells whether a row is held. An index that is
  repeated within a call or already held raises `ValueError` on `append`; an index that is not held raises `KeyError`
  on `update` and `remove`. A call that raises changes nothing.
- **Subgoals and history.** `subgoals` and `history` are as in `encode` and are optional per row: a row without them
  has empty layers and no history, the same batch as `encode` gives for it. A stream stores `max_goal_level` layers per
  row and the valid steps of each history, and a flush pads the histories to the longest, so rows with histories of
  every length can share a stream.
- **Per-row inputs.** `derived`, `goals` and `actions` are as in `encode`, with one rule: every row of a stream gives
  the same inputs. The first rows appended to a stream without rows decide (a stream with `goals` keeps a goal for
  every row, a stream with `actions` the action rows of every state, a single `[1, W]` goal is copied for every row),
  and a later `append` or `update` that gives other inputs raises `ValueError`. A stream that holds no rows again takes
  any inputs. Rows narrower than their instance are refused when appended, with the same errors as `encode`; errors
  that need the encoder (a goal bit with no atom slot, action labels of another instance, an action order that is not
  canonical) are raised by `flush`, which keeps the rows.
- **Action rows.** The action rows of the appended expansions are copied and joined in row order, so a flushed batch
  is that of one expansion over all rows: the `entity_ref` of an action entity is what `encode` gives for the same
  states and action rows. The row of an action entity in the expansion it was appended with is `x.offsets[k] +
  entity_ref`, where `x` is the expansion given to the `append` (or `update`) that stored the state and `k` is the
  state's row in that call, so `x.action(x.offsets[k] + entity_ref)` is the entity's action. The flushed batch does not
  number its graphs as `x` does: graph `j` of a flush is the `j`-th row appended, and of a mutable flush the row of
  `indices()[j]`.
- **Threads.** The methods of one stream take turns (a flush holds the stream until its batch is encoded); a stream is
  not a way to encode concurrently, use one stream per producer or `encode` itself.
- **Shared statics** and **capacities** work as in `encode`; batches and streams are not picklable.

### Collation

`encoding.collate(batches)` concatenates host or CUDA batches in list order. Empty slices retain their schema; the
list must contain at least one batch. The result equals one encode of all rows in the same order. Entity references
stay local to their graph; entity ids and offsets receive the appropriate prefixes. Relation-major cells are
interleaved by relation across the slices. Mixed schemas, tables (`table_fingerprint`), layouts, index types,
static representations or devices raise `ValueError`; so does a device batch of an encoder with subgoal layers or
history steps. Host collation also takes batches with [shared statics](#shared-statics): the result holds one static
block per instance of its graphs, in ascending instance order, as one encode of all rows does; slices whose static
relations differ, or whose blocks of one instance differ in their tuples or arguments, raise `ValueError`. It takes
`numeric_values` (each value follows its tuple into the result's segment order; the slices agree on
`numeric_relations` and `numeric_dtype`) and the [Simple graph](#the-simple-graph) (the node ids in `edge_index` are
shifted by the nodes of the batches before).
Overflowed inputs are refused. Host collation allocates exact arrays; device collation accepts
`capacities=` and reports overflow through `device_sizes`. All accumulated totals use uint64 before index checks.

## Encoding on the device

Pass CUDA rows to `encoder.encode` to run the encoder on their device and receive a `DeviceFlatBatch` of torch
tensors. This needs a CUDA build of mymyr and torch. The arrays have the host batch's names, dtypes, ordering and
values, for both layouts and index widths. Inputs must share one device. State words may be `uint64`/`int64`
`[N, W]` or `uint32`/`int32` halves `[N, 2 W]`; task ids are contiguous `int32` `[N]`.
DLPack and CUDA array interface inputs are imported in place. `framework` is `None` or `"torch"`; `pool` and `into`
are refused for device rows, and `threads` has no effect.

`literal_slots` and `literal_masks` also accept CUDA torch tensors, with an encoder as their source. They resolve
table domain predicate ids (`table.predicate_names`; `schema.relation_predicates` maps decoded relation ids to
predicates) and instance-local objects with the same semantics as the host helpers, under both `reachable` and
`dense` atom layouts. They allocate through torch on the current stream and perform no host synchronization.

| Argument or result | CUDA tensor |
| --- | --- |
| `predicates`, `objects` | contiguous int32 `[K]`, `[K, k]`, objects padded with -1 |
| `positive`, `rows`, optional `layers` | bool `[K]`, int32 `[K]`, int32 `[K]` |
| `task_ids` | int32 `[K]` for slots, `[num_rows]` for masks; required for several instances |
| slots result | `(kinds, slots, status)`: uint8 `[K]`, int32 `[K]` (-1 without a slot), int64 scalar |
| mask result | `GoalMasks` or `LayerMasks`, int64 words, bool `unsatisfiable`, int64 scalar `status` |

Mask shapes come from the host integer `num_rows`, the table's word counts, and an explicit host integer
`num_layers` when `layers` is supplied.
`word_dtype=torch.uint32` returns low/high halves of each word. CUDA masks require `fixed="decide"`: static and
unreachable literals that fail set `unsatisfiable`; those that hold contribute no bit. Invalid literals, task ids,
rows or layers are skipped and OR value 4 into `status`; reading that scalar is the caller's choice. Host masks
raise for invalid inputs and have `status=None`.

```python needs-torch skip-if-no-cuda
import torch
from mymyr import rl

predicate = task.atom(0)
predicates = torch.tensor([rl.as_table(task).predicate_names.index(predicate.predicate.name)],
                          dtype=torch.int32, device="cuda:0")
objects = torch.tensor([list(predicate.object_indices)], dtype=torch.int32, device="cuda:0")
encoder = encoding.FlatRelationEncoder(task)
kinds, slots, status = encoding.literal_slots(encoder, predicates, objects)
goals = encoding.literal_masks(encoder, predicates, objects,
                               torch.ones(1, dtype=torch.bool, device="cuda:0"),
                               torch.zeros(1, dtype=torch.int32, device="cuda:0"), num_rows=1)
```

Warm the helper on the capture stream before CUDA graph capture to upload and acquire its context. Keep the
encoder, input buffers and outputs alive through the last replay. Importing `mymyr.rl.torch` also registers
`torch.ops.mymyr.literal_slots` and `literal_masks`, with fake implementations for `torch.compile(fullgraph=True)`.
Their first argument is a registered encoder handle. The mask op returns six tensors: positive and negative fluent
masks, positive and negative derived masks (width zero without derived slots), `unsatisfiable`, and `status`.

```python needs-torch skip-if-no-cuda
import torch

encoder = encoding.FlatRelationEncoder(task)
rows = np.asarray(task.encode([task.initial_state, task.successor_states(task.initial_state)[0]]))
device_rows = torch.as_tensor(rows.view(np.int64).copy(), device="cuda:0")
batch = encoder.encode(device_rows)
assert batch.relation_args.is_cuda
assert np.array_equal(batch.as_dict("numpy")["relation_args"], encoder.encode(rows).relation_args)

capacities = dict(graphs=2, entities=256, args=2048)
fixed = encoder.encode(device_rows, capacities=capacities)
assert fixed.relation_args.shape == (2048,)
assert fixed.nbytes_by_array["relation_args"] == 2048 * 4
assert fixed.nbytes == sum(fixed.nbytes_by_array.values())  # metadata only, no device read
assert not fixed.sizes["overflow"]                       # explicitly reads device status
```

### What the device encodes

| Output or input | Torch device path | JAX FFI, CPU and CUDA |
|---|---|---|
| Relations, goals, derived atoms, subgoal layers, history, targets | Yes | Yes |
| Graph/relation layouts, int32/int64 indices, omitted counts/entities | Yes | Yes; int64 needs `jax_enable_x64` |
| Numeric relations and values | Yes | Numeric tables refused |
| Simple alone or beside relations | Yes; refused with numeric relations (`include_numeric=False` encodes it) | Refused |
| Shared static blocks | Yes, including streams and collation | Refused |
| Supplied action labels and action targets | Yes, relational output on classical tables | Refused |
| Horizons, rooted horizons, transitions | Yes, classical relational output | Refused |
| HGraph view | Relation-major batches, created outside capture | View a host or torch batch |

[Action inputs](#actions), [subgoals](#subgoals), [history](#history) and [horizons](#horizons-on-the-device) use
their host forms with device arrays. Missing derived words are evaluated on the encoding stream when needed. Supply
`derived=` for instances whose axioms exceed the device evaluator's limits; exact calls otherwise raise, and capacity
calls flag the affected rows as invalid. JAX checks the whole table when tracing and requires supplied words if any
instance needs them. A schema exceeding a kernel's shared-memory budget is refused before launch.

### Capacities, status and synchronization

An exact call counts its outputs, downloads the counts and waits before allocating and filling exact arrays. It
cannot run during CUDA graph capture. [Capacities](#capacities-and-buffer-reuse) allocate variable arrays at their
host-known bounds and keep counts and status on the device. After context setup, the capacity path has no
device-to-host wait. Ordinary calls use
`N` graph slots even if `capacities['graphs']` is larger. Live stream and suite batches reserve slots for every
possible input row and expose their current count as `device_graphs`.

`device_sizes` is `int64 [6]`: entities, tuples, args, targets, history and status; numeric batches append a seventh
word, the numeric value count. Shared statics add `device_static_sizes`, the three counts instances, tuples and args.
The status is a bitmask, with bits that may be combined:

| Value | Meaning | Host `sizes` key |
|---|---|---|
| 1 | An output or upstream action/ring capacity was exceeded | `capacity_overflow` |
| 2 | An index does not fit the chosen index dtype | `index_overflow` |
| 4 | Invalid task id, row/mask width or atom bit, history count, action label or horizon structure | `invalid_input` |

Any nonzero status makes `sizes['overflow']` true. Ordinary encodes zero every output array on failure; rings have
the separate tail contract below. Successful ordinary capacity calls
zero unused variable tails; shared static offset tails repeat the closing offset. Shape/type errors and graph counts
above the host-provided graph capacity raise before encoding. Status stays on the device until explicitly read.

| Access | Host wait |
|---|---|
| Array properties, `device_sizes`, `device_graphs`, `nbytes`, `nbytes_by_array` | None |
| `sizes`, `num_graphs` with a live device count | Reads device counters |
| `as_dict()` of a live batch | Reads counts and returns compact live arrays |
| `as_dict('numpy')`, `entity_names`, `describe` | Copies requested data to the host |
| Creating a device HGraph view | One metadata transfer; recreate it after rewriting the backing batch |

### Streams and CUDA graphs

Torch encodes allocate and enqueue work on torch's current stream for the input device. Operations on that stream
are ordered. When another stream produced an input or consumes an output, establish a dependency with
`wait_stream` or an event, as with torch tensors. A device row ring and a prepared collator must be used on the
stream that constructed them. JAX FFI kernels use XLA's stream.

Warm a capacity call on the capture stream to upload the encoder's context before recording. Keep the encoder,
input arrays and their owners, and output tensors alive through completion of the final replay. Replay reads
the captured addresses: update inputs in place. Host exports, exact encoding and HGraph construction belong outside
capture. The capacity encoder allocates scratch through torch; the graph's private pool retains captured scratch.
Exact encodes reuse scratch per stream; `encoder.reserve_device(graphs, device=..., stream=...)` grows it in advance
and `encoder.release_device(...)` releases it. Reserve/release outside capture with no outstanding uses.

```python needs-torch skip-if-no-cuda
capture_stream = torch.cuda.Stream()
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    encoder.encode(device_rows, capacities=capacities)
capture_stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    captured = encoder.encode(device_rows, capacities=capacities)
    captured_bytes = captured.nbytes
with torch.cuda.stream(capture_stream):
    device_rows.copy_(device_rows.flip(0))
    graph.replay()
capture_stream.synchronize()
assert captured_bytes == captured.nbytes
assert not captured.sizes["overflow"]
```

Captured action expansion has its own [workspace ownership and fixed-shape requirements](rl.md#encoded-observations).

### The torch op

`torch.ops.mymyr.encode_flat` is the capacity encoder as a custom op. Import `mymyr.rl.torch` to register it and
`register(encoder)` to obtain its encoder handle. Its signature lists all 16 optional input tensors in order:

```text
torch.ops.mymyr.encode_flat(
    encoder, states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg,
    subgoal_pos, subgoal_neg, subgoal_derived_pos, subgoal_derived_neg,
    history_pos, history_neg, history_dt, history_count, history_derived_pos, history_derived_neg,
    graphs, entities, args, targets, history, numeric=0, static_instances=0, static_args=0, *,
    action_schema=None, action_binding=None, action_offsets=None, action_parent=None,
    node_offsets=None, edge_offsets=None, edge_src=None, edge_dst=None,
    edge_schema=None, edge_binding=None, action_succ=None, expand_actions=False)
```

Absent input tensors are `None`. The capacities are `graphs, entities, args, targets, history`, with optional
`numeric`, `static_instances` and `static_args`; unused capacities are zero. Optional keyword tensors supply
`action_schema`, `action_binding`, `action_offsets`, `action_parent`, horizon node/edge columns or `action_succ`
for transitions. `expand_actions=True` requests bounded observation actions and takes no explicit action/horizon
arrays. Its fake implementation derives shapes from metadata, so `torch.compile(fullgraph=True)` can trace it.

The result list is ordered by `DeviceFlatBatch.array_names`: output arrays, `static_sizes`, then `sizes`.
Absent arrays are empty tensors. Wrap it with `DeviceFlatBatch.from_arrays`. The handle registry owns the encoder
until `unregister(handle)`; keep the handle registered while compiled functions or CUDA graphs can call it.
A compiled function specializes the handle: calling it with another registered encoder traces a graph for that
encoder's schema.

```python needs-torch skip-if-no-cuda
import mymyr.rl.torch as rt

handle = rt.register(encoder)
outputs = torch.ops.mymyr.encode_flat(
    handle, device_rows, task_ids=None, derived=None, goal_pos=None, goal_neg=None,
    goal_derived_pos=None, goal_derived_neg=None, subgoal_pos=None, subgoal_neg=None,
    subgoal_derived_pos=None, subgoal_derived_neg=None, history_pos=None, history_neg=None,
    history_dt=None, history_count=None, history_derived_pos=None, history_derived_neg=None,
    graphs=2, entities=256, args=2048, targets=0, history=0)
op_batch = encoding.DeviceFlatBatch.from_arrays(encoder, outputs)
assert op_batch.nbytes == fixed.nbytes
assert not op_batch.sizes["overflow"]
rt.unregister(handle)
```

### Device row streams and collation

`encoder.device_stream(capacity, device=..., goals=False, derived=False, derived_goals=False)` makes a
`DeviceFlatStream` ring on the current stream. Choose optional columns at construction; every append supplies those
columns. The ring copies rows, task ids and optional masks. It accepts shared statics and Simple output, but no
subgoal layers or history. Mutable streams are host-only.

```python needs-torch skip-if-no-cuda
ring = encoder.device_stream(16, device=0)
ring.append(device_rows)
stream_batch = ring.flush(capacities=dict(graphs=16, entities=512, args=4096))
assert stream_batch.device_graphs is not None
assert stream_batch.nbytes == sum(stream_batch.nbytes_by_array.values())
assert stream_batch.num_graphs == 2
```

An append either copies every row or sets sticky `device_overflow` without changing the ring. `clear()` clears both
rows and overflow. Flush encodes the live rows, advances the ring and empties it; graph capacity must cover the ring.
`device_status` is int64 `[head, live_rows, overflow]`. `len(ring)` waits; `device_graphs` does not. Per-graph and
per-cell tails are zero, but **variable tails are unwritten**: read only the live sizes/offsets. A ring overflow
zeroes every per-graph output and reports capacity overflow. Construct the ring outside capture; append and capacity
flush can be recorded. Keep the ring, inputs and returned batches alive through the last replay's completion.

`encoding.collate(batches, capacities=...)` accepts CUDA batches; subgoal/history schemas are refused. Prepare
`encoding.DeviceCollator(batches)` outside capture for repeated collation, then call `collator.encode(capacities=...)`
on its construction stream. The collator fixes each input's graph count at preparation (reading a live count waits
then); replay uses those counts. Keep the collator, source tensors and outputs alive. Capacity collation has no
host wait and zeroes unused tails; exact collation downloads output sizes. Shared blocks are checked for consistent
instance contents. See [Collation](#collation) for schema/table checks.

### Several domains and resident datasets

`SuiteEncoder.encode(cuda_rows, global_task_ids)` groups rows stably on the device and returns a `SuiteBatch`: one
batch per domain, plus device `domain`, `index`, `rows` and global `task_ids` mappings. Each domain batch uses its
local instance ids. Exact multi-domain calls download the per-domain row counts once, omit empty domains and then
wait for each domain's exact encoding sizes. A single-domain call knows its row count without a grouping transfer.

For asynchronous encoding, pass a capacity dict per domain, with **each** `graphs >= N`, the whole input batch size.
Every domain is returned, including empty ones. Each batch has `N` graph slots, a live `device_graphs` count and zero
unused tails; `rows[d]` is -1 past the live prefix. Invalid global ids are grouped with instance 0's domain and flag
that batch's `invalid_input`; host calls raise instead. Capacity calls and their mappings remain resident and can be
captured after warm-up. `actions=True` uses bounded device expansion; supplied per-domain expansions require exact
calls. Automatic actions require an asynchronous expansion backend for every domain, on both exact and capacity
calls. Their expansions contain labels, parents, offsets and status with `succ=None`; successor states are not
allocated. The device live-row count bounds action expansion: staged padding contributes no successors or
overflow. Raw expansion status is preserved for the live rows.
Device suites accept Simple output as well as relations, but refuse horizon and transition schemas; encode those
with a separate view per domain.

[`StateRows` and `EncodingCollate`](datasets.md#training-data-encoded-on-load) use this path for resident datasets.
[`BatchedEnv`](rl.md#encoded-observations) returns the corresponding domain batches with `domain`/`index` mappings;
multi-domain torch observations support manual capture and `BatchedEnv.capture`, while `torch.compile` handles
one-domain observations.

## Encoding inside JAX

`mymyr.encoding.jax.encode(encoder, states, task_ids=None, *, derived=None, goals=None, subgoals=None, history=None,
capacities=None, threads=1, live=None)` is
`encoder.encode` as the XLA FFI target `mymyr_encode_flat`, so a jitted function, a `lax.scan` body or a `jax.vmap`
can encode states. On the CPU the target runs the host encoder, on CUDA the device kernels on XLA's stream; both return
the arrays of the host encoder's capacity form bit for bit, so a function gives the same batch wherever it runs. It
needs a mymyr built with jaxlib's XLA FFI headers (as `mymyr.rl.jax`).

```python needs-jax
import jax
import jax.numpy as jnp
import mymyr.encoding.jax as ej

gripper = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
gripper_encoder = encoding.FlatRelationEncoder(gripper)
words = np.asarray(gripper.encode([gripper.initial_state] + gripper.successor_states(gripper.initial_state)))
rows = jnp.asarray(words.view(np.uint32))                 # uint32 halves of the u64 words: [N, 2 W]

host = gripper_encoder.encode(words)
capacities = dict(graphs=len(words), entities=int(host.entity_offsets[-1]), args=2 * int(host.relation_arg_offsets[-1]))
jax_batch = jax.jit(lambda r: ej.encode(gripper_encoder, r, capacities=capacities))(rows)
assert isinstance(jax_batch, ej.JaxFlatBatch) and jax_batch.relation_args.shape == (capacities["args"],)
assert int(jax_batch.overflow) == 0                       # usable inside jit; jax_batch.sizes reads it on the host
assert np.array_equal(np.asarray(jax_batch.relation_args)[: jax_batch.sizes["args"]], host.relation_args)

# two history steps per row: the row's own atoms, then none
history_encoder = encoding.FlatRelationEncoder(gripper, history=True)
steps = np.zeros((len(words), 2, words.shape[1]), np.uint64)
steps[:, 0] = words
history = (steps, np.zeros_like(steps), np.full((len(words), 2), 1, np.int32))
host = history_encoder.encode(words, history=history)
capacities = dict(graphs=len(words), entities=int(host.entity_offsets[-1]), args=int(host.relation_arg_offsets[-1]),
                  history=int(host.history_offsets[-1]))
jax_history = (jnp.asarray(steps.view(np.uint32)), jnp.zeros(steps.shape[:2] + (2 * steps.shape[2],), jnp.uint32),
               jnp.asarray(history[2]))
jax_batch = jax.jit(lambda r, h: ej.encode(history_encoder, r, history=h, capacities=capacities))(rows, jax_history)
assert jax_batch.sizes["history"] == 2 * len(words)
assert np.array_equal(np.asarray(jax_batch.history_dt), host.history_dt)
```

- **Rows.** JAX has no 64-bit types by default, so rows are the u64 words of each state as `uint32` halves, low half
  first (`[N, 2 W]`, or one row `[2 W]`), as `mymyr.rl.jax` takes them; `uint64` or `int64` words (`[N, W]`, with
  `jax_enable_x64` or as NumPy arrays) and `int32` halves are accepted too. `task_ids` is `int32` `[N]` (required for a
  table of several instances), `derived` the rows' derived words, and `goals` the masks `(goal_pos, goal_neg)` or all
  four, with `N` rows or one row for all, as in `encoder.encode`. `subgoals` (`(pos, neg)` or all four masks, `[N, L,
  2 W]` halves or `[N, L, W]` words, or one row `[1, L, ...]` for all) and `history` (`(pos, neg, dt)`, `(pos, neg,
  dt, count)` or all six, masks `[N, H, ...]`, `dt` `int32` `[N, H]` and `count` `int32` `[N]`) are the subgoal layers
  and history steps of `encoder.encode`, on both platforms. A wider integer `dt` or `count` is taken where int32 holds
  its values: a concrete value outside int32 raises `ValueError`, a traced one marks its row invalid.
  With `live=`, subgoal layers and history steps are read only for the live prefix on both platforms; their unused
  per-graph entries and variable output tails are zero.
- **Capacities.** Shapes are static under `jax.jit`, so a traced call needs `capacities=dict(graphs=, entities=,
  args=, targets=, history=)` (`targets` only with target sources, `history` only with history steps; `graphs` at
  least `N`): the per-graph arrays have `N` entries, the variable-size arrays (`history_entity` and `history_dt` among
  them) the capacities' sizes, zero past the batch. `batch.device_sizes` is `uint32` `[S,
  2]`, the `S` 64-bit words of `DeviceFlatBatch.sizes` (entities, tuples, args, targets, history and the overflow
  flags: 1 a capacity, 2 the index type, 4 an invalid row) as halves, low half first; `batch.overflow` is the flags as
  a `uint32` scalar for use inside the function, and `batch.sizes` reads them on the host (NumPy arrays over the
  stacked axes for a batch stacked by `lax.scan` or `jax.vmap`). A batch that does not fit, or has an invalid row, is
  all zero with its flag set. An eager call without capacities waits once for the batch's sizes (the first call for a
  number of rows also counts the batch; later ones reuse the capacities held for it, which only grow) and returns
  exact arrays (`ValueError` for an invalid row, `OverflowError` when `int32` indices do not suffice); a traced call
  without them raises `ValueError`.
- **Live rows.** With capacities, `live=` is an integer scalar clipped to `[0, N]`. Only its prefix is encoded,
  including subgoal layers and history; per-graph entries beyond it and variable output tails are zero.
  `device_graphs` retains the live scalar. Array properties and byte reports keep capacity shapes without a host
  read; `num_graphs` and compact `as_dict()` read the count.
- **`JaxFlatBatch`** is a pytree with the arrays of `FlatBatch` under the same names, dtypes and shapes (an array the
  schema does not store raises `AttributeError`), `num_graphs`, `num_relations`, `schema`, `implicit_entities`,
  `nbytes`, `nbytes_by_array` and
  `as_dict(framework=None)` (`'jax'`, `'dlpack'` or `'numpy'`, with the scalars of `FlatBatch.as_dict`, the table's
  fingerprint included). It is returned from jitted functions and stacked by
  `lax.scan`; under `jax.vmap` every vmapped batch is encoded on its own with the same capacities (the target loops over
  the mapped axis), and every array gains the leading axis. `index_dtype='int64'` needs `jax_enable_x64`.
- **What it encodes.** What [the device encodes](#what-the-device-encodes) but numeric tables, the Simple graph, action
  rows, horizons and transitions, on both platforms: the rows with their goals, subgoal layers and history steps.
  Action relations and action targets are accepted without expansion rows (the action relations then have no tuples);
  `actions=`, horizons and transitions are refused, since the FFI operand list has no expansion labels, node offsets or
  edges. A schema or input the device refuses raises `ValueError` (`mymyr: encoding: ...`) when the function is
  traced. Where an instance of the table has axioms the device cannot evaluate, a call
  without `derived` raises `ValueError` when traced, whatever its rows (the torch path flags those rows in
  `invalid_input` instead). Otherwise the device evaluates missing derived words, as the CPU encoder does. In
  a mymyr built without the CUDA backend, only the CPU target exists, and no device refusal applies, except that an
  encoder of a numeric table (with or without `include_numeric`: the result layout has no numeric buffer), of the
  Simple graph (it has no node or edge arrays) or with shared statics (`statics='shared'`) has no form on either
  target: registering it raises `ValueError`.
- **Handles and lifetime.** The target finds its encoder through the handle registry of the FFI targets.
  `ej.register(encoder)` returns the encoder's handle (`encode` registers it on first use, and takes the handle in place
  of the encoder, so a function can hold the integer only); the registry keeps the encoder, its table and its device
  contexts until `ej.unregister(encoder or handle)`. Every compiled function that calls the encoder holds it as well,
  so it keeps working after the encoder is unregistered and every Python reference to it is gone; the encoder is freed
  when the last of these executables is (`jax.clear_caches()` and dropping the jitted functions). A function traced
  after `unregister` registers the encoder again, or fails if it holds only the handle; one that is being traced or
  compiled while its encoder is unregistered fails with `no encoder is registered` (an executable looks its encoder up
  when it is created), so unregister an encoder only after the functions that use it are compiled. Destroying the last
  reference of an encoder on a CUDA device waits for the device; where the device cannot synchronize (CUDA refuses the
  wait while another thread captures a CUDA graph, which invalidates that capture) the encoder's device memory is kept
  rather than freed under work that may still read it.
- **On CUDA** the first call of a compiled function on a device uploads the encoder's context there (before XLA runs
  or traces the call); every call then enqueues the encoding kernels on XLA's stream (they zero what the batch leaves
  unused), with the device scratch as one more result of the call that XLA allocates, and nothing else: no allocation,
  synchronization or event. The target is registered command-buffer compatible, as the environment targets of
  `mymyr.rl.jax`, so XLA may trace it into its CUDA graphs (`--xla_gpu_enable_command_buffer` with `CUSTOM_CALL`).
- **On the CPU** `threads` sets the host encoder's worker threads (0: one per hardware thread); the arrays do not
  depend on it. A call holds the encoder's host scratch, so the CPU calls of one encoder run one at a time.

## Choosing an encoding

Choose the arrays your model consumes. [Omitting counts and implicit entity arrays](#layouts-and-omitted-arrays)
reduces the relational payload; [shared statics](#shared-statics) avoid repeating static tuples per state.
[HGraph](#the-hgraph-view) exposes typed relations, while the [Simple graph](#the-simple-graph) stores a homogeneous
graph with explicit node/edge labels. [Action targets](#actions) add selectable actions;
[horizons](#horizons-and-transitions) include successor context.

The following measurements use an **NVIDIA L4 (GPU 0)** and batches of **1,024 and 8,192 states**, with int32 indices.
Rows use the **Childsnack domain, problem p16-easy**, except the numeric row, which uses
`tests/data/numeric_tasks/cs-counters.txt`. Each batch cyclically repeats eight distinct states from seeded walks
(`seed=0`); this repetition favors cache reuse compared with equally sized batches of distinct states.
Every row sets `goal_derivations=()` rather than using the default. Capacity outputs reserve 25% extra variable
entries. Byte columns are output payload per state at 1,024 rows; device capacity bytes include device counters.

Times are median CUDA-event nanoseconds per state over 3 rounds of 10 captured encodes, with prepared inputs.
They exclude expansion, HGraph metadata construction, allocation and host dispatch. At 1,024 states, a fixed cost
of roughly 40 µs per call dominates the light forms; dividing it by the row count makes per-state time fall sharply
with larger batches. These are batch latency divided by the number of states, not a per-state throughput slope.
The GPU is shared, so times are indicative.

| Form | CPU exact B/state | Device exact B/state | Device capacity B/state | Encode ns/state, 1,024 | Encode ns/state, 8,192 |
|---|---:|---:|---:|---:|---:|
| Relations | 1520.5 | 1520.5 | 1690.7 | 42.3 | 15.3 |
| Relations with omitted arrays | 1011.5 | 1011.5 | 1135.4 | 41.2 | 12.9 |
| Shared statics | 1117.1 | 1117.1 | 1186.4 | 91.7 | 16.8 |
| HGraph backing | 1520.5 | 1520.5 | 1690.7 | 48.0 | 16.4 |
| Simple graph | 3785.5 | 3785.5 | 4771.9 | 57.3 | 34.5 |
| Relations and Simple graph | 5109.0 | 5109.0 | 6219.3 | 58.5 | 36.9 |
| Action targets | 6542.3 | 6542.3 | 7966.9 | 451.9 | 200.7 |
| Full horizon | 41624.3 | 41624.3 | 51945.4 | 11533.4 | 3328.1 |
| Delta horizon | 12958.3 | 12958.3 | 16064.9 | 14753.5 | 3484.7 |
| Numeric relations (four counters) | 240.0 | 240.0 | 261.1 | 36.9 | 5.6 |

The options for each row follow; all other encoder options use their defaults, including graph-major layout except
for HGraph. Numeric values are float32. Horizons have depth one.

| Form | Encoder and options, in addition to `goal_derivations=()` |
|---|---|
| Relations | `FlatRelationEncoder` |
| Relations with omitted arrays | `FlatRelationEncoder(emit_counts=False, entity_arrays='auto', include_actions=False)` |
| Shared statics | `FlatRelationEncoder(statics='shared')` |
| HGraph backing | `FlatRelationEncoder(layout='relation')`; time covers encoding its backing arrays |
| Simple graph | `SimpleEncoder` |
| Relations and Simple graph | `FlatRelationEncoder(outputs=('relations', 'simple'))` |
| Action targets | `FlatRelationEncoder(target_sources=('action',))`, with a prepared expansion as `actions=` |
| Full horizon | `FlatHorizonEncoder(horizon_mode='full')`, with a prepared `horizon(..., depth=1)` |
| Delta horizon | `FlatHorizonEncoder(horizon_mode='delta')`, with a prepared `horizon(..., depth=1)` |
| Numeric relations (four counters) | `FlatRelationEncoder` over the counter fixture |

The [byte benchmark](../bench/python/bench_encoding_bytes.py) reports every encoding GPU benchmark task, CPU
capacity bytes, per-array JSON breakdowns and unsupported forms. Run it with
`--tasks blocks-p01e --rows 1024 8192` for a vendored task, or `--data` pointing to the IPC directory for the full task
set; `--numeric` adds the counter fixture. Size grows with facts, objects, action fanout and horizon nodes, so use
`nbytes` on representative batches before choosing capacities.

## Names, pickling and persistence

A batch stores no names. `encoder.entity_names(batch, graph)` returns the names of the entities of graph `graph` in
entity order, and `encoder.describe(batch, graph)` a `mymyr.encoding.GraphDescription`: `entities` lists `(role, name)`
for each entity, `relations` maps every relation with tuples to its tuples as tuples of entity names, in catalog order
and in the order of the encoding, and `str(description)` prints it. Both work on a batch of either layout, with or
without the omitted entity arrays, on a batch of the capacity form and on a device batch (copied to the host); the
batch must come from this encoder's schema and hold the relation arrays. A graph out of range raises `IndexError`; a
batch of another schema, a Simple-only batch or one that overflowed its capacities `ValueError`.

```python
named = encoding.FlatRelationEncoder(task, target_sources=("goal",))
description = named.describe(named.encode([task.initial_state]), 0)
print(description)
assert description.entities[0] == ("object", task.objects[0])
assert ("goal", "[+](at o2 o1)[g]") in description.entities
```

The name of an entity follows its role:

| Entity | Name |
|---|---|
| object | its name in the task (`task.objects[entity_ref]`) |
| auxiliary | `auxiliary` |
| goal | `[+](on c b)[g]` or `[-](clear a)[g]`: the sign, then the predicate and objects of the plain goal tuple the entity stands for; a nullary literal is `[+](handempty)[g]` |
| subgoal | `[+](on c b)[sg]`: like a goal, with the level of its relation (`[sg]`, `[ssg]`, `[sssg]`) |
| action | `(unstack a b)`: the action schema, then the objects of its action tuple |
| history step | `history:-2#0`: the `dt` of the step and its index among the graph's encoded steps (`history_entity`), counted from 0 |
| history target | `history:-2#0:[+](on c b)`: the step, then the sign, predicate and objects of the history tuple |
| state | `target:k`: the graph-local node index of a horizon's node |
| predicate virtual node | `predicate:on` |

The names are a function of the batch and the task: they do not depend on the layout, the process or the hash seed.

**Pickling.** A `FlatRelationEncoder` (or any other flat encoder: `SimpleEncoder`, the horizon, rooted horizon and
transition encoders) pickles as its class, its table (or task), its options and the fingerprint of its schema, so an
encoder can go to worker processes with `multiprocessing` or `torch.multiprocessing`. Loading rebuilds the encoder
from the table and the options and compares the schema fingerprint with the pickled one: an encoder that would encode
otherwise (one that a different mymyr version builds with another schema, say) raises `ValueError` instead of
producing batches that differ from the sender's. The loaded encoder's output equals the original's bit for bit.
Batches and streams are not picklable: encode in the worker, or send the arrays.

**Persistence.** A schema is stored as a record of its version, options, fingerprint and catalog
(`encoding.schema_to_dict`, `encoding.schema_from_dict(record, table)`, which checks it over a table of the domain),
and a batch as its arrays and scalars (`batch.as_dict()`, read back and checked by
`encoding.FlatBatch.from_arrays(schema, arrays, *, sizes=None, framework=None)`);
[Encoders, schemas and batches](datasets.md#encoders-schemas-and-batches) shows all three forms.

## C++

`mymyr/encoding/encoder.hpp`, linked with `mymyr::encoding`: `FlatEncoder::create(table_or_task, options)` returns the
encoder, and `encode(EncodeInput{states, task_ids, derived, derived_stride, goal_pos, goal_neg, goal_stride,
goal_words, ..., actions, subgoals, history})` returns the batch as an `rl::ArrayBundle` with the arrays and scalars
above. `mymyr/encoding/validate.hpp` checks a batch against the contract. `encode(in, pool, scratch)` takes a
`ThreadPool` (the batch is encoded by up to `ceil(graphs / k_graphs_per_member)` of its members, on the calling thread
alone for a small batch, byte-identically for any pool) and a reusable `EncodeScratch`. `count(in, pool, scratch)`
returns the `FlatSizes` of a batch (entities, tuples, args, targets, history); `fill(in, sizes, buffers, pool,
scratch)` then writes it into caller-provided `FlatBuffers`, and `encode_into(in, capacities, buffers, pool, scratch)`
does both within `BatchShape{graphs, entities, args, targets, history, numeric}`, reporting `overflow` in the sizes
when a capacity is too small. `encode_capacity(in, capacities, pool, scratch, sizes)` returns a bundle whose variable
arrays have the capacity sizes, zero after the batch; `validate` accepts such a bundle. The host encoder runs on the
CPU. `mymyr/encoding/schema.hpp` is the catalog (`EncodingSchema`), `mymyr/encoding/options.hpp` the options
(`FlatOptions`).

Literals resolve through the context: `detail::resolve_literal(context.view(), instance, predicate, objects, width)`
(`mymyr/encoding/detail/literals.hpp`, host and device code) gives the `LiteralKind` and slot of an atom given as a
table domain predicate id and instance-local objects, by binary search in the context's literal section (every domain
predicate's slot ranges and static rows, whatever the options leave out). `mymyr/encoding/literals.hpp` has the host
loops: `resolve_literals` over arrays of literals and `literal_masks`, the goal or layer masks of rows of literals
(`LiteralMasks` with `unsatisfiable`), with `FixedLiterals::Decide` or `Raise` for the literals of fixed truth.

With `FlatOptions::statics = Statics::Shared` (`schema.shared_statics()`) the calls take the capacity
`StaticShape{instances, args}` and report `ExtraSizes::statics` (`StaticSizes{instances, tuples, args}`) after the
batch's: `count(in, pool, scratch, &extra)`, `encode_capacity(..., sizes, &static_shape, &extra)`, `encode_into(...,
scratch, &static_shape, &extra)` and the stream `flush` overloads. `FlatBuffers` has the five static arrays
(`static_task_ids`, `static_offsets`, `static_arg_offsets`, `static_args`, `static_relations`), null iff the statics
are repeated. `require_repeated_statics(options, family)` is the refusal for an encoding family whose graphs need the
static tuples among their own tuples.

The numeric lane is a cell kind of the same passes: a numeric cell has one tuple per numeric slot `[a, b)` of an
instance, written like the tuples of a static cell, and its values come from the numeric words of the row
(`RowInput::numeric`, `numeric_slot_value`, `fill_numeric_cell` in `mymyr/encoding/detail/lanes.hpp`). `GraphOut`
holds the destination of the values (`numeric_f32` or `numeric_f64`, `numeric_block`) and `SimpleOut::node_value` that
of the node values; `FillCursor::numeric_at` is the index of the next value. `FlatBuffers::numeric_values` and
`node_value` are the arrays, `BatchShape::numeric` the capacity, and `ExtraSizes::numeric` the true count: `count`,
`encode_capacity`, `encode_into` and the stream `flush` overloads take an `ExtraSizes*` last (the `FlatSizes` of a
batch do not carry it, so that the device ABI keeps its six words).

`FlatOptions::outputs` (`Output::Relations`, `Output::Simple`) and `FlatOptions::constant_labels` select the
[Simple graph](#the-simple-graph); `FlatBuffers` has its arrays (`node_offsets`, `node_label`, `edge_offsets`,
`edge_index` with the row distance `edge_stride`, `edge_label`) and `EncodingSchema` its vocabulary
(`simple_label_names()`, `simple_edge_label_names()`, `simple_max_arity()`, `simple_constants()`). The graph is written
in the count, scan and fill passes of the relation arrays, after the tuples of each graph, by `MYMYR_HD` functions over
plain views in `mymyr/encoding/detail/lanes.hpp` (`simple_fill_graph` and its parts): they read the tuples' arguments
the fill has just written, so a Simple-only encoding fills a scratch copy of the relation arrays, not the batch.
`validate` checks the Simple arrays against the contract and, when both outputs are stored, against the relation
arrays.

`mymyr/encoding/horizon_graphs.hpp` builds horizons: `build_horizon_graphs(table, roots, task_ids,
HorizonBuild{depth, max_nodes, truncate, merge}[, pool, scratch])` (and an overload over a `SuccessorGraph`, the
transitions of a state space) returns `HorizonGraphs`, the arrays of a view with `input()` the `HorizonInput`;
`transition_graphs(table, parents, task_ids, expansion)` makes the depth-one graphs of an existing expansion. The
kinship of `_sibling_` and `_cousin_` is built per graph from the canonical parent edges by `MYMYR_HD` functions over
scratch words (`mymyr/encoding/detail/horizon.hpp`: `horizon_kin_words`, `horizon_kin_prepare`,
`horizon_kin_fill_far`).

`mymyr/encoding/hgraph.hpp` has the [HGraph view](#the-hgraph-view): `HGraphView::of(batch, schema)` (a relation-major
`rl::ArrayBundle` and the `EncodingSchema` it was encoded for; `std::invalid_argument` for what the view refuses)
holds the batch alive and answers in constant time per type: `num_nodes(type)`, `symbol_ptr()`, `relation_ptr(relation,
out)`, `edge_type(i)` and `find_edge_type(relation, slot)`, `edge_index_1(i)` (an `IndexRow`: a pointer, a size and a
stride into the batch's `relation_args`, with `wide()` for int64) and `edge_index_0(i, out)`, which writes the arange.
The node types are the relations in relation order and then `symbol_type()`.

`mymyr/encoding/stream.hpp` has the streams: `FlatStream(encoder)` with `append(EncodeInput)`, `size()`, `clear()`
and `flush(pool)` (or `flush(pool, capacities, sizes)`), and `FlatMutableStream(encoder)` with `append(indices,
input)`, `update(indices, input)`, `remove(indices)`, `contains(index)`, `indices()` and the same `flush`. They throw
`std::invalid_argument` for invalid rows or indices and `std::out_of_range` for an index that is not held; their
methods take a mutex, and a flush encodes with the calling thread's `EncodeScratch`, which it keeps between calls.
Host `encoding::collate` is declared in `mymyr/encoding/collate.hpp` and takes a span of shared array bundles.

`mymyr/cuda/encoding.hpp`, linked with `mymyr::cuda`, has the [device encoder](#encoding-on-the-device):
`cuda::DeviceEncoder(context, encoder)` uploads the encoder's context, `reserve` and `acquire` give an
`EncodingWorkspace` per stream, and `encode(workspace, in, out)` encodes device rows within the capacities of `out`
(no allocation, no synchronization, capturable in a CUDA graph). `count(workspace, in, out)` followed by
`fill(workspace, in, out)` is the same encoding in two steps for a caller that sizes its outputs by the batch: the
count writes the `FlatSizes` of the rows to device memory, the caller waits for them and allocates, and the fill writes
the arrays.

The host and device capacity forms use `encoding::BatchShape`. `DeviceOutput::capacities.graphs` bounds all
per-graph and per-cell destinations; `encode`, `count` and `fill` reject a larger input before enqueuing work.
For runtimes that own their scratch, `DeviceEncoder::scratch_bytes(shape)` reports the required byte count. Acquire
the context with `device.context().acquire(stream)` outside capture, then call
`encode(acquisition, scratch, bytes, shape, in, out)` (or the corresponding `count` and `fill` forms). The scratch
address is aligned to 256 bytes. Keep the acquisition and buffers alive through the last captured replay, and use
separate scratch for concurrent calls. A workspace owns the same scratch and delegates to the same implementation.
The scratch layout and workspace storage are private.

`scratch_bytes(acquisition, shape)` sizes scratch for that acquired generation. Its bounds, launch classes and derived
word requirements remain attached to the acquisition across refreshes. A split count/fill uses the same acquisition,
scratch address and shape in both calls; a refresh between them preserves that layout.

`DeviceEncoder::scratch_bytes(encoder, shape)` also sizes caller scratch before constructing a device encoder.
Derived evaluation storage is included in every scratch size query, with table-width derived rows and the largest
instance view, unless the query passes `cuda::DerivedWords::Supplied`: that size holds the encodes that pass their
derived words, and an encode that would evaluate them in such scratch is refused before any launch. The Python
device encoder sizes its scratch that way when `derived=` is given. Split count/fill reuses the derived rows produced
by count. Acquired generations retain their axiom tables and scratch requirements across refreshes.

Automatic device launches choose a width class from each instance's own word count, cell count and argument bound.
A graph-major row uses a thread up to 8 words, 128 cells and 128 arguments; relation-major uses the same word and cell
limits and up to 1,024 arguments. Wider work uses a warp. Count starts with four lanes and tries eight or 32 if its
shared memory does not fit. The shared-memory budget includes static kernel storage. Class launches share scans in
batch order, so a launch permutation changes the workers assigned to graphs and preserves every output position.

`bench/python/bench_encoding_gpu.py --mixed-widths --device-bench PATH` also measures small rows in a table containing
both small and wide instances. The C++ driver accepts `--secondary-problem PATH` for that table and reports the
captured thread and warp fill launches.

The device row streams and collation are `cuda::DeviceRowStream(context, encoder, capacity, stream, goals, derived,
derived_goals)` with `append`, `flush(DeviceOutput)`, `clear`, and `status()`; and `cuda::DeviceCollator(context,
batch_views, stream)` with caller scratch sized by `scratch_bytes()`. The batch views come from
`cuda::collation_batch(encoder, arrays, sizes, graphs)`, which refuses an encoder that the device does not encode.
`DeviceOutput::graphs` optionally receives a live graph count.
The native Python encoder handle's `graph_count` pointer is both input and output: it reads an int64 device scalar
as an unsigned count and writes `min(count, N)` back into the same scalar. Counts above `N`, including negative
int64 values interpreted as unsigned, become `N`.

For CUDA capacity encoding with shared statics, `DeviceOutput::static_capacities` bounds the instance blocks and their
arguments, and `static_sizes` points to a device `StaticSizes`. The overflow flag includes these capacities. The free
`cuda::output_layout` takes a `StaticShape`; `DeviceEncoder::output_layout` reserves blocks for the entire table.
`cuda::collation_batch` accepts the source static sizes and capacities alongside its ordinary batch sizes.
