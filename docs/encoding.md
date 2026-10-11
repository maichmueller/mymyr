# Native relational encoding

`mymyr.encoding` encodes states of classical tasks into relational graphs: one graph per state, with the task's objects
as entities and, for every relation of an `EncodingSchema`, the tuples of the state's atoms. It runs in C++, reads
native state words directly, and needs neither a model framework nor mifrost. The result is a batch of flat integer
arrays that every framework imports without a copy.

```python
import numpy as np
import mymyr
from mymyr import encoding

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
encoder = encoding.FlatRelationEncoder(task, goal_derivations=())
state = task.initial_state
batch = encoder.encode([state])
assert batch.num_graphs == 1

at = encoder.relation_id("at")
assert batch.relation_counts[0, at] == sum(str(atom).startswith("(at ") for atom in state.atoms())
assert not batch.relation_args.flags.writeable
assert np.array_equal(encoder.encode(state).relation_args, batch.relation_args)
```

## The encoder

`FlatRelationEncoder(table_or_task, *, <options>)` takes a `Task`, a `TaskHandle` or a one-domain `rl.TaskTable` and the
options of the [schema](#encoding-schema). `encoder.encode(states, task_ids=None, *, derived=None, framework=None)`
returns a `FlatBatch`. It is reentrant: one encoder serves any number of concurrent calls.

- **`states`:**
  - an array of state words `[N, W]` of any DLPack framework: u64 words, or u32 halves `[N, 2W]` that are reinterpreted
    as the little-endian halves of the words (the layout JAX uses);
  - or a `State` or a list of `State`s, which are packed first (the output is then NumPy).

  A row may be wider than its instance needs: the words past the instance's own are padding and are not read. A row
  narrower than its instance, or one that sets an atom bit its instance does not have, raises `ValueError`. Tasks with
  lazy atom slots are refused, since their rows have no fixed layout ([State encodings](parsing-and-tasks.md#state-encodings)).
- **`task_ids`:** the instance of each row, an int32 array or a sequence of ints. It is required for a table of several
  instances; a `State` must belong to the task of its row.
- **`derived`:** the derived atom words of each row, `[N, D]` u64 (or u32 halves), as `rl.expand` computes them. By
  default they are evaluated here, and only for rows whose instance has derived relations in the schema.
- **`framework`:** `"numpy"`, `"torch"`, `"jax"` or `"dlpack"`; by default the framework of `states` (NumPy for `State`s).

Options that this encoder does not encode raise `ValueError` with a `mymyr: encoding:` message: `goal_derivations`,
`max_goal_level` above 0, `target_sources`, `predicate_virtual_nodes`, `horizon_mode` and `layout='relation'`. The
default of `goal_derivations` is among them: **pass `goal_derivations=()`**. Numeric tasks raise as well. Every
other invalid input (a task id out of range, a missing `task_ids`, a `State` of another task, a malformed `derived`
array) raises `ValueError` too; a batch that needs more indices than the index type holds raises `OverflowError`
(`index_dtype='int64'` lifts the limit).

## Lanes

A graph has the objects of its instance as entities, in the task's object order. Its relations hold:

- the tuples of the **fluent** atoms of the row, the **derived** atoms (from `derived` or evaluated), and the **static**
  atoms of the instance, each atom in the relation of its predicate, with its arguments as the tuple. The types are
  static unary predicates, so `object` has one tuple per object. The `=` predicate has relations only with
  `equality=True`.
- the **nullary** atoms (`nullary=True`, the default): all nullary atoms of a graph share one auxiliary entity, and each
  one is the unary tuple over it. The graph has the auxiliary entity exactly when it has such a tuple. With
  `nullary=False` there are no nullary relations and no auxiliary entity.

A graph contains nothing that its state and instance do not have: no entity or tuple stands for something absent.
The goal, history and action relations of the schema have no tuples in a batch.

## The batch

For `G` graphs and `R = schema.num_relations` relations, with `I` the index type (`int32` by default, `int64` with
`index_dtype='int64'`); entity ids are batch-global, and graph `g`'s first entity is `entity_offsets[g]`:

| Array | dtype, shape | Meaning |
|---|---|---|
| `task_ids` | int32 `[G]` | the instance of each row |
| `num_objects` | I `[G]` | the objects of the instance (the auxiliary entity is not counted) |
| `entity_offsets` | I `[G + 1]` | graph `g`'s entities are `[entity_offsets[g], entity_offsets[g + 1])` |
| `entity_role` | uint8 `[E]` | the role of each entity: 0 object, 8 auxiliary (`schema.entity_role_names`) |
| `entity_ref` | I `[E]` | an object's instance-local id; -1 for the auxiliary entity |
| `relation_counts` | I `[G, R]` | the tuples of cell `(g, r)` |
| `relation_offsets` | I `[G * R + 1]` | cell `g * R + r` holds the tuples `[relation_offsets[c], relation_offsets[c + 1])` |
| `relation_arg_offsets` | I `[G * R + 1]` | the same offsets into `relation_args` |
| `relation_args` | I `[A]` | the entities of the tuples, tuple after tuple, `relation_arities[r]` per tuple |

The scalars `version`, `num_graphs`, `num_relations`, `layout` (0: graph), `index_dtype` (0: int32, 1: int64),
`schema_fingerprint` (the schema's fingerprint as a signed 64-bit value) and `statics` (0: the static tuples are
repeated in every graph) come with `batch.as_dict()`.

The tuples of a cell are the atoms of its predicate in the lexicographic order of their objects, so the atom slot
layout of a task (`atoms='auto'`, `'reachable'`, `'dense'`) never reaches a model. A static tuple appears once per
graph. Encoding a slice of a batch gives the slice of the batch's encoding with the offsets rebased. An object's
instance-local id is `entity - entity_offsets[g]`, and `task.objects[entity_ref]` is its name.

`batch.as_dict(framework)` exports the scalars and arrays without a copy; `batch.<array name>` exports one array in the
batch's framework. NumPy arrays are read-only; do not mutate the shared buffers through frameworks whose tensor APIs
cannot enforce that flag. The batch owns its memory and stays valid after the encoder, the task and the inputs are gone.
`batch.schema` is its `EncodingSchema`.

## Encoding schema

`mymyr.encoding.EncodingSchema(table_or_task, **options)` (C++: `mymyr/encoding/schema.hpp`) is the relation catalog of
an encoding: which relations a batch holds, with their arities, lanes and slot roles. It is a function of the domain
and the options, never of the instances of a table, so tables over one domain give the same relation ids to the same
relations. An encoder has the same properties.

- **Names and order.** Relations are named `[+]on[g][sat]`: a sign for goal and history relations, the predicate or
  action schema name, `[hist]`, a goal level (`[g]`, `[sg]`, `[ssg]`, `[sssg]`) and a derivation (`[sat]`, `[unsat]`,
  `[sat+]`, `[sat-]`). State relations carry the plain name. The catalog is sorted by name, bytewise; a relation's id
  is its position. `relation_lanes`, `relation_signs`, `relation_levels`, `relation_derivations`,
  `relation_predicates` and `relation_schemas` describe each relation, `relation_slot_roles` each position of its tuples.
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

## C++

`mymyr/encoding/encoder.hpp`, linked with `mymyr::encoding`: `FlatEncoder::create(table_or_task, options)` returns the
encoder (pass `options.goal_derivations = {}`), and `encode(EncodeInput{states, task_ids, derived, derived_stride})`
returns the batch as an `rl::ArrayBundle` with the arrays and scalars above. `mymyr/encoding/validate.hpp` checks a
batch against the contract. The encoder runs on the calling thread and is not a CUDA or JAX-jitted operation.
