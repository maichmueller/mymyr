# Native relational encoding

`mymyr.encoding` encodes states of classical tasks into relational graphs: one graph per state, with the task's objects
as entities and, for every relation of an `EncodingSchema`, the tuples of the state's atoms. It runs in C++, reads
native state words directly, and needs no model framework. The result is a batch of flat integer
arrays that every framework imports without a copy. Options follow the [API conventions](conventions.md).

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

## The encoder

`FlatRelationEncoder(table_or_task, *, <options>)` takes a `Task`, a `TaskHandle` or a one-domain `rl.TaskTable` and the
options of the [schema](#encoding-schema).
`encoder.encode(states, task_ids=None, *, derived=None, goals=None, actions=None, framework=None, threads=1, pool=None, capacities=None)`
returns a `FlatBatch`. It is reentrant: one encoder serves any number of concurrent calls.

- **`states`:**
  - an array of state words `[N, W]` of any DLPack framework: u64 words, or u32 halves `[N, 2W]` that are reinterpreted
    as the little-endian halves of the words (the layout JAX uses);
  - or a `State` or a list of `State`s, which are packed first (the output is then NumPy).

  A row may be wider than its instance needs: the words past the instance's own are padding and are not read. A row
  narrower than its instance, or one that sets an atom bit its instance does not have, raises `ValueError`. Tasks with
  lazy atom slots are refused, since their rows have no fixed layout ([State encodings](parsing-and-tasks.md#state-encodings)).
- **`task_ids`:** the instance of each row, an int32 array or a sequence of ints. It is required for a table of several
  instances. A `State` must be a state of the task of its row (the table's instance named by its task id): a `State` of
another `Task` object is accepted when that task has the same `fingerprint()` and `encoding_fingerprint()` (for example a
second parse of the same file), and refused with a `ValueError` naming the row and the instance otherwise.
- **`derived`:** the derived atom words of each row, `[N, D]` u64 (or u32 halves), as `rl.expand` computes them. By
  default they are evaluated here, and only for rows whose instance has derived relations in the schema.
- **`goals`:** a goal for every row instead of its instance's goal: `(goal_pos, goal_neg)` or `(goal_pos, goal_neg,
  goal_derived_pos, goal_derived_neg)`, masks over the fluent (derived) atom slots as `rl.goal_masks` returns them.
  The arrays are `[N, W]` (`[N, D]` for the derived pair), u64 words or u32 halves, or `[1, W]` for one goal that every
  row shares. A row's goal relations then hold exactly its masked literals, with no [constant part](#goals), and its
  `goal_complete` is 1. The masks of a row may be wider than its instance needs (the extra words are padding); a mask
  that is narrower, or one that sets an atom bit its instance does not have, raises `ValueError`. Without the derived
  pair a row's derived goal part is empty. `goals` applies to every row of the call. The masks of `rl.goal_masks`
  (`table.goal_masks()`) are the instance's goal over fluent slots only: they give no derived masks, and a goal with
  such a part keeps only its fluent literals. A call with these masks gives the same batch as a call without `goals`
  only for an instance whose goal is complete and has no literal over a static predicate, over a derived predicate or
  over an atom without a slot (the constant part, the derived part), so that the masks are the whole goal.
- **`actions`:** the `rl.Expansion` of the same states over the same table (`rl.expand(table, states, task_ids)`), on
  the host: the encoder reads its action rows, the labels of the successors, instead of recomputing the applicable
  actions. Without it a schema with action relations has no action tuples. The expansion must hold the states of the
  call in order (`actions.num_states` equals the rows), with every row's actions in the canonical order that
  `rl.expand` writes (ascending action schema, then ascending binding), and must not have overflowed. It must be over
  the encoder's own table, or over a table whose instance for each row has the same `fingerprint()` and
  `encoding_fingerprint()`; any other expansion, a label narrower than the table's, or a schema or object the instance
  lacks raises `ValueError`. `actions` needs the action relations
  (`include_actions=True`, the default) and applies to every row of the call. An expansion that `rl.expand` pruned
  (a witness expansion, which keeps the rows of a subset of the actions) encodes exactly the rows it has.
- **`framework`:** `"numpy"`, `"torch"`, `"jax"` or `"dlpack"`; by default the framework of `states` (NumPy for `State`s).
- **`threads`:** the threads that encode the batch (default 1; 0 means the hardware threads). The result does not depend
  on it: a batch is byte-identical for every thread count. Threads pay off for large batches only: waking a thread
  costs as much as encoding hundreds of graphs, so every thread gets at least 1024 graphs, and a batch of up to 1024
  graphs (a single state included) is encoded on the calling thread whatever the count. The encoder keeps one pool of
  worker threads, rebuilt when the count changes (calls of one encoder with different counts, from several threads,
  rebuild it back and forth: give each such caller its own `pool`). The pool belongs to the process that made it: after
  `os.fork()` the child makes its own.
- **`pool`:** a `mymyr.rl.ThreadPool` that you hold, like `rl.expand`'s: it is used instead of `threads` and shared by
  the calls that name it (one batch runs on it at a time).
- **`capacities`:** `None` (default) sizes every array to the batch. A dict with the keys `graphs`, `entities` and
  `args` (and `targets` when the schema has targets) gives the capacity of the variable-size arrays instead:
  `entity_role` and `entity_ref` have `entities` entries, `relation_args` has `args` entries, `target_entity` and
  `target_group` have `targets` entries; the offset arrays keep their sizes (`G + 1`, `G * R + 1`) and `graphs`
  bounds the number of input rows. Exceeding the graph capacity raises before encoding; tuple counts have no
  separate capacity because every tuple has at least one argument. The batch fills the first entries of each array and the rest is zero, so batches
  of different sizes share array shapes (for example under `jax.jit`). A batch that does not fit (a capacity too small,
  or an index type too narrow) leaves every array zero and sets `batch.sizes['overflow']`; `sizes['capacity_overflow']`
  and `sizes['index_overflow']` tell which, and `sizes['entities']`, `['tuples']`, `['args']` and `['targets']` are the
  true sizes.

Options that this encoder does not encode raise `ValueError` with a `mymyr: encoding:` message: `max_goal_level`
above 0, subgoal and history targets and `horizon_mode`. Numeric tasks raise as well.
Every other invalid input (a task id out of range, a missing `task_ids`, a `State` of another task, a malformed
`derived` array) raises `ValueError` too; a batch that needs more indices than the index type holds raises
`OverflowError` (`index_dtype='int64'` lifts the limit).

## Lanes

A graph has the objects of its instance as entities, in the task's object order, then the auxiliary entity, the goal
entities, the action entities and the predicate virtual nodes, each when the options and the state give it one. Its
relations hold:

- the tuples of the **fluent** atoms of the row, the **derived** atoms (from `derived` or evaluated), and the **static**
  atoms of the instance, each atom in the relation of its predicate, with its arguments as the tuple. The types are
  static unary predicates, so `object` has one tuple per object. The `=` predicate has relations only with
  `equality=True`.
- the **nullary** atoms (`nullary=True`, the default): all nullary atoms of a graph share one auxiliary entity, and each
  one is the unary tuple over it. The graph has the auxiliary entity exactly when it has such a tuple. With
  `nullary=False` there are no nullary relations and no auxiliary entity.

- the **goal** relations, below.
- the **action** relations, from `actions`: one tuple per action of the state, below.

A graph contains nothing that its state and instance do not have: no entity or tuple stands for something absent.
The history relations of the schema have no tuples in a batch.

### Goals

The goal of an instance is `Task.encoding_goal()` read as a set of ground literals (a conjunction, with `forall`
expanded where the front end can). `goal_complete` is 0 for an instance whose goal had a conjunct it could not expand
and for one with a literal that the options leave out although a state can change its truth (a nullary literal under
`nullary=False`); a left-out literal of a static predicate has a fixed truth and leaves the goal complete.
A literal is placed in the relations of its predicate, `[+]` for a positive and `[-]` for a negative one. With the
default `goal_derivations=("plain", "satisfied")` the relations are, per predicate and sign:

- the plain relation `[+]on[g]` holds the tuple of every goal literal;
- `[+]on[g][sat]` holds the literals that hold in the state (a positive literal whose atom is true, a negative one
  whose atom is false); `goal_derivations` may add `"unsatisfied"`, `[+]on[g][unsat]`, the literals that do not.

The literals of fluent and derived predicates (derived atoms from `derived` or evaluated) are evaluated against the
state words with the goal's masks; those of static predicates and of atoms without a slot (not reachable by any state)
have a fixed truth and form the constant part of the goal. A statically false literal is still emitted: in its plain
relation, and in `[unsat]` for a positive literal over a false atom or `[sat]` for a negative one. Goal literals whose
predicate has no goal relation (equality without `equality=True`, nullary atoms with `nullary=False`, the types) are
not encoded. A cell lists its tuples in the lexicographic order of their arguments, constant part and masked
literals merged.

Nullary goal literals are unary tuples over the auxiliary entity, which the graph then has.

With `target_sources=("goal",)` every plain goal tuple gets a **goal entity** (entity role 2, after the objects and the
auxiliary entity), numbered in the order of the plain goal relations (relation id, then tuple) so that the id is
`entity_offsets[g] + num_objects[g] + has_auxiliary + k` for the `k`-th plain goal tuple of graph `g`. The entity is the
first argument of its tuple; its `entity_ref` is the tuple's index among the tuples of graph `g` in graph-major order:
the tuples of its cells counted cell after cell in relation order, so the graph's first tuple has index 0 in either
layout. In the graph layout the tuple is `relation_offsets[g * R] + entity_ref`; in the relation layout, subtract the
counts `relation_counts[g, 0]`, `relation_counts[g, 1]`, ... from `entity_ref` until the rest `j` falls into a cell
`(g, r)`, and the tuple is `relation_offsets[r * G + g] + j`. The batch lists the goal entities as targets:
`target_offsets` `[G + 1]`, `target_entity` `[T]` and `target_group` `[T]` (0: goal, an index into
`encoder.target_group_names`).

### Actions

Given `actions`, the applicable actions of each state are the rows of its expansion. Every distinct action of the state
is one tuple of the action relation of its schema `name`, and the tuple is always `(action entity, o_1, ..., o_k)`: the
**action entity** (entity role 4, after the goal entities) followed by the binding `(o_1, ..., o_k)`. A schema with no
parameters has the unary tuple `(action entity)` and takes no auxiliary entity. A schema that the front end split in
several (a disjunctive precondition) shares one relation, and an action that two of its schemas both give is one tuple:
the relation lists each binding once. There is no limit on the schemas of a relation.

The action entities exist whenever the batch has action tuples, with or without `target_sources=("action",)`. They are
numbered in the order of the action relations (relation id, then tuple). The entity's `entity_ref` is the row of the
action among the rows of its own state in the expansion (the position among the state's rows, the first one for a
binding listed by several schemas), the same in either layout and whatever the graph's place in the batch. The row of
the expansion is `actions.offsets[g] + entity_ref`, so `actions.action(row)` is the action and `actions.succ[row]` its
successor: a policy that picks an action entity picks that row. With `target_sources=("action",)` (or `("action",
"goal")`) the batch also lists the action entities as targets of group 2 (`encoder.target_group_names`), after the goal
entities of their graph; without it they are entities and tuple arguments only. A call without `actions` has no action
tuples and no action entities.

### Predicate virtual nodes

With `predicate_virtual_nodes=True` a graph has one **predicate virtual node** (entity role 1, the last entities of
the graph) for each predicate that one of its tuples has a relation of: the relations of a predicate are its state
relation and its goal relations (`[+]on[g]`, `[+]on[g][sat]`, ...). A tuple of such a relation has the node as an
extra argument right before the predicate's own arguments (after the goal entity of a goal tuple that has one);
action tuples have no predicate slot. A node exists only if some tuple uses it, so a predicate without tuples in the
graph has no node, and the nodes are numbered by the name of their predicate (ascending, bytewise), whatever relations
the schema keeps. The node's `entity_ref` is the id of the predicate's key relation: its plain state relation, or for
a predicate without one (a static or derived predicate whose state relation is dropped) its first relation in the
catalog (C++: `EncodingSchema::virtual_key`). The nullary atoms of a predicate are the tuple over its node and the
auxiliary entity. Actions are not virtual-node predicates.

## The batch

For `G` graphs and `R = schema.num_relations` relations, with `I` the index type (`int32` by default, `int64` with
`index_dtype='int64'`); entity ids are batch-global, and graph `g`'s first entity is `entity_offsets[g]`:

| Array | dtype, shape | Meaning |
|---|---|---|
| `task_ids` | int32 `[G]` | the instance of each row |
| `num_objects` | I `[G]` | the objects of the instance (the auxiliary entity is not counted) |
| `entity_offsets` | I `[G + 1]` | graph `g`'s entities are `[entity_offsets[g], entity_offsets[g + 1])` |
| `entity_role` | uint8 `[E]` | the role of each entity: 0 object, 1 predicate virtual node, 2 goal, 4 action, 8 auxiliary; omitted when implicit |
| `entity_ref` | I `[E]` | an object's instance-local id; a goal entity's tuple index within its graph (graph-major order); an action entity's row within its graph's rows of `actions`; a virtual node's key relation id; -1 for the auxiliary entity; omitted like `entity_role` |
| `relation_counts` | I `[G, R]` | the tuples of cell `(g, r)`, in both layouts; omitted with `emit_counts=False` |
| `relation_offsets` | I `[G * R + 1]` | segment `s` holds the tuples `[offsets[s], offsets[s + 1])`; cell `(g, r)` is `s = g * R + r`, or `r * G + g` in the relation layout |
| `relation_arg_offsets` | I `[G * R + 1]` | the same offsets into `relation_args` |
| `relation_args` | I `[A]` | the entities of the tuples, tuple after tuple, `relation_arities[r]` per tuple |
| `goal_complete` | uint8 `[G]` | 1 iff the goal relations hold the whole goal; present iff the schema has goal relations |
| `target_offsets` | I `[G + 1]` | graph `g`'s targets are `[target_offsets[g], target_offsets[g + 1])`; present iff the schema has targets |
| `target_entity` | I `[T]` | the entity of each target, graph after graph |
| `target_group` | uint8 `[T]` | the group of each target (0: goal, 2: action) |

The scalars `version`, `num_graphs`, `num_relations`, `layout` (0: graph, 1: relation), `index_dtype` (0: int32,
1: int64),
`schema_fingerprint` (the schema's fingerprint as a signed 64-bit value) and `statics` (0: the static tuples are
repeated in every graph) come with `batch.as_dict()`.

The tuples of a cell are the atoms of its predicate in the lexicographic order of their objects, so the atom slot
layout of a task (`atoms='auto'`, `'reachable'`, `'dense'`) never reaches a model. A static tuple appears once per
graph. In the graph layout, encoding a slice of a batch gives the slice of the batch's encoding with the offsets
rebased (`entity_ref` needs no rebase); in the relation layout a slice's cells lie in `R` separate runs, so it is
encoded, not cut. An object's
instance-local id is `entity - entity_offsets[g]`, and `task.objects[entity_ref]` is its name.

`batch.as_dict(framework)` exports the scalars and arrays without a copy; `batch.<array name>` exports one array in the
batch's framework. NumPy arrays are read-only; do not mutate the shared buffers through frameworks whose tensor APIs
cannot enforce that flag. The batch owns its memory and stays valid after the encoder, the task and the inputs are gone.
`batch.schema` is its `EncodingSchema`.

## Memory of batches

A batch is one block of memory. Fresh memory costs a kernel page fault per 4 KiB page, which for a batch of tens of
megabytes is several times the work of filling it, so released blocks are kept and handed out again with their pages
still mapped. The same cache serves the other batch outputs (`mymyr.rl.expand`, table batches). It is bounded:

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
big = encoder.encode([state] * 20000)  # a batch of several megabytes
del big  # its block goes back to the cache
assert mymyr.block_cache_bytes() > 0
mymyr.trim_block_cache()  # and back to the system
assert mymyr.block_cache_bytes() == 0
```

## Layouts and omitted arrays

`layout='graph'` (the default) stores the tuples graph by graph; `layout='relation'` stores them relation by relation
across the graphs, so one relation's tuples of the whole batch are contiguous (a good fit for per-relation message
passing). Both hold the same tuples in the same order within a cell; only the order of the cells differs: a permutation
of the segments. `relation_counts`, the entity arrays (`entity_ref` included) and the target arrays are equal in both.

`emit_counts=False` drops `relation_counts`; the count of a cell is the difference of two consecutive
`relation_offsets`. `entity_arrays='auto'` drops `entity_role` and `entity_ref` when they carry no information: every
entity is an object (role 0, `entity_ref` the instance-local id) or the auxiliary entity (role 8, ref -1), which holds
without `target_sources`, `predicate_virtual_nodes`, `include_actions` and a horizon mode (`schema.implicit_entities`
and `encoder.implicit_entities` tell, `batch.implicit_entities` is the batch's flag). Omitted arrays raise
`AttributeError` on the batch and are absent from `as_dict()`.

## Streams

Rows that arrive one by one (a search expanding states, an actor stepping several environments) are encoded in one call
by collecting them in a stream. `encoder.stream(*, framework=None)` returns an append-only `FlatStream`,
`encoder.mutable_stream(*, framework=None)` a `FlatMutableStream` whose rows have indices you choose. A stream copies
the rows it is given, so the caller's arrays can be reused at once, and `flush` encodes all rows it holds in a single
`encode` call: the batch equals `encoder.encode(rows, ...)` over the same rows bit for bit, whatever the way the rows
arrived.

```python
import numpy as np
import mymyr
from mymyr import encoding, rl

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
encoder = encoding.FlatRelationEncoder(task, target_sources=("goal", "action"))
table = rl.as_table(task)
states = [task.initial_state]
for _ in range(3):
    states.append(task.successor_states(states[-1])[0])

stream = encoder.stream()
stream.append(states[:2], actions=rl.expand(table, states[:2]))
stream.append(states[2:], actions=rl.expand(table, states[2:]))
assert len(stream) == 4
batch = stream.flush()                       # the stream is empty again
assert len(stream) == 0 and batch.num_graphs == 4
expected = encoder.encode(states, actions=rl.expand(table, states))
assert np.array_equal(batch.relation_args, expected.relation_args)
assert np.array_equal(batch.entity_ref, expected.entity_ref)

mutable = encoder.mutable_stream()
mutable.append([7, 3], states[:2], actions=rl.expand(table, states[:2]))
mutable.update(3, states[2], actions=rl.expand(table, states[2:3]))
mutable.remove(7)
assert list(mutable.indices()) == [3] and 3 in mutable
assert mutable.flush().num_graphs == 1       # the rows are kept
assert len(mutable) == 1
```

- **`FlatStream.append(states, task_ids=None, *, derived=None, goals=None, actions=None)`** takes the arguments of
  `encode` and adds the rows. `flush(*, framework=None, threads=1, pool=None, capacities=None)` takes the remaining
  arguments of `encode`, encodes the rows in the order of the appends and **empties** the stream. A flush that
  raises, or whose batch overflowed its `capacities`, keeps the rows. `clear()` drops the rows, `len(stream)` counts
  them, `stream.encoder` is the encoder.
- **`FlatMutableStream`** stores each row under an index: `append(indices, states, task_ids=None, *, ...)` adds rows
  under new indices, `update(indices, states, ...)` replaces the rows under held ones (with all their inputs), and
  `remove(indices)` drops them. An index is an int, an integer array or a sequence of ints; any 64-bit integer,
  negative ones included. `flush` encodes the rows in **ascending order of their indices** and **keeps** them, so
  graph `j` of the batch is the row of `indices()[j]`; `index in stream` tells whether a row is held. An index that is
  repeated within a call or already held raises `ValueError` on `append`; an index that is not held raises `KeyError`
  on `update` and `remove`. A call that raises changes nothing.
- **Per-row inputs.** `derived`, `goals` and `actions` are as in `encode`, with one rule: every row of a stream gives
  the same inputs. The first rows appended to a stream without rows decide (a stream with `goals` keeps a goal for every
  row, a stream with `actions` the action rows of every state, a single `[1, W]` goal is copied for every row), and a
  later `append` or `update` that gives other inputs raises `ValueError`. A stream that holds no rows again takes
  any inputs. Rows narrower than their instance are refused when appended, with the same errors as `encode`; errors
  that need the encoder (a goal bit with no atom slot, action labels of another instance, an action order that is not
  canonical) are raised by `flush`, which keeps the rows.
- **Action rows.** The action rows of the appended expansions are copied and joined in row order, so a flushed batch is
  that of one expansion over all rows: the `entity_ref` of an action entity is what `encode` gives for the same states
  and action rows.
- **Threads.** The methods of one stream take turns (a flush holds the stream until its batch is encoded); a stream is
  not a way to encode concurrently, use one stream per producer or `encode` itself.

## Encoding on the device

`encoder.encode` takes rows in CUDA memory too and encodes them on the GPU with the device kernels, into torch tensors
on the same device. The result equals the host encoder's batch bit for bit, in both layouts and both index types. It
needs a CUDA build of mymyr and torch; the arrays are allocated on torch's current stream and the kernels run on it,
in stream order, like torch's own operations, so no synchronization is needed before or after the call.

```python needs-torch skip-if-no-cuda
import numpy as np
import torch
import mymyr
from mymyr import encoding

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
encoder = encoding.FlatRelationEncoder(task)
rows = np.asarray(task.encode([task.initial_state, task.successor_states(task.initial_state)[0]]))
device_rows = torch.from_numpy(rows.view(np.int64)).cuda()

batch = encoder.encode(device_rows)                       # a DeviceFlatBatch: tensors on the GPU
assert batch.relation_args.is_cuda and batch.num_graphs == 2
assert np.array_equal(batch.as_dict("numpy")["relation_args"], encoder.encode(rows).relation_args)

# with capacities nothing waits for the device: the arrays have the capacities' sizes
capacities = dict(graphs=2, entities=256, args=2048)
fixed = encoder.encode(device_rows, capacities=capacities)
assert fixed.relation_args.shape == (2048,)
assert not fixed.sizes["overflow"]                        # reads the sizes: waits for the device
```

- **Inputs.** `states` is a CUDA torch tensor (`int64` or `uint64` words, or `int32` halves of them; `[N, W]` with
  contiguous rows of any stride, or one row `[W]`) or any CUDA array that speaks DLPack or `__cuda_array_interface__`,
  which is read in place. `task_ids` is a contiguous `int32` `[N]` tensor, `derived` the derived words of the rows
  (`[N, D]`, for instance `rl.expand(table, rows, derived=True).succ` and `.derived` moved to the device), `goals` the
  goal masks `(goal_pos, goal_neg)` or `(goal_pos, goal_neg, goal_derived_pos, goal_derived_neg)` with `N` rows or a
  single shared row. All of them are on one device or all on the host; a call mixing the two raises `ValueError`.
  The derived words must be given when a graph reads derived atoms (a schema with derived relations or derived goal
  literals): the device does not evaluate derived atoms.
- **Refused.** `actions=` (the action relations are not encoded on the device), `pool=` (a pool of host threads
  means nothing for device rows) and a `framework` other than `'torch'` raise `ValueError` with the `mymyr: encoding:`
  prefix; `threads` has no effect. A row that is invalid (a task id outside the table, a row or goal mask narrower
  than its instance, a goal bit its instance has no atom for) raises `ValueError` without capacities and sets
  `invalid_input` in `sizes` with them.
- **One wait without capacities.** An encoding without `capacities` must size its arrays by the batch: it counts the
  batch on the stream, copies the five sizes to the host and waits for them (the only host wait of the call), allocates
  exact arrays and fills them. A batch that needs more indices than `int32` holds raises `OverflowError`; use
  `index_dtype='int64'`.
- **No wait with capacities.** With `capacities=dict(graphs=, entities=, args=, targets=)` (`targets` only with target
  sources) every variable-size array is allocated at its capacity, zero-filled, and the kernels write the true sizes into
  `batch.device_sizes` (an `int64` `[5]` tensor: entities, tuples, args, targets and the overflow flags, 1 for a
  capacity, 2 for the index type, 4 for an invalid row). The call never waits and can be captured in a CUDA graph. A batch
  that does not fit its capacities stays all zero with the flag set. `batch.sizes` reads `device_sizes` into a dict (and
  waits).
- **`DeviceFlatBatch`** has the accessors of `FlatBatch`: the arrays (an array the schema does not store raises
  `AttributeError`), `num_graphs`, `num_relations`, `schema`, `implicit_entities`, `as_dict(framework=None)` (torch
  tensors, `'dlpack'` or, copying to the host, `'numpy'`) and `device`. `encoder.entity_names(batch, g)` and
  `encoder.describe(batch, g)` accept it and copy it to the host.
- **Workspaces.** The encoder keeps one device workspace (the scratch of the kernels and its uploaded context) per
  stream it runs on, made by the first call on the stream and reused after it; a batch of more rows than the workspace
  holds makes the next call reserve a larger one (the size is a power of two, at least 1024 rows).
  `encoder.reserve_device(graphs, *, device=None, stream=None)` reserves one in advance and
  `encoder.release_device(*, device=None, stream=None)` frees the workspace of a stream, or of every stream of the
  device. Workspaces are freed with the encoder.

### The torch op

`torch.ops.mymyr.encode_flat(encoder, states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg,
graphs, entities, args, targets)` is the capacity form as a torch custom op, for `torch.compile` and CUDA graphs
(`import mymyr.rl.torch` registers it). `encoder` is the handle `mymyr.rl.torch.register(encoder)` returns; the tensors
are as above (`None` for an input that is absent) and the four ints are the capacities (`targets` 0 without target
sources). It returns the list of 14 tensors that `DeviceFlatBatch.array_names` names, the arrays in the order of the
batch table and `sizes` last; an array the schema does not store is an empty tensor, and
`DeviceFlatBatch.from_arrays(encoder, outputs)` wraps the list into a batch. The op waits for nothing, allocates only
its outputs and has a fake implementation that gives their shapes from the capacities, so `torch.compile(fullgraph=True)`
traces it without a graph break. The registry keeps the encoder alive until `unregister(handle)`. A compiled function
holds its handle and capacities as constants: compile one function per encoder, since one that is called with another
handle is traced again with symbolic integers, which the fake implementation cannot resolve to an encoder.

To capture the op (or `encode` with capacities) in a `torch.cuda.CUDAGraph`, reserve the workspace on the capture
stream first, since a reservation allocates and a capture cannot: `encoder.reserve_device(max_rows, stream=capture_stream)`.
A capture without it raises a `ValueError` that says so, and so does an encoding without capacities. The graph reads
its input tensors where they were captured: write new rows in place and replay. Keep the encoder registered while the
graph can replay, and do not release its workspace.

## Names

A batch stores no names. `encoder.entity_names(batch, graph)` returns the names of the entities of graph `graph` in
entity order, and `encoder.describe(batch, graph)` a `mymyr.encoding.GraphDescription`: `entities` lists `(role, name)`
for each entity, `relations` maps every relation with tuples to its tuples as tuples of entity names, in catalog order and
in the order of the encoding, and `str(description)` prints it. Both work on a batch of either layout, with or without
the omitted entity arrays, and on a batch of the capacity form; the batch must come from this encoder's schema.
A graph out of range raises `IndexError`, a batch of another schema or one that overflowed its capacities
`ValueError`.

```python
description = encoder.describe(batch, 0)
print(description)
assert description.entities[0] == ("object", task.objects[0])
```

The name of an entity follows its role:

| Entity | Name |
|---|---|
| object | its name in the task (`task.objects[entity_ref]`) |
| auxiliary | `auxiliary` |
| goal | `[+](on c b)[g]` or `[-](clear a)[g]`: the sign, then the predicate and objects of the plain goal tuple the entity stands for; a nullary literal is `[+](handempty)[g]` |
| action | `(unstack a b)`: the action schema, then the objects of its action tuple |
| predicate virtual node | `predicate:on` |

The names are a function of the batch and the task: they do not depend on the layout, the process or the hash seed.

## Pickling

A `FlatRelationEncoder` pickles as its table (or task), its options and the fingerprint of its schema, so an encoder
can go to worker processes with `multiprocessing` or `torch.multiprocessing`. Loading rebuilds the encoder from the
table and the options and compares the schema fingerprint with the pickled one: an encoder that would encode otherwise
(one that a different mymyr version builds with another schema, say) raises `ValueError` instead of producing batches that differ from the
sender's. The loaded encoder's output equals the original's bit for bit. Batches and streams are not picklable: encode in
the worker, or send the arrays.

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
encoder, and `encode(EncodeInput{states, task_ids, derived, derived_stride, goal_pos, goal_neg, goal_stride, goal_words, ...})`
returns the batch as an `rl::ArrayBundle` with the arrays and scalars above. `mymyr/encoding/validate.hpp` checks a
batch against the contract. `encode(in, pool, scratch)` takes a `ThreadPool` (the batch is encoded by up to
`ceil(graphs / k_graphs_per_member)` of its members, on the calling thread alone for a small batch, byte-identically for
any pool) and a reusable `EncodeScratch`. `count(in, pool, scratch)` returns the `FlatSizes` of a batch (entities,
tuples, args, targets); `fill(in, sizes, buffers, pool, scratch)` then writes it into caller-provided `FlatBuffers`, and
`encode_into(in, capacities, buffers, pool, scratch)` does both within `BatchShape{graphs, entities, args, targets}`, reporting `overflow` in the
sizes when a capacity is too small. `encode_capacity(in, capacities, pool, scratch, sizes)` returns a bundle whose
variable arrays have the capacity sizes, zero after the batch; `validate` accepts such a bundle. The host encoder
runs on the CPU.

`mymyr/cuda/encoding.hpp`, linked with `mymyr::cuda`, has the [device encoder](#encoding-on-the-device):
`cuda::DeviceEncoder(context, encoder)` uploads the encoder's context, `reserve` and `acquire` give an
`EncodingWorkspace` per stream, and `encode(workspace, in, out)` encodes device rows within the capacities of `out`
(no allocation, no synchronization, capturable in a CUDA graph). `count(workspace, in, out)` followed by
`fill(workspace, in, out)` is the same encoding in two steps for a caller that sizes its outputs by the batch: the
count writes the `FlatSizes` of the rows to device memory, the caller waits for them and allocates, and the fill writes
the arrays.

`mymyr/encoding/stream.hpp` has the streams: `FlatStream(encoder)` with `append(EncodeInput)`, `size()`, `clear()`
and `flush(pool)` (or `flush(pool, capacities, sizes)`), and `FlatMutableStream(encoder)` with
`append(indices, input)`, `update(indices, input)`, `remove(indices)`, `contains(index)`, `indices()` and the same
`flush`. They throw `std::invalid_argument` for invalid rows or indices and `std::out_of_range` for an index that is not
held; their methods take a mutex, and each stream has its own `EncodeScratch`.

The host and device capacity forms use `encoding::BatchShape`. `DeviceOutput::capacities.graphs` bounds all
per-graph and per-cell destinations; `encode`, `count` and `fill` reject a larger input before enqueuing work.
For runtimes that own their scratch, `DeviceEncoder::scratch_bytes(shape)` reports the required byte count. Acquire
the context with `device.context().acquire(stream)` outside capture, then call
`encode(acquisition, scratch, bytes, shape, in, out)` (or the corresponding `count` and `fill` forms). The scratch
address is aligned to 256 bytes. Keep the acquisition and buffers alive through the last captured replay, and use
separate scratch for concurrent calls. A workspace owns the same scratch and delegates to the same implementation.
The scratch layout and workspace storage are private.

Automatic device launches choose a width class from each instance's own word count, cell count and argument bound.
A graph-major row uses a thread up to 8 words, 128 cells and 128 arguments; relation-major uses the same word and cell
limits and up to 1,024 arguments. Wider work uses a warp. Count starts with four lanes and tries eight or 32 if its
shared memory does not fit. The shared-memory budget includes static kernel storage. Class launches share scans in
batch order, so a launch permutation changes the workers assigned to graphs and preserves every output position.

`bench/python/bench_encoding_gpu.py --mixed-widths --device-bench PATH` also measures small rows in a table containing
both small and wide instances. The C++ driver accepts `--secondary-problem PATH` for that table and reports the
captured thread and warp fill launches.

`scratch_bytes(acquisition, shape)` sizes scratch for that acquired generation. Its bounds, launch classes and derived
word requirements remain attached to the acquisition across refreshes. A split count/fill uses the same acquisition,
scratch address and shape in both calls; a refresh between them preserves that layout.
