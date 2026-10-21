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
`encoder.encode(states, task_ids=None, *, derived=None, goals=None, actions=None, subgoals=None, history=None, framework=None, threads=1, pool=None, capacities=None)`
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
- **`subgoals`:** the subgoal layers of every row, `(pos, neg)` or `(pos, neg, derived_pos, derived_neg)`: masks over the
  fluent (derived) atom slots, `[N, L, W]` (`[N, L, D]` for the derived pair) with `L` at most `max_goal_level`, or
  `[1, L, W]` for layers that every row shares; u64 words or u32 halves. Layer `l` fills the relations of level `l + 1`
  ([Subgoals](#subgoals)). A row may have fewer layers than the schema has levels (the missing ones are empty). It
  needs `max_goal_level >= 1` and goal relations. Masks narrower than the instance, or with a bit the instance does not
  have, raise `ValueError`, as for `goals`.
  `mymyr.encoding.subgoal_masks(encoder, layers, task_ids=None)` builds the arrays from lists of literals.
- **`history`:** the history of every row, `(pos, neg, dt)`, `(pos, neg, dt, count)` or `(pos, neg, dt, count,
  derived_pos, derived_neg)`: masks `[N, H, W]` (`[N, H, D]`) of the steps of a row, oldest first, `dt` an integer array
  `[N, H]` (the time of each step relative to the state, as the caller defines it; the encoder only stores it) and `count`
  an integer array `[N]` with the valid steps of each row, a prefix of its `H` (`None`: all `H`). Steps past a row's count
  are padding and never read. It needs `history=True` ([History](#history)).
  `mymyr.encoding.history_masks(encoder, steps, task_ids=None)` builds the arrays from lists of `(dt, literals)`.
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
  `args` (and `targets` when the schema has targets, `history` when it has history) gives the capacity of the
  variable-size arrays instead: `entity_role` and `entity_ref` have `entities` entries, `relation_args` has `args`
  entries, `target_entity` and `target_group` have `targets` entries, `history_entity` and `history_dt` have `history`
  entries; the offset arrays keep their sizes (`G + 1`, `G * R + 1`) and `graphs` bounds the number of input rows.
  Exceeding the graph capacity raises before encoding; tuple counts have no separate capacity because every tuple has
  at least one argument. The batch fills the first entries of each array and the rest is zero, so batches
  of different sizes share array shapes (for example under `jax.jit`). A batch that does not fit (a capacity too small,
  or an index type too narrow) leaves every array zero and sets `batch.sizes['overflow']`; `sizes['capacity_overflow']`
  and `sizes['index_overflow']` tell which, and `sizes['entities']`, `['tuples']`, `['args']`, `['targets']` and
  (with history) `['history']` are the true sizes.

`FlatRelationEncoder` takes no horizon options (`horizon_mode`, `root_policy`, `parent_relation`: `TypeError`); the
[horizon encoder](#horizons) encodes them. Numeric tasks raise `ValueError`.
Every other invalid input (a task id out of range, a missing `task_ids`, a `State` of another task, a malformed
`derived` array) raises `ValueError` too; a batch that needs more indices than the index type holds raises
`OverflowError` (`index_dtype='int64'` lifts the limit).

## Lanes

A graph has the objects of its instance as entities, in the task's object order, then the auxiliary entity, the goal
entities, the subgoal entities, the action entities, the history step entities, the history target entities and the
predicate virtual nodes, each when the options and the inputs give it one. Its relations hold:

- the tuples of the **fluent** atoms of the row, the **derived** atoms (from `derived` or evaluated), and the **static**
  atoms of the instance, each atom in the relation of its predicate, with its arguments as the tuple. The types are
  static unary predicates, so `object` has one tuple per object. The `=` predicate has relations only with
  `equality=True`.
- the **nullary** atoms (`nullary=True`, the default): all nullary atoms of a graph share one auxiliary entity, and each
  one is the unary tuple over it. The graph has the auxiliary entity exactly when it has such a tuple. With
  `nullary=False` there are no nullary relations and no auxiliary entity.

- the **goal** relations, below.
- the **subgoal** relations, from `subgoals`: the goal relations of the levels 1 to 3, below.
- the **action** relations, from `actions`: one tuple per action of the state, below.
- the **history** relations, from `history`: the literals of the steps of the row's history, below.

A graph contains nothing that its inputs and instance do not have: no entity or tuple stands for something absent.
A call without `subgoals` and `history` has no tuples in their relations.

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

### Subgoals

With `max_goal_level = L` the schema has, for every predicate and sign, goal relations of the levels 1 to `L` besides the
goal's: `[+]on[sg]`, `[+]on[ssg]`, `[+]on[sssg]` and their `[sat]` and `[unsat]` variants, as `goal_derivations` says.
Layer `l` of a row's `subgoals` (`l = 0, 1, ...`) fills the relations of level `l + 1`. A level holds the literals
of its layer **minus every literal of the goal and of the lower layers**: a literal is encoded once, at the lowest level
that has it. A literal that the goal has is only in the goal relations, one that layers 0 and 2 both have is only in
level 1, and so is one that layers 0 and 1 both have. The layers of a row are masks, so a literal listed twice in one
layer is one tuple. The `[sat]` and `[unsat]` variants are the literals of the level that hold, or do not hold, in the
row's state. A row's own `goals`, when given, are the goal that is subtracted instead of the instance's.

With `target_sources` including `"subgoal"` (it needs the plain goal derivation and `max_goal_level >= 1`) every tuple
of a plain subgoal relation gets a **subgoal entity** (entity role 3, after the goal entities), numbered in the order of
the plain subgoal relations (relation id, then tuple), the first argument of its tuple, with the tuple's index among the
graph's tuples in graph-major order as `entity_ref`, exactly as a goal entity. The batch lists them as targets of group 1,
after the goal entities of their graph.

Nullary literals in a layer are unary tuples over the auxiliary entity, which the graph then has.

### History

With `history=True` the history relations `[+]on[hist]` and `[-]on[hist]` of the schema hold the literals of the steps of
a row's `history`. Of a row's valid steps only the **newest** `history_max_steps` are encoded (`None`, the default: all of
them; otherwise at least 1). Each encoded step is one **history step entity** (entity role 5, after the action entities), in step order,
oldest first; the batch lists them in `history_offsets` `[G + 1]`, `history_entity` `[S]` (the entity of each step) and
`history_dt` int32 `[S]` (its `dt`). The `entity_ref` of a step entity is the step's index among the row's valid steps
(not among the encoded ones: step `entity_ref` of the row's `history` is `history[2][g, entity_ref]` for its `dt` and
`history[0][g, entity_ref]` for its positive mask, whatever `history_max_steps` is).

A history tuple is `(history step, [predicate node,] o_1, ..., o_k)`, and with `target_sources` including `"history"` it
starts with a **history target entity** (entity role 6, after the history steps) of its own: `(history target, history
step, ...)`. The tuples of a cell are in the order of the steps, then of the atom slots (which is the lexicographic order of
the objects). History target entities are numbered in the order of the history relations (relation id, then tuple) and
their `entity_ref` is the tuple's index among the graph's tuples in graph-major order. That is the order in which the
tuples are filled, not the order of the steps, so the target of a literal is found through its tuple. The batch lists
them as targets of group 3, after the action targets of their graph. A literal over a nullary predicate is the tuple over
the auxiliary entity.

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

## Horizons

`FlatHorizonEncoder(table_or_task, *, horizon_mode, <options>)` encodes **horizons**: one graph per root state, holding
the root and the nodes below it (states reached from the root by actions) as state entities, so that a model sees a
root and its candidate successors in one graph. It has the options of `FlatRelationEncoder` except `history` (a horizon
has none), with `horizon_mode` required (`'full'`, `'delta'` or `'action'`; `None` raises `ValueError`, there is no
horizon without a mode) and `root_policy` and `parent_relation` new. It shares the batch format,
`schema`, `table`, `entity_names`, `describe` and pickling of the flat encoder, and a `FlatBatch` of a horizon encoder
has `batch.schema.options['horizon_mode']` set. `EncodingSchema` takes the same three options.
`encoder.encode(view, *, derived=None, goals=None, subgoals=None, framework=None, threads=1, pool=None,
capacities=None)` returns a `FlatBatch` of `G` graphs, one per root.

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

**The edges.** The graph is a dag: a node may have several parents, an edge may lead back to the root or to its own
source, and the edges may come in any order and more than once. The **action tuples** of a relation are the distinct
`(dst node, objects)` of the edges whose schema belongs to it, ordered by destination node and then objects, and the
**`_parent_` tuples** are the distinct `(src, dst)` pairs in ascending order; so reordering or repeating the edges does
not change the batch, a node that several edges reach with the same action has one action tuple, and a node with several
parents has one `_parent_` tuple for each. (The tuples of every other relation are in the canonical order of the flat
encoder.) `rl.expand` rows are a valid edge list: the rows of a state's successors, `schema`, `binding`, with `edge_src`
the state's node and `edge_dst` the node you made of the successor. The edge arrays a schema does not read (all of them
without action and parent relations, `edge_schema` and `edge_binding` without action relations) and the arrays of a batch
without any edge may be `None`. `derived` is per node row `[M, D]`; `goals` and `subgoals` are
per graph (`[G, ...]`, or one row for all graphs) and replace the instance's goal and layers for that graph.
Streams do not encode horizons (a horizon is a batch of graphs with offsets, not a row): the horizon encoder has no
`stream` and C++ `FlatStream(encoder)` refuses it with `std::invalid_argument`. The device encoder refuses a horizon
schema (`ValueError`, "mymyr: encoding: ..."). Invalid views raise `ValueError` naming the graph
(`graph 3 of the horizon has no nodes`, `edge 2 of graph 0 of the horizon has an endpoint outside the graph's nodes`,
`node 1 of graph 2 sets an atom bit its instance does not have`).

**State entities.** The graph has an entity of role 7 (`state`) per node, after the objects, the auxiliary entity and
before the predicate virtual nodes. Its `entity_ref` is the **graph-local node index** `k`, so the node is row
`horizon.node_offsets[g] + entity_ref` of the view's `nodes`, whatever the layout or the graph's place in the batch.
Names are `target:k` (`encoder.entity_names`). The nodes that are candidates for a model's choice (`target_group` 4,
`encoder.target_group_names`) are all state entities under `root_policy='include'` and all but the root otherwise; the
batch lists them in `target_entity`, so the schema has targets (the capacity form needs `targets`).

**Root policies.** `include` (the default) makes the root node 0 a state entity and a target candidate. `encode_only`
gives it a state entity but does not list it as a target. `exclude` has no state entity for the root: the graph's
entities are the nodes 1 and up, and the root's relations are the plain ones (no state slot): the root's atoms are
tuples without a state argument. An edge into the root gives no action tuple (the action tuples start with the state of
their destination) and an edge that touches the root no `_parent_` tuple; an edge out of the root keeps its action tuple,
which names the node it leads to. The root has no entity at all; its relations say what it holds.

**Modes.** Every relation of a node has the node's state entity as its first argument (the slot role `state`):

- `full`: the atoms of every node, `on(node, a, b)`, the root included under `include` and `encode_only`; under
  `exclude` the root's atoms are the plain relations `on(a, b)` and the nodes' `on[state](node, a, b)`. The static
  atoms are the root's only (the same for every node): `object(target:0, a)`, or plain under `exclude`. The goal
  relations hold the goal's tuples once, at the root, and the satisfaction relations `[g][sat]`, `[g][unsat]` the
  satisfaction of the goal literals at every node.
- `delta`: the facts the node adds to or deletes from its **root** (not its parent): `[+]on(node, a, b)` for an atom
  that holds in the node and not in the root, `[-]on(node, a, b)` for one that holds in the root and not in the node.
  The root's own atoms are in the plain relations (with the root's state entity unless it is excluded). With
  `goal_derivations` `added_satisfied` and
  `added_unsatisfied`, the relations `[+]on[g][sat+]` and `[sat-]` hold the goal literals that the node makes true and
  those it makes false, relative to the root. A node equal to its root (a self-loop's end) has no delta tuples.
- `action`: the root's state and, for every edge, the action that leads to its target node: the action relation of the
  schema holds `(dst node, o_1, ..., o_k)`. The nodes carry no facts of their own.

The catalog lists only relations a graph can have tuples in: a static predicate has no delta relation and no relation
of the nodes (its atoms are the root's), and the added-satisfaction relations exist per goal level.

**Nullary relations.** The atoms and goal literals of nullary predicates are tuples over the graph's auxiliary entity,
in front of which stands the state entity (`handempty(node, auxiliary)`); the graph has the auxiliary entity exactly when
a nullary tuple exists. With `nullary=False` they are absent.

```python
import numpy as np
from mymyr import encoding, rl

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
batch = horizon.encode(view)
assert horizon.entity_names(batch, 0)[-n - 1:] == [f"target:{k}" for k in range(n + 1)]
```

## The batch

For `G` graphs and `R = schema.num_relations` relations, with `I` the index type (`int32` by default, `int64` with
`index_dtype='int64'`); entity ids are batch-global, and graph `g`'s first entity is `entity_offsets[g]`:

| Array | dtype, shape | Meaning |
|---|---|---|
| `task_ids` | int32 `[G]` | the instance of each row |
| `num_objects` | I `[G]` | the objects of the instance (the auxiliary entity is not counted) |
| `entity_offsets` | I `[G + 1]` | graph `g`'s entities are `[entity_offsets[g], entity_offsets[g + 1])` |
| `entity_role` | uint8 `[E]` | the role of each entity: 0 object, 1 predicate virtual node, 2 goal, 3 subgoal, 4 action, 5 history step, 6 history target, 7 state (a horizon's node), 8 auxiliary; omitted when implicit |
| `entity_ref` | I `[E]` | an object's instance-local id; a goal, subgoal or history target entity's tuple index within its graph (graph-major order); an action entity's row within its graph's rows of `actions`; a history step's index among the row's valid steps; a virtual node's key relation id; -1 for the auxiliary entity; omitted like `entity_role` |
| `relation_counts` | I `[G, R]` | the tuples of cell `(g, r)`, in both layouts; omitted with `emit_counts=False` |
| `relation_offsets` | I `[G * R + 1]` | segment `s` holds the tuples `[offsets[s], offsets[s + 1])`; cell `(g, r)` is `s = g * R + r`, or `r * G + g` in the relation layout |
| `relation_arg_offsets` | I `[G * R + 1]` | the same offsets into `relation_args` |
| `relation_args` | I `[A]` | the entities of the tuples, tuple after tuple, `relation_arities[r]` per tuple |
| `goal_complete` | uint8 `[G]` | 1 iff the goal relations hold the whole goal; present iff the schema has goal relations |
| `target_offsets` | I `[G + 1]` | graph `g`'s targets are `[target_offsets[g], target_offsets[g + 1])`; present iff the schema has targets |
| `target_entity` | I `[T]` | the entity of each target, graph after graph, in entity order |
| `target_group` | uint8 `[T]` | the group of each target (0: goal, 1: subgoal, 2: action, 3: history) |
| `history_offsets` | I `[G + 1]` | graph `g`'s history steps are `[history_offsets[g], history_offsets[g + 1])`; present iff the schema has history |
| `history_entity` | I `[S]` | the entity of each history step, graph after graph, oldest first |
| `history_dt` | int32 `[S]` | the `dt` of each history step |

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

- **`FlatStream.append(states, task_ids=None, *, derived=None, goals=None, actions=None, subgoals=None, history=None)`** takes the arguments of
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
- **Subgoals and history.** `subgoals` and `history` are as in `encode` and are optional per row: a row without them has
  empty layers and no history, the same batch as `encode` gives for it. A stream stores `max_goal_level` layers per row
  and the valid steps of each history, and a flush pads the histories to the longest, so rows with histories of every
  length can share a stream.
- **Per-row inputs.** `derived`, `goals` and `actions` are as in `encode`, with one rule: every row of a stream gives
  the same inputs. The first rows appended to a stream without rows decide (a stream with `goals` keeps a goal for every
  row, a stream with `actions` the action rows of every state, a single `[1, W]` goal is copied for every row), and a
  later `append` or `update` that gives other inputs raises `ValueError`. A stream that holds no rows again takes
  any inputs. Rows narrower than their instance are refused when appended, with the same errors as `encode`; errors
  that need the encoder (a goal bit with no atom slot, action labels of another instance, an action order that is not
  canonical) are raised by `flush`, which keeps the rows.
- **Action rows.** The action rows of the appended expansions are copied and joined in row order, so a flushed batch is
  that of one expansion over all rows: the `entity_ref` of an action entity is what `encode` gives for the same states
  and action rows. The row of an action entity in the expansion it was appended with is `x.offsets[k] + entity_ref`,
  where `x` is the expansion given to the `append` (or `update`) that stored the state and `k` is the state's row in
  that call, so `x.action(x.offsets[k] + entity_ref)` is the entity's action. The flushed batch does not number its
  graphs as `x` does: graph `j` of a flush is the `j`-th row appended, and of a mutable flush the row of `indices()[j]`.
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
  The derived words must be given when a graph reads derived atoms (a schema with derived relations, derived goal
  literals or subgoal layers over derived atoms): the device does not evaluate derived atoms.
- **Subgoal layers and history.** `subgoals=` and `history=` take CUDA tensors of the forms of the host call (the
  masks `[N, L, W]` of 64-bit words, `[1, L, W]` for subgoal layers that every row shares, `dt` `int32` `[N, H]` and the
  valid step `count` `int32` `[N]`, with the derived masks last), and the device encodes the subgoal levels, the history
  steps and their targets exactly as the host does: the levels as layers minus the goal and the lower layers, the newest
  `history_max_steps` of each row's valid steps (all of them when it is not set), history targets numbered relation,
  then step, then slot. A row may use fewer layers than the call has (an empty layer adds nothing) and have fewer valid
  steps than `H`, down to none. The subgoal levels of a graph are computed once into the workspace's scratch, so a
  schema with levels needs scratch of three planes per level and sign over the instance's words for every graph of a
  batch, on top of the scratch of a schema without them.
- **Refused.** `actions=` (the action relations are not encoded on the device), `pool=` (a pool of host threads
  means nothing for device rows) and a `framework` other than `'torch'` raise `ValueError` with the `mymyr: encoding:`
  prefix, as do an encoder with action targets or a horizon mode and layers or history given to an encoder whose
  schema has no subgoal or history relations; `threads` has no effect. A row that is invalid (a task id outside the
  table, a row, goal mask, subgoal layer or history step narrower than its instance, a goal, layer or step bit its
  instance has no atom for, a history count outside `[0, H]`) raises `ValueError` without capacities and sets
  `invalid_input` in `sizes` with them. The steps of a history that the row does not use (beyond its count, or older
  than `history_max_steps` newest) are not read.
- **One wait without capacities.** An encoding without `capacities` must size its arrays by the batch: it counts the
  batch on the stream, copies the sizes to the host and waits for them (the only host wait of the call), allocates
  exact arrays and fills them. A batch that needs more indices than `int32` holds raises `OverflowError`; use
  `index_dtype='int64'`.
- **No wait with capacities.** With `capacities=dict(graphs=, entities=, args=, targets=, history=)` (`targets` only with
  target sources, `history` only with a history) every variable-size array is allocated at its capacity, zero-filled,
  and the kernels write the true sizes into `batch.device_sizes` (an `int64` `[6]` tensor: entities, tuples, args,
  targets, history steps and the overflow flags, 1 for a capacity, 2 for the index type, 4 for an invalid row). The call never waits and can be captured in a CUDA graph. A batch
  that does not fit its capacities stays all zero with the flag set. `batch.sizes` reads `device_sizes` into a dict (and
  waits).
- **`DeviceFlatBatch`** has the accessors of `FlatBatch`: the arrays (an array the schema does not store raises
  `AttributeError`), `num_graphs`, `num_relations`, `schema`, `implicit_entities`, `as_dict(framework=None)` (torch
  tensors, `'dlpack'` or, copying to the host, `'numpy'`) and `device`. `encoder.entity_names(batch, g)` and
  `encoder.describe(batch, g)` accept it and copy it to the host.
- **Workspaces.** Exact encodes keep one device workspace per stream, made by the first exact call and reused after it;
  a batch of more rows than the workspace
  holds makes the next call reserve a larger one (the size is a power of two, at least 1024 rows).
  `encoder.reserve_device(graphs, *, device=None, stream=None)` reserves exact-call scratch in advance and
  `encoder.release_device(*, device=None, stream=None)` frees the workspace of a stream, or of every stream of the
  device. Workspaces are freed with the encoder. Capacity encodes allocate scratch through torch's caching allocator
  per call; captured scratch belongs to the graph's private pool and survives later calls and workspace growth.

### The torch op

`torch.ops.mymyr.encode_flat(encoder, states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg,
subgoal_pos, subgoal_neg, subgoal_derived_pos, subgoal_derived_neg, history_pos, history_neg, history_dt, history_count,
history_derived_pos, history_derived_neg, graphs, entities, args, targets, history)` is the capacity form as a torch
custom op, for `torch.compile` and CUDA graphs (`import mymyr.rl.torch` registers it). `encoder` is the handle
`mymyr.rl.torch.register(encoder)` returns; the tensors are as above (`None` for an input that is absent) and the five
ints are the capacities (`targets` 0 without target sources, `history` 0 without history). It returns the list of 17
tensors that `DeviceFlatBatch.array_names` names, the arrays in the order of the batch table and `sizes` last; an array the schema does not store is an empty tensor, and
`DeviceFlatBatch.from_arrays(encoder, outputs)` wraps the list into a batch. The op waits for nothing, allocates only
its outputs and scratch and has a fake implementation that gives their shapes from the capacities, so `torch.compile(fullgraph=True)`
traces it without a graph break. The registry keeps the encoder alive until `unregister(handle)`. A compiled function
specializes the handle: calling it with another registered encoder traces a graph for that encoder's schema.

To capture the op (or `encode` with capacities) in a `torch.cuda.CUDAGraph`, warm it once on the capture stream to upload
and acquire its context. No scratch reservation is needed. An encoding without capacities waits for the sizes and
cannot be captured. The graph reads its input tensors where they were captured: write new rows in place and replay.
Keep the encoder registered or referenced while the graph can replay: a captured `encode` holds raw context pointers
and does not retain the Python encoder. Graphs of different sizes can share a capture stream; each owns its scratch.

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
| subgoal | `[+](on c b)[sg]`: like a goal, with the level of its relation (`[sg]`, `[ssg]`, `[sssg]`) |
| action | `(unstack a b)`: the action schema, then the objects of its action tuple |
| history step | `history:-2#0`: the `dt` of the step and its index among the graph's encoded steps (`history_entity`), counted from 0 |
| history target | `history:-2#0:[+](on c b)`: the step, then the sign, predicate and objects of the history tuple |
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
  targets together with a horizon mode; history targets or `history_max_steps` without `history`; a `history_max_steps` of 0; `history` together with a
  horizon mode; the action horizon mode without `include_actions`; `parent_relation` or a
  `root_policy` other than `include` without a horizon mode.
- **Fingerprint.** `schema_fingerprint` hashes the catalog (names, lanes, arities, slot roles) and the options that
  shape the output but not the catalog (target sources, layout, index type, history and `history_max_steps`, horizon
  mode and root policy), and which arrays are omitted: `relation_counts` (`emit_counts=False`) and `entity_role` /
  `entity_ref` (implicit entity arrays); an encoding that stores every array has one fingerprint however it was asked
  for. It does not depend on declaration order, on predicate ids, on the process or on the build.

## C++

`mymyr/encoding/encoder.hpp`, linked with `mymyr::encoding`: `FlatEncoder::create(table_or_task, options)` returns the
encoder, and `encode(EncodeInput{states, task_ids, derived, derived_stride, goal_pos, goal_neg, goal_stride, goal_words, ..., actions, subgoals, history})`
returns the batch as an `rl::ArrayBundle` with the arrays and scalars above. `mymyr/encoding/validate.hpp` checks a
batch against the contract. `encode(in, pool, scratch)` takes a `ThreadPool` (the batch is encoded by up to
`ceil(graphs / k_graphs_per_member)` of its members, on the calling thread alone for a small batch, byte-identically for
any pool) and a reusable `EncodeScratch`. `count(in, pool, scratch)` returns the `FlatSizes` of a batch (entities,
tuples, args, targets, history); `fill(in, sizes, buffers, pool, scratch)` then writes it into caller-provided
`FlatBuffers`, and `encode_into(in, capacities, buffers, pool, scratch)` does both within
`BatchShape{graphs, entities, args, targets, history}`, reporting `overflow` in the
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
held; their methods take a mutex, and a flush encodes with the calling thread's `EncodeScratch`, which it keeps between calls.

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
