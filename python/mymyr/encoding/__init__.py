"""Native relational encodings of states, independent of model frameworks.

FlatRelationEncoder encodes states of a Task or a one-domain rl.TaskTable into graphs: one graph per state, with the
objects as entities and, for every relation of its EncodingSchema, the tuples of the state's atoms. The result is a
FlatBatch of read-only arrays (batch.as_dict(framework) exports them to NumPy, torch, JAX or DLPack without a copy).
For a numeric task, every fluent numeric function f has a relation ``[num]f`` with a tuple for each of its ground terms
in the instance, and ``batch.numeric_values`` holds their values (``numeric_dtype``; NaN for a term without a value).

encoder.stream() and encoder.mutable_stream() collect rows over time and encode them in one call per flush
(FlatStream, FlatMutableStream); encoder.entity_names(batch, g) and encoder.describe(batch, g) name the entities and
tuples of one graph. An encoder pickles as its table and options.

The rows of an encoding can carry subgoal layers and a history (``encode(..., subgoals=..., history=...)``, also on the
streams): arrays of masks over the atom slots, which ``subgoal_masks`` and ``history_masks`` build from lists of
literals.

FlatHorizonEncoder encodes horizons: a root state and the nodes below it, given as a HorizonView (node rows, offsets and
edges; ``rl.expand`` rows have this form), into one graph each with the facts of the nodes, the goal's satisfaction at
them, their deltas to the root and the action and parent edges. ``horizon(table, roots, depth)`` and
``horizon_of_space(space, roots, depth)`` build the HorizonView of states by expanding them or from a state space (nodes
merged by state, in the order they are reached; ``max_nodes`` bounds the size). FlatRootedHorizonEncoder is the
horizon encoder for rooted graphs, with the root as an encoded node, the ``_parent_`` relation and, on request, the
``_sibling_`` and ``_cousin_`` relations between nodes. FlatTransitionEncoder and FlatTransitionEffectsEncoder encode
the transitions of states (the horizons of depth one, with the successors' facts or their deltas) from the states
themselves. Streams do not encode horizons; the device encodes depth-one horizons into torch arrays.

SimpleEncoder encodes states into homogeneous graphs with integer node and edge labels, the form that standard graph
neural networks consume: a node for each entity and for each tuple of the relational encoding and, for every position
of a tuple, an edge from its entity to the tuple's node and one back (``outputs=('relations', 'simple')`` on any
encoder gives the Simple graph next to the relation arrays). ``schema.simple_label_names`` and
``schema.simple_edge_label_names`` name the labels.

``hgraph(batch)`` views a relation-major batch as a heterogeneous graph (HGraphView): a node type per relation and one
for the entities, an edge type per relation and tuple position, the edges' entity ids a strided view of the batch's
arguments. The view copies no tuple and keeps the batch alive.

``action_rows``, ``action_indices`` and ``action_entities`` map the action entities of a batch (the actions a policy
chooses among) back to the rows of the expansion they were encoded from and to the actions an environment takes.
``SuiteEncoder`` encodes the rows of a task suite, one encoder and one batch per domain, with each row's place in its
batch (``SuiteBatch``). ``encode(..., capacities=..., into=batch)`` writes into the memory of an earlier batch of the
same capacities (the arrays it exported change with it). ``schema_to_dict`` and ``schema_from_dict`` serialize a
schema as options, catalog, fingerprint and version and check it when it is rebuilt over a table;
``FlatBatch.from_arrays(schema, arrays)`` reads back a batch from ``batch.as_dict()`` and checks it against the schema.

EncodingSchema is the relation catalog of an encoding: the relations a batch holds, sorted by name, with their
arities, lanes and slot roles, and a fingerprint. It is a function of the domain and the options alone.
"""

from mymyr._core._encoding import (
    EncodingSchema,
    FlatBatch,
    FlatHorizonEncoder,
    FlatMutableStream,
    FlatRelationEncoder,
    FlatRootedHorizonEncoder,
    FlatStream,
    FlatTransitionEffectsEncoder,
    FlatTransitionEncoder,
    HGraphView,
    SimpleEncoder,
)
from mymyr.encoding._actions import action_entities, action_indices, action_rows
from mymyr.encoding._device import DeviceCollator, DeviceFlatBatch, DeviceFlatStream
from mymyr.encoding._hgraph import hgraph
from mymyr.encoding._horizon import HorizonView, horizon, horizon_of_space
from mymyr.encoding._literals import history_masks, subgoal_masks
from mymyr.encoding._names import GraphDescription
from mymyr.encoding._record import schema_from_dict, schema_to_dict
from mymyr.encoding._suite import SuiteBatch, SuiteEncoder

__all__ = [
    "DeviceCollator",
    "DeviceFlatBatch",
    "DeviceFlatStream",
    "EncodingSchema",
    "FlatBatch",
    "FlatHorizonEncoder",
    "FlatMutableStream",
    "FlatRelationEncoder",
    "FlatRootedHorizonEncoder",
    "FlatStream",
    "FlatTransitionEffectsEncoder",
    "FlatTransitionEncoder",
    "GraphDescription",
    "HGraphView",
    "HorizonView",
    "SimpleEncoder",
    "SuiteBatch",
    "SuiteEncoder",
    "action_entities",
    "action_indices",
    "action_rows",
    "collate",
    "hgraph",
    "history_masks",
    "horizon",
    "horizon_of_space",
    "schema_from_dict",
    "schema_to_dict",
    "subgoal_masks",
]


def collate(batches, *, capacities=None):
    """Concatenate encoded graphs in batch order, rebuilding offsets and shifting entity ids.

    Batches must share a table, schema, layout, index type and static representation. Empty slices are allowed;
    the list itself must contain a batch to establish the schema. Host and CUDA batches cannot be mixed.
    """
    batches = list(batches)
    if not batches:
        raise ValueError("mymyr: encoding: collate needs at least one batch")
    if all(isinstance(batch, FlatBatch) for batch in batches):
        if capacities is not None:
            raise ValueError("mymyr: encoding: host collation allocates exactly; capacities are for device batches")
        from mymyr._core._encoding import _collate
        return _collate(batches)
    if all(isinstance(batch, DeviceFlatBatch) for batch in batches):
        from mymyr.encoding._device import collate_device
        return collate_device(batches, capacities)
    raise ValueError("mymyr: encoding: collate received mixed host and device batches or an invalid batch")
