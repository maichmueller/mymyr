"""Native relational encodings of states, independent of model frameworks.

FlatRelationEncoder encodes states of a Task or a one-domain rl.TaskTable into graphs: one graph per state, with the
objects as entities and, for every relation of its EncodingSchema, the tuples of the state's atoms. The result is a
FlatBatch of read-only arrays (batch.as_dict(framework) exports them to NumPy, torch, JAX or DLPack without a copy).
Numeric tasks are rejected.

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
themselves. Streams and the device do not encode horizons.

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
)
from mymyr.encoding._device import DeviceCollator, DeviceFlatBatch, DeviceFlatStream
from mymyr.encoding._horizon import HorizonView, horizon, horizon_of_space
from mymyr.encoding._literals import history_masks, subgoal_masks
from mymyr.encoding._names import GraphDescription

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
    "HorizonView",
    "collate",
    "history_masks",
    "horizon",
    "horizon_of_space",
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
