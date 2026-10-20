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

``hgraph(batch)`` views a relation-major batch as a heterogeneous graph (HGraphView): a node type per relation and one
for the entities, an edge type per relation and tuple position, the edges' entity ids a strided view of the batch's
arguments. The view copies no tuple and keeps the batch alive.

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
)
from mymyr.encoding._device import DeviceFlatBatch
from mymyr.encoding._hgraph import hgraph
from mymyr.encoding._horizon import HorizonView, horizon, horizon_of_space
from mymyr.encoding._literals import history_masks, subgoal_masks
from mymyr.encoding._names import GraphDescription

__all__ = [
    "DeviceFlatBatch",
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
    "hgraph",
    "history_masks",
    "horizon",
    "horizon_of_space",
    "subgoal_masks",
]
