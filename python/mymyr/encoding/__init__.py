"""Native relational encodings of states, independent of model frameworks.

FlatRelationEncoder encodes states of a Task or a one-domain rl.TaskTable into graphs: one graph per state, with the
objects as entities and, for every relation of its EncodingSchema, the tuples of the state's atoms. The result is a
FlatBatch of read-only arrays (batch.as_dict(framework) exports them to NumPy, torch, JAX or DLPack without a copy).
Numeric tasks are rejected.

encoder.stream() and encoder.mutable_stream() collect rows over time and encode them in one call per flush
(FlatStream, FlatMutableStream); encoder.entity_names(batch, g) and encoder.describe(batch, g) name the entities and
tuples of one graph. An encoder pickles as its table and options.

EncodingSchema is the relation catalog of an encoding: the relations a batch holds, sorted by name, with their
arities, lanes and slot roles, and a fingerprint. It is a function of the domain and the options alone.
"""

from mymyr._core._encoding import EncodingSchema, FlatBatch, FlatMutableStream, FlatRelationEncoder, FlatStream
from mymyr.encoding._device import DeviceFlatBatch
from mymyr.encoding._names import GraphDescription

__all__ = [
    "DeviceFlatBatch",
    "EncodingSchema",
    "FlatBatch",
    "FlatMutableStream",
    "FlatRelationEncoder",
    "FlatStream",
    "GraphDescription",
]
