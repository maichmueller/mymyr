"""Native relational encodings of states, independent of model frameworks.

FlatRelationEncoder encodes states of a Task or a one-domain rl.TaskTable into graphs: one graph per state, with the
objects as entities and, for every relation of its EncodingSchema, the tuples of the state's atoms. The result is a
FlatBatch of read-only arrays (batch.as_dict(framework) exports them to NumPy, torch, JAX or DLPack without a copy).
Numeric tasks are rejected.

EncodingSchema is the relation catalog of an encoding: the relations a batch holds, sorted by name, with their
arities, lanes and slot roles, and a fingerprint. It is a function of the domain and the options alone.
"""

from mymyr._core._encoding import EncodingSchema, FlatBatch, FlatRelationEncoder

__all__ = ["EncodingSchema", "FlatBatch", "FlatRelationEncoder"]
