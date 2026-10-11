"""Native packed relational observations, independent of model frameworks.

FlatRelationEncoder accepts one Task or a one-domain rl.TaskTable. Its owned
FlatBatch arrays encode classical facts, signed goal satisfaction and optional
action labels from an existing rl.Expansion. Numeric tasks are rejected.

EncodingSchema is the relation catalog of an encoding: the relations a batch holds, sorted by name, with their
arities, lanes and slot roles, and a fingerprint. It is a function of the domain and the options alone.
"""

from mymyr._core._encoding import EncodingSchema, FlatBatch, FlatRelationEncoder

__all__ = ["EncodingSchema", "FlatBatch", "FlatRelationEncoder"]
