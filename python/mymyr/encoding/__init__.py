"""Native packed relational observations, independent of model frameworks.

FlatRelationEncoder accepts one Task or a one-domain rl.TaskTable. Its owned
FlatBatch arrays encode classical facts, signed goal satisfaction and optional
action labels from an existing rl.Expansion. Numeric tasks are rejected.
"""

from mymyr._core._encoding import FlatBatch, FlatRelationEncoder

__all__ = ["FlatBatch", "FlatRelationEncoder"]
