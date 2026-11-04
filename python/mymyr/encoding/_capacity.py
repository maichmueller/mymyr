"""Static capacity bounds of an encoder: FlatRelationEncoder.capacity_bounds."""

from __future__ import annotations

from typing import Any, Optional

from mymyr._core import _encoding
from mymyr._core._rl import TaskSuite

# which capacities some batch of the rows reaches exactly; the others are upper bounds
_EXACT = {
    "graphs": True,
    "entities": False,
    "args": False,
    "targets": False,
    "history": True,
    "numeric": True,
    "static_instances": True,
    "static_args": True,
    "successors": False,
}


class CapacityBounds(dict):
    """Capacities that no batch of ``graphs`` rows over a table exceeds, from ``FlatRelationEncoder.capacity_bounds``.

    The dict holds the keys that the encoder's ``capacities`` need (``graphs``, ``entities``, ``args``, and
    ``targets``, ``history``, ``numeric``, ``static_instances`` / ``static_args`` where the schema has them) and, when
    the bounds were made with ``actions`` and ``drop_noops``, ``successors``: pass it as ``capacities=`` of ``encode``
    or as ``BatchedEnv(obs_capacities=...)``. A value is None where no bound is known (the action bounds of an instance
    whose grounding exceeds its budget; ``reason`` says why), and such a dict is refused as capacities.

    - ``relation_tuples``: ``{relation name: tuples}`` in catalog order, the tuples of each relation in the rows;
    - ``relation_exact``: ``{relation name: bool}``, True where some batch has exactly that many tuples (the static
      and numeric relations), False for an upper bound;
    - ``exact``: ``{key: bool}`` for the keys of the dict, likewise (``graphs``, ``history``, ``numeric`` and the
      static blocks are exact, the others upper bounds);
    - ``unbounded``: the keys and relation names without a bound;
    - ``reason``: why a bound is unknown (empty when all are known).
    """

    __slots__ = ("relation_tuples", "relation_exact", "exact", "unbounded", "reason")

    def __init__(self, values: dict, relation_tuples: dict, relation_exact: dict, reason: str) -> None:
        super().__init__(values)
        self.relation_tuples = relation_tuples
        self.relation_exact = relation_exact
        self.exact = {key: _EXACT[key] for key in values}
        self.unbounded = tuple(k for k, v in values.items() if v is None) + tuple(
            k for k, v in relation_tuples.items() if v is None)
        self.reason = reason

    def __reduce__(self) -> tuple:
        return (CapacityBounds, (dict(self), self.relation_tuples, self.relation_exact, self.reason))

    def __repr__(self) -> str:
        return f"CapacityBounds({dict.__repr__(self)})"


def _bigger(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else max(a, b)


def _tables(encoder: Any, table_or_suite: Any) -> list:
    """The tables to bound: ``table_or_suite`` itself, or the tables of a suite whose catalog is the encoder's."""
    if not isinstance(table_or_suite, TaskSuite):
        return [table_or_suite]
    options = encoder.options
    tables = [t for t in table_or_suite.tables
              if _encoding.EncodingSchema(t, **options).schema_fingerprint == encoder.schema_fingerprint]
    if not tables:
        raise ValueError("mymyr: encoding: the suite has no table of this encoder's domain")
    return tables


def _count(name: str, value: Any, least: int) -> int:
    if isinstance(value, bool) or not hasattr(value, "__index__") or value.__index__() < least:
        raise ValueError(f"mymyr: encoding: capacity_bounds needs an int {name} of at least {least}, not {value!r}")
    return value.__index__()


def capacity_bounds(encoder: Any, table_or_suite: Any, graphs: Any, actions: bool = True, drop_noops: bool = False,
                    history: Any = 0) -> CapacityBounds:
    """The bounds of ``encoder.capacity_bounds`` (its docstring has the arguments)."""
    graphs = _count("graphs", graphs, 1)
    history = _count("history", history, 0)
    merged: Optional[dict] = None
    for table in _tables(encoder, table_or_suite):
        counts = _encoding._capacity_counts(encoder, table, graphs, bool(actions), history)
        if merged is None:
            merged = counts
            continue
        for key, value in counts.items():
            if key == "relation_tuples":
                merged[key] = [_bigger(a, b) for a, b in zip(merged[key], value)]
            elif key == "relation_exact":
                merged[key] = [a and b for a, b in zip(merged[key], value)]
            elif key == "reason":
                merged[key] = merged[key] or value
            else:
                merged[key] = _bigger(merged[key], value)
    assert merged is not None
    names = encoder.relation_names
    tuples = dict(zip(names, merged.pop("relation_tuples"), strict=True))
    exact = dict(zip(names, merged.pop("relation_exact"), strict=True))
    reason = merged.pop("reason")
    if not drop_noops:
        # without the drop the kept rows are all the successors, which the entity capacity bounds
        merged.pop("successors", None)
    return CapacityBounds(merged, tuples, exact, reason)
