"""The serialized form of an encoding schema: a plain dictionary of options, catalog, fingerprint and version."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mymyr._core import _encoding as _core

if TYPE_CHECKING:
    from mymyr.encoding import EncodingSchema
    from mymyr.rl import TaskTable

__all__ = ["schema_from_dict", "schema_to_dict"]

_CATALOG = {
    "names": "relation_names",
    "arities": "relation_arities",
    "logical_arities": "relation_logical_arities",
    "lanes": "relation_lanes",
    "signs": "relation_signs",
    "levels": "relation_levels",
    "derivations": "relation_derivations",
    "slot_roles": "relation_slot_roles",
}


def _plain(value: Any) -> Any:
    """`value` with tuples and sets as lists, so that it equals its own JSON round trip."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    return value


def schema_to_dict(schema: EncodingSchema) -> dict[str, Any]:
    """The serialized form of a schema, a dictionary that ``json`` writes and reads back unchanged:

    - ``version``: the version of the catalog rules (``schema.version``);
    - ``options``: the schema's options in normal form (``schema.options``), including ``repeated_literals``;
    - ``fingerprint``: ``schema.schema_fingerprint`` as 16 hex digits (the unsigned 64-bit value);
    - ``catalog``: the relations in id order, one list per column: ``names``, ``arities``, ``logical_arities``,
      ``lanes``, ``signs``, ``levels``, ``derivations`` and ``slot_roles`` (a list of role codes per relation, named
      by ``schema.slot_role_names``).

    A schema is a function of a domain and its options, so the record holds no domain: ``schema_from_dict(record,
    table)`` rebuilds the schema over a table of the domain and checks it against the record.
    """
    return {
        "version": schema.version,
        "options": _plain(schema.options),
        "fingerprint": f"{schema.schema_fingerprint & (2**64 - 1):016x}",
        "catalog": {key: _plain(getattr(schema, attribute)) for key, attribute in _CATALOG.items()},
    }


def _bad(message: str) -> ValueError:
    return ValueError(f"mymyr: encoding: {message}")


def _check_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise _bad(f"a schema record is a dict, not {type(record).__name__}")
    for key, kind in (("version", int), ("options", dict), ("fingerprint", str), ("catalog", dict)):
        if not isinstance(record.get(key), kind) or isinstance(record.get(key), bool):
            raise _bad(f"the schema record has no {kind.__name__} '{key}'")
    try:
        int(record["fingerprint"], 16)
    except ValueError:
        raise _bad("the schema record's fingerprint is not a hexadecimal number") from None
    catalog = record["catalog"]
    for key in _CATALOG:
        if not isinstance(catalog.get(key), list):
            raise _bad(f"the schema record's catalog has no list '{key}'")
    if len({len(catalog[key]) for key in _CATALOG}) != 1:
        raise _bad("the columns of the schema record's catalog differ in length")


def _first_difference(recorded: dict[str, list[Any]], rebuilt: dict[str, list[Any]]) -> str:
    """Where the catalog of a record and the rebuilt one first differ, or an empty string."""
    if len(recorded["names"]) != len(rebuilt["names"]):
        return f"the record has {len(recorded['names'])} relations, the schema over the table {len(rebuilt['names'])}"
    for key in _CATALOG:
        for at, (a, b) in enumerate(zip(recorded[key], rebuilt[key])):
            if a != b:
                return (
                    f"relation {at} ({rebuilt['names'][at]!r}) differs in its {key}: the record has {a!r}, the schema "
                    f"over the table {b!r}"
                )
    return ""


def schema_from_dict(record: Mapping[str, Any], table: TaskTable | Any) -> EncodingSchema:
    """The schema that ``record`` (the output of ``schema_to_dict``, also after a JSON round trip) describes, built over
    ``table`` (a Task, TaskHandle or one-domain TaskTable of the domain it was made for) with the record's options.

    Raises ``ValueError`` ("mymyr: encoding: ...") for a malformed record, for a record of another version of the
    catalog rules, and when the schema rebuilt over ``table`` does not have the record's fingerprint or catalog (the
    message names the first relation that differs): the domain, the options' meaning or the rules have changed.
    """
    _check_record(record)
    record = dict(record)
    schema = _core.EncodingSchema(table, **record["options"])
    if record["version"] != schema.version:
        raise _bad(
            f"the schema record is of version {record['version']} of the catalog rules, this build has version "
            f"{schema.version}"
        )
    rebuilt = schema_to_dict(schema)
    difference = _first_difference(record["catalog"], rebuilt["catalog"])
    if difference or rebuilt["fingerprint"] != record["fingerprint"].lower().rjust(16, "0"):
        raise _bad(
            f"the schema record's fingerprint is {record['fingerprint']}, the schema rebuilt over the table has "
            f"{rebuilt['fingerprint']} ("
            + (difference or "the catalogs agree: an option that shapes the arrays, or the rules, differ")
            + ")"
        )
    return schema
