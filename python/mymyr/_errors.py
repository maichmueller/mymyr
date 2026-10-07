"""mymyr.PddlError: PDDL that mymyr cannot read."""

from __future__ import annotations

__all__ = ["PddlError"]


class PddlError(ValueError):
    """PDDL that mymyr cannot read: a syntax error, an undefined or mismatched name, a wrong number of arguments, or a
    construct mymyr does not support (durative actions, processes and events, preferences, trajectory constraints,
    object fluents, timed initial literals, non-deterministic or probabilistic effects, an unknown requirement flag).

    ``str(error)`` is one line, ``"<path>:<line>: <message> (in action <name>)"`` with the parts that are known.
    A file that cannot be opened raises OSError (FileNotFoundError) instead.

    Attributes:
        message: what is wrong, without the location.
        path: the file, or None for PDDL given as a string.
        line: the 1-based line in the file, or None when unknown.
        action: the action the error is in, or None.
    """

    message: str
    path: str | None
    line: int | None
    action: str | None

    def __init__(self, message: str, path: str | None = None, line: int | None = None, action: str | None = None):
        super().__init__(message, path, line, action)
        self.message = message
        self.path = path
        self.line = line
        self.action = action

    def __str__(self) -> str:
        where = self.path or ""
        if self.line is not None:
            where += f":{self.line}" if where else f"line {self.line}"
        out = f"{where}: {self.message}" if where else self.message
        if self.action:
            out += f" (in action {self.action})"
        return out
