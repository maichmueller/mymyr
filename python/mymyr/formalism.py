"""The normalized task and its formulas.

Read-only views of the normalized task: every type, object, predicate, function, schema, axiom and initial/goal entry.
Views keep their task alive; they are immutable, hashable and cheap.

Formulas are values over the predicates, objects, types and functions of one task: Variable, Atom, Literal and
ConjunctiveCondition (lifted), GroundAtom, GroundLiteral and GroundCondition (ground), and the Expression and
NumericConstraint views of their numeric constraints. A schema's precondition is a ConjunctiveCondition, the goal a
GroundCondition, the initial atoms GroundAtoms; Task.atom, Task.literal, Task.condition and Task.ground_condition make
new ones.
"""

from mymyr._core._formalism import (
    Atom,
    Axiom,
    ConditionalEffect,
    ConjunctiveCondition,
    Expression,
    Function,
    GroundAtom,
    GroundCondition,
    GroundFunctionValue,
    GroundLiteral,
    Literal,
    NormalizedTask,
    NumericConstraint,
    NumericEffect,
    Object,
    Parameter,
    Predicate,
    Schema,
    Type,
    Variable,
    read_task_text,
)

CLASSES = (
    Type, Object, Variable, Parameter, Predicate, Function, Atom, Literal, GroundAtom, GroundLiteral, Expression,
    NumericConstraint, NumericEffect, ConjunctiveCondition, GroundCondition, ConditionalEffect, Schema, Axiom,
    GroundFunctionValue, NormalizedTask,
)
"""Every class of this module; each has a `fields` tuple naming its data attributes."""

__all__ = [c.__name__ for c in CLASSES] + ["CLASSES", "read_task_text"]
