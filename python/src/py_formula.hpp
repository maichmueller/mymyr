#pragma once
// The formulas of mymyr.formalism (formula_bindings.cpp registers them): values over the predicates, objects, types and
// functions of one normalized task (successor/conditions.hpp).
//   - Variable, Atom and Literal (lifted: terms are objects or variables), ConjunctiveCondition.
//   - GroundAtom, GroundLiteral, GroundCondition.
//   - Expression and NumericConstraint: views of the expression trees of a task or a condition.
// A value holds its task's data and, when a Task (or TaskHandle) made it, that owner: the owner pickles it. Equality and
// hashing use the task data's identity plus the content, never the owner. Operations in a state (holds, ground) take the
// task from the state.

#include "formalism_views.hpp"
#include "py_task.hpp"

#include "mymyr/successor/conditions.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mymyr::python
{
namespace nb = nanobind;

using DataPtr = std::shared_ptr<const formalism::TaskData>;
using Variables = std::vector<ConjunctiveCondition::Variable>;
using VariablesPtr = std::shared_ptr<const Variables>;

struct FormulaOwner
{
    DataPtr data;
    nb::object task;  // the Task or TaskHandle that made the value, or None
    PyTaskCore* core = nullptr;  // its core (null without one)
};

/// A variable of a condition (or of a lifted atom made on its own): its position in the variable list.
struct PyVariable
{
    FormulaOwner o;
    u32 position = 0;
    std::string name;  // without '?'
    std::vector<TypeId> types;
};

/// A lifted atom; a variable term (>= 0) indexes `vars`.
struct PyLiftedAtom
{
    FormulaOwner o;
    PredicateId predicate;
    std::vector<formalism::Term> terms;
    VariablesPtr vars;
};

struct PyLiteral
{
    PyLiftedAtom atom;
    bool positive = true;
};

struct PyGroundAtom
{
    FormulaOwner o;
    GroundAtom a;
};

struct PyGroundLiteral
{
    FormulaOwner o;
    GroundLiteral l;
};

struct PyConjunctiveCondition
{
    FormulaOwner o;
    std::shared_ptr<const ConjunctiveCondition> c;
};

struct PyGroundCondition
{
    FormulaOwner o;
    std::shared_ptr<const GroundCondition> c;
};

/// The expression pools an Expression or NumericConstraint indexes: the task's (formalism::TaskData) or a condition's.
struct ExprPool
{
    FormulaOwner o;
    std::shared_ptr<const void> keep;  // owns the pools
    const std::vector<formalism::Expr>* exprs = nullptr;
    const std::vector<formalism::Term>* terms = nullptr;
    const std::vector<formalism::NumericConstraint>* constraints = nullptr;
    VariablesPtr vars;  // the variables of variable terms (null: none in scope)
};
struct PyExpression
{
    ExprPool p;
    u32 i = 0;
};
struct PyNumericConstraint
{
    ExprPool p;
    u32 i = 0;
};

/// A ground atom: a GroundAtom, '(on a b)', ('on', 'a', 'b'), ('on', ['a', 'b']), or a fluent slot.
using AtomTuple = nb::typed<nb::tuple, std::variant<std::string, int, nb::typed<nb::sequence, ObjectKey>>, nb::ellipsis>;
using AtomLike = Arg<std::variant<PyGroundAtom, std::string, int, AtomTuple>>;
/// A ground literal: a GroundLiteral, a ground atom (positive), or '(not (on a b))'.
using GroundLiteralLike = std::variant<PyGroundLiteral, PyGroundAtom, std::string, AtomTuple>;

// ------------------------------------------------------------------------------------------------ makers

[[nodiscard]] FormulaOwner bare_owner(DataPtr data);
[[nodiscard]] FormulaOwner task_owner(const Owner& o);
/// The variables of a parameter scope of the normalized task: `first`, then `second` (a conditional effect's).
[[nodiscard]] VariablesPtr scope_variables(const formalism::TaskData& d, formalism::Range first, formalism::Range second = {});
/// An Object (a term < 0) or a Variable (a term >= 0) of `vars`.
[[nodiscard]] nb::object term_object(const FormulaOwner& o, formalism::Term x, const VariablesPtr& vars);
/// The task's pools (exprs, terms, constraints) with the variables of a scope.
[[nodiscard]] ExprPool task_pool(const FormulaOwner& o, VariablesPtr vars);

[[nodiscard]] nb::object make_lifted_atom(const FormulaOwner& o, PredicateId predicate, std::vector<formalism::Term> terms,
                                          VariablesPtr vars);
[[nodiscard]] nb::object make_literal(const FormulaOwner& o, PredicateId predicate, bool positive,
                                      std::vector<formalism::Term> terms, VariablesPtr vars);
[[nodiscard]] nb::object make_ground_atom(const FormulaOwner& o, GroundAtom a);
[[nodiscard]] nb::object make_ground_literal(const FormulaOwner& o, GroundLiteral l);
[[nodiscard]] nb::object make_conjunctive_condition(const FormulaOwner& o, ConjunctiveCondition c);
[[nodiscard]] nb::object make_ground_condition(const FormulaOwner& o, GroundCondition c);

/// PDDL text of a term, literal or condition part (variables as "?name").
[[nodiscard]] std::string literal_text(const formalism::TaskData& d, PredicateId predicate, bool positive,
                                       std::span<const formalism::Term> terms, const Variables* vars);
[[nodiscard]] std::string ground_atom_text(const formalism::TaskData& d, const GroundAtom& a);

/// The task of a state that a value of `o` is evaluated in: ValueError if the state belongs to another task.
[[nodiscard]] const PyState& formula_state(const FormulaOwner& o, nb::handle state);

// ------------------------------------------------------------------------------------------------ from task_bindings.cpp

/// A partial binding of a condition (Task.bindings' partial=), std::nullopt per free variable; empty for None.
[[nodiscard]] std::vector<std::optional<ObjectId>> condition_partial(PyTaskCore& core, const std::shared_ptr<const ConjunctiveCondition>& c,
                                                                     nb::handle partial);
/// A limit (None: no limit).
[[nodiscard]] u64 limit_value(nb::handle limit);
/// The ground atom of anything Task.atom accepts that is ground; ValueError for a lifted atom.
[[nodiscard]] GroundAtom ground_atom_of(const Owner& o, nb::handle atom);

/// Registers the formula classes in mymyr._core._formalism (m) and _restore_formula in mymyr._core (parent).
void bind_formulas(nb::module_& m, nb::module_& parent);
}  // namespace mymyr::python
