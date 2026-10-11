#pragma once
// Conditions as values over the predicates, objects, types and functions of one task: lifted conjunctive conditions
// (ConjunctiveCondition) and ground ones (GroundAtom, GroundLiteral, GroundCondition), with their evaluation in a
// state. Making one from the normalized task and printing it need the task's data only (formalism::TaskData); the
// overloads taking a Task read its data.
//
//   GroundCondition g;
//   g.literals.push_back({{on, {a, b}}, true});            // (on a b)
//   g.literals.push_back({{clear, {c}}, false});           // (not (clear c))
//   g.add_constraint(*task, "(>= (fuel a) 1)");
//   const bool now = holds(*task, state, g);
//   const ConjunctiveCondition c = g.lift();              // (?x0 ?x1 ?x2) (and (on ?x0 ?x1) (not (clear ?x2)) ...)
//   const WorkspaceLease ws = task->workspace();
//   for_each_binding(*task, *ws, c, state, {},  // successor/bindings.hpp
//                    [&](std::span<const ObjectId> b) { const GroundCondition back = c.ground(b); ... });
//
// Truth in a state (holds): a static literal by the task's static facts, a fluent literal by the state's atoms, a
// derived literal by the state's closure under the axioms, a numeric constraint by mimir's semantics (task/numeric.hpp:
// an undefined function value or a division by zero makes the comparison false). A ground atom need not be one of the
// task's atoms: one outside the reachable domains of its predicate never holds, and so does a fluent or derived atom
// that no state has produced yet.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/state.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr
{
class Task;
struct GroundCondition;

/// A lifted conjunctive condition (a value type). Terms use formalism::Term: >= 0 is a variable index, < 0 the object
/// -(t + 1) (formalism::object_term).
struct ConjunctiveCondition
{
    struct Variable
    {
        std::string name;           // without '?'; may be empty
        std::vector<TypeId> types;  // the variable ranges over the objects of any of these types; empty: every object
    };
    struct Literal
    {
        PredicateId predicate;
        bool positive = true;
        std::vector<formalism::Term> terms;  // one per argument of the predicate
    };
    /// lhs == rhs (positive) or lhs != rhs, on object identity.
    struct Equality
    {
        formalism::Term lhs = 0, rhs = 0;
        bool positive = true;
    };

    std::vector<Variable> variables;
    std::vector<Literal> literals;
    std::vector<Equality> equalities;
    /// Numeric constraints: comparisons of two expressions of `exprs` (flat trees as in formalism::TaskData; a
    /// Function node takes its argument terms from `expr_terms`). Static and fluent functions only.
    std::vector<formalism::NumericConstraint> constraints;
    std::vector<formalism::Expr> exprs;
    std::vector<formalism::Term> expr_terms;

    [[nodiscard]] u32 arity() const noexcept { return static_cast<u32>(variables.size()); }

    /// The precondition of a schema: its parameters (with their names and declared types), the literals (each once)
    /// and the numeric constraints of its precondition. Its bindings are the bindings of the schema's precondition
    /// alone (the schema's own bindings also apply the numeric effect rules).
    [[nodiscard]] static ConjunctiveCondition precondition(const formalism::TaskData& task, SchemaId schema);
    [[nodiscard]] static ConjunctiveCondition precondition(const Task& task, SchemaId schema);
    /// The condition of a schema's conditional effect (effect: index into the task's conditional effects, one of the
    /// schema's): the schema's parameters followed by the effect's forall parameters, and the effect's condition.
    [[nodiscard]] static ConjunctiveCondition effect_condition(const formalism::TaskData& task, SchemaId schema, u32 effect);
    [[nodiscard]] static ConjunctiveCondition effect_condition(const Task& task, SchemaId schema, u32 effect);
    /// The body of an axiom over its parameters.
    [[nodiscard]] static ConjunctiveCondition axiom_body(const formalism::TaskData& task, u32 axiom);
    [[nodiscard]] static ConjunctiveCondition axiom_body(const Task& task, u32 axiom);
    /// The task's goal: no variables; its single (empty) binding exists iff the goal holds (Task::is_goal).
    [[nodiscard]] static ConjunctiveCondition goal(const formalism::TaskData& task);
    [[nodiscard]] static ConjunctiveCondition goal(const Task& task);

    /// Appends the numeric constraint written in PDDL, e.g. "(>= (fuel ?x) (* 2 (distance ?x a)))": a comparator
    /// (=, !=, <, <=, >, >=) of two expressions built from numbers, function terms "(f t1 ... tk)" and +, -, *, /
    /// (binary; "(- e)" negates). A term is an object name or a variable "?name" of `variables`; names are matched
    /// case-insensitively. Throws std::invalid_argument("mymyr: constraint ...: ...") naming the first error.
    void add_constraint(const Task& task, std::string_view pddl);

    /// The ground condition of a binding (one object per variable): the literals and numeric constraints with every
    /// variable replaced by its object. Equalities are not part of a ground condition: the binding must satisfy them,
    /// as every binding of successor/bindings.hpp does. Throws std::invalid_argument for a binding of the wrong size
    /// or one that violates an equality.
    [[nodiscard]] GroundCondition ground(std::span<const ObjectId> binding) const;

    /// Checks the condition against the task: predicate, object, type and function ids in range, literal and
    /// function arities, variable indices below arity(), expression trees well formed (children in range, no cycles),
    /// no total-cost function. Throws std::invalid_argument("mymyr: condition: ...") naming the first violation.
    void validate(const Task& task) const;

    /// PDDL-like text, e.g. "(?x ?y - block) (and (on ?x ?y) (not (clear ?y)) (!= ?x a) (>= (fuel ?x) 1))". Numbers
    /// are written in their shortest exact form, so add_constraint() reads a constraint back unchanged.
    [[nodiscard]] std::string str(const formalism::TaskData& task) const;
    [[nodiscard]] std::string str(const Task& task) const;

    friend bool operator==(const ConjunctiveCondition& a, const ConjunctiveCondition& b);
};

/// A ground atom: a predicate applied to objects (one per argument).
struct GroundAtom
{
    PredicateId predicate;
    std::vector<ObjectId> objects;

    friend bool operator==(const GroundAtom&, const GroundAtom&) = default;
};

/// A ground atom and its polarity.
struct GroundLiteral
{
    GroundAtom atom;
    bool positive = true;

    friend bool operator==(const GroundLiteral&, const GroundLiteral&) = default;
};

/// A ground conjunctive condition (a value type): ground literals over static, fluent and derived predicates (the kind
/// comes from the predicate) and numeric constraints over ground expressions.
struct GroundCondition
{
    std::vector<GroundLiteral> literals;
    /// Comparisons of two expressions of `exprs` (flat trees as in formalism::TaskData); a Function node takes its
    /// arguments from `expr_terms`, which holds objects only (formalism::object_term). Static and fluent functions.
    std::vector<formalism::NumericConstraint> constraints;
    std::vector<formalism::Expr> exprs;
    std::vector<formalism::Term> expr_terms;

    /// The task's goal (each literal once, then its numeric constraints).
    [[nodiscard]] static GroundCondition goal(const formalism::TaskData& task);
    [[nodiscard]] static GroundCondition goal(const Task& task);

    /// Appends a ground numeric constraint written in PDDL (ConjunctiveCondition::add_constraint without variables).
    void add_constraint(const Task& task, std::string_view pddl);

    /// The lifted condition with every object replaced by a variable: variable i (named "x<i>", untyped) stands for
    /// the i-th distinct object in order of first appearance over the literals, then the constraints (as mimir's
    /// GroundConjunctiveCondition.lift). With add_inequalities, every pair of variables gets a disequality, so that
    /// the bindings map distinct variables to distinct objects. Grounding the result under the binding that maps each
    /// variable back to its object gives this condition back.
    [[nodiscard]] ConjunctiveCondition lift(bool add_inequalities = false) const;

    /// Checks the condition against the task, as ConjunctiveCondition::validate (every term an object). Throws
    /// std::invalid_argument("mymyr: condition: ...").
    void validate(const Task& task) const;

    /// PDDL text, e.g. "(and (on a b) (not (clear c)) (>= (fuel a) 1))".
    [[nodiscard]] std::string str(const formalism::TaskData& task) const;
    [[nodiscard]] std::string str(const Task& task) const;

    friend bool operator==(const GroundCondition& a, const GroundCondition& b);
};

/// Truth of a ground atom, literal or condition in s (see the top of this file). Derived atoms are evaluated in a
/// workspace leased for the call (Task::workspace()), so these may be called from inside a successor or binding
/// enumeration. Throws std::invalid_argument for ids out of range or a wrong number of objects.
[[nodiscard]] bool holds(const Task& task, StateView s, const GroundAtom& atom);
[[nodiscard]] bool holds(const Task& task, StateView s, const GroundLiteral& literal);
[[nodiscard]] bool holds(const Task& task, StateView s, const GroundCondition& condition);
}  // namespace mymyr
