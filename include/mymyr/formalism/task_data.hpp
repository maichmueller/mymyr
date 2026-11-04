#pragma once
// TaskData: the complete normalized task as flat tables.
//
// This is the contract between the front end (loki bridge, text readers) and everything downstream (compiled plans,
// Python bindings, device export). It is plain host data: pools of POD records addressed by u32 offsets, no
// pointers, no interning tables. Names are kept for the Python API and for portability (atom keys by name).
//
// Conventions
//   - Terms: `Term` >= 0 is a parameter index of the enclosing schema/axiom/effect; < 0 encodes object o as -(o+1).
//     Ground structures (initial atoms, goal) use the same encoding with only objects.
//   - A schema's parameter list is [original parameters..., parameters introduced by normalization...]; the
//     parameters of a conditional effect's forall come after the schema's (indices arity .. arity+extra-1).
//   - Objects: domain constants first, then problem objects. Predicates and axioms: domain first, then the ones the
//     problem (or goal normalization) introduces.
//   - Types are also compiled into static unary predicates by normalization (as in loki); `Object::types` and
//     `Parameter::types` keep the declared typing for the API.
//   - Numeric expressions are flat trees (`Expr` nodes referencing children by index), compiled to bytecode later.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::formalism
{
using Term = i32;

[[nodiscard]] constexpr Term object_term(ObjectId o) { return -static_cast<Term>(o.v) - 1; }
[[nodiscard]] constexpr bool is_object(Term t) { return t < 0; }
[[nodiscard]] constexpr ObjectId term_object(Term t) { return ObjectId{static_cast<u32>(-(t + 1))}; }
[[nodiscard]] constexpr u32 term_parameter(Term t) { return static_cast<u32>(t); }

enum class PredKind : u8
{
    Static,
    Fluent,
    Derived,
};
enum class FuncKind : u8
{
    Static,
    Fluent,
    Auxiliary,  // total-cost: feeds the metric / g-value, not part of the state
};
enum class Comparator : u8
{
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
};
enum class AssignOp : u8
{
    Assign,
    Increase,
    Decrease,
    ScaleUp,
    ScaleDown,
};
enum class ExprOp : u8
{
    Number,    // value
    Function,  // func applied to terms
    Add,
    Sub,
    Mul,
    Div,
    Neg,       // unary minus of a
};

struct Range  // [begin, begin + count) into a pool
{
    u32 begin = 0;
    u32 count = 0;
    [[nodiscard]] constexpr u32 end() const { return begin + count; }
};

struct Str  // into TaskData::chars
{
    u32 begin = 0;
    u32 size = 0;
};

struct Type
{
    Str name;
    Range bases;  // into type_ids (direct supertypes)
};

struct Object
{
    Str name;
    Range types;  // into type_ids
    bool constant = false;
};

/// The most parameters of a predicate or function, and the most variables of a schema (its parameters with those of
/// one conditional effect) or an axiom. Binding enumeration recurses once per variable, so this bounds its depth.
inline constexpr u32 k_max_parameters = 1024;

struct Parameter
{
    Str name;     // variable name without '?'
    Range types;  // into type_ids
};

struct Predicate
{
    Str name;
    PredKind kind = PredKind::Static;
    u32 arity = 0;
    Range params;  // into params
};

struct Function
{
    Str name;
    FuncKind kind = FuncKind::Static;
    u32 arity = 0;
    Range params;  // into params
};

struct Literal
{
    PredicateId pred;
    bool positive = true;
    Range terms;  // into terms; count == arity of pred
};

struct Expr
{
    ExprOp op = ExprOp::Number;
    f64 value = 0;     // Number
    FunctionId func;   // Function
    Range terms;       // Function arguments, into terms
    u32 a = 0, b = 0;  // children (expr indices) for Add/Sub/Mul/Div (a, b) and Neg (a)
};

struct NumericConstraint
{
    Comparator cmp = Comparator::Eq;
    u32 lhs = 0, rhs = 0;  // expr indices
};

struct NumericEffect
{
    AssignOp op = AssignOp::Assign;
    FunctionId func;
    Range terms;   // target function arguments
    u32 expr = 0;  // expr index
};

/// Conjunction of literals (static, fluent and derived mixed; kind comes from the predicate) and numeric constraints.
struct Condition
{
    Range literals;     // into literals
    Range constraints;  // into constraints
};

struct ConditionalEffect
{
    Range extra_params;  // forall parameters, into params; term indices continue after the schema's parameters
    Condition condition;
    Range effects;          // fluent literals (positive = add, negative = delete), into literals
    Range numeric_effects;  // fluent numeric effects, into numeric_effects
    std::optional<NumericEffect> auxiliary;  // total-cost effect
};

/// A parameter of an action schema that nothing but its type mentions: no literal of the precondition (equality
/// included), no effect, no condition or effect of a conditional effect, no numeric expression. Every object of its
/// type gives the same transitions, so normalization removes it from the schema (collapse_unused_parameters) and
/// keeps this record: the schema's actions bind the other parameters only, and the action as written names the
/// representative here.
struct CollapsedParameter
{
    u32 param = 0;  // into params: its name and declared types
    /// Its index in the schema's parameter list before collapsing ([written..., introduced by normalization...]).
    u32 position = 0;
    /// Whether the PDDL action declares it (not a parameter that normalization introduced for an existential).
    bool written = true;
    /// The first object (in object order: constants first, then the problem's objects) that satisfies its type
    /// literals. Invalid when no object does: then the schema has no action.
    ObjectId representative;
};

struct Schema
{
    Str name;
    /// Parameters written in the PDDL among `params` (they come first); the rest came from normalization. The written
    /// parameters that normalization collapsed are not counted (written_arity counts them).
    u32 original_arity = 0;
    Range params;  // into params (arity = params.count)
    Condition precondition;
    Range effects;  // into conditional_effects (an unconditional effect has an empty condition and no extras)
    Range collapsed;  // into collapsed_params, ascending in position
    [[nodiscard]] u32 arity() const { return params.count; }
};

struct Axiom
{
    Range params;  // into params
    Literal head;  // positive literal over a derived predicate
    Condition body;
    bool from_problem = false;
};

struct GroundAtom
{
    PredicateId pred;
    Range objects;  // into object_ids
};

struct GroundFunctionValue
{
    FunctionId func;
    Range objects;  // into object_ids
    f64 value = 0;
};

struct Metric
{
    bool minimize = true;
    u32 expr = 0;  // ground expression (terms are objects)
};

struct TaskData
{
    // identification
    std::string domain_name, problem_name;
    std::vector<std::string> requirements;

    // pools
    std::string chars;
    std::vector<TypeId> type_ids;
    std::vector<ObjectId> object_ids;
    std::vector<Term> terms;
    std::vector<Parameter> params;
    std::vector<Literal> literals;
    std::vector<Expr> exprs;
    std::vector<NumericConstraint> constraints;
    std::vector<NumericEffect> numeric_effects;
    std::vector<ConditionalEffect> conditional_effects;
    std::vector<CollapsedParameter> collapsed_params;

    // entities
    std::vector<Type> types;
    std::vector<Object> objects;
    // Whether `Object::constant` marks the domain's constants. The front end records them; mymyr's text format does
    // not, so every object of a task read from it is unmarked whether or not its domain has constants.
    bool constants_recorded = false;
    std::vector<Predicate> predicates;
    std::vector<Function> functions;
    std::vector<Schema> schemas;
    std::vector<Axiom> axioms;
    /// Whether normalization collapsed the unused parameters of the schemas (collapse_unused_parameters). A task with
    /// it set and one without are different tasks even when no schema has such a parameter: their fingerprints differ.
    bool unused_parameters_collapsed = false;

    // problem
    std::vector<GroundAtom> static_init, fluent_init;
    std::vector<GroundFunctionValue> static_values, fluent_values;
    std::optional<f64> auxiliary_initial;  // initial total-cost, if declared
    Condition goal;                        // ground: every term is an object
    std::optional<Metric> metric;

    // The goal as an encoding sees it: signed ground literals (into `literals`; every term is an object), sorted by
    // (sign: positive first, predicate id, objects) and free of duplicates. It is the goal as written, with
    // conjunctions flattened, `forall` expanded over the objects of the quantified types (constants included, in
    // object order) and `imply` with a static antecedent that holds expanded (one that fails contributes nothing).
    // Negation is pushed inward (`not or`, `not exists`, `not imply` with a static antecedent that holds, `not not`).
    // Literals over static predicates are evaluated: a true one is kept, a false one is kept too and sets
    // `encoding_goal_unsatisfiable`. A disjunction (`or`, `not and`, `exists` and `not forall` over their bindings,
    // `imply` with an antecedent that static atoms do not decide) is decided by static atoms and equality where they
    // can: a disjunct that holds makes it contribute nothing, and when all disjuncts but one fail, that one is
    // expanded. A conjunct that cannot be expanded (a disjunction with two or more undecided disjuncts, a numeric
    // constraint) is left out and clears `encoding_goal_complete`; so is one whose quantifiers would take the goal
    // past the budget of quantifier bindings. The planner's goal (`goal`) is not affected.
    // `encoding_goal_recorded` is false for a TaskData whose front end did not record it (`encoding_goal` is then
    // empty); a Task derives it with set_encoding_goal_from_normalized.
    Range encoding_goal;
    bool encoding_goal_recorded = false;
    bool encoding_goal_complete = true;
    bool encoding_goal_unsatisfiable = false;

    // ------------------------------------------------------------------ accessors
    [[nodiscard]] std::string_view str(Str s) const { return std::string_view(chars).substr(s.begin, s.size); }
    template<class T>
    [[nodiscard]] static std::span<const T> slice(const std::vector<T>& pool, Range r)
    {
        return std::span<const T>(pool.data() + r.begin, r.count);
    }
    [[nodiscard]] std::span<const Term> terms_of(const Literal& l) const { return slice(terms, l.terms); }
    [[nodiscard]] std::span<const Literal> literals_of(const Condition& c) const { return slice(literals, c.literals); }
    [[nodiscard]] std::span<const NumericConstraint> constraints_of(const Condition& c) const
    {
        return slice(constraints, c.constraints);
    }
    [[nodiscard]] std::span<const ConditionalEffect> effects_of(const Schema& s) const
    {
        return slice(conditional_effects, s.effects);
    }
    [[nodiscard]] std::span<const CollapsedParameter> collapsed_of(const Schema& s) const
    {
        return slice(collapsed_params, s.collapsed);
    }
    [[nodiscard]] std::span<const Literal> encoding_goal_literals() const { return slice(literals, encoding_goal); }
    [[nodiscard]] std::span<const ObjectId> objects_of(const GroundAtom& a) const { return slice(object_ids, a.objects); }
    [[nodiscard]] const Predicate& predicate(PredicateId p) const { return predicates[p.v]; }
    [[nodiscard]] u32 num_objects() const { return static_cast<u32>(objects.size()); }
    [[nodiscard]] bool has_numerics() const { return !functions.empty(); }

    // ------------------------------------------------------------------ building helpers
    Str intern_string(std::string_view s)
    {
        Str r{static_cast<u32>(chars.size()), static_cast<u32>(s.size())};
        chars.append(s);
        return r;
    }
    template<class T>
    static Range append(std::vector<T>& pool, std::span<const T> items)
    {
        Range r{static_cast<u32>(pool.size()), static_cast<u32>(items.size())};
        pool.insert(pool.end(), items.begin(), items.end());
        return r;
    }
};

/// A signed ground literal, as the front ends hand it to set_encoding_goal.
struct GroundLiteralData
{
    PredicateId pred;
    bool positive = true;
    std::vector<ObjectId> objects;
};

/// Records the encoding goal (see TaskData::encoding_goal): sorts and deduplicates `literals`, stores them, and sets
/// `encoding_goal_unsatisfiable` from the literals over static predicates (against `static_init`).
void set_encoding_goal(TaskData& t, std::vector<GroundLiteralData> literals, bool complete);

/// The encoding goal of a task whose goal as written is not available: the literals of `goal` except those over the
/// derived predicates that goal normalization introduced (loki names them "axiom_<k>"), complete iff nothing was left
/// out and the goal has no numeric constraints. For a conjunctive goal this is the goal itself. The derived
/// predicates are recognized by their name, so a user predicate called "axiom_<k>" in the goal is left out too.
void set_encoding_goal_from_normalized(TaskData& t);

/// Whether the encoding goal of `t` is the one set_encoding_goal_from_normalized would give (true if none is
/// recorded).
[[nodiscard]] bool has_default_encoding_goal(const TaskData& t);

/// Collapses the unused parameters of every schema (CollapsedParameter): removes them from the schema's parameters
/// (the terms of the other parameters and of the forall parameters are renumbered) and their type literals from its
/// precondition, records them with their representatives in `collapsed`, and sets `unused_parameters_collapsed`.
/// `type_predicate[p]` tells whether predicate p is a type predicate: a positive literal of one over a parameter alone
/// is a type literal of that parameter and does not count as a mention. A schema whose collapsed parameter has no
/// object of its type has no action (has_representatives). Parameters that only a schema's own type literals mention
/// are collapsed whether written or introduced by normalization. A task with `unused_parameters_collapsed` set is left as it is.
void collapse_unused_parameters(TaskData& t, std::span<const u8> type_predicate);

/// Whether every collapsed parameter of `s` has a representative; a schema without has no action.
[[nodiscard]] bool has_representatives(const TaskData& t, const Schema& s);

/// The number of parameters the PDDL action of `s` declares: original_arity plus its written collapsed parameters.
[[nodiscard]] u32 written_arity(const TaskData& t, const Schema& s);

/// The binding of the action of `s` before collapsing: `binding` (arity() objects) with each collapsed parameter's
/// representative inserted at its position. Its first written_arity(t, s) objects are the action's arguments as
/// written in PDDL.
[[nodiscard]] std::vector<ObjectId> full_binding(const TaskData& t, const Schema& s, std::span<const ObjectId> binding);

/// Whether object o is of one of the types `types` (into type_ids): one of its declared types or a supertype of one is
/// among them. Every object is of type `object`, and of an empty list of types.
[[nodiscard]] bool has_type(const TaskData& t, ObjectId o, Range types);

/// Structural validation (ranges in bounds, arities consistent, derived heads, ground goal). Throws on violation.
void validate(const TaskData& t);
}  // namespace mymyr::formalism
