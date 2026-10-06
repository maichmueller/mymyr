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

struct Schema
{
    Str name;
    u32 original_arity = 0;  // parameters written in the PDDL; the rest came from normalization
    Range params;            // into params (arity = params.count)
    Condition precondition;
    Range effects;  // into conditional_effects (an unconditional effect has an empty condition and no extras)
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

    // entities
    std::vector<Type> types;
    std::vector<Object> objects;
    std::vector<Predicate> predicates;
    std::vector<Function> functions;
    std::vector<Schema> schemas;
    std::vector<Axiom> axioms;

    // problem
    std::vector<GroundAtom> static_init, fluent_init;
    std::vector<GroundFunctionValue> static_values, fluent_values;
    std::optional<f64> auxiliary_initial;  // initial total-cost, if declared
    Condition goal;                        // ground: every term is an object
    std::optional<Metric> metric;

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

/// Structural validation (ranges in bounds, arities consistent, derived heads, ground goal). Throws on violation.
void validate(const TaskData& t);
}  // namespace mymyr::formalism
