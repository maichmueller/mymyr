#pragma once
// Binding generators: the groundings of a lifted conjunctive condition in a state, and the applicable ground actions of
// one schema, with any subset of the variables fixed in advance (partial bindings).
//
//   const ConjunctiveCondition c = ConjunctiveCondition::precondition(*task, SchemaId{2});
//   std::vector<std::optional<ObjectId>> partial(c.arity());
//   partial[1] = ObjectId{4};                                   // fix the second parameter
//   for_each_binding(*task, task->workspace(), c, state, partial,
//                    [&](std::span<const ObjectId> binding) { ...; return true; });  // false stops
//
// Targets:
//   - a ConjunctiveCondition: variables (optionally named and typed), literals over static, fluent and derived
//     predicates whose terms are variables or objects (either polarity), equalities and disequalities between terms,
//     and numeric constraints over the task's functions. A binding assigns an object to every variable such that every
//     literal, equality and constraint holds in the state (derived atoms are those of the state's axiom closure);
//   - a schema (SchemaId): the bindings are exactly the parameter tuples of the schema's applicable ground actions in
//     the state, every parameter enumerated (no witness pruning): the precondition, numeric constraints included, and
//     for a schema with numeric or total-cost effects mimir's applicability rules on those effects
//     (task/numeric.hpp). Conditional effects do not restrict applicability otherwise.
// Partial bindings: a span with one entry per variable (or empty: every variable free); a fixed entry restricts the
// variable to that object. A fixed object that violates the variable's type, a static literal or any other part of
// the condition gives no binding (not an error).
// Order: deterministic. The free variables are bound in an order the compiler chooses per (target, set of fixed
// variables), and the bindings come in lexicographic order over the free variables in that order, by object index.
// The order does not depend on the state, the thread, the workspace or earlier calls, so a later call can continue an
// enumeration after a binding it returned (BindingOptions::resume_after).
// Workspaces: the enumeration uses scratch of `ws` (a workspace of the task, one per thread: Task::workspace()). The
// callback may use the workspace's successor generator, but must not start another binding enumeration on the same
// workspace (std::logic_error). Compiled targets are cached per workspace, keyed by (target, set of fixed variables).
//
// Complexity: compiling a target costs about one pass over the static atoms of its predicates; each call then
// prepares the state (as Successors::prepare: its true atoms and the axioms) and searches the bindings with the
// relation tables of the state.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/state.hpp"

#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mymyr
{
class Task;
class Workspace;

/// A lifted conjunctive condition over the predicates, objects, types and functions of one task (a value type).
/// Terms use formalism::Term: >= 0 is a variable index, < 0 the object -(t + 1) (formalism::object_term).
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
    [[nodiscard]] static ConjunctiveCondition precondition(const Task& task, SchemaId schema);
    /// The task's goal: no variables; its single (empty) binding exists iff the goal holds (Task::is_goal).
    [[nodiscard]] static ConjunctiveCondition goal(const Task& task);

    /// Checks the condition against the task: predicate, object, type and function ids in range, literal and
    /// function arities, variable indices below arity(), expression trees well formed (children in range, no cycles),
    /// no total-cost function. Throws std::invalid_argument("mymyr: condition: ...") naming the first violation.
    void validate(const Task& task) const;

    /// PDDL-like text, e.g. "(?x ?y - block) (and (on ?x ?y) (not (clear ?y)) (!= ?x a) (>= (fuel ?x) 1))".
    [[nodiscard]] std::string str(const Task& task) const;

    friend bool operator==(const ConjunctiveCondition& a, const ConjunctiveCondition& b);
};

/// A ground literal of a binding (GroundConjunction); `objects` is valid during the callback that received it.
struct GroundLiteral
{
    PredicateId predicate;
    bool positive = true;
    std::span<const ObjectId> objects;
};

/// A binding with the condition's literals grounded under it, split by predicate kind in the condition's literal order
/// (as the fork's create_ground_conjunction_generator). Equalities and numeric constraints are not literals and do not
/// appear. Valid during the callback that received it.
struct GroundConjunction
{
    std::span<const ObjectId> binding;
    std::span<const GroundLiteral> static_literals, fluent_literals, derived_literals;
};

/// Variables fixed in advance: one entry per variable (std::nullopt: free), or empty for none.
using PartialBinding = std::span<const std::optional<ObjectId>>;

struct BindingOptions
{
    /// At most this many bindings.
    u64 limit = ~u64{0};
    /// Continue after this binding: one that an earlier call with the same target, state and partial binding
    /// produced. The enumeration resumes strictly after it in the documented order (cheaply: the search descends to
    /// it without enumerating what came before). Empty: start at the first binding.
    std::span<const ObjectId> resume_after = {};
};

namespace detail
{
/// A non-owning callable reference (std::function_ref is not yet in every standard library mymyr builds with).
template<class Sig>
class FunctionRef;
template<class R, class... A>
class FunctionRef<R(A...)>
{
public:
    template<class F>
    FunctionRef(F& f) noexcept  // NOLINT(google-explicit-constructor)
        : m_obj(static_cast<void*>(&f)), m_call([](void* o, A... a) -> R { return (*static_cast<F*>(o))(static_cast<A>(a)...); })
    {
    }
    R operator()(A... a) const { return m_call(m_obj, static_cast<A>(a)...); }

private:
    void* m_obj;
    R (*m_call)(void*, A...);
};

/// What to enumerate: a schema's applicable actions (condition == nullptr) or a condition's bindings.
struct BindingTarget
{
    const ConjunctiveCondition* condition = nullptr;
    SchemaId schema;
};

using BindingSink = FunctionRef<bool(std::span<const ObjectId>)>;
using ConjunctionSink = FunctionRef<bool(const GroundConjunction&)>;

u64 enumerate_bindings(const Task& task, Workspace& ws, const BindingTarget& target, StateView s, PartialBinding partial,
                       const BindingOptions& options, BindingSink* bindings, ConjunctionSink* conjunctions);

/// Adapts a callback returning bool or void to one returning bool.
template<class Arg, class F>
auto bool_callback(F& f)
{
    return [&f](Arg a) -> bool
    {
        if constexpr (std::is_same_v<std::invoke_result_t<F&, Arg>, bool>)
            return f(a);
        else
        {
            f(a);
            return true;
        }
    };
}
}  // namespace detail

/// Calls fn(std::span<const ObjectId> binding) for every binding of `condition` in s consistent with `partial`, in the
/// documented order; if fn returns bool, false stops. The span is valid during the call. Returns the number of
/// bindings passed to fn. Throws std::invalid_argument for an invalid condition (ConjunctiveCondition::validate), a
/// partial binding of the wrong size, an object index out of range, or a resume_after binding that does not fit.
template<class F>
u64 for_each_binding(const Task& task, Workspace& ws, const ConjunctiveCondition& condition, StateView s,
                     PartialBinding partial, F&& fn, const BindingOptions& options = {})
{
    auto f = detail::bool_callback<std::span<const ObjectId>>(fn);
    detail::BindingSink sink(f);
    return detail::enumerate_bindings(task, ws, {&condition, SchemaId{}}, s, partial, options, &sink, nullptr);
}

/// for_each_binding over a schema: fn receives the parameter tuple of every applicable ground action of the schema in
/// s that agrees with `partial` (every parameter, including those normalization introduced). Throws
/// std::invalid_argument for a schema id out of range and as above.
template<class F>
u64 for_each_binding(const Task& task, Workspace& ws, SchemaId schema, StateView s, PartialBinding partial, F&& fn,
                     const BindingOptions& options = {})
{
    auto f = detail::bool_callback<std::span<const ObjectId>>(fn);
    detail::BindingSink sink(f);
    return detail::enumerate_bindings(task, ws, {nullptr, schema}, s, partial, options, &sink, nullptr);
}

/// for_each_binding with the ground literals: fn(const GroundConjunction&) per binding.
template<class F>
u64 for_each_ground_conjunction(const Task& task, Workspace& ws, const ConjunctiveCondition& condition, StateView s,
                                PartialBinding partial, F&& fn, const BindingOptions& options = {})
{
    auto f = detail::bool_callback<const GroundConjunction&>(fn);
    detail::ConjunctionSink sink(f);
    return detail::enumerate_bindings(task, ws, {&condition, SchemaId{}}, s, partial, options, nullptr, &sink);
}

/// for_each_ground_conjunction over a schema (the literals of its precondition).
template<class F>
u64 for_each_ground_conjunction(const Task& task, Workspace& ws, SchemaId schema, StateView s, PartialBinding partial,
                                F&& fn, const BindingOptions& options = {})
{
    auto f = detail::bool_callback<const GroundConjunction&>(fn);
    detail::ConjunctionSink sink(f);
    return detail::enumerate_bindings(task, ws, {nullptr, schema}, s, partial, options, nullptr, &sink);
}

/// The bindings as owning tuples (at most `limit`).
[[nodiscard]] std::vector<std::vector<ObjectId>> bindings(const Task& task, Workspace& ws, const ConjunctiveCondition& condition,
                                                          StateView s, PartialBinding partial = {}, u64 limit = ~u64{0});
[[nodiscard]] std::vector<std::vector<ObjectId>> bindings(const Task& task, Workspace& ws, SchemaId schema, StateView s,
                                                          PartialBinding partial = {}, u64 limit = ~u64{0});

/// The number of bindings (at most `limit`).
[[nodiscard]] u64 count_bindings(const Task& task, Workspace& ws, const ConjunctiveCondition& condition, StateView s,
                                 PartialBinding partial = {}, u64 limit = ~u64{0});
[[nodiscard]] u64 count_bindings(const Task& task, Workspace& ws, SchemaId schema, StateView s, PartialBinding partial = {},
                                 u64 limit = ~u64{0});
}  // namespace mymyr
