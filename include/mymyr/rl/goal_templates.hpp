#pragma once
// Lifted goal templates: per instance, a few lifted conjunctions of the instance's goal literals, the goals of lifted
// hindsight relabelling (a template matched in a reached state and grounded under its first binding is a goal that
// state achieves).
//
//   rl::TemplateOptions o;                                         // the defaults below
//   const rl::GoalTemplates t = rl::lifted_goal_templates(suite, o); // host, deterministic under o.seed
//   for (const ConjunctiveCondition& c : t.conditions(g)) ...        // instance g's templates, largest first
//
// Generation, per instance g of the suite (global id), from its goal literals (GroundCondition::goal(task).literals:
// every literal once, both polarities, static, fluent and derived; numeric constraints are not literals):
//   1. the literal graph: an edge between two literals whose atoms share an object; its connected components ordered
//      by their smallest literal index;
//   2. per component and size k = 1 .. min(|C|, max_size): the first max_subsets_per_size connected k-subsets of the
//      component in lexicographic order of their (ascending) literal index tuples;
//   3. for c = 1 .. min(max_size, #components): min(max_component_samples, binom(#components, c)) draws of a uniform
//      c-subset of the components, deduplicated and taken in lexicographic order; per draw the first max_products
//      elements of the product of the drawn components' subset lists (each list by size, then in the order of step 2;
//      the last component varies fastest), each the concatenation of its subsets; those with more than max_size
//      literals are dropped;
//   4. per size, the first max_subgoals_per_size of these subgoals in that order; sizes largest first;
//   5. each subgoal lifted (GroundCondition::lift): every object, constants included, becomes a variable numbered by
//      first appearance; with distinct (Distinct::All, or Distinct::Equality when the domain declares :equality) every
//      pair of variables gets a disequality, so that distinct variables bind distinct objects.
// Draw d (0 <= d < c) of draw s of count c is the Philox block of rl/rng.hpp (seed; draw = c * 2^32 + s * c + d,
// env = g, purpose k_templates), and the c-subset is a partial Fisher-Yates shuffle of [0, #components) with index
// d + below(bits, #components - d) at step d, sorted. Templates therefore depend on (task, g, options) only: the other
// instances of the suite do not change them.
//
// Cost: subsets above max_size are never enumerated. The connected subsets come from a depth-first search over index
// tuples that extends a prefix only while the remaining picks, among literals of larger index, can still join its
// connected parts into one (a bounded search that never rules out a prefix that extends) and its component among them
// holds enough literals; the last pick is any literal adjacent to every part. A goal of hundreds of literals costs
// milliseconds.
//
// Packed form (arrays(), one aligned block, the layout a device upload copies): K templates of I instances, L literals
//   instance_offsets [I + 1]  instance g's templates are [instance_offsets[g], instance_offsets[g + 1]), largest first
//   template_size [K]         literals of template k
//   num_vars [K]              variables of template k
//   distinct [K]              1 if distinct variables of template k must bind distinct objects
//   literal_offsets [K + 1]   template k's literals are [literal_offsets[k], literal_offsets[k + 1])
//   order [K, V]              the binding order of template k's variables (ConjunctiveCondition binding order on the
//                             instance's task, successor/bindings.hpp binding_order), -1 padding; V = max num_vars
//   predicate [L]             the literal's predicate as a predicate id of the instance's domain table
//   polarity [L]              1 positive, 0 negative
//   kind [L]                  k_pred_static, k_pred_fluent or k_pred_derived
//   arity [L]
//   terms [L, A]              formalism::Term (>= 0 a variable, < 0 the object -(t + 1)), k_no_term padding; A = the
//                             largest literal arity (generated templates hold variables only)
//   goal_literal [L]          the literal's index in the instance's goal literals (-1: from_conditions)
// with the scalars num_instances, num_templates, num_literals, max_vars, max_arity.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/successor/conditions.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace mymyr::rl
{
/// Which variables of a template must bind distinct objects.
enum class Distinct : u8
{
    None = 0,      // none: two variables may bind one object
    All = 1,       // every pair of variables
    Equality = 2,  // every pair when the instance's domain declares :equality, else none
};

struct TemplateOptions
{
    u32 max_subgoals_per_size = 2;
    u32 max_size = 10;
    u32 max_subsets_per_size = 100;
    u32 max_component_samples = 5;
    u32 max_products = 1000;
    Distinct distinct = Distinct::All;
    u64 seed = 0;

    /// Throws std::invalid_argument ("mymyr: lifted_goal_templates: ...") unless every count is at least 1 and
    /// max_component_samples * max_size < 2^32 (the draw counters).
    void validate() const;
};

/// The padding of GoalTemplates' terms.
inline constexpr i32 k_no_term = std::numeric_limits<i32>::min();

/// The subgoals of steps 1-4 for global instance `instance` of a task: index tuples into the task's goal literals, sizes
/// largest first, each in its literal order (the components in ascending order, each subset ascending).
[[nodiscard]] std::vector<std::vector<u32>> goal_subgoals(const Task& task, u64 instance, const TemplateOptions& options);

/// The packed templates of every instance of a suite (see the top of the file). Immutable once built.
class GoalTemplates
{
public:
    /// The generated templates (steps 1-5) of every instance. Throws std::invalid_argument for invalid options.
    [[nodiscard]] static GoalTemplates generate(TaskSuitePtr suite, const TemplateOptions& options);
    /// Templates given as conditions: conditions[g] holds instance g's, in their match order (largest first is the
    /// convention, not a requirement). A condition's literals become the template's, its variables untyped, and its
    /// equalities either none or exactly one disequality per pair of distinct variables (distinct = 1). Throws
    /// std::invalid_argument ("mymyr: GoalTemplates: ...") for a list per instance of the wrong length, an invalid
    /// condition (ConjunctiveCondition::validate), typed variables, numeric constraints, other equalities or a
    /// condition without literals.
    [[nodiscard]] static GoalTemplates from_conditions(TaskSuitePtr suite,
                                                       const std::vector<std::vector<ConjunctiveCondition>>& conditions);

    /// The arrays of the packed form, as arrays() holds them (for a restore from saved arrays).
    struct Arrays
    {
        std::vector<i32> instance_offsets, template_size, num_vars, literal_offsets, predicate, kind, arity, terms, goal_literal;
        std::vector<u8> distinct, polarity;
        u32 max_arity = 0;  // terms is [num_literals, max_arity]
    };
    /// The templates of saved arrays (order is recomputed), with the options they were generated with (null: none):
    /// checks the shapes and ranges, then rebuilds every template as from_conditions does. Throws
    /// std::invalid_argument for arrays that do not describe templates of this suite.
    [[nodiscard]] static GoalTemplates from_arrays(TaskSuitePtr suite, const Arrays& arrays,
                                                   const TemplateOptions* options = nullptr);

    [[nodiscard]] const TaskSuitePtr& suite() const noexcept { return m_suite; }
    [[nodiscard]] const ArrayBundle& arrays() const noexcept { return *m_arrays; }
    [[nodiscard]] const std::shared_ptr<const ArrayBundle>& shared_arrays() const noexcept { return m_arrays; }
    [[nodiscard]] u32 num_instances() const noexcept { return static_cast<u32>(m_instance_offsets.size() - 1); }
    [[nodiscard]] u32 num_templates() const noexcept { return m_instance_offsets.back(); }
    [[nodiscard]] u32 num_literals() const noexcept { return m_literal_offsets.back(); }
    [[nodiscard]] u32 max_vars() const noexcept { return m_max_vars; }
    [[nodiscard]] u32 max_arity() const noexcept { return m_max_arity; }
    /// Instance g's templates: [first(g), first(g + 1)).
    [[nodiscard]] u32 first(u32 instance) const { return m_instance_offsets.at(instance); }
    /// The options of generate() (from_conditions and from_arrays: none).
    [[nodiscard]] const TemplateOptions* options() const noexcept { return m_generated ? &m_options : nullptr; }

    /// Template k as a condition over its instance's task: variables "x<i>" (untyped), its literals, and with distinct
    /// one disequality per pair of variables (the condition Task bindings match: its first binding is the template's
    /// first binding in the binding order).
    [[nodiscard]] ConjunctiveCondition condition(u32 k) const;
    /// Instance g's templates as conditions, in order.
    [[nodiscard]] std::vector<ConjunctiveCondition> conditions(u32 instance) const;
    /// The instance's goal literal indices of template k (empty for templates from conditions).
    [[nodiscard]] std::vector<u32> goal_literals(u32 k) const;

private:
    GoalTemplates() = default;
    struct Packer;

    TaskSuitePtr m_suite;
    std::shared_ptr<const ArrayBundle> m_arrays;
    std::vector<u32> m_instance_offsets{0}, m_literal_offsets{0};
    u32 m_max_vars = 0, m_max_arity = 0;
    TemplateOptions m_options;
    bool m_generated = false;
};

/// GoalTemplates::generate.
[[nodiscard]] inline GoalTemplates lifted_goal_templates(TaskSuitePtr suite, const TemplateOptions& options = {})
{
    return GoalTemplates::generate(std::move(suite), options);
}
}  // namespace mymyr::rl
