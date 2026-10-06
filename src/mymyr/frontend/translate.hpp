#pragma once
// Port of mimir's translation layer (`ToMimirStructures` + `EncodeParameterIndexInVariables`), from loki's
// normalized structures to `formalism::TaskData`.
//
// Mimir runs two full passes, each into fresh hash-consing repositories, and then orders every list (literals of
// a condition, conditional effects, initial atoms, goal literals, numeric constraints, multi-operator operands) by
// the creation index in the *second* pass's repositories. The second pass visits the first pass's lists in
// first-pass index order. We replay both passes on `Repo` tables that only assign those indices (repo.hpp), in
// mimir's traversal order (Apple Clang evaluates call arguments left to right, which fixes the order where mimir
// creates several entities inside one call), and emit the TaskData from the second pass.
//
// Repository topology, as in mimir (`formalism::Parser`):
//   domain:  pass 1 -> R1d (fresh), pass 2 -> R2d (fresh; the final domain repositories)
//   problem: pass 1 -> R1p (child of R2d), pass 2 -> R2p (child of R2d)

#include "mymyr/formalism/task_data.hpp"
#include "repo.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
// loki's entity headers only: loki/loki.hpp would also pull in the spirit x3 AST (parser.hpp, in domain.cpp)
#include <loki/details/pddl/action.hpp>
#include <loki/details/pddl/atom.hpp>
#include <loki/details/pddl/axiom.hpp>
#include <loki/details/pddl/conditions.hpp>
#include <loki/details/pddl/domain.hpp>
#include <loki/details/pddl/effects.hpp>
#include <loki/details/pddl/function.hpp>
#include <loki/details/pddl/function_expressions.hpp>
#include <loki/details/pddl/function_skeleton.hpp>
#include <loki/details/pddl/function_value.hpp>
#include <loki/details/pddl/literal.hpp>
#include <loki/details/pddl/metric.hpp>
#include <loki/details/pddl/object.hpp>
#include <loki/details/pddl/parameter.hpp>
#include <loki/details/pddl/predicate.hpp>
#include <loki/details/pddl/problem.hpp>
#include <loki/details/pddl/requirements.hpp>
#include <loki/details/pddl/term.hpp>
#include <loki/details/pddl/type.hpp>
#include <loki/details/pddl/variable.hpp>

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr::frontend::detail
{
/// Ground problem input in loki's final order (`loki::ProblemImpl::get_initial_literals()` /
/// `get_initial_function_values()` of the translated problem). Filled from loki (full path) or by the fast `:init`
/// reader.
struct GroundInit
{
    struct Atom
    {
        u32 pred;
        bool positive;
        u32 begin, count;  // into objects
    };
    struct Value
    {
        u32 func;
        u32 begin, count;  // into objects
        f64 value;
    };
    std::vector<Atom> atoms;
    std::vector<Value> values;
    std::vector<u32> objects;

    void add_atom(u32 pred, bool positive, std::span<const u32> objs)
    {
        atoms.push_back({pred, positive, static_cast<u32>(objects.size()), static_cast<u32>(objs.size())});
        objects.insert(objects.end(), objs.begin(), objs.end());
    }
    void add_value(u32 func, std::span<const u32> objs, f64 value)
    {
        values.push_back({func, static_cast<u32>(objects.size()), static_cast<u32>(objs.size()), value});
        objects.insert(objects.end(), objs.begin(), objs.end());
    }
};

/// Everything the problem translation needs from the domain. Immutable after `translate_domain`.
struct DomainState
{
    // entity maps (values are TaskData ids)
    absl::flat_hash_map<const loki::PredicateImpl*, u32> pred_of;
    absl::flat_hash_map<std::string, u32> pred_by_name;
    absl::flat_hash_map<const loki::FunctionSkeletonImpl*, u32> func_of;
    absl::flat_hash_map<std::string, u32> func_by_name;
    absl::flat_hash_map<const loki::ObjectImpl*, u32> const_of;
    absl::flat_hash_map<std::string, u32> const_by_name;
    absl::flat_hash_map<const loki::TypeImpl*, u32> type_of;
    absl::flat_hash_map<std::string, u32> type_by_name;

    // mimir's static analysis (`ToMimirStructures::prepare`) of the domain
    absl::flat_hash_set<std::string> fluent_predicates, derived_predicates, effect_functions;

    // canonical type order (see domain.hpp): for each predicate, the type whose AddTypePredicates predicate it is
    // (~0u for other predicates)
    std::vector<u32> pred_type;
    // canonical argument order of the derived predicates that loki's RemoveUniversalQuantifiers generates
    // ("axiom_<k>"): their parameters are the free variables of a forall body in the iteration order of an
    // std::unordered_set of variable pointers. perm[j] = loki's position of canonical argument j (arguments sorted by
    // variable name).
    absl::flat_hash_map<u32, std::vector<u32>> pred_perm;

    std::unique_ptr<Repo> r2d;  // final domain repositories (parent of every problem's)
    formalism::TaskData data;   // domain part of every task
};

/// Which predicates loki generated, and the argument order to give them.
struct PredicateOrigin
{
    absl::flat_hash_set<std::string> declared;  // predicate names of the parsed domain (and problem); the others are loki's
    const std::map<std::string, std::vector<std::string>>* argument_order = nullptr;  // DomainOptions override
};

std::unique_ptr<DomainState> translate_domain(const loki::Domain& translated_domain, const PredicateOrigin& origin);

/// Canonical argument order of a derived predicate that loki generated (DomainState::pred_perm): empty if the
/// predicate is not generated by RemoveUniversalQuantifiers or has fewer than two parameters. The arguments are
/// sorted by variable name unless `origin.argument_order` names an order.
std::vector<u32> generated_predicate_perm(const loki::Predicate& p, const PredicateOrigin& origin);

/// The canonical order of the types in a type hierarchy (see domain.hpp): the reverse of the order in which loki's
/// `collect_types_from_hierarchy` inserts them (each declared type, then its bases, depth first), which is the order
/// loki's std::unordered_set yields whenever no two of them share a hash bucket. Returns rank[type] (~0u if absent).
std::vector<u32> hierarchy_ranks(const formalism::TaskData& t, std::span<const TypeId> declared);

using ObjectMap = absl::flat_hash_map<const loki::ObjectImpl*, u32>;

/// Maps the problem's objects: constants first, then `problem->get_objects()` in order.
ObjectMap map_objects(const DomainState& ds, const loki::Problem& problem);

/// The translated problem's initial literals and function values (the full loki path).
GroundInit read_loki_init(const DomainState& ds, const loki::Problem& problem, const ObjectMap& object_of);

/// Builds the task from the translated problem, with `init` in place of the problem's initial literals and values.
/// `origin.declared`: the predicate names of the parsed domain and problem (the others are loki's).
formalism::TaskData translate_problem(const DomainState& ds, const loki::Problem& problem, const GroundInit& init,
                                      const ObjectMap& object_of, const PredicateOrigin& origin);

}  // namespace mymyr::frontend::detail
