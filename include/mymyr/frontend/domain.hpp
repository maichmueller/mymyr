#pragma once
// PDDL front end: loki parses and normalizes, then a translation layer matching mimir's
// (`ToMimirStructures` + `EncodeParameterIndexInVariables`) produces a complete `formalism::TaskData`.
//
//   auto domain = mymyr::frontend::Domain::from_file("domain.pddl");   // parse + normalize once
//   std::shared_ptr<const formalism::TaskData> t = domain->instantiate_file("p01.pddl");  // per problem; thread-safe
//
// Parity with mimir. Every id and every list order in the TaskData equals what mimir produces for the same PDDL,
// token for token, wherever mimir's own output is deterministic: objects (constants first, then problem objects in
// declaration order), predicates (static, fluent, derived by kind, then problem-derived), functions (static,
// fluent, auxiliary), schemas, axioms, and the order of literals, conditional effects, initial atoms and goal
// literals. mimir orders everything by the creation index of its hash-consing repositories; the translator replays
// both of mimir's passes over loki's output to reproduce those indices (see translate.hpp). Checked by
// tests/cpp/frontend (suite, numeric and IPC goldens) and tests/data/check_fork_parity.py (every IPC-2023 task of
// mimir's data). Orders mimir does not define itself (they depend on heap addresses there, and change between runs
// of mimir) are reproduced as far as they are deterministic there, and are otherwise made deterministic here:
//   - loki visits a type hierarchy through an std::unordered_set of type pointers (`collect_types_from_hierarchy`),
//     so in typed domains the order of the type predicates, of the type literals of one object in the initial state
//     and of the type literals of one parameter in a condition depends on heap addresses (and differs between runs
//     of mimir). mymyr puts each such group in hierarchy order instead, a function of the type names and the
//     hierarchy alone (translate.hpp, hierarchy_ranks). The types themselves are numbered in that name order
//     (`object`, `number`, then alphabetical), because loki lists them in the iteration order of a hash table of
//     type names, which differs between standard libraries;
//   - loki's ToEffectNormalForm sums the numeric effects per (operator, function) in an unordered_map keyed by
//     pointers; mymyr orders each run of such effects by (function name, arguments, operator);
//   - the arguments of the derived predicates that loki generates for universal quantifiers ("axiom_<k>") are the
//     free variables in the iteration order of an unordered_set of variable pointers; mymyr sorts them by variable
//     name (DomainOptions::generated_argument_order reproduces a particular mimir run), and orders the type literals
//     of those axioms' parameters by parameter;
//   - conditional effects of one schema are grouped in an std::unordered_map keyed by a condition *pointer*; for at
//     most two groups libc++ iterates in reverse insertion order (reproduced), for three or more the order depends
//     on heap addresses (we keep reverse insertion order);
//   - parameters that normalization appends to a schema (moved existentials) are sorted by pointer value; we sort
//     by creation order, which is what the allocator produces in practice;
//   - loki de-duplicates the branches of a disjunction (disjunctive normal form) and the literal and `when` effects
//     of an effect conjunction (ToEffectNormalForm) through an std::unordered_set. The order of the schemas that one
//     `or` splits into and of the literal effects of one schema follows the hash function and the bucket policy of
//     the standard library: it is the same on every run of one build, but not between libstdc++ and libc++, and
//     mymyr does not canonicalize it.
// Conditions list each nullary literal twice, as mimir's ConjunctiveCondition does (once as a lifted literal,
// once as a "nullary ground literal"); consumers may dedupe.
//
// Thread safety. A Domain is immutable after construction and may be shared; `instantiate_*` may be called
// concurrently from any number of threads. loki keeps no global state, but `loki::Parser::parse_problem` mutates
// the parser (error handler, problem counter), so problem *parsing* is serialized by a per-Domain mutex. The fast
// path (`InstantiateOptions::fast_init`, the default) hands loki only the problem without its `:init` section, so the
// lock is held for well under a millisecond on typical problems; `:init` itself is read without the lock. With
// `fast_init = false` the whole problem goes through loki and the lock covers loki's parse of the full problem
// (loki's normalization and the translation run unlocked).
//
// Text preprocessing. `from_file` / `instantiate_file` read files like loki does (comments stripped, lower-cased);
// `from_string` / `instantiate_string` hand the text to loki as it is (loki's grammar has no comments and is
// case-sensitive), which is also what mimir does with strings.

#include "mymyr/formalism/task_data.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::frontend
{
using TaskPtr = std::shared_ptr<const formalism::TaskData>;

struct DomainOptions
{
    /// Argument order of the derived predicates that loki generates for universal quantifiers ("axiom_<k>"), as
    /// variable names (with or without '?'), by predicate name. mymyr sorts their arguments by variable name; mimir
    /// takes an order that depends on heap addresses. Naming the order of one mimir run here reproduces that run
    /// (used by the parity tests). Predicates not listed keep the default.
    std::map<std::string, std::vector<std::string>> generated_argument_order;
};

struct InstantiateOptions
{
    /// One-pass `:objects`/`:init` path: loki normalizes the problem without its `:init`; the initial atoms and
    /// function values are read straight into the TaskData, merged with loki's type and equality atoms in loki's
    /// order. Produces exactly the TaskData of the full loki path (guarded by golden tests).
    bool fast_init = true;
};

class Domain
{
public:
    static std::shared_ptr<const Domain> from_file(const std::filesystem::path& path, const DomainOptions& options = {});
    /// `path` is used for error messages only.
    static std::shared_ptr<const Domain> from_string(std::string_view text, const std::filesystem::path& path = "",
                                                     const DomainOptions& options = {});

    /// Instantiates a problem of this domain. Thread-safe. Throws std::runtime_error (or a loki exception) on
    /// malformed input.
    [[nodiscard]] TaskPtr instantiate_file(const std::filesystem::path& problem_path, const InstantiateOptions& options = {}) const;
    [[nodiscard]] TaskPtr instantiate_string(std::string_view text, const std::filesystem::path& path = "",
                                             const InstantiateOptions& options = {}) const;

    /// The normalized domain alone (problem sections empty): types, constants, predicates, functions, schemas,
    /// domain axioms.
    [[nodiscard]] const formalism::TaskData& domain_data() const;
    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] const std::filesystem::path& path() const;

    ~Domain();
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    struct Impl;

private:
    explicit Domain(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

/// Convenience: parse the domain and instantiate one problem.
[[nodiscard]] TaskPtr load_task(const std::filesystem::path& domain_path, const std::filesystem::path& problem_path,
                                const InstantiateOptions& options = {});
}  // namespace mymyr::frontend
