#pragma once
// Exact delete-relaxed reachability over ground atoms without grounding, matching mimir 0.16.3's semantics.
//
//   auto rr = mymyr::reachability::RelaxedReachability::create(*task);
//   bool r = rr->table().is_reachable(cid);                          // canonical id (task/atom_index.hpp)
//   bool g = rr->goal_reachable_without(std::vector<CanonicalAtom>{cid});  // the complete landmark test
//
// The reading is mimir's: the schemas are Datalog rules over object tuples, never instantiated as actions.
//   - Rules: one per (schema, conditional effect, positive effect literal), head = the effect atom, body = the positive
//     fluent and derived literals of the schema's precondition and of the effect's condition plus their static
//     literals of both polarities; one per axiom (head = the derived atom). Negative fluent and derived literals are
//     dropped (the delete relaxation), negative static ones are kept (exact: no action changes a static atom). PDDL
//     equality is evaluated on object identity. A variable no positive literal binds ranges over its declared type.
//   - EDB: the static atoms and the initial fluent atoms. The fixpoint is monotone (negation reaches only statics).
//
// Evaluation (mymyr's own): semi-naive rounds over compiled matchers and bitset views (the successor engine's
// row tables, reachability/join.hpp). Round 0 enumerates every rule over the initial atoms; round r joins every rule
// anchored on each atom derived in round r-1: the anchor literal is unified with the atom, literals before it read the
// atoms known before round r-1 (Old), literals after it every atom known so far (New), so a binding is enumerated
// exactly once (at its first new literal). Head variables are bound first; once only witness variables remain one
// completion suffices, and a binding whose head is already derived is cut where its head variables are bound.
//
// Restricted fixpoints (the Richter-Helmert-Westphal "possible first achievers"): forbidden atoms are removed from the
// initial state and never derived. Witnesses: the unrestricted fixpoint records per derived atom its first derivation
// (rule and binding); witness_query answers "reachable without these atoms" from the recorded derivation trees
// (ReachableWithout is sound, Unknown carries no information). Conjunctive queries: per variable the objects it takes
// in some solution of a conjunction over a table (the lifted landmark generator's joint disambiguation).
//
// Nothing is interned into the task. Tables and queries are safe to use from several threads at once.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::reachability
{
struct Options
{
    /// Evaluate negative static literals (including PDDL (not (= ?x ?y))). false reproduces mimir's
    /// LiftedGrounder exploration, which drops every negative literal.
    bool enforce_negative_static = true;
    /// Record the first derivation of every derived atom (witness_query).
    bool record_witnesses = true;
};

struct Statistics
{
    u64 rules = 0;          // one per schema, conditional effect and positive effect literal, plus one per axiom
    u64 dropped_rules = 0;  // statically unsatisfiable
    u64 plans = 0;          // compiled matchers (one unanchored per rule plus one per relation literal)
    u32 rounds = 0;         // semi-naive rounds of the unrestricted fixpoint
    u64 fluent_atoms = 0;   // reachable fluent atoms
    u64 derived_atoms = 0;  // reachable derived atoms
    u64 bindings = 0;       // bindings emitted by the unrestricted fixpoint
    double compile_ms = 0;
    double fixpoint_ms = 0;
};

enum class WitnessVerdict : u8
{
    ReachableWithout,  // a recorded derivation avoids every forbidden atom: reachable in the restricted fixpoint
    Unknown,           // no information (unreachable, or the recorded derivation uses a forbidden atom)
};

/// A term of a conjunctive query: >= 0 a variable index, < 0 the object -(o + 1).
using QueryTerm = i32;
[[nodiscard]] inline constexpr QueryTerm query_object(ObjectId o) { return -static_cast<i32>(o.v) - 1; }
[[nodiscard]] inline constexpr QueryTerm query_variable(u32 v) { return static_cast<i32>(v); }

struct QueryLiteral
{
    PredicateId predicate;
    std::vector<QueryTerm> terms;
    bool positive = true;  // negative: static predicates only (negative fluent and derived literals are ignored)
};

/// A conjunction of positive literals over static, fluent and derived predicates, negative static literals, and
/// equalities / disequalities between terms. Variables no positive literal binds range over every object.
struct ConjunctiveQuery
{
    u32 num_variables = 0;
    std::vector<QueryLiteral> literals;
    std::vector<std::pair<QueryTerm, QueryTerm>> equalities, disequalities;
};

namespace detail
{
struct Program;
struct TableData;
struct WitnessMemo;
}  // namespace detail

class WitnessQuery;

/// The atoms one fixpoint reached. Shares the compiled rules with the RelaxedReachability that produced it (and
/// keeps them alive); the task must outlive it.
class Table
{
public:
    Table(std::shared_ptr<const detail::Program> program, std::unique_ptr<detail::TableData> data);
    Table(Table&&) noexcept;
    Table& operator=(Table&&) noexcept;
    ~Table();

    /// Membership of a fluent or derived atom by canonical id (ids outside the task's canonical space: false).
    [[nodiscard]] bool is_reachable(CanonicalAtom atom) const;
    [[nodiscard]] bool is_reachable(PredicateId predicate, std::span<const ObjectId> objects) const;
    /// The reachable atoms of a fluent or derived predicate, in derivation order (empty for a static predicate).
    [[nodiscard]] std::span<const CanonicalAtom> atoms(PredicateId predicate) const;
    /// The same as object tuples (flat, arity objects each).
    [[nodiscard]] std::span<const u32> tuples(PredicateId predicate) const;

    [[nodiscard]] u64 num_atoms() const;
    [[nodiscard]] u64 num_fluent_atoms() const;
    [[nodiscard]] u64 num_derived_atoms() const;
    /// The static goal holds and every positive fluent and derived goal atom is reachable.
    [[nodiscard]] bool goal_reachable() const;
    [[nodiscard]] u32 rounds() const;

    /// Per query variable, the objects (ascending) it takes in some solution of the whole conjunction over this
    /// table. Throws std::invalid_argument for a variable index >= num_variables.
    [[nodiscard]] std::vector<std::vector<ObjectId>> project(const ConjunctiveQuery& query) const;

    [[nodiscard]] bool has_witnesses() const;
    /// Throws std::logic_error on a table without witnesses.
    [[nodiscard]] WitnessQuery witness_query(std::span<const CanonicalAtom> forbidden) const;

    [[nodiscard]] const detail::TableData& data() const noexcept { return *m_data; }
    [[nodiscard]] const detail::Program* program() const noexcept { return m_program.get(); }

private:
    std::shared_ptr<const detail::Program> m_program;
    std::unique_ptr<detail::TableData> m_data;
};

/// "Is this atom reachable without the forbidden atoms?" from the recorded derivations; verdicts are memoized per
/// query. Holds a pointer to its table, which must outlive it. Not thread-safe (one query per thread).
class WitnessQuery
{
public:
    WitnessQuery(const Table& table, std::span<const CanonicalAtom> forbidden);
    WitnessQuery(WitnessQuery&&) noexcept;
    WitnessQuery& operator=(WitnessQuery&&) noexcept;
    ~WitnessQuery();

    [[nodiscard]] WitnessVerdict avoids(CanonicalAtom atom) const;
    [[nodiscard]] u64 memoized() const;

private:
    const Table* m_table;
    std::unique_ptr<detail::WitnessMemo> m_memo;
};

class RelaxedReachability
{
public:
    /// Compiles the rules of `task` and runs the unrestricted fixpoint. The task must outlive the result.
    [[nodiscard]] static std::shared_ptr<const RelaxedReachability> create(const Task& task, const Options& options = {});

    RelaxedReachability(const RelaxedReachability&) = delete;
    RelaxedReachability& operator=(const RelaxedReachability&) = delete;
    ~RelaxedReachability();

    [[nodiscard]] const Table& table() const noexcept { return *m_table; }
    [[nodiscard]] bool is_reachable(CanonicalAtom atom) const { return m_table->is_reachable(atom); }
    [[nodiscard]] bool goal_reachable() const { return m_table->goal_reachable(); }

    /// The fixpoint with `forbidden` removed from the initial state and never derived (shares the compiled rules).
    [[nodiscard]] Table restricted(std::span<const CanonicalAtom> forbidden) const;
    /// The complete delete-relaxation landmark test: is the goal reachable without `forbidden`? Stops as soon as the
    /// goal is reached.
    [[nodiscard]] bool goal_reachable_without(std::span<const CanonicalAtom> forbidden) const;

    [[nodiscard]] const Task& task() const noexcept;
    [[nodiscard]] const Options& options() const noexcept { return m_options; }
    [[nodiscard]] const Statistics& statistics() const noexcept { return m_stats; }

    struct Private
    {
    };
    RelaxedReachability(Private, const Task& task, const Options& options);

private:
    Options m_options;
    Statistics m_stats;
    std::shared_ptr<detail::Program> m_program;
    std::unique_ptr<Table> m_table;
};
}  // namespace mymyr::reachability
