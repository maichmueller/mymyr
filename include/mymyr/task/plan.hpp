#pragma once
// Compiled plans: the immutable, flat form of schemas, axioms and the goal.
//
// A Matcher enumerates the bindings of parameters [first_free, total) that satisfy a conjunction of literals, given
// values for the prebound parameters [0, first_free) (the action parameters, for conditional effects):
//   - per free parameter a static domain `dom0` (types, static unary literals, static p(x, c), projections of static
//     k-ary literals), plus per-state unary constraints (fluent unary tables, fluent rows of constants);
//   - per binding step the row constraints (base, src, neg) to earlier or prebound parameters, and the literals that are
//     checked once their last free parameter is bound (arity >= 3, repeated variables);
//   - for fail-first forward checking the rows in both directions between free parameters.
// Pointers to per-state tables are TableRefs (a view row), resolved by each thread's workspace (successor/detail/
// engine.hpp); static tables are word offsets into Compiled::static_words.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/task/atom_index.hpp"
#include "mymyr/task/numeric.hpp"

#include <vector>

namespace mymyr::plan
{
enum class LitKind : u8
{
    Static,
    Fluent,
    Derived,
};

/// One variable position of a pattern: key += rs[bind[param]].
struct PatternVar
{
    u32 param = 0;
    const u64* rs = nullptr;
};

/// A lifted atom. Its key under a binding is base + sum over vars of rs[bind[param]]: the canonical id for fluent and
/// derived atoms, the key of the static relation for static ones. A key >= the space size means "cannot hold".
struct Pattern
{
    LitKind kind = LitKind::Fluent;
    u32 pred = 0;
    u64 base = 0;
    u32 var_begin = 0, var_count = 0;  // into Compiled::pattern_vars
    formalism::Range terms;            // the literal's terms (into TaskData::terms), kept for the API
};

struct Check
{
    Pattern pat;
    bool pos = true;
};

/// A table pointer: row r of the per-thread view (bit 63 set), or a word offset into Compiled::static_words.
struct TableRef
{
    static constexpr u64 k_view = u64{1} << 63;
    u64 v = 0;
    [[nodiscard]] static TableRef view_row(u32 row) { return {k_view | row}; }
    [[nodiscard]] static TableRef static_words(u64 offset) { return {offset}; }
    [[nodiscard]] bool is_view() const { return (v & k_view) != 0; }
    [[nodiscard]] u64 value() const { return v & ~k_view; }
};

struct Unary
{
    TableRef ptr;  // object bitset (OW words)
    bool neg = false;
};
struct Row
{
    TableRef base;  // row(v) = base + v * OW
    u32 src = 0;    // earlier bound (or prebound) parameter whose value selects the row
    bool neg = false;
};
struct Edge
{
    u32 to = 0;
    TableRef base;  // row(value of this parameter) = admissible values of `to`
    bool neg = false;
};
struct Step
{
    u32 param = 0;
    u32 row_begin = 0, row_end = 0;      // into Matcher::rows
    u32 check_begin = 0, check_end = 0;  // into Matcher::step_checks
    u32 ncheck_begin = 0, ncheck_end = 0;  // into Matcher::step_nchecks
};

struct Matcher
{
    u32 total = 0, first_free = 0;
    bool never = false;   // statically unsatisfiable
    bool use_fc = false;  // fail-first forward checking instead of the fixed order
    u32 first_exist = 0;  // step from which parameters are witness-only (== steps.size() when none)
    bool binds_in_order = false;  // fixed order over parameters first_free, first_free + 1, ...: lexicographic output
    std::vector<u64> dom0;         // total * OW (zero for prebound parameters)
    std::vector<u32> unary_begin;  // total + 1
    std::vector<Unary> unary;
    std::vector<Check> pre_checks;  // literals without free parameters
    std::vector<Check> checks;      // literals checked when their last free parameter is bound
    std::vector<Step> steps;        // free parameters in binding order
    std::vector<Row> rows;
    std::vector<u32> step_checks;  // indices into checks
    // forward checking
    std::vector<u32> free_params;     // binding order (effect-relevant first)
    std::vector<u8> relevant;         // per parameter
    std::vector<u32> fc_out_begin;    // total + 1
    std::vector<Edge> fc_out;
    std::vector<Row> fc_pre;          // prebound -> free rows, applied to the initial domains
    std::vector<u32> fc_pre_to;
    std::vector<u32> fc_checks_begin;  // total + 1
    std::vector<u32> fc_checks;        // per parameter: checks mentioning it
    std::vector<u32> check_vars_begin;  // checks.size() + 1
    std::vector<u32> check_vars;        // per check: its free parameters
    // numeric constraints: checked as soon as their last free parameter is bound
    std::vector<NumCheck> npre;          // without free parameters
    std::vector<NumCheck> nchecks;       // the others
    std::vector<u32> step_nchecks;       // indices into nchecks, per step (Step::ncheck_begin/end)
    std::vector<u32> fc_nchecks_begin;   // total + 1 (forward checking: numeric checks mentioning each parameter)
    std::vector<u32> fc_nchecks;
    std::vector<u32> ncheck_vars_begin;  // nchecks.size() + 1
    std::vector<u32> ncheck_vars;        // per numeric check: its free parameters
};

struct CondEffect
{
    Matcher cond;  // prebound: the schema parameters; free: the forall parameters
    std::vector<Pattern> adds, dels;
    // numeric part: fluent effects, then the optional total-cost effect
    std::vector<NumEffect> neffs;
    bool has_aux = false;
    AuxEffect aux;
    bool numeric = false;  // neffs or aux: evaluated by the numeric path (effect families, applicability)
    bool extras = false;   // has forall parameters
    // Atom-driven enumeration: a conditional effect without numeric effects whose forall parameters are all effect-
    // relevant and all occur in one positive fluent condition literal takes its forall bindings from the true atoms of
    // that literal's predicate in the current state (Engine::true_atoms), not from the matcher's search over the object
    // product (e.g. rubiks-cube: forall (?x ?y ?z) (when (cube5 ?x ?y ?z) ...)). Each binding is then checked against
    // the forall parameters' static domains, every condition literal and every numeric condition constraint.
    static constexpr u32 k_no_driver = ~u32{0};
    u32 driver = k_no_driver;                   // index into Compiled::drivers
    std::vector<formalism::Term> driver_terms;  // the literal's terms: parameter (>= 0) or object (< 0)
    std::vector<Check> lits;                    // every condition literal (driven effects)
};

/// A numeric effect group of an unconditional effect of a schema.
struct NumGroup
{
    std::vector<NumEffect> neffs;
    bool has_aux = false;
    AuxEffect aux;
};

/// A schema's effects with numeric content, in the task's order of conditional effects (mimir's family order).
struct NumRef
{
    bool conditional = false;
    u32 index = 0;  // into Schema::ces or Schema::uncond_num
};

struct Schema
{
    u32 arity = 0;
    u32 bind_size = 0;  // arity plus the largest forall block
    Matcher pre[2];     // [0] witness pruning on, [1] off
    std::vector<Pattern> adds, dels;  // unconditional effects
    std::vector<CondEffect> ces;
    std::vector<Check> pre_lits;  // every precondition literal (applicability of a full binding)
    std::vector<NumCheck> pre_nums;  // every numeric precondition (applicability of a full binding)
    std::vector<NumGroup> uncond_num;
    std::vector<NumRef> num_order;  // non-empty iff the schema has numeric or total-cost effects
    [[nodiscard]] bool numeric() const noexcept { return !num_order.empty(); }
};

struct Axiom
{
    Pattern head;
    Matcher body;
};
struct Stratum
{
    std::vector<Axiom> axioms;
    bool recursive = false;  // a body literal is derived from this stratum: naive rounds to the fixpoint
};

struct Goal
{
    std::vector<Check> lits;  // fluent and derived literals (static ones are folded)
    bool unsatisfiable = false;
    bool uses_derived = false;
};

/// Membership of a static predicate: keys are typed-dense over the objects occurring at each position.
struct StaticRelation
{
    static constexpr u64 k_outside = u64{1} << 56;
    u32 arity = 0;
    u64 size = 0;            // key space
    std::vector<u64> rs;     // arity * n: rank * stride, or k_outside
    std::vector<u64> bits;   // dense membership (when the key space is small)
    std::vector<u64> table;  // otherwise open addressing of key + 1
    u64 mask = 0;

    [[nodiscard]] const u64* position_table(u32 i, u32 n) const { return rs.data() + static_cast<u64>(i) * n; }
    [[nodiscard]] bool contains(u64 key) const
    {
        if (key >= size)
            return false;
        if (!bits.empty())
            return (bits[key >> 6] >> (key & 63)) & 1;
        for (u64 j = hash_key(key) & mask;; j = (j + 1) & mask)
        {
            if (table[j] == key + 1)
                return true;
            if (table[j] == 0)
                return false;
        }
    }
    [[nodiscard]] static u64 hash_key(u64 k)
    {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        return k;
    }
};

struct Compiled
{
    u32 num_objects = 0;
    u32 ow = 0;  // words per object bitset
    std::vector<formalism::PredKind> kinds;  // per predicate
    std::vector<u32> arity;                  // per predicate
    CanonicalLayout layout;
    ViewLayout view;
    std::vector<u64> static_words;         // static unary bitsets and row tables
    std::vector<StaticRelation> statics;   // per predicate (empty for non-static)
    std::vector<PatternVar> pattern_vars;
    std::vector<Schema> schemas;
    std::vector<Stratum> strata;
    Goal goal;
    u32 max_bind = 1;
    std::vector<std::vector<u32>> initial;  // initial fluent atoms: [pred, args...]
    bool has_conditional_effects = false;
    std::vector<u32> drivers;  // predicates of atom-driven conditional effects (CondEffect::driver)
    Numeric num;  // numeric fluents, constraints, effects, costs and the metric
};
}  // namespace mymyr::plan
