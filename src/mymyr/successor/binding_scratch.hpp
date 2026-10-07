#pragma once
// Internal: the per-workspace scratch of the binding generators (successor/bindings.hpp).
//
// A target (a condition, or a schema's precondition) compiles to a conjunctive matcher of reachability/join.hpp with
// the fixed variables prebound: join plans take any subset of prebound variables, equalities, and custom checks
// (here the numeric constraints, evaluated on the condition's expression trees). The plans read the relation rows of a
// private Workspace's engine view after Successors::prepare (fluent atoms plus the axiom closure), so the caller's
// successor generator stays free for the callback. Compiled targets are cached per (target, fixed variables).

#include "../reachability/join.hpp"

#include "mymyr/successor/bindings.hpp"
#include "mymyr/task/workspace.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mymyr::detail
{
/// A compiled target for one set of fixed variables.
struct BindingQuery
{
    ConjunctiveCondition cond;  // the condition (for a schema: its precondition); the numeric checks read its pools
    bool schema_target = false;
    u32 schema = 0;
    bool numeric_effects = false;  // a schema with numeric or total-cost effects: Successors::effects decides
    std::vector<u8> fixed;
    std::vector<reach::Lit> lits;
    reach::Plan plan;
    /// The plan with, per step d, a check that the binding of steps [0, d] is not below resume_after's (lexicographic in
    /// step order; strictly above once every step is bound); made on the first resumed call.
    reach::Plan resume;
    bool has_resume = false;
    std::vector<u32> order;  // the free variables in step order
    std::vector<formalism::PredKind> kinds;  // per literal
    u32 terms = 0;                           // total literal terms (ground conjunction scratch)
};

struct KeyHash
{
    usize operator()(const std::vector<u64>& k) const noexcept
    {
        u64 h = 0x9e3779b97f4a7c15ULL ^ k.size();
        for (u64 x : k)
        {
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h *= 0xff51afd7ed558ccdULL;
        }
        return static_cast<usize>(h ^ (h >> 33));
    }
};

class BindingScratch
{
public:
    /// Cached compiled targets per workspace; beyond this many the cache (and the static tables) start over.
    static constexpr usize k_max_cached = 256;

    explicit BindingScratch(const Task& task);
    BindingScratch(const BindingScratch&) = delete;
    BindingScratch& operator=(const BindingScratch&) = delete;
    ~BindingScratch();

    const Task* task;
    std::unique_ptr<Workspace> inner;  // the engine whose view the plans read
    std::unique_ptr<reach::StaticTables> tables;
    std::unordered_map<std::vector<u64>, std::unique_ptr<BindingQuery>, KeyHash> cache;
    reach::Executor exec;
    bool busy = false;
    // per call
    std::vector<u64> key;
    std::vector<u32> resume;  // resume_after as u32 (empty: none)
    std::vector<ObjectId> binding;
    std::vector<ObjectId> lit_objects;
    std::vector<GroundLiteralView> lits[3];
};
}  // namespace mymyr::detail
