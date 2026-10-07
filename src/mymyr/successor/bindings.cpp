#include "binding_scratch.hpp"
#include "expr_eval.hpp"

#include "mymyr/successor/bindings.hpp"
#include "mymyr/task/numeric.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr
{
using namespace formalism;

// ================================================================================================ scratch
namespace detail
{
BindingScratch::BindingScratch(const Task& t) : task(&t), inner(std::make_unique<Workspace>(t)) {}
BindingScratch::~BindingScratch() = default;

namespace
{
constexpr u64 k_condition_key = 2, k_schema_key = 1;

/// Appends the condition's content to the cache key (with the section sizes, so that distinct conditions differ).
void append_key(const ConjunctiveCondition& c, std::vector<u64>& k)
{
    k.push_back(c.variables.size());
    for (const auto& v : c.variables)
    {
        k.push_back(v.types.size());
        for (TypeId t : v.types)
            k.push_back(t.v);
    }
    k.push_back(c.literals.size());
    for (const auto& l : c.literals)
    {
        k.push_back((static_cast<u64>(l.predicate.v) << 1) | (l.positive ? 1 : 0));
        k.push_back(l.terms.size());
        for (Term t : l.terms)
            k.push_back(static_cast<u64>(static_cast<i64>(t)));
    }
    k.push_back(c.equalities.size());
    for (const auto& e : c.equalities)
    {
        k.push_back(static_cast<u64>(static_cast<i64>(e.lhs)));
        k.push_back(static_cast<u64>(static_cast<i64>(e.rhs)));
        k.push_back(e.positive ? 1 : 0);
    }
    k.push_back(c.constraints.size());
    for (const auto& n : c.constraints)
    {
        k.push_back(static_cast<u64>(n.cmp));
        k.push_back(n.lhs);
        k.push_back(n.rhs);
    }
    k.push_back(c.exprs.size());
    for (const auto& x : c.exprs)
    {
        k.push_back(static_cast<u64>(x.op));
        k.push_back(std::bit_cast<u64>(x.value));
        k.push_back(x.func.v);
        k.push_back((static_cast<u64>(x.terms.begin) << 32) | x.terms.count);
        k.push_back((static_cast<u64>(x.a) << 32) | x.b);
    }
    k.push_back(c.expr_terms.size());
    for (Term t : c.expr_terms)
        k.push_back(static_cast<u64>(static_cast<i64>(t)));
}

/// The free variables a numeric constraint reads.
void expr_vars(const ConjunctiveCondition& c, u32 e, std::vector<i32>& out)
{
    const Expr& x = c.exprs[e];
    switch (x.op)
    {
        case ExprOp::Function:
            for (u32 j = x.terms.begin; j < x.terms.end(); ++j)
                if (!is_object(c.expr_terms[j]))
                    out.push_back(c.expr_terms[j]);
            break;
        case ExprOp::Add:
        case ExprOp::Sub:
        case ExprOp::Mul:
        case ExprOp::Div:
            expr_vars(c, x.a, out);
            expr_vars(c, x.b, out);
            break;
        case ExprOp::Neg: expr_vars(c, x.a, out); break;
        case ExprOp::Number: break;
    }
}

/// Narrows the plan's domains to the variables' declared types.
void restrict_types(reach::Plan& p, reach::StaticTables& st, const ConjunctiveCondition& c, const std::vector<u8>& fixed)
{
    const u32 OW = p.ow;
    for (u32 v = 0; v < c.arity(); ++v)
    {
        if (c.variables[v].types.empty())
            continue;
        std::vector<u32> types;
        for (TypeId t : c.variables[v].types)
            types.push_back(t.v);
        const u64* ts = st.objects_of_types(types);
        u64* d = p.dom0.data() + static_cast<usize>(v) * OW;
        for (u32 w = 0; w < OW; ++w)
            d[w] &= ts[w];
        p.dom_rng[v] = reach::word_range(d, OW);
        if (!fixed[v] && p.dom_rng[v] == 0)
            p.never = true;
    }
}

reach::CompileSpec spec_of(const BindingQuery& q)
{
    reach::CompileSpec spec;
    spec.num_vars = q.cond.arity();
    spec.prebound = q.fixed;
    spec.slot_mode = {0};
    spec.equality_by_identity = true;
    return spec;
}

void compile_query(BindingScratch& sc, BindingQuery& q)
{
    const Task& task = *sc.task;
    const ConjunctiveCondition& c = q.cond;
    const plan::Compiled& C = task.compiled();
    for (const auto& l : c.literals)
    {
        reach::Lit x;
        const PredKind k = C.kinds[l.predicate.v];
        x.type = k == PredKind::Static ? reach::LitType::Static : reach::LitType::Rel;
        x.positive = l.positive;
        x.pred = l.predicate.v;
        x.terms.assign(l.terms.begin(), l.terms.end());
        q.lits.push_back(std::move(x));
        q.kinds.push_back(k);
        q.terms += static_cast<u32>(l.terms.size());
    }
    for (const auto& e : c.equalities)
    {
        reach::Lit x;
        x.type = reach::LitType::Eq;
        x.positive = e.positive;
        x.terms = {e.lhs, e.rhs};
        q.lits.push_back(std::move(x));
    }
    for (u32 i = 0; i < c.constraints.size(); ++i)
    {
        reach::Lit x;
        x.type = reach::LitType::Custom;
        x.slot = i;
        expr_vars(c, c.constraints[i].lhs, x.terms);
        expr_vars(c, c.constraints[i].rhs, x.terms);
        q.lits.push_back(std::move(x));
    }
    if (!sc.tables)
        sc.tables = std::make_unique<reach::StaticTables>(task);
    q.plan = reach::compile(*sc.tables, q.lits, spec_of(q));
    restrict_types(q.plan, *sc.tables, c, q.fixed);
    for (const reach::Step& st : q.plan.steps)
        q.order.push_back(st.var);
}

void compile_resume(BindingScratch& sc, BindingQuery& q)
{
    std::vector<reach::Lit> lits = q.lits;
    const u32 nc = static_cast<u32>(q.cond.constraints.size());
    for (u32 d = 0; d < q.order.size(); ++d)
    {
        reach::Lit x;
        x.type = reach::LitType::Custom;
        x.slot = nc + d;
        for (u32 j = 0; j <= d; ++j)
            x.terms.push_back(static_cast<i32>(q.order[j]));
        lits.push_back(std::move(x));
    }
    q.resume = reach::compile(*sc.tables, lits, spec_of(q));
    restrict_types(q.resume, *sc.tables, q.cond, q.fixed);
    // the step order depends on the literals' rows and domains only, never on custom checks
    bool same = q.resume.steps.size() == q.order.size();
    for (usize d = 0; same && d < q.order.size(); ++d)
        same = q.resume.steps[d].var == q.order[d];
    if (!same)
        throw std::logic_error("mymyr: bindings: the resumable plan binds its variables in another order");
    q.has_resume = true;
}

/// The engine's view rows as join.hpp reads relation rows (every row over its full word range).
struct EngineRows
{
    const Engine* e;
    [[nodiscard]] const u64* row(u32 r) const noexcept { return e->view_row(r); }
    [[nodiscard]] u64 range(u32) const noexcept { return reach::pack_range(0, e->ow()); }
};

/// Membership of fluent and derived atoms (canonical ids) in the engine's current state and its axiom closure.
struct EngineAtoms
{
    const Engine* e;
    u64 fluent_count;
    [[nodiscard]] bool test(u64 key) const noexcept
    {
        const u32 s = e->atoms().find(key);
        if (s == AtomIndex::k_empty)
            return false;
        return key < fluent_count ? bits::test(e->state(), e->state_words(), s) : bits::test(e->derived(), e->derived_words(), s);
    }
};

struct Run
{
    BindingScratch& sc;
    BindingQuery& q;
    const plan::Numeric& N;
    const u64* num;
    EngineRows rows_;
    EngineAtoms atoms_;
    u64 limit;
    BindingSink* bsink;
    ConjunctionSink* csink;
    u64 count = 0;

    [[nodiscard]] const EngineRows& rows(u32) const noexcept { return rows_; }
    [[nodiscard]] const EngineAtoms& view(u32) const noexcept { return atoms_; }
    bool head_ok(const u32*) const noexcept { return true; }

    bool custom(u32 id, const u32* bind)
    {
        const u32 nc = static_cast<u32>(q.cond.constraints.size());
        if (id < nc)
        {
            const NumericConstraint& k = q.cond.constraints[id];
            return plan::compare(N, k.cmp, eval_expr(N, q.cond.exprs, q.cond.expr_terms, k.lhs, bind, num),
                                 eval_expr(N, q.cond.exprs, q.cond.expr_terms, k.rhs, bind, num));
        }
        // resume check of step d: the bound prefix is not below resume_after's, and above it once every step is bound
        const u32 d = id - nc;
        const u32* r = sc.resume.data();
        for (u32 j = 0; j <= d; ++j)
        {
            const u32 v = q.order[j];
            if (bind[v] != r[v])
                return bind[v] > r[v];
        }
        return d + 1 < q.order.size();
    }

    bool emit(const u32* bind)
    {
        const u32 V = q.cond.arity();
        ObjectId* b = sc.binding.data();
        for (u32 v = 0; v < V; ++v)
            b[v] = ObjectId{bind[v]};
        if (q.numeric_effects)
        {
            Successors& succ = sc.inner->successors();
            std::copy(b, b + V, succ.engine().bind());
            bool ok = true;
            (void) succ.effects(q.schema, &ok);
            if (!ok)
                return true;
        }
        ++count;
        const std::span<const ObjectId> binding(b, V);
        bool go = true;
        if (bsink)
            go = (*bsink)(binding);
        else
        {
            sc.lit_objects.clear();
            sc.lit_objects.reserve(q.terms);
            for (auto& l : sc.lits)
                l.clear();
            for (usize i = 0; i < q.cond.literals.size(); ++i)
            {
                const auto& l = q.cond.literals[i];
                const usize at = sc.lit_objects.size();
                for (Term t : l.terms)
                    sc.lit_objects.push_back(is_object(t) ? term_object(t) : b[t]);
                sc.lits[static_cast<u32>(q.kinds[i])].push_back(
                    {l.predicate, l.positive, std::span<const ObjectId>(sc.lit_objects.data() + at, l.terms.size())});
            }
            const GroundConjunction g{binding, sc.lits[static_cast<u32>(PredKind::Static)], sc.lits[static_cast<u32>(PredKind::Fluent)],
                                      sc.lits[static_cast<u32>(PredKind::Derived)]};
            go = (*csink)(g);
        }
        return go && count < limit;
    }
};

struct BusyGuard
{
    bool& b;
    explicit BusyGuard(bool& flag) : b(flag) { b = true; }
    ~BusyGuard() { b = false; }
    BusyGuard(const BusyGuard&) = delete;
    BusyGuard& operator=(const BusyGuard&) = delete;
};
}  // namespace

u64 enumerate_bindings(const Task& task, Workspace& ws, const BindingTarget& target, StateView s, PartialBinding partial,
                       const BindingOptions& options, BindingSink* bsink, ConjunctionSink* csink)
{
    if (&ws.engine().task() != &task)
        throw std::invalid_argument("mymyr: bindings: the workspace belongs to another task");
    BindingScratch& sc = ws.bindings();
    if (sc.busy)
        throw std::logic_error("mymyr: bindings: an enumeration is already running on this workspace (do not start one "
                               "from inside a binding callback; use another workspace)");
    const BusyGuard guard(sc.busy);
    const u32 n = task.num_objects();
    u32 arity = 0;
    if (target.condition)
        arity = target.condition->arity();
    else
    {
        if (target.schema.v >= task.num_schemas())
            throw std::invalid_argument("mymyr: bindings: schema index out of range");
        arity = task.compiled().schemas[target.schema.v].arity;
    }
    if (!partial.empty() && partial.size() != arity)
        throw std::invalid_argument("mymyr: bindings: the partial binding has " + std::to_string(partial.size()) +
                                    " entries, the target " + std::to_string(arity) + " variables");
    for (const auto& o : partial)
        if (o && o->v >= n)
            throw std::invalid_argument("mymyr: bindings: object index " + std::to_string(o->v) + " out of range");
    const bool resuming = !options.resume_after.empty();
    if (resuming)
    {
        if (options.resume_after.size() != arity)
            throw std::invalid_argument("mymyr: bindings: resume_after has the wrong number of objects");
        for (u32 v = 0; v < arity; ++v)
            if (options.resume_after[v].v >= n || (!partial.empty() && partial[v] && *partial[v] != options.resume_after[v]))
                throw std::invalid_argument("mymyr: bindings: resume_after does not agree with the partial binding");
    }
    if (options.limit == 0)
        return 0;

    // the compiled query of (target, fixed variables)
    std::vector<u64>& key = sc.key;
    key.clear();
    key.push_back(target.condition ? k_condition_key : k_schema_key);
    key.push_back(target.condition ? 0 : target.schema.v);
    for (u32 v = 0; v < arity; ++v)
        key.push_back(!partial.empty() && partial[v] ? 1 : 0);
    if (target.condition)
        append_key(*target.condition, key);
    auto it = sc.cache.find(key);
    if (it == sc.cache.end())
    {
        if (target.condition)
            target.condition->validate(task);
        if (sc.cache.size() >= BindingScratch::k_max_cached)
        {
            sc.cache.clear();
            sc.tables.reset();
        }
        auto q = std::make_unique<BindingQuery>();
        if (target.condition)
            q->cond = *target.condition;
        else
        {
            q->cond = ConjunctiveCondition::precondition(task, target.schema);
            q->schema_target = true;
            q->schema = target.schema.v;
            q->numeric_effects = task.compiled().schemas[target.schema.v].numeric();
        }
        q->fixed.assign(arity, 0);
        for (u32 v = 0; v < arity; ++v)
            q->fixed[v] = !partial.empty() && partial[v] ? 1 : 0;
        compile_query(sc, *q);
        it = sc.cache.emplace(key, std::move(q)).first;
    }
    BindingQuery& q = *it->second;
    if (resuming && q.order.empty())
        return 0;  // every variable is fixed: the only binding is resume_after itself
    if (resuming && !q.has_resume)
        compile_resume(sc, q);
    const reach::Plan& plan = resuming ? q.resume : q.plan;
    if (plan.never)
        return 0;

    Successors& succ = sc.inner->successors();
    succ.prepare(s);
    const Engine& e = succ.engine();
    sc.binding.resize(std::max<u32>(1, arity));
    sc.resume.clear();
    if (resuming)
        for (ObjectId o : options.resume_after)
            sc.resume.push_back(o.v);
    sc.exec.reserve(std::max<u32>(1, arity), static_cast<u32>(plan.steps.size()), plan.ow);
    u32* b = sc.exec.bind();
    for (u32 v = 0; v < arity; ++v)
        b[v] = !partial.empty() && partial[v] ? partial[v]->v : 0;
    Run run{sc, q, task.numeric(), s.num, EngineRows{&e}, EngineAtoms{&e, task.compiled().layout.fluent_count},
            options.limit, bsink, csink};
    sc.exec.run(plan, run);
    return run.count;
}
}  // namespace detail

// ================================================================================================ conveniences
namespace
{
std::vector<std::vector<ObjectId>> collect(const Task& task, Workspace& ws, const detail::BindingTarget& t, StateView s,
                                           PartialBinding partial, u64 limit)
{
    std::vector<std::vector<ObjectId>> out;
    auto f = [&](std::span<const ObjectId> b)
    {
        out.emplace_back(b.begin(), b.end());
        return true;
    };
    detail::BindingSink sink(f);
    BindingOptions o;
    o.limit = limit;
    detail::enumerate_bindings(task, ws, t, s, partial, o, &sink, nullptr);
    return out;
}

u64 count(const Task& task, Workspace& ws, const detail::BindingTarget& t, StateView s, PartialBinding partial, u64 limit)
{
    auto f = [](std::span<const ObjectId>) { return true; };
    detail::BindingSink sink(f);
    BindingOptions o;
    o.limit = limit;
    return detail::enumerate_bindings(task, ws, t, s, partial, o, &sink, nullptr);
}
}  // namespace

std::vector<std::vector<ObjectId>> bindings(const Task& task, Workspace& ws, const ConjunctiveCondition& condition, StateView s,
                                            PartialBinding partial, u64 limit)
{
    return collect(task, ws, {&condition, SchemaId{}}, s, partial, limit);
}

std::vector<std::vector<ObjectId>> bindings(const Task& task, Workspace& ws, SchemaId schema, StateView s, PartialBinding partial,
                                            u64 limit)
{
    return collect(task, ws, {nullptr, schema}, s, partial, limit);
}

u64 count_bindings(const Task& task, Workspace& ws, const ConjunctiveCondition& condition, StateView s, PartialBinding partial,
                   u64 limit)
{
    return count(task, ws, {&condition, SchemaId{}}, s, partial, limit);
}

u64 count_bindings(const Task& task, Workspace& ws, SchemaId schema, StateView s, PartialBinding partial, u64 limit)
{
    return count(task, ws, {nullptr, schema}, s, partial, limit);
}
}  // namespace mymyr
