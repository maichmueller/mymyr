// TaskData -> plan::Compiled. The matcher compiler (compile_matcher, compile_schema, the axiom stratification and
// the typed-dense domain rule of freeze_dense_ids) uses a fixed literal classification, binding order heuristic and
// forward-checking data, so that it enumerates exactly the same bindings in the same order on every run.

#include "compile.hpp"

#include "mymyr/core/bitset.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mymyr::detail
{
using namespace formalism;
using plan::TableRef;

void check_supported(const TaskData& t)
{
    for (const auto& p : t.predicates)
        if (p.arity > 64)
            throw std::invalid_argument("mymyr: predicate arity above 64 is not supported");
    for (const auto& f : t.functions)
        if (f.arity > 64)
            throw std::invalid_argument("mymyr: function arity above 64 is not supported");
}

namespace
{
std::vector<u32> distinct_params(std::span<const Term> terms)
{
    std::vector<u32> v;
    for (Term x : terms)
        if (x >= 0 && std::find(v.begin(), v.end(), static_cast<u32>(x)) == v.end())
            v.push_back(static_cast<u32>(x));
    return v;
}

u64 sat_mul(u64 a, u64 b)
{
    constexpr u64 cap = u64{1} << 62;
    if (a == 0 || b == 0)
        return 0;
    if (a > cap / b)
        return cap;
    return std::min(cap, a * b);
}

class Compiler
{
public:
    Compiler(const TaskData& t, const TaskOptions& opt, plan::Compiled& c) : T(t), O(opt), C(c) {}

    void run()
    {
        n = T.num_objects();
        OW = std::max<u32>(1, bits::words_for(n));
        C.num_objects = n;
        C.ow = OW;
        const u32 m = static_cast<u32>(T.predicates.size());
        C.kinds.resize(m);
        C.arity.resize(m);
        for (u32 p = 0; p < m; ++p)
        {
            C.kinds[p] = T.predicates[p].kind;
            C.arity[p] = T.predicates[p].arity;
        }
        setup_statics();
        setup_view();
        setup_numeric();
        compile_schemas();
        compile_axioms();
        compile_goal();
        compile_metric();
        build_layout();
        resolve_patterns();
        for (const auto& a : T.fluent_init)
        {
            std::vector<u32> rec{a.pred.v};
            for (ObjectId o : T.objects_of(a))
                rec.push_back(o.v);
            C.initial.push_back(std::move(rec));
        }
    }

private:
    // ------------------------------------------------------------------------------------------------ statics
    void setup_statics()
    {
        const u32 m = static_cast<u32>(T.predicates.size());
        s_unary.assign(m, {});
        static_atoms.assign(m, {});
        for (const auto& a : T.static_init)
            static_atoms[a.pred.v].push_back(&a);
        C.statics.resize(m);
        for (u32 p = 0; p < m; ++p)
        {
            if (C.kinds[p] != PredKind::Static)
                continue;
            if (C.arity[p] == 1)
            {
                s_unary[p].assign(OW, 0);
                for (const GroundAtom* a : static_atoms[p])
                    bits::set(s_unary[p].data(), T.objects_of(*a)[0].v);
            }
            build_relation(p);
        }
    }

    void build_relation(u32 p)
    {
        plan::StaticRelation& R = C.statics[p];
        const u32 ar = C.arity[p];
        R.arity = ar;
        std::vector<std::vector<u32>> rank(ar, std::vector<u32>(n, ~0u));
        std::vector<u32> dsize(ar, 0);
        for (const GroundAtom* a : static_atoms[p])
        {
            auto objs = T.objects_of(*a);
            for (u32 i = 0; i < ar; ++i)
                if (rank[i][objs[i].v] == ~0u)
                    rank[i][objs[i].v] = 0;  // mark; ranks assigned in object order below
        }
        for (u32 i = 0; i < ar; ++i)
            for (u32 o = 0; o < n; ++o)
                if (rank[i][o] == 0)
                    rank[i][o] = dsize[i]++;
        u64 size = 1;
        for (u32 i = 0; i < ar; ++i)
            size = sat_mul(size, dsize[i]);
        if (size >= plan::StaticRelation::k_outside)
            throw std::invalid_argument("mymyr: static predicate '" + std::string(T.str(T.predicates[p].name)) +
                                        "' has a key space of 2^56 or more");
        R.size = size;
        R.rs.assign(static_cast<usize>(ar) * n, plan::StaticRelation::k_outside);
        u64 stride = 1;
        for (u32 i = ar; i-- > 0;)
        {
            for (u32 o = 0; o < n; ++o)
                if (rank[i][o] != ~0u)
                    R.rs[static_cast<usize>(i) * n + o] = rank[i][o] * stride;
            stride *= std::max<u32>(1, dsize[i]);
        }
        auto key_of = [&](const GroundAtom& a)
        {
            u64 k = 0;
            auto objs = T.objects_of(a);
            for (u32 i = 0; i < ar; ++i)
                k += R.rs[static_cast<usize>(i) * n + objs[i].v];
            return k;
        };
        const u64 count = static_atoms[p].size();
        if (size <= (u64{1} << 24))
        {
            R.bits.assign(std::max<u32>(1, bits::words_for(size)), 0);
            for (const GroundAtom* a : static_atoms[p])
                bits::set(R.bits.data(), key_of(*a));
        }
        else
        {
            u64 cap = 64;
            while (cap < 2 * count)
                cap *= 2;
            R.table.assign(cap, 0);
            R.mask = cap - 1;
            for (const GroundAtom* a : static_atoms[p])
            {
                const u64 k = key_of(*a);
                for (u64 j = plan::StaticRelation::hash_key(k) & R.mask;; j = (j + 1) & R.mask)
                {
                    if (R.table[j] == k + 1)
                        break;
                    if (R.table[j] == 0)
                    {
                        R.table[j] = k + 1;
                        break;
                    }
                }
            }
        }
    }

    bool static_ground_holds(const Literal& l) const
    {
        const plan::StaticRelation& R = C.statics[l.pred.v];
        u64 k = 0;
        auto terms = T.terms_of(l);
        for (u32 i = 0; i < R.arity; ++i)
            k += R.rs[static_cast<usize>(i) * n + term_object(terms[i]).v];
        return R.contains(k);
    }

    /// Word offset (into C.static_words) of the n * OW row table of static `pred`: row(a[from]) has bit a[to].
    u64 static_rows(u32 pred, u32 from, u32 to)
    {
        const auto key = std::make_tuple(pred, from, to);
        if (auto it = row_tables.find(key); it != row_tables.end())
            return it->second;
        const u64 off = C.static_words.size();
        C.static_words.resize(off + static_cast<usize>(n) * OW, 0);
        for (const GroundAtom* a : static_atoms[pred])
        {
            auto objs = T.objects_of(*a);
            bits::set(C.static_words.data() + off + static_cast<usize>(objs[from].v) * OW, objs[to].v);
        }
        row_tables.emplace(key, off);
        return off;
    }

    void setup_view()
    {
        const u32 m = static_cast<u32>(T.predicates.size());
        ViewLayout& V = C.view;
        V.num_objects = n;
        V.ow = OW;
        V.unary_row.assign(m, ViewLayout::k_none);
        V.fwd_row.assign(m, ViewLayout::k_none);
        V.bwd_row.assign(m, ViewLayout::k_none);
        u32 rows = 0;
        for (u32 p = 0; p < m; ++p)
        {
            if (C.kinds[p] == PredKind::Static)
                continue;
            if (C.arity[p] == 1)
                V.unary_row[p] = rows++;
            else if (C.arity[p] == 2)
            {
                V.fwd_row[p] = rows;
                rows += n;
                V.bwd_row[p] = rows;
                rows += n;
            }
        }
        V.rows = rows;
    }

    // ------------------------------------------------------------------------------------------------ numerics
    /// Parameters (terms >= 0) of the function terms of expression e.
    void expr_params(u32 e, std::vector<u32>& out) const
    {
        const Expr& x = T.exprs[e];
        switch (x.op)
        {
            case ExprOp::Number: return;
            case ExprOp::Function:
                for (Term t : TaskData::slice(T.terms, x.terms))
                    if (t >= 0 && std::find(out.begin(), out.end(), static_cast<u32>(t)) == out.end())
                        out.push_back(static_cast<u32>(t));
                return;
            case ExprOp::Neg: expr_params(x.a, out); return;
            default:
                expr_params(x.a, out);
                expr_params(x.b, out);
        }
    }
    std::vector<u32> constraint_params(const NumericConstraint& c) const
    {
        std::vector<u32> v;
        expr_params(c.lhs, v);
        expr_params(c.rhs, v);
        return v;
    }
    void effect_params(const NumericEffect& e, std::vector<u32>& out) const
    {
        for (Term t : TaskData::slice(T.terms, e.terms))
            if (t >= 0 && std::find(out.begin(), out.end(), static_cast<u32>(t)) == out.end())
                out.push_back(static_cast<u32>(t));
        expr_params(e.expr, out);
    }

    /// Whether expression e keeps integral values integral (integral numbers and static values, no division).
    bool integral_expr(u32 e) const
    {
        const Expr& x = T.exprs[e];
        switch (x.op)
        {
            case ExprOp::Number: return std::isfinite(x.value) && x.value == std::nearbyint(x.value);
            case ExprOp::Function:
            {
                const Function& f = T.functions[x.func.v];
                if (f.kind == FuncKind::Static)
                    return static_integral[x.func.v] != 0;
                return f.kind == FuncKind::Fluent;
            }
            case ExprOp::Neg: return integral_expr(x.a);
            case ExprOp::Div: return false;
            default: return integral_expr(x.a) && integral_expr(x.b);
        }
    }

    u64 function_key(const plan::FunctionTable& F, std::span<const ObjectId> objs) const
    {
        u64 k = 0;
        for (u32 i = 0; i < F.arity; ++i)
            k += F.rs[static_cast<usize>(i) * n + objs[i].v];
        return k;
    }
    u32 new_slot(u32 f, const GroundFunctionValue* v)
    {
        const u32 s = N.slots++;
        N.initial.push_back(0);
        N.slot_function.push_back(f);
        const auto objs = TaskData::slice(T.object_ids, v->objects);
        N.slot_args.insert(N.slot_args.end(), objs.begin(), objs.end());
        N.slot_args_begin.push_back(static_cast<u32>(N.slot_args.size()));
        return s;
    }

    /// The argument table of function f from its defined values (fluent: new slots in the order of `vals`).
    void build_function_table(u32 f, const std::vector<const GroundFunctionValue*>& vals)
    {
        plan::FunctionTable& F = N.tables[f];
        const u32 ar = T.functions[f].arity;
        F.arity = ar;
        F.fluent = T.functions[f].kind == FuncKind::Fluent;
        std::vector<std::vector<u32>> rank(ar, std::vector<u32>(n, ~0u));
        std::vector<u32> dsize(ar, 0);
        for (const GroundFunctionValue* v : vals)
        {
            const auto objs = TaskData::slice(T.object_ids, v->objects);
            for (u32 i = 0; i < ar; ++i)
                rank[i][objs[i].v] = 0;
        }
        for (u32 i = 0; i < ar; ++i)
            for (u32 o = 0; o < n; ++o)
                if (rank[i][o] == 0)
                    rank[i][o] = dsize[i]++;
        u64 size = vals.empty() ? 0 : 1;
        for (u32 i = 0; i < ar && size; ++i)
            size = sat_mul(size, dsize[i]);
        if (size >= plan::FunctionTable::k_outside)
            throw std::invalid_argument("mymyr: function '" + std::string(T.str(T.functions[f].name)) + "' has a key space of 2^56 or more");
        F.size = size;
        F.rs.assign(static_cast<usize>(ar) * n, plan::FunctionTable::k_outside);
        u64 stride = 1;
        for (u32 i = ar; i-- > 0;)
        {
            for (u32 o = 0; o < n; ++o)
                if (rank[i][o] != ~0u)
                    F.rs[static_cast<usize>(i) * n + o] = rank[i][o] * stride;
            stride *= std::max<u32>(1, dsize[i]);
        }
        const bool dense = size <= (u64{1} << 24);
        if (!dense)
        {
            u64 cap = 64;
            while (cap < 2 * vals.size())
                cap *= 2;
            F.hkeys.assign(cap, 0);
            F.hmask = cap - 1;
            if (F.fluent)
                F.hslot.assign(cap, plan::FunctionTable::k_none);
            else
                F.hvalue.assign(cap, std::numeric_limits<f64>::quiet_NaN());
        }
        else if (F.fluent)
            F.slot.assign(size, plan::FunctionTable::k_none);
        else
            F.value.assign(size, std::numeric_limits<f64>::quiet_NaN());
        for (const GroundFunctionValue* v : vals)
        {
            const u64 k = function_key(F, TaskData::slice(T.object_ids, v->objects));
            if (dense)
            {
                if (F.fluent)
                {
                    if (F.slot[k] == plan::FunctionTable::k_none)
                        F.slot[k] = new_slot(f, v);
                    N.initial[F.slot[k]] = v->value;  // a repeated value overrides
                }
                else
                    F.value[k] = v->value;
                continue;
            }
            u64 j = plan::FunctionTable::hash_key(k) & F.hmask;
            while (F.hkeys[j] != 0 && F.hkeys[j] != k + 1)
                j = (j + 1) & F.hmask;
            if (F.hkeys[j] == 0)
            {
                F.hkeys[j] = k + 1;
                if (F.fluent)
                    F.hslot[j] = new_slot(f, v);
            }
            if (F.fluent)
                N.initial[F.hslot[j]] = v->value;
            else
                F.hvalue[j] = v->value;
        }
    }

    void setup_numeric()
    {
        N.num_objects = n;
        N.quantum = O.numeric_quantum;
        N.tolerant = O.numeric_tolerant;
        N.slot_args_begin.assign(1, 0);
        const u32 m = static_cast<u32>(T.functions.size());
        N.tables.assign(m, {});
        static_integral.assign(m, 1);
        std::vector<std::vector<const GroundFunctionValue*>> svals(m), fvals(m);
        for (const auto& v : T.static_values)
        {
            svals[v.func.v].push_back(&v);
            if (!(std::isfinite(v.value) && v.value == std::nearbyint(v.value)))
                static_integral[v.func.v] = 0;
        }
        for (const auto& v : T.fluent_values)
            fvals[v.func.v].push_back(&v);
        for (u32 f = 0; f < m; ++f)  // slots in (function, arguments) order
            std::stable_sort(fvals[f].begin(), fvals[f].end(),
                             [&](const GroundFunctionValue* a, const GroundFunctionValue* b)
                             {
                                 const auto x = TaskData::slice(T.object_ids, a->objects);
                                 const auto y = TaskData::slice(T.object_ids, b->objects);
                                 return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
                             });
        for (u32 f = 0; f < m; ++f)
            switch (T.functions[f].kind)
            {
                case FuncKind::Static: build_function_table(f, svals[f]); break;
                case FuncKind::Fluent: build_function_table(f, fvals[f]); break;
                case FuncKind::Auxiliary: N.has_aux = true; break;
            }
        N.aux_initial = T.auxiliary_initial.value_or(0.0);
        // storage: I32 when the task is statically integral and exact
        bool integral = O.numeric_quantum == 0 && !O.numeric_tolerant;
        for (f64 v : N.initial)
            integral = integral && std::isfinite(v) && v == std::nearbyint(v) && v >= -2147483648.0 && v <= 2147483647.0;
        for (const NumericEffect& e : T.numeric_effects)
            integral = integral && e.op != AssignOp::ScaleDown && integral_expr(e.expr);
        switch (O.numeric_storage)
        {
            case TaskOptions::NumericStorageMode::F64: N.storage = NumericStorage::F64; break;
            case TaskOptions::NumericStorageMode::I32: N.storage = NumericStorage::I32; break;
            case TaskOptions::NumericStorageMode::Auto: N.storage = integral ? NumericStorage::I32 : NumericStorage::F64; break;
        }
        N.words = plan::Numeric::words_for(N.slots, N.storage);
        if (N.storage == NumericStorage::I32)
            for (f64 v : N.initial)
                if (!(v >= -2147483648.0 && v <= 2147483647.0) || v != std::nearbyint(v))
                    throw std::invalid_argument("mymyr: numeric_storage I32, but an initial value is not an int32");
    }

    void emit_const(f64 v)
    {
        N.code.push_back({plan::NumOp::Const, 0, static_cast<u32>(N.consts.size()), 0});
        N.consts.push_back(v);
    }
    [[nodiscard]] bool single_const_since(usize start) const
    {
        return N.code.size() == start + 1 && N.code.back().op == plan::NumOp::Const;
    }
    /// Structural key of code[b, e): operators, constant bit patterns, slots, tables and their argument terms (the
    /// offsets into N.consts / N.terms do not matter).
    [[nodiscard]] std::string code_key(usize b, usize e) const
    {
        std::string k;
        auto put = [&](u64 v) { k.append(reinterpret_cast<const char*>(&v), sizeof v); };
        for (usize i = b; i < e; ++i)
        {
            const plan::NumIns& in = N.code[i];
            put(static_cast<u64>(in.op) | (static_cast<u64>(in.ar) << 8));
            switch (in.op)
            {
                case plan::NumOp::Const: put(std::bit_cast<u64>(N.consts[in.a])); break;
                case plan::NumOp::SlotF64:
                case plan::NumOp::SlotI32: put(in.a); break;
                case plan::NumOp::LoadF64:
                case plan::NumOp::LoadI32:
                case plan::NumOp::LoadStatic:
                    put(in.a);
                    for (u32 j = 0; j < in.ar; ++j)
                        put(static_cast<u64>(static_cast<i64>(N.terms[in.b + j])));
                    break;
                default: break;
            }
        }
        return k;
    }

    /// Postfix code of expression e with constant folding (static functions of objects, operators over constants).
    void emit_expr(u32 e)
    {
        const Expr& x = T.exprs[e];
        const f64 nan = std::numeric_limits<f64>::quiet_NaN();
        switch (x.op)
        {
            case ExprOp::Number: emit_const(x.value); return;
            case ExprOp::Function:
            {
                const Function& f = T.functions[x.func.v];
                const auto terms = TaskData::slice(T.terms, x.terms);
                if (f.kind == FuncKind::Auxiliary)
                {
                    emit_const(0.0);  // total-cost is not part of the state (it only feeds the metric)
                    return;
                }
                const plan::FunctionTable& F = N.tables[x.func.v];
                if (std::all_of(terms.begin(), terms.end(), [](Term t) { return t < 0; }))
                {
                    std::vector<ObjectId> objs;
                    for (Term t : terms)
                        objs.push_back(term_object(t));
                    const u64 k = function_key(F, objs);
                    if (!F.fluent)
                        emit_const(F.value_of(k));
                    else if (const u32 s = F.slot_of(k); s == plan::FunctionTable::k_none)
                        emit_const(nan);
                    else
                        N.code.push_back({N.storage == NumericStorage::I32 ? plan::NumOp::SlotI32 : plan::NumOp::SlotF64, 0, s, 0});
                    return;
                }
                const plan::NumOp op = !F.fluent ? plan::NumOp::LoadStatic
                                                 : (N.storage == NumericStorage::I32 ? plan::NumOp::LoadI32 : plan::NumOp::LoadF64);
                N.code.push_back({op, static_cast<u8>(terms.size()), x.func.v, static_cast<u32>(N.terms.size())});
                N.terms.insert(N.terms.end(), terms.begin(), terms.end());
                return;
            }
            case ExprOp::Neg:
            {
                const usize start = N.code.size();
                emit_expr(x.a);
                if (single_const_since(start))
                    N.consts[N.code.back().a] = -N.consts[N.code.back().a];
                else
                    N.code.push_back({plan::NumOp::Neg, 0, 0, 0});
                return;
            }
            default:
            {
                const usize start = N.code.size();
                emit_expr(x.a);
                const bool ca = single_const_since(start);
                const usize mid = N.code.size();
                emit_expr(x.b);
                const bool cb = single_const_since(mid);
                if (ca && cb)
                {
                    const f64 r = N.consts[N.code.back().a];
                    N.code.pop_back();
                    f64& l = N.consts[N.code.back().a];
                    switch (x.op)
                    {
                        case ExprOp::Add: l = l + r; break;
                        case ExprOp::Sub: l = l - r; break;
                        case ExprOp::Mul: l = l * r; break;
                        default: l = r == 0 ? nan : l / r; break;
                    }
                    if (N.tolerant && !std::isfinite(l))
                        l = nan;
                    return;
                }
                plan::NumOp op = plan::NumOp::Div;
                if (x.op == ExprOp::Add)
                    op = plan::NumOp::Add;
                else if (x.op == ExprOp::Sub)
                    op = plan::NumOp::Sub;
                else if (x.op == ExprOp::Mul)
                    op = plan::NumOp::Mul;
                // Commutative operands in a canonical order, so that equal expressions compile to equal
                // code (compile_expr shares it); a + b and b + a (a * b, b * a) are the same IEEE-754 result
                if ((op == plan::NumOp::Add || op == plan::NumOp::Mul) && code_key(mid, N.code.size()) < code_key(start, mid))
                    std::rotate(N.code.begin() + static_cast<std::ptrdiff_t>(start), N.code.begin() + static_cast<std::ptrdiff_t>(mid),
                                N.code.end());
                N.code.push_back({op, 0, 0, 0});
            }
        }
    }
    plan::NumProg compile_expr(u32 e)
    {
        plan::NumProg p;
        const usize consts = N.consts.size(), terms = N.terms.size();
        p.begin = static_cast<u32>(N.code.size());
        emit_expr(e);
        p.end = static_cast<u32>(N.code.size());
        // equal expressions share one program
        std::string key = code_key(p.begin, p.end);
        if (const auto it = m_programs.find(key); it != m_programs.end())
        {
            N.code.resize(p.begin);
            N.consts.resize(consts);
            N.terms.resize(terms);
            return it->second;
        }
        m_programs.emplace(std::move(key), p);
        u32 sp = 0, mx = 0;
        for (u32 i = p.begin; i < p.end; ++i)
            switch (N.code[i].op)
            {
                case plan::NumOp::Add:
                case plan::NumOp::Sub:
                case plan::NumOp::Mul:
                case plan::NumOp::Div: --sp; break;
                case plan::NumOp::Neg: break;
                default: mx = std::max(mx, ++sp); break;
            }
        if (mx > plan::Numeric::k_max_stack)
            throw std::invalid_argument("mymyr: a numeric expression is nested too deeply");
        return p;
    }
    plan::NumCheck compile_check(const NumericConstraint& c) { return {c.cmp, compile_expr(c.lhs), compile_expr(c.rhs)}; }
    plan::NumEffect compile_effect(const NumericEffect& e)
    {
        plan::NumEffect c;
        c.op = e.op;
        c.expr = compile_expr(e.expr);
        const auto terms = TaskData::slice(T.terms, e.terms);
        const plan::FunctionTable& F = N.tables[e.func.v];
        c.arity = static_cast<u8>(terms.size());
        if (std::all_of(terms.begin(), terms.end(), [](Term t) { return t < 0; }))
        {
            std::vector<ObjectId> objs;
            for (Term t : terms)
                objs.push_back(term_object(t));
            c.slot = F.slot_of(function_key(F, objs));
        }
        else
        {
            c.lifted = true;
            c.table = e.func.v;
            c.terms = static_cast<u32>(N.terms.size());
            N.terms.insert(N.terms.end(), terms.begin(), terms.end());
        }
        return c;
    }

    void compile_metric()
    {
        if (!T.metric)
            return;
        N.has_metric = true;
        N.minimize = T.metric->minimize;
        N.metric = compile_expr(T.metric->expr);
    }

    // ------------------------------------------------------------------------------------------------ matchers
    plan::LitKind kind_of(u32 p) const
    {
        switch (C.kinds[p])
        {
            case PredKind::Static: return plan::LitKind::Static;
            case PredKind::Fluent: return plan::LitKind::Fluent;
            case PredKind::Derived: return plan::LitKind::Derived;
        }
        return plan::LitKind::Static;
    }
    plan::Pattern pattern(const Literal& l) const
    {
        plan::Pattern pt;
        pt.kind = kind_of(l.pred.v);
        pt.pred = l.pred.v;
        pt.terms = l.terms;
        return pt;
    }
    plan::Check check(const Literal& l) const { return {pattern(l), l.positive}; }

    u64* dom(plan::Matcher& m, u32 v) { return m.dom0.data() + static_cast<usize>(v) * OW; }
    void and_words(u64* d, const u64* s, bool neg)
    {
        for (u32 w = 0; w < OW; ++w)
            d[w] &= neg ? ~s[w] : s[w];
    }
    long popc(const plan::Matcher& m, u32 v) const
    {
        long k = 0;
        for (u32 w = 0; w < OW; ++w)
            k += std::popcount(m.dom0[static_cast<usize>(v) * OW + w]);
        return k;
    }

    plan::Matcher matcher(std::span<const Literal> lits, u32 total, u32 first_free, const std::vector<u8>& relevant,
                          std::span<const NumericConstraint> cons = {})
    {
        plan::Matcher m;
        m.total = total;
        m.first_free = first_free;
        m.dom0.assign(static_cast<usize>(total) * OW, 0);
        for (u32 v = first_free; v < total; ++v)
            for (u32 o = 0; o < n; ++o)
                bits::set(dom(m, v), o);
        std::vector<std::vector<plan::Unary>> unary(total);
        auto is_free = [&](Term x) { return x >= 0 && static_cast<u32>(x) >= first_free; };

        struct Binary
        {
            u32 a, b;  // parameters at positions pa, pb of pred
            u32 pred, pa, pb;
            bool fluent, pos;
        };
        std::vector<Binary> binaries;
        std::vector<const Literal*> leaf;
        for (const Literal& l : lits)
        {
            const u32 p = l.pred.v;
            const bool fluent = C.kinds[p] != PredKind::Static;  // fluent or derived: per-state
            const auto terms = T.terms_of(l);
            const auto vars = distinct_params(terms);
            std::vector<u32> fvars;
            for (u32 v : vars)
                if (v >= first_free)
                    fvars.push_back(v);
            if (fvars.empty())
            {
                if (!fluent && vars.empty())
                {
                    if (static_ground_holds(l) != l.positive)
                        m.never = true;
                }
                else
                    m.pre_checks.push_back(check(l));
                continue;
            }
            const u32 ar = C.arity[p];
            if (ar == 1)
            {
                const u32 v = fvars[0];
                if (fluent)
                    unary[v].push_back({TableRef::view_row(C.view.unary_row[p]), !l.positive});
                else
                    and_words(dom(m, v), s_unary[p].data(), !l.positive);
                continue;
            }
            if (ar == 2 && terms[0] != terms[1])
            {
                const bool f0 = is_free(terms[0]), f1 = is_free(terms[1]);
                if (f0 && f1)
                {
                    binaries.push_back({static_cast<u32>(terms[0]), static_cast<u32>(terms[1]), p, 0, 1, fluent, l.positive});
                    continue;
                }
                // exactly one free parameter; the other term is a constant or a prebound parameter
                const u32 fpos = f0 ? 0 : 1, opos = 1 - fpos;
                const u32 v = static_cast<u32>(terms[fpos]);
                const Term other = terms[opos];
                if (other >= 0)
                {
                    binaries.push_back({static_cast<u32>(other), v, p, opos, fpos, fluent, l.positive});
                    continue;
                }
                const u32 cobj = term_object(other).v;
                if (fluent)
                {
                    const u32 row = fpos == 0 ? C.view.bwd_row[p] + cobj : C.view.fwd_row[p] + cobj;
                    unary[v].push_back({TableRef::view_row(row), !l.positive});
                }
                else
                {
                    const u64 off = static_rows(p, opos, fpos);
                    and_words(dom(m, v), C.static_words.data() + off + static_cast<usize>(cobj) * OW, !l.positive);
                }
                continue;
            }
            // arity >= 3 or repeated variables: checked when bound. Static positive k-ary literals additionally
            // contribute projections as row / domain constraints (sound over-approximations).
            if (!fluent && l.positive && ar >= 3)
            {
                for (u32 i = 0; i < ar; ++i)
                    for (u32 j = 0; j < ar; ++j)
                        if (i != j && is_free(terms[j]) && terms[i] >= 0 && terms[i] != terms[j] && (!is_free(terms[i]) || i < j))
                            binaries.push_back({static_cast<u32>(terms[i]), static_cast<u32>(terms[j]), p, i, j, false, true});
                for (u32 i = 0; i < ar; ++i)
                    if (is_free(terms[i]))
                    {
                        std::vector<u64> proj(OW, 0);
                        for (const GroundAtom* a : static_atoms[p])
                        {
                            auto objs = T.objects_of(*a);
                            bool ok = true;
                            for (u32 j = 0; j < ar && ok; ++j)
                                if (terms[j] < 0 && objs[j] != term_object(terms[j]))
                                    ok = false;
                            if (ok)
                                bits::set(proj.data(), objs[i].v);
                        }
                        and_words(dom(m, static_cast<u32>(terms[i])), proj.data(), false);
                    }
            }
            leaf.push_back(&l);
        }
        for (u32 v = first_free; v < total; ++v)
            if (!bits::any(dom(m, v), OW))
                m.never = true;

        // Order free parameters: effect-relevant first, then witness-only; inside each group greedily pick the
        // parameter with the most constraints to already placed parameters, ties broken by fewest candidates.
        std::vector<u32> order;
        std::vector<bool> placed(total, false);
        for (u32 v = 0; v < first_free; ++v)
            placed[v] = true;
        for (int group = 0; group < 2; ++group)
            for (;;)
            {
                int best = -1;
                long best_score = 0;
                for (u32 v = first_free; v < total; ++v)
                {
                    if (placed[v] || (relevant[v] != 0) != (group == 0))
                        continue;
                    long links = 0;
                    for (const auto& b : binaries)
                        if ((b.a == v && placed[b.b]) || (b.b == v && placed[b.a]))
                            ++links;
                    const long score = links * 1000000 + static_cast<long>(unary[v].size()) * 10000 - popc(m, v);
                    if (best < 0 || score > best_score)
                        best = static_cast<int>(v), best_score = score;
                }
                if (best < 0)
                    break;
                placed[best] = true;
                order.push_back(static_cast<u32>(best));
            }
        m.first_exist = 0;
        while (m.first_exist < order.size() && relevant[order[m.first_exist]])
            ++m.first_exist;
        std::vector<i32> pos_of(total, -1);  // prebound parameters come "before" step 0
        for (u32 d = 0; d < order.size(); ++d)
            pos_of[order[d]] = static_cast<i32>(d);

        std::vector<std::vector<plan::Row>> step_rows(order.size());
        std::vector<std::vector<u32>> step_checks(order.size());
        for (const auto& b : binaries)
        {
            const bool a_first = pos_of[b.a] < pos_of[b.b];
            const u32 later = a_first ? b.b : b.a, earlier = a_first ? b.a : b.b;
            const u32 epos = a_first ? b.pa : b.pb, lpos = a_first ? b.pb : b.pa;
            TableRef base;
            if (b.fluent)  // binary fluent: row indexed by the value at epos, bits over values at lpos
                base = TableRef::view_row(epos == 0 ? C.view.fwd_row[b.pred] : C.view.bwd_row[b.pred]);
            else
                base = TableRef::static_words(static_rows(b.pred, epos, lpos));
            step_rows[pos_of[later]].push_back({base, earlier, !b.pos});
        }
        for (const Literal* l : leaf)
        {
            i32 last = -1;
            for (u32 v : distinct_params(T.terms_of(*l)))
                last = std::max(last, pos_of[v]);
            step_checks[last].push_back(static_cast<u32>(m.checks.size()));
            m.checks.push_back(check(*l));
        }
        // numeric constraints: checked at the step of their last free parameter (the probe's early placement)
        std::vector<std::vector<u32>> step_nchecks(order.size());
        std::vector<std::vector<u32>> nfree;
        for (const NumericConstraint& c : cons)
        {
            std::vector<u32> fv;
            for (u32 v : constraint_params(c))
                if (v >= first_free)
                    fv.push_back(v);
            const plan::NumCheck nc = compile_check(c);
            if (fv.empty())
            {
                m.npre.push_back(nc);
                continue;
            }
            i32 last = -1;
            for (u32 v : fv)
                last = std::max(last, pos_of[v]);
            step_nchecks[last].push_back(static_cast<u32>(m.nchecks.size()));
            m.nchecks.push_back(nc);
            nfree.push_back(std::move(fv));
        }
        for (u32 d = 0; d < order.size(); ++d)
        {
            plan::Step st;
            st.param = order[d];
            st.row_begin = static_cast<u32>(m.rows.size());
            m.rows.insert(m.rows.end(), step_rows[d].begin(), step_rows[d].end());
            st.row_end = static_cast<u32>(m.rows.size());
            st.check_begin = static_cast<u32>(m.step_checks.size());
            m.step_checks.insert(m.step_checks.end(), step_checks[d].begin(), step_checks[d].end());
            st.check_end = static_cast<u32>(m.step_checks.size());
            st.ncheck_begin = static_cast<u32>(m.step_nchecks.size());
            m.step_nchecks.insert(m.step_nchecks.end(), step_nchecks[d].begin(), step_nchecks[d].end());
            st.ncheck_end = static_cast<u32>(m.step_nchecks.size());
            m.steps.push_back(st);
        }
        {
            std::vector<std::vector<u32>> nchecks_of(total);
            m.ncheck_vars_begin.assign(1, 0);
            for (u32 ci = 0; ci < m.nchecks.size(); ++ci)
            {
                for (u32 v : nfree[ci])
                {
                    m.ncheck_vars.push_back(v);
                    nchecks_of[v].push_back(ci);
                }
                m.ncheck_vars_begin.push_back(static_cast<u32>(m.ncheck_vars.size()));
            }
            m.fc_nchecks_begin.assign(total + 1, 0);
            for (u32 v = 0; v < total; ++v)
            {
                m.fc_nchecks_begin[v] = static_cast<u32>(m.fc_nchecks.size());
                m.fc_nchecks.insert(m.fc_nchecks.end(), nchecks_of[v].begin(), nchecks_of[v].end());
            }
            m.fc_nchecks_begin[total] = static_cast<u32>(m.fc_nchecks.size());
        }
        m.unary_begin.assign(total + 1, 0);
        for (u32 v = 0; v < total; ++v)
        {
            m.unary_begin[v] = static_cast<u32>(m.unary.size());
            m.unary.insert(m.unary.end(), unary[v].begin(), unary[v].end());
        }
        m.unary_begin[total] = static_cast<u32>(m.unary.size());

        // Forward-checking data: rows in both directions between free parameters, prebound rows folded up front.
        m.relevant = relevant;
        m.free_params = order;
        std::vector<std::vector<plan::Edge>> fc_out(total);
        auto row_base = [&](const Binary& b, bool from_a) -> TableRef
        {
            const u32 fpos = from_a ? b.pa : b.pb, tpos = from_a ? b.pb : b.pa;
            if (b.fluent)
                return TableRef::view_row(fpos == 0 ? C.view.fwd_row[b.pred] : C.view.bwd_row[b.pred]);
            return TableRef::static_words(static_rows(b.pred, fpos, tpos));
        };
        for (const auto& b : binaries)
        {
            const bool a_free = b.a >= first_free, b_free = b.b >= first_free;
            if (a_free && b_free)
            {
                fc_out[b.a].push_back({b.b, row_base(b, true), !b.pos});
                fc_out[b.b].push_back({b.a, row_base(b, false), !b.pos});
            }
            else if (a_free)
            {
                m.fc_pre.push_back({row_base(b, false), b.b, !b.pos});
                m.fc_pre_to.push_back(b.a);
            }
            else
            {
                m.fc_pre.push_back({row_base(b, true), b.a, !b.pos});
                m.fc_pre_to.push_back(b.b);
            }
        }
        m.fc_out_begin.assign(total + 1, 0);
        for (u32 v = 0; v < total; ++v)
        {
            m.fc_out_begin[v] = static_cast<u32>(m.fc_out.size());
            m.fc_out.insert(m.fc_out.end(), fc_out[v].begin(), fc_out[v].end());
        }
        m.fc_out_begin[total] = static_cast<u32>(m.fc_out.size());
        std::vector<std::vector<u32>> checks_of(total);
        m.check_vars_begin.push_back(0);
        for (u32 ci = 0; ci < m.checks.size(); ++ci)
        {
            for (u32 v : distinct_params(T.terms_of(*leaf[ci])))
                if (v >= first_free)
                {
                    m.check_vars.push_back(v);
                    checks_of[v].push_back(ci);
                }
            m.check_vars_begin.push_back(static_cast<u32>(m.check_vars.size()));
        }
        m.fc_checks_begin.assign(total + 1, 0);
        for (u32 v = 0; v < total; ++v)
        {
            m.fc_checks_begin[v] = static_cast<u32>(m.fc_checks.size());
            m.fc_checks.insert(m.fc_checks.end(), checks_of[v].begin(), checks_of[v].end());
        }
        m.fc_checks_begin[total] = static_cast<u32>(m.fc_checks.size());

        switch (O.matching)
        {
            case TaskOptions::Matching::FixedOrder: m.use_fc = false; break;
            case TaskOptions::Matching::ForwardChecking: m.use_fc = true; break;
            case TaskOptions::Matching::Auto: m.use_fc = order.size() > O.fc_auto_free_params; break;
        }
        m.binds_in_order = !m.use_fc;
        for (u32 d = 0; d < order.size(); ++d)
            if (order[d] != first_free + d)
                m.binds_in_order = false;
        return m;
    }

    void compile_schemas()
    {
        for (const Schema& s : T.schemas)
        {
            plan::Schema c;
            c.arity = s.arity();
            c.bind_size = c.arity;
            // Effect-relevant parameters: in literal effects, conditions of conditional effects (literals and numeric
            // constraints), numeric effects (targets and values) and total-cost effects: bindings that differ only in
            // witness parameters must not differ in any effect, applicability rule or cost.
            std::vector<u8> relevant(c.arity, 0);
            auto mark = [&](const std::vector<u32>& ps)
            {
                for (u32 x : ps)
                    if (x < c.arity)
                        relevant[x] = 1;
            };
            for (const auto& ce : T.effects_of(s))
            {
                for (Range r : {ce.condition.literals, ce.effects})
                    for (const Literal& l : TaskData::slice(T.literals, r))
                        for (Term x : T.terms_of(l))
                            if (x >= 0 && static_cast<u32>(x) < c.arity)
                                relevant[x] = 1;
                for (const NumericConstraint& nc : T.constraints_of(ce.condition))
                    mark(constraint_params(nc));
                std::vector<u32> ps;
                for (const NumericEffect& e : TaskData::slice(T.numeric_effects, ce.numeric_effects))
                    effect_params(e, ps);
                if (ce.auxiliary)
                    effect_params(*ce.auxiliary, ps);
                mark(ps);
            }
            const auto pre = T.literals_of(s.precondition);
            const auto pre_cons = T.constraints_of(s.precondition);
            c.pre[0] = matcher(pre, c.arity, 0, relevant, pre_cons);
            c.pre[1] = matcher(pre, c.arity, 0, std::vector<u8>(c.arity, 1), pre_cons);
            for (const Literal& l : pre)
                c.pre_lits.push_back(check(l));
            for (const NumericConstraint& nc : pre_cons)
                c.pre_nums.push_back(compile_check(nc));
            for (const auto& ce : T.effects_of(s))
            {
                const auto effs = TaskData::slice(T.literals, ce.effects);
                const auto neffs = TaskData::slice(T.numeric_effects, ce.numeric_effects);
                const bool numeric = !neffs.empty() || ce.auxiliary.has_value();
                const u32 extra = ce.extra_params.count;
                if (extra == 0 && ce.condition.literals.count == 0 && ce.condition.constraints.count == 0)
                {
                    for (const Literal& l : effs)
                        (l.positive ? c.adds : c.dels).push_back(pattern(l));
                    if (numeric)
                    {
                        plan::NumGroup g;
                        for (const NumericEffect& e : neffs)
                            g.neffs.push_back(compile_effect(e));
                        if (ce.auxiliary)
                        {
                            g.has_aux = true;
                            g.aux = {ce.auxiliary->op, compile_expr(ce.auxiliary->expr)};
                        }
                        c.num_order.push_back({false, static_cast<u32>(c.uncond_num.size())});
                        c.uncond_num.push_back(std::move(g));
                    }
                    continue;
                }
                const u32 total = c.arity + extra;
                c.bind_size = std::max(c.bind_size, total);
                std::vector<u8> rel(total, 1);
                for (u32 v = c.arity; v < total; ++v)
                    rel[v] = numeric ? 1 : 0;  // every firing of a numeric (non-idempotent) effect counts
                for (const Literal& l : effs)
                    for (Term x : T.terms_of(l))
                        if (x >= 0)
                            rel[x] = 1;
                plan::CondEffect cce;
                cce.cond = matcher(T.literals_of(ce.condition), total, c.arity, rel, T.constraints_of(ce.condition));
                for (const Literal& l : effs)
                    (l.positive ? cce.adds : cce.dels).push_back(pattern(l));
                cce.extras = extra > 0;
                if (extra > 0 && !numeric && !cce.cond.never)
                    choose_driver(cce, T.literals_of(ce.condition), c.arity, total, rel);
                if (numeric)
                {
                    cce.numeric = true;
                    for (const NumericEffect& e : neffs)
                        cce.neffs.push_back(compile_effect(e));
                    if (ce.auxiliary)
                    {
                        cce.has_aux = true;
                        cce.aux = {ce.auxiliary->op, compile_expr(ce.auxiliary->expr)};
                    }
                    c.num_order.push_back({true, static_cast<u32>(c.ces.size())});
                }
                c.ces.push_back(std::move(cce));
                C.has_conditional_effects = true;
            }
            C.max_bind = std::max(C.max_bind, c.bind_size);
            C.schemas.push_back(std::move(c));
        }
    }

    /// Atom-driven enumeration of a conditional effect's forall bindings (plan::CondEffect::driver): the first positive
    /// fluent condition literal of arity >= 3 mentioning every forall parameter, if every forall parameter is effect-
    /// relevant (each binding fires once, as with the matcher, which then emits no witness-pruned duplicates either).
    /// Unary and binary literals stay with the matcher: its view rows already restrict the candidates to true atoms,
    /// while a literal of arity >= 3 is only tested once all its parameters are bound (the object product).
    void choose_driver(plan::CondEffect& cce, std::span<const Literal> cond, u32 arity, u32 total, const std::vector<u8>& rel)
    {
        for (u32 v = arity; v < total; ++v)
            if (!rel[v])
                return;
        for (const Literal& l : cond)
        {
            if (!l.positive || C.kinds[l.pred.v] != PredKind::Fluent || T.terms_of(l).size() < 3)
                continue;
            const auto terms = T.terms_of(l);
            bool covers = true;
            for (u32 v = arity; v < total && covers; ++v)
                covers = std::find(terms.begin(), terms.end(), static_cast<Term>(v)) != terms.end();
            if (!covers)
                continue;
            u32 id = 0;
            while (id < C.drivers.size() && C.drivers[id] != l.pred.v)
                ++id;
            if (id == C.drivers.size())
                C.drivers.push_back(l.pred.v);
            cce.driver = id;
            cce.driver_terms.assign(terms.begin(), terms.end());
            for (const Literal& x : cond)
                cce.lits.push_back(check(x));
            return;
        }
    }

    void compile_axioms()
    {
        const auto& xs = T.axioms;
        if (xs.empty())
            return;
        const u32 m = static_cast<u32>(T.predicates.size());
        std::vector<u32> rank(m, 0);
        for (bool changed = true; changed;)
        {
            changed = false;
            for (const auto& x : xs)
                for (const Literal& l : T.literals_of(x.body))
                    if (C.kinds[l.pred.v] == PredKind::Derived)
                    {
                        const u32 need = rank[l.pred.v] + (l.positive ? 0 : 1);
                        if (rank[x.head.pred.v] < need)
                        {
                            rank[x.head.pred.v] = need;
                            changed = true;
                            if (need > m)
                                throw std::invalid_argument("mymyr: axioms are not stratifiable");
                        }
                    }
        }
        u32 top = 0;
        for (const auto& x : xs)
            top = std::max(top, rank[x.head.pred.v]);
        C.strata.resize(top + 1);
        for (const auto& x : xs)
        {
            const u32 arity = x.params.count;
            std::vector<u8> rel(arity, 0);
            for (Term v : T.terms_of(x.head))
                if (v >= 0)
                    rel[v] = 1;
            C.strata[rank[x.head.pred.v]].axioms.push_back({pattern(x.head), matcher(T.literals_of(x.body), arity, 0, rel, T.constraints_of(x.body))});
            C.max_bind = std::max(C.max_bind, arity);
        }
        for (const auto& x : xs)
            for (const Literal& l : T.literals_of(x.body))
                if (C.kinds[l.pred.v] == PredKind::Derived && rank[l.pred.v] == rank[x.head.pred.v])
                    C.strata[rank[x.head.pred.v]].recursive = true;
        // drop empty strata (ranks without axioms)
        std::erase_if(C.strata, [](const plan::Stratum& s) { return s.axioms.empty(); });
    }

    void compile_goal()
    {
        for (const Literal& l : T.literals_of(T.goal))
        {
            if (C.kinds[l.pred.v] == PredKind::Static)
            {
                if (static_ground_holds(l) != l.positive)
                    C.goal.unsatisfiable = true;
                continue;
            }
            C.goal.lits.push_back(check(l));
            C.goal.uses_derived |= C.kinds[l.pred.v] == PredKind::Derived;
        }
        for (const NumericConstraint& c : T.constraints_of(T.goal))
            N.goal.push_back(compile_check(c));
    }

    // ------------------------------------------------------------------------------------------------ layout
    void build_layout()
    {
        const u32 m = static_cast<u32>(T.predicates.size());
        std::vector<std::vector<std::vector<u64>>> bits(m);
        for (u32 p = 0; p < m; ++p)
            if (C.kinds[p] != PredKind::Static)
                bits[p].assign(C.arity[p], std::vector<u64>(OW, 0));
        auto add_term = [&](u32 p, u32 i, Term t, const plan::Matcher& mt)
        {
            if (t < 0)
                bits::set(bits[p][i].data(), term_object(t).v);
            else
                for (u32 w = 0; w < OW; ++w)
                    bits[p][i][w] |= mt.dom0[static_cast<usize>(t) * OW + w];
        };
        auto add_pattern = [&](const plan::Pattern& pt, auto&& matcher_of)
        {
            const auto terms = TaskData::slice(T.terms, pt.terms);
            for (u32 i = 0; i < terms.size(); ++i)
                add_term(pt.pred, i, terms[i], matcher_of(terms[i]));
        };
        for (const plan::Schema& s : C.schemas)
        {
            for (const auto& a : s.adds)
                add_pattern(a, [&](Term) -> const plan::Matcher& { return s.pre[0]; });
            for (const auto& ce : s.ces)
                for (const auto& a : ce.adds)
                    add_pattern(a, [&](Term t) -> const plan::Matcher&
                                { return (t < 0 || static_cast<u32>(t) < s.arity) ? s.pre[0] : ce.cond; });
        }
        for (const auto& st : C.strata)
            for (const auto& x : st.axioms)
                add_pattern(x.head, [&](Term) -> const plan::Matcher& { return x.body; });
        for (const auto& a : T.fluent_init)
        {
            auto objs = T.objects_of(a);
            for (u32 i = 0; i < objs.size(); ++i)
                bits::set(bits[a.pred.v][i].data(), objs[i].v);
        }

        CanonicalLayout& L = C.layout;
        L.num_objects = n;
        L.arity = C.arity;
        L.offset.assign(m, CanonicalLayout::k_none);
        L.size.assign(m, 0);
        L.pos_begin.assign(m, 0);
        u64 running = 0, positions = 0;
        std::vector<u32> preds;
        for (PredKind k : {PredKind::Fluent, PredKind::Derived})
        {
            for (u32 p = 0; p < m; ++p)
                if (C.kinds[p] == k)
                    preds.push_back(p);
        }
        for (u32 p : preds)
        {
            L.pos_begin[p] = positions;
            positions += C.arity[p];
            L.max_arity = std::max(L.max_arity, C.arity[p]);
        }
        L.rs.assign(positions * n, CanonicalLayout::k_outside);
        L.stride.assign(positions, 0);
        L.domain_size.assign(positions, 0);
        L.domain_begin.assign(positions, 0);
        for (u32 p : preds)
        {
            const u32 ar = C.arity[p];
            u64 size = 1;
            for (u32 i = 0; i < ar; ++i)
            {
                const u64 pos = L.pos_begin[p] + i;
                L.domain_begin[pos] = L.domain.size();
                for (u32 o = 0; o < n; ++o)
                    if (bits::test(bits[p][i].data(), OW, o))
                        L.domain.push_back(o);
                L.domain_size[pos] = static_cast<u32>(L.domain.size() - L.domain_begin[pos]);
                size = sat_mul(size, L.domain_size[pos]);
            }
            u64 st = 1;
            for (u32 i = ar; i-- > 0;)
            {
                const u64 pos = L.pos_begin[p] + i;
                L.stride[pos] = st;
                for (u32 r = 0; r < L.domain_size[pos]; ++r)
                    L.rs[pos * n + L.domain[L.domain_begin[pos] + r]] = r * st;
                st = sat_mul(st, std::max<u32>(1, L.domain_size[pos]));
            }
            L.size[p] = size;
            L.offset[p] = running;
            running = std::min<u64>(running + size, u64{1} << 62);
            if (C.kinds[p] == PredKind::Fluent)
                L.fluent_count = running;  // fluent predicates come first: the end of the fluent block
        }
        L.total = running;
        for (u32 p : preds)
            if (L.size[p] > 0)
                L.by_offset.push_back(p);
    }

    void resolve(plan::Pattern& pt)
    {
        const auto terms = TaskData::slice(T.terms, pt.terms);
        pt.var_begin = static_cast<u32>(C.pattern_vars.size());
        pt.var_count = 0;
        pt.base = 0;
        if (pt.kind == plan::LitKind::Static)
        {
            const plan::StaticRelation& R = C.statics[pt.pred];
            for (u32 i = 0; i < terms.size(); ++i)
            {
                const u64* tab = R.position_table(i, n);
                if (terms[i] < 0)
                    pt.base += tab[term_object(terms[i]).v];
                else
                    C.pattern_vars.push_back({static_cast<u32>(terms[i]), tab}), ++pt.var_count;
            }
            return;
        }
        const CanonicalLayout& L = C.layout;
        pt.base = L.offset[pt.pred];
        for (u32 i = 0; i < terms.size(); ++i)
        {
            const u64* tab = L.position_table(pt.pred, i);
            if (terms[i] < 0)
                pt.base += tab[term_object(terms[i]).v];
            else
                C.pattern_vars.push_back({static_cast<u32>(terms[i]), tab}), ++pt.var_count;
        }
    }

    void resolve(plan::Matcher& m)
    {
        for (auto& c : m.pre_checks)
            resolve(c.pat);
        for (auto& c : m.checks)
            resolve(c.pat);
    }

    void resolve_patterns()
    {
        // pattern_vars holds raw pointers into C.layout.rs / C.statics[p].rs, which are final at this point
        for (auto& s : C.schemas)
        {
            resolve(s.pre[0]);
            resolve(s.pre[1]);
            for (auto* v : {&s.adds, &s.dels})
                for (auto& pt : *v)
                    resolve(pt);
            for (auto& c : s.pre_lits)
                resolve(c.pat);
            for (auto& ce : s.ces)
            {
                resolve(ce.cond);
                for (auto* v : {&ce.adds, &ce.dels})
                    for (auto& pt : *v)
                        resolve(pt);
                for (auto& c : ce.lits)
                    resolve(c.pat);
            }
        }
        for (auto& st : C.strata)
            for (auto& x : st.axioms)
            {
                resolve(x.head);
                resolve(x.body);
            }
        for (auto& c : C.goal.lits)
            resolve(c.pat);
    }

    const TaskData& T;
    const TaskOptions& O;
    plan::Compiled& C;
    u32 n = 0, OW = 1;
    std::vector<std::vector<u64>> s_unary;
    std::vector<std::vector<const GroundAtom*>> static_atoms;
    std::map<std::tuple<u32, u32, u32>, u64> row_tables;
    plan::Numeric& N = C.num;
    std::unordered_map<std::string, plan::NumProg> m_programs;  // compile_expr: code key -> program
    std::vector<u8> static_integral;  // per function: every static value is an integer
};
}  // namespace

void compile_task(const TaskData& t, const TaskOptions& options, plan::Compiled& out)
{
    Compiler(t, options, out).run();
}
}  // namespace mymyr::detail
