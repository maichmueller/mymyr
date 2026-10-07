#pragma once
// Internal: evaluation of the flat expression trees of ConjunctiveCondition and GroundCondition (formalism::Expr) on a
// state's numeric words, with the variables of a Function node's terms read from a binding.

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/task/numeric.hpp"

#include <cmath>
#include <limits>
#include <span>

namespace mymyr::detail
{
/// Value of expression e (mimir's semantics, as plan::eval): an undefined function value and a division by zero are
/// NaN; with tolerant semantics a non-finite intermediate is NaN. `bind` holds the object of each variable (nullptr
/// for ground expressions); `num` the state's numeric words.
[[nodiscard]] inline f64 eval_expr(const plan::Numeric& N, std::span<const formalism::Expr> exprs,
                                   std::span<const formalism::Term> terms, u32 e, const u32* bind, const u64* num)
{
    using formalism::ExprOp;
    const formalism::Expr& x = exprs[e];
    f64 r = 0;
    switch (x.op)
    {
        case ExprOp::Number: return x.value;
        case ExprOp::Function:
        {
            const plan::FunctionTable& F = N.tables[x.func.v];
            u64 k = 0;
            for (u32 j = 0; j < x.terms.count; ++j)
            {
                const formalism::Term t = terms[x.terms.begin + j];
                const u32 o = formalism::is_object(t) ? formalism::term_object(t).v : bind[t];
                k += F.rs[static_cast<usize>(j) * N.num_objects + o];
            }
            if (!F.fluent)
                return F.value_of(k);
            const u32 s = F.slot_of(k);
            return s == plan::FunctionTable::k_none ? std::numeric_limits<f64>::quiet_NaN() : plan::load(N, num, s);
        }
        case ExprOp::Add: r = eval_expr(N, exprs, terms, x.a, bind, num) + eval_expr(N, exprs, terms, x.b, bind, num); break;
        case ExprOp::Sub: r = eval_expr(N, exprs, terms, x.a, bind, num) - eval_expr(N, exprs, terms, x.b, bind, num); break;
        case ExprOp::Mul: r = eval_expr(N, exprs, terms, x.a, bind, num) * eval_expr(N, exprs, terms, x.b, bind, num); break;
        case ExprOp::Div:
        {
            const f64 a = eval_expr(N, exprs, terms, x.a, bind, num), b = eval_expr(N, exprs, terms, x.b, bind, num);
            r = b == 0 ? std::numeric_limits<f64>::quiet_NaN() : a / b;
            break;
        }
        case ExprOp::Neg: r = -eval_expr(N, exprs, terms, x.a, bind, num); break;
    }
    if (N.tolerant && !std::isfinite(r))
        return std::numeric_limits<f64>::quiet_NaN();
    return r;
}
}  // namespace mymyr::detail
