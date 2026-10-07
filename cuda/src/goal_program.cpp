#include "goal_program.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/successor/conditions.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mymyr::cuda::detail
{
namespace
{
template<class T>
const T* upload(DeviceBuffer& buffer, const ContextPtr& ctx, const std::vector<T>& values, cudaStream_t stream)
{
    if (values.empty()) return nullptr;
    buffer = DeviceBuffer(ctx, values.size() * sizeof(T), stream);
    check(cudaMemcpyAsync(buffer.data(), values.data(), buffer.size(), cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync (goal programs)");
    return static_cast<const T*>(buffer.data());
}

struct Programs
{
    const plan::Numeric& numeric;
    std::vector<u32> code, checks;
    std::vector<f64> consts;
    u32 stack = 0;

    void emit(plan::NumOp op, u32 operand = 0)
    {
        code.insert(code.end(), {static_cast<u32>(op), 0, operand, 0});
        if (op == plan::NumOp::Const || op == plan::NumOp::SlotF64)
        {
            if (++stack > plan::Numeric::k_max_stack)
                throw std::invalid_argument("mymyr: device goal: numeric expressions need more than 64 stack values");
        }
        else if (op != plan::NumOp::Neg)
            --stack;
    }

    void constant(f64 value)
    {
        emit(plan::NumOp::Const, static_cast<u32>(consts.size()));
        consts.push_back(value);
    }

    void expression(const GroundCondition& condition, u32 id)
    {
        using formalism::ExprOp;
        const formalism::Expr& e = condition.exprs[id];
        switch (e.op)
        {
            case ExprOp::Number: constant(e.value); return;
            case ExprOp::Function:
            {
                const plan::FunctionTable& f = numeric.tables[e.func.v];
                u64 key = 0;
                for (u32 j = 0; j < e.terms.count; ++j)
                    key += f.rs[u64{j} * numeric.num_objects +
                                formalism::term_object(condition.expr_terms[e.terms.begin + j]).v];
                if (!f.fluent)
                    constant(f.value_of(key));
                else if (const u32 slot = f.slot_of(key); slot != plan::FunctionTable::k_none)
                    emit(plan::NumOp::SlotF64, slot);
                else
                    constant(std::numeric_limits<f64>::quiet_NaN());
                return;
            }
            case ExprOp::Neg: expression(condition, e.a); emit(plan::NumOp::Neg); return;
            case ExprOp::Add:
            case ExprOp::Sub:
            case ExprOp::Mul:
            case ExprOp::Div:
                expression(condition, e.a);
                expression(condition, e.b);
                emit(e.op == ExprOp::Add ? plan::NumOp::Add : e.op == ExprOp::Sub ? plan::NumOp::Sub :
                     e.op == ExprOp::Mul ? plan::NumOp::Mul : plan::NumOp::Div);
                return;
        }
    }

    plan::NumProg program(const GroundCondition& condition, u32 id)
    {
        stack = 0;
        const u32 begin = static_cast<u32>(code.size() / 4);
        expression(condition, id);
        return {begin, static_cast<u32>(code.size() / 4)};
    }
};
}  // namespace

GoalPrograms::GoalPrograms(const ContextPtr& ctx, const Task& task, std::span<const search::GoalSpec::AtomGoal> goals,
                           cudaStream_t stream)
{
    const AtomIndex& atoms = task.atoms();
    const auto& layout = atoms.layout();
    u32 words = 1, derived_words = 0;
    for (const auto& g : goals)
    {
        auto fluent = [&](std::span<const SlotId> slots)
        {
            for (SlotId x : slots)
            {
                if (x.v >= atoms.fluent_slots())
                    throw std::invalid_argument("mymyr: device goal: a slot is not a fluent atom slot of the task");
                words = std::max(words, bits::word_of(x.v) + 1);
            }
        };
        fluent(g.positive); fluent(g.negative);
        auto derived = [&](std::span<const CanonicalAtom> ids)
        {
            for (CanonicalAtom id : ids)
            {
                if (id < layout.fluent_count || id >= layout.total)
                    throw std::invalid_argument("mymyr: device goal: a canonical id is not a derived atom of the task");
                derived_words = std::max(derived_words, bits::word_of(atoms.intern(id)) + 1);
            }
        };
        derived(g.derived_positive); derived(g.derived_negative);
        if (g.numeric) g.numeric->validate(task);
    }
    std::vector<u64> positive(goals.size() * words), negative(positive.size()),
                     derived_positive(goals.size() * derived_words), derived_negative(derived_positive.size());
    std::vector<u32> offsets{0};
    Programs programs{task.compiled().num, {}, {}, {}};
    for (u32 i = 0; i < goals.size(); ++i)
    {
        const auto& g = goals[i];
        for (SlotId x : g.positive) bits::set(positive.data() + u64{i} * words, x.v);
        for (SlotId x : g.negative) bits::set(negative.data() + u64{i} * words, x.v);
        for (CanonicalAtom id : g.derived_positive)
            bits::set(derived_positive.data() + u64{i} * derived_words, atoms.find(id));
        for (CanonicalAtom id : g.derived_negative)
            bits::set(derived_negative.data() + u64{i} * derived_words, atoms.find(id));
        if (g.numeric)
            for (const formalism::NumericConstraint& c : g.numeric->constraints)
            {
                const auto lhs = programs.program(*g.numeric, c.lhs), rhs = programs.program(*g.numeric, c.rhs);
                programs.checks.insert(programs.checks.end(), {static_cast<u32>(c.cmp), lhs.begin, lhs.end, rhs.begin, rhs.end});
            }
        offsets.push_back(static_cast<u32>(programs.checks.size() / 5));
    }
    m_view.count = static_cast<u32>(goals.size());
    m_view.words = words;
    m_view.derived_words = derived_words;
    m_view.positive = upload(m_positive, ctx, positive, stream);
    m_view.negative = upload(m_negative, ctx, negative, stream);
    m_view.derived_positive = upload(m_derived_positive, ctx, derived_positive, stream);
    m_view.derived_negative = upload(m_derived_negative, ctx, derived_negative, stream);
    m_view.offsets = upload(m_offsets, ctx, offsets, stream);
    m_view.numeric.slots = task.numeric_slots();
    m_view.numeric.tolerant = task.compiled().num.tolerant;
    m_view.numeric.code = upload(m_code, ctx, programs.code, stream);
    m_view.numeric.consts = upload(m_consts, ctx, programs.consts, stream);
    m_view.numeric.checks = upload(m_checks, ctx, programs.checks, stream);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize (goal programs)");
}
}  // namespace mymyr::cuda::detail
