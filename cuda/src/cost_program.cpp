// The device action costs' programs (mymyr/cuda/cost_program.hpp, cost_program_build.hpp).

#include "cost_program_build.hpp"

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace mymyr::cuda::costs
{
using namespace formalism;

namespace
{
void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("mymyr: device action costs: ") + what + ": " + cudaGetErrorString(e));
}

struct Host
{
    std::vector<u32> begin{0};
    std::vector<Ins> code;
    std::vector<f64> nums;
    std::vector<u32> terms;
    std::vector<u64> func_base;
    std::vector<u32> objects;
    std::vector<u64> table;
    std::vector<u64> keys;
    std::vector<f64> values;
};

/// Appends the postfix program of expression e to `code` (its numbers and terms to h); returns the operands it holds at
/// once. Of a binary operation's operands the one that needs more is evaluated first (a reversed operation for - and /;
/// + and * commute exactly in IEEE arithmetic), so a program holds at most log2(leaves) + 1 operands.
u32 compile(const TaskData& T, u32 e, Host& h, std::vector<Ins>& code)
{
    const Expr& x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number:
            code.push_back({k_op_num, static_cast<u32>(h.nums.size()), 0, 0});
            h.nums.push_back(x.value);
            return 1;
        case ExprOp::Function:
        {
            const auto terms = TaskData::slice(T.terms, x.terms);
            if (terms.size() > k_max_arity)
            {
                // (ActionCosts::evaluate: undefined)
                code.push_back({k_op_num, static_cast<u32>(h.nums.size()), 0, 0});
                h.nums.push_back(std::numeric_limits<f64>::quiet_NaN());
                return 1;
            }
            code.push_back({k_op_fn, static_cast<u32>(h.terms.size()), x.func.v, static_cast<u32>(terms.size())});
            for (const Term t : terms)
                h.terms.push_back(is_object(t) ? term_object(t).v : (k_param | term_parameter(t)));
            return 1;
        }
        case ExprOp::Neg:
        {
            const u32 d = compile(T, x.a, h, code);
            code.push_back({k_op_neg, 0, 0, 0});
            return d;
        }
        default:
        {
            std::vector<Ins> ca, cb;
            const u32 da = compile(T, x.a, h, ca);
            const u32 db = compile(T, x.b, h, cb);
            const bool reversed = db > da;
            const std::vector<Ins>& first = reversed ? cb : ca;
            const std::vector<Ins>& second = reversed ? ca : cb;
            code.insert(code.end(), first.begin(), first.end());
            code.insert(code.end(), second.begin(), second.end());
            u32 op = k_op_add;
            switch (x.op)
            {
                case ExprOp::Add: op = k_op_add; break;
                case ExprOp::Mul: op = k_op_mul; break;
                case ExprOp::Sub: op = reversed ? k_op_rsub : k_op_sub; break;
                default: op = reversed ? k_op_rdiv : k_op_div; break;  // ExprOp::Div
            }
            code.push_back({op, 0, 0, 0});
            return da == db ? da + 1 : std::max(da, db);
        }
    }
}

/// The instance's function keys (ActionCosts::function_key: the first key of each function, then the mixed radix over
/// max(1, objects)) and its static values (none for unit costs) in a hash table. ActionCosts' constructor has checked
/// the key space.
void add_values(const TaskData& T, bool unit, Host& h)
{
    const u32 objects = T.num_objects();
    const u64 n = std::max<u32>(1, objects);
    std::vector<u64> base(T.functions.size() + 1, 0);
    for (usize f = 0; !unit && f < T.functions.size(); ++f)
    {
        u64 size = 1;
        for (u32 i = 0; i < T.functions[f].arity; ++i)
            size *= n;
        base[f + 1] = base[f] + size;
    }
    h.func_base.insert(h.func_base.end(), base.begin(), base.end());
    h.objects.push_back(objects);
    const usize count = unit ? 0 : T.static_values.size();
    const u64 slots = std::bit_ceil(std::max<u64>(2 * count, 2));
    const u64 first = h.keys.size();
    h.table.push_back(first);
    h.table.push_back(slots - 1);
    h.keys.resize(first + slots, 0);
    h.values.resize(first + slots, 0);
    for (const GroundFunctionValue& v : std::span(T.static_values).first(count))
    {
        const auto objs = TaskData::slice(T.object_ids, v.objects);
        u64 key = 0, mul = 1;
        bool inside = v.func.v < T.functions.size();
        for (usize i = 0; inside && i < objs.size(); ++i)
        {
            inside = objs[i].v < objects;
            key += mul * objs[i].v;
            mul *= n;
        }
        if (!inside)
            continue;  // (ActionCosts::function_key fails: never looked up)
        key += base[v.func.v];
        for (u64 j = hash_key(key) & (slots - 1);; j = (j + 1) & (slots - 1))
        {
            u64& k = h.keys[first + j];
            if (k == 0 || k == key + 1)
            {
                k = key + 1;
                h.values[first + j] = v.value;  // (the last of equal keys, as ActionCosts' map)
                break;
            }
        }
    }
}

template<class T>
void put(std::vector<u8>& bytes, const std::vector<T>& v, u64& at)
{
    at = (bytes.size() + 7) & ~u64{7};
    bytes.resize(at + v.size() * sizeof(T));
    if (!v.empty())
        std::memcpy(bytes.data() + at, v.data(), v.size() * sizeof(T));
}
}  // namespace

Programs build(const ContextPtr& ctx, std::span<const Task* const> tasks,
               std::span<const heuristics::ActionCosts* const> costs, cudaStream_t s)
{
    Programs out;
    if (tasks.empty() || tasks.size() != costs.size())
        throw std::invalid_argument("mymyr: device action costs: one ActionCosts per task");
    const u32 S = static_cast<u32>(tasks.front()->data().schemas.size());
    const u32 F = static_cast<u32>(tasks.front()->data().functions.size());
    Host h;
    for (usize i = 0; i < tasks.size(); ++i)
    {
        const TaskData& T = tasks[i]->data();
        const heuristics::ActionCosts& c = *costs[i];
        if (T.schemas.size() != S || T.functions.size() != F)
            throw std::invalid_argument("mymyr: device action costs: the instances' schemas or functions differ");
        if (!c.unit() && !(c.kind() == heuristics::ActionCosts::Kind::TotalCost && c.state_independent()))
        {
            out.why = "action costs that depend on the state";
            return out;
        }
        out.integral = out.integral && c.integral();
        add_values(T, c.unit(), h);
        for (u32 x = 0; x < S; ++x)
        {
            if (c.unit() || c.constant(x))
            {
                h.code.push_back({k_op_num, static_cast<u32>(h.nums.size()), 0, 0});
                h.nums.push_back(c.unit() ? 1.0 : c.cost(x, nullptr));
            }
            else
            {
                // the schema's one unconditional increase of total-cost (ActionCosts: Cost::Expr)
                u32 expr = ~u32{0};
                for (const ConditionalEffect& ce : T.effects_of(T.schemas[x]))
                    if (ce.auxiliary)
                        expr = ce.auxiliary->expr;
                if (expr == ~u32{0})
                    throw std::logic_error("mymyr: device action costs: a schema's cost expression is missing "
                                           "(internal error)");
                if (compile(T, expr, h, h.code) > k_stack)
                    throw std::length_error("mymyr: device action costs: an expression of more than 2^31 operands");
            }
            h.begin.push_back(static_cast<u32>(h.code.size()));
        }
    }
    // one allocation for every array
    std::vector<u8> bytes;
    u64 at_begin = 0, at_code = 0, at_nums = 0, at_terms = 0, at_base = 0, at_objects = 0, at_table = 0, at_keys = 0,
        at_values = 0;
    put(bytes, h.begin, at_begin);
    put(bytes, h.code, at_code);
    put(bytes, h.nums, at_nums);
    put(bytes, h.terms, at_terms);
    put(bytes, h.func_base, at_base);
    put(bytes, h.objects, at_objects);
    put(bytes, h.table, at_table);
    put(bytes, h.keys, at_keys);
    put(bytes, h.values, at_values);
    out.buf = DeviceBuffer(ctx, std::max<u64>(bytes.size(), 8), s);
    check(cudaMemcpyAsync(out.buf.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // (the host bytes go)
    auto* d = static_cast<const u8*>(out.buf.data());
    Program& p = out.view;
    p.begin = reinterpret_cast<const u32*>(d + at_begin);
    p.code = reinterpret_cast<const Ins*>(d + at_code);
    p.nums = reinterpret_cast<const f64*>(d + at_nums);
    p.terms = reinterpret_cast<const u32*>(d + at_terms);
    p.func_base = reinterpret_cast<const u64*>(d + at_base);
    p.objects = reinterpret_cast<const u32*>(d + at_objects);
    p.table = reinterpret_cast<const u64*>(d + at_table);
    p.keys = reinterpret_cast<const u64*>(d + at_keys);
    p.values = reinterpret_cast<const f64*>(d + at_values);
    p.num_schemas = S;
    p.num_functions = F;
    return out;
}
}  // namespace mymyr::cuda::costs
