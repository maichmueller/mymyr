#pragma once
// Device action costs: the state-independent cost of an action (heuristics::ActionCosts::cost) evaluated on
// the device from its binding, as a postfix program per (instance, schema) over numbers and static function values,
// with each instance's static values in a hash table. Any number of cost parameters, any key space (the tables over
// the cost parameters' objects they replace held at most 4 parameters and 2^24 entries). The device state space
// (cuda/state_space.hpp) and the best-first searches (cuda/astar.hpp, cuda/gbfs.hpp) share them; the builder is
// cuda/src/cost_program_build.hpp.
//
// Semantics: ActionCosts::evaluate's, operation by operation in IEEE double precision (round to nearest, no
// contraction): a static function value that is undefined (missing, an argument beyond the objects, more than 64
// arguments) is NaN, so is a division by zero, and NaN propagates. evaluate() returns the value as is; the callers decide about NaN and negative
// costs (ActionCosts::cost throws std::domain_error for both). Unit instances cost 1.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20, and by the host builder.

#include "mymyr/core/types.hpp"

#include <limits>

namespace mymyr::cuda::costs
{
enum : u32
{
    k_op_num = 0,  // push nums[a]
    k_op_fn = 1,   // push static function b at the terms[a .. a + ar)
    k_op_add = 2,
    k_op_sub = 3,
    k_op_mul = 4,
    k_op_div = 5,
    k_op_neg = 6,
    k_op_rsub = 7,  // b - a for the operands a (below) and b (top): the builder pushes the deeper operand first
    k_op_rdiv = 8,
};
inline constexpr u32 k_param = 0x80000000u;  // a term that is a parameter (the low bits), else an object id
inline constexpr u32 k_stack = 32;          // operands held at once, at most (deeper first: log2(leaves) + 1)
inline constexpr u32 k_max_arity = 64;      // function arguments at most (ActionCosts::evaluate: more is undefined)

struct Ins
{
    u32 op = 0;
    u32 a = 0;
    u32 b = 0;
    u32 ar = 0;
};

/// The device view of the programs of I instances of one domain (S schemas, F functions).
struct Program
{
    const u32* begin = nullptr;      // [I * S + 1] the instructions of (instance, schema): code[begin[i * S + s] ..)
    const Ins* code = nullptr;
    const f64* nums = nullptr;
    const u32* terms = nullptr;
    const u64* func_base = nullptr;  // [I * (F + 1)] the first key of each function
    const u32* objects = nullptr;    // [I] the instance's objects (the key radix is max(1, objects))
    const u64* table = nullptr;      // [I * 2] the instance's hash table: first slot, slot mask
    const u64* keys = nullptr;       // key + 1 (0: an empty slot)
    const f64* values = nullptr;
    u32 num_schemas = 0;
    u32 num_functions = 0;
};

/// The hash of a static function key (the murmur3 finalizer).
MYMYR_HD u64 hash_key(u64 k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

#if defined(__CUDA_ARCH__)
MYMYR_HD f64 nan_value() { return __longlong_as_double(0x7FF8000000000000ll); }
MYMYR_HD f64 add(f64 a, f64 b) { return __dadd_rn(a, b); }
MYMYR_HD f64 sub(f64 a, f64 b) { return __dsub_rn(a, b); }
MYMYR_HD f64 mul(f64 a, f64 b) { return __dmul_rn(a, b); }
MYMYR_HD f64 div(f64 a, f64 b) { return b == 0 ? nan_value() : __ddiv_rn(a, b); }
#else
MYMYR_HD f64 nan_value() { return std::numeric_limits<f64>::quiet_NaN(); }
MYMYR_HD f64 add(f64 a, f64 b) { return a + b; }
MYMYR_HD f64 sub(f64 a, f64 b) { return a - b; }
MYMYR_HD f64 mul(f64 a, f64 b) { return a * b; }
MYMYR_HD f64 div(f64 a, f64 b) { return b == 0 ? nan_value() : a / b; }
#endif

/// The value of static function f of instance i at `args` (ActionCosts::function_key and its map; NaN: undefined).
MYMYR_HD f64 lookup(const Program& p, u32 i, u32 f, const u32* args, u32 arity)
{
    const u32 n = p.objects[i];
    u64 key = 0, radix = 1;
    for (u32 k = 0; k < arity; ++k)
    {
        if (args[k] >= n)
            return nan_value();
        key += radix * args[k];
        radix *= n;  // (n >= 1 here)
    }
    key += p.func_base[u64{i} * (p.num_functions + 1) + f];
    const u64 first = p.table[u64{i} * 2], mask = p.table[u64{i} * 2 + 1];
    for (u64 j = hash_key(key) & mask;; j = (j + 1) & mask)
    {
        const u64 x = p.keys[first + j];
        if (x == key + 1)
            return p.values[first + j];
        if (x == 0)
            return nan_value();
    }
}

/// The cost of action (schema s, binding) of instance i: ActionCosts::evaluate's value (NaN: undefined).
MYMYR_HD f64 evaluate(const Program& p, u32 i, u32 s, const u32* binding)
{
    f64 st[k_stack];
    u32 sp = 0;
    const u64 at = u64{i} * p.num_schemas + s;
    for (u32 k = p.begin[at]; k < p.begin[at + 1]; ++k)
    {
        const Ins x = p.code[k];
        switch (x.op)
        {
            case k_op_num: st[sp++] = p.nums[x.a]; break;
            case k_op_fn:
            {
                u32 args[k_max_arity];
                for (u32 j = 0; j < x.ar; ++j)
                {
                    const u32 t = p.terms[x.a + j];
                    args[j] = t & k_param ? binding[t & ~k_param] : t;
                }
                st[sp++] = lookup(p, i, x.b, args, x.ar);
                break;
            }
            case k_op_add: --sp; st[sp - 1] = add(st[sp - 1], st[sp]); break;
            case k_op_sub: --sp; st[sp - 1] = sub(st[sp - 1], st[sp]); break;
            case k_op_mul: --sp; st[sp - 1] = mul(st[sp - 1], st[sp]); break;
            case k_op_div: --sp; st[sp - 1] = div(st[sp - 1], st[sp]); break;
            case k_op_rsub: --sp; st[sp - 1] = sub(st[sp], st[sp - 1]); break;
            case k_op_rdiv: --sp; st[sp - 1] = div(st[sp], st[sp - 1]); break;
            default: st[sp - 1] = -st[sp - 1]; break;  // k_op_neg
        }
    }
    return st[0];
}
}  // namespace mymyr::cuda::costs
