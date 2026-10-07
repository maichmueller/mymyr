#pragma once
// Offset-addressed numeric programs of a device task. State values are canonical IEEE-754 doubles, one per fluent
// slot. Programs preserve the CPU's postfix evaluation order; function tables map lifted arguments to slots or
// static values. All pointers address the same immutable task block.

#include "mymyr/core/types.hpp"

#include <bit>
#include <cmath>

namespace mymyr::rl::dev
{
struct NumericView
{
    u32 slots = 0;
    u32 storage = 0;  // CPU encoding: 0 double, 1 packed int32
    u32 tolerant = 0;
    f64 quantum = 0;
    const u32* code = nullptr;     // [instructions, 4]: op, arity, a, b
    const f64* consts = nullptr;
    const i32* terms = nullptr;
    const u64* table = nullptr;    // [functions, 8]: size, rs, dense, dense count, hash keys, hash values, mask, fluent
    const u64* rs = nullptr;
    const u64* payload = nullptr;  // static double bits or fluent slot ids; hashed keys share this array
    const u32* checks = nullptr;   // [checks, 5]: comparator, lhs begin/end, rhs begin/end
    const u32* matcher = nullptr;  // [matchers, 8]: pre begin/count, checks begin/count, steps, masks, unused, unused
    const u32* steps = nullptr;    // [plan steps, 2]: numeric check indices begin/end
    const u32* index = nullptr;
    const u32* masks = nullptr;    // per numeric check: its free parameters as bits
    const u32* effects = nullptr;  // [effects, 8]: op, slot, lifted, table, terms, arity, program begin/end
    const u32* groups = nullptr;   // [groups, 5]: effect begin/count, aux op (0xFFFFFFFF absent), program begin/end
    const u32* schema = nullptr;   // [schemas, 2]: ordered groups begin/count
    const u32* order = nullptr;    // [ordered groups, 2]: conditional (0/1), group or conditional-effect index
    const u32* ce = nullptr;       // [conditional effects, 3]: group, numeric, extras
    const f64* initial = nullptr;
    u32 goal_begin = 0, goal_count = 0;
};

[[nodiscard]] MYMYR_HD f64 numeric_double(u64 bits)
{
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    return std::bit_cast<f64>(bits);
#endif
}

[[nodiscard]] MYMYR_HD u64 numeric_bits(f64 value)
{
#if defined(__CUDA_ARCH__)
    return static_cast<u64>(__double_as_longlong(value));
#else
    return std::bit_cast<u64>(value);
#endif
}

[[nodiscard]] MYMYR_HD f64 numeric_nan() { return numeric_double(0x7FF8000000000000ULL); }
[[nodiscard]] MYMYR_HD f64 numeric_value(const u64* values, u32 slot) { return numeric_double(values[slot]); }

[[nodiscard]] MYMYR_HD u64 numeric_key(const NumericView& n, u32 table, u32 off, u32 arity,
                                             u32 objects, const u32* bind)
{
    const u64* t = n.table + u64{table} * 8;
    u64 key = 0;
    for (u32 i = 0; i < arity; ++i)
    {
        const i32 term = n.terms[off + i];
        const u32 object = term >= 0 ? bind[term] : static_cast<u32>(-1 - term);
        key += n.rs[t[1] + u64{i} * objects + object];
    }
    return key;
}

[[nodiscard]] MYMYR_HD u64 numeric_lookup(const NumericView& n, u32 table, u64 key)
{
    const u64* t = n.table + u64{table} * 8;
    if (key >= t[0])
        return t[7] ? 0xFFFFFFFFu : numeric_bits(numeric_nan());
    if (t[3])
        return n.payload[t[2] + key];
    if (t[4] == ~u64{0})
        return t[7] ? 0xFFFFFFFFu : numeric_bits(numeric_nan());
    u64 h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    for (u64 j = h & t[6];; j = (j + 1) & t[6])
    {
        const u64 k = n.payload[t[4] + j];
        if (k == key + 1)
            return n.payload[t[5] + j];
        if (!k)
            return t[7] ? 0xFFFFFFFFu : numeric_bits(numeric_nan());
    }
}

[[nodiscard]] MYMYR_HD f64 numeric_eval(const NumericView& n, u32 begin, u32 end, const u64* values,
                                               const u32* bind, u32 objects)
{
    f64 stack[64];
    u32 sp = 0;
    for (u32 i = begin; i < end; ++i)
    {
        const u32* in = n.code + u64{i} * 4;
        switch (in[0])
        {
            case 0: stack[sp++] = n.consts[in[2]]; break;
            case 1:
            case 2: stack[sp++] = numeric_value(values, in[2]); break;
            case 3:
            case 4:
            case 5:
            {
                const u64 v = numeric_lookup(n, in[2], numeric_key(n, in[2], in[3], in[1], objects, bind));
                stack[sp++] = in[0] == 5 ? numeric_double(v) : v == 0xFFFFFFFFu ? numeric_nan()
                                                                                  : numeric_value(values, static_cast<u32>(v));
                break;
            }
            case 6: --sp; stack[sp - 1] += stack[sp]; break;
            case 7: --sp; stack[sp - 1] -= stack[sp]; break;
            case 8: --sp; stack[sp - 1] *= stack[sp]; break;
            case 9: --sp; stack[sp - 1] = stack[sp] == 0 ? numeric_nan() : stack[sp - 1] / stack[sp]; break;
            case 10: stack[sp - 1] = -stack[sp - 1]; break;
        }
        if (n.tolerant && in[0] >= 6 && !std::isfinite(stack[sp - 1]))
            stack[sp - 1] = numeric_nan();
    }
    return stack[0];
}

[[nodiscard]] MYMYR_HD bool numeric_holds(const NumericView& n, u32 check, const u64* values,
                                                 const u32* bind, u32 objects)
{
    const u32* c = n.checks + u64{check} * 5;
    const f64 l = numeric_eval(n, c[1], c[2], values, bind, objects);
    const f64 r = numeric_eval(n, c[3], c[4], values, bind, objects);
    if (std::isnan(l) || std::isnan(r))
        return false;
    const f64 e = n.tolerant ? 1e-9 : 0;
    switch (c[0])
    {
        case 0: return n.tolerant ? std::fabs(l - r) <= e : l == r;
        case 1: return n.tolerant ? std::fabs(l - r) > e : l != r;
        case 2: return l < r - e;
        case 3: return l <= r + e;
        case 4: return l > r + e;
        case 5: return l >= r - e;
    }
    return false;
}

[[nodiscard]] MYMYR_HD f64 numeric_canonical(const NumericView& n, f64 value)
{
    if (n.tolerant)
    {
        if (!std::isfinite(value))
            return numeric_nan();
        if (std::fabs(value) < 1e6)
            value = std::nearbyint(value * 1e9) / 1e9;
    }
    if (n.quantum > 0 && std::isfinite(value))
        value = std::nearbyint(value / n.quantum) * n.quantum;
    return value == 0 ? 0.0 : std::isnan(value) ? numeric_nan() : value;
}

[[nodiscard]] MYMYR_HD f64 numeric_assign(u32 op, f64 old, f64 value)
{
    switch (op)
    {
        case 0: return value;
        case 1: return old + value;
        case 2: return old - value;
        case 3: return old * value;
        case 4: return old / value;
    }
    return value;
}
}  // namespace mymyr::rl::dev
