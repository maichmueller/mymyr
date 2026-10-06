#pragma once
// Numeric fluents.
//
// Slots. Every ground fluent function with a value in the initial state gets a numeric slot, in (function, arguments)
// order. A state carries one value per slot, in its *numeric words* after its atom bits: the logical row of a state
// is [bits | slots]. The storage type is fixed per task:
//   - I32 when the task is statically integral (integral initial values, integral constants and static values in
//     fluent effects, no division and no scale-down in fluent effects) and no quantization is set: two slots per word,
//     slot 2k in the low half of word k (little endian, as the state bits);
//   - F64 otherwise: one IEEE-754 double per word (its bit pattern).
//   Values are evaluated in double precision in either case (mimir's semantics); an I32 task whose value leaves the
//   int32 range or stops being integral throws std::overflow_error (choose TaskOptions::numeric_storage = F64).
// Canonical bit patterns (dedup compares and hashes words): -0 becomes +0 and every NaN one quiet NaN (mimir compares
// raw bit patterns, so it keeps -0 and +0 apart; no numeric task of the parity set produces -0). A slot value is never
// NaN in practice: an effect that would produce NaN makes its action inapplicable.
//
// Expressions compile to postfix bytecode with constant folding: numbers, static functions of constants and operators
// over constants become constants, fluent functions of constants become direct slot loads, the rest look up a
// per-function table (typed-dense over the objects that occur at each argument position; hashed above 2^24 keys).
// Semantics (mimir's):
//   - an undefined function value is NaN, so is a division by zero; a comparison with a NaN side is false;
//   - a binding is inapplicable when a numeric effect of a conditional effect that fires (or of the unconditional
//     effect) targets a function without a value, evaluates to NaN, or conflicts with an earlier effect of the same
//     action on the same target (assign with anything, additive with multiplicative); the effect families are recorded
//     in mimir's order: every conditional effect in turn, fluent effects then the total-cost effect, whether or not
//     it fires (a failure only matters if it fires). total-cost effects obey the same rules (NaN, families);
//   - effect values are evaluated on the parent state and applied in order to a copy.
// Deviation: mimir also lets `assign` give a value to a function that has none (its states grow a numeric variable);
// here such an action is inapplicable, because the slots are fixed per task (the ground functions with an initial
// value), so that every state row of a task has the same width. No task of the parity set does this.
// Options (TaskOptions): `numeric_quantum` q > 0 snaps every stored value to the grid q * round(v / q) (it changes the
// state space: never silent); `numeric_tolerant` gives the numeric semantics of Mimir-C#: values below 1e6 in
// magnitude snap to a 1e-9 grid, non-finite results are undefined, comparisons use a 1e-9 tolerance.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/successor/action.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr
{
enum class NumericStorage : u8
{
    F64,
    I32,
};

namespace plan
{
enum class NumOp : u8
{
    Const,     // a: constant index
    SlotF64,   // a: slot
    SlotI32,   // a: slot
    LoadF64,   // a: table, b: terms offset, ar: arity (fluent function over parameters)
    LoadI32,
    LoadStatic,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
};

struct NumIns
{
    NumOp op = NumOp::Const;
    u8 ar = 0;
    u32 a = 0;
    u32 b = 0;
};

/// A postfix program: code[begin, end).
struct NumProg
{
    u32 begin = 0, end = 0;
};

struct NumCheck
{
    formalism::Comparator cmp = formalism::Comparator::Eq;
    NumProg lhs, rhs;
};

struct NumEffect
{
    formalism::AssignOp op = formalism::AssignOp::Assign;
    u32 slot = ~u32{0};  // target slot when every target term is an object (~0: no value, the effect never applies)
    bool lifted = false;
    u32 table = 0;  // lifted: function table
    u32 terms = 0;  // lifted: target terms (offset into Numeric::terms)
    u8 arity = 0;
    NumProg expr;
};

struct AuxEffect
{
    formalism::AssignOp op = formalism::AssignOp::Increase;
    NumProg expr;
};

/// Argument lookup of one function: key = sum over positions of rs[i * n + object]; k_outside marks an object that
/// never occurs at the position (no value).
struct FunctionTable
{
    static constexpr u64 k_outside = u64{1} << 56;
    static constexpr u32 k_none = ~u32{0};
    bool fluent = false;
    u32 arity = 0;
    u64 size = 0;
    std::vector<u64> rs;     // arity * n
    std::vector<f64> value;  // static, dense: NaN = undefined
    std::vector<u32> slot;   // fluent, dense: k_none = undefined
    // hashed (key spaces above 2^24): open addressing over key + 1
    std::vector<u64> hkeys;
    std::vector<f64> hvalue;
    std::vector<u32> hslot;
    u64 hmask = 0;

    [[nodiscard]] static u64 hash_key(u64 k) noexcept
    {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        return k;
    }
    [[nodiscard]] u64 find_hashed(u64 key) const noexcept
    {
        for (u64 j = hash_key(key) & hmask;; j = (j + 1) & hmask)
        {
            if (hkeys[j] == key + 1)
                return j;
            if (hkeys[j] == 0)
                return ~u64{0};
        }
    }
    [[nodiscard]] u32 slot_of(u64 key) const noexcept
    {
        if (key >= size)
            return k_none;
        if (!slot.empty())
            return slot[key];
        if (hkeys.empty())
            return k_none;
        const u64 j = find_hashed(key);
        return j == ~u64{0} ? k_none : hslot[j];
    }
    [[nodiscard]] f64 value_of(u64 key) const noexcept
    {
        if (key >= size)
            return std::numeric_limits<f64>::quiet_NaN();
        if (!value.empty())
            return value[key];
        if (hkeys.empty())
            return std::numeric_limits<f64>::quiet_NaN();
        const u64 j = find_hashed(key);
        return j == ~u64{0} ? std::numeric_limits<f64>::quiet_NaN() : hvalue[j];
    }
};

/// The compiled numeric part of a task. Empty (slots == 0, no code) for classical tasks.
struct Numeric
{
    static constexpr u32 k_max_stack = 64;

    u32 num_objects = 0;
    u32 slots = 0;  // fluent numeric slots
    u32 words = 0;  // numeric words per state (0 without slots)
    NumericStorage storage = NumericStorage::F64;
    f64 quantum = 0;
    bool tolerant = false;  // TaskOptions::numeric_tolerant

    std::vector<NumIns> code;
    std::vector<f64> consts;
    std::vector<formalism::Term> terms;
    std::vector<FunctionTable> tables;  // per function (Auxiliary: empty)
    std::vector<f64> initial;           // per slot
    std::vector<u32> slot_function;     // per slot
    std::vector<u32> slot_args_begin;   // slots + 1
    std::vector<ObjectId> slot_args;
    std::vector<NumCheck> goal;         // ground numeric goal constraints

    bool has_aux = false;  // the domain declares total-cost
    f64 aux_initial = 0;
    bool has_metric = false;
    bool minimize = true;
    NumProg metric;  // ground

    [[nodiscard]] bool any() const noexcept { return slots > 0 || !code.empty(); }
    [[nodiscard]] static u32 words_for(u32 slots, NumericStorage st) noexcept
    {
        return st == NumericStorage::I32 ? (slots + 1) / 2 : slots;
    }
};

// ------------------------------------------------------------------------------------------------ slot access
[[nodiscard]] inline f64 load_f64(const u64* num, u32 slot) noexcept { return std::bit_cast<f64>(num[slot]); }
[[nodiscard]] inline f64 load_i32(const u64* num, u32 slot) noexcept
{
    return static_cast<f64>(static_cast<i32>(static_cast<u32>(num[slot >> 1] >> ((slot & 1) * 32))));
}
[[nodiscard]] inline f64 load(const Numeric& N, const u64* num, u32 slot) noexcept
{
    return N.storage == NumericStorage::I32 ? load_i32(num, slot) : load_f64(num, slot);
}

/// The canonical double of v under the task's options (exact: -0 -> +0, one NaN).
[[nodiscard]] inline f64 canonical(const Numeric& N, f64 v) noexcept
{
    if (N.tolerant)
    {
        if (!std::isfinite(v))
            return std::numeric_limits<f64>::quiet_NaN();
        if (std::fabs(v) < 1e6)
            v = std::nearbyint(v * 1e9) / 1e9;
    }
    if (N.quantum > 0 && std::isfinite(v))
        v = std::nearbyint(v / N.quantum) * N.quantum;
    if (v == 0)
        return 0.0;
    if (std::isnan(v))
        return std::numeric_limits<f64>::quiet_NaN();
    return v;
}

/// Stores the canonical value v into slot `slot` of the numeric words. Throws std::overflow_error when an I32 task
/// meets a value that is not an int32.
inline void store(const Numeric& N, u64* num, u32 slot, f64 v)
{
    v = canonical(N, v);
    if (N.storage == NumericStorage::F64)
    {
        num[slot] = std::bit_cast<u64>(v);
        return;
    }
    if (!(v >= -2147483648.0 && v <= 2147483647.0) || v != std::nearbyint(v))
        throw std::overflow_error("mymyr: a numeric value is not an int32 (" + std::to_string(v) +
                                  "); build the task with TaskOptions::numeric_storage = F64");
    const u64 bits = static_cast<u32>(static_cast<i32>(v));
    const u32 sh = (slot & 1) * 32;
    u64& w = num[slot >> 1];
    w = (w & ~(u64{0xFFFFFFFFULL} << sh)) | (bits << sh);
}

// The evaluation stack needs no initialization: C++26 makes uninitialized automatic variables erroneous, and GCC 16
// zero-fills them (a 512-byte memset per evaluation) unless told otherwise.
#if defined(__clang__)
#define MYMYR_NUMERIC_UNINIT [[clang::uninitialized]]
#elif defined(__GNUC__)
#define MYMYR_NUMERIC_UNINIT [[gnu::uninitialized]]
#else
#define MYMYR_NUMERIC_UNINIT
#endif

/// Evaluates a program. `num`: the state's numeric words (unused by ground programs over constants); `bind`: the
/// parameter values (unused by programs without lifted loads).
[[nodiscard]] inline f64 eval(const Numeric& N, NumProg p, const u64* num, const ObjectId* bind) noexcept
{
    const NumIns* code = N.code.data();
    if (p.end == p.begin + 1)  // a constant or a ground fluent: most programs
    {
        const NumIns& in = code[p.begin];
        if (in.op == NumOp::Const)
            return N.consts[in.a];
        if (in.op == NumOp::SlotI32)
            return load_i32(num, in.a);
        if (in.op == NumOp::SlotF64)
            return load_f64(num, in.a);
    }
    MYMYR_NUMERIC_UNINIT f64 st[Numeric::k_max_stack];
    u32 sp = 0;
    const u32 n = N.num_objects;
    for (u32 i = p.begin; i < p.end; ++i)
    {
        const NumIns& in = code[i];
        switch (in.op)
        {
            case NumOp::Const: st[sp++] = N.consts[in.a]; break;
            case NumOp::SlotF64: st[sp++] = load_f64(num, in.a); break;
            case NumOp::SlotI32: st[sp++] = load_i32(num, in.a); break;
            case NumOp::LoadF64:
            case NumOp::LoadI32:
            case NumOp::LoadStatic:
            {
                const FunctionTable& T = N.tables[in.a];
                const formalism::Term* t = N.terms.data() + in.b;
                u64 k = 0;
                for (u32 j = 0; j < in.ar; ++j)
                {
                    const u32 o = t[j] >= 0 ? bind[t[j]].v : formalism::term_object(t[j]).v;
                    k += T.rs[static_cast<usize>(j) * n + o];
                }
                if (in.op == NumOp::LoadStatic)
                    st[sp++] = T.value_of(k);
                else
                {
                    const u32 s = T.slot_of(k);
                    st[sp++] = s == FunctionTable::k_none ? std::numeric_limits<f64>::quiet_NaN()
                                                           : (in.op == NumOp::LoadF64 ? load_f64(num, s) : load_i32(num, s));
                }
                break;
            }
            case NumOp::Add:
                --sp;
                st[sp - 1] += st[sp];
                break;
            case NumOp::Sub:
                --sp;
                st[sp - 1] -= st[sp];
                break;
            case NumOp::Mul:
                --sp;
                st[sp - 1] *= st[sp];
                break;
            case NumOp::Div:
                --sp;
                st[sp - 1] = st[sp] == 0 ? std::numeric_limits<f64>::quiet_NaN() : st[sp - 1] / st[sp];
                break;
            case NumOp::Neg: st[sp - 1] = -st[sp - 1]; break;
        }
        if (N.tolerant && in.op >= NumOp::Add && !std::isfinite(st[sp - 1]))
            st[sp - 1] = std::numeric_limits<f64>::quiet_NaN();
    }
    return st[0];
}

/// A comparison: false when a side is NaN (mimir); with tolerant semantics (Mimir-C#) within a 1e-9 tolerance.
[[nodiscard]] inline bool compare(const Numeric& N, formalism::Comparator cmp, f64 l, f64 r) noexcept
{
    using formalism::Comparator;
    if (std::isnan(l) || std::isnan(r))
        return false;
    if (N.tolerant)
    {
        constexpr f64 e = 1e-9;
        switch (cmp)
        {
            case Comparator::Gt: return l > r + e;
            case Comparator::Lt: return l < r - e;
            case Comparator::Eq: return std::fabs(l - r) <= e;
            case Comparator::Ne: return std::fabs(l - r) > e;
            case Comparator::Ge: return l >= r - e;
            case Comparator::Le: return l <= r + e;
        }
    }
    switch (cmp)
    {
        case Comparator::Gt: return l > r;
        case Comparator::Lt: return l < r;
        case Comparator::Eq: return l == r;
        case Comparator::Ne: return l != r;
        case Comparator::Ge: return l >= r;
        case Comparator::Le: return l <= r;
    }
    return false;
}

[[nodiscard]] inline bool holds(const Numeric& N, const NumCheck& c, const u64* num, const ObjectId* bind) noexcept
{
    return compare(N, c.cmp, eval(N, c.lhs, num, bind), eval(N, c.rhs, num, bind));
}

/// The effect family of an assign operator (mimir's EffectFamily): 1 assign, 2 additive, 3 multiplicative.
[[nodiscard]] constexpr u8 effect_family(formalism::AssignOp op) noexcept
{
    return op == formalism::AssignOp::Assign ? 1 : (op == formalism::AssignOp::Increase || op == formalism::AssignOp::Decrease) ? 2 : 3;
}
/// Whether an effect of family f may follow the recorded family (0 = none yet).
[[nodiscard]] constexpr bool compatible_family(u8 recorded, u8 f) noexcept { return recorded == 0 || (recorded == f && f != 1); }
}  // namespace plan
}  // namespace mymyr
