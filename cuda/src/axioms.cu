// Axiom strata on the device (include/mymyr/cuda/lifted.hpp, launch_axioms): the CPU's AxiomEvaluator
// (src/mymyr/axioms/evaluator.cpp) per parent, over section "plan" of the device task, with the matcher executors of the
// successor kernels (lifted_device.cuh). Also the goal test over derived bitsets and a row gather.
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted
{
namespace
{
constexpr unsigned k_axiom_block = 128;

/// The body of axiom k over one parent: `emit` gets each binding of its body (Split: the part split_i of split_n, see
/// run_fixed), with the device's matcher (rl::dev::device_fc: derived atoms are a set, the order never matters).
template<u32 OW, bool FC, bool Split = false, class Emit>
__device__ __forceinline__ void run_body(const TaskView& t, u32 k, const StateRef& s, const u64* view, u32* bind, Emit&& emit,
                                         u32 split_n = 1, u32 split_i = 0)
{
    const u32* mx = t.matcher + u64{t.axiom[u64{k} * 2 + 1]} * k_mc_count;
    if constexpr (FC)
    {
        if (device_fc(mx[k_mc_flags], MatchOrder::Free))
        {
            run_fc<OW, Split>(t, mx, s, view, bind, emit, split_n, split_i);
            return;
        }
    }
    run_fixed<OW, k_max_depth, Split>(t, mx, s, view, bind, emit, split_n, split_i);
}

/// The slot of the head of axiom k under binding b, or k_none (reported as missing under lazy slots).
__device__ __forceinline__ u32 head_slot(const TaskView& t, u32 k, const u32* b, const Derived& d)
{
    const u64 key = pattern_key(t, t.axiom[u64{k} * 2], b);
    const u32 slot = slot_of(t, key);
    if (slot == k_none)
    {
        if (d.missing_bits && key < t.atom_total)
            atomicOr(d.missing_bits + (key >> 5), 1u << (key & 31));
        if (d.missing_flag)
            *d.missing_flag = 1;
        return k_none;
    }
    return (slot >> 6) < d.words ? slot : k_none;  // (slot_of covers only this upload's derived slots)
}

/// One lane per parent through strata [s_begin, s_end), as the CPU's AxiomEvaluator does. A new derived atom is set in
/// the bitset and pushed into the parent's view at once (Engine::add_derived), so the axioms after it in the same round
/// see it.
template<u32 OW, bool FC>
__global__ void __launch_bounds__(k_axiom_block) k_axioms(TaskView t, Parents p, Views v, Derived d, u32 s_begin, u32 s_end)
{
    const u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= live_rows(p))
        return;
    u64* der = d.data + i * d.words;
    u64* view = v.data + i * v.words;
    const StateRef s{p.data + i * p.stride, p.words, der, d.words};
    u32 bind[k_max_depth];
    for (u32 si = s_begin; si < s_end; ++si)
    {
        const u32* st = t.stratum + u64{si} * 3;
        const u32 begin = st[0], end = st[0] + st[1];
        const bool recursive = st[2] != 0;
        u64 prev = ~u64{0}, cur = 0;
        bool first = true;
        for (;;)
        {
            cur = 0;
            for (u32 k = begin; k < end; ++k)
            {
                const u64 reads = d.reads ? d.reads[u64{k} * 2] : ~u64{0};
                if (!first && (reads & (prev | cur)) == 0)
                    continue;  // nothing it reads changed since its last run
                const u64 head_bit = d.reads ? d.reads[u64{k} * 2 + 1] : ~u64{0};
                auto emit = [&](const u32* b)
                {
                    const u32 slot = head_slot(t, k, b, d);
                    if (slot == k_none)
                        return;
                    const u64 bit = u64{1} << (slot & 63);
                    if (der[slot >> 6] & bit)
                        return;
                    der[slot >> 6] |= bit;
                    const u32* op = t.derived_view_op + u64{slot} * 4;
                    if (op[0] != k_none)
                        view[u64{op[0]} * OW + (op[1] >> 6)] |= u64{1} << (op[1] & 63);
                    if (op[2] != k_none)
                        view[u64{op[2]} * OW + (op[3] >> 6)] |= u64{1} << (op[3] & 63);
                    cur |= head_bit;
                };
                run_body<OW, FC>(t, k, s, view, bind, emit);
            }
            first = false;
            prev = cur;
            if (!recursive || cur == 0)
                break;
        }
    }
}

/// A flat stratum (lifted::Strata): axioms [a_begin, a_end), each body searched by `split` lanes per parent (run_fixed's
/// Split). A block takes 32 parents, its warps the (axiom, part) jobs in turn. The stratum's bodies read none of its
/// heads, so the lanes of one parent only share the words they set, atomically, and a binding found twice sets nothing
/// new.
template<u32 OW, bool FC>
__global__ void __launch_bounds__(k_axiom_block)
    k_axioms_flat(TaskView t, Parents p, Views v, Derived d, u32 a_begin, u32 a_end, u32 split)
{
    const u64 i = u64{blockIdx.x} * 32 + (threadIdx.x & 31);
    if (i >= live_rows(p))
        return;
    const u32 warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    u64* der = d.data + i * d.words;
    u64* view = v.data + i * v.words;
    const StateRef s{p.data + i * p.stride, p.words, der, d.words};
    u32 bind[k_max_depth];
    auto set = [](u64* w, u64 bit) { atomicOr(reinterpret_cast<unsigned long long*>(w), static_cast<unsigned long long>(bit)); };
    const u32 jobs = (a_end - a_begin) * split;
    for (u32 j = warp; j < jobs; j += warps)
    {
        const u32 k = a_begin + j / split;
        auto emit = [&](const u32* b)
        {
            const u32 slot = head_slot(t, k, b, d);
            if (slot == k_none)
                return;
            const u64 bit = u64{1} << (slot & 63);
            if (der[slot >> 6] & bit)
                return;
            set(der + (slot >> 6), bit);
            const u32* op = t.derived_view_op + u64{slot} * 4;
            if (op[0] != k_none)
                set(view + u64{op[0]} * OW + (op[1] >> 6), u64{1} << (op[1] & 63));
            if (op[2] != k_none)
                set(view + u64{op[2]} * OW + (op[3] >> 6), u64{1} << (op[3] & 63));
        };
        run_body<OW, FC, true>(t, k, s, view, bind, emit, split, j % split);
    }
}

template<u32 OW, bool FC>
cudaError_t launch_axioms_ow(const TaskView& t, Parents p, Views v, Derived d, const Strata& st, cudaStream_t s)
{
    const u64 seq_blocks = (u64{p.rows} + k_axiom_block - 1) / k_axiom_block;
    const u64 flat_blocks = (u64{p.rows} + 31) / 32;
    if (flat_blocks > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    for (u32 a = 0; a < st.n;)
    {
        if (st.flat[a])
        {
            if (st.count[a])
            {
                // strata of fewer axioms than a block has warps: the lanes split each body's search
                constexpr u32 warps = k_axiom_block / 32;
                const u32 split = st.count[a] < warps ? warps / st.count[a] : 1;
                const u32 jobs = st.count[a] * split;
                const u32 threads = 32 * (jobs < warps ? jobs : warps);
                k_axioms_flat<OW, FC><<<static_cast<unsigned>(flat_blocks), threads, 0, s>>>(t, p, v, d, st.begin[a],
                                                                                           st.begin[a] + st.count[a], split);
                if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
                    return e;
            }
            ++a;
            continue;
        }
        u32 b = a + 1;
        while (b < st.n && !st.flat[b])
            ++b;
        k_axioms<OW, FC><<<static_cast<unsigned>(seq_blocks), k_axiom_block, 0, s>>>(t, p, v, d, a, b);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            return e;
        a = b;
    }
    return cudaSuccess;
}

__global__ void k_goal_rows_derived(TaskView t, const u64* states, u32 words, u64 n, const u64* derived, u32 dw, u8* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        out[i] = goal_holds(t, states + i * words, words, derived ? derived + i * dw : nullptr, dw) ? 1 : 0;
}

__global__ void k_gather_rows(const u64* src, u64 stride, u32 words, u64 src_rows, const u32* order, u64 n, u64* dst)
{
    const u64 total = n * words;
    for (u64 at = u64{blockIdx.x} * blockDim.x + threadIdx.x; at < total; at += u64{gridDim.x} * blockDim.x)
    {
        const u64 i = at / words;
        const u32 w = static_cast<u32>(at % words);
        const u64 r = order ? order[i] : i;
        dst[at] = r >= src_rows ? 0 : src[r * stride + w];
    }
}
}  // namespace

cudaError_t launch_axioms(const rl::dev::TaskView& t, Parents p, Views v, Derived d, bool fc, const Strata& st,
                          cudaStream_t s)
{
    if (p.rows == 0 || t.n_strata == 0)
        return cudaSuccess;
    if (d.words == 0 || !d.data || st.n != t.n_strata || !st.begin || !st.count || !st.flat)
        return cudaErrorInvalidValue;
    if (const cudaError_t e = cudaMemsetAsync(d.data, 0, u64{p.rows} * d.words * sizeof(u64), s); e != cudaSuccess)
        return e;
    if (fc)
    {
        switch (t.ow)
        {
            case 1: return launch_axioms_ow<1, true>(t, p, v, d, st, s);
            case 2: return launch_axioms_ow<2, true>(t, p, v, d, st, s);
            case 3: return launch_axioms_ow<3, true>(t, p, v, d, st, s);
            case 4: return launch_axioms_ow<4, true>(t, p, v, d, st, s);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (t.ow)
    {
        case 1: return launch_axioms_ow<1, false>(t, p, v, d, st, s);
        case 2: return launch_axioms_ow<2, false>(t, p, v, d, st, s);
        case 3: return launch_axioms_ow<3, false>(t, p, v, d, st, s);
        case 4: return launch_axioms_ow<4, false>(t, p, v, d, st, s);
        case 5: return launch_axioms_ow<5, false>(t, p, v, d, st, s);
        case 6: return launch_axioms_ow<6, false>(t, p, v, d, st, s);
        case 7: return launch_axioms_ow<7, false>(t, p, v, d, st, s);
        case 8: return launch_axioms_ow<8, false>(t, p, v, d, st, s);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t launch_goal_rows_derived(const rl::dev::TaskView& t, const u64* states, u32 words, u64 n, const u64* derived,
                                     u32 derived_words, u8* out, cudaStream_t s)
{
    if (n)
        k_goal_rows_derived<<<grid_for(n), k_block, 0, s>>>(t, states, words, n, derived, derived_words, out);
    return cudaGetLastError();
}

cudaError_t launch_gather_rows(const u64* src, u64 stride, u32 words, u64 src_rows, const u32* order, u64 n, u64* dst,
                               cudaStream_t s)
{
    if (n && words)
        k_gather_rows<<<grid_for(n * words), k_block, 0, s>>>(src, stride, words, src_rows, order, n, dst);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::lifted
