// h² over triangular proposition-pair tables. A block evaluates one state with Jacobi sweeps; the two cost tables
// live in global scratch, and blocks reuse their slices while walking the batch.

#include "mymyr/cuda/heuristics_kernels.hpp"

namespace mymyr::cuda::hk
{
namespace
{
__device__ __forceinline__ u64 pair_index(u32 p, u32 q)
{
    const u32 x = p < q ? p : q, y = p < q ? q : p;
    return u64{y} * (y + 1) / 2 + x;
}

__device__ __forceinline__ u32 add_cost(u32 c, u32 a)
{
    return u64{c} + a >= k_inf ? k_inf - 1 : c + a;
}

__device__ bool excludes(const Relaxed& r, u32 o, u32 p)
{
    u32 b = r.excl_begin[o], e = r.excl_begin[o + 1];
    while (b < e)
    {
        const u32 mid = b + (e - b) / 2;
        if (r.excl[mid] < p)
            b = mid + 1;
        else
            e = mid;
    }
    return b < r.excl_begin[o + 1] && r.excl[b] == p;
}

__global__ void k_evaluate_h2(Relaxed r, Rows rows, u64 group_bytes, void* scratch, Out out)
{
    __shared__ u32 flag, changed, value;
    const u32 tid = threadIdx.x, nt = blockDim.x;
    const u64 pairs = u64{r.P} * (r.P + 1) / 2;
    auto* cost = reinterpret_cast<u32*>(static_cast<unsigned char*>(scratch) + u64{blockIdx.x} * group_bytes);
    u32* next = cost + pairs;
    u32* truth = next + pairs;
    u32* base = truth + r.P;
    const u64 n = rows.count && *rows.count < rows.n ? *rows.count : rows.n;
    for (u64 state = blockIdx.x; state < n; state += gridDim.x)
    {
        if (tid == 0)
            flag = 0;
        for (u32 p = tid; p < r.P; p += nt)
            truth[p] = r.negative[p] ? 1 : 0;
        __syncthreads();
        const u64* row = rows.data + state * rows.stride;
        for (u32 w = tid; w < rows.words; w += nt)
        {
            u64 bits = row[w];
            while (bits)
            {
                const u32 slot = w * 64 + static_cast<u32>(__ffsll(static_cast<long long>(bits)) - 1);
                bits &= bits - 1;
                if (slot >= r.slots)
                {
                    atomicOr(&flag, u32{k_invalid});
                    continue;
                }
                const u32 p = r.slot_pos[slot];
                if (p == k_none)
                {
                    atomicOr(&flag, u32{k_outside});
                    continue;
                }
                truth[p] = 1;
                const u32 neg = r.slot_neg[slot];
                if (neg != k_none)
                    truth[neg] = 0;
            }
        }
        __syncthreads();
        if (flag || r.goal_unreachable)
        {
            if (tid == 0)
            {
                out.h[state] = k_inf;
                const u8 status = flag & k_invalid ? k_invalid : flag ? k_outside : k_ok;
                if (out.status)
                    out.status[state] = status;
                if (status != k_ok)
                    atomicAdd(out.counters + (status == k_invalid ? 1 : 0), 1u);
            }
            __syncthreads();
            continue;
        }
        // Initially true goals have zero-cost pairs, so their answer needs no pair-table sweep.
        if (tid == 0)
            value = 0;
        __syncthreads();
        for (u32 i = tid; i < r.G; i += nt)
            if (!truth[r.goal[i]])
                atomicOr(&value, 1u);
        __syncthreads();
        if (!value)
        {
            if (tid == 0)
            {
                out.h[state] = 0;
                if (out.status)
                    out.status[state] = k_ok;
            }
            __syncthreads();
            continue;
        }
        // A warp writes one triangular row so that adjacent lanes write adjacent pair costs.
        for (u32 q = tid / 32; q < r.P; q += nt / 32)
            for (u32 p = tid % 32; p <= q; p += 32)
            {
                const u64 at = pair_index(p, q);
                cost[at] = next[at] = truth[p] && truth[q] ? 0 : k_inf;
            }
        __syncthreads();
        for (;;)
        {
            if (tid == 0)
                changed = 0;
            for (u32 o = tid; o < r.O; o += nt)
            {
                u32 c = 0;
                for (u32 i = r.pre_begin[o]; i < r.pre_begin[o + 1]; ++i)
                    for (u32 j = i; j < r.pre_begin[o + 1]; ++j)
                    {
                        const u32 v = cost[pair_index(r.pre[i], r.pre[j])];
                        c = v > c ? v : c;
                    }
                base[o] = c;
            }
            __syncthreads();
            for (u32 o = tid; o < r.O; o += nt)
            {
                if (base[o] == k_inf)
                    continue;
                const u32 v = add_cost(base[o], r.opcost[o]);
                for (u32 i = r.eff_begin[o]; i < r.eff_begin[o + 1]; ++i)
                    for (u32 j = i; j < r.eff_begin[o + 1]; ++j)
                    {
                        const u32 p = r.eff[i], q = r.eff[j];
                        if (r.complement[p] != q)
                            atomicMin(next + pair_index(p, q), v);
                    }
                if (!r.axiom[o])
                {
                    const u32 ga = r.op_ga[o];
                    for (u32 j = r.ga_ops_begin[ga]; j < r.ga_ops_begin[ga + 1]; ++j)
                    {
                        const u32 other = r.ga_ops[j];
                        if (other == o || base[other] == k_inf)
                            continue;
                        const u32 c = base[o] > base[other] ? base[o] : base[other];
                        const u32 both = add_cost(c, r.opcost[o]);
                        for (u32 i = r.eff_begin[o]; i < r.eff_begin[o + 1]; ++i)
                            for (u32 k = r.eff_begin[other]; k < r.eff_begin[other + 1]; ++k)
                            {
                                const u32 p = r.eff[i], q = r.eff[k];
                                if (p != q && r.complement[p] != q)
                                    atomicMin(next + pair_index(p, q), both);
                            }
                    }
                }
            }
            for (u64 i = tid; i < u64{r.O} * r.P; i += nt)
            {
                const u32 o = static_cast<u32>(i / r.P), p = static_cast<u32>(i % r.P);
                u32 c = base[o];
                if (c == k_inf || excludes(r, o, p))
                    continue;
                if (r.pre_begin[o] == r.pre_begin[o + 1])
                    c = cost[pair_index(p, p)];
                else
                    for (u32 j = r.pre_begin[o]; j < r.pre_begin[o + 1]; ++j)
                    {
                        const u32 v = cost[pair_index(r.pre[j], p)];
                        c = v > c ? v : c;
                    }
                if (c == k_inf)
                    continue;
                const u32 v = add_cost(c, r.opcost[o]);
                for (u32 j = r.eff_begin[o]; j < r.eff_begin[o + 1]; ++j)
                    if (r.complement[r.eff[j]] != p)
                        atomicMin(next + pair_index(r.eff[j], p), v);
            }
            __syncthreads();
            for (u64 i = tid; i < pairs; i += nt)
            {
                if (next[i] < cost[i])
                    atomicOr(&changed, 1u);
                cost[i] = next[i];
            }
            __syncthreads();
            if (!changed)
                break;
        }
        if (tid == 0)
            value = 0;
        __syncthreads();
        for (u32 i = tid; i < r.G; i += nt)
            for (u32 j = i; j < r.G; ++j)
                atomicMax(&value, cost[pair_index(r.goal[i], r.goal[j])]);
        __syncthreads();
        if (tid == 0)
        {
            out.h[state] = value;
            if (out.status)
                out.status[state] = k_ok;
        }
        __syncthreads();
    }
}
}  // namespace

cudaError_t launch_h2(const Relaxed& r, Rows rows, Launch l, Out out, cudaStream_t s)
{
    if (rows.n == 0)
        return cudaSuccess;
    if (!l.scratch || !out.h || !out.counters || !l.blocks || l.warp || l.shared)
        return cudaErrorInvalidValue;
    k_evaluate_h2<<<l.blocks, l.threads, 0, s>>>(r, rows, l.group_bytes, l.scratch, out);
    return cudaGetLastError();
}

cudaError_t occupancy_h2(const Launch& l, int* blocks_per_sm)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, k_evaluate_h2, static_cast<int>(l.threads), 0);
}
}  // namespace mymyr::cuda::hk
