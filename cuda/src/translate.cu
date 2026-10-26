#include "mymyr/cuda/translate.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <climits>

namespace mymyr::cuda
{
namespace
{
constexpr u32 missing = ~u32{0}, checked_static = ~u32{0} - 1;

__device__ f64 load(const u64* data, u32 slot, bool i32)
{
    return i32 ? static_cast<f64>(static_cast<mymyr::i32>(data[slot / 2] >> ((slot % 2) * 32)))
               : __longlong_as_double(static_cast<long long>(data[slot]));
}

__device__ bool convert(f64 value, const TranslateInstance& inst, f64& result)
{
    result = value;
    if (inst.tolerant)
    {
        if (!isfinite(result))
            result = nan("");
        else if (fabs(result) < 1e6)
            result = nearbyint(result * 1e9) / 1e9;
    }
    if (inst.quantum > 0 && isfinite(result))
        result = nearbyint(result / inst.quantum) * inst.quantum;
    if (result == 0)
        result = 0;
    if (isnan(result))
        result = nan("");
    return (value == result || (isnan(value) && isnan(result))) &&
           (!inst.target_i32 || (result >= -2147483648.0 && result <= 2147483647.0 && result == nearbyint(result)));
}

__device__ void store(u64* data, u32 slot, bool i32, f64 value)
{
    if (!i32)
        data[slot] = static_cast<u64>(__double_as_longlong(value));
    else
    {
        const u32 shift = (slot % 2) * 32;
        data[slot / 2] = (data[slot / 2] & ~(u64{0xffffffff} << shift)) |
                         (u64{static_cast<u32>(static_cast<mymyr::i32>(value))} << shift);
    }
}

__global__ void init_status(i32* status)
{
    status[0] = 0;
    status[1] = INT_MAX;
    status[2] = -1;
}

// One thread owns a whole row, including packed pairs of numeric values.
__global__ void translate(TranslateView map, const u64* source, u64 rows, u64 stride, u32 sw,
                          const i32* ids, u64* target, u32 tw, u32 rw, u8* lossless, i32* status)
{
    for (u64 row = u64{blockIdx.x} * blockDim.x + threadIdx.x; row < rows; row += u64{blockDim.x} * gridDim.x)
    {
        u64* out = target + row * rw;
        for (u32 w = 0; w < rw; ++w)
            out[w] = 0;
        const i32 id = ids ? ids[row] : 0;
        u32 flags = 0;
        if (id < 0 || static_cast<u32>(id) >= map.count)
            flags = 2;
        else
        {
            const auto inst = map.instances[id];
            flags |= inst.missing_constants ? 1u : 0u;
            for (u32 w = 0; w < inst.target_words; ++w)
                out[w] = map.constants[inst.constant_begin + w];
            const u32 nn = inst.target_i32 ? (inst.target_numeric_slots + 1) / 2 : inst.target_numeric_slots;
            for (u32 w = 0; w < nn; ++w)
                out[tw + w] = map.initial[inst.initial_begin + w];
            const u64* in = source + row * stride;
            for (u32 w = 0; w < inst.source_words; ++w)
                for (u64 bits = in[w]; bits; bits &= bits - 1)
                {
                    const u32 slot = w * 64 + static_cast<u32>(__ffsll(static_cast<long long>(bits)) - 1);
                    if (slot >= inst.slots)
                        flags |= 2;
                    else
                    {
                        const u32 mapped = map.slots[inst.slots_begin + slot];
                        if (mapped == missing)
                            flags |= 1;
                        else if (mapped != checked_static)
                        {
                            if (mapped >= inst.target_slots)
                                flags |= 2;
                            else
                                out[mapped / 64] |= u64{1} << (mapped % 64);
                        }
                    }
                }
            for (u32 n = 0; n < inst.numeric_slots; ++n)
            {
                const u32 mapped = map.numeric[inst.numeric_begin + n];
                f64 value = 0;
                if (mapped == missing)
                    flags |= 1;
                else if (!convert(load(in + sw, n, inst.source_i32), inst, value))
                    flags |= 4;
                else
                    store(out + tw, mapped, inst.target_i32, value);
            }
        }
        lossless[row] = flags == 0;
        if (flags)
        {
            atomicOr(status, static_cast<i32>(flags));
            atomicMin(status + 1, static_cast<i32>(row));
        }
    }
}

__global__ void finish_status(TranslateView map, const u64* source, u64 stride, u32 sw, const i32* ids, i32* status)
{
    if (status[1] == INT_MAX)
    {
        status[1] = -1;
        return;
    }
    const u32 row = static_cast<u32>(status[1]);
    const i32 id = ids ? ids[row] : 0;
    if (id < 0 || static_cast<u32>(id) >= map.count)
        return;
    const auto inst = map.instances[id];
    const u64* in = source + u64{row} * stride;
    for (u32 w = 0; w < inst.source_words; ++w)
        for (u64 bits = in[w]; bits; bits &= bits - 1)
        {
            const u32 slot = w * 64 + static_cast<u32>(__ffsll(static_cast<long long>(bits)) - 1);
            if (slot >= inst.slots || map.slots[inst.slots_begin + slot] == missing)
            {
                status[2] = static_cast<i32>(slot);
                return;
            }
        }
    for (u32 n = 0; n < inst.numeric_slots; ++n)
    {
        f64 value = 0;
        if (map.numeric[inst.numeric_begin + n] == missing || !convert(load(in + sw, n, inst.source_i32), inst, value))
        {
            status[2] = static_cast<i32>(inst.slots + n);
            return;
        }
    }
    if (inst.missing_constants)
        status[2] = static_cast<i32>(inst.slots + inst.numeric_slots);
}
}

cudaError_t translate_rows(TranslateView map, const u64* source, u64 rows, u64 stride, u32 source_words,
                            const i32* task_ids, u64* target, u32 target_words, u32 target_row_words,
                            u8* lossless, i32* status, cudaStream_t stream)
{
    if (rows > INT_MAX)
        return cudaErrorInvalidValue;
    init_status<<<1, 1, 0, stream>>>(status);
    if (rows)
        translate<<<static_cast<u32>(rows / 128 + (rows % 128 != 0)), 128, 0, stream>>>(
            map, source, rows, stride, source_words, task_ids, target, target_words, target_row_words, lossless, status);
    finish_status<<<1, 1, 0, stream>>>(map, source, stride, source_words, task_ids, status);
    return cudaGetLastError();
}
}
