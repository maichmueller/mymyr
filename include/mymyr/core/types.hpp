#pragma once
// Fixed-width scalar aliases and the host/device function qualifier used by every core header.
// Core data is POD and shared verbatim between the CPU and CUDA executors.

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define MYMYR_HD __host__ __device__ __forceinline__
#define MYMYR_HD_HOT __host__ __device__ __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define MYMYR_HD inline
// Small functions on the successor/dedup hot path (hashes): always inlined on the host too.
#define MYMYR_HD_HOT inline __attribute__((always_inline))
#else
#define MYMYR_HD inline
#define MYMYR_HD_HOT inline
#endif

// Loops over a few words (states, object bitsets, view rows): auto-vectorizing them with masked AVX-512 prologues and
// epilogues (GCC >= 14 even vectorizes early-exit loops) costs about 30% on the lifted BrFS at -march=native.
#if defined(__clang__)
#define MYMYR_NOVECTOR _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(__GNUC__) && __GNUC__ >= 14 && !defined(__CUDACC__)
#define MYMYR_NOVECTOR _Pragma("GCC novector")
#else
#define MYMYR_NOVECTOR
#endif

namespace mymyr
{
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;
using usize = std::size_t;
// 128-bit unsigned for 64x64 -> 128 products. A GNU extension (GCC, Clang, nvcc); __extension__ keeps -Wpedantic quiet.
__extension__ typedef unsigned __int128 u128;
}  // namespace mymyr
