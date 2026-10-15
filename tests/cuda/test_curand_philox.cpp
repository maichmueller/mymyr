// The device Philox-4x32-10 of rl::rng (cuda::envk::launch_philox) equals cuRAND's curand_Philox4x32_10 on the same
// counters and keys. GPU 0 (CUDA_VISIBLE_DEVICES=0); skips without a device.

#include "curand_philox.hpp"

#include "mymyr/core/hash.hpp"
#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mymyr;

TEST(DevicePhiloxRng, DeviceEqualsCurand)
{
    if (cuda::device_count() == 0)
        GTEST_SKIP() << "no CUDA device";
    cuda::ContextOptions o;
    o.max_bytes = u64{1} << 30;
    auto ctx = cuda::DeviceContext::create(0, o);
    const cudaStream_t s = ctx->stream();
    const u64 n = 4096;
    std::vector<u32> ctr(n * 4), key(n * 2);
    for (u64 i = 0; i < n; ++i)
    {
        for (u32 j = 0; j < 4; ++j)
            ctr[i * 4 + j] = static_cast<u32>(hash::mix64(i * 8 + j));
        for (u32 j = 0; j < 2; ++j)
            key[i * 2 + j] = static_cast<u32>(hash::mix64(i * 8 + 4 + j) >> 7);
    }
    cuda::DeviceBuffer dc(ctx, n * 16, s), dk(ctx, n * 8, s), ours(ctx, n * 16, s), reference(ctx, n * 16, s);
    cuda::check(cudaMemcpyAsync(dc.data(), ctr.data(), n * 16, cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaMemcpyAsync(dk.data(), key.data(), n * 8, cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cuda::envk::launch_philox(static_cast<u32*>(dc.data()), static_cast<u32*>(dk.data()), n,
                                          static_cast<u32*>(ours.data()), s), "philox");
    cuda::check(test::launch_curand_philox(static_cast<u32*>(dc.data()), static_cast<u32*>(dk.data()), n,
                                           static_cast<u32*>(reference.data()), s), "curand");
    std::vector<u32> a(n * 4), b(n * 4);
    cuda::check(cudaMemcpyAsync(a.data(), ours.data(), n * 16, cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaMemcpyAsync(b.data(), reference.data(), n * 16, cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    EXPECT_EQ(a, b) << "rl::rng differs from curand_Philox4x32_10";
}
