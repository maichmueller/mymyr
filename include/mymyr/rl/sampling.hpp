#pragma once
// Counter-based instance sampling, the draws of mymyr.rl.InstanceSampler bit for bit: an environment's k-th sampled
// instance is a function of (seed, env id, k) alone, not of the batch, the thread or the action RNG. Uniform over the
// instances, or weighted: `thresholds` [m - 1] are the cumulative distribution of the positive weights over 2^32 and
// `indices` [m] the instances of positive weight (InstanceSampler's arrays).

#include "mymyr/core/types.hpp"

#include <algorithm>
#include <vector>

namespace mymyr::rl
{
struct InstanceSampling
{
    u32 instances = 0;            // 0: no sampling
    std::vector<u32> thresholds;  // weighted (indices not empty)
    std::vector<i32> indices;

    [[nodiscard]] bool enabled() const noexcept { return instances > 0; }

    /// The instance of env `env_id`'s draw number `reset_count` under `seed`.
    [[nodiscard]] i32 sample(u64 seed, u64 env_id, u32 reset_count) const noexcept
    {
        const u32 value = mix(static_cast<u32>(seed) ^ mix(static_cast<u32>(seed >> 32) ^ 0x53414D50u));
        const u32 bits = mix(value ^ mix(static_cast<u32>(env_id)) ^ mix(reset_count ^ 0x9E3779B9u));
        if (indices.empty())
            return static_cast<i32>((u64{bits} * instances) >> 32);
        // the thresholds ascend: the count of those at most `bits`
        const auto k = std::upper_bound(thresholds.begin(), thresholds.end(), bits) - thresholds.begin();
        return indices[static_cast<usize>(k)];
    }

    [[nodiscard]] static constexpr u32 mix(u32 x) noexcept
    {
        x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
        x = (x ^ (x >> 13)) * 0xC2B2AE35u;
        return x ^ (x >> 16);
    }
};
}  // namespace mymyr::rl
