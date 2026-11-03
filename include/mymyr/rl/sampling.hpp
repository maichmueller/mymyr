#pragma once
// Counter-based instance sampling, the draws of mymyr.rl.InstanceSampler bit for bit: an environment's k-th sampled
// instance is a function of (seed, env id, k) alone, not of the batch, the thread, the device or the action RNG. Uniform
// over the instances, or weighted: `thresholds` [m - 1] are the cumulative distribution of the positive weights over
// 2^32 and `indices` [m] the instances of positive weight (InstanceSampler's arrays).
//
// The environments draw inside reset and step (rl/env.hpp: EnvConfig::sampling, EnvBatch::reset_counts): the host env,
// the device env (rl::rng::instance_draw on device copies of the arrays), the CPU pool, and the torch and JAX
// environments built on them.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/rng.hpp"

#include <string>
#include <vector>

namespace mymyr::rl
{
struct InstanceSampling
{
    u32 instances = 0;            // 0: no sampling
    std::vector<u32> thresholds;  // weighted (indices not empty)
    std::vector<i32> indices;

    [[nodiscard]] bool enabled() const noexcept { return instances > 0; }
    /// The number of instances of positive weight (0: uniform).
    [[nodiscard]] u32 weighted() const noexcept { return static_cast<u32>(indices.size()); }

    /// The instance of env `env_id`'s draw number `reset_count` under `seed`.
    [[nodiscard]] i32 sample(u64 seed, u64 env_id, u32 reset_count) const noexcept
    {
        return rng::instance_draw(seed, env_id, reset_count, instances, thresholds.data(), indices.data(), weighted());
    }
};

/// Throws std::invalid_argument unless `s` is disabled or a uniform or weighted sampler over `instances` instances
/// (`what` names the caller in the message).
void check_sampling(const InstanceSampling& s, u32 instances, const std::string& what);
}  // namespace mymyr::rl
