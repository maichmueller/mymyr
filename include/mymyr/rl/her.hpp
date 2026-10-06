#pragma once
// Hindsight experience replay with atom-set goals (Andrychowicz et al., NeurIPS 2017): goals are
// sets of atoms (state-word masks), so a relabelled goal is (a subset of) the atoms of a state the trajectory reached,
// and the relabelled reward is the goal test of the transition's reached state.
//
// Input: a window of T steps of N environments: states[t, i] the state step t of environment i reached (its final state
// before an autoreset) and done[t, i] whether that step ended the episode. For every transition (t, i) and j < k:
//   1. the source step t' of the same episode: "future" uniform in [t, end], "episode" uniform in [start, end], "final"
//      end, where start is the step after the previous done (0) and end the next done at or after t (T - 1 when the
//      episode continues past the window: only the window's steps are drawn);
//   2. the goal: the atoms of states[t', i] (restricted to goal_atoms if given), or `subset` of them drawn one by one
//      without replacement (the r-th set bit in ascending slot order, r uniform over the remaining ones);
//   3. achieved = (states[t, i] & goal) == goal; reward = step_reward, plus goal_reward if achieved.
// Every draw is a Philox block of rl/rng.hpp: (seed; draw, env, purpose k_her) with env = env_ids[i] (or first_env + i)
// and draw = (t k + j)(1 + subset) + d, d = 0 for the step and d = 1 .. subset for the atoms. A relabel therefore depends
// on (seed, env id, t, j) and the window only, not on the batch, the thread or the device: the host loop, the device
// kernel (cuda/rl_ops.hpp) and the jnp version (mymyr.rl.jax.her_relabel) agree bit for bit.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/rng.hpp"

namespace mymyr::rl
{
enum class HerStrategy : u8
{
    Future = 0,
    Final = 1,
    Episode = 2,
};

struct HerConfig
{
    HerStrategy strategy = HerStrategy::Future;
    u32 k = 4;            // relabels per transition
    u32 subset = 0;       // atoms per goal (0: all atoms of the source state)
    u64 seed = 0;
    u64 first_env = 0;    // env id of column 0 (without env_ids)
    f32 step_reward = -1.0f;
    f32 goal_reward = 0.0f;
};

/// The window and the outputs (caller-owned).
struct HerBatch
{
    u64 steps = 0;                    // T
    u64 envs = 0;                     // N
    u32 words = 0;                    // goal words (the first `words` words of a state row are read)
    u32 row_words = 0;                // state row stride (>= words)
    const u64* states = nullptr;      // [T, N, row_words]
    const u8* done = nullptr;         // [T, N]
    const u64* goal_atoms = nullptr;  // [words], [N, words] (goal_atoms_stride = words) or null (all atoms)
    u32 goal_atoms_stride = 0;        // 0: one row shared by every env; words: env i's row i (its instance's atoms)
    const u32* env_ids = nullptr;     // [N] or null (first_env + i)
    // outputs [T, N, k] (null: not written, except goal)
    u64* goal = nullptr;              // [T, N, k, words]
    i32* source = nullptr;
    u8* achieved = nullptr;
    f32* reward = nullptr;
};

/// The episode bounds of step t of environment i: (start, end) as in the file comment.
MYMYR_HD void her_bounds(const HerBatch& b, u64 t, u64 i, u64& start, u64& end)
{
    start = t;
    while (start > 0 && !b.done[(start - 1) * b.envs + i])
        --start;
    end = t;
    while (end + 1 < b.steps && !b.done[end * b.envs + i])
        ++end;
}

/// Relabel j of transition (t, i).
MYMYR_HD void her_one(const HerBatch& b, const HerConfig& c, u64 t, u64 i, u32 j)
{
    const u64 env = b.env_ids ? u64{b.env_ids[i]} : c.first_env + i;
    const u64 base = (t * c.k + j) * (u64{1} + c.subset);
    u64 start = 0, end = 0;
    her_bounds(b, t, i, start, end);
    const u64 lo = c.strategy == HerStrategy::Future ? t : c.strategy == HerStrategy::Final ? end : start;
    const u64 pick = rng::below(rng::bits64(c.seed, env, base, rng::k_her), static_cast<u32>(end - lo + 1));
    const u64 src = lo + pick;
    const u64 out = (t * b.envs + i) * c.k + j;
    const u64* s = b.states + (src * b.envs + i) * b.row_words;
    u64* g = b.goal + out * b.words;
    const u64* atoms = b.goal_atoms ? b.goal_atoms + i * b.goal_atoms_stride : nullptr;
    if (c.subset == 0)
        for (u32 w = 0; w < b.words; ++w)
            g[w] = s[w] & (atoms ? atoms[w] : ~u64{0});
    else
    {
        // `subset` draws among the candidates (the source's atoms not drawn yet: s & atoms & ~g)
        u32 total = 0;
        for (u32 w = 0; w < b.words; ++w)
        {
            g[w] = 0;
            total += static_cast<u32>(detail::popcount_word(s[w] & (atoms ? atoms[w] : ~u64{0})));
        }
        for (u32 d = 1; d <= c.subset && total > 0; ++d, --total)
        {
            const u32 r = rng::below(rng::bits64(c.seed, env, base + d, rng::k_her), total);
            u32 before = 0;
            for (u32 w = 0; w < b.words; ++w)
            {
                const u64 x0 = s[w] & (atoms ? atoms[w] : ~u64{0}) & ~g[w];
                const u32 pc = static_cast<u32>(detail::popcount_word(x0));
                if (r < before + pc)
                {
                    u64 x = x0;
                    for (u32 q = r - before; q > 0; --q)
                        x &= x - 1;  // drop the lowest set bits before the r-th
                    g[w] |= x & (~x + 1);
                    break;
                }
                before += pc;
            }
        }
    }
    const u64* reached = b.states + (t * b.envs + i) * b.row_words;
    bool ok = true;
    for (u32 w = 0; w < b.words; ++w)
        ok = ok && (reached[w] & g[w]) == g[w];
    if (b.source)
        b.source[out] = static_cast<i32>(src);
    if (b.achieved)
        b.achieved[out] = ok ? 1 : 0;
    if (b.reward)
        b.reward[out] = c.step_reward + (ok ? c.goal_reward : 0.0f);
}

/// Every relabel of the window (host loop; T * N * k independent relabels). Throws std::invalid_argument for a
/// missing array or k == 0.
void her_relabel(const HerBatch& b, const HerConfig& c);
}  // namespace mymyr::rl
