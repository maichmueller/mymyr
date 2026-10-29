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
//   3. achieved = (states[t, i] & goal) == goal; reward = step_reward, plus goal_reward if achieved;
//   4. valid: under the relabelled goal the episode ends at its first state that holds the goal, so the transition is
//      part of that episode only if no earlier state of the episode holds it: none of the reached states of steps
//      start .. t - 1 and, when prev_states (the states the steps started from) is given, not the episode's first state
//      prev_states[start, i] (its reset state, or the state before the window). Without prev_states that first state
//      is not tested. Callers drop the relabels with valid == 0 (they would follow a terminal state).
// Every draw is a Philox block of rl/rng.hpp: (seed; draw, env, purpose k_her) with env = env_ids[i] (or first_env + i)
// and draw = (t k + j)(1 + subset) + d, d = 0 for the step and d = 1 .. subset for the atoms. A relabel therefore depends
// on (seed, env id, t, j) and the window only, not on the batch, the thread or the device: the host loop, the device
// kernel (cuda/rl_ops.hpp) and the jnp version (mymyr.rl.jax.her_relabel) agree bit for bit.

#include "mymyr/core/types.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/rl/match.hpp"
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
    f32 step_reward = 0.0f;
    f32 goal_reward = 1.0f;
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
    const u64* prev_states = nullptr; // [T, N, row_words]: the states the steps started from, or null (see valid)
    const u64* goal_atoms = nullptr;  // [words], [N, words] (goal_atoms_stride = words) or null (all atoms)
    u32 goal_atoms_stride = 0;        // 0: one row shared by every env; words: env i's row i (its instance's atoms)
    const u32* env_ids = nullptr;     // [N] or null (first_env + i)
    // outputs [T, N, k] (null: not written, except goal)
    u64* goal = nullptr;              // [T, N, k, words]
    i32* source = nullptr;
    u8* achieved = nullptr;
    f32* reward = nullptr;
    u8* valid = nullptr;
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

/// Whether the state row s holds every atom of the goal g (the first `words` words).
MYMYR_HD bool her_holds(const u64* s, const u64* g, u32 words)
{
    bool ok = true;
    for (u32 w = 0; w < words; ++w)
        ok = ok && (s[w] & g[w]) == g[w];
    return ok;
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
    const bool ok = her_holds(b.states + (t * b.envs + i) * b.row_words, g, b.words);
    if (b.valid)
    {
        bool first = !(b.prev_states && her_holds(b.prev_states + (start * b.envs + i) * b.row_words, g, b.words));
        for (u64 u = start; first && u < t; ++u)
            first = !her_holds(b.states + (u * b.envs + i) * b.row_words, g, b.words);
        b.valid[out] = first ? 1 : 0;
    }
    if (b.source)
        b.source[out] = static_cast<i32>(src);
    if (b.achieved)
        b.achieved[out] = ok ? 1 : 0;
    if (b.reward)
        b.reward[out] = c.step_reward + (ok ? c.goal_reward : 0.0f);
}

/// Every relabel of the window (host loop; T * N * k independent relabels, valid tests O(t - start) states each).
/// Throws std::invalid_argument for a missing array or k == 0.
void her_relabel(const HerBatch& b, const HerConfig& c);
}  // namespace mymyr::rl


namespace mymyr::rl
{
/// Lifted hindsight relabelling. Episodes are runs between done flags, including partial episodes at window edges.
/// Environments and episodes are processed in ascending order; segments within an episode are generated backwards.
/// A matching reached state's first template/binding defines the goal. The backward scan excludes the first step
/// whose starting state either satisfies that goal or equals a successor already scanned (hashes only prefilter
/// exact equality, including numeric words). Scanning continues before that excluded step.
struct HerLiftedConfig
{
    u32 max_segments = 1;
    u64 max_generated = ~u64{0};
    u64 max_steps = 100000000;  // matcher budget per reached row and template
    f32 step_reward = 0.0f, goal_reward = 1.0f;
};

/// Caller-owned contiguous window, outputs and scratch. task_ids is [N] (task_stride=0) or [T,N] (task_stride=N);
/// the instance must be constant within each episode. State/derived row strides also apply to their prev arrays.
/// Outputs: segment/reward/achieved [T,N]; segment metadata [N,S], binding [N,S,V], fluent masks [N,S,W],
/// derived masks [N,S,DW], count [N] (retained segments). Invalid metadata and binding are -1; masks/rewards are zero.
/// Status is cleared per call: bit 0 segment overflow, 1 device variable limit, 2 invalid instance, 3 matcher budget
/// exhaustion. An exhausted template is skipped; later templates may match. Counts describe completed searches.
/// Scratch sizes depend only on shapes. All arrays, including scratch, must remain live through graph replay.
struct HerLiftedBatch
{
    u64 steps = 0, envs = 0;
    u32 row_words = 0, words = 0, derived_words = 0, task_stride = 0;
    const u64 *states = nullptr, *prev_states = nullptr, *derived = nullptr, *prev_derived = nullptr;
    const u8* done = nullptr;
    const i32* task_ids = nullptr;
    i32 *segment = nullptr, *seg_start = nullptr, *seg_end = nullptr, *seg_template = nullptr, *seg_binding = nullptr;
    u8 *achieved = nullptr, *seg_valid = nullptr;
    f32* reward = nullptr;
    u64 *goal_pos = nullptr, *goal_neg = nullptr, *goal_pos_derived = nullptr, *goal_neg_derived = nullptr;
    i32* count = nullptr;
    u64* status = nullptr;
    // Scratch: hashes [T,N], previous hashes [T,N], ids [T,N], matches with capacity=1, work goals [N,2*(W+DW)].
    u64 *hashes = nullptr, *prev_hashes = nullptr, *work_goals = nullptr;
    i32* ids = nullptr;
    MatchBatch matches{};
};

namespace her_lifted_detail
{
struct Solo
{
    MYMYR_HD u32 lane() const { return 0; }
    MYMYR_HD u32 size() const { return 1; }
    MYMYR_HD u32 bit_or(u32 x) const { return x; }
    MYMYR_HD void sync() const {}
};

MYMYR_HD void prepare(const HerLiftedBatch& b, u64 row)
{
    b.hashes[row] = hash::state_words(b.states + row * b.row_words, b.row_words);
    b.prev_hashes[row] = hash::state_words(b.prev_states + row * b.row_words, b.row_words);
    b.ids[row] = b.task_ids ? b.task_ids[(row / b.envs) * b.task_stride + row % b.envs] : 0;
}

MYMYR_HD MatchBatch matches(const HerLiftedBatch& b, const HerLiftedConfig& c)
{
    MatchBatch m = b.matches;
    m.states = b.states;
    m.derived = b.derived;
    m.task_ids = b.ids;
    m.rows = b.steps * b.envs;
    m.row_words = b.row_words;
    m.derived_words = b.derived_words;
    m.capacity = 1;
    m.first = true;
    m.max_steps = c.max_steps;
    m.status = b.status;
    return m;
}

/// Clear padding as well as live entries, so replays never expose stale outputs.
template<class Group>
MYMYR_HD void clear_slot(const Group& grp, const MatchTemplates& t, const HerLiftedBatch& b,
                         const HerLiftedConfig& c, u64 i, u32 j)
{
    const u64 at = i * c.max_segments + j;
    if (!grp.lane())
    {
        b.seg_valid[at] = 0;
        b.seg_start[at] = b.seg_end[at] = b.seg_template[at] = -1;
    }
    for (u32 v = grp.lane(); v < t.max_vars; v += grp.size())
        b.seg_binding[at * t.max_vars + v] = -1;
    for (u32 w = grp.lane(); w < b.words; w += grp.size())
        b.goal_pos[at * b.words + w] = b.goal_neg[at * b.words + w] = 0;
    for (u32 w = grp.lane(); w < b.derived_words; w += grp.size())
        b.goal_pos_derived[at * b.derived_words + w] = b.goal_neg_derived[at * b.derived_words + w] = 0;
}

template<class Group>
MYMYR_HD void scan(const Group& grp, const MatchTemplates& t, const HerLiftedBatch& b,
                   const HerLiftedConfig& c, u64 i)
{
    namespace ed = encoding::detail;
    const u32 GW = 2 * (b.words + b.derived_words);
    u64* goals = GW ? b.work_goals + i * GW : nullptr;
    for (u32 j = 0; j < c.max_segments; ++j)
        clear_slot(grp, t, b, c, i, j);
    for (u64 q = grp.lane(); q < b.steps; q += grp.size())
    {
        const u64 at = q * b.envs + i;
        b.segment[at] = -1;
        b.reward[at] = 0;
        b.achieved[at] = 0;
    }
    u32 generated = 0;
    for (i64 first = 0; first < static_cast<i64>(b.steps);)
    {
        i64 last = first;
        while (last + 1 < static_cast<i64>(b.steps) && !b.done[u64(last) * b.envs + i])
            ++last;
        i64 end = last;
        while (end >= first)
        {
            const u64 row = u64(end) * b.envs + i;
            const i32 k = b.matches.template_id[row];
            if (k < 0)
            {
                --end;
                continue;
            }
            for (u32 w = grp.lane(); w < GW; w += grp.size())
                goals[w] = 0;
            grp.sync();
            if (!grp.lane())
                for (i32 l = t.literal_offsets[k]; l < t.literal_offsets[k + 1]; ++l)
                {
                    const match_detail::Objects objects{t.max_arity ? t.terms + u64(l) * t.max_arity : nullptr,
                                                       t.max_vars ? b.matches.binding + row * t.max_vars : nullptr};
                    const auto slot = ed::resolve_literal(t.literals, static_cast<u32>(b.ids[row]), t.predicate[l],
                                                          objects, static_cast<u32>(t.arity[l]));
                    if (slot.kind == ed::LiteralKind::Fluent || slot.kind == ed::LiteralKind::Derived)
                    {
                        const u32 offset = slot.kind == ed::LiteralKind::Fluent ? (t.polarity[l] ? 0 : b.words)
                            : 2 * b.words + (t.polarity[l] ? 0 : b.derived_words);
                        goals[offset + slot.slot / 64] |= u64{1} << (slot.slot % 64);
                    }
                }
            grp.sync();
            i64 start = end;
            while (start >= first)
            {
                const u64 at = u64(start) * b.envs + i;
                u32 loop = 0, failed = 0;
                for (i64 q = start + grp.lane(); q <= end; q += grp.size())
                {
                    const u64 past = u64(q) * b.envs + i;
                    if (b.prev_hashes[at] != b.hashes[past])
                        continue;
                    bool equal = true;
                    for (u32 w = 0; w < b.row_words && equal; ++w)
                        equal = b.prev_states[at * b.row_words + w] == b.states[past * b.row_words + w];
                    loop |= equal;
                }
                for (u32 w = grp.lane(); w < b.words; w += grp.size())
                {
                    const u64 value = b.prev_states[at * b.row_words + w];
                    failed |= (value & goals[w]) != goals[w] || (value & goals[b.words + w]) != 0;
                }
                for (u32 w = grp.lane(); w < b.derived_words; w += grp.size())
                {
                    const u64 value = b.prev_derived[at * b.derived_words + w];
                    failed |= (value & goals[2 * b.words + w]) != goals[2 * b.words + w] ||
                              (value & goals[2 * b.words + b.derived_words + w]) != 0;
                }
                if (grp.bit_or(loop) || !grp.bit_or(failed))
                    break;
                --start;
            }
            if (start < end)
            {
                if (generated < c.max_segments)
                {
                    const u64 at = i * c.max_segments + generated;
                    if (!grp.lane())
                    {
                        b.seg_valid[at] = 1;
                        b.seg_start[at] = static_cast<i32>(start + 1);
                        b.seg_end[at] = static_cast<i32>(end);
                        b.seg_template[at] = k;
                    }
                    for (u32 v = grp.lane(); v < static_cast<u32>(t.num_vars[k]); v += grp.size())
                        b.seg_binding[at * t.max_vars + v] = b.matches.binding[row * t.max_vars + v];
                    for (u32 w = grp.lane(); w < b.words; w += grp.size())
                    {
                        b.goal_pos[at * b.words + w] = goals[w];
                        b.goal_neg[at * b.words + w] = goals[b.words + w];
                    }
                    for (u32 w = grp.lane(); w < b.derived_words; w += grp.size())
                    {
                        b.goal_pos_derived[at * b.derived_words + w] = goals[2 * b.words + w];
                        b.goal_neg_derived[at * b.derived_words + w] = goals[2 * b.words + b.derived_words + w];
                    }
                    for (i64 q = start + 1 + grp.lane(); q <= end; q += grp.size())
                    {
                        const u64 out = u64(q) * b.envs + i;
                        b.segment[out] = static_cast<i32>(generated);
                        b.achieved[out] = q == end;
                        b.reward[out] = q == end ? c.goal_reward : c.step_reward;
                    }
                }
                ++generated;
            }
            end = start - 1;
        }
        first = last + 1;
    }
    if (!grp.lane())
        b.count[i] = static_cast<i32>(generated);
}

/// Exclusive prefix in environment order, counting generated segments even when their output capacity overflows.
MYMYR_HD u64 cap_counts(const HerLiftedBatch& b, const HerLiftedConfig& c)
{
    u64 remaining = c.max_generated, flags = 0;
    for (u64 i = 0; i < b.envs; ++i)
    {
        const u64 total = static_cast<u64>(b.count[i]);
        const u64 allowed = total < remaining ? total : remaining;
        remaining -= allowed;
        if (allowed > c.max_segments)
            flags |= 1;
        b.count[i] = static_cast<i32>(allowed < c.max_segments ? allowed : c.max_segments);
    }
    return flags;
}

template<class Group>
MYMYR_HD void trim(const Group& grp, const MatchTemplates& t, const HerLiftedBatch& b,
                   const HerLiftedConfig& c, u64 i)
{
    for (u32 j = static_cast<u32>(b.count[i]); j < c.max_segments; ++j)
        clear_slot(grp, t, b, c, i, j);
    for (u64 q = grp.lane(); q < b.steps; q += grp.size())
    {
        const u64 at = q * b.envs + i;
        if (b.segment[at] >= b.count[i])
        {
            b.segment[at] = -1;
            b.reward[at] = 0;
            b.achieved[at] = 0;
        }
    }
}
}  // namespace her_lifted_detail

/// Destination-passing host form; a pool partitions reached rows and environment scans among its workers.
void her_relabel_lifted(const MatchTemplates& templates, const HerLiftedBatch& batch, const HerLiftedConfig& config,
                        ThreadPool* pool = nullptr);
}  // namespace mymyr::rl
