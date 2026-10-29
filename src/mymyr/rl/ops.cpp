// Host loops of the RL helpers: novelty rewards (rl/novelty.hpp), prefix masks (rl/prefix.hpp) and hindsight
// relabels (rl/her.hpp). Each runs the element function its device kernel runs (cuda/rl_ops.hpp).

#include "mymyr/rl/her.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"

#include <algorithm>
#include <stdexcept>

namespace mymyr::rl
{
void novelty_update(u64* seen, const u64* states, u64 rows, u32 words, u32 state_stride, i32* reward)
{
    if (rows == 0 || words == 0)
    {
        if (reward)
            std::fill_n(reward, rows, 0);
        return;
    }
    if (!seen || !states || state_stride < words)
        throw std::invalid_argument("mymyr: novelty_update needs seen [rows, words] and states [rows, >= words]");
    for (u64 i = 0; i < rows; ++i)
    {
        const i32 r = novelty_update_row(seen + i * words, states + i * state_stride, words);
        if (reward)
            reward[i] = r;
    }
}

void prefix_masks(const LabelRows& x, const PrefixQuery& q, u8* mask)
{
    std::fill_n(mask, q.rows * q.num_objects, u8{0});
    if (q.depth > 0 && (!q.prefix || q.prefix_stride < q.depth))
        throw std::invalid_argument("mymyr: prefix_masks needs a prefix [rows, >= depth]");
    for (u64 j = 0; j < x.size; ++j)
    {
        u64 row = 0;
        u32 o = 0;
        if (prefix_hit(x, q, j, row, o))
            mask[row * q.num_objects + o] = 1;
    }
}

void schema_masks(const LabelRows& x, u64 rows, u32 num_schemas, u8* mask)
{
    std::fill_n(mask, rows * num_schemas, u8{0});
    for (u64 j = 0; j < x.size; ++j)
    {
        u64 row = 0;
        u32 s = 0;
        if (schema_hit(x, rows, num_schemas, j, row, s))
            mask[row * num_schemas + s] = 1;
    }
}

void her_relabel(const HerBatch& b, const HerConfig& c)
{
    if (c.k == 0)
        throw std::invalid_argument("mymyr: her_relabel needs k >= 1");
    if (b.steps == 0 || b.envs == 0)
        return;
    if (!b.states || !b.done || !b.goal || b.row_words < b.words)
        throw std::invalid_argument("mymyr: her_relabel needs states [T, N, >= words], done [T, N] and goal outputs");
    for (u64 t = 0; t < b.steps; ++t)
        for (u64 i = 0; i < b.envs; ++i)
            for (u32 j = 0; j < c.k; ++j)
                her_one(b, c, t, i, j);
}
}  // namespace mymyr::rl


namespace mymyr::rl
{
void her_relabel_lifted(const MatchTemplates& t, const HerLiftedBatch& b, const HerLiftedConfig& c, ThreadPool* pool)
{
    if (!b.status || (!b.count && b.envs) || b.row_words < b.words || b.steps >= u64{1} << 31 || c.max_segments >= u32{1} << 31)
        throw std::invalid_argument("mymyr: her_relabel_lifted: invalid shapes or missing count/status");
    if (b.steps && b.envs && (!b.states || !b.prev_states || !b.done || !b.hashes || !b.prev_hashes || !b.ids ||
        !b.segment || !b.achieved || !b.reward || (b.words + b.derived_words && !b.work_goals)))
        throw std::invalid_argument("mymyr: her_relabel_lifted: missing window, outputs or scratch");
    if (b.steps && b.envs && (!b.matches.found || !b.matches.template_id || !b.matches.count ||
        (t.max_vars && !b.matches.binding)))
        throw std::invalid_argument("mymyr: her_relabel_lifted: missing matcher scratch");
    if (b.envs && c.max_segments && (!b.seg_start || !b.seg_end || !b.seg_valid || !b.seg_template ||
        (t.max_vars && !b.seg_binding) || (b.words && (!b.goal_pos || !b.goal_neg)) ||
        (b.derived_words && (!b.goal_pos_derived || !b.goal_neg_derived))))
        throw std::invalid_argument("mymyr: her_relabel_lifted: missing segment outputs");
    if (b.derived_words && b.steps && b.envs && (!b.derived || !b.prev_derived))
        throw std::invalid_argument("mymyr: her_relabel_lifted: derived and prev_derived are required");
    auto run = [&](u64 size, auto function)
    {
        auto work = [&](u32 worker)
        {
            const auto [begin, end] = ThreadPool::slice(size, worker, pool ? pool->size() : 1);
            for (u64 i = begin; i < end; ++i)
                function(i);
        };
        if (pool)
            pool->run(work);
        else
            work(0);
    };
    run(b.steps * b.envs, [&](u64 row) { her_lifted_detail::prepare(b, row); });
    match_first(t, her_lifted_detail::matches(b, c), pool);
    run(b.envs, [&](u64 i) { her_lifted_detail::scan(her_lifted_detail::Solo{}, t, b, c, i); });
    *b.status |= her_lifted_detail::cap_counts(b, c);
    run(b.envs, [&](u64 i) { her_lifted_detail::trim(her_lifted_detail::Solo{}, t, b, c, i); });
}
}  // namespace mymyr::rl
