// Host loops of the RL helpers: novelty rewards (rl/novelty.hpp), prefix masks (rl/prefix.hpp) and hindsight
// relabels (rl/her.hpp). Each runs the element function its device kernel runs (cuda/rl_ops.hpp).

#include "mymyr/rl/her.hpp"
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
