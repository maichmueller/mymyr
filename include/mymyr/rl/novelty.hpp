#pragma once
// Width-1 novelty rewards for RL (Lipovetzky, Ramirez, Geffner, IJCAI 2015): one atom table per environment,
// `seen[N, S]` (S state words: bit a of a row = atom slot a was seen in
// that environment). An update of environment i with state s gives
//
//   r_int[i] = popcount(s & ~seen[i])   (the atoms s adds to the table)   and   seen[i] |= s,
//
// so r_int > 0 is exactly the verdict of a width-1 novelty table (novelty/novelty_table.hpp, NoveltyTable(1, ...)
// mark_state) fed that environment's states in order. Only the first S words of a state row are read (the atom words;
// numeric words and wider rows are ignored). Rows are independent: the result does not depend on the batch or the
// thread, and the device kernel (cuda/rl_ops.hpp) runs the same row function.
//
// Width 2 (pairs) needs F^2 bits per environment (125 KB at F = 1000) and is not provided.

#include "mymyr/core/types.hpp"

namespace mymyr::rl
{
namespace detail
{
MYMYR_HD int popcount_word(u64 x)
{
#if defined(__CUDA_ARCH__)
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}
}  // namespace detail

/// The update of one environment's table: the number of atoms of `state` not in `seen` (S words each), then seen |= s.
MYMYR_HD i32 novelty_update_row(u64* seen, const u64* state, u32 words)
{
    i32 r = 0;
    for (u32 w = 0; w < words; ++w)
    {
        const u64 s = state[w];
        r += detail::popcount_word(s & ~seen[w]);
        seen[w] |= s;
    }
    return r;
}

/// The novelty update of `rows` environments: seen [rows, words] (in place), states [rows, state_stride] (the first
/// `words` words of each row are read; state_stride >= words), reward [rows] (r_int; null: not written).
void novelty_update(u64* seen, const u64* states, u64 rows, u32 words, u32 state_stride, i32* reward);
}  // namespace mymyr::rl
