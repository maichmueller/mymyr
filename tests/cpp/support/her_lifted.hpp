#pragma once
#include "mymyr/rl/her.hpp"
#include <vector>

namespace mymyr::test
{
/// Caller-owned padded hindsight outputs and fixed scratch, useful for host/device comparisons.
struct LiftedRelabels
{
    std::vector<i32> segment, start, end, which, binding, count, ids, match_binding, match_template, match_count;
    std::vector<u8> achieved, valid, found;
    std::vector<f32> reward;
    std::vector<u64> pos, neg, dpos, dneg, hashes, prev_hashes, work, status;
    rl::HerLiftedBatch batch;

    LiftedRelabels(u64 T, u64 N, u32 W, u32 RW, u32 D, u32 S, u32 V)
        : segment(T*N), start(N*S), end(N*S), which(N*S), binding(N*S*V), count(N), ids(T*N),
          match_binding(T*N*V), match_template(T*N), match_count(T*N), achieved(T*N), valid(N*S),
          found(T*N), reward(T*N), pos(N*S*W), neg(N*S*W), dpos(N*S*D), dneg(N*S*D),
          hashes(T*N), prev_hashes(T*N), work(N*2*(W+D)), status(1, 99)
    {
        batch.steps = T;
        batch.envs = N;
        batch.words = W;
        batch.row_words = RW;
        batch.derived_words = D;
        batch.segment = segment.data();
        batch.seg_start = start.data();
        batch.seg_end = end.data();
        batch.seg_template = which.data();
        batch.seg_binding = binding.data();
        batch.count = count.data();
        batch.achieved = achieved.data();
        batch.seg_valid = valid.data();
        batch.reward = reward.data();
        batch.goal_pos = pos.data();
        batch.goal_neg = neg.data();
        batch.goal_pos_derived = dpos.data();
        batch.goal_neg_derived = dneg.data();
        batch.hashes = hashes.data();
        batch.prev_hashes = prev_hashes.data();
        batch.work_goals = work.data();
        batch.ids = ids.data();
        batch.status = status.data();
        batch.matches.found = found.data();
        batch.matches.binding = match_binding.data();
        batch.matches.template_id = match_template.data();
        batch.matches.count = match_count.data();
    }
};
}  // namespace mymyr::test
