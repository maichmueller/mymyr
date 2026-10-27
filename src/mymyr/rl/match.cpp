#include "mymyr/rl/match.hpp"
#include "mymyr/core/thread_pool.hpp"
#include <algorithm>
#include <atomic>
#include <vector>

namespace mymyr::rl
{
void match_first(const MatchTemplates& t, const MatchBatch& b, ThreadPool* pool)
{
    if (b.status)
        *b.status = 0;
    auto work = [&](u32 worker)
    {
        const auto [begin, end] = ThreadPool::slice(b.rows, worker, pool ? pool->size() : 1);
        std::vector<i32> binding(t.max_vars);
        std::vector<u32> base(t.max_vars), masks(t.max_vars);
        u64 flags = 0;
        for (u64 row = begin; row < end; ++row)
        {
            b.count[row] = 0;
            for (u32 j = 0; j < b.capacity; ++j)
            {
                const u64 at = row * b.capacity + j;
                b.found[at] = 0;
                b.template_id[at] = -1;
                for (u32 v = 0; v < t.max_vars; ++v)
                    b.binding[at * t.max_vars + v] = -1;
            }
            const i32 g = b.task_ids ? b.task_ids[row] : 0;
            if (g < 0 || static_cast<u32>(g) >= t.num_instances)
            {
                flags |= 4;
                continue;
            }
            for (i32 k = t.instance_offsets[g]; k < t.instance_offsets[g + 1]; ++k)
            {
                const bool found = match_detail::search(encoding::detail::SoloGroup{}, t, static_cast<u32>(g),
                    static_cast<u32>(k), b.states + row * b.row_words,
                    b.derived ? b.derived + row * b.derived_words : nullptr, binding.data(), base.data(), masks.data());
                if (!found)
                    continue;
                ++b.count[row];
                const u32 j = b.first ? 0 : static_cast<u32>(k - t.instance_offsets[g]);
                if (j >= b.capacity)
                    flags |= 1;
                else
                {
                    const u64 at = row * b.capacity + j;
                    b.found[at] = 1;
                    b.template_id[at] = k;
                    if (t.num_vars[k])
                        std::copy_n(binding.data(), t.num_vars[k], b.binding + at * t.max_vars);
                }
                if (b.first)
                    break;
            }
        }
        if (b.status)
            std::atomic_ref(*b.status).fetch_or(flags, std::memory_order_relaxed);
    };
    if (pool)
        pool->run(work);
    else
        work(0);
}
}  // namespace mymyr::rl
