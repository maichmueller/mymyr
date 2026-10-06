// Device rollouts (include/mymyr/cuda/rollouts.hpp): seeded DeviceMultiIw runs from one start row.

#include "mymyr/cuda/rollouts.hpp"

#include "mymyr/core/bitset.hpp"

#include <algorithm>
#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
MultiIwOptions tracking(MultiIwOptions o)
{
    o.track_reached = true;
    return o;
}

ContextPtr checked(ContextPtr ctx, const TaskPtr& task)
{
    if (!ctx || !task)
        throw std::invalid_argument("mymyr: device rollouts: null context or task");
    return ctx;
}
}  // namespace

DeviceRollouts::DeviceRollouts(ContextPtr ctx, TaskPtr task, const MultiIwOptions& iw)
    : m_ctx(checked(std::move(ctx), task)), m_task(std::move(task)), m_iw(m_ctx, m_task, tracking(iw))
{
}

MultiIwBatch DeviceRollouts::run(std::span<const u64> seeds, const State* start, std::span<const search::GoalSpec::AtomGoal> goals)
{
    if (!goals.empty() && goals.size() != seeds.size())
        throw std::invalid_argument("mymyr: device rollouts: " + std::to_string(goals.size()) + " goals for " +
                                    std::to_string(seeds.size()) + " seeds (pass none, or one per seed)");
    const State s = start ? *start : m_task->initial_state();
    if (s.numeric_words())
        throw std::invalid_argument("mymyr: device rollouts: a start state with numeric values (numeric tasks are not supported)");
    // one start row, broadcast to every rollout (stride 0)
    const u32 w = std::max<u32>({1, m_task->words(), s.size_words()});
    std::vector<u64> row(w, 0);
    std::copy_n(s.data(), s.size_words(), row.begin());
    const cudaStream_t st = m_ctx->stream();
    if (m_start_words < w)
    {
        m_start = DeviceBuffer(m_ctx, w * sizeof(u64), st);
        m_start_words = w;
    }
    check(cudaMemcpyAsync(m_start.data(), row.data(), w * sizeof(u64), cudaMemcpyHostToDevice, st), "cudaMemcpyAsync (H2D)");
    const u32 n = static_cast<u32>(seeds.size());
    MultiIwBatch out = m_iw.run(DeviceStarts{static_cast<const u64*>(m_start.data()), 0, w, n}, goals, seeds, st);
    return out;  // run() synchronized the stream: `row` may go
}

MultiIwBatch rollouts_batch(ContextPtr ctx, TaskPtr task, const DeviceRolloutOptions& options)
{
    DeviceRollouts r(std::move(ctx), std::move(task), options.iw);
    return r.run(options.seeds, options.start ? &*options.start : nullptr, options.goals);
}

std::vector<CanonicalAtom> reached_atoms(const MultiIwBatch& batch, u32 i, const Task& task)
{
    std::vector<CanonicalAtom> out;
    if (batch.reached.empty())
        return out;
    const u64* r = batch.reached.data() + static_cast<u64>(i) * batch.reached_words;
    bits::for_each(r, batch.reached_words, [&](u64 s) { out.push_back(task.atoms().canonical(AtomKind::Fluent, static_cast<u32>(s))); });
    std::sort(out.begin(), out.end());  // canonical ids are unique: no ties
    return out;
}

DeviceRolloutsResult find_rollouts(ContextPtr ctx, TaskPtr task, const DeviceRolloutOptions& options)
{
    const MultiIwBatch b = rollouts_batch(ctx, task, options);
    DeviceRolloutsResult out;
    out.stats = b.stats;
    out.message = b.message;
    const State start = options.start ? *options.start : task->initial_state();
    out.rollouts.resize(b.n);
    for (u32 i = 0; i < b.n; ++i)
    {
        out.rollouts[i].search = b.result(i, *task, start, options.iw.costs);
        out.rollouts[i].reached_fluent_atoms = reached_atoms(b, i, *task);
    }
    return out;
}
}  // namespace mymyr::cuda
