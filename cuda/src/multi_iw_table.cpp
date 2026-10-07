// IW over task tables (include/mymyr/cuda/multi_iw.hpp): the searches grouped by instance, each group run by
// the DeviceMultiIw of its instance, the results put back in input order.

#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/multi_iw.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mymyr::cuda
{
struct DeviceTableIw::Impl
{
    ContextPtr ctx;
    rl::TaskTablePtr table;
    MultiIwOptions o;
    std::vector<std::unique_ptr<DeviceMultiIw>> iw;  // per instance, made on first use
    DeviceBuffer order, rows;

    DeviceMultiIw& of(u32 i)
    {
        if (!iw[i])
            iw[i] = std::make_unique<DeviceMultiIw>(ctx, table->task(i), o);
        return *iw[i];
    }

    /// The searches of each instance (ascending search index).
    [[nodiscard]] std::vector<std::vector<u32>> groups(std::span<const u32> task_ids, u64 n) const
    {
        if (task_ids.size() != n)
            throw std::invalid_argument("mymyr: device IW over a table: " + std::to_string(task_ids.size()) + " task ids for " +
                                        std::to_string(n) + " searches");
        std::vector<std::vector<u32>> g(table->size());
        for (u64 i = 0; i < n; ++i)
        {
            if (task_ids[i] >= table->size())
                throw std::out_of_range("mymyr: device IW over a table: task id " + std::to_string(task_ids[i]) + " of search " +
                                        std::to_string(i) + " is not an instance of the table (" +
                                        std::to_string(table->size()) + " instances)");
            g[task_ids[i]].push_back(static_cast<u32>(i));
        }
        return g;
    }

    template<class T>
    static std::vector<T> pick(std::span<const T> v, const std::vector<u32>& idx)
    {
        std::vector<T> out;
        if (v.empty())
            return out;
        out.reserve(idx.size());
        for (const u32 i : idx)
            out.push_back(v[i]);
        return out;
    }

    /// The batch of all searches from the groups' batches.
    struct Merge
    {
        MultiIwBatch out;
        std::vector<std::vector<u32>> plans;
        std::vector<std::vector<u64>> reached;

        Merge(u32 n, u32 words)
        {
            out.n = n;
            out.words = words;
            out.status.assign(n, search::SearchStatus::Exhausted);
            out.effective_width.assign(n, 0);
            out.plan_length.assign(n, -1);
            out.goal_rows.assign(u64{n} * words, 0);
            out.num_passes.assign(n, 0);
            plans.assign(n, {});
            reached.assign(n, {});
        }

        void add(const MultiIwBatch& b, const std::vector<u32>& idx)
        {
            out.label_width = std::max(out.label_width, b.label_width);
            out.reached_words = std::max(out.reached_words, b.reached_words);
            const u32 LW = 1 + b.label_width;
            if (b.pass_slots > out.pass_slots)
            {
                std::vector<search::IwPassStatistics> p(u64{out.n} * b.pass_slots);
                for (u64 i = 0; i < out.n; ++i)
                    std::copy_n(out.pass_stats.begin() + static_cast<std::ptrdiff_t>(i * out.pass_slots), out.num_passes[i],
                                p.begin() + static_cast<std::ptrdiff_t>(i * b.pass_slots));
                out.pass_stats.swap(p);
                out.pass_slots = b.pass_slots;
            }
            for (u32 j = 0; j < b.n; ++j)
            {
                const u32 i = idx[j];
                out.status[i] = b.status[j];
                out.effective_width[i] = b.effective_width[j];
                out.plan_length[i] = b.plan_length[j];
                const std::span<const search::IwPassStatistics> ps = b.passes(j);
                std::copy(ps.begin(), ps.end(), out.pass_stats.begin() + static_cast<std::ptrdiff_t>(u64{i} * out.pass_slots));
                out.num_passes[i] = static_cast<u8>(ps.size());
                std::copy_n(b.goal_rows.begin() + static_cast<std::ptrdiff_t>(u64{j} * b.words), std::min(b.words, out.words),
                            out.goal_rows.begin() + static_cast<std::ptrdiff_t>(u64{i} * out.words));
                if (!b.plan_offsets.empty())
                {
                    // labels of width b.label_width, widened at finish()
                    plans[i].assign(b.plan_labels.begin() + static_cast<std::ptrdiff_t>(b.plan_offsets[j] * LW),
                                    b.plan_labels.begin() + static_cast<std::ptrdiff_t>(b.plan_offsets[j + 1] * LW));
                    plans[i].push_back(b.label_width);
                }
                if (b.reached_words)
                    reached[i].assign(b.reached.begin() + static_cast<std::ptrdiff_t>(u64{j} * b.reached_words),
                                      b.reached.begin() + static_cast<std::ptrdiff_t>(u64{j + 1} * b.reached_words));
            }
            if (!b.message.empty() && out.message.find(b.message) == std::string::npos)
                out.message += (out.message.empty() ? "" : "; ") + b.message;
            MultiIwStats& s = out.stats;
            s.seconds += b.stats.seconds;
            s.host_ms += b.stats.host_ms;
            s.groups += b.stats.groups;
            s.passes += b.stats.passes;
            s.layers += b.stats.layers;
            s.chunks += b.stats.chunks;
            s.splits += b.stats.splits;
            s.redone += b.stats.redone;
            s.distinct += b.stats.distinct;
            s.replays += b.stats.replays;
            s.captures += b.stats.captures;
            s.device_loops += b.stats.device_loops;
            s.loop_handoffs += b.stats.loop_handoffs;
            s.nodes += b.stats.nodes;
            s.candidates += b.stats.candidates;
            s.uploads += b.stats.uploads;
            s.widenings += b.stats.widenings;
            s.host_schemas = std::max(s.host_schemas, b.stats.host_schemas);
            s.device_bytes = std::max(s.device_bytes, b.stats.device_bytes);
        }

        MultiIwBatch finish()
        {
            const u32 n = out.n, LW = 1 + out.label_width;
            out.plan_offsets.assign(u64{n} + 1, 0);
            for (u32 i = 0; i < n; ++i)
            {
                std::vector<u32>& p = plans[i];
                u64 steps = 0;
                if (!p.empty())
                {
                    const u32 w = p.back();
                    p.pop_back();
                    steps = p.size() / (1 + w);
                    for (u64 k = 0; k < steps; ++k)
                    {
                        for (u32 c = 0; c < 1 + w; ++c)
                            out.plan_labels.push_back(p[k * (1 + w) + c]);
                        for (u32 c = 1 + w; c < LW; ++c)
                            out.plan_labels.push_back(0xFFFFFFFFu);
                    }
                }
                out.plan_offsets[i + 1] = out.plan_offsets[i] + steps;
            }
            if (out.reached_words)
            {
                out.reached.assign(u64{n} * out.reached_words, 0);
                for (u32 i = 0; i < n; ++i)
                    std::copy(reached[i].begin(), reached[i].end(),
                              out.reached.begin() + static_cast<std::ptrdiff_t>(u64{i} * out.reached_words));
            }
            return std::move(out);
        }
    };
};

DeviceTableIw::DeviceTableIw(ContextPtr ctx, rl::TaskTablePtr table, const MultiIwOptions& options) : m(std::make_unique<Impl>())
{
    if (!ctx || !table)
        throw std::invalid_argument("mymyr: DeviceTableIw: null context or table");
    for (u32 i = 0; i < table->size(); ++i)
        if (const std::string why = multi_iw_unsupported(*table->task(i), options); !why.empty())
            throw std::invalid_argument("mymyr: the CUDA backend cannot run these searches: instance " + std::to_string(i) + ": " + why);
    m->ctx = std::move(ctx);
    m->table = std::move(table);
    m->o = options;
    m->iw.resize(m->table->size());
}

DeviceTableIw::~DeviceTableIw() = default;

const rl::TaskTablePtr& DeviceTableIw::table() const noexcept { return m->table; }
const MultiIwOptions& DeviceTableIw::options() const noexcept { return m->o; }

MultiIwBatch DeviceTableIw::run(DeviceStarts starts, std::span<const u32> task_ids, std::span<const search::GoalSpec::AtomGoal> goals,
                                std::span<const u64> seeds, cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    const u32 n = starts.rows;
    if (!goals.empty() && goals.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(goals.size()) + " goals for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    for (const search::GoalSpec::AtomGoal& g : goals)
        if (!g.fluent_only())
            throw std::invalid_argument("mymyr: device IW: a goal with derived literals or numeric constraints (the device "
                                        "searches test fluent literals only)");
    if (!seeds.empty() && seeds.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(seeds.size()) + " seeds for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    if (n && (!starts.data || starts.words == 0 || (starts.stride != 0 && starts.stride < starts.words)))
        throw std::invalid_argument("mymyr: device IW: malformed start rows");
    const auto g = m->groups(task_ids, n);
    const cudaStream_t s = m->ctx->stream();
    if (stream && stream != s)
        stream_wait(s, stream);  // the starts were written on `stream`
    Impl::Merge merge(n, m->table->words());
    for (u32 i = 0; i < m->table->size(); ++i)
    {
        const std::vector<u32>& idx = g[i];
        if (idx.empty())
            continue;
        const u64 k = idx.size();
        if (m->order.size() < k * sizeof(u32))
            m->order = DeviceBuffer(m->ctx, k * sizeof(u32), s);
        if (m->rows.size() < k * starts.words * sizeof(u64))
            m->rows = DeviceBuffer(m->ctx, k * starts.words * sizeof(u64), s);
        // the group's start rows (stride 0: every search starts from the one row)
        const std::vector<u32> rows_of = starts.stride ? idx : std::vector<u32>(k, 0);
        check(cudaMemcpyAsync(m->order.data(), rows_of.data(), k * sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the pageable source
        check(lifted::launch_gather_rows(starts.data, starts.stride, starts.words, starts.stride ? n : 1,
                                         static_cast<const u32*>(m->order.data()), k, static_cast<u64*>(m->rows.data()), s),
              "launch_gather_rows");
        const std::vector<search::GoalSpec::AtomGoal> gg = Impl::pick(goals, idx);
        const std::vector<u64> gs = Impl::pick(seeds, idx);
        const DeviceStarts part{static_cast<const u64*>(m->rows.data()), starts.words, starts.words, static_cast<u32>(k)};
        const MultiIwBatch b = m->of(i).run(part, gg, gs, s);
        merge.add(b, idx);
    }
    return merge.finish();
}

MultiIwBatch DeviceTableIw::run(std::span<const State> starts, std::span<const u32> task_ids,
                                std::span<const search::GoalSpec::AtomGoal> goals, std::span<const u64> seeds)
{
    DeviceGuard guard(m->ctx->device());
    const auto n = static_cast<u32>(starts.size());
    if (!goals.empty() && goals.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(goals.size()) + " goals for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    for (const search::GoalSpec::AtomGoal& g : goals)
        if (!g.fluent_only())
            throw std::invalid_argument("mymyr: device IW: a goal with derived literals or numeric constraints (the device "
                                        "searches test fluent literals only)");
    if (!seeds.empty() && seeds.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(seeds.size()) + " seeds for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    const auto g = m->groups(task_ids, n);
    Impl::Merge merge(n, m->table->words());
    for (u32 i = 0; i < m->table->size(); ++i)
    {
        const std::vector<u32>& idx = g[i];
        if (idx.empty())
            continue;
        const std::vector<State> part = Impl::pick(starts, idx);
        const std::vector<search::GoalSpec::AtomGoal> gg = Impl::pick(goals, idx);
        const std::vector<u64> gs = Impl::pick(seeds, idx);
        merge.add(m->of(i).run(part, gg, gs), idx);
    }
    return merge.finish();
}

std::vector<search::IwResult> multi_iw(ContextPtr ctx, rl::TaskTablePtr table, std::span<const u32> task_ids,
                                       std::span<const State> starts, const MultiIwOptions& options,
                                       std::span<const search::GoalSpec::AtomGoal> goals)
{
    DeviceTableIw x(std::move(ctx), table, options);
    const MultiIwBatch b = x.run(starts, task_ids, goals);
    std::vector<search::IwResult> out;
    out.reserve(starts.size());
    for (u32 i = 0; i < b.n; ++i)
        out.push_back(b.result(i, *table->task(task_ids[i]), starts[i], options.costs));
    return out;
}

MultiIwBatch batched_iw1(ContextPtr ctx, rl::TaskTablePtr table, DeviceStarts starts, std::span<const u32> task_ids,
                         MultiIwOptions options, cudaStream_t stream)
{
    options.max_arity = 1;
    DeviceTableIw x(std::move(ctx), std::move(table), options);
    return x.run(starts, task_ids, {}, {}, stream);
}
}  // namespace mymyr::cuda
