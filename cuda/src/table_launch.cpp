// The launch plan of multi-instance batches over a task table (table_launch.hpp).

#include "table_launch.hpp"

#include "mymyr/cuda/generator.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace mymyr::cuda::detail
{
namespace
{
template<class T>
DeviceBuffer upload(const ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    DeviceBuffer b(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        check(cudaMemcpyAsync(b.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (plan)");
    return b;
}
}  // namespace

std::string multi_unsupported(const rl::TaskTable& table)
{
    for (u32 i = 0; i < table.size(); ++i)
    {
        const Task& task = *table.task(i);
        const std::string at = "instance " + std::to_string(i) + ": ";
        if (task.numeric_slots())
            return at + "numeric states require per-instance kernels";
        if (task.has_axioms())
            return at + "axioms";
        if (task.atoms().mode() != AtomMode::Frozen)
            return at + "lazy atom slots";
        for (const bool witness : {true, false})
        {
            const SchemaPlacement pl = ChunkGenerator::place(task, witness);
            if (pl.host_count)
                return at + "a schema on the CPU fallback";
            if (pl.ce_count)
                return at + "conditional effects";
            if (pl.deep_count)
                return at + "a matcher deeper than " + std::to_string(lifted::k_max_depth) + " parameters";
        }
    }
    return {};
}

lifted::SchemaSet TableLaunch::set(bool witness, bool canonical, bool fc) const
{
    const Kinds& k = w[witness ? 0 : 1][canonical ? 0 : 1];
    const auto* base = static_cast<const u32*>(k.sets.data());
    return lifted::SchemaSet{fc ? base + k.n_fixed : base, fc ? k.n_fc : k.n_fixed, num_schemas, witness ? 1u : 0u,
                             canonical ? 1u : 0u, fc ? 1u : 0u};
}

lifted::Multi TableLaunch::multi(const DeviceTaskTable& dt, bool witness, bool canonical, const i32* inst) const
{
    lifted::Multi m;
    m.views = dt.device_views();
    m.inst = reinterpret_cast<const u32*>(inst);
    m.fc_of = static_cast<const u8*>(w[witness ? 0 : 1][canonical ? 0 : 1].kinds.data());
    m.num_schemas = num_schemas;
    m.instances = instances;
    return m;
}

void TableLaunch::record_stream(cudaStream_t s)
{
    key.record_stream(s);
    for (auto& per_order : w)
        for (Kinds& k : per_order)
        {
            k.kinds.record_stream(s);
            k.sets.record_stream(s);
        }
}

TableLaunch plan_launches(const ContextPtr& ctx, const DeviceTaskTable& dt, cudaStream_t s, u32 num_schemas)
{
    TableLaunch p;
    const rl::TaskTable& table = *dt.table();
    const u32 I = table.size(), S = std::max(num_schemas, table.num_schemas());
    p.instances = I;
    p.num_schemas = S;
    std::vector<u32> ow(I), wb(I);
    for (u32 i = 0; i < I; ++i)
    {
        const rl::dev::TaskView& v = dt.view(i);
        ow[i] = v.ow;
        wb[i] = bucket_of(std::max<u32>(1, v.state_words));
        p.view_words = std::max<u64>(p.view_words, u64{v.view_rows} * v.ow);
    }
    // launch order: instances ranked by (OW, bucket, id)
    std::vector<u32> rank(I);
    std::iota(rank.begin(), rank.end(), 0u);
    std::sort(rank.begin(), rank.end(), [&](u32 a, u32 b) { return ow[a] != ow[b] ? ow[a] < ow[b] : wb[a] != wb[b] ? wb[a] < wb[b] : a < b; });
    p.host_key.resize(I);
    for (u32 r = 0; r < I; ++r)
        p.host_key[rank[r]] = r;
    p.key = upload(ctx, p.host_key, s);
    // the keys of the ranks with OW o and a bucket in (lo, hi]
    auto keys = [&](u32 o, u32 lo, u32 hi)
    {
        TableLaunch::Keys k{I, 0};
        for (u32 r = 0; r < I; ++r)
            if (ow[rank[r]] == o && wb[rank[r]] > lo && wb[rank[r]] <= hi)
            {
                k.lo = std::min(k.lo, r);
                k.hi = r + 1;
            }
        return k;
    };
    // launches: per OW group, at its widest bucket or per bucket
    std::set<u32> ows(ow.begin(), ow.end());
    for (const u32 o : ows)
    {
        std::set<u32> buckets;
        for (u32 i = 0; i < I; ++i)
            if (ow[i] == o)
                buckets.insert(wb[i]);
        const u32 widest = *buckets.rbegin();
        p.ows.push_back(o);
        p.ow_keys.push_back(keys(o, 0, widest));
        p.widest.push_back({o, 0, widest, widest, p.ow_keys.back()});
        for (const u32 b : buckets)
            p.per_bucket.push_back({o, bucket_floor(b), b, b, keys(o, bucket_floor(b), b)});
    }
    // the device's matcher kinds per instance (rl::dev::device_fc; the env and the expander write binding labels:
    // under witness pruning with the CPU's witnesses), and the union sets
    for (u32 k = 0; k < 4; ++k)
    {
        const u32 wi = k / 2, ci = k % 2;
        const rl::dev::MatchOrder order = ci == 1 ? rl::dev::MatchOrder::Matcher
                                          : wi == 0 ? rl::dev::MatchOrder::Witness
                                                    : rl::dev::MatchOrder::Free;
        std::vector<u8> kinds(u64{I} * S, 2);
        std::vector<u8> any_fixed(S, 0), any_fc(S, 0);
        bool sorts = false;
        for (u32 i = 0; i < I; ++i)
        {
            const Task& task = *table.task(i);
            const rl::dev::TaskView tv = rl::task_view(dt.bundle(i));
            const SchemaPlacement pl = ChunkGenerator::place(task, wi == 0);
            for (const u32 x : pl.device)
            {
                const u32 flags = rl::dev::schema_matcher_flags(tv, x, wi == 0);
                const bool fc = rl::dev::device_fc(flags, order);
                kinds[u64{i} * S + x] = fc ? 1 : 0;
                sorts = sorts || (ci == 0 && task.compiled().schemas[x].arity > 1 && rl::dev::device_sorts(flags, fc));
            }
        }
        for (u32 i = 0; i < I; ++i)
            for (u32 x = 0; x < S; ++x)
                (kinds[u64{i} * S + x] == 1 ? any_fc : any_fixed)[x] = 1;
        std::vector<u32> sets;
        for (u32 x = 0; x < S; ++x)
            if (any_fixed[x])
                sets.push_back(x);
        const auto n_fixed = static_cast<u32>(sets.size());
        for (u32 x = 0; x < S; ++x)
            if (any_fc[x])
                sets.push_back(x);
        TableLaunch::Kinds& kd = p.w[wi][ci];
        kd.kinds = upload(ctx, kinds, s);
        kd.sets = upload(ctx, sets, s);
        kd.n_fixed = n_fixed;
        kd.n_fc = static_cast<u32>(sets.size()) - n_fixed;
        kd.sorts = sorts;
    }
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the pageable sources above
    return p;
}

bool capturing(cudaStream_t s)
{
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(s, &st), "cudaStreamIsCapturing");
    return st != cudaStreamCaptureStatusNone;
}

void check_order(const ContextPtr& ctx, const u32* inst, u32 instances, lifted::OrderKeys keys, u64 n, const u32* order,
                 const u32* pos, u64 stride, cudaStream_t s)
{
#ifndef NDEBUG
    if (n == 0 || capturing(s))
        return;
    DeviceBuffer bad(ctx, sizeof(u32), s);
    check(cudaMemsetAsync(bad.data(), 0, sizeof(u32), s), "cudaMemsetAsync");
    check(lifted::launch_check_order_keys(inst, instances, keys, n, order, pos, stride, static_cast<u32*>(bad.data()), s),
          "launch_check_order_keys");
    u32 h = 0;
    check(cudaMemcpyAsync(&h, bad.data(), sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    if (h)
        throw std::logic_error("mymyr: a launch order is not a stable sort of the batch's rows by key (internal error)");
#else
    (void)ctx, (void)inst, (void)instances, (void)keys, (void)n, (void)order, (void)pos, (void)stride, (void)s;
#endif
}

void Fanout::reserve(u32 n)
{
    while (m_aux.size() + 1 < n)
    {
        m_aux.push_back(std::make_unique<Stream>());
        m_done.emplace_back();
    }
}

void Fanout::fork(cudaStream_t s, u32 n)
{
    if (n <= 1)
        return;
    reserve(n);
    m_forked.record(s);
    for (u32 i = 1; i < n; ++i)
        m_forked.wait_on(m_aux[i - 1]->get());
}

void Fanout::join(cudaStream_t s, u32 n)
{
    for (u32 i = 1; i < n; ++i)
    {
        m_done[i - 1].record(m_aux[i - 1]->get());
        m_done[i - 1].wait_on(s);
    }
}
}  // namespace mymyr::cuda::detail
