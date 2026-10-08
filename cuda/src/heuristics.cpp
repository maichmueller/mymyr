// Batched grounded heuristics on the device (include/mymyr/cuda/heuristics.hpp): the upload of the relaxed grounding,
// the launch configuration, the CPU fallback for states outside the grounding, and the CPU reference of the device's
// h_FF and set-additive tie-breaking.

#include "mymyr/cuda/heuristics.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/heuristics_kernels.hpp"
#include "mymyr/heuristics/action_costs.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace mymyr::cuda
{
namespace
{
constexpr u32 k_inf = hk::k_inf;
constexpr u32 k_none = hk::k_none;

u32 kind_code(heuristics::Kind k)
{
    switch (k)
    {
        case heuristics::Kind::Max: return hk::k_max;
        case heuristics::Kind::Add: return hk::k_add;
        case heuristics::Kind::FF: return hk::k_ff;
        case heuristics::Kind::SetAdditive: return hk::k_set_additive;
        case heuristics::Kind::H2: return hk::k_h2;
        default: return ~u32{0};
    }
}

[[nodiscard]] inline u32 sat_add(u64 a, u64 b) { return a + b >= k_inf ? k_inf - 1 : static_cast<u32>(a + b); }

/// A device copy of host vectors appended one after the other (each section 16-byte aligned).
template<class T>
class Packed
{
public:
    /// Appends v; returns its offset in elements.
    u64 add(const std::vector<T>& v)
    {
        const u64 at = m_host.size();
        m_host.insert(m_host.end(), v.begin(), v.end());
        while (m_host.size() * sizeof(T) % 16)
            m_host.push_back(T{});
        return at;
    }
    void upload(const ContextPtr& ctx, cudaStream_t s)
    {
        m_buf = DeviceBuffer(ctx, std::max<u64>(m_host.size() * sizeof(T), 16), s);
        if (!m_host.empty())
            check(cudaMemcpyAsync(m_buf.data(), m_host.data(), m_host.size() * sizeof(T), cudaMemcpyHostToDevice, s),
                  "cudaMemcpyAsync (relaxed task)");
    }
    [[nodiscard]] const T* at(u64 offset) const noexcept { return static_cast<const T*>(m_buf.data()) + offset; }
    void release_host() { m_host = {}; }

private:
    std::vector<T> m_host;
    DeviceBuffer m_buf;
};
}  // namespace

const char* to_string(HeuristicVariant v) noexcept
{
    switch (v)
    {
        case HeuristicVariant::Auto: return "auto";
        case HeuristicVariant::Sweep: return "sweep";
        case HeuristicVariant::Frontier: return "frontier";
    }
    return "?";
}

std::string DeviceHeuristic::unsupported(const Task& task, const DeviceHeuristicOptions& o)
{
    if (kind_code(o.kind) == ~u32{0})
        return std::string("the heuristic '") + heuristics::to_string(o.kind) +
               "' (the device evaluates max, add, ff, h2 and set_additive)";
    if (o.costs == heuristics::Costs::Real)
    {
        const heuristics::ActionCosts costs(task);
        if (!costs.state_independent())
            return "real costs that depend on the state";
        if (!costs.integral())
            return "real costs that are not integral";
    }
    return {};
}

struct DeviceHeuristic::Impl
{
    ContextPtr ctx;
    TaskPtr task;
    DeviceHeuristicOptions o;
    std::shared_ptr<const heuristics::RelaxedTask> R;
    DeviceHeuristicStats st;
    hk::Relaxed view{};
    hk::Launch launch{};
    Packed<u32> words32;
    Packed<u8> bytes8;
    DeviceBuffer slot_tables;  // slot_pos then slot_neg
    u32 slots = 0;
    std::vector<u32> h_slot_pos, h_slot_neg;
    DeviceBuffer scratch;       // global mode
    DeviceBuffer out_u32, status, counters, rows_buf;
    u64 out_cap = 0, status_cap = 0, rows_cap = 0;
    PinnedBuffer pinned{64};
    std::unique_ptr<heuristics::Heuristic> cpu;  // fallback and reference (h_max, h_add)
    // reference scratch
    std::vector<u32> r_cost, r_init, r_cnt, r_acc, r_lvl, r_cand, r_supp, r_touched, r_avail, r_next;
    std::vector<u8> r_settled, r_pmark, r_gmark, r_is_goal;

    heuristics::Heuristic& cpu_heuristic()
    {
        if (!cpu)
        {
            heuristics::Options ho;
            ho.kind = o.kind;
            ho.costs = o.costs;
            ho.evaluation = heuristics::Evaluation::Auto;
            ho.budget = o.budget;
            ho.relaxed = R;
            cpu = heuristics::make_heuristic(*task, ho);
        }
        return *cpu;
    }

    void upload(cudaStream_t s);
    void refresh_slots(cudaStream_t s);
    void configure();
    void run(const u64* rows, u64 stride, u32 words, u64 n, u32* out, cudaStream_t s);
    /// The device's view of a state (propositions; false if an atom lies outside the grounding).
    bool convert(StateView s, std::vector<u32>& init) const;
    f64 reference_supporters(StateView s);
};

DeviceHeuristic::DeviceHeuristic(ContextPtr ctx, TaskPtr task, const DeviceHeuristicOptions& options) : m(std::make_unique<Impl>())
{
    if (!ctx || !task)
        throw std::invalid_argument("mymyr: DeviceHeuristic: null context or task");
    m->ctx = std::move(ctx);
    m->task = std::move(task);
    m->o = options;
    if (const std::string why = unsupported(*m->task, m->o); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot evaluate this heuristic: " + why);
    DeviceGuard guard(m->ctx->device());
    const auto t0 = std::chrono::steady_clock::now();
    m->R = m->o.relaxed;
    if (!m->R)
    {
        heuristics::GroundingStats gs;
        m->R = heuristics::RelaxedTask::build(*m->task, m->o.budget, &gs);
        if (!m->R)
            throw std::invalid_argument("mymyr: the CUDA backend cannot evaluate this heuristic: the relaxed grounding "
                                        "exceeds the budget (" +
                                        gs.reason + "); use the CPU heuristic (lifted evaluation)");
    }
    else if (&m->R->task() != m->task.get())
        throw std::invalid_argument("mymyr: DeviceHeuristic: the grounding belongs to another task");
    m->st.grounding_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (m->o.costs == heuristics::Costs::Real && !m->R->real_costs_available())
        throw std::invalid_argument("mymyr: the CUDA backend cannot evaluate this heuristic: real costs need non-negative "
                                    "integer action costs below 2^31");
    if (m->o.kind == heuristics::Kind::H2 && m->R->num_props() > k_h2_max_props)
        throw std::invalid_argument("mymyr: CUDA h2 grounding has " + std::to_string(m->R->num_props()) +
                                    " propositions; limit is " + std::to_string(k_h2_max_props));
    const cudaStream_t s = m->ctx->stream();
    m->upload(s);
    m->configure();
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

DeviceHeuristic::~DeviceHeuristic()
{
    if (m && m->ctx)
    {
        try
        {
            DeviceGuard g(m->ctx->device());
            m->ctx->synchronize();
        }
        catch (...)
        {
        }
    }
}

heuristics::Kind DeviceHeuristic::kind() const noexcept { return m->o.kind; }
const heuristics::RelaxedTask& DeviceHeuristic::relaxed() const noexcept { return *m->R; }
const std::shared_ptr<const heuristics::RelaxedTask>& DeviceHeuristic::grounding() const noexcept { return m->R; }
const DeviceHeuristicStats& DeviceHeuristic::stats() const noexcept { return m->st; }
const ContextPtr& DeviceHeuristic::context() const noexcept { return m->ctx; }
const TaskPtr& DeviceHeuristic::task() const noexcept { return m->task; }

namespace
{
/// Lanes per list of a CSR (begin[0..n]): the mean length of the list an entry is in (the sum of the squared lengths
/// over the entries), rounded up to a power of two, at most 32.
u32 mean_lanes(const std::vector<u32>& begin)
{
    u64 sq = 0;
    for (u64 p = 0; p + 1 < begin.size(); ++p)
    {
        const u64 len = begin[p + 1] - begin[p];
        sq += len * len;
    }
    u32 lanes = 1;
    while (lanes < 32 && u64{lanes} * begin.back() < sq)
        lanes *= 2;
    return lanes;
}
}  // namespace

void DeviceHeuristic::Impl::upload(cudaStream_t s)
{
    const heuristics::RelaxedTask& X = *R;
    const bool real = o.costs == heuristics::Costs::Real;
    const u32 P = X.num_props(), O = X.num_ops(), GA = X.num_ground_actions();
    std::vector<u32> pre_begin{0}, pre, eff_begin{0}, eff, opcost(O), op_ga(O), ga_cost(GA), pre_of_begin{0}, pre_of;
    std::vector<u8> axiom(O), negative(P), is_goal(P, 0);
    for (u32 op = 0; op < O; ++op)
    {
        for (u32 q : X.pre(op))
            pre.push_back(q);
        pre_begin.push_back(static_cast<u32>(pre.size()));
        for (u32 q : X.eff(op))
            eff.push_back(q);
        eff_begin.push_back(static_cast<u32>(eff.size()));
        axiom[op] = X.is_axiom(op) ? 1 : 0;
        op_ga[op] = X.ground_action(op);
        opcost[op] = axiom[op] ? 0 : (real ? X.real_cost(op_ga[op]) : 1);
    }
    for (u32 ga = 0; ga < GA; ++ga)
        ga_cost[ga] = real ? X.real_cost(ga) : 1;
    for (u32 p = 0; p < P; ++p)
    {
        for (u32 op : X.pre_of(p))
            pre_of.push_back(op);
        pre_of_begin.push_back(static_cast<u32>(pre_of.size()));
        negative[p] = X.negative(p) ? 1 : 0;
    }
    // the achievers (ascending), and the uniform cost: every operator an action of one positive cost
    std::vector<u32> ach_begin(P + 1, 0), ach;
    for (u32 op = 0; op < O; ++op)
        for (u32 j = eff_begin[op]; j < eff_begin[op + 1]; ++j)
            ++ach_begin[eff[j] + 1];
    for (u32 p = 0; p < P; ++p)
        ach_begin[p + 1] += ach_begin[p];
    {
        ach.resize(ach_begin[P]);
        std::vector<u32> fill(ach_begin.begin(), ach_begin.end() - 1);
        for (u32 op = 0; op < O; ++op)
            for (u32 j = eff_begin[op]; j < eff_begin[op + 1]; ++j)
                ach[fill[eff[j]]++] = op;
    }
    u32 uniform = O ? opcost[0] : 0;
    for (u32 op = 0; op < O && uniform; ++op)
        if (axiom[op] || opcost[op] != uniform)
            uniform = 0;
    std::vector<u32> zero(X.zero_ops().begin(), X.zero_ops().end()), goal(X.goal().begin(), X.goal().end());
    for (u32 g : goal)
        is_goal[g] = 1;
    const u64 o_pre_begin = words32.add(pre_begin), o_pre = words32.add(pre), o_eff_begin = words32.add(eff_begin),
              o_eff = words32.add(eff), o_opcost = words32.add(opcost), o_op_ga = words32.add(op_ga),
              o_ga_cost = words32.add(ga_cost), o_pre_of_begin = words32.add(pre_of_begin), o_pre_of = words32.add(pre_of),
              o_zero = words32.add(zero), o_goal = words32.add(goal), o_ach_begin = words32.add(ach_begin),
              o_ach = words32.add(ach);
    const u64 o_axiom = bytes8.add(axiom), o_negative = bytes8.add(negative), o_is_goal = bytes8.add(is_goal);
    u64 o_comp = 0, o_excl_begin = 0, o_excl = 0, o_ga_begin = 0, o_ga_ops = 0;
    if (o.kind == heuristics::Kind::H2)
    {
        std::vector<u32> comp(P, k_none), ga_begin(GA + 1, 0), ga_ops;
        for (u32 p = 0; p < P; ++p)
            if (X.negative(p))
            {
                const u32 pos = X.fluent_props(X.prop_atom(p)).first;
                comp[p] = pos;
                if (pos != k_none)
                    comp[pos] = p;
            }
        for (u32 op = 0; op < O; ++op)
            if (!X.is_axiom(op))
                ++ga_begin[X.ground_action(op) + 1];
        for (u32 ga = 0; ga < GA; ++ga)
            ga_begin[ga + 1] += ga_begin[ga];
        ga_ops.resize(ga_begin[GA]);
        std::vector<u32> fill = ga_begin;
        for (u32 op = 0; op < O; ++op)
            if (!X.is_axiom(op))
                ga_ops[fill[X.ground_action(op)]++] = op;
        std::vector<u32> excl_begin{0}, excl, tmp;
        auto falsified = [&](u32 op)
        {
            tmp.insert(tmp.end(), X.del(op).begin(), X.del(op).end());
            for (u32 p : X.eff(op))
                if (comp[p] != k_none)
                    tmp.push_back(comp[p]);
        };
        for (u32 op = 0; op < O; ++op)
        {
            tmp.assign(X.eff(op).begin(), X.eff(op).end());
            falsified(op);
            if (!X.is_axiom(op))
            {
                const u32 ga = X.ground_action(op);
                for (u32 j = ga_begin[ga]; j < ga_begin[ga + 1]; ++j)
                    if (ga_ops[j] != op && X.unconditional(ga_ops[j]))
                        falsified(ga_ops[j]);
            }
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            excl.insert(excl.end(), tmp.begin(), tmp.end());
            excl_begin.push_back(static_cast<u32>(excl.size()));
        }
        o_comp = words32.add(comp);
        o_excl_begin = words32.add(excl_begin);
        o_excl = words32.add(excl);
        o_ga_begin = words32.add(ga_begin);
        o_ga_ops = words32.add(ga_ops);
    }
    words32.upload(ctx, s);
    bytes8.upload(ctx, s);
    if (o.kind == heuristics::Kind::H2)
    {
        view.complement = words32.at(o_comp);
        view.excl_begin = words32.at(o_excl_begin);
        view.excl = words32.at(o_excl);
        view.ga_ops_begin = words32.at(o_ga_begin);
        view.ga_ops = words32.at(o_ga_ops);
    }
    view.P = P;
    view.O = O;
    view.G = static_cast<u32>(goal.size());
    view.GA = GA;
    view.pre_begin = words32.at(o_pre_begin);
    view.pre = words32.at(o_pre);
    view.eff_begin = words32.at(o_eff_begin);
    view.eff = words32.at(o_eff);
    view.opcost = words32.at(o_opcost);
    view.op_ga = words32.at(o_op_ga);
    view.ga_cost = words32.at(o_ga_cost);
    view.pre_of_begin = words32.at(o_pre_of_begin);
    view.pre_of = words32.at(o_pre_of);
    // the frontier reads a settled proposition's pre_of list with this many lanes (folding: thousands of entries per
    // list; most groundings: a few, with some long ones), the relaxed plan its achievers (uniform costs)
    view.pre_of_lanes = mean_lanes(pre_of_begin);
    view.ach_begin = words32.at(o_ach_begin);
    view.ach = words32.at(o_ach);
    view.uniform_cost = uniform;
    view.ach_lanes = mean_lanes(ach_begin);
    view.zero_ops = words32.at(o_zero);
    view.num_zero = static_cast<u32>(zero.size());
    view.goal = words32.at(o_goal);
    view.axiom = bytes8.at(o_axiom);
    view.negative = bytes8.at(o_negative);
    view.is_goal = bytes8.at(o_is_goal);
    view.goal_unreachable = X.goal_unreachable() ? 1 : 0;
    st.propositions = P;
    st.operators = O;
    st.ground_actions = GA;
    refresh_slots(s);
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the host vectors go
    words32.release_host();
    bytes8.release_host();
}

void DeviceHeuristic::Impl::refresh_slots(cudaStream_t s)
{
    const AtomIndex& A = task->atoms();
    const u32 n = A.fluent_slots();
    if (n <= slots && slot_tables.data())
        return;
    for (u32 slot = static_cast<u32>(h_slot_pos.size()); slot < n; ++slot)
    {
        const auto [pp, np] = R->fluent_props(A.canonical(AtomKind::Fluent, slot));
        h_slot_pos.push_back(pp);
        h_slot_neg.push_back(np);
    }
    DeviceBuffer t(ctx, std::max<u64>(u64{2} * n * sizeof(u32), 16), s);
    if (n)
    {
        check(cudaMemcpyAsync(t.data(), h_slot_pos.data(), u64{n} * sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        check(cudaMemcpyAsync(static_cast<u32*>(t.data()) + n, h_slot_neg.data(), u64{n} * sizeof(u32), cudaMemcpyHostToDevice, s),
              "cudaMemcpyAsync");
    }
    slot_tables = std::move(t);  // the old tables are freed after the work enqueued on s
    slots = n;
    view.slot_pos = static_cast<const u32*>(slot_tables.data());
    view.slot_neg = view.slot_pos + n;
    view.slots = n;
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the host copies are pageable: finish them
}

void DeviceHeuristic::Impl::configure()
{
    const u32 P = view.P, O = view.O;
    launch.kind = kind_code(o.kind);
    // Auto: the frontier wins on the large groundings and on those of many propositions with few operators each,
    // the sweeps elsewhere (their rounds are cheap when they sweep a few hundred operators); a warp per state wins
    // on the smallest groundings, wider blocks on the largest.
    HeuristicVariant v = o.variant;
    if (v == HeuristicVariant::Auto)
        v = O >= 2048 || (P >= 128 && O <= 2 * P) ? HeuristicVariant::Frontier : HeuristicVariant::Sweep;
    const bool h2 = launch.kind == hk::k_h2;
    if (h2)
        v = HeuristicVariant::Sweep;
    st.variant = v;
    launch.variant = v == HeuristicVariant::Frontier ? hk::k_frontier : hk::k_sweep;
    launch.group_bytes = hk::scratch_bytes(P, O, view.GA, launch.kind, launch.variant, view.uniform_cost != 0);
    const bool warp = !h2 && (o.warp_groups >= 0 ? o.warp_groups == 1 : (P <= 56 && O <= 200));
    launch.warp = warp ? 1 : 0;
    launch.threads = o.threads ? o.threads : (warp ? 128 : (O >= 2048 ? 512 : O >= 1024 ? 256 : 128));
    if (launch.threads % 32 != 0 || launch.threads > 1024)
        throw std::invalid_argument("mymyr: DeviceHeuristic: threads must be a multiple of 32, at most 1024");
    const u64 groups_per_block = warp ? launch.threads / 32 : 1;
    const u64 shared_limit = o.shared_bytes ? o.shared_bytes : u64{48} << 10;
    if (h2 && launch.group_bytes > o.max_scratch_bytes)
        throw std::invalid_argument("mymyr: CUDA h2 grounding has " + std::to_string(P) + " propositions and needs " +
                                    std::to_string(launch.group_bytes) + " scratch bytes per state; limit is " +
                                    std::to_string(o.max_scratch_bytes) + " bytes (max_scratch_bytes)");
    launch.shared = !h2 && !o.force_global && launch.group_bytes * groups_per_block <= shared_limit ? 1 : 0;
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device()), "cudaDeviceGetAttribute");
    int per_sm = 0;
    check(hk::occupancy(launch, &per_sm), "hk::occupancy");
    if (per_sm <= 0)
    {
        // does not fit shared memory after all: global scratch
        launch.shared = 0;
        check(hk::occupancy(launch, &per_sm), "hk::occupancy");
        if (per_sm <= 0)
            throw std::runtime_error("mymyr: DeviceHeuristic: no resident block for the launch configuration");
    }
    u64 blocks = u64{static_cast<u32>(per_sm)} * static_cast<u32>(sms);
    if (!launch.shared)
        blocks = std::min<u64>(blocks, std::max<u64>(1, o.max_scratch_bytes / (launch.group_bytes * groups_per_block)));
    if (o.max_blocks)
        blocks = std::min<u64>(blocks, o.max_blocks);
    launch.blocks = static_cast<u32>(std::max<u64>(blocks, 1));
    if (!launch.shared)
    {
        const u64 bytes = launch.group_bytes * groups_per_block * launch.blocks;
        scratch = DeviceBuffer(ctx, bytes, ctx->stream());
        check(cudaMemsetAsync(scratch.data(), 0, bytes, ctx->stream()), "cudaMemsetAsync (heuristic scratch)");
        launch.scratch = scratch.data();
    }
    st.threads = launch.threads;
    st.blocks = launch.blocks;
    st.warp_groups = warp;
    st.shared = launch.shared != 0;
    st.group_bytes = launch.group_bytes;
    st.supporter_levels = (launch.kind == hk::k_ff || launch.kind == hk::k_set_additive) && !view.uniform_cost;
}

void DeviceHeuristic::Impl::run(const u64* rows, u64 stride, u32 words, u64 n, u32* out, cudaStream_t s)
{
    refresh_slots(s);
    if (n == 0)
        return;
    if (!launch.shared && scratch.stream() != s)
        stream_wait(s, scratch.stream());  // the scratch is ordered on the stream of its last use
    if (status_cap < n)
    {
        status = DeviceBuffer(ctx, n, s);
        status_cap = n;
    }
    if (!counters.data())
        counters = DeviceBuffer(ctx, 16, s);
    check(cudaMemsetAsync(counters.data(), 0, 8, s), "cudaMemsetAsync");
    hk::Out ho{out, static_cast<u8*>(status.data()), static_cast<u32*>(counters.data())};
    check(hk::launch_evaluate(view, hk::Rows{rows, stride, words, n}, launch, ho, s), "hk::launch_evaluate");
    ++st.launches;
    st.evaluations += n;
    auto* c = static_cast<u32*>(pinned.data());
    check(cudaMemcpyAsync(c, counters.data(), 8, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    if (c[1])
        throw std::invalid_argument("mymyr: DeviceHeuristic: a state sets an atom slot the task has not assigned (" +
                                    std::to_string(c[1]) + " of " + std::to_string(n) + " rows)");
    if (c[0] == 0)
        return;
    // states outside the grounding: the CPU heuristic
    std::vector<u8> stat(n);
    check(cudaMemcpyAsync(stat.data(), status.data(), n, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    std::vector<u64> row(words);
    for (u64 i = 0; i < n; ++i)
    {
        if (stat[i] != hk::k_outside)
            continue;
        check(cudaMemcpyAsync(row.data(), rows + i * stride, u64{words} * sizeof(u64), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        const f64 v = cpu_heuristic().evaluate(StateView{row.data(), bits::trimmed_size(row.data(), words), nullptr, 0});
        const u32 h = v == heuristics::k_dead_end || v >= static_cast<f64>(k_inf) ? k_inf : static_cast<u32>(v);
        check(cudaMemcpyAsync(out + i, &h, sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        ++st.fallbacks;
    }
}

void DeviceHeuristic::prepare(u64 n, cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    const cudaStream_t s = stream ? stream : m->ctx->stream();
    m->refresh_slots(s);
    if (!m->launch.shared && m->scratch.stream() != s)
        stream_wait(s, m->scratch.stream());  // the scratch is ordered on the stream of its last use
    (void)n;
}

void DeviceHeuristic::evaluate_async(const u64* rows, u64 stride, u32 words, u64 n, const u32* count, u32* out, u32* flags,
                                     cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    if (n && (!rows || !out || !flags || stride < words))
        throw std::invalid_argument("mymyr: DeviceHeuristic::evaluate_async: malformed rows or output");
    if (n == 0)
        return;
    const cudaStream_t s = stream ? stream : m->ctx->stream();
    hk::Out ho{out, nullptr, flags};
    check(hk::launch_evaluate(m->view, hk::Rows{rows, stride, words, n, count}, m->launch, ho, s), "hk::launch_evaluate");
    ++m->st.launches;
}

void DeviceHeuristic::evaluate(const u64* rows, u64 stride, u32 words, u64 n, u32* out, cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    if (n && (!rows || !out || stride < words))
        throw std::invalid_argument("mymyr: DeviceHeuristic::evaluate: malformed rows or output");
    m->run(rows, stride, words, n, out, stream ? stream : m->ctx->stream());
}

void DeviceHeuristic::evaluate(const u64* rows, u64 stride, u32 words, u64 n, f64* out, cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    if (n && (!rows || !out || stride < words))
        throw std::invalid_argument("mymyr: DeviceHeuristic::evaluate: malformed rows or output");
    const cudaStream_t s = stream ? stream : m->ctx->stream();
    if (m->out_cap < n || m->out_u32.stream() != s)
    {
        m->out_u32 = DeviceBuffer(m->ctx, std::max<u64>(n, 1) * sizeof(u32), s);
        m->out_cap = std::max<u64>(n, 1);
    }
    m->run(rows, stride, words, n, static_cast<u32*>(m->out_u32.data()), s);
    check(hk::launch_to_f64(static_cast<const u32*>(m->out_u32.data()), n, out, s), "hk::launch_to_f64");
}

std::vector<f64> DeviceHeuristic::evaluate(std::span<const State> states)
{
    DeviceGuard guard(m->ctx->device());
    const cudaStream_t s = m->ctx->stream();
    u32 w = std::max<u32>(1, m->task->words());
    for (const State& x : states)
    {
        w = std::max<u32>(w, x.size_words());
    }
    const u64 n = states.size();
    std::vector<f64> out(n);
    if (n == 0)
        return out;
    std::vector<u64> rows(n * w, 0);
    for (u64 i = 0; i < n; ++i)
        std::copy_n(states[i].data(), states[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * w));
    if (m->rows_cap < rows.size() || m->rows_buf.stream() != s)
    {
        m->rows_buf = DeviceBuffer(m->ctx, rows.size() * sizeof(u64), s);
        m->rows_cap = rows.size();
    }
    check(cudaMemcpyAsync(m->rows_buf.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    DeviceBuffer vals(m->ctx, n * sizeof(f64), s);
    evaluate(static_cast<const u64*>(m->rows_buf.data()), w, w, n, static_cast<f64*>(vals.data()), s);
    check(cudaMemcpyAsync(out.data(), vals.data(), n * sizeof(f64), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    return out;
}

// ================================================================================================= the CPU reference

bool DeviceHeuristic::Impl::convert(StateView s, std::vector<u32>& init) const
{
    const heuristics::RelaxedTask& X = *R;
    const AtomIndex& A = task->atoms();
    const u32 P = X.num_props();
    init.assign(P, k_inf);
    for (u32 p = 0; p < P; ++p)
        if (X.negative(p))
            init[p] = 0;
    bool ok = true;
    bits::for_each(s.w, s.nw,
                   [&](u64 slot)
                   {
                       const auto [pp, np] = X.fluent_props(A.canonical(AtomKind::Fluent, static_cast<u32>(slot)));
                       if (pp == heuristics::RelaxedTask::k_none)
                       {
                           ok = false;
                           return;
                       }
                       init[pp] = 0;
                       if (np != k_none)
                           init[np] = k_inf;
                   });
    return ok;
}

f64 DeviceHeuristic::reference(StateView s)
{
    if (m->o.kind != heuristics::Kind::FF && m->o.kind != heuristics::Kind::SetAdditive)
        return m->cpu_heuristic().evaluate(s);
    return m->reference_supporters(s);
}

f64 DeviceHeuristic::Impl::reference_supporters(StateView s)
{
    const heuristics::RelaxedTask& X = *R;
    const u32 P = X.num_props(), O = X.num_ops();
    if (!convert(s, r_init))
        return cpu_heuristic().evaluate(s);
    if (X.goal_unreachable())
        return heuristics::k_dead_end;
    auto opcost = [&](u32 op) -> u64
    { return X.is_axiom(op) ? 0 : (o.costs == heuristics::Costs::Real ? X.real_cost(X.ground_action(op)) : 1); };
    auto free0 = [&](u32 q) { return X.negative(q) && r_init[q] == 0; };

    // h_max costs: a generalized Dijkstra with counters (the values do not depend on its order)
    r_cost = r_init;
    r_cnt.assign(O, 0);
    r_acc.assign(O, 0);
    r_settled.assign(P, 0);
    for (u32 op = 0; op < O; ++op)
        for (u32 q : X.pre(op))
            r_cnt[op] += free0(q) ? 0 : 1;
    using Item = std::pair<u32, u32>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
    for (u32 p = 0; p < P; ++p)
        if (!X.negative(p) && r_init[p] == 0)
            q.push({0, p});
    auto fire = [&](u32 op)
    {
        const u32 v = sat_add(r_acc[op], opcost(op));
        for (u32 e : X.eff(op))
            if (v < r_cost[e])
            {
                r_cost[e] = v;
                q.push({v, e});
            }
    };
    for (u32 op = 0; op < O; ++op)
        if (r_cnt[op] == 0)
            fire(op);
    while (!q.empty())
    {
        const auto [c, p] = q.top();
        q.pop();
        if (c != r_cost[p] || r_settled[p])
            continue;
        r_settled[p] = 1;
        for (u32 op : X.pre_of(p))
        {
            r_acc[op] = std::max(r_acc[op], c);
            if (--r_cnt[op] == 0)
                fire(op);
        }
    }
    for (u32 g : X.goal())
        if (r_cost[g] == k_inf)
            return heuristics::k_dead_end;

    // supporters: BFS levels over the tight achievers, the smallest operator id per level
    r_lvl.assign(P, k_inf);
    r_supp.assign(P, k_none);
    r_cand.assign(P, k_none);
    for (u32 p = 0; p < P; ++p)
        if (r_init[p] == 0)
            r_lvl[p] = 0;
    r_cnt.assign(O, 0);
    r_avail.clear();
    for (u32 op = 0; op < O; ++op)
    {
        for (u32 x : X.pre(op))
            r_cnt[op] += r_lvl[x] == k_inf ? 1 : 0;
        if (r_cnt[op] == 0)
            r_avail.push_back(op);
    }
    u32 left = 0;
    r_is_goal.assign(P, 0);
    for (u32 g : X.goal())
    {
        r_is_goal[g] = 1;
        left += r_lvl[g] == k_inf ? 1 : 0;
    }
    for (u32 lv = 1; left > 0 && !r_avail.empty(); ++lv)
    {
        r_touched.clear();
        for (u32 op : r_avail)
        {
            u32 cm = 0;
            for (u32 x : X.pre(op))
                cm = std::max(cm, r_cost[x]);
            const u32 v = sat_add(cm, opcost(op));
            for (u32 e : X.eff(op))
                if (r_lvl[e] == k_inf && r_cost[e] == v)
                {
                    if (r_cand[e] == k_none)
                        r_touched.push_back(e);
                    r_cand[e] = std::min(r_cand[e], op);
                }
        }
        r_next.clear();
        for (u32 e : r_touched)
            r_lvl[e] = lv;  // first all levels of the round (an axiom's preconditions are of lower levels anyway)
        for (u32 e : r_touched)
        {
            const u32 op = r_cand[e];
            u32 sup = op;
            if (o.kind == heuristics::Kind::FF && X.is_axiom(op))
            {
                u32 best = k_none;
                for (u32 x : X.pre(op))
                    if (r_lvl[x] + 1 == lv)
                        best = std::min(best, x);
                sup = best == k_none ? k_none : r_supp[best];
            }
            r_supp[e] = sup;
            left -= r_is_goal[e];
            for (u32 op2 : X.pre_of(e))
                if (--r_cnt[op2] == 0)
                    r_next.push_back(op2);
        }
        r_avail.swap(r_next);
    }

    // The goal closure under x -> pre(supp(x)): FF counts ground actions, set-additive supported propositions.
    r_pmark.assign(P, 0);
    r_gmark.assign(X.num_ground_actions(), 0);
    std::vector<u32> stack(X.goal().begin(), X.goal().end());
    u64 h = 0;
    while (!stack.empty())
    {
        const u32 x = stack.back();
        stack.pop_back();
        if (r_pmark[x])
            continue;
        r_pmark[x] = 1;
        const u32 op = r_supp[x];
        if (op == k_none)
            continue;
        const u32 ga = X.ground_action(op);
        if (!X.is_axiom(op) && (o.kind == heuristics::Kind::SetAdditive || !r_gmark[ga]))
        {
            r_gmark[ga] = 1;
            h += o.costs == heuristics::Costs::Real ? X.real_cost(ga) : 1;
        }
        for (u32 y : X.pre(op))
            stack.push_back(y);
    }
    return h >= k_inf ? heuristics::k_dead_end : static_cast<f64>(h);
}
}  // namespace mymyr::cuda
