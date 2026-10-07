// The environment step on the device (include/mymyr/cuda/env.hpp): the host driver of the fast path (cached counts,
// views and launch order; the single-instance kernels for a table of one, the multi-instance kernels otherwise; over a
// suite of several domains each domain's multi-instance kernels on its rows) and of the general path
// (SuiteExpander: a table's DeviceExpander, or one per domain), over the kernels of cuda/env_kernels.hpp.

#include "mymyr/cuda/env.hpp"

#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#include "mymyr/cuda/suite_kernels.hpp"
#include "table_launch.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::cuda
{
namespace
{
/// Views of one chunk of the fast path at most. A chunk sized to the L2 (48 MB on the L4) leaves the count kernel
/// with too few blocks per launch.
constexpr u64 k_view_chunk_bytes = u64{256} << 20;

/// 64-bit words per region of the count cache (lifted::CountCache): a pick of a recorded schema is read from the
/// cache when the region holds the bitmap of the segment's canonical ranks (a rank space of k_rank_min to k_rank_bits
/// bindings, any segment length: gripper's pick and drop over 80 balls), or its keys (at most this many bindings).
constexpr u32 k_seg_cap = 8;
constexpr u32 k_rank_bits = 64 * std::min(k_seg_cap, lifted::k_max_rank_words);
/// The smallest rank space recorded by its bitmaps: more than one bitmap word. Below it the pick's search is short and
/// ranking every binding in the count costs more than it saves.
constexpr u32 k_rank_min = 65;
/// Sorted segments recorded by their keys: those whose matcher binds at least this many free parameters, or forward
/// checks (a pick there repeats a costly search: sokoban's push and move; recording the keys of shallow searches costs
/// more memory traffic than the repeated search).
constexpr usize k_deep = 4;
/// Sorted segments recorded by their rank bitmaps: those whose matcher binds at least this many free parameters (the
/// count then ranks every binding, about half a search again: it pays where the pick's search is costly, as in
/// gripper's three-level pick and drop; on two levels a binding is cheap to find, as in miconic's up / down).
constexpr usize k_rank_steps = 3;

// control words (device and pinned host)
constexpr u32 k_err = 0, k_words_needed = 1, k_order_bad = 2, k_ctl_words = 3;

rl::ExpandOptions expand_options(const rl::EnvConfig& c)
{
    rl::ExpandOptions o;
    o.canonical_order = c.canonical_order;
    o.witness_pruning = c.witness_pruning;
    o.validate = false;  // the rows come from reset() and step()
    return o;
}

template<class T>
DeviceBuffer upload(const ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    DeviceBuffer b(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        check(cudaMemcpyAsync(b.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    return b;
}


/// Why the fast path does not run one instance (empty: it does).
std::string fast_why(const Task& task, const rl::EnvConfig& config)
{
    if (const std::string why = ChunkGenerator::unsupported(task); !why.empty())
        return why;
    if (task.has_axioms())
        return "axioms (their derived atoms are evaluated on the CPU)";
    if (task.atoms().mode() != AtomMode::Frozen)
        return "lazy atom slots (the device may meet atoms without a slot)";
    const plan::Compiled& C = task.compiled();
    const u32 wi = config.witness_pruning ? 0 : 1;
    for (const plan::Schema& ps : C.schemas)
    {
        const plan::Matcher& m = ps.pre[wi];
        if (!ps.ces.empty() || m.total > lifted::k_max_depth || m.steps.size() > lifted::k_max_depth ||
            ps.arity > lifted::k_max_label || (m.use_fc && C.ow > lifted::k_max_fc_ow))
            return "a schema on the CPU fallback (conditional effects or a matcher beyond the device limits)";
        if (config.canonical_order && ps.arity > 1 && !m.binds_in_order && !lifted::pick_keys_fit(ps.arity, C.num_objects))
            return "canonical-order keys wider than 64 bits";
    }
    return {};
}

/// Schema k's rank table in instance view t (lifted::CountCache::rank_table: the size of its canonical rank space, then
/// [arity, num_objects] the objects' shares of a binding's rank), or empty when the rank space holds fewer than
/// k_rank_min or more than k_rank_bits ranks. The ranks are lifted_device.cuh's RankSpace's (the pick decodes them with it): over the
/// matcher's static domains (dom0), parameter 0 the most significant.
std::vector<u32> rank_table(const rl::dev::TaskView& t, u32 k, bool witness, u32 arity)
{
    namespace d = rl::dev;
    const u32* mx = t.matcher + u64{t.schema[u64{k} * d::k_sc_count + (witness ? d::k_sc_pre0 : d::k_sc_pre1)]} * d::k_mc_count;
    const u64* dom = t.dom0 + mx[d::k_mc_dom0];
    auto in = [&](u32 i, u32 o) { return o / 64 < t.ow && ((dom[u64{i} * t.ow + o / 64] >> (o % 64)) & 1) != 0; };
    std::vector<u64> radix(arity, 0);
    u64 size = 1;
    for (u32 i = 0; i < arity && size <= k_rank_bits; ++i)
    {
        for (u32 w = 0; w < t.ow; ++w)
            radix[i] += static_cast<u64>(std::popcount(dom[u64{i} * t.ow + w]));
        size *= radix[i];
    }
    if (size < k_rank_min || size > k_rank_bits)
        return {};
    std::vector<u32> table(1 + u64{arity} * t.num_objects, lifted::k_rank_outside);
    table[0] = static_cast<u32>(size);
    u64 stride = size;
    for (u32 i = 0; i < arity; ++i)
    {
        stride /= radix[i];
        u32 r = 0;
        for (u32 o = 0; o < t.num_objects; ++o)
            if (in(i, o))
                table[1 + u64{i} * t.num_objects + o] = static_cast<u32>(r++ * stride);
    }
    return table;
}

/// Why the fast path does not run a table (empty: it does), for its first instance it does not run.
std::string table_fast_why(const rl::TaskTable& table, const rl::EnvConfig& config)
{
    for (u32 i = 0; i < table.size(); ++i)
        if (const std::string why = fast_why(*table.task(i), config); !why.empty())
            return table.size() > 1 ? "instance " + std::to_string(i) + ": " + why : why;
    return {};
}
}  // namespace

struct DeviceEnv::Impl
{
    /// One domain's table on the fast path (a table: the only one): its device table, launch plan and count cache.
    struct Part
    {
        rl::TaskTablePtr table;
        DeviceTaskTablePtr dt;
        bool multi = false;         // the multi-instance kernels (several instances, or a domain of a suite)
        detail::TableLaunch plan;   // multi
        detail::Fanout fan;         // multi: the launches over the plan's row groups, concurrently
        u32 regions = 0, seg_cap = 0;  // lifted::CountCache (regions 0: no schema is recorded)
        u64 C0 = 0;                 // its count cache words (lifted::cache_stride(S, regions, seg_cap))
        DeviceBuffer region;        // [instances, S] recorded regions
        DeviceBuffer rank, rank_table;  // [instances, S] the offsets of the rank tables (k_none: keys), and the tables
        DeviceBuffer sets, fc_of;   // a single instance: its placement's schema sets and matcher kinds
        u32 n_fixed = 0, n_fc = 0;
        u32 key_base = 0;           // the env's launch-order key of its plan's key 0 (several domains: the keys before)
    };

    ContextPtr ctx;
    rl::TaskSuitePtr suite;
    rl::EnvConfig cfg;
    bool fast = false;
    u32 I = 1, D = 1, W = 1, L = 1, S = 0;  // instances (global), domains, row words, label width, segments per state
    u64 V = 0;
    u64 C0 = 0;  // words of the count cache proper (the most any domain's layout takes)
    u64 C = 0;   // columns of EnvBatch::counts: C0, then the launch order's two over several instances
    u64 chunk_rows = 0;
    BucketLaunch launch = BucketLaunch::Widest;
    std::vector<u32> init_count;          // per instance
    std::vector<Part> parts;              // fast path: per domain
    std::unique_ptr<SuiteExpander> x;     // general path
    cudaStream_t s = nullptr;     // the stream of the calls (set_stream)
    cudaStream_t home = nullptr;  // the context's stream: the env's buffers are allocated there and used on s
    // constants on the device: per instance the initial row, count, cache and view, and the goal masks
    DeviceBuffer init_row, init_counts, init_cache, init_view, goal_pos, goal_neg;
    // the fast path's per-instance view words and goal words (envk::Instances)
    DeviceBuffer inst_view_words, goal_off, goal_word, goal_wpos, goal_wneg, goal_never;
    bool mask_goals = false;
    // several domains: the launch-order key of every instance (its rank by domain, then by its table's key; in [0, I)),
    // its domain and id in the domain's table, its view (its domain table's: the views of a whole batch in one launch),
    // and [D, I] the suite's instances as each domain's ids (another domain's: ~0)
    DeviceBuffer okey, dom_of, loc_of, all_views, init_ids;
    DeviceBuffer iota;  // [I] 0, 1, ..., I - 1
    DeviceBuffer starts;  // [I + 1] the launch order's key starts (a ranged() step's select writes them)
    // the RNG key and the control words (k_ctl_words u32, zeroed at construction; check_errors() clears them)
    DeviceBuffer seed, ctl;
    // scratch (grow())
    DeviceBuffer pick_schema, pick_rank, pick_row, status, goal_flag, order_temp, part_ids;
    DeviceBuffer f_succ, f_schema, f_binding, f_goal, f_offsets, f_offsets2;
    PinnedBuffer host_ctl{k_ctl_words * sizeof(u32)};
    detail::Fanout lanes;  // several domains: one stream per domain, forked from s and joined back
    // CUDA graphs: the current call is being captured; some call was (buffers replaced since then stay in `retired`:
    // graphs captured before may still use them)
    bool cap = false, captured = false;
    std::vector<DeviceBuffer> retired;

    /// Several instances: the cached launch order (the counts' two order columns).
    [[nodiscard]] bool multi() const noexcept { return I > 1; }
    [[nodiscard]] bool several() const noexcept { return D > 1; }
    /// Some launch of a step covers part of the keys (several domains, OW groups or, under the launch mode, buckets):
    /// the step's select writes the key starts for its range launches.
    [[nodiscard]] bool ranged() const
    {
        return fast && (several() || (multi() && parts[0].plan.launches(launch).size() > 1));
    }

    /// Every device buffer of the env (their frees are ordered after the work of the streams they were used on).
    [[nodiscard]] std::vector<DeviceBuffer*> buffers()
    {
        std::vector<DeviceBuffer*> v{&init_row,    &init_counts, &init_cache,  &init_view, &goal_pos,   &goal_neg,
                                     &inst_view_words, &goal_off, &goal_word, &goal_wpos, &goal_wneg,  &goal_never,
                                     &okey,        &dom_of,      &loc_of,      &all_views, &init_ids,   &iota,
                                     &starts,      &seed,        &ctl,
                                     &pick_schema, &pick_rank,   &pick_row,    &status,    &goal_flag,  &order_temp,
                                     &part_ids,    &f_succ,      &f_schema,    &f_binding, &f_goal,     &f_offsets,
                                     &f_offsets2};
        for (Part& P : parts)
            for (DeviceBuffer* b : {&P.region, &P.rank, &P.rank_table, &P.sets, &P.fc_of})
                v.push_back(b);
        return v;
    }

    /// The start of a call on stream s: whether it is being captured.
    void begin_call()
    {
        cap = detail::capturing(s);
        captured = captured || cap;
    }

    /// At least n Ts of scratch in `b` (contents undefined after growth), allocated on the context's stream for use on
    /// s. Never during a capture (a captured call allocates nothing: reserve() or a call of that size first). A buffer
    /// replaced after a capture stays alive until the env goes.
    template<class T>
    T* grow(DeviceBuffer& b, u64 n)
    {
        const u64 bytes = std::max<u64>(n, 1) * sizeof(T);
        if (b.size() >= bytes)
            return static_cast<T*>(b.data());
        if (cap)
            throw std::logic_error("mymyr: device env: a captured call needs " + std::to_string(bytes) + " bytes of "
                                   "scratch the env has not sized: reserve(rows), or run one call of this batch size "
                                   "before the capture");
        DeviceBuffer nb(ctx, std::max(bytes, b.size() + b.size() / 2), home);
        nb.record_stream(s);
        if (captured && b.data())
            retired.push_back(std::move(b));
        b = std::move(nb);
        stream_wait(s, home);
        return static_cast<T*>(b.data());
    }

    /// Throws when the current call is captured (a call that synchronizes).
    void no_capture(const char* what) const
    {
        if (cap)
            throw std::logic_error(std::string("mymyr: device env: ") + what +
                                   " cannot be captured into a CUDA graph: " +
                                   (fast ? std::string("it synchronizes")
                                         : "the general path synchronizes on the successor count"));
    }

    [[nodiscard]] u32* dctl() const { return static_cast<u32*>(ctl.data()); }
    [[nodiscard]] const u64* dseed() const { return static_cast<const u64*>(seed.data()); }
    /// The order of the env's bindings: its actions are labels (under witness pruning, with the CPU's witnesses).
    [[nodiscard]] rl::dev::MatchOrder env_order() const noexcept
    {
        return !cfg.canonical_order ? rl::dev::MatchOrder::Matcher
               : cfg.witness_pruning ? rl::dev::MatchOrder::Witness
                                     : rl::dev::MatchOrder::Free;
    }

    [[nodiscard]] lifted::SchemaSet set(const Part& P, bool fc) const
    {
        if (P.multi)
            return P.plan.set(cfg.witness_pruning, cfg.canonical_order, fc);
        const auto* base = static_cast<const u32*>(P.sets.data());
        return lifted::SchemaSet{fc ? base + P.n_fixed : base, fc ? P.n_fc : P.n_fixed, S, cfg.witness_pruning ? 1u : 0u,
                                 cfg.canonical_order ? 1u : 0u, fc ? 1u : 0u};
    }

    /// A domain's count cache of `counts` (rows `stride` words apart).
    [[nodiscard]] lifted::CountCache cache(const Part& P, u32* counts, u64 stride) const
    {
        lifted::CountCache c{counts, stride, S, P.regions, P.seg_cap, static_cast<const u32*>(P.region.data()),
                             P.multi ? S : 0};
        if (P.regions && P.rank_table.size())
        {
            c.rank = static_cast<const u32*>(P.rank.data());
            c.rank_table = static_cast<const u32*>(P.rank_table.data());
        }
        return c;
    }

    /// A domain's multi-instance launch over rows with its instance ids `inst`; `counts` (rows C apart) holds the
    /// cached launch order, or null (batch order).
    [[nodiscard]] lifted::Multi multi_of(const Part& P, const u32* inst, const u32* counts)
    {
        lifted::Multi m = P.plan.multi(*P.dt, cfg.witness_pruning, cfg.canonical_order, reinterpret_cast<const i32*>(inst));
        if (counts)
        {
            m.order = counts + C0;
            m.order_stride = C;
            m.order_bad = dctl() + k_order_bad;
        }
        return m;
    }

    /// m as a range launch over keys k of domain P's plan (lifted::Multi::starts: the key starts the step's select
    /// writes), unless they are every key: a launch over every row takes every position anyway, and the range's loads
    /// would cost each of its threads.
    void range(const Part& P, lifted::Multi& m, detail::TableLaunch::Keys k) const
    {
        if (!ranged() || (P.key_base + k.lo == 0 && P.key_base + k.hi >= I))
            return;
        m.starts = static_cast<const u32*>(starts.data());
        m.key_lo = P.key_base + k.lo;
        m.key_hi = P.key_base + k.hi;
    }

    /// The launch-order keys of the env's instances (lifted::launch_order; several instances).
    [[nodiscard]] lifted::OrderKeys order_keys() const
    {
        return several() ? lifted::OrderKeys{static_cast<const u32*>(okey.data()), I} : parts[0].plan.order_keys();
    }

    /// The rows per chunk of the views and counts: the views of a chunk stay in L2 for its counts.
    [[nodiscard]] u64 chunk() const
    {
        return chunk_rows ? chunk_rows : std::max<u64>(1024, k_view_chunk_bytes / std::max<u64>(8, V * 8));
    }

    /// Views and count caches of domain P's rows among [0, n) of `states` (the fast path), in L2-sized chunks of
    /// launch positions, on stream `ls`. The sets cover every schema (the fast path has no CPU-fallback schema), so
    /// every count is written. Several instances: row i is of instance inst[i] of the domain (0xFFFFFFFF: another
    /// domain's row, skipped), each chunk's views in one launch, then the OW groups' counts concurrently; `ordered`: in
    /// the cached launch order (counts' order columns); `ranged` (the step's, ordered): in range launches, the views
    /// over the domain's keys and each group's counts over its keys. (Each group's views and counts as one chain on its
    /// lane, so that the widest instance's counts wait for their own views only, was tried: a graph's parallel
    /// branches share the device's few hardware queues, and a chain's waiting kernel holds up the others'; not kept.)
    void part_views_and_counts(Part& P, cudaStream_t ls, const u64* states, u64 n, u64* views, u32* counts, u64 stride,
                               const u32* inst, bool ordered, bool ranged)
    {
        const u64 per = chunk();
        const lifted::SchemaSet fixed = set(P, false), fc = set(P, true);
        const auto groups = static_cast<u32>(P.plan.ows.size());
        for (u64 b = 0; b < n; b += per)
        {
            const u64 m = std::min(per, n - b);
            if (!P.multi)
            {
                const rl::dev::TaskView& t = P.dt->view(0);
                const lifted::Parents p{states + b * W, W, W, static_cast<u32>(m), nullptr, 0};
                const lifted::Views v{views + b * V, V};
                const lifted::CountCache c = cache(P, counts + b * stride, stride);
                check(lifted::launch_view(t, p, v, ls), "launch_view");
                // the first launch writes the fingerprints (a task without schemas still needs one launch for them)
                if (fixed.count || !fc.count)
                    check(lifted::launch_count_keys(t, p, v, fixed, c, true, ls), "launch_count_keys");
                if (fc.count)
                    check(lifted::launch_count_keys(t, p, v, fc, c, fixed.count == 0, ls), "launch_count_keys (fc)");
                continue;
            }
            lifted::Multi mm = multi_of(P, inst, ordered ? counts : nullptr);
            mm.first = b;
            if (ranged)
                range(P, mm, detail::TableLaunch::Keys{0, P.plan.instances});
            check(lifted::launch_view_multi(mm, lifted::Parents{states, W, W, static_cast<u32>(m), nullptr, 0},
                                            lifted::Views{views, V}, ls),
                  "launch_view_multi");
            P.fan.fork(ls, groups);
            for (u32 gi = 0; gi < groups; ++gi)
                group_counts(P, gi, P.fan.lane(ls, gi), states, b, m, views, counts, stride, inst, ordered, ranged);
            P.fan.join(ls, groups);
        }
    }

    /// The count caches of OW group gi of multi-instance domain P among launch positions [b, b + m) (their views
    /// written), on stream ls. Row i is of instance inst[i] of the domain (0xFFFFFFFF: another domain's row, skipped);
    /// `ordered`: positions in the cached launch order; `ranged`: in range launches over the group's keys.
    void group_counts(Part& P, u32 gi, cudaStream_t ls, const u64* states, u64 b, u64 m, u64* views, u32* counts,
                      u64 stride, const u32* inst, bool ordered, bool ranged)
    {
        const lifted::SchemaSet fixed = set(P, false), fc = set(P, true);
        lifted::Multi mm = multi_of(P, inst, ordered ? counts : nullptr);
        mm.first = b;
        mm.ow = P.plan.ows[gi];
        if (ranged)
            range(P, mm, P.plan.ow_keys[gi]);
        const lifted::Parents p{states, W, W, static_cast<u32>(m), nullptr, 0};
        const lifted::Views v{views, V};
        const lifted::CountCache c = cache(P, counts, stride);
        if (fixed.count || !fc.count)
            check(lifted::launch_count_keys_multi(mm, p, v, fixed, c, true, ls), "launch_count_keys_multi");
        if (fc.count && mm.ow <= lifted::k_max_fc_ow)
            check(lifted::launch_count_keys_multi(mm, p, v, fc, c, fixed.count == 0, ls), "launch_count_keys_multi (fc)");
    }

    /// Views and count caches of rows [0, n) of instances `ids` (a table's or the suite's ids; `ordered`: in the
    /// cached launch order; `ranged`: the counts in range launches, part_views_and_counts): a table's on s; several
    /// domains: in L2-sized chunks of launch positions, each chunk's views of every domain's rows in one launch (the
    /// instances' views, all_views), then each domain's counts on its own stream (dom_ids: domain d's ids at
    /// dom_ids + d * dom_stride, domain_ids()). One view launch, not one per domain after the domain's picks: the views
    /// are bandwidth-bound and ran one after another anyway, and a graph's branches of a domain's views and counts
    /// start their counts well after their views.
    void views_and_counts(const u64* states, u64 n, u64* views, u32* counts, u64 stride, const u32* ids,
                          const u32* dom_ids, u64 dom_stride, bool ordered, bool ranged)
    {
        if (!several())
        {
            part_views_and_counts(parts[0], s, states, n, views, counts, stride, ids, ordered, ranged);
            return;
        }
        const u64 per = chunk();
        for (u64 b = 0; b < n; b += per)
        {
            const u64 m = std::min(per, n - b);
            lifted::Multi mm;
            mm.views = static_cast<const rl::dev::TaskView*>(all_views.data());
            mm.inst = ids;
            if (ordered)
            {
                mm.order = counts + C0;
                mm.order_stride = C;
                mm.order_bad = dctl() + k_order_bad;
            }
            mm.first = b;
            mm.instances = I;
            check(lifted::launch_view_multi(mm, lifted::Parents{states, W, W, static_cast<u32>(m), nullptr, 0},
                                            lifted::Views{views, V}, s),
                  "launch_view_multi");
            lanes.fork(s, D);
            for (u32 d = 0; d < D; ++d)
            {
                Part& P = parts[d];
                const cudaStream_t ls = lanes.lane(s, d);
                const auto groups = static_cast<u32>(P.plan.ows.size());
                P.fan.fork(ls, groups);
                for (u32 gi = 0; gi < groups; ++gi)
                    group_counts(P, gi, P.fan.lane(ls, gi), states, b, m, views, counts, stride, dom_ids + d * dom_stride,
                                 ordered, ranged);
                P.fan.join(ls, groups);
            }
            lanes.join(s, D);
        }
    }

    /// Several domains: each domain's instance ids of the batch's rows ([D, n], suitek::launch_domain_ids).
    [[nodiscard]] const u32* domain_ids(const i32* task_ids, u64 n)
    {
        auto* out = grow<u32>(part_ids, u64{D} * n);
        check(suitek::launch_domain_ids(task_ids, n, static_cast<const u32*>(dom_of.data()),
                                        static_cast<const u32*>(loc_of.data()), I, D, out, n, s),
              "launch_domain_ids");
        return out;
    }

    /// The launch order of the batch's rows (several instances): the counts' order columns; over several domains the
    /// rows are grouped by domain first.
    void order_rows(const i32* inst, u64 n, u32* counts)
    {
        if (!multi() || !fast || n == 0)
            return;
        const u64 tb = lifted::order_temp_bytes(n);
        const lifted::OrderKeys keys = order_keys();
        const auto* ids = reinterpret_cast<const u32*>(inst);
        check(lifted::launch_order(ids, I, keys, n, grow<u8>(order_temp, tb), tb, counts + C0, counts + C0 + 1, C, s),
              "launch_order");
        detail::check_order(ctx, ids, I, keys, n, counts + C0, counts + C0 + 1, C, s);
    }

    /// The step's picks of multi-instance domain P's rows (rows of other domains have inst 0xFFFFFFFF), written over
    /// the parents, on stream ls: the launch groups' concurrently, in range launches over the cached launch order (the
    /// select's key starts).
    void part_picks(Part& P, cudaStream_t ls, rl::EnvBatch& b, const u32* inst, const lifted::Picks& pk0,
                    const lifted::Labels& lab, const lifted::SuccessorRows& rows)
    {
        const lifted::Parents p{b.states, W, W, static_cast<u32>(b.rows), nullptr, 0};
        const lifted::Views v{b.views, V};
        lifted::Picks pk = pk0;
        pk.cache = cache(P, b.counts, C);
        const lifted::SchemaSet fixed = set(P, false), fc = set(P, true);
        const auto& launches = P.plan.launches(launch);
        const auto n = static_cast<u32>(launches.size());
        P.fan.fork(ls, n);
        for (u32 li = 0; li < n; ++li)
        {
            const detail::TableLaunch::Launch& l = launches[li];
            lifted::Multi mm = multi_of(P, inst, b.counts);
            mm.ow = l.ow;
            mm.words_lo = l.lo;
            mm.words_hi = l.hi;
            range(P, mm, l.keys);
            if (fixed.count)
                check(lifted::launch_pick_multi(mm, p, v, fixed, pk, lab, rows, l.wb, P.fan.lane(ls, li)),
                      "launch_pick_multi");
            if (fc.count && l.ow <= lifted::k_max_fc_ow)
                check(lifted::launch_pick_multi(mm, p, v, fc, pk, lab, rows, l.wb, P.fan.lane(ls, li)),
                      "launch_pick_multi (fc)");
        }
        P.fan.join(ls, n);
    }

    /// The picks of a one-instance table over the parents (written over them), on s.
    void single_picks(rl::EnvBatch& b, const lifted::Picks& pk0, const lifted::Labels& lab,
                      const lifted::SuccessorRows& rows)
    {
        Part& P = parts[0];
        const lifted::Parents p{b.states, W, W, static_cast<u32>(b.rows), nullptr, 0};
        const lifted::Views v{b.views, V};
        lifted::Picks pk = pk0;
        pk.fc_of = static_cast<const u8*>(P.fc_of.data());
        pk.cache = cache(P, b.counts, C);
        const lifted::SchemaSet fixed = set(P, false), fc = set(P, true);
        const rl::dev::TaskView& t = P.dt->view(0);
        if (fixed.count)
            check(lifted::launch_pick(t, p, v, fixed, pk, lab, rows, s), "launch_pick");
        if (fc.count)
            check(lifted::launch_pick(t, p, v, fc, pk, lab, rows, s), "launch_pick (fc)");
    }

    /// The instances of a batch with task ids `inst`, as the env kernels read them.
    [[nodiscard]] envk::Instances instances(i32* inst) const
    {
        envk::Instances in;
        in.count = I;
        in.inst = inst;
        in.views = fast && multi() && !several() ? parts[0].dt->device_views() : nullptr;
        in.init_row = static_cast<const u64*>(init_row.data());
        in.init_count = static_cast<const u32*>(init_counts.data());
        if (fast)
        {
            in.init_cache = static_cast<const u32*>(init_cache.data());
            in.cache_words = C0;
            in.init_view = static_cast<const u64*>(init_view.data());
            in.view_words = V;
            in.inst_view_words = static_cast<const u64*>(inst_view_words.data());
            if (mask_goals)
            {
                in.goal_off = static_cast<const u32*>(goal_off.data());
                in.goal_word = static_cast<const u32*>(goal_word.data());
                in.goal_wpos = static_cast<const u64*>(goal_wpos.data());
                in.goal_wneg = static_cast<const u64*>(goal_wneg.data());
                in.goal_never = static_cast<const u8*>(goal_never.data());
            }
        }
        in.goal_pos = static_cast<const u64*>(goal_pos.data());
        in.goal_neg = static_cast<const u64*>(goal_neg.data());
        return in;
    }

    /// General path: the offsets (and, with `full`, the rows) of the expansion of `rows` states into the flat buffers.
    rl::Expansion expand(const u64* states, const i32* inst, u64 rows, bool full, i32* offsets)
    {
        const u64 total = x->count(rl::StateBatchView{states, rows, W, 0, 0}, inst, expand_options(cfg));
        rl::Expansion e;
        e.words = W;
        e.label_width = L;
        e.offsets = offsets;
        if (full)
        {
            e.capacity = total;
            e.succ = grow<u64>(f_succ, total * W);
            e.schema = grow<i32>(f_schema, total);
            e.binding = grow<i32>(f_binding, total * L);
            e.goal = grow<u8>(f_goal, total);
        }
        x->write(e);
        if (e.words_needed > W)
            throw std::logic_error("mymyr: device env: a successor is wider than the rows (internal error)");
        return e;
    }

    void check_batch(const rl::EnvBatch& b, const rl::StepOutputs* out) const
    {
        rl::check_env(*suite, b, out);
        if (b.words != W)
            throw std::invalid_argument("mymyr: device env: state rows of " + std::to_string(b.words) + " words; the device "
                                        "env takes exactly " + std::to_string(W));
        if (fast && b.rows && (!b.counts || !b.views))
            throw std::invalid_argument("mymyr: device env: the fast path needs the cache arrays (counts [rows, " +
                                        std::to_string(C) + "], views [rows, " + std::to_string(V) + "])");
    }

    /// The fast path's constants: per domain its device table, launch plan, schema sets and recorded regions; per
    /// instance its view words, goal words and initial cache (checked against the host's initial counts).
    void setup_fast()
    {
        S = suite->max_schemas();
        parts.resize(D);
        for (u32 d = 0; d < D; ++d)
        {
            Part& P = parts[d];
            P.table = suite->table(d);
            P.dt = DeviceTaskTable::upload(ctx, P.table);
            P.dt->acquire(s);
            P.multi = several() || P.table->size() > 1;
        }
        const u32 wi = cfg.witness_pruning ? 0 : 1;
        std::vector<u64> ivw(I), wpos, wneg;
        std::vector<u32> off{0}, word;
        std::vector<u8> never(I);
        mask_goals = true;
        for (u32 g = 0; g < I; ++g)
        {
            const rl::TaskSuite::Ref r = suite->ref(g);
            const rl::dev::TaskView& tv = parts[r.domain].dt->view(r.local);
            const rl::TaskTable::Instance& in = suite->instance(g);
            ivw[g] = u64{tv.view_rows} * tv.ow;
            V = std::max<u64>(V, ivw[g]);
            never[g] = in.goal_unsatisfiable ? 1 : 0;
            mask_goals = mask_goals && !in.goal_derived;  // the fast path has no axioms
            for (u32 w = 0; w < in.goal_pos.size(); ++w)
                if (in.goal_pos[w] | in.goal_neg[w])
                {
                    word.push_back(w);
                    wpos.push_back(in.goal_pos[w]);
                    wneg.push_back(in.goal_neg[w]);
                }
            off.push_back(static_cast<u32>(word.size()));
        }
        if (several() && !mask_goals)
            throw std::logic_error("mymyr: device env: a goal with derived literals on the fast path (internal error)");
        inst_view_words = upload(ctx, ivw, s);
        goal_off = upload(ctx, off, s);
        goal_word = upload(ctx, word, s);
        goal_wpos = upload(ctx, wpos, s);
        goal_wneg = upload(ctx, wneg, s);
        goal_never = upload(ctx, never, s);
        // the recorded schemas of each instance: those whose segments the device sorts (their picks would search every
        // binding), where the region holds their rank bitmap, or their keys pay for the region's traffic
        const rl::dev::MatchOrder order = env_order();
        for (Part& P : parts)
        {
            const u32 Id = P.table->size();
            std::vector<u32> reg(u64{std::max<u32>(S, 1)} * Id, rl::dev::k_none), rank(reg.size(), rl::dev::k_none);
            std::vector<u32> tables;
            for (u32 i = 0; i < Id; ++i)
            {
                const plan::Compiled& CC = P.table->task(i)->compiled();
                const rl::dev::TaskView tv = rl::task_view(P.dt->bundle(i));
                u32 n = 0;
                for (u32 k = 0; k < S && k < CC.schemas.size(); ++k)
                {
                    const plan::Schema& ps = CC.schemas[k];
                    const plan::Matcher& m = ps.pre[wi];
                    const u32 flags = rl::dev::schema_matcher_flags(tv, k, cfg.witness_pruning);
                    const bool fc = rl::dev::device_fc(flags, order);
                    if (!cfg.canonical_order || ps.arity <= 1 || !rl::dev::device_sorts(flags, fc))
                        continue;
                    const std::vector<u32> rt = m.steps.size() >= k_rank_steps
                                                    ? rank_table(tv, k, cfg.witness_pruning, ps.arity)
                                                    : std::vector<u32>{};
                    const bool keys = (fc || m.steps.size() >= k_deep) && lifted::pick_keys_fit(ps.arity, CC.num_objects);
                    if (rt.empty() && !keys)
                        continue;
                    reg[u64{i} * S + k] = n++;
                    if (!rt.empty())
                    {
                        rank[u64{i} * S + k] = static_cast<u32>(tables.size());
                        tables.insert(tables.end(), rt.begin(), rt.end());
                    }
                }
                P.regions = std::max(P.regions, n);
            }
            if (tables.size() > std::numeric_limits<u32>::max())
                throw std::length_error("mymyr: device env: the rank tables exceed 2^32 words");
            P.seg_cap = P.regions ? k_seg_cap : 0;
            P.region = upload(ctx, reg, s);
            if (!tables.empty())
            {
                P.rank = upload(ctx, rank, s);
                P.rank_table = upload(ctx, tables, s);
            }
            P.C0 = lifted::cache_stride(S, P.regions, P.seg_cap);
            C0 = std::max(C0, P.C0);
        }
        C = multi() ? C0 + 2 : C0;
        for (Part& P : parts)
        {
            if (P.multi)
            {
                // a suite's domains lay their caches out with the suite's S (their schemas past their own: count 0)
                P.plan = detail::plan_launches(ctx, *P.dt, s, several() ? S : 0);
                // the auxiliary streams of the launch groups exist before any capture
                P.fan.reserve(static_cast<u32>(std::max({P.plan.ows.size(), P.plan.widest.size(), P.plan.per_bucket.size()})));
                continue;
            }
            // the device's matcher per schema (rl::dev::device_fc)
            const SchemaPlacement pl = ChunkGenerator::place(*P.table->task(0), cfg.witness_pruning);
            const rl::dev::TaskView tv = rl::task_view(P.dt->bundle(0));
            std::vector<u32> sv, fcs;
            for (const u32 k : pl.device)
                (rl::dev::device_fc(rl::dev::schema_matcher_flags(tv, k, cfg.witness_pruning), order) ? fcs : sv).push_back(k);
            P.n_fixed = static_cast<u32>(sv.size());
            P.n_fc = static_cast<u32>(fcs.size());
            std::vector<u8> fc(std::max<u32>(S, 1), 0);
            for (const u32 k : fcs)
                fc[k] = 1;
            sv.insert(sv.end(), fcs.begin(), fcs.end());
            P.sets = upload(ctx, sv, s);
            P.fc_of = upload(ctx, fc, s);
        }
        if (several())
        {
            // the launch order groups the rows by domain, then by the domain's own key (OW, bucket, instance): an
            // instance's key is its rank in that order, the domains' ranks one after another, so the keys span [0, I)
            // (the narrowest key range for lifted::launch_order's counting sort)
            std::vector<u32> base(D + 1, 0);
            for (u32 d = 0; d < D; ++d)
            {
                parts[d].key_base = base[d];
                base[d + 1] = base[d] + parts[d].table->size();
            }
            std::vector<u32> key(I), dom(I), loc(I);
            for (u32 g = 0; g < I; ++g)
            {
                const rl::TaskSuite::Ref r = suite->ref(g);
                dom[g] = r.domain;
                loc[g] = r.local;
                key[g] = base[r.domain] + parts[r.domain].plan.host_key[r.local];
            }
            okey = upload(ctx, key, s);
            dom_of = upload(ctx, dom, s);
            loc_of = upload(ctx, loc, s);
            std::vector<rl::dev::TaskView> tv(I);
            std::vector<u32> ids(u64{D} * I, ~u32{0});
            for (u32 g = 0; g < I; ++g)
            {
                tv[g] = parts[dom[g]].dt->view(loc[g]);
                ids[u64{dom[g]} * I + g] = loc[g];
            }
            all_views = upload(ctx, tv, s);
            init_ids = upload(ctx, ids, s);
            lanes.reserve(D);
        }
        // the key starts of either launch mode's ranged() steps (set_launch)
        if (fast && (several() || (multi() && parts[0].plan.per_bucket.size() > 1)))
            starts = DeviceBuffer(ctx, (u64{I} + 1) * sizeof(u32), s);
        // the initial states' caches (instance g's at row g; rows C0 apart, no launch order)
        init_cache = DeviceBuffer(ctx, std::max<u64>(u64{I} * C0, 1) * sizeof(u32), s);
        check(cudaMemsetAsync(init_cache.data(), 0, std::max<u64>(u64{I} * C0, 1) * sizeof(u32), s), "cudaMemsetAsync");
        init_view = DeviceBuffer(ctx, std::max<u64>(u64{I} * V, 1) * sizeof(u64), s);
        {
            std::vector<i32> ids(I);
            std::iota(ids.begin(), ids.end(), 0);
            iota = upload(ctx, ids, s);
            views_and_counts(static_cast<const u64*>(init_row.data()), I, static_cast<u64*>(init_view.data()),
                             static_cast<u32*>(init_cache.data()), C0, static_cast<const u32*>(iota.data()),
                             static_cast<const u32*>(init_ids.data()), I, false, false);
        }
        std::vector<u32> cache(u64{I} * C0);
        if (!cache.empty())
            check(cudaMemcpyAsync(cache.data(), init_cache.data(), cache.size() * sizeof(u32), cudaMemcpyDeviceToHost, s),
                  "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        for (u32 g = 0; g < I; ++g)
        {
            u64 sum = 0;
            for (u32 k = 0; k < S; ++k)
                sum += cache[u64{g} * C0 + k];
            if (sum != init_count[g])
                throw std::logic_error("mymyr: device env: the device counts " + std::to_string(sum) + " successors of the "
                                       "initial state of instance " + std::to_string(g) + ", the host " +
                                       std::to_string(init_count[g]) + " (internal error)");
        }
    }
};

std::string DeviceEnv::fast_unsupported(const rl::TaskSuite& suite, const rl::EnvConfig& config)
{
    if (suite.single_domain())
        return table_fast_why(*suite.table(0), config);
    for (u32 d = 0; d < suite.num_domains(); ++d)
    {
        std::string why = table_fast_why(*suite.table(d), config);
        if (why.empty())
            why = detail::multi_unsupported(*suite.table(d));
        if (!why.empty())
            return "domain " + std::to_string(d) + " (" + suite.domain_name(d) + "): " + why;
    }
    return {};
}

DeviceEnv::DeviceEnv(ContextPtr ctx, rl::TaskSuitePtr suite, const rl::EnvConfig& config, Path path, cudaStream_t stream)
    : m(std::make_unique<Impl>())
{
    if (!ctx || !suite)
        throw std::invalid_argument("mymyr: DeviceEnv: null context or suite");
    Impl& I = *m;
    I.ctx = std::move(ctx);
    I.suite = std::move(suite);
    I.cfg = config;
    I.home = I.ctx->stream();
    I.s = I.home;  // the setup runs on the context's stream; set_stream(stream) at the end
    if (I.suite->numeric())
        throw std::invalid_argument("mymyr: device environments with numeric fluents are not supported");
    if (const std::string why = SuiteExpander::unsupported(*I.suite); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run " + why);
    I.I = I.suite->size();
    I.D = I.suite->num_domains();
    I.W = I.suite->words();
    if (I.W > lifted::k_max_words)
        throw std::invalid_argument("mymyr: the CUDA backend cannot run this " + std::string(I.suite->noun()) +
                                    ": states of " + std::to_string(I.W) + " words (the device env takes at most " +
                                    std::to_string(lifted::k_max_words) + ")");
    I.L = std::max<u32>(1, I.suite->label_width());
    const std::string why = fast_unsupported(*I.suite, I.cfg);
    if (path == Path::Fast && !why.empty())
        throw std::invalid_argument("mymyr: device env: the fast path cannot run this " + std::string(I.suite->noun()) +
                                    ": " + why);
    I.fast = path != Path::General && why.empty();
    DeviceGuard g(I.ctx->device());
    if (detail::capturing(I.home))
        throw std::logic_error("mymyr: DeviceEnv: constructed while the context's stream captures a graph");
    // the RNG key and the control words
    I.seed = upload(I.ctx, std::vector<u64>{I.cfg.seed}, I.home);
    I.ctl = DeviceBuffer(I.ctx, k_ctl_words * sizeof(u32), I.home);
    check(cudaMemsetAsync(I.ctl.data(), 0, k_ctl_words * sizeof(u32), I.home), "cudaMemsetAsync");
    // the initial rows, their successor counts (the host engine's: both paths agree with it) and the goal masks, at the
    // suite's width (an instance's domain table may be narrower: zero words)
    {
        const rl::HostEnv host(I.suite, I.cfg);
        std::vector<u64> rows(u64{I.I} * I.W, 0), gp(u64{I.I} * I.W, 0), gn(u64{I.I} * I.W, 0);
        for (u32 i = 0; i < I.I; ++i)
        {
            const rl::TaskTable::Instance& in = I.suite->instance(i);
            const u32 w = I.suite->table_of(i).words();
            std::copy_n(in.init.begin(), w, rows.begin() + static_cast<std::ptrdiff_t>(u64{i} * I.W));
            std::copy_n(in.goal_pos.begin(), w, gp.begin() + static_cast<std::ptrdiff_t>(u64{i} * I.W));
            std::copy_n(in.goal_neg.begin(), w, gn.begin() + static_cast<std::ptrdiff_t>(u64{i} * I.W));
            I.init_count.push_back(host.initial_count(i));
        }
        I.init_row = upload(I.ctx, rows, I.s);
        I.init_counts = upload(I.ctx, I.init_count, I.s);
        I.goal_pos = upload(I.ctx, gp, I.s);
        I.goal_neg = upload(I.ctx, gn, I.s);
        check(cudaStreamSynchronize(I.s), "cudaStreamSynchronize");  // the pageable sources
    }
    if (!I.fast)
        I.x = std::make_unique<SuiteExpander>(I.ctx, I.suite, I.s);
    else
        I.setup_fast();
    set_stream(stream);
}

DeviceEnv::~DeviceEnv()
{
    if (m && m->ctx)
    {
        try
        {
            DeviceGuard g(m->ctx->device());
            check(cudaStreamSynchronize(m->s), "cudaStreamSynchronize");
        }
        catch (...)
        {
        }
    }
}

const ContextPtr& DeviceEnv::context() const noexcept { return m->ctx; }
const rl::TaskSuitePtr& DeviceEnv::suite() const noexcept { return m->suite; }
const rl::EnvConfig& DeviceEnv::config() const noexcept { return m->cfg; }
bool DeviceEnv::fast() const noexcept { return m->fast; }
cudaStream_t DeviceEnv::stream() const noexcept { return m->s; }
void DeviceEnv::set_chunk_rows(u64 rows) noexcept
{
    m->chunk_rows = rows;
    if (m->x)
        m->x->set_chunk_rows(rows);
}
void DeviceEnv::set_launch(BucketLaunch launch) noexcept
{
    m->launch = launch;
    if (m->x)
        m->x->set_launch(launch);
}
u32 DeviceEnv::words() const noexcept { return m->W; }
u32 DeviceEnv::label_width() const noexcept { return m->L; }
u32 DeviceEnv::cache_schemas() const noexcept { return m->fast ? static_cast<u32>(m->C) : 0; }
u64 DeviceEnv::cache_view_words() const noexcept { return m->fast ? m->V : 0; }
u32 DeviceEnv::cache_regions() const noexcept
{
    u32 r = 0;
    for (const Impl::Part& P : m->parts)
        r = std::max(r, P.regions);
    return m->fast ? r : 0;
}
u32 DeviceEnv::initial_count(u32 instance) const { return m->init_count.at(instance); }

void DeviceEnv::set_stream(cudaStream_t s)
{
    Impl& I = *m;
    s = s ? s : I.home;
    if (s == I.s)
        return;
    DeviceGuard g(I.ctx->device());
    // a capturing stream cannot wait on work outside its graph (the capture was begun after the work it depends on:
    // the calls before, and the upload of the device table, which the constructor waited for)
    const bool cap = detail::capturing(s);
    if (!cap)
        stream_wait(s, I.s);
    for (DeviceBuffer* b : I.buffers())
        if (b->data())
            b->record_stream(s);
    for (Impl::Part& P : I.parts)
    {
        if (!cap)
            P.dt->acquire(s);
        if (P.multi)
            P.plan.record_stream(s);
    }
    if (I.x)
        I.x->set_stream(s);
    I.s = s;
}

void DeviceEnv::set_seed(u64 seed)
{
    Impl& I = *m;
    DeviceGuard g(I.ctx->device());
    I.cfg.seed = seed;
    check(envk::launch_store(static_cast<u64*>(I.seed.data()), seed, I.s), "launch_store (seed)");
}

std::string DeviceEnv::capture_unsupported() const
{
    if (!m->fast)
        return "the general path synchronizes on the successor count (DeviceExpander): " +
               (fast_unsupported(*m->suite, m->cfg).empty() ? std::string("path 'general' was chosen")
                                                            : fast_unsupported(*m->suite, m->cfg));
    return {};
}

void DeviceEnv::reserve(u64 rows)
{
    Impl& I = *m;
    DeviceGuard g(I.ctx->device());
    I.begin_call();
    (void)I.grow<u8>(I.status, rows);
    if (I.fast)
    {
        (void)I.grow<u32>(I.pick_schema, rows);
        (void)I.grow<u32>(I.pick_rank, rows);
        if (I.multi())
            (void)I.grow<u8>(I.order_temp, lifted::order_temp_bytes(rows));
        if (I.several())
            (void)I.grow<u32>(I.part_ids, u64{I.D} * rows);
        return;
    }
    (void)I.grow<i64>(I.pick_row, rows);
    (void)I.grow<u8>(I.goal_flag, rows);
    (void)I.grow<i32>(I.f_offsets, rows + 1);
    (void)I.grow<i32>(I.f_offsets2, rows + 1);
}

void DeviceEnv::random_actions(const u64* draws, const i32* count, u64 rows, u64 first_env, u32 max_actions, i64* out)
{
    Impl& I = *m;
    DeviceGuard g(I.ctx->device());
    check(envk::launch_random_actions(draws, count, rows, I.cfg.seed, I.dseed(), first_env, max_actions, out, I.s),
          "launch_random_actions");
}

void DeviceEnv::reset(rl::EnvBatch& b, const u8* mask, i32* count, bool keep_goals)
{
    Impl& I = *m;
    I.check_batch(b, nullptr);
    DeviceGuard g(I.ctx->device());
    I.begin_call();
    envk::Reset r;
    r.rows = b.rows;
    r.words = I.W;
    r.row_words = I.W;
    r.states = b.states;
    r.steps = b.steps;
    r.mask = mask;
    r.count = count;
    r.in = I.instances(b.task_ids);
    if (I.fast)
    {
        r.cache_counts = b.counts;
        r.count_stride = I.C;
        r.views = b.views;
    }
    r.goal_pos = b.goal_pos;
    r.goal_neg = b.goal_neg;
    r.keep_goals = keep_goals ? 1 : 0;
    r.error = I.dctl() + k_err;
    check(envk::launch_reset(r, I.s), "launch_reset");
    I.order_rows(b.task_ids, b.rows, b.counts);
}

void DeviceEnv::refresh(rl::EnvBatch& b, i32* count)
{
    Impl& I = *m;
    I.check_batch(b, nullptr);
    DeviceGuard g(I.ctx->device());
    I.begin_call();
    if (I.fast)
    {
        I.order_rows(b.task_ids, b.rows, b.counts);
        const auto* ids = reinterpret_cast<const u32*>(b.task_ids);
        if (!I.several())
            I.views_and_counts(b.states, b.rows, b.views, b.counts, I.C, ids, nullptr, 0, true, false);
        else if (b.rows)
        {
            // a row may have changed its domain: the columns another domain's layout used are cleared first (a domain
            // writes the counts of its own schemas only)
            check(cudaMemset2DAsync(b.counts, I.C * sizeof(u32), 0, I.C0 * sizeof(u32), b.rows, I.s), "cudaMemset2DAsync");
            I.views_and_counts(b.states, b.rows, b.views, b.counts, I.C, ids, I.domain_ids(b.task_ids, b.rows), b.rows,
                               true, false);
        }
        if (count)
            check(envk::launch_row_counts(b.counts, I.S, I.C, nullptr, b.rows, count, I.s), "launch_row_counts");
        return;
    }
    I.no_capture("refresh");
    auto* off = I.grow<i32>(I.f_offsets2, b.rows + 1);
    (void)I.expand(b.states, b.task_ids, b.rows, false, off);
    if (count)
        check(envk::launch_row_counts(nullptr, 0, 0, off, b.rows, count, I.s), "launch_row_counts");
}

void DeviceEnv::step(rl::EnvBatch& b, const rl::StepOutputs& out, rl::Actions action, const i32* next_task_ids)
{
    Impl& I = *m;
    I.check_batch(b, &out);
    if (action.v64 && action.v32)
        throw std::invalid_argument("mymyr: env: actions given as int64 and as int32");
    if (!action.given() && b.rows && !b.draws)
        throw std::invalid_argument("mymyr: env: the random policy needs the draw counters");
    if (I.cfg.max_steps && b.rows && !b.steps)
        throw std::invalid_argument("mymyr: env: truncation (max_steps) needs the step counters");
    const u64 N = b.rows;
    if (N == 0)
        return;
    DeviceGuard g(I.ctx->device());
    I.begin_call();
    if (!I.fast)
        I.no_capture("step");
    const cudaStream_t s = I.s;
    u32* ctl = I.dctl();
    auto* status = I.grow<u8>(I.status, N);
    envk::Select sel;
    sel.rows = N;
    sel.action = action.v64;
    sel.action32 = action.v32;
    sel.draws = b.draws;
    sel.seed = I.cfg.seed;
    sel.seed_dev = I.dseed();
    sel.first_env = b.first_env;
    sel.seeds = b.seeds;
    sel.env_ids = b.env_ids;
    sel.status = status;
    envk::Finish f;
    f.rows = N;
    f.words = I.W;
    f.row_words = I.W;
    f.states = b.states;
    f.steps = b.steps;
    f.status = status;
    f.goal_pos = b.goal_pos;
    f.goal_neg = b.goal_neg;
    f.step_reward = I.cfg.step_reward;
    f.goal_reward = I.cfg.goal_reward;
    f.dead_end_reward = I.cfg.dead_end_reward;
    f.max_steps = I.cfg.max_steps;
    f.autoreset = I.cfg.autoreset ? 1 : 0;
    f.dead_end = I.cfg.dead_end == rl::DeadEnd::NoSuccessors ? 1 : 0;
    f.dead_end_terminal = I.cfg.dead_end_terminal ? 1 : 0;
    f.in = I.instances(b.task_ids);
    f.next_task_ids = next_task_ids;
    f.error = ctl + k_err;
    f.reward = out.reward;
    f.terminated = out.terminated;
    f.truncated = out.truncated;
    f.goal = out.goal;
    f.invalid = out.invalid;
    f.count = out.count;
    f.final_states = out.final_states;
    f.schema = out.schema;
    f.binding = out.binding;
    f.label_width = out.label_width;
    if (I.fast)
    {
        // 1. choose: (schema, rank) from the cached counts (and, over several instances, check the launch order)
        sel.counts = b.counts;
        sel.num_schemas = I.S;
        sel.count_stride = I.C;
        sel.pick_schema = I.grow<u32>(I.pick_schema, N);
        sel.pick_rank = I.grow<u32>(I.pick_rank, N);
        if (I.multi())
        {
            sel.order_col = static_cast<u32>(I.C0);
            sel.order_bad = ctl + k_order_bad;
            if (I.ranged())
            {
                const lifted::OrderKeys keys = I.order_keys();
                sel.inst = reinterpret_cast<const u32*>(b.task_ids);
                sel.key = keys.key;
                sel.instances = I.I;
                sel.keys = keys.count;
                sel.starts = static_cast<u32*>(I.starts.data());
            }
        }
        check(envk::launch_select(sel, s), "launch_select");
        // 2. the picked successors over the parents, with their labels. (Sorting the parents by schema first, so that
        //    a warp's lanes run one matcher, was tried: it cost more on some domains and gained little elsewhere; not
        //    kept.)
        lifted::Picks pk{sel.pick_schema, sel.pick_rank, nullptr, ctl + k_err, {}};
        lifted::Labels lab;
        lab.binding = reinterpret_cast<u32*>(out.binding);
        lab.schema = reinterpret_cast<u32*>(out.schema);
        lab.capacity = N;
        lab.label_width = out.binding ? out.label_width : I.L;
        lifted::SuccessorRows rows;
        rows.words = b.states;
        rows.out_words = I.W;
        rows.words_needed = ctl + k_words_needed;
        if (!I.multi())
        {
            I.single_picks(b, pk, lab, rows);
            // 3. the reached states' views and counts (the cache of the next step)
            I.views_and_counts(b.states, N, b.views, b.counts, I.C, reinterpret_cast<const u32*>(b.task_ids), nullptr,
                               0, true, false);
        }
        else if (!I.several())
        {
            // 2 and 3 over several instances: in range launches (the select's key starts)
            const auto* ids = reinterpret_cast<const u32*>(b.task_ids);
            I.part_picks(I.parts[0], s, b, ids, pk, lab, rows);
            I.views_and_counts(b.states, N, b.views, b.counts, I.C, ids, nullptr, 0, true, true);
        }
        else
        {
            // over several domains: each domain's picks on its own stream (its rows are disjoint from the others'), then
            // the views of every row in one launch and each domain's counts on its stream
            const u32* ids = I.domain_ids(b.task_ids, N);
            I.lanes.fork(s, I.D);
            for (u32 d = 0; d < I.D; ++d)
                I.part_picks(I.parts[d], I.lanes.lane(s, d), b, ids + d * N, pk, lab, rows);
            I.lanes.join(s, I.D);
            I.views_and_counts(b.states, N, b.views, b.counts, I.C, reinterpret_cast<const u32*>(b.task_ids), ids, N,
                               true, true);
        }
        // 4. finish: goal test on the device; autoresets copy their instance's initial cache
        f.counts = b.counts;
        f.num_schemas = I.S;
        f.count_stride = I.C;
        f.cache_counts = b.counts;
        f.views = b.views;
        if (I.multi())
            f.order_bad = ctl + k_order_bad;
        check(envk::launch_finish(I.several() ? rl::dev::TaskView{} : I.parts[0].dt->view(0), f, s), "launch_finish");
        // autoresets into other instances regroup the rows
        if (next_task_ids)
            I.order_rows(b.task_ids, N, b.counts);
        return;
    }
    // general path: 1. the flat expansion of the current states (canonical order, goal flags)
    auto* off = I.grow<i32>(I.f_offsets, N + 1);
    const rl::Expansion e = I.expand(b.states, b.task_ids, N, true, off);
    // 2. choose and move
    sel.offsets = off;
    sel.pick_row = I.grow<i64>(I.pick_row, N);
    check(envk::launch_select(sel, s), "launch_select");
    auto* goal = I.grow<u8>(I.goal_flag, N);
    envk::Move mv;
    mv.rows = N;
    mv.row_words = I.W;
    mv.states = b.states;
    mv.pick_row = sel.pick_row;
    mv.succ = e.succ;
    mv.succ_goal = e.goal;
    mv.succ_schema = e.schema;
    mv.succ_binding = e.binding;
    mv.flat_width = I.L;
    mv.goal_flag = goal;
    mv.schema = out.schema;
    mv.binding = out.binding;
    mv.label_width = out.label_width;
    check(envk::launch_move(mv, s), "launch_move");
    // 3. the successor counts of the reached states
    auto* off2 = I.grow<i32>(I.f_offsets2, N + 1);
    (void)I.expand(b.states, b.task_ids, N, false, off2);
    // 4. finish with the given goal flags
    f.offsets = off2;
    f.goal_flag = goal;
    check(envk::launch_finish(rl::dev::TaskView{}, f, s), "launch_finish");
}

void DeviceEnv::check_errors()
{
    Impl& I = *m;
    DeviceGuard g(I.ctx->device());
    I.begin_call();
    I.no_capture("check_errors");
    u32* d = I.dctl();
    auto* h = static_cast<u32*>(I.host_ctl.data());
    check(cudaMemcpyAsync(h, d, k_ctl_words * sizeof(u32), cudaMemcpyDeviceToHost, I.s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(I.s), "cudaStreamSynchronize");
    const u32 err = h[k_err], wide = h[k_words_needed];
    if (err || wide)
    {
        check(cudaMemsetAsync(d + k_err, 0, 2 * sizeof(u32), I.s), "cudaMemsetAsync");
        if (err & 4u)
            throw std::invalid_argument("mymyr: device env: a task id outside the " + std::string(I.suite->noun()) +
                                        "'s " + std::to_string(I.I) + " instances (reset, a batch's task ids or "
                                        "next_task_ids)");
        if (err)
            throw std::logic_error("mymyr: device env: a step found a state whose cached counts do not belong to it "
                                   "(states written without refresh())");
        throw std::logic_error("mymyr: device env: a successor was wider than the rows (internal error)");
    }
}
}  // namespace mymyr::cuda
