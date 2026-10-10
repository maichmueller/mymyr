// Many IW searches at once on the device (include/mymyr/cuda/multi_iw.hpp): the host driver of the kernels in
// cuda/src/multi_iw.cu over the lifted successor kernels (cuda/generator.hpp).
//
// A chunk is one launch sequence. Its sizes are capacities from the driver's estimates (candidates, candidates of
// the distinct parents and unseen tuples per chunk row, grown from what earlier chunks reported), the kernels count on
// the device and launch_check marks the chunk aborted when a count exceeds its capacity: every committing kernel then
// does nothing and the driver redoes the chunk at the reported sizes. The host reads the chunk's control block (a few
// u32) once, after the last kernel; the last chunk of a layer also ends the layer (segments, shuffles, exhaustion).

#include "mymyr/cuda/multi_iw.hpp"

#include "goal_program.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"
#include "mymyr/cuda/multi_iw_kernels.hpp"
#include "mymyr/cuda/state_set.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/search/goal.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

constexpr u32 k_none = miw::k_none;

/// Row width of the device state rows: fixed buckets of 1, 2, 4, 8, 16, then 32 and 64 words (as the device BrFS).
u32 bucket(u32 words)
{
    const u32 w = std::bit_ceil(std::max<u32>(words, 1));
    if (w > lifted::k_max_words)
        throw std::invalid_argument("mymyr: device IW: the states grew wider than " + std::to_string(lifted::k_max_words) +
                                    " words (the device kernels' limit); use search::iw");
    return w;
}

/// A grow-only typed device array; ensure(n, keep) preserves the first `keep` elements (growing by half at least:
/// `exact` allocates n).
template<class T>
class DevArray
{
public:
    T* ensure(const ContextPtr& ctx, u64 n, cudaStream_t s, u64 keep = 0, bool exact = false)
    {
        if (n <= m_cap && m_buf.stream() == s)
            return data();
        const u64 cap = exact ? n : std::max<u64>({n, m_cap + m_cap / 2, 64});
        DeviceBuffer nb(ctx, cap * sizeof(T), s);
        if (keep)
            check(cudaMemcpyAsync(nb.data(), m_buf.data(), std::min(keep, m_cap) * sizeof(T), cudaMemcpyDeviceToDevice, s),
                  "cudaMemcpyAsync (grow)");
        m_buf = std::move(nb);  // the old buffer is freed after the copy (stream-ordered on s)
        m_cap = cap;
        return data();
    }
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(m_buf.data()); }
    [[nodiscard]] u64 capacity() const noexcept { return m_cap; }
    [[nodiscard]] u64 bytes() const noexcept { return m_cap * sizeof(T); }

private:
    DeviceBuffer m_buf;
    u64 m_cap = 0;
};

template<class T>
void to_device(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n)
        check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (H2D)");
}
template<class T>
void to_host(T* dst, const T* src, u64 n, cudaStream_t s)
{
    if (n)
        check(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (D2H)");
}

search::SearchStatus host_status(u32 code)
{
    switch (code)
    {
        case miw::k_solved: return search::SearchStatus::Solved;
        case miw::k_out_of_states: return search::SearchStatus::OutOfStates;
        case miw::k_out_of_time: return search::SearchStatus::OutOfTime;
        default: return search::SearchStatus::Exhausted;
    }
}

// The control block, on the device and mirrored in pinned memory (bytes): the chunk's control words, its miw::Step, a
// device loop's miw::Loop, and (pinned only) a count the host path reads.
constexpr usize k_blk_step = miw::k_ctl_words * sizeof(u32);
constexpr usize k_blk_loop = k_blk_step + sizeof(miw::Step);
constexpr usize k_blk_device = k_blk_loop + sizeof(miw::Loop);
constexpr usize k_blk_count = 512;
constexpr usize k_blk_pinned = 1024;
static_assert(sizeof(miw::Step) % 8 == 0 && k_blk_device <= k_blk_count, "the control block's layout");
constexpr usize k_max_graphs = 16;

/// The key of a captured chunk: every scalar and address its launches embed.
struct Key
{
    std::vector<u64> v;
    void add(u64 x) { v.push_back(x); }
    void add(const void* p) { v.push_back(static_cast<u64>(reinterpret_cast<std::uintptr_t>(p))); }
    template<class... T>
    void all(const T&... xs)
    {
        (add(xs), ...);
    }
};

/// A capacity for `rows` chunk rows at `per_row` entries per row.
u64 cap_for(double per_row, u64 rows)
{
    const double c = std::ceil(per_row * static_cast<double>(rows));
    if (c >= 0x7FFFFFF0)
        throw std::length_error("mymyr: device IW: more than 2^31 transitions or tuples in one chunk (lower chunk_states)");
    return std::max<u64>(1, static_cast<u64>(c));
}

/// n rounded up to one of 8 steps per octave (at most 12.5% more): capacities that repeat across chunks, so that their
/// captured graphs replay.
u64 quantize(u64 n)
{
    const u64 q = std::max<u64>(1, std::bit_ceil(std::max<u64>(n, 1)) / 8);
    return (n + q - 1) / q * q;
}

/// Rows [first, first + count) of a host array of rows [n, w] from `src` (count rows of src_w words); the array widens
/// to src_w first when it is narrower.
void put_rows(std::vector<u64>& a, u32& w, u32 n, u32 first, const u64* src, u32 count, u32 src_w)
{
    if (src_w > w)
    {
        std::vector<u64> b(u64{n} * src_w, 0);
        for (u64 k = 0; k < n && w; ++k)
            std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(k * w), w, b.begin() + static_cast<std::ptrdiff_t>(k * src_w));
        a.swap(b);
        w = src_w;
    }
    for (u64 k = 0; k < count; ++k)
        std::copy_n(src + k * src_w, src_w, a.begin() + static_cast<std::ptrdiff_t>((first + k) * w));
}
}  // namespace

// ================================================================================================= MultiIwBatch

std::span<const search::IwPassStatistics> MultiIwBatch::passes(u32 i) const
{
    return {pass_stats.data() + static_cast<u64>(i) * pass_slots, num_passes.at(i)};
}

u64 MultiIwBatch::expanded(u32 i) const
{
    u64 n = 0;
    for (const search::IwPassStatistics& p : passes(i))
        n += p.expanded;
    return n;
}

u64 MultiIwBatch::generated(u32 i) const
{
    u64 n = 0;
    for (const search::IwPassStatistics& p : passes(i))
        n += p.generated;
    return n;
}

std::vector<Action> MultiIwBatch::plan(u32 i, const Task& task) const
{
    std::vector<Action> out;
    if (plan_length.at(i) < 0)
        return out;
    const u32 LW = 1 + label_width;
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    for (u64 k = plan_offsets[i]; k < plan_offsets[i + 1]; ++k)
    {
        const u32* l = plan_labels.data() + k * LW;
        const u32 arity = succ.arity(l[0]);
        std::vector<ObjectId> b(arity);
        for (u32 j = 0; j < arity; ++j)
            b[j] = ObjectId{l[1 + j]};
        out.emplace_back(SchemaId{l[0]}, std::move(b));
    }
    return out;
}

search::IwResult MultiIwBatch::result(u32 i, const Task& task, const State& start, bool costs) const
{
    search::IwResult r;
    r.status = status.at(i);
    const std::span<const search::IwPassStatistics> ps = passes(i);
    r.passes.assign(ps.begin(), ps.end());
    for (const search::IwPassStatistics& p : r.passes)
    {
        const search::SearchStatistics s = p.statistics();
        r.total.expanded += s.expanded;
        r.total.generated += s.generated;
        r.total.states += s.states;
        r.total.pruned += s.pruned;
        r.total.seconds += s.seconds;
    }
    r.effective_width = effective_width.at(i);
    r.message = message;
    r.fluent_slots = task.atoms().fluent_slots();
    if (r.status == search::SearchStatus::Solved)
    {
        r.plan = plan(i, task);
        const u64* row = goal_rows.data() + static_cast<u64>(i) * (words + numeric_words);
        r.goal_state = State(row, words, numeric_words ? row + words : nullptr, task.numeric_words());
        if (costs)
        {
            const heuristics::ActionCosts c(task);
            const WorkspaceLease lease = task.workspace();
            r.cost = heuristics::plan_metric(lease->successors(), c, start, c.initial(start.view()), r.plan);
        }
    }
    return r;
}

// ================================================================================================= the driver

std::string multi_iw_unsupported(const Task& task, const MultiIwOptions& o)
{
    if (std::string why = ChunkGenerator::unsupported(task); !why.empty())
        return why;
    if (o.max_arity > 2)
        return "IW arity " + std::to_string(o.max_arity) + " (the device runs arities 0, 1 and 2)";
    return {};
}

struct DeviceMultiIw::Impl
{
    ContextPtr ctx;
    TaskPtr task;
    MultiIwOptions o;
    cudaStream_t s = nullptr;
    std::unique_ptr<ChunkGenerator> gen;
    u32 S = 0, L = 1, LW = 2;
    u32 W = 1;   // state row words
    u32 RW = 1;  // table row words
    bool host_goal = false;  // goals evaluated outside the captured chunk
    bool cpu_goal = false;   // ... tested on the CPU (the axioms are not on the device): host mirrors of the rows
    bool needs_host = false;
    u64 chunk_cap = 0;
    // layer entries per chunk (exact novelty): chunk_cap, lowered where a chunk's unseen tuples exceed emit_budget (the
    // following chunks start there instead of being redone one by one) and doubled back while they stay under a quarter
    u64 row_cap = 0;
    // capacity estimates per chunk row: the largest ratio seen, with a margin; a chunk over them is redone
    double est_cand = 0, est_ucand = 0, est_emit = 0;
    // distinct rows per chunk row over the chunks of k_ratio_rows rows or more, and the most distinct rows of a
    // chunk (dedup; none before the first chunk: the generator is sized for every gathered row)
    double est_distinct = 0;
    u64 most_distinct = 0;
    bool randomized = false;
    bool per_search_goals = false;
    MultiIwStats st;
    Clock::time_point deadline{};
    bool timed = false;
    u32 layer_running = 0;  // searches running after the last layer end

    // nodes of the current pass
    DevArray<u64> n_rows;
    DevArray<u32> n_parent, n_search, n_label;
    DevArray<u8> n_flags;
    u64 n_cap = 0;
    u32 n_tail = 0;
    DevArray<u32> order_a, order_b;
    // per search (group-local)
    u32 B = 0;
    DevArray<u32> sa_u32;  // the u32 arrays, B each
    DevArray<unsigned long long> sa_u64;
    DevArray<u64> rng, goal_pos, goal_neg, reached, table;
    miw::Searches sv;
    // chunk scratch
    DevArray<u8> row_flags, host_goal_dev, c_flags;
    DevArray<u64> scan_ws;  // the scans' tile states (miw::ScanState sites; chunk_begin zeroes their flags)
    u64 scan_tiles = 0;     // per site
    DevArray<u32> live_scan, gen_scan;
    DevArray<u64> g_rows, c_rows;
    DevArray<u32> g_node, g_crow, g_search, c_parent, c_schema, c_binding, c_src;
    DevArray<u32> e_count, e_offset, e_slot;
    DevArray<u64> map_keys;
    DevArray<u32> map_vals;
    DevArray<u64> root_slots;
    DevArray<u32> root_owner;
    DevArray<u64> scan_pair, kept_pair;  // (admitted, entries) scans, packed
    DevArray<u64> blk_dev;  // the control block (k_blk_*)
    // parent dedup: the set, the distinct rows and their candidates, the candidates per gathered row
    DevArray<u32> d_table, d_owner, d_rank, d_map, u_schema, u_binding, row_off;
    DevArray<u8> d_flags;
    DevArray<u64> u_rows, u_cand;
    PinnedBuffer ctl{k_blk_pinned};  // the control block's mirror
    PinnedBuffer stage;  // per-pass results
    // host mirrors (goals over derived atoms, CPU-fallback parents)
    std::vector<u64> h_rows;
    std::vector<u32> h_order;
    std::vector<u64> h_gathered;
    std::vector<u8> h_goal;
    std::vector<u32> h_search;
    std::vector<search::GoalSpec::AtomGoal> h_goals;
    std::unique_ptr<detail::GoalPrograms> goal_programs;

    Impl(ContextPtr c, TaskPtr t, const MultiIwOptions& opts) : ctx(std::move(c)), task(std::move(t)), o(opts) {}

    [[nodiscard]] bool out_of_time() const { return timed && Clock::now() >= deadline; }
    [[nodiscard]] u32* ctl_host() const { return static_cast<u32*>(ctl.data()); }
    [[nodiscard]] miw::Step* step_host() const { return reinterpret_cast<miw::Step*>(static_cast<u8*>(ctl.data()) + k_blk_step); }
    [[nodiscard]] miw::Loop* loop_host() const { return reinterpret_cast<miw::Loop*>(static_cast<u8*>(ctl.data()) + k_blk_loop); }
    [[nodiscard]] u32* count_host() const { return reinterpret_cast<u32*>(static_cast<u8*>(ctl.data()) + k_blk_count); }
    [[nodiscard]] u32* ctl_d() const { return reinterpret_cast<u32*>(blk_dev.data()); }
    [[nodiscard]] miw::Step* step_d() const { return reinterpret_cast<miw::Step*>(reinterpret_cast<u8*>(blk_dev.data()) + k_blk_step); }
    [[nodiscard]] miw::Loop* loop_d() const { return reinterpret_cast<miw::Loop*>(reinterpret_cast<u8*>(blk_dev.data()) + k_blk_loop); }
    void sync() { check(cudaStreamSynchronize(s), "cudaStreamSynchronize"); }
    u8* staging(u64 bytes)
    {
        if (stage.size() < bytes)
            stage = PinnedBuffer(std::max<u64>(bytes, 2 * stage.size()));
        return static_cast<u8*>(stage.data());
    }

    [[nodiscard]] miw::Nodes nodes() const
    {
        return {n_rows.data(), n_parent.data(), n_search.data(), n_label.data(), n_flags.data(), W, LW};
    }
    [[nodiscard]] miw::Table table_view(u32 arity) const { return {table.data(), arity, RW}; }

    void setup();
    void alloc_searches(u32 count);
    void ensure_nodes(u64 n);
    /// Both order lists for n entries, at one capacity (a captured chunk embeds it, and the lists swap at every
    /// layer end: with two capacities a chunk's key would alternate between runs, each shape captured twice), keeping
    /// the first keep_a / keep_b entries.
    void ensure_lists(u64 n, u64 keep_a, u64 keep_b);
    void widen(u32 nw, u32 arity);
    void grow_table(u32 arity);
    /// The scans' states for scans of up to n items (zeroed when allocated; not while a chunk is captured).
    void ensure_scan(u64 n);
    [[nodiscard]] miw::ScanState scan_state(u32 site) const { return {scan_ws.data(), scan_tiles, site}; }
    void scan(const u8* flags, u8 mask, u64 n, const u32* n_dev, u32* out, u32 site);
    /// The dedup set of `slots` slots (all k_none: filled when allocated, emptied by every chunk after its use).
    u32* dedup_table(u64 slots);
    /// Host results of a run (flat, index = search).
    struct Host
    {
        u32 goal_w = 0;
        std::vector<u64> goal;     // [n, goal_w] goal rows of the solved searches
        std::vector<u32> labels;   // plan labels of every extraction, appended ([steps, LW])
        std::vector<u64> plan_at;  // [n] first step of search i's plan in labels (~0: none extracted)
        u32 reached_w = 0;
        std::vector<u64> reached;  // [n, reached_w]
    };

    MultiIwBatch run(DeviceStarts starts, std::span<const search::GoalSpec::AtomGoal> goals, std::span<const u64> seeds);
    void run_group(DeviceStarts starts, u32 g0, u32 count, std::span<const search::GoalSpec::AtomGoal> goals,
                   std::span<const u64> seeds, MultiIwBatch& out, Host& host);
    /// One pass of the searches `roots` (group-local, ascending).
    void run_pass(DeviceStarts starts, u32 arity, u32 root_rule, const std::vector<u32>& roots, bool init_reached);
    /// Where a pass is: the layer, its first entry not expanded yet, its entries (in order_a) and the next layer's so
    /// far (in order_b).
    struct Pos
    {
        u32 layer = 0, pb = 0, entries = 0, next = 0;
    };
    /// Whether the chunks from `p` on can run in a device loop: captured chunks of one rule.
    [[nodiscard]] bool loop_ok(u32 arity, u32 root_rule, u32 layer) const;
    /// The chunks from `p` on in a device loop (a captured WHILE graph). `p`, n_tail and layer_running are where it
    /// stopped, and `done` is set when the pass ended. False when the host runs the chunk at `p` next: the loop did
    /// not run (its shape's first occurrence, or no capture), or a capture of more rows, which caps the emission at
    /// emit_budget, stopped at a one-row chunk whose unseen tuples exceed it (one row is not split).
    bool device_loop(u32 arity, Pos& p, bool& done);
    /// One chunk of the current layer (entries [pb, pb + rows) of `cur`; the last one of the layer ends it); false if it
    /// must be redone (at larger capacities, after a widening, or with fewer rows: `rows` is lowered when the chunk's
    /// unseen tuples exceed emit_budget).
    bool chunk(u32 arity, u32 rule, u32 layer, u32 pb, u32& rows, u32 n_entries, DevArray<u32>& cur, DevArray<u32>& nxt,
               u32& n_next, bool& ended);

    /// The candidates' part of a chunk: the gathered rows' candidates, from the lifted kernels over their distinct rows
    /// (dedup_parents) or over all of them, at the estimated capacities.
    struct Expansion
    {
        bool dedup = false, write = false;
        u32 cap = 0;       // gathered rows (capacity)
        u32 gen_rows = 0;  // the generator's rows (capacity): the distinct rows' with dedup
        u64 ucap = 0, mcap = 0, slots = 0;
        miw::Dedup d{};
        const u64* rows = nullptr;  // the generator's parents (the distinct rows with dedup)
        const u32* rows_dev = nullptr;
        lifted::Labels ulab{}, lab{};  // the distinct rows' labels (dedup), or the candidates'
        u64* u_words = nullptr;
        u32* off = nullptr;
        miw::Broadcast b{};
        miw::Candidates c{};
    };
    /// Sizes and allocates an expansion of `g` (capacity g.n, live *g.n_dev); `write`: rows and labels (else counts);
    /// scan_n: the largest scan of the caller.
    void build_expansion(Expansion& x, const miw::Gathered& g, bool write, u64 scan_n);
    /// Enqueues it, with the counts and overflow bits in the control block. False when the chunk must be redone at once
    /// (the CPU fallback's rows widened the states).
    bool enqueue_expansion(Expansion& x, const miw::Gathered& g, u32 arity);

    /// One chunk's launch sequence: its arrays, capacities and flags, fixed before anything is enqueued, and the
    /// key under which it replays as a captured graph.
    struct Run
    {
        u32 arity = 0, rule = 0, pcap = 0;
        bool root_rule = false, last = false, write = false, novelty = false, exact = false, cut = false, close = false;
        bool loop = false;  // a device loop's body: the layer's end is the step's (Step::last)
        miw::Limits lim{};
        miw::Admission adm{};
        miw::Chunk ch{};
        miw::Gathered g{};
        Expansion x{};
        u8* flags = nullptr;
        miw::Table t{};
        miw::Emission e{};
        miw::OwnerMap map{};
        miw::RootSet root{};
        u64 root_cap = 0;
        u64* cut_scan = nullptr;
        u64* kept = nullptr;
        miw::Compact cm{};
        u32* nxt = nullptr;
        u32 nxt_cap = 0;
        std::vector<u64> key;
    };
    Run run_;
    struct Cached
    {
        std::vector<u64> key;
        GraphExec exec;
        u32 seen = 0;
        u64 used = 0;
        bool failed = false;
    };
    std::vector<Cached> graphs;
    u64 graph_clock = 0;
    /// Sizes a chunk of `rows` rows (with r.loop: of a device loop from `p`, whose lists and nodes it sizes too).
    void build_chunk(Run& r, u32 arity, u32 rule, u32 layer, u32 rows, bool last, const Pos& p);
    /// The chunk's launches, the step's upload first and the control block's download last (captured as they are);
    /// false as enqueue_expansion.
    bool enqueue_chunk(Run& r, u32 pb);
    /// The launches of a chunk from k_rows_live on (the host chunks' and, with `loop` and its conditional handle, the
    /// device loops' body); false as enqueue_expansion.
    bool enqueue_body(Run& r, miw::Loop* loop, unsigned long long handle);
    /// The cached graph of a key (null: none yet; its first occurrence is recorded, and it is captured at the next).
    Cached* cached(const std::vector<u64>& key);
    /// After an aborted chunk of `rows` rows: the estimates, a smaller chunk when its unseen tuples exceeded emit_budget
    /// (rows lowered), lazy interning.
    void after_abort(const u32* hc, u32& rows, u32 arity);
    u64 loop_rows = 0;  // the largest layer a device loop met (its captures' row capacity)
    u64 loop_room = 1;  // nodes and entries a device loop makes room for (a chunk's bound after a k_abort_space)
    /// Rows per chunk under chunk_bytes at the estimated capacities (`emits`: with exact novelty's tuples: a slot, and
    /// up to four map slots of a key and a value each, the map being a power of two of at least twice the tuples).
    [[nodiscard]] u64 budget_rows(bool emits) const
    {
        const double cand = static_cast<double>(W) * 8 + 4.0 * L + 40, ucand = static_cast<double>(W) * 8 + 4.0 * L + 8;
        const double per = est_cand * cand + (o.dedup_parents ? est_ucand * ucand : 0) + (emits ? est_emit * (4 + 4 * 12) : 0);
        return std::max<u64>(1, static_cast<u64>(static_cast<double>(o.chunk_bytes) / std::max(per, 1.0)));
    }
    /// After a chunk (aborted or not): the estimates from its counts (those of the stages before an abort are exact).
    void learn(const u32* ctl_h, u64 rows);
    /// After an aborted chunk: lazy interning when atoms without a slot were met; true when the rows were widened.
    bool recover(const u32* ctl_h, u32 arity);
    /// Plans and goal rows of the searches solved in the last pass; returns their total steps.
    u64 extract_plans(const std::vector<u32>& solved, const u32* depth, u32 g0, Host& host);
    /// Rollouts with truncated layers: the successors of the plans' non-goal states join the reached atoms (mimir's
    /// plan extraction creates them; after extract_plans, whose offsets it reuses).
    void extract_reached(u32 solved, u64 total, u32 arity);

    DevArray<u32> roots_dev, plan_meta, plan_labels;
    DevArray<u64> goal_dev, x_rows;
    DevArray<u32> x_search;
};

DeviceMultiIw::DeviceMultiIw(ContextPtr ctx, TaskPtr task, const MultiIwOptions& options) : m(std::make_unique<Impl>(ctx, task, options))
{
    if (!m->ctx || !m->task)
        throw std::invalid_argument("mymyr: DeviceMultiIw: null context or task");
    if (const std::string why = multi_iw_unsupported(*m->task, m->o); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run these searches: " + why);
    if (m->o.chunk_states == 0)
        throw std::invalid_argument("mymyr: DeviceMultiIw: chunk_states must be at least 1");
    if (m->o.max_next_layer_states == 0)
        throw std::invalid_argument("mymyr: DeviceMultiIw: max_next_layer_states must be positive");
    m->setup();
}

DeviceMultiIw::~DeviceMultiIw()
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

const MultiIwOptions& DeviceMultiIw::options() const noexcept { return m->o; }

void DeviceMultiIw::Impl::setup()
{
    DeviceGuard guard(ctx->device());
    s = ctx->stream();
    gen = std::make_unique<ChunkGenerator>(ctx, task, s);
    S = gen->num_schemas();
    L = gen->label_width();
    LW = 1 + L;
    needs_host = gen->needs_host(o.witness_pruning);
    st.host_schemas = gen->placement(o.witness_pruning).host_count;
    int l2 = 0;
    check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device()), "cudaDeviceGetAttribute");
    const u64 dw = gen->device_axioms() ? std::max<u32>(1, bits::words_for(task->atoms().max_derived_slots())) : 0;
    const u64 per = std::max<u64>(8, (gen->view_words() + dw) * 8);
    chunk_cap = o.view_bytes ? std::max<u64>(1, o.view_bytes / per) : std::max<u64>(8192, std::max<u64>(static_cast<u64>(l2) / 2, 1) / per);
    row_cap = chunk_cap;
    est_cand = est_ucand = est_emit = std::max(o.capacity_per_row, 0.0);
    blk_dev.ensure(ctx, k_blk_device / sizeof(u64) + 1, s);
}

void DeviceMultiIw::Impl::alloc_searches(u32 count)
{
    B = count;
    constexpr u32 k_u32_arrays = 17;
    u32* a = sa_u32.ensure(ctx, u64{k_u32_arrays} * count, s);
    unsigned long long* b = sa_u64.ensure(ctx, u64{4} * count, s);
    u32** fields[k_u32_arrays] = {&sv.status,     &sv.tree,      &sv.next,       &sv.layer_cut, &sv.goal_node, &sv.goal_depth,
                                  &sv.seg_begin,  &sv.seg_end,   &sv.first_live, &sv.first_goal, &sv.first_gen, &sv.cut,
                                  &sv.cut_row,    &sv.last_live, &sv.last_gen,   nullptr,       nullptr};
    for (u32 i = 0; i < k_u32_arrays; ++i)
        if (fields[i])
            *fields[i] = a + u64{i} * count;
    sv.count = count;
    sv.expanded = b;
    sv.generated = b + count;
    sv.in_tree = b + 2 * u64{count};
    sv.skipped = b + 3 * u64{count};
}

void DeviceMultiIw::Impl::ensure_nodes(u64 n)
{
    if (n <= n_cap && n_rows.capacity() >= n_cap * W)  // (a later run may have wider rows: lazy slots)
        return;
    const u64 cap = std::max<u64>({n, n_cap + n_cap / 2, 1024});
    n_rows.ensure(ctx, cap * W, s, u64{n_tail} * W);
    n_parent.ensure(ctx, cap, s, n_tail);
    n_search.ensure(ctx, cap, s, n_tail);
    n_label.ensure(ctx, cap * LW, s, u64{n_tail} * LW);
    n_flags.ensure(ctx, cap, s, n_tail);
    n_cap = cap;
}

void DeviceMultiIw::Impl::ensure_lists(u64 n, u64 keep_a, u64 keep_b)
{
    u64 cap = std::max(order_a.capacity(), order_b.capacity());
    if (n > cap)
        cap = std::max<u64>({n, cap + cap / 2, 64});
    order_a.ensure(ctx, cap, s, keep_a, true);
    order_b.ensure(ctx, cap, s, keep_b, true);
}

void DeviceMultiIw::Impl::ensure_scan(u64 n)
{
    const u64 tiles = miw::scan_tiles(n);
    if (tiles <= scan_tiles && scan_ws.capacity() >= miw::scan_buffer_words(scan_tiles))
        return;
    scan_tiles = std::max(tiles, scan_tiles);
    u64* w = scan_ws.ensure(ctx, miw::scan_buffer_words(scan_tiles), s);
    check(cudaMemsetAsync(w, 0, miw::scan_flag_words(scan_tiles) * sizeof(u64), s), "cudaMemsetAsync");  // (a new layout)
}

void DeviceMultiIw::Impl::scan(const u8* flags, u8 mask, u64 n, const u32* n_dev, u32* out, u32 site)
{
    check(miw::launch_scan_flags(flags, mask, n, n_dev, nullptr, out, scan_state(site), s), "launch_scan_flags");
}

u32* DeviceMultiIw::Impl::dedup_table(u64 slots)
{
    const u32* before = d_table.data();
    const u64 cap = d_table.capacity();
    u32* t = d_table.ensure(ctx, slots, s);
    if (t != before || d_table.capacity() != cap)
        check(cudaMemsetAsync(t, 0xFF, d_table.bytes(), s), "cudaMemsetAsync");
    return t;
}

/// Re-lays the node rows (and the per-search reached bitsets) out at nw words, and grows the tables to the task's
/// width (lazy slots).
void DeviceMultiIw::Impl::widen(u32 nw, u32 arity)
{
    DevArray<u64> rows;
    rows.ensure(ctx, std::max<u64>(n_cap, 1) * nw, s);
    if (task->numeric_slots())
    {
        auto view = gen->view();
        view.numeric.storage = 0;
        check(numeric::launch_convert(view, n_rows.data(), W, W - task->numeric_slots(), rows.data(), nw,
                                       nw - task->numeric_slots(), n_tail, true, s), "numeric relayout");
    }
    else
        check(state_set::launch_relayout(n_rows.data(), W, rows.data(), nw, n_tail, s), "launch_relayout");
    n_rows = std::move(rows);
    if (sv.reached)
    {
        DevArray<u64> r;
        r.ensure(ctx, u64{B} * nw, s);
        check(state_set::launch_relayout(reached.data(), sv.reached_words, r.data(), nw - task->numeric_slots(), B, s), "launch_relayout");
        reached = std::move(r);
        sv.reached = reached.data();
        sv.reached_words = nw - task->numeric_slots();
    }
    if (!h_rows.empty() || cpu_goal)
    {
        std::vector<u64> h(static_cast<u64>(n_tail) * nw, 0);
        for (u64 i = 0; i < n_tail && i * W < h_rows.size(); ++i)
        {
            const u32 slots = task->numeric_slots();
            std::copy_n(h_rows.begin() + static_cast<std::ptrdiff_t>(i * W), W - slots,
                        h.begin() + static_cast<std::ptrdiff_t>(i * nw));
            std::copy_n(h_rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * W - slots), slots,
                        h.begin() + static_cast<std::ptrdiff_t>((i + 1) * nw - slots));
        }
        h_rows.swap(h);
    }
    W = nw;
    ++st.widenings;
    grow_table(arity);
}

void DeviceMultiIw::Impl::grow_table(u32 arity)
{
    const u32 tw = task->words();
    if (arity == 0 || tw <= RW)
        return;
    const u64 per = arity == 1 ? tw : u64{64} * tw * tw;
    DevArray<u64> t;
    t.ensure(ctx, u64{B} * per, s);
    check(miw::launch_table_relayout(table.data(), t.data(), arity, B, RW, tw, s), "launch_table_relayout");
    table = std::move(t);
    RW = tw;
}

MultiIwBatch DeviceMultiIw::run(DeviceStarts starts, std::span<const search::GoalSpec::AtomGoal> goals, std::span<const u64> seeds,
                                cudaStream_t stream)
{
    DeviceGuard guard(m->ctx->device());
    if (stream && stream != m->s)
        stream_wait(m->s, stream);  // the starts were written on `stream`
    MultiIwBatch out = m->run(starts, goals, seeds);
    return out;
}

MultiIwBatch DeviceMultiIw::run(std::span<const State> starts, std::span<const search::GoalSpec::AtomGoal> goals, std::span<const u64> seeds)
{
    DeviceGuard guard(m->ctx->device());
    u32 w = std::max<u32>(1, m->task->words());
    for (const State& x : starts)
    {
        if (x.numeric_words() != m->task->numeric_words())
            throw std::invalid_argument("mymyr: device IW: start state has the wrong numeric width");
        w = std::max<u32>(w, x.size_words());
    }
    w += m->task->numeric_slots();
    std::vector<u64> rows(starts.size() * static_cast<u64>(w), 0);
    for (usize i = 0; i < starts.size(); ++i)
        numeric::encode(*m->task, starts[i].view(), rows.data() + i * w, w);
    DevArray<u64> dev;
    dev.ensure(m->ctx, std::max<u64>(rows.size(), 1), m->s);
    to_device(dev.data(), rows.data(), rows.size(), m->s);
    MultiIwBatch out = m->run(DeviceStarts{dev.data(), w, w, static_cast<u32>(starts.size())}, goals, seeds);
    return out;
}


MultiIwBatch DeviceMultiIw::Impl::run(DeviceStarts starts, std::span<const search::GoalSpec::AtomGoal> goals, std::span<const u64> seeds)
{
    const auto t0 = Clock::now();
    st = MultiIwStats{};
    st.host_schemas = gen->placement(o.witness_pruning).host_count;
    const u32 n = starts.rows;
    if (!goals.empty() && goals.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(goals.size()) + " goals for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    if (!seeds.empty() && seeds.size() != n)
        throw std::invalid_argument("mymyr: device IW: " + std::to_string(seeds.size()) + " seeds for " + std::to_string(n) +
                                    " searches (pass none, or one per search)");
    if (starts.rows && (!starts.data || starts.words == 0 || (starts.stride != 0 && starts.stride < starts.words)))
        throw std::invalid_argument("mymyr: device IW: malformed start rows");
    randomized = !seeds.empty();
    per_search_goals = !goals.empty();
    const bool derived_goal = per_search_goals ? std::ranges::any_of(goals, [](const auto& g) { return g.needs_view(); })
                                             : task->compiled().goal.uses_derived;
    host_goal = per_search_goals ? std::ranges::any_of(goals, [](const auto& g) { return !g.fluent_only(); })
                                : derived_goal || task->numeric_slots();
    cpu_goal = host_goal && derived_goal && !gen->device_axioms();
    const double secs = o.budget.max_seconds;
    timed = secs < 1e15;
    if (timed)
        deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)));

    MultiIwBatch out;
    out.n = n;
    out.label_width = L;
    out.status.assign(n, search::SearchStatus::Exhausted);
    out.effective_width.assign(n, 0);
    out.plan_length.assign(n, -1);
    out.pass_slots = o.max_arity + 1;
    out.pass_stats.resize(u64{n} * out.pass_slots);
    out.num_passes.assign(n, 0);
    Host host;
    host.plan_at.assign(n, ~u64{0});
    const u32 slots = task->numeric_slots();
    if (starts.words < (starts.numeric_words ? starts.numeric_words : slots))
        throw std::invalid_argument("mymyr: device IW: start rows have no numeric block");
    const u32 input_atoms = starts.words - (starts.numeric_words ? starts.numeric_words : slots);
    W = bucket(std::max({task->words(), input_atoms, 1u}) + slots);
    DevArray<u64> numeric_starts;
    if (slots)
    {
        auto view = gen->view();
        if (!starts.numeric_words)
            view.numeric.storage = 0;
        else if (starts.numeric_words != task->numeric_words())
            throw std::invalid_argument("mymyr: device IW: start rows have the wrong numeric width");
        auto* converted = numeric_starts.ensure(ctx, std::max<u64>(u64{n} * W, 1), s);
        check(numeric::launch_convert(view, starts.data, starts.stride, input_atoms, converted, W, W - slots, n, true, s),
              "numeric starts");
        starts = DeviceStarts{converted, W, W, n};
    }
    // group size: the novelty tables of a group fit table_bytes
    const u64 tw = std::max<u32>(1, task->words());
    const u64 per = o.max_arity >= 2 ? u64{512} * tw * tw : o.max_arity == 1 ? 8 * tw : 8;
    u32 group = static_cast<u32>(std::clamp<u64>(o.table_bytes / std::max<u64>(per, 1), 1, u64{1} << 24));
    if (o.max_searches)
        group = std::min(group, o.max_searches);
    for (u32 g0 = 0; g0 < n; g0 += group)
    {
        const u32 count = std::min(group, n - g0);
        const DeviceStarts gs{starts.data + static_cast<u64>(g0) * starts.stride, starts.stride, starts.words, count};
        run_group(gs, g0, count, goals.empty() ? goals : goals.subspan(g0, count), seeds.empty() ? seeds : seeds.subspan(g0, count),
                  out, host);
        ++st.groups;
    }
    // flatten: goal rows at the final width, plans in search order
    out.words = std::max<u32>(1, task->words());
    out.numeric_words = task->numeric_words();
    out.goal_rows.assign(static_cast<u64>(n) * (out.words + out.numeric_words), 0);
    if (host.goal_w)
        for (u32 i = 0; i < n; ++i)
        {
            const u64* src = host.goal.data() + u64{i} * host.goal_w;
            u64* dst = out.goal_rows.data() + u64{i} * (out.words + out.numeric_words);
            std::copy_n(src, std::min(host.goal_w - slots, out.words), dst);
            if (slots && out.status[i] == search::SearchStatus::Solved)
            {
                const State decoded = numeric::decode(*task, src, host.goal_w);
                std::copy_n(decoded.numeric().data(), out.numeric_words, dst + out.words);
            }
        }
    out.plan_offsets.assign(static_cast<u64>(n) + 1, 0);
    for (u32 i = 0; i < n; ++i)
        out.plan_offsets[i + 1] = out.plan_offsets[i] + (host.plan_at[i] != ~u64{0} ? static_cast<u64>(out.plan_length[i]) : 0);
    out.plan_labels.resize(out.plan_offsets[n] * LW);
    for (u32 i = 0; i < n; ++i)
        if (host.plan_at[i] != ~u64{0})
            std::copy_n(host.labels.begin() + static_cast<std::ptrdiff_t>(host.plan_at[i] * LW), (out.plan_offsets[i + 1] - out.plan_offsets[i]) * LW,
                        out.plan_labels.begin() + static_cast<std::ptrdiff_t>(out.plan_offsets[i] * LW));
    if (o.track_reached)
    {
        out.reached_words = out.words;
        out.reached.assign(static_cast<u64>(n) * out.words, 0);
        if (host.reached_w)
            for (u32 i = 0; i < n; ++i)
                std::copy_n(host.reached.begin() + static_cast<std::ptrdiff_t>(u64{i} * host.reached_w), std::min(host.reached_w, out.words),
                            out.reached.begin() + static_cast<std::ptrdiff_t>(u64{i} * out.words));
    }
    st.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    st.uploads = gen->uploads();
    st.device_bytes = ctx->usage().used_high;
    out.stats = st;
    return out;
}

void DeviceMultiIw::Impl::run_group(DeviceStarts starts, u32 g0, u32 count, std::span<const search::GoalSpec::AtomGoal> goals,
                                    std::span<const u64> seeds, MultiIwBatch& out, Host& host)
{
    alloc_searches(count);
    h_goals.assign(goals.begin(), goals.end());
    goal_programs.reset();
    if (per_search_goals && host_goal)
    {
        goal_programs = std::make_unique<detail::GoalPrograms>(ctx, *task, goals, s);
        gen->refresh();
    }
    // per-search RNG streams, goals and reached sets
    sv.rng = nullptr;
    if (randomized)
    {
        sv.rng = rng.ensure(ctx, count, s);
        to_device(sv.rng, seeds.data(), count, s);  // (pageable uploads are staged before the call returns)
    }
    sv.goal_pos = sv.goal_neg = nullptr;
    sv.goal_words = 0;
    if (per_search_goals)
    {
        u32 gw = 1;
        for (const auto& g : goals)
        {
            for (SlotId x : g.positive)
                gw = std::max(gw, bits::word_of(x.v) + 1);
            for (SlotId x : g.negative)
                gw = std::max(gw, bits::word_of(x.v) + 1);
        }
        std::vector<u64> pos(static_cast<u64>(count) * gw, 0), neg(pos.size(), 0);
        for (u32 i = 0; i < count; ++i)
        {
            for (SlotId x : goals[i].positive)
                bits::set(pos.data() + u64{i} * gw, x.v);
            for (SlotId x : goals[i].negative)
                bits::set(neg.data() + u64{i} * gw, x.v);
        }
        sv.goal_pos = goal_pos.ensure(ctx, pos.size(), s);
        sv.goal_neg = goal_neg.ensure(ctx, neg.size(), s);
        to_device(goal_pos.data(), pos.data(), pos.size(), s);
        to_device(goal_neg.data(), neg.data(), neg.size(), s);
        sv.goal_words = gw;
    }
    sv.reached = nullptr;
    sv.reached_words = 0;
    const bool optimize = o.optimize_iw1 && o.max_arity == 1;
    std::vector<u32> pending(count);
    for (u32 i = 0; i < count; ++i)
        pending[i] = i;
    auto push = [&](u32 i, const search::IwPassStatistics& p)
    {
        u8& k = out.num_passes[i];
        if (k >= out.pass_slots)
            throw std::logic_error("mymyr: device IW: more passes than the ladder has (internal error)");
        out.pass_stats[u64{i} * out.pass_slots + k++] = p;
    };
    auto push_all = [&](const search::IwPassStatistics& p)
    {
        for (u32 id : pending)
            push(g0 + id, p);
    };
    if (optimize)
    {
        search::IwPassStatistics p;
        p.arity = 0;
        p.placeholder = true;
        push_all(p);
    }
    const u32 first = optimize ? 1 : 0;
    if (!per_search_goals && task->compiled().goal.unsatisfiable)
    {
        // mimir's first pass returns UNSOLVABLE before expanding anything
        search::IwPassStatistics p;
        p.arity = first;
        p.status = search::SearchStatus::Unsolvable;
        push_all(p);
        for (u32 id : pending)
            out.status[g0 + id] = search::SearchStatus::Unsolvable;
        return;
    }
    // per-pass results in pinned memory: counters [4, count] u64, then codes and depths [count] u32
    u8* staged = staging(u64{count} * (4 * sizeof(u64) + 2 * sizeof(u32)));
    auto* counters = reinterpret_cast<unsigned long long*>(staged);
    auto* codes = reinterpret_cast<u32*>(staged + u64{count} * 4 * sizeof(u64));
    u32* depth = codes + count;
    for (u32 k = first; k <= o.max_arity && !pending.empty(); ++k)
    {
        if (out_of_time())
        {
            for (u32 id : pending)
                out.status[g0 + id] = search::SearchStatus::OutOfTime;
            pending.clear();
            break;
        }
        const u32 rule = k == 0 ? miw::k_rule_zero_root : (optimize && k == 1 ? miw::k_rule_continuation : miw::k_rule_normal);
        const auto tp = Clock::now();
        if (o.track_reached && k == first)
        {
            // every search's start state is created first: its atoms are reached (the roots are copied in by
            // run_pass, which knows the rows)
            sv.reached = reached.ensure(ctx, u64{count} * W, s);
            sv.reached_words = W - task->numeric_slots();
        }
        run_pass(starts, k, rule, pending, k == first);
        to_host(counters, sv.expanded, 4 * u64{count}, s);
        to_host(codes, sv.status, count, s);
        to_host(depth, sv.goal_depth, count, s);
        sync();
        const double secs = std::chrono::duration<double>(Clock::now() - tp).count();
        ++st.passes;
        std::vector<u32> next, solved;
        for (u32 id : pending)
        {
            search::IwPassStatistics p;
            p.arity = k;
            p.status = host_status(codes[id]);
            p.expanded = counters[id];
            p.generated = counters[count + id];
            p.generated_in_tree = counters[2 * u64{count} + id];
            p.skipped = counters[3 * u64{count} + id];
            p.seconds = secs;
            push(g0 + id, p);
            if (codes[id] == miw::k_solved)
            {
                out.status[g0 + id] = search::SearchStatus::Solved;
                out.effective_width[g0 + id] = k;
                out.plan_length[g0 + id] = static_cast<i32>(depth[id]);
                solved.push_back(id);
            }
            else if (codes[id] == miw::k_exhausted)
                next.push_back(id);
            else
                out.status[g0 + id] = p.status;
        }
        const bool extraction = o.track_reached && o.max_next_layer_states != ~u32{0};
        if (!solved.empty() && (o.plans || extraction))
        {
            const u64 total = extract_plans(solved, depth, g0, host);
            if (extraction)
                extract_reached(static_cast<u32>(solved.size()), total, k);
        }
        pending.swap(next);
    }
    if (o.track_reached && sv.reached)
    {
        std::vector<u64> r(u64{count} * sv.reached_words);
        to_host(r.data(), sv.reached, r.size(), s);  // (pageable: returns when the copy is done)
        put_rows(host.reached, host.reached_w, out.n, g0, r.data(), count, sv.reached_words);
    }
}

u64 DeviceMultiIw::Impl::extract_plans(const std::vector<u32>& solved, const u32* depth, u32 g0, Host& host)
{
    const u32 n = static_cast<u32>(solved.size());
    std::vector<u32> meta(2 * u64{n});
    u64 total = 0;
    for (u32 i = 0; i < n; ++i)
    {
        meta[i] = solved[i];
        meta[n + i] = static_cast<u32>(total);
        total += depth[solved[i]];
    }
    if (total >= k_none)
        throw std::length_error("mymyr: device IW: more than 2^32 plan steps");
    u32* dm = plan_meta.ensure(ctx, meta.size(), s);
    to_device(dm, meta.data(), meta.size(), s);
    u32* labels = plan_labels.ensure(ctx, std::max<u64>(total * LW, 1), s);
    u64* rows = goal_dev.ensure(ctx, u64{n} * W, s);
    check(miw::launch_plans(nodes(), sv, miw::PlanOut{dm, dm + n, n, labels, rows, W}, s), "launch_plans");
    const u64 base = host.labels.size() / LW;
    host.labels.resize(host.labels.size() + total * LW);
    std::vector<u64> hr(u64{n} * W);
    to_host(host.labels.data() + base * LW, labels, total * LW, s);
    to_host(hr.data(), rows, hr.size(), s);  // (pageable: returns when the copy is done)
    for (u32 i = 0; i < n; ++i)
    {
        const u32 id = solved[i];
        host.plan_at[g0 + id] = base + meta[n + i];
        put_rows(host.goal, host.goal_w, static_cast<u32>(host.plan_at.size()), g0 + id, hr.data() + u64{i} * W, 1, W);
    }
    return total;
}

void DeviceMultiIw::Impl::extract_reached(u32 n, u64 total, u32 arity)
{
    if (total == 0)
        return;
    for (bool restart = true; restart;)  // redone from the start after a widening (OR-ing into reached is idempotent)
    {
        restart = false;
        u64* rows = x_rows.ensure(ctx, total * W, s);
        u32* search = x_search.ensure(ctx, total, s);
        const u32* dm = plan_meta.data();
        check(miw::launch_plan_states(nodes(), sv, miw::PlanOut{dm, dm + n, n, nullptr, nullptr, 0},
                                      miw::Gathered{rows, nullptr, nullptr, search, static_cast<u32>(total), W}, s),
              "launch_plan_states");
        for (u64 b = 0; b < total;)
        {
            const u32 P = static_cast<u32>(std::min<u64>({total - b, chunk_cap, o.chunk_states, budget_rows(false)}));
            const miw::Gathered g{rows + b * W, nullptr, nullptr, search + b, P, W};
            Expansion x;
            build_expansion(x, g, true, P);
            check(cudaMemsetAsync(ctl_d(), 0, miw::k_ctl_words * sizeof(u32), s), "cudaMemsetAsync");
            check(cudaMemsetAsync(scan_ws.data(), 0, miw::scan_flag_words(scan_tiles) * sizeof(u64), s), "cudaMemsetAsync");
            if (!enqueue_expansion(x, g, arity))
            {
                restart = true;
                break;
            }
            miw::Candidates c = x.c;
            c.flags = c_flags.ensure(ctx, c.n + 1, s);
            check(cudaMemsetAsync(c.flags, miw::k_cand_kept, c.n, s), "cudaMemsetAsync");
            check(miw::launch_reached(g, c, sv, s), "launch_reached");
            to_host(ctl_host(), ctl_d(), miw::k_ctl_words, s);
            sync();
            const u32* hc = ctl_host();
            learn(hc, P);
            if (hc[miw::k_ctl_abort])
            {
                if (recover(hc, arity))
                {
                    restart = true;
                    break;
                }
                continue;  // this chunk again, at the new capacities
            }
            if (hc[miw::k_ctl_err])
                throw std::logic_error("mymyr: device IW: a sorted segment found no scratch (internal error)");
            b += P;
        }
    }
}

void DeviceMultiIw::Impl::learn(const u32* hc, u64 rows)
{
    const double r = static_cast<double>(std::max<u64>(rows, 1));
    const u32 ab = hc[miw::k_ctl_abort];
    est_ucand = std::max(est_ucand, 1.25 * hc[miw::k_ctl_ucand] / r);
    est_cand = std::max(est_cand, 1.25 * hc[miw::k_ctl_cand] / r);
    most_distinct = std::max<u64>(most_distinct, hc[miw::k_ctl_distinct]);
    if (rows >= miw::k_ratio_rows)
        est_distinct = std::max(est_distinct, 1.25 * hc[miw::k_ctl_distinct] / r);
    if ((ab & ~u32{miw::k_abort_emit | miw::k_abort_space}) == 0)  // the emission ran
        est_emit = std::max(est_emit, 1.25 * hc[miw::k_ctl_emit] / r);
}

bool DeviceMultiIw::Impl::recover(const u32* hc, u32 arity)
{
    ++st.redone;
    if (!(hc[miw::k_ctl_abort] & miw::k_abort_missing))
        return false;
    const auto tm = Clock::now();
    (void)gen->resolve_missing();
    st.host_ms += ms_since(tm);
    if (bucket(task->words() + task->numeric_slots()) > W)
    {
        widen(bucket(task->words() + task->numeric_slots()), arity);
        return true;
    }
    grow_table(arity);
    return false;
}

void DeviceMultiIw::Impl::build_expansion(Expansion& x, const miw::Gathered& g, bool write, u64 scan_n)
{
    u32* const ctl = ctl_d();
    const u32 cap = g.n;
    const u64 P = cap;
    x.dedup = o.dedup_parents;
    x.write = write;
    x.cap = cap;
    x.mcap = quantize(cap_for(est_cand, P));
    x.ucap = quantize(cap_for(est_ucand, P));
    // with dedup the generator expands the distinct rows: at most the estimate's (the chunk aborts past it)
    x.gen_rows = cap;
    if (x.dedup && most_distinct > 0)
    {
        const u64 want = std::max<u64>({64, (most_distinct * 5 + 3) / 4, est_distinct > 0 ? cap_for(est_distinct, P) : 0});
        x.gen_rows = static_cast<u32>(std::min<u64>(P, quantize(want)));
    }
    gen->prepare(x.gen_rows);
    x.c = miw::Candidates{};
    miw::Candidates& c = x.c;
    c.num_schemas = S;
    c.label_width = L;
    c.abort = ctl + miw::k_ctl_abort;
    c.n = x.mcap;
    c.src = nullptr;
    if (write)
    {
        c.parent = c_parent.ensure(ctx, x.mcap, s);
        if (!x.dedup)
        {
            c.rows = c_rows.ensure(ctx, x.mcap * W, s);
            c.schema = c_schema.ensure(ctx, x.mcap, s);
            c.binding = c_binding.ensure(ctx, x.mcap * L, s);
        }
    }
    const u32* seg = gen->seg_offsets();  // (the generator's scratch for `cap` rows: prepare() sized it)
    if (x.dedup)
    {
        x.slots = std::bit_ceil(std::max<u64>(2 * P, 64));
        miw::Dedup& d = x.d;
        d.table = dedup_table(x.slots);
        d.mask = x.slots - 1;
        d.owner = d_owner.ensure(ctx, cap, s);
        d.flags = d_flags.ensure(ctx, cap, s);
        d.rank = d_rank.ensure(ctx, P + 1, s);
        d.map = d_map.ensure(ctx, cap, s);
        d.rows = u_rows.ensure(ctx, u64{x.gen_rows} * W, s);
        d.rows_capacity = x.gen_rows;
        d.ctl = ctl;
        x.rows = d.rows;
        x.rows_dev = ctl + miw::k_ctl_gen_rows;
        x.ulab = lifted::Labels{};
        x.ulab.label_width = L;
        x.ulab.error = ctl + miw::k_ctl_err;
        if (write)
        {
            x.ulab.binding = u_binding.ensure(ctx, x.ucap * L, s);
            x.ulab.schema = u_schema.ensure(ctx, x.ucap, s);
            x.ulab.capacity = x.ucap;
            x.u_words = u_cand.ensure(ctx, x.ucap * W, s);
        }
        x.off = row_off.ensure(ctx, P + 1, s);
        miw::Broadcast& b = x.b;
        b = miw::Broadcast{};
        b.rows = x.u_words;
        b.schema = x.ulab.schema;
        b.binding = x.ulab.binding;
        b.seg = seg;
        b.num_schemas = S;
        b.row_offsets = x.off;
        b.capacity = x.mcap;
        b.abort = c.abort;
        b.ctl = ctl;
        b.ucand_capacity = x.ucap;
        if (write)
        {
            // the candidates stay the distinct rows' (Candidates::src)
            b.out_parent = c_parent.data();
            b.out_src = c_src.ensure(ctx, x.mcap, s);
            c.rows = x.u_words;
            c.schema = x.ulab.schema;
            c.binding = x.ulab.binding;
            c.src = b.out_src;
        }
        c.seg_offsets = x.off;
        c.num_schemas = 1;
        c.n_dev = x.off + P;
    }
    else
    {
        x.rows = g.rows;
        x.rows_dev = g.n_dev;
        x.lab = lifted::Labels{};
        x.lab.label_width = L;
        x.lab.error = ctl + miw::k_ctl_err;
        if (write)
        {
            x.lab.binding = c_binding.data();
            x.lab.schema = c_schema.data();
            x.lab.parent = c_parent.data();
            x.lab.capacity = x.mcap;
        }
        c.seg_offsets = seg;
        c.n_dev = seg + P * S;
    }
    ensure_scan(std::max<u64>({scan_n, P, x.mcap}));
}

bool DeviceMultiIw::Impl::enqueue_expansion(Expansion& x, const miw::Gathered& g, u32 arity)
{
    u32* const ctl = ctl_d();
    const u64 P = x.cap;
    if (x.dedup)
    {
        // (the set is empty: dedup_compact empties the slots it used)
        check(miw::launch_dedup_insert(g, x.d, s), "launch_dedup_insert");
        check(miw::launch_dedup_owner(g, x.d, s), "launch_dedup_owner");
        scan(x.d.flags, 1, P, g.n_dev, x.d.rank, miw::k_scan_rank);
        check(miw::launch_dedup_compact(g, x.d, s), "launch_dedup_compact");
    }
    ChunkInput in{x.rows, W, W, x.gen_rows, nullptr, W, 0, x.rows_dev};
    // the CPU-fallback work runs while the device computes the views and counts (frozen slots, no CPU
    // axioms: ChunkGenerator::can_defer_host)
    const bool overlap = needs_host && gen->can_defer_host();
    if (needs_host)
    {
        // the CPU fallback needs the parents on the host: their count, then the rows (not captured: synchronizes)
        const auto th = Clock::now();
        if (in.rows_dev)
        {
            to_host(count_host(), in.rows_dev, 1, s);
            sync();
            in.rows = *count_host();
            in.rows_dev = nullptr;
        }
        h_gathered.resize(u64{in.rows} * W);
        to_host(h_gathered.data(), x.rows, h_gathered.size(), s);  // (pageable: returns when the copy is done)
        in.host_states = h_gathered.data();
        if (overlap)
            gen->begin_device(in, o.witness_pruning, o.canonical_order);
        else
            gen->begin(in, o.witness_pruning, o.canonical_order);
        st.host_ms += ms_since(th);
        if (bucket(task->words() + task->numeric_slots()) > W)
        {
            widen(bucket(task->words() + task->numeric_slots()), arity);  // the CPU fallback interned atoms beyond the rows' width
            return false;
        }
        grow_table(arity);
    }
    else
        gen->begin(in, o.witness_pruning, o.canonical_order);
    gen->views();
    if (overlap)
    {
        gen->count_device();
        const auto th = Clock::now();
        gen->host_work();
        st.host_ms += ms_since(th);
        gen->count_host();
    }
    else
        gen->count();
    const u32* seg = gen->seg_offsets();
    const u32* total = seg + u64{in.rows} * S;  // (in.rows: the capacity, or the host path's count)
    if (x.dedup)
    {
        // the distinct rows' candidates are checked after their write (which stops at the capacity), by broadcast
        if (x.write)
            gen->write(x.ulab, x.u_words, W);
        check(miw::launch_dedup_offsets(g, x.d, seg, S, x.off, scan_state(miw::k_scan_offsets), s), "launch_dedup_offsets");
        if (x.write)
        {
            x.b.ucand = total;
            check(miw::launch_broadcast(g, x.d, x.b, s), "launch_broadcast");  // (checks both totals)
        }
        else
        {
            check(miw::launch_check(miw::Check{ctl, miw::k_ctl_ucand, total, static_cast<u32>(x.ucap), miw::k_abort_ucand}, s),
                  "launch_check");
            check(miw::launch_check(miw::Check{ctl, miw::k_ctl_cand, x.off + P, static_cast<u32>(x.mcap), miw::k_abort_cand}, s),
                  "launch_check");
        }
    }
    else
    {
        x.c.n_dev = total;
        check(miw::launch_check(miw::Check{ctl, miw::k_ctl_cand, total, static_cast<u32>(x.mcap), miw::k_abort_cand}, s),
              "launch_check");
        if (x.write)
            gen->write(x.lab, c_rows.data(), W);
    }
    if (const u32* missing = gen->missing_flag())
        check(miw::launch_check(miw::Check{ctl, miw::k_ctl_missing, missing, 0, miw::k_abort_missing}, s), "launch_check");
    return true;
}

void DeviceMultiIw::Impl::run_pass(DeviceStarts starts, u32 arity, u32 root_rule, const std::vector<u32>& roots, bool init_reached)
{
    DevArray<u64> resized_starts;
    if (task->numeric_slots() && starts.words != W)
    {
        auto view = gen->view();
        view.numeric.storage = 0;
        const u32 slots = task->numeric_slots();
        auto* converted = resized_starts.ensure(ctx, std::max<u64>(u64{starts.rows} * W, 1), s);
        check(numeric::launch_convert(view, starts.data, starts.stride, starts.words - slots, converted, W,
                                      W - slots, starts.rows, true, s), "numeric starts relayout");
        starts = DeviceStarts{converted, W, W, starts.rows};
    }
    const u32 R = static_cast<u32>(roots.size());
    n_tail = 0;
    ensure_nodes(std::max<u64>(R, 1024));
    u32* rd = roots_dev.ensure(ctx, std::max<u32>(R, 1), s);
    to_device(rd, roots.data(), R, s);
    ensure_lists(std::max<u32>(R, 1), 0, 0);
    check(miw::launch_init_pass(nodes(), sv, miw::Roots{starts.data, starts.stride, starts.words, rd, R}, order_a.data(), s),
          "launch_init_pass");
    n_tail = R;
    if (init_reached && sv.reached)
    {
        // the first pass has every search as a root, in order: the roots are the start states
        if (task->numeric_slots())
            check(cudaMemcpy2DAsync(sv.reached, u64{sv.reached_words} * sizeof(u64), n_rows.data(), u64{W} * sizeof(u64),
                                    u64{sv.reached_words} * sizeof(u64), R, cudaMemcpyDeviceToDevice, s), "cudaMemcpy2DAsync");
        else
            check(cudaMemcpyAsync(sv.reached, n_rows.data(), u64{R} * W * sizeof(u64), cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync");
    }
    if (arity >= 1)
    {
        RW = std::max<u32>(1, task->words());
        const u64 per = arity == 1 ? RW : u64{64} * RW * RW;
        table.ensure(ctx, u64{B} * per, s);
        check(miw::launch_table_init(table_view(arity), nodes(), R, s), "launch_table_init");
    }
    if (cpu_goal)
    {
        h_rows.assign(u64{R} * W, 0);
        to_host(h_rows.data(), n_rows.data(), h_rows.size(), s);
        h_search.resize(R);
        to_host(h_search.data(), n_search.data(), R, s);
        h_order.resize(R);
        for (u32 i = 0; i < R; ++i)
            h_order[i] = i;
    }
    Pos p;
    p.entries = R;
    while (true)
    {
        if (out_of_time())
        {
            check(miw::launch_stop_running(sv, miw::k_out_of_time, s), "launch_stop_running");
            ++st.layers;
            break;
        }
        if (loop_ok(arity, root_rule, p.layer))
        {
            bool done = false;
            if (device_loop(arity, p, done))
            {
                if (done)
                    break;
                continue;
            }
        }
        const u32 rule = p.layer == 0 ? root_rule : (arity == 0 ? miw::k_rule_zero_below : miw::k_rule_normal);
        const bool emits = o.exact && arity >= 1;
        u32 rows = static_cast<u32>(std::min<u64>({o.chunk_states, row_cap, u64{p.entries} - p.pb, budget_rows(emits)}));
        bool ended = false;
        while (!chunk(arity, rule, p.layer, p.pb, rows, p.entries, order_a, order_b, p.next, ended))
            rows = static_cast<u32>(std::min<u64>(rows, budget_rows(emits)));  // (the estimates grew)
        p.pb += rows;
        if (!ended)
        {
            if (p.pb >= p.entries)
                throw std::logic_error("mymyr: device IW: a layer without its last chunk (internal error)");
            continue;
        }
        ++st.layers;
        if (cpu_goal)
        {
            h_order.resize(p.next);
            to_host(h_order.data(), order_b.data(), p.next, s);  // (pageable: returns when the copy is done)
        }
        std::swap(order_a, order_b);
        p = Pos{p.layer + 1, 0, p.next, 0};
        if (layer_running == 0 || p.entries == 0)
            break;
    }
}

bool DeviceMultiIw::Impl::loop_ok(u32 arity, u32 root_rule, u32 layer) const
{
    // one rule from here on (the root layer's differs, but for a normal pass), every chunk captured, no time limit
    // (checked between chunks); MYMYR_CUDA_DEVICE_LOOPS=0 replays the chunks from the host, MYMYR_CUDA_GRAPHS=0 launches
    // them (GraphExec::loops_enabled, enabled)
    const bool one_rule = layer >= 1 || (root_rule == miw::k_rule_normal && arity >= 1);
    return one_rule && o.graphs && GraphExec::loops_enabled() && !host_goal && !timed && gen->capturable(o.witness_pruning);
}

void DeviceMultiIw::Impl::build_chunk(Run& r, u32 arity, u32 rule, u32 layer, u32 rows, bool last, const Pos& p)
{
    u32* ctl = ctl_d();
    const u32 pcap = static_cast<u32>(quantize(rows));
    r.arity = arity;
    r.rule = rule;
    r.pcap = pcap;
    r.last = last;
    r.root_rule = layer == 0 && (rule == miw::k_rule_continuation || rule == miw::k_rule_zero_root);
    r.write = !(rule == miw::k_rule_zero_below && !o.track_reached);
    r.novelty = r.write && arity >= 1 && (rule == miw::k_rule_normal || rule == miw::k_rule_continuation);
    r.exact = r.novelty && o.exact;
    r.close = arity == 0 && layer == 0;
    r.lim = miw::Limits{};
    r.lim.max_expanded = o.budget.max_expanded;
    r.lim.max_states = o.budget.max_states >= k_none ? k_none : static_cast<u32>(o.budget.max_states);
    r.lim.max_next = o.max_next_layer_states;
    r.lim.max_depth = o.budget.max_depth;
    r.cut = r.write && (r.lim.max_states != k_none || r.lim.max_next != k_none);
    // the novelty kernels admit unless the root rules' flags come in between; without a cut every candidate is kept
    r.adm = miw::Admission{rule, o.width_zero == search::WidthZero::RootOnly, r.novelty && !r.root_rule, r.write && !r.cut};
    miw::Chunk& ch = r.ch;
    ch = miw::Chunk{};
    ch.step = step_d();
    ch.rows = pcap;
    ch.row_flags = row_flags.ensure(ctx, pcap, s);
    // dead entries (width 0's closed duplicates) only exist in width-0 passes: elsewhere a search's rows from its first
    // live row to its last are all live, and their ranks need no scan
    ch.live_scan = arity == 0 ? live_scan.ensure(ctx, u64{pcap} + 1, s) : nullptr;
    ch.gen_scan = gen_scan.ensure(ctx, u64{pcap} + 1, s);
    ch.host_goal = host_goal ? host_goal_dev.ensure(ctx, pcap, s) : nullptr;
    ch.abort = ctl + miw::k_ctl_abort;
    ch.ctl = ctl;
    // the generating rows: at most pcap of them, their count on the device
    r.g = miw::Gathered{g_rows.ensure(ctx, u64{pcap} * W, s), g_node.ensure(ctx, pcap, s), g_crow.ensure(ctx, pcap, s),
                        g_search.ensure(ctx, pcap, s), pcap, W, ch.gen_scan + pcap};
    build_expansion(r.x, r.g, r.write, pcap);
    const u64 M = r.x.mcap;
    r.flags = c_flags.ensure(ctx, M + 1, s);
    r.t = table_view(arity);
    r.e = miw::Emission{};
    r.map = miw::OwnerMap{};
    if (r.exact)
    {
        r.e.count = e_count.ensure(ctx, M + 1, s);
        r.e.offset = e_offset.ensure(ctx, M + 1, s);
        // (with more than one row, a chunk over emit_budget is split: no more is allocated)
        u64 ecap = quantize(cap_for(est_emit, pcap));
        if (rows > 1)
            ecap = std::min<u64>(ecap, std::max<u64>(o.emit_budget, 1));
        r.e.total = r.e.offset + M;
        r.e.capacity = ecap;
        r.e.ctl = ctl;
        r.e.slot = e_slot.ensure(ctx, ecap, s);
        if (r.x.dedup && r.write)
        {
            // the broadcast counts the candidates' unseen tuples as it writes them
            r.x.b.table = r.t;
            r.x.b.emit_count = r.e.count;
        }
        const u64 slots = std::bit_ceil(std::max<u64>(2 * ecap, 1024));
        if (map_keys.capacity() < slots)
        {
            // the whole map is empty between chunks: fresh memory is cleared, commit clears what it used
            map_keys.ensure(ctx, slots, s);
            map_vals.ensure(ctx, slots, s);
            check(cudaMemsetAsync(map_keys.data(), 0xFF, map_keys.bytes(), s), "cudaMemsetAsync");
            check(cudaMemsetAsync(map_vals.data(), 0xFF, map_vals.bytes(), s), "cudaMemsetAsync");
        }
        r.map = miw::OwnerMap{map_keys.data(), map_vals.data(), slots - 1};
    }
    r.root = miw::RootSet{};
    r.root_cap = 0;
    if (r.write && r.root_rule)
    {
        r.root_cap = std::bit_ceil(std::max<u64>(2 * M, 1024));
        r.root = miw::RootSet{root_slots.ensure(ctx, r.root_cap, s), r.root_cap - 1, root_owner.ensure(ctx, M, s)};
    }
    r.cut_scan = r.cut ? scan_pair.ensure(ctx, M + 1, s) : nullptr;
    r.kept = kept_pair.ensure(ctx, M + 1, s);
    if (r.loop)
    {
        // a device loop: the chunks that do not fit abort (compact checks; the lists swap at each layer end: the
        // step's bound is the smaller capacity), and the loop after it makes room for the chunk's bound
        const u64 room = std::min<u64>(std::max<u64>(loop_room, 1), M);
        loop_room = 1;
        if (u64{n_tail} + room >= k_none)
            throw std::length_error("mymyr: device IW: more than 2^32 nodes in one pass");
        ensure_nodes(u64{n_tail} + room);
        ensure_lists(std::max<u64>({p.entries, u64{p.next} + room, 1}), p.entries, p.next);
    }
    else
    {
        if (r.write)
            ensure_nodes(u64{n_tail} + M);
        if (r.write || last)
            ensure_lists(std::max<u64>(u64{p.next} + M, 1), p.entries, p.next);
    }
    r.cm = miw::Compact{};
    r.cm.kept = r.kept;
    r.cm.root_owner = r.root_rule ? r.root.owner : nullptr;
    r.cm.step = step_d();
    r.cm.abort = ctl + miw::k_ctl_abort;
    r.cm.reached = sv.reached;
    r.cm.reached_words = sv.reached_words;
    r.cm.table = r.t;
    r.cm.map = r.map;
    r.cm.emission = r.exact ? r.e : miw::Emission{};
    r.nxt_cap = static_cast<u32>(std::min<u64>(order_b.capacity(), k_none));
    // the key of a captured replay: every scalar and address the launches embed (the lists are the step's)
    Key k;
    k.all(arity, rule, pcap, u64{r.last}, u64{r.loop}, u64{r.root_rule}, u64{r.write}, u64{r.novelty}, u64{r.exact}, u64{r.cut},
          u64{r.close}, u64{randomized}, u64{o.width_zero == search::WidthZero::RootOnly}, u64{o.dedup_parents}, W, LW, L, S, RW, B);
    k.all(r.lim.max_expanded, r.lim.max_states, r.lim.max_next, r.lim.max_depth);
    const miw::Nodes nd = nodes();
    k.all(nd.rows, nd.parent, nd.search, nd.label, nd.flags, u64{r.nxt_cap}, ch.step, ctl, ch.row_flags, ch.live_scan, ch.gen_scan,
          ch.host_goal);
    k.all(r.g.rows, r.g.node, r.g.crow, r.g.search);
    const Expansion& x = r.x;
    k.all(x.cap, x.gen_rows, x.ucap, x.mcap, x.slots, x.d.table, x.d.owner, x.d.flags, x.d.rank, x.d.map, x.d.rows, x.u_words,
          x.ulab.binding, x.ulab.schema, x.off, x.b.emit_count, x.c.rows, x.c.parent, x.c.schema, x.c.binding, x.c.src,
          x.c.seg_offsets, x.c.n_dev, r.flags);
    k.all(r.e.count, r.e.offset, r.e.slot, r.e.capacity, r.map.keys, r.map.vals, r.map.mask, r.root.slots, r.root.owner,
          r.root_cap, r.cut_scan, r.kept, scan_ws.data(), scan_tiles);
    k.all(sa_u32.data(), sa_u64.data(), sv.rng, sv.goal_pos, sv.goal_neg, sv.goal_words, sv.reached, sv.reached_words,
          r.t.bits, r.t.arity, r.t.row_words, gen->capture_key());
    r.key.swap(k.v);
}

bool DeviceMultiIw::Impl::enqueue_chunk(Run& r, u32 pb)
{
    to_device(step_d(), step_host(), 1, s);
    check(miw::launch_chunk_begin(sv, ctl_d(), step_d(), scan_state(0), s), "launch_chunk_begin");
    miw::Chunk& ch = r.ch;
    if (host_goal && !cpu_goal)
    {
        // the rows' axioms and goal on the device (not captured: the rows and count are host values); dead entries
        // (k_dead) read as zero rows, and k_rows_live skips them
        if (goal_programs)
            gen->goal_flags(n_rows.data(), W, W, n_tail, order_a.data() + pb, step_host()->rows,
                            const_cast<u8*>(ch.host_goal), goal_programs->view(), n_search.data());
        else
            gen->goal_flags(n_rows.data(), W, W, n_tail, order_a.data() + pb, step_host()->rows, const_cast<u8*>(ch.host_goal));
    }
    else if (host_goal)
    {
        const auto th = Clock::now();
        const WorkspaceLease lease = task->workspace();
        Successors& succ = lease->successors();
        const u32 P = step_host()->rows;
        h_goal.assign(P, 0);
        for (u32 i = 0; i < P; ++i)
        {
            const u32 node = h_order[pb + i];
            if (node == miw::k_dead)
                continue;
            const u64* row = h_rows.data() + u64{node} * W;
            const State decoded = numeric::decode(*task, row, W);
            if (per_search_goals)
            {
                succ.prepare(decoded.view());
                h_goal[i] = search::holds(h_goals[h_search[node]], succ, decoded.view());
            }
            else
                h_goal[i] = succ.is_goal(decoded.view()) ? 1 : 0;
        }
        to_device(const_cast<u8*>(ch.host_goal), h_goal.data(), P, s);
        st.host_ms += ms_since(th);
    }
    if (!enqueue_body(r, nullptr, 0))
        return false;  // redo the chunk at the new width
    to_host(ctl_host(), ctl_d(), miw::k_ctl_words, s);
    return true;
}

bool DeviceMultiIw::Impl::enqueue_body(Run& r, miw::Loop* loop, unsigned long long handle)
{
    miw::Chunk& ch = r.ch;
    const u32 pcap = r.pcap;
    const u32* rows = &step_d()->rows;  // (the scans of the rows read the step's)
    check(miw::launch_rows_live(gen->view(), nodes(), sv, ch, s), "launch_rows_live");
    if (ch.live_scan)
        scan(ch.row_flags, miw::k_row_live, pcap, rows, ch.live_scan, miw::k_scan_live);
    check(miw::launch_rows_select(nodes(), sv, r.lim, ch, s), "launch_rows_select");
    scan(ch.row_flags, miw::k_row_gen, pcap, rows, ch.gen_scan, miw::k_scan_gen);
    check(miw::launch_gather(nodes(), sv, ch, r.g, s), "launch_gather");
    if (!enqueue_expansion(r.x, r.g, r.arity))
        return false;  // redo the chunk at the new width
    miw::Candidates c = r.x.c;
    c.flags = r.flags;
    const u64 M = c.n;  // the capacity
    const miw::Gathered& g = r.g;
    if (!r.novelty)
        check(cudaMemsetAsync(c.flags, 0, M + 1, s), "cudaMemsetAsync");  // (novelty sets the flags of every candidate)
    // exact novelty with admission and keep and nothing between them: the kept scan computes the flags
    const bool novel_kept = r.exact && r.adm.admit && r.adm.keep;
    if (r.exact)
    {
        if (!r.x.b.emit_count)
            check(miw::launch_emit_count(g, c, r.t, r.e, s), "launch_emit_count");
        check(miw::launch_scan_u32(r.e.count, M, c.n_dev, c.abort, r.e.offset, scan_state(miw::k_scan_emit), s),
              "launch_scan_u32");
        check(miw::launch_emit_insert(g, c, r.t, r.e, r.map, s), "launch_emit_insert");  // (checks the tuples' total)
        if (!novel_kept)
            check(miw::launch_novel_flags(c, r.e, r.map, r.adm, s), "launch_novel_flags");
    }
    else if (r.novelty)
        check(miw::launch_relaxed(g, c, r.t, step_d(), ctl_d() + miw::k_ctl_abort, r.adm, s), "launch_relaxed");
    if (r.write)
    {
        if (r.root_rule)
        {
            check(cudaMemsetAsync(r.root.slots, 0, r.root_cap * sizeof(u64), s), "cudaMemsetAsync");
            check(miw::launch_root_insert(g, c, r.root, s), "launch_root_insert");
            check(miw::launch_root_flags(g, c, r.root, s), "launch_root_flags");
        }
        if (!r.adm.admit)
            check(miw::launch_admit(c, r.adm, s), "launch_admit");
        if (r.cut)
        {
            check(miw::launch_scan_flag_pair(c.flags, miw::k_cand_adm, miw::k_cand_ent, M, c.n_dev, c.abort, r.cut_scan,
                                             scan_state(miw::k_scan_cut), s),
                  "launch_scan_flag_pair");
            check(miw::launch_cut(g, c, sv, r.lim, r.cut_scan, s), "launch_cut");
            check(miw::launch_keep(g, c, sv, r.lim, s), "launch_keep");
        }
    }
    if (novel_kept)
        check(miw::launch_novel_kept(c, r.e, r.map, r.adm, r.kept, scan_state(miw::k_scan_kept), s), "launch_novel_kept");
    else
        check(miw::launch_scan_flag_pair(c.flags, miw::k_cand_kept | miw::k_cand_adm, miw::k_cand_kept | miw::k_cand_ent, M,
                                         c.n_dev, c.abort, r.kept, scan_state(miw::k_scan_kept), s),
              "launch_scan_flag_pair");
    if (r.write)
        check(miw::launch_compact(nodes(), g, c, r.cm, s), "launch_compact");  // (and the reached atoms, the commit)
    // the counts (the kept totals into the control block); a layer's last chunk ends the layer (Step::last), a device
    // loop's advances the step
    const miw::ChunkEnd end{nodes(), randomized, r.close, step_d(), loop, handle};
    check(miw::launch_search_update(g, c, sv, r.lim, ch, r.kept, end, s), "launch_search_update");
    return true;
}

DeviceMultiIw::Impl::Cached* DeviceMultiIw::Impl::cached(const std::vector<u64>& key)
{
    for (Cached& gc : graphs)
        if (gc.key == key)
        {
            ++gc.seen;
            gc.used = ++graph_clock;
            return &gc;
        }
    if (graphs.size() >= k_max_graphs)
        graphs.erase(std::min_element(graphs.begin(), graphs.end(), [](const Cached& a, const Cached& b) { return a.used < b.used; }));
    graphs.push_back(Cached{key, GraphExec{}, 1, ++graph_clock, false});
    return nullptr;
}

void DeviceMultiIw::Impl::after_abort(const u32* hc, u32& rows, u32 arity)
{
    learn(hc, rows);
    const u32 ab = hc[miw::k_ctl_abort];
    if ((ab & miw::k_abort_emit) && !(ab & ~u32{miw::k_abort_emit}) && rows > 1 && hc[miw::k_ctl_emit] > o.emit_budget)
    {
        const double f = static_cast<double>(o.emit_budget) / static_cast<double>(hc[miw::k_ctl_emit]) / 2;
        rows = static_cast<u32>(std::max<u64>(1, static_cast<u64>(static_cast<double>(rows) * f)));
        row_cap = rows;
        ++st.splits;
    }
    (void)recover(hc, arity);
}

bool DeviceMultiIw::Impl::device_loop(u32 arity, Pos& p, bool& done)
{
    const u32 rule = p.layer == 0 ? miw::k_rule_normal : (arity == 0 ? miw::k_rule_zero_below : miw::k_rule_normal);
    const bool emits = o.exact && arity >= 1;
    const u64 limit = std::min<u64>({o.chunk_states, row_cap, budget_rows(emits)});
    // the capture's rows: the largest layer a loop met (so that the shape repeats), under the limit
    loop_rows = std::max<u64>(loop_rows, u64{p.entries} - p.pb);
    const u32 rows = static_cast<u32>(std::max<u64>(1, std::min(limit, loop_rows)));
    Run& r = run_;
    r.loop = true;
    build_chunk(r, arity, rule, p.layer, rows, false, p);
    Cached* hit = cached(r.key);
    if (!hit || hit->failed)
        return false;  // (its first occurrence runs as host chunks: one-off shapes are not captured)
    if (!hit->exec)
    {
        try
        {
            hit->exec = GraphExec::capture_while(s, [&](unsigned long long handle)
            {
                check(miw::launch_chunk_begin(sv, ctl_d(), step_d(), scan_state(0), s), "launch_chunk_begin");
                (void)enqueue_body(r, loop_d(), handle);  // (its search_update advances the step)
            });
            ++st.captures;
        }
        catch (const CudaError&)
        {
            hit->failed = true;  // (not capturable after all: host chunks from now on)
            return false;
        }
    }
    // the first chunk's step and the loop's bounds, then the loop, then its control block (one read)
    const u32 pcap = r.pcap;
    const u32 lrows = std::min<u32>(rows, pcap);
    miw::Step* sp = step_host();
    *sp = miw::Step{};
    sp->begin = p.pb;
    sp->rows = std::min<u32>(lrows, p.entries - p.pb);
    sp->node_base = n_tail;
    sp->entry_base = p.next;
    sp->layer = p.layer;
    sp->entries = p.entries;
    sp->last = u64{p.pb} + sp->rows >= p.entries ? 1 : 0;
    sp->next_cap = r.nxt_cap;
    sp->node_cap = static_cast<u32>(std::min<u64>(n_cap, k_none - 1));
    sp->cur = order_a.data();
    sp->next = order_b.data();
    miw::Loop* lp = loop_host();
    *lp = miw::Loop{};
    lp->rows = lrows;
    lp->grow_above = pcap < limit ? pcap : k_none;
    lp->emission = r.exact ? 1 : 0;
    check(cudaMemcpyAsync(step_d(), sp, k_blk_device - k_blk_step, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    hit->exec.launch(s);
    ++st.replays;
    ++st.device_loops;
    check(cudaMemcpyAsync(ctl_host(), ctl_d(), k_blk_device, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    sync();
    const miw::Loop& l = *loop_host();
    const miw::Step& e = *step_host();
    const u32* hc = ctl_host();
    st.chunks += l.chunks;
    st.candidates += l.candidates;
    st.distinct += l.distinct;
    st.nodes += l.nodes;
    st.layers += l.layers;
    est_ucand = std::max(est_ucand, 1.25 * l.ucand);
    est_cand = std::max(est_cand, 1.25 * l.cand_rows);
    est_distinct = std::max(est_distinct, 1.25 * l.distinct_rows);
    most_distinct = std::max<u64>(most_distinct, l.max_distinct);
    est_emit = std::max(est_emit, 1.25 * l.emit);
    if (l.layers)
        layer_running = l.running;
    // where it stopped: the step it would run next (an aborted chunk's own)
    if (e.cur != order_a.data())
        std::swap(order_a, order_b);
    p = Pos{e.layer, e.begin, e.entries, e.entry_base};
    n_tail = e.node_base;
    if (l.stop == miw::k_loop_err)
        throw std::logic_error("mymyr: device IW: a sorted segment found no scratch (internal error)");
    if (l.stop == miw::k_loop_abort)
    {
        if (hc[miw::k_ctl_abort] & miw::k_abort_space)
            loop_room = r.x.mcap;  // (the next loop's first chunk fits)
        u32 r_rows = e.rows;
        after_abort(hc, r_rows, arity);
        const bool emit_over = (hc[miw::k_ctl_abort] & miw::k_abort_emit) && hc[miw::k_ctl_emit] > o.emit_budget;
        if (rows > 1 && e.rows <= 1 && emit_over)
        {
            // one row over emit_budget under a capture of more rows (a layer's last entry): one row is not split,
            // and the captures of more rows cap its emission at emit_budget (build_chunk), so no loop runs it. The
            // next loop would capture as many rows and, once its key recurs, abort on the same row again: a livelock.
            // The host runs the row alone, its emission sized from the estimates (learnt: at least 1.25 x its tuples).
            ++st.loop_handoffs;
            return false;
        }
    }
    if (l.stop == miw::k_loop_grow)
        loop_rows = std::max<u64>(loop_rows, p.entries);
    if (emits && lrows >= row_cap && l.max_emit < o.emit_budget / 4)
        row_cap = std::min(chunk_cap, row_cap * 2);
    done = l.stop == miw::k_loop_done;
    return true;
}

bool DeviceMultiIw::Impl::chunk(u32 arity, u32 rule, u32 layer, u32 pb, u32& rows, u32 n_entries, DevArray<u32>& cur,
                                DevArray<u32>& nxt, u32& n_next, bool& ended)
{
    const u32 P = rows;
    const bool last = u64{pb} + P >= n_entries;
    run_.loop = false;
    build_chunk(run_, arity, rule, layer, P, last, Pos{layer, pb, n_entries, n_next});
    miw::Step* sp = step_host();
    *sp = miw::Step{};
    sp->begin = pb;
    sp->rows = P;
    sp->node_base = n_tail;
    sp->entry_base = n_next;
    sp->layer = layer;
    sp->entries = n_entries;
    sp->last = last ? 1 : 0;
    sp->next_cap = run_.nxt_cap;
    sp->node_cap = static_cast<u32>(std::min<u64>(n_cap, k_none - 1));
    sp->cur = cur.data();
    sp->next = nxt.data();
    const bool graph = o.graphs && GraphExec::enabled() && !host_goal && gen->capturable(o.witness_pruning);
    bool enqueued = false;
    if (graph)
    {
        // a chunk shape seen before replays as a graph (captured at its second occurrence: one-off shapes run eagerly)
        Cached* hit = cached(run_.key);
        if (hit && !hit->exec && !hit->failed)
        {
            try
            {
                hit->exec = GraphExec::capture(s, [&] { (void)enqueue_chunk(run_, pb); });
                ++st.captures;
            }
            catch (const CudaError&)
            {
                hit->failed = true;  // (not capturable after all: eager from now on)
            }
        }
        if (hit && hit->exec)
        {
            hit->exec.launch(s);
            ++st.replays;
            enqueued = true;
        }
    }
    if (!enqueued && !enqueue_chunk(run_, pb))
        return false;  // redo the chunk at the new width
    sync();
    const u32* hc = ctl_host();
    if (hc[miw::k_ctl_abort])
    {
        after_abort(hc, rows, arity);
        return false;  // nothing was committed
    }
    learn(hc, P);
    if (run_.write && hc[miw::k_ctl_err])
        throw std::logic_error("mymyr: device IW: a sorted segment found no scratch (internal error)");
    if (run_.exact && rows >= row_cap && hc[miw::k_ctl_emit] < o.emit_budget / 4)
        row_cap = std::min(chunk_cap, row_cap * 2);
    const u32 n_adm = hc[miw::k_ctl_adm], n_ent = hc[miw::k_ctl_ent];
    if (cpu_goal && n_adm)
    {
        h_rows.resize(u64{n_tail + n_adm} * W);
        to_host(h_rows.data() + u64{n_tail} * W, n_rows.data() + u64{n_tail} * W, u64{n_adm} * W, s);
        h_search.resize(u64{n_tail} + n_adm);
        to_host(h_search.data() + n_tail, n_search.data() + n_tail, n_adm, s);
    }
    n_tail += n_adm;
    n_next += n_ent;
    ++st.chunks;
    st.candidates += hc[miw::k_ctl_cand];
    st.distinct += hc[miw::k_ctl_distinct];
    st.nodes += n_adm;
    if (last)
    {
        ended = true;
        layer_running = hc[miw::k_ctl_running];
    }
    return true;
}

// ================================================================================================= free functions

std::vector<search::IwResult> multi_iw(ContextPtr ctx, TaskPtr task, std::span<const State> starts, const MultiIwOptions& options,
                                       std::span<const search::GoalSpec::AtomGoal> goals)
{
    DeviceMultiIw x(ctx, task, options);
    const MultiIwBatch b = x.run(starts, goals);
    std::vector<search::IwResult> out;
    out.reserve(starts.size());
    for (u32 i = 0; i < b.n; ++i)
        out.push_back(b.result(i, *task, starts[i], options.costs));
    return out;
}

MultiIwBatch batched_iw1(ContextPtr ctx, TaskPtr task, DeviceStarts starts, MultiIwOptions options, cudaStream_t stream)
{
    options.max_arity = 1;
    DeviceMultiIw x(std::move(ctx), std::move(task), options);
    return x.run(starts, {}, {}, stream);
}
}  // namespace mymyr::cuda
