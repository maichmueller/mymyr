// The device layer BrFS (include/mymyr/cuda/brfs.hpp).

#include "row_layout.hpp"

#include "mymyr/cuda/brfs.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/threads.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"
#include "mymyr/cuda/state_set.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>

namespace mymyr::cuda
{
namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

/// Arena rows use power-of-two buckets through 64 words, then whole 64-word chunks.
u32 bucket(const Task& task) { return search_row_words(task.words(), task.numeric_slots()); }

constexpr u32 k_no_parent = 0xFFFFFFFFu;

/// Timed chunks between two folds of their events into the statistics (each CUDA event holds host memory).
constexpr usize k_timed_chunks = 256;
/// The fewest rows of a successor window held to the arenas' spare rows before the search refuses (a window costs a
/// few synchronizations and launches whatever its rows).
constexpr u64 k_min_window_rows = 64;

template<class T>
T* as(Scratch& s)
{
    return static_cast<T*>(s.data());
}
}  // namespace

struct DeviceBrfs::Impl
{
    ContextPtr ctx;
    TaskPtr task;
    DeviceBrfsOptions o;
    cudaStream_t s = nullptr;
    std::unique_ptr<ChunkGenerator> gen;
    std::unique_ptr<DeviceArena> states, nodes;
    u32 W = 1;
    DeviceBuffer table;
    u64 slots = 0;
    Scratch sort, parent, cand, result, rank, scan, ctl;
    Scratch window_row, window_total;  // a windowed state's own copy of its row, and a window's candidates (a u32)
    PinnedLease pin;  // the group's control block and records, read once per group
    bool ran = false;
    DeviceBrfsStats st;
    std::deque<Event> events;  // 4 per chunk with timings: view, gen, dedup boundaries (stable references); folded
    usize next_event = 0;

    state_set::Table tab() const { return {static_cast<u64*>(table.data()), slots - 1}; }
    state_set::Rows rows() const { return {reinterpret_cast<const u64*>(states->device_data()), W}; }

    Event& event()
    {
        if (next_event == events.size())
            events.emplace_back(true);
        return events[next_event++];
    }

    /// Adds the recorded chunks' phase times to st and reuses their events (waits for the last one).
    void fold_events()
    {
        if (next_event == 0)
            return;
        events[next_event - 1].synchronize();
        for (usize i = 0; i + 3 < next_event; i += 4)
        {
            st.view_ms += events[i + 1].elapsed_ms(events[i]);
            st.gen_ms += events[i + 2].elapsed_ms(events[i + 1]);
            st.dedup_ms += events[i + 3].elapsed_ms(events[i + 2]);
        }
        next_event = 0;
    }

    /// A table of `n` slots (the search is unchanged when the allocation fails).
    void alloc_table(u64 n)
    {
        DeviceBuffer t(ctx, n * sizeof(u64), s);
        check(cudaMemsetAsync(t.data(), 0, n * sizeof(u64), s), "cudaMemsetAsync (state table)");
        table = std::move(t);
        slots = n;
    }
    /// The slots ensure_table(count, more) grows the table to (0: it has room).
    [[nodiscard]] u64 table_growth(u64 count, u64 more) const
    {
        if (count + more <= table_limit())
            return 0;
        u64 n = slots;
        while (count + more > n / 4)
            n *= 2;
        return n;
    }

    /// Makes room for `more` new states (an upper bound: the chunk's candidates) at load <= 3/4, growing to load
    /// <= 1/4 (few rehashes). The device sizes grouped chunks: it halts a chunk whose candidates exceed
    /// table_limit() (state_set::launch_chunk_size), which the host then sizes here with its exact candidates, so that
    /// the table grows where and as it would chunk by chunk.
    void ensure_table(u64 count, u64 more)
    {
        const u64 n = table_growth(count, more);
        if (n == 0)
            return;
        alloc_table(n);
        check(state_set::launch_rehash(tab(), rows(), count, s), "launch_rehash");
        ++st.rehashes;
    }
    /// Stored states plus a chunk's candidates the table takes (load <= 3/4).
    u64 table_limit() const { return slots / 4 * 3; }

    /// Re-lays the arena out at `nw` words per row (lazy slots outgrew the rows) and rehashes the table.
    void widen(u32 nw, u64 count)
    {
        auto grown = std::make_unique<DeviceArena>(ctx, nw * 8, std::max(states->capacity(), count + 1024), states->has_mirror());
        std::byte* dst = grown->tail(count);
        if (task->numeric_slots())
        {
            auto view = gen->view();
            view.numeric.storage = 0;
            check(numeric::launch_convert(view, reinterpret_cast<const u64*>(states->device_data()), W,
                                           W - task->numeric_slots(), reinterpret_cast<u64*>(dst), nw,
                                           nw - task->numeric_slots(), count, true, s), "numeric relayout");
        }
        else
            check(state_set::launch_relayout(reinterpret_cast<const u64*>(states->device_data()), W, reinterpret_cast<u64*>(dst), nw,
                                             count, s), "launch_relayout");
        grown->commit(count);
        widened_growths += states->growths();
        widened_mirror_growths += states->mirror_growths();
        states = std::move(grown);
        W = nw;
        check(cudaMemsetAsync(table.data(), 0, slots * sizeof(u64), s), "cudaMemsetAsync (state table)");
        check(state_set::launch_rehash(tab(), rows(), count, s), "launch_rehash");
        ++st.widenings;
    }

    u64 widened_growths = 0, widened_mirror_growths = 0;  // of the states' arenas replaced by widen()

    /// A chunk of a layer: parents [b, b + ns); its candidate capacity (0: from the estimate), whether its last
    /// parent precedes the search's first goal state (the parents before it, redone after a goal halt), and whether the
    /// generator still holds its count (it halted for room as the last chunk of its group: it resumes at its rows, with
    /// its goal count).
    struct Chunk
    {
        u64 b = 0;
        u32 ns = 0;
        u64 known = 0;
        bool before_goal = false;
        bool counted = false;
        u32 goals = 0;
        u32 first = 0;
        bool exact = false;  // sized after its count, alone (its group's estimated reservations did not fit)
    };

    /// Pinned room for the control block and `n` records.
    void pin_room(u32 n)
    {
        const u64 bytes = sizeof(state_set::ChunkCtl) + u64{n} * sizeof(state_set::ChunkRecord);
        if (pin.size() == 0 || bytes > pin_bytes)
        {
            pin_bytes = std::max<u64>(bytes, 2 * pin_bytes);
            pin = ctx->lease_pinned(1, pin_bytes);
        }
    }
    u64 pin_bytes = 0;

    /// Grows an arena for `more` records: doubling while the doubled arena takes at most a third of what the context
    /// can still allocate, otherwise to what that third holds, but by at least a quarter while that takes at most half
    /// of it, and at least to what it needs (the old generation is held until the copy is done). Growth by exactly
    /// what each chunk needs would copy the whole arena at every chunk: quadratic in the states.
    void grow(DeviceArena& a, u64 more) const
    {
        if (a.device_size() + more <= a.capacity())
            return;
        const u64 room = device_room(*ctx) / a.record_bytes(), step = a.capacity() + a.capacity() / 4;
        a.reserve(more, std::max(room / 3, step <= room / 2 ? step : 0));
    }
    /// At least `bytes` of `x`; with `exact`, exactly max(bytes, 64): a buffer of another size is freed first (no
    /// growth margin, and the old one is not held while the new one is allocated).
    void* size_buffer(Scratch& x, u64 bytes, bool exact) const
    {
        if (exact && x.size() != std::max<u64>(bytes, 64))
            x.reset();
        return x.ensure(ctx, bytes, s);
    }
    [[nodiscard]] u64 buffer_bytes() const
    {
        u64 n = 0;
        for (const Scratch* x : {&sort, &parent, &cand, &result, &rank, &scan, &ctl, &window_row, &window_total})
            n += x->size();
        return n;
    }
    /// What the search holds of the context's memory (refusals).
    [[nodiscard]] std::string held() const
    {
        return held_text(*ctx, {{"states", states ? states->capacity() * states->record_bytes() : 0},
                                {"search nodes", nodes ? nodes->capacity() * nodes->record_bytes() : 0},
                                {"state table", table.size()},
                                {"candidate buffers", buffer_bytes()}});
    }
    /// What the search was allocating (refusals of an allocation that failed).
    const char* phase = "the task's device tables and the successor generator";

    DeviceBrfsResult run();
    std::vector<Action> plan_to(u64 id);
};

DeviceBrfs::DeviceBrfs(ContextPtr ctx, TaskPtr task, const DeviceBrfsOptions& options) : m(std::make_unique<Impl>())
{
    if (!ctx || !task)
        throw std::invalid_argument("mymyr: DeviceBrfs: null context or task");
    m->ctx = std::move(ctx);
    m->task = std::move(task);
    m->o = options;
    m->s = m->ctx->stream();  // the arenas' writer stream
    if (m->o.chunk_states == 0)
        throw std::invalid_argument("mymyr: DeviceBrfs: chunk_states must be at least 1");
    if (m->o.expected_states > state_set::k_max_states)
        throw std::invalid_argument("mymyr: DeviceBrfs: expected_states above the state id limit (" +
                                    std::to_string(state_set::k_max_states) + ")");
}

DeviceBrfs::~DeviceBrfs()
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

const DeviceArena& DeviceBrfs::states() const
{
    if (!m->states)
        throw std::logic_error("mymyr: DeviceBrfs: run() first");
    return *m->states;
}
const DeviceArena& DeviceBrfs::nodes() const
{
    if (!m->nodes)
        throw std::logic_error("mymyr: DeviceBrfs: run() first");
    return *m->nodes;
}
u32 DeviceBrfs::words() const noexcept { return m->W; }
std::vector<Action> DeviceBrfs::plan_to(u64 id) const { return m->plan_to(id); }
DeviceBrfsResult DeviceBrfs::run()
{
    try
    {
        return m->run();
    }
    catch (const CudaError& e)
    {
        // (an allocation the search did not size against the context's room: its phase and what it holds)
        if (!out_of_memory(e))
            throw;
        refuse_allocation_failure(*m->ctx, "device BrFS", m->phase, m->held(),
                                  "Raise the Context's max_bytes, or use the CPU BrFS.");
    }
}

DeviceBrfsResult brfs(ContextPtr ctx, TaskPtr task, const DeviceBrfsOptions& options)
{
    return DeviceBrfs(std::move(ctx), std::move(task), options).run();
}

DeviceBrfsResult DeviceBrfs::Impl::run()
{
    if (ran)
        throw std::logic_error("mymyr: DeviceBrfs::run() may be called once");
    ran = true;
    DeviceGuard guard(ctx->device());
    const auto t_start = Clock::now();
    BrfsResult r;
    r.store = "device";
    r.threads_used = 0;
    const bool witness = o.witness_pruning, canonical = o.canonical_order;

    if (const std::string why = ChunkGenerator::search_unsupported(*task); !why.empty())
        throw std::invalid_argument("mymyr: DeviceBrfs: the device cannot run " + why);
    gen = std::make_unique<ChunkGenerator>(ctx, task, s);
    gen->set_timing(o.timings);
    st.host_schemas = gen->placement(witness).host_count;
    st.ce_schemas = gen->placement(witness).ce_count;
    st.host_ce_schemas = gen->placement(witness).host_ce_count;
    st.device_axioms = task->has_axioms() && gen->device_axioms();
    const u32 S = gen->num_schemas(), L = gen->label_width();
    const bool host = gen->needs_host(witness);
    W = bucket(*task);
    const u64 budget = std::clamp<u64>(o.max_states, 1, state_set::k_max_states);
    const u64 expected = o.expected_states ? o.expected_states : (o.max_states != ~u64{0} ? budget : 0);
    const u64 initial = expected ? expected + expected / 8 + 1024 : u64{1} << 16;
    // the arenas and the table start within a quarter of what the context can still allocate each (they grow as the
    // states need)
    phase = "the search's initial arenas and state table";
    const u64 room0 = device_room(*ctx);
    const u64 cap0 = std::clamp<u64>(!lifted::wide_row(W) ? initial : std::min(initial, (u64{16} << 20) / (u64{W} * 8)), 1,
                                     std::max<u64>(1, room0 / 4 / (u64{W} * 8 + 8)));
    states = std::make_unique<DeviceArena>(ctx, W * 8, cap0, host);  // the mirror only when the CPU reads parents
    nodes = std::make_unique<DeviceArena>(ctx, 8, cap0, false);
    u64 slots0 = expected ? std::bit_ceil(std::max<u64>(u64{1} << 16, 2 * expected + 2)) : u64{1} << 20;
    while (slots0 > 1024 && slots0 * sizeof(u64) > room0 / 4)
        slots0 /= 2;
    alloc_table(slots0);

    // the root
    {
        const State s0 = task->initial_state();
        std::vector<u64> row(W, 0);
        numeric::encode(*task, s0.view(), row.data(), W);
        states->append_from_host(row.data(), 1);
        const u32 root[2] = {k_no_parent, 0};
        nodes->append_from_host(root, 1);
        check(state_set::launch_rehash(tab(), rows(), 1, s), "launch_rehash");
        if (host)
            states->sync_to_host();
    }
    // chunk size: the views of a chunk (and its derived bitsets) fit the budget (by default half of L2, so they are
    // read from L2, not DRAM), but a chunk has at least 65536 parents while their views take at most 256 MB: with
    // views of kilobytes (philosophers: 3.5-4 KB) an L2-sized chunk would leave most of the device idle; and the
    // views of a chunk take at most an eighth of what the context can still hold (view_parents)
    u64 chunk_cap = 0, view_per = 8;
    {
        int l2 = 0;
        check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device()), "cudaDeviceGetAttribute");
        const u64 dw = gen->device_axioms() ? std::max<u32>(1, bits::words_for(task->atoms().max_derived_slots())) : 0;
        const u64 per = std::max<u64>(8, (gen->view_words() + dw + (lifted::wide_row(W) ? W : 0)) * 8);
        view_per = per;
        const u64 floor = std::min<u64>(65536, (u64{256} << 20) / per);
        chunk_cap = o.view_bytes ? std::max<u64>(1, o.view_bytes / per)
                                 : std::max<u64>({lifted::wide_row(W) ? 1u : 8192u, floor, std::max<u64>(static_cast<u64>(l2) / 2, 1) / per});
    }
    u64 count = 1, lb = 0, le = 1;
    bool stop = false, budget_hit = false;
    // the last windowed state and its transitions (the cut back to the budget need not enumerate them again)
    u64 windowed_state = ~u64{0}, windowed_transitions = 0;
    u64 goal_id = 0;
    u64 first_goal = ~u64{0};  // the first goal state expanded (stop_at_goal stops before it instead)
    u64 b = 0;             // the layer [lb, le)'s next parent
    bool started = false;  // the layer [lb, le) was counted
    bool exhausted = false;
    // the chunks of a layer run in groups without a host read in between: each chunk is sized on the device
    // (state_set::ChunkCtl: its candidates against a capacity estimated from the last chunks, the first new id a device
    // count) and the host reads the group's records once. A chunk past its capacity, or one whose parents hold a goal
    // state when the search stops at goals, writes and commits nothing and halts the group; the host redoes it: with
    // room, from its rows when it was the group's last chunk (the generator holds its count; the host's exact sizing
    // then costs what a host-sized chunk does), and the parents before the goal state. Lazy slots synchronize per chunk
    // anyway (interning, width growth on the host's count): groups of one chunk there. With CPU-fallback work (host) a
    // redo would repeat that work: the host reads each chunk's candidates before its rows (exact capacities, groups of
    // one chunk).
    const bool exact = host;
    const bool grouped = gen->missing_flag() == nullptr && !exact;
    // the CPU-fallback work of a chunk runs on the host while the device computes the chunk's views, goal
    // count and the device schemas' counts (frozen slots, no CPU axioms: ChunkGenerator::can_defer_host)
    const bool overlap = host && gen->can_defer_host();
    // the capacities' estimate: the most candidates per parent among the last k_window committed chunks (successive
    // layers alternate: gripper's 4 to 11 per parent), with a margin, up to o.candidate_bytes of candidate rows (a
    // chunk past it halts for room and is redone with its candidates known, a loop's chunk is cut, launch_loop_size;
    // unbounded, organic-synthesis' 32 per parent from a layer of 6 parents sized 524288 candidates of 64 words and
    // 1.8 GB of arenas and scratch for 41311 states)
    constexpr usize k_window = 4;
    std::array<double, k_window> ratios{8, 0, 0, 0};
    usize ratio_at = 0;
    auto estimate = [&](u64 ns)
    {
        const u64 est = static_cast<u64>(std::ceil(*std::ranges::max_element(ratios) * 1.25 * static_cast<double>(ns))) + 256;
        return std::min(est, std::max<u64>(lifted::wide_row(W) ? 1u : 256u, o.candidate_bytes / (u64{W} * sizeof(u64))));
    };
    // (estimates are cut to what the context can still hold, cap_cut, set per group: a chunk past its capacity halts
    // and is redone at its exact size)
    u64 cap_cut = ~u64{0};
    auto capacity = [&](const Chunk& k) { return k.known ? k.known : std::min(estimate(k.ns), cap_cut); };
    std::vector<Chunk> plan;
    // the scratch of a chunk's candidates (one chunk at a time on the stream)
    u32 *d_parent = nullptr, *d_sort = nullptr, *d_result = nullptr, *d_rank = nullptr;
    u64* d_cand = nullptr;
    void* d_scan = nullptr;
    u64 tb = 0;
    // (exact: buffers that grow are freed first and take exactly what n rows need, size_buffer; sort_rows: the sort
    // scratch's rows when they are not n, a successor window's)
    auto size_scratch = [&](u64 n, bool exact = false, u64 sort_rows = 0)
    {
        d_parent = static_cast<u32*>(size_buffer(parent, n * sizeof(u32), exact));
        d_cand = static_cast<u64*>(size_buffer(cand, n * W * sizeof(u64), exact));
        d_sort = canonical && gen->sorts(witness)
                     ? static_cast<u32*>(size_buffer(sort, (sort_rows ? sort_rows : n) * L * sizeof(u32), exact))
                     : nullptr;
        d_result = static_cast<u32*>(size_buffer(result, n * sizeof(u32), exact));
        d_rank = static_cast<u32*>(size_buffer(rank, (n + 1) * sizeof(u32), exact));
        tb = state_set::rank_temp_bytes(n);
        d_scan = size_buffer(scan, tb, exact);
    };
    // the halt of a chunk: the record's reason, applied to its plan entry (redone with room, the parents before its
    // first goal state, or the search stops); `resumes`: the generator still holds its count
    auto apply_halt = [&](Chunk& ck, const state_set::ChunkRecord& x, bool resumes)
    {
        if (x.error)
            throw std::logic_error("mymyr: device BrFS: a sorted segment found no scratch (internal error)");
        if (x.halt & state_set::k_halt_budget)
        {
            budget_hit = true;
            stop = true;
        }
        else if (x.halt & state_set::k_halt_room)
        {
            ck.known = x.candidates;
            ck.counted = resumes;
            ck.goals = x.goals;
            ck.first = x.first;
            if (!ck.counted)
                ++st.redone;
        }
        else  // k_halt_goal: redo the parents before the goal state
        {
            if (x.first != 0)
                ++st.redone;
            ck.ns = x.first;
            ck.known = 0;
            ck.before_goal = true;
            if (x.first == 0)
            {
                ++st.chunks;
                r.num_goal_states += 1;
                stop = true;
                r.solved = true;
                goal_id = ck.b;
            }
        }
    };

    // Device memory per candidate row of a chunk: its row, parent, result and rank, and at most a new state of the
    // arenas with its table slots (load 1/4 after growth); sort rows apart. At the current width (lazy slots widen it).
    const bool sorting = canonical && gen->sorts(witness);
    auto per_candidate = [&] { return u64{W} * 16 + 4 * 3 + 8 + 32 + (sorting ? u64{L} * 4 : 0); };
    // What a chunk's candidates may take: what the context can still allocate, with the candidate buffers the chunk
    // reuses; a chunk holds a quarter of it (the arenas and the table grow in steps, and a grown buffer is allocated
    // before the one it replaces is freed).
    auto reused_bytes = [&] { return parent.size() + cand.size() + sort.size() + result.size() + rank.size() + scan.size(); };
    auto room_bytes = [&] { return device_room(*ctx) + reused_bytes(); };
    auto room_rows = [&] { return room_bytes() / 4 / per_candidate(); };
    // the most parents whose views take an eighth of the room (the generator's buffers of a chunk follow its parents)
    auto view_parents = [&] { return std::max<u64>(1, room_bytes() / 8 / view_per); };
    // A state whose successors exceed room_rows(): its successors a window at a time (lifted::Labels::first), each
    // window deduplicated and committed before the next, so that ids, counts and the plan are the ones of a chunk
    // that holds them all. The generator reads the state from a copy of its own (arena growth moves the arena).
    // Returns false when the search stops (budget, goal).
    auto run_windows = [&](const Chunk& ck, state_set::ChunkCtl* d_ctl, state_set::ChunkRecord* d_rec, state_set::ChunkCtl* h_ctl,
                           state_set::ChunkRecord* h_rec)
    {
        const u64 b = ck.b;
        std::vector<u32> off(u64{S} + 1);
        std::vector<u64> host_row;
        u64* prow = nullptr;
        auto* d_total = static_cast<u32*>(window_total.ensure(ctx, sizeof(u32), s));
        // the state's views and counts, and its goal count in the record, from its own row
        auto begin_state = [&]
        {
            const auto th = Clock::now();
            prow = static_cast<u64*>(window_row.ensure(ctx, u64{W} * sizeof(u64), s));
            check(cudaMemcpyAsync(prow, states->device_data() + b * W * sizeof(u64), u64{W} * sizeof(u64),
                                  cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync (window row)");
            if (host)
            {
                if (states->host_size() < b + 1)
                    states->wait();
                const auto* h = reinterpret_cast<const u64*>(states->host_data()) + b * W;
                host_row.assign(h, h + W);
            }
            *h_rec = state_set::ChunkRecord{};
            check(cudaMemcpyAsync(d_rec, h_rec, sizeof(state_set::ChunkRecord), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
            gen->begin(ChunkInput{prow, W, W, 1, host ? host_row.data() : nullptr, W, static_cast<u32>(b)}, witness, canonical);
            st.host_ms += ms_since(th);
            gen->views();
            gen->goal_count(&d_rec->goals);
            gen->count();
            check(cudaMemcpyAsync(off.data(), gen->seg_offsets(), off.size() * sizeof(u32), cudaMemcpyDeviceToHost, s),
                  "cudaMemcpyAsync (window offsets)");
            check(cudaMemcpyAsync(h_rec, d_rec, sizeof(state_set::ChunkRecord), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        };
        begin_state();
        const u32 goals = h_rec->goals, first_goal_row = h_rec->first;
        const u64 M = off[S];
        windowed_state = b;
        windowed_transitions = M;
        if (M > 0x7FFFFFFFu)  // (the counts' prefixes saturate past the device's 31-bit row index)
            throw std::length_error("mymyr: device BrFS: a state has more than 2^31 - 1 successors; use the CPU BrFS");
        phase = "a successor window";
        // A window's buffers at their exact sizes: the candidate buffers (those of another size are freed first,
        // window_room counts them as free), the table's growth, and the arenas' growth when they have no room for its
        // new states.
        Scratch* const buffers[] = {&parent, &cand, &sort, &result, &rank, &scan};
        auto window_room = [&]
        {
            u64 n = device_room(*ctx);
            for (const Scratch* x : buffers)
                n += x->size();
            return n;
        };
        auto window_buffers = [&](const SuccessorWindow& w)
        {
            const u64 n = w.rows;
            auto at_least = [](u64 x) { return std::max<u64>(x, 64); };
            u64 bytes = at_least(n * sizeof(u32)) * 2 + at_least(n * W * sizeof(u64)) + at_least((n + 1) * sizeof(u32)) +
                        at_least(state_set::rank_temp_bytes(n)) + table_growth(count, n) * sizeof(u64);
            if (sorting)
                bytes += at_least(std::max<u64>(w.scratch_rows, 1) * L * sizeof(u32));
            return bytes;
        };
        auto window_bytes = [&](const SuccessorWindow& w)
        {
            return window_buffers(w) + states->growth_bytes(w.rows, count + w.rows) + nodes->growth_bytes(w.rows, count + w.rows);
        };
        // Every new state of the state in arenas reserved at once, with an eighth more for the states after it, as far
        // as they fit beside a sixteenth of the room for the windows, and as many again (as doubling would) as far as
        // they fit beside a quarter: arenas grown per window, or for the next layer's first states, would hold the
        // stored states twice at each growth. A window adds at most as many states as the arenas have room for, so that only a partial
        // reservation grows them again.
        {
            for (Scratch* x : buffers)
                x->reset();
            const u64 room = device_room(*ctx), row = states->record_bytes() + nodes->record_bytes();
            const u64 one = window_buffers(successor_window(off, 0, 1)), limit = std::max<u64>(budget, count + 1);
            auto rows_beside = [&](u64 keep) { return room > keep ? (room - keep) / row : 0; };
            const u64 total = std::max(std::min({count + M + M / 8, limit, rows_beside(std::max(one, room / 16))}),
                                       std::min({2 * (count + M), limit, rows_beside(std::max(one, room / 4))}));
            if (total > states->capacity())
                (void)try_allocate(*ctx, [&](int)
                                   {
                                       states->reserve(total - count, total);
                                       nodes->reserve(total - count, total);
                                   });
        }
        // a window of one successor that does not fit: the arenas' growth when they have no room for it (they hold the
        // stored states twice while they grow), otherwise the successor's own buffers (`failed`: their allocation
        // failed although the bytes are free: the pool holds them in smaller pieces)
        auto refuse_window = [&](const SuccessorWindow& w1, u64 room, bool failed)
        {
            const u64 bytes = window_bytes(w1);
            if (states->growth_bytes(1, count + 1) + nodes->growth_bytes(1, count + 1) != 0)
                refuse_memory(*ctx, "device BrFS",
                              "the arenas of more than " + std::to_string(count) + " states with a successor's buffers (" +
                                  std::to_string(bytes) + (failed ? " bytes, which the pool could not allocate in its free pieces)" : " bytes)"),
                              held(), "Raise the Context's max_bytes, or use the CPU BrFS.");
            refuse_successors(*ctx, "device BrFS", "a state", M, failed ? 0 : bytes, room,
                              "Raise the Context's max_bytes, or use the CPU BrFS.", per_candidate(), held());
        };
        u64 window_cap = M;  // halved when a window's buffers do not fit after all
        for (u64 f = 0; f < M;)
        {
            if (count >= budget)
            {
                // (the state's later windows count as generated, as a chunk's candidates do: the cut back to the
                // budget recounts them)
                if (f > 0)
                    r.generated += M - f;
                budget_hit = true;
                stop = true;
                return false;
            }
            // the window: the most rows whose buffers fit 15/16 of the room (a margin for the pool's granularity)
            const u64 room = window_room(), usable = room - room / 16;
            u64 hi = std::min(window_cap, M - f);
            const u64 spare = std::min(states->capacity(), nodes->capacity()) - count;
            if (spare > 0)
                hi = std::min(hi, spare);
            SuccessorWindow w = successor_window(off, f, 1);
            if (window_bytes(w) > usable)
                refuse_window(w, room, false);
            for (u64 lo = 1; lo < hi;)
            {
                const u64 mid = lo + (hi - lo + 1) / 2;
                if (const SuccessorWindow x = successor_window(off, f, mid); window_bytes(x) <= usable)
                {
                    lo = mid;
                    w = x;
                }
                else
                    hi = mid - 1;
            }
            // A window held to a few spare rows of arenas that cannot grow by one more: the state's other successors
            // would take a window per few rows (each with its synchronizations), and the search fails at the first
            // window whose new states exceed the spare rows unless all the rest are duplicates
            if (spare > 0 && w.rows == spare && spare < k_min_window_rows && spare < M - f)
                if (const u64 more = window_bytes(successor_window(off, f, spare + 1)); more > usable)
                    refuse_memory(*ctx, "device BrFS",
                                  "the arenas of more than " + std::to_string(count + spare) +
                                      " states with a successor window's buffers (" + std::to_string(more) + " bytes)",
                                  held(), "Raise the Context's max_bytes, or use the CPU BrFS.");
            // (numeric successor rows are written whole: a numeric state's successors fit at once or are refused)
            if (task->numeric_slots() && w.rows < M)
                refuse_successors(*ctx, "device BrFS", "a state", M, window_bytes(successor_window(off, 0, M)), room,
                                  "Raise the Context's max_bytes, or use the CPU BrFS.", per_candidate(), held());
            if (count + w.rows > state_set::k_max_states)
                throw std::length_error("mymyr: device BrFS: more than 2^31 - 2 states (with a chunk's candidates)");
            // (the largest first: freed candidate buffers and the arenas' growth, then the table and the buffers)
            const bool fits = try_allocate(*ctx, [&](int)
                                           {
                                               if (states->growth_bytes(w.rows, count + w.rows) != 0)
                                                   for (Scratch* x : buffers)
                                                       x->reset();
                                               states->reserve(w.rows, count + w.rows);
                                               nodes->reserve(w.rows, count + w.rows);
                                               ensure_table(count, w.rows);
                                               size_scratch(w.rows, true, std::max<u64>(w.scratch_rows, 1));
                                           });
            if (!fits)
            {
                // (nothing of the window ran: a smaller one, down to a single successor)
                if (w.rows == 1)
                    refuse_window(w, window_room(), true);
                window_cap = w.rows / 2;
                continue;
            }
            // the window's control block and record (the state's goal count goes with its first window)
            *h_ctl = state_set::ChunkCtl{static_cast<u32>(count), 0, 0, 0};
            *h_rec = state_set::ChunkRecord{};
            if (f == 0)
            {
                h_rec->goals = goals;
                h_rec->first = first_goal_row;
            }
            const u32 n32 = static_cast<u32>(w.rows);
            check(cudaMemcpyAsync(d_ctl, h_ctl, sizeof(state_set::ChunkCtl) + sizeof(state_set::ChunkRecord),
                                  cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
            check(cudaMemcpyAsync(d_total, &n32, sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
            check(state_set::launch_chunk_size(d_ctl, d_rec, d_total, {w.rows, table_limit(), budget, o.stop_at_goal}, s),
                  "launch_chunk_size");
            lifted::Labels labels;
            labels.parent = d_parent;
            labels.first = w.first;
            labels.capacity = w.first + w.rows;
            labels.label_width = L;
            labels.parent_base = static_cast<u32>(b);
            labels.scratch = d_sort;
            labels.scratch_first = w.scratch_first;
            labels.scratch_rows = d_sort ? w.scratch_rows : 0;
            labels.scratch_indexed = d_sort ? 1 : 0;
            labels.error = &d_rec->error;
            labels.live = &d_ctl->live;
            gen->write(labels, d_cand, W);
            bool widened = false;
            for (;;)
            {
                const auto tm = Clock::now();
                const bool missing = gen->resolve_missing();
                st.host_ms += ms_since(tm);
                if (!missing)
                    break;
                if (bucket(*task) > W)
                {
                    widen(bucket(*task), count);
                    widened = true;
                    break;
                }
                gen->write(labels, d_cand, W);
            }
            if (widened)
            {
                // the window again at the new width, from the state's own row (nothing of it is committed yet)
                if (host)
                    states->sync_to_host();
                begin_state();
                continue;
            }
            const state_set::Live live{&d_ctl->live};
            check(state_set::launch_insert(tab(), rows(), {d_cand, W}, w.rows, d_result, s, live), "launch_insert");
            check(state_set::launch_rank(tab(), d_result, w.rows, d_rank, d_scan, tb, s, live), "launch_rank");
            state_set::Compact cm;
            cm.result = d_result;
            cm.rank = d_rank;
            cm.cand = d_cand;
            cm.parent = d_parent;
            cm.seg_offsets = gen->seg_offsets();
            cm.num_schemas = S;
            cm.parent_base = static_cast<u32>(b);
            cm.arena_tail = reinterpret_cast<u64*>(states->device_data());
            cm.nodes_tail = reinterpret_cast<u32*>(nodes->device_data());
            cm.base = 0;
            cm.words = W;
            cm.offset = &d_ctl->count;
            cm.first_row = static_cast<u32>(w.first);
            check(state_set::launch_compact(tab(), cm, w.rows, s, live), "launch_compact");
            check(state_set::launch_advance(d_rank, w.rows, &d_ctl->count, &d_rec->fresh, s), "launch_advance");
            check(cudaMemcpyAsync(h_rec, d_rec, sizeof(state_set::ChunkRecord), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            ++st.groups;
            const state_set::ChunkRecord x = *h_rec;
            if (x.error)
                throw std::logic_error("mymyr: device BrFS: a sorted segment found no scratch (internal error)");
            if (x.halt & state_set::k_halt_goal)
            {
                // a goal state, stopped at before its expansion (only the first window carries the goal count)
                ++st.chunks;
                r.num_goal_states += 1;
                stop = true;
                r.solved = true;
                goal_id = b;
                return false;
            }
            if (x.halt)
                throw std::logic_error("mymyr: device BrFS: a successor window halted (internal error)");
            count += x.fresh;
            states->commit(x.fresh);
            nodes->commit(x.fresh);
            if (host)
                states->sync_to_host();
            r.generated += w.rows;
            ++st.chunks;
            ++st.windows;
            if (f == 0)
            {
                r.expanded += 1;
                r.num_goal_states += x.goals;
                if (x.goals)
                    first_goal = std::min<u64>(first_goal, b + x.first);
            }
            f += w.rows;
        }
        if (ck.before_goal)
        {
            // the parents before the first goal state are expanded: stop there (the CPU's order)
            r.num_goal_states += 1;
            stop = true;
            r.solved = true;
            goal_id = b + 1;
            return false;
        }
        return true;
    };

    // A device loop over the chunks of the small layers. A WHILE graph replays a captured chunk of up to C
    // parents (state_set::LoopCtl: the cursor, the next layer from the device count) until the search ends, a chunk
    // halts (past its capacity, a goal, the budget: the host redoes it as in a group), the table or the arenas could not
    // take another chunk, or a layer outgrows C; the host reads the loop's counts once. Without host work and lazy
    // slots (grouped), timings or graphs (MYMYR_CUDA_GRAPHS=0, MYMYR_CUDA_DEVICE_LOOPS=0), and for layers of more than
    // k_loop_layer parents (their chunks are sized on the host, in groups), the host drives the chunks as before.
    // The loops start after the first host-driven chunk (`warm`): every kernel of the body has had a plain launch
    // before the first WHILE launch, as in the A* and multi-IW loops, which start after host-driven steps. Under
    // compute-sanitizer memcheck (2025.3), a BrFS WHILE body whose kernels had not run plainly faulted the GPU on every
    // run (an axiom kernel needs 2.5 KB of stack against the context's default 1 KB); warm, it runs clean. memcheck
    // can still fault once inside a warm loop, at an address outside the application's allocations, so the sanitizers
    // run without device loops (GraphExec::loops_enabled).
    constexpr u64 k_loop_layer = 16384;
    constexpr usize k_loop_graphs = 4;
    const bool loops = o.max_depth == ~u32{0} && grouped && !o.timings && GraphExec::enabled() && GraphExec::loops_enabled() && gen->capturable(witness);
    bool warm = false;
    struct LoopGraph
    {
        std::vector<u64> key;
        GraphExec exec;
    };
    std::vector<LoopGraph> loop_graphs;  // the most recent last
    Scratch loop_rows, loop_ctl;
    PinnedLease loop_pin;
    // runs the loop from the cursor; false when it ended at a halted chunk (the cursor's; its record in `halted`)
    // the loop's chunk (parents) and its candidates' capacity at the cursor: the layer twice over (shapes repeat)
    auto loop_chunk = [&]
    { return std::min<u64>({o.chunk_states, chunk_cap, view_parents(), k_loop_layer, std::bit_ceil(std::max<u64>(2048, 2 * (le - lb)))}); };
    auto loop_capacity = [&](u64 C) { return std::bit_ceil(estimate(C)); };
    // the arenas' room for a loop: four chunks' capacities (a loop ends when the next chunk might not fit, k_loop_room;
    // the next one starts in a grown arena), but no more than o.candidate_bytes of rows past one chunk's (organic-
    // synthesis' 64-word rows: four 65536-row chunks would hold 128 MB of arena and its mirror for 41311 states)
    auto loop_room = [&](u64 cap) { return std::max(cap, std::min(4 * cap, o.candidate_bytes / (u64{W} * sizeof(u64)))); };
    // the loop's arenas and buffers; false when the context cannot hold them (the host then sizes the chunks)
    auto loop_fits = [&](u64 C, u64 cap)
    {
        return try_allocate(*ctx, [&](int)
                            {
                                grow(*states, loop_room(cap));
                                grow(*nodes, loop_room(cap));
                                size_scratch(cap);
                                (void)loop_rows.ensure(ctx, C * W * sizeof(u64), s);
                            });
    };
    auto device_loop = [&](u64 C, u64 cap, state_set::ChunkRecord& halted, u32& halted_ns) -> bool
    {
        grow(*states, loop_room(cap));
        grow(*nodes, loop_room(cap));
        size_scratch(cap);
        auto* prow = static_cast<u64*>(loop_rows.ensure(ctx, C * W * sizeof(u64), s));
        auto* lc = static_cast<state_set::LoopCtl*>(loop_ctl.ensure(ctx, sizeof(state_set::LoopCtl), s));
        if (loop_pin.size() == 0)
            loop_pin = ctx->lease_pinned(1, sizeof(state_set::LoopCtl));
        auto* hl = static_cast<state_set::LoopCtl*>(loop_pin.data(0));
        gen->prepare(static_cast<u32>(C));
        state_set::LoopLimits lim;
        lim.chunk = static_cast<u32>(C);
        lim.capacity = cap;
        lim.table = table_limit();
        lim.arena = std::min(states->capacity(), nodes->capacity());
        lim.budget = budget;
        lim.stop_at_goal = o.stop_at_goal;
        lim.max_layer = C;
        const std::vector<u64> key = {C,
                                      cap,
                                      W,
                                      slots,
                                      lim.table,
                                      lim.arena,
                                      reinterpret_cast<u64>(table.data()),
                                      reinterpret_cast<u64>(states->device_data()),
                                      reinterpret_cast<u64>(nodes->device_data()),
                                      reinterpret_cast<u64>(d_parent),
                                      reinterpret_cast<u64>(d_cand),
                                      reinterpret_cast<u64>(d_sort),
                                      reinterpret_cast<u64>(d_result),
                                      reinterpret_cast<u64>(d_rank),
                                      reinterpret_cast<u64>(d_scan),
                                      tb,
                                      reinterpret_cast<u64>(prow),
                                      reinterpret_cast<u64>(lc),
                                      gen->capture_key()};
        auto it = std::ranges::find_if(loop_graphs, [&](const LoopGraph& x) { return x.key == key; });
        if (it == loop_graphs.end())
        {
            if (loop_graphs.size() == k_loop_graphs)
                loop_graphs.erase(loop_graphs.begin());
            auto body = [&](unsigned long long handle)
            {
                check(state_set::launch_loop_plan(lc, lim, s), "launch_loop_plan");
                check(state_set::launch_gather_parents(rows(), lc, prow, static_cast<u32>(C), s), "launch_gather_parents");
                gen->begin(ChunkInput{prow, W, W, static_cast<u32>(C), nullptr, W, 0, &lc->ns}, witness, canonical);
                gen->views();
                gen->goal_count(&lc->rec.goals);
                gen->count();
                // the chunk's parents whose candidates fit the capacity
                check(state_set::launch_loop_size(lc, gen->seg_offsets(), S, {cap, lim.table, budget, o.stop_at_goal}, s),
                      "launch_loop_size");
                lifted::Labels labels;
                labels.parent = d_parent;
                labels.capacity = cap;
                labels.label_width = L;
                labels.parent_base = 0;
                labels.scratch = d_sort;
                labels.scratch_rows = d_sort ? cap : 0;
                labels.scratch_indexed = d_sort ? 1 : 0;
                labels.error = &lc->rec.error;
                labels.live = &lc->chunk.live;
                gen->write(labels, d_cand, W);
                const state_set::Live live{&lc->chunk.live};
                check(state_set::launch_insert(tab(), rows(), {d_cand, W}, cap, d_result, s, live), "launch_insert");
                check(state_set::launch_rank(tab(), d_result, cap, d_rank, d_scan, tb, s, live), "launch_rank");
                state_set::Compact cm;
                cm.result = d_result;
                cm.rank = d_rank;
                cm.cand = d_cand;
                cm.parent = d_parent;
                cm.seg_offsets = gen->seg_offsets();
                cm.num_schemas = S;
                cm.parent_base = 0;
                cm.arena_tail = reinterpret_cast<u64*>(states->device_data());
                cm.nodes_tail = reinterpret_cast<u32*>(nodes->device_data());
                cm.base = 0;
                cm.words = W;
                cm.offset = &lc->chunk.count;
                cm.parent_offset = &lc->b;
                check(state_set::launch_compact(tab(), cm, cap, s, live), "launch_compact");
                check(state_set::launch_advance(d_rank, cap, &lc->chunk.count, &lc->rec.fresh, s), "launch_advance");
                check(state_set::launch_loop_next(lc, lim, handle, s), "launch_loop_next");
            };
            loop_graphs.push_back(LoopGraph{key, GraphExec::capture_while(s, body)});
            it = loop_graphs.end() - 1;
            ++st.captures;
        }
        *hl = state_set::LoopCtl{};
        hl->chunk.count = static_cast<u32>(count);
        hl->lb = static_cast<u32>(lb);
        hl->le = static_cast<u32>(le);
        hl->b = static_cast<u32>(b);
        hl->started = 1;
        hl->steps_left = 0xFFFFFFFFu;
        check(cudaMemcpyAsync(lc, hl, sizeof(state_set::LoopCtl), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
        it->exec.launch(s);
        check(cudaMemcpyAsync(hl, lc, sizeof(state_set::LoopCtl), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        ++st.loops;
        ++st.groups;
        st.chunks += hl->chunks;
        st.cuts += hl->cuts;
        r.layers += hl->layers;
        r.expanded += hl->expanded;
        r.generated += hl->generated;
        r.num_goal_states += hl->goal_states;
        first_goal = std::min<u64>(first_goal, hl->first_goal);
        if (hl->chunks)
            ratios[ratio_at++ % k_window] = static_cast<double>(hl->max_ratio) / 256;
        states->commit(hl->chunk.count - count);
        nodes->commit(hl->chunk.count - count);
        count = hl->chunk.count;
        lb = hl->lb;
        le = hl->le;
        b = hl->b;
        started = hl->started != 0;
        if (hl->end != state_set::k_loop_halt)
            return true;
        halted = hl->rec;
        halted_ns = hl->ns;
        return false;
    };
    const auto t0 = Clock::now();

    while (!stop)
    {
        if (b == le && started)
        {
            lb = le;
            le = count;
            started = false;
        }
        if (lb == le)
        {
            exhausted = true;
            break;
        }
        if (!started)
        {
            if (r.layers >= o.max_depth)
                break;
            if (count >= budget)
            {
                budget_hit = true;
                break;
            }
            ++r.layers;
            started = true;
        }
        state_set::ChunkRecord halted;
        u32 halted_ns = 0;
        // (the loop leaves the table's growth to the host's chunks: it runs while a chunk fits the table)
        const u64 loop_c = loop_chunk(), loop_cap = loop_capacity(loop_c);
        phase = "the successors of a chunk of states";
        const bool looped = loops && warm && le - b <= k_loop_layer && count + loop_cap <= table_limit() &&
                            count + loop_room(loop_cap) < state_set::k_max_states && loop_fits(loop_c, loop_cap);
        if (looped && device_loop(loop_c, loop_cap, halted, halted_ns))
            continue;  // (the cursor moved)
        plan.clear();
        const u64 parents = std::min(chunk_cap, view_parents());
        for (u64 x = b; x < le;)
        {
            // (a loop's halted chunk first: its parents, as the loop ran it, launch_loop_size, its record holds)
            const u64 most = looped && x == b ? halted_ns : o.chunk_states;
            const u32 ns = static_cast<u32>(std::min<u64>({most, le - x, parents}));
            plan.push_back({x, ns});
            x += ns;
        }
        if (looped)
        {
            apply_halt(plan[0], halted, false);
            if (stop)
                break;
        }
        for (usize i = 0; i < plan.size() && !stop;)
        {
            // the group [i, j): its candidates' room at most a quarter of the stored states (at least 2^20, at least
            // one chunk), so that the arena grows about as it would chunk by chunk; a chunk redone after it halted for
            // room (its candidates known; resumed at its rows when counted) runs alone, sized as with exact capacities.
            // The estimates and the room stay within what the context can still hold (room_rows); a group whose
            // reservations fail all the same runs its first chunk alone at its exact size.
            const bool exact_group = exact || plan[i].known || plan[i].exact;
            u64 room_limit = lifted::wide_row(W) ? std::max<u64>(1, o.candidate_bytes / (u64{W} * 8))
                                    : std::max<u64>(count / 4, u64{1} << 20);
            cap_cut = ~u64{0};
            if (!exact_group)
            {
                cap_cut = std::max<u64>(1, room_rows());
                room_limit = std::min(room_limit, cap_cut);
            }
            usize j = i;
            u64 room = 0, cap_max = 0;
            while (j < plan.size() && (j == i || (grouped && !exact_group && room + capacity(plan[j]) <= room_limit)))
            {
                room += capacity(plan[j]);
                cap_max = std::max(cap_max, capacity(plan[j]));
                ++j;
            }
            if (count + room > state_set::k_max_states)
                throw std::length_error("mymyr: device BrFS: more than 2^31 - 2 states (with a chunk's candidates)");
            // the scratch of the group's chunks and the arenas for its new states (the device checks the table's room
            // per chunk: ensure_table)
            if (!exact_group && !try_allocate(*ctx, [&](int)
                                              {
                                                  grow(*states, room);
                                                  grow(*nodes, room);
                                                  size_scratch(cap_max);
                                              }))
            {
                plan[i].exact = true;
                continue;
            }
            const u32 G = static_cast<u32>(j - i);
            auto* d_ctl = static_cast<state_set::ChunkCtl*>(
                ctl.ensure(ctx, sizeof(state_set::ChunkCtl) + u64{G} * sizeof(state_set::ChunkRecord), s));
            auto* d_rec = reinterpret_cast<state_set::ChunkRecord*>(d_ctl + 1);
            pin_room(G + 1);
            auto* h_ctl = static_cast<state_set::ChunkCtl*>(pin.data(0));
            auto* h_rec = reinterpret_cast<state_set::ChunkRecord*>(h_ctl + 1);
            auto* h_cand = reinterpret_cast<u32*>(h_rec + G);  // exact: the chunk's candidates
            // the control block and the records, in one upload (state_set::ChunkCtl, ChunkRecord; a resumed chunk's
            // record keeps its goal count)
            auto init_ctl = [&]
            {
                *h_ctl = state_set::ChunkCtl{static_cast<u32>(count), 0, 0, 0};
                std::fill_n(h_rec, G, state_set::ChunkRecord{});
                if (plan[i].counted)
                {
                    h_rec[0].goals = plan[i].goals;
                    h_rec[0].first = plan[i].first;
                }
                static_assert(sizeof(state_set::ChunkCtl) % alignof(state_set::ChunkRecord) == 0);
                check(cudaMemcpyAsync(d_ctl, h_ctl, sizeof(state_set::ChunkCtl) + u64{G} * sizeof(state_set::ChunkRecord),
                                      cudaMemcpyHostToDevice, s),
                      "cudaMemcpyAsync");
            };
            init_ctl();
            bool resplit = false, windowed = false;  // (exact groups: one chunk)
            for (usize k = i; k < j && !resplit && !windowed; ++k)
            {
                Chunk& ck = plan[k];
                const u64 b = ck.b;
                u64 cap = capacity(ck);
                const u32 ns = ck.ns;
                state_set::ChunkRecord* rec = d_rec + (k - i);
                for (;;)  // the chunk's launches (redone after a width growth: lazy slots, a group of one)
                {
                    Event* ev[4] = {nullptr, nullptr, nullptr, nullptr};
                    if (o.timings)
                    {
                        if (next_event >= 4 * k_timed_chunks)
                            fold_events();
                        for (auto& e : ev)
                            e = &event();
                    }
                    if (ev[0])
                        ev[0]->record(s);
                    if (!ck.counted)
                    {
                        const auto th = Clock::now();
                        if (host && states->host_size() < b + ns)
                            states->wait();
                        const auto* dev_rows = reinterpret_cast<const u64*>(states->device_data()) + b * W;
                        const auto* host_rows = host ? reinterpret_cast<const u64*>(states->host_data()) + b * W : nullptr;
                        const ChunkInput in{dev_rows, W, W, ns, host_rows, W, static_cast<u32>(b)};
                        if (overlap)
                            gen->begin_device(in, witness, canonical);
                        else
                            gen->begin(in, witness, canonical);
                        st.host_ms += ms_since(th);
                        gen->views();
                    }
                    if (ev[1])
                        ev[1]->record(s);
                    if (!ck.counted)
                    {
                        gen->goal_count(&rec->goals);
                        if (overlap)
                        {
                            gen->count_device();
                            const auto th = Clock::now();
                            gen->host_work();  // (while the device runs the views and counts)
                            st.host_ms += ms_since(th);
                            gen->count_host();
                        }
                        else
                            gen->count();
                    }
                    // more candidates than the device has room for (`fit` rows), found before anything of the chunk is
                    // committed: fewer parents (the rest is the next chunk), or one parent's successors a window at a
                    // time (run_windows)
                    // the new states the chunk can add (none without candidates: its one-row buffers hold none)
                    u64 adds = cap;
                    auto split = [&](u64 fit)
                    {
                        if (o.timings)
                            next_event -= 4;
                        if (adds == 0)  // (a chunk without candidates needs only its one-row buffers)
                            refuse_memory(*ctx, "device BrFS", "the candidate buffers of " + std::to_string(ns) + " states",
                                          held(), "Raise the Context's max_bytes, or use the CPU BrFS.");
                        if (ns == 1)
                        {
                            windowed = true;
                            return;
                        }
                        std::vector<u32> firsts(u64{ns} + 1);
                        check(cudaMemcpy2DAsync(firsts.data(), sizeof(u32), gen->seg_offsets(), u64{S} * sizeof(u32), sizeof(u32),
                                                u64{ns} + 1, cudaMemcpyDeviceToHost, s),
                              "cudaMemcpy2DAsync (parent offsets)");
                        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
                        u32 n1 = 1;
                        while (n1 < ns && firsts[n1 + 1] <= fit)
                            ++n1;
                        if (n1 == ns)  // (fewer parents need no less: the chunk's own buffers do not fit)
                            refuse_memory(*ctx, "device BrFS", "the candidate buffers of " + std::to_string(ns) + " states",
                                          held(), "Raise the Context's max_bytes, or use the CPU BrFS.");
                        Chunk rest{b + n1, ns - n1};
                        rest.before_goal = ck.before_goal;
                        ck.ns = n1;
                        ck.known = firsts[n1];
                        ck.counted = false;
                        ck.before_goal = false;
                        plan.insert(plan.begin() + static_cast<std::ptrdiff_t>(k) + 1, rest);
                        ++st.redone;
                        resplit = true;
                    };
                    if (exact_group)
                    {
                        if (!ck.known)  // (CPU-fallback work: its candidates before its rows)
                        {
                            check(cudaMemcpyAsync(h_cand, gen->seg_offsets() + u64{ns} * S, sizeof(u32), cudaMemcpyDeviceToHost,
                                                  s),
                                  "cudaMemcpyAsync");
                            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
                            cap = std::max<u64>(*h_cand, 1);
                            adds = *h_cand;
                        }
                        u64 fit = cap * per_candidate() > reused_bytes() ? room_rows() : cap;
                        if (cap <= fit || adds == 0)
                        {
                            if (count + cap > state_set::k_max_states)
                                throw std::length_error("mymyr: device BrFS: more than 2^31 - 2 states");
                            fit = cap;
                            if (!try_allocate(*ctx, [&](int attempt)
                                              {
                                                  ensure_table(count, adds);
                                                  size_scratch(cap, attempt == 1);
                                              }))
                                fit = cap / 2;
                        }
                        if (cap > fit)
                        {
                            split(fit);
                            break;
                        }
                    }
                    check(state_set::launch_chunk_size(d_ctl, rec, gen->seg_offsets() + u64{ns} * S,
                                                       {cap, table_limit(), budget, o.stop_at_goal}, s),
                          "launch_chunk_size");
                    // successor rows and their parents, in canonical order (long sorted segments use the sort scratch)
                    lifted::Labels labels;
                    labels.parent = d_parent;
                    labels.capacity = cap;
                    labels.label_width = L;
                    labels.parent_base = static_cast<u32>(b);
                    labels.scratch = d_sort;
                    labels.scratch_rows = d_sort ? cap : 0;
                    labels.scratch_indexed = d_sort ? 1 : 0;
                    labels.error = &rec->error;
                    labels.live = &d_ctl->live;  // (a halted chunk writes nothing)
                    gen->write(labels, d_cand, W);
                    bool widened = false;
                    for (;;)
                    {
                        const auto tm = Clock::now();
                        const bool missing = gen->resolve_missing();
                        st.host_ms += ms_since(tm);
                        if (!missing)
                            break;
                        if (bucket(*task) > W)
                        {
                            widen(bucket(*task), count);
                            widened = true;
                            break;
                        }
                        gen->write(labels, d_cand, W);
                    }
                    if (widened)
                    {
                        if (host)
                            states->sync_to_host();
                        if (o.timings)
                            next_event -= 4;  // the redo records this chunk's events again
                        if (!exact_group)
                        {
                            grow(*states, room);
                            size_scratch(cap_max);  // (the rows are wider)
                        }
                        ck.counted = false;
                        init_ctl();
                        continue;  // redo the chunk at the new width (nothing of it was committed)
                    }
                    if (ev[2])
                        ev[2]->record(s);
                    // after the rows: the parents' arena generation is read up to here
                    if (exact_group && !try_allocate(*ctx, [&](int)
                                                     {
                                                         grow(*states, adds);
                                                         grow(*nodes, adds);
                                                     }))
                    {
                        split(cap / 2);
                        break;
                    }
                    // dedup: deterministic ids in candidate order, the new states after the stored ones (device count)
                    const state_set::Live live{&d_ctl->live};
                    check(state_set::launch_insert(tab(), rows(), {d_cand, W}, cap, d_result, s, live), "launch_insert");
                    check(state_set::launch_rank(tab(), d_result, cap, d_rank, d_scan, tb, s, live), "launch_rank");
                    state_set::Compact cm;
                    cm.result = d_result;
                    cm.rank = d_rank;
                    cm.cand = d_cand;
                    cm.parent = d_parent;
                    cm.seg_offsets = gen->seg_offsets();
                    cm.num_schemas = S;
                    cm.parent_base = static_cast<u32>(b);
                    cm.arena_tail = reinterpret_cast<u64*>(states->device_data());
                    cm.nodes_tail = reinterpret_cast<u32*>(nodes->device_data());
                    cm.base = 0;
                    cm.words = W;
                    cm.offset = &d_ctl->count;
                    check(state_set::launch_compact(tab(), cm, cap, s, live), "launch_compact");
                    check(state_set::launch_advance(d_rank, cap, &d_ctl->count, &rec->fresh, s), "launch_advance");
                    if (ev[3])
                        ev[3]->record(s);
                    break;
                }
            }
            if (resplit)
                continue;  // the chunk again with fewer parents
            if (windowed)
            {
                const bool more = run_windows(plan[i], d_ctl, d_rec, h_ctl, h_rec);
                phase = "the successors of a chunk of states";
                if (more)
                    ++i;
                continue;
            }
            // the group's records: commit the chunks up to the first halted one
            check(cudaMemcpyAsync(h_rec, d_rec, u64{G} * sizeof(state_set::ChunkRecord), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            ++st.groups;
            const u64 before = count;
            usize next = j;
            for (usize k = i; k < j; ++k)
            {
                const state_set::ChunkRecord& x = h_rec[k - i];
                Chunk& ck = plan[k];
                if (ck.counted)
                    ++st.resumed;
                ck.counted = false;
                if (x.halt == 0)
                {
                    if (x.error)
                        throw std::logic_error("mymyr: device BrFS: a sorted segment found no scratch (internal error)");
                    r.expanded += ck.ns;
                    r.generated += x.candidates;
                    r.num_goal_states += x.goals;
                    if (x.goals)
                        first_goal = std::min<u64>(first_goal, ck.b + x.first);
                    count += x.fresh;
                    ++st.chunks;
                    if (ck.ns)
                        ratios[ratio_at++ % k_window] = static_cast<double>(x.candidates) / ck.ns;
                    if (ck.before_goal)
                    {
                        // the parents before the first goal state are expanded: stop there (the CPU's order)
                        r.num_goal_states += 1;
                        stop = true;
                        r.solved = true;
                        goal_id = ck.b + ck.ns;
                        next = plan.size();
                        break;
                    }
                    continue;
                }
                // redo it: with room from its rows when the generator still holds its count (the group's last)
                next = k;
                apply_halt(ck, x, k + 1 == j);
                break;
            }
            states->commit(count - before);
            nodes->commit(count - before);
            if (host)
                states->sync_to_host();
            i = next;
        }
        warm = true;
        if (!stop)
            b = le;
    }
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    if (budget_hit && count >= budget)
    {
        // The CPU stops at the state that fills the store, inside the expansion of its parent; the device commits
        // whole chunks. The last chunk is cut back to that state: the states keep the ids [0, budget) (the CPU's), and
        // the counts become the CPU's, with the parents after that state's parent and the parent's later transitions
        // recounted on the host.
        u32 rec[2];
        check(cudaMemcpyAsync(rec, nodes->device_data() + (budget - 1) * nodes->record_bytes(), sizeof rec,
                              cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (budget)");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        if (rec[0] != k_no_parent)
        {
            const u64 p = rec[0], E = r.expanded;  // the parents [0, E) were expanded
            std::vector<u64> rows((E - p) * W);
            check(cudaMemcpyAsync(rows.data(), states->device_data() + p * W * sizeof(u64), rows.size() * sizeof(u64),
                                  cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (budget)");
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            const WorkspaceLease lease = task->workspace();
            Successors& succ = lease->successors();
            for (u64 id = p; id < E; ++id)
            {
                u64 transitions = 0;
                if (id == p && id == windowed_state)
                    transitions = windowed_transitions;
                else
                {
                    const State state = numeric::decode(*task, rows.data() + (id - p) * W, W);
                    succ.prepare(state.view());
                    succ.generate<true>([&](u32, const ObjectId*, const Delta&) { ++transitions; return true; }, witness,
                                        canonical);
                }
                if (id == p)
                    r.generated -= transitions - (u64{rec[1]} + 1);
                else
                {
                    r.generated -= transitions;
                    r.num_goal_states -= succ.goal_holds() ? 1 : 0;
                }
            }
            r.expanded = p + 1;
            if (r.solved && goal_id > p)  // the goal state after the parents of a chunk: never reached
            {
                r.solved = false;
                r.num_goal_states -= 1;
            }
            if (first_goal > p)
                first_goal = ~u64{0};
        }
        count = budget;
        states->truncate(count);
        nodes->truncate(count);
    }
    r.search_s = std::chrono::duration<double>(Clock::now() - t0).count();
    r.states = count;
    r.exhausted = exhausted;
    r.words = task->words();
    r.fluent_slots = task->atoms().fluent_slots();
    r.memory_bytes = states->capacity() * W * 8 + nodes->capacity() * 8 + slots * 8;
    fold_events();
    st.uploads = gen->uploads();
    const GeneratorStats gs = gen->stats();
    st.axiom_ms = gs.device_axiom_ms;
    st.host_axiom_ms = gs.host_axiom_ms;
    st.axiom_reruns = gs.axiom_reruns;
    st.table_slots = slots;
    st.growths = widened_growths + states->growths() + nodes->growths();
    st.mirror_growths = widened_mirror_growths + states->mirror_growths();
    st.device_bytes = ctx->usage().used_high;
    if (!r.solved && first_goal != ~u64{0})
    {
        r.solved = true;  // a goal state was expanded: its plan, as the CPU BrFS without stop_at_goal
        goal_id = first_goal;
    }
    if (r.solved)
        r.plan = plan_to(goal_id);
    if (o.fingerprint)
    {
        std::vector<u64> all(count * W);
        check(cudaMemcpyAsync(all.data(), states->device_data(), count * W * sizeof(u64), cudaMemcpyDeviceToHost, s),
              "cudaMemcpyAsync (fingerprint)");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        const u64* h = all.data();
        const u32 T = std::clamp<u32>(std::thread::hardware_concurrency() / 4, 1, 8);
        std::vector<u64> part(T, 0);
        std::vector<std::thread> threads;
        auto hash_part = [&](u32 t)
                {
                    u64 x = 0;
                    for (u64 id = t; id < count; id += T)
                    {
                        const u64* row = h + id * W;
                        if (task->numeric_slots())
                        {
                            const State state = numeric::decode(*task, row, W);
                            x ^= brfs_fingerprint_term(id, task->canonical_hash(state.view()));
                        }
                        else
                            x ^= brfs_fingerprint_term(id, task->canonical_hash(StateView{row, bits::trimmed_size(row, W), nullptr, 0}));
                    }
                    part[t] = x;
                };
        start_threads(threads, T, hash_part, [] {});
        for (auto& th : threads)
            th.join();
        for (u64 x : part)
            r.fingerprint ^= x;
    }
    (void)t_start;
    return {std::move(r), st};
}

std::vector<Action> DeviceBrfs::Impl::plan_to(u64 id)
{
    if (!states || id >= states->device_size())
        throw std::out_of_range("mymyr: DeviceBrfs::plan_to: no such state");
    DeviceGuard guard(ctx->device());
    auto fetch = [&](const DeviceArena& a, u64 i, void* dst)
    {
        check(cudaMemcpyAsync(dst, a.device_data() + i * a.record_bytes(), a.record_bytes(), cudaMemcpyDeviceToHost, s),
              "cudaMemcpyAsync (plan)");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    };
    std::vector<std::pair<u64, u32>> chain;  // (state, successor index within its parent)
    for (u64 v = id;;)
    {
        u32 rec[2];
        fetch(*nodes, v, rec);
        if (rec[0] == k_no_parent)
            break;
        chain.emplace_back(v, rec[1]);
        v = rec[0];
    }
    std::reverse(chain.begin(), chain.end());
    std::vector<Action> plan;
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    std::vector<u64> prow(W), crow(W), tmp;
    u64 p = 0;
    for (const auto& [v, k] : chain)
    {
        fetch(*states, p, prow.data());
        fetch(*states, v, crow.data());
        const State parent = numeric::decode(*task, prow.data(), W);
        const u32 nw = parent.size_words();
        succ.prepare(parent.view());
        u32 i = 0;
        bool found = false;
        succ.generate<true>(
            [&](u32 schema, const ObjectId* b, const Delta& d) -> bool
            {
                if (i++ != k)
                    return true;
                const u32 n = apply_delta(prow.data(), nw, d, tmp);
                tmp.resize(std::max<usize>(tmp.size(), W), 0);
                if (task->numeric_slots())
                {
                    const State next(tmp.data(), n, d.num, d.nnum);
                    std::vector<u64> encoded(W);
                    numeric::encode(*task, next.view(), encoded.data(), W);
                    found = encoded == crow;
                }
                else
                    found = n <= W && std::equal(tmp.begin(), tmp.begin() + n, crow.begin()) &&
                            std::all_of(crow.begin() + n, crow.end(), [](u64 x) { return x == 0; });
                plan.emplace_back(SchemaId{schema}, std::vector<ObjectId>(b, b + succ.arity(schema)));
                return false;
            },
            o.witness_pruning, o.canonical_order);
        if (!found)
            throw std::logic_error("mymyr: device BrFS: the plan replay does not reproduce state " + std::to_string(v) +
                                   " (device and CPU successor orders differ)");
        p = v;
    }
    return plan;
}
}  // namespace mymyr::cuda
