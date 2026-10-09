// The device state spaces (include/mymyr/cuda/state_space.hpp).

#include "mymyr/cuda/state_space.hpp"
#include "mymyr/cuda/numeric.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"

#include "cost_program_build.hpp"
#include "state_space_kernels.hpp"
#include "table_launch.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/state_set.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace mymyr::cuda
{
struct DeviceStateSpace::Storage
{
    ContextPtr ctx;
    cudaStream_t s = nullptr;
    DeviceBuffer rows, foff, targets, schemas, bindings, costs, boff, bsrc, bedges, unit, cost, goal, unsolvable, alive;
};

namespace
{
using Clock = std::chrono::steady_clock;
using datasets::StateSpaceStatus;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
double s_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

template<class T>
T* ptr(const DeviceBuffer& b)
{
    return static_cast<T*>(b.data());
}

template<class T>
T* ptr(Scratch& b, const ContextPtr& ctx, u64 n, cudaStream_t s)
{
    return static_cast<T*>(b.ensure(ctx, std::max<u64>(n, 1) * sizeof(T), s));
}

DeviceBuffer alloc(const ContextPtr& ctx, u64 bytes, cudaStream_t s) { return DeviceBuffer(ctx, std::max<u64>(bytes, 8), s); }

template<class T>
DeviceBuffer upload(const ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    DeviceBuffer b = alloc(ctx, v.size() * sizeof(T), s);
    if (!v.empty())
        check(cudaMemcpyAsync(b.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (upload)");
    return b;
}

template<class T>
void download(const T* src, u64 n, std::vector<T>& dst, cudaStream_t s)
{
    dst.resize(n);
    if (n)
        check(cudaMemcpyAsync(dst.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (download)");
}

void sync(cudaStream_t s) { check(cudaStreamSynchronize(s), "cudaStreamSynchronize"); }

/// A device array that grows (doubling at least: its growth copies stay below its final size), keeping its first
/// elements.
template<class T>
struct Grow
{
    DeviceBuffer buf;
    u64 cap = 0;

    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(buf.data()); }
    void reserve(const ContextPtr& ctx, u64 n, u64 keep, cudaStream_t s)
    {
        if (n <= cap)
            return;
        const u64 nc = std::max<u64>(n, 2 * cap + 1024);
        DeviceBuffer nb(ctx, nc * sizeof(T), s);
        keep = std::min(keep, cap);
        if (keep)
            check(cudaMemcpyAsync(nb.data(), buf.data(), keep * sizeof(T), cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync (grow)");
        buf = std::move(nb);
        cap = nc;
    }
    void release() noexcept
    {
        buf.reset();
        cap = 0;
    }
};

/// Bits of the values [0, n).
u32 bits_for(u64 n)
{
    u32 b = 1;
    while (b < 64 && (u64{1} << b) < n)
        ++b;
    return b;
}

/// Row width of a lazy task's rows (fixed buckets; lazy slots widen by relayout).
u32 bucket(u32 words)
{
    const u32 w = std::bit_ceil(std::max<u32>(words, 1));
    if (w > lifted::k_max_words)
        throw std::invalid_argument("mymyr: device state space: the states grew wider than " +
                                    std::to_string(lifted::k_max_words) + " words (the device kernels' limit)");
    return w;
}

u32 max_arity(const Task& task)
{
    u32 k = 0;
    for (const auto& sc : task.data().schemas)
        k = std::max(k, sc.arity());
    return k;
}

u32 resolve_threads(u32 threads) { return threads == 0 ? std::max<u32>(1, std::thread::hardware_concurrency()) : threads; }

// ------------------------------------------------------------------------------------------------ host output
/// One device-to-host copy of a download.
struct Copy
{
    void* dst = nullptr;
    const void* src = nullptr;
    u64 bytes = 0;
};

/// Asks for transparent huge pages over the whole pages of [p, p + bytes) (Linux; a hint): the first touch of large
/// host arrays then faults 2 MB at a time.
void advise_huge(void* p, u64 bytes)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr u64 page = u64{2} << 20;
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t lo = (a + page - 1) / page * page, hi = (a + bytes) / page * page;
    if (hi > lo)
        (void)madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
#else
    (void)p;
    (void)bytes;
#endif
}

/// Runs `jobs` (0 .. n) on up to `threads` threads (the first exception is rethrown).
void parallel(u64 n, u32 threads, const std::function<void(u64)>& job)
{
    std::atomic<u64> next{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto worker = [&]
    {
        try
        {
            for (u64 i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n && !failed.load(std::memory_order_relaxed);)
                job(i);
        }
        catch (...)
        {
            std::lock_guard lock(error_mutex);
            if (!error)
                error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    const u32 T = static_cast<u32>(std::min<u64>(std::max<u32>(threads, 1), std::max<u64>(n, 1)));
    std::vector<std::thread> pool;
    for (u32 t = 1; t < T; ++t)
        pool.emplace_back(worker);
    worker();
    for (auto& th : pool)
        th.join();
    if (error)
        std::rethrow_exception(error);
}

/// The sizes of a space's host arrays (the CPU generator's layout).
struct Shape
{
    u64 n = 0, e = 0;
    u32 words = 1, label_width = 0;
    bool labels = false, costs = false;
};

Shape shape_of(const DeviceStateSpace& d)
{
    return {d.num_states(), d.num_transitions(), d.row_words(), d.label_width(), d.has_labels(), !d.unit_costs()};
}

/// The host arrays of a space, in the CPU generator's order (for_arrays).
enum Array : u32
{
    a_rows,
    a_foff,
    a_targets,
    a_schemas,
    a_bindings,
    a_costs,
    a_boff,
    a_bsrc,
    a_bedges,
    a_unit,
    a_cost,
    a_goal,
    a_unsolvable,
    a_alive,
};

/// Where array `a` of the space at `part` starts in its wave's storage (DeviceStateSpace's accessors).
const void* array_source(const DeviceStateSpace::Storage& st, const DeviceStateSpace::Part& part, u32 label_width, u32 a)
{
    switch (a)
    {
        case a_rows: return ptr<u64>(st.rows) + part.rows;
        case a_foff: return ptr<u64>(st.foff) + part.offsets;
        case a_targets: return ptr<u32>(st.targets) + part.edges;
        case a_schemas: return ptr<u32>(st.schemas) + part.edges;
        case a_bindings: return ptr<u32>(st.bindings) + part.edges * label_width;
        case a_costs: return ptr<f64>(st.costs) + part.edges;
        case a_boff: return ptr<u64>(st.boff) + part.offsets;
        case a_bsrc: return ptr<u32>(st.bsrc) + part.edges;
        case a_bedges: return ptr<u32>(st.bedges) + part.edges;
        case a_unit: return ptr<i32>(st.unit) + part.states;
        case a_cost: return ptr<f64>(st.cost) + part.states;
        case a_goal: return ptr<u8>(st.goal) + part.states;
        case a_unsolvable: return ptr<u8>(st.unsolvable) + part.states;
        default: return ptr<u8>(st.alive) + part.states;
    }
}

/// Array `a` of a device space (its accessors).
const void* device_array(const DeviceStateSpace& d, u32 a)
{
    switch (a)
    {
        case a_rows: return d.state_words();
        case a_foff: return d.forward_offsets();
        case a_targets: return d.forward_targets();
        case a_schemas: return d.label_schemas();
        case a_bindings: return d.label_bindings();
        case a_costs: return d.costs();
        case a_boff: return d.backward_offsets();
        case a_bsrc: return d.backward_sources();
        case a_bedges: return d.backward_edges();
        case a_unit: return d.unit_goal_distances();
        case a_cost: return d.cost_goal_distances();
        case a_goal: return d.goal_flags();
        case a_unsolvable: return d.unsolvable_flags();
        default: return d.alive_flags();
    }
}

/// Calls f(array, vector, elements) for every host array of a space of shape `s` (Array order); arrays the space does
/// not have get 0 elements.
template <class F>
void for_arrays(datasets::StateSpaceArrays& a, const Shape& s, F&& f)
{
    const u64 N = s.n, E = s.e;
    f(a_rows, a.state_words, N * s.words);
    f(a_foff, a.forward_offsets, N + 1);
    f(a_targets, a.forward_targets, E);
    f(a_schemas, a.label_schemas, s.labels ? E : 0);
    f(a_bindings, a.label_bindings, s.labels ? E * s.label_width : 0);
    f(a_costs, a.costs, s.costs ? E : 0);
    f(a_boff, a.backward_offsets, N + 1);
    f(a_bsrc, a.backward_sources, E);
    f(a_bedges, a.backward_edges, E);
    f(a_unit, a.unit_goal_distances, N);
    f(a_cost, a.cost_goal_distances, N);
    f(a_goal, a.goal_flags, N);
    f(a_unsolvable, a.unsolvable_flags, N);
    f(a_alive, a.alive_flags, N);
}

/// The flags of states [lo, hi) and, with `cost`, their unit-cost goal distances, from the unit goal distances and
/// the goal flags (the device's k_flags).
void derive_flags(datasets::StateSpaceArrays& a, u64 lo, u64 hi, bool cost)
{
    for (u64 i = lo; i < hi; ++i)
    {
        const i32 d = a.unit_goal_distances[i];
        const bool g = a.goal_flags[i] != 0, u = d < 0;
        a.unsolvable_flags[i] = u ? 1 : 0;
        a.alive_flags[i] = !g && !u ? 1 : 0;
        if (cost)
            a.cost_goal_distances[i] = u ? std::numeric_limits<f64>::infinity() : static_cast<f64>(d);
    }
}

/// The backward CSR from the forward one: the (target, edge) pairs sorted stably by target (the device's sort).
void derive_backward(datasets::StateSpaceArrays& a)
{
    const u64 N = a.num_states;
    std::vector<u64>& off = a.backward_offsets;
    std::fill(off.begin(), off.end(), u64{0});
    for (const u32 t : a.forward_targets)
        ++off[u64{t} + 1];
    for (u64 i = 0; i < N; ++i)
        off[i + 1] += off[i];
    std::vector<u64> pos(off.begin(), off.end() - 1);
    for (u64 v = 0; v < N; ++v)
        for (u64 e = a.forward_offsets[v]; e < a.forward_offsets[v + 1]; ++e)
        {
            const u64 j = pos[a.forward_targets[e]]++;
            a.backward_edges[j] = static_cast<u32>(e);
            a.backward_sources[j] = static_cast<u32>(v);
        }
}

/// Sizes `v` to n elements (first touch: transparent huge pages asked for, then value-initialized).
template <class T>
void size_array(std::vector<T>& v, u64 n)
{
    if (v.size() == n)
        return;
    if (n == 0)
    {
        std::vector<T>().swap(v);
        return;
    }
    v.reserve(n);
    advise_huge(v.data(), n * sizeof(T));
    v.resize(n);
}

/// Device-to-host copies through pinned staging (pageable destinations copy at about half the bus's rate): `workers`
/// threads, each with a stream and two pinned pieces of the context's, download one piece while copying the other
/// into its destination. The sources' work must be complete.
void staged_download(DeviceContext& ctx, std::span<const Copy> copies, u32 workers)
{
    constexpr u64 piece = u64{8} << 20;
    constexpr u64 none = ~u64{0};
    std::vector<Copy> pieces;
    for (const Copy& c : copies)
        for (u64 o = 0; o < c.bytes; o += piece)
            pieces.push_back({static_cast<std::byte*>(c.dst) + o, static_cast<const std::byte*>(c.src) + o, std::min(piece, c.bytes - o)});
    if (pieces.empty())
        return;
    const u32 T = static_cast<u32>(std::min<u64>(std::max<u32>(workers, 1), pieces.size()));
    DeviceGuard guard(ctx.device());
    const PinnedLease stage = ctx.lease_pinned(2 * T, piece);
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<Event> events;
    for (u32 t = 0; t < T; ++t)
    {
        streams.push_back(std::make_unique<Stream>());
        events.emplace_back();
        events.emplace_back();
    }
    std::atomic<u64> next{0};
    parallel(T, T,
             [&](u64 t)
             {
                 DeviceGuard g(ctx.device());
                 const cudaStream_t st = streams[t]->get();
                 u64 pend[2] = {none, none};
                 auto finish = [&](u32 b)
                 {
                     events[2 * t + b].synchronize();
                     const Copy& c = pieces[pend[b]];
                     std::memcpy(c.dst, stage.data(static_cast<u32>(2 * t + b)), c.bytes);
                     pend[b] = none;
                 };
                 try
                 {
                     for (u32 b = 0;; b ^= 1)
                     {
                         const u64 i = next.fetch_add(1, std::memory_order_relaxed);
                         const bool more = i < pieces.size();
                         if (more)
                         {
                             check(cudaMemcpyAsync(stage.data(static_cast<u32>(2 * t + b)), pieces[i].src, pieces[i].bytes,
                                                   cudaMemcpyDeviceToHost, st),
                                   "cudaMemcpyAsync (host output)");
                             events[2 * t + b].record(st);
                             pend[b] = i;
                         }
                         if (pend[b ^ 1] != none)
                             finish(b ^ 1);  // (its download ran while this one was enqueued)
                         if (!more)
                             break;
                     }
                 }
                 catch (...)
                 {
                     (void)cudaStreamSynchronize(st);  // (the staging buffers go back to the context)
                     throw;
                 }
             });
}

/// Host output, overlapped with the device's post-processing. presize() sizes the host arrays of a wave's members
/// in the background (the first touch of fresh pages, in parallel), in phases of arrays; early() downloads arrays that
/// are final before the post-processing ends, in the background too (each once its phase is sized, through pinned
/// staging: staged_download), while later phases are sized; finish() downloads the rest, derives some on the host and
/// builds the spaces. Without presize, finish sizes the arrays first. A member's arrays are sized to its space's exact
/// shape either way (a presize guess that differs, as costs of a member whose costs turn out unit, is corrected; an
/// early download of an array whose size changed is redone).
class HostOutput
{
public:
    HostOutput(ContextPtr ctx, u32 threads) : m_ctx(std::move(ctx)), m_threads(std::max<u32>(1, threads)) {}
    ~HostOutput()
    {
        // (a background error is dropped: finish was not reached, the caller has an exception of its own)
        if (m_toucher.joinable())
            m_toucher.join();
        if (m_downloader.joinable())
            m_downloader.join();
    }
    HostOutput(const HostOutput&) = delete;
    HostOutput& operator=(const HostOutput&) = delete;

    /// Sizes slot i's arrays to shapes[i] (n == 0: none) in the background: the arrays of phases[0] (Array bits)
    /// first, then those of phases[1], ..., then the others.
    void presize(std::vector<Shape> shapes, std::vector<u32> phases)
    {
        m_shapes = std::move(shapes);
        m_arrays.assign(m_shapes.size(), {});
        m_early.assign(m_shapes.size(), 0);
        u32 all = 0;
        for (const u32 p : phases)
            all |= p;
        phases.push_back(~all);
        m_phases = phases;
        m_sized.clear();
        std::vector<std::promise<void>> done(phases.size());
        for (auto& d : done)
            m_sized.push_back(d.get_future().share());
        // one pool over the arrays of every phase, handed out in phase order: a phase is done when its last array is
        // (phase by phase with a pool each, a wave of few spaces sized its arrays one after another)
        m_toucher = std::thread(
            [this, phases = std::move(phases), done = std::move(done)]() mutable
            {
                const usize np = phases.size();
                std::vector<std::function<void()>> jobs;
                std::vector<u32> phase_of;
                std::vector<std::atomic<u64>> left(np);
                std::vector<std::atomic<bool>> set(np);
                auto finish = [&](usize p) { if (!set[p].exchange(true)) done[p].set_value(); };
                try
                {
                    for (usize i = 0; i < np; ++i)
                        for (usize j = 0; j < m_shapes.size(); ++j)
                            if (m_shapes[j].n)
                                for_arrays(m_arrays[j], m_shapes[j],
                                           [&]<class T>(u32 a, std::vector<T>& v, u64 n)
                                           {
                                               if (n && (phases[i] >> a & 1))
                                               {
                                                   jobs.emplace_back([&v, n] { size_array(v, n); });
                                                   phase_of.push_back(static_cast<u32>(i));
                                                   ++left[i];
                                               }
                                           });
                    for (usize i = 0; i < np; ++i)
                        if (left[i] == 0)
                            finish(i);
                    parallel(jobs.size(), m_threads,
                             [&](u64 k)
                             {
                                 jobs[k]();
                                 if (left[phase_of[k]].fetch_sub(1) == 1)
                                     finish(phase_of[k]);
                             });
                }
                catch (...)
                {
                    for (usize i = 0; i < np; ++i)
                        if (!set[i].exchange(true))
                            done[i].set_exception(std::current_exception());
                }
            });
    }

    /// Downloads the arrays `arrays` (Array bits) of every presized slot from `storage` (parts[i]: slot i's part) once
    /// they are sized and the work enqueued on `s` so far is done; in the background, after the downloads before it.
    void early(u32 arrays, std::shared_ptr<const DeviceStateSpace::Storage> storage, std::vector<DeviceStateSpace::Part> parts,
               u32 label_width, cudaStream_t s)
    {
        auto done = std::make_shared<Event>();
        done->record(s);
        m_downloader = std::thread(
            [this, prev = std::move(m_downloader), arrays, storage = std::move(storage), parts = std::move(parts), label_width,
             done]() mutable
            {
                if (prev.joinable())
                    prev.join();
                if (m_error)
                    return;
                try
                {
                    done->synchronize();
                    // phase by phase: a phase's downloads overlap the sizing of the next
                    for (usize p = 0; p < m_phases.size(); ++p)
                    {
                        const u32 mine = m_phases[p] & arrays;
                        if (!mine)
                            continue;
                        m_sized[p].get();
                        std::vector<Copy> copies;
                        for (usize i = 0; i < m_shapes.size(); ++i)
                            if (m_shapes[i].n)
                                for_arrays(m_arrays[i], m_shapes[i],
                                           [&]<class T>(u32 a, std::vector<T>& v, u64 n)
                                           {
                                               if (!(mine >> a & 1) || n == 0)
                                                   return;
                                               copies.push_back({v.data(), array_source(*storage, parts[i], label_width, a), n * sizeof(T)});
                                               m_early[i] |= 1u << a;
                                           });
                        staged_download(*m_ctx, copies, std::min<u32>(m_threads, 4));
                    }
                }
                catch (...)
                {
                    m_error = std::current_exception();
                }
            });
    }

    /// The host spaces of `spaces` (their storage's work must be complete); spaces[j] fills slot slots[j]'s arrays.
    std::vector<datasets::StateSpacePtr> finish(std::span<const DeviceStateSpace* const> spaces, std::span<const u32> slots)
    {
        if (m_toucher.joinable())
            m_toucher.join();
        if (m_downloader.joinable())
            m_downloader.join();
        for (const auto& f : m_sized)
            f.get();  // (a sizing error)
        if (m_error)
            std::rethrow_exception(std::exchange(m_error, nullptr));
        const u64 n = spaces.size();
        std::vector<datasets::StateSpaceArrays> arrays(n);
        std::vector<u32> early(n, 0);
        for (u64 j = 0; j < n; ++j)
            if (slots[j] < m_arrays.size())
            {
                arrays[j] = std::move(m_arrays[slots[j]]);
                early[j] = m_early[slots[j]];
            }
        m_arrays.clear();
        // the arrays to size, then the copies of those not downloaded early
        std::vector<std::function<void()>> sizes;
        for (u64 j = 0; j < n; ++j)
        {
            const DeviceStateSpace& d = *spaces[j];
            datasets::StateSpaceArrays& a = arrays[j];
            a.task = d.task();
            a.num_states = d.num_states();
            a.words = d.words();
            a.numeric_words = d.numeric_words();
            a.labels = d.has_labels();
            a.label_width = d.label_width();
            a.threads = 0;
            a.layers = d.layers();
            for_arrays(a, shape_of(d),
                       [&]<class T>(u32 k, std::vector<T>& v, u64 count)
                       {
                           if (v.size() != count)
                           {
                               early[j] &= ~(1u << k);
                               sizes.emplace_back([&v, count] { size_array(v, count); });
                           }
                       });
        }
        parallel(sizes.size(), m_threads, [&](u64 i) { sizes[i](); });
        // arrays derived on the host instead of downloaded (the bus is the bottleneck, the CPU idle): the flags and the
        // unit-cost goal distances from the unit distances and the goal flags; the backward CSR from the forward one
        // where the spaces are small enough to share the threads (a counting sort per space)
        u64 e_total = 0;
        for (u64 j = 0; j < n; ++j)
            e_total += spaces[j]->num_transitions();
        std::vector<u32> derived(n, 0);
        for (u64 j = 0; j < n; ++j)
        {
            derived[j] = 1u << a_unsolvable | 1u << a_alive | (spaces[j]->unit_costs() ? 1u << a_cost : 0u);
            if (spaces[j]->num_transitions() * 8 <= e_total)
                derived[j] |= 1u << a_boff | 1u << a_bsrc | 1u << a_bedges;
        }
        std::vector<Copy> copies;
        for (u64 j = 0; j < n; ++j)
        {
            const DeviceStateSpace& d = *spaces[j];
            for_arrays(arrays[j], shape_of(d),
                       [&]<class T>(u32 k, std::vector<T>& v, u64 count)
                       {
                           if (count && !((early[j] | derived[j]) >> k & 1))
                               copies.push_back({v.data(), device_array(d, k), count * sizeof(T)});
                       });
        }
        staged_download(*m_ctx, copies, std::min<u32>(m_threads, 4));
        std::vector<std::function<void()>> jobs;
        for (u64 j = 0; j < n; ++j)
            if (derived[j] >> a_bsrc & 1)
                jobs.emplace_back([&a = arrays[j]] { derive_backward(a); });
        constexpr u64 block = u64{1} << 20;
        for (u64 j = 0; j < n; ++j)
            for (u64 lo = 0; lo < arrays[j].num_states; lo += block)
                jobs.emplace_back([&a = arrays[j], lo, hi = std::min<u64>(arrays[j].num_states, lo + block), cost = derived[j] >> a_cost & 1]
                                  { derive_flags(a, lo, hi, cost != 0); });
        parallel(jobs.size(), m_threads, [&](u64 i) { jobs[i](); });
        std::vector<datasets::StateSpacePtr> out(n);
        parallel(n, m_threads, [&](u64 j) { out[j] = datasets::StateSpace::create(std::move(arrays[j])); });
        return out;
    }

private:
    ContextPtr m_ctx;
    u32 m_threads = 1;
    std::vector<Shape> m_shapes;
    std::vector<datasets::StateSpaceArrays> m_arrays;
    std::vector<u32> m_early;                     // per slot: the arrays early() downloaded (Array bits)
    std::vector<u32> m_phases;                    // presize's phases (Array bits; the last: the others)
    std::vector<std::shared_future<void>> m_sized;  // per phase: sized
    std::thread m_toucher;                        // presize
    std::thread m_downloader;                     // the last early() (each joins the one before it)
    std::exception_ptr m_error;                   // the downloads' first error
};

enum class CostMode : u8
{
    Unit,    // no costs (every instance's ActionCosts is Unit)
    Device,  // state-independent costs from programs (mymyr/cuda/cost_program.hpp), on the device
    Host,    // regenerated on the CPU with the generator's formula
};

/// Candidates per chunk at most (the chunk is cut when its successors exceed it, as long as it has two parents).
constexpr u64 k_candidate_bytes = u64{768} << 20;

/// Per-wave generation state: the states of the wave's instances (members, consecutive table ids from `first`) in one
/// id space, their transitions, the state set.
struct Wave
{
    u32 first = 0, k = 0;
    u32 W = 1;
    Grow<u64> rows;
    Grow<u32> inst, depth;
    Grow<u8> goal;
    Grow<u64> off;
    Grow<u32> targets, schemas, bindings;
    Grow<f64> costs;
    u64 count = 0, edges = 0;
    u64 offsets = 0;  // off[0, offsets) written: the expanded parents' offsets and the next one's
    u64 goals = 0;    // goal[0, goals) written (a chunk's parents' flags are written when it is counted)
    DeviceBuffer table;
    u64 slots = 0;
    DeviceBuffer live;          // [capacity] member -> table id (k_no_instance: not expanded)
    std::vector<u32> host_live;
    std::vector<u32> icount;    // states per member (after the last chunk)
    std::vector<u64> last;      // an upper bound of the member's largest state id
    std::vector<u8> dead;
    std::vector<StateSpaceStatus> status;
    std::vector<Clock::time_point> t0;
    Clock::time_point start;

    [[nodiscard]] ssk::Table tab() const { return {ptr<u64>(table), slots - 1}; }
    [[nodiscard]] ssk::Store store() const { return {rows.data(), W, inst.data()}; }
};

class Engine
{
public:
    Engine(ContextPtr ctx, std::vector<TaskPtr> tasks, rl::TaskTablePtr table, const DeviceStateSpaceOptions& o)
        : m_ctx(std::move(ctx)), m_o(o), m_tasks(std::move(tasks)), m_table(std::move(table))
    {
        if (!m_ctx)
            throw std::invalid_argument("mymyr: device state space: null context");
        if (m_o.space.symmetry_pruning)
            throw std::invalid_argument("mymyr: device state space: symmetry pruning runs on the CPU only "
                                        "(datasets::generate_state_space)");
        if (m_o.chunk_states == 0)
            throw std::invalid_argument("mymyr: device state space: chunk_states must be at least 1");
        m_s = m_o.stream ? m_o.stream : m_ctx->stream();
        m_threads = resolve_threads(m_o.space.threads);
        for (u32 i = 0; i < m_tasks.size(); ++i)
        {
            const std::string why = state_space_unsupported(*m_tasks[i]);
            if (!why.empty())
                throw std::invalid_argument("mymyr: device state space" +
                                            (m_tasks.size() > 1 ? " (instance " + std::to_string(i) + ")" : std::string()) +
                                            ": " + why);
        }
        m_L = max_arity(*m_tasks.front());
        m_S = static_cast<u32>(m_tasks.front()->data().schemas.size());
        m_multi = m_table && !m_table->numeric() && detail::multi_unsupported(*m_table).empty();
        m_st.multi = m_multi;
        setup_costs();
    }

    DeviceStateSpaces run()
    {
        DeviceGuard guard(m_ctx->device());
        const u32 I = static_cast<u32>(m_tasks.size());
        m_results.assign(I, {});
        if (m_multi)
        {
            m_dt = DeviceTaskTable::upload(m_ctx, m_table);
            m_dt->acquire(m_s);
            m_plan = detail::plan_launches(m_ctx, *m_dt, m_s);
            u32 next = 0;
            while (next < I)
                next = run_wave(next);
        }
        else
            for (u32 i = 0; i < I; ++i)
                run_wave(i);
        sync(m_s);
        m_st.device_bytes = m_ctx->usage().used_high;
        return {std::move(m_results), m_st};
    }

private:
    // ------------------------------------------------------------------------------------------ costs
    void setup_costs()
    {
        const u32 I = static_cast<u32>(m_tasks.size());
        m_costs.reserve(I);
        bool unit = true, independent = true;
        for (const TaskPtr& t : m_tasks)
        {
            m_costs.emplace_back(*t);
            const heuristics::ActionCosts& c = m_costs.back();
            unit = unit && c.unit();
            independent = independent && (c.unit() || (c.kind() == heuristics::ActionCosts::Kind::TotalCost && c.state_independent()));
        }
        if (unit)
        {
            m_cost_mode = CostMode::Unit;
            return;
        }
        m_cost_mode = CostMode::Host;
        if (!independent)
            return;
        // the programs of every instance's costs (any number of cost parameters, any static function key space)
        std::vector<const Task*> ts;
        std::vector<const heuristics::ActionCosts*> cs;
        for (u32 i = 0; i < I; ++i)
        {
            ts.push_back(m_tasks[i].get());
            cs.push_back(&m_costs[i]);
        }
        m_programs = costs::build(m_ctx, ts, cs, m_s);
        if (!m_programs.why.empty())
            return;  // (costs that depend on the state stay on the host; the check above already sent them there)
        m_cost_mode = CostMode::Device;
    }

    // ------------------------------------------------------------------------------------------ waves
    void alloc_table(Wave& w, u64 n)
    {
        w.slots = n;
        w.table = DeviceBuffer(m_ctx, n * sizeof(u64), m_s);
        check(cudaMemsetAsync(w.table.data(), 0, n * sizeof(u64), m_s), "cudaMemsetAsync (state table)");
    }

    /// Room for `more` new states at load <= 1/2 (growing to load <= 1/4).
    void ensure_table(Wave& w, u64 more)
    {
        if (w.count + more <= w.slots / 2)
            return;
        u64 n = w.slots;
        while (w.count + more > n / 4)
            n *= 2;
        alloc_table(w, n);
        check(ssk::launch_rehash(w.tab(), w.store(), 0, w.count, m_s), "launch_rehash");
        ++m_st.rehashes;
    }

    /// The per-state arrays hold `n` states, the offsets n + 1.
    void reserve_states(Wave& w, u64 n)
    {
        w.rows.reserve(m_ctx, n * w.W, w.count * w.W, m_s);
        w.inst.reserve(m_ctx, n, w.count, m_s);
        w.depth.reserve(m_ctx, n, w.count, m_s);
        w.goal.reserve(m_ctx, n, w.goals, m_s);
        w.off.reserve(m_ctx, n + 1, w.offsets, m_s);  // unexpanded states have no offsets yet
    }

    /// Admits members [w.k, w.k + n) (table ids first + w.k ...): their initial states.
    void admit(Wave& w, u32 n)
    {
        if (n == 0)
            return;
        std::vector<u64> rows(u64{n} * w.W, 0);
        std::vector<u32> inst(n), zero(n, 0);
        for (u32 j = 0; j < n; ++j)
        {
            const u32 m = w.k + j, id = w.first + m;
            if (m_multi)
            {
                const std::vector<u64>& init = m_table->instance(id).init;
                std::copy_n(init.begin(), std::min<u64>(w.W, init.size()), rows.begin() + static_cast<std::ptrdiff_t>(u64{j} * w.W));
            }
            else
            {
                const State s0 = m_tasks[id]->initial_state();
                numeric::encode(*m_tasks[id], s0.view(), rows.data() + u64{j} * w.W, w.W);
            }
            inst[j] = m;
            w.host_live[m] = m_multi ? id : 0;
            w.icount.push_back(1);
            w.last.push_back(w.count + j);
            w.dead.push_back(0);
            w.status.push_back(StateSpaceStatus::Ok);
            w.t0.push_back(Clock::now());
        }
        reserve_states(w, w.count + n);
        ensure_table(w, n);
        check(cudaMemcpyAsync(w.rows.data() + w.count * w.W, rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, m_s),
              "cudaMemcpyAsync (roots)");
        check(cudaMemcpyAsync(w.inst.data() + w.count, inst.data(), n * sizeof(u32), cudaMemcpyHostToDevice, m_s),
              "cudaMemcpyAsync (roots)");
        check(cudaMemcpyAsync(w.depth.data() + w.count, zero.data(), n * sizeof(u32), cudaMemcpyHostToDevice, m_s),
              "cudaMemcpyAsync (roots)");
        if (m_multi)
        {
            check(cudaMemcpyAsync(ptr<u32>(w.live) + w.k, w.host_live.data() + w.k, n * sizeof(u32), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (live)");
            std::vector<u32> one(n, 1);
            check(cudaMemcpyAsync(counts_dev() + w.k, one.data(), n * sizeof(u32), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (counts)");
            sync(m_s);  // the pageable sources
        }
        else
            sync(m_s);
        check(ssk::launch_rehash(w.tab(), w.store(), w.count, w.count + n, m_s), "launch_rehash");
        w.count += n;
        w.k += n;
    }

    /// Marks member m as no longer expanded.
    void kill(Wave& w, u32 m, StateSpaceStatus why)
    {
        if (w.dead[m])
            return;
        w.dead[m] = 1;
        w.status[m] = why;
        w.host_live[m] = ssk::k_no_instance;
        if (m_multi)
            check(cudaMemcpyAsync(ptr<u32>(w.live) + m, &w.host_live[m], sizeof(u32), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (live)");
    }

    /// Runs the wave starting at table id `first`; returns the next table id not admitted.
    u32 run_wave(u32 first)
    {
        const u32 I = static_cast<u32>(m_tasks.size());
        Wave w;
        w.first = first;
        w.start = Clock::now();
        ++m_st.waves;
        // instances whose goal is statically false are Unsolvable at once (as in the CPU generator: the BrFS ends
        // before expanding), and are not admitted; they end the wave's range like others
        const u32 cap = m_multi ? I - first : 1;
        w.host_live.assign(cap, ssk::k_no_instance);
        m_dctl = alloc(m_ctx, (u64{k_ctl_counts} + cap) * sizeof(u32), m_s);
        check(cudaMemsetAsync(m_dctl.data(), 0, (u64{k_ctl_counts} + cap) * sizeof(u32), m_s), "cudaMemsetAsync");
        m_hctl.assign(k_ctl_counts + cap, 0);
        if (m_multi)
        {
            w.live = alloc(m_ctx, u64{cap} * sizeof(u32), m_s);
            w.W = std::max<u32>(1, m_table->words());
        }
        else
        {
            const Task& t = *m_tasks[first];
            w.W = t.atoms().mode() == AtomMode::Frozen ? std::max<u32>(1, t.words()) + t.numeric_slots()
                                                      : bucket(t.words() + t.numeric_slots());
            m_gen = std::make_unique<ChunkGenerator>(m_ctx, m_tasks[first], m_s);
        }
        const u64 expected = m_o.expected_states;
        const u64 cap0 = expected ? expected + expected / 8 + 1024 : u64{1} << 14;
        reserve_states(w, cap0);
        alloc_table(w, expected ? std::bit_ceil(std::max<u64>(u64{1} << 16, 4 * expected)) : u64{1} << 16);
        const u64 limit = std::max<u64>(m_o.space.max_states, 2);
        const bool timed = std::isfinite(m_o.space.max_seconds);
        u32 next = first;
        // admission: members are consecutive table ids; an instance whose goal is statically false is Unsolvable at
        // once (its BrFS ends before expanding, as in the CPU generator) and takes a member slot without states
        auto admit_more = [&]
        {
            u32 n = 0;
            auto room = [&]
            {
                return next + n < I && next + n - first < cap && (m_o.wave_instances == 0 || w.k + n < m_o.wave_instances) &&
                       (w.count + n < m_o.wave_states || w.k + n == 0);
            };
            while (room())
            {
                if (unsatisfiable(next + n))
                {
                    admit(w, n);
                    next += n;
                    n = 0;
                    admit_placeholder(w);
                    ++next;
                    continue;
                }
                ++n;
            }
            admit(w, n);
            next += n;
        };
        admit_more();
        const auto tg = Clock::now();
        u64 b = 0;
        u32 chunk_cap = chunk_capacity();
        for (;;)
        {
            if (b == w.count)
            {
                if (m_multi)
                    admit_more();
                if (b == w.count)
                    break;
            }
            bool all_dead = true;
            for (u32 m = 0; m < w.k; ++m)
                all_dead = all_dead && w.dead[m];
            if (all_dead && next >= I)
                break;
            const auto ns = static_cast<u32>(std::min<u64>({m_o.chunk_states, w.count - b, chunk_cap}));
            const ChunkStart at{b, w.count, w.edges, w.icount};
            const u32 done = expand(w, b, ns);
            b += done;
            ++m_st.chunks;
            // budgets: max_states (fails iff the space has max(M, 2) states or more; `states` is the count after
            // the parent whose expansion reached it) and max_seconds (checked between chunks)
            ChunkTail tail;
            for (u32 m = 0; m < w.k; ++m)
            {
                if (w.dead[m])
                    continue;
                if (w.icount[m] >= limit)
                {
                    w.icount[m] = failure_count(w, m, at, done, limit, tail);
                    kill(w, m, StateSpaceStatus::OutOfStates);
                }
                else if (timed && b <= w.last[m] && s_since(w.t0[m]) > m_o.space.max_seconds)
                    kill(w, m, StateSpaceStatus::Timeout);
            }
            if (!m_multi && w.dead[0])
                break;
            if (m_multi)
                admit_more();
        }
        read_control(0);
        check_costs_flag();
        m_st.generate_ms += ms_since(tg);
        m_st.states += w.count;
        m_st.transitions += w.edges;
        w.table.reset();
        release_scratch();
        post(w);
        if (!m_multi)
            m_gen.reset();
        return m_multi ? next : first + 1;
    }

    /// A chunk's start: its first parent, the wave's states and transitions, and each member's count then.
    struct ChunkStart
    {
        u64 b = 0, count = 0, edges = 0;
        std::vector<u32> icount;
    };
    /// The chunk's transitions on the host, downloaded once for the members that failed in it.
    struct ChunkTail
    {
        bool loaded = false;
        std::vector<u64> off;                  // per parent of the chunk, and its end: absolute edge offsets
        std::vector<u32> targets;              // the chunk's transitions
        std::vector<u32> parent_inst, new_inst;  // multi: the parents' members and the new states' members
    };

    /// The CPU generator's `states` when member m fails with OutOfStates in the chunk that expanded `done` parents
    /// from `at`: the CPU generator (one thread, as the instance pool) expands a parent completely and fails after
    /// the parent whose successors made the count reach `limit`; its count then. The new states of a chunk are
    /// numbered in (parent, canonical index) order, so those m holds after parent p are the ones up to the largest
    /// target of m's parents up to p.
    u32 failure_count(Wave& w, u32 m, const ChunkStart& at, u32 done, u64 limit, ChunkTail& t)
    {
        if (!t.loaded)
        {
            download(w.off.data() + at.b, u64{done} + 1, t.off, m_s);
            download(w.targets.data() + at.edges, w.edges - at.edges, t.targets, m_s);
            if (m_multi)
            {
                download(w.inst.data() + at.b, done, t.parent_inst, m_s);
                download(w.inst.data() + at.count, w.count - at.count, t.new_inst, m_s);
            }
            sync(m_s);
            t.loaded = true;
        }
        u64 top = 0;   // 1 + the largest target so far
        u64 seen = 0;  // new states (ids at.count + j) below `top`
        u64 held = 0;  // those of member m
        u64 count = at.icount[m];
        for (u32 p = 0; p < done; ++p)
        {
            if (m_multi && t.parent_inst[p] != m)
                continue;
            for (u64 e = t.off[p] - at.edges; e < t.off[u64{p} + 1] - at.edges; ++e)
                top = std::max<u64>(top, u64{t.targets[e]} + 1);
            for (; at.count + seen < top && at.count + seen < w.count; ++seen)
                if (!m_multi || t.new_inst[seen] == m)
                    ++held;
            count = at.icount[m] + held;
            if (count >= limit)
                return static_cast<u32>(count);
        }
        return w.icount[m];  // (not reached: the member's count after the chunk is at least the limit)
    }

    [[nodiscard]] bool unsatisfiable(u32 id) const
    {
        return m_multi ? m_table->instance(id).goal_unsatisfiable : m_tasks[id]->compiled().goal.unsatisfiable;
    }

    /// A member that is never expanded (statically false goal): no state, status Unsolvable.
    void admit_placeholder(Wave& w)
    {
        w.icount.push_back(0);
        w.last.push_back(0);
        w.dead.push_back(1);
        w.status.push_back(StateSpaceStatus::Unsolvable);
        w.t0.push_back(Clock::now());
        w.host_live[w.k] = ssk::k_no_instance;
        if (m_multi)
        {
            const u32 zero = 0;
            check(cudaMemcpyAsync(ptr<u32>(w.live) + w.k, &w.host_live[w.k], sizeof(u32), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (live)");
            check(cudaMemcpyAsync(counts_dev() + w.k, &zero, sizeof(u32), cudaMemcpyHostToDevice, m_s),
                  "cudaMemcpyAsync (counts)");
            sync(m_s);
        }
        ++w.k;
    }

    void release_scratch()
    {
        for (Scratch* x : {&m_parent, &m_cand, &m_rank, &m_scan, &m_sort, &m_ids, &m_order, &m_pos, &m_order_temp, &m_views,
                           &m_counts_sc, &m_seg, &m_seg_scan, &m_sch_sc, &m_bind_sc})
            x->reset();
    }

    /// Parents per chunk at most: their views fit the budget (as cuda::DeviceBrfs).
    [[nodiscard]] u32 chunk_capacity() const
    {
        int l2 = 0;
        check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, m_ctx->device()), "cudaDeviceGetAttribute");
        u64 vw = 0;
        if (m_multi)
            vw = m_plan.view_words;
        else
        {
            const Task& t = *m_gen->task();
            const u64 dw = m_gen->device_axioms() ? std::max<u32>(1, bits::words_for(t.atoms().max_derived_slots())) : 0;
            vw = m_gen->view_words() + dw;
        }
        const u64 per = std::max<u64>(8, vw * 8);
        const u64 floor = std::min<u64>(65536, (u64{256} << 20) / per);
        const u64 cap = m_o.view_bytes ? std::max<u64>(1, m_o.view_bytes / per)
                                       : std::max<u64>({8192, floor, std::max<u64>(static_cast<u64>(l2) / 2, 1) / per});
        return static_cast<u32>(std::min<u64>(cap, 0x7FFFFFFFull));
    }

    // ------------------------------------------------------------------------------------------ a chunk
    /// Expands parents [b, b + ns) (fewer when their successors exceed the candidate budget); returns the parents
    /// expanded.
    u32 expand(Wave& w, u64 b, u32 ns)
    {
        const u64 budget = std::max<u64>(u64{1} << 20, k_candidate_bytes / (u64{w.W} * 8 + 16 + 4 * u64{m_L}));
        for (;;)  // one chunk (redone after a width growth, or cut)
        {
            u64 M = m_multi ? count_multi(w, b, ns) : count_single(w, b, ns);
            if (M > budget && ns > 1)
            {
                ns = std::max<u32>(1, static_cast<u32>(static_cast<double>(ns) * static_cast<double>(budget) / static_cast<double>(M)));
                continue;
            }
            if (w.edges + M >= (u64{1} << 32))
                throw std::length_error("mymyr: device state space: more than 2^32 - 1 transitions in a wave (state spaces "
                                        "are limited to 2^32 - 1 transitions; lower wave_states for tables)");
            const u32* seg = m_multi ? ptr<u32>(m_seg, m_ctx, 1, m_s) : m_gen->seg_offsets();
            // the transitions' arrays at the chunk's first edge; the labels go there directly
            w.targets.reserve(m_ctx, w.edges + M, w.edges, m_s);
            const bool labels = m_o.space.labels;
            const bool dev_costs = m_cost_mode == CostMode::Device;
            if (labels)
            {
                w.schemas.reserve(m_ctx, w.edges + M, w.edges, m_s);
                w.bindings.reserve(m_ctx, (w.edges + M) * m_L, w.edges * m_L, m_s);
            }
            if (dev_costs)
                w.costs.reserve(m_ctx, w.edges + M, w.edges, m_s);
            u32* parent = ptr<u32>(m_parent, m_ctx, M, m_s);
            u64* cand = ptr<u64>(m_cand, m_ctx, M * w.W, m_s);
            u32* sch = labels ? w.schemas.data() + w.edges : dev_costs ? ptr<u32>(m_sch_sc, m_ctx, M, m_s) : nullptr;
            u32* bnd = m_L == 0 ? nullptr
                       : labels ? w.bindings.data() + w.edges * m_L
                       : dev_costs ? ptr<u32>(m_bind_sc, m_ctx, M * m_L, m_s)
                                   : nullptr;
            const bool sorts = m_multi ? m_plan.w[1][0].sorts : m_gen->sorts(false);
            u32* sort = !bnd && sorts && m_L ? ptr<u32>(m_sort, m_ctx, M * m_L, m_s) : nullptr;
            lifted::Labels lab;
            lab.binding = bnd;
            lab.schema = sch;
            lab.parent = parent;
            lab.capacity = M;
            lab.label_width = m_L;
            lab.parent_base = static_cast<u32>(b);
            lab.scratch = sort;
            lab.scratch_rows = sort ? M : 0;
            lab.scratch_indexed = sort ? 1 : 0;
            lab.error = ptr<u32>(m_dctl);
            if (M)
            {
                if (m_multi)
                    write_multi(w, b, ns, lab, cand);
                else if (!write_single(w, b, ns, lab, cand))
                    continue;  // widened: redo the chunk at the new width
            }
            dedup(w, b, ns, M, seg, parent, cand, sch, bnd);
            return ns;
        }
    }

    // single instance (ChunkGenerator) ---------------------------------------------------------
    u64 count_single(Wave& w, u64 b, u32 ns)
    {
        const u64* rows_b = w.rows.data() + b * w.W;
        const u64* host_rows = nullptr;
        if (m_gen->needs_host(false))
        {
            const u64 bytes = u64{ns} * w.W * sizeof(u64);
            if (m_host_rows.size() < bytes)
                m_host_rows = PinnedBuffer(std::max<u64>(bytes, m_host_rows.size() * 2));
            check(cudaMemcpyAsync(m_host_rows.data(), rows_b, bytes, cudaMemcpyDeviceToHost, m_s), "cudaMemcpyAsync (parents)");
            sync(m_s);
            host_rows = static_cast<const u64*>(m_host_rows.data());
        }
        const ChunkInput in{rows_b, w.W, w.W, ns, host_rows, w.W, static_cast<u32>(b)};
        m_gen->begin(in, false, true);
        m_gen->views();
        const lifted::Parents& p = m_gen->parents();
        if (m_gen->task()->compiled().goal.uses_derived && !m_gen->device_axioms())
        {
            std::vector<u64> values;
            if (!host_rows)
            {
                values.resize(u64{ns} * w.W);
                check(cudaMemcpyAsync(values.data(), rows_b, values.size() * 8, cudaMemcpyDeviceToHost, m_s), "goal rows");
                sync(m_s);
                host_rows = values.data();
            }
            std::vector<u8> flags(ns);
            for (u32 i = 0; i < ns; ++i)
            {
                const State state = numeric::decode(*m_gen->task(), host_rows + u64{i} * w.W, w.W);
                flags[i] = m_gen->task()->is_goal(state.view());
            }
            check(cudaMemcpyAsync(w.goal.data() + b, flags.data(), ns, cudaMemcpyHostToDevice, m_s), "goal flags");
            sync(m_s);
        }
        else if (m_gen->task()->numeric_slots())
            m_gen->goal_flags(rows_b, w.W, w.W, ns, nullptr, ns, w.goal.data() + b);
        else if (p.derived)
            check(lifted::launch_goal_rows_derived(m_gen->view(), rows_b, w.W, ns, p.derived, p.derived_words, w.goal.data() + b, m_s),
                  "launch_goal_rows_derived");
        else
            check(lifted::launch_goal_rows(m_gen->view(), rows_b, w.W, ns, nullptr, w.goal.data() + b, m_s), "launch_goal_rows");
        w.goals = std::max(w.goals, b + ns);
        m_gen->count();
        return read_u32(m_gen->seg_offsets() + u64{ns} * m_S);
    }

    /// Writes the chunk's rows; false if the rows were widened (lazy slots): redo the chunk.
    bool write_single(Wave& w, u64 b, u32 ns, const lifted::Labels& lab, u64* cand)
    {
        (void)b;
        (void)ns;
        m_gen->write(lab, cand, w.W);
        const Task& task = *m_gen->task();
        for (;;)
        {
            if (!m_gen->resolve_missing())
                return true;
            if (bucket(task.words() + task.numeric_slots()) > w.W)
            {
                widen(w, bucket(task.words() + task.numeric_slots()));
                return false;
            }
            m_gen->write(lab, cand, w.W);
        }
    }

    /// Re-lays the rows out at `nw` words (lazy slots outgrew them) and rehashes the table.
    void widen(Wave& w, u32 nw)
    {
        Grow<u64> grown;
        grown.reserve(m_ctx, std::max<u64>(w.rows.cap / w.W, w.count + 1) * nw, 0, m_s);
        const u32 slots = m_gen->task()->numeric_slots();
        if (slots)
        {
            auto view = m_gen->view();
            view.numeric.storage = 0;
            check(numeric::launch_convert(view, w.rows.data(), w.W, w.W - slots, grown.data(), nw, nw - slots,
                                           w.count, true, m_s), "numeric relayout");
        }
        else
            check(state_set::launch_relayout(w.rows.data(), w.W, grown.data(), nw, w.count, m_s), "launch_relayout");
        w.rows = std::move(grown);
        w.W = nw;
        check(cudaMemsetAsync(w.table.data(), 0, w.slots * sizeof(u64), m_s), "cudaMemsetAsync (state table)");
        check(ssk::launch_rehash(w.tab(), w.store(), 0, w.count, m_s), "launch_rehash");
        ++m_st.widenings;
    }

    // multi-instance kernels ---------------------------------------------------------------------
    [[nodiscard]] lifted::Multi multi_of(const u32* ids, const u32* order) const
    {
        lifted::Multi m = m_plan.multi(*m_dt, false, true, reinterpret_cast<const i32*>(ids));
        m.order = order;
        return m;
    }

    u64 count_multi(Wave& w, u64 b, u32 ns)
    {
        u32* ids = ptr<u32>(m_ids, m_ctx, ns, m_s);
        check(ssk::launch_chunk_ids(w.inst.data() + b, ptr<u32>(w.live), ns, ids, m_s), "launch_chunk_ids");
        u32* ord = ptr<u32>(m_order, m_ctx, ns, m_s);
        u32* pos = ptr<u32>(m_pos, m_ctx, ns, m_s);
        const u64 tb = lifted::order_temp_bytes(ns);
        check(lifted::launch_order(ids, m_plan.instances, m_plan.order_keys(), ns,
                                   m_order_temp.ensure(m_ctx, std::max<u64>(tb, 1), m_s), tb, ord, pos, 1, m_s),
              "launch_order");
        const lifted::Parents p{w.rows.data() + b * w.W, w.W, w.W, ns, nullptr, 0};
        const lifted::Views v{ptr<u64>(m_views, m_ctx, u64{ns} * m_plan.view_words, m_s), m_plan.view_words};
        lifted::Multi m = multi_of(ids, ord);
        check(lifted::launch_view_multi(m, p, v, m_s), "launch_view_multi");
        const u64 nseg = u64{ns} * m_S + 1;
        if (nseg > 0x7FFFFFFFull)
            throw std::length_error("mymyr: device state space: more than 2^31 (state, schema) segments in one chunk");
        u32* cnt = ptr<u32>(m_counts_sc, m_ctx, nseg, m_s);
        check(cudaMemsetAsync(cnt, 0, nseg * sizeof(u32), m_s), "cudaMemsetAsync");
        const lifted::SchemaSet fixed = m_plan.set(false, true, false);
        const lifted::SchemaSet fc = m_plan.set(false, true, true);
        const auto groups = static_cast<u32>(m_plan.ows.size());
        m_fan.fork(m_s, groups);
        for (u32 gi = 0; gi < groups; ++gi)
        {
            m.ow = m_plan.ows[gi];
            if (fixed.count)
                check(lifted::launch_count_multi(m, p, v, fixed, cnt, m_fan.lane(m_s, gi)), "launch_count_multi");
            if (fc.count && m.ow <= lifted::k_max_fc_ow)
                check(lifted::launch_count_multi(m, p, v, fc, cnt, m_fan.lane(m_s, gi)), "launch_count_multi (fc)");
        }
        m_fan.join(m_s, groups);
        u32* seg = ptr<u32>(m_seg, m_ctx, nseg, m_s);
        const u64 stb = lifted::scan_temp_bytes(nseg);
        check(lifted::launch_scan(cnt, seg, nseg, m_seg_scan.ensure(m_ctx, std::max<u64>(stb, 1), m_s), stb, m_s), "launch_scan");
        check(lifted::launch_goal_rows_multi(m_dt->device_views(), m_plan.instances, ids, p.data, w.W, ns, w.goal.data() + b, m_s),
              "launch_goal_rows_multi");
        w.goals = std::max(w.goals, b + ns);
        return read_u32(seg + u64{ns} * m_S);
    }

    void write_multi(Wave& w, u64 b, u32 ns, const lifted::Labels& lab, u64* cand)
    {
        const u32* ids = ptr<u32>(m_ids, m_ctx, ns, m_s);
        const u32* ord = ptr<u32>(m_order, m_ctx, ns, m_s);
        const lifted::Parents p{w.rows.data() + b * w.W, w.W, w.W, ns, nullptr, 0};
        const lifted::Views v{ptr<u64>(m_views, m_ctx, u64{ns} * m_plan.view_words, m_s), m_plan.view_words};
        const u32* seg = ptr<u32>(m_seg, m_ctx, 1, m_s);
        lifted::SuccessorRows rows;
        rows.words = cand;
        rows.out_words = w.W;
        const lifted::SchemaSet fixed = m_plan.set(false, true, false);
        const lifted::SchemaSet fc = m_plan.set(false, true, true);
        lifted::Multi m = multi_of(ids, ord);
        const auto& launches = m_plan.launches(BucketLaunch::PerBucket);
        const auto groups = static_cast<u32>(launches.size());
        m_fan.fork(m_s, groups);
        for (u32 gi = 0; gi < groups; ++gi)
        {
            const detail::TableLaunch::Launch& l = launches[gi];
            m.ow = l.ow;
            m.words_lo = l.lo;
            m.words_hi = l.hi;
            if (fixed.count)
                check(lifted::launch_write_multi(m, p, v, fixed, seg, lab, rows, l.wb, m_fan.lane(m_s, gi)), "launch_write_multi");
            if (fc.count && l.ow <= lifted::k_max_fc_ow)
                check(lifted::launch_write_multi(m, p, v, fc, seg, lab, rows, l.wb, m_fan.lane(m_s, gi)),
                      "launch_write_multi (fc)");
        }
        m_fan.join(m_s, groups);
    }

    // dedup ---------------------------------------------------------------------------------------
    void dedup(Wave& w, u64 b, u32 ns, u64 M, const u32* seg, const u32* parent, const u64* cand, const u32* sch,
               const u32* bnd)
    {
        u32* dctl = ptr<u32>(m_dctl);
        u64 fresh = 0;
        u32* result = w.targets.data() + w.edges;
        if (M)
        {
            ensure_table(w, M);
            u32* rank = ptr<u32>(m_rank, m_ctx, M + 1, m_s);
            const u64 tb = ssk::rank_temp_bytes(M);
            void* scan = m_scan.ensure(m_ctx, std::max<u64>(tb, 1), m_s);
            check(ssk::launch_insert(w.tab(), w.store(), cand, parent, M, result, m_s), "launch_insert");
            check(ssk::launch_rank(w.tab(), result, M, rank, scan, tb, m_s), "launch_rank");
            if (m_multi)
                check(ssk::launch_owner_counts(w.tab(), result, parent, w.inst.data(), M, counts_dev(), m_s), "launch_owner_counts");
            check(cudaMemcpyAsync(dctl + k_ctl_fresh, rank + M, sizeof(u32), cudaMemcpyDeviceToDevice, m_s), "cudaMemcpyAsync");
            read_control(m_multi ? w.k : 0);
            if (m_hctl[k_ctl_sort])
                throw std::logic_error("mymyr: device state space: a sorted segment found no scratch (internal error)");
            check_costs_flag();
            fresh = m_hctl[k_ctl_fresh];
            if (w.count + fresh > state_set::k_max_states)
                throw std::length_error("mymyr: device state space: more than 2^31 - 2 states");
            reserve_states(w, w.count + fresh);
            ssk::Compact cm;
            cm.result = result;
            cm.rank = rank;
            cm.cand = cand;
            cm.parent = parent;
            cm.words = w.W;
            cm.base = w.count;
            cm.rows = w.rows.data();
            cm.inst = w.inst.data();
            cm.depth = w.depth.data();
            check(ssk::launch_compact(w.tab(), cm, M, m_s), "launch_compact");
            check(ssk::launch_resolve(w.tab(), result, M, m_s), "launch_resolve");
            if (m_cost_mode == CostMode::Device)
            {
                check(ssk::launch_costs(m_programs.view, w.first, sch, bnd, m_L, parent, w.inst.data(), w.depth.data(), M,
                                        w.costs.data() + w.edges, dctl + k_ctl_cost, m_s),
                      "launch_costs");
            }
        }
        check(ssk::launch_forward_offsets(seg, m_S, ns, w.edges, w.off.data() + b, m_s), "launch_forward_offsets");
        w.offsets = b + ns + 1;
        if (m_multi)
        {
            if (M)
                for (u32 m = 0; m < w.k; ++m)
                {
                    if (w.dead[m])
                        continue;  // (a failed member keeps its count at the failure)
                    const u32 now = m_hctl[k_ctl_counts + m];
                    if (now != w.icount[m])
                        w.last[m] = w.count + fresh - 1;
                    w.icount[m] = now;
                }
        }
        else
        {
            w.icount[0] = static_cast<u32>(w.count + fresh);
            w.last[0] = w.count + fresh - 1;
        }
        w.count += fresh;
        w.edges += M;
    }

    // control ------------------------------------------------------------------------------------
    static constexpr u32 k_ctl_sort = 0, k_ctl_cost = 1, k_ctl_fresh = 2, k_ctl_counts = 8;
    [[nodiscard]] u32* counts_dev() const { return ptr<u32>(m_dctl) + k_ctl_counts; }

    /// Copies the control block (with `members` counts) to the host; waits for the stream (a pageable copy).
    void read_control(u32 members)
    {
        check(cudaMemcpyAsync(m_hctl.data(), m_dctl.data(), (u64{k_ctl_counts} + members) * sizeof(u32), cudaMemcpyDeviceToHost, m_s),
              "cudaMemcpyAsync (control)");
        sync(m_s);
    }

    /// One device u32 (waits for the stream).
    [[nodiscard]] u32 read_u32(const u32* p)
    {
        u32 v = 0;
        check(cudaMemcpyAsync(&v, p, sizeof(u32), cudaMemcpyDeviceToHost, m_s), "cudaMemcpyAsync");
        sync(m_s);
        return v;
    }

    void check_costs_flag() const
    {
        if (m_cost_mode == CostMode::Device && m_hctl[k_ctl_cost])
            throw std::domain_error("mymyr: device state space: an action's cost is undefined (a static function value is "
                                    "missing); the CPU successor generator drops such actions, the device does not");
    }

    // ------------------------------------------------------------------------------------------ post-processing
    void host_costs(Wave& w)
    {
        const auto t0 = Clock::now();
        const u64 N = w.count, E = w.edges, W = w.W;
        std::vector<u64> rows, off;
        std::vector<u32> inst, depth;
        download(w.rows.data(), N * W, rows, m_s);
        download(w.off.data(), N + 1, off, m_s);
        download(w.inst.data(), N, inst, m_s);
        download(w.depth.data(), N, depth, m_s);
        sync(m_s);
        std::vector<f64> costs(E, 0.0);
        std::atomic<u64> next{0};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::exception_ptr error;
        auto worker = [&]
        {
            try
            {
                std::vector<u64> tmp;
                for (;;)
                {
                    const u64 a = next.fetch_add(256, std::memory_order_relaxed);
                    if (a >= N || failed.load(std::memory_order_relaxed))
                        return;
                    for (u64 g = a; g < std::min<u64>(N, a + 256); ++g)
                    {
                        const u32 m = inst[g];
                        if (w.dead[m] || off[g + 1] == off[g])
                            continue;
                        const Task& task = *m_tasks[w.first + m];
                        const heuristics::ActionCosts& ac = m_costs[w.first + m];
                        Successors& succ = task.workspace().successors();
                        const u64* row = rows.data() + g * W;
                        const State state = numeric::decode(task, row, static_cast<u32>(W));
                        const StateView rec = state.view();
                        succ.prepare(rec);
                        const bool metric = ac.kind() == heuristics::ActionCosts::Kind::StateMetric;
                        const f64 gv = metric ? ac.initial(rec) : static_cast<f64>(depth[g]);
                        u64 e = off[g];
                        succ.generate<false>(
                            [&](u32, const ObjectId*, const Delta& d)
                            {
                                if (e < off[g + 1])
                                    costs[e] = ac.unit() ? 1.0 : ac.transition(gv, d);
                                ++e;
                                return true;
                            },
                            false, true);
                        if (e != off[g + 1])
                            throw std::logic_error("mymyr: device state space: the CPU regenerates " + std::to_string(e - off[g]) +
                                                   " transitions of a state the device expanded into " +
                                                   std::to_string(off[g + 1] - off[g]) + " (internal error)");
                    }
                }
            }
            catch (...)
            {
                std::lock_guard lock(error_mutex);
                if (!error)
                    error = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        };
        std::vector<std::thread> pool;
        const u32 T = static_cast<u32>(std::min<u64>(m_threads, std::max<u64>(1, N / 256)));
        for (u32 t = 1; t < T; ++t)
            pool.emplace_back(worker);
        worker();
        for (auto& th : pool)
            th.join();
        if (error)
            std::rethrow_exception(error);
        w.costs.reserve(m_ctx, E, 0, m_s);
        if (E)
            check(cudaMemcpyAsync(w.costs.data(), costs.data(), E * sizeof(f64), cudaMemcpyHostToDevice, m_s), "cudaMemcpyAsync (costs)");
        sync(m_s);
        m_st.host_cost_ms += ms_since(t0);
    }

    void post(Wave& w)
    {
        const auto tp = Clock::now();
        const u64 N = w.count, E = w.edges;
        const u32 k = w.k;
        const cudaStream_t s = m_s;
        bool any = false;
        for (u32 m = 0; m < k; ++m)
            any = any || !w.dead[m];
        if (!any)
        {
            finish_failed(w);
            return;
        }
        // layers per member
        std::vector<u32> maxd;
        {
            DeviceBuffer d = alloc(m_ctx, u64{k} * sizeof(u32), s);
            check(cudaMemsetAsync(d.data(), 0, u64{k} * sizeof(u32), s), "cudaMemsetAsync");
            check(ssk::launch_max_depth(w.depth.data(), k > 1 ? w.inst.data() : nullptr, N, ptr<u32>(d), s), "launch_max_depth");
            download(ptr<u32>(d), k, maxd, s);
            sync(s);
        }
        const bool has_costs = m_cost_mode != CostMode::Unit;
        if (m_cost_mode == CostMode::Host)
            host_costs(w);
        w.depth.release();
        const bool labels = m_o.space.labels;
        // instance-major order (a stable partition by member): perm[p] = the state at union position p
        std::vector<u64> P(k + 1, 0), Q(k + 1, 0);
        DeviceBuffer perm, sinst;
        if (k > 1)
        {
            {
                DeviceBuffer k1 = alloc(m_ctx, N * 4, s), v0 = alloc(m_ctx, N * 4, s), v1 = alloc(m_ctx, N * 4, s);
                DeviceBuffer k0 = alloc(m_ctx, N * 4, s);
                check(cudaMemcpyAsync(k0.data(), w.inst.data(), N * 4, cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync");
                check(ssk::launch_iota(ptr<u32>(v0), N, s), "launch_iota");
                const u64 tb = ssk::sort_temp_bytes(N);
                DeviceBuffer temp = alloc(m_ctx, tb, s);
                int sel = 0;
                check(ssk::launch_sort_pairs(ptr<u32>(k0), ptr<u32>(k1), ptr<u32>(v0), ptr<u32>(v1), N, bits_for(k), temp.data(), tb,
                                             &sel, s),
                      "launch_sort_pairs (instances)");
                perm = std::move(sel ? v1 : v0);
                sinst = std::move(sel ? k1 : k0);
            }
            {
                DeviceBuffer cnt = alloc(m_ctx, u64{k} * 4, s);
                check(cudaMemsetAsync(cnt.data(), 0, u64{k} * 4, s), "cudaMemsetAsync");
                check(ssk::launch_histogram_grouped(w.inst.data(), N, ptr<u32>(cnt), s), "launch_histogram_grouped");
                std::vector<u32> n;
                download(ptr<u32>(cnt), k, n, s);
                sync(s);
                for (u32 m = 0; m < k; ++m)
                    P[m + 1] = P[m] + n[m];
            }
            w.inst.release();
            DeviceBuffer pos = alloc(m_ctx, N * 4, s);
            check(ssk::launch_invert(ptr<u32>(perm), N, ptr<u32>(pos), s), "launch_invert");
            DeviceBuffer noff = alloc(m_ctx, (N + 1) * 8, s);
            {
                DeviceBuffer deg = alloc(m_ctx, N * 8, s);
                check(ssk::launch_degrees(w.off.data(), ptr<u32>(perm), N, ptr<u64>(deg), s), "launch_degrees");
                const u64 tb = ssk::scan_temp_bytes(N);
                DeviceBuffer temp = alloc(m_ctx, tb, s);
                check(ssk::launch_scan_u64(ptr<u64>(deg), ptr<u64>(noff), N, temp.data(), tb, s), "launch_scan_u64");
            }
            {
                DeviceBuffer emap = alloc(m_ctx, E * 4, s);
                check(ssk::launch_edge_map(w.off.data(), ptr<u64>(noff), ptr<u32>(pos), N, ptr<u32>(emap), s), "launch_edge_map");
                auto move_u32 = [&](Grow<u32>& a, u32 width, const u32* map)
                {
                    Grow<u32> b;
                    b.reserve(m_ctx, E * width, 0, s);
                    check(ssk::launch_scatter_u32(a.data(), ptr<u32>(emap), map, E, width, b.data(), s), "launch_scatter_u32");
                    a = std::move(b);
                };
                move_u32(w.targets, 1, ptr<u32>(pos));
                if (labels)
                {
                    move_u32(w.schemas, 1, nullptr);
                    if (m_L)
                        move_u32(w.bindings, m_L, nullptr);
                }
                if (has_costs)
                {
                    Grow<f64> b;
                    b.reserve(m_ctx, E, 0, s);
                    check(ssk::launch_scatter_f64(w.costs.data(), ptr<u32>(emap), E, b.data(), s), "launch_scatter_f64");
                    w.costs = std::move(b);
                }
            }
            {
                Grow<u8> g;
                g.reserve(m_ctx, N, 0, s);
                check(ssk::launch_gather_u8(w.goal.data(), ptr<u32>(perm), N, g.data(), s), "launch_gather_u8");
                w.goal = std::move(g);
            }
            pos.reset();
            {
                Grow<u64> o;
                o.buf = std::move(noff);
                o.cap = N + 1;
                w.off = std::move(o);
            }
            DeviceBuffer dP = upload(m_ctx, P, s), dq = alloc(m_ctx, (u64{k} + 1) * 8, s);
            check(ssk::launch_pick_u64(w.off.data(), ptr<u64>(dP), k, ptr<u64>(dq), s), "launch_pick_u64");
            std::vector<u64> q;
            download(ptr<u64>(dq), u64{k} + 1, q, s);  // Q[m] = the first edge of member m
            sync(s);
            Q = std::move(q);
        }
        else
        {
            P[1] = N;
            Q[1] = E;
            w.inst.release();
        }
        // the wave's storage, filled as its arrays become final: with host output, the members' host arrays are sized
        // in the background meanwhile, and the final arrays are downloaded while the device works on the others
        std::vector<u32> words(k);
        std::vector<u64> row_base(k + 1, 0);
        for (u32 m = 0; m < k; ++m)
        {
            words[m] = std::max<u32>(1, m_tasks[w.first + m]->words()) + m_tasks[w.first + m]->numeric_words();
            row_base[m + 1] = row_base[m] + (P[m + 1] - P[m]) * words[m];
        }
        std::vector<DeviceStateSpace::Part> parts(k);
        for (u32 m = 0; m < k; ++m)
            parts[m] = {P[m], Q[m], row_base[m], P[m] + m};
        auto storage = std::make_shared<DeviceStateSpace::Storage>();
        storage->ctx = m_ctx;
        storage->s = s;
        const bool host_out = m_o.output != StateSpaceOutput::Device;
        const u32 label_width = labels ? m_L : 0;
        constexpr u32 early_a = 1u << a_rows | 1u << a_foff | 1u << a_schemas | 1u << a_bindings | 1u << a_costs | 1u << a_goal;
        constexpr u32 early_b = 1u << a_targets;
        HostOutput out(m_ctx, m_threads);
        if (host_out)
        {
            std::vector<Shape> shapes(k);
            for (u32 m = 0; m < k; ++m)
                if (!w.dead[m])
                    shapes[m] = {P[m + 1] - P[m], Q[m + 1] - Q[m], words[m], label_width, labels, has_costs};
            // the early arrays one by one (their downloads overlap the sizing of the next), the others last
            std::vector<u32> phases;
            for (u32 a = 0; a < 32; ++a)
                if ((early_a | early_b) >> a & 1)
                    phases.push_back(1u << a);
            out.presize(std::move(shapes), std::move(phases));
        }
        const u32* spos = k > 1 ? ptr<u32>(sinst) : nullptr;  // the member of every union position
        DeviceBuffer dP = upload(m_ctx, P, s), dQ = upload(m_ctx, Q, s), dwords = upload(m_ctx, words, s),
                     dbase = upload(m_ctx, row_base, s);
        const ssk::Layout lay{ptr<u64>(dP), ptr<u64>(dQ), ptr<u32>(dwords), ptr<u64>(dbase), spos};
        // final now: the rows and the forward offsets (in local ids), the labels, the costs and the goal flags
        storage->rows = alloc(m_ctx, row_base[k] * 8, s);
        if (!m_multi && m_tasks[w.first]->numeric_slots())
            check(numeric::launch_convert(m_gen->view(), w.rows.data(), w.W, w.W - m_tasks[w.first]->numeric_slots(),
                                           ptr<u64>(storage->rows), words[0], words[0] - m_tasks[w.first]->numeric_words(),
                                           N, false, s), "numeric state-space output");
        else
            check(ssk::launch_local_rows(w.rows.data(), w.W, k > 1 ? ptr<u32>(perm) : nullptr, lay, N, ptr<u64>(storage->rows), s),
                  "launch_local_rows");
        w.rows.release();
        perm.reset();
        storage->foff = alloc(m_ctx, (N + k) * 8, s);
        check(ssk::launch_local_offsets(w.off.data(), lay, N, ptr<u64>(storage->foff), s), "launch_local_offsets");
        check(ssk::launch_local_ends(w.off.data(), lay, k, ptr<u64>(storage->foff), s), "launch_local_ends");
        if (labels)
        {
            storage->schemas = std::move(w.schemas.buf);
            storage->bindings = std::move(w.bindings.buf);
        }
        if (has_costs)
            storage->costs = std::move(w.costs.buf);
        storage->goal = std::move(w.goal.buf);
        const u8* goal = ptr<u8>(storage->goal);
        const f64* costs = has_costs ? ptr<f64>(storage->costs) : nullptr;
        if (host_out)
            out.early(early_a, storage, parts, label_width, s);
        // reverse CSR: (target, edge) pairs sorted stably by target; the targets in local ids (final) after their copy
        // and the in-degrees
        DeviceBuffer bedges, bsrc, boff;
        {
            DeviceBuffer k0 = alloc(m_ctx, E * 4, s);
            if (E)
                check(cudaMemcpyAsync(k0.data(), w.targets.data(), E * 4, cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync");
            {
                DeviceBuffer indeg = alloc(m_ctx, N * 4, s);
                check(cudaMemsetAsync(indeg.data(), 0, N * 4, s), "cudaMemsetAsync");
                check(ssk::launch_histogram(w.targets.data(), E, ptr<u32>(indeg), s), "launch_histogram");
                boff = alloc(m_ctx, (N + 1) * 8, s);
                const u64 tb = ssk::scan_temp_bytes(N);
                DeviceBuffer temp = alloc(m_ctx, tb, s);
                check(ssk::launch_scan_u32(ptr<u32>(indeg), ptr<u64>(boff), N, temp.data(), tb, s), "launch_scan_u32");
            }
            if (k > 1)
                check(ssk::launch_local_states(w.targets.data(), lay, E, s), "launch_local_states");
            storage->targets = std::move(w.targets.buf);
            DeviceBuffer k1 = alloc(m_ctx, E * 4, s), v0 = alloc(m_ctx, E * 4, s), v1 = alloc(m_ctx, E * 4, s);
            check(ssk::launch_iota(ptr<u32>(v0), E, s), "launch_iota");
            int sel = 0;
            if (E)
            {
                const u64 tb = ssk::sort_temp_bytes(E);
                DeviceBuffer temp = alloc(m_ctx, tb, s);
                check(ssk::launch_sort_pairs(ptr<u32>(k0), ptr<u32>(k1), ptr<u32>(v0), ptr<u32>(v1), E, bits_for(N), temp.data(),
                                             tb, &sel, s),
                      "launch_sort_pairs (reverse)");
            }
            bedges = std::move(sel ? v1 : v0);
        }
        storage->boff = alloc(m_ctx, (N + k) * 8, s);
        check(ssk::launch_local_offsets(ptr<u64>(boff), lay, N, ptr<u64>(storage->boff), s), "launch_local_offsets");
        check(ssk::launch_local_ends(ptr<u64>(boff), lay, k, ptr<u64>(storage->boff), s), "launch_local_ends");
        if (host_out)
            out.early(early_b, storage, parts, label_width, s);
        bsrc = alloc(m_ctx, E * 4, s);
        check(ssk::launch_edge_sources(w.off.data(), N, ptr<u32>(bedges), E, ptr<u32>(bsrc), s), "launch_edge_sources");
        w.off.release();
        // unit goal distances
        DeviceBuffer unit = alloc(m_ctx, N * 4, s);
        DeviceBuffer fa = alloc(m_ctx, N * 4, s), fb = alloc(m_ctx, N * 4, s);
        DeviceBuffer frontier_ctl = alloc(m_ctx, 2 * sizeof(u32), s);
        u32* dctl = ptr<u32>(frontier_ctl);
        u32 c[1] = {0};
        {
            check(cudaMemsetAsync(dctl, 0, 2 * sizeof(u32), s), "cudaMemsetAsync");
            check(ssk::launch_bfs_init(goal, N, ptr<i32>(unit), ptr<u32>(fa), dctl, s), "launch_bfs_init");
            check(cudaMemcpyAsync(c, dctl, 4, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            sync(s);
            u64 nf = c[0];
            i32 d = 0;
            while (nf)
            {
                ++d;
                check(cudaMemsetAsync(dctl + 1, 0, 4, s), "cudaMemsetAsync");
                check(ssk::launch_bfs_step(ptr<u64>(boff), ptr<u32>(bsrc), ptr<u32>(fa), nf, d, ptr<i32>(unit), ptr<u32>(fb), dctl + 1, s),
                      "launch_bfs_step");
                check(cudaMemcpyAsync(c, dctl + 1, 4, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
                sync(s);
                nf = c[0];
                std::swap(fa, fb);
            }
        }
        // cost goal distances
        DeviceBuffer cost = alloc(m_ctx, N * 8, s);
        std::vector<u32> not_unit(k, 0), bad(k, 0);
        if (has_costs)
        {
            DeviceBuffer flags = alloc(m_ctx, u64{k} * 8, s);
            check(cudaMemsetAsync(flags.data(), 0, u64{k} * 8, s), "cudaMemsetAsync");
            check(ssk::launch_unit_costs(costs, ptr<u64>(dQ), k, E, ptr<u32>(flags), s), "launch_unit_costs");
            check(ssk::launch_check_costs(costs, ptr<u64>(dQ), k, E, ptr<u32>(flags) + k, s), "launch_check_costs");
            std::vector<u32> f;
            download(ptr<u32>(flags), 2 * u64{k}, f, s);
            sync(s);
            std::copy_n(f.begin(), k, not_unit.begin());
            std::copy_n(f.begin() + k, k, bad.begin());
            DeviceBuffer queued = alloc(m_ctx, N * 4, s);
            check(cudaMemsetAsync(queued.data(), 0, N * 4, s), "cudaMemsetAsync");
            check(cudaMemsetAsync(dctl, 0, 2 * sizeof(u32), s), "cudaMemsetAsync");
            check(ssk::launch_sssp_init(goal, N, ptr<u64>(cost), ptr<u32>(fa), dctl, s), "launch_sssp_init");
            check(cudaMemcpyAsync(c, dctl, 4, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            sync(s);
            u64 nf = c[0];
            while (nf)
            {
                check(cudaMemsetAsync(dctl + 1, 0, 4, s), "cudaMemsetAsync");
                check(ssk::launch_sssp_step(ptr<u64>(boff), ptr<u32>(bsrc), ptr<u32>(bedges), costs, ptr<u32>(fa), nf, ptr<u64>(cost),
                                            ptr<u32>(queued), ptr<u32>(fb), dctl + 1, s),
                      "launch_sssp_step");
                check(cudaMemcpyAsync(c, dctl + 1, 4, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
                sync(s);
                nf = c[0];
                check(ssk::launch_sssp_clear(ptr<u32>(fb), nf, ptr<u32>(queued), s), "launch_sssp_clear");
                std::swap(fa, fb);
            }
        }
        fa.reset();
        fb.reset();
        // flags and per-member statistics
        DeviceBuffer unsolvable = alloc(m_ctx, N, s), alive = alloc(m_ctx, N, s);
        std::vector<u32> stats;
        std::vector<i32> root_unit(k, -1);
        {
            DeviceBuffer ds = alloc(m_ctx, u64{k} * 3 * 4, s);
            check(cudaMemsetAsync(ds.data(), 0, u64{k} * 3 * 4, s), "cudaMemsetAsync");
            check(ssk::launch_flags(goal, ptr<i32>(unit), spos, N, !has_costs, ptr<u8>(unsolvable), ptr<u8>(alive), ptr<f64>(cost),
                                    ptr<u32>(ds), s),
                  "launch_flags");
            DeviceBuffer dr = alloc(m_ctx, u64{k} * 4, s);
            check(ssk::launch_pick_roots(ptr<i32>(unit), ptr<u64>(dP), k, ptr<i32>(dr), s), "launch_pick_roots");
            download(ptr<i32>(dr), k, root_unit, s);
            download(ptr<u32>(ds), u64{k} * 3, stats, s);
            sync(s);
        }
        // the backward edges in local ids
        if (k > 1)
        {
            check(ssk::launch_local_edges(ptr<u32>(bedges), ptr<u32>(bsrc), lay, E, s), "launch_local_edges");
            check(ssk::launch_local_states(ptr<u32>(bsrc), lay, E, s), "launch_local_states");
        }
        boff.reset();
        storage->bsrc = std::move(bsrc);
        storage->bedges = std::move(bedges);
        storage->unit = std::move(unit);
        storage->cost = std::move(cost);
        storage->unsolvable = std::move(unsolvable);
        storage->alive = std::move(alive);
        sinst.reset();
        sync(s);
        const double post_ms = ms_since(tp);
        m_st.post_ms += post_ms;
        // results
        const auto to = Clock::now();
        const bool keep_device = m_o.output != StateSpaceOutput::Host;
        std::vector<const DeviceStateSpace*> down;
        std::vector<u32> down_id;
        std::vector<DeviceStateSpacePtr> down_keep;
        for (u32 m = 0; m < k; ++m)
        {
            const u32 id = w.first + m;
            DeviceStateSpaceResult& r = m_results[id];
            r.states = w.icount[m];
            StateSpaceStatus st = w.status[m];
            const u32 n_goal = stats[m * 3], n_unsolvable = stats[m * 3 + 1];
            if (st == StateSpaceStatus::Ok && m_o.space.remove_if_unsolvable && (n_goal == 0 || root_unit[m] < 0))
                st = StateSpaceStatus::Unsolvable;
            if (st == StateSpaceStatus::Ok && bad[m])
                throw std::domain_error("mymyr: the state space has a negative or NaN transition cost; goal distances by "
                                        "Dijkstra need non-negative costs");
            r.status = st;
            if (st == StateSpaceStatus::Ok)
            {
                auto space = std::make_shared<DeviceStateSpace>(
                    storage, parts[m], m_tasks[id], static_cast<u32>(P[m + 1] - P[m]), Q[m + 1] - Q[m], words[m], m_L, labels,
                    !has_costs || !not_unit[m], n_goal, n_unsolvable, static_cast<i32>(stats[m * 3 + 2]) - 1, maxd[m] + 1);
                if (host_out)
                {
                    down.push_back(space.get());
                    down_id.push_back(id);
                    down_keep.push_back(space);
                }
                if (keep_device)
                    r.space = std::move(space);
            }
            r.seconds = s_since(w.t0[m]);
        }
        if (!down.empty())
        {
            std::vector<u32> slots(down_id.size());
            for (usize j = 0; j < slots.size(); ++j)
                slots[j] = down_id[j] - w.first;
            const std::vector<datasets::StateSpacePtr> hs = out.finish(down, slots);
            for (usize j = 0; j < hs.size(); ++j)
            {
                m_results[down_id[j]].host = hs[j];
                m_results[down_id[j]].seconds = s_since(w.t0[down_id[j] - w.first]);
            }
        }
        m_st.output_ms += ms_since(to);
    }

    /// Every member failed: no post-processing.
    void finish_failed(Wave& w)
    {
        for (u32 m = 0; m < w.k; ++m)
        {
            DeviceStateSpaceResult& r = m_results[w.first + m];
            r.status = w.status[m];
            r.states = w.icount[m];
            r.seconds = s_since(w.t0[m]);
        }
    }

    ContextPtr m_ctx;
    DeviceStateSpaceOptions m_o;
    cudaStream_t m_s = nullptr;
    u32 m_threads = 1;
    std::vector<TaskPtr> m_tasks;
    rl::TaskTablePtr m_table;
    bool m_multi = false;
    u32 m_L = 0, m_S = 0;
    DeviceStateSpaceStats m_st;
    std::vector<DeviceStateSpaceResult> m_results;
    PinnedBuffer m_host_rows;
    DeviceBuffer m_dctl;          // the control block (k_ctl_*)
    std::vector<u32> m_hctl;      // its host copy
    // costs
    std::vector<heuristics::ActionCosts> m_costs;
    CostMode m_cost_mode = CostMode::Unit;
    costs::Programs m_programs;  // CostMode::Device
    // generators
    std::unique_ptr<ChunkGenerator> m_gen;
    std::shared_ptr<DeviceTaskTable> m_dt;
    detail::TableLaunch m_plan;
    detail::Fanout m_fan;
    // chunk scratch
    Scratch m_parent, m_cand, m_rank, m_scan, m_sort, m_ids, m_order, m_pos, m_order_temp, m_views, m_counts_sc, m_seg,
        m_seg_scan, m_sch_sc, m_bind_sc;
};
}  // namespace

// ---------------------------------------------------------------------------------------------- DeviceStateSpace
DeviceStateSpace::DeviceStateSpace(std::shared_ptr<const Storage> storage, Part part, TaskPtr task, u32 num_states,
                                   u64 num_transitions, u32 words, u32 label_width, bool labels, bool unit_costs,
                                   u32 num_goal, u32 num_unsolvable, i32 max_goal_distance, u32 layers)
    : m_storage(std::move(storage)), m_part(part), m_task(std::move(task)), m_n(num_states), m_e(num_transitions),
      m_words(words - m_task->numeric_words()), m_label_width(labels ? label_width : 0), m_labels(labels), m_unit_costs(unit_costs),
      m_num_goal(num_goal), m_num_unsolvable(num_unsolvable), m_max_unit(max_goal_distance), m_layers(layers)
{
}

const u64* DeviceStateSpace::state_words() const noexcept { return ptr<u64>(m_storage->rows) + m_part.rows; }
const u64* DeviceStateSpace::forward_offsets() const noexcept { return ptr<u64>(m_storage->foff) + m_part.offsets; }
const u32* DeviceStateSpace::forward_targets() const noexcept { return ptr<u32>(m_storage->targets) + m_part.edges; }
const u32* DeviceStateSpace::label_schemas() const noexcept
{
    return m_labels ? ptr<u32>(m_storage->schemas) + m_part.edges : nullptr;
}
const u32* DeviceStateSpace::label_bindings() const noexcept
{
    return m_labels && m_label_width ? ptr<u32>(m_storage->bindings) + m_part.edges * m_label_width : nullptr;
}
const f64* DeviceStateSpace::costs() const noexcept { return m_unit_costs ? nullptr : ptr<f64>(m_storage->costs) + m_part.edges; }
const u64* DeviceStateSpace::backward_offsets() const noexcept { return ptr<u64>(m_storage->boff) + m_part.offsets; }
const u32* DeviceStateSpace::backward_sources() const noexcept { return ptr<u32>(m_storage->bsrc) + m_part.edges; }
const u32* DeviceStateSpace::backward_edges() const noexcept { return ptr<u32>(m_storage->bedges) + m_part.edges; }
const i32* DeviceStateSpace::unit_goal_distances() const noexcept { return ptr<i32>(m_storage->unit) + m_part.states; }
const f64* DeviceStateSpace::cost_goal_distances() const noexcept { return ptr<f64>(m_storage->cost) + m_part.states; }
const u8* DeviceStateSpace::goal_flags() const noexcept { return ptr<u8>(m_storage->goal) + m_part.states; }
const u8* DeviceStateSpace::unsolvable_flags() const noexcept { return ptr<u8>(m_storage->unsolvable) + m_part.states; }
const u8* DeviceStateSpace::alive_flags() const noexcept { return ptr<u8>(m_storage->alive) + m_part.states; }
const ContextPtr& DeviceStateSpace::context() const noexcept { return m_storage->ctx; }
cudaStream_t DeviceStateSpace::stream() const noexcept { return m_storage->s; }

void DeviceStateSpace::use_on(cudaStream_t consumer) const
{
    DeviceGuard guard(m_storage->ctx->device());
    stream_wait(consumer, m_storage->s);
    // record_stream only adds the consumer to a buffer's user list (under the buffer's mutex): the arrays are unchanged
    auto& st = const_cast<Storage&>(*m_storage);  // created non-const (make_shared<Storage>)
    for (DeviceBuffer* b : {&st.rows, &st.foff, &st.targets, &st.schemas, &st.bindings, &st.costs, &st.boff, &st.bsrc,
                            &st.bedges, &st.unit, &st.cost, &st.goal, &st.unsolvable, &st.alive})
        if (b->data())
            b->record_stream(consumer);
}

datasets::StateSpacePtr DeviceStateSpace::to_host() const
{
    DeviceGuard guard(m_storage->ctx->device());
    sync(m_storage->s);
    const DeviceStateSpace* self = this;
    const u32 slot = 0;
    return HostOutput(m_storage->ctx, std::max<u32>(1, std::thread::hardware_concurrency()))
        .finish(std::span<const DeviceStateSpace* const>(&self, 1), std::span<const u32>(&slot, 1))
        .front();
}

// ---------------------------------------------------------------------------------------------- entry points
std::string state_space_unsupported(const Task& task)
{
    return ChunkGenerator::unsupported(task);
}

DeviceStateSpaceResult state_space(ContextPtr ctx, TaskPtr task, const DeviceStateSpaceOptions& options, DeviceStateSpaceStats* stats)
{
    if (!task)
        throw std::invalid_argument("mymyr: device state space: null task");
    Engine e(std::move(ctx), {std::move(task)}, nullptr, options);
    DeviceStateSpaces r = e.run();
    if (stats)
        *stats = r.stats;
    return std::move(r.results.front());
}

DeviceStateSpaces state_spaces(ContextPtr ctx, rl::TaskTablePtr table, const DeviceStateSpaceOptions& options)
{
    if (!table)
        throw std::invalid_argument("mymyr: device state spaces: null table");
    std::vector<TaskPtr> tasks;
    for (u32 i = 0; i < table->size(); ++i)
        tasks.push_back(table->task(i));
    Engine e(std::move(ctx), std::move(tasks), std::move(table), options);
    return e.run();
}

std::shared_ptr<const datasets::GeneralizedStateSpace> generalized_state_space(std::span<const DeviceStateSpaceResult> results,
                                                                               bool sort_ascending)
{
    std::vector<datasets::StateSpaceResult> host(results.size());
    for (usize i = 0; i < results.size(); ++i)
    {
        const DeviceStateSpaceResult& r = results[i];
        host[i].status = r.status;
        host[i].states = r.states;
        host[i].seconds = r.seconds;
        host[i].space = r.host ? r.host : r.space ? r.space->to_host() : nullptr;
        if (r.status == datasets::StateSpaceStatus::Ok && !host[i].space)
            throw std::invalid_argument("mymyr: generalized_state_space: a successful result without a space");
    }
    return datasets::GeneralizedStateSpace::create(datasets::ordered_spaces(host, sort_ascending));
}
}  // namespace mymyr::cuda
