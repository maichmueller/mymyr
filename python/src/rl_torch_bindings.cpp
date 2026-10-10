// mymyr._core._rl_torch: the native side of mymyr.rl.torch.
//   Env(table, device=None | cuda index, ...)      batched planning environments over a TaskSuite or a TaskTable (or a
//                                                  Task: its table of one): reset / refresh / step on caller-owned arrays
//                                                  (rl::HostEnv on the host, cuda::DeviceEnv on a CUDA device; the same
//                                                  results bit for bit)
//   philox(counters, key)                          Philox-4x32-10 blocks (rl/rng.hpp; equal to cuRAND's)
//   successor_indices(seed, envs, draws, counts)   the random policy's choices (rl::rng::successor_index)
//   fast_unsupported(table, ...)                   why the device fast path does not run a table or suite ('' if it
//                                                  does)
// Arrays are taken in place through DLPack (destination passing): host arrays (NumPy, torch CPU tensors) for a
// host Env, CUDA arrays on its device for a device Env. Device arrays are imported with the array API's stream handoff
// on the call's stream (stream=None: the Env's context's stream; torch passes its current stream) and released only
// after the work enqueued there has completed; nothing synchronizes the host except check_errors(). mymyr.rl.torch
// wraps these calls into torch custom ops (mymyr::reset, mymyr::step, ...) and the TorchRL environment PlanningEnv.
//
// The environment state of a batch over a table or suite: states [N, row_words], task_ids [N] int32 (the rows'
// instances, global over a suite; optional over a table of one), steps, draws, the fast path's cache (counts, views)
// and, optionally, per-env goal masks goal_pos / goal_neg [N, words] (the step's goal test is then the mask test).

#include "arrays.hpp"
#include "dlpack.hpp"
#include "py_table.hpp"
#include "py_task.hpp"
#include "rl_imports.hpp"
#include "rl_typing.hpp"
#include "typing.hpp"

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/rng.hpp"

#if defined(MYMYR_HAS_CUDA)
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/cuda/runtime.hpp"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <deque>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

#if defined(MYMYR_HAS_CUDA)
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device);  // cuda_bindings.cpp
#endif

namespace
{
using namespace rlimp;
using ArrayArg = Arg<ann::ArrayLike>;
using ContextArg = Arg<ann::Any>;
using PathArg = Arg<ann::EnvPath>;
using ArrayOut = Arg<ann::Any>;

// ------------------------------------------------------------------------------------------------ Env

class PyEnv
{
public:
    PyEnv(SuiteArg table, nb::object device, ContextArg ctx, const rl::EnvConfig& cfg, PathArg path, u32 threads)
        : ref(suite_of(table))
    {
        const rl::TaskSuitePtr& t = ref.suite;
        const std::string p = nb::cast<std::string>(path);
        if (p != "auto" && p != "fast" && p != "general")
            throw nb::value_error("mymyr: path must be 'auto', 'fast' or 'general'");
        m_init_count.resize(t->size());
        if (!device.is_none())
        {
            m_device = int_arg<int>(device, "device", 0);
            if (m_device < 0)
                throw nb::value_error("mymyr: device must be a CUDA device index (>= 0) or None (the CPU)");
#if defined(MYMYR_HAS_CUDA)
            cuda::ContextPtr c = table_device_context(ref.obj, ctx, m_device);
            const auto mode = p == "fast" ? cuda::DeviceEnv::Path::Fast
                              : p == "general" ? cuda::DeviceEnv::Path::General
                                               : cuda::DeviceEnv::Path::Auto;
            nb::gil_scoped_release release;
            m_dev = std::make_unique<cuda::DeviceEnv>(std::move(c), t, cfg, mode);
            m_W = m_dev->words();
            m_L = m_dev->label_width();
            m_S = m_dev->cache_schemas();
            m_V = m_dev->cache_view_words();
            for (u32 i = 0; i < t->size(); ++i)
                m_init_count[i] = m_dev->initial_count(i);
#else
            (void)ctx;
            throw nb::value_error("mymyr: this mymyr was built without the CUDA backend; use device=None (the CPU)");
#endif
        }
        else
        {
            if (!ctx.is_none())
                throw nb::value_error("mymyr: ctx is for device environments (device=None runs on the CPU)");
            if (p == "fast")
                throw nb::value_error("mymyr: path='fast' is a device path (device=None runs on the CPU)");
            {
                nb::gil_scoped_release release;
                m_host = std::make_unique<rl::HostEnv>(t, cfg);
            }
            if (threads > 1)
                m_pool = std::make_unique<ThreadPool>(threads);
            m_W = m_host->words();
            m_L = std::max<u32>(1, t->label_width());
            for (u32 i = 0; i < t->size(); ++i)
                m_init_count[i] = m_host->initial_count(i);
        }
        m_NN = t->numeric_words();
    }
    ~PyEnv()
    {
#if defined(MYMYR_HAS_CUDA)
        m_dev.reset();  // synchronizes its stream: the pending imports are free to go
#endif
    }
    PyEnv(const PyEnv&) = delete;
    PyEnv& operator=(const PyEnv&) = delete;

    SuiteRef ref;

    [[nodiscard]] bool on_device() const noexcept { return m_device >= 0; }
    [[nodiscard]] std::optional<int> device() const
    {
        return m_device < 0 ? std::nullopt : std::optional<int>(m_device);
    }
    [[nodiscard]] const rl::EnvConfig& config() const
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
            return m_dev->config();
#endif
        return m_host->config();
    }
    [[nodiscard]] bool fast() const
    {
#if defined(MYMYR_HAS_CUDA)
        return m_dev && m_dev->fast();
#else
        return false;
#endif
    }
    [[nodiscard]] u32 num_instances() const noexcept { return ref.suite->size(); }
    [[nodiscard]] u32 words() const noexcept { return m_W; }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_NN; }
    [[nodiscard]] u32 row_words() const noexcept { return m_W + m_NN; }
    [[nodiscard]] u32 label_width() const noexcept { return m_L; }
    [[nodiscard]] u32 cache_schemas() const noexcept { return m_S; }
    [[nodiscard]] u64 cache_view_words() const noexcept { return m_V; }
    [[nodiscard]] const std::vector<u32>& initial_counts() const noexcept { return m_init_count; }

    void set_seed(u64 seed, StreamArg stream)
    {
        const std::intptr_t dl = select_stream(stream);
        std::lock_guard lock(m_mutex);
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
        {
            nb::gil_scoped_release release;
            m_dev->set_stream(reinterpret_cast<cudaStream_t>(dl));
            m_dev->set_seed(seed);
            return;
        }
#else
        (void)dl;
#endif
        m_host->set_seed(seed);
    }

    /// Sizes a device Env's scratch for batches of up to `rows` rows (calls then allocate nothing: CUDA graphs).
    void reserve(u64 rows)
    {
#if defined(MYMYR_HAS_CUDA)
        std::lock_guard lock(m_mutex);
        if (m_dev)
        {
            nb::gil_scoped_release release;
            m_dev->reserve(rows);
        }
#else
        (void)rows;
#endif
    }

    [[nodiscard]] std::string capture_unsupported() const
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
            return m_dev->capture_unsupported();
#endif
        return "a CPU Env (CUDA graphs capture device work)";
    }

    void set_launch(const std::string& launch)
    {
#if defined(MYMYR_HAS_CUDA)
        if (launch != "widest" && launch != "per_bucket")
            throw nb::value_error("mymyr: launch must be 'widest' or 'per_bucket'");
        std::lock_guard lock(m_mutex);
        if (m_dev)
            m_dev->set_launch(launch == "widest" ? cuda::BucketLaunch::Widest : cuda::BucketLaunch::PerBucket);
#else
        (void)launch;
#endif
    }

    /// The instances' initial states [I, row_words] (NumPy uint64).
    ArrayOut initial_states() const
    {
        const rl::TaskSuite& t = *ref.suite;
        const u64 I = t.size(), RW = row_words();
        auto block = Block::make(std::max<u64>(I * RW, 1) * sizeof(u64));
        auto* w = reinterpret_cast<u64*>(block->data());
        for (u64 i = 0; i < I; ++i)
            t.initial_row(static_cast<u32>(i), w + i * RW, m_W, m_NN);
        ArraySpec spec{block, w, rl::DType::U64, {static_cast<i64>(I), static_cast<i64>(RW)}, {}, false, true};
        return export_array(std::move(spec), Framework::Numpy, default_words(Framework::Numpy));
    }

    void reset(nb::handle states, nb::handle task_ids, nb::handle steps, nb::handle counts, nb::handle views,
               nb::handle goal_pos, nb::handle goal_neg, nb::handle count, nb::handle mask, bool keep_goals,
               nb::handle stream)
    {
        Call c(*this, stream);
        rl::EnvBatch b = batch(c.in, states, task_ids, steps, nb::none(), counts, views, goal_pos, goal_neg);
        const i64 n = static_cast<i64>(b.rows);
        const u8* m = c.in.get<const u8>(mask, "mask", {boolt, u8t}, {n}, false);
        i32* cnt = c.in.get<i32>(count, "count", {i32t}, {n}, true);
        c.run([&] { run_reset(b, m, cnt, keep_goals); });
    }

    void refresh(nb::handle states, nb::handle task_ids, nb::handle counts, nb::handle views, nb::handle count,
                 nb::handle stream)
    {
        Call c(*this, stream);
        rl::EnvBatch b = batch(c.in, states, task_ids, nb::none(), nb::none(), counts, views, nb::none(), nb::none());
        i32* cnt = c.in.get<i32>(count, "count", {i32t}, {static_cast<i64>(b.rows)}, true);
        c.run([&] { run_refresh(b, cnt); });
    }

    void step(nb::handle states, nb::handle task_ids, nb::handle steps, nb::handle draws, nb::handle counts,
              nb::handle views, nb::handle goal_pos, nb::handle goal_neg, nb::handle action, nb::handle next_task_ids,
              nb::handle reward, nb::handle terminated, nb::handle truncated, nb::handle count, nb::handle final_states,
              nb::handle schema, nb::handle binding, nb::handle invalid, nb::handle goal, u64 first_env,
              nb::handle stream)
    {
        Call c(*this, stream);
        rl::EnvBatch b = batch(c.in, states, task_ids, steps, draws, counts, views, goal_pos, goal_neg);
        b.first_env = first_env;
        const i64 n = static_cast<i64>(b.rows);
        const i64* act = c.in.get<const i64>(action, "action", {i64t}, {n}, false);
        if (!act && !b.draws && n > 0)
            throw nb::value_error("mymyr: step: the random policy (action=None) needs the draw counters 'draws'");
        const i32* next = c.in.get<const i32>(next_task_ids, "next_task_ids", {i32t}, {n}, false);
        rl::StepOutputs o;
        o.reward = c.in.get<f32>(reward, "reward", {f32t}, {n}, true);
        o.terminated = c.in.get<u8>(terminated, "terminated", {boolt, u8t}, {n}, true);
        o.truncated = c.in.get<u8>(truncated, "truncated", {boolt, u8t}, {n}, true);
        o.count = c.in.get<i32>(count, "count", {i32t}, {n}, true);
        o.final_states = c.in.words(final_states, "final_states", n, row_words(), true);
        o.schema = c.in.get<i32>(schema, "schema", {i32t}, {n}, true);
        std::vector<i64> ext;
        o.binding = c.in.get<i32>(binding, "binding", {i32t}, {n, -1}, true, &ext);
        o.label_width = o.binding ? static_cast<u32>(ext[1]) : 0;
        o.invalid = c.in.get<u8>(invalid, "invalid", {boolt, u8t}, {n}, true);
        o.goal = c.in.get<u8>(goal, "goal", {boolt, u8t}, {n}, true);
        c.run([&] { run_step(b, o, act, next); });
    }

    // The entry points of the torch ops (mymyr.rl.torch._ops): the same calls on the data pointers of tensors that
    // the caller has checked (dtype, shape, device, contiguity) and keeps alive, in the order of the array methods (0:
    // None). They run on `stream` (torch's current stream) in stream order, like torch's own kernels, and skip the
    // DLPack imports (about 10 us per torch tensor).
    using Ptrs = std::vector<std::uintptr_t>;

    // states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask
    void reset_ptrs(u64 rows, const Ptrs& p, bool keep_goals, StreamArg stream)
    {
        check_ptrs(p, 9);
        Call c(*this, stream);
        rl::EnvBatch b = raw_batch(rows, p[0], p[1], p[2], 0, p[3], p[4], p[5], p[6]);
        c.run([&] { run_reset(b, ptr<const u8>(p[8]), ptr<i32>(p[7]), keep_goals); });
    }

    void refresh_ptrs(u64 rows, const Ptrs& p, StreamArg stream)  // states, task_ids, counts, views, count
    {
        check_ptrs(p, 5);
        Call c(*this, stream);
        rl::EnvBatch b = raw_batch(rows, p[0], p[1], 0, 0, p[2], p[3], 0, 0);
        c.run([&] { run_refresh(b, ptr<i32>(p[4])); });
    }

    // states, task_ids, steps, draws, counts, views, goal_pos, goal_neg, action, next_task_ids, reward, terminated,
    // truncated, count, final_states, schema, binding, invalid, goal
    void step_ptrs(u64 rows, const Ptrs& p, u32 label_width, u64 first_env, StreamArg stream)
    {
        check_ptrs(p, 19);
        Call c(*this, stream);
        rl::EnvBatch b = raw_batch(rows, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        b.first_env = first_env;
        const i64* act = ptr<const i64>(p[8]);
        if (!act && !b.draws && rows > 0)
            throw nb::value_error("mymyr: step: the random policy (action=None) needs the draw counters 'draws'");
        rl::StepOutputs o;
        o.reward = ptr<f32>(p[10]);
        o.terminated = ptr<u8>(p[11]);
        o.truncated = ptr<u8>(p[12]);
        o.count = ptr<i32>(p[13]);
        o.final_states = ptr<u64>(p[14]);
        o.schema = ptr<i32>(p[15]);
        o.binding = ptr<i32>(p[16]);
        o.label_width = o.binding ? label_width : 0;
        o.invalid = ptr<u8>(p[17]);
        o.goal = ptr<u8>(p[18]);
        c.run([&] { run_step(b, o, act, ptr<const i32>(p[9])); });
    }

    /// The random policy's next choices (rl::rng::successor_index at the current draw counters), limited to
    /// max_actions successors (0: no limit), without stepping: out [N] int64 (0 where there is no successor).
    void random_actions(nb::handle draws, nb::handle count, nb::handle out, u64 first_env, u32 max_actions,
                        nb::handle stream)
    {
        Call c(*this, stream);
        std::vector<i64> ext;
        const u64* d = c.in.get<const u64>(draws, "draws", {i64t, u64t}, {-1}, false, &ext);
        if (!d)
            throw nb::type_error("mymyr: 'draws' is required");
        const i64 n = ext[0];
        const i32* cnt = c.in.get<const i32>(count, "count", {i32t}, {n}, false);
        i64* o = c.in.get<i64>(out, "out", {i64t}, {n}, true);
        if (!cnt || !o)
            throw nb::type_error("mymyr: 'count' and 'out' are required");
        c.run([&] { run_random_actions(d, cnt, static_cast<u64>(n), first_env, max_actions, o); });
    }

    // draws, count, out
    void random_actions_ptrs(u64 rows, const Ptrs& p, u64 first_env, u32 max_actions, StreamArg stream)
    {
        check_ptrs(p, 3);
        Call c(*this, stream);
        c.run([&] { run_random_actions(ptr<const u64>(p[0]), ptr<const i32>(p[1]), rows, first_env, max_actions, ptr<i64>(p[2])); });
    }

    void check_errors()
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
        {
            std::lock_guard lock(m_mutex);
            {
                nb::gil_scoped_release release;
                m_dev->check_errors();
            }
            drain(true);
        }
#endif
    }

private:
    /// One call: the imports on its stream, the Env's lock while it runs, and the release of the device imports once
    /// the work enqueued on the stream has completed.
    class Call
    {
    public:
        Call(PyEnv& e, nb::handle stream) : m_env(e), m_dl(e.select_stream(stream)), in(e.m_device, m_dl) {}
        template<class F>
        void run(F&& f)
        {
            {
                std::lock_guard lock(m_env.m_mutex);
                nb::gil_scoped_release release;
#if defined(MYMYR_HAS_CUDA)
                if (m_env.m_dev)
                    m_env.m_dev->set_stream(reinterpret_cast<cudaStream_t>(m_dl));  // DLPack 1, 2: legacy, per-thread
#endif
                f();
            }
            m_env.defer(std::move(in.keep), m_dl);
        }

    private:
        PyEnv& m_env;
        std::intptr_t m_dl;  // the call's stream as a DLPack stream value (host Env: k_stream_default)

    public:
        Imports in;
    };

    void run_reset(rl::EnvBatch& b, const u8* mask, i32* count, bool keep_goals)
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
            return m_dev->reset(b, mask, count, keep_goals);
#endif
        m_host->reset(b, mask, count, keep_goals);
    }
    void run_refresh(rl::EnvBatch& b, i32* count)
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
            return m_dev->refresh(b, count);
#endif
        if (count)
            m_host->count(b, count, m_pool.get());
    }
    void run_step(rl::EnvBatch& b, const rl::StepOutputs& o, const i64* action, const i32* next_task_ids)
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
            return m_dev->step(b, o, rl::Actions{action, nullptr}, next_task_ids);
#endif
        m_host->step(b, o, rl::Actions{action, nullptr}, next_task_ids, m_pool.get());
    }
    void run_random_actions(const u64* d, const i32* cnt, u64 rows, u64 first_env, u32 max_actions, i64* o)
    {
        const u64 seed = config().seed;
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
        {
            m_dev->random_actions(d, cnt, rows, first_env, max_actions, o);
            return;
        }
#endif
        for (u64 i = 0; i < rows; ++i)
        {
            u32 k = cnt[i] > 0 ? static_cast<u32>(cnt[i]) : 0u;
            if (max_actions && k > max_actions)
                k = max_actions;
            o[i] = k ? i64{rl::rng::successor_index(seed, first_env + i, d[i], k)} : i64{0};
        }
    }

    template<class T>
    static T* ptr(std::uintptr_t p)
    {
        return reinterpret_cast<T*>(p);
    }
    static void check_ptrs(const Ptrs& p, usize n)
    {
        if (p.size() != n)
            throw nb::value_error(("mymyr: expected " + std::to_string(n) + " data pointers").c_str());
    }
    rl::EnvBatch raw_batch(u64 rows, std::uintptr_t states, std::uintptr_t task_ids, std::uintptr_t steps,
                           std::uintptr_t draws, std::uintptr_t counts, std::uintptr_t views, std::uintptr_t goal_pos,
                           std::uintptr_t goal_neg) const
    {
        rl::EnvBatch b;
        b.rows = rows;
        b.states = ptr<u64>(states);
        b.words = m_W;
        b.numeric_words = m_NN;
        b.task_ids = ptr<i32>(task_ids);
        b.steps = ptr<i32>(steps);
        b.draws = ptr<u64>(draws);
        b.counts = ptr<u32>(counts);
        b.views = ptr<u64>(views);
        b.goal_pos = ptr<u64>(goal_pos);
        b.goal_neg = ptr<u64>(goal_neg);
        return b;
    }

    rl::EnvBatch batch(Imports& in, nb::handle states, nb::handle task_ids, nb::handle steps, nb::handle draws,
                       nb::handle counts, nb::handle views, nb::handle goal_pos, nb::handle goal_neg)
    {
        rl::EnvBatch b;
        if (states.is_none())
            throw nb::type_error("mymyr: 'states' is required");
        b.states = in.words(states, "states", -1, row_words(), true, &b.rows);
        b.words = m_W;
        b.numeric_words = m_NN;
        const i64 n = static_cast<i64>(b.rows);
        b.task_ids = in.get<i32>(task_ids, "task_ids", {i32t}, {n}, true);
        b.steps = in.get<i32>(steps, "steps", {i32t}, {n}, true);
        b.draws = in.get<u64>(draws, "draws", {i64t, u64t}, {n}, true);
        b.counts = in.get<u32>(counts, "counts", {i32t, u32t}, {n, static_cast<i64>(m_S)}, true);
        b.views = in.get<u64>(views, "views", {i64t, u64t}, {n, static_cast<i64>(m_V)}, true);
        b.goal_pos = in.words(goal_pos, "goal_pos", n, m_W, true);
        b.goal_neg = in.words(goal_neg, "goal_neg", n, m_W, true);
        return b;
    }

    /// The call's stream as a DLPack stream value (host Env: unused), after dropping the completed imports.
    std::intptr_t select_stream(nb::handle stream)
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev)
        {
            std::intptr_t v = 0;
            if (stream.is_none())
                v = reinterpret_cast<std::intptr_t>(m_dev->context()->stream());
            else
            {
                nb::object h = nb::getattr(stream, "cuda_stream", nb::none());
                v = nb::cast<std::intptr_t>(h.is_none() ? nb::borrow(stream) : h);
                if (v == dl::k_stream_none)
                    throw nb::value_error("mymyr: stream -1 means 'no synchronization'; pass a stream to run on");
                if (v == 0)
                    v = dl::k_stream_legacy;  // torch reports its default stream as 0
            }
            // the completed imports go (not during a CUDA graph capture: no event queries then)
            if (!capturing(v))
                drain(false);
            return v;
        }
#endif
        if (!stream.is_none())
            throw nb::value_error("mymyr: stream is for device environments (this Env runs on the CPU)");
        return dl::k_stream_default;
    }

    /// Keeps the call's imports until the work enqueued on its stream so far has completed, or, when the call was
    /// captured into a CUDA graph, until the Env goes (the graph's replays read them).
    void defer(std::vector<std::shared_ptr<void>> keep, std::intptr_t stream)
    {
#if defined(MYMYR_HAS_CUDA)
        if (m_dev && !keep.empty())
        {
            cuda::DeviceGuard g(m_device);
            if (capturing(stream))
            {
                std::lock_guard lock(m_pending_mutex);
                for (auto& k : keep)
                    m_captured.push_back(std::move(k));
                return;
            }
            cuda::Event e;
            e.record(reinterpret_cast<cudaStream_t>(stream));
            std::lock_guard lock(m_pending_mutex);
            m_pending.push_back({std::move(e), std::move(keep)});
        }
#else
        (void)keep;
        (void)stream;
#endif
    }

#if defined(MYMYR_HAS_CUDA)
    /// Whether the stream of a DLPack stream value captures a CUDA graph.
    [[nodiscard]] bool capturing(std::intptr_t dl) const
    {
        cuda::DeviceGuard g(m_device);
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cuda::check(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(dl), &st), "cudaStreamIsCapturing");
        return st != cudaStreamCaptureStatusNone;
    }
    struct Pending
    {
        cuda::Event event;
        std::vector<std::shared_ptr<void>> keep;
    };
    /// Drops the imports whose work has completed (all of them after waiting, with `all`).
    void drain(bool all)
    {
        std::deque<Pending> done;
        {
            std::lock_guard lock(m_pending_mutex);
            while (!m_pending.empty() && (all || m_pending.front().event.done()))
            {
                if (all)
                    m_pending.front().event.synchronize();
                done.push_back(std::move(m_pending.front()));
                m_pending.pop_front();
            }
        }
    }
    std::mutex m_pending_mutex;
    std::deque<Pending> m_pending;  // declared before m_dev: destroyed after it (its destructor synchronizes)
    std::vector<std::shared_ptr<void>> m_captured;  // the imports of captured calls
    std::unique_ptr<cuda::DeviceEnv> m_dev;
#endif
    std::unique_ptr<rl::HostEnv> m_host;
    std::unique_ptr<ThreadPool> m_pool;
    std::mutex m_mutex;  // one call at a time (the Env's scratch)
    int m_device = -1;
    u32 m_W = 1, m_NN = 0, m_L = 1, m_S = 0;
    u64 m_V = 0;
    std::vector<u32> m_init_count;
};

nb::object host_u32_array(std::shared_ptr<Block> block, std::vector<i64> shape)
{
    ArraySpec s{block, block->data(), rl::DType::U32, std::move(shape), {}, false, false};
    return export_array(std::move(s), Framework::Numpy, default_words(Framework::Numpy));
}
}  // namespace

void bind_rl_torch(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_rl_torch", "Batched planning environments and the counter-based RNG (mymyr.rl.torch)");

    nb::class_<PyEnv>(m, "Env",
                      "N planning environments over a TaskSuite or a TaskTable (or a Task), stepped in lockstep on "
                      "caller-owned arrays (see mymyr.rl.torch). device=None runs them on the CPU (rl::HostEnv, any "
                      "table), a CUDA device index on that device (cuda::DeviceEnv: classical tables); both give the "
                      "same results bit for bit. Over several instances every call takes the rows' task_ids [N] int32 "
                      "(global ids over a suite).")
        .def(
            "__init__",
            [](PyEnv* self, SuiteArg table, Arg<int> device, ContextArg ctx, IntArg seed_in, IntArg max_steps_in, f32 step_reward,
               f32 goal_reward, Arg<ann::DeadEnd> dead_end, f32 dead_end_reward, bool dead_end_terminal, bool autoreset,
               bool canonical, bool witness, PathArg path, IntArg threads_in) {
                const u64 seed = int_arg<u64>(seed_in, "seed");
                const u32 max_steps = int_arg<u32>(max_steps_in, "max_steps");
                const u32 threads = threads_arg(threads_in);
                rl::EnvConfig c;
                c.seed = seed;
                c.max_steps = max_steps;
                c.step_reward = step_reward;
                c.goal_reward = goal_reward;
                c.dead_end = parse_dead_end(nb::cast<std::string>(dead_end));
                c.dead_end_reward = dead_end_reward;
                c.dead_end_terminal = dead_end_terminal;
                c.autoreset = autoreset;
                c.canonical_order = canonical;
                c.witness_pruning = witness;
                new (self) PyEnv(std::move(table), std::move(device), std::move(ctx), c, std::move(path), threads);
            },
            "table"_a, nb::kw_only(), "device"_a = nb::none(), "ctx"_a = nb::none(), "seed"_a = 0, "max_steps"_a = 0,
            "step_reward"_a = -1.0f, "goal_reward"_a = 0.0f, "dead_end"_a = "no_successors", "dead_end_reward"_a = 0.0f,
            "dead_end_terminal"_a = true, "autoreset"_a = true, "canonical"_a = true, "witness"_a = false,
            "path"_a = "auto", "threads"_a = 1)
        .def_prop_ro("table", [](const PyEnv& e) { return SuiteArg(e.ref.obj); })
        .def_prop_ro("num_instances", &PyEnv::num_instances)
        .def_prop_ro("device", &PyEnv::device, "The CUDA device index, or None (the CPU).")
        .def_prop_ro("fast", &PyEnv::fast, "Whether a device Env takes the fast path (the per-environment cache).")
        .def_prop_ro("words", &PyEnv::words, "Atom words per state row (the table's words, at least 1).")
        .def_prop_ro("numeric_words", &PyEnv::numeric_words, "Numeric words per row after the atom words.")
        .def_prop_ro("row_words", &PyEnv::row_words, "Words per state row: words + numeric_words.")
        .def_prop_ro("label_width", &PyEnv::label_width, "Columns of step()'s binding output (the largest arity, >= 1).")
        .def_prop_ro("cache_schemas", &PyEnv::cache_schemas, "Columns of the fast path's counts cache (0 otherwise).")
        .def_prop_ro("cache_view_words", &PyEnv::cache_view_words, "Columns of the fast path's views cache (0 otherwise).")
        .def_prop_ro("initial_counts", &PyEnv::initial_counts, "Successors of each instance's initial state [I].")
        .def_prop_ro("seed", [](const PyEnv& e) { return e.config().seed; })
        .def_prop_ro("max_steps", [](const PyEnv& e) { return e.config().max_steps; })
        .def_prop_ro("step_reward", [](const PyEnv& e) { return e.config().step_reward; })
        .def_prop_ro("goal_reward", [](const PyEnv& e) { return e.config().goal_reward; })
        .def_prop_ro("dead_end", [](const PyEnv& e) { return std::string(dead_end_name(e.config().dead_end)); })
        .def_prop_ro("dead_end_reward", [](const PyEnv& e) { return e.config().dead_end_reward; })
        .def_prop_ro("dead_end_terminal", [](const PyEnv& e) { return e.config().dead_end_terminal; })
        .def_prop_ro("autoreset", [](const PyEnv& e) { return e.config().autoreset; })
        .def_prop_ro("canonical", [](const PyEnv& e) { return e.config().canonical_order; })
        .def_prop_ro("witness", [](const PyEnv& e) { return e.config().witness_pruning; })
        .def("set_seed", &PyEnv::set_seed, "seed"_a, nb::kw_only(), "stream"_a = nb::none(),
             "The key of the successor draws; the draw counters are the caller's (zero them to restart the streams). A "
             "device Env writes it on `stream` (None: the Env's context's stream) in stream order: steps enqueued "
             "before draw with the old key, steps and replays of captured CUDA graphs after with the new one.")
        .def("reserve", &PyEnv::reserve, "rows"_a,
             "Sizes a device Env's scratch for batches of up to `rows` rows, so that its calls allocate nothing (a call "
             "captured into a CUDA graph cannot allocate; a call of more rows grows the scratch). A no-op on the CPU.")
        .def_prop_ro("capture_unsupported", &PyEnv::capture_unsupported,
                     "'' if reset / refresh / step can be captured into a CUDA graph (torch.cuda.graph, torch.compile's "
                     "CUDA graphs; a device Env on the fast path), else why not.")
        .def("set_launch", &PyEnv::set_launch, "launch"_a,
             "How a device Env launches its pick over the row-width buckets of a table of several instances: 'widest' "
             "(one launch per object-word group at its widest bucket; the default) or 'per_bucket'. The results do not "
             "depend on it; a no-op on the CPU.")
        .def("initial_states", &PyEnv::initial_states, "The instances' initial states as rows: NumPy uint64 [I, row_words].")
        .def(
            "reset",
            [](PyEnv& e, ArrayArg states, ArrayArg task_ids, ArrayArg steps, ArrayArg counts, ArrayArg views,
               ArrayArg goal_pos, ArrayArg goal_neg, ArrayArg count, ArrayArg mask, bool keep_goals, StreamArg stream) {
                e.reset(states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask, keep_goals, stream);
            },
            "states"_a, "task_ids"_a = nb::none(), "steps"_a = nb::none(), nb::kw_only(), "counts"_a = nb::none(),
            "views"_a = nb::none(), "goal_pos"_a = nb::none(), "goal_neg"_a = nb::none(), "count"_a = nb::none(),
            "mask"_a = nb::none(), "keep_goals"_a = false, "stream"_a = nb::none(),
            "Resets the rows with mask[i] (all rows without a mask) to the initial states of their instances "
            "(task_ids [N] int32, set by the caller first; None over a table of one): states [N, row_words], steps [N] "
            "int32 (0), the fast path's cache counts [N, cache_schemas] and views [N, cache_view_words], per-env goal "
            "masks goal_pos / goal_neg [N, words] (the instances' goals, unless keep_goals), count [N] int32 (the "
            "initial states' successors). The draw counters are kept.")
        .def(
            "refresh",
            [](PyEnv& e, ArrayArg states, ArrayArg task_ids, ArrayArg counts, ArrayArg views, ArrayArg count,
               StreamArg stream) { e.refresh(states, task_ids, counts, views, count, stream); },
            "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "counts"_a = nb::none(), "views"_a = nb::none(),
            "count"_a = nb::none(), "stream"_a = nb::none(),
            "Recomputes the fast path's cache and the successor counts [N] of states (or task ids) the caller wrote.")
        .def(
            "step",
            [](PyEnv& e, ArrayArg states, ArrayArg task_ids, ArrayArg steps, ArrayArg draws, ArrayArg counts,
               ArrayArg views, ArrayArg goal_pos, ArrayArg goal_neg, ArrayArg action, ArrayArg next_task_ids,
               ArrayArg reward, ArrayArg terminated, ArrayArg truncated, ArrayArg count, ArrayArg final_states,
               ArrayArg schema, ArrayArg binding, ArrayArg invalid, ArrayArg goal, IntArg first_env_in, StreamArg stream) {
                const u64 first_env = int_arg<u64>(first_env_in, "first_env");
                e.step(states, task_ids, steps, draws, counts, views, goal_pos, goal_neg, action, next_task_ids, reward,
                       terminated, truncated, count, final_states, schema, binding, invalid, goal, first_env, stream);
            },
            "states"_a, "task_ids"_a = nb::none(), "steps"_a = nb::none(), "draws"_a = nb::none(), nb::kw_only(),
            "counts"_a = nb::none(), "views"_a = nb::none(), "goal_pos"_a = nb::none(), "goal_neg"_a = nb::none(),
            "action"_a = nb::none(), "next_task_ids"_a = nb::none(), "reward"_a = nb::none(),
            "terminated"_a = nb::none(), "truncated"_a = nb::none(), "count"_a = nb::none(),
            "final_states"_a = nb::none(), "schema"_a = nb::none(), "binding"_a = nb::none(), "invalid"_a = nb::none(),
            "goal"_a = nb::none(), "first_env"_a = 0, "stream"_a = nb::none(),
            "One step of every environment (rl/env.hpp): action [N] int64 indexes the canonical successor order "
            "(None: the random policy, drawing from draws [N] (u)int64 with the env ids first_env + i); next_task_ids "
            "[N] int32 names the instances autoresetting rows restart in. In place: states, task_ids, steps, draws, "
            "the cache and the goal masks; outputs (None: not written): reward [N] float32, terminated, truncated, "
            "invalid and goal [N] bool, count [N] int32, final_states [N, row_words], schema [N] int32, binding "
            "[N, >= label_width] int32.")
        .def("_reset_ptrs", &PyEnv::reset_ptrs, "rows"_a, "ptrs"_a, "keep_goals"_a = false, "stream"_a = nb::none(),
             "reset() on data pointers (states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask; 0: "
             "None) of checked tensors; the torch ops' entry point.")
        .def("_refresh_ptrs", &PyEnv::refresh_ptrs, "rows"_a, "ptrs"_a, "stream"_a = nb::none(),
             "refresh() on data pointers (states, task_ids, counts, views, count; 0: None) of checked tensors.")
        .def("_step_ptrs", &PyEnv::step_ptrs, "rows"_a, "ptrs"_a, "label_width"_a, "first_env"_a = 0,
             "stream"_a = nb::none(),
             "step() on data pointers (states, task_ids, steps, draws, counts, views, goal_pos, goal_neg, action, "
             "next_task_ids, reward, terminated, truncated, count, final_states, schema, binding, invalid, goal; 0: "
             "None) of checked tensors.")
        .def("_random_actions_ptrs", &PyEnv::random_actions_ptrs, "rows"_a, "ptrs"_a, "first_env"_a = 0,
             "max_actions"_a = 0, "stream"_a = nb::none(),
             "random_actions() on data pointers (draws, count, out) of checked tensors.")
        .def(
            "random_actions",
            [](PyEnv& e, ArrayArg draws, ArrayArg count, ArrayArg out, IntArg first_env_in, IntArg max_actions_in, StreamArg stream) {
                const u64 first_env = int_arg<u64>(first_env_in, "first_env");
                const u32 max_actions = int_arg<u32>(max_actions_in, "max_actions");
                e.random_actions(draws, count, out, first_env, max_actions, stream);
            },
            "draws"_a, "count"_a, "out"_a, nb::kw_only(), "first_env"_a = 0, "max_actions"_a = 0, "stream"_a = nb::none(),
            "The random policy's next choices without stepping: out[i] = successor_indices(seed, first_env + i, "
            "draws[i], min(count[i], max_actions)) (max_actions 0: no limit; 0 where there is no successor), int64 [N]. "
            "A step with these actions takes the same successors as the random policy's step (count <= max_actions).")
        .def("check_errors", &PyEnv::check_errors,
             "Device Env: waits for its work and raises RuntimeError if a fast-path step met a state its cache did not "
             "describe (states written without refresh()), ValueError for task ids outside the table. A no-op on the "
             "CPU.")
        .def("__repr__", [](const PyEnv& e) {
            return std::string("Env(") + (e.on_device() ? "cuda:" + std::to_string(*e.device()) : std::string("cpu")) +
                   (e.fast() ? ", fast" : "") + ", instances=" + std::to_string(e.num_instances()) +
                   ", words=" + std::to_string(e.row_words()) + ")";
        });

    m.def(
        "fast_unsupported",
        [](SuiteArg table, bool canonical, bool witness) -> std::string {
            const SuiteRef ref = suite_of(table);
#if defined(MYMYR_HAS_CUDA)
            rl::EnvConfig c;
            c.canonical_order = canonical;
            c.witness_pruning = witness;
            return cuda::DeviceEnv::fast_unsupported(*ref.suite, c);
#else
            (void)ref;
            (void)canonical;
            (void)witness;
            return "a CPU build (no CUDA backend)";
#endif
        },
        "table"_a, nb::kw_only(), "canonical"_a = true, "witness"_a = false,
        "Why the device fast path does not run the table ('' if it does).");

    m.def(
        "philox",
        [](ArrayArg counters, IntArg seed_in) {
            const u64 seed = int_arg<u64>(seed_in, "seed");
            Imports in(-1, dl::k_stream_default);
            std::vector<i64> ext;
            const u32* c = in.get<const u32>(counters, "counters", {u32t, i32t}, {-1, 4}, false, &ext);
            const u64 n = static_cast<u64>(ext[0]);
            auto block = Block::make(std::max<u64>(n, 1) * 4 * sizeof(u32));
            auto* out = reinterpret_cast<u32*>(block->data());
            for (u64 i = 0; i < n; ++i)
            {
                const rl::rng::Block4 r = rl::rng::philox4x32_10({{c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]}},
                                                                 static_cast<u32>(seed), static_cast<u32>(seed >> 32));
                std::copy_n(r.v, 4, out + 4 * i);
            }
            return ArrayOut(host_u32_array(std::move(block), {static_cast<i64>(n), 4}));
        },
        "counters"_a, "seed"_a,
        "Philox-4x32-10 blocks of host counters [n, 4] (uint32) under the key (seed low, seed high): NumPy uint32 "
        "[n, 4], equal to cuRAND's curand_Philox4x32_10.");

    m.def(
        "successor_indices",
        [](IntArg seed_in, ArrayArg envs, ArrayArg draws, ArrayArg counts) {
            const u64 seed = int_arg<u64>(seed_in, "seed");
            Imports in(-1, dl::k_stream_default);
            std::vector<i64> ext;
            const u64* e = in.get<const u64>(envs, "envs", {i64t, u64t}, {-1}, false, &ext);
            const i64 n = ext[0];
            const u64* d = in.get<const u64>(draws, "draws", {i64t, u64t}, {n}, false);
            const i32* c = in.get<const i32>(counts, "counts", {i32t}, {n}, false);
            auto block = Block::make(std::max<u64>(static_cast<u64>(n), 1) * sizeof(i64));
            auto* out = reinterpret_cast<i64*>(block->data());
            for (i64 i = 0; i < n; ++i)
                out[i] = c[i] <= 0 ? i64{-1} : i64{rl::rng::successor_index(seed, e[i], d[i], static_cast<u32>(c[i]))};
            ArraySpec s{block, out, rl::DType::I64, {n}, {}, false, false};
            return ArrayOut(export_array(std::move(s), Framework::Numpy, default_words(Framework::Numpy)));
        },
        "seed"_a, "envs"_a, "draws"_a, "counts"_a,
        "The random policy's choices (rl/rng.hpp): for env ids [n] (u)int64, draw counters [n] (u)int64 and successor "
        "counts [n] int32, the uniform index in [0, count) of each draw (-1 for count 0), as NumPy int64 [n]. "
        "Env.step draws index successor_indices(seed, first_env + i, draws[i], count[i]) and then advances draws[i].");
}
}  // namespace mymyr::python
