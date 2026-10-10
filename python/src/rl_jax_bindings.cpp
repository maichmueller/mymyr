// mymyr._core._rl_jax: the native side of mymyr.rl.jax.
//   Env(table, device=None | cuda index, ...)  a registered environment over a TaskSuite or a TaskTable (or a Task:
//                                               its table of one): `handle` is the FFI attribute of its calls
//   ffi_targets()                               the XLA FFI handlers as (name, platform, capsule), for
//                                               jax.ffi.register_ffi_target (empty without the XLA FFI headers)
//
// The FFI targets (typed XLA FFI, api version 4; CPU: rl::HostEnv and rl::expand, CUDA: cuda::DeviceEnv and
// cuda::SuiteExpander on XLA's stream). Every array is XLA's (destination passing): the handlers allocate no
// output, and a step on the device fast path enqueues its kernels on XLA's stream without synchronizing the host.
// State words are uint32 pairs (JAX has no uint64 without x64): a row of RW = words + numeric words u64 words is
// uint32[2 RW]; the draw counters and the per-row seeds are uint32[2] (low word first), the views uint32[2 V], the
// per-env goal masks uint32[2 W] (or [N, 0]: none, the instances' own goal tests). task_ids are int32 [N] (the rows'
// instances; not read over a table of one). The leading dimensions of every array are the environments (flattened:
// vmap's broadcast_all batching adds one), the last one the columns.
//   mymyr_env_init     (task_ids) -> states, steps, count, counts, views, goal_pos, goal_neg: every row at the
//                      initial state of its instance (goal masks: the instance's goal)
//   mymyr_env_reset    (states, task_ids, steps, count, counts, views, goal_pos, goal_neg, mask; attr keep_goals)
//                      -> states, steps, count, counts, views, goal_pos, goal_neg: the rows with mask[i] reset into
//                      instance task_ids[i] (goal masks: the instance's goal unless keep_goals)
//   mymyr_env_refresh  (states, task_ids, counts, views) -> count, counts, views: the cache and counts of written
//                      states
//   mymyr_env_step     (states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg, env_id, seed,
//                      action, next_task_ids)
//                      -> states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg, reward, terminated,
//                         truncated, goal, invalid, schema, binding, final_states
//                      action int32 [N] indexes the canonical successor order; an empty action array is the random
//                      policy (Philox of (seed[i], env_id[i], draws[i]), rl/rng.hpp); next_task_ids int32 [N] (empty:
//                      none) names the instances autoresetting rows restart in. binding [0, L] and final_states
//                      [0, 2 RW] are not written (the caller asked for neither)
//   mymyr_expand       (states, task_ids) -> succ [N, K, 2 RW], schema, binding [N, K, L], goal, mask [N, K], count [N]
//   mymyr_expand_flat  (states, task_ids) -> succ [M, 2 RW], parent, schema, binding [M, L], goal [M], offsets
//                      [N + 1]: the first M successors in canonical order; rows past the total are padding (succ 0,
//                      parent, schema and binding -1, goal false); offsets keep the true counts
// The in-place results (the environment state) alias their operands (ffi_call's input_output_aliases); a handler that
// gets distinct buffers copies the operand first. The cache columns decide the device path: [N, cache_schemas] and
// [N, 2 cache_view_words] take the fast path, empty ones the general path (which synchronizes on the successor count);
// on the CPU they must be empty.
//
// The handle registry: FFI attributes are scalars, so an Env enters its calls as an int64 handle into a process-wide
// table of the native environments (as mymyr.rl.torch's ops take handles). Handles are never reused; a call whose handle is
// gone (its Env was collected) fails with an error.

#include "py_table.hpp"
#include "py_task.hpp"
#include "rl_typing.hpp"
#include "typing.hpp"

#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"

#if defined(MYMYR_HAS_CUDA)
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#endif

#if defined(MYMYR_HAS_XLA_FFI)
#include "xla/ffi/api/ffi.h"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

#if defined(MYMYR_HAS_CUDA)
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device);  // cuda_bindings.cpp
#endif

namespace
{
using ContextArg = Arg<ann::Any>;

// ------------------------------------------------------------------------------------------------ registry

/// A registered environment: the suite (a table's: TaskSuite::of) and configuration, the host environment (CPU calls)
/// and, per CUDA device, the device environments and the expander (CUDA calls). Each backend serves one call at a time
/// (its scratch).
struct Entry
{
    rl::TaskSuitePtr suite;
    rl::EnvConfig cfg;
    std::string path = "auto";
    u32 W = 1, NN = 0, L = 1;
    std::mutex host_mutex;
    std::unique_ptr<rl::HostEnv> host;
    std::vector<u64> scratch_succ;
    std::vector<i32> scratch_schema, scratch_binding, scratch_offsets;
    std::vector<u8> scratch_goal;
#if defined(MYMYR_HAS_CUDA)
    struct Device
    {
        std::mutex mutex;
        cuda::ContextPtr ctx;
        std::unique_ptr<cuda::DeviceEnv> fast, general;
        std::unique_ptr<cuda::SuiteExpander> expander;
        cuda::Scratch succ, schema, binding, goal, offsets;
    };
    std::mutex devices_mutex;
    std::map<int, std::shared_ptr<Device>> devices;

    /// The device slot of CUDA device `ordinal` (created on first use, with the device's default context).
    std::shared_ptr<Device> device(int ordinal, cuda::ContextPtr ctx = nullptr)
    {
        std::lock_guard lock(devices_mutex);
        std::shared_ptr<Device>& d = devices[ordinal];
        if (!d)
        {
            d = std::make_shared<Device>();
            d->ctx = ctx ? std::move(ctx) : cuda::DeviceContext::create(ordinal);
        }
        return d;
    }
#endif
    [[nodiscard]] u32 row_words() const noexcept { return W + NN; }
    [[nodiscard]] bool multi() const noexcept { return suite->size() > 1; }
};
using EntryPtr = std::shared_ptr<Entry>;

class Registry
{
public:
    i64 add(EntryPtr e)
    {
        std::lock_guard lock(m_mutex);
        const i64 h = m_next++;
        m_entries.emplace(h, std::move(e));
        return h;
    }
    void remove(i64 h)
    {
        EntryPtr keep;  // destroyed after the lock (a device env synchronizes its stream)
        {
            std::lock_guard lock(m_mutex);
            auto it = m_entries.find(h);
            if (it == m_entries.end())
                return;
            keep = std::move(it->second);
            m_entries.erase(it);
        }
    }
    [[nodiscard]] EntryPtr find(i64 h)
    {
        std::lock_guard lock(m_mutex);
        auto it = m_entries.find(h);
        return it == m_entries.end() ? nullptr : it->second;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<i64, EntryPtr> m_entries;
    i64 m_next = 1;
};

Registry& registry()
{
    static Registry* r = new Registry();  // never destroyed: XLA may call a handler during interpreter shutdown
    return *r;
}

// ------------------------------------------------------------------------------------------------ Env (Python)

class PyJaxEnv
{
public:
    PyJaxEnv(SuiteArg table, nb::object device, ContextArg ctx, const rl::EnvConfig& cfg, const std::string& path)
        : m_ref(suite_of(table))
    {
        if (path != "auto" && path != "fast" && path != "general")
            throw nb::value_error("mymyr: path must be 'auto', 'fast' or 'general'");
        auto e = std::make_shared<Entry>();
        e->suite = m_ref.suite;
        e->cfg = cfg;
        e->path = path;
        e->NN = e->suite->numeric_words();
        {
            nb::gil_scoped_release release;
            e->host = std::make_unique<rl::HostEnv>(e->suite, cfg);
        }
        e->W = e->host->words();
        e->L = std::max<u32>(1, e->suite->label_width());
        for (u32 i = 0; i < e->suite->size(); ++i)
            m_init_count.push_back(e->host->initial_count(i));
        if (!device.is_none())
        {
            m_device = int_arg<int>(device, "device", 0);
            if (m_device < 0)
                throw nb::value_error("mymyr: device must be a CUDA device index (>= 0) or None (the CPU)");
#if defined(MYMYR_HAS_CUDA)
            cuda::ContextPtr c = table_device_context(m_ref.obj, ctx, m_device);
            nb::gil_scoped_release release;
            auto d = e->device(m_device, std::move(c));
            std::lock_guard lock(d->mutex);
            if (path == "general")
                d->general = std::make_unique<cuda::DeviceEnv>(d->ctx, e->suite, cfg, cuda::DeviceEnv::Path::General);
            else
            {
                const auto mode = path == "fast" ? cuda::DeviceEnv::Path::Fast : cuda::DeviceEnv::Path::Auto;
                auto env = std::make_unique<cuda::DeviceEnv>(d->ctx, e->suite, cfg, mode);
                (env->fast() ? d->fast : d->general) = std::move(env);
            }
            if (d->fast)
            {
                m_fast = true;
                m_C = d->fast->cache_schemas();
                m_V = d->fast->cache_view_words();
            }
#else
            (void)ctx;
            throw nb::value_error("mymyr: this mymyr was built without the CUDA backend; use device=None (the CPU)");
#endif
        }
        else if (!ctx.is_none())
            throw nb::value_error("mymyr: ctx is for device environments (device=None runs on the CPU)");
        else if (path == "fast")
            throw nb::value_error("mymyr: path='fast' is a device path (device=None runs on the CPU)");
        m_entry = e;
        m_handle = registry().add(std::move(e));
    }
    ~PyJaxEnv() { registry().remove(m_handle); }
    PyJaxEnv(const PyJaxEnv&) = delete;
    PyJaxEnv& operator=(const PyJaxEnv&) = delete;

    [[nodiscard]] i64 handle() const noexcept { return m_handle; }
    [[nodiscard]] std::optional<int> device() const
    {
        return m_device < 0 ? std::nullopt : std::optional<int>(m_device);
    }
    [[nodiscard]] const Entry& entry() const noexcept { return *m_entry; }
    [[nodiscard]] const SuiteRef& ref() const noexcept { return m_ref; }
    [[nodiscard]] bool fast() const noexcept { return m_fast; }
    [[nodiscard]] u32 cache_schemas() const noexcept { return m_C; }
    [[nodiscard]] u64 cache_view_words() const noexcept { return m_V; }
    [[nodiscard]] const std::vector<u32>& initial_counts() const noexcept { return m_init_count; }

    /// The instances' initial states as uint32 halves [I, 2 row_words] (low word first), row-major.
    [[nodiscard]] std::vector<u32> initial_states() const
    {
        const rl::TaskSuite& t = *m_entry->suite;
        const u32 RW = m_entry->row_words();
        std::vector<u32> out(static_cast<usize>(t.size()) * 2 * RW, 0);
        std::vector<u64> r(RW);
        for (u32 i = 0; i < t.size(); ++i)
        {
            t.initial_row(i, r.data(), m_entry->W, m_entry->NN);
            for (usize k = 0; k < RW; ++k)
            {
                out[(usize{i} * RW + k) * 2] = static_cast<u32>(r[k]);
                out[(usize{i} * RW + k) * 2 + 1] = static_cast<u32>(r[k] >> 32);
            }
        }
        return out;
    }

    void set_launch(const std::string& launch)
    {
        if (launch != "widest" && launch != "per_bucket")
            throw nb::value_error("mymyr: launch must be 'widest' or 'per_bucket'");
#if defined(MYMYR_HAS_CUDA)
        const auto l = launch == "widest" ? cuda::BucketLaunch::Widest : cuda::BucketLaunch::PerBucket;
        std::vector<std::shared_ptr<Entry::Device>> ds;
        {
            std::lock_guard lock(m_entry->devices_mutex);
            for (auto& [k, d] : m_entry->devices)
                ds.push_back(d);
        }
        for (auto& d : ds)
        {
            std::lock_guard lock(d->mutex);
            for (cuda::DeviceEnv* env : {d->fast.get(), d->general.get()})
                if (env)
                    env->set_launch(l);
            if (d->expander)
                d->expander->set_launch(l);
        }
#endif
    }

    /// Waits for the device work of this Env's calls and raises if a fast-path step met a stale cache.
    void check_errors()
    {
#if defined(MYMYR_HAS_CUDA)
        std::vector<std::shared_ptr<Entry::Device>> ds;
        {
            std::lock_guard lock(m_entry->devices_mutex);
            for (auto& [k, d] : m_entry->devices)
                ds.push_back(d);
        }
        nb::gil_scoped_release release;
        for (auto& d : ds)
        {
            std::lock_guard lock(d->mutex);
            for (cuda::DeviceEnv* env : {d->fast.get(), d->general.get()})
                if (env)
                    env->check_errors();
        }
#endif
    }

private:
    SuiteRef m_ref;
    EntryPtr m_entry;
    i64 m_handle = 0;
    int m_device = -1;
    bool m_fast = false;
    u32 m_C = 0;
    u64 m_V = 0;
    std::vector<u32> m_init_count;
};

// ------------------------------------------------------------------------------------------------ FFI handlers

#if defined(MYMYR_HAS_XLA_FFI)
namespace ffi = xla::ffi;

/// A handler's failure (an ffi::Error with the message).
struct Failure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& msg) { throw Failure("mymyr: " + msg); }

const char* dtype_name(ffi::DataType t)
{
    switch (t)
    {
        case ffi::DataType::PRED: return "bool";
        case ffi::DataType::S32: return "int32";
        case ffi::DataType::U32: return "uint32";
        case ffi::DataType::F32: return "float32";
        case ffi::DataType::S64: return "int64";
        case ffi::DataType::U64: return "uint64";
        default: return "another dtype";
    }
}

/// A buffer as rows (the product of the leading dimensions) x columns (the last dimension; 1 for a vector).
struct View
{
    void* data = nullptr;
    u64 rows = 0, cols = 1;
    u64 count = 0;  // elements

    template<class T>
    [[nodiscard]] T* as() const noexcept
    {
        return count ? static_cast<T*>(data) : nullptr;
    }
};

/// `buf` checked against a dtype and a shape: `vector` buffers have rows elements (any rank; all dimensions are
/// environments), others [..., cols] with rows = the product of the leading dimensions (cols = -1: any).
View view(const ffi::AnyBuffer& buf, const char* name, ffi::DataType dtype, bool vector, i64 cols = -1)
{
    if (buf.element_type() != dtype)
        fail(std::string("'") + name + "' must be " + dtype_name(dtype) + ", not " + dtype_name(buf.element_type()));
    const auto dims = buf.dimensions();
    View v;
    v.data = buf.untyped_data();
    v.count = buf.element_count();
    if (vector)
    {
        v.rows = v.count;
        return v;
    }
    if (dims.size() == 0)
        fail(std::string("'") + name + "' must have at least one dimension");
    v.cols = static_cast<u64>(dims.back());
    v.rows = v.cols ? v.count / v.cols : 1;
    if (!v.cols)
    {
        v.rows = 1;
        for (usize i = 0; i + 1 < dims.size(); ++i)
            v.rows *= static_cast<u64>(dims[i]);
    }
    if (cols >= 0 && v.cols != static_cast<u64>(cols))
        fail(std::string("'") + name + "' must have " + std::to_string(cols) + " columns, not " + std::to_string(v.cols));
    if (v.count && reinterpret_cast<std::uintptr_t>(v.data) % 8 != 0)
        fail(std::string("'") + name + "' is not 8-byte aligned");
    return v;
}

void expect_rows(const View& v, u64 rows, const char* name)
{
    if (v.rows != rows)
        fail(std::string("'") + name + "' has " + std::to_string(v.rows) + " rows; the states have " + std::to_string(rows));
}

EntryPtr entry(i64 handle)
{
    EntryPtr e = registry().find(handle);
    if (!e)
        fail("no environment is registered under handle " + std::to_string(handle) +
             " (its mymyr.rl.jax.Env was collected; keep the Env alive while its functions run)");
    return e;
}

/// Runs `f` and turns exceptions into an ffi::Error.
template<class F>
ffi::Error guarded(F&& f)
{
    try
    {
        f();
        return ffi::Error::Success();
    }
    catch (const Failure& x)
    {
        return ffi::Error::InvalidArgument(x.what());
    }
    catch (const std::invalid_argument& x)
    {
        return ffi::Error::InvalidArgument(x.what());
    }
    catch (const std::exception& x)
    {
        return ffi::Error::Internal(std::string(x.what()));
    }
}

/// The platform of a call: how in-place operands reach their results, and where the environment runs.
struct Host
{
    static void copy(void* dst, const void* src, u64 bytes)
    {
        if (dst != src && bytes)
            std::memmove(dst, src, bytes);
    }
};

/// The task ids of a call: int32 [N] (read over tables of several instances; null for a table of one).
i32* task_ids_of(const Entry& e, const View& v, u64 N, const char* name = "task_ids")
{
    expect_rows(v, N, name);
    return e.multi() ? v.as<i32>() : nullptr;
}

/// The environment state of a call (the in-place arrays after their operands reached them).
struct StateBufs
{
    View states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg;
};

/// Which in-place arrays a call has, in this order: states, [task_ids], [steps], [draws], count, counts, views,
/// goal_pos, goal_neg.
struct Fields
{
    bool task_ids = false, steps = true, draws = false;
};

rl::EnvBatch make_batch(const Entry& e, const StateBufs& s, u64 N, i32* task_ids)
{
    rl::EnvBatch b;
    b.states = s.states.as<u64>();
    b.rows = N;
    b.words = e.W;
    b.numeric_words = e.NN;
    b.task_ids = task_ids;
    b.steps = s.steps.as<i32>();
    b.draws = s.draws.as<u64>();
    b.counts = s.counts.cols ? s.counts.as<u32>() : nullptr;
    b.views = s.views.cols ? s.views.as<u64>() : nullptr;
    b.goal_pos = s.goal_pos.cols ? s.goal_pos.as<u64>() : nullptr;
    b.goal_neg = s.goal_neg.cols ? s.goal_neg.as<u64>() : nullptr;
    return b;
}

/// Views of the state arrays (results; `args`: their operands, copied into them, or null: no operands).
template<class Copy>
StateBufs state_results(const Entry& e, ffi::Result<ffi::AnyBuffer>* res, const ffi::AnyBuffer* args, Fields f, Copy copy)
{
    StateBufs s;
    const u32 RW2 = 2 * e.row_words();
    usize k = 0;
    auto next = [&](const char* name, ffi::DataType t, bool vec, i64 cols) -> View
    {
        View v = view(*res[k], name, t, vec, cols);
        if (args)
        {
            const View a = view(args[k], name, t, vec, cols);
            if (a.count != v.count)
                fail(std::string("'") + name + "': operand and result sizes differ");
            copy(v.data, a.data, a.count * (t == ffi::DataType::PRED ? 1 : 4));
        }
        ++k;
        return v;
    };
    s.states = next("states", ffi::DataType::U32, false, RW2);
    if (f.task_ids)
        s.task_ids = next("task_ids", ffi::DataType::S32, true, -1);
    if (f.steps)
        s.steps = next("steps", ffi::DataType::S32, true, -1);
    if (f.draws)
        s.draws = next("draws", ffi::DataType::U32, false, 2);
    s.count = next("count", ffi::DataType::S32, true, -1);
    s.counts = next("counts", ffi::DataType::U32, false, -1);
    s.views = next("views", ffi::DataType::U32, false, -1);
    s.goal_pos = next("goal_pos", ffi::DataType::U32, false, -1);
    s.goal_neg = next("goal_neg", ffi::DataType::U32, false, -1);
    const u64 N = s.states.rows;
    if (f.task_ids)
        expect_rows(s.task_ids, N, "task_ids");
    if (f.steps)
        expect_rows(s.steps, N, "steps");
    if (f.draws)
        expect_rows(s.draws, N, "draws");
    expect_rows(s.count, N, "count");
    if (s.counts.cols)
        expect_rows(s.counts, N, "counts");
    if (s.views.cols)
        expect_rows(s.views, N, "views");
    if (s.views.cols % 2)
        fail("'views' must have an even number of uint32 columns (u64 words)");
    if (s.goal_pos.cols != s.goal_neg.cols)
        fail("'goal_pos' and 'goal_neg' must have the same columns (2 words, or 0: no per-env goals)");
    if (s.goal_pos.cols && s.goal_pos.cols != 2 * static_cast<u64>(e.W))
        fail("the per-env goal masks must have " + std::to_string(2 * e.W) + " uint32 columns (or 0: none)");
    if (s.goal_pos.cols)
    {
        expect_rows(s.goal_pos, N, "goal_pos");
        expect_rows(s.goal_neg, N, "goal_neg");
    }
    return s;
}

void require_host_cache(const StateBufs& s)
{
    if (s.counts.cols || s.views.cols)
        fail("the CPU environment keeps no cache: 'counts' and 'views' must have 0 columns on the CPU (a state made "
             "for a CUDA device's fast path runs there, not here)");
}

// ---- CPU

ffi::Error env_init_cpu(i64 handle, ffi::AnyBuffer task_ids, ffi::Result<ffi::AnyBuffer> states,
                        ffi::Result<ffi::AnyBuffer> steps, ffi::Result<ffi::AnyBuffer> count,
                        ffi::Result<ffi::AnyBuffer> counts, ffi::Result<ffi::AnyBuffer> views,
                        ffi::Result<ffi::AnyBuffer> goal_pos, ffi::Result<ffi::AnyBuffer> goal_neg)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        ffi::Result<ffi::AnyBuffer> res[] = {states, steps, count, counts, views, goal_pos, goal_neg};
        const StateBufs s = state_results(*e, res, nullptr, Fields{}, Host::copy);
        require_host_cache(s);
        const u64 N = s.states.rows;
        rl::EnvBatch b = make_batch(*e, s, N, task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), N));
        std::lock_guard lock(e->host_mutex);
        e->host->reset(b, nullptr, s.count.as<i32>());
    });
}

ffi::Error env_reset_cpu(i64 handle, bool keep_goals, ffi::AnyBuffer states_in, ffi::AnyBuffer task_ids,
                         ffi::AnyBuffer steps_in, ffi::AnyBuffer count_in, ffi::AnyBuffer counts_in,
                         ffi::AnyBuffer views_in, ffi::AnyBuffer goal_pos_in, ffi::AnyBuffer goal_neg_in,
                         ffi::AnyBuffer mask, ffi::Result<ffi::AnyBuffer> states, ffi::Result<ffi::AnyBuffer> steps,
                         ffi::Result<ffi::AnyBuffer> count, ffi::Result<ffi::AnyBuffer> counts,
                         ffi::Result<ffi::AnyBuffer> views, ffi::Result<ffi::AnyBuffer> goal_pos,
                         ffi::Result<ffi::AnyBuffer> goal_neg)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        ffi::Result<ffi::AnyBuffer> res[] = {states, steps, count, counts, views, goal_pos, goal_neg};
        const ffi::AnyBuffer args[] = {states_in, steps_in, count_in, counts_in, views_in, goal_pos_in, goal_neg_in};
        const StateBufs s = state_results(*e, res, args, Fields{}, Host::copy);
        require_host_cache(s);
        const u64 N = s.states.rows;
        const View m = view(mask, "mask", ffi::DataType::PRED, true);
        expect_rows(m, N, "mask");
        rl::EnvBatch b = make_batch(*e, s, N, task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), N));
        std::lock_guard lock(e->host_mutex);
        e->host->reset(b, m.as<u8>(), s.count.as<i32>(), keep_goals);
    });
}

ffi::Error env_refresh_cpu(i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids, ffi::AnyBuffer counts_in,
                           ffi::AnyBuffer views_in, ffi::Result<ffi::AnyBuffer> count,
                           ffi::Result<ffi::AnyBuffer> counts, ffi::Result<ffi::AnyBuffer> views)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const View st = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        const View c = view(*count, "count", ffi::DataType::S32, true);
        expect_rows(c, st.rows, "count");
        if (view(*counts, "counts", ffi::DataType::U32, false).cols || view(*views, "views", ffi::DataType::U32, false).cols)
            fail("the CPU environment keeps no cache: 'counts' and 'views' must have 0 columns on the CPU");
        (void)counts_in;
        (void)views_in;
        rl::EnvBatch b;
        b.states = st.as<u64>();
        b.rows = st.rows;
        b.words = e->W;
        b.numeric_words = e->NN;
        b.task_ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), st.rows);
        std::lock_guard lock(e->host_mutex);
        if (b.rows)
        {
            e->suite->check_task_ids(b.task_ids, b.rows);
            e->host->count(b, c.as<i32>());
        }
    });
}

/// The outputs of a step, checked against N rows.
struct StepBufs
{
    View reward, terminated, truncated, goal, invalid, schema, binding, final_states;
};

StepBufs step_results(const Entry& e, u64 N, ffi::Result<ffi::AnyBuffer> reward, ffi::Result<ffi::AnyBuffer> terminated,
                      ffi::Result<ffi::AnyBuffer> truncated, ffi::Result<ffi::AnyBuffer> goal,
                      ffi::Result<ffi::AnyBuffer> invalid, ffi::Result<ffi::AnyBuffer> schema,
                      ffi::Result<ffi::AnyBuffer> binding, ffi::Result<ffi::AnyBuffer> final_states)
{
    StepBufs o;
    o.reward = view(*reward, "reward", ffi::DataType::F32, true);
    o.terminated = view(*terminated, "terminated", ffi::DataType::PRED, true);
    o.truncated = view(*truncated, "truncated", ffi::DataType::PRED, true);
    o.goal = view(*goal, "goal", ffi::DataType::PRED, true);
    o.invalid = view(*invalid, "invalid", ffi::DataType::PRED, true);
    o.schema = view(*schema, "schema", ffi::DataType::S32, true);
    o.binding = view(*binding, "binding", ffi::DataType::S32, false);
    o.final_states = view(*final_states, "final_states", ffi::DataType::U32, false, 2 * e.row_words());
    for (auto [v, name] : {std::pair{&o.reward, "reward"}, {&o.terminated, "terminated"}, {&o.truncated, "truncated"},
                           {&o.goal, "goal"}, {&o.invalid, "invalid"}, {&o.schema, "schema"}})
        expect_rows(*v, N, name);
    if (o.binding.count && o.binding.rows != N)
        expect_rows(o.binding, N, "binding");
    if (o.binding.count && o.binding.cols < e.L)
        fail("'binding' needs at least " + std::to_string(e.L) + " columns (the largest schema arity)");
    if (o.final_states.count)
        expect_rows(o.final_states, N, "final_states");
    return o;
}

rl::StepOutputs make_outputs(const StepBufs& o)
{
    rl::StepOutputs out;
    out.reward = o.reward.as<f32>();
    out.terminated = o.terminated.as<u8>();
    out.truncated = o.truncated.as<u8>();
    out.goal = o.goal.as<u8>();
    out.invalid = o.invalid.as<u8>();
    out.count = nullptr;  // set by the caller (the state's count)
    out.schema = o.schema.as<i32>();
    out.binding = o.binding.as<i32>();
    out.label_width = out.binding ? static_cast<u32>(o.binding.cols) : 0;
    out.final_states = o.final_states.as<u64>();
    return out;
}

/// The per-row inputs of a step: env ids, seeds, the action (empty: the random policy) and the next task ids (empty:
/// none).
struct StepInputs
{
    View env_id, seed, action, next;
};

StepInputs step_inputs(u64 N, const ffi::AnyBuffer& env_id, const ffi::AnyBuffer& seed, const ffi::AnyBuffer& action,
                       const ffi::AnyBuffer& next_task_ids)
{
    StepInputs in;
    in.env_id = view(env_id, "env_id", ffi::DataType::U32, true);
    in.seed = view(seed, "seed", ffi::DataType::U32, false, 2);
    in.action = view(action, "action", ffi::DataType::S32, true);
    in.next = view(next_task_ids, "next_task_ids", ffi::DataType::S32, true);
    expect_rows(in.env_id, N, "env_id");
    expect_rows(in.seed, N, "seed");
    if (in.action.count)
        expect_rows(in.action, N, "action");
    if (in.next.count)
        expect_rows(in.next, N, "next_task_ids");
    return in;
}

#define MYMYR_STEP_PARAMS                                                                                               \
    ffi::AnyBuffer states_in, ffi::AnyBuffer task_ids_in, ffi::AnyBuffer steps_in, ffi::AnyBuffer draws_in,             \
        ffi::AnyBuffer count_in, ffi::AnyBuffer counts_in, ffi::AnyBuffer views_in, ffi::AnyBuffer goal_pos_in,         \
        ffi::AnyBuffer goal_neg_in, ffi::AnyBuffer env_id, ffi::AnyBuffer seed, ffi::AnyBuffer action,                  \
        ffi::AnyBuffer next_task_ids, ffi::Result<ffi::AnyBuffer> states, ffi::Result<ffi::AnyBuffer> task_ids,         \
        ffi::Result<ffi::AnyBuffer> steps, ffi::Result<ffi::AnyBuffer> draws, ffi::Result<ffi::AnyBuffer> count,        \
        ffi::Result<ffi::AnyBuffer> counts, ffi::Result<ffi::AnyBuffer> views, ffi::Result<ffi::AnyBuffer> goal_pos,    \
        ffi::Result<ffi::AnyBuffer> goal_neg, ffi::Result<ffi::AnyBuffer> reward,                                       \
        ffi::Result<ffi::AnyBuffer> terminated, ffi::Result<ffi::AnyBuffer> truncated, ffi::Result<ffi::AnyBuffer> goal, \
        ffi::Result<ffi::AnyBuffer> invalid, ffi::Result<ffi::AnyBuffer> schema, ffi::Result<ffi::AnyBuffer> binding,   \
        ffi::Result<ffi::AnyBuffer> final_states

/// The state arrays of a step (results with their operands copied in), its inputs and its outputs.
struct StepCall
{
    StateBufs s;
    StepInputs in;
    StepBufs o;
    u64 N = 0;
};

template<class Copy>
StepCall step_call(const Entry& e, MYMYR_STEP_PARAMS, Copy copy)
{
    StepCall c;
    ffi::Result<ffi::AnyBuffer> res[] = {states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg};
    const ffi::AnyBuffer args[] = {states_in, task_ids_in, steps_in, draws_in, count_in, counts_in, views_in, goal_pos_in,
                                   goal_neg_in};
    c.s = state_results(e, res, args, Fields{true, true, true}, copy);
    c.N = c.s.states.rows;
    c.in = step_inputs(c.N, env_id, seed, action, next_task_ids);
    c.o = step_results(e, c.N, reward, terminated, truncated, goal, invalid, schema, binding, final_states);
    return c;
}

ffi::Error env_step_cpu(i64 handle, MYMYR_STEP_PARAMS)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const StepCall c = step_call(*e, states_in, task_ids_in, steps_in, draws_in, count_in, counts_in, views_in,
                                     goal_pos_in, goal_neg_in, env_id, seed, action, next_task_ids, states, task_ids,
                                     steps, draws, count, counts, views, goal_pos, goal_neg, reward, terminated,
                                     truncated, goal, invalid, schema, binding, final_states, Host::copy);
        require_host_cache(c.s);
        rl::EnvBatch b = make_batch(*e, c.s, c.N, e->multi() ? c.s.task_ids.as<i32>() : nullptr);
        b.seeds = c.in.seed.as<u64>();
        b.env_ids = c.in.env_id.as<u32>();
        rl::StepOutputs out = make_outputs(c.o);
        out.count = c.s.count.as<i32>();
        std::lock_guard lock(e->host_mutex);
        e->host->step(b, out, rl::Actions{nullptr, c.in.action.as<i32>()}, e->multi() ? c.in.next.as<i32>() : nullptr);
    });
}

/// rl::expand of the states into the entry's host scratch (all successors): the offsets and the total.
u64 host_expand(Entry& e, const View& st, const i32* ids)
{
    const rl::StateBatchView in{st.as<u64>(), st.rows, e.W, 0, e.NN};
    rl::ExpandOptions opt;
    opt.canonical_order = e.cfg.canonical_order;
    opt.witness_pruning = e.cfg.witness_pruning;
    e.scratch_offsets.assign(st.rows + 1, 0);
    rl::Expansion x;
    x.words = e.W;
    x.numeric_words = e.NN;
    x.label_width = e.L;
    x.offsets = e.scratch_offsets.data();
    rl::expand(*e.suite, in, ids, x, opt);  // counts only
    const u64 total = x.total;
    e.scratch_succ.resize(std::max<u64>(total, 1) * e.row_words());
    e.scratch_schema.resize(std::max<u64>(total, 1));
    e.scratch_binding.resize(std::max<u64>(total, 1) * e.L);
    e.scratch_goal.resize(std::max<u64>(total, 1));
    x.capacity = total;
    x.succ = e.scratch_succ.data();
    x.schema = e.scratch_schema.data();
    x.binding = e.scratch_binding.data();
    x.goal = e.scratch_goal.data();
    rl::expand(*e.suite, in, ids, x, opt);
    if (x.words_needed > e.W)
        fail("a successor needs " + std::to_string(x.words_needed) + " words; the rows have " + std::to_string(e.W));
    return total;
}

/// The padded results [N, K]: succ [N, K, 2 RW], schema, binding [N, K, L'], goal, mask, count.
struct PaddedBufs
{
    View succ, schema, binding, goal, mask, count;
    u32 K = 0;
};

PaddedBufs padded_results(const Entry& e, u64 N, ffi::Result<ffi::AnyBuffer> succ, ffi::Result<ffi::AnyBuffer> schema,
                          ffi::Result<ffi::AnyBuffer> binding, ffi::Result<ffi::AnyBuffer> goal,
                          ffi::Result<ffi::AnyBuffer> mask, ffi::Result<ffi::AnyBuffer> count)
{
    PaddedBufs p;
    p.succ = view(*succ, "succ", ffi::DataType::U32, false, 2 * e.row_words());
    p.schema = view(*schema, "schema", ffi::DataType::S32, false);
    p.binding = view(*binding, "binding", ffi::DataType::S32, false);
    p.goal = view(*goal, "goal", ffi::DataType::PRED, false);
    p.mask = view(*mask, "mask", ffi::DataType::PRED, false);
    p.count = view(*count, "count", ffi::DataType::S32, true);
    p.K = static_cast<u32>(p.schema.cols);
    expect_rows(p.schema, N, "schema");
    expect_rows(p.count, N, "count");
    if (p.succ.rows != N * p.K || p.goal.count != N * p.K || p.mask.count != N * p.K || p.binding.rows != N * p.K)
        fail("the padded results must be [N, K, ...] with the same K");
    if (p.binding.cols < e.L)
        fail("'binding' needs at least " + std::to_string(e.L) + " columns (the largest schema arity)");
    return p;
}

rl::PaddedExpansion make_padded(const Entry& e, const PaddedBufs& p)
{
    rl::PaddedExpansion out;
    out.K = p.K;
    out.mask = p.mask.as<u8>();
    out.count = p.count.as<i32>();
    out.words = e.W;
    out.numeric_words = e.NN;
    out.succ = p.succ.as<u64>();
    out.schema = p.schema.as<i32>();
    out.label_width = static_cast<u32>(p.binding.cols);
    out.binding = p.binding.as<i32>();
    out.goal = p.goal.as<u8>();
    return out;
}

ffi::Error expand_cpu(i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids, ffi::Result<ffi::AnyBuffer> succ,
                      ffi::Result<ffi::AnyBuffer> schema, ffi::Result<ffi::AnyBuffer> binding,
                      ffi::Result<ffi::AnyBuffer> goal, ffi::Result<ffi::AnyBuffer> mask,
                      ffi::Result<ffi::AnyBuffer> count)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const View st = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        const i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), st.rows);
        const PaddedBufs p = padded_results(*e, st.rows, succ, schema, binding, goal, mask, count);
        std::lock_guard lock(e->host_mutex);
        const u64 total = host_expand(*e, st, ids);
        rl::Expansion x;
        x.capacity = total;
        x.words = e->W;
        x.numeric_words = e->NN;
        x.label_width = e->L;
        x.succ = e->scratch_succ.data();
        x.schema = e->scratch_schema.data();
        x.binding = e->scratch_binding.data();
        x.goal = e->scratch_goal.data();
        x.offsets = e->scratch_offsets.data();
        x.total = total;
        rl::PaddedExpansion out = make_padded(*e, p);
        rl::pad(x, st.rows, out);
    });
}

/// The flat results with capacity M: succ [M, 2 RW], parent, schema, binding [M, L'], goal [M], offsets [N + 1].
struct FlatBufs
{
    View succ, parent, schema, binding, goal, offsets;
    u64 M = 0;
};

FlatBufs flat_results(const Entry& e, u64 N, ffi::Result<ffi::AnyBuffer> succ, ffi::Result<ffi::AnyBuffer> parent,
                      ffi::Result<ffi::AnyBuffer> schema, ffi::Result<ffi::AnyBuffer> binding,
                      ffi::Result<ffi::AnyBuffer> goal, ffi::Result<ffi::AnyBuffer> offsets)
{
    FlatBufs f;
    f.succ = view(*succ, "succ", ffi::DataType::U32, false, 2 * e.row_words());
    f.parent = view(*parent, "parent", ffi::DataType::S32, true);
    f.schema = view(*schema, "schema", ffi::DataType::S32, true);
    f.binding = view(*binding, "binding", ffi::DataType::S32, false);
    f.goal = view(*goal, "goal", ffi::DataType::PRED, true);
    f.offsets = view(*offsets, "offsets", ffi::DataType::S32, true);
    f.M = f.parent.rows;
    if (f.succ.rows != f.M || f.schema.rows != f.M || f.goal.rows != f.M || f.binding.rows != f.M)
        fail("the flat results must have the same capacity M");
    if (f.offsets.rows != N + 1)
        fail("'offsets' must have N + 1 entries");
    if (f.binding.cols < e.L)
        fail("'binding' needs at least " + std::to_string(e.L) + " columns (the largest schema arity)");
    return f;
}

rl::Expansion make_flat(const Entry& e, const FlatBufs& f)
{
    rl::Expansion x;
    x.capacity = f.M;
    x.words = e.W;
    x.numeric_words = e.NN;
    x.label_width = static_cast<u32>(f.binding.cols);
    x.succ = f.succ.as<u64>();
    x.parent = f.parent.as<i32>();
    x.schema = f.schema.as<i32>();
    x.binding = f.binding.as<i32>();
    x.goal = f.goal.as<u8>();
    x.offsets = f.offsets.as<i32>();
    return x;
}

ffi::Error expand_flat_cpu(i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids, ffi::Result<ffi::AnyBuffer> succ,
                           ffi::Result<ffi::AnyBuffer> parent, ffi::Result<ffi::AnyBuffer> schema,
                           ffi::Result<ffi::AnyBuffer> binding, ffi::Result<ffi::AnyBuffer> goal,
                           ffi::Result<ffi::AnyBuffer> offsets)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const View st = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        const i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), st.rows);
        const FlatBufs f = flat_results(*e, st.rows, succ, parent, schema, binding, goal, offsets);
        // padding first: rows past the total keep it
        if (f.succ.count)
            std::memset(f.succ.data, 0, f.succ.count * 4);
        for (const View* v : {&f.parent, &f.schema, &f.binding})
            if (v->count)
                std::memset(v->data, 0xFF, v->count * 4);
        if (f.goal.count)
            std::memset(f.goal.data, 0, f.goal.count);
        rl::Expansion x = make_flat(*e, f);
        rl::ExpandOptions opt;
        opt.canonical_order = e->cfg.canonical_order;
        opt.witness_pruning = e->cfg.witness_pruning;
        rl::expand(*e->suite, rl::StateBatchView{st.as<u64>(), st.rows, e->W, 0, e->NN}, ids, x, opt);
    });
}

#define MYMYR_RETS7                                                                                                     \
    .Ret<ffi::AnyBuffer>()                                                                                              \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()
#define MYMYR_INIT_BINDING .Arg<ffi::AnyBuffer>() MYMYR_RETS7
#define MYMYR_RESET_BINDING                                                                                             \
    .Attr<bool>("keep_goals")                                                                                           \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>() MYMYR_RETS7
#define MYMYR_REFRESH_BINDING                                                                                           \
    .Arg<ffi::AnyBuffer>()                                                                                              \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()
#define MYMYR_STEP_BINDING                                                                                              \
    .Arg<ffi::AnyBuffer>()                                                                                              \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()
#define MYMYR_EXPAND_BINDING                                                                                            \
    .Arg<ffi::AnyBuffer>()                                                                                              \
        .Arg<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()                                                                                          \
        .Ret<ffi::AnyBuffer>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_init_cpu, env_init_cpu, ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_INIT_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_reset_cpu, env_reset_cpu,
                              ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_RESET_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_refresh_cpu, env_refresh_cpu,
                              ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_REFRESH_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_step_cpu, env_step_cpu, ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_STEP_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_expand_cpu, expand_cpu, ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_EXPAND_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_expand_flat_cpu, expand_flat_cpu,
                              ffi::Ffi::Bind().Attr<i64>("handle") MYMYR_EXPAND_BINDING);

// ---- CUDA

#if defined(MYMYR_HAS_CUDA)
/// Device calls: in-place operands reach their results by a stream-ordered copy.
struct OnStream
{
    cudaStream_t s;
    void operator()(void* dst, const void* src, u64 bytes) const
    {
        if (dst != src && bytes)
            cuda::check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync");
    }
};

/// The device environment of a call: the fast path for the fast path's cache columns, the general path for none.
cuda::DeviceEnv& device_env(Entry& e, Entry::Device& d, const StateBufs& s)
{
    if (!s.counts.cols && !s.views.cols)
    {
        if (!d.general)
            d.general = std::make_unique<cuda::DeviceEnv>(d.ctx, e.suite, e.cfg, cuda::DeviceEnv::Path::General);
        return *d.general;
    }
    if (!d.fast)
    {
        if (e.path == "general")
            fail("the environment runs the general path (path='general'): 'counts' and 'views' must have 0 columns");
        d.fast = std::make_unique<cuda::DeviceEnv>(d.ctx, e.suite, e.cfg, cuda::DeviceEnv::Path::Fast);
    }
    if (s.counts.cols != d.fast->cache_schemas() || s.views.cols != 2 * d.fast->cache_view_words())
        fail("the cache arrays of the fast path are counts [N, " + std::to_string(d.fast->cache_schemas()) +
             "] and views [N, " + std::to_string(2 * d.fast->cache_view_words()) + "] (uint32), or both [N, 0] for the "
             "general path");
    return *d.fast;
}

ffi::Error env_init_cuda(cudaStream_t st, i32 ordinal, i64 handle, ffi::AnyBuffer task_ids,
                         ffi::Result<ffi::AnyBuffer> states, ffi::Result<ffi::AnyBuffer> steps,
                         ffi::Result<ffi::AnyBuffer> count, ffi::Result<ffi::AnyBuffer> counts,
                         ffi::Result<ffi::AnyBuffer> views, ffi::Result<ffi::AnyBuffer> goal_pos,
                         ffi::Result<ffi::AnyBuffer> goal_neg)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        ffi::Result<ffi::AnyBuffer> res[] = {states, steps, count, counts, views, goal_pos, goal_neg};
        const StateBufs s = state_results(*e, res, nullptr, Fields{}, OnStream{st});
        const u64 N = s.states.rows;
        i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), N);
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceEnv& env = device_env(*e, *d, s);
        env.set_stream(st);
        // the env writes a row's own instance's views (V_i of the widest V words): zero the rest, so that every byte
        // of the state pytree is defined (XLA's output buffers are not initialized)
        if (views->size_bytes())
            cuda::check(cudaMemsetAsync(views->untyped_data(), 0, views->size_bytes(), st), "cudaMemsetAsync (views)");
        rl::EnvBatch b = make_batch(*e, s, N, ids);
        env.reset(b, nullptr, s.count.as<i32>());
    });
}

ffi::Error env_reset_cuda(cudaStream_t st, i32 ordinal, i64 handle, bool keep_goals, ffi::AnyBuffer states_in,
                          ffi::AnyBuffer task_ids, ffi::AnyBuffer steps_in, ffi::AnyBuffer count_in,
                          ffi::AnyBuffer counts_in, ffi::AnyBuffer views_in, ffi::AnyBuffer goal_pos_in,
                          ffi::AnyBuffer goal_neg_in, ffi::AnyBuffer mask, ffi::Result<ffi::AnyBuffer> states,
                          ffi::Result<ffi::AnyBuffer> steps, ffi::Result<ffi::AnyBuffer> count,
                          ffi::Result<ffi::AnyBuffer> counts, ffi::Result<ffi::AnyBuffer> views,
                          ffi::Result<ffi::AnyBuffer> goal_pos, ffi::Result<ffi::AnyBuffer> goal_neg)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        ffi::Result<ffi::AnyBuffer> res[] = {states, steps, count, counts, views, goal_pos, goal_neg};
        const ffi::AnyBuffer args[] = {states_in, steps_in, count_in, counts_in, views_in, goal_pos_in, goal_neg_in};
        const StateBufs s = state_results(*e, res, args, Fields{}, OnStream{st});
        const u64 N = s.states.rows;
        const View m = view(mask, "mask", ffi::DataType::PRED, true);
        expect_rows(m, N, "mask");
        i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), N);
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceEnv& env = device_env(*e, *d, s);
        env.set_stream(st);
        rl::EnvBatch b = make_batch(*e, s, N, ids);
        env.reset(b, m.as<u8>(), s.count.as<i32>(), keep_goals);
    });
}

ffi::Error env_refresh_cuda(cudaStream_t st, i32 ordinal, i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids,
                            ffi::AnyBuffer counts_in, ffi::AnyBuffer views_in, ffi::Result<ffi::AnyBuffer> count,
                            ffi::Result<ffi::AnyBuffer> counts, ffi::Result<ffi::AnyBuffer> views)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        StateBufs s;
        s.states = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        s.count = view(*count, "count", ffi::DataType::S32, true);
        s.counts = view(*counts, "counts", ffi::DataType::U32, false);
        s.views = view(*views, "views", ffi::DataType::U32, false);
        expect_rows(s.count, s.states.rows, "count");
        i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), s.states.rows);
        // the cache is recomputed: the operands' contents do not matter, only that the results are the arrays
        (void)counts_in;
        (void)views_in;
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceEnv& env = device_env(*e, *d, s);
        env.set_stream(st);
        rl::EnvBatch b = make_batch(*e, s, s.states.rows, ids);
        env.refresh(b, s.count.as<i32>());
    });
}

ffi::Error env_step_cuda(cudaStream_t st, i32 ordinal, i64 handle, MYMYR_STEP_PARAMS)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const StepCall c = step_call(*e, states_in, task_ids_in, steps_in, draws_in, count_in, counts_in, views_in,
                                     goal_pos_in, goal_neg_in, env_id, seed, action, next_task_ids, states, task_ids,
                                     steps, draws, count, counts, views, goal_pos, goal_neg, reward, terminated,
                                     truncated, goal, invalid, schema, binding, final_states, OnStream{st});
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceEnv& env = device_env(*e, *d, c.s);
        env.set_stream(st);
        rl::EnvBatch b = make_batch(*e, c.s, c.N, e->multi() ? c.s.task_ids.as<i32>() : nullptr);
        b.seeds = c.in.seed.as<u64>();
        b.env_ids = c.in.env_id.as<u32>();
        rl::StepOutputs out = make_outputs(c.o);
        out.count = c.s.count.as<i32>();
        env.step(b, out, rl::Actions{nullptr, c.in.action.as<i32>()}, e->multi() ? c.in.next.as<i32>() : nullptr);
    });
}

cuda::SuiteExpander& device_expander(Entry& e, Entry::Device& d, cudaStream_t s)
{
    if (!d.expander)
        d.expander = std::make_unique<cuda::SuiteExpander>(d.ctx, e.suite, s);
    d.expander->set_stream(s);
    return *d.expander;
}

/// The expansion of `st` on the device into the slot's flat scratch (all successors): the flat arrays.
rl::Expansion device_expand(Entry& e, Entry::Device& d, const View& st, const i32* ids, cudaStream_t s)
{
    cuda::SuiteExpander& x0 = device_expander(e, d, s);
    rl::ExpandOptions opt;
    opt.canonical_order = e.cfg.canonical_order;
    opt.witness_pruning = e.cfg.witness_pruning;
    const u64 N = st.rows;
    const u64 total = x0.count(rl::StateBatchView{st.as<u64>(), N, e.W, 0, e.NN}, ids, opt);
    rl::Expansion x;
    x.capacity = std::min(total, rl::k_max_rows);  // more successors: the write raises
    x.words = e.W;
    x.numeric_words = e.NN;
    x.label_width = e.L;
    const u64 rows = std::max<u64>(x.capacity, 1);
    x.succ = static_cast<u64*>(d.succ.ensure(d.ctx, rows * e.row_words() * 8, s));
    x.schema = static_cast<i32*>(d.schema.ensure(d.ctx, rows * 4, s));
    x.binding = static_cast<i32*>(d.binding.ensure(d.ctx, rows * e.L * 4, s));
    x.goal = static_cast<u8*>(d.goal.ensure(d.ctx, rows, s));
    x.offsets = static_cast<i32*>(d.offsets.ensure(d.ctx, (N + 1) * 4, s));
    x0.write(x);
    if (x.words_needed > e.W)
        fail("a successor needs " + std::to_string(x.words_needed) + " words; the rows have " + std::to_string(e.W));
    return x;
}

ffi::Error expand_cuda(cudaStream_t st, i32 ordinal, i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids,
                       ffi::Result<ffi::AnyBuffer> succ, ffi::Result<ffi::AnyBuffer> schema,
                       ffi::Result<ffi::AnyBuffer> binding, ffi::Result<ffi::AnyBuffer> goal,
                       ffi::Result<ffi::AnyBuffer> mask, ffi::Result<ffi::AnyBuffer> count)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const View sv = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        const i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), sv.rows);
        const PaddedBufs p = padded_results(*e, sv.rows, succ, schema, binding, goal, mask, count);
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceGuard g(ordinal);
        if (sv.rows == 0)
            return;
        const rl::Expansion x = device_expand(*e, *d, sv, ids, st);
        rl::PaddedExpansion out = make_padded(*e, p);
        d->expander->pad(x, sv.rows, out);
    });
}

ffi::Error expand_flat_cuda(cudaStream_t st, i32 ordinal, i64 handle, ffi::AnyBuffer states, ffi::AnyBuffer task_ids,
                            ffi::Result<ffi::AnyBuffer> succ, ffi::Result<ffi::AnyBuffer> parent,
                            ffi::Result<ffi::AnyBuffer> schema, ffi::Result<ffi::AnyBuffer> binding,
                            ffi::Result<ffi::AnyBuffer> goal, ffi::Result<ffi::AnyBuffer> offsets)
{
    return guarded([&] {
        EntryPtr e = entry(handle);
        const View sv = view(states, "states", ffi::DataType::U32, false, 2 * e->row_words());
        const i32* ids = task_ids_of(*e, view(task_ids, "task_ids", ffi::DataType::S32, true), sv.rows);
        const FlatBufs f = flat_results(*e, sv.rows, succ, parent, schema, binding, goal, offsets);
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        cuda::DeviceGuard g(ordinal);
        if (f.succ.count)
            cuda::check(cudaMemsetAsync(f.succ.data, 0, f.succ.count * 4, st), "cudaMemsetAsync");
        for (const View* v : {&f.parent, &f.schema, &f.binding})
            if (v->count)
                cuda::check(cudaMemsetAsync(v->data, 0xFF, v->count * 4, st), "cudaMemsetAsync");
        if (f.goal.count)
            cuda::check(cudaMemsetAsync(f.goal.data, 0, f.goal.count, st), "cudaMemsetAsync");
        cuda::SuiteExpander& x0 = device_expander(*e, *d, st);
        rl::ExpandOptions opt;
        opt.canonical_order = e->cfg.canonical_order;
        opt.witness_pruning = e->cfg.witness_pruning;
        rl::Expansion x = make_flat(*e, f);
        x0.expand(rl::StateBatchView{sv.as<u64>(), sv.rows, e->W, 0, e->NN}, ids, x, opt);
    });
}

/// The initialize stage of the environment calls (before XLA runs a call, or traces it into a command buffer by
/// stream capture): sizes the fast path's scratch for the call's rows, so that the execute stage allocates nothing.
/// `first` is the call's first operand: the states [..., 2 RW], or (vector) the task ids of mymyr_env_init.
ffi::Error env_initialize_cuda(i32 ordinal, const ffi::Dictionary& attrs, const ffi::AnyBuffer& first, bool vector)
{
    return guarded([&] {
        const auto handle = attrs.get<i64>("handle");
        if (!handle.has_value())
            fail("an environment call without its 'handle' attribute");
        EntryPtr e = entry(handle.value());
        const auto dims = first.dimensions();
        u64 rows = first.element_count();
        if (!vector && dims.size() > 0)
        {
            rows = 1;
            for (usize i = 0; i + 1 < dims.size(); ++i)
                rows *= static_cast<u64>(dims[i]);
        }
        auto d = e->device(ordinal);
        std::lock_guard lock(d->mutex);
        if (d->fast)
            d->fast->reserve(rows);
    });
}
ffi::Error env_initialize_rows_cuda(i32 ordinal, ffi::Dictionary attrs, ffi::AnyBuffer states, ffi::RemainingArgs,
                                    ffi::RemainingRets)
{
    return env_initialize_cuda(ordinal, attrs, states, false);
}
ffi::Error env_initialize_ids_cuda(i32 ordinal, ffi::Dictionary attrs, ffi::AnyBuffer task_ids, ffi::RemainingArgs,
                                   ffi::RemainingRets)
{
    return env_initialize_cuda(ordinal, attrs, task_ids, true);
}

// The environment calls enqueue the same kernels for the same shapes and attributes, synchronize nothing and allocate
// nothing on the fast path (the initialize stage sized the scratch), so XLA may trace them into command buffers (CUDA
// graphs); init and reset do so on the general path too. The general path's refresh and step synchronize on the
// successor count: the targets mymyr_env_refresh_sync and mymyr_env_step_sync run them outside command buffers (the
// same handlers without the trait; mymyr.rl.jax picks them for states without the cache columns), and so do the
// expansions.
#define MYMYR_CUDA_CTX .Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::DeviceOrdinal>().Attr<i64>("handle")
#define MYMYR_CMD_BUFFER {ffi::Traits::kCmdBufferCompatible}
#define MYMYR_INITIALIZE                                                                                                \
    ffi::Ffi::BindInitialize().Ctx<ffi::DeviceOrdinal>().Attrs().Arg<ffi::AnyBuffer>().RemainingArgs().RemainingRets()
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_initialize_rows_cuda, env_initialize_rows_cuda, MYMYR_INITIALIZE);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_initialize_ids_cuda, env_initialize_ids_cuda, MYMYR_INITIALIZE);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_init_cuda, env_init_cuda, ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_INIT_BINDING,
                              MYMYR_CMD_BUFFER);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_reset_cuda, env_reset_cuda,
                              ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_RESET_BINDING, MYMYR_CMD_BUFFER);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_refresh_cuda, env_refresh_cuda,
                              ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_REFRESH_BINDING, MYMYR_CMD_BUFFER);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_step_cuda, env_step_cuda, ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_STEP_BINDING,
                              MYMYR_CMD_BUFFER);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_refresh_sync_cuda, env_refresh_cuda,
                              ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_REFRESH_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_env_step_sync_cuda, env_step_cuda,
                              ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_STEP_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_expand_cuda, expand_cuda, ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_EXPAND_BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_expand_flat_cuda, expand_flat_cuda,
                              ffi::Ffi::Bind() MYMYR_CUDA_CTX MYMYR_EXPAND_BINDING);
#undef MYMYR_CUDA_CTX
#undef MYMYR_CMD_BUFFER
#undef MYMYR_INITIALIZE
#endif  // MYMYR_HAS_CUDA

#undef MYMYR_STEP_PARAMS
#undef MYMYR_RETS7
#undef MYMYR_INIT_BINDING
#undef MYMYR_RESET_BINDING
#undef MYMYR_REFRESH_BINDING
#undef MYMYR_STEP_BINDING
#undef MYMYR_EXPAND_BINDING
#endif  // MYMYR_HAS_XLA_FFI

/// A target: its name, platform, handlers per execution stage ("execute", "initialize"), and whether XLA may trace
/// its calls into command buffers (the FFI trait kCmdBufferCompatible, which JAX's registration takes explicitly).
using Target = std::tuple<std::string, std::string, std::map<std::string, nb::capsule>, bool>;

std::vector<Target> ffi_targets()
{
    std::vector<Target> out;
#if defined(MYMYR_HAS_XLA_FFI)
    auto add = [&](const char* name, const char* platform, XLA_FFI_Handler* h, XLA_FFI_Handler* init = nullptr) {
        std::map<std::string, nb::capsule> stages;
        stages.emplace("execute", nb::capsule(reinterpret_cast<void*>(h)));
        if (init)
            stages.emplace("initialize", nb::capsule(reinterpret_cast<void*>(init)));
        out.emplace_back(name, platform, std::move(stages), init != nullptr);  // the CUDA env targets
    };
    add("mymyr_env_init", "cpu", mymyr_ffi_env_init_cpu);
    add("mymyr_env_reset", "cpu", mymyr_ffi_env_reset_cpu);
    add("mymyr_env_refresh", "cpu", mymyr_ffi_env_refresh_cpu);
    add("mymyr_env_refresh_sync", "cpu", mymyr_ffi_env_refresh_cpu);
    add("mymyr_env_step", "cpu", mymyr_ffi_env_step_cpu);
    add("mymyr_env_step_sync", "cpu", mymyr_ffi_env_step_cpu);
    add("mymyr_expand", "cpu", mymyr_ffi_expand_cpu);
    add("mymyr_expand_flat", "cpu", mymyr_ffi_expand_flat_cpu);
#if defined(MYMYR_HAS_CUDA)
    add("mymyr_env_init", "CUDA", mymyr_ffi_env_init_cuda, mymyr_ffi_env_initialize_ids_cuda);
    add("mymyr_env_reset", "CUDA", mymyr_ffi_env_reset_cuda, mymyr_ffi_env_initialize_rows_cuda);
    add("mymyr_env_refresh", "CUDA", mymyr_ffi_env_refresh_cuda, mymyr_ffi_env_initialize_rows_cuda);
    add("mymyr_env_refresh_sync", "CUDA", mymyr_ffi_env_refresh_sync_cuda);
    add("mymyr_env_step", "CUDA", mymyr_ffi_env_step_cuda, mymyr_ffi_env_initialize_rows_cuda);
    add("mymyr_env_step_sync", "CUDA", mymyr_ffi_env_step_sync_cuda);
    add("mymyr_expand", "CUDA", mymyr_ffi_expand_cuda);
    add("mymyr_expand_flat", "CUDA", mymyr_ffi_expand_flat_cuda);
#endif
#endif
    return out;
}
}  // namespace

void bind_rl_jax(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_rl_jax", "XLA FFI targets of the planning environments (mymyr.rl.jax)");

    nb::class_<PyJaxEnv>(m, "Env",
                         "A planning environment over a TaskSuite or a TaskTable (or a Task) registered for the XLA FFI "
                         "targets "
                         "(mymyr.rl.jax.Env builds on it): its calls carry `handle`. device=None runs its calls on the "
                         "CPU (rl::HostEnv, any table), a CUDA device index there (cuda::DeviceEnv: classical tables); "
                         "the same results bit for bit.")
        .def(
            "__init__",
            [](PyJaxEnv* self, SuiteArg table, Arg<int> device, ContextArg ctx, IntArg max_steps_in, f32 step_reward,
               f32 goal_reward, Arg<ann::DeadEnd> dead_end, f32 dead_end_reward, bool dead_end_terminal, bool autoreset,
               bool canonical, bool witness, Arg<ann::EnvPath> path) {
                const u32 max_steps = int_arg<u32>(max_steps_in, "max_steps");
                rl::EnvConfig c;
                c.max_steps = max_steps;
                c.step_reward = step_reward;
                c.goal_reward = goal_reward;
                c.dead_end = parse_dead_end(nb::cast<std::string>(dead_end));
                c.dead_end_reward = dead_end_reward;
                c.dead_end_terminal = dead_end_terminal;
                c.autoreset = autoreset;
                c.canonical_order = canonical;
                c.witness_pruning = witness;
                new (self) PyJaxEnv(std::move(table), std::move(device), std::move(ctx), c, nb::cast<std::string>(path));
            },
            "table"_a, nb::kw_only(), "device"_a = nb::none(), "ctx"_a = nb::none(), "max_steps"_a = 0,
            "step_reward"_a = -1.0f, "goal_reward"_a = 0.0f, "dead_end"_a = "no_successors", "dead_end_reward"_a = 0.0f,
            "dead_end_terminal"_a = true, "autoreset"_a = true, "canonical"_a = true, "witness"_a = false,
            "path"_a = "auto")
        .def_prop_ro("handle", &PyJaxEnv::handle, "The FFI attribute `handle` of this environment's calls.")
        .def_prop_ro("table", [](const PyJaxEnv& e) { return SuiteArg(e.ref().obj); })
        .def_prop_ro("num_instances", [](const PyJaxEnv& e) { return e.entry().suite->size(); })
        .def_prop_ro("device", &PyJaxEnv::device, "The CUDA device index, or None (the CPU).")
        .def_prop_ro("fast", &PyJaxEnv::fast, "Whether the device environment takes the fast path.")
        .def_prop_ro("words", [](const PyJaxEnv& e) { return e.entry().W; }, "Atom words per state row.")
        .def_prop_ro("numeric_words", [](const PyJaxEnv& e) { return e.entry().NN; }, "Numeric words per row.")
        .def_prop_ro("row_words", [](const PyJaxEnv& e) { return e.entry().row_words(); },
                     "u64 words per state row (a JAX row is 2 row_words uint32).")
        .def_prop_ro("label_width", [](const PyJaxEnv& e) { return e.entry().L; }, "Columns of the binding labels.")
        .def_prop_ro("cache_schemas", &PyJaxEnv::cache_schemas,
                     "uint32 columns of the fast path's count cache (0 on the CPU and on the general path).")
        .def_prop_ro("cache_view_words", &PyJaxEnv::cache_view_words,
                     "u64 columns of the fast path's views cache (a JAX row is 2 cache_view_words uint32).")
        .def_prop_ro("initial_counts", &PyJaxEnv::initial_counts, "Successors of each instance's initial state [I].")
        .def_prop_ro("max_objects", [](const PyJaxEnv& e) { return e.entry().suite->max_objects(); })
        .def_prop_ro("num_schemas", [](const PyJaxEnv& e) { return e.entry().suite->max_schemas(); })
        .def_prop_ro("max_steps", [](const PyJaxEnv& e) { return e.entry().cfg.max_steps; })
        .def_prop_ro("step_reward", [](const PyJaxEnv& e) { return e.entry().cfg.step_reward; })
        .def_prop_ro("goal_reward", [](const PyJaxEnv& e) { return e.entry().cfg.goal_reward; })
        .def_prop_ro("dead_end", [](const PyJaxEnv& e) { return std::string(dead_end_name(e.entry().cfg.dead_end)); })
        .def_prop_ro("dead_end_reward", [](const PyJaxEnv& e) { return e.entry().cfg.dead_end_reward; })
        .def_prop_ro("dead_end_terminal", [](const PyJaxEnv& e) { return e.entry().cfg.dead_end_terminal; })
        .def_prop_ro("autoreset", [](const PyJaxEnv& e) { return e.entry().cfg.autoreset; })
        .def_prop_ro("canonical", [](const PyJaxEnv& e) { return e.entry().cfg.canonical_order; })
        .def_prop_ro("witness", [](const PyJaxEnv& e) { return e.entry().cfg.witness_pruning; })
        .def("initial_states", &PyJaxEnv::initial_states,
             "The instances' initial states as uint32 halves, row-major [I * 2 row_words] (low word first).")
        .def("set_launch", &PyJaxEnv::set_launch, "launch"_a,
             "How the device calls launch over the row-width buckets of a table of several instances: 'widest' or "
             "'per_bucket' (defaults: the steps' pick 'widest', the expansions 'per_bucket'; cuda::BucketLaunch). The "
             "results do not depend on it.")
        .def("check_errors", &PyJaxEnv::check_errors,
             "Waits for this environment's device work and raises RuntimeError if a fast-path step met a state whose "
             "cache did not describe it (states written without refresh), ValueError for task ids outside the table. "
             "A no-op on the CPU.")
        .def("__repr__", [](const PyJaxEnv& e) {
            return std::string("Env(") + (e.device() ? "cuda:" + std::to_string(*e.device()) : std::string("cpu")) +
                   (e.fast() ? ", fast" : "") + ", instances=" + std::to_string(e.entry().suite->size()) +
                   ", handle=" + std::to_string(e.handle()) + ")";
        });

    m.def("ffi_targets", &ffi_targets,
          "The XLA FFI handlers as (target name, platform, {stage: capsule}, command-buffer compatible) for "
          "jax.ffi.register_ffi_target (stages 'execute' and 'initialize'); empty when this mymyr was built without the "
          "XLA FFI headers (jaxlib's).");
    m.def(
        "has_ffi",
        [] {
#if defined(MYMYR_HAS_XLA_FFI)
            return true;
#else
            return false;
#endif
        },
        "Whether this mymyr was built with the XLA FFI headers.");
}
}  // namespace mymyr::python
