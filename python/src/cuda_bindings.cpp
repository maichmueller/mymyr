// mymyr._core._cuda (CUDA builds only; mymyr.cuda re-exports it): device contexts, the device task upload, device
// arenas with tail syncs, and smoke-test device operations; the device BrFS and the device path of rl.expand /
// rl.expand_into (installed through device_hooks.hpp). Zero-copy DLPack in both directions:
//   - outputs are device DLArrays (torch / JAX / dlpack) produced on the operation's stream; __dlpack__(stream=s)
//     makes the consumer's stream wait for it and records s as a user of the memory;
//   - inputs are imported with __dlpack__(stream=<the operation's stream>), so the producer (torch, JAX, mymyr) makes
//     its pending writes visible there, and read in place; the imports are released only after the operation's work
//     has completed (a per-context list of (event, import) pairs drained by later calls), never while a kernel may
//     still read them.
// Streams: stream=None runs an operation on the context's stream; an int is a cudaStream_t (1 and 2 are the legacy
// and per-thread default streams); an object with `cuda_stream` (torch.cuda.Stream) is taken by its handle. External
// streams must outlive the arrays produced on them.

#include "arrays.hpp"
#include "completed_release.hpp"
#include "capture_scope.hpp"
#include "cuda_stream.hpp"
#include "device_hooks.hpp"
#include "dlpack.hpp"
#include "py_table.hpp"
#include "py_task.hpp"
#include "translate_bindings.hpp"

#include "mymyr/cuda/arena.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"
#include "mymyr/cuda/device_task.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/rl_ops.hpp"
#include "mymyr/cuda/kernels.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/masks.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/translate.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#include "mymyr/rl/masks.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/validate.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <type_traits>
#include <vector>

namespace mymyr::python::ann
{
/// A memory kind of to_memory.
struct Memory
{
};
}  // namespace mymyr::python::ann

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::Memory>
{
    static constexpr auto Name = const_name("typing.Literal['device', 'managed', 'pinned']");
};
}  // namespace nanobind::detail

namespace mymyr::python
{
using namespace nb::literals;
namespace k = mymyr::cuda::kernels;

namespace
{
// ------------------------------------------------------------------------------------------------ streams

struct OpStream
{
    cudaStream_t s = nullptr;
    std::intptr_t dl = 0;  // the same stream as a DLPack stream value
};

// ------------------------------------------------------------------------------------------------ context

/// A device context plus the imports waiting for the work that reads them.
class PyContext
{
public:
    explicit PyContext(cuda::ContextPtr c) : ctx(std::move(c)) {}
    ~PyContext()
    {
        cuda::detail::RelaxedCaptureScope capture_mode;
        try
        {
            drain(true);
        }
        catch (...)
        {
        }
    }
    PyContext(const PyContext&) = delete;
    PyContext& operator=(const PyContext&) = delete;

    cuda::ContextPtr ctx;

    OpStream stream_of(nb::handle stream) const
    {
        if (stream.is_none())
            return {ctx->stream(), reinterpret_cast<std::intptr_t>(ctx->stream())};
        nb::object h = nb::getattr(stream, "cuda_stream", nb::none());
        const std::intptr_t v = nb::cast<std::intptr_t>(h.is_none() ? nb::borrow(stream) : h);
        if (v == dl::k_stream_none)
            throw nb::value_error("mymyr: stream -1 means 'no synchronization'; pass a stream to run on");
        if (v == 0)
            return {cudaStreamLegacy, dl::k_stream_legacy};  // torch reports its default stream as 0
        return {cuda_stream(v), v};
    }

    /// Releases `keep` once the work enqueued on s so far has completed.
    void release_after(cudaStream_t s, std::vector<std::shared_ptr<void>> keep)
    {
        if (keep.empty())
            return;
        cuda::DeviceGuard g(ctx->device());
        cuda::Event e;
        e.record(s);
        std::lock_guard lock(m_mutex);
        m_items.push_back({std::move(e), std::move(keep)});
    }
    /// Drops the imports whose work has completed (all of them after waiting, with `all`).
    void drain(bool all = false)
    {
        std::deque<Item> done;
        {
            std::lock_guard lock(m_mutex);
            for (auto it = m_items.begin(); it != m_items.end();)
            {
                if (all)
                    it->event.synchronize();
                if (all || it->event.done())
                {
                    done.push_back(std::move(*it));
                    it = m_items.erase(it);
                }
                else
                    ++it;
            }
        }
        // the producers' deleters run here, outside the lock
    }
    [[nodiscard]] usize pending() const
    {
        std::lock_guard lock(m_mutex);
        return m_items.size();
    }

private:
    struct Item
    {
        cuda::Event event;
        std::vector<std::shared_ptr<void>> keep;
    };
    mutable std::mutex m_mutex;
    std::deque<Item> m_items;
};
using PyContextPtr = std::shared_ptr<PyContext>;

struct PyContextObj
{
    PyContextPtr p;
};

PyContextPtr context_of(nb::handle ctx, int device = 0)
{
    if (ctx.is_none())
        return std::make_shared<PyContext>(cuda::DeviceContext::create(device));
    if (!nb::isinstance<PyContextObj>(ctx))
        throw nb::type_error("mymyr: ctx must be a mymyr.cuda.Context or None");
    return nb::inst_ptr<PyContextObj>(ctx)->p;
}

// ------------------------------------------------------------------------------------------------ exports

/// Handoff of memory written on `producer`: the consumer's stream waits for it and becomes a user of the buffer.
class BufferSync final : public DeviceSync
{
public:
    BufferSync(std::shared_ptr<cuda::DeviceBuffer> buffer, cudaStream_t producer) : m_buffer(std::move(buffer)), m_producer(producer) {}
    void handoff(std::intptr_t consumer) override
    {
        const cudaStream_t c = cuda_stream(consumer);
        cuda::DeviceGuard g(m_buffer->context()->device());
        cuda::stream_wait(c, m_producer);
        m_buffer->record_stream(c);
    }

private:
    std::shared_ptr<cuda::DeviceBuffer> m_buffer;
    cudaStream_t m_producer;
};

/// Handoff of the uploaded task block: the consumer waits for the upload and becomes a user of the block.
class TaskSync final : public DeviceSync
{
public:
    explicit TaskSync(std::shared_ptr<cuda::DeviceTask> dt) : m_dt(std::move(dt)) {}
    void handoff(std::intptr_t consumer) override
    {
        cuda::DeviceGuard g(m_dt->context()->device());
        m_dt->acquire(cuda_stream(consumer));
    }

private:
    std::shared_ptr<cuda::DeviceTask> m_dt;
};

/// A fresh device output of `bytes` bytes allocated on the operation's stream.
std::shared_ptr<cuda::DeviceBuffer> device_output(const PyContext& c, u64 bytes, const OpStream& os)
{
    return std::make_shared<cuda::DeviceBuffer>(c.ctx, std::max<u64>(bytes, 1), os.s);
}

nb::object export_buffer(const std::shared_ptr<cuda::DeviceBuffer>& buf, const OpStream& os, rl::DType dtype, std::vector<i64> shape,
                         bool words, Framework fw, WordEncoding enc, bool readonly = false)
{
    ArraySpec spec;
    spec.readonly = readonly;
    spec.owner = buf;
    spec.data = buf->data();
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.words = words;
    DeviceExport dev{dl::k_cuda, buf->context()->device(), std::make_shared<BufferSync>(buf, os.s)};
    return export_device_array(std::move(spec), std::move(dev), fw, enc);
}

// ------------------------------------------------------------------------------------------------ imports

/// An input read in place on the device.
struct DeviceInput
{
    dl::Imported im;
    Framework fw = Framework::DLPack;
};

/// The stream handoffs of one call. torch's __dlpack__(stream=s) makes s wait on torch's current stream as a whole,
/// not on the one tensor, so once a call's first torch import has done that, its later torch imports pass no stream
/// (each handoff costs about 8 us of host time).
struct ImportSync
{
    bool torch = false;
};

DeviceInput import_device(nb::handle obj, const PyContext& c, const OpStream& os, const char* name,
                          ImportSync* sync = nullptr)
{
    const dl::Device d = dl::dlpack_device(obj);
    if (d.device_type != dl::k_cuda && d.device_type != dl::k_cuda_managed)
        throw nb::type_error((std::string("mymyr: '") + name + "' must be a CUDA device (or managed) array on cuda:" +
                              std::to_string(c.ctx->device()) + "; move host data with tensor.cuda() / jax.device_put")
                                 .c_str());
    if (d.device_id != c.ctx->device())
        throw nb::value_error((std::string("mymyr: '") + name + "' lives on cuda:" + std::to_string(d.device_id) +
                               ", the context on cuda:" + std::to_string(c.ctx->device()))
                                  .c_str());
    DeviceInput in;
    in.fw = framework_of(obj);
    const bool torch = sync && in.fw == Framework::Torch;
    in.im = dl::import_dlpack(obj, torch && sync->torch ? dl::k_stream_none : os.dl);
    if (torch)
        sync->torch = true;
    return in;
}

WordsLayout device_words(const DeviceInput& in)
{
    const dl::Imported& im = in.im;
    return words_layout(im.data, im.dtype.code, im.dtype.bits, im.dtype.lanes, im.shape.size(), im.shape.data(), im.strides.data());
}

/// A C-contiguous 32-bit integer array with `ndim` dimensions.
const u32* device_u32(const DeviceInput& in, usize ndim, const char* name)
{
    const dl::Imported& im = in.im;
    const bool ok_type = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.bits == 32 && im.dtype.lanes == 1;
    bool contiguous = im.shape.size() == ndim;
    i64 expect = 1;
    for (usize d = im.shape.size(); contiguous && d-- > 0;)
    {
        if (im.shape[d] > 1 && im.strides[d] != expect)
            contiguous = false;
        expect *= im.shape[d];
    }
    if (!ok_type || !contiguous)
        throw nb::type_error((std::string("mymyr: '") + name + "' must be a C-contiguous " + std::to_string(ndim) +
                              "-d int32 array")
                                 .c_str());
    return static_cast<const u32*>(im.data);
}

k::DeviceStates to_states(const WordsLayout& w) { return {w.data, w.rows, w.words, w.stride}; }

std::pair<Framework, WordEncoding> output_kind(const DeviceInput& in, const WordsLayout& w, nb::handle framework)
{
    if (!framework.is_none())
    {
        const Framework fw = parse_framework(framework, nb::none());
        return {fw, fw == in.fw || fw == Framework::DLPack ? w.enc : default_words(fw)};
    }
    return {in.fw == Framework::Numpy ? Framework::DLPack : in.fw, w.enc};
}

// ------------------------------------------------------------------------------------------------ DeviceTask

struct PyDeviceTask
{
    PyContextPtr ctx;
    std::shared_ptr<cuda::DeviceTask> dt;
    TaskPtr task;
};

struct SmokeInputs
{
    OpStream os;
    DeviceInput states, derived;
    WordsLayout w, dw;
    bool has_derived = false;
    std::shared_ptr<cuda::DeviceBuffer> numeric_rows, numeric_views, numeric_error;
    std::vector<std::shared_ptr<void>> keep;
};

SmokeInputs smoke_inputs(PyDeviceTask& t, nb::handle states, nb::handle derived, nb::handle stream)
{
    SmokeInputs in;
    t.ctx->drain();
    in.os = t.ctx->stream_of(stream);
    in.states = import_device(states, *t.ctx, in.os, "states");
    in.w = device_words(in.states);
    in.keep.push_back(in.states.im.keep);
    if (!derived.is_none())
    {
        in.derived = import_device(derived, *t.ctx, in.os, "derived");
        in.dw = device_words(in.derived);
        if (in.dw.rows != in.w.rows && !(in.w.single && in.dw.single))
            throw nb::value_error("mymyr: 'derived' needs one row per state");
        if (in.dw.stride != in.dw.words && in.dw.rows > 1)
            throw nb::type_error("mymyr: 'derived' rows must be contiguous");
        in.has_derived = true;
        in.keep.push_back(in.derived.im.keep);
    }
    else if (t.task->has_axioms())
        throw nb::value_error("mymyr: the task has axioms: pass derived=<[N, DW] derived words> (e.g. from "
                              "DeviceTask.host_derived); these smoke operations do not evaluate axiom strata "
                              "themselves (the search and expand paths do)");
    if (t.task->numeric_slots())
    {
        const u32 nn = t.task->numeric_words();
        if (in.w.words <= nn || in.w.rows > 0xFFFFFFFFu)
            throw nb::value_error("mymyr: malformed numeric smoke state rows");
        const u32 atoms = in.w.words - nn, rw = atoms + t.task->numeric_slots();
        if (rw > cuda::lifted::k_max_words || t.task->compiled().max_bind > cuda::lifted::k_max_depth)
            throw nb::value_error("mymyr: numeric smoke operations exceed the device row or binding limit; use rl.expand");
        const auto& view = t.dt->acquire(in.os.s);
        in.numeric_rows = device_output(*t.ctx, in.w.rows * rw * sizeof(u64), in.os);
        in.numeric_views = device_output(*t.ctx, checked_mul({in.w.rows, view.view_rows, view.ow, 8}, "the numeric views"), in.os);
        in.numeric_error = device_output(*t.ctx, sizeof(u32), in.os);
        cuda::check(cudaMemsetAsync(in.numeric_error->data(), 0, sizeof(u32), in.os.s), "numeric smoke error");
        cuda::check(cuda::numeric::launch_convert(view, in.w.data, in.w.stride, atoms,
            static_cast<u64*>(in.numeric_rows->data()), rw, atoms, in.w.rows, true, in.os.s), "numeric smoke input");
        in.w.data = static_cast<const u64*>(in.numeric_rows->data()); in.w.words = atoms; in.w.stride = rw;
        cuda::lifted::Parents parents{in.w.data, rw, atoms, static_cast<u32>(in.w.rows),
                                     in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0};
        cuda::check(cuda::lifted::launch_view(view, parents,
            {static_cast<u64*>(in.numeric_views->data()), u64{view.view_rows} * view.ow}, in.os.s), "numeric smoke views");
        in.keep.push_back(in.numeric_rows); in.keep.push_back(in.numeric_views); in.keep.push_back(in.numeric_error);
    }
    return in;
}

void check_numeric_smoke(const SmokeInputs& in)
{
    u32 error = 0;
    cuda::check(cudaMemcpyAsync(&error, in.numeric_error->data(), sizeof(u32), cudaMemcpyDeviceToHost, in.os.s), "numeric smoke error read");
    cuda::check(cudaStreamSynchronize(in.os.s), "numeric smoke sync");
    if (error) throw std::domain_error("mymyr: numeric smoke successor overflows int32 storage");
}

k::DeviceLabels labels_of(PyDeviceTask& t, SmokeInputs& in, nb::handle state_index, nb::handle schema, nb::handle binding)
{
    DeviceInput si = import_device(state_index, *t.ctx, in.os, "state_index");
    DeviceInput sc = import_device(schema, *t.ctx, in.os, "schema");
    DeviceInput bi = import_device(binding, *t.ctx, in.os, "binding");
    k::DeviceLabels l;
    l.state_index = device_u32(si, 1, "state_index");
    l.schema = device_u32(sc, 1, "schema");
    l.binding = device_u32(bi, 2, "binding");
    l.count = static_cast<u64>(si.im.shape[0]);
    l.width = static_cast<u32>(bi.im.shape[1]);
    if (static_cast<u64>(sc.im.shape[0]) != l.count || static_cast<u64>(bi.im.shape[0]) != l.count)
        throw nb::value_error("mymyr: state_index, schema and binding need the same number of rows");
    for (DeviceInput* x : {&si, &sc, &bi})
        in.keep.push_back(x->im.keep);
    return l;
}

void launched(PyDeviceTask& t, SmokeInputs& in, cudaError_t e, const char* what)
{
    cuda::check(e, what);  // the task block was acquired for in.os.s before the launch
    t.ctx->release_after(in.os.s, std::move(in.keep));
}

// ------------------------------------------------------------------------------------------------ StateArena

struct PyArena
{
    PyContextPtr ctx;
    std::unique_ptr<cuda::DeviceArena> arena;
    u32 words = 0;
};

// ------------------------------------------------------------------------------------------------ to_memory

enum class MemoryKind
{
    Device,
    Managed,
    Pinned,
};

MemoryKind parse_memory(nb::handle memory)
{
    if (!nb::isinstance<nb::str>(memory))
        throw nb::type_error("mymyr: memory must be 'device', 'managed' or 'pinned'");
    const std::string s = nb::cast<std::string>(memory);
    if (s == "device")
        return MemoryKind::Device;
    if (s == "managed")
        return MemoryKind::Managed;
    if (s == "pinned")
        return MemoryKind::Pinned;
    throw nb::value_error("mymyr: memory must be 'device', 'managed' or 'pinned'");
}

rl::DType dtype_of(const nb::dlpack::dtype& d)
{
    const bool sgn = d.code == static_cast<u8>(nb::dlpack::dtype_code::Int);
    const bool uns = d.code == static_cast<u8>(nb::dlpack::dtype_code::UInt);
    if (d.code == static_cast<u8>(nb::dlpack::dtype_code::Bool) && d.bits == 8)
        return rl::DType::Bool;
    if ((sgn || uns) && d.bits == 8)
        return sgn ? rl::DType::I8 : rl::DType::U8;
    if ((sgn || uns) && d.bits == 32)
        return sgn ? rl::DType::I32 : rl::DType::U32;
    if ((sgn || uns) && d.bits == 64)
        return sgn ? rl::DType::I64 : rl::DType::U64;
    throw nb::type_error("mymyr: to_memory: unsupported dtype (bool or 8/32/64-bit integers)");
}

// ------------------------------------------------------------------------------------------------ device expand

using ExpansionShape = std::array<u64, 13>;

struct ExpansionWorkspace;

struct WorkspaceRelease
{
    std::shared_ptr<ExpansionWorkspace> workspace;
    ExpansionWorkspace* completed = nullptr;
    WorkspaceRelease* next = nullptr;
};

struct ExpansionWorkspace
{
    std::unique_ptr<cuda::SuiteExpander> expander;
    std::optional<ExpansionShape> shape;
    std::unique_ptr<cuda::Event> ready;
    std::unique_ptr<WorkspaceRelease> release;
    std::shared_ptr<std::atomic<bool>> released = std::make_shared<std::atomic<bool>>(false);
};

struct WorkspaceReference
{
    std::weak_ptr<ExpansionWorkspace> workspace;
    std::shared_ptr<std::atomic<bool>> released;
};

/// CUDA user-object callbacks may only signal host work: workspace destruction calls CUDA.
class WorkspaceReclaimer
{
public:
    using Item = WorkspaceRelease;

    static WorkspaceReclaimer& get()
    {
        // CUDA can release graph references during runtime teardown, after C++ static destructors.
        static auto* instance = new WorkspaceReclaimer;
        return *instance;
    }

    void release(Item* item) noexcept
    {
        {
            std::lock_guard lock(m_mutex);
            item->next = m_head;
            m_head = item;
        }
        m_ready.notify_one();
    }

private:
    WorkspaceReclaimer()
    {
        std::thread([this] {
            for (;;)
            {
                Item* items;
                {
                    std::unique_lock lock(m_mutex);
                    m_ready.wait(lock, [this] { return m_head != nullptr; });
                    items = std::exchange(m_head, nullptr);
                }
                while (items)
                {
                    Item* next = items->next;
                    if (items->completed)
                    {
                        auto* workspace = items->completed;
                        try
                        {
                            cuda::detail::release_completed(workspace->expander, *workspace->ready);
                        }
                        catch (...)
                        {
                            // Runtime teardown and sticky errors cannot be retried. Keep CUDA resources alive
                            // without invoking their destructors or making any further CUDA calls.
                            (void)workspace->expander.release();
                            (void)workspace->ready.release();
                        }
                        auto released = workspace->released;
                        delete workspace;
                        released->store(true, std::memory_order_release);
                    }
                    delete items;
                    items = next;
                }
            }
        }).detach();
    }

    std::mutex m_mutex;
    std::condition_variable m_ready;
    Item* m_head = nullptr;
};

std::shared_ptr<ExpansionWorkspace> make_expansion_workspace()
{
    return std::shared_ptr<ExpansionWorkspace>(new ExpansionWorkspace, [](ExpansionWorkspace* workspace) noexcept {
        if (auto* item = workspace->release.release())
        {
            item->completed = workspace;
            WorkspaceReclaimer::get().release(item);
        }
        else
            delete workspace;
    });
}

/// The graph and every executable clone retain the workspace until their final launch completes.
void retain_workspace(cudaGraph_t graph, const std::shared_ptr<ExpansionWorkspace>& workspace)
{
    (void)WorkspaceReclaimer::get();
    auto item = std::make_unique<WorkspaceReclaimer::Item>();
    item->workspace = workspace;
    cudaUserObject_t object;
    cuda::check(cudaUserObjectCreate(&object, item.get(), [](void* p) {
        WorkspaceReclaimer::get().release(static_cast<WorkspaceReclaimer::Item*>(p));
    }, 1, cudaUserObjectNoDestructorSync), "cudaUserObjectCreate");
    (void)item.release();  // the user object owns the callback argument
    const cudaError_t result = cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove);
    if (result != cudaSuccess)
        (void)cudaUserObjectRelease(object);
    cuda::check(result, "cudaGraphRetainUserObject");
}

/// Per-table CUDA state (a Task's on its core, a TaskTable's on the table, a TaskSuite's on the suite): a default
/// context per device, and per context the idle device expanders and the device masks (they keep their context alive
/// until the table goes).
struct TranslateUpload;

struct TaskCuda
{
    std::map<int, PyContextPtr> contexts;
    std::map<std::vector<u64>, std::shared_ptr<TranslateUpload>> translations;
    std::map<const cuda::DeviceContext*, std::vector<std::shared_ptr<ExpansionWorkspace>>> idle;
    std::map<std::pair<const cuda::DeviceContext*, unsigned long long>, std::weak_ptr<ExpansionWorkspace>> captures;
    std::optional<std::string> masks_why;  // DeviceMasks::unsupported of the suite
    std::map<const cuda::DeviceContext*, std::shared_ptr<const cuda::DeviceMasks>> masks;
};

TaskCuda& slot_cuda(std::shared_ptr<void>& slot)  // with the slot's mutex held
{
    if (!slot)
        slot = std::make_shared<TaskCuda>();
    return *static_cast<TaskCuda*>(slot.get());
}

/// The context of a device operation on a table: `ctx`, or the table's default context on the input's device.
PyContextPtr slot_context(std::mutex& mutex, std::shared_ptr<void>& slot, nb::handle ctx, int device)
{
    if (!ctx.is_none())
        return context_of(ctx);
    std::lock_guard lock(mutex);
    PyContextPtr& c = slot_cuda(slot).contexts[device];
    if (!c)
        c = std::make_shared<PyContext>(cuda::DeviceContext::create(device));
    return c;
}

PyContextPtr task_context(PyTaskCore& core, nb::handle ctx, int device)
{
    return slot_context(core.cuda_mutex, core.cuda, ctx, device);
}

PyContextPtr table_context(const SuiteRef& ref, nb::handle ctx, int device)
{
    return slot_context(ref.cuda_mutex(), ref.cuda(), ctx, device);
}

/// An idle device expander of (suite, context), or a new one; it goes back to the idle list when the lease ends. An
/// idle expander last used on stream s is preferred, so that callers on their own streams keep their expanders (a
/// stream change orders the new stream after the old one and moves the expander's scratch).
class ExpanderLease
{
public:
    ExpanderLease(const SuiteRef& ref, const PyContextPtr& c, cudaStream_t s, bool inspect = false) : m_ref(ref), m_ctx(c->ctx)
    {
        {
            std::lock_guard lock(ref.cuda_mutex());
            auto& v = slot_cuda(ref.cuda()).idle[m_ctx.get()];
            if (!v.empty())
            {
                const cudaStream_t want = s ? s : m_ctx->stream();
                auto it = std::find_if(v.rbegin(), v.rend(), [&](const auto& x) { return x->expander->stream() == want; });
                auto at = it == v.rend() ? v.end() - 1 : std::prev(it.base());
                m_x = std::move(*at);
                v.erase(at);
            }
        }
        if (!m_x)
        {
            cuda::DeviceGuard guard(m_ctx->device());
            m_x = make_expansion_workspace();
            m_x->ready = std::make_unique<cuda::Event>();
            m_x->expander = std::make_unique<cuda::SuiteExpander>(m_ctx, ref.suite);
        }
        if (!inspect)
            m_x->shape.reset();
    }
    ~ExpanderLease()
    {
        if (!m_x)
            return;
        std::lock_guard lock(m_ref.cuda_mutex());
        slot_cuda(m_ref.cuda()).idle[m_ctx.get()].push_back(std::move(m_x));
    }
    ExpanderLease(const ExpanderLease&) = delete;
    ExpanderLease& operator=(const ExpanderLease&) = delete;
    cuda::SuiteExpander* operator->() const { return m_x->expander.get(); }
    void warmed(const ExpansionShape& shape)
    {
        cuda::DeviceGuard guard(m_ctx->device());
        m_x->ready->record(m_x->expander->stream());
        m_x->shape = shape;
    }

private:
    const SuiteRef& m_ref;
    cuda::ContextPtr m_ctx;
    std::shared_ptr<ExpansionWorkspace> m_x;
};

/// Each capture uses an exclusive warmed workspace per table and context. Shapes stay fixed so captured pointers
/// cannot move. Successive launches of one executable graph serialize workspace accesses.
std::shared_ptr<ExpansionWorkspace> captured_workspace(const SuiteRef& ref, const PyContextPtr& c, cudaStream_t s,
                                                        unsigned long long id, const ExpansionShape* shape = nullptr)
{
    std::lock_guard lock(ref.cuda_mutex());
    auto& state = slot_cuda(ref.cuda());
    std::erase_if(state.captures, [](const auto& entry) { return entry.second.expired(); });
    const auto key = std::make_pair(c->ctx.get(), id);
    if (auto owner = state.captures[key].lock())
    {
        if (owner->expander->stream() != s || (shape && owner->shape != *shape))
            throw nb::value_error("mymyr: captured expansions over one table need the same stream, shapes and options");
        return owner;
    }
    auto& idle = state.idle[c->ctx.get()];
    auto it = std::find_if(idle.begin(), idle.end(), [&](const auto& owner) {
        return owner->expander->stream() == s && owner->shape && (!shape || owner->shape == *shape);
    });
    if (it == idle.end())
        throw nb::value_error("mymyr: warm expand_into with the same shapes and options on the capture stream before "
                              "CUDA graph capture");
    auto owner = *it;
    if (shape)
    {
        cudaStreamCaptureStatus capture;
        cudaGraph_t graph = nullptr;
        cuda::check(cudaStreamGetCaptureInfo(s, &capture, nullptr, &graph), "cudaStreamGetCaptureInfo");
        auto release = std::make_unique<WorkspaceRelease>();
        retain_workspace(graph, owner);
        owner->release = std::move(release);
        idle.erase(it);
        state.captures[key] = owner;
    }
    return owner;
}

nb::capsule workspace_reference(const std::shared_ptr<ExpansionWorkspace>& workspace)
{
    return nb::capsule(new WorkspaceReference{workspace, workspace->released}, "mymyr.expansion_workspace_weak",
                       [](void* p) noexcept { delete static_cast<WorkspaceReference*>(p); });
}

/// The device task ids of a batch: an int32 CUDA array [rows] on the context's device, or None (null: a table of one).
const i32* device_task_ids(nb::handle obj, const PyContext& c, const OpStream& os, u64 rows,
                           std::vector<std::shared_ptr<void>>& keep)
{
    if (obj.is_none())
        return nullptr;
    const dl::Device d = dl::dlpack_device(obj);
    if (d.device_type != dl::k_cuda && d.device_type != dl::k_cuda_managed)
        throw nb::type_error("mymyr: 'task_ids' must be an int32 CUDA array like the states (device states take device "
                             "task ids)");
    DeviceInput in = import_device(obj, c, os, "task_ids");
    const dl::Imported& im = in.im;
    if (im.dtype.code != dl::k_int || im.dtype.bits != 32 || im.dtype.lanes != 1)
        throw nb::type_error("mymyr: device 'task_ids' must be int32");
    if (im.shape.size() != 1 || static_cast<u64>(im.shape[0]) != rows || (rows > 1 && im.strides[0] != 1))
        throw nb::value_error(("mymyr: 'task_ids' must be a contiguous int32 array [" + std::to_string(rows) + "]").c_str());
    keep.push_back(im.keep);
    return static_cast<const i32*>(im.data);
}

/// A region of a device buffer exported as a device array produced on os.
nb::object export_region(const std::shared_ptr<cuda::DeviceBuffer>& buf, u64 off, const OpStream& os, rl::DType dtype,
                         std::vector<i64> shape, bool words, Framework fw, WordEncoding enc, bool readonly = false)
{
    ArraySpec spec;
    spec.readonly = readonly;
    const std::byte* p = static_cast<const std::byte*>(buf->data()) + off;
    spec.owner = std::shared_ptr<const void>(buf, p);
    spec.data = p;
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.words = words;
    DeviceExport dev{dl::k_cuda, buf->context()->device(), std::make_shared<BufferSync>(buf, os.s)};
    return export_device_array(std::move(spec), std::move(dev), fw, enc);
}

/// The flat result of one device expand: one device buffer holding every array (rl_bindings' FlatResult on the device).
struct DeviceFlat
{
    std::shared_ptr<cuda::DeviceBuffer> buf;
    OpStream os;
    u64 rows = 0, capacity = 0, total = 0, generated = 0;
    u64 off_hash = 0, off_noop = 0, off_index = 0;
    bool has_hash = false, has_noop = false, has_index = false;
    u32 words = 0, label_width = 0, words_needed = 0, numeric_words = 0;
    u64 off_succ = 0, off_parent = 0, off_schema = 0, off_binding = 0, off_goal = 0, off_derived = 0, off_offsets = 0;
    u32 derived_words = 0;
    bool has_goal = false, has_derived = false;

    [[nodiscard]] u64 valid() const noexcept { return std::min(total, capacity); }
    template<class T>
    [[nodiscard]] T* at(u64 off) const noexcept
    {
        return reinterpret_cast<T*>(static_cast<std::byte*>(buf->data()) + off);
    }
    [[nodiscard]] rl::Expansion view() const
    {
        rl::Expansion x;
        x.capacity = capacity;
        x.words = words;
        x.numeric_words = numeric_words;
        x.label_width = label_width;
        x.succ = at<u64>(off_succ);
        x.parent = at<i32>(off_parent);
        x.schema = at<i32>(off_schema);
        x.binding = at<i32>(off_binding);
        x.goal = has_goal ? at<u8>(off_goal) : nullptr;
        x.derived = has_derived && derived_words ? at<u64>(off_derived) : nullptr;
        x.derived_words = derived_words;
        x.offsets = at<i32>(off_offsets);
        x.succ_hash = has_hash ? at<u64>(off_hash) : nullptr;
        x.noop = has_noop ? at<u8>(off_noop) : nullptr;
        x.index = has_index ? at<i32>(off_index) : nullptr;
        x.total = total;
        x.generated = generated;
        x.words_needed = words_needed;
        return x;
    }
    /// The CSR offsets on the host (synchronizes os).
    [[nodiscard]] std::vector<i32> host_offsets() const
    {
        std::vector<i32> h(rows + 1);
        cuda::DeviceGuard g(buf->context()->device());
        cuda::check(cudaMemcpyAsync(h.data(), at<i32>(off_offsets), h.size() * sizeof(i32), cudaMemcpyDeviceToHost, os.s),
                    "cudaMemcpyAsync");
        cuda::check(cudaStreamSynchronize(os.s), "cudaStreamSynchronize");
        return h;
    }
};

DeviceFlat allocate_device_flat(const PyContext& c, const OpStream& os, u64 rows, u64 cap, u32 W, u32 NN, u32 L, bool goal,
                                std::optional<u32> derived, bool hashes, bool noops, bool drop)
{
    DeviceFlat f;
    BlockLayout lay;
    f.os = os;
    f.rows = rows;
    f.capacity = cap;
    f.words = W;
    f.numeric_words = NN;
    f.label_width = L;
    f.has_goal = goal;
    f.has_derived = derived.has_value();
    f.derived_words = derived.value_or(0);
    f.off_succ = lay.add(cap, (u64{W} + NN) * sizeof(u64));
    f.off_parent = lay.add(cap, sizeof(i32));
    f.off_schema = lay.add(cap, sizeof(i32));
    f.off_binding = lay.add(cap, u64{L} * sizeof(i32));
    f.off_goal = lay.add(goal ? cap : 0, 1);
    f.off_derived = lay.add(derived ? cap : 0, u64{f.derived_words} * sizeof(u64));
    f.has_hash = hashes;
    f.has_noop = noops || drop;
    f.has_index = drop;
    f.off_hash = lay.add(hashes ? cap : 0, sizeof(u64));
    f.off_noop = lay.add(f.has_noop ? cap : 0, 1);
    f.off_index = lay.add(drop ? cap : 0, sizeof(i32));
    f.off_offsets = lay.add(checked_add(rows, 1, "the offsets"), sizeof(i32));
    f.buf = device_output(c, lay.bytes, os);
    return f;
}

/// The padded view of a device expansion, on the device.
struct DevicePadded
{
    std::shared_ptr<cuda::DeviceBuffer> buf;
    OpStream os;
    u64 rows = 0;
    u32 k = 0, words = 0, label_width = 0, numeric_words = 0;
    u64 off_index = 0, off_mask = 0, off_count = 0, off_succ = 0, off_schema = 0, off_binding = 0, off_goal = 0, off_derived = 0;
    u32 derived_words = 0;
    bool has_goal = false, has_derived = false, overflow = false, has_hash = false, has_noop = false, has_canonical = false;
    u64 off_hash = 0, off_noop = 0, off_canonical = 0;
    Framework fw = Framework::DLPack;
    WordEncoding enc;

    nb::object array(u64 off, rl::DType dt, std::vector<i64> shape, bool w = false) const
    {
        return export_region(buf, off, os, dt, std::move(shape), w, fw, enc);
    }
};

struct PyDeviceExpansion
{
    DeviceFlat flat;
    PyContextPtr ctx;
    Framework fw = Framework::DLPack;
    WordEncoding enc;
    SuiteRef ref;
    std::shared_ptr<cuda::DeviceBuffer> task_ids;  // a copy of the batch's task ids (tables of several instances)
    std::optional<DevicePadded> padded;

    nb::object array(u64 off, rl::DType dt, std::vector<i64> shape, bool words = false) const
    {
        return export_region(flat.buf, off, flat.os, dt, std::move(shape), words, fw, enc);
    }
};

u32 device_auto_k(const DeviceFlat& f)
{
    const std::vector<i32> off = f.host_offsets();
    i32 m = 1;
    for (u64 i = 0; i < f.rows; ++i)
        m = std::max(m, off[i + 1] - off[i]);
    return std::bit_ceil(static_cast<u32>(m));
}

DevicePadded make_device_padded(const SuiteRef& ref, const PyContextPtr& c, const DeviceFlat& f, u32 k, Framework fw, WordEncoding enc)
{
    DevicePadded p;
    BlockLayout lay;
    p.os = f.os;
    p.rows = f.rows;
    p.k = k;
    p.words = f.words;
    p.numeric_words = f.numeric_words;
    p.label_width = f.label_width;
    p.has_goal = f.has_goal;
    p.has_derived = f.has_derived;
    p.has_hash = f.has_hash;
    p.has_noop = f.has_noop;
    p.has_canonical = f.has_index;
    p.derived_words = f.derived_words;
    p.fw = fw;
    p.enc = enc;
    const u64 NK = checked_mul({f.rows, k}, "a padded expansion (rows x k)");
    p.off_index = lay.add(NK, sizeof(i32));
    p.off_mask = lay.add(NK, 1);
    p.off_count = lay.add(f.rows, sizeof(i32));
    p.off_succ = lay.add(NK, (u64{f.words} + f.numeric_words) * sizeof(u64));
    p.off_schema = lay.add(NK, sizeof(i32));
    p.off_binding = lay.add(NK, u64{f.label_width} * sizeof(i32));
    p.off_goal = lay.add(f.has_goal ? NK : 0, 1);
    p.off_derived = lay.add(f.has_derived ? NK : 0, u64{f.derived_words} * sizeof(u64));
    p.off_hash = lay.add(f.has_hash ? NK : 0, sizeof(u64));
    p.off_noop = lay.add(f.has_noop ? NK : 0, 1);
    p.off_canonical = lay.add(f.has_index ? NK : 0, sizeof(i32));
    p.buf = device_output(*c, lay.bytes, f.os);
    auto* b = static_cast<std::byte*>(p.buf->data());
    rl::PaddedExpansion out;
    out.k = k;
    out.index = reinterpret_cast<i32*>(b + p.off_index);
    out.mask = reinterpret_cast<u8*>(b + p.off_mask);
    out.count = reinterpret_cast<i32*>(b + p.off_count);
    out.words = f.words;
    out.numeric_words = f.numeric_words;
    out.succ = reinterpret_cast<u64*>(b + p.off_succ);
    out.schema = reinterpret_cast<i32*>(b + p.off_schema);
    out.label_width = f.label_width;
    out.binding = reinterpret_cast<i32*>(b + p.off_binding);
    out.goal = f.has_goal ? reinterpret_cast<u8*>(b + p.off_goal) : nullptr;
    out.derived_words = f.derived_words;
    out.derived = f.has_derived && f.derived_words ? reinterpret_cast<u64*>(b + p.off_derived) : nullptr;
    out.succ_hash = f.has_hash ? reinterpret_cast<u64*>(b + p.off_hash) : nullptr;
    out.noop = f.has_noop ? reinterpret_cast<u8*>(b + p.off_noop) : nullptr;
    out.canonical = f.has_index ? reinterpret_cast<i32*>(b + p.off_canonical) : nullptr;
    {
        ExpanderLease ex(ref, c, f.os.s);
        nb::gil_scoped_release release;
        ex->set_stream(f.os.s);
        ex->pad(f.view(), f.rows, out);
    }
    p.overflow = out.overflow || f.generated > f.capacity || f.words_needed > f.words;
    return p;
}

nb::object device_expand(const ExpandArgs& a)
{
    SuiteRef ref = suite_of(a.table);
    const rl::TaskSuite& tt = *ref.suite;
    const dl::Device dev = dl::dlpack_device(a.states);
    PyContextPtr c = table_context(ref, a.ctx, dev.device_id);
    c->drain();
    const OpStream os = c->stream_of(a.stream);
    DeviceInput in = import_device(a.states, *c, os, "states");
    const WordsLayout w = device_words(in);
    const u32 NN = tt.numeric_words();
    if (w.words < NN)
        throw nb::value_error("mymyr: numeric states have no numeric block");
    const u32 atom_words = w.words - NN;
    std::vector<std::shared_ptr<void>> keep{in.im.keep};
    const i32* ids = device_task_ids(a.task_ids, *c, os, w.rows, keep);
    const auto [fw, enc] = output_kind(in, w, a.framework);
    const rl::ExpandOptions opt{a.canonical, a.witness, a.validate, a.drop_noops};
    const std::optional<u64> fixed_cap = opt_int_arg<u64>(a.capacity, "capacity", 0, rl::k_max_rows);
    const std::optional<u32> fixed_words = opt_int_arg<u32>(a.words, "words", 0);
    const std::optional<u32> k = opt_int_arg<u32>(a.k, "k", 1);
    const u32 L = tt.label_width();
    DeviceFlat f;
    {
        ExpanderLease ex(ref, c, os.s);
        nb::gil_scoped_release release;
        ex->set_stream(os.s);
        const u64 total = ex->count({w.data, w.rows, atom_words, w.stride, NN}, ids, opt);
        const u64 cap = fixed_cap.value_or(std::min(total, rl::k_max_rows));  // more: the write raises
        u32 W = fixed_words.value_or(std::max(atom_words, tt.words()));
        if (W == 0)
            W = 1;
        for (int attempt = 0;; ++attempt)
        {
            f = allocate_device_flat(*c, os, w.rows, cap, W, NN, L, a.goal,
                                     a.derived ? std::optional<u32>(tt.derived_words()) : std::nullopt, a.hashes, a.noops, a.drop_noops);
            rl::Expansion x = f.view();
            ex->write(x);
            f.total = x.total;
            f.generated = x.generated;
            f.words_needed = x.words_needed;
            if (x.words_needed <= W || fixed_words || attempt >= 3)
                break;
            W = x.words_needed;
        }
    }
    PyDeviceExpansion e;
    if (ids && tt.size() > 1 && w.rows)
    {
        // a copy of the task ids for action() (the caller may write its array later)
        e.task_ids = device_output(*c, w.rows * sizeof(i32), os);
        cuda::DeviceGuard g(c->ctx->device());
        cuda::check(cudaMemcpyAsync(e.task_ids->data(), ids, w.rows * sizeof(i32), cudaMemcpyDeviceToDevice, os.s),
                    "cudaMemcpyAsync");
    }
    c->release_after(os.s, std::move(keep));
    e.flat = std::move(f);
    e.ctx = c;
    e.fw = fw;
    e.enc = enc;
    e.ref = std::move(ref);
    if (k)
        e.padded = make_device_padded(e.ref, c, e.flat, *k, fw, enc);
    return nb::cast(std::move(e), nb::rv_policy::move);
}

/// A destination of expand_into on the device: typed pointer after checking dtype, rank, contiguity and writability.
template<class T>
T* device_dest(const DeviceInput& d, const char* name, std::initializer_list<std::pair<u8, u8>> dtypes, usize ndim)
{
    const dl::Imported& im = d.im;
    bool ok = false;
    for (const auto& [code, bits] : dtypes)
        ok |= im.dtype.code == code && im.dtype.bits == bits && im.dtype.lanes == 1;
    if (!ok)
        throw nb::type_error((std::string("mymyr: '") + name + "' has the wrong dtype").c_str());
    if (im.shape.size() != ndim)
        throw nb::type_error((std::string("mymyr: '") + name + "' has the wrong number of dimensions").c_str());
    // An array without elements is contiguous whatever its strides (torch gives a (N, 0) tensor strides (1, 1)).
    const bool empty = std::find(im.shape.begin(), im.shape.end(), 0) != im.shape.end();
    i64 expect = 1;
    for (usize k = ndim; !empty && k-- > 0;)
    {
        if (im.shape[k] > 1 && im.strides[k] != expect)
            throw nb::type_error((std::string("mymyr: '") + name + "' must be C-contiguous").c_str());
        expect *= im.shape[k];
    }
    if (im.readonly)
        throw nb::type_error((std::string("mymyr: '") + name + "' is read-only").c_str());
    return static_cast<T*>(im.data);
}

nb::dict device_expand_into(const ExpandIntoArgs& a)
{
    const SuiteRef ref = suite_of(a.table);
    if (!is_cuda_array(a.states))
        throw nb::type_error("mymyr: expand_into: device destinations need device states (CUDA arrays on the same "
                             "device); host states take host destinations");
    const dl::Device dev = dl::dlpack_device(a.states);
    PyContextPtr c = table_context(ref, a.ctx, dev.device_id);
    const OpStream os = c->stream_of(a.stream);
    cudaStreamCaptureStatus capture;
    {
        cuda::DeviceGuard guard(dev.device_id);
        cuda::check(cudaStreamIsCapturing(os.s, &capture), "cudaStreamIsCapturing");
    }
    if (capture == cudaStreamCaptureStatusNone)
        c->drain();
    else if (a.status.is_none())
        throw nb::value_error("mymyr: expand_into under CUDA graph capture needs status= and stream=");
    ImportSync sync;
    DeviceInput in = import_device(a.states, *c, os, "states", &sync);
    const WordsLayout w = device_words(in);
    std::vector<std::shared_ptr<void>> keep{in.im.keep};
    const i32* ids = device_task_ids(a.task_ids, *c, os, w.rows, keep);
    rl::Expansion x;
    const u32 NN = ref.suite->numeric_words();
    if (w.words < NN)
        throw nb::value_error("mymyr: numeric states have no numeric block");
    x.numeric_words = NN;
    x.words = w.words - NN;
    std::optional<u64> cap;
    auto take_cap = [&](i64 n) { cap = cap ? std::min<u64>(*cap, static_cast<u64>(n)) : static_cast<u64>(n); };
    auto dest = [&](nb::handle obj, const char* name) {
        if (!is_cuda_array(obj))
            throw nb::type_error((std::string("mymyr: expand_into: '") + name +
                                  "' must be a CUDA device array like the states (device states write device destinations)")
                                     .c_str());
        DeviceInput d = import_device(obj, *c, os, name, &sync);
        keep.push_back(d.im.keep);
        return d;
    };
    constexpr std::pair<u8, u8> i32t{dl::k_int, 32}, u32t{dl::k_uint, 32}, i64t{dl::k_int, 64}, u64t{dl::k_uint, 64},
        u8t{dl::k_uint, 8}, boolt{dl::k_bool, 8};
    if (!a.succ.is_none())
    {
        DeviceInput d = dest(a.succ, "succ");
        if (d.im.shape.size() != 2)
            throw nb::type_error("mymyr: 'succ' must be [capacity, W] (64-bit) or [capacity, 2W] (32-bit)");
        if (d.im.dtype.bits == 64)
        {
            x.succ = device_dest<u64>(d, "succ", {u64t, i64t}, 2);
            x.words = static_cast<u32>(d.im.shape[1]);
        }
        else if (d.im.dtype.bits == 32)
        {
            x.succ = reinterpret_cast<u64*>(device_dest<u32>(d, "succ", {u32t, i32t}, 2));
            if (d.im.shape[1] % 2 != 0 || reinterpret_cast<std::uintptr_t>(d.im.data) % 8 != 0)
                throw nb::type_error("mymyr: 32-bit 'succ' needs an even number of columns and 8-byte alignment");
            x.words = static_cast<u32>(d.im.shape[1] / 2);
        }
        else
            throw nb::type_error("mymyr: 'succ' must hold 64-bit or 32-bit integers");
        if (x.words < NN)
            throw nb::value_error("mymyr: numeric successor rows have no numeric block");
        x.words -= NN;
        take_cap(d.im.shape[0]);
    }
    if (!a.parent.is_none())
    {
        DeviceInput d = dest(a.parent, "parent");
        x.parent = device_dest<i32>(d, "parent", {i32t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.schema.is_none())
    {
        DeviceInput d = dest(a.schema, "schema");
        x.schema = device_dest<i32>(d, "schema", {i32t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.binding.is_none())
    {
        DeviceInput d = dest(a.binding, "binding");
        x.binding = device_dest<i32>(d, "binding", {i32t}, 2);
        x.label_width = static_cast<u32>(d.im.shape[1]);
        take_cap(d.im.shape[0]);
    }
    if (!a.goal.is_none())
    {
        DeviceInput d = dest(a.goal, "goal");
        x.goal = device_dest<u8>(d, "goal", {boolt, u8t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.derived.is_none())
    {
        DeviceInput d = dest(a.derived, "derived");
        if (d.im.shape.size() != 2)
            throw nb::type_error("mymyr: 'derived' must be [capacity, D] (64-bit) or [capacity, 2D] (32-bit)");
        if (d.im.dtype.bits == 64)
        {
            x.derived = device_dest<u64>(d, "derived", {u64t, i64t}, 2);
            x.derived_words = static_cast<u32>(d.im.shape[1]);
        }
        else if (d.im.dtype.bits == 32)
        {
            x.derived = reinterpret_cast<u64*>(device_dest<u32>(d, "derived", {u32t, i32t}, 2));
            if (d.im.shape[1] % 2 != 0 || reinterpret_cast<std::uintptr_t>(d.im.data) % 8 != 0)
                throw nb::type_error("mymyr: 32-bit 'derived' needs an even number of columns and 8-byte alignment");
            x.derived_words = static_cast<u32>(d.im.shape[1] / 2);
        }
        else
            throw nb::type_error("mymyr: 'derived' must hold 64-bit or 32-bit integers");
        // an array without elements may carry no data pointer: the width is checked here, not by the expansion
        if (x.derived_words < ref.suite->derived_words())
            throw nb::value_error(("mymyr: expand_into: 'derived' rows of " + std::to_string(x.derived_words) +
                                   " words; the table has instances of " + std::to_string(ref.suite->derived_words()))
                                      .c_str());
        take_cap(d.im.shape[0]);
    }
    if (!a.succ_hash.is_none())
    {
        DeviceInput d = dest(a.succ_hash, "succ_hash");
        x.succ_hash = device_dest<u64>(d, "succ_hash", {u64t, i64t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.noop.is_none())
    {
        DeviceInput d = dest(a.noop, "noop");
        x.noop = device_dest<u8>(d, "noop", {boolt, u8t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.index.is_none())
    {
        DeviceInput d = dest(a.index, "index");
        x.index = device_dest<i32>(d, "index", {i32t}, 1);
        take_cap(d.im.shape[0]);
    }
    if (!a.offsets.is_none())
    {
        DeviceInput d = dest(a.offsets, "offsets");
        x.offsets = device_dest<i32>(d, "offsets", {i32t}, 1);
        if (static_cast<u64>(d.im.shape[0]) < w.rows + 1)
            throw nb::value_error("mymyr: 'offsets' needs N + 1 entries");
    }
    const u32* rows_dev = nullptr;
    if (!a.row_count.is_none())
    {
        if (a.status.is_none())
            throw nb::value_error("mymyr: expand_into: row_count requires status=");
        DeviceInput d = dest(a.row_count, "row_count");
        rows_dev = device_dest<u32>(d, "row_count", {i32t, u32t}, 1);
        if (d.im.shape[0] != 1)
            throw nb::value_error("mymyr: expand_into: row_count needs one entry");
    }
    x.capacity = cap.value_or(0);
    const rl::ExpandOptions opt{a.canonical, a.witness, a.validate, a.drop_noops};
    const rl::StateBatchView view{w.data, w.rows, w.words - NN, w.stride, NN};
    if (!a.status.is_none())
    {
        DeviceInput d = dest(a.status, "status");
        auto* status = device_dest<u32>(d, "status", {i32t, u32t}, 1);
        if (d.im.shape[0] < 1)
            throw nb::value_error("mymyr: expand_into: 'status' needs one entry");
        if (!x.offsets)
            throw nb::value_error("mymyr: expand_into: an expansion with 'status' needs 'offsets' (they carry its total)");
        // nothing waits for the outputs, so they are written in the order of the stream that reads them
        if (a.stream.is_none())
            throw nb::value_error("mymyr: expand_into: an expansion with 'status' does not wait for the device: pass "
                                  "stream=, the stream that reads its outputs (for instance torch.cuda.current_stream())");
        const u64 outputs = (x.succ ? 1u : 0u) | (x.parent ? 2u : 0u) | (x.schema ? 4u : 0u) |
                            (x.binding ? 8u : 0u) | (x.goal ? 16u : 0u) | (x.derived ? 32u : 0u) |
                            (x.succ_hash ? 64u : 0u) | (x.noop ? 128u : 0u) | (x.index ? 256u : 0u) |
                            (a.drop_noops ? 512u : 0u);
        const ExpansionShape shape{w.rows, w.words, w.stride, x.capacity, x.words, x.label_width, x.derived_words,
                                   outputs, ids ? 1u : 0u, a.canonical, a.witness, a.validate, rows_dev ? 1u : 0u};
        bool waited = false;
        std::shared_ptr<ExpansionWorkspace> owner;
        if (capture != cudaStreamCaptureStatusNone)
        {
            unsigned long long id = 0;
            cuda::check(cudaStreamGetCaptureInfo(os.s, &capture, &id), "cudaStreamGetCaptureInfo");
            owner = captured_workspace(ref, c, os.s, id, &shape);
            nb::gil_scoped_release release;
            owner->expander->expand_async(view, ids, x, status, opt, rows_dev);
        }
        else
        {
            ExpanderLease ex(ref, c, os.s);
            nb::gil_scoped_release release;
            ex->set_stream(os.s);
            if (ex->async_unsupported(w.rows, opt).empty())
            {
                ex->expand_async(view, ids, x, status, opt, rows_dev);
                ex.warmed(shape);
            }
            else
            {
                if (rows_dev)
                    throw nb::value_error("mymyr: expand_into: row_count requires an asynchronous device expansion");
                ex->expand(view, ids, x, opt);
                const u32 flags = (x.overflow() ? 1u : 0u) | (x.words_needed > x.words ? 2u : 0u);
                cuda::check(cudaMemcpyAsync(status, &flags, sizeof(u32), cudaMemcpyHostToDevice, os.s), "cudaMemcpyAsync");
                cuda::check(cudaStreamSynchronize(os.s), "cudaStreamSynchronize");  // the pageable source
                waited = true;
            }
        }
        if (capture == cudaStreamCaptureStatusNone)
            c->release_after(os.s, std::move(keep));
        nb::dict r;
        if (owner)
            r["_workspace"] = workspace_reference(owner);
        if (waited)
        {
            r["total"] = x.total;
            r["generated"] = x.generated;
            r["words_needed"] = x.words_needed;
            r["overflow"] = x.overflow();
        }
        r["capacity"] = x.capacity;
        return r;
    }
    {
        ExpanderLease ex(ref, c, os.s);
        nb::gil_scoped_release release;
        ex->set_stream(os.s);
        ex->expand(view, ids, x, opt);
    }
    c->release_after(os.s, std::move(keep));
    nb::dict r;
    r["total"] = x.total;
    r["generated"] = x.generated;
    r["words_needed"] = x.words_needed;
    r["overflow"] = x.overflow();
    r["capacity"] = x.capacity;
    return r;
}

nb::object device_hash_rows(nb::handle states, unsigned numeric_words, nb::handle stream)
{
    const int device = dl::dlpack_device(states).device_id;
    static std::mutex mutex;
    static std::map<int, PyContextPtr> contexts;
    PyContextPtr c;
    {
        std::lock_guard lock(mutex);
        auto& entry = contexts[device];
        if (!entry) entry = std::make_shared<PyContext>(cuda::DeviceContext::create(device));
        c = entry;
    }
    const OpStream os = c->stream_of(stream);
    cudaStreamCaptureStatus capture;
    {
        cuda::DeviceGuard guard(device);
        cuda::check(cudaStreamIsCapturing(os.s, &capture), "cudaStreamIsCapturing");
    }
    if (capture == cudaStreamCaptureStatusNone) c->drain();
    DeviceInput in = import_device(states, *c, os, "states");
    const WordsLayout w = device_words(in);
    if (numeric_words > w.words)
        throw nb::value_error("mymyr: hash_rows: numeric_words exceeds the row width");
    const auto [fw, enc] = output_kind(in, w, nb::none());
    auto buf = device_output(*c, w.rows * sizeof(u64), os);
    {
        nb::gil_scoped_release release;
        cuda::DeviceGuard guard(device);
        cuda::check(cuda::launch_hash_rows(w.data, w.rows, w.stride, w.words - numeric_words, numeric_words,
                                          static_cast<u64*>(buf->data()), os.s), "launch_hash_rows");
    }
    if (capture == cudaStreamCaptureStatusNone)
        c->release_after(os.s, {in.im.keep});
    std::vector<i64> shape{static_cast<i64>(w.rows)};
    if (fw == Framework::Jax) shape.push_back(1);
    return export_region(buf, 0, os, fw == Framework::Torch ? rl::DType::I64 : rl::DType::U64,
                         std::move(shape), fw == Framework::Jax, fw, enc);
}

// ------------------------------------------------------------------------------------------------ action masks

/// The device masks of (suite, context), cached on the table; null when the kernels do not run the suite (`why`).
std::shared_ptr<const cuda::DeviceMasks> table_masks(const SuiteRef& ref, const PyContextPtr& c,
                                                     std::string* why)
{
    std::lock_guard lock(ref.cuda_mutex());
    TaskCuda& tc = slot_cuda(ref.cuda());
    if (!tc.masks_why)
        tc.masks_why = cuda::DeviceMasks::unsupported(*ref.suite);
    if (why)
        *why = *tc.masks_why;
    if (!tc.masks_why->empty())
        return nullptr;
    std::shared_ptr<const cuda::DeviceMasks>& m = tc.masks[c->ctx.get()];
    if (!m)
        m = std::make_shared<const cuda::DeviceMasks>(c->ctx, ref.suite);
    return m;
}

/// An action-mask call over device rows: the arguments of DeviceMasks plus the numeric words the host functions read.
struct MaskCall
{
    MaskKind kind = MaskKind::Schemas;
    cuda::DeviceMasks::Batch b;
    u32 numeric_words = 0;
    cuda::DeviceMasks::Schemas schema;
    const i32* labels = nullptr;  // the prefix (objects) or the binding (applicable), [rows, width]
    u32 width = 0;
    u32 depth = 0;
    u32 num_objects = 0;
    u8* out = nullptr;
    u64* views = nullptr;  // [rows, view_words] (schemas and objects on the device)
    u32* status = nullptr;

    [[nodiscard]] u64 columns(const rl::TaskSuite& suite) const
    {
        return kind == MaskKind::Schemas ? suite.max_schemas() : kind == MaskKind::Objects ? num_objects : 1;
    }
};

/// The host functions' checks that read no rows, with their errors.
void check_mask_call(const MaskCall& m, const rl::TaskSuite& suite)
{
    if (m.kind == MaskKind::Objects)
    {
        if (m.num_objects < suite.max_objects())
            throw std::invalid_argument("mymyr: parameter_masks: num_objects is below the suite's maximum");
        if (m.width < m.depth || (m.b.rows && m.depth && !m.labels))
            throw std::invalid_argument("mymyr: parameter_masks: prefix needs at least depth columns");
    }
    if (m.kind == MaskKind::Applicable && m.b.rows && m.width && !m.labels)
        throw std::invalid_argument("mymyr: is_applicable: null label or output buffer");
}

/// The host functions over host copies of the call's device inputs, the result copied into m.out: the path of tasks
/// the device masks do not run, and of calls whose status reports an error (the host function raises it). Waits for s.
void host_masks(const rl::TaskSuite& suite, const MaskCall& m, cudaStream_t s)
{
    const u64 n = m.b.rows;
    const u64 row = u64{m.b.words} + m.numeric_words;
    std::vector<u64> rows(n ? (n - 1) * m.b.stride + row : 0);
    std::vector<i32> ids(m.b.task_ids ? n : 0), schema(n, m.schema.value), labels(m.labels ? n * m.width : 0);
    auto get = [&](void* dst, const void* src, u64 bytes)
    {
        if (bytes)
            cuda::check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    };
    get(rows.data(), m.b.states, rows.size() * sizeof(u64));
    get(ids.data(), m.b.task_ids, ids.size() * sizeof(i32));
    if (m.schema.schema)
        get(schema.data(), m.schema.schema, n * sizeof(i32));
    get(labels.data(), m.labels, labels.size() * sizeof(i32));
    cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    const rl::StateBatchView view{rows.data(), n, m.b.words, m.b.stride, m.numeric_words};
    const i32* id = m.b.task_ids ? ids.data() : nullptr;
    const i32* lab = m.labels ? labels.data() : nullptr;
    std::vector<u8> out(n * m.columns(suite));
    switch (m.kind)
    {
    case MaskKind::Schemas:
        rl::schema_masks(suite, view, id, out.data(), nullptr);
        break;
    case MaskKind::Objects:
        suite.check_task_ids(id, n);
        if (id || suite.size() == 1)
            for (u64 i = 0; i < n; ++i)
            {
                const auto& plans = suite.task_of(id, i).compiled().schemas;
                if (schema[i] >= 0 && static_cast<u32>(schema[i]) < plans.size() && m.depth >= plans[schema[i]].arity)
                    throw std::invalid_argument("mymyr: depth must be below the schema arity");
            }
        rl::parameter_masks(suite, view, id, schema.data(), lab, m.width, m.depth, out.data(), m.num_objects, nullptr);
        break;
    case MaskKind::Applicable:
        rl::is_applicable(suite, view, id, schema.data(), lab, m.width, out.data(), nullptr);
        break;
    }
    if (out.empty())
        return;
    cuda::check(cudaMemcpyAsync(m.out, out.data(), out.size(), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

/// Runs a mask call on stream s: on the device masks dm, or the host functions without them (they wait for s and raise
/// their errors; `why` the device masks do not run the suite; refused while s is capturing). On the device, errors are
/// status bits (*m.status) and nothing waits; with `raise` the call waits for the status and raises the host
/// function's error for a nonzero one.
void run_masks(const rl::TaskSuite& suite, const cuda::DeviceMasks* dm, const std::string& why, const MaskCall& m,
               cudaStream_t s, bool raise)
{
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!dm)
        cuda::check(cudaStreamIsCapturing(s, &capture), "cudaStreamIsCapturing");
    if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("mymyr: action masks: this " + std::string(suite.noun()) + " runs on the host (" +
                                    why + "), which cannot be captured into a CUDA graph");
    check_mask_call(m, suite);
    if (!dm)
    {
        if (m.status)
            cuda::check(cudaMemsetAsync(m.status, 0, sizeof(u32), s), "cudaMemsetAsync");
        host_masks(suite, m, s);
        return;
    }
    switch (m.kind)
    {
    case MaskKind::Schemas:
        dm->schemas(m.b, m.out, m.views, m.status, s);
        break;
    case MaskKind::Objects:
        dm->objects(m.b, m.schema, m.labels, m.width, m.depth, m.out, m.num_objects, m.views, m.status, s);
        break;
    case MaskKind::Applicable:
        dm->applicable(m.b, m.schema, m.labels, m.width, m.out, m.status, s);
        break;
    }
    if (!raise)
        return;
    u32 status = 0;
    cuda::check(cudaMemcpyAsync(&status, m.status, sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    if (status)
        host_masks(suite, m, s);
}

/// An int32 CUDA array [rows] (matrix: [rows, cols], *cols set) read in place on the operation's stream.
const i32* device_i32(nb::handle obj, const PyContext& c, const OpStream& os, u64 rows, bool matrix, u32* cols,
                      std::vector<std::shared_ptr<void>>& keep, const char* name)
{
    DeviceInput in = import_device(obj, c, os, name);
    const dl::Imported& im = in.im;
    const std::string q = std::string("mymyr: '") + name + "'";
    if (im.dtype.code != dl::k_int || im.dtype.bits != 32 || im.dtype.lanes != 1)
        throw nb::type_error((q + " must be an int32 CUDA array like the states").c_str());
    if (im.shape.size() != (matrix ? 2u : 1u) || static_cast<u64>(im.shape[0]) != rows)
        throw nb::value_error((q + " must have shape [" + std::to_string(rows) + (matrix ? ", L]" : "]")).c_str());
    const bool empty = std::find(im.shape.begin(), im.shape.end(), 0) != im.shape.end();
    i64 expect = 1;
    for (usize d = im.shape.size(); !empty && d-- > 0;)
    {
        if (im.shape[d] > 1 && im.strides[d] != expect)
            throw nb::value_error((q + " must be C-contiguous").c_str());
        expect *= im.shape[d];
    }
    if (matrix)
        *cols = static_cast<u32>(im.shape[1]);
    keep.push_back(im.keep);
    return static_cast<const i32*>(im.data);
}

/// rl.applicable_schemas / rl.applicable_objects / rl.is_applicable on CUDA arrays: the device masks on the table's
/// context stream, the host functions for tasks the kernels do not run; waits for the result and raises the host
/// functions' errors.
nb::object device_action_masks(const MaskArgs& a)
{
    SuiteRef ref = suite_of(a.table);
    const rl::TaskSuite& tt = *ref.suite;
    PyContextPtr c = table_context(ref, nb::none(), dl::dlpack_device(a.states).device_id);
    c->drain();
    const OpStream os = c->stream_of(nb::none());
    DeviceInput in = import_device(a.states, *c, os, "states");
    const WordsLayout w = device_words(in);
    const u32 NN = tt.numeric_words();
    if (w.words < NN)
        throw nb::value_error("mymyr: numeric states have no numeric block");
    std::vector<std::shared_ptr<void>> keep{in.im.keep};
    MaskCall m;
    m.kind = a.kind;
    m.b = {static_cast<const u64*>(w.data), w.rows, w.words - NN, w.stride, nullptr};
    m.numeric_words = NN;
    m.b.task_ids = device_task_ids(a.task_ids, *c, os, w.rows, keep);
    if (a.kind != MaskKind::Schemas)
    {
        if (a.schema.is_none())
            throw nb::type_error("mymyr: schema must be an int or an int32 array [N]");
        if (PyIndex_Check(a.schema.ptr()) && dl::dlpack_device(a.schema).device_type < 0)
            m.schema.value = int_arg<i32>(a.schema, "schema");
        else
            m.schema.schema = device_i32(a.schema, *c, os, w.rows, false, nullptr, keep, "schema");
    }
    if (a.kind == MaskKind::Objects)
    {
        m.depth = int_arg<u32>(a.depth, "depth");
        m.num_objects = opt_int_arg<u32>(a.num_objects, "num_objects").value_or(tt.max_objects());
    }
    if (a.kind == MaskKind::Applicable && a.labels.is_none())
        throw nb::type_error("mymyr: binding must be an int32 array [N, L]");
    if (!a.labels.is_none())
        m.labels = device_i32(a.labels, *c, os, w.rows, true, &m.width, keep,
                              a.kind == MaskKind::Objects ? "prefix" : "binding");
    std::string why;
    const std::shared_ptr<const cuda::DeviceMasks> dm = table_masks(ref, c, &why);
    const u64 cols = m.columns(tt);
    auto out = device_output(*c, checked_mul({w.rows, cols}, "action masks"), os);
    auto status = device_output(*c, sizeof(u32), os);
    std::shared_ptr<cuda::DeviceBuffer> views;
    if (dm && a.kind != MaskKind::Applicable)
        views = device_output(*c, checked_mul({w.rows, dm->view_words(), sizeof(u64)}, "action masks"), os);
    m.out = static_cast<u8*>(out->data());
    m.status = static_cast<u32*>(status->data());
    m.views = views ? static_cast<u64*>(views->data()) : nullptr;
    {
        nb::gil_scoped_release release;
        cuda::DeviceGuard g(c->ctx->device());
        run_masks(tt, dm.get(), why, m, os.s, true);
    }
    c->release_after(os.s, std::move(keep));
    const auto [fw, enc] = output_kind(in, w, nb::none());
    std::vector<i64> shape{static_cast<i64>(w.rows)};
    if (a.kind != MaskKind::Applicable)
        shape.push_back(static_cast<i64>(cols));
    return export_buffer(out, os, rl::DType::Bool, std::move(shape), false, fw, enc);
}

// ------------------------------------------------------------------------------------------------ device BrFS

struct PyDeviceBrfs
{
    std::shared_ptr<cuda::DeviceBuffer> numeric_rows;
    std::shared_ptr<cuda::DeviceBrfs> b;
    cuda::DeviceBrfsResult r;
    Owner o;
    PyContextPtr ctx;
    double cost = 0;                  // of the plan, by replaying it on the CPU (0 unless solved)
    std::optional<State> goal_state;  // the state the plan reaches (unless solved: none)
};

nb::object action_object(const Owner& o, const Action& a)
{
    PyAction p;
    p.schema = a.schema.v;
    p.binding.resize(a.binding.size());
    for (usize i = 0; i < a.binding.size(); ++i)
        p.binding[i] = a.binding[i].v;
    p.owner = o.obj;
    p.core = o.core;
    return nb::cast(std::move(p), nb::rv_policy::move);
}

/// Rows [0, n) of a device arena as a zero-copy device array (keeps the arena's generation alive).
nb::object arena_array(const PyDeviceBrfs& x, const cuda::DeviceArena& a, rl::DType dtype, i64 cols, bool words, nb::handle framework)
{
    const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
    const auto& gen = a.generation();
    ArraySpec s;
    s.owner = std::shared_ptr<const void>(gen, a.device_data());
    s.data = a.device_data();
    s.dtype = dtype;
    s.shape = {static_cast<i64>(a.device_size()), cols};
    s.readonly = true;
    s.words = words;
    return export_device_array(std::move(s), DeviceExport{dl::k_cuda, x.ctx->ctx->device(), std::make_shared<BufferSync>(gen, a.stream())},
                               fw, default_words(fw));
}

// ------------------------------------------------------------------------------------------------ argument types
// (typing.hpp, py_task.hpp: rendered in the stubs; the functions check their arguments themselves)
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using ContextArg = Arg<PyContextObj>;
using MemoryArg = Arg<ann::Memory>;
using ArrayArg = Arg<ann::ArrayLike>;
using SizeArg = Arg<u64>;
using ArrayOut = Arg<ann::Any>;
using MaybeArray = Arg<std::optional<ann::Any>>;
using ArrayPair = nb::typed<nb::tuple, ann::Any, ann::Any>;
using ArrayDict = nb::typed<nb::dict, std::string, ann::Any>;
using ActionList = nb::typed<nb::list, PyAction>;
}  // namespace

namespace
{
struct TranslateUpload
{
    cuda::ContextPtr context;
    cuda::TranslateView view;
    std::vector<std::shared_ptr<cuda::DeviceBuffer>> buffers;
    cuda::Event ready;
    std::mutex use_mutex;
    std::map<cudaStream_t, std::shared_ptr<cuda::LastUse>> uses;
};
}

nb::object translate_device(PySlotMap& map, nb::handle rows, nb::handle task_ids, bool drop, nb::handle status,
                            nb::handle stream, nb::handle ctx)
{
    const dl::Device dev = dl::dlpack_device(rows);
    const bool table = nb::isinstance<PyTable>(map.source);
    PyTaskCore* core = table ? nullptr : owner_of(map.source).core;
    PyTable* py = table ? nb::inst_ptr<PyTable>(map.source) : nullptr;
    auto& mutex = table ? py->cuda_mutex : core->cuda_mutex;
    auto& slot = table ? py->cuda : core->cuda;
    PyContextPtr c = slot_context(mutex, slot, ctx, dev.device_id);
    cuda::DeviceGuard guard(dev.device_id);
    const OpStream os = c->stream_of(stream);
    cudaStreamCaptureStatus capture;
    cuda::check(cudaStreamIsCapturing(os.s, &capture), "translate capture status");
    if (capture != cudaStreamCaptureStatusNone && (status.is_none() || stream.is_none()))
        throw nb::value_error("mymyr: translate under CUDA graph capture needs status= and stream=");
    if (capture == cudaStreamCaptureStatusNone)
        c->drain();
    std::vector<std::shared_ptr<const std::vector<u32>>> snapshots;
    std::vector<u64> key{reinterpret_cast<std::uintptr_t>(c->ctx.get())};
    for (u32 i = 0; i < map.maps.size(); ++i)
    {
        if (!map.targets[i]->atoms().frozen())
            throw nb::value_error("mymyr: translate: a device target must have fixed atom slots (device rows cannot "
                                  "intern lazy slots; translate into target.frozen() or translate host rows)");
        snapshots.push_back(map.maps[i]->snapshot());
        key.push_back(map.targets[i]->uid());
        key.push_back(snapshots.back()->size());
    }
    std::shared_ptr<TranslateUpload> upload;
    {
        std::lock_guard lock(mutex);
        auto& cache = slot_cuda(slot).translations;
        auto it = cache.find(key);
        if (it != cache.end())
            upload = it->second;
        else
        {
            if (capture != cudaStreamCaptureStatusNone)
                throw nb::value_error("mymyr: translate: warm the slot map on this context before capture");
            upload = std::make_shared<TranslateUpload>();
            upload->context = c->ctx;
            std::vector<cuda::TranslateInstance> instances;
            std::vector<u32> slots, numeric;
            std::vector<u64> constants, initial;
            for (u32 i = 0; i < map.maps.size(); ++i)
            {
                const auto& src = *map.sources[i];
                const auto& dst = *map.targets[i];
                cuda::TranslateInstance inst;
                inst.slots_begin = static_cast<u32>(slots.size());
                inst.slots = static_cast<u32>(snapshots[i]->size());
                inst.numeric_begin = static_cast<u32>(numeric.size());
                inst.numeric_slots = src.numeric_slots();
                inst.constant_begin = static_cast<u32>(constants.size());
                inst.initial_begin = static_cast<u32>(initial.size());
                inst.target_slots = dst.atoms().fluent_slots();
                inst.target_numeric_slots = dst.numeric_slots();
                inst.missing_constants = static_cast<u32>(map.maps[i]->missing_constants().size());
                inst.source_words = bits::words_for(inst.slots);
                inst.target_words = dst.words();
                inst.source_i32 = src.numeric_storage() == NumericStorage::I32;
                inst.target_i32 = dst.numeric_storage() == NumericStorage::I32;
                inst.quantum = dst.numeric().quantum;
                inst.tolerant = dst.numeric().tolerant;
                instances.push_back(inst);
                slots.insert(slots.end(), snapshots[i]->begin(), snapshots[i]->end());
                const auto& ns = map.maps[i]->numeric_slots();
                numeric.insert(numeric.end(), ns.begin(), ns.end());
                constants.resize(constants.size() + dst.words(), 0);
                const auto& fixed = map.maps[i]->constant_words();
                std::copy(fixed.begin(), fixed.end(), constants.begin() + inst.constant_begin);
                const State init = dst.initial_state();
                initial.insert(initial.end(), init.numeric().begin(), init.numeric().end());
            }
            const auto copy = [&](const auto& data) -> const void* {
                const u64 bytes = data.size() * sizeof(typename std::decay_t<decltype(data)>::value_type);
                auto buffer = std::make_shared<cuda::DeviceBuffer>(c->ctx, std::max<u64>(bytes, 1));
                if (bytes)
                    cuda::check(cudaMemcpyAsync(buffer->data(), data.data(), bytes, cudaMemcpyHostToDevice, c->ctx->stream()),
                                "translate map upload");
                const void* ptr = buffer->data();
                upload->buffers.push_back(std::move(buffer));
                return ptr;
            };
            upload->view.instances = static_cast<const cuda::TranslateInstance*>(copy(instances));
            upload->view.slots = static_cast<const u32*>(copy(slots));
            upload->view.numeric = static_cast<const u32*>(copy(numeric));
            upload->view.constants = static_cast<const u64*>(copy(constants));
            upload->view.initial = static_cast<const u64*>(copy(initial));
            upload->view.count = static_cast<u32>(instances.size());
            upload->ready.record(c->ctx->stream());
            upload->ready.synchronize();
            // Outputs retain older uploads while captured graphs still reference them.
            for (auto prior = cache.begin(); prior != cache.end();)
            {
                bool same_pair = prior->first.size() == key.size() && prior->first[0] == key[0];
                for (usize k = 1; same_pair && k < key.size(); k += 2)
                    same_pair = prior->first[k] == key[k];
                if (same_pair)
                    prior = cache.erase(prior);
                else
                    ++prior;
            }
            cache.emplace(key, upload);
        }
    }
    DeviceInput in = import_device(rows, *c, os, "states");
    const WordsLayout w = device_words(in);
    const u32 sn = table ? map.source_numeric : map.sources[0]->numeric_words();
    const u32 sw = table ? map.source_words : bits::words_for(static_cast<u32>(snapshots[0]->size()));
    if (w.words < sn || w.words - sn < sw || w.rows > 0x7fffffff)
        throw nb::value_error("mymyr: translate: source rows have the wrong width or too many rows");
    if (map.maps.size() > 1 && task_ids.is_none())
        throw nb::value_error("mymyr: translate: task_ids is required for a table with multiple instances");
    std::vector<std::shared_ptr<void>> keep{in.im.keep, upload};
    const i32* ids = device_task_ids(task_ids, *c, os, w.rows, keep);
    const u32 tw = table ? map.target_words : std::max<u32>(1, map.targets[0]->words());
    const u32 tn = table ? map.target_numeric : map.targets[0]->numeric_words();
    const u32 rw = tw + tn;
    auto output = device_output(*c, checked_mul({w.rows, rw, 8}, "translated device rows"), os);
    auto mask = device_output(*c, w.rows, os);
    auto internal_status = status.is_none() ? device_output(*c, 3 * sizeof(i32), os) : nullptr;
    i32* error = internal_status ? static_cast<i32*>(internal_status->data()) : nullptr;
    if (!status.is_none())
    {
        DeviceInput im = import_device(status, *c, os, "status");
        error = device_dest<i32>(im, "status", {{dl::k_int, 32}, {dl::k_uint, 32}}, 1);
        if (im.im.shape[0] != 3)
            throw nb::value_error("mymyr: translate: status must have shape [3]");
        keep.push_back(im.im.keep);
    }
    cuda::check(cuda::translate_rows(upload->view, w.data, w.rows, w.stride, w.words - sn, ids,
                                      static_cast<u64*>(output->data()), tw, rw,
                                      static_cast<u8*>(mask->data()), error, os.s), "translate_rows");
    {
        std::lock_guard lock(upload->use_mutex);
        auto& last = upload->uses[os.s];
        if (!last)
            last = std::make_shared<cuda::LastUse>(dev.device_id);
        last->record(os.s);
        for (auto& buffer : upload->buffers)
            buffer->retire_stream(os.s, last);
    }
    if (internal_status)
    {
        std::array<i32, 3> flags{};
        cuda::check(cudaMemcpyAsync(flags.data(), error, sizeof(flags), cudaMemcpyDeviceToHost, os.s), "translate status");
        cuda::check(cudaStreamSynchronize(os.s), "translate status wait");
        if (flags[0] & 2)
        {
            if (flags[2] < 0)
                throw nb::value_error(("mymyr: translate: task_ids[" + std::to_string(flags[1]) +
                                       "] is outside the source table").c_str());
            throw nb::value_error(("mymyr: translate: row " + std::to_string(flags[1]) + " sets an unknown source slot " +
                                   std::to_string(flags[2])).c_str());
        }
        if (flags[0] && !drop)
        {
            i32 id = 0;
            if (ids && flags[1] >= 0)
            {
                cuda::check(cudaMemcpyAsync(&id, ids + flags[1], sizeof(id), cudaMemcpyDeviceToHost, os.s), "translate error instance");
                cuda::check(cudaStreamSynchronize(os.s), "translate error instance wait");
            }
            std::string name = "a source value";
            if (id >= 0 && static_cast<usize>(id) < map.sources.size() && flags[2] >= 0)
            {
                const auto& src = *map.sources[static_cast<usize>(id)];
                const u32 slot_id = static_cast<u32>(flags[2]);
                const u32 fluent_slots = static_cast<u32>(snapshots[static_cast<usize>(id)]->size());
                if (slot_id < fluent_slots)
                    name = src.format(SlotId{slot_id});
                else if (slot_id - fluent_slots < src.numeric_slots())
                    name = src.numeric_name(slot_id - fluent_slots);
                else if (slot_id - fluent_slots - src.numeric_slots() < map.maps[static_cast<usize>(id)]->missing_constants().size())
                    name = map.maps[static_cast<usize>(id)]->missing_constants()[slot_id - fluent_slots - src.numeric_slots()];
            }
            throw nb::value_error(("mymyr: translate: row " + std::to_string(flags[1]) + ": " + name +
                                    " has no representable value in the target").c_str());
        }
    }
    struct ResultOwner
    {
        std::shared_ptr<cuda::DeviceBuffer> buffer;
        std::vector<std::shared_ptr<void>> keep;
    };
    const auto emit = [&](const std::shared_ptr<cuda::DeviceBuffer>& buffer, rl::DType dtype,
                           std::vector<i64> shape, bool words) {
        auto owner = std::make_shared<ResultOwner>(buffer, keep);
        ArraySpec spec{owner, buffer->data(), dtype, std::move(shape), {}, false, words};
        DeviceExport device{dl::k_cuda, dev.device_id, std::make_shared<BufferSync>(buffer, os.s)};
        return export_device_array(std::move(spec), std::move(device), in.fw, w.enc);
    };
    nb::object result = emit(output, rl::DType::U64, {static_cast<i64>(w.rows), rw}, true);
    if (drop)
        return nb::make_tuple(result, emit(mask, rl::DType::Bool, {static_cast<i64>(w.rows)}, false));
    return result;
}

// Multi-search IW, device rollouts and batched IW(1) (cuda_search_bindings.cpp); `lookup` is task_context's
// device context.
void bind_cuda_search(nb::module_& m, cuda::ContextPtr (*lookup)(PyTaskCore& core, nb::handle ctx, int device));
// Device heuristics, A* and GBFS (cuda_heuristics_bindings.cpp), with the same lookup.
void bind_cuda_heuristics(nb::module_& m, cuda::ContextPtr (*lookup)(PyTaskCore& core, nb::handle ctx, int device));

/// The device context of an operation on a table or suite (the device environments of rl_torch_bindings.cpp and
/// rl_jax_bindings.cpp, the table searches and state spaces): `ctx` (a mymyr.cuda.Context), or the table's (suite's)
/// default context on `device`.
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device)
{
    return table_context(suite_of(table), ctx, device)->ctx;
}

// ------------------------------------------------------------------------------------------------ bindings

void bind_cuda(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_cuda", "CUDA backend: contexts, device tasks, arenas, DLPack");

    m.def("device_count", [] { return cuda::device_count(); }, "The number of visible CUDA devices.");
    m.def("available", [] { return cuda::device_count() > 0; }, "True if a CUDA device is visible");

    nb::class_<PyContextObj>(m, "Context",
                             "A device context: the device, its own stream-ordered memory pool (max_bytes caps it), a "
                             "compute stream and a copy stream. Owned by the user; device tasks and arrays keep it alive.")
        .def(
            "__init__",
            [](PyContextObj* self, IntArg device_in, IntArg max_bytes_in, IntArg release_threshold_in) {
                const int device = int_arg<int>(device_in, "device", 0);
                const u64 max_bytes = int_arg<u64>(max_bytes_in, "max_bytes");
                const std::optional<u64> release_threshold = opt_int_arg<u64>(release_threshold_in, "release_threshold");
                cuda::ContextOptions o;
                o.max_bytes = max_bytes;
                if (release_threshold)
                    o.release_threshold = *release_threshold;  // (default: the pool keeps its memory until trim())
                new (self) PyContextObj{std::make_shared<PyContext>(cuda::DeviceContext::create(device, o))};
            },
            "device"_a = 0, "max_bytes"_a = 0, "release_threshold"_a = nb::none())
        .def_prop_ro("device", [](const PyContextObj& c) { return c.p->ctx->device(); }, "The CUDA device index.")
        .def_prop_ro("name", [](const PyContextObj& c) { return c.p->ctx->name(); }, "The device's name.")
        .def_prop_ro("compute_capability", [](const PyContextObj& c) { return c.p->ctx->compute_capability(); },
                     "The device's compute capability as major * 10 + minor (89 for 8.9).")
        .def_prop_ro("stream", [](const PyContextObj& c) { return reinterpret_cast<std::intptr_t>(c.p->ctx->stream()); },
                     "the compute stream (a cudaStream_t, usable as torch.cuda.ExternalStream(ctx.stream))")
        .def_prop_ro("copy_stream", [](const PyContextObj& c) { return reinterpret_cast<std::intptr_t>(c.p->ctx->copy_stream()); },
                     "The copy stream (a cudaStream_t) of the arenas' host syncs.")
        .def("usage", [](const PyContextObj& c) {
            const cuda::PoolUsage u = c.p->ctx->usage();
            nb::typed<nb::dict, std::string, u64> d{nb::dict()};
            d["reserved"] = u.reserved;
            d["used"] = u.used;
            d["reserved_high"] = u.reserved_high;
            d["used_high"] = u.used_high;
            return d;
        },
             "The memory pool's bytes: a dict of reserved, used and their high-water marks reserved_high, used_high.")
        .def("synchronize", [](const PyContextObj& c) {
            c.p->ctx->synchronize();
            c.p->drain(true);
        },
             "Blocks until the compute and copy streams are idle and releases the inputs held for their work.")
        .def("trim", [](const PyContextObj& c, IntArg keep_in) { const u64 keep = int_arg<u64>(keep_in, "keep"); c.p->ctx->trim(keep); }, "keep"_a = 0,
             "Returns reserved pool memory beyond `keep` bytes to the driver (after synchronizing the streams) and "
             "frees the cached pinned buffers.")
        .def_prop_ro("pending_imports", [](const PyContextObj& c) {
            c.p->drain();
            return c.p->pending();
        }, "inputs still held until the work that reads them completes");

    nb::class_<PyDeviceTask>(m, "DeviceTask",
                             "A task's device_arrays (version 2) uploaded once to the device; shareable across "
                             "streams and threads. DeviceTask(task, ctx=None): ctx None creates a context on device 0.")
        .def(
            "__init__",
            [](PyDeviceTask* self, TaskArg task, ContextArg ctx) {
                const Owner o = owner_of(task);
                PyContextPtr c = context_of(ctx);
                std::shared_ptr<const rl::ArrayBundle> bundle = o.core->device_arrays(rl::k_device_arrays_version);
                auto dt = cuda::DeviceTask::upload(c->ctx, bundle);
                new (self) PyDeviceTask{std::move(c), std::move(dt), o.core->task};
            },
            "task"_a, "ctx"_a = nb::none())
        .def_prop_ro("bytes", [](const PyDeviceTask& t) { return t.dt->bytes(); },
                     "Device bytes of the uploaded arrays.")
        .def_prop_ro("version", [](const PyDeviceTask& t) { return t.dt->version(); },
                     "The layout version of the device arrays.")
        .def_prop_ro("upload_seconds", [](const PyDeviceTask& t) { return t.dt->upload_seconds(); },
                     "Wall time of the upload, in seconds.")
        .def_prop_ro("upload_device_ms", [](const PyDeviceTask& t) { return t.dt->upload_device_ms(); },
                     "Device time of the upload copies, in milliseconds.")
        .def_prop_ro("context", [](const PyDeviceTask& t) { return PyContextObj{t.ctx}; },
                     "The Context the task lives in.")
        .def_prop_ro("data_ptr", [](const PyDeviceTask& t) { return reinterpret_cast<std::uintptr_t>(t.dt->device_block()); },
                     "The device address of the uploaded block (an integer).")
        .def(
            "download",
            [](const PyDeviceTask& t) {
                auto block = std::make_shared<std::vector<std::byte>>(t.dt->bytes());
                t.dt->download(block->data());
                ArraySpec s{std::shared_ptr<const void>(block, block->data()), block->data(), rl::DType::U8,
                            {static_cast<i64>(block->size())}, {}, false, false};
                return ArrayOut(export_array(std::move(s), Framework::Numpy, default_words(Framework::Numpy)));
            },
            "The device block copied back to the host (uint8 NumPy): the round-trip check")
        .def(
            "host_block",
            [](const PyDeviceTask& t) {
                const auto& b = t.dt->bundle_ptr();
                ArraySpec s{std::shared_ptr<const void>(b, b->block()), b->block(), rl::DType::U8, {static_cast<i64>(b->bytes())},
                            {}, true, false};
                return ArrayOut(export_array(std::move(s), Framework::Numpy, default_words(Framework::Numpy)));
            },
            "The uploaded host block (read-only uint8 NumPy view)")
        .def(
            "arrays",
            [](const PyDeviceTask& t, FrameworkArg framework) {
                const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
                const rl::ArrayBundle& b = t.dt->bundle();
                ArrayDict d{nb::dict()};
                for (const auto& [name, value] : b.scalars())
                    d[nb::str(name.data(), name.size())] = value;
                auto sync = std::make_shared<TaskSync>(t.dt);
                for (const rl::ArrayInfo& a : b.arrays())
                {
                    ArraySpec s;
                    const std::byte* p = t.dt->device_block() + a.offset;
                    s.owner = std::shared_ptr<const void>(t.dt, p);
                    s.data = p;
                    s.dtype = a.dtype;
                    s.shape = a.shape.vec();
                    s.readonly = true;
                    s.words = a.words;
                    d[nb::str(a.name.data(), a.name.size())] =
                        export_device_array(std::move(s), DeviceExport{dl::k_cuda, t.ctx->ctx->device(), sync}, fw, default_words(fw));
                }
                return d;
            },
            "framework"_a = nb::none(),
            "The uploaded arrays as zero-copy device arrays (torch, jax or dlpack), plus the scalars")
        .def(
            "validate",
            [](PyDeviceTask& t, StreamArg stream) {
                const OpStream os = t.ctx->stream_of(stream);
                const rl::dev::TaskView& v = t.dt->acquire(os.s);
                auto out = device_output(*t.ctx, 8, os);
                const u32 init[2] = {0, rl::dev::k_none};
                cuda::check(cudaMemcpyAsync(out->data(), init, 8, cudaMemcpyHostToDevice, os.s), "cudaMemcpyAsync");
                cuda::check(k::launch_validate(v, static_cast<u32*>(out->data()), os.s), "validate");
                u32 res[2];
                cuda::check(cudaMemcpyAsync(res, out->data(), 8, cudaMemcpyDeviceToHost, os.s), "cudaMemcpyAsync");
                cuda::check(cudaStreamSynchronize(os.s), "cudaStreamSynchronize");
                return std::pair<u32, u32>{res[0], res[0] ? res[1] : 0u};
            },
            "stream"_a = nb::none(), "(failed items, first failed rule) of the structural validation on the device")
        .def(
            "applicable",
            [](PyDeviceTask& t, ArrayArg states, ArrayArg state_index, ArrayArg schema, ArrayArg binding,
               ArrayArg derived, StreamArg stream, FrameworkArg framework) -> ArrayOut {
                SmokeInputs in = smoke_inputs(t, states, derived, stream);
                const k::DeviceLabels l = labels_of(t, in, state_index, schema, binding);
                const rl::dev::TaskView& v = t.dt->acquire(in.os.s);
                auto out = device_output(*t.ctx, l.count, in.os);
                if (t.task->numeric_slots())
                {
                    cuda::lifted::Parents p{in.w.data, in.w.stride, in.w.words, static_cast<u32>(in.w.rows),
                                            in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0};
                    cuda::check(cuda::numeric::launch_labels(v, p,
                        {static_cast<u64*>(in.numeric_views->data()), u64{v.view_rows} * v.ow}, l,
                        static_cast<u8*>(out->data()), nullptr, 0, nullptr, static_cast<u32*>(in.numeric_error->data()), in.os.s), "numeric applicable");
                    check_numeric_smoke(in);
                    launched(t, in, cudaSuccess, "numeric applicable");
                }
                else
                    launched(t, in,
                             k::launch_applicable(v, to_states(in.w), in.has_derived ? in.dw.data : nullptr,
                                                  in.has_derived ? in.dw.words : 0, l, static_cast<u8*>(out->data()), in.os.s),
                             "applicable");
                const auto [fw, enc] = output_kind(in.states, in.w, framework);
                return export_buffer(out, in.os, rl::DType::Bool, {static_cast<i64>(l.count)}, false, fw, enc);
            },
            "states"_a, "state_index"_a, "schema"_a, "binding"_a, "derived"_a = nb::none(), "stream"_a = nb::none(),
            "framework"_a = nb::none(),
            "Smoke kernel: out[i] = label i (schema[i], binding[i]) applicable in states[state_index[i]]. Device inputs "
            "are read in place on `stream`; the bool output is produced there.")
        .def(
            "apply",
            [](PyDeviceTask& t, ArrayArg states, ArrayArg state_index, ArrayArg schema, ArrayArg binding,
               ArrayArg derived, IntArg words_in, StreamArg stream, FrameworkArg framework) {
                const u32 words = int_arg<u32>(words_in, "words");
                SmokeInputs in = smoke_inputs(t, states, derived, stream);
                const k::DeviceLabels l = labels_of(t, in, state_index, schema, binding);
                const rl::dev::TaskView& v = t.dt->acquire(in.os.s);
                const u32 W = words ? words : std::max(in.w.words, v.state_words) + t.task->numeric_words();
                auto succ = device_output(*t.ctx, checked_mul({l.count, W, 8}, "apply (labels x words)"), in.os);
                auto status = device_output(*t.ctx, l.count * 4, in.os);
                if (t.task->numeric_slots())
                {
                    const u32 nn = t.task->numeric_words();
                    if (W <= nn || W - nn + t.task->numeric_slots() > cuda::lifted::k_max_words)
                        throw nb::value_error("mymyr: numeric smoke output width exceeds the device limit");
                    const u32 rw = W - nn + t.task->numeric_slots();
                    auto internal = device_output(*t.ctx, l.count * rw * sizeof(u64), in.os);
                    in.keep.push_back(internal);
                    cuda::lifted::Parents p{in.w.data, in.w.stride, in.w.words, static_cast<u32>(in.w.rows),
                                            in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0};
                    cuda::check(cuda::numeric::launch_labels(v, p,
                        {static_cast<u64*>(in.numeric_views->data()), u64{v.view_rows} * v.ow}, l, nullptr,
                        static_cast<u64*>(internal->data()), rw, static_cast<u32*>(status->data()),
                        static_cast<u32*>(in.numeric_error->data()), in.os.s), "numeric apply");
                    cuda::check(cuda::numeric::launch_convert(v, static_cast<const u64*>(internal->data()), rw, rw - t.task->numeric_slots(),
                        static_cast<u64*>(succ->data()), W, W - nn, l.count, false, in.os.s), "numeric smoke output");
                    check_numeric_smoke(in);
                    launched(t, in, cudaSuccess, "numeric apply");
                }
                else
                    launched(t, in,
                             k::launch_apply(v, to_states(in.w), in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0,
                                             l, static_cast<u64*>(succ->data()), W, static_cast<u32*>(status->data()), in.os.s),
                             "apply");
                const auto [fw, enc] = output_kind(in.states, in.w, framework);
                return ArrayPair(nb::make_tuple(
                    export_buffer(succ, in.os, rl::DType::U64, {static_cast<i64>(l.count), static_cast<i64>(W)}, true, fw, enc),
                    export_buffer(status, in.os, rl::DType::I32, {static_cast<i64>(l.count)}, false, fw, enc)));
            },
            "states"_a, "state_index"_a, "schema"_a, "binding"_a, "derived"_a = nb::none(), "words"_a = 0,
            "stream"_a = nb::none(), "framework"_a = nb::none(),
            "Successors in CPU state encoding and status per label (0 ok, 1 classical conditional effects: CPU, 2 no slot, 3 too "
            "wide, -1 not applicable)")
        .def(
            "goal",
            [](PyDeviceTask& t, ArrayArg states, ArrayArg derived, StreamArg stream, FrameworkArg framework) -> ArrayOut {
                SmokeInputs in = smoke_inputs(t, states, derived, stream);
                const rl::dev::TaskView& v = t.dt->acquire(in.os.s);
                auto out = device_output(*t.ctx, in.w.rows, in.os);
                if (t.task->numeric_slots())
                {
                    cuda::lifted::Parents p{in.w.data, in.w.stride, in.w.words, static_cast<u32>(in.w.rows),
                                            in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0};
                    launched(t, in, cuda::numeric::launch_goals(v, p, nullptr, nullptr, in.w.rows,
                                                              static_cast<u8*>(out->data()), in.os.s), "numeric goal");
                }
                else
                    launched(t, in,
                             k::launch_goal(v, to_states(in.w), in.has_derived ? in.dw.data : nullptr, in.has_derived ? in.dw.words : 0,
                                            static_cast<u8*>(out->data()), in.os.s),
                             "goal");
                const auto [fw, enc] = output_kind(in.states, in.w, framework);
                return export_buffer(out, in.os, rl::DType::Bool, {static_cast<i64>(in.w.rows)}, false, fw, enc);
            },
            "states"_a, "derived"_a = nb::none(), "stream"_a = nb::none(), "framework"_a = nb::none(),
            "Smoke kernel: goal test per state")
        .def(
            "host_derived",
            [](const PyDeviceTask& t, StatesLike states) {
                StateBatch b = import_task_states(states, *t.task);
                const u32 DW = t.task->has_axioms() ? std::max<u32>(1, bits::words_for(t.task->atoms().max_derived_slots())) : 1;
                auto block = std::make_shared<std::vector<u64>>(b.view.rows * DW, 0);
                {
                    nb::gil_scoped_release release;
                    const WorkspaceLease lease = t.task->workspace();
                    Successors& succ = lease->successors();
                    for (u64 i = 0; i < b.view.rows && t.task->has_axioms(); ++i)
                    {
                        succ.prepare(StateView{b.view.row(i), b.view.words, b.view.numeric_words ? b.view.row(i) + b.view.words : nullptr, b.view.numeric_words});
                        const detail::Engine& e = succ.engine();
                        std::copy_n(e.derived(), std::min(e.derived_words(), DW), block->begin() + static_cast<std::ptrdiff_t>(i * DW));
                    }
                }
                ArraySpec s{std::shared_ptr<const void>(block, block->data()), block->data(), rl::DType::U64,
                            {static_cast<i64>(b.view.rows), static_cast<i64>(DW)}, {}, false, true};
                return ArrayOut(export_array(std::move(s), Framework::Numpy, default_words(Framework::Numpy)));
            },
            "states"_a, "The states' derived bitsets computed by the CPU axiom evaluator ([N, DW] uint64 NumPy)");

    nb::class_<PyArena>(m, "StateArena",
                        "An append-only device array of state rows [N, W] with a pinned host mirror kept in sync by "
                        "tail copies. One writer; not thread-safe.")
        .def(
            "__init__",
            [](PyArena* self, ContextArg ctx, IntArg words_in, IntArg capacity_in) {
                const u32 words = int_arg<u32>(words_in, "words", 1, ~u32{0} / 8);  // a record's bytes fit in a u32
                const u64 capacity = int_arg<u64>(capacity_in, "capacity");
                PyContextPtr c = context_of(ctx);
                auto a = std::make_unique<cuda::DeviceArena>(c->ctx, words * 8, capacity);  // checks capacity x 8 words
                new (self) PyArena{std::move(c), std::move(a), words};
            },
            "ctx"_a, "words"_a, "capacity"_a = 1024)
        .def_prop_ro("words", [](const PyArena& a) { return a.words; }, "Words per state row (W).")
        .def_prop_ro("device_size", [](const PyArena& a) { return a.arena->device_size(); }, "Rows on the device.")
        .def_prop_ro("host_size", [](const PyArena& a) { return a.arena->host_size(); },
                     "Rows valid in the host mirror (advanced by poll() and wait()).")
        .def_prop_ro("capacity", [](const PyArena& a) { return a.arena->capacity(); },
                     "Rows the device array holds before it grows (it at least doubles).")
        .def_prop_ro("synced_upto", [](const PyArena& a) { return a.arena->synced_upto(); },
                     "The end of the last sync issued (host_size once every sync in flight has landed).")
        .def(
            "append",
            [](PyArena& a, StatesLike states) {
                a.ctx->drain();
                const dl::Device d = dl::dlpack_device(states);
                if (d.device_type == dl::k_cuda || d.device_type == dl::k_cuda_managed)
                {
                    // imported with the arena's stream: the producer hands off to it, the copy runs there
                    const OpStream os{a.arena->stream(), reinterpret_cast<std::intptr_t>(a.arena->stream())};
                    DeviceInput in = import_device(states, *a.ctx, os, "states");
                    const WordsLayout w = device_words(in);
                    if (w.words != a.words || (w.rows > 1 && w.stride != w.words))
                        throw nb::value_error("mymyr: StateArena.append needs contiguous rows of the arena's width");
                    a.arena->append_from_device(w.data, w.rows, os.s);
                    a.ctx->release_after(os.s, {in.im.keep});
                    return;
                }
                StateBatch b = import_states(states, a.words);
                if (b.view.words != a.words || (b.view.rows > 1 && b.view.stride != b.view.words))
                    throw nb::value_error("mymyr: StateArena.append needs contiguous rows of the arena's width");
                a.arena->append_from_host(b.view.data, b.view.rows);
            },
            "states"_a, "Appends rows: device arrays by a device copy (read in place), host arrays by an upload")
        .def("sync", [](PyArena& a) { a.arena->sync_to_host(); }, "Starts copying the new tail to the host mirror")
        .def("poll", [](PyArena& a) { return a.arena->poll(); },
             "Advances host_size over the syncs that have landed; True when none is in flight.")
        .def("wait", [](PyArena& a) { a.arena->wait(); }, "Blocks until every sync in flight has landed.")
        .def(
            "device_view",
            [](const PyArena& a, IntArg lo_in, SizeArg hi_obj, FrameworkArg framework) -> ArrayOut {
                const u64 lo = int_arg<u64>(lo_in, "lo");
                const u64 hi = hi_obj.is_none() ? a.arena->device_size() : int_arg<u64>(hi_obj, "hi");
                if (lo > hi || hi > a.arena->device_size())
                    throw nb::index_error("mymyr: StateArena.device_view: range outside [0, device_size]");
                const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
                const auto& gen = a.arena->generation();
                ArraySpec s;
                // offset folded into the pointer (lo <= device_size: within the checked capacity x record bytes)
                const std::byte* p = a.arena->device_data() + lo * a.arena->record_bytes();
                s.owner = std::shared_ptr<const void>(gen, p);
                s.data = p;
                s.dtype = rl::DType::U64;
                s.shape = {static_cast<i64>(hi - lo), static_cast<i64>(a.words)};
                s.words = true;
                const OpStream os{a.arena->stream(), 0};
                return export_device_array(std::move(s),
                                           DeviceExport{dl::k_cuda, a.ctx->ctx->device(), std::make_shared<BufferSync>(gen, os.s)},
                                           fw, default_words(fw));
            },
            "lo"_a = 0, "hi"_a = nb::none(), "framework"_a = nb::none(),
            "Records [lo, hi) as a zero-copy device array (keeps its generation alive when the arena grows)")
        .def(
            "host_view",
            [](const PyArena& a, FrameworkArg framework) -> ArrayOut {
                const Framework fw = framework.is_none() ? Framework::Numpy : parse_framework(framework, nb::none());
                const auto& mirror = a.arena->mirror();
                ArraySpec s;
                s.owner = std::shared_ptr<const void>(mirror, mirror->data());
                s.data = mirror->data();
                s.dtype = rl::DType::U64;
                s.shape = {static_cast<i64>(a.arena->host_size()), static_cast<i64>(a.words)};
                s.readonly = true;
                s.words = true;
                return export_array(std::move(s), fw, default_words(fw));
            },
            "framework"_a = nb::none(), "The synced records [0, host_size) of the pinned mirror (read-only, zero-copy)");

    m.def(
        "to_memory",
        [](ArrayArg array, MemoryArg memory, ContextArg ctx, FrameworkArg framework) -> ArrayOut {
            nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> arr;
            if (!nb::try_cast(array, arr))
                throw nb::type_error("mymyr: to_memory takes a C-contiguous CPU array");
            PyContextPtr c = context_of(ctx);
            const MemoryKind kind = parse_memory(memory);
            const rl::DType dtype = dtype_of(arr.dtype());
            const u64 bytes = arr.nbytes();
            std::vector<i64> shape;
            for (usize i = 0; i < arr.ndim(); ++i)
                shape.push_back(static_cast<i64>(arr.shape(i)));
            const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
            const OpStream os{c->ctx->stream(), reinterpret_cast<std::intptr_t>(c->ctx->stream())};
            ArraySpec s;
            s.dtype = dtype;
            s.shape = shape;
            if (kind == MemoryKind::Device)
            {
                auto buf = device_output(*c, bytes, os);
                cuda::check(cudaMemcpyAsync(buf->data(), arr.data(), bytes, cudaMemcpyHostToDevice, os.s), "cudaMemcpyAsync");
                cuda::check(cudaStreamSynchronize(os.s), "cudaStreamSynchronize");  // pageable source
                return export_buffer(buf, os, dtype, shape, false, fw, default_words(fw));
            }
            MemoryResource& mr = kind == MemoryKind::Managed ? c->ctx->managed_memory() : c->ctx->pinned_memory();
            void* p = mr.allocate(std::max<u64>(bytes, 1), 64);
            std::memcpy(p, arr.data(), bytes);
            PyContextPtr keep = c;
            std::shared_ptr<void> owner(p, [keep, kind](void* q) {
                MemoryResource& r = kind == MemoryKind::Managed ? keep->ctx->managed_memory() : keep->ctx->pinned_memory();
                r.deallocate(q, 0, 64);
            });
            s.owner = owner;
            s.data = p;
            // written by the host: nothing to hand off
            return export_device_array(std::move(s),
                                       DeviceExport{kind == MemoryKind::Managed ? dl::k_cuda_managed : dl::k_cuda_host,
                                                    c->ctx->device(), nullptr},
                                       fw, default_words(fw));
        },
        "array"_a, "memory"_a = "device", "ctx"_a = nb::none(), "framework"_a = nb::none(),
        "A copy of a CPU array in device, managed or pinned-host memory, exported with its DLPack device type (tests "
        "and staging)");

    m.def(
        "_delayed_iota",
        [](u64 n, u64 value, u64 cycles, ContextArg ctx, StreamArg stream, FrameworkArg framework) -> ArrayOut {
            PyContextPtr c = context_of(ctx);
            const OpStream os = c->stream_of(stream);
            auto buf = device_output(*c, checked_mul({n, 8}, "_delayed_iota"), os);
            cuda::check(k::launch_delayed_iota(static_cast<u64*>(buf->data()), n, value, cycles, os.s), "delayed_iota");
            const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
            return export_buffer(buf, os, rl::DType::U64, {static_cast<i64>(n)}, false, fw, default_words(fw));
        },
        "n"_a, "value"_a, "cycles"_a, "ctx"_a, "stream"_a = nb::none(), "framework"_a = nb::none(),
        "Test helper: a fresh device array [n] uint64 with x[i] = value + i, written on `stream` after spinning about "
        "`cycles` clock cycles (stream-semantics tests)");

    m.def(
        "_import_info",
        [](ArrayArg obj, ContextArg ctx, StreamArg stream) {
            PyContextPtr c = context_of(ctx);
            const OpStream os = c->stream_of(stream);
            DeviceInput in = import_device(obj, *c, os, "array");
            const WordsLayout w = device_words(in);
            c->release_after(os.s, {in.im.keep});
            return std::make_tuple(reinterpret_cast<std::uintptr_t>(w.data), w.rows, w.words, w.stride, framework_name(in.fw),
                                   in.im.device.device_type, in.im.device.device_id);
        },
        "obj"_a, "ctx"_a, "stream"_a = nb::none(),
        "(data pointer, rows, words, row stride in words, framework, DLPack device type, device id) of a device state "
        "array as the CUDA operations import it: tests use it to check that inputs are read in place");

    device_hooks().hash_rows = &device_hash_rows;
    device_hooks().expand = &device_expand;
    device_hooks().expand_into = &device_expand_into;
    device_hooks().masks = &device_action_masks;
    m.def(
        "_masks_info",
        [](SuiteArg table, int device) {
            const SuiteRef ref = suite_of(table);
            std::string why;
            const auto dm = table_masks(ref, table_context(ref, nb::none(), device), &why);
            return std::make_tuple(dm ? dm->view_words() : u64{0}, why, ref.suite->max_schemas(),
                                   ref.suite->max_objects());
        },
        "table"_a, "device"_a,
        "(views words per row, why not, schema columns, object columns) of the table's device masks on a device: the "
        "scratch the mask ops take, an empty string if the kernels run every instance (else the ops run the host "
        "functions, which wait and raise), and the widths of the schema and object masks.");
    m.def(
        "_masks_ptrs",
        [](SuiteArg table, int kind, int device, std::intptr_t stream, u64 rows, u32 row_words, std::uintptr_t states,
           std::uintptr_t task_ids, std::uintptr_t schema, i32 value, std::uintptr_t labels, u32 width, u32 depth,
           std::uintptr_t out, u32 num_objects, std::uintptr_t views, std::uintptr_t status) {
            if (kind < 0 || kind > 2)
                throw nb::value_error("mymyr: unknown mask kind");
            const SuiteRef ref = suite_of(table);
            const rl::TaskSuite& tt = *ref.suite;
            const u32 NN = tt.numeric_words();
            if (row_words < NN)
                throw nb::value_error("mymyr: numeric states have no numeric block");
            MaskCall m;
            m.kind = static_cast<MaskKind>(kind);
            m.b = {reinterpret_cast<const u64*>(states), rows, row_words - NN, row_words,
                   reinterpret_cast<const i32*>(task_ids)};
            m.numeric_words = NN;
            m.schema = {reinterpret_cast<const i32*>(schema), value};
            m.labels = reinterpret_cast<const i32*>(labels);
            m.width = width;
            m.depth = depth;
            m.num_objects = num_objects;
            m.out = reinterpret_cast<u8*>(out);
            m.views = reinterpret_cast<u64*>(views);
            m.status = reinterpret_cast<u32*>(status);
            std::string why;
            const auto dm = table_masks(ref, table_context(ref, nb::none(), device), &why);
            nb::gil_scoped_release release;
            cuda::DeviceGuard g(device);
            run_masks(tt, dm.get(), why, m, cuda_stream(stream), false);
        },
        "table"_a, "kind"_a, "device"_a, "stream"_a, "rows"_a, "row_words"_a, "states"_a, "task_ids"_a, "schema"_a,
        "value"_a, "labels"_a, "width"_a, "depth"_a, "out"_a, "num_objects"_a, "views"_a, "status"_a,
        "The mask ops' native entry on device pointers (kind 0 for schemas, 1 for objects, 2 for applicable), on "
        "`stream`: status bits instead of errors on the device; the host functions (waiting, raising) for tables the "
        "kernels do not run.");
    m.def("_check_async_expansion",
          [](TableArg table, u64 rows, int device, std::intptr_t stream)
          {
              const SuiteRef ref = suite_of(table);
              PyContextPtr c = table_context(ref, nb::none(), device);
              const auto s = cuda_stream(stream);
              cudaStreamCaptureStatus capture;
              unsigned long long id = 0;
              cuda::DeviceGuard guard(device);
              cuda::check(cudaStreamGetCaptureInfo(s, &capture, &id), "cudaStreamGetCaptureInfo");
              auto check = [&](const cuda::SuiteExpander& ex) {
                  if (const std::string why = ex.async_unsupported(rows, rl::ExpandOptions{}); !why.empty())
                      throw std::invalid_argument("mymyr: device action expansion needs an asynchronous backend: " + why);
              };
              if (capture != cudaStreamCaptureStatusNone)
                  check(*captured_workspace(ref, c, s, id)->expander);
              else
              {
                  ExpanderLease ex(ref, c, s, true);
                  nb::gil_scoped_release release;
                  check(*ex.operator->());
              }
          },
          "table"_a, "rows"_a, "device"_a, "stream"_a,
          "Checks that the table can expand this many rows without waiting for the device.");

    m.def("_expansion_workspace_guard", [](nb::capsule capsule) {
        if (std::strcmp(capsule.name(), "mymyr.expansion_workspace_weak") != 0)
            throw nb::type_error("mymyr: expected a weak expansion workspace reference");
        auto owner = static_cast<WorkspaceReference*>(capsule.data())->workspace.lock();
        if (!owner)
            throw nb::value_error("mymyr: expansion workspace has expired");
        return nb::capsule(new std::shared_ptr<ExpansionWorkspace>(std::move(owner)), "mymyr.expansion_workspace_guard",
                           [](void* p) noexcept { delete static_cast<std::shared_ptr<ExpansionWorkspace>*>(p); });
    }, "reference"_a, "An independent safety owner for checking graph lifetimes during pending launches.");
    m.def("_expansion_workspace_alive", [](nb::capsule capsule) {
        if (std::strcmp(capsule.name(), "mymyr.expansion_workspace_weak") != 0)
            throw nb::type_error("mymyr: expected a weak expansion workspace reference");
        return !static_cast<WorkspaceReference*>(capsule.data())->workspace.expired();
    }, "reference"_a, "Whether a captured expansion workspace still has an owner.");

    m.def("_expansion_workspace_owners", [](nb::capsule capsule) {
        if (std::strcmp(capsule.name(), "mymyr.expansion_workspace_weak") != 0)
            throw nb::type_error("mymyr: expected a weak expansion workspace reference");
        return static_cast<WorkspaceReference*>(capsule.data())->workspace.use_count();
    }, "reference"_a, "The number of independent owners of a captured expansion workspace.");

    m.def("_expansion_workspace_released", [](nb::capsule capsule) {
        if (std::strcmp(capsule.name(), "mymyr.expansion_workspace_weak") != 0)
            throw nb::type_error("mymyr: expected a weak expansion workspace reference");
        return static_cast<WorkspaceReference*>(capsule.data())->released->load(std::memory_order_acquire);
    }, "reference"_a, "Whether workspace cleanup has finished, including abandonment after a CUDA failure.");

    nb::class_<DevicePadded>(m, "DevicePaddedExpansion",
                             "The padded [N, K] view of a device expansion (rl.PaddedExpansion's arrays, on the device).")
        .def_prop_ro("k", [](const DevicePadded& p) { return p.k; },
                     "The padded successor count per state (the second dimension, K).")
        .def_prop_ro("words", [](const DevicePadded& p) { return p.words; }, "Atom words per row (W).")
        .def_prop_ro("numeric_words", [](const DevicePadded& p) { return p.numeric_words; },
                     "Numeric words per row after the atom words (0 for classical tables).")
        .def_prop_ro("overflow", [](const DevicePadded& p) { return p.overflow; },
                     "Whether the flat or padded capacity or the atom-word width omitted successor data.")
        .def_prop_ro("index", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_index, rl::DType::I32, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] int32: row of the flat arrays, -1 = none")
        .def_prop_ro("mask", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_mask, rl::DType::Bool, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] bool")
        .def_prop_ro("count", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_count, rl::DType::I32, {static_cast<i64>(p.rows)});
        }, "[N] int32 true successor counts")
        .def_prop_ro("succ", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_succ, rl::DType::U64, {static_cast<i64>(p.rows), p.k, static_cast<i64>(p.words) + p.numeric_words}, true);
        }, "[N, K, W + numeric_words] state rows in the task's CPU encoding (zero padding)")
        .def_prop_ro("schema", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_schema, rl::DType::I32, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] int32, -1 padding")
        .def_prop_ro("binding", [](const DevicePadded& p) -> ArrayOut {
            return p.array(p.off_binding, rl::DType::I32, {static_cast<i64>(p.rows), p.k, p.label_width});
        }, "[N, K, L] int32 objects, -1 padding")
        .def_prop_ro("goal", [](const DevicePadded& p) -> MaybeArray {
            if (!p.has_goal)
                return nb::none();
            return p.array(p.off_goal, rl::DType::Bool, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] bool goal flags (if requested)")
        .def_prop_ro("succ_hash", [](const DevicePadded& p) -> MaybeArray {
            if (!p.has_hash) return nb::none();
            std::vector<i64> shape{static_cast<i64>(p.rows), p.k};
            if (p.fw == Framework::Jax) shape.push_back(1);
            return p.array(p.off_hash, p.fw == Framework::Torch ? rl::DType::I64 : rl::DType::U64,
                           std::move(shape), p.fw == Framework::Jax);
        }, "[N, K] native successor hashes, zero padding")
        .def_prop_ro("noop", [](const DevicePadded& p) -> MaybeArray {
            if (!p.has_noop) return nb::none();
            return p.array(p.off_noop, rl::DType::Bool, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] bitwise equality with the parent, false padding")
        .def_prop_ro("canonical", [](const DevicePadded& p) -> MaybeArray {
            if (!p.has_canonical) return nb::none();
            return p.array(p.off_canonical, rl::DType::I32, {static_cast<i64>(p.rows), p.k});
        }, "[N, K] int32: the flat index (position in the parent's full successor order, the environment action), "
           "-1 padding; None if the expansion has no index")
        .def_prop_ro("derived", [](const DevicePadded& p) -> MaybeArray {
            if (!p.has_derived)
                return nb::none();
            return p.array(p.off_derived, rl::DType::U64, {static_cast<i64>(p.rows), p.k, static_cast<i64>(p.derived_words)}, true);
        }, "[N, K, D] derived words of the successors (zero padding), if requested")
        .def("__repr__", [](const DevicePadded& p) {
            return "DevicePaddedExpansion(states=" + std::to_string(p.rows) + ", k=" + std::to_string(p.k) +
                   (p.overflow ? ", overflow" : "") + ")";
        });

    nb::class_<PyDeviceExpansion>(m, "DeviceExpansion", nb::is_weak_referenceable(),
                                  "rl.expand of CUDA device states: rl.Expansion's arrays on the device, produced on "
                                  "the call's stream (a consumer's __dlpack__(stream=s) waits for them), byte-equal to "
                                  "the CPU expansion.")
        .def_prop_ro("succ", [](const PyDeviceExpansion& e) -> ArrayOut {
            return e.array(e.flat.off_succ, rl::DType::U64, {static_cast<i64>(e.flat.valid()), static_cast<i64>(e.flat.words) + e.flat.numeric_words}, true);
        }, "[M, W + numeric_words] successor state rows in the task's CPU encoding")
        .def_prop_ro("parent", [](const PyDeviceExpansion& e) -> ArrayOut {
            return e.array(e.flat.off_parent, rl::DType::I32, {static_cast<i64>(e.flat.valid())});
        }, "[M] int32 index of the expanded state")
        .def_prop_ro("schema", [](const PyDeviceExpansion& e) -> ArrayOut {
            return e.array(e.flat.off_schema, rl::DType::I32, {static_cast<i64>(e.flat.valid())});
        }, "[M] int32 schema of the label")
        .def_prop_ro("binding", [](const PyDeviceExpansion& e) -> ArrayOut {
            return e.array(e.flat.off_binding, rl::DType::I32, {static_cast<i64>(e.flat.valid()), e.flat.label_width});
        }, "[M, L] int32 objects of the label, -1 past the schema's arity")
        .def_prop_ro("offsets", [](const PyDeviceExpansion& e) -> ArrayOut {
            return e.array(e.flat.off_offsets, rl::DType::I32, {static_cast<i64>(e.flat.rows + 1)});
        }, "[N + 1] int32 CSR offsets (true counts, even past an overflow)")
        .def_prop_ro("goal", [](const PyDeviceExpansion& e) -> MaybeArray {
            if (!e.flat.has_goal)
                return nb::none();
            return e.array(e.flat.off_goal, rl::DType::Bool, {static_cast<i64>(e.flat.valid())});
        }, "[M] bool goal flags of the successors (expand(..., goal=True))")
        .def_prop_ro("derived", [](const PyDeviceExpansion& e) -> MaybeArray {
            if (!e.flat.has_derived)
                return nb::none();
            return e.array(e.flat.off_derived, rl::DType::U64, {static_cast<i64>(e.flat.valid()), static_cast<i64>(e.flat.derived_words)}, true);
        }, "[M, D] derived words of the successors (expand(..., derived=True))")
        .def_prop_ro("derived_words", [](const PyDeviceExpansion& e) { return e.flat.derived_words; },
                     "Derived words per row (D) of the derived array (0 if not requested).")
        .def_prop_ro("succ_hash", [](const PyDeviceExpansion& e) -> MaybeArray {
            if (!e.flat.has_hash) return nb::none();
            std::vector<i64> shape{static_cast<i64>(e.flat.valid())};
            if (e.fw == Framework::Jax) shape.push_back(1);
            return e.array(e.flat.off_hash, e.fw == Framework::Torch ? rl::DType::I64 : rl::DType::U64,
                           std::move(shape), e.fw == Framework::Jax);
        }, "[M] native successor hashes, if requested")
        .def_prop_ro("noop", [](const PyDeviceExpansion& e) -> MaybeArray {
            if (!e.flat.has_noop) return nb::none();
            return e.array(e.flat.off_noop, rl::DType::Bool, {static_cast<i64>(e.flat.valid())});
        }, "[M] bitwise equality with the parent, if requested")
        .def_prop_ro("index", [](const PyDeviceExpansion& e) -> MaybeArray {
            if (!e.flat.has_index) return nb::none();
            return e.array(e.flat.off_index, rl::DType::I32, {static_cast<i64>(e.flat.valid())});
        }, "[M] canonical successor positions with drop_noops")
        .def_prop_ro("counts", [](const PyDeviceExpansion& e) -> ArrayOut {
            const std::vector<i32> off = e.flat.host_offsets();
            std::vector<i32> c(e.flat.rows);
            for (u64 i = 0; i < e.flat.rows; ++i)
                c[i] = off[i + 1] - off[i];
            auto buf = device_output(*e.ctx, c.size() * sizeof(i32), e.flat.os);
            cuda::DeviceGuard g(e.ctx->ctx->device());
            cuda::check(cudaMemcpyAsync(buf->data(), c.data(), c.size() * sizeof(i32), cudaMemcpyHostToDevice, e.flat.os.s),
                        "cudaMemcpyAsync");
            cuda::check(cudaStreamSynchronize(e.flat.os.s), "cudaStreamSynchronize");  // pageable source
            return export_region(buf, 0, e.flat.os, rl::DType::I32, {static_cast<i64>(e.flat.rows)}, false, e.fw, e.enc);
        }, "[N] int32 successors per state (a fresh array)")
        .def_prop_ro("total", [](const PyDeviceExpansion& e) { return e.flat.total; },
                     "The successor rows of the batch, those beyond the capacity included.")
        .def_prop_ro("generated", [](const PyDeviceExpansion& e) { return e.flat.generated; },
                     "Successors before drop_noops; generated > capacity is an overflow.")
        .def_prop_ro("capacity", [](const PyDeviceExpansion& e) { return e.flat.capacity; },
                     "The successor rows the outputs hold; total > capacity is an overflow.")
        .def_prop_ro("num_states", [](const PyDeviceExpansion& e) { return e.flat.rows; },
                     "The number of expanded states (the batch size).")
        .def_prop_ro("words", [](const PyDeviceExpansion& e) { return e.flat.words; }, "Atom words per row (W).")
        .def_prop_ro("numeric_words", [](const PyDeviceExpansion& e) { return e.flat.numeric_words; },
                     "Numeric words per row after the atom words (0 for classical tables).")
        .def_prop_ro("words_needed", [](const PyDeviceExpansion& e) { return e.flat.words_needed; },
                     "The atom words every successor row needs; more than words is an overflow (new atoms under lazy "
                     "slots).")
        .def_prop_ro("label_width", [](const PyDeviceExpansion& e) { return e.flat.label_width; },
                     "Binding objects per successor (the largest schema arity).")
        .def_prop_ro("num_objects", [](const PyDeviceExpansion& e) { return e.ref.suite->max_objects(); },
                     "The most objects of an instance of the table (the columns of rl.prefix_masks).")
        .def_prop_ro("num_schemas", [](const PyDeviceExpansion& e) { return e.ref.suite->max_schemas(); },
                     "Action schemas of the table (the columns of rl.schema_masks).")
        .def_prop_ro("table", [](const PyDeviceExpansion& e) { return SuiteArg(e.ref.obj); },
                     "The table (or Task) the expansion was made over.")
        .def_prop_ro("overflow", [](const PyDeviceExpansion& e) {
            return e.flat.generated > e.flat.capacity || e.flat.words_needed > e.flat.words;
        },
                     "Whether the outputs miss rows or words: total > capacity or words_needed > words.")
        .def_prop_ro("framework", [](const PyDeviceExpansion& e) { return framework_name(e.fw); },
                     "The framework of the arrays: 'torch', 'jax' or 'dlpack'.")
        .def_prop_ro("context", [](const PyDeviceExpansion& e) { return PyContextObj{e.ctx}; },
                     "The Context the arrays live in.")
        .def_prop_ro("stream", [](const PyDeviceExpansion& e) { return e.flat.os.dl; },
                     "the stream the arrays were produced on (a DLPack stream value)")
        .def_prop_ro("padded", [](const PyDeviceExpansion& e) -> Arg<std::optional<DevicePadded>> {
            if (!e.padded)
                return nb::none();
            return nb::cast(*e.padded);
        }, "The padded view, if expand() was called with k")
        .def("pad",
             [](const PyDeviceExpansion& e, IntArg k_in) {
                 const std::optional<u32> k = opt_int_arg<u32>(k_in, "k", 1);
                 return make_device_padded(e.ref, e.ctx, e.flat, k ? *k : device_auto_k(e.flat), e.fw, e.enc);
             },
             "k"_a = nb::none(), "The padded [N, k] view on the device (None: k is the next power of two at or above the largest count).")
        .def("action",
             [](const PyDeviceExpansion& e, IntArg j_in) {
                 const u64 j = int_arg<u64>(j_in, "row");
                 if (j >= e.flat.valid())
                     throw nb::index_error("mymyr: successor row out of range");
                 std::vector<i32> b(e.flat.label_width + 1);
                 i32 parent = 0, inst = 0;
                 cuda::DeviceGuard g(e.ctx->ctx->device());
                 cuda::check(cudaMemcpyAsync(b.data(), e.flat.at<i32>(e.flat.off_schema) + j, sizeof(i32), cudaMemcpyDeviceToHost,
                                             e.flat.os.s),
                             "cudaMemcpyAsync");
                 cuda::check(cudaMemcpyAsync(b.data() + 1, e.flat.at<i32>(e.flat.off_binding) + j * e.flat.label_width,
                                             e.flat.label_width * sizeof(i32), cudaMemcpyDeviceToHost, e.flat.os.s),
                             "cudaMemcpyAsync");
                 if (e.task_ids)
                     cuda::check(cudaMemcpyAsync(&parent, e.flat.at<i32>(e.flat.off_parent) + j, sizeof(i32),
                                                 cudaMemcpyDeviceToHost, e.flat.os.s),
                                 "cudaMemcpyAsync");
                 cuda::check(cudaStreamSynchronize(e.flat.os.s), "cudaStreamSynchronize");
                 if (e.task_ids)
                 {
                     cuda::check(cudaMemcpyAsync(&inst, static_cast<const i32*>(e.task_ids->data()) + parent, sizeof(i32),
                                                 cudaMemcpyDeviceToHost, e.flat.os.s),
                                 "cudaMemcpyAsync");
                     cuda::check(cudaStreamSynchronize(e.flat.os.s), "cudaStreamSynchronize");
                 }
                 const u32 s = static_cast<u32>(b[0]);
                 const Owner o = e.ref.owner(static_cast<u32>(inst));
                 return make_label(o, s, b.data() + 1, o.core->data->schemas[s].arity());
             },
             "row"_a, "The Action (label) of flat row j (copies it to the host).")
        .def("actions",
             [](nb::pointer_and_handle<PyDeviceExpansion> self, IntArg i_in) {
                 const u64 i = int_arg<u64>(i_in, "state");
                 const PyDeviceExpansion& e = *self.p;
                 if (i >= e.flat.rows)
                     throw nb::index_error("mymyr: state index out of range");
                 const std::vector<i32> off = e.flat.host_offsets();
                 ActionList out{nb::list()};
                 for (i64 j = off[i]; j < off[i + 1] && static_cast<u64>(j) < e.flat.valid(); ++j)
                     out.append(self.h.attr("action")(j));
                 return out;
             },
             "state"_a, "The Actions of state i's successors, in order.")
        .def("__len__", [](const PyDeviceExpansion& e) { return e.flat.valid(); })
        .def("__repr__", [](const PyDeviceExpansion& e) {
            return "DeviceExpansion(states=" + std::to_string(e.flat.rows) + ", successors=" + std::to_string(e.flat.total) +
                   ", words=" + std::to_string(e.flat.words) +
                   (e.flat.total > e.flat.capacity || e.flat.words_needed > e.flat.words ? ", overflow" : "") +
                   ", framework=" + framework_name(e.fw) + ", device=cuda:" + std::to_string(e.ctx->ctx->device()) + ")";
        });

    nb::class_<PyDeviceBrfs>(m, "BrfsResult",
                             "The device BrFS's result (mymyr.search.BrfsResult's fields, plus per-phase statistics and "
                             "the state space, which stays on the device).")
        .def_prop_ro("status", [](const PyDeviceBrfs& x) {
            const BrfsResult& r = x.r.result;
            return r.solved ? search::SearchStatus::Solved
                            : (r.exhausted ? search::SearchStatus::Exhausted : search::SearchStatus::OutOfStates);
        }, "How the search ended (mymyr.search.Status).")
        .def_prop_ro("solved", [](const PyDeviceBrfs& x) { return x.r.result.solved; },
                     "True when status is Status.SOLVED (there is a plan).")
        .def_prop_ro("plan", [](const PyDeviceBrfs& x) {
            ActionList out{nb::list()};
            for (const Action& a : x.r.result.plan)
                out.append(action_object(x.o, a));
            return out;
        }, "The plan to the first goal state expanded, a list of Actions (empty without a plan).")
        .def_prop_ro("cost", [](const PyDeviceBrfs& x) { return x.cost; },
                     "The plan's cost under the task's metric (0.0 without a plan).")
        .def_prop_ro("goal_state",
                     [](const PyDeviceBrfs& x) -> Arg<std::optional<PyState>> {
                         if (!x.goal_state)
                             return Arg<std::optional<PyState>>(nb::none());
                         return Arg<std::optional<PyState>>(make_state(x.o, State(*x.goal_state)));
                     },
                     "The State the plan reaches (None without a plan).")
        .def_prop_ro("stats",
                     [](const PyDeviceBrfs& x) {
                         const BrfsResult& r = x.r.result;
                         search::SearchStatistics s;
                         s.expanded = r.expanded;
                         s.generated = r.generated;
                         s.states = r.states;
                         s.pruned = r.pruned;
                         s.seconds = r.search_s;
                         s.memory_bytes = r.memory_bytes;
                         return s;
                     },
                     "The search's counts and time (mymyr.search.Statistics; memory_bytes: the device store).")
        .def_prop_ro("message", [](const PyDeviceBrfs& x) { return x.r.result.message; },
                     "Why the search stopped early (empty when it ran to its end).")
        .def_prop_ro("exhausted", [](const PyDeviceBrfs& x) { return x.r.result.exhausted; },
                     "The whole reachable space was expanded (with stop_at_goal=False also when solved).")
        .def_prop_ro("num_goal_states", [](const PyDeviceBrfs& x) { return x.r.result.num_goal_states; },
                     "Goal states expanded.")
        .def_prop_ro("layers", [](const PyDeviceBrfs& x) { return x.r.result.layers; }, "Layers expanded.")
        .def_prop_ro("store", [](const PyDeviceBrfs& x) { return x.r.result.store; }, "The state store used.")
        .def_prop_ro("fingerprint", [](const PyDeviceBrfs& x) { return x.r.result.fingerprint; },
                     "With fingerprint=True: the hash of mymyr.search.BrfsResult.fingerprint; else 0.")
        .def_prop_ro("words", [](const PyDeviceBrfs& x) { return std::max<u32>(1, x.b->words() - x.o.core->task->numeric_slots()); }, "Atom words per stored state row.")
        .def_prop_ro("numeric_words", [](const PyDeviceBrfs& x) { return x.o.core->task->numeric_words(); },
                     "Numeric words per stored state row (0 for a classical task).")
        .def_prop_ro("counters", [](const PyDeviceBrfs& x) {
            const cuda::DeviceBrfsStats& s = x.r.stats;
            nb::typed<nb::dict, std::string, std::variant<double, u64>> d{nb::dict()};
            d["view_ms"] = s.view_ms;
            d["gen_ms"] = s.gen_ms;
            d["dedup_ms"] = s.dedup_ms;
            d["host_ms"] = s.host_ms;
            d["chunks"] = s.chunks;
            d["groups"] = s.groups;
            d["resumed"] = s.resumed;
            d["redone"] = s.redone;
            d["windows"] = s.windows;
            d["loops"] = s.loops;
            d["captures"] = s.captures;
            d["rehashes"] = s.rehashes;
            d["growths"] = s.growths;
            d["mirror_growths"] = s.mirror_growths;
            d["uploads"] = s.uploads;
            d["widenings"] = s.widenings;
            d["host_schemas"] = s.host_schemas;
            d["table_slots"] = s.table_slots;
            d["device_bytes"] = s.device_bytes;
            d["ce_schemas"] = s.ce_schemas;
            d["host_ce_schemas"] = s.host_ce_schemas;
            d["device_axioms"] = static_cast<u64>(s.device_axioms ? 1 : 0);
            d["axiom_ms"] = s.axiom_ms;
            d["host_axiom_ms"] = s.host_axiom_ms;
            d["axiom_reruns"] = s.axiom_reruns;
            return d;
        }, "A dict of per-phase device times (ms) and counters: view_ms (with axiom_ms, the device axioms), gen_ms, dedup_ms, "
           "host_ms (with host_axiom_ms, CPU time on axioms); host_schemas (CPU fallback; host_ce_schemas of them with "
           "conditional effects), ce_schemas (conditional effects on the device), device_axioms (1: the axioms run on the "
           "device), axiom_reruns (lazy slots: axiom evaluations redone after interning derived atoms); chunks, groups "
           "(host reads of the chunks' device-sized records), resumed / redone (chunks past their capacity, redone from "
           "their rows / views), windows (a state's successors generated a window at a time: more than fit the device "
           "at once), growths / mirror_growths (new generations of the arenas and of the states' pinned mirror, each "
           "a copy of what they hold), loops / captures (device loops over small layers, their graphs). The times are 0 "
           "unless brfs() ran with timings=True")
        .def("state_words",
             [](const PyDeviceBrfs& x, FrameworkArg framework) -> ArrayOut {
                 if (x.numeric_rows)
                 {
                     const u32 words = std::max<u32>(1, x.b->words() - x.o.core->task->numeric_slots()) + x.o.core->task->numeric_words();
                     const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
                     return export_region(x.numeric_rows, 0, OpStream{x.b->states().stream()}, rl::DType::U64,
                                          {static_cast<i64>(x.r.result.states), words}, true, fw, default_words(fw), true);
                 }
                 return arena_array(x, x.b->states(), rl::DType::U64, x.b->words(), true, framework);
             },
             "framework"_a = nb::none(), "The stored states [states, words + numeric_words] in id order (read-only, on the device); numeric values use the task's CPU encoding")
        .def("nodes",
             [](const PyDeviceBrfs& x, FrameworkArg framework) -> ArrayOut {
                 return arena_array(x, x.b->nodes(), rl::DType::I32, 2, false, framework);
             },
             "framework"_a = nb::none(),
             "The node records [states, 2] int32: parent id (-1 for the root) and the index among the parent's "
             "successors in canonical order (zero-copy, read-only, on the device)")
        .def("plan_to",
             [](const PyDeviceBrfs& x, IntArg id_in) {
                 const u64 id = int_arg<u64>(id_in, "id");
                 std::vector<Action> plan;
                 {
                     nb::gil_scoped_release release;
                     plan = x.b->plan_to(id);
                 }
                 ActionList out{nb::list()};
                 for (const Action& a : plan)
                     out.append(action_object(x.o, a));
                 return out;
             },
             "id"_a, "The plan from the initial state to state `id` (replayed on the CPU)")
        .def("__repr__", [](const PyDeviceBrfs& x) {
            const BrfsResult& r = x.r.result;
            return std::string("BrfsResult(status=") +
                   (r.solved ? "solved" : (r.exhausted ? "exhausted" : "out_of_states")) +
                   ", plan_length=" + std::to_string(r.plan.size()) + ", layers=" + std::to_string(r.layers) +
                   ", expanded=" + std::to_string(r.expanded) + ", generated=" + std::to_string(r.generated) +
                   ", states=" + std::to_string(r.states) + ", store=device)";
        });

    m.def(
        "brfs",
        [](TaskArg task, ContextArg ctx, bool witness_pruning, bool canonical_order, SizeArg max_states,
           bool stop_at_goal, bool fingerprint, SizeArg chunk_states, SizeArg expected_states, SizeArg max_depth, bool timings) {
            const Owner o = owner_of(task);
            PyContextPtr c = task_context(*o.core, ctx, 0);
            cuda::DeviceBrfsOptions opts;
            opts.witness_pruning = witness_pruning;
            opts.canonical_order = canonical_order;
            if (!max_states.is_none())
                opts.max_states = int_arg<u64>(max_states, "max_states");
            opts.stop_at_goal = stop_at_goal;
            opts.fingerprint = fingerprint;
            if (!chunk_states.is_none())
                opts.chunk_states = int_arg<u32>(chunk_states, "chunk_states", 1);
            if (!expected_states.is_none())
                opts.expected_states = int_arg<u64>(expected_states, "expected_states");
            if (!max_depth.is_none())
                opts.max_depth = int_arg<u32>(max_depth, "max_depth");
            opts.timings = timings;
            PyDeviceBrfs x;
            x.o = o;
            x.ctx = c;
            {
                nb::gil_scoped_release release;
                x.b = std::make_shared<cuda::DeviceBrfs>(c->ctx, o.core->task, opts);
                x.r = x.b->run();
                if (x.r.result.solved)
                {
                    search::PlanValidation v = search::validate_plan(*o.core->task, x.r.result.plan);
                    x.cost = v.cost;
                    x.goal_state = std::move(v.state);
                }
                if (o.core->task->numeric_slots())
                {
                    const auto& t = *o.core->task;
                    const u32 atoms = x.b->words() - t.numeric_slots();
                    const u32 output_atoms = std::max<u32>(1, atoms);
                    const cudaStream_t stream = x.b->states().stream();
                    x.numeric_rows = device_output(*c, x.r.result.states * (output_atoms + t.numeric_words()) * sizeof(u64), OpStream{stream});
                    rl::dev::TaskView view;
                    view.numeric.slots = t.numeric_slots();
                    view.numeric.storage = t.numeric_storage() == NumericStorage::I32 ? 1 : 0;
                    cuda::DeviceGuard guard(c->ctx->device());
                    cuda::check(cuda::numeric::launch_convert(view,
                                    reinterpret_cast<const u64*>(x.b->states().device_data()), x.b->words(), atoms,
                                    static_cast<u64*>(x.numeric_rows->data()), output_atoms + t.numeric_words(), output_atoms,
                                    x.r.result.states, false, stream), "numeric BrFS output");
                }
            }
            return x;
        },
        "task"_a, nb::kw_only(), "ctx"_a = nb::none(), "witness_pruning"_a = true, "canonical_order"_a = true,
        "max_states"_a = nb::none(), "stop_at_goal"_a = true, "fingerprint"_a = false, "chunk_states"_a = nb::none(),
        "expected_states"_a = nb::none(), "max_depth"_a = nb::none(), "timings"_a = false,
        "The device layer BrFS: the ids equal the CPU BrFS's deterministic ids (mymyr.search.brfs with any thread "
        "count), whatever the chunk size; a search that expands a goal state is SOLVED with the CPU's plan to the first one, "
        "and stop_at_goal (the default) stops there. The state space stays on the device "
        "(state_words(), nodes()). Small layers run in device loops (one host read per loop); timings=True records "
        "the per-phase device times instead (counters; every chunk then runs from the host).");
    bind_cuda_search(m, [](PyTaskCore& core, nb::handle ctx, int device) { return task_context(core, ctx, device)->ctx; });
    bind_cuda_heuristics(m, [](PyTaskCore& core, nb::handle ctx, int device) { return task_context(core, ctx, device)->ctx; });
}
}  // namespace mymyr::python
