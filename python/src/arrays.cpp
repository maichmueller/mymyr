// Zero-copy array interop: our own DLPack exporter (DLArray), framework detection, state batch import. See arrays.hpp.

#include "arrays.hpp"

#include "dlpack.hpp"
#include "py_task.hpp"

#if defined(MYMYR_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

namespace
{
dl::DataType dl_dtype(rl::DType d)
{
    switch (d)
    {
        case rl::DType::Bool: return {dl::k_bool, 8, 1};
        case rl::DType::I8: return {dl::k_int, 8, 1};
        case rl::DType::U8: return {dl::k_uint, 8, 1};
        case rl::DType::I32: return {dl::k_int, 32, 1};
        case rl::DType::U32: return {dl::k_uint, 32, 1};
        case rl::DType::I64: return {dl::k_int, 64, 1};
        case rl::DType::U64: return {dl::k_uint, 64, 1};
        case rl::DType::F32: return {dl::k_float, 32, 1};
        case rl::DType::F64: return {dl::k_float, 64, 1};
    }
    return {dl::k_int, 32, 1};
}

const char* dtype_name(rl::DType d)
{
    switch (d)
    {
        case rl::DType::Bool: return "bool";
        case rl::DType::I8: return "int8";
        case rl::DType::U8: return "uint8";
        case rl::DType::I32: return "int32";
        case rl::DType::U32: return "uint32";
        case rl::DType::I64: return "int64";
        case rl::DType::U64: return "uint64";
        case rl::DType::F32: return "float32";
        case rl::DType::F64: return "float64";
    }
    return "?";
}

/// Our exporter. Memory is owned by `owner` (a C++ block); nothing here references a Python object, so capsule
/// deleters may run on any thread.
struct DLArray
{
    std::shared_ptr<const void> owner;
    const void* data = nullptr;
    rl::DType dtype = rl::DType::I32;
    std::vector<i64> shape, strides;  // strides in elements, always set
    dl::Device device{dl::k_cpu, 0};
    bool readonly = false;
    std::shared_ptr<DeviceSync> sync;  // device memory: the producer's stream handoff

    [[nodiscard]] u64 elements() const
    {
        u64 n = 1;
        for (i64 s : shape)
            n *= static_cast<u64>(s);
        return n;
    }
};

struct Ctx
{
    std::shared_ptr<const void> owner;
    std::vector<int64_t> shape, strides;
};

template<class M>
void delete_managed(M* m)
{
    delete static_cast<Ctx*>(m->manager_ctx);
    delete m;
}

template<class M>
void fill(M& m, const DLArray& a, const void* data, Ctx* ctx)
{
    m.dl_tensor.data = const_cast<void*>(data);
    m.dl_tensor.device = a.device;
    m.dl_tensor.ndim = static_cast<int32_t>(a.shape.size());
    m.dl_tensor.dtype = dl_dtype(a.dtype);
    m.dl_tensor.shape = ctx->shape.data();
    m.dl_tensor.strides = ctx->strides.data();
    m.dl_tensor.byte_offset = 0;  // offsets are folded into the pointer (torch rejects a byte_offset)
    m.manager_ctx = ctx;
    m.deleter = &delete_managed<M>;
}

void capsule_destructor_legacy(PyObject* cap) noexcept
{
    if (PyCapsule_IsValid(cap, "dltensor"))  // not consumed: we still own it
    {
        auto* m = static_cast<dl::ManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (m && m->deleter)
            m->deleter(m);
    }
}

void capsule_destructor_versioned(PyObject* cap) noexcept
{
    if (PyCapsule_IsValid(cap, "dltensor_versioned"))
    {
        auto* m = static_cast<dl::ManagedTensorVersioned*>(PyCapsule_GetPointer(cap, "dltensor_versioned"));
        if (m && m->deleter)
            m->deleter(m);
    }
}

nb::object dlpack_capsule(const DLArray& a, nb::handle stream, nb::handle max_version, nb::handle dl_device, nb::handle copy)
{
    // --- device / stream (DLPack array API)
    if (dl::is_cuda_family(a.device.device_type))
    {
        std::intptr_t s = dl::k_stream_legacy;  // None: the legacy default stream
        if (!stream.is_none())
        {
            if (!nb::isinstance<nb::int_>(stream))
                throw nb::type_error("mymyr: __dlpack__: stream must be an int or None");
            s = nb::cast<std::intptr_t>(stream);
        }
        if (s == 0)
            throw nb::buffer_error("mymyr: __dlpack__: stream 0 is ambiguous for CUDA (use 1 for the legacy default "
                                   "stream, 2 for the per-thread default stream)");
        if (s != dl::k_stream_none && a.sync)
            a.sync->handoff(s);
    }
    else if (a.device.device_type != dl::k_cpu)
        throw nb::buffer_error("mymyr: __dlpack__: unsupported device");
    else if (!stream.is_none())
    {
        // CPU arrays take no stream (DLPack spec); -1 means "no synchronization" and is accepted too.
        const bool ok = nb::isinstance<nb::int_>(stream) && nb::cast<i64>(stream) == -1;
        if (!ok)
            throw nb::buffer_error("mymyr: a stream was passed for a CPU array");
    }
    if (!dl_device.is_none())
    {
        auto t = nb::cast<nb::tuple>(dl_device);
        if (nb::len(t) != 2 || nb::cast<i32>(t[0]) != a.device.device_type || nb::cast<i32>(t[1]) != a.device.device_id)
            throw nb::buffer_error("mymyr: __dlpack__: only the array's own device is supported");
    }
    if (a.device.device_type != dl::k_cpu && !copy.is_none() && nb::cast<bool>(copy))
        throw nb::buffer_error("mymyr: __dlpack__: copy=True is not supported for device arrays (copy in the consumer)");
    bool versioned = false;
    if (!max_version.is_none())
    {
        auto t = nb::cast<nb::tuple>(max_version);
        versioned = nb::len(t) >= 1 && nb::cast<i64>(t[0]) >= 1;
    }
    // --- copy
    const void* data = a.data;
    std::shared_ptr<const void> owner = a.owner;
    bool copied = false;
    if (!copy.is_none() && nb::cast<bool>(copy))
    {
        // strides are always C-contiguous or row-strided views of our own blocks: copy element by element row-wise
        const u64 bytes = a.elements() * rl::dtype_bytes(a.dtype);
        auto block = std::shared_ptr<std::byte[]>(new std::byte[bytes ? bytes : 1]);
        if (bytes)
        {
            // general strided gather over the last dimension
            const u64 isz = rl::dtype_bytes(a.dtype);
            const usize nd = a.shape.size();
            std::vector<i64> idx(nd, 0);
            for (u64 e = 0; e < a.elements(); ++e)
            {
                i64 off = 0;
                for (usize d = 0; d < nd; ++d)
                    off += idx[d] * a.strides[d];
                std::memcpy(block.get() + e * isz, static_cast<const std::byte*>(a.data) + off * static_cast<i64>(isz), isz);
                for (usize d = nd; d-- > 0;)
                    if (++idx[d] < a.shape[d])
                        break;
                    else
                        idx[d] = 0;
            }
        }
        data = block.get();
        owner = std::shared_ptr<const void>(block, block.get());
        copied = true;
    }
    auto* ctx = new Ctx{owner, {a.shape.begin(), a.shape.end()}, {}};
    if (copied)
    {
        ctx->strides.assign(a.shape.size(), 1);
        for (usize d = a.shape.size(); d-- > 1;)
            ctx->strides[d - 1] = ctx->strides[d] * a.shape[d];
    }
    else
        ctx->strides.assign(a.strides.begin(), a.strides.end());
    if (versioned)
    {
        auto* m = new dl::ManagedTensorVersioned{};
        m->version = {dl::k_major, dl::k_minor};
        m->flags = (a.readonly && !copied ? dl::k_flag_read_only : 0) | (copied ? dl::k_flag_is_copied : 0);
        fill(*m, a, data, ctx);
        PyObject* cap = PyCapsule_New(m, "dltensor_versioned", &capsule_destructor_versioned);
        if (!cap)
        {
            delete_managed(m);
            throw nb::python_error();
        }
        return nb::steal(cap);
    }
    auto* m = new dl::ManagedTensor{};
    fill(*m, a, data, ctx);
    PyObject* cap = PyCapsule_New(m, "dltensor", &capsule_destructor_legacy);
    if (!cap)
    {
        delete_managed(m);
        throw nb::python_error();
    }
    return nb::steal(cap);
}

/// A Python capsule owning a copy of a shared_ptr: the base object of NumPy views.
nb::object owner_capsule(std::shared_ptr<const void> owner)
{
    auto* heap = new std::shared_ptr<const void>(std::move(owner));
    return nb::capsule(heap, [](void* p) noexcept { delete static_cast<std::shared_ptr<const void>*>(p); });
}

nb::object framework_convert(nb::object dlarray, Framework fw)
{
    // torch.from_dlpack / jax.dlpack.from_dlpack through a small Python helper (import lookups are cheap, and no
    // Python object is cached in C++ statics).
    return nb::module_::import_("mymyr._interop").attr("from_dlpack")(dlarray, framework_name(fw));
}
}  // namespace

// ------------------------------------------------------------------------------------------------ frameworks

Framework framework_of(nb::handle obj)
{
    if (nb::isinstance<DLArray>(obj))
        return Framework::DLPack;
    // Static and extension types carry the module in tp_name ("numpy.ndarray", "jaxlib._jax.ArrayImpl"). Reading
    // it avoids type.__module__, which for a static type builds and interns a string under an interpreter-wide lock
    // on every call (the hot path of rl.expand from many threads).
    const std::string_view tp_name = Py_TYPE(obj.ptr())->tp_name;
    if (tp_name.starts_with("numpy."))
        return Framework::Numpy;
    if (tp_name.starts_with("jax"))
        return Framework::Jax;
    if (tp_name.starts_with("torch."))
        return Framework::Torch;
    nb::object mod = nb::getattr(obj.type(), "__module__", nb::none());  // Python classes (torch.Tensor)
    if (!nb::isinstance<nb::str>(mod))
        return Framework::Numpy;
    const std::string_view m = nb::borrow<nb::str>(mod).c_str();
    if (m.starts_with("torch"))
        return Framework::Torch;
    if (m.starts_with("jax") || m.starts_with("jaxlib"))
        return Framework::Jax;
    return Framework::Numpy;
}

Framework parse_framework(nb::handle name, nb::handle like)
{
    if (name.is_none())
        return like.is_none() ? Framework::Numpy : framework_of(like);
    const std::string_view s = nb::cast<std::string_view>(name);
    if (s == "numpy" || s == "np")
        return Framework::Numpy;
    if (s == "torch" || s == "pytorch")
        return Framework::Torch;
    if (s == "jax")
        return Framework::Jax;
    if (s == "dlpack" || s == "array_api")
        return Framework::DLPack;
    throw nb::value_error("framework must be 'numpy', 'torch', 'jax' or 'dlpack'");
}

WordEncoding default_words(Framework fw)
{
    switch (fw)
    {
        case Framework::Torch: return {64, true};  // torch's uint64 has few operators
        case Framework::Jax: return {32, false};   // JAX runs without x64
        default: return {64, false};
    }
}

const char* framework_name(Framework fw)
{
    switch (fw)
    {
        case Framework::Numpy: return "numpy";
        case Framework::Torch: return "torch";
        case Framework::Jax: return "jax";
        case Framework::DLPack: return "dlpack";
    }
    return "numpy";
}

// ------------------------------------------------------------------------------------------------ blocks

namespace
{
/// Block memory released on this thread, reused by its next Block::make. Allocating fresh memory for every
/// rl.expand result cost a page fault per 4 KB page (a third of a 256-state expand) and serialized the threads on
/// the address-space lock. Keeps at most k_keep buffers of at most k_max_bytes each.
struct BlockCache
{
    static constexpr usize k_keep = 4;
    static constexpr u64 k_max_bytes = u64{4} << 20;

    struct Entry
    {
        std::unique_ptr<std::byte[]> raw;
        u64 capacity = 0;
    };
    std::vector<Entry> entries;

    ~BlockCache() { gone() = true; }
    /// Trivially destructible, so readable while and after the thread's cache is destroyed.
    static bool& gone()
    {
        thread_local bool flag = false;
        return flag;
    }
    static BlockCache* get()
    {
        if (gone())
            return nullptr;
        thread_local BlockCache cache;
        return &cache;
    }
};
}  // namespace

std::shared_ptr<Block> Block::make(u64 bytes)
{
    auto b = std::make_shared<Block>();
    const u64 need = bytes + 64;
    if (BlockCache* cache = BlockCache::get())
    {
        // the smallest cached buffer that fits and is not more than 4x the request
        auto best = cache->entries.end();
        for (auto it = cache->entries.begin(); it != cache->entries.end(); ++it)
            if (it->capacity >= need && it->capacity / 4 <= need && (best == cache->entries.end() || it->capacity < best->capacity))
                best = it;
        if (best != cache->entries.end())
        {
            b->m_raw = std::move(best->raw);
            b->m_capacity = best->capacity;
            cache->entries.erase(best);
        }
    }
    if (!b->m_raw)
    {
        // power-of-two sizes, so the varying sizes of successive batches reuse each other's buffers
        b->m_capacity = need <= BlockCache::k_max_bytes ? std::bit_ceil(need) : need;
        b->m_raw.reset(new std::byte[b->m_capacity]);
    }
    b->m_size = bytes;
    const auto addr = reinterpret_cast<std::uintptr_t>(b->m_raw.get());
    b->m_base = b->m_raw.get() + ((64 - addr % 64) % 64);
    return b;
}

Block::~Block()
{
    if (!m_raw || m_capacity > BlockCache::k_max_bytes)
        return;
    BlockCache* cache = BlockCache::get();
    if (!cache)
        return;
    if (cache->entries.size() >= BlockCache::k_keep)
        cache->entries.erase(cache->entries.begin());  // the oldest
    cache->entries.push_back({std::move(m_raw), m_capacity});
}

// ------------------------------------------------------------------------------------------------ export

namespace
{
nb::object export_impl(ArraySpec spec, const DeviceExport* device, Framework fw, WordEncoding enc)
{
    if (spec.strides.empty())
    {
        spec.strides.assign(spec.shape.size(), 1);
        for (usize d = spec.shape.size(); d-- > 1;)
            spec.strides[d - 1] = spec.strides[d] * spec.shape[d];
    }
    // 64-bit arrays that are not state words go to JAX as uint32 pairs as well: JAX without x64 would narrow them
    // to 32 bits, by a copy that drops the high halves
    const bool wide = spec.dtype == rl::DType::U64 || spec.dtype == rl::DType::I64;
    const bool contiguous_last = spec.shape.empty() || spec.shape.back() <= 1 || spec.strides.back() == 1;
    const bool pairs = spec.words ? enc.bits == 32 : (fw == Framework::Jax && wide && contiguous_last);
    if (pairs)
    {
        // [.., W] u64 -> [.., 2W] u32 over the same bytes (little-endian: low half first)
        spec.dtype = spec.words && enc.is_signed ? rl::DType::I32 : rl::DType::U32;
        for (i64& s : spec.strides)
            s *= 2;
        if (!spec.shape.empty())
        {
            spec.shape.back() *= 2;
            spec.strides.back() = 1;
        }
    }
    else if (spec.words)
        spec.dtype = enc.is_signed ? rl::DType::I64 : rl::DType::U64;
    const bool on_host = !device || device->device_type == dl::k_cpu;
    if (fw == Framework::Numpy)
    {
        if (!on_host)
            throw nb::type_error("mymyr: NumPy arrays hold host memory only: export device arrays with "
                                 "framework='torch', 'jax' or 'dlpack'");
        std::vector<size_t> shape(spec.shape.begin(), spec.shape.end());
        std::vector<int64_t> strides(spec.strides.begin(), spec.strides.end());
        const dl::DataType d = dl_dtype(spec.dtype);
        const nb::dlpack::dtype nd{d.code, d.bits, d.lanes};
        nb::object base = owner_capsule(spec.owner);
        if (spec.readonly)
            return nb::ndarray<nb::numpy, nb::ro>(spec.data, shape.size(), shape.data(), base, strides.data(), nd,
                                                  nb::device::cpu::value)
                .cast();
        return nb::ndarray<nb::numpy>(const_cast<void*>(spec.data), shape.size(), shape.data(), base, strides.data(), nd,
                                      nb::device::cpu::value)
            .cast();
    }
    DLArray a;
    a.owner = std::move(spec.owner);
    a.data = spec.data;
    a.dtype = spec.dtype;
    a.shape = std::move(spec.shape);
    a.strides = std::move(spec.strides);
    a.readonly = spec.readonly;
    if (device)
    {
        a.device = {device->device_type, device->device_id};
        a.sync = device->sync;
    }
    nb::object obj = nb::cast(std::move(a), nb::rv_policy::move);
    if (fw == Framework::DLPack)
        return obj;
    return framework_convert(obj, fw);
}
}  // namespace

nb::object export_array(ArraySpec spec, Framework fw, WordEncoding enc) { return export_impl(std::move(spec), nullptr, fw, enc); }

nb::object export_device_array(ArraySpec spec, DeviceExport device, Framework fw, WordEncoding enc)
{
    return export_impl(std::move(spec), &device, fw, enc);
}

nb::object export_bundle(const std::shared_ptr<const rl::ArrayBundle>& bundle, Framework fw, WordEncoding enc)
{
    nb::dict d;
    for (const auto& [name, value] : bundle->scalars())
        d[nb::str(name.c_str())] = value;
    for (const rl::ArrayInfo& a : bundle->arrays())
    {
        ArraySpec s;
        s.owner = std::shared_ptr<const void>(bundle, bundle->data(a));
        s.data = bundle->data(a);
        s.dtype = a.dtype;
        s.shape = a.shape;
        s.readonly = true;  // snapshots are shared and immutable
        s.words = a.words;
        d[nb::str(a.name.c_str())] = export_array(std::move(s), fw, enc);
    }
    return d;
}

// ------------------------------------------------------------------------------------------------ import

namespace dl
{
Device dlpack_device(nb::handle obj)
{
    nb::object f = nb::getattr(obj, "__dlpack_device__", nb::none());
    if (f.is_none())
        return {-1, -1};
    nb::tuple t = nb::cast<nb::tuple>(f());
    return {nb::cast<int32_t>(t[0]), nb::cast<int32_t>(t[1])};
}

Imported import_dlpack(nb::handle obj, std::intptr_t stream)
{
    nb::object f = nb::getattr(obj, "__dlpack__", nb::none());
    if (f.is_none())
        throw nb::type_error("mymyr: expected an array that speaks DLPack (__dlpack__)");
    nb::object cap;
    const nb::object st = stream == k_stream_default ? nb::none() : nb::object(nb::int_(stream));
    try
    {
        cap = f("stream"_a = st, "max_version"_a = nb::make_tuple(k_major, k_minor));
    }
    catch (nb::python_error& e)
    {
        if (!e.matches(PyExc_TypeError))
            throw;
        cap = f("stream"_a = st);  // producers predating max_version (array API < 2023.12)
    }
    Imported im;
    Tensor* t = nullptr;
    PyObject* c = cap.ptr();
    if (PyCapsule_IsValid(c, "dltensor_versioned"))
    {
        auto* m = static_cast<ManagedTensorVersioned*>(PyCapsule_GetPointer(c, "dltensor_versioned"));
        if (m->version.major > k_major)
            throw nb::buffer_error("mymyr: the producer's DLPack major version is newer than 1");
        if (PyCapsule_SetName(c, "used_dltensor_versioned") != 0)
            throw nb::python_error();
        im.keep = std::shared_ptr<void>(m, [](void* p) {
            auto* x = static_cast<ManagedTensorVersioned*>(p);
            if (x->deleter)
                x->deleter(x);
        });
        im.readonly = (m->flags & k_flag_read_only) != 0;
        t = &m->dl_tensor;
    }
    else if (PyCapsule_IsValid(c, "dltensor"))
    {
        auto* m = static_cast<ManagedTensor*>(PyCapsule_GetPointer(c, "dltensor"));
        if (PyCapsule_SetName(c, "used_dltensor") != 0)
            throw nb::python_error();
        im.keep = std::shared_ptr<void>(m, [](void* p) {
            auto* x = static_cast<ManagedTensor*>(p);
            if (x->deleter)
                x->deleter(x);
        });
        t = &m->dl_tensor;
    }
    else
        throw nb::type_error("mymyr: __dlpack__ did not return a DLPack capsule");
    im.data = static_cast<std::byte*>(t->data) + t->byte_offset;
    im.device = t->device;
    im.dtype = t->dtype;
    im.shape.assign(t->shape, t->shape + t->ndim);
    if (t->strides)
        im.strides.assign(t->strides, t->strides + t->ndim);
    else
    {
        im.strides.assign(static_cast<usize>(t->ndim), 1);
        for (int32_t d = t->ndim; d-- > 1;)
            im.strides[static_cast<usize>(d - 1)] = im.strides[static_cast<usize>(d)] * im.shape[static_cast<usize>(d)];
    }
    return im;
}
}  // namespace dl

WordsLayout words_layout(const void* data, u8 code, u8 bits, u16 lanes, usize ndim, const i64* shape, const i64* strides)
{
    const bool integer = code == dl::k_int || code == dl::k_uint;
    if (!integer || (bits != 64 && bits != 32) || lanes != 1)
        throw nb::type_error("mymyr: state words must be 64-bit integers [N, W] or 32-bit integers [N, 2W]");
    if (ndim != 1 && ndim != 2)
        throw nb::type_error("mymyr: state words must be 1-d (one state) or 2-d (a batch)");
    const usize last = ndim - 1;
    const i64 cols = shape[last];
    const i64 rows = ndim == 2 ? shape[0] : 1;
    if (cols > 1 && strides[last] != 1)
        throw nb::type_error("mymyr: the words of a state must be contiguous (last-dimension stride 1)");
    i64 row_stride = ndim == 2 ? strides[0] : cols;
    if (row_stride < 0)
        throw nb::type_error("mymyr: negative row strides are not supported");
    u32 words = static_cast<u32>(cols);
    if (bits == 32)
    {
        if (cols % 2 != 0 || (rows > 1 && row_stride % 2 != 0))
            throw nb::type_error("mymyr: 32-bit state words need an even number of columns and an even row stride "
                                 "([N, 2W])");
        words = static_cast<u32>(cols / 2);
        row_stride /= 2;
    }
    if (reinterpret_cast<std::uintptr_t>(data) % 8 != 0 && rows > 0 && words > 0)
        throw nb::type_error("mymyr: state words must be 8-byte aligned");
    WordsLayout w;
    w.data = static_cast<const u64*>(data);
    w.rows = static_cast<u64>(rows);
    w.words = words;
    w.stride = static_cast<u64>(rows > 1 ? row_stride : words);
    if (w.stride == 0)
        w.stride = words;
    w.single = ndim == 1;
    w.enc = {bits, code == dl::k_int};
    return w;
}

namespace
{
[[noreturn]] void device_not_host(int32_t type, int32_t id)
{
    if (type == dl::k_cuda)
        throw nb::type_error(("mymyr: this operation runs on the CPU and cannot read a CUDA device array (cuda:" +
                              std::to_string(id) + "; rl.expand and rl.expand_into take device arrays in CUDA builds); "
                              "copy it to host memory (tensor.cpu(), jax.device_get) or pass pinned or managed memory, "
                              "which is read in place")
                                 .c_str());
    throw nb::type_error("mymyr: only CPU, pinned-host and managed CUDA arrays can be read by host operations");
}

/// Host-accessible CUDA memory (pinned, managed): the producer hands off to the legacy default stream, which the host
/// then waits for, so every write the producer enqueued before is visible to the host reads that follow.
dl::Imported import_host_accessible(nb::handle obj)
{
#if defined(MYMYR_HAS_CUDA)
    dl::Imported im;
    try
    {
        im = dl::import_dlpack(obj, dl::k_stream_legacy);
    }
    catch (nb::python_error&)
    {
        // torch exports pinned tensors through its CPU path, which takes no stream (ordering its own asynchronous
        // writes to them is the caller's job, as in torch): retry with the producer's default
        im = dl::import_dlpack(obj, dl::k_stream_default);
    }
    if (!dl::host_accessible(im.device.device_type))
        device_not_host(im.device.device_type, im.device.device_id);
    const cudaError_t e = cudaStreamSynchronize(cudaStreamLegacy);
    if (e != cudaSuccess)
    {
        (void)cudaGetLastError();
        throw nb::type_error((std::string("mymyr: synchronizing with the producer of a CUDA array failed: ") +
                              cudaGetErrorString(e))
                                 .c_str());
    }
    return im;
#else
    (void)obj;
    throw nb::type_error("mymyr: pinned-host and managed CUDA arrays need a CUDA build of mymyr "
                         "(CMAKE_ARGS=\"-DMYMYR_CUDA=ON\") to synchronize with their producer; pass CPU arrays");
#endif
}
}  // namespace

StateBatch import_states(nb::handle obj, u32 task_words)
{
    StateBatch b;
    if (is_state(obj))
    {
        const State& s = state_of(obj).s;
        b.packed = std::make_shared<std::vector<u64>>(s.words().begin(), s.words().end());
        b.view = {b.packed->data(), 1, static_cast<u32>(b.packed->size()), 0};
        b.single = true;
        b.enc = {64, false};
        b.owners = {state_of(obj).core->task->uid()};
        return b;
    }
    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))
    {
        nb::sequence seq = nb::borrow<nb::sequence>(obj);
        const usize n = nb::len(seq);
        u32 W = task_words;
        std::vector<const State*> states(n);
        for (usize i = 0; i < n; ++i)
        {
            nb::handle item = seq[i];
            if (!is_state(item))
                throw nb::type_error("mymyr: expected an array of state words, a State, or a sequence of States");
            states[i] = &state_of(item).s;
            b.owners.push_back(state_of(item).core->task->uid());
            W = std::max(W, states[i]->size_words());
        }
        b.packed = std::make_shared<std::vector<u64>>(n * W, 0);
        for (usize i = 0; i < n; ++i)
            std::copy(states[i]->words().begin(), states[i]->words().end(), b.packed->begin() + static_cast<std::ptrdiff_t>(i * W));
        b.view = {b.packed->data(), n, W, 0};
        b.enc = {64, false};
        return b;
    }
    nb::ndarray<nb::ro> arr;
    if (!nb::try_cast(obj, arr))
        throw nb::type_error("mymyr: expected an array of state words (NumPy, torch or JAX; [N, W] 64-bit or [N, 2W] "
                             "32-bit integers), a State, or a sequence of States");
    WordsLayout w;
    if (arr.device_type() == nb::device::cpu::value)
    {
        std::vector<i64> shape(arr.ndim()), strides(arr.ndim());
        for (usize i = 0; i < arr.ndim(); ++i)
        {
            shape[i] = static_cast<i64>(arr.shape(i));
            strides[i] = arr.stride(i);
        }
        const nb::dlpack::dtype dt = arr.dtype();
        w = words_layout(arr.data(), dt.code, dt.bits, dt.lanes, arr.ndim(), shape.data(), strides.data());
        b.keep_native = std::make_shared<nb::ndarray<nb::ro>>(std::move(arr));
    }
    else
    {
        // nanobind imported without a stream: re-import through our DLPack path with the stream handoff
        const int32_t type = arr.device_type(), id = arr.device_id();
        arr = nb::ndarray<nb::ro>();
        if (type != dl::k_cuda_host && type != dl::k_cuda_managed)
            device_not_host(type, id);
        dl::Imported im = import_host_accessible(obj);
        w = words_layout(im.data, im.dtype.code, im.dtype.bits, im.dtype.lanes, im.shape.size(), im.shape.data(),
                         im.strides.data());
        b.keep_native = std::move(im.keep);
    }
    b.view = {w.data, w.rows, w.words, w.stride};
    b.single = w.single;
    b.fw = framework_of(obj);
    b.enc = w.enc;
    return b;
}

Dest import_dest(nb::handle obj, const char* name)
{
    Dest d;
    if (framework_of(obj) == Framework::Jax)  // JAX arrays are immutable values; rl.expand returns fresh ones
        throw nb::type_error((std::string("mymyr: '") + name +
                              "' is a JAX array; JAX arrays are immutable, use rl.expand(..., framework='jax')")
                                 .c_str());
    nb::ndarray<> arr;
    if (!nb::try_cast(obj, arr))
        throw nb::type_error((std::string("mymyr: '") + name + "' must be a writable CPU array").c_str());
    if (arr.device_type() != nb::device::cpu::value)
        throw nb::type_error((std::string("mymyr: '") + name +
                              "' must be a writable CPU array: host operations write host memory (device outputs "
                              "arrive with the device expand)")
                                 .c_str());
    const nb::dlpack::dtype dt = arr.dtype();
    const bool sgn = dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    const bool uns = dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
    const bool boo = dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool);
    if (boo && dt.bits == 8)
        d.dtype = rl::DType::Bool;
    else if ((sgn || uns) && dt.bits == 8)
        d.dtype = sgn ? rl::DType::I8 : rl::DType::U8;
    else if ((sgn || uns) && dt.bits == 32)
        d.dtype = sgn ? rl::DType::I32 : rl::DType::U32;
    else if ((sgn || uns) && dt.bits == 64)
        d.dtype = sgn ? rl::DType::I64 : rl::DType::U64;
    else
        throw nb::type_error((std::string("mymyr: '") + name + "' has an unsupported dtype").c_str());
    d.data = arr.data();
    for (usize i = 0; i < arr.ndim(); ++i)
    {
        d.shape.push_back(static_cast<i64>(arr.shape(i)));
        d.strides.push_back(arr.stride(i));
    }
    d.keep = nb::cast(arr);
    return d;
}

// ------------------------------------------------------------------------------------------------ bindings

void bind_arrays(nb::module_& m)
{
    nb::class_<DLArray>(m, "DLArray",
                        "A read-only or fresh array owned by mymyr, exported through DLPack (__dlpack__ / "
                        "__dlpack_device__). Frameworks wrap it zero-copy: numpy.from_dlpack, torch.from_dlpack, "
                        "jax.dlpack.from_dlpack.")
        .def("__dlpack__", &dlpack_capsule, nb::kw_only(), "stream"_a = nb::none(), "max_version"_a = nb::none(),
             "dl_device"_a = nb::none(), "copy"_a = nb::none())
        .def("__dlpack_device__", [](const DLArray& a) { return nb::make_tuple(a.device.device_type, a.device.device_id); })
        .def_prop_ro("shape", [](const DLArray& a) {
            nb::list l;
            for (i64 s : a.shape)
                l.append(s);
            return nb::tuple(l);
        })
        .def_prop_ro("strides", [](const DLArray& a) {
            nb::list l;
            for (i64 s : a.strides)
                l.append(s * static_cast<i64>(rl::dtype_bytes(a.dtype)));
            return nb::tuple(l);
        }, "strides in bytes")
        .def_prop_ro("dtype", [](const DLArray& a) { return std::string(dtype_name(a.dtype)); })
        .def_prop_ro("ndim", [](const DLArray& a) { return a.shape.size(); })
        .def_prop_ro("readonly", [](const DLArray& a) { return a.readonly; })
        .def_prop_ro("data_ptr", [](const DLArray& a) { return reinterpret_cast<std::uintptr_t>(a.data); })
        .def_prop_ro("nbytes", [](const DLArray& a) { return a.elements() * rl::dtype_bytes(a.dtype); })
        .def("__len__", [](const DLArray& a) -> i64 {
            if (a.shape.empty())
                throw nb::type_error("len() of a 0-d array");
            return a.shape[0];
        })
        .def("__repr__", [](const DLArray& a) {
            std::string s = "DLArray(shape=(";
            for (usize i = 0; i < a.shape.size(); ++i)
                s += (i ? ", " : "") + std::to_string(a.shape[i]);
            if (a.shape.size() == 1)
                s += ",";
            return s + "), dtype=" + dtype_name(a.dtype) + ")";
        });

    m.def(
        "_import_info",
        [](nb::handle obj) {
            StateBatch b = import_states(obj, 0);
            return nb::make_tuple(reinterpret_cast<std::uintptr_t>(b.view.data), b.view.rows, b.view.words, b.view.stride,
                                  framework_name(b.fw), static_cast<int>(b.enc.bits), b.enc.is_signed);
        },
        "obj"_a, "(data pointer, rows, words, row stride in words, framework, word bits, signed) as imported: tests use "
                 "it to check that inputs are not copied");
}
}  // namespace mymyr::python
