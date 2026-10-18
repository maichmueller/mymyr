// mymyr._core._encoding.DeviceEncoder (CUDA builds only): the native side of mymyr.encoding's device path
// (mymyr/encoding/_device.py), which encodes rows in CUDA memory into torch tensors. The class wraps a
// cuda::DeviceEncoder and its context upload for one FlatRelationEncoder on one device, and runs it on raw device
// pointers that the Python side has checked (dtype, shape, device, contiguity) and keeps alive, on a stream the caller
// names (torch's current stream): in stream order, like torch's own kernels, and without the DLPack imports.
//
// Each stream the encoder runs on has a slot with one held workspace (the scratch and a lease of the context
// generation), made by the first call on the stream or by reserve() and kept until release(): a call takes nothing new
// after that, which is what a CUDA graph capture needs. A capture on a stream whose slot cannot hold its batch is
// refused (a reservation allocates and records events, which a capture does not allow). The calls of one slot take turns:
// they share the scratch.
//
//   encode:        everything on the stream, outputs sized by the capacities: no host synchronization, capturable.
//   encode_exact:  counts, copies the sizes to the host and waits (the one synchronization), has `allocate` make the
//                  outputs of exactly that size, and fills them.

#include "arrays.hpp"
#include "dlpack.hpp"
#include "typing.hpp"

#include "mymyr/cuda/encoding.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/encoding/encoder.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

std::shared_ptr<const encoding::FlatEncoder> flat_encoder_of(nb::handle obj);  // encoding_bindings.cpp
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device);  // cuda_bindings.cpp

namespace
{
[[noreturn]] void refuse(const std::string& why) { throw nb::value_error(("mymyr: encoding: " + why).c_str()); }

/// The device pointers and strides of one call's rows, in the order of the tuple that the Python side passes.
struct Rows
{
    u64 graphs = 0;
    std::uintptr_t states = 0;
    u64 state_stride = 0;
    u32 state_words = 0;
    std::uintptr_t task_ids = 0, derived = 0;
    u64 derived_stride = 0;
    std::uintptr_t goal_pos = 0, goal_neg = 0;
    u64 goal_stride = 0;
    u32 goal_words = 0;
    std::uintptr_t goal_derived_pos = 0, goal_derived_neg = 0;
    u64 goal_derived_stride = 0;
    u32 goal_derived_words = 0;
};
constexpr usize k_row_fields = 15;

Rows rows_of(const std::vector<std::uintptr_t>& v)
{
    if (v.size() != k_row_fields)
        throw nb::value_error(("mymyr: expected " + std::to_string(k_row_fields) + " row fields").c_str());
    Rows r;
    usize i = 0;
    r.graphs = v[i++];
    r.states = v[i++];
    r.state_stride = v[i++];
    r.state_words = static_cast<u32>(v[i++]);
    r.task_ids = v[i++];
    r.derived = v[i++];
    r.derived_stride = v[i++];
    r.goal_pos = v[i++];
    r.goal_neg = v[i++];
    r.goal_stride = v[i++];
    r.goal_words = static_cast<u32>(v[i++]);
    r.goal_derived_pos = v[i++];
    r.goal_derived_neg = v[i++];
    r.goal_derived_stride = v[i++];
    r.goal_derived_words = static_cast<u32>(v[i++]);
    return r;
}

template<class T>
T* ptr(std::uintptr_t p)
{
    return reinterpret_cast<T*>(p);
}

cuda::DeviceEncodeInput input_of(const Rows& r)
{
    cuda::DeviceEncodeInput in;
    in.graphs = r.graphs;
    in.states = ptr<const u64>(r.states);
    in.state_stride = r.state_stride;
    in.state_words = r.state_words;
    in.task_ids = ptr<const i32>(r.task_ids);
    in.derived = ptr<const u64>(r.derived);
    in.derived_stride = r.derived_stride;
    in.goal_pos = ptr<const u64>(r.goal_pos);
    in.goal_neg = ptr<const u64>(r.goal_neg);
    in.goal_stride = r.goal_stride;
    in.goal_words = r.goal_words;
    in.goal_derived_pos = ptr<const u64>(r.goal_derived_pos);
    in.goal_derived_neg = ptr<const u64>(r.goal_derived_neg);
    in.goal_derived_stride = r.goal_derived_stride;
    in.goal_derived_words = r.goal_derived_words;
    return in;
}

/// The destination arrays, in the order of encoding._device: task_ids, num_objects, entity_offsets, entity_role,
/// entity_ref, relation_counts, relation_offsets, relation_arg_offsets, relation_args, goal_complete, target_offsets,
/// target_entity, target_group, sizes (0: the array is absent or has no entries).
constexpr usize k_output_fields = 14;

cuda::DeviceOutput output_of(const std::vector<std::uintptr_t>& v, const encoding::BatchShape& capacities)
{
    if (v.size() != k_output_fields)
        throw nb::value_error(("mymyr: expected " + std::to_string(k_output_fields) + " output pointers").c_str());
    cuda::DeviceOutput out;
    encoding::FlatBuffers& a = out.arrays;
    a.task_ids = ptr<i32>(v[0]);
    a.num_objects = ptr<void>(v[1]);
    a.entity_offsets = ptr<void>(v[2]);
    a.entity_role = ptr<u8>(v[3]);
    a.entity_ref = ptr<void>(v[4]);
    a.relation_counts = ptr<void>(v[5]);
    a.relation_offsets = ptr<void>(v[6]);
    a.relation_arg_offsets = ptr<void>(v[7]);
    a.relation_args = ptr<void>(v[8]);
    a.goal_complete = ptr<u8>(v[9]);
    a.target_offsets = ptr<void>(v[10]);
    a.target_entity = ptr<void>(v[11]);
    a.target_group = ptr<u8>(v[12]);
    out.sizes = ptr<encoding::FlatSizes>(v[13]);
    out.capacities = capacities;
    return out;
}

/// The cudaStream_t of a stream handle: torch reports its default stream as 0 and the array API's handles 1 and 2 are the
/// legacy and the per-thread default streams (the library would take 0 for the context's own stream).
cudaStream_t stream_of(std::intptr_t handle)
{
    if (handle == 0 || handle == 1)
        return cudaStreamLegacy;
    if (handle == 2)
        return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(handle);
}

bool capturing(int device, cudaStream_t stream)
{
    cuda::DeviceGuard guard(device);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cuda::check(cudaStreamIsCapturing(stream, &status), "cudaStreamIsCapturing");
    return status != cudaStreamCaptureStatusNone;
}

/// What the encoder keeps for one stream.
struct StreamSlot
{
    std::mutex mutex;  // one call at a time: the calls share the scratch (and count and fill share its totals)
    std::optional<cuda::EncodingWorkspace> held;
    std::shared_ptr<cuda::DeviceBuffer> sizes;  // the FlatSizes of a count
    cuda::PinnedLease staging;                  // its host copy
    u64 reservations = 0;
    u64 syncs = 0;
};

class DeviceEncoderHandle
{
public:
    DeviceEncoderHandle(nb::handle encoder, int device, nb::handle ctx) : m_device(device)
    {
        m_encoder = flat_encoder_of(encoder);
        m_instances = m_encoder->table().size();
        cuda::ContextPtr c = table_device_context(nb::getattr(encoder, "table"), ctx, device);
        if (c->device() != device)
            refuse("the context is on cuda:" + std::to_string(c->device()) + ", not cuda:" + std::to_string(device));
        nb::gil_scoped_release release;
        m_device_encoder = std::make_unique<cuda::DeviceEncoder>(std::move(c), m_encoder);
    }
    ~DeviceEncoderHandle()
    {
        try
        {
            m_slots.clear();
        }
        catch (...)
        {
        }
    }
    DeviceEncoderHandle(const DeviceEncoderHandle&) = delete;
    DeviceEncoderHandle& operator=(const DeviceEncoderHandle&) = delete;

    [[nodiscard]] int device() const noexcept { return m_device; }

    /// Makes the stream's workspace hold batches of up to `graphs` graphs. Not during a capture.
    void reserve(std::intptr_t stream, u64 graphs)
    {
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure(*slot, stream_of(stream), graphs);
    }

    /// Frees the workspace of `stream` (stream != 0) or of every stream, and with `all` also the uploaded context.
    void release(std::optional<std::intptr_t> stream)
    {
        nb::gil_scoped_release release;
        std::map<cudaStream_t, std::shared_ptr<StreamSlot>> dropped;
        {
            std::lock_guard lock(m_mutex);
            if (stream)
            {
                const auto it = m_slots.find(stream_of(*stream));
                if (it != m_slots.end())
                {
                    dropped.emplace(it->first, std::move(it->second));
                    m_slots.erase(it);
                }
            }
            else
                dropped.swap(m_slots);
        }
        for (auto& [s, slot] : dropped)
        {
            std::lock_guard lock(slot->mutex);
            slot->held.reset();
            m_device_encoder->release(s);
        }
    }

    /// The graphs reserved on a stream, the bytes of its scratch, the reservations made and the host waits so far.
    [[nodiscard]] std::vector<u64> state(std::intptr_t stream) const
    {
        std::shared_ptr<StreamSlot> slot;
        {
            std::lock_guard lock(m_mutex);
            const auto it = m_slots.find(stream_of(stream));
            if (it != m_slots.end())
                slot = it->second;
        }
        if (!slot)
            return {0, 0, 0, 0};
        std::lock_guard lock(slot->mutex);
        return {slot->held ? slot->held->shape().graphs : 0, slot->held ? slot->held->scratch_bytes() : 0, slot->reservations,
                slot->syncs};
    }

    /// Encodes the rows into `out` with `capacities` {graphs, entities, args, targets}. No synchronization.
    void encode(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities)
    {
        if (capacities.size() != 4)
            throw nb::value_error("mymyr: expected four capacities");
        const Rows rows = rows_of(row_fields);
        const encoding::BatchShape caps{capacities[0], capacities[1], capacities[2], capacities[3]};
        const cuda::DeviceOutput out = output_of(outputs, caps);
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        const cudaStream_t s = stream_of(stream);
        ensure(*slot, s, rows.graphs);
        check_rows(rows);
        m_device_encoder->encode(*slot->held, input_of(rows), out);
    }

    /// Counts the rows, waits for the sizes, calls `allocate(entities, tuples, args, targets)` for the output pointers
    /// (the order of `encode`'s `outputs`, the sizes pointer ignored) and fills them.
    void encode_exact(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                      const std::function<std::vector<std::uintptr_t>(u64, u64, u64, u64)>& allocate)
    {
        const Rows rows = rows_of(row_fields);
        std::vector<std::uintptr_t> outputs;
        {
            nb::gil_scoped_release release;
            const auto slot = slot_of(stream);
            std::lock_guard lock(slot->mutex);
            const cudaStream_t s = stream_of(stream);
            if (capturing(m_device, s))
                refuse("an encode without capacities waits for the batch's sizes and cannot be captured in a CUDA graph: "
                       "pass capacities");
            ensure(*slot, s, rows.graphs);
            check_rows(rows);
            const cuda::DeviceEncodeInput in = input_of(rows);
            auto* device_sizes = static_cast<encoding::FlatSizes*>(slot->sizes->data());
            cuda::DeviceOutput counted;
            counted.capacities.graphs = rows.graphs;
            counted.sizes = device_sizes;
            m_device_encoder->count(*slot->held, in, counted);
            cuda::DeviceGuard guard(m_device);
            cuda::check(cudaMemcpyAsync(slot->staging.data(0), device_sizes, sizeof(encoding::FlatSizes), cudaMemcpyDeviceToHost, s),
                        "cudaMemcpyAsync (sizes)");
            cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize (sizes)");
            ++slot->syncs;
            const encoding::FlatSizes sizes = *static_cast<const encoding::FlatSizes*>(slot->staging.data(0));
            if (sizes.overflow & encoding::k_overflow_input)
                refuse("a row is invalid: a task id outside the table, a state row or goal mask narrower than its instance, "
                       "or a bit that its instance has no atom for");
            if (sizes.overflow & encoding::k_overflow_index)
                throw std::overflow_error(
                    "mymyr: encoding: the batch needs more indices than its index type holds (entities " +
                    std::to_string(sizes.entities) + ", tuples " + std::to_string(sizes.tuples) + ", args " +
                    std::to_string(sizes.args) + ", targets " + std::to_string(sizes.targets) +
                    "): use index_dtype='int64'");
            outputs = allocate(sizes.entities, sizes.tuples, sizes.args, sizes.targets);  // takes the GIL itself
            const encoding::BatchShape caps{rows.graphs, sizes.entities, sizes.args, sizes.targets};
            cuda::DeviceOutput out = output_of(outputs, caps);
            out.sizes = device_sizes;
            m_device_encoder->fill(*slot->held, in, out);
        }
    }

private:
    std::shared_ptr<StreamSlot> slot_of(std::intptr_t stream)
    {
        std::lock_guard lock(m_mutex);
        auto& slot = m_slots[stream_of(stream)];
        if (!slot)
            slot = std::make_shared<StreamSlot>();
        return slot;
    }

    /// The held workspace of the slot holds `graphs` graphs: reserved and acquired now unless it does already. A
    /// capture refuses that, naming the reservation that the caller makes before it.
    void ensure(StreamSlot& slot, cudaStream_t stream, u64 graphs)
    {
        if (slot.held && slot.held->shape().graphs >= graphs)
            return;
        if (capturing(m_device, stream))
            refuse("a CUDA graph capture needs the encoder's device workspace for at least " + std::to_string(graphs) +
                   " rows on its stream, which it does not have: call encoder.reserve_device(" + std::to_string(graphs) +
                   ", stream=<the capture stream>) before the capture");
        // sized to a power of two so that batches that grow a little reserve again rarely
        const u64 want = std::bit_ceil(std::max<u64>(graphs, 1024));
        cuda::DeviceGuard guard(m_device);
        if (!slot.sizes)
        {
            slot.sizes = std::make_shared<cuda::DeviceBuffer>(m_device_encoder->context().context(),
                                                              sizeof(encoding::FlatSizes), stream);
            slot.staging = m_device_encoder->context().context()->lease_pinned(1, sizeof(encoding::FlatSizes));
        }
        m_device_encoder->reserve(cuda::BatchShape{want, 0, 0, 0}, stream);
        slot.held.reset();
        slot.held.emplace(m_device_encoder->acquire(stream));
        ++slot.reservations;
    }

    /// What the host can check about the rows without reading device memory.
    void check_rows(const Rows& r) const
    {
        if (r.graphs && !r.task_ids && m_instances > 1)
            refuse("task_ids are required for a table of several instances");
    }

    int m_device;
    std::shared_ptr<const encoding::FlatEncoder> m_encoder;
    u32 m_instances = 1;
    std::unique_ptr<cuda::DeviceEncoder> m_device_encoder;
    mutable std::mutex m_mutex;
    std::map<cudaStream_t, std::shared_ptr<StreamSlot>> m_slots;
};
}  // namespace

void bind_encoding_device(nb::module_& parent)
{
    nb::module_ module = nb::borrow<nb::module_>(parent.attr("_encoding"));
    nb::class_<DeviceEncoderHandle>(
        module, "DeviceEncoder",
        "The device side of a FlatRelationEncoder on one CUDA device (mymyr.encoding keeps one per encoder and device). "
        "It runs on raw device pointers that its caller has checked, on the stream the caller names; not part of the "
        "public API.")
        .def(
            "__init__",
            [](DeviceEncoderHandle* self, Arg<ann::Any> encoder, int device, Arg<ann::Any> ctx)
            { new (self) DeviceEncoderHandle(encoder, device, ctx); },
            "encoder"_a, "device"_a, "ctx"_a = nb::none(),
             "Uploads the encoder's context to `device` (on `ctx`, a mymyr.cuda.Context, or the table's default context). "
             "Raises ValueError for options the device does not encode.")
        .def_prop_ro("device", &DeviceEncoderHandle::device, "The CUDA device index.")
        .def("reserve", &DeviceEncoderHandle::reserve, "stream"_a, "graphs"_a,
             "Makes the workspace of `stream` (a cudaStream_t, 0: the default stream) hold at least `graphs` graphs. "
             "Not during a capture.")
        .def("release", &DeviceEncoderHandle::release, "stream"_a = nb::none(),
             "Frees the workspace of `stream`, or of every stream.")
        .def("state", &DeviceEncoderHandle::state, "stream"_a,
             "[graphs reserved, scratch bytes, reservations made, host waits made] of a stream.")
        .def("encode", &DeviceEncoderHandle::encode, "stream"_a, "rows"_a, "outputs"_a, "capacities"_a,
             "Encodes the rows (15 integers: the graphs, then the pointers and strides of the arrays) into the 14 output "
             "pointers, with the capacities (entities, tuples, args, targets), on `stream`. No host wait.")
        .def("encode_exact", &DeviceEncoderHandle::encode_exact, "stream"_a, "rows"_a, "allocate"_a,
             "Counts the rows, waits once for their sizes, calls allocate(entities, tuples, args, targets) for the 14 "
             "output pointers and fills them, on `stream`.");
}
}  // namespace mymyr::python
