// mymyr._core._encoding.DeviceEncoder (CUDA builds only): the native side of mymyr.encoding's device path
// (mymyr/encoding/_device.py), which encodes rows in CUDA memory into torch tensors. The class wraps a
// cuda::DeviceEncoder and its context upload for one FlatRelationEncoder on one device, and runs it on raw device
// pointers that the Python side has checked (dtype, shape, device, contiguity) and keeps alive, on a stream the caller
// names (torch's current stream): in stream order, like torch's own kernels, and without the DLPack imports.
//
// Each stream holds a context acquisition through release(). Capacity calls take caller-owned scratch, which can
// belong to a graph's private allocator pool. Exact calls reuse a workspace and pinned sizes staging per stream.
//
//   encode:        everything on the stream, outputs sized by the capacities: no host synchronization, capturable.
//   encode_exact:  counts, copies the sizes to the host and waits (the one synchronization), has `allocate` make the
//                  outputs of exactly that size, and fills them.

#include "arrays.hpp"
#include "cuda_stream.hpp"
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
    std::optional<cuda::EncodingAcquisition> context;
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

    void append(std::intptr_t stream, const std::vector<std::uintptr_t>& source,
                const std::vector<std::uintptr_t>& ring, std::uintptr_t status)
    {
        const auto in = input_of(rows_of(source));
        const auto dst = input_of(rows_of(ring));
        nb::gil_scoped_release release;
        cuda::DeviceGuard guard(m_device);
        check_rows(rows_of(source));
        cuda::append_rows(in, dst, ptr<u64>(status), cuda_stream(stream));
    }

    void clear(std::intptr_t stream, std::uintptr_t status, u64 capacity)
    {
        nb::gil_scoped_release release;
        cuda::DeviceGuard guard(m_device);
        cuda::clear_rows(ptr<u64>(status), capacity, cuda_stream(stream));
    }

    void flush(std::intptr_t stream, const std::vector<std::uintptr_t>& ring, std::uintptr_t status,
               const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities,
               std::uintptr_t graphs, std::uintptr_t scratch, u64 bytes)
    {
        if (capacities.size() != 4) refuse("expected four capacities");
        auto in = input_of(rows_of(ring));
        in.graphs_device = ptr<const u64>(status) + 1;
        in.row_start = ptr<const u64>(status);
        in.dropped = ptr<const u64>(status) + 2;
        in.row_capacity = in.graphs;
        auto out = output_of(outputs, {capacities[0], capacities[1], capacities[2], capacities[3]});
        out.graphs = ptr<u64>(graphs);
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure_context(*slot, cuda_stream(stream));
        m_device_encoder->encode(*slot->context, ptr<void>(scratch), bytes, {in.graphs, 0, 0, 0}, in, out);
        cuda::finish_rows(ptr<u64>(status), in.graphs, cuda_stream(stream));
    }

    /// Makes the stream's workspace hold batches of up to `graphs` graphs. Not during a capture.
    void reserve(std::intptr_t stream, u64 graphs)
    {
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure(*slot, cuda_stream(stream), graphs);
    }

    /// Frees the exact workspace and acquisition of `stream`, or of every stream when absent.
    void release(std::optional<std::intptr_t> stream)
    {
        nb::gil_scoped_release release;
        std::map<cudaStream_t, std::shared_ptr<StreamSlot>> dropped;
        {
            std::lock_guard lock(m_mutex);
            if (stream)
            {
                const auto it = m_slots.find(cuda_stream(*stream));
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
            const auto it = m_slots.find(cuda_stream(stream));
            if (it != m_slots.end())
                slot = it->second;
        }
        if (!slot)
            return {0, 0, 0, 0};
        std::lock_guard lock(slot->mutex);
        return {slot->held ? slot->held->shape().graphs : 0, slot->held ? slot->held->scratch_bytes() : 0, slot->reservations,
                slot->syncs};
    }

    /// Scratch bytes for these rows and this stream's acquired generation. No workspace reservation.
    u64 scratch_bytes(std::intptr_t stream, u64 graphs, bool derived)
    {
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure_context(*slot, cuda_stream(stream));
        return m_device_encoder->scratch_bytes(*slot->context, cuda::BatchShape{graphs, 0, 0, 0},
                                               derived ? cuda::DerivedWords::Supplied : cuda::DerivedWords::Evaluated);
    }

    void check_stream_schema() const { cuda::check_stream_schema(m_encoder->schema()); }

    /// Encodes with caller-owned scratch. Its owner keeps it alive until the stream or captured graph is done.
    void encode(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities,
                std::uintptr_t scratch, u64 bytes)
    {
        if (capacities.size() != 4)
            throw nb::value_error("mymyr: expected four capacities");
        const Rows rows = rows_of(row_fields);
        const encoding::BatchShape caps{capacities[0], capacities[1], capacities[2], capacities[3]};
        cuda::DeviceOutput out = output_of(outputs, caps);
        out.zero_unused = true;
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure_context(*slot, cuda_stream(stream));
        check_rows(rows);
        m_device_encoder->encode(*slot->context, ptr<void>(scratch), bytes, cuda::BatchShape{rows.graphs, 0, 0, 0},
                                 input_of(rows), out);
    }

    /// Counts, waits for the sizes, and calls `allocate(entities, tuples, args, targets, flags)` to check the flags
    /// and return the graph extent and output pointers (the sizes pointer ignored), then fills them.
    void encode_exact(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                      const std::function<std::vector<std::uintptr_t>(u64, u64, u64, u64, u32)>& allocate)
    {
        const Rows rows = rows_of(row_fields);
        std::vector<std::uintptr_t> outputs;
        {
            nb::gil_scoped_release release;
            const auto slot = slot_of(stream);
            std::lock_guard lock(slot->mutex);
            const cudaStream_t s = cuda_stream(stream);
            if (capturing(m_device, s))
                refuse("an encode without capacities waits for the batch's sizes and cannot be captured in a CUDA graph: "
                       "pass capacities");
            ensure(*slot, s, rows.graphs);
            if (!slot->sizes)
            {
                slot->sizes = std::make_shared<cuda::DeviceBuffer>(m_device_encoder->context().context(),
                                                                 sizeof(encoding::FlatSizes), s);
                slot->staging = m_device_encoder->context().context()->lease_pinned(1, sizeof(encoding::FlatSizes));
            }
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
            outputs = allocate(sizes.entities, sizes.tuples, sizes.args, sizes.targets, sizes.overflow);
            if (outputs.size() != k_output_fields + 1)
                refuse("expected the output graph extent and fourteen pointers");
            const encoding::BatchShape caps{outputs.front(), sizes.entities, sizes.args, sizes.targets};
            outputs.erase(outputs.begin());
            cuda::DeviceOutput out = output_of(outputs, caps);
            out.sizes = device_sizes;
            m_device_encoder->fill(*slot->held, in, out);
        }
    }

private:
    std::shared_ptr<StreamSlot> slot_of(std::intptr_t stream)
    {
        std::lock_guard lock(m_mutex);
        auto& slot = m_slots[cuda_stream(stream)];
        if (!slot)
            slot = std::make_shared<StreamSlot>();
        return slot;
    }

    /// A context acquisition is independent of the scratch size and survives workspace growth.
    void ensure_context(StreamSlot& slot, cudaStream_t stream)
    {
        if (slot.context)
            return;
        if (capturing(m_device, stream))
            refuse("warm the encoder on the capture stream before capture to acquire its uploaded context");
        slot.context.emplace(m_device_encoder->context().acquire(stream));
    }

    /// The reusable exact workspace holds at least `graphs` rows; exact calls cannot be captured.
    void ensure(StreamSlot& slot, cudaStream_t stream, u64 graphs)
    {
        if (slot.held && slot.held->shape().graphs >= graphs)
            return;
        if (capturing(m_device, stream))
            refuse("cannot reserve an exact workspace during CUDA graph capture");
        // sized to a power of two so that batches that grow a little reserve again rarely
        const u64 want = std::bit_ceil(std::max<u64>(graphs, 1024));
        cuda::DeviceGuard guard(m_device);
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

class DeviceCollatorHandle
{
public:
    DeviceCollatorHandle(const std::vector<nb::handle>& encoders, int device, std::intptr_t stream,
                         const std::vector<std::vector<std::uintptr_t>>& inputs, const std::vector<u64>& graphs)
        : m_device(device), m_stream(cuda_stream(stream))
    {
        if (inputs.size() != graphs.size() || inputs.size() != encoders.size() || encoders.empty())
            refuse("collation needs one encoder and one graph count per batch");
        m_context = table_device_context(nb::getattr(encoders.front(), "table"), nb::none(), device);
        std::vector<cuda::DeviceCollationBatch> batches;
        for (usize i = 0; i < inputs.size(); ++i)
        {
            const auto output = output_of(inputs[i], {});
            batches.push_back(cuda::collation_batch(*flat_encoder_of(encoders[i]), output.arrays, output.sizes, graphs[i]));
        }
        nb::gil_scoped_release release;
        m_collator = std::make_unique<cuda::DeviceCollator>(m_context, std::move(batches), m_stream);
    }
    u64 scratch_bytes() const { return m_collator->scratch_bytes(); }
    void encode(const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities,
                std::uintptr_t scratch, u64 bytes)
    {
        if (capacities.size() != 4) refuse("expected four capacities");
        auto out = output_of(outputs, {capacities[0], capacities[1], capacities[2], capacities[3]});
        out.zero_unused = true;
        nb::gil_scoped_release release;
        m_collator->encode(ptr<void>(scratch), bytes, out);
    }
    void encode_exact(std::uintptr_t scratch, u64 bytes,
                       const std::function<std::vector<std::uintptr_t>(u64, u64, u64, u64, u32)>& allocate)
    {
        nb::gil_scoped_release release;
        std::lock_guard lock(m_mutex);
        cuda::DeviceGuard guard(m_device);
        if (capturing(m_device, m_stream)) refuse("exact collation cannot be captured: pass capacities");
        if (!m_sizes)
        {
            m_sizes = std::make_unique<cuda::DeviceBuffer>(m_context, sizeof(encoding::FlatSizes), m_stream);
            m_staging = m_context->lease_pinned(1, sizeof(encoding::FlatSizes));
        }
        auto* sizes = static_cast<encoding::FlatSizes*>(m_sizes->data());
        m_collator->count(ptr<void>(scratch), bytes, sizes);
        cuda::check(cudaMemcpyAsync(m_staging.data(0), sizes, sizeof(*sizes), cudaMemcpyDeviceToHost, m_stream), "copy collated sizes");
        cuda::check(cudaStreamSynchronize(m_stream), "wait for collated sizes");
        const auto counted = *static_cast<const encoding::FlatSizes*>(m_staging.data(0));
        const auto outputs = allocate(counted.entities, counted.tuples, counted.args, counted.targets, counted.overflow);
        auto out = output_of(outputs, {m_collator->graphs(), counted.entities, counted.args, counted.targets});
        out.sizes = sizes;
        m_collator->fill(ptr<void>(scratch), bytes, out);
    }
private:
    int m_device;
    cudaStream_t m_stream;
    cuda::ContextPtr m_context;
    std::unique_ptr<cuda::DeviceCollator> m_collator;
    std::unique_ptr<cuda::DeviceBuffer> m_sizes;
    cuda::PinnedLease m_staging;
    std::mutex m_mutex;
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
        .def("append", &DeviceEncoderHandle::append, "stream"_a, "source"_a, "ring"_a, "status"_a,
             "Copies source rows into the device ring, then advances its tail.")
        .def("clear", &DeviceEncoderHandle::clear, "stream"_a, "status"_a, "capacity"_a,
             "Empties the ring and clears its overflow flag on the stream.")
        .def("flush", &DeviceEncoderHandle::flush, "stream"_a, "ring"_a, "status"_a, "outputs"_a,
             "capacities"_a, "graphs"_a, "scratch"_a, "bytes"_a,
             "Encodes the live ring and empties it on the stream.")
        .def("reserve", &DeviceEncoderHandle::reserve, "stream"_a, "graphs"_a,
             "Makes the workspace of `stream` (a cudaStream_t, 0: the default stream) hold at least `graphs` graphs. "
             "Not during a capture.")
        .def("release", &DeviceEncoderHandle::release, "stream"_a = nb::none(),
             "Frees the workspace of `stream`, or of every stream.")
        .def("state", &DeviceEncoderHandle::state, "stream"_a,
             "[graphs reserved, scratch bytes, reservations made, host waits made] of a stream.")
        .def("scratch_bytes", &DeviceEncoderHandle::scratch_bytes, "stream"_a, "graphs"_a, "derived"_a = false,
             "Scratch bytes for this stream's acquired context and graph count; `derived`: for rows whose derived "
             "words are given.")
        .def("check_stream_schema", &DeviceEncoderHandle::check_stream_schema,
             "Raises ValueError for a schema that a device stream cannot carry (subgoal layers, history steps).")
        .def("encode", &DeviceEncoderHandle::encode, "stream"_a, "rows"_a, "outputs"_a, "capacities"_a,
             "scratch"_a, "bytes"_a,
             "Encodes the rows (15 integers: the graphs, then the pointers and strides of the arrays) into the 14 output "
             "pointers, with the capacities (graphs, entities, args, targets), on `stream`. No host wait.")
        .def("encode_exact", &DeviceEncoderHandle::encode_exact, "stream"_a, "rows"_a, "allocate"_a,
             "Counts the rows, waits once for their sizes, calls allocate(entities, tuples, args, targets, flags) "
             "to check the flags and return the graph extent and fourteen pointers, then fills them on `stream`.");

    nb::class_<DeviceCollatorHandle>(module, "DeviceCollator", "An uploaded list of device batch views.")
        .def("__init__", [](DeviceCollatorHandle* self, const std::vector<nb::handle>& encoders, int device, std::intptr_t stream,
                            const std::vector<std::vector<std::uintptr_t>>& inputs, const std::vector<u64>& graphs)
             { new (self) DeviceCollatorHandle(encoders, device, stream, inputs, graphs); },
             "encoders"_a, "device"_a, "stream"_a, "inputs"_a, "graphs"_a)
        .def_prop_ro("scratch_bytes", &DeviceCollatorHandle::scratch_bytes)
        .def("encode", &DeviceCollatorHandle::encode, "outputs"_a, "capacities"_a, "scratch"_a, "bytes"_a)
        .def("encode_exact", &DeviceCollatorHandle::encode_exact, "scratch"_a, "bytes"_a, "allocate"_a);
}
}  // namespace mymyr::python
