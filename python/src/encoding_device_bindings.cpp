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
#include <array>
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
    // the subgoal layers, then the masks of the history: pos, neg, stride, layers, words, derived pos, derived neg,
    // derived stride, derived words; then the history's dt and count
    std::array<std::uintptr_t, 9> subgoals{}, history{};
    std::uintptr_t history_dt = 0, history_count = 0;
    // the expansion rows: offsets, schema, binding, parent, capacity, label width, status
    std::array<std::uintptr_t, 7> actions{};
    // the horizon view: node offsets, edge offsets, src, dst, schema, binding, nodes, edges, label width, scratch,
    // scratch words, status (mymyr.encoding._device._ROW_HORIZON and after)
    std::array<std::uintptr_t, 12> horizon{};
    static constexpr usize k_horizon_nodes = 6;
    // the rows the scratch covers: the graphs, or every node of a horizon view
    [[nodiscard]] u64 input_rows() const noexcept
    {
        return horizon[0] ? std::max<u64>(graphs, horizon[k_horizon_nodes]) : graphs;
    }
};
constexpr usize k_row_fields = 54;

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
    for (auto* layers : {&r.subgoals, &r.history})
        for (std::uintptr_t& field : *layers)
            field = v[i++];
    r.history_dt = v[i++];
    r.history_count = v[i++];
    for (auto& field : r.actions)
        field = v[i++];
    for (auto& field : r.horizon)
        field = v[i++];
    return r;
}

template<class T>
T* ptr(std::uintptr_t p)
{
    return reinterpret_cast<T*>(p);
}

encoding::LayerInput layers_of(const std::array<std::uintptr_t, 9>& f)
{
    encoding::LayerInput l;
    l.pos = ptr<const u64>(f[0]);
    l.neg = ptr<const u64>(f[1]);
    l.stride = f[2];
    l.layers = static_cast<u32>(f[3]);
    l.words = static_cast<u32>(f[4]);
    l.derived_pos = ptr<const u64>(f[5]);
    l.derived_neg = ptr<const u64>(f[6]);
    l.derived_stride = f[7];
    l.derived_words = static_cast<u32>(f[8]);
    return l;
}

cuda::DeviceEncodeInput input_of(const Rows& r, u32 numeric_words = 0)
{
    cuda::DeviceEncodeInput in;
    in.graphs = r.graphs;
    in.states = ptr<const u64>(r.states);
    in.state_stride = r.state_stride;
    if (r.state_words < numeric_words) refuse("the state row is narrower than its numeric words");
    in.state_words = r.state_words - numeric_words;
    in.numeric_words = numeric_words;
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
    in.subgoals = layers_of(r.subgoals);
    in.history.masks = layers_of(r.history);
    in.history.dt = ptr<const i32>(r.history_dt);
    in.history.count = ptr<const i32>(r.history_count);
    const auto& a = r.actions;
    in.actions = {ptr<const i32>(a[0]), ptr<const i32>(a[1]), ptr<const i32>(a[2]), ptr<const i32>(a[3]),
                  a[4], static_cast<u32>(a[5]), ptr<const u64>(a[6])};
    const auto& h = r.horizon;
    in.horizon = {ptr<const i32>(h[0]), ptr<const i32>(h[1]), ptr<const i32>(h[2]), ptr<const i32>(h[3]),
                  ptr<const i32>(h[4]), ptr<const i32>(h[5]), h[6], h[7], static_cast<u32>(h[8]),
                  ptr<u32>(h[9]), h[10], ptr<const u32>(h[11])};
    return in;
}

/// The destination arrays, in the order of encoding._device: task_ids, num_objects, entity_offsets, entity_role,
/// entity_ref, relation_counts, relation_offsets, relation_arg_offsets, relation_args, goal_complete, target_offsets,
/// target_entity, target_group, history_offsets, history_entity, history_dt, the five Simple arrays (node_offsets,
/// node_label, edge_offsets, edge_index, edge_label), numeric_values, the five static arrays (static_task_ids,
/// static_offsets, static_arg_offsets, static_args, static_relations), static_sizes, sizes, then the three private
/// relation arrays of a Simple-only fill and the edge_index row stride (0: the array is absent or has no entries). With
/// numeric values the sizes are followed by the numeric value count.
constexpr usize k_output_fields = 33;
/// The counts an exact encoding or collation reads back in one transfer.
struct CountedSizes
{
    encoding::FlatSizes flat;
    u64 numeric = 0;
    encoding::StaticSizes statics;
};
/// allocate(entities, tuples, args, targets, history, numeric, flags, static_instances, static_tuples, static_args)
using ExactAllocation = std::function<std::vector<std::uintptr_t>(u64, u64, u64, u64, u64, u64, u32, u64, u64, u64)>;
/// allocate(entities, tuples, args, targets, numeric, flags, static_instances, static_tuples, static_args)
using CollationAllocation = std::function<std::vector<std::uintptr_t>(u64, u64, u64, u64, u64, u32, u64, u64, u64)>;

cuda::DeviceOutput output_of(const std::vector<std::uintptr_t>& v, const encoding::BatchShape& capacities,
                              bool numeric = false, const encoding::StaticShape& statics = {})
{
    if (v.size() != k_output_fields)
        throw nb::value_error(("mymyr: expected " + std::to_string(k_output_fields) + " output fields").c_str());
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
    a.history_offsets = ptr<void>(v[13]);
    a.history_entity = ptr<void>(v[14]);
    a.history_dt = ptr<i32>(v[15]);
    a.node_offsets = ptr<void>(v[16]);
    a.node_label = ptr<i32>(v[17]);
    a.edge_offsets = ptr<void>(v[18]);
    a.edge_index = ptr<void>(v[19]);
    a.edge_label = ptr<i32>(v[20]);
    a.numeric_values = ptr<void>(v[21]);
    a.static_task_ids = ptr<i32>(v[22]);
    a.static_offsets = ptr<void>(v[23]);
    a.static_arg_offsets = ptr<void>(v[24]);
    a.static_args = ptr<void>(v[25]);
    a.static_relations = ptr<void>(v[26]);
    out.static_sizes = ptr<encoding::StaticSizes>(v[27]);
    out.static_capacities = statics;
    out.sizes = ptr<encoding::FlatSizes>(v[28]);
    if (numeric && out.sizes) out.numeric_size = reinterpret_cast<u64*>(out.sizes + 1);
    out.scratch_relation_offsets = ptr<void>(v[29]);
    out.scratch_relation_arg_offsets = ptr<void>(v[30]);
    out.scratch_relation_args = ptr<void>(v[31]);
    a.edge_stride = v[32];
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

    void build_space_horizons(std::intptr_t stream, const std::vector<std::uintptr_t>& inputs,
                              const std::vector<std::uintptr_t>& outputs, u32 states, u64 edges, u32 words,
                              u32 label_width, u32 graphs, u32 depth, u32 limit, bool truncate, u64 scratch_bytes)
    {
        if (inputs.size() != 6 || outputs.size() != 9)
            refuse("the state-space horizon builder needs six input arrays and nine output arrays");
        const cuda::DeviceHorizonSpace space{ptr<const u64>(inputs[0]), ptr<const u64>(inputs[1]),
            ptr<const u32>(inputs[2]), ptr<const u32>(inputs[3]), ptr<const u32>(inputs[4]), states, words, label_width, edges};
        const cuda::DeviceSpaceHorizonBuild out{ptr<const i32>(inputs[5]), graphs, depth, limit, truncate,
            ptr<u64>(outputs[0]), ptr<i32>(outputs[1]), ptr<i32>(outputs[2]), ptr<i32>(outputs[3]),
            ptr<i32>(outputs[4]), ptr<i32>(outputs[5]), ptr<i32>(outputs[6]), ptr<void>(outputs[7]), scratch_bytes, ptr<u8>(outputs[8])};
        nb::gil_scoped_release release;
        cuda::DeviceGuard guard(m_device);
        cuda::build_space_horizons(space, out, cuda_stream(stream));
    }

    void build_horizon(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                       std::uintptr_t children, u64 child_stride, const std::vector<std::uintptr_t>& outputs,
                       bool merge, std::uintptr_t scratch, u64 scratch_bytes)
    {
        if (outputs.size() != 6)
            refuse("the horizon builder needs nodes, node offsets, sources, destinations, schemas and bindings");
        const auto in = input_of(rows_of(row_fields));
        const cuda::DeviceHorizonBuild out{ptr<u64>(outputs[0]), ptr<i32>(outputs[1]), ptr<i32>(outputs[2]),
                                            ptr<i32>(outputs[3]), merge,              ptr<i32>(outputs[4]),
                                            ptr<i32>(outputs[5]), ptr<void>(scratch), scratch_bytes};
        nb::gil_scoped_release release;
        cuda::DeviceGuard guard(m_device);
        cuda::build_depth_one_horizon(in, ptr<const u64>(children), child_stride, out, cuda_stream(stream));
    }

    void append(std::intptr_t stream, const std::vector<std::uintptr_t>& source,
                const std::vector<std::uintptr_t>& ring, std::uintptr_t status)
    {
        const auto in = input_of(rows_of(source), m_encoder->table().numeric_words());
        const auto dst = input_of(rows_of(ring), m_encoder->table().numeric_words());
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
               std::uintptr_t graphs, std::uintptr_t scratch, u64 bytes, bool zero_unused)
    {
        if (capacities.size() != 8) refuse("expected eight capacities");
        auto in = input_of(rows_of(ring), m_encoder->table().numeric_words());
        in.graphs_device = ptr<const u64>(status) + 1;
        in.row_start = ptr<const u64>(status);
        in.dropped = ptr<const u64>(status) + 2;
        in.row_capacity = in.graphs;
        auto out = output_of(outputs, {capacities[0], capacities[1], capacities[2], capacities[3], capacities[4], capacities[5]},
                             numeric(), {capacities[6], capacities[7]});
        out.graphs = ptr<u64>(graphs);
        out.zero_unused = zero_unused;
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
        return m_device_encoder->scratch_bytes(*slot->context, cuda::BatchShape{graphs, 0, 0, 0, 0},
                                               derived ? cuda::DerivedWords::Supplied : cuda::DerivedWords::Evaluated);
    }

    /// Encodes with caller-owned scratch. Its owner keeps it alive until the stream or captured graph is done.
    void encode(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities,
                std::uintptr_t scratch, u64 bytes, std::uintptr_t graph_count)
    {
        if (capacities.size() != 8)
            throw nb::value_error("mymyr: expected eight capacities");
        const Rows rows = rows_of(row_fields);
        const encoding::BatchShape caps{capacities[0], capacities[1], capacities[2], capacities[3], capacities[4], capacities[5]};
        cuda::DeviceOutput out = output_of(outputs, caps, numeric(), {capacities[6], capacities[7]});
        out.zero_unused = true;
        out.graphs = ptr<u64>(graph_count);
        auto in = input_of(rows, m_encoder->table().numeric_words());
        in.graphs_device = ptr<const u64>(graph_count);
        nb::gil_scoped_release release;
        const auto slot = slot_of(stream);
        std::lock_guard lock(slot->mutex);
        ensure_context(*slot, cuda_stream(stream));
        check_rows(rows);
        m_device_encoder->encode(*slot->context, ptr<void>(scratch), bytes,
                                 cuda::BatchShape{rows.input_rows(), 0, 0, 0, 0, 0},
                                 in, out);
    }

    /// Counts, waits once for the dynamic, numeric and static sizes, and calls `allocate(entities, tuples, args,
    /// targets, history, numeric, flags, static_instances, static_tuples, static_args)` to check the flags and return the
    /// graph extent and output pointers (the sizes pointer ignored), then fills them.
    void encode_exact(std::intptr_t stream, const std::vector<std::uintptr_t>& row_fields,
                      const ExactAllocation& allocate)
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
            ensure(*slot, s, rows.input_rows());
            if (!slot->sizes)
            {
                slot->sizes = std::make_shared<cuda::DeviceBuffer>(m_device_encoder->context().context(),
                                                                 sizeof(CountedSizes), s);
                slot->staging = m_device_encoder->context().context()->lease_pinned(1, sizeof(CountedSizes));
            }
            check_rows(rows);
            const cuda::DeviceEncodeInput in = input_of(rows, m_encoder->table().numeric_words());
            auto* device_sizes = static_cast<encoding::FlatSizes*>(slot->sizes->data());
            cuda::DeviceOutput counted;
            counted.capacities.graphs = rows.graphs;
            counted.sizes = device_sizes;
            counted.numeric_size = reinterpret_cast<u64*>(device_sizes + 1);
            if (m_device_encoder->schema().shared_statics())
                counted.static_sizes = &static_cast<CountedSizes*>(slot->sizes->data())->statics;
            m_device_encoder->count(*slot->held, in, counted);
            cuda::DeviceGuard guard(m_device);
            cuda::check(cudaMemcpyAsync(slot->staging.data(0), device_sizes, sizeof(CountedSizes), cudaMemcpyDeviceToHost, s),
                        "cudaMemcpyAsync (sizes)");
            cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize (sizes)");
            ++slot->syncs;
            const encoding::FlatSizes sizes = *static_cast<const encoding::FlatSizes*>(slot->staging.data(0));
            const auto& counts = *static_cast<const CountedSizes*>(slot->staging.data(0));
            const u64 n = counts.numeric;
            const auto statics = m_device_encoder->schema().shared_statics() ? counts.statics : encoding::StaticSizes{};
            outputs = allocate(sizes.entities, sizes.tuples, sizes.args, sizes.targets, sizes.history, n, sizes.overflow,
                               statics.instances, statics.tuples, statics.args);
            if (outputs.size() != k_output_fields + 1)
                refuse("expected the output graph extent and " + std::to_string(k_output_fields) + " output fields");
            const encoding::BatchShape caps{outputs.front(), sizes.entities, sizes.args, sizes.targets, sizes.history, n};
            outputs.erase(outputs.begin());
            cuda::DeviceOutput out = output_of(outputs, caps, numeric(), {statics.instances, statics.args});
            out.sizes = device_sizes;
            m_device_encoder->fill(*slot->held, in, out);
        }
    }

private:
    bool numeric() const { return m_encoder->schema().stores_numeric_values(); }
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
        m_device_encoder->reserve(cuda::BatchShape{want, 0, 0, 0, 0}, stream);
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
                         const std::vector<std::vector<std::uintptr_t>>& inputs, const std::vector<std::vector<u64>>& shapes)
        : m_device(device), m_stream(cuda_stream(stream))
    {
        if (inputs.size() != shapes.size() || inputs.size() != encoders.size() || encoders.empty())
            refuse("collation needs one encoder and one graph count per batch");
        m_context = table_device_context(nb::getattr(encoders.front(), "table"), nb::none(), device);
        m_numeric = flat_encoder_of(encoders.front())->schema().stores_numeric_values();
        std::vector<cuda::DeviceCollationBatch> batches;
        for (usize i = 0; i < inputs.size(); ++i)
        {
            const auto output = output_of(inputs[i], {}, m_numeric);
            if (shapes[i].size() != 3) refuse("collation needs graph and static capacities for every batch");
            auto batch = cuda::collation_batch(*flat_encoder_of(encoders[i]), output.arrays, output.sizes, shapes[i][0]);
            batch.static_capacities = {shapes[i][1], shapes[i][2]};
            batch.static_sizes = output.static_sizes;
            batches.push_back(batch);
        }
        m_shared = batches.front().shared_statics;
        nb::gil_scoped_release release;
        m_collator = std::make_unique<cuda::DeviceCollator>(m_context, std::move(batches), m_stream);
    }
    u64 scratch_bytes() const { return m_collator->scratch_bytes(); }
    void encode(const std::vector<std::uintptr_t>& outputs, const std::vector<u64>& capacities,
                std::uintptr_t scratch, u64 bytes)
    {
        if (capacities.size() != 8) refuse("expected eight capacities");
        auto out = output_of(outputs, {capacities[0], capacities[1], capacities[2], capacities[3], capacities[4], capacities[5]},
                             numeric(), {capacities[6], capacities[7]});
        out.zero_unused = true;
        nb::gil_scoped_release release;
        m_collator->encode(ptr<void>(scratch), bytes, out);
    }
    void encode_exact(std::uintptr_t scratch, u64 bytes,
                       const CollationAllocation& allocate)
    {
        nb::gil_scoped_release release;
        std::lock_guard lock(m_mutex);
        cuda::DeviceGuard guard(m_device);
        if (capturing(m_device, m_stream)) refuse("exact collation cannot be captured: pass capacities");
        if (!m_sizes)
        {
            m_sizes = std::make_unique<cuda::DeviceBuffer>(m_context, sizeof(CountedSizes), m_stream);
            m_staging = m_context->lease_pinned(1, sizeof(CountedSizes));
        }
        auto* sizes = static_cast<encoding::FlatSizes*>(m_sizes->data());
        m_collator->count(ptr<void>(scratch), bytes, sizes, reinterpret_cast<u64*>(sizes + 1),
                          m_shared ? &static_cast<CountedSizes*>(m_sizes->data())->statics : nullptr);
        cuda::check(cudaMemcpyAsync(m_staging.data(0), sizes, sizeof(CountedSizes), cudaMemcpyDeviceToHost, m_stream), "copy collated sizes");
        cuda::check(cudaStreamSynchronize(m_stream), "wait for collated sizes");
        const auto& counts = *static_cast<const CountedSizes*>(m_staging.data(0));
        const auto counted = counts.flat;
        const u64 n = counts.numeric;
        const auto statics = m_shared ? counts.statics : encoding::StaticSizes{};
        const auto outputs = allocate(counted.entities, counted.tuples, counted.args, counted.targets, n, counted.overflow,
                                      statics.instances, statics.tuples, statics.args);
        auto out = output_of(outputs, {m_collator->graphs(), counted.entities, counted.args, counted.targets, 0, n}, m_numeric,
                             {statics.instances, statics.args});
        out.sizes = sizes;
        m_collator->fill(ptr<void>(scratch), bytes, out);
    }
private:
    bool numeric() const { return m_numeric; }
    bool m_numeric = false;
    int m_device;
    bool m_shared = false;
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
        .def("build_space_horizons", &DeviceEncoderHandle::build_space_horizons, "stream"_a, "inputs"_a,
             "outputs"_a, "states"_a, "edges"_a, "words"_a, "label_width"_a, "graphs"_a, "depth"_a,
             "limit"_a, "truncate"_a, "scratch_bytes"_a, "Traverses recorded state ids into compact device horizon slices on the stream.")
        .def_static("space_horizon_scratch_bytes", &cuda::space_horizon_scratch_bytes, "states"_a, "edges"_a, "graphs"_a,
                    "The scratch bytes for independent traversals of recorded state ids.")
        .def("build_horizon", &DeviceEncoderHandle::build_horizon, "stream"_a, "rows"_a, "children"_a,
             "child_stride"_a, "outputs"_a, "merge"_a, "scratch"_a, "scratch_bytes"_a,
             "Builds compact depth-one node slices from a device expansion without a host wait.")
        .def_static("horizon_scratch_bytes", &cuda::depth_one_horizon_scratch_bytes, "capacity"_a, "merge"_a,
                    "The scratch bytes of build_horizon over an expansion of `capacity` rows.")
        .def("append", &DeviceEncoderHandle::append, "stream"_a, "source"_a, "ring"_a, "status"_a,
             "Copies source rows into the device ring, then advances its tail.")
        .def("clear", &DeviceEncoderHandle::clear, "stream"_a, "status"_a, "capacity"_a,
             "Empties the ring and clears its overflow flag on the stream.")
        .def("flush", &DeviceEncoderHandle::flush, "stream"_a, "ring"_a, "status"_a, "outputs"_a,
             "capacities"_a, "graphs"_a, "scratch"_a, "bytes"_a, "zero_unused"_a = false,
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
        .def_static("check_stream_schema",
                    [](Arg<ann::Any> encoder) { cuda::check_stream_schema(flat_encoder_of(encoder)->schema()); }, "encoder"_a,
                    "Raises ValueError for an encoder whose schema a device stream cannot carry (subgoal layers, "
                    "history steps).")
        .def("encode", &DeviceEncoderHandle::encode, "stream"_a, "rows"_a, "outputs"_a, "capacities"_a,
             "scratch"_a, "bytes"_a, "graph_count"_a = 0,
             "Encodes the rows (54 integers: the graphs, then the pointers and strides of the arrays) into the 29 output "
             "arrays and their private storage, with the capacities (graphs, entities, args, targets, history, numeric, "
             "static_instances, static_args), on `stream`. Optional graph_count is the pointer to an int64 device "
             "scalar counting the live prefix, bounded by the row count; the clamped count is written back to that "
             "scalar. Negative int64 counts are read as unsigned and clamp to the row count. Per-graph tails are "
             "zeroed. No host wait.")
        .def("encode_exact", &DeviceEncoderHandle::encode_exact, "stream"_a, "rows"_a, "allocate"_a,
             "Counts the rows, waits once for their sizes, calls allocate(entities, tuples, args, targets, history, numeric, "
             "flags, static_instances, static_tuples, static_args) to check the flags and return the graph extent and "
             "the output fields, then fills them on `stream`.");

    nb::class_<DeviceCollatorHandle>(module, "DeviceCollator", "An uploaded list of device batch views.")
        .def("__init__", [](DeviceCollatorHandle* self, const std::vector<nb::handle>& encoders, int device, std::intptr_t stream,
                            const std::vector<std::vector<std::uintptr_t>>& inputs, const std::vector<std::vector<u64>>& shapes)
             { new (self) DeviceCollatorHandle(encoders, device, stream, inputs, shapes); },
             "encoders"_a, "device"_a, "stream"_a, "inputs"_a, "shapes"_a)
        .def_prop_ro("scratch_bytes", &DeviceCollatorHandle::scratch_bytes)
        .def("encode", &DeviceCollatorHandle::encode, "outputs"_a, "capacities"_a, "scratch"_a, "bytes"_a)
        .def("encode_exact", &DeviceCollatorHandle::encode_exact, "scratch"_a, "bytes"_a, "allocate"_a);
}
}  // namespace mymyr::python
