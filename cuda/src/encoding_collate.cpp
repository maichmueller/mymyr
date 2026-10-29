#include "capture_scope.hpp"
#include "encoding_collate.hpp"
#include "mymyr/cuda/encoding.hpp"


#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(std::string("mymyr: encoding: ") + message); }
u64 add(u64 a, u64 b)
{
    if (b > std::numeric_limits<u64>::max() - a)
        throw std::overflow_error("mymyr: encoding: device collation size overflow");
    return a + b;
}
u64 mul(u64 a, u64 b)
{
    if (b && a > std::numeric_limits<u64>::max() / b)
        throw std::overflow_error("mymyr: encoding: device collation size overflow");
    return a * b;
}
u64 aligned(u64 bytes) { return add(bytes, 255) & ~u64{255}; }
/// Resident blocks of the grid-stride count and fill per multiprocessor.
constexpr u32 k_blocks_per_sm = 8;
}
struct DeviceCollator::Impl
{
    ContextPtr context;
    cudaStream_t stream;
    detail::CollationParams params;
    OutputArrays stored;
    DeviceBuffer slices;
    PinnedLease staging;
    Event uploaded;
    bool has_upload = false;
    u64 total_bytes = 0, cell_offset = 0, bad_offset = 0, temp_offset = 0;

    Impl(ContextPtr ctx, std::vector<DeviceCollationBatch> batches, cudaStream_t s)
        : context(std::move(ctx)), stream(s ? s : context->stream())
    {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "query collation capture");
        if (capture != cudaStreamCaptureStatusNone) invalid("prepare the device collator outside capture");
        if (batches.empty()) invalid("collate needs at least one batch");
        const auto& first = batches.front();
        int sms = 0;
        check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, context->device()),
              "query the multiprocessors");
        params.grid = static_cast<u32>(std::max(sms, 1)) * k_blocks_per_sm;
        stored = first.stored;
        params.relations = first.relations;
        params.relations_stored = stored.relations;
        params.simple = stored.simple;
        params.index64 = first.index == encoding::IndexType::I64;
        params.relation_major = first.layout == encoding::Layout::Relation;
        params.batches = batches.size();
        params.numeric_relations = first.numeric_relations;
        params.numeric_f64 = stored.numeric_dtype == encoding::NumericDtype::F64;
        params.shared_statics = first.shared_statics;
        params.static_relations = first.static_relations;
        params.instances = first.instances;
        staging = context->lease_pinned(1, mul(batches.size(), sizeof(detail::CollationSlice)));
        auto* host = static_cast<detail::CollationSlice*>(staging.data(0));
        for (u64 i = 0; i < batches.size(); ++i)
        {
            const auto& b = batches[i];
            if (b.layout != first.layout) invalid("collate received mixed layouts");
            if (b.index != first.index) invalid("collate received mixed index types");
            if (b.shared_statics != first.shared_statics) invalid("collate received mixed statics");
            if (b.schema_fingerprint != first.schema_fingerprint) invalid("collate received mixed schemas");
            if (b.table_fingerprint != first.table_fingerprint)
                invalid("collate received batches of different tables: their task ids name different instances");
            if (b.stored.numeric != (b.numeric_relations != 0))
                invalid("collate received inconsistent numeric metadata");
            if (b.numeric_relations != first.numeric_relations || b.numeric_relations > b.relations)
                invalid("collate received different numeric_relations");
            if (b.static_relations != first.static_relations || b.instances != first.instances)
                invalid("collate received different static catalogs");
            if (b.shared_statics && (!b.static_sizes || !b.arrays.static_offsets || !b.arrays.static_arg_offsets ||
                (b.static_capacities.instances && !b.arrays.static_task_ids) ||
                (b.static_capacities.args && !b.arrays.static_args) ||
                (b.static_relations && !b.arrays.static_relations)))
                invalid("shared collation needs the source static blocks and their sizes");
            if (b.max_goal_level != 0 || b.history || b.arrays.history_offsets || b.arrays.history_entity ||
                b.arrays.history_dt)
                invalid("device collation carries no subgoal layers or history steps");
            if (b.relations != first.relations || b.stored != first.stored) invalid("collate received different stored arrays");
            host[i] = {b, params.graphs};
            params.graphs = add(params.graphs, b.graphs);
        }
        params.cells = stored.relations ? mul(params.graphs, params.relations) : 0;
        (void)add(params.cells, 1);
        cell_offset = aligned(mul(add(params.graphs, 1), sizeof(detail::GraphCounts)));
        bad_offset = add(cell_offset, aligned(mul(add(params.cells, 1), sizeof(detail::CellCounts))));
        temp_offset = add(bad_offset, aligned(sizeof(u64)));
        params.temp_bytes = detail::collation_scan_bytes(params.graphs, params.cells);
        total_bytes = add(temp_offset, params.temp_bytes);
        slices = DeviceBuffer(context, mul(batches.size(), sizeof(detail::CollationSlice)), stream);
        params.slices = static_cast<const detail::CollationSlice*>(slices.data());
        try
        {
            check(cudaMemcpyAsync(slices.data(), host, slices.size(), cudaMemcpyHostToDevice, stream),
                  "upload collation slices");
            uploaded.record(stream);
            has_upload = true;
        }
        catch (...)
        {
            // Do not recycle pinned staging while a descriptor copy is outstanding.
            (void)cudaStreamSynchronize(stream);
            throw;
        }
    }

    ~Impl()
    {
        detail::RelaxedCaptureScope capture_mode;
        // Staging may be recycled only after the descriptor copy has finished.
        if (has_upload)
            try { uploaded.synchronize(); } catch (...) { (void)cudaStreamSynchronize(stream); }
    }

    void run(int phase, void* scratch, u64 bytes, const DeviceOutput& output) const
    {
        DeviceGuard guard(context->device());
        if (!scratch || bytes < total_bytes || reinterpret_cast<std::uintptr_t>(scratch) % 256)
            invalid("device collation scratch is null, too small, or not aligned to 256 bytes");
        if (!output.sizes) invalid("device collation needs output sizes");
        if (params.shared_statics && !output.static_sizes)
            invalid("shared collation needs output static sizes");
        const auto& static_arrays = output.arrays;
        if (!params.shared_statics &&
            (output.static_sizes || output.static_capacities.instances || output.static_capacities.args ||
             static_arrays.static_task_ids || static_arrays.static_offsets || static_arrays.static_arg_offsets ||
             static_arrays.static_args || static_arrays.static_relations))
            invalid("repeated statics take no static block outputs");
        if (phase != 1)
        {
            const auto& a = output.arrays;
            if (output.capacities.graphs < params.graphs) invalid("collated graphs exceed the output capacity");
            if (!a.entity_offsets || (stored.relations && (!a.relation_offsets || !a.relation_arg_offsets)) ||
                (params.graphs && (!a.task_ids || !a.num_objects)))
                invalid("device collation needs the batch output arrays");
            if (stored.entity_arrays ? (output.capacities.entities && (!a.entity_role || !a.entity_ref))
                                     : (a.entity_role || a.entity_ref))
                invalid("collation entity arrays do not match the schema");
            if (stored.counts ? (params.graphs && params.relations && !a.relation_counts) : bool(a.relation_counts))
                invalid("collation counts do not match the schema");
            if (stored.goal_complete ? (params.graphs && !a.goal_complete) : bool(a.goal_complete))
                invalid("collation goal flags do not match the schema");
            if (bool(a.target_offsets) != stored.targets ||
                (stored.targets ? (output.capacities.targets && (!a.target_entity || !a.target_group))
                                : (a.target_entity || a.target_group)))
                invalid("collation target arrays do not match the schema");
            if (stored.relations && output.capacities.args && !a.relation_args) invalid("collation arguments are null");
            if (stored.simple && (!a.node_offsets || !a.edge_offsets ||
                ((output.capacities.entities || output.capacities.args) && !a.node_label) ||
                (output.capacities.args && (!a.edge_index || !a.edge_label)) ||
                a.edge_stride < mul(2, output.capacities.args)))
                invalid("collation Simple arrays do not hold the capacities");
            if (stored.numeric ? (output.capacities.numeric && !a.numeric_values) : a.numeric_values != nullptr)
                invalid("collation numeric values do not match the schema");
            if (params.shared_statics && (!a.static_offsets || !a.static_arg_offsets ||
                (output.static_capacities.instances && !a.static_task_ids) ||
                (output.static_capacities.args && !a.static_args) ||
                (params.static_relations && !a.static_relations)))
                invalid("shared collation needs the static output arrays");
            if (a.history_offsets || a.history_entity || a.history_dt)
                invalid("device collation carries no subgoal layers or history steps");
        }
        auto p = params;
        auto* base = static_cast<std::byte*>(scratch);
        p.graph_counts = reinterpret_cast<detail::GraphCounts*>(base);
        p.cell_counts = reinterpret_cast<detail::CellCounts*>(base + cell_offset);
        p.bad = reinterpret_cast<u64*>(base + bad_offset);
        p.static_totals = reinterpret_cast<encoding::StaticSizes*>(base + bad_offset + sizeof(u64));
        p.temp = base + temp_offset;
        p.output = output;
        p.node_capacity = encoding::simple_node_capacity(output.capacities, stored.argumentless_relations);
        check(detail::launch_collation(p, phase, stream), "device collation kernels");
    }
};

DeviceCollator::DeviceCollator(ContextPtr ctx, std::vector<DeviceCollationBatch> batches, cudaStream_t stream)
{
    if (!ctx) invalid("device collate needs a context");
    DeviceGuard guard(ctx->device());
    m_impl = std::make_unique<Impl>(std::move(ctx), std::move(batches), stream);
}
DeviceCollator::~DeviceCollator() = default;
u64 DeviceCollator::graphs() const noexcept { return m_impl->params.graphs; }
u64 DeviceCollator::scratch_bytes() const noexcept { return m_impl->total_bytes; }
void DeviceCollator::encode(void* scratch, u64 bytes, const DeviceOutput& out) const { m_impl->run(0, scratch, bytes, out); }
void DeviceCollator::count(void* scratch, u64 bytes, encoding::FlatSizes* sizes, u64* numeric_size,
                           encoding::StaticSizes* statics) const
{
    DeviceOutput out;
    out.sizes = sizes;
    out.numeric_size = numeric_size;
    out.static_sizes = statics;
    if (m_impl->params.shared_statics)
        out.static_capacities = {~u64{0}, ~u64{0}};
    out.capacities = {graphs(), ~u64{0}, ~u64{0}, ~u64{0}, 0, ~u64{0}};
    m_impl->run(1, scratch, bytes, out);
}
void DeviceCollator::fill(void* scratch, u64 bytes, const DeviceOutput& out) const { m_impl->run(2, scratch, bytes, out); }
}
