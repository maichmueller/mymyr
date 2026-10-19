#include "encoding_collate.hpp"
#include "mymyr/cuda/encoding.hpp"

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
    u64 total_bytes = 0, temp_offset = 0;

    Impl(ContextPtr ctx, std::vector<DeviceCollationBatch> batches, cudaStream_t s)
        : context(std::move(ctx)), stream(s ? s : context->stream())
    {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "query collation capture");
        if (capture != cudaStreamCaptureStatusNone) invalid("prepare the device collator outside capture");
        if (batches.empty()) invalid("collate needs at least one batch");
        const auto& first = batches.front();
        stored = first.stored;
        params.relations = first.relations;
        params.index64 = first.index == encoding::IndexType::I64;
        params.relation_major = first.layout == encoding::Layout::Relation;
        params.batches = batches.size();
        staging = context->lease_pinned(1, mul(batches.size(), sizeof(detail::CollationSlice)));
        auto* host = static_cast<detail::CollationSlice*>(staging.data(0));
        for (u64 i = 0; i < batches.size(); ++i)
        {
            const auto& b = batches[i];
            if (b.layout != first.layout) invalid("collate received mixed layouts");
            if (b.index != first.index) invalid("collate received mixed index types");
            if (b.shared_statics != first.shared_statics) invalid("collate received mixed statics");
            if (b.schema_fingerprint != first.schema_fingerprint) invalid("collate received mixed schemas");
            if (b.shared_statics) invalid("device collate requires repeated statics");
            if (b.relations != first.relations || b.stored != first.stored) invalid("collate received different stored arrays");
            host[i] = {b, params.graphs};
            params.graphs = add(params.graphs, b.graphs);
        }
        params.entries = add(add(params.graphs, mul(params.graphs, params.relations)), 1);
        const u64 nbytes = mul(params.entries, sizeof(detail::CollationTotals));
        temp_offset = add(nbytes, 255) & ~u64{255};
        params.temp_bytes = detail::collation_scan_bytes(params.entries);
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
        if (phase != 1)
        {
            const auto& a = output.arrays;
            if (output.capacities.graphs < params.graphs) invalid("collated graphs exceed the output capacity");
            if (!a.entity_offsets || !a.relation_offsets || !a.relation_arg_offsets ||
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
            if (output.capacities.args && !a.relation_args) invalid("collation arguments are null");
        }
        auto p = params;
        p.totals = static_cast<detail::CollationTotals*>(scratch);
        p.temp = static_cast<std::byte*>(scratch) + temp_offset;
        p.output = output;
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
void DeviceCollator::count(void* scratch, u64 bytes, encoding::FlatSizes* sizes) const
{
    DeviceOutput out;
    out.sizes = sizes;
    out.capacities = {graphs(), ~u64{0}, ~u64{0}, ~u64{0}};
    m_impl->run(1, scratch, bytes, out);
}
void DeviceCollator::fill(void* scratch, u64 bytes, const DeviceOutput& out) const { m_impl->run(2, scratch, bytes, out); }
}
