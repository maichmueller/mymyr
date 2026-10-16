#include "mymyr/cuda/encoding.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mymyr::cuda
{
namespace
{
u64 add(u64 a, u64 b)
{
    if (b > std::numeric_limits<u64>::max() - a)
        throw std::overflow_error("mymyr: encoding: byte size overflow");
    return a + b;
}
u64 mul(u64 a, u64 b)
{
    if (b && a > std::numeric_limits<u64>::max() / b)
        throw std::overflow_error("mymyr: encoding: size product overflow");
    return a * b;
}
u64 round_up(u64 n) { return add(n, 63) & ~u64{63}; }

void finish_use(const std::shared_ptr<LastUse>& last, cudaStream_t stream) noexcept
{
    if (!last)
        return;
    try { last->record(stream); }
    catch (...) { (void)cudaStreamSynchronize(stream); }
}

void refuse_capture(cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &status), "cudaStreamIsCapturing");
    if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument(
            "mymyr: encoding: acquire outside a stream capture and hold the acquisition through the last replay");
}

[[noreturn]] void not_an_extension(const std::string& why)
{
    throw std::invalid_argument("mymyr: encoding: the grown context is not an extension of the uploaded one: " + why);
}
void check_extension(const encoding::EncodingContext& current, const encoding::EncodingContext& grown)
{
    if (grown.schema().fingerprint() != current.schema().fingerprint())
        not_an_extension("the schema differs");
    if (grown.max_arity() != current.max_arity())
        not_an_extension("the widest relation differs");
    if (grown.table().size() < current.table().size())
        not_an_extension("it has fewer instances");
    for (const rl::ArrayInfo& a : current.arrays().arrays())
    {
        const rl::ArrayInfo* b = grown.arrays().find(a.name);
        if (!b || b->dtype != a.dtype || b->elements() < a.elements())
            not_an_extension("the array " + a.name.str() + " is missing or shorter");
        if (std::memcmp(current.arrays().data(a), grown.arrays().data(*b), mul(a.elements(), rl::dtype_bytes(a.dtype))))
            not_an_extension("the array " + a.name.str() + " differs in the rows of the uploaded instances");
    }
}
void check_schema(const encoding::EncodingSchema& schema)
{
    for (const auto& relation : schema.relations())
        if (schema.options().include_derived && relation.lane == encoding::Lane::Derived)
            throw std::invalid_argument("mymyr: encoding: the device does not encode derived relations: "
                                        "set include_derived = false");
    if (schema.has_goal_complete())
        throw std::invalid_argument("mymyr: encoding: the device does not encode goal relations: "
                                    "set goal_derivations = {}");
    if (schema.options().predicate_virtual_nodes)
        throw std::invalid_argument("mymyr: encoding: the device does not encode predicate virtual nodes: "
                                    "set predicate_virtual_nodes = false");
    if (schema.options().targets.contains(encoding::TargetSource::Action))
        throw std::invalid_argument("mymyr: encoding: the device does not encode action targets: "
                                    "remove 'action' from target_sources");
    if (schema.options().layout != encoding::Layout::Graph)
        throw std::invalid_argument("mymyr: encoding: device scratch needs graph-major layout");
}
// The state, static and nullary lanes; the goal, target and virtual-node lanes must add their bounds here.
GraphBounds graph_bounds(const encoding::EncodingContext& host)
{
    const auto v = host.view();
    GraphBounds bounds;
    for (u32 i = 0; i < host.table().size(); ++i)
    {
        const auto inst = encoding::detail::instance_view(v, i);
        GraphBounds b;
        b.cells = inst.num_cells;
        bool nullary = false;
        for (u32 c = 0; c < inst.num_cells; ++c)
        {
            const auto& cell = inst.cells[c];
            if (encoding::detail::is_action_op(cell.op))  // the device has no action input: such a cell has no tuple
                continue;
            b.tuples = add(b.tuples, cell.b - cell.a);
            b.args = add(b.args, mul(cell.b - cell.a, cell.arity));
            nullary = nullary || cell.logical_arity == 0;
        }
        b.entities = add(inst.num_objects, nullary ? 1 : 0);
        bounds.entities = std::max(bounds.entities, b.entities);
        bounds.tuples = std::max(bounds.tuples, b.tuples);
        bounds.args = std::max(bounds.args, b.args);
        bounds.cells = std::max(bounds.cells, b.cells);
    }
    return bounds;
}
BatchShape worst_shape(u64 graphs, const GraphBounds& bounds)
{
    return {graphs, mul(graphs, bounds.entities), mul(graphs, bounds.args)};
}
}  // namespace

EncodingAcquisition::EncodingAcquisition(encoding::detail::ContextView v, std::shared_ptr<const DeviceBuffer> gen,
                                         u64 n, std::shared_ptr<LastUse> last, cudaStream_t stream)
    : view(v), generation(std::move(gen)), number(n), m_last(std::move(last)), m_stream(stream)
{
}
EncodingAcquisition::EncodingAcquisition(EncodingAcquisition&& other) noexcept
    : view(other.view), generation(std::move(other.generation)), number(other.number),
      m_last(std::move(other.m_last)), m_stream(other.m_stream)
{
}
EncodingAcquisition::~EncodingAcquisition() { finish_use(m_last, m_stream); }

std::shared_ptr<DeviceEncodingContext> DeviceEncodingContext::upload(ContextPtr ctx,
    std::shared_ptr<const encoding::EncodingContext> host, cudaStream_t stream)
{
    if (!ctx || !host)
        throw std::invalid_argument("mymyr: encoding: a device context needs a device context and a host context");
    DeviceGuard guard(ctx->device());
    auto out = std::shared_ptr<DeviceEncodingContext>(new DeviceEncodingContext(std::move(ctx), host));
    out->copy(host, stream ? stream : out->m_ctx->stream());
    return out;
}
DeviceEncodingContext::DeviceEncodingContext(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host)
    : m_ctx(std::move(ctx)), m_host(std::move(host))
{
}
DeviceEncodingContext::~DeviceEncodingContext()
{
    if (m_generation)
    {
        try { DeviceGuard guard(m_ctx->device()); m_ready.synchronize(); }
        catch (...) {}
    }
}
void DeviceEncodingContext::copy(const std::shared_ptr<const encoding::EncodingContext>& host, cudaStream_t stream)
{
    if (m_generation)
    {
        m_ready.synchronize();
        m_staging = PinnedLease();
    }
    auto staging = m_ctx->lease_pinned(1, host->arrays().bytes());
    std::memcpy(staging.data(0), host->arrays().block(), host->arrays().bytes());
    auto block = std::make_shared<DeviceBuffer>(m_ctx, host->arrays().bytes());
    Event allocated;
    allocated.record(m_ctx->stream());
    allocated.wait_on(stream);
    auto last = std::make_shared<LastUse>(m_ctx->device());
    block->track(last);
    try
    {
        check(cudaMemcpyAsync(block->data(), staging.data(0), block->size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync (encoding context)");
        last->record(stream);
        m_ready.record(stream);
    }
    catch (...)
    {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    m_view = host->view(static_cast<const std::byte*>(block->data()));
    m_block = std::move(block);
    m_staging = std::move(staging);
    m_host = host;
    m_uses.clear();
    ++m_generation;
}
bool DeviceEncodingContext::refresh(std::shared_ptr<const encoding::EncodingContext> grown, cudaStream_t stream)
{
    if (!grown)
        throw std::invalid_argument("mymyr: encoding: a refresh needs a host context");
    std::lock_guard lock(m_mutex);
    check_extension(*m_host, *grown);
    if (grown->table().size() == m_host->table().size())
        return false;
    DeviceGuard guard(m_ctx->device());
    copy(grown, stream ? stream : m_ctx->stream());
    return true;
}
std::shared_ptr<const encoding::EncodingContext> DeviceEncodingContext::host() const
{
    std::lock_guard lock(m_mutex);
    return m_host;
}
u32 DeviceEncodingContext::num_instances() const
{
    std::lock_guard lock(m_mutex);
    return m_host->table().size();
}
u64 DeviceEncodingContext::bytes() const
{
    std::lock_guard lock(m_mutex);
    return m_block->size();
}
u64 DeviceEncodingContext::generation() const
{
    std::lock_guard lock(m_mutex);
    return m_generation;
}
EncodingAcquisition DeviceEncodingContext::acquire(cudaStream_t stream, std::shared_ptr<LastUse> use) const
{
    std::lock_guard lock(m_mutex);
    DeviceGuard guard(m_ctx->device());
    if (!stream)
        stream = m_ctx->stream();
    refuse_capture(stream);
    m_ready.wait_on(stream);
    auto& last = m_uses[stream];
    if (use ? last != use : !last)
    {
        last = use ? std::move(use) : std::make_shared<LastUse>(m_ctx->device());
        m_block->track(last);
    }
    return {m_view, m_block, m_generation, last, stream};
}
void DeviceEncodingContext::release(cudaStream_t stream) const
{
    std::lock_guard lock(m_mutex);
    m_uses.erase(stream ? stream : m_ctx->stream());
}
std::vector<std::byte> DeviceEncodingContext::download(cudaStream_t stream) const
{
    auto lease = acquire(stream);
    DeviceGuard guard(m_ctx->device());
    if (!stream)
        stream = m_ctx->stream();
    std::vector<std::byte> out(lease.generation->size());
    auto staging = m_ctx->lease_pinned(1, out.size());
    check(cudaMemcpyAsync(staging.data(0), lease.generation->data(), out.size(), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync (encoding context)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::memcpy(out.data(), staging.data(0), out.size());
    return out;
}

OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations)
{
    const u64 I = index == encoding::IndexType::I32 ? 4 : 8;
    const u64 G = shape.graphs, E = shape.entities, A = shape.args, GR = mul(G, relations);
    OutputLayout out;
    u64 at = 0;
    auto place = [&](u64& offset, u64 bytes)
    {
        offset = at;
        at = add(at, round_up(bytes));
    };
    place(out.task_ids, mul(4, G));
    place(out.num_objects, mul(I, G));
    place(out.entity_offsets, mul(I, add(G, 1)));
    place(out.entity_role, E);
    place(out.entity_ref, mul(I, E));
    place(out.relation_counts, mul(I, GR));
    place(out.relation_offsets, mul(I, add(GR, 1)));
    place(out.relation_arg_offsets, mul(I, add(GR, 1)));
    place(out.relation_args, mul(I, A));
    out.bytes = at;
    return out;
}
ScratchLayout scratch_layout(const BatchShape& shape, const GraphBounds& bounds, encoding::Layout layout)
{
    if (layout != encoding::Layout::Graph)
        throw std::invalid_argument("mymyr: encoding: device scratch needs graph-major layout");
    ScratchLayout out;
    const u64 G = shape.graphs;
    const u64 scanned = add(add(mul(G, bounds.entities), mul(G, bounds.tuples)), mul(G, bounds.args));
    out.wide = scanned > std::numeric_limits<u32>::max();
    const u64 row = mul(out.wide ? 8 : 4, add(G, 1));
    out.tuples = row;
    out.args = mul(2, row);
    out.scan_length = mul(3, add(G, 1));
    out.cell_counts = round_up(mul(3, row));
    out.cell_entries = mul(G, bounds.cells);
    out.scan_temp = add(out.cell_counts, round_up(mul(4, out.cell_entries)));
    out.scan_temp_bytes = totals_scan_temp_bytes(out.scan_length, out.wide);
    out.bytes = add(out.scan_temp, round_up(out.scan_temp_bytes));
    return out;
}

EncodingWorkspace::EncodingWorkspace(EncodingAcquisition c, std::shared_ptr<const DeviceBuffer> s, BatchShape sh,
                                     ScratchLayout l)
    : context(std::move(c)), scratch(std::move(s)), shape(sh), layout(l)
{
}
ScratchView EncodingWorkspace::scratch_view() const noexcept
{
    auto* base = static_cast<std::byte*>(scratch->data());
    ScratchView out;
    out.totals = base + layout.entities;
    out.totals_stride = shape.graphs + 1;
    out.scan_length = layout.scan_length;
    out.cell_counts = reinterpret_cast<u32*>(base + layout.cell_counts);
    out.cell_entries = layout.cell_entries;
    out.scan_temp = base + layout.scan_temp;
    out.scan_temp_bytes = layout.scan_temp_bytes;
    out.wide = layout.wide;
    return out;
}
cudaError_t EncodingWorkspace::scan_totals() const
{
    const ScratchView v = scratch_view();
    return launch_totals_scan(v.totals, v.scan_length, v.wide, v.scan_temp, v.scan_temp_bytes, stream());
}

DeviceEncoder::DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder)
    : m_ctx(std::move(ctx)), m_encoder(std::move(encoder))
{
    if (!m_ctx || !m_encoder)
        throw std::invalid_argument("mymyr: encoding: a device encoder needs a device context and an encoder");
    m_schema = m_encoder->schema_ptr();
    check_schema(*m_schema);
    m_bounds = graph_bounds(m_encoder->context());
    m_context = DeviceEncodingContext::upload(m_ctx, m_encoder->context_ptr());
}
std::shared_ptr<const encoding::FlatEncoder> DeviceEncoder::encoder() const
{
    std::shared_lock lock(m_state_mutex);
    return m_encoder;
}
GraphBounds DeviceEncoder::bounds() const
{
    std::shared_lock lock(m_state_mutex);
    return m_bounds;
}
BatchShape DeviceEncoder::worst_case(u64 graphs) const { return worst_shape(graphs, bounds()); }
ScratchLayout DeviceEncoder::scratch_layout(const BatchShape& shape) const
{
    DeviceGuard guard(m_ctx->device());
    return cuda::scratch_layout(shape, bounds(), options().layout);
}
u64 DeviceEncoder::bytes_needed(const BatchShape& shape) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto output = output_layout(shape, options().index, schema().num_relations());
    return add(cuda::scratch_layout(shape, m_bounds, options().layout).bytes, output.bytes);
}
u64 DeviceEncoder::bytes_needed(u64 graphs) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto shape = worst_shape(graphs, m_bounds);
    return add(cuda::scratch_layout(shape, m_bounds, options().layout).bytes,
               output_layout(shape, options().index, schema().num_relations()).bytes);
}

std::shared_ptr<DeviceEncoder::Workspace> DeviceEncoder::workspace(cudaStream_t stream) const
{
    std::lock_guard lock(m_workspace_mutex);
    const auto it = m_workspaces.find(stream);
    return it == m_workspaces.end() ? nullptr : it->second;
}
void DeviceEncoder::reserve_workspace(Workspace& work, const BatchShape& shape)
{
    std::lock_guard lock(work.mutex);
    const BatchShape want{std::max(shape.graphs, work.shape.graphs), std::max(shape.entities, work.shape.entities),
                          std::max(shape.args, work.shape.args)};
    (void)output_layout(want, options().index, schema().num_relations());
    const auto layout = cuda::scratch_layout(want, m_bounds, options().layout);
    if (!work.scratch || layout.bytes != work.layout.bytes || layout.wide != work.layout.wide ||
        layout.tuples != work.layout.tuples || layout.cell_counts != work.layout.cell_counts ||
        layout.scan_temp != work.layout.scan_temp)
    {
        auto scratch = std::make_shared<DeviceBuffer>(m_ctx, layout.bytes);
        auto last = std::make_shared<LastUse>(m_ctx->device());
        scratch->track(last);
        work.ready.record(m_ctx->stream());
        work.scratch = std::move(scratch);
        work.last = std::move(last);
    }
    work.shape = want;
    work.layout = layout;
}
void DeviceEncoder::reserve_locked(const BatchShape& shape, cudaStream_t stream)
{
    if (!stream)
        stream = m_ctx->stream();
    // Validate output sizes as well, before any device allocation.
    (void)output_layout(shape, options().index, schema().num_relations());
    auto work = workspace(stream);
    if (!work)
    {
        // The stream gains an entry only once its first reservation succeeded.
        auto fresh = std::make_shared<Workspace>();
        reserve_workspace(*fresh, shape);
        std::lock_guard map_lock(m_workspace_mutex);
        work = m_workspaces.try_emplace(stream, fresh).first->second;
        if (work != fresh)
            reserve_workspace(*work, shape);
    }
    else
        reserve_workspace(*work, shape);
    std::lock_guard work_lock(work->mutex);
    work->ready.wait_on(stream);
}
void DeviceEncoder::reserve(const BatchShape& shape, cudaStream_t stream)
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    reserve_locked(shape, stream);
}
void DeviceEncoder::reserve(u64 graphs, cudaStream_t stream)
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    reserve_locked(worst_shape(graphs, m_bounds), stream);
}
BatchShape DeviceEncoder::reserved(cudaStream_t stream) const
{
    auto work = workspace(stream ? stream : m_ctx->stream());
    if (!work)
        return {};
    std::lock_guard lock(work->mutex);
    return work->shape;
}
u64 DeviceEncoder::reserved_bytes(cudaStream_t stream) const
{
    auto work = workspace(stream ? stream : m_ctx->stream());
    if (!work)
        return 0;
    std::lock_guard lock(work->mutex);
    return work->scratch ? work->scratch->size() : 0;
}
OutputLayout DeviceEncoder::reserved_layout(cudaStream_t stream) const
{
    return output_layout(reserved(stream), options().index, schema().num_relations());
}
EncodingWorkspace DeviceEncoder::acquire(cudaStream_t stream) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    if (!stream)
        stream = m_ctx->stream();
    refuse_capture(stream);
    auto work = workspace(stream);
    if (!work)
        throw std::invalid_argument("mymyr: encoding: reserve scratch on this stream before acquire");
    std::lock_guard work_lock(work->mutex);
    const auto needed = cuda::scratch_layout(work->shape, m_bounds, options().layout);
    if (!work->scratch || needed.bytes != work->layout.bytes || needed.wide != work->layout.wide ||
        needed.scan_temp != work->layout.scan_temp)
        throw std::invalid_argument("mymyr: encoding: reserve scratch again after a failed refresh");
    work->ready.wait_on(stream);
    return {m_context->acquire(stream, work->last), work->scratch, work->shape, work->layout};
}
void DeviceEncoder::release(cudaStream_t stream)
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    if (!stream)
        stream = m_ctx->stream();
    std::shared_ptr<Workspace> dropped;
    {
        std::lock_guard map_lock(m_workspace_mutex);
        const auto it = m_workspaces.find(stream);
        if (it != m_workspaces.end())
        {
            dropped = std::move(it->second);
            m_workspaces.erase(it);
        }
    }
    m_context->release(stream);
}
bool DeviceEncoder::refresh(std::shared_ptr<const encoding::FlatEncoder> grown, cudaStream_t stream)
{
    if (!grown)
        throw std::invalid_argument("mymyr: encoding: a refresh needs an encoder");
    std::unique_lock lock(m_state_mutex);
    if (grown->schema().fingerprint() != m_schema->fingerprint())
        throw std::invalid_argument("mymyr: encoding: the grown encoder has another schema or options");
    check_schema(grown->schema());
    const auto bounds = graph_bounds(grown->context());
    const bool copied = m_context->refresh(grown->context_ptr(), stream);
    m_encoder = std::move(grown);
    m_bounds = bounds;
    std::vector<std::shared_ptr<Workspace>> workspaces;
    {
        std::lock_guard map_lock(m_workspace_mutex);
        for (const auto& [s, work] : m_workspaces)
            workspaces.push_back(work);
    }
    DeviceGuard guard(m_ctx->device());
    for (const auto& work : workspaces)
        reserve_workspace(*work, {});
    return copied;
}
}  // namespace mymyr::cuda
