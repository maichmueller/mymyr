#include "mymyr/cuda/encoding.hpp"

#include "encoding_kernels.hpp"

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
/// The flags at the front of the scratch, cleared with the cell row before each batch.
constexpr u64 k_flag_bytes = 64;

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
    if (schema.has_horizon())
        throw std::invalid_argument("mymyr: encoding: the device does not encode horizon relations (horizons, rooted "
                                    "horizons, transitions, and their parent, sibling and cousin relations): "
                                    "set horizon_mode = None");
    if (schema.has_subgoals())
        throw std::invalid_argument("mymyr: encoding: the device does not encode subgoal relations: "
                                    "set max_goal_level = 0");
    if (schema.has_history())
        throw std::invalid_argument("mymyr: encoding: the device does not encode history relations: "
                                    "set history = false");
    if (schema.options().targets.contains(encoding::TargetSource::Action))
        throw std::invalid_argument("mymyr: encoding: the device does not encode action targets: "
                                    "remove 'action' from target_sources");
}

/// What the device needs to know about a context: the bounds of its graphs, and whether the graphs of some instance read
/// derived words, by the goals of the rows (none, over the fluent slots, also over the derived slots).
struct ContextFacts
{
    GraphBounds bounds;
    std::array<bool, 3> reads_derived{};
};
ContextFacts context_facts(const encoding::EncodingContext& host)
{
    namespace ed = encoding::detail;
    const auto v = host.view();
    ContextFacts facts;
    GraphBounds& bounds = facts.bounds;
    for (u32 i = 0; i < host.table().size(); ++i)
    {
        const ed::InstanceView inst = ed::instance_view(v, i);
        bounds.words = std::max<u64>(bounds.words, std::max(inst.words, inst.derived_words));
        facts.reads_derived[0] = facts.reads_derived[0] || ed::reads_derived(inst, false, false);
        facts.reads_derived[1] = facts.reads_derived[1] || ed::reads_derived(inst, true, false);
        facts.reads_derived[2] = facts.reads_derived[2] || ed::reads_derived(inst, true, true);
        // the graph of a row has the cells of the instance's goal or of the row's own goal, and no action cells
        for (const ed::CellList list : {ed::cell_list(inst, false, false), ed::cell_list(inst, true, false)})
        {
            u64 tuples = 0, args = 0, targets = 0;
            bool nullary = false;
            for (u32 c = 0; c < list.count; ++c)
            {
                const ed::Cell& cell = list.cells[c];
                const u64 n = u64{cell.b - cell.a} + (ed::is_goal_op(cell.op) ? cell.d - cell.c : 0u);
                tuples = add(tuples, n);
                args = add(args, mul(n, cell.arity));
                if (ed::is_goal_op(cell.op) && ed::goal_op_mode(cell.op) == ed::GoalMode::PlainTarget)
                    targets = add(targets, n);
                nullary = nullary || (n > 0 && cell.logical_arity == 0);
            }
            // objects, the auxiliary entity, one entity per goal target and one per predicate virtual node
            const u64 entities = add(add(inst.num_objects, nullary ? 1 : 0), add(targets, inst.num_virtual));
            bounds.entities = std::max(bounds.entities, entities);
            bounds.tuples = std::max(bounds.tuples, tuples);
            bounds.args = std::max(bounds.args, args);
            bounds.targets = std::max(bounds.targets, targets);
            bounds.cells = std::max<u64>(bounds.cells, list.count);
        }
    }
    return facts;
}
BatchShape worst_shape(u64 graphs, const GraphBounds& bounds)
{
    return {graphs, mul(graphs, bounds.entities), mul(graphs, bounds.args), mul(graphs, bounds.targets)};
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

OutputArrays output_arrays(const encoding::EncodingSchema& schema)
{
    OutputArrays arrays;
    arrays.entity_arrays = !schema.implicit_entity_arrays();
    arrays.counts = schema.emits_counts();
    arrays.goal_complete = schema.has_goal_complete();
    arrays.targets = schema.has_targets();
    return arrays;
}

OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations, const OutputArrays& arrays)
{
    const u64 I = index == encoding::IndexType::I32 ? 4 : 8;
    const u64 G = shape.graphs, E = shape.entities, A = shape.args, T = shape.targets, GR = mul(G, relations);
    OutputLayout out;
    out.arrays = arrays;
    u64 at = 0;
    auto place = [&](u64& offset, u64 bytes)
    {
        offset = at;
        at = add(at, round_up(bytes));
    };
    place(out.task_ids, mul(4, G));
    place(out.num_objects, mul(I, G));
    place(out.entity_offsets, mul(I, add(G, 1)));
    place(out.entity_role, arrays.entity_arrays ? E : 0);
    place(out.entity_ref, arrays.entity_arrays ? mul(I, E) : 0);
    place(out.relation_counts, arrays.counts ? mul(I, GR) : 0);
    place(out.relation_offsets, mul(I, add(GR, 1)));
    place(out.relation_arg_offsets, mul(I, add(GR, 1)));
    place(out.relation_args, mul(I, A));
    place(out.goal_complete, arrays.goal_complete ? G : 0);
    place(out.target_offsets, arrays.targets ? mul(I, add(G, 1)) : 0);
    place(out.target_entity, arrays.targets ? mul(I, T) : 0);
    place(out.target_group, arrays.targets ? T : 0);
    place(out.sizes, sizeof(encoding::FlatSizes));
    out.bytes = at;
    return out;
}
DeviceOutput output_of(const OutputLayout& layout, const BatchShape& shape, void* block)
{
    auto* base = static_cast<std::byte*>(block);
    auto at = [&](u64 offset, bool present) -> void* { return present ? base + offset : nullptr; };
    const OutputArrays& a = layout.arrays;
    DeviceOutput out;
    encoding::FlatBuffers& b = out.arrays;
    b.task_ids = static_cast<i32*>(at(layout.task_ids, true));
    b.num_objects = at(layout.num_objects, true);
    b.entity_offsets = at(layout.entity_offsets, true);
    b.entity_role = static_cast<u8*>(at(layout.entity_role, a.entity_arrays));
    b.entity_ref = at(layout.entity_ref, a.entity_arrays);
    b.relation_counts = at(layout.relation_counts, a.counts);
    b.relation_offsets = at(layout.relation_offsets, true);
    b.relation_arg_offsets = at(layout.relation_arg_offsets, true);
    b.relation_args = at(layout.relation_args, true);
    b.goal_complete = static_cast<u8*>(at(layout.goal_complete, a.goal_complete));
    b.target_offsets = at(layout.target_offsets, a.targets);
    b.target_entity = at(layout.target_entity, a.targets);
    b.target_group = static_cast<u8*>(at(layout.target_group, a.targets));
    out.capacities = {shape.entities, shape.args, shape.args, shape.targets};
    out.sizes = static_cast<encoding::FlatSizes*>(at(layout.sizes, true));
    return out;
}

u64 ScratchLayout::scanned(u64 graphs) const
{
    if (!relation_major)
        return mul(4, add(graphs, 1));
    return add(add(mul(2, add(graphs, 1)), mul(graphs, relations)), 1);
}
ScratchLayout scratch_layout(const BatchShape& shape, const GraphBounds& bounds, encoding::Layout layout, u64 relations)
{
    ScratchLayout out;
    out.relation_major = layout == encoding::Layout::Relation;
    out.relations = relations;
    const u64 G = shape.graphs;
    // the largest value of the scan: the sum of the totals of every row (the argument row only graph-major)
    u64 scanned = add(add(mul(G, bounds.entities), mul(G, bounds.targets)), mul(G, bounds.tuples));
    if (!out.relation_major)
        scanned = add(scanned, mul(G, bounds.args));
    out.wide = scanned > std::numeric_limits<u32>::max();
    out.scan_length = out.scanned(G);
    out.flags = 0;
    out.totals = k_flag_bytes;
    out.cell_counts = round_up(add(k_flag_bytes, mul(out.wide ? 8 : 4, out.scan_length)));
    out.cell_stride = bounds.cells;
    out.cell_entries = out.relation_major ? 0 : mul(G, bounds.cells);
    out.scan_temp = add(out.cell_counts, round_up(mul(4, out.cell_entries)));
    out.scan_temp_bytes = detail::totals_scan_temp_bytes(out.scan_length, out.wide);
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
    out.flags = reinterpret_cast<u32*>(base + layout.flags);
    out.totals = base + layout.totals;
    out.cell_counts = reinterpret_cast<u32*>(base + layout.cell_counts);
    out.cell_stride = layout.cell_stride;
    out.scan_temp = base + layout.scan_temp;
    out.scan_temp_bytes = layout.scan_temp_bytes;
    out.wide = layout.wide;
    return out;
}

DeviceEncoder::DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder)
    : m_ctx(std::move(ctx)), m_encoder(std::move(encoder))
{
    if (!m_ctx || !m_encoder)
        throw std::invalid_argument("mymyr: encoding: a device encoder needs a device context and an encoder");
    m_schema = m_encoder->schema_ptr();
    check_schema(*m_schema);
    const ContextFacts facts = context_facts(m_encoder->context());
    m_bounds = facts.bounds;
    m_reads_derived = facts.reads_derived;
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
OutputLayout DeviceEncoder::output_layout(const BatchShape& shape) const
{
    return cuda::output_layout(shape, options().index, schema().num_relations(), output_arrays(schema()));
}
ScratchLayout DeviceEncoder::scratch_layout(const BatchShape& shape) const
{
    DeviceGuard guard(m_ctx->device());
    return cuda::scratch_layout(shape, bounds(), options().layout, schema().num_relations());
}
u64 DeviceEncoder::bytes_needed(const BatchShape& shape) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto output = output_layout(shape);
    return add(cuda::scratch_layout(shape, m_bounds, options().layout, schema().num_relations()).bytes, output.bytes);
}
u64 DeviceEncoder::bytes_needed(u64 graphs) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto shape = worst_shape(graphs, m_bounds);
    return add(cuda::scratch_layout(shape, m_bounds, options().layout, schema().num_relations()).bytes,
               output_layout(shape).bytes);
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
                          std::max(shape.args, work.shape.args), std::max(shape.targets, work.shape.targets)};
    (void)output_layout(want);
    const auto layout = cuda::scratch_layout(want, m_bounds, options().layout, schema().num_relations());
    if (!work.scratch || layout != work.layout)
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
    (void)output_layout(shape);
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
OutputLayout DeviceEncoder::reserved_layout(cudaStream_t stream) const { return output_layout(reserved(stream)); }
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
    const auto needed = cuda::scratch_layout(work->shape, m_bounds, options().layout, schema().num_relations());
    if (!work->scratch || needed != work->layout)
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
    const ContextFacts facts = context_facts(grown->context());
    const bool copied = m_context->refresh(grown->context_ptr(), stream);
    m_encoder = std::move(grown);
    m_bounds = facts.bounds;
    m_reads_derived = facts.reads_derived;
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

namespace
{
[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }

/// The lanes of the count kernel for rows of at most `words` words.
u32 count_lanes_for(u64 words) { return words <= 2 ? 4 : words <= 8 ? 8 : 32; }
}  // namespace

void DeviceEncoder::encode(const EncodingWorkspace& work, const DeviceEncodeInput& in, const DeviceOutput& out,
                           const LaunchConfig& launch) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const encoding::EncodingSchema& schema = *m_schema;
    const encoding::FlatOptions& opts = schema.options();
    const u64 graphs = in.graphs;
    const u64 R = schema.num_relations();
    const bool relation_major = opts.layout == encoding::Layout::Relation;

    // what can be checked without reading device memory
    if (graphs > work.shape.graphs)
        invalid("the batch has " + std::to_string(graphs) + " graphs, more than the " +
                std::to_string(work.shape.graphs) + " the workspace was reserved for");
    if (work.layout.relation_major != relation_major || work.layout.relations != R)
        invalid("the workspace does not belong to this encoder");
    if (graphs && !in.states)
        invalid("the state buffer is null");
    if (in.state_stride < in.state_words)
        invalid("the state row stride is smaller than its width");
    if (in.derived && in.derived_stride < m_encoder->context().max_derived_words())
        invalid("the derived row stride is smaller than the derived words of an instance");
    if (in.goal_pos && !schema.has_goal_complete())
        invalid("goals are given, but the schema has no goal relations");
    if ((in.goal_pos == nullptr) != (in.goal_neg == nullptr))
        invalid("the goal needs both goal_pos and goal_neg");
    if ((in.goal_derived_pos == nullptr) != (in.goal_derived_neg == nullptr))
        invalid("the derived goal needs both goal_derived_pos and goal_derived_neg");
    if (in.goal_derived_pos && !in.goal_pos)
        invalid("the derived goal masks need the goal masks over the fluent slots");
    if (!in.derived && m_reads_derived[in.goal_pos ? (in.goal_derived_pos ? 2 : 1) : 0])
        invalid("the device does not evaluate derived atoms: pass the derived words of the rows (derived)");
    const OutputArrays arrays = output_arrays(schema);
    const encoding::FlatBuffers& a = out.arrays;
    if (!a.task_ids || !a.num_objects || !a.entity_offsets || !a.relation_offsets || !a.relation_arg_offsets || !out.sizes)
        invalid("the output needs task_ids, num_objects, entity_offsets, relation_offsets, relation_arg_offsets and sizes");
    if ((a.entity_role != nullptr) != arrays.entity_arrays || (a.entity_ref != nullptr) != arrays.entity_arrays)
        invalid("entity_role and entity_ref must be given iff the schema stores entity arrays");
    if ((a.relation_counts != nullptr) != arrays.counts)
        invalid("relation_counts must be given iff the schema stores counts");
    if ((a.goal_complete != nullptr) != arrays.goal_complete)
        invalid("goal_complete must be given iff the schema has goal relations");
    if ((a.target_offsets != nullptr) != arrays.targets || (a.target_entity != nullptr) != arrays.targets ||
        (a.target_group != nullptr) != arrays.targets)
        invalid("the target arrays must be given iff the schema has target sources");
    if (out.capacities.args && !a.relation_args)
        invalid("relation_args is null");
    if (arrays.targets && out.capacities.targets && (!a.target_entity || !a.target_group))
        invalid("the target arrays are null");
    if (launch.count_lanes != 0 && launch.count_lanes != 4 && launch.count_lanes != 8 && launch.count_lanes != 32)
        invalid("the count kernel runs 4, 8 or 32 lanes per graph");

    // how it is launched
    const u32 virtual_nodes = work.view().num_virtual;
    const u32 lanes = launch.count_lanes ? launch.count_lanes : count_lanes_for(m_bounds.words);
    bool warp = launch.fill == FillMode::Warp;
    if (launch.fill == FillMode::Auto)
        warp = m_bounds.words > 8 || m_bounds.cells > 128 || m_bounds.args > 1024 ||
               detail::fill_shared_bytes(virtual_nodes, static_cast<u32>(R), false, relation_major) > detail::k_shared_limit;
    if (detail::count_shared_bytes(virtual_nodes, lanes) > detail::k_shared_limit ||
        detail::fill_shared_bytes(virtual_nodes, static_cast<u32>(R), warp, relation_major) > detail::k_shared_limit)
        invalid("the schema has too many predicate virtual nodes or relations for the device kernels");

    const ScratchView scratch = work.scratch_view();
    const encoding::IndexType index = opts.index;
    const u64 limit = index == encoding::IndexType::I64 ? static_cast<u64>(std::numeric_limits<i64>::max())
                                                       : static_cast<u64>(std::numeric_limits<i32>::max());
    detail::EncodeParams p;
    p.context = work.view();
    p.graphs = graphs;
    p.states = in.states;
    p.state_stride = in.state_stride;
    p.state_words = in.state_words;
    p.task_ids = in.task_ids;
    p.derived = in.derived;
    p.derived_stride = in.derived_stride;
    p.goal_pos = in.goal_pos;
    p.goal_neg = in.goal_neg;
    p.goal_stride = in.goal_stride;
    p.goal_words = in.goal_words;
    p.goal_derived_pos = in.goal_derived_pos;
    p.goal_derived_neg = in.goal_derived_neg;
    p.goal_derived_stride = in.goal_derived_stride;
    p.goal_derived_words = in.goal_derived_words;
    p.order = launch.order;
    p.out = out.arrays;
    p.sizes = out.sizes;
    p.capacities = out.capacities;
    p.index64 = index == encoding::IndexType::I64;
    p.relation_major = relation_major;
    if (mul(graphs, R) >= limit || graphs >= limit)
        p.static_overflow |= encoding::k_overflow_index;
    p.totals = scratch.totals;
    p.wide = scratch.wide;
    p.segments = mul(2, add(graphs, 1));
    p.flags = scratch.flags;
    p.cell_counts = scratch.cell_counts;
    p.cell_stride = scratch.cell_stride;

    detail::LaunchShape shape;
    shape.lanes = lanes;
    shape.warp_fill = warp;
    shape.scan_length = work.layout.scanned(graphs);
    shape.scan_temp = scratch.scan_temp;
    shape.scan_temp_bytes = scratch.scan_temp_bytes;
    shape.zero_begin = scratch.flags;
    shape.zero_bytes = k_flag_bytes;
    if (relation_major)
    {
        // the flags and the zero-opened cell row, which the count kernel fills only where a graph has a cell
        const u64 entry = work.layout.wide ? 8 : 4;
        shape.zero_bytes = add(k_flag_bytes, mul(entry, shape.scan_length));
    }
    check(detail::launch_encode(p, shape, work.stream()), "encoding kernels");
}
void DeviceEncoder::encode(const DeviceEncodeInput& in, const DeviceOutput& out, cudaStream_t stream,
                           const LaunchConfig& launch) const
{
    const EncodingWorkspace work = acquire(stream);
    encode(work, in, out, launch);
}
}  // namespace mymyr::cuda
