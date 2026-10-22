#include "mymyr/cuda/encoding.hpp"

#include "axiom_reads.hpp"
#include "encoding_kernels.hpp"
#include "encoding_scratch.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/rl/task_arrays.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
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
/// The flags at the front of the scratch, cleared with the scanned totals before each batch.
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

/// The facts of a context (detail::EncodingFacts). With `ctx`, the axioms of the instances whose graphs read derived
/// atoms are exported and uploaded, those of `previous` (the facts of a context this one extends) reused; without it
/// only their sizes are computed, which needs no export.
std::shared_ptr<detail::EncodingFacts> context_facts(const encoding::EncodingContext& host, const ContextPtr& ctx = {},
                                                     const detail::EncodingFacts* previous = nullptr)
{
    namespace ed = encoding::detail;
    const auto v = host.view();
    const encoding::EncodingSchema& schema = host.schema();
    auto out = std::make_shared<detail::EncodingFacts>();
    detail::EncodingFacts& facts = *out;
    facts.derived_words = host.max_derived_words();
    GraphBounds& bounds = facts.bounds;
    const bool levels = schema.has_subgoals();
    const bool history = schema.has_history();
    bounds.levels = levels ? schema.options().max_goal_level : 0;
    bounds.history = history;
    const u32 instances = host.table().size();
    std::vector<u32> modes(instances, 0);
    for (u32 i = 0; i < instances; ++i)
    {
        const ed::InstanceView inst = ed::instance_view(v, i);
        facts.widths.push_back(ed::graph_width(inst));
        bounds.words = std::max<u64>(bounds.words, std::max(inst.words, inst.derived_words));
        bounds.state_words = std::max<u64>(bounds.state_words, inst.words);
        bounds.derived_words = std::max<u64>(bounds.derived_words, inst.derived_words);
        // subgoal layers over derived slots read derived atoms under every goal mode (evaluation modes 3 to 5)
        const bool layer_reads = levels && (inst.needs_derived & ed::k_needs_row_goal_derived) != 0;
        facts.reads_derived_layers = facts.reads_derived_layers || layer_reads;
        for (u32 m = 0; m < 3; ++m)
        {
            const bool goal_reads = ed::reads_derived(inst, m != 0, m == 2);
            facts.reads_derived[m] = facts.reads_derived[m] || goal_reads;
            if (goal_reads)
                modes[i] |= 1u << m;
            if (goal_reads || layer_reads)
                modes[i] |= 1u << (m + 3);
        }
        // the graph of a row has the cells of the instance's goal or of the row's own goal and, with subgoal layers
        // or history steps, their cells, and no action cells
        for (const ed::CellList list : {ed::cell_list(inst, false, false, levels, history),
                                        ed::cell_list(inst, true, false, levels, history)})
        {
            u64 tuples = 0, args = 0, targets = 0;
            u64 step_tuples = 0, step_args = 0, step_targets = 0;
            bool nullary = false;
            for (u32 c = 0; c < list.count; ++c)
            {
                const ed::Cell& cell = list.cells[c];
                if (ed::is_history_op(cell.op))
                {
                    // every step of a row may have a tuple for each slot of the cell
                    const u64 n = cell.b - cell.a;
                    step_tuples = add(step_tuples, n);
                    step_args = add(step_args, mul(n, cell.arity));
                    if (ed::history_op_target(cell.op))
                        step_targets = add(step_targets, n);
                    nullary = nullary || (n > 0 && cell.logical_arity == 0);
                    continue;
                }
                const u64 n = u64{cell.b - cell.a} + (ed::is_goal_op(cell.op) ? cell.d - cell.c : 0u);
                tuples = add(tuples, n);
                args = add(args, mul(n, cell.arity));
                if (ed::is_goal_op(cell.op) && ed::goal_op_mode(cell.op) == ed::GoalMode::PlainTarget)
                    targets = add(targets, n);
                nullary = nullary || (n > 0 && cell.logical_arity == 0);
            }
            // objects, the auxiliary entity, one entity per goal or subgoal target and one per predicate virtual node
            const u64 entities = add(add(inst.num_objects, nullary ? 1 : 0), add(targets, inst.num_virtual));
            bounds.entities = std::max(bounds.entities, entities);
            bounds.tuples = std::max(bounds.tuples, tuples);
            bounds.args = std::max(bounds.args, args);
            bounds.targets = std::max(bounds.targets, targets);
            bounds.cells = std::max<u64>(bounds.cells, list.count);
            // a history step has an entity, and one more per tuple of a history target cell
            bounds.step_entities = std::max(bounds.step_entities, add(history ? 1 : 0, step_targets));
            bounds.step_tuples = std::max(bounds.step_tuples, step_tuples);
            bounds.step_args = std::max(bounds.step_args, step_args);
            bounds.step_targets = std::max(bounds.step_targets, step_targets);
        }
    }
    if (!facts.derived_words ||
        !(facts.reads_derived[0] || facts.reads_derived[1] || facts.reads_derived[2] || facts.reads_derived_layers))
        return out;
    facts.axioms.resize(instances);
    for (u32 i = 0; i < instances; ++i)
    {
        auto& axioms = facts.axioms[i];
        axioms.modes = modes[i];
        if (!axioms.modes)
            continue;
        if (previous && i < previous->axioms.size() && previous->axioms[i].modes == axioms.modes &&
            (previous->axioms[i].task || !previous->axioms[i].refusal.empty() || !ctx))
        {
            axioms = previous->axioms[i];
            if (axioms.refusal.empty())
                facts.axiom_view_words = std::max(facts.axiom_view_words, axioms.view_words);
            continue;
        }
        const Task& task = *host.table().instance(i).task;
        for (const auto& st : task.compiled().strata)
            for (const auto& a : st.axioms)
                if (a.body.total > lifted::k_max_depth || a.body.steps.size() > lifted::k_max_depth ||
                    !a.body.npre.empty() || !a.body.nchecks.empty())
                    axioms.refusal = "the axiom body exceeds the device matcher limits: pass derived words";
        axioms.ow = host.table().instance(i).ow;
        if (!(axioms.ow <= 8 || axioms.ow == 12 || axioms.ow == 16 || axioms.ow == 24 || axioms.ow == 32))
            axioms.refusal = "the axiom object bitsets exceed the device limits: pass derived words";
        if (!axioms.refusal.empty())
            continue;
        axioms.view_words = mul(rl::device_view_rows(task), axioms.ow);
        facts.axiom_view_words = std::max(facts.axiom_view_words, axioms.view_words);
        if (!ctx)
            continue;
        auto bundle = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(task));
        const auto view = rl::task_view(*bundle);
        if (view.ow != axioms.ow || mul(view.view_rows, view.ow) != axioms.view_words)
            throw std::logic_error("mymyr: encoding: an instance's exported view differs from its sizing");
        for (u32 a = 0; a < view.n_axioms; ++a)
            axioms.fc = axioms.fc || rl::dev::device_fc(rl::dev::axiom_matcher_flags(view, a), rl::dev::MatchOrder::Free);
        axioms.fc = axioms.fc && axioms.ow <= lifted::k_max_fc_ow;
        axioms.reads = detail::axiom_reads(task.compiled());
        axioms.task = DeviceTask::upload(ctx, std::move(bundle));
    }
    // the launches of each evaluation mode: one per object-bitset width of its evaluated instances, and one that only
    // flags the rows of refused instances when there is none
    for (u32 m = 0; m < detail::k_evaluation_modes; ++m)
    {
        bool refused = false;
        for (const auto& axioms : facts.axioms)
        {
            if (!((axioms.modes >> m) & 1u))
                continue;
            if (!axioms.refusal.empty())
            {
                refused = true;
                continue;
            }
            const auto it = std::ranges::find(facts.class_ow[m], axioms.ow);
            if (it == facts.class_ow[m].end())
            {
                facts.class_ow[m].push_back(axioms.ow);
                facts.class_fc[m].push_back(axioms.fc ? 1 : 0);
            }
            else if (axioms.fc)
                facts.class_fc[m][static_cast<usize>(it - facts.class_ow[m].begin())] = 1;
        }
        if (refused && facts.class_ow[m].empty())
        {
            facts.class_ow[m].push_back(1);
            facts.class_fc[m].push_back(0);
        }
    }
    return out;
}

/// The instances' lifted::EncodingAxiomInstance records followed by their reads, as uploaded: the bytes into `staging`
/// (null: only the size) with the reads at device address `device`.
u64 axiom_records(const detail::EncodingFacts& facts, std::byte* staging, const std::byte* device)
{
    const u64 head = round_up(mul(facts.axioms.size(), sizeof(lifted::EncodingAxiomInstance)));
    u64 at = head;
    for (u64 i = 0; i < facts.axioms.size(); ++i)
    {
        const auto& axioms = facts.axioms[i];
        lifted::EncodingAxiomInstance record;
        record.modes = axioms.modes;
        record.refused = axioms.modes && !axioms.refusal.empty() ? 1 : 0;
        if (axioms.task)
        {
            record.view = axioms.task->view();
            if (!axioms.reads.empty())
            {
                record.reads = reinterpret_cast<const u64*>(device + at);
                if (staging)
                    std::memcpy(staging + at, axioms.reads.data(), axioms.reads.size() * sizeof(u64));
                at = add(at, mul(axioms.reads.size(), sizeof(u64)));
            }
        }
        if (staging)
            std::memcpy(staging + i * sizeof(record), &record, sizeof(record));
    }
    return at;
}
/// The steps of a row's history a worst case allows: those of the longest history given, at most history_max_steps.
/// With `strict`, a schema that encodes every step needs the longest history.
u64 history_steps_of(const encoding::EncodingSchema& schema, u64 history_steps, bool strict)
{
    if (!schema.has_history())
        return 0;
    const std::optional<u32> limit = schema.options().history_max_steps;
    if (limit)
        return history_steps ? std::min<u64>(history_steps, *limit) : *limit;
    if (history_steps == 0 && strict)
        throw std::invalid_argument(
            "mymyr: encoding: the schema encodes every history step: give the most steps one row's history has");
    return history_steps;
}
BatchShape worst_shape(u64 graphs, const GraphBounds& bounds, u64 steps)
{
    return {graphs,
            mul(graphs, add(bounds.entities, mul(steps, bounds.step_entities))),
            mul(graphs, add(bounds.args, mul(steps, bounds.step_args))),
            mul(graphs, add(bounds.targets, mul(steps, bounds.step_targets))),
            mul(graphs, steps)};
}
}  // namespace

EncodingAcquisition::EncodingAcquisition(encoding::detail::ContextView v, std::shared_ptr<const DeviceBuffer> gen,
                                         u64 n, std::shared_ptr<LastUse> last, cudaStream_t stream)
    : view(v), generation(std::move(gen)), number(n), m_last(std::move(last)), m_stream(stream)
{
}
EncodingAcquisition::EncodingAcquisition(EncodingAcquisition&& other) noexcept
    : view(other.view), generation(std::move(other.generation)), number(other.number),
      m_facts(std::move(other.m_facts)), m_owner(other.m_owner), m_last(std::move(other.m_last)), m_stream(other.m_stream)
{
}
EncodingAcquisition::~EncodingAcquisition()
{
    finish_use(m_last, m_stream);
    if (m_facts)
        for (const auto& axioms : m_facts->axioms)
            if (axioms.task)
                try { axioms.task->release(m_stream, m_last); }
                catch (...) { (void)cudaStreamSynchronize(m_stream); }
}

std::shared_ptr<DeviceEncodingContext> DeviceEncodingContext::upload(ContextPtr ctx,
    std::shared_ptr<const encoding::EncodingContext> host, cudaStream_t stream)
{
    if (!ctx || !host)
        throw std::invalid_argument("mymyr: encoding: a device context needs a device context and a host context");
    return create(std::move(ctx), std::move(host), false, stream);
}
std::shared_ptr<DeviceEncodingContext> DeviceEncodingContext::create(ContextPtr ctx,
    std::shared_ptr<const encoding::EncodingContext> host, bool facts, cudaStream_t stream)
{
    DeviceGuard guard(ctx->device());
    auto out = std::shared_ptr<DeviceEncodingContext>(new DeviceEncodingContext(std::move(ctx), host));
    out->copy(host, stream ? stream : out->m_ctx->stream(), facts);
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
void DeviceEncodingContext::copy(const std::shared_ptr<const encoding::EncodingContext>& host, cudaStream_t stream,
                                 bool with_facts)
{
    if (m_generation)
    {
        m_ready.synchronize();
        m_staging = PinnedLease();
    }
    // the instances a refresh adds are exported and uploaded; the others keep their uploads
    std::shared_ptr<detail::EncodingFacts> facts = with_facts ? context_facts(*host, m_ctx, m_facts.get()) : nullptr;
    const u64 context_bytes = host->arrays().bytes();
    const u64 records_at = round_up(context_bytes);
    const u64 record_bytes = facts && !facts->axioms.empty() ? axiom_records(*facts, nullptr, nullptr) : 0;
    auto staging = m_ctx->lease_pinned(1, record_bytes ? add(records_at, record_bytes) : context_bytes);
    std::memcpy(staging.data(0), host->arrays().block(), context_bytes);
    auto block = std::make_shared<DeviceBuffer>(m_ctx, context_bytes);
    if (record_bytes)
    {
        facts->axiom_block = std::make_shared<DeviceBuffer>(m_ctx, record_bytes);
        (void)axiom_records(*facts, static_cast<std::byte*>(staging.data(0)) + records_at,
                            static_cast<const std::byte*>(facts->axiom_block->data()));
    }
    Event allocated;
    allocated.record(m_ctx->stream());
    allocated.wait_on(stream);
    auto last = std::make_shared<LastUse>(m_ctx->device());
    block->track(last);
    if (record_bytes)
        facts->axiom_block->track(last);
    try
    {
        check(cudaMemcpyAsync(block->data(), staging.data(0), block->size(), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync (encoding context)");
        if (record_bytes)
            check(cudaMemcpyAsync(facts->axiom_block->data(), static_cast<std::byte*>(staging.data(0)) + records_at,
                                  record_bytes, cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync (encoding axioms)");
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
    m_facts = std::move(facts);
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
    copy(grown, stream ? stream : m_ctx->stream(), m_facts != nullptr);
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
        if (m_facts && m_facts->axiom_block)
            m_facts->axiom_block->track(last);
    }
    EncodingAcquisition held(m_view, m_block, m_generation, last, stream);
    held.m_facts = m_facts;
    held.m_owner = this;
    if (m_facts)
        for (const auto& axioms : m_facts->axioms)
            if (axioms.task)
                (void)axioms.task->acquire(stream);
    return held;
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

DeviceCollationBatch collation_batch(const encoding::FlatEncoder& encoder, const encoding::FlatBuffers& arrays,
                                     const encoding::FlatSizes* sizes, u64 graphs)
{
    const auto& schema = encoder.schema();
    DeviceCollationBatch batch;
    batch.arrays = arrays;
    batch.sizes = sizes;
    batch.graphs = graphs;
    batch.relations = schema.num_relations();
    batch.schema_fingerprint = schema.fingerprint();
    batch.table_fingerprint = encoder.table_fingerprint();
    batch.layout = schema.options().layout;
    batch.index = schema.options().index;
    batch.shared_statics = schema.shared_statics();
    batch.stored = output_arrays(schema);
    batch.max_goal_level = schema.options().max_goal_level;
    batch.history = schema.options().history;
    return batch;
}

OutputArrays output_arrays(const encoding::EncodingSchema& schema)
{
    OutputArrays arrays;
    arrays.entity_arrays = !schema.implicit_entity_arrays();
    arrays.counts = schema.emits_counts();
    arrays.goal_complete = schema.has_goal_complete();
    arrays.targets = schema.has_targets();
    arrays.history = schema.has_history();
    return arrays;
}

OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations, const OutputArrays& arrays)
{
    const u64 I = index == encoding::IndexType::I32 ? 4 : 8;
    const u64 G = shape.graphs, E = shape.entities, A = shape.args, T = shape.targets, S = shape.history,
              GR = mul(G, relations);
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
    place(out.history_offsets, arrays.history ? mul(I, add(G, 1)) : 0);
    place(out.history_entity, arrays.history ? mul(I, S) : 0);
    place(out.history_dt, arrays.history ? mul(4, S) : 0);
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
    b.history_offsets = at(layout.history_offsets, a.history);
    b.history_entity = at(layout.history_entity, a.history);
    b.history_dt = static_cast<i32*>(at(layout.history_dt, a.history));
    out.capacities = shape;
    out.sizes = static_cast<encoding::FlatSizes*>(at(layout.sizes, true));
    return out;
}

namespace detail
{
u64 ScratchLayout::history_at(u64 graphs) const
{
    if (!history)
        return 0;
    if (!relation_major)
        return mul(4, add(graphs, 1));
    return add(add(mul(2, add(graphs, 1)), mul(graphs, relations)), 1);
}
u64 ScratchLayout::scanned(u64 graphs) const
{
    return history ? add(history_at(graphs), add(graphs, 1))
                   : (relation_major ? add(add(mul(2, add(graphs, 1)), mul(graphs, relations)), 1)
                                     : mul(4, add(graphs, 1)));
}
ScratchLayout scratch_layout(const BatchShape& shape, const GraphBounds& bounds, encoding::Layout layout, u64 relations)
{
    ScratchLayout out;
    out.relation_major = layout == encoding::Layout::Relation;
    out.relations = relations;
    out.history = bounds.history;
    const u64 G = shape.graphs;
    // the largest value of the scan: the sum of the totals of every row (the argument row only graph-major)
    u64 scanned = add(add(mul(G, bounds.entities), mul(G, bounds.targets)), mul(G, bounds.tuples));
    if (!out.relation_major)
        scanned = add(scanned, mul(G, bounds.args));
    // the steps of a history are not bounded by the schema
    out.wide = bounds.history || scanned > std::numeric_limits<u32>::max();
    out.scan_length = out.scanned(G);
    out.flags = 0;
    out.totals = k_flag_bytes;
    out.cell_counts = round_up(add(k_flag_bytes, mul(out.wide ? 8 : 4, out.scan_length)));
    out.cell_stride = bounds.cells;
    out.cell_entries = out.relation_major ? 0 : mul(G, bounds.cells);
    // the levels of the subgoal layers of a graph: the plain mask of each level, sign and slot kind, over the instance's
    // words (and its derived words)
    out.level_stride = mul(mul(bounds.levels, 2), add(bounds.state_words, bounds.derived_words));
    out.level_storage = add(out.cell_counts, round_up(mul(4, out.cell_entries)));
    out.scan_temp = add(out.level_storage, round_up(mul(8, mul(G, out.level_stride))));
    out.scan_temp_bytes = totals_scan_temp_bytes(out.scan_length, out.wide);
    out.bytes = add(out.scan_temp, round_up(out.scan_temp_bytes));
    return out;
}

ScratchLayout encoding_scratch_layout(const BatchShape& shape, const EncodingFacts& facts, encoding::Layout layout,
                                      u64 relations)
{
    auto out = scratch_layout(shape, facts.bounds, layout, relations);
    if (!facts.axioms.empty())
    {
        out.derived = out.bytes;
        out.axiom_views = add(out.derived, round_up(mul(mul(shape.graphs, facts.derived_words), sizeof(u64))));
        out.bytes = add(out.axiom_views, round_up(mul(mul(shape.graphs, facts.axiom_view_words), sizeof(u64))));
    }
    return out;
}

u64 required_bytes(const ScratchLayout& layout, bool evaluate) noexcept
{
    return evaluate || !layout.derived ? layout.bytes : layout.derived;
}

ScratchView scratch_view(const ScratchLayout& layout, void* block) noexcept
{
    auto* base = static_cast<std::byte*>(block);
    ScratchView out;
    out.flags = reinterpret_cast<u32*>(base + layout.flags);
    out.totals = base + layout.totals;
    out.cell_counts = reinterpret_cast<u32*>(base + layout.cell_counts);
    out.cell_stride = layout.cell_stride;
    out.level_storage = reinterpret_cast<u64*>(base + layout.level_storage);
    out.level_stride = layout.level_stride;
    out.scan_temp = base + layout.scan_temp;
    out.scan_temp_bytes = layout.scan_temp_bytes;
    out.wide = layout.wide;
    return out;
}

}

cudaStream_t EncodingWorkspace::stream() const noexcept { return m_state->context.stream(); }
const encoding::detail::ContextView& EncodingWorkspace::view() const noexcept { return m_state->context.view; }
u64 EncodingWorkspace::generation() const noexcept { return m_state->context.number; }
BatchShape EncodingWorkspace::shape() const noexcept { return m_state->shape; }
u64 EncodingWorkspace::scratch_bytes() const noexcept { return m_state->layout.bytes; }

struct DeviceEncoder::Workspace
{
    BatchShape shape;
    detail::ScratchLayout layout;
    std::shared_ptr<DeviceBuffer> scratch;
    std::shared_ptr<LastUse> last;
    Event ready;
    std::mutex mutex;
};

void DeviceEncoder::check_schema(const encoding::EncodingSchema& schema)
{
    const auto& o = schema.options();
    auto refuse = [](const char* what) { throw std::invalid_argument(std::string("mymyr: encoding: the device does not encode ") + what); };
    if (o.targets.contains(encoding::TargetSource::Action)) refuse("action targets");
    if (schema.has_numeric()) refuse("numeric relations");
    // added_satisfied and added_unsatisfied need a delta horizon (FlatOptions): refused with it
    if (o.horizon != encoding::HorizonOptions{}) refuse("horizons");
    if (o.outputs != encoding::Outputs{encoding::Output::Relations}) refuse("Simple");
    if (o.statics == encoding::Statics::Shared) refuse("shared statics (statics='shared')");
}

DeviceEncoder::DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder)
    : m_ctx(std::move(ctx)), m_encoder(std::move(encoder))
{
    if (!m_ctx || !m_encoder)
        throw std::invalid_argument("mymyr: encoding: a device encoder needs a device context and an encoder");
    m_schema = m_encoder->schema_ptr();
    check_schema(*m_schema);
    if (m_encoder->table().numeric())
        throw std::invalid_argument("mymyr: encoding: the device does not encode numeric relations (the table has numeric state words)");
    m_context = DeviceEncodingContext::create(m_ctx, m_encoder->context_ptr(), true, nullptr);
    m_bounds = m_context->m_facts->bounds;
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
BatchShape DeviceEncoder::worst_case(u64 graphs, u64 history_steps) const
{
    return worst_shape(graphs, bounds(), history_steps_of(schema(), history_steps, true));
}
OutputLayout DeviceEncoder::output_layout(const BatchShape& shape) const
{
    return cuda::output_layout(shape, options().index, schema().num_relations(), output_arrays(schema()));
}
u64 DeviceEncoder::scratch_bytes(const BatchShape& shape, DerivedWords derived) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    return detail::required_bytes(
        detail::encoding_scratch_layout(shape, *m_context->m_facts, options().layout, schema().num_relations()),
        derived == DerivedWords::Evaluated);
}
u64 DeviceEncoder::scratch_bytes(const encoding::FlatEncoder& encoder, const BatchShape& shape, DerivedWords derived)
{
    const auto facts = context_facts(encoder.context());
    return detail::required_bytes(detail::encoding_scratch_layout(shape, *facts, encoder.schema().options().layout,
                                                                  encoder.schema().num_relations()),
                                  derived == DerivedWords::Evaluated);
}
u64 DeviceEncoder::scratch_bytes(const EncodingAcquisition& context, const BatchShape& shape, DerivedWords derived) const
{
    if (context.m_owner != m_context.get() || !context.m_facts)
        throw std::invalid_argument("mymyr: encoding: acquire the context from this device encoder");
    DeviceGuard guard(m_ctx->device());
    return detail::required_bytes(
        detail::encoding_scratch_layout(shape, *context.m_facts, options().layout, schema().num_relations()),
        derived == DerivedWords::Evaluated);
}
bool DeviceEncoder::needs_derived(const encoding::FlatEncoder& encoder, bool goal_masks, bool derived_goal_masks,
                                  bool derived_layer_masks)
{
    const auto facts = context_facts(encoder.context());
    // the evaluation mode of an encode (as in run): the goal mode, plus 3 with derived subgoal masks
    const u32 mode = (goal_masks ? (derived_goal_masks ? 2 : 1) : 0) + (derived_layer_masks ? 3 : 0);
    return std::ranges::any_of(facts->axioms, [&](const detail::EncodingFacts::Axioms& axioms) {
        return ((axioms.modes >> mode) & 1u) && !axioms.refusal.empty();
    });
}
u64 DeviceEncoder::bytes_needed(const BatchShape& shape) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto output = output_layout(shape);
    return add(detail::encoding_scratch_layout(shape, *m_context->m_facts, options().layout, schema().num_relations()).bytes, output.bytes);
}
u64 DeviceEncoder::bytes_needed(u64 graphs, u64 history_steps) const
{
    std::shared_lock lock(m_state_mutex);
    DeviceGuard guard(m_ctx->device());
    const auto shape = worst_shape(graphs, m_bounds, history_steps_of(schema(), history_steps, true));
    return add(detail::encoding_scratch_layout(shape, *m_context->m_facts, options().layout, schema().num_relations()).bytes,
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
                          std::max(shape.args, work.shape.args), std::max(shape.targets, work.shape.targets),
                          std::max(shape.history, work.shape.history)};
    (void)output_layout(want);
    const auto layout = detail::encoding_scratch_layout(want, *m_context->m_facts, options().layout, schema().num_relations());
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
    reserve_locked(worst_shape(graphs, m_bounds, history_steps_of(schema(), 0, false)), stream);
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
    const auto needed = detail::encoding_scratch_layout(work->shape, *m_context->m_facts, options().layout, schema().num_relations());
    if (!work->scratch || needed != work->layout)
        throw std::invalid_argument("mymyr: encoding: reserve scratch again after a failed refresh");
    work->ready.wait_on(stream);
    return EncodingWorkspace(std::make_shared<detail::EncodingWorkspaceState>(
        work->scratch, m_context->acquire(stream, work->last), work->shape, work->layout));
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
    const bool copied = m_context->refresh(grown->context_ptr(), stream);
    m_encoder = std::move(grown);
    m_bounds = m_context->facts()->bounds;
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

/// The count kernel's lanes per graph and the fill classes (bit 0: one thread per graph, bit 1: a warp per graph) that
/// the first `instances` instances of `widths` launch with. Throws std::invalid_argument when the shared memory of a
/// block does not hold what a kernel needs.
struct LaunchPlan
{
    u32 lanes;
    u32 fill_mask;
};
LaunchPlan plan_launch(u32 virtual_nodes, u64 relations, bool relation_major, bool index64,
                       const std::vector<encoding::detail::GraphWidth>& widths, u64 instances, const LaunchConfig& launch)
{
    u32 lanes = launch.count_lanes ? launch.count_lanes : 4;
    if (!launch.count_lanes)
        for (const u32 candidate : {4u, 8u, 32u})
            if (detail::count_shared_required(virtual_nodes, candidate) <= detail::k_shared_limit)
            {
                lanes = candidate;
                break;
            }
    if (detail::count_shared_required(virtual_nodes, lanes) > detail::k_shared_limit)
        invalid("the schema has too many predicate virtual nodes for the count kernel");
    auto fits = [&](bool warp)
    {
        return detail::fill_shared_required(virtual_nodes, static_cast<u32>(relations), warp, relation_major, index64) <=
               detail::k_shared_limit;
    };
    u32 fill_mask = 0;
    for (u64 instance = 0; instance < std::min<u64>(instances, widths.size()); ++instance)
    {
        bool warp = launch.fill == FillMode::Warp ||
                    (launch.fill == FillMode::Auto && detail::warp_fill_for(widths[instance], relation_major));
        if (!fits(warp) && launch.fill == FillMode::Auto)
            warp = true;
        if (!fits(warp))
            invalid("the schema has too many predicate virtual nodes or relations for the fill kernel");
        fill_mask |= warp ? 2u : 1u;
    }
    return {lanes, fill_mask ? fill_mask : 1u};
}

}  // namespace

void DeviceEncoder::check_encoder(const encoding::FlatEncoder& encoder)
{
    check_schema(encoder.schema());
    if (encoder.table().numeric())
        throw std::invalid_argument("mymyr: encoding: the device does not encode numeric relations (the table has numeric state words)");
    if (device_count() == 0)  // the shared memory of the kernels is read from the device
        return;
    const auto facts = context_facts(encoder.context());
    const auto& options = encoder.schema().options();
    (void)plan_launch(encoder.context().view().num_virtual, encoder.schema().num_relations(),
                      options.layout == encoding::Layout::Relation, options.index == encoding::IndexType::I64, facts->widths,
                      facts->widths.size(), {});
}

void DeviceEncoder::run(Phase phase, const detail::EncodingExecution& execution, const DeviceEncodeInput& in,
                        const DeviceOutput& out, const LaunchConfig& launch) const
{
    const auto& [context, block, bytes, scratch_shape, layout, stream] = execution;
    DeviceGuard guard(m_ctx->device());
    if (context.m_owner != m_context.get() || !context.m_facts)
        invalid("acquire the context from this device encoder");
    const auto& facts = *context.m_facts;
    const encoding::EncodingSchema& schema = *m_schema;
    const encoding::FlatOptions& opts = schema.options();
    const u64 graphs = in.graphs;
    const u64 R = schema.num_relations();
    const bool relation_major = opts.layout == encoding::Layout::Relation;
    const encoding::LayerInput& sub = in.subgoals;
    const encoding::HistoryInput& hist = in.history;
    const u32 goal_mode = in.goal_pos ? (in.goal_derived_pos ? 2 : 1) : 0;
    const bool layer_derived = sub.layers != 0 && sub.derived_pos;
    const u32 mode = goal_mode + (layer_derived ? 3 : 0);
    const bool evaluate =
        graphs && !in.derived && (facts.reads_derived[goal_mode] || (layer_derived && facts.reads_derived_layers));
    const bool live = in.graphs_device || in.row_start || in.dropped || out.graphs;

    if (!block || reinterpret_cast<std::uintptr_t>(block) % 256 || bytes < detail::required_bytes(layout, false))
        invalid("the scratch is null, too small, or not aligned to 256 bytes");
    if (evaluate && (!layout.derived || !layout.axiom_views || bytes < layout.bytes))
        invalid("the scratch has no room to evaluate the derived atoms the graphs read: pass derived words, or size the "
                "scratch with DerivedWords::Evaluated");

    // what can be checked without reading device memory
    if (graphs > scratch_shape.graphs)
        invalid("the batch has " + std::to_string(graphs) + " graphs, more than the " +
                std::to_string(scratch_shape.graphs) + " the workspace was reserved for");
    if (graphs > out.capacities.graphs)
        invalid("the batch has more graphs than the output capacity");
    if (layout.relation_major != relation_major || layout.relations != R)
        invalid("the workspace does not belong to this encoder");
    if (in.row_start && !in.row_capacity)
        invalid("a circular row view needs a positive capacity");
    if (live && launch.order)
        invalid("a launch order needs the graph count on the host: it cannot order a live row count");
    if (graphs && !in.states)
        invalid("the state buffer is null");
    if (in.state_stride < in.state_words)
        invalid("the state row stride is smaller than its width");
    if (in.derived && in.derived_stride < facts.derived_words)
        invalid("the derived row stride is smaller than the derived words of an instance");
    if (in.goal_pos && !schema.has_goal_complete())
        invalid("goals are given, but the schema has no goal relations");
    if ((in.goal_pos == nullptr) != (in.goal_neg == nullptr))
        invalid("the goal needs both goal_pos and goal_neg");
    if ((in.goal_derived_pos == nullptr) != (in.goal_derived_neg == nullptr))
        invalid("the derived goal needs both goal_derived_pos and goal_derived_neg");
    if (in.goal_pos && in.goal_stride && in.goal_stride < in.goal_words)
        invalid("the goal row stride is smaller than its width");
    if (in.goal_derived_pos && in.goal_derived_stride && in.goal_derived_stride < in.goal_derived_words)
        invalid("the derived goal row stride is smaller than its width");
    if (in.goal_derived_pos && !in.goal_pos)
        invalid("the derived goal masks need the goal masks over the fluent slots");
    // the masks of an empty batch may be null (an empty tensor has no data pointer), but not only one of them
    const auto check_layers = [graphs](const encoding::LayerInput& l, const char* what, bool rows_share)
    {
        if (l.layers == 0)
            return;
        if ((l.pos == nullptr) != (l.neg == nullptr) || (graphs != 0 && !l.pos))
            invalid(std::string(what) + " need both their positive and negative masks");
        if ((l.derived_pos == nullptr) != (l.derived_neg == nullptr))
            invalid(std::string(what) + " need both their positive and negative derived masks");
        if (l.stride < u64{l.layers} * l.words && !(rows_share && l.stride == 0))
            invalid(std::string("the row stride of the ") + what + " is smaller than their layers");
        if (l.derived_pos && l.derived_stride < u64{l.layers} * l.derived_words &&
            !(rows_share && l.derived_stride == 0))
            invalid(std::string("the derived row stride of the ") + what + " is smaller than their layers");
    };
    if (sub.layers != 0)
    {
        if (!schema.has_subgoals())
            invalid("subgoal layers are given, but the schema has no subgoal relations");
        if (sub.layers > opts.max_goal_level)
            invalid(std::to_string(sub.layers) + " subgoal layers are given, the schema has " +
                    std::to_string(opts.max_goal_level) + " levels");
        check_layers(sub, "subgoal layers", true);
    }
    if (hist.masks.layers != 0 || hist.count)
    {
        if (!schema.has_history())
            invalid("history steps are given, but the schema has no history relations");
        if (hist.masks.layers != 0 && graphs != 0 && !hist.dt)
            invalid("the history steps need their dt");
        if (hist.masks.layers == 0 && (hist.masks.pos || hist.masks.neg || hist.dt))
            invalid("history masks or dt are given without steps");
        check_layers(hist.masks, "history steps", false);
    }
    if (live && (layout.history || sub.layers != 0))
        invalid("a live row view carries no subgoal layers or history steps");
    // the rows of a batch without task ids are instance 0's: a refusal of its axioms refuses the call; otherwise the
    // rows of a refused instance are invalid rows (the input flag)
    if (evaluate && !in.task_ids && !facts.axioms.empty() && ((facts.axioms[0].modes >> mode) & 1u) &&
        !facts.axioms[0].refusal.empty())
        invalid(facts.axioms[0].refusal);
    const OutputArrays arrays = output_arrays(schema);
    const encoding::FlatBuffers& a = out.arrays;
    if (!out.sizes)
        invalid("the output needs sizes");
    if (phase != Phase::Count)
    {
        if ((graphs && (!a.task_ids || !a.num_objects)) || !a.entity_offsets || !a.relation_offsets ||
            !a.relation_arg_offsets)
            invalid("the output needs task_ids, num_objects, entity_offsets, relation_offsets, relation_arg_offsets and sizes");
        // an array without entries may be null, so a batch of rows without objects needs no entity arrays
        const bool entity_given = (a.entity_role != nullptr) || (a.entity_ref != nullptr);
        if (!arrays.entity_arrays ? entity_given
                                  : (out.capacities.entities && (!a.entity_role || !a.entity_ref)))
            invalid("entity_role and entity_ref must be given iff the schema stores entity arrays");
        if (arrays.counts ? (graphs && R && !a.relation_counts) : a.relation_counts != nullptr)
            invalid("relation_counts must be given iff the schema stores counts");
        if (arrays.goal_complete ? (graphs && !a.goal_complete) : a.goal_complete != nullptr)
            invalid("goal_complete must be given iff the schema has goal relations");
        if ((a.target_offsets != nullptr) != arrays.targets ||
            (!arrays.targets && (a.target_entity != nullptr || a.target_group != nullptr)))
            invalid("the target arrays must be given iff the schema has target sources");
        if (out.capacities.args && !a.relation_args)
            invalid("relation_args is null");
        if (arrays.targets && out.capacities.targets && (!a.target_entity || !a.target_group))
            invalid("the target arrays are null");
        if ((a.history_offsets != nullptr) != arrays.history ||
            (!arrays.history && (a.history_entity != nullptr || a.history_dt != nullptr)))
            invalid("the history arrays must be given iff the schema has history");
        if (arrays.history && out.capacities.history && (!a.history_entity || !a.history_dt))
            invalid("the history arrays are null");
    }
    if (launch.count_lanes != 0 && launch.count_lanes != 4 && launch.count_lanes != 8 && launch.count_lanes != 32)
        invalid("the count kernel runs 4, 8 or 32 lanes per graph");

    // how it is launched
    const u64 launched_instances = in.task_ids ? facts.widths.size() : std::min<usize>(1, facts.widths.size());
    const auto [lanes, fill_mask] =
        plan_launch(context.view.num_virtual, R, relation_major, opts.index == encoding::IndexType::I64, facts.widths,
                    launched_instances, launch);

    const detail::ScratchView scratch = detail::scratch_view(layout, block);
    const encoding::IndexType index = opts.index;
    const u64 limit = index == encoding::IndexType::I64 ? static_cast<u64>(std::numeric_limits<i64>::max())
                                                       : static_cast<u64>(std::numeric_limits<i32>::max());
    detail::EncodeParams p;
    p.context = context.view;
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
    p.layers = {sub, hist, opts.history_max_steps.value_or(0)};
    p.layered = p.layers.layered();
    p.level_storage = scratch.level_storage;
    p.level_stride = scratch.level_stride;
    p.order = launch.order;
    p.sizes = out.sizes;
    if (phase == Phase::Count)
    {
        // counting reports the sizes against no capacity
        constexpr u64 unbounded = std::numeric_limits<u64>::max();
        p.capacities = {graphs, unbounded, unbounded, unbounded, unbounded};
    }
    else
    {
        p.out = out.arrays;
        p.capacities = out.capacities;
    }
    p.index64 = index == encoding::IndexType::I64;
    p.zero_unused = out.zero_unused;
    p.relation_major = relation_major;
    if (mul(graphs, R) >= limit || graphs >= limit)
        p.static_overflow |= encoding::k_overflow_index;
    p.totals = scratch.totals;
    p.wide = scratch.wide;
    p.segments = mul(2, add(graphs, 1));
    p.history_at = layout.history_at(graphs);
    p.flags = scratch.flags;
    p.cell_counts = scratch.cell_counts;
    p.cell_stride = scratch.cell_stride;

    detail::LaunchShape shape;
    shape.lanes = lanes;
    shape.fill_mask = fill_mask;
    shape.scan_length = layout.scanned(graphs);
    shape.scan_temp = scratch.scan_temp;
    shape.scan_temp_bytes = scratch.scan_temp_bytes;
    shape.zero_begin = scratch.flags;
    shape.live = live;
    // An unassigned graph in an invalid launch order contributes zero totals. A live count writes all of its totals.
    const u64 entry = layout.wide ? 8 : 4;
    shape.zero_bytes = live ? k_flag_bytes : add(k_flag_bytes, mul(entry, shape.scan_length));
    if (live)
    {
        p.graphs_device = in.graphs_device;
        p.row_start = in.row_start;
        p.row_capacity = in.row_capacity;
        p.rows_overflow = in.dropped;
        p.output_graphs = out.graphs;
        p.scan_state = scratch.scan_temp;
        p.scan_tiles = detail::live_scan_tiles(shape.scan_length);
    }
    if (evaluate)
    {
        auto* base = static_cast<std::byte*>(block);
        p.derived = reinterpret_cast<u64*>(base + layout.derived);
        p.derived_stride = facts.derived_words;
        p.derived_logical = true;
        if (phase != Phase::Fill && !facts.class_ow[mode].empty())
        {
            // the evaluation flags the rows of instances it cannot evaluate: the clear goes first
            check(detail::launch_clear(shape, stream), "encoding kernels (clear)");
            shape.cleared = true;
            lifted::EncodingAxioms ax;
            ax.instances = static_cast<const lifted::EncodingAxiomInstance*>(facts.axiom_block->data());
            ax.count = static_cast<u32>(facts.axioms.size());
            ax.states = in.states;
            ax.stride = in.state_stride;
            ax.words = in.state_words;
            ax.rows = graphs;
            ax.task_ids = in.task_ids;
            ax.mode = mode;
            ax.views = reinterpret_cast<u64*>(base + layout.axiom_views);
            ax.view_words = facts.axiom_view_words;
            ax.derived = reinterpret_cast<u64*>(base + layout.derived);
            ax.derived_words = facts.derived_words;
            ax.invalid = scratch.flags;
            ax.live = in.graphs_device;
            ax.row_start = in.row_start;
            ax.row_capacity = in.row_capacity;
            ax.class_ow = facts.class_ow[mode].data();
            ax.class_fc = facts.class_fc[mode].data();
            ax.launches = static_cast<u32>(facts.class_ow[mode].size());
            check(lifted::launch_encoding_axioms(ax, stream), "encoding axiom kernels");
        }
    }
    switch (phase)
    {
    case Phase::Whole: check(detail::launch_encode(p, shape, stream), "encoding kernels"); break;
    case Phase::Count: check(detail::launch_count_sizes(p, shape, stream), "encoding kernels (count)"); break;
    case Phase::Fill: check(detail::launch_fill(p, shape, stream), "encoding kernels (fill)"); break;
    }
}
void DeviceEncoder::encode(const EncodingWorkspace& work, const DeviceEncodeInput& in, const DeviceOutput& out,
                           const LaunchConfig& launch) const
{
    const auto& s = *work.m_state;
    run(Phase::Whole, {s.context, s.scratch->data(), s.layout.bytes, s.shape, s.layout, s.context.stream()}, in, out,
        launch);
}
void DeviceEncoder::count(const EncodingWorkspace& work, const DeviceEncodeInput& in, const DeviceOutput& out,
                          const LaunchConfig& launch) const
{
    const auto& s = *work.m_state;
    run(Phase::Count, {s.context, s.scratch->data(), s.layout.bytes, s.shape, s.layout, s.context.stream()}, in, out,
        launch);
}
void DeviceEncoder::fill(const EncodingWorkspace& work, const DeviceEncodeInput& in, const DeviceOutput& out,
                         const LaunchConfig& launch) const
{
    const auto& s = *work.m_state;
    run(Phase::Fill, {s.context, s.scratch->data(), s.layout.bytes, s.shape, s.layout, s.context.stream()}, in, out,
        launch);
}

void DeviceEncoder::encode(const DeviceEncodeInput& in, const DeviceOutput& out, cudaStream_t stream,
                           const LaunchConfig& launch) const
{
    const EncodingWorkspace work = acquire(stream);
    encode(work, in, out, launch);
}
void DeviceEncoder::encode(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
                          const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch) const
{
    DeviceGuard guard(m_ctx->device());
    if (context.m_owner != m_context.get() || !context.m_facts)
        invalid("acquire the context from this device encoder");
    const auto layout = detail::encoding_scratch_layout(shape, *context.m_facts, options().layout, schema().num_relations());
    run(Phase::Whole, {context, scratch, bytes, shape, layout, context.stream()}, in, out, launch);
}
void DeviceEncoder::encode(const EncodingAcquisition& context, cudaStream_t stream, void* scratch, u64 bytes,
                           const BatchShape& shape, const DeviceEncodeInput& in, const DeviceOutput& out,
                           const LaunchConfig& launch) const
{
    DeviceGuard guard(m_ctx->device());
    if (context.m_owner != m_context.get() || !context.m_facts)
        invalid("acquire the context from this device encoder");
    const auto layout = detail::encoding_scratch_layout(shape, *context.m_facts, options().layout, schema().num_relations());
    run(Phase::Whole, {context, scratch, bytes, shape, layout, stream}, in, out, launch);
}
void DeviceEncoder::count(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
                          const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch) const
{
    DeviceGuard guard(m_ctx->device());
    if (context.m_owner != m_context.get() || !context.m_facts)
        invalid("acquire the context from this device encoder");
    const auto layout = detail::encoding_scratch_layout(shape, *context.m_facts, options().layout, schema().num_relations());
    run(Phase::Count, {context, scratch, bytes, shape, layout, context.stream()}, in, out, launch);
}
void DeviceEncoder::fill(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
                          const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch) const
{
    DeviceGuard guard(m_ctx->device());
    if (context.m_owner != m_context.get() || !context.m_facts)
        invalid("acquire the context from this device encoder");
    const auto layout = detail::encoding_scratch_layout(shape, *context.m_facts, options().layout, schema().num_relations());
    run(Phase::Fill, {context, scratch, bytes, shape, layout, context.stream()}, in, out, launch);
}
}  // namespace mymyr::cuda
