// DeviceEncodingContext and DeviceEncoder (include/mymyr/cuda/encoding.hpp).

#include "mymyr/cuda/encoding.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mymyr::cuda
{
namespace detail
{
u64 scan_temp_bytes(u64 n);  // encoding.cu
}

namespace
{
using encoding::detail::Cell;

constexpr u64 round_up(u64 x) noexcept { return (x + 63) & ~u64{63}; }

[[noreturn]] void not_an_extension(const std::string& why)
{
    throw std::invalid_argument("mymyr: encoding: the grown context is not an extension of the uploaded one: " + why);
}

/// Throws unless `grown` has the schema of `current`, the instances of `current` at the same rows of every array, and
/// at least as many instances.
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
            not_an_extension("the array " + a.name + " is missing or shorter");
        if (std::memcmp(current.arrays().data(a), grown.arrays().data(*b), a.elements() * rl::dtype_bytes(a.dtype)) != 0)
            not_an_extension("the array " + a.name + " differs in the rows of the uploaded instances");
    }
}

[[noreturn]] void derived_refused(u32 instance)
{
    throw std::invalid_argument("mymyr: encoding: the device does not encode derived relations (instance " +
                                std::to_string(instance) + " has derived atoms): set include_derived = false");
}
}  // namespace

// ------------------------------------------------------------------------------------------------ context

std::shared_ptr<DeviceEncodingContext> DeviceEncodingContext::upload(ContextPtr ctx,
                                                                      std::shared_ptr<const encoding::EncodingContext> host,
                                                                      cudaStream_t stream)
{
    if (!ctx || !host)
        throw std::invalid_argument("mymyr: encoding: a device context needs a device context and a host context");
    DeviceGuard g(ctx->device());
    std::shared_ptr<DeviceEncodingContext> out(new DeviceEncodingContext(std::move(ctx), std::move(host)));
    out->copy(stream ? stream : out->m_ctx->stream());
    return out;
}

DeviceEncodingContext::DeviceEncodingContext(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host)
    : m_ctx(std::move(ctx)), m_host(std::move(host))
{
}

DeviceEncodingContext::~DeviceEncodingContext()
{
    // the staging buffer is the source of the copy in flight; the block frees itself after the work of its users
    if (m_uploads)
        cudaEventSynchronize(m_ready.get());
}

void DeviceEncodingContext::copy(cudaStream_t stream)
{
    const rl::ArrayBundle& arrays = m_host->arrays();
    const u64 bytes = arrays.bytes();
    if (m_uploads)
        m_ready.synchronize();  // the staging buffer of the previous upload is free once its copy ran
    PinnedBuffer staging(bytes);
    std::memcpy(staging.data(), arrays.block(), bytes);
    DeviceBuffer block(m_ctx, bytes, stream);
    check(cudaMemcpyAsync(block.data(), staging.data(), bytes, cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync (encoding context)");
    m_ready.record(stream);
    m_block = std::move(block);  // the previous block frees after the work of the streams that used it
    m_staging = std::move(staging);
    ++m_uploads;
}

bool DeviceEncodingContext::refresh(std::shared_ptr<const encoding::EncodingContext> grown, cudaStream_t stream)
{
    if (!grown)
        throw std::invalid_argument("mymyr: encoding: a refresh needs a host context");
    std::lock_guard lock(m_mutex);
    check_extension(*m_host, *grown);
    if (grown->table().size() == m_host->table().size())
        return false;
    DeviceGuard g(m_ctx->device());
    m_host = std::move(grown);
    copy(stream ? stream : m_ctx->stream());
    return true;
}

encoding::detail::ContextView DeviceEncodingContext::view() const
{
    std::lock_guard lock(m_mutex);
    return m_host->view(static_cast<const std::byte*>(m_block.data()));
}

void DeviceEncodingContext::acquire(cudaStream_t s) const
{
    std::lock_guard lock(m_mutex);
    if (s != m_block.stream())
        m_ready.wait_on(s);
    m_block.record_stream(s);
    if (m_staging.data() && m_ready.done())
        m_staging = PinnedBuffer();
}

std::vector<std::byte> DeviceEncodingContext::download(cudaStream_t s) const
{
    std::lock_guard lock(m_mutex);
    DeviceGuard g(m_ctx->device());
    if (!s)
        s = m_ctx->stream();
    m_ready.wait_on(s);
    std::vector<std::byte> out(m_block.size());
    PinnedBuffer staging(out.size());
    check(cudaMemcpyAsync(staging.data(), m_block.data(), out.size(), cudaMemcpyDeviceToHost, s),
          "cudaMemcpyAsync (encoding context)");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    std::memcpy(out.data(), staging.data(), out.size());
    return out;
}

// ------------------------------------------------------------------------------------------------ sizes

OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations) noexcept
{
    const u64 I = index == encoding::IndexType::I32 ? 4 : 8;
    const u64 G = shape.graphs, E = shape.entities, A = shape.args, R = relations;
    OutputLayout out;
    u64 at = 0;
    auto place = [&](u64& offset, u64 bytes)
    {
        offset = at;
        at += round_up(bytes);
    };
    place(out.task_ids, 4 * G);
    place(out.num_objects, I * G);
    place(out.entity_offsets, I * (G + 1));
    place(out.entity_role, E);
    place(out.entity_ref, I * E);
    place(out.relation_counts, I * G * R);
    place(out.relation_offsets, I * (G * R + 1));
    place(out.relation_arg_offsets, I * (G * R + 1));
    place(out.relation_args, I * A);
    out.bytes = at;
    return out;
}

u64 scratch_bytes(u64 graphs, u64 max_cells)
{
    return round_up(24 * graphs) + round_up(24 * graphs) + round_up(4 * graphs * max_cells) +
           round_up(detail::scan_temp_bytes(graphs + 1));
}

// ------------------------------------------------------------------------------------------------ encoder

DeviceEncoder::DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder)
    : m_ctx(std::move(ctx)), m_encoder(std::move(encoder))
{
    if (!m_ctx || !m_encoder)
        throw std::invalid_argument("mymyr: encoding: a device encoder needs a device context and an encoder");
    const encoding::EncodingContext& host = m_encoder->context();
    for (u32 i = 0; i < host.table().size(); ++i)
        if (host.needs_derived(i))
            derived_refused(i);
    m_context = DeviceEncodingContext::upload(m_ctx, m_encoder->context_ptr());
    set_bounds();
}

void DeviceEncoder::set_bounds()
{
    const encoding::EncodingContext& host = m_encoder->context();
    const encoding::detail::ContextView v = host.view();
    GraphBounds bounds;
    for (u32 i = 0; i < host.table().size(); ++i)
    {
        const encoding::detail::InstanceView inst = encoding::detail::instance_view(v, i);
        GraphBounds b;
        b.cells = inst.num_cells;
        bool nullary = false;
        for (u32 c = 0; c < inst.num_cells; ++c)
        {
            const Cell& cell = inst.cells[c];
            b.tuples += cell.b - cell.a;
            b.args += u64{cell.b - cell.a} * cell.arity;
            nullary = nullary || cell.logical_arity == 0;
        }
        b.entities = inst.num_objects + (nullary ? 1 : 0);
        bounds.entities = std::max(bounds.entities, b.entities);
        bounds.tuples = std::max(bounds.tuples, b.tuples);
        bounds.args = std::max(bounds.args, b.args);
        bounds.cells = std::max(bounds.cells, b.cells);
    }
    m_bounds = bounds;
}

BatchShape DeviceEncoder::worst_case(u64 graphs) const noexcept
{
    return {graphs, graphs * m_bounds.entities, graphs * m_bounds.args};
}

u64 DeviceEncoder::bytes_needed(const BatchShape& shape) const
{
    return scratch_bytes(shape.graphs, m_bounds.cells) + output_layout(shape, options().index, schema().num_relations()).bytes;
}

u64 DeviceEncoder::bytes_needed(u64 graphs) const { return bytes_needed(worst_case(graphs)); }

OutputLayout DeviceEncoder::reserved_layout() const noexcept
{
    return output_layout(m_reserved, options().index, schema().num_relations());
}

void DeviceEncoder::reserve(const BatchShape& shape, cudaStream_t stream)
{
    std::lock_guard lock(m_mutex);
    if (m_output.data() && shape.graphs <= m_reserved.graphs && shape.entities <= m_reserved.entities &&
        shape.args <= m_reserved.args)
        return;
    const BatchShape want{std::max(shape.graphs, m_reserved.graphs), std::max(shape.entities, m_reserved.entities),
                          std::max(shape.args, m_reserved.args)};
    DeviceGuard g(m_ctx->device());
    if (!stream)
        stream = m_ctx->stream();
    DeviceBuffer scratch(m_ctx, scratch_bytes(want.graphs, m_bounds.cells), stream);
    DeviceBuffer output(m_ctx, output_layout(want, options().index, schema().num_relations()).bytes, stream);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    m_scratch = std::move(scratch);
    m_output = std::move(output);
    m_reserved = want;
}

void DeviceEncoder::reserve(u64 max_graphs, cudaStream_t stream) { reserve(worst_case(max_graphs), stream); }

void DeviceEncoder::acquire(cudaStream_t s) const
{
    m_context->acquire(s);
    std::lock_guard lock(m_mutex);
    if (m_output.data())
    {
        m_scratch.record_stream(s);
        m_output.record_stream(s);
    }
}

bool DeviceEncoder::refresh(std::shared_ptr<const encoding::FlatEncoder> grown, cudaStream_t stream)
{
    if (!grown)
        throw std::invalid_argument("mymyr: encoding: a refresh needs an encoder");
    if (grown->schema().options() != options())
        throw std::invalid_argument("mymyr: encoding: the grown encoder has other options");
    for (u32 i = 0; i < grown->table().size(); ++i)
        if (grown->context().needs_derived(i))
            derived_refused(i);
    const bool copied = m_context->refresh(grown->context_ptr(), stream);
    std::lock_guard lock(m_mutex);
    m_encoder = std::move(grown);
    set_bounds();
    return copied;
}
}  // namespace mymyr::cuda
