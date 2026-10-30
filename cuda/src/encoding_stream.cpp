#include "mymyr/cuda/encoding.hpp"
#include "mymyr/encoding/detail/lanes.hpp"
#include "encoding_rows.hpp"

#include <limits>
#include <optional>
#include <stdexcept>

namespace mymyr::cuda
{
void append_rows(const DeviceEncodeInput& in, const DeviceEncodeInput& ring, u64* status, cudaStream_t stream)
{
    if (!in.graphs) return;
    if (in.actions.offsets || in.horizon.node_offsets || ring.actions.offsets || ring.horizon.node_offsets)
        throw std::invalid_argument("mymyr: encoding: a device ring carries no expansion labels or horizon views");
    if (in.subgoals.layers || in.history.masks.layers || in.history.count || ring.subgoals.layers ||
        ring.history.masks.layers || ring.history.count)
        throw std::invalid_argument("mymyr: encoding: a device ring carries no subgoal layers or history steps");
    auto invalid = [] { throw std::invalid_argument("mymyr: encoding: appended rows must match the device ring's columns and widths"); };
    if (!status || !ring.graphs || !ring.states || !ring.task_ids || in.graphs_device || in.row_start || in.dropped ||
        !in.states || in.state_words != ring.state_words || in.numeric_words != ring.numeric_words ||
        in.state_stride < u64{in.state_words} + in.numeric_words ||
        ring.state_stride < u64{ring.state_words} + ring.numeric_words)
        invalid();
    if (bool(in.derived) != bool(ring.derived) || (in.derived && in.derived_stride < ring.derived_stride) ||
        bool(in.goal_pos) != bool(ring.goal_pos) || bool(in.goal_neg) != bool(ring.goal_neg) ||
        bool(in.goal_derived_pos) != bool(ring.goal_derived_pos) || bool(in.goal_derived_neg) != bool(ring.goal_derived_neg) ||
        (in.goal_pos && (in.goal_words != ring.goal_words || (in.goal_stride && in.goal_stride < in.goal_words))) ||
        (in.goal_derived_pos && (in.goal_derived_words != ring.goal_derived_words ||
                               (in.goal_derived_stride && in.goal_derived_stride < in.goal_derived_words))))
        invalid();
    if (const encoding::ConstantInput& c = in.goal_constants; c.rows)
    {
        if (!ring.goal_constants.rows || !in.goal_pos)
            throw std::invalid_argument("mymyr: encoding: the device ring holds no constant goal literals: construct it "
                                        "with room for them");
        if (c.width != ring.goal_constants.width || c.capacity > ring.goal_constants.capacity ||
            (c.stride != 0 && c.stride < u64{c.capacity} * c.width))
            throw std::invalid_argument(
                "mymyr: encoding: the constant goal literals of the appended rows are " + std::to_string(c.capacity) +
                " rows of " + std::to_string(c.width) + " columns, the device ring holds at most " +
                std::to_string(ring.goal_constants.capacity) + " of " + std::to_string(ring.goal_constants.width));
    }
    check(launch_append_rows(in, ring, status, stream), "append encoding rows");
}

void finish_rows(u64* status, u64 capacity, cudaStream_t stream)
{
    if (!status || !capacity)
        throw std::invalid_argument("mymyr: encoding: finish_rows needs ring status and a positive capacity");
    check(launch_clear_rows(status, capacity, false, stream), "finish encoding rows");
}

void check_stream_schema(const encoding::EncodingSchema& schema)
{
    const auto& o = schema.options();
    if (schema.has_horizon())
        throw std::invalid_argument("mymyr: encoding: a device stream carries no horizon nodes or edges");
    if (o.max_goal_level != 0 || o.history)
        throw std::invalid_argument("mymyr: encoding: a device stream carries no subgoal layers or history steps");
}

void clear_rows(u64* status, u64 capacity, cudaStream_t stream)
{
    if (!status || !capacity)
        throw std::invalid_argument("mymyr: encoding: clear_rows needs ring status and a positive capacity");
    check(launch_clear_rows(status, capacity, true, stream), "clear encoding rows");
}

struct DeviceRowStream::Impl
{
    static const ContextPtr& checked(const ContextPtr& ctx, const std::shared_ptr<const encoding::FlatEncoder>& host)
    {
        if (!host)
            throw std::invalid_argument("mymyr: encoding: a device stream needs an encoder");
        check_stream_schema(host->schema());
        return ctx;
    }

    ContextPtr context;
    DeviceEncoder encoder;
    cudaStream_t stream;
    DeviceEncodeInput ring;
    std::vector<DeviceBuffer> columns;
    std::optional<EncodingWorkspace> workspace;

    Impl(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> host, u64 capacity, cudaStream_t s,
         bool goals, bool derived, bool derived_goals, u32 constant_literals)
        : context(std::move(ctx)), encoder(checked(context, host), host), stream(s ? s : context->stream())
    {
        if (!capacity || (derived_goals && !goals))
            throw std::invalid_argument("mymyr: encoding: a device stream needs positive capacity and derived goals need fluent goals");
        if (constant_literals && !goals)
            throw std::invalid_argument("mymyr: encoding: constant goal literals need goal masks");
        DeviceGuard guard(context->device());
        ring.graphs = capacity;
        ring.row_capacity = capacity;
        ring.state_words = host->table().words();
        ring.numeric_words = host->table().numeric_words();
        ring.state_stride = u64{ring.state_words} + ring.numeric_words;
        ring.derived_stride = host->table().derived_words();
        ring.goal_words = ring.state_words;
        ring.goal_stride = ring.state_words;
        ring.goal_derived_words = host->table().derived_words();
        ring.goal_derived_stride = ring.goal_derived_words;
        columns.reserve(10);
        auto allocate = [&](u64 count, u64 width) -> void*
        {
            if (width && count > std::numeric_limits<u64>::max() / width)
                throw std::overflow_error("mymyr: encoding: device stream size overflow");
            columns.emplace_back(context, count * width, stream);
            return columns.back().data();
        };
        auto* status = static_cast<u64*>(allocate(3, sizeof(u64)));
        check(cudaMemsetAsync(status, 0, 3 * sizeof(u64), stream), "initialize encoding ring");
        ring.row_start = status;
        ring.graphs_device = status + 1;
        ring.dropped = status + 2;
        ring.states = static_cast<u64*>(allocate(capacity, ring.state_stride * sizeof(u64)));
        ring.task_ids = static_cast<i32*>(allocate(capacity, sizeof(i32)));
        if (derived && ring.derived_stride)
            ring.derived = static_cast<u64*>(allocate(capacity, ring.derived_stride * sizeof(u64)));
        if (goals)
        {
            ring.goal_pos = static_cast<u64*>(allocate(capacity, ring.goal_words * sizeof(u64)));
            ring.goal_neg = static_cast<u64*>(allocate(capacity, ring.goal_words * sizeof(u64)));
        }
        if (derived_goals && ring.goal_derived_words)
        {
            ring.goal_derived_pos = static_cast<u64*>(allocate(capacity, ring.goal_derived_words * sizeof(u64)));
            ring.goal_derived_neg = static_cast<u64*>(allocate(capacity, ring.goal_derived_words * sizeof(u64)));
        }
        if (constant_literals)
        {
            const u32 width = encoding::detail::k_constant_objects + host->context().max_arity();
            const u64 stride = u64{constant_literals} * width;
            ring.goal_constants = {static_cast<i32*>(allocate(capacity, stride * sizeof(i32))), stride,
                                   constant_literals, width};
        }
        encoder.reserve(BatchShape{capacity, 0, 0, 0}, stream);
        workspace.emplace(encoder.acquire(stream));
    }
};

DeviceRowStream::DeviceRowStream(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder, u64 capacity,
                                 cudaStream_t stream, bool goals, bool derived, bool derived_goals,
                                 u32 constant_literals)
    : m_impl(std::make_unique<Impl>(std::move(ctx), std::move(encoder), capacity, stream, goals, derived, derived_goals,
                                    constant_literals))
{
}
DeviceRowStream::~DeviceRowStream() = default;
u64 DeviceRowStream::capacity() const noexcept { return m_impl->ring.graphs; }
cudaStream_t DeviceRowStream::stream() const noexcept { return m_impl->stream; }
const u64* DeviceRowStream::status() const noexcept { return m_impl->ring.row_start; }
void DeviceRowStream::append(const DeviceEncodeInput& in)
{
    DeviceGuard guard(m_impl->context->device());
    append_rows(in, m_impl->ring, const_cast<u64*>(status()), stream());
}
void DeviceRowStream::flush(const DeviceOutput& output)
{
    DeviceGuard guard(m_impl->context->device());
    m_impl->encoder.encode(*m_impl->workspace, m_impl->ring, output);
    finish_rows(const_cast<u64*>(status()), capacity(), stream());
}
void DeviceRowStream::clear()
{
    DeviceGuard guard(m_impl->context->device());
    clear_rows(const_cast<u64*>(status()), capacity(), stream());
}
}
