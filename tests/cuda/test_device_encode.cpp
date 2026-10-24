// The device encoder against the host encoder, byte for byte: every supported configuration, both layouts and index types,
// batch sizes around the launch widths, tables of mixed instances and widths, adversarial rows, every launch
// configuration, the capacity form, invalid rows, and the encoding inside a CUDA graph. MYMYR_TEST_FULL_SUITE=1 runs
// the full list of batch sizes.

#include "../cpp/encoding/matrix.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/cuda/encoding.hpp"
#include "../../cuda/src/encoding_scratch.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/horizon_graphs.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace mymyr;
using namespace mymyr::test::encoding_matrix;

#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (cuda::device_count() == 0)                                                                                  \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

namespace
{
cuda::ContextPtr device_context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{2} << 30;  // the GPU is shared
    return cuda::DeviceContext::create(0, o);
}

bool full_suite()
{
    const char* e = std::getenv("MYMYR_TEST_FULL_SUITE");
    return e && *e && std::string(e) != "0";
}

// ------------------------------------------------------------------------------------------------ tables

struct DeviceRows
{
    cuda::DeviceBuffer states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg;
    cuda::DeviceBuffer layer_pos, layer_neg, layer_derived_pos, layer_derived_neg;
    cuda::DeviceBuffer step_pos, step_neg, step_derived_pos, step_derived_neg, step_dt, step_count;

    DeviceRows(const cuda::ContextPtr& ctx, cudaStream_t stream, const Batch& b)
    {
        auto upload = [&](cuda::DeviceBuffer& buffer, const auto& v)
        {
            buffer = cuda::DeviceBuffer(ctx, std::max<u64>(v.size() * sizeof(v[0]), 8), stream);
            if (!v.empty())
                cuda::check(cudaMemcpyAsync(buffer.data(), v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice, stream),
                            "upload rows");
        };
        upload(states, b.states);
        upload(task_ids, b.task_ids);
        upload(derived, b.derived);
        upload(goal_pos, b.goal_pos);
        upload(goal_neg, b.goal_neg);
        upload(goal_derived_pos, b.goal_derived_pos);
        upload(goal_derived_neg, b.goal_derived_neg);
        upload(layer_pos, b.layer_pos);
        upload(layer_neg, b.layer_neg);
        upload(layer_derived_pos, b.layer_derived_pos);
        upload(layer_derived_neg, b.layer_derived_neg);
        upload(step_pos, b.step_pos);
        upload(step_neg, b.step_neg);
        upload(step_derived_pos, b.step_derived_pos);
        upload(step_derived_neg, b.step_derived_neg);
        upload(step_dt, b.step_dt);
        upload(step_count, b.step_count);
        cuda::check(cudaStreamSynchronize(stream), "upload rows");
    }

    [[nodiscard]] cuda::DeviceEncodeInput input(const Batch& b, u64 n, const Inputs& inputs) const
    {
        cuda::DeviceEncodeInput in;
        in.graphs = n;
        in.states = static_cast<const u64*>(states.data());
        in.state_stride = b.words;
        in.state_words = b.words;
        in.task_ids = static_cast<const i32*>(task_ids.data());
        if (inputs.derived && b.derived_words)
        {
            in.derived = static_cast<const u64*>(derived.data());
            in.derived_stride = b.derived_words;
        }
        if (inputs.row_goals)
        {
            in.goal_pos = static_cast<const u64*>(goal_pos.data());
            in.goal_neg = static_cast<const u64*>(goal_neg.data());
            in.goal_stride = b.words;
            in.goal_words = b.words;
            if (inputs.derived_goals && b.derived_words)
            {
                in.goal_derived_pos = static_cast<const u64*>(goal_derived_pos.data());
                in.goal_derived_neg = static_cast<const u64*>(goal_derived_neg.data());
                in.goal_derived_stride = b.derived_words;
                in.goal_derived_words = b.derived_words;
            }
        }
        const auto words = [](const cuda::DeviceBuffer& buffer) { return static_cast<const u64*>(buffer.data()); };
        if (inputs.layers)
            in.subgoals = layer_view(b, inputs.layers, words(layer_pos), words(layer_neg), words(layer_derived_pos),
                                     words(layer_derived_neg));
        if (inputs.history)
            in.history = history_view(b, inputs, words(step_pos), words(step_neg), words(step_derived_pos),
                                      words(step_derived_neg), static_cast<const i32*>(step_dt.data()),
                                      static_cast<const i32*>(step_count.data()));
        return in;
    }
};

// ------------------------------------------------------------------------------------------------ running

/// What a device encode left in its output block.
struct DeviceResult
{
    std::vector<std::byte> block;
    cuda::OutputLayout layout;
    encoding::FlatSizes sizes;
};

/// Encodes on `stream` into a zeroed block laid out for `shape`, with capacities `caps` (default: the shape's), and reads
/// the block back.
DeviceResult run_device(const cuda::DeviceEncoder& device, const cuda::ContextPtr& ctx, cudaStream_t stream,
                        const cuda::DeviceEncodeInput& in, const cuda::BatchShape& shape,
                        const encoding::BatchShape* caps = nullptr, const cuda::LaunchConfig& launch = {}, int canary = 0)
{
    DeviceResult result;
    result.layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, result.layout.bytes, stream);
    cuda::check(cudaMemsetAsync(block.data(), canary, block.size(), stream), "fill the output");
    cuda::DeviceOutput out = cuda::output_of(result.layout, shape, block.data());
    if (caps)
        out.capacities = *caps;
    device.encode(in, out, stream, launch);
    result.block.resize(block.size());
    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream),
                "download the output");
    cuda::check(cudaStreamSynchronize(stream), "download the output");
    std::memcpy(&result.sizes, result.block.data() + result.layout.sizes, sizeof(encoding::FlatSizes));
    return result;
}

/// The arrays of a device result against the bundle of the host: the same arrays, the same bytes.
void expect_same_arrays(const rl::ArrayBundle& host, const DeviceResult& device)
{
    const cuda::OutputLayout& l = device.layout;
    const cuda::OutputArrays& a = l.arrays;
    struct Array
    {
        const char* name;
        u64 offset;
        bool present;
    };
    const Array arrays[] = {{"task_ids", l.task_ids, true},
                            {"num_objects", l.num_objects, true},
                            {"entity_offsets", l.entity_offsets, true},
                            {"entity_role", l.entity_role, a.entity_arrays},
                            {"entity_ref", l.entity_ref, a.entity_arrays},
                            {"relation_counts", l.relation_counts, a.counts},
                            {"relation_offsets", l.relation_offsets, a.relations},
                            {"relation_arg_offsets", l.relation_arg_offsets, a.relations},
                            {"relation_args", l.relation_args, a.relations},
                            {"goal_complete", l.goal_complete, a.goal_complete},
                            {"target_offsets", l.target_offsets, a.targets},
                            {"target_entity", l.target_entity, a.targets},
                            {"target_group", l.target_group, a.targets},
                            {"history_offsets", l.history_offsets, a.history},
                            {"history_entity", l.history_entity, a.history},
                            {"history_dt", l.history_dt, a.history},
                            {"node_offsets", l.node_offsets, a.simple},
                            {"node_label", l.node_label, a.simple},
                            {"edge_offsets", l.edge_offsets, a.simple},
                            {"edge_index", l.edge_index, a.simple},
                            {"edge_label", l.edge_label, a.simple}};
    for (const Array& array : arrays)
    {
        const rl::ArrayInfo* info = host.find(array.name);
        ASSERT_EQ(info != nullptr, array.present) << array.name;
        if (!info)
            continue;
        const u64 bytes = info->elements() * rl::dtype_bytes(info->dtype);
        ASSERT_LE(array.offset + bytes, device.block.size()) << array.name;
        EXPECT_EQ(std::memcmp(device.block.data() + array.offset, host.data(*info), bytes), 0) << array.name;
    }
}

void expect_sizes(const encoding::FlatSizes& expected, const encoding::FlatSizes& actual)
{
    EXPECT_EQ(actual.entities, expected.entities);
    EXPECT_EQ(actual.tuples, expected.tuples);
    EXPECT_EQ(actual.args, expected.args);
    EXPECT_EQ(actual.targets, expected.targets);
    EXPECT_EQ(actual.history, expected.history);
    EXPECT_EQ(actual.overflow, expected.overflow);
}

/// A host encoder's reference for a batch: the bundle and the sizes.
struct Reference
{
    std::shared_ptr<const rl::ArrayBundle> bundle;
    encoding::FlatSizes sizes;
};
Reference host_reference(const encoding::FlatEncoder& encoder, const encoding::EncodeInput& in)
{
    ThreadPool pool(1);
    encoding::EncodeScratch scratch;
    Reference r;
    r.sizes = encoder.count(in, pool, scratch);
    r.bundle = encoder.encode(in);
    return r;
}

cuda::BatchShape shape_of(u64 graphs, const encoding::FlatSizes& s)
{
    return {graphs, s.entities, s.args, s.targets, s.history};
}

/// The configurations of the matrix and the layered ones, each with the batches they run on: the rows of every batch
/// have subgoal layers and history steps.
std::vector<Config> all_configs()
{
    std::vector<Config> out = configs();
    for (Config& c : layered_configs())
        out.push_back(std::move(c));
    return out;
}
Batch layered_batch(const rl::TaskTablePtr& table, u64 count, bool walk, u64 seed)
{
    Batch b = make_batch(table, count, walk, seed);
    add_layered_rows(b, 3, 4, seed + 1000);
    return b;
}

/// The output block of an encode, read back.
DeviceResult download_encoding(const cuda::ContextPtr& ctx, cudaStream_t stream, const cuda::OutputLayout& layout,
                               const cuda::DeviceBuffer& block)
{
    (void)ctx;
    DeviceResult result;
    result.layout = layout;
    result.block.resize(block.size());
    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download batch");
    cuda::check(cudaStreamSynchronize(stream), "wait for batch");
    std::memcpy(&result.sizes, result.block.data() + layout.sizes, sizeof(result.sizes));
    return result;
}

// ------------------------------------------------------------------------------------------------ configurations

/// The batch sizes around the widths of the launches: one graph, a warp, a block and a few blocks.
std::vector<u64> batch_sizes()
{
    if (full_suite())
        return {1, 2, 31, 32, 33, 255, 256, 257, 4097, 16384};
    return {1, 33, 257, 4097};
}

// ------------------------------------------------------------------------------------------------ the matrix

/// The largest batch of a corpus entry and configuration that the matrix encodes: the output of a batch of the wide
/// instances with subgoal levels or history steps takes about 1 GB at 4097 rows and 4 GB at 16384, more than the shared
/// GPU gives one test.
u64 matrix_batch_cap(const Corpus& c, const Config& config)
{
    const bool layered = config.options.max_goal_level != 0 || config.options.history;
    return layered && std::string_view(c.name) == "wide" ? 4097 : std::numeric_limits<u64>::max();
}

TEST(DeviceEncode, EqualsTheHostEncoderBitForBit)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const std::vector<u64> sizes = batch_sizes();
    const u64 max_n = *std::max_element(sizes.begin(), sizes.end());
    u64 compared = 0, capped = 0;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, max_n, c.walk, 13);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : all_configs())
        {
            SCOPED_TRACE(config.name);
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    SCOPED_TRACE(std::string(layout == encoding::Layout::Graph ? "graph-major " : "relation-major ") +
                                 (index == encoding::IndexType::I32 ? "int32" : "int64"));
                    const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, index));
                    cuda::DeviceEncoder device(ctx, encoder);
                    const auto host_bytes = device.context().download();
                    ASSERT_EQ(host_bytes.size(), encoder->context().arrays().bytes());
                    ASSERT_EQ(std::memcmp(host_bytes.data(), encoder->context().arrays().block(), host_bytes.size()), 0);
                    const u64 cap = matrix_batch_cap(c, config);
                    device.reserve(cuda::BatchShape{std::min(max_n, cap), 0, 0, 0}, stream);
                    for (const u64 n : sizes)
                    {
                        SCOPED_TRACE("N = " + std::to_string(n));
                        if (n > cap)
                        {
                            ++capped;
                            continue;
                        }
                        const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
                        ASSERT_FALSE(host.sizes.overflowed());
                        const DeviceResult result =
                            run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape_of(n, host.sizes));
                        expect_sizes(host.sizes, result.sizes);
                        expect_same_arrays(*host.bundle, result);
                        ++compared;
                        if (::testing::Test::HasFailure())
                            return;
                    }
                }
        }
    }
    EXPECT_GT(compared, 0u);
    std::printf("[ batches  ] %llu compared, %llu above the batch cap of their corpus entry not encoded\n",
                static_cast<unsigned long long>(compared), static_cast<unsigned long long>(capped));
    RecordProperty("compared", static_cast<int>(compared));
    RecordProperty("capped", static_cast<int>(capped));
}

// ------------------------------------------------------------------------------------------------ launches

/// The results of every launch configuration are the same bytes.
TEST(DeviceEncode, EveryLaunchConfigurationGivesTheSameBytes)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 1000;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 17);
        const DeviceRows rows(ctx, stream, batch);
        // rich configurations: goals with targets, virtual nodes, row goals; and with subgoal levels and history
        for (const Config& config : {configs().back(), layered_configs()[9]})
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        {
            SCOPED_TRACE(std::string(config.name) + (layout == encoding::Layout::Graph ? " graph-major" : " relation-major"));
            const auto encoder =
                encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
            cuda::DeviceEncoder device(ctx, encoder);
            device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
            const auto input = rows.input(batch, n, config.inputs);
            const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
            const cuda::BatchShape shape = shape_of(n, host.sizes);
            // permutations of the graphs: reversed, a stride, a shuffle
            std::vector<u32> reversed(n), stride(n), shuffled(n);
            std::iota(reversed.begin(), reversed.end(), 0u);
            std::reverse(reversed.begin(), reversed.end());
            for (u64 i = 0; i < n; ++i)
                stride[i] = static_cast<u32>((i * 7) % n);  // 7 is coprime with 1000
            std::iota(shuffled.begin(), shuffled.end(), 0u);
            std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937_64(5));
            std::vector<std::pair<const char*, const std::vector<u32>*>> orders = {
                {"batch order", nullptr}, {"reversed", &reversed}, {"stride", &stride}, {"shuffled", &shuffled}};
            for (const auto& [order_name, order] : orders)
            {
                cuda::DeviceBuffer device_order;
                if (order)
                {
                    device_order = cuda::DeviceBuffer(ctx, n * 4, stream);
                    cuda::check(cudaMemcpyAsync(device_order.data(), order->data(), n * 4, cudaMemcpyHostToDevice, stream),
                                "upload the order");
                }
                for (const u32 lanes : {0u, 4u, 8u, 32u})
                    for (const auto fill : {cuda::FillMode::Auto, cuda::FillMode::Thread, cuda::FillMode::Group,
                                            cuda::FillMode::Warp})
                    {
                        SCOPED_TRACE(std::string(order_name) + ", lanes " + std::to_string(lanes) + ", fill " +
                                     std::to_string(static_cast<int>(fill)));
                        cuda::LaunchConfig launch;
                        launch.count_lanes = lanes;
                        launch.fill = fill;
                        launch.order = order ? static_cast<const u32*>(device_order.data()) : nullptr;
                        const DeviceResult result = run_device(device, ctx, stream, input, shape, nullptr, launch);
                        expect_sizes(host.sizes, result.sizes);
                        expect_same_arrays(*host.bundle, result);
                        if (::testing::Test::HasFailure())
                            return;
                    }
            }
        }
    }
}

TEST(DeviceEncode, ALaunchRefusesWhatItCannotRun)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus().front();
    const Batch batch = make_batch(c.table, 16, true, 3);
    const DeviceRows rows(ctx, stream, batch);
    const Inputs inputs;
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{16, 0, 0, 0}, stream);
    const Reference host = host_reference(*encoder, host_input(batch, 16, inputs));
    cuda::LaunchConfig launch;
    launch.count_lanes = 16;
    EXPECT_THROW((void)run_device(device, ctx, stream, rows.input(batch, 16, inputs), shape_of(16, host.sizes), nullptr, launch),
                 std::invalid_argument);
    // more graphs than the reservation
    cuda::DeviceEncoder small(ctx, encoder);
    small.reserve(cuda::BatchShape{8, 0, 0, 0}, stream);
    EXPECT_THROW((void)run_device(small, ctx, stream, rows.input(batch, 16, inputs), shape_of(16, host.sizes)),
                 std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ capacities

TEST(DeviceEncode, CapacitiesReportAsTheHostDoes)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[2];  // goals: with targets
    const u64 n = 257;
    const Batch batch = make_batch(c.table, n, true, 23);
    const DeviceRows rows(ctx, stream, batch);
    Config config = configs()[2];
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
        {
            SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
            const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, index));
            cuda::DeviceEncoder device(ctx, encoder);
            device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
            const auto host_in = host_input(batch, n, config.inputs);
            const auto device_in = rows.input(batch, n, config.inputs);
            ThreadPool pool(1);
            encoding::EncodeScratch scratch;
            const encoding::FlatSizes exact = encoder->count(host_in, pool, scratch);
            ASSERT_GT(exact.targets, 0u);
            const encoding::BatchShape fits{n, exact.entities, exact.args, exact.targets};
            struct Case
            {
                const char* name;
                encoding::BatchShape caps;
            };
            const Case cases[] = {
                {"exact", fits},
                {"roomy", {n, exact.entities + 7, exact.args + 5, exact.targets + 9}},
                {"short entities", {n, exact.entities - 1, exact.args, exact.targets}},
                {"short args", {n, exact.entities, exact.args - 1, exact.targets}},
                {"short targets", {n, exact.entities, exact.args, exact.targets - 1}},
                {"zero", {n, 0, 0, 0}},
            };
            for (const Case& k : cases)
            {
                SCOPED_TRACE(k.name);
                encoding::FlatSizes host_sizes;
                const auto host = encoder->encode_capacity(host_in, k.caps, pool, scratch, host_sizes);
                // the block holds the capacities' entries (zero beyond the batch), and the host's too
                const cuda::BatchShape shape{n, k.caps.entities, k.caps.args, k.caps.targets};
                const DeviceResult result = run_device(device, ctx, stream, device_in, shape, &k.caps, {}, host_sizes.overflow ? 0xa5 : 0);
                expect_sizes(host_sizes, result.sizes);
                if (host_sizes.overflow)
                {
                    for (u64 b = 0; b < result.layout.sizes; ++b)
                        ASSERT_EQ(result.block[b], std::byte{0xa5}) << b;
                }
                else
                    expect_same_arrays(*host, result);
                EXPECT_EQ(result.sizes.overflow != 0, std::string(k.name).find("short") != std::string::npos ||
                                                          std::string(k.name) == "zero");
            }
        }
}

TEST(DeviceEncode, AnEmptyBatchFitsAnyCapacity)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus().front();
    const Batch batch = make_batch(c.table, 4, true, 3);
    const DeviceRows rows(ctx, stream, batch);
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        const auto encoder = encoding::FlatEncoder::create(c.table, with(test::encoder_options(), layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{4, 0, 0, 0}, stream);
        const auto host = host_reference(*encoder, host_input(batch, 0, Inputs{}));
        const DeviceResult result = run_device(device, ctx, stream, rows.input(batch, 0, Inputs{}), {});
        expect_sizes(host.sizes, result.sizes);
        EXPECT_EQ(result.sizes.entities + result.sizes.tuples + result.sizes.args, 0u);
        expect_same_arrays(*host.bundle, result);
    }
}

// ------------------------------------------------------------------------------------------------ invalid rows

/// Whether the host refuses the rows, and whether the device flags them and writes nothing.
void expect_refused(const cuda::DeviceEncoder& device, const encoding::FlatEncoder& encoder, const cuda::ContextPtr& ctx,
                    cudaStream_t stream, const encoding::EncodeInput& host_in, const cuda::DeviceEncodeInput& device_in)
{
    EXPECT_THROW((void)encoder.encode(host_in), std::invalid_argument);
    const cuda::BatchShape shape{device_in.graphs, 64, 256, 64};
    const DeviceResult result = run_device(device, ctx, stream, device_in, shape, nullptr, {}, 0xa5);
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
    // nothing of the batch is written
    const cuda::OutputLayout& l = result.layout;
    for (u64 i = 0; i < l.sizes; ++i)
        ASSERT_EQ(result.block[i], std::byte{0xa5}) << "byte " << i;
}

TEST(DeviceEncode, InvalidRowsAreFlaggedAndNothingIsWritten)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const u64 n = 40;
    Batch batch = make_batch(c.table, n, true, 29);
    Config config = configs()[1];
    config.inputs = Inputs{true, true, false};
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
        const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
        const u64 words = batch.words;
        {
            SCOPED_TRACE("a task id outside the table");
            Batch bad = batch;
            bad.task_ids[7] = static_cast<i32>(c.table->size());
            const DeviceRows rows(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs), rows.input(bad, n, config.inputs));
            bad.task_ids[7] = -1;
            const DeviceRows negative(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs),
                           negative.input(bad, n, config.inputs));
        }
        {
            SCOPED_TRACE("a state bit past the instance's slots");
            Batch bad = batch;
            const u32 instance = static_cast<u32>(bad.task_ids[9]);
            const u32 past = c.table->task(instance)->atoms().fluent_slots();
            if (past < c.table->instance(instance).words * 64)
            {
                set_bit(bad.states.data() + 9 * words, past);
                const DeviceRows rows(ctx, stream, bad);
                expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs),
                               rows.input(bad, n, config.inputs));
            }
        }
        {
            SCOPED_TRACE("a row narrower than its instance");
            const DeviceRows rows(ctx, stream, batch);
            auto in = rows.input(batch, n, config.inputs);
            in.state_words = 1;
            auto host_in = host_input(batch, n, config.inputs);
            host_in.states.words = 1;
            host_in.states.stride = words;
            // instance 1 of blocks needs two words
            ASSERT_GT(c.table->instance(1).words, 1u);
            expect_refused(device, *encoder, ctx, stream, host_in, in);
        }
        {
            SCOPED_TRACE("a goal mask narrower than its instance");
            const DeviceRows rows(ctx, stream, batch);
            auto in = rows.input(batch, n, config.inputs);
            in.goal_words = 1;
            auto host_in = host_input(batch, n, config.inputs);
            host_in.goal_words = 1;
            expect_refused(device, *encoder, ctx, stream, host_in, in);
        }
        {
            SCOPED_TRACE("a goal bit the instance does not have");
            Batch bad = batch;
            const u32 instance = static_cast<u32>(bad.task_ids[5]);
            const u32 past = c.table->task(instance)->atoms().fluent_slots();
            if (past < c.table->instance(instance).words * 64)
            {
                set_bit(bad.goal_pos.data() + 5 * words, past);
                const DeviceRows rows(ctx, stream, bad);
                expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs),
                               rows.input(bad, n, config.inputs));
            }
        }
        // the batch itself is fine
        const DeviceRows rows(ctx, stream, batch);
        const auto host = host_reference(*encoder, host_input(batch, n, config.inputs));
        const DeviceResult result =
            run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape_of(n, host.sizes));
        expect_same_arrays(*host.bundle, result);
    }
}

TEST(DeviceEncode, DerivedWordsAreEvaluatedWhereAGraphReadsThem)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[1];  // features
    ASSERT_GT(c.table->derived_words(), 0u);
    const Batch batch = make_batch(c.table, 16, true, 31);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{16, 0, 0, 0}, stream);
    const Inputs without{false, false, false};
    const auto reference = host_reference(*encoder, host_input(batch, 16, without));
    expect_same_arrays(*reference.bundle, run_device(device, ctx, stream, rows.input(batch, 16, without),
                                                    shape_of(16, reference.sizes)));
    // a derived stride narrower than an instance's words
    auto narrow = rows.input(batch, 16, Inputs{});
    narrow.derived_stride = 0;
    EXPECT_THROW((void)run_device(device, ctx, stream, narrow, {16, 1, 1, 1}), std::invalid_argument);
    // without derived relations nothing is read, so nothing is required
    encoding::FlatOptions options = test::encoder_options();
    options.include_derived = false;
    const auto lean = encoding::FlatEncoder::create(c.table, options);
    cuda::DeviceEncoder lean_device(ctx, lean);
    lean_device.reserve(cuda::BatchShape{16, 0, 0, 0}, stream);
    const Reference host = host_reference(*lean, host_input(batch, 16, without));
    const DeviceResult result = run_device(lean_device, ctx, stream, rows.input(batch, 16, without), shape_of(16, host.sizes));
    expect_same_arrays(*host.bundle, result);
}

TEST(DeviceEncode, SuppliedDerivedScratchRefusesAnEvaluation)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[1];  // features
    ASSERT_GT(c.table->derived_words(), 0u);
    const Batch batch = make_batch(c.table, 16, true, 37);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    const auto reference = host_reference(*encoder, host_input(batch, 16, Inputs{}));
    const auto shape = shape_of(16, reference.sizes);
    const u64 evaluated = device.scratch_bytes(shape), supplied = device.scratch_bytes(shape, cuda::DerivedWords::Supplied);
    EXPECT_LT(supplied, evaluated);
    EXPECT_EQ(cuda::DeviceEncoder::scratch_bytes(*encoder, shape, cuda::DerivedWords::Supplied), supplied);
    EXPECT_EQ(cuda::DeviceEncoder::scratch_bytes(*encoder, shape), evaluated);
    const auto output_layout = device.output_layout(shape);
    const auto context = device.context().acquire(stream);
    cuda::DeviceBuffer scratch(ctx, supplied, stream), block(ctx, output_layout.bytes, stream);
    const auto out = cuda::output_of(output_layout, shape, block.data());
    // supplied derived words fit the smaller scratch
    device.encode(context, scratch.data(), supplied, shape, rows.input(batch, 16, Inputs{}), out);
    expect_same_arrays(*reference.bundle, download_encoding(ctx, stream, output_layout, block));
    // an evaluation needs the derived and view regions
    try
    {
        device.encode(context, scratch.data(), supplied, shape, rows.input(batch, 16, Inputs{false, false, false}), out);
        ADD_FAILURE() << "evaluated the derived words without room for them";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("DerivedWords::Evaluated"), std::string::npos) << e.what();
    }
    // without derived relations nothing is evaluated and both sizes agree
    encoding::FlatOptions options = test::encoder_options();
    options.include_derived = false;
    const cuda::DeviceEncoder lean(ctx, encoding::FlatEncoder::create(c.table, options));
    EXPECT_EQ(lean.scratch_bytes(shape), lean.scratch_bytes(shape, cuda::DerivedWords::Supplied));
}

/// An instance of `objects` objects of a domain whose axiom the device evaluates (`q` holds where `p` does).
TaskPtr gated_task(u32 objects)
{
    static const auto domain = frontend::Domain::from_string(
        "(define (domain gated) (:requirements :strips :derived-predicates) (:predicates (p ?x) (q ?x))"
        " (:derived (q ?x) (p ?x))"
        " (:action set :parameters (?x) :precondition (and) :effect (p ?x)))");
    std::string names;
    for (u32 i = 0; i < objects; ++i)
        names += " o" + std::to_string(i);
    const std::string problem = "(define (problem g" + std::to_string(objects) + ") (:domain gated) (:objects" + names +
                                ") (:init (p o0)) (:goal (and (q o1))))";
    return Task::create(*domain->instantiate_string(problem));
}

/// The rows of `b` whose instance is `id`.
Batch rows_of(const Batch& b, i32 id)
{
    Batch out = b;
    out.states.clear(); out.task_ids.clear(); out.derived.clear(); out.goal_pos.clear(); out.goal_neg.clear();
    out.goal_derived_pos.clear(); out.goal_derived_neg.clear();
    out.count = 0;
    for (u64 g = 0; g < b.count; ++g)
    {
        if (b.task_ids[g] != id)
            continue;
        const auto words = [&](const std::vector<u64>& from, std::vector<u64>& to, u64 w)
        { to.insert(to.end(), from.begin() + static_cast<std::ptrdiff_t>(g * w), from.begin() + static_cast<std::ptrdiff_t>((g + 1) * w)); };
        words(b.states, out.states, b.words);
        words(b.goal_pos, out.goal_pos, b.words);
        words(b.goal_neg, out.goal_neg, b.words);
        words(b.derived, out.derived, b.derived_words);
        words(b.goal_derived_pos, out.goal_derived_pos, b.derived_words);
        words(b.goal_derived_neg, out.goal_derived_neg, b.derived_words);
        out.task_ids.push_back(id);
        ++out.count;
    }
    return out;
}

TEST(DeviceEncode, AnInstanceOverTheDeviceLimitsFlagsOnlyItsOwnRows)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    // 2100 objects need 33 object words, more than the device evaluates
    const auto table = rl::TaskTable::create({gated_task(3), gated_task(2100)});
    ASSERT_GT(table->instance(1).ow, 32u);
    const auto encoder = encoding::FlatEncoder::create(table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{24, 0, 0, 0}, stream);
    const Batch batch = make_batch(table, 24, false, 41);
    const Inputs without{false, false, false};
    {
        const DeviceRows rows(ctx, stream, batch);
        const auto host = host_reference(*encoder, host_input(batch, 24, Inputs{}));
        const auto result = run_device(device, ctx, stream, rows.input(batch, 24, without), shape_of(24, host.sizes));
        EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
        // with their derived words supplied, the same rows encode
        expect_same_arrays(*host.bundle,
                           run_device(device, ctx, stream, rows.input(batch, 24, Inputs{}), shape_of(24, host.sizes)));
    }
    const Batch small = rows_of(batch, 0);
    ASSERT_GT(small.count, 0u);
    const DeviceRows rows(ctx, stream, small);
    const auto host = host_reference(*encoder, host_input(small, small.count, without));
    expect_same_arrays(*host.bundle, run_device(device, ctx, stream, rows.input(small, small.count, without),
                                                shape_of(small.count, host.sizes)));
    // rows without task ids are instance 0's: a table whose instance 0 is refused refuses them before any launch
    const auto refused = encoding::FlatEncoder::create(rl::TaskTable::create({gated_task(2100), gated_task(3)}),
                                                       test::encoder_options());
    cuda::DeviceEncoder refused_device(ctx, refused);
    refused_device.reserve(cuda::BatchShape{24, 0, 0, 0}, stream);
    auto in = rows.input(small, small.count, without);
    in.task_ids = nullptr;
    try
    {
        (void)run_device(refused_device, ctx, stream, in, shape_of(small.count, host.sizes));
        ADD_FAILURE() << "encoded rows of a refused instance 0";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("exceed the device limits"), std::string::npos) << e.what();
    }
}

TEST(DeviceEncode, TheCallRefusesInputThatDoesNotMatchTheSchema)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const Batch batch = make_batch(c.table, 8, true, 37);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{8, 0, 0, 0}, stream);
    const cuda::BatchShape shape{8, 100, 100, 0};
    const cuda::OutputLayout layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, layout.bytes, stream);
    const cuda::DeviceOutput good = cuda::output_of(layout, shape, block.data());
    const auto input = rows.input(batch, 8, Inputs{});
    EXPECT_NO_THROW(device.encode(input, good, stream));
    auto expect_invalid = [&](cuda::DeviceEncodeInput in, cuda::DeviceOutput out)
    { EXPECT_THROW(device.encode(in, out, stream), std::invalid_argument); };
    {
        auto out = good;
        out.sizes = nullptr;
        expect_invalid(input, out);
    }
    {
        auto out = good;
        out.arrays.relation_counts = nullptr;  // the schema stores counts
        expect_invalid(input, out);
    }
    {
        auto out = good;
        out.arrays.goal_complete = reinterpret_cast<u8*>(block.data());  // the schema has no goal relations
        expect_invalid(input, out);
    }
    {
        auto out = good;
        out.arrays.entity_ref = nullptr;
        expect_invalid(input, out);
    }
    {
        auto in = input;
        in.states = nullptr;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.state_stride = 0;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.goal_pos = in.states;  // a goal without its pair
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.goal_pos = in.states;
        in.goal_neg = in.states;  // goals for a schema without goal relations
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.graphs = 9;  // more graphs than reserved
        expect_invalid(in, good);
    }
}

// ------------------------------------------------------------------------------------------------ capture

/// The kinds of nodes in a graph.
struct NodeCounts
{
    u32 kernels = 0, memsets = 0, others = 0;
};
NodeCounts count_nodes(cudaGraph_t graph)
{
    size_t count = 0;
    cuda::check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    cuda::check(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
    NodeCounts counts;
    for (const cudaGraphNode_t node : nodes)
    {
        cudaGraphNodeType type;
        cuda::check(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        (type == cudaGraphNodeTypeKernel ? counts.kernels : type == cudaGraphNodeTypeMemset ? counts.memsets : counts.others) += 1;
    }
    return counts;
}

TEST(DeviceEncode, ACapturedEncodeReplaysWithinTheWidthClassLaunchBudget)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 600;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 41);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : {configs()[1], configs()[4], configs().back(), layered_configs()[7], layered_configs()[9]})
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            {
                SCOPED_TRACE(std::string(config.name) + (layout == encoding::Layout::Graph ? " graph-major" : " relation-major"));
                const auto encoder =
                    encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
                cuda::DeviceEncoder device(ctx, encoder);
                device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
                const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
                const cuda::BatchShape shape = shape_of(n, host.sizes);
                const auto input = rows.input(batch, n, config.inputs);
                const DeviceResult eager = run_device(device, ctx, stream, input, shape);
                expect_same_arrays(*host.bundle, eager);

                // the workspace is acquired before the capture and held through the replays
                const cuda::OutputLayout layout_of = device.output_layout(shape);
                cuda::DeviceBuffer block(ctx, layout_of.bytes, stream);
                // with the unused tails zeroed: the history arrays have a zeroing kernel of their own
                cuda::DeviceOutput out = cuda::output_of(layout_of, shape, block.data());
                out.zero_unused = true;
                const auto work = device.acquire(stream);
                cudaGraph_t graph = nullptr;
                cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                std::exception_ptr error;
                try { device.encode(work, input, out); }
                catch (...) { error = std::current_exception(); }
                cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
                if (error)
                    std::rethrow_exception(error);
                const NodeCounts nodes = count_nodes(graph);
                // at most two fill classes, four kernels per class, and the zeroing of the history arrays
                EXPECT_LE(nodes.kernels, 8u + (layout_of.arrays.history ? 1u : 0u));
                EXPECT_EQ(nodes.others, 0u) << "an allocation, copy or other node";
                cudaGraphExec_t exec = nullptr;
                cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                for (int replay = 0; replay < 2; ++replay)
                {
                    cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
                    cuda::check(cudaGraphLaunch(exec, stream), "replay");
                    std::vector<std::byte> bytes(block.size());
                    cuda::check(cudaMemcpyAsync(bytes.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
                    cuda::check(cudaStreamSynchronize(stream), "replay");
                    EXPECT_EQ(std::memcmp(bytes.data(), eager.block.data(), eager.layout.sizes), 0) << "replay " << replay;
                    encoding::FlatSizes sizes;
                    std::memcpy(&sizes, bytes.data() + layout_of.sizes, sizeof sizes);
                    expect_sizes(host.sizes, sizes);
                }
                cuda::check(cudaGraphExecDestroy(exec), "destroy the executable graph");
                cuda::check(cudaGraphDestroy(graph), "destroy the graph");
                if (::testing::Test::HasFailure())
                    return;
            }
    }
}

TEST(DeviceEncode, TheScratchOfAnEncoderEqualsThatOfItsDeviceEncoderAndGrowsWithTheBatch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
            {
                const auto encoder = encoding::FlatEncoder::create(c.table, with(configs()[1].options, layout, index));
                cuda::DeviceEncoder device(ctx, encoder);
                const auto acquisition = device.context().acquire();
                u64 last = 0;
                for (const u64 graphs : {u64{1}, u64{64}, u64{4097}})
                {
                    const cuda::BatchShape shape{graphs, 100 * graphs, 300 * graphs, 10 * graphs};
                    for (const auto derived : {cuda::DerivedWords::Evaluated, cuda::DerivedWords::Supplied})
                    {
                        const u64 bytes = cuda::DeviceEncoder::scratch_bytes(*encoder, shape, derived);
                        EXPECT_EQ(bytes, device.scratch_bytes(shape, derived));
                        EXPECT_EQ(bytes, device.scratch_bytes(acquisition, shape, derived));
                        EXPECT_GT(bytes, 0u);
                    }
                    const u64 evaluated = cuda::DeviceEncoder::scratch_bytes(*encoder, shape);
                    EXPECT_GE(evaluated, cuda::DeviceEncoder::scratch_bytes(*encoder, shape, cuda::DerivedWords::Supplied));
                    EXPECT_GT(evaluated, last);
                    last = evaluated;
                }
            }
    }
    // a table whose graphs read derived atoms needs more room to evaluate them than to be given them
    const Corpus features = corpus()[1];
    const auto encoder = encoding::FlatEncoder::create(features.table, test::encoder_options());
    const cuda::BatchShape shape{64, 0, 0, 0};
    EXPECT_GT(cuda::DeviceEncoder::scratch_bytes(*encoder, shape),
              cuda::DeviceEncoder::scratch_bytes(*encoder, shape, cuda::DerivedWords::Supplied));
}

TEST(DeviceEncode, DerivedWordsAreNeededOnlyWhereTheDeviceCannotEvaluateTheAxioms)
{
    SKIP_WITHOUT_GPU();
    auto options = test::encoder_options();
    options.max_goal_level = 2;  // so that subgoal layers can read derived atoms
    // the evaluation modes: goal masks, derived goal masks, derived subgoal masks
    constexpr std::array<std::tuple<bool, bool, bool>, 6> k_derived_modes{{{false, false, false}, {true, false, false},
                                                                           {true, true, false}, {false, false, true},
                                                                           {true, false, true}, {true, true, true}}};
    // the device evaluates the axioms of tasks whose graphs read derived atoms
    const auto evaluable = encoding::FlatEncoder::create(rl::TaskTable::create({gated_task(3), gated_task(5)}), options);
    ASSERT_TRUE(evaluable->context().needs_derived(0));
    for (const auto& [goals, derived_goals, layers] : k_derived_modes)
        EXPECT_FALSE(cuda::DeviceEncoder::needs_derived(*evaluable, goals, derived_goals, layers));
    // 2100 objects need 33 object words, more than it evaluates: rows of that instance need their derived words
    const auto beyond = encoding::FlatEncoder::create(rl::TaskTable::create({gated_task(3), gated_task(2100)}), options);
    for (const auto& [goals, derived_goals, layers] : k_derived_modes)
        EXPECT_TRUE(cuda::DeviceEncoder::needs_derived(*beyond, goals, derived_goals, layers));
    // a table without derived atoms never does
    const auto plain = encoding::FlatEncoder::create(corpus()[0].table, options);
    for (const auto& [goals, derived_goals, layers] : k_derived_modes)
        EXPECT_FALSE(cuda::DeviceEncoder::needs_derived(*plain, goals, derived_goals, layers));
    // the host answer agrees with what the device does: the refused instance's rows are flagged without derived words
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, beyond);
    device.reserve(cuda::BatchShape{24, 0, 0, 0}, stream);
    const Batch batch = make_batch(beyond->context().table_ptr(), 24, false, 41);
    const DeviceRows rows(ctx, stream, batch);
    const auto host = host_reference(*beyond, host_input(batch, 24, Inputs{}));
    const auto result =
        run_device(device, ctx, stream, rows.input(batch, 24, Inputs{false, false, false}), shape_of(24, host.sizes));
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
}

TEST(DeviceEncode, TheStreamFormLaunchesOnItsStreamAndNotTheAcquiredOne)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream owner, launch;
    const Corpus c = corpus()[2];
    const auto config = configs()[4];
    const Batch batch = make_batch(c.table, 33, c.walk, 79);
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
        const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        const Reference host = host_reference(*encoder, host_input(batch, 33, config.inputs));
        const cuda::BatchShape shape = shape_of(33, host.sizes);
        const cuda::OutputLayout layout_of = device.output_layout(shape);
        // the generation is acquired on one stream, the rows, scratch and output belong to another
        const auto acquisition = device.context().acquire(owner);
        const DeviceRows rows(ctx, launch, batch);
        const auto input = rows.input(batch, 33, config.inputs);
        const u64 bytes = device.scratch_bytes(acquisition, shape);
        cuda::DeviceBuffer scratch(ctx, bytes, launch), block(ctx, layout_of.bytes, launch);
        const cuda::DeviceOutput out = cuda::output_of(layout_of, shape, block.data());

        // capturing the launch stream finds the kernels only if the encode was enqueued on it
        cudaGraph_t graph = nullptr;
        cuda::check(cudaStreamBeginCapture(launch, cudaStreamCaptureModeThreadLocal), "begin capture");
        std::exception_ptr error;
        try { device.encode(acquisition, launch, scratch.data(), bytes, shape, input, out); }
        catch (...) { error = std::current_exception(); }
        cuda::check(cudaStreamEndCapture(launch, &graph), "end capture");
        if (error)
            std::rethrow_exception(error);
        const NodeCounts nodes = count_nodes(graph);
        EXPECT_GT(nodes.kernels, 0u);
        EXPECT_EQ(nodes.others, 0u);
        cudaGraphExec_t exec = nullptr;
        cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), launch), "clear the output");
        cuda::check(cudaGraphLaunch(exec, launch), "replay");
        expect_same_arrays(*host.bundle, download_encoding(ctx, launch, layout_of, block));
        cuda::check(cudaGraphExecDestroy(exec), "destroy the executable graph");
        cuda::check(cudaGraphDestroy(graph), "destroy the graph");

        // eager, with the rows written on the launch stream right before the encode and nothing synchronized between
        cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), launch), "clear the output");
        cuda::check(cudaMemcpyAsync(const_cast<u64*>(input.states), batch.states.data(), batch.states.size() * sizeof(u64),
                                    cudaMemcpyHostToDevice, launch),
                    "write the rows");
        device.encode(acquisition, launch, scratch.data(), bytes, shape, input, out);
        const DeviceResult eager = download_encoding(ctx, launch, layout_of, block);
        expect_sizes(host.sizes, eager.sizes);
        expect_same_arrays(*host.bundle, eager);
    }
}

/// A host callback holds a stream, without reading device memory, until the gate is destroyed (or a minute passed, so
/// that a test waiting on the held stream fails instead of hanging).
class StreamGate
{
public:
    explicit StreamGate(cudaStream_t stream) : m_stream(stream)
    {
        cuda::check(cudaLaunchHostFunc(stream, [](void* p)
        {
            auto& gate = *static_cast<StreamGate*>(p);
            std::unique_lock lock(gate.m_mutex);
            (void)gate.m_changed.wait_for(lock, std::chrono::minutes(1), [&] { return gate.m_open; });
        }, this), "cudaLaunchHostFunc");
    }
    StreamGate(const StreamGate&) = delete;
    StreamGate& operator=(const StreamGate&) = delete;
    ~StreamGate()
    {
        {
            std::lock_guard lock(m_mutex);
            m_open = true;
        }
        m_changed.notify_all();
        (void)cudaStreamSynchronize(m_stream);
    }
private:
    cudaStream_t m_stream;
    std::mutex m_mutex;
    std::condition_variable m_changed;
    bool m_open = false;
};

TEST(DeviceEncode, TheStreamFormEvaluatesTheDerivedAtomsOnItsStream)
{
    // Rows without derived words: the scratch clear and the axiom evaluation are launched on the launch stream with
    // the count and fill, so a capture of that stream holds them and a held acquisition stream delays none of them.
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream owner, launch;
    const u64 n = 33;
    std::vector<Config> chosen{configs().back()};  // derived goal masks
    for (const Config& c : layered_configs())
        if (std::string_view(c.name) == "levels_row_goals_derived_targets_virtual")
        {
            chosen.push_back(c);  // derived goal and subgoal masks
            Config layers_only = c;  // derived subgoal masks alone read the derived atoms
            layers_only.name = "derived subgoal masks alone";
            layers_only.options.include_derived = false;
            layers_only.inputs.derived_goals = false;
            chosen.push_back(layers_only);
        }
    ASSERT_EQ(chosen.size(), 3u);
    for (const auto& c : corpus())
    {
        if (!c.table->derived_words())
            continue;
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 87);
        for (Config config : chosen)
        {
            SCOPED_TRACE(config.name);
            config.inputs.derived = false;
            const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
            cuda::DeviceEncoder device(ctx, encoder);
            const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
            const cuda::BatchShape shape = shape_of(n, host.sizes);
            const cuda::OutputLayout layout = device.output_layout(shape);
            const auto acquisition = device.context().acquire(owner);
            const DeviceRows rows(ctx, launch, batch);
            const auto input = rows.input(batch, n, config.inputs);
            const u64 bytes = device.scratch_bytes(acquisition, shape);
            ASSERT_EQ(bytes, device.scratch_bytes(acquisition, shape, cuda::DerivedWords::Evaluated));
            cuda::DeviceBuffer scratch(ctx, bytes, launch), block(ctx, layout.bytes, launch);
            const cuda::DeviceOutput out = cuda::output_of(layout, shape, block.data());
            const auto capture = [&](cudaStream_t stream, auto&& enqueue)
            {
                cudaGraph_t graph = nullptr;
                cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                std::exception_ptr error;
                try { enqueue(); }
                catch (...) { error = std::current_exception(); }
                cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
                if (error)
                    std::rethrow_exception(error);
                return graph;
            };

            // the capture of the stream form has the nodes of the acquisition's own form, and its replay evaluates the
            // rows the graph reads when it runs: they are written after the capture
            cuda::check(cudaMemsetAsync(const_cast<u64*>(input.states), 0, batch.states.size() * sizeof(u64), launch),
                        "clear the rows");
            cuda::check(cudaStreamSynchronize(launch), "clear the rows");
            cudaGraph_t own = capture(owner, [&] { device.encode(acquisition, scratch.data(), bytes, shape, input, out); });
            cudaGraph_t graph =
                capture(launch, [&] { device.encode(acquisition, launch, scratch.data(), bytes, shape, input, out); });
            const NodeCounts expected = count_nodes(own), nodes = count_nodes(graph);
            EXPECT_EQ(nodes.kernels, expected.kernels);
            EXPECT_EQ(nodes.memsets, expected.memsets);
            EXPECT_EQ(nodes.others, 0u);
            cuda::check(cudaGraphDestroy(own), "destroy the graph");
            cuda::check(cudaStreamSynchronize(owner), "wait for the owner");
            cudaGraphExec_t exec = nullptr;
            cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            cuda::check(cudaMemcpyAsync(const_cast<u64*>(input.states), batch.states.data(),
                                        batch.states.size() * sizeof(u64), cudaMemcpyHostToDevice, launch),
                        "write the rows");
            cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), launch), "clear the output");
            cuda::check(cudaGraphLaunch(exec, launch), "replay");
            const DeviceResult replayed = download_encoding(ctx, launch, layout, block);
            cuda::check(cudaGraphExecDestroy(exec), "destroy the executable graph");
            cuda::check(cudaGraphDestroy(graph), "destroy the graph");
            expect_sizes(host.sizes, replayed.sizes);
            expect_same_arrays(*host.bundle, replayed);

            // eager, while the acquisition's stream is held: the launch stream finishes the batch on its own
            cuda::check(cudaMemsetAsync(scratch.data(), 0xa5, bytes, launch), "poison the scratch");
            cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), launch), "clear the output");
            DeviceResult eager;
            {
                const StreamGate gate(owner);
                device.encode(acquisition, launch, scratch.data(), bytes, shape, input, out);
                eager = download_encoding(ctx, launch, layout, block);
            }
            expect_sizes(host.sizes, eager.sizes);
            expect_same_arrays(*host.bundle, eager);
            if (::testing::Test::HasFailure())
                return;
        }
    }
}

TEST(DeviceEncode, TheWorkspaceFormEqualsTheStreamForm)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 100;
    const Corpus c = corpus()[0];
    const Batch batch = make_batch(c.table, n, true, 43);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
    const Reference host = host_reference(*encoder, host_input(batch, n, Inputs{}));
    const cuda::BatchShape shape = shape_of(n, host.sizes);
    const auto input = rows.input(batch, n, Inputs{});
    const DeviceResult by_stream = run_device(device, ctx, stream, input, shape);
    const cuda::OutputLayout layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, layout.bytes, stream);
    cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
    {
        const auto work = device.acquire(stream);
        device.encode(work, input, cuda::output_of(layout, shape, block.data()));
    }
    std::vector<std::byte> bytes(block.size());
    cuda::check(cudaMemcpyAsync(bytes.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
    cuda::check(cudaStreamSynchronize(stream), "download");
    EXPECT_EQ(std::memcmp(bytes.data(), by_stream.block.data(), by_stream.layout.sizes), 0);
    expect_same_arrays(*host.bundle, by_stream);
}

/// The device's FlatSizes of a count() on `stream`, downloaded.
encoding::FlatSizes count_sizes(const cuda::DeviceEncoder& device, const cuda::ContextPtr& ctx, cudaStream_t stream,
                                const cuda::EncodingWorkspace& work, const cuda::DeviceEncodeInput& in,
                                const cuda::LaunchConfig& launch = {})
{
    cuda::DeviceBuffer buffer(ctx, sizeof(encoding::FlatSizes), stream);
    cuda::check(cudaMemsetAsync(buffer.data(), 0xff, buffer.size(), stream), "clear the sizes");
    cuda::DeviceOutput out;
    out.capacities.graphs = work.shape().graphs;
    out.sizes = static_cast<encoding::FlatSizes*>(buffer.data());
    device.count(work, in, out, launch);
    encoding::FlatSizes sizes;
    cuda::check(cudaMemcpyAsync(&sizes, buffer.data(), sizeof(sizes), cudaMemcpyDeviceToHost, stream), "download the sizes");
    cuda::check(cudaStreamSynchronize(stream), "download the sizes");
    return sizes;
}

/// Counting and then filling is encoding: the sizes of the count are the host's, the filled arrays are the host's bytes,
/// in every configuration and layout.
TEST(DeviceEncode, CountingThenFillingEqualsTheHostEncoderBitForBit)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 300;
    u64 compared = 0;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 53);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : all_configs())
        {
            SCOPED_TRACE(config.name);
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, index));
                    cuda::DeviceEncoder device(ctx, encoder);
                    device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
                    const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
                    const cuda::BatchShape shape = shape_of(n, host.sizes);
                    const auto input = rows.input(batch, n, config.inputs);
                    const cuda::OutputLayout out_layout = device.output_layout(shape);
                    cuda::DeviceBuffer block(ctx, out_layout.bytes, stream);
                    cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
                    const auto work = device.acquire(stream);
                    const encoding::FlatSizes counted = count_sizes(device, ctx, stream, work, input);
                    expect_sizes(host.sizes, counted);
                    EXPECT_EQ(counted.reserved, 0u);
                    device.fill(work, input, cuda::output_of(out_layout, shape, block.data()));
                    DeviceResult result;
                    result.layout = out_layout;
                    result.block.resize(block.size());
                    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream),
                                "download the output");
                    cuda::check(cudaStreamSynchronize(stream), "download the output");
                    std::memcpy(&result.sizes, result.block.data() + out_layout.sizes, sizeof(encoding::FlatSizes));
                    expect_sizes(host.sizes, result.sizes);
                    expect_same_arrays(*host.bundle, result);
                    ++compared;
                    if (::testing::Test::HasFailure())
                        return;
                }
        }
    }
    EXPECT_GT(compared, 0u);
}

/// A count reports the index and input flags and never the capacity flag; a fill with short capacities reports it and
/// writes nothing.
TEST(DeviceEncode, CountingReportsTheFlagsOfTheRowsAndFillingThoseOfTheCapacities)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const u64 n = 40;
    Batch batch = make_batch(c.table, n, true, 59);
    const Config config = configs()[1];
    const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, encoding::Layout::Graph, encoding::IndexType::I32));
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{n, 0, 0, 0}, stream);
    const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
    const cuda::BatchShape shape = shape_of(n, host.sizes);
    {
        const DeviceRows rows(ctx, stream, batch);
        const auto input = rows.input(batch, n, config.inputs);
        const auto work = device.acquire(stream);
        const encoding::FlatSizes counted = count_sizes(device, ctx, stream, work, input);
        EXPECT_EQ(counted.overflow, 0u);
        // capacities one short: the fill reports it and writes no array
        const cuda::OutputLayout layout = device.output_layout(shape);
        cuda::DeviceBuffer block(ctx, layout.bytes, stream);
        cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
        cuda::DeviceOutput out = cuda::output_of(layout, shape, block.data());
        out.capacities.args = host.sizes.args - 1;
        device.fill(work, input, out);
        std::vector<std::byte> bytes(block.size());
        cuda::check(cudaMemcpyAsync(bytes.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
        cuda::check(cudaStreamSynchronize(stream), "download");
        encoding::FlatSizes sizes;
        std::memcpy(&sizes, bytes.data() + layout.sizes, sizeof(sizes));
        EXPECT_EQ(sizes.overflow, encoding::k_overflow_capacity);
        EXPECT_EQ(sizes.args, host.sizes.args);
        for (u64 i = 0; i < layout.sizes; ++i)
            ASSERT_EQ(bytes[i], std::byte{0}) << "byte " << i;
    }
    {
        batch.task_ids[3] = static_cast<i32>(c.table->size());
        const DeviceRows rows(ctx, stream, batch);
        const auto input = rows.input(batch, n, config.inputs);
        const auto work = device.acquire(stream);
        const encoding::FlatSizes counted = count_sizes(device, ctx, stream, work, input);
        EXPECT_EQ(counted.overflow, encoding::k_overflow_input);
    }
}

TEST(DeviceEncode, EncodingAfterAGrowthOfTheTableUsesTheNewInstances)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto small = table_of("blocks", {"smedium"});
    const auto grown = table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto before = encoding::FlatEncoder::create(small, test::encoder_options());
    const auto after = encoding::FlatEncoder::create(grown, test::encoder_options());
    cuda::DeviceEncoder device(ctx, before);
    device.reserve(cuda::BatchShape{64, 0, 0, 0}, stream);
    const Batch batch = make_batch(grown, 64, true, 47);
    const DeviceRows rows(ctx, stream, batch);
    ASSERT_TRUE(device.refresh(after));
    device.reserve(cuda::BatchShape{64, 0, 0, 0}, stream);
    const Reference host = host_reference(*after, host_input(batch, 64, Inputs{}));
    const DeviceResult result = run_device(device, ctx, stream, rows.input(batch, 64, Inputs{}), shape_of(64, host.sizes));
    expect_sizes(host.sizes, result.sizes);
    expect_same_arrays(*host.bundle, result);
}
TEST(DeviceEncode, WideTotalsEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus().back();  // wide
    const u64 n = 4097;
    const Batch batch = make_batch(c.table, n, c.walk, 13);
    const DeviceRows rows(ctx, stream, batch);
    u64 compared = 0;
    for (const Config& config : configs())
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
            {
                SCOPED_TRACE(std::string(config.name) + (layout == encoding::Layout::Graph ? " graph" : " relation"));
                const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, index));
                cuda::DeviceEncoder device(ctx, encoder);
                const cuda::GraphBounds b = device.bounds();
                const u64 per = b.entities + b.targets + b.tuples + (layout == encoding::Layout::Graph ? b.args : 0);
                const u64 graphs = (u64{1} << 32) / per + 2;
                const auto sl = cuda::detail::scratch_layout({graphs, 0, 0, 0}, b, layout, encoder->schema().num_relations());
                ASSERT_TRUE(sl.wide);
                if (sl.bytes > (u64{512} << 20))
                {
                    ADD_FAILURE() << "scratch too big " << sl.bytes;
                    continue;
                }
                device.reserve(cuda::BatchShape{graphs, 0, 0, 0}, stream);
                for (const u64 k : {u64{1}, u64{33}, n})
                {
                    const Reference host = host_reference(*encoder, host_input(batch, k, config.inputs));
                    for (const auto fill : {cuda::FillMode::Thread, cuda::FillMode::Group, cuda::FillMode::Warp})
                    {
                        cuda::LaunchConfig launch;
                        launch.fill = fill;
                        const DeviceResult r = run_device(device, ctx, stream, rows.input(batch, k, config.inputs),
                                                          shape_of(k, host.sizes), nullptr, launch);
                        expect_sizes(host.sizes, r.sizes);
                        expect_same_arrays(*host.bundle, r);
                        ++compared;
                    }
                }
                if (::testing::Test::HasFailure())
                    return;
            }
    EXPECT_GT(compared, 0u);
}
}  // namespace

namespace
{
TEST(DeviceEncode, GraphCapacityIsCheckedBeforeAnyLaunch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus().front();
    const auto batch = make_batch(c.table, 33, true, 73);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(33, stream);
    const auto work = device.acquire(stream);
    const auto shape = device.worst_case(32);
    const auto layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, layout.bytes, stream);
    const auto out = cuda::output_of(layout, shape, block.data());
    const auto in = rows.input(batch, 33, {});
    cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    EXPECT_THROW(device.encode(work, in, out), std::invalid_argument);
    EXPECT_THROW(device.count(work, in, out), std::invalid_argument);
    EXPECT_THROW(device.fill(work, in, out), std::invalid_argument);
    cudaGraph_t graph = nullptr;
    cuda::check(cudaStreamEndCapture(stream, &graph), "capture");
    size_t nodes = 0;
    cuda::check(cudaGraphGetNodes(graph, nullptr, &nodes), "nodes");
    EXPECT_EQ(nodes, 0u);
    cuda::check(cudaGraphDestroy(graph), "destroy");
}

TEST(DeviceEncode, CallerScratchEqualsTheWorkspaceAndCanBeCaptured)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[2];
    const auto batch = make_batch(c.table, 33, true, 79);
    const DeviceRows rows(ctx, stream, batch);
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        const auto config = configs()[4];
        const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I64));
        cuda::DeviceEncoder device(ctx, encoder);
        const auto input = rows.input(batch, 33, config.inputs);
        const auto reference = host_reference(*encoder, host_input(batch, 33, config.inputs));
        const auto shape = shape_of(33, reference.sizes);
        const auto output_layout = device.output_layout(shape);
        const auto context = device.context().acquire(stream);
        const u64 bytes = device.scratch_bytes(shape);
        cuda::DeviceBuffer scratch(ctx, bytes, stream), block(ctx, output_layout.bytes, stream);
        const auto out = cuda::output_of(output_layout, shape, block.data());
        // rows with their derived words need the scratch of supplied derived words only
        const u64 needed = input.derived ? device.scratch_bytes(shape, cuda::DerivedWords::Supplied) : bytes;
        EXPECT_THROW(device.encode(context, scratch.data(), needed - 1, shape, input, out), std::invalid_argument);
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        device.encode(context, scratch.data(), bytes, shape, input, out);
        cudaGraph_t graph = nullptr;
        cuda::check(cudaStreamEndCapture(stream, &graph), "capture");
        cudaGraphExec_t exec = nullptr;
        cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        cuda::check(cudaGraphLaunch(exec, stream), "launch");
        DeviceResult result;
        result.layout = output_layout;
        result.block.resize(block.size());
        cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
        cuda::check(cudaStreamSynchronize(stream), "download");
        std::memcpy(&result.sizes, result.block.data() + output_layout.sizes, sizeof result.sizes);
        expect_sizes(reference.sizes, result.sizes);
        expect_same_arrays(*reference.bundle, result);
        device.count(context, scratch.data(), bytes, shape, input, out);
        device.fill(context, scratch.data(), bytes, shape, input, out);
        cuda::check(cudaStreamSynchronize(stream), "fill");
        cuda::check(cudaGraphExecDestroy(exec), "destroy");
        cuda::check(cudaGraphDestroy(graph), "destroy");
    }
}

TEST(DeviceEncode, InvalidLaunchIndicesSetTheInputFlagAndLeaveTheCanary)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus().front();
    const auto batch = make_batch(c.table, 33, true, 83);
    const DeviceRows rows(ctx, stream, batch);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(33, stream);
    std::vector<u32> order(33);
    std::iota(order.begin(), order.end(), 0u);
    order[17] = 33;
    cuda::DeviceBuffer indices(ctx, order.size() * sizeof(u32), stream);
    cuda::check(cudaMemcpyAsync(indices.data(), order.data(), indices.size(), cudaMemcpyHostToDevice, stream), "order");
    cuda::LaunchConfig launch;
    launch.order = static_cast<const u32*>(indices.data());
    const auto result = run_device(device, ctx, stream, rows.input(batch, 33, {}), device.worst_case(33), nullptr, launch, 0xa5);
    EXPECT_EQ(result.sizes.overflow, encoding::k_overflow_input);
    for (u64 i = 0; i < result.layout.sizes; ++i)
        ASSERT_EQ(result.block[i], std::byte{0xa5}) << i;
}

TEST(DeviceEncode, MixedWidthsUseTheirFillClassesAndKeepBatchOrder)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 4097;
    const auto table = wide_table();
    const auto batch = make_batch(table, n, false, 89);
    const DeviceRows rows(ctx, stream, batch);
    std::vector<u32> order(n);
    std::iota(order.rbegin(), order.rend(), 0u);
    cuda::DeviceBuffer indices(ctx, n * sizeof(u32), stream);
    cuda::check(cudaMemcpyAsync(indices.data(), order.data(), indices.size(), cudaMemcpyHostToDevice, stream), "order");
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        const auto encoder = encoding::FlatEncoder::create(table, with(test::encoder_options(), layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        const auto reference = host_reference(*encoder, host_input(batch, n, {}));
        const auto shape = shape_of(n, reference.sizes);
        const auto output_layout = device.output_layout(shape);
        device.reserve(shape, stream);
        const auto work = device.acquire(stream);
        cuda::DeviceBuffer block(ctx, output_layout.bytes, stream);
        cuda::LaunchConfig launch;
        launch.order = static_cast<const u32*>(indices.data());
        const auto input = rows.input(batch, n, {});
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        device.encode(work, input, cuda::output_of(output_layout, shape, block.data()), launch);
        cudaGraph_t graph = nullptr;
        cuda::check(cudaStreamEndCapture(stream, &graph), "capture");
        size_t count = 0;
        cuda::check(cudaGraphGetNodes(graph, nullptr, &count), "nodes");
        std::vector<cudaGraphNode_t> nodes(count);
        cuda::check(cudaGraphGetNodes(graph, nodes.data(), &count), "nodes");
        u32 kernels = 0, thread_fills = 0, group_fills = 0, warp_fills = 0;
        for (const auto node : nodes)
        {
            cudaGraphNodeType type;
            cuda::check(cudaGraphNodeGetType(node, &type), "node type");
            if (type != cudaGraphNodeTypeKernel)
                continue;
            ++kernels;
            cudaKernelNodeParams params{};
            cuda::check(cudaGraphKernelNodeGetParams(node, &params), "kernel parameters");
            thread_fills += params.gridDim.x == (n + 127) / 128;
            group_fills += params.gridDim.x == (n * 8 + 127) / 128;
            warp_fills += params.gridDim.x == (n * 32 + 127) / 128;
        }
        // the narrow instance takes a thread per graph, the wide ones more lanes; at most four kernels per class
        const u32 classes = thread_fills + group_fills + warp_fills;
        EXPECT_EQ(thread_fills, 1u);
        EXPECT_GE(classes, 2u);
        EXPECT_LE(group_fills, 1u);
        EXPECT_LE(warp_fills, 1u);
        EXPECT_LE(kernels, 4 * classes);
        cuda::check(cudaGraphDestroy(graph), "destroy");
        const auto result = run_device(device, ctx, stream, input, shape, nullptr, launch);
        expect_same_arrays(*reference.bundle, result);
    }
}
}

namespace
{
TEST(DeviceEncode, GoalStridesAreCheckedOnBothBackends)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    const auto batch = make_batch(c.table, 33, true, 97);
    const auto config = configs().back();
    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(33, stream);
    const DeviceRows rows(ctx, stream, batch);
    auto host_in = host_input(batch, 33, config.inputs);
    auto device_in = rows.input(batch, 33, config.inputs);
    ASSERT_GT(batch.words, 0u);
    ASSERT_GT(batch.derived_words, 0u);
    ++host_in.goal_words;
    ++device_in.goal_words;
    EXPECT_THROW((void)encoder->encode(host_in), std::invalid_argument);
    EXPECT_THROW((void)run_device(device, ctx, stream, device_in, device.worst_case(33)), std::invalid_argument);
    --host_in.goal_words;
    --device_in.goal_words;
    ++host_in.goal_derived_words;
    ++device_in.goal_derived_words;
    EXPECT_THROW((void)encoder->encode(host_in), std::invalid_argument);
    EXPECT_THROW((void)run_device(device, ctx, stream, device_in, device.worst_case(33)), std::invalid_argument);
}
}

namespace
{
TEST(DeviceEncode, CallerScratchKeepsTheAcquiredGenerationAcrossRefresh)
{
    SKIP_WITHOUT_GPU();
    const auto domain = frontend::Domain::from_string(
        "(define (domain growing) (:requirements :strips) (:predicates (fixed ?x) (flag ?x))"
        " (:action set :parameters (?x) :precondition (and) :effect (flag ?x)))");
    const auto first = Task::create(*domain->instantiate_string(
        "(define (problem small) (:domain growing) (:objects a) (:init) (:goal (and)))"));
    const auto second = Task::create(*domain->instantiate_string(
        "(define (problem larger) (:domain growing) (:objects a) (:init (fixed a)) (:goal (and)))"));
    const auto small = encoding::FlatEncoder::create(rl::TaskTable::single(first), test::encoder_options());
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create({first, second}), test::encoder_options());
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, small);
    const u64 n = 257;
    const auto batch = make_batch(small->table_ptr(), n, true, 103);
    const DeviceRows rows(ctx, stream, batch);
    const auto input = rows.input(batch, n, {});
    const auto reference = host_reference(*small, host_input(batch, n, {}));
    const auto shape = shape_of(n, reference.sizes);
    const auto context = device.context().acquire(stream);
    const u64 bytes = device.scratch_bytes(context, shape);
    const auto layout = device.output_layout(shape);
    cuda::DeviceBuffer scratch(ctx, bytes, stream), block(ctx, layout.bytes, stream);
    const auto out = cuda::output_of(layout, shape, block.data());
    device.count(context, scratch.data(), bytes, shape, input, out);
    ASSERT_TRUE(device.refresh(grown, stream));
    EXPECT_GT(device.scratch_bytes(shape), bytes);
    EXPECT_EQ(device.scratch_bytes(context, shape), bytes);
    device.fill(context, scratch.data(), bytes, shape, input, out);
    DeviceResult result;
    result.layout = layout;
    result.block.resize(block.size());
    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
    cuda::check(cudaStreamSynchronize(stream), "download");
    std::memcpy(&result.sizes, result.block.data() + layout.sizes, sizeof result.sizes);
    expect_sizes(reference.sizes, result.sizes);
    expect_same_arrays(*reference.bundle, result);
    device.encode(context, scratch.data(), bytes, shape, input, out);
    cuda::check(cudaStreamSynchronize(stream), "encode");
}

TEST(DeviceEncode, CallerScratchRejectsAForeignAcquisitionBeforeLaunch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus().front();
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder first(ctx, encoder), second(ctx, encoder);
    const auto foreign = first.context().acquire(stream);
    const auto shape = second.worst_case(0);
    const auto layout = second.output_layout(shape);
    cuda::DeviceBuffer scratch(ctx, second.scratch_bytes(shape), stream), block(ctx, layout.bytes, stream);
    const auto out = cuda::output_of(layout, shape, block.data());
    cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    EXPECT_THROW((void)second.scratch_bytes(foreign, shape), std::invalid_argument);
    EXPECT_THROW(second.encode(foreign, scratch.data(), scratch.size(), shape, {}, out), std::invalid_argument);
    EXPECT_THROW(second.count(foreign, scratch.data(), scratch.size(), shape, {}, out), std::invalid_argument);
    EXPECT_THROW(second.fill(foreign, scratch.data(), scratch.size(), shape, {}, out), std::invalid_argument);
    cudaGraph_t graph = nullptr;
    cuda::check(cudaStreamEndCapture(stream, &graph), "capture");
    size_t nodes = 0;
    cuda::check(cudaGraphGetNodes(graph, nullptr, &nodes), "nodes");
    EXPECT_EQ(nodes, 0u);
    cuda::check(cudaGraphDestroy(graph), "destroy");
}
}

namespace
{
TEST(DeviceEncode, EvaluatedDerivedWordsEqualTheHostForMixedInstancesAndGoals)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    for (const auto& c : corpus())
    {
        if (!c.table->derived_words())
            continue;
        SCOPED_TRACE(c.name);
        const auto batch = make_batch(c.table, 33, c.walk, 123);
        const DeviceRows rows(ctx, stream, batch);
        for (auto config : configs())
        {
            config.inputs.derived = false;
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    config.options.layout = layout;
                    config.options.index = index;
                    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                    cuda::DeviceEncoder device(ctx, encoder);
                    const auto reference = host_reference(*encoder, host_input(batch, 33, config.inputs));
                    const auto shape = shape_of(33, reference.sizes);
                    device.reserve(shape, stream);
                    EXPECT_EQ(device.scratch_bytes(shape), cuda::DeviceEncoder::scratch_bytes(*encoder, shape));
                    expect_same_arrays(*reference.bundle,
                        run_device(device, ctx, stream, rows.input(batch, 33, config.inputs), shape));
                }
        }
    }
}

TEST(DeviceEncode, EvaluatedDerivedWordsEqualTheHostThroughSubgoalLayers)
{
    // derived atoms that only the subgoal layers over derived slots read are evaluated as well
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 300;
    for (const auto& c : corpus())
    {
        if (!c.table->derived_words())
            continue;
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 151);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& base : layered_configs())
            for (const bool state_derived : {true, false})
            {
                SCOPED_TRACE(std::string(base.name) + (state_derived ? "" : " without derived state relations"));
                Config config = base;
                config.inputs.derived = false;
                if (!state_derived)
                {
                    // row goals over the fluent slots only: the subgoal layers alone read the derived atoms
                    if (config.inputs.layers == 0)
                        continue;
                    config.options.include_derived = false;
                    config.inputs.row_goals = true;
                    config.inputs.derived_goals = false;
                }
                for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                {
                    const auto encoder =
                        encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
                    cuda::DeviceEncoder device(ctx, encoder);
                    const Reference reference = host_reference(*encoder, host_input(batch, n, config.inputs));
                    const cuda::BatchShape shape = shape_of(n, reference.sizes);
                    device.reserve(shape, stream);
                    const DeviceResult result = run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape);
                    expect_sizes(reference.sizes, result.sizes);
                    expect_same_arrays(*reference.bundle, result);
                    if (::testing::Test::HasFailure())
                        return;
                }
            }
    }
}

TEST(DeviceEncode, EvaluatedDerivedWordsEqualTheHostForManyRows)
{
    // the evaluation gives fewer rows more workers each: these batches cover every split of a block
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const std::vector<u64> sizes{1100, 2100, 4200, 8400, 17000};
    for (const auto& c : corpus())
    {
        if (!c.table->derived_words())
            continue;
        SCOPED_TRACE(c.name);
        const auto batch = make_batch(c.table, sizes.back(), c.walk, 131);
        const DeviceRows rows(ctx, stream, batch);
        auto config = configs().back();
        config.inputs.derived = false;
        const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{sizes.back(), 0, 0, 0}, stream);
        for (const u64 n : sizes)
        {
            SCOPED_TRACE("N = " + std::to_string(n));
            const auto reference = host_reference(*encoder, host_input(batch, n, config.inputs));
            expect_same_arrays(*reference.bundle,
                               run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape_of(n, reference.sizes)));
        }
    }
}

TEST(DeviceEncode, EvaluatedDerivedWordsReplayWithCallerScratch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    ASSERT_GT(c.table->derived_words(), 0u);
    const auto batch = make_batch(c.table, 33, c.walk, 127);
    const DeviceRows rows(ctx, stream, batch);
    auto config = configs().back();
    config.inputs.derived = false;
    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
    cuda::DeviceEncoder device(ctx, encoder);
    const auto reference = host_reference(*encoder, host_input(batch, 33, config.inputs));
    const auto shape = shape_of(33, reference.sizes);
    const auto acquisition = device.context().acquire(stream);
    const auto bytes = device.scratch_bytes(acquisition, shape);
    const auto layout = device.output_layout(shape);
    cuda::DeviceBuffer scratch(ctx, bytes, stream), block(ctx, layout.bytes, stream);
    const auto input = rows.input(batch, 33, config.inputs);
    const auto output = cuda::output_of(layout, shape, block.data());
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    device.encode(acquisition, scratch.data(), bytes, shape, input, output);
    cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda::check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
    for (int i = 0; i < 3; ++i)
        cuda::check(cudaGraphLaunch(executable, stream), "replay");
    DeviceResult result;
    result.layout = layout;
    result.block.resize(layout.bytes);
    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
    cuda::check(cudaStreamSynchronize(stream), "wait");
    expect_same_arrays(*reference.bundle, result);
    cuda::check(cudaGraphExecDestroy(executable), "destroy executable");
    cuda::check(cudaGraphDestroy(graph), "destroy graph");
}
}

namespace
{
/// Every entry past what `graphs` graphs of `sizes` use is zero, up to the capacities of `shape`: the per-graph and
/// per-cell arrays from the graphs on (the offset arrays after their closing entry) and the variable arrays from their
/// sizes on.
/// The per-graph and per-cell entries past `graphs` are zero up to the capacities, and every byte of the variable arrays
/// past `sizes` is `variable` (none: not checked).
void expect_tails(const DeviceResult& r, const cuda::BatchShape& shape, u64 graphs, u64 relations,
                  encoding::IndexType index, const encoding::FlatSizes& sizes, std::optional<std::byte> variable)
{
    const u64 w = index == encoding::IndexType::I64 ? 8 : 4;
    const cuda::OutputLayout& l = r.layout;
    const cuda::OutputArrays& a = l.arrays;
    const u64 cells = graphs * relations, capacity_cells = shape.graphs * relations;
    struct Tail
    {
        const char* name;
        u64 offset;
        bool present;
        u64 begin, end, width;
        bool per_graph = true;
    };
    const Tail tails[] = {{"task_ids", l.task_ids, true, graphs, shape.graphs, 4},
                          {"num_objects", l.num_objects, true, graphs, shape.graphs, w},
                          {"entity_offsets", l.entity_offsets, true, graphs + 1, shape.graphs + 1, w},
                          {"goal_complete", l.goal_complete, a.goal_complete, graphs, shape.graphs, 1},
                          {"target_offsets", l.target_offsets, a.targets, graphs + 1, shape.graphs + 1, w},
                          {"relation_counts", l.relation_counts, a.counts, cells, capacity_cells, w},
                          {"relation_offsets", l.relation_offsets, true, cells + 1, capacity_cells + 1, w},
                          {"relation_arg_offsets", l.relation_arg_offsets, true, cells + 1, capacity_cells + 1, w},
                          {"entity_role", l.entity_role, a.entity_arrays, sizes.entities, shape.entities, 1, false},
                          {"entity_ref", l.entity_ref, a.entity_arrays, sizes.entities, shape.entities, w, false},
                          {"relation_args", l.relation_args, true, sizes.args, shape.args, w, false},
                          {"target_entity", l.target_entity, a.targets, sizes.targets, shape.targets, w, false},
                          {"target_group", l.target_group, a.targets, sizes.targets, shape.targets, 1, false}};
    for (const Tail& t : tails)
    {
        if (!t.present || (!t.per_graph && !variable))
            continue;
        ASSERT_LE(t.offset + t.end * t.width, r.block.size()) << t.name;
        const std::byte expected = t.per_graph ? std::byte{0} : *variable;
        u64 differ = 0;
        for (u64 b = t.begin * t.width; b < t.end * t.width; ++b)
            differ += r.block[t.offset + b] != expected;
        EXPECT_EQ(differ, 0u) << t.name << " entries [" << t.begin << ", " << t.end << ")";
    }
}

u64 device_value(const cuda::DeviceBuffer& buffer, cudaStream_t stream)
{
    u64 value = 0;
    cuda::check(cudaMemcpyAsync(&value, buffer.data(), sizeof(value), cudaMemcpyDeviceToHost, stream), "read a count");
    cuda::check(cudaStreamSynchronize(stream), "read a count");
    return value;
}

std::array<u64, 3> ring_status(const cuda::DeviceRowStream& ring, cudaStream_t stream)
{
    std::array<u64, 3> status;
    cuda::check(cudaMemcpyAsync(status.data(), ring.status(), sizeof(status), cudaMemcpyDeviceToHost, stream), "read ring status");
    cuda::check(cudaStreamSynchronize(stream), "wait for ring status");
    return status;
}

TEST(DeviceRowStream, WrapsAndReportsEveryFlushAfterADroppedAppendUntilClear)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    const auto batch = make_batch(c.table, 33, c.walk, 151);
    const DeviceRows rows(ctx, stream, batch);
    for (auto config : configs())
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
            {
                config.options.layout = layout;
                config.options.index = index;
                const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                const auto host = host_reference(*encoder, host_input(batch, 33, config.inputs));
                const cuda::BatchShape shape{37, host.sizes.entities, host.sizes.args, host.sizes.targets};
                const auto output_layout = cuda::output_layout(shape, index, encoder->schema().num_relations(),
                                                               cuda::output_arrays(encoder->schema()));
                cuda::DeviceBuffer block(ctx, output_layout.bytes, stream), graph_count(ctx, sizeof(u64), stream);
                auto out = cuda::output_of(output_layout, shape, block.data());
                out.graphs = static_cast<u64*>(graph_count.data());
                cuda::DeviceRowStream ring(ctx, encoder, 37, stream, config.inputs.row_goals,
                                            config.inputs.derived && batch.derived_words,
                                            config.inputs.derived_goals && config.inputs.row_goals);
                const auto input = rows.input(batch, 33, config.inputs);
                for (int pass = 0; pass < 3; ++pass)
                {
                    // the head moves by 33 rows a pass: the rows wrap around the end of the ring
                    ring.append(input);
                    ring.flush(out);
                    expect_same_arrays(*host.bundle, download_encoding(ctx, stream, output_layout, block));
                    EXPECT_EQ(device_value(graph_count, stream), 33u);
                }
                ring.append(input);
                ring.append(rows.input(batch, 5, config.inputs));
                auto status = ring_status(ring, stream);
                EXPECT_EQ(status[1], 33u);
                EXPECT_EQ(status[2], 1u);
                for (int flush = 0; flush < 2; ++flush)
                {
                    // the batch of the dropped append and every one after it until clear() report the overflow
                    if (flush) ring.append(rows.input(batch, 5, config.inputs));
                    ring.flush(out);
                    const auto dropped = download_encoding(ctx, stream, output_layout, block);
                    EXPECT_NE(dropped.sizes.overflow & encoding::k_overflow_capacity, 0u) << flush;
                    expect_tails(dropped, shape, 0, encoder->schema().num_relations(), index, {}, std::nullopt);
                }
                ring.clear();
                status = ring_status(ring, stream);
                EXPECT_EQ(status[1], 0u);
                EXPECT_EQ(status[2], 0u);
                ring.flush(out);
                auto empty_input = host_input(batch, 0, config.inputs);
                const auto empty = encoder->encode(empty_input);
                expect_same_arrays(*empty, download_encoding(ctx, stream, output_layout, block));
                EXPECT_EQ(device_value(graph_count, stream), 0u);
                ring.append(input);
                ring.flush(out);
                expect_same_arrays(*host.bundle, download_encoding(ctx, stream, output_layout, block));
            }
}

TEST(DeviceRowStream, AFlushZeroesThePerGraphTailsAndWritesTheVariableTailsOnlyWhenAsked)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    const auto batch = make_batch(c.table, 33, c.walk, 167);
    const DeviceRows rows(ctx, stream, batch);
    auto config = configs().back();
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
        {
            config.options.layout = layout;
            config.options.index = index;
            const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
            const u64 relations = encoder->schema().num_relations();
            const auto host = host_reference(*encoder, host_input(batch, 33, config.inputs));
            const cuda::BatchShape shape{64, host.sizes.entities + 9, host.sizes.args + 17, host.sizes.targets + 3};
            const auto output_layout = cuda::output_layout(shape, index, relations, cuda::output_arrays(encoder->schema()));
            cuda::DeviceBuffer block(ctx, output_layout.bytes, stream), graph_count(ctx, sizeof(u64), stream);
            auto out = cuda::output_of(output_layout, shape, block.data());
            out.graphs = static_cast<u64*>(graph_count.data());
            cuda::DeviceRowStream ring(ctx, encoder, 64, stream, config.inputs.row_goals,
                                        config.inputs.derived && batch.derived_words,
                                        config.inputs.derived_goals && config.inputs.row_goals);
            // 33 rows, then 5 rows from a circular start, then none: every flush into a block of 0xff bytes, whose
            // variable tails a flush leaves as they are unless asked to zero them
            for (const bool zero_unused : {false, true})
                for (const u64 n : {33u, 5u, 0u})
                {
                    SCOPED_TRACE(testing::Message() << static_cast<int>(layout) << " " << static_cast<int>(index) << " "
                                                    << zero_unused << " " << n);
                    cuda::check(cudaMemsetAsync(block.data(), 0xff, block.size(), stream), "fill the output");
                    if (n) ring.append(rows.input(batch, n, config.inputs));
                    out.zero_unused = zero_unused;
                    ring.flush(out);
                    const auto result = download_encoding(ctx, stream, output_layout, block);
                    const auto reference = host_reference(*encoder, host_input(batch, n, config.inputs));
                    expect_sizes(reference.sizes, result.sizes);
                    expect_same_arrays(*reference.bundle, result);
                    EXPECT_EQ(device_value(graph_count, stream), n);
                    expect_tails(result, shape, n, relations, index, reference.sizes,
                                 zero_unused ? std::byte{0} : std::byte{0xff});
                }
        }
}

TEST(DeviceRowStream, RefusesSchemasWithSubgoalLayersOrHistory)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    const auto c = corpus().front();
    for (const int change : {0, 1})
    {
        auto options = test::encoder_options();
        if (change == 0) options.max_goal_level = 1;
        else options.history = true;
        const auto encoder = encoding::FlatEncoder::create(c.table, options);
        EXPECT_THROW(cuda::DeviceRowStream(ctx, encoder, 8), std::invalid_argument) << change;
    }
    u64 status[3] = {};
    EXPECT_THROW(cuda::finish_rows(nullptr, 8, nullptr), std::invalid_argument);
    EXPECT_THROW(cuda::finish_rows(status, 0, nullptr), std::invalid_argument);
}

TEST(DeviceRowStream, CapturedAppendAndFlushReplayWithCircularRows)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    const auto batch = make_batch(c.table, 33, c.walk, 157);
    const DeviceRows rows(ctx, stream, batch);
    for (const bool given : {false, true})
    {
        auto config = configs().back();
        config.inputs.derived = given;
        const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
        const auto host = host_reference(*encoder, host_input(batch, 33, config.inputs));
        const cuda::BatchShape shape{37, host.sizes.entities, host.sizes.args, host.sizes.targets};
        const auto layout = cuda::output_layout(shape, config.options.index, encoder->schema().num_relations(),
                                                cuda::output_arrays(encoder->schema()));
        cuda::DeviceBuffer block(ctx, layout.bytes, stream);
        const auto out = cuda::output_of(layout, shape, block.data());
        cuda::DeviceRowStream ring(ctx, encoder, 37, stream, true, given, true);
        const auto input = rows.input(batch, 33, config.inputs);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        ring.append(input);
        ring.flush(out);
        cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda::check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
        for (int pass = 0; pass < 4; ++pass)
            cuda::check(cudaGraphLaunch(executable, stream), "replay");
        expect_same_arrays(*host.bundle, download_encoding(ctx, stream, layout, block));
        cuda::check(cudaGraphExecDestroy(executable), "destroy executable");
        cuda::check(cudaGraphDestroy(graph), "destroy graph");
    }
}

TEST(DeviceCollate, RolloutSlicesEqualOneEncodeWithEmptySlicesAndCapture)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto c = corpus()[3];
    const auto batch = make_batch(c.table, 33, c.walk, 163);
    const DeviceRows rows(ctx, stream, batch);
    for (auto config : configs())
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
            {
                config.options.layout = layout;
                config.options.index = index;
                const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                cuda::DeviceEncoder device(ctx, encoder);
                device.reserve(33, stream);
                const auto host = host_reference(*encoder, host_input(batch, 33, config.inputs));
                std::vector<cuda::DeviceBuffer> blocks;
                std::vector<cuda::DeviceCollationBatch> inputs;
                u64 start = 0;
                for (u64 n : {3u, 0u, 11u, 19u})
                {
                    auto in = rows.input(batch, n, config.inputs);
                    in.states += start * in.state_stride;
                    in.task_ids += start;
                    if (in.derived) in.derived += start * in.derived_stride;
                    if (in.goal_pos) { in.goal_pos += start * in.goal_stride; in.goal_neg += start * in.goal_stride; }
                    if (in.goal_derived_pos)
                    {
                        in.goal_derived_pos += start * in.goal_derived_stride;
                        in.goal_derived_neg += start * in.goal_derived_stride;
                    }
                    const auto shape = device.worst_case(n);
                    const auto part_layout = device.output_layout(shape);
                    blocks.emplace_back(ctx, part_layout.bytes, stream);
                    const auto out = cuda::output_of(part_layout, shape, blocks.back().data());
                    device.encode(in, out, stream);
                    inputs.push_back(cuda::collation_batch(*encoder, out.arrays, out.sizes, n));
                    start += n;
                }
                cuda::DeviceCollator collator(ctx, inputs, stream);
                const auto shape = shape_of(33, host.sizes);
                const auto out_layout = device.output_layout(shape);
                cuda::DeviceBuffer scratch(ctx, collator.scratch_bytes(), stream), block(ctx, out_layout.bytes, stream);
                auto out = cuda::output_of(out_layout, shape, block.data());
                collator.encode(scratch.data(), scratch.size(), out);
                expect_same_arrays(*host.bundle, download_encoding(ctx, stream, out_layout, block));
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t executable = nullptr;
                cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                collator.encode(scratch.data(), scratch.size(), out);
                cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
                cuda::check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
                for (int pass = 0; pass < 3; ++pass) cuda::check(cudaGraphLaunch(executable, stream), "replay");
                expect_same_arrays(*host.bundle, download_encoding(ctx, stream, out_layout, block));
                cuda::check(cudaGraphExecDestroy(executable), "destroy executable");
                cuda::check(cudaGraphDestroy(graph), "destroy graph");
                for (int change = 0; change < 5; ++change)
                {
                    auto mixed = inputs;
                    if (change == 0) mixed.back().schema_fingerprint ^= 1;
                    if (change == 1) mixed.back().table_fingerprint ^= 1;
                    if (change == 2) mixed.back().max_goal_level = 1;
                    if (change == 3) mixed.back().history = true;
                    if (change == 4) mixed.back().arrays.history_offsets = mixed.back().arrays.entity_offsets;
                    EXPECT_THROW((void)cuda::DeviceCollator(ctx, mixed, stream), std::invalid_argument) << change;
                }
                // capacities past the collated graphs and sizes: the tails are zero
                const cuda::BatchShape wide{40, host.sizes.entities + 5, host.sizes.args + 11, host.sizes.targets + 2};
                const auto wide_layout = device.output_layout(wide);
                cuda::DeviceBuffer wide_block(ctx, wide_layout.bytes, stream), graph_count(ctx, sizeof(u64), stream);
                cuda::check(cudaMemsetAsync(wide_block.data(), 0xff, wide_block.size(), stream), "fill the output");
                auto wide_out = cuda::output_of(wide_layout, wide, wide_block.data());
                wide_out.zero_unused = true;
                wide_out.graphs = static_cast<u64*>(graph_count.data());
                collator.encode(scratch.data(), scratch.size(), wide_out);
                const auto zeroed = download_encoding(ctx, stream, wide_layout, wide_block);
                expect_sizes(host.sizes, zeroed.sizes);
                expect_same_arrays(*host.bundle, zeroed);
                EXPECT_EQ(device_value(graph_count, stream), 33u);
                expect_tails(zeroed, wide, 33, encoder->schema().num_relations(), index, host.sizes, std::byte{0});
            }
}
}

TEST(DeviceCollate, ChecksWideTotalsBeforeAnyIndexWrite)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const std::array<i32, 2> offsets{0, 0x7fffffff};
    const std::array<i32, 1> zero{0};
    cuda::DeviceBuffer source(ctx, sizeof(offsets), stream), empty_offsets(ctx, sizeof(zero), stream);
    cuda::check(cudaMemcpyAsync(source.data(), offsets.data(), sizeof(offsets), cudaMemcpyHostToDevice, stream), "upload offsets");
    cuda::check(cudaMemcpyAsync(empty_offsets.data(), zero.data(), sizeof(zero), cudaMemcpyHostToDevice, stream), "upload offsets");
    cuda::DeviceCollationBatch batch;
    batch.graphs = 1;
    // Implicit entity arrays represent the large count without entity storage.
    batch.stored.entity_arrays = false;
    batch.arrays.task_ids = static_cast<i32*>(empty_offsets.data());
    batch.arrays.num_objects = static_cast<i32*>(source.data()) + 1;
    batch.arrays.entity_offsets = source.data();
    batch.arrays.relation_offsets = empty_offsets.data();
    batch.arrays.relation_arg_offsets = empty_offsets.data();
    cuda::DeviceCollator collator(ctx, {batch, batch}, stream);
    const cuda::BatchShape shape{2, 0, 0, 0};
    const auto layout = cuda::output_layout(shape, encoding::IndexType::I32, 0, batch.stored);
    cuda::DeviceBuffer scratch(ctx, collator.scratch_bytes(), stream), block(ctx, layout.bytes, stream);
    cuda::check(cudaMemsetAsync(block.data(), 0x7b, block.size(), stream), "initialize canary");
    const auto out = cuda::output_of(layout, shape, block.data());
    collator.encode(scratch.data(), scratch.size(), out);
    const auto result = download_encoding(ctx, stream, layout, block);
    EXPECT_EQ(result.sizes.entities, 2 * u64{0x7fffffff});
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_index, 0u);
    EXPECT_EQ(result.block[layout.task_ids], std::byte{0x7b});
}

TEST(DeviceRowStream, MixedWidthsReadOnlyTheirInstanceWords)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto table = wide_table();
    const auto batch = make_batch(table, 33, false, 173);
    ASSERT_GT(batch.words, 64u);
    const DeviceRows rows(ctx, stream, batch);
    auto config = configs().back();
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
        {
            config.options.layout = layout;
            config.options.index = index;
            const auto encoder = encoding::FlatEncoder::create(table, config.options);
            const auto reference = host_reference(*encoder, host_input(batch, 33, config.inputs));
            const cuda::BatchShape shape{37, reference.sizes.entities, reference.sizes.args, reference.sizes.targets};
            const auto out_layout = cuda::output_layout(shape, index, encoder->schema().num_relations(),
                                                        cuda::output_arrays(encoder->schema()));
            cuda::DeviceBuffer block(ctx, out_layout.bytes, stream);
            const auto output = cuda::output_of(out_layout, shape, block.data());
            cuda::DeviceRowStream ring(ctx, encoder, 37, stream, true, false, true);
            for (int pass = 0; pass < 3; ++pass)
            {
                ring.append(rows.input(batch, 33, config.inputs));
                ring.flush(output);
                expect_same_arrays(*reference.bundle, download_encoding(ctx, stream, out_layout, block));
            }
        }
}

// ------------------------------------------------------------------------------------------------ layers and history

namespace
{
/// The layered configuration with subgoal levels, history, their targets and row goals.
Config rich_layered_config() { return layered_configs()[9]; }

TEST(DeviceEncode, LayeredCapacitiesReportAsTheHostDoes)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[1];  // features: with derived atoms
    const u64 n = 257;
    const Batch batch = layered_batch(c.table, n, true, 109);
    const DeviceRows rows(ctx, stream, batch);
    const Config config = rich_layered_config();
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
        {
            SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
            const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, index));
            cuda::DeviceEncoder device(ctx, encoder);
            device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
            const auto host_in = host_input(batch, n, config.inputs);
            const auto device_in = rows.input(batch, n, config.inputs);
            ThreadPool pool(1);
            encoding::EncodeScratch scratch;
            const encoding::FlatSizes exact = encoder->count(host_in, pool, scratch);
            ASSERT_GT(exact.targets, 0u);
            ASSERT_GT(exact.history, 0u);
            const encoding::BatchShape worst = device.worst_case(n, batch.steps);
            EXPECT_GE(worst.entities, exact.entities);
            EXPECT_GE(worst.args, exact.args);
            EXPECT_GE(worst.targets, exact.targets);
            EXPECT_GE(worst.history, exact.history);
            struct Case
            {
                const char* name;
                encoding::BatchShape caps;
            };
            const Case cases[] = {
                {"exact", {n, exact.entities, exact.args, exact.targets, exact.history}},
                {"roomy", {n, exact.entities + 7, exact.args + 5, exact.targets + 9, exact.history + 3}},
                {"worst case", worst},
                {"short entities", {n, exact.entities - 1, exact.args, exact.targets, exact.history}},
                {"short args", {n, exact.entities, exact.args - 1, exact.targets, exact.history}},
                {"short targets", {n, exact.entities, exact.args, exact.targets - 1, exact.history}},
                {"short history", {n, exact.entities, exact.args, exact.targets, exact.history - 1}},
                {"zero", {n, 0, 0, 0, 0}},
            };
            for (const Case& k : cases)
            {
                SCOPED_TRACE(k.name);
                encoding::FlatSizes host_sizes;
                const auto host = encoder->encode_capacity(host_in, k.caps, pool, scratch, host_sizes);
                const DeviceResult result =
                    run_device(device, ctx, stream, device_in, k.caps, &k.caps, {}, host_sizes.overflow ? 0xa5 : 0);
                expect_sizes(host_sizes, result.sizes);
                if (host_sizes.overflow)
                {
                    for (u64 b = 0; b < result.layout.sizes; ++b)
                        ASSERT_EQ(result.block[b], std::byte{0xa5}) << b;
                }
                else
                    expect_same_arrays(*host, result);
                EXPECT_EQ(result.sizes.overflow != 0, std::string(k.name).find("short") != std::string::npos ||
                                                          std::string(k.name) == "zero");
            }
        }
}

TEST(DeviceEncode, TheWorstCaseNeedsTheStepsOfAHistoryThatIsNotTruncated)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    const Corpus c = corpus()[0];
    encoding::FlatOptions options = test::encoder_options();
    options.history = true;
    const auto all_steps = encoding::FlatEncoder::create(c.table, options);
    cuda::DeviceEncoder device(ctx, all_steps);
    EXPECT_THROW((void)device.worst_case(8), std::invalid_argument);
    EXPECT_THROW((void)device.bytes_needed(8), std::invalid_argument);
    const auto four = device.worst_case(8, 4);
    EXPECT_EQ(four.history, 32u);
    EXPECT_EQ(device.worst_case(8, 9).history, 72u);
    EXPECT_GT(device.bytes_needed(8, 4), device.bytes_needed(8, 1));
    options.history_max_steps = 3;
    const auto limited = encoding::FlatEncoder::create(c.table, options);
    cuda::DeviceEncoder limited_device(ctx, limited);
    EXPECT_EQ(limited_device.worst_case(8).history, 24u);
    EXPECT_EQ(limited_device.worst_case(8, 2).history, 16u);
    EXPECT_EQ(limited_device.worst_case(8, 9).history, 24u);
    EXPECT_NO_THROW(limited_device.reserve(8));
    // a schema without history has no history capacity
    const auto plain = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder plain_device(ctx, plain);
    EXPECT_EQ(plain_device.worst_case(8, 4).history, 0u);
}

TEST(DeviceEncode, AnEmptyLayeredBatchFitsAnyCapacity)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus().front();
    const Batch batch = layered_batch(c.table, 4, true, 113);
    const DeviceRows rows(ctx, stream, batch);
    const Config config = rich_layered_config();
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{4, 0, 0, 0, 0}, stream);
        const auto host = host_reference(*encoder, host_input(batch, 0, config.inputs));
        const DeviceResult result = run_device(device, ctx, stream, rows.input(batch, 0, config.inputs), {});
        expect_sizes(host.sizes, result.sizes);
        EXPECT_EQ(result.sizes.entities + result.sizes.tuples + result.sizes.args + result.sizes.history, 0u);
        expect_same_arrays(*host.bundle, result);
    }
}

TEST(DeviceEncode, InvalidLayersAndHistoryAreFlaggedAndNothingIsWritten)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];  // blocks: the second instance has more words
    const u64 n = 40;
    const Batch batch = layered_batch(c.table, n, true, 127);
    const Config config = rich_layered_config();  // the newest two steps of a history are encoded
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
    {
        SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
        const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
        const u64 W = batch.words;
        const u32 H = batch.steps, L = batch.layers;
        auto past_slots = [&](u64 row)
        {
            const u32 instance = static_cast<u32>(batch.task_ids[row]);
            return c.table->task(instance)->atoms().fluent_slots();
        };
        auto has_room = [&](u64 row)
        {
            return past_slots(row) < c.table->instance(static_cast<u32>(batch.task_ids[row])).words * 64;
        };
        {
            SCOPED_TRACE("a history count past the steps");
            Batch bad = batch;
            bad.step_count[7] = static_cast<i32>(H) + 1;
            const DeviceRows rows(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs), rows.input(bad, n, config.inputs));
        }
        {
            SCOPED_TRACE("a negative history count");
            Batch bad = batch;
            bad.step_count[7] = -1;
            const DeviceRows rows(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs), rows.input(bad, n, config.inputs));
        }
        {
            SCOPED_TRACE("a subgoal bit the instance does not have");
            Batch bad = batch;
            ASSERT_TRUE(has_room(9));
            set_bit(bad.layer_pos.data() + (9 * L + 1) * W, past_slots(9));
            const DeviceRows rows(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs), rows.input(bad, n, config.inputs));
        }
        {
            SCOPED_TRACE("a bit the instance does not have in a step that is encoded");
            Batch bad = batch;
            ASSERT_TRUE(has_room(11));
            bad.step_count[11] = static_cast<i32>(H);
            set_bit(bad.step_neg.data() + (11 * H + H - 1) * W, past_slots(11));
            const DeviceRows rows(ctx, stream, bad);
            expect_refused(device, *encoder, ctx, stream, host_input(bad, n, config.inputs), rows.input(bad, n, config.inputs));
        }
        for (const char* what : {"a step beyond the valid ones", "a step older than the newest two"})
        {
            SCOPED_TRACE(what);
            // these steps are not read, whatever they hold
            Batch bad = batch;
            ASSERT_TRUE(has_room(13));
            const bool beyond = what[7] == 'b';
            bad.step_count[13] = beyond ? 1 : static_cast<i32>(H);
            set_bit(bad.step_pos.data() + (13 * H + (beyond ? H - 1 : 0)) * W, past_slots(13));
            const DeviceRows rows(ctx, stream, bad);
            const auto host = host_reference(*encoder, host_input(bad, n, config.inputs));
            const DeviceResult result =
                run_device(device, ctx, stream, rows.input(bad, n, config.inputs), shape_of(n, host.sizes));
            expect_sizes(host.sizes, result.sizes);
            expect_same_arrays(*host.bundle, result);
        }
        ASSERT_GT(c.table->instance(1).words, 1u);
        {
            SCOPED_TRACE("subgoal layers narrower than the instance");
            const DeviceRows rows(ctx, stream, batch);
            auto in = rows.input(batch, n, config.inputs);
            in.subgoals.words = 1;
            auto host_in = host_input(batch, n, config.inputs);
            host_in.subgoals.words = 1;
            expect_refused(device, *encoder, ctx, stream, host_in, in);
        }
        {
            SCOPED_TRACE("history steps narrower than the instance");
            const DeviceRows rows(ctx, stream, batch);
            auto in = rows.input(batch, n, config.inputs);
            in.history.masks.words = 1;
            auto host_in = host_input(batch, n, config.inputs);
            host_in.history.masks.words = 1;
            expect_refused(device, *encoder, ctx, stream, host_in, in);
        }
        {
            // the batch itself is fine
            const DeviceRows rows(ctx, stream, batch);
            const auto host = host_reference(*encoder, host_input(batch, n, config.inputs));
            const DeviceResult result =
                run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape_of(n, host.sizes));
            expect_same_arrays(*host.bundle, result);
        }
    }
}

TEST(DeviceEncode, TheCallRefusesLayersAndHistoryThatDoNotMatchTheSchema)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const u64 n = 8;
    const Batch batch = layered_batch(c.table, n, true, 131);
    const DeviceRows rows(ctx, stream, batch);
    const Config config = rich_layered_config();
    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
    const cuda::BatchShape shape = device.worst_case(n, batch.steps);
    const cuda::OutputLayout layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, layout.bytes, stream);
    const cuda::DeviceOutput good = cuda::output_of(layout, shape, block.data());
    const auto input = rows.input(batch, n, config.inputs);
    EXPECT_NO_THROW(device.encode(input, good, stream));
    auto expect_invalid = [&](const cuda::DeviceEncodeInput& in, const cuda::DeviceOutput& out)
    { EXPECT_THROW(device.encode(in, out, stream), std::invalid_argument); };
    {
        auto in = input;
        in.subgoals.layers = 4;  // the schema has three levels
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.subgoals.neg = nullptr;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.subgoals.stride = 1;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.history.dt = nullptr;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.history.masks.pos = nullptr;
        expect_invalid(in, good);
    }
    {
        auto in = input;
        in.history.masks.stride = 0;  // the rows of a history never share
        expect_invalid(in, good);
    }
    {
        auto out = good;
        out.arrays.history_offsets = nullptr;  // the schema has history
        expect_invalid(input, out);
    }
    {
        auto out = good;
        out.arrays.history_dt = nullptr;
        expect_invalid(input, out);
    }
    // layers and steps for a schema that has neither
    const auto plain = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder plain_device(ctx, plain);
    plain_device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
    const cuda::BatchShape plain_shape{n, 100, 100, 0, 0};
    const cuda::OutputLayout plain_layout = plain_device.output_layout(plain_shape);
    cuda::DeviceBuffer plain_block(ctx, plain_layout.bytes, stream);
    const cuda::DeviceOutput plain_out = cuda::output_of(plain_layout, plain_shape, plain_block.data());
    EXPECT_NO_THROW(plain_device.encode(rows.input(batch, n, Inputs{}), plain_out, stream));
    EXPECT_THROW(plain_device.encode(rows.input(batch, n, Inputs{true, false, false, 1}), plain_out, stream),
                 std::invalid_argument);
    EXPECT_THROW(plain_device.encode(rows.input(batch, n, Inputs{true, false, false, 0, true, true}), plain_out, stream),
                 std::invalid_argument);
    auto with_history_arrays = plain_out;
    with_history_arrays.arrays.history_offsets = good.arrays.history_offsets;
    EXPECT_THROW(plain_device.encode(rows.input(batch, n, Inputs{}), with_history_arrays, stream), std::invalid_argument);
    // a schema with levels does not need layers in every call, and one with history not history
    DeviceRows no_layers(ctx, stream, batch);
    EXPECT_NO_THROW(device.encode(no_layers.input(batch, n, Inputs{true, true, true}), good, stream));
    // a circular row view carries no subgoal layers and no history steps
    cuda::DeviceBuffer live(ctx, 3 * sizeof(u64), stream);
    cuda::check(cudaMemsetAsync(live.data(), 0, live.size(), stream), "clear the live count");
    {
        auto in = input;
        in.graphs_device = static_cast<const u64*>(live.data());
        in.row_start = static_cast<const u64*>(live.data()) + 1;
        in.row_capacity = n;
        expect_invalid(in, good);
        in = no_layers.input(batch, n, Inputs{true, true, true});
        in.graphs_device = static_cast<const u64*>(live.data());
        in.row_start = static_cast<const u64*>(live.data()) + 1;
        in.row_capacity = n;
        expect_invalid(in, good);  // the schema has history steps
    }
    EXPECT_THROW(cuda::append_rows(input, input, static_cast<u64*>(live.data()), stream), std::invalid_argument);
    cuda::check(cudaStreamSynchronize(stream), "wait");
}

TEST(DeviceEncode, LinearLiveRowsWithLayersAndHistoryEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const u64 n = 8;
    const Batch batch = layered_batch(c.table, n, true, 131);
    const DeviceRows rows(ctx, stream, batch);
    const Config config = rich_layered_config();
    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
    const auto shape = device.worst_case(n, batch.steps);
    cuda::DeviceBuffer live(ctx, sizeof(u64), stream);
    const auto layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, layout.bytes, stream);
    for (const u64 count : {u64{0}, n / 2, n})
    {
        SCOPED_TRACE(count);
        cuda::check(cudaMemcpyAsync(live.data(), &count, sizeof count, cudaMemcpyHostToDevice, stream), "set live count");
        auto in = rows.input(batch, n, config.inputs);
        in.graphs_device = static_cast<const u64*>(live.data());
        const auto host = host_reference(*encoder, host_input(batch, count, config.inputs));
        cuda::check(cudaMemsetAsync(block.data(), 0xa5, block.size(), stream), "poison output");
        auto out = cuda::output_of(layout, shape, block.data());
        out.zero_unused = true;
        device.encode(in, out, stream);
        const auto result = download_encoding(ctx, stream, layout, block);
        expect_sizes(host.sizes, result.sizes);
        expect_same_arrays(*host.bundle, result);
        expect_tails(result, shape, count, encoder->schema().num_relations(), config.options.index, host.sizes,
                     std::byte{0});
        const u64 at = result.layout.history_offsets + (count + 1) * sizeof(i32);
        const u64 end = result.layout.history_offsets + (n + 1) * sizeof(i32);
        for (u64 b = at; b < end; ++b)
            EXPECT_EQ(result.block[b], std::byte{0});
    }
}

TEST(DeviceEncode, TheCountsOfAnEmptyHistoryAreCheckedAsTheHostChecksThem)
{
    // H = 0: no count, zero counts and a count past the zero steps, with and without a history schema
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[0];
    const u64 n = 8;
    const Batch batch = make_batch(c.table, n, true, 137);
    const DeviceRows rows(ctx, stream, batch);
    for (const bool history : {false, true})
    {
        SCOPED_TRACE(history ? "history schema" : "no history schema");
        auto options = test::encoder_options();
        options.history = history;
        const auto encoder = encoding::FlatEncoder::create(c.table, options);
        cuda::DeviceEncoder device(ctx, encoder);
        device.reserve(cuda::BatchShape{n, 0, 0, 0, 0}, stream);
        for (const int counts : {-1, 0, 1})
        {
            SCOPED_TRACE(counts < 0 ? "no counts" : counts == 0 ? "zero counts" : "a count past the steps");
            std::vector<i32> count(n, 0);
            count[3] = counts > 0 ? 1 : 0;
            cuda::DeviceBuffer device_count(ctx, n * sizeof(i32), stream);
            cuda::check(cudaMemcpyAsync(device_count.data(), count.data(), n * sizeof(i32), cudaMemcpyHostToDevice, stream),
                        "upload the counts");
            auto host_in = host_input(batch, n, Inputs{});
            auto in = rows.input(batch, n, Inputs{});
            if (counts >= 0)
            {
                host_in.history.count = count.data();
                in.history.count = static_cast<const i32*>(device_count.data());
            }
            if (counts > 0)
            {
                EXPECT_THROW((void)encoder->encode(host_in), std::invalid_argument);
                const DeviceResult result = run_device(device, ctx, stream, in, {n, 64, 64, 0, 0});
                EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
                continue;
            }
            const Reference host = host_reference(*encoder, host_in);
            const DeviceResult result = run_device(device, ctx, stream, in, shape_of(n, host.sizes));
            expect_sizes(host.sizes, result.sizes);
            expect_same_arrays(*host.bundle, result);
        }
    }
}

/// The scratch of a layered batch holds the levels of every graph; scratch_bytes sizes exactly that and a call with a
/// scratch of that size equals the host.
TEST(DeviceEncode, TheScratchOfLayeredSchemasHoldsTheLevels)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const Corpus c = corpus()[1];
    const u64 n = 300;
    const Batch batch = layered_batch(c.table, n, true, 137);
    const DeviceRows rows(ctx, stream, batch);
    const Config config = rich_layered_config();
    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
    const auto plain = encoding::FlatEncoder::create(c.table, test::encoder_options());
    cuda::DeviceEncoder device(ctx, encoder), plain_device(ctx, plain);
    const cuda::BatchShape shape{n, 0, 0, 0, 0};
    // three levels over the words of the widest instance take scratch beyond that of a schema without levels
    EXPECT_GT(device.scratch_bytes(shape), plain_device.scratch_bytes(shape));
    const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
    const cuda::BatchShape out_shape = shape_of(n, host.sizes);
    const auto context = device.context().acquire(stream);
    // the rows pass their derived words: the scratch has no room for an evaluation
    const u64 bytes = device.scratch_bytes(shape, config.inputs.derived ? cuda::DerivedWords::Supplied
                                                                        : cuda::DerivedWords::Evaluated);
    cuda::DeviceBuffer scratch(ctx, bytes, stream);
    const cuda::OutputLayout out_layout = device.output_layout(out_shape);
    cuda::DeviceBuffer block(ctx, out_layout.bytes, stream);
    cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
    EXPECT_THROW(device.encode(context, scratch.data(), bytes - 64, shape, rows.input(batch, n, config.inputs),
                               cuda::output_of(out_layout, out_shape, block.data())),
                 std::invalid_argument);
    device.encode(context, scratch.data(), bytes, shape, rows.input(batch, n, config.inputs),
                  cuda::output_of(out_layout, out_shape, block.data()));
    DeviceResult result;
    result.layout = out_layout;
    result.block.resize(block.size());
    cuda::check(cudaMemcpyAsync(result.block.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "download");
    cuda::check(cudaStreamSynchronize(stream), "download");
    std::memcpy(&result.sizes, result.block.data() + out_layout.sizes, sizeof result.sizes);
    expect_sizes(host.sizes, result.sizes);
    expect_same_arrays(*host.bundle, result);
}
}  // namespace

namespace
{
TEST(DeviceEncode, RowsWithHistoryAndWithoutSubgoalLayersEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    // a batch big enough that instances of both fill classes have rows, in which some rows have no history steps and none has
    // subgoal layers: the layer storage of such a row is never bound
    const u64 n = 600;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, n, c.walk, 41);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : {layered_configs()[7], layered_configs()[9]})
        {
            SCOPED_TRACE(config.name);
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            {
                const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
                cuda::DeviceEncoder device(ctx, encoder);
                const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
                const cuda::BatchShape shape = shape_of(n, host.sizes);
                device.reserve(shape, stream);
                const DeviceResult result = run_device(device, ctx, stream, rows.input(batch, n, config.inputs), shape);
                expect_sizes(host.sizes, result.sizes);
                expect_same_arrays(*host.bundle, result);
            }
        }
    }
}

TEST(DeviceEncode, LiveScratchRegionsNeverOverlap)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        for (const Config& config : all_configs())
        {
            SCOPED_TRACE(config.name);
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            {
                const auto encoder = encoding::FlatEncoder::create(c.table, with(config.options, layout, encoding::IndexType::I32));
                cuda::DeviceEncoder device(ctx, encoder);
                for (const u64 graphs : {u64{0}, u64{1}, u64{33}, u64{600}, u64{4097}})
                {
                    SCOPED_TRACE((layout == encoding::Layout::Graph ? "graph-major " : "relation-major ") + std::to_string(graphs));
                    // the facts of a context whose graphs read derived atoms: the scratch has the regions of their
                    // evaluation as well
                    cuda::detail::EncodingFacts facts;
                    facts.bounds = device.bounds();
                    facts.derived_words = std::max<u32>(encoder->context().max_derived_words(), 1);
                    facts.axiom_view_words = 5;
                    facts.axioms.resize(1);
                    const cuda::detail::ScratchLayout s = cuda::detail::encoding_scratch_layout(
                        cuda::BatchShape{graphs, 0, 0, 0}, facts, layout, encoder->schema().num_relations());
                    struct Region
                    {
                        const char* name;
                        u64 begin, size;
                    };
                    const u64 entry = s.wide ? 8 : 4;
                    const Region regions[] = {{"flags", s.flags, 64},
                                              {"totals", s.totals, entry * s.scan_length},
                                              {"cell_counts", s.cell_counts, 4 * s.cell_entries},
                                              {"level_storage", s.level_storage, 8 * graphs * s.level_stride},
                                              {"scan_temp", s.scan_temp, s.scan_temp_bytes},
                                              {"derived", s.derived, 8 * graphs * facts.derived_words},
                                              {"axiom_views", s.axiom_views, 8 * graphs * facts.axiom_view_words}};
                    for (const Region& a : regions)
                    {
                        EXPECT_EQ(a.begin % 64, 0u) << a.name;
                        EXPECT_LE(a.begin + a.size, s.bytes) << a.name;
                        for (const Region& b : regions)
                        {
                            if (&a < &b && a.size != 0 && b.size != 0)
                            {
                                EXPECT_TRUE(a.begin + a.size <= b.begin || b.begin + b.size <= a.begin)
                                    << a.name << " [" << a.begin << ", " << a.begin + a.size << ") overlaps " << b.name << " ["
                                    << b.begin << ", " << b.begin + b.size << ")";
                            }
                        }
                    }
                    if (HasFailure())
                        return;
                }
            }
        }
    }
}
}  // namespace

namespace
{
struct LabelUploads
{
    cuda::ContextPtr context;
    cudaStream_t stream;
    std::vector<cuda::DeviceBuffer> buffers;
    template<class T> const T* put(const std::vector<T>& data)
    {
        buffers.emplace_back(context, std::max<u64>(8, data.size() * sizeof(T)), stream);
        if (!data.empty())
            cuda::check(cudaMemcpyAsync(buffers.back().data(), data.data(), data.size() * sizeof(T),
                                        cudaMemcpyHostToDevice, stream), "upload labels");
        return static_cast<const T*>(buffers.back().data());
    }
};

TEST(DeviceEncodeActions, SplitRunsAndMixedInstancesEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto domain = frontend::Domain::from_string(
        "(define (domain split) (:requirements :strips :typing :disjunctive-preconditions)"
        " (:types item) (:predicates (p ?x - item) (q ?x - item) (done ?x - item) (flag))"
        " (:action go :parameters (?x - item) :precondition (or (p ?x) (q ?x)) :effect (done ?x))"
        " (:action arm :parameters () :precondition (and) :effect (flag)))");
    std::vector<TaskPtr> tasks;
    for (const std::string& objects : {std::string("a b c"), std::string("a b c d e")})
        tasks.push_back(Task::create(*domain->instantiate_string(
            "(define (problem p) (:domain split) (:objects " + objects + " - item)"
            " (:init (p a) (q a) (p b) (q c)) (:goal (done a)))")));
    const auto table = rl::TaskTable::create(tasks);
    std::vector<u64> states;
    std::vector<i32> ids{0, 1, 0};
    for (const i32 id : ids)
        states.insert(states.end(), table->instance(u32(id)).init.begin(), table->instance(u32(id)).init.end());
    const u64 capacity = 128;
    std::vector<u64> successors(capacity * table->words());
    std::vector<i32> schema(capacity), binding(capacity * table->label_width()), parent(capacity), offsets(ids.size() + 1);
    rl::Expansion x;
    x.capacity = capacity;
    x.words = table->words();
    x.label_width = table->label_width();
    x.succ = successors.data(); x.schema = schema.data(); x.binding = binding.data();
    x.parent = parent.data(); x.offsets = offsets.data();
    const rl::StateBatchView rows{states.data(), ids.size(), table->words(), 0, 0};
    rl::expand(*table, rows, ids.data(), x);
    ASSERT_FALSE(x.overflow());
    const auto ctx = device_context();
    cuda::Stream stream;
    LabelUploads upload{ctx, stream, {}};
    cuda::DeviceEncodeInput input;
    input.graphs = ids.size();
    input.states = upload.put(states); input.task_ids = upload.put(ids);
    input.state_stride = input.state_words = table->words();
    input.actions = {upload.put(offsets), upload.put(schema), upload.put(binding), upload.put(parent), capacity,
                     table->label_width(), nullptr};
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
        {
            encoding::FlatOptions options;
            options.layout = layout; options.index = index;
            options.targets = {encoding::TargetSource::Goal, encoding::TargetSource::Action};
            options.predicate_virtual_nodes = true;
            const auto encoder = encoding::FlatEncoder::create(table, options);
            encoding::EncodeInput host;
            host.states = rows; host.task_ids = ids.data(); host.actions = &x;
            const auto reference = host_reference(*encoder, host);
            cuda::DeviceEncoder device(ctx, encoder);
            device.reserve({input.graphs, 0, 0, 0}, stream);
            const auto result = run_device(device, ctx, stream, input, shape_of(input.graphs, reference.sizes));
            expect_sizes(reference.sizes, result.sizes);
            expect_same_arrays(*reference.bundle, result);
        }
    stream.synchronize();
}

TEST(DeviceEncodeHorizon, DepthOneModesRootPoliciesAndDerivedEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    for (const auto& table : {table_of("blocks", {"smedium", "probBLOCKS-8-0"}),
                              table_of("philosophers", {"p03-phil4"})})
    {
        auto roots = test::random_walk(*table, 0, 3, 2);
        if (table->size() > 1)
        {
            const auto other = test::random_walk(*table, 1, 1, 3);
            roots.data.insert(roots.data.end(), other.data.begin(), other.data.end());
            roots.task_ids.insert(roots.task_ids.end(), other.task_ids.begin(), other.task_ids.end());
        }
        const auto view = test::horizon_of(*table, roots, 1);
        LabelUploads upload{ctx, stream, {}};
        cuda::DeviceEncodeInput input;
        input.graphs = view.graphs(); input.states = upload.put(view.nodes); input.task_ids = upload.put(view.task_ids);
        input.state_stride = input.state_words = view.words;
        const u64 words = cuda::horizon_scratch_words(view.graphs(), view.num_nodes(), view.edge_src.size());
        cuda::DeviceBuffer canonical(ctx, words * sizeof(u32), stream);
        input.horizon = {upload.put(view.node_offsets), upload.put(view.edge_offsets), upload.put(view.edge_src),
                         upload.put(view.edge_dst), upload.put(view.edge_schema), upload.put(view.edge_binding),
                         view.num_nodes(), view.edge_src.size(), view.label_width,
                         static_cast<u32*>(canonical.data()), words};
        for (const auto mode : {encoding::HorizonMode::Full, encoding::HorizonMode::Delta, encoding::HorizonMode::Action})
            for (const auto root : {encoding::RootPolicy::Include, encoding::RootPolicy::Exclude, encoding::RootPolicy::EncodeOnly})
                for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                    for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                    {
                        encoding::FlatOptions options;
                        options.layout = layout; options.index = index;
                        options.horizon = {mode, root, true, true, true};
                        options.predicate_virtual_nodes = true;
                        const auto encoder = encoding::FlatEncoder::create(table, options);
                        const auto reference = host_reference(*encoder, view.input());
                        cuda::DeviceEncoder device(ctx, encoder);
                        device.reserve({view.num_nodes(), 0, 0, 0}, stream);
                        const auto result = run_device(device, ctx, stream, input, shape_of(input.graphs, reference.sizes));
                        expect_sizes(reference.sizes, result.sizes);
                        expect_same_arrays(*reference.bundle, result);
                        if (HasFailure())
                            return;
                    }
    }
}

TEST(DeviceEncodeHorizon, DeepModesRootPoliciesAndDerivedEqualTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    for (const u32 depth : {2u, 3u})
    for (const auto& table : {table_of("blocks", {"smedium", "probBLOCKS-8-0"}),
                              table_of("philosophers", {"p03-phil4"})})
    {
        auto roots = test::random_walk(*table, 0, 3, 2);
        if (table->size() > 1)
        {
            const auto other = test::random_walk(*table, 1, 1, 3);
            roots.data.insert(roots.data.end(), other.data.begin(), other.data.end());
            roots.task_ids.insert(roots.task_ids.end(), other.task_ids.begin(), other.task_ids.end());
        }
        const auto view = test::horizon_of(*table, roots, depth);
        LabelUploads upload{ctx, stream, {}};
        cuda::DeviceEncodeInput input;
        input.graphs = view.graphs(); input.states = upload.put(view.nodes); input.task_ids = upload.put(view.task_ids);
        input.state_stride = input.state_words = view.words;
        const u64 words = cuda::horizon_scratch_words(view.graphs(), view.num_nodes(), view.edge_src.size());
        cuda::DeviceBuffer canonical(ctx, words * sizeof(u32), stream);
        input.horizon = {upload.put(view.node_offsets), upload.put(view.edge_offsets), upload.put(view.edge_src),
                         upload.put(view.edge_dst), upload.put(view.edge_schema), upload.put(view.edge_binding),
                         view.num_nodes(), view.edge_src.size(), view.label_width,
                         static_cast<u32*>(canonical.data()), words};
        for (const auto mode : {encoding::HorizonMode::Full, encoding::HorizonMode::Delta, encoding::HorizonMode::Action})
            for (const auto root : {encoding::RootPolicy::Include, encoding::RootPolicy::Exclude, encoding::RootPolicy::EncodeOnly})
                for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                    for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                    {
                        encoding::FlatOptions options;
                        options.layout = layout; options.index = index;
                        options.horizon = {mode, root, true, true, true};
                        options.predicate_virtual_nodes = true;
                        const auto encoder = encoding::FlatEncoder::create(table, options);
                        const auto reference = host_reference(*encoder, view.input());
                        cuda::DeviceEncoder device(ctx, encoder);
                        device.reserve({view.num_nodes(), 0, 0, 0}, stream);
                        const auto result = run_device(device, ctx, stream, input, shape_of(input.graphs, reference.sizes));
                        expect_sizes(reference.sizes, result.sizes);
                        expect_same_arrays(*reference.bundle, result);
                        if (HasFailure())
                            return;
                    }
    }
}

TEST(DeviceEncodeHorizon, InvalidEndpointsAndExpansionOffsetsSetTheInputFlag)
{
    SKIP_WITHOUT_GPU();
    const auto table = table_of("blocks", {"smedium"});
    const auto roots = test::random_walk(*table, 0, 1, 1);
    auto view = test::horizon_of(*table, roots, 2, 2);
    view.edge_src[0] = -1;
    const auto ctx = device_context();
    cuda::Stream stream;
    LabelUploads upload{ctx, stream, {}};
    encoding::FlatOptions options;
    options.horizon.mode = encoding::HorizonMode::Full;
    const auto encoder = encoding::FlatEncoder::create(table, options);
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve({view.num_nodes(), 0, 0, 0}, stream);
    const u64 words = cuda::horizon_scratch_words(view.graphs(), view.num_nodes(), view.edge_src.size());
    cuda::DeviceBuffer canonical(ctx, words * sizeof(u32), stream);
    cuda::DeviceEncodeInput input;
    input.graphs = view.graphs(); input.states = upload.put(view.nodes);
    input.state_stride = input.state_words = view.words;
    input.horizon = {upload.put(view.node_offsets), upload.put(view.edge_offsets), upload.put(view.edge_src),
                     upload.put(view.edge_dst), upload.put(view.edge_schema), upload.put(view.edge_binding),
                     view.num_nodes(), view.edge_src.size(), view.label_width, static_cast<u32*>(canonical.data()), words};
    auto result = run_device(device, ctx, stream, input, {input.graphs, 0, 0, 0});
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
    options.horizon = {};
    const auto flat = encoding::FlatEncoder::create(table, options);
    cuda::DeviceEncoder actions(ctx, flat);
    actions.reserve({1, 0, 0, 0}, stream);
    input.horizon = {};
    const std::vector<i32> invalid_offsets{0, -1};
    input.actions.offsets = upload.put(invalid_offsets);
    input.actions.label_width = table->label_width();
    result = run_device(actions, ctx, stream, input, {1, 0, 0, 0});
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
}
}

namespace
{
/// The device builder's view of `roots` (merged or not) from their host expansion, read back.
encoding::HorizonGraphs device_build(const cuda::ContextPtr& ctx, cudaStream_t stream, const rl::TaskTable& table,
                                     const test::Rows& roots, bool merge)
{
    const u64 G = roots.size();
    const u32 W = roots.words, L = std::max<u32>(1, table.label_width());
    const rl::StateBatchView rows{roots.data.data(), G, W, 0, 0};
    rl::Expansion count;
    rl::expand(table, rows, roots.task_ids.data(), count);
    const u64 M = count.total;
    std::vector<u64> succ(std::max<u64>(M, 1) * W);
    std::vector<i32> schema(std::max<u64>(M, 1)), binding(std::max<u64>(M, 1) * L), parent(std::max<u64>(M, 1)),
        offsets(G + 1);
    rl::Expansion x;
    x.capacity = M;
    x.words = W;
    x.label_width = L;
    x.succ = succ.data(); x.schema = schema.data(); x.binding = binding.data(); x.parent = parent.data();
    x.offsets = offsets.data();
    rl::expand(table, rows, roots.task_ids.data(), x);
    LabelUploads upload{ctx, stream, {}};
    cuda::DeviceEncodeInput in;
    in.graphs = G;
    in.states = upload.put(roots.data);
    in.task_ids = upload.put(roots.task_ids);
    in.state_stride = in.state_words = W;
    in.actions = {upload.put(offsets), upload.put(schema), upload.put(binding), upload.put(parent), M, L, nullptr};
    const u64 bytes = cuda::depth_one_horizon_scratch_bytes(M, merge);
    cuda::DeviceBuffer nodes(ctx, std::max<u64>(G + M, 1) * W * sizeof(u64), stream),
        node_offsets(ctx, (G + 1) * sizeof(i32), stream), src(ctx, std::max<u64>(M, 1) * sizeof(i32), stream),
        dst(ctx, std::max<u64>(M, 1) * sizeof(i32), stream), out_schema(ctx, std::max<u64>(M, 1) * sizeof(i32), stream),
        out_binding(ctx, std::max<u64>(M, 1) * L * sizeof(i32), stream), scratch(ctx, bytes, stream);
    const cuda::DeviceHorizonBuild out{static_cast<u64*>(nodes.data()), static_cast<i32*>(node_offsets.data()),
                                       static_cast<i32*>(src.data()), static_cast<i32*>(dst.data()), merge,
                                       static_cast<i32*>(out_schema.data()), static_cast<i32*>(out_binding.data()),
                                       scratch.data(), bytes};
    cuda::build_depth_one_horizon(in, upload.put(succ), W, out, stream);
    auto down = [&]<class T>(const cuda::DeviceBuffer& b, u64 n, T) {
        std::vector<T> v(n);
        if (n)
            cuda::check(cudaMemcpyAsync(v.data(), b.data(), n * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
        cuda::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        return v;
    };
    encoding::HorizonGraphs h;
    h.words = W;
    h.label_width = L;
    h.task_ids = roots.task_ids;
    h.node_offsets = down(node_offsets, G + 1, i32{});
    h.edge_offsets = offsets;
    const u64 n = h.node_offsets.back() > 0 ? static_cast<u64>(h.node_offsets.back()) : 0;
    h.nodes = down(nodes, n * W, u64{});
    h.edge_src = down(src, M, i32{});
    h.edge_dst = down(dst, M, i32{});
    h.edge_schema = down(out_schema, M, i32{});
    h.edge_binding = down(out_binding, M * L, i32{});
    return h;
}
}  // namespace

TEST(DeviceEncodeHorizon, TheParallelBuilderEqualsTheHostBuilderMergedAndUnmerged)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    for (const auto& table : {table_of("blocks", {"smedium", "probBLOCKS-8-0"}), table_of("philosophers", {"p03-phil4"}),
                              table_of("gripper", {"p02"})})
    {
        auto roots = test::random_walk(*table, 0, 6, 3);
        if (table->size() > 1)
        {
            const auto other = test::random_walk(*table, 1, 2, 3);
            roots.data.insert(roots.data.end(), other.data.begin(), other.data.end());
            roots.task_ids.insert(roots.task_ids.end(), other.task_ids.begin(), other.task_ids.end());
        }
        // a repeated root: its graph is built again
        roots.data.insert(roots.data.end(), roots.data.begin(), roots.data.begin() + roots.words);
        roots.task_ids.push_back(roots.task_ids.front());
        for (const bool merge : {false, true})
        {
            SCOPED_TRACE(merge ? "merged" : "unmerged");
            encoding::HorizonBuild options;
            options.depth = 1;
            options.merge = merge;
            const auto host = encoding::build_horizon_graphs(*table, {roots.data.data(), roots.size(), roots.words, 0, 0},
                                                             roots.task_ids.data(), options);
            const auto device = device_build(ctx, stream, *table, roots, merge);
            EXPECT_EQ(device.node_offsets, host.node_offsets);
            EXPECT_EQ(device.edge_offsets, host.edge_offsets);
            EXPECT_EQ(device.nodes, host.nodes);
            EXPECT_EQ(device.edge_src, host.edge_src);
            EXPECT_EQ(device.edge_dst, host.edge_dst);
            EXPECT_EQ(device.edge_schema, host.edge_schema);
            EXPECT_EQ(device.edge_binding, host.edge_binding);
        }
    }
}

TEST(DeviceEncodeHorizon, RecordedStateTraversalMatchesTheHostAndChecksScratch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const std::vector<u64> rows{0, 1, 2, 3}, offsets{0, 3, 5, 8, 8};
    const std::vector<u32> targets{1, 1, 2, 0, 3, 1, 3, 2}, schemas{0, 1, 0, 0, 0, 0, 1, 2};
    const std::vector<u32> bindings{0, ~0u, 1, 2, 2, ~0u, 0, ~0u, 1, ~0u, 2, ~0u, 1, 2, 0, 1};
    const encoding::SuccessorGraph host_space{4, 1, rows.data(), 1, offsets.data(), targets.data(), schemas.data(),
                                               bindings.data(), 2};
    LabelUploads upload{ctx, stream, {}};
    const cuda::DeviceHorizonSpace space{upload.put(rows), upload.put(offsets), upload.put(targets), upload.put(schemas),
                                          upload.put(bindings), 4, 1, 2, 8};
    for (const u32 graphs : {0u, 1u, 5u, 129u})
        for (const u32 depth : {0u, 2u, 3u})
            for (const bool truncate : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << graphs << " roots, depth " << depth << ", truncate " << truncate);
                std::vector<u32> roots(graphs);
                std::vector<i32> ids(graphs);
                for (u32 g = 0; g < graphs; ++g)
                    ids[g] = static_cast<i32>(roots[g] = g % 4);
                encoding::HorizonBuild options;
                options.depth = depth;
                options.max_nodes = truncate ? 2 : 4;
                options.truncate = truncate;
                const auto reference = encoding::build_horizon_graphs(host_space, roots, 0, options);
                const u64 bytes = cuda::space_horizon_scratch_bytes(4, 8, graphs);
                cuda::DeviceBuffer nodes(ctx, std::max<u64>(1, graphs * 4) * sizeof(u64), stream),
                    no(ctx, (u64(graphs) + 1) * sizeof(i32), stream), eo(ctx, (u64(graphs) + 1) * sizeof(i32), stream),
                    src(ctx, std::max<u64>(1, graphs * 8) * sizeof(i32), stream),
                    dst(ctx, src.size(), stream), schema(ctx, src.size(), stream), binding(ctx, src.size() * 2, stream),
                    cut(ctx, std::max<u64>(1, graphs), stream), scratch(ctx, bytes, stream);
                cuda::DeviceSpaceHorizonBuild out{upload.put(ids), graphs, depth, truncate ? 2u : 0u, truncate,
                    static_cast<u64*>(nodes.data()), static_cast<i32*>(no.data()), static_cast<i32*>(eo.data()),
                    static_cast<i32*>(src.data()), static_cast<i32*>(dst.data()), static_cast<i32*>(schema.data()),
                    static_cast<i32*>(binding.data()), scratch.data(), bytes, static_cast<u8*>(cut.data())};
                if (graphs)
                {
                    auto bad = out;
                    bad.scratch_bytes = bytes - 1;
                    EXPECT_THROW(cuda::build_space_horizons(space, bad, stream), std::invalid_argument);
                    bad = out;
                    bad.scratch = static_cast<std::byte*>(scratch.data()) + 4;
                    EXPECT_THROW(cuda::build_space_horizons(space, bad, stream), std::invalid_argument);
                    bad.scratch = nullptr;
                    EXPECT_THROW(cuda::build_space_horizons(space, bad, stream), std::invalid_argument);
                }
                cuda::build_space_horizons(space, out, stream);
                auto down = [&]<class T>(const cuda::DeviceBuffer& buffer, u64 n, T)
                {
                    std::vector<T> values(n);
                    if (n)
                        cuda::check(cudaMemcpyAsync(values.data(), buffer.data(), n * sizeof(T), cudaMemcpyDeviceToHost,
                                                    stream), "download horizon");
                    cuda::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
                    return values;
                };
                EXPECT_EQ(down(no, u64(graphs) + 1, i32{}), reference.node_offsets);
                EXPECT_EQ(down(eo, u64(graphs) + 1, i32{}), reference.edge_offsets);
                EXPECT_EQ(down(nodes, reference.nodes.size(), u64{}), reference.nodes);
                EXPECT_EQ(down(src, reference.edge_src.size(), i32{}), reference.edge_src);
                EXPECT_EQ(down(dst, reference.edge_dst.size(), i32{}), reference.edge_dst);
                EXPECT_EQ(down(schema, reference.edge_schema.size(), i32{}), reference.edge_schema);
                EXPECT_EQ(down(binding, reference.edge_binding.size(), i32{}), reference.edge_binding);
                EXPECT_EQ(down(cut, graphs, u8{}), reference.truncated);
            }
}

TEST(DeviceEncodeHorizon, AnExpansionStatusReportsItsOverflowAsACapacityOverflow)
{
    SKIP_WITHOUT_GPU();
    const auto table = table_of("blocks", {"smedium"});
    const auto view = test::horizon_of(*table, test::random_walk(*table, 0, 2, 2), 1);
    const auto ctx = device_context();
    cuda::Stream stream;
    LabelUploads upload{ctx, stream, {}};
    encoding::FlatOptions options;
    options.horizon.mode = encoding::HorizonMode::Delta;
    const auto encoder = encoding::FlatEncoder::create(table, options);
    const auto reference = host_reference(*encoder, view.input());
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve({view.num_nodes(), 0, 0, 0}, stream);
    const u64 words = cuda::horizon_scratch_words(view.graphs(), view.num_nodes(), view.edge_src.size());
    cuda::DeviceBuffer canonical(ctx, words * sizeof(u32), stream);
    cuda::DeviceEncodeInput input;
    input.graphs = view.graphs(); input.states = upload.put(view.nodes); input.task_ids = upload.put(view.task_ids);
    input.state_stride = input.state_words = view.words;
    std::vector<i32> invalid_offsets = view.node_offsets;
    invalid_offsets.back() = -1;  // what the builder writes past an expansion that overflowed
    for (const u32 status : {0u, 1u, 2u, 3u})
        for (const bool valid : {true, false})
        {
            SCOPED_TRACE("status " + std::to_string(status) + (valid ? " valid" : " invalid offsets"));
            input.horizon = {upload.put(valid ? view.node_offsets : invalid_offsets), upload.put(view.edge_offsets),
                             upload.put(view.edge_src), upload.put(view.edge_dst), upload.put(view.edge_schema),
                             upload.put(view.edge_binding), view.num_nodes(), view.edge_src.size(), view.label_width,
                             static_cast<u32*>(canonical.data()), words, upload.put(std::vector<u32>{status})};
            const auto result = run_device(device, ctx, stream, input, shape_of(input.graphs, reference.sizes));
            const bool capacity = status & 1u, input_flag = (status & 2u) || (!valid && !capacity);
            EXPECT_EQ((result.sizes.overflow & encoding::k_overflow_capacity) != 0, capacity);
            EXPECT_EQ((result.sizes.overflow & encoding::k_overflow_input) != 0, input_flag);
            if (status == 0 && valid)
                expect_same_arrays(*reference.bundle, result);
        }
}

TEST(DeviceEncode, SimpleEqualsTheHostForEveryCoreConfiguration)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    u32 compared = 0;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = layered_batch(c.table, 17, c.walk, 13);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : all_configs())
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                    for (const bool only : {false, true})
                    {
                        if (config.options.predicate_virtual_nodes || (only && layout == encoding::Layout::Relation))
                            continue;
                        SCOPED_TRACE(config.name);
                        SCOPED_TRACE(std::string(only ? "simple-only " : "both ") +
                                     (layout == encoding::Layout::Graph ? "graph " : "relation ") +
                                     (index == encoding::IndexType::I32 ? "int32" : "int64"));
                        auto options = with(config.options, layout, index);
                        options.outputs = {encoding::Output::Simple};
                        if (!only) options.outputs.insert(encoding::Output::Relations);
                        options.entity_arrays = encoding::EntityArrays::Always;
                        if (only) options.emit_counts = true;
                        const auto encoder = encoding::FlatEncoder::create(c.table, options);
                        cuda::DeviceEncoder device(ctx, encoder);
                        device.reserve(cuda::BatchShape{17, 0, 0, 0, 0}, stream);
                        for (const u64 n : {u64{0}, u64{1}, u64{17}})
                        {
                            SCOPED_TRACE("N = " + std::to_string(n));
                            const Reference host = host_reference(*encoder, host_input(batch, n, config.inputs));
                            const auto result = run_device(device, ctx, stream, rows.input(batch, n, config.inputs),
                                                           shape_of(n, host.sizes));
                            expect_sizes(host.sizes, result.sizes);
                            expect_same_arrays(*host.bundle, result);
                            ++compared;
                            if (::testing::Test::HasFailure()) return;
                        }
                    }
    }
    EXPECT_GT(compared, 0u);
    RecordProperty("compared", compared);
}

TEST(DeviceEncoderSchema, SimpleOutputsHaveTheirArraysAndPrivateRelationStorage)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    for (const bool only : {false, true})
    {
        auto options = test::encoder_options();
        options.outputs = {encoding::Output::Simple};
        if (!only) options.outputs.insert(encoding::Output::Relations);
        const auto encoder = encoding::FlatEncoder::create(table, options);
        EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(encoder->schema()));
        const auto stored = cuda::output_arrays(encoder->schema());
        EXPECT_TRUE(stored.simple);
        EXPECT_EQ(stored.relations, !only);
        EXPECT_EQ(stored.counts, !only);
        const auto layout = cuda::output_layout({3, 20, 80}, options.index, encoder->schema().num_relations(), stored);
        std::vector<std::byte> bytes(layout.bytes);
        const auto out = cuda::output_of(layout, {3, 20, 80}, bytes.data());
        EXPECT_NE(out.arrays.node_offsets, nullptr);
        EXPECT_NE(out.arrays.edge_offsets, nullptr);
        EXPECT_EQ(out.arrays.edge_stride, 160u);
        EXPECT_EQ(out.arrays.relation_offsets != nullptr, !only);
        EXPECT_EQ(out.scratch_relation_offsets != nullptr, only);
    }
}

TEST(DeviceEncoderSchema, SimpleRefusesNumericRelationsAndHorizonsAndTakesActionTargets)
{
    const auto numeric = rl::TaskTable::single(test::pddl_task("counters", "p01"));
    const auto plain = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    for (const bool only : {false, true})
    {
        auto options = test::encoder_options();
        options.outputs = {encoding::Output::Simple};
        if (!only) options.outputs.insert(encoding::Output::Relations);
        for (const bool include : {false, true})
        {
            // the rows of a numeric table carry numeric words; only numeric relations have a node value
            options.include_numeric = include;
            const auto encoder = encoding::FlatEncoder::create(numeric, options);
            if (include)
            {
                EXPECT_THROW(cuda::DeviceEncoder::check_encoder(*encoder), std::invalid_argument);
                EXPECT_THROW((void)cuda::collation_batch(*encoder, {}, nullptr, 0), std::invalid_argument);
            }
            else
            {
                EXPECT_NO_THROW(cuda::DeviceEncoder::check_encoder(*encoder));
                EXPECT_NO_THROW((void)cuda::collation_batch(*encoder, {}, nullptr, 0));
            }
        }
        options.include_numeric = true;
        // the nodes and edges of a horizon are encoded into relations only
        for (const auto mode : {encoding::HorizonMode::Full, encoding::HorizonMode::Delta,
                                encoding::HorizonMode::Action})
        {
            options.horizon.mode = mode;
            const auto encoder = encoding::FlatEncoder::create(plain, options);
            try
            {
                cuda::DeviceEncoder::check_schema(encoder->schema());
                ADD_FAILURE() << "a Simple horizon was accepted";
            }
            catch (const std::invalid_argument& e)
            {
                EXPECT_NE(std::string(e.what()).find("mymyr: encoding: the device does not encode Simple horizons"),
                          std::string::npos)
                    << e.what();
            }
        }
        // action relations and action targets of rows without supplied expansions
        options.horizon = {};
        options.include_actions = true;
        options.targets = {encoding::TargetSource::Action};
        const auto encoder = encoding::FlatEncoder::create(plain, options);
        EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(encoder->schema()));
    }
}
