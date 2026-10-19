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
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
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
                            {"relation_offsets", l.relation_offsets, true},
                            {"relation_arg_offsets", l.relation_arg_offsets, true},
                            {"relation_args", l.relation_args, true},
                            {"goal_complete", l.goal_complete, a.goal_complete},
                            {"target_offsets", l.target_offsets, a.targets},
                            {"target_entity", l.target_entity, a.targets},
                            {"target_group", l.target_group, a.targets},
                            {"history_offsets", l.history_offsets, a.history},
                            {"history_entity", l.history_entity, a.history},
                            {"history_dt", l.history_dt, a.history}};
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

// ------------------------------------------------------------------------------------------------ configurations

/// The batch sizes around the widths of the launches: one graph, a warp, a block and a few blocks.
std::vector<u64> batch_sizes()
{
    if (full_suite())
        return {1, 2, 31, 32, 33, 255, 256, 257, 4097, 16384};
    return {1, 33, 257, 4097};
}

// ------------------------------------------------------------------------------------------------ the matrix

TEST(DeviceEncode, EqualsTheHostEncoderBitForBit)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const std::vector<u64> sizes = batch_sizes();
    const u64 max_n = *std::max_element(sizes.begin(), sizes.end());
    u64 compared = 0;
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
                    device.reserve(cuda::BatchShape{max_n, 0, 0, 0}, stream);
                    for (const u64 n : sizes)
                    {
                        SCOPED_TRACE("N = " + std::to_string(n));
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
                    for (const auto fill : {cuda::FillMode::Auto, cuda::FillMode::Thread, cuda::FillMode::Warp})
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

TEST(DeviceEncode, DerivedWordsAreRequiredWhereAGraphReadsThem)
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
    // the host evaluates the axioms itself; the device refuses a call without the derived words
    const Inputs without{false, false, false};
    EXPECT_THROW((void)run_device(device, ctx, stream, rows.input(batch, 16, without), {16, 1, 1, 1}),
                 std::invalid_argument);
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
                const cuda::DeviceOutput out = cuda::output_of(layout_of, shape, block.data());
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
                EXPECT_LE(nodes.kernels, 8u);  // at most two fill classes, four kernels per class
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
                    for (const auto fill : {cuda::FillMode::Thread, cuda::FillMode::Warp})
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
        EXPECT_THROW(device.encode(context, scratch.data(), bytes - 1, shape, input, out), std::invalid_argument);
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

TEST(DeviceEncode, MixedWidthsUseBothFillClassesAndKeepBatchOrder)
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
        u32 kernels = 0, thread_fills = 0, warp_fills = 0;
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
            warp_fills += params.gridDim.x == (n * 32 + 127) / 128;
        }
        EXPECT_LE(kernels, 8u);
        EXPECT_EQ(thread_fills, 1u);
        EXPECT_EQ(warp_fills, 1u);
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
    const u64 bytes = device.scratch_bytes(shape);
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
