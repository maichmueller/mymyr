// The device encoder against the host encoder, byte for byte: every supported configuration, both layouts and index types,
// batch sizes around the launch widths, tables of mixed instances and widths, adversarial rows, every launch
// configuration, the capacity form, invalid rows, and the encoding inside a CUDA graph. MYMYR_TEST_FULL_SUITE=1 runs
// the full list of batch sizes.

#include "../cpp/encoding/support.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/cuda/encoding.hpp"
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

rl::TaskTablePtr table_of(const std::string& domain, const std::vector<std::string>& problems)
{
    std::vector<TaskPtr> tasks;
    for (const std::string& problem : problems)
        tasks.push_back(test::pddl_task(domain, problem));
    return rl::TaskTable::create(tasks);
}

/// Instances of 1, 42 and 184 words: `link` over n objects has n * n slots, `flag` n more, and `armed` one.
rl::TaskTablePtr wide_table()
{
    const auto domain = frontend::Domain::from_string(
        "(define (domain wide) (:requirements :strips)"
        " (:predicates (link ?x ?y) (flag ?x) (armed) (fixed ?x ?y))"
        " (:action connect :parameters (?x ?y) :precondition (and) :effect (link ?x ?y))"
        " (:action mark :parameters (?x) :precondition (and) :effect (flag ?x))"
        " (:action arm :parameters () :precondition (and) :effect (armed)))");
    std::vector<TaskPtr> tasks;
    for (const u32 n : {5u, 51u, 108u})
    {
        std::string objects, init = "(fixed o0 o1) (fixed o1 o2) (armed)";
        for (u32 i = 0; i < n; ++i)
            objects += " o" + std::to_string(i);
        const std::string problem = "(define (problem p" + std::to_string(n) + ") (:domain wide) (:objects" + objects +
                                    ") (:init " + init + ") (:goal (and (link o0 o1) (flag o2) (armed))))";
        tasks.push_back(Task::create(*domain->instantiate_string(problem)));
    }
    return rl::TaskTable::create(tasks);
}

// ------------------------------------------------------------------------------------------------ rows

/// The rows of a batch over a table with everything an encode call can take besides the options.
struct Batch
{
    rl::TaskTablePtr table;
    u64 count = 0;
    u32 words = 0, derived_words = 0;
    std::vector<u64> states;  // [count, words]
    std::vector<i32> task_ids;
    std::vector<u64> derived;  // [count, derived_words], the axiom evaluation of each row; empty without derived atoms
    std::vector<u64> goal_pos, goal_neg;                  // [count, words] masks over a subset of the row's fluent slots
    std::vector<u64> goal_derived_pos, goal_derived_neg;  // [count, derived_words]
};

void set_bit(u64* row, u32 slot) { row[slot >> 6] |= u64{1} << (slot & 63); }
bool test_bit(const u64* row, u32 slot) { return (row[slot >> 6] >> (slot & 63)) & 1u; }

/// All zeros, all ones within the instance's slots, single bits at 63, 64 and 65 and at the last slot, and the last
/// partial word set.
std::vector<std::vector<u64>> adversarial_rows(const rl::TaskTable& table, u32 instance)
{
    const u32 slots = table.task(instance)->atoms().fluent_slots();
    std::vector<std::vector<u64>> rows;
    auto blank = [&] { return std::vector<u64>(table.words(), 0); };
    rows.push_back(blank());
    {
        auto row = blank();
        for (u32 s = 0; s < slots; ++s)
            set_bit(row.data(), s);
        rows.push_back(row);
    }
    for (const u32 bit : {63u, 64u, 65u, slots ? slots - 1 : 0u})
        if (bit < slots)
        {
            auto row = blank();
            set_bit(row.data(), bit);
            rows.push_back(row);
        }
    if (slots)
    {
        auto row = blank();
        for (u32 s = ((slots - 1) / 64) * 64; s < slots; ++s)
            set_bit(row.data(), s);
        rows.push_back(row);
    }
    return rows;
}

/// The derived words of a state row by a fresh axiom evaluation, as the host encoder does it.
std::vector<u64> derived_words_of(const rl::TaskTable& table, u32 instance, const u64* row, u32 width)
{
    const Task& task = *table.task(instance);
    std::vector<u64> out(width, 0);
    const WorkspaceLease workspace = task.workspace();
    auto& engine = workspace->engine();
    engine.set_state(row, table.instance(instance).words);
    engine.build_view();
    workspace->axioms().evaluate();
    std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), out.begin());
    return out;
}

/// `count` rows: the adversarial rows of every instance first, then rows interleaved by instance: random walks from
/// the initial states (`walk`) or random bit patterns of mixed density with garbage in the padding words past a narrower
/// instance's words.
Batch make_batch(const rl::TaskTablePtr& table, u64 count, bool walk, u64 seed)
{
    Batch b;
    b.table = table;
    b.words = table->words();
    b.derived_words = table->derived_words();
    const u32 instances = table->size();
    std::mt19937_64 rng(seed);
    std::vector<test::Rows> walks;
    if (walk)
        for (u32 i = 0; i < instances; ++i)
            walks.push_back(test::random_walk(*table, i, count / instances + 1, seed + i));
    std::vector<std::vector<u64>> special;
    std::vector<i32> special_ids;
    for (u32 i = 0; i < instances; ++i)
        for (auto& row : adversarial_rows(*table, i))
        {
            special.push_back(std::move(row));
            special_ids.push_back(static_cast<i32>(i));
        }
    for (u64 k = 0; k < count; ++k)
    {
        std::vector<u64> row(b.words, 0);
        i32 id;
        if (k < special.size())
        {
            row = special[k];
            id = special_ids[k];
        }
        else
        {
            const u64 j = k - special.size();
            id = static_cast<i32>(j % instances);
            const u32 slots = table->task(static_cast<u32>(id))->atoms().fluent_slots();
            if (walk)
            {
                const u64* from = walks[static_cast<u32>(id)].data.data() + (j / instances) * b.words;
                std::copy(from, from + b.words, row.begin());
            }
            else
            {
                const double density = std::array{0.002, 0.01, 0.05}[rng() % 3];
                for (u32 s = 0; s < slots; ++s)
                    if (std::uniform_real_distribution<double>(0, 1)(rng) < density)
                        set_bit(row.data(), s);
                if (rng() % 2)
                    for (u32 w = table->instance(static_cast<u32>(id)).words; w < b.words; ++w)
                        row[w] = rng();
            }
        }
        b.states.insert(b.states.end(), row.begin(), row.end());
        b.task_ids.push_back(id);
    }
    b.count = count;
    // derived words and goals of the rows
    std::vector<u64> goal_pos(count * b.words, 0), goal_neg(count * b.words, 0);
    std::vector<u64> goal_derived_pos(count * b.derived_words, 0), goal_derived_neg(count * b.derived_words, 0);
    b.derived.assign(count * b.derived_words, 0);
    for (u64 g = 0; g < count; ++g)
    {
        const u32 instance = static_cast<u32>(b.task_ids[g]);
        const Task& task = *table->task(instance);
        const u64* row = b.states.data() + g * b.words;
        for (u32 s = 0; s < task.atoms().fluent_slots(); ++s)
        {
            if (rng() % 5)
                continue;
            set_bit(test_bit(row, s) ? goal_pos.data() + g * b.words : goal_neg.data() + g * b.words, s);
        }
        if (b.derived_words)
        {
            const auto derived = derived_words_of(*table, instance, row, b.derived_words);
            std::copy(derived.begin(), derived.end(), b.derived.begin() + static_cast<std::ptrdiff_t>(g * b.derived_words));
            for (u32 s = 0; s < task.atoms().derived_slots(); ++s)
            {
                if (rng() % 3)
                    continue;
                set_bit(test_bit(derived.data(), s) ? goal_derived_pos.data() + g * b.derived_words
                                                    : goal_derived_neg.data() + g * b.derived_words,
                        s);
            }
        }
    }
    b.goal_pos = std::move(goal_pos);
    b.goal_neg = std::move(goal_neg);
    b.goal_derived_pos = std::move(goal_derived_pos);
    b.goal_derived_neg = std::move(goal_derived_neg);
    return b;
}

/// Which optional inputs a call has.
struct Inputs
{
    bool derived = true;
    bool row_goals = false;
    bool derived_goals = false;
};

encoding::EncodeInput host_input(const Batch& b, u64 n, const Inputs& inputs)
{
    encoding::EncodeInput in;
    in.states = {b.states.data(), n, b.words, 0, 0};
    in.task_ids = b.task_ids.data();
    if (inputs.derived && b.derived_words)
    {
        in.derived = b.derived.data();
        in.derived_stride = b.derived_words;
    }
    if (inputs.row_goals)
    {
        in.goal_pos = b.goal_pos.data();
        in.goal_neg = b.goal_neg.data();
        in.goal_stride = b.words;
        in.goal_words = b.words;
        if (inputs.derived_goals && b.derived_words)
        {
            in.goal_derived_pos = b.goal_derived_pos.data();
            in.goal_derived_neg = b.goal_derived_neg.data();
            in.goal_derived_stride = b.derived_words;
            in.goal_derived_words = b.derived_words;
        }
    }
    return in;
}

/// The rows of a batch in device memory.
struct DeviceRows
{
    cuda::DeviceBuffer states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg;

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
                        const encoding::FlatCapacities* caps = nullptr, const cuda::LaunchConfig& launch = {})
{
    DeviceResult result;
    result.layout = device.output_layout(shape);
    cuda::DeviceBuffer block(ctx, result.layout.bytes, stream);
    cuda::check(cudaMemsetAsync(block.data(), 0, block.size(), stream), "clear the output");
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
                            {"target_group", l.target_group, a.targets}};
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

cuda::BatchShape shape_of(u64 graphs, const encoding::FlatSizes& s) { return {graphs, s.entities, s.args, s.targets}; }

// ------------------------------------------------------------------------------------------------ configurations

struct Config
{
    const char* name;
    encoding::FlatOptions options;
    Inputs inputs;
};

using encoding::GoalDerivation;
using encoding::TargetSource;

/// The configurations the device encodes. Rows for the first one carry derived words whenever the table has derived atoms.
std::vector<Config> configs()
{
    const encoding::GoalDerivations all = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    std::vector<Config> out;
    auto add = [&](const char* name, auto&& change, Inputs inputs = {})
    {
        encoding::FlatOptions o = test::encoder_options();
        change(o);
        out.push_back({name, o, inputs});
    };
    add("state", [](auto&) {});
    add("goal_plain_satisfied", [](auto& o) { o.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied}; });
    add("goal_all_targets",
        [&](auto& o)
        {
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
        });
    add("virtual_nodes", [](auto& o) { o.predicate_virtual_nodes = true; });
    add("virtual_goal_targets",
        [&](auto& o)
        {
            o.predicate_virtual_nodes = true;
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
        });
    add("no_nullary_equality",
        [](auto& o)
        {
            o.nullary = false;
            o.equality = true;
        });
    add("no_static_no_derived",
        [](auto& o)
        {
            o.include_static = false;
            o.include_derived = false;
        });
    add("implicit_entities_no_counts",
        [](auto& o)
        {
            o.emit_counts = false;
            o.entity_arrays = encoding::EntityArrays::Auto;
        });
    add("row_goals", [&](auto& o) { o.goal_derivations = all; }, Inputs{true, true, false});
    add("row_goals_derived_targets_virtual",
        [&](auto& o)
        {
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
            o.predicate_virtual_nodes = true;
        },
        Inputs{true, true, true});
    return out;
}

/// The batch sizes around the widths of the launches: one graph, a warp, a block and a few blocks.
std::vector<u64> batch_sizes()
{
    if (full_suite())
        return {1, 2, 31, 32, 33, 255, 256, 257, 4097, 16384};
    return {1, 33, 257};
}

struct Corpus
{
    const char* name;
    rl::TaskTablePtr table;
    bool walk;
};

std::vector<Corpus> corpus()
{
    std::vector<Corpus> out;
    out.push_back({"blocks", table_of("blocks", {"smedium", "probBLOCKS-8-0"}), true});
    out.push_back({"features", table_of("features", {"p01", "p02", "p03", "p04"}), true});
    out.push_back({"goals", table_of("goals", {"conjunctive", "forall", "or", "exists", "not_and", "static_true"}), true});
    out.push_back({"philosophers", table_of("philosophers", {"p03-phil4", "p03-phil4-one-blocked"}), true});
    out.push_back({"gripper", table_of("gripper", {"p01", "p02"}), true});
    out.push_back({"wide", wide_table(), false});
    return out;
}

encoding::FlatOptions with(encoding::FlatOptions o, encoding::Layout layout, encoding::IndexType index)
{
    o.layout = layout;
    o.index = index;
    return o;
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
        const Batch batch = make_batch(c.table, max_n, c.walk, 13);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : configs())
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
        const Batch batch = make_batch(c.table, n, c.walk, 17);
        const DeviceRows rows(ctx, stream, batch);
        // a rich configuration: goals with targets, virtual nodes, row goals
        const Config config = configs().back();
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        {
            SCOPED_TRACE(layout == encoding::Layout::Graph ? "graph-major" : "relation-major");
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
            const encoding::FlatCapacities fits{exact.entities, exact.tuples, exact.args, exact.targets};
            struct Case
            {
                const char* name;
                encoding::FlatCapacities caps;
            };
            const Case cases[] = {
                {"exact", fits},
                {"roomy", {exact.entities + 7, exact.tuples + 3, exact.args + 5, exact.targets + 9}},
                {"short entities", {exact.entities - 1, exact.tuples, exact.args, exact.targets}},
                {"short tuples", {exact.entities, exact.tuples - 1, exact.args, exact.targets}},
                {"short args", {exact.entities, exact.tuples, exact.args - 1, exact.targets}},
                {"short targets", {exact.entities, exact.tuples, exact.args, exact.targets - 1}},
                {"zero", {0, 0, 0, 0}},
            };
            for (const Case& k : cases)
            {
                SCOPED_TRACE(k.name);
                encoding::FlatSizes host_sizes;
                const auto host = encoder->encode_capacity(host_in, k.caps, pool, scratch, host_sizes);
                // the block holds the capacities' entries (zero beyond the batch), and the host's too
                const cuda::BatchShape shape{n, k.caps.entities, k.caps.args, k.caps.targets};
                const DeviceResult result = run_device(device, ctx, stream, device_in, shape, &k.caps);
                expect_sizes(host_sizes, result.sizes);
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
    const DeviceResult result = run_device(device, ctx, stream, device_in, shape);
    EXPECT_NE(result.sizes.overflow & encoding::k_overflow_input, 0u);
    // nothing of the batch is written
    const cuda::OutputLayout& l = result.layout;
    for (u64 i = 0; i < l.sizes; ++i)
        ASSERT_EQ(result.block[i], std::byte{0}) << "byte " << i;
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

TEST(DeviceEncode, ACapturedEncodeReplaysToTheEagerResultInAtMostFourKernels)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    const u64 n = 600;
    for (const Corpus& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const Batch batch = make_batch(c.table, n, c.walk, 41);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : {configs()[1], configs()[4], configs().back()})
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
                EXPECT_LE(nodes.kernels, 4u);
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
    device.count(work, in, static_cast<encoding::FlatSizes*>(buffer.data()), launch);
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
        const Batch batch = make_batch(c.table, n, c.walk, 53);
        const DeviceRows rows(ctx, stream, batch);
        for (const Config& config : configs())
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
}  // namespace
