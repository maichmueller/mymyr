// Device context bytes, generation ownership, per-stream scratch and checked batch sizes.

#include "../cpp/encoding/support.hpp"
#include "context_probe.hpp"
#include "mymyr/cuda/encoding.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <thread>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace mymyr;
namespace fs = std::filesystem;

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

/// A PDDL file under tests/data.
std::optional<fs::path> find_file(const std::string& relative)
{
    for (const fs::path& root : {fs::path(MYMYR_TEST_DATA_DIR), fs::path(MYMYR_TEST_DATA_DIR) / "benchmarks/classic"})
        if (fs::exists(root / relative))
            return root / relative;
    return std::nullopt;
}

/// The tasks of one domain directory under tests/data (`domain` is relative to it, the problems
/// are file names next to it); empty if a file is missing.
std::vector<TaskPtr> load_tasks(const std::string& directory, const std::vector<std::string>& problems)
{
    std::vector<TaskPtr> tasks;
    const auto domain = find_file(directory + "/domain.pddl");
    if (!domain)
        return {};
    for (const std::string& problem : problems)
    {
        const auto file = find_file(directory + "/" + problem + ".pddl");
        if (!file)
            return {};
        tasks.push_back(Task::create(*frontend::load_task(domain->string(), file->string())));
    }
    return tasks;
}

struct Group
{
    std::string name;
    std::string directory;
    std::vector<std::string> problems;
};

/// The fixture tasks of the encoder tests and the benchmark tasks, grouped by domain: one table per group.
const std::vector<Group>& groups()
{
    static const std::vector<Group> g = {
        {"blocks", "pddl/blocks", {"smedium", "probBLOCKS-8-0"}},
        {"logistics", "pddl/logistics00", {"probLOGISTICS-6-1"}},
        {"childsnack", "ipc/childsnack-ipc/test", {"p01-easy"}},
        {"philosophers", "pddl/philosophers", {"p03-phil4"}},
        {"rovers", "ipc/rovers-ipc/test", {"p01-easy"}},
        {"satellite", "ipc/satellite-ipc/test", {"p01-easy"}},
        {"blocks_three_sizes", "ipc/blocksworld-ipc/test", {"p01-easy", "p01-medium", "p20-easy"}},
        {"childsnack_larger", "ipc/childsnack-ipc/test", {"p20-easy"}},
        {"rovers_larger", "ipc/rovers-ipc/test", {"p20-easy"}},
        {"sokoban_larger", "ipc/sokoban-ipc/test", {"p20-easy"}},
        {"transport_larger", "ipc/transport-ipc/test", {"p30-easy"}},
        {"miconic_larger", "ipc/miconic-ipc/test", {"p16-medium"}},
    };
    return g;
}

std::shared_ptr<const encoding::EncodingContext> host_context(const rl::TaskTablePtr& table,
                                                              const encoding::FlatOptions& options = test::encoder_options())
{
    return encoding::EncodingContext::create(table, encoding::EncodingSchema::create(*table, options));
}

std::vector<std::byte> host_bytes(const encoding::EncodingContext& context)
{
    const rl::ArrayBundle& arrays = context.arrays();
    return {arrays.block(), arrays.block() + arrays.bytes()};
}

/// The probe of every instance of `view`, run on `stream` and read back.
std::vector<u64> device_probe(const encoding::detail::ContextView& view, u32 instances, const cuda::ContextPtr& ctx,
                              cudaStream_t stream)
{
    cuda::DeviceBuffer out(ctx, std::max<u64>(u64{instances} * test::k_probe_width * 8, 8), stream);
    test::launch_context_probe(view, instances, static_cast<u64*>(out.data()), stream);
    std::vector<u64> host(u64{instances} * test::k_probe_width);
    if (instances)
        cuda::check(cudaMemcpyAsync(host.data(), out.data(), host.size() * 8, cudaMemcpyDeviceToHost, stream), "D2H");
    cuda::check(cudaStreamSynchronize(stream), "sync");
    return host;
}

std::vector<u64> host_probe(const encoding::detail::ContextView& view, u32 instances)
{
    std::vector<u64> host(u64{instances} * test::k_probe_width);
    for (u32 i = 0; i < instances; ++i)
        test::probe_instance(view, i, host.data() + u64{i} * test::k_probe_width);
    return host;
}

u64 round_up(u64 x) { return (x + 63) / 64 * 64; }

// ------------------------------------------------------------------------------------------------ the block

TEST(DeviceEncodingContext, DownloadedBlockEqualsTheHostBlock)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream other;  // outlives the contexts it acquires
    u32 checked = 0;
    for (const Group& group : groups())
    {
        SCOPED_TRACE(group.name);
        const std::vector<TaskPtr> tasks = load_tasks(group.directory, group.problems);
        if (tasks.empty())
        {
            std::cerr << "skipped " << group.name << ": its PDDL is missing\n";
            continue;
        }
        const auto table = rl::TaskTable::create(tasks);
        const auto host = host_context(table);
        const auto device = cuda::DeviceEncodingContext::upload(ctx, host);
        EXPECT_EQ(device->uploads(), 1u);
        EXPECT_EQ(device->bytes(), host->arrays().bytes());
        EXPECT_EQ(device->num_instances(), table->size());
        EXPECT_TRUE(device->download() == host_bytes(*host));
        // a second stream sees the block after acquire()
        EXPECT_TRUE(device->download(other) == host_bytes(*host));
        EXPECT_EQ(device->uploads(), 1u);
        ++checked;
    }
    EXPECT_EQ(checked, groups().size());
}

TEST(DeviceEncodingContext, ABlockWithDerivedAtomsIsCopiedToo)
{
    SKIP_WITHOUT_GPU();
    const auto task = test::pddl_task("features", "p01");
    const auto table = rl::TaskTable::single(task);
    const auto host = host_context(table);
    ASSERT_GT(host->max_derived_words(), 0u);
    const auto device = cuda::DeviceEncodingContext::upload(device_context(), host);
    EXPECT_TRUE(device->download() == host_bytes(*host));
}

// ------------------------------------------------------------------------------------------------ append

struct Grown
{
    rl::TaskTablePtr small, grown;
    std::shared_ptr<const encoding::EncodingContext> before, after;
};

std::optional<Grown> blocks_grown()
{
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0", "smedium"});
    if (tasks.empty())
        return std::nullopt;
    Grown g;
    g.small = rl::TaskTable::create({tasks[0]});
    g.grown = rl::TaskTable::create(tasks);
    const auto schema = encoding::EncodingSchema::create(*g.small, test::encoder_options());
    g.before = encoding::EncodingContext::create(g.small, schema);
    g.after = encoding::EncodingContext::create(g.grown, schema);
    return g;
}

TEST(DeviceEncodingContext, AppendingInstancesKeepsTheOffsetsAndBytesOfTheExistingOnes)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    if (!g)
        GTEST_SKIP() << "no PDDL data";
    const auto ctx = device_context();
    const auto device = cuda::DeviceEncodingContext::upload(ctx, g->before);
    const std::vector<std::byte> first = device->download();
    const auto first_view = g->before->view();
    ASSERT_TRUE(first == host_bytes(*g->before));
    EXPECT_EQ(device->uploads(), 1u);

    EXPECT_TRUE(device->refresh(g->after));
    EXPECT_EQ(device->uploads(), 2u);
    EXPECT_EQ(device->num_instances(), 3u);
    const std::vector<std::byte> second = device->download();
    EXPECT_TRUE(second == host_bytes(*g->after));

    // every array of the first context is a prefix of the same array of the second, the instance table included
    for (const rl::ArrayInfo& a : g->before->arrays().arrays())
    {
        const rl::ArrayInfo& b = *g->after->arrays().find(a.name);
        ASSERT_GE(b.elements(), a.elements()) << a.name;
        const u64 bytes = a.elements() * rl::dtype_bytes(a.dtype);
        EXPECT_EQ(std::memcmp(first.data() + a.offset, second.data() + b.offset, bytes), 0) << a.name;
    }
    // the rows of the first instance in the instance table are unchanged, so are the rows after them for a repeat
    const auto v0 = host_probe(first_view, 1);
    const auto v1 = host_probe(g->after->view(), 3);
    EXPECT_TRUE(std::equal(v0.begin(), v0.end(), v1.begin()));
    EXPECT_TRUE(std::equal(v0.begin(), v0.end(), v1.begin() + 2 * test::k_probe_width));  // the third is the first again
    const auto lease = device->acquire();
    EXPECT_EQ(device_probe(lease.view, 3, ctx, ctx->stream()), v1);
}

TEST(DeviceEncodingContext, ARefreshWithoutNewInstancesUploadsNothing)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    if (!g)
        GTEST_SKIP() << "no PDDL data";
    const auto device = cuda::DeviceEncodingContext::upload(device_context(), g->after);
    EXPECT_FALSE(device->refresh(g->after));
    EXPECT_FALSE(device->refresh(host_context(g->grown)));  // equal bytes, another object
    EXPECT_EQ(device->uploads(), 1u);
}

TEST(DeviceEncodingContext, OneUploadPerRefresh)
{
    SKIP_WITHOUT_GPU();
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0", "smedium", "probBLOCKS-8-0"});
    if (tasks.empty())
        GTEST_SKIP() << "no PDDL data";
    const auto schema = encoding::EncodingSchema::create(*rl::TaskTable::create({tasks[0]}), test::encoder_options());
    auto context_of = [&](u32 n)
    { return encoding::EncodingContext::create(rl::TaskTable::create({tasks.begin(), tasks.begin() + n}), schema); };
    const auto device = cuda::DeviceEncodingContext::upload(device_context(), context_of(1));
    EXPECT_EQ(device->uploads(), 1u);
    for (u32 n = 2; n <= 4; ++n)
    {
        EXPECT_TRUE(device->refresh(context_of(n)));
        EXPECT_EQ(device->uploads(), n);
        EXPECT_EQ(device->num_instances(), n);
    }
    EXPECT_TRUE(device->download() == host_bytes(*device->host()));
}

TEST(DeviceEncodingContext, OnlyAnExtensionIsAppended)
{
    SKIP_WITHOUT_GPU();
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0"});
    if (tasks.empty())
        GTEST_SKIP() << "no PDDL data";
    const auto schema = encoding::EncodingSchema::create(*rl::TaskTable::create(tasks), test::encoder_options());
    auto context_of = [&](std::vector<TaskPtr> t) { return encoding::EncodingContext::create(rl::TaskTable::create(std::move(t)), schema); };
    const auto device = cuda::DeviceEncodingContext::upload(device_context(), context_of({tasks[0], tasks[1]}));
    // the instances in another order, fewer instances
    EXPECT_THROW(device->refresh(context_of({tasks[1], tasks[0]})), std::invalid_argument);
    EXPECT_THROW(device->refresh(context_of({tasks[0]})), std::invalid_argument);
    EXPECT_THROW(device->refresh(nullptr), std::invalid_argument);
    // another schema
    encoding::FlatOptions other = test::encoder_options();
    other.nullary = false;
    const auto table = rl::TaskTable::create({tasks[0], tasks[1], tasks[0]});
    EXPECT_THROW(device->refresh(host_context(table, other)), std::invalid_argument);
    EXPECT_EQ(device->uploads(), 1u);
    EXPECT_TRUE(device->download() == host_bytes(*device->host()));
}

// ------------------------------------------------------------------------------------------------ views

TEST(DeviceEncodingContext, DeviceViewsReadTheValuesOfTheHostViews)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    u32 checked = 0;
    for (const Group& group : groups())
    {
        SCOPED_TRACE(group.name);
        const std::vector<TaskPtr> tasks = load_tasks(group.directory, group.problems);
        if (tasks.empty())
            continue;
        const auto table = rl::TaskTable::create(tasks);
        const auto host = host_context(table);
        const auto device = cuda::DeviceEncodingContext::upload(ctx, host);
        const auto lease = device->acquire();
        const auto dv = lease.view;
        EXPECT_EQ(dv.num_instances, table->size());
        EXPECT_EQ(dv.max_arity, host->max_arity());
        EXPECT_EQ(dv.num_relations, host->schema().num_relations());
        // the device pointers are the host pointers rebased onto the copy
        const auto hv = host->view();
        EXPECT_EQ(reinterpret_cast<const std::byte*>(dv.cells) - static_cast<const std::byte*>(lease.generation->data()),
                  reinterpret_cast<const std::byte*>(hv.cells) - host->arrays().block());
        EXPECT_EQ(device_probe(dv, table->size(), ctx, ctx->stream()), host_probe(hv, table->size()));
        ++checked;
    }
    EXPECT_EQ(checked, groups().size());
}

TEST(DeviceEncodingContext, AViewTakenBeforeARefreshReadsOnForTheWorkEnqueuedBefore)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    if (!g)
        GTEST_SKIP() << "no PDDL data";
    const auto ctx = device_context();
    const auto device = cuda::DeviceEncodingContext::upload(ctx, g->before);
    const cudaStream_t s = ctx->stream();
    const auto lease = device->acquire(s);
    cuda::DeviceBuffer out(ctx, test::k_probe_width * 8, s);
    test::launch_context_probe(lease.view, 1, static_cast<u64*>(out.data()), s);
    EXPECT_TRUE(device->refresh(g->after, s));  // the old block frees after the probe
    std::vector<u64> probe(test::k_probe_width);
    cuda::check(cudaMemcpyAsync(probe.data(), out.data(), probe.size() * 8, cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    EXPECT_EQ(probe, host_probe(g->before->view(), 1));
}

// ------------------------------------------------------------------------------------------------ lifetime

TEST(DeviceEncodingContext, GenerationsAreHeldAcrossRefresh)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    ASSERT_TRUE(g);
    const auto ctx = device_context();
    const auto device = cuda::DeviceEncodingContext::upload(ctx, g->before);
    std::weak_ptr<const cuda::DeviceBuffer> old;
    {
        auto lease = device->acquire();
        old = lease.generation;
        const void* address = lease.generation->data();
        EXPECT_TRUE(device->refresh(g->after));
        const auto next = device->acquire();
        EXPECT_EQ(lease.number, 1u);
        EXPECT_EQ(next.number, 2u);
        EXPECT_EQ(lease.view.num_instances, 1u);
        EXPECT_EQ(next.view.num_instances, 3u);
        EXPECT_EQ(old.lock()->data(), address);
        EXPECT_NE(next.generation->data(), address);
    }
    EXPECT_TRUE(old.expired());
    ctx->synchronize();
}

/// A host callback holds a stream without reading any device memory.
class StreamGate
{
public:
    explicit StreamGate(cudaStream_t stream) : m_stream(stream)
    {
        cuda::check(cudaLaunchHostFunc(stream, [](void* p)
        {
            auto& gate = *static_cast<StreamGate*>(p);
            std::unique_lock lock(gate.m_mutex);
            gate.m_changed.wait(lock, [&] { return gate.m_open; });
        }, this), "cudaLaunchHostFunc");
    }
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

void expect_blocked(const cuda::ContextPtr& ctx)
{
    cuda::Event done;
    done.record(ctx->stream());
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!done.done() && std::chrono::steady_clock::now() < end)
        std::this_thread::yield();
    EXPECT_FALSE(done.done());
    EXPECT_EQ(cudaStreamQuery(ctx->stream()), cudaErrorNotReady);
}

TEST(DeviceEncodingContext, FreeWaitsForTheAcquiredStream)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    ASSERT_TRUE(g);
    const auto ctx = device_context();
    cuda::Stream stream;
    auto device = cuda::DeviceEncodingContext::upload(ctx, g->after);
    ctx->synchronize();
    {
        StreamGate gate(stream);
        { auto lease = device->acquire(stream); }
        device.reset();
        expect_blocked(ctx);
    }
    ctx->synchronize();
}

TEST(DeviceEncodingContext, CallerUploadAndRefreshStreamsMayBeDestroyedFirst)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    ASSERT_TRUE(g);
    const auto ctx = device_context();
    std::shared_ptr<cuda::DeviceEncodingContext> device;
    {
        cuda::Stream stream;
        device = cuda::DeviceEncodingContext::upload(ctx, g->before, stream);
        EXPECT_TRUE(device->refresh(g->after, stream));
    }
    EXPECT_EQ(device->download(), host_bytes(*g->after));
    device.reset();
    ctx->synchronize();
}

TEST(DeviceEncodingContext, DestroyedRightAfterTheUpload)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    if (!g)
        GTEST_SKIP() << "no PDDL data";
    const auto ctx = device_context();
    for (int i = 0; i < 20; ++i)
        cuda::DeviceEncodingContext::upload(ctx, g->after);
    ctx->synchronize();
}

// ------------------------------------------------------------------------------------------------ sizes

TEST(DeviceEncoder, OutputLayoutPinsTheBytesOfEveryArray)
{
    // G = 10 graphs, E = 100 entities, A = 1000 argument entries, R = 7 relations
    const cuda::BatchShape shape{10, 100, 1000};
    const cuda::OutputLayout i32 = cuda::output_layout(shape, encoding::IndexType::I32, 7);
    // 4 G = 40, I G = 40, I (G + 1) = 44, E = 100, I E = 400, I G R = 280, I (G R + 1) = 284 (twice), I A = 4000,
    // each rounded up to 64
    EXPECT_EQ(i32.task_ids, 0u);
    EXPECT_EQ(i32.num_objects, 64u);
    EXPECT_EQ(i32.entity_offsets, 128u);
    EXPECT_EQ(i32.entity_role, 192u);
    EXPECT_EQ(i32.entity_ref, 320u);
    EXPECT_EQ(i32.relation_counts, 768u);
    EXPECT_EQ(i32.relation_offsets, 1088u);
    EXPECT_EQ(i32.relation_arg_offsets, 1408u);
    EXPECT_EQ(i32.relation_args, 1728u);
    EXPECT_EQ(i32.bytes, 5760u);
    const cuda::OutputLayout i64 = cuda::output_layout(shape, encoding::IndexType::I64, 7);
    // 40, 80, 88, 100, 800, 560, 568, 568, 8000
    EXPECT_EQ(i64.num_objects, 64u);
    EXPECT_EQ(i64.entity_offsets, 192u);
    EXPECT_EQ(i64.entity_role, 320u);
    EXPECT_EQ(i64.entity_ref, 448u);
    EXPECT_EQ(i64.relation_counts, 1280u);
    EXPECT_EQ(i64.relation_offsets, 1856u);
    EXPECT_EQ(i64.relation_arg_offsets, 2432u);
    EXPECT_EQ(i64.relation_args, 3008u);
    EXPECT_EQ(i64.bytes, 11008u);
    // an empty batch has the offsets arrays of one entry
    EXPECT_EQ(cuda::output_layout({}, encoding::IndexType::I32, 7).bytes, 64u * 3);
}

TEST(DeviceEncoder, ScratchUsesThreeInPlaceScans)
{
    SKIP_WITHOUT_GPU();
    for (const u64 graphs : {u64{1}, u64{10}, u64{1000}, u64{16384}})
        for (const u64 cells : {u64{0}, u64{6}, u64{170}})
        {
            const cuda::GraphBounds bounds{9, 97, 161, cells};
            const auto narrow = cuda::scratch_layout({graphs, 9 * graphs, 161 * graphs}, bounds, encoding::Layout::Graph);
            EXPECT_FALSE(narrow.wide);
            EXPECT_EQ(narrow.tuples, 4 * (graphs + 1));
            EXPECT_EQ(narrow.args, 8 * (graphs + 1));
            EXPECT_EQ(narrow.cell_counts, round_up(12 * (graphs + 1)));
            EXPECT_EQ(narrow.scan_temp - narrow.cell_counts, round_up(4 * graphs * cells));
            EXPECT_EQ(narrow.bytes - narrow.scan_temp, round_up(narrow.scan_temp_bytes));
            if (graphs >= 64)
            {
                EXPECT_LE(narrow.scan_temp, 16 * graphs + round_up(4 * graphs * cells));
            }
            const auto wide = cuda::scratch_layout({graphs, u64{1} << 32, 0}, bounds, encoding::Layout::Graph);
            EXPECT_TRUE(wide.wide);
            EXPECT_EQ(wide.tuples, 8 * (graphs + 1));
            EXPECT_EQ(wide.args, 16 * (graphs + 1));
            EXPECT_EQ(wide.cell_counts, round_up(24 * (graphs + 1)));
        }
}

TEST(DeviceEncoder, OverflowAndUnsupportedLayoutsAreRejectedBeforeAllocation)
{
    const u64 huge = std::numeric_limits<u64>::max();
    EXPECT_THROW((void)cuda::output_layout({huge, 0, 0}, encoding::IndexType::I32, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({1, huge, 0}, encoding::IndexType::I64, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({1, 0, huge}, encoding::IndexType::I32, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({2, 0, 0}, encoding::IndexType::I64, huge), std::overflow_error);
    EXPECT_THROW((void)cuda::scratch_layout({}, {}, encoding::Layout::Relation), std::invalid_argument);
    EXPECT_THROW((void)cuda::scratch_layout({huge, 0, 0}, {}, encoding::Layout::Graph), std::overflow_error);
    EXPECT_THROW((void)cuda::scratch_layout({2, 0, 0}, {0, 0, 0, huge}, encoding::Layout::Graph), std::overflow_error);
}

std::shared_ptr<const encoding::FlatEncoder> blocks_encoder(encoding::IndexType index = encoding::IndexType::I32)
{
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0"});
    if (tasks.empty())
        return nullptr;
    encoding::FlatOptions options = test::encoder_options();
    options.index = index;
    return encoding::FlatEncoder::create(rl::TaskTable::create(tasks), options);
}

TEST(DeviceEncoder, BytesNeededOfTheWorstCaseBatch)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    if (!encoder)
        GTEST_SKIP() << "no PDDL data";
    const cuda::DeviceEncoder device(device_context(), encoder);
    const cuda::GraphBounds& b = device.bounds();
    // the bounds recomputed from the context's cells
    const auto view = encoder->context().view();
    cuda::GraphBounds expected;
    for (u32 i = 0; i < encoder->table().size(); ++i)
    {
        const auto inst = encoding::detail::instance_view(view, i);
        u64 tuples = 0, args = 0;
        bool nullary = false;
        for (u32 c = 0; c < inst.num_cells; ++c)
        {
            tuples += inst.cells[c].b - inst.cells[c].a;
            args += u64{inst.cells[c].b - inst.cells[c].a} * inst.cells[c].arity;
            nullary = nullary || inst.cells[c].logical_arity == 0;
        }
        expected.entities = std::max<u64>(expected.entities, inst.num_objects + nullary);
        expected.tuples = std::max(expected.tuples, tuples);
        expected.args = std::max(expected.args, args);
        expected.cells = std::max<u64>(expected.cells, inst.num_cells);
    }
    EXPECT_EQ(b.entities, expected.entities);
    EXPECT_EQ(b.tuples, expected.tuples);
    EXPECT_EQ(b.args, expected.args);
    EXPECT_EQ(b.cells, expected.cells);

    const u64 G = 1000, R = encoder->schema().num_relations();
    const cuda::BatchShape worst = device.worst_case(G);
    EXPECT_EQ(worst, (cuda::BatchShape{G, G * b.entities, G * b.args}));
    // scratch + output, with the output in four-byte indices
    const u64 output = round_up(4 * G) + round_up(4 * G) + round_up(4 * (G + 1)) + round_up(G * b.entities) +
                       round_up(4 * G * b.entities) + round_up(4 * G * R) + 2 * round_up(4 * (G * R + 1)) +
                       round_up(4 * G * b.args);
    EXPECT_EQ(device.bytes_needed(worst), device.scratch_layout(worst).bytes + output);
    EXPECT_EQ(device.bytes_needed(G), device.bytes_needed(worst));
    // a smaller batch needs less, a batch with fewer entities and arguments needs less
    EXPECT_LT(device.bytes_needed(G / 2), device.bytes_needed(G));
    EXPECT_LT(device.bytes_needed(cuda::BatchShape{G, 1, 1}), device.bytes_needed(G));
}

TEST(DeviceEncoder, PinnedBytesNeededOfBlocks)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    if (!encoder)
        GTEST_SKIP() << "no PDDL data";
    const cuda::DeviceEncoder device(device_context(), encoder);
    EXPECT_EQ(device.bounds().entities, 9u);
    EXPECT_EQ(device.bounds().tuples, 97u);
    EXPECT_EQ(device.bounds().args, 161u);
    EXPECT_EQ(device.bounds().cells, 6u);
    EXPECT_EQ(encoder->schema().num_relations(), 25u);
    const auto scratch = device.scratch_layout(device.worst_case(1000));
    EXPECT_EQ(device.bytes_needed(u64{1000}) - round_up(scratch.scan_temp_bytes), 1037312u);
}

TEST(DeviceEncoder, ReserveRetainsScratchAndOutputsBelongToTheCaller)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder(encoding::IndexType::I64);
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::DeviceEncoder device(ctx, encoder);
    EXPECT_EQ(device.reserved_bytes(), 0u);
    EXPECT_THROW((void)device.acquire(), std::invalid_argument);
    device.reserve(u64{4096});
    const auto lease = device.acquire();
    const auto address = lease.scratch->data();
    const u64 bytes = device.scratch_layout(device.reserved()).bytes;
    EXPECT_EQ(device.reserved_bytes(), bytes);
    EXPECT_EQ(device.bytes_needed(device.reserved()), bytes + device.reserved_layout().bytes);
    cuda::DeviceBuffer output(ctx, device.reserved_layout().bytes);
    EXPECT_NE(output.data(), address);
    device.reserve(u64{1});
    device.reserve(u64{4096});
    device.reserve(cuda::BatchShape{10, 1, 1});
    EXPECT_EQ(device.acquire().scratch->data(), address);
    EXPECT_EQ(device.reserved(), device.worst_case(4096));
    device.reserve(cuda::BatchShape{8192, 1, 1});
    EXPECT_EQ(device.reserved().graphs, 8192u);
    EXPECT_EQ(device.reserved().entities, device.worst_case(4096).entities);
    EXPECT_EQ(device.reserved_bytes(), device.scratch_layout(device.reserved()).bytes);
    EXPECT_NE(device.acquire().scratch->data(), address);
    EXPECT_EQ(lease.scratch->data(), address);
    ctx->synchronize();
}

TEST(DeviceEncoder, StreamsHaveIndependentScratch)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream first, second;
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(1000, first);
    device.reserve(2000, second);
    {
        auto one = device.acquire(first);
        auto two = device.acquire(second);
        EXPECT_NE(one.scratch->data(), two.scratch->data());
        EXPECT_EQ(one.context.generation, two.context.generation);
        EXPECT_EQ(one.shape.graphs, 1000u);
        EXPECT_EQ(two.shape.graphs, 2000u);
        cuda::check(cudaMemsetAsync(one.scratch->data(), 1, one.scratch->size(), first), "memset first");
        cuda::check(cudaMemsetAsync(two.scratch->data(), 2, two.scratch->size(), second), "memset second");
    }
    first.synchronize();
    second.synchronize();
    EXPECT_EQ(device.reserved().graphs, 0u);
    EXPECT_EQ(device.reserved(first).graphs, 1000u);
}

TEST(DeviceEncoder, RefreshResizesScratchWhenTheCellBoundGrows)
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
    device.reserve(1000, stream);
    const auto old = device.acquire(stream);
    const u64 cells = device.bounds().cells;
    const u64 before = device.reserved_bytes(stream);
    EXPECT_TRUE(device.refresh(grown));
    EXPECT_GT(device.bounds().cells, cells);
    EXPECT_GT(device.reserved_bytes(stream), before);
    EXPECT_EQ(device.reserved_bytes(stream), device.scratch_layout(device.reserved(stream)).bytes);
    const auto next = device.acquire(stream);
    EXPECT_NE(next.scratch, old.scratch);
    EXPECT_NE(next.context.generation, old.context.generation);
    EXPECT_EQ(next.context.number, old.context.number + 1);
    EXPECT_EQ(old.scratch->size(), before);
    device.reserve(1000, stream);
    EXPECT_EQ(device.reserved_bytes(stream), device.scratch_layout(device.reserved(stream)).bytes);
}

TEST(DeviceEncoder, DerivedRefusalDoesNotDependOnTheInstancesSlots)
{
    SKIP_WITHOUT_GPU();
    const auto domain = frontend::Domain::from_string(
        "(define (domain derived) (:requirements :strips :derived-predicates)"
        " (:predicates (flag ?x) (available ?x)) (:derived (available ?x) (flag ?x))"
        " (:action set :parameters (?x) :precondition (and) :effect (flag ?x)))");
    const auto empty = Task::create(*domain->instantiate_string(
        "(define (problem empty) (:domain derived) (:init) (:goal (and)))"));
    const auto populated = Task::create(*domain->instantiate_string(
        "(define (problem populated) (:domain derived) (:objects a) (:init) (:goal (and)))"));
    auto options = test::encoder_options();
    const auto encoder = encoding::FlatEncoder::create(rl::TaskTable::single(empty), options);
    EXPECT_FALSE(encoder->context().needs_derived(0));
    EXPECT_THROW(cuda::DeviceEncoder(device_context(), encoder), std::invalid_argument);
    options.include_derived = false;
    const auto without = encoding::FlatEncoder::create(rl::TaskTable::single(empty), options);
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create({empty, populated}), options);
    cuda::DeviceEncoder device(device_context(), without);
    EXPECT_TRUE(device.refresh(grown));
}

TEST(DeviceEncoder, OversizedReservationsLeaveExistingScratchIntact)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    cuda::DeviceEncoder device(device_context(), encoder);
    device.reserve(10);
    const auto old = device.acquire();
    const u64 huge = std::numeric_limits<u64>::max();
    EXPECT_THROW((void)device.worst_case(huge), std::overflow_error);
    EXPECT_THROW(device.reserve(cuda::BatchShape{huge, 0, 0}), std::overflow_error);
    EXPECT_EQ(device.acquire().scratch, old.scratch);
    EXPECT_EQ(device.reserved().graphs, 10u);
}

TEST(DeviceEncoder, CombinedCapacityOverflowLeavesScratchIntact)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    cuda::DeviceEncoder device(device_context(), encoder);
    const u64 large = std::numeric_limits<u64>::max() / 8;
    device.reserve(cuda::BatchShape{1, large, 0});
    const auto old = device.acquire();
    // Each output shape fits u64 on its own; their componentwise maximum does not.
    (void)cuda::output_layout({1, 0, large}, encoding::IndexType::I32, encoder->schema().num_relations());
    EXPECT_THROW(device.reserve(cuda::BatchShape{1, 0, large}), std::overflow_error);
    EXPECT_EQ(device.acquire().scratch, old.scratch);
    EXPECT_EQ(device.reserved(), (cuda::BatchShape{1, large, 0}));
}

TEST(DeviceEncoder, ConcurrentStreamsReserveAndAcquireIndependently)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream first, second;
    cuda::DeviceEncoder device(ctx, encoder);
    std::exception_ptr errors[2];
    auto run = [&](u32 i, cudaStream_t stream)
    {
        try
        {
            for (u64 n = 1; n <= 16; ++n)
            {
                device.reserve(n * (i + 1), stream);
                auto lease = device.acquire(stream);
                cuda::check(cudaMemsetAsync(lease.scratch->data(), i, lease.scratch->size(), stream), "memset scratch");
            }
        }
        catch (...) { errors[i] = std::current_exception(); }
    };
    std::thread one(run, 0, first.get()), two(run, 1, second.get());
    one.join();
    two.join();
    EXPECT_FALSE(errors[0]);
    EXPECT_FALSE(errors[1]);
    first.synchronize();
    second.synchronize();
    EXPECT_EQ(device.reserved(first).graphs, 16u);
    EXPECT_EQ(device.reserved(second).graphs, 32u);
}

TEST(DeviceEncoder, CallerReserveStreamMayBeDestroyedFirst)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    auto device = std::make_unique<cuda::DeviceEncoder>(ctx, encoder);
    {
        cuda::Stream stream;
        device->reserve(1000, stream);
        auto lease = device->acquire(stream);
        cuda::check(cudaMemsetAsync(lease.scratch->data(), 0, lease.scratch->size(), stream), "memset scratch");
    }
    device.reset();
    ctx->synchronize();
}

TEST(DeviceEncoder, ScratchFreeWaitsForTheAcquiredStream)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream stream;
    auto device = std::make_unique<cuda::DeviceEncoder>(ctx, encoder);
    device->reserve(1000, stream);
    ctx->synchronize();
    const auto keep_context = device->context().acquire();
    {
        StreamGate gate(stream);
        { auto lease = device->acquire(stream); }
        device.reset();
        expect_blocked(ctx);
    }
    ctx->synchronize();
}

// ------------------------------------------------------------------------------------------------ encoder

TEST(DeviceEncoder, GrowsWithItsTable)
{
    SKIP_WITHOUT_GPU();
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0"});
    if (tasks.empty())
        GTEST_SKIP() << "no PDDL data";
    const auto options = test::encoder_options();
    const auto small = encoding::FlatEncoder::create(rl::TaskTable::create({tasks[0]}), options);
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create(tasks), options);
    cuda::DeviceEncoder device(device_context(), small);
    const u64 entities = device.bounds().entities;
    EXPECT_TRUE(device.refresh(grown));
    EXPECT_EQ(device.context().uploads(), 2u);
    EXPECT_EQ(device.context().num_instances(), 2u);
    EXPECT_GE(device.bounds().entities, entities);
    EXPECT_FALSE(device.refresh(grown));
    EXPECT_TRUE(device.context().download() == host_bytes(grown->context()));
    // other options
    encoding::FlatOptions other = options;
    other.nullary = false;
    EXPECT_THROW(device.refresh(encoding::FlatEncoder::create(rl::TaskTable::create(tasks), other)), std::invalid_argument);
}

TEST(DeviceEncoder, RefusesWhatTheDeviceDoesNotEncode)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    EXPECT_THROW(cuda::DeviceEncoder(nullptr, blocks_encoder()), std::invalid_argument);
    EXPECT_THROW(cuda::DeviceEncoder(ctx, nullptr), std::invalid_argument);
    const auto table = rl::TaskTable::single(test::pddl_task("features", "p01"));
    // derived relations of a table with derived atoms
    try
    {
        cuda::DeviceEncoder device(ctx, encoding::FlatEncoder::create(table, test::encoder_options()));
        ADD_FAILURE() << "derived relations were accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("mymyr: encoding: the device does not encode derived relations"), std::string::npos)
            << e.what();
    }
    // without them the same table is fine
    encoding::FlatOptions options = test::encoder_options();
    options.include_derived = false;
    const cuda::DeviceEncoder device(ctx, encoding::FlatEncoder::create(table, options));
    EXPECT_EQ(device.context().num_instances(), 1u);
    EXPECT_TRUE(device.context().download() == host_bytes(device.encoder()->context()));
}
}  // namespace
