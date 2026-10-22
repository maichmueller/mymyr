// Device context bytes, generation ownership, per-stream scratch and checked batch sizes.

#include "../cpp/encoding/support.hpp"
#include "context_probe.hpp"
#include "mymyr/cuda/encoding.hpp"
#include "../../cuda/src/encoding_scratch.hpp"
#include "../../cuda/src/encoding_kernels.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
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
        EXPECT_EQ(device->generation(), 1u);
        EXPECT_EQ(device->bytes(), host->arrays().bytes());
        EXPECT_EQ(device->num_instances(), table->size());
        EXPECT_TRUE(device->download() == host_bytes(*host));
        // a second stream sees the block after acquire()
        EXPECT_TRUE(device->download(other) == host_bytes(*host));
        EXPECT_EQ(device->generation(), 1u);
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
    EXPECT_EQ(device->generation(), 1u);

    EXPECT_TRUE(device->refresh(g->after));
    EXPECT_EQ(device->generation(), 2u);
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
    EXPECT_EQ(device->generation(), 1u);
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
    EXPECT_EQ(device->generation(), 1u);
    for (u32 n = 2; n <= 4; ++n)
    {
        EXPECT_TRUE(device->refresh(context_of(n)));
        EXPECT_EQ(device->generation(), n);
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
    EXPECT_EQ(device->generation(), 1u);
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

TEST(DeviceEncodingContext, RefreshFreesTheOldGenerationAfterTheAcquiredStream)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    ASSERT_TRUE(g);
    const auto ctx = device_context();
    cuda::Stream stream;
    const auto device = cuda::DeviceEncodingContext::upload(ctx, g->before);
    ctx->synchronize();
    {
        StreamGate gate(stream);
        { auto lease = device->acquire(stream); }
        EXPECT_TRUE(device->refresh(g->after));  // nothing holds the old generation: its free is enqueued here
        expect_blocked(ctx);
    }
    ctx->synchronize();
}

TEST(DeviceEncodingContext, ReleasingAStreamKeepsTheGenerationFreeOrdered)
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
        device->release(stream);
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

// ------------------------------------------------------------------------------------------------ capture

u32 kernel_nodes(cudaGraph_t graph)
{
    size_t count = 0;
    cuda::check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    cuda::check(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
    u32 kernels = 0;
    for (const cudaGraphNode_t node : nodes)
    {
        cudaGraphNodeType type;
        cuda::check(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        kernels += type == cudaGraphNodeTypeKernel;
    }
    return kernels;
}

/// Captures what `enqueue` launches on `stream` and returns the graph's kernel nodes. The graph is never launched.
template <class Enqueue>
u32 captured_kernels(cudaStream_t stream, Enqueue&& enqueue)
{
    cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    std::exception_ptr error;
    try { enqueue(); }
    catch (...) { error = std::current_exception(); }
    cudaGraph_t graph = nullptr;
    cuda::check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    if (error)
    {
        (void)cudaGraphDestroy(graph);
        std::rethrow_exception(error);
    }
    const u32 kernels = kernel_nodes(graph);
    cuda::check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    return kernels;
}

std::shared_ptr<const encoding::FlatEncoder> blocks_encoder(encoding::IndexType index = encoding::IndexType::I32);

TEST(DeviceEncoder, AHeldAcquisitionIsEnoughToLaunchInsideACapture)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(64, stream);
    const auto work = device.acquire(stream);
    cuda::DeviceBuffer out(ctx, test::k_probe_width * 8, stream);
    const u32 kernels = captured_kernels(stream, [&]
    {
        EXPECT_GT(work.scratch_bytes(), 0u);
        test::launch_context_probe(work.view(), 1, static_cast<u64*>(out.data()), work.stream());
    });
    EXPECT_EQ(kernels, 1u);  // the probe; the memset is not a kernel node
}

TEST(DeviceEncoder, AcquireRefusesACapturingStream)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(64, stream);
    cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    EXPECT_THROW((void)device.acquire(stream), std::invalid_argument);
    EXPECT_THROW((void)device.context().acquire(stream), std::invalid_argument);
    cudaGraph_t graph = nullptr;
    EXPECT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);  // the refusals left the capture valid
    if (graph)
        cuda::check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    EXPECT_NO_THROW((void)device.acquire(stream));
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
    EXPECT_EQ(i32.sizes, 5760u);  // without goals and targets, the sizes follow the arguments
    EXPECT_EQ(i32.bytes, 5824u);
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
    EXPECT_EQ(i64.bytes, 11072u);
    // an empty batch has the offsets arrays of one entry and the sizes
    EXPECT_EQ(cuda::output_layout({}, encoding::IndexType::I32, 7).bytes, 64u * 4);
}

TEST(DeviceEncoder, OutputLayoutHasTheArraysOfItsSchemaOnly)
{
    const cuda::BatchShape shape{10, 100, 1000, 50};
    // every array: the goal flags (G = 10), the target offsets I (G + 1) = 44, the targets I T = 200 and T = 50
    const cuda::OutputArrays all{true, true, true, true};
    const cuda::OutputLayout full = cuda::output_layout(shape, encoding::IndexType::I32, 7, all);
    EXPECT_EQ(full.goal_complete, 5760u);
    EXPECT_EQ(full.target_offsets, 5824u);
    EXPECT_EQ(full.target_entity, 5888u);
    EXPECT_EQ(full.target_group, 6144u);
    EXPECT_EQ(full.sizes, 6208u);
    EXPECT_EQ(full.bytes, 6272u);
    // implicit entity arrays and no counts take no bytes
    const cuda::OutputLayout lean = cuda::output_layout(shape, encoding::IndexType::I32, 7, {false, false, false, false});
    EXPECT_EQ(lean.entity_role, lean.entity_ref);
    EXPECT_EQ(lean.entity_ref, lean.relation_counts);
    EXPECT_EQ(lean.relation_counts, lean.relation_offsets);
    EXPECT_EQ(lean.bytes, 5824u - 128 - 448 - 320);
    // the arrays of a schema
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    auto options = test::encoder_options();
    EXPECT_EQ(cuda::output_arrays(*encoding::EncodingSchema::create(*table, options)), (cuda::OutputArrays{true, true, false, false}));
    options.goal_derivations = {encoding::GoalDerivation::Plain};
    options.targets = {encoding::TargetSource::Goal};
    options.emit_counts = false;
    EXPECT_EQ(cuda::output_arrays(*encoding::EncodingSchema::create(*table, options)), (cuda::OutputArrays{true, false, true, true}));
}

TEST(DeviceEncoder, ScratchHoldsOneScanOfTheTotals)
{
    SKIP_WITHOUT_GPU();
    for (const u64 graphs : {u64{1}, u64{10}, u64{1000}, u64{16384}})
        for (const u64 cells : {u64{0}, u64{6}, u64{170}})
        {
            const cuda::GraphBounds bounds{9, 97, 161, cells};
            const auto narrow = cuda::detail::scratch_layout({graphs, 9 * graphs, 161 * graphs}, bounds, encoding::Layout::Graph, 7);
            EXPECT_FALSE(narrow.wide);
            EXPECT_EQ(narrow.flags, 0u);
            EXPECT_EQ(narrow.totals, 64u);
            EXPECT_EQ(narrow.scan_length, 4 * (graphs + 1));
            EXPECT_EQ(narrow.cell_counts, round_up(64 + 16 * (graphs + 1)));
            EXPECT_EQ(narrow.cell_entries, graphs * cells);
            EXPECT_EQ(narrow.cell_stride, cells);
            EXPECT_EQ(narrow.scan_temp - narrow.cell_counts, round_up(4 * graphs * cells));
            EXPECT_GT(narrow.scan_temp_bytes, 0u);
            EXPECT_EQ(narrow.bytes - narrow.scan_temp, round_up(narrow.scan_temp_bytes));
            // 16 bytes per graph and 4 per cell, besides the flags and the scan's storage
            EXPECT_LE(narrow.scan_temp, 64 + 63 + 16 * (graphs + 1) + 4 * graphs * cells + 63);
            // the width follows the bounds: larger output capacities never switch to u64
            const auto roomy =
                cuda::detail::scratch_layout({graphs, u64{1} << 40, u64{1} << 40, u64{1} << 40}, bounds, encoding::Layout::Graph, 7);
            EXPECT_FALSE(roomy.wide);
            EXPECT_EQ(roomy.bytes, narrow.bytes);
            // entities, targets, tuples and arguments together decide it
            const cuda::GraphBounds big{u64{1} << 32, 1, 1, cells};
            const auto wide = cuda::detail::scratch_layout({graphs, 0, 0}, big, encoding::Layout::Graph, 7);
            EXPECT_TRUE(wide.wide);
            EXPECT_EQ(wide.cell_counts, round_up(64 + 32 * (graphs + 1)));
            // relation-major: the entity and target rows and the cell row; no cell counts
            const auto rm = cuda::detail::scratch_layout({graphs, 9 * graphs, 161 * graphs}, bounds, encoding::Layout::Relation, 7);
            EXPECT_FALSE(rm.wide);
            EXPECT_TRUE(rm.relation_major);
            EXPECT_EQ(rm.scan_length, 2 * (graphs + 1) + 7 * graphs + 1);
            EXPECT_EQ(rm.cell_entries, 0u);
            EXPECT_EQ(rm.scan_temp, rm.cell_counts);
            EXPECT_LE(rm.scan_temp, 64 + 63 + 8 * (graphs + 1) + 4 * (7 * graphs + 1) + 63);
        }
    const u64 quarter = std::numeric_limits<u32>::max() / 4;
    EXPECT_FALSE(cuda::detail::scratch_layout({1, 0, 0}, {quarter, quarter, quarter, 0, quarter}, encoding::Layout::Graph).wide);
    EXPECT_TRUE(cuda::detail::scratch_layout({1, 0, 0}, {quarter + 1, quarter + 1, quarter + 1, 0, quarter + 1}, encoding::Layout::Graph).wide);
    // the argument row is not scanned relation-major
    EXPECT_FALSE(cuda::detail::scratch_layout({1, 0, 0}, {quarter, quarter, u64{1} << 40, 0, quarter}, encoding::Layout::Relation, 1).wide);
}

TEST(DeviceEncoder, OverflowIsRejectedBeforeAllocation)
{
    const u64 huge = std::numeric_limits<u64>::max();
    EXPECT_THROW((void)cuda::output_layout({huge, 0, 0}, encoding::IndexType::I32, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({1, huge, 0}, encoding::IndexType::I64, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({1, 0, huge}, encoding::IndexType::I32, 1), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({1, 0, 0, huge}, encoding::IndexType::I32, 1, {true, true, true, true}), std::overflow_error);
    EXPECT_THROW((void)cuda::output_layout({2, 0, 0}, encoding::IndexType::I64, huge), std::overflow_error);
    EXPECT_THROW((void)cuda::detail::scratch_layout({huge, 0, 0}, {}, encoding::Layout::Graph), std::overflow_error);
    EXPECT_THROW((void)cuda::detail::scratch_layout({huge, 0, 0}, {}, encoding::Layout::Relation, 3), std::overflow_error);
    EXPECT_THROW((void)cuda::detail::scratch_layout({2, 0, 0}, {0, 0, 0, huge}, encoding::Layout::Graph), std::overflow_error);
    EXPECT_THROW((void)cuda::detail::scratch_layout({u64{1} << 40, 0, 0}, {}, encoding::Layout::Relation, u64{1} << 40), std::overflow_error);
}

std::shared_ptr<const encoding::FlatEncoder> blocks_encoder(encoding::IndexType index)
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
    // Fixed-row output bounds exclude expansion tuples; cell scratch also holds the optional action cells.
    // These options have no goal, so the instance and row goal lists agree.
    const auto view = encoder->context().view();
    cuda::GraphBounds expected;
    for (u32 i = 0; i < encoder->table().size(); ++i)
    {
        const auto inst = encoding::detail::instance_view(view, i);
        u64 tuples = 0, args = 0;
        bool nullary = false;
        for (u32 c = 0; c < inst.num_plain_cells; ++c)
        {
            tuples += inst.plain_cells[c].b - inst.plain_cells[c].a;
            args += u64{inst.plain_cells[c].b - inst.plain_cells[c].a} * inst.plain_cells[c].arity;
            nullary = nullary || inst.plain_cells[c].logical_arity == 0;
        }
        expected.entities = std::max<u64>(expected.entities, inst.num_objects + nullary);
        expected.tuples = std::max(expected.tuples, tuples);
        expected.args = std::max(expected.args, args);
        expected.cells = std::max<u64>(expected.cells, inst.num_cells);
        expected.words = std::max<u64>(expected.words, inst.words);
        expected.state_words = std::max<u64>(expected.state_words, inst.words);
        expected.derived_words = std::max<u64>(expected.derived_words, inst.derived_words);
    }
    EXPECT_EQ(b.entities, expected.entities);
    EXPECT_EQ(b.tuples, expected.tuples);
    EXPECT_EQ(b.args, expected.args);
    EXPECT_EQ(b.cells, expected.cells);
    EXPECT_EQ(b.words, expected.words);
    EXPECT_EQ(b.state_words, expected.state_words);
    EXPECT_EQ(b.derived_words, expected.derived_words);
    EXPECT_EQ(b.targets, 0u);
    EXPECT_EQ(b.levels, 0u);
    EXPECT_FALSE(b.history);
    EXPECT_EQ(b.step_entities + b.step_tuples + b.step_args + b.step_targets, 0u);

    const u64 G = 1000, R = encoder->schema().num_relations();
    const cuda::BatchShape worst = device.worst_case(G);
    EXPECT_EQ(worst, (cuda::BatchShape{G, G * b.entities, G * b.args, 0}));
    // scratch + output, with the output in four-byte indices and the sizes
    const u64 output = round_up(4 * G) + round_up(4 * G) + round_up(4 * (G + 1)) + round_up(G * b.entities) +
                       round_up(4 * G * b.entities) + round_up(4 * G * R) + 2 * round_up(4 * (G * R + 1)) +
                       round_up(4 * G * b.args) + 64;
    EXPECT_EQ(device.bytes_needed(worst), cuda::detail::scratch_layout(worst, device.bounds(), device.options().layout, device.schema().num_relations()).bytes + output);
    EXPECT_EQ(device.bytes_needed(G), device.bytes_needed(worst));
    // a smaller batch needs less, a batch with fewer entities and arguments needs less
    EXPECT_LT(device.bytes_needed(G / 2), device.bytes_needed(G));
    EXPECT_LT(device.bytes_needed(cuda::BatchShape{G, 1, 1}), device.bytes_needed(G));
}

TEST(DeviceEncoder, BoundsCoverGoalsTargetsAndVirtualNodes)
{
    SKIP_WITHOUT_GPU();
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto plain = encoding::FlatEncoder::create(table, test::encoder_options());
    encoding::FlatOptions options = test::encoder_options();
    options.goal_derivations = {encoding::GoalDerivation::Plain, encoding::GoalDerivation::Satisfied,
                                encoding::GoalDerivation::Unsatisfied};
    options.targets = {encoding::TargetSource::Goal};
    options.predicate_virtual_nodes = true;
    const auto rich = encoding::FlatEncoder::create(table, options);
    const auto ctx = device_context();
    const cuda::DeviceEncoder base(ctx, plain), full(ctx, rich);
    const auto inst = encoding::detail::instance_view(rich->context().view(), 0);
    // goal cells add tuples, goal targets add entities and targets, virtual nodes add entities
    EXPECT_GT(full.bounds().tuples, base.bounds().tuples);
    EXPECT_GT(full.bounds().targets, 0u);
    EXPECT_GE(full.bounds().entities, base.bounds().entities + full.bounds().targets + inst.num_virtual);
    EXPECT_GT(full.bounds().cells, base.bounds().cells);
    EXPECT_EQ(full.worst_case(10).targets, 10 * full.bounds().targets);
    EXPECT_GT(full.bytes_needed(u64{1000}), base.bytes_needed(u64{1000}));
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
    EXPECT_EQ(device.bounds().cells, 10u);
    EXPECT_EQ(encoder->schema().num_relations(), 25u);
    const auto scratch = cuda::detail::scratch_layout(device.worst_case(1000), device.bounds(), device.options().layout, device.schema().num_relations());
    EXPECT_EQ(device.bytes_needed(u64{1000}) - round_up(scratch.scan_temp_bytes), 1117504u);
}

void touch_scratch(const cuda::DeviceEncoder& device, const cuda::EncodingWorkspace& work, const cuda::ContextPtr& ctx)
{
    cuda::DeviceBuffer buffer(ctx, sizeof(encoding::FlatSizes), work.stream());
    cuda::DeviceOutput out;
    out.capacities.graphs = work.shape().graphs;
    out.sizes = static_cast<encoding::FlatSizes*>(buffer.data());
    device.count(work, {}, out);
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
    const auto held_bytes = lease.scratch_bytes();
    const u64 bytes = cuda::detail::scratch_layout(device.reserved(), device.bounds(), device.options().layout, device.schema().num_relations()).bytes;
    EXPECT_EQ(device.reserved_bytes(), bytes);
    EXPECT_EQ(device.bytes_needed(device.reserved()), bytes + device.reserved_layout().bytes);
    cuda::DeviceBuffer output(ctx, device.reserved_layout().bytes);
    EXPECT_NE(output.data(), nullptr);
    device.reserve(u64{1});
    device.reserve(u64{4096});
    device.reserve(cuda::BatchShape{10, 1, 1});
    EXPECT_EQ(device.acquire().scratch_bytes(), held_bytes);
    EXPECT_EQ(device.reserved(), device.worst_case(4096));
    device.reserve(cuda::BatchShape{8192, 1, 1});
    EXPECT_EQ(device.reserved().graphs, 8192u);
    EXPECT_EQ(device.reserved().entities, device.worst_case(4096).entities);
    EXPECT_EQ(device.reserved_bytes(), cuda::detail::scratch_layout(device.reserved(), device.bounds(), device.options().layout, device.schema().num_relations()).bytes);
    EXPECT_GT(device.acquire().scratch_bytes(), held_bytes);
    EXPECT_EQ(lease.scratch_bytes(), held_bytes);
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
        EXPECT_NE(one.scratch_bytes(), two.scratch_bytes());
        EXPECT_EQ(one.view().inst, two.view().inst);
        EXPECT_EQ(one.shape().graphs, 1000u);
        EXPECT_EQ(two.shape().graphs, 2000u);
        touch_scratch(device, one, ctx);
        touch_scratch(device, two, ctx);
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
    EXPECT_EQ(device.reserved_bytes(stream), cuda::detail::scratch_layout(device.reserved(stream), device.bounds(), device.options().layout, device.schema().num_relations()).bytes);
    const auto next = device.acquire(stream);
    EXPECT_NE(next.scratch_bytes(), old.scratch_bytes());
    EXPECT_NE(next.view().inst, old.view().inst);
    EXPECT_EQ(next.generation(), old.generation() + 1);
    EXPECT_EQ(old.scratch_bytes(), before);
    device.reserve(1000, stream);
    EXPECT_EQ(device.reserved_bytes(stream), cuda::detail::scratch_layout(device.reserved(stream), device.bounds(), device.options().layout, device.schema().num_relations()).bytes);
}

TEST(DeviceEncoder, DerivedRelationsAreAcceptedWhateverTheInstancesSlots)
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
    const auto options = test::encoder_options();
    const auto encoder = encoding::FlatEncoder::create(rl::TaskTable::single(empty), options);
    EXPECT_FALSE(encoder->context().needs_derived(0));
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create({empty, populated}), options);
    cuda::DeviceEncoder device(device_context(), encoder);
    EXPECT_TRUE(device.refresh(grown));
    EXPECT_TRUE(grown->context().needs_derived(1));
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
    EXPECT_EQ(device.acquire().scratch_bytes(), old.scratch_bytes());
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
    EXPECT_EQ(device.acquire().scratch_bytes(), old.scratch_bytes());
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
                touch_scratch(device, lease, ctx);
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
        touch_scratch(*device, lease, ctx);
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

TEST(DeviceEncoder, ReleasingAHeldWorkspaceOrdersItsScratchFree)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, blocks_encoder());
    device.reserve(1000, stream);
    ctx->synchronize();
    const auto keep_context = device.context().acquire();
    {
        StreamGate gate(stream);
        {
            auto lease = device.acquire(stream);
            device.release(stream);
        }
        expect_blocked(ctx);
    }
    ctx->synchronize();
}

TEST(DeviceEncoder, AFailedFirstReserveLeavesTheStreamUnreserved)
{
    SKIP_WITHOUT_GPU();
    const std::vector<TaskPtr> tasks = load_tasks("pddl/blocks", {"smedium", "probBLOCKS-8-0"});
    if (tasks.empty())
        GTEST_SKIP() << "no PDDL data";
    const auto options = test::encoder_options();
    const auto small = encoding::FlatEncoder::create(rl::TaskTable::create({tasks[0]}), options);
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create(tasks), options);
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, small);
    // far beyond the pool of the context
    EXPECT_THROW(device.reserve(u64{1} << 28, stream), std::exception);
    EXPECT_EQ(device.reserved(stream), cuda::BatchShape{});
    EXPECT_EQ(device.reserved_bytes(stream), 0u);
    EXPECT_TRUE(device.refresh(grown));
    EXPECT_EQ(device.reserved_bytes(stream), 0u);
    try
    {
        (void)device.acquire(stream);
        ADD_FAILURE() << "acquire succeeded without a reservation";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("reserve scratch on this stream before acquire"), std::string::npos) << e.what();
    }
    device.reserve(10, stream);
    EXPECT_NO_THROW((void)device.acquire(stream));
}

TEST(DeviceEncoder, ReleasingAStreamFreesItsScratchAfterItsLastWork)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, encoder);
    device.reserve(1000, stream);
    ctx->synchronize();
    {
        StreamGate gate(stream);
        { auto work = device.acquire(stream); }
        device.release(stream);
        expect_blocked(ctx);
        EXPECT_EQ(device.reserved_bytes(stream), 0u);
        EXPECT_THROW((void)device.acquire(stream), std::invalid_argument);
    }
    ctx->synchronize();
    device.reserve(10, stream);  // the stream reserves again
    EXPECT_NO_THROW((void)device.acquire(stream));
}

TEST(DeviceEncoder, WorkOnAnotherStreamIsOrderedIntoTheAcquiredStream)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder();
    ASSERT_TRUE(encoder);
    const auto ctx = device_context();
    cuda::Stream stream, other;
    auto device = std::make_unique<cuda::DeviceEncoder>(ctx, encoder);
    device->reserve(1000, stream);
    ctx->synchronize();
    {
        StreamGate gate(other);  // the replays run here
        cuda::Event replayed;
        replayed.record(other);
        {
            auto work = device->acquire(stream);
            replayed.wait_on(work.stream());  // the acquisition's stream waits before the release
        }
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
    EXPECT_EQ(device.context().generation(), 2u);
    EXPECT_EQ(device.context().num_instances(), 2u);
    EXPECT_GE(device.bounds().entities, entities);
    EXPECT_FALSE(device.refresh(grown));
    EXPECT_TRUE(device.context().download() == host_bytes(grown->context()));
    // other options
    encoding::FlatOptions other = options;
    other.nullary = false;
    EXPECT_THROW(device.refresh(encoding::FlatEncoder::create(rl::TaskTable::create(tasks), other)), std::invalid_argument);
}

TEST(DeviceEncoder, RefusesToGrowIntoATableWithNumericStateWords)
{
    // the first instance has no object of the numeric function's type: without include_numeric both tables have one
    // schema, and only the grown one has numeric state words
    SKIP_WITHOUT_GPU();
    const auto domain = frontend::Domain::from_string(
        "(define (domain grow) (:requirements :strips :typing :numeric-fluents) (:types a)"
        " (:predicates (p)) (:functions (f ?x - a))"
        " (:action inc :parameters (?x - a) :precondition (p) :effect (increase (f ?x) 1)))");
    const auto empty = Task::create(*domain->instantiate_string(
        "(define (problem empty) (:domain grow) (:init (p)) (:goal (p)))"));
    const auto one = Task::create(*domain->instantiate_string(
        "(define (problem one) (:domain grow) (:objects x - a) (:init (p) (= (f x) 0)) (:goal (p)))"));
    auto options = test::encoder_options();
    options.include_numeric = false;
    const auto base = encoding::FlatEncoder::create(rl::TaskTable::single(empty), options);
    const auto grown = encoding::FlatEncoder::create(rl::TaskTable::create({empty, one}), options);
    ASSERT_FALSE(base->table().numeric());
    ASSERT_TRUE(grown->table().numeric());
    ASSERT_EQ(base->schema().fingerprint(), grown->schema().fingerprint());
    cuda::DeviceEncoder device(device_context(), base);
    try
    {
        (void)device.refresh(grown);
        ADD_FAILURE() << "the device grew into a table with numeric state words";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("mymyr: encoding: the device does not encode numeric relations"),
                  std::string::npos)
            << e.what();
    }
    // nothing was uploaded
    EXPECT_EQ(device.context().generation(), 1u);
    EXPECT_EQ(device.context().num_instances(), 1u);
    EXPECT_EQ(device.encoder(), base);
}

TEST(DeviceEncoder, RefusesWhatTheDeviceDoesNotEncode)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = device_context();
    EXPECT_THROW(cuda::DeviceEncoder(nullptr, blocks_encoder()), std::invalid_argument);
    EXPECT_THROW(cuda::DeviceEncoder(ctx, nullptr), std::invalid_argument);
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    // numeric relations, and the numeric state words of a table whose encoder leaves the relations out
    {
        const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
        const auto numeric_table = rl::TaskTable::single(
            Task::create(*frontend::load_task(dir + "domain.pddl", dir + "p01.pddl")));
        for (const bool include_numeric : {true, false})
        {
            encoding::FlatOptions numeric = test::encoder_options();
            numeric.include_numeric = include_numeric;
            try
            {
                cuda::DeviceEncoder device(ctx, encoding::FlatEncoder::create(numeric_table, numeric));
                ADD_FAILURE() << "numeric relations were accepted (include_numeric " << include_numeric << ")";
            }
            catch (const std::invalid_argument& e)
            {
                EXPECT_NE(std::string(e.what()).find("mymyr: encoding: the device does not encode numeric relations"),
                          std::string::npos)
                    << e.what();
            }
        }
    }
    // horizon relations: every mode, the rooted horizon (parent, sibling and cousin relations), the transitions (the
    // depth-one horizons: full or delta, the root excluded)
    struct HorizonCase
    {
        const char* what;
        encoding::HorizonMode mode;
        encoding::RootPolicy root;
        bool parent, sibling, cousin;
    };
    using enum encoding::HorizonMode;
    using enum encoding::RootPolicy;
    for (const HorizonCase& c : {HorizonCase{"full horizon", Full, Include, false, false, false},
                                 HorizonCase{"delta horizon", Delta, Include, false, false, false},
                                 HorizonCase{"action horizon", Action, Include, false, false, false},
                                 HorizonCase{"rooted horizon", Full, EncodeOnly, true, true, true},
                                 HorizonCase{"cousin relation", Action, Include, false, false, true},
                                 HorizonCase{"transitions", Full, Exclude, false, false, false},
                                 HorizonCase{"transition effects", Delta, Exclude, true, false, false}})
    {
        encoding::FlatOptions horizon = test::encoder_options();
        horizon.horizon.mode = c.mode;
        horizon.horizon.root = c.root;
        horizon.horizon.parent_relation = c.parent;
        horizon.horizon.sibling_relation = c.sibling;
        horizon.horizon.cousin_relation = c.cousin;
        EXPECT_NO_THROW(cuda::DeviceEncoder(ctx, encoding::FlatEncoder::create(table, horizon))) << c.what;
    }
    // subgoal and history relations are encoded
    {
        encoding::FlatOptions with_subgoals;
        with_subgoals.max_goal_level = 1;
        encoding::FlatOptions with_history;
        with_history.history = true;
        encoding::FlatOptions with_both;
        with_both.max_goal_level = 3;
        with_both.history = true;
        with_both.history_max_steps = 2;
        with_both.targets = {encoding::TargetSource::Goal, encoding::TargetSource::Subgoal,
                             encoding::TargetSource::History};
        for (const auto& accepted_options : {with_subgoals, with_history, with_both})
        {
            const cuda::DeviceEncoder device(ctx, encoding::FlatEncoder::create(table, accepted_options));
            EXPECT_EQ(device.bounds().levels, accepted_options.max_goal_level);
            EXPECT_EQ(device.bounds().history, accepted_options.history);
        }
    }
    // everything else is encoded: goal relations, goal targets, virtual nodes, derived relations, both layouts
    const auto derived_table = rl::TaskTable::single(test::pddl_task("features", "p01"));
    for (const auto& t : {table, derived_table})
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        {
            encoding::FlatOptions accepted;
            accepted.layout = layout;
            accepted.targets = {encoding::TargetSource::Goal};
            accepted.predicate_virtual_nodes = true;
            const cuda::DeviceEncoder device(ctx, encoding::FlatEncoder::create(t, accepted));
            EXPECT_EQ(device.context().num_instances(), 1u);
            EXPECT_TRUE(device.context().download() == host_bytes(device.encoder()->context()));
        }
}
}  // namespace

namespace
{
void expect_schema_refusal(encoding::FlatOptions options, const char* message)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto schema = encoding::EncodingSchema::create(*table, options);
    try
    {
        cuda::DeviceEncoder::check_schema(*schema);
        FAIL() << "the schema was accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(std::string(e.what()), std::string("mymyr: encoding: the device does not encode ") + message);
    }
}
TEST(DeviceEncoderSchema, AcceptsSubgoalLevelsHistoryAndTheirTargets)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    encoding::FlatOptions o;
    o.max_goal_level = 3;
    o.history = true;
    o.targets = {encoding::TargetSource::Goal, encoding::TargetSource::Subgoal, encoding::TargetSource::History};
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
    o.history_max_steps = 3;
    o.targets = {encoding::TargetSource::History};
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
}
TEST(DeviceEncoderSchema, AcceptsActionTargets)
{
    encoding::FlatOptions o;
    o.targets = {encoding::TargetSource::Action};
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
}
TEST(DeviceEncoderSchema, RefusesNumericRelations)
{
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
    const auto table = rl::TaskTable::single(Task::create(*frontend::load_task(dir + "domain.pddl", dir + "p01.pddl")));
    const auto schema = encoding::EncodingSchema::create(*table, test::encoder_options());
    ASSERT_TRUE(schema->has_numeric());
    try
    {
        cuda::DeviceEncoder::check_schema(*schema);
        FAIL() << "the schema was accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(std::string(e.what()), "mymyr: encoding: the device does not encode numeric relations");
    }
}
TEST(DeviceEncoderSchema, RefusesSharedStatics)
{
    encoding::FlatOptions o;
    o.statics = encoding::Statics::Shared;
    expect_schema_refusal(o, "shared statics (statics='shared')");
}
TEST(DeviceEncoderSchema, AcceptsFullHorizons)
{
    encoding::FlatOptions o;
    o.horizon.mode = encoding::HorizonMode::Full;
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
}
TEST(DeviceEncoderSchema, AcceptsDeltaHorizons)
{
    encoding::FlatOptions o;
    o.horizon.mode = encoding::HorizonMode::Delta;
    o.goal_derivations = {encoding::GoalDerivation::AddedSatisfied};
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
}
TEST(DeviceEncoderSchema, AcceptsActionHorizons)
{
    encoding::FlatOptions o;
    o.horizon.mode = encoding::HorizonMode::Action;
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*encoding::EncodingSchema::create(*table, o)));
}
TEST(DeviceEncoderSchema, TheEncoderAndTheRowStreamRefuseSharedStaticsWhenConstructed)
{
    SKIP_WITHOUT_GPU();
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    encoding::FlatOptions o;
    o.statics = encoding::Statics::Shared;
    const auto encoder = encoding::FlatEncoder::create(table, o);
    const auto ctx = device_context();
    cuda::Stream stream;
    EXPECT_THROW(cuda::DeviceEncoder(ctx, encoder), std::invalid_argument);
    EXPECT_THROW(cuda::DeviceRowStream(ctx, encoder, 8, stream), std::invalid_argument);
}
TEST(DeviceEncoderSchema, TheCollatorRefusesTheBatchesOfSharedStaticsWhenConstructed)
{
    // the descriptor of an empty batch, from the public helper: nothing is read and no kernel is launched
    SKIP_WITHOUT_GPU();
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    encoding::FlatOptions o;
    o.statics = encoding::Statics::Shared;
    const auto encoder = encoding::FlatEncoder::create(table, o);
    const auto ctx = device_context();
    cuda::Stream stream;
    try
    {
        const cuda::DeviceCollator collator(ctx, {cuda::collation_batch(*encoder, {}, nullptr, 0)}, stream);
        ADD_FAILURE() << "the collator accepted a batch of shared statics";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("mymyr: encoding: the device does not encode shared statics"),
                  std::string::npos)
            << e.what();
    }
}
TEST(DeviceEncoderSchema, TheCollationDescriptorRefusesWhatTheDeviceDoesNotEncode)
{
    // descriptors of empty batches, on the host: no device is needed and nothing is launched
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
    const auto numeric_table =
        rl::TaskTable::single(Task::create(*frontend::load_task(dir + "domain.pddl", dir + "p01.pddl")));
    struct Refused
    {
        const char* what;
        rl::TaskTablePtr table;
        encoding::FlatOptions options;
    };
    std::vector<Refused> cases;
    auto add = [&](const char* what, const rl::TaskTablePtr& t, auto&& change)
    {
        encoding::FlatOptions o = test::encoder_options();
        change(o);
        cases.push_back({what, t, o});
    };
    add("numeric relations", numeric_table, [](auto&) {});
    add("numeric relations (the table has numeric state words)", numeric_table, [](auto& o) { o.include_numeric = false; });
    add("Simple", table, [](auto& o) { o.outputs = {encoding::Output::Simple}; });
    add("Simple", table, [](auto& o) { o.outputs = {encoding::Output::Relations, encoding::Output::Simple}; });
    add("shared statics", table, [](auto& o) { o.statics = encoding::Statics::Shared; });
    for (const Refused& c : cases)
    {
        const auto encoder = encoding::FlatEncoder::create(c.table, c.options);
        try
        {
            (void)cuda::collation_batch(*encoder, {}, nullptr, 0);
            ADD_FAILURE() << "the descriptor accepted " << c.what;
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find(std::string("mymyr: encoding: the device does not encode ") + c.what),
                      std::string::npos)
                << e.what();
        }
    }
    // what the device encodes is described
    const auto accepted = encoding::FlatEncoder::create(table, test::encoder_options());
    const cuda::DeviceCollationBatch batch = cuda::collation_batch(*accepted, {}, nullptr, 0);
    EXPECT_EQ(batch.schema_fingerprint, accepted->schema().fingerprint());
    EXPECT_FALSE(batch.shared_statics);
}
}

namespace
{
TEST(DeviceEncoder, AutomaticCountLanesTryTheNextSharedMemoryConfiguration)
{
    SKIP_WITHOUT_GPU();
    std::string predicates, effects, init;
    for (u32 p = 0; p < 400; ++p)
    {
        const auto name = "p" + std::to_string(p);
        predicates += " (" + name + ")";
        effects += " (not (" + name + "))";
        init += " (" + name + ")";
    }
    const auto domain = frontend::Domain::from_string("(define (domain many) (:requirements :strips) (:predicates" +
        predicates + ") (:action clear :parameters () :precondition (and) :effect (and" + effects + ")))" );
    const auto task = Task::create(*domain->instantiate_string("(define (problem p) (:domain many) (:init" + init +
        ") (:goal (and)))"));
    auto options = test::encoder_options();
    options.predicate_virtual_nodes = true;
    const auto host = encoding::FlatEncoder::create(task, options);
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceEncoder device(ctx, host);
    const auto shape = device.worst_case(1);
    device.reserve(shape, stream);
    const auto output_layout = device.output_layout(shape);
    const auto& initial = host->table().instance(0).init;
    cuda::DeviceBuffer states(ctx, initial.size() * sizeof(u64), stream), block(ctx, output_layout.bytes, stream);
    cuda::check(cudaMemcpyAsync(states.data(), initial.data(), states.size(), cudaMemcpyHostToDevice, stream), "state");
    cuda::DeviceEncodeInput in;
    in.graphs = 1;
    in.states = static_cast<const u64*>(states.data());
    in.state_stride = host->table().words();
    in.state_words = host->table().words();
    const auto out = cuda::output_of(output_layout, shape, block.data());
    cuda::LaunchConfig forced;
    forced.count_lanes = 4;
    EXPECT_THROW(device.encode(in, out, stream, forced), std::invalid_argument);
    EXPECT_NO_THROW(device.encode(in, out, stream));
    encoding::FlatSizes sizes;
    cuda::check(cudaMemcpyAsync(&sizes, out.sizes, sizeof sizes, cudaMemcpyDeviceToHost, stream), "sizes");
    cuda::check(cudaStreamSynchronize(stream), "sizes");
    EXPECT_EQ(sizes.overflow, 0u);
}
TEST(DeviceEncoder, AnEncoderWhoseLaunchesNeedMoreSharedMemoryThanABlockHasIsRefusedWithoutAContext)
{
    SKIP_WITHOUT_GPU();
    auto many = [](u32 predicates)
    {
        std::string names, effects, init;
        for (u32 p = 0; p < predicates; ++p)
        {
            const auto name = "p" + std::to_string(p);
            names += " (" + name + ")";
            effects += " (not (" + name + "))";
            init += " (" + name + ")";
        }
        const auto domain = frontend::Domain::from_string("(define (domain many) (:requirements :strips) (:predicates" +
            names + ") (:action clear :parameters () :precondition (and) :effect (and" + effects + ")))");
        auto options = test::encoder_options();
        options.predicate_virtual_nodes = true;
        return encoding::FlatEncoder::create(
            Task::create(*domain->instantiate_string("(define (problem p) (:domain many) (:init" + init + ") (:goal (and)))")),
            options);
    };
    // 400 virtual nodes fit with one of the count kernel configurations, 4000 with none
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_encoder(*many(400)));
    try
    {
        cuda::DeviceEncoder::check_encoder(*many(4000));
        ADD_FAILURE() << "accepted an encoder whose count kernel needs more shared memory than a block has";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("too many predicate virtual nodes"), std::string::npos) << e.what();
    }
    // an ordinary encoder passes, and a schema the device does not encode is refused as well
    const auto blocks = load_tasks("pddl/blocks", {"smedium"});
    if (!blocks.empty())
    {
        EXPECT_NO_THROW(cuda::DeviceEncoder::check_encoder(*encoding::FlatEncoder::create(blocks.front(), test::encoder_options())));
        auto refused = test::encoder_options();
        refused.targets = {encoding::TargetSource::Action};
        EXPECT_THROW(cuda::DeviceEncoder::check_encoder(*encoding::FlatEncoder::create(blocks.front(), refused)),
                     std::invalid_argument);
    }
}

TEST(DeviceEncoder, SharedMemoryIncludesTheStaticScanStorage)
{
    SKIP_WITHOUT_GPU();
    for (const bool index64 : {false, true})
        for (const bool warp : {false, true})
        {
            const u64 dynamic = cuda::detail::fill_shared_bytes(0, 32, warp, true);
            const u64 total = cuda::detail::fill_shared_required(0, 32, warp, true, index64);
            ASSERT_GT(total, dynamic);
            const u64 extra = total - dynamic;
            const u32 relations = static_cast<u32>((cuda::detail::k_shared_limit - extra) / 8 + 1);
            EXPECT_LE(cuda::detail::fill_shared_bytes(0, relations, warp, true), cuda::detail::k_shared_limit);
            EXPECT_GT(cuda::detail::fill_shared_required(0, relations, warp, true, index64), cuda::detail::k_shared_limit);
        }
}

}

namespace
{
TEST(DeviceEncoder, FillZerosUnusedTailsAndOverflowOutputsWhenRequested)
{
    SKIP_WITHOUT_GPU();
    const auto task = test::pddl_task("blocks", "smedium");
    const auto ctx = device_context();
    cuda::Stream stream;
    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
        for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
            for (const u64 graphs : {0u, 2u})
            {
                encoding::FlatOptions options;
                options.layout = layout;
                options.index = index;
                options.targets = {encoding::TargetSource::Goal};
                const auto host = encoding::FlatEncoder::create(task, options);
                cuda::DeviceEncoder device(ctx, host);
                auto shape = device.worst_case(graphs);
                if (!graphs)
                    shape.entities = shape.args = shape.targets = 10;
                device.reserve(shape, stream);
                const auto& initial = host->table().instance(0).init;
                std::vector<u64> rows(initial.begin(), initial.end());
                rows.insert(rows.end(), initial.begin(), initial.end());
                cuda::DeviceBuffer states(ctx, rows.size() * sizeof(u64), stream);
                cuda::check(cudaMemcpyAsync(states.data(), rows.data(), states.size(), cudaMemcpyHostToDevice, stream), "states");
                cuda::DeviceEncodeInput in;
                in.graphs = graphs;
                in.states = static_cast<const u64*>(states.data());
                in.state_stride = in.state_words = host->table().words();
                for (const bool overflow : {false, true})
                {
                    auto capacity = shape;
                    if (overflow)
                    {
                        ++capacity.graphs;
                        capacity.entities = capacity.args = capacity.targets = 1;
                    }
                    const auto offsets = device.output_layout(capacity);
                    cuda::DeviceBuffer block(ctx, offsets.bytes, stream);
                    cuda::check(cudaMemsetAsync(block.data(), 0x5a, block.size(), stream), "dirty outputs");
                    auto out = cuda::output_of(offsets, capacity, block.data());
                    out.zero_unused = true;
                    device.encode(in, out, stream);
                    std::vector<std::byte> bytes(block.size());
                    cuda::check(cudaMemcpyAsync(bytes.data(), block.data(), block.size(), cudaMemcpyDeviceToHost, stream), "outputs");
                    cuda::check(cudaStreamSynchronize(stream), "outputs");
                    encoding::FlatSizes sizes;
                    std::memcpy(&sizes, bytes.data() + offsets.sizes, sizeof sizes);
                    const bool failed = overflow && graphs != 0;
                    EXPECT_EQ(sizes.overflow != 0, failed);
                    const u64 I = index == encoding::IndexType::I32 ? 4 : 8;
                    const auto zeros = [&](u64 at, u64 begin, u64 end, u64 width)
                    {
                        EXPECT_TRUE(std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(at + begin * width),
                                                bytes.begin() + static_cast<std::ptrdiff_t>(at + end * width),
                                                [](std::byte value) { return value == std::byte{0}; }));
                    };
                    zeros(offsets.entity_role, failed ? 0 : sizes.entities, capacity.entities, 1);
                    zeros(offsets.entity_ref, failed ? 0 : sizes.entities, capacity.entities, I);
                    zeros(offsets.relation_args, failed ? 0 : sizes.args, capacity.args, I);
                    zeros(offsets.target_entity, failed ? 0 : sizes.targets, capacity.targets, I);
                    zeros(offsets.target_group, failed ? 0 : sizes.targets, capacity.targets, 1);
                    if (failed)
                    {
                        const u64 cells = capacity.graphs * host->schema().num_relations();
                        zeros(offsets.task_ids, 0, capacity.graphs, 4);
                        zeros(offsets.num_objects, 0, capacity.graphs, I);
                        zeros(offsets.entity_offsets, 0, capacity.graphs + 1, I);
                        zeros(offsets.relation_counts, 0, cells, I);
                        zeros(offsets.relation_offsets, 0, cells + 1, I);
                        zeros(offsets.relation_arg_offsets, 0, cells + 1, I);
                        zeros(offsets.goal_complete, 0, capacity.graphs, 1);
                        zeros(offsets.target_offsets, 0, capacity.graphs + 1, I);
                    }
                }
            }
}
}


TEST(DeviceEncoderSchema, RowStreamsRefuseHorizonGraphs)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    encoding::FlatOptions options;
    options.horizon.mode = encoding::HorizonMode::Full;
    const auto schema = encoding::EncodingSchema::create(*table, options);
    EXPECT_NO_THROW(cuda::DeviceEncoder::check_schema(*schema));
    EXPECT_THROW(cuda::check_stream_schema(*schema), std::invalid_argument);
}
