// Device encoding context and encoder skeleton tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the downloaded block equals the host block byte for byte, for the fixture tasks and the benchmark tasks (PDDL
//     under tests/data and the fork's data; a task whose files are missing is skipped);
//   - a grown table's context is appended: the existing instances keep their rows, offsets and bytes; one upload per
//     refresh;
//   - views rebased onto the device read the same values as the host views (a probe kernel);
//   - the sizes of a batch: the formula of bytes_needed, pinned on a task, and reserve() allocating exactly that once;
//   - lifetime with work in flight, and the refusals.

#include "../cpp/encoding/support.hpp"
#include "../cpp/frontend/golden.hpp"
#include "context_probe.hpp"
#include "mymyr/cuda/encoding.hpp"

#include <gtest/gtest.h>

#include <cstring>
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

/// A PDDL file of tests/data or, under ipc/, the fork's data.
std::optional<fs::path> find_file(const std::string& relative)
{
    for (const fs::path& root : {test::test_data_dir(), test::fork_data_dir()})
        if (fs::exists(root / relative))
            return root / relative;
    return std::nullopt;
}

/// The tasks of one domain directory under tests/data or the fork's data (`domain` is relative to it, the problems
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
        {"F1_F2_blocks", "pddl/blocks", {"smedium", "probBLOCKS-8-0"}},
        {"F3_logistics", "pddl/logistics00", {"probLOGISTICS-6-1"}},
        {"F4_childsnack", "ipc/childsnack-ipc/test", {"p01-easy"}},
        {"F5_philosophers", "pddl/philosophers", {"p03-phil4"}},
        {"F6_rovers", "ipc/rovers-ipc/test", {"p01-easy"}},
        {"F7_satellite", "ipc/satellite-ipc/test", {"p01-easy"}},
        {"blocks_p01e_p05m_p20m", "ipc/blocksworld-ipc/test", {"p01-easy", "p05-medium", "p20-medium"}},
        {"childsnack_p16e", "ipc/childsnack-ipc/test", {"p16-easy"}},
        {"rovers_p16e", "ipc/rovers-ipc/test", {"p16-easy"}},
        {"sokoban_p16e", "ipc/sokoban-ipc/test", {"p16-easy"}},
        {"transport_p30m", "ipc/transport-ipc/test", {"p30-medium"}},
        {"miconic_p30m", "ipc/miconic-ipc/test", {"p30-medium"}},
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
        device->acquire(other);
        EXPECT_TRUE(device->download(other) == host_bytes(*host));
        EXPECT_EQ(device->uploads(), 1u);
        ++checked;
    }
    if (checked == 0)
        GTEST_SKIP() << "no PDDL data";
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
    EXPECT_EQ(device_probe(device->view(), 3, ctx, ctx->stream()), v1);
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
        const auto dv = device->view();
        EXPECT_EQ(dv.num_instances, table->size());
        EXPECT_EQ(dv.max_arity, host->max_arity());
        EXPECT_EQ(dv.num_relations, host->schema().num_relations());
        // the device pointers are the host pointers rebased onto the copy
        const auto hv = host->view();
        EXPECT_EQ(reinterpret_cast<const std::byte*>(dv.cells) - static_cast<const std::byte*>(device->data()),
                  reinterpret_cast<const std::byte*>(hv.cells) - host->arrays().block());
        EXPECT_EQ(device_probe(dv, table->size(), ctx, ctx->stream()), host_probe(hv, table->size()));
        ++checked;
    }
    if (checked == 0)
        GTEST_SKIP() << "no PDDL data";
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
    device->acquire(s);
    cuda::DeviceBuffer out(ctx, test::k_probe_width * 8, s);
    test::launch_context_probe(device->view(), 1, static_cast<u64*>(out.data()), s);
    EXPECT_TRUE(device->refresh(g->after, s));  // the old block frees after the probe
    std::vector<u64> probe(test::k_probe_width);
    cuda::check(cudaMemcpyAsync(probe.data(), out.data(), probe.size() * 8, cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    EXPECT_EQ(probe, host_probe(g->before->view(), 1));
}

// ------------------------------------------------------------------------------------------------ lifetime

TEST(DeviceEncodingContext, DestroyedWithWorkInFlightOnAnotherStream)
{
    SKIP_WITHOUT_GPU();
    const auto g = blocks_grown();
    if (!g)
        GTEST_SKIP() << "no PDDL data";
    const auto ctx = device_context();
    cuda::Stream stream;
    cuda::DeviceBuffer out(ctx, 3 * test::k_probe_width * 8, stream);
    {
        const auto device = cuda::DeviceEncodingContext::upload(ctx, g->after);
        device->acquire(stream);
        test::launch_context_probe(device->view(), 3, static_cast<u64*>(out.data()), stream);
    }  // destroyed before the probe has necessarily run
    std::vector<u64> probe(3 * test::k_probe_width);
    cuda::check(cudaMemcpyAsync(probe.data(), out.data(), probe.size() * 8, cudaMemcpyDeviceToHost, stream), "D2H");
    stream.synchronize();
    EXPECT_EQ(probe, host_probe(g->after->view(), 3));
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

TEST(DeviceEncoder, ScratchBytesAreTheFourTermsRoundedUp)
{
    // 24 G + 24 G + 4 G C + the scan of G + 1 values, each rounded up to 64
    for (const u64 graphs : {u64{1}, u64{10}, u64{1000}, u64{16384}})
        for (const u64 cells : {u64{0}, u64{6}, u64{170}})
        {
            const u64 fixed = round_up(24 * graphs) * 2 + round_up(4 * graphs * cells);
            const u64 total = cuda::scratch_bytes(graphs, cells);
            ASSERT_GT(total, fixed);
            const u64 scan = total - fixed;
            EXPECT_EQ(scan % 64, 0u);
            EXPECT_LE(scan, u64{1} << 20);
        }
    // the scan term does not depend on the cells
    EXPECT_EQ(cuda::scratch_bytes(1000, 6) - cuda::scratch_bytes(1000, 0), round_up(4 * 1000 * 6));
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
    EXPECT_EQ(device.bytes_needed(worst), cuda::scratch_bytes(G, b.cells) + output);
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
    EXPECT_EQ(device.bytes_needed(u64{1000}), 1074560u);
}

TEST(DeviceEncoder, ReserveAllocatesOnceAndNothingAfter)
{
    SKIP_WITHOUT_GPU();
    const auto encoder = blocks_encoder(encoding::IndexType::I64);
    if (!encoder)
        GTEST_SKIP() << "no PDDL data";
    const auto ctx = device_context();
    cuda::Stream s1, s2;  // outlive the encoder
    cuda::DeviceEncoder device(ctx, encoder);
    EXPECT_EQ(device.context().uploads(), 1u);
    EXPECT_EQ(device.scratch(), nullptr);
    EXPECT_EQ(device.reserved_bytes(), 0u);
    ctx->synchronize();
    const u64 before = ctx->usage().used;
    device.reserve(u64{4096});
    const u64 needed = device.bytes_needed(u64{4096});
    EXPECT_EQ(device.reserved_bytes(), needed);
    EXPECT_EQ(device.reserved(), device.worst_case(4096));
    EXPECT_EQ(device.reserved_layout().bytes, cuda::output_layout(device.reserved(), encoding::IndexType::I64, encoder->schema().num_relations()).bytes);
    EXPECT_GE(ctx->usage().used - before, needed);
    const u64 after = ctx->usage().used;
    void* const scratch = device.scratch();
    void* const output = device.output();
    ASSERT_NE(scratch, nullptr);
    ASSERT_NE(output, nullptr);
    // smaller and equal batches, and acquires from new streams, allocate nothing and move nothing
    device.reserve(u64{1});
    device.reserve(u64{4096});
    device.reserve(cuda::BatchShape{10, 1, 1});
    device.acquire(s1);
    device.acquire(s2);
    EXPECT_EQ(ctx->usage().used, after);
    EXPECT_EQ(device.scratch(), scratch);
    EXPECT_EQ(device.output(), output);
    EXPECT_EQ(device.reserved(), device.worst_case(4096));
    // a larger batch replaces the memory, to the elementwise maximum
    device.reserve(cuda::BatchShape{8192, 1, 1});
    EXPECT_EQ(device.reserved().graphs, 8192u);
    EXPECT_EQ(device.reserved().entities, device.worst_case(4096).entities);
    EXPECT_EQ(device.reserved_bytes(), device.bytes_needed(device.reserved()));
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
