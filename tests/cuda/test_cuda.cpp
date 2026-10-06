// CUDA backend tests (run on GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the context, its pool and the stream-ordered buffers (record_stream orders a free after another stream's work);
//   - device arenas: appends, kernel-written tails, tail syncs to the pinned mirror, growth;
//   - the device task upload of every suite task round-trips byte for byte, validates on the device, and the
//     smoke kernels (applicability, apply, goal test) agree with the CPU engine along random walks, frozen and lazy.

#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "mymyr/cuda/arena.hpp"
#include "mymyr/cuda/device_task.hpp"
#include "mymyr/cuda/kernels.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;
namespace k = mymyr::cuda::kernels;

namespace
{
#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (cuda::device_count() == 0)                                                                                  \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

cuda::ContextPtr context()
{
    // one context per test: the pool and streams are torn down with it (no process-global state)
    cuda::ContextOptions o;
    o.max_bytes = u64{4} << 30;  // well below the 16 GB this machine allows
    return cuda::DeviceContext::create(0, o);
}

template<class T>
std::vector<T> to_host(const void* d, u64 n, cudaStream_t s)
{
    std::vector<T> out(n);
    if (n)
    {
        cuda::check(cudaMemcpyAsync(out.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, s), "D2H");
        cuda::check(cudaStreamSynchronize(s), "sync");
    }
    return out;
}

template<class T>
cuda::DeviceBuffer to_device(const cuda::ContextPtr& ctx, const std::vector<T>& v)
{
    cuda::DeviceBuffer b(ctx, std::max<u64>(1, v.size() * sizeof(T)));
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(b.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, ctx->stream()), "H2D");
    cuda::check(cudaStreamSynchronize(ctx->stream()), "sync");
    return b;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ runtime

TEST(CudaRuntime, ContextPoolAndBuffers)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    EXPECT_EQ(ctx->device(), 0);
    EXPECT_GE(ctx->compute_capability(), 75);
    {
        cuda::DeviceBuffer a(ctx, u64{64} << 20);
        ASSERT_NE(a.data(), nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 256, 0u);
        cuda::check(k::launch_iota(static_cast<u64*>(a.data()), (u64{64} << 20) / 8, 5, ctx->stream()), "iota");
        const auto h = to_host<u64>(a.data(), 1024, ctx->stream());
        for (u64 i = 0; i < h.size(); ++i)
            ASSERT_EQ(h[i], 5 + i);
        EXPECT_GE(ctx->usage().used, u64{64} << 20);
    }
    ctx->synchronize();
    EXPECT_EQ(ctx->usage().used, 0u);
    // the MemoryResource interface: device memory usable on any stream, pinned and managed host-accessible
    void* d = ctx->device_memory().allocate(4096, 256);
    ctx->device_memory().deallocate(d, 4096, 256);
    auto* p = static_cast<u64*>(ctx->pinned_memory().allocate(4096, 64));
    p[0] = 7;
    ctx->pinned_memory().deallocate(p, 4096, 64);
    auto* m = static_cast<u64*>(ctx->managed_memory().allocate(4096, 64));
    cuda::check(k::launch_iota(m, 512, 9, ctx->stream()), "iota");
    ctx->synchronize();
    EXPECT_EQ(m[511], 9u + 511);
    ctx->managed_memory().deallocate(m, 4096, 64);
    ctx->trim(0);
    EXPECT_EQ(ctx->usage().reserved, 0u);
}

// Provokes cudaErrorMemoryAllocation on purpose: compute-sanitizer reports it as an API error, so the memcheck
// sanitizer run excludes this test.
TEST(CudaRuntimeLimits, PoolUpperBoundIsEnforced)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    EXPECT_THROW(cuda::DeviceBuffer(ctx, u64{8} << 30), cuda::CudaError);
    cuda::DeviceBuffer ok(ctx, u64{1} << 20);  // the context stays usable after the failure
    cuda::check(k::launch_iota(static_cast<u64*>(ok.data()), 16, 3, ctx->stream()), "iota");
    EXPECT_EQ(to_host<u64>(ok.data(), 16, ctx->stream())[15], 18u);
}

TEST(CudaRuntime, RecordStreamOrdersTheFreeAfterTheUsersWork)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    cuda::Stream other;
    const u64 n = 1 << 16;
    void* first = nullptr;
    {
        cuda::DeviceBuffer a(ctx, n * 8);
        first = a.data();
        cuda::stream_wait(other, ctx->stream());  // `other` may use a from here on
        a.record_stream(other);
        // a long kernel on `other` writes a after the buffer is gone from the host's point of view
        cuda::check(k::launch_delayed_iota(static_cast<u64*>(a.data()), n, 1000, 200'000'000, other), "delayed");
    }
    // same size, same stream: the pool may hand out the same memory, but only after the delayed kernel
    cuda::DeviceBuffer b(ctx, n * 8);
    cuda::check(k::launch_iota(static_cast<u64*>(b.data()), n, 7, ctx->stream()), "iota");
    cuda::check(cudaStreamSynchronize(other), "sync");
    const auto h = to_host<u64>(b.data(), n, ctx->stream());
    for (u64 i = 0; i < n; ++i)
        ASSERT_EQ(h[i], 7 + i) << "reused before the other stream finished (same memory: " << (first == b.data()) << ")";
}

TEST(CudaRuntime, LastUseFreesAfterTheCallersStreamIsGone)
{
    // A buffer allocated on a caller's stream that is destroyed before the buffer: the free runs on a live stream
    // after the owner's last use (the delayed kernel), and the pool does not hand the memory out earlier
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const u64 n = 1 << 16;
    auto last = std::make_shared<cuda::LastUse>(ctx->device());
    for (bool explicit_stream : {false, true})
    {
        SCOPED_TRACE(explicit_stream ? "reset(stream)" : "destructor");
        std::optional<cuda::DeviceBuffer> a;
        {
            cuda::Stream side;
            a.emplace(ctx, n * 8, side.get());
            a->track(last);
            cuda::check(k::launch_delayed_iota(static_cast<u64*>(a->data()), n, 1000, 200'000'000, side), "delayed");
            last->record(side);
        }  // the caller's stream is gone; its kernel still runs
        cuda::Stream releaser;
        if (explicit_stream)
            a->reset(releaser);
        a.reset();
        cuda::stream_wait(ctx->stream(), releaser);
        cuda::DeviceBuffer b(ctx, n * 8);
        cuda::check(k::launch_iota(static_cast<u64*>(b.data()), n, 7, ctx->stream()), "iota");
        const auto h = to_host<u64>(b.data(), n, ctx->stream());
        for (u64 i = 0; i < n; ++i)
            ASSERT_EQ(h[i], 7 + i) << "reused before the last use finished";
        cuda::check(cudaDeviceSynchronize(), "sync");
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
}

// ------------------------------------------------------------------------------------------------ arenas

TEST(CudaArena, AppendsTailsSyncsAndGrows)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    cuda::Stream other;  // outlives every buffer that records it
    const u32 W = 3;
    cuda::DeviceArena arena(ctx, W * 8, 4);
    std::vector<u64> expect;
    // host appends
    std::vector<u64> rows(5 * W);
    std::iota(rows.begin(), rows.end(), 100);
    arena.append_from_host(rows.data(), 5);
    expect.insert(expect.end(), rows.begin(), rows.end());
    EXPECT_EQ(arena.device_size(), 5u);
    EXPECT_GE(arena.capacity(), 5u);
    arena.sync_to_host();
    EXPECT_EQ(arena.synced_upto(), 5u);
    arena.wait();
    ASSERT_EQ(arena.host_size(), 5u);
    EXPECT_EQ(std::memcmp(arena.host_data(), expect.data(), expect.size() * 8), 0);
    // kernel-written tails, several syncs in flight, growth in between (the old generation stays with its exports)
    std::shared_ptr<cuda::DeviceBuffer> old = arena.generation();
    for (u32 batch = 0; batch < 6; ++batch)
    {
        const u64 n = 7 + 13 * batch;
        auto* t = reinterpret_cast<u64*>(arena.tail(n));
        const u64 first = 1'000 * (batch + 1);
        cuda::check(k::launch_iota(t, n * W, first, arena.stream()), "iota");
        arena.commit(n);
        for (u64 i = 0; i < n * W; ++i)
            expect.push_back(first + i);
        arena.sync_to_host();
    }
    EXPECT_NE(arena.generation(), old);  // grew at least once
    arena.wait();
    EXPECT_TRUE(arena.poll());
    ASSERT_EQ(arena.host_size(), expect.size() / W);
    EXPECT_EQ(std::memcmp(arena.host_data(), expect.data(), expect.size() * 8), 0);
    const auto dev = to_host<u64>(arena.device_data(), expect.size(), arena.stream());
    EXPECT_EQ(dev, expect);
    // the first generation still holds its 5 records (a slice exported before growth stays valid)
    const auto first = to_host<u64>(old->data(), 5 * W, arena.stream());
    EXPECT_TRUE(std::equal(first.begin(), first.end(), rows.begin()));
    // device-to-device appends from another stream's work
    cuda::DeviceBuffer src(ctx, 4 * W * 8, other);
    cuda::check(k::launch_delayed_iota(static_cast<u64*>(src.data()), 4 * W, 77, 50'000'000, other), "delayed");
    arena.append_from_device(src.data(), 4, other);
    src.record_stream(arena.stream());
    arena.sync_to_host();
    arena.wait();
    const u64* h = reinterpret_cast<const u64*>(arena.host_data()) + (arena.host_size() - 4) * W;
    for (u64 i = 0; i < 4 * W; ++i)
        ASSERT_EQ(h[i], 77 + i);
    ctx->synchronize();
}

// ------------------------------------------------------------------------------------------------ gate 2

namespace
{
std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const SuiteTask& t : suite())
        for (bool frozen : {true, false})
            out.emplace_back(t.name, frozen);
    return out;
}

class CudaDeviceTask : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
protected:
    TaskPtr load() const
    {
        TaskOptions o;
        o.atoms = std::get<1>(GetParam()) ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
        return Task::from_text_file(task_path(std::get<0>(GetParam())), o);
    }
};
}  // namespace

TEST_P(CudaDeviceTask, RoundTripValidateAndSmokeKernelsAgreeWithTheEngine)
{
    SKIP_WITHOUT_GPU();
    const auto task = load();
    const std::vector<State> states = device_ref_walks(*task, 3, 30, 29);
    const DeviceRef ref = device_ref(*task, states, 1, 31);
    auto ctx = context();
    cuda::Stream s;  // a second stream shares the task; declared first, so it outlives the buffers recording it
    auto bundle = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*task));
    auto dt = cuda::DeviceTask::upload(ctx, bundle);
    ASSERT_EQ(dt->version(), rl::k_device_arrays_version);

    // round trip: byte-identical
    std::vector<std::byte> back(dt->bytes());
    dt->download(back.data());
    ASSERT_EQ(std::memcmp(back.data(), bundle->block(), bundle->bytes()), 0);
    EXPECT_GE(dt->upload_device_ms(), 0.0f);

    const rl::dev::TaskView& v = dt->acquire(s);

    // validation on the device equals the host's (no failures)
    std::vector<u32> init{0, rl::dev::k_none};
    auto vout = to_device(ctx, init);
    cuda::check(k::launch_validate(v, static_cast<u32*>(vout.data()), s), "validate");
    const auto vres = to_host<u32>(vout.data(), 2, s);
    EXPECT_EQ(vres[0], 0u) << "first failed rule " << vres[1];

    // smoke kernels
    auto d_states = to_device(ctx, ref.states);
    auto d_derived = to_device(ctx, ref.derived);
    auto d_ls = to_device(ctx, ref.label_state);
    auto d_lsch = to_device(ctx, ref.label_schema);
    auto d_lb = to_device(ctx, ref.label_binding);
    const k::DeviceStates st{static_cast<const u64*>(d_states.data()), ref.n(), ref.W, ref.W};
    const k::DeviceLabels lab{static_cast<const u32*>(d_ls.data()), static_cast<const u32*>(d_lsch.data()),
                              static_cast<const u32*>(d_lb.data()), ref.k(), ref.L};
    const u64* der = ref.DW ? static_cast<const u64*>(d_derived.data()) : nullptr;
    cuda::DeviceBuffer app(ctx, ref.k(), s), succ(ctx, ref.k() * ref.W * 8, s), status(ctx, ref.k() * 4, s),
        goal(ctx, ref.n(), s);
    for (auto* b : {&d_states, &d_derived, &d_ls, &d_lsch, &d_lb})
        b->record_stream(s);
    cuda::check(k::launch_applicable(v, st, der, ref.DW, lab, static_cast<u8*>(app.data()), s), "applicable");
    cuda::check(k::launch_apply(v, st, der, ref.DW, lab, static_cast<u64*>(succ.data()), ref.W, static_cast<u32*>(status.data()), s),
                "apply");
    cuda::check(k::launch_goal(v, st, der, ref.DW, static_cast<u8*>(goal.data()), s), "goal");
    const auto h_app = to_host<u8>(app.data(), ref.k(), s);
    const auto h_succ = to_host<u64>(succ.data(), ref.k() * ref.W, s);
    const auto h_status = to_host<u32>(status.data(), ref.k(), s);
    const auto h_goal = to_host<u8>(goal.data(), ref.n(), s);
    u64 applicable = 0, applied = 0, conditional = 0;
    for (u64 i = 0; i < ref.k(); ++i)
    {
        ASSERT_EQ(h_app[i], ref.applicable[i]) << "label " << i << " schema " << ref.label_schema[i];
        if (!ref.applicable[i])
        {
            ASSERT_EQ(h_status[i], rl::dev::k_none) << "label " << i;
            continue;
        }
        ++applicable;
        if (h_status[i] == rl::dev::k_apply_conditional)
        {
            ++conditional;
            continue;
        }
        ASSERT_EQ(h_status[i], rl::dev::k_apply_ok) << "label " << i;
        ASSERT_TRUE(std::equal(h_succ.begin() + static_cast<std::ptrdiff_t>(i * ref.W),
                               h_succ.begin() + static_cast<std::ptrdiff_t>((i + 1) * ref.W),
                               ref.successor.begin() + static_cast<std::ptrdiff_t>(i * ref.W)))
            << "label " << i;
        ++applied;
    }
    EXPECT_EQ(h_goal, ref.goal);
    EXPECT_GT(applicable, 0u);
    EXPECT_EQ(applied + conditional, applicable);
    if (!task->compiled().has_conditional_effects)
    {
        EXPECT_EQ(conditional, 0u);
    }
    RecordProperty("labels", static_cast<int>(ref.k()));
    RecordProperty("states", static_cast<int>(ref.n()));
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaDeviceTask, ::testing::ValuesIn(params()),
                         [](const auto& info)
                         {
                             std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
                             std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                             return n;
                         });

TEST(CudaDeviceTaskSharing, ManyThreadsAndStreamsLaunchOverOneUpload)
{
    SKIP_WITHOUT_GPU();
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    const std::vector<State> states = device_ref_walks(*task, 4, 40, 3);
    const DeviceRef ref = device_ref(*task, states, 1, 5);
    auto ctx = context();
    constexpr int T = 8;
    std::vector<std::unique_ptr<cuda::Stream>> streams;  // outlive the task and buffers that record them
    for (int t = 0; t < T; ++t)
        streams.push_back(std::make_unique<cuda::Stream>());
    auto dt = cuda::DeviceTask::upload(ctx, *task);
    auto d_states = to_device(ctx, ref.states);
    auto d_ls = to_device(ctx, ref.label_state);
    auto d_lsch = to_device(ctx, ref.label_schema);
    auto d_lb = to_device(ctx, ref.label_binding);
    std::vector<std::vector<u8>> results(T);
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t)
        threads.emplace_back(
            [&, t]
            {
                const cudaStream_t s = *streams[t];
                const rl::dev::TaskView& v = dt->acquire(s);
                d_states.record_stream(s);
                d_ls.record_stream(s);
                d_lsch.record_stream(s);
                d_lb.record_stream(s);
                cuda::DeviceBuffer out(ctx, ref.k(), s);
                const k::DeviceStates st{static_cast<const u64*>(d_states.data()), ref.n(), ref.W, ref.W};
                const k::DeviceLabels lab{static_cast<const u32*>(d_ls.data()), static_cast<const u32*>(d_lsch.data()),
                                          static_cast<const u32*>(d_lb.data()), ref.k(), ref.L};
                for (int r = 0; r < 20; ++r)
                    cuda::check(k::launch_applicable(v, st, nullptr, 0, lab, static_cast<u8*>(out.data()), s), "applicable");
                results[t] = to_host<u8>(out.data(), ref.k(), s);
                out.reset();
            });
    for (auto& th : threads)
        th.join();
    for (int t = 0; t < T; ++t)
        EXPECT_EQ(results[t], ref.applicable) << "thread " << t;
}
