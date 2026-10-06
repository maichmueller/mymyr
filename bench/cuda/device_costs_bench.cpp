// Costs of the CUDA foundations on this machine, as JSON lines.
//   mymyr_device_costs_bench upload [--reps R] [--atoms frozen|lazy]   per suite task: export (host), upload (wall, device), bytes
//   mymyr_device_costs_bench sync [--reps R]                          the probe's round trip, and tail syncs of DeviceArena
//                                                           against the raw pinned copies of probes/gpu/sync_cost.cu
// Run with CUDA_VISIBLE_DEVICES=0.

#include "../../tests/cpp/support/suite.hpp"
#include "mymyr/cuda/arena.hpp"
#include "mymyr/cuda/device_task.hpp"
#include "mymyr/cuda/kernels.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mymyr;
namespace k = mymyr::cuda::kernels;

namespace
{
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template<class F>
double median_us(int reps, F&& f)
{
    std::vector<double> t;
    for (int r = 0; r < reps; ++r)
    {
        const double a = now_s();
        f();
        t.push_back((now_s() - a) * 1e6);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int upload(int reps, bool frozen)
{
    auto ctx = cuda::DeviceContext::create(0);
    {
        // warm the pool and the copy engine once, outside the measurements
        auto warm = cuda::DeviceTask::upload(ctx, *Task::from_text_file(test::task_path("blocks__probBLOCKS-8-0")));
        ctx->synchronize();
    }
    for (const test::SuiteTask& st : test::suite())
    {
        TaskOptions o;
        o.atoms = frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
        const auto task = Task::from_text_file(test::task_path(st.name), o);
        std::vector<double> export_s, wall_s, device_ms, total_s;
        u64 bytes = 0;
        bool round_trip = true;
        for (int r = 0; r < reps; ++r)
        {
            const double t0 = now_s();
            auto bundle = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*task));
            const double t1 = now_s();
            auto dt = cuda::DeviceTask::upload(ctx, bundle);
            dt->acquire(ctx->stream());
            cuda::check(cudaStreamSynchronize(ctx->stream()), "sync");
            const double t2 = now_s();
            export_s.push_back(t1 - t0);
            wall_s.push_back(t2 - t1);  // allocation + H2D + completion
            total_s.push_back(t2 - t0);
            device_ms.push_back(dt->upload_device_ms());
            bytes = dt->bytes();
            if (r == 0)
            {
                std::vector<std::byte> back(bytes);
                dt->download(back.data());
                round_trip = std::memcmp(back.data(), bundle->block(), bytes) == 0;
            }
        }
        std::printf("{\"bench\":\"upload\",\"task\":\"%s\",\"atoms\":\"%s\",\"bytes\":%llu,\"export_ms\":%.3f,"
                    "\"upload_ms\":%.3f,\"h2d_device_ms\":%.4f,\"total_ms\":%.3f,\"h2d_GBps\":%.2f,\"round_trip\":%s,\"reps\":%d}\n",
                    st.name.c_str(), frozen ? "frozen" : "lazy", static_cast<unsigned long long>(bytes),
                    median(export_s) * 1e3, median(wall_s) * 1e3, median(device_ms), median(total_s) * 1e3,
                    bytes / (median(device_ms) * 1e6), round_trip ? "true" : "false", reps);
        std::fflush(stdout);
    }
    return 0;
}

int sync(int reps)
{
    auto ctx = cuda::DeviceContext::create(0);
    const cudaStream_t s = ctx->stream();
    // (a) the probe's control round trip: tiny kernel + 4-byte D2H, and kernel + stream sync
    {
        cuda::DeviceBuffer c(ctx, 4);
        cuda::check(cudaMemsetAsync(c.data(), 0, 4, s), "memset");
        cuda::PinnedBuffer h(4);
        const double rt = median_us(2000, [&] {
            cuda::check(k::launch_touch(static_cast<u32*>(c.data()), s), "touch");
            cuda::check(cudaMemcpyAsync(h.data(), c.data(), 4, cudaMemcpyDeviceToHost, s), "d2h");
            cuda::check(cudaStreamSynchronize(s), "sync");
        });
        const double ks = median_us(2000, [&] {
            cuda::check(k::launch_touch(static_cast<u32*>(c.data()), s), "touch");
            cuda::check(cudaStreamSynchronize(s), "sync");
        });
        std::printf("{\"bench\":\"roundtrip\",\"kernel_plus_4B_d2h_us\":%.1f,\"kernel_plus_sync_us\":%.1f,"
                    "\"probe_kernel_plus_4B_d2h_us\":8.6,\"probe_kernel_plus_sync_us\":5.6}\n",
                    rt, ks);
    }
    // (b) tail syncs: a kernel writes a tail of `bytes` into the arena, then sync_to_host + wait. Against the raw
    // pinned cudaMemcpy of the same bytes (sync_cost's "pinned") measured in the same process.
    const u64 probe_bytes[] = {u64{4} << 10, u64{64} << 10, u64{1} << 20, u64{16} << 20, u64{256} << 20};
    const double probe_pinned_us[] = {5.3, 7.6, 45.3, 641.9, 10179.0};
    const u64 max_bytes = u64{256} << 20;
    for (usize bi = 0; bi < std::size(probe_bytes); ++bi)
    {
        const u64 bytes = probe_bytes[bi];
        const int r = bytes >= (u64{256} << 20) ? std::max(5, reps / 10) : reps;
        const u32 rb = 16;  // W = 2 state words per record
        const u64 recs = bytes / rb;
        auto arena = std::make_unique<cuda::DeviceArena>(ctx, rb, (max_bytes / rb) + 1);
        cuda::PinnedBuffer pin(bytes);
        // raw pinned copy (the probe's number), same buffer sizes
        const double raw = median_us(r, [&] {
            cuda::check(cudaMemcpyAsync(pin.data(), arena->device_data(), bytes, cudaMemcpyDeviceToHost, ctx->copy_stream()), "d2h");
            cuda::check(cudaStreamSynchronize(ctx->copy_stream()), "sync");
        });
        // the arena: each repetition appends a fresh tail (kernel), then syncs it; timed from the sync call to landing
        std::vector<double> t;
        for (int i = 0; i < r; ++i)
        {
            if (arena->device_size() + recs > arena->capacity())
            {
                arena.reset();  // full: start over with a fresh arena of the same capacity
                arena = std::make_unique<cuda::DeviceArena>(ctx, rb, (max_bytes / rb) + 1);
            }
            auto* tail = reinterpret_cast<u64*>(arena->tail(recs));
            cuda::check(k::launch_iota(tail, recs * 2, 0, s), "iota");
            arena->commit(recs);
            cuda::check(cudaStreamSynchronize(s), "sync");  // measure the sync alone, not the kernel before it
            const double a = now_s();
            arena->sync_to_host();
            arena->wait();
            t.push_back((now_s() - a) * 1e6);
        }
        const double tail = median(t);
        bool ok = true;
        {
            const auto* h = reinterpret_cast<const u64*>(arena->host_data());
            const u64 n = arena->host_size() * 2;
            for (u64 i = 0; i < std::min<u64>(n, 1024); ++i)
                ok &= (h[i] & 0xFFFFFFFFull) == (i % (recs * 2));
        }
        auto gbs = [&](double us) { return bytes / (us * 1e3); };
        std::printf("{\"bench\":\"tail_sync\",\"bytes\":%llu,\"arena_sync_us\":%.1f,\"arena_GBps\":%.2f,\"raw_pinned_us\":%.1f,"
                    "\"raw_GBps\":%.2f,\"overhead_us\":%.1f,\"probe_pinned_us\":%.1f,\"probe_GBps\":%.2f,\"content_ok\":%s,\"reps\":%zu}\n",
                    static_cast<unsigned long long>(bytes), tail, gbs(tail), raw, gbs(raw), tail - raw, probe_pinned_us[bi],
                    gbs(probe_pinned_us[bi]), ok ? "true" : "false", t.size());
        std::fflush(stdout);
    }
    // (c) overlap: the tail sync of layer k (256 MB) on the copy stream while a DRAM-heavy kernel (layer k + 1) runs
    // on the compute stream: sync_cost's (d), through the arena
    {
        const u64 bytes = u64{256} << 20;
        const u32 rb = 16;
        const u64 recs = bytes / rb;
        cuda::DeviceBuffer busy(ctx, u64{256} << 20);
        cuda::check(cudaMemsetAsync(busy.data(), 0, u64{256} << 20, s), "memset");
        const u64 nb = (u64{256} << 20) / 8;
        const int iters = 200;
        const double kernel = median_us(5, [&] {
            cuda::check(k::launch_busy(static_cast<u64*>(busy.data()), nb, iters, s), "busy");
            cuda::check(cudaStreamSynchronize(s), "sync");
        });
        std::vector<double> copy_t, both_t;
        for (int i = 0; i < 5; ++i)
        {
            cuda::DeviceArena arena(ctx, rb, recs);
            auto* tail = reinterpret_cast<u64*>(arena.tail(recs));
            cuda::check(k::launch_iota(tail, recs * 2, 1, s), "iota");
            arena.commit(recs);
            cuda::check(cudaStreamSynchronize(s), "sync");
            double a = now_s();
            arena.sync_to_host();
            arena.wait();
            copy_t.push_back((now_s() - a) * 1e6);
            // again with the next layer's kernel behind the sync's event
            cuda::DeviceArena arena2(ctx, rb, recs);
            tail = reinterpret_cast<u64*>(arena2.tail(recs));
            cuda::check(k::launch_iota(tail, recs * 2, 1, s), "iota");
            arena2.commit(recs);
            cuda::check(cudaStreamSynchronize(s), "sync");
            a = now_s();
            arena2.sync_to_host();  // waits for the work on s so far (the tail), not for the kernel below
            cuda::check(k::launch_busy(static_cast<u64*>(busy.data()), nb, iters, s), "busy");
            arena2.wait();
            cuda::check(cudaStreamSynchronize(s), "sync");
            both_t.push_back((now_s() - a) * 1e6);
        }
        const double copy = median(copy_t), both = median(both_t);
        std::printf("{\"bench\":\"overlap\",\"bytes\":%llu,\"kernel_us\":%.0f,\"sync_us\":%.0f,\"both_us\":%.0f,"
                    "\"serial_sum_us\":%.0f,\"hidden_fraction\":%.2f,\"probe_hidden_fraction\":0.38}\n",
                    static_cast<unsigned long long>(bytes), kernel, copy, both, kernel + copy,
                    (kernel + copy - both) / std::min(kernel, copy));
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s upload|sync [--reps R] [--atoms frozen|lazy]\n", argv[0]);
        return 2;
    }
    int reps = 20;
    bool frozen = true;
    for (int i = 2; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--reps")
            reps = std::stoi(argv[i + 1]);
        else if (a == "--atoms")
            frozen = std::string(argv[i + 1]) != "lazy";
    }
    const std::string mode = argv[1];
    try
    {
        if (mode == "upload")
            return upload(reps, frozen);
        if (mode == "sync")
            return sync(reps);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
