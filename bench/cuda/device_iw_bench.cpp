// mymyr_device_iw_bench: device rollouts and exact batched IW(1) against the CPU, printing one JSON line.
//
//   mymyr_device_iw_bench task.txt [--mode rollouts|iw1] [--n N] [--next K] [--arity A] [--sample S] [--threads T]
//                  [--reps R] [--atoms auto|lazy|frozen] [--chunk N] [--max-bytes B] [--walk-steps S] [--max-states N]
//                  [--graphs 0|1] [--dedup 0|1]
//
// rollouts: N device rollouts (seeds 1 + i * golden) from the initial state (cuda::rollouts_batch; max_arity A,
// max_next_layer_states K) against search::find_rollouts_parallel on the first S seeds with 1 thread (the one-core
// rate) and with T threads (0: every core) on min(N, S * T) seeds. iw1: exact batched IW(1) (cuda::batched_iw1) from
// N random-walk states (walks of --walk-steps steps from the initial state) uploaded once, against search::iw() from
// the first S starts on one core and from min(N, S * T) starts on T threads. Device times are the median of R runs
// (a fresh DeviceMultiIw each: the task upload is included, the context creation is not). The CPU rates are
// extrapolated to N (per-search time x N). The device results of the sampled searches are checked against the CPU's
// (status, plan length, reached atoms): "equal" in the output. The load average and the core count are reported.

#include "mymyr/core/random.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/rollouts.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include <stdlib.h>  // getloadavg

using namespace mymyr;

namespace
{
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

/// FNV-1a over the device batch (statuses, widths, plans, goal rows, pass statistics without their times, reached
/// sets): equal batches of two builds have equal prints.
u64 batch_fp(const cuda::MultiIwBatch& b)
{
    u64 h = 0xcbf29ce484222325ull;
    auto add = [&](u64 x)
    {
        for (int k = 0; k < 8; ++k)
        {
            h ^= (x >> (8 * k)) & 0xFF;
            h *= 0x100000001b3ull;
        }
    };
    add(b.n);
    for (u32 i = 0; i < b.n; ++i)
        add(static_cast<u64>(b.status[i]));
    for (const u32 x : b.effective_width)
        add(x);
    for (const i32 x : b.plan_length)
        add(static_cast<u64>(static_cast<u32>(x)));
    for (const u64 x : b.plan_offsets)
        add(x);
    for (const u32 x : b.plan_labels)
        add(x);
    for (const u64 x : b.goal_rows)
        add(x);
    for (const auto& x : b.pass_stats)
    {
        add(x.arity);
        add(static_cast<u64>(x.status));
        add(x.expanded);
        add(x.generated);
        add(x.generated_in_tree);
        add(x.skipped);
        add(x.blocked);
        add(x.placeholder ? 1 : 0);
    }
    for (const u8 x : b.num_passes)
        add(x);
    for (const u64 x : b.reached)
        add(x);
    return h;
}

[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_iw_bench task.txt [--mode rollouts|iw1] [--n N] [--next K] [--arity A] [--sample S]\n"
                 "       [--threads T] [--reps R] [--atoms auto|lazy|frozen] [--chunk N] [--max-bytes B] [--walk-steps S]\n",
                 msg);
    std::exit(2);
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

/// n random-walk states: walk k takes steps uniformly among the successors (SplitMix64(seed + k)), its end state is
/// start k.
std::vector<State> walk_states(const Task& task, u32 n, u32 steps, u64 seed)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    std::vector<State> out;
    std::vector<u64> tmp;
    for (u32 k = 0; k < n; ++k)
    {
        SplitMix64 rng(seed + k);
        State s = task.initial_state();
        const u32 len = static_cast<u32>(rng.bounded(steps + 1));
        for (u32 i = 0; i < len; ++i)
        {
            std::vector<State> kids;
            succ.prepare(s.view());
            succ.generate<false>(
                [&](u32, const ObjectId*, const Delta& d)
                {
                    const u32 w = apply_delta(s.data(), s.size_words(), d, tmp);
                    kids.emplace_back(tmp.data(), w);
                    return true;
                },
                false, true);
            if (kids.empty())
                break;
            s = kids[rng.bounded(kids.size())];
        }
        out.push_back(std::move(s));
    }
    return out;
}

/// f(i) for i < n on t threads.
template<class F>
void parallel_for(u32 n, u32 t, F&& f)
{
    std::atomic<u32> next{0};
    std::vector<std::thread> pool;
    for (u32 k = 0; k < t; ++k)
        pool.emplace_back(
            [&]
            {
                for (u32 i; (i = next.fetch_add(1)) < n;)
                    f(i);
            });
    for (std::thread& th : pool)
        th.join();
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing task file");
    const std::string path = argv[1];
    TaskOptions to;
    std::string mode = "rollouts";
    u32 n = 4096, next = ~u32{0}, arity = 1, sample = 256, threads = 0, chunk = 0, walk_steps = 30;
    int reps = 3;
    u64 max_bytes = u64{6} << 30;
    u64 max_states = ~u64{0};
    int graphs = -1, dedup = -1;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--mode")
            mode = value();
        else if (a == "--n")
            n = static_cast<u32>(std::stoul(value()));
        else if (a == "--next")
            next = static_cast<u32>(std::stoul(value()));
        else if (a == "--arity")
            arity = static_cast<u32>(std::stoul(value()));
        else if (a == "--sample")
            sample = static_cast<u32>(std::stoul(value()));
        else if (a == "--threads")
            threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--reps")
            reps = std::stoi(value());
        else if (a == "--chunk")
            chunk = static_cast<u32>(std::stoul(value()));
        else if (a == "--max-states")
            max_states = std::stoull(value());
        else if (a == "--walk-steps")
            walk_steps = static_cast<u32>(std::stoul(value()));
        else if (a == "--max-bytes")
            max_bytes = std::stoull(value());
        else if (a == "--graphs")
            graphs = std::stoi(value());
        else if (a == "--dedup")
            dedup = std::stoi(value());
        else if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else
            usage(("unknown option " + a).c_str());
    }
    if (mode != "rollouts" && mode != "iw1")
        usage("--mode must be rollouts or iw1");
    try
    {
        const TaskPtr task = Task::from_text_file(path, to);
        const u32 cores = std::max(1u, std::thread::hardware_concurrency());
        const u32 T = threads ? threads : cores;
        sample = std::min(sample, n);
        const u32 n_all = std::min<u32>(n, sample * T);
        double load[3] = {0, 0, 0};
        if (getloadavg(load, 3) < 0)
            load[0] = load[1] = load[2] = -1;

        cuda::ContextOptions co;
        co.max_bytes = max_bytes;
        const auto ctx = cuda::DeviceContext::create(0, co);
        cuda::MultiIwOptions mo;
        mo.max_arity = arity;
        mo.max_next_layer_states = next;
        mo.budget.max_states = max_states;
        if (chunk)
            mo.chunk_states = chunk;
        if (graphs >= 0)
            mo.graphs = graphs != 0;
        if (dedup >= 0)
            mo.dedup_parents = dedup != 0;

        std::vector<double> dev_times;
        cuda::MultiIwBatch batch;
        double cpu1 = 0, cpuT = 0, first = 0;
        bool equal = true;
        u64 solved = 0, device_bytes = 0;
        double host_ms = 0;
        if (mode == "rollouts")
        {
            cuda::DeviceRolloutOptions ro;
            ro.iw = mo;
            for (u32 i = 0; i < n; ++i)
                ro.seeds.push_back(1 + i * 0x9e3779b97f4a7c15ULL);
            {
                auto t0 = Clock::now();
                batch = cuda::rollouts_batch(ctx, task, ro);  // cold: the task upload and every allocation
                first = since(t0);
                cuda::DeviceRollouts x(ctx, task, mo);
                (void)x.run(ro.seeds);
                for (int r = 0; r < reps; ++r)
                {
                    t0 = Clock::now();
                    batch = x.run(ro.seeds);
                    dev_times.push_back(since(t0));
                }
            }
            {
                // CPU warm-up (workspaces, lazy slots)
                search::ParallelRolloutOptions w;
                w.iw.max_arity = arity;
                w.iw.control.budget.max_states = max_states;
                w.max_next_layer_states = next;
                w.seeds.assign(ro.seeds.begin(), ro.seeds.begin() + std::min<u32>(n, 8));
                w.num_threads = 1;
                (void)search::find_rollouts_parallel(*task, w);
            }
            search::ParallelRolloutOptions c;
            c.iw.max_arity = arity;
            c.iw.control.budget.max_states = max_states;
            c.max_next_layer_states = next;
            c.seeds.assign(ro.seeds.begin(), ro.seeds.begin() + sample);
            c.num_threads = 1;
            auto t0 = Clock::now();
            const search::ParallelRolloutsResult one = search::find_rollouts_parallel(*task, c);
            cpu1 = since(t0) / sample;
            c.seeds.assign(ro.seeds.begin(), ro.seeds.begin() + n_all);
            c.num_threads = T;
            t0 = Clock::now();
            const search::ParallelRolloutsResult all = search::find_rollouts_parallel(*task, c);
            cpuT = since(t0) / n_all;
            for (u32 i = 0; i < sample; ++i)
            {
                const search::RolloutResult& x = one.rollouts[i];
                const i32 len = x.search.status == search::SearchStatus::Solved ? static_cast<i32>(x.search.plan.size()) : -1;
                equal = equal && batch.status[i] == x.search.status && batch.plan_length[i] == len &&
                        cuda::reached_atoms(batch, i, *task) == x.reached_fluent_atoms;
            }
            (void)all;
        }
        else
        {
            const std::vector<State> starts = walk_states(*task, n, walk_steps, 12345);
            const u32 W = std::max<u32>(1, task->words());
            std::vector<u64> rows(u64{n} * W, 0);
            for (u32 i = 0; i < n; ++i)
                std::copy_n(starts[i].data(), std::min(starts[i].size_words(), W), rows.begin() + static_cast<std::ptrdiff_t>(u64{i} * W));
            cuda::DeviceBuffer dev(ctx, std::max<u64>(rows.size(), 1) * sizeof(u64), ctx->stream());
            cuda::check(cudaMemcpyAsync(dev.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, ctx->stream()), "H2D");
            cuda::check(cudaStreamSynchronize(ctx->stream()), "sync");
            const cuda::DeviceStarts ds{static_cast<const u64*>(dev.data()), W, W, n};
            {
                auto t0 = Clock::now();
                batch = cuda::batched_iw1(ctx, task, ds, mo, ctx->stream());  // cold
                first = since(t0);
                mo.max_arity = 1;
                cuda::DeviceMultiIw x(ctx, task, mo);
                (void)x.run(ds);
                for (int r = 0; r < reps; ++r)
                {
                    t0 = Clock::now();
                    batch = x.run(ds);
                    dev_times.push_back(since(t0));
                }
            }
            auto cpu_iw = [&](u32 i)
            {
                search::IwOptions c;
                c.max_arity = 1;
                c.control.budget.max_states = max_states;
                c.start = starts[i];
                return search::iw(*task, c);
            };
            (void)cpu_iw(0);  // CPU warm-up
            std::vector<search::IwResult> one(sample);
            auto t0 = Clock::now();
            for (u32 i = 0; i < sample; ++i)
                one[i] = cpu_iw(i);
            cpu1 = since(t0) / sample;
            t0 = Clock::now();
            parallel_for(n_all, T, [&](u32 i) { (void)cpu_iw(i); });
            cpuT = since(t0) / n_all;
            for (u32 i = 0; i < sample; ++i)
            {
                const search::IwResult& x = one[i];
                const i32 len = x.status == search::SearchStatus::Solved ? static_cast<i32>(x.plan.size()) : -1;
                equal = equal && batch.status[i] == x.status && batch.plan_length[i] == len && batch.expanded(i) == x.total.expanded &&
                        batch.generated(i) == x.total.generated;
            }
        }
        for (u32 i = 0; i < batch.n; ++i)
            solved += batch.status[i] == search::SearchStatus::Solved;
        device_bytes = batch.stats.device_bytes;
        host_ms = batch.stats.host_ms;
        const double dev = median(dev_times);
        std::printf("{\"task\": \"%s\", \"mode\": \"%s\", \"n\": %u, \"arity\": %u, \"next\": %lld, \"solved\": %llu, "
                    "\"device_first_s\": %.6f, \"device_s\": %.6f, \"device_per_s\": %.1f, \"cpu1_per_search_ms\": %.4f, \"cpu1_s_extrapolated\": %.3f, "
                    "\"cpuT_threads\": %u, \"cpuT_per_search_ms\": %.4f, \"cpuT_s_extrapolated\": %.3f, "
                    "\"speedup_vs_1core\": %.1f, \"speedup_vs_all_cores\": %.1f, \"sample\": %u, \"sample_all\": %u, \"equal\": %s, "
                    "\"host_ms\": %.1f, \"chunks\": %llu, \"splits\": %llu, \"redone\": %llu, \"distinct\": %llu, \"replays\": %llu, "
                    "\"captures\": %u, \"device_loops\": %llu, \"loop_handoffs\": %llu, \"nodes\": %llu, \"candidates\": %llu, \"device_bytes\": %llu, "
                    "\"cores\": %u, \"load1\": %.2f, \"load5\": %.2f, \"batch_fp\": \"%016llx\", \"device_runs\": [",
                    path.c_str(), mode.c_str(), n, arity, next == ~u32{0} ? -1LL : static_cast<long long>(next),
                    static_cast<unsigned long long>(solved), first, dev, n / dev, cpu1 * 1e3, cpu1 * n, T, cpuT * 1e3, cpuT * n,
                    cpu1 * n / dev, cpuT * n / dev, sample, n_all, equal ? "true" : "false", host_ms,
                    static_cast<unsigned long long>(batch.stats.chunks), static_cast<unsigned long long>(batch.stats.splits),
                    static_cast<unsigned long long>(batch.stats.redone), static_cast<unsigned long long>(batch.stats.distinct),
                    static_cast<unsigned long long>(batch.stats.replays), batch.stats.captures, static_cast<unsigned long long>(batch.stats.device_loops),
                    static_cast<unsigned long long>(batch.stats.loop_handoffs),
                    static_cast<unsigned long long>(batch.stats.nodes), static_cast<unsigned long long>(batch.stats.candidates),
                    static_cast<unsigned long long>(device_bytes), cores, load[0], load[1], static_cast<unsigned long long>(batch_fp(batch)));
        for (usize i = 0; i < dev_times.size(); ++i)
            std::printf("%s%.6f", i ? ", " : "", dev_times[i]);
        std::printf("]}\n");
        return equal ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
