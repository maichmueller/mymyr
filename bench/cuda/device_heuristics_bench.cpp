// mymyr_device_heuristics_bench: the batched device heuristics against the CPU heuristic, and device A* / GBFS against the CPU
// searches, printing one JSON line per batch size.
//
//   mymyr_device_heuristics_bench task.txt [--mode heur|astar|gbfs] [--h max|add|ff|blind] [--b 1000,10000,100000]
//                  [--variant auto|sweep|frontier] [--warp -1|0|1] [--threads T] [--global] [--reps R] [--sample S]
//                  [--cpu-threads T] [--walk-steps S] [--atoms auto|lazy|frozen] [--distinct N]
//                  [--budget N] [--seconds S] [--single-bucket] [--hcosts unit|real]
//   mymyr_device_heuristics_bench domain.pddl --problem problem.pddl [...]   (PDDL: needs the front end)
//
// heur (the default):
// States: N distinct random-walk states (walk k from the initial state with SplitMix64(seed + k), a uniform length up
// to --walk-steps), tiled to B rows on the device. Device: DeviceHeuristic::evaluate of the B rows (u32 values), the
// median of R timed runs after one warm-up (the grounding and upload are not included; the wall time includes the
// launch and the synchronization). CPU: heuristics::make_heuristic (grounded, sharing the grounding) on the first S
// distinct states on one core, and on T threads (0: every core) each evaluating the S states with its own heuristic.
// The device values of the sampled states are checked against the CPU's (h_max, h_add) or DeviceHeuristic::reference
// (h_FF): "equal" in the output. The load average and the core count are reported.
//
// astar / gbfs: search::astar_eager / gbfs_eager on one core (max_expanded --budget, max_seconds --seconds), then the
// device search (cuda::astar / cuda::gbfs) with the same budgets for each batch size B. Reported: status, expansions,
// search seconds (without the grounding: BestFirstResult::stats.seconds) and costs of both, the expansion overhead
// (device / CPU), the speedup of the search time, and whether the device plan replays on the CPU to a goal at its cost.

#include "mymyr/core/random.hpp"
#include "mymyr/cuda/astar.hpp"
#include "mymyr/cuda/gbfs.hpp"
#include "mymyr/cuda/heuristics.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <stdlib.h>  // getloadavg

using namespace mymyr;

namespace
{
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_heuristics_bench task.txt [--h max|add|ff] [--b 1000,10000,100000] [--variant auto|sweep|frontier]\n"
                 "       [--warp -1|0|1] [--threads T] [--global] [--reps R] [--sample S] [--cpu-threads T] [--walk-steps S]\n"
                 "       [--atoms auto|lazy|frozen] [--distinct N]\n",
                 msg);
    std::exit(2);
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

/// n random-walk states (walk k: SplitMix64(seed + k), a uniform length in [0, steps]).
std::vector<State> walk_states(const Task& task, u32 n, u32 steps, u64 seed)
{
    Successors& succ = task.workspace().successors();
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

/// The plan's cost when it replays from `start` to a goal state, else -1.
f64 replay(const Task& task, const State& start, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    const heuristics::ActionCosts costs(task);
    f64 g = costs.initial();
    State s = start;
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s.view(), a.label()))
            return -1;
        g += costs.unit() ? 1.0 : costs.cost(a.schema.v, a.binding.data());
        s = succ.apply(s.view(), a.label());
    }
    return succ.is_goal(s.view()) ? g : -1;
}

/// FNV-1a over the plan's schemas and bindings: plans are equal iff their fingerprints are (in practice).
u64 plan_fp(const std::vector<Action>& plan)
{
    u64 h = 1469598103934665603ull;
    const auto mix = [&](u64 x) {
        h ^= x;
        h *= 1099511628211ull;
    };
    for (const Action& a : plan)
    {
        mix(a.schema.v);
        mix(a.binding.size());
        for (const ObjectId o : a.binding)
            mix(o.v);
    }
    return h;
}

int run_search(const std::string& path, const TaskPtr& task, const std::string& mode, const std::string& hname,
               heuristics::Costs hcosts, const std::vector<u64>& batches, u64 budget, double seconds, bool single_bucket, int reps,
               const cuda::DeviceHeuristicOptions& launch)
{
    const bool greedy = mode == "gbfs";
    search::BestFirstOptions co;
    co.heuristic.kind = heuristics::parse_kind(hname);
    co.heuristic.costs = hcosts;
    co.control.budget.max_expanded = budget;
    co.control.budget.max_seconds = seconds;
    double load[3] = {0, 0, 0};
    if (getloadavg(load, 3) < 0)
        load[0] = load[1] = load[2] = -1;
    auto t0 = Clock::now();
    const search::BestFirstResult c = greedy ? search::gbfs_eager(*task, co) : search::astar_eager(*task, co);
    const double cpu_total = since(t0);
    cuda::ContextOptions cto;
    cto.max_bytes = u64{3} << 30;
    const auto ctx = cuda::DeviceContext::create(0, cto);
    const State s0 = task->initial_state();
    bool ok = true;
    {
        // warm-up: module loading, the first allocations of the context
        cuda::DeviceBestFirstOptions w;
        w.search = co;
        w.search.control.budget.max_expanded = std::min<u64>(budget, 2000);
        w.batch = 100;
        (void)(greedy ? cuda::gbfs(ctx, task, w) : cuda::astar(ctx, task, w));
    }
    for (u64 B : batches)
    {
        cuda::DeviceBestFirstOptions o;
        o.search = co;
        o.batch = static_cast<u32>(B);
        o.single_bucket = single_bucket;
        o.heuristic = launch;
        std::vector<double> times;
        cuda::DeviceBestFirstResult d;
        double total = 0;
        for (int r = 0; r < std::max(1, reps); ++r)
        {
            t0 = Clock::now();
            d = greedy ? cuda::gbfs(ctx, task, o) : cuda::astar(ctx, task, o);
            total = since(t0);
            times.push_back(d.result.stats.seconds);
        }
        const double dev = median(times);
        const bool solved = d.result.status == search::SearchStatus::Solved;
        const f64 cost = solved ? replay(*task, s0, d.result.plan) : 0;
        const bool valid = !solved || cost == d.result.cost || greedy;  // GBFS: the fork's plan extraction may differ
        ok = ok && valid && (greedy || !solved || c.status != search::SearchStatus::Solved || d.result.cost == c.cost);
        std::printf("{\"task\": \"%s\", \"mode\": \"%s\", \"h\": \"%s\", \"B\": %llu, \"single_bucket\": %s, "
                    "\"cpu_status\": \"%s\", \"cpu_expanded\": %llu, \"cpu_generated\": %llu, \"cpu_search_s\": %.4f, "
                    "\"cpu_total_s\": %.4f, \"cpu_cost\": %g, \"dev_status\": \"%s\", \"dev_expanded\": %llu, "
                    "\"dev_generated\": %llu, \"dev_search_s\": %.4f, \"dev_total_s\": %.4f, \"dev_cost\": %g, \"replay_cost\": %g, "
                    "\"steps\": %llu, \"stale\": %llu, \"max_batch\": %llu, \"overhead\": %.4f, \"speedup\": %.3f, "
                    "\"heuristic_ms\": %.1f, \"host_ms\": %.1f, \"device_bytes\": %llu, \"graph_steps\": %llu, \"host_steps\": %llu, "
                    "\"loops\": %llu, \"captures\": %llu, \"aborts\": %llu, \"popped\": %llu, \"open_entries\": %llu, \"plan_fp\": \"%016llx\", \"valid\": %s, \"load1\": %.2f}\n",
                    path.c_str(), mode.c_str(), hname.c_str(), static_cast<unsigned long long>(B), single_bucket ? "true" : "false",
                    search::to_string(c.status), static_cast<unsigned long long>(c.stats.expanded),
                    static_cast<unsigned long long>(c.stats.generated), c.stats.seconds, cpu_total, c.cost,
                    search::to_string(d.result.status), static_cast<unsigned long long>(d.result.stats.expanded),
                    static_cast<unsigned long long>(d.result.stats.generated), dev, total, d.result.cost, cost,
                    static_cast<unsigned long long>(d.device.steps), static_cast<unsigned long long>(d.device.stale),
                    static_cast<unsigned long long>(d.device.max_batch),
                    c.stats.expanded ? static_cast<double>(d.result.stats.expanded) / static_cast<double>(c.stats.expanded) : 0.0,
                    dev > 0 ? c.stats.seconds / dev : 0.0, d.device.heuristic_ms, d.device.host_ms,
                    static_cast<unsigned long long>(d.device.device_bytes), static_cast<unsigned long long>(d.device.graph_steps),
                    static_cast<unsigned long long>(d.device.host_steps), static_cast<unsigned long long>(d.device.loops),
                    static_cast<unsigned long long>(d.device.captures), static_cast<unsigned long long>(d.device.aborts),
                    static_cast<unsigned long long>(d.device.popped), static_cast<unsigned long long>(d.device.open_entries),
                    static_cast<unsigned long long>(plan_fp(d.result.plan)), valid ? "true" : "false", load[0]);
        std::fflush(stdout);
    }
    return ok ? 0 : 1;
}

std::vector<u64> parse_list(const std::string& s)
{
    std::vector<u64> out;
    std::stringstream in(s);
    std::string x;
    while (std::getline(in, x, ','))
        out.push_back(std::stoull(x));
    return out;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing task file");
    const std::string path = argv[1];
    TaskOptions to;
    std::string hname = "ff", variant = "auto", mode = "heur", problem, hcosts = "unit";
    u64 budget = ~u64{0};
    double seconds = std::numeric_limits<double>::infinity();
    bool single_bucket = false;
    std::vector<u64> batch_sizes{1000, 10000, 100000};
    u32 threads = 0, sample = 2000, cpu_threads = 0, walk_steps = 40, distinct = 20000;
    int warp = -1, reps = 5;
    bool global = false;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--h")
            hname = value();
        else if (a == "--mode")
            mode = value();
        else if (a == "--problem")
            problem = value();
        else if (a == "--budget")
            budget = std::stoull(value());
        else if (a == "--seconds")
            seconds = std::stod(value());
        else if (a == "--single-bucket")
            single_bucket = true;
        else if (a == "--hcosts")
            hcosts = value();
        else if (a == "--b")
            batch_sizes = parse_list(value());
        else if (a == "--variant")
            variant = value();
        else if (a == "--warp")
            warp = std::stoi(value());
        else if (a == "--threads")
            threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--global")
            global = true;
        else if (a == "--reps")
            reps = std::stoi(value());
        else if (a == "--sample")
            sample = static_cast<u32>(std::stoul(value()));
        else if (a == "--cpu-threads")
            cpu_threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--walk-steps")
            walk_steps = static_cast<u32>(std::stoul(value()));
        else if (a == "--distinct")
            distinct = static_cast<u32>(std::stoul(value()));
        else if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        TaskPtr task;
        if (problem.empty())
            task = Task::from_text_file(path, to);
        else
        {
#if defined(MYMYR_HAS_FRONTEND)
            task = Task::create(*frontend::load_task(path, problem), to);
#else
            usage("built without the PDDL front end: pass an exported task file");
#endif
        }
        if (mode == "astar" || mode == "gbfs")
        {
            cuda::DeviceHeuristicOptions launch;
            launch.variant = variant == "sweep" ? cuda::HeuristicVariant::Sweep
                             : variant == "frontier" ? cuda::HeuristicVariant::Frontier
                                                     : cuda::HeuristicVariant::Auto;
            launch.warp_groups = warp;
            launch.threads = threads;
            launch.force_global = global;
            return run_search(problem.empty() ? path : problem, task, mode, hname,
                              hcosts == "real" ? heuristics::Costs::Real : heuristics::Costs::Unit, batch_sizes, budget, seconds,
                              single_bucket, reps == 5 ? 1 : reps, launch);
        }
        if (mode != "heur")
            usage("--mode must be heur, astar or gbfs");
        const heuristics::Kind kind = heuristics::parse_kind(hname);
        const u32 cores = std::max(1u, std::thread::hardware_concurrency());
        const u32 T = cpu_threads ? cpu_threads : cores;
        double load[3] = {0, 0, 0};
        if (getloadavg(load, 3) < 0)
            load[0] = load[1] = load[2] = -1;

        u64 max_b = 1;
        for (u64 b : batch_sizes)
            max_b = std::max(max_b, b);
        distinct = static_cast<u32>(std::min<u64>(distinct, max_b));
        const std::vector<State> states = walk_states(*task, distinct, walk_steps, 4242);
        sample = std::min(sample, distinct);

        cuda::ContextOptions co;
        co.max_bytes = u64{3} << 30;
        const auto ctx = cuda::DeviceContext::create(0, co);
        cuda::DeviceHeuristicOptions ho;
        ho.kind = kind;
        ho.variant = variant == "sweep" ? cuda::HeuristicVariant::Sweep
                     : variant == "frontier" ? cuda::HeuristicVariant::Frontier
                                             : cuda::HeuristicVariant::Auto;
        ho.warp_groups = warp;
        ho.threads = threads;
        ho.force_global = global;
        auto t0 = Clock::now();
        cuda::DeviceHeuristic dh(ctx, task, ho);
        const double setup = since(t0);

        // rows: the distinct states tiled to max_b
        u32 W = std::max<u32>(1, task->words());
        for (const State& s : states)
            W = std::max(W, s.size_words());
        std::vector<u64> rows(max_b * W, 0);
        for (u64 i = 0; i < max_b; ++i)
        {
            const State& s = states[i % distinct];
            std::copy_n(s.data(), s.size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * W));
        }
        const cudaStream_t st = ctx->stream();
        cuda::DeviceBuffer d_rows(ctx, rows.size() * sizeof(u64), st);
        cuda::DeviceBuffer d_out(ctx, max_b * sizeof(u32), st);
        cuda::check(cudaMemcpyAsync(d_rows.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, st), "H2D");
        cuda::check(cudaStreamSynchronize(st), "sync");
        const auto* dr = static_cast<const u64*>(d_rows.data());
        auto* dout = static_cast<u32*>(d_out.data());

        // CPU: one core, then T threads
        auto cpu = heuristics::make_heuristic(*task, {.kind = kind, .costs = heuristics::Costs::Unit,
                                                      .evaluation = heuristics::Evaluation::Grounded,
                                                      .budget = {}, .relaxed = dh.grounding()});
        std::vector<f64> cpu_vals(sample);
        for (u32 i = 0; i < std::min<u32>(sample, 64); ++i)
            (void)cpu->evaluate(states[i].view());  // warm-up
        t0 = Clock::now();
        for (u32 i = 0; i < sample; ++i)
            cpu_vals[i] = cpu->evaluate(states[i].view());
        const double cpu1 = since(t0) / sample;
        std::vector<std::unique_ptr<heuristics::Heuristic>> hs;
        for (u32 k = 0; k < T; ++k)
            hs.push_back(heuristics::make_heuristic(*task, {.kind = kind, .costs = heuristics::Costs::Unit,
                                                            .evaluation = heuristics::Evaluation::Grounded,
                                                            .budget = {}, .relaxed = dh.grounding()}));
        std::atomic<u32> go{0};
        std::vector<std::thread> pool;
        t0 = Clock::now();
        for (u32 k = 0; k < T; ++k)
            pool.emplace_back(
                [&, k]
                {
                    go.fetch_add(1);
                    for (u32 i = 0; i < sample; ++i)
                        (void)hs[k]->evaluate(states[i].view());
                });
        for (std::thread& th : pool)
            th.join();
        const double cpuT = since(t0) / (static_cast<double>(sample) * T);

        // device values of the sample against the CPU / the reference
        bool equal = true;
        {
            dh.evaluate(dr, W, W, sample, dout);
            std::vector<u32> got(sample);
            cuda::check(cudaMemcpyAsync(got.data(), dout, sample * sizeof(u32), cudaMemcpyDeviceToHost, st), "D2H");
            cuda::check(cudaStreamSynchronize(st), "sync");
            u64 ff_equal_cpu = 0;
            for (u32 i = 0; i < sample; ++i)
            {
                const f64 want = kind == heuristics::Kind::FF ? dh.reference(states[i].view()) : cpu_vals[i];
                const u32 w = want == heuristics::k_dead_end ? cuda::DeviceHeuristic::k_dead_end : static_cast<u32>(want);
                equal = equal && got[i] == w;
                ff_equal_cpu += cpu_vals[i] == want ? 1 : 0;
            }
            if (kind == heuristics::Kind::FF)
                std::fprintf(stderr, "h_FF equal to the CPU's on %llu of %u states\n", static_cast<unsigned long long>(ff_equal_cpu), sample);
        }
        const cuda::DeviceHeuristicStats& S = dh.stats();
        for (u64 B : batch_sizes)
        {
            dh.evaluate(dr, W, W, B, dout);  // warm-up
            std::vector<double> times;
            for (int r = 0; r < reps; ++r)
            {
                t0 = Clock::now();
                dh.evaluate(dr, W, W, B, dout);
                times.push_back(since(t0));
            }
            const double dev = median(times) / static_cast<double>(B);
            std::printf("{\"task\": \"%s\", \"h\": \"%s\", \"B\": %llu, \"P\": %u, \"O\": %u, \"variant\": \"%s\", \"warp\": %s, "
                        "\"threads\": %u, \"blocks\": %u, \"shared\": %s, \"group_bytes\": %llu, \"levels\": %s, \"device_us\": %.4f, "
                        "\"cpu1_us\": %.4f, \"cpuT_us\": %.4f, \"cpu_threads\": %u, \"speedup_1core\": %.2f, \"speedup_all\": %.2f, "
                        "\"equal\": %s, \"sample\": %u, \"distinct\": %u, \"fallbacks\": %llu, \"setup_s\": %.3f, \"cores\": %u, "
                        "\"load1\": %.2f, \"load5\": %.2f}\n",
                        path.c_str(), hname.c_str(), static_cast<unsigned long long>(B), S.propositions, S.operators,
                        cuda::to_string(S.variant), S.warp_groups ? "true" : "false", S.threads, S.blocks, S.shared ? "true" : "false",
                        static_cast<unsigned long long>(S.group_bytes), S.supporter_levels ? "true" : "false", dev * 1e6, cpu1 * 1e6, cpuT * 1e6, T, cpu1 / dev, cpuT / dev,
                        equal ? "true" : "false", sample, distinct, static_cast<unsigned long long>(S.fallbacks), setup, cores,
                        load[0], load[1]);
            std::fflush(stdout);
        }
        return equal ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
