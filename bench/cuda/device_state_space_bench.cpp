// mymyr_device_state_space_bench (one JSON line per measurement).
//
//   mymyr_device_state_space_bench single [--tasks a,b] [--threads 1,64] [--reps 3] [--output device|host|both]
//   mymyr_device_state_space_bench table  [--instances 210] [--blocks 5,6,7] [--threads 64] [--reps 3] [--waves 0]
//                           [--output device,host] [--no-baselines]  (no pool or one-at-a-time device runs)
//
// single: the state space of lifted-suite tasks (tests/data/tasks, frozen slots) by the host
// datasets::generate_state_space at each thread count against cuda::state_space (device output: the arrays stay on
// the device; host output: datasets::StateSpace). The median of `reps` runs after one warm-up run of the
// device (the host build at one thread runs once).
// table: `instances` random blocksworld problems (the 4-operator domain of blocksworld-ipc; n blocks, n cycling over
// --blocks: 866, 7057 and 65990 states for 5, 6 and 7 blocks), one table: the host's instance pool
// (datasets::generate_state_spaces on `threads` workers) against cuda::state_spaces over the table (device and host
// output) and against one cuda::state_space per instance.
// Every line reports the load average before the measurement (the machine and the GPU are shared).

#include "../../tests/cpp/support/suite.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/state_space.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;
namespace fs = std::filesystem;

namespace
{
using Clock = std::chrono::steady_clock;

[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_state_space_bench single [--tasks a,b] [--threads 1,64] [--reps 3] [--output device|host|both]\n"
                 "       mymyr_device_state_space_bench table [--instances 210] [--blocks 5,6,7] [--threads 64] [--reps 3] [--waves 0] "
                 "[--output device,host] [--no-baselines]\n",
                 msg);
    std::exit(2);
}

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');)
        if (!x.empty())
            out.push_back(x);
    return out;
}

std::vector<u32> split_u32(const std::string& s)
{
    std::vector<u32> out;
    for (const std::string& x : split(s))
        out.push_back(static_cast<u32>(std::stoul(x)));
    return out;
}

std::string loadavg()
{
    std::ifstream f("/proc/loadavg");
    std::string a;
    f >> a;
    return a.empty() ? "?" : a;
}

double seconds(const std::function<void()>& f)
{
    const auto t0 = Clock::now();
    f();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

cuda::StateSpaceOutput output_of(const std::string& s)
{
    if (s == "device")
        return cuda::StateSpaceOutput::Device;
    if (s == "host")
        return cuda::StateSpaceOutput::Host;
    if (s == "both")
        return cuda::StateSpaceOutput::Both;
    usage("--output is device, host or both");
}

cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{5} << 30;  // gate: at most 6 GB of GPU memory
    return cuda::DeviceContext::create(0, o);
}

// ------------------------------------------------------------------------------------------------ single
void single(const std::vector<std::string>& names, const std::vector<u32>& threads, u32 reps, const std::string& out)
{
    const auto ctx = context();
    for (const std::string& name : names)
    {
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        const TaskPtr task = Task::from_text_file(test::task_path(name), to);
        datasets::StateSpaceOptions so;
        so.remove_if_unsolvable = false;
        for (const u32 T : threads)
        {
            so.threads = T;
            const std::string load = loadavg();
            std::vector<double> ts;
            u64 states = 0, transitions = 0;
            for (u32 r = 0; r < (T == 1 ? 1 : reps); ++r)
                ts.push_back(seconds(
                    [&]
                    {
                        const datasets::StateSpaceResult res = datasets::generate_state_space(task, so);
                        states = res.space ? res.space->num_states() : res.states;
                        transitions = res.space ? res.space->num_transitions() : 0;
                    }));
            std::printf("{\"bench\":\"single\",\"task\":\"%s\",\"impl\":\"m7\",\"threads\":%u,\"states\":%llu,\"transitions\":%llu,"
                        "\"s\":%.4f,\"load\":\"%s\"}\n",
                        name.c_str(), T, static_cast<unsigned long long>(states), static_cast<unsigned long long>(transitions),
                        median(ts), load.c_str());
            std::fflush(stdout);
        }
        for (const std::string& o : split(out))
        {
            cuda::DeviceStateSpaceOptions d;
            d.space = so;
            d.space.threads = 0;
            d.output = output_of(o);
            (void)cuda::state_space(ctx, task, d);  // warm-up
            const std::string load = loadavg();
            std::vector<double> ts;
            cuda::DeviceStateSpaceStats st;
            u64 states = 0, transitions = 0;
            for (u32 r = 0; r < reps; ++r)
                ts.push_back(seconds(
                    [&]
                    {
                        const cuda::DeviceStateSpaceResult res = cuda::state_space(ctx, task, d, &st);
                        states = res.states;
                        transitions = res.space ? res.space->num_transitions() : res.host ? res.host->num_transitions() : 0;
                    }));
            std::printf("{\"bench\":\"single\",\"task\":\"%s\",\"impl\":\"device\",\"output\":\"%s\",\"states\":%llu,"
                        "\"transitions\":%llu,\"s\":%.4f,\"generate_ms\":%.1f,\"post_ms\":%.1f,\"output_ms\":%.1f,\"chunks\":%llu,"
                        "\"device_mb\":%.0f,\"load\":\"%s\"}\n",
                        name.c_str(), o.c_str(), static_cast<unsigned long long>(states), static_cast<unsigned long long>(transitions),
                        median(ts), st.generate_ms, st.post_ms, st.output_ms, static_cast<unsigned long long>(st.chunks),
                        static_cast<double>(st.device_bytes) / (1 << 20), load.c_str());
            std::fflush(stdout);
        }
    }
}

// ------------------------------------------------------------------------------------------------ table
u64 splitmix(u64& x)
{
    u64 z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Goal towers of n blocks: `on` atoms of a random permutation cut at random places.
std::string goal_towers(u32 n, u64& rng)
{
    std::vector<u32> p(n);
    for (u32 i = 0; i < n; ++i)
        p[i] = i;
    for (u32 i = n; i > 1; --i)
        std::swap(p[i - 1], p[splitmix(rng) % i]);
    std::string s;
    for (u32 i = 1; i < n; ++i)
        if (splitmix(rng) % 3 != 0)
            s += " (on b" + std::to_string(p[i] + 1) + " b" + std::to_string(p[i - 1] + 1) + ")";
    return s;
}

std::string blocks_problem(u32 n, u32 k, u64& rng)
{
    // initial towers with (clear) on the tops; goal towers without the table atoms
    std::vector<u32> p(n);
    for (u32 i = 0; i < n; ++i)
        p[i] = i;
    for (u32 i = n; i > 1; --i)
        std::swap(p[i - 1], p[splitmix(rng) % i]);
    std::vector<bool> starts(n, false);
    starts[0] = true;
    for (u32 i = 1; i < n; ++i)
        starts[i] = splitmix(rng) % 3 == 0;
    std::string init = "(arm-empty)";
    for (u32 i = 0; i < n; ++i)
    {
        const std::string b = "b" + std::to_string(p[i] + 1);
        init += starts[i] ? " (on-table " + b + ")" : " (on " + b + " b" + std::to_string(p[i - 1] + 1) + ")";
        if (i + 1 == n || starts[i + 1])
            init += " (clear " + b + ")";
    }
    std::string objects;
    for (u32 i = 0; i < n; ++i)
        objects += " b" + std::to_string(i + 1);
    const std::string goal = goal_towers(n, rng);
    return "(define (problem bw-" + std::to_string(n) + "-" + std::to_string(k) + ") (:domain blocksworld)\n (:objects" + objects +
           " - object)\n (:init " + init + ")\n (:goal (and" + (goal.empty() ? " (arm-empty)" : goal) + ")))\n";
}

void table(u32 instances, const std::vector<u32>& blocks, u32 threads, u32 reps, u64 waves, const std::string& outs,
           bool baselines)
{
    const fs::path dom = fs::path(MYMYR_FORK_DATA_DIR) / "ipc" / "blocksworld-ipc" / "train" / "domain.pddl";
    const auto domain = frontend::Domain::from_file(dom);
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Frozen;
    std::vector<TaskPtr> tasks;
    u64 rng = 0x5eed;
    for (u32 k = 0; k < instances; ++k)
    {
        const u32 n = blocks[k % blocks.size()];
        tasks.push_back(Task::create(*domain->instantiate_string(blocks_problem(n, k, rng), "p.pddl"), to));
    }
    const rl::TaskTablePtr tab = rl::TaskTable::create(tasks);
    datasets::StateSpaceOptions so;
    so.remove_if_unsolvable = false;
    so.threads = 1;
    // the host's instance pool
    u64 states = 0, transitions = 0;
    if (baselines)
    {
        const std::string load = loadavg();
        std::vector<double> ts, runs, frees;
        for (u32 r = 0; r < reps; ++r)
        {
            std::optional<std::vector<datasets::StateSpaceResult>> rs;
            runs.push_back(seconds([&] { rs.emplace(datasets::generate_state_spaces(tasks, so, threads)); }));
            states = transitions = 0;
            for (const auto& x : *rs)
            {
                states += x.space ? x.space->num_states() : 0;
                transitions += x.space ? x.space->num_transitions() : 0;
            }
            frees.push_back(seconds([&] { rs.reset(); }));
            ts.push_back(runs.back() + frees.back());
        }
        std::printf("{\"bench\":\"table\",\"impl\":\"cpu_pool\",\"instances\":%u,\"threads\":%u,\"states\":%llu,\"transitions\":%llu,"
                    "\"s\":%.4f,\"run_ms\":%.1f,\"free_ms\":%.1f,\"load\":\"%s\"}\n",
                    instances, threads, static_cast<unsigned long long>(states), static_cast<unsigned long long>(transitions),
                    median(ts), median(runs) * 1e3, median(frees) * 1e3, load.c_str());
        std::fflush(stdout);
    }
    const auto ctx = context();
    for (const std::string& oname_s : split(outs))
    {
        const cuda::StateSpaceOutput out = output_of(oname_s);
        const char* oname = oname_s.c_str();
        cuda::DeviceStateSpaceOptions d;
        d.space = so;
        d.space.threads = 0;
        d.output = out;
        if (waves)
            d.wave_states = waves;
        (void)cuda::state_spaces(ctx, tab, d);  // warm-up
        std::vector<double> ts, runs, frees;
        cuda::DeviceStateSpaceStats st;
        const std::string load = loadavg();
        for (u32 r = 0; r < reps; ++r)
        {
            // (the results' release counts: host output frees large host arrays)
            std::optional<cuda::DeviceStateSpaces> rs;
            runs.push_back(seconds([&] { rs.emplace(cuda::state_spaces(ctx, tab, d)); }));
            st = rs->stats;
            frees.push_back(seconds([&] { rs.reset(); }));
            ts.push_back(runs.back() + frees.back());
        }
        std::printf("{\"bench\":\"table\",\"impl\":\"device_table\",\"output\":\"%s\",\"instances\":%u,\"states\":%llu,"
                    "\"transitions\":%llu,\"s\":%.4f,\"run_ms\":%.1f,\"free_ms\":%.1f,\"waves\":%u,\"multi\":%d,\"generate_ms\":%.1f,"
                    "\"post_ms\":%.1f,\"output_ms\":%.1f,\"chunks\":%llu,\"device_mb\":%.0f,\"load\":\"%s\"}\n",
                    oname, instances, static_cast<unsigned long long>(st.states), static_cast<unsigned long long>(st.transitions),
                    median(ts), median(runs) * 1e3, median(frees) * 1e3, st.waves, st.multi ? 1 : 0, st.generate_ms, st.post_ms, st.output_ms,
                    static_cast<unsigned long long>(st.chunks), static_cast<double>(st.device_bytes) / (1 << 20), load.c_str());
        std::fflush(stdout);
        if (!baselines)
            continue;
        // one device run per instance
        std::vector<double> ts1;
        const std::string load1 = loadavg();
        for (u32 r = 0; r < reps; ++r)
            ts1.push_back(seconds(
                [&]
                {
                    for (const TaskPtr& t : tasks)
                        (void)cuda::state_space(ctx, t, d);
                }));
        std::printf("{\"bench\":\"table\",\"impl\":\"device_singles\",\"output\":\"%s\",\"instances\":%u,\"s\":%.4f,\"load\":\"%s\"}\n",
                    oname, instances, median(ts1), load1.c_str());
        std::fflush(stdout);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("no mode");
    const std::string mode = argv[1];
    std::vector<std::string> tasks = {"zenotravel__p05", "transport-opt08-strips__p23", "snake-opt18-strips__p05"};
    std::vector<u32> threads = {1, std::max<u32>(1, std::thread::hardware_concurrency())};
    std::vector<u32> blocks = {5, 6, 7};
    u32 reps = 3, instances = 210;
    u64 waves = 0;
    std::string out = "device,host";
    bool threads_set = false, baselines = true;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        if (a == "--tasks")
            tasks = split(next());
        else if (a == "--threads")
        {
            threads = split_u32(next());
            threads_set = true;
        }
        else if (a == "--reps")
            reps = static_cast<u32>(std::stoul(next()));
        else if (a == "--output")
            out = next();
        else if (a == "--instances")
            instances = static_cast<u32>(std::stoul(next()));
        else if (a == "--blocks")
            blocks = split_u32(next());
        else if (a == "--waves")
            waves = std::stoull(next());
        else if (a == "--no-baselines")
            baselines = false;
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        if (mode == "single")
            single(tasks, threads, reps, out);
        else if (mode == "table")
            table(instances, blocks, threads_set ? threads.back() : std::max<u32>(1, std::thread::hardware_concurrency()), reps, waves,
                  out, baselines);
        else
            usage("unknown mode");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
