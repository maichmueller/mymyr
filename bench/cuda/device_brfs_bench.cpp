// mymyr_device_brfs_bench: the device BrFS on an exported task text file, printing one JSON line.
//
//   mymyr_device_brfs_bench task.txt [--atoms auto|lazy|frozen] [--match auto|fixed|fc] [--no-witness] [--no-canonical]
//                 [--chunk N] [--expected N] [--max-states N] [--stop-at-goal] [--fp] [--cpu T] [--reps R]
//                 [--max-bytes B] [--timings] [--candidate-bytes B]
//
// --cpu T also runs the CPU BrFS with T threads (deterministic ids) and reports its counts, time and fingerprint
// next to the device's; --reps R repeats the device search R times (fresh context each) and reports every time and
// the median. --timings records the per-phase device times (events per chunk: the host drives every chunk, no device
// loops). --candidate-bytes B bounds a chunk's candidate capacity (DeviceBrfsOptions::candidate_bytes; "cuts" counts
// the device loops' chunks cut to it).

#include "mymyr/cuda/brfs.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_brfs_bench task.txt [--atoms auto|lazy|frozen] [--match auto|fixed|fc] [--no-witness]\n"
                 "       [--no-canonical] [--chunk N] [--expected N] [--max-states N] [--stop-at-goal] [--fp] [--cpu T]\n"
                 "       [--reps R] [--max-bytes B] [--timings] [--candidate-bytes B]\n",
                 msg);
    std::exit(2);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing task file");
    TaskOptions to;
    cuda::DeviceBrfsOptions o;
    u32 cpu_threads = 0;
    int reps = 1;
    u64 max_bytes = u64{12} << 30;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else if (a == "--match")
        {
            const std::string v = value();
            to.matching = v == "fixed" ? TaskOptions::Matching::FixedOrder
                           : v == "fc" ? TaskOptions::Matching::ForwardChecking
                                       : TaskOptions::Matching::Auto;
        }
        else if (a == "--no-witness")
            o.witness_pruning = false;
        else if (a == "--no-canonical")
            o.canonical_order = false;
        else if (a == "--chunk")
            o.chunk_states = static_cast<u32>(std::stoul(value()));
        else if (a == "--expected")
            o.expected_states = std::stoull(value());
        else if (a == "--max-states")
            o.max_states = std::stoull(value());
        else if (a == "--stop-at-goal")
            o.stop_at_goal = true;
        else if (a == "--fp")
            o.fingerprint = true;
        else if (a == "--cpu")
            cpu_threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--reps")
            reps = std::stoi(value());
        else if (a == "--max-bytes")
            max_bytes = std::stoull(value());
        else if (a == "--timings")
            o.timings = true;
        else if (a == "--candidate-bytes")
            o.candidate_bytes = std::stoull(value());
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        const auto task = Task::from_text_file(argv[1], to);
        std::string name = argv[1];
        name = name.substr(name.find_last_of('/') + 1);
        cuda::ContextOptions co;
        co.max_bytes = max_bytes;
        std::vector<double> times;
        cuda::DeviceBrfsResult d;
        for (int k = 0; k < reps; ++k)
        {
            auto ctx = cuda::DeviceContext::create(0, co);
            d = cuda::brfs(ctx, task, o);
            times.push_back(d.result.search_s);
        }
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        std::string ts;
        for (double t : times)
            ts += (ts.empty() ? "" : ",") + std::to_string(t);
        const BrfsResult& r = d.result;
        const cuda::DeviceBrfsStats& st = d.stats;
        std::printf("{\"task\":\"%s\",\"atoms\":\"%s\",\"witness\":%s,\"canonical\":%s,\"chunk\":%u,\"expected\":%llu,"
                    "\"states\":%llu,\"expanded\":%llu,\"generated\":%llu,\"goal_states\":%llu,\"layers\":%u,\"exhausted\":%s,"
                    "\"solved\":%s,\"plan_len\":%zu,\"search_s\":%.4f,\"search_s_all\":[%s],\"states_per_s\":%.0f,"
                    "\"fp\":\"%016llx\",\"view_ms\":%.2f,\"gen_ms\":%.2f,\"dedup_ms\":%.2f,\"host_ms\":%.2f,\"chunks\":%llu,"
                    "\"groups\":%llu,\"resumed\":%llu,\"redone\":%llu,\"loops\":%llu,\"captures\":%llu,\"cuts\":%llu,"
                    "\"rehashes\":%u,\"uploads\":%u,\"widenings\":%u,\"host_schemas\":%u,\"table_slots\":%llu,"
                    "\"device_mb\":%.1f,\"words\":%u",
                    name.c_str(), task->atoms().mode() == AtomMode::Frozen ? "frozen" : "lazy", o.witness_pruning ? "true" : "false",
                    o.canonical_order ? "true" : "false", o.chunk_states, static_cast<unsigned long long>(o.expected_states),
                    static_cast<unsigned long long>(r.states), static_cast<unsigned long long>(r.expanded),
                    static_cast<unsigned long long>(r.generated), static_cast<unsigned long long>(r.goal_states), r.layers,
                    r.exhausted ? "true" : "false", r.solved ? "true" : "false", r.plan.size(), median, ts.c_str(),
                    static_cast<double>(r.states) / median, static_cast<unsigned long long>(r.fingerprint), st.view_ms, st.gen_ms,
                    st.dedup_ms, st.host_ms, static_cast<unsigned long long>(st.chunks), static_cast<unsigned long long>(st.groups),
                    static_cast<unsigned long long>(st.resumed), static_cast<unsigned long long>(st.redone),
                    static_cast<unsigned long long>(st.loops), static_cast<unsigned long long>(st.captures),
                    static_cast<unsigned long long>(st.cuts), st.rehashes, st.uploads,
                    st.widenings,
                    st.host_schemas, static_cast<unsigned long long>(st.table_slots), static_cast<double>(st.device_bytes) / 1048576.0,
                    r.words);
        if (cpu_threads)
        {
            BrfsOptions bo;
            bo.threads = cpu_threads;
            bo.witness_pruning = o.witness_pruning;
            bo.canonical_order = o.canonical_order;
            bo.max_states = o.max_states;
            bo.stop_at_goal = o.stop_at_goal;
            bo.fingerprint = o.fingerprint;
            const BrfsResult c = brfs(*task, bo);
            std::printf(",\"cpu_threads\":%u,\"cpu_states\":%llu,\"cpu_generated\":%llu,\"cpu_goal_states\":%llu,"
                        "\"cpu_layers\":%u,\"cpu_search_s\":%.4f,\"cpu_fp\":\"%016llx\",\"equal\":%s",
                        cpu_threads, static_cast<unsigned long long>(c.states), static_cast<unsigned long long>(c.generated),
                        static_cast<unsigned long long>(c.goal_states), c.layers, c.search_s,
                        static_cast<unsigned long long>(c.fingerprint),
                        c.states == r.states && c.generated == r.generated && c.goal_states == r.goal_states &&
                                c.layers == r.layers && c.fingerprint == r.fingerprint
                            ? "true"
                            : "false");
        }
        std::printf("}\n");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
