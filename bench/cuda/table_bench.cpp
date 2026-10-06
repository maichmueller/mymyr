// mymyr_table_bench: table measurements (one JSON line per measurement).
//
//   mymyr_table_bench env    [--sets a,b,..] [--envs 16384,65536] [--steps 200] [--warmup 10] [--reps 3] [--graph K]
//   mymyr_table_bench expand [--sets a,b,..] [--envs 16384,65536] [--reps 5] [--walk 20]
//   mymyr_table_bench views  [--envs 65536]
//   mymyr_table_bench table  [--instances 100,1000] [--reps 3]
//
// env: random-policy rollouts of cuda::DeviceEnv (fast path where the table allows it) over the mixed table of a set
// (tests/cpp/rl/table_instance_sets.hpp, plus "sokoban": sokoban-ipc instances of 54 to 177 objects), env i in instance i mod I
// (equal env weights), under both bucket launches; then each instance alone (its table of one) at the same N. The gate
// compares the mixed time per step with the env-weighted mean of the singles' (ratio = t_mixed / sum_k w_k t_k).
// A rep: a fresh batch reset to the initial states, `warmup` steps, then `steps` timed steps between two stream
// synchronizations (as mymyr_device_env_rollout_bench); the median rep is reported. --graph K: the timed steps are replays of a CUDA
// graph of K steps (stream capture of the env's stream; the launch groups' streams fork and join inside it), for
// the singles and the mixed batch alike.
// expand: cuda::DeviceExpander (count + write: successors, parents, labels, offsets) of N states, reached by `walk`
// random steps from the initial states (mixed: env i in instance i mod I; singles: every row of the one instance), the
// median of `reps` expansions; the same ratio.
// views: the fast path's per-environment cache at N envs for sokoban-ipc tables of small instances with one larger
// one (the views are padded to the widest instance's V).
// table: TaskTable::create, DeviceTaskTable::upload and DeviceEnv construction for tables of 100 / 1000 distinct small
// instances (blocksworld-ipc train, each problem instantiated again as needed).

#include "../../tests/cpp/rl/table_instance_sets.hpp"
#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace mymyr::test
{
std::filesystem::path fork_data_dir()
{
    if (const char* d = std::getenv("MYMYR_FORK_DATA"); d && *d)
        return d;
    return std::filesystem::path(MYMYR_FORK_DATA_DIR);
}
}  // namespace mymyr::test

using namespace mymyr;
using namespace mymyr::test;
namespace fs = std::filesystem;

namespace
{
[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_table_bench env|expand|views|table [--sets a,b] [--envs N,N] [--steps S]\n"
                 "       [--warmup W] [--reps R] [--walk K] [--instances I,I] [--graph K]\n",
                 msg);
    std::exit(2);
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/// FNV-1a over device bytes, chained (`fp` in the JSON lines: equal outputs give equal fingerprints across builds).
constexpr u64 k_fnv0 = 14695981039346656037ull;
u64 fnv(u64 h, const void* d, u64 bytes, cudaStream_t s)
{
    std::vector<unsigned char> b(bytes);
    if (bytes)
    {
        cuda::check(cudaMemcpyAsync(b.data(), d, bytes, cudaMemcpyDeviceToHost, s), "D2H");
        cuda::check(cudaStreamSynchronize(s), "sync");
    }
    for (const unsigned char c : b)
    {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<u64> parse_list(const std::string& s)
{
    std::vector<u64> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        out.push_back(std::stoull(x));
    return out;
}

std::vector<std::string> parse_names(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        out.push_back(x);
    return out;
}

fs::path sokoban_dir() { return fork_data_dir() / "ipc" / "sokoban-ipc"; }

/// The bench's sets: table_instance_sets() plus "sokoban" (sokoban-ipc: 54, 87, 106, 151, 177 objects; 1 to 9 words).
std::vector<InstanceSet> bench_sets()
{
    std::vector<InstanceSet> out = table_instance_sets();
    const fs::path d = sokoban_dir();
    out.push_back({"sokoban",
                   d / "train" / "domain.pddl",
                   {d / "train" / "p01.pddl", d / "train" / "p36.pddl", d / "train" / "p56.pddl", d / "test" / "p22-easy.pddl",
                    d / "test" / "p29-easy.pddl"}});
    return out;
}

cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{3} << 30;  // the GPU may be shared: cap this process at 4 GB
    return cuda::DeviceContext::create(0, o);
}

std::string g_only;  // --only mixed|singles (profiling)
int g_ids = -1;      // --ids k: every env of the mixed batch in instance k (profiling)
bool g_blocked = false;  // --layout blocked: env i in instance i * I / N (default interleaved: i mod I)
int g_graph = 0;         // --graph K: replays of a captured graph of K steps

const char* launch_name(cuda::BucketLaunch l) { return l == cuda::BucketLaunch::Widest ? "widest" : "per_bucket"; }

/// A device batch of N environments over a table (env i in ids[i]).
struct Batch
{
    cuda::DeviceBuffer states, ids, steps, draws, counts, views, reward, term, trunc, count;
    rl::EnvBatch b;
    rl::StepOutputs out;

    Batch(const cuda::ContextPtr& ctx, const cuda::DeviceEnv& env, const std::vector<i32>& h_ids, cudaStream_t s) :
        states(ctx, h_ids.size() * env.words() * 8, s),
        ids(ctx, h_ids.size() * 4, s),
        steps(ctx, h_ids.size() * 4, s),
        draws(ctx, h_ids.size() * 8, s),
        counts(ctx, std::max<u64>(h_ids.size() * env.cache_schemas(), 1) * 4, s),
        views(ctx, std::max<u64>(h_ids.size() * env.cache_view_words(), 1) * 8, s),
        reward(ctx, h_ids.size() * 4, s),
        term(ctx, h_ids.size(), s),
        trunc(ctx, h_ids.size(), s),
        count(ctx, h_ids.size() * 4, s)
    {
        const u64 N = h_ids.size();
        cuda::check(cudaMemcpyAsync(ids.data(), h_ids.data(), N * 4, cudaMemcpyHostToDevice, s), "H2D");
        cuda::check(cudaMemsetAsync(draws.data(), 0, N * 8, s), "cudaMemsetAsync");
        b.states = static_cast<u64*>(states.data());
        b.rows = N;
        b.words = env.words();
        b.task_ids = static_cast<i32*>(ids.data());
        b.steps = static_cast<i32*>(steps.data());
        b.draws = static_cast<u64*>(draws.data());
        b.counts = env.cache_schemas() ? static_cast<u32*>(counts.data()) : nullptr;
        b.views = env.cache_view_words() ? static_cast<u64*>(views.data()) : nullptr;
        out.reward = static_cast<f32*>(reward.data());
        out.terminated = static_cast<u8*>(term.data());
        out.truncated = static_cast<u8*>(trunc.data());
        out.count = static_cast<i32*>(count.data());
    }

    /// The batch's state and the last step's outputs.
    [[nodiscard]] u64 fingerprint(cudaStream_t s) const
    {
        const u64 N = b.rows;
        u64 h = fnv(k_fnv0, states.data(), N * b.words * 8, s);
        h = fnv(h, ids.data(), N * 4, s);
        h = fnv(h, steps.data(), N * 4, s);
        h = fnv(h, draws.data(), N * 8, s);
        h = fnv(h, reward.data(), N * 4, s);
        h = fnv(h, term.data(), N, s);
        h = fnv(h, trunc.data(), N, s);
        return fnv(h, count.data(), N * 4, s);
    }
};

struct Loaded
{
    std::string name;
    std::vector<TaskPtr> tasks;
    rl::TaskTablePtr table;
};

std::vector<Loaded> load(const std::vector<std::string>& names)
{
    std::vector<Loaded> out;
    for (const InstanceSet& s : bench_sets())
    {
        if (!names.empty() && std::find(names.begin(), names.end(), s.name) == names.end())
            continue;
        std::vector<TaskPtr> tasks = load_set(s, TaskOptions::Atoms::Frozen);
        if (tasks.empty())
            throw std::runtime_error("missing PDDL of set " + s.name);
        auto table = rl::TaskTable::create(tasks);
        out.push_back({s.name, std::move(tasks), std::move(table)});
    }
    return out;
}

std::string objects_of(const rl::TaskTable& t)
{
    std::string s;
    for (const auto& in : t.instances())
        s += (s.empty() ? "" : ",") + std::to_string(in.num_objects);
    return s;
}

std::string words_of(const rl::TaskTable& t)
{
    std::string s;
    for (const auto& in : t.instances())
        s += (s.empty() ? "" : ",") + std::to_string(in.words);
    return s;
}

/// Seconds per step (median of reps) of N random-policy envs over `table`.
double time_env(const cuda::ContextPtr& ctx, const rl::TaskTablePtr& table, const std::vector<i32>& ids, int steps,
                int warmup, int reps, cuda::BucketLaunch launch, bool* fast, u64* V, u32* S, std::vector<double>* all,
                u64* fp)
{
    rl::EnvConfig cfg;
    cfg.seed = 1;
    cuda::DeviceEnv env(ctx, table, cfg, cuda::DeviceEnv::Path::Auto);
    env.set_launch(launch);
    const cudaStream_t s = env.stream();
    *fast = env.fast();
    *V = env.cache_view_words();
    *S = env.cache_schemas();
    std::vector<double> secs;
    for (int r = 0; r < reps; ++r)
    {
        Batch B(ctx, env, ids, s);
        env.reset(B.b);
        for (int t = 0; t < warmup; ++t)
            env.step(B.b, B.out);
        cudaGraph_t g = nullptr;
        cudaGraphExec_t gx = nullptr;
        if (g_graph > 0)
        {
            cuda::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
            for (int t = 0; t < g_graph; ++t)
                env.step(B.b, B.out);
            cuda::check(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
            cuda::check(cudaGraphInstantiate(&gx, g, 0), "cudaGraphInstantiate");
            cuda::check(cudaGraphLaunch(gx, s), "cudaGraphLaunch");  // uploads the graph
        }
        const int timed = g_graph > 0 ? steps / g_graph * g_graph : steps;
        cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        const double t0 = now_s();
        if (g_graph > 0)
            for (int t = 0; t < steps / g_graph; ++t)
                cuda::check(cudaGraphLaunch(gx, s), "cudaGraphLaunch");
        else
            for (int t = 0; t < steps; ++t)
                env.step(B.b, B.out);
        cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        secs.push_back((now_s() - t0) / timed);
        if (gx)
        {
            cuda::check(cudaGraphExecDestroy(gx), "cudaGraphExecDestroy");
            cuda::check(cudaGraphDestroy(g), "cudaGraphDestroy");
        }
        env.check_errors();
        *fp = B.fingerprint(s);
    }
    *all = secs;
    return median(secs);
}

std::string join(const std::vector<double>& v)
{
    std::string s;
    for (double x : v)
    {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.4g", x);
        s += (s.empty() ? "" : ",") + std::string(buf);
    }
    return s;
}

void run_env(const std::vector<std::string>& names, const std::vector<u64>& envs, int steps, int warmup, int reps)
{
    auto ctx = context();
    for (const Loaded& L : load(names))
    {
        const u32 I = L.table->size();
        for (u64 N : envs)
        {
            std::vector<i32> ids(N);
            for (u64 i = 0; i < N; ++i)
                ids[i] = g_ids >= 0 ? g_ids : g_blocked ? static_cast<i32>(i * I / N) : static_cast<i32>(i % I);
            // singles: instance k alone at N envs
            std::vector<double> single(I);
            double ref = 0;
            std::string singles_json;
            for (u32 k = 0; k < I && g_only != "mixed"; ++k)
            {
                bool fast = false;
                u64 V = 0;
                u32 S = 0;
                std::vector<double> all;
                u64 fp = 0;
                single[k] = time_env(ctx, rl::TaskTable::single(L.tasks[k]), std::vector<i32>(N, 0), steps, warmup,
                                     reps, cuda::BucketLaunch::Widest, &fast, &V, &S, &all, &fp);
                const auto n_k = static_cast<u64>(std::count(ids.begin(), ids.end(), static_cast<i32>(k)));
                ref += static_cast<double>(n_k) / static_cast<double>(N) * single[k];
                char buf[256];
                std::snprintf(buf, sizeof buf, "%s{\"k\":%u,\"objects\":%u,\"W\":%u,\"V\":%llu,\"S\":%u,\"path\":\"%s\","
                              "\"us_per_step\":%.1f,\"env_steps_per_s\":%.4g,\"fp\":\"%016llx\"}",
                              k ? "," : "", k, L.table->instance(k).num_objects, L.table->instance(k).words,
                              static_cast<unsigned long long>(V), S, fast ? "fast" : "general", 1e6 * single[k],
                              static_cast<double>(N) / single[k], static_cast<unsigned long long>(fp));
                singles_json += buf;
            }
            for (cuda::BucketLaunch launch : {cuda::BucketLaunch::Widest, cuda::BucketLaunch::PerBucket})
            {
                if (g_only == "singles")
                    break;
                bool fast = false;
                u64 V = 0;
                u32 S = 0;
                std::vector<double> all;
                u64 fp = 0;
                const double t = time_env(ctx, L.table, ids, steps, warmup, reps, launch, &fast, &V, &S, &all, &fp);
                std::printf("{\"tool\":\"table_bench\",\"mode\":\"env\",\"layout\":\"%s\",\"set\":\"%s\",\"N\":%llu,\"instances\":%u,"
                            "\"objects\":[%s],\"words\":[%s],\"W\":%u,\"launch\":\"%s\",\"path\":\"%s\",\"V\":%llu,\"S\":%u,"
                            "\"steps\":%d,\"us_per_step\":%.1f,\"us_all\":[%s],\"env_steps_per_s\":%.4g,"
                            "\"weighted_single_us\":%.1f,\"weighted_single_env_steps_per_s\":%.4g,\"ratio\":%.3f,"
                            "\"graph\":%d,\"fp\":\"%016llx\",\"singles\":[%s]}\n",
                            g_blocked ? "blocked" : "interleaved", L.name.c_str(), static_cast<unsigned long long>(N), I,
                            objects_of(*L.table).c_str(),
                            words_of(*L.table).c_str(), L.table->words(), launch_name(launch), fast ? "fast" : "general",
                            static_cast<unsigned long long>(V), S, steps, 1e6 * t, join([&] {
                                std::vector<double> u;
                                for (double x : all)
                                    u.push_back(1e6 * x);
                                return u;
                            }()).c_str(),
                            static_cast<double>(N) / t, 1e6 * ref, static_cast<double>(N) / ref, t / ref, g_graph,
                            static_cast<unsigned long long>(fp), singles_json.c_str());
                std::fflush(stdout);
            }
        }
    }
}

/// N states over `table` (env i of ids[i]) after `walk` random steps; returns them on the device.
cuda::DeviceBuffer walked_states(const cuda::ContextPtr& ctx, const rl::TaskTablePtr& table, const std::vector<i32>& ids,
                                 int walk, cudaStream_t s)
{
    rl::EnvConfig cfg;
    cfg.seed = 7;
    cuda::DeviceEnv env(ctx, table, cfg, cuda::DeviceEnv::Path::Auto, s);
    Batch B(ctx, env, ids, s);
    env.reset(B.b);
    for (int t = 0; t < walk; ++t)
        env.step(B.b, B.out);
    env.check_errors();
    cuda::DeviceBuffer out(ctx, ids.size() * env.words() * 8, s);
    cuda::check(cudaMemcpyAsync(out.data(), B.states.data(), ids.size() * env.words() * 8, cudaMemcpyDeviceToDevice, s),
                "D2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return out;
}

struct ExpandTime
{
    double secs = 0;
    u64 total = 0;
    int mode = 0;
    u64 fp = 0;  // the outputs and words_needed
};

ExpandTime time_expand(const cuda::ContextPtr& ctx, const rl::TaskTablePtr& table, const std::vector<i32>& ids, int walk,
                       int reps, cuda::BucketLaunch launch)
{
    cuda::Stream stream;
    const cudaStream_t s = stream.get();
    const u64 N = ids.size();
    const u32 W = table->words(), L = std::max<u32>(table->label_width(), 1);
    cuda::DeviceBuffer states = walked_states(ctx, table, ids, walk, s);
    cuda::DeviceBuffer d_ids(ctx, N * 4, s);
    cuda::check(cudaMemcpyAsync(d_ids.data(), ids.data(), N * 4, cudaMemcpyHostToDevice, s), "H2D");
    cuda::DeviceExpander ex(ctx, table, s);
    ex.set_launch(launch);
    const rl::StateBatchView in{static_cast<const u64*>(states.data()), N, W, 0};
    const i32* tid = table->size() > 1 ? static_cast<const i32*>(d_ids.data()) : nullptr;
    const u64 total = ex.count(in, tid);
    const u64 cap = std::max<u64>(total, 1);
    cuda::DeviceBuffer succ(ctx, cap * W * 8, s), parent(ctx, cap * 4, s), schema(ctx, cap * 4, s),
        binding(ctx, cap * L * 4, s), offsets(ctx, (N + 1) * 4, s);
    rl::Expansion x;
    x.capacity = cap;
    x.words = W;
    x.label_width = L;
    x.succ = static_cast<u64*>(succ.data());
    x.parent = static_cast<i32*>(parent.data());
    x.schema = static_cast<i32*>(schema.data());
    x.binding = static_cast<i32*>(binding.data());
    x.offsets = static_cast<i32*>(offsets.data());
    ex.write(x);  // warm up
    cuda::check(cudaStreamSynchronize(s), "sync");
    std::vector<double> secs;
    for (int r = 0; r < reps; ++r)
    {
        const double t0 = now_s();
        ex.expand(in, tid, x);
        cuda::check(cudaStreamSynchronize(s), "sync");
        secs.push_back(now_s() - t0);
        if (x.total != total)
            throw std::logic_error("expand totals differ between reps");
    }
    u64 h = fnv(k_fnv0, succ.data(), total * W * 8, s);
    h = fnv(h, parent.data(), total * 4, s);
    h = fnv(h, schema.data(), total * 4, s);
    h = fnv(h, binding.data(), total * L * 4, s);
    h = fnv(h, offsets.data(), (N + 1) * 4, s);
    h = (h ^ x.words_needed) * 1099511628211ull;
    return {median(secs), total, static_cast<int>(ex.mode()), h};
}

void run_expand(const std::vector<std::string>& names, const std::vector<u64>& envs, int reps, int walk)
{
    auto ctx = context();
    for (const Loaded& L : load(names))
    {
        const u32 I = L.table->size();
        for (u64 N : envs)
        {
            std::vector<i32> ids(N);
            for (u64 i = 0; i < N; ++i)
                ids[i] = g_ids >= 0 ? g_ids : g_blocked ? static_cast<i32>(i * I / N) : static_cast<i32>(i % I);
            double ref = 0;
            std::string singles_json;
            for (u32 k = 0; k < I && g_only != "mixed"; ++k)
            {
                const ExpandTime e = time_expand(ctx, rl::TaskTable::single(L.tasks[k]), std::vector<i32>(N, 0), walk,
                                                 reps, cuda::BucketLaunch::Widest);
                const auto n_k = static_cast<u64>(std::count(ids.begin(), ids.end(), static_cast<i32>(k)));
                ref += static_cast<double>(n_k) / static_cast<double>(N) * e.secs;
                char buf[256];
                std::snprintf(buf, sizeof buf, "%s{\"k\":%u,\"objects\":%u,\"W\":%u,\"ms\":%.3f,\"successors\":%llu,"
                              "\"states_per_s\":%.4g,\"fp\":\"%016llx\"}",
                              k ? "," : "", k, L.table->instance(k).num_objects, L.table->instance(k).words, 1e3 * e.secs,
                              static_cast<unsigned long long>(e.total), static_cast<double>(N) / e.secs,
                              static_cast<unsigned long long>(e.fp));
                singles_json += buf;
            }
            for (cuda::BucketLaunch launch : {cuda::BucketLaunch::Widest, cuda::BucketLaunch::PerBucket})
            {
                if (g_only == "singles")
                    break;
                const ExpandTime e = time_expand(ctx, L.table, ids, walk, reps, launch);
                std::printf("{\"tool\":\"table_bench\",\"mode\":\"expand\",\"layout\":\"%s\",\"set\":\"%s\",\"N\":%llu,\"instances\":%u,"
                            "\"objects\":[%s],\"words\":[%s],\"W\":%u,\"launch\":\"%s\",\"expander_mode\":%d,\"walk\":%d,"
                            "\"ms\":%.3f,\"successors\":%llu,\"states_per_s\":%.4g,\"successors_per_s\":%.4g,"
                            "\"weighted_single_ms\":%.3f,\"ratio\":%.3f,\"fp\":\"%016llx\",\"singles\":[%s]}\n",
                            g_blocked ? "blocked" : "interleaved", L.name.c_str(), static_cast<unsigned long long>(N), I,
                            objects_of(*L.table).c_str(),
                            words_of(*L.table).c_str(), L.table->words(), launch_name(launch), e.mode, walk,
                            1e3 * e.secs, static_cast<unsigned long long>(e.total), static_cast<double>(N) / e.secs,
                            static_cast<double>(e.total) / e.secs, 1e3 * ref, e.secs / ref,
                            static_cast<unsigned long long>(e.fp), singles_json.c_str());
                std::fflush(stdout);
            }
        }
    }
}

TaskPtr sokoban(const std::string& split, const std::string& problem)
{
    const fs::path d = sokoban_dir();
    const auto dom = frontend::Domain::from_file(d / split / "domain.pddl");
    TaskOptions o;
    o.atoms = TaskOptions::Atoms::Frozen;
    return Task::create(*dom->instantiate_file(d / split / problem), o);
}

void run_views(const std::vector<u64>& envs)
{
    auto ctx = context();
    const std::vector<TaskPtr> small{sokoban("train", "p01.pddl"), sokoban("train", "p06.pddl"),
                                     sokoban("train", "p11.pddl"), sokoban("train", "p16.pddl")};
    const std::vector<std::pair<std::string, std::string>> large{
        {"train", "p66.pddl"}, {"test", "p22-easy.pddl"}, {"test", "p29-easy.pddl"}, {"test", "p05-medium.pddl"}};
    rl::EnvConfig cfg;
    std::vector<u64> small_v;
    for (const TaskPtr& t : small)
    {
        cuda::DeviceEnv e(ctx, rl::TaskTable::single(t), cfg);
        small_v.push_back(e.cache_view_words());
    }
    for (const auto& [split, prob] : large)
    {
        const TaskPtr big = sokoban(split, prob);
        std::vector<TaskPtr> tasks = small;
        tasks.push_back(big);
        const auto table = rl::TaskTable::create(tasks);
        const std::string why = cuda::DeviceEnv::fast_unsupported(*table, cfg);
        u64 V = 0, V_big = 0;
        u32 S = 0;
        if (why.empty())
        {
            cuda::DeviceEnv e(ctx, table, cfg);
            V = e.cache_view_words();
            S = e.cache_schemas();
            cuda::DeviceEnv eb(ctx, rl::TaskTable::single(big), cfg);
            V_big = eb.cache_view_words();
        }
        std::string sv;
        for (u64 v : small_v)
            sv += (sv.empty() ? "" : ",") + std::to_string(v);
        const u32 I = table->size();
        for (u64 N : envs)
        {
            // padded: every env carries V words; per-group sizing: the large instance's envs V_big, the others their own
            // V (equal env weights)
            u64 per_group = 0;
            for (u32 k = 0; k < I; ++k)
            {
                const u64 n_k = (N + I - 1 - k) / I;  // equal env weights
                per_group += n_k * (k + 1 < I ? small_v[k] : V_big);
            }
            const double states_mb = static_cast<double>(N * table->words() * 8) / 1048576.0;
            std::printf("{\"tool\":\"table_bench\",\"mode\":\"views\",\"large\":\"%s/%s\",\"objects\":[%s],\"words\":[%s],"
                        "\"W\":%u,\"fast\":%s,\"why\":\"%s\",\"V_small\":[%s],\"V_large\":%llu,\"V\":%llu,\"S\":%u,\"N\":%llu,"
                        "\"views_mb\":%.1f,\"views_per_group_mb\":%.1f,\"counts_mb\":%.1f,\"states_mb\":%.1f}\n",
                        split.c_str(), prob.c_str(), objects_of(*table).c_str(), words_of(*table).c_str(), table->words(),
                        why.empty() ? "true" : "false", why.c_str(), sv.c_str(), static_cast<unsigned long long>(V_big),
                        static_cast<unsigned long long>(V), S, static_cast<unsigned long long>(N),
                        static_cast<double>(N * V * 8) / 1048576.0, static_cast<double>(per_group * 8) / 1048576.0,
                        static_cast<double>(N * S * 4) / 1048576.0, states_mb);
            std::fflush(stdout);
        }
    }
}

void run_table(const std::vector<u64>& sizes, int reps)
{
    auto ctx = context();
    const fs::path d = fork_data_dir() / "ipc" / "blocksworld-ipc" / "train";
    std::vector<fs::path> probs;
    for (const auto& e : fs::directory_iterator(d))
        if (e.path().filename().string().starts_with("p") && e.path().extension() == ".pddl")
            probs.push_back(e.path());
    std::sort(probs.begin(), probs.end());
    if (probs.empty())
        throw std::runtime_error("no blocksworld-ipc problems in " + d.string());
    const auto dom = frontend::Domain::from_file(d / "domain.pddl");
    for (u64 n : sizes)
    {
        std::vector<double> t_tasks, t_table, t_upload, t_env;
        u64 bytes = 0, max_objects = 0;
        u32 W = 0;
        for (int r = 0; r < reps; ++r)
        {
            TaskOptions o;
            o.atoms = TaskOptions::Atoms::Frozen;
            double t0 = now_s();
            std::vector<TaskPtr> tasks;
            for (u64 i = 0; i < n; ++i)
                tasks.push_back(Task::create(*dom->instantiate_file(probs[i % probs.size()]), o));
            t_tasks.push_back(now_s() - t0);
            t0 = now_s();
            auto table = rl::TaskTable::create(tasks);
            t_table.push_back(now_s() - t0);
            t0 = now_s();
            auto dt = cuda::DeviceTaskTable::upload(ctx, table);
            cuda::check(cudaStreamSynchronize(ctx->stream()), "sync");
            t_upload.push_back(now_s() - t0);
            bytes = dt->bytes();
            t0 = now_s();
            {
                rl::EnvConfig cfg;
                cuda::DeviceEnv env(ctx, table, cfg);
                cuda::check(cudaStreamSynchronize(env.stream()), "sync");
            }
            t_env.push_back(now_s() - t0);
            W = table->words();
            max_objects = table->max_objects();
        }
        std::printf("{\"tool\":\"table_bench\",\"mode\":\"table\",\"instances\":%llu,\"distinct_problems\":%zu,\"W\":%u,"
                    "\"max_objects\":%llu,\"tasks_ms\":%.1f,\"table_create_ms\":%.2f,\"upload_ms\":%.2f,\"device_bytes\":%llu,"
                    "\"device_env_ms\":%.2f,\"reps\":%d}\n",
                    static_cast<unsigned long long>(n), std::min<size_t>(n, probs.size()), W,
                    static_cast<unsigned long long>(max_objects), 1e3 * median(t_tasks), 1e3 * median(t_table),
                    1e3 * median(t_upload), static_cast<unsigned long long>(bytes), 1e3 * median(t_env), reps);
        std::fflush(stdout);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing mode");
    const std::string mode = argv[1];
    std::vector<std::string> sets;
    std::vector<u64> envs{16384, 65536}, instances{100, 1000};
    int steps = 200, warmup = 10, reps = mode == "expand" ? 5 : 3, walk = 20;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--sets")
            sets = parse_names(value());
        else if (a == "--envs")
            envs = parse_list(value());
        else if (a == "--steps")
            steps = std::stoi(value());
        else if (a == "--warmup")
            warmup = std::stoi(value());
        else if (a == "--reps")
            reps = std::max(1, std::stoi(value()));
        else if (a == "--walk")
            walk = std::stoi(value());
        else if (a == "--ids")
            g_ids = std::stoi(value());
        else if (a == "--layout")
            g_blocked = value() == "blocked";
        else if (a == "--only")
            g_only = value();
        else if (a == "--graph")
            g_graph = std::stoi(value());
        else if (a == "--instances")
            instances = parse_list(value());
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        if (mode == "env")
            run_env(sets, envs, steps, warmup, reps);
        else if (mode == "expand")
            run_expand(sets, envs, reps, walk);
        else if (mode == "views")
            run_views(envs);
        else if (mode == "table")
            run_table(instances, reps);
        else
            usage(("unknown mode " + mode).c_str());
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
