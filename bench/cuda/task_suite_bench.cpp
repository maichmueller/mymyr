// mymyr_task_suite_bench (one JSON line per measurement): a batch over a task suite (instances of several domains)
// against the same rows as one batch per domain, run one after another on one stream.
//
//   mymyr_task_suite_bench env    [--suite a,b,..] [--envs 16384,65536] [--steps 200] [--warmup 10] [--reps 3] [--graph K]
//                          [--layout interleaved|grouped] [--only suite|apart]
//   mymyr_task_suite_bench expand [--suite a,b,..] [--envs 16384,65536] [--reps 5] [--walk 20] [--layout ...] [--only ...]
//
// The suite's domains are tables of the instance sets of tests/cpp/rl/table_instance_sets.hpp, plus "sokoban": sokoban-ipc instances of
// 54 to 177 objects), five instances each. Env i is in global instance i mod I (interleaved: every domain in turn), or
// the same multiset of instances grouped by domain (grouped: a stable sort by domain). The reference ("apart") is one
// batch per domain of exactly the suite batch's rows of that domain (local ids in the same order, its table's own row
// width), every domain's batch stepped (expanded) one after another on one stream.
// env: random-policy steps of cuda::DeviceEnv (fast path), a rep = fresh batches reset, `warmup` steps, `steps` timed
// steps between two synchronizations; the median rep. --graph K: the timed steps are replays of a CUDA graph of K steps
// (the suite: K suite steps; apart: K steps of every domain's env, in one graph).
// expand: count + write (successors, parents, labels, offsets) of N states reached by `walk` random steps (the suite's
// env, then split by domain for the reference): cuda::SuiteExpander over the suite batch against each domain's
// cuda::DeviceExpander on its rows; the median of `reps`.
// ratio = t_suite / t_apart (gate: <= 1.3).

#include "../../tests/cpp/rl/table_instance_sets.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_suite.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
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
                 "error: %s\nusage: mymyr_task_suite_bench env|expand [--suite a,b] [--envs N,N] [--steps S] [--warmup W]\n"
                 "       [--reps R] [--walk K] [--graph K] [--layout interleaved|grouped] [--only suite|apart]\n",
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

/// table_instance_sets() plus "sokoban" (as mymyr_table_bench).
std::vector<InstanceSet> bench_sets()
{
    std::vector<InstanceSet> out = table_instance_sets();
    const fs::path d = fork_data_dir() / "ipc" / "sokoban-ipc";
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

bool g_grouped = false;
int g_graph = 0;
std::string g_only;

rl::TaskSuitePtr load_suite(const std::vector<std::string>& names)
{
    const std::vector<InstanceSet> all = bench_sets();
    std::vector<rl::TaskTablePtr> tables;
    for (const std::string& n : names)
    {
        const auto it = std::find_if(all.begin(), all.end(), [&](const InstanceSet& s) { return s.name == n; });
        if (it == all.end())
            throw std::runtime_error("unknown set " + n);
        std::vector<TaskPtr> tasks = load_set(*it, TaskOptions::Atoms::Frozen);
        if (tasks.empty())
            throw std::runtime_error("missing PDDL of set " + n);
        tables.push_back(rl::TaskTable::create(tasks));
    }
    return rl::TaskSuite::create(tables);
}

/// The suite batch's global ids, and per domain its rows' local ids (in batch order).
struct Layout
{
    std::vector<i32> ids;
    std::vector<std::vector<i32>> local;
    std::vector<std::vector<u64>> rows;  // per domain: the batch rows
};

Layout layout(const rl::TaskSuite& S, u64 N)
{
    Layout L;
    const u32 I = S.size(), D = S.num_domains();
    L.ids.resize(N);
    for (u64 i = 0; i < N; ++i)
        L.ids[i] = static_cast<i32>(i % I);
    if (g_grouped)
        std::stable_sort(L.ids.begin(), L.ids.end(), [&](i32 a, i32 b)
                         { return S.domain_of(static_cast<u32>(a)) < S.domain_of(static_cast<u32>(b)); });
    L.local.resize(D);
    L.rows.resize(D);
    for (u64 i = 0; i < N; ++i)
    {
        const rl::TaskSuite::Ref r = S.ref(static_cast<u32>(L.ids[i]));
        L.local[r.domain].push_back(static_cast<i32>(r.local));
        L.rows[r.domain].push_back(i);
    }
    return L;
}

/// A device batch of environments (the fast path's arrays).
struct Batch
{
    cuda::DeviceBuffer states, ids, steps, draws, counts, views, reward, term, trunc, count;
    rl::EnvBatch b;
    rl::StepOutputs out;

    Batch(const cuda::ContextPtr& ctx, const cuda::DeviceEnv& env, const std::vector<i32>& h_ids, cudaStream_t s) :
        states(ctx, std::max<u64>(h_ids.size(), 1) * env.words() * 8, s),
        ids(ctx, std::max<u64>(h_ids.size(), 1) * 4, s),
        steps(ctx, std::max<u64>(h_ids.size(), 1) * 4, s),
        draws(ctx, std::max<u64>(h_ids.size(), 1) * 8, s),
        counts(ctx, std::max<u64>(h_ids.size() * env.cache_schemas(), 1) * 4, s),
        views(ctx, std::max<u64>(h_ids.size() * env.cache_view_words(), 1) * 8, s),
        reward(ctx, std::max<u64>(h_ids.size(), 1) * 4, s),
        term(ctx, std::max<u64>(h_ids.size(), 1), s),
        trunc(ctx, std::max<u64>(h_ids.size(), 1), s),
        count(ctx, std::max<u64>(h_ids.size(), 1) * 4, s)
    {
        const u64 N = h_ids.size();
        if (N)
            cuda::check(cudaMemcpyAsync(ids.data(), h_ids.data(), N * 4, cudaMemcpyHostToDevice, s), "H2D");
        cuda::check(cudaMemsetAsync(draws.data(), 0, std::max<u64>(N, 1) * 8, s), "cudaMemsetAsync");
        cuda::check(cudaStreamSynchronize(s), "sync");
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

    /// The batch's state and the last step's outputs, chained onto h.
    [[nodiscard]] u64 fingerprint(u64 h, cudaStream_t s) const
    {
        const u64 N = b.rows;
        h = fnv(h, states.data(), N * b.words * 8, s);
        h = fnv(h, ids.data(), N * 4, s);
        h = fnv(h, steps.data(), N * 4, s);
        h = fnv(h, draws.data(), N * 8, s);
        h = fnv(h, reward.data(), N * 4, s);
        h = fnv(h, term.data(), N, s);
        h = fnv(h, trunc.data(), N, s);
        return fnv(h, count.data(), N * 4, s);
    }
};

/// Seconds per step (median of reps) of the envs stepped one after another on stream s (one env: the suite; several:
/// the domains apart).
double time_envs(const cuda::ContextPtr& ctx, const std::vector<cuda::DeviceEnv*>& envs,
                 const std::vector<std::vector<i32>>& ids, int steps, int warmup, int reps, cudaStream_t s, u64* fp)
{
    std::vector<double> secs;
    for (int r = 0; r < reps; ++r)
    {
        std::vector<std::unique_ptr<Batch>> B;
        for (usize k = 0; k < envs.size(); ++k)
        {
            B.push_back(std::make_unique<Batch>(ctx, *envs[k], ids[k], s));
            envs[k]->reserve(ids[k].size());
            envs[k]->reset(B[k]->b);
        }
        auto step_all = [&]
        {
            for (usize k = 0; k < envs.size(); ++k)
                if (B[k]->b.rows)
                    envs[k]->step(B[k]->b, B[k]->out);
        };
        for (int t = 0; t < warmup; ++t)
            step_all();
        cudaGraph_t g = nullptr;
        cudaGraphExec_t gx = nullptr;
        if (g_graph > 0)
        {
            cuda::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
            for (int t = 0; t < g_graph; ++t)
                step_all();
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
                step_all();
        cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        secs.push_back((now_s() - t0) / timed);
        if (gx)
        {
            cuda::check(cudaGraphExecDestroy(gx), "cudaGraphExecDestroy");
            cuda::check(cudaGraphDestroy(g), "cudaGraphDestroy");
        }
        for (cuda::DeviceEnv* e : envs)
            e->check_errors();
        *fp = k_fnv0;
        for (const auto& b : B)
            *fp = b->fingerprint(*fp, s);
    }
    return median(secs);
}

std::string suite_name(const std::vector<std::string>& names)
{
    std::string s;
    for (const auto& n : names)
        s += (s.empty() ? "" : "+") + n;
    return s;
}

void run_env(const std::vector<std::string>& names, const std::vector<u64>& envs, int steps, int warmup, int reps)
{
    auto ctx = context();
    const rl::TaskSuitePtr S = load_suite(names);
    const u32 D = S->num_domains();
    cuda::Stream stream;
    const cudaStream_t s = stream.get();
    rl::EnvConfig cfg;
    cfg.seed = 1;
    for (u64 N : envs)
    {
        const Layout L = layout(*S, N);
        double t_suite = 0, t_apart = 0;
        u64 fp_suite = 0, fp_apart = 0, fp_one = 0;
        u32 C = 0;
        u64 V = 0;
        if (g_only != "apart")
        {
            cuda::DeviceEnv env(ctx, S, cfg, cuda::DeviceEnv::Path::Fast, s);
            C = env.cache_schemas();
            V = env.cache_view_words();
            t_suite = time_envs(ctx, {&env}, {L.ids}, steps, warmup, reps, s, &fp_suite);
        }
        std::string per;
        if (g_only != "suite")
        {
            std::vector<std::unique_ptr<cuda::DeviceEnv>> apart;
            std::vector<cuda::DeviceEnv*> ptrs;
            for (u32 d = 0; d < D; ++d)
            {
                apart.push_back(std::make_unique<cuda::DeviceEnv>(ctx, S->table(d), cfg, cuda::DeviceEnv::Path::Fast, s));
                ptrs.push_back(apart.back().get());
            }
            t_apart = time_envs(ctx, ptrs, L.local, steps, warmup, reps, s, &fp_apart);
            // each domain's batch alone (where the time goes)
            for (u32 d = 0; d < D; ++d)
            {
                const double t = time_envs(ctx, {apart[d].get()}, {L.local[d]}, steps, warmup, reps, s, &fp_one);
                char buf[160];
                std::snprintf(buf, sizeof buf, "%s{\"domain\":\"%s\",\"N\":%zu,\"W\":%u,\"us_per_step\":%.1f}", d ? "," : "",
                              names[d].c_str(), L.local[d].size(), S->table(d)->words(), 1e6 * t);
                per += buf;
            }
        }
        std::printf("{\"tool\":\"task_suite_bench\",\"mode\":\"env\",\"suite\":\"%s\",\"layout\":\"%s\",\"N\":%llu,\"domains\":%u,"
                    "\"instances\":%u,\"W\":%u,\"C\":%u,\"V\":%llu,\"graph\":%d,\"steps\":%d,\"suite_us_per_step\":%.1f,"
                    "\"apart_us_per_step\":%.1f,\"ratio\":%.3f,\"suite_env_steps_per_s\":%.4g,\"suite_fp\":\"%016llx\","
                    "\"apart_fp\":\"%016llx\",\"per_domain\":[%s]}\n",
                    suite_name(names).c_str(), g_grouped ? "grouped" : "interleaved", static_cast<unsigned long long>(N), D,
                    S->size(), S->words(), C, static_cast<unsigned long long>(V), g_graph, steps, 1e6 * t_suite,
                    1e6 * t_apart, t_apart > 0 ? t_suite / t_apart : 0.0, t_suite > 0 ? static_cast<double>(N) / t_suite : 0.0,
                    static_cast<unsigned long long>(fp_suite), static_cast<unsigned long long>(fp_apart), per.c_str());
        std::fflush(stdout);
    }
}

/// The states of the suite batch after `walk` random steps (host copy, rows of S.words()).
std::vector<u64> walked(const cuda::ContextPtr& ctx, const rl::TaskSuitePtr& S, const std::vector<i32>& ids, int walk,
                        cudaStream_t s)
{
    rl::EnvConfig cfg;
    cfg.seed = 7;
    cuda::DeviceEnv env(ctx, S, cfg, cuda::DeviceEnv::Path::Auto, s);
    Batch B(ctx, env, ids, s);
    env.reset(B.b);
    for (int t = 0; t < walk; ++t)
        env.step(B.b, B.out);
    env.check_errors();
    std::vector<u64> out(ids.size() * env.words());
    cuda::check(cudaMemcpyAsync(out.data(), B.states.data(), out.size() * 8, cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return out;
}

/// The flat outputs of an expansion of up to `cap` successors.
struct Out
{
    cuda::DeviceBuffer succ, parent, schema, binding, offsets;
    rl::Expansion x;
    Out(const cuda::ContextPtr& ctx, u64 cap, u64 N, u32 W, u32 L, cudaStream_t s) :
        succ(ctx, std::max<u64>(cap, 1) * W * 8, s), parent(ctx, std::max<u64>(cap, 1) * 4, s),
        schema(ctx, std::max<u64>(cap, 1) * 4, s), binding(ctx, std::max<u64>(cap, 1) * L * 4, s),
        offsets(ctx, (N + 1) * 4, s)
    {
        x.capacity = cap;
        x.words = W;
        x.label_width = L;
        x.succ = static_cast<u64*>(succ.data());
        x.parent = static_cast<i32*>(parent.data());
        x.schema = static_cast<i32*>(schema.data());
        x.binding = static_cast<i32*>(binding.data());
        x.offsets = static_cast<i32*>(offsets.data());
    }

    /// The written outputs (x.total rows below the capacity) and words_needed, chained onto h.
    [[nodiscard]] u64 fingerprint(u64 h, u64 N, cudaStream_t s) const
    {
        const u64 n = std::min(x.total, x.capacity);
        h = fnv(h, succ.data(), n * x.words * 8, s);
        h = fnv(h, parent.data(), n * 4, s);
        h = fnv(h, schema.data(), n * 4, s);
        h = fnv(h, binding.data(), n * x.label_width * 4, s);
        h = fnv(h, offsets.data(), (N + 1) * 4, s);
        return (h ^ x.words_needed) * 1099511628211ull;
    }
};

template<class T>
cuda::DeviceBuffer to_device(const cuda::ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    cuda::DeviceBuffer d(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return d;
}

void run_expand(const std::vector<std::string>& names, const std::vector<u64>& envs, int reps, int walk)
{
    auto ctx = context();
    const rl::TaskSuitePtr S = load_suite(names);
    const u32 D = S->num_domains(), W = S->words(), L = std::max<u32>(1, S->label_width());
    cuda::Stream stream;
    const cudaStream_t s = stream.get();
    for (u64 N : envs)
    {
        const Layout lay = layout(*S, N);
        const std::vector<u64> rows = walked(ctx, S, lay.ids, walk, s);
        double t_suite = 0, t_apart = 0;
        u64 total = 0, fp_suite = 0, fp_apart = 0;
        if (g_only != "apart")
        {
            const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s), d_ids = to_device(ctx, lay.ids, s);
            cuda::SuiteExpander ex(ctx, S, s);
            const rl::StateBatchView in{static_cast<const u64*>(d_rows.data()), N, W, 0};
            total = ex.count(in, static_cast<const i32*>(d_ids.data()));
            Out o(ctx, total, N, W, L, s);
            ex.write(o.x);  // warm up
            std::vector<double> secs;
            for (int r = 0; r < reps; ++r)
            {
                const double t0 = now_s();
                ex.expand(in, static_cast<const i32*>(d_ids.data()), o.x);
                cuda::check(cudaStreamSynchronize(s), "sync");
                secs.push_back(now_s() - t0);
            }
            t_suite = median(secs);
            fp_suite = o.fingerprint(k_fnv0, N, s);
        }
        if (g_only != "suite")
        {
            // each domain's rows at its table's width, its own expander and outputs
            std::vector<cuda::DeviceBuffer> d_rows, d_ids;
            std::vector<std::unique_ptr<cuda::DeviceExpander>> ex;
            std::vector<std::unique_ptr<Out>> outs;
            for (u32 d = 0; d < D; ++d)
            {
                const u32 Wd = S->table(d)->words();
                std::vector<u64> r(lay.rows[d].size() * Wd);
                for (u64 k = 0; k < lay.rows[d].size(); ++k)
                    std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(lay.rows[d][k] * W), Wd,
                                r.begin() + static_cast<std::ptrdiff_t>(k * Wd));
                d_rows.push_back(to_device(ctx, r, s));
                d_ids.push_back(to_device(ctx, lay.local[d], s));
                ex.push_back(std::make_unique<cuda::DeviceExpander>(ctx, S->table(d), s));
                const rl::StateBatchView in{static_cast<const u64*>(d_rows[d].data()), lay.rows[d].size(), Wd, 0};
                const u64 t = ex[d]->count(in, static_cast<const i32*>(d_ids[d].data()));
                outs.push_back(std::make_unique<Out>(ctx, t, lay.rows[d].size(), Wd, std::max<u32>(1, S->table(d)->label_width()), s));
                ex[d]->write(outs[d]->x);  // warm up
            }
            std::vector<double> secs;
            for (int r = 0; r < reps; ++r)
            {
                const double t0 = now_s();
                for (u32 d = 0; d < D; ++d)
                {
                    const rl::StateBatchView in{static_cast<const u64*>(d_rows[d].data()), lay.rows[d].size(),
                                                S->table(d)->words(), 0};
                    ex[d]->expand(in, static_cast<const i32*>(d_ids[d].data()), outs[d]->x);
                }
                cuda::check(cudaStreamSynchronize(s), "sync");
                secs.push_back(now_s() - t0);
            }
            t_apart = median(secs);
            fp_apart = k_fnv0;
            for (u32 d = 0; d < D; ++d)
                fp_apart = outs[d]->fingerprint(fp_apart, lay.rows[d].size(), s);
        }
        std::printf("{\"tool\":\"task_suite_bench\",\"mode\":\"expand\",\"suite\":\"%s\",\"layout\":\"%s\",\"N\":%llu,\"domains\":%u,"
                    "\"instances\":%u,\"W\":%u,\"walk\":%d,\"successors\":%llu,\"suite_ms\":%.3f,\"apart_ms\":%.3f,"
                    "\"ratio\":%.3f,\"suite_states_per_s\":%.4g,\"suite_fp\":\"%016llx\",\"apart_fp\":\"%016llx\"}\n",
                    suite_name(names).c_str(), g_grouped ? "grouped" : "interleaved", static_cast<unsigned long long>(N), D,
                    S->size(), W, walk, static_cast<unsigned long long>(total), 1e3 * t_suite, 1e3 * t_apart,
                    t_apart > 0 ? t_suite / t_apart : 0.0, t_suite > 0 ? static_cast<double>(N) / t_suite : 0.0,
                    static_cast<unsigned long long>(fp_suite), static_cast<unsigned long long>(fp_apart));
        std::fflush(stdout);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing mode");
    const std::string mode = argv[1];
    std::vector<std::string> suite{"gripper", "blocks"};
    std::vector<u64> envs{16384, 65536};
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
        if (a == "--suite")
            suite = parse_names(value());
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
        else if (a == "--graph")
            g_graph = std::stoi(value());
        else if (a == "--layout")
            g_grouped = value() == "grouped";
        else if (a == "--only")
            g_only = value();
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        if (mode == "env")
            run_env(suite, envs, steps, warmup, reps);
        else if (mode == "expand")
            run_expand(suite, envs, reps, walk);
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
