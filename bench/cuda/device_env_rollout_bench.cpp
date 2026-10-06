// mymyr_device_env_rollout_bench: random-policy rollouts of N device environments (cuda::DeviceEnv), one JSON line per
// batch size:
//
//   mymyr_device_env_rollout_bench task.txt [--envs 16384,65536] [--steps 200] [--warmup 10] [--reps 3] [--path auto|fast|general]
//                     [--atoms auto|lazy|frozen] [--max-steps M] [--no-canonical] [--witness] [--labels] [--final]
//                     [--chunk R] [--max-bytes B] [--graph K]
//
// A step is the full environment step: choose (the counter-based RNG), generate the successor (canonical order by
// default), goal test, dead end, rewards, termination, truncation, autoreset, and the successor count of the reached
// state; the outputs reward, terminated, truncated and count are written every step (--labels and --final add the
// action labels and final states). Every rep builds a
// fresh batch at the initial state, warms up, then times `steps` steps between two stream synchronizations; the median
// rep is reported. --graph K captures K steps into a CUDA graph (stream capture of the env's stream) and times
// steps / K replays of it instead (steps rounded down to a multiple of K).

#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
/// FNV-1a over device bytes, chained (`fp` in the JSON line: equal outputs give equal fingerprints across builds).
constexpr u64 k_fnv0 = 14695981039346656037ull;
u64 fnv(u64 h, const void* d, u64 bytes)
{
    std::vector<unsigned char> b(bytes);
    if (bytes)
        cuda::check(cudaMemcpy(b.data(), d, bytes, cudaMemcpyDeviceToHost), "D2H");
    for (const unsigned char c : b)
    {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_env_rollout_bench task.txt [--envs N,N,...] [--steps S] [--warmup W] [--reps R]\n"
                 "       [--path auto|fast|general] [--atoms auto|lazy|frozen] [--max-steps M] [--no-canonical] [--witness]\n"
                 "       [--labels] [--final] [--chunk R] [--max-bytes B] [--graph K]\n",
                 msg);
    std::exit(2);
}

double now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing task file");
    TaskOptions to;
    rl::EnvConfig cfg;
    cfg.seed = 1;
    cuda::DeviceEnv::Path path = cuda::DeviceEnv::Path::Auto;
    std::vector<u64> envs{16384, 65536};
    int steps = 200, warmup = 10, reps = 3, graph = 0;
    bool labels = false, finals = false;
    u64 chunk = 0, max_bytes = u64{6} << 30;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--envs")
        {
            envs.clear();
            std::stringstream ss(value());
            std::string x;
            while (std::getline(ss, x, ','))
                envs.push_back(std::stoull(x));
        }
        else if (a == "--steps")
            steps = std::stoi(value());
        else if (a == "--warmup")
            warmup = std::stoi(value());
        else if (a == "--reps")
            reps = std::max(1, std::stoi(value()));
        else if (a == "--path")
        {
            const std::string v = value();
            path = v == "fast" ? cuda::DeviceEnv::Path::Fast : v == "general" ? cuda::DeviceEnv::Path::General : cuda::DeviceEnv::Path::Auto;
        }
        else if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else if (a == "--max-steps")
            cfg.max_steps = static_cast<u32>(std::stoul(value()));
        else if (a == "--no-canonical")
            cfg.canonical_order = false;
        else if (a == "--witness")
            cfg.witness_pruning = true;
        else if (a == "--labels")
            labels = true;
        else if (a == "--final")
            finals = true;
        else if (a == "--chunk")
            chunk = std::stoull(value());
        else if (a == "--max-bytes")
            max_bytes = std::stoull(value());
        else if (a == "--graph")
            graph = std::stoi(value());
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        const auto task = Task::from_text_file(argv[1], to);
        cuda::ContextOptions co;
        co.max_bytes = max_bytes;
        auto ctx = cuda::DeviceContext::create(0, co);
        cuda::DeviceEnv env(ctx, rl::TaskTable::single(task), cfg, path);
        env.set_chunk_rows(chunk);
        const cudaStream_t s = env.stream();
        const u32 W = env.words(), L = env.label_width(), S = env.cache_schemas();
        const u64 V = env.cache_view_words();
        for (u64 N : envs)
        {
            std::vector<double> secs;
            double branching = 0;
            u64 ended = 0, fp = 0;
            for (int r = 0; r < reps; ++r)
            {
                cuda::DeviceBuffer states(ctx, N * W * 8, s), steps_(ctx, N * 4, s), draws(ctx, N * 8, s),
                    counts(ctx, std::max<u64>(N * S, 1) * 4, s), views(ctx, std::max<u64>(N * V, 1) * 8, s),
                    reward(ctx, N * 4, s), term(ctx, N, s), trunc(ctx, N, s), count(ctx, N * 4, s),
                    schema(ctx, labels ? N * 4 : 4, s), binding(ctx, labels ? N * L * 4 : 4, s),
                    final_states(ctx, finals ? N * W * 8 : 8, s);
                cuda::check(cudaMemsetAsync(draws.data(), 0, N * 8, s), "cudaMemsetAsync");
                rl::EnvBatch b;
                b.states = static_cast<u64*>(states.data());
                b.rows = N;
                b.words = W;
                b.steps = static_cast<i32*>(steps_.data());
                b.draws = static_cast<u64*>(draws.data());
                b.counts = S ? static_cast<u32*>(counts.data()) : nullptr;
                b.views = V ? static_cast<u64*>(views.data()) : nullptr;
                rl::StepOutputs out;
                out.reward = static_cast<f32*>(reward.data());
                out.terminated = static_cast<u8*>(term.data());
                out.truncated = static_cast<u8*>(trunc.data());
                out.count = static_cast<i32*>(count.data());
                if (labels)
                {
                    out.schema = static_cast<i32*>(schema.data());
                    out.binding = static_cast<i32*>(binding.data());
                    out.label_width = L;
                }
                if (finals)
                    out.final_states = static_cast<u64*>(final_states.data());
                env.reset(b);
                for (int t = 0; t < warmup; ++t)
                    env.step(b, out);
                cudaGraph_t g = nullptr;
                cudaGraphExec_t gx = nullptr;
                if (graph > 0)
                {
                    cuda::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
                    for (int t = 0; t < graph; ++t)
                        env.step(b, out);
                    cuda::check(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
                    cuda::check(cudaGraphInstantiate(&gx, g, 0), "cudaGraphInstantiate");
                    cuda::check(cudaGraphLaunch(gx, s), "cudaGraphLaunch");  // uploads the graph
                }
                cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
                const double t0 = now_s();
                if (graph > 0)
                    for (int t = 0; t < steps / graph; ++t)
                        cuda::check(cudaGraphLaunch(gx, s), "cudaGraphLaunch");
                else
                    for (int t = 0; t < steps; ++t)
                        env.step(b, out);
                cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
                secs.push_back(now_s() - t0);
                if (gx)
                {
                    cuda::check(cudaGraphExecDestroy(gx), "cudaGraphExecDestroy");
                    cuda::check(cudaGraphDestroy(g), "cudaGraphDestroy");
                }
                env.check_errors();
                // the mean successor count at the last step, and the environments that ended in it
                std::vector<i32> c(N);
                std::vector<u8> e(N), u(N);
                cuda::check(cudaMemcpy(c.data(), count.data(), N * 4, cudaMemcpyDeviceToHost), "D2H");
                cuda::check(cudaMemcpy(e.data(), term.data(), N, cudaMemcpyDeviceToHost), "D2H");
                cuda::check(cudaMemcpy(u.data(), trunc.data(), N, cudaMemcpyDeviceToHost), "D2H");
                double sum = 0;
                ended = 0;
                for (u64 i = 0; i < N; ++i)
                {
                    sum += c[i];
                    ended += e[i] | u[i];
                }
                branching = sum / static_cast<double>(N);
                // the batch and the last step's outputs
                fp = fnv(k_fnv0, states.data(), N * W * 8);
                for (const cuda::DeviceBuffer* x : {&steps_, &draws, &reward, &count})
                    fp = fnv(fp, x->data(), x == &draws ? N * 8 : N * 4);
                fp = fnv(fnv(fp, term.data(), N), trunc.data(), N);
                if (labels)
                    fp = fnv(fnv(fp, schema.data(), N * 4), binding.data(), N * L * 4);
                if (finals)
                    fp = fnv(fp, final_states.data(), N * W * 8);
            }
            const int timed = graph > 0 ? steps / graph * graph : steps;
            std::vector<double> sorted = secs;
            std::sort(sorted.begin(), sorted.end());
            const double med = sorted[sorted.size() / 2];
            std::string all;
            for (double x : secs)
                all += (all.empty() ? "" : ",") + std::to_string(x);
            std::printf("{\"tool\":\"device_env_rollout_bench\",\"task\":\"%s\",\"N\":%llu,\"path\":\"%s\",\"canonical\":%s,\"W\":%u,\"S\":%u,"
                        "\"V\":%llu,\"steps\":%d,\"secs\":%.5f,\"secs_all\":[%s],\"env_steps_per_s\":%.4g,\"us_per_step\":%.1f,"
                        "\"branching_last\":%.2f,\"ended_last\":%llu,\"max_steps\":%u,\"labels\":%s,\"final\":%s,\"graph\":%d,"
                        "\"fp\":\"%016llx\"}\n",
                        argv[1], static_cast<unsigned long long>(N), env.fast() ? "fast" : "general",
                        cfg.canonical_order ? "true" : "false", W, S, static_cast<unsigned long long>(V), timed, med,
                        all.c_str(), static_cast<double>(N) * timed / med, 1e6 * med / timed, branching,
                        static_cast<unsigned long long>(ended), cfg.max_steps, labels ? "true" : "false",
                        finals ? "true" : "false", graph, static_cast<unsigned long long>(fp));
            std::fflush(stdout);
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
