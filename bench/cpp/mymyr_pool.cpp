// mymyr_pool: rl::CpuEnvPool env-steps/s over a mixed task table (the native side of
// bench/python/bench_table_pool.py), random policy with autoreset, one JSON line per (driver, threads):
//
//   mymyr_pool domain.pddl problem.pddl [problem.pddl ...] [--envs 16384] [--threads 1,8,32] [--steps 50]
//
// Drivers: "pool" (one caller: CpuEnvPool::step over every env, each batch recycled), "pool_async" (one caller, two
// halves of the envs in flight: recv one, send it again, then the other; EnvPool's asynchronous use), "rows" (T plain threads, each stepping its own slice
// of one EnvBatch with HostEnv::step_row: the scaling the row steps allow without the pool's scheduling), "rows_sync"
// (the same with a barrier after every step, as a lockstep batch has).

#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/pool.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;

namespace
{
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<u32> parse_list(const std::string& s)
{
    std::vector<u32> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        out.push_back(static_cast<u32>(std::stoul(x)));
    return out;
}
}  // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> files;
    u32 envs = 16384;
    int steps = 50, warmup = 5;
    std::vector<u32> threads{1, 8, 32};
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "error: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--envs")
            envs = static_cast<u32>(std::stoul(value()));
        else if (a == "--steps")
            steps = std::stoi(value());
        else if (a == "--warmup")
            warmup = std::stoi(value());
        else if (a == "--threads")
            threads = parse_list(value());
        else
            files.push_back(a);
    }
    if (files.size() < 2)
    {
        std::fprintf(stderr, "usage: mymyr_pool domain.pddl problem.pddl [...] [--envs N] [--threads 1,8,32] [--steps S]\n");
        return 2;
    }
    try
    {
        const auto dom = frontend::Domain::from_file(files[0]);
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        std::vector<TaskPtr> tasks;
        for (usize k = 1; k < files.size(); ++k)
            tasks.push_back(Task::create(*dom->instantiate_file(files[k]), to));
        const auto table = rl::TaskTable::create(tasks);
        const u32 I = table->size();
        std::vector<i32> ids(envs);
        for (u32 e = 0; e < envs; ++e)
            ids[e] = static_cast<i32>(e % I);
        rl::EnvConfig cfg;
        cfg.seed = 1;
        cfg.max_steps = 100;
        for (const std::string driver : {"pool", "pool_async", "rows", "rows_sync"})
        {
            double base = 0;
            for (const u32 T : threads)
            {
                double rate = 0, send_ms = 0;
                if (driver == "pool_async")
                {
                    // two halves in flight: the workers always have a queued send
                    rl::CpuEnvPool pool(table, cfg, envs, {.threads = T, .goals = false});
                    (void)pool.reset(nullptr, 0, ids.data(), nullptr, nullptr);
                    std::vector<u32> h0, h1;
                    for (u32 e = 0; e < envs; ++e)
                        (e < envs / 2 ? h0 : h1).push_back(e);
                    u64 t0k = pool.send(h0.data(), h0.size(), {}, nullptr), t1k = pool.send(h1.data(), h1.size(), {}, nullptr);
                    double t0 = 0;
                    for (int t = 0; t < warmup + steps; ++t)
                    {
                        if (t == warmup)
                            t0 = now_s();
                        pool.recycle(pool.recv_ticket(t0k));
                        t0k = pool.send(h0.data(), h0.size(), {}, nullptr);
                        pool.recycle(pool.recv_ticket(t1k));
                        t1k = pool.send(h1.data(), h1.size(), {}, nullptr);
                    }
                    rate = static_cast<double>(envs) * steps / (now_s() - t0);
                    pool.recycle(pool.recv_ticket(t0k));
                    pool.recycle(pool.recv_ticket(t1k));
                }
                else if (driver == "pool")
                {
                    rl::CpuEnvPool pool(table, cfg, envs, {.threads = T, .goals = false});
                    (void)pool.reset(nullptr, 0, ids.data(), nullptr, nullptr);
                    for (int t = 0; t < warmup; ++t)
                        pool.recycle(pool.step(nullptr, 0, {}, nullptr));
                    const double t0 = now_s();
                    double t_send = 0;
                    for (int t = 0; t < steps; ++t)
                    {
                        const double a = now_s();
                        const u64 tk = pool.send(nullptr, 0, {}, nullptr);
                        t_send += now_s() - a;
                        pool.recycle(pool.recv_ticket(tk));
                    }
                    rate = static_cast<double>(envs) * steps / (now_s() - t0);
                    send_ms = 1e3 * t_send / steps;
                }
                else
                {
                    const rl::HostEnv env(table, cfg);
                    const u32 W = table->words(), L = std::max<u32>(1, table->label_width());
                    std::vector<u64> states(u64{envs} * W), draws(envs, 0);
                    std::vector<i32> tid = ids, stp(envs, 0), count(envs), schema(envs), binding(u64{envs} * L);
                    std::vector<f32> reward(envs);
                    std::vector<u8> term(envs), trunc(envs), inval(envs), goal(envs);
                    rl::EnvBatch b;
                    b.states = states.data();
                    b.rows = envs;
                    b.words = W;
                    b.task_ids = tid.data();
                    b.steps = stp.data();
                    b.draws = draws.data();
                    for (u32 e = 0; e < envs; ++e)
                        (void)env.reset_row(b, e, static_cast<u32>(tid[e]), false);
                    const rl::StepOutputs out{reward.data(), term.data(), trunc.data(), count.data(), nullptr,
                                              schema.data(), binding.data(), L, inval.data(), goal.data()};
                    std::barrier sync(static_cast<std::ptrdiff_t>(T) + 1);
                    std::barrier step_sync(static_cast<std::ptrdiff_t>(T));
                    const bool lockstep = driver == "rows_sync";
                    std::vector<std::thread> ths;
                    for (u32 t = 0; t < T; ++t)
                        ths.emplace_back(
                            [&, t]
                            {
                                rl::HostEnv::Scratch s;
                                const u64 lo = u64{envs} * t / T, hi = u64{envs} * (t + 1) / T;
                                for (int k = 0; k < warmup + steps; ++k)
                                {
                                    if (k == warmup)
                                        sync.arrive_and_wait();
                                    for (u64 r = lo; r < hi; ++r)
                                        env.step_row(b, r, out, r, 0, true, -1, s);
                                    if (lockstep)
                                        step_sync.arrive_and_wait();
                                }
                                sync.arrive_and_wait();
                            });
                    sync.arrive_and_wait();
                    const double t0 = now_s();
                    sync.arrive_and_wait();
                    rate = static_cast<double>(envs) * steps / (now_s() - t0);
                    for (std::thread& th : ths)
                        th.join();
                }
                base = base > 0 ? base : rate;
                double load[3] = {0, 0, 0};
                (void)getloadavg(load, 3);
                std::printf("{\"tool\":\"mymyr_pool\",\"driver\":\"%s\",\"threads\":%u,\"envs\":%u,\"instances\":%u,"
                            "\"steps\":%d,\"env_steps_per_s\":%.4g,\"speedup\":%.2f,\"send_ms\":%.3f,\"load1\":%.2f}\n",
                            driver.c_str(), T, envs, I, steps, rate, rate / base, send_ms, load[0]);
                std::fflush(stdout);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
