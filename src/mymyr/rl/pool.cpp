// CpuEnvPool: EnvPool-style asynchronous environments on the CPU (include/mymyr/rl/pool.hpp).

#include "mymyr/rl/pool.hpp"

#include "mymyr/core/threads.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace mymyr::rl
{
namespace
{
struct Job
{
    u64 ticket = 0;
    bool all = false;          // every env, in order (env_ids is empty)
    std::vector<u32> env_ids;
    std::vector<i64> actions;  // empty: the random policy
    std::vector<i32> next;     // empty: autoresets keep the instance
    PoolBatch out;
    StepOutputs so;
    // the rows in home ranges, one per worker: worker t claims pieces of range t first, then of the others (so an
    // env stays on its worker's cores from step to step while the loads balance)
    struct alignas(64) Range
    {
        std::atomic<u64> next{0};  // the range's next unclaimed row (may run past hi)
        u64 lo = 0, hi = 0;
    };
    std::unique_ptr<Range[]> ranges;
    u32 parts = 0;
    u64 piece = 1;                      // rows per claim
    alignas(64) std::atomic<u64> finished{0};  // rows done

    [[nodiscard]] bool claimable() const
    {
        for (u32 p = 0; p < parts; ++p)
            if (ranges[p].next.load(std::memory_order_relaxed) < ranges[p].hi)
                return true;
        return false;
    }
    bool receiving = false;        // a recv owns it (under the pool's lock)
    std::exception_ptr error;      // the first step error (under the pool's lock)
};

void size_batch(PoolBatch& b, u64 n, u32 W, u32 NN, u32 L, bool goals, bool step)
{
    b.rows = n;
    b.words = W;
    b.numeric_words = NN;
    b.label_width = L;
    b.goals = goals;
    b.step = step;
    b.env_ids.resize(n);
    b.states.resize(n * (W + NN));
    b.task_ids.resize(n);
    b.count.resize(n);
    b.steps.resize(n);
    b.goal_pos.resize(goals ? n * W : 0);
    b.goal_neg.resize(goals ? n * W : 0);
    b.reward.resize(step ? n : 0);
    b.terminated.resize(step ? n : 0);
    b.truncated.resize(step ? n : 0);
    b.invalid.resize(step ? n : 0);
    b.goal.resize(step ? n : 0);
    b.final_states.resize(step ? n * (W + NN) : 0);
    b.schema.resize(step ? n : 0);
    b.binding.resize(step ? n * L : 0);
}

template<class V>
void append(V& dst, const V& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

/// Re-lays `rows` rows of `from` words, each followed by `tail` words, as rows of `to` >= `from` words (zeros between).
template<class V>
void widen_rows(V& v, u64 rows, u32 from, u32 to, u32 tail)
{
    if (rows == 0 || from == to || v.empty())
        return;
    v.resize(rows * (to + tail));
    for (u64 r = rows; r-- > 0;)
    {
        u64* src = v.data() + r * (from + tail);
        u64* dst = v.data() + r * (to + tail);
        std::copy_backward(src + from, src + from + tail, dst + to + tail);
        std::fill(dst + from, dst + to, u64{0});
        std::copy_backward(src, src + from, dst + from);
    }
}

/// Widens a batch's rows to W atom words (W >= b.words).
void widen(PoolBatch& b, u32 W)
{
    widen_rows(b.states, b.rows, b.words, W, b.numeric_words);
    widen_rows(b.final_states, b.rows, b.words, W, b.numeric_words);
    widen_rows(b.goal_pos, b.rows, b.words, W, 0);
    widen_rows(b.goal_neg, b.rows, b.words, W, 0);
    b.words = W;
}
}  // namespace

struct CpuEnvPool::Impl
{
    HostEnv env;
    u32 N = 0, T = 1, W = 1, NN = 0, L = 1;
    u32 fixed_words = 0;  // PoolOptions::words: the batches' row width (0: each batch's widest instance's)
    std::vector<u32> inst_words;  // the atom words of each instance (global ids)
    bool uniform = true;          // every instance has W atom words
    bool goals = false;
    std::vector<u64> states, draws, gpos, gneg;
    std::vector<i32> task_ids, steps;
    EnvBatch batch;

    // under mu: the envs held by sends in flight, and the marks of the env id checks
    mutable std::mutex mu;
    std::vector<u8> busy;
    u64 busy_count = 0;
    std::vector<u32> mark;
    u32 stamp = 0;
    std::condition_variable work_cv;
    std::deque<std::shared_ptr<Job>> jobs;  // sent, not yet received, in send order
    u64 next_ticket = 1;
    bool quit = false;
    std::vector<std::thread> workers;
    // per worker, a cache line apart (a row's step writes its scratch vectors' sizes)
    struct alignas(64) WorkerScratch
    {
        HostEnv::Scratch s;
    };
    std::vector<WorkerScratch> scratch;
    // recycled batches (recycle()), under spare_mu
    std::mutex spare_mu;
    std::vector<PoolBatch> spare;

    Impl(TaskSuitePtr suite, const EnvConfig& config) : env(std::move(suite), config) {}

    /// Ends the workers' loops and joins them.
    void stop_workers() noexcept
    {
        {
            std::lock_guard lock(mu);
            quit = true;
        }
        work_cv.notify_all();
        for (std::thread& t : workers)
            t.join();
        workers.clear();
    }

    /// A recycled batch (arrays with capacity), or an empty one.
    PoolBatch take_spare()
    {
        std::lock_guard lock(spare_mu);
        if (spare.empty())
            return {};
        PoolBatch b = std::move(spare.back());
        spare.pop_back();
        return b;
    }

    void put_spare(PoolBatch&& b)
    {
        std::lock_guard lock(spare_mu);
        if (spare.size() < std::max<usize>(8, 2 * usize{T}))
            spare.push_back(std::move(b));
    }

    /// The atom words of instance g's rows.
    [[nodiscard]] u32 instance_words(i32 g) const { return inst_words[static_cast<u32>(g)]; }

    /// The row width of a batch of envs `ids` (under the lock: none in flight) whose rows end in instances `next`
    /// (null: their current ones; else both count, the step may autoreset into `next`).
    [[nodiscard]] u32 batch_words(const u32* ids, u64 n, const i32* next, bool current) const
    {
        if (fixed_words)
            return fixed_words;
        if (uniform)
            return W;
        u32 w = 1;
        for (u64 p = 0; p < n; ++p)
        {
            const u32 e = ids ? ids[p] : static_cast<u32>(p);
            if (current)
                w = std::max(w, instance_words(task_ids[e]));
            if (next)
                w = std::max(w, instance_words(next[p]));
        }
        return w;
    }

    /// Copies env `e`'s observation into row `pos` of `out`: its instance's words, zeros up to out.words, its numeric
    /// words.
    void observe(u32 e, PoolBatch& out, u64 pos) const
    {
        const u32 RW = W + NN, OW = out.words, w = instance_words(task_ids[e]);
        const u64* s = states.data() + u64{e} * RW;
        u64* d = out.states.data() + pos * (OW + NN);
        std::copy_n(s, w, d);
        std::fill(d + w, d + OW, u64{0});
        std::copy_n(s + W, NN, d + OW);
        out.task_ids[pos] = task_ids[e];
        out.steps[pos] = steps[e];
        out.env_ids[pos] = e;
        if (goals)
        {
            u64* gp = out.goal_pos.data() + pos * OW;
            u64* gn = out.goal_neg.data() + pos * OW;
            std::copy_n(gpos.data() + u64{e} * W, w, gp);
            std::fill(gp + w, gp + OW, u64{0});
            std::copy_n(gneg.data() + u64{e} * W, w, gn);
            std::fill(gn + w, gn + OW, u64{0});
        }
    }

    void run_rows(Job& j, u64 lo, u64 hi, HostEnv::Scratch& s)
    {
        const bool random = j.actions.empty();
        for (u64 p = lo; p < hi; ++p)
        {
            const u32 e = j.all ? static_cast<u32>(p) : j.env_ids[p];
            env.step_row(batch, e, j.so, p, random ? 0 : j.actions[p], random, j.next.empty() ? -1 : j.next[p], s);
            observe(e, j.out, p);
        }
    }

    /// Workers: take the oldest send with unclaimed rows (under the lock), then claim pieces of its home ranges (their
    /// own first) with the ranges' atomic cursors until none is left.
    void worker(u32 t)
    {
        for (;;)
        {
            std::shared_ptr<Job> j;
            {
                std::unique_lock lock(mu);
                work_cv.wait(lock,
                             [&]
                             {
                                 if (quit)
                                     return true;
                                 for (const auto& x : jobs)
                                     if (x->claimable())
                                     {
                                         j = x;
                                         return true;
                                     }
                                 return false;
                             });
                if (quit)
                    return;
            }
            for (u32 k = 0; k < j->parts; ++k)
            {
                Job::Range& r = j->ranges[(t + k) % j->parts];
                for (u64 lo = r.next.fetch_add(j->piece); lo < r.hi; lo = r.next.fetch_add(j->piece))
                {
                    const u64 hi = std::min(r.hi, lo + j->piece);
                    try
                    {
                        run_rows(*j, lo, hi, scratch[t].s);
                    }
                    catch (...)
                    {
                        std::lock_guard lock(mu);
                        if (!j->error)
                            j->error = std::current_exception();
                    }
                    // (release: the rows' results before the count; the receiver waits on the count)
                    if (j->finished.fetch_add(hi - lo, std::memory_order_acq_rel) + (hi - lo) == j->out.rows)
                        j->finished.notify_all();
                }
            }
        }
    }

    /// Waits for the given jobs (owned by the caller: receiving; without the lock), removes them, frees their envs and
    /// merges them.
    PoolBatch collect(const std::vector<std::shared_ptr<Job>>& take)
    {
        for (const auto& j : take)
            for (u64 f = j->finished.load(std::memory_order_acquire); f != j->out.rows;
                 f = j->finished.load(std::memory_order_acquire))
                j->finished.wait(f, std::memory_order_acquire);
        std::unique_lock lock(mu);
        std::exception_ptr err;
        for (const auto& j : take)
        {
            if (j->all)
                std::fill(busy.begin(), busy.end(), u8{0});
            else
                for (u32 e : j->env_ids)
                    busy[e] = 0;
            busy_count -= j->out.rows;
            std::erase(jobs, j);
            if (j->error && !err)
                err = j->error;
        }
        lock.unlock();
        if (err)
            std::rethrow_exception(err);
        if (take.size() == 1)
            return std::move(take[0]->out);
        u32 width = 1;
        for (const auto& j : take)
            width = std::max(width, j->out.words);
        PoolBatch out = std::move(take[0]->out);
        widen(out, width);
        for (usize k = 1; k < take.size(); ++k)
        {
            PoolBatch& o = take[k]->out;
            widen(o, width);
            out.rows += o.rows;
            append(out.env_ids, o.env_ids);
            append(out.states, o.states);
            append(out.task_ids, o.task_ids);
            append(out.count, o.count);
            append(out.steps, o.steps);
            append(out.goal_pos, o.goal_pos);
            append(out.goal_neg, o.goal_neg);
            append(out.reward, o.reward);
            append(out.terminated, o.terminated);
            append(out.truncated, o.truncated);
            append(out.invalid, o.invalid);
            append(out.goal, o.goal);
            append(out.final_states, o.final_states);
            append(out.schema, o.schema);
            append(out.binding, o.binding);
            put_spare(std::move(o));
        }
        return out;
    }

    /// The env ids of a call: `ids` [n], or every env (null).
    [[nodiscard]] std::vector<u32> env_list(const u32* ids, u64 n) const
    {
        std::vector<u32> v;
        if (!ids)
        {
            v.resize(N);
            for (u32 e = 0; e < N; ++e)
                v[e] = e;
        }
        else
            v.assign(ids, ids + n);
        return v;
    }

    /// Checks env ids under the lock: known, distinct, not in flight (`all`: they are every env, in order).
    void check_envs(const std::vector<u32>& v, bool all, const char* what)
    {
        auto fail = [&](u32 e, const char* why)
        {
            throw std::invalid_argument(std::string("mymyr: CpuEnvPool.") + what + ": env " + std::to_string(e) + why);
        };
        if (all)
        {
            if (busy_count)
                for (u32 e = 0; e < N; ++e)
                    if (busy[e])
                        fail(e, " has a step in flight (recv its results first)");
            return;
        }
        if (++stamp == 0)
        {
            std::fill(mark.begin(), mark.end(), 0u);
            stamp = 1;
        }
        for (u32 e : v)
        {
            if (e >= N)
                throw std::invalid_argument(std::string("mymyr: CpuEnvPool.") + what + ": env id " + std::to_string(e) +
                                            " is outside the pool's " + std::to_string(N) + " envs");
            if (mark[e] == stamp)
                fail(e, " is given twice");
            if (busy[e])
                fail(e, " has a step in flight (recv its results first)");
            mark[e] = stamp;
        }
    }
};

CpuEnvPool::CpuEnvPool(TaskSuitePtr suite, const EnvConfig& config, u32 num_envs, const PoolOptions& options)
    : m_impl(std::make_unique<Impl>(std::move(suite), config))
{
    Impl& I = *m_impl;
    const TaskSuite& tt = *I.env.suite();
    if (num_envs == 0)
        throw std::invalid_argument("mymyr: CpuEnvPool: no envs");
    if (num_envs > static_cast<u32>(std::numeric_limits<i32>::max()))
        throw std::invalid_argument("mymyr: CpuEnvPool: more than 2^31 - 1 envs");
    I.N = num_envs;
    I.T = resolve_threads(options.threads);
    I.W = tt.words();
    I.NN = tt.numeric_words();
    if (options.words && options.words < I.W)
        throw std::invalid_argument("mymyr: CpuEnvPool: rows of " + std::to_string(options.words) + " words; the " +
                                    tt.noun() + "'s widest instance has " + std::to_string(I.W));
    I.fixed_words = options.words;
    I.inst_words.resize(tt.size());
    for (u32 g = 0; g < tt.size(); ++g)
    {
        I.inst_words[g] = tt.instance(g).words;
        I.uniform = I.uniform && I.inst_words[g] == I.W;
    }
    I.L = std::max<u32>(1, tt.label_width());
    I.goals = options.goals;
    const u64 RW = I.W + I.NN;
    I.states.assign(u64{I.N} * RW, 0);
    I.draws.assign(I.N, 0);
    I.task_ids.assign(I.N, 0);
    I.steps.assign(I.N, 0);
    if (I.goals)
    {
        I.gpos.assign(u64{I.N} * I.W, 0);
        I.gneg.assign(u64{I.N} * I.W, 0);
    }
    I.busy.assign(I.N, 0);
    I.mark.assign(I.N, 0);
    EnvBatch& b = I.batch;
    b.states = I.states.data();
    b.rows = I.N;
    b.words = I.W;
    b.numeric_words = I.NN;
    b.task_ids = I.task_ids.data();
    b.goal_pos = I.goals ? I.gpos.data() : nullptr;
    b.goal_neg = I.goals ? I.gneg.data() : nullptr;
    b.steps = I.steps.data();
    b.draws = I.draws.data();
    b.first_env = 0;
    check_env(tt, b, nullptr);
    I.env.reset(b);  // every env at instance 0's initial state
    I.scratch.resize(I.T);
    I.workers.reserve(I.T);
    for (u32 t = 0; t < I.T; ++t)
    {
        try
        {
            I.workers.emplace_back([&I, t] { I.worker(t); });
        }
        catch (const std::exception& e)
        {
            I.stop_workers();
            throw ThreadStartError(t, I.T, e);
        }
    }
}

CpuEnvPool::~CpuEnvPool() { m_impl->stop_workers(); }

const TaskSuitePtr& CpuEnvPool::suite() const noexcept { return m_impl->env.suite(); }
const EnvConfig& CpuEnvPool::config() const noexcept { return m_impl->env.config(); }
u32 CpuEnvPool::num_envs() const noexcept { return m_impl->N; }
u32 CpuEnvPool::threads() const noexcept { return m_impl->T; }
bool CpuEnvPool::goals() const noexcept { return m_impl->goals; }
u32 CpuEnvPool::words() const noexcept { return m_impl->fixed_words; }
u32 CpuEnvPool::numeric_words() const noexcept { return m_impl->NN; }
u32 CpuEnvPool::label_width() const noexcept { return m_impl->L; }

PoolBatch CpuEnvPool::reset(const u32* env_ids, u64 n, const i32* task_ids, const u64* goal_pos, const u64* goal_neg,
                            u32 goal_words)
{
    Impl& I = *m_impl;
    if ((goal_pos == nullptr) != (goal_neg == nullptr))
        throw std::invalid_argument("mymyr: CpuEnvPool.reset: goals need both goal_pos and goal_neg");
    if (goal_pos && !I.goals)
        throw std::invalid_argument("mymyr: CpuEnvPool.reset: goal masks given, but the pool has no per-env goals");
    const std::vector<u32> ids = I.env_list(env_ids, n);
    I.env.suite()->check_task_ids(task_ids, ids.size());
    std::lock_guard lock(I.mu);
    I.check_envs(ids, env_ids == nullptr, "reset");
    const bool current = task_ids == nullptr;
    const u32 W = I.batch_words(ids.data(), ids.size(), task_ids, current);
    if (goal_pos)
        for (u64 p = 0; p < ids.size(); ++p)
        {
            const u32 w = I.instance_words(task_ids ? task_ids[p] : I.task_ids[ids[p]]);
            if (goal_words < w)
                throw std::invalid_argument("mymyr: CpuEnvPool.reset: goal masks of " + std::to_string(goal_words) +
                                            " words; env " + std::to_string(ids[p]) + "'s instance has " +
                                            std::to_string(w));
        }
    PoolBatch out = I.take_spare();
    size_batch(out, ids.size(), W, I.NN, I.L, I.goals, false);
    for (u64 p = 0; p < ids.size(); ++p)
    {
        const u32 e = ids[p];
        const u32 inst = task_ids ? static_cast<u32>(task_ids[p]) : static_cast<u32>(I.task_ids[e]);
        out.count[p] = static_cast<i32>(I.env.reset_row(I.batch, e, inst, goal_pos != nullptr));
        if (goal_pos)
        {
            // the instance's words of the given masks; the rest of the env's masks is zero
            const u32 w = I.instance_words(static_cast<i32>(inst));
            u64* gp = I.gpos.data() + u64{e} * I.W;
            u64* gn = I.gneg.data() + u64{e} * I.W;
            std::copy_n(goal_pos + p * goal_words, w, gp);
            std::fill(gp + w, gp + I.W, u64{0});
            std::copy_n(goal_neg + p * goal_words, w, gn);
            std::fill(gn + w, gn + I.W, u64{0});
        }
        I.observe(e, out, p);
    }
    return out;
}

u64 CpuEnvPool::send(const u32* env_ids, u64 n, Actions actions, const i32* next_task_ids)
{
    Impl& I = *m_impl;
    if (actions.v64 && actions.v32)
        throw std::invalid_argument("mymyr: CpuEnvPool.send: actions given as int64 and as int32");
    auto job = std::make_shared<Job>();
    job->all = env_ids == nullptr;
    if (!job->all)
        job->env_ids.assign(env_ids, env_ids + n);
    const u64 B = job->all ? I.N : n;
    I.env.suite()->check_task_ids(next_task_ids, B);
    if (actions.given())
    {
        job->actions.resize(B);
        for (u64 p = 0; p < B; ++p)
            job->actions[p] = actions.at(p);
    }
    if (next_task_ids)
        job->next.assign(next_task_ids, next_task_ids + B);
    job->out = I.take_spare();
    // home ranges of at least 128 rows, at most one per worker, claimed in pieces of about a quarter of a range
    job->parts = static_cast<u32>(std::clamp<u64>(B / 128, 1, I.T));
    job->ranges = std::make_unique<Job::Range[]>(job->parts);
    for (u32 p = 0; p < job->parts; ++p)
    {
        job->ranges[p].lo = B * p / job->parts;
        job->ranges[p].hi = B * (p + 1) / job->parts;
        job->ranges[p].next.store(job->ranges[p].lo, std::memory_order_relaxed);
    }
    job->piece = std::clamp<u64>(B / (u64{job->parts} * 4), 16, 256);
    {
        std::lock_guard lock(I.mu);
        I.check_envs(job->env_ids, job->all, "send");
        // the batch's width: the envs' instances' and those they may autoreset into (read under the lock: no env of
        // the send is in flight)
        const u32 W = I.batch_words(job->all ? nullptr : job->env_ids.data(), B, next_task_ids, true);
        size_batch(job->out, B, W, I.NN, I.L, I.goals, true);
        PoolBatch& o = job->out;
        job->so = StepOutputs{o.reward.data(), o.terminated.data(), o.truncated.data(), o.count.data(),
                              o.final_states.data(), o.schema.data(), o.binding.data(), I.L, o.invalid.data(),
                              o.goal.data(), W};
        job->ticket = I.next_ticket++;
        if (job->all)
            std::fill(I.busy.begin(), I.busy.end(), u8{1});
        else
            for (u32 e : job->env_ids)
                I.busy[e] = 1;
        I.busy_count += B;
        I.jobs.push_back(job);
    }
    // as many workers as the send has home ranges
    if (job->parts >= I.T)
        I.work_cv.notify_all();
    else
        for (u32 p = 0; p < job->parts; ++p)
            I.work_cv.notify_one();
    return job->ticket;
}

PoolBatch CpuEnvPool::recv(u64 min_rows)
{
    Impl& I = *m_impl;
    std::unique_lock lock(I.mu);
    std::vector<std::shared_ptr<Job>> take;
    u64 rows = 0;
    for (const auto& j : I.jobs)
    {
        if (j->receiving)
            continue;
        take.push_back(j);
        rows += j->out.rows;
        if (rows >= min_rows)
            break;
    }
    if (take.empty() || rows < min_rows)
        throw std::invalid_argument("mymyr: CpuEnvPool.recv: " + std::to_string(rows) + " rows pending, " +
                                    std::to_string(std::max<u64>(min_rows, 1)) + " asked for (send first)");
    for (const auto& j : take)
        j->receiving = true;
    lock.unlock();
    return I.collect(take);
}

PoolBatch CpuEnvPool::recv_ticket(u64 ticket)
{
    Impl& I = *m_impl;
    std::shared_ptr<Job> job;
    {
        std::lock_guard lock(I.mu);
        for (const auto& j : I.jobs)
            if (j->ticket == ticket && !j->receiving)
            {
                j->receiving = true;
                job = j;
                break;
            }
    }
    if (!job)
        throw std::invalid_argument("mymyr: CpuEnvPool.recv: no pending send has ticket " + std::to_string(ticket));
    return I.collect({job});
}

PoolBatch CpuEnvPool::step(const u32* env_ids, u64 n, Actions actions, const i32* next_task_ids)
{
    return recv_ticket(send(env_ids, n, actions, next_task_ids));
}

void CpuEnvPool::recycle(PoolBatch&& batch) { m_impl->put_spare(std::move(batch)); }

u64 CpuEnvPool::pending() const
{
    std::lock_guard lock(m_impl->mu);
    return m_impl->jobs.size();
}
}  // namespace mymyr::rl
