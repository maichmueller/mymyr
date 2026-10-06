// mymyr_device_conditional_bench: the device BrFS on tasks with conditional effects and axioms, printing one JSON
// line. The task is an exported text file or a PDDL domain and problem (the loki front end).
//
//   mymyr_device_conditional_bench task.txt | domain.pddl problem.pddl [--atoms auto|lazy|frozen] [--info] [--no-witness]
//                  [--no-canonical] [--chunk N] [--expected N] [--max-states N] [--stop-at-goal] [--fp] [--reps R]
//                  [--cpu T[,T...]] [--cpu-reps R] [--max-bytes B] [--view-bytes B] [--matching auto|fixed|fc]
//                  [--timings]
//
// --info prints the task's conditional effects, axioms and the device placement of its schemas, and stops.
// --cpu T runs the CPU BrFS with T threads (deterministic ids; 0 = all hardware threads) after the device runs, and
// reports its counts, time (median of --cpu-reps) and whether it equals the device's result.
// --matching sets TaskOptions::matching (the matchers of the CPU and the device; auto: forward checking above
// fc_auto_free_params free parameters). --timings records the per-phase device times (events per chunk: the host drives
// every chunk, no device loops).

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"
#if MYMYR_DEVICE_CONDITIONAL_FRONTEND
#include "mymyr/frontend/domain.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;

namespace
{
[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_conditional_bench task.txt | domain.pddl problem.pddl [--atoms auto|lazy|frozen] [--info]\n"
                 "       [--no-witness] [--no-canonical] [--chunk N] [--expected N] [--max-states N] [--stop-at-goal] [--fp]\n"
                 "       [--reps R] [--cpu T[,T...]] [--cpu-reps R] [--max-bytes B] [--view-bytes B] [--matching auto|fixed|fc]\n"
                 "       [--timings]\n",
                 msg);
    std::exit(2);
}

std::string base_name(const std::string& p) { return p.substr(p.find_last_of('/') + 1); }

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

std::string join(const std::vector<double>& v)
{
    std::string s;
    for (double x : v)
        s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
}

/// The task's conditional effects, axioms and schema placement as JSON members (no braces).
std::string info(const TaskPtr& tp)
{
    const Task& task = *tp;
    const plan::Compiled& C = task.compiled();
    u32 ce_schemas = 0, ces = 0, ce_forall = 0, ce_max_steps = 0, ce_fc = 0, ce_driven = 0, ce_max_total = 0;
    for (const plan::Schema& s : C.schemas)
    {
        ce_schemas += s.ces.empty() ? 0 : 1;
        for (const plan::CondEffect& ce : s.ces)
        {
            ++ces;
            ce_forall += ce.extras ? 1 : 0;
            ce_max_steps = std::max<u32>(ce_max_steps, static_cast<u32>(ce.cond.steps.size()));
            ce_max_total = std::max(ce_max_total, ce.cond.total);
            ce_fc += ce.cond.use_fc ? 1 : 0;
            ce_driven += ce.driver != plan::CondEffect::k_no_driver ? 1 : 0;
        }
    }
    u32 axioms = 0, recursive = 0, ax_max_steps = 0, ax_fc = 0, ax_max_per_stratum = 0;
    std::string bodies;  // per axiom: [stratum, parameters, steps, fc, first witness-only step, static domain sizes]
    for (usize si = 0; si < C.strata.size(); ++si)
    {
        const plan::Stratum& st = C.strata[si];
        axioms += static_cast<u32>(st.axioms.size());
        recursive += st.recursive ? 1 : 0;
        ax_max_per_stratum = std::max<u32>(ax_max_per_stratum, static_cast<u32>(st.axioms.size()));
        for (const plan::Axiom& x : st.axioms)
        {
            const plan::Matcher& m = x.body;
            ax_max_steps = std::max<u32>(ax_max_steps, static_cast<u32>(m.steps.size()));
            ax_fc += m.use_fc ? 1 : 0;
            std::string doms;
            for (const plan::Step& sp : m.steps)
            {
                u32 n = 0;
                for (u32 w = 0; w < C.ow; ++w)
                    n += static_cast<u32>(std::popcount(m.dom0[u64{sp.param} * C.ow + w]));
                doms += std::string(doms.empty() ? "" : ",") + std::to_string(n);
            }
            bodies += std::string(bodies.empty() ? "" : ",") + "[" + std::to_string(si) + "," + std::to_string(m.total) + "," +
                      std::to_string(m.steps.size()) + "," + (m.use_fc ? "1" : "0") + "," + std::to_string(m.first_exist) +
                      ",[" + doms + "]]";
        }
    }
    // per schema and witness setting (with, without): the plan's matcher, [use_fc, the fixed order's steps as
    // [parameter, static domain size, backward rows, checks], first witness-only step, forward-checking edges]
    std::string matchers;
    const rl::DeviceSearchCosts costs = rl::device_search_costs(task);
    for (u32 si = 0; si < C.schemas.size(); ++si)
    {
        std::string ms;
        for (u32 wi = 0; wi < 2; ++wi)
        {
            const plan::Matcher& m = C.schemas[si].pre[wi];
            std::string steps;
            for (const plan::Step& sp : m.steps)
            {
                u32 n = 0;
                for (u32 w = 0; w < C.ow; ++w)
                    n += static_cast<u32>(std::popcount(m.dom0[u64{sp.param} * C.ow + w]));
                steps += std::string(steps.empty() ? "" : ",") + "[" + std::to_string(sp.param) + "," + std::to_string(n) + "," +
                         std::to_string(sp.row_end - sp.row_begin) + "," + std::to_string(sp.check_end - sp.check_begin) + "]";
            }
            const rl::SearchCost sc = costs.schemas[si][wi];
            ms += std::string(ms.empty() ? "" : ",") + "[" + (m.use_fc ? "1" : "0") + ",[" + steps + "]," +
                  std::to_string(m.first_exist) + "," + std::to_string(m.fc_out.size()) + ",[" + std::to_string(sc.fixed) +
                  "," + std::to_string(sc.fc) + "]]";
        }
        matchers += std::string(matchers.empty() ? "" : ",") + "\"" + task.schema_name(SchemaId{si}) + "\":[" + ms + "]";
    }
    // the device placement (witness pruning on) and the reasons of the CPU fallback
    std::string placement = "\"host_schemas\":null";
    if (cuda::device_count() > 0 && cuda::ChunkGenerator::unsupported(task).empty())
    {
        auto ctx = cuda::DeviceContext::create(0);
        const cuda::ChunkGenerator gen(ctx, tp);
        const cuda::SchemaPlacement& pl = gen.placement(true);
        std::string why;
        for (u32 s = 0; s < pl.host.size(); ++s)
            if (pl.host[s])
                why += std::string(why.empty() ? "" : ",") + "\"" + task.schema_name(SchemaId{s}) + ": " +
                       cuda::ChunkGenerator::host_reason(task, s, true) + "\"";
        placement = "\"host_schemas\":" + std::to_string(pl.host_count) + ",\"device_ce_schemas\":" + std::to_string(pl.ce_count) +
                    ",\"host_ce_schemas\":" + std::to_string(pl.host_ce_count) + ",\"host_why\":[" + why +
                    "],\"device_axioms\":" + (task.has_axioms() && gen.device_axioms() ? "true" : "false") +
                    ",\"host_axioms_why\":\"" + gen.host_axioms_reason() + "\",\"view_words\":" + std::to_string(gen.view_words()) +
                    ",\"cpu_view_words\":" + std::to_string(u64{C.view.rows} * C.ow) + ",\"flat_strata\":" + std::to_string(gen.flat_strata());
    }
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "\"objects\":%u,\"ow\":%u,\"schemas\":%zu,\"ce_schemas\":%u,\"ces\":%u,\"ce_forall\":%u,\"ce_max_steps\":%u,"
                  "\"ce_max_total\":%u,\"ce_fc\":%u,\"ce_driven\":%u,\"strata\":%zu,\"recursive_strata\":%u,\"axioms\":%u,"
                  "\"axioms_max_per_stratum\":%u,\"axiom_max_steps\":%u,\"axiom_fc\":%u,\"derived_dense\":%llu,"
                  "\"derived_goal\":%s,\"words\":%u,\"max_words\":%u,",
                  C.num_objects, C.ow, C.schemas.size(), ce_schemas, ces, ce_forall, ce_max_steps, ce_max_total, ce_fc, ce_driven,
                  C.strata.size(), recursive, axioms, ax_max_per_stratum, ax_max_steps, ax_fc,
                  static_cast<unsigned long long>(C.layout.total - C.layout.fluent_count), C.goal.uses_derived ? "true" : "false",
                  task.words(), task.max_words());
    std::string axiom_costs;
    for (const rl::SearchCost& sc : costs.axioms)
        axiom_costs += std::string(axiom_costs.empty() ? "" : ",") + "[" + std::to_string(sc.fixed) + "," + std::to_string(sc.fc) + "]";
    return buf + placement + ",\"axiom_bodies\":[" + bodies + "],\"matchers\":{" + matchers + "},\"axiom_costs\":[" +
           axiom_costs + "],\"probe_states\":" + std::to_string(costs.states);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        usage("missing task file");
    std::vector<std::string> files;
    int i = 1;
    for (; i < argc && argv[i][0] != '-'; ++i)
        files.emplace_back(argv[i]);
    if (files.empty() || files.size() > 2)
        usage("give one task.txt or a domain.pddl and a problem.pddl");
    TaskOptions to;
    cuda::DeviceBrfsOptions o;
    std::vector<u32> cpu_threads;
    int reps = 1, cpu_reps = 1;
    bool only_info = false;
    u64 max_bytes = u64{3} << 30;
    for (; i < argc; ++i)
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
        else if (a == "--info")
            only_info = true;
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
        else if (a == "--reps")
            reps = std::stoi(value());
        else if (a == "--cpu")
        {
            const std::string v = value();
            for (usize b = 0; b <= v.size();)
            {
                const usize e = std::min(v.find(',', b), v.size());
                cpu_threads.push_back(static_cast<u32>(std::stoul(v.substr(b, e - b))));
                b = e + 1;
            }
        }
        else if (a == "--cpu-reps")
            cpu_reps = std::stoi(value());
        else if (a == "--timings")
            o.timings = true;
        else if (a == "--view-bytes")
            o.view_bytes = std::stoull(value());
        else if (a == "--max-bytes")
            max_bytes = std::stoull(value());
        else if (a == "--matching")
        {
            const std::string v = value();
            to.matching = v == "fixed" ? TaskOptions::Matching::FixedOrder
                        : v == "fc"    ? TaskOptions::Matching::ForwardChecking
                                       : TaskOptions::Matching::Auto;
        }
        else
            usage(("unknown option " + a).c_str());
    }
    try
    {
        TaskPtr task;
        std::string name;
        if (files.size() == 1)
        {
            task = Task::from_text_file(files[0], to);
            name = base_name(files[0]);
        }
        else
        {
#if MYMYR_DEVICE_CONDITIONAL_FRONTEND
            task = Task::create(*frontend::load_task(files[0], files[1]), to);
            name = base_name(files[0].substr(0, files[0].find_last_of('/'))) + "/" + base_name(files[1]);
#else
            usage("this build has no PDDL front end");
#endif
        }
        std::printf("{\"task\":\"%s\",\"atoms\":\"%s\",%s", name.c_str(), task->atoms().mode() == AtomMode::Frozen ? "frozen" : "lazy",
                    info(task).c_str());
        if (only_info)
        {
            std::printf("}\n");
            return 0;
        }
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
        const BrfsResult& r = d.result;
        const cuda::DeviceBrfsStats& st = d.stats;
        std::printf(",\"witness\":%s,\"canonical\":%s,\"states\":%llu,\"expanded\":%llu,\"generated\":%llu,\"goal_states\":%llu,"
                    "\"layers\":%u,\"exhausted\":%s,\"solved\":%s,\"plan_len\":%zu,\"search_s\":%.4f,\"search_s_all\":[%s],"
                    "\"fp\":\"%016llx\",\"view_ms\":%.2f,\"gen_ms\":%.2f,\"dedup_ms\":%.2f,\"host_ms\":%.2f,\"chunks\":%llu,"
                    "\"uploads\":%u,\"widenings\":%u,\"device_mb\":%.1f,\"run_host_schemas\":%u,\"run_ce_schemas\":%u,"
                    "\"run_host_ce_schemas\":%u,\"run_device_axioms\":%s,\"axiom_ms\":%.2f,\"host_axiom_ms\":%.2f,\"axiom_reruns\":%llu,"
                    "\"groups\":%llu,\"loops\":%llu,\"captures\":%llu,\"resumed\":%llu,\"redone\":%llu",
                    o.witness_pruning ? "true" : "false", o.canonical_order ? "true" : "false",
                    static_cast<unsigned long long>(r.states), static_cast<unsigned long long>(r.expanded),
                    static_cast<unsigned long long>(r.generated), static_cast<unsigned long long>(r.goal_states), r.layers,
                    r.exhausted ? "true" : "false", r.solved ? "true" : "false", r.plan.size(), median(times), join(times).c_str(),
                    static_cast<unsigned long long>(r.fingerprint), st.view_ms, st.gen_ms, st.dedup_ms, st.host_ms,
                    static_cast<unsigned long long>(st.chunks), st.uploads, st.widenings,
                    static_cast<double>(st.device_bytes) / 1048576.0, st.host_schemas, st.ce_schemas, st.host_ce_schemas,
                    st.device_axioms ? "true" : "false", st.axiom_ms, st.host_axiom_ms,
                    static_cast<unsigned long long>(st.axiom_reruns), static_cast<unsigned long long>(st.groups),
                    static_cast<unsigned long long>(st.loops), static_cast<unsigned long long>(st.captures),
                    static_cast<unsigned long long>(st.resumed), static_cast<unsigned long long>(st.redone));
        for (u32 T : cpu_threads)
        {
            BrfsOptions bo;
            bo.threads = T ? T : std::max(1u, std::thread::hardware_concurrency());
            bo.witness_pruning = o.witness_pruning;
            bo.canonical_order = o.canonical_order;
            bo.max_states = o.max_states;
            bo.stop_at_goal = o.stop_at_goal;
            bo.fingerprint = o.fingerprint;
            std::vector<double> ct;
            BrfsResult c;
            for (int k = 0; k < cpu_reps; ++k)
            {
                c = brfs(*task, bo);
                ct.push_back(c.search_s);
            }
            const bool equal = c.states == r.states && c.generated == r.generated && c.goal_states == r.goal_states &&
                               c.layers == r.layers && c.fingerprint == r.fingerprint && c.plan == r.plan;
            std::printf(",\"cpu%u_s\":%.4f,\"cpu%u_s_all\":[%s],\"cpu%u_equal\":%s", bo.threads, median(ct), bo.threads,
                        join(ct).c_str(), bo.threads, equal ? "true" : "false");
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
