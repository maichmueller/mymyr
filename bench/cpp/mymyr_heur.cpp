// mymyr_heur: heuristic searches and heuristic evaluation, printing one JSON line with the fields of the fork tool
// the same fields as the fork's own heuristic tool (lib, algo, h, status, plan_cost, plan_length, expanded, generated,
// deadends, parse_s, setup_s, search_s, states, peak_rss_mb) and of the probe heur_probe (hcosts, impl, evals,
// ground_ops, ground_s, expanded_per_s, eval_us), plus mymyr's own.
//
//   mymyr_heur task.txt                  --algo ALGO [options]
//   mymyr_heur --domain D --problem P    --algo ALGO [options]      (when built with the PDDL front end)
//
// ALGO: astar (eager), astar_lazy, gbfs (eager), gbfs_lazy, beam: one search;
//       h0: h of the initial state for every heuristic, grounded and lifted;
//       evalbench: evaluation cost of --h over the first --samples states of a BrFS, grounded and lifted.
// options: --h blind|gc|max|add|ff|set_additive|h2 (default: max for astar*, ff otherwise)  --hcosts unit|real
//          --impl auto|grounded|lifted  --store auto|flat|chunked|compact  --queue auto|bucket|heap
//          --timeout S  --max-states N  --max-expanded N  --beam-width B  --no-requeue  --no-preferred
//          --witness  --atoms auto|lazy|frozen  --samples N  --instance NAME  --plan (plan lines on stderr)

#include "mymyr/formalism/text_format.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

double peak_rss_mb()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<double>(ru.ru_maxrss) / 1048576.0;
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

[[noreturn]] void usage(const std::string& msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_heur (task.txt | --domain D --problem P) --algo "
                 "astar|astar_lazy|gbfs|gbfs_lazy|beam|h0|evalbench\n"
                 "       [--h blind|gc|max|add|ff|set_additive|h2] [--hcosts unit|real] [--impl auto|grounded|lifted]\n"
                 "       [--store auto|flat|chunked|compact] [--queue auto|bucket|heap] [--timeout S] [--max-states N]\n"
                 "       [--max-expanded N] [--beam-width B] [--no-requeue] [--no-preferred] [--witness]\n"
                 "       [--atoms auto|lazy|frozen] [--samples N] [--instance NAME] [--plan]\n",
                 msg.c_str());
    std::exit(2);
}

std::string jstr(const std::string& s)
{
    std::string o = "\"";
    for (char ch : s)
    {
        if (ch == '"' || ch == '\\')
            o += '\\';
        if (static_cast<unsigned char>(ch) < 0x20)
            continue;
        o += ch;
    }
    return o + "\"";
}

std::string jnum(double v)
{
    if (!std::isfinite(v))
        return v > 0 ? "\"inf\"" : "null";
    std::ostringstream o;
    o.precision(10);
    o << v;
    return o.str();
}

/// The fork tool's status names.
const char* status_name(SearchStatus s) { return to_string(s); }

struct Json
{
    std::string body;
    void add(const char* k, const std::string& raw) { body += (body.empty() ? "" : ", ") + std::string("\"") + k + "\": " + raw; }
    void str(const char* k, const std::string& v) { add(k, jstr(v)); }
    void num(const char* k, double v) { add(k, jnum(v)); }
    void u(const char* k, u64 v) { add(k, std::to_string(v)); }
    void print() const { std::printf("{%s}\n", body.c_str()); }
};

/// The first `n` states of a BrFS from the initial state (the probe's evalbench samples).
std::vector<State> brfs_sample(const Task& task, u32 n)
{
    Successors& succ = task.workspace().successors();
    FlatStateStore store(std::max<u32>(1, task.words()));
    const State s0 = task.initial_state();
    store.insert(s0.data(), s0.size_words());
    std::vector<u64> cur, next;
    for (u32 id = 0; id < store.size() && store.size() < n; ++id)
    {
        const u64* rec = store.words(StateId{id});
        cur.assign(rec, rec + store.stride());
        const u32 cn = bits::trimmed_size(cur.data(), store.stride());
        succ.prepare({cur.data(), cn, nullptr, 0});
        succ.generate<true>(
            [&](u32, const ObjectId*, const Delta& d)
            {
                const u32 nn = apply_delta(cur.data(), cn, d, next);
                store.insert(next.data(), nn);
                return store.size() < n;
            },
            true, true);
    }
    std::vector<State> out;
    for (u32 i = 0; i < store.size(); ++i)
        out.push_back(store.state(StateId{i}));
    return out;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string task_file, domain, problem, algo, hname, instance;
    std::string hcosts = "unit", impl = "auto", store = "auto", queue = "auto";
    double timeout = std::numeric_limits<double>::infinity();
    u64 max_states = std::numeric_limits<u64>::max(), max_expanded = std::numeric_limits<u64>::max();
    u32 beam_width = 1000, samples = 2000;
    bool requeue = true, preferred = true, witness = false, print_plan = false;
    TaskOptions to;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage("missing value for " + a);
            return argv[++i];
        };
        if (a == "--domain")
            domain = value();
        else if (a == "--problem")
            problem = value();
        else if (a == "--algo")
            algo = value();
        else if (a == "--h")
            hname = value();
        else if (a == "--hcosts")
            hcosts = value();
        else if (a == "--impl")
            impl = value();
        else if (a == "--store")
            store = value();
        else if (a == "--queue")
            queue = value();
        else if (a == "--timeout")
            timeout = std::stod(value());
        else if (a == "--max-states")
            max_states = std::stoull(value());
        else if (a == "--max-expanded")
            max_expanded = std::stoull(value());
        else if (a == "--beam-width")
            beam_width = static_cast<u32>(std::stoul(value()));
        else if (a == "--samples")
            samples = static_cast<u32>(std::stoul(value()));
        else if (a == "--instance")
            instance = value();
        else if (a == "--no-requeue")
            requeue = false;
        else if (a == "--no-preferred")
            preferred = false;
        else if (a == "--witness")
            witness = true;
        else if (a == "--plan")
            print_plan = true;
        else if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else if (!a.empty() && a[0] != '-' && task_file.empty())
            task_file = a;
        else
            usage("unknown argument " + a);
    }
    if (algo.empty())
        usage("missing --algo");
    if (hname.empty())
        hname = algo.rfind("astar", 0) == 0 ? "max" : "ff";
    if (instance.empty())
        instance = !problem.empty() ? problem : task_file;

    Json j;
    j.str("lib", "mymyr");
    j.str("instance", instance);
    j.str("algo", algo);
    j.str("h", hname);
    j.str("hcosts", hcosts);
    j.str("impl", impl);
    try
    {
        const auto t_parse = Clock::now();
        std::shared_ptr<const Task> task;
        if (!task_file.empty())
            task = Task::from_text_file(task_file, to);
        else
        {
#if defined(MYMYR_HAS_FRONTEND)
            if (domain.empty() || problem.empty())
                usage("need a task file or --domain and --problem");
            const auto data = frontend::load_task(domain, problem);
            task = Task::create(*data, to);
#else
            usage("built without the PDDL front end: pass an exported task file");
#endif
        }
        const double parse_s = seconds_since(t_parse);
        j.num("parse_s", parse_s);

        heuristics::Options ho;
        ho.kind = heuristics::parse_kind(hname);
        ho.costs = hcosts == "real" ? heuristics::Costs::Real : heuristics::Costs::Unit;
        ho.evaluation = impl == "grounded" ? heuristics::Evaluation::Grounded
                        : impl == "lifted" ? heuristics::Evaluation::Lifted
                                           : heuristics::Evaluation::Auto;

        if (algo == "h0")
        {
            const State s0 = task->initial_state();
            for (heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::GoalCount, heuristics::Kind::Max,
                                       heuristics::Kind::Add, heuristics::Kind::FF})
            {
                for (const auto ev : {heuristics::Evaluation::Auto, heuristics::Evaluation::Lifted})
                {
                    heuristics::Options o = ho;
                    o.kind = k;
                    o.evaluation = ev;
                    auto h = heuristics::make_heuristic(*task, o);
                    const auto t0 = Clock::now();
                    const double v = h->evaluate(s0);
                    const std::string key = std::string(heuristics::to_string(k)) + (ev == heuristics::Evaluation::Lifted ? "_lifted" : "");
                    j.num(key.c_str(), v);
                    j.num((key + "_s").c_str(), seconds_since(t0));
                    if (auto R = h->relaxed(); R && ev == heuristics::Evaluation::Auto && k == heuristics::Kind::Max)
                    {
                        j.u("ground_ops", R->num_ops());
                        j.u("ground_actions", R->num_ground_actions());
                        j.u("reached_atoms", R->stats().reached_atoms);
                        j.num("ground_s", R->stats().seconds);
                    }
                }
            }
            j.num("peak_rss_mb", peak_rss_mb());
            j.print();
            return 0;
        }
        if (algo == "evalbench")
        {
            const std::vector<State> states = brfs_sample(*task, samples);
            j.u("samples", states.size());
            for (const auto ev : {heuristics::Evaluation::Auto, heuristics::Evaluation::Lifted})
            {
                if (ev == heuristics::Evaluation::Lifted && (ho.kind == heuristics::Kind::SetAdditive || ho.kind == heuristics::Kind::H2))
                    continue;  // grounded only
                heuristics::Options o = ho;
                o.evaluation = ev;
                auto h = heuristics::make_heuristic(*task, o);
                const auto tg = Clock::now();
                double sum = h->evaluate(states.front());  // grounds (Auto)
                const double first_s = seconds_since(tg);
                const auto t0 = Clock::now();
                for (const State& s : states)
                    sum += std::isfinite(h->evaluate(s)) ? 1 : 0;
                const double us = seconds_since(t0) * 1e6 / static_cast<double>(states.size());
                const std::string key = ev == heuristics::Evaluation::Lifted ? "lifted" : "grounded";
                j.num((key + "_eval_us").c_str(), us);
                j.num((key + "_first_s").c_str(), first_s);
                if (auto R = h->relaxed())
                {
                    j.u("ground_ops", R->num_ops());
                    j.num("ground_s", R->stats().seconds);
                }
                (void)sum;
            }
            j.num("peak_rss_mb", peak_rss_mb());
            j.print();
            return 0;
        }

        BestFirstOptions o;
        o.heuristic = ho;
        o.control.budget.max_seconds = timeout;
        o.control.budget.max_states = max_states;
        o.control.budget.max_expanded = max_expanded;
        o.store = store == "flat"      ? BestFirstOptions::Store::Flat
                  : store == "chunked" ? BestFirstOptions::Store::Chunked
                  : store == "compact" ? BestFirstOptions::Store::Compact
                                       : BestFirstOptions::Store::Auto;
        o.queue = queue == "bucket" ? BestFirstOptions::Queue::Bucket
                  : queue == "heap" ? BestFirstOptions::Queue::Heap
                                    : BestFirstOptions::Queue::Auto;
        o.witness_pruning = witness;
        o.lazy_requeue = requeue;
        o.preferred_operators = preferred;
        o.beam_width = beam_width;
        BestFirstResult r;
        if (algo == "astar" || algo == "astar_eager")
            r = astar_eager(*task, o);
        else if (algo == "astar_lazy")
            r = astar_lazy(*task, o);
        else if (algo == "gbfs" || algo == "gbfs_eager")
            r = gbfs_eager(*task, o);
        else if (algo == "gbfs_lazy")
            r = gbfs_lazy(*task, o);
        else if (algo == "beam")
            r = beam(*task, o);
        else
            usage("unknown --algo " + algo);
        const bool solved = r.status == SearchStatus::Solved;
        j.str("status", status_name(r.status));
        j.add("plan_cost", solved ? jnum(r.cost) : "null");
        j.add("plan_length", solved ? std::to_string(r.plan.size()) : "null");
        j.u("expanded", r.stats.expanded);
        j.u("generated", r.stats.generated);
        j.u("states", r.stats.states);
        j.u("deadends", r.dead_ends);
        j.u("pruned", r.stats.pruned);
        j.u("evals", r.evaluations);
        j.u("reopened", r.reopened);
        if (algo == "beam")
            j.u("layers", r.layers);
        j.num("initial_h", r.initial_h);
        j.num("setup_s", r.setup_seconds);
        j.num("ground_s", r.heuristic.grounding_seconds);
        j.num("search_s", r.stats.seconds);
        j.num("expanded_per_s", r.stats.seconds > 0 ? std::round(static_cast<double>(r.stats.expanded) / r.stats.seconds) : 0);
        j.u("h_grounded", r.heuristic.grounded);
        j.u("h_lifted", r.heuristic.lifted);
        j.str("store", r.store);
        j.str("queue", r.queue);
        j.num("store_mb", static_cast<double>(r.store_bytes) / 1048576.0);
        j.u("fluent_slots", r.fluent_slots);
        if (!r.message.empty())
            j.str("message", r.message);
        j.num("peak_rss_mb", peak_rss_mb());
        j.print();
        if (print_plan && solved)
            for (const Action& a : r.plan)
                std::fprintf(stderr, "%s\n", task->format(a.label()).c_str());
        return solved || r.status == SearchStatus::Exhausted || r.status == SearchStatus::Unsolvable ||
                       r.status == SearchStatus::OutOfTime || r.status == SearchStatus::OutOfStates
                   ? 0
                   : 1;
    }
    catch (const std::exception& e)
    {
        j.str("status", "error");
        j.str("message", e.what());
        j.print();
        return 1;
    }
}
