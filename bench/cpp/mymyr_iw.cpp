// mymyr_iw: one IW-family search, printing one JSON line with the fork's own fields (status names,
// expanded / generated / generated_in_tree, passes,
// per_pass "arity:expanded/generated/generated_in_tree;...", plan_len, plan_cost (the fork's: total-cost), num_subproblems,
// max_effective_width), plus mymyr's own.
//
//   mymyr_iw task.txt                        --algo iw|siw|iwpass [--k K] [options]
//   mymyr_iw --domain D.pddl --problem P.pddl --algo ...            (when built with the PDDL front end)
//
// options: --atoms auto|lazy|frozen  --match auto|fixed|fc  --width-zero fork|root  --no-opt-iw1  --witness
//          --no-canonical  --max-states N  --max-expanded N  --max-depth D  --timeout-ms T  --dense-mb M  --plan
//          --trace FILE: the event trace of bench/fork_iw_trace (E <sorted atoms> per expansion, G +|- <action> per
//          transition, S <expanded>/<generated>/<in_tree> per pass); runs the observed (slow) path
//          --order-from FILE: test each expanded state's transitions in the order a fork_iw_trace file lists them
//          (IwOptions::successor_order; states missing from the file keep the generation order: order_misses)
// SIW: the totals (expanded, generated, generated_in_tree, passes, num_subproblems) cover the solved subproblems only,
// as the fork's SIW statistics do; all_* fields include a last, failed subproblem.

#include "mymyr/formalism/text_format.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/task/task.hpp"
#if defined(MYMYR_HAVE_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <sys/resource.h>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
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
                 "error: %s\nusage: mymyr_iw (task.txt | --domain D --problem P) --algo iw|siw|iwpass [--k K]\n"
                 "       [--atoms auto|lazy|frozen] [--match auto|fixed|fc] [--width-zero fork|root] [--no-opt-iw1]\n"
                 "       [--witness] [--no-canonical] [--max-states N] [--max-expanded N] [--max-depth D]\n"
                 "       [--timeout-ms T] [--dense-mb M] [--plan]\n",
                 msg.c_str());
    std::exit(2);
}

struct Sums
{
    u64 expanded = 0, generated = 0, in_tree = 0, skipped = 0, passes = 0;
    std::string per_pass;
    std::string list;
    void add(const IwPassStatistics& p, bool record)
    {
        expanded += p.expanded;
        generated += p.generated;
        in_tree += p.generated_in_tree;
        skipped += p.skipped;
        ++passes;
        if (!record)
            return;
        if (!per_pass.empty())
            per_pass += ";";
        per_pass += std::to_string(p.arity) + ":" + std::to_string(p.expanded) + "/" + std::to_string(p.generated) + "/" +
                    std::to_string(p.generated_in_tree);
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "%s{\"arity\":%u,\"status\":\"%s\",\"expanded\":%llu,\"generated\":%llu,\"generated_in_tree\":%llu,"
                      "\"skipped\":%llu,\"blocked\":%llu,\"placeholder\":%s,\"seconds\":%.6f}",
                      list.empty() ? "" : ",", p.arity, to_string(p.status), static_cast<unsigned long long>(p.expanded),
                      static_cast<unsigned long long>(p.generated), static_cast<unsigned long long>(p.generated_in_tree),
                      static_cast<unsigned long long>(p.skipped), static_cast<unsigned long long>(p.blocked),
                      p.placeholder ? "true" : "false", p.seconds);
        list += buf;
    }
};

/// The fork_iw_trace event format (bench/fork_iw_trace/fork_iw_trace.cpp).
class TraceObserver final : public SearchObserver
{
public:
    TraceObserver(const Task& task, std::FILE* f) : m_task(task), m_f(f) {}
    void on_expand(u64, StateView s) override
    {
        std::vector<std::string> atoms = m_task.format_atoms(s);
        std::sort(atoms.begin(), atoms.end());
        std::string line = "E";
        for (const std::string& a : atoms)
            line += " " + a;
        std::fprintf(m_f, "%s\n", line.c_str());
    }
    void on_generate(u64, const Action& a, u64, StateView, bool is_new) override
    {
        std::fprintf(m_f, "G %c %s\n", is_new ? '+' : '-', m_task.format(a.label()).c_str());
    }
    void on_pass(u32, const SearchStatistics& s) override
    {
        std::fprintf(m_f, "S %llu/%llu/%llu\n", static_cast<unsigned long long>(s.expanded), static_cast<unsigned long long>(s.generated),
                     static_cast<unsigned long long>(s.generated - s.pruned));
    }

private:
    const Task& m_task;
    std::FILE* m_f;
};

u64 fnv1a(std::string_view s)
{
    u64 h = 0xcbf29ce484222325ULL;
    for (unsigned char ch : s)
        h = (h ^ ch) * 0x100000001b3ULL;
    return h;
}

/// The successor order of a fork_iw_trace file: per expanded state (hash of its sorted atom list), the hashes of its
/// actions in the fork's generation order.
class TraceOrder
{
public:
    TraceOrder(const Task& task, const std::string& path) : m_task(task)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("cannot open " + path);
        std::vector<u64>* cur = nullptr;
        std::string line;
        while (std::getline(in, line))
        {
            std::string_view v(line);
            while (!v.empty() && (v.back() == '\n' || v.back() == '\r'))
                v.remove_suffix(1);
            if (!v.empty() && v[0] == 'E' && (v.size() == 1 || v[1] == ' '))
            {
                const auto [it, fresh] = m_order.try_emplace(fnv1a(v.size() > 2 ? v.substr(2) : std::string_view{}));
                cur = fresh ? &it->second : nullptr;  // the first expansion of a state defines its order
            }
            else if (v.size() > 4 && v[0] == 'G' && cur)
                cur->push_back(fnv1a(v.substr(4)));
        }
    }

    void operator()(StateView s, std::span<const Action> actions, std::vector<u32>& order)
    {
        std::vector<std::string> atoms = m_task.format_atoms(s);
        std::sort(atoms.begin(), atoms.end());
        std::string key;
        for (const std::string& a : atoms)
            key += (key.empty() ? "" : " ") + a;
        const auto it = m_order.find(fnv1a(key));
        if (it == m_order.end())
        {
            ++state_misses;
            return;
        }
        std::unordered_map<u64, u32> pos;
        for (u32 i = 0; i < it->second.size(); ++i)
            pos.emplace(it->second[i], i);
        std::vector<std::pair<u64, u32>> keyed;
        for (u32 i = 0; i < actions.size(); ++i)
        {
            const auto p = pos.find(fnv1a(m_task.format(actions[i].label())));
            if (p == pos.end())
                ++action_misses;
            keyed.emplace_back(p == pos.end() ? (u64{1} << 32) + i : p->second, i);
        }
        std::sort(keyed.begin(), keyed.end());
        for (const auto& [k, i] : keyed)
            order.push_back(i);
    }

    u64 state_misses = 0, action_misses = 0;

private:
    const Task& m_task;
    std::unordered_map<u64, std::vector<u64>> m_order;
};

std::string plan_json(const Task& task, const std::vector<Action>& plan)
{
    std::string s = "[";
    for (usize i = 0; i < plan.size(); ++i)
    {
        std::string a = task.format(plan[i].label());
        std::string e;
        for (char ch : a)
            e += ch == '"' ? std::string("\\\"") : std::string(1, ch);
        s += (i ? ",\"" : "\"") + e + "\"";
    }
    return s + "]";
}
}  // namespace

int main(int argc, char** argv)
{
    std::string task_file, domain, problem, algo = "iw";
    u32 k = 1;
    bool print_plan = false;
    std::string trace_file, order_file;
    TaskOptions to;
    IwOptions io;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage("missing value for " + a);
            return argv[++i];
        };
        if (a == "--algo")
            algo = value();
        else if (a == "--k")
            k = static_cast<u32>(std::stoul(value()));
        else if (a == "--domain")
            domain = value();
        else if (a == "--problem")
            problem = value();
        else if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else if (a == "--match")
        {
            const std::string v = value();
            to.matching = v == "fixed" ? TaskOptions::Matching::FixedOrder
                          : v == "fc"  ? TaskOptions::Matching::ForwardChecking
                                       : TaskOptions::Matching::Auto;
        }
        else if (a == "--width-zero")
            io.width_zero = value() == "root" ? WidthZero::RootOnly : WidthZero::ExpandDepthOne;
        else if (a == "--no-opt-iw1")
            io.optimize_iw1 = false;
        else if (a == "--witness")
            io.witness_pruning = true;
        else if (a == "--no-canonical")
            io.canonical_order = false;
        else if (a == "--max-states")
            io.control.budget.max_states = std::stoull(value());
        else if (a == "--max-expanded")
            io.control.budget.max_expanded = std::stoull(value());
        else if (a == "--max-depth")
            io.control.budget.max_depth = static_cast<u32>(std::stoul(value()));
        else if (a == "--timeout-ms")
            io.control.budget.max_seconds = std::stod(value()) / 1000.0;
        else if (a == "--dense-mb")
            io.tables.max_dense_bytes = static_cast<u64>(std::stod(value()) * 1048576.0);
        else if (a == "--plan")
            print_plan = true;
        else if (a == "--trace")
            trace_file = value();
        else if (a == "--order-from")
            order_file = value();
        else if (!a.empty() && a[0] != '-' && task_file.empty())
            task_file = a;
        else
            usage("unknown argument " + a);
    }
    if (task_file.empty() && (domain.empty() || problem.empty()))
        usage("missing task (task.txt or --domain/--problem)");
    io.max_arity = k;
    try
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<const Task> task;
        if (!task_file.empty())
            task = Task::from_text_file(task_file, to);
        else
        {
#if defined(MYMYR_HAVE_FRONTEND)
            const auto data = frontend::load_task(domain, problem);
            task = Task::create(*data, to);
#else
            usage("built without the PDDL front end: pass an exported task.txt");
#endif
        }
        const double prep = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::FILE* trace = nullptr;
        std::optional<TraceObserver> tracer;
        if (!trace_file.empty())
        {
            trace = std::fopen(trace_file.c_str(), "w");
            if (!trace)
                usage("cannot open " + trace_file);
            tracer.emplace(*task, trace);
            io.control.observer = &*tracer;
        }
        std::shared_ptr<TraceOrder> replay;
        if (!order_file.empty())
        {
            replay = std::make_shared<TraceOrder>(*task, order_file);
            io.successor_order = [replay](StateView st, std::span<const Action> acts, std::vector<u32>& ord) { (*replay)(st, acts, ord); };
        }

        Sums fork_sums, all_sums;
        SearchStatus status;
        std::string extra, message;
        std::vector<Action> plan;
        double cost = 0;
        bool cost_exact = true;
        u32 fluent_slots = 0;
        u64 table_bytes = 0, node_bytes = 0;
        const auto ts = std::chrono::steady_clock::now();
        if (algo == "iw" || algo == "iwpass")
        {
            const IwResult r = algo == "iw" ? iw(*task, io) : iw_pass(*task, k, io);
            status = r.status;
            for (const IwPassStatistics& p : r.passes)
            {
                fork_sums.add(p, true);
                all_sums.add(p, false);
            }
            plan = r.plan;
            cost = r.cost;
            cost_exact = r.cost_exact;
            message = r.message;
            fluent_slots = r.fluent_slots;
            table_bytes = r.peak_table_bytes;
            node_bytes = r.peak_node_bytes;
            extra += ",\"effective_width\":" + std::to_string(r.effective_width);
        }
        else if (algo == "siw")
        {
            const SiwResult r = siw(*task, io);
            status = r.status;
            u32 solved = 0;
            for (const SiwSubproblem& sub : r.subproblems)
            {
                for (const IwPassStatistics& p : sub.passes)
                {
                    if (sub.status == SearchStatus::Solved)
                        fork_sums.add(p, false);
                    all_sums.add(p, false);
                }
                solved += sub.status == SearchStatus::Solved;
            }
            plan = r.plan;
            cost = r.cost;
            cost_exact = r.cost_exact;
            message = r.message;
            fluent_slots = r.fluent_slots;
            table_bytes = r.peak_table_bytes;
            node_bytes = r.peak_node_bytes;
            extra += ",\"num_subproblems\":" + std::to_string(solved) + ",\"all_subproblems\":" + std::to_string(r.subproblems.size()) +
                     ",\"max_effective_width\":" + std::to_string(solved ? static_cast<int>(r.max_effective_width) : -1);
        }
        else
            usage("unknown algo " + algo);
        const double search_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        if (trace)
            std::fclose(trace);

        std::printf("{\"lib\":\"mymyr\",\"algo\":\"%s\",\"k\":%u,\"status\":\"%s\",\"mymyr_status\":\"%s\",\"prep_s\":%.6f,"
                    "\"search_s\":%.6f,\"expanded\":%llu,\"generated\":%llu,\"generated_in_tree\":%llu,\"skipped\":%llu,"
                    "\"passes\":%llu,\"per_pass\":\"%s\",\"pass_list\":[%s],\"all_expanded\":%llu,\"all_generated\":%llu,"
                    "\"all_generated_in_tree\":%llu%s",
                    algo.c_str(), k, mimir_status_name(status), to_string(status), prep, search_s,
                    static_cast<unsigned long long>(fork_sums.expanded), static_cast<unsigned long long>(fork_sums.generated),
                    static_cast<unsigned long long>(fork_sums.in_tree), static_cast<unsigned long long>(fork_sums.skipped),
                    static_cast<unsigned long long>(fork_sums.passes), fork_sums.per_pass.c_str(), fork_sums.list.c_str(),
                    static_cast<unsigned long long>(all_sums.expanded), static_cast<unsigned long long>(all_sums.generated),
                    static_cast<unsigned long long>(all_sums.in_tree), extra.c_str());
        if (status == SearchStatus::Solved)
            std::printf(",\"plan_len\":%zu,\"plan_cost\":%.17g,\"cost_exact\":%s", plan.size(), cost, cost_exact ? "true" : "false");
        if (print_plan)
            std::printf(",\"plan\":%s", plan_json(*task, plan).c_str());
        if (!message.empty())
            std::printf(",\"message\":\"%s\"", message.c_str());
        if (replay)
            std::printf(",\"order_misses\":{\"states\":%llu,\"actions\":%llu}", static_cast<unsigned long long>(replay->state_misses),
                        static_cast<unsigned long long>(replay->action_misses));
        std::printf(",\"atoms\":\"%s\",\"F\":%u,\"dense_F\":%llu,\"witness\":%s,\"canonical\":%s,\"table_mb\":%.2f,\"tree_mb\":%.2f,"
                    "\"peak_rss_mb\":%.1f}\n",
                    task->atoms().mode() == AtomMode::Frozen ? "frozen" : "lazy", fluent_slots,
                    static_cast<unsigned long long>(task->info().dense_fluent), io.witness_pruning ? "true" : "false",
                    io.canonical_order ? "true" : "false", static_cast<double>(table_bytes) / 1048576.0,
                    static_cast<double>(node_bytes) / 1048576.0, peak_rss_mb());
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
