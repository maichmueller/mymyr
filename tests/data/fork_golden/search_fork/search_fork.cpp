// search_fork: one search of the mimir fork (0.16.x) on a PDDL task, for the expectations of mymyr's parity tests:
// the numeric best-first test (tests/data/numeric_tasks/fork_best_first.json, made by run_numeric.py) and the layer
// ordering test (tests/data/layer_orders/fork_layer_orders.json, made by run_layer_orders.py).
//
//   search_fork --algo astar_eager|astar_lazy|gbfs_eager|gbfs_lazy --h blind|max|add|ff --domain D --problem P
//               [--max-ms T] [--max-states N]
//   search_fork --algo iw|brfs --order in_order|reverse|goal_count|goal_count_fewer [--k K] [--limit L]
//               --domain D --problem P [--max-ms T] [--max-states N]
//
// Prints one line "RESULT {...}" with status, plan_cost, plan_length, plan (ground action strings), expanded and
// generated (best-first: also deadends; iw: also per-pass statistics), or "ERROR <message>". Successor generation is
// lifted KPKC with symmetry pruning off; h_max, h_add and h_FF are the fork's grounded heuristics over a
// LiftedGrounder (unit action costs), blind is its BlindHeuristic; the searches use the fork's default event handlers
// and strategies. iw is iw::find_solution with max_arity K and the layer ordering strategy (max_next_layer_states L);
// brfs is brfs::find_solution with stop_if_goal and the same ordering.

#include <mimir/mimir.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mimir;
using namespace mimir::search;
using namespace mimir::formalism;

namespace
{
const char* status_name(SearchStatus s)
{
    switch (s)
    {
        case SearchStatus::IN_PROGRESS: return "in_progress";
        case SearchStatus::OUT_OF_TIME: return "out_of_time";
        case SearchStatus::OUT_OF_MEMORY: return "out_of_memory";
        case SearchStatus::OUT_OF_STATES: return "out_of_states";
        case SearchStatus::FAILED: return "failed";
        case SearchStatus::EXHAUSTED: return "exhausted";
        case SearchStatus::SOLVED: return "solved";
        case SearchStatus::UNSOLVABLE: return "unsolvable";
        case SearchStatus::CANCELED: return "canceled";
    }
    return "unknown";
}

std::string jstr(const std::string& s)
{
    std::string o = "\"";
    for (unsigned char c : s)
    {
        if (c == '"' || c == '\\')
        {
            o += '\\';
            o += static_cast<char>(c);
        }
        else if (c < 0x20)
        {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        }
        else
            o += static_cast<char>(c);
    }
    return o + "\"";
}

std::string jnum(double v)
{
    if (!std::isfinite(v))
        return "null";
    std::ostringstream s;
    s.precision(17);
    s << v;
    return s.str();
}

std::string action_str(GroundAction a)
{
    std::string s = "(" + a->get_action()->get_name();
    for (auto o : a->get_objects())
        s += " " + o->get_name();
    return s + ")";
}

template<typename Options, typename Handler, typename Find>
std::string run(const SearchContext& context, const Heuristic& h, Handler eh, uint32_t max_ms, uint32_t max_states, Find find)
{
    Options opts;
    opts.event_handler = eh;
    opts.max_time_in_ms = max_ms;
    opts.max_num_states = max_states;
    auto result = find(context, h, opts);
    const auto& st = eh->get_statistics();
    std::string o = "\"status\":" + jstr(status_name(result.status));
    if (result.plan)
    {
        o += ",\"plan_cost\":" + jnum(result.plan->get_cost()) + ",\"plan_length\":" + std::to_string(result.plan->get_actions().size());
        o += ",\"plan\":[";
        bool first = true;
        for (const auto& a : result.plan->get_actions())
        {
            o += (first ? "" : ",") + jstr(action_str(a));
            first = false;
        }
        o += "]";
    }
    else
        o += ",\"plan_cost\":null,\"plan_length\":null,\"plan\":null";
    o += ",\"expanded\":" + std::to_string(st.get_num_expanded()) + ",\"generated\":" + std::to_string(st.get_num_generated()) +
         ",\"deadends\":" + std::to_string(st.get_num_deadends());
    return o;
}

LayerOrderingStrategy make_order(const Problem& problem, const std::string& order)
{
    if (order == "in_order")
        return InOrderLayerOrderingStrategyImpl::create();
    if (order == "reverse")
        return ReverseOrderLayerOrderingStrategyImpl::create();
    if (order == "goal_count")
        return GoalCountLayerOrderingStrategyImpl::create(problem, true);
    if (order == "goal_count_fewer")
        return GoalCountLayerOrderingStrategyImpl::create(problem, false);
    throw std::invalid_argument("unknown order " + order);
}

std::string plan_json(const SearchResult& result)
{
    if (!result.plan)
        return ",\"plan_cost\":null,\"plan_length\":null,\"plan\":null";
    std::string o = ",\"plan_cost\":" + jnum(result.plan->get_cost()) + ",\"plan_length\":" + std::to_string(result.plan->get_actions().size());
    o += ",\"plan\":[";
    bool first = true;
    for (const auto& a : result.plan->get_actions())
    {
        o += (first ? "" : ",") + jstr(action_str(a));
        first = false;
    }
    return o + "]";
}

int run_layered(const std::string& domain, const std::string& problem_file, const std::string& algo, const std::string& order, size_t k,
                uint32_t limit, uint32_t max_ms, uint32_t max_states)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    std::string body;
    if (algo == "iw")
    {
        auto iw_eh = iw::DefaultEventHandlerImpl::create(problem, true);
        auto brfs_eh = brfs::DefaultEventHandlerImpl::create(problem, true);
        auto opts = iw::Options();
        opts.max_arity = k;
        opts.iw_event_handler = iw_eh;
        opts.brfs_event_handler = brfs_eh;
        opts.layer_ordering_strategy = make_order(problem, order);
        opts.max_next_layer_states = limit;
        opts.max_time_in_ms = max_ms;
        opts.max_num_states = max_states;
        auto result = iw::find_solution(context, opts);
        body = "\"status\":" + jstr(status_name(result.status)) + plan_json(result) + ",\"passes\":[";
        uint64_t exp = 0, gen = 0;
        const auto& by_arity = iw_eh->get_statistics().get_brfs_statistics_by_arity();
        for (size_t a = 0; a < by_arity.size(); ++a)
        {
            const auto& st = by_arity[a];
            body += std::string(a ? "," : "") + "{\"arity\":" + std::to_string(a) + ",\"expanded\":" + std::to_string(st.get_num_expanded())
                    + ",\"generated\":" + std::to_string(st.get_num_generated())
                    + ",\"generated_in_tree\":" + std::to_string(st.get_num_generated_in_search_tree()) + "}";
            exp += st.get_num_expanded();
            gen += st.get_num_generated();
        }
        body += "],\"expanded\":" + std::to_string(exp) + ",\"generated\":" + std::to_string(gen);
    }
    else if (algo == "brfs")
    {
        auto eh = brfs::DefaultEventHandlerImpl::create(problem, true);
        auto opts = brfs::Options();
        opts.event_handler = eh;
        opts.stop_if_goal = true;
        opts.layer_ordering_strategy = make_order(problem, order);
        opts.max_next_layer_states = limit;
        opts.max_time_in_ms = max_ms;
        opts.max_num_states = max_states;
        auto result = brfs::find_solution(context, opts);
        const auto& st = eh->get_statistics();
        body = "\"status\":" + jstr(status_name(result.status)) + plan_json(result) + ",\"expanded\":" + std::to_string(st.get_num_expanded())
               + ",\"generated\":" + std::to_string(st.get_num_generated());
    }
    else
    {
        std::cerr << "unknown algo " << algo << "\n";
        return 2;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "RESULT {\"algo\":" << jstr(algo) << ",\"order\":" << jstr(order) << ",\"k\":" << k << ",\"limit\":"
              << (limit == UINT32_MAX ? std::string("null") : std::to_string(limit)) << "," << body << ",\"seconds\":" << jnum(secs) << "}"
              << std::endl;
    return 0;
}

int run_search(const std::string& domain, const std::string& problem_file, const std::string& algo, const std::string& hname, uint32_t max_ms,
           uint32_t max_states)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    std::unique_ptr<LiftedGrounder> grounder;
    Heuristic h;
    if (hname == "blind")
        h = BlindHeuristicImpl::create(problem);
    else
    {
        grounder = std::make_unique<LiftedGrounder>(problem);
        if (hname == "max")
            h = MaxHeuristicImpl::create(*grounder);
        else if (hname == "add")
            h = AddHeuristicImpl::create(*grounder);
        else if (hname == "ff")
            h = FFHeuristicImpl::create(*grounder);
        else
        {
            std::cerr << "unknown heuristic " << hname << "\n";
            return 2;
        }
    }
    std::string body;
    if (algo == "astar_eager")
        body = run<astar_eager::Options>(context, h, astar_eager::DefaultEventHandlerImpl::create(problem, true), max_ms, max_states,
                                         [](auto&& c, auto&& hh, auto&& o) { return astar_eager::find_solution(c, hh, o); });
    else if (algo == "astar_lazy")
        body = run<astar_lazy::Options>(context, h, astar_lazy::DefaultEventHandlerImpl::create(problem, true), max_ms, max_states,
                                        [](auto&& c, auto&& hh, auto&& o) { return astar_lazy::find_solution(c, hh, o); });
    else if (algo == "gbfs_eager")
        body = run<gbfs_eager::Options>(context, h, gbfs_eager::DefaultEventHandlerImpl::create(problem, true), max_ms, max_states,
                                        [](auto&& c, auto&& hh, auto&& o) { return gbfs_eager::find_solution(c, hh, o); });
    else if (algo == "gbfs_lazy")
        body = run<gbfs_lazy::Options>(context, h, gbfs_lazy::DefaultEventHandlerImpl::create(problem, true), max_ms, max_states,
                                       [](auto&& c, auto&& hh, auto&& o) { return gbfs_lazy::find_solution(c, hh, o); });
    else
    {
        std::cerr << "unknown algo " << algo << "\n";
        return 2;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "RESULT {\"algo\":" << jstr(algo) << ",\"h\":" << jstr(hname) << "," << body << ",\"seconds\":" << jnum(secs) << "}"
              << std::endl;
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string algo, hname, domain, problem, order;
    uint32_t max_ms = 120000, max_states = UINT32_MAX, limit = UINT32_MAX;
    size_t k = 1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (i + 1 >= argc)
        {
            std::cerr << "missing value for " << a << "\n";
            return 2;
        }
        const std::string v = argv[++i];
        if (a == "--algo")
            algo = v;
        else if (a == "--h")
            hname = v;
        else if (a == "--domain")
            domain = v;
        else if (a == "--problem")
            problem = v;
        else if (a == "--max-ms")
            max_ms = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--max-states")
            max_states = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--order")
            order = v;
        else if (a == "--k")
            k = std::stoul(v);
        else if (a == "--limit")
            limit = static_cast<uint32_t>(std::stoul(v));
        else
        {
            std::cerr << "unknown argument " << a << "\n";
            return 2;
        }
    }
    const bool layered = algo == "iw" || algo == "brfs";
    if (algo.empty() || domain.empty() || problem.empty() || (layered ? order.empty() : hname.empty()))
    {
        std::cerr << "usage: search_fork --algo A (--h H | --order O [--k K] [--limit L]) --domain D --problem P [--max-ms T] "
                     "[--max-states N]\n";
        return 2;
    }
    int rc = 2;
    try
    {
        rc = layered ? run_layered(domain, problem, algo, order, k, limit, max_ms, max_states)
                     : run_search(domain, problem, algo, hname, max_ms, max_states);
    }
    catch (const std::exception& ex)
    {
        std::cout << "ERROR " << jstr(ex.what()) << std::endl;
        std::_Exit(1);
    }
    std::cout.flush();
    std::_Exit(rc);  // skips the teardown of large repositories
}
