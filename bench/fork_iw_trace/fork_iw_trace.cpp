/*
 * fork_iw_trace: the event trace of the mimir fork (v0.16.3) IW ladder on ONE task, for parity checks.
 * Built out of source against the fork install (see CMakeLists.txt); the fork's source
 * tree is never touched. Successor generation is lifted KPKC with symmetry pruning off, matching the fork.
 *
 *   fork_iw_trace --domain D --problem P --k K [--algo iw|siw] [--timeout-ms T]
 *
 * One line per event of every BrFS pass that iw::find_solution (or siw::find_solution: every pass of every
 * subproblem) runs (quiet IW/SIW handlers, tracing BrFS handler):
 *   P                     a pass starts (the optimized IW(1) placeholder pass runs no BrFS and prints nothing)
 *   E <atoms>             a state is expanded (its fluent atoms, sorted, space separated)
 *   X <atoms>             a popped state is a goal
 *   G + <action>          a generated transition enters the search tree
 *   G - <action>          a generated transition does not
 *   S <expanded>/<generated>/<in_tree>   after the search, per pass of the IW ladder (IW only)
 *   R <status> <plan length>             the result (the fork's status names)
 * mymyr_iw --trace writes the same format.
 */

#include <mimir/mimir.hpp>

#include <algorithm>
#include <cstdio>
#include <iostream>
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

std::string action_str(GroundAction a)
{
    std::string s = "(" + a->get_action()->get_name();
    for (auto o : a->get_objects())
        s += " " + o->get_name();
    return s + ")";
}

std::string atoms_str(const State& state)
{
    std::vector<std::string> out;
    const auto& repos = state.get_problem().get_repositories();
    for (auto a : repos.get_ground_atoms_from_indices<FluentTag>(state.get_atoms<FluentTag>()))
    {
        std::string s = "(" + a->get_predicate()->get_name();
        for (auto o : a->get_objects())
            s += " " + o->get_name();
        out.push_back(s + ")");
    }
    std::sort(out.begin(), out.end());
    std::string r;
    for (const auto& s : out)
        r += (r.empty() ? "" : " ") + s;
    return r;
}

class TraceHandler : public brfs::EventHandlerBase<TraceHandler>
{
    friend class brfs::EventHandlerBase<TraceHandler>;

    void on_expand_state_impl(const State& state) const { std::printf("E %s\n", atoms_str(state).c_str()); }
    void on_expand_goal_state_impl(const State& state) const { std::printf("X %s\n", atoms_str(state).c_str()); }
    void on_generate_state_impl(const State&, GroundAction, ContinuousCost, const State&) const {}
    void on_generate_state_in_search_tree_impl(const State&, GroundAction action, ContinuousCost, const State&) const
    {
        std::printf("G + %s\n", action_str(action).c_str());
    }
    void on_generate_state_not_in_search_tree_impl(const State&, GroundAction action, ContinuousCost, const State&) const
    {
        std::printf("G - %s\n", action_str(action).c_str());
    }
    void on_finish_g_layer_impl(uint32_t, uint64_t, uint64_t) const {}
    void on_start_search_impl(const State&) const { std::printf("P\n"); }
    void on_end_search_impl(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) const {}
    void on_solved_impl(const Plan&) const {}
    void on_unsolvable_impl() const {}
    void on_exhausted_impl() const {}

public:
    explicit TraceHandler(Problem problem) : brfs::EventHandlerBase<TraceHandler>(problem, false) {}
};
}  // namespace

int main(int argc, char** argv)
{
    std::string domain, problem_path, algo = "iw";
    size_t k = 1;
    uint32_t timeout_ms = 120000;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
                std::exit(2);
            return argv[++i];
        };
        if (a == "--domain")
            domain = next();
        else if (a == "--problem")
            problem_path = next();
        else if (a == "--algo")
            algo = next();
        else if (a == "--k")
            k = std::stoul(next());
        else if (a == "--timeout-ms")
            timeout_ms = std::stoul(next());
        else
        {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    auto problem = ProblemImpl::create(domain, problem_path);
    auto context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    auto iw_eh = iw::DefaultEventHandlerImpl::create(problem, true);
    auto brfs_eh = std::make_shared<TraceHandler>(problem);
    SearchResult result;
    if (algo == "siw")
    {
        auto opts = siw::Options();
        opts.max_arity = k;
        opts.siw_event_handler = siw::DefaultEventHandlerImpl::create(problem, true);
        opts.iw_event_handler = iw_eh;
        opts.brfs_event_handler = brfs_eh;
        result = siw::find_solution(context, opts);
    }
    else
    {
        auto opts = iw::Options();
        opts.max_arity = k;
        opts.iw_event_handler = iw_eh;
        opts.brfs_event_handler = brfs_eh;
        opts.max_time_in_ms = timeout_ms;
        result = iw::find_solution(context, opts);
        for (const auto& st : iw_eh->get_statistics().get_brfs_statistics_by_arity())
            std::printf("S %lu/%lu/%lu\n", (unsigned long) st.get_num_expanded(), (unsigned long) st.get_num_generated(),
                        (unsigned long) st.get_num_generated_in_search_tree());
    }
    std::printf("R %s %zu\n", status_name(result.status), result.plan ? result.plan->get_actions().size() : size_t(0));
    std::fflush(stdout);
    std::_Exit(0);
}
