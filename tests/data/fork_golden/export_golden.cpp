/*
 * export_golden: golden expectations from the mimir fork (v0.16.3) for ONE task, for the mymyr parity tests.
 * Built out-of-source against an install of the fork (see build.sh); the fork's source tree is never touched.
 *
 * Successor generation is always LIFTED (KPKC, symmetry pruning OFF).
 * Every search phase parses the task afresh (its own Problem, repositories and SearchContext), so the numbers equal
 * those of a separately run phase and do not depend on which phases ran before.
 *
 * Output: one line per finished phase on stdout, "PHASE <name> <json>", flushed immediately, so a driver that kills
 * the process on a budget keeps every phase that finished. export_all.py assembles the lines into
 * tests/data/expected/<domain>__<problem>.json (format: README.md).
 *
 * Phases (--phases, comma separated, in this order):
 *   task   names (objects and schemas in the fork's order, which equals the text export's; predicates by kind, static
 *          ones sorted by name)
 *   walks  seeded random walks: per step the fluent atoms, the derived atoms (for normalization's axiom_<k>
 *          predicates with the objects sorted), the applicable ground actions, blind and goal-count values, the goal
 *          flag and the action taken
 *   iw1    iw::find_solution with max_arity 1 (per-pass statistics, status, plan length and cost)
 *   iw2    iw::find_solution with max_arity 2
 *   brfs   brfs::find_solution with stop_if_goal = false (exhaustive state count within the budgets)
 *   heur   h_max, h_add and h_ff (grounded through LiftedGrounder, as the fork's planner_astar) on the walk states
 *   astar  astar_eager::find_solution: optimal cost (blind, or h_max on unit-cost tasks when --astar-h hmax/auto)
 *
 * Conventions (README.md): atom "(pred o1 ... ok)"; ground action "(schema o1 ... on)" over ALL normalized
 * parameters; lists sorted by std::string byte order; set hash = sum mod 2^64 of FNV-1a-64(item) over the items,
 * written as 16 hex digits; RNG = splitmix64(seed), next action = sorted_actions[next() % count].
 */

#include <mimir/mimir.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mimir;
using namespace mimir::search;
using namespace mimir::formalism;

namespace
{

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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

/* ---------------------------------------------------------------- JSON helpers */

std::string jstr(const std::string& s)
{
    std::string o = "\"";
    for (unsigned char c : s)
    {
        if (c == '"' || c == '\\')
        {
            o += '\\';
            o += (char) c;
        }
        else if (c < 0x20)
        {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        }
        else
            o += (char) c;
    }
    return o + "\"";
}

std::string jnum(double v)
{
    if (!std::isfinite(v))
        return "null";  // h = infinity (dead end) is written as null
    if (v == std::floor(v) && std::fabs(v) < 1e15)
        return std::to_string((long long) v);
    std::ostringstream s;
    s.precision(17);
    s << v;
    return s.str();
}

std::string jlist(const std::vector<std::string>& xs)
{
    std::string o = "[";
    for (size_t i = 0; i < xs.size(); ++i)
        o += (i ? "," : "") + jstr(xs[i]);
    return o + "]";
}

uint64_t fnv1a64(const std::string& s)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s)
    {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string set_hash(const std::vector<std::string>& xs)
{
    uint64_t h = 0;
    for (const auto& x : xs)
        h += fnv1a64(x);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long) h);
    return buf;
}

/// {"count": n, "hash": "..", "items": [...]} (items only when n <= full_limit)
std::string jset(const std::vector<std::string>& sorted, size_t full_limit)
{
    std::string o = "{\"count\":" + std::to_string(sorted.size()) + ",\"hash\":" + jstr(set_hash(sorted));
    if (sorted.size() <= full_limit)
        o += ",\"items\":" + jlist(sorted);
    return o + "}";
}

void emit(const std::string& phase, const std::string& json)
{
    std::cout << "PHASE " << phase << " " << json << "\n";
    std::cout.flush();
}

/* ---------------------------------------------------------------- strings */

template<typename P>
std::string atom_str(GroundAtom<P> a)
{
    std::string s = "(" + a->get_predicate()->get_name();
    for (auto o : a->get_objects())
        s += " " + o->get_name();
    return s + ")";
}

std::string action_str(GroundAction a)
{
    std::string s = "(" + a->get_action()->get_name();
    for (auto o : a->get_objects())
        s += " " + o->get_name();
    return s + ")";
}

template<typename P>
std::vector<std::string> state_atoms(const State& state)
{
    std::vector<std::string> out;
    const auto& repos = state.get_problem().get_repositories();
    for (auto a : repos.get_ground_atoms_from_indices<P>(state.get_atoms<P>()))
        out.push_back(atom_str(a));
    std::sort(out.begin(), out.end());
    return out;
}

/* Derived predicates that normalization introduces (loki's axiom_<k>) are not in the domain file. The fork orders their
   arguments by heap address, so the argument order changes from run to run. Their atoms are written with the objects
   sorted, which is deterministic (and lossy); atoms of declared derived predicates keep their argument order. */
std::set<std::string> g_domain_tokens;

void load_domain_tokens(const std::string& path)
{
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string tok;
    bool comment = false;
    for (char c : text)
    {
        if (comment)
        {
            comment = (c != '\n');
            continue;
        }
        if (c == ';' || std::isspace((unsigned char) c) || c == '(' || c == ')')
        {
            if (!tok.empty())
                g_domain_tokens.insert(tok);
            tok.clear();
            comment = (c == ';');
            continue;
        }
        tok += (char) std::tolower((unsigned char) c);
    }
    if (!tok.empty())
        g_domain_tokens.insert(tok);
}

bool is_introduced(const std::string& predicate_name)
{
    std::string lower;
    for (char c : predicate_name)
        lower += (char) std::tolower((unsigned char) c);
    return !g_domain_tokens.count(lower);
}

/// The derived atoms of a state, sorted: "(pred o1 ... ok)", with the objects sorted for introduced predicates.
std::vector<std::string> derived_atoms(const State& state)
{
    std::vector<std::string> out;
    const auto& repos = state.get_problem().get_repositories();
    for (auto a : repos.get_ground_atoms_from_indices<DerivedTag>(state.get_atoms<DerivedTag>()))
    {
        if (!is_introduced(a->get_predicate()->get_name()))
        {
            out.push_back(atom_str(a));
            continue;
        }
        std::vector<std::string> objs;
        for (auto o : a->get_objects())
            objs.push_back(o->get_name());
        std::sort(objs.begin(), objs.end());
        std::string s = "(" + a->get_predicate()->get_name();
        for (const auto& o : objs)
            s += " " + o;
        out.push_back(s + ")");
    }
    std::sort(out.begin(), out.end());
    return out;
}

/* ---------------------------------------------------------------- RNG */

struct SplitMix64
{
    uint64_t x;
    uint64_t next()
    {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

/* ---------------------------------------------------------------- task setup */

struct Setup
{
    Problem problem;
    SearchContext context;
    double parse_s = 0, context_s = 0;
};

Setup make_setup(const std::string& domain, const std::string& problem)
{
    Setup s;
    double t0 = now_s();
    s.problem = ProblemImpl::create(domain, problem);
    s.parse_s = now_s() - t0;
    t0 = now_s();
    s.context = SearchContextImpl::create(
        s.problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    s.context_s = now_s() - t0;
    return s;
}

/* ---------------------------------------------------------------- phases */

std::string phase_task(const Problem& problem)
{
    const auto& dom = problem->get_domain();
    std::string o = "{\"objects\":[";
    const auto& objs = problem->get_problem_and_domain_objects();
    for (size_t i = 0; i < objs.size(); ++i)
        o += (i ? "," : "") + jstr(objs[i]->get_name());
    o += "],\"predicates\":[";
    // The kinds in the order of export_lifted_fork.cpp: static, fluent, derived (domain), then the problem's derived
    // predicates. Fluent and derived predicates keep the fork's order; static ones are sorted by name, because the fork
    // orders them (the type predicates) by heap address.
    std::map<std::pair<char, std::string>, int> seen;
    std::vector<std::tuple<char, size_t, std::string>> entries;
    std::vector<std::string> introduced;
    auto add = [&](char kind, const auto& ps)
    {
        for (auto p : ps)
            if (seen.emplace(std::make_pair(kind, p->get_name()), (int) seen.size()).second)
            {
                entries.emplace_back(kind, p->get_arity(), p->get_name());
                if (kind == 'D' && is_introduced(p->get_name()))
                    introduced.push_back(p->get_name());
            }
    };
    add('S', dom->get_predicates<StaticTag>());
    std::sort(entries.begin(), entries.end(), [](const auto& l, const auto& r) { return std::get<2>(l) < std::get<2>(r); });
    add('F', dom->get_predicates<FluentTag>());
    add('D', dom->get_predicates<DerivedTag>());
    add('D', problem->get_problem_and_domain_derived_predicates());
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const auto& [kind, arity, name] = entries[i];
        o += std::string(i ? "," : "") + "[" + jstr(std::string(1, kind)) + "," + std::to_string(arity) + "," + jstr(name) + "]";
    }
    o += "],\"introduced_derived_predicates\":" + jlist(introduced) + ",\"schemas\":[";
    std::map<std::string, int> name_count;
    const auto& actions = dom->get_actions();
    for (size_t i = 0; i < actions.size(); ++i)
    {
        auto a = actions[i];
        ++name_count[a->get_name()];
        o += std::string(i ? "," : "") + "{\"name\":" + jstr(a->get_name()) + ",\"arity\":" + std::to_string(a->get_arity())
             + ",\"original_arity\":" + std::to_string(a->get_original_arity()) + "}";
    }
    std::vector<std::string> dup;
    for (auto& [n, c] : name_count)
        if (c > 1)
            dup.push_back(n);
    size_t n_goal = problem->get_goal_literals<StaticTag>().size() + problem->get_goal_literals<FluentTag>().size()
                    + problem->get_goal_literals<DerivedTag>().size();
    o += "],\"duplicate_schema_names\":" + jlist(dup) + ",\"num_axioms\":" + std::to_string(problem->get_problem_and_domain_axioms().size())
         + ",\"num_static_init_atoms\":" + std::to_string(problem->get_static_initial_atoms().size())
         + ",\"num_fluent_init_atoms\":" + std::to_string(problem->get_fluent_initial_atoms().size())
         + ",\"num_goal_literals\":" + std::to_string(n_goal) + ",\"has_metric\":" + (problem->get_optimization_metric() ? "true" : "false") + "}";
    return o;
}

struct WalkStep
{
    State state;
    bool is_goal = false;
    size_t goal_count = 0;
    std::vector<std::string> fluent, derived, actions;  // sorted
    std::string taken;                                  // empty at the last step / dead end
};

struct Walk
{
    uint64_t seed;
    std::vector<WalkStep> steps;
    bool dead_end = false;
    bool duplicate_action_strings = false;
};

std::vector<Walk> run_walks(const Setup& s, size_t num_walks, uint64_t seed_base, size_t num_steps)
{
    auto& aag = *s.context->get_applicable_action_generator();
    auto& repo = *s.context->get_state_repository();
    auto goal_strategy = ProblemGoalStrategyImpl::create(s.problem);
    const bool static_goal = goal_strategy->test_static_goal();
    const auto& fg = s.problem->get_goal_literals<FluentTag>();
    const auto& dg = s.problem->get_goal_literals<DerivedTag>();

    std::vector<Walk> walks;
    for (size_t w = 0; w < num_walks; ++w)
    {
        Walk walk;
        walk.seed = seed_base + w;
        SplitMix64 rng { walk.seed };
        auto [state, metric] = repo.get_or_create_initial_state();
        for (size_t i = 0; i <= num_steps; ++i)
        {
            WalkStep st { state };
            st.fluent = state_atoms<FluentTag>(state);
            st.derived = derived_atoms(state);
            for (auto l : fg)
                st.goal_count += !state.literal_holds(l);
            for (auto l : dg)
                st.goal_count += !state.literal_holds(l);
            st.is_goal = static_goal && goal_strategy->test_dynamic_goal(state);
            std::vector<std::pair<std::string, GroundAction>> acts;
            for (auto a : aag.create_applicable_action_generator(state))
                acts.emplace_back(action_str(a), a);
            std::stable_sort(acts.begin(), acts.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
            for (size_t j = 0; j < acts.size(); ++j)
            {
                st.actions.push_back(acts[j].first);
                if (j && acts[j].first == acts[j - 1].first)
                    walk.duplicate_action_strings = true;
            }
            if (i < num_steps && !acts.empty())
            {
                const auto& [name, action] = acts[rng.next() % acts.size()];
                st.taken = name;
                auto [succ, succ_metric] = repo.get_or_create_successor_state(state, action, metric);
                walk.steps.push_back(std::move(st));
                state = succ;
                metric = succ_metric;
                continue;
            }
            walk.dead_end = acts.empty();
            walk.steps.push_back(std::move(st));
            break;
        }
        walks.push_back(std::move(walk));
    }
    return walks;
}

std::string walk_json(const Walk& walk, size_t full_limit)
{
    std::string o = "{\"seed\":" + std::to_string(walk.seed) + ",\"dead_end\":" + (walk.dead_end ? "true" : "false")
                    + ",\"duplicate_action_strings\":" + (walk.duplicate_action_strings ? "true" : "false") + ",\"steps\":[";
    for (size_t i = 0; i < walk.steps.size(); ++i)
    {
        const auto& st = walk.steps[i];
        o += std::string(i ? "," : "") + "{\"fluent_atoms\":" + jset(st.fluent, SIZE_MAX) + ",\"derived_atoms\":" + jset(st.derived, SIZE_MAX)
             + ",\"is_goal\":" + (st.is_goal ? "true" : "false") + ",\"h\":{\"blind\":0,\"goal_count\":" + std::to_string(st.goal_count)
             + "},\"applicable\":" + jset(st.actions, full_limit) + ",\"taken\":" + (st.taken.empty() ? "null" : jstr(st.taken)) + "}";
    }
    return o + "]}";
}

std::string brfs_pass_json(const brfs::Statistics& st, size_t arity)
{
    return "{\"arity\":" + std::to_string(arity) + ",\"expanded\":" + std::to_string(st.get_num_expanded())
           + ",\"generated\":" + std::to_string(st.get_num_generated()) + ",\"generated_in_tree\":" + std::to_string(st.get_num_generated_in_search_tree())
           + "}";
}

std::string plan_json(const SearchResult& r)
{
    if (!r.plan)
        return "\"plan_length\":null,\"plan_cost\":null";
    return "\"plan_length\":" + std::to_string(r.plan->get_actions().size()) + ",\"plan_cost\":" + jnum(r.plan->get_cost());
}

std::string phase_iw(const Setup& s, size_t k, uint32_t max_ms)
{
    auto iw_eh = iw::DefaultEventHandlerImpl::create(s.problem, true);
    auto brfs_eh = brfs::DefaultEventHandlerImpl::create(s.problem, true);
    auto opts = iw::Options();
    opts.max_arity = k;
    opts.iw_event_handler = iw_eh;
    opts.brfs_event_handler = brfs_eh;
    opts.max_time_in_ms = max_ms;
    double t0 = now_s();
    auto result = iw::find_solution(s.context, opts);
    double t = now_s() - t0;
    std::string passes;
    uint64_t exp = 0, gen = 0;
    const auto& by_arity = iw_eh->get_statistics().get_brfs_statistics_by_arity();
    for (size_t a = 0; a < by_arity.size(); ++a)
    {
        passes += (a ? "," : "") + brfs_pass_json(by_arity[a], a);
        exp += by_arity[a].get_num_expanded();
        gen += by_arity[a].get_num_generated();
    }
    return "{\"k\":" + std::to_string(k) + ",\"status\":" + jstr(status_name(result.status)) + "," + plan_json(result) + ",\"expanded\":" + std::to_string(exp)
           + ",\"generated\":" + std::to_string(gen) + ",\"passes\":[" + passes + "],\"budget\":{\"max_time_ms\":" + std::to_string(max_ms)
           + "},\"meta\":{\"search_s\":" + jnum(t) + ",\"parse_s\":" + jnum(s.parse_s) + "}}";
}

std::string phase_brfs(const Setup& s, uint32_t max_states, uint32_t max_ms)
{
    auto eh = brfs::DefaultEventHandlerImpl::create(s.problem, true);
    auto opts = brfs::Options();
    opts.stop_if_goal = false;
    opts.event_handler = eh;
    opts.max_num_states = max_states;
    opts.max_time_in_ms = max_ms;
    double t0 = now_s();
    auto result = brfs::find_solution(s.context, opts);
    double t = now_s() - t0;
    const auto& st = eh->get_statistics();
    const bool exhausted = result.status == SearchStatus::EXHAUSTED;
    return "{\"status\":" + jstr(status_name(result.status)) + ",\"states\":" + (exhausted ? std::to_string(st.get_num_expanded()) : std::string("null"))
           + ",\"expanded\":" + std::to_string(st.get_num_expanded()) + ",\"generated\":" + std::to_string(st.get_num_generated())
           + ",\"num_states_in_repository\":" + std::to_string(s.context->get_state_repository()->get_state_count())
           + ",\"budget\":{\"max_states\":" + std::to_string(max_states) + ",\"max_time_ms\":" + std::to_string(max_ms) + "},\"meta\":{\"search_s\":" + jnum(t)
           + "}}";
}

std::string phase_heur(const Setup& s, const std::vector<Walk>& walks)
{
    double t0 = now_s();
    auto grounder = std::make_unique<LiftedGrounder>(s.problem);
    auto hmax = MaxHeuristicImpl::create(*grounder);
    auto hadd = AddHeuristicImpl::create(*grounder);
    auto hff = FFHeuristicImpl::create(*grounder);
    double setup = now_s() - t0;
    t0 = now_s();
    std::string o = "{\"walks\":[";
    for (size_t w = 0; w < walks.size(); ++w)
    {
        o += std::string(w ? "," : "") + "[";
        for (size_t i = 0; i < walks[w].steps.size(); ++i)
        {
            const auto& state = walks[w].steps[i].state;
            o += std::string(i ? "," : "") + "{\"hmax\":" + jnum(hmax->compute_heuristic(state)) + ",\"hadd\":" + jnum(hadd->compute_heuristic(state))
                 + ",\"hff\":" + jnum(hff->compute_heuristic(state)) + "}";
        }
        o += "]";
    }
    return o + "],\"meta\":{\"grounding_s\":" + jnum(setup) + ",\"eval_s\":" + jnum(now_s() - t0) + "}}";
}

std::string phase_astar(const Setup& s, const std::string& hname_req, uint32_t max_states, uint32_t max_ms)
{
    std::string hname = hname_req;
    if (hname == "auto")
        hname = s.problem->get_optimization_metric() ? "blind" : "hmax";  // unit-cost h_max is admissible only without costs
    std::unique_ptr<LiftedGrounder> grounder;
    Heuristic h;
    double t0 = now_s();
    if (hname == "hmax")
    {
        grounder = std::make_unique<LiftedGrounder>(s.problem);
        h = MaxHeuristicImpl::create(*grounder);
    }
    else
        h = BlindHeuristicImpl::create(s.problem);
    double setup = now_s() - t0;
    auto eh = astar_eager::DefaultEventHandlerImpl::create(s.problem, true);
    auto opts = astar_eager::Options();
    opts.event_handler = eh;
    opts.max_num_states = max_states;
    opts.max_time_in_ms = max_ms;
    t0 = now_s();
    auto result = astar_eager::find_solution(s.context, h, opts);
    double t = now_s() - t0;
    std::string cost = result.status == SearchStatus::SOLVED ? jnum(result.plan->get_cost()) : std::string("null");
    return "{\"status\":" + jstr(status_name(result.status)) + ",\"optimal_cost\":" + cost + "," + plan_json(result) + ",\"heuristic\":" + jstr(hname)
           + ",\"expanded\":" + std::to_string(eh->get_statistics().get_num_expanded()) + ",\"budget\":{\"max_states\":" + std::to_string(max_states)
           + ",\"max_time_ms\":" + std::to_string(max_ms) + "},\"meta\":{\"setup_s\":" + jnum(setup) + ",\"search_s\":" + jnum(t) + "}}";
}

}  // namespace

int main(int argc, char** argv)
{
    std::string domain, problem, phases = "task,walks,iw1,iw2,brfs,heur,astar", astar_h = "auto";
    size_t num_walks = 3, num_steps = 25, full_limit = 5000;
    uint64_t seed_base = 1;
    uint32_t brfs_states = 3'000'000, brfs_ms = 300'000, iw_ms = 300'000, astar_states = 3'000'000, astar_ms = 300'000;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << a << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--domain")
            domain = next();
        else if (a == "--problem")
            problem = next();
        else if (a == "--phases")
            phases = next();
        else if (a == "--walks")
            num_walks = std::stoul(next());
        else if (a == "--steps")
            num_steps = std::stoul(next());
        else if (a == "--seed")
            seed_base = std::stoull(next());
        else if (a == "--full-limit")
            full_limit = std::stoul(next());
        else if (a == "--brfs-max-states")
            brfs_states = std::stoul(next());
        else if (a == "--brfs-max-ms")
            brfs_ms = std::stoul(next());
        else if (a == "--iw-max-ms")
            iw_ms = std::stoul(next());
        else if (a == "--astar-max-states")
            astar_states = std::stoul(next());
        else if (a == "--astar-max-ms")
            astar_ms = std::stoul(next());
        else if (a == "--astar-h")
            astar_h = next();
        else
        {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (domain.empty() || problem.empty())
    {
        std::cerr << "usage: export_golden --domain D --problem P [--phases task,walks,iw1,iw2,brfs,heur,astar] [--walks 3] [--steps 25] [--seed 1]\n"
                     "       [--full-limit 5000] [--brfs-max-states N] [--brfs-max-ms T] [--iw-max-ms T] [--astar-max-states N] [--astar-max-ms T]\n"
                     "       [--astar-h auto|blind|hmax]\n";
        return 2;
    }
    auto want = [&](const std::string& p) { return ("," + phases + ",").find("," + p + ",") != std::string::npos; };
    load_domain_tokens(domain);

    try
    {
        std::optional<Setup> walk_setup;
        std::vector<Walk> walks;
        auto ensure_walks = [&]()
        {
            if (!walk_setup)
            {
                walk_setup = make_setup(domain, problem);
                walks = run_walks(*walk_setup, num_walks, seed_base, num_steps);
            }
        };
        if (want("task"))
        {
            auto s = make_setup(domain, problem);
            emit("task", phase_task(s.problem));
        }
        if (want("walks"))
        {
            double t0 = now_s();
            ensure_walks();
            std::string o = "{\"num_steps\":" + std::to_string(num_steps) + ",\"seed_base\":" + std::to_string(seed_base) + ",\"full_limit\":"
                            + std::to_string(full_limit) + ",\"walks\":[";
            for (size_t w = 0; w < walks.size(); ++w)
                o += (w ? "," : "") + walk_json(walks[w], full_limit);
            emit("walks", o + "],\"meta\":{\"walk_s\":" + jnum(now_s() - t0) + "}}");
        }
        for (size_t k : { 1, 2 })
            if (want("iw" + std::to_string(k)))
                emit("iw" + std::to_string(k), phase_iw(make_setup(domain, problem), k, iw_ms));
        if (want("brfs"))
            emit("brfs", phase_brfs(make_setup(domain, problem), brfs_states, brfs_ms));
        if (want("heur"))
        {
            ensure_walks();
            emit("heur", phase_heur(*walk_setup, walks));
        }
        if (want("astar"))
            emit("astar", phase_astar(make_setup(domain, problem), astar_h, astar_states, astar_ms));
    }
    catch (const std::exception& ex)
    {
        std::cout << "ERROR " << jstr(ex.what()) << std::endl;
        std::_Exit(1);
    }
    std::cout << "DONE" << std::endl;
    // Skip destructors: tearing down large repositories is slow and irrelevant here.
    std::_Exit(0);
}
