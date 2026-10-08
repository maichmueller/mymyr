// search_fork: one search of the mimir fork (0.16.x) on a PDDL task, for the expectations of mymyr's parity tests:
// the numeric best-first test (tests/data/numeric_tasks/fork_best_first.json, made by run_numeric.py), the layer
// ordering test (tests/data/layer_orders/fork_layer_orders.json, made by run_layer_orders.py), the beam test
// (tests/data/beam/fork_beam.json and fork_beam_relaxed.json, made by run_beam.py and run_beam_relaxed.py) and the
// heuristics test
// (tests/data/heuristics/fork_heuristics.json, made by run_heuristics.py), the binding generator test
// (tests/data/bindings/fork_bindings.json, made by run_bindings.py) and the tuple graph tests
// (tests/data/tuple_graphs/fork_tuple_graphs.json, made by run_tuple_graphs.py) and the k-FWL certificate tests
// (tests/data/kfwl/fork_kfwl.json, made by run_kfwl.py).
//
//   search_fork --algo astar_eager|astar_lazy|gbfs_eager|gbfs_lazy --h blind|max|add|ff|setadd|perfect --domain D
//               --problem P [--max-ms T] [--max-states N]
//   search_fork --algo iw|brfs --order in_order|reverse|goal_count|goal_count_fewer [--k K] [--limit L]
//               [--beam W [--beam-mode all_tested|survivors_only|relaxed]] [--tie-seed S] [--threads N] [--chunk C]
//               [--iw1-knobs precheck|atom_first|incremental] --domain D --problem P [--max-ms T] [--max-states N]
//   search_fork --algo astar_iw --h blind|hmax --width K --features classical|abstracted|base_abstracted
//               --domain D --problem P [--max-ms T] [--max-states N]
//   search_fork --algo walk_h --h setadd|h2|perfect --domain D --problem P [--walks W] [--steps S] [--seed B]
//               [--max-states N]
//   search_fork --algo walk_ground --domain D --problem P [--walks W] [--steps S] [--seed B]
//   search_fork --algo tuple_graphs --domain D --problem P [--max-states N] [--sample S] [--max-width W]
//               [--time-width W [--time-pruning 0|1]]
//   search_fork --algo kfwl --domain D --problem P [--max-states N] [--sample S] [--max-n2 M] [--max-n3 M] [--max-n4 M]
//
// Prints one line "RESULT {...}" with status, plan_cost, plan_length, plan (ground action strings), expanded and
// generated (best-first: also deadends; iw: also per-pass statistics), or "ERROR <message>". Successor generation is
// lifted KPKC with symmetry pruning off; h_max, h_add, h_FF, set-additive and h² are the fork's grounded heuristics
// over a LiftedGrounder (unit action costs), blind is its BlindHeuristic, perfect its PerfectHeuristic over the
// search context's state space (built first with --max-states as its limit: a larger space is an ERROR); the searches
// use the fork's default event handlers and strategies. iw is iw::find_solution with max_arity K and the layer
// ordering strategy (max_next_layer_states L, or beam_width W with beam_novelty_mode; relaxed is SURVIVORS_ONLY with
// relaxed_survivors_only_beam; --tie-seed S randomizes equal-score ties with seed S; --threads and --chunk are
// parallel_beam_num_threads and parallel_beam_chunk_size); brfs is brfs::find_solution with stop_if_goal and the same
// ordering. --iw1-knobs adds IW(1)'s iw1_precheck_add_effect_novelty, then iw1_atom_first_mode, then
// iw1_incremental_first_applicability (each implies the ones before).
// walk_h evaluates the heuristic on the states of the seeded random walks of tests/data/fork_golden/README.md (walk w
// uses seed B + w, the next action is sorted_applicable[splitmix64() % count]; W = 3, S = 25, B = 1 as in the
// golden files) and prints per walk and step the fluent atom count, the set hash of their strings and h.
// walk_ground prints on the same walks, per step, the groundings (count, set hash of the binding strings, ground
// literals per kind) of the goal literals as a ConjunctiveCondition and, per action schema in domain order, of its
// precondition (ConjunctiveConditionSatisficingBindingGenerator) and of the action (ActionSatisficingBindingGenerator),
// and the action taken.
// tuple_graphs prints the state space's size and the digests of the fork's tuple graphs (TupleGraphImpl::create) of
// width 0, and of widths 1 and 2 (up to --max-width) with and without dominance pruning, of every
// ceil(N / S)-th vertex (tests/data/fork_golden/README.md, "Tuple graphs"); with --time-width only that width and
// pruning, timed over every vertex, with the peak RSS before and after.
// kfwl prints, for every ceil(N / S)-th vertex of the state space, the state key, the number of vertices of its object
// graph (datasets::create_object_graph), its class among these states by nauty canonical form and by the fork's
// k-FWL certificate for k = 2, 3, 4 (only on graphs of at most M vertices; M = 0 skips k), the seconds and peak RSS of
// the certificates, and the size of the fork's symmetry-reduced state space (nauty).

#include <mimir/mimir.hpp>
#include <mimir/search/algorithms/astar_iw.hpp>
#include <mimir/search/algorithms/astar_iw/event_handlers/default.hpp>
#include <mimir/graphs/algorithms/folklore_weisfeiler_leman.hpp>
#include <mimir/graphs/algorithms/nauty.hpp>
#include <mimir/search/heuristics/h2.hpp>  // not in mimir.hpp

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/resource.h>

using namespace mimir;
using namespace mimir::search;
using namespace mimir::formalism;

namespace
{
/// The symmetry pruning of every search context (--symmetry off|wl1).
SearchContextImpl::SymmetryPruning g_symmetry = SearchContextImpl::SymmetryPruning::OFF;
struct BeamKnobs
{
    long tie_seed = -1;  // -1: ties in generation order
    uint32_t threads = 1;
    uint32_t chunk = 1024;
    int iw1 = 0;  // 1 precheck, 2 + atom first, 3 + incremental first applicability
};
BeamKnobs g_beam;

template<class Options>
void set_beam_knobs(Options& opts, const std::string& mode)
{
    opts.relaxed_survivors_only_beam = mode == "relaxed";
    opts.randomize_equal_score_ties = g_beam.tie_seed >= 0;
    opts.equal_score_tie_seed = g_beam.tie_seed >= 0 ? static_cast<uint64_t>(g_beam.tie_seed) : 0;
    opts.parallel_beam_num_threads = g_beam.threads;
    opts.parallel_beam_chunk_size = g_beam.chunk;
    opts.iw1_precheck_add_effect_novelty = g_beam.iw1 >= 1;
    opts.iw1_atom_first_mode = g_beam.iw1 >= 2;
    opts.iw1_incremental_first_applicability = g_beam.iw1 >= 3;
}

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

/// The number of fluent atoms of a state and the set hash of their strings (README.md: sum of FNV-1a-64).
std::pair<size_t, std::string> fluent_hash(const State& state)
{
    const auto& repos = state.get_problem().get_repositories();
    uint64_t h = 0;
    size_t n = 0;
    for (auto a : repos.get_ground_atoms_from_indices<FluentTag>(state.get_atoms<FluentTag>()))
    {
        std::string str = "(" + a->get_predicate()->get_name();
        for (auto o : a->get_objects())
            str += " " + o->get_name();
        h += fnv1a64(str + ")");
        ++n;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return {n, buf};
}

/// The fork's PerfectHeuristic, after checking that the state space has fewer than max_states states.
Heuristic perfect_heuristic(const SearchContext& context, uint32_t max_states)
{
    auto options = datasets::StateSpaceImpl::Options();
    options.remove_if_unsolvable = false;
    options.max_num_states = max_states;
    if (!datasets::StateSpaceImpl::create(context, options))
        throw std::runtime_error("the state space has at least " + std::to_string(max_states) + " states");
    return PerfectHeuristicImpl::create(context);
}

int run_walk_h(const std::string& domain, const std::string& problem_file, const std::string& hname, size_t walks, size_t steps,
               uint64_t seed_base, uint32_t max_states)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(g_symmetry))));
    std::unique_ptr<LiftedGrounder> grounder;
    Heuristic h;
    if (hname == "perfect")
        h = perfect_heuristic(context, max_states);
    else
    {
        grounder = std::make_unique<LiftedGrounder>(problem);
        if (hname == "setadd")
            h = SetAddHeuristicImpl::create(*grounder);
        else if (hname == "h2")
            h = H2HeuristicImpl::create(*grounder);
        else
        {
            std::cerr << "unknown heuristic " << hname << "\n";
            return 2;
        }
    }
    auto& aag = *context->get_applicable_action_generator();
    auto& repo = *context->get_state_repository();
    std::string body = "\"walks\":[";
    for (size_t w = 0; w < walks; ++w)
    {
        SplitMix64 rng { seed_base + w };
        auto [state, metric] = repo.get_or_create_initial_state();
        body += std::string(w ? "," : "") + "{\"seed\":" + std::to_string(seed_base + w) + ",\"steps\":[";
        for (size_t i = 0; i <= steps; ++i)
        {
            const auto [n, hash] = fluent_hash(state);
            body += std::string(i ? "," : "") + "{\"atoms\":" + std::to_string(n) + ",\"hash\":" + jstr(hash) +
                    ",\"h\":" + jnum(h->compute_heuristic(state)) + "}";
            std::vector<std::pair<std::string, GroundAction>> acts;
            for (auto a : aag.create_applicable_action_generator(state))
                acts.emplace_back(action_str(a), a);
            std::stable_sort(acts.begin(), acts.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
            if (i == steps || acts.empty())
                break;
            const auto& action = acts[rng.next() % acts.size()].second;
            auto [succ, succ_metric] = repo.get_or_create_successor_state(state, action, metric);
            state = succ;
            metric = succ_metric;
        }
        body += "]}";
    }
    body += "]";
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "RESULT {\"algo\":\"walk_h\",\"h\":" << jstr(hname) << "," << body << ",\"seconds\":" << jnum(secs) << "}" << std::endl;
    return 0;
}

template<typename P>
LiteralList<P> lift_literals(const Problem& problem, const GroundLiteralList<P>& ground)
{
    LiteralList<P> out;
    for (const auto& l : ground)
    {
        TermList terms;
        for (auto o : l->get_atom()->get_objects())
            terms.push_back(problem->get_or_create_term(o));
        out.push_back(problem->get_or_create_literal(l->get_polarity(), problem->get_or_create_atom(l->get_atom()->get_predicate(), terms)));
    }
    return out;
}

/// The goal's literals as a ConjunctiveCondition without parameters (numeric goal constraints left out).
ConjunctiveCondition goal_condition(const Problem& problem)
{
    LiteralList<StaticTag> st = lift_literals<StaticTag>(problem, problem->get_goal_literals<StaticTag>());
    LiteralList<FluentTag> fl = lift_literals<FluentTag>(problem, problem->get_goal_literals<FluentTag>());
    LiteralList<DerivedTag> de = lift_literals<DerivedTag>(problem, problem->get_goal_literals<DerivedTag>());
    return problem->get_or_create_conjunctive_condition(
        ParameterList {},
        boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c<StaticTag>, std::move(st)),
                              boost::hana::make_pair(boost::hana::type_c<FluentTag>, std::move(fl)),
                              boost::hana::make_pair(boost::hana::type_c<DerivedTag>, std::move(de))),
        NumericConstraintList {});
}

/// The number of groundings of a binding generator in a state, the set hash of the strings "(name o1 ... ok)" of the
/// bindings, and the number of static, fluent and derived ground literals summed over the groundings.
template<typename Generator>
std::string ground_counts(Generator& g, const State& state, const std::string& name)
{
    size_t n = 0, lits[3] = { 0, 0, 0 };
    uint64_t h = 0;
    for (const auto& [binding, literals] : g.create_ground_conjunction_generator(state))
    {
        std::string s = "(" + name;
        for (auto o : binding)
            s += " " + o->get_name();
        h += fnv1a64(s + ")");
        lits[0] += std::get<0>(literals).size();
        lits[1] += std::get<1>(literals).size();
        lits[2] += std::get<2>(literals).size();
        ++n;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return "{\"n\":" + std::to_string(n) + ",\"hash\":" + jstr(buf) + ",\"literals\":[" + std::to_string(lits[0]) + "," +
           std::to_string(lits[1]) + "," + std::to_string(lits[2]) + "]}";
}

/// walk_ground: along the seeded walks of walk_h, per step the groundings of the goal (as a ConjunctiveCondition) and,
/// per action schema, of its precondition (ConjunctiveConditionSatisficingBindingGenerator) and of the action itself
/// (ActionSatisficingBindingGenerator: also the numeric effect checks).
int run_walk_ground(const std::string& domain, const std::string& problem_file, size_t walks, size_t steps, uint64_t seed_base)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(g_symmetry))));
    const ActionList& actions = problem->get_domain()->get_actions();
    std::vector<ConjunctiveConditionSatisficingBindingGenerator> pre;
    std::vector<ActionSatisficingBindingGenerator> act;
    for (const auto& a : actions)
    {
        pre.emplace_back(a->get_conjunctive_condition(), problem);
        act.emplace_back(a, problem);
    }
    ConjunctiveConditionSatisficingBindingGenerator goal(goal_condition(problem), problem);
    auto& aag = *context->get_applicable_action_generator();
    auto& repo = *context->get_state_repository();
    std::string body = "\"schemas\":[";
    for (size_t i = 0; i < actions.size(); ++i)
        body += std::string(i ? "," : "") + jstr(actions[i]->get_name());
    body += "],\"walks\":[";
    for (size_t w = 0; w < walks; ++w)
    {
        SplitMix64 rng { seed_base + w };
        auto [state, metric] = repo.get_or_create_initial_state();
        body += std::string(w ? "," : "") + "{\"seed\":" + std::to_string(seed_base + w) + ",\"steps\":[";
        for (size_t i = 0; i <= steps; ++i)
        {
            const auto [n, hash] = fluent_hash(state);
            body += std::string(i ? "," : "") + "{\"atoms\":" + std::to_string(n) + ",\"hash\":" + jstr(hash) +
                    ",\"goal\":" + ground_counts(goal, state, "goal") + ",\"pre\":[";
            for (size_t k = 0; k < actions.size(); ++k)
                body += std::string(k ? "," : "") + ground_counts(pre[k], state, actions[k]->get_name());
            body += "],\"act\":[";
            for (size_t k = 0; k < actions.size(); ++k)
                body += std::string(k ? "," : "") + ground_counts(act[k], state, actions[k]->get_name());
            body += "]";
            std::vector<std::pair<std::string, GroundAction>> acts;
            for (auto a : aag.create_applicable_action_generator(state))
                acts.emplace_back(action_str(a), a);
            std::stable_sort(acts.begin(), acts.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
            if (i == steps || acts.empty())
            {
                body += "}";
                break;
            }
            const auto& [label, action] = acts[rng.next() % acts.size()];
            body += ",\"action\":" + jstr(label) + "}";
            auto [succ, succ_metric] = repo.get_or_create_successor_state(state, action, metric);
            state = succ;
            metric = succ_metric;
        }
        body += "]}";
    }
    body += "]";
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "RESULT {\"algo\":\"walk_ground\"," << body << ",\"seconds\":" << jnum(secs) << "}" << std::endl;
    return 0;
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

/// symmetry_states: every state reachable without pruning (breadth-first, at most max_states), each with its fluent
/// atoms (and fluent function values), its applicable actions under the WL1 symmetry pruning of the KPKC generator, and the WL1 colour class of
/// every object (the colour refinement of the state's object graph, classes numbered by first occurrence in object
/// order); then, with WL1 pruning, brfs exhaustively and stopping at a goal, and astar_eager with the blind
/// heuristic.
int run_symmetry_states(const std::string& domain, const std::string& problem_file, uint32_t max_ms, uint32_t max_states)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    using Kpkc = SearchContextImpl::LiftedOptions::KPKCOptions;
    SearchContext context = SearchContextImpl::create(
        problem, SearchContextImpl::Options(SearchContextImpl::LiftedOptions(Kpkc(SearchContextImpl::SymmetryPruning::OFF))));
    auto pruned = KPKCLiftedApplicableActionGeneratorImpl::create(problem, Kpkc(SearchContextImpl::SymmetryPruning::WL1));
    auto& aag = *context->get_applicable_action_generator();
    auto& repo = *context->get_state_repository();
    const auto& objects = problem->get_problem_and_domain_objects();

    std::vector<std::string> atom_names, action_names;
    std::unordered_map<std::string, size_t> atom_ids, action_ids;
    auto intern = [](std::vector<std::string>& names, std::unordered_map<std::string, size_t>& ids, const std::string& s)
    {
        const auto [it, fresh] = ids.emplace(s, names.size());
        if (fresh)
            names.push_back(s);
        return it->second;
    };
    std::string states = "[";
    std::unordered_set<Index> seen;
    std::deque<std::pair<State, ContinuousCost>> queue;
    queue.push_back(repo.get_or_create_initial_state());
    seen.insert(queue.front().first.get_index());
    size_t num_states = 0, sum_all = 0, sum_pruned = 0;
    bool complete = true;
    while (!queue.empty())
    {
        const auto [state, metric] = queue.front();
        queue.pop_front();
        std::vector<size_t> atoms, acts;
        for (auto a : repo.get_problem()->get_repositories().get_ground_atoms_from_indices<FluentTag>(state.get_atoms<FluentTag>()))
        {
            std::string s = "(" + a->get_predicate()->get_name();
            for (auto o : a->get_objects())
                s += " " + o->get_name();
            atoms.push_back(intern(atom_names, atom_ids, s + ")"));
        }
        // numeric tasks: the fluent function values, as "(f o1 ... ok)=v" (%.17g; undefined values are left out)
        for (const auto& [f, v] : repo.get_problem()->get_repositories().get_ground_function_values<FluentTag>(state.get_numeric_variables()))
        {
            if (std::isnan(v))
                continue;
            std::string s = "(" + f->get_function_skeleton()->get_name();
            for (auto o : f->get_objects())
                s += " " + o->get_name();
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.17g", v);
            atoms.push_back(intern(atom_names, atom_ids, s + ")=" + buf));
        }
        for (auto a : pruned->create_applicable_action_generator(state))
            acts.push_back(intern(action_names, action_ids, action_str(a)));
        std::sort(atoms.begin(), atoms.end());
        std::sort(acts.begin(), acts.end());
        const auto graph = datasets::create_object_graph(state, *problem);
        const auto certificate = graphs::color_refinement::compute_certificate(graph);
        std::unordered_map<Index, size_t> class_of_color;
        std::string classes = "[";
        for (size_t o = 0; o < objects.size(); ++o)
        {
            const auto [it, fresh] = class_of_color.emplace(certificate->get_hash_to_color()[o], class_of_color.size());
            classes += std::string(o ? "," : "") + std::to_string(it->second);
        }
        states += std::string(num_states ? "," : "") + "{\"atoms\":[";
        for (size_t i = 0; i < atoms.size(); ++i)
            states += std::string(i ? "," : "") + std::to_string(atoms[i]);
        states += "],\"actions\":[";
        for (size_t i = 0; i < acts.size(); ++i)
            states += std::string(i ? "," : "") + std::to_string(acts[i]);
        states += "],\"classes\":" + classes + "]}";
        ++num_states;
        sum_pruned += acts.size();
        for (auto a : aag.create_applicable_action_generator(state))
        {
            ++sum_all;
            auto next = repo.get_or_create_successor_state(state, a, metric);
            if (seen.size() >= max_states && !seen.contains(next.first.get_index()))
            {
                complete = false;
                continue;
            }
            if (seen.insert(next.first.get_index()).second)
                queue.push_back(next);
        }
    }
    states += "]";

    // the searches with WL1 pruning, each on its own context
    auto wl1_context = [&]
    { return SearchContextImpl::create(problem, SearchContextImpl::Options(SearchContextImpl::LiftedOptions(Kpkc(SearchContextImpl::SymmetryPruning::WL1)))); };
    std::string searches = "{";
    for (const bool stop : { false, true })
    {
        auto eh = brfs::DefaultEventHandlerImpl::create(problem, true);
        auto opts = brfs::Options();
        opts.event_handler = eh;
        opts.stop_if_goal = stop;
        opts.max_time_in_ms = max_ms;
        auto result = brfs::find_solution(wl1_context(), opts);
        const auto& st = eh->get_statistics();
        searches += std::string(stop ? ",\"brfs\":{" : "\"brfs_exhaustive\":{") + "\"status\":" + jstr(status_name(result.status)) +
                    plan_json(result) + ",\"expanded\":" + std::to_string(st.get_num_expanded()) +
                    ",\"generated\":" + std::to_string(st.get_num_generated()) + "}";
    }
    {
        const auto ctx = wl1_context();
        searches += ",\"astar_blind\":{" +
                    run<astar_eager::Options>(ctx, BlindHeuristicImpl::create(problem), astar_eager::DefaultEventHandlerImpl::create(problem, true),
                                              max_ms, UINT32_MAX,
                                              [](auto&& c, auto&& hh, auto&& o) { return astar_eager::find_solution(c, hh, o); }) +
                    "}";
    }
    searches += "}";

    std::string names = "[";
    for (size_t o = 0; o < objects.size(); ++o)
        names += std::string(o ? "," : "") + jstr(objects[o]->get_name());
    std::string atoms_json = "[", actions_json = "[";
    for (size_t i = 0; i < atom_names.size(); ++i)
        atoms_json += std::string(i ? "," : "") + jstr(atom_names[i]);
    for (size_t i = 0; i < action_names.size(); ++i)
        actions_json += std::string(i ? "," : "") + jstr(action_names[i]);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "RESULT {\"algo\":\"symmetry_states\",\"complete\":" << (complete ? "true" : "false") << ",\"num_states\":" << num_states
              << ",\"num_actions\":" << sum_all << ",\"num_pruned_actions\":" << sum_pruned << ",\"objects\":" << names
              << "],\"atom_names\":" << atoms_json << "],\"action_names\":" << actions_json << "],\"states\":" << states
              << ",\"searches\":" << searches << ",\"seconds\":" << jnum(secs) << "}" << std::endl;
    return 0;
}

BeamNoveltyMode beam_mode(const std::string& m)
{
    if (m == "all_tested")
        return BeamNoveltyMode::ALL_TESTED;
    if (m == "survivors_only" || m == "relaxed")
        return BeamNoveltyMode::SURVIVORS_ONLY;
    throw std::invalid_argument("unknown beam mode " + m);
}

int run_layered(const std::string& domain, const std::string& problem_file, const std::string& algo, const std::string& order, size_t k,
                uint32_t limit, uint32_t beam, const std::string& mode, uint32_t max_ms, uint32_t max_states)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(g_symmetry))));
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
        opts.beam_width = beam;
        opts.beam_novelty_mode = beam_mode(mode);
        set_beam_knobs(opts, mode);
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
        opts.beam_width = beam;
        opts.beam_novelty_mode = beam_mode(mode);
        set_beam_knobs(opts, mode);
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
              << (limit == UINT32_MAX ? std::string("null") : std::to_string(limit));
    if (beam != UINT32_MAX)
        std::cout << ",\"beam\":" << beam << ",\"beam_mode\":" << jstr(mode);
    std::cout << "," << body << ",\"seconds\":" << jnum(secs) << "}" << std::endl;
    return 0;
}

int run_search(const std::string& domain, const std::string& problem_file, const std::string& algo, const std::string& hname, uint32_t max_ms,
           uint32_t max_states, size_t width, const std::string& features)
{
    const auto t0 = std::chrono::steady_clock::now();
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(g_symmetry))));
    std::unique_ptr<LiftedGrounder> grounder;
    Heuristic h;
    if (hname == "blind")
        h = BlindHeuristicImpl::create(problem);
    else if (hname == "perfect")
        h = perfect_heuristic(context, max_states);
    else
    {
        grounder = std::make_unique<LiftedGrounder>(problem);
        if (hname == "max" || hname == "hmax")
            h = MaxHeuristicImpl::create(*grounder);
        else if (hname == "add")
            h = AddHeuristicImpl::create(*grounder);
        else if (hname == "ff")
            h = FFHeuristicImpl::create(*grounder);
        else if (hname == "setadd")
            h = SetAddHeuristicImpl::create(*grounder);
        else
        {
            std::cerr << "unknown heuristic " << hname << "\n";
            return 2;
        }
    }
    std::string body;
    if (algo == "astar_iw")
    {
        astar_iw::NoveltyFeatureMode feature_mode;
        if (features == "classical")
            feature_mode = astar_iw::NoveltyFeatureMode::CLASSICAL;
        else if (features == "abstracted")
            feature_mode = astar_iw::NoveltyFeatureMode::ABSTRACTED;
        else if (features == "base_abstracted")
            feature_mode = astar_iw::NoveltyFeatureMode::BASE_ABSTRACTED;
        else
            throw std::invalid_argument("unknown features " + features);
        body = run<astar_iw::Options>(context, h, astar_iw::DefaultEventHandlerImpl::create(problem, true), max_ms, max_states,
            [&](auto&& c, auto&& hh, auto o)
            {
                o.width = width;
                o.novelty_feature_mode = feature_mode;
                return astar_iw::find_solution(c, hh, o);
            });
        body += ",\"width\":" + std::to_string(width) + ",\"features\":" + jstr(features);
    }
    else if (algo == "astar_eager")
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

// ------------------------------------------------------------------------------------------------ tuple graphs
std::string fluent_atom_str(const Problem& problem, Index index)
{
    const auto a = problem->get_repositories().get_ground_atom<FluentTag>(index);
    std::string s = "(" + a->get_predicate()->get_name();
    for (auto o : a->get_objects())
        s += " " + o->get_name();
    return s + ")";
}

/// FNV-1a-64 of the state's fluent atom strings, sorted and joined by newlines, followed by "\n=%.17g" per numeric
/// variable (the state hash of the fork's state-space fingerprints in tests/cpp/datasets/fork_cases.inc), as 16 hex
/// digits.
std::string state_key(const Problem& problem, const State& state)
{
    std::vector<std::string> atoms;
    for (const auto a : state.get_atoms<FluentTag>())
        atoms.push_back(fluent_atom_str(problem, a));
    std::sort(atoms.begin(), atoms.end());
    std::string joined;
    for (size_t i = 0; i < atoms.size(); ++i)
        joined += (i ? "\n" : "") + atoms[i];
    for (const double x : state.get_numeric_variables())
    {
        char b[64];
        std::snprintf(b, sizeof b, "\n=%.17g", x);
        joined += b;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fnv1a64(joined)));
    return buf;
}

long peak_rss_kb()
{
    rusage u {};
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss;
}

/// Per (distance, sorted problem vertices): every tuple of size <= width whose states of first novelty are these, by a
/// breadth-first search from `root` over the state space graph with the fork's DynamicNoveltyTable (the tuples of the
/// states of a layer are novel iff no state of an earlier layer has them).
std::map<std::pair<size_t, IndexList>, std::vector<iw::AtomIndexList>> tuple_classes(const datasets::StateSpace& space, Index root, size_t width)
{
    const auto& graph = space->get_graph();
    iw::DynamicNoveltyTable table(width);
    std::map<iw::AtomIndexList, std::pair<size_t, std::set<Index>>> novel;
    std::vector<Index> layer { root }, next;
    std::set<Index> visited { root };
    std::vector<iw::AtomIndexList> tuples;
    for (size_t d = 0; !layer.empty(); ++d)
    {
        for (const auto s : layer)
        {
            table.compute_novel_tuples(graphs::get_state(graph.get_vertex(s)), tuples);
            for (const auto& t : tuples)
            {
                auto& entry = novel[t];
                entry.first = d;
                entry.second.insert(s);
            }
        }
        for (const auto s : layer)
            table.test_novelty_and_update_table(graphs::get_state(graph.get_vertex(s)));
        next.clear();
        for (const auto s : layer)
            for (const auto t : graph.get_adjacent_vertex_indices<graphs::ForwardTag>(s))
                if (visited.insert(t).second)
                    next.push_back(t);
        std::swap(layer, next);
    }
    std::map<std::pair<size_t, IndexList>, std::vector<iw::AtomIndexList>> classes;
    for (const auto& [t, entry] : novel)
        classes[{ entry.first, IndexList(entry.second.begin(), entry.second.end()) }].push_back(t);
    return classes;
}

/// The fork's tuple graphs (TupleGraphImpl::create) of the state space (remove_if_unsolvable = false, no symmetry
/// pruning) for width 0, and widths 1 and 2 with and without dominance pruning, of every state-space vertex (every
/// ceil(N / sample)-th when the space has more than `sample` vertices). With time_width >= 0: only that width and
/// pruning, timed, without the graphs.
int run_tuple_graphs(const std::string& domain, const std::string& problem_file, uint32_t max_states, size_t sample, size_t max_width,
                     long time_width, bool time_pruning)
{
    Problem problem = ProblemImpl::create(domain, problem_file);
    SearchContext context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    auto ss_options = datasets::StateSpaceImpl::Options();
    ss_options.remove_if_unsolvable = false;
    ss_options.max_num_states = max_states;
    const auto t0 = std::chrono::steady_clock::now();
    auto result = datasets::StateSpaceImpl::create(context, ss_options);
    if (!result)
        throw std::runtime_error("no state space (max_states reached or a statically false goal)");
    const auto space = result->first;
    auto certificate_maps = result->second;
    const double space_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto& graph = space->get_graph();
    const size_t N = graph.get_num_vertices();

    if (time_width >= 0)
    {
        const long rss_before = peak_rss_kb();
        const auto t1 = std::chrono::steady_clock::now();
        const auto tuple_graphs =
            datasets::TupleGraphImpl::create(space, certificate_maps, datasets::TupleGraphImpl::Options(static_cast<size_t>(time_width), time_pruning));
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        size_t vertices = 0, edges = 0;
        for (const auto& tg : tuple_graphs)
        {
            vertices += tg->get_graph().get_num_vertices();
            edges += tg->get_graph().get_num_edges();
        }
        std::cout << "RESULT {\"states\":" << N << ",\"width\":" << time_width << ",\"pruning\":" << (time_pruning ? "true" : "false")
                  << ",\"seconds\":" << jnum(secs) << ",\"state_space_seconds\":" << jnum(space_secs) << ",\"tuple_vertices\":" << vertices
                  << ",\"tuple_edges\":" << edges << ",\"peak_rss_kb_before\":" << rss_before << ",\"peak_rss_kb_after\":" << peak_rss_kb()
                  << "}" << std::endl;
        return 0;
    }

    const size_t step = (sample > 0 && N > sample) ? (N + sample - 1) / sample : 1;
    std::vector<Index> roots;
    for (Index v = 0; v < N; v += step)
        roots.push_back(v);
    std::vector<std::string> keys(N);
    for (Index v = 0; v < N; ++v)
        keys[v] = state_key(problem, graphs::get_state(graph.get_vertex(v)));
    std::string body = "\"states\":" + std::to_string(N) + ",\"sample_step\":" + std::to_string(step) + ",\"roots\":[";
    for (size_t i = 0; i < roots.size(); ++i)
        body += std::string(i ? "," : "") + jstr(keys[roots[i]]);
    body += "]";

    std::map<Index, std::string> atom_names;
    auto atom_name = [&](Index a) -> const std::string&
    {
        auto it = atom_names.find(a);
        if (it == atom_names.end())
            it = atom_names.emplace(a, fluent_atom_str(problem, a)).first;
        return it->second;
    };
    // a tuple as its atom names, sorted
    auto names = [&](const iw::AtomIndexList& tuple)
    {
        std::vector<std::string> out;
        for (const auto a : tuple)
            out.push_back(atom_name(a));
        std::sort(out.begin(), out.end());
        return out;
    };
    auto concat = [](const std::vector<std::string>& v)
    {
        std::string s;
        for (const auto& x : v)
            s += x;
        return s;
    };
    auto hex = [](uint64_t h)
    {
        char buf[17];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
        return std::string(buf);
    };
    auto set_hash = [&](const std::set<std::string>& items)
    {
        uint64_t h = 0;
        for (const auto& s : items)
            h += fnv1a64(s);
        return hex(h);
    };
    auto counts_json = [](const std::vector<size_t>& v)
    {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i)
            s += std::string(i ? "," : "") + std::to_string(v[i]);
        return s + "]";
    };

    std::vector<std::pair<size_t, bool>> configs = { { 0, true }, { 1, true }, { 1, false } };
    if (max_width >= 2)
    {
        configs.emplace_back(2, true);
        configs.emplace_back(2, false);
    }
    size_t dropped_duplicates = 0;
    body += ",\"graphs\":{";
    for (size_t c = 0; c < configs.size(); ++c)
    {
        const auto [width, pruning] = configs[c];
        const auto tuple_graphs = datasets::TupleGraphImpl::create(space, certificate_maps, datasets::TupleGraphImpl::Options(width, pruning));
        body += std::string(c ? "," : "") + "\"w" + std::to_string(width) + (width == 0 ? "" : (pruning ? "p1" : "p0")) + "\":[";
        for (size_t r = 0; r < roots.size(); ++r)
        {
            const auto& tg = *tuple_graphs.at(roots[r]);
            const auto& tgraph = tg.get_graph();
            const auto& groups = tg.get_tuple_vertex_indices_grouped_by_distance();
            std::vector<size_t> distance(tgraph.get_num_vertices(), 0);
            for (size_t d = 0; d < groups.size(); ++d)
                for (const auto v : groups[d])
                    distance.at(v) = d;
            std::map<std::pair<size_t, IndexList>, std::vector<iw::AtomIndexList>> classes;
            if (pruning && width > 0)
                classes = tuple_classes(space, roots[r], width);
            // the vertex of a tuple graph by its distance and tuple: with dominance pruning the tuple is the smallest
            // (fewest atoms, then lexicographically smallest sorted atom names) of its class
            std::vector<std::string> vertex_id(tgraph.get_num_vertices());
            std::set<std::string> vertex_items, edge_items, layer_items;
            std::vector<std::set<std::string>> per_distance(groups.size()), per_layer_edges(groups.size());
            for (Index v = 0; v < tgraph.get_num_vertices(); ++v)
            {
                const auto& vertex = tgraph.get_vertex(v);
                auto tuple = names(graphs::get_atom_tuple(vertex));
                if (pruning && width > 0)
                {
                    auto problem_vertices = graphs::get_problem_vertices(vertex);
                    std::sort(problem_vertices.begin(), problem_vertices.end());
                    const auto it = classes.find({ distance[v], problem_vertices });
                    if (it == classes.end())
                        throw std::runtime_error("a tuple vertex without a novelty class");
                    const auto fork_tuple = tuple;
                    bool member = false;
                    for (const auto& t : it->second)
                    {
                        auto candidate = names(t);
                        member = member || candidate == fork_tuple;
                        if (&t == &it->second.front() || candidate.size() < tuple.size() || (candidate.size() == tuple.size() && candidate < tuple))
                            tuple = std::move(candidate);
                    }
                    if (!member)
                        throw std::runtime_error("the fork's tuple is not in its novelty class");
                }
                std::vector<std::string> problem_keys;
                for (const auto p : graphs::get_problem_vertices(vertex))
                    problem_keys.push_back(keys.at(p));
                std::sort(problem_keys.begin(), problem_keys.end());
                std::string pk;
                for (size_t i = 0; i < problem_keys.size(); ++i)
                    pk += (i ? "," : "") + problem_keys[i];
                vertex_id[v] = std::to_string(distance[v]) + ":" + concat(tuple);
                vertex_items.insert(vertex_id[v] + ":" + pk);
                per_distance[distance[v]].insert(vertex_id[v]);
            }
            for (const auto& e : tgraph.get_edges())
            {
                const auto item = vertex_id[e.get_source()] + ">" + vertex_id[e.get_target()];
                edge_items.insert(item);
                per_layer_edges.at(distance[e.get_target()]).insert(item);
            }
            // the problem vertices by distance; the fork leaves the root out of distance 0 at width 0
            const auto& pgroups = tg.get_problem_vertex_indices_grouped_by_distance();
            std::vector<size_t> pcounts;
            for (size_t d = 0; d < pgroups.size(); ++d)
            {
                std::set<std::string> layer;
                for (const auto p : pgroups[d])
                    layer.insert(std::to_string(d) + ":" + keys.at(p));
                if (d == 0)
                    layer.insert("0:" + keys.at(roots[r]));
                pcounts.push_back(layer.size());
                layer_items.insert(layer.begin(), layer.end());
            }
            std::vector<size_t> vcounts, ecounts;
            for (size_t d = 0; d < groups.size(); ++d)
            {
                vcounts.push_back(per_distance[d].size());
                if (d > 0)
                    ecounts.push_back(per_layer_edges[d].size());
            }
            // width 0: the fork opens distance 1 also when the root has no successor
            while (vcounts.size() > 1 && vcounts.back() == 0)
                vcounts.pop_back();
            pcounts.resize(vcounts.size());
            ecounts.resize(vcounts.size() - 1);
            dropped_duplicates += tgraph.get_num_vertices() - vertex_items.size();
            body += std::string(r ? "," : "") + "{\"n\":" + counts_json(vcounts) + ",\"m\":" + counts_json(ecounts) + ",\"p\":" +
                    counts_json(pcounts) + ",\"v\":" + jstr(set_hash(vertex_items)) + ",\"e\":" + jstr(set_hash(edge_items)) + ",\"q\":" +
                    jstr(set_hash(layer_items)) + "}";
        }
        body += "]";
    }
    body += "},\"dropped_duplicates\":" + std::to_string(dropped_duplicates);
    std::cout << "RESULT {\"algo\":\"tuple_graphs\"," << body << "}" << std::endl;
    return 0;
}

/// Class ids by first occurrence of each value.
template<class T>
std::vector<long> first_occurrence_classes(const std::vector<std::optional<T>>& values)
{
    std::map<T, long> ids;
    std::vector<long> out;
    for (const auto& v : values)
        out.push_back(v ? ids.emplace(*v, static_cast<long>(ids.size())).first->second : -1);
    return out;
}

std::string json_list(const std::vector<long>& v)
{
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i)
        s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

/// The fork's k-FWL certificates (kfwl::compute_certificate<K>, one IsomorphismTypeCompressionFunction per K) of the
/// object graphs of the sampled states with at most max_n vertices; a state is identified with the 64-bit loki hash of
/// its certificate (the hash of its identifying members). Adds "k<K>" (class ids, -1 above max_n), "k<K>_max_n",
/// "k<K>_seconds", "k<K>_state_seconds" (per state, null above max_n) and "k<K>_peak_rss_kb" to `body`.
template<size_t K>
void kfwl_classes(const std::vector<graphs::StaticGraph<graphs::Vertex<graphs::PropertyValue>, graphs::Edge<>>>& graphs, size_t max_n,
                  std::string& body)
{
    auto iso = graphs::kfwl::IsomorphismTypeCompressionFunction();
    std::vector<std::optional<size_t>> hashes;
    std::string state_seconds = "[";
    double secs = 0;
    for (const auto& g : graphs)
    {
        if (g.get_num_vertices() > max_n)
        {
            hashes.emplace_back();
            state_seconds += std::string(hashes.size() > 1 ? "," : "") + "null";
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto certificate = graphs::kfwl::compute_certificate<K>(g, iso);
        hashes.emplace_back(loki::Hash<graphs::kfwl::CertificateImpl<K>>()(*certificate));
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        secs += s;
        state_seconds += std::string(hashes.size() > 1 ? "," : "") + jnum(s);
    }
    state_seconds += "]";
    const std::string k = std::to_string(K);
    body += ",\"k" + k + "\":" + json_list(first_occurrence_classes(hashes)) + ",\"k" + k + "_max_n\":" + std::to_string(max_n) + ",\"k" + k +
            "_seconds\":" + jnum(secs) + ",\"k" + k + "_state_seconds\":" + state_seconds + ",\"k" + k + "_peak_rss_kb\":" + std::to_string(peak_rss_kb());
}

/// The fork's k-FWL (k = 2, 3, 4) and nauty classes of the object graphs (datasets::create_object_graph) of every
/// ceil(N / sample)-th state of the state space (remove_if_unsolvable = false, no symmetry pruning), and the size of
/// the fork's symmetry-reduced state space (nauty canonical forms).
int run_kfwl(const std::string& domain, const std::string& problem_file, uint32_t max_states, size_t sample, const std::array<size_t, 5>& max_n)
{
    Problem problem = ProblemImpl::create(domain, problem_file);
    auto options = SearchContextImpl::Options(
        SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF)));
    auto ss_options = datasets::StateSpaceImpl::Options();
    ss_options.remove_if_unsolvable = false;
    ss_options.max_num_states = max_states;
    auto result = datasets::StateSpaceImpl::create(SearchContextImpl::create(problem, options), ss_options);
    if (!result)
        throw std::runtime_error("no state space (max_states reached or a statically false goal)");
    const auto space = result->first;
    const auto& graph = space->get_graph();
    const size_t N = graph.get_num_vertices();
    const size_t step = (sample > 0 && N > sample) ? (N + sample - 1) / sample : 1;

    std::vector<std::string> keys;
    std::vector<long> sizes;
    std::vector<graphs::StaticGraph<graphs::Vertex<graphs::PropertyValue>, graphs::Edge<>>> object_graphs;
    std::vector<std::optional<graphs::nauty::SparseGraph>> canonical;  // optional: SparseGraph is not copy-assignable
    const auto t0 = std::chrono::steady_clock::now();
    for (Index v = 0; v < N; v += step)
    {
        const auto& state = graphs::get_state(graph.get_vertex(v));
        keys.push_back(state_key(problem, state));
        object_graphs.push_back(datasets::create_object_graph(state, *problem));
        sizes.push_back(static_cast<long>(object_graphs.back().get_num_vertices()));
        canonical.emplace_back(graphs::nauty::SparseGraph(object_graphs.back()).canonize());
    }
    const double nauty_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string body = "\"states\":" + std::to_string(N) + ",\"sample_step\":" + std::to_string(step) + ",\"keys\":[";
    for (size_t i = 0; i < keys.size(); ++i)
        body += std::string(i ? "," : "") + jstr(keys[i]);
    body += "],\"n\":" + json_list(sizes);
    UnorderedMap<graphs::nauty::SparseGraph, long> nauty_ids;  // loki::Hash / loki::EqualTo of the canonical forms
    std::vector<long> nauty_classes;
    for (const auto& c : canonical)
        nauty_classes.push_back(nauty_ids.emplace(*c, static_cast<long>(nauty_ids.size())).first->second);
    body += ",\"nauty\":" + json_list(nauty_classes) + ",\"nauty_seconds\":" + jnum(nauty_secs);
    if (max_n[2])
        kfwl_classes<2>(object_graphs, max_n[2], body);
    if (max_n[3])
        kfwl_classes<3>(object_graphs, max_n[3], body);
    if (max_n[4])
        kfwl_classes<4>(object_graphs, max_n[4], body);
    // the fork's symmetry-reduced state space (nauty)
    ss_options.symmetry_pruning = true;
    auto symmetric = datasets::StateSpaceImpl::create(SearchContextImpl::create(ProblemImpl::create(domain, problem_file), options), ss_options);
    body += ",\"symmetric_states\":" + (symmetric ? std::to_string(symmetric->first->get_graph().get_num_vertices()) : std::string("null"));
    std::cout << "RESULT {\"algo\":\"kfwl\"," << body << "}" << std::endl;
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string algo, hname, domain, problem, order, mode = "all_tested", features = "classical";
    uint32_t max_ms = 120000, max_states = UINT32_MAX, limit = UINT32_MAX, beam = UINT32_MAX;
    size_t k = 1, walks = 3, steps = 25;
    uint64_t seed = 1;
    size_t sample = 0, max_width = 2;
    long time_width = -1;
    bool time_pruning = true;
    std::array<size_t, 5> max_n { 0, 0, 1000, 64, 24 };  // per k: object graphs above are not certified
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
        else if (a == "--features")
            features = v;
        else if (a == "--k" || a == "--width")
            k = std::stoul(v);
        else if (a == "--limit")
            limit = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--beam")
            beam = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--beam-mode")
            mode = v;
        else if (a == "--tie-seed")
            g_beam.tie_seed = std::stol(v);
        else if (a == "--threads")
            g_beam.threads = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--chunk")
            g_beam.chunk = static_cast<uint32_t>(std::stoul(v));
        else if (a == "--iw1-knobs")
        {
            if (v == "precheck")
                g_beam.iw1 = 1;
            else if (v == "atom_first")
                g_beam.iw1 = 2;
            else if (v == "incremental")
                g_beam.iw1 = 3;
            else
            {
                std::cerr << "unknown IW(1) knobs " << v << "\n";
                return 2;
            }
        }
        else if (a == "--walks")
            walks = std::stoul(v);
        else if (a == "--steps")
            steps = std::stoul(v);
        else if (a == "--seed")
            seed = std::stoull(v);
        else if (a == "--symmetry")
        {
            if (v == "off")
                g_symmetry = SearchContextImpl::SymmetryPruning::OFF;
            else if (v == "wl1")
                g_symmetry = SearchContextImpl::SymmetryPruning::WL1;
            else
            {
                std::cerr << "unknown symmetry pruning " << v << "\n";
                return 2;
            }
        }
        else if (a == "--sample")
            sample = std::stoul(v);
        else if (a == "--max-width")
            max_width = std::stoul(v);
        else if (a == "--time-width")
            time_width = std::stol(v);
        else if (a == "--time-pruning")
            time_pruning = v != "0";
        else if (a == "--max-n2" || a == "--max-n3" || a == "--max-n4")
            max_n[a.back() - '0'] = std::stoul(v);
        else
        {
            std::cerr << "unknown argument " << a << "\n";
            return 2;
        }
    }
    const bool layered = algo == "iw" || algo == "brfs";
    if (algo.empty() || domain.empty() || problem.empty() || (layered ? order.empty() : algo != "walk_ground" && algo != "symmetry_states" && algo != "tuple_graphs" && algo != "kfwl" && hname.empty()))
    {
        std::cerr << "usage: search_fork --algo A (--h H | --order O [--k K] [--limit L] [--beam W] [--beam-mode M]) --domain D --problem P [--max-ms T] "
                     "[--max-states N] [--walks W] [--steps S] [--seed B]\n";
        return 2;
    }
    int rc = 2;
    try
    {
        if (algo == "kfwl")
            rc = run_kfwl(domain, problem, max_states, sample, max_n);
        else if (algo == "tuple_graphs")
            rc = run_tuple_graphs(domain, problem, max_states, sample, max_width, time_width, time_pruning);
        else if (algo == "walk_ground")
            rc = run_walk_ground(domain, problem, walks, steps, seed);
        else if (algo == "symmetry_states")
            rc = run_symmetry_states(domain, problem, max_ms, max_states);
        else if (algo == "walk_h")
            rc = run_walk_h(domain, problem, hname, walks, steps, seed, max_states);
        else
            rc = layered ? run_layered(domain, problem, algo, order, k, limit, beam, mode, max_ms, max_states)
                         : run_search(domain, problem, algo, hname, max_ms, max_states, k, features);
    }
    catch (const std::exception& ex)
    {
        std::cout << "ERROR " << jstr(ex.what()) << std::endl;
        std::_Exit(1);
    }
    std::cout.flush();
    std::_Exit(rc);  // skips the teardown of large repositories
}
