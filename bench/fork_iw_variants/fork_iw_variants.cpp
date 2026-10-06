/*
 * fork_iw_variants: one IW-family run of the C++ mimir fork (v0.16.3), printing one `RESULT_JSON {...}` line with its
 * fields, for comparison with mymyr_iw_variants. Built out of source against the fork install (CMakeLists.txt); the
 * fork's source tree is never touched. Successor generation is lifted (KPKC, symmetry pruning off), except --native
 * parallel rollouts (grounded, which the fork requires there).
 *
 *   fork_iw_variants --domain D --problem P --algo A [options]
 *
 * Algorithms:
 *   aiw        one brfs::find_solution pass with AbstractedNoveltyPruningStrategyImpl(width --k, --base,
 *              preserve goal atoms unless --no-preserve-goal, --keep-depth-one, the landmark graph of --landmarks with
 *              its disjunctive grouping, preserve landmark atoms unless --no-preserve-landmarks): AIW, BAIW, the
 *              projective_iw alias (--base --no-preserve-goal), abstracted LIW
 *   liw        iw::find_solution with landmark_novelty_graph (--landmarks), --lm-disjunctive, --lm-all-private
 *   rollout_iw rollout_iw::find_solution; --ordering in_order|randomized|dgaf|regression|mixed, --seed; randomized
 *              and mixed use SplitMix64 (the same generator, bounded draw and Fisher-Yates shuffle as mymyr's
 *              core/random.hpp) instead of std::mt19937_64 + std::shuffle, which are not portable; mixed shuffles and
 *              then applies the fork's own GoalRegressionRelevance strategy to the shuffled list (a stable sort by rank)
 *   rollouts   the per-rollout body of iw::find_rollouts_parallel (a private state repository with first-achiever and
 *              co-occurrence tracking, iw::find_solution with a randomized layer order and the tie seed), run serially
 *              per --seeds, with RandomizedLayerOrderingStrategyImpl replaced by a SplitMix64 shuffle of the layer;
 *              --max-next-layer-states. With --native T: the fork's own find_rollouts_parallel on T threads (grounded
 *              context, std::shuffle: timing only)
 *   portfolio  iw::find_solution_atomic_goal_portfolio, --workers K, --threads T (1: serial), --base-seed,
 *              --orderings kind:seed,... (deterministic kinds keep a serial run comparable)
 *   order_oracle  no search: for each "E <atoms>" line of --states FILE, the state of those fluent atoms in a fresh
 *              repository and its applicable actions in the lifted generator's order, written to --trace in the
 *              trace format below (the portfolio has no hook for a trace; its replay asks this oracle for the states
 *              mymyr_iw_variants --misses-out reports, until none is missing)
 * Common: --k, --timeout-ms, --goal-atom "(p a b)" (an atomic goal instead of the problem's; rollout_iw, rollouts,
 * portfolio), --max-states N, --landmarks none|lifted (LiftedFactLandmarkGenerator, default options),
 * --landmarks-out FILE (the graph as "F <atom>" and "D <atom> <atom> ..." lines, read by mymyr_iw_variants --landmarks),
 * --export-names (traces name objects as the text exports do, "o<i>"),
 * --trace FILE (bench/fork_iw_trace's format: "E <sorted atoms>" per expansion / materialization, then "G + <action>" or
 * "G - <action>" per transition in the fork's generation order; read by mymyr_iw_variants --order-from).
 */

#include <mimir/mimir.hpp>
#include <mimir/search/algorithms/iw/atomic_goal_portfolio.hpp>
#include <mimir/search/algorithms/iw/landmark_novelty_table.hpp>
#include <mimir/search/algorithms/iw/parallel_rollouts.hpp>
#include <mimir/search/algorithms/iw/pruning_strategy.hpp>
#include <mimir/search/algorithms/rollout_iw.hpp>
#include <mimir/search/algorithms/strategies/goal_strategy.hpp>
#include <mimir/search/algorithms/strategies/layer_ordering_strategy.hpp>
#include <mimir/search/landmarks/fact_landmark_graph.hpp>
#include <mimir/search/landmarks/lifted_fact_landmark_generator.hpp>
#include <mimir/search/state_repository.hpp>

#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>

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

/// mymyr core/random.hpp, bit for bit.
class SplitMix64
{
public:
    explicit SplitMix64(uint64_t seed) : m_state(seed) {}
    uint64_t next()
    {
        m_state += 0x9e3779b97f4a7c15ULL;
        uint64_t x = m_state;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }
    uint64_t bounded(uint64_t n)
    {
        unsigned __int128 m = static_cast<unsigned __int128>(next()) * n;
        uint64_t low = static_cast<uint64_t>(m);
        if (low < n)
        {
            const uint64_t threshold = (0 - n) % n;
            while (low < threshold)
            {
                m = static_cast<unsigned __int128>(next()) * n;
                low = static_cast<uint64_t>(m);
            }
        }
        return static_cast<uint64_t>(m >> 64);
    }
    template<class T>
    void shuffle(std::vector<T>& a)
    {
        for (size_t i = a.size(); i > 1; --i)
        {
            const size_t j = static_cast<size_t>(bounded(i));
            std::swap(a[i - 1], a[j]);
        }
    }

private:
    uint64_t m_state;
};

uint64_t fnv1a(const std::string& s, uint64_t h = 0xcbf29ce484222325ULL)
{
    for (unsigned char ch : s)
        h = (h ^ ch) * 0x100000001b3ULL;
    return h;
}

/// --export-names: objects are named as in the text exports mymyr reads for tasks its front end rejects
/// ("o<i>", i the object's position in get_problem_and_domain_objects()).
std::unordered_map<std::string, std::string> g_export_names;

const std::string& object_name(const Object& o)
{
    const auto it = g_export_names.find(o->get_name());
    return it == g_export_names.end() ? o->get_name() : it->second;
}

std::string atom_str(GroundAtom<FluentTag> a)
{
    std::string s = "(" + a->get_predicate()->get_name();
    for (auto o : a->get_objects())
        s += " " + object_name(o);
    return s + ")";
}

std::string action_str(GroundAction a)
{
    std::string s = "(" + a->get_action()->get_name();
    for (auto o : a->get_objects())
        s += " " + object_name(o);
    return s + ")";
}

std::string atoms_str(const State& state)
{
    std::vector<std::string> out;
    const auto& repos = state.get_problem().get_repositories();
    for (auto a : repos.get_ground_atoms_from_indices<FluentTag>(state.get_atoms<FluentTag>()))
        out.push_back(atom_str(a));
    std::sort(out.begin(), out.end());
    std::string r;
    for (const auto& s : out)
        r += (r.empty() ? "" : " ") + s;
    return r;
}

/// "(p a b)" -> the fluent ground atom of the problem.
GroundAtom<FluentTag> parse_atom(const Problem& problem, const std::string& text)
{
    std::string t = text;
    for (char& c : t)
        if (c == '(' || c == ')')
            c = ' ';
    std::istringstream in(t);
    std::string pred;
    in >> pred;
    const auto& preds = problem->get_domain()->get_name_to_predicate<FluentTag>();
    const auto it = preds.find(pred);
    if (it == preds.end())
        throw std::runtime_error("unknown fluent predicate in " + text);
    const auto objects_by_name = problem->get_name_to_problem_or_domain_object();
    ObjectList objs;
    std::string o;
    while (in >> o)
    {
        const auto jt = objects_by_name.find(o);
        if (jt == objects_by_name.end())
            throw std::runtime_error("unknown object in " + text);
        objs.push_back(jt->second);
    }
    return problem->get_or_create_ground_atom(it->second, objs);
}

GroundConjunctiveCondition atom_goal(const Problem& problem, const std::string& text)
{
    auto literals = GroundLiteralLists<StaticTag, FluentTag, DerivedTag> {};
    auto& fl = boost::hana::at_key(literals, boost::hana::type<FluentTag> {});
    fl.push_back(problem->get_or_create_ground_literal<FluentTag>(true, parse_atom(problem, text)));
    return problem->get_or_create_ground_conjunctive_condition(std::move(literals), GroundNumericConstraintList {});
}

/// The brfs trace of bench/fork_iw_trace, written to a file, statistics kept by the base.
class TraceHandler : public brfs::EventHandlerBase<TraceHandler>
{
    friend class brfs::EventHandlerBase<TraceHandler>;
    std::FILE* m_f;

    void on_expand_state_impl(const State& state) const { std::fprintf(m_f, "E %s\n", atoms_str(state).c_str()); }
    void on_expand_goal_state_impl(const State& state) const { std::fprintf(m_f, "X %s\n", atoms_str(state).c_str()); }
    void on_generate_state_impl(const State&, GroundAction, ContinuousCost, const State&) const {}
    void on_generate_state_in_search_tree_impl(const State&, GroundAction action, ContinuousCost, const State&) const
    {
        std::fprintf(m_f, "G + %s\n", action_str(action).c_str());
    }
    void on_generate_state_not_in_search_tree_impl(const State&, GroundAction action, ContinuousCost, const State&) const
    {
        std::fprintf(m_f, "G - %s\n", action_str(action).c_str());
    }
    void on_finish_g_layer_impl(uint32_t, uint64_t, uint64_t) const {}
    void on_start_search_impl(const State&) const { std::fprintf(m_f, "P\n"); }
    void on_end_search_impl(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) const {}
    void on_solved_impl(const Plan&) const {}
    void on_unsolvable_impl() const {}
    void on_exhausted_impl() const {}

public:
    TraceHandler(Problem problem, std::FILE* f) : brfs::EventHandlerBase<TraceHandler>(problem, false), m_f(f) {}
};

brfs::EventHandler make_brfs_handler(const Problem& problem, std::FILE* trace)
{
    if (trace)
        return std::make_shared<TraceHandler>(problem, trace);
    return brfs::DefaultEventHandlerImpl::create(problem, true);
}

/// RandomizedLayerOrderingStrategyImpl with SplitMix64 in place of std::mt19937_64 + std::shuffle.
class PortableRandomizedLayerOrdering : public ILayerOrderingStrategy
{
    SplitMix64 m_rng;

public:
    explicit PortableRandomizedLayerOrdering(uint64_t seed) : m_rng(seed) {}
    void order_layer(StateList& states, DiscreteCost) override { m_rng.shuffle(states); }
};

/// Rollout IW action orderings: the fork's deterministic ones, portable randomized ones, and a trace of the
/// generation order (the order of the action list a node's rank() receives) per materialized state.
class PortableOrdering : public rollout_iw::IActionOrderingStrategy
{
public:
    PortableOrdering(std::string kind, uint64_t seed, const Problem& problem, std::optional<GroundConjunctiveCondition> goal, std::FILE* trace) :
        m_kind(std::move(kind)),
        m_rng(seed),
        m_trace(trace)
    {
        using K = rollout_iw::ActionOrderingKind;
        if (m_kind == "dgaf")
            m_inner = rollout_iw::create_action_ordering_strategy(rollout_iw::ActionOrderingConfiguration(K::DIRECT_GOAL_ACHIEVER_FIRST), problem, goal);
        else if (m_kind == "regression" || m_kind == "mixed")
            m_inner = rollout_iw::create_action_ordering_strategy(rollout_iw::ActionOrderingConfiguration(K::GOAL_REGRESSION_RELEVANCE), problem, goal);
        else if (m_kind != "in_order" && m_kind != "randomized")
            throw std::runtime_error("unknown ordering " + m_kind);
    }

    void rank(const State& state, const std::vector<GroundAction>& actions, std::vector<uint32_t>& out) override
    {
        if (m_trace)
        {
            // rank() runs at every visit of a node: test the state's index (one repository per search) and format
            // the state only at its first visit, so that tracing does not slow the search down
            const auto index = static_cast<size_t>(state.get_index());
            if (index >= m_seen.size())
                m_seen.resize(index + 1, 0);
            if (!m_seen[index])
            {
                m_seen[index] = 1;
                std::fprintf(m_trace, "E %s\n", atoms_str(state).c_str());
                for (const auto& a : actions)
                    std::fprintf(m_trace, "G + %s\n", action_str(a).c_str());
            }
        }
        out.resize(actions.size());
        std::iota(out.begin(), out.end(), 0u);
        if (m_kind == "in_order")
            return;
        if (m_kind == "randomized")
        {
            m_rng.shuffle(out);
            return;
        }
        if (m_kind == "mixed")
        {
            // shuffle, then the fork's regression ranking of the shuffled list (a stable sort by schema rank)
            std::vector<uint32_t> perm(actions.size());
            std::iota(perm.begin(), perm.end(), 0u);
            m_rng.shuffle(perm);
            std::vector<GroundAction> shuffled;
            shuffled.reserve(actions.size());
            for (uint32_t p : perm)
                shuffled.push_back(actions[p]);
            std::vector<uint32_t> tmp;
            m_inner->rank(state, shuffled, tmp);
            for (size_t i = 0; i < tmp.size(); ++i)
                out[i] = perm[tmp[i]];
            return;
        }
        m_inner->rank(state, actions, out);
    }

private:
    std::string m_kind;
    SplitMix64 m_rng;
    std::FILE* m_trace;
    rollout_iw::ActionOrderingStrategy m_inner;
    std::vector<uint8_t> m_seen;  // per state index: already traced
};

struct Counts
{
    uint64_t expanded = 0, generated = 0, generated_in_tree = 0, passes = 0;
    std::string per_pass;
    void add(const brfs::Statistics& st, size_t arity)
    {
        expanded += st.get_num_expanded();
        generated += st.get_num_generated();
        generated_in_tree += st.get_num_generated_in_search_tree();
        ++passes;
        if (!per_pass.empty())
            per_pass += ";";
        per_pass += std::to_string(arity) + ":" + std::to_string(st.get_num_expanded()) + "/" + std::to_string(st.get_num_generated()) + "/" +
                    std::to_string(st.get_num_generated_in_search_tree());
    }
};

std::string rollout_stats_json(const rollout_iw::Statistics& s)
{
    std::ostringstream o;
    o << "{\"rollouts\":" << s.num_rollouts << ",\"generated\":" << s.num_generated_states << ",\"expanded\":" << s.num_expanded_nodes
      << ",\"feature_depth_improvements\":" << s.num_feature_depth_improvements << ",\"case1\":" << s.num_case_1 << ",\"case2\":" << s.num_case_2
      << ",\"case3\":" << s.num_case_3 << ",\"case4\":" << s.num_case_4 << ",\"solved_propagations\":" << s.num_solved_propagations
      << ",\"dead_ends\":" << s.num_dead_ends << ",\"depth_bound_prunings\":" << s.num_depth_bound_prunings
      << ",\"incumbent_bound_prunings\":" << s.num_incumbent_bound_prunings << ",\"max_rollout_depth\":" << s.max_rollout_depth
      << ",\"tree_nodes\":" << s.num_tree_nodes << "}";
    return o.str();
}

rollout_iw::ActionOrderingKind ordering_kind(const std::string& k)
{
    using K = rollout_iw::ActionOrderingKind;
    if (k == "in_order")
        return K::IN_ORDER;
    if (k == "randomized")
        return K::RANDOMIZED;
    if (k == "dgaf")
        return K::DIRECT_GOAL_ACHIEVER_FIRST;
    if (k == "regression")
        return K::GOAL_REGRESSION_RELEVANCE;
    if (k == "mixed")
        return K::MIXED_REGRESSION_RANDOM;
    throw std::runtime_error("unknown ordering " + k);
}

/// A rollout's private repository, described portably: sorted atom names and hashes over them.
std::string repository_json(StateRepositoryImpl& repo, const ApplicableActionGenerator& generator, const Problem& problem, bool landing, bool co)
{
    const auto& repos = problem->get_repositories();
    auto name = [&](Index i) { return atom_str(repos.get_ground_atom<FluentTag>(i)); };
    std::vector<std::string> reached;
    for (const auto i : repo.get_reached_fluent_ground_atoms_bitset())
        reached.push_back(name(i));
    std::sort(reached.begin(), reached.end());
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const auto& s : reached)
        h = fnv1a(s + "\n", h);
    size_t derived = 0;
    for (const auto i : repo.get_reached_derived_ground_atoms_bitset())
    {
        (void) i;
        ++derived;
    }
    std::ostringstream o;
    o << "\"num_states\":" << repo.get_state_count() << ",\"reached_fluent\":" << reached.size() << ",\"reached_fluent_hash\":\"" << std::hex << h
      << std::dec << "\",\"reached_derived\":" << derived;
    if (landing)
    {
        // landing state per atom: the first created state holding it, as its sorted atom list
        const auto& by_atom = repo.get_first_achiever_state_by_atom();
        std::unordered_map<Index, std::string> desc;
        std::vector<std::string> pairs;
        size_t dead = 0;
        for (size_t a = 0; a < by_atom.size(); ++a)
        {
            if (by_atom[a] == MAX_INDEX)
                continue;
            auto it = desc.find(by_atom[a]);
            if (it == desc.end())
            {
                const auto state = repo.get_state(*repo.get_packed_state(by_atom[a]));
                it = desc.emplace(by_atom[a], atoms_str(state)).first;
                auto gen = generator->create_applicable_action_generator(state);
                dead += gen.begin() == gen.end();
            }
            pairs.push_back(name(a) + " -> " + it->second);
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<std::string> states;
        for (const auto& [i, s] : desc)
            states.push_back(s);
        std::sort(states.begin(), states.end());
        uint64_t hp = 0xcbf29ce484222325ULL, hs = hp;
        for (const auto& s : pairs)
            hp = fnv1a(s + "\n", hp);
        for (const auto& s : states)
            hs = fnv1a(s + "\n", hs);
        o << ",\"landing_states\":" << states.size() << ",\"landing_dead_ends\":" << dead << ",\"landing_states_hash\":\"" << std::hex << hs
          << "\",\"landing_by_atom_hash\":\"" << hp << std::dec << "\"";
    }
    if (co)
    {
        const auto& rows = repo.get_co_occurrence_by_atom();
        std::vector<std::string> lines;
        size_t pairs = 0;
        for (size_t a = 0; a < rows.size(); ++a)
        {
            std::vector<std::string> row;
            for (const auto b : rows[a])
                row.push_back(name(b));
            if (row.empty())
                continue;
            std::sort(row.begin(), row.end());
            pairs += row.size();
            std::string l = name(a) + ":";
            for (const auto& s : row)
                l += " " + s;
            lines.push_back(l);
        }
        std::sort(lines.begin(), lines.end());
        uint64_t hc = 0xcbf29ce484222325ULL;
        for (const auto& s : lines)
            hc = fnv1a(s + "\n", hc);
        o << ",\"co_occurrence_pairs\":" << pairs << ",\"co_occurrence_hash\":\"" << std::hex << hc << std::dec << "\"";
    }
    return o.str();
}
}  // namespace

int main(int argc, char** argv)
{
    std::string algo, domain, problem_path, landmarks = "none", landmarks_out, ordering = "in_order", goal_atom, seeds_arg, orderings_arg,
                                           trace_path, states_path;
    size_t k = 1;
    uint32_t timeout_ms = 120000, max_next_layer_states = std::numeric_limits<uint32_t>::max(), workers = 4, threads = 1, native = 0;
    uint64_t seed = 0, base_seed = 0, max_states = std::numeric_limits<uint64_t>::max(), max_rollouts = std::numeric_limits<uint64_t>::max();
    bool base = false, preserve_goal = true, keep_depth_one = false, preserve_landmarks = true, lm_disjunctive = false, lm_all_private = false,
         report = true, export_names = false;
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
        if (a == "--algo")
            algo = next();
        else if (a == "--k")
            k = std::stoul(next());
        else if (a == "--domain")
            domain = next();
        else if (a == "--problem")
            problem_path = next();
        else if (a == "--timeout-ms")
            timeout_ms = std::stoul(next());
        else if (a == "--base")
            base = true;
        else if (a == "--no-preserve-goal")
            preserve_goal = false;
        else if (a == "--keep-depth-one")
            keep_depth_one = true;
        else if (a == "--no-preserve-landmarks")
            preserve_landmarks = false;
        else if (a == "--landmarks")
            landmarks = next();
        else if (a == "--landmarks-out")
            landmarks_out = next();
        else if (a == "--lm-disjunctive")
            lm_disjunctive = true;
        else if (a == "--lm-all-private")
            lm_all_private = true;
        else if (a == "--ordering")
            ordering = next();
        else if (a == "--seed")
            seed = std::stoull(next());
        else if (a == "--goal-atom")
            goal_atom = next();
        else if (a == "--seeds")
            seeds_arg = next();
        else if (a == "--max-next-layer-states")
            max_next_layer_states = std::stoul(next());
        else if (a == "--no-report")
            report = false;
        else if (a == "--native")
            native = std::stoul(next());
        else if (a == "--workers")
            workers = std::stoul(next());
        else if (a == "--threads")
            threads = std::stoul(next());
        else if (a == "--base-seed")
            base_seed = std::stoull(next());
        else if (a == "--orderings")
            orderings_arg = next();
        else if (a == "--max-states")
            max_states = std::stoull(next());
        else if (a == "--max-rollouts")
            max_rollouts = std::stoull(next());
        else if (a == "--trace")
            trace_path = next();
        else if (a == "--states")
            states_path = next();
        else if (a == "--export-names")
            export_names = true;
        else
        {
            std::cerr << "unknown arg " << a << "\n";
            return 2;
        }
    }
    if (algo.empty() || domain.empty() || problem_path.empty())
    {
        std::cerr << "usage: fork_iw_variants --algo {aiw,liw,rollout_iw,rollouts,portfolio} --domain D --problem P [options]\n";
        return 2;
    }
    std::FILE* trace = nullptr;
    if (!trace_path.empty())
    {
        trace = std::fopen(trace_path.c_str(), "w");
        if (!trace)
        {
            std::cerr << "cannot open " << trace_path << "\n";
            return 2;
        }
    }

    std::ostringstream js;
    js.precision(9);
    double t0 = now_s();
    auto problem = ProblemImpl::create(domain, problem_path);
    const double parse_s = now_s() - t0;
    t0 = now_s();
    auto context = SearchContextImpl::create(
        problem,
        SearchContextImpl::Options(SearchContextImpl::LiftedOptions(SearchContextImpl::LiftedOptions::KPKCOptions(SearchContextImpl::SymmetryPruning::OFF))));
    const double context_s = now_s() - t0;
    if (export_names)
    {
        const auto& objs = problem->get_problem_and_domain_objects();
        for (size_t i = 0; i < objs.size(); ++i)
            g_export_names.emplace(objs[i]->get_name(), "o" + std::to_string(i));
    }

    if (goal_atom == "@first" || goal_atom == "@last")
    {
        // the smallest / largest name among the positive fluent goal atoms (mymyr_iw_variants picks the same)
        std::vector<std::string> names;
        const auto& repos = problem->get_repositories();
        for (const auto i : problem->get_goal_condition()->get_compressed_precondition<PositiveTag, FluentTag>()->compressed_range())
            names.push_back(atom_str(repos.get_ground_atom<FluentTag>(i)));
        if (names.empty())
        {
            std::cerr << "no positive fluent goal atom\n";
            return 3;
        }
        std::sort(names.begin(), names.end());
        goal_atom = goal_atom == "@first" ? names.front() : names.back();
    }
    std::optional<GroundConjunctiveCondition> goal;
    if (!goal_atom.empty())
        goal = atom_goal(problem, goal_atom);

    landmarks::FactLandmarkGraph graph = nullptr;
    double setup_s = 0;
    if (landmarks == "lifted")
    {
        t0 = now_s();
        graph = landmarks::LiftedFactLandmarkGenerator::create(problem);
        setup_s = now_s() - t0;
        if (!landmarks_out.empty())
        {
            std::FILE* f = std::fopen(landmarks_out.c_str(), "w");
            const auto& repos = problem->get_repositories();
            for (const auto i : graph->get_landmark_atom_indices())
                std::fprintf(f, "F %s\n", atom_str(repos.get_ground_atom<FluentTag>(i)).c_str());
            for (const auto& set : graph->get_disjunctive_landmarks())
            {
                std::string line = "D";
                for (const auto i : set)
                    line += " " + atom_str(repos.get_ground_atom<FluentTag>(i));
                std::fprintf(f, "%s\n", line.c_str());
            }
            std::fclose(f);
        }
    }
    else if (landmarks != "none")
    {
        std::cerr << "unknown --landmarks " << landmarks << "\n";
        return 2;
    }

    if (algo == "order_oracle")
    {
        if (!trace || states_path.empty())
        {
            std::cerr << "order_oracle needs --states and --trace\n";
            return 2;
        }
        std::ifstream in(states_path);
        const auto repo = context->get_state_repository();
        const auto generator = context->get_applicable_action_generator();
        std::string line;
        size_t n = 0;
        while (std::getline(in, line))
        {
            if (line.size() < 1 || line[0] != 'E')
                continue;
            GroundAtomList<FluentTag> atoms;
            for (size_t open = line.find('('); open != std::string::npos; open = line.find('(', open + 1))
                atoms.push_back(parse_atom(problem, line.substr(open, line.find(')', open) - open + 1)));
            const auto state = repo->get_or_create_state(atoms, FlatDoubleList {}).first;
            std::fprintf(trace, "E %s\n", atoms_str(state).c_str());
            for (const auto& action : generator->create_applicable_action_generator(state))
                std::fprintf(trace, "G + %s\n", action_str(action).c_str());
            ++n;
        }
        std::fclose(trace);
        std::cout << "RESULT_JSON {\"lib\":\"fork\",\"algo\":\"order_oracle\",\"states\":" << n << "}" << std::endl;
        std::fflush(stdout);
        std::_Exit(0);
    }

    SearchResult result;
    Counts c;
    double search_s = 0;
    std::string extra;
    if (algo == "aiw")
    {
        auto eh = make_brfs_handler(problem, trace);
        auto opts = brfs::Options();
        opts.event_handler = eh;
        opts.max_time_in_ms = timeout_ms;
        if (max_states != std::numeric_limits<uint64_t>::max())
            opts.max_num_states = static_cast<uint32_t>(max_states);
        auto grouping = iw::LandmarkGrouping {};
        if (graph && lm_disjunctive)
        {
            grouping.disjunctive_landmarks = graph->get_disjunctive_landmarks();
            if (lm_all_private)
                grouping.mode = iw::LandmarkGroupingMode::ALL_PRIVATE;
        }
        opts.pruning_strategy =
            iw::AbstractedNoveltyPruningStrategyImpl::create(problem, k, base, preserve_goal, keep_depth_one, graph, grouping, preserve_landmarks);
        t0 = now_s();
        result = brfs::find_solution(context, opts);
        search_s = now_s() - t0;
        c.add(eh->get_statistics(), k);
    }
    else if (algo == "liw")
    {
        auto iw_eh = iw::DefaultEventHandlerImpl::create(problem, true);
        auto brfs_eh = make_brfs_handler(problem, trace);
        auto opts = iw::Options();
        opts.max_arity = k;
        opts.iw_event_handler = iw_eh;
        opts.brfs_event_handler = brfs_eh;
        opts.max_time_in_ms = timeout_ms;
        if (max_states != std::numeric_limits<uint64_t>::max())
            opts.max_num_states = static_cast<uint32_t>(max_states);
        opts.landmark_novelty_graph = graph;
        opts.landmark_novelty_disjunctive = lm_disjunctive;
        opts.landmark_novelty_all_private = lm_all_private;
        t0 = now_s();
        result = iw::find_solution(context, opts);
        search_s = now_s() - t0;
        const auto& by_arity = iw_eh->get_statistics().get_brfs_statistics_by_arity();
        for (size_t a = 0; a < by_arity.size(); ++a)
            c.add(by_arity[a], a);
    }
    else if (algo == "rollout_iw")
    {
        auto opts = rollout_iw::Options();
        opts.seed = seed;
        opts.goal_condition = goal;
        opts.max_time_in_ms = timeout_ms;
        opts.max_num_states = max_states;
        opts.max_rollouts = max_rollouts;
        opts.action_ordering = std::make_shared<PortableOrdering>(ordering, seed, problem, goal, trace);
        t0 = now_s();
        auto r = rollout_iw::find_solution(context, opts);
        search_s = now_s() - t0;
        result = r.search_result;
        c.expanded = r.statistics.num_expanded_nodes;
        c.generated = r.statistics.num_generated_states;
        extra += ",\"rollout\":" + rollout_stats_json(r.statistics) + ",\"root_solved\":" + (r.root_solved ? "true" : "false") + ",\"stop_reason\":\"" +
                 r.stop_reason + "\",\"num_rollouts\":" + std::to_string(r.statistics.num_rollouts) + ",\"num_tree_nodes\":" +
                 std::to_string(r.statistics.num_tree_nodes);
    }
    else if (algo == "rollouts")
    {
        std::vector<uint64_t> seeds;
        {
            std::istringstream in(seeds_arg);
            std::string s;
            while (std::getline(in, s, ','))
                if (!s.empty())
                    seeds.push_back(std::stoull(s));
        }
        if (native)
        {
            // the fork's own function (grounded, std::shuffle): timing only
            auto gproblem = ProblemImpl::create(domain, problem_path);
            auto gcontext = SearchContextImpl::create(gproblem, SearchContextImpl::Options());
            auto po = iw::ParallelRolloutOptions();
            po.seeds = seeds;
            po.num_threads = native;
            po.options.max_arity = k;
            po.options.max_time_in_ms = timeout_ms;
            po.options.max_next_layer_states = max_next_layer_states;
            if (goal_atom.size())
                po.options.goal_strategy = ProblemGoalStrategyImpl::create(gproblem, atom_goal(gproblem, goal_atom));
            po.report_landing_states = report;
            po.report_co_occurrence = report;
            t0 = now_s();
            const auto rs = iw::find_rollouts_parallel(gcontext, po);
            search_s = now_s() - t0;
            size_t solved = 0;
            for (const auto& r : rs)
                solved += r.status == SearchStatus::SOLVED;
            extra += ",\"native_threads\":" + std::to_string(native) + ",\"rollouts_solved\":" + std::to_string(solved);
            result.status = SearchStatus::IN_PROGRESS;
        }
        else
        {
            const auto generator = context->get_applicable_action_generator();
            const auto axiom_evaluator = context->get_state_repository()->get_axiom_evaluator();
            std::string list;
            double searching = 0;  // the rollouts without the JSON (names and hashes) written after each
            for (const uint64_t s : seeds)
            {
                const double r0 = now_s();
                auto repo = StateRepositoryImpl::create(axiom_evaluator, StateRepositoryImpl::PrivateInterningTables {});
                if (report)
                {
                    repo->enable_first_achiever_tracking();
                    repo->enable_co_occurrence_tracking();
                }
                auto rctx = SearchContextImpl::create(problem, generator, repo);
                auto iw_eh = iw::DefaultEventHandlerImpl::create(problem, true);
                auto brfs_eh = make_brfs_handler(problem, trace);
                auto opts = iw::Options();
                opts.max_arity = k;
                opts.iw_event_handler = iw_eh;
                opts.brfs_event_handler = brfs_eh;
                opts.max_time_in_ms = timeout_ms;
                opts.max_next_layer_states = max_next_layer_states;
                opts.layer_ordering_strategy = std::make_shared<PortableRandomizedLayerOrdering>(s);
                opts.randomize_equal_score_ties = true;
                opts.equal_score_tie_seed = s;
                if (goal)
                    opts.goal_strategy = ProblemGoalStrategyImpl::create(problem, goal);
                opts.start_state = repo->get_or_create_initial_state().first;
                const auto r = iw::find_solution(rctx, opts);
                searching += now_s() - r0;
                Counts rc;
                const auto& by_arity = iw_eh->get_statistics().get_brfs_statistics_by_arity();
                for (size_t a = 0; a < by_arity.size(); ++a)
                    rc.add(by_arity[a], a);
                std::ostringstream o;
                o << "{\"seed\":" << s << ",\"status\":\"" << status_name(r.status) << "\",\"per_pass\":\"" << rc.per_pass << "\"";
                if (r.plan)
                    o << ",\"plan_len\":" << r.plan->get_actions().size();
                o << "," << repository_json(*repo, generator, problem, report, report) << "}";
                list += (list.empty() ? "" : ",") + o.str();
            }
            search_s = searching;
            extra += ",\"rollouts\":[" + list + "]";
            result.status = SearchStatus::IN_PROGRESS;
        }
    }
    else if (algo == "portfolio")
    {
        auto po = iw::AtomicGoalPortfolioOptions();
        if (!goal_atom.empty())
            po.atomic_goal_atoms.push_back(parse_atom(problem, goal_atom));
        po.num_rollout_workers = workers;
        po.num_threads = threads;
        po.base_seed = base_seed;
        po.max_time_in_ms = timeout_ms;
        {
            std::istringstream in(orderings_arg);
            std::string s;
            while (std::getline(in, s, ','))
            {
                if (s.empty())
                    continue;
                const auto colon = s.find(':');
                po.rollout_orderings.emplace_back(ordering_kind(s.substr(0, colon)), colon == std::string::npos ? 0 : std::stoull(s.substr(colon + 1)));
            }
        }
        t0 = now_s();
        const auto r = iw::find_solution_atomic_goal_portfolio(context, po);
        search_s = now_s() - t0;
        result.status = r.status;
        result.plan = r.plan;
        c.expanded = r.iw_statistics.get_num_expanded();
        c.generated = r.iw_statistics.get_num_generated();
        c.generated_in_tree = r.iw_statistics.get_num_generated_in_search_tree();
        std::string stats, statuses;
        for (size_t i = 0; i < r.rollout_statistics.size(); ++i)
        {
            stats += (i ? "," : "") + rollout_stats_json(r.rollout_statistics[i]);
            statuses += std::string(i ? "," : "") + "\"" + status_name(r.rollout_statuses[i]) + "\"";
        }
        extra += ",\"plan_length\":" + std::to_string(r.plan ? r.plan_length : 0) + ",\"certified_optimal\":" + (r.certified_optimal ? "true" : "false") +
                 ",\"iw_lower_bound\":" + std::to_string(r.iw_lower_bound) + ",\"iw_completed_depth\":" + std::to_string(r.iw_completed_depth) +
                 ",\"winning_worker\":" + std::to_string(r.winning_worker) + ",\"certifier_status\":\"" + status_name(r.certifier_status) +
                 "\",\"total_expansions\":" + std::to_string(r.total_expansions) + ",\"stop_reason\":\"" + r.stop_reason + "\",\"rollout_statistics\":[" +
                 stats + "],\"rollout_statuses\":[" + statuses + "]";
    }
    else
    {
        std::cerr << "unknown algo " << algo << "\n";
        return 2;
    }
    if (trace)
        std::fclose(trace);

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    js << "{\"lib\":\"fork\",\"mode\":\"lifted\",\"algo\":\"" << algo << "\",\"k\":" << k << ",\"status\":\"" << status_name(result.status) << "\""
       << ",\"parse_s\":" << parse_s << ",\"context_s\":" << context_s << ",\"setup_s\":" << setup_s << ",\"search_s\":" << search_s
       << ",\"expanded\":" << c.expanded << ",\"generated\":" << c.generated << ",\"generated_in_tree\":" << c.generated_in_tree << ",\"passes\":" << c.passes
       << ",\"per_pass\":\"" << c.per_pass << "\"";
    if (graph)
        js << ",\"landmarks\":" << graph->get_landmark_atom_indices().size() << ",\"disjunctive_landmarks\":" << graph->get_disjunctive_landmarks().size();
    if (result.plan)
        js << ",\"plan_len\":" << result.plan->get_actions().size() << ",\"plan_cost\":" << result.plan->get_cost();
    if (!goal_atom.empty())
        js << ",\"goal_atom\":\"" << goal_atom << "\"";
    js << extra << ",\"ru_maxrss_mb\":" << (ru.ru_maxrss / 1024.0) << "}";
    std::cout << "RESULT_JSON " << js.str() << std::endl;
    std::fflush(stdout);
    std::_Exit(0);
}
