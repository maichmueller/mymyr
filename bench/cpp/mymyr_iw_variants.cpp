// mymyr_iw_variants: one IW-family run, printing one JSON line with the fields of bench/fork_iw_variants (the fork's status names
// and per_pass "arity:expanded/generated/generated_in_tree", plan_len, plan_cost, and per algorithm the fork's
// statistics), plus mymyr's own (plan: the solved plan's action names, per run and per rollout).
//
//   mymyr_iw_variants (task.txt | --domain D.pddl --problem P.pddl) --algo aiw|liw|rollout_iw|rollouts|portfolio [options]
//
// The options are fork_iw_variants's: --k, --base, --no-preserve-goal, --keep-depth-one, --no-preserve-landmarks,
// --landmarks FILE (fork_iw_variants --landmarks-out: "F <atom>" fact landmarks, "D <atom> ..." disjunctive sets),
// --lm-disjunctive, --lm-all-private, --ordering in_order|randomized|dgaf|regression|mixed, --seed, --goal-atom "(p a)",
// --seeds a,b,..., --threads T, --max-next-layer-states N, --no-report, --workers K, --base-seed S,
// --orderings kind:seed,..., --max-states N, --max-rollouts N, --timeout-ms T, plus
// --order-from FILE (a fork trace: each expanded / materialized state's successors in the fork's generation order,
// through the searches' successor_order hooks; order_misses counts states and actions the file does not list),
// --misses-out FILE (with --order-from: every state the file does not list, once, as an "E <atoms>" line, for
// fork_iw_variants --algo order_oracle) and --reps N (repeat the search, report the best search_s; timing).

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/task/task.hpp"
#if defined(MYMYR_HAVE_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
    std::fprintf(stderr, "error: %s\nusage: mymyr_iw_variants (task.txt | --domain D --problem P) --algo aiw|liw|rollout_iw|rollouts|portfolio [options]\n",
                 msg.c_str());
    std::exit(2);
}

u64 fnv1a(std::string_view s, u64 h = 0xcbf29ce484222325ULL)
{
    for (unsigned char ch : s)
        h = (h ^ ch) * 0x100000001b3ULL;
    return h;
}

std::string hex(u64 h)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string json_escape(const std::string& s)
{
    std::string e;
    for (char ch : s)
        e += ch == '"' ? std::string("\\\"") : std::string(1, ch);
    return e;
}

/// Atom names <-> canonical ids.
class Names
{
public:
    explicit Names(const Task& task) : m_task(task)
    {
        const formalism::TaskData& t = task.data();
        for (u32 p = 0; p < t.predicates.size(); ++p)
            m_pred.emplace(std::string(t.str(t.predicates[p].name)), p);
        for (u32 o = 0; o < t.objects.size(); ++o)
            m_obj.emplace(std::string(t.str(t.objects[o].name)), o);
    }
    /// "(p a b)" -> canonical id; nullopt when unknown or not a fluent atom.
    [[nodiscard]] std::optional<CanonicalAtom> parse(const std::string& text) const
    {
        std::string s = text;
        for (char& c : s)
            if (c == '(' || c == ')')
                c = ' ';
        std::istringstream in(s);
        std::string p, o;
        in >> p;
        const auto it = m_pred.find(p);
        if (it == m_pred.end())
            return std::nullopt;
        std::vector<u32> args;
        while (in >> o)
        {
            const auto jt = m_obj.find(o);
            if (jt == m_obj.end())
                return std::nullopt;
            args.push_back(jt->second);
        }
        const CanonicalLayout& L = m_task.atoms().layout();
        if (m_task.data().predicate(PredicateId{it->second}).kind != formalism::PredKind::Fluent)
            return std::nullopt;
        const CanonicalAtom c = L.encode(it->second, args.data());
        if (c >= L.fluent_count)
            return std::nullopt;
        return c;
    }
    [[nodiscard]] std::string name(CanonicalAtom c) const { return m_task.format(SlotId{m_task.atoms().intern(c)}); }
    /// A plan as a JSON list of action names "(schema o1 o2)" (Python's str(Action)).
    [[nodiscard]] std::string plan_json(const std::vector<Action>& plan) const
    {
        const formalism::TaskData& t = m_task.data();
        std::string o = "[";
        for (usize i = 0; i < plan.size(); ++i)
        {
            o += (i ? ",\"(" : "\"(") + std::string(t.str(t.schemas[plan[i].schema.v].name));
            for (ObjectId x : plan[i].binding)
                o += " " + std::string(t.str(t.objects[x.v].name));
            o += ")\"";
        }
        return o + "]";
    }

private:
    const Task& m_task;
    std::unordered_map<std::string, u32> m_pred, m_obj;
};

/// The successor order of a fork trace (bench/fork_iw_trace format): per expanded state (hash of its sorted atom
/// list), the hashes of its actions in the fork's generation order. Thread-safe (parallel rollouts).
class TraceOrder
{
public:
    TraceOrder(const Task& task, const std::string& path) : m_task(task)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("cannot open " + path);
        // The fork's order is a function of the state, so a later expansion of a state lists the same actions in the
        // same order; it can list more (an expansion cut short by a truncated layer lists a prefix), which extend it.
        std::vector<u64>* cur = nullptr;
        bool fresh = false;
        std::string line;
        while (std::getline(in, line))
        {
            std::string_view v(line);
            while (!v.empty() && (v.back() == '\n' || v.back() == '\r'))
                v.remove_suffix(1);
            if (!v.empty() && v[0] == 'E' && (v.size() == 1 || v[1] == ' '))
            {
                const auto r = m_order.try_emplace(fnv1a(v.size() > 2 ? v.substr(2) : std::string_view{}));
                cur = &r.first->second;
                fresh = r.second;
            }
            else if (v.size() > 4 && v[0] == 'G' && cur)
            {
                const u64 h = fnv1a(v.substr(4));
                if (fresh || std::find(cur->begin(), cur->end(), h) == cur->end())
                    cur->push_back(h);
            }
        }
    }

    void operator()(StateView s, std::span<const Action> actions, std::vector<u32>& order)
    {
        if (actions.empty())
            return;  // a dead end has no order (the fork's rollout traces never list one)
        std::vector<std::string> atoms = m_task.format_atoms(s);
        std::sort(atoms.begin(), atoms.end());
        std::string key;
        for (const std::string& a : atoms)
            key += (key.empty() ? "" : " ") + a;
        const auto it = m_order.find(fnv1a(key));
        if (it == m_order.end())
        {
            ++state_misses;
            if (misses_out)
            {
                const std::lock_guard lock(m_mutex);
                if (m_missed.insert(fnv1a(key)).second)
                    std::fprintf(misses_out, "E %s\n", key.c_str());
            }
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

    std::atomic<u64> state_misses{0}, action_misses{0};
    std::FILE* misses_out = nullptr;

private:
    const Task& m_task;
    std::unordered_map<u64, std::vector<u64>> m_order;
    std::mutex m_mutex;
    std::unordered_set<u64> m_missed;
};

using OrderHook = std::function<void(StateView, std::span<const Action>, std::vector<u32>&)>;

struct Sums
{
    u64 expanded = 0, generated = 0, in_tree = 0, passes = 0;
    std::string per_pass;
    void add(const IwPassStatistics& p, u32 arity)
    {
        expanded += p.expanded;
        generated += p.generated;
        in_tree += p.generated_in_tree;
        ++passes;
        if (!per_pass.empty())
            per_pass += ";";
        per_pass += std::to_string(arity) + ":" + std::to_string(p.expanded) + "/" + std::to_string(p.generated) + "/" + std::to_string(p.generated_in_tree);
    }
};

/// The fork's status name for a single brfs pass (abstracted IW) and for rollout IW.
const char* brfs_status_name(SearchStatus s)
{
    return s == SearchStatus::Exhausted ? "exhausted" : mimir_status_name(s);
}
const char* rollout_status_name(SearchStatus s)
{
    switch (s)
    {
        case SearchStatus::Exhausted: return "exhausted";
        case SearchStatus::Failed: return "failed";  // the rollout budget
        default: return mimir_status_name(s);
    }
}

std::string rollout_stats_json(const RolloutIwStatistics& s)
{
    char buf[640];
    std::snprintf(buf, sizeof buf,
                  "{\"rollouts\":%llu,\"generated\":%llu,\"expanded\":%llu,\"feature_depth_improvements\":%llu,\"case1\":%llu,\"case2\":%llu,"
                  "\"case3\":%llu,\"case4\":%llu,\"solved_propagations\":%llu,\"dead_ends\":%llu,\"depth_bound_prunings\":%llu,"
                  "\"incumbent_bound_prunings\":%llu,\"max_rollout_depth\":%u,\"tree_nodes\":%llu,\"blocked\":%llu}",
                  static_cast<unsigned long long>(s.rollouts), static_cast<unsigned long long>(s.generated), static_cast<unsigned long long>(s.expanded),
                  static_cast<unsigned long long>(s.feature_depth_improvements), static_cast<unsigned long long>(s.case1),
                  static_cast<unsigned long long>(s.case2), static_cast<unsigned long long>(s.case3), static_cast<unsigned long long>(s.case4),
                  static_cast<unsigned long long>(s.solved_propagations), static_cast<unsigned long long>(s.dead_ends),
                  static_cast<unsigned long long>(s.depth_bound_prunings), static_cast<unsigned long long>(s.incumbent_bound_prunings),
                  s.max_rollout_depth, static_cast<unsigned long long>(s.tree_nodes), static_cast<unsigned long long>(s.blocked));
    return buf;
}

ActionOrdering ordering_of(const std::string& k)
{
    if (k == "in_order")
        return ActionOrdering::InOrder;
    if (k == "randomized")
        return ActionOrdering::Randomized;
    if (k == "dgaf")
        return ActionOrdering::DirectGoalAchieverFirst;
    if (k == "regression")
        return ActionOrdering::GoalRegressionRelevance;
    if (k == "mixed")
        return ActionOrdering::MixedRegressionRandom;
    usage("unknown ordering " + k);
}

std::string rollout_json(const Names& names, const RolloutResult& r, u64 seed, bool report)
{
    std::string o = "{\"seed\":" + std::to_string(seed) + ",\"status\":\"" + mimir_status_name(r.search.status) + "\",\"per_pass\":\"";
    Sums s;
    for (const IwPassStatistics& p : r.search.passes)
        s.add(p, p.arity);
    o += s.per_pass + "\"";
    if (r.search.status == SearchStatus::Solved)
        o += ",\"plan_len\":" + std::to_string(r.search.plan.size()) + ",\"plan\":" + names.plan_json(r.search.plan);
    std::vector<std::string> reached;
    for (CanonicalAtom c : r.reached_fluent_atoms)
        reached.push_back(names.name(c));
    std::sort(reached.begin(), reached.end());
    u64 h = 0xcbf29ce484222325ULL;
    for (const std::string& x : reached)
        h = fnv1a(x + "\n", h);
    o += ",\"num_states\":" + std::to_string(r.num_states) + ",\"reached_fluent\":" + std::to_string(reached.size()) + ",\"reached_fluent_hash\":\"" +
         hex(h) + "\",\"reached_derived\":" + std::to_string(r.reached_derived_atoms.size());
    if (report)
    {
        std::vector<std::string> states(r.landing_states.size());
        u64 dead = 0;
        for (usize i = 0; i < r.landing_states.size(); ++i)
        {
            std::vector<std::string> atoms;
            for (CanonicalAtom c : r.landing_states[i].atoms)
                atoms.push_back(names.name(c));
            std::sort(atoms.begin(), atoms.end());
            for (const std::string& a : atoms)
                states[i] += (states[i].empty() ? "" : " ") + a;
            dead += r.landing_states[i].direct_dead_end;
        }
        std::vector<std::string> pairs;
        for (const auto& [c, i] : r.landing_state_by_atom)
            pairs.push_back(names.name(c) + " -> " + states[i]);
        std::sort(pairs.begin(), pairs.end());
        std::vector<std::string> sorted = states;
        std::sort(sorted.begin(), sorted.end());
        u64 hp = 0xcbf29ce484222325ULL, hs = hp;
        for (const std::string& x : pairs)
            hp = fnv1a(x + "\n", hp);
        for (const std::string& x : sorted)
            hs = fnv1a(x + "\n", hs);
        o += ",\"landing_states\":" + std::to_string(states.size()) + ",\"landing_dead_ends\":" + std::to_string(dead) + ",\"landing_states_hash\":\"" +
             hex(hs) + "\",\"landing_by_atom_hash\":\"" + hex(hp) + "\"";
        std::vector<std::string> lines;
        u64 pairs_n = 0;
        for (const auto& [c, row] : r.co_occurrence)
        {
            std::vector<std::string> names_row;
            for (CanonicalAtom x : row)
                names_row.push_back(names.name(x));
            std::sort(names_row.begin(), names_row.end());
            pairs_n += names_row.size();
            std::string l = names.name(c) + ":";
            for (const std::string& x : names_row)
                l += " " + x;
            lines.push_back(l);
        }
        std::sort(lines.begin(), lines.end());
        u64 hc = 0xcbf29ce484222325ULL;
        for (const std::string& x : lines)
            hc = fnv1a(x + "\n", hc);
        o += ",\"co_occurrence_pairs\":" + std::to_string(pairs_n) + ",\"co_occurrence_hash\":\"" + hex(hc) + "\"";
    }
    return o + "}";
}
}  // namespace

int main(int argc, char** argv)
{
    std::string task_file, domain, problem, algo, landmarks_file, ordering = "in_order", goal_atom, seeds_arg, orderings_arg, order_file, misses_file;
    u32 k = 1, threads = 1, workers = 4, reps = 1;
    u32 max_next_layer_states = ~u32{0};
    u64 seed = 0, base_seed = 0, max_states = ~u64{0}, max_rollouts = ~u64{0};
    double timeout_s = std::numeric_limits<double>::infinity();
    bool base = false, preserve_goal = true, keep_depth_one = false, preserve_landmarks = true, lm_disjunctive = false, lm_all_private = false,
         report = true;
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
        else if (a == "--base")
            base = true;
        else if (a == "--no-preserve-goal")
            preserve_goal = false;
        else if (a == "--keep-depth-one")
            keep_depth_one = true;
        else if (a == "--no-preserve-landmarks")
            preserve_landmarks = false;
        else if (a == "--landmarks")
            landmarks_file = value();
        else if (a == "--lm-disjunctive")
            lm_disjunctive = true;
        else if (a == "--lm-all-private")
            lm_all_private = true;
        else if (a == "--ordering")
            ordering = value();
        else if (a == "--seed")
            seed = std::stoull(value());
        else if (a == "--goal-atom")
            goal_atom = value();
        else if (a == "--seeds")
            seeds_arg = value();
        else if (a == "--threads")
            threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--max-next-layer-states")
            max_next_layer_states = static_cast<u32>(std::stoul(value()));
        else if (a == "--no-report")
            report = false;
        else if (a == "--workers")
            workers = static_cast<u32>(std::stoul(value()));
        else if (a == "--base-seed")
            base_seed = std::stoull(value());
        else if (a == "--orderings")
            orderings_arg = value();
        else if (a == "--max-states")
            max_states = std::stoull(value());
        else if (a == "--max-rollouts")
            max_rollouts = std::stoull(value());
        else if (a == "--timeout-ms")
            timeout_s = std::stod(value()) / 1000.0;
        else if (a == "--order-from")
            order_file = value();
        else if (a == "--misses-out")
            misses_file = value();
        else if (a == "--reps")
            reps = static_cast<u32>(std::stoul(value()));
        else if (!a.empty() && a[0] != '-' && task_file.empty())
            task_file = a;
        else
            usage("unknown argument " + a);
    }
    if (task_file.empty() && (domain.empty() || problem.empty()))
        usage("missing task (task.txt or --domain/--problem)");
    try
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<const Task> task;
        if (!task_file.empty())
            task = Task::from_text_file(task_file);
        else
        {
#if defined(MYMYR_HAVE_FRONTEND)
            const auto data = frontend::load_task(domain, problem);
            task = Task::create(*data);
#else
            usage("built without the PDDL front end: pass an exported task.txt");
#endif
        }
        const double prep = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const Names names(*task);

        SearchControl control;
        control.budget.max_seconds = timeout_s;
        if (goal_atom == "@first" || goal_atom == "@last")
        {
            // the smallest / largest name among the positive fluent goal atoms (fork_iw_variants picks the same)
            const formalism::TaskData& t = task->data();
            std::vector<std::string> goal_names;
            for (const formalism::Literal& l : t.literals_of(t.goal))
            {
                if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
                    continue;
                std::vector<u32> args;
                for (formalism::Term x : t.terms_of(l))
                    args.push_back(formalism::term_object(x).v);
                const CanonicalAtom c = task->atoms().layout().encode(l.pred.v, args.data());
                if (c < task->atoms().layout().fluent_count)
                    goal_names.push_back(names.name(c));
            }
            if (goal_names.empty())
            {
                std::fprintf(stderr, "error: no positive fluent goal atom\n");
                return 3;
            }
            std::sort(goal_names.begin(), goal_names.end());
            goal_atom = goal_atom == "@first" ? goal_names.front() : goal_names.back();
        }
        if (!goal_atom.empty())
        {
            const auto c = names.parse(goal_atom);
            if (!c)
                usage("unknown goal atom " + goal_atom);
            control.goal.kind = GoalSpec::Kind::AnyOf;
            control.goal.goals.push_back({{SlotId{task->atoms().intern(*c)}}, {}});
        }
        LandmarkNovelty lm;
        u64 lm_unknown = 0, lm_facts = 0, lm_sets = 0;
        if (!landmarks_file.empty())
        {
            std::ifstream in(landmarks_file);
            if (!in)
                usage("cannot open " + landmarks_file);
            std::vector<CanonicalAtom> facts;
            std::vector<std::vector<CanonicalAtom>> sets;
            std::string line;
            while (std::getline(in, line))
            {
                if (line.size() < 2)
                    continue;
                std::vector<CanonicalAtom> atoms;
                for (usize p = line.find('('); p != std::string::npos; p = line.find('(', p + 1))
                {
                    const auto c = names.parse(line.substr(p, line.find(')', p) - p + 1));
                    if (c)
                        atoms.push_back(*c);
                    else
                        ++lm_unknown;
                }
                if (line[0] == 'F')
                    facts.insert(facts.end(), atoms.begin(), atoms.end());
                else if (line[0] == 'D')
                    sets.push_back(std::move(atoms));
            }
            lm_facts = facts.size();
            lm_sets = sets.size();
            lm.graph = std::make_shared<const landmarks::FactLandmarkGraph>(landmarks::FactLandmarkGraph::create(std::move(facts), std::move(sets)));
            lm.disjunctive = lm_disjunctive;
            lm.all_private = lm_all_private;
        }
        std::shared_ptr<TraceOrder> replay;
        OrderHook hook;
        if (!order_file.empty())
        {
            replay = std::make_shared<TraceOrder>(*task, order_file);
            if (!misses_file.empty() && !(replay->misses_out = std::fopen(misses_file.c_str(), "w")))
                throw std::runtime_error("cannot open " + misses_file);
            hook = [replay](StateView st, std::span<const Action> acts, std::vector<u32>& ord) { (*replay)(st, acts, ord); };
        }

        std::string status, extra, message;
        Sums sums;
        std::vector<Action> plan;
        bool solved = false;
        double cost = 0, best_s = std::numeric_limits<double>::infinity();
        for (u32 rep = 0; rep < std::max<u32>(reps, 1); ++rep)
        {
            sums = Sums{};
            extra.clear();
            const auto ts = std::chrono::steady_clock::now();
            double call_s = -1;  // parallel rollouts: the call alone, without the JSON (names and hashes) of its results
            if (algo == "aiw")
            {
                AbstractedIwOptions o;
                o.control = control;
                if (max_states != ~u64{0})
                    o.control.budget.max_states = max_states;
                o.width = k;
                o.base_abstracted = base;
                o.preserve_goal_atoms = preserve_goal;
                o.keep_depth_one_novel = keep_depth_one;
                o.landmarks = lm;
                o.preserve_landmark_atoms = preserve_landmarks;
                o.successor_order = hook;
                const IwResult r = abstracted_iw(*task, o);
                for (const IwPassStatistics& p : r.passes)
                    sums.add(p, k);
                status = brfs_status_name(r.status);
                solved = r.status == SearchStatus::Solved;
                plan = r.plan;
                cost = r.cost;
                message = r.message;
            }
            else if (algo == "liw")
            {
                LiwOptions o;
                o.control = control;
                if (max_states != ~u64{0})
                    o.control.budget.max_states = max_states;
                o.max_arity = k;
                o.landmarks = lm;
                o.successor_order = hook;
                const IwResult r = liw(*task, o);
                for (const IwPassStatistics& p : r.passes)
                    sums.add(p, p.arity);
                status = mimir_status_name(r.status);
                solved = r.status == SearchStatus::Solved;
                plan = r.plan;
                cost = r.cost;
                message = r.message;
            }
            else if (algo == "rollout_iw")
            {
                RolloutIwOptions o;
                o.control = control;
                o.control.budget.max_states = max_states;
                o.ordering = ordering_of(ordering);
                o.seed = seed;
                o.max_rollouts = max_rollouts;
                o.successor_order = hook;
                const RolloutIwResult r = rollout_iw(*task, o);
                sums.expanded = r.statistics.expanded;
                sums.generated = r.statistics.generated;
                status = rollout_status_name(r.status);
                solved = r.status == SearchStatus::Solved;
                plan = r.plan;
                cost = r.cost;
                message = r.message;
                extra += ",\"rollout\":" + rollout_stats_json(r.statistics) + ",\"root_solved\":" + (r.root_solved ? "true" : "false") +
                         ",\"num_rollouts\":" + std::to_string(r.statistics.rollouts) + ",\"num_tree_nodes\":" + std::to_string(r.statistics.tree_nodes);
            }
            else if (algo == "rollouts")
            {
                ParallelRolloutOptions o;
                o.iw.control = control;
                if (max_states != ~u64{0})
                    o.iw.control.budget.max_states = max_states;
                o.iw.max_arity = k;
                o.iw.successor_order = hook;
                o.num_threads = threads;
                o.max_next_layer_states = max_next_layer_states;
                o.report_landing_states = report;
                o.report_co_occurrence = report;
                std::istringstream in(seeds_arg);
                std::string s;
                while (std::getline(in, s, ','))
                    if (!s.empty())
                        o.seeds.push_back(std::stoull(s));
                const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                call_s = secs;
                std::string list;
                u64 n_solved = 0;
                for (usize i = 0; i < r.rollouts.size(); ++i)
                {
                    if (rep + 1 == std::max<u32>(reps, 1))
                        list += (i ? "," : "") + rollout_json(names, r.rollouts[i], o.seeds[i], report);
                    n_solved += r.rollouts[i].search.status == SearchStatus::Solved;
                }
                status = "in_progress";
                message = r.message;
                extra += ",\"threads_used\":" + std::to_string(r.threads_used) + ",\"rollouts_solved\":" + std::to_string(n_solved) +
                         ",\"batch_s\":" + std::to_string(secs) + ",\"rollouts\":[" + list + "]";
            }
            else if (algo == "portfolio")
            {
                PortfolioOptions o;
                o.control = control;
                if (max_states != ~u64{0})
                    o.control.budget.max_states = max_states;
                o.num_rollout_workers = workers;
                o.num_threads = threads;
                o.base_seed = base_seed;
                o.successor_order = hook;
                std::istringstream in(orderings_arg);
                std::string s;
                while (std::getline(in, s, ','))
                {
                    if (s.empty())
                        continue;
                    const auto colon = s.find(':');
                    o.rollout_orderings.push_back({ordering_of(s.substr(0, colon)), colon == std::string::npos ? 0 : std::stoull(s.substr(colon + 1))});
                }
                const PortfolioResult r = atomic_goal_portfolio(*task, o);
                for (const IwPassStatistics& p : r.certifier.passes)
                {
                    sums.expanded += p.expanded;
                    sums.generated += p.generated;
                    sums.in_tree += p.generated_in_tree;
                }
                // the fork passes its certifier's IW status through (an exhausted ladder is FAILED)
                status = r.status == SearchStatus::Exhausted ? "failed" : mimir_status_name(r.status);
                solved = r.status == SearchStatus::Solved;
                plan = r.plan;
                cost = r.cost;
                message = r.message;
                std::string stats, statuses;
                for (usize i = 0; i < r.rollout_statistics.size(); ++i)
                {
                    stats += (i ? "," : "") + rollout_stats_json(r.rollout_statistics[i]);
                    statuses += std::string(i ? "," : "") + "\"" + (r.rollout_rounds[i] ? rollout_status_name(r.rollout_statuses[i]) : "in_progress") + "\"";
                }
                extra += ",\"plan_length\":" + std::to_string(solved ? r.plan_length : 0) + ",\"certified_optimal\":" + (r.certified_optimal ? "true" : "false") +
                         ",\"iw_lower_bound\":" + std::to_string(r.iw_lower_bound) + ",\"iw_completed_depth\":" + std::to_string(r.iw_completed_depth) +
                         ",\"winning_worker\":" + std::to_string(r.winning_worker) + ",\"certifier_status\":\"" +
                         (r.certifier_ran ? mimir_status_name(r.certifier_status) : "in_progress") + "\",\"total_expansions\":" +
                         std::to_string(r.total_expansions) + ",\"threads_used\":" + std::to_string(r.threads_used) + ",\"rollout_statistics\":[" + stats +
                         "],\"rollout_statuses\":[" + statuses + "],\"rollout_rounds\":[";
                for (usize i = 0; i < r.rollout_rounds.size(); ++i)
                    extra += (i ? "," : "") + std::to_string(r.rollout_rounds[i]);
                extra += "]";
            }
            else
                usage("unknown algo " + algo);
            best_s = std::min(best_s, call_s >= 0 ? call_s : std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
        }

        std::printf("{\"lib\":\"mymyr\",\"algo\":\"%s\",\"k\":%u,\"status\":\"%s\",\"prep_s\":%.6f,\"search_s\":%.6f,\"expanded\":%llu,"
                    "\"generated\":%llu,\"generated_in_tree\":%llu,\"passes\":%llu,\"per_pass\":\"%s\"%s",
                    algo.c_str(), k, status.c_str(), prep, best_s, static_cast<unsigned long long>(sums.expanded),
                    static_cast<unsigned long long>(sums.generated), static_cast<unsigned long long>(sums.in_tree),
                    static_cast<unsigned long long>(sums.passes), sums.per_pass.c_str(), extra.c_str());
        if (lm.graph)
            std::printf(",\"landmarks\":%llu,\"disjunctive_landmarks\":%llu,\"landmark_atoms_unknown\":%llu", static_cast<unsigned long long>(lm_facts),
                        static_cast<unsigned long long>(lm_sets), static_cast<unsigned long long>(lm_unknown));
        if (solved)
            std::printf(",\"plan_len\":%zu,\"plan_cost\":%.17g,\"plan\":%s", plan.size(), cost, names.plan_json(plan).c_str());
        if (!message.empty())
            std::printf(",\"message\":\"%s\"", json_escape(message).c_str());
        if (replay && replay->misses_out)
            std::fclose(replay->misses_out);
        if (replay)
            std::printf(",\"order_misses\":{\"states\":%llu,\"actions\":%llu}", static_cast<unsigned long long>(replay->state_misses.load()),
                        static_cast<unsigned long long>(replay->action_misses.load()));
        if (!goal_atom.empty())
            std::printf(",\"goal_atom\":\"%s\"", goal_atom.c_str());
        std::printf(",\"F\":%u,\"peak_rss_mb\":%.1f}\n", task->atoms().fluent_slots(), peak_rss_mb());
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
