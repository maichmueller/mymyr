// mymyr_landmarks: relaxed reachability, fact landmarks and the landmark transition ordering, printing one line
// "RESULT_JSON {...}" with the same fields as the fork's own landmark tool (same modes, same
// hashes over the same sorted names), plus mymyr's own.
//
//   mymyr_landmarks (task.txt | --domain D --problem P) --mode rr|without|approx|lifted|order [options]
//
//   rr       reachable fluent and derived atoms: counts, hashes of the sorted names, per predicate counts, goal
//            reachability, compile and fixpoint milliseconds (best of --reps)
//   without  goal_reachable_without({a}) for every reachable fluent atom a (the complete landmark test), and the
//            restricted tables of the first --queries of them: count of landmarks, milliseconds (best of --reps)
//   approx   the approximate (grounded) generator: fact and disjunctive landmarks, orderings, achievers
//            [--disj-size K] [--disj-depth D]
//   lifted   the lifted generator [--disambiguation off|per_literal|joint] [--complete off|members|all]
//            [--no-first-achievers] [--no-filter-members] [--no-verify] [--no-static-filter] [--max-combinations N]
//            [--max-members N]
//   order    IW(1) with the landmark transition ordering over the approximate graph: per layer of the width-1 pass the
//            generated transitions in admission order [--max-layers L] [--order-from FORK_DUMP]
//            --order-from replays the fork's generation order within each parent state from the fork tool's --dump of
//            the same task (its admission order restricted to one parent, which agrees with its generation order on
//            every tie the stable sort keeps)
//   --atoms lazy|frozen|auto (default lazy: no pilot search) is passed to Task::create.
//   --dump FILE appends the names behind every hash to FILE; atom names are "pred(o1,o2)", the objects of introduced
//   derived predicates (axiom_<k>) sorted.

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/landmarks/transition_ordering.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/reachability/relaxed_reachability.hpp"
#include "mymyr/task/task.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>

using namespace mymyr;

namespace
{
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

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
                 "error: %s\nusage: mymyr_landmarks (task.txt | --domain D --problem P) --mode rr|without|approx|lifted|order\n"
                 "       [--reps N] [--dump FILE] [--queries N] [--disj-size K] [--disj-depth D] [--max-layers L] [--order-from F]\n"
                 "       [--disambiguation off|per_literal|joint] [--complete off|members|all] [--no-first-achievers]\n"
                 "       [--no-filter-members] [--no-verify] [--no-static-filter] [--max-combinations N] [--max-members N]\n",
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
        return "null";
    std::ostringstream o;
    o.precision(9);
    o << v;
    return o.str();
}

std::string jlist(const std::vector<std::string>& xs)
{
    std::string o = "[";
    for (std::size_t i = 0; i < xs.size(); ++i)
        o += (i ? "," : "") + jstr(xs[i]);
    return o + "]";
}

u64 fnv(const std::string& s, u64 h = 1469598103934665603ull)
{
    for (unsigned char c : s)
        h = (h ^ c) * 1099511628211ull;
    return h;
}

std::string hex(u64 x)
{
    char buf[20];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(x));
    return buf;
}

/// Hash of a set of strings (sorted, newline-joined): the fork tool's set_hash.
std::string set_hash(std::vector<std::string> v)
{
    std::sort(v.begin(), v.end());
    u64 h = 1469598103934665603ull;
    for (const auto& s : v)
        h = fnv(s + "\n", h);
    return hex(h);
}

bool introduced(std::string_view pred) { return pred.starts_with("axiom_"); }

struct Args
{
    std::string task_file, domain, problem, mode, dump;
    int reps = 1;
    u32 queries = 0;
    usize disj_size = 0, disj_depth = 0;
    landmarks::LiftedFactLandmarkOptions lifted;
    u32 max_layers = 1000;
    std::string order_from;
    TaskOptions to = lazy_atoms();

    static TaskOptions lazy_atoms()
    {
        TaskOptions o;
        o.atoms = TaskOptions::Atoms::Lazy;  // Auto runs a pilot BrFS, which alone takes minutes on childsnack p30-hard
        return o;
    }
};

void dump_lines(const Args& a, const std::string& header, const std::vector<std::string>& lines)
{
    if (a.dump.empty())
        return;
    std::ofstream f(a.dump, std::ios::app);
    f << "# " << header << "\n";
    for (const auto& l : lines)
        f << l << "\n";
}

std::string atom_name(const Task& task, PredicateId p, std::span<const u32> args)
{
    const auto& d = task.data();
    const std::string pred(d.str(d.predicate(p).name));
    std::vector<std::string> objs;
    for (u32 o : args)
        objs.push_back(task.object_name(ObjectId{o}));
    if (introduced(pred))
        std::sort(objs.begin(), objs.end());
    std::string s = pred + "(";
    for (std::size_t i = 0; i < objs.size(); ++i)
        s += (i ? "," : "") + objs[i];
    return s + ")";
}

std::string cid_name(const Task& task, CanonicalAtom c)
{
    const CanonicalLayout& L = task.atoms().layout();
    std::vector<u32> args(std::max<u32>(1, L.max_arity));
    const u32 p = L.decode(c, args.data());
    return atom_name(task, PredicateId{p}, std::span<const u32>(args.data(), L.arity[p]));
}

std::string action_name(const Task& task, const ActionLabel& a)
{
    std::string s = "(" + task.schema_name(a.schema);
    for (ObjectId o : a.binding)
        s += " " + task.object_name(o);
    return s + ")";
}

// ------------------------------------------------------------------------------------------------ graphs
struct GraphOut
{
    std::vector<std::string> landmarks, disjunctive, orderings, achievers, first_achievers, lifted;
};

GraphOut describe(const Task& task, const landmarks::FactLandmarkGraph& g)
{
    GraphOut o;
    for (CanonicalAtom c : g.landmarks())
        o.landmarks.push_back(cid_name(task, c));
    for (const auto& members : g.disjunctive())
    {
        std::vector<std::string> m;
        for (CanonicalAtom c : members)
            m.push_back(cid_name(task, c));
        std::sort(m.begin(), m.end());
        std::string x = "{";
        for (std::size_t k = 0; k < m.size(); ++k)
            x += (k ? "|" : "") + m[k];
        o.disjunctive.push_back(x + "}");
    }
    for (const auto& [b, c] : g.orderings())
        o.orderings.push_back(cid_name(task, b) + " < " + cid_name(task, c));
    if (g.has_achiever_index())
    {
        const auto& ix = g.achievers();
        for (CanonicalAtom c : g.landmarks())
        {
            std::vector<std::string> ac, fa;
            for (u32 x : ix.achievers_of(c))
                ac.push_back(action_name(task, ix.actions[x].label()));
            for (u32 x : ix.first_achievers_of(c))
                fa.push_back(action_name(task, ix.actions[x].label()));
            std::sort(ac.begin(), ac.end());
            std::sort(fa.begin(), fa.end());
            std::string s1 = cid_name(task, c) + " :", s2 = s1;
            for (const auto& x : ac)
                s1 += " " + x;
            for (const auto& x : fa)
                s2 += " " + x;
            o.achievers.push_back(s1);
            o.first_achievers.push_back(s2);
        }
    }
    const auto& d = task.data();
    for (const auto& l : g.lifted())
    {
        std::vector<std::string> objs;
        for (u32 x : l.binding)
            objs.push_back(x == landmarks::k_free ? std::string("?") : task.object_name(ObjectId{x}));
        const std::string pred(d.str(d.predicate(l.predicate).name));
        if (introduced(pred))
            std::sort(objs.begin(), objs.end());
        std::string s = pred + "(";
        for (std::size_t i = 0; i < objs.size(); ++i)
            s += (i ? "," : "") + objs[i];
        o.lifted.push_back(s + ") members=" + std::to_string(l.members.size()) + " initially_true=" + std::to_string(l.initially_true ? 1 : 0) +
                           " fact=" + std::to_string(l.is_fact() ? 1 : 0));
    }
    for (auto* v : {&o.landmarks, &o.disjunctive, &o.orderings, &o.achievers, &o.first_achievers, &o.lifted})
        std::sort(v->begin(), v->end());
    return o;
}

void print_graph(const Args& a, const GraphOut& o, const std::string& extra)
{
    dump_lines(a, "landmarks", o.landmarks);
    dump_lines(a, "disjunctive", o.disjunctive);
    dump_lines(a, "orderings", o.orderings);
    dump_lines(a, "achievers", o.achievers);
    dump_lines(a, "first_achievers", o.first_achievers);
    dump_lines(a, "lifted", o.lifted);
    std::cout << "RESULT_JSON {\"lib\":\"mymyr\",\"mode\":" << jstr(a.mode) << ",\"landmarks\":" << o.landmarks.size()
              << ",\"landmarks_hash\":" << jstr(set_hash(o.landmarks)) << ",\"disjunctive\":" << o.disjunctive.size()
              << ",\"disjunctive_hash\":" << jstr(set_hash(o.disjunctive)) << ",\"orderings\":" << o.orderings.size()
              << ",\"orderings_hash\":" << jstr(set_hash(o.orderings)) << ",\"achievers_hash\":" << jstr(set_hash(o.achievers))
              << ",\"first_achievers_hash\":" << jstr(set_hash(o.first_achievers)) << ",\"lifted\":" << o.lifted.size()
              << ",\"lifted_hash\":" << jstr(set_hash(o.lifted)) << ",\"landmark_names\":" << (o.landmarks.size() <= 400 ? jlist(o.landmarks) : "null")
              << extra << ",\"peak_rss_mb\":" << jnum(peak_rss_mb()) << "}" << std::endl;
}

void mode_approx(const Args& a, const Task& task)
{
    landmarks::ApproximateFactLandmarkOptions o;
    o.max_disjunctive_landmark_size = a.disj_size;
    o.max_disjunctive_landmark_depth = a.disj_depth;
    double best = 1e300;
    landmarks::FactLandmarkGraph g;
    for (int r = 0; r < a.reps; ++r)
    {
        const auto t0 = Clock::now();
        g = landmarks::approximate_fact_landmarks(task, o);
        best = std::min(best, ms_since(t0));
    }
    print_graph(a, describe(task, g), ",\"generator_ms\":" + jnum(best));
}

void mode_lifted(const Args& a, const Task& task)
{
    double best = 1e300;
    landmarks::FactLandmarkGraph g;
    for (int r = 0; r < a.reps; ++r)
    {
        const auto t0 = Clock::now();
        g = landmarks::lifted_fact_landmarks(task, a.lifted);
        best = std::min(best, ms_since(t0));
    }
    print_graph(a, describe(task, g), ",\"generator_ms\":" + jnum(best));
}

// ------------------------------------------------------------------------------------------------ order
/// The fork tool's state key: hash of the sorted fluent atom names, 12 hex digits.
std::string state_key(const Task& task, StateView s)
{
    std::vector<std::string> names;
    const AtomIndex& ix = task.atoms();
    bits::for_each(s.w, s.nw,
                   [&](u64 slot)
                   {
                       const u32* rec = ix.record(AtomKind::Fluent, static_cast<u32>(slot));
                       names.push_back(atom_name(task, PredicateId{rec[0]}, std::span<const u32>(rec + 1, ix.layout().arity[rec[0]])));
                   });
    return set_hash(names).substr(0, 12);
}

std::string score_str(const landmarks::LandmarkTransitionScore& sc)
{
    return "[" + std::to_string(sc.num_new_landmarks) + "," + std::to_string(sc.num_new_unique_landmarks) + "," +
           std::to_string(sc.unique_achiever_action ? 1 : 0) + "," + std::to_string(sc.num_deleted_achieved_landmarks) + "]";
}

/// The landmark ordering over a replayed generation order: per parent (by key), the action order of the fork's dump.
class ReplayOrdering final : public search::TransitionOrdering
{
public:
    ReplayOrdering(std::shared_ptr<const landmarks::LandmarkTransitionOrdering> inner, const std::string& dump) : m_inner(std::move(inner))
    {
        std::ifstream in(dump);
        if (!in)
            throw std::runtime_error("cannot open " + dump);
        std::string line;
        bool section = false;
        while (std::getline(in, line))
        {
            if (line.starts_with("# "))
            {
                section = line == "# order";
                continue;
            }
            if (!section)
                continue;
            // "depth key (action ...) [a,b,c,d]( +)"
            const auto k0 = line.find(' ');
            if (k0 == std::string::npos)
                continue;
            const auto k1 = line.find(' ', k0 + 1);
            if (k1 == std::string::npos)
                continue;
            const auto a1 = line.find(") [", k1);
            if (a1 == std::string::npos)
                continue;
            auto& m = m_rank[line.substr(k0 + 1, k1 - k0 - 1)];
            m.try_emplace(line.substr(k1 + 1, a1 - k1), static_cast<u32>(m.size()));
        }
    }

    void order(const Task& task, std::span<const search::LayerTransition> layer, std::vector<u32>& order) const override
    {
        // replayed generation order: parents in layer order (their transitions are contiguous), then the fork's rank
        std::vector<std::pair<u64, u32>> keyed;
        u64 group = 0;
        const u64* last = nullptr;
        const std::unordered_map<std::string, u32>* ranks = nullptr;
        for (u32 i = 0; i < layer.size(); ++i)
        {
            if (layer[i].parent.w != last)
            {
                last = layer[i].parent.w;
                ++group;
                const auto it = m_rank.find(state_key(task, layer[i].parent));
                ranks = it == m_rank.end() ? nullptr : &it->second;
                if (!ranks)
                    ++state_misses;
            }
            u64 r = (u64{1} << 31) + i;
            if (ranks)
            {
                const auto it = ranks->find(action_name(task, layer[i].action));
                if (it != ranks->end())
                    r = it->second;
                else
                    ++action_misses;
            }
            keyed.emplace_back((group << 32) | r, i);
        }
        std::sort(keyed.begin(), keyed.end());
        std::vector<search::LayerTransition> replayed;
        replayed.reserve(layer.size());
        for (const auto& [k, i] : keyed)
            replayed.push_back(layer[i]);
        std::vector<u32> inner;
        m_inner->order(task, replayed, inner);
        order.clear();
        for (u32 j : inner)
            order.push_back(keyed[j].second);
    }

    mutable u64 state_misses = 0, action_misses = 0;

private:
    std::shared_ptr<const landmarks::LandmarkTransitionOrdering> m_inner;
    std::unordered_map<std::string, std::unordered_map<std::string, u32>> m_rank;
};

/// Records the admission-order events of the width-1 pass: per transition "key (action) [score]", " +" if admitted.
class OrderRecorder final : public search::SearchObserver
{
public:
    OrderRecorder(const Task& task, const landmarks::LandmarkTransitionOrdering& ordering) : m_task(task), m_ordering(ordering) {}

    void on_expand(u64 id, StateView state) override
    {
        Node& e = m_nodes[id];
        e.key = state_key(m_task, state);
        e.words.assign(state.w, state.w + state.nw);
    }
    void on_generate(u64 parent, const Action& action, u64 child, StateView child_state, bool is_new) override
    {
        const Node& p = m_nodes.at(parent);
        const StateView pv{p.words.data(), static_cast<u32>(p.words.size()), nullptr, 0};
        const auto sc = m_ordering.score(pv, action.label(), child_state);
        const u32 depth = p.depth;
        if (layers.size() <= depth)
        {
            layers.resize(depth + 1);
            admitted.resize(depth + 1, 0);
        }
        layers[depth].push_back(p.key + " " + action_name(m_task, action.label()) + " " + score_str(sc) + (is_new ? " +" : ""));
        if (is_new)
        {
            ++admitted[depth];
            m_nodes[child].depth = depth + 1;
        }
    }

    std::vector<std::vector<std::string>> layers;
    std::vector<u64> admitted;

private:
    struct Node
    {
        std::string key;
        std::vector<u64> words;
        u32 depth = 0;
    };
    const Task& m_task;
    const landmarks::LandmarkTransitionOrdering& m_ordering;
    std::unordered_map<u64, Node> m_nodes;
};

void mode_order(const Args& a, const Task& task)
{
    landmarks::ApproximateFactLandmarkOptions lo;
    lo.max_disjunctive_landmark_size = a.disj_size;
    lo.max_disjunctive_landmark_depth = a.disj_depth;
    const auto graph = landmarks::approximate_fact_landmarks(task, lo);
    const auto ordering = std::make_shared<const landmarks::LandmarkTransitionOrdering>(task, graph);
    std::shared_ptr<ReplayOrdering> replay;
    if (!a.order_from.empty())
        replay = std::make_shared<ReplayOrdering>(ordering, a.order_from);
    OrderRecorder rec(task, *ordering);
    search::IwOptions io;
    io.max_arity = 1;
    io.control.budget.max_depth = a.max_layers;
    io.control.observer = &rec;
    if (replay)
        io.transition_ordering = replay;
    else
        io.transition_ordering = ordering;
    const auto t0 = Clock::now();
    const search::IwResult r = search::iw(task, io);
    const double search_s = ms_since(t0) / 1000.0;
    std::vector<std::string> lines, layer_hashes;
    usize transitions = 0;
    std::string adm = "[";
    for (usize l = 0; l < rec.layers.size(); ++l)
    {
        if (rec.layers[l].empty())
            continue;
        const usize depth = layer_hashes.size();
        u64 h = 1469598103934665603ull;
        for (const auto& t : rec.layers[l])
        {
            lines.push_back(std::to_string(depth) + " " + t);
            h = fnv(t + "\n", h);
        }
        transitions += rec.layers[l].size();
        adm += (depth ? "," : "") + std::to_string(rec.admitted[l]);
        layer_hashes.push_back(hex(h));
    }
    adm += "]";
    dump_lines(a, "order", lines);
    u64 all = 1469598103934665603ull;
    for (const auto& l : lines)
        all = fnv(l + "\n", all);
    std::cout << "RESULT_JSON {\"lib\":\"mymyr\",\"mode\":\"order\",\"status\":" << jstr(search::mimir_status_name(r.status))
              << ",\"plan_length\":" << (r.status == search::SearchStatus::Solved ? std::to_string(r.plan.size()) : std::string("null"))
              << ",\"layers\":" << layer_hashes.size() << ",\"transitions\":" << transitions << ",\"order_hash\":" << jstr(hex(all))
              << ",\"layer_hashes\":" << jlist(layer_hashes) << ",\"admitted\":" << adm << ",\"landmarks\":" << graph.landmarks().size()
              << ",\"search_s\":" << jnum(search_s);
    if (replay)
        std::cout << ",\"replay_misses\":{\"states\":" << replay->state_misses << ",\"actions\":" << replay->action_misses << "}";
    std::cout << ",\"peak_rss_mb\":" << jnum(peak_rss_mb()) << "}" << std::endl;
}

// ------------------------------------------------------------------------------------------------ rr
void mode_rr(const Args& a, const Task& task)
{
    double best_compile = 1e300, best_fix = 1e300, best_total = 1e300;
    std::shared_ptr<const reachability::RelaxedReachability> rr;
    for (int r = 0; r < a.reps; ++r)
    {
        rr.reset();
        const auto t0 = Clock::now();
        rr = reachability::RelaxedReachability::create(task);
        best_total = std::min(best_total, ms_since(t0));
        best_compile = std::min(best_compile, rr->statistics().compile_ms);
        best_fix = std::min(best_fix, rr->statistics().fixpoint_ms);
    }
    const auto& d = task.data();
    std::vector<std::string> fluent, derived;
    std::map<std::string, std::size_t> per_pred;
    for (u32 p = 0; p < d.predicates.size(); ++p)
    {
        const auto& pr = d.predicates[p];
        if (pr.kind == formalism::PredKind::Static)
            continue;
        const auto tuples = rr->table().tuples(PredicateId{p});
        const std::size_t n = pr.arity == 0 ? rr->table().atoms(PredicateId{p}).size() : tuples.size() / pr.arity;
        per_pred[std::string(d.str(pr.name))] += n;
        for (std::size_t i = 0; i < n; ++i)
        {
            auto name = atom_name(task, PredicateId{p}, tuples.subspan(i * pr.arity, pr.arity));
            (pr.kind == formalism::PredKind::Derived ? derived : fluent).push_back(std::move(name));
        }
    }
    std::sort(fluent.begin(), fluent.end());
    std::sort(derived.begin(), derived.end());
    dump_lines(a, "fluent", fluent);
    dump_lines(a, "derived", derived);
    std::string pp = "{";
    bool first = true;
    for (const auto& [k, v] : per_pred)
    {
        if (v == 0)
            continue;  // the fork lists every predicate; compare the nonzero entries
        pp += (first ? "" : ",") + jstr(k) + ":" + std::to_string(v);
        first = false;
    }
    pp += "}";
    const auto& st = rr->statistics();
    std::cout << "RESULT_JSON {\"lib\":\"mymyr\",\"mode\":\"rr\",\"fluent\":" << fluent.size() << ",\"derived\":" << derived.size()
              << ",\"fluent_hash\":" << jstr(set_hash(fluent)) << ",\"derived_hash\":" << jstr(set_hash(derived))
              << ",\"goal\":" << (rr->goal_reachable() ? "true" : "false") << ",\"per_pred\":" << pp << ",\"rounds\":" << st.rounds
              << ",\"rules\":" << st.rules << ",\"dropped_rules\":" << st.dropped_rules << ",\"plans\":" << st.plans
              << ",\"bindings\":" << st.bindings << ",\"compile_ms\":" << jnum(best_compile) << ",\"fixpoint_ms\":" << jnum(best_fix)
              << ",\"total_ms\":" << jnum(best_total) << ",\"peak_rss_mb\":" << jnum(peak_rss_mb()) << "}" << std::endl;
}

// ------------------------------------------------------------------------------------------------ without
void mode_without(const Args& a, const Task& task)
{
    const auto rr = reachability::RelaxedReachability::create(task);
    const auto& d = task.data();
    std::vector<CanonicalAtom> candidates;
    for (u32 p = 0; p < d.predicates.size(); ++p)
        if (d.predicates[p].kind == formalism::PredKind::Fluent)
            for (CanonicalAtom c : rr->table().atoms(PredicateId{p}))
                candidates.push_back(c);
    std::sort(candidates.begin(), candidates.end());
    double best_goal = 1e300, best_restricted = 1e300;
    std::vector<std::string> landmarks;
    const u32 nq = std::min<u32>(a.queries, static_cast<u32>(candidates.size()));
    for (int r = 0; r < a.reps; ++r)
    {
        std::vector<CanonicalAtom> lm;
        const auto t0 = Clock::now();
        for (CanonicalAtom c : candidates)
            if (!rr->goal_reachable_without(std::span<const CanonicalAtom>(&c, 1)))
                lm.push_back(c);
        best_goal = std::min(best_goal, ms_since(t0));
        const auto t1 = Clock::now();
        u64 sink = 0;
        for (u32 i = 0; i < nq; ++i)
            sink += rr->restricted(std::span<const CanonicalAtom>(&candidates[i], 1)).num_atoms();
        best_restricted = std::min(best_restricted, ms_since(t1));
        if (r == 0)
        {
            (void)sink;
            for (CanonicalAtom c : lm)
            {
                // name through the table's tuple store: find the predicate block of c
                for (u32 p = 0; p < d.predicates.size(); ++p)
                {
                    const auto atoms = rr->table().atoms(PredicateId{p});
                    const auto it = std::find(atoms.begin(), atoms.end(), c);
                    if (it == atoms.end())
                        continue;
                    const u32 ar = d.predicates[p].arity;
                    const auto tuples = rr->table().tuples(PredicateId{p});
                    landmarks.push_back(
                        atom_name(task, PredicateId{p}, tuples.subspan(static_cast<std::size_t>(it - atoms.begin()) * ar, ar)));
                    break;
                }
            }
        }
    }
    std::sort(landmarks.begin(), landmarks.end());
    dump_lines(a, "complete_landmarks", landmarks);
    std::cout << "RESULT_JSON {\"lib\":\"mymyr\",\"mode\":\"without\",\"candidates\":" << candidates.size()
              << ",\"complete_landmarks\":" << landmarks.size() << ",\"complete_landmarks_hash\":" << jstr(set_hash(landmarks))
              << ",\"goal_without_ms\":" << jnum(best_goal) << ",\"goal_without_us_per_query\":"
              << jnum(candidates.empty() ? 0.0 : best_goal * 1000.0 / static_cast<double>(candidates.size())) << ",\"restricted_queries\":" << nq
              << ",\"restricted_ms\":" << jnum(best_restricted) << ",\"peak_rss_mb\":" << jnum(peak_rss_mb()) << "}" << std::endl;
}
}  // namespace

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage("missing value for " + s);
            return argv[++i];
        };
        if (s == "--domain")
            a.domain = value();
        else if (s == "--problem")
            a.problem = value();
        else if (s == "--mode")
            a.mode = value();
        else if (s == "--dump")
            a.dump = value();
        else if (s == "--reps")
            a.reps = std::max(1, std::stoi(value()));
        else if (s == "--queries")
            a.queries = static_cast<u32>(std::stoul(value()));
        else if (s == "--disj-size")
            a.disj_size = std::stoul(value());
        else if (s == "--disj-depth")
            a.disj_depth = std::stoul(value());
        else if (s == "--disambiguation")
        {
            const std::string v = value();
            a.lifted.reachability_disambiguation = v == "off"           ? landmarks::ReachabilityDisambiguation::Off
                                                   : v == "per_literal" ? landmarks::ReachabilityDisambiguation::PerLiteral
                                                                        : landmarks::ReachabilityDisambiguation::Joint;
        }
        else if (s == "--complete")
        {
            const std::string v = value();
            a.lifted.complete_fact_landmarks = v == "off"   ? landmarks::CompleteFactLandmarks::Off
                                               : v == "all" ? landmarks::CompleteFactLandmarks::All
                                                            : landmarks::CompleteFactLandmarks::Members;
        }
        else if (s == "--max-layers")
            a.max_layers = static_cast<u32>(std::stoul(value()));
        else if (s == "--order-from")
            a.order_from = value();
        else if (s == "--no-first-achievers")
            a.lifted.first_achievers_restricted = false;
        else if (s == "--no-filter-members")
            a.lifted.reachability_filter_members = false;
        else if (s == "--no-verify")
            a.lifted.verify_pi_plus = false;
        else if (s == "--no-static-filter")
            a.lifted.use_static_filter = false;
        else if (s == "--max-combinations")
            a.lifted.max_occurrence_combinations = std::stoul(value());
        else if (s == "--max-members")
            a.lifted.max_disjunctive_members = std::stoul(value());
        else if (s == "--atoms")
        {
            const std::string v = value();
            a.to.atoms = v == "auto" ? TaskOptions::Atoms::Auto : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
        }
        else if (!s.empty() && s[0] != '-' && a.task_file.empty())
            a.task_file = s;
        else
            usage("unknown argument " + s);
    }
    if (a.mode.empty())
        usage("missing --mode");
    try
    {
        std::shared_ptr<const Task> task;
        if (!a.task_file.empty())
            task = Task::from_text_file(a.task_file, a.to);
        else
        {
#if defined(MYMYR_HAS_FRONTEND)
            if (a.domain.empty() || a.problem.empty())
                usage("need a task file or --domain and --problem");
            const auto t0 = Clock::now();
            const auto data = frontend::load_task(a.domain, a.problem);
            std::fprintf(stderr, "parse_ms %.3f\n", ms_since(t0));
            const auto t1 = Clock::now();
            task = Task::create(*data, a.to);
            std::fprintf(stderr, "task_ms %.3f\n", ms_since(t1));
#else
            usage("built without the PDDL front end: pass a task file");
#endif
        }
        if (a.mode == "rr")
            mode_rr(a, *task);
        else if (a.mode == "approx")
            mode_approx(a, *task);
        else if (a.mode == "without")
            mode_without(a, *task);
        else if (a.mode == "lifted")
            mode_lifted(a, *task);
        else if (a.mode == "order")
            mode_order(a, *task);
        else
            usage("unknown mode " + a.mode);
    }
    catch (const std::exception& e)
    {
        std::cout << "RESULT_JSON {\"lib\":\"mymyr\",\"mode\":" << jstr(a.mode) << ",\"status\":\"error\",\"error\":" << jstr(e.what()) << "}"
                  << std::endl;
        return 1;
    }
    return 0;
}
