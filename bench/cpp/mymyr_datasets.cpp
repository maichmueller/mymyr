// mymyr_datasets: state spaces, generalized state spaces and samplers on PDDL instances, printing one JSON line
// per measurement, for parity, speed and determinism checks against the fork.
//
//   mymyr_datasets ss LIST [--threads T] [--no-fp] [--hash] [--max-states N] [--atoms auto|lazy|frozen] [--text]
//                          [--remove-if-unsolvable] [--no-labels]
//       One JSON line per line "domain.pddl problem.pddl [task.txt]" of LIST (--text: read the exported task.txt of
//       the third column instead of the PDDL). Defaults as the fork_ss configuration: remove_if_unsolvable = false,
//       frozen atoms. --hash adds "arrays_hash": FNV-1a over the bytes of every array of the space
//       (state words, forward CSR, labels, costs, reverse CSR, distances, flags), for the determinism gate.
//   mymyr_datasets pool LIST --threads T [--repeat R] [--text] [--atoms ...] [--quiet]
//       The instance pool (datasets::for_each_state_space) over LIST repeated R times: one JSON line per task unless
//       --quiet, then the probe's summary line (bench "pool": wall_s, states_per_s, ...).
//   mymyr_datasets gss DOMAIN PROBLEM... [--threads T]
//   mymyr_datasets sampler DOMAIN PROBLEM [--seed S] [--samples N]
//       As fork_datasets gss / sampler.
//   mymyr_datasets tg DOMAIN PROBLEM [--width W] [--no-pruning] [--tg-threads T] [state space options]
//       The tuple graphs of every vertex of the state space (datasets/tuple_graph.hpp) on T threads: seconds, peak RSS
//       before and after, total vertices and edges; as search_fork --algo tuple_graphs --time-width W.
//
// Fingerprints (FNV-1a 64 over bytes, u64 as 8 little-endian bytes):
//   atom string  "(pred o1 ... ok)" of every true fluent atom; state hash = fnv(sorted atom strings joined by '\n'),
//   numeric values appended as "\n=<%.17g>" in slot order; content = fnv(sorted state hashes); transitions =
//   fnv(sorted fnv(src hash, fnv("(schema o1 ... ok)"), dst hash)); goal = fnv(sorted goal state hashes);
//   vstar = fnv(sorted fnv(state hash, unit distance)), with the fork's INT32_MAX for unsolvable states.

#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

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
                 "error: %s\nusage: mymyr_datasets ss LIST [--threads T] [--no-fp] [--hash] [--max-states N]\n"
                 "                          [--atoms auto|lazy|frozen] [--text] [--remove-if-unsolvable] [--no-labels]\n"
                 "       mymyr_datasets pool LIST --threads T [--repeat R] [--text] [--atoms ...] [--quiet]\n"
                 "       mymyr_datasets gss DOMAIN PROBLEM... [--threads T]\n"
                 "       mymyr_datasets sampler DOMAIN PROBLEM [--seed S] [--samples N]\n"
                 "       mymyr_datasets tg DOMAIN PROBLEM [--width W] [--no-pruning] [--tg-threads T]\n",
                 msg.c_str());
    std::exit(2);
}

struct Fnv
{
    u64 h = 0xcbf29ce484222325ULL;
    void bytes(const void* p, usize n)
    {
        const auto* b = static_cast<const unsigned char*>(p);
        for (usize i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 0x100000001b3ULL;
        }
    }
    void str(const std::string& s) { bytes(s.data(), s.size()); }
    void u64le(u64 x)
    {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i)
            b[i] = static_cast<unsigned char>(x >> (8 * i));
        bytes(b, 8);
    }
    template <class T>
    void span(std::span<const T> s)
    {
        u64le(s.size());
        bytes(s.data(), s.size_bytes());
    }
};
u64 fnv_str(const std::string& s)
{
    Fnv f;
    f.str(s);
    return f.h;
}
u64 fnv_list(std::vector<u64> v)
{
    std::sort(v.begin(), v.end());
    Fnv f;
    for (u64 x : v)
        f.u64le(x);
    return f.h;
}
std::string hex(u64 x)
{
    char b[32];
    std::snprintf(b, sizeof b, "%016" PRIx64, x);
    return b;
}

/// Per-state hashes of a space in id order (the fork driver's state_hash).
std::vector<u64> state_hashes(const StateSpace& S)
{
    const Task& task = *S.task();
    std::vector<std::string> names(static_cast<usize>(S.words()) * 64);  // by fluent slot, formatted on first use
    std::vector<u64> out(S.num_states());
    std::vector<const std::string*> atoms;
    std::string key;
    for (u32 id = 0; id < S.num_states(); ++id)
    {
        const StateView s = S.state(id);
        atoms.clear();
        for (u32 w = 0; w < s.nw; ++w)
            for (u64 bits = s.w[w]; bits; bits &= bits - 1)
            {
                const u32 slot = w * 64 + static_cast<u32>(std::countr_zero(bits));
                if (names[slot].empty())
                    names[slot] = task.format(SlotId{slot});
                atoms.push_back(&names[slot]);
            }
        std::sort(atoms.begin(), atoms.end(), [](const std::string* a, const std::string* b) { return *a < *b; });
        key.clear();
        for (usize i = 0; i < atoms.size(); ++i)
        {
            if (i)
                key += '\n';
            key += *atoms[i];
        }
        if (task.numeric_slots())
            for (f64 v : task.numeric_values(s))
            {
                char b[64];
                std::snprintf(b, sizeof b, "\n=%.17g", v);
                key += b;
            }
        out[id] = fnv_str(key);
    }
    return out;
}

std::string histogram_json(const std::map<long long, u64>& h)
{
    std::string o = "{";
    bool first = true;
    for (const auto& [k, v] : h)
    {
        o += (first ? "\"" : ",\"") + std::to_string(k) + "\":" + std::to_string(v);
        first = false;
    }
    return o + "}";
}
std::string dhist_json(const std::map<double, u64>& h)
{
    std::string o = "{";
    bool first = true;
    for (const auto& [k, v] : h)
    {
        char b[64];
        if (std::isinf(k))
            std::snprintf(b, sizeof b, "inf");
        else
            std::snprintf(b, sizeof b, "%.17g", k);
        o += (first ? "\"" : ",\"") + std::string(b) + "\":" + std::to_string(v);
        first = false;
    }
    return o + "}";
}

/// FNV over every array of the space (the determinism gate).
u64 arrays_hash(const StateSpace& S)
{
    Fnv f;
    f.u64le(S.num_states());
    f.u64le(S.row_words());
    f.span(S.state_words());
    f.span(S.forward_offsets());
    f.span(S.forward_targets());
    f.span(S.label_schemas());
    f.span(S.label_bindings());
    f.span(S.costs());
    f.span(S.backward_offsets());
    f.span(S.backward_sources());
    f.span(S.backward_edges());
    f.span(S.unit_goal_distances());
    f.span(S.cost_goal_distances());
    f.span(S.goal_flags());
    f.span(S.unsolvable_flags());
    f.span(S.alive_flags());
    return f.h;
}

std::string state_space_json(const StateSpace& S, bool fp, bool hash)
{
    std::map<long long, u64> vstar;
    std::map<double, u64> cstar;
    const auto unit = S.unit_goal_distances();
    const auto cost = S.cost_goal_distances();
    u64 alive = 0;
    for (u32 v = 0; v < S.num_states(); ++v)
    {
        ++vstar[unit[v]];
        ++cstar[cost[v]];
        alive += S.is_alive(v);
    }
    std::string o = ",\"states\":" + std::to_string(S.num_states()) + ",\"transitions\":" + std::to_string(S.num_transitions()) +
                    ",\"n_goal\":" + std::to_string(S.num_goal_states()) + ",\"n_unsolvable\":" + std::to_string(S.num_unsolvable_states()) +
                    ",\"n_alive\":" + std::to_string(alive) + ",\"max_goal_dist\":" + std::to_string(S.max_goal_distance()) +
                    ",\"vstar_hist\":" + histogram_json(vstar) + ",\"cstar_hist\":" + dhist_json(cstar);
    if (fp)
    {
        const Task& task = *S.task();
        const std::vector<u64> hs = state_hashes(S);
        std::vector<u64> trans, goals, vst;
        trans.reserve(S.num_transitions());
        const auto off = S.forward_offsets();
        const auto tgt = S.forward_targets();
        for (u32 v = 0; v < S.num_states(); ++v)
            for (u64 e = off[v]; e < off[v + 1]; ++e)
            {
                Fnv f;
                f.u64le(hs[v]);
                f.u64le(fnv_str(task.format(S.label(e).label())));
                f.u64le(hs[tgt[e]]);
                trans.push_back(f.h);
            }
        for (u32 v = 0; v < S.num_states(); ++v)
        {
            if (S.is_goal(v))
                goals.push_back(hs[v]);
            Fnv f;
            f.u64le(hs[v]);
            const i64 d = unit[v] == k_unsolvable_distance ? std::numeric_limits<i32>::max() : unit[v];
            f.u64le(static_cast<u64>(d));
            vst.push_back(f.h);
        }
        std::vector<u64> sorted = hs;
        std::sort(sorted.begin(), sorted.end());
        const bool unique = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        o += ",\"fp_content\":\"" + hex(fnv_list(hs)) + "\",\"fp_transitions\":\"" + hex(fnv_list(trans)) + "\",\"fp_goal\":\"" +
             hex(fnv_list(goals)) + "\",\"fp_vstar\":\"" + hex(fnv_list(vst)) + "\",\"state_hashes_unique\":" + (unique ? "true" : "false");
    }
    if (hash)
        o += ",\"arrays_hash\":\"" + hex(arrays_hash(S)) + "\"";
    return o;
}

struct Line
{
    std::string domain, problem, text;
};
std::vector<Line> read_list(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        usage("cannot read " + path);
    std::vector<Line> out;
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream ls(line);
        Line l;
        ls >> l.domain >> l.problem >> l.text;
        if (!l.domain.empty())
            out.push_back(std::move(l));
    }
    return out;
}

TaskOptions::Atoms parse_atoms(const std::string& v)
{
    if (v == "lazy")
        return TaskOptions::Atoms::Lazy;
    if (v == "frozen")
        return TaskOptions::Atoms::Frozen;
    if (v == "auto")
        return TaskOptions::Atoms::Auto;
    usage("unknown --atoms " + v);
}

/// Loads one task: the exported text (text) or the PDDL pair. Returns the parse and task-creation seconds.
TaskPtr load(const Line& l, bool text, const TaskOptions& to, double* parse_s, double* task_s)
{
    const auto t0 = Clock::now();
    if (text)
    {
        if (l.text.empty())
            usage("--text needs a third column (task.txt) in the list");
        TaskPtr t = Task::from_text_file(l.text, to);
        *parse_s = seconds_since(t0);
        *task_s = 0;
        return t;
    }
#if defined(MYMYR_HAS_FRONTEND)
    const auto data = frontend::load_task(l.domain, l.problem);
    *parse_s = seconds_since(t0);
    const auto t1 = Clock::now();
    TaskPtr t = Task::create(*data, to);
    *task_s = seconds_since(t1);
    return t;
#else
    (void)to;
    (void)parse_s;
    (void)task_s;
    usage("built without the PDDL front end: use --text");
#endif
}

struct Common
{
    StateSpaceOptions so;
    TaskOptions to;
    bool fp = true, hash = false, text = false, quiet = false;
    u32 repeat = 1;
    u64 seed = 1, samples = 20000;
    TupleGraphOptions tg;
    std::vector<std::string> positional;
};

Common parse_common(int argc, char** argv)
{
    Common c;
    c.so.remove_if_unsolvable = false;
    c.to.atoms = TaskOptions::Atoms::Frozen;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage("missing value for " + a);
            return argv[++i];
        };
        if (a == "--threads")
            c.so.threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--no-fp")
            c.fp = false;
        else if (a == "--hash")
            c.hash = true;
        else if (a == "--max-states")
            c.so.max_states = std::stoull(value());
        else if (a == "--max-seconds")
            c.so.max_seconds = std::stod(value());
        else if (a == "--atoms")
            c.to.atoms = parse_atoms(value());
        else if (a == "--text")
            c.text = true;
        else if (a == "--remove-if-unsolvable")
            c.so.remove_if_unsolvable = true;
        else if (a == "--no-labels")
            c.so.labels = false;
        else if (a == "--repeat")
            c.repeat = static_cast<u32>(std::stoul(value()));
        else if (a == "--quiet")
            c.quiet = true;
        else if (a == "--seed")
            c.seed = std::stoull(value());
        else if (a == "--samples")
            c.samples = std::stoull(value());
        else if (a == "--width")
            c.tg.width = static_cast<u32>(std::stoul(value()));
        else if (a == "--no-pruning")
            c.tg.dominance_pruning = false;
        else if (a == "--tg-threads")
            c.tg.threads = static_cast<u32>(std::stoul(value()));
        else if (!a.starts_with("--"))
            c.positional.push_back(a);
        else
            usage("unknown argument " + a);
    }
    return c;
}

int run_ss(const Common& c)
{
    if (c.positional.size() != 1)
        usage("ss needs one LIST");
    for (const Line& l : read_list(c.positional[0]))
    {
        double parse_s = 0, task_s = 0;
        const auto t0 = Clock::now();
        const TaskPtr task = load(l, c.text, c.to, &parse_s, &task_s);
        const auto t1 = Clock::now();
        const StateSpaceResult r = generate_state_space(task, c.so);
        const double ss_s = seconds_since(t1), total_s = seconds_since(t0);
        std::string o = "{\"lib\":\"mymyr\",\"mode\":\"" + std::string(c.text ? "text" : "pddl") + "\",\"domain\":\"" + l.domain +
                        "\",\"problem\":\"" + l.problem + "\",\"ok\":" + (r.space ? "true" : "false") + ",\"status\":\"" + to_string(r.status) + "\"";
        char tb[512];
        std::snprintf(tb, sizeof tb, ",\"threads\":%u,\"parse_s\":%.5f,\"task_s\":%.5f,\"statespace_s\":%.5f,\"total_s\":%.5f", c.so.threads,
                      parse_s, task_s, ss_s, total_s);
        o += tb;
        if (r.space)
        {
            const StateSpace& S = *r.space;
            std::snprintf(tb, sizeof tb, ",\"search_s\":%.5f,\"post_s\":%.5f,\"layers\":%u,\"words\":%u,\"bytes\":%llu", S.search_seconds(),
                          S.post_seconds(), S.layers(), S.row_words(), static_cast<unsigned long long>(S.bytes()));
            o += tb;
            o += state_space_json(S, c.fp, c.hash);
        }
        std::snprintf(tb, sizeof tb, ",\"peak_rss_mb\":%.1f}", peak_rss_mb());
        o += tb;
        std::printf("%s\n", o.c_str());
        std::fflush(stdout);
    }
    return 0;
}

int run_pool(const Common& c)
{
    if (c.positional.size() != 1)
        usage("pool needs one LIST");
    const std::vector<Line> lines = read_list(c.positional[0]);
    const u64 count = lines.size() * c.repeat;
    std::vector<std::string> per(count);
    std::vector<double> per_s(count, 0);
    std::atomic<u64> states{0}, transitions{0};
    std::vector<Clock::time_point> started(count);
    StateSpaceOptions so = c.so;
    const u32 T = so.threads;
    so.threads = 1;
    const auto t0 = Clock::now();
    for_each_state_space(
        count,
        [&](u64 i)
        {
            started[i] = Clock::now();
            double p = 0, t = 0;
            return load(lines[i % lines.size()], c.text, c.to, &p, &t);
        },
        [&](u64 i, StateSpaceResult&& r)
        {
            per_s[i] = seconds_since(started[i]);
            const u64 n = r.space ? r.space->num_states() : 0, e = r.space ? r.space->num_transitions() : 0;
            states += n;
            transitions += e;
            if (!c.quiet)
            {
                const Line& l = lines[i % lines.size()];
                char b[256];
                std::snprintf(b, sizeof b, "\"states\":%llu,\"transitions\":%llu,\"n_goal\":%u,\"n_unsolvable\":%u,\"total_s\":%.5f}",
                              static_cast<unsigned long long>(n), static_cast<unsigned long long>(e),
                              r.space ? r.space->num_goal_states() : 0, r.space ? r.space->num_unsolvable_states() : 0, per_s[i]);
                per[i] = "{\"domain\":\"" + l.domain + "\",\"problem\":\"" + l.problem + "\",\"ok\":" + (r.space ? "true," : "false,") + b;
            }
        },
        so, T);
    const double wall = seconds_since(t0);
    double sum = 0;
    for (double x : per_s)
        sum += x;
    if (!c.quiet)
        for (const std::string& s : per)
            std::printf("%s\n", s.c_str());
    std::printf("{\"bench\":\"pool\",\"lib\":\"mymyr\",\"T\":%u,\"input\":\"%s\",\"instances\":%llu,\"states\":%llu,\"transitions\":%llu,"
                "\"wall_s\":%.3f,\"sum_task_s\":%.3f,\"instances_per_s\":%.1f,\"states_per_s\":%.0f,\"transitions_per_s\":%.0f,"
                "\"peak_rss_mb\":%.1f}\n",
                T, c.text ? "text" : "pddl", static_cast<unsigned long long>(count), static_cast<unsigned long long>(states.load()),
                static_cast<unsigned long long>(transitions.load()), wall, sum, static_cast<double>(count) / wall,
                static_cast<double>(states.load()) / wall, static_cast<double>(transitions.load()) / wall, peak_rss_mb());
    return 0;
}

int run_gss(const Common& c)
{
#if defined(MYMYR_HAS_FRONTEND)
    if (c.positional.size() < 2)
        usage("gss needs DOMAIN PROBLEM...");
    const auto t0 = Clock::now();
    const auto domain = frontend::Domain::from_file(c.positional[0]);
    std::vector<TaskPtr> tasks;
    std::unordered_map<const Task*, std::string> path_of;
    for (usize i = 1; i < c.positional.size(); ++i)
    {
        const auto data = domain->instantiate_file(c.positional[i]);
        tasks.push_back(Task::create(*data, c.to));
        path_of[tasks.back().get()] = c.positional[i];
    }
    StateSpaceOptions so = c.so;
    const u32 T = so.threads;
    so.threads = 1;
    const std::vector<StateSpaceResult> results = generate_state_spaces(tasks, so, T);
    const auto gss = GeneralizedStateSpace::create(ordered_spaces(results, true));
    const double total_s = seconds_since(t0);
    std::vector<std::vector<u64>> hashes;
    std::string order = "[";
    for (usize i = 0; i < gss->spaces().size(); ++i)
    {
        hashes.push_back(state_hashes(*gss->spaces()[i]));
        order += (i ? ",\"" : "\"") + path_of.at(gss->spaces()[i]->task().get()) + "\"";
    }
    order += "]";
    auto rep = [&](u32 v) -> std::pair<u64, u64>
    {
        const u32 p = gss->problem_of(v);
        return {p, hashes.at(p).at(v - gss->vertex_offsets()[p])};
    };
    std::vector<u64> vfp, efp, mfp, ifp, gfp, ufp;
    for (u32 v = 0; v < gss->num_vertices(); ++v)
    {
        auto [p, h] = rep(v);
        Fnv f;
        f.u64le(p);
        f.u64le(h);
        vfp.push_back(f.h);
        if (gss->initial_flags()[v])
            ifp.push_back(f.h);
        if (gss->goal_flags()[v])
            gfp.push_back(f.h);
        if (gss->unsolvable_flags()[v])
            ufp.push_back(f.h);
    }
    for (u32 v = 0; v < gss->num_vertices(); ++v)
        for (u64 e = gss->forward_offsets()[v]; e < gss->forward_offsets()[v + 1]; ++e)
        {
            auto [ps, hs] = rep(v);
            auto [pt, ht] = rep(gss->forward_targets()[e]);
            Fnv f;
            f.u64le(ps);
            f.u64le(hs);
            f.u64le(pt);
            f.u64le(ht);
            efp.push_back(f.h);
        }
    for (u32 p = 0; p < gss->spaces().size(); ++p)
        for (u32 v = 0; v < gss->spaces()[p]->num_states(); ++v)
        {
            auto [cp, ch] = rep(gss->vertex(p, v));
            Fnv f;
            f.u64le(p);
            f.u64le(hashes[p][v]);
            f.u64le(cp);
            f.u64le(ch);
            mfp.push_back(f.h);
        }
    std::printf("{\"lib\":\"mymyr\",\"mode\":\"gss\",\"domain\":\"%s\",\"problems\":%zu,\"state_spaces\":%zu,\"order\":%s,"
                "\"class_vertices\":%u,\"class_edges\":%llu,\"initial\":%zu,\"goal\":%zu,\"unsolvable\":%zu,"
                "\"fp_vertices\":\"%s\",\"fp_edges\":\"%s\",\"fp_mapping\":\"%s\",\"fp_initial\":\"%s\",\"fp_goal\":\"%s\",\"fp_unsolvable\":\"%s\","
                "\"total_s\":%.4f,\"peak_rss_mb\":%.1f}\n",
                c.positional[0].c_str(), c.positional.size() - 1, gss->spaces().size(), order.c_str(),
                gss->num_vertices(), static_cast<unsigned long long>(gss->num_edges()), ifp.size(), gfp.size(), ufp.size(),
                hex(fnv_list(vfp)).c_str(), hex(fnv_list(efp)).c_str(), hex(fnv_list(mfp)).c_str(), hex(fnv_list(ifp)).c_str(),
                hex(fnv_list(gfp)).c_str(), hex(fnv_list(ufp)).c_str(), total_s, peak_rss_mb());
    return 0;
#else
    (void)c;
    usage("gss needs the PDDL front end");
#endif
}

int run_sampler(const Common& c)
{
#if defined(MYMYR_HAS_FRONTEND)
    if (c.positional.size() != 2)
        usage("sampler needs DOMAIN PROBLEM");
    const auto data = frontend::load_task(c.positional[0], c.positional[1]);
    StateSpaceOptions so;  // the fork's StateSpaceSampler defaults: remove_if_unsolvable = true
    const StateSpaceResult r = generate_state_space(Task::create(*data, c.to), so);
    const std::string& problem = c.positional[1];
    if (!r.space)
    {
        std::printf("{\"lib\":\"mymyr\",\"mode\":\"sampler\",\"problem\":\"%s\",\"ok\":false}\n", problem.c_str());
        return 0;
    }
    StateSpaceSampler sampler(r.space, c.seed);
    const std::vector<u64> hs = state_hashes(*r.space);
    const auto unit = r.space->unit_goal_distances();
    auto vstar = [&](u32 s) -> long long { return unit[s] == k_unsolvable_distance ? -1 : unit[s]; };
    std::map<u64, u64> hits_any, hits_dead;
    std::map<long long, u64> d_any;
    Fnv seq;
    for (u64 i = 0; i < c.samples; ++i)
    {
        const u32 s = sampler.sample_state();
        seq.u64le(s);
        ++hits_any[hs[s]];
        ++d_any[vstar(s)];
    }
    u64 dead_wrong = 0;
    if (sampler.num_dead_end_states() > 0)
        for (u64 i = 0; i < c.samples; ++i)
        {
            const u32 s = sampler.sample_dead_end_state();
            seq.u64le(s);
            dead_wrong += !r.space->is_unsolvable(s);
            ++hits_dead[hs[s]];
        }
    std::string per_n = "{";
    for (u32 n = 0; n <= sampler.max_steps_to_goal(); ++n)
    {
        std::map<u64, u64> hits;
        u64 wrong = 0;
        for (u64 i = 0; i < c.samples / 10; ++i)
        {
            const u32 s = sampler.sample_state_n_steps_from_goal(static_cast<i32>(n));
            seq.u64le(s);
            wrong += vstar(s) != n;
            ++hits[hs[s]];
        }
        per_n += (n ? ",\"" : "\"") + std::to_string(n) + "\":{\"support\":" + std::to_string(hits.size()) + ",\"wrong\":" + std::to_string(wrong) + "}";
    }
    per_n += "}";
    std::map<long long, u64> by_d;
    for (u32 s = 0; s < r.space->num_states(); ++s)
        ++by_d[vstar(s)];
    u64 max_hits = 0, min_hits = ~u64{0};
    for (const auto& [h, k] : hits_any)
        max_hits = std::max(max_hits, k), min_hits = std::min(min_hits, k);
    std::printf("{\"lib\":\"mymyr\",\"mode\":\"sampler\",\"problem\":\"%s\",\"ok\":true,\"seed\":%llu,\"samples\":%llu,"
                "\"num_states\":%u,\"num_dead_end_states\":%u,\"num_alive_states\":%u,\"max_steps_to_goal\":%u,"
                "\"states_by_vstar\":%s,\"sample_state_vstar_hist\":%s,\"sample_state_support\":%zu,\"sample_state_min_hits\":%llu,"
                "\"sample_state_max_hits\":%llu,\"dead_end_support\":%zu,\"dead_end_wrong\":%llu,\"n_steps\":%s,\"samples_hash\":\"%s\"}\n",
                problem.c_str(), static_cast<unsigned long long>(c.seed), static_cast<unsigned long long>(c.samples), sampler.num_states(),
                sampler.num_dead_end_states(), sampler.num_alive_states(), sampler.max_steps_to_goal(), histogram_json(by_d).c_str(),
                histogram_json(d_any).c_str(), hits_any.size(), static_cast<unsigned long long>(min_hits),
                static_cast<unsigned long long>(max_hits), hits_dead.size(), static_cast<unsigned long long>(dead_wrong), per_n.c_str(),
                hex(seq.h).c_str());
    return 0;
#else
    (void)c;
    usage("sampler needs the PDDL front end");
#endif
}

int run_tg(const Common& c)
{
#if defined(MYMYR_HAS_FRONTEND)
    if (c.positional.size() != 2)
        usage("tg needs DOMAIN PROBLEM");
    const auto data = frontend::load_task(c.positional[0], c.positional[1]);
    const auto t0 = Clock::now();
    const StateSpaceResult r = generate_state_space(Task::create(*data, c.to), c.so);
    const double space_s = seconds_since(t0);
    if (!r.space)
    {
        std::printf("{\"lib\":\"mymyr\",\"mode\":\"tg\",\"problem\":\"%s\",\"ok\":false}\n", c.positional[1].c_str());
        return 0;
    }
    const double rss_before = peak_rss_mb();
    const auto t1 = Clock::now();
    const std::vector<TupleGraph> graphs = tuple_graphs(r.space, c.tg);
    const double tg_s = seconds_since(t1);
    u64 vertices = 0, edges = 0, bytes = 0;
    for (const TupleGraph& g : graphs)
        vertices += g.num_vertices(), edges += g.num_edges(), bytes += g.bytes();
    std::printf("{\"lib\":\"mymyr\",\"mode\":\"tg\",\"problem\":\"%s\",\"ok\":true,\"states\":%u,\"width\":%u,\"pruning\":%s,"
                "\"threads\":%u,\"seconds\":%.6f,\"state_space_seconds\":%.6f,\"tuple_vertices\":%llu,\"tuple_edges\":%llu,"
                "\"graph_bytes\":%llu,\"peak_rss_mb_before\":%.1f,\"peak_rss_mb_after\":%.1f}\n",
                c.positional[1].c_str(), r.space->num_states(), c.tg.width, c.tg.dominance_pruning ? "true" : "false", c.tg.threads, tg_s,
                space_s, static_cast<unsigned long long>(vertices), static_cast<unsigned long long>(edges),
                static_cast<unsigned long long>(bytes), rss_before, peak_rss_mb());
    return 0;
#else
    (void)c;
    usage("tg needs the PDDL front end");
#endif
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
        usage("missing mode");
    const std::string mode = argv[1];
    try
    {
        const Common c = parse_common(argc, argv);
        if (mode == "ss")
            return run_ss(c);
        if (mode == "pool")
            return run_pool(c);
        if (mode == "gss")
            return run_gss(c);
        if (mode == "sampler")
            return run_sampler(c);
        if (mode == "tg")
            return run_tg(c);
        usage("unknown mode " + mode);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
