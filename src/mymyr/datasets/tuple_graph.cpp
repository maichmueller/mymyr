#include "mymyr/datasets/tuple_graph.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/novelty/novelty_table.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace mymyr::datasets
{
// ----------------------------------------------------------------------------------------------------- TupleGraph
u32 TupleGraph::distance(u32 v) const
{
    if (v >= num_vertices())
        throw std::out_of_range("mymyr: tuple graph vertex out of range");
    return static_cast<u32>(std::upper_bound(m_distance_offsets.begin(), m_distance_offsets.end(), v) - m_distance_offsets.begin()) - 1;
}

namespace
{
std::span<const u32> row(const std::vector<u32>& offsets, const std::vector<u32>& values, u32 i, const char* what)
{
    if (i + 1 >= offsets.size())
        throw std::out_of_range(std::string("mymyr: tuple graph ") + what + " out of range");
    return {values.data() + offsets[i], offsets[i + 1] - offsets[i]};
}
}  // namespace

std::span<const u32> TupleGraph::tuple(u32 v) const { return row(m_tuple_offsets, m_tuple_atoms, v, "vertex"); }
std::span<const u32> TupleGraph::problem_vertices(u32 v) const { return row(m_problem_offsets, m_problem_vertices, v, "vertex"); }
std::span<const u32> TupleGraph::successors(u32 v) const { return row(m_succ_offsets, m_succ, v, "vertex"); }
std::span<const u32> TupleGraph::predecessors(u32 v) const { return row(m_pred_offsets, m_pred, v, "vertex"); }
std::span<const u32> TupleGraph::problem_vertices_at(u32 d) const { return row(m_layer_offsets, m_layer_vertices, d, "distance"); }

u64 TupleGraph::bytes() const noexcept
{
    u64 b = 0;
    for (const auto* v : {&m_distance_offsets, &m_tuple_offsets, &m_tuple_atoms, &m_problem_offsets, &m_problem_vertices, &m_succ_offsets,
                          &m_succ, &m_pred_offsets, &m_pred, &m_layer_offsets, &m_layer_vertices})
        b += v->capacity() * sizeof(u32);
    return b;
}

bool operator==(const TupleGraph& a, const TupleGraph& b) noexcept
{
    return a.m_space == b.m_space && a.m_root == b.m_root && a.m_width == b.m_width && a.m_pruning == b.m_pruning &&
           a.m_distance_offsets == b.m_distance_offsets && a.m_tuple_offsets == b.m_tuple_offsets && a.m_tuple_atoms == b.m_tuple_atoms &&
           a.m_problem_offsets == b.m_problem_offsets && a.m_problem_vertices == b.m_problem_vertices && a.m_succ_offsets == b.m_succ_offsets &&
           a.m_succ == b.m_succ && a.m_pred_offsets == b.m_pred_offsets && a.m_pred == b.m_pred && a.m_layer_offsets == b.m_layer_offsets &&
           a.m_layer_vertices == b.m_layer_vertices;
}

// ----------------------------------------------------------------------------------------------------- construction
namespace
{
constexpr u32 K = novelty::k_max_arity;

/// A tuple of 0..K fluent atom slots, ascending; ordered by size, then slots (an internal order for sorting and
/// grouping, not the canonical one).
struct Tuple
{
    u32 n = 0;
    std::array<u32, K> a{};
    [[nodiscard]] std::span<const u32> atoms() const noexcept { return {a.data(), n}; }
    friend bool operator==(const Tuple& x, const Tuple& y) noexcept { return x.n == y.n && std::equal(x.a.begin(), x.a.begin() + x.n, y.a.begin()); }
    friend std::strong_ordering operator<=>(const Tuple& x, const Tuple& y) noexcept
    {
        if (x.n != y.n)
            return x.n <=> y.n;
        for (u32 i = 0; i < x.n; ++i)
            if (x.a[i] != y.a[i])
                return x.a[i] <=> y.a[i];
        return std::strong_ordering::equal;
    }
};

/// The atoms (fluent slots) of a state's words, ascending.
void atoms_of(const u64* w, u32 nw, std::vector<u32>& out)
{
    out.clear();
    bits::for_each(w, nw, [&](u64 b) { out.push_back(static_cast<u32>(b)); });
}

/// Calls f(tuple) for every tuple of 1..width atoms of `atoms` (ascending) that the table has not seen.
template<class F>
void for_each_novel(const novelty::NoveltyTable& table, std::span<const u32> atoms, u32 width, F&& f)
{
    const u32 n = static_cast<u32>(atoms.size());
    Tuple t;
    std::array<u32, K> idx{};
    for (u32 j = 1; j <= width && j <= n; ++j)
    {
        for (u32 i = 0; i < j; ++i)
            idx[i] = i;
        while (true)
        {
            t.n = j;
            for (u32 i = 0; i < j; ++i)
                t.a[i] = atoms[idx[i]];
            if (!table.seen(t.atoms()))
                f(t);
            // next combination of j indices out of n
            i32 i = static_cast<i32>(j) - 1;
            while (i >= 0 && idx[i] == n - j + static_cast<u32>(i))
                --i;
            if (i < 0)
                break;
            ++idx[i];
            for (u32 k = static_cast<u32>(i) + 1; k < j; ++k)
                idx[k] = idx[k - 1] + 1;
        }
    }
}

/// Every tuple of 0..width atoms of `atoms`.
void all_tuples(std::span<const u32> atoms, u32 width, std::vector<Tuple>& out)
{
    out.clear();
    out.push_back(Tuple{});
    const u32 n = static_cast<u32>(atoms.size());
    std::array<u32, K> idx{};
    for (u32 j = 1; j <= width && j <= n; ++j)
    {
        for (u32 i = 0; i < j; ++i)
            idx[i] = i;
        while (true)
        {
            Tuple t;
            t.n = j;
            for (u32 i = 0; i < j; ++i)
                t.a[i] = atoms[idx[i]];
            out.push_back(t);
            i32 i = static_cast<i32>(j) - 1;
            while (i >= 0 && idx[i] == n - j + static_cast<u32>(i))
                --i;
            if (i < 0)
                break;
            ++idx[i];
            for (u32 k = static_cast<u32>(i) + 1; k < j; ++k)
                idx[k] = idx[k - 1] + 1;
        }
    }
}

/// Shared, read-only inputs of one tuple_graphs() call.
struct Shared
{
    StateSpacePtr space;
    TupleGraphOptions options;
};
}  // namespace

/// Builds the tuple graphs of one space on one thread (scratch reused across roots).
class TupleGraphBuilder
{
public:
    explicit TupleGraphBuilder(const Shared& shared)
        : m_sh(shared), m_S(*shared.space), m_task(*m_S.task()), m_width(shared.options.width), m_pruning(shared.options.dominance_pruning),
          m_N(m_S.num_states())
    {
        if (m_width > 0)
        {
            m_table.emplace(m_width, std::max<u32>(m_task.atoms().max_fluent_slots(), 1));
            m_table->reserve(std::max<u32>(m_task.atoms().fluent_slots(), 1));
        }
        m_visit.assign(m_N, 0);
        m_cstamp.assign(m_N, 0);
        m_cpos.assign(m_N, 0);
        m_lstamp.assign(m_N, 0);
        m_lpos.assign(m_N, 0);
    }

    TupleGraph build(u32 root)
    {
        m_out = TupleGraph{};
        m_out.m_space = m_sh.space;
        m_out.m_root = root;
        m_out.m_width = m_width;
        m_out.m_pruning = m_pruning;
        m_preds.clear();
        if (m_width == 0)
            build_width_zero(root);
        else
            build_layers(root);
        finish();
        return std::move(m_out);
    }

private:
    // ------------------------------------------------------------------------------------------- output assembly
    void add_vertex(const Tuple& t, std::span<const u32> problem_vertices, std::span<const u32> preds)
    {
        m_out.m_tuple_atoms.insert(m_out.m_tuple_atoms.end(), t.a.begin(), t.a.begin() + t.n);
        m_out.m_tuple_offsets.push_back(static_cast<u32>(m_out.m_tuple_atoms.size()));
        m_out.m_problem_vertices.insert(m_out.m_problem_vertices.end(), problem_vertices.begin(), problem_vertices.end());
        m_out.m_problem_offsets.push_back(static_cast<u32>(m_out.m_problem_vertices.size()));
        m_preds.insert(m_preds.end(), preds.begin(), preds.end());
        m_out.m_pred_offsets.push_back(static_cast<u32>(m_preds.size()));
    }
    void close_distance(std::span<const u32> layer)
    {
        m_out.m_distance_offsets.push_back(static_cast<u32>(m_out.m_tuple_offsets.size() - 1));
        m_out.m_layer_vertices.insert(m_out.m_layer_vertices.end(), layer.begin(), layer.end());
        m_out.m_layer_offsets.push_back(static_cast<u32>(m_out.m_layer_vertices.size()));
    }
    void finish()
    {
        TupleGraph& g = m_out;
        g.m_pred = std::move(m_preds);
        m_preds = {};
        const u32 V = g.num_vertices();
        std::vector<u32> count(V + 1, 0);
        for (u32 u : g.m_pred)
            ++count[u + 1];
        for (u32 v = 0; v < V; ++v)
            count[v + 1] += count[v];
        g.m_succ_offsets = count;
        g.m_succ.assign(g.m_pred.size(), 0);
        for (u32 v = 0; v < V; ++v)
            for (u32 e = g.m_pred_offsets[v]; e < g.m_pred_offsets[v + 1]; ++e)
                g.m_succ[count[g.m_pred[e]]++] = v;
    }

    // ------------------------------------------------------------------------------------------- canonical order
    const std::string& name(u32 slot)
    {
        if (slot >= m_names.size())
            m_names.resize(slot + 1);
        std::string& s = m_names[slot];
        if (s.empty())
            s = m_task.format(SlotId{slot});
        return s;
    }
    /// The canonical tuple order: fewer atoms first, then the sorted atom names lexicographically.
    bool canonical_less(const Tuple& x, const Tuple& y)
    {
        if (x.n != y.n)
            return x.n < y.n;
        if (x.n == 0)
            return false;
        // name() may grow m_names: size it for both tuples before taking addresses
        (void) name(std::max(*std::max_element(x.a.begin(), x.a.begin() + x.n), *std::max_element(y.a.begin(), y.a.begin() + y.n)));
        std::array<const std::string*, K> nx{}, ny{};
        for (u32 i = 0; i < x.n; ++i)
        {
            nx[i] = &name(x.a[i]);
            ny[i] = &name(y.a[i]);
        }
        // insertion sort: at most K names
        auto sort_names = [](std::array<const std::string*, K>& a, u32 n)
        {
            for (u32 i = 1; i < n; ++i)
                for (u32 j = i; j > 0 && *a[j] < *a[j - 1]; --j)
                    std::swap(a[j], a[j - 1]);
        };
        sort_names(nx, x.n);
        sort_names(ny, y.n);
        for (u32 i = 0; i < x.n; ++i)
            if (*nx[i] != *ny[i])
                return *nx[i] < *ny[i];
        return false;
    }

    // ------------------------------------------------------------------------------------------- width 0
    void build_width_zero(u32 r)
    {
        const u32 root_layer[1] = {r};
        add_vertex(Tuple{}, root_layer, {});
        close_distance(root_layer);
        const auto off = m_S.forward_offsets();
        const auto tgt = m_S.forward_targets();
        std::vector<u32> next(tgt.begin() + static_cast<i64>(off[r]), tgt.begin() + static_cast<i64>(off[r + 1]));
        std::erase(next, r);
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        if (next.empty())
            return;
        const u32 root_vertex[1] = {0};
        for (u32 t : next)
            add_vertex(Tuple{}, std::span<const u32>(&t, 1), root_vertex);
        close_distance(next);
    }

    // ------------------------------------------------------------------------------------------- width >= 1
    void reserve_table(const std::vector<u32>& atoms)
    {
        if (!atoms.empty())
            m_table->reserve(atoms.back() + 1);
    }

    void build_layers(u32 r)
    {
        novelty::NoveltyTable& table = *m_table;
        table.clear();
        ++m_epoch;
        const u32 epoch_visit = m_epoch;

        // distance 0
        const StateView root_state = m_S.state(r);
        atoms_of(root_state.w, root_state.nw, m_atoms);
        reserve_table(m_atoms);
        const u32 root_layer[1] = {r};
        if (m_pruning)
            add_vertex(Tuple{}, root_layer, {});
        else
        {
            all_tuples(m_atoms, m_width, m_tuples);
            std::sort(m_tuples.begin(), m_tuples.end(), [&](const Tuple& a, const Tuple& b) { return canonical_less(a, b); });
            for (const Tuple& t : m_tuples)
                add_vertex(t, root_layer, {});
        }
        close_distance(root_layer);
        table.mark_state(root_state.w, root_state.nw);
        m_layer.assign(1, r);
        m_visit[r] = epoch_visit;

        const auto off = m_S.forward_offsets();
        const auto tgt = m_S.forward_targets();
        while (true)
        {
            // the next breadth-first layer of problem vertices
            m_next_layer.clear();
            for (u32 p : m_layer)
                for (u64 e = off[p]; e < off[p + 1]; ++e)
                    if (const u32 c = tgt[e]; m_visit[c] != epoch_visit)
                    {
                        m_visit[c] = epoch_visit;
                        m_next_layer.push_back(c);
                    }
            if (m_next_layer.empty())
                return;

            // the novel tuples of this layer's states: (tuple, problem vertex) pairs
            m_pairs.clear();
            for (u32 c : m_next_layer)
            {
                const StateView s = m_S.state(c);
                atoms_of(s.w, s.nw, m_atoms);
                reserve_table(m_atoms);
                for_each_novel(table, m_atoms, m_width, [&](const Tuple& t) { m_pairs.emplace_back(t, c); });
            }
            for (u32 c : m_next_layer)
            {
                const StateView s = m_S.state(c);
                table.mark_state(s.w, s.nw);
            }
            std::swap(m_layer, m_next_layer);
            std::sort(m_layer.begin(), m_layer.end());
            if (m_pairs.empty())
                return;
            if (!extend_and_add())
                return;
            close_distance(m_layer);
        }
    }

    /// From this layer's (tuple, problem vertex) pairs: the extended tuples, after dominance pruning, as the vertices
    /// of the next distance. Returns false if there are none.
    bool extend_and_add()
    {
        // tuples T_i with their problem vertices P_i (sorted, unique)
        std::sort(m_pairs.begin(), m_pairs.end());
        m_pairs.erase(std::unique(m_pairs.begin(), m_pairs.end()), m_pairs.end());
        m_tup.clear();
        m_poff.assign(1, 0);
        m_pv.clear();
        for (usize i = 0; i < m_pairs.size(); ++i)
        {
            if (i == 0 || !(m_pairs[i].first == m_pairs[i - 1].first))
            {
                if (i > 0)
                    m_poff.push_back(static_cast<u32>(m_pv.size()));
                m_tup.push_back(m_pairs[i].first);
            }
            m_pv.push_back(m_pairs[i].second);
        }
        m_poff.push_back(static_cast<u32>(m_pv.size()));
        const u32 T = static_cast<u32>(m_tup.size());

        // per problem vertex c with novel tuples: its tuple ids (CSR over the distinct c)
        ++m_epoch;
        const u32 ec = m_epoch;
        m_ct.clear();
        for (u32 i = 0; i < T; ++i)
            for (u32 k = m_poff[i]; k < m_poff[i + 1]; ++k)
                m_ct.emplace_back(m_pv[k], i);
        std::sort(m_ct.begin(), m_ct.end());
        m_coff.clear();
        for (usize k = 0; k < m_ct.size(); ++k)
            if (k == 0 || m_ct[k].first != m_ct[k - 1].first)
            {
                m_cstamp[m_ct[k].first] = ec;
                m_cpos[m_ct[k].first] = static_cast<u32>(m_coff.size());
                m_coff.push_back(static_cast<u32>(k));
            }
        m_coff.push_back(static_cast<u32>(m_ct.size()));

        // extension: t extends u iff every problem vertex of u has a successor with t novel
        const auto off = m_S.forward_offsets();
        const auto tgt = m_S.forward_targets();
        m_cnt.assign(T, 0);
        m_mark.assign(T, 0);
        m_ext_preds.assign(T, {});
        u32 mark = 0;
        const u32 prev_begin = m_out.m_distance_offsets[m_out.m_distance_offsets.size() - 2];
        const u32 prev_end = m_out.m_distance_offsets.back();
        for (u32 u = prev_begin; u < prev_end; ++u)
        {
            const u32 pb = m_out.m_problem_offsets[u], pe = m_out.m_problem_offsets[u + 1];
            m_touched.clear();
            for (u32 k = pb; k < pe; ++k)
            {
                const u32 p = m_out.m_problem_vertices[k];
                ++mark;
                for (u64 e = off[p]; e < off[p + 1]; ++e)
                {
                    const u32 c = tgt[e];
                    if (m_cstamp[c] != ec)
                        continue;
                    const u32 q = m_cpos[c];
                    for (u32 j = m_coff[q]; j < m_coff[q + 1]; ++j)
                    {
                        const u32 i = m_ct[j].second;
                        if (m_mark[i] == mark)
                            continue;
                        m_mark[i] = mark;
                        if (m_cnt[i]++ == 0)
                            m_touched.push_back(i);
                    }
                }
            }
            for (u32 i : m_touched)
            {
                if (m_cnt[i] == pe - pb)
                    m_ext_preds[i].push_back(u);
                m_cnt[i] = 0;
            }
        }
        m_kept.clear();
        for (u32 i = 0; i < T; ++i)
            if (!m_ext_preds[i].empty())
                m_kept.push_back(i);
        if (m_kept.empty())
            return false;
        if (m_pruning)
            prune();

        std::sort(m_kept.begin(), m_kept.end(), [&](u32 a, u32 b) { return canonical_less(m_tup[a], m_tup[b]); });
        for (u32 i : m_kept)
            add_vertex(m_tup[i], std::span<const u32>(m_pv.data() + m_poff[i], m_poff[i + 1] - m_poff[i]), m_ext_preds[i]);
        return true;
    }

    std::span<const u32> pset(u32 i) const { return {m_pv.data() + m_poff[i], m_poff[i + 1] - m_poff[i]}; }

    /// Dominance pruning of m_kept: one tuple (the canonical smallest) per set of problem vertices, and only the sets
    /// with no strict subset among them.
    void prune()
    {
        // equal sets
        std::sort(m_kept.begin(), m_kept.end(),
                  [&](u32 a, u32 b)
                  {
                      const auto x = pset(a), y = pset(b);
                      if (const auto c = std::lexicographical_compare_three_way(x.begin(), x.end(), y.begin(), y.end()); c != 0)
                          return c < 0;
                      return a < b;
                  });
        m_reps.clear();
        for (usize k = 0; k < m_kept.size(); ++k)
        {
            const u32 i = m_kept[k];
            if (k > 0 && std::ranges::equal(pset(i), pset(m_reps.back())))
            {
                if (canonical_less(m_tup[i], m_tup[m_reps.back()]))
                    m_reps.back() = i;
                continue;
            }
            m_reps.push_back(i);
        }
        // strict subsets, by an inverted index: set j is a subset of set i iff all of j's elements are in i
        const u32 R = static_cast<u32>(m_reps.size());
        ++m_epoch;
        const u32 el = m_epoch;
        m_inv.clear();
        for (u32 j = 0; j < R; ++j)
            for (u32 x : pset(m_reps[j]))
                m_inv.emplace_back(x, j);
        std::sort(m_inv.begin(), m_inv.end());
        m_ioff.clear();
        for (usize k = 0; k < m_inv.size(); ++k)
            if (k == 0 || m_inv[k].first != m_inv[k - 1].first)
            {
                m_lstamp[m_inv[k].first] = el;
                m_lpos[m_inv[k].first] = static_cast<u32>(m_ioff.size());
                m_ioff.push_back(static_cast<u32>(k));
            }
        m_ioff.push_back(static_cast<u32>(m_inv.size()));
        m_cnt.assign(R, 0);
        m_kept.clear();
        for (u32 i = 0; i < R; ++i)
        {
            const auto si = pset(m_reps[i]);
            m_touched.clear();
            for (u32 x : si)
            {
                const u32 q = m_lpos[x];
                for (u32 k = m_ioff[q]; k < m_ioff[q + 1]; ++k)
                {
                    const u32 j = m_inv[k].second;
                    if (m_cnt[j]++ == 0)
                        m_touched.push_back(j);
                }
            }
            bool dominated = false;
            for (u32 j : m_touched)
            {
                if (j != i && m_cnt[j] == pset(m_reps[j]).size() && pset(m_reps[j]).size() < si.size())
                    dominated = true;
                m_cnt[j] = 0;
            }
            if (!dominated)
                m_kept.push_back(m_reps[i]);
        }
    }

    const Shared& m_sh;
    const StateSpace& m_S;
    const Task& m_task;
    const u32 m_width;
    const bool m_pruning;
    const u32 m_N;
    std::optional<novelty::NoveltyTable> m_table;
    TupleGraph m_out;
    std::vector<u32> m_preds;
    std::vector<std::string> m_names;

    // breadth-first search and per-layer scratch; stamps compare against m_epoch, so they are never cleared
    u32 m_epoch = 0;
    std::vector<u32> m_visit, m_cstamp, m_cpos, m_lstamp, m_lpos;
    std::vector<u32> m_layer, m_next_layer, m_atoms;
    std::vector<Tuple> m_tuples, m_tup;
    std::vector<std::pair<Tuple, u32>> m_pairs;
    std::vector<u32> m_poff, m_pv, m_coff, m_ioff, m_cnt, m_mark, m_touched, m_kept, m_reps;
    std::vector<std::pair<u32, u32>> m_ct, m_inv;
    std::vector<std::vector<u32>> m_ext_preds;
};

namespace
{
void check(const StateSpacePtr& space, const TupleGraphOptions& options)
{
    if (!space)
        throw std::invalid_argument("mymyr: tuple graph: no state space");
    if (options.width > novelty::k_max_arity)
        throw std::invalid_argument("mymyr: tuple graph width must be in 0.." + std::to_string(novelty::k_max_arity) + ", got " +
                                    std::to_string(options.width));
}
}  // namespace

std::vector<TupleGraph> tuple_graphs(const StateSpacePtr& space, const TupleGraphOptions& options)
{
    check(space, options);
    const u32 N = space->num_states();
    const u32 threads = options.threads == 0 ? std::max<u32>(1, std::thread::hardware_concurrency()) : options.threads;
    const Shared sh{space, options};
    std::vector<TupleGraph> out(N);
    std::atomic<u32> next{0};
    ThreadPool pool(std::min<u32>(threads, std::max<u32>(N, 1)));
    pool.run(
        [&](u32)
        {
            TupleGraphBuilder builder(sh);
            for (u32 v; (v = next.fetch_add(1, std::memory_order_relaxed)) < N;)
                out[v] = builder.build(v);
        });
    return out;
}

TupleGraph tuple_graph(const StateSpacePtr& space, u32 vertex, const TupleGraphOptions& options)
{
    check(space, options);
    if (vertex >= space->num_states())
        throw std::invalid_argument("mymyr: tuple graph: vertex " + std::to_string(vertex) + " is not in the state space (" +
                                    std::to_string(space->num_states()) + " states)");
    const Shared sh{space, options};
    TupleGraphBuilder builder(sh);
    return builder.build(vertex);
}
}  // namespace mymyr::datasets
