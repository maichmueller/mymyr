// Lifted goal templates (rl/goal_templates.hpp).

#include "mymyr/rl/goal_templates.hpp"

#include "mymyr/rl/rng.hpp"
#include "mymyr/encoding/literal_index.hpp"
#include "mymyr/successor/bindings.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mymyr::rl
{
namespace
{
[[noreturn]] void invalid(const std::string& what) { throw std::invalid_argument("mymyr: GoalTemplates: " + what); }

/// binom(n, c), saturated at cap.
u64 binom_capped(u64 n, u64 c, u64 cap)
{
    if (c > n)
        return 0;
    c = std::min(c, n - c);
    u64 v = 1;
    for (u64 i = 1; i <= c; ++i)  // v = binom(n, i) is increasing in i for i <= n / 2
    {
        v = v * (n - i + 1) / i;
        if (v >= cap)
            return cap;
    }
    return std::min(v, cap);
}

/// The connected k-subsets of one component in lexicographic order. Vertices are the component's literals in
/// ascending order (local ids), adjacency is a bitset row per vertex.
class SubsetSearch
{
public:
    SubsetSearch(u32 m, u32 words, const std::vector<u64>& adj) : m_m(m), m_w(words), m_adj(adj), m_after(words), m_cand(words),
                                                                 m_seen(words), m_frontier(words), m_nb(words)
    {
    }

    /// Appends the first `limit` connected k-subsets (local ids) to out.
    void run(u32 k, u32 limit, std::vector<std::vector<u32>>& out)
    {
        m_k = k;
        m_limit = limit;
        m_found = 0;
        m_out = &out;
        m_set.clear();
        m_bits.assign(m_w, 0);
        // joinable and rec recurse at most k deep; sized up front, so that pointers and references into the
        // per-level scratch stay valid across the recursion
        m_level_nb.resize(std::max<usize>(m_level_nb.size(), k + 1));
        m_level_cand.resize(std::max<usize>(m_level_cand.size(), k + 1));
        m_level_after.resize(std::max<usize>(m_level_after.size(), static_cast<usize>(k + 1) * m_w));
        m_children.resize(std::max<usize>(m_children.size(), static_cast<usize>(k + 1) * m_w));
        rec(0);
    }

private:
    [[nodiscard]] const u64* row(u32 v) const noexcept { return m_adj.data() + static_cast<usize>(v) * m_w; }
    [[nodiscard]] u64* scratch(std::vector<u64>& v, usize slots)
    {
        if (v.size() < slots * m_w)
            v.resize(slots * m_w);
        return v.data();
    }

    /// m_after = the vertices >= start: the vertices a completion of the current prefix may add.
    void set_after(u32 start)
    {
        for (u32 w = 0; w < m_w; ++w)
        {
            const u32 lo = w * 64;
            m_after[w] = start <= lo ? ~u64{0} : start >= lo + 64 ? 0 : ~u64{0} << (start - lo);
            if (w == m_w - 1 && m_m % 64)
                m_after[w] &= (u64{1} << (m_m % 64)) - 1;
        }
    }

    /// The parts (connected components) of the vertex set m_set (m_bits): per part, the neighbours of its vertices in
    /// m_after outside the set, into nbs (one row of m_w words per part). Returns the number of parts.
    u32 parts(std::vector<u64>& nbs)
    {
        const u32 n = static_cast<u32>(m_set.size());
        m_part.assign(n, 0);
        u32 count = 0;
        for (u32 i = 0; i < n; ++i)
        {
            if (m_part[i])
                continue;
            u64* nb = scratch(nbs, count + 1) + static_cast<usize>(count) * m_w;
            std::fill(nb, nb + m_w, 0);
            m_part[i] = 1;
            m_stack.assign(1, i);
            while (!m_stack.empty())
            {
                const u64* r = row(m_set[m_stack.back()]);
                m_stack.pop_back();
                for (u32 u = 0; u < m_w; ++u)
                    nb[u] |= r[u];
                for (u32 q = 0; q < n; ++q)
                    if (!m_part[q] && (r[m_set[q] / 64] >> (m_set[q] % 64) & 1))
                    {
                        m_part[q] = 1;
                        m_stack.push_back(q);
                    }
            }
            for (u32 u = 0; u < m_w; ++u)
                nb[u] &= m_after[u] & ~m_bits[u];
            ++count;
        }
        return count;
    }

    void add(u32 v)
    {
        m_set.push_back(v);
        m_bits[v / 64] |= u64{1} << (v % 64);
    }
    void remove_last()
    {
        const u32 v = m_set.back();
        m_set.pop_back();
        m_bits[v / 64] &= ~(u64{1} << (v % 64));
    }

    /// A family of parts no two of which share a neighbour (nbs: t rows from parts()), chosen greedily by increasing
    /// neighbourhood size so that it tends to be large: each of its parts needs a joining vertex of its own. Returns
    /// its size (~0u if a part has no neighbour) and leaves the union of its neighbourhoods in m_seen.
    u32 independent_parts(const std::vector<u64>& nbs, u32 t)
    {
        m_order.resize(t);
        m_count.resize(t);
        for (u32 p = 0; p < t; ++p)
        {
            const u64* nb = nbs.data() + static_cast<usize>(p) * m_w;
            u32 count = 0;
            for (u32 u = 0; u < m_w; ++u)
                count += static_cast<u32>(std::popcount(nb[u]));
            if (count == 0)
                return ~0u;
            m_order[p] = p;
            m_count[p] = count;
        }
        std::ranges::sort(m_order, [&](u32 a, u32 b) { return m_count[a] < m_count[b] || (m_count[a] == m_count[b] && a < b); });
        std::fill(m_seen.begin(), m_seen.end(), 0);
        u32 independent = 0;
        for (u32 p : m_order)
        {
            const u64* nb = nbs.data() + static_cast<usize>(p) * m_w;
            bool shared = false;
            for (u32 u = 0; u < m_w && !shared; ++u)
                shared = (nb[u] & m_seen[u]) != 0;
            if (shared)
                continue;
            ++independent;
            for (u32 u = 0; u < m_w; ++u)
                m_seen[u] |= nb[u];
        }
        return independent;
    }

    /// Whether at most r vertices of m_after may join m_set into one connected set: false only if none do. Every
    /// joining set holds a neighbour of each part (parts are not adjacent to each other), so the search branches over
    /// the neighbours of the part with the fewest; a vertex adjacent to every part settles it at once, and a set of parts
    /// no two of which share a neighbour (each needs a vertex of its own) bounds it from below. After m_budget calls
    /// the answer is true (not ruled out), which keeps dense components whose parts cannot be joined cheap.
    bool joinable(u32 r, u32 depth)
    {
        if (m_budget == 0)
            return true;
        --m_budget;
        std::vector<u64>& nbs = m_level_nb[depth];
        const u32 t = parts(nbs);
        if (t == 1)
            return true;
        if (r == 0)
            return false;
        // a vertex adjacent to every part; the attachment bound; the part with the fewest neighbours
        const u32 independent = independent_parts(nbs, t);
        if (independent == ~0u)
            return false;
        const u32 best = m_order[0];
        bool joiner = false;
        std::fill(m_cand.begin(), m_cand.end(), ~u64{0});
        for (u32 p = 0; p < t; ++p)
            for (u32 u = 0; u < m_w; ++u)
                m_cand[u] &= nbs[static_cast<usize>(p) * m_w + u];
        for (u32 u = 0; u < m_w && !joiner; ++u)
            joiner = m_cand[u] != 0;
        if (joiner)
            return true;
        if (independent > r || r == 1)
            return false;
        // branch over the neighbours of part `best`, those adjacent to more of the other parts first (they join
        // several parts at once, so a joining set is found early when there is one)
        std::vector<std::pair<u32, u32>>& cand = m_level_cand[depth];  // (parts adjacent, vertex)
        cand.clear();
        const u64* nb_best = nbs.data() + static_cast<usize>(best) * m_w;
        for (u32 w = 0; w < m_w; ++w)
            for (u64 x = nb_best[w]; x; x &= x - 1)
            {
                const u32 v = w * 64 + static_cast<u32>(std::countr_zero(x));
                u32 adjacent = 0;
                for (u32 p = 0; p < t; ++p)
                    adjacent += static_cast<u32>(nbs[static_cast<usize>(p) * m_w + v / 64] >> (v % 64) & 1);
                cand.emplace_back(adjacent, v);
            }
        std::ranges::sort(cand, [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        // as many independent parts as picks: every pick is a neighbour of one of them, so the deeper levels choose
        // among the union of their neighbourhoods only
        u64* saved = scratch(m_level_after, depth + 1) + static_cast<usize>(depth) * m_w;
        std::copy(m_after.begin(), m_after.end(), saved);
        if (independent == r)
            for (u32 u = 0; u < m_w; ++u)
                m_after[u] &= m_seen[u];
        bool ok = false;
        for (usize i = 0; i < cand.size() && !ok; ++i)
        {
            add(cand[i].second);
            ok = joinable(r - 1, depth + 1);
            remove_last();
        }
        std::copy(saved, saved + m_w, m_after.begin());
        return ok;
    }

    /// Whether the component of m_set[0] among m_set and m_after holds at least k vertices.
    bool component_holds_k()
    {
        std::fill(m_seen.begin(), m_seen.end(), 0);
        m_seen[m_set[0] / 64] |= u64{1} << (m_set[0] % 64);
        m_frontier = m_seen;
        u32 count = 1;
        while (count < m_k)
        {
            std::fill(m_nb.begin(), m_nb.end(), 0);
            for (u32 w = 0; w < m_w; ++w)
                for (u64 x = m_frontier[w]; x; x &= x - 1)
                {
                    const u64* r = row(w * 64 + static_cast<u32>(std::countr_zero(x)));
                    for (u32 u = 0; u < m_w; ++u)
                        m_nb[u] |= r[u];
                }
            bool any = false;
            for (u32 u = 0; u < m_w; ++u)
            {
                m_frontier[u] = m_nb[u] & (m_after[u] | m_bits[u]) & ~m_seen[u];
                m_seen[u] |= m_frontier[u];
                count += static_cast<u32>(std::popcount(m_frontier[u]));
                any |= m_frontier[u] != 0;
            }
            if (!any)
                break;
        }
        return count >= m_k;
    }

    /// Whether the prefix m_set (its last vertex `last`) may extend to a connected k-subset with vertices above `last`
    /// (false only if it does not): at most k - |prefix| such vertices join it (joinable), and its component among them
    /// holds k vertices (a connected set of at most k vertices then grows to exactly k inside it).
    bool feasible(u32 last) { return extends_from(last + 1); }

    /// Whether the prefix m_set may extend to a connected k-subset with vertices >= start, and in `within` if given
    /// (false only if it does not).
    bool extends_from(u32 start, const u64* within = nullptr)
    {
        set_after(start);
        if (within)
            for (u32 u = 0; u < m_w; ++u)
                m_after[u] &= within[u];
        m_budget = k_budget;
        return joinable(m_k - static_cast<u32>(m_set.size()), 0) && component_holds_k();
    }

    void rec(u32 start)
    {
        const u32 need = m_k - static_cast<u32>(m_set.size());  // >= 1
        if (need == 1)
        {
            // the last vertex: those adjacent to every part of the prefix (any vertex for k = 1)
            set_after(start);
            m_cand = m_after;
            const u32 t = m_set.empty() ? 0 : parts(m_level_nb[0]);
            for (u32 p = 0; p < t; ++p)
                for (u32 u = 0; u < m_w; ++u)
                    m_cand[u] &= m_level_nb[0][static_cast<usize>(p) * m_w + u];
            for (u32 w = 0; w < m_w; ++w)
                for (u64 x = m_cand[w]; x; x &= x - 1)
                {
                    add(w * 64 + static_cast<u32>(std::countr_zero(x)));
                    m_out->push_back(m_set);
                    remove_last();
                    if (++m_found >= m_limit)
                        return;
                }
            return;
        }
        // when as many parts of the prefix need a vertex of their own as there are picks left, every pick is a
        // neighbour of one of them: the children lie in the union of their neighbourhoods
        const bool cut = !m_set.empty();
        bool tight = false;
        const usize depth = m_set.size();
        if (cut)
        {
            set_after(start);
            const u32 t = parts(m_level_nb[0]);
            tight = t > 1 && independent_parts(m_level_nb[0], t) == need;
            if (tight)
                std::copy(m_seen.begin(), m_seen.end(), scratch(m_children, depth + 1) + depth * m_w);
        }
        // otherwise completions using vertices >= s exist for s up to some last value (fewer vertices, fewer
        // completions), and a child j needs one with vertices >= j: the children stop there
        u32 end = m_m - need;
        if (cut && !tight)
        {
            u32 lo = start;  // a completion with vertices >= start exists (the parent's check)
            while (lo < end)
            {
                const u32 mid = lo + (end - lo + 1) / 2;
                if (extends_from(mid))
                    lo = mid;
                else
                    end = mid - 1;
            }
        }
        // below the cutoff, the component of the prefix among the vertices >= j holds k vertices (it only grows as j
        // falls), so a child needs only the joining check; a child of a tight prefix gets the whole check, its picks
        // within the same union
        for (u32 j = start; j <= end; ++j)
        {
            if (tight && !(m_children[depth * m_w + j / 64] >> (j % 64) & 1))
                continue;
            add(j);
            bool ok = false;
            if (cut && !tight)
            {
                set_after(j + 1);
                m_budget = k_budget;
                ok = joinable(need - 1, 0);
            }
            else if (tight)
                ok = extends_from(j + 1, m_children.data() + depth * m_w);
            else
                ok = feasible(j);
            if (ok)
                rec(j + 1);
            remove_last();
            if (m_found >= m_limit)
                return;
        }
    }

    u32 m_m, m_w;
    const std::vector<u64>& m_adj;
    std::vector<u64> m_after, m_cand, m_seen, m_frontier, m_nb, m_bits;
    std::vector<std::vector<u64>> m_level_nb;  // per depth of joinable: the parts' neighbourhoods
    std::vector<std::vector<std::pair<u32, u32>>> m_level_cand;  // per depth of joinable: the branching candidates
    std::vector<u64> m_children;     // per prefix size: the children a tight prefix allows
    std::vector<u64> m_level_after;  // per depth of joinable: m_after before a tight level narrowed it
    std::vector<u32> m_set, m_stack, m_order, m_count;
    std::vector<u8> m_part;
    u32 m_k = 0, m_limit = 0, m_found = 0, m_budget = 0;
    std::vector<std::vector<u32>>* m_out = nullptr;
    /// joinable calls per check: enough for the sparse components where branching settles a check, a bound for dense
    /// ones where it cannot
    static constexpr u32 k_budget = 512;
};

/// Per component: its subsets (goal literal indices), by size, then lexicographic.
std::vector<std::vector<std::vector<u32>>> component_subsets(const GroundCondition& goal, const TemplateOptions& o)
{
    const u32 n = static_cast<u32>(goal.literals.size());
    // components by union-find over shared objects, ordered by their smallest literal
    std::vector<u32> parent(n);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](u32 x)
    {
        while (parent[x] != x)
            x = parent[x] = parent[parent[x]];
        return x;
    };
    std::vector<std::vector<u32>> literals_of;  // object -> literals mentioning it (ascending, each once)
    for (u32 i = 0; i < n; ++i)
        for (ObjectId ob : goal.literals[i].atom.objects)
        {
            if (ob.v >= literals_of.size())
                literals_of.resize(ob.v + 1);
            auto& l = literals_of[ob.v];
            if (l.empty() || l.back() != i)
                l.push_back(i);
        }
    for (const auto& l : literals_of)
        for (usize j = 1; j < l.size(); ++j)
        {
            const u32 a = find(l[0]), b = find(l[j]);
            if (a != b)
                parent[std::max(a, b)] = std::min(a, b);
        }
    std::vector<std::vector<u32>> comps;
    std::vector<u32> comp_of_root(n, ~0u);
    for (u32 i = 0; i < n; ++i)
    {
        const u32 r = find(i);
        if (comp_of_root[r] == ~0u)
        {
            comp_of_root[r] = static_cast<u32>(comps.size());
            comps.emplace_back();
        }
        comps[comp_of_root[r]].push_back(i);
    }

    std::vector<std::vector<std::vector<u32>>> out(comps.size());
    std::vector<u32> local(n, 0);
    for (usize c = 0; c < comps.size(); ++c)
    {
        const std::vector<u32>& comp = comps[c];
        const u32 m = static_cast<u32>(comp.size());
        for (u32 j = 0; j < m; ++j)
            local[comp[j]] = j;
        const u32 words = (m + 63) / 64;
        // adjacency: the OR of the literal sets of each literal's objects, without the literal itself
        std::vector<u64> adj(static_cast<usize>(m) * words, 0);
        for (u32 j = 0; j < m; ++j)
        {
            u64* r = adj.data() + static_cast<usize>(j) * words;
            for (ObjectId ob : goal.literals[comp[j]].atom.objects)
                for (u32 i : literals_of[ob.v])
                    r[local[i] / 64] |= u64{1} << (local[i] % 64);
            r[j / 64] &= ~(u64{1} << (j % 64));
        }
        SubsetSearch search(m, words, adj);
        std::vector<std::vector<u32>> found;
        for (u32 k = 1; k <= std::min(m, o.max_size); ++k)
            search.run(k, o.max_subsets_per_size, found);
        for (std::vector<u32>& s : found)
            for (u32& v : s)
                v = comp[v];
        out[c] = std::move(found);
    }
    return out;
}
}  // namespace

void TemplateOptions::validate() const
{
    auto positive = [](u32 v, const char* name)
    {
        if (v == 0)
            throw std::invalid_argument(std::string("mymyr: lifted_goal_templates: ") + name + " must be at least 1");
    };
    positive(max_subgoals_per_size, "max_subgoals_per_size");
    positive(max_size, "max_size");
    positive(max_subsets_per_size, "max_subsets_per_size");
    positive(max_component_samples, "max_component_samples");
    positive(max_products, "max_products");
    if (static_cast<u64>(max_component_samples) * max_size >= (u64{1} << 32))
        throw std::invalid_argument("mymyr: lifted_goal_templates: max_component_samples * max_size must be below 2^32");
    if (distinct != Distinct::None && distinct != Distinct::All && distinct != Distinct::Equality)
        throw std::invalid_argument("mymyr: lifted_goal_templates: unknown distinct mode");
}

std::vector<std::vector<u32>> goal_subgoals(const Task& task, u64 instance, const TemplateOptions& o)
{
    o.validate();
    const GroundCondition goal = GroundCondition::goal(task);
    const auto subs = component_subsets(goal, o);
    const u32 nc = static_cast<u32>(subs.size());

    std::vector<std::vector<std::vector<u32>>> by_size(o.max_size + 1);
    u32 open = o.max_size;  // sizes whose quota is not full yet
    std::vector<u32> perm(nc), pos;
    std::vector<std::vector<u32>> draws;
    for (u32 c = 1; c <= std::min(o.max_size, nc) && open > 0; ++c)
    {
        const u32 samples = static_cast<u32>(binom_capped(nc, c, o.max_component_samples));
        draws.clear();
        for (u32 s = 0; s < samples; ++s)
        {
            std::iota(perm.begin(), perm.end(), 0u);
            for (u32 d = 0; d < c; ++d)
            {
                const u64 draw = (static_cast<u64>(c) << 32) | (static_cast<u64>(s) * c + d);
                const u32 j = d + rng::below(rng::bits64(o.seed, instance, draw, rng::k_templates), nc - d);
                std::swap(perm[d], perm[j]);
            }
            std::vector<u32> pick(perm.begin(), perm.begin() + c);
            std::sort(pick.begin(), pick.end());
            draws.push_back(std::move(pick));
        }
        std::sort(draws.begin(), draws.end());
        draws.erase(std::unique(draws.begin(), draws.end()), draws.end());
        for (const std::vector<u32>& pick : draws)
        {
            // the first max_products elements of the product, the last component varying fastest
            pos.assign(c, 0);
            for (u32 p = 0; p < o.max_products && open > 0; ++p)
            {
                u32 size = 0;
                for (u32 i = 0; i < c; ++i)
                    size += static_cast<u32>(subs[pick[i]][pos[i]].size());
                if (size <= o.max_size && by_size[size].size() < o.max_subgoals_per_size)
                {
                    std::vector<u32> sg;
                    sg.reserve(size);
                    for (u32 i = 0; i < c; ++i)
                        sg.insert(sg.end(), subs[pick[i]][pos[i]].begin(), subs[pick[i]][pos[i]].end());
                    by_size[size].push_back(std::move(sg));
                    if (by_size[size].size() == o.max_subgoals_per_size)
                        --open;
                }
                u32 i = c;
                while (i > 0 && ++pos[i - 1] == subs[pick[i - 1]].size())
                    pos[--i] = 0;
                if (i == 0)
                    break;  // the product is exhausted
            }
        }
    }
    std::vector<std::vector<u32>> out;
    for (u32 size = o.max_size; size >= 1; --size)
        for (std::vector<u32>& sg : by_size[size])
            out.push_back(std::move(sg));
    return out;
}

// ------------------------------------------------------------------------------------------------ packing

struct GoalTemplates::Packer
{
    const TaskSuite& suite;
    std::vector<i32> instance_offsets{0}, template_size, num_vars, literal_offsets{0}, predicate, kind, arity, goal_literal;
    std::vector<std::vector<i32>> terms;
    std::vector<u8> distinct, polarity;
    std::vector<std::vector<u32>> orders;
    u32 max_vars = 0, max_arity = 0;

    explicit Packer(const TaskSuite& s) : suite(s) {}

    /// Adds a template of instance g: the literals of c (instance predicate ids), num_vars variables, distinct, and
    /// the goal literal of each literal (or none).
    void add(u32 g, Workspace& ws, const ConjunctiveCondition& c, bool dist, const std::vector<u32>* goal_lits)
    {
        const TaskSuite::Ref ref = suite.ref(g);
        const TaskTable& table = *suite.table(ref.domain);
        const TaskTable::Instance& inst = table.instance(ref.local);
        const Task& task = *inst.task;
        for (usize i = 0; i < c.literals.size(); ++i)
        {
            const auto& l = c.literals[i];
            const u32 tp = inst.pred_map.at(l.predicate.v);
            predicate.push_back(static_cast<i32>(tp));
            polarity.push_back(l.positive ? 1 : 0);
            kind.push_back(table.predicate_kinds()[tp]);
            arity.push_back(static_cast<i32>(l.terms.size()));
            terms.emplace_back(l.terms.begin(), l.terms.end());
            goal_literal.push_back(goal_lits ? static_cast<i32>((*goal_lits)[i]) : -1);
            max_arity = std::max<u32>(max_arity, static_cast<u32>(l.terms.size()));
        }
        template_size.push_back(static_cast<i32>(c.literals.size()));
        num_vars.push_back(static_cast<i32>(c.arity()));
        distinct.push_back(dist ? 1 : 0);
        literal_offsets.push_back(static_cast<i32>(predicate.size()));
        max_vars = std::max(max_vars, c.arity());
        // the order of the condition as condition(k) returns it
        ConjunctiveCondition canon;
        canon.literals = c.literals;
        for (u32 v = 0; v < c.arity(); ++v)
            canon.variables.push_back({"x" + std::to_string(v), {}});
        if (dist)
            for (u32 i = 0; i < c.arity(); ++i)
                for (u32 j = i + 1; j < c.arity(); ++j)
                    canon.equalities.push_back({static_cast<formalism::Term>(i), static_cast<formalism::Term>(j), false});
        orders.push_back(binding_order(task, ws, canon));
    }
    void end_instance() { instance_offsets.push_back(static_cast<i32>(template_size.size())); }

    GoalTemplates finish(TaskSuitePtr owner)
    {
        const i64 K = static_cast<i64>(template_size.size()), L = static_cast<i64>(predicate.size());
        std::vector<i32> order(static_cast<usize>(K) * max_vars, -1), flat(static_cast<usize>(L) * max_arity, k_no_term);
        for (i64 k = 0; k < K; ++k)
            std::copy(orders[k].begin(), orders[k].end(), order.begin() + k * max_vars);
        for (i64 l = 0; l < L; ++l)
            std::copy(terms[l].begin(), terms[l].end(), flat.begin() + l * max_arity);
        const i64 I = static_cast<i64>(instance_offsets.size()) - 1;
        ArrayBundle::Builder b;
        b.add("instance_offsets", DType::I32, {I + 1}, instance_offsets);
        b.add("template_size", DType::I32, {K}, template_size);
        b.add("num_vars", DType::I32, {K}, num_vars);
        b.add("distinct", DType::U8, {K}, distinct);
        b.add("literal_offsets", DType::I32, {K + 1}, literal_offsets);
        b.add("order", DType::I32, {K, static_cast<i64>(max_vars)}, order);
        b.add("predicate", DType::I32, {L}, predicate);
        b.add("polarity", DType::U8, {L}, polarity);
        b.add("kind", DType::I32, {L}, kind);
        b.add("arity", DType::I32, {L}, arity);
        b.add("terms", DType::I32, {L, static_cast<i64>(max_arity)}, flat);
        b.add("goal_literal", DType::I32, {L}, goal_literal);
        b.scalar("num_instances", I);
        b.scalar("num_templates", K);
        b.scalar("num_literals", L);
        b.scalar("max_vars", max_vars);
        b.scalar("max_arity", max_arity);
        namespace ed = encoding::detail;
        u32 index_arity = 1, predicates = 0, capacity = 0;
        for (u32 g = 0; g < owner->size(); ++g)
        {
            const auto& table = owner->table_of(g);
            for (u32 ar : table.predicate_arities())
                index_arity = std::max(index_arity, ar);
            predicates = std::max(predicates, table.num_domain_predicates());
            capacity = std::max(capacity, static_cast<u32>(instance_offsets[g + 1] - instance_offsets[g]));
        }
        std::vector<u64> inst(static_cast<usize>(I) * ed::k_instance_columns, 0);
        std::vector<u32> ranges(static_cast<usize>(I) * predicates * ed::k_literal_columns, 0);
        std::vector<i32> slots, derived_slots, statics;
        for (u32 g = 0; g < owner->size(); ++g)
        {
            const auto& table = owner->table_of(g);
            const auto local = owner->local_id(g);
            const auto meta = atom_metadata(*owner->task(g));
            u64* row = inst.data() + u64{g} * ed::k_instance_columns;
            row[ed::k_num_objects] = table.instance(local).num_objects;
            row[ed::k_slot_args_offset] = slots.size() / index_arity;
            row[ed::k_derived_args_offset] = derived_slots.size() / index_arity;
            row[ed::k_static_args_offset] = statics.size() / index_arity;
            auto append = [&](const char* pred_name, const char* args_name, std::vector<i32>& dst)
            {
                const auto pred = ed::literal_ints(meta, pred_name), args = ed::literal_ints(meta, args_name);
                const usize width = pred.empty() ? 0 : args.size() / pred.size();
                for (usize n = 0; n < pred.size(); ++n)
                    for (u32 j = 0; j < index_arity; ++j)
                        dst.push_back(j < width ? args[n * width + j] : -1);
            };
            append("atom_pred", "atom_args_padded", slots);
            append("derived_pred", "derived_args_padded", derived_slots);
            const auto sp = ed::literal_ints(meta, "static_pred"), sa = ed::literal_ints(meta, "static_args_padded");
            const usize sw = sp.empty() ? 0 : sa.size() / sp.size();
            auto static_rows = [&](u32 ip, u32 ar)
            {
                std::vector<std::vector<i32>> tuples;
                for (usize n = 0; n < sp.size(); ++n)
                    if (static_cast<u32>(sp[n]) == ip)
                        tuples.emplace_back(sa.begin() + n * sw, sa.begin() + n * sw + ar);
                std::sort(tuples.begin(), tuples.end());
                tuples.erase(std::unique(tuples.begin(), tuples.end()), tuples.end());
                const u32 begin = static_cast<u32>(statics.size() / index_arity - row[ed::k_static_args_offset]);
                for (const auto& tuple : tuples)
                    for (u32 j = 0; j < index_arity; ++j)
                        statics.push_back(j < ar ? tuple[j] : -1);
                return std::pair{begin, static_cast<u32>(statics.size() / index_arity - row[ed::k_static_args_offset])};
            };
            const auto r = ed::build_literal_ranges(table, local, meta, static_rows);
            std::copy(r.begin(), r.end(), ranges.begin() + u64{g} * predicates * ed::k_literal_columns);
        }
        b.add("literal_inst", DType::U64, {I, ed::k_instance_columns}, inst);
        b.add("literal_ranges", DType::U32, {I, predicates, ed::k_literal_columns}, ranges);
        b.add("literal_slots", DType::I32, {static_cast<i64>(slots.size() / index_arity), index_arity}, slots);
        b.add("literal_derived", DType::I32, {static_cast<i64>(derived_slots.size() / index_arity), index_arity}, derived_slots);
        b.add("literal_static", DType::I32, {static_cast<i64>(statics.size() / index_arity), index_arity}, statics);
        b.scalar("capacity", capacity);
        GoalTemplates t;
        t.m_suite = std::move(owner);
        t.m_arrays = std::make_shared<const ArrayBundle>(b.build());
        t.m_instance_offsets.assign(instance_offsets.begin(), instance_offsets.end());
        t.m_literal_offsets.assign(literal_offsets.begin(), literal_offsets.end());
        t.m_max_vars = max_vars;
        t.m_max_arity = max_arity;
        t.m_capacity = capacity;
        t.m_index_arity = index_arity;
        t.m_predicates = predicates;
        t.m_needs_derived = std::ranges::find(kind, static_cast<i32>(k_pred_derived)) != kind.end();
        return t;
    }
};

namespace
{
bool declares_equality(const Task& task)
{
    for (std::string_view r : task.data().requirements)
    {
        while (!r.empty() && r.front() == ':')
            r.remove_prefix(1);
        if (r == "equality")
            return true;
    }
    return false;
}

const i32* i32_array(const ArrayBundle& b, std::string_view name)
{
    return static_cast<const i32*>(b.data(*b.find(name)));
}

const u8* u8_array(const ArrayBundle& b, std::string_view name) { return static_cast<const u8*>(b.data(*b.find(name))); }

/// Instance predicate id of a table predicate of instance `inst` (~0u: none).
u32 instance_predicate(const TaskTable::Instance& inst, u32 table_pred)
{
    const auto it = std::find(inst.pred_map.begin(), inst.pred_map.end(), table_pred);
    return it == inst.pred_map.end() ? ~0u : static_cast<u32>(it - inst.pred_map.begin());
}
}  // namespace

GoalTemplates GoalTemplates::generate(TaskSuitePtr suite, const TemplateOptions& options)
{
    options.validate();
    if (!suite)
        invalid("no suite");
    Packer p(*suite);
    for (u32 g = 0; g < suite->size(); ++g)
    {
        const Task& task = *suite->task(g);
        const GroundCondition goal = GroundCondition::goal(task);
        const bool dist = options.distinct == Distinct::All || (options.distinct == Distinct::Equality && declares_equality(task));
        const WorkspaceLease ws = task.workspace();
        for (const std::vector<u32>& sg : goal_subgoals(task, g, options))
        {
            GroundCondition sub;
            for (u32 i : sg)
                sub.literals.push_back(goal.literals[i]);
            p.add(g, *ws, sub.lift(dist), dist, &sg);
        }
        p.end_instance();
    }
    GoalTemplates t = p.finish(std::move(suite));
    t.m_options = options;
    t.m_generated = true;
    return t;
}

GoalTemplates GoalTemplates::from_conditions(TaskSuitePtr suite, const std::vector<std::vector<ConjunctiveCondition>>& conditions)
{
    if (!suite)
        invalid("no suite");
    if (conditions.size() != suite->size())
        invalid("conditions for " + std::to_string(conditions.size()) + " instances, the " + suite->noun() + " has " +
                std::to_string(suite->size()));
    Packer p(*suite);
    for (u32 g = 0; g < suite->size(); ++g)
    {
        const Task& task = *suite->task(g);
        const WorkspaceLease ws = task.workspace();
        for (usize k = 0; k < conditions[g].size(); ++k)
        {
            const ConjunctiveCondition& c = conditions[g][k];
            const std::string where = "instance " + std::to_string(g) + ", condition " + std::to_string(k) + ": ";
            c.validate(task);
            if (c.literals.empty())
                invalid(where + "no literals");
            if (!c.constraints.empty())
                invalid(where + "numeric constraints are not supported");
            for (const auto& v : c.variables)
                if (!v.types.empty())
                    invalid(where + "typed variables are not supported");
            const u32 n = c.arity();
            if (!c.equalities.empty())
            {
                std::vector<u8> pair(static_cast<usize>(n) * n, 0);
                bool ok = c.equalities.size() == static_cast<usize>(n) * (n - 1) / 2;
                for (const auto& e : c.equalities)
                {
                    if (!ok || e.positive || e.lhs < 0 || e.rhs < 0 || e.lhs == e.rhs)
                    {
                        ok = false;
                        break;
                    }
                    const u32 a = static_cast<u32>(std::min(e.lhs, e.rhs)), b = static_cast<u32>(std::max(e.lhs, e.rhs));
                    ok = pair[a * n + b] == 0;
                    pair[a * n + b] = 1;
                }
                if (!ok)
                    invalid(where + "equalities must be one disequality per pair of distinct variables, or none");
            }
            p.add(g, *ws, c, !c.equalities.empty(), nullptr);
        }
        p.end_instance();
    }
    return p.finish(std::move(suite));
}

GoalTemplates GoalTemplates::from_arrays(TaskSuitePtr suite, const Arrays& a, const TemplateOptions* options)
{
    if (options)
        options->validate();
    if (!suite)
        invalid("no suite");
    const usize I = suite->size();
    if (a.instance_offsets.size() != I + 1 || a.instance_offsets[0] != 0)
        invalid("instance_offsets must have num_instances + 1 entries, starting at 0");
    for (usize g = 0; g < I; ++g)
        if (a.instance_offsets[g + 1] < a.instance_offsets[g])
            invalid("instance_offsets must be nondecreasing");
    const usize K = static_cast<usize>(a.instance_offsets[I]);
    if (a.template_size.size() != K || a.num_vars.size() != K || a.distinct.size() != K || a.literal_offsets.size() != K + 1 ||
        a.literal_offsets[0] != 0)
        invalid("the per-template arrays do not match instance_offsets");
    for (usize k = 0; k < K; ++k)
        if (a.literal_offsets[k + 1] - a.literal_offsets[k] != a.template_size[k] || a.template_size[k] < 1 || a.num_vars[k] < 0)
            invalid("template " + std::to_string(k) + ": literal_offsets, template_size and num_vars disagree");
    const usize L = static_cast<usize>(a.literal_offsets[K]);
    if (a.predicate.size() != L || a.polarity.size() != L || a.kind.size() != L || a.arity.size() != L ||
        a.goal_literal.size() != L || a.terms.size() != L * a.max_arity)
        invalid("the per-literal arrays do not match literal_offsets");

    Packer p(*suite);
    for (u32 g = 0; g < I; ++g)
    {
        const TaskSuite::Ref ref = suite->ref(g);
        const TaskTable& table = *suite->table(ref.domain);
        const TaskTable::Instance& inst = table.instance(ref.local);
        const Task& task = *inst.task;
        const GroundCondition goal = GroundCondition::goal(task);
        const u32 goal_size = static_cast<u32>(goal.literals.size());
        const WorkspaceLease ws = task.workspace();
        for (usize k = static_cast<usize>(a.instance_offsets[g]); k < static_cast<usize>(a.instance_offsets[g + 1]); ++k)
        {
            const std::string where = "template " + std::to_string(k) + ": ";
            ConjunctiveCondition c;
            const u32 n = static_cast<u32>(a.num_vars[k]);
            for (u32 v = 0; v < n; ++v)
                c.variables.push_back({"x" + std::to_string(v), {}});
            std::vector<u32> goal_lits;
            bool from_goal = true;
            for (usize l = static_cast<usize>(a.literal_offsets[k]); l < static_cast<usize>(a.literal_offsets[k + 1]); ++l)
            {
                const i32 tp = a.predicate[l];
                if (tp < 0 || static_cast<usize>(tp) >= table.predicate_names().size())
                    invalid(where + "predicate out of range");
                const u32 ip = instance_predicate(inst, static_cast<u32>(tp));
                if (ip == ~0u)
                    invalid(where + "predicate " + std::to_string(tp) + " is not one of the instance's");
                const i32 ar = a.arity[l];
                if (ar < 0 || static_cast<u32>(ar) > a.max_arity || static_cast<u32>(ar) != table.predicate_arities()[tp] ||
                    a.kind[l] != table.predicate_kinds()[tp] || a.polarity[l] > 1)
                    invalid(where + "arity, kind or polarity disagree with the predicate");
                ConjunctiveCondition::Literal x{PredicateId{ip}, a.polarity[l] == 1, {}};
                for (u32 i = 0; i < a.max_arity; ++i)
                {
                    const i32 t = a.terms[l * a.max_arity + i];
                    if (i >= static_cast<u32>(ar))
                    {
                        if (t != k_no_term)
                            invalid(where + "terms past the arity must be padding");
                        continue;
                    }
                    if (t == k_no_term || (t >= 0 && static_cast<u32>(t) >= n) ||
                        (t < 0 && static_cast<u32>(-(t + 1)) >= task.num_objects()))
                        invalid(where + "term out of range");
                    x.terms.push_back(t);
                }
                c.literals.push_back(std::move(x));
                const i32 gl = a.goal_literal[l];
                if (gl < -1 || gl >= static_cast<i32>(goal_size))
                    invalid(where + "goal_literal out of range");
                if (gl >= 0)
                {
                    const GroundLiteral& want = goal.literals[static_cast<u32>(gl)];
                    if (want.atom.predicate.v != ip || want.positive != (a.polarity[l] == 1))
                        invalid(where + "goal_literal names a goal literal of another predicate or polarity");
                }
                if (l == static_cast<usize>(a.literal_offsets[k]))
                    from_goal = gl >= 0;
                else if ((gl >= 0) != from_goal)
                    invalid(where + "goal_literal must be set for every literal of a template or for none");
                if (gl >= 0)
                    goal_lits.push_back(static_cast<u32>(gl));
            }
            if (a.distinct[k] > 1)
                invalid(where + "distinct must be 0 or 1");
            if (a.distinct[k])
                for (u32 i = 0; i < n; ++i)
                    for (u32 j = i + 1; j < n; ++j)
                        c.equalities.push_back({static_cast<formalism::Term>(i), static_cast<formalism::Term>(j), false});
            c.validate(task);
            p.add(g, *ws, c, a.distinct[k] == 1, from_goal ? &goal_lits : nullptr);
        }
        p.end_instance();
    }
    GoalTemplates t = p.finish(std::move(suite));
    if (options)
    {
        t.m_options = *options;
        t.m_generated = true;
    }
    return t;
}

ConjunctiveCondition GoalTemplates::condition(u32 k) const
{
    if (k >= num_templates())
        throw std::out_of_range("mymyr: GoalTemplates: template index out of range");
    const u32 g = static_cast<u32>(std::upper_bound(m_instance_offsets.begin(), m_instance_offsets.end(), k) - m_instance_offsets.begin()) - 1;
    const TaskSuite::Ref ref = m_suite->ref(g);
    const TaskTable::Instance& inst = m_suite->table(ref.domain)->instance(ref.local);
    const ArrayBundle& b = *m_arrays;
    const i32* pred = i32_array(b, "predicate");
    const i32* ar = i32_array(b, "arity");
    const i32* terms = i32_array(b, "terms");
    const u8* pol = u8_array(b, "polarity");
    const u32 n = static_cast<u32>(i32_array(b, "num_vars")[k]);
    ConjunctiveCondition c;
    for (u32 v = 0; v < n; ++v)
        c.variables.push_back({"x" + std::to_string(v), {}});
    for (u32 l = m_literal_offsets[k]; l < m_literal_offsets[k + 1]; ++l)
    {
        ConjunctiveCondition::Literal x{PredicateId{instance_predicate(inst, static_cast<u32>(pred[l]))}, pol[l] == 1, {}};
        x.terms.assign(terms + static_cast<usize>(l) * m_max_arity, terms + static_cast<usize>(l) * m_max_arity + ar[l]);
        c.literals.push_back(std::move(x));
    }
    if (u8_array(b, "distinct")[k])
        for (u32 i = 0; i < n; ++i)
            for (u32 j = i + 1; j < n; ++j)
                c.equalities.push_back({static_cast<formalism::Term>(i), static_cast<formalism::Term>(j), false});
    return c;
}

std::vector<ConjunctiveCondition> GoalTemplates::conditions(u32 instance) const
{
    if (instance >= num_instances())
        throw std::out_of_range("mymyr: GoalTemplates: instance index out of range");
    std::vector<ConjunctiveCondition> out;
    for (u32 k = m_instance_offsets[instance]; k < m_instance_offsets[instance + 1]; ++k)
        out.push_back(condition(k));
    return out;
}

std::vector<u32> GoalTemplates::goal_literals(u32 k) const
{
    if (k >= num_templates())
        throw std::out_of_range("mymyr: GoalTemplates: template index out of range");
    const i32* gl = i32_array(*m_arrays, "goal_literal");
    std::vector<u32> out;
    for (u32 l = m_literal_offsets[k]; l < m_literal_offsets[k + 1]; ++l)
        if (gl[l] >= 0)
            out.push_back(static_cast<u32>(gl[l]));
    return out;
}
}  // namespace mymyr::rl

namespace mymyr::rl
{
MatchTemplates GoalTemplates::view(const std::byte* block) const noexcept
{
    if (!block)
        block = m_arrays->block();
    auto at = [&](const char* name) { return block + m_arrays->find(name)->offset; };
    MatchTemplates v;
    v.num_instances = num_instances();
    v.capacity = m_capacity;
    v.max_vars = m_max_vars;
    v.max_arity = m_max_arity;
    v.instance_offsets = reinterpret_cast<const i32*>(at("instance_offsets"));
    v.num_vars = reinterpret_cast<const i32*>(at("num_vars"));
    v.literal_offsets = reinterpret_cast<const i32*>(at("literal_offsets"));
    v.order = reinterpret_cast<const i32*>(at("order"));
    v.predicate = reinterpret_cast<const i32*>(at("predicate"));
    v.arity = reinterpret_cast<const i32*>(at("arity"));
    v.terms = reinterpret_cast<const i32*>(at("terms"));
    v.polarity = reinterpret_cast<const u8*>(at("polarity"));
    v.distinct = reinterpret_cast<const u8*>(at("distinct"));
    v.literals.inst = reinterpret_cast<const u64*>(at("literal_inst"));
    v.literals.num_instances = num_instances();
    v.literals.num_predicates = m_predicates;
    v.literals.max_arity = m_index_arity;
    v.literals.literal_ranges = reinterpret_cast<const u32*>(at("literal_ranges"));
    v.literals.slot_args = reinterpret_cast<const i32*>(at("literal_slots"));
    v.literals.derived_args = reinterpret_cast<const i32*>(at("literal_derived"));
    v.literals.static_args = reinterpret_cast<const i32*>(at("literal_static"));
    return v;
}
}
