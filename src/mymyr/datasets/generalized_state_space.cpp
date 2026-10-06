// Generalized state spaces (datasets/generalized_state_space.hpp).

#include "mymyr/datasets/generalized_state_space.hpp"

#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/object_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace mymyr::datasets
{
std::vector<StateSpacePtr> ordered_spaces(std::span<const StateSpaceResult> results, bool sort_ascending)
{
    std::vector<StateSpacePtr> spaces;
    for (const StateSpaceResult& r : results)
        if (r.space)
            spaces.push_back(r.space);
    if (sort_ascending)
        std::stable_sort(spaces.begin(), spaces.end(),
                         [](const StateSpacePtr& a, const StateSpacePtr& b) { return a->num_states() < b->num_states(); });
    return spaces;
}

namespace
{
std::vector<u32> flagged(std::span<const u8> flags)
{
    std::vector<u32> out;
    for (usize i = 0; i < flags.size(); ++i)
        if (flags[i])
            out.push_back(static_cast<u32>(i));
    return out;
}

struct PairHash
{
    usize operator()(u64 x) const noexcept { return static_cast<usize>((x ^ (x >> 29)) * 0xbf58476d1ce4e5b9ULL); }
};
}  // namespace

std::vector<u32> GeneralizedStateSpace::initial_vertices() const { return flagged(m_initial); }
std::vector<u32> GeneralizedStateSpace::goal_vertices() const { return flagged(m_goal); }
std::vector<u32> GeneralizedStateSpace::unsolvable_vertices() const { return flagged(m_unsolvable); }

std::shared_ptr<const GeneralizedStateSpace> GeneralizedStateSpace::create(std::vector<StateSpacePtr> spaces)
{
    for (const StateSpacePtr& s : spaces)
        if (!s)
            throw std::invalid_argument("mymyr: a generalized state space needs non-null state spaces");
    for (usize i = 1; i < spaces.size(); ++i)
        if (spaces[i]->task()->data().domain_name != spaces[0]->task()->data().domain_name)
            throw std::invalid_argument("mymyr: the state spaces of a generalized state space must share one domain ('" +
                                        spaces[0]->task()->data().domain_name + "' and '" +
                                        spaces[i]->task()->data().domain_name + "')");
    auto G = std::make_shared<GeneralizedStateSpace>();
    const bool symmetric =
        !spaces.empty() && std::all_of(spaces.begin(), spaces.end(), [](const StateSpacePtr& s) { return s->symmetry_reduced(); });
    G->m_symmetric = symmetric;
    auto add_vertex = [&](u32 p, u32 v, bool initial, bool goal, bool unsolvable)
    {
        G->m_vproblem.push_back(p);
        G->m_vvertex.push_back(v);
        G->m_initial.push_back(initial);
        G->m_goal.push_back(goal);
        G->m_unsolvable.push_back(unsolvable);
    };
    auto add_edge = [&](u32 s, u32 t, u32 p, u64 e)
    {
        if (G->m_esource.size() >= 0xffffffffULL || e >= 0xffffffffULL)
            throw std::length_error("mymyr: a generalized state space supports fewer than 2^32 class edges");
        G->m_esource.push_back(s);
        G->m_etarget.push_back(t);
        G->m_eproblem.push_back(p);
        G->m_eedge.push_back(static_cast<u32>(e));
        return static_cast<u32>(G->m_esource.size() - 1);
    };
    if (!symmetric)
    {
        for (u32 p = 0; p < spaces.size(); ++p)
        {
            const StateSpace& S = *spaces[p];
            const u64 voff = G->m_vproblem.size();
            if (voff + S.num_states() > 0xffffffffULL)
                throw std::length_error("mymyr: a generalized state space supports fewer than 2^32 class vertices");
            std::vector<u32>& vmap = G->m_vmap.emplace_back(S.num_states());
            std::vector<u32>& emap = G->m_emap.emplace_back(S.num_transitions());
            for (u32 v = 0; v < S.num_states(); ++v)
            {
                add_vertex(p, v, v == S.initial_state(), S.is_goal(v), S.is_unsolvable(v));
                vmap[v] = static_cast<u32>(voff + v);
            }
            const auto off = S.forward_offsets();
            const auto tgt = S.forward_targets();
            for (u32 v = 0; v < S.num_states(); ++v)
                for (u64 e = off[v]; e < off[v + 1]; ++e)
                    emap[e] = add_edge(static_cast<u32>(voff + v), static_cast<u32>(voff + tgt[e]), p, e);
        }
        G->m_spaces = std::move(spaces);
    }
    else
    {
        std::unordered_map<Certificate, u32, CertificateHash> class_of;
        std::unordered_map<u64, u32, PairHash> class_edge;  // (source class << 32 | target class) -> class edge
        std::vector<StateSpacePtr> kept;
        std::vector<Certificate> certs;
        for (const StateSpacePtr& sp : spaces)
        {
            const StateSpace& S = *sp;
            if (S.certificate() != spaces[0]->certificate() || S.fwl_k() != spaces[0]->fwl_k())
                throw std::invalid_argument("mymyr: the state spaces of a generalized state space must use one certificate kind");
            ObjectGraphBuilder ogb(*S.task());
            ObjectGraph graph;
            certs.resize(S.num_states());
            for (u32 v = 0; v < S.num_states(); ++v)
            {
                ogb.build(S.state(v), graph);
                certs[v] = S.certificate() == CertificateKind::KFwl ? kfwl_certificate(graph, S.fwl_k()) : color_refinement_certificate(graph);
            }
            if (class_of.contains(certs[S.initial_state()]))
                continue;  // isomorphic to an earlier problem
            const u32 p = static_cast<u32>(kept.size());
            kept.push_back(sp);
            std::vector<u32>& vmap = G->m_vmap.emplace_back(S.num_states());
            std::vector<u32>& emap = G->m_emap.emplace_back(S.num_transitions());
            for (u32 v = 0; v < S.num_states(); ++v)
            {
                auto [it, fresh] = class_of.try_emplace(certs[v], static_cast<u32>(G->m_vproblem.size()));
                if (fresh)
                {
                    if (G->m_vproblem.size() >= 0xffffffffULL)
                        throw std::length_error("mymyr: a generalized state space supports fewer than 2^32 class vertices");
                    add_vertex(p, v, v == S.initial_state(), S.is_goal(v), S.is_unsolvable(v));
                }
                vmap[v] = it->second;
            }
            const auto off = S.forward_offsets();
            const auto tgt = S.forward_targets();
            for (u32 v = 0; v < S.num_states(); ++v)
                for (u64 e = off[v]; e < off[v + 1]; ++e)
                {
                    const u32 cs = vmap[v], ct = vmap[tgt[e]];
                    const u64 key = (static_cast<u64>(cs) << 32) | ct;
                    if (auto it = class_edge.find(key); it != class_edge.end())
                        emap[e] = it->second;
                    else
                    {
                        const u32 ce = add_edge(cs, ct, p, e);
                        class_edge.emplace(key, ce);
                        emap[e] = ce;
                    }
                }
        }
        G->m_spaces = std::move(kept);
    }
    // forward CSR over the class vertices (edges in ascending index order per source)
    const u32 V = G->num_vertices();
    G->m_foffsets.assign(static_cast<usize>(V) + 1, 0);
    for (u32 s : G->m_esource)
        ++G->m_foffsets[static_cast<usize>(s) + 1];
    for (u32 v = 0; v < V; ++v)
        G->m_foffsets[v + 1] += G->m_foffsets[v];
    G->m_fedges.resize(G->m_esource.size());
    std::vector<u64> fill(G->m_foffsets.begin(), G->m_foffsets.end() - 1);
    for (u32 e = 0; e < G->m_esource.size(); ++e)
        G->m_fedges[fill[G->m_esource[e]]++] = e;
    return G;
}
}  // namespace mymyr::datasets
