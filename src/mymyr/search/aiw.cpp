// Abstracted IW and abstracted LIW (search/aiw.hpp): the IW family engine with the abstracted novelty pruner, a port
// of mimir's AbstractedNoveltyPruningStrategyImpl (src/search/algorithms/iw/pruning_strategy.cpp).

#include "mymyr/search/aiw.hpp"

#include "iw_family_detail.hpp"

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace mymyr::search
{
namespace detail
{
namespace
{
struct KeyHash
{
    usize operator()(const std::vector<u32>& k) const noexcept
    {
        u64 h = k.size();
        for (u32 x : k)
            h = hash::combine(h, x);
        return static_cast<usize>(h);
    }
};

struct AbstractedConfig
{
    u32 width = 1;
    u64 max_dense_bytes = u64{256} << 20;  // width 2: dense pair tables (all ranks together) up to this size
    bool base_abstracted = false;
    bool preserve_goal_atoms = true;
    bool preserve_landmark_atoms = true;
};

class AbstractedPruner
{
public:
    /// coords: the landmark coordinates of abstracted LIW, or null.
    AbstractedPruner(const Task& task, const AbstractedConfig& cfg, const novelty::LandmarkCoordinates* coords)
        : m_task(task), m_cfg(cfg), m_coords(coords), m_single(coords ? coords->num_ranks() : 1)
    {
        if (cfg.preserve_goal_atoms)
        {
            const formalism::TaskData& t = task.data();
            const CanonicalLayout& L = task.atoms().layout();
            std::vector<u32> args;
            for (const formalism::Literal& l : t.literals_of(t.goal))
            {
                if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
                    continue;
                args.clear();
                for (formalism::Term x : t.terms_of(l))
                    args.push_back(formalism::term_object(x).v);
                const CanonicalAtom c = L.encode(l.pred.v, args.data());
                if (c < L.fluent_count)
                    m_preserved.push_back(task.atoms().intern(c));
            }
        }
        if (coords && cfg.preserve_landmark_atoms)
            m_preserved.insert(m_preserved.end(), coords->atoms().begin(), coords->atoms().end());
        std::sort(m_preserved.begin(), m_preserved.end());
        m_preserved.erase(std::unique(m_preserved.begin(), m_preserved.end()), m_preserved.end());
        if (coords)
        {
            m_pmask.resize(coords->mask_words());
            m_smask.resize(coords->mask_words());
            m_fmask.resize(coords->mask_words());
            m_kmask.resize(coords->mask_words());
            m_mirror = cfg.width == 1;
            m_rw = coords->mask_words();
        }
        m_dense2 = cfg.width == 2 && cfg.max_dense_bytes > 0;
        if (m_dense2)
            m_prow.resize(m_single.size());
    }

    bool init(const u64* w, u32 n)
    {
        if (m_coords)
        {
            const bool any = m_coords->mask(w, n, m_pmask.data());
            m_coords->collect(m_pmask.data(), any, m_ranks);
            for (u32 r : m_ranks)
                state_update<true>(w, n, r);
            return true;
        }
        state_update<true>(w, n, 0);
        return true;  // mimir's test_prune_initial_state returns false (never prunes the start state)
    }
    void begin(const u64* w, u32 n)
    {
        if (m_coords)
            m_pany = m_coords->mask(w, n, m_pmask.data());
        if (incremental() || m_mirror)
            begin_incremental(w, n);
    }
    bool test(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        return run<true>(pw, pn, cw, cn, add, d);
    }
    /// test() without marking (BeamNovelty::SurvivorsOnly).
    bool peek(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        return run<false>(pw, pn, cw, cn, add, d);
    }
    [[nodiscard]] u64 bytes() const
    {
        u64 b = m_pairs.bytes() + m_triples.bytes() + m_by_feature.capacity() * 8 + m_dense_bytes;
        for (const auto& s : m_single)
            b += s.capacity() * 8;
        for (const auto& f : m_features)
            b += f.capacity() * 4;
        return b;
    }

private:
    /// The novelty of a transition; with Mark, marks its tuples.
    template<bool Mark>
    bool run(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        if (!m_coords && incremental())
            return test_incremental<Mark>(pw, pn, cw, cn, add, d);
        if (!m_coords)
            return transition_update<Mark>(pw, pn, cw, cn, add, 0);
        const bool sany = m_coords->child_mask(m_pmask.data(), cw, cn, add, d.del, m_smask.data());
        if (m_cfg.width == 1)
        {
            // pairs: flipped ranks x the successor's features, kept ranks x the added atoms' features. Tested first
            // (kept ranks against the feature-major mirror, flipped ones against the rank's feature set), marked only
            // if one is new (marking pairs that are all marked changes nothing)
            m_coords->split_masks(m_pmask.data(), m_pany, m_smask.data(), sany, m_fmask.data(), m_kmask.data());
            const bool flipped = bits::any(m_fmask.data(), m_rw);
            if (flipped)
                apply_delta_features(pw, pn, cw, cn, add, d);  // m_pu: the successor's features
            const bool novel = novel1(add, flipped);
            if (Mark && novel)
            {
                if (flipped)
                    bits::for_each(m_fmask.data(), m_rw,
                                   [&](u64 r)
                                   {
                                       std::vector<u64>& set = m_single[r];
                                       if (set.size() < m_pu.size())
                                           set.resize(m_pu.size(), 0);
                                       for (usize i = 0; i < m_pu.size(); ++i)
                                           set[i] |= m_pu[i];
                                       bits::for_each(m_pu.data(), static_cast<u32>(m_pu.size()),
                                                      [&](u64 x) { bits::set(mirror_row(static_cast<u32>(x)), r); });
                                   });
                bits::for_each(m_kmask.data(), m_rw,
                               [&](u64 r)
                               {
                                   for (u32 a : add)
                                       for (u32 x : features(a))
                                           insert1(static_cast<u32>(r), x);
                               });
            }
            if (flipped)
                revert_delta_features(add);
            return novel;
        }
        m_coords->split(m_pmask.data(), m_pany, m_smask.data(), sany, m_flipped, m_kept);
        bool novel = false;
        for (u32 r : m_flipped)
            novel = state_update<Mark>(cw, cn, r) || novel;
        for (u32 r : m_kept)
            novel = transition_update<Mark>(pw, pn, cw, cn, add, r) || novel;
        return novel;
    }

    // ------------------------------------------------------------------ features
    u32 intern(const std::vector<u32>& key)
    {
        const auto [it, inserted] = m_ids.emplace(key, static_cast<u32>(m_ids.size()));
        return it->second;
    }
    void append_signature(u32 object, std::vector<u32>& out) const
    {
        const formalism::TaskData& t = m_task.data();
        const formalism::Range r = t.objects[object].types;
        if (r.count == 0)
        {
            out.push_back(1);
            out.push_back(~u32{0});
            return;
        }
        out.push_back(r.count);
        for (u32 i = 0; i < r.count; ++i)
            out.push_back(t.type_ids[r.begin + i].v);
    }
    std::vector<u32> compute(u32 slot)
    {
        const AtomIndex& ix = m_task.atoms();
        const u32 pred = ix.predicate(SlotId{slot}).v;
        const std::span<const u32> args = ix.arguments(SlotId{slot});
        std::vector<u32> out;
        auto full = [&]()
        {
            m_key.assign({1u, pred});
            m_key.insert(m_key.end(), args.begin(), args.end());
            return intern(m_key);
        };
        if (std::binary_search(m_preserved.begin(), m_preserved.end(), slot))
        {
            out.push_back(full());
            if (args.size() <= 1)
                return out;
        }
        if (args.empty())
            return {full()};
        for (u32 i = 0; i < args.size(); ++i)
        {
            m_key.assign({0u, pred, i, args[i]});
            if (!m_cfg.base_abstracted)
                for (u32 j = 0; j < args.size(); ++j)
                    if (j != i)
                        append_signature(args[j], m_key);
            out.push_back(intern(m_key));
        }
        return out;
    }
    std::span<const u32> features(u32 slot)
    {
        if (slot >= m_features.size())
            m_features.resize(slot + 1);
        if (m_features[slot].empty())
            m_features[slot] = compute(slot);
        return m_features[slot];
    }

    // ------------------------------------------------------------------ tables (one per rank)
    bool insert1(u32 rank, u32 f)
    {
        std::vector<u64>& s = m_single[rank];
        if ((f >> 6) >= s.size())
            s.resize(std::max<usize>((f >> 6) + 1, s.size() * 2), 0);
        const u64 bit = u64{1} << (f & 63);
        const bool fresh = (s[f >> 6] & bit) == 0;
        s[f >> 6] |= bit;
        if (m_mirror)
            bits::set(mirror_row(f), rank);
        return fresh;
    }
    /// The feature-major mirror of the singleton sets (width 1 with landmarks): per feature, its marked ranks.
    u64* mirror_row(u32 f)
    {
        if (static_cast<usize>(f + 1) * m_rw > m_by_feature.size())
            m_by_feature.resize(std::max<usize>(static_cast<usize>(f + 1) * m_rw, m_by_feature.size() * 2), 0);
        return m_by_feature.data() + static_cast<usize>(f) * m_rw;
    }
    /// Width 1 with landmarks: whether a pair of the transition (m_fmask, m_kmask) is unmarked; with flipped ranks
    /// m_pu holds the successor's features.
    bool novel1(std::span<const u32> add, bool flipped)
    {
        if (bits::any(m_kmask.data(), m_rw))
            for (u32 a : add)
                for (u32 x : features(a))
                {
                    const u64* row = mirror_row(x);
                    for (u32 i = 0; i < m_rw; ++i)
                        if (m_kmask[i] & ~row[i])
                            return true;
                }
        if (!flipped)
            return false;
        for (u32 i = 0; i < m_rw; ++i)
            for (u64 y = m_fmask[i]; y; y &= y - 1)
            {
                const std::vector<u64>& set = m_single[i * 64 + static_cast<u32>(bits::ctz64(y))];
                for (usize j = 0; j < m_pu.size(); ++j)
                    if (m_pu[j] & ~(j < set.size() ? set[j] : u64{0}))
                        return true;
            }
        return false;
    }
    bool insert2(u32 rank, u32 a, u32 b) { return m_pairs.insert({(u64{a} << 32) | b, rank}); }
    bool insert3(u32 rank, u32 a, u32 b, u32 c) { return m_triples.insert({(u64{a} << 32) | b, (u64{c} << 32) | rank}); }
    /// With Mark: insert; without: whether the tuple is unmarked (nothing changes).
    template<bool Mark>
    bool touch1(u32 rank, u32 f)
    {
        if constexpr (Mark)
            return insert1(rank, f);
        const std::vector<u64>& s = m_single[rank];
        return (f >> 6) >= s.size() || ((s[f >> 6] >> (f & 63)) & 1) == 0;
    }
    template<bool Mark>
    bool touch2(u32 rank, u32 a, u32 b)
    {
        if constexpr (Mark)
            return insert2(rank, a, b);
        return !m_pairs.contains({(u64{a} << 32) | b, rank});
    }
    template<bool Mark>
    bool touch3(u32 rank, u32 a, u32 b, u32 c)
    {
        if constexpr (Mark)
            return insert3(rank, a, b, c);
        return !m_triples.contains({(u64{a} << 32) | b, (u64{c} << 32) | rank});
    }

    /// Loads the successor's atoms as groups; `parent` null: every atom counts as added.
    void load_groups(const u64* w, u32 n, const u64* pw, u32 pn)
    {
        m_gatoms.clear();
        m_gadded.clear();
        bits::for_each(w, n,
                       [&](u64 a)
                       {
                           m_gatoms.push_back(static_cast<u32>(a));
                           m_gadded.push_back(pw == nullptr || !bits::test(pw, pn, a));
                       });
        if (!m_gatoms.empty())
            features(m_gatoms.back());  // sizes m_features: the spans below stay valid
        m_gfeat.clear();
        for (u32 a : m_gatoms)
            m_gfeat.push_back(features(a));
    }
    /// Mimir's generate_tuples + insert over the loaded groups, for one rank.
    template<bool Mark>
    bool group_tuples(u32 rank)
    {
        if (m_dense2)
            return dense_pairs<Mark>(rank);
        const u32 G = static_cast<u32>(m_gatoms.size());
        bool novel = false;
        for (u32 g = 0; g < G; ++g)
            if (m_gadded[g])
                for (u32 f : m_gfeat[g])
                    novel = touch1<Mark>(rank, f) || novel;
        if (m_cfg.width < 2)
            return novel;
        auto other = [&](u32 ia, u32 o) { return o != ia && !(o < ia && m_gadded[o]); };
        for (u32 ia = 0; ia < G; ++ia)
        {
            if (!m_gadded[ia])
                continue;
            for (u32 o = 0; o < G; ++o)
            {
                if (!other(ia, o))
                    continue;
                for (u32 fa : m_gfeat[ia])
                    for (u32 fo : m_gfeat[o])
                        if (fa != fo)
                            novel = touch2<Mark>(rank, std::min(fa, fo), std::max(fa, fo)) || novel;
            }
        }
        if (m_cfg.width < 3)
            return novel;
        for (u32 ia = 0; ia < G; ++ia)
        {
            if (!m_gadded[ia])
                continue;
            for (u32 o2 = 0; o2 < G; ++o2)
            {
                if (!other(ia, o2))
                    continue;
                for (u32 o3 = o2 + 1; o3 < G; ++o3)
                {
                    if (!other(ia, o3))
                        continue;
                    for (u32 f1 : m_gfeat[ia])
                        for (u32 f2 : m_gfeat[o2])
                            for (u32 f3 : m_gfeat[o3])
                            {
                                u32 t[3] = {f1, f2, f3};
                                if (t[0] > t[1])
                                    std::swap(t[0], t[1]);
                                if (t[1] > t[2])
                                    std::swap(t[1], t[2]);
                                if (t[0] > t[1])
                                    std::swap(t[0], t[1]);
                                if (t[0] == t[1] || t[1] == t[2])
                                    continue;
                                novel = touch3<Mark>(rank, t[0], t[1], t[2]) || novel;
                            }
                }
            }
        }
        return novel;
    }
    /// Width 2 on dense tables: the same singles and pairs as group_tuples, as bitsets. Per rank and feature fa, a row
    /// holds the features fo with {fa, fo} marked (symmetric). The pairs of an added group ia are fa x (U_ia - {fa}),
    /// fa a feature of ia and U_ia the features of the other groups (the loaded features minus those only ia has), so
    /// the test is word-parallel; all tuples are marked only if one is new (marking tuples that are all marked changes
    /// nothing). Above max_dense_bytes the tables move to the pair set for good.
    template<bool Mark>
    bool dense_pairs(u32 rank)
    {
        const u32 G = static_cast<u32>(m_gatoms.size());
        const u32 F = static_cast<u32>(m_ids.size());  // every loaded feature is interned
        const u32 W = bits::words_for(F);
        m_u.assign(W, 0);
        if (m_cnt.size() < F)
            m_cnt.resize(F, 0);
        u32 lo = W, hi = 0;
        for (u32 g = 0; g < G; ++g)
            for (u32 f : m_gfeat[g])
                if (m_cnt[f]++ == 0)
                {
                    bits::set(m_u.data(), f);
                    lo = std::min(lo, f >> 6);
                    hi = std::max(hi, (f >> 6) + 1);
                }
        const bool novel = dense_core<Mark>(rank, m_u.data(), W, lo, hi);
        for (u32 g = 0; g < G; ++g)
            for (u32 f : m_gfeat[g])
                m_cnt[f] = 0;
        if (m_dense_bytes > m_cfg.max_dense_bytes)
            leave_dense();
        return novel;
    }
    /// The test and marking of dense_pairs: U (W words, bits only in [lo, hi)) and m_cnt hold the successor's
    /// features and, per feature, how many of its atoms have it; the added groups are those of m_gfeat with m_gadded.
    template<bool Mark>
    bool dense_core(u32 rank, u64* U, u32 W, u32 lo, u32 hi)
    {
        const u32 G = static_cast<u32>(m_gfeat.size());
        std::vector<u64>& single = m_single[rank];
        auto marked1 = [&](u32 f) { return (f >> 6) < single.size() && ((single[f >> 6] >> (f & 63)) & 1) != 0; };
        std::vector<std::vector<u64>>& rows = m_prow[rank];
        // the features only group ia has leave U while its pairs are formed
        auto unique_off = [&](u32 ia, bool off)
        {
            for (u32 f : m_gfeat[ia])
                if (m_cnt[f] == 1)
                {
                    if (off)
                        bits::reset(U, f);
                    else
                        bits::set(U, f);
                }
        };
        bool novel = false;
        for (u32 g = 0; g < G && !novel; ++g)
            if (m_gadded[g])
                for (u32 f : m_gfeat[g])
                    if (!marked1(f))
                    {
                        novel = true;
                        break;
                    }
        for (u32 ia = 0; ia < G && !novel; ++ia)
        {
            if (!m_gadded[ia])
                continue;
            unique_off(ia, true);
            for (u32 fa : m_gfeat[ia])
            {
                const bool in = bits::test(U, W, fa);
                bits::reset(U, fa);
                const std::vector<u64>* row = fa < rows.size() ? &rows[fa] : nullptr;
                for (u32 i = lo; i < hi; ++i)
                    if (U[i] & ~(row && i < row->size() ? (*row)[i] : u64{0}))
                    {
                        novel = true;
                        break;
                    }
                if (in)
                    bits::set(U, fa);
                if (novel)
                    break;
            }
            unique_off(ia, false);
        }
        if (Mark && novel)
        {
            for (u32 g = 0; g < G; ++g)
                if (m_gadded[g])
                    for (u32 f : m_gfeat[g])
                        insert1(rank, f);
            const u32 F = static_cast<u32>(m_ids.size());
            if (rows.size() < F)
                rows.resize(F);
            for (u32 ia = 0; ia < G; ++ia)
            {
                if (!m_gadded[ia])
                    continue;
                unique_off(ia, true);
                for (u32 fa : m_gfeat[ia])
                {
                    const bool in = bits::test(U, W, fa);
                    bits::reset(U, fa);
                    std::vector<u64>& row = grow_row(rows[fa], W);
                    for (u32 i = lo; i < hi; ++i)
                        row[i] |= U[i];
                    for (u32 i = lo; i < hi; ++i)
                        for (u64 y = U[i]; y; y &= y - 1)
                            bits::set(grow_row(rows[i * 64 + static_cast<u32>(bits::ctz64(y))], W).data(), fa);
                    if (in)
                        bits::set(U, fa);
                }
                unique_off(ia, false);
            }
        }
        return novel;
    }
    // ------------------------------------------------------------------ width 2 without landmarks: incremental
    // The successor's feature counts are the expanded state's (begin) with the transition's deleted and added atoms
    // applied, so a transition costs its delta, not its whole state.
    [[nodiscard]] bool incremental() const noexcept { return m_dense2 && !m_coords; }
    void begin_incremental(const u64* w, u32 n)
    {
        for (u32 a : m_patoms)
            for (u32 f : features(a))
                m_cnt[f] = 0;
        m_patoms.clear();
        bits::for_each(w, n, [&](u64 a) { m_patoms.push_back(static_cast<u32>(a)); });
        for (u32 a : m_patoms)
            features(a);  // interns first: the ids below are final
        const u32 F = static_cast<u32>(m_ids.size());
        if (m_cnt.size() < F)
            m_cnt.resize(F, 0);
        m_pu.assign(bits::words_for(F), 0);
        m_plo = static_cast<u32>(m_pu.size());
        m_phi = 0;
        for (u32 a : m_patoms)
            for (u32 f : features(a))
                if (m_cnt[f]++ == 0)
                {
                    bits::set(m_pu.data(), f);
                    m_plo = std::min(m_plo, f >> 6);
                    m_phi = std::max(m_phi, (f >> 6) + 1);
                }
    }
    /// m_cnt and m_pu: the expanded state's features (begin_incremental) with the transition's deleted and added atoms
    /// applied; extends [m_lo, m_hi) to the added features. revert_delta_features undoes it.
    void apply_delta_features(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        m_dels.clear();
        for (SlotId x : d.del)
            if (bits::test(pw, pn, x.v) && !bits::test(cw, cn, x.v) && std::find(m_dels.begin(), m_dels.end(), x.v) == m_dels.end())
                m_dels.push_back(x.v);
        for (u32 a : add)
            features(a);  // interns first
        const u32 F = static_cast<u32>(m_ids.size());
        if (m_cnt.size() < F)
            m_cnt.resize(F, 0);
        if (m_pu.size() < bits::words_for(F))
            m_pu.resize(bits::words_for(F), 0);
        m_lo = m_plo;
        m_hi = m_phi;
        for (u32 x : m_dels)
            for (u32 f : features(x))
                if (--m_cnt[f] == 0)
                    bits::reset(m_pu.data(), f);
        for (u32 a : add)
            for (u32 f : features(a))
                if (m_cnt[f]++ == 0)
                {
                    bits::set(m_pu.data(), f);
                    m_lo = std::min(m_lo, f >> 6);
                    m_hi = std::max(m_hi, (f >> 6) + 1);
                }
    }
    void revert_delta_features(std::span<const u32> add)
    {
        for (u32 a : add)
            for (u32 f : features(a))
                if (--m_cnt[f] == 0)
                    bits::reset(m_pu.data(), f);
        for (u32 x : m_dels)
            for (u32 f : features(x))
                if (m_cnt[f]++ == 0)
                    bits::set(m_pu.data(), f);
    }
    template<bool Mark>
    bool test_incremental(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d)
    {
        apply_delta_features(pw, pn, cw, cn, add, d);
        m_gfeat.clear();
        m_gadded.clear();
        for (u32 a : add)
        {
            m_gfeat.push_back(features(a));
            m_gadded.push_back(1);
        }
        const bool novel = dense_core<Mark>(0, m_pu.data(), static_cast<u32>(m_pu.size()), m_lo, m_hi);
        revert_delta_features(add);
        if (m_dense_bytes > m_cfg.max_dense_bytes)
        {
            for (u32 a : m_patoms)
                for (u32 f : features(a))
                    m_cnt[f] = 0;
            m_patoms.clear();
            leave_dense();
        }
        return novel;
    }
    std::vector<u64>& grow_row(std::vector<u64>& row, u32 words)
    {
        if (row.size() < words)
        {
            m_dense_bytes += (words - row.size()) * 8;
            row.resize(words, 0);
        }
        return row;
    }
    /// The dense pair tables into the pair set (one way).
    void leave_dense()
    {
        for (u32 r = 0; r < m_prow.size(); ++r)
            for (u32 a = 0; a < m_prow[r].size(); ++a)
                bits::for_each(m_prow[r][a].data(), static_cast<u32>(m_prow[r][a].size()),
                               [&](u64 b)
                               {
                                   if (b > a)
                                       insert2(r, a, static_cast<u32>(b));
                               });
        m_prow.clear();
        m_prow.shrink_to_fit();
        m_dense_bytes = 0;
        m_dense2 = false;
    }
    /// Every tuple of a state (mimir's test_state_novelty_and_update_table).
    template<bool Mark>
    bool state_update(const u64* w, u32 n, u32 rank)
    {
        if (m_cfg.width == 1)
        {
            bool novel = false;
            bits::for_each(w, n,
                           [&](u64 a)
                           {
                               for (u32 f : features(static_cast<u32>(a)))
                                   novel = touch1<Mark>(rank, f) || novel;
                           });
            return novel;
        }
        load_groups(w, n, nullptr, 0);
        return group_tuples<Mark>(rank);
    }
    /// The tuples of a transition that contain an added atom (mimir's test_transition_novelty_and_update_table).
    template<bool Mark>
    bool transition_update(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, u32 rank)
    {
        if (m_cfg.width == 1)
        {
            bool novel = false;
            for (u32 a : add)
                for (u32 f : features(a))
                    novel = touch1<Mark>(rank, f) || novel;
            return novel;
        }
        load_groups(cw, cn, pw, pn);
        return group_tuples<Mark>(rank);
    }

    const Task& m_task;
    AbstractedConfig m_cfg;
    const novelty::LandmarkCoordinates* m_coords;
    std::vector<u32> m_preserved;  // sorted slots with a full-identity feature
    std::unordered_map<std::vector<u32>, u32, KeyHash> m_ids;
    std::vector<u32> m_key;
    std::vector<std::vector<u32>> m_features;  // per slot (empty: not computed yet)
    std::vector<std::vector<u64>> m_single;    // per rank: bitset over feature ids
    novelty::TupleSet m_pairs, m_triples;
    // scratch
    std::vector<u64> m_pmask, m_smask;
    bool m_pany = false;
    std::vector<u32> m_ranks, m_flipped, m_kept;
    std::vector<u64> m_fmask, m_kmask;
    bool m_mirror = false;
    u32 m_rw = 0;
    std::vector<u64> m_by_feature;  // feature-major mirror of m_single (m_mirror)
    std::vector<u32> m_gatoms;
    std::vector<u8> m_gadded;
    std::vector<std::span<const u32>> m_gfeat;
    // width 2, dense (dense_pairs)
    bool m_dense2 = false;
    std::vector<std::vector<std::vector<u64>>> m_prow;  // per rank, per feature: its marked partners
    u64 m_dense_bytes = 0;
    std::vector<u64> m_u;    // scratch: the loaded features
    std::vector<u32> m_cnt;  // scratch: per feature, the loaded groups (incremental: the successor's atoms) that have it
    std::vector<u32> m_patoms, m_dels;  // incremental: the expanded state's atoms, a transition's deleted atoms
    std::vector<u64> m_pu;              // incremental: the successor's features (the expanded state's between tests)
    u32 m_plo = 0, m_phi = 0, m_lo = 0, m_hi = 0;
};
}  // namespace
}  // namespace detail

IwResult abstracted_iw(const Task& task, const AbstractedIwOptions& o)
{
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: abstracted IW does not support tasks with numeric fluents (IW and SIW do)");
    if (o.width < 1 || o.width > 3)
        return detail::failed("abstracted IW width must be 1, 2 or 3 (got " + std::to_string(o.width) + ")");
    Successors& succ = task.workspace().successors();
    const detail::GoalTest goal = detail::GoalTest::from_spec(task, o.control.goal);
    const detail::BlockedSet blocked(o.control.blocked_states);
    detail::Env env(task, succ, goal, blocked, o.control);
    env.witness = o.witness_pruning;
    env.canonical = o.canonical_order;
    env.successor_order = &o.successor_order;
    detail::LayerOrderer layers;
    if (std::string e = detail::apply_layers(env, o.layers, layers); !e.empty())
        return detail::failed(std::move(e));
    std::optional<novelty::LandmarkCoordinates> coords;
    try
    {
        if (o.landmarks.graph)
            coords = detail::make_coordinates(task, o.landmarks);
    }
    catch (const std::invalid_argument& e)
    {
        return detail::failed(e.what());
    }
    const detail::AbstractedConfig cfg{.width = o.width,
                                       .max_dense_bytes = o.landmarks.tables.max_dense_bytes,
                                       .base_abstracted = o.base_abstracted,
                                       .preserve_goal_atoms = o.preserve_goal_atoms,
                                       .preserve_landmark_atoms = o.preserve_landmark_atoms};
    const State root = o.start ? *o.start : task.initial_state();
    if (env.root)
        env.root->on_start(root);
    IwResult r = detail::novelty_ladder(env, o.width, o.width, false,
                                        [&](u32 k, detail::PassOut& po)
                                        {
                                            detail::AbstractedPruner p(task, cfg, coords ? &*coords : nullptr);
                                            detail::novelty_pass(env, root, p,
                                                                 detail::PassConfig{.arity = k,
                                                                                    .root = detail::RootRule::Continuation,
                                                                                    .keep_depth_one = o.keep_depth_one_novel},
                                                                 po);
                                        });
    detail::finish_result(env, root, r);
    return r;
}

IwResult projective_iw(const Task& task, const ProjectiveIwOptions& o)
{
    AbstractedIwOptions a = o;
    a.width = 1;
    a.base_abstracted = !o.typed_projection;
    a.preserve_goal_atoms = o.keep_goal_nonunary_atoms;
    return abstracted_iw(task, a);
}
}  // namespace mymyr::search
