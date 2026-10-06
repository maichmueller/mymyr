// Lifted fact landmarks (landmarks/lifted.hpp).

#include "mymyr/landmarks/lifted.hpp"

#include "../reachability/join.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/reachability/relaxed_reachability.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <deque>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace mymyr::landmarks
{
using namespace formalism;
using reach::is_var;
using reach::term_obj;
using reachability::RelaxedReachability;
using reachability::Table;

namespace
{
struct VecHash
{
    usize operator()(const std::vector<u32>& v) const noexcept
    {
        u64 h = 0x9E3779B97F4A7C15ULL ^ v.size();
        for (u32 x : v)
            h = hash::mix64(h ^ (x + 0x632BE59BD9B4E019ULL));
        return static_cast<usize>(h);
    }
};
template<class V>
using VecMap = std::unordered_map<std::vector<u32>, V, VecHash>;

struct LitRef
{
    u32 pred = 0;
    bool positive = true;
    std::vector<i32> terms;
};

/// One (schema, conditional effect) pair with its literals split: slots are the schema's parameters, then the
/// effect's own (the order of TaskData terms).
struct SchemaEffect
{
    u32 schema = 0, ce = 0, num_slots = 0;
    std::vector<const u64*> types;  // per slot: the objects of its declared types
    std::vector<LitRef> pos_fluent, statics, pos_effects;
};

struct Adder
{
    u32 se = 0, eff = 0, id = 0;
};

/// Ground atoms of one kind (the static atoms, the initial fluent atoms) per predicate, with (position, object) buckets.
class GroundAtoms
{
public:
    void init(const TaskData& T, const std::vector<GroundAtom>& atoms)
    {
        m_args.assign(T.predicates.size(), {});
        m_ids.assign(T.predicates.size(), {});
        m_arity.assign(T.predicates.size(), 0);
        for (u32 p = 0; p < T.predicates.size(); ++p)
            m_arity[p] = T.predicates[p].arity;
        for (const GroundAtom& a : atoms)
        {
            const u32 p = a.pred.v;
            const u32 id = static_cast<u32>(m_ids[p].size());
            m_ids[p].push_back(id);
            u32 i = 0;
            for (ObjectId o : T.objects_of(a))
            {
                m_args[p].push_back(o.v);
                m_buckets[key(p, i++, o.v)].push_back(id);
            }
        }
    }
    [[nodiscard]] const u32* args(u32 p, u32 id) const { return m_args[p].data() + static_cast<usize>(id) * m_arity[p]; }
    /// The atoms of p that can match `pattern` (k_free: any): the smallest bucket a bound position selects.
    [[nodiscard]] std::span<const u32> candidates(u32 p, std::span<const u32> pattern) const
    {
        const std::vector<u32>* best = nullptr;
        for (u32 i = 0; i < pattern.size(); ++i)
        {
            if (pattern[i] == k_free)
                continue;
            const auto it = m_buckets.find(key(p, i, pattern[i]));
            if (it == m_buckets.end())
                return {};
            if (!best || it->second.size() < best->size())
                best = &it->second;
        }
        return best ? std::span<const u32>(*best) : std::span<const u32>(m_ids[p]);
    }

private:
    static u64 key(u32 p, u32 pos, u32 o) { return (static_cast<u64>(p) << 44) | (static_cast<u64>(pos) << 32) | o; }
    std::vector<std::vector<u32>> m_args, m_ids;
    std::vector<u32> m_arity;
    std::unordered_map<u64, std::vector<u32>> m_buckets;
};

class Generator
{
public:
    Generator(const Task& task, const RelaxedReachability* rr, const LiftedFactLandmarkOptions& o)
        : m_task(task), T(task.data()), L(task.atoms().layout()), O(o), m_rr(rr), m_st(task), OW(m_st.ow())
    {
        m_statics.init(T, T.static_init);
        m_finit.init(T, T.fluent_init);
        std::vector<u32> args;
        for (const GroundAtom& a : T.fluent_init)
        {
            args.clear();
            for (ObjectId ob : T.objects_of(a))
                args.push_back(ob.v);
            const CanonicalAtom c = encode(a.pred.v, args);
            if (c != k_no_atom)
                m_init.push_back(c);
        }
        std::sort(m_init.begin(), m_init.end());
        m_init.erase(std::unique(m_init.begin(), m_init.end()), m_init.end());
        collect_schema_effects();
    }

    FactLandmarkGraph run();

private:
    // ------------------------------------------------------------------------------------ basics
    [[nodiscard]] CanonicalAtom encode(u32 pred, std::span<const u32> args) const
    {
        if (L.offset[pred] == CanonicalLayout::k_none || L.size[pred] == 0)
            return k_no_atom;
        const CanonicalAtom c = L.encode(pred, args.data());
        return c < L.fluent_count ? c : k_no_atom;
    }
    [[nodiscard]] bool initially(CanonicalAtom c) const { return std::binary_search(m_init.begin(), m_init.end(), c); }
    [[nodiscard]] bool any_initially(std::span<const CanonicalAtom> cs) const
    {
        return std::any_of(cs.begin(), cs.end(), [&](CanonicalAtom c) { return initially(c); });
    }
    [[nodiscard]] static u32 resolve(i32 t, const std::vector<u32>& sigma)
    {
        return is_var(t) ? (static_cast<u32>(t) < sigma.size() ? sigma[t] : k_free) : term_obj(t);
    }
    [[nodiscard]] static u32 slot_of(i32 t) { return is_var(t) ? static_cast<u32>(t) : k_free; }
    [[nodiscard]] bool static_has(u32 pred, std::span<const u32> objects) { return m_st.contains(pred, objects.data()); }

    void decode_members(std::span<const CanonicalAtom> members, std::vector<std::vector<u32>>& out) const
    {
        out.clear();
        std::vector<u32> args(std::max<u32>(1, L.max_arity));
        for (CanonicalAtom c : members)
        {
            const u32 p = L.decode(c, args.data());
            out.emplace_back(args.begin(), args.begin() + L.arity[p]);
        }
    }

    void collect_schema_effects()
    {
        auto add_lits = [&](std::span<const Literal> lits, SchemaEffect& se)
        {
            const usize f0 = se.pos_fluent.size(), s0 = se.statics.size();
            for (const Literal& l : lits)
            {
                LitRef r{l.pred.v, l.positive, {}};
                for (Term t : T.terms_of(l))
                    r.terms.push_back(t);
                const PredKind k = T.predicates[l.pred.v].kind;
                auto dup = [&](const std::vector<LitRef>& v, usize from)
                {
                    for (usize i = from; i < v.size(); ++i)
                        if (v[i].pred == r.pred && v[i].positive == r.positive && v[i].terms == r.terms)
                            return true;
                    return false;
                };
                // a condition is a set of literals in mimir (TaskData repeats nullary ones)
                if (k == PredKind::Fluent && l.positive && !dup(se.pos_fluent, f0))
                    se.pos_fluent.push_back(std::move(r));
                else if (k == PredKind::Static && !dup(se.statics, s0))
                    se.statics.push_back(std::move(r));
            }
        };
        m_adders.assign(T.predicates.size(), {});
        u32 next_id = 0;
        for (u32 s = 0; s < T.schemas.size(); ++s)
        {
            const Schema& sc = T.schemas[s];
            const auto ces = T.effects_of(sc);
            for (u32 k = 0; k < ces.size(); ++k)
            {
                const ConditionalEffect& ce = ces[k];
                SchemaEffect se;
                se.schema = s;
                se.ce = k;
                se.num_slots = sc.arity() + ce.extra_params.count;
                for (Range r : {sc.params, ce.extra_params})
                    for (const Parameter& par : TaskData::slice(T.params, r))
                    {
                        std::vector<u32> types;
                        for (TypeId t : TaskData::slice(T.type_ids, par.types))
                            types.push_back(t.v);
                        se.types.push_back(m_st.objects_of_types(types));
                    }
                add_lits(T.literals_of(sc.precondition), se);
                add_lits(T.literals_of(ce.condition), se);
                for (const Literal& l : TaskData::slice(T.literals, ce.effects))
                    if (l.positive)
                    {
                        LitRef r{l.pred.v, true, {}};
                        for (Term t : T.terms_of(l))
                            r.terms.push_back(t);
                        se.pos_effects.push_back(std::move(r));
                    }
                m_schemas.push_back(std::move(se));
            }
        }
        for (u32 i = 0; i < m_schemas.size(); ++i)
            for (u32 e = 0; e < m_schemas[i].pos_effects.size(); ++e)
                m_adders[m_schemas[i].pos_effects[e].pred].push_back({i, e, next_id++});
    }

    // ------------------------------------------------------------------------------------ achievers
    struct Achiever
    {
        u32 se = 0, eff = 0;
        std::vector<u32> sigma;  // k_free: unbound
        std::vector<u64> cand;   // num_slots * OW: candidates of the unbound slots
        std::vector<u32> count;
    };
    [[nodiscard]] u64* cand(Achiever& a, u32 slot) const { return a.cand.data() + static_cast<usize>(slot) * OW; }
    [[nodiscard]] const u64* cand(const Achiever& a, u32 slot) const { return a.cand.data() + static_cast<usize>(slot) * OW; }
    [[nodiscard]] bool in(const u64* bits, u32 o) const { return (bits[o >> 6] >> (o & 63)) & 1; }

    /// Unifies the add literal `eff` of `se` with `binding` (k_free: free) and optionally runs the static filter.
    std::optional<Achiever> try_build(u32 se, u32 eff, std::span<const u32> binding, bool filter)
    {
        const SchemaEffect& S = m_schemas[se];
        const LitRef& e = S.pos_effects[eff];
        if (e.terms.size() != binding.size())
            return std::nullopt;
        Achiever a;
        a.se = se;
        a.eff = eff;
        a.sigma.assign(S.num_slots, k_free);
        for (u32 i = 0; i < binding.size(); ++i)
        {
            if (binding[i] == k_free)
                continue;
            const i32 t = e.terms[i];
            if (!is_var(t))
            {
                if (term_obj(t) != binding[i])
                    return std::nullopt;
                continue;
            }
            const u32 slot = static_cast<u32>(t);
            if ((a.sigma[slot] != k_free && a.sigma[slot] != binding[i]) || !in(S.types[slot], binding[i]))
                return std::nullopt;
            a.sigma[slot] = binding[i];
        }
        a.cand.assign(static_cast<usize>(S.num_slots) * OW, 0);
        a.count.assign(S.num_slots, 0);
        for (u32 slot = 0; slot < S.num_slots; ++slot)
            if (a.sigma[slot] == k_free)
            {
                std::copy(S.types[slot], S.types[slot] + OW, cand(a, slot));
                a.count[slot] = static_cast<u32>(bits::count(cand(a, slot), OW));
            }
        if (filter && !static_filter(a))
            return std::nullopt;
        return a;
    }

    /// Binds every unbound slot with a single candidate. Returns whether one was bound.
    bool bind_singletons(Achiever& a) const
    {
        bool changed = false;
        for (u32 slot = 0; slot < a.sigma.size(); ++slot)
            if (a.sigma[slot] == k_free && a.count[slot] == 1)
            {
                const u64* c = cand(a, slot);
                for (u32 w = 0; w < OW; ++w)
                    if (c[w])
                    {
                        a.sigma[slot] = w * 64 + static_cast<u32>(bits::ctz64(c[w]));
                        break;
                    }
                changed = true;
            }
        return changed;
    }

    /// Replaces the candidates of `slot` by `now` (a subset); `changed` if it shrank.
    void narrow_to(Achiever& a, u32 slot, const u64* now, bool& changed) const
    {
        u64* c = cand(a, slot);
        std::copy(now, now + OW, c);
        const u32 k = static_cast<u32>(bits::count(c, OW));
        changed |= k < a.count[slot];
        a.count[slot] = k;
    }

    /// Scratch for the per-literal narrowings: per slot an object bitset, touched slots listed.
    struct Narrowed
    {
        std::vector<u64> bits;
        std::vector<u32> slots, local;
        std::vector<u32> local_slots;
        void reset(u32 num_slots, u32 ow)
        {
            if (bits.size() < static_cast<usize>(num_slots) * ow)
                bits.resize(static_cast<usize>(num_slots) * ow);
            for (u32 s : slots)
                std::fill(bits.begin() + static_cast<std::ptrdiff_t>(s) * ow, bits.begin() + static_cast<std::ptrdiff_t>(s + 1) * ow, 0);
            slots.clear();
            local.assign(num_slots, k_free);
            local_slots.clear();
        }
    };

    /// Matches one ground atom (args) against a literal's terms under sigma and the candidates: collects the free
    /// slots' objects into `nw.local` (repeated slots agree). False if inconsistent.
    bool match(const LitRef& l, const u32* args, const Achiever& a, std::span<const u32> pattern, Narrowed& nw) const
    {
        nw.local_slots.clear();
        bool ok = true;
        for (u32 i = 0; ok && i < l.terms.size(); ++i)
        {
            if (pattern[i] != k_free)
            {
                ok = pattern[i] == args[i];
                continue;
            }
            const u32 slot = slot_of(l.terms[i]);
            if (slot == k_free || slot >= a.sigma.size())
                continue;
            if (!in(cand(a, slot), args[i]))
            {
                ok = false;
                break;
            }
            if (nw.local[slot] != k_free)
                ok = nw.local[slot] == args[i];
            else
            {
                nw.local[slot] = args[i];
                nw.local_slots.push_back(slot);
            }
        }
        return ok;
    }
    void collect(Narrowed& nw) const
    {
        for (u32 slot : nw.local_slots)
        {
            u64* b = nw.bits.data() + static_cast<usize>(slot) * OW;
            if (std::find(nw.slots.begin(), nw.slots.end(), slot) == nw.slots.end())
                nw.slots.push_back(slot);
            b[nw.local[slot] >> 6] |= u64{1} << (nw.local[slot] & 63);
        }
    }
    void unmatch(Narrowed& nw) const
    {
        for (u32 slot : nw.local_slots)
            nw.local[slot] = k_free;
    }
    std::vector<u32> pattern_of(const LitRef& l, const std::vector<u32>& sigma) const
    {
        std::vector<u32> p;
        p.reserve(l.terms.size());
        for (i32 t : l.terms)
            p.push_back(resolve(t, sigma));
        return p;
    }

    /// The static filter (static_filter below), for one positive static literal.
    bool narrow_positive_static(const LitRef& l, Achiever& a, bool& changed)
    {
        const std::vector<u32> pattern = pattern_of(l, a.sigma);
        m_nw.reset(static_cast<u32>(a.sigma.size()), OW);
        bool any = false;
        for (u32 id : m_statics.candidates(l.pred, pattern))
        {
            const bool ok = match(l, m_statics.args(l.pred, id), a, pattern, m_nw);
            if (ok)
            {
                any = true;
                collect(m_nw);
            }
            unmatch(m_nw);
        }
        if (!any)
            return false;
        for (u32 slot : m_nw.slots)
            narrow_to(a, slot, m_nw.bits.data() + static_cast<usize>(slot) * OW, changed);
        return true;
    }

    /// The static filter, for PDDL equality, on object identity.
    bool narrow_equality(const LitRef& l, Achiever& a, bool& changed) const
    {
        if (l.terms.size() != 2)
            return true;
        const u32 lhs = resolve(l.terms[0], a.sigma), rhs = resolve(l.terms[1], a.sigma);
        if (!l.positive)
            return !(lhs != k_free && rhs != k_free && lhs == rhs);
        if (lhs != k_free && rhs != k_free)
            return lhs == rhs;
        const u32 bound = lhs != k_free ? lhs : rhs;
        const u32 free_slot = lhs != k_free ? slot_of(l.terms[1]) : slot_of(l.terms[0]);
        if (bound != k_free && free_slot != k_free && free_slot < a.sigma.size())
        {
            u64* c = cand(a, free_slot);
            if (!in(c, bound))
                return false;
            changed |= a.count[free_slot] > 1;
            std::fill(c, c + OW, 0);
            c[bound >> 6] |= u64{1} << (bound & 63);
            a.count[free_slot] = 1;
        }
        return true;
    }

    /// Runs the static filter to a fixpoint. False drops the achiever.
    bool static_filter(Achiever& a)
    {
        const SchemaEffect& S = m_schemas[a.se];
        for (bool changed = true; changed;)
        {
            changed = false;
            for (const LitRef& l : S.statics)
            {
                if (m_st.is_equality(l.pred))
                {
                    if (!narrow_equality(l, a, changed))
                        return false;
                }
                else if (l.positive)
                {
                    if (!narrow_positive_static(l, a, changed))
                        return false;
                }
                else
                {
                    const std::vector<u32> objs = pattern_of(l, a.sigma);
                    if (std::find(objs.begin(), objs.end(), k_free) == objs.end() && static_has(l.pred, objs))
                        return false;
                }
            }
            changed |= bind_singletons(a);
        }
        return true;
    }

    /// Per-member test: sigma statically consistent for the schema effect.
    bool statically_consistent(const SchemaEffect& S, const std::vector<u32>& sigma)
    {
        for (const LitRef& l : S.statics)
        {
            const std::vector<u32> objs = pattern_of(l, sigma);
            const bool bound = std::find(objs.begin(), objs.end(), k_free) == objs.end();
            if (m_st.is_equality(l.pred))
            {
                if (l.terms.size() == 2 && bound && ((objs[0] == objs[1]) != l.positive))
                    return false;
                continue;
            }
            if (!l.positive)
            {
                if (bound && static_has(l.pred, objs))
                    return false;
                continue;
            }
            if (bound)
            {
                if (!static_has(l.pred, objs))
                    return false;
                continue;
            }
            bool matched = false;
            for (u32 id : m_statics.candidates(l.pred, objs))
            {
                const u32* args = m_statics.args(l.pred, id);
                bool ok = true;
                for (u32 i = 0; ok && i < objs.size(); ++i)
                    ok = objs[i] == k_free || objs[i] == args[i];
                if (ok)
                {
                    matched = true;
                    break;
                }
            }
            if (!matched)
                return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------------------ self-dependence
    bool adder_needs_pattern(const SchemaEffect& S, const std::vector<u32>& sigma, u32 pred, std::span<const u32> pattern) const
    {
        for (const LitRef& l : S.pos_fluent)
        {
            if (l.pred != pred || l.terms.size() != pattern.size())
                continue;
            bool covers = true;
            for (u32 i = 0; covers && i < pattern.size(); ++i)
                covers = pattern[i] == k_free || resolve(l.terms[i], sigma) == pattern[i];
            if (covers)
                return true;
        }
        return false;
    }

    const std::optional<std::vector<u32>>& adder_substitution(const Adder& ad, const std::vector<u32>& pattern)
    {
        std::vector<u32> key{ad.id};
        key.insert(key.end(), pattern.begin(), pattern.end());
        if (auto it = m_adder_cache.find(key); it != m_adder_cache.end())
            return it->second;
        auto a = try_build(ad.se, ad.eff, pattern, true);
        std::optional<std::vector<u32>> sigma;
        if (a)
            sigma = std::move(a->sigma);
        return m_adder_cache.emplace(std::move(key), std::move(sigma)).first->second;
    }

    bool self_dependent_rule(Achiever& a, u32 pred, std::span<const u32> binding, VecMap<bool>& decided, bool& changed)
    {
        const SchemaEffect& S = m_schemas[a.se];
        for (const LitRef& l : S.pos_fluent)
        {
            const std::vector<u32> pp = pattern_of(l, a.sigma);
            bool every = true;
            std::vector<u32> id{l.pred};
            id.insert(id.end(), pp.begin(), pp.end());
            if (auto it = decided.find(id); it != decided.end())
                every = it->second;
            else
            {
                for (const Adder& ad : m_adders[l.pred])
                {
                    const auto& sigma = adder_substitution(ad, pp);
                    if (!sigma)
                        continue;
                    if (!adder_needs_pattern(m_schemas[ad.se], *sigma, pred, binding))
                    {
                        every = false;
                        break;
                    }
                }
                decided.emplace(std::move(id), every);
            }
            if (!every)
                continue;
            // every way of producing this precondition needs the pattern: the first achiever used an initial atom
            m_nw.reset(static_cast<u32>(a.sigma.size()), OW);
            bool any = false;
            for (u32 aid : m_finit.candidates(l.pred, pp))
            {
                const bool ok = match(l, m_finit.args(l.pred, aid), a, pp, m_nw);
                if (ok)
                {
                    any = true;
                    collect(m_nw);
                }
                unmatch(m_nw);
            }
            if (!any)
                return false;
            for (u32 slot : m_nw.slots)
                narrow_to(a, slot, m_nw.bits.data() + static_cast<usize>(slot) * OW, changed);
        }
        changed |= bind_singletons(a);
        return true;
    }

    // ------------------------------------------------------------------------------------ reachability
    /// The unrestricted table's tuples of a predicate, bucketed by (position, object).
    struct Reachable
    {
        std::vector<u32> all;
        std::unordered_map<u64, std::vector<u32>> buckets;
        bool built = false;
    };
    std::span<const u32> reachable_candidates(u32 pred, std::span<const u32> pattern)
    {
        if (m_reach.empty())
            m_reach.resize(T.predicates.size());
        Reachable& R = m_reach[pred];
        const u32 ar = T.predicates[pred].arity;
        if (!R.built)
        {
            R.built = true;
            const auto atoms = m_rr->table().atoms(PredicateId{pred});
            const auto tuples = m_rr->table().tuples(PredicateId{pred});
            for (u32 i = 0; i < atoms.size(); ++i)
            {
                R.all.push_back(i);
                for (u32 k = 0; k < ar; ++k)
                    R.buckets[(static_cast<u64>(k) << 32) | tuples[static_cast<usize>(i) * ar + k]].push_back(i);
            }
        }
        const std::vector<u32>* best = nullptr;
        for (u32 i = 0; i < pattern.size(); ++i)
        {
            if (pattern[i] == k_free)
                continue;
            const auto it = R.buckets.find((static_cast<u64>(i) << 32) | pattern[i]);
            if (it == R.buckets.end())
                return {};
            if (!best || it->second.size() < best->size())
                best = &it->second;
        }
        return best ? std::span<const u32>(*best) : std::span<const u32>(R.all);
    }
    [[nodiscard]] const u32* reachable_args(u32 pred, u32 i) const
    {
        return m_rr->table().tuples(PredicateId{pred}).data() + static_cast<usize>(i) * T.predicates[pred].arity;
    }

    /// Membership in R_{¬M}: the witness store where it can answer, the restricted fixpoint (at most once) otherwise.
    class Membership
    {
    public:
        Membership(const RelaxedReachability& e, std::vector<CanonicalAtom> forbidden) : m_e(e), m_forbidden(std::move(forbidden))
        {
            if (e.table().has_witnesses())
                m_witness.emplace(e.table().witness_query(m_forbidden));
        }
        bool contains(CanonicalAtom c)
        {
            if (c == k_no_atom)
                return false;
            if (m_witness && m_witness->avoids(c) == reachability::WitnessVerdict::ReachableWithout)
                return true;
            return table().is_reachable(c);
        }
        const Table& table()
        {
            if (!m_table)
                m_table.emplace(m_e.restricted(m_forbidden));
            return *m_table;
        }

    private:
        const RelaxedReachability& m_e;
        std::vector<CanonicalAtom> m_forbidden;
        std::optional<reachability::WitnessQuery> m_witness;
        std::optional<Table> m_table;
    };

    /// Narrows by reachability, per literal.
    bool narrow_reachable_literal(const LitRef& l, Achiever& a, Membership* membership, bool& changed)
    {
        const std::vector<u32> pattern = pattern_of(l, a.sigma);
        m_nw.reset(static_cast<u32>(a.sigma.size()), OW);
        bool any = false;
        for (u32 i : reachable_candidates(l.pred, pattern))
        {
            const u32* args = reachable_args(l.pred, i);
            bool ok = match(l, args, a, pattern, m_nw);
            if (ok && membership && !membership->contains(encode(l.pred, std::span<const u32>(args, l.terms.size()))))
                ok = false;
            if (ok)
            {
                any = true;
                collect(m_nw);
            }
            unmatch(m_nw);
        }
        if (!any)
            return false;
        for (u32 slot : m_nw.slots)
            narrow_to(a, slot, m_nw.bits.data() + static_cast<usize>(slot) * OW, changed);
        return true;
    }

    bool narrow_per_literal(Achiever& a, Membership* membership, bool& changed)
    {
        for (const LitRef& l : m_schemas[a.se].pos_fluent)
            if (!narrow_reachable_literal(l, a, membership, changed))
                return false;
        changed |= bind_singletons(a);
        return true;
    }

    /// Narrows by reachability jointly: the projection of the positive fluent preconditions and the static literals
    /// over `table`.
    bool narrow_joint(Achiever& a, const Table& table, bool& changed)
    {
        const SchemaEffect& S = m_schemas[a.se];
        std::vector<u32> var_of(a.sigma.size(), k_free), slot_of_var;
        auto declare = [&](i32 t)
        {
            if (resolve(t, a.sigma) != k_free)
                return;
            const u32 slot = slot_of(t);
            if (slot == k_free || slot >= var_of.size() || var_of[slot] != k_free)
                return;
            var_of[slot] = static_cast<u32>(slot_of_var.size());
            slot_of_var.push_back(slot);
        };
        for (const LitRef& l : S.pos_fluent)
            for (i32 t : l.terms)
                declare(t);
        for (const LitRef& l : S.statics)
            for (i32 t : l.terms)
                declare(t);
        if (slot_of_var.empty())
        {
            // nothing free, but the achiever still has to be possible: its ground preconditions must be reachable
            for (const LitRef& l : S.pos_fluent)
            {
                const std::vector<u32> objs = pattern_of(l, a.sigma);
                if (std::find(objs.begin(), objs.end(), k_free) != objs.end())
                    continue;
                const CanonicalAtom c = encode(l.pred, objs);
                if (c == k_no_atom || !table.is_reachable(c))
                    return false;
            }
            return true;
        }
        reachability::ConjunctiveQuery q;
        q.num_variables = static_cast<u32>(slot_of_var.size());
        auto to_term = [&](i32 t) -> reachability::QueryTerm
        {
            const u32 o = resolve(t, a.sigma);
            if (o != k_free)
                return reachability::query_object(ObjectId{o});
            return reachability::query_variable(var_of[slot_of(t)]);
        };
        for (const LitRef& l : S.pos_fluent)
        {
            reachability::QueryLiteral ql{PredicateId{l.pred}, {}, true};
            for (i32 t : l.terms)
                ql.terms.push_back(to_term(t));
            q.literals.push_back(std::move(ql));
        }
        for (const LitRef& l : S.statics)
        {
            std::vector<reachability::QueryTerm> terms;
            for (i32 t : l.terms)
                terms.push_back(to_term(t));
            if (m_st.is_equality(l.pred) && terms.size() == 2)
            {
                (l.positive ? q.equalities : q.disequalities).emplace_back(terms[0], terms[1]);
                continue;
            }
            q.literals.push_back({PredicateId{l.pred}, std::move(terms), l.positive});
        }
        const auto proj = table.project(q);
        std::vector<u64> admissible(OW);
        for (u32 v = 0; v < slot_of_var.size() && v < proj.size(); ++v)
        {
            const u32 slot = slot_of_var[v];
            std::fill(admissible.begin(), admissible.end(), 0);
            for (ObjectId o : proj[v])
                admissible[o.v >> 6] |= u64{1} << (o.v & 63);
            const u64* c = cand(a, slot);
            bool any = false;
            for (u32 w = 0; w < OW; ++w)
            {
                admissible[w] &= c[w];
                any |= admissible[w] != 0;
            }
            if (!any)
                return false;
            narrow_to(a, slot, admissible.data(), changed);
        }
        changed |= bind_singletons(a);
        return true;
    }

    bool narrow_configured(Achiever& a, Membership* membership, bool& changed)
    {
        switch (O.reachability_disambiguation)
        {
            case ReachabilityDisambiguation::Off: return true;
            case ReachabilityDisambiguation::PerLiteral: return narrow_per_literal(a, membership, changed);
            case ReachabilityDisambiguation::Joint: return narrow_joint(a, membership ? membership->table() : m_rr->table(), changed);
        }
        return true;
    }

    /// First narrowing step: the achiever's pattern positions agree with some member (position-wise).
    bool narrow_to_member_projection(Achiever& a, const std::vector<std::vector<u32>>& members, bool& changed)
    {
        const LitRef& e = m_schemas[a.se].pos_effects[a.eff];
        std::vector<u64> proj(OW);
        for (u32 pos = 0; pos < e.terms.size(); ++pos)
        {
            const u32 slot = slot_of(e.terms[pos]);
            if (slot == k_free || slot >= a.sigma.size() || a.sigma[slot] != k_free)
                continue;
            std::fill(proj.begin(), proj.end(), 0);
            for (const auto& m : members)
                if (pos < m.size())
                    proj[m[pos] >> 6] |= u64{1} << (m[pos] & 63);
            const u64* c = cand(a, slot);
            bool any = false;
            for (u32 w = 0; w < OW; ++w)
            {
                proj[w] &= c[w];
                any |= proj[w] != 0;
            }
            if (!any)
                return false;
            narrow_to(a, slot, proj.data(), changed);
        }
        changed |= bind_singletons(a);
        return true;
    }

    /// Static reachability of a candidate member: initially true, or added by some statically consistent,
    /// type-compatible instance of some schema effect.
    bool can_ever_hold(u32 pred, const std::vector<u32>& objects)
    {
        if (initially(encode(pred, objects)))
            return true;
        if (m_adders[pred].empty())
            return false;
        std::vector<u32> key{pred};
        key.insert(key.end(), objects.begin(), objects.end());
        if (auto it = m_hold.find(key); it != m_hold.end())
            return it->second;
        bool reachable = false;
        for (const Adder& ad : m_adders[pred])
            if (try_build(ad.se, ad.eff, objects, true))
            {
                reachable = true;
                break;
            }
        m_hold.emplace(std::move(key), reachable);
        return reachable;
    }

    // ------------------------------------------------------------------------------------ records
    std::optional<u32> insert(u32 pred, std::vector<u32> binding, std::vector<CanonicalAtom> members, std::optional<u32> parent);
    void insert_derived(u32 pred, const std::vector<u32>& binding, std::vector<CanonicalAtom> members, u32 parent);
    void expand(u32 position);

    const Task& m_task;
    const TaskData& T;
    const CanonicalLayout& L;
    const LiftedFactLandmarkOptions& O;
    const RelaxedReachability* m_rr;
    reach::StaticTables m_st;
    u32 OW;
    GroundAtoms m_statics, m_finit;
    std::vector<CanonicalAtom> m_init;
    std::vector<SchemaEffect> m_schemas;
    std::vector<std::vector<Adder>> m_adders;  // per predicate
    Narrowed m_nw;
    VecMap<std::optional<std::vector<u32>>> m_adder_cache;
    VecMap<bool> m_hold;
    std::vector<Reachable> m_reach;
    bool m_member_set_identity = false;
    // records
    std::vector<LiftedLandmark> m_records;
    VecMap<u32> m_position;
    std::deque<u32> m_work;
    std::vector<CanonicalAtom> m_facts;
    std::vector<std::pair<CanonicalAtom, CanonicalAtom>> m_orderings;
};

std::optional<u32> Generator::insert(u32 pred, std::vector<u32> binding, std::vector<CanonicalAtom> members, std::optional<u32> parent)
{
    const bool bound = std::find(binding.begin(), binding.end(), k_free) == binding.end();
    if (bound)
    {
        const CanonicalAtom c = encode(pred, binding);
        if (c == k_no_atom)
            return std::nullopt;  // outside the typed atom space: no state holds it
        members.assign(1, c);
    }
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());

    std::vector<u32> identity{pred};
    identity.insert(identity.end(), binding.begin(), binding.end());
    if (m_member_set_identity && !bound)
        for (CanonicalAtom c : members)
        {
            identity.push_back(static_cast<u32>(c));
            identity.push_back(static_cast<u32>(c >> 32));
        }
    if (auto it = m_position.find(identity); it != m_position.end())
    {
        LiftedLandmark& r = m_records[it->second];
        r.members.insert(r.members.end(), members.begin(), members.end());
        std::sort(r.members.begin(), r.members.end());
        r.members.erase(std::unique(r.members.begin(), r.members.end()), r.members.end());
        if (parent && std::find(r.parents.begin(), r.parents.end(), *parent) == r.parents.end())
            r.parents.push_back(*parent);
        return it->second;
    }
    // subsumption (partial landmarks only)
    if (!bound)
        for (const LiftedLandmark& other : m_records)
        {
            if (other.predicate.v != pred)
                continue;
            if (m_member_set_identity)
            {
                if (std::includes(members.begin(), members.end(), other.members.begin(), other.members.end()))
                    return std::nullopt;
                continue;
            }
            if (other.binding.size() != binding.size())
                continue;
            bool subsumes = true;
            for (u32 i = 0; subsumes && i < binding.size(); ++i)
                subsumes = binding[i] == k_free || other.binding[i] == binding[i];
            if (subsumes)
                return std::nullopt;
        }
    LiftedLandmark r;
    r.predicate = PredicateId{pred};
    r.binding = std::move(binding);
    r.members = std::move(members);
    if (bound)
    {
        r.fact = r.members.front();
        m_facts.push_back(r.fact);
    }
    if (parent)
        r.parents.push_back(*parent);
    const u32 position = static_cast<u32>(m_records.size());
    m_records.push_back(std::move(r));
    m_position.emplace(std::move(identity), position);
    m_work.push_back(position);
    return position;
}

void Generator::insert_derived(u32 pred, const std::vector<u32>& binding, std::vector<CanonicalAtom> members, u32 parent)
{
    std::vector<u32> effective = binding;
    if (std::find(binding.begin(), binding.end(), k_free) != binding.end())
    {
        std::sort(members.begin(), members.end());
        members.erase(std::unique(members.begin(), members.end()), members.end());
        if (members.empty())
        {
            if (m_rr && m_rr->goal_reachable())
            {
                LiftedLandmark x;
                x.predicate = PredicateId{pred};
                x.binding = binding;
                throw std::logic_error("mymyr: lifted landmark extraction derived an empty member set for " + format_lifted(m_task, x) +
                                       ", which contradicts the reachability of the parent's first achiever");
            }
            return;
        }
        if (members.size() == 1)
        {
            std::vector<u32> args(std::max<u32>(1, L.max_arity));
            const u32 p = L.decode(members.front(), args.data());
            effective.assign(args.begin(), args.begin() + L.arity[p]);
        }
    }
    const auto position = insert(pred, std::move(effective), std::move(members), parent);
    if (!position)
        return;
    if (O.compute_greedy_necessary_orderings && m_records[*position].is_fact() && m_records[parent].is_fact())
        m_orderings.emplace_back(m_records[*position].fact, m_records[parent].fact);
}

void Generator::expand(u32 position)
{
    const u32 pred = m_records[position].predicate.v;
    const std::vector<u32> binding = m_records[position].binding;  // by value: m_records grows below

    // already true in the initial state: no achievers needed
    if (any_initially(m_records[position].members))
    {
        m_records[position].initially_true = true;
        return;
    }

    // every achiever, statically filtered
    std::vector<Achiever> achievers;
    for (const Adder& ad : m_adders[pred])
        if (auto a = try_build(ad.se, ad.eff, binding, O.use_static_filter))
            achievers.push_back(std::move(*a));

    // narrowed to members reachable at this point, and to this landmark's own members
    std::optional<Membership> membership;
    if (m_rr && O.first_achievers_restricted)
    {
        membership.emplace(*m_rr, m_records[position].members);
        std::vector<std::vector<u32>> member_objects;
        decode_members(m_records[position].members, member_objects);
        const bool had = !achievers.empty();
        std::vector<Achiever> surviving;
        for (Achiever& a : achievers)
        {
            bool keep = true;
            for (bool changed = true; keep && changed;)
            {
                changed = false;
                keep = narrow_to_member_projection(a, member_objects, changed) && narrow_configured(a, &*membership, changed);
                if (keep && changed)
                    keep = static_filter(a);
            }
            if (keep)
                surviving.push_back(std::move(a));
        }
        if (surviving.empty() && had && m_rr->goal_reachable())
        {
            LiftedLandmark x;
            x.predicate = PredicateId{pred};
            x.binding = binding;
            throw std::logic_error("mymyr: reachability narrowing dropped every achiever of " + format_lifted(m_task, x) +
                                   ", which contradicts the reachability of its first member-adder");
        }
        achievers = std::move(surviving);
    }
    else if (m_rr && O.reachability_disambiguation != ReachabilityDisambiguation::Off)
    {
        std::vector<Achiever> surviving;
        for (Achiever& a : achievers)
        {
            bool keep = true;
            for (bool changed = true; keep && changed;)
            {
                changed = false;
                keep = narrow_configured(a, nullptr, changed);
                if (keep && changed)
                    keep = static_filter(a);
            }
            if (keep)
                surviving.push_back(std::move(a));
        }
        achievers = std::move(surviving);
    }

    // self-dependence check (gated on the pattern having no initial instance)
    if (!(m_rr && O.first_achievers_restricted))
    {
        bool initial_instance = false;
        for (u32 id : m_finit.candidates(pred, binding))
        {
            const u32* args = m_finit.args(pred, id);
            bool ok = true;
            for (u32 i = 0; ok && i < binding.size(); ++i)
                ok = binding[i] == k_free || binding[i] == args[i];
            if (ok)
            {
                initial_instance = true;
                break;
            }
        }
        if (!initial_instance)
        {
            VecMap<bool> decided;
            std::vector<Achiever> surviving;
            for (Achiever& a : achievers)
            {
                bool keep = true;
                for (bool changed = true; keep && changed;)
                {
                    changed = false;
                    keep = self_dependent_rule(a, pred, binding, decided, changed);
                    if (keep && changed)
                        keep = static_filter(a);
                }
                if (keep)
                    surviving.push_back(std::move(a));
            }
            achievers = std::move(surviving);
        }
    }

    if (achievers.empty())
        return;

    // predicates every achiever's positive fluent precondition mentions, ascending
    std::vector<std::vector<std::vector<u32>>> occ(achievers.size());  // [achiever][predicate] -> literal indices
    for (u32 j = 0; j < achievers.size(); ++j)
    {
        occ[j].assign(T.predicates.size(), {});
        const auto& pf = m_schemas[achievers[j].se].pos_fluent;
        for (u32 li = 0; li < pf.size(); ++li)
            occ[j][pf[li].pred].push_back(li);
    }
    for (u32 q = 0; q < T.predicates.size(); ++q)
    {
        if (occ[0][q].empty())
            continue;
        bool everywhere = true;
        for (u32 j = 1; j < achievers.size() && everywhere; ++j)
            everywhere = !occ[j][q].empty();
        if (!everywhere)
            continue;
        usize combos = 1;
        for (u32 j = 0; j < achievers.size(); ++j)
        {
            const usize k = occ[j][q].size();
            combos = combos > O.max_occurrence_combinations / std::max<usize>(k, 1) ? static_cast<usize>(-1) : combos * k;
        }
        const bool all = O.max_occurrence_combinations > 1 && combos <= O.max_occurrence_combinations;
        const usize vectors = all ? combos : 1;
        const u32 arity = T.predicates[q].arity;
        for (usize combination = 0; combination < vectors; ++combination)
        {
            std::vector<u32> choice(achievers.size(), 0);
            usize rem = combination;
            for (u32 j = static_cast<u32>(achievers.size()); j-- > 0;)
            {
                choice[j] = static_cast<u32>(rem % occ[j][q].size());
                rem /= occ[j][q].size();
            }
            auto lit = [&](u32 j) -> const LitRef& { return m_schemas[achievers[j].se].pos_fluent[occ[j][q][choice[j]]]; };
            std::vector<u32> derived(arity, k_free);
            for (u32 p = 0; p < arity; ++p)
            {
                u32 agreed = k_free;
                bool agrees = true;
                for (u32 j = 0; agrees && j < achievers.size(); ++j)
                {
                    const u32 o = resolve(lit(j).terms[p], achievers[j].sigma);
                    agrees = o != k_free && (agreed == k_free || agreed == o);
                    agreed = o;
                }
                derived[p] = agrees ? agreed : k_free;
            }
            // the members
            std::vector<CanonicalAtom> members;
            if (std::find(derived.begin(), derived.end(), k_free) != derived.end())
            {
                for (u32 j = 0; j < achievers.size(); ++j)
                {
                    const Achiever& a = achievers[j];
                    const LitRef& l = lit(j);
                    std::vector<u32> free_slots;
                    for (i32 t : l.terms)
                    {
                        const u32 slot = slot_of(t);
                        if (slot != k_free && slot < a.sigma.size() && a.sigma[slot] == k_free &&
                            std::find(free_slots.begin(), free_slots.end(), slot) == free_slots.end())
                            free_slots.push_back(slot);
                    }
                    if (std::any_of(free_slots.begin(), free_slots.end(), [&](u32 s) { return a.count[s] == 0; }))
                        continue;
                    // odometer over the candidates (ascending objects), last slot fastest
                    std::vector<std::vector<u32>> lists(free_slots.size());
                    for (u32 k = 0; k < free_slots.size(); ++k)
                        bits::for_each(cand(a, free_slots[k]), OW, [&](u64 o) { lists[k].push_back(static_cast<u32>(o)); });
                    std::vector<u32> completion = a.sigma, odo(free_slots.size(), 0), objs;
                    for (bool exhausted = false; !exhausted;)
                    {
                        for (u32 k = 0; k < free_slots.size(); ++k)
                            completion[free_slots[k]] = lists[k][odo[k]];
                        if (statically_consistent(m_schemas[a.se], completion))
                        {
                            objs = pattern_of(l, completion);
                            const CanonicalAtom c = encode(q, objs);
                            if (c != k_no_atom && can_ever_hold(q, objs) &&
                                (!m_rr || !O.reachability_filter_members || m_rr->is_reachable(c)) && (!membership || membership->contains(c)))
                                members.push_back(c);
                        }
                        exhausted = true;
                        for (u32 k = static_cast<u32>(free_slots.size()); k-- > 0;)
                        {
                            if (++odo[k] < lists[k].size())
                            {
                                exhausted = false;
                                break;
                            }
                            odo[k] = 0;
                        }
                    }
                }
            }
            insert_derived(q, derived, std::move(members), position);
        }
    }
}

FactLandmarkGraph Generator::run()
{
    m_member_set_identity = m_rr && O.first_achievers_restricted;
    if (O.include_positive_goal_facts)
    {
        std::vector<u32> args;
        for (const Literal& l : T.literals_of(T.goal))
        {
            if (!l.positive || T.predicates[l.pred.v].kind != PredKind::Fluent)
                continue;
            args.clear();
            for (Term t : T.terms_of(l))
                args.push_back(term_object(t).v);
            insert(l.pred.v, args, {}, std::nullopt);
        }
    }
    while (!m_work.empty())
    {
        const u32 position = m_work.front();
        m_work.pop_front();
        expand(position);
    }

    for (LiftedLandmark& r : m_records)
        r.initially_true = any_initially(r.members);
    std::vector<std::vector<CanonicalAtom>> disjunctive;
    for (const LiftedLandmark& r : m_records)
    {
        if (r.is_fact() || r.members.size() < 2)
            continue;
        if (O.max_disjunctive_members > 0 && r.members.size() > O.max_disjunctive_members)
            continue;
        disjunctive.push_back(r.members);
    }
    if (m_member_set_identity)
    {
        std::vector<std::vector<CanonicalAtom>> kept;
        for (const auto& m : disjunctive)
        {
            const bool strict_superset = std::any_of(disjunctive.begin(), disjunctive.end(), [&](const auto& other)
                                                     { return other.size() < m.size() && std::includes(m.begin(), m.end(), other.begin(), other.end()); });
            if (!strict_superset)
                kept.push_back(m);
        }
        disjunctive = std::move(kept);
    }

    // completion: candidates whose removal makes the goal unreachable become fact landmarks too
    if (m_rr && O.complete_fact_landmarks != CompleteFactLandmarks::Off && m_rr->goal_reachable())
    {
        std::vector<CanonicalAtom> facts = m_facts;
        std::sort(facts.begin(), facts.end());
        std::vector<CanonicalAtom> candidates, seen;
        auto consider = [&](CanonicalAtom c)
        {
            if (std::binary_search(facts.begin(), facts.end(), c) || initially(c))
                return;
            if (std::find(seen.begin(), seen.end(), c) != seen.end())
                return;
            seen.push_back(c);
            candidates.push_back(c);
        };
        if (O.complete_fact_landmarks == CompleteFactLandmarks::Members)
        {
            for (const LiftedLandmark& r : m_records)
                if (!r.is_fact())
                    for (CanonicalAtom c : r.members)
                        consider(c);
        }
        else
        {
            for (u32 p = 0; p < T.predicates.size(); ++p)
                if (T.predicates[p].kind == PredKind::Fluent)
                    for (CanonicalAtom c : m_rr->table().atoms(PredicateId{p}))
                        consider(c);
        }
        std::vector<u32> args(std::max<u32>(1, L.max_arity));
        for (CanonicalAtom c : candidates)
        {
            if (m_rr->goal_reachable_without(std::span<const CanonicalAtom>(&c, 1)))
                continue;
            m_facts.push_back(c);
            LiftedLandmark r;
            const u32 p = L.decode(c, args.data());
            r.predicate = PredicateId{p};
            r.binding.assign(args.begin(), args.begin() + L.arity[p]);
            r.members = {c};
            r.fact = c;
            m_records.push_back(std::move(r));
        }
    }
    return FactLandmarkGraph::create(m_facts, std::move(disjunctive), std::move(m_orderings), std::move(m_records));
}
}  // namespace

bool needs_relaxed_reachability(const LiftedFactLandmarkOptions& o)
{
    return o.reachability_filter_members || o.reachability_disambiguation != ReachabilityDisambiguation::Off || o.first_achievers_restricted ||
           o.verify_pi_plus || o.complete_fact_landmarks != CompleteFactLandmarks::Off;
}

FactLandmarkGraph lifted_fact_landmarks(const Task& task, const LiftedFactLandmarkOptions& options)
{
    if (!needs_relaxed_reachability(options))
        return Generator(task, nullptr, options).run();
    const auto rr = RelaxedReachability::create(task);
    return lifted_fact_landmarks(*rr, options);
}

FactLandmarkGraph lifted_fact_landmarks(const RelaxedReachability& engine, const LiftedFactLandmarkOptions& options)
{
    const bool use = needs_relaxed_reachability(options);
    FactLandmarkGraph g = Generator(engine.task(), use ? &engine : nullptr, options).run();
    if (options.verify_pi_plus)
        verify_pi_plus_fact_landmarks(engine, g);
    return g;
}

void verify_pi_plus_fact_landmarks(const Task& task, const FactLandmarkGraph& graph)
{
    const auto rr = RelaxedReachability::create(task);
    verify_pi_plus_fact_landmarks(*rr, graph);
}

void verify_pi_plus_fact_landmarks(const RelaxedReachability& engine, const FactLandmarkGraph& graph)
{
    if (!engine.goal_reachable())
        return;
    const Task& task = engine.task();
    const TaskData& T = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    std::vector<CanonicalAtom> skip;
    std::vector<u32> args;
    auto add = [&](u32 pred, std::span<const ObjectId> objs)
    {
        if (L.offset[pred] == CanonicalLayout::k_none)
            return;
        args.clear();
        for (ObjectId o : objs)
            args.push_back(o.v);
        const CanonicalAtom c = L.encode(pred, args.data());
        if (c < L.fluent_count)
            skip.push_back(c);
    };
    for (const GroundAtom& a : T.fluent_init)
        add(a.pred.v, T.objects_of(a));
    std::vector<ObjectId> objs;
    for (const Literal& l : T.literals_of(T.goal))
        if (l.positive && T.predicates[l.pred.v].kind == PredKind::Fluent)
        {
            objs.clear();
            for (Term t : T.terms_of(l))
                objs.push_back(term_object(t));
            add(l.pred.v, objs);
        }
    std::sort(skip.begin(), skip.end());
    for (CanonicalAtom c : graph.landmarks())
    {
        if (std::binary_search(skip.begin(), skip.end(), c))
            continue;
        if (engine.goal_reachable_without(std::span<const CanonicalAtom>(&c, 1)))
            throw std::logic_error("mymyr: lifted landmark extraction produced a fact landmark that is not one: the goal is still "
                                   "delete-relaxed reachable without " +
                                   format_atom(task, c));
    }
}
}  // namespace mymyr::landmarks
