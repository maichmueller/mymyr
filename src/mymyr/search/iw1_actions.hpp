#pragma once

#include "mymyr/task/task.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/layer_ordering.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace mymyr::search::detail
{
inline bool has_conditional_effects(const Task& task)
{
    for (const auto& e : task.data().conditional_effects)
        if (e.condition.literals.count || e.condition.constraints.count)
            return true;
    return false;
}

inline void check_iw1_actions(const Task& task, bool incremental, bool precheck, u32 width,
                            bool landmarks, bool deferred, const LayerOrdering& layers, u32 threads)
{
    if (!incremental && !precheck)
        return;
    if (width != 1)
        throw std::invalid_argument("incremental_first_applicability and precheck_add_effects require width 1");
    if (has_conditional_effects(task))
        throw std::invalid_argument("incremental_first_applicability and precheck_add_effects do not support conditional effects");
    if (landmarks || deferred)
        throw std::invalid_argument("incremental_first_applicability and precheck_add_effects require atom novelty without landmark novelty or transition ordering");
    if (incremental && (layers.kind != LayerOrdering::Kind::Queue || layers.beam() ||
                        layers.max_next_layer_states != ~u32{0} || threads != 1))
        throw std::invalid_argument("incremental_first_applicability requires queued layers and one thread");
}

/// Width-1 actions whose effects have not yet been novelty-tested. Once tested, every add feature is marked,
/// so another application cannot enter the tree. Blocked successors must not call tested().
class Iw1Actions
{
    struct Hash
    {
        using is_transparent = void;
        usize operator()(ActionLabel a) const noexcept
        {
            usize h = a.schema.v;
            for (ObjectId o : a.binding)
                h = h * 1000003 + o.v + 1;
            return h;
        }
        usize operator()(const Action& a) const noexcept { return (*this)(a.label()); }
    };
    struct Equal
    {
        using is_transparent = void;
        bool operator()(const Action& a, const Action& b) const { return a == b; }
        bool operator()(const Action& a, ActionLabel b) const
        {
            return a.schema == b.schema && std::equal(a.binding.begin(), a.binding.end(), b.binding.begin(), b.binding.end());
        }
        bool operator()(ActionLabel a, const Action& b) const { return (*this)(b, a); }
    };
    struct Trigger
    {
        u32 schema;
        const plan::Check* literal;
    };
    struct Matcher
    {
        mymyr::detail::ExecMatcher exec;
        std::vector<u64> domains;
    };

public:
    explicit Iw1Actions(Successors& succ) : m_succ(succ), m_task(succ.engine().task())
    {
        const auto& c = m_task.compiled();
        m_positive.resize(m_task.data().predicates.size());
        m_negative.resize(m_positive.size());
        m_matchers.resize(c.schemas.size());
        for (u32 s = 0; s < c.schemas.size(); ++s)
            for (const auto& l : c.schemas[s].pre_lits)
                if (l.pat.kind == plan::LitKind::Fluent)
                    (l.pos ? m_positive : m_negative)[l.pat.pred].push_back({s, &l});
    }

    [[nodiscard]] bool contains(u32 s, const ObjectId* b) const
    {
        return m_tested.contains(ActionLabel{SchemaId{s}, {b, m_succ.arity(s)}});
    }
    void tested(u32 s, const ObjectId* b)
    {
        m_tested.emplace(SchemaId{s}, std::vector<ObjectId>(b, b + m_succ.arity(s)));
    }
    [[nodiscard]] u64 bytes() const
    {
        auto storage = [](const auto& v) { return v.capacity() * sizeof(typename std::remove_cvref_t<decltype(v)>::value_type); };
        u64 n = m_tested.bucket_count() * sizeof(void*) + m_tested.size() * (sizeof(Action) + sizeof(void*));
        for (const Action& a : m_tested)
            n += storage(a.binding);
        n += storage(m_candidates) + storage(m_matchers) + storage(m_positive) + storage(m_negative);
        for (const Action& a : m_candidates)
            n += storage(a.binding);
        for (const auto& row : m_positive)
            n += storage(row);
        for (const auto& row : m_negative)
            n += storage(row);
        for (const auto& m : m_matchers)
            if (m)
            {
                const auto& x = m->exec;
                n += sizeof(Matcher) + storage(m->domains) + storage(x.unary) + storage(x.rows) + storage(x.fc_out) +
                     storage(x.fc_pre) + storage(x.steps) + storage(x.cand) + storage(x.dom1) + storage(x.fc_doms) + storage(x.bound);
            }
        return n;
    }

    /// Parent and child are the incoming tree edge. Null parent: root, fully enumerate. Derived and numeric
    /// applicability, witness selection, noncanonical order and blocked successors retain full enumeration;
    /// filtering tested actions still applies. An ordering callback must likewise receive its full action list.
    template<class Emit, class Interrupted>
    void run(StateView parent, StateView child, bool root, bool witness, bool canonical, bool blocked,
             Emit&& emit, Interrupted&& interrupted)
    {
        u64 visited = 0;
        auto check = [&] { return ++visited % k_check_transitions != 0 || !interrupted(); };
        if (root || witness || !canonical || blocked || m_task.has_axioms() || m_task.numeric_words())
        {
            m_succ.generate<true>(
                [&](u32 s, const ObjectId* b, const Delta& d)
                { return check() && (contains(s, b) || emit(s, b, d)); }, witness, canonical);
            return;
        }
        m_candidates.clear();
        bool stop = false;
        auto changed = [&](u64 slot, bool positive)
        {
            if (stop)
                return;
            const u32* atom = m_task.atoms().record(AtomKind::Fluent, static_cast<u32>(slot));
            for (const Trigger& t : (positive ? m_positive : m_negative)[atom[0]])
            {
                if (!check())
                {
                    stop = true;
                    return;
                }
                seed(t, atom + 1,
                     [&]()
                     {
                         if (!check())
                         {
                             stop = true;
                             return false;
                         }
                         const ObjectId* b = m_succ.engine().bind();
                         if (!contains(t.schema, b))
                             m_candidates.emplace_back(SchemaId{t.schema}, std::vector<ObjectId>(b, b + m_succ.arity(t.schema)));
                         return true;
                     });
                if (stop)
                    return;
            }
        };
        for (u32 i = 0; i < std::max(parent.nw, child.nw) && !stop; ++i)
        {
            const u64 p = i < parent.nw ? parent.w[i] : 0;
            const u64 c = i < child.nw ? child.w[i] : 0;
            for (u64 b = c & ~p; b; b &= b - 1)
                changed(u64{i} * 64 + bits::ctz64(b), true);
            for (u64 b = p & ~c; b; b &= b - 1)
                changed(u64{i} * 64 + bits::ctz64(b), false);
        }
        if (stop)
            return;
        std::sort(m_candidates.begin(), m_candidates.end());
        m_candidates.erase(std::unique(m_candidates.begin(), m_candidates.end()), m_candidates.end());
        for (const Action& a : m_candidates)
        {
            if (!check())
                return;
            if (contains(a.schema.v, a.binding.data()))
                continue;
            std::copy(a.binding.begin(), a.binding.end(), m_succ.engine().bind());
            bool ok;
            const Delta d = m_succ.effects(a.schema.v, &ok);
            if (!ok)
                continue;
            if (!emit(a.schema.v, a.binding.data(), d))
                return;
        }
    }

private:
    /// Restricts a private matcher's static domains to a literal's binding. The remaining constraints and both
    /// matcher execution modes stay intact, including repeated parameters, constants and parameter types.
    template<class Emit>
    void seed(const Trigger& t, const u32* args, Emit&& emit)
    {
        const u32 s = t.schema;
        const auto& plan = m_task.compiled().schemas[s].pre[1];
        const u32 ow = m_succ.engine().ow();
        if (!m_matchers[s])
        {
            auto m = std::make_unique<Matcher>();
            m_succ.engine().instantiate(m->exec, plan);
            m->domains.resize(usize{plan.total} * ow);
            for (auto& st : m->exec.steps)
                st.dom0 = m->domains.data() + usize{st.param} * ow;
            m_matchers[s] = std::move(m);
        }
        Matcher& m = *m_matchers[s];
        std::copy(plan.dom0.begin(), plan.dom0.end(), m.domains.begin());
        const auto range = t.literal->pat.terms;
        for (u32 i = 0; i < range.count; ++i)
        {
            const auto term = m_task.data().terms[range.begin + i];
            if (term < 0)
            {
                if (formalism::term_object(term).v != args[i])
                    return;
                continue;
            }
            u64* row = m.domains.data() + usize{static_cast<u32>(term)} * ow;
            if (!bits::test(row, ow, args[i]))
                return;
            std::fill_n(row, ow, u64{0});
            bits::set(row, args[i]);
        }
        m_succ.engine().clear_stop();
        m_succ.engine().run<true>(m.exec, emit);
        m_succ.engine().clear_stop();
    }

    Successors& m_succ;
    const Task& m_task;
    std::vector<std::vector<Trigger>> m_positive, m_negative;
    std::vector<std::unique_ptr<Matcher>> m_matchers;
    std::unordered_set<Action, Hash, Equal> m_tested;
    std::vector<Action> m_candidates;
};
}  // namespace mymyr::search::detail
