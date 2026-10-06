#pragma once
// Test support: the lifted suite (tests/data/tasks) with its expected BrFS counts, and an exhaustive
// oracle of the applicable actions and successors that shares no code with the compiled matchers: it enumerates every
// binding in parameter order over all objects and checks each literal of the normalized task directly.

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::test
{
struct SuiteTask
{
    std::string name;  // file stem, e.g. "gripper__prob05"
    u64 states;        // exhaustive BrFS state count (brfs_final_table.md)
    u64 generated;     // with witness pruning on (brfs_final_20260927.txt)
    u64 goal_states;
};

// Recorded from an exhaustive BrFS run against the fork.
inline const std::vector<SuiteTask>& suite()
{
    static const std::vector<SuiteTask> s = {
        {"gripper__prob05", 376832, 2031616, 2},
        {"blocks__probBLOCKS-8-0", 695417, 2094752, 1},
        {"logistics00__probLOGISTICS-6-1", 941192, 10487568, 8},
        {"miconic__s7-4", 229376, 3153920, 1792},
        {"visitall__visitall_x-6_y-3_r-100", 193456, 597411, 18},
        {"sokoban-opt08-strips__p14", 348955, 866027, 27},
        {"depot__p02", 40320, 376560, 9},
        {"driverlog__p03", 348750, 2826000, 40},
        {"rovers__p02", 198288, 1807920, 7776},
        {"zenotravel__p05", 1016064, 17224704, 784},
        {"transport-opt08-strips__p23", 1185759, 6993720, 81},
        {"freecell__p02", 159924, 1371273, 1},
        {"snake-opt18-strips__p05", 1320037, 2200339, 88669},
        {"parcprinter-opt11-strips__p03", 173942, 488868, 1},
        {"pegsol-08-strips__p22", 99457, 189967, 2},
        {"miconic-simpleadl__s10-2", 316680, 6333600, 20},
        {"caldera-split-opt18-adl__p04", 468041, 630395, 4472},
        {"pathways__p02", 669078, 11237052, 12288},
        {"folding-opt23-adl__p01", 148260, 170776, 1},
        {"openstacks-opt08-adl__p03", 97953, 548161, 6},
        {"philosophers__p03-phil4", 47938, 121820, 1},
        {"organic-synthesis-opt18-strips__p20", 41311, 52548, 24},
    };
    return s;
}

inline std::string task_path(const std::string& name)
{
    return std::string(MYMYR_TEST_DATA_DIR) + "/tasks/" + name + ".txt";
}

/// Full-suite tests are opt-in (MYMYR_TEST_FULL_SUITE=1); by default only tasks up to this many states run.
inline u64 suite_state_limit()
{
    const char* e = std::getenv("MYMYR_TEST_FULL_SUITE");
    return e && std::string(e) == "1" ? ~u64{0} : 100000;
}

// ------------------------------------------------------------------------------------------------------ oracle
using Atom = std::vector<u32>;  // [pred, args...]
using AtomSet = std::set<Atom>;

class Oracle
{
public:
    explicit Oracle(const formalism::TaskData& t) : T(t), n(t.num_objects())
    {
        statics.resize(T.predicates.size());
        for (const auto& a : T.static_init)
        {
            Atom x;
            for (ObjectId o : T.objects_of(a))
                x.push_back(o.v);
            statics[a.pred.v].insert(x);
        }
        // stratification of the axioms (rank of each derived predicate)
        rank.assign(T.predicates.size(), 0);
        for (bool changed = true; changed;)
        {
            changed = false;
            for (const auto& x : T.axioms)
                for (const auto& l : T.literals_of(x.body))
                    if (T.predicates[l.pred.v].kind == formalism::PredKind::Derived)
                    {
                        const u32 need = rank[l.pred.v] + (l.positive ? 0 : 1);
                        if (rank[x.head.pred.v] < need)
                        {
                            rank[x.head.pred.v] = need;
                            changed = true;
                        }
                    }
        }
        for (const auto& x : T.axioms)
            top = std::max(top, rank[x.head.pred.v]);
    }

    u64 budget = 50'000'000;  // binding nodes per call before giving up (std::runtime_error)

    AtomSet initial() const
    {
        AtomSet s;
        for (const auto& a : T.fluent_init)
        {
            Atom x{a.pred.v};
            for (ObjectId o : T.objects_of(a))
                x.push_back(o.v);
            s.insert(x);
        }
        return s;
    }

    AtomSet derive(const AtomSet& fl)
    {
        AtomSet der;
        if (T.axioms.empty())
            return der;
        for (u32 st = 0; st <= top; ++st)
            for (bool changed = true; changed;)
            {
                changed = false;
                for (const auto& x : T.axioms)
                {
                    if (rank[x.head.pred.v] != st)
                        continue;
                    std::vector<u32> bind(x.params.count, 0);
                    std::vector<Atom> heads;
                    enumerate(T.literals_of(x.body), 0, x.params.count, bind, fl, der,
                              [&] { heads.push_back(instantiate(x.head, bind)); });
                    for (auto& h : heads)
                        changed |= der.insert(h).second;
                }
            }
        return der;
    }

    bool is_goal(const AtomSet& fl, const AtomSet& der) const
    {
        std::vector<u32> none;
        for (const auto& l : T.literals_of(T.goal))
            if (!holds(l, none, fl, der))
                return false;
        return true;
    }

    /// Applicable actions in canonical order (schema, then binding).
    std::vector<Action> applicable(const AtomSet& fl, const AtomSet& der)
    {
        std::vector<Action> out;
        for (u32 s = 0; s < T.schemas.size(); ++s)
        {
            const auto& sc = T.schemas[s];
            std::vector<u32> bind(sc.arity(), 0);
            enumerate(T.literals_of(sc.precondition), 0, sc.arity(), bind, fl, der,
                      [&]
                      {
                          std::vector<ObjectId> b;
                          for (u32 v : bind)
                              b.push_back(ObjectId{v});
                          out.emplace_back(SchemaId{s}, std::move(b));
                      });
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    AtomSet successor(const AtomSet& fl, const AtomSet& der, const Action& a)
    {
        const auto& sc = T.schemas[a.schema.v];
        AtomSet adds, dels;
        for (const auto& ce : T.effects_of(sc))
        {
            const u32 total = sc.arity() + ce.extra_params.count;
            std::vector<u32> bind(total, 0);
            for (u32 i = 0; i < sc.arity(); ++i)
                bind[i] = a.binding[i].v;
            enumerate(T.literals_of(ce.condition), sc.arity(), total, bind, fl, der,
                      [&]
                      {
                          for (const auto& l : formalism::TaskData::slice(T.literals, ce.effects))
                              (l.positive ? adds : dels).insert(instantiate(l, bind));
                      });
        }
        AtomSet next = fl;
        for (const auto& d : dels)
            next.erase(d);
        for (const auto& x : adds)
            next.insert(x);
        return next;
    }

    /// Parameters of a schema that occur in some effect (condition or effect literal): witness pruning keeps one
    /// binding of the others.
    std::vector<bool> relevant(u32 schema) const
    {
        const auto& sc = T.schemas[schema];
        std::vector<bool> r(sc.arity(), false);
        for (const auto& ce : T.effects_of(sc))
            for (formalism::Range rg : {ce.condition.literals, ce.effects})
                for (const auto& l : formalism::TaskData::slice(T.literals, rg))
                    for (formalism::Term x : T.terms_of(l))
                        if (x >= 0 && static_cast<u32>(x) < sc.arity())
                            r[x] = true;
        return r;
    }

private:
    Atom instantiate(const formalism::Literal& l, const std::vector<u32>& bind) const
    {
        Atom a{l.pred.v};
        for (formalism::Term x : T.terms_of(l))
            a.push_back(x < 0 ? formalism::term_object(x).v : bind[x]);
        return a;
    }
    bool holds(const formalism::Literal& l, const std::vector<u32>& bind, const AtomSet& fl, const AtomSet& der) const
    {
        Atom a = instantiate(l, bind);
        bool t;
        switch (T.predicates[l.pred.v].kind)
        {
            case formalism::PredKind::Static: t = statics[l.pred.v].count(Atom(a.begin() + 1, a.end())) > 0; break;
            case formalism::PredKind::Fluent: t = fl.count(a) > 0; break;
            default: t = der.count(a) > 0; break;
        }
        return t == l.positive;
    }
    /// Every binding of parameters [first, total) (each over all objects) satisfying all literals. Parameters are bound
    /// in a static order that follows the literals (the next one shares the most literals with those already bound;
    /// ties by index), and each literal is checked as soon as all its parameters are bound.
    void enumerate(std::span<const formalism::Literal> lits, u32 first, u32 total, std::vector<u32>& bind, const AtomSet& fl,
                   const AtomSet& der, const std::function<void()>& f)
    {
        const u32 k = total - first;
        std::vector<std::vector<u32>> vars(lits.size());
        for (usize li = 0; li < lits.size(); ++li)
            for (formalism::Term x : T.terms_of(lits[li]))
                if (x >= 0 && static_cast<u32>(x) >= first &&
                    std::find(vars[li].begin(), vars[li].end(), static_cast<u32>(x)) == vars[li].end())
                    vars[li].push_back(static_cast<u32>(x));
        std::vector<u32> order;
        std::vector<bool> placed(total, false);
        for (u32 step = 0; step < k; ++step)
        {
            i64 best = -1, best_score = -1;
            for (u32 v = first; v < total; ++v)
            {
                if (placed[v])
                    continue;
                // positive literals only (negated ones, such as the many (not (= x y)), barely prune): a join with a
                // bound parameter weighs most, then a unary constraint
                i64 score = 0;
                for (usize li = 0; li < vars.size(); ++li)
                {
                    const auto& vs = vars[li];
                    if (!lits[li].positive || std::find(vs.begin(), vs.end(), v) == vs.end())
                        continue;
                    u32 bound = 0;
                    for (u32 u : vs)
                        bound += placed[u];
                    score += bound > 0 ? 1000 : vs.size() == 1 ? 1 : 0;
                }
                if (score > best_score)
                    best = v, best_score = score;
            }
            placed[best] = true;
            order.push_back(static_cast<u32>(best));
        }
        std::vector<u32> pos(total, 0);
        for (u32 d = 0; d < k; ++d)
            pos[order[d]] = d + 1;
        // at[d]: literals whose parameters are all bound after d binding steps (d = 0: no free parameter)
        std::vector<std::vector<const formalism::Literal*>> at(k + 1);
        for (usize li = 0; li < lits.size(); ++li)
        {
            u32 last = 0;
            for (u32 v : vars[li])
                last = std::max(last, pos[v]);
            at[last].push_back(&lits[li]);
        }
        // Candidates of the parameter bound at step d: the values that the true atoms of a positive literal allow
        // once the literal's other parameters are bound (a naive join over the atom sets); all objects otherwise.
        std::vector<std::vector<usize>> sources(k);
        for (usize li = 0; li < lits.size(); ++li)
        {
            if (!lits[li].positive || vars[li].empty())
                continue;
            u32 last = 0;
            for (u32 v : vars[li])
                last = std::max(last, pos[v]);
            sources[last - 1].push_back(li);  // the literal's last parameter is bound at step last - 1
        }
        auto candidates = [&](u32 d, std::vector<u32>& out)
        {
            const u32 v = order[d];
            bool restricted = false;
            for (usize li : sources[d])
            {
                const formalism::Literal& l = lits[li];
                const auto terms = T.terms_of(l);
                std::set<u32> vals;
                auto consider = [&](const u32* args)
                {
                    u32 val = ~u32{0};
                    for (usize i = 0; i < terms.size(); ++i)
                    {
                        const formalism::Term x = terms[i];
                        if (x < 0)
                        {
                            if (args[i] != formalism::term_object(x).v)
                                return;
                        }
                        else if (static_cast<u32>(x) == v)
                        {
                            if (val != ~u32{0} && val != args[i])
                                return;
                            val = args[i];
                        }
                        else if (args[i] != bind[x])
                            return;
                    }
                    vals.insert(val);
                };
                switch (T.predicates[l.pred.v].kind)
                {
                    case formalism::PredKind::Static:
                        for (const Atom& a : statics[l.pred.v])
                            consider(a.data());
                        break;
                    default:
                    {
                        const AtomSet& src = T.predicates[l.pred.v].kind == formalism::PredKind::Fluent ? fl : der;
                        for (auto it = src.lower_bound(Atom{l.pred.v}); it != src.end() && (*it)[0] == l.pred.v; ++it)
                            consider(it->data() + 1);
                    }
                }
                if (!restricted)
                    out.assign(vals.begin(), vals.end());
                else
                {
                    std::vector<u32> keep;
                    std::set_intersection(out.begin(), out.end(), vals.begin(), vals.end(), std::back_inserter(keep));
                    out.swap(keep);
                }
                restricted = true;
            }
            if (!restricted)
            {
                out.resize(n);
                for (u32 o = 0; o < n; ++o)
                    out[o] = o;
            }
        };
        u64 nodes = 0;
        std::function<void(u32)> rec = [&](u32 d)
        {
            if (++nodes > budget)
                throw std::runtime_error("oracle budget exceeded");
            for (const auto* l : at[d])
                if (!holds(*l, bind, fl, der))
                    return;
            if (d == k)
            {
                f();
                return;
            }
            std::vector<u32> cand;
            candidates(d, cand);
            for (u32 o : cand)
            {
                bind[order[d]] = o;
                rec(d + 1);
            }
        };
        rec(0);
    }

    const formalism::TaskData& T;
    u32 n;
    std::vector<std::set<Atom>> statics;
    std::vector<u32> rank;
    u32 top = 0;
};

/// The fluent atoms of a mymyr state as oracle atoms.
inline AtomSet atoms_of(const Task& task, StateView s)
{
    AtomSet out;
    bits::for_each(s.w, s.nw,
                   [&](u64 slot)
                   {
                       const u32* r = task.atoms().record(AtomKind::Fluent, static_cast<u32>(slot));
                       Atom a{r[0]};
                       for (u32 i = 0; i < task.compiled().arity[r[0]]; ++i)
                           a.push_back(r[1 + i]);
                       out.insert(a);
                   });
    return out;
}
}  // namespace mymyr::test
