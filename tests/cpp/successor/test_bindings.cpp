// Binding generators (successor/bindings.hpp).
//
// Walk oracle: on the fork's golden walks (tests/data/expected, replayed on the text exports of tests/data/tasks and
// tests/data/tasks_iw by the names of the actions taken), for every applicable action of every walk state, enumerating
// its schema with the action's own parameters partially fixed (every subset of at most two parameters, the empty and
// the full one) returns exactly the applicable actions that agree with the partial binding; a parameter fixed to an
// object that no applicable action has there gives none; resumed enumerations (two bindings per call) give the same
// sequence as one call; each schema's precondition as a condition with an action's binding fixed agrees with
// is_applicable (applicable actions and random bindings); the goal as a condition has a binding iff is_goal.
//
// Random conditions: on random walks of the BrFS suite and of the numeric tasks, random conditions (literals over every
// predicate kind with variables and objects, either polarity, equalities, declared types, numeric constraints) have
// exactly the bindings of a brute-force evaluation over all object tuples that shares no code with the compiled
// matchers (oracle atom sets, the axioms closed by test::Oracle, numeric values by slot), with and without partial
// bindings; ground conjunctions instantiate the literals. Concurrent enumeration over one shared task from 8 threads
// gives the single-threaded results.

#include "../support/json.hpp"
#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/successor/bindings.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;
using namespace mymyr::test;

namespace
{
using Partial = std::vector<std::optional<ObjectId>>;
using Tuple = std::vector<ObjectId>;

#if defined(MYMYR_SANITIZED)
constexpr bool k_sanitized = true;
#else
constexpr bool k_sanitized = false;
#endif

fs::path data_dir() { return fs::path(MYMYR_TEST_DATA_DIR); }

/// The text export of a golden task: tests/data/tasks/<name>.txt or tests/data/tasks_iw/<tag>__<name>.txt.
std::optional<fs::path> text_of(const std::string& name)
{
    const fs::path a = data_dir() / "tasks" / (name + ".txt");
    if (fs::exists(a))
        return a;
    for (const auto& e : fs::directory_iterator(data_dir() / "tasks_iw"))
    {
        const std::string f = e.path().filename().string();
        const std::string suffix = "__" + name + ".txt";
        if (f.size() > suffix.size() && f.compare(f.size() - suffix.size(), suffix.size(), suffix) == 0)
            return e.path();
    }
    return std::nullopt;
}

std::vector<std::string> golden_names()
{
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(data_dir() / "expected"))
        if (e.path().extension() == ".json")
            out.push_back(e.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

/// An action by the golden file's object names (the text exports name their objects o0, o1, ...).
std::string golden_format(const Task& task, const ActionLabel& a, const test::json::Value& names)
{
    std::string s = "(" + task.schema_name(a.schema);
    for (ObjectId o : a.binding)
        s += " " + names["objects"][o.v].str;
    return s + ")";
}

/// The states of the golden walks, replayed by the names of the actions taken.
std::vector<State> walk_states(const Task& task, const test::json::Value& doc)
{
    Successors& succ = task.workspace().successors();
    const auto& names = doc["names"];
    std::vector<State> out;
    for (const auto& w : doc["walks"]["walks"].arr)
    {
        State s = task.initial_state();
        for (const auto& st : w["steps"].arr)
        {
            out.push_back(s);
            const auto& taken = st["taken"];
            if (taken.is_null())
                break;
            std::optional<Action> next;
            succ.set_witness_pruning(false);
            succ.for_each_applicable(s,
                                     [&](const ActionLabel& a, const Delta&) -> bool
                                     {
                                         if (golden_format(task, a, names) != taken.str)
                                             return true;
                                         next = Action(a);
                                         return false;
                                     });
            if (!next)
            {
                ADD_FAILURE() << "walk replay: " << taken.str << " not applicable";
                return out;
            }
            s = succ.apply(s, next->label());
        }
    }
    return out;
}

std::vector<Action> applicable(const Task& task, StateView s)
{
    Successors& succ = task.workspace().successors();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);
    return succ.applicable_actions(s);
}

bool agrees(const Tuple& b, const Partial& p)
{
    for (usize i = 0; i < p.size(); ++i)
        if (p[i] && b[i] != *p[i])
            return false;
    return true;
}

std::vector<Tuple> sorted(std::vector<Tuple> v)
{
    std::sort(v.begin(), v.end(), [](const Tuple& a, const Tuple& b)
              { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), [](ObjectId x, ObjectId y) { return x.v < y.v; }); });
    return v;
}

std::string show(const Task& task, const Tuple& b)
{
    std::string s = "(";
    for (usize i = 0; i < b.size(); ++i)
        s += (i ? " " : "") + task.object_name(b[i]);
    return s + ")";
}

/// The enumeration in pieces of `step` bindings, each call resuming after the last binding of the previous one.
std::vector<Tuple> resumed(const Task& task, Workspace& ws, SchemaId schema, StateView s, const Partial& p, u64 step)
{
    std::vector<Tuple> out;
    for (;;)
    {
        BindingOptions o;
        o.limit = step;
        if (!out.empty())
            o.resume_after = out.back();
        const u64 n = for_each_binding(task, ws, schema, s, p, [&](std::span<const ObjectId> b) { out.emplace_back(b.begin(), b.end()); }, o);
        if (n < step)
            return out;
    }
}

// ------------------------------------------------------------------------------------------------ walk oracle

class BindingsWalk : public ::testing::TestWithParam<std::string>
{
};

TEST_P(BindingsWalk, SchemaBindingsAreTheApplicableActionsThatAgree)
{
    const std::string name = GetParam();
    const auto txt = text_of(name);
    if (!txt)
        GTEST_SKIP() << "no text export of " << name;
    const auto doc = test::json::parse_file((data_dir() / "expected" / (name + ".json")).string());
    const auto task = Task::from_text_file(txt->string());
    Workspace& ws = task->workspace();
    std::vector<State> states = walk_states(*task, doc);
    if (k_sanitized && states.size() > 12)
        states.resize(12);
    const u32 S = task->num_schemas();
    std::vector<ConjunctiveCondition> pre;
    for (u32 sc = 0; sc < S; ++sc)
        pre.push_back(ConjunctiveCondition::precondition(*task, SchemaId{sc}));
    const ConjunctiveCondition goal = ConjunctiveCondition::goal(*task);
    std::mt19937_64 rng(7);
    u64 cases = 0, empty_cases = 0, pre_checks = 0, resumes = 0;
    int shown = 0;
    auto fail = [&](const std::string& what)
    {
        if (shown++ < 6)
            ADD_FAILURE() << name << ": " << what;
    };
    for (usize si = 0; si < states.size(); ++si)
    {
        const State& s = states[si];
        const std::vector<Action> acts = applicable(*task, s.view());
        std::vector<std::vector<Tuple>> by_schema(S);
        for (const Action& a : acts)
            by_schema[a.schema.v].push_back(a.binding);
        // (schema, partial) cases from the applicable actions' own parameters
        std::set<std::pair<u32, std::vector<i64>>> seen;
        auto run_case = [&](u32 sc, const Partial& p, bool expect_empty)
        {
            std::vector<i64> key;
            for (const auto& o : p)
                key.push_back(o ? static_cast<i64>(o->v) : -1);
            if (!seen.insert({sc, key}).second)
                return;
            ++cases;
            empty_cases += expect_empty;
            const std::vector<Tuple> got = sorted(bindings(*task, ws, SchemaId{sc}, s.view(), p));
            std::vector<Tuple> want;
            for (const Tuple& b : by_schema[sc])
                if (agrees(b, p))
                    want.push_back(b);
            if (got != want)
                fail("state " + std::to_string(si) + " schema " + task->schema_name(SchemaId{sc}) + ": " + std::to_string(got.size()) +
                     " bindings, expected " + std::to_string(want.size()) + (got.empty() ? "" : ", first " + show(*task, got[0])));
            if (expect_empty && !want.empty())
                fail("an empty case has applicable actions");
        };
        for (u32 sc = 0; sc < S; ++sc)
            run_case(sc, Partial(task->compiled().schemas[sc].arity), false);
        for (const Action& a : acts)
        {
            const u32 k = static_cast<u32>(a.binding.size());
            Partial full(k);
            for (u32 i = 0; i < k; ++i)
                full[i] = a.binding[i];
            run_case(a.schema.v, full, false);
            for (u32 i = 0; i < k; ++i)
            {
                Partial p(k);
                p[i] = a.binding[i];
                run_case(a.schema.v, p, false);
                for (u32 j = i + 1; j < k; ++j)
                {
                    Partial q = p;
                    q[j] = a.binding[j];
                    run_case(a.schema.v, q, false);
                }
            }
            // the precondition as a condition, the action's binding fixed
            ++pre_checks;
            if (count_bindings(*task, ws, pre[a.schema.v], s.view(), full) != 1)
                fail("the precondition of an applicable action " + task->format(a.label()) + " does not hold");
        }
        // a parameter fixed to an object no applicable action has there: no binding
        for (u32 sc = 0; sc < S; ++sc)
        {
            const u32 k = task->compiled().schemas[sc].arity;
            for (u32 i = 0; i < k; ++i)
            {
                std::vector<u8> used(task->num_objects(), 0);
                for (const Tuple& b : by_schema[sc])
                    used[b[i].v] = 1;
                for (u32 o = 0; o < task->num_objects(); ++o)
                    if (!used[o])
                    {
                        Partial p(k);
                        p[i] = ObjectId{o};
                        run_case(sc, p, true);
                        break;
                    }
            }
        }
        // random full bindings: the precondition holds iff the action is applicable (classical schemas)
        Successors& succ = ws.successors();
        for (u32 sc = 0; sc < S; ++sc)
        {
            if (task->compiled().schemas[sc].numeric())
                continue;
            const u32 k = task->compiled().schemas[sc].arity;
            for (int r = 0; r < 4; ++r)
            {
                Tuple b(k, ObjectId{0});
                if (r == 0 && !by_schema[sc].empty())
                    b = by_schema[sc][rng() % by_schema[sc].size()];
                for (u32 i = 0; i < k; ++i)
                    if (r > 0 && (r == 3 || rng() % 2))
                        b[i] = ObjectId{static_cast<u32>(rng() % task->num_objects())};
                Partial full(k);
                for (u32 i = 0; i < k; ++i)
                    full[i] = b[i];
                ++pre_checks;
                const bool holds = count_bindings(*task, ws, pre[sc], s.view(), full) == 1;
                if (holds != succ.is_applicable(s.view(), ActionLabel{SchemaId{sc}, b}))
                    fail("precondition and is_applicable disagree on " + task->schema_name(SchemaId{sc}) + show(*task, b));
            }
        }
        // the goal
        if ((count_bindings(*task, ws, goal, s.view()) == 1) != task->is_goal(s.view()))
            fail("state " + std::to_string(si) + ": the goal condition disagrees with is_goal");
        // resumed enumerations equal one enumeration, in the same order
        for (u32 sc = 0; sc < S; ++sc)
        {
            const Partial none(task->compiled().schemas[sc].arity);
            std::vector<Tuple> one;
            for_each_binding(*task, ws, SchemaId{sc}, s.view(), none, [&](std::span<const ObjectId> b) { one.emplace_back(b.begin(), b.end()); });
            if (one.size() < 2)
                continue;
            ++resumes;
            if (resumed(*task, ws, SchemaId{sc}, s.view(), none, 2) != one)
                fail("a resumed enumeration differs from one enumeration (" + task->schema_name(SchemaId{sc}) + ")");
        }
    }
    std::printf("BINDINGS %-40s states %3zu cases %7llu (empty %6llu) precondition checks %6llu resumed %4llu\n", name.c_str(),
                states.size(), static_cast<unsigned long long>(cases), static_cast<unsigned long long>(empty_cases),
                static_cast<unsigned long long>(pre_checks), static_cast<unsigned long long>(resumes));
}

INSTANTIATE_TEST_SUITE_P(Golden, BindingsWalk, ::testing::ValuesIn(golden_names()),
                         [](const auto& info)
                         {
                             std::string s = info.param;
                             for (char& c : s)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return s;
                         });

// ------------------------------------------------------------------------------------------------ random conditions

/// Brute-force evaluation of a condition: every literal, equality, type and numeric constraint checked directly.
class BruteForce
{
public:
    BruteForce(const Task& task, Oracle& oracle) : m_task(task), T(task.data()), m_oracle(oracle)
    {
        // objects of each type (reflexive-transitive supertypes of each object's declared types)
        const u32 nt = static_cast<u32>(T.types.size());
        m_of_type.assign(nt, std::vector<u8>(T.num_objects(), 0));
        for (u32 o = 0; o < T.num_objects(); ++o)
        {
            std::vector<u32> stack;
            for (TypeId t : formalism::TaskData::slice(T.type_ids, T.objects[o].types))
                stack.push_back(t.v);
            while (!stack.empty())
            {
                const u32 t = stack.back();
                stack.pop_back();
                if (m_of_type[t][o])
                    continue;
                m_of_type[t][o] = 1;
                for (TypeId b : formalism::TaskData::slice(T.type_ids, T.types[t].bases))
                    stack.push_back(b.v);
            }
        }
        for (const auto& a : T.static_init)
        {
            Atom x{a.pred.v};
            for (ObjectId o : T.objects_of(a))
                x.push_back(o.v);
            m_static.insert(x);
        }
        for (const auto& v : T.static_values)
        {
            std::vector<u32> k{v.func.v};
            for (ObjectId o : formalism::TaskData::slice(T.object_ids, v.objects))
                k.push_back(o.v);
            m_static_values[k] = v.value;
        }
        const plan::Numeric& N = task.numeric();
        for (u32 s = 0; s < N.slots; ++s)
        {
            std::vector<u32> k{N.slot_function[s]};
            for (u32 j = N.slot_args_begin[s]; j < N.slot_args_begin[s + 1]; ++j)
                k.push_back(N.slot_args[j].v);
            m_slot[k] = s;
        }
    }

    void set_state(StateView s)
    {
        m_state = s;
        m_fl = atoms_of(m_task, s);
        m_der = m_oracle.derive(m_fl);
    }

    /// The true static, fluent and derived atoms of the state.
    [[nodiscard]] std::vector<test::Atom> truths() const
    {
        std::vector<test::Atom> out(m_static.begin(), m_static.end());
        out.insert(out.end(), m_fl.begin(), m_fl.end());
        out.insert(out.end(), m_der.begin(), m_der.end());
        return out;
    }

    std::vector<Tuple> solve(const ConjunctiveCondition& c) const
    {
        std::vector<Tuple> out;
        const u32 V = c.arity(), n = T.num_objects();
        Tuple b(V, ObjectId{0});
        for (;;)
        {
            if (holds(c, b))
                out.push_back(b);
            u32 i = V;
            while (i > 0 && ++b[i - 1].v == n)
                b[--i].v = 0;
            if (i == 0)
                return out;
        }
    }

private:
    using Atom = test::Atom;

    u32 obj(formalism::Term t, const Tuple& b) const { return formalism::is_object(t) ? formalism::term_object(t).v : b[t].v; }

    bool holds(const ConjunctiveCondition& c, const Tuple& b) const
    {
        for (u32 v = 0; v < c.arity(); ++v)
        {
            bool ok = c.variables[v].types.empty();
            for (TypeId t : c.variables[v].types)
                ok = ok || m_of_type[t.v][b[v].v];
            if (!ok)
                return false;
        }
        for (const auto& l : c.literals)
        {
            Atom a{l.predicate.v};
            for (formalism::Term t : l.terms)
                a.push_back(obj(t, b));
            bool t;
            switch (T.predicates[l.predicate.v].kind)
            {
                case formalism::PredKind::Static: t = m_static.count(a) > 0; break;
                case formalism::PredKind::Fluent: t = m_fl.count(a) > 0; break;
                default: t = m_der.count(a) > 0; break;
            }
            if (t != l.positive)
                return false;
        }
        for (const auto& e : c.equalities)
            if ((obj(e.lhs, b) == obj(e.rhs, b)) != e.positive)
                return false;
        for (const auto& k : c.constraints)
        {
            const double l = value(c, k.lhs, b), r = value(c, k.rhs, b);
            if (std::isnan(l) || std::isnan(r))
                return false;
            bool t = false;
            switch (k.cmp)
            {
                case formalism::Comparator::Eq: t = l == r; break;
                case formalism::Comparator::Ne: t = l != r; break;
                case formalism::Comparator::Lt: t = l < r; break;
                case formalism::Comparator::Le: t = l <= r; break;
                case formalism::Comparator::Gt: t = l > r; break;
                case formalism::Comparator::Ge: t = l >= r; break;
            }
            if (!t)
                return false;
        }
        return true;
    }

    double value(const ConjunctiveCondition& c, u32 e, const Tuple& b) const
    {
        const formalism::Expr& x = c.exprs[e];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (x.op)
        {
            case formalism::ExprOp::Number: return x.value;
            case formalism::ExprOp::Function:
            {
                std::vector<u32> k{x.func.v};
                for (u32 j = x.terms.begin; j < x.terms.end(); ++j)
                    k.push_back(obj(c.expr_terms[j], b));
                if (T.functions[x.func.v].kind == formalism::FuncKind::Static)
                {
                    const auto it = m_static_values.find(k);
                    return it == m_static_values.end() ? nan : it->second;
                }
                const auto it = m_slot.find(k);
                return it == m_slot.end() ? nan : m_task.numeric_value(m_state, it->second);
            }
            case formalism::ExprOp::Add: return value(c, x.a, b) + value(c, x.b, b);
            case formalism::ExprOp::Sub: return value(c, x.a, b) - value(c, x.b, b);
            case formalism::ExprOp::Mul: return value(c, x.a, b) * value(c, x.b, b);
            case formalism::ExprOp::Div:
            {
                const double d = value(c, x.b, b);
                return d == 0 ? nan : value(c, x.a, b) / d;
            }
            case formalism::ExprOp::Neg: return -value(c, x.a, b);
        }
        return nan;
    }

    const Task& m_task;
    const formalism::TaskData& T;
    Oracle& m_oracle;
    StateView m_state;
    std::vector<std::vector<u8>> m_of_type;
    std::set<Atom> m_static;
    std::map<std::vector<u32>, double> m_static_values;
    std::map<std::vector<u32>, u32> m_slot;
    AtomSet m_fl, m_der;
};

/// A random condition with up to `max_vars` variables over the task's predicates, types and functions. With `truths`
/// (true atoms [pred, args...]), half of the literals are positive and lift a true atom (each argument becomes a
/// variable or stays), so that more conditions have bindings.
ConjunctiveCondition random_condition(const Task& task, std::mt19937_64& rng, u32 max_vars,
                                      const std::vector<test::Atom>* truths = nullptr)
{
    const formalism::TaskData& T = task.data();
    const u32 n = T.num_objects();
    ConjunctiveCondition c;
    const u32 V = static_cast<u32>(rng() % (max_vars + 1));
    for (u32 v = 0; v < V; ++v)
    {
        ConjunctiveCondition::Variable x;
        x.name = "v" + std::to_string(v);
        if (!T.types.empty() && rng() % 5 == 0)
            x.types.push_back(TypeId{static_cast<u32>(rng() % T.types.size())});
        c.variables.push_back(x);
    }
    auto term = [&]() -> formalism::Term
    {
        if (V > 0 && rng() % 5 != 0)
            return static_cast<formalism::Term>(rng() % V);
        return formalism::object_term(ObjectId{static_cast<u32>(rng() % n)});
    };
    std::vector<u32> preds;
    for (u32 p = 0; p < T.predicates.size(); ++p)
        if (T.predicates[p].arity <= 3 && (n > 0 || T.predicates[p].arity == 0))
            preds.push_back(p);
    const u32 L = 1 + static_cast<u32>(rng() % 4);
    for (u32 i = 0; i < L && !preds.empty(); ++i)
    {
        ConjunctiveCondition::Literal l;
        if (truths && !truths->empty() && rng() % 2 == 0)
        {
            const test::Atom& a = (*truths)[rng() % truths->size()];
            l.predicate = PredicateId{a[0]};
            for (usize j = 1; j < a.size(); ++j)
                l.terms.push_back(V > 0 && rng() % 3 != 0 ? static_cast<formalism::Term>(rng() % V) : formalism::object_term(ObjectId{a[j]}));
            c.literals.push_back(std::move(l));
            continue;
        }
        l.predicate = PredicateId{preds[rng() % preds.size()]};
        l.positive = rng() % 10 < 7;
        for (u32 j = 0; j < T.predicates[l.predicate.v].arity; ++j)
            l.terms.push_back(term());
        c.literals.push_back(std::move(l));
    }
    if (V > 0 && rng() % 3 == 0)
        c.equalities.push_back({term(), term(), rng() % 2 == 0});
    // numeric constraints: (cmp (f t...) k) or (cmp (+ (f t...) (g t...)) k)
    std::vector<u32> funcs;
    for (u32 f = 0; f < T.functions.size(); ++f)
        if (T.functions[f].kind != formalism::FuncKind::Auxiliary && (n > 0 || T.functions[f].arity == 0))
            funcs.push_back(f);
    if (!funcs.empty() && rng() % 2 == 0)
    {
        auto fn = [&]() -> u32
        {
            formalism::Expr x;
            x.op = formalism::ExprOp::Function;
            x.func = FunctionId{funcs[rng() % funcs.size()]};
            x.terms = {static_cast<u32>(c.expr_terms.size()), T.functions[x.func.v].arity};
            for (u32 j = 0; j < T.functions[x.func.v].arity; ++j)
                c.expr_terms.push_back(term());
            c.exprs.push_back(x);
            return static_cast<u32>(c.exprs.size() - 1);
        };
        u32 lhs = fn();
        if (rng() % 3 == 0)
        {
            const u32 r = fn();
            formalism::Expr add;
            add.op = rng() % 2 ? formalism::ExprOp::Add : formalism::ExprOp::Sub;
            add.a = lhs;
            add.b = r;
            c.exprs.push_back(add);
            lhs = static_cast<u32>(c.exprs.size() - 1);
        }
        formalism::Expr k;
        k.op = formalism::ExprOp::Number;
        k.value = static_cast<double>(static_cast<int>(rng() % 9) - 2);
        c.exprs.push_back(k);
        formalism::NumericConstraint nc;
        nc.cmp = static_cast<formalism::Comparator>(rng() % 6);
        nc.lhs = lhs;
        nc.rhs = static_cast<u32>(c.exprs.size() - 1);
        c.constraints.push_back(nc);
    }
    return c;
}

/// Random walk states of a task (successors chosen uniformly).
std::vector<State> random_states(const Task& task, u32 walks, u32 steps, u64 seed)
{
    std::mt19937_64 rng(seed);
    std::vector<State> out;
    for (u32 w = 0; w < walks; ++w)
    {
        State s = task.initial_state();
        for (u32 i = 0; i <= steps; ++i)
        {
            out.push_back(s);
            const std::vector<Action> acts = applicable(task, s.view());
            if (acts.empty())
                break;
            s = task.workspace().successors().apply(s, acts[rng() % acts.size()].label());
        }
    }
    return out;
}

std::vector<std::string> random_tasks()
{
    std::vector<std::string> out;
    for (const auto& t : suite())
        out.push_back("tasks/" + t.name);
    for (const auto& e : fs::directory_iterator(data_dir() / "numeric_tasks"))
        if (e.path().extension() == ".txt")
            out.push_back("numeric_tasks/" + e.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

class BindingsRandom : public ::testing::TestWithParam<std::string>
{
};

TEST_P(BindingsRandom, ConditionsMatchBruteForce)
{
    const auto task = Task::from_text_file((data_dir() / (GetParam() + ".txt")).string());
    Workspace& ws = task->workspace();
    Oracle oracle(task->data());
    BruteForce brute(*task, oracle);
    const u32 n = task->num_objects();
    // variables such that n^V tuples stay small
    u32 max_vars = 0;
    for (u64 p = n; n > 0 && max_vars < 3 && p <= 30000; p *= n)
        ++max_vars;
    std::mt19937_64 rng(11);
    const std::vector<State> states = random_states(*task, 2, k_sanitized ? 3 : 8, 5);
    u64 conditions = 0, nonempty = 0, partials = 0;
    int shown = 0;
    for (const State& s : states)
    {
        brute.set_state(s.view());
        const std::vector<test::Atom> truths = brute.truths();
        for (int r = 0; r < (k_sanitized ? 10 : 40); ++r)
        {
            const ConjunctiveCondition c = random_condition(*task, rng, max_vars, r % 2 ? &truths : nullptr);
            ASSERT_NO_THROW(c.validate(*task)) << c.str(*task);
            const std::vector<Tuple> want = brute.solve(c);
            const std::vector<Tuple> got = sorted(bindings(*task, ws, c, s.view()));
            ++conditions;
            nonempty += !want.empty();
            if (got != want && shown++ < 5)
                ADD_FAILURE() << GetParam() << ": " << c.str(*task) << ": " << got.size() << " bindings, brute force "
                              << want.size();
            // partial bindings: fix a random subset to the values of a solution, or to random objects
            for (int k = 0; k < 3 && c.arity() > 0; ++k)
            {
                Partial p(c.arity());
                for (u32 v = 0; v < c.arity(); ++v)
                    if (rng() % 2)
                        p[v] = !want.empty() && k < 2 ? want[rng() % want.size()][v] : ObjectId{static_cast<u32>(rng() % n)};
                std::vector<Tuple> wp;
                for (const Tuple& b : want)
                    if (agrees(b, p))
                        wp.push_back(b);
                ++partials;
                if (sorted(bindings(*task, ws, c, s.view(), p)) != wp && shown++ < 5)
                    ADD_FAILURE() << GetParam() << ": " << c.str(*task) << " with a partial binding";
            }
            // ground conjunctions instantiate the literals
            for_each_ground_conjunction(*task, ws, c, s.view(), {},
                                        [&](const GroundConjunction& g)
                                        {
                                            usize at[3] = {0, 0, 0};
                                            const std::span<const GroundLiteral> by[3] = {g.static_literals, g.fluent_literals,
                                                                                          g.derived_literals};
                                            for (const auto& l : c.literals)
                                            {
                                                const u32 k = static_cast<u32>(task->data().predicates[l.predicate.v].kind);
                                                const GroundLiteral& x = by[k][at[k]++];
                                                EXPECT_EQ(x.predicate, l.predicate);
                                                EXPECT_EQ(x.positive, l.positive);
                                                for (usize j = 0; j < l.terms.size(); ++j)
                                                    EXPECT_EQ(x.objects[j], formalism::is_object(l.terms[j]) ? formalism::term_object(l.terms[j])
                                                                                                           : g.binding[l.terms[j]]);
                                            }
                                            return true;
                                        }, BindingOptions{3, {}});
        }
    }
    std::printf("RANDOM %-40s states %3zu conditions %5llu (nonempty %5llu) partial %5llu\n", GetParam().c_str(), states.size(),
                static_cast<unsigned long long>(conditions), static_cast<unsigned long long>(nonempty),
                static_cast<unsigned long long>(partials));
}

INSTANTIATE_TEST_SUITE_P(Tasks, BindingsRandom, ::testing::ValuesIn(random_tasks()),
                         [](const auto& info)
                         {
                             std::string s = info.param;
                             for (char& c : s)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return s;
                         });

// ------------------------------------------------------------------------------------------------ hand-made conditions

TaskPtr suite_task(const std::string& name) { return Task::from_text_file((data_dir() / "tasks" / (name + ".txt")).string()); }

u32 pred(const Task& t, std::string_view name)
{
    for (u32 p = 0; p < t.data().predicates.size(); ++p)
        if (t.data().str(t.data().predicates[p].name) == name)
            return p;
    ADD_FAILURE() << "no predicate " << name;
    return 0;
}

/// An object of a suite task by the golden file's name (the text exports name their objects o0, o1, ...).
ObjectId object(const std::string& task, std::string_view name)
{
    const auto doc = test::json::parse_file((data_dir() / "expected" / (task + ".json")).string());
    const auto& objs = doc["names"]["objects"].arr;
    for (u32 o = 0; o < objs.size(); ++o)
        if (objs[o].str == name)
            return ObjectId{o};
    ADD_FAILURE() << "no object " << name;
    return ObjectId{0};
}

TEST(Bindings, ConstantsNegationEqualityAndStaticPredicates)
{
    // gripper: (at ?b rooma) (not (= ?b ball1)), (ball ?b) static, (free ?g) fluent negated
    const auto task = suite_task("gripper__prob05");
    Workspace& ws = task->workspace();
    const State s0 = task->initial_state();
    ConjunctiveCondition c;
    c.variables = {{"b", {}}};
    c.literals.push_back({PredicateId{pred(*task, "at")}, true, {0, formalism::object_term(object("gripper__prob05", "rooma"))}});
    c.literals.push_back({PredicateId{pred(*task, "ball")}, true, {0}});
    c.equalities.push_back({0, formalism::object_term(object("gripper__prob05", "ball1")), false});
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view()), 11u);  // the twelve balls start in rooma
    c.literals.push_back({PredicateId{pred(*task, "at")}, false, {0, formalism::object_term(object("gripper__prob05", "roomb"))}});
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view()), 11u);
    // a fixed variable that violates the static literal: no binding, no error
    const std::optional<ObjectId> rooma[] = {object("gripper__prob05", "rooma")};
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view(), rooma), 0u);
    const std::optional<ObjectId> ball1[] = {object("gripper__prob05", "ball1")};
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view(), ball1), 0u);
    const std::optional<ObjectId> ball2[] = {object("gripper__prob05", "ball2")};
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view(), ball2), 1u);
    // two variables, an inequality between them: ordered pairs of distinct balls in rooma (ball1 excluded for ?b)
    c.variables.push_back({"c", {}});
    c.literals.push_back({PredicateId{pred(*task, "at")}, true, {1, formalism::object_term(object("gripper__prob05", "rooma"))}});
    c.literals.push_back({PredicateId{pred(*task, "ball")}, true, {1}});
    c.equalities.push_back({0, 1, false});
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view()), 11u * 11u);
    EXPECT_EQ(count_bindings(*task, ws, c, s0.view(), {}, 5), 5u);
    const std::string roomb = task->object_name(object("gripper__prob05", "roomb"));
    EXPECT_NE(c.str(*task).find("(not (at ?b " + roomb + "))"), std::string::npos) << c.str(*task);
}

TEST(Bindings, DeclaredTypes)
{
    // gripper with declared types (the text format has none): ball and gripper, from the type predicates
    formalism::TaskData data = formalism::read_task_text_file((data_dir() / "tasks" / "gripper__prob05.txt").string());
    for (const char* tn : {"ball", "gripper"})
    {
        formalism::Type t;
        t.name = data.intern_string(tn);
        data.types.push_back(t);
    }
    std::vector<std::vector<TypeId>> types(data.num_objects());
    for (const auto& a : data.static_init)
        for (u32 k = 0; k < 2; ++k)
            if (data.str(data.predicates[a.pred.v].name) == (k == 0 ? "ball" : "gripper"))
                types[data.objects_of(a)[0].v].push_back(TypeId{k});
    for (u32 o = 0; o < data.num_objects(); ++o)
        data.objects[o].types = formalism::TaskData::append(data.type_ids, std::span<const TypeId>(types[o]));
    const auto task = Task::create(std::move(data));
    Workspace& ws = task->workspace();
    const StateView s = task->initial_state().view();
    ConjunctiveCondition c;
    c.variables = {{"x", {TypeId{0}}}};
    EXPECT_EQ(count_bindings(*task, ws, c, s), 12u);
    c.variables[0].types.push_back(TypeId{1});  // either
    EXPECT_EQ(count_bindings(*task, ws, c, s), 14u);
    c.variables[0].types = {TypeId{1}};
    const std::optional<ObjectId> ball1[] = {object("gripper__prob05", "ball1")};
    EXPECT_EQ(count_bindings(*task, ws, c, s, ball1), 0u);
    const std::optional<ObjectId> left[] = {object("gripper__prob05", "left")};
    EXPECT_EQ(count_bindings(*task, ws, c, s, left), 1u);
    EXPECT_NE(c.str(*task).find("?x - gripper"), std::string::npos) << c.str(*task);
    c.variables[0].types = {TypeId{7}};
    EXPECT_THROW((void) count_bindings(*task, ws, c, s), std::invalid_argument);
}

TEST(Bindings, DerivedPredicates)
{
    const auto task = suite_task("philosophers__p03-phil4");
    ASSERT_TRUE(task->has_axioms());
    Workspace& ws = task->workspace();
    Oracle oracle(task->data());
    BruteForce brute(*task, oracle);
    u64 derived_literals = 0, nonempty = 0;
    for (const State& s : random_states(*task, 2, 10, 3))
    {
        brute.set_state(s.view());
        for (u32 p = 0; p < task->data().predicates.size(); ++p)
        {
            const auto& P = task->data().predicates[p];
            if (P.kind != formalism::PredKind::Derived || P.arity > 2)
                continue;
            for (bool positive : {true, false})
            {
                ConjunctiveCondition c;
                for (u32 v = 0; v < P.arity; ++v)
                    c.variables.push_back({"x" + std::to_string(v), {}});
                ConjunctiveCondition::Literal l{PredicateId{p}, positive, {}};
                for (u32 v = 0; v < P.arity; ++v)
                    l.terms.push_back(static_cast<formalism::Term>(v));
                c.literals.push_back(l);
                ++derived_literals;
                const auto want = brute.solve(c);
                nonempty += !want.empty();
                EXPECT_EQ(sorted(bindings(*task, ws, c, s.view())), want) << c.str(*task);
            }
        }
    }
    EXPECT_GT(nonempty, 0u);
    std::printf("derived literals %llu (nonempty %llu)\n", static_cast<unsigned long long>(derived_literals),
                static_cast<unsigned long long>(nonempty));
}

TEST(Bindings, NumericConstraints)
{
    // counters: (<= (+ (value ?c) 1) (max_int)) and value comparisons between counters
    const auto task = Task::from_text_file((data_dir() / "numeric_tasks" / "cs-counters.txt").string());
    Workspace& ws = task->workspace();
    const auto& T = task->data();
    u32 value = ~u32{0}, max_int = ~u32{0};
    for (u32 f = 0; f < T.functions.size(); ++f)
    {
        if (T.str(T.functions[f].name) == "value")
            value = f;
        if (T.str(T.functions[f].name) == "max_int")
            max_int = f;
    }
    ASSERT_NE(value, ~u32{0});
    ConjunctiveCondition c;
    c.variables = {{"c", {}}, {"d", {}}};
    auto fn = [&](u32 f, std::vector<formalism::Term> terms)
    {
        formalism::Expr x;
        x.op = formalism::ExprOp::Function;
        x.func = FunctionId{f};
        x.terms = {static_cast<u32>(c.expr_terms.size()), static_cast<u32>(terms.size())};
        c.expr_terms.insert(c.expr_terms.end(), terms.begin(), terms.end());
        c.exprs.push_back(x);
        return static_cast<u32>(c.exprs.size() - 1);
    };
    const u32 vc = fn(value, {0}), vd = fn(value, {1});
    c.constraints.push_back({formalism::Comparator::Lt, vc, vd});
    if (max_int != ~u32{0})
    {
        const u32 m = fn(max_int, {});
        formalism::Expr one;
        one.op = formalism::ExprOp::Number;
        one.value = 1;
        c.exprs.push_back(one);
        formalism::Expr add;
        add.op = formalism::ExprOp::Add;
        add.a = vd;
        add.b = static_cast<u32>(c.exprs.size() - 1);
        c.exprs.push_back(add);
        c.constraints.push_back({formalism::Comparator::Le, static_cast<u32>(c.exprs.size() - 1), m});
    }
    ASSERT_NO_THROW(c.validate(*task));
    Oracle oracle(T);
    BruteForce brute(*task, oracle);
    u64 nonempty = 0;
    for (const State& s : random_states(*task, 2, 12, 9))
    {
        brute.set_state(s.view());
        const auto want = brute.solve(c);
        nonempty += !want.empty();
        EXPECT_EQ(sorted(bindings(*task, ws, c, s.view())), want) << c.str(*task);
    }
    EXPECT_GT(nonempty, 0u);
    EXPECT_NE(c.str(*task).find("(< (value ?c) (value ?d))"), std::string::npos) << c.str(*task);
}

TEST(Bindings, NumericSchemasFollowTheApplicableActions)
{
    for (const auto& e : fs::directory_iterator(data_dir() / "numeric_tasks"))
    {
        if (e.path().extension() != ".txt")
            continue;
        const auto task = Task::from_text_file(e.path().string());
        Workspace& ws = task->workspace();
        for (const State& s : random_states(*task, 1, k_sanitized ? 4 : 10, 2))
        {
            const std::vector<Action> acts = applicable(*task, s.view());
            for (u32 sc = 0; sc < task->num_schemas(); ++sc)
            {
                std::vector<Tuple> want;
                for (const Action& a : acts)
                    if (a.schema.v == sc)
                        want.push_back(a.binding);
                EXPECT_EQ(sorted(bindings(*task, ws, SchemaId{sc}, s.view())), want)
                    << e.path().stem().string() << " " << task->schema_name(SchemaId{sc});
            }
        }
    }
}

TEST(Bindings, InvalidInputThrows)
{
    const auto task = suite_task("gripper__prob05");
    Workspace& ws = task->workspace();
    const State s = task->initial_state();
    ConjunctiveCondition c;
    c.variables = {{"x", {}}};
    c.literals.push_back({PredicateId{pred(*task, "at")}, true, {0}});  // wrong arity
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view()), std::invalid_argument);
    c.literals[0].terms = {0, 1};  // variable out of range
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view()), std::invalid_argument);
    c.literals[0].terms = {0, formalism::object_term(ObjectId{task->num_objects()})};  // object out of range
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view()), std::invalid_argument);
    c.literals[0] = {PredicateId{static_cast<u32>(task->data().predicates.size())}, true, {}};
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view()), std::invalid_argument);
    c.literals.clear();
    c.exprs.push_back({});
    c.constraints.push_back({formalism::Comparator::Eq, 0, 5});  // expression index out of range
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view()), std::invalid_argument);
    c.constraints.clear();
    const std::optional<ObjectId> two[] = {std::nullopt, std::nullopt};
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view(), two), std::invalid_argument);
    const std::optional<ObjectId> far[] = {ObjectId{task->num_objects()}};
    EXPECT_THROW((void) count_bindings(*task, ws, c, s.view(), far), std::invalid_argument);
    EXPECT_THROW((void) count_bindings(*task, ws, SchemaId{task->num_schemas()}, s.view()), std::invalid_argument);
    // a nested enumeration on the same workspace
    EXPECT_THROW(for_each_binding(*task, ws, c, s.view(), {},
                                  [&](std::span<const ObjectId>) { (void) count_bindings(*task, ws, c, s.view()); return false; }),
                 std::logic_error);
    // ... is fine on another workspace, and the successor generator of the same workspace stays usable
    Workspace other(*task);
    u64 inner = 0;
    for_each_binding(*task, ws, c, s.view(), {},
                     [&](std::span<const ObjectId>)
                     {
                         inner += count_bindings(*task, other, c, s.view());
                         inner += ws.successors().applicable_actions(s.view()).size();
                         return false;
                     });
    EXPECT_GT(inner, 0u);
    // resume_after must fit the partial binding
    const std::optional<ObjectId> fixed[] = {ObjectId{1}};
    const ObjectId r[] = {ObjectId{2}};
    EXPECT_THROW(for_each_binding(*task, ws, c, s.view(), fixed, [](std::span<const ObjectId>) {}, BindingOptions{~u64{0}, r}),
                 std::invalid_argument);
}

TEST(Bindings, PreconditionAndGoalConditions)
{
    const auto task = suite_task("blocks__probBLOCKS-8-0");
    const auto& T = task->data();
    for (u32 sc = 0; sc < task->num_schemas(); ++sc)
    {
        const ConjunctiveCondition c = ConjunctiveCondition::precondition(*task, SchemaId{sc});
        EXPECT_EQ(c.arity(), T.schemas[sc].arity());
        EXPECT_EQ(c.literals.size(), T.schemas[sc].precondition.literals.count);
        EXPECT_NO_THROW(c.validate(*task));
        EXPECT_TRUE(c == ConjunctiveCondition::precondition(*task, SchemaId{sc}));
    }
    const ConjunctiveCondition g = ConjunctiveCondition::goal(*task);
    EXPECT_EQ(g.arity(), 0u);
    EXPECT_EQ(g.literals.size(), T.goal.literals.count);
    EXPECT_FALSE(g == ConjunctiveCondition::precondition(*task, SchemaId{0}));
    // the goal does not hold initially: no binding; an empty condition has the one empty binding
    EXPECT_EQ(count_bindings(*task, task->workspace(), g, task->initial_state().view()), 0u);
    EXPECT_EQ(count_bindings(*task, task->workspace(), ConjunctiveCondition{}, task->initial_state().view()), 1u);
}

TEST(Bindings, ConcurrentEnumerationOverOneTask)
{
    const auto task = suite_task("blocks__probBLOCKS-8-0");
    const std::vector<State> states = random_states(*task, 2, k_sanitized ? 4 : 10, 4);
    std::mt19937_64 rng(3);
    std::vector<ConjunctiveCondition> conds;
    for (int i = 0; i < 8; ++i)
        conds.push_back(random_condition(*task, rng, 3));
    for (u32 sc = 0; sc < task->num_schemas(); ++sc)
        conds.push_back(ConjunctiveCondition::precondition(*task, SchemaId{sc}));
    auto run = [&](Workspace& ws)
    {
        std::vector<std::vector<Tuple>> out;
        for (const State& s : states)
        {
            for (u32 sc = 0; sc < task->num_schemas(); ++sc)
                out.push_back(bindings(*task, ws, SchemaId{sc}, s.view()));
            for (const auto& c : conds)
                out.push_back(bindings(*task, ws, c, s.view()));
        }
        return out;
    };
    Workspace ref_ws(*task);
    const auto ref = run(ref_ws);
    std::vector<std::vector<std::vector<Tuple>>> got(8);
    std::vector<std::thread> threads;
    for (u32 t = 0; t < 8; ++t)
        threads.emplace_back([&, t] { got[t] = run(task->workspace()); });
    for (auto& th : threads)
        th.join();
    for (u32 t = 0; t < 8; ++t)
        EXPECT_EQ(got[t], ref) << "thread " << t;
}
}  // namespace
