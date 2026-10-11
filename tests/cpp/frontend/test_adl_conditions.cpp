// Conditions of the ADL fragment (quantifiers, implication, disjunction, negation over compounds, nesting) in goals,
// preconditions, conditional effects (under forall effects, nested in each other) and derived predicate bodies: on small
// tasks, in every state the base actions reach, the goal test, the applicable actions and the successors agree with the
// formula as written, evaluated here; search finds plans for guarded forall goals that satisfy the formula.

#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace mymyr;

namespace
{
const std::string kDomain = R"((define (domain adl) (:requirements :adl :derived-predicates)
 (:predicates (s ?x) (p ?x) (q ?x) (r ?x ?y) (m ?x) (n ?x) (k ?x) (ok ?x))
 (:derived (ok ?x) (forall (?y) (imply (r ?x ?y) (p ?y))))
 (:action set-p :parameters (?x) :precondition (not (p ?x)) :effect (p ?x))
 (:action del-p :parameters (?x) :precondition (p ?x) :effect (not (p ?x)))
 (:action set-q :parameters (?x) :precondition (not (q ?x)) :effect (q ?x))
 (:action act :parameters (?x) :precondition (or (q ?x) (forall (?y) (imply (s ?y) (p ?y))))
  :effect (and (n ?x) (forall (?w) (when (forall (?y) (imply (r ?w ?y) (q ?y))) (m ?w)))
   (when (exists (?z) (and (q ?z) (not (p ?z)))) (forall (?w) (when (not (exists (?y) (and (r ?w ?y) (not (p ?y))))) (k ?w))))))
 (:action check :parameters (?x) :precondition (and (ok ?x) (not (exists (?y) (and (r ?y ?x) (not (q ?y))))))
  :effect (n ?x))))";

const std::array<std::string, 3> kObjects = {"a", "b", "c"};

std::string problem(const std::string& goal)
{
    return "(define (problem p) (:domain adl) (:objects a b c) (:init (s a) (s c) (r a b) (r b c) (r c c)) (:goal " + goal +
           "))";
}

std::shared_ptr<const Task> make(const std::string& goal)
{
    const auto d = frontend::Domain::from_string(kDomain, "d.pddl");
    return Task::create(*d->instantiate_string(problem(goal), "p.pddl"));
}

/// The true atoms of a state, static ones included.
struct Atoms
{
    std::set<std::string> set;
    bool operator()(const std::string& pred, const std::string& x) const { return set.contains("(" + pred + " " + x + ")"); }
    bool operator()(const std::string& pred, const std::string& x, const std::string& y) const
    {
        return set.contains("(" + pred + " " + x + " " + y + ")");
    }
    template<class F>
    static bool all(F&& f)
    {
        return std::ranges::all_of(kObjects, f);
    }
    template<class F>
    static bool any(F&& f)
    {
        return std::ranges::any_of(kObjects, f);
    }
};

Atoms atoms_of(const Task& t, StateView s)
{
    Atoms out;
    for (const std::string& a : t.format_atoms(s))
        out.set.insert(a);
    for (const char* a : {"(s a)", "(s c)", "(r a b)", "(r b c)", "(r c c)"})
        out.set.insert(a);
    return out;
}

/// The states the actions set-p, del-p and set-q reach from the initial state: every combination of p and q.
std::vector<State> base_states(const Task& t)
{
    std::unordered_set<State> seen{t.initial_state()};
    std::vector<State> frontier{t.initial_state()}, out;
    const WorkspaceLease ws = t.workspace();
    while (!frontier.empty())
    {
        const State s = frontier.back();
        frontier.pop_back();
        out.push_back(s);
        for (const Action& a : ws->successors().applicable_actions(s.view()))
        {
            const std::string name = t.format(a.label());
            if (name.starts_with("(act") || name.starts_with("(check"))
                continue;
            State n = ws->successors().apply(s.view(), a.label());
            if (seen.insert(n).second)
                frontier.push_back(n);
        }
    }
    return out;
}

using Formula = std::function<bool(const Atoms&)>;

/// "(p a)" or "(not (p a))".
std::string literal_text(const formalism::TaskData& d, const formalism::Literal& l)
{
    std::string s = "(" + std::string(d.str(d.predicate(l.pred).name));
    for (formalism::Term x : d.terms_of(l))
        s += " " + std::string(d.str(d.objects[formalism::term_object(x).v].name));
    s += ")";
    return l.positive ? s : "(not " + s + ")";
}

bool literal_holds(const formalism::TaskData& d, const formalism::Literal& l, const Atoms& at)
{
    const auto terms = d.terms_of(l);
    const bool value = d.str(d.predicate(l.pred).name) == "="
                           ? formalism::term_object(terms[0]).v == formalism::term_object(terms[1]).v
                           : at.set.contains(literal_text(d, {l.pred, true, l.terms}));
    return value == l.positive;
}

/// The goal test agrees with `f` in every base state. The encoding goal (the goal as written, expanded) is `encoding`
/// and `complete`: its literals hold in every goal state, and when it is complete they hold in the goal states only.
void expect_goal(const std::string& goal, const Formula& f, const std::set<std::string>& encoding, bool complete)
{
    const auto t = make(goal);
    const formalism::TaskData& d = t->data();
    std::set<std::string> got;
    for (const formalism::Literal& l : t->encoding_goal())
        got.insert(literal_text(d, l));
    EXPECT_EQ(got, encoding) << goal;
    EXPECT_EQ(t->encoding_goal_complete(), complete) << goal;
    EXPECT_FALSE(t->encoding_goal_unsatisfiable()) << goal;
    const std::vector<State> states = base_states(*t);
    ASSERT_EQ(states.size(), 64u);
    u32 holds = 0;
    for (const State& s : states)
    {
        const Atoms at = atoms_of(*t, s.view());
        EXPECT_EQ(t->is_goal(s.view()), f(at)) << goal;
        holds += f(at) ? 1 : 0;
        const bool expanded = std::ranges::all_of(t->encoding_goal(), [&](const formalism::Literal& l) { return literal_holds(d, l, at); });
        EXPECT_TRUE(expanded || !f(at)) << goal;
        if (complete)
        {
            EXPECT_EQ(expanded, f(at)) << goal;
        }
    }
    EXPECT_GT(holds, 0u) << goal;
    EXPECT_LT(holds, states.size()) << goal;
}
}  // namespace

TEST(AdlConditions, GoalsMatchTheFormula)
{
    using A = Atoms;
    // a static guard (s, r) is decided per binding in the encoding goal; a fluent guard, or a disjunction or an
    // existential with two or more disjuncts that static atoms do not decide, leaves the encoding goal incomplete
    expect_goal("(forall (?x) (imply (s ?x) (p ?x)))", [](const A& at) { return A::all([&](auto& x) { return !at("s", x) || at("p", x); }); },
                {"(p a)", "(p c)"}, true);
    expect_goal("(and (q a) (forall (?x) (imply (s ?x) (p ?x))))",
                [](const A& at) { return at("q", "a") && A::all([&](auto& x) { return !at("s", x) || at("p", x); }); },
                {"(q a)", "(p a)", "(p c)"}, true);
    expect_goal("(forall (?x) (imply (s ?x) (forall (?y) (imply (r ?x ?y) (q ?y)))))",
                [](const A& at)
                { return A::all([&](auto& x) { return !at("s", x) || A::all([&](auto& y) { return !at("r", x, y) || at("q", y); }); }); },
                {"(q b)", "(q c)"}, true);
    expect_goal("(forall (?x) (or (p ?x) (q ?x)))", [](const A& at) { return A::all([&](auto& x) { return at("p", x) || at("q", x); }); }, {},
                false);
    // disjunctions that static atoms decide but for one disjunct
    expect_goal("(forall (?x) (or (not (s ?x)) (p ?x)))", [](const A& at) { return A::all([&](auto& x) { return !at("s", x) || at("p", x); }); },
                {"(p a)", "(p c)"}, true);
    expect_goal("(not (exists (?x) (and (s ?x) (q ?x))))", [](const A& at) { return !A::any([&](auto& x) { return at("s", x) && at("q", x); }); },
                {"(not (q a))", "(not (q c))"}, true);
    expect_goal("(exists (?x) (and (r ?x b) (p ?x)))", [](const A& at) { return A::any([&](auto& x) { return at("r", x, "b") && at("p", x); }); },
                {"(r a b)", "(p a)"}, true);
    expect_goal("(exists (?x) (and (p ?x) (not (q ?x))))", [](const A& at) { return A::any([&](auto& x) { return at("p", x) && !at("q", x); }); },
                {}, false);
    expect_goal("(or (and (p a) (q b)) (not (exists (?x) (q ?x))))",
                [](const A& at) { return (at("p", "a") && at("q", "b")) || !A::any([&](auto& x) { return at("q", x); }); }, {}, false);
    expect_goal("(forall (?x) (imply (p ?x) (exists (?y) (and (r ?x ?y) (q ?y)))))",
                [](const A& at)
                { return A::all([&](auto& x) { return !at("p", x) || A::any([&](auto& y) { return at("r", x, y) && at("q", y); }); }); },
                {}, false);
    expect_goal("(not (forall (?x) (or (p ?x) (exists (?y) (and (r ?y ?x) (q ?y))))))",
                [](const A& at)
                { return !A::all([&](auto& x) { return at("p", x) || A::any([&](auto& y) { return at("r", y, x) && at("q", y); }); }); },
                {}, false);
    expect_goal("(forall (?x ?y) (imply (and (r ?x ?y) (p ?x)) (or (p ?y) (= ?x ?y))))",
                [](const A& at)
                {
                    return A::all([&](auto& x)
                                  { return A::all([&](auto& y) { return !(at("r", x, y) && at("p", x)) || at("p", y) || x == y; }); });
                },
                {}, false);
}

TEST(AdlConditions, PreconditionsEffectConditionsAndDerivedBodiesMatchTheFormula)
{
    const auto t = make("(p a)");
    const std::vector<State> states = base_states(*t);
    ASSERT_EQ(states.size(), 64u);
    const WorkspaceLease ws = t->workspace();
    u32 acts = 0, checks = 0, marks = 0, nested_marks = 0;
    for (const State& s : states)
    {
        const Atoms at = atoms_of(*t, s.view());
        const bool guarded = Atoms::all([&](auto& y) { return !at("s", y) || at("p", y); });
        auto ok = [&](const std::string& x) { return Atoms::all([&](auto& y) { return !at("r", x, y) || at("p", y); }); };
        auto mark = [&](const std::string& w) { return Atoms::all([&](auto& y) { return !at("r", w, y) || at("q", y); }); };
        const bool nested = Atoms::any([&](auto& z) { return at("q", z) && !at("p", z); });
        std::set<std::string> want, got;
        for (const std::string& x : kObjects)
        {
            if (at("q", x) || guarded)
                want.insert("(act " + x + ")");
            if (ok(x) && !Atoms::any([&](auto& y) { return at("r", y, x) && !at("q", y); }))
                want.insert("(check " + x + ")");
        }
        for (const Action& a : ws->successors().applicable_actions(s.view()))
        {
            const std::string name = t->format(a.label());
            if (!name.starts_with("(act") && !name.starts_with("(check"))
                continue;
            got.insert(name);
            if (!name.starts_with("(act"))
            {
                ++checks;
                continue;
            }
            ++acts;
            // the successor adds (n x), (m w) for every w whose condition holds in s, and (k w) for every w with ok(w)
            // if some z has q but not p
            std::vector<std::string> expected = t->format_atoms(s.view());
            expected.push_back("(n " + name.substr(5, name.size() - 6) + ")");
            for (const std::string& w : kObjects)
                if (mark(w))
                {
                    expected.push_back("(m " + w + ")");
                    ++marks;
                }
            for (const std::string& w : kObjects)
                if (nested && ok(w))
                {
                    expected.push_back("(k " + w + ")");
                    ++nested_marks;
                }
            const State n = ws->successors().apply(s.view(), a.label());
            std::vector<std::string> succ = t->format_atoms(n.view());
            std::ranges::sort(expected);
            std::ranges::sort(succ);
            EXPECT_EQ(succ, expected) << name;
        }
        EXPECT_EQ(got, want);
    }
    EXPECT_GT(acts, 0u);
    EXPECT_GT(checks, 0u);
    EXPECT_GT(marks, 0u);
    EXPECT_GT(nested_marks, 0u);
}

TEST(AdlConditions, SearchFindsPlansForGuardedForallGoals)
{
    const std::string goal = "(and (forall (?x) (imply (s ?x) (p ?x))) (forall (?x) (imply (r ?x ?x) (m ?x))) (not (q b)))";
    const auto t = make(goal);
    auto holds = [&](const Atoms& at)
    {
        return Atoms::all([&](auto& x) { return (!at("s", x) || at("p", x)) && (!at("r", x, x) || at("m", x)); }) && !at("q", "b");
    };
    const BrfsResult b = brfs(*t);
    const search::BestFirstResult a = search::astar_eager(*t);
    ASSERT_TRUE(b.solved);
    ASSERT_EQ(a.status, search::SearchStatus::Solved) << a.message;
    for (const std::vector<Action>* plan : {&b.plan, &a.plan})
    {
        const WorkspaceLease ws = t->workspace();
        State s = t->initial_state();
        EXPECT_FALSE(holds(atoms_of(*t, s.view())));
        for (const Action& x : *plan)
            s = ws->successors().apply(s.view(), x.label());
        EXPECT_TRUE(holds(atoms_of(*t, s.view())));
        EXPECT_TRUE(t->is_goal(s.view()));
        // (set-p a), (set-p c), (set-q c) and one act, whose forall effect marks c once q c holds
        EXPECT_EQ(plan->size(), 4u);
    }
}

TEST(AdlConditions, ConditionsOfForallEffectsBecomeHelperPredicates)
{
    // the condition of the forall effect quantifies universally over its variable ?w: it becomes a derived predicate
    // axiom_<k> over ?w; the derived predicate ok and the other quantifiers compile as loki compiles them
    const auto d = frontend::Domain::from_string(kDomain, "d.pddl");
    const auto data = d->instantiate_string(problem("(forall (?x) (imply (s ?x) (p ?x)))"), "p.pddl");
    u32 helpers = 0, unary = 0;
    for (const formalism::Predicate& p : data->predicates)
        if (const std::string name(data->str(p.name)); name.starts_with("axiom_"))
        {
            ++helpers;
            EXPECT_EQ(p.kind, formalism::PredKind::Derived) << name;
            unary += p.arity == 1 ? 1 : 0;
        }
    EXPECT_GE(helpers, 4u);
    EXPECT_GE(unary, 1u);
    // the same text from a second parse (helper names, argument orders and rule orders are fixed)
    const auto again = frontend::Domain::from_string(kDomain, "d.pddl")->instantiate_string(problem("(forall (?x) (imply (s ?x) (p ?x)))"), "p.pddl");
    EXPECT_EQ(formalism::write_task_text(*data), formalism::write_task_text(*again));
}
