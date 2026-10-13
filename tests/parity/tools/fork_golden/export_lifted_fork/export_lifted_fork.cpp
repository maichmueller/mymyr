// Export a normalized lifted task (incl. axioms) with the C++ API of the mimir fork (0.16.x). Needed because no
// pymimir build binds Domain/Problem::get_axioms().
//
//   X k   then per axiom:  head_pred arity  H terms..  L k lits..   (terms >= 0 parameter index, < 0 object -o-1)
#include <mimir/mimir.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mimir;
using namespace mimir::formalism;

struct Exporter
{
    Problem problem;
    std::unordered_map<std::string, int> obj_id;
    std::map<std::pair<char, std::string>, int> pred_id;
    std::vector<std::tuple<char, size_t, std::string>> preds;
    std::ostringstream out;

    template<typename P>
    void add_preds(char kind, const PredicateList<P>& ps)
    {
        for (auto p : ps)
            if (pred_id.emplace(std::make_pair(kind, p->get_name()), (int) preds.size()).second)
                preds.emplace_back(kind, p->get_arity(), p->get_name());
    }
    template<typename P>
    static char kind_of()
    {
        if constexpr (std::is_same_v<P, StaticTag>)
            return 'S';
        else if constexpr (std::is_same_v<P, FluentTag>)
            return 'F';
        else
            return 'D';
    }
    template<typename P>
    int pid(Predicate<P> p) const
    {
        return pred_id.at(std::make_pair(kind_of<P>(), p->get_name()));
    }
    int term(Term t, const std::unordered_map<const void*, int>& var_pos) const
    {
        const auto& v = t->get_variant();
        if (std::holds_alternative<Object>(v))
            return -obj_id.at(std::get<Object>(v)->get_name()) - 1;
        return var_pos.at(std::get<Variable>(v));
    }
    template<typename P>
    std::string lifted(Literal<P> l, const std::unordered_map<const void*, int>& vp) const
    {
        std::ostringstream s;
        s << pid(l->get_atom()->get_predicate()) << " " << (l->get_polarity() ? 1 : 0);
        for (auto t : l->get_atom()->get_terms())
            s << " " << term(t, vp);
        return s.str();
    }
    template<typename P>
    std::string ground(GroundAtom<P> a) const
    {
        std::ostringstream s;
        s << pid(a->get_predicate());
        for (auto o : a->get_objects())
            s << " " << obj_id.at(o->get_name());
        return s.str();
    }
    std::vector<std::string> condition(ConjunctiveCondition c, const std::unordered_map<const void*, int>& vp) const
    {
        std::vector<std::string> ls;
        for (auto l : c->get_literals<StaticTag>())
            ls.push_back(lifted(l, vp));
        for (auto l : c->get_literals<FluentTag>())
            ls.push_back(lifted(l, vp));
        for (auto l : c->get_literals<DerivedTag>())
            ls.push_back(lifted(l, vp));
        auto nullary = [&](const auto& gls)
        {
            for (auto gl : gls)
                ls.push_back(std::to_string(pid(gl->get_atom()->get_predicate())) + " " + (gl->get_polarity() ? "1" : "0"));
        };
        nullary(c->get_nullary_ground_literals<StaticTag>());
        nullary(c->get_nullary_ground_literals<FluentTag>());
        nullary(c->get_nullary_ground_literals<DerivedTag>());
        if (!c->get_numeric_constraints().empty() || !c->get_nullary_ground_numeric_constraints().empty())
            throw std::runtime_error("unsupported: numeric constraints");
        return ls;
    }
    void lits(const char* tag, const std::vector<std::string>& ls)
    {
        out << tag << " " << ls.size() << "\n";
        for (auto& l : ls)
            out << l << "\n";
    }

    explicit Exporter(Problem p) : problem(std::move(p))
    {
        const auto& objs = problem->get_problem_and_domain_objects();
        for (size_t i = 0; i < objs.size(); ++i)
            obj_id.emplace(objs[i]->get_name(), (int) i);
        const auto& dom = problem->get_domain();
        add_preds('S', dom->get_predicates<StaticTag>());
        add_preds('F', dom->get_predicates<FluentTag>());
        add_preds('D', dom->get_predicates<DerivedTag>());
        add_preds('D', problem->get_problem_and_domain_derived_predicates());

        out << "O " << objs.size() << "\nP " << preds.size() << "\n";
        for (auto& [k, a, n] : preds)
            out << k << " " << a << " " << n << "\n";
        out << "SI " << problem->get_static_initial_atoms().size() << "\n";
        for (auto a : problem->get_static_initial_atoms())
            out << ground(a) << "\n";
        out << "FI " << problem->get_fluent_initial_atoms().size() << "\n";
        for (auto a : problem->get_fluent_initial_atoms())
            out << ground(a) << "\n";
        std::vector<std::string> goal;
        auto goal_lits = [&](const auto& gls)
        {
            for (auto gl : gls)
            {
                std::string g = ground(gl->get_atom());
                auto sp = g.find(' ');
                std::string head = sp == std::string::npos ? g : g.substr(0, sp);
                std::string rest = sp == std::string::npos ? "" : g.substr(sp);
                goal.push_back(head + " " + (gl->get_polarity() ? "1" : "0") + rest);
            }
        };
        goal_lits(problem->get_goal_literals<StaticTag>());
        goal_lits(problem->get_goal_literals<FluentTag>());
        goal_lits(problem->get_goal_literals<DerivedTag>());
        lits("G", goal);

        const auto& actions = dom->get_actions();
        out << "A " << actions.size() << "\n";
        for (auto a : actions)
        {
            std::unordered_map<const void*, int> vp;
            const auto& ps = a->get_parameters();
            for (size_t i = 0; i < ps.size(); ++i)
                vp[ps[i]->get_variable()] = (int) i;
            out << a->get_name() << " " << ps.size() << "\n";
            lits("L", condition(a->get_conjunctive_condition(), vp));
            out << "E " << a->get_conditional_effects().size() << "\n";
            for (auto ce : a->get_conditional_effects())
            {
                auto cvp = vp;
                const auto& extra = ce->get_conjunctive_condition()->get_parameters();
                for (size_t j = 0; j < extra.size(); ++j)
                    cvp[extra[j]->get_variable()] = (int) (ps.size() + j);
                const auto& eextra = ce->get_conjunctive_effect()->get_parameters();
                for (size_t j = 0; j < eextra.size(); ++j)
                    cvp.emplace(eextra[j]->get_variable(), (int) (ps.size() + j));
                out << extra.size() << "\n";
                lits("L", condition(ce->get_conjunctive_condition(), cvp));
                std::vector<std::string> el;
                for (auto l : ce->get_conjunctive_effect()->get_literals())
                    el.push_back(lifted(l, cvp));
                lits("F", el);
            }
        }
        const auto& axioms = problem->get_problem_and_domain_axioms();
        out << "X " << axioms.size() << "\n";
        for (auto x : axioms)
        {
            std::unordered_map<const void*, int> vp;
            const auto& ps = x->get_parameters();
            for (size_t i = 0; i < ps.size(); ++i)
                vp[ps[i]->get_variable()] = (int) i;
            auto head = x->get_literal();
            out << pid(head->get_atom()->get_predicate()) << " " << ps.size() << "\nH";
            for (auto t : head->get_atom()->get_terms())
                out << " " << term(t, vp);
            out << "\n";
            lits("L", condition(x->get_conjunctive_condition(), vp));
        }
    }
};

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: export_lifted_fork domain.pddl problem.pddl out.txt\n");
        return 2;
    }
    try
    {
        Exporter e(ProblemImpl::create(argv[1], argv[2]));
        std::ofstream(argv[3]) << e.out.str();
        std::printf("%s: objects=%zu preds=%zu schemas=%zu axioms=%zu\n", argv[3], e.obj_id.size(), e.preds.size(),
                    e.problem->get_domain()->get_actions().size(), e.problem->get_problem_and_domain_axioms().size());
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
