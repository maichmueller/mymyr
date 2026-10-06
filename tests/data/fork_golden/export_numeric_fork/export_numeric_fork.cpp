// Export a normalized lifted task WITH numerics from the mimir fork (0.16.x). A variant of
// ../export_lifted_fork/export_lifted_fork.cpp extended with numeric fluents.
//
// Format = the lifted format (O, P, SI, FI, G, A, X) plus numeric sections:
//   N k            functions: kind(S|F|A) arity name       (function ids 0..k-1, all kinds in one space)
//   NI k           initial values: fid obj.. value         (static and fluent; auxiliary as fid with no args)
//   NG k           goal numeric constraints (ground, objects encoded -o-1)
//   M none | M min|max <expr>                               metric
// and per condition (after every L block, also for axioms) an   NC k   block of constraints, per effect
// (after every F block) an   NE k   block of fluent numeric effects and an   NA -|<op> <expr>   line.
//   constraint := <cmp> <expr> <expr>        cmp in > < = != >= <=
//   effect     := <op> fid terms.. <expr>    op in assign scale-up scale-down increase decrease
//   expr       := n tok..   postfix: c <val> | s fid terms.. | f fid terms.. | a fid | + | - | * | / | u (negate)
// Terms: >= 0 parameter index, < 0 object -o-1 (as in literals).
#include <mimir/mimir.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mimir;
using namespace mimir::formalism;

static std::string num(double v)
{
    char b[64];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

struct Exporter
{
    Problem problem;
    std::unordered_map<std::string, int> obj_id;
    std::map<std::pair<char, std::string>, int> pred_id;
    std::vector<std::tuple<char, size_t, std::string>> preds;
    std::map<std::pair<char, std::string>, int> fun_id;
    std::vector<std::tuple<char, size_t, std::string>> funs;
    std::ostringstream out;
    size_t n_constraints = 0, n_numeric_effects = 0, n_aux_effects = 0;

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
        else if constexpr (std::is_same_v<P, AuxiliaryTag>)
            return 'A';
        else
            return 'D';
    }
    template<typename P>
    int pid(Predicate<P> p) const
    {
        return pred_id.at(std::make_pair(kind_of<P>(), p->get_name()));
    }
    template<typename F>
    int fid(FunctionSkeleton<F> f)
    {
        auto key = std::make_pair(kind_of<F>(), f->get_name());
        auto it = fun_id.find(key);
        if (it != fun_id.end())
            return it->second;
        int id = (int) funs.size();
        fun_id.emplace(key, id);
        funs.emplace_back(kind_of<F>(), f->get_arity(), f->get_name());
        return id;
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

    // ---- numeric expressions (postfix token lists)
    static const char* bin_op(loki::BinaryOperatorEnum op)
    {
        switch (op)
        {
            case loki::BinaryOperatorEnum::PLUS: return "+";
            case loki::BinaryOperatorEnum::MINUS: return "-";
            case loki::BinaryOperatorEnum::MUL: return "*";
            case loki::BinaryOperatorEnum::DIV: return "/";
        }
        throw std::runtime_error("bin op");
    }
    static const char* multi_op(loki::MultiOperatorEnum op) { return op == loki::MultiOperatorEnum::PLUS ? "+" : "*"; }
    static const char* cmp(loki::BinaryComparatorEnum c)
    {
        switch (c)
        {
            case loki::BinaryComparatorEnum::GREATER: return ">";
            case loki::BinaryComparatorEnum::LESS: return "<";
            case loki::BinaryComparatorEnum::EQUAL: return "=";
            case loki::BinaryComparatorEnum::UNEQUAL: return "!=";
            case loki::BinaryComparatorEnum::GREATER_EQUAL: return ">=";
            case loki::BinaryComparatorEnum::LESS_EQUAL: return "<=";
        }
        throw std::runtime_error("cmp");
    }
    static const char* assign_op(loki::AssignOperatorEnum a)
    {
        switch (a)
        {
            case loki::AssignOperatorEnum::ASSIGN: return "assign";
            case loki::AssignOperatorEnum::SCALE_UP: return "scale-up";
            case loki::AssignOperatorEnum::SCALE_DOWN: return "scale-down";
            case loki::AssignOperatorEnum::INCREASE: return "increase";
            case loki::AssignOperatorEnum::DECREASE: return "decrease";
        }
        throw std::runtime_error("assign op");
    }
    // lifted
    void expr_rec(FunctionExpression e, const std::unordered_map<const void*, int>& vp, std::vector<std::string>& toks)
    {
        std::visit(
            [&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, FunctionExpressionNumber>)
                    toks.push_back("c " + num(arg->get_number()));
                else if constexpr (std::is_same_v<T, FunctionExpressionBinaryOperator>)
                {
                    expr_rec(arg->get_left_function_expression(), vp, toks);
                    expr_rec(arg->get_right_function_expression(), vp, toks);
                    toks.push_back(bin_op(arg->get_binary_operator()));
                }
                else if constexpr (std::is_same_v<T, FunctionExpressionMultiOperator>)
                {
                    const auto& es = arg->get_function_expressions();
                    if (es.empty())
                        throw std::runtime_error("empty multi operator");
                    expr_rec(es[0], vp, toks);
                    for (size_t i = 1; i < es.size(); ++i)
                    {
                        expr_rec(es[i], vp, toks);
                        toks.push_back(multi_op(arg->get_multi_operator()));
                    }
                }
                else if constexpr (std::is_same_v<T, FunctionExpressionMinus>)
                {
                    expr_rec(arg->get_function_expression(), vp, toks);
                    toks.push_back("u");
                }
                else if constexpr (std::is_same_v<T, FunctionExpressionFunction<StaticTag>>)
                    toks.push_back(fterm('s', arg->get_function(), vp));
                else if constexpr (std::is_same_v<T, FunctionExpressionFunction<FluentTag>>)
                    toks.push_back(fterm('f', arg->get_function(), vp));
                else
                    static_assert(sizeof(T) == 0, "variant");
            },
            e->get_variant());
    }
    template<typename F>
    std::string fterm(char tag, Function<F> f, const std::unordered_map<const void*, int>& vp)
    {
        std::ostringstream s;
        s << tag << " " << fid(f->get_function_skeleton());
        for (auto t : f->get_terms())
            s << " " << term(t, vp);
        return s.str();
    }
    std::string expr(FunctionExpression e, const std::unordered_map<const void*, int>& vp)
    {
        std::vector<std::string> toks;
        expr_rec(e, vp, toks);
        std::string s = std::to_string(toks.size());
        for (auto& t : toks)
            s += " " + t;
        return s;
    }
    // ground
    template<typename F>
    std::string gfterm(char tag, GroundFunction<F> f)
    {
        std::ostringstream s;
        s << tag << " " << fid(f->get_function_skeleton());
        if (tag != 'a')
            for (auto o : f->get_objects())
                s << " " << (-obj_id.at(o->get_name()) - 1);
        return s.str();
    }
    void gexpr_rec(GroundFunctionExpression e, std::vector<std::string>& toks)
    {
        std::visit(
            [&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, GroundFunctionExpressionNumber>)
                    toks.push_back("c " + num(arg->get_number()));
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionBinaryOperator>)
                {
                    gexpr_rec(arg->get_left_function_expression(), toks);
                    gexpr_rec(arg->get_right_function_expression(), toks);
                    toks.push_back(bin_op(arg->get_binary_operator()));
                }
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionMultiOperator>)
                {
                    const auto& es = arg->get_function_expressions();
                    gexpr_rec(es[0], toks);
                    for (size_t i = 1; i < es.size(); ++i)
                    {
                        gexpr_rec(es[i], toks);
                        toks.push_back(multi_op(arg->get_multi_operator()));
                    }
                }
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionMinus>)
                {
                    gexpr_rec(arg->get_function_expression(), toks);
                    toks.push_back("u");
                }
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionFunction<StaticTag>>)
                    toks.push_back(gfterm('s', arg->get_function()));
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionFunction<FluentTag>>)
                    toks.push_back(gfterm('f', arg->get_function()));
                else if constexpr (std::is_same_v<T, GroundFunctionExpressionFunction<AuxiliaryTag>>)
                    toks.push_back(gfterm('a', arg->get_function()));
                else
                    static_assert(sizeof(T) == 0, "variant");
            },
            e->get_variant());
    }
    std::string gexpr(GroundFunctionExpression e)
    {
        std::vector<std::string> toks;
        gexpr_rec(e, toks);
        std::string s = std::to_string(toks.size());
        for (auto& t : toks)
            s += " " + t;
        return s;
    }
    std::string gconstraint(GroundNumericConstraint c)
    {
        return std::string(cmp(c->get_binary_comparator())) + " " + gexpr(c->get_left_function_expression()) + " "
               + gexpr(c->get_right_function_expression());
    }

    std::vector<std::string> condition(ConjunctiveCondition c, const std::unordered_map<const void*, int>& vp,
                                       std::vector<std::string>& ncs)
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
        for (auto nc : c->get_numeric_constraints())
            ncs.push_back(std::string(cmp(nc->get_binary_comparator())) + " " + expr(nc->get_left_function_expression(), vp) + " "
                          + expr(nc->get_right_function_expression(), vp));
        for (auto gnc : c->get_nullary_ground_numeric_constraints())
            ncs.push_back(gconstraint(gnc));
        n_constraints += ncs.size();
        return ls;
    }
    void lits(const char* tag, const std::vector<std::string>& ls)
    {
        out << tag << " " << ls.size() << "\n";
        for (auto& l : ls)
            out << l << "\n";
    }
    void cond_block(ConjunctiveCondition c, const std::unordered_map<const void*, int>& vp)
    {
        std::vector<std::string> ncs;
        lits("L", condition(c, vp, ncs));
        lits("NC", ncs);
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
        for (auto f : dom->get_function_skeletons<StaticTag>())
            fid(f);
        for (auto f : dom->get_function_skeletons<FluentTag>())
            fid(f);
        if (dom->get_auxiliary_function_skeleton())
            fid(dom->get_auxiliary_function_skeleton().value());

        std::ostringstream body;  // everything after the function table (functions may be discovered late)
        std::swap(out, body);

        out << "SI " << problem->get_static_initial_atoms().size() << "\n";
        for (auto a : problem->get_static_initial_atoms())
            out << ground(a) << "\n";
        out << "FI " << problem->get_fluent_initial_atoms().size() << "\n";
        for (auto a : problem->get_fluent_initial_atoms())
            out << ground(a) << "\n";
        std::vector<std::string> ni;
        auto init_vals = [&](const auto& vals, char tag)
        {
            for (auto v : vals)
            {
                std::string t = gfterm(tag, v->get_function());
                ni.push_back(t.substr(2) + " " + num(v->get_number()));
            }
        };
        init_vals(problem->get_initial_function_values<StaticTag>(), 's');
        init_vals(problem->get_initial_function_values<FluentTag>(), 'f');
        if (problem->get_auxiliary_function_value())
            init_vals(std::vector { problem->get_auxiliary_function_value().value() }, 'a');
        lits("NI", ni);

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
        std::vector<std::string> ng;
        for (auto c : problem->get_goal_numeric_constraints())
            ng.push_back(gconstraint(c));
        lits("NG", ng);
        if (const auto& m = problem->get_optimization_metric(); m.has_value())
            out << "M " << (m.value()->get_optimization_metric() == loki::OptimizationMetricEnum::MINIMIZE ? "min" : "max") << " "
                << gexpr(m.value()->get_function_expression()) << "\n";
        else
            out << "M none\n";

        const auto& actions = dom->get_actions();
        out << "A " << actions.size() << "\n";
        for (auto a : actions)
        {
            std::unordered_map<const void*, int> vp;
            const auto& ps = a->get_parameters();
            for (size_t i = 0; i < ps.size(); ++i)
                vp[ps[i]->get_variable()] = (int) i;
            out << a->get_name() << " " << ps.size() << "\n";
            cond_block(a->get_conjunctive_condition(), vp);
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
                cond_block(ce->get_conjunctive_condition(), cvp);
                std::vector<std::string> el;
                for (auto l : ce->get_conjunctive_effect()->get_literals())
                    el.push_back(lifted(l, cvp));
                lits("F", el);
                std::vector<std::string> ne;
                for (auto e : ce->get_conjunctive_effect()->get_fluent_numeric_effects())
                {
                    std::string t = fterm('f', e->get_function(), cvp);
                    ne.push_back(std::string(assign_op(e->get_assign_operator())) + " " + t.substr(2) + " " + expr(e->get_function_expression(), cvp));
                }
                n_numeric_effects += ne.size();
                lits("NE", ne);
                if (const auto& ax = ce->get_conjunctive_effect()->get_auxiliary_numeric_effect(); ax.has_value())
                {
                    ++n_aux_effects;
                    out << "NA " << assign_op(ax.value()->get_assign_operator()) << " " << expr(ax.value()->get_function_expression(), cvp)
                        << "\n";
                }
                else
                    out << "NA -\n";
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
            cond_block(x->get_conjunctive_condition(), vp);
        }

        std::swap(out, body);
        out << "O " << objs.size() << "\nP " << preds.size() << "\n";
        for (auto& [k, a, n] : preds)
            out << k << " " << a << " " << n << "\n";
        out << "N " << funs.size() << "\n";
        for (auto& [k, a, n] : funs)
            out << k << " " << a << " " << n << "\n";
        out << body.str();
    }
};

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: export_numeric_fork domain.pddl problem.pddl out.txt\n");
        return 2;
    }
    try
    {
        auto t0 = std::chrono::steady_clock::now();
        auto problem = ProblemImpl::create(argv[1], argv[2]);
        auto t1 = std::chrono::steady_clock::now();
        Exporter e(problem);
        std::ofstream(argv[3]) << e.out.str();
        auto t2 = std::chrono::steady_clock::now();
        std::printf("{\"out\":\"%s\",\"objects\":%zu,\"preds\":%zu,\"functions\":%zu,\"schemas\":%zu,\"axioms\":%zu,\"constraints\":%zu,"
                    "\"numeric_effects\":%zu,\"aux_effects\":%zu,\"goal_constraints\":%zu,\"fork_create_ms\":%.3f,\"export_ms\":%.3f}\n",
                    argv[3], e.obj_id.size(), e.preds.size(), e.funs.size(), e.problem->get_domain()->get_actions().size(),
                    e.problem->get_problem_and_domain_axioms().size(), e.n_constraints, e.n_numeric_effects, e.n_aux_effects,
                    e.problem->get_goal_numeric_constraints().size(), std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
