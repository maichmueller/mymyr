// PDDL writer (formalism/pddl_writer.hpp).

#include "mymyr/formalism/pddl_writer.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace mymyr::formalism
{
namespace
{
/// The shortest decimal (fixed notation, no exponent) that reads back as exactly v.
std::string number(f64 v)
{
    if (!std::isfinite(v))
        throw std::invalid_argument("mymyr write_pddl: a number is not finite");
    std::array<char, 400> buf{};  // a double in fixed notation has at most 309 integral and 767 fractional digits,
                                  // but its shortest round-trip form has at most 17 significant ones
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v, std::chars_format::fixed);
    if (r.ec != std::errc{})
        throw std::invalid_argument("mymyr write_pddl: a number has no short decimal form");
    return std::string(buf.data(), r.ptr);
}

std::string requirement(std::string_view r)
{
    while (!r.empty() && r.front() == ':')
        r.remove_prefix(1);
    return std::string(r);
}

class Writer
{
public:
    explicit Writer(const TaskData& t) : T(t)
    {
        m_type_pred.assign(t.predicates.size(), false);
        m_problem_pred.assign(t.predicates.size(), false);
        for (u32 p = 0; p < t.predicates.size(); ++p)
        {
            const Predicate& P = t.predicates[p];
            const std::string_view name = t.str(P.name);
            // a task read from text has no types, but its predicates of the built-in types object and number
            if (P.kind == PredKind::Static && P.arity == 1 &&
                (name == "object" || name == "number" || either_predicate(P) ||
                 std::ranges::any_of(t.types, [&](const Type& ty) { return t.str(ty.name) == name; })))
                m_type_pred[p] = true;
            if (P.kind == PredKind::Static && P.arity == 2 && name == "=")
                m_eq = p;
        }
        for (const Axiom& a : t.axioms)
            if (a.from_problem)
                m_problem_pred[a.head.pred.v] = true;
        for (const Type& ty : t.types)
            if (const std::string_view n = t.str(ty.name); n != "object" && n != "number")
                m_typing = true;
        // constants: the objects declared so, and those the domain names (a task read from text marks none)
        m_constant.resize(t.objects.size());
        for (u32 i = 0; i < t.objects.size(); ++i)
            m_constant[i] = t.objects[i].constant;
        m_collect = true;
        (void)domain({});
        m_collect = false;
    }

    /// The domain with the requirements `req` (without the problem's goal axioms and their predicates).
    std::string domain(const std::vector<std::string>& req) const
    {
        std::string o = "(define (domain " + name_or(T.domain_name, "domain") + ")\n";
        o += "  (:requirements";
        for (const std::string& r : req)
            o += " :" + r;
        o += ")\n";
        if (m_typing)
        {
            o += "  (:types";
            for (const Type& ty : T.types)
            {
                const std::string_view n = T.str(ty.name);
                if (n == "object" || n == "number")
                    continue;
                o += " " + std::string(n);
                if (ty.bases.count > 0)
                    o += " - " + type_text(ty.bases);
            }
            o += ")\n";
        }
        std::string constants;
        for (u32 i = 0; i < T.objects.size(); ++i)
            if (m_constant[i])
                constants += " " + typed_object(i);
        if (!constants.empty())
            o += "  (:constants" + constants + ")\n";
        o += "  (:predicates";
        bool any = false;
        for (u32 p = 0; p < T.predicates.size(); ++p)
        {
            if (m_type_pred[p] || p == m_eq || m_problem_pred[p])
                continue;
            const Predicate& P = T.predicates[p];
            o += "\n    (" + std::string(T.str(P.name)) + typed_params(P.params, P.arity) + ")";
            any = true;
        }
        o += any ? "\n  )\n" : ")\n";
        std::string functions;
        for (const Function& f : T.functions)
            functions += "\n    (" + std::string(T.str(f.name)) + typed_params(f.params, f.arity) + ")";
        if (!functions.empty())
            o += "  (:functions" + functions + "\n  )\n";
        for (const Axiom& a : T.axioms)
            if (!a.from_problem)
                o += axiom(a);
        // the schemas of one name are one action (normalization splits a disjunctive precondition)
        std::vector<bool> done(T.schemas.size(), false);
        for (u32 s = 0; s < T.schemas.size(); ++s)
        {
            if (done[s])
                continue;
            std::vector<u32> group;
            for (u32 x = s; x < T.schemas.size(); ++x)
                if (!done[x] && T.str(T.schemas[x].name) == T.str(T.schemas[s].name))
                {
                    group.push_back(x);
                    done[x] = true;
                }
            o += action(group);
        }
        return o + ")\n";
    }

    /// The problem, its goal axioms written into the goal as the formula they stand for.
    std::string problem() const
    {
        std::string o = "(define (problem " + name_or(T.problem_name, "problem") + ")\n";
        o += "  (:domain " + name_or(T.domain_name, "domain") + ")\n";
        std::string objects;
        for (u32 i = 0; i < T.objects.size(); ++i)
            if (!m_constant[i])
                objects += "\n    " + typed_object(i);
        if (!objects.empty())
            o += "  (:objects" + objects + "\n  )\n";
        o += "  (:init";
        auto atoms = [&](const std::vector<GroundAtom>& v) {
            for (const GroundAtom& a : v)
            {
                if (m_type_pred[a.pred.v] || a.pred.v == m_eq)
                    continue;
                o += "\n    (" + std::string(T.str(T.predicates[a.pred.v].name));
                for (const ObjectId x : T.objects_of(a))
                    o += " " + object_name(x);
                o += ")";
            }
        };
        atoms(T.static_init);
        atoms(T.fluent_init);
        auto values = [&](const std::vector<GroundFunctionValue>& v) {
            for (const GroundFunctionValue& f : v)
            {
                o += "\n    (= (" + std::string(T.str(T.functions[f.func.v].name));
                for (u32 j = f.objects.begin; j < f.objects.end(); ++j)
                    o += " " + object_name(T.object_ids[j]);
                o += ") " + number(f.value) + ")";
            }
        };
        values(T.static_values);
        values(T.fluent_values);
        if (T.auxiliary_initial)
            for (const Function& f : T.functions)
                if (f.kind == FuncKind::Auxiliary)
                    o += "\n    (= (" + std::string(T.str(f.name)) + ") " + number(*T.auxiliary_initial) + ")";
        o += "\n  )\n";
        const Vars none;
        std::vector<std::string> goal;
        for (const Literal& l : T.literals_of(T.goal))
        {
            if (m_problem_pred[l.pred.v])
            {
                std::vector<std::string> args;
                for (const Term t : T.terms_of(l))
                    args.push_back(term(t, none));
                std::string f = inline_atom(l.pred, args, 0);
                goal.push_back(l.positive ? f : "(not " + f + ")");
            }
            else
                goal.push_back(literal(l, none));
        }
        for (const NumericConstraint& c : T.constraints_of(T.goal))
            goal.push_back(constraint(c, none));
        o += "  (:goal " + (goal.empty() ? std::string("(and)") : conjunction(goal)) + ")\n";
        if (T.metric)
            o += "  (:metric " + std::string(T.metric->minimize ? "minimize " : "maximize ") + expr(T.metric->expr, none) +
                 ")\n";
        return o + ")\n";
    }

    /// The predicate the front end adds for a parameter of type (either t1 ... tk): "either_t1_..._tk" over that type.
    bool either_predicate(const Predicate& P) const
    {
        const Range types = T.params[P.params.begin].types;
        if (types.count < 2)
            return false;
        std::string expected = "either";
        for (u32 j = types.begin; j < types.end(); ++j)
            expected += "_" + std::string(T.str(T.types[T.type_ids[j].v].name));
        return T.str(P.name) == expected;
    }

    /// The requirements of the task and those its domain and problem as written need (the goal axioms inlined into
    /// the goal), without repeats.
    std::vector<std::string> requirements() const
    {
        std::vector<std::string> req;
        auto add = [&](std::string r) {
            if (std::ranges::find(req, r) == req.end())
                req.push_back(std::move(r));
        };
        for (const std::string& r : T.requirements)
            add(requirement(r));
        add("strips");
        if (m_typing)
            add("typing");
        bool negative = false, conditional = false, existential = false, disjunctive = false;
        for (const Literal& l : T.literals)
            negative |= !l.positive;
        for (const NumericConstraint& c : T.constraints)
            negative |= c.cmp == Comparator::Ne;  // written as (not (= ...))
        for (const Schema& s : T.schemas)
        {
            existential |= s.arity() > s.original_arity;
            for (const ConditionalEffect& e : T.effects_of(s))
                conditional |= e.condition.literals.count + e.condition.constraints.count > 0 || e.extra_params.count > 0;
        }
        for (u32 s = 0; s < T.schemas.size(); ++s)
            for (u32 x = s + 1; x < T.schemas.size(); ++x)
                disjunctive |= T.str(T.schemas[s].name) == T.str(T.schemas[x].name);
        if (negative)
            add("negative-preconditions");
        if (m_eq != ~u32{0})
            add("equality");
        if (conditional)
            add("conditional-effects");
        if (existential || std::ranges::any_of(T.axioms, [](const Axiom& a) { return a.params.count > 0; }))
            add("existential-preconditions");
        if (disjunctive || std::ranges::any_of(T.axioms, &Axiom::from_problem))
            add("disjunctive-preconditions");
        if (std::ranges::any_of(T.axioms, [&](const Axiom& a) { return !a.from_problem; }))
            add("derived-predicates");
        if (std::ranges::any_of(T.functions, [](const Function& f) { return f.kind != FuncKind::Auxiliary; }))
            add("numeric-fluents");
        if (std::ranges::any_of(T.functions, [](const Function& f) { return f.kind == FuncKind::Auxiliary; }))
            add("action-costs");
        return req;
    }

private:
    using Vars = std::vector<std::string>;  // the name of each variable of a scope, by index

    const TaskData& T;
    std::vector<bool> m_type_pred, m_problem_pred;
    u32 m_eq = ~u32{0};
    bool m_typing = false;
    mutable u32 m_fresh = 0;  // fresh variables of inlined goal axioms
    mutable std::vector<bool> m_constant;  // per object: written as a constant of the domain
    mutable bool m_collect = false;        // writing the domain marks the objects it names as constants

    std::string name_or(const std::string& s, const char* fallback) const { return s.empty() ? fallback : s; }
    std::string object_name(ObjectId o) const { return std::string(T.str(T.objects[o.v].name)); }


    std::string type_text(Range types) const
    {
        if (types.count == 1)
            return std::string(T.str(T.types[T.type_ids[types.begin].v].name));
        std::string o = "(either";
        for (u32 j = types.begin; j < types.end(); ++j)
            o += " " + std::string(T.str(T.types[T.type_ids[j].v].name));
        return o + ")";
    }

    std::string typed(std::string_view name, Range types) const
    {
        std::string o(name);
        if (m_typing && types.count > 0)
            o += " - " + type_text(types);
        return o;
    }

    std::string typed_object(u32 i) const { return typed(T.str(T.objects[i].name), T.objects[i].types); }

    std::string var_name(u32 param, u32 index) const
    {
        const std::string_view n = T.str(T.params[param].name);
        return n.empty() ? "?x" + std::to_string(index) : "?" + std::string(n);
    }

    /// " ?a - t ?b - t" for the parameters [range.begin, range.begin + count)
    std::string typed_params(Range params, u32 count) const
    {
        std::string o;
        for (u32 i = 0; i < count; ++i)
        {
            if (i >= params.count)
            {
                o += " ?x" + std::to_string(i);
                continue;
            }
            const Parameter& p = T.params[params.begin + i];
            o += " " + typed(var_name(params.begin + i, i), p.types);
        }
        return o;
    }

    std::string typed_vars(const Vars& vars, u32 from, u32 to, Range params, u32 param_offset) const
    {
        std::string o;
        for (u32 i = from; i < to; ++i)
        {
            if (!o.empty())
                o += " ";
            o += typed(vars[i], T.params[params.begin + i - param_offset].types);
        }
        return o;
    }

    std::string term(Term t, const Vars& vars) const
    {
        if (is_object(t))
        {
            if (m_collect)
                m_constant[term_object(t).v] = true;
            return object_name(term_object(t));
        }
        const u32 i = term_parameter(t);
        return i < vars.size() ? vars[i] : "?x" + std::to_string(i);
    }

    std::string atom(PredicateId p, std::span<const Term> terms, const Vars& vars) const
    {
        std::string o = "(" + (p.v == m_eq ? std::string("=") : std::string(T.str(T.predicates[p.v].name)));
        for (const Term t : terms)
            o += " " + term(t, vars);
        return o + ")";
    }

    std::string literal(const Literal& l, const Vars& vars) const
    {
        const std::string a = atom(l.pred, T.terms_of(l), vars);
        return l.positive ? a : "(not " + a + ")";
    }

    /// A type literal over one of the variables [from, to) of its scope: normalization adds it again from the
    /// variable's declared type, and a type is no predicate in PDDL.
    bool type_literal(const Literal& l, u32 from, u32 to) const
    {
        if (!m_type_pred[l.pred.v])
            return false;
        const Term t = T.terms[l.terms.begin];
        return !is_object(t) && term_parameter(t) >= from && term_parameter(t) < to;
    }

    std::string expr(u32 e, const Vars& vars) const
    {
        const Expr& x = T.exprs[e];
        switch (x.op)
        {
            case ExprOp::Number: return number(x.value);
            case ExprOp::Function:
            {
                std::string o = "(" + std::string(T.str(T.functions[x.func.v].name));
                for (u32 j = x.terms.begin; j < x.terms.end(); ++j)
                    o += " " + term(T.terms[j], vars);
                return o + ")";
            }
            case ExprOp::Add: return "(+ " + expr(x.a, vars) + " " + expr(x.b, vars) + ")";
            case ExprOp::Sub: return "(- " + expr(x.a, vars) + " " + expr(x.b, vars) + ")";
            case ExprOp::Mul: return "(* " + expr(x.a, vars) + " " + expr(x.b, vars) + ")";
            case ExprOp::Div: return "(/ " + expr(x.a, vars) + " " + expr(x.b, vars) + ")";
            case ExprOp::Neg: return "(- " + expr(x.a, vars) + ")";
        }
        throw std::invalid_argument("mymyr write_pddl: an expression of unknown kind");
    }

    std::string constraint(const NumericConstraint& c, const Vars& vars) const
    {
        const std::string l = expr(c.lhs, vars), r = expr(c.rhs, vars);
        switch (c.cmp)
        {
            case Comparator::Eq: return "(= " + l + " " + r + ")";
            case Comparator::Ne: return "(not (= " + l + " " + r + "))";
            case Comparator::Lt: return "(< " + l + " " + r + ")";
            case Comparator::Le: return "(<= " + l + " " + r + ")";
            case Comparator::Gt: return "(> " + l + " " + r + ")";
            case Comparator::Ge: return "(>= " + l + " " + r + ")";
        }
        throw std::invalid_argument("mymyr write_pddl: a comparison of unknown kind");
    }

    static std::string conjunction(const std::vector<std::string>& parts)
    {
        if (parts.size() == 1)
            return parts[0];
        std::string o = "(and";
        for (const std::string& p : parts)
            o += " " + p;
        return o + ")";
    }

    /// The literals and constraints of c, without the type literals over the variables [from, to) and without
    /// repeats (the first occurrence stays).
    std::vector<std::string> condition(const Condition& c, const Vars& vars, u32 from, u32 to) const
    {
        std::vector<std::string> parts;
        auto add = [&](std::string p)
        {
            if (std::find(parts.begin(), parts.end(), p) == parts.end())
                parts.push_back(std::move(p));
        };
        for (const Literal& l : T.literals_of(c))
            if (!type_literal(l, from, to))
                add(literal(l, vars));
        for (const NumericConstraint& k : T.constraints_of(c))
            add(constraint(k, vars));
        return parts;
    }

    std::string numeric_effect(const NumericEffect& e, const Vars& vars) const
    {
        static constexpr std::array<const char*, 5> ops{"assign", "increase", "decrease", "scale-up", "scale-down"};
        std::string f = "(" + std::string(T.str(T.functions[e.func.v].name));
        for (u32 j = e.terms.begin; j < e.terms.end(); ++j)
            f += " " + term(T.terms[j], vars);
        f += ")";
        return "(" + std::string(ops[static_cast<usize>(e.op)]) + " " + f + " " + expr(e.expr, vars) + ")";
    }

    /// The literal and numeric effects of e over `vars`.
    std::vector<std::string> effect_parts(const ConditionalEffect& e, const Vars& vars) const
    {
        std::vector<std::string> parts;
        for (u32 j = e.effects.begin; j < e.effects.end(); ++j)
            parts.push_back(literal(T.literals[j], vars));
        for (u32 j = e.numeric_effects.begin; j < e.numeric_effects.end(); ++j)
            parts.push_back(numeric_effect(T.numeric_effects[j], vars));
        if (e.auxiliary)
            parts.push_back(numeric_effect(*e.auxiliary, vars));
        return parts;
    }

    static bool unconditional(const ConditionalEffect& e)
    {
        return e.extra_params.count == 0 && e.condition.literals.count == 0 && e.condition.constraints.count == 0;
    }

    std::string effect(const ConditionalEffect& e, Vars vars) const
    {
        const u32 from = static_cast<u32>(vars.size());
        for (u32 j = 0; j < e.extra_params.count; ++j)
            vars.push_back(var_name(e.extra_params.begin + j, from + j));
        const std::vector<std::string> parts = effect_parts(e, vars);
        std::string o = parts.empty() ? "(and)" : conjunction(parts);
        const std::vector<std::string> cond = condition(e.condition, vars, from, static_cast<u32>(vars.size()));
        if (!cond.empty())
            o = "(when " + conjunction(cond) + " " + o + ")";
        if (e.extra_params.count > 0)
            o = "(forall (" + typed_vars(vars, from, static_cast<u32>(vars.size()), e.extra_params, from) + ") " + o +
                ")";
        return o;
    }

    Vars schema_vars(const Schema& s) const
    {
        Vars vars;
        for (u32 i = 0; i < s.arity(); ++i)
            vars.push_back(var_name(s.params.begin + i, i));
        return vars;
    }

    std::string precondition(const Schema& s) const
    {
        const Vars vars = schema_vars(s);
        const std::vector<std::string> parts = condition(s.precondition, vars, 0, s.arity());
        std::string o = parts.empty() ? "(and)" : conjunction(parts);
        if (s.arity() > s.original_arity)
            o = "(exists (" + typed_vars(vars, s.original_arity, s.arity(), s.params, 0) + ") " + o + ")";
        return o;
    }

    std::string effects(const Schema& s) const
    {
        const Vars vars = schema_vars(s);
        // In reverse order: the front end lists an action's effects in the reverse of their order in the PDDL, so
        // that the written action reads back with its effects in this order.
        std::vector<std::string> parts;
        const std::span<const ConditionalEffect> effs = T.effects_of(s);
        for (const ConditionalEffect& e : std::views::reverse(effs))
        {
            if (unconditional(e))  // its effects at the top level, as the action declared them
            {
                for (std::string& x : effect_parts(e, vars))
                    parts.push_back(std::move(x));
            }
            else
                parts.push_back(effect(e, vars));
        }
        return parts.empty() ? "(and)" : conjunction(parts);
    }

    std::string action(const std::vector<u32>& group) const
    {
        const Schema& s = T.schemas[group[0]];
        const Vars vars = schema_vars(s);
        const std::string params = typed_vars(vars, 0, s.original_arity, s.params, 0);
        const std::string eff = effects(s);
        std::vector<std::string> pre;
        for (const u32 x : group)
        {
            const Schema& o = T.schemas[x];
            if (o.original_arity != s.original_arity ||
                typed_vars(schema_vars(o), 0, o.original_arity, o.params, 0) != params || effects(o) != eff)
                throw std::invalid_argument("mymyr write_pddl: the schemas named '" + std::string(T.str(s.name)) +
                                            "' differ in their parameters or effects: they are no split action");
            pre.push_back(precondition(o));
        }
        std::string p;
        if (pre.size() == 1)
            p = pre[0];
        else
        {
            p = "(or";
            for (const std::string& x : pre)
                p += " " + x;
            p += ")";
        }
        return "  (:action " + std::string(T.str(s.name)) + "\n    :parameters (" + params + ")\n    :precondition " + p +
               "\n    :effect " + eff + ")\n";
    }

    /// The head variables of axiom a (its head's terms, each a distinct parameter) and the order of its variables:
    /// head variables first, then the others.
    std::string axiom(const Axiom& a) const
    {
        Vars vars;
        for (u32 i = 0; i < a.params.count; ++i)
            vars.push_back(var_name(a.params.begin + i, i));
        std::vector<bool> in_head(a.params.count, false);
        std::string head = "(" + std::string(T.str(T.predicates[a.head.pred.v].name));
        for (const Term t : T.terms_of(a.head))
        {
            if (is_object(t) || term_parameter(t) >= a.params.count || in_head[term_parameter(t)])
                throw std::invalid_argument("mymyr write_pddl: the head of a rule for '" +
                                            std::string(T.str(T.predicates[a.head.pred.v].name)) +
                                            "' is not a list of distinct variables");
            const u32 i = term_parameter(t);
            in_head[i] = true;
            head += " " + typed(vars[i], T.params[a.params.begin + i].types);
        }
        head += ")";
        const std::vector<std::string> parts = condition(a.body, vars, 0, a.params.count);
        std::string body = parts.empty() ? "(and)" : conjunction(parts);
        std::string ex;
        for (u32 i = 0; i < a.params.count; ++i)
            if (!in_head[i])
                ex += (ex.empty() ? "" : " ") + typed(vars[i], T.params[a.params.begin + i].types);
        if (!ex.empty())
            body = "(exists (" + ex + ") " + body + ")";
        return "  (:derived " + head + "\n    " + body + ")\n";
    }

    /// The formula a goal axiom's predicate stands for under `args` (one term text per head variable): the
    /// disjunction over its rules, each an exists over its other variables.
    std::string inline_atom(PredicateId pred, const std::vector<std::string>& args, u32 depth) const
    {
        if (depth > T.axioms.size())
            throw std::invalid_argument("mymyr write_pddl: the goal axioms are recursive");
        std::vector<std::string> disjuncts;
        for (const Axiom& a : T.axioms)
        {
            if (!a.from_problem || a.head.pred != pred)
                continue;
            Vars vars(a.params.count);
            std::vector<bool> bound(a.params.count, false);
            const std::span<const Term> head = T.terms_of(a.head);
            for (usize j = 0; j < head.size() && j < args.size(); ++j)
                if (!is_object(head[j]) && term_parameter(head[j]) < a.params.count)
                {
                    vars[term_parameter(head[j])] = args[j];
                    bound[term_parameter(head[j])] = true;
                }
            std::string ex;
            for (u32 i = 0; i < a.params.count; ++i)
                if (!bound[i])
                {
                    vars[i] = "?g" + std::to_string(m_fresh++);
                    ex += (ex.empty() ? "" : " ") + typed(vars[i], T.params[a.params.begin + i].types);
                }
            std::vector<std::string> parts;
            std::vector<std::string> seen;  // the body's literals over the rule's variables, each written once
            for (const Literal& l : T.literals_of(a.body))
            {
                std::string key = literal(l, vars);
                if (std::find(seen.begin(), seen.end(), key) != seen.end())
                    continue;
                seen.push_back(std::move(key));
                if (m_type_pred[l.pred.v])
                {
                    const Term t = T.terms[l.terms.begin];
                    if (!is_object(t))
                        continue;  // the variable's declared type, or a head variable's object of that type
                }
                if (m_problem_pred[l.pred.v])
                {
                    std::vector<std::string> inner;
                    for (const Term t : T.terms_of(l))
                        inner.push_back(term(t, vars));
                    const std::string f = inline_atom(l.pred, inner, depth + 1);
                    parts.push_back(l.positive ? f : "(not " + f + ")");
                }
                else
                    parts.push_back(literal(l, vars));
            }
            for (const NumericConstraint& k : T.constraints_of(a.body))
                parts.push_back(constraint(k, vars));
            std::string body = parts.empty() ? "(and)" : conjunction(parts);
            if (!ex.empty())
                body = "(exists (" + ex + ") " + body + ")";
            disjuncts.push_back(std::move(body));
        }
        if (disjuncts.empty())
            return "(or)";
        if (disjuncts.size() == 1)
            return disjuncts[0];
        std::string o = "(or";
        for (const std::string& d : disjuncts)
            o += " " + d;
        return o + ")";
    }
};
}  // namespace

PddlFiles write_pddl(const TaskData& task)
{
    const TaskData* const one[1] = {&task};
    return write_pddl(std::span<const TaskData* const>(one));
}

PddlFiles write_pddl(std::span<const TaskData* const> tasks)
{
    if (tasks.empty())
        throw std::invalid_argument("mymyr write_pddl: no tasks");
    // one domain: the tasks' domains agree but for their requirements, which it takes all of
    PddlFiles out;
    std::vector<std::string> req;
    std::string body;
    for (usize i = 0; i < tasks.size(); ++i)
    {
        const Writer w(*tasks[i]);
        for (std::string& r : w.requirements())
            if (std::ranges::find(req, r) == req.end())
                req.push_back(std::move(r));
        std::string d = w.domain({});
        if (i == 0)
            body = std::move(d);
        else if (d != body)
            throw std::invalid_argument("mymyr write_pddl: task " + std::to_string(i) + " (" + tasks[i]->problem_name +
                                        ") has another domain than task 0 (" + tasks[0]->problem_name + ")");
        out.problems.push_back(w.problem());
    }
    out.domain = Writer(*tasks[0]).domain(req);
    return out;
}
}  // namespace mymyr::formalism
