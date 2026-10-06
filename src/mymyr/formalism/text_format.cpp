#include "mymyr/formalism/text_format.hpp"

#include <charconv>
#include <cstdio>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::formalism
{
namespace
{
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("task text: " + msg); }

class Reader
{
public:
    explicit Reader(std::istream& in) : m_in(in) {}

    TaskData read()
    {
        expect("O");
        const u32 n = u();
        for (u32 i = 0; i < n; ++i)
        {
            Object o;
            o.name = t.intern_string("o" + std::to_string(i));
            t.objects.push_back(o);
        }
        expect("P");
        const u32 np = u();
        for (u32 i = 0; i < np; ++i)
        {
            const std::string kind = word();
            Predicate p;
            p.kind = kind == "S" ? PredKind::Static : kind == "F" ? PredKind::Fluent : kind == "D" ? PredKind::Derived
                                                                                                   : (fail("bad predicate kind " + kind), PredKind::Static);
            p.arity = u();
            p.name = t.intern_string(word());
            p.params = anonymous_params(p.arity);
            t.predicates.push_back(p);
        }
        std::string tag = word();
        m_numeric = tag == "N";
        if (m_numeric)
        {
            const u32 nf = u();
            for (u32 i = 0; i < nf; ++i)
            {
                const std::string kind = word();
                Function f;
                f.kind = kind == "S" ? FuncKind::Static : kind == "F" ? FuncKind::Fluent : kind == "A" ? FuncKind::Auxiliary
                                                                                                       : (fail("bad function kind " + kind), FuncKind::Static);
                f.arity = u();
                f.name = t.intern_string(word());
                f.params = anonymous_params(f.arity);
                t.functions.push_back(f);
            }
            tag = word();
        }
        if (tag != "SI")
            fail("expected SI, got " + tag);
        read_atoms_body(t.static_init);
        expect("FI");
        read_atoms_body(t.fluent_init);
        if (m_numeric)
        {
            expect("NI");
            const u32 k = u();
            for (u32 i = 0; i < k; ++i)
            {
                GroundFunctionValue v;
                v.func = FunctionId{u()};
                const Function& f = t.functions.at(v.func.v);
                const u32 ar = f.kind == FuncKind::Auxiliary ? 0 : f.arity;
                std::vector<ObjectId> objs(ar);
                for (auto& o : objs)
                    o = term_object(i32v());
                v.objects = TaskData::append<ObjectId>(t.object_ids, objs);
                v.value = number();
                if (f.kind == FuncKind::Auxiliary)
                    t.auxiliary_initial = v.value;
                else
                    (f.kind == FuncKind::Static ? t.static_values : t.fluent_values).push_back(v);
            }
        }
        expect("G");
        {
            const u32 k = u();
            std::vector<Literal> goal;
            for (u32 i = 0; i < k; ++i)
                goal.push_back(literal(/*ground=*/true));
            t.goal.literals = TaskData::append<Literal>(t.literals, goal);
        }
        if (m_numeric)
        {
            t.goal.constraints = constraints("NG");
            expect("M");
            const std::string m = word();
            if (m != "none")
            {
                Metric metric;
                metric.minimize = m == "min";
                metric.expr = expr();
                t.metric = metric;
            }
        }
        expect("A");
        const u32 ns = u();
        for (u32 i = 0; i < ns; ++i)
            t.schemas.push_back(schema());
        if (m_in >> tag)
        {
            if (tag != "X")
                fail("expected X, got " + tag);
            const u32 nx = u();
            for (u32 i = 0; i < nx; ++i)
            {
                Axiom a;
                a.head.pred = PredicateId{u()};
                a.head.positive = true;
                const u32 arity = u();
                a.params = anonymous_params(arity);
                expect("H");
                std::vector<Term> head(t.predicates.at(a.head.pred.v).arity);
                for (auto& x : head)
                    x = i32v();
                a.head.terms = TaskData::append<Term>(t.terms, head);
                a.body = condition();
                t.axioms.push_back(a);
            }
        }
        validate(t);
        return std::move(t);
    }

private:
    void expect(const char* s)
    {
        const std::string w = word();
        if (w != s)
            fail(std::string("expected ") + s + ", got '" + w + "'");
    }
    std::string word()
    {
        std::string w;
        if (!(m_in >> w))
            fail("unexpected end of input");
        return w;
    }
    u32 u()
    {
        const std::string w = word();
        u32 v = 0;
        auto [p, ec] = std::from_chars(w.data(), w.data() + w.size(), v);
        if (ec != std::errc{} || p != w.data() + w.size())
            fail("expected unsigned integer, got '" + w + "'");
        return v;
    }
    i32 i32v()
    {
        const std::string w = word();
        i32 v = 0;
        auto [p, ec] = std::from_chars(w.data(), w.data() + w.size(), v);
        if (ec != std::errc{} || p != w.data() + w.size())
            fail("expected integer, got '" + w + "'");
        return v;
    }
    f64 number()
    {
        const std::string w = word();
        char* end = nullptr;
        const f64 v = std::strtod(w.c_str(), &end);
        if (end != w.c_str() + w.size())
            fail("expected number, got '" + w + "'");
        return v;
    }

    Range anonymous_params(u32 n)
    {
        std::vector<Parameter> ps(n);
        for (u32 i = 0; i < n; ++i)
            ps[i].name = t.intern_string("x" + std::to_string(i));
        return TaskData::append<Parameter>(t.params, ps);
    }

    void read_atoms_body(std::vector<GroundAtom>& out)
    {
        const u32 k = u();
        for (u32 i = 0; i < k; ++i)
        {
            GroundAtom a;
            a.pred = PredicateId{u()};
            std::vector<ObjectId> objs(t.predicates.at(a.pred.v).arity);
            for (auto& o : objs)
                o = ObjectId{u()};
            a.objects = TaskData::append<ObjectId>(t.object_ids, objs);
            out.push_back(a);
        }
    }

    Literal literal(bool ground)
    {
        Literal l;
        l.pred = PredicateId{u()};
        l.positive = u() != 0;
        std::vector<Term> ts(t.predicates.at(l.pred.v).arity);
        for (auto& x : ts)
            x = ground ? object_term(ObjectId{u()}) : i32v();
        l.terms = TaskData::append<Term>(t.terms, ts);
        return l;
    }

    Range literals(const char* tag)
    {
        expect(tag);
        const u32 k = u();
        std::vector<Literal> ls;
        for (u32 i = 0; i < k; ++i)
            ls.push_back(literal(false));
        return TaskData::append<Literal>(t.literals, ls);
    }

    Condition condition()
    {
        Condition c;
        c.literals = literals("L");
        if (m_numeric)
            c.constraints = constraints("NC");
        return c;
    }

    // postfix token count followed by tokens; returns the root expr index
    u32 expr()
    {
        const u32 nt = u();
        std::vector<u32> stack;
        for (u32 i = 0; i < nt; ++i)
        {
            const std::string s = word();
            Expr e;
            if (s == "c")
            {
                e.op = ExprOp::Number;
                e.value = number();
            }
            else if (s == "s" || s == "f" || s == "a")
            {
                e.op = ExprOp::Function;
                e.func = FunctionId{u()};
                const u32 ar = s == "a" ? 0 : t.functions.at(e.func.v).arity;
                std::vector<Term> ts(ar);
                for (auto& x : ts)
                    x = i32v();
                e.terms = TaskData::append<Term>(t.terms, ts);
            }
            else if (s == "+" || s == "-" || s == "*" || s == "/")
            {
                if (stack.size() < 2)
                    fail("expression stack underflow");
                e.op = s == "+" ? ExprOp::Add : s == "-" ? ExprOp::Sub : s == "*" ? ExprOp::Mul : ExprOp::Div;
                e.b = stack.back();
                stack.pop_back();
                e.a = stack.back();
                stack.pop_back();
            }
            else if (s == "u")
            {
                if (stack.empty())
                    fail("expression stack underflow");
                e.op = ExprOp::Neg;
                e.a = stack.back();
                stack.pop_back();
            }
            else
                fail("bad expression token '" + s + "'");
            stack.push_back(static_cast<u32>(t.exprs.size()));
            t.exprs.push_back(e);
        }
        if (stack.size() != 1)
            fail("malformed expression");
        return stack.back();
    }

    Comparator comparator()
    {
        const std::string s = word();
        if (s == "=") return Comparator::Eq;
        if (s == "!=") return Comparator::Ne;
        if (s == "<") return Comparator::Lt;
        if (s == "<=") return Comparator::Le;
        if (s == ">") return Comparator::Gt;
        if (s == ">=") return Comparator::Ge;
        fail("bad comparator '" + s + "'");
    }

    AssignOp assign_op(const std::string& s)
    {
        if (s == "assign") return AssignOp::Assign;
        if (s == "increase") return AssignOp::Increase;
        if (s == "decrease") return AssignOp::Decrease;
        if (s == "scale-up") return AssignOp::ScaleUp;
        if (s == "scale-down") return AssignOp::ScaleDown;
        fail("bad assign operator '" + s + "'");
    }

    Range constraints(const char* tag)
    {
        expect(tag);
        const u32 k = u();
        std::vector<NumericConstraint> cs;
        for (u32 i = 0; i < k; ++i)
        {
            NumericConstraint c;
            c.cmp = comparator();
            c.lhs = expr();
            c.rhs = expr();
            cs.push_back(c);
        }
        return TaskData::append<NumericConstraint>(t.constraints, cs);
    }

    Schema schema()
    {
        Schema s;
        s.name = t.intern_string(word());
        const u32 arity = u();
        s.original_arity = arity;  // the format does not record which parameters normalization introduced
        s.params = anonymous_params(arity);
        s.precondition = condition();
        expect("E");
        const u32 ne = u();
        std::vector<ConditionalEffect> ces;
        for (u32 i = 0; i < ne; ++i)
        {
            ConditionalEffect ce;
            const u32 extra = u();
            std::vector<Parameter> ps(extra);
            for (u32 j = 0; j < extra; ++j)
                ps[j].name = t.intern_string("x" + std::to_string(arity + j));
            ce.extra_params = TaskData::append<Parameter>(t.params, ps);
            ce.condition = condition();
            ce.effects = literals("F");
            if (m_numeric)
            {
                expect("NE");
                const u32 nn = u();
                std::vector<NumericEffect> nes;
                for (u32 j = 0; j < nn; ++j)
                {
                    NumericEffect e;
                    e.op = assign_op(word());
                    e.func = FunctionId{u()};
                    std::vector<Term> ts(t.functions.at(e.func.v).arity);
                    for (auto& x : ts)
                        x = i32v();
                    e.terms = TaskData::append<Term>(t.terms, ts);
                    e.expr = expr();
                    nes.push_back(e);
                }
                ce.numeric_effects = TaskData::append<NumericEffect>(t.numeric_effects, nes);
                expect("NA");
                const std::string op = word();
                if (op != "-")
                {
                    NumericEffect aux;
                    aux.op = assign_op(op);
                    for (u32 f = 0; f < t.functions.size(); ++f)
                        if (t.functions[f].kind == FuncKind::Auxiliary)
                            aux.func = FunctionId{f};
                    aux.expr = expr();
                    ce.auxiliary = aux;
                }
            }
            ces.push_back(ce);
        }
        s.effects = TaskData::append<ConditionalEffect>(t.conditional_effects, ces);
        return s;
    }

    std::istream& m_in;
    TaskData t;
    bool m_numeric = false;
};

// ------------------------------------------------------------------------------------------------ writer

class Writer
{
public:
    explicit Writer(const TaskData& t) : t(t), m_numeric(t.has_numerics() || t.metric.has_value()) {}

    std::string write()
    {
        o << "O " << t.objects.size() << "\nP " << t.predicates.size() << "\n";
        for (const auto& p : t.predicates)
            o << (p.kind == PredKind::Static ? 'S' : p.kind == PredKind::Fluent ? 'F' : 'D') << " " << p.arity << " "
              << t.str(p.name) << "\n";
        if (m_numeric)
        {
            o << "N " << t.functions.size() << "\n";
            for (const auto& f : t.functions)
                o << (f.kind == FuncKind::Static ? 'S' : f.kind == FuncKind::Fluent ? 'F' : 'A') << " " << f.arity << " "
                  << t.str(f.name) << "\n";
        }
        atoms("SI", t.static_init);
        atoms("FI", t.fluent_init);
        if (m_numeric)
        {
            std::vector<const GroundFunctionValue*> vals;
            for (const auto& v : t.static_values)
                vals.push_back(&v);
            for (const auto& v : t.fluent_values)
                vals.push_back(&v);
            u32 aux_fid = ~0u;
            for (u32 f = 0; f < t.functions.size(); ++f)
                if (t.functions[f].kind == FuncKind::Auxiliary)
                    aux_fid = f;
            const bool aux = t.auxiliary_initial.has_value() && aux_fid != ~0u;
            o << "NI " << vals.size() + (aux ? 1 : 0) << "\n";
            for (const auto* v : vals)
            {
                o << v->func.v;
                for (ObjectId x : TaskData::slice(t.object_ids, v->objects))
                    o << " " << object_term(x);
                o << " " << num(v->value) << "\n";
            }
            if (aux)
                o << aux_fid << " " << num(*t.auxiliary_initial) << "\n";
        }
        o << "G " << t.goal.literals.count << "\n";
        for (const auto& l : t.literals_of(t.goal))
        {
            o << l.pred.v << " " << (l.positive ? 1 : 0);
            for (Term x : t.terms_of(l))
                o << " " << term_object(x).v;
            o << "\n";
        }
        if (m_numeric)
        {
            constraints("NG", t.goal.constraints);
            if (t.metric)
            {
                o << "M " << (t.metric->minimize ? "min" : "max") << " ";
                expr(t.metric->expr);
                o << "\n";
            }
            else
                o << "M none\n";
        }
        o << "A " << t.schemas.size() << "\n";
        for (const auto& s : t.schemas)
        {
            o << t.str(s.name) << " " << s.arity() << "\n";
            condition(s.precondition);
            o << "E " << s.effects.count << "\n";
            for (const auto& ce : t.effects_of(s))
            {
                o << ce.extra_params.count << "\n";
                condition(ce.condition);
                lits("F", ce.effects);
                if (m_numeric)
                {
                    o << "NE " << ce.numeric_effects.count << "\n";
                    for (const auto& e : TaskData::slice(t.numeric_effects, ce.numeric_effects))
                    {
                        o << assign_op(e.op) << " " << e.func.v;
                        for (Term x : TaskData::slice(t.terms, e.terms))
                            o << " " << x;
                        o << " ";
                        expr(e.expr);
                        o << "\n";
                    }
                    if (ce.auxiliary)
                    {
                        o << "NA " << assign_op(ce.auxiliary->op) << " ";
                        expr(ce.auxiliary->expr);
                        o << "\n";
                    }
                    else
                        o << "NA -\n";
                }
            }
        }
        if (m_numeric || !t.axioms.empty())
        {
            o << "X " << t.axioms.size() << "\n";
            for (const auto& a : t.axioms)
            {
                o << a.head.pred.v << " " << a.params.count << "\nH";
                for (Term x : t.terms_of(a.head))
                    o << " " << x;
                o << "\n";
                condition(a.body);
            }
        }
        return o.str();
    }

private:
    static std::string num(f64 v)
    {
        char b[64];
        std::snprintf(b, sizeof b, "%.17g", v);
        return b;
    }
    static const char* assign_op(AssignOp op)
    {
        switch (op)
        {
            case AssignOp::Assign: return "assign";
            case AssignOp::Increase: return "increase";
            case AssignOp::Decrease: return "decrease";
            case AssignOp::ScaleUp: return "scale-up";
            case AssignOp::ScaleDown: return "scale-down";
        }
        return "?";
    }
    static const char* comparator(Comparator c)
    {
        switch (c)
        {
            case Comparator::Eq: return "=";
            case Comparator::Ne: return "!=";
            case Comparator::Lt: return "<";
            case Comparator::Le: return "<=";
            case Comparator::Gt: return ">";
            case Comparator::Ge: return ">=";
        }
        return "?";
    }
    void atoms(const char* tag, const std::vector<GroundAtom>& as)
    {
        o << tag << " " << as.size() << "\n";
        for (const auto& a : as)
        {
            o << a.pred.v;
            for (ObjectId x : t.objects_of(a))
                o << " " << x.v;
            o << "\n";
        }
    }
    void lits(const char* tag, Range r)
    {
        o << tag << " " << r.count << "\n";
        for (const auto& l : TaskData::slice(t.literals, r))
        {
            o << l.pred.v << " " << (l.positive ? 1 : 0);
            for (Term x : t.terms_of(l))
                o << " " << x;
            o << "\n";
        }
    }
    void condition(const Condition& c)
    {
        lits("L", c.literals);
        if (m_numeric)
            constraints("NC", c.constraints);
    }
    void constraints(const char* tag, Range r)
    {
        o << tag << " " << r.count << "\n";
        for (const auto& c : TaskData::slice(t.constraints, r))
        {
            o << comparator(c.cmp) << " ";
            expr(c.lhs);
            o << " ";
            expr(c.rhs);
            o << "\n";
        }
    }
    void postfix(u32 e, std::vector<std::string>& toks) const
    {
        const Expr& x = t.exprs[e];
        switch (x.op)
        {
            case ExprOp::Number: toks.push_back("c " + num(x.value)); return;
            case ExprOp::Function:
            {
                const Function& f = t.functions[x.func.v];
                std::string s = f.kind == FuncKind::Static ? "s " : f.kind == FuncKind::Fluent ? "f " : "a ";
                s += std::to_string(x.func.v);
                if (f.kind != FuncKind::Auxiliary)
                    for (Term tt : TaskData::slice(t.terms, x.terms))
                        s += " " + std::to_string(tt);
                toks.push_back(std::move(s));
                return;
            }
            case ExprOp::Neg:
                postfix(x.a, toks);
                toks.push_back("u");
                return;
            default:
                postfix(x.a, toks);
                postfix(x.b, toks);
                toks.push_back(x.op == ExprOp::Add ? "+" : x.op == ExprOp::Sub ? "-" : x.op == ExprOp::Mul ? "*" : "/");
        }
    }
    void expr(u32 e)
    {
        std::vector<std::string> toks;
        postfix(e, toks);
        o << toks.size();
        for (const auto& s : toks)
            o << " " << s;
    }

    const TaskData& t;
    bool m_numeric;
    std::ostringstream o;
};
}  // namespace

TaskData read_task_text(std::istream& in) { return Reader(in).read(); }

TaskData read_task_text_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("task text: cannot open " + path);
    return read_task_text(in);
}

std::string write_task_text(const TaskData& t) { return Writer(t).write(); }
}  // namespace mymyr::formalism
