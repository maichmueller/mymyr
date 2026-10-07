// The formulas of mymyr.formalism (py_formula.hpp): variables, atoms, literals and conjunctive conditions, lifted and
// ground, and the expression views of numeric constraints.

#include "py_formula.hpp"

#include "mymyr/successor/bindings.hpp"
#include "mymyr/task/workspace.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <bit>
#include <charconv>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace mymyr::python
{
namespace fm = formalism;
using formalism::TaskData;

namespace
{
// ------------------------------------------------------------------------------------------------ text and hashing

const char* kind_name(fm::PredKind k) { return k == fm::PredKind::Static ? "static" : k == fm::PredKind::Fluent ? "fluent" : "derived"; }

const char* cmp_name(fm::Comparator c)
{
    switch (c)
    {
        case fm::Comparator::Eq: return "=";
        case fm::Comparator::Ne: return "!=";
        case fm::Comparator::Lt: return "<";
        case fm::Comparator::Le: return "<=";
        case fm::Comparator::Gt: return ">";
        case fm::Comparator::Ge: return ">=";
    }
    return "?";
}

std::string number_text(f64 v)
{
    std::array<char, 64> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v);
    return std::string(buf.data(), r.ptr);
}

std::string name_text(const TaskData& d, fm::Str s) { return std::string(d.str(s)); }

std::string variable_text(const Variables* vars, u32 v)
{
    if (vars && v < vars->size() && !(*vars)[v].name.empty())
        return "?" + (*vars)[v].name;
    return "?x" + std::to_string(v);
}

std::string term_text(const TaskData& d, fm::Term x, const Variables* vars)
{
    if (fm::is_object(x))
        return name_text(d, d.objects[fm::term_object(x).v].name);
    return variable_text(vars, fm::term_parameter(x));
}

std::string expr_text(const ExprPool& p, u32 e)
{
    const TaskData& d = *p.o.data;
    const fm::Expr& x = (*p.exprs)[e];
    switch (x.op)
    {
        case fm::ExprOp::Number: return number_text(x.value);
        case fm::ExprOp::Function:
        {
            std::string out = "(" + name_text(d, d.functions[x.func.v].name);
            for (u32 j = x.terms.begin; j < x.terms.end(); ++j)
                out += " " + term_text(d, (*p.terms)[j], p.vars.get());
            return out + ")";
        }
        case fm::ExprOp::Neg: return "(- " + expr_text(p, x.a) + ")";
        default:
        {
            const char* op = x.op == fm::ExprOp::Add ? "+" : x.op == fm::ExprOp::Sub ? "-" : x.op == fm::ExprOp::Mul ? "*" : "/";
            return std::string("(") + op + " " + expr_text(p, x.a) + " " + expr_text(p, x.b) + ")";
        }
    }
}

std::string constraint_text(const ExprPool& p, u32 i)
{
    const fm::NumericConstraint& k = (*p.constraints)[i];
    return std::string("(") + cmp_name(k.cmp) + " " + expr_text(p, k.lhs) + " " + expr_text(p, k.rhs) + ")";
}

u64 mix(u64 h, u64 x) { return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)); }
nb::ssize_t to_hash(u64 h)
{
    const auto v = static_cast<nb::ssize_t>(h);
    return v == -1 ? -2 : v;
}
u64 data_hash(const FormulaOwner& o) { return std::hash<const void*>{}(o.data.get()); }

/// A term for equality and hashing of lifted atoms: an object, or a variable by name.
std::string term_key(fm::Term x, const Variables* vars)
{
    if (fm::is_object(x))
        return "#" + std::to_string(fm::term_object(x).v);
    return variable_text(vars, fm::term_parameter(x));
}

bool same_atom(const PyLiftedAtom& a, const PyLiftedAtom& b)
{
    if (a.o.data.get() != b.o.data.get() || a.predicate != b.predicate || a.terms.size() != b.terms.size())
        return false;
    for (usize i = 0; i < a.terms.size(); ++i)
        if (term_key(a.terms[i], a.vars.get()) != term_key(b.terms[i], b.vars.get()))
            return false;
    return true;
}

u64 atom_hash(const PyLiftedAtom& a)
{
    u64 h = mix(data_hash(a.o), a.predicate.v);
    for (fm::Term x : a.terms)
        h = mix(h, std::hash<std::string>{}(term_key(x, a.vars.get())));
    return h;
}

u64 ground_atom_hash(const GroundAtom& a)
{
    u64 h = mix(0x51ed27, a.predicate.v);
    for (ObjectId o : a.objects)
        h = mix(h, o.v);
    return h;
}

// ------------------------------------------------------------------------------------------------ views

nb::tuple type_tuple(const DataPtr& t, std::span<const TypeId> types)
{
    nb::list l;
    for (TypeId ty : types)
        l.append(TypeView{{t, ty.v}});
    return nb::tuple(l);
}

nb::tuple object_tuple(const DataPtr& t, std::span<const ObjectId> objects)
{
    nb::list l;
    for (ObjectId o : objects)
        l.append(ObjectView{{t, o.v}});
    return nb::tuple(l);
}

nb::tuple term_tuple(const FormulaOwner& o, std::span<const fm::Term> terms, const VariablesPtr& vars)
{
    nb::list l;
    for (fm::Term x : terms)
        l.append(term_object(o, x, vars));
    return nb::tuple(l);
}

bool all_objects(std::span<const fm::Term> terms)
{
    for (fm::Term x : terms)
        if (!fm::is_object(x))
            return false;
    return true;
}

ExprPool condition_pool(const FormulaOwner& o, const std::shared_ptr<const ConjunctiveCondition>& c)
{
    return ExprPool{o, c, &c->exprs, &c->expr_terms, &c->constraints, VariablesPtr(c, &c->variables)};
}

ExprPool condition_pool(const FormulaOwner& o, const std::shared_ptr<const GroundCondition>& c)
{
    return ExprPool{o, c, &c->exprs, &c->expr_terms, &c->constraints, nullptr};
}

nb::tuple constraint_tuple(const ExprPool& p)
{
    nb::list l;
    for (u32 i = 0; i < p.constraints->size(); ++i)
        l.append(PyNumericConstraint{p, i});
    return nb::tuple(l);
}

/// The literals of a condition whose predicate has the kind (all with `any`).
nb::tuple literal_tuple(const PyConjunctiveCondition& c, std::optional<fm::PredKind> kind)
{
    const VariablesPtr vars(c.c, &c.c->variables);
    nb::list l;
    for (const auto& x : c.c->literals)
        if (!kind || c.o.data->predicates[x.predicate.v].kind == *kind)
            l.append(make_literal(c.o, x.predicate, x.positive, x.terms, vars));
    return nb::tuple(l);
}

nb::tuple ground_literal_tuple(const PyGroundCondition& c, std::optional<fm::PredKind> kind)
{
    nb::list l;
    for (const GroundLiteral& x : c.c->literals)
        if (!kind || c.o.data->predicates[x.atom.predicate.v].kind == *kind)
            l.append(make_ground_literal(c.o, x));
    return nb::tuple(l);
}

template<class V>
void set_fields(nb::class_<V>& cls, std::initializer_list<const char*> names)
{
    nb::list l;
    for (const char* n : names)
        l.append(nb::str(n));
    cls.attr("fields") = nb::tuple(l);
}

/// `other` as a V if it is one, else nullptr (the comparison then returns NotImplemented).
template<class V>
const V* as(nb::handle other)
{
    return nb::isinstance<V>(other) ? nb::inst_ptr<V>(other) : nullptr;
}

// ------------------------------------------------------------------------------------------------ pickling

nb::tuple variables_payload(const Variables& vars)
{
    nb::list l;
    for (const auto& v : vars)
    {
        nb::list ts;
        for (TypeId t : v.types)
            ts.append(t.v);
        l.append(nb::make_tuple(v.name, nb::tuple(ts)));
    }
    return nb::tuple(l);
}

template<class T>
nb::tuple ints(const std::vector<T>& v)
{
    nb::list l;
    for (const T& x : v)
    {
        if constexpr (std::is_integral_v<T>)
            l.append(x);
        else
            l.append(x.v);
    }
    return nb::tuple(l);
}

nb::tuple numeric_payload(const std::vector<fm::NumericConstraint>& cs, const std::vector<fm::Expr>& es, const std::vector<fm::Term>& ts)
{
    nb::list kl, el;
    for (const auto& k : cs)
        kl.append(nb::make_tuple(static_cast<int>(k.cmp), k.lhs, k.rhs));
    for (const auto& e : es)
        el.append(nb::make_tuple(static_cast<int>(e.op), e.value, e.func.v, e.terms.begin, e.terms.count, e.a, e.b));
    return nb::make_tuple(nb::tuple(kl), nb::tuple(el), ints(ts));
}

nb::object reduce(const FormulaOwner& o, const char* type, const char* kind, nb::object payload)
{
    if (o.task.is_none() || !o.core)
        throw nb::type_error((std::string("mymyr: a ") + type +
                              " pickles with the Task that made it; this one comes from a NormalizedTask "
                              "(make it with the Task's methods: atom, literal, condition, ground_condition, "
                              "precondition, goal_condition)")
                                 .c_str());
    return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_formula"),
                          nb::make_tuple(task_object(o.task), o.core->task->fingerprint(), kind, payload));
}

Variables variables_of(nb::handle payload)
{
    Variables out;
    for (nb::handle v : payload)
    {
        ConjunctiveCondition::Variable var;
        var.name = nb::cast<std::string>(v[0]);
        for (nb::handle t : v[1])
            var.types.push_back(TypeId{nb::cast<u32>(t)});
        out.push_back(std::move(var));
    }
    return out;
}

template<class T>
std::vector<T> ints_of(nb::handle payload)
{
    std::vector<T> out;
    for (nb::handle x : payload)
    {
        if constexpr (std::is_integral_v<T>)
            out.push_back(nb::cast<T>(x));
        else
            out.push_back(T{nb::cast<u32>(x)});
    }
    return out;
}

template<class C>
void numeric_of(nb::handle payload, C& c)
{
    for (nb::handle k : payload[0])
        c.constraints.push_back({static_cast<fm::Comparator>(nb::cast<int>(k[0])), nb::cast<u32>(k[1]), nb::cast<u32>(k[2])});
    for (nb::handle e : payload[1])
    {
        const int op = nb::cast<int>(e[0]);
        if (op < 0 || op > static_cast<int>(fm::ExprOp::Neg))
            throw nb::value_error("mymyr: corrupt pickle (expression operator)");
        fm::Expr x;
        x.op = static_cast<fm::ExprOp>(op);
        x.value = nb::cast<f64>(e[1]);
        x.func = FunctionId{nb::cast<u32>(e[2])};
        x.terms = fm::Range{nb::cast<u32>(e[3]), nb::cast<u32>(e[4])};
        x.a = nb::cast<u32>(e[5]);
        x.b = nb::cast<u32>(e[6]);
        c.exprs.push_back(x);
    }
    c.expr_terms = ints_of<fm::Term>(payload[2]);
    for (const auto& k : c.constraints)
        if (static_cast<int>(k.cmp) > static_cast<int>(fm::Comparator::Ge))
            throw nb::value_error("mymyr: corrupt pickle (comparator)");
}

/// Checks a value against the task (through a condition that holds it); invalid_argument becomes ValueError.
void check(const Task& task, const ConjunctiveCondition& c) { c.validate(task); }

nb::object restore_formula(nb::handle task, u64 fingerprint, std::string_view kind, nb::tuple payload)
{
    if (!nb::isinstance<PyTask>(task))
        throw nb::type_error("mymyr: _restore_formula needs a Task");
    PyTaskCore& core = *nb::inst_ptr<PyTask>(task)->core;
    if (core.task->fingerprint() != fingerprint)
        throw nb::value_error("mymyr: the formula was pickled from a different task (content fingerprint mismatch)");
    const FormulaOwner o = task_owner(Owner{&core, nb::borrow(task)});
    const Task& T = *core.task;
    if (kind == "variable")
    {
        ConjunctiveCondition c;
        c.variables = variables_of(payload[1]);
        const u32 position = nb::cast<u32>(payload[0]);
        if (position >= c.variables.size())
            throw nb::value_error("mymyr: corrupt pickle (variable position)");
        check(T, c);
        return nb::cast(PyVariable{o, position, c.variables[position].name, c.variables[position].types}, nb::rv_policy::move);
    }
    if (kind == "atom" || kind == "literal")
    {
        ConjunctiveCondition c;
        c.variables = variables_of(payload[3]);
        c.literals.push_back({PredicateId{nb::cast<u32>(payload[0])}, nb::cast<bool>(payload[1]), ints_of<fm::Term>(payload[2])});
        check(T, c);
        auto vars = std::make_shared<const Variables>(std::move(c.variables));
        const auto& l = c.literals[0];
        if (kind == "atom")
            return make_lifted_atom(o, l.predicate, l.terms, vars);
        return make_literal(o, l.predicate, l.positive, l.terms, vars);
    }
    if (kind == "ground_atom" || kind == "ground_literal")
    {
        GroundCondition g;
        g.literals.push_back({{PredicateId{nb::cast<u32>(payload[0])}, ints_of<ObjectId>(payload[2])}, nb::cast<bool>(payload[1])});
        g.validate(T);
        if (kind == "ground_atom")
            return make_ground_atom(o, g.literals[0].atom);
        return make_ground_literal(o, g.literals[0]);
    }
    if (kind == "conjunctive_condition")
    {
        ConjunctiveCondition c;
        c.variables = variables_of(payload[0]);
        for (nb::handle l : payload[1])
            c.literals.push_back({PredicateId{nb::cast<u32>(l[0])}, nb::cast<bool>(l[1]), ints_of<fm::Term>(l[2])});
        for (nb::handle e : payload[2])
            c.equalities.push_back({nb::cast<fm::Term>(e[0]), nb::cast<fm::Term>(e[1]), nb::cast<bool>(e[2])});
        numeric_of(payload[3], c);
        check(T, c);
        return make_conjunctive_condition(o, std::move(c));
    }
    if (kind == "ground_condition")
    {
        GroundCondition g;
        for (nb::handle l : payload[0])
            g.literals.push_back({{PredicateId{nb::cast<u32>(l[0])}, ints_of<ObjectId>(l[2])}, nb::cast<bool>(l[1])});
        numeric_of(payload[1], g);
        g.validate(T);
        return make_ground_condition(o, std::move(g));
    }
    throw nb::value_error("mymyr: corrupt pickle (unknown formula kind)");
}

nb::tuple literal_payload(const PyLiftedAtom& a, bool positive)
{
    return nb::make_tuple(a.predicate.v, positive, ints(a.terms), a.vars ? variables_payload(*a.vars) : nb::tuple());
}
}  // namespace

// ------------------------------------------------------------------------------------------------ makers

FormulaOwner bare_owner(DataPtr data) { return FormulaOwner{std::move(data), nb::none(), nullptr}; }

FormulaOwner task_owner(const Owner& o) { return FormulaOwner{o.core->data, o.obj, o.core}; }

VariablesPtr scope_variables(const TaskData& d, fm::Range first, fm::Range second)
{
    auto out = std::make_shared<Variables>();
    for (const fm::Range& r : {first, second})
        for (const fm::Parameter& p : TaskData::slice(d.params, r))
        {
            const auto types = TaskData::slice(d.type_ids, p.types);
            out->push_back({name_text(d, p.name), std::vector<TypeId>(types.begin(), types.end())});
        }
    return out;
}

nb::object term_object(const FormulaOwner& o, fm::Term x, const VariablesPtr& vars)
{
    if (fm::is_object(x))
        return nb::cast(ObjectView{{o.data, fm::term_object(x).v}});
    const u32 v = fm::term_parameter(x);
    PyVariable var{o, v, "x" + std::to_string(v), {}};
    if (vars && v < vars->size())
    {
        if (!(*vars)[v].name.empty())
            var.name = (*vars)[v].name;
        var.types = (*vars)[v].types;
    }
    return nb::cast(std::move(var), nb::rv_policy::move);
}

ExprPool task_pool(const FormulaOwner& o, VariablesPtr vars)
{
    return ExprPool{o, o.data, &o.data->exprs, &o.data->terms, &o.data->constraints, std::move(vars)};
}

nb::object make_lifted_atom(const FormulaOwner& o, PredicateId predicate, std::vector<fm::Term> terms, VariablesPtr vars)
{
    return nb::cast(PyLiftedAtom{o, predicate, std::move(terms), std::move(vars)}, nb::rv_policy::move);
}

nb::object make_literal(const FormulaOwner& o, PredicateId predicate, bool positive, std::vector<fm::Term> terms, VariablesPtr vars)
{
    return nb::cast(PyLiteral{PyLiftedAtom{o, predicate, std::move(terms), std::move(vars)}, positive}, nb::rv_policy::move);
}

nb::object make_ground_atom(const FormulaOwner& o, GroundAtom a) { return nb::cast(PyGroundAtom{o, std::move(a)}, nb::rv_policy::move); }

nb::object make_ground_literal(const FormulaOwner& o, GroundLiteral l)
{
    return nb::cast(PyGroundLiteral{o, std::move(l)}, nb::rv_policy::move);
}

nb::object make_conjunctive_condition(const FormulaOwner& o, ConjunctiveCondition c)
{
    return nb::cast(PyConjunctiveCondition{o, std::make_shared<const ConjunctiveCondition>(std::move(c))}, nb::rv_policy::move);
}

nb::object make_ground_condition(const FormulaOwner& o, GroundCondition c)
{
    return nb::cast(PyGroundCondition{o, std::make_shared<const GroundCondition>(std::move(c))}, nb::rv_policy::move);
}

std::string literal_text(const TaskData& d, PredicateId predicate, bool positive, std::span<const fm::Term> terms, const Variables* vars)
{
    std::string a = "(" + name_text(d, d.predicates[predicate.v].name);
    for (fm::Term x : terms)
        a += " " + term_text(d, x, vars);
    a += ")";
    return positive ? a : "(not " + a + ")";
}

std::string ground_atom_text(const TaskData& d, const GroundAtom& a)
{
    std::string s = "(" + name_text(d, d.predicates[a.predicate.v].name);
    for (ObjectId o : a.objects)
        s += " " + name_text(d, d.objects[o.v].name);
    return s + ")";
}

const PyState& formula_state(const FormulaOwner& o, nb::handle state)
{
    if (!is_state(state))
        throw nb::type_error("mymyr: expected a State");
    const PyState& s = state_of(state);
    if (s.core->data.get() != o.data.get())
        throw nb::value_error("mymyr: the state belongs to another task than the formula");
    return s;
}

// ------------------------------------------------------------------------------------------------ registration

void bind_formulas(nb::module_& m, nb::module_& parent)
{
    // --- Variable
    {
        nb::class_<PyVariable> c(m, "Variable", "A variable of a condition: its position in the variable list, its name and types.");
        c.def_prop_ro("position", [](const PyVariable& v) { return v.position; }, "index in the enclosing variable list")
            .def_prop_ro("name", [](const PyVariable& v) { return v.name; }, "without '?'")
            .def_prop_ro("types", [](const PyVariable& v) { return type_tuple(v.o.data, v.types); },
                         "the variable ranges over the objects of any of these types; empty: every object")
            .def("__eq__",
                 [](const PyVariable& a, nb::handle b) -> nb::object {
                     const PyVariable* o = as<PyVariable>(b);
                     if (!o)
                         return nb::borrow(Py_NotImplemented);
                     return nb::bool_(a.o.data.get() == o->o.data.get() && a.position == o->position && a.name == o->name &&
                                      a.types == o->types);
                 })
            .def("__hash__", [](const PyVariable& v) { return to_hash(mix(mix(data_hash(v.o), v.position), std::hash<std::string>{}(v.name))); })
            .def("__reduce__", [](const PyVariable& v) {
                Variables vars(v.position + 1);
                vars[v.position] = {v.name, v.types};
                return reduce(v.o, "Variable", "variable", nb::make_tuple(v.position, variables_payload(vars)));
            })
            .def("__str__", [](const PyVariable& v) { return "?" + v.name; })
            .def("__repr__", [](const PyVariable& v) { return "Variable(?" + v.name + ")"; });
        set_fields(c, {"position", "name", "types"});
    }
    // --- Atom
    {
        nb::class_<PyLiftedAtom> c(m, "Atom", "A lifted atom: a predicate applied to objects and variables (Task.atom with '?x' terms).");
        c.def_prop_ro("predicate", [](const PyLiftedAtom& a) { return PredicateView{{a.o.data, a.predicate.v}}; })
            .def_prop_ro("terms", [](const PyLiftedAtom& a) { return term_tuple(a.o, a.terms, a.vars); }, "an Object or a Variable per argument")
            .def_prop_ro("is_ground", [](const PyLiftedAtom& a) { return all_objects(a.terms); })
            .def("__eq__",
                 [](const PyLiftedAtom& a, nb::handle b) -> nb::object {
                     const PyLiftedAtom* o = as<PyLiftedAtom>(b);
                     return o ? nb::object(nb::bool_(same_atom(a, *o))) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyLiftedAtom& a) { return to_hash(atom_hash(a)); })
            .def("__reduce__", [](const PyLiftedAtom& a) { return reduce(a.o, "Atom", "atom", literal_payload(a, true)); })
            .def("__str__", [](const PyLiftedAtom& a) { return literal_text(*a.o.data, a.predicate, true, a.terms, a.vars.get()); })
            .def("__repr__", [](const PyLiftedAtom& a) { return "Atom" + literal_text(*a.o.data, a.predicate, true, a.terms, a.vars.get()); });
        set_fields(c, {"predicate", "terms", "is_ground"});
    }
    // --- Literal
    {
        nb::class_<PyLiteral> c(m, "Literal", "A lifted literal: an Atom and its polarity.");
        c.def_prop_ro("atom", [](const PyLiteral& l) { return l.atom; })
            .def_prop_ro("predicate", [](const PyLiteral& l) { return PredicateView{{l.atom.o.data, l.atom.predicate.v}}; })
            .def_prop_ro("positive", [](const PyLiteral& l) { return l.positive; })
            .def_prop_ro("terms", [](const PyLiteral& l) { return term_tuple(l.atom.o, l.atom.terms, l.atom.vars); },
                         "an Object or a Variable per argument")
            .def_prop_ro("is_ground", [](const PyLiteral& l) { return all_objects(l.atom.terms); })
            .def("__eq__",
                 [](const PyLiteral& a, nb::handle b) -> nb::object {
                     const PyLiteral* o = as<PyLiteral>(b);
                     return o ? nb::object(nb::bool_(a.positive == o->positive && same_atom(a.atom, o->atom))) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyLiteral& l) { return to_hash(mix(atom_hash(l.atom), l.positive)); })
            .def("__reduce__", [](const PyLiteral& l) { return reduce(l.atom.o, "Literal", "literal", literal_payload(l.atom, l.positive)); })
            .def("__str__",
                 [](const PyLiteral& l) { return literal_text(*l.atom.o.data, l.atom.predicate, l.positive, l.atom.terms, l.atom.vars.get()); })
            .def("__repr__", [](const PyLiteral& l) {
                return "Literal" + literal_text(*l.atom.o.data, l.atom.predicate, l.positive, l.atom.terms, l.atom.vars.get());
            });
        set_fields(c, {"atom", "predicate", "positive", "terms", "is_ground"});
    }
    // --- GroundAtom
    {
        nb::class_<PyGroundAtom> c(m, "GroundAtom", "A ground atom: a predicate applied to objects.");
        c.def_prop_ro("predicate", [](const PyGroundAtom& a) { return PredicateView{{a.o.data, a.a.predicate.v}}; })
            .def_prop_ro("objects", [](const PyGroundAtom& a) { return object_tuple(a.o.data, a.a.objects); })
            .def_prop_ro("predicate_index", [](const PyGroundAtom& a) { return a.a.predicate.v; })
            .def_prop_ro("object_indices", [](const PyGroundAtom& a) {
                std::vector<u32> v;
                for (ObjectId o : a.a.objects)
                    v.push_back(o.v);
                return v;
            })
            .def_prop_ro("kind", [](const PyGroundAtom& a) { return std::string(kind_name(a.o.data->predicates[a.a.predicate.v].kind)); },
                         "'static', 'fluent' or 'derived'")
            .def_prop_ro(
                "slot",
                [](const PyGroundAtom& a) -> std::optional<u32> {
                    if (!a.o.core || a.o.data->predicates[a.a.predicate.v].kind != fm::PredKind::Fluent)
                        return std::nullopt;
                    const SlotId s = a.o.core->task->find_atom(a.a.predicate, a.a.objects);
                    return s.valid() ? std::optional<u32>(s.v) : std::nullopt;
                },
                "The state bit of a fluent atom of a Task, or None (static and derived atoms, atoms without a slot yet "
                "under lazy slots, atoms of a NormalizedTask).")
            .def(
                "holds", [](const PyGroundAtom& a, nb::handle state) {
                    const PyState& s = formula_state(a.o, state);
                    return mymyr::holds(*s.core->task, s.s.view(), a.a);
                },
                "state"_a, "Truth in a state (State.holds).")
            .def("__eq__",
                 [](const PyGroundAtom& a, nb::handle b) -> nb::object {
                     const PyGroundAtom* o = as<PyGroundAtom>(b);
                     return o ? nb::object(nb::bool_(a.o.data.get() == o->o.data.get() && a.a == o->a)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyGroundAtom& a) { return to_hash(mix(data_hash(a.o), ground_atom_hash(a.a))); })
            .def("__reduce__", [](const PyGroundAtom& a) {
                return reduce(a.o, "GroundAtom", "ground_atom", nb::make_tuple(a.a.predicate.v, true, ints(a.a.objects)));
            })
            .def("__str__", [](const PyGroundAtom& a) { return ground_atom_text(*a.o.data, a.a); })
            .def("__repr__", [](const PyGroundAtom& a) { return "GroundAtom" + ground_atom_text(*a.o.data, a.a); });
        set_fields(c, {"predicate", "objects", "kind"});
    }
    // --- GroundLiteral
    {
        nb::class_<PyGroundLiteral> c(m, "GroundLiteral", "A ground literal: a GroundAtom and its polarity.");
        c.def_prop_ro("atom", [](const PyGroundLiteral& l) { return PyGroundAtom{l.o, l.l.atom}; })
            .def_prop_ro("predicate", [](const PyGroundLiteral& l) { return PredicateView{{l.o.data, l.l.atom.predicate.v}}; })
            .def_prop_ro("positive", [](const PyGroundLiteral& l) { return l.l.positive; })
            .def_prop_ro("objects", [](const PyGroundLiteral& l) { return object_tuple(l.o.data, l.l.atom.objects); })
            .def(
                "holds", [](const PyGroundLiteral& l, nb::handle state) {
                    const PyState& s = formula_state(l.o, state);
                    return mymyr::holds(*s.core->task, s.s.view(), l.l);
                },
                "state"_a, "Truth in a state (State.holds).")
            .def("__eq__",
                 [](const PyGroundLiteral& a, nb::handle b) -> nb::object {
                     const PyGroundLiteral* o = as<PyGroundLiteral>(b);
                     return o ? nb::object(nb::bool_(a.o.data.get() == o->o.data.get() && a.l == o->l)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyGroundLiteral& l) { return to_hash(mix(mix(data_hash(l.o), ground_atom_hash(l.l.atom)), l.l.positive)); })
            .def("__reduce__", [](const PyGroundLiteral& l) {
                return reduce(l.o, "GroundLiteral", "ground_literal", nb::make_tuple(l.l.atom.predicate.v, l.l.positive, ints(l.l.atom.objects)));
            })
            .def("__str__", [](const PyGroundLiteral& l) {
                const std::string a = ground_atom_text(*l.o.data, l.l.atom);
                return l.l.positive ? a : "(not " + a + ")";
            })
            .def("__repr__", [](const PyGroundLiteral& l) {
                const std::string a = ground_atom_text(*l.o.data, l.l.atom);
                return "GroundLiteral" + (l.l.positive ? a : "(not " + a + ")");
            });
        set_fields(c, {"atom", "predicate", "positive", "objects"});
    }
    // --- Expression
    {
        nb::class_<PyExpression> c(m, "Expression", "A node of a numeric expression tree.");
        c.def_prop_ro("index", [](const PyExpression& e) { return e.i; })
            .def_prop_ro("op", [](const PyExpression& e) {
                switch ((*e.p.exprs)[e.i].op)
                {
                    case fm::ExprOp::Number: return std::string("number");
                    case fm::ExprOp::Function: return std::string("function");
                    case fm::ExprOp::Add: return std::string("+");
                    case fm::ExprOp::Sub: return std::string("-");
                    case fm::ExprOp::Mul: return std::string("*");
                    case fm::ExprOp::Div: return std::string("/");
                    case fm::ExprOp::Neg: return std::string("neg");
                }
                return std::string("?");
            })
            .def_prop_ro("value", [](const PyExpression& e) -> std::optional<f64> {
                const auto& x = (*e.p.exprs)[e.i];
                return x.op == fm::ExprOp::Number ? std::optional<f64>(x.value) : std::nullopt;
            })
            .def_prop_ro("function", [](const PyExpression& e) -> std::optional<FunctionView> {
                const auto& x = (*e.p.exprs)[e.i];
                return x.op == fm::ExprOp::Function ? std::optional<FunctionView>(FunctionView{{e.p.o.data, x.func.v}}) : std::nullopt;
            })
            .def_prop_ro("terms", [](const PyExpression& e) {
                const auto& x = (*e.p.exprs)[e.i];
                if (x.op != fm::ExprOp::Function)
                    return nb::tuple();
                return term_tuple(e.p.o, std::span(*e.p.terms).subspan(x.terms.begin, x.terms.count), e.p.vars);
            })
            .def_prop_ro("children", [](const PyExpression& e) {
                nb::list l;
                const auto& x = (*e.p.exprs)[e.i];
                if (x.op == fm::ExprOp::Neg)
                    l.append(PyExpression{e.p, x.a});
                else if (x.op != fm::ExprOp::Number && x.op != fm::ExprOp::Function)
                {
                    l.append(PyExpression{e.p, x.a});
                    l.append(PyExpression{e.p, x.b});
                }
                return nb::tuple(l);
            })
            .def("__eq__",
                 [](const PyExpression& a, nb::handle b) -> nb::object {
                     const PyExpression* o = as<PyExpression>(b);
                     return o ? nb::object(nb::bool_(a.p.keep.get() == o->p.keep.get() && a.i == o->i)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyExpression& e) { return to_hash(mix(std::hash<const void*>{}(e.p.keep.get()), e.i)); })
            .def("__str__", [](const PyExpression& e) { return expr_text(e.p, e.i); })
            .def("__repr__", [](const PyExpression& e) { return "Expression" + expr_text(e.p, e.i); });
        set_fields(c, {"op", "value", "function", "terms", "children"});
    }
    // --- NumericConstraint
    {
        nb::class_<PyNumericConstraint> c(m, "NumericConstraint", "A comparison of two numeric expressions.");
        c.def_prop_ro("index", [](const PyNumericConstraint& k) { return k.i; })
            .def_prop_ro("comparator", [](const PyNumericConstraint& k) { return std::string(cmp_name((*k.p.constraints)[k.i].cmp)); })
            .def_prop_ro("lhs", [](const PyNumericConstraint& k) { return PyExpression{k.p, (*k.p.constraints)[k.i].lhs}; })
            .def_prop_ro("rhs", [](const PyNumericConstraint& k) { return PyExpression{k.p, (*k.p.constraints)[k.i].rhs}; })
            .def("__eq__",
                 [](const PyNumericConstraint& a, nb::handle b) -> nb::object {
                     const PyNumericConstraint* o = as<PyNumericConstraint>(b);
                     return o ? nb::object(nb::bool_(a.p.keep.get() == o->p.keep.get() && a.i == o->i)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyNumericConstraint& k) { return to_hash(mix(std::hash<const void*>{}(k.p.keep.get()), k.i ^ 0x4b)); })
            .def("__str__", [](const PyNumericConstraint& k) { return constraint_text(k.p, k.i); })
            .def("__repr__", [](const PyNumericConstraint& k) { return "NumericConstraint" + constraint_text(k.p, k.i); });
        set_fields(c, {"comparator", "lhs", "rhs"});
    }
    // --- ConjunctiveCondition
    {
        nb::class_<PyConjunctiveCondition> c(
            m, "ConjunctiveCondition",
            "A lifted conjunctive condition: variables, literals (static, fluent and derived), equalities and numeric "
            "constraints. Task.bindings enumerates its bindings in a state; ground() makes the ground conditions.");
        c.def_prop_ro("variables",
                      [](const PyConjunctiveCondition& x) {
                          const VariablesPtr vars(x.c, &x.c->variables);
                          nb::list l;
                          for (u32 v = 0; v < x.c->arity(); ++v)
                              l.append(term_object(x.o, static_cast<fm::Term>(v), vars));
                          return nb::tuple(l);
                      })
            .def_prop_ro("arity", [](const PyConjunctiveCondition& x) { return x.c->arity(); }, "The number of variables.")
            .def_prop_ro("literals", [](const PyConjunctiveCondition& x) { return literal_tuple(x, std::nullopt); })
            .def_prop_ro("static_literals", [](const PyConjunctiveCondition& x) { return literal_tuple(x, fm::PredKind::Static); })
            .def_prop_ro("fluent_literals", [](const PyConjunctiveCondition& x) { return literal_tuple(x, fm::PredKind::Fluent); })
            .def_prop_ro("derived_literals", [](const PyConjunctiveCondition& x) { return literal_tuple(x, fm::PredKind::Derived); })
            .def_prop_ro(
                "equalities",
                [](const PyConjunctiveCondition& x) {
                    const VariablesPtr vars(x.c, &x.c->variables);
                    nb::list l;
                    for (const auto& e : x.c->equalities)
                        l.append(nb::make_tuple(term_object(x.o, e.lhs, vars), term_object(x.o, e.rhs, vars), e.positive));
                    return nb::tuple(l);
                },
                "(lhs, rhs, positive) per (in)equality of two terms: lhs == rhs if positive, else lhs != rhs")
            .def_prop_ro("numeric_constraints", [](const PyConjunctiveCondition& x) { return constraint_tuple(condition_pool(x.o, x.c)); })
            .def(
                "ground",
                [](const PyConjunctiveCondition& x, nb::handle state, nb::handle limit, nb::handle partial) {
                    const PyState& s = formula_state(x.o, state);
                    const Task& task = *s.core->task;
                    const std::vector<std::optional<ObjectId>> fixed = condition_partial(*s.core, x.c, partial);
                    const u64 cap = limit_value(limit);
                    std::vector<GroundCondition> found;
                    if (cap > 0)
                        for_each_binding(task, task.workspace(), *x.c, s.s.view(), fixed, [&](std::span<const ObjectId> b) {
                            found.push_back(x.c->ground(b));
                            return found.size() < cap;
                        });
                    const FormulaOwner o = x.o.task.is_none() ? task_owner(Owner{s.core, s.owner}) : x.o;
                    nb::list out;
                    for (GroundCondition& g : found)
                        out.append(make_ground_condition(o, std::move(g)));
                    return out;
                },
                "state"_a, "limit"_a = nb::none(), "partial"_a = nb::none(),
                "The ground conditions of the bindings in a state (at most limit; partial fixes variables as in "
                "Task.bindings), in the order of Task.bindings.")
            .def("__len__", [](const PyConjunctiveCondition& x) { return x.c->literals.size() + x.c->equalities.size() + x.c->constraints.size(); },
                 "The number of conjuncts: literals, equalities and numeric constraints.")
            .def("__eq__",
                 [](const PyConjunctiveCondition& a, nb::handle b) -> nb::object {
                     const PyConjunctiveCondition* o = as<PyConjunctiveCondition>(b);
                     return o ? nb::object(nb::bool_(a.o.data.get() == o->o.data.get() && *a.c == *o->c)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyConjunctiveCondition& x) {
                u64 h = mix(data_hash(x.o), x.c->arity());
                for (const auto& l : x.c->literals)
                {
                    h = mix(mix(h, l.predicate.v), l.positive);
                    for (fm::Term t : l.terms)
                        h = mix(h, static_cast<u32>(t));
                }
                return to_hash(mix(h, x.c->constraints.size()));
            })
            .def("__reduce__", [](const PyConjunctiveCondition& x) {
                nb::list ls, es;
                for (const auto& l : x.c->literals)
                    ls.append(nb::make_tuple(l.predicate.v, l.positive, ints(l.terms)));
                for (const auto& e : x.c->equalities)
                    es.append(nb::make_tuple(e.lhs, e.rhs, e.positive));
                return reduce(x.o, "ConjunctiveCondition", "conjunctive_condition",
                              nb::make_tuple(variables_payload(x.c->variables), nb::tuple(ls), nb::tuple(es),
                                             numeric_payload(x.c->constraints, x.c->exprs, x.c->expr_terms)));
            })
            .def("__str__", [](const PyConjunctiveCondition& x) { return x.c->str(*x.o.data); })
            .def("__repr__", [](const PyConjunctiveCondition& x) { return "ConjunctiveCondition" + x.c->str(*x.o.data); });
        set_fields(c, {"variables", "literals", "static_literals", "fluent_literals", "derived_literals", "equalities",
                       "numeric_constraints"});
    }
    // --- GroundCondition
    {
        nb::class_<PyGroundCondition> c(m, "GroundCondition",
                                        "A ground conjunctive condition: ground literals (static, fluent and derived) and "
                                        "ground numeric constraints. A goal of the searches (goal=).");
        c.def_prop_ro("literals", [](const PyGroundCondition& x) { return ground_literal_tuple(x, std::nullopt); })
            .def_prop_ro("static_literals", [](const PyGroundCondition& x) { return ground_literal_tuple(x, fm::PredKind::Static); })
            .def_prop_ro("fluent_literals", [](const PyGroundCondition& x) { return ground_literal_tuple(x, fm::PredKind::Fluent); })
            .def_prop_ro("derived_literals", [](const PyGroundCondition& x) { return ground_literal_tuple(x, fm::PredKind::Derived); })
            .def_prop_ro("numeric_constraints", [](const PyGroundCondition& x) { return constraint_tuple(condition_pool(x.o, x.c)); })
            .def(
                "holds",
                [](const PyGroundCondition& x, nb::handle state) {
                    const PyState& s = formula_state(x.o, state);
                    return mymyr::holds(*s.core->task, s.s.view(), *x.c);
                },
                "state"_a,
                "Truth in a state: static literals by the task's static facts, fluent ones by the state, derived ones by "
                "the axioms, numeric constraints by the state's values (an undefined value makes a comparison false).")
            .def(
                "lift", [](const PyGroundCondition& x, bool add_inequalities) { return make_conjunctive_condition(x.o, x.c->lift(add_inequalities)); },
                "add_inequalities"_a = false,
                "The ConjunctiveCondition with every object replaced by a variable ?x<i>, numbered by first appearance "
                "(literals, then numeric constraints), untyped. With add_inequalities, every pair of variables gets an "
                "inequality, so that the bindings map distinct variables to distinct objects. Grounding it under the "
                "binding that maps each variable back to its object gives this condition back.")
            .def("__len__", [](const PyGroundCondition& x) { return x.c->literals.size() + x.c->constraints.size(); },
                 "The number of conjuncts: literals and numeric constraints.")
            .def("__eq__",
                 [](const PyGroundCondition& a, nb::handle b) -> nb::object {
                     const PyGroundCondition* o = as<PyGroundCondition>(b);
                     return o ? nb::object(nb::bool_(a.o.data.get() == o->o.data.get() && *a.c == *o->c)) : nb::borrow(Py_NotImplemented);
                 })
            .def("__hash__", [](const PyGroundCondition& x) {
                u64 h = data_hash(x.o);
                for (const auto& l : x.c->literals)
                    h = mix(mix(h, ground_atom_hash(l.atom)), l.positive);
                return to_hash(mix(h, x.c->constraints.size()));
            })
            .def("__reduce__", [](const PyGroundCondition& x) {
                nb::list ls;
                for (const auto& l : x.c->literals)
                    ls.append(nb::make_tuple(l.atom.predicate.v, l.positive, ints(l.atom.objects)));
                return reduce(x.o, "GroundCondition", "ground_condition",
                              nb::make_tuple(nb::tuple(ls), numeric_payload(x.c->constraints, x.c->exprs, x.c->expr_terms)));
            })
            .def("__str__", [](const PyGroundCondition& x) { return x.c->str(*x.o.data); })
            .def("__repr__", [](const PyGroundCondition& x) { return "GroundCondition" + x.c->str(*x.o.data); });
        set_fields(c, {"literals", "static_literals", "fluent_literals", "derived_literals", "numeric_constraints"});
    }

    parent.def("_restore_formula", &restore_formula);
}
}  // namespace mymyr::python
