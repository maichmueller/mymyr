// Read-only Python views of the normalized task (TaskData): every entity is reachable from Python,
// the bindings generated from one entity-descriptor table.
//
// Design:
//   - A view is (shared_ptr<const TaskData>, index [, context]). Views keep the task alive, are immutable, hashable
//     and comparable by (task identity, kind, index), and are cheap to create. Nothing is interned or cached.
//   - Each class is declared once in a descriptor table (class name, then named fields with getters). The table
//     drives the property registration, __repr__, and the `fields` class attribute that the coverage test walks.
//   - Terms render as Object views (constants/objects) or Variable (parameter index + name from the enclosing scope).

#include "formalism_task.hpp"
#include "formalism_views.hpp"

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/formalism/text_format.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace mymyr::python
{
using formalism::TaskData;
using TaskPtr = std::shared_ptr<const TaskData>;

namespace
{
namespace fm = formalism;

// ------------------------------------------------------------------------------------------------ view types

using View = FormalismView;

struct TypeV : View {};
using ObjectV = ObjectView;
struct PredicateV : View {};
struct FunctionV : View {};
struct SchemaV : View {};
struct AxiomV : View {};

/// A parameter in scope: index into TaskData::params, plus its position in the enclosing parameter list.
struct ParameterV : View
{
    u32 position = 0;
};

/// A variable term: position in the enclosing scope, with the scope's parameter range for its name.
struct VariableV
{
    TaskPtr t;
    u32 position = 0;
    std::string name;
};

/// Parameter scope of an expression/literal: schema params followed by (optionally) a conditional effect's extras.
struct Scope
{
    fm::Range first, second;  // second may be empty
    [[nodiscard]] std::string name(const TaskData& d, u32 pos) const
    {
        const fm::Range& r = pos < first.count ? first : second;
        const u32 k = pos < first.count ? pos : pos - first.count;
        if (k >= r.count)
            return "?" + std::to_string(pos);
        return std::string(d.str(d.params[r.begin + k].name));
    }
};

struct LiteralV : View  // i = index into literals
{
    Scope scope;
};
struct ExprV : View  // i = index into exprs
{
    Scope scope;
};
struct ConstraintV : View  // i = index into constraints
{
    Scope scope;
};
struct NumericEffectV : View  // i = index into numeric_effects, or aux (stored inline)
{
    Scope scope;
    std::optional<fm::NumericEffect> inline_effect;  // for auxiliary effects stored in ConditionalEffect
    [[nodiscard]] const fm::NumericEffect& e() const { return inline_effect ? *inline_effect : d().numeric_effects[i]; }
};
struct ConditionV
{
    TaskPtr t;
    fm::Condition c;
    Scope scope;
    std::string owner;  // for repr
};
struct ConditionalEffectV : View  // i = index into conditional_effects
{
    fm::Range schema_params;
};
struct GroundAtomV : View  // i = index into static_init or fluent_init
{
    bool fluent = false;
    [[nodiscard]] const fm::GroundAtom& a() const { return fluent ? d().fluent_init[i] : d().static_init[i]; }
};
struct GroundValueV : View
{
    bool fluent = false;
    [[nodiscard]] const fm::GroundFunctionValue& v() const { return fluent ? d().fluent_values[i] : d().static_values[i]; }
};

const char* kind_name(fm::PredKind k) { return k == fm::PredKind::Static ? "static" : k == fm::PredKind::Fluent ? "fluent" : "derived"; }
const char* kind_name(fm::FuncKind k) { return k == fm::FuncKind::Static ? "static" : k == fm::FuncKind::Fluent ? "fluent" : "auxiliary"; }
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
const char* op_name(fm::AssignOp o)
{
    switch (o)
    {
        case fm::AssignOp::Assign: return "assign";
        case fm::AssignOp::Increase: return "increase";
        case fm::AssignOp::Decrease: return "decrease";
        case fm::AssignOp::ScaleUp: return "scale-up";
        case fm::AssignOp::ScaleDown: return "scale-down";
    }
    return "?";
}

nb::object term_object(const TaskPtr& t, fm::Term x, const Scope& s)
{
    if (fm::is_object(x))
        return nb::cast(ObjectV{{t, fm::term_object(x).v}});
    return nb::cast(VariableV{t, fm::term_parameter(x), s.name(*t, fm::term_parameter(x))});
}

std::string term_str(const TaskData& d, fm::Term x, const Scope& s)
{
    if (fm::is_object(x))
        return std::string(d.str(d.objects[fm::term_object(x).v].name));
    return "?" + s.name(d, fm::term_parameter(x));
}

std::string literal_str(const TaskData& d, const fm::Literal& l, const Scope& s)
{
    std::string out = l.positive ? "(" : "(not (";
    out += d.str(d.predicates[l.pred.v].name);
    for (fm::Term x : d.terms_of(l))
        out += " " + term_str(d, x, s);
    out += l.positive ? ")" : "))";
    return out;
}

std::string expr_str(const TaskData& d, u32 e, const Scope& s)
{
    const fm::Expr& x = d.exprs[e];
    switch (x.op)
    {
        case fm::ExprOp::Number:
        {
            std::ostringstream o;
            o << x.value;
            return o.str();
        }
        case fm::ExprOp::Function:
        {
            std::string out = "(" + std::string(d.str(d.functions[x.func.v].name));
            for (fm::Term tt : TaskData::slice(d.terms, x.terms))
                out += " " + term_str(d, tt, s);
            return out + ")";
        }
        case fm::ExprOp::Neg: return "(- " + expr_str(d, x.a, s) + ")";
        default:
        {
            const char* op = x.op == fm::ExprOp::Add ? "+" : x.op == fm::ExprOp::Sub ? "-" : x.op == fm::ExprOp::Mul ? "*" : "/";
            return std::string("(") + op + " " + expr_str(d, x.a, s) + " " + expr_str(d, x.b, s) + ")";
        }
    }
}

Scope schema_scope(const TaskData& d, u32 schema) { return Scope{d.schemas[schema].params, {}}; }

// ------------------------------------------------------------------------------------------------ descriptor table

/// Registers a class from a list of (field name, getter) and records the field names in `fields`.
template<class V>
struct ClassDescriptor
{
    nb::class_<V>& cls;
    std::vector<std::string> names;

    explicit ClassDescriptor(nb::class_<V>& c) : cls(c) {}

    template<class F>
    ClassDescriptor& field(const char* name, F&& getter, const char* doc = "")
    {
        cls.def_prop_ro(name, std::forward<F>(getter), doc);
        names.emplace_back(name);
        return *this;
    }
    void finish()
    {
        nb::list l;
        for (auto& n : names)
            l.append(nb::str(n.c_str()));
        cls.attr("fields") = nb::tuple(l);
    }
};

template<class V>
void identity_protocol(nb::class_<V>& cls, const char* kind)
{
    cls.def("__eq__", [](const V& a, const V& b) { return a.t.get() == b.t.get() && a.i == b.i; }, nb::is_operator());
    cls.def("__hash__", [kind](const V& a) {
        return static_cast<nb::ssize_t>(std::hash<const void*>{}(a.t.get()) ^ (std::hash<u32>{}(a.i) * 0x9e3779b97f4a7c15ULL) ^
                                        std::hash<std::string_view>{}(kind));
    });
    cls.def_prop_ro("index", [](const V& a) { return a.i; });
}

template<class V, class Vec>
nb::list views_of(const TaskPtr& t, const Vec& pool)
{
    nb::list l;
    for (u32 i = 0; i < pool.size(); ++i)
        l.append(V{{t, i}});
    return l;
}

nb::list literal_list(const TaskPtr& t, fm::Range r, const Scope& s)
{
    nb::list l;
    for (u32 k = 0; k < r.count; ++k)
        l.append(LiteralV{{t, r.begin + k}, s});
    return l;
}
nb::list constraint_list(const TaskPtr& t, fm::Range r, const Scope& s)
{
    nb::list l;
    for (u32 k = 0; k < r.count; ++k)
        l.append(ConstraintV{{t, r.begin + k}, s});
    return l;
}
nb::list parameter_list(const TaskPtr& t, fm::Range r, u32 offset = 0)
{
    nb::list l;
    for (u32 k = 0; k < r.count; ++k)
        l.append(ParameterV{{t, r.begin + k}, offset + k});
    return l;
}
nb::list type_list(const TaskPtr& t, fm::Range r)
{
    nb::list l;
    for (TypeId ty : TaskData::slice(t->type_ids, r))
        l.append(TypeV{{t, ty.v}});
    return l;
}
nb::list object_list(const TaskPtr& t, fm::Range r)
{
    nb::list l;
    for (ObjectId o : TaskData::slice(t->object_ids, r))
        l.append(ObjectV{{t, o.v}});
    return l;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ Task-level wrapper

void bind_formalism(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_formalism", "Read-only views of the normalized task (complete: every entity)");

    // --- Type
    {
        nb::class_<TypeV> c(m, "Type");
        identity_protocol(c, "type");
        ClassDescriptor<TypeV>{c}
            .field("name", [](const TypeV& v) { return std::string(v.d().str(v.d().types[v.i].name)); })
            .field("bases", [](const TypeV& v) { return type_list(v.t, v.d().types[v.i].bases); }, "direct supertypes")
            .finish();
        c.def("__repr__", [](const TypeV& v) { return "Type(" + std::string(v.d().str(v.d().types[v.i].name)) + ")"; });
    }
    // --- Object
    {
        nb::class_<ObjectV> c(m, "Object");
        identity_protocol(c, "object");
        ClassDescriptor<ObjectV>{c}
            .field("name", [](const ObjectV& v) { return std::string(v.d().str(v.d().objects[v.i].name)); })
            .field("types", [](const ObjectV& v) { return type_list(v.t, v.d().objects[v.i].types); })
            .field("is_constant", [](const ObjectV& v) { return v.d().objects[v.i].constant; })
            .finish();
        c.def("__repr__", [](const ObjectV& v) { return "Object(" + std::string(v.d().str(v.d().objects[v.i].name)) + ")"; });
        c.def("__str__", [](const ObjectV& v) { return std::string(v.d().str(v.d().objects[v.i].name)); });
    }
    // --- Variable (a parameter reference inside a literal/expression)
    {
        nb::class_<VariableV> c(m, "Variable");
        c.def_prop_ro("position", [](const VariableV& v) { return v.position; }, "index in the enclosing parameter list");
        c.def_prop_ro("name", [](const VariableV& v) { return v.name; });
        c.attr("fields") = nb::make_tuple("position", "name");
        c.def("__repr__", [](const VariableV& v) { return "Variable(?" + v.name + ")"; });
        c.def("__str__", [](const VariableV& v) { return "?" + v.name; });
        c.def("__eq__", [](const VariableV& a, const VariableV& b) { return a.t.get() == b.t.get() && a.position == b.position && a.name == b.name; }, nb::is_operator());
        c.def("__hash__", [](const VariableV& a) { return static_cast<nb::ssize_t>(std::hash<std::string>{}(a.name) ^ a.position); });
    }
    // --- Parameter
    {
        nb::class_<ParameterV> c(m, "Parameter");
        identity_protocol(c, "parameter");
        ClassDescriptor<ParameterV>{c}
            .field("name", [](const ParameterV& v) { return std::string(v.d().str(v.d().params[v.i].name)); })
            .field("position", [](const ParameterV& v) { return v.position; })
            .field("types", [](const ParameterV& v) { return type_list(v.t, v.d().params[v.i].types); })
            .finish();
        c.def("__repr__", [](const ParameterV& v) { return "Parameter(?" + std::string(v.d().str(v.d().params[v.i].name)) + ")"; });
    }
    // --- Predicate
    {
        nb::class_<PredicateV> c(m, "Predicate");
        identity_protocol(c, "predicate");
        ClassDescriptor<PredicateV>{c}
            .field("name", [](const PredicateV& v) { return std::string(v.d().str(v.d().predicates[v.i].name)); })
            .field("kind", [](const PredicateV& v) { return std::string(kind_name(v.d().predicates[v.i].kind)); }, "'static', 'fluent' or 'derived'")
            .field("arity", [](const PredicateV& v) { return v.d().predicates[v.i].arity; })
            .field("parameters", [](const PredicateV& v) { return parameter_list(v.t, v.d().predicates[v.i].params); })
            .finish();
        c.def("__repr__", [](const PredicateV& v) {
            const auto& p = v.d().predicates[v.i];
            return "Predicate(" + std::string(v.d().str(p.name)) + "/" + std::to_string(p.arity) + ", " + kind_name(p.kind) + ")";
        });
    }
    // --- Function
    {
        nb::class_<FunctionV> c(m, "Function");
        identity_protocol(c, "function");
        ClassDescriptor<FunctionV>{c}
            .field("name", [](const FunctionV& v) { return std::string(v.d().str(v.d().functions[v.i].name)); })
            .field("kind", [](const FunctionV& v) { return std::string(kind_name(v.d().functions[v.i].kind)); }, "'static', 'fluent' or 'auxiliary'")
            .field("arity", [](const FunctionV& v) { return v.d().functions[v.i].arity; })
            .field("parameters", [](const FunctionV& v) { return parameter_list(v.t, v.d().functions[v.i].params); })
            .finish();
        c.def("__repr__", [](const FunctionV& v) {
            const auto& f = v.d().functions[v.i];
            return "Function(" + std::string(v.d().str(f.name)) + "/" + std::to_string(f.arity) + ", " + kind_name(f.kind) + ")";
        });
    }
    // --- Literal
    {
        nb::class_<LiteralV> c(m, "Literal");
        identity_protocol(c, "literal");
        ClassDescriptor<LiteralV>{c}
            .field("predicate", [](const LiteralV& v) { return PredicateV{{v.t, v.d().literals[v.i].pred.v}}; })
            .field("positive", [](const LiteralV& v) { return v.d().literals[v.i].positive; })
            .field("terms", [](const LiteralV& v) {
                nb::list l;
                for (fm::Term x : v.d().terms_of(v.d().literals[v.i]))
                    l.append(term_object(v.t, x, v.scope));
                return l;
            }, "Object or Variable per argument")
            .field("is_ground", [](const LiteralV& v) {
                for (fm::Term x : v.d().terms_of(v.d().literals[v.i]))
                    if (!fm::is_object(x))
                        return false;
                return true;
            })
            .finish();
        c.def("__repr__", [](const LiteralV& v) { return "Literal" + literal_str(v.d(), v.d().literals[v.i], v.scope); });
        c.def("__str__", [](const LiteralV& v) { return literal_str(v.d(), v.d().literals[v.i], v.scope); });
    }
    // --- Expression
    {
        nb::class_<ExprV> c(m, "Expression");
        identity_protocol(c, "expr");
        ClassDescriptor<ExprV>{c}
            .field("op", [](const ExprV& v) {
                switch (v.d().exprs[v.i].op)
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
            .field("value", [](const ExprV& v) -> std::optional<f64> {
                const auto& e = v.d().exprs[v.i];
                return e.op == fm::ExprOp::Number ? std::optional<f64>(e.value) : std::nullopt;
            })
            .field("function", [](const ExprV& v) -> nb::object {
                const auto& e = v.d().exprs[v.i];
                return e.op == fm::ExprOp::Function ? nb::cast(FunctionV{{v.t, e.func.v}}) : nb::none();
            })
            .field("terms", [](const ExprV& v) {
                nb::list l;
                const auto& e = v.d().exprs[v.i];
                if (e.op == fm::ExprOp::Function)
                    for (fm::Term x : TaskData::slice(v.d().terms, e.terms))
                        l.append(term_object(v.t, x, v.scope));
                return l;
            })
            .field("children", [](const ExprV& v) {
                nb::list l;
                const auto& e = v.d().exprs[v.i];
                if (e.op == fm::ExprOp::Neg)
                    l.append(ExprV{{v.t, e.a}, v.scope});
                else if (e.op != fm::ExprOp::Number && e.op != fm::ExprOp::Function)
                {
                    l.append(ExprV{{v.t, e.a}, v.scope});
                    l.append(ExprV{{v.t, e.b}, v.scope});
                }
                return l;
            })
            .finish();
        c.def("__str__", [](const ExprV& v) { return expr_str(v.d(), v.i, v.scope); });
        c.def("__repr__", [](const ExprV& v) { return "Expression" + expr_str(v.d(), v.i, v.scope); });
    }
    // --- NumericConstraint
    {
        nb::class_<ConstraintV> c(m, "NumericConstraint");
        identity_protocol(c, "constraint");
        ClassDescriptor<ConstraintV>{c}
            .field("comparator", [](const ConstraintV& v) { return std::string(cmp_name(v.d().constraints[v.i].cmp)); })
            .field("lhs", [](const ConstraintV& v) { return ExprV{{v.t, v.d().constraints[v.i].lhs}, v.scope}; })
            .field("rhs", [](const ConstraintV& v) { return ExprV{{v.t, v.d().constraints[v.i].rhs}, v.scope}; })
            .finish();
        c.def("__str__", [](const ConstraintV& v) {
            const auto& k = v.d().constraints[v.i];
            return std::string("(") + cmp_name(k.cmp) + " " + expr_str(v.d(), k.lhs, v.scope) + " " + expr_str(v.d(), k.rhs, v.scope) + ")";
        });
    }
    // --- NumericEffect
    {
        nb::class_<NumericEffectV> c(m, "NumericEffect");
        c.def_prop_ro("index", [](const NumericEffectV& v) { return v.i; });
        ClassDescriptor<NumericEffectV>{c}
            .field("operator", [](const NumericEffectV& v) { return std::string(op_name(v.e().op)); })
            .field("function", [](const NumericEffectV& v) { return FunctionV{{v.t, v.e().func.v}}; })
            .field("terms", [](const NumericEffectV& v) {
                nb::list l;
                for (fm::Term x : TaskData::slice(v.d().terms, v.e().terms))
                    l.append(term_object(v.t, x, v.scope));
                return l;
            })
            .field("expression", [](const NumericEffectV& v) { return ExprV{{v.t, v.e().expr}, v.scope}; })
            .finish();
        c.def("__str__", [](const NumericEffectV& v) {
            const auto& e = v.e();
            std::string out = std::string("(") + op_name(e.op) + " (" + std::string(v.d().str(v.d().functions[e.func.v].name));
            for (fm::Term x : TaskData::slice(v.d().terms, e.terms))
                out += " " + term_str(v.d(), x, v.scope);
            return out + ") " + expr_str(v.d(), e.expr, v.scope) + ")";
        });
    }
    // --- Condition
    {
        nb::class_<ConditionV> c(m, "Condition");
        ClassDescriptor<ConditionV>{c}
            .field("literals", [](const ConditionV& v) { return literal_list(v.t, v.c.literals, v.scope); })
            .field("static_literals", [](const ConditionV& v) {
                nb::list l;
                for (u32 k = 0; k < v.c.literals.count; ++k)
                    if (v.t->predicates[v.t->literals[v.c.literals.begin + k].pred.v].kind == fm::PredKind::Static)
                        l.append(LiteralV{{v.t, v.c.literals.begin + k}, v.scope});
                return l;
            })
            .field("fluent_literals", [](const ConditionV& v) {
                nb::list l;
                for (u32 k = 0; k < v.c.literals.count; ++k)
                    if (v.t->predicates[v.t->literals[v.c.literals.begin + k].pred.v].kind == fm::PredKind::Fluent)
                        l.append(LiteralV{{v.t, v.c.literals.begin + k}, v.scope});
                return l;
            })
            .field("derived_literals", [](const ConditionV& v) {
                nb::list l;
                for (u32 k = 0; k < v.c.literals.count; ++k)
                    if (v.t->predicates[v.t->literals[v.c.literals.begin + k].pred.v].kind == fm::PredKind::Derived)
                        l.append(LiteralV{{v.t, v.c.literals.begin + k}, v.scope});
                return l;
            })
            .field("numeric_constraints", [](const ConditionV& v) { return constraint_list(v.t, v.c.constraints, v.scope); })
            .finish();
        c.def("__len__", [](const ConditionV& v) { return v.c.literals.count + v.c.constraints.count; });
        c.def("__str__", [](const ConditionV& v) {
            std::string out = "(and";
            for (const auto& l : v.t->literals_of(v.c))
                out += " " + literal_str(*v.t, l, v.scope);
            for (const auto& k : v.t->constraints_of(v.c))
                out += std::string(" (") + cmp_name(k.cmp) + " " + expr_str(*v.t, k.lhs, v.scope) + " " + expr_str(*v.t, k.rhs, v.scope) + ")";
            return out + ")";
        });
    }
    // --- ConditionalEffect
    {
        nb::class_<ConditionalEffectV> c(m, "ConditionalEffect");
        identity_protocol(c, "conditional_effect");
        auto scope_of = [](const ConditionalEffectV& v) { return Scope{v.schema_params, v.d().conditional_effects[v.i].extra_params}; };
        ClassDescriptor<ConditionalEffectV>{c}
            .field("parameters", [](const ConditionalEffectV& v) {
                return parameter_list(v.t, v.d().conditional_effects[v.i].extra_params, v.schema_params.count);
            }, "forall parameters (positions continue after the schema's)")
            .field("condition", [scope_of](const ConditionalEffectV& v) {
                return ConditionV{v.t, v.d().conditional_effects[v.i].condition, scope_of(v), "effect"};
            })
            .field("add_effects", [scope_of](const ConditionalEffectV& v) {
                nb::list l;
                const auto r = v.d().conditional_effects[v.i].effects;
                for (u32 k = 0; k < r.count; ++k)
                    if (v.d().literals[r.begin + k].positive)
                        l.append(LiteralV{{v.t, r.begin + k}, scope_of(v)});
                return l;
            })
            .field("delete_effects", [scope_of](const ConditionalEffectV& v) {
                nb::list l;
                const auto r = v.d().conditional_effects[v.i].effects;
                for (u32 k = 0; k < r.count; ++k)
                    if (!v.d().literals[r.begin + k].positive)
                        l.append(LiteralV{{v.t, r.begin + k}, scope_of(v)});
                return l;
            })
            .field("numeric_effects", [scope_of](const ConditionalEffectV& v) {
                nb::list l;
                const auto r = v.d().conditional_effects[v.i].numeric_effects;
                for (u32 k = 0; k < r.count; ++k)
                    l.append(NumericEffectV{{v.t, r.begin + k}, scope_of(v), std::nullopt});
                return l;
            })
            .field("auxiliary_effect", [scope_of](const ConditionalEffectV& v) -> nb::object {
                const auto& ce = v.d().conditional_effects[v.i];
                if (!ce.auxiliary)
                    return nb::none();
                return nb::cast(NumericEffectV{{v.t, 0}, scope_of(v), ce.auxiliary});
            }, "the total-cost effect, if any")
            .finish();
    }
    // --- Schema (action)
    {
        nb::class_<SchemaV> c(m, "Schema");
        identity_protocol(c, "schema");
        ClassDescriptor<SchemaV>{c}
            .field("name", [](const SchemaV& v) { return std::string(v.d().str(v.d().schemas[v.i].name)); })
            .field("arity", [](const SchemaV& v) { return v.d().schemas[v.i].arity(); })
            .field("original_arity", [](const SchemaV& v) { return v.d().schemas[v.i].original_arity; },
                   "parameters written in the PDDL; the rest were introduced by normalization")
            .field("parameters", [](const SchemaV& v) { return parameter_list(v.t, v.d().schemas[v.i].params); })
            .field("precondition", [](const SchemaV& v) {
                return ConditionV{v.t, v.d().schemas[v.i].precondition, schema_scope(v.d(), v.i), "schema"};
            })
            .field("effects", [](const SchemaV& v) {
                nb::list l;
                const auto& s = v.d().schemas[v.i];
                for (u32 k = 0; k < s.effects.count; ++k)
                    l.append(ConditionalEffectV{{v.t, s.effects.begin + k}, s.params});
                return l;
            }, "conditional effects; an unconditional effect has an empty condition and no parameters")
            .finish();
        c.def("__repr__", [](const SchemaV& v) {
            const auto& s = v.d().schemas[v.i];
            return "Schema(" + std::string(v.d().str(s.name)) + "/" + std::to_string(s.arity()) + ")";
        });
    }
    // --- Axiom
    {
        nb::class_<AxiomV> c(m, "Axiom");
        identity_protocol(c, "axiom");
        auto scope_of = [](const AxiomV& v) { return Scope{v.d().axioms[v.i].params, {}}; };
        ClassDescriptor<AxiomV>{c}
            .field("parameters", [](const AxiomV& v) { return parameter_list(v.t, v.d().axioms[v.i].params); })
            .field("head", [scope_of](const AxiomV& v) {
                // the head is stored inline; expose it through a temporary one-literal view
                const auto& a = v.d().axioms[v.i];
                nb::list terms;
                for (fm::Term x : v.d().terms_of(a.head))
                    terms.append(term_object(v.t, x, scope_of(v)));
                return nb::make_tuple(PredicateV{{v.t, a.head.pred.v}}, terms);
            }, "(derived predicate, terms)")
            .field("body", [scope_of](const AxiomV& v) { return ConditionV{v.t, v.d().axioms[v.i].body, scope_of(v), "axiom"}; })
            .field("from_problem", [](const AxiomV& v) { return v.d().axioms[v.i].from_problem; })
            .finish();
        c.def("__repr__", [scope_of](const AxiomV& v) {
            const auto& a = v.d().axioms[v.i];
            return "Axiom(" + literal_str(v.d(), a.head, scope_of(v)) + ")";
        });
    }
    // --- GroundAtom (initial state)
    {
        nb::class_<GroundAtomV> c(m, "GroundAtom");
        c.def_prop_ro("index", [](const GroundAtomV& v) { return v.i; });
        ClassDescriptor<GroundAtomV>{c}
            .field("predicate", [](const GroundAtomV& v) { return PredicateV{{v.t, v.a().pred.v}}; })
            .field("objects", [](const GroundAtomV& v) { return object_list(v.t, v.a().objects); })
            .finish();
        c.def("__str__", [](const GroundAtomV& v) {
            std::string out = "(" + std::string(v.d().str(v.d().predicates[v.a().pred.v].name));
            for (ObjectId o : v.d().objects_of(v.a()))
                out += " " + std::string(v.d().str(v.d().objects[o.v].name));
            return out + ")";
        });
        c.def("__repr__", [](const GroundAtomV& v) { return "GroundAtom(" + nb::cast<std::string>(nb::str(nb::cast(v))) + ")"; });
    }
    // --- GroundFunctionValue
    {
        nb::class_<GroundValueV> c(m, "GroundFunctionValue");
        ClassDescriptor<GroundValueV>{c}
            .field("function", [](const GroundValueV& v) { return FunctionV{{v.t, v.v().func.v}}; })
            .field("objects", [](const GroundValueV& v) { return object_list(v.t, v.v().objects); })
            .field("value", [](const GroundValueV& v) { return v.v().value; })
            .finish();
    }
    // --- Task (formalism root)
    {
        nb::class_<FormalismTask> c(m, "NormalizedTask", "The complete normalized task (read-only).");
        ClassDescriptor<FormalismTask>{c}
            .field("domain_name", [](const FormalismTask& v) { return v.t->domain_name; })
            .field("problem_name", [](const FormalismTask& v) { return v.t->problem_name; })
            .field("requirements", [](const FormalismTask& v) { return v.t->requirements; })
            .field("types", [](const FormalismTask& v) { return views_of<TypeV>(v.t, v.t->types); })
            .field("objects", [](const FormalismTask& v) { return views_of<ObjectV>(v.t, v.t->objects); })
            .field("predicates", [](const FormalismTask& v) { return views_of<PredicateV>(v.t, v.t->predicates); })
            .field("functions", [](const FormalismTask& v) { return views_of<FunctionV>(v.t, v.t->functions); })
            .field("schemas", [](const FormalismTask& v) { return views_of<SchemaV>(v.t, v.t->schemas); })
            .field("axioms", [](const FormalismTask& v) { return views_of<AxiomV>(v.t, v.t->axioms); })
            .field("static_init", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->static_init.size(); ++i)
                    l.append(GroundAtomV{{v.t, i}, false});
                return l;
            })
            .field("fluent_init", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->fluent_init.size(); ++i)
                    l.append(GroundAtomV{{v.t, i}, true});
                return l;
            })
            .field("static_values", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->static_values.size(); ++i)
                    l.append(GroundValueV{{v.t, i}, false});
                return l;
            })
            .field("fluent_values", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->fluent_values.size(); ++i)
                    l.append(GroundValueV{{v.t, i}, true});
                return l;
            })
            .field("auxiliary_initial", [](const FormalismTask& v) { return v.t->auxiliary_initial; })
            .field("goal", [](const FormalismTask& v) { return ConditionV{v.t, v.t->goal, Scope{}, "goal"}; })
            .field("metric", [](const FormalismTask& v) -> nb::object {
                if (!v.t->metric)
                    return nb::none();
                return nb::make_tuple(v.t->metric->minimize ? "minimize" : "maximize", ExprV{{v.t, v.t->metric->expr}, Scope{}});
            }, "('minimize'|'maximize', Expression) or None")
            .finish();
        c.def("to_text", [](const FormalismTask& v) { return formalism::write_task_text(*v.t); },
              "mymyr's normalized text format (formalism::write_task_text writes it).");
        c.def("__repr__", [](const FormalismTask& v) {
            return "NormalizedTask(" + std::to_string(v.t->objects.size()) + " objects, " + std::to_string(v.t->predicates.size()) +
                   " predicates, " + std::to_string(v.t->schemas.size()) + " schemas, " + std::to_string(v.t->axioms.size()) + " axioms)";
        });
    }

    m.def("read_task_text", [](const std::string& path) {
        return FormalismTask{std::make_shared<const TaskData>(formalism::read_task_text_file(path)), nullptr};
    }, "path"_a, "Load a task in mymyr's normalized text format (formalism::write_task_text writes it).");
}
}  // namespace mymyr::python
