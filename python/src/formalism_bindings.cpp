// Read-only Python views of the normalized task (TaskData): every entity is reachable from Python,
// the bindings generated from one entity-descriptor table.
//
// Design:
//   - A view is (shared_ptr<const TaskData>, index [, context]). Views keep the task alive, are immutable, hashable
//     and comparable by (task identity, kind, index), and are cheap to create. Nothing is interned or cached.
//   - Each class is declared once in a descriptor table (class name, then named fields with getters). The table
//     drives the property registration, __repr__, and the `fields` class attribute that the coverage test walks.
//   - Conditions, literals, atoms and expressions are the formula values of py_formula.hpp (formula_bindings.cpp):
//     a schema's precondition is a ConjunctiveCondition, the goal a GroundCondition, the initial atoms GroundAtoms.
//     Terms render as Object views or Variables (position, name and types from the enclosing parameter list).

#include "formalism_task.hpp"
#include "formalism_views.hpp"
#include "py_formula.hpp"

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

namespace
{
namespace fm = formalism;

// ------------------------------------------------------------------------------------------------ view types

using View = FormalismView;

using TypeV = TypeView;
using ObjectV = ObjectView;
using PredicateV = PredicateView;
using FunctionV = FunctionView;
struct SchemaV : View {};
struct AxiomV : View {};

/// A parameter in scope: index into TaskData::params, plus its position in the enclosing parameter list.
struct ParameterV : View
{
    u32 position = 0;
};

struct NumericEffectV : View  // i = index into numeric_effects, or aux (stored inline)
{
    VariablesPtr vars;  // the schema's parameters, then the conditional effect's
    std::optional<fm::NumericEffect> inline_effect;  // for auxiliary effects stored in ConditionalEffect
    [[nodiscard]] const fm::NumericEffect& e() const { return inline_effect ? *inline_effect : d().numeric_effects[i]; }
};
struct ConditionalEffectV : View  // i = index into conditional_effects
{
    u32 schema = 0;
    fm::Range schema_params;
};
struct GroundValueV : View
{
    bool fluent = false;
    [[nodiscard]] const fm::GroundFunctionValue& v() const { return fluent ? d().fluent_values[i] : d().static_values[i]; }
};

const char* kind_name(fm::PredKind k) { return k == fm::PredKind::Static ? "static" : k == fm::PredKind::Fluent ? "fluent" : "derived"; }
const char* kind_name(fm::FuncKind k) { return k == fm::FuncKind::Static ? "static" : k == fm::FuncKind::Fluent ? "fluent" : "auxiliary"; }
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

std::string term_str(const TaskData& d, fm::Term x, const Variables& vars)
{
    if (fm::is_object(x))
        return std::string(d.str(d.objects[fm::term_object(x).v].name));
    const u32 p = fm::term_parameter(x);
    return "?" + (p < vars.size() ? vars[p].name : std::to_string(p));
}

/// The literals of a pool range as Literal values over `vars`; `positive`: only those of that polarity.
nb::typed<nb::tuple, PyLiteral, nb::ellipsis> literal_tuple(const DataPtr& t, fm::Range r, const VariablesPtr& vars, std::optional<bool> positive)
{
    nb::list l;
    const FormulaOwner o = bare_owner(t);
    for (u32 k = 0; k < r.count; ++k)
    {
        const fm::Literal& x = t->literals[r.begin + k];
        if (positive && x.positive != *positive)
            continue;
        const auto terms = t->terms_of(x);
        l.append(make_literal(o, x.pred, x.positive, std::vector<fm::Term>(terms.begin(), terms.end()), vars));
    }
    return nb::typed<nb::tuple, PyLiteral, nb::ellipsis>(nb::tuple(l));
}

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
    cls.def_prop_ro("index", [](const V& a) { return a.i; },
                    "The position in the normalized task's table of this kind (NormalizedTask.types, objects, ...); "
                    "views of one task are equal when their indices are.");
}

template<class V, class Vec>
nb::list views_of(const DataPtr& t, const Vec& pool)
{
    nb::list l;
    for (u32 i = 0; i < pool.size(); ++i)
        l.append(V{{t, i}});
    return l;
}

nb::list parameter_list(const DataPtr& t, fm::Range r, u32 offset = 0)
{
    nb::list l;
    for (u32 k = 0; k < r.count; ++k)
        l.append(ParameterV{{t, r.begin + k}, offset + k});
    return l;
}
nb::list type_list(const DataPtr& t, fm::Range r)
{
    nb::list l;
    for (TypeId ty : TaskData::slice(t->type_ids, r))
        l.append(TypeV{{t, ty.v}});
    return l;
}
nb::list object_list(const DataPtr& t, fm::Range r)
{
    nb::list l;
    for (ObjectId o : TaskData::slice(t->object_ids, r))
        l.append(ObjectV{{t, o.v}});
    return l;
}
nb::typed<nb::tuple, PyGroundAtom, nb::ellipsis> ground_atoms(const DataPtr& t, const std::vector<fm::GroundAtom>& atoms)
{
    nb::list l;
    const FormulaOwner o = bare_owner(t);
    for (const fm::GroundAtom& a : atoms)
    {
        const auto objects = t->objects_of(a);
        l.append(make_ground_atom(o, GroundAtom{a.pred, std::vector<ObjectId>(objects.begin(), objects.end())}));
    }
    return nb::typed<nb::tuple, PyGroundAtom, nb::ellipsis>(nb::tuple(l));
}
}  // namespace

// ------------------------------------------------------------------------------------------------ Task-level wrapper

void bind_formalism(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_formalism", "Read-only views of the normalized task (complete: every entity)");
    bind_formulas(m, parent);

    // --- Type
    {
        nb::class_<TypeV> c(m, "Type",
                            "A type of the normalized task (NormalizedTask.types).");
        identity_protocol(c, "type");
        ClassDescriptor<TypeV>{c}
            .field("name", [](const TypeV& v) { return std::string(v.d().str(v.d().types[v.i].name)); },
                   "The type name.")
            .field("bases", [](const TypeV& v) { return type_list(v.t, v.d().types[v.i].bases); }, "direct supertypes")
            .finish();
        c.def("__repr__", [](const TypeV& v) { return "Type(" + std::string(v.d().str(v.d().types[v.i].name)) + ")"; });
    }
    // --- Object
    {
        nb::class_<ObjectV> c(m, "Object",
                              "An object or constant of the normalized task (NormalizedTask.objects).");
        identity_protocol(c, "object");
        ClassDescriptor<ObjectV>{c}
            .field("name", [](const ObjectV& v) { return std::string(v.d().str(v.d().objects[v.i].name)); },
                   "The object name.")
            .field("types", [](const ObjectV& v) { return type_list(v.t, v.d().objects[v.i].types); },
                   "The Types the object was declared with.")
            .field("is_constant", [](const ObjectV& v) { return v.d().objects[v.i].constant; },
                   "Whether the domain declares it (a constant) rather than the problem.")
            .finish();
        c.def("__repr__", [](const ObjectV& v) { return "Object(" + std::string(v.d().str(v.d().objects[v.i].name)) + ")"; });
        c.def("__str__", [](const ObjectV& v) { return std::string(v.d().str(v.d().objects[v.i].name)); });
    }
    // --- Parameter
    {
        nb::class_<ParameterV> c(m, "Parameter",
                                 "A parameter of a predicate, function, schema or axiom.");
        identity_protocol(c, "parameter");
        ClassDescriptor<ParameterV>{c}
            .field("name", [](const ParameterV& v) { return std::string(v.d().str(v.d().params[v.i].name)); },
                   "The parameter name without the '?'.")
            .field("position", [](const ParameterV& v) { return v.position; },
                   "The 0-based position in its owner's parameter list.")
            .field("types", [](const ParameterV& v) { return type_list(v.t, v.d().params[v.i].types); },
                   "The Types of the parameter (several for 'either').")
            .finish();
        c.def("__repr__", [](const ParameterV& v) { return "Parameter(?" + std::string(v.d().str(v.d().params[v.i].name)) + ")"; });
    }
    // --- Predicate
    {
        nb::class_<PredicateV> c(m, "Predicate",
                                 "A predicate of the normalized task (NormalizedTask.predicates).");
        identity_protocol(c, "predicate");
        ClassDescriptor<PredicateV>{c}
            .field("name", [](const PredicateV& v) { return std::string(v.d().str(v.d().predicates[v.i].name)); },
                   "The predicate name.")
            .field("kind", [](const PredicateV& v) { return std::string(kind_name(v.d().predicates[v.i].kind)); }, "'static', 'fluent' or 'derived'")
            .field("arity", [](const PredicateV& v) { return v.d().predicates[v.i].arity; },
                   "The number of parameters.")
            .field("parameters", [](const PredicateV& v) { return parameter_list(v.t, v.d().predicates[v.i].params); },
                   "The Parameters.")
            .finish();
        c.def("__repr__", [](const PredicateV& v) {
            const auto& p = v.d().predicates[v.i];
            return "Predicate(" + std::string(v.d().str(p.name)) + "/" + std::to_string(p.arity) + ", " + kind_name(p.kind) + ")";
        });
    }
    // --- Function
    {
        nb::class_<FunctionV> c(m, "Function",
                                "A numeric function of the normalized task (NormalizedTask.functions).");
        identity_protocol(c, "function");
        ClassDescriptor<FunctionV>{c}
            .field("name", [](const FunctionV& v) { return std::string(v.d().str(v.d().functions[v.i].name)); },
                   "The function name.")
            .field("kind", [](const FunctionV& v) { return std::string(kind_name(v.d().functions[v.i].kind)); }, "'static', 'fluent' or 'auxiliary'")
            .field("arity", [](const FunctionV& v) { return v.d().functions[v.i].arity; },
                   "The number of parameters.")
            .field("parameters", [](const FunctionV& v) { return parameter_list(v.t, v.d().functions[v.i].params); },
                   "The Parameters.")
            .finish();
        c.def("__repr__", [](const FunctionV& v) {
            const auto& f = v.d().functions[v.i];
            return "Function(" + std::string(v.d().str(f.name)) + "/" + std::to_string(f.arity) + ", " + kind_name(f.kind) + ")";
        });
    }
    // --- NumericEffect
    {
        nb::class_<NumericEffectV> c(m, "NumericEffect",
                                     "A numeric effect of a schema: a function updated by an expression.");
        c.def_prop_ro("index", [](const NumericEffectV& v) { return v.i; },
                      "The effect's index in the numeric effect table of its owner.");
        ClassDescriptor<NumericEffectV>{c}
            .field("operator", [](const NumericEffectV& v) { return std::string(op_name(v.e().op)); },
                   "The update: 'assign', 'increase', 'decrease', 'scale-up' or 'scale-down'.")
            .field("function", [](const NumericEffectV& v) { return FunctionV{{v.t, v.e().func.v}}; },
                   "The Function it updates.")
            .field("terms", [](const NumericEffectV& v) {
                nb::list l;
                for (fm::Term x : TaskData::slice(v.d().terms, v.e().terms))
                    l.append(term_object(bare_owner(v.t), x, v.vars));
                return nb::tuple(l);
            },
                   "The arguments of the updated function (Objects or the schema's Variables).")
            .field("expression", [](const NumericEffectV& v) { return PyExpression{task_pool(bare_owner(v.t), v.vars), v.e().expr}; },
                   "The Expression of the new value.")
            .finish();
        c.def("__str__", [](const NumericEffectV& v) {
            const auto& e = v.e();
            std::string out = std::string("(") + op_name(e.op) + " (" + std::string(v.d().str(v.d().functions[e.func.v].name));
            for (fm::Term x : TaskData::slice(v.d().terms, e.terms))
                out += " " + term_str(v.d(), x, *v.vars);
            return out + ") " + nb::cast<std::string>(nb::str(nb::cast(PyExpression{task_pool(bare_owner(v.t), v.vars), e.expr}))) + ")";
        });
    }
    // --- ConditionalEffect
    {
        nb::class_<ConditionalEffectV> c(m, "ConditionalEffect",
                                         "An effect of a schema with its own parameters and condition (an "
                                         "unconditional effect has neither).");
        identity_protocol(c, "conditional_effect");
        auto vars_of = [](const ConditionalEffectV& v) { return scope_variables(v.d(), v.schema_params, v.d().conditional_effects[v.i].extra_params); };
        ClassDescriptor<ConditionalEffectV>{c}
            .field("parameters", [](const ConditionalEffectV& v) {
                return parameter_list(v.t, v.d().conditional_effects[v.i].extra_params, v.schema_params.count);
            }, "forall parameters (positions continue after the schema's)")
            .field("condition", [](const ConditionalEffectV& v) {
                return Arg<PyConjunctiveCondition>(make_conjunctive_condition(bare_owner(v.t), ConjunctiveCondition::effect_condition(v.d(), SchemaId{v.schema}, v.i)));
            }, "a ConjunctiveCondition over the schema's parameters followed by the forall parameters")
            .field("add_effects", [vars_of](const ConditionalEffectV& v) {
                return literal_tuple(v.t, v.d().conditional_effects[v.i].effects, vars_of(v), true);
            },
                   "The atoms it makes true.")
            .field("delete_effects", [vars_of](const ConditionalEffectV& v) {
                return literal_tuple(v.t, v.d().conditional_effects[v.i].effects, vars_of(v), false);
            },
                   "The atoms it makes false.")
            .field("numeric_effects", [vars_of](const ConditionalEffectV& v) {
                nb::list l;
                const auto r = v.d().conditional_effects[v.i].numeric_effects;
                const VariablesPtr vars = vars_of(v);
                for (u32 k = 0; k < r.count; ++k)
                    l.append(NumericEffectV{{v.t, r.begin + k}, vars, std::nullopt});
                return l;
            },
                   "Its NumericEffects.")
            .field("auxiliary_effect", [vars_of](const ConditionalEffectV& v) -> nb::object {
                const auto& ce = v.d().conditional_effects[v.i];
                if (!ce.auxiliary)
                    return nb::none();
                return nb::cast(NumericEffectV{{v.t, 0}, vars_of(v), ce.auxiliary});
            }, "the total-cost effect, if any")
            .finish();
    }
    // --- Schema (action)
    {
        nb::class_<SchemaV> c(m, "Schema",
                              "An action schema of the normalized task (NormalizedTask.schemas); a PDDL action may "
                              "become several (one per branch of a disjunctive precondition).");
        identity_protocol(c, "schema");
        ClassDescriptor<SchemaV>{c}
            .field("name", [](const SchemaV& v) { return std::string(v.d().str(v.d().schemas[v.i].name)); },
                   "The name of the PDDL action it comes from.")
            .field("arity", [](const SchemaV& v) { return v.d().schemas[v.i].arity(); },
                   "The number of parameters, those normalization added included.")
            .field("original_arity", [](const SchemaV& v) { return v.d().schemas[v.i].original_arity; },
                   "parameters written in the PDDL; the rest were introduced by normalization")
            .field("parameters", [](const SchemaV& v) { return parameter_list(v.t, v.d().schemas[v.i].params); },
                   "The Parameters.")
            .field("precondition", [](const SchemaV& v) {
                return Arg<PyConjunctiveCondition>(make_conjunctive_condition(bare_owner(v.t), ConjunctiveCondition::precondition(v.d(), SchemaId{v.i})));
            }, "a ConjunctiveCondition over the parameters")
            .field("effects", [](const SchemaV& v) {
                nb::list l;
                const auto& s = v.d().schemas[v.i];
                for (u32 k = 0; k < s.effects.count; ++k)
                    l.append(ConditionalEffectV{{v.t, s.effects.begin + k}, v.i, s.params});
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
        nb::class_<AxiomV> c(m, "Axiom",
                             "An axiom: a rule of a derived predicate (NormalizedTask.axioms).");
        identity_protocol(c, "axiom");
        ClassDescriptor<AxiomV>{c}
            .field("parameters", [](const AxiomV& v) { return parameter_list(v.t, v.d().axioms[v.i].params); },
                   "The Parameters of the rule.")
            .field("head", [](const AxiomV& v) {
                const auto& a = v.d().axioms[v.i];
                const auto terms = v.d().terms_of(a.head);
                return Arg<PyLiftedAtom>(make_lifted_atom(bare_owner(v.t), a.head.pred, std::vector<fm::Term>(terms.begin(), terms.end()),
                                                          scope_variables(v.d(), a.params)));
            }, "an Atom of a derived predicate over the parameters")
            .field("body", [](const AxiomV& v) {
                return Arg<PyConjunctiveCondition>(make_conjunctive_condition(bare_owner(v.t), ConjunctiveCondition::axiom_body(v.d(), v.i)));
            }, "a ConjunctiveCondition over the parameters")
            .field("from_problem", [](const AxiomV& v) { return v.d().axioms[v.i].from_problem; },
                   "Whether normalization made the rule from the problem (a goal formula compiled into a derived "
                   "predicate) rather than from the domain.")
            .finish();
        c.def("__repr__", [](const AxiomV& v) {
            const auto& a = v.d().axioms[v.i];
            const VariablesPtr vars = scope_variables(v.d(), a.params);
            return "Axiom(" + literal_text(v.d(), a.head.pred, true, v.d().terms_of(a.head), vars.get()) + ")";
        });
    }
    // --- GroundFunctionValue
    {
        nb::class_<GroundValueV> c(m, "GroundFunctionValue",
                                   "The initial value of a ground function term.");
        ClassDescriptor<GroundValueV>{c}
            .field("function", [](const GroundValueV& v) { return FunctionV{{v.t, v.v().func.v}}; },
                   "The Function.")
            .field("objects", [](const GroundValueV& v) { return object_list(v.t, v.v().objects); },
                   "The Objects of the arguments.")
            .field("value", [](const GroundValueV& v) { return v.v().value; },
                   "The value.")
            .finish();
    }
    // --- Task (formalism root)
    {
        nb::class_<FormalismTask> c(m, "NormalizedTask", "The complete normalized task (read-only).");
        ClassDescriptor<FormalismTask>{c}
            .field("domain_name", [](const FormalismTask& v) { return v.t->domain_name; },
                   "The name of the PDDL domain.")
            .field("problem_name", [](const FormalismTask& v) { return v.t->problem_name; },
                   "The name of the PDDL problem.")
            .field("requirements", [](const FormalismTask& v) { return v.t->requirements; },
                   "The PDDL requirements the domain and problem declare.")
            .field("types", [](const FormalismTask& v) { return views_of<TypeV>(v.t, v.t->types); },
                   "Every Type, 'object' and 'number' included.")
            .field("objects", [](const FormalismTask& v) { return views_of<ObjectV>(v.t, v.t->objects); },
                   "Every Object: the domain's constants and the problem's objects.")
            .field("predicates", [](const FormalismTask& v) { return views_of<PredicateV>(v.t, v.t->predicates); },
                   "Every Predicate (static, fluent and derived).")
            .field("functions", [](const FormalismTask& v) { return views_of<FunctionV>(v.t, v.t->functions); },
                   "Every Function.")
            .field("schemas", [](const FormalismTask& v) { return views_of<SchemaV>(v.t, v.t->schemas); },
                   "Every action Schema.")
            .field("axioms", [](const FormalismTask& v) { return views_of<AxiomV>(v.t, v.t->axioms); },
                   "Every Axiom.")
            .field("static_init", [](const FormalismTask& v) { return ground_atoms(v.t, v.t->static_init); }, "GroundAtoms")
            .field("fluent_init", [](const FormalismTask& v) { return ground_atoms(v.t, v.t->fluent_init); }, "GroundAtoms")
            .field("static_values", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->static_values.size(); ++i)
                    l.append(GroundValueV{{v.t, i}, false});
                return l;
            },
                   "The GroundFunctionValues of the static functions.")
            .field("fluent_values", [](const FormalismTask& v) {
                nb::list l;
                for (u32 i = 0; i < v.t->fluent_values.size(); ++i)
                    l.append(GroundValueV{{v.t, i}, true});
                return l;
            },
                   "The initial GroundFunctionValues of the fluent functions.")
            .field("auxiliary_initial", [](const FormalismTask& v) { return v.t->auxiliary_initial; },
                   "The initial value of total-cost when the problem declares one, else None.")
            .field("goal", [](const FormalismTask& v) { return Arg<PyGroundCondition>(make_ground_condition(bare_owner(v.t), GroundCondition::goal(*v.t))); },
                   "a GroundCondition")
            .field("encoding_goal", [](const FormalismTask& v) { return make_encoding_goal(bare_owner(v.t)); },
                   "tuple of GroundLiteral: the goal as written, expanded (Task.encoding_goal)")
            .field("encoding_goal_complete", [](const FormalismTask& v) { return v.t->encoding_goal_complete; },
                   "Whether the goal's ground literals (quantifiers expanded, disjunctions decided by static "
                   "atoms) are the whole goal; False when a disjunction or a numeric constraint was left out of "
                   "them.")
            .field("encoding_goal_unsatisfiable", [](const FormalismTask& v) { return v.t->encoding_goal_unsatisfiable; },
                   "Whether a goal literal over a static predicate is false, so the goal can never hold.")
            .field("metric", [](const FormalismTask& v) -> nb::object {
                if (!v.t->metric)
                    return nb::none();
                return nb::make_tuple(v.t->metric->minimize ? "minimize" : "maximize",
                                     PyExpression{task_pool(bare_owner(v.t), nullptr), v.t->metric->expr});
            }, "('minimize'|'maximize', Expression) or None")
            .finish();
        c.def("to_text", [](const FormalismTask& v, bool encoding_goal) { return formalism::write_task_text(*v.t, encoding_goal); },
              "encoding_goal"_a = false,
              "mymyr's normalized text format (formalism::write_task_text writes it). With encoding_goal, a final section "
              "keeps the encoding goal, which Task.from_text then reads; without it, from_text derives the encoding goal "
              "from the planner's goal.");
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
