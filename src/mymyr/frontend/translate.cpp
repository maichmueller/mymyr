// Port of mimir's ToMimirStructures (src/formalism/to_mimir_structures.cpp) and
// EncodeParameterIndexInVariables (src/formalism/translator/encode_parameter_index_in_variables.cpp plus the default
// traversal of translator/recursive_base.hpp), replayed on index-only tables. See translate.hpp for the overview.
// Comments of the form "mimir:" point at the behaviour being reproduced.

#include "translate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>

namespace mymyr::frontend::detail
{
namespace
{
using namespace formalism;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("mymyr frontend: " + msg); }

enum : u8
{
    kStatic = 0,
    kFluent = 1,
    kDerived = 2,  // predicates
    kAuxiliary = 2,  // functions
};

/// Kinds of the predicates and functions visible to a translation (domain, plus the problem's derived predicates).
struct Kinds
{
    std::vector<u8> pred;
    std::vector<u8> func;
    std::vector<u32> pred_arity;
};

u32 key_u32(u64 v) { return static_cast<u32>(v); }

/// Entity construction shared by both passes: the parts of mimir's `Repositories::get_or_create_*` that canonicalize
/// (sorting by index, grounding nullary parts) and `Repositories::ground` (constant folding).
class Core
{
public:
    Core(Repo& r, const Kinds& k) : r(r), k(k) {}

    // ------------------------------------------------------------ ground function expressions
    u32 gnum(f64 v) { return r.gfe.id(Key{kFeNumber, r.gnum.id(Key{f64_bits(v)})}); }
    u32 gbin(u64 op, u32 a, u32 b) { return r.gfe.id(Key{kFeBinary, r.gbin.id(Key{op, a, b})}); }
    u32 gmulti(u64 op, std::vector<u32> xs)
    {
        // mimir: sorted by the index of the variant alternative (each alternative has its own index space), so
        // operands of different alternatives can tie. stable_sort keeps ties in operand order on every standard
        // library; up to 16 operands that is also what mimir's libstdc++ std::sort does.
        std::stable_sort(xs.begin(), xs.end(), [&](u32 a, u32 b) { return r.gfe[a].back() < r.gfe[b].back(); });
        Key key{op};
        key.insert(key.end(), xs.begin(), xs.end());
        return r.gfe.id(Key{kFeMulti, r.gmulti.id(key)});
    }
    u32 gminus(u32 a) { return r.gfe.id(Key{kFeMinus, r.gminus.id(Key{a})}); }
    u32 gfunc(u32 f, std::span<const u32> objs)
    {
        const u8 kind = k.func.at(f);
        Key key{f};
        key.insert(key.end(), objs.begin(), objs.end());
        return r.gfe.id(Key{kFeFunction, kind, r.gfunc[kind].id(key)});
    }
    bool gis_number(u32 g) const { return r.gfe[g][0] == kFeNumber; }
    f64 gnumber(u32 g) const { return bits_f64(r.gnum[key_u32(r.gfe[g][1])][0]); }

    // ------------------------------------------------------------ lifted function expressions
    u32 fnum(f64 v) { return r.fe.id(Key{kFeNumber, f64_bits(v)}); }
    u32 fbin(u64 op, u32 a, u32 b)
    {
        // mimir: operands of commutative operators are ordered by index
        if ((op == static_cast<u64>(loki::BinaryOperatorEnum::MUL) || op == static_cast<u64>(loki::BinaryOperatorEnum::PLUS)) && a > b)
            std::swap(a, b);
        return r.fe.id(Key{kFeBinary, op, a, b});
    }
    u32 fmulti(u64 op, std::vector<u32> xs)
    {
        std::sort(xs.begin(), xs.end());
        Key key{kFeMulti, op};
        key.insert(key.end(), xs.begin(), xs.end());
        return r.fe.id(key);
    }
    u32 fminus(u32 a) { return r.fe.id(Key{kFeMinus, a}); }
    u32 ffunc(u32 f, std::span<const u64> terms)
    {
        Key key{kFeFunction, f};
        key.insert(key.end(), terms.begin(), terms.end());
        return r.fe.id(key);
    }

    static f64 eval_binary(u64 op, f64 a, f64 b)
    {
        switch (static_cast<loki::BinaryOperatorEnum>(op))
        {
            case loki::BinaryOperatorEnum::PLUS: return a + b;
            case loki::BinaryOperatorEnum::MINUS: return a - b;
            case loki::BinaryOperatorEnum::MUL: return a * b;
            case loki::BinaryOperatorEnum::DIV: return b == 0 ? std::numeric_limits<f64>::quiet_NaN() : a / b;  // undefined
        }
        fail("bad binary operator");
    }
    static f64 eval_multi(u64 op, f64 a, f64 b)
    {
        return static_cast<loki::MultiOperatorEnum>(op) == loki::MultiOperatorEnum::PLUS ? a + b : a * b;
    }

    /// mimir: Repositories::ground(FunctionExpression, {}) -- used for nullary numeric constraints; folds constants.
    /// Unlike mimir, a unary minus over a non-constant operand is kept (mimir drops it, which negates the constraint).
    u32 ground(u32 f)
    {
        const Key& key = r.fe[f];
        switch (key[0])
        {
            case kFeNumber: return gnum(bits_f64(key[1]));
            case kFeBinary:
            {
                const u32 a = ground(key_u32(key[2]));
                const u32 b = ground(key_u32(key[3]));
                if (gis_number(a) && gis_number(b))
                    return gnum(eval_binary(key[1], gnumber(a), gnumber(b)));
                return gbin(key[1], a, b);
            }
            case kFeMulti:
            {
                std::vector<u32> numbers, others;
                for (size_t i = 2; i < key.size(); ++i)
                {
                    const u32 g = ground(key_u32(key[i]));
                    (gis_number(g) ? numbers : others).push_back(g);
                }
                if (!numbers.empty())
                {
                    f64 v = gnumber(numbers.front());
                    for (size_t i = 1; i < numbers.size(); ++i)
                        v = eval_multi(key[1], v, gnumber(numbers[i]));
                    others.push_back(gnum(v));
                }
                return gmulti(key[1], std::move(others));
            }
            case kFeMinus:
            {
                const u32 g = ground(key_u32(key[1]));
                return gis_number(g) ? gnum(-gnumber(g)) : gminus(g);
            }
            case kFeFunction:
            {
                std::vector<u32> objs;
                for (size_t i = 2; i < key.size(); ++i)
                {
                    if (!is_obj_term(key[i]))
                        fail("grounding a nullary constraint over a variable");
                    objs.push_back(term_obj(key[i]));
                }
                return gfunc(key_u32(key[1]), objs);
            }
        }
        fail("bad function expression");
    }

    // ------------------------------------------------------------ literals, conditions, effects
    u32 glit(u32 pred, bool pos, std::span<const u32> objs)
    {
        Key key{pred, pos ? 1u : 0u};
        key.insert(key.end(), objs.begin(), objs.end());
        return r.glit.id(key);
    }
    u32 lit_arity(u32 l) const { return static_cast<u32>(r.lit[l].size() - 2); }
    u8 lit_kind(u32 l) const { return k.pred.at(key_u32(r.lit[l][0])); }

    /// mimir: Repositories::get_or_create_conjunctive_condition
    u32 cond(const std::vector<u32>& params, std::vector<u32> lits[3], std::vector<u32> ncs)
    {
        Repo::CondExtra extra;
        for (int kind = 0; kind < 3; ++kind)
        {
            std::sort(lits[kind].begin(), lits[kind].end());
            for (u32 l : lits[kind])
                if (lit_arity(l) == 0)
                    extra.nullary[kind].push_back(glit(key_u32(r.lit[l][0]), r.lit[l][1] != 0, {}));
            std::sort(extra.nullary[kind].begin(), extra.nullary[kind].end());
        }
        for (u32 n : ncs)  // mimir: grounded before the constraints are sorted
            if (r.nc[n][3] == 0)
            {
                const Key& nk = r.nc[n];
                const u32 a = ground(key_u32(nk[1]));
                const u32 b = ground(key_u32(nk[2]));
                extra.gncs.push_back(r.gnc.id(Key{nk[0], a, b}));
            }
        std::sort(ncs.begin(), ncs.end());
        std::sort(extra.gncs.begin(), extra.gncs.end());

        Key key{params.size()};
        key.insert(key.end(), params.begin(), params.end());
        for (int kind = 0; kind < 3; ++kind)
        {
            key.push_back(lits[kind].size());
            key.insert(key.end(), lits[kind].begin(), lits[kind].end());
        }
        key.push_back(ncs.size());
        key.insert(key.end(), ncs.begin(), ncs.end());
        auto [id, created] = r.cond.get_or_create(key);
        if (created)
            r.cond_extra.push_back(std::move(extra));
        return id;
    }

    /// mimir: Repositories::get_or_create_conjunctive_effect
    u32 ceff(const std::vector<u32>& params, std::vector<u32> lits, std::vector<u32> nes, std::optional<u32> aux)
    {
        std::sort(lits.begin(), lits.end());
        std::sort(nes.begin(), nes.end());
        Key key{params.size()};
        key.insert(key.end(), params.begin(), params.end());
        key.push_back(lits.size());
        key.insert(key.end(), lits.begin(), lits.end());
        key.push_back(nes.size());
        key.insert(key.end(), nes.begin(), nes.end());
        key.push_back(aux ? 1 : 0);
        key.push_back(aux.value_or(0));
        return r.ceff.id(key);
    }

    u32 ce(u32 c, u32 e) { return r.ce.id(Key{c, e}); }

    Repo& r;
    const Kinds& k;
};

/// Decoded conjunctive condition key.
struct CondView
{
    std::vector<u32> params;
    std::vector<u32> lits[3];
    std::vector<u32> ncs;
};
CondView decode_cond(const Key& key)
{
    CondView v;
    size_t i = 0;
    auto take = [&](std::vector<u32>& out)
    {
        const size_t n = key[i++];
        for (size_t j = 0; j < n; ++j)
            out.push_back(key_u32(key[i++]));
    };
    take(v.params);
    for (auto& l : v.lits)
        take(l);
    take(v.ncs);
    return v;
}

struct CeffView
{
    std::vector<u32> params, lits, nes;
    std::optional<u32> aux;
};
CeffView decode_ceff(const Key& key)
{
    CeffView v;
    size_t i = 0;
    auto take = [&](std::vector<u32>& out)
    {
        const size_t n = key[i++];
        for (size_t j = 0; j < n; ++j)
            out.push_back(key_u32(key[i++]));
    };
    take(v.params);
    take(v.lits);
    take(v.nes);
    if (key[i] != 0)
        v.aux = key_u32(key[i + 1]);
    return v;
}

struct Action1
{
    std::string name;
    u32 original_arity = 0;
    std::vector<u32> params;
    u32 cond = 0;
    std::vector<u32> ces;
};

std::string strip_question(std::string_view s) { return std::string(!s.empty() && s[0] == '?' ? s.substr(1) : s); }

/// mimir: which predicates are fluent / derived and which functions appear in effects (ToMimirStructures::prepare).
struct Classifier
{
    absl::flat_hash_set<std::string>& fluent;
    absl::flat_hash_set<std::string>& derived;
    absl::flat_hash_set<std::string>& effect_functions;

    void condition(loki::Condition c)
    {
        if (const auto* a = std::get_if<loki::ConditionAnd>(&c->get_condition()))
        {
            for (const auto& part : (*a)->get_conditions())
                if (!std::holds_alternative<loki::ConditionLiteral>(part->get_condition())
                    && !std::holds_alternative<loki::ConditionNumericConstraint>(part->get_condition()))
                    throw std::logic_error("Expected literal in conjunctive condition.");
        }
        else if (!std::holds_alternative<loki::ConditionLiteral>(c->get_condition())
                 && !std::holds_alternative<loki::ConditionNumericConstraint>(c->get_condition()))
            throw std::logic_error("Expected conjunctive condition.");
    }
    void effect(loki::Effect effect)
    {
        auto one = [&](loki::Effect e)
        {
            if (const auto* f = std::get_if<loki::EffectCompositeForall>(&e->get_effect()))
                e = (*f)->get_effect();
            if (const auto* w = std::get_if<loki::EffectCompositeWhen>(&e->get_effect()))
            {
                const auto& c = (*w)->get_condition()->get_condition();
                if (const auto* a = std::get_if<loki::ConditionAnd>(&c))
                    for (const auto& part : (*a)->get_conditions())
                        if (!std::holds_alternative<loki::ConditionLiteral>(part->get_condition())
                            && !std::holds_alternative<loki::ConditionNumericConstraint>(part->get_condition()))
                            throw std::logic_error("Expected literal in conjunctive condition.");
                e = (*w)->get_effect();
            }
            if (const auto* l = std::get_if<loki::EffectLiteral>(&e->get_effect()))
                fluent.insert((*l)->get_literal()->get_atom()->get_predicate()->get_name());
            else if (const auto* n = std::get_if<loki::EffectNumeric>(&e->get_effect()))
                effect_functions.insert((*n)->get_function()->get_function_skeleton()->get_name());
            else
                throw std::logic_error("Expected simple effect.");
        };
        if (const auto* a = std::get_if<loki::EffectAnd>(&effect->get_effect()))
            for (const auto& e : (*a)->get_effects())
                one(e);
        else
            one(effect);
    }
    void domain(const loki::Domain& d)
    {
        for (const auto& a : d->get_actions())
        {
            if (a->get_condition())
                condition(*a->get_condition());
            if (a->get_effect())
                effect(*a->get_effect());
        }
        for (const auto& x : d->get_axioms())
        {
            condition(x->get_condition());
            derived.insert(x->get_literal()->get_atom()->get_predicate()->get_name());
        }
    }
};

// ================================================================================================ pass 1 (loki -> R1)

/// Lookups from loki entities to TaskData ids for one translation.
struct Lookups
{
    const DomainState* ds = nullptr;
    const ObjectMap* objects = nullptr;
    const absl::flat_hash_map<const loki::PredicateImpl*, u32>* problem_preds = nullptr;
    const absl::flat_hash_map<std::string, u32>* problem_preds_by_name = nullptr;
    const absl::flat_hash_map<u32, std::vector<u32>>* problem_perm = nullptr;

    /// Canonical argument order of a generated derived predicate (DomainState::pred_perm), or nullptr.
    const std::vector<u32>* perm(u32 pred) const
    {
        if (auto it = ds->pred_perm.find(pred); it != ds->pred_perm.end())
            return &it->second;
        if (problem_perm)
            if (auto it = problem_perm->find(pred); it != problem_perm->end())
                return &it->second;
        return nullptr;
    }

    u32 pred(loki::Predicate p) const
    {
        if (auto it = ds->pred_of.find(p); it != ds->pred_of.end())
            return it->second;
        if (problem_preds)
            if (auto it = problem_preds->find(p); it != problem_preds->end())
                return it->second;
        if (auto it = ds->pred_by_name.find(p->get_name()); it != ds->pred_by_name.end())
            return it->second;
        if (problem_preds_by_name)
            if (auto it = problem_preds_by_name->find(p->get_name()); it != problem_preds_by_name->end())
                return it->second;
        fail("unknown predicate " + p->get_name());
    }
    u32 func(loki::FunctionSkeleton f) const
    {
        if (auto it = ds->func_of.find(f); it != ds->func_of.end())
            return it->second;
        if (auto it = ds->func_by_name.find(f->get_name()); it != ds->func_by_name.end())
            return it->second;
        fail("unknown function " + f->get_name());
    }
    u32 object(loki::Object o) const
    {
        if (objects)
        {
            if (auto it = objects->find(o); it != objects->end())
                return it->second;
        }
        if (auto it = ds->const_of.find(o); it != ds->const_of.end())
            return it->second;
        if (auto it = ds->const_by_name.find(o->get_name()); it != ds->const_by_name.end())
            return it->second;
        if (objects)
            for (const auto& [lo, id] : *objects)
                if (lo->get_name() == o->get_name())
                    return id;
        fail("unknown object " + o->get_name());
    }
    u32 type(loki::Type t) const
    {
        if (auto it = ds->type_of.find(t); it != ds->type_of.end())
            return it->second;
        if (auto it = ds->type_by_name.find(t->get_name()); it != ds->type_by_name.end())
            return it->second;
        fail("unknown type " + t->get_name());
    }
};

class Pass1 : public Core
{
public:
    Pass1(Repo& r, const Kinds& k, const Lookups& lk) : Core(r, k), lk(lk) {}

    u32 name(const std::string& s) { return r.names.id(s); }
    u32 var(loki::Variable v) { return r.var.id(Key{name(v->get_name()), 0}); }
    u64 term(loki::Term t)
    {
        const auto& v = t->get_object_or_variable();
        if (const auto* o = std::get_if<loki::Object>(&v))
            return obj_term(lk.object(*o));
        return var(std::get<loki::Variable>(v));
    }
    std::vector<u64> terms(const loki::TermList& ts)
    {
        std::vector<u64> out;
        out.reserve(ts.size());
        for (const auto& t : ts)
            out.push_back(term(t));
        return out;
    }
    u32 param(loki::Parameter p)
    {
        const u32 v = var(p->get_variable());
        Key key{v};  // mimir: variable first, then the types
        std::vector<u32> types;
        for (const auto& t : p->get_bases())
            types.push_back(lk.type(t));
        // the declared types of the variable in the scope being translated (canonical type order of conditions)
        if (v >= m_var_types.size())
            m_var_types.resize(v + 1);
        m_var_types[v].assign(types.begin(), types.end());
        std::sort(types.begin(), types.end());
        key.insert(key.end(), types.begin(), types.end());
        return r.param.id(key);
    }

    /// Canonical type order (domain.hpp): loki emits the type literals of one parameter (AddTypePredicates) in the
    /// iteration order of an std::unordered_set of type pointers. They are consecutive parts of the condition; each
    /// such run is put in hierarchy order before any literal is created. In the axioms of generated predicates the
    /// parameters themselves come in an address-dependent order, so there a run spans the type literals of all
    /// parameters and is ordered by (parameter, hierarchy rank).
    void canonicalize_type_runs(loki::ConditionList& parts) const
    {
        const size_t n = parts.size();
        std::vector<const loki::VariableImpl*> var(n, nullptr);  // positive type literal over a variable, else null
        std::vector<u32> type(n, ~0u);
        for (size_t i = 0; i < n; ++i)
        {
            const auto* cl = std::get_if<loki::ConditionLiteral>(&parts[i]->get_condition());
            if (!cl || !(*cl)->get_literal()->get_polarity())
                continue;
            const auto& atom = (*cl)->get_literal()->get_atom();
            const auto pit = lk.ds->pred_of.find(atom->get_predicate());
            if (atom->get_terms().size() != 1 || pit == lk.ds->pred_of.end() || pit->second >= lk.ds->pred_type.size()
                || lk.ds->pred_type[pit->second] == ~0u)
                continue;
            if (const auto* v = std::get_if<loki::Variable>(&atom->get_terms()[0]->get_object_or_variable()))
            {
                var[i] = *v;
                type[i] = lk.ds->pred_type[pit->second];
            }
        }
        // per variable: its parameter position in a generated axiom (else 0) and the hierarchy ranks of its types
        struct Scope
        {
            bool in_scope = false;
            u32 position = 0;
            std::vector<u32> rank;
        };
        absl::flat_hash_map<const loki::VariableImpl*, Scope> scopes;
        auto scope = [&](const loki::VariableImpl* v) -> const Scope&
        {
            auto [it, fresh] = scopes.try_emplace(v);
            if (fresh)
            {
                std::vector<TypeId> declared;
                if (const auto name = r.names.find(v->get_name()))
                    if (const auto id = r.var.find(Key{*name, 0}))
                    {
                        if (*id < m_var_types.size())
                            for (u32 t : m_var_types[*id])
                                declared.push_back(TypeId{t});
                        if (auto p = m_scope_pos.find(*id); p != m_scope_pos.end())
                            it->second = Scope{true, p->second, {}};
                    }
                it->second.rank = hierarchy_ranks(lk.ds->data, declared);
            }
            return it->second;
        };
        const bool across = !m_scope_pos.empty();
        auto starts_run = [&](size_t i) { return var[i] && (!across || scope(var[i]).in_scope); };
        auto joins = [&](size_t i, size_t j) { return var[j] && (across ? scope(var[j]).in_scope : var[j] == var[i]); };
        for (size_t i = 0; i < n;)
        {
            size_t j = i + 1;
            if (starts_run(i))
                while (j < n && joins(i, j))
                    ++j;
            if (j - i > 1)
            {
                auto key = [&](size_t k)
                {
                    const Scope& sc = scope(var[k]);
                    return std::pair(across ? sc.position : 0u, type[k] < sc.rank.size() ? sc.rank[type[k]] : ~0u);
                };
                std::vector<size_t> idx(j - i);
                std::iota(idx.begin(), idx.end(), i);
                std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return key(x) < key(y); });
                loki::ConditionList sorted;
                for (size_t k : idx)
                    sorted.push_back(parts[k]);
                std::copy(sorted.begin(), sorted.end(), parts.begin() + static_cast<std::ptrdiff_t>(i));
            }
            i = j;
        }
    }
    std::vector<u32> params(const loki::ParameterList& ps)
    {
        std::vector<u32> out;
        for (const auto& p : ps)
            out.push_back(param(p));
        return out;
    }
    u32 lit(loki::Literal l)
    {
        const u32 p = lk.pred(l->get_atom()->get_predicate());
        Key key{p, l->get_polarity() ? 1u : 0u};
        const auto& ts = l->get_atom()->get_terms();
        if (const auto* pm = lk.perm(p); pm && pm->size() == ts.size())
            for (u32 j : *pm)
                key.push_back(term(ts[j]));
        else
            for (const auto& t : ts)
                key.push_back(term(t));
        return r.lit.id(key);
    }

    u32 fexpr(loki::FunctionExpression e)
    {
        return std::visit(
            [&](auto&& arg) -> u32
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, loki::FunctionExpressionNumber>)
                    return fnum(arg->get_number());
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionBinaryOperator>)
                {
                    const u32 a = fexpr(arg->get_left_function_expression());
                    const u32 b = fexpr(arg->get_right_function_expression());
                    return fbin(static_cast<u64>(arg->get_binary_operator()), a, b);
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMultiOperator>)
                {
                    std::vector<u32> xs;
                    for (const auto& x : arg->get_function_expressions())
                        xs.push_back(fexpr(x));
                    return fmulti(static_cast<u64>(arg->get_multi_operator()), std::move(xs));
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMinus>)
                    return fminus(fexpr(arg->get_function_expression()));
                else
                {
                    const u32 f = lk.func(arg->get_function()->get_function_skeleton());
                    if (k.func.at(f) == kAuxiliary)
                        throw std::logic_error("ToMimirStructures::translate_lifted: FunctionExpressionFunction<AuxiliaryTag> must not exist.");
                    const auto ts = terms(arg->get_function()->get_terms());
                    return ffunc(f, ts);
                }
            },
            e->get_function_expression());
    }

    /// Distinct terms of an expression (mimir's NumericConstraint::get_terms, only its size matters).
    void collect_terms(u32 f, absl::flat_hash_set<u64>& out) const
    {
        const Key& key = r.fe[f];
        switch (key[0])
        {
            case kFeBinary:
                collect_terms(key_u32(key[2]), out);
                collect_terms(key_u32(key[3]), out);
                return;
            case kFeMulti:
                for (size_t i = 2; i < key.size(); ++i)
                    collect_terms(key_u32(key[i]), out);
                return;
            case kFeMinus: collect_terms(key_u32(key[1]), out); return;
            case kFeFunction:
                for (size_t i = 2; i < key.size(); ++i)
                    out.insert(key[i]);
                return;
            default: return;
        }
    }

    u32 nc(loki::ConditionNumericConstraint c)
    {
        const u32 a = fexpr(c->get_left_function_expression());
        const u32 b = fexpr(c->get_right_function_expression());
        absl::flat_hash_set<u64> ts;
        collect_terms(a, ts);
        collect_terms(b, ts);
        return r.nc.id(Key{static_cast<u64>(c->get_binary_comparator()), a, b, ts.size()});
    }

    /// mimir: ToMimirStructures::translate_lifted(Condition, parameters)
    u32 cond(loki::Condition c, const std::vector<u32>& ps)
    {
        std::vector<u32> lits[3], ncs;
        auto add_lit = [&](loki::Literal l)
        {
            const u32 id = lit(l);
            lits[lit_kind(id)].push_back(id);
        };
        if (const auto* a = std::get_if<loki::ConditionAnd>(&c->get_condition()))
        {
            loki::ConditionList parts = (*a)->get_conditions();
            canonicalize_type_runs(parts);
            for (const auto& part : parts)
            {
                if (const auto* l = std::get_if<loki::ConditionLiteral>(&part->get_condition()))
                    add_lit((*l)->get_literal());
                else if (const auto* n = std::get_if<loki::ConditionNumericConstraint>(&part->get_condition()))
                    ncs.push_back(nc(*n));
                else
                    throw std::logic_error("Expected literal in conjunctive condition.");
            }
        }
        else if (const auto* l = std::get_if<loki::ConditionLiteral>(&c->get_condition()))
            add_lit((*l)->get_literal());
        else if (const auto* n = std::get_if<loki::ConditionNumericConstraint>(&c->get_condition()))
            ncs.push_back(nc(*n));
        else
            throw std::logic_error("Expected conjunctive condition.");
        return Core::cond(ps, lits, std::move(ncs));
    }
    u32 empty_cond(const std::vector<u32>& ps = {})
    {
        std::vector<u32> lits[3];
        return Core::cond(ps, lits, {});
    }

    /// mimir: ToMimirStructures::translate_lifted(Effect, parameters). Effects are grouped by condition in an
    /// std::unordered_map keyed by the condition pointer, and the conditional effects are created while iterating it.
    std::vector<u32> effects(loki::Effect effect, const std::vector<u32>& action_params)
    {
        struct Group
        {
            u32 cond;
            std::vector<u32> lits, nes;
            std::optional<u32> aux;
        };
        std::vector<Group> groups;
        absl::flat_hash_map<u32, size_t> group_of;
        auto one = [&](loki::Effect e)
        {
            std::vector<u32> forall_params;
            if (const auto* f = std::get_if<loki::EffectCompositeForall>(&e->get_effect()))
            {
                forall_params = params((*f)->get_parameters());
                e = (*f)->get_effect();
            }
            std::optional<u32> c;
            if (const auto* w = std::get_if<loki::EffectCompositeWhen>(&e->get_effect()))
            {
                c = cond((*w)->get_condition(), forall_params);
                e = (*w)->get_effect();
            }
            if (!c)
                c = empty_cond();  // mimir: forall parameters without a `when` are dropped here
            auto [it, inserted] = group_of.try_emplace(*c, groups.size());
            if (inserted)
                groups.push_back(Group{*c, {}, {}, std::nullopt});
            Group& g = groups[it->second];
            if (const auto* l = std::get_if<loki::EffectLiteral>(&e->get_effect()))
            {
                const u32 id = lit((*l)->get_literal());
                if (lit_kind(id) == kDerived)
                    throw std::runtime_error("Only fluent literals are allowed in effects!");
                if (lit_kind(id) == kStatic)
                    throw std::logic_error("Expected fluent effect literal but it was static!");
                g.lits.push_back(id);
            }
            else if (const auto* n = std::get_if<loki::EffectNumeric>(&e->get_effect()))
            {
                const u32 f = lk.func((*n)->get_function()->get_function_skeleton());
                const auto ts = terms((*n)->get_function()->get_terms());
                const u32 x = fexpr((*n)->get_function_expression());
                Key key{static_cast<u64>((*n)->get_assign_operator()), f, ts.size()};
                key.insert(key.end(), ts.begin(), ts.end());
                key.push_back(x);
                if (k.func.at(f) == kFluent)
                    g.nes.push_back(r.ne.id(key));
                else if (k.func.at(f) == kAuxiliary)
                    g.aux = r.nea.id(key);
                else
                    throw std::logic_error("ToMimirStructures::translate_lifted: Function<StaticTag> must not exist.");
            }
            else
                throw std::logic_error("Unexpected effect type. This error indicates a bug in the translation.");
        };
        if (const auto* a = std::get_if<loki::EffectAnd>(&effect->get_effect()))
        {
            loki::EffectList parts = (*a)->get_effects();
            canonicalize_numeric_runs(parts);
            for (const auto& e : parts)
                one(e);
        }
        else
            one(effect);

        // mimir's unordered_map (libstdc++; libc++ does the same) iterates at most two groups in reverse insertion
        // order: a node for an empty bucket goes to the list front, a node for an occupied bucket in front of that
        // bucket's nodes. With three or more groups a shared bucket changes the order (address-dependent), see
        // domain.hpp.
        std::vector<u32> out;
        for (auto it = groups.rbegin(); it != groups.rend(); ++it)
            out.push_back(ce(it->cond, ceff(action_params, it->lits, it->nes, it->aux)));
        return out;
    }

    /// Canonical order of numeric effects (domain.hpp): loki's `flatten` (ToEffectNormalForm) sums the numeric
    /// effects per (assign operator, function) in an std::unordered_map keyed by a Function pointer and creates the
    /// sums in its iteration order. They are consecutive parts of the effect; each run of numeric effects under the
    /// same forall/when is put in the order of (function, arguments, operator) before any id is created.
    static void canonicalize_numeric_runs(loki::EffectList& parts)
    {
        struct Info
        {
            std::vector<const void*> scope;  // forall parameters, when condition
            std::string key;
        };
        auto info = [](loki::Effect e) -> std::optional<Info>
        {
            Info out;
            if (const auto* f = std::get_if<loki::EffectCompositeForall>(&e->get_effect()))
            {
                for (const auto& p : (*f)->get_parameters())
                    out.scope.push_back(p);
                e = (*f)->get_effect();
            }
            out.scope.push_back(nullptr);
            if (const auto* w = std::get_if<loki::EffectCompositeWhen>(&e->get_effect()))
            {
                out.scope.push_back((*w)->get_condition());
                e = (*w)->get_effect();
            }
            const auto* n = std::get_if<loki::EffectNumeric>(&e->get_effect());
            if (!n)
                return std::nullopt;
            out.key = (*n)->get_function()->get_function_skeleton()->get_name();
            for (const auto& t : (*n)->get_function()->get_terms())
                out.key += " " + std::visit([](auto&& x) { return x->get_name(); }, t->get_object_or_variable());
            out.key += " " + std::to_string(static_cast<int>((*n)->get_assign_operator()));
            return out;
        };
        std::vector<std::optional<Info>> infos;
        for (const auto& e : parts)
            infos.push_back(info(e));
        for (size_t i = 0; i < parts.size();)
        {
            size_t j = i + 1;
            if (infos[i])
                while (j < parts.size() && infos[j] && infos[j]->scope == infos[i]->scope)
                    ++j;
            if (j - i > 1)
            {
                std::vector<size_t> idx(j - i);
                std::iota(idx.begin(), idx.end(), i);
                std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return infos[a]->key < infos[b]->key; });
                loki::EffectList sorted;
                for (size_t k : idx)
                    sorted.push_back(parts[k]);
                std::copy(sorted.begin(), sorted.end(), parts.begin() + static_cast<std::ptrdiff_t>(i));
            }
            i = j;
        }
    }

    Action1 action(loki::Action a)
    {
        Action1 out;
        out.name = a->get_name();
        out.original_arity = static_cast<u32>(a->get_original_arity());
        out.params = params(a->get_parameters());
        // mimir: `std::sort(params.begin() + original_arity, params.end())` over Parameter pointers, i.e. allocation
        // order; creation order is our proxy.
        std::sort(out.params.begin() + std::min<size_t>(out.original_arity, out.params.size()), out.params.end());
        out.cond = a->get_condition() ? cond(*a->get_condition(), out.params) : empty_cond(out.params);
        if (a->get_effect())
            out.ces = effects(*a->get_effect(), out.params);
        std::sort(out.ces.begin(), out.ces.end());
        return out;
    }

    u32 axiom(loki::Axiom x)
    {
        // canonical argument order of a generated head predicate: the parameters bound to the head's arguments come
        // first, in canonical argument order, then the others (moved existentials) in loki's order
        loki::ParameterList lps = x->get_parameters();
        const auto& head_terms = x->get_literal()->get_atom()->get_terms();
        const auto* pm = lk.perm(lk.pred(x->get_literal()->get_atom()->get_predicate()));
        if (pm && pm->size() == head_terms.size())
        {
            loki::ParameterList reordered;
            for (u32 j : *pm)
                if (const auto* v = std::get_if<loki::Variable>(&head_terms[j]->get_object_or_variable()))
                    for (const auto& q : lps)
                        if (q->get_variable() == *v && std::find(reordered.begin(), reordered.end(), q) == reordered.end())
                            reordered.push_back(q);
            for (const auto& q : lps)
                if (std::find(reordered.begin(), reordered.end(), q) == reordered.end())
                    reordered.push_back(q);
            lps = std::move(reordered);
        }
        const auto ps = params(lps);
        if (pm)  // loki types the parameters in their (address-dependent) order: sort type literals across variables
            for (u32 i = 0; i < ps.size(); ++i)
                m_scope_pos[key_u32(r.param[ps[i]][0])] = i;
        const u32 c = cond(x->get_condition(), ps);
        m_scope_pos.clear();
        const u32 h = lit(x->get_literal());
        if (lit_kind(h) != kDerived)
            throw std::runtime_error("ToMimirStructures::translate_lifted: Expected Literal<DerivedTag> in axiom head.");
        return r.axiom.id(Key{c, h});
    }

    // ------------------------------------------------------------ ground (problem)
    /// [pred, polarity, objects...] of a ground literal, as Core::glit keys it
    Key glit_key(loki::Literal l) const
    {
        const u32 p = lk.pred(l->get_atom()->get_predicate());
        std::vector<u32> objs;
        for (const auto& t : l->get_atom()->get_terms())
        {
            const auto* o = std::get_if<loki::Object>(&t->get_object_or_variable());
            if (!o)
                throw std::logic_error("Expected ground term.");
            objs.push_back(lk.object(*o));
        }
        Key key{p, l->get_polarity() ? 1u : 0u};
        if (const auto* pm = lk.perm(p); pm && pm->size() == objs.size())
            for (u32 j : *pm)
                key.push_back(objs[j]);
        else
            key.insert(key.end(), objs.begin(), objs.end());
        return key;
    }
    u32 glit_of(loki::Literal l) { return r.glit.id(glit_key(l)); }

    /// mimir: ToMimirStructures::translate_grounded(FunctionExpression) -- no folding
    u32 gfexpr(loki::FunctionExpression e)
    {
        return std::visit(
            [&](auto&& arg) -> u32
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, loki::FunctionExpressionNumber>)
                    return gnum(arg->get_number());
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionBinaryOperator>)
                {
                    const u32 a = gfexpr(arg->get_left_function_expression());
                    const u32 b = gfexpr(arg->get_right_function_expression());
                    return gbin(static_cast<u64>(arg->get_binary_operator()), a, b);
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMultiOperator>)
                {
                    std::vector<u32> xs;
                    for (const auto& x : arg->get_function_expressions())
                        xs.push_back(gfexpr(x));
                    return gmulti(static_cast<u64>(arg->get_multi_operator()), std::move(xs));
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMinus>)
                    return gminus(gfexpr(arg->get_function_expression()));
                else
                {
                    const u32 f = lk.func(arg->get_function()->get_function_skeleton());
                    std::vector<u32> objs;
                    for (const auto& t : arg->get_function()->get_terms())
                    {
                        const auto* o = std::get_if<loki::Object>(&t->get_object_or_variable());
                        if (!o)
                            throw std::logic_error("Expected ground term.");
                        objs.push_back(lk.object(*o));
                    }
                    return gfunc(f, objs);
                }
            },
            e->get_function_expression());
    }
    u32 gnc_of(loki::ConditionNumericConstraint c)
    {
        const u32 a = gfexpr(c->get_left_function_expression());
        const u32 b = gfexpr(c->get_right_function_expression());
        return r.gnc.id(Key{static_cast<u64>(c->get_binary_comparator()), a, b});
    }

    const Lookups& lk;

private:
    std::vector<std::vector<u32>> m_var_types;    // R1 variable -> declared types in the current scope
    absl::flat_hash_map<u32, u32> m_scope_pos;   // R1 variable -> parameter position (generated axioms only)
};

// ================================================================================================ pass 2 (R1 -> R2)

/// mimir: EncodeParameterIndexInVariables on top of RecursiveBaseTranslator (no caching).
class Pass2 : public Core
{
public:
    Pass2(Repo& r2, const Repo& r1, const Kinds& k) : Core(r2, k), r1(r1) {}

    u32 var(u32 v1)
    {
        const Key& key = r1.var[v1];
        const std::string& name = r1.names[key_u32(key[0])];
        if (m_encode)
            if (auto it = m_index.find(v1); it != m_index.end())
                return r.var.id(Key{r.names.id(name + "_" + std::to_string(it->second)), it->second});
        return r.var.id(Key{r.names.id(name), 0});
    }
    u64 term(u64 t) { return is_obj_term(t) ? t : var(key_u32(t)); }
    u32 param(u32 p1)
    {
        const Key& key1 = r1.param[p1];
        Key key{var(key_u32(key1[0]))};
        key.insert(key.end(), key1.begin() + 1, key1.end());
        return r.param.id(key);
    }
    std::vector<u32> params(const std::vector<u32>& ps)
    {
        std::vector<u32> out;
        for (u32 p : ps)
            out.push_back(param(p));
        return out;
    }
    /// mimir: predicates and function skeletons are translated with the encoding disabled
    void unencoded_params(const std::vector<u32>& ps)
    {
        m_encode = false;
        params(ps);
        m_encode = true;
    }
    u32 lit(u32 l1)
    {
        const Key& key1 = r1.lit[l1];
        Key key{key1[0], key1[1]};
        for (size_t i = 2; i < key1.size(); ++i)
            key.push_back(term(key1[i]));
        return r.lit.id(key);
    }
    u32 fexpr(u32 f1)
    {
        const Key& key1 = r1.fe[f1];
        switch (key1[0])
        {
            case kFeNumber: return fnum(bits_f64(key1[1]));
            case kFeBinary:
            {
                const u32 a = fexpr(key_u32(key1[2]));
                const u32 b = fexpr(key_u32(key1[3]));
                return fbin(key1[1], a, b);
            }
            case kFeMulti:
            {
                std::vector<u32> xs;
                for (size_t i = 2; i < key1.size(); ++i)
                    xs.push_back(fexpr(key_u32(key1[i])));
                return fmulti(key1[1], std::move(xs));
            }
            case kFeMinus: return fminus(fexpr(key_u32(key1[1])));
            case kFeFunction:
            {
                std::vector<u64> ts;
                for (size_t i = 2; i < key1.size(); ++i)
                    ts.push_back(term(key1[i]));
                return ffunc(key_u32(key1[1]), ts);
            }
        }
        fail("bad function expression");
    }
    u32 nc(u32 n1)
    {
        const Key& key1 = r1.nc[n1];
        const u32 a = fexpr(key_u32(key1[1]));
        const u32 b = fexpr(key_u32(key1[2]));
        return r.nc.id(Key{key1[0], a, b, key1[3]});
    }
    u32 cond(u32 c1)
    {
        const CondView v = decode_cond(r1.cond[c1]);
        const auto ps = params(v.params);
        std::vector<u32> lits[3];
        for (int kind = 0; kind < 3; ++kind)
            for (u32 l : v.lits[kind])
                lits[kind].push_back(lit(l));
        std::vector<u32> ncs;
        for (u32 n : v.ncs)
            ncs.push_back(nc(n));
        return Core::cond(ps, lits, std::move(ncs));
    }
    u32 numeric_effect(const Table<Key>& from, Table<Key>& to, u32 e1)
    {
        const Key& key1 = from[e1];
        const size_t nt = key1[2];
        Key key{key1[0], key1[1], nt};
        for (size_t i = 0; i < nt; ++i)
            key.push_back(term(key1[3 + i]));
        key.push_back(fexpr(key_u32(key1[3 + nt])));
        return to.id(key);
    }
    u32 ceff(u32 e1)
    {
        const CeffView v = decode_ceff(r1.ceff[e1]);
        const auto ps = params(v.params);
        std::vector<u32> lits, nes;
        for (u32 l : v.lits)
            lits.push_back(lit(l));
        for (u32 n : v.nes)
            nes.push_back(numeric_effect(r1.ne, r.ne, n));
        std::optional<u32> aux;
        if (v.aux)
            aux = numeric_effect(r1.nea, r.nea, *v.aux);
        return Core::ceff(ps, std::move(lits), std::move(nes), aux);
    }
    /// mimir: EncodeParameterIndexInVariables::translate_level_2(ConditionalEffect)
    u32 ce(u32 ce1)
    {
        const Key& key1 = r1.ce[ce1];
        const CondView cv = decode_cond(r1.cond[key_u32(key1[0])]);
        const u32 start = static_cast<u32>(m_index.size());
        for (size_t i = 0; i < cv.params.size(); ++i)
            m_index[key_u32(r1.param[cv.params[i]][0])] = start + static_cast<u32>(i);
        const u32 c = cond(key_u32(key1[0]));
        const u32 e = ceff(key_u32(key1[1]));
        const u32 id = Core::ce(c, e);
        for (u32 p : cv.params)
            m_index.erase(key_u32(r1.param[p][0]));
        return id;
    }
    struct Action2
    {
        std::string name;
        u32 original_arity;
        std::vector<u32> params1;
        u32 cond;
        std::vector<u32> ces;
    };
    Action2 action(const Action1& a)
    {
        m_index.clear();
        for (size_t i = 0; i < a.params.size(); ++i)
            m_index[key_u32(r1.param[a.params[i]][0])] = static_cast<u32>(i);
        Action2 out{a.name, a.original_arity, a.params, cond(a.cond), {}};
        for (u32 c : a.ces)
            out.ces.push_back(ce(c));
        std::sort(out.ces.begin(), out.ces.end());
        m_index.clear();
        return out;
    }
    u32 axiom(u32 x1)
    {
        const Key& key1 = r1.axiom[x1];
        const CondView cv = decode_cond(r1.cond[key_u32(key1[0])]);
        m_index.clear();
        for (size_t i = 0; i < cv.params.size(); ++i)
            m_index[key_u32(r1.param[cv.params[i]][0])] = static_cast<u32>(i);
        const u32 c = cond(key_u32(key1[0]));
        const u32 h = lit(key_u32(key1[1]));
        m_index.clear();
        return r.axiom.id(Key{c, h});
    }

    // ------------------------------------------------------------ ground (default translation)
    u32 glit(u32 g1) { return r.glit.id(r1.glit[g1]); }
    u32 gfexpr(u32 g1)
    {
        const Key& key1 = r1.gfe[g1];
        switch (key1[0])
        {
            case kFeNumber: return gnum(bits_f64(r1.gnum[key_u32(key1[1])][0]));
            case kFeBinary:
            {
                const Key& in = r1.gbin[key_u32(key1[1])];
                const u32 a = gfexpr(key_u32(in[1]));
                const u32 b = gfexpr(key_u32(in[2]));
                return gbin(in[0], a, b);
            }
            case kFeMulti:
            {
                const Key& in = r1.gmulti[key_u32(key1[1])];
                std::vector<u32> xs;
                for (size_t i = 1; i < in.size(); ++i)
                    xs.push_back(gfexpr(key_u32(in[i])));
                return gmulti(in[0], std::move(xs));
            }
            case kFeMinus: return gminus(gfexpr(key_u32(r1.gminus[key_u32(key1[1])][0])));
            case kFeFunction:
            {
                const Key& in = r1.gfunc[key1[1]][key_u32(key1[2])];
                std::vector<u32> objs;
                for (size_t i = 1; i < in.size(); ++i)
                    objs.push_back(key_u32(in[i]));
                return gfunc(key_u32(in[0]), objs);
            }
        }
        fail("bad ground function expression");
    }
    u32 gnc(u32 n1)
    {
        const Key& key1 = r1.gnc[n1];
        const u32 a = gfexpr(key_u32(key1[1]));
        const u32 b = gfexpr(key_u32(key1[2]));
        return r.gnc.id(Key{key1[0], a, b});
    }

    const Repo& r1;

private:
    absl::flat_hash_map<u32, u32> m_index;  // pass-1 variable -> parameter index
    bool m_encode = true;
};

// ================================================================================================ emission (R2 -> TaskData)

class Emitter
{
public:
    Emitter(const Repo& r, const Kinds& k, TaskData& t) : r(r), k(k), t(t) {}

    /// Display name of a (pass-2, parameter-encoded) variable: without '?' and the "_<index>" suffix.
    std::string var_name(u32 v) const
    {
        const Key& key = r.var[v];
        std::string s = r.names[key_u32(key[0])];
        const std::string suffix = "_" + std::to_string(key[1]);
        if (s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0)
            s.resize(s.size() - suffix.size());
        return strip_question(s);
    }
    Range params(const std::vector<u32>& ps)
    {
        std::vector<Parameter> out;
        for (u32 p : ps)
        {
            const Key& key = r.param[p];
            Parameter x;
            x.name = t.intern_string(var_name(key_u32(key[0])));
            std::vector<TypeId> types;
            for (size_t i = 1; i < key.size(); ++i)
                types.push_back(TypeId{key_u32(key[i])});
            x.types = TaskData::append<TypeId>(t.type_ids, types);
            out.push_back(x);
        }
        return TaskData::append<Parameter>(t.params, out);
    }
    Term term(u64 x) const
    {
        if (is_obj_term(x))
            return object_term(ObjectId{term_obj(x)});
        return static_cast<Term>(r.var[key_u32(x)][1]);
    }
    Literal lit(u32 l)
    {
        const Key& key = r.lit[l];
        Literal out;
        out.pred = PredicateId{key_u32(key[0])};
        out.positive = key[1] != 0;
        std::vector<Term> ts;
        for (size_t i = 2; i < key.size(); ++i)
            ts.push_back(term(key[i]));
        out.terms = TaskData::append<Term>(t.terms, ts);
        return out;
    }
    Literal glit(u32 g)
    {
        const Key& key = r.glit[g];
        Literal out;
        out.pred = PredicateId{key_u32(key[0])};
        out.positive = key[1] != 0;
        std::vector<Term> ts;
        for (size_t i = 2; i < key.size(); ++i)
            ts.push_back(object_term(ObjectId{key_u32(key[i])}));
        out.terms = TaskData::append<Term>(t.terms, ts);
        return out;
    }
    GroundAtom atom(u32 pred, std::span<const u32> objects)
    {
        GroundAtom a;
        a.pred = PredicateId{pred};
        a.objects = Range{static_cast<u32>(t.object_ids.size()), static_cast<u32>(objects.size())};
        for (u32 o : objects)
            t.object_ids.push_back(ObjectId{o});
        return a;
    }

    u32 push(const Expr& e)
    {
        t.exprs.push_back(e);
        return static_cast<u32>(t.exprs.size() - 1);
    }
    static ExprOp binary_op(u64 op)
    {
        switch (static_cast<loki::BinaryOperatorEnum>(op))
        {
            case loki::BinaryOperatorEnum::PLUS: return ExprOp::Add;
            case loki::BinaryOperatorEnum::MINUS: return ExprOp::Sub;
            case loki::BinaryOperatorEnum::MUL: return ExprOp::Mul;
            case loki::BinaryOperatorEnum::DIV: return ExprOp::Div;
        }
        fail("bad binary operator");
    }
    static ExprOp multi_op(u64 op)
    {
        return static_cast<loki::MultiOperatorEnum>(op) == loki::MultiOperatorEnum::PLUS ? ExprOp::Add : ExprOp::Mul;
    }
    /// A multi-operator becomes a left-associated chain, as mimir's exporter writes it.
    u32 chain(ExprOp op, const std::vector<u32>& xs)
    {
        if (xs.empty())
            fail("empty multi operator");
        u32 acc = xs[0];
        for (size_t i = 1; i < xs.size(); ++i)
        {
            Expr e;
            e.op = op;
            e.a = acc;
            e.b = xs[i];
            acc = push(e);
        }
        return acc;
    }
    u32 expr(u32 f)
    {
        const Key& key = r.fe[f];
        Expr e;
        switch (key[0])
        {
            case kFeNumber:
                e.op = ExprOp::Number;
                e.value = bits_f64(key[1]);
                return push(e);
            case kFeBinary:
                e.op = binary_op(key[1]);
                e.a = expr(key_u32(key[2]));
                e.b = expr(key_u32(key[3]));
                return push(e);
            case kFeMulti:
            {
                std::vector<u32> xs;
                for (size_t i = 2; i < key.size(); ++i)
                    xs.push_back(expr(key_u32(key[i])));
                return chain(multi_op(key[1]), xs);
            }
            case kFeMinus:
                e.op = ExprOp::Neg;
                e.a = expr(key_u32(key[1]));
                return push(e);
            case kFeFunction:
            {
                e.op = ExprOp::Function;
                e.func = FunctionId{key_u32(key[1])};
                std::vector<Term> ts;
                for (size_t i = 2; i < key.size(); ++i)
                    ts.push_back(term(key[i]));
                e.terms = TaskData::append<Term>(t.terms, ts);
                return push(e);
            }
        }
        fail("bad function expression");
    }
    u32 gexpr(u32 g)
    {
        const Key& key = r.gfe[g];
        Expr e;
        switch (key[0])
        {
            case kFeNumber:
                e.op = ExprOp::Number;
                e.value = bits_f64(r.gnum[key_u32(key[1])][0]);
                return push(e);
            case kFeBinary:
            {
                const Key& in = r.gbin[key_u32(key[1])];
                e.op = binary_op(in[0]);
                e.a = gexpr(key_u32(in[1]));
                e.b = gexpr(key_u32(in[2]));
                return push(e);
            }
            case kFeMulti:
            {
                const Key& in = r.gmulti[key_u32(key[1])];
                std::vector<u32> xs;
                for (size_t i = 1; i < in.size(); ++i)
                    xs.push_back(gexpr(key_u32(in[i])));
                return chain(multi_op(in[0]), xs);
            }
            case kFeMinus:
                e.op = ExprOp::Neg;
                e.a = gexpr(key_u32(r.gminus[key_u32(key[1])][0]));
                return push(e);
            case kFeFunction:
            {
                const Key& in = r.gfunc[key[1]][key_u32(key[2])];
                e.op = ExprOp::Function;
                e.func = FunctionId{key_u32(in[0])};
                std::vector<Term> ts;
                if (k.func.at(e.func.v) != kAuxiliary)
                    for (size_t i = 1; i < in.size(); ++i)
                        ts.push_back(object_term(ObjectId{key_u32(in[i])}));
                e.terms = TaskData::append<Term>(t.terms, ts);
                return push(e);
            }
        }
        fail("bad ground function expression");
    }
    static Comparator comparator(u64 c)
    {
        switch (static_cast<loki::BinaryComparatorEnum>(c))
        {
            case loki::BinaryComparatorEnum::GREATER: return Comparator::Gt;
            case loki::BinaryComparatorEnum::LESS: return Comparator::Lt;
            case loki::BinaryComparatorEnum::EQUAL: return Comparator::Eq;
            case loki::BinaryComparatorEnum::UNEQUAL: return Comparator::Ne;
            case loki::BinaryComparatorEnum::GREATER_EQUAL: return Comparator::Ge;
            case loki::BinaryComparatorEnum::LESS_EQUAL: return Comparator::Le;
        }
        fail("bad comparator");
    }
    static AssignOp assign_op(u64 op)
    {
        switch (static_cast<loki::AssignOperatorEnum>(op))
        {
            case loki::AssignOperatorEnum::ASSIGN: return AssignOp::Assign;
            case loki::AssignOperatorEnum::SCALE_UP: return AssignOp::ScaleUp;
            case loki::AssignOperatorEnum::SCALE_DOWN: return AssignOp::ScaleDown;
            case loki::AssignOperatorEnum::INCREASE: return AssignOp::Increase;
            case loki::AssignOperatorEnum::DECREASE: return AssignOp::Decrease;
        }
        fail("bad assign operator");
    }
    NumericConstraint nc(u32 n)
    {
        const Key& key = r.nc[n];
        NumericConstraint c;
        c.cmp = comparator(key[0]);
        c.lhs = expr(key_u32(key[1]));
        c.rhs = expr(key_u32(key[2]));
        return c;
    }
    NumericConstraint gnc(u32 n)
    {
        const Key& key = r.gnc[n];
        NumericConstraint c;
        c.cmp = comparator(key[0]);
        c.lhs = gexpr(key_u32(key[1]));
        c.rhs = gexpr(key_u32(key[2]));
        return c;
    }
    NumericEffect numeric_effect(const Key& key)
    {
        NumericEffect e;
        e.op = assign_op(key[0]);
        e.func = FunctionId{key_u32(key[1])};
        const size_t nt = key[2];
        std::vector<Term> ts;
        for (size_t i = 0; i < nt; ++i)
            ts.push_back(term(key[3 + i]));
        e.terms = TaskData::append<Term>(t.terms, ts);
        e.expr = expr(key_u32(key[3 + nt]));
        return e;
    }

    /// Lifted literals by kind, then the nullary ground literals by kind (mimir's exporter order), then the numeric
    /// constraints and the grounded nullary ones.
    Condition cond(u32 c)
    {
        const CondView v = decode_cond(r.cond[c]);
        const Repo::CondExtra& x = r.extra(c);
        std::vector<Literal> ls;
        for (const auto& kind : v.lits)
            for (u32 l : kind)
                ls.push_back(lit(l));
        for (const auto& kind : x.nullary)
            for (u32 g : kind)
            {
                Literal l = glit(g);
                ls.push_back(l);
            }
        std::vector<NumericConstraint> ncs;
        for (u32 n : v.ncs)
            ncs.push_back(nc(n));
        for (u32 n : x.gncs)
            ncs.push_back(gnc(n));
        Condition out;
        out.literals = TaskData::append<Literal>(t.literals, ls);
        out.constraints = TaskData::append<NumericConstraint>(t.constraints, ncs);
        return out;
    }

    Schema schema(const Pass2::Action2& a)
    {
        Schema s;
        s.name = t.intern_string(a.name);
        s.original_arity = a.original_arity;
        const CondView pv = decode_cond(r.cond[a.cond]);
        s.params = params(pv.params);
        s.precondition = cond(a.cond);
        std::vector<ConditionalEffect> ces;
        for (u32 id : a.ces)
        {
            const Key& key = r.ce[id];
            const CondView cv = decode_cond(r.cond[key_u32(key[0])]);
            const CeffView ev = decode_ceff(r.ceff[key_u32(key[1])]);
            ConditionalEffect ce;
            ce.extra_params = params(cv.params);
            ce.condition = cond(key_u32(key[0]));
            std::vector<Literal> es;
            for (u32 l : ev.lits)
                es.push_back(lit(l));
            ce.effects = TaskData::append<Literal>(t.literals, es);
            std::vector<NumericEffect> nes;
            for (u32 n : ev.nes)
                nes.push_back(numeric_effect(r.ne[n]));
            ce.numeric_effects = TaskData::append<NumericEffect>(t.numeric_effects, nes);
            if (ev.aux)
                ce.auxiliary = numeric_effect(r.nea[*ev.aux]);
            ces.push_back(ce);
        }
        s.effects = TaskData::append<ConditionalEffect>(t.conditional_effects, ces);
        return s;
    }

    Axiom axiom(u32 x, bool from_problem)
    {
        const Key& key = r.axiom[x];
        const CondView cv = decode_cond(r.cond[key_u32(key[0])]);
        Axiom a;
        a.params = params(cv.params);
        a.head = lit(key_u32(key[1]));
        a.body = cond(key_u32(key[0]));
        a.from_problem = from_problem;
        return a;
    }

    const Repo& r;
    const Kinds& k;
    TaskData& t;
};

/// The canonical order of type names: the built-in types `object` and `number` first, the others by name.
int type_name_class(std::string_view name) { return name == "object" ? 0 : name == "number" ? 1 : 2; }
bool type_name_less(std::string_view a, std::string_view b)
{
    const int ca = type_name_class(a), cb = type_name_class(b);
    return ca != cb ? ca < cb : a < b;
}

void add_types(DomainState& ds, TaskData& t, loki::Type ty)
{
    if (ds.type_of.contains(ty))
        return;
    if (auto it = ds.type_by_name.find(ty->get_name()); it != ds.type_by_name.end())
    {
        ds.type_of.emplace(ty, it->second);
        return;
    }
    loki::TypeList sorted_bases = ty->get_bases();  // loki's order of the bases of one type is a hash-table order
    std::sort(sorted_bases.begin(), sorted_bases.end(), [](const loki::Type& a, const loki::Type& b) { return type_name_less(a->get_name(), b->get_name()); });
    for (const auto& b : sorted_bases)
        add_types(ds, t, b);
    const u32 id = static_cast<u32>(t.types.size());
    Type out;
    out.name = t.intern_string(ty->get_name());
    std::vector<TypeId> bases;
    for (const auto& b : sorted_bases)
        bases.push_back(TypeId{ds.type_of.at(b)});
    out.bases = TaskData::append<TypeId>(t.type_ids, bases);
    t.types.push_back(out);
    ds.type_of.emplace(ty, id);
    ds.type_by_name.emplace(ty->get_name(), id);
}

Range loki_params(DomainState& ds, TaskData& t, const loki::ParameterList& ps)
{
    std::vector<Parameter> out;
    for (const auto& p : ps)
    {
        Parameter x;
        x.name = t.intern_string(strip_question(p->get_variable()->get_name()));
        std::vector<TypeId> types;
        for (const auto& b : p->get_bases())
        {
            add_types(ds, t, b);
            types.push_back(TypeId{ds.type_of.at(b)});
        }
        x.types = TaskData::append<TypeId>(t.type_ids, types);
        out.push_back(x);
    }
    return TaskData::append<Parameter>(t.params, out);
}

std::vector<std::string> requirement_strings(loki::Requirements r)
{
    std::vector<std::string> out;
    if (r)
        for (auto e : r->get_requirements())
            out.push_back(":" + loki::requirement_enum_to_string.at(e));
    return out;
}

Kinds domain_kinds(const DomainState& ds)
{
    Kinds k;
    for (const auto& p : ds.data.predicates)
    {
        k.pred.push_back(p.kind == PredKind::Static ? kStatic : p.kind == PredKind::Fluent ? kFluent : kDerived);
        k.pred_arity.push_back(p.arity);
    }
    for (const auto& f : ds.data.functions)
        k.func.push_back(f.kind == FuncKind::Static ? kStatic : f.kind == FuncKind::Fluent ? kFluent : kAuxiliary);
    return k;
}
}  // namespace

// ================================================================================================ canonical type order

std::vector<u32> hierarchy_ranks(const formalism::TaskData& t, std::span<const TypeId> declared)
{
    // visit the declared types in the canonical name order, each followed by its bases (in the same order)
    auto by_name = [&](std::vector<u32>& ids)
    { std::sort(ids.begin(), ids.end(), [&](u32 a, u32 b) { return type_name_less(t.str(t.types[a].name), t.str(t.types[b].name)); }); };
    std::vector<u32> order;
    std::vector<bool> seen(t.types.size(), false);
    auto visit = [&](auto&& self, u32 ty) -> void
    {
        if (ty >= t.types.size() || seen[ty])
            return;
        seen[ty] = true;
        order.push_back(ty);
        std::vector<u32> bases;
        for (TypeId b : TaskData::slice(t.type_ids, t.types[ty].bases))
            bases.push_back(b.v);
        by_name(bases);
        for (u32 b : bases)
            self(self, b);
    };
    std::vector<u32> roots;
    for (TypeId ty : declared)
        if (ty.v < t.types.size())
            roots.push_back(ty.v);
    by_name(roots);
    for (u32 ty : roots)
        visit(visit, ty);
    std::vector<u32> rank(t.types.size(), ~0u);
    for (size_t i = 0; i < order.size(); ++i)
        rank[order[order.size() - 1 - i]] = static_cast<u32>(i);  // reverse insertion order
    return rank;
}

// ================================================================================================ domain

std::vector<u32> generated_predicate_perm(const loki::Predicate& p, const PredicateOrigin& origin)
{
    const std::string& name = p->get_name();
    const auto& ps = p->get_parameters();
    if (ps.size() < 2 || origin.declared.contains(name) || name.size() <= 6 || name.compare(0, 6, "axiom_") != 0
        || name.find_first_not_of("0123456789", 6) != std::string::npos)
        return {};
    std::vector<u32> perm(ps.size());
    std::iota(perm.begin(), perm.end(), 0u);
    if (origin.argument_order)
        if (auto it = origin.argument_order->find(name); it != origin.argument_order->end())
        {
            const auto& order = it->second;
            if (order.size() != ps.size())
                fail("argument order for " + name + ": expected " + std::to_string(ps.size()) + " variables");
            for (size_t j = 0; j < order.size(); ++j)
            {
                const std::string v = order[j].starts_with('?') ? order[j] : "?" + order[j];
                u32 k = 0;
                while (k < ps.size() && ps[k]->get_variable()->get_name() != v)
                    ++k;
                if (k == ps.size() || std::find(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(j), k) != perm.begin() + static_cast<std::ptrdiff_t>(j))
                    fail("argument order for " + name + ": '" + order[j] + "' is not a (distinct) parameter");
                perm[j] = k;
            }
            return perm;
        }
    std::stable_sort(perm.begin(), perm.end(),
                     [&](u32 a, u32 b) { return ps[a]->get_variable()->get_name() < ps[b]->get_variable()->get_name(); });
    return perm;
}

std::unique_ptr<DomainState> translate_domain(const loki::Domain& dl, const PredicateOrigin& origin)
{
    auto ds = std::make_unique<DomainState>();
    TaskData& t = ds->data;
    t.domain_name = dl->get_name();
    t.requirements = requirement_strings(dl->get_requirements());

    {
        // type ids in the canonical name order: loki lists the types in the iteration order of a hash table of
        // type names, which differs between standard libraries
        loki::TypeList types = dl->get_types();
        std::sort(types.begin(), types.end(), [](const loki::Type& a, const loki::Type& b) { return type_name_less(a->get_name(), b->get_name()); });
        for (const auto& ty : types)
            add_types(*ds, t, ty);
    }

    Classifier{ds->fluent_predicates, ds->derived_predicates, ds->effect_functions}.domain(dl);

    // predicates: ids by kind (static, fluent, derived), each in loki's order (mimir: DomainBuilder sorts each kind by
    // index, and both passes create them in the domain's order)
    auto pred_kind = [&](const std::string& name)
    {
        if (ds->fluent_predicates.contains(name) && !ds->derived_predicates.contains(name))
            return PredKind::Fluent;
        if (ds->derived_predicates.contains(name))
            return PredKind::Derived;
        return PredKind::Static;
    };
    // canonical type order (domain.hpp): AddTypePredicates creates the type predicates in the iteration order of an
    // std::unordered_set of type pointers, before every other predicate; that block is put in hierarchy order
    auto type_of_pred = [&](const loki::Predicate& p) -> const loki::TypeImpl*
    {
        const auto& ps = p->get_parameters();
        if (ps.size() != 1 || ps[0]->get_bases().size() != 1 || ps[0]->get_variable()->get_name() != "?arg")
            return nullptr;
        const loki::Type b = ps[0]->get_bases()[0];
        return b->get_name() == p->get_name() ? b : nullptr;
    };
    std::vector<loki::Predicate> preds(dl->get_predicates().begin(), dl->get_predicates().end());
    {
        std::vector<TypeId> declared;
        for (const auto& ty : dl->get_types())
            declared.push_back(TypeId{ds->type_of.at(ty)});
        const std::vector<u32> rank = hierarchy_ranks(t, declared);
        std::vector<size_t> pos;
        std::vector<loki::Predicate> block;
        for (size_t i = 0; i < preds.size(); ++i)
            if (const auto* ty = type_of_pred(preds[i]); ty && ds->type_of.contains(ty))
            {
                pos.push_back(i);
                block.push_back(preds[i]);
            }
        std::stable_sort(block.begin(), block.end(), [&](const loki::Predicate& a, const loki::Predicate& b)
                         { return rank[ds->type_of.at(type_of_pred(a))] < rank[ds->type_of.at(type_of_pred(b))]; });
        for (size_t k = 0; k < pos.size(); ++k)
            preds[pos[k]] = block[k];
    }
    std::vector<loki::Predicate> by_kind[3];
    for (const auto& p : preds)
        by_kind[static_cast<int>(pred_kind(p->get_name()))].push_back(p);
    auto canonical_params = [&](const loki::Predicate& p) -> loki::ParameterList
    {
        const std::vector<u32> perm = generated_predicate_perm(p, origin);
        if (perm.empty())
            return p->get_parameters();
        loki::ParameterList out;
        for (u32 j : perm)
            out.push_back(p->get_parameters()[j]);
        return out;
    };
    for (int kind = 0; kind < 3; ++kind)
        for (const auto& p : by_kind[kind])
        {
            Predicate out;
            out.name = t.intern_string(p->get_name());
            out.kind = static_cast<PredKind>(kind);
            out.arity = static_cast<u32>(p->get_parameters().size());
            out.params = loki_params(*ds, t, canonical_params(p));
            if (auto perm = generated_predicate_perm(p, origin); !perm.empty())
                ds->pred_perm.emplace(static_cast<u32>(t.predicates.size()), std::move(perm));
            ds->pred_of.emplace(p, static_cast<u32>(t.predicates.size()));
            ds->pred_by_name.emplace(p->get_name(), static_cast<u32>(t.predicates.size()));
            const auto* ty = type_of_pred(p);
            ds->pred_type.push_back(ty && ds->type_of.contains(ty) ? ds->type_of.at(ty) : ~0u);
            t.predicates.push_back(out);
        }

    // functions: static, fluent, auxiliary (mimir: `total-cost` is auxiliary, fluent iff it appears in an effect)
    auto func_kind = [&](const std::string& name)
    {
        if (name == "total-cost")
            return FuncKind::Auxiliary;
        return ds->effect_functions.contains(name) ? FuncKind::Fluent : FuncKind::Static;
    };
    std::vector<loki::FunctionSkeleton> funcs_by_kind[3];
    for (const auto& f : dl->get_function_skeletons())
        funcs_by_kind[static_cast<int>(func_kind(f->get_name()))].push_back(f);
    for (int kind = 0; kind < 3; ++kind)
        for (const auto& f : funcs_by_kind[kind])
        {
            Function out;
            out.name = t.intern_string(f->get_name());
            out.kind = static_cast<FuncKind>(kind);
            out.arity = static_cast<u32>(f->get_parameters().size());
            out.params = loki_params(*ds, t, f->get_parameters());
            ds->func_of.emplace(f, static_cast<u32>(t.functions.size()));
            ds->func_by_name.emplace(f->get_name(), static_cast<u32>(t.functions.size()));
            t.functions.push_back(out);
        }

    // constants
    for (const auto& o : dl->get_constants())
    {
        Object out;
        out.name = t.intern_string(o->get_name());
        std::vector<TypeId> types;
        for (const auto& b : o->get_bases())
        {
            add_types(*ds, t, b);
            types.push_back(TypeId{ds->type_of.at(b)});
        }
        out.types = TaskData::append<TypeId>(t.type_ids, types);
        out.constant = true;
        ds->const_of.emplace(o, static_cast<u32>(t.objects.size()));
        ds->const_by_name.emplace(o->get_name(), static_cast<u32>(t.objects.size()));
        t.objects.push_back(out);
    }

    const Kinds kinds = domain_kinds(*ds);
    Lookups lk;
    lk.ds = ds.get();

    // ---- pass 1 (ToMimirStructures): predicates, function skeletons, actions, axioms, in loki's order
    Repo r1;
    Pass1 p1(r1, kinds, lk);
    std::vector<std::vector<u32>> pred_params1(t.predicates.size()), func_params1(t.functions.size());
    for (const auto& p : preds)
        pred_params1[ds->pred_of.at(p)] = p1.params(canonical_params(p));
    for (const auto& f : dl->get_function_skeletons())
        func_params1[ds->func_of.at(f)] = p1.params(f->get_parameters());
    std::vector<Action1> actions1;
    for (const auto& a : dl->get_actions())
        actions1.push_back(p1.action(a));
    std::vector<u32> axioms1;
    for (const auto& x : dl->get_axioms())
        axioms1.push_back(p1.axiom(x));
    std::sort(axioms1.begin(), axioms1.end());
    axioms1.erase(std::unique(axioms1.begin(), axioms1.end()), axioms1.end());

    // ---- pass 2 (EncodeParameterIndexInVariables) into the final domain repositories
    ds->r2d = std::make_unique<Repo>();
    Pass2 p2(*ds->r2d, r1, kinds);
    for (u32 p = 0; p < t.predicates.size(); ++p)  // ids are already ordered static, fluent, derived
        p2.unencoded_params(pred_params1[p]);
    for (u32 f = 0; f < t.functions.size(); ++f)
        p2.unencoded_params(func_params1[f]);
    std::vector<Pass2::Action2> actions2;
    for (const auto& a : actions1)
        actions2.push_back(p2.action(a));
    std::vector<u32> axioms2;
    for (u32 x : axioms1)
        axioms2.push_back(p2.axiom(x));
    std::sort(axioms2.begin(), axioms2.end());

    // ---- emission
    Emitter em(*ds->r2d, kinds, t);
    for (const auto& a : actions2)
        t.schemas.push_back(em.schema(a));
    for (u32 x : axioms2)
        t.axioms.push_back(em.axiom(x, false));
    validate(t);
    return ds;
}

// ================================================================================================ problem

ObjectMap map_objects(const DomainState& ds, const loki::Problem& problem)
{
    ObjectMap out(ds.const_of.begin(), ds.const_of.end());
    u32 next = static_cast<u32>(ds.data.objects.size());
    for (const auto& o : problem->get_objects())
        if (out.try_emplace(o, next).second)
            ++next;
    return out;
}

GroundInit read_loki_init(const DomainState& ds, const loki::Problem& problem, const ObjectMap& object_of)
{
    GroundInit init;
    Lookups lk;
    lk.ds = &ds;
    lk.objects = &object_of;
    std::vector<u32> objs;
    for (const auto& l : problem->get_initial_literals())
    {
        objs.clear();
        for (const auto& term : l->get_atom()->get_terms())
        {
            const auto* o = std::get_if<loki::Object>(&term->get_object_or_variable());
            if (!o)
                throw std::logic_error("Expected ground term.");
            objs.push_back(lk.object(*o));
        }
        init.add_atom(lk.pred(l->get_atom()->get_predicate()), l->get_polarity(), objs);
    }
    for (const auto& v : problem->get_initial_function_values())
    {
        objs.clear();
        for (const auto& term : v->get_function()->get_terms())
        {
            const auto* o = std::get_if<loki::Object>(&term->get_object_or_variable());
            if (!o)
                throw std::logic_error("Expected ground term.");
            objs.push_back(lk.object(*o));
        }
        init.add_value(lk.func(v->get_function()->get_function_skeleton()), objs, v->get_number());
    }
    return init;
}

TaskData translate_problem(const DomainState& ds, const loki::Problem& pl, const GroundInit& init, const ObjectMap& object_of,
                           const PredicateOrigin& origin)
{
    TaskData t = ds.data;  // the domain part
    t.problem_name = pl->get_name();
    for (const auto& r : requirement_strings(pl->get_requirements()))
        if (std::find(t.requirements.begin(), t.requirements.end(), r) == t.requirements.end())
            t.requirements.push_back(r);

    // problem objects (constants are already in the domain part; problem types are always domain types)
    t.objects.resize(ds.data.objects.size());
    {
        std::vector<std::pair<u32, loki::Object>> objs;
        for (const auto& o : pl->get_objects())
            objs.emplace_back(object_of.at(o), o);
        std::sort(objs.begin(), objs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [id, o] : objs)
        {
            if (id != t.objects.size())
                continue;  // declared twice
            Object out;
            out.name = t.intern_string(o->get_name());
            std::vector<TypeId> types;
            for (const auto& b : o->get_bases())
            {
                // the translated problem's types are the translated domain's (ProblemBuilder copies them), with a
                // by-name fallback
                const auto it = ds.type_of.find(b);
                const auto by_name = ds.type_by_name.find(b->get_name());
                if (it == ds.type_of.end() && by_name == ds.type_by_name.end())
                    fail("object " + o->get_name() + " has the undeclared type " + b->get_name());
                types.push_back(TypeId{it != ds.type_of.end() ? it->second : by_name->second});
            }
            out.types = TaskData::append<TypeId>(t.type_ids, types);
            t.objects.push_back(out);
        }
    }

    // problem-level derived predicates (mimir: every problem predicate is derived)
    Kinds kinds = domain_kinds(ds);
    absl::flat_hash_map<const loki::PredicateImpl*, u32> problem_preds;
    absl::flat_hash_map<std::string, u32> problem_preds_by_name;
    absl::flat_hash_map<u32, std::vector<u32>> problem_perm;
    std::vector<loki::ParameterList> problem_pred_params;  // canonical argument order
    for (const auto& p : pl->get_predicates())
    {
        if (ds.pred_by_name.contains(p->get_name()))
            fail("problem predicate " + p->get_name() + " shadows a domain predicate");
        const u32 id = static_cast<u32>(t.predicates.size());
        Predicate out;
        out.name = t.intern_string(p->get_name());
        out.kind = PredKind::Derived;
        out.arity = static_cast<u32>(p->get_parameters().size());
        loki::ParameterList lps = p->get_parameters();
        if (auto perm = generated_predicate_perm(p, origin); !perm.empty())
        {
            lps.clear();
            for (u32 j : perm)
                lps.push_back(p->get_parameters()[j]);
            problem_perm.emplace(id, std::move(perm));
        }
        problem_pred_params.push_back(lps);
        std::vector<Parameter> ps;
        for (const auto& q : lps)
        {
            Parameter x;
            x.name = t.intern_string(strip_question(q->get_variable()->get_name()));
            std::vector<TypeId> types;
            for (const auto& b : q->get_bases())
                types.push_back(TypeId{ds.type_by_name.at(b->get_name())});
            x.types = TaskData::append<TypeId>(t.type_ids, types);
            ps.push_back(x);
        }
        out.params = TaskData::append<Parameter>(t.params, ps);
        t.predicates.push_back(out);
        problem_preds.emplace(p, id);
        problem_preds_by_name.emplace(p->get_name(), id);
        kinds.pred.push_back(kDerived);
        kinds.pred_arity.push_back(out.arity);
    }
    for (const auto& x : pl->get_axioms())
        if (!problem_preds_by_name.contains(x->get_literal()->get_atom()->get_predicate()->get_name()))
            fail("problem axiom over a domain predicate is not supported");

    // mimir: ToMimirStructures::prepare(problem) rejects non-conjunctive goals and axiom bodies
    {
        absl::flat_hash_set<std::string> f, d, e;
        Classifier cl{f, d, e};
        if (pl->get_goal_condition())
            cl.condition(*pl->get_goal_condition());
        for (const auto& x : pl->get_axioms())
            cl.condition(x->get_condition());
    }

    Lookups lk;
    lk.ds = &ds;
    lk.objects = &object_of;
    lk.problem_preds = &problem_preds;
    lk.problem_preds_by_name = &problem_preds_by_name;
    lk.problem_perm = &problem_perm;

    // ---- pass 1 (ToMimirStructures::translate(problem)) into R1p, a child of the final domain repositories
    Repo r1(ds.r2d.get());
    Pass1 p1(r1, kinds, lk);
    std::vector<std::vector<u32>> pred_params1;
    for (const auto& ps : problem_pred_params)
        pred_params1.push_back(p1.params(ps));

    // canonical type order (domain.hpp): loki emits the type atoms of one object (AddTypePredicates) in the iteration
    // order of an std::unordered_set of type pointers; they are consecutive, and each such run is put in hierarchy order
    std::vector<u32> init_order(init.atoms.size());
    std::iota(init_order.begin(), init_order.end(), 0u);
    {
        auto type_obj = [&](u32 i) -> std::pair<u32, u32>  // (type, object) of a type atom, else (~0u, ~0u)
        {
            const auto& a = init.atoms[i];
            if (a.count != 1 || !a.positive || a.pred >= ds.pred_type.size() || ds.pred_type[a.pred] == ~0u)
                return {~0u, ~0u};
            return {ds.pred_type[a.pred], init.objects[a.begin]};
        };
        for (size_t i = 0; i < init_order.size();)
        {
            const u32 o = type_obj(static_cast<u32>(i)).second;
            size_t j = i + 1;
            if (o != ~0u)
                while (j < init_order.size() && type_obj(static_cast<u32>(j)).second == o)
                    ++j;
            if (j - i > 1 && o < t.objects.size())
            {
                const std::vector<u32> rank = hierarchy_ranks(t, TaskData::slice(t.type_ids, t.objects[o].types));
                std::stable_sort(init_order.begin() + static_cast<std::ptrdiff_t>(i), init_order.begin() + static_cast<std::ptrdiff_t>(j),
                                 [&](u32 a, u32 b) { return rank[type_obj(a).first] < rank[type_obj(b).first]; });
            }
            i = j;
        }
    }

    // Initial atoms. Mimir creates them in R1p, uniquifies, sorts by R1p index, recreates them in R2p and sorts
    // again; R1p and R2p are children of R2d, and nothing creates a ground literal in them before the initial atoms.
    // So per kind the order is: the atoms R2d already has (nullary literals of domain conditions) by R2d index, then
    // the others in order of first occurrence. That order is computed directly; only the atoms that later ground
    // literals of this problem can meet (nullary ones, goal literals, nullary literals of problem axioms) are entered
    // into R1p/R2p, which keeps every later id relation of mimir (rovers p30-hard has 232,000 initial atoms).
    absl::flat_hash_set<Key> later;  // keys of ground literals created after the initial atoms
    {
        auto collect = [&](const loki::Condition& c, bool nullary_only)
        {
            auto one = [&](const loki::Literal& l)
            {
                if (!l->get_polarity() || (nullary_only && !l->get_atom()->get_terms().empty()))
                    return;
                if (nullary_only)
                    later.insert(Key{lk.pred(l->get_atom()->get_predicate()), 1u});
                else if (std::all_of(l->get_atom()->get_terms().begin(), l->get_atom()->get_terms().end(),
                                     [](const loki::Term& t) { return std::holds_alternative<loki::Object>(t->get_object_or_variable()); }))
                    later.insert(p1.glit_key(l));
            };
            if (const auto* a = std::get_if<loki::ConditionAnd>(&c->get_condition()))
            {
                for (const auto& part : (*a)->get_conditions())
                    if (const auto* l = std::get_if<loki::ConditionLiteral>(&part->get_condition()))
                        one((*l)->get_literal());
            }
            else if (const auto* l = std::get_if<loki::ConditionLiteral>(&c->get_condition()))
                one((*l)->get_literal());
        };
        if (pl->get_goal_condition())
            collect(*pl->get_goal_condition(), false);
        for (const auto& x : pl->get_axioms())
            collect(x->get_condition(), true);
    }
    struct InitAtom
    {
        u32 index;  // into init.atoms
        u32 rank;   // R2d id, or ~0u
    };
    std::vector<InitAtom> order2[2];  // per kind, the final order
    std::vector<u32> tracked1;        // atoms entered into R1p, in creation order
    {
        // first occurrence: packed (pred, up to three objects < 2^16) where possible
        absl::flat_hash_set<u64> seen_packed;
        absl::flat_hash_set<Key> seen_keys;
        auto first_time = [&](const GroundInit::Atom& a)
        {
            const u32* o = init.objects.data() + a.begin;
            if (a.count <= 3 && a.pred < (1u << 16) && std::all_of(o, o + a.count, [](u32 x) { return x < (1u << 16); }))
            {
                u64 k = a.pred;
                for (u32 i = 0; i < a.count; ++i)
                    k |= u64{o[i]} << (16 * (i + 1));
                return seen_packed.insert(k).second;
            }
            Key key{a.pred};
            key.insert(key.end(), o, o + a.count);
            return seen_keys.insert(std::move(key)).second;
        };
        std::vector<InitAtom> members[2];
        seen_packed.reserve(init.atoms.size());
        for (const u32 ai : init_order)
        {
            const auto& a = init.atoms[ai];
            const u8 kind = kinds.pred.at(a.pred);
            if (kind == kDerived)
                fail("derived atom in the initial state");
            if (!a.positive)
                fail("negative literals in the initial state are not supported");  // mimir: ProblemImpl
            if (!first_time(a))
                continue;
            const std::span<const u32> objs(init.objects.data() + a.begin, a.count);
            const bool track = a.count == 0 || (!later.empty() && [&] {
                Key key{a.pred, 1u};
                key.insert(key.end(), objs.begin(), objs.end());
                return later.contains(key);
            }());
            if (track)
            {
                const u32 id = p1.glit(a.pred, true, objs);
                if (id < r1.glit.base())
                {
                    members[kind].push_back({ai, id});
                    continue;
                }
                tracked1.push_back(ai);
            }
            order2[kind].push_back({ai, ~0u});
        }
        for (int kind = 0; kind < 2; ++kind)
        {
            // a duplicated init atom ties on its rank: keep file order on every standard library
            std::stable_sort(members[kind].begin(), members[kind].end(), [](const InitAtom& x, const InitAtom& y) { return x.rank < y.rank; });
            order2[kind].insert(order2[kind].begin(), members[kind].begin(), members[kind].end());
        }
    }
    std::vector<u32> values1[3];
    for (const auto& v : init.values)
    {
        const u8 kind = kinds.func.at(v.func);
        Key key{v.func};
        key.insert(key.end(), init.objects.begin() + v.begin, init.objects.begin() + v.begin + v.count);
        key.push_back(f64_bits(v.value));
        values1[kind].push_back(r1.gfv[kind].id(key));
    }
    if (values1[kAuxiliary].size() > 1)
        fail("more than one initial value of the auxiliary function");
    for (auto& v : values1)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    std::vector<u32> goal1[3], goal_ncs1;
    if (pl->get_goal_condition())
    {
        auto add = [&](loki::Literal l)
        {
            const u32 id = p1.glit_of(l);
            goal1[kinds.pred.at(key_u32(r1.glit[id][0]))].push_back(id);
        };
        const auto& c = (*pl->get_goal_condition())->get_condition();
        if (const auto* a = std::get_if<loki::ConditionAnd>(&c))
        {
            for (const auto& part : (*a)->get_conditions())
            {
                if (const auto* l = std::get_if<loki::ConditionLiteral>(&part->get_condition()))
                    add((*l)->get_literal());
                else if (const auto* n = std::get_if<loki::ConditionNumericConstraint>(&part->get_condition()))
                    goal_ncs1.push_back(p1.gnc_of(*n));
                else
                    throw std::logic_error("Expected literal in conjunctive condition.");
            }
        }
        else if (const auto* l = std::get_if<loki::ConditionLiteral>(&c))
            add((*l)->get_literal());
        else if (const auto* n = std::get_if<loki::ConditionNumericConstraint>(&c))
            goal_ncs1.push_back(p1.gnc_of(*n));
        else
            throw std::logic_error("Expected conjunctive condition.");
    }
    for (auto& v : goal1)
        std::sort(v.begin(), v.end());
    std::sort(goal_ncs1.begin(), goal_ncs1.end());

    std::optional<std::pair<bool, u32>> metric1;
    if (pl->get_optimization_metric())
    {
        const auto& m = *pl->get_optimization_metric();
        metric1 = std::make_pair(m->get_optimization_metric() == loki::OptimizationMetricEnum::MINIMIZE,
                                 p1.gfexpr(m->get_function_expression()));
    }
    std::vector<u32> axioms1;
    for (const auto& x : pl->get_axioms())
        axioms1.push_back(p1.axiom(x));
    std::sort(axioms1.begin(), axioms1.end());
    axioms1.erase(std::unique(axioms1.begin(), axioms1.end()), axioms1.end());

    // ---- pass 2 (EncodeParameterIndexInVariables) into R2p, a child of the final domain repositories
    Repo r2(ds.r2d.get());
    Pass2 p2(r2, r1, kinds);
    for (const auto& ps : pred_params1)
        p2.unencoded_params(ps);
    // the tracked initial atoms, in mimir's pass-2 order (static, then fluent, each in its final order)
    if (!tracked1.empty())
    {
        const absl::flat_hash_set<u32> tracked(tracked1.begin(), tracked1.end());
        for (int kind = 0; kind < 2; ++kind)
            for (const InitAtom& ia : order2[kind])
                if (ia.rank != ~0u || tracked.contains(ia.index))
                {
                    const auto& a = init.atoms[ia.index];
                    Key key{a.pred, 1u};
                    key.insert(key.end(), init.objects.begin() + a.begin, init.objects.begin() + a.begin + a.count);
                    p2.glit(*r1.glit.find(key));
                }
    }
    std::vector<u32> values2[3];
    for (int kind = 0; kind < 3; ++kind)
    {
        for (u32 v : values1[kind])
            values2[kind].push_back(r2.gfv[kind].id(r1.gfv[kind][v]));
        std::sort(values2[kind].begin(), values2[kind].end());
    }
    std::vector<u32> goal2[3], goal_ncs2;
    for (int kind = 0; kind < 3; ++kind)
    {
        for (u32 g : goal1[kind])
            goal2[kind].push_back(p2.glit(g));
        std::sort(goal2[kind].begin(), goal2[kind].end());
    }
    for (u32 n : goal_ncs1)
        goal_ncs2.push_back(p2.gnc(n));
    std::sort(goal_ncs2.begin(), goal_ncs2.end());
    std::optional<std::pair<bool, u32>> metric2;
    if (metric1)
        metric2 = std::make_pair(metric1->first, p2.gfexpr(metric1->second));
    std::vector<u32> axioms2;
    for (u32 x : axioms1)
        axioms2.push_back(p2.axiom(x));
    std::sort(axioms2.begin(), axioms2.end());

    // ---- emission
    Emitter em(r2, kinds, t);
    t.object_ids.reserve(t.object_ids.size() + init.objects.size());
    for (int kind = 0; kind < 2; ++kind)
    {
        auto& out = kind == kStatic ? t.static_init : t.fluent_init;
        out.reserve(order2[kind].size());
        for (const InitAtom& ia : order2[kind])
        {
            const auto& a = init.atoms[ia.index];
            out.push_back(em.atom(a.pred, std::span<const u32>(init.objects.data() + a.begin, a.count)));
        }
    }
    auto values = [&](int kind, std::vector<GroundFunctionValue>& out)
    {
        for (u32 v : values2[kind])
        {
            const Key& key = r2.gfv[kind][v];
            GroundFunctionValue x;
            x.func = FunctionId{key_u32(key[0])};
            std::vector<ObjectId> objs;
            for (size_t i = 1; i + 1 < key.size(); ++i)
                objs.push_back(ObjectId{key_u32(key[i])});
            x.objects = TaskData::append<ObjectId>(t.object_ids, objs);
            x.value = bits_f64(key.back());
            out.push_back(x);
        }
    };
    values(kStatic, t.static_values);
    values(kFluent, t.fluent_values);
    if (!values2[kAuxiliary].empty())
        t.auxiliary_initial = bits_f64(r2.gfv[kAuxiliary][values2[kAuxiliary][0]].back());
    {
        std::vector<Literal> ls;
        for (const auto& kind : goal2)
            for (u32 g : kind)
                ls.push_back(em.glit(g));
        std::vector<NumericConstraint> ncs;
        for (u32 n : goal_ncs2)
            ncs.push_back(em.gnc(n));
        t.goal.literals = TaskData::append<Literal>(t.literals, ls);
        t.goal.constraints = TaskData::append<NumericConstraint>(t.constraints, ncs);
    }
    if (metric2)
    {
        Metric m;
        m.minimize = metric2->first;
        m.expr = em.gexpr(metric2->second);
        t.metric = m;
    }
    for (u32 x : axioms2)
        t.axioms.push_back(em.axiom(x, true));
    apply_union_types(ds, t, true);
    validate(t);
    return t;
}

// ================================================================================================ union types

bool apply_union_types(const DomainState& ds, TaskData& t, bool init)
{
    bool any = false;
    const u32 nt = static_cast<u32>(t.types.size());
    // anc[a * nt + b]: b is a or a supertype of a
    std::vector<u8> anc(static_cast<usize>(nt) * nt, 0);
    for (u32 a = 0; a < nt; ++a)
    {
        std::vector<u32> stack{a};
        while (!stack.empty())
        {
            const u32 x = stack.back();
            stack.pop_back();
            if (anc[static_cast<usize>(a) * nt + x])
                continue;
            anc[static_cast<usize>(a) * nt + x] = 1;
            for (TypeId b : TaskData::slice(t.type_ids, t.types[x].bases))
                stack.push_back(b.v);
        }
    }
    auto is_super = [&](u32 a, u32 b) { return a < nt && b < nt && anc[static_cast<usize>(a) * nt + b]; };
    auto type_of_pred = [&](u32 p) { return p < ds.pred_type.size() ? ds.pred_type[p] : ~0u; };

    struct Union
    {
        std::vector<u32> members;  // sorted type ids, none a subtype of another
        u32 pred = 0;
    };
    std::vector<Union> unions;
    auto union_pred = [&](const std::vector<u32>& members) -> u32
    {
        for (const Union& u : unions)
            if (u.members == members)
                return u.pred;
        std::string base = "either";
        for (u32 m : members)
            base += "_" + std::string(t.str(t.types[m].name));
        auto taken = [&](const std::string& name)
        {
            for (const Predicate& q : t.predicates)
                if (t.str(q.name) == name)
                    return true;
            return false;
        };
        std::string name = base;
        for (u32 k = 2; taken(name); ++k)
            name = base + "_" + std::to_string(k);
        Predicate q;
        q.name = t.intern_string(name);
        q.kind = PredKind::Static;
        q.arity = 1;
        Parameter par;
        par.name = t.intern_string("arg");
        std::vector<TypeId> ids;
        for (u32 m : members)
            ids.push_back(TypeId{m});
        par.types = TaskData::append<TypeId>(t.type_ids, ids);
        q.params = TaskData::append<Parameter>(t.params, std::span<const Parameter>(&par, 1));
        unions.push_back(Union{members, static_cast<u32>(t.predicates.size())});
        t.predicates.push_back(q);
        return unions.back().pred;
    };

    // The type literals of variable `var` (declared types `declared`, two or more) in condition c.
    auto rewrite = [&](Condition& c, u32 var, std::span<const TypeId> declared)
    {
        std::vector<u32> members;
        for (TypeId d : declared)
        {
            bool subsumed = false;
            for (TypeId e : declared)
                subsumed = subsumed || (e.v != d.v && is_super(d.v, e.v) && !(is_super(e.v, d.v) && e.v > d.v));
            if (!subsumed && std::find(members.begin(), members.end(), d.v) == members.end())
                members.push_back(d.v);
        }
        std::sort(members.begin(), members.end());
        const std::vector<Literal> lits(t.literals_of(c).begin(), t.literals_of(c).end());
        std::vector<Literal> out;
        bool changed = false, placed = false;
        for (const Literal& l : lits)
        {
            const u32 ty = type_of_pred(l.pred.v);
            const auto terms = t.terms_of(l);
            const bool type_lit = l.positive && ty != ~0u && terms.size() == 1 && terms[0] == static_cast<Term>(var) &&
                                  std::any_of(declared.begin(), declared.end(), [&](TypeId d) { return is_super(d.v, ty); });
            if (!type_lit)
            {
                out.push_back(l);
                continue;
            }
            if (members.size() == 1 && is_super(members[0], ty))
            {
                out.push_back(l);  // the type literals of the one remaining type
                continue;
            }
            changed = true;
            if (members.size() > 1 && !placed)
            {
                Literal u = l;  // the same single term
                u.pred = PredicateId{union_pred(members)};
                out.push_back(u);
                placed = true;
            }
        }
        if (changed)
        {
            c.literals = TaskData::append<Literal>(t.literals, out);
            any = true;
        }
    };
    auto rewrite_scope = [&](Condition& c, Range params, u32 first)
    {
        const std::vector<Parameter> ps(TaskData::slice(t.params, params).begin(), TaskData::slice(t.params, params).end());
        for (u32 i = 0; i < ps.size(); ++i)
            if (ps[i].types.count > 1)
            {
                const std::vector<TypeId> declared(TaskData::slice(t.type_ids, ps[i].types).begin(),
                                                   TaskData::slice(t.type_ids, ps[i].types).end());
                rewrite(c, first + i, declared);
            }
    };
    for (Schema& s : t.schemas)
    {
        rewrite_scope(s.precondition, s.params, 0);
        for (u32 e = s.effects.begin; e < s.effects.end(); ++e)
            rewrite_scope(t.conditional_effects[e].condition, t.conditional_effects[e].extra_params, s.arity());
    }
    for (Axiom& x : t.axioms)
        rewrite_scope(x.body, x.params, 0);

    if (!init || unions.empty())
        return any;
    // the objects of any member type: loki gives every object the type atoms of its declared types and their supertypes
    std::vector<u32> type_pred(nt, ~0u);
    for (u32 p = 0; p < ds.pred_type.size(); ++p)
        if (ds.pred_type[p] < nt)
            type_pred[ds.pred_type[p]] = p;
    for (const Union& u : unions)
    {
        std::vector<u8> member(t.objects.size(), 0);
        for (const GroundAtom& a : t.static_init)
            for (u32 m : u.members)
                if (a.pred.v == type_pred[m] && a.objects.count == 1)
                    member[t.object_ids[a.objects.begin].v] = 1;
        for (u32 o = 0; o < member.size(); ++o)
            if (member[o])
            {
                const ObjectId id{o};
                t.static_init.push_back(GroundAtom{PredicateId{u.pred}, TaskData::append<ObjectId>(t.object_ids, std::span<const ObjectId>(&id, 1))});
            }
    }
    return any;
}
}  // namespace mymyr::frontend::detail
