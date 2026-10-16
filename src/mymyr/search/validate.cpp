// Plan validation (search/validate.hpp).

#include "mymyr/search/validate.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/conditions.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <stdexcept>

namespace mymyr::search
{
namespace
{
using formalism::TaskData;

/// "1 argument", "2 arguments".
std::string arguments(usize n)
{
    return std::format("{} argument{}", n, n == 1 ? "" : "s");
}

/// "first", "second", ... for the k-th argument (0-based k).
std::string ordinal(usize k)
{
    static constexpr std::array<const char*, 10> words{"first",   "second",  "third",  "fourth", "fifth",
                                                       "sixth",   "seventh", "eighth", "ninth",  "tenth"};
    return k < words.size() ? std::string(words[k]) : std::format("{}th", k + 1);
}

bool same_name(std::string_view a, std::string_view b)
{
    return std::ranges::equal(a, b, [](char x, char y) {
        return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
    });
}

std::string_view trim(std::string_view s)
{
    const auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while (!s.empty() && space(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && space(s.back()))
        s.remove_suffix(1);
    return s;
}

std::vector<std::string_view> tokens(std::string_view s)
{
    std::vector<std::string_view> out;
    usize i = 0;
    while (i < s.size())
    {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
            ++i;
        const usize b = i;
        while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i])))
            ++i;
        if (i > b)
            out.push_back(s.substr(b, i - b));
    }
    return out;
}

/// The text of a one-element ground condition without its "(and ...)".
std::string unwrap(std::string s)
{
    if (s.starts_with("(and ") && s.ends_with(")"))
        return s.substr(5, s.size() - 6);
    return s;
}

std::string literal_text(const TaskData& D, const GroundLiteral& l)
{
    GroundCondition c;
    c.literals.push_back(l);
    return unwrap(c.str(D));
}

/// Ground condition `g` reduced to its constraint i (the expressions are shared).
GroundCondition only_constraint(const GroundCondition& g, usize i)
{
    GroundCondition c;
    c.exprs = g.exprs;
    c.expr_terms = g.expr_terms;
    c.constraints.push_back(g.constraints[i]);
    return c;
}

/// Whether object o is of type `ty`: one of its declared types or a supertype of one is `ty`.
bool of_type(const TaskData& D, ObjectId o, TypeId ty)
{
    if (D.str(D.types[ty.v].name) == "object")
        return true;
    std::vector<TypeId> todo(D.type_ids.begin() + D.objects[o.v].types.begin,
                             D.type_ids.begin() + D.objects[o.v].types.end());
    std::vector<bool> seen(D.types.size(), false);
    while (!todo.empty())
    {
        const TypeId t = todo.back();
        todo.pop_back();
        if (t == ty)
            return true;
        if (t.v >= D.types.size() || seen[t.v])
            continue;
        seen[t.v] = true;
        const formalism::Type& T = D.types[t.v];
        todo.insert(todo.end(), D.type_ids.begin() + T.bases.begin, D.type_ids.begin() + T.bases.end());
    }
    return false;
}

/// The first parameter (of the first `count`) whose declared types the binding's object has none of: its text
/// "o - t1" (or "o - (either t1 t2)"), else empty.
std::string type_violation(const TaskData& D, const formalism::Schema& sc, std::span<const ObjectId> binding, u32 count)
{
    for (u32 k = 0; k < count && k < binding.size(); ++k)
    {
        const formalism::Parameter& p = D.params[sc.params.begin + k];
        if (p.types.count == 0)
            continue;
        bool ok = false;
        for (u32 j = p.types.begin; j < p.types.end() && !ok; ++j)
            ok = of_type(D, binding[k], D.type_ids[j]);
        if (ok)
            continue;
        std::string ty;
        if (p.types.count == 1)
            ty = std::string(D.str(D.types[D.type_ids[p.types.begin].v].name));
        else
        {
            ty = "(either";
            for (u32 j = p.types.begin; j < p.types.end(); ++j)
                ty += " " + std::string(D.str(D.types[D.type_ids[j].v].name));
            ty += ")";
        }
        return std::string(D.str(D.objects[binding[k].v].name)) + " - " + ty;
    }
    return {};
}

/// Why schema `schema` under `binding` (its first `bound` parameters; the others are the existential parameters
/// normalization added) is not applicable in s: the first precondition literal, equality or numeric constraint that
/// fails. Empty when every condition over the bound parameters holds.
std::string unsatisfied_precondition(const Task& task, StateView s, SchemaId schema, std::span<const ObjectId> binding,
                                     u32 bound)
{
    const TaskData& D = task.data();
    const ConjunctiveCondition pre = ConjunctiveCondition::precondition(task, schema);
    auto object_of = [&](formalism::Term t, ObjectId& o) {
        if (formalism::is_object(t))
        {
            o = formalism::term_object(t);
            return true;
        }
        const u32 v = formalism::term_parameter(t);
        if (v >= bound)
            return false;
        o = binding[v];
        return true;
    };
    for (const ConjunctiveCondition::Equality& e : pre.equalities)
    {
        ObjectId a, b;
        if (!object_of(e.lhs, a) || !object_of(e.rhs, b))
            continue;
        if ((a == b) != e.positive)
            return std::format("({} {} {})", e.positive ? "=" : "!=", D.str(D.objects[a.v].name),
                               D.str(D.objects[b.v].name));
    }
    for (const ConjunctiveCondition::Literal& l : pre.literals)
    {
        GroundLiteral g{{l.predicate, {}}, l.positive};
        bool ground = true;
        for (const formalism::Term t : l.terms)
        {
            ObjectId o;
            if (!object_of(t, o))
            {
                ground = false;
                break;
            }
            g.atom.objects.push_back(o);
        }
        if (ground && !holds(task, s, g))
            return literal_text(D, g);
    }
    if (bound == pre.arity() && !pre.constraints.empty())
    {
        std::vector<ObjectId> full(binding.begin(), binding.end());
        const GroundCondition g = pre.ground(full);
        for (usize i = 0; i < g.constraints.size(); ++i)
        {
            const GroundCondition c = only_constraint(g, i);
            if (!holds(task, s, c))
                return unwrap(c.str(D));
        }
    }
    return {};
}

std::string step_text(const TaskData& D, const Action& a, u32 count)
{
    std::string out = "(" + std::string(D.str(D.schemas[a.schema.v].name));
    for (u32 k = 0; k < count && k < a.binding.size(); ++k)
        out += " " + std::string(D.str(D.objects[a.binding[k].v].name));
    return out + ")";
}

/// Replays plans and records the outcome.
class Validator
{
public:
    Validator(const Task& task, const std::optional<State>& start)
        : m_task(task), m_D(task.data()), m_lease(task.workspace()), m_succ(m_lease->successors()), m_costs(task)
    {
        if (start && start->numeric_words() != task.numeric_words())
            throw std::invalid_argument("mymyr validate_plan: the start state has no numeric values of this task");
        m_r.state = start ? *start : task.initial_state();
        m_r.cost = m_costs.initial(m_r.state.view());
    }

    /// Ends the validation at step `step` with `error`.
    PlanValidation& fail(PlanError error, u64 step, u64 line, std::string action, std::string condition,
                         std::string why)
    {
        m_r.valid = false;
        m_r.error = error;
        m_r.step = step;
        m_r.line = line;
        m_r.action = std::move(action);
        m_r.condition = std::move(condition);
        m_r.message = line ? std::format("plan[{}] {} (line {}): {}", step, m_r.action, line, why)
                           : std::format("plan[{}] {}: {}", step, m_r.action, why);
        return m_r;
    }

    /// The reason a resolved, well-typed step is not applicable.
    PlanValidation& not_applicable(u64 step, u64 line, std::string text, SchemaId schema,
                                   std::span<const ObjectId> binding, u32 bound)
    {
        std::string cond = unsatisfied_precondition(m_task, m_r.state.view(), schema, binding, bound);
        const formalism::Schema& sc = m_D.schemas[schema.v];
        std::string why;
        if (!cond.empty())
            why = "the precondition " + cond + " does not hold";
        else if (bound < sc.arity())
            why = "no objects for its existential variables satisfy the precondition";
        else
            why = "its effects are undefined in this state (an undefined numeric value, a division by zero or "
                  "conflicting effects)";
        return fail(PlanError::NotApplicable, step, line, std::move(text), std::move(cond), std::move(why));
    }

    void apply(const Action& a)
    {
        StateBuilder b;
        const Delta d = m_succ.apply_with_delta(m_r.state.view(), a.label(), b);
        m_r.cost = m_costs.next(m_r.cost, d);
        m_r.state = b.build();
        m_r.plan.push_back(a);
    }

    /// After the last step: the goal.
    PlanValidation& finish(u64 steps)
    {
        m_r.step = steps;
        if (m_succ.is_goal(m_r.state.view()))
        {
            m_r.valid = true;
            return m_r;
        }
        const GroundCondition goal = GroundCondition::goal(m_task);
        std::string cond;
        for (const GroundLiteral& l : goal.literals)
            if (!holds(m_task, m_r.state.view(), l))
            {
                cond = literal_text(m_D, l);
                break;
            }
        for (usize i = 0; i < goal.constraints.size() && cond.empty(); ++i)
        {
            const GroundCondition c = only_constraint(goal, i);
            if (!holds(m_task, m_r.state.view(), c))
                cond = unwrap(c.str(m_D));
        }
        m_r.valid = false;
        m_r.error = PlanError::GoalNotReached;
        m_r.action.clear();
        m_r.condition = cond;
        const std::string where =
            steps == 0 ? std::string("in the start state (the plan is empty)")
                       : std::format("after the plan's {} action{}", steps, steps == 1 ? "" : "s");
        m_r.message = cond.empty() ? std::format("the goal does not hold {}", where)
                                   : std::format("the goal does not hold {}: {} is false", where, cond);
        return m_r;
    }

    [[nodiscard]] const Task& task() const { return m_task; }
    [[nodiscard]] const TaskData& data() const { return m_D; }
    [[nodiscard]] Successors& succ() { return m_succ; }
    [[nodiscard]] StateView state() const { return m_r.state.view(); }
    [[nodiscard]] PlanValidation& result() { return m_r; }

private:
    const Task& m_task;
    const TaskData& m_D;
    WorkspaceLease m_lease;
    Successors& m_succ;
    heuristics::ActionCosts m_costs;
    PlanValidation m_r;
};
}  // namespace

const char* to_string(PlanError e) noexcept
{
    switch (e)
    {
        case PlanError::None: return "none";
        case PlanError::Syntax: return "syntax";
        case PlanError::UnknownAction: return "unknown_action";
        case PlanError::WrongArity: return "wrong_arity";
        case PlanError::UnknownObject: return "unknown_object";
        case PlanError::WrongType: return "wrong_type";
        case PlanError::NotApplicable: return "not_applicable";
        case PlanError::GoalNotReached: return "goal_not_reached";
    }
    return "?";
}

PlanValidation validate_plan(const Task& task, std::span<const Action> plan, const std::optional<State>& start)
{
    Validator v(task, start);
    const TaskData& D = v.data();
    for (usize i = 0; i < plan.size(); ++i)
    {
        const Action& a = plan[i];
        if (a.schema.v >= D.schemas.size())
            return v.fail(PlanError::UnknownAction, i, 0, std::format("(<schema {}>)", a.schema.v), {},
                          std::format("the task has no action schema {} (it has {})", a.schema.v, D.schemas.size()));
        const formalism::Schema& sc = D.schemas[a.schema.v];
        const std::string name(D.str(sc.name));
        if (a.binding.size() != sc.arity())
            return v.fail(PlanError::WrongArity, i, 0, "(" + name + " ...)", {},
                          std::format("{} takes {}, the action has {}", name, arguments(sc.arity()), a.binding.size()));
        for (usize k = 0; k < a.binding.size(); ++k)
            if (a.binding[k].v >= D.num_objects())
                return v.fail(PlanError::UnknownObject, i, 0, "(" + name + " ...)", {},
                              std::format("the {} argument is object {}, but the task has {} objects", ordinal(k),
                                          a.binding[k].v, D.num_objects()));
        const std::string text = step_text(D, a, sc.arity());
        if (std::string t = type_violation(D, sc, a.binding, sc.arity()); !t.empty())
            return v.fail(PlanError::WrongType, i, 0, text, t, "the argument " + t + " is not of its parameter's type");
        if (!v.succ().is_applicable(v.state(), a.label()))
            return v.not_applicable(i, 0, text, a.schema, a.binding, sc.arity());
        v.apply(a);
    }
    return v.finish(plan.size());
}

PlanValidation validate_plan_text(const Task& task, std::string_view text, const std::optional<State>& start)
{
    Validator v(task, start);
    const TaskData& D = v.data();
    u64 line_no = 0, step = 0;
    while (!text.empty())
    {
        const usize nl = text.find('\n');
        const std::string_view line = trim(text.substr(0, nl));
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        ++line_no;
        if (line.empty() || line.front() == ';')
            continue;
        const usize open = line.find('('), close = line.rfind(')');
        if (open != 0 || close != line.size() - 1)
            return v.fail(PlanError::Syntax, step, line_no, std::string(line), {}, "not an action '(name o1 ... ok)'");
        const std::vector<std::string_view> t = tokens(line.substr(1, line.size() - 2));
        if (t.empty())
            return v.fail(PlanError::Syntax, step, line_no, std::string(line), {}, "names no action");
        const std::string written(line);
        const usize given = t.size() - 1;
        // the schemas of this name, and those of them that take `given` arguments
        std::vector<u32> named, fitting;
        for (u32 s = 0; s < D.schemas.size(); ++s)
            if (same_name(D.str(D.schemas[s].name), t[0]))
            {
                named.push_back(s);
                if (given == D.schemas[s].original_arity || given == D.schemas[s].arity())
                    fitting.push_back(s);
            }
        if (named.empty())
            return v.fail(PlanError::UnknownAction, step, line_no, written, {},
                          std::format("the task has no action '{}'", t[0]));
        if (fitting.empty())
            return v.fail(PlanError::WrongArity, step, line_no, written, {},
                          std::format("{} takes {}, the action has {}", D.str(D.schemas[named[0]].name),
                                      arguments(D.schemas[named[0]].original_arity), given));
        std::vector<ObjectId> args;
        for (usize k = 0; k < given; ++k)
        {
            u32 o = 0;
            while (o < D.num_objects() && !same_name(D.str(D.objects[o].name), t[k + 1]))
                ++o;
            if (o == D.num_objects())
                return v.fail(PlanError::UnknownObject, step, line_no, written, {},
                              std::format("the {} argument '{}' is no object of the task", ordinal(k), t[k + 1]));
            args.push_back(ObjectId{o});
        }
        // the declared types: the step is well typed when it is for one of the fitting schemas
        std::string type_error;
        bool typed = false;
        for (const u32 s : fitting)
        {
            std::string e = type_violation(D, D.schemas[s], args, static_cast<u32>(given));
            if (e.empty())
                typed = true;
            else if (type_error.empty())
                type_error = std::move(e);
        }
        if (!typed)
            return v.fail(PlanError::WrongType, step, line_no, written, type_error,
                          "the argument " + type_error + " is not of its parameter's type");
        std::optional<Action> found;
        v.succ().for_each_applicable(v.state(), [&](const ActionLabel& a, const Delta&) {
            if (std::ranges::find(fitting, a.schema.v) == fitting.end())
                return;
            const formalism::Schema& sc = D.schemas[a.schema.v];
            if (given != sc.original_arity && given != a.binding.size())
                return;
            if (!std::ranges::equal(args, a.binding.subspan(0, given)))
                return;
            Action x(a);
            if (!found || x < *found)
                found = std::move(x);
        });
        if (!found)
        {
            // the reason: of the first schema the step names (canonical order), over the arguments it gives
            const u32 s = fitting.front();
            return v.not_applicable(step, line_no, written, SchemaId{s}, args, static_cast<u32>(given));
        }
        v.apply(*found);
        ++step;
    }
    return v.finish(step);
}
}  // namespace mymyr::search
