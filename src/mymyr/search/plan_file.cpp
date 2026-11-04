// IPC plan text (search/plan_file.hpp).

#include "mymyr/search/plan_file.hpp"

#include "plan_line.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace mymyr::search
{
namespace
{
State start_state(const Task& task, const std::optional<State>& start)
{
    if (!start)
        return task.initial_state();
    if (start->numeric_words() != task.numeric_words())
        throw std::invalid_argument("mymyr plan: the start state has no numeric values of this task");
    return *start;
}
}  // namespace

std::string format_plan(const Task& task, std::span<const Action> plan, const std::optional<State>& start)
{
    const formalism::TaskData& D = task.data();
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    const heuristics::ActionCosts costs(task);
    State s = start_state(task, start);
    f64 g = costs.initial(s.view());
    std::string out;
    for (usize i = 0; i < plan.size(); ++i)
    {
        const Action& a = plan[i];
        if (a.schema.v >= D.schemas.size() || a.binding.size() != D.schemas[a.schema.v].arity())
            throw std::invalid_argument(std::format("mymyr plan: action {} is not an action of this task", i + 1));
        if (!succ.is_applicable(s.view(), a.label()))
            throw std::invalid_argument(std::format("mymyr plan: action {} is not applicable in its state", i + 1));
        const formalism::Schema& sc = D.schemas[a.schema.v];
        out += '(';
        out += D.str(sc.name);
        const std::vector<ObjectId> args = formalism::full_binding(D, sc, a.binding);
        for (u32 k = 0, n = formalism::written_arity(D, sc); k < n && k < args.size(); ++k)
        {
            out += ' ';
            out += D.str(D.objects[args[k].v].name);
        }
        out += ")\n";
        StateBuilder b;
        const Delta d = succ.apply_with_delta(s.view(), a.label(), b);
        g = costs.next(g, d);
        s = b.build();
    }
    out += std::format("; cost = {} ({})\n", g, costs.unit() ? "unit cost" : "general cost");
    return out;
}

namespace detail
{
std::vector<ObjectId> binding_of_line(const formalism::TaskData& D, const formalism::Schema& sc, std::span<const ObjectId> args)
{
    const auto collapsed = D.collapsed_of(sc);
    std::vector<ObjectId> out;
    usize c = 0;
    for (usize k = 0; k < args.size(); ++k)
    {
        while (c < collapsed.size() && collapsed[c].position < k)
            ++c;
        if (c < collapsed.size() && collapsed[c].position == k)
            continue;
        out.push_back(args[k]);
    }
    return out;
}

const formalism::Parameter& line_parameter(const formalism::TaskData& D, const formalism::Schema& sc, u32 k)
{
    u32 kept = k;
    for (const formalism::CollapsedParameter& c : D.collapsed_of(sc))
    {
        if (c.position == k)
            return c.param;
        if (c.position < k)
            --kept;
    }
    return D.params[sc.params.begin + kept];
}

bool of_type(const formalism::TaskData& D, ObjectId o, TypeId ty)
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

std::string type_violation(const formalism::TaskData& D, const formalism::Schema& sc, std::span<const ObjectId> args,
                           u32 count, bool line)
{
    for (u32 k = 0; k < count && k < args.size(); ++k)
    {
        const formalism::Parameter& p = line ? line_parameter(D, sc, k) : D.params[sc.params.begin + k];
        if (p.types.count == 0)
            continue;
        bool ok = false;
        for (u32 j = p.types.begin; j < p.types.end() && !ok; ++j)
            ok = of_type(D, args[k], D.type_ids[j]);
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
        return std::string(D.str(D.objects[args[k].v].name)) + " - " + ty;
    }
    return {};
}

PlanLine resolve_plan_line(const formalism::TaskData& D, std::span<const std::string_view> t)
{
    PlanLine r;
    const usize given = t.size() - 1;
    for (u32 s = 0; s < D.schemas.size(); ++s)
        if (same_name(D.str(D.schemas[s].name), t[0]))
        {
            r.named.push_back(s);
            if (given == formalism::written_arity(D, D.schemas[s]) || given == D.schemas[s].arity() + D.schemas[s].collapsed.count)
                r.fitting.push_back(s);
        }
    if (r.named.empty())
    {
        r.status = PlanLine::Status::UnknownAction;
        return r;
    }
    if (r.fitting.empty())
    {
        r.status = PlanLine::Status::WrongArity;
        return r;
    }
    for (usize k = 0; k < given; ++k)
    {
        u32 o = 0;
        while (o < D.num_objects() && !same_name(D.str(D.objects[o].name), t[k + 1]))
            ++o;
        if (o == D.num_objects())
        {
            r.status = PlanLine::Status::UnknownObject;
            r.unknown = k;
            return r;
        }
        r.args.push_back(ObjectId{o});
    }
    return r;
}

std::optional<Action> match_plan_line(const formalism::TaskData& D, Successors& succ, StateView s,
                                      std::span<const u32> schemas, std::span<const ObjectId> line_args)
{
    // ascending schemas, and per schema the first binding in canonical order: the first match is the least action
    for (const u32 schema : schemas)
    {
        const formalism::Schema& sc = D.schemas[schema];
        if (sc.collapsed.count != 0 && !type_violation(D, sc, line_args, static_cast<u32>(line_args.size()), true).empty())
            continue;  // an object not of its parameter's type: not an action of this schema
        const std::vector<ObjectId> bound = binding_of_line(D, sc, line_args);
        const std::span<const ObjectId> args = bound;
        const u32 arity = sc.arity();
        if (args.size() == arity)
        {
            if (succ.is_applicable(s, ActionLabel{SchemaId{schema}, args}))
                return Action(SchemaId{schema}, std::vector<ObjectId>(args.begin(), args.end()));
            continue;
        }
        if (args.size() > arity)
            continue;
        std::optional<Action> found;
        succ.prepare(s);
        succ.generate<true>(
            [&](u32 x, const ObjectId* binding, const Delta&) -> bool {
                if (!std::equal(args.begin(), args.end(), binding))
                    return true;
                found.emplace(SchemaId{x}, std::vector<ObjectId>(binding, binding + arity));
                return false;
            },
            /*witness_pruning=*/false, /*canonical_order=*/true, schema, schema + 1);
        if (found)
            return found;
    }
    return std::nullopt;
}
}  // namespace detail

std::vector<Action> parse_plan(const Task& task, std::string_view text, const std::optional<State>& start)
{
    const formalism::TaskData& D = task.data();
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = start_state(task, start);
    std::vector<Action> plan;
    usize line_no = 0;
    while (!text.empty())
    {
        const usize nl = text.find('\n');
        const std::string_view line = detail::trim(text.substr(0, nl));
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        ++line_no;
        if (line.empty() || line.front() == ';')
            continue;
        const usize open = line.find('('), close = line.rfind(')');
        if (open == std::string_view::npos || close == std::string_view::npos || close < open)
            throw std::invalid_argument(std::format("mymyr plan: line {} is not an action '(name o1 ... on)'", line_no));
        const std::vector<std::string_view> t = detail::tokens(line.substr(open + 1, close - open - 1));
        if (t.empty())
            throw std::invalid_argument(std::format("mymyr plan: line {} names no action", line_no));
        const detail::PlanLine r = detail::resolve_plan_line(D, t);
        switch (r.status)
        {
            case detail::PlanLine::Status::Ok: break;
            case detail::PlanLine::Status::UnknownAction:
                throw std::invalid_argument(
                    std::format("mymyr plan: line {} '{}': the task has no action '{}'", line_no, line, t[0]));
            case detail::PlanLine::Status::WrongArity:
            {
                const formalism::Schema& sc = D.schemas[r.named.front()];
                throw std::invalid_argument(std::format("mymyr plan: line {} '{}': {} takes {} argument{}, the line "
                                                        "has {}",
                                                        line_no, line, D.str(sc.name), formalism::written_arity(D, sc),
                                                        formalism::written_arity(D, sc) == 1 ? "" : "s", t.size() - 1));
            }
            case detail::PlanLine::Status::UnknownObject:
                throw std::invalid_argument(
                    std::format("mymyr plan: line {} '{}': argument {} '{}' is no object of the task", line_no, line,
                                r.unknown + 1, t[r.unknown + 1]));
        }
        std::optional<Action> found = detail::match_plan_line(D, succ, s.view(), r.fitting, r.args);
        if (!found)
        {
            // an argument of the wrong type for every schema the line fits
            std::string type_error;
            bool typed = false;
            for (const u32 x : r.fitting)
            {
                std::string e = detail::type_violation(D, D.schemas[x], r.args, static_cast<u32>(r.args.size()), true);
                typed = typed || e.empty();
                if (type_error.empty())
                    type_error = std::move(e);
            }
            if (!typed)
                throw std::invalid_argument(std::format("mymyr plan: line {} '{}': the argument {} is not of its parameter's type",
                                                        line_no, line, type_error));
        }
        if (!found)
            throw std::invalid_argument(std::format(
                "mymyr plan: line {} '{}': the action is not applicable in its state (validate_plan names the failing "
                "precondition)",
                line_no, line));
        s = succ.apply(s.view(), found->label());
        plan.push_back(std::move(*found));
    }
    return plan;
}
}  // namespace mymyr::search
