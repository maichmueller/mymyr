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
        for (u32 k = 0; k < sc.original_arity && k < a.binding.size(); ++k)
        {
            out += ' ';
            out += D.str(D.objects[a.binding[k].v].name);
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
PlanLine resolve_plan_line(const formalism::TaskData& D, std::span<const std::string_view> t)
{
    PlanLine r;
    const usize given = t.size() - 1;
    for (u32 s = 0; s < D.schemas.size(); ++s)
        if (same_name(D.str(D.schemas[s].name), t[0]))
        {
            r.named.push_back(s);
            if (given == D.schemas[s].original_arity || given == D.schemas[s].arity())
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
                                      std::span<const u32> schemas, std::span<const ObjectId> args)
{
    // ascending schemas, and per schema the first binding in canonical order: the first match is the least action
    for (const u32 schema : schemas)
    {
        const u32 arity = D.schemas[schema].arity();
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
                                                        line_no, line, D.str(sc.name), sc.original_arity,
                                                        sc.original_arity == 1 ? "" : "s", t.size() - 1));
            }
            case detail::PlanLine::Status::UnknownObject:
                throw std::invalid_argument(
                    std::format("mymyr plan: line {} '{}': argument {} '{}' is no object of the task", line_no, line,
                                r.unknown + 1, t[r.unknown + 1]));
        }
        std::optional<Action> found = detail::match_plan_line(D, succ, s.view(), r.fitting, r.args);
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
