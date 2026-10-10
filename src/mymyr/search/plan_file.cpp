// IPC plan text (search/plan_file.hpp).

#include "mymyr/search/plan_file.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <stdexcept>

namespace mymyr::search
{
namespace
{
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
        const std::string_view line = trim(text.substr(0, nl));
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        ++line_no;
        if (line.empty() || line.front() == ';')
            continue;
        const usize open = line.find('('), close = line.rfind(')');
        if (open == std::string_view::npos || close == std::string_view::npos || close < open)
            throw std::invalid_argument(std::format("mymyr plan: line {} is not an action '(name o1 ... on)'", line_no));
        const std::vector<std::string_view> t = tokens(line.substr(open + 1, close - open - 1));
        if (t.empty())
            throw std::invalid_argument(std::format("mymyr plan: line {} names no action", line_no));
        const usize given = t.size() - 1;
        std::optional<Action> found;
        succ.for_each_applicable(s.view(), [&](const ActionLabel& a, const Delta&) {
            const formalism::Schema& sc = D.schemas[a.schema.v];
            if (!same_name(D.str(sc.name), t[0]) || (given != sc.original_arity && given != a.binding.size()))
                return;
            for (usize k = 0; k < given; ++k)
                if (!same_name(D.str(D.objects[a.binding[k].v].name), t[k + 1]))
                    return;
            Action x(a);
            if (!found || x < *found)
                found = std::move(x);
        });
        if (!found)
            throw std::invalid_argument(
                std::format("mymyr plan: line {}: no applicable action matches '{}'", line_no, line));
        s = succ.apply(s.view(), found->label());
        plan.push_back(std::move(*found));
    }
    return plan;
}
}  // namespace mymyr::search
