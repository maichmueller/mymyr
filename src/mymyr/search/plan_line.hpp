#pragma once
// One line of IPC plan text resolved to an action of a task (search::parse_plan, search::validate_plan_text).
//
// A line "(name o1 ... ok)" names a schema and objects. It names either the PDDL action's own parameters
// (Schema::original_arity) or all of the schema's parameters (normalization adds parameters for existential
// preconditions and may split an action into several schemas of one name). The action it names in a state is, of the
// applicable actions of the fitting schemas whose binding starts with the named objects, the first in canonical order.
// The result does not depend on the successor generator's witness pruning: a line that names a full binding is that
// binding, and a line that leaves parameters open has every applicable completion as a candidate.

#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mymyr
{
class Successors;
namespace formalism
{
struct TaskData;
}
}  // namespace mymyr

namespace mymyr::search::detail
{
/// Whether two names are equal without regard to case.
inline bool same_name(std::string_view a, std::string_view b)
{
    return std::ranges::equal(a, b, [](char x, char y) {
        return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
    });
}

inline std::string_view trim(std::string_view s)
{
    const auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while (!s.empty() && space(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && space(s.back()))
        s.remove_suffix(1);
    return s;
}

/// The whitespace-separated tokens of s.
inline std::vector<std::string_view> tokens(std::string_view s)
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

/// A line's tokens (the action name, then the objects) resolved against the task's names.
struct PlanLine
{
    enum class Status : u8
    {
        Ok,
        UnknownAction,  // no schema has the name
        WrongArity,     // no schema of the name takes that many arguments
        UnknownObject,  // the argument `unknown` names no object
    };
    Status status = Status::Ok;
    std::vector<u32> named;      // the schemas of the line's name, ascending
    std::vector<u32> fitting;    // those of them whose original_arity or arity is the number of arguments
    std::vector<ObjectId> args;  // the objects the arguments name (complete when status is Ok)
    usize unknown = 0;           // UnknownObject: the 0-based argument
};

/// Resolves the tokens of a line ("name", "o1", ..., "ok"; at least the name).
[[nodiscard]] PlanLine resolve_plan_line(const formalism::TaskData& D, std::span<const std::string_view> tokens);

/// The action that `args` names among `schemas` (ascending) in state s: of the applicable actions of those schemas
/// whose binding starts with args, the first in canonical order; nullopt if there is none. A schema args binds fully is
/// tested with is_applicable; a schema whose normalization-added parameters args leaves open is enumerated with
/// witness pruning off. Independent of succ's witness pruning and canonical order settings.
[[nodiscard]] std::optional<Action> match_plan_line(const formalism::TaskData& D, Successors& succ, StateView s,
                                                    std::span<const u32> schemas, std::span<const ObjectId> args);
}  // namespace mymyr::search::detail
