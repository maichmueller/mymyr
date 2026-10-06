#pragma once
// Plans as text in the IPC format that planners write and plan validators (VAL) read:
//
//   (pick ball1 rooma left)
//   (move rooma roomb)
//   (drop ball1 roomb left)
//   ; cost = 3 (unit cost)
//
// One line per action with the PDDL action's own parameters (mimir's plan format: the parameters that normalization
// added, Schema::original_arity onwards, are not written), then the cost line: "(unit cost)" for a task without
// action costs, "(general cost)" otherwise.
//
//   std::string text = mymyr::search::format_plan(*task, result.plan);
//   std::vector<mymyr::Action> plan = mymyr::search::parse_plan(*task, text);

#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::search
{
/// The plan as IPC plan text. The cost is mimir's metric value of the state the plan reaches from `start` (default:
/// the initial state; heuristics::ActionCosts), as BestFirstResult::cost. Throws std::invalid_argument when an
/// action is not applicable in its state.
[[nodiscard]] std::string format_plan(const Task& task, std::span<const Action> plan,
                                      const std::optional<State>& start = std::nullopt);

/// The actions of IPC plan text, replayed from `start` (default: the initial state). Blank lines and lines starting
/// with ';' are skipped; names are compared without regard to case. A line names the PDDL action's own parameters
/// or all of the schema's parameters; of the applicable actions that match it the first in canonical order is taken
/// (several match when normalization added parameters or split the action into schemas of one name). Throws
/// std::invalid_argument, naming the line, when no applicable action matches.
[[nodiscard]] std::vector<Action> parse_plan(const Task& task, std::string_view text,
                                             const std::optional<State>& start = std::nullopt);
}  // namespace mymyr::search
