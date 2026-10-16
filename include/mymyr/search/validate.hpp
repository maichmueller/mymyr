#pragma once
// Plan validation: replays a plan on a task, step by step, as a plan validator (VAL) does.
//
//   const mymyr::search::PlanValidation v = mymyr::search::validate_plan_text(*task, text);
//   if (!v.valid)
//       std::cerr << v.message << "\n";   // "step 3 (stack a b): the precondition (clear b) does not hold"
//
// Each step is checked in the state the steps before it reached: the action must exist (its name, for plan text),
// take that many arguments, name objects of the task of its parameters' declared types, and be applicable (its
// precondition holds, static, fluent and derived literals and numeric constraints alike, derived atoms by the state's
// closure under the axioms, and its effects are defined). Its effects are then applied (conditional effects that fire,
// numeric effects). After the last step the goal must hold. The plan's cost is the metric value of the last state
// (heuristics::ActionCosts: the problem's total-cost or metric, else the number of steps), counted from the start
// state's metric value, as the searches report it. The first failing step ends the validation.
//
// Plan text is the IPC format (search/plan_file.hpp): one "(name o1 ... ok)" per line, blank lines and ';' comments
// skipped, names compared without regard to case. A line names the PDDL action's own parameters or all of the schema's
// parameters (normalization adds parameters for existential preconditions and may split an action into several
// schemas of one name); of the applicable actions that match it the first in canonical order is taken, as
// parse_plan does.

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
/// Why a plan is invalid.
enum class PlanError : u8
{
    None,            // the plan is valid
    Syntax,          // plan text: a line is not "(name o1 ... ok)"
    UnknownAction,   // no action of the task has this name (a plan of actions: the schema id is out of range)
    WrongArity,      // the step has a different number of arguments than the action's parameters
    UnknownObject,   // an argument names no object of the task
    WrongType,       // an argument is not of its parameter's declared type
    NotApplicable,   // the precondition does not hold in the step's state, or the effects are undefined there
    GoalNotReached,  // every step applies, but the goal does not hold after the last one
};

[[nodiscard]] const char* to_string(PlanError e) noexcept;

struct PlanValidation
{
    bool valid = false;
    PlanError error = PlanError::None;
    /// The failing step: its index in the plan (0-based); the plan's length when the goal is not reached or the plan
    /// is valid.
    u64 step = 0;
    /// Plan text: the line of the failing step (1-based); 0 when the goal is not reached, the plan is valid or the
    /// plan is a sequence of actions.
    u64 line = 0;
    std::string action;   // the failing step as text, e.g. "(stack a b)"; empty when the goal is not reached
    /// What does not hold: the precondition literal, numeric constraint or equality, the argument's type
    /// ("c - block"), or the goal literal or constraint, as PDDL text; empty when there is none to name.
    std::string condition;
    std::string message;  // the whole reason in one sentence; empty when valid
    double cost = 0;      // the metric value of `state` (the plan's cost when valid)
    std::vector<Action> plan;  // the steps that applied, with every schema parameter bound
    State state;               // the state the steps that applied reached (the goal state when valid)
};

/// Validates a plan of actions (as the searches return them) from `start` (default: the task's initial state).
/// Throws std::invalid_argument for a start state of another task (its numeric width differs).
[[nodiscard]] PlanValidation validate_plan(const Task& task, std::span<const Action> plan,
                                           const std::optional<State>& start = std::nullopt);

/// Validates IPC plan text (see above) from `start`.
[[nodiscard]] PlanValidation validate_plan_text(const Task& task, std::string_view text,
                                                const std::optional<State>& start = std::nullopt);
}  // namespace mymyr::search
