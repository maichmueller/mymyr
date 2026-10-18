#include "mymyr/encoding/options.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void invalid(const std::string& why)
{
    throw std::invalid_argument("mymyr: encoding: " + why);
}

template<class E, usize N>
std::optional<E> parse(const std::array<std::string_view, N>& names, std::string_view text) noexcept
{
    for (usize i = 0; i < N; ++i)
        if (names[i] == text)
            return static_cast<E>(i);
    return std::nullopt;
}

constexpr std::array<std::string_view, 5> k_derivations = {"plain", "satisfied", "unsatisfied", "added_satisfied",
                                                           "added_unsatisfied"};
constexpr std::array<std::string_view, 4> k_sources = {"goal", "subgoal", "action", "history"};
constexpr std::array<std::string_view, 2> k_layouts = {"graph", "relation"};
constexpr std::array<std::string_view, 2> k_entity_arrays = {"auto", "always"};
constexpr std::array<std::string_view, 2> k_index_types = {"int32", "int64"};
constexpr std::array<std::string_view, 4> k_modes = {"none", "full", "delta", "action"};
constexpr std::array<std::string_view, 3> k_roots = {"include", "exclude", "encode_only"};
}

std::string_view to_string(GoalDerivation value) noexcept { return k_derivations[static_cast<usize>(value)]; }
std::string_view to_string(TargetSource value) noexcept { return k_sources[static_cast<usize>(value)]; }
std::string_view to_string(Layout value) noexcept { return k_layouts[static_cast<usize>(value)]; }
std::string_view to_string(EntityArrays value) noexcept { return k_entity_arrays[static_cast<usize>(value)]; }
std::string_view to_string(IndexType value) noexcept { return k_index_types[static_cast<usize>(value)]; }
std::string_view to_string(HorizonMode value) noexcept { return k_modes[static_cast<usize>(value)]; }
std::string_view to_string(RootPolicy value) noexcept { return k_roots[static_cast<usize>(value)]; }

std::optional<GoalDerivation> parse_goal_derivation(std::string_view text) noexcept
{
    return parse<GoalDerivation>(k_derivations, text);
}
std::optional<TargetSource> parse_target_source(std::string_view text) noexcept
{
    return parse<TargetSource>(k_sources, text);
}
std::optional<Layout> parse_layout(std::string_view text) noexcept { return parse<Layout>(k_layouts, text); }
std::optional<EntityArrays> parse_entity_arrays(std::string_view text) noexcept
{
    return parse<EntityArrays>(k_entity_arrays, text);
}
std::optional<IndexType> parse_index_type(std::string_view text) noexcept
{
    return parse<IndexType>(k_index_types, text);
}
std::optional<HorizonMode> parse_horizon_mode(std::string_view text) noexcept
{
    return parse<HorizonMode>(k_modes, text);
}
std::optional<RootPolicy> parse_root_policy(std::string_view text) noexcept
{
    return parse<RootPolicy>(k_roots, text);
}

FlatOptions normalized(const FlatOptions& options)
{
    FlatOptions out = options;
    const bool horizon = out.horizon.mode != HorizonMode::None;
    if (out.max_goal_level > 3)
        invalid("max_goal_level must be in [0, 3], got " + std::to_string(out.max_goal_level));
    if (out.targets.contains(TargetSource::Subgoal) && out.max_goal_level == 0)
        invalid("subgoal targets need max_goal_level >= 1");
    const bool added = out.goal_derivations.contains(GoalDerivation::AddedSatisfied) ||
                       out.goal_derivations.contains(GoalDerivation::AddedUnsatisfied);
    if (added && out.horizon.mode != HorizonMode::Delta)
        invalid("the goal derivations added_satisfied and added_unsatisfied need the delta horizon mode");
    if (out.targets.contains(TargetSource::Action) && !out.include_actions)
        invalid("action targets need the action relations (include_actions)");
    if ((out.targets.contains(TargetSource::Goal) || out.targets.contains(TargetSource::Subgoal)) &&
        !out.goal_derivations.contains(GoalDerivation::Plain))
        invalid("goal and subgoal targets need the plain goal derivation: only its relations carry the target slot");
    if (!out.history && out.targets.contains(TargetSource::History))
        invalid("history targets need history");
    if (!out.history && out.history_max_steps != 0)
        invalid("history_max_steps needs history");
    if (out.history && horizon)
        invalid("history cannot be combined with a horizon mode");
    if (horizon && !out.targets.empty())
        invalid("target sources cannot be combined with a horizon mode: the state nodes of a horizon are its targets");
    if (out.horizon.mode == HorizonMode::Action && !out.include_actions)
        invalid("the action horizon mode needs the action relations (include_actions)");
    if (!horizon)
    {
        if (out.horizon.parent_relation)
            invalid("parent_relation needs a horizon mode");
        if (out.horizon.sibling_relation)
            invalid("sibling_relation needs a horizon mode");
        if (out.horizon.cousin_relation)
            invalid("cousin_relation needs a horizon mode");
        if (out.horizon.root != RootPolicy::Include)
            invalid("root_policy '" + std::string(to_string(out.horizon.root)) + "' needs a horizon mode");
    }
    return out;
}
}
