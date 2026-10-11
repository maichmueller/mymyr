#pragma once

// Options of the relational encodings. They decide which relations an EncodingSchema (schema.hpp) lists and how the
// encoded arrays are laid out; every option that changes the encoded output is part of the schema fingerprint.

#include "mymyr/core/types.hpp"

#include <initializer_list>
#include <optional>
#include <string_view>
#include <vector>

namespace mymyr::encoding
{
/// A small set of enumerators of E (at most 32 of them), ordered by enumerator value when iterated.
template<class E>
class EnumSet
{
public:
    constexpr EnumSet() noexcept = default;
    constexpr EnumSet(std::initializer_list<E> members) noexcept
    {
        for (E member : members)
            insert(member);
    }

    constexpr void insert(E member) noexcept { m_bits |= bit(member); }
    constexpr void erase(E member) noexcept { m_bits &= ~bit(member); }
    [[nodiscard]] constexpr bool contains(E member) const noexcept { return (m_bits & bit(member)) != 0; }
    [[nodiscard]] constexpr bool empty() const noexcept { return m_bits == 0; }
    /// The members as a bit mask (bit i: the enumerator with value i).
    [[nodiscard]] constexpr u32 bits() const noexcept { return m_bits; }
    /// The members in enumerator order.
    [[nodiscard]] std::vector<E> members() const
    {
        std::vector<E> out;
        for (u32 i = 0; i < 32; ++i)
            if ((m_bits >> i) & 1u)
                out.push_back(static_cast<E>(i));
        return out;
    }
    [[nodiscard]] friend constexpr bool operator==(EnumSet, EnumSet) noexcept = default;

private:
    [[nodiscard]] static constexpr u32 bit(E member) noexcept { return u32{1} << static_cast<u32>(member); }
    u32 m_bits = 0;
};

/// The variants of a goal relation. Plain lists the goal literals themselves; Satisfied / Unsatisfied list the literals
/// that hold / do not hold in the encoded state (without the goal target slot). AddedSatisfied / AddedUnsatisfied list
/// the goal literals a horizon node newly satisfies / newly falsifies relative to its root (delta horizons only).
enum class GoalDerivation : u8
{
    Plain,
    Satisfied,
    Unsatisfied,
    AddedSatisfied,
    AddedUnsatisfied,
};

/// What becomes an entity of its own that tuples can refer to (a "target"). A Goal target is one entity per goal
/// literal, a Subgoal target one per subgoal literal, an Action target one per ground action, a History target one per
/// history literal.
enum class TargetSource : u8
{
    Goal,
    Subgoal,
    Action,
    History,
};

using GoalDerivations = EnumSet<GoalDerivation>;
using TargetSources = EnumSet<TargetSource>;

/// Order of the tuples of a batch: Graph groups them by graph and then by relation, Relation by relation and then by
/// graph (all tuples of a relation are contiguous across the graphs).
enum class Layout : u8
{
    Graph,
    Relation,
};

/// The integer type of every index, count and offset array.
enum class IndexType : u8
{
    I32,
    I64,
};

/// Which states a horizon encoding relates. Full encodes the complete state of every node, Delta the atoms a node
/// adds to / deletes from its root, Action the root's state plus the actions that lead to the nodes.
enum class HorizonMode : u8
{
    None,  // no horizon: single states
    Full,
    Delta,
    Action,
};

/// How the root state of a horizon takes part: Include encodes it with a state slot and makes it a target candidate,
/// EncodeOnly encodes it with a state slot but not as a candidate, Exclude encodes its relations without a state slot.
enum class RootPolicy : u8
{
    Include,
    Exclude,
    EncodeOnly,
};

struct HorizonOptions
{
    HorizonMode mode = HorizonMode::None;
    /// Include without a horizon mode: any other policy needs one.
    RootPolicy root = RootPolicy::Include;
    /// Adds the relation `_parent_` (state slot, state slot) between a node and its parent. Needs a horizon mode.
    bool parent_relation = false;

    [[nodiscard]] friend constexpr bool operator==(const HorizonOptions&, const HorizonOptions&) noexcept = default;
};

/// Options of the flat relational encoding and its horizon variants.
struct FlatOptions
{
    /// Goal relation variants per predicate and sign. The default lists the goal literals and which of them hold.
    GoalDerivations goal_derivations{GoalDerivation::Plain, GoalDerivation::Satisfied};
    /// Subgoal layers beyond the goal: relations of level 1 .. max_goal_level, named `[sg]`, `[ssg]`, `[sssg]`.
    /// At most 3.
    u32 max_goal_level = 0;
    /// Relations of the static predicates (the types included) in the state lane.
    bool include_static = true;
    /// Relations of the derived predicates in the state lane.
    bool include_derived = true;
    /// Relations of the nullary predicates. A nullary atom is a tuple over the graph's auxiliary entity, so that the
    /// fact has a node to attach to.
    bool nullary = true;
    /// Relations of the equality predicate `=`.
    bool equality = false;
    /// A predicate slot in the tuples of the predicate relations: a graph gets an entity for each predicate that has a
    /// tuple in it (and none for the others), listed in relation order.
    bool predicate_virtual_nodes = false;
    /// One relation per action schema.
    bool include_actions = true;
    /// Entities for the goal literals, subgoal literals, ground actions and history literals; the tuples that mention
    /// them get a target slot. Goal and subgoal targets need the plain goal derivation, action targets include_actions.
    TargetSources targets{};
    Layout layout = Layout::Graph;
    IndexType index = IndexType::I32;
    HorizonOptions horizon{};

    [[nodiscard]] friend bool operator==(const FlatOptions&, const FlatOptions&) noexcept = default;
};

/// Checks the options and returns them in normal form. Throws std::invalid_argument ("mymyr: encoding: ...") for
/// - max_goal_level above 3, or subgoal targets without subgoal levels;
/// - action targets without action relations (include_actions);
/// - goal or subgoal targets without the plain goal derivation: only the plain goal relations carry the target slot;
/// - added goal derivations without the delta horizon mode;
/// - targets together with a horizon mode (the state nodes are the targets of a horizon);
/// - the action horizon mode without action relations;
/// - a parent relation, or a root policy other than Include, without a horizon mode.
[[nodiscard]] FlatOptions normalized(const FlatOptions& options);

/// The text of an enumerator ("plain", "satisfied", "unsatisfied", "added_satisfied", "added_unsatisfied"; "goal",
/// "subgoal", "action", "history"; "graph", "relation"; "int32", "int64"; "none", "full", "delta", "action";
/// "include", "exclude", "encode_only"), and its inverse: nullopt for text that names no enumerator.
[[nodiscard]] std::string_view to_string(GoalDerivation value) noexcept;
[[nodiscard]] std::string_view to_string(TargetSource value) noexcept;
[[nodiscard]] std::string_view to_string(Layout value) noexcept;
[[nodiscard]] std::string_view to_string(IndexType value) noexcept;
[[nodiscard]] std::string_view to_string(HorizonMode value) noexcept;
[[nodiscard]] std::string_view to_string(RootPolicy value) noexcept;
[[nodiscard]] std::optional<GoalDerivation> parse_goal_derivation(std::string_view text) noexcept;
[[nodiscard]] std::optional<TargetSource> parse_target_source(std::string_view text) noexcept;
[[nodiscard]] std::optional<Layout> parse_layout(std::string_view text) noexcept;
[[nodiscard]] std::optional<IndexType> parse_index_type(std::string_view text) noexcept;
[[nodiscard]] std::optional<HorizonMode> parse_horizon_mode(std::string_view text) noexcept;
[[nodiscard]] std::optional<RootPolicy> parse_root_policy(std::string_view text) noexcept;
}
