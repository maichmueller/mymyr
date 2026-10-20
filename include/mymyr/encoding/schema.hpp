#pragma once

// The relation catalog of an encoding: which relations a batch of encoded states holds, in which order, with which
// arities and slot roles. A schema is a function of the domain (its predicates and action schemas) and the options
// alone, never of the instances of a task table, so a table of training instances and one of evaluation instances over
// the same domain give the same relation ids to the same relations.
//
// Relations are named in the convention `[+]on[g][sat]`: a sign prefix `[+]` / `[-]` for goal and history relations,
// the predicate or action schema name, a modifier `[hist]` for history relations, a goal level `[g]` (the goal) or
// `[sg]`, `[ssg]`, `[sssg]` (subgoal layers), a derivation `[sat]` / `[unsat]` / `[sat+]` / `[sat-]`, and `[state]` for
// the state-anchored variant of a full horizon without its root. The catalog is sorted by name, bytewise, which makes
// the relation ids independent of the declaration order in the domain. State relations carry the plain predicate name.

#include "mymyr/encoding/options.hpp"
#include "mymyr/rl/task_table.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::encoding
{
/// Version of the catalog rules; part of the fingerprint.
inline constexpr u32 k_schema_version = 1;

enum class PredicateKind : u8
{
    Static,
    Fluent,
    Derived,
};

/// What a schema is built from: the predicates and action schemas of a domain, in declaration order.
struct DomainSignature
{
    struct Predicate
    {
        std::string name;
        u32 arity = 0;
        PredicateKind kind = PredicateKind::Fluent;
    };
    struct ActionSchema
    {
        std::string name;
        /// The parameters of the normalized schema: the written ones and those normalization adds for the variables of
        /// an existential condition. An action relation holds all of them.
        u32 arity = 0;
    };
    std::vector<Predicate> predicates;
    std::vector<ActionSchema> schemas;
    /// The constants of the domain, in declaration order: they are the first objects of every instance, so the
    /// constant with index c is the object with id c.
    std::vector<std::string> constants;

    /// The domain of a task table: its domain predicates (the front end's helper predicates for quantified conditions
    /// among them; the predicates only a problem's axioms define are not part of it) and its action schemas. Throws
    /// std::invalid_argument for a table with numeric functions or constraints.
    [[nodiscard]] static DomainSignature of(const rl::TaskTable& table);
};

/// The part of a batch a relation belongs to.
enum class Lane : u8
{
    Fluent,           // the atoms of a fluent predicate in the state
    Static,           // the atoms of a static predicate (the types included)
    Derived,          // the atoms of a derived predicate in the state
    Goal,             // goal literals (and subgoal literals of a level above 0)
    GoalDerivation,   // goal literals that hold / do not hold in the state
    GoalSatisfaction, // the same for the nodes of a horizon, and the goals a delta node newly satisfies / falsifies
    History,          // literals of earlier steps
    Action,           // the ground actions of one action schema
    Parent,           // the parent relation of a horizon
    Sibling,          // the sibling relation of a horizon
    Cousin,           // the cousin relation of a horizon
};

[[nodiscard]] std::string_view to_string(Lane lane) noexcept;

/// What one position of a tuple holds. Argument positions come last, behind the positions of the other roles.
enum class SlotRole : u8
{
    Argument,
    Predicate,      // the entity of the tuple's predicate (predicate virtual nodes)
    State,          // the state node of a horizon
    Action,         // the entity of the ground action
    GoalTarget,
    SubgoalTarget,
    HistoryTarget,
    History,        // the entity of the history step
    Auxiliary,      // the graph's auxiliary entity: the only argument of a nullary relation
};

/// The roles of the entities of a graph. Objects come first.
enum class EntityRole : u8
{
    Object,
    PredicateVirtual,
    Goal,
    Subgoal,
    Action,
    History,
    HistoryTarget,
    State,
    Auxiliary,  // one per graph; the entity that nullary atoms are tuples over
};

/// The name of the auxiliary entity of a graph.
inline constexpr std::string_view k_auxiliary_entity_name = "auxiliary";

/// The codes of the target groups: the group of an entity that can be a target. They are fixed, not numbered by the
/// groups that an encoder has.
enum class TargetGroup : u8
{
    Goal,
    Subgoal,
    Action,
    History,
    State,
};

inline constexpr std::array<std::string_view, 9> k_slot_role_names = {
    "argument_slot",    "predicate_slot",      "state_slot",          "action_slot",   "goal_target_slot",
    "subgoal_target_slot", "history_target_slot", "history_slot", "auxiliary_slot"};
inline constexpr std::array<std::string_view, 9> k_entity_role_names = {
    "object", "predicate_virtual", "goal", "subgoal", "action", "history", "history_target", "state",
    k_auxiliary_entity_name};
inline constexpr std::array<std::string_view, 5> k_target_group_names = {"goal", "subgoal", "action", "history",
                                                                          "state"};

[[nodiscard]] inline std::string_view to_string(SlotRole role) noexcept
{
    return k_slot_role_names[static_cast<usize>(role)];
}
[[nodiscard]] inline std::string_view to_string(EntityRole role) noexcept
{
    return k_entity_role_names[static_cast<usize>(role)];
}
[[nodiscard]] inline std::string_view to_string(TargetGroup group) noexcept
{
    return k_target_group_names[static_cast<usize>(group)];
}

/// One relation of the catalog.
struct Relation
{
    std::string name;
    Lane lane = Lane::Fluent;
    i8 sign = 0;                                // +1, -1, or 0 for a relation without a sign
    std::optional<u8> level;                    // goal level (0: the goal); none for other relations
    std::optional<GoalDerivation> derivation;   // none for plain goal relations and relations without a goal
    bool state_anchored = false;                // the `[state]` variant of a full horizon without its root
    i32 predicate = -1;                         // domain predicate id, or -1
    i32 schema = -1;                            // first action schema of this name and arity, or -1
    u32 arity = 0;                              // encoded arity: the slots of the other roles plus the arguments
                                                // (one auxiliary slot for a nullary predicate)
    u32 logical_arity = 0;                      // arguments: the arity of the predicate or action schema
};

class EncodingSchema
{
public:
    /// Builds the catalog of `domain` under `options` (checked and normalized, see normalized()). Throws
    /// std::invalid_argument ("mymyr: encoding: ...") for invalid options, a domain with two predicates of one name, or
    /// two relations that would get the same name. Action schemas may share a name: those of one name and arity share
    /// a relation.
    [[nodiscard]] static std::shared_ptr<const EncodingSchema> create(const DomainSignature& domain,
                                                                     const FlatOptions& options = {});
    [[nodiscard]] static std::shared_ptr<const EncodingSchema> create(const rl::TaskTable& table,
                                                                     const FlatOptions& options = {});

    /// The options in normal form.
    [[nodiscard]] const FlatOptions& options() const noexcept { return m_options; }

    /// The relations, sorted by name (bytewise); a relation's id is its position.
    [[nodiscard]] u32 num_relations() const noexcept { return static_cast<u32>(m_relations.size()); }
    [[nodiscard]] const std::vector<Relation>& relations() const noexcept { return m_relations; }
    [[nodiscard]] const Relation& relation(u32 id) const { return m_relations.at(id); }
    /// The id of the relation with this name.
    [[nodiscard]] std::optional<u32> find(std::string_view name) const;

    /// The slot roles of a relation, one per encoded position: its leading slots (state, action, targets, history),
    /// then the predicate slot, then the arguments (the auxiliary slot for a nullary predicate).
    [[nodiscard]] std::span<const SlotRole> slot_roles(u32 id) const;
    /// All slot roles, relation after relation, and the offset of each relation's roles (num_relations() + 1 entries).
    [[nodiscard]] std::span<const SlotRole> all_slot_roles() const noexcept { return m_roles; }
    [[nodiscard]] std::span<const u32> slot_role_offsets() const noexcept { return m_role_offsets; }

    /// The relation of each action schema of the domain (empty without action relations). The front end splits an
    /// action whose precondition has a disjunction into one schema per disjunct: schemas of one name and arity share a
    /// relation, and a name with several arities has one relation `name#arity` per arity.
    [[nodiscard]] std::span<const i32> schema_relations() const noexcept { return m_schema_relation; }

    /// The key of a relation's predicate for the predicate virtual nodes, a relation id: the predicate's plain state
    /// relation, or its first relation if the schema has none; -1 for a relation without a predicate. The entity_ref
    /// of a virtual node is its key.
    [[nodiscard]] i32 virtual_key(u32 relation) const { return m_virtual_key.at(relation); }

    /// The rank of a relation's predicate among the predicates that have relations, by ascending name of the domain
    /// predicate; -1 for a relation without a predicate. A graph numbers its virtual nodes in the order of these ranks.
    [[nodiscard]] i32 virtual_rank(u32 relation) const { return m_virtual_rank.at(relation); }

    /// The predicates that have relations (the distinct ranks).
    [[nodiscard]] u32 num_virtual_predicates() const noexcept { return m_num_virtual_predicates; }

    /// The names of all entity roles and slot roles, indexed by role value. The codes are fixed: the auxiliary role is
    /// listed whether or not the schema has nullary relations.
    [[nodiscard]] static std::span<const std::string_view> entity_role_names() noexcept { return k_entity_role_names; }
    [[nodiscard]] static std::span<const std::string_view> slot_role_names() noexcept { return k_slot_role_names; }

    /// Whether a batch of this schema omits `entity_role` and `entity_ref`: the options say `entity_arrays = auto` and
    /// configure nothing that makes an entity of another role than object or auxiliary: no target sources, no predicate
    /// virtual nodes, no action relations (their ground actions are entities), no history (its steps are entities) and
    /// no horizon mode. Every entity of every graph is then an object or the auxiliary entity. The omitted arrays are
    /// then implicit: the entity with local index j of graph g (entity id entity_offsets[g] + j) has role Object and
    /// ref j if j < num_objects[g], and otherwise role Auxiliary and ref -1 (it is the graph's last and only entity
    /// past the objects).
    [[nodiscard]] bool implicit_entity_arrays() const noexcept { return m_implicit_entities; }
    /// Whether a batch stores `relation_counts` (FlatOptions::emit_counts).
    [[nodiscard]] bool emits_counts() const noexcept { return m_options.emit_counts; }
    /// Whether a batch stores `goal_complete`: the schema has goal relations (of any level or derivation).
    [[nodiscard]] bool has_goal_complete() const noexcept { return m_goal_complete; }
    /// Whether a batch stores the target arrays `target_offsets`, `target_entity` and `target_group`: the options name
    /// at least one target source, or a horizon mode (its candidate state nodes are the targets).
    [[nodiscard]] bool has_targets() const noexcept
    {
        return !m_options.targets.empty() || m_options.horizon.mode != HorizonMode::None;
    }
    /// Whether the schema is that of a horizon: batches then hold state entities (see FlatHorizonEncoder).
    [[nodiscard]] bool has_horizon() const noexcept { return m_options.horizon.mode != HorizonMode::None; }

    /// Whether a batch stores the history arrays `history_offsets`, `history_entity` and `history_dt`
    /// (FlatOptions::history).
    [[nodiscard]] bool has_history() const noexcept { return m_options.history; }
    /// Whether the schema has subgoal relations: goal relations of a level above 0, including the ones of a horizon.
    [[nodiscard]] bool has_subgoals() const noexcept { return m_subgoals; }

    /// Whether a batch stores the relation arrays (relation_counts, relation_offsets, relation_arg_offsets and
    /// relation_args): FlatOptions::outputs has Relations.
    [[nodiscard]] bool emits_relations() const noexcept { return m_options.outputs.contains(Output::Relations); }
    /// Whether a batch stores the Simple graph (node_offsets, node_label, edge_offsets, edge_index and edge_label):
    /// FlatOptions::outputs has Simple.
    [[nodiscard]] bool emits_simple() const noexcept { return m_options.outputs.contains(Output::Simple); }
    /// The largest encoded arity of a relation, K: slot k of an atom's tuple labels its edge to the entity k, and the
    /// edge back has the label K + k. At least 1.
    [[nodiscard]] u32 simple_max_arity() const noexcept { return m_simple_max_arity; }
    /// The domain's constants a node label of its own is given with FlatOptions::constant_labels (otherwise 0).
    [[nodiscard]] u32 simple_constants() const noexcept { return m_simple_constants; }
    /// The number of node labels of the Simple graph: the 9 entity roles, then one per relation, then one per
    /// constant with constant_labels.
    [[nodiscard]] u32 simple_node_labels() const noexcept
    {
        return static_cast<u32>(k_entity_role_names.size()) + num_relations() + m_simple_constants;
    }
    /// The number of edge labels of the Simple graph, 2K.
    [[nodiscard]] u32 simple_edge_labels() const noexcept { return 2 * m_simple_max_arity; }
    /// The name of each node label: the entity role names, then the relation names, then the constant names. Label
    /// 9 + r is relation r; with constant_labels label 9 + R + c is the constant c.
    [[nodiscard]] std::vector<std::string> simple_label_names() const;
    /// The name of each edge label: "<k>" for the edge from the entity at position k of an atom to the atom, "<k>^-1"
    /// for the edge back (label K + k).
    [[nodiscard]] std::vector<std::string> simple_edge_label_names() const;

    /// A 64-bit hash of the catalog (names, lanes, arities, slot roles), the rule version, and the options that shape
    /// the encoded arrays beyond the catalog: the layout, the index type, whether relation_counts and the entity arrays
    /// are stored, the target sources, the history lane with its step limit, the horizon mode and root policy, and the
    /// outputs and constant labels when they differ from the default (relations only, no constant labels). It
    /// does not depend on the declaration order of the domain, on predicate or schema ids, or on the process or the
    /// build.
    [[nodiscard]] u64 fingerprint() const noexcept { return m_fingerprint; }

private:
    EncodingSchema() = default;

    FlatOptions m_options;
    std::vector<Relation> m_relations;
    std::vector<SlotRole> m_roles;
    std::vector<u32> m_role_offsets;
    std::vector<i32> m_schema_relation;
    std::vector<i32> m_virtual_key;
    std::vector<i32> m_virtual_rank;
    u32 m_num_virtual_predicates = 0;
    u32 m_simple_max_arity = 1;
    u32 m_simple_constants = 0;
    std::vector<std::string> m_constants;
    bool m_implicit_entities = false;
    bool m_goal_complete = false;
    bool m_subgoals = false;
    u64 m_fingerprint = 0;
};
}
