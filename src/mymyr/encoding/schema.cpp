#include "mymyr/encoding/schema.hpp"

#include "mymyr/core/hash.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void invalid(const std::string& why)
{
    throw std::invalid_argument("mymyr: encoding: " + why);
}

/// The relation name of an action schema: its name, with the arity behind a `#` when the name has several arities.
std::string action_relation_name(const DomainSignature::ActionSchema& action, bool split)
{
    return split ? action.name + "#" + std::to_string(action.arity) : action.name;
}

/// The predicates every domain has for its types; they have no goal relations.
bool is_top_type(std::string_view name) { return name == "object" || name == "number"; }

constexpr std::array<std::string_view, 4> k_level_suffix = {"[g]", "[sg]", "[ssg]", "[sssg]"};

std::string_view derivation_suffix(GoalDerivation derivation)
{
    switch (derivation)
    {
    case GoalDerivation::Plain: return "";
    case GoalDerivation::Satisfied: return "[sat]";
    case GoalDerivation::Unsatisfied: return "[unsat]";
    case GoalDerivation::AddedSatisfied: return "[sat+]";
    case GoalDerivation::AddedUnsatisfied: return "[sat-]";
    }
    return "";
}

struct Pending
{
    Relation relation;
    std::vector<SlotRole> roles;
};

/// The catalog under construction: relations are added in any order and sorted by name in the end.
class Catalog
{
public:
    Catalog(const DomainSignature& domain, const FlatOptions& options) : m_domain(domain), m_options(options) {}

    void build();
    std::vector<Pending> take() { return std::move(m_pending); }

private:
    struct Key
    {
        i8 sign = 0;
        std::optional<u8> level;
        std::optional<GoalDerivation> derivation;
        std::string_view modifier;
        bool anchored = false;
    };
    [[nodiscard]] static Key key(i8 sign = 0, std::optional<u8> level = {}, std::optional<GoalDerivation> derivation = {},
                                 std::string_view modifier = {}, bool anchored = false)
    {
        return Key{sign, level, derivation, modifier, anchored};
    }

    /// Adds a predicate relation: `lead` roles, the predicate slot with virtual nodes, then the arguments.
    void add_predicate(u32 predicate, Lane lane, const Key& key, std::initializer_list<SlotRole> lead);
    void add_action(u32 schema, bool split, std::initializer_list<SlotRole> lead);
    void add(Pending&& pending);

    [[nodiscard]] static Lane state_lane(PredicateKind kind)
    {
        return kind == PredicateKind::Static ? Lane::Static : kind == PredicateKind::Derived ? Lane::Derived : Lane::Fluent;
    }

    void build_flat(u32 predicate, bool state_relation, bool top);
    void build_horizon(u32 predicate, bool state_relation, bool top);

    const DomainSignature& m_domain;
    const FlatOptions& m_options;
    std::vector<Pending> m_pending;
};

void Catalog::add(Pending&& pending)
{
    pending.relation.arity = static_cast<u32>(pending.roles.size());
    m_pending.push_back(std::move(pending));
}

void Catalog::add_predicate(u32 predicate, Lane lane, const Key& key, std::initializer_list<SlotRole> lead)
{
    const DomainSignature::Predicate& source = m_domain.predicates[predicate];
    Pending pending;
    Relation& relation = pending.relation;
    relation.lane = lane;
    relation.sign = key.sign;
    relation.level = key.level;
    relation.derivation = key.derivation;
    relation.state_anchored = key.anchored;
    relation.predicate = static_cast<i32>(predicate);
    relation.logical_arity = source.arity;
    relation.name.reserve(source.name.size() + 16);
    if (key.sign != 0)
        relation.name += key.sign > 0 ? "[+]" : "[-]";
    relation.name += source.name;
    relation.name += key.modifier;
    if (key.level)
        relation.name += k_level_suffix[*key.level];
    if (key.derivation)
        relation.name += derivation_suffix(*key.derivation);
    if (key.anchored)
        relation.name += "[state]";
    pending.roles.assign(lead.begin(), lead.end());
    if (m_options.predicate_virtual_nodes)
        pending.roles.push_back(SlotRole::Predicate);
    if (source.arity == 0)
        pending.roles.push_back(SlotRole::Auxiliary);
    else
        pending.roles.insert(pending.roles.end(), source.arity, SlotRole::Argument);
    add(std::move(pending));
}

void Catalog::add_action(u32 schema, bool split, std::initializer_list<SlotRole> lead)
{
    const DomainSignature::ActionSchema& source = m_domain.schemas[schema];
    Pending pending;
    pending.relation.name = action_relation_name(source, split);
    pending.relation.lane = Lane::Action;
    pending.relation.schema = static_cast<i32>(schema);
    pending.relation.logical_arity = source.arity;
    pending.roles.assign(lead.begin(), lead.end());
    pending.roles.insert(pending.roles.end(), source.arity, SlotRole::Argument);
    add(std::move(pending));
}

void Catalog::build_flat(u32 predicate, bool state_relation, bool top)
{
    const DomainSignature::Predicate& source = m_domain.predicates[predicate];
    const FlatOptions& o = m_options;
    if (state_relation)
        add_predicate(predicate, state_lane(source.kind), key(), {});
    if (!top)
        for (u32 level = 0; level <= o.max_goal_level; ++level)
        {
            const bool target = level == 0 ? o.targets.contains(TargetSource::Goal)
                                           : o.targets.contains(TargetSource::Subgoal);
            const SlotRole target_role = level == 0 ? SlotRole::GoalTarget : SlotRole::SubgoalTarget;
            for (i8 sign : {i8{1}, i8{-1}})
                for (GoalDerivation derivation : o.goal_derivations.members())
                {
                    Key goal = key(sign, static_cast<u8>(level));
                    if (derivation == GoalDerivation::Plain)
                    {
                        if (target)
                            add_predicate(predicate, Lane::Goal, goal, {target_role});
                        else
                            add_predicate(predicate, Lane::Goal, goal, {});
                    }
                    else
                    {
                        goal.derivation = derivation;
                        add_predicate(predicate, Lane::GoalDerivation, goal, {});
                    }
                }
        }
    for (i8 sign : {i8{1}, i8{-1}})
    {
        const Key hist = key(sign, {}, {}, "[hist]");
        if (o.targets.contains(TargetSource::History))
            add_predicate(predicate, Lane::History, hist, {SlotRole::HistoryTarget, SlotRole::History});
        else
            add_predicate(predicate, Lane::History, hist, {SlotRole::History});
    }
}

void Catalog::build_horizon(u32 predicate, bool state_relation, bool top)
{
    const DomainSignature::Predicate& source = m_domain.predicates[predicate];
    const FlatOptions& o = m_options;
    const bool delta = o.horizon.mode == HorizonMode::Delta;
    const bool root_slot = o.horizon.root != RootPolicy::Exclude;
    const bool split = o.horizon.mode == HorizonMode::Full && !root_slot;
    // The atoms a delta node adds and deletes only exist for predicates that change.
    const bool changing = source.kind != PredicateKind::Static;
    // The relations of the root carry a state slot unless the root is excluded; those of a candidate node always do.
    auto add_root = [&](Lane lane, const Key& key)
    {
        if (root_slot)
            add_predicate(predicate, lane, key, {SlotRole::State});
        else
            add_predicate(predicate, lane, key, {});
    };
    auto add_candidate = [&](Lane lane, const Key& key) { add_predicate(predicate, lane, key, {SlotRole::State}); };

    if (state_relation)
    {
        add_root(state_lane(source.kind), key());
        if (split)
            add_candidate(state_lane(source.kind), key(0, {}, {}, {}, true));
    }
    if (top)
        return;
    for (i8 sign : {i8{1}, i8{-1}})
        for (u32 level = 0; level <= o.max_goal_level; ++level)
        {
            Key goal = key(sign, static_cast<u8>(level));
            if (o.goal_derivations.contains(GoalDerivation::Plain))
                add_root(Lane::Goal, goal);
            for (GoalDerivation derivation : {GoalDerivation::Satisfied, GoalDerivation::Unsatisfied})
            {
                if (!o.goal_derivations.contains(derivation))
                    continue;
                goal.derivation = derivation;
                add_root(Lane::GoalSatisfaction, goal);
                if (split)
                {
                    Key anchored = goal;
                    anchored.anchored = true;
                    add_candidate(Lane::GoalSatisfaction, anchored);
                }
            }
            for (GoalDerivation derivation : {GoalDerivation::AddedSatisfied, GoalDerivation::AddedUnsatisfied})
                if (o.goal_derivations.contains(derivation) && changing)
                {
                    goal.derivation = derivation;
                    add_candidate(Lane::GoalSatisfaction, goal);
                }
        }
    if (!delta)
        return;
    // A delta horizon also lists the literals themselves: the atoms a node adds (+) and deletes (-) relative to its
    // root, and the satisfaction of those literals.
    for (i8 sign : {i8{1}, i8{-1}})
    {
        if (changing)
            add_candidate(state_lane(source.kind), key(sign));
        for (GoalDerivation derivation : o.goal_derivations.members())
        {
            if (derivation == GoalDerivation::Plain)
                continue;
            const bool added = derivation == GoalDerivation::AddedSatisfied || derivation == GoalDerivation::AddedUnsatisfied;
            if (added && !changing)
                continue;
            const Key literal = key(sign, {}, derivation);
            if (added)
                add_candidate(Lane::GoalSatisfaction, literal);
            else
                add_root(Lane::GoalSatisfaction, literal);
        }
    }
}

void Catalog::build()
{
    const FlatOptions& o = m_options;
    const bool horizon = o.horizon.mode != HorizonMode::None;
    for (u32 predicate = 0; predicate < m_domain.predicates.size(); ++predicate)
    {
        const DomainSignature::Predicate& source = m_domain.predicates[predicate];
        if ((source.arity == 0 && !o.nullary) || (source.name == "=" && !o.equality))
            continue;
        const bool state_relation = !(source.kind == PredicateKind::Static && !o.include_static) &&
                                    !(source.kind == PredicateKind::Derived && !o.include_derived);
        const bool top = is_top_type(source.name);
        if (horizon)
            build_horizon(predicate, state_relation, top);
        else
            build_flat(predicate, state_relation, top);
    }
    if (o.include_actions)
    {
        // Schemas of one name and arity (the front end splits an action with a disjunctive precondition) share a
        // relation; a name with several arities gets one relation per arity, named `name#arity`.
        std::unordered_map<std::string_view, std::set<u32>> arities;
        for (const DomainSignature::ActionSchema& source : m_domain.schemas)
            arities[source.name].insert(source.arity);
        std::set<std::string> seen;
        for (u32 schema = 0; schema < m_domain.schemas.size(); ++schema)
        {
            const DomainSignature::ActionSchema& source = m_domain.schemas[schema];
            if (!seen.insert(action_relation_name(source, arities[source.name].size() > 1)).second)
                continue;
            if (horizon)
                add_action(schema, arities[source.name].size() > 1, {SlotRole::State});
            else
                add_action(schema, arities[source.name].size() > 1, {SlotRole::Action});
        }
    }
    if (o.horizon.parent_relation)
    {
        Pending parent;
        parent.relation.name = "_parent_";
        parent.relation.lane = Lane::Parent;
        parent.roles = {SlotRole::State, SlotRole::State};
        add(std::move(parent));
    }
}

/// FNV-1a over a canonical byte stream: integers as 8 little-endian bytes, strings with their length in front.
class Hasher
{
public:
    void bytes(const void* data, usize size)
    {
        const auto* p = static_cast<const unsigned char*>(data);
        for (usize i = 0; i < size; ++i)
        {
            m_state ^= p[i];
            m_state *= 0x100000001b3ULL;
        }
    }
    void integer(u64 value)
    {
        unsigned char b[8];
        for (u32 i = 0; i < 8; ++i)
            b[i] = static_cast<unsigned char>(value >> (8 * i));
        bytes(b, 8);
    }
    void text(std::string_view value)
    {
        integer(value.size());
        bytes(value.data(), value.size());
    }
    [[nodiscard]] u64 value() const { return hash::mix64(m_state); }

private:
    u64 m_state = 0xcbf29ce484222325ULL;
};
}

DomainSignature DomainSignature::of(const rl::TaskTable& table)
{
    for (const auto& instance : table.instances())
        if (instance.task->data().has_numerics() || !instance.task->data().constraints.empty())
            invalid("numeric functions and constraints are not supported");
    DomainSignature domain;
    const auto& names = table.predicate_names();
    for (u32 p = 0; p < table.num_domain_predicates(); ++p)
    {
        const i32 code = table.predicate_kinds()[p];
        Predicate predicate;
        predicate.name = names[p];
        predicate.arity = table.predicate_arities()[p];
        predicate.kind = code == rl::k_pred_static ? PredicateKind::Static
                         : code == rl::k_pred_fluent ? PredicateKind::Fluent
                                                     : PredicateKind::Derived;
        domain.predicates.push_back(std::move(predicate));
    }
    for (u32 s = 0; s < table.num_schemas(); ++s)
        domain.schemas.push_back({table.schema_names()[s], table.schema_arities()[s]});
    return domain;
}

std::string_view to_string(Lane lane) noexcept
{
    switch (lane)
    {
    case Lane::Fluent: return "fluent";
    case Lane::Static: return "static";
    case Lane::Derived: return "derived";
    case Lane::Goal: return "goal";
    case Lane::GoalDerivation: return "goal_derivation";
    case Lane::GoalSatisfaction: return "goal_satisfaction";
    case Lane::History: return "history";
    case Lane::Action: return "action";
    case Lane::Parent: return "parent";
    }
    return "";
}

std::shared_ptr<const EncodingSchema> EncodingSchema::create(const rl::TaskTable& table, const FlatOptions& options)
{
    return create(DomainSignature::of(table), options);
}

std::shared_ptr<const EncodingSchema> EncodingSchema::create(const DomainSignature& domain, const FlatOptions& options)
{
    std::shared_ptr<EncodingSchema> schema(new EncodingSchema);
    schema->m_options = normalized(options);
    {
        std::unordered_set<std::string_view> seen;
        for (const auto& predicate : domain.predicates)
            if (!seen.insert(predicate.name).second)
                invalid("the domain has two predicates named '" + predicate.name + "'");
    }

    Catalog catalog(domain, schema->m_options);
    catalog.build();
    std::vector<Pending> pending = catalog.take();
    std::ranges::sort(pending, {}, [](const Pending& p) -> const std::string& { return p.relation.name; });
    for (usize i = 1; i < pending.size(); ++i)
        if (pending[i].relation.name == pending[i - 1].relation.name)
            invalid("two relations are named '" + pending[i].relation.name + "'");

    schema->m_role_offsets.reserve(pending.size() + 1);
    schema->m_role_offsets.push_back(0);
    for (Pending& entry : pending)
    {
        schema->m_roles.insert(schema->m_roles.end(), entry.roles.begin(), entry.roles.end());
        schema->m_role_offsets.push_back(static_cast<u32>(schema->m_roles.size()));
        schema->m_relations.push_back(std::move(entry.relation));
    }
    if (schema->m_options.include_actions)
        for (const auto& action : domain.schemas)
        {
            const bool split = std::ranges::any_of(domain.schemas, [&](const auto& other)
                                                   { return other.name == action.name && other.arity != action.arity; });
            schema->m_schema_relation.push_back(static_cast<i32>(*schema->find(action_relation_name(action, split))));
        }
    const FlatOptions& o = schema->m_options;
    schema->m_implicit_entities = o.entity_arrays == EntityArrays::Auto && o.targets.empty() && !o.predicate_virtual_nodes &&
                                  !o.include_actions && o.horizon.mode == HorizonMode::None;
    Hasher hasher;
    hasher.text("mymyr.encoding.schema");
    hasher.integer(k_schema_version);
    hasher.integer(static_cast<u64>(o.layout));
    hasher.integer(static_cast<u64>(o.index));
    hasher.integer(o.emit_counts ? 1 : 0);
    hasher.integer(schema->m_implicit_entities ? 1 : 0);
    hasher.integer(o.targets.bits());
    hasher.integer(static_cast<u64>(o.horizon.mode));
    hasher.integer(o.horizon.mode == HorizonMode::None ? 0 : static_cast<u64>(o.horizon.root));
    hasher.integer(schema->m_relations.size());
    for (u32 id = 0; id < schema->m_relations.size(); ++id)
    {
        const Relation& relation = schema->m_relations[id];
        hasher.text(relation.name);
        hasher.integer(static_cast<u64>(relation.lane));
        hasher.integer(relation.arity);
        for (SlotRole role : schema->slot_roles(id))
            hasher.integer(static_cast<u64>(role));
    }
    schema->m_fingerprint = hasher.value();
    return schema;
}

std::optional<u32> EncodingSchema::find(std::string_view name) const
{
    const auto it = std::ranges::lower_bound(m_relations, name, {}, [](const Relation& r) -> std::string_view { return r.name; });
    if (it == m_relations.end() || it->name != name)
        return std::nullopt;
    return static_cast<u32>(it - m_relations.begin());
}

std::span<const SlotRole> EncodingSchema::slot_roles(u32 id) const
{
    if (id >= m_relations.size())
        throw std::out_of_range("mymyr: encoding: relation id out of range");
    return {m_roles.data() + m_role_offsets[id], m_role_offsets[id + 1] - m_role_offsets[id]};
}
}
