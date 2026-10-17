#include "mymyr/encoding/validate.hpp"

#include <bit>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("mymyr: encoding: " + what); }

/// An index array of either width.
struct Ints
{
    const void* data = nullptr;
    u64 size = 0;
    bool wide = false;
    [[nodiscard]] i64 operator[](u64 i) const
    {
        return wide ? static_cast<const i64*>(data)[i] : static_cast<i64>(static_cast<const i32*>(data)[i]);
    }
};

/// The entity role an argument position of this slot role must hold.
EntityRole entity_role_of(SlotRole role)
{
    switch (role)
    {
        case SlotRole::Argument: return EntityRole::Object;
        case SlotRole::Predicate: return EntityRole::PredicateVirtual;
        case SlotRole::State: return EntityRole::State;
        case SlotRole::Action: return EntityRole::Action;
        case SlotRole::GoalTarget: return EntityRole::Goal;
        case SlotRole::SubgoalTarget: return EntityRole::Subgoal;
        case SlotRole::HistoryTarget: return EntityRole::HistoryTarget;
        case SlotRole::History: return EntityRole::History;
        case SlotRole::Auxiliary: return EntityRole::Auxiliary;
    }
    return EntityRole::Object;
}

const rl::ArrayInfo* array(const rl::ArrayBundle& batch, const char* name, rl::DType dtype, const std::vector<i64>& shape)
{
    const rl::ArrayInfo* info = batch.find(name);
    if (!info)
        fail(std::string("the array ") + name + " is missing");
    if (info->dtype != dtype)
        fail(std::string("the array ") + name + " has the wrong dtype");
    if (info->shape != shape)
        fail(std::string("the array ") + name + " has the wrong shape");
    return info;
}

/// A one-dimensional array of at least `size` entries (a capacity bundle pads it) whose entries from `size` on are
/// zero.
const rl::ArrayInfo* padded_array(const rl::ArrayBundle& batch, const char* name, rl::DType dtype, u64 size, bool wide)
{
    const rl::ArrayInfo* info = batch.find(name);
    if (!info)
        fail(std::string("the array ") + name + " is missing");
    if (info->dtype != dtype)
        fail(std::string("the array ") + name + " has the wrong dtype");
    if (info->shape.size() != 1 || info->shape[0] < static_cast<i64>(size))
        fail(std::string("the array ") + name + " is shorter than the batch needs");
    const void* data = batch.data(*info);
    for (u64 i = size; i < info->elements(); ++i)
    {
        const bool nonzero = dtype == rl::DType::U8 ? static_cast<const u8*>(data)[i] != 0
                             : wide                 ? static_cast<const i64*>(data)[i] != 0
                                                    : static_cast<const i32*>(data)[i] != 0;
        if (nonzero)
            fail(std::string("the array ") + name + " is not zero past the batch");
    }
    return info;
}

void check_offsets(const Ints& offsets, u64 end, const char* name)
{
    if (offsets.size == 0 || offsets[0] != 0)
        fail(std::string(name) + " does not start at 0");
    for (u64 i = 1; i < offsets.size; ++i)
        if (offsets[i] < offsets[i - 1])
            fail(std::string(name) + " decreases at " + std::to_string(i));
    if (static_cast<u64>(offsets[offsets.size - 1]) != end)
        fail(std::string(name) + " does not end at the size of the array it indexes");
}
}  // namespace

void validate(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const i64 version = batch.scalar("version");
    if (version != 1)
        fail("unknown batch version " + std::to_string(version));
    const i64 graphs_scalar = batch.scalar("num_graphs");
    if (graphs_scalar < 0)
        fail("num_graphs is missing");
    const u64 G = static_cast<u64>(graphs_scalar);
    const u64 R = schema.num_relations();
    if (batch.scalar("num_relations") != static_cast<i64>(R))
        fail("num_relations does not match the schema");
    const bool relation_major = schema.options().layout == Layout::Relation;
    if (batch.scalar("layout") != (relation_major ? 1 : 0))
        fail(std::string("the layout is not ") + (relation_major ? "relation-major" : "graph-major"));
    const i64 index_dtype = batch.scalar("index_dtype");
    if (index_dtype != 0 && index_dtype != 1)
        fail("index_dtype is neither 0 nor 1");
    if ((index_dtype == 1) != (schema.options().index == IndexType::I64))
        fail("index_dtype does not match the schema's index type");
    if (batch.scalar("schema_fingerprint", 0) != std::bit_cast<i64>(schema.fingerprint()))
        fail("schema_fingerprint does not match the schema");
    if (batch.scalar("statics") != 0)
        fail("statics is not 0");
    const bool wide = index_dtype == 1;
    const rl::DType itype = wide ? rl::DType::I64 : rl::DType::I32;
    auto ints = [&](const rl::ArrayInfo& info) { return Ints{batch.data(info), info.elements(), wide}; };
    auto g_i64 = static_cast<i64>(G);
    auto r_i64 = static_cast<i64>(R);

    array(batch, "task_ids", rl::DType::I32, {g_i64});
    const Ints num_objects = ints(*array(batch, "num_objects", itype, {g_i64}));
    const Ints entity_offsets = ints(*array(batch, "entity_offsets", itype, {g_i64 + 1}));
    if (entity_offsets[0] != 0)
        fail("entity_offsets does not start at 0");
    const u64 E = static_cast<u64>(entity_offsets[G]);
    const bool implicit_entities = schema.implicit_entity_arrays();
    if (implicit_entities != (batch.find("entity_role") == nullptr) ||
        implicit_entities != (batch.find("entity_ref") == nullptr))
        fail(implicit_entities ? "entity_role and entity_ref must be omitted: the schema's entity arrays are implicit"
                               : "entity_role and entity_ref are missing");
    Ints entity_ref;
    const u8* entity_role = nullptr;
    if (!implicit_entities)
    {
        entity_ref = ints(*padded_array(batch, "entity_ref", itype, E, wide));
        entity_role = static_cast<const u8*>(batch.data(*padded_array(batch, "entity_role", rl::DType::U8, E, wide)));
    }
    const bool stored_counts = schema.emits_counts();
    if (stored_counts != (batch.find("relation_counts") != nullptr))
        fail(stored_counts ? "relation_counts is missing"
                           : "relation_counts must be omitted: the schema does not store counts");
    Ints counts;
    if (stored_counts)
        counts = ints(*array(batch, "relation_counts", itype, {g_i64, r_i64}));
    const Ints tuple_offsets = ints(*array(batch, "relation_offsets", itype, {g_i64 * r_i64 + 1}));
    const Ints arg_offsets = ints(*array(batch, "relation_arg_offsets", itype, {g_i64 * r_i64 + 1}));
    const u64 total_args = static_cast<u64>(arg_offsets[G * R]);
    const Ints args = ints(*padded_array(batch, "relation_args", itype, total_args, wide));

    const bool stored_complete = schema.has_goal_complete();
    if (stored_complete != (batch.find("goal_complete") != nullptr))
        fail(stored_complete ? "goal_complete is missing" : "goal_complete must be omitted: the schema has no goal relations");
    if (stored_complete)
    {
        const auto* info = array(batch, "goal_complete", rl::DType::U8, {g_i64});
        for (u64 g = 0; g < G; ++g)
            if (static_cast<const u8*>(batch.data(*info))[g] > 1)
                fail("goal_complete of graph " + std::to_string(g) + " is neither 0 nor 1");
    }
    const bool stored_targets = schema.has_targets();
    if (stored_targets != (batch.find("target_offsets") != nullptr) || stored_targets != (batch.find("target_entity") != nullptr) ||
        stored_targets != (batch.find("target_group") != nullptr))
        fail(stored_targets ? "a target array is missing" : "the target arrays must be omitted: the schema has no targets");
    Ints target_offsets, target_entity;
    const u8* target_group = nullptr;
    if (stored_targets)
    {
        target_offsets = ints(*array(batch, "target_offsets", itype, {g_i64 + 1}));
        const u64 T = static_cast<u64>(target_offsets[G]);
        target_entity = ints(*padded_array(batch, "target_entity", itype, T, wide));
        target_group = static_cast<const u8*>(batch.data(*padded_array(batch, "target_group", rl::DType::U8, T, wide)));
        check_offsets(target_offsets, T, "target_offsets");
    }
    check_offsets(entity_offsets, E, "entity_offsets");
    check_offsets(tuple_offsets, static_cast<u64>(tuple_offsets[G * R]), "relation_offsets");
    check_offsets(arg_offsets, total_args, "relation_arg_offsets");

    constexpr u8 k_object = static_cast<u8>(EntityRole::Object);
    constexpr u8 k_auxiliary = static_cast<u8>(EntityRole::Auxiliary);
    constexpr u8 k_goal = static_cast<u8>(EntityRole::Goal);
    constexpr u8 k_action = static_cast<u8>(EntityRole::Action);
    constexpr u8 k_virtual = static_cast<u8>(EntityRole::PredicateVirtual);
    const bool action_targets = schema.options().targets.contains(TargetSource::Action);
    for (u64 g = 0; g < G; ++g)
    {
        const i64 first = entity_offsets[g], last = entity_offsets[g + 1];
        const i64 objects = num_objects[g];
        if (objects < 0 || first + objects > last)
            fail("graph " + std::to_string(g) + " has more objects than entities");
        // The role of an entity: stored, or implied (an object below num_objects, else the auxiliary entity).
        auto role_of = [&](i64 e)
        { return entity_role ? entity_role[e] : (e < first + objects ? k_object : k_auxiliary); };
        if (implicit_entities && last - first - objects > 1)
            fail("graph " + std::to_string(g) + " has more than one entity past its objects");
        for (i64 e = first; e < last && !implicit_entities; ++e)
        {
            const bool is_object = e < first + objects;
            if (is_object && (entity_role[e] != k_object || entity_ref[e] != e - first))
                fail("entity " + std::to_string(e) + " of graph " + std::to_string(g) + " is not its object");
            if (!is_object && entity_role[e] == k_auxiliary && (e != first + objects || entity_ref[e] != -1))
                fail("graph " + std::to_string(g) + " has an auxiliary entity that is not at local index num_objects with ref -1");
            if (!is_object && entity_role[e] == k_object)
                fail("graph " + std::to_string(g) + " has an object after its objects");
        }
        u32 auxiliaries = 0;
        i64 goal_entities = 0, action_entities = 0, virtual_entities = 0;
        for (i64 e = first; e < last; ++e)
        {
            const u8 role = role_of(e);
            auxiliaries += role == k_auxiliary;
            goal_entities += role == k_goal;
            action_entities += role == k_action;
            virtual_entities += role == k_virtual;
        }
        if (auxiliaries > 1)
            fail("graph " + std::to_string(g) + " has more than one auxiliary entity");
        if (last - first != objects + auxiliaries + goal_entities + action_entities + virtual_entities)
            fail("graph " + std::to_string(g) + " has entities that are neither objects, auxiliary, goal, action nor predicate "
                                               "virtual entities");
        // The entities past the objects: the auxiliary entity, the goal entities, the action entities, the virtual entities.
        const i64 goal_start = first + objects + auxiliaries;
        const i64 action_start = goal_start + goal_entities;
        const i64 virtual_start = action_start + action_entities;
        for (i64 e = goal_start; e < last; ++e)
        {
            const u8 expected = e < action_start ? k_goal : e < virtual_start ? k_action : k_virtual;
            if (role_of(e) != expected)
                fail("the entities of graph " + std::to_string(g) + " past its objects are not in the order auxiliary, goal, "
                                                                  "action, predicate virtual");
        }
        // The goal entities follow the objects and the auxiliary entity, one per tuple of the plain goal relations with a
        // goal target, numbered in the order of those relations and of their tuples; each refers to its tuple, by the
        // index of the tuple among the graph's tuples in graph-major order. The action entities follow them, one per
        // tuple of an action relation, numbered the same way.
        i64 goals = 0, actions = 0, graph_tuples = 0;
        bool auxiliary_used = false;
        std::vector<bool> virtual_used(static_cast<usize>(virtual_entities), false);
        for (u64 r = 0; r < R; ++r)
        {
            const u64 cell = relation_major ? r * G + g : g * R + r;
            const u64 n = static_cast<u64>(tuple_offsets[cell + 1] - tuple_offsets[cell]);
            const u32 arity = schema.relation(static_cast<u32>(r)).arity;
            const auto roles = schema.slot_roles(static_cast<u32>(r));
            const std::string where = "cell (" + std::to_string(g) + ", " + schema.relation(static_cast<u32>(r)).name + ")";
            if (stored_counts && counts[g * R + r] != tuple_offsets[cell + 1] - tuple_offsets[cell])
                fail("relation_offsets and relation_counts disagree in " + where);
            if (static_cast<u64>(arg_offsets[cell + 1] - arg_offsets[cell]) != n * arity)
                fail("relation_arg_offsets and relation_counts disagree in " + where);
            const bool goal_target = roles.front() == SlotRole::GoalTarget;
            const bool action = roles.front() == SlotRole::Action;
            for (u64 t = 0; t < n; ++t)
            {
                const u64 at = static_cast<u64>(arg_offsets[cell]) + t * arity;
                if (goal_target)
                {
                    const i64 entity = args[at];
                    if (goals >= goal_entities || entity != goal_start + goals)
                        fail("a tuple of " + where + " does not have the next goal entity");
                    if (entity_role && entity_ref[entity] != graph_tuples + static_cast<i64>(t))
                        fail("the goal entity of a tuple of " + where + " does not refer to the tuple");
                    if (stored_targets && (target_offsets[g] + goals >= target_offsets[g + 1] ||
                                           target_entity[static_cast<u64>(target_offsets[g] + goals)] != entity ||
                                           target_group[target_offsets[g] + goals] != 0))
                        fail("the targets of graph " + std::to_string(g) + " do not list its goal entities in order");
                    ++goals;
                }
                if (action)
                {
                    const i64 entity = args[at];
                    if (actions >= action_entities || entity != action_start + actions)
                        fail("a tuple of " + where + " does not have the next action entity");
                    if (entity_role && entity_ref[entity] < 0)
                        fail("the action entity of a tuple of " + where +
                             " does not refer to a row of the graph's expansion");
                    if (action_targets)
                    {
                        const i64 position = target_offsets[g] + goal_entities + actions;
                        if (position >= target_offsets[g + 1] || target_entity[static_cast<u64>(position)] != entity ||
                            target_group[position] != static_cast<u8>(TargetGroup::Action))
                            fail("the targets of graph " + std::to_string(g) + " do not list its action entities in order");
                    }
                    ++actions;
                }
                for (u32 k = 0; k < arity; ++k)
                {
                    const i64 id = args[at + k];
                    if (id < first || id >= last)
                        fail("a tuple of " + where + " has an entity outside its graph");
                    if (role_of(id) != static_cast<u8>(entity_role_of(roles[k])))
                        fail("a tuple of " + where + " has an entity of the wrong role at position " + std::to_string(k));
                    if (roles[k] == SlotRole::Auxiliary)
                        auxiliary_used = true;
                    if (roles[k] == SlotRole::Predicate)
                    {
                        if (entity_ref[id] != schema.virtual_key(static_cast<u32>(r)))
                            fail("a tuple of " + where + " has the virtual node of another predicate");
                        virtual_used[static_cast<usize>(id - virtual_start)] = true;
                    }
                }
                if (t > 0)
                {
                    u32 k = 0;
                    while (k < arity && args[at + k] == args[at - arity + k])
                        ++k;
                    if (k == arity)
                        fail("a tuple occurs twice in " + where);
                    if (args[at + k] < args[at - arity + k])
                        fail("the tuples of " + where + " are not ascending");
                }
            }
            graph_tuples += static_cast<i64>(n);
        }
        const i64 expected_targets = goal_entities + (action_targets ? action_entities : 0);
        if (stored_targets && target_offsets[g] + expected_targets != target_offsets[g + 1])
            fail("graph " + std::to_string(g) + " has targets that are not its goal and action entities");
        if (goals != goal_entities)
            fail("graph " + std::to_string(g) + " has goal entities that no tuple of a plain goal relation starts with");
        if (actions != action_entities)
            fail("graph " + std::to_string(g) + " has action entities that no action tuple starts with");
        for (i64 v = 0; v < virtual_entities; ++v)
        {
            if (!virtual_used[static_cast<usize>(v)])
                fail("graph " + std::to_string(g) + " has a virtual node of a predicate that has no tuple");
            if (v > 0 && entity_ref[virtual_start + v] <= entity_ref[virtual_start + v - 1])
                fail("the virtual nodes of graph " + std::to_string(g) + " are not in the order of their predicates' keys");
        }
        if ((auxiliaries == 1) != auxiliary_used)
            fail("graph " + std::to_string(g) + (auxiliary_used ? " uses an auxiliary entity it does not have"
                                                                : " has an auxiliary entity no tuple uses"));
    }
}
}  // namespace mymyr::encoding
