#include "mymyr/encoding/validate.hpp"

#include "mymyr/encoding/detail/lanes.hpp"

#include <algorithm>
#include <bit>
#include <map>
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

/// A two-row array [2, capacity] of an index type holding at least `size` entries per row, zero past them.
void edge_rows(const rl::ArrayBundle& batch, const char* name, rl::DType dtype, u64 size, bool wide, Ints& row0, Ints& row1)
{
    const rl::ArrayInfo* info = batch.find(name);
    if (!info)
        fail(std::string("the array ") + name + " is missing");
    if (info->dtype != dtype)
        fail(std::string("the array ") + name + " has the wrong dtype");
    if (info->shape.size() != 2 || info->shape[0] != 2 || info->shape[1] < static_cast<i64>(size))
        fail(std::string("the array ") + name + " is not [2, edges] or is shorter than the batch needs");
    const u64 capacity = static_cast<u64>(info->shape[1]);
    const auto* base = static_cast<const std::byte*>(batch.data(*info));
    row0 = Ints{base, capacity, wide};
    row1 = Ints{base + capacity * (wide ? 8 : 4), capacity, wide};
    for (u64 i = size; i < capacity; ++i)
        if (row0[i] != 0 || row1[i] != 0)
            fail(std::string("the array ") + name + " is not zero past the batch");
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
/// An offsets array of at least `count` entries (a capacity bundle pads it) whose entries from `count` on repeat the
/// last entry before them.
const rl::ArrayInfo* padded_offsets(const rl::ArrayBundle& batch, const char* name, rl::DType dtype, u64 count, bool wide)
{
    const rl::ArrayInfo* info = batch.find(name);
    if (!info)
        fail(std::string("the array ") + name + " is missing");
    if (info->dtype != dtype)
        fail(std::string("the array ") + name + " has the wrong dtype");
    if (info->shape.size() != 1 || info->shape[0] < static_cast<i64>(count))
        fail(std::string("the array ") + name + " is shorter than the batch needs");
    const Ints data{batch.data(*info), info->elements(), wide};
    for (u64 i = count; i < info->elements(); ++i)
        if (data[i] != data[count - 1])
            fail(std::string("the array ") + name + " does not repeat its last entry past the batch");
    return info;
}

/// The static blocks of a batch with shared statics, checked against the schema and the graphs: the instance, the
/// objects and what the tuples use of each block.
struct StaticBlocks
{
    std::vector<i32> ids;                 // the instance of each block, ascending
    std::vector<i64> objects;             // the objects of each block's instance
    std::vector<bool> auxiliary;          // some tuple of the block uses the auxiliary entity
    std::vector<std::vector<i64>> keys;   // the predicate keys the tuples of the block use, ascending, each once
    [[nodiscard]] usize find(i32 id) const
    {
        return static_cast<usize>(std::ranges::lower_bound(ids, id) - ids.begin());
    }
};

StaticBlocks check_static_blocks(const rl::ArrayBundle& batch, const EncodingSchema& schema, const i32* task_ids,
                                 const Ints& num_objects, u64 G, rl::DType itype, bool wide, BatchForm form)
{
    constexpr const char* names[] = {"static_task_ids", "static_offsets", "static_arg_offsets", "static_args",
                                     "static_relations"};
    const std::span<const u32> relations = schema.static_relations();
    const u64 Rs = relations.size();
    // the instances of the graphs, with their objects
    std::map<i32, i64> instances;
    for (u64 g = 0; g < G; ++g)
    {
        const auto [at, inserted] = instances.emplace(task_ids[g], num_objects[g]);
        if (!inserted && at->second != num_objects[g])
            fail("graphs " + std::to_string(g) + " and an earlier one of instance " + std::to_string(task_ids[g]) +
                 " have different num_objects");
    }
    StaticBlocks blocks;
    for (const auto& [id, objects] : instances)
    {
        blocks.ids.push_back(id);
        blocks.objects.push_back(objects);
    }
    const u64 Ib = blocks.ids.size();
    blocks.auxiliary.assign(Ib, false);
    blocks.keys.resize(Ib);
    const Ints stored_relations = Ints{batch.data(*array(batch, names[4], itype, {static_cast<i64>(Rs)})), Rs, wide};
    for (u64 k = 0; k < Rs && form == BatchForm::Encoded; ++k)
        if (stored_relations[k] != static_cast<i64>(relations[k]))
            fail("static_relations does not list the schema's static relations");
    const auto* ids_info = padded_array(batch, names[0], rl::DType::I32, Ib, false);
    const i32* stored_ids = static_cast<const i32*>(batch.data(*ids_info));
    for (u64 b = 0; b < Ib; ++b)
        if (stored_ids[b] != blocks.ids[b])
            fail("static_task_ids is not the ascending list of the graphs' instances");
    const u64 cells = Ib * Rs + 1;
    const rl::ArrayInfo* tuple_info = padded_offsets(batch, names[1], itype, cells, wide);
    const rl::ArrayInfo* arg_info = padded_offsets(batch, names[2], itype, cells, wide);
    const Ints tuples{batch.data(*tuple_info), cells, wide};
    const Ints arg_offsets{batch.data(*arg_info), cells, wide};
    check_offsets(tuples, static_cast<u64>(tuples[cells - 1]), "static_offsets");
    check_offsets(arg_offsets, static_cast<u64>(arg_offsets[cells - 1]), "static_arg_offsets");
    const u64 total_args = static_cast<u64>(arg_offsets[cells - 1]);
    const Ints args{batch.data(*padded_array(batch, names[3], itype, total_args, wide)), total_args, wide};
    for (u64 b = 0; b < Ib; ++b)
        for (u64 k = 0; k < Rs; ++k)
        {
            const u64 segment = b * Rs + k;
            const u32 relation = relations[k];
            const u32 arity = schema.relation(relation).arity;
            const auto roles = schema.slot_roles(relation);
            const u64 n = static_cast<u64>(tuples[segment + 1] - tuples[segment]);
            const std::string where = "the static block of instance " + std::to_string(blocks.ids[b]) + ", relation " +
                                      schema.relation(relation).name;
            if (static_cast<u64>(arg_offsets[segment + 1] - arg_offsets[segment]) != n * arity)
                fail("static_offsets and static_arg_offsets disagree in " + where);
            for (u64 t = 0; t < n; ++t)
            {
                const u64 at = static_cast<u64>(arg_offsets[segment]) + t * arity;
                for (u32 position = 0; position < arity; ++position)
                {
                    const i64 id = args[at + position];
                    switch (roles[position])
                    {
                        case SlotRole::Argument:
                            if (id < 0 || id >= blocks.objects[b])
                                fail("a tuple of " + where + " has an object outside its instance");
                            break;
                        case SlotRole::Auxiliary:
                            if (id != blocks.objects[b])
                                fail("a tuple of " + where + " has an auxiliary entity that is not num_objects");
                            blocks.auxiliary[b] = true;
                            break;
                        case SlotRole::Predicate:
                            if (id != schema.virtual_key(relation))
                                fail("a tuple of " + where + " has the key of another predicate");
                            blocks.keys[b].push_back(id);
                            break;
                        default: fail("a static relation has a slot of another role");
                    }
                }
                if (t > 0)
                {
                    u32 position = 0;
                    while (position < arity && args[at + position] == args[at - arity + position])
                        ++position;
                    if (position == arity)
                        fail("a tuple occurs twice in " + where);
                    if (args[at + position] < args[at - arity + position])
                        fail("the tuples of " + where + " are not ascending");
                }
            }
        }
    // each key once, so that a graph looks up its virtual nodes once per predicate and not once per tuple
    for (std::vector<i64>& keys : blocks.keys)
    {
        std::ranges::sort(keys);
        keys.erase(std::ranges::unique(keys).begin(), keys.end());
    }
    return blocks;
}
}  // namespace

static void validate_relations(const rl::ArrayBundle& batch, const EncodingSchema& schema, BatchForm form)
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
    const bool shared = schema.shared_statics();
    if (batch.scalar("statics") != (shared ? 1 : 0))
        fail(std::string("statics is not ") + (shared ? "1" : "0"));
    for (const char* name : {"static_task_ids", "static_offsets", "static_arg_offsets", "static_args", "static_relations"})
        if ((batch.find(name) != nullptr) != shared)
            fail(shared ? std::string("the array ") + name + " is missing"
                        : std::string("the array ") + name + " must be omitted: the schema repeats its statics");
    const bool wide = index_dtype == 1;
    const rl::DType itype = wide ? rl::DType::I64 : rl::DType::I32;
    auto ints = [&](const rl::ArrayInfo& info) { return Ints{batch.data(info), info.elements(), wide}; };
    auto g_i64 = static_cast<i64>(G);
    auto r_i64 = static_cast<i64>(R);

    const i32* task_ids = static_cast<const i32*>(batch.data(*array(batch, "task_ids", rl::DType::I32, {g_i64})));
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
    const bool stored_history = schema.has_history();
    if (stored_history != (batch.find("history_offsets") != nullptr) ||
        stored_history != (batch.find("history_entity") != nullptr) ||
        stored_history != (batch.find("history_dt") != nullptr))
        fail(stored_history ? "a history array is missing"
                            : "the history arrays must be omitted: the schema has no history");
    Ints history_offsets, history_entity;
    if (stored_history)
    {
        history_offsets = ints(*array(batch, "history_offsets", itype, {g_i64 + 1}));
        const u64 S = static_cast<u64>(history_offsets[G]);
        history_entity = ints(*padded_array(batch, "history_entity", itype, S, wide));
        padded_array(batch, "history_dt", rl::DType::I32, S, false);
        check_offsets(history_offsets, S, "history_offsets");
    }
    check_offsets(entity_offsets, E, "entity_offsets");
    check_offsets(tuple_offsets, static_cast<u64>(tuple_offsets[G * R]), "relation_offsets");
    check_offsets(arg_offsets, total_args, "relation_arg_offsets");

    StaticBlocks blocks;
    if (shared)
        blocks = check_static_blocks(batch, schema, task_ids, num_objects, G, itype, wide, form);
    constexpr u8 k_object = static_cast<u8>(EntityRole::Object);
    constexpr u8 k_auxiliary = static_cast<u8>(EntityRole::Auxiliary);
    constexpr u8 k_goal = static_cast<u8>(EntityRole::Goal);
    constexpr u8 k_subgoal = static_cast<u8>(EntityRole::Subgoal);
    constexpr u8 k_action = static_cast<u8>(EntityRole::Action);
    constexpr u8 k_history = static_cast<u8>(EntityRole::History);
    constexpr u8 k_history_target = static_cast<u8>(EntityRole::HistoryTarget);
    constexpr u8 k_virtual = static_cast<u8>(EntityRole::PredicateVirtual);
    constexpr u8 k_state = static_cast<u8>(EntityRole::State);
    const bool horizon = schema.has_horizon();
    const bool root_target = schema.options().horizon.root == RootPolicy::Include;
    const bool root_entity = schema.options().horizon.root != RootPolicy::Exclude;
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
        i64 goal_entities = 0, subgoal_entities = 0, action_entities = 0, history_entities = 0, history_targets = 0,
            state_entities = 0, virtual_entities = 0;
        for (i64 e = first; e < last; ++e)
        {
            const u8 role = role_of(e);
            auxiliaries += role == k_auxiliary;
            goal_entities += role == k_goal;
            subgoal_entities += role == k_subgoal;
            action_entities += role == k_action;
            history_entities += role == k_history;
            history_targets += role == k_history_target;
            state_entities += role == k_state;
            virtual_entities += role == k_virtual;
        }
        if (auxiliaries > 1)
            fail("graph " + std::to_string(g) + " has more than one auxiliary entity");
        if (last - first != objects + auxiliaries + goal_entities + subgoal_entities + action_entities +
                                history_entities + history_targets + state_entities + virtual_entities)
            fail("graph " + std::to_string(g) +
                 " has entities that are neither objects, auxiliary, goal, subgoal, action, history step, history "
                 "target, state nor predicate virtual entities");
        if (state_entities != 0 && !horizon)
            fail("graph " + std::to_string(g) + " has state entities, the schema has no horizon");
        // The entities past the objects: the auxiliary entity, the goal entities, the subgoal entities, the action
        // entities, the history step entities, the history target entities, the state entities, the virtual entities.
        const i64 goal_start = first + objects + auxiliaries;
        const i64 subgoal_start = goal_start + goal_entities;
        const i64 action_start = subgoal_start + subgoal_entities;
        const i64 history_start = action_start + action_entities;
        const i64 history_target_start = history_start + history_entities;
        const i64 state_start = history_target_start + history_targets;
        const i64 virtual_start = state_start + state_entities;
        for (i64 e = goal_start; e < last; ++e)
        {
            const u8 expected = e < subgoal_start          ? k_goal
                                : e < action_start         ? k_subgoal
                                : e < history_start        ? k_action
                                : e < history_target_start ? k_history
                                : e < state_start          ? k_history_target
                                : e < virtual_start        ? k_state
                                                           : k_virtual;
            if (role_of(e) != expected)
                fail("the entities of graph " + std::to_string(g) + " past its objects are not in the order auxiliary, goal, "
                                                                  "subgoal, action, history step, history target, "
                                                                  "state, predicate virtual");
        }
        // The state entities of a horizon refer to their nodes: consecutively from the first node that has an entity.
        for (i64 j = 0; j < state_entities; ++j)
            if (entity_ref[state_start + j] != j + (root_entity ? 0 : 1))
                fail("the state entities of graph " + std::to_string(g) + " do not refer to consecutive nodes from " +
                     (root_entity ? "the root" : "the first node below the root"));
        if (stored_history)
        {
            const i64 steps = history_offsets[g + 1] - history_offsets[g];
            if (steps != history_entities)
                fail("graph " + std::to_string(g) + " has history_offsets that do not count its history step entities");
            for (i64 j = 0; j < steps; ++j)
            {
                if (history_entity[static_cast<u64>(history_offsets[g] + j)] != history_start + j)
                    fail("history_entity of graph " + std::to_string(g) + " does not list its history steps in order");
                if (entity_role && j > 0 && entity_ref[history_start + j] <= entity_ref[history_start + j - 1])
                    fail("the history steps of graph " + std::to_string(g) + " do not have ascending references");
            }
        }
        else if (history_entities != 0 || history_targets != 0)
            fail("graph " + std::to_string(g) + " has history entities, the schema has no history");
        // The goal entities follow the objects and the auxiliary entity, one per tuple of the plain goal relations with a
        // goal target, numbered in the order of those relations and of their tuples; each refers to its tuple, by the
        // index of the tuple among the graph's tuples in graph-major order. The action entities follow them, one per
        // tuple of an action relation, numbered the same way.
        i64 goals = 0, subgoals = 0, actions = 0, history_target_count = 0, graph_tuples = 0;
        const i64 target_start = stored_targets ? static_cast<i64>(target_offsets[g]) : 0;
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
            if (shared && n != 0 && schema.relation(static_cast<u32>(r)).lane == Lane::Static)
                fail(where + " has tuples, the schema shares its static tuples");
            if (static_cast<u64>(arg_offsets[cell + 1] - arg_offsets[cell]) != n * arity)
                fail("relation_arg_offsets and relation_counts disagree in " + where);
            const bool goal_target = roles.front() == SlotRole::GoalTarget;
            const bool subgoal_target = roles.front() == SlotRole::SubgoalTarget;
            const bool history_target = roles.front() == SlotRole::HistoryTarget;
            const bool action = roles.front() == SlotRole::Action;
            const bool leading = goal_target || subgoal_target || history_target || action;
            // the tuples of a target or action relation start with their own entity, which is ascending by
            // construction, so that the tuples are compared past it
            const u32 lead = leading ? 1u : 0u;
            // checks that the target entry at `position` lists `entity` in `group`
            auto check_target = [&](i64 position, i64 entity, TargetGroup group, const char* what)
            {
                if (stored_targets && (position >= target_offsets[g + 1] ||
                                       target_entity[static_cast<u64>(position)] != entity ||
                                       target_group[position] != static_cast<u8>(group)))
                    fail("the targets of graph " + std::to_string(g) + " do not list its " + what +
                         " entities in order");
            };
            for (u64 t = 0; t < n; ++t)
            {
                const u64 at = static_cast<u64>(arg_offsets[cell]) + t * arity;
                if (leading && (args[at] < first || args[at] >= last))
                    fail("a tuple of " + where + " has an entity outside its graph");
                if (goal_target)
                {
                    const i64 entity = args[at];
                    if (goals >= goal_entities || entity != goal_start + goals)
                        fail("a tuple of " + where + " does not have the next goal entity");
                    if (entity_role && entity_ref[entity] != graph_tuples + static_cast<i64>(t))
                        fail("the goal entity of a tuple of " + where + " does not refer to the tuple");
                    check_target(target_start + goals, entity, TargetGroup::Goal, "goal");
                    ++goals;
                }
                if (subgoal_target)
                {
                    const i64 entity = args[at];
                    if (subgoals >= subgoal_entities || entity != subgoal_start + subgoals)
                        fail("a tuple of " + where + " does not have the next subgoal entity");
                    if (entity_role && entity_ref[entity] != graph_tuples + static_cast<i64>(t))
                        fail("the subgoal entity of a tuple of " + where + " does not refer to the tuple");
                    check_target(target_start + goal_entities + subgoals, entity, TargetGroup::Subgoal, "subgoal");
                    ++subgoals;
                }
                if (history_target)
                {
                    const i64 entity = args[at];
                    if (history_target_count >= history_targets ||
                        entity != history_target_start + history_target_count)
                        fail("a tuple of " + where + " does not have the next history target entity");
                    if (entity_role && entity_ref[entity] != graph_tuples + static_cast<i64>(t))
                        fail("the history target entity of a tuple of " + where + " does not refer to the tuple");
                    check_target(target_start + goal_entities + subgoal_entities +
                                     (action_targets ? action_entities : 0) + history_target_count,
                                 entity, TargetGroup::History, "history target");
                    ++history_target_count;
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
                        check_target(target_start + goal_entities + subgoal_entities + actions, entity,
                                     TargetGroup::Action, "action");
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
                    u32 k = lead;
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
        const i64 candidates = root_target ? state_entities : (state_entities > 0 ? state_entities - (root_entity ? 1 : 0) : 0);
        const i64 expected_targets =
            goal_entities + subgoal_entities + (action_targets ? action_entities : 0) + history_targets + candidates;
        // the state targets: the candidate nodes' entities in order, the root's only with the root policy `include`
        if (stored_targets && horizon)
            for (i64 j = 0; j < candidates; ++j)
            {
                const i64 position = target_start + j;
                const i64 entity = state_start + (state_entities - candidates) + j;
                if (position >= target_offsets[g + 1] || target_entity[static_cast<u64>(position)] != entity ||
                    target_group[position] != static_cast<u8>(TargetGroup::State))
                    fail("the targets of graph " + std::to_string(g) + " do not list its candidate state entities in order");
            }
        if (stored_targets && target_offsets[g] + expected_targets != target_offsets[g + 1])
            fail("graph " + std::to_string(g) +
                 " has targets that are not its goal, subgoal, action and history target entities");
        if (goals != goal_entities)
            fail("graph " + std::to_string(g) + " has goal entities that no tuple of a plain goal relation starts with");
        if (subgoals != subgoal_entities)
            fail("graph " + std::to_string(g) +
                 " has subgoal entities that no tuple of a plain subgoal relation starts with");
        if (history_target_count != history_targets)
            fail("graph " + std::to_string(g) + " has history target entities that no history tuple starts with");
        if (actions != action_entities)
            fail("graph " + std::to_string(g) + " has action entities that no action tuple starts with");
        if (shared)
        {
            // the static tuples of the graph's instance are among its tuples: they use the entities they name
            const usize block = blocks.find(task_ids[g]);
            auxiliary_used = auxiliary_used || blocks.auxiliary[block];
            for (const i64 key : blocks.keys[block])
            {
                i64 node = 0;
                while (node < virtual_entities && entity_ref[virtual_start + node] != key)
                    ++node;
                if (node == virtual_entities)
                    fail("graph " + std::to_string(g) + " has no virtual node for the predicate of a static tuple");
                virtual_used[static_cast<usize>(node)] = true;
            }
        }
        for (i64 v = 0; v < virtual_entities; ++v)
        {
            if (!virtual_used[static_cast<usize>(v)])
                fail("graph " + std::to_string(g) + " has a virtual node of a predicate that has no tuple");
            if (v > 0 && schema.virtual_rank(static_cast<u32>(entity_ref[virtual_start + v])) <=
                             schema.virtual_rank(static_cast<u32>(entity_ref[virtual_start + v - 1])))
                fail("the virtual nodes of graph " + std::to_string(g) +
                     " are not in the order of their predicates' names");
        }
        if ((auxiliaries == 1) != auxiliary_used)
            fail("graph " + std::to_string(g) + (auxiliary_used ? " uses an auxiliary entity it does not have"
                                                                : " has an auxiliary entity no tuple uses"));
    }
}

namespace
{
/// The checks of the Simple graph of `batch`: its arrays, and, edge by edge, that it is the incidence graph of the
/// tuples it encodes (those of the relation arrays when the batch stores them).
void validate_simple(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const i64 graphs_scalar = batch.scalar("num_graphs");
    if (graphs_scalar < 0)
        fail("num_graphs is missing");
    const u64 G = static_cast<u64>(graphs_scalar);
    const u64 R = schema.num_relations();
    const u64 K = schema.simple_max_arity();
    const u64 constants = schema.simple_constants();
    const bool wide = batch.scalar("index_dtype") == 1;
    const rl::DType itype = wide ? rl::DType::I64 : rl::DType::I32;
    const auto g_i64 = static_cast<i64>(G);
    auto ints = [&](const rl::ArrayInfo& info) { return Ints{batch.data(info), info.elements(), wide}; };
    if (batch.scalar("simple_node_labels") != static_cast<i64>(schema.simple_node_labels()) ||
        batch.scalar("simple_edge_labels") != static_cast<i64>(schema.simple_edge_labels()))
        fail("simple_node_labels or simple_edge_labels does not match the schema");

    const Ints entity_offsets = ints(*array(batch, "entity_offsets", itype, {g_i64 + 1}));
    const u64 E = static_cast<u64>(entity_offsets[G]);
    if (batch.find("entity_role") == nullptr)
        fail("entity_role is missing: the Simple graph labels its entity nodes with the roles");
    const u8* entity_role =
        static_cast<const u8*>(batch.data(*padded_array(batch, "entity_role", rl::DType::U8, E, wide)));
    const Ints node_offsets = ints(*array(batch, "node_offsets", itype, {g_i64 + 1}));
    const Ints edge_offsets = ints(*array(batch, "edge_offsets", itype, {g_i64 + 1}));
    const u64 V = static_cast<u64>(node_offsets[G]), M = static_cast<u64>(edge_offsets[G]);
    check_offsets(node_offsets, V, "node_offsets");
    check_offsets(edge_offsets, M, "edge_offsets");
    const i32* node_label =
        static_cast<const i32*>(batch.data(*padded_array(batch, "node_label", rl::DType::I32, V, false)));
    const i32* edge_label =
        static_cast<const i32*>(batch.data(*padded_array(batch, "edge_label", rl::DType::I32, M, false)));
    Ints source, target;
    edge_rows(batch, "edge_index", itype, M, wide, source, target);
    if (V < E)
        fail("the Simple graph has fewer nodes than the batch has entities");

    Ints relation_offsets, relation_arg_offsets, relation_args;
    const bool relations = schema.emits_relations();
    const bool relation_major = schema.options().layout == Layout::Relation;
    if (relations)
    {
        relation_offsets = ints(*array(batch, "relation_offsets", itype, {g_i64 * static_cast<i64>(R) + 1}));
        relation_arg_offsets = ints(*array(batch, "relation_arg_offsets", itype, {g_i64 * static_cast<i64>(R) + 1}));
        relation_args = ints(
            *padded_array(batch, "relation_args", itype, static_cast<u64>(relation_arg_offsets[G * R]), wide));
    }
    u64 tuple_base = 0;
    std::vector<u64> seen(R);
    for (u64 g = 0; g < G; ++g)
    {
        const std::string where = "graph " + std::to_string(g);
        const u64 first = static_cast<u64>(entity_offsets[g]);
        const u64 entities = static_cast<u64>(entity_offsets[g + 1]) - first;
        const u64 node_first = static_cast<u64>(node_offsets[g]);
        const u64 nodes = static_cast<u64>(node_offsets[g + 1]) - node_first;
        if (node_first != first + tuple_base)
            fail("node_offsets of " + where + " is not its first entity plus the tuples before it");
        if (nodes < entities)
            fail(where + " has fewer nodes than entities");
        const u64 tuples = nodes - entities;
        for (u64 j = 0; j < entities; ++j)
        {
            const u8 role = entity_role[first + j];
            const bool constant = role == static_cast<u8>(EntityRole::Object) && j < constants;
            const i32 expected = constant ? static_cast<i32>(detail::k_simple_roles + R + j) : static_cast<i32>(role);
            if (node_label[node_first + j] != expected)
                fail("the label of entity " + std::to_string(j) + " of " + where + " is not its role" +
                     (constants ? " or constant" : ""));
        }
        const u64 edge_first = static_cast<u64>(edge_offsets[g]);
        const u64 edges = static_cast<u64>(edge_offsets[g + 1]) - edge_first;
        if (edge_first % 2 != 0 || edges % 2 != 0)
            fail("the edges of " + where + " are not forward and reverse in pairs");
        const u64 args = edges / 2;
        std::fill(seen.begin(), seen.end(), u64{0});
        u64 arg = 0;
        i64 previous = -1;
        for (u64 t = 0; t < tuples; ++t)
        {
            const i32 label = node_label[node_first + entities + t];
            if (label < static_cast<i32>(detail::k_simple_roles) ||
                label >= static_cast<i32>(detail::k_simple_roles + R))
                fail("atom " + std::to_string(t) + " of " + where + " has no relation label");
            const u32 r = static_cast<u32>(label) - detail::k_simple_roles;
            if (static_cast<i64>(r) < previous)
                fail("the atoms of " + where + " are not in the order of their relations");
            previous = r;
            const u32 arity = schema.relation(r).arity;
            const auto roles = schema.slot_roles(r);
            const i64 atom = static_cast<i64>(node_first + entities + t);
            if (arg + arity > args)
                fail("the edges of " + where + " are fewer than its atoms' arguments");
            for (u32 k = 0; k < arity; ++k)
            {
                const u64 forward = edge_first + arg + k, back = forward + args;
                const i64 entity = source[forward];
                if (target[forward] != atom || source[back] != atom || target[back] != entity ||
                    edge_label[forward] != static_cast<i32>(k) || edge_label[back] != static_cast<i32>(K + k))
                    fail("the edges of atom " + std::to_string(t) + " of " + where + " at position " +
                         std::to_string(k) + " are not an edge and its reverse with the labels k and K + k");
                if (entity < static_cast<i64>(node_first) || entity >= static_cast<i64>(node_first + entities))
                    fail("an edge of " + where + " leaves its entity nodes");
                if (entity_role[first + static_cast<u64>(entity) - node_first] !=
                    static_cast<u8>(entity_role_of(roles[k])))
                    fail("an atom of " + where + " has an entity of the wrong role at position " + std::to_string(k));
                if (relations)
                {
                    const u64 cell = relation_major ? r * G + g : g * R + r;
                    const u64 at = static_cast<u64>(relation_arg_offsets[cell]) + seen[r] * arity + k;
                    if (relation_args[at] != entity - static_cast<i64>(tuple_base))
                        fail("the edges of " + where + " differ from the tuples of its relation " +
                             schema.relation(r).name);
                }
            }
            ++seen[r];
            arg += arity;
        }
        if (arg != args)
            fail(where + " has edges that no atom's position explains");
        if (relations)
            for (u64 r = 0; r < R; ++r)
            {
                const u64 cell = relation_major ? r * G + g : g * R + r;
                if (static_cast<u64>(relation_offsets[cell + 1] - relation_offsets[cell]) != seen[r])
                    fail(where + " has atoms that differ in number from the tuples of relation " +
                         schema.relation(static_cast<u32>(r)).name);
            }
        tuple_base += tuples;
    }
    if (V != E + tuple_base)
        fail("node_offsets does not end at the entities plus the tuples");
}
}  // namespace

namespace
{
/// The value at `i` of a float array of either width, as a double.
f64 real_at(const void* data, bool wide, u64 i)
{
    if (wide)
        return static_cast<const f64*>(data)[i];
    return static_cast<f64>(static_cast<const f32*>(data)[i]);
}

bool same_real(f64 a, f64 b) { return (a != a && b != b) || a == b; }

/// Whether the entries [from, to) of `width` bytes each are zero.
bool zero_past(const void* data, u64 from, u64 to, usize width)
{
    const auto* bytes = static_cast<const std::byte*>(data);
    for (u64 i = from * width; i < to * width; ++i)
        if (bytes[i] != std::byte{0})
            return false;
    return true;
}

/// The checks of the numeric values: their arrays exist iff the schema stores them, hold the value of every numeric tuple
/// in the order of the numeric segments, are zero past the batch, and node_value agrees with them.
void validate_numeric(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const bool values = schema.stores_numeric_values(), node_values = schema.stores_node_value();
    if (values != (batch.find("numeric_values") != nullptr))
        fail(values ? "numeric_values is missing" : "numeric_values must be omitted: the schema does not store it");
    if (node_values != (batch.find("node_value") != nullptr))
        fail(node_values ? "node_value is missing" : "node_value must be omitted: the schema does not store it");
    if (!values && !node_values)
        return;
    const u64 G = static_cast<u64>(batch.scalar("num_graphs"));
    const u64 R = schema.num_relations();
    const bool wide_index = batch.scalar("index_dtype") == 1;
    const bool relation_major = schema.options().layout == Layout::Relation;
    const bool wide_values = schema.options().numeric_dtype == NumericDtype::F64;
    auto ints = [&](const char* name)
    {
        const rl::ArrayInfo* info = batch.find(name);
        if (!info)
            fail(std::string("the array ") + name + " is missing");
        return Ints{batch.data(*info), info->elements(), wide_index};
    };
    // where the values of each numeric segment start
    std::vector<u64> start(G * R, 0);
    u64 total = 0;
    Ints tuple_offsets;
    if (values)
    {
        tuple_offsets = ints("relation_offsets");
        auto visit = [&](u64 g, u32 r)
        {
            const u64 cell = relation_major ? r * G + g : g * R + r;
            start[g * R + r] = total;
            total += static_cast<u64>(tuple_offsets[cell + 1] - tuple_offsets[cell]);
        };
        if (relation_major)
            for (u32 r = schema.first_numeric_relation(); r < R; ++r)
                for (u64 g = 0; g < G; ++g)
                    visit(g, r);
        else
            for (u64 g = 0; g < G; ++g)
                for (u32 r = schema.first_numeric_relation(); r < R; ++r)
                    visit(g, r);
        const rl::ArrayInfo* info = batch.find("numeric_values");
        if (info->dtype != (wide_values ? rl::DType::F64 : rl::DType::F32))
            fail("numeric_values has the wrong dtype");
        if (info->shape.size() != 1 || info->shape[0] < static_cast<i64>(total))
            fail("numeric_values is shorter than the batch needs");
        if (!zero_past(batch.data(*info), total, info->elements(), wide_values ? 8 : 4))
            fail("numeric_values is not zero past the batch");
        if (batch.scalar("numeric_dtype") != (wide_values ? 1 : 0))
            fail("numeric_dtype does not match the schema's numeric dtype");
        if (batch.scalar("numeric_relations") != static_cast<i64>(R - schema.first_numeric_relation()))
            fail("numeric_relations does not match the schema's numeric relations");
    }
    if (!node_values)
        return;
    const Ints node_offsets = ints("node_offsets");
    const Ints entity_offsets = ints("entity_offsets");
    const u64 V = static_cast<u64>(node_offsets[G]);
    const rl::ArrayInfo* info = batch.find("node_value");
    if (info->dtype != rl::DType::F32)
        fail("node_value has the wrong dtype");
    if (info->shape.size() != 1 || info->shape[0] < static_cast<i64>(V))
        fail("node_value is shorter than the batch needs");
    if (!zero_past(batch.data(*info), V, info->elements(), 4))
        fail("node_value is not zero past the batch");
    const auto* node_value = static_cast<const f32*>(batch.data(*info));
    const rl::ArrayInfo* labels = batch.find("node_label");
    if (!labels)
        fail("the array node_label is missing");
    const auto* node_label = static_cast<const i32*>(batch.data(*labels));
    const void* numeric = values ? batch.data(*batch.find("numeric_values")) : nullptr;
    std::vector<u64> seen(R);
    for (u64 g = 0; g < G; ++g)
    {
        const u64 first = static_cast<u64>(node_offsets[g]);
        const u64 entities = static_cast<u64>(entity_offsets[g + 1] - entity_offsets[g]);
        std::fill(seen.begin(), seen.end(), u64{0});
        for (u64 n = first; n < static_cast<u64>(node_offsets[g + 1]); ++n)
        {
            const bool atom = n >= first + entities;
            const i64 relation = atom ? node_label[n] - static_cast<i64>(detail::k_simple_roles) : -1;
            if (!atom || relation < static_cast<i64>(schema.first_numeric_relation()) || relation >= static_cast<i64>(R))
            {
                if (node_value[n] != 0.0f)
                    fail("node_value of a node of graph " + std::to_string(g) + " that is no numeric atom is not 0");
                continue;
            }
            if (numeric)
            {
                const u64 at = start[g * R + static_cast<u64>(relation)] + seen[static_cast<u64>(relation)]++;
                const f32 want = static_cast<f32>(real_at(numeric, wide_values, at));
                if (!same_real(node_value[n], want))
                    fail("node_value of a numeric atom of graph " + std::to_string(g) + " differs from its numeric value");
            }
        }
    }
}
}  // namespace

void validate(const rl::ArrayBundle& batch, const EncodingSchema& schema, BatchForm form)
{
    if (schema.emits_relations())
        validate_relations(batch, schema, form);
    if (schema.emits_simple())
        validate_simple(batch, schema);
    validate_numeric(batch, schema);
    if (form == BatchForm::Overflowed)
        for (const rl::ArrayInfo& info : batch.arrays())
        {
            const auto* bytes = static_cast<const u8*>(batch.data(info));
            const u64 size = info.elements() * rl::dtype_bytes(info.dtype);
            if (std::any_of(bytes, bytes + size, [](u8 b) { return b != 0; }))
                fail("the batch overflowed, but '" + std::string(info.name.view()) + "' is not zero");
        }
}
}  // namespace mymyr::encoding
