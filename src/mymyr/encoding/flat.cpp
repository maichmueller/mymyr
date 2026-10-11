#include "mymyr/encoding/flat.hpp"

#include "mymyr/core/checked.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mymyr::encoding
{
namespace
{
i32 index_of(u64 value)
{
    if (value > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::length_error("mymyr: encoding: a packed index exceeds int32 capacity");
    return static_cast<i32>(value);
}

struct Tuple
{
    u32 relation = 0;
    u32 begin = 0;
    u32 arity = 0;
    i32 action = -1;
};

bool atom_holds(const Task& task, const detail::Engine& engine, const GroundAtom& atom)
{
    const u32 predicate = atom.predicate.v;
    std::vector<u32> objects;
    objects.reserve(atom.objects.size());
    for (ObjectId object : atom.objects)
        objects.push_back(object.v);
    if (task.data().predicates[predicate].kind == formalism::PredKind::Static)
    {
        const plan::StaticRelation& relation = task.compiled().statics[predicate];
        u64 key = 0;
        for (u32 position = 0; position < objects.size(); ++position)
            key += relation.position_table(position, task.num_objects())[objects[position]];
        return relation.contains(key);
    }
    const CanonicalAtom canonical = task.atoms().layout().encode(predicate, objects.data());
    const u32 slot = task.atoms().find(canonical);
    if (slot >= AtomIndex::k_pending)
        return false;
    if (task.data().predicates[predicate].kind == formalism::PredKind::Derived)
        return bits::test(engine.derived(), bits::words_for(task.atoms().derived_slots()), slot);
    return bits::test(engine.state(), engine.state_words(), slot);
}
}

FlatRelationEncoder::FlatRelationEncoder(TaskPtr task, FlatOptions options)
    : FlatRelationEncoder(rl::TaskTable::single(std::move(task)), options)
{
}

FlatRelationEncoder::FlatRelationEncoder(rl::TaskTablePtr table, FlatOptions options)
    : m_table(std::move(table)), m_options(options)
{
    if (!m_table)
        throw std::invalid_argument("mymyr: encoding: the task table is null");
    auto schema = std::make_shared<FlatSchema>();
    schema->domain_predicates = m_table->num_domain_predicates();
    const u32 predicates = static_cast<u32>(m_table->predicate_names().size());
    index_of(checked_add(checked_mul({predicates, 5}, "encoding schema"), m_table->num_schemas(), "encoding schema"));
    for (u32 predicate = 0; predicate < predicates; ++predicate)
    {
        std::string name = m_table->predicate_names()[predicate];
        const i32 instance = m_table->predicate_instances()[predicate];
        if (instance >= 0)
            name += "[@instance:" + std::to_string(instance) + "]";
        for (u32 lane = 0; lane < 5; ++lane)
        {
            const bool positive = lane <= 2;
            const bool satisfied = lane == 1 || lane == 3;
            schema->names.push_back(lane == 0 ? name : std::string(positive ? "[+]" : "[-]") + name + "[g]" +
                                                          (satisfied ? "[sat]" : "[unsat]"));
            schema->sources.emplace_back(lane == 0 ? "state" : satisfied ? "goal_satisfied" : "goal_unsatisfied");
            schema->arities.push_back(index_of(m_table->predicate_arities()[predicate]));
            schema->predicates.push_back(index_of(predicate));
            schema->schemas.push_back(-1);
            schema->kinds.push_back(m_table->predicate_kinds()[predicate]);
            schema->instances.push_back(instance);
            schema->polarities.push_back(lane == 0 ? -1 : positive ? 1 : 0);
        }
    }
    for (u32 action = 0; action < m_table->num_schemas(); ++action)
    {
        schema->names.push_back(m_table->schema_names()[action] + "[action]");
        schema->sources.emplace_back("action");
        schema->arities.push_back(index_of(m_table->schema_arities()[action]));
        schema->predicates.push_back(-1);
        schema->schemas.push_back(index_of(action));
        schema->kinds.push_back(-1);
        schema->instances.push_back(-1);
        schema->polarities.push_back(-1);
    }
    for (const auto& instance : m_table->instances())
    {
        const Task& task = *instance.task;
        if (task.data().has_numerics() || !task.data().constraints.empty())
            throw std::invalid_argument("mymyr: encoding: numeric functions and constraints are not supported");
        m_goals.push_back(GroundCondition::goal(task));
    }
    m_schema = std::move(schema);
}

FlatBatch FlatRelationEncoder::encode(rl::StateBatchView states, const i32* task_ids, ActionBatchView actions,
                                      std::span<const GroundCondition* const> goals) const
{
    const u32 graphs = static_cast<u32>(index_of(states.rows));
    const u32 relations = static_cast<u32>(m_schema->names.size());
    const u32 predicate_relations = static_cast<u32>(m_table->predicate_names().size()) * 5;
    const u64 cells = checked_mul({graphs, relations}, "encoding counts");
    index_of(cells);
    if (graphs && states.words && !states.data)
        throw std::invalid_argument("mymyr: encoding: the state buffer is null");
    if (states.numeric_words)
        throw std::invalid_argument("mymyr: encoding: numeric state rows are not supported");
    if (states.stride && states.stride < states.words)
        throw std::invalid_argument("mymyr: encoding: the state row stride is smaller than its width");
    if (graphs && states.words)
    {
        const u64 extent = checked_add(checked_mul({graphs - 1, states.stride ? states.stride : states.words},
                                                  "encoding state rows"), states.words, "encoding state rows");
        if (extent > static_cast<u64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(u64))
            throw std::length_error("mymyr: encoding: the state row span exceeds pointer capacity");
    }
    if (graphs && m_table->size() > 1 && !task_ids)
        throw std::invalid_argument("mymyr: encoding: task_ids are required for a multi-instance table");
    m_table->check_task_ids(task_ids, graphs);
    if (!goals.empty() && goals.size() != graphs)
        throw std::invalid_argument("mymyr: encoding: goals need one condition per state row");
    if (actions.size && (!actions.offsets || !actions.schema || (actions.label_width && !actions.binding)))
        throw std::invalid_argument("mymyr: encoding: incomplete action arrays");
    if (actions.offsets)
    {
        index_of(actions.size);
        if (actions.offsets[0] != 0 || actions.offsets[graphs] != static_cast<i32>(actions.size))
            throw std::invalid_argument("mymyr: encoding: action offsets must cover every action exactly");
        for (u32 graph = 0; graph < graphs; ++graph)
            if (actions.offsets[graph] < 0 || actions.offsets[graph] > actions.offsets[graph + 1])
                throw std::invalid_argument("mymyr: encoding: action offsets must be nonnegative and monotone");
    }

    FlatBatch output;
    output.schema = m_schema;
    output.graphs = graphs;
    std::vector<i32> counts(cells, 0), offsets{0}, tuple_offsets{0}, arguments, object_offsets{0}, ids;
    std::vector<i32> action_offsets{0}, action_schema, action_binding, action_parent, action_tuple(actions.size, -1);
    std::vector<Tuple> tuples;
    std::vector<u32> local_arguments;
    const u32 label_width = actions.offsets ? actions.label_width : m_table->label_width();
    index_of(checked_mul({actions.size, label_width}, "encoding action bindings"));
    u32 tuple_count = 0;
    for (u32 graph = 0; graph < graphs; ++graph)
    {
        const u32 instance_id = task_ids ? static_cast<u32>(task_ids[graph]) : 0;
        const auto& instance = m_table->instance(instance_id);
        const Task& task = *instance.task;
        const u64* state = states.words ? states.row(graph) : nullptr;
        const u32 assigned = task.atoms().fluent_slots();
        for (u32 word_index = 0; word_index < states.words; ++word_index)
        {
            u64 word = state[word_index];
            while (word)
            {
                const u64 slot = static_cast<u64>(word_index) * 64 + std::countr_zero(word);
                if (slot >= assigned)
                    throw std::invalid_argument("mymyr: encoding: a state sets an unassigned atom slot");
                word &= word - 1;
            }
        }
        const GroundCondition& condition = goals.empty() || !goals[graph] ? m_goals[instance_id] : *goals[graph];
        condition.validate(task);
        if (!condition.constraints.empty())
            throw std::invalid_argument("mymyr: encoding: numeric goals are not supported");
        const WorkspaceLease workspace = task.workspace();
        detail::Engine& engine = workspace->engine();
        engine.set_state(state, states.words);
        engine.build_view();
        workspace->axioms().evaluate();
        tuples.clear();
        local_arguments.clear();
        auto append = [&](u32 relation, u32 arity, auto object_at, i32 action = -1)
        {
            const u32 begin = static_cast<u32>(index_of(local_arguments.size()));
            index_of(checked_add(begin, arity, "encoding tuple arguments"));
            tuples.push_back({relation, begin, arity, action});
            for (u32 position = 0; position < arity; ++position)
                local_arguments.push_back(object_at(position));
        };
        auto append_slots = [&](AtomKind kind, const u64* words, u32 width)
        {
            for (u32 word_index = 0; word_index < width; ++word_index)
                for (u64 word = words[word_index]; word; word &= word - 1)
                {
                    const u32 slot = word_index * 64 + std::countr_zero(word);
                    const u32* record = task.atoms().record(kind, slot);
                    const u32 predicate = record[0];
                    append(instance.pred_map[predicate] * 5, task.data().predicates[predicate].arity,
                           [&](u32 position) { return record[position + 1]; });
                }
        };
        append_slots(AtomKind::Fluent, state, states.words);
        if (m_options.include_static)
            for (const formalism::GroundAtom& atom : task.data().static_init)
            {
                const auto objects = task.data().objects_of(atom);
                append(instance.pred_map[atom.pred.v] * 5, static_cast<u32>(objects.size()),
                       [&](u32 position) { return objects[position].v; });
            }
        if (m_options.include_derived)
            append_slots(AtomKind::Derived, engine.derived(), bits::words_for(task.atoms().derived_slots()));
        for (const GroundLiteral& literal : condition.literals)
        {
            const bool satisfied = atom_holds(task, engine, literal.atom) == literal.positive;
            const u32 lane = literal.positive ? (satisfied ? 1 : 2) : (satisfied ? 3 : 4);
            append(instance.pred_map[literal.atom.predicate.v] * 5 + lane,
                   static_cast<u32>(literal.atom.objects.size()),
                   [&](u32 position) { return literal.atom.objects[position].v; });
        }
        if (actions.offsets)
            for (i32 action = actions.offsets[graph]; action < actions.offsets[graph + 1]; ++action)
            {
                const i32 schema = actions.schema[action];
                if (schema < 0 || static_cast<u32>(schema) >= m_table->num_schemas())
                    throw std::invalid_argument("mymyr: encoding: an action schema is out of range");
                const u32 arity = m_table->schema_arities()[schema];
                if (label_width < arity)
                    throw std::invalid_argument("mymyr: encoding: an action binding is too narrow");
                const i32* binding = label_width ? actions.binding + static_cast<u64>(action) * label_width : nullptr;
                for (u32 position = 0; position < label_width; ++position)
                {
                    if (position < arity ? binding[position] < 0 || static_cast<u32>(binding[position]) >= task.num_objects()
                                         : binding[position] != -1)
                        throw std::invalid_argument("mymyr: encoding: an action object or padding is invalid");
                    action_binding.push_back(binding[position]);
                }
                action_schema.push_back(schema);
                action_parent.push_back(index_of(graph));
                append(predicate_relations + schema, arity, [&](u32 position) { return static_cast<u32>(binding[position]); }, action);
            }
        std::stable_sort(tuples.begin(), tuples.end(), [&](const Tuple& first, const Tuple& second)
        {
            if (first.relation != second.relation)
                return first.relation < second.relation;
            const auto first_args = std::span(local_arguments).subspan(first.begin, first.arity);
            const auto second_args = std::span(local_arguments).subspan(second.begin, second.arity);
            return std::lexicographical_compare(first_args.begin(), first_args.end(), second_args.begin(), second_args.end());
        });
        tuples.erase(std::unique(tuples.begin(), tuples.end(), [&](const Tuple& first, const Tuple& second)
        {
            return first.relation == second.relation && first.relation < predicate_relations && first.relation % 5 == 0 &&
                   std::ranges::equal(std::span(local_arguments).subspan(first.begin, first.arity),
                                      std::span(local_arguments).subspan(second.begin, second.arity));
        }), tuples.end());
        auto current = tuples.begin();
        for (u32 relation = 0; relation < relations; ++relation)
        {
            while (current != tuples.end() && current->relation == relation)
            {
                const Tuple& tuple = *current++;
                if (tuple.action >= 0)
                    action_tuple[tuple.action] = index_of(tuple_count);
                tuple_count = static_cast<u32>(index_of(static_cast<u64>(tuple_count) + 1));
                ++counts[static_cast<u64>(graph) * relations + relation];
                for (u32 position = 0; position < tuple.arity; ++position)
                    arguments.push_back(index_of(static_cast<u64>(object_offsets.back()) + local_arguments[tuple.begin + position]));
            }
            offsets.push_back(index_of(arguments.size()));
            tuple_offsets.push_back(index_of(tuple_count));
        }
        object_offsets.push_back(index_of(static_cast<u64>(object_offsets.back()) + task.num_objects()));
        action_offsets.push_back(index_of(action_schema.size()));
        ids.push_back(index_of(instance_id));
        output.task_fingerprints.push_back(task.fingerprint());
        for (const auto& object : task.data().objects)
            output.object_names.emplace_back(task.data().str(object.name));
    }
    rl::ArrayBundle::Builder builder;
    builder.scalar("version", k_flat_version);
    builder.scalar("num_graphs", graphs);
    builder.scalar("num_relations", relations);
    builder.scalar("num_tuples", tuple_count);
    builder.scalar("num_domain_predicates", m_schema->domain_predicates);
    auto add = [&](const char* name, std::vector<i64> shape, const std::vector<i32>& values)
    {
        builder.add(name, rl::DType::I32, std::move(shape), values);
    };
    add("relation_counts", {graphs, relations}, counts);
    add("relation_offsets", {static_cast<i64>(offsets.size())}, offsets);
    add("relation_tuple_offsets", {static_cast<i64>(tuple_offsets.size())}, tuple_offsets);
    add("relation_args", {static_cast<i64>(arguments.size())}, arguments);
    add("object_offsets", {static_cast<i64>(object_offsets.size())}, object_offsets);
    add("task_ids", {graphs}, ids);
    add("action_offsets", {static_cast<i64>(action_offsets.size())}, action_offsets);
    add("action_schema", {static_cast<i64>(action_schema.size())}, action_schema);
    add("action_binding", {static_cast<i64>(action_schema.size()), label_width}, action_binding);
    add("action_parent", {static_cast<i64>(action_parent.size())}, action_parent);
    add("action_tuple", {static_cast<i64>(action_tuple.size())}, action_tuple);
    add("relation_arities", {relations}, m_schema->arities);
    add("relation_predicates", {relations}, m_schema->predicates);
    add("relation_schemas", {relations}, m_schema->schemas);
    add("relation_kinds", {relations}, m_schema->kinds);
    add("relation_instances", {relations}, m_schema->instances);
    add("relation_polarities", {relations}, m_schema->polarities);
    output.arrays = std::make_shared<const rl::ArrayBundle>(builder.build());
    return output;
}
}
