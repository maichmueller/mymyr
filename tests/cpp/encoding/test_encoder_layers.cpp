// The subgoal layers and the history of the relational encoder: placement across goal levels, the history steps and their truncation, the entities and targets of both, the layouts, thread counts,
// capacities, streams, and the inputs that are refused.

#include "support.hpp"

#include "mymyr/encoding/collate.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <random>
#include <set>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
using Tuple = std::vector<i64>;

std::vector<i64> ints(const rl::ArrayBundle& bundle, const char* name)
{
    const rl::ArrayInfo* info = bundle.find(name);
    std::vector<i64> out;
    if (!info)
    {
        ADD_FAILURE() << "missing array " << name;
        return out;
    }
    for (u64 i = 0; i < info->elements(); ++i)
    {
        const std::byte* at = static_cast<const std::byte*>(bundle.data(*info)) + i * rl::dtype_bytes(info->dtype);
        switch (info->dtype)
        {
            case rl::DType::U8: out.push_back(std::to_integer<u8>(*at)); break;
            case rl::DType::I32: { i32 v; std::memcpy(&v, at, 4); out.push_back(v); break; }
            case rl::DType::I64: { i64 v; std::memcpy(&v, at, 8); out.push_back(v); break; }
            default: ADD_FAILURE() << "unexpected dtype of " << name;
        }
    }
    return out;
}

/// One graph of a batch with instance-local entity ids.
struct Graph
{
    std::vector<std::vector<Tuple>> tuples;  // [relation][tuple]
    std::vector<i64> role, ref;              // per entity
    std::vector<i64> history_entity, history_dt;  // the graph's history steps (local entity ids)
    std::vector<i64> target_entity, target_group;  // the graph's targets (local entity ids)
    i64 num_objects = 0;
};

std::vector<Graph> decode(const rl::ArrayBundle& bundle, const EncodingSchema& schema)
{
    const u64 R = schema.num_relations();
    const bool relation_major = schema.options().layout == Layout::Relation;
    const auto entity_offsets = ints(bundle, "entity_offsets");
    const auto num_objects = ints(bundle, "num_objects");
    const auto role = ints(bundle, "entity_role");
    const auto ref = ints(bundle, "entity_ref");
    const auto offsets = ints(bundle, "relation_offsets");
    const auto arg_offsets = ints(bundle, "relation_arg_offsets");
    const auto args = ints(bundle, "relation_args");
    const u64 G = entity_offsets.size() - 1;
    std::vector<i64> history_offsets, history_entity, history_dt, target_offsets, target_entity, target_group;
    if (schema.has_history())
    {
        history_offsets = ints(bundle, "history_offsets");
        history_entity = ints(bundle, "history_entity");
        history_dt = ints(bundle, "history_dt");
    }
    if (schema.has_targets())
    {
        target_offsets = ints(bundle, "target_offsets");
        target_entity = ints(bundle, "target_entity");
        target_group = ints(bundle, "target_group");
    }
    std::vector<Graph> out(G);
    for (u64 g = 0; g < G; ++g)
    {
        Graph& graph = out[g];
        graph.num_objects = num_objects[g];
        graph.tuples.assign(R, {});
        for (i64 e = entity_offsets[g]; e < entity_offsets[g + 1]; ++e)
        {
            graph.role.push_back(role[e]);
            graph.ref.push_back(ref[e]);
        }
        for (u64 r = 0; r < R; ++r)
        {
            const u64 s = relation_major ? r * G + g : g * R + r;
            const u32 arity = schema.relation(static_cast<u32>(r)).arity;
            for (i64 t = 0; t < offsets[s + 1] - offsets[s]; ++t)
            {
                Tuple tuple;
                for (u32 k = 0; k < arity; ++k)
                    tuple.push_back(args[arg_offsets[s] + t * arity + k] - entity_offsets[g]);
                graph.tuples[r].push_back(tuple);
            }
        }
        if (schema.has_history())
            for (i64 j = history_offsets[g]; j < history_offsets[g + 1]; ++j)
            {
                graph.history_entity.push_back(history_entity[static_cast<usize>(j)] - entity_offsets[g]);
                graph.history_dt.push_back(history_dt[static_cast<usize>(j)]);
            }
        if (schema.has_targets())
            for (i64 j = target_offsets[g]; j < target_offsets[g + 1]; ++j)
            {
                graph.target_entity.push_back(target_entity[static_cast<usize>(j)] - entity_offsets[g]);
                graph.target_group.push_back(target_group[static_cast<usize>(j)]);
            }
    }
    return out;
}

std::string predicate_name(const Task& task, u32 predicate)
{
    return std::string(task.data().str(task.data().predicates[predicate].name));
}

/// A literal: predicate name, sign and objects.
struct Literal
{
    std::string predicate;
    bool positive;
    Tuple objects;
    [[nodiscard]] auto key() const { return std::tie(predicate, positive, objects); }
    bool operator<(const Literal& o) const { return key() < o.key(); }
    bool operator==(const Literal& o) const { return key() == o.key(); }
};

using AtomKey = std::pair<std::string, Tuple>;

/// The true atoms of a row, by predicate name: fluent and derived ones.
std::set<AtomKey> true_atoms(const rl::TaskTable& table, u32 instance, const u64* row)
{
    const Task& task = *table.task(instance);
    std::set<AtomKey> out;
    auto add = [&](const u32* record)
    {
        Tuple objects;
        for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
            objects.push_back(record[1 + k]);
        out.insert({predicate_name(task, record[0]), std::move(objects)});
    };
    for (u32 slot = 0; slot < task.atoms().fluent_slots(); ++slot)
        if ((row[slot >> 6] >> (slot & 63)) & 1)
            add(task.atoms().record(AtomKind::Fluent, slot));
    if (task.atoms().derived_slots() > 0)
    {
        const WorkspaceLease workspace = task.workspace();
        auto& engine = workspace->engine();
        engine.set_state(row, table.instance(instance).words);
        engine.build_view();
        workspace->axioms().evaluate();
        const u64* derived = engine.derived();
        for (u32 slot = 0; slot < task.atoms().derived_slots(); ++slot)
            if ((derived[slot >> 6] >> (slot & 63)) & 1)
                add(task.atoms().record(AtomKind::Derived, slot));
    }
    return out;
}

std::vector<Literal> encoding_goal(const Task& task)
{
    std::vector<Literal> out;
    for (const formalism::Literal& literal : task.encoding_goal())
    {
        Literal l{predicate_name(task, literal.pred.v), literal.positive, {}};
        for (const auto term : task.data().terms_of(literal))
            l.objects.push_back(formalism::term_object(term).v);
        out.push_back(std::move(l));
    }
    return out;
}

bool is_derived_predicate(const Task& task, const std::string& name)
{
    for (u32 p = 0; p < task.data().predicates.size(); ++p)
        if (predicate_name(task, p) == name)
            return task.data().predicates[p].kind == formalism::PredKind::Derived;
    return false;
}

/// Masks of a batch of layers: per row, `layers` masks of `words` words (and `derived_words` over the derived slots).
struct Masks
{
    std::vector<u64> pos, neg, derived_pos, derived_neg;
    u32 layers = 0, words = 0, derived_words = 0;
};

Masks make_masks(const rl::TaskTable& table, const std::vector<u32>& instances,
                 const std::vector<std::vector<std::vector<Literal>>>& rows)
{
    Masks m;
    m.words = table.words();
    m.derived_words = std::max(1u, table.derived_words());
    for (const auto& row : rows)
        m.layers = std::max<u32>(m.layers, static_cast<u32>(row.size()));
    const u64 n = rows.size();
    m.pos.assign(n * m.layers * m.words, 0);
    m.neg = m.pos;
    m.derived_pos.assign(n * m.layers * m.derived_words, 0);
    m.derived_neg = m.derived_pos;
    for (u64 r = 0; r < n; ++r)
    {
        const Task& task = *table.task(instances[r]);
        for (u32 l = 0; l < rows[r].size(); ++l)
            for (const Literal& literal : rows[r][l])
            {
                const bool derived = is_derived_predicate(task, literal.predicate);
                const u32 slot = test::atom_slot(task, derived ? AtomKind::Derived : AtomKind::Fluent,
                                                 literal.predicate, literal.objects);
                std::vector<u64>& words = derived ? (literal.positive ? m.derived_pos : m.derived_neg)
                                                  : (literal.positive ? m.pos : m.neg);
                const u32 width = derived ? m.derived_words : m.words;
                words[(r * m.layers + l) * width + (slot >> 6)] |= u64{1} << (slot & 63);
            }
    }
    return m;
}

LayerInput layer_input(const Masks& m)
{
    LayerInput in;
    in.pos = m.pos.data();
    in.neg = m.neg.data();
    in.layers = m.layers;
    in.words = m.words;
    in.stride = u64{m.layers} * m.words;
    in.derived_pos = m.derived_pos.data();
    in.derived_neg = m.derived_neg.data();
    in.derived_words = m.derived_words;
    in.derived_stride = u64{m.layers} * m.derived_words;
    return in;
}

/// The history of the rows: steps of (dt, literals), oldest first.
using Steps = std::vector<std::pair<i32, std::vector<Literal>>>;

struct History
{
    Masks masks;
    std::vector<i32> dt, count;
    [[nodiscard]] HistoryInput input() const
    {
        HistoryInput in;
        in.masks = layer_input(masks);
        in.dt = dt.data();
        in.count = count.data();
        return in;
    }
};

History make_history(const rl::TaskTable& table, const std::vector<u32>& instances, const std::vector<Steps>& rows)
{
    std::vector<std::vector<std::vector<Literal>>> layers;
    for (const Steps& steps : rows)
    {
        layers.emplace_back();
        for (const auto& step : steps)
            layers.back().push_back(step.second);
    }
    History h;
    h.masks = make_masks(table, instances, layers);
    h.dt.assign(rows.size() * h.masks.layers, 0);
    for (u64 r = 0; r < rows.size(); ++r)
    {
        h.count.push_back(static_cast<i32>(rows[r].size()));
        for (u32 j = 0; j < rows[r].size(); ++j)
            h.dt[r * h.masks.layers + j] = rows[r][j].first;
    }
    return h;
}

EncodeInput input_of(const test::Rows& rows)
{
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    return in;
}

std::string level_suffix(u32 level)
{
    return level == 0 ? "[g]" : level == 1 ? "[sg]" : level == 2 ? "[ssg]" : "[sssg]";
}

using Named = std::map<std::string, std::vector<Tuple>>;

/// The subgoal relations of a row, placed by the schema's policy, with variants for their truth in `atoms`.
Named subgoal_reference(const EncodingSchema& schema, i64 num_objects, const std::vector<Literal>& goal,
                        const std::vector<std::vector<Literal>>& layers, const std::set<AtomKey>& atoms)
{
    Named out;
    std::set<Literal> seen(goal.begin(), goal.end());
    for (u32 l = 0; l < layers.size(); ++l)
    {
        std::set<Literal> level;
        for (const Literal& literal : layers[l])
        {
            bool include = true;
            if (schema.options().repeated_literals == RepeatedLiterals::Lowest)
                include = !seen.count(literal);
            else if (schema.options().repeated_literals == RepeatedLiterals::Highest)
                for (u32 higher = l + 1; higher < layers.size(); ++higher)
                    if (std::find(layers[higher].begin(), layers[higher].end(), literal) != layers[higher].end())
                        include = false;
            if (include)
                level.insert(literal);
        }
        seen.insert(layers[l].begin(), layers[l].end());
        for (const Literal& literal : level)
        {
            const bool holds = (atoms.count({literal.predicate, literal.objects}) != 0) == literal.positive;
            const std::string base =
                std::string(literal.positive ? "[+]" : "[-]") + literal.predicate + level_suffix(l + 1);
            const Tuple tuple = literal.objects.empty() ? Tuple{num_objects} : literal.objects;
            for (const std::string& name : {base, base + (holds ? "[sat]" : "[unsat]")})
                if (schema.find(name))
                    out[name].push_back(tuple);
        }
    }
    for (auto& entry : out)
        std::sort(entry.second.begin(), entry.second.end());
    return out;
}

/// The history relations of a row: for each used step in order, the literals of the step by relation, ascending.
struct HistoryTuple
{
    i64 step;
    Tuple objects;
    bool operator<(const HistoryTuple& o) const { return std::tie(step, objects) < std::tie(o.step, o.objects); }
    bool operator==(const HistoryTuple& o) const = default;
};
using HistoryNamed = std::map<std::string, std::vector<HistoryTuple>>;

HistoryNamed history_reference(const EncodingSchema& schema, i64 num_objects, const Steps& steps, u32 max_steps)
{
    HistoryNamed out;
    const usize used = max_steps == 0 ? steps.size() : std::min<usize>(steps.size(), max_steps);
    const usize first = steps.size() - used;
    for (usize j = 0; j < used; ++j)
    {
        std::set<Literal> literals(steps[first + j].second.begin(), steps[first + j].second.end());
        for (const Literal& literal : literals)
        {
            const std::string name = std::string(literal.positive ? "[+]" : "[-]") + literal.predicate + "[hist]";
            if (schema.find(name))
                out[name].push_back(
                    {static_cast<i64>(j), literal.objects.empty() ? Tuple{num_objects} : literal.objects});
        }
    }
    for (auto& entry : out)
        std::sort(entry.second.begin(), entry.second.end());
    return out;
}

/// Checks the subgoal relations, entities and targets of a graph.
void expect_subgoals(const EncodingSchema& schema, const Graph& graph, const Named& expected, const char* where)
{
    i64 subgoal_entities = 0;
    for (u32 r = 0; r < schema.num_relations(); ++r)
    {
        const Relation& relation = schema.relation(r);
        const bool level =
            (relation.lane == Lane::Goal || relation.lane == Lane::GoalDerivation) && relation.level.value_or(0) > 0;
        if (!level)
            continue;
        const bool target = schema.slot_roles(r).front() == SlotRole::SubgoalTarget;
        std::vector<Tuple> got;
        for (Tuple tuple : graph.tuples[r])
        {
            if (target)
            {
                EXPECT_EQ(graph.role[static_cast<usize>(tuple.front())], 3) << where << " " << relation.name;
                ++subgoal_entities;
                tuple.erase(tuple.begin());
            }
            got.push_back(tuple);
        }
        const auto found = expected.find(relation.name);
        const std::vector<Tuple> want = found == expected.end() ? std::vector<Tuple>{} : found->second;
        EXPECT_EQ(got, want) << where << " relation " << relation.name;
    }
    const i64 role_count = std::count(graph.role.begin(), graph.role.end(), 3);
    EXPECT_EQ(role_count, subgoal_entities) << where;
}

struct Counts
{
    u64 subgoal_tuples = 0, history_tuples = 0;
};

FlatOptions layer_options(u32 levels = 2, bool history = true, u32 max_steps = 0, Layout layout = Layout::Graph)
{
    FlatOptions options;
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    options.max_goal_level = levels;
    options.targets = {TargetSource::Goal};
    if (levels > 0)
        options.targets.insert(TargetSource::Subgoal);
    if (history)
    {
        options.history = true;
        if (max_steps != 0)
            options.history_max_steps = max_steps;
        options.targets.insert(TargetSource::History);
    }
    options.include_actions = false;
    options.layout = layout;
    return options;
}

Literal on(const char* p, bool positive, Tuple objects) { return {p, positive, std::move(objects)}; }

// ------------------------------------------------------------------------------------------------ subgoals

// blocks smedium: the objects are b a c = 0 1 2; the state is (clear a) (on a b) (on b c) (ontable c) (handempty); the
// goal is (on c b) (on b a)
TEST(EncoderLayers, SubgoalLevelsKeepEachLayer)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);  // the initial state
    const std::vector<std::vector<Literal>> layers = {
        {on("on", true, {2, 0}), on("on", true, {1, 0}), on("clear", true, {1}), on("clear", false, {0}),
         on("on", false, {0, 1})},
        {on("on", true, {1, 0}), on("clear", true, {1}), on("ontable", true, {2}), on("on", true, {0, 2}),
         on("clear", false, {0}), on("on", true, {0, 1})}};
    const Masks masks = make_masks(*table, {0}, {layers});
    const auto encoder = FlatEncoder::create(table, layer_options(2, false));
    EncodeInput in = input_of(rows);
    in.subgoals = layer_input(masks);
    const auto bundle = encoder->encode(in);
    const EncodingSchema& schema = encoder->schema();
    validate(*bundle, schema);
    const Graph g = decode(*bundle, schema)[0];
    auto relation = [&](const char* name) { return g.tuples[*schema.find(name)]; };
    auto bare = [](std::vector<Tuple> tuples)
    {
        for (Tuple& t : tuples)
            t.erase(t.begin());
        return tuples;
    };
    EXPECT_EQ(bare(relation("[+]on[g]")), (std::vector<Tuple>{{0, 1}, {2, 0}}));
    EXPECT_EQ(bare(relation("[+]on[sg]")), (std::vector<Tuple>{{1, 0}, {2, 0}}));
    EXPECT_EQ(bare(relation("[+]clear[sg]")), (std::vector<Tuple>{{1}}));
    EXPECT_EQ(bare(relation("[-]clear[sg]")), (std::vector<Tuple>{{0}}));
    EXPECT_EQ(bare(relation("[-]on[sg]")), (std::vector<Tuple>{{0, 1}}));
    EXPECT_EQ(bare(relation("[+]on[ssg]")), (std::vector<Tuple>{{0, 1}, {0, 2}, {1, 0}}));
    EXPECT_EQ(bare(relation("[+]ontable[ssg]")), (std::vector<Tuple>{{2}}));
    EXPECT_EQ(bare(relation("[+]clear[ssg]")), (std::vector<Tuple>{{1}}));
    EXPECT_EQ(bare(relation("[-]clear[ssg]")), (std::vector<Tuple>{{0}}));
    // (on a b), (clear a), (on b c), (ontable c) hold in the state; (not (clear b)) and (not (on b a)) hold; (on c b)
    // and (on b a) do not
    EXPECT_EQ(relation("[+]on[sg][sat]"), (std::vector<Tuple>{{1, 0}}));
    EXPECT_EQ(relation("[+]on[sg][unsat]"), (std::vector<Tuple>{{2, 0}}));
    EXPECT_EQ(relation("[-]on[sg][sat]"), bare(relation("[-]on[sg]")));
    EXPECT_EQ(relation("[-]clear[sg][sat]"), bare(relation("[-]clear[sg]")));
    EXPECT_EQ(relation("[+]on[ssg][sat]"), (std::vector<Tuple>{{0, 2}, {1, 0}}));
    EXPECT_EQ(relation("[+]on[ssg][unsat]"), (std::vector<Tuple>{{0, 1}}));
    // 3 objects, the auxiliary entity ((handempty) holds in the state), the goal entities 2, the subgoal entities:
    // level 1 has 5 tuples, level 2 has 6
    EXPECT_EQ(g.role.size(), 3u + 1u + 2u + 11u);
    EXPECT_EQ(std::count(g.role.begin(), g.role.end(), 3), 11);
    // the subgoal entities follow the goal entities, in the order of their relations (sorted by name)
    EXPECT_EQ(g.target_group, (std::vector<i64>{0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    for (u64 k = 0; k < g.target_entity.size(); ++k)
        EXPECT_EQ(g.target_entity[k], 4 + static_cast<i64>(k));
}

TEST(EncoderLayers, RepeatedGoalLiteralsFollowThePlacementPolicy)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);
    const Masks masks = make_masks(*table, {0}, {{{on("on", true, {2, 0}), on("on", true, {2, 0})}, {},
                                               {on("on", true, {2, 0})}}});
    for (const auto policy : {RepeatedLiterals::Every, RepeatedLiterals::Lowest, RepeatedLiterals::Highest})
        for (const Layout layout : {Layout::Graph, Layout::Relation})
        {
            SCOPED_TRACE(to_string(policy));
            auto options = layer_options(3, false, 0, layout);
            options.repeated_literals = policy;
            const auto encoder = FlatEncoder::create(table, options);
            EncodeInput in = input_of(rows);
            in.subgoals = layer_input(masks);
            const auto bundle = encoder->encode(in);
            const EncodingSchema& schema = encoder->schema();
            validate(*bundle, schema);
            const Graph g = decode(*bundle, schema)[0];
            auto count = [&](const char* name) { return g.tuples[*schema.find(name)].size(); };
            EXPECT_EQ(count("[+]on[g]"), policy == RepeatedLiterals::Highest ? 1u : 2u);
            EXPECT_EQ(count("[+]on[sg]"), policy == RepeatedLiterals::Every ? 1u : 0u);
            EXPECT_EQ(count("[+]on[ssg]"), 0u);
            EXPECT_EQ(count("[+]on[sssg]"), policy == RepeatedLiterals::Lowest ? 0u : 1u);
            EXPECT_EQ(count("[+]on[sg][sat]"), 0u);
            EXPECT_EQ(count("[+]on[sg][unsat]"), count("[+]on[sg]"));
            EXPECT_EQ(count("[+]on[sssg][unsat]"), count("[+]on[sssg]"));
            EXPECT_EQ(ints(*bundle, "goal_complete"), (std::vector<i64>{1}));
            EXPECT_EQ(g.target_group.size(), policy == RepeatedLiterals::Every ? 4u : 2u);
            for (usize i = 0; i < g.target_entity.size(); ++i)
                EXPECT_EQ(g.target_entity[i], 4 + static_cast<i64>(i));
        }
}

TEST(EncoderLayers, FewerLayersThanLevelsAndNoLayersAreTheSameAsEmptyLayers)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 4, 2);
    const auto encoder = FlatEncoder::create(table, layer_options(3, false));
    const auto plain = encoder->encode(input_of(rows));
    // one layer of nothing, three empty layers: the bytes of an encoding without layers
    for (const u32 layers : {0u, 1u, 3u})
    {
        const Masks masks = make_masks(*table, {0, 0, 0, 0}, {{}, {}, {}, {}});
        Masks sized = masks;
        sized.layers = layers;
        sized.pos.assign(4 * layers * sized.words, 0);
        sized.neg = sized.pos;
        sized.derived_pos.assign(4 * layers * sized.derived_words, 0);
        sized.derived_neg = sized.derived_pos;
        EncodeInput in = input_of(rows);
        in.subgoals = layer_input(sized);
        const auto bundle = encoder->encode(in);
        ASSERT_EQ(bundle->bytes(), plain->bytes());
        EXPECT_EQ(std::memcmp(bundle->block(), plain->block(), bundle->bytes()), 0) << layers;
    }
}

/// Random subgoal layers over the fluent atoms (and derived ones for wide_predicates) of a task, per row.
std::vector<std::vector<std::vector<Literal>>> random_layers(const Task& task, u64 rows, u32 levels, u64 seed)
{
    std::vector<Literal> pool;
    for (u32 slot = 0; slot < task.atoms().fluent_slots(); ++slot)
    {
        const u32* record = task.atoms().record(AtomKind::Fluent, slot);
        Tuple objects;
        for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
            objects.push_back(record[1 + k]);
        for (const bool positive : {true, false})
            pool.push_back({predicate_name(task, record[0]), positive, objects});
    }
    for (u32 slot = 0; slot < task.atoms().derived_slots(); ++slot)
    {
        const u32* record = task.atoms().record(AtomKind::Derived, slot);
        Tuple objects;
        for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
            objects.push_back(record[1 + k]);
        for (const bool positive : {true, false})
            pool.push_back({predicate_name(task, record[0]), positive, objects});
    }
    std::mt19937_64 rng(seed);
    std::vector<std::vector<std::vector<Literal>>> out(rows);
    for (auto& row : out)
    {
        row.resize(rng() % (levels + 1));
        for (auto& layer : row)
            for (u64 n = rng() % 9; n > 0; --n)
                layer.push_back(pool[rng() % pool.size()]);
    }
    return out;
}

/// Random histories: per row up to `max_steps` steps with a random dt and a layer of random literals.
std::vector<Steps> random_steps(const Task& task, u64 rows, u32 max_steps, u64 seed)
{
    std::mt19937_64 rng(seed);
    std::vector<Steps> out(rows);
    for (Steps& steps : out)
    {
        const u32 count = static_cast<u32>(rng() % (max_steps + 1));
        const auto layers = random_layers(task, count, 1, rng());
        for (u32 j = 0; j < count; ++j)
            steps.push_back(
                {static_cast<i32>(rng() % 9) - 8, layers[j].empty() ? std::vector<Literal>{} : layers[j][0]});
    }
    return out;
}

struct SubgoalProblem
{
    const char* domain;
    const char* problem;
};

TEST(EncoderLayers, RandomLayersMatchTheSetReferenceInBothLayouts)
{
    u64 tuples = 0;
    for (const SubgoalProblem& p : {SubgoalProblem{"blocks", "smedium"}, SubgoalProblem{"wide_predicates", "p01"},
                                     SubgoalProblem{"gripper", "p01"}, SubgoalProblem{"features", "p01"}})
        for (const Layout layout : {Layout::Graph, Layout::Relation})
            for (const auto policy : {RepeatedLiterals::Every, RepeatedLiterals::Lowest, RepeatedLiterals::Highest})
            {
                SCOPED_TRACE(to_string(policy));
                SCOPED_TRACE(std::string(p.domain) + (layout == Layout::Graph ? " graph" : " relation"));
                const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
                const test::Rows rows = test::random_walk(*table, 0, 40, 5);
                const auto layers = random_layers(*table->task(0), rows.size(), 3, 17);
                const Masks masks = make_masks(*table, std::vector<u32>(rows.size(), 0), layers);
                FlatOptions options = layer_options(3, false, 0, layout);
                options.repeated_literals = policy;
                const auto encoder = FlatEncoder::create(table, options);
                EncodeInput in = input_of(rows);
                in.subgoals = layer_input(masks);
                const auto bundle = encoder->encode(in);
                validate(*bundle, encoder->schema());
                const auto graphs = decode(*bundle, encoder->schema());
                for (u64 g = 0; g < rows.size(); ++g)
                {
                    const Named expected = subgoal_reference(
                        encoder->schema(), graphs[g].num_objects, encoding_goal(*table->task(0)), layers[g],
                        true_atoms(*table, 0, rows.data.data() + g * rows.words));
                    expect_subgoals(encoder->schema(), graphs[g], expected, "row");
                    for (const auto& entry : expected)
                        tuples += entry.second.size();
                }
            }
    EXPECT_GT(tuples, 2000u);
}

TEST(EncoderLayers, LayersSharedByEveryRowAreOneSetOfMasks)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 6, 3);
    const std::vector<std::vector<Literal>> layers = {{on("clear", true, {1}), on("handempty", true, {})},
                                                      {on("ontable", true, {2})}};
    const Masks one = make_masks(*table, {0}, {layers});
    const Masks many =
        make_masks(*table, std::vector<u32>(6, 0), std::vector<std::vector<std::vector<Literal>>>(6, layers));
    const auto encoder = FlatEncoder::create(table, layer_options(2, false));
    EncodeInput shared = input_of(rows), repeated = input_of(rows);
    shared.subgoals = layer_input(one);
    shared.subgoals.stride = 0;
    shared.subgoals.derived_stride = 0;
    repeated.subgoals = layer_input(many);
    const auto a = encoder->encode(shared);
    const auto b = encoder->encode(repeated);
    ASSERT_EQ(a->bytes(), b->bytes());
    EXPECT_EQ(std::memcmp(a->block(), b->block(), a->bytes()), 0);
}

TEST(EncoderLayers, NullaryLiteralsInALayerAreTuplesOverTheAuxiliaryEntity)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);  // (handempty) holds
    const Masks masks = make_masks(*table, {0}, {{{on("handempty", true, {})}}});
    // the state has no nullary tuple of its own: without nullary relations the state has no auxiliary entity
    FlatOptions options = layer_options(1, false);
    options.nullary = Nullary::Auxiliary;
    const auto encoder = FlatEncoder::create(table, options);
    EncodeInput in = input_of(rows);
    in.subgoals = layer_input(masks);
    const auto bundle = encoder->encode(in);
    const EncodingSchema& schema = encoder->schema();
    validate(*bundle, schema);
    const Graph g = decode(*bundle, schema)[0];
    // (handempty) is a state tuple: the auxiliary entity is the entity 3; the subgoal entity follows the goal's
    EXPECT_EQ(g.role[3], 8);
    const u32 sg = *schema.find("[+]handempty[sg]");
    ASSERT_EQ(g.tuples[sg].size(), 1u);
    // the subgoal entity comes after 3 objects, the auxiliary entity and 2 goal entities
    EXPECT_EQ(g.tuples[sg][0], (Tuple{3 + 1 + 2, 3}));
    EXPECT_EQ(g.tuples[*schema.find("[+]handempty[sg][sat]")], (std::vector<Tuple>{{3}}));
    options.nullary = Nullary::Omit;
    const auto without = FlatEncoder::create(table, options);
    EXPECT_EQ(without->schema().find("[+]handempty[sg]"), std::nullopt);
}

// ------------------------------------------------------------------------------------------------ history

TEST(EncoderLayers, HistoryStepsAreEntitiesInStepOrderAndTheirLiteralsTuples)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);
    const Steps steps = {{-2, {on("on", true, {1, 0}), on("clear", true, {1}), on("handempty", true, {})}},
                         {-1, {on("clear", false, {0}), on("clear", true, {1})}},
                         {0, {on("on", true, {0, 2}), on("on", true, {1, 0})}}};
    const History history = make_history(*table, {0}, {steps});
    const auto encoder = FlatEncoder::create(table, layer_options(0, true));
    EncodeInput in = input_of(rows);
    in.history = history.input();
    const auto bundle = encoder->encode(in);
    const EncodingSchema& schema = encoder->schema();
    validate(*bundle, schema);
    const Graph g = decode(*bundle, schema)[0];
    // entities: 3 objects, the auxiliary entity (handempty in a step, and in the state), 2 goal entities, 3 history
    // steps, and a history target per literal: 3 + 2 + 2 = 7
    ASSERT_EQ(g.role.size(), 3u + 1u + 2u + 3u + 7u);
    EXPECT_EQ(g.history_dt, (std::vector<i64>{-2, -1, 0}));
    const i64 step0 = 3 + 1 + 2;
    EXPECT_EQ(g.history_entity, (std::vector<i64>{step0, step0 + 1, step0 + 2}));
    for (i64 j = 0; j < 3; ++j)
    {
        EXPECT_EQ(g.role[static_cast<usize>(step0 + j)], 5);
        EXPECT_EQ(g.ref[static_cast<usize>(step0 + j)], j);
    }
    // history targets in relation order (relations sorted by name), then in step order: [+]clear, [+]handempty, [+]on,
    // [-]clear (sorted '[+' before '[-')
    auto relation = [&](const char* name) { return g.tuples[*schema.find(name)]; };
    const i64 target0 = step0 + 3;
    EXPECT_EQ(relation("[+]clear[hist]"), (std::vector<Tuple>{{target0, step0, 1}, {target0 + 1, step0 + 1, 1}}));
    EXPECT_EQ(relation("[+]handempty[hist]"), (std::vector<Tuple>{{target0 + 2, step0, 3}}));
    EXPECT_EQ(relation("[+]on[hist]"),
              (std::vector<Tuple>{
                  {target0 + 3, step0, 1, 0}, {target0 + 4, step0 + 2, 0, 2}, {target0 + 5, step0 + 2, 1, 0}}));
    EXPECT_EQ(relation("[-]clear[hist]"), (std::vector<Tuple>{{target0 + 6, step0 + 1, 0}}));
    for (i64 k = 0; k < 7; ++k)
    {
        EXPECT_EQ(g.role[static_cast<usize>(target0 + k)], 6);
        EXPECT_EQ(g.ref[static_cast<usize>(target0 + k)] >= 0, true);
    }
    // targets: the two goal entities, then the history targets
    EXPECT_EQ(g.target_group, (std::vector<i64>{0, 0, 3, 3, 3, 3, 3, 3, 3}));
}

TEST(EncoderLayers, TheNewestStepsAreKeptAndEntityRefsCountTheValidSteps)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 3, 1);
    const Steps long_history = {{-4, {on("clear", true, {0})}}, {-3, {on("clear", true, {1})}},
                                {-2, {on("clear", true, {2})}}, {-1, {on("ontable", true, {2})}},
                                {0, {on("ontable", true, {1})}}};
    const std::vector<Steps> per_row = {long_history, {}, {long_history.begin(), long_history.begin() + 2}};
    const History history = make_history(*table, {0, 0, 0}, per_row);
    for (const u32 k : {0u, 1u, 2u, 3u, 9u})
    {
        SCOPED_TRACE(k);
        const auto encoder = FlatEncoder::create(table, layer_options(0, true, k));
        EncodeInput in = input_of(rows);
        in.history = history.input();
        const auto bundle = encoder->encode(in);
        const EncodingSchema& schema = encoder->schema();
        validate(*bundle, schema);
        const auto graphs = decode(*bundle, schema);
        for (u64 r = 0; r < 3; ++r)
        {
            const u32 valid = static_cast<u32>(per_row[r].size());
            const u32 used = k == 0 ? valid : std::min(valid, k);
            ASSERT_EQ(graphs[r].history_dt.size(), used) << r;
            const i64 first_step = graphs[r].num_objects + 1 + 2;  // objects, the auxiliary entity, 2 goal entities
            for (u32 j = 0; j < used; ++j)
            {
                // the newest `used` steps of the valid ones, oldest first; the ref is the index among the valid steps
                EXPECT_EQ(graphs[r].history_dt[j], per_row[r][valid - used + j].first);
                EXPECT_EQ(graphs[r].ref[static_cast<usize>(first_step + j)], static_cast<i64>(valid - used + j));
            }
            const HistoryNamed expected = history_reference(schema, graphs[r].num_objects, per_row[r], k);
            for (u32 id = 0; id < schema.num_relations(); ++id)
            {
                if (schema.relation(id).lane != Lane::History)
                    continue;
                std::vector<HistoryTuple> got;
                for (const Tuple& t : graphs[r].tuples[id])
                    got.push_back({t[1] - first_step, Tuple(t.begin() + 2, t.end())});
                const auto found = expected.find(schema.relation(id).name);
                EXPECT_EQ(got, found == expected.end() ? std::vector<HistoryTuple>{} : found->second)
                    << r << " " << schema.relation(id).name;
            }
        }
    }
}

TEST(EncoderLayers, AHistoryWithoutACountHasAllItsLayersValid)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 2, 1);
    const Steps steps = {{-1, {on("clear", true, {0})}}, {0, {on("clear", true, {1})}}};
    const History history = make_history(*table, {0, 0}, {steps, steps});
    const auto encoder = FlatEncoder::create(table, layer_options(0, true));
    EncodeInput counted = input_of(rows), all = input_of(rows);
    counted.history = history.input();
    all.history = history.input();
    all.history.count = nullptr;
    const auto a = encoder->encode(counted);
    const auto b = encoder->encode(all);
    ASSERT_EQ(a->bytes(), b->bytes());
    EXPECT_EQ(std::memcmp(a->block(), b->block(), a->bytes()), 0);
}

TEST(EncoderLayers, HistoryTargetsAreNumberedInFillOrder)
{
    // the history targets of a graph follow its other entities: relations in catalog order (sorted by name), within a
    // relation by step, then slot
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);
    const Steps steps = {{-1, {on("ontable", true, {2}), on("clear", true, {1})}}, {0, {on("clear", true, {0})}}};
    const History history = make_history(*table, {0}, {steps});
    FlatOptions options = layer_options(0, true);
    options.targets = {TargetSource::History};
    const auto encoder = FlatEncoder::create(table, options);
    EncodeInput in = input_of(rows);
    in.history = history.input();
    const auto bundle = encoder->encode(in);
    const EncodingSchema& schema = encoder->schema();
    validate(*bundle, schema);
    const Graph g = decode(*bundle, schema)[0];
    const i64 targets = 3 + 1;  // objects, the auxiliary entity; no goal relations: the steps start here
    const i64 step0 = targets;
    const i64 target0 = step0 + 2;
    EXPECT_EQ(g.tuples[*schema.find("[+]clear[hist]")],
              (std::vector<Tuple>{{target0, step0, 1}, {target0 + 1, step0 + 1, 0}}));
    EXPECT_EQ(g.tuples[*schema.find("[+]ontable[hist]")], (std::vector<Tuple>{{target0 + 2, step0, 2}}));
    EXPECT_EQ(g.target_entity, (std::vector<i64>{target0, target0 + 1, target0 + 2}));
}

TEST(EncoderLayers, HistoryOfEveryLengthEqualsOneRowAtATimeAndEveryLayoutAndThreadCount)
{
    for (const SubgoalProblem& p : {SubgoalProblem{"blocks", "smedium"}, SubgoalProblem{"wide_predicates", "p01"}})
    {
        SCOPED_TRACE(p.domain);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const u64 n = 300;  // more graphs than one member's share
        const test::Rows rows = test::random_walk(*table, 0, n, 7);
        const std::vector<Steps> per_row = random_steps(*table->task(0), n, 5, 3);
        const std::vector<u32> instances(n, 0);
        const History history = make_history(*table, instances, per_row);
        for (const u32 k : {0u, 2u})
        {
            std::vector<std::shared_ptr<const rl::ArrayBundle>> bundles;
            for (const Layout layout : {Layout::Graph, Layout::Relation})
            {
                const auto encoder = FlatEncoder::create(table, layer_options(0, true, k, layout));
                EncodeInput in = input_of(rows);
                in.history = history.input();
                const auto reference = encoder->encode(in);
                validate(*reference, encoder->schema());
                bundles.push_back(reference);
                for (const u32 threads : {2u, 3u, 5u})
                {
                    ThreadPool pool(threads);
                    EncodeScratch scratch;
                    const auto bundle = encoder->encode(in, pool, scratch);
                    ASSERT_EQ(bundle->bytes(), reference->bytes());
                    EXPECT_EQ(std::memcmp(bundle->block(), reference->block(), bundle->bytes()), 0) << threads;
                }
                // one row at a time gives the same graphs
                const auto graphs = decode(*reference, encoder->schema());
                for (u64 g = 0; g < n; g += 37)
                {
                    test::Rows one;
                    one.words = rows.words;
                    one.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(g * rows.words),
                                    rows.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * rows.words));
                    one.task_ids = {0};
                    const History single = make_history(*table, {0}, {per_row[g]});
                    EncodeInput row = input_of(one);
                    row.history = single.input();
                    const auto alone = decode(*encoder->encode(row), encoder->schema());
                    EXPECT_EQ(alone[0].tuples, graphs[g].tuples) << g;
                    EXPECT_EQ(alone[0].role, graphs[g].role) << g;
                    EXPECT_EQ(alone[0].ref, graphs[g].ref) << g;
                    EXPECT_EQ(alone[0].history_dt, graphs[g].history_dt) << g;
                    EXPECT_EQ(alone[0].history_entity, graphs[g].history_entity) << g;
                    EXPECT_EQ(alone[0].target_entity, graphs[g].target_entity) << g;
                }
            }
            // the layouts hold the same graphs
            const FlatOptions graph_options = layer_options(0, true, k, Layout::Graph);
            const FlatOptions relation_options = layer_options(0, true, k, Layout::Relation);
            const auto ga = decode(*bundles[0], FlatEncoder::create(table, graph_options)->schema());
            const auto gb = decode(*bundles[1], FlatEncoder::create(table, relation_options)->schema());
            ASSERT_EQ(ga.size(), gb.size());
            for (u64 g = 0; g < ga.size(); ++g)
            {
                EXPECT_EQ(ga[g].tuples, gb[g].tuples);
                EXPECT_EQ(ga[g].role, gb[g].role);
                EXPECT_EQ(ga[g].ref, gb[g].ref);
                EXPECT_EQ(ga[g].history_entity, gb[g].history_entity);
            }
        }
    }
}

TEST(EncoderLayers, SubgoalsAndHistoryTogetherWithActionsAndVirtualNodes)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 20, 4);
    const auto layers = random_layers(*table->task(0), rows.size(), 2, 5);
    const Masks masks = make_masks(*table, std::vector<u32>(rows.size(), 0), layers);
    const std::vector<Steps> per_row = random_steps(*table->task(0), rows.size(), 3, 2);
    const History history = make_history(*table, std::vector<u32>(rows.size(), 0), per_row);
    rl::Expansion actions;
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    {
        actions.words = rows.words;
        actions.capacity = 1 << 12;
        actions.label_width = table->label_width();
        succ.assign(actions.capacity * rows.words, 0);
        parent.assign(actions.capacity, 0);
        schema.assign(actions.capacity, 0);
        binding.assign(actions.capacity * table->label_width(), 0);
        offsets.assign(rows.size() + 1, 0);
        actions.succ = succ.data();
        actions.parent = parent.data();
        actions.schema = schema.data();
        actions.binding = binding.data();
        actions.offsets = offsets.data();
        rl::expand(*table, rows.view(), rows.task_ids.data(), actions);
        ASSERT_FALSE(actions.overflow());
    }
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions options = layer_options(2, true, 3, layout);
        options.include_actions = true;
        options.predicate_virtual_nodes = true;
        options.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::Action, TargetSource::History};
        const auto encoder = FlatEncoder::create(table, options);
        EncodeInput in = input_of(rows);
        in.actions = &actions;
        in.subgoals = layer_input(masks);
        in.history = history.input();
        const auto bundle = encoder->encode(in);
        validate(*bundle, encoder->schema());
        const auto graphs = decode(*bundle, encoder->schema());
        u64 with_everything = 0;
        for (const Graph& g : graphs)
        {
            std::set<i64> roles(g.role.begin(), g.role.end());
            with_everything += roles.count(1) && roles.count(2) && roles.count(3) && roles.count(4) && roles.count(5) &&
                               roles.count(6);
        }
        EXPECT_GT(with_everything, 0u);
    }
}

TEST(EncoderLayers, TheCapacityFormKeepsTheHistoryArrays)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 5, 2);
    const Steps steps = {{-1, {on("clear", true, {0})}}, {0, {on("clear", true, {1}), on("ontable", true, {2})}}};
    const History history = make_history(*table, std::vector<u32>(5, 0), std::vector<Steps>(5, steps));
    const auto encoder = FlatEncoder::create(table, layer_options(0, true));
    EncodeInput in = input_of(rows);
    in.history = history.input();
    const auto exact = encoder->encode(in);
    ThreadPool pool(1);
    EncodeScratch scratch;
    FlatSizes sizes;
    const u64 entries = ints(*exact, "history_entity").size();
    ASSERT_EQ(entries, 10u);
    BatchShape fits{5, ints(*exact, "entity_ref").size() + 3, ints(*exact, "relation_args").size() + 7,
                    ints(*exact, "target_entity").size() + 2, entries + 4};
    const auto padded = encoder->encode_capacity(in, fits, pool, scratch, sizes);
    EXPECT_FALSE(sizes.overflowed());
    EXPECT_EQ(sizes.history, entries);
    validate(*padded, encoder->schema());
    const auto history_entity = ints(*padded, "history_entity");
    const auto history_dt = ints(*padded, "history_dt");
    ASSERT_EQ(history_entity.size(), entries + 4);
    EXPECT_EQ(std::vector<i64>(history_entity.begin(), history_entity.begin() + 10), ints(*exact, "history_entity"));
    EXPECT_EQ(std::vector<i64>(history_entity.begin() + 10, history_entity.end()), (std::vector<i64>(4, 0)));
    EXPECT_EQ(std::vector<i64>(history_dt.begin() + 10, history_dt.end()), (std::vector<i64>(4, 0)));
    EXPECT_EQ(ints(*padded, "history_offsets"), ints(*exact, "history_offsets"));
    // a capacity below the history steps overflows, and the batch is all zero
    fits.history = entries - 1;
    const auto over = encoder->encode_capacity(in, fits, pool, scratch, sizes);
    EXPECT_TRUE(sizes.overflowed());
    EXPECT_EQ(sizes.history, entries);
    for (const i64 v : ints(*over, "history_entity"))
        EXPECT_EQ(v, 0);
}

// ------------------------------------------------------------------------------------------------ streams

TEST(EncoderLayers, StreamsEqualTheBatchWithSubgoalsAndHistory)
{
    const auto table = rl::TaskTable::single(test::pddl_task("wide_predicates", "p01"));
    const u64 n = 24;
    const test::Rows rows = test::random_walk(*table, 0, n, 9);
    const auto layers = random_layers(*table->task(0), n, 2, 11);
    const std::vector<u32> instances(n, 0);
    const Masks masks = make_masks(*table, instances, layers);
    const std::vector<Steps> per_row = random_steps(*table->task(0), n, 5, 1);
    const History history = make_history(*table, instances, per_row);
    const auto encoder = FlatEncoder::create(table, layer_options(2, true, 3));
    EncodeInput all = input_of(rows);
    all.subgoals = layer_input(masks);
    all.history = history.input();
    const auto reference = encoder->encode(all);
    ThreadPool pool(1);

    // the rows of `from` to `to` as an input
    struct Slice
    {
        test::Rows rows;
        Masks masks;
        History history;
        EncodeInput input;
    };
    auto slice = [&](u64 from, u64 to)
    {
        auto s = std::make_unique<Slice>();
        s->rows.words = rows.words;
        s->rows.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(from * rows.words),
                            rows.data.begin() + static_cast<std::ptrdiff_t>(to * rows.words));
        s->rows.task_ids.assign(to - from, 0);
        const auto first = static_cast<std::ptrdiff_t>(from), last = static_cast<std::ptrdiff_t>(to);
        s->masks = make_masks(*table, std::vector<u32>(to - from, 0), {layers.begin() + first, layers.begin() + last});
        s->history = make_history(*table, std::vector<u32>(to - from, 0),
                                  {per_row.begin() + first, per_row.begin() + last});
        s->input = input_of(s->rows);
        s->input.subgoals = layer_input(s->masks);
        s->input.history = s->history.input();
        return s;
    };
    {
        FlatStream stream(encoder);
        for (u64 from = 0; from < n; from += 7)
            stream.append(slice(from, std::min(n, from + 7))->input);
        EXPECT_EQ(stream.size(), n);
        const auto flushed = stream.flush(pool);
        ASSERT_EQ(flushed->bytes(), reference->bytes());
        EXPECT_EQ(std::memcmp(flushed->block(), reference->block(), reference->bytes()), 0);
    }
    {
        // a mutable stream: rows under indices out of order, replaced and removed
        FlatMutableStream stream(encoder);
        std::vector<i64> indices(n);
        for (u64 g = 0; g < n; ++g)
            indices[g] = static_cast<i64>(n - g) * 10;  // descending: the flush sorts ascending
        for (u64 g = 0; g < n; ++g)
        {
            const i64 index = indices[g];
            stream.append(std::span<const i64>(&index, 1), slice(g, g + 1)->input);
        }
        // update every third row to the next row's inputs, then back
        for (u64 g = 0; g < n; g += 3)
        {
            const i64 index = indices[g];
            const u64 other = (g + 1) % n;
            stream.update(std::span<const i64>(&index, 1), slice(other, other + 1)->input);
        }
        for (u64 g = 0; g < n; g += 3)
        {
            const i64 index = indices[g];
            stream.update(std::span<const i64>(&index, 1), slice(g, g + 1)->input);
        }
        const auto flushed = stream.flush(pool);
        // the graph j of the flush is the row of the j-th smallest index: the rows in reverse
        std::vector<u64> order(n);
        for (u64 j = 0; j < n; ++j)
            order[j] = n - 1 - j;
        test::Rows reversed;
        reversed.words = rows.words;
        std::vector<std::vector<std::vector<Literal>>> reversed_layers;
        std::vector<Steps> reversed_steps;
        for (const u64 g : order)
        {
            reversed.data.insert(reversed.data.end(), rows.data.begin() + static_cast<std::ptrdiff_t>(g * rows.words),
                                 rows.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * rows.words));
            reversed.task_ids.push_back(0);
            reversed_layers.push_back(layers[g]);
            reversed_steps.push_back(per_row[g]);
        }
        const Masks reversed_masks = make_masks(*table, instances, reversed_layers);
        const History reversed_history = make_history(*table, instances, reversed_steps);
        EncodeInput in = input_of(reversed);
        in.subgoals = layer_input(reversed_masks);
        in.history = reversed_history.input();
        const auto expected = encoder->encode(in);
        ASSERT_EQ(flushed->bytes(), expected->bytes());
        EXPECT_EQ(std::memcmp(flushed->block(), expected->block(), expected->bytes()), 0);
    }
    {
        // rows without history and layers join a stream of an encoder with both
        FlatStream stream(encoder);
        stream.append(input_of(rows));
        const auto flushed = stream.flush(pool);
        const auto expected = encoder->encode(input_of(rows));
        ASSERT_EQ(flushed->bytes(), expected->bytes());
        EXPECT_EQ(std::memcmp(flushed->block(), expected->block(), expected->bytes()), 0);
    }
}

// ------------------------------------------------------------------------------------------------ refusals

TEST(EncoderLayers, InputsTheSchemaOrTheInstanceCannotTakeAreRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 3, 1);
    const Steps steps = {{-1, {on("clear", true, {0})}}, {0, {on("clear", true, {1})}}};
    const History history = make_history(*table, {0, 0, 0}, {steps, steps, steps});
    const Masks masks = make_masks(*table, {0, 0, 0}, {{{on("clear", true, {0})}}, {}, {}});
    auto refuses = [&](const FlatEncoder& encoder, EncodeInput in, const char* needle)
    {
        try
        {
            (void)encoder.encode(in);
            ADD_FAILURE() << "accepted: " << needle;
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("mymyr: encoding: "), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << error.what();
        }
    };
    const auto with_both = FlatEncoder::create(table, layer_options(1, true));
    const auto plain = FlatEncoder::create(table, FlatOptions{.goal_derivations = {}});
    {
        EncodeInput in = input_of(rows);
        in.subgoals = layer_input(masks);
        refuses(*plain, in, "no subgoal relations");
        in.history = history.input();
        in.subgoals = {};
        refuses(*plain, in, "no history relations");
    }
    {
        EncodeInput in = input_of(rows);
        in.subgoals = layer_input(masks);
        in.subgoals.layers = 2;  // the schema has one level
        refuses(*with_both, in, "subgoal layers are given");
        in.subgoals = layer_input(masks);
        in.subgoals.neg = nullptr;
        refuses(*with_both, in, "positive and negative masks");
        in.subgoals = layer_input(masks);
        in.subgoals.words = 0;
        in.subgoals.stride = 0;
        refuses(*with_both, in, "narrower than its instance");
        in.subgoals = layer_input(masks);
        Masks bad = masks;
        bad.pos[0] |= u64{1} << 63;  // a slot past the instance's atoms
        in.subgoals = layer_input(bad);
        refuses(*with_both, in, "subgoal layers of row 0 set an atom bit");
        in.subgoals = layer_input(masks);
        in.subgoals.layers = 1;
        in.subgoals.words = 3;
        in.subgoals.stride = 2;
        refuses(*with_both, in, "row stride");
    }
    {
        EncodeInput in = input_of(rows);
        in.history = history.input();
        in.history.dt = nullptr;
        refuses(*with_both, in, "need their dt");
        in.history = history.input();
        std::vector<i32> counts = {1, 3, 0};
        in.history.count = counts.data();
        refuses(*with_both, in, "history count of row 1");
        counts = {-1, 0, 0};
        refuses(*with_both, in, "history count of row 0");
        in.history = history.input();
        in.history.masks.words = 0;
        in.history.masks.stride = 0;
        refuses(*with_both, in, "history steps of row 0 are narrower");
        in.history = history.input();
        History bad = history;
        bad.masks.neg[bad.masks.layers * bad.masks.words + 0] |= u64{1} << 63;  // row 1 step 0
        in.history = bad.input();
        refuses(*with_both, in, "history steps of row 1 set an atom bit");
        // bits of a step beyond the row's count are not read
        History beyond = history;
        std::vector<i32> short_counts = {1, 1, 1};
        beyond.masks.pos[(0 * beyond.masks.layers + 1) * beyond.masks.words] |= u64{1} << 63;
        in.history = beyond.input();
        in.history.count = short_counts.data();
        EXPECT_NO_THROW((void)with_both->encode(in));
    }
    // the options
    try
    {
        FlatOptions options = layer_options(0, false);
        options.history_max_steps = 2;
        (void)FlatEncoder::create(table, options);
        ADD_FAILURE() << "history_max_steps without history";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("history_max_steps needs history"), std::string::npos);
    }
    try
    {
        FlatOptions options = layer_options(0, true);
        options.history_max_steps = 0;
        (void)FlatEncoder::create(table, options);
        ADD_FAILURE() << "history_max_steps of 0";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("history_max_steps must be at least 1"), std::string::npos);
    }
    {
        // all steps and the newest k steps are different schemas, and so are different k
        FlatOptions all = layer_options(0, true), one = all, two = all;
        one.history_max_steps = 1;
        two.history_max_steps = 2;
        const auto fingerprint = [&](const FlatOptions& o)
        { return FlatEncoder::create(table, o)->schema().fingerprint(); };
        EXPECT_NE(fingerprint(all), fingerprint(one));
        EXPECT_NE(fingerprint(one), fingerprint(two));
    }
    try
    {
        FlatOptions options = layer_options(0, false);
        options.targets = {TargetSource::History};
        (void)FlatEncoder::create(table, options);
        ADD_FAILURE() << "history targets without history";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("history targets need history"), std::string::npos);
    }
}

TEST(EncoderLayers, AStreamRefusesTheHistoryCountsTheBatchRefuses)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 2, 1);
    const auto encoder = FlatEncoder::create(table, layer_options(0, true));
    ThreadPool pool(1);
    // no steps given, but a count of 2: both the batch and the stream refuse it
    const std::vector<i32> counts = {2, 0};
    EncodeInput in = input_of(rows);
    in.history.count = counts.data();
    EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    FlatStream stream(encoder);
    EXPECT_THROW(stream.append(in), std::invalid_argument);
    FlatMutableStream mutable_stream(encoder);
    const std::vector<i64> indices = {0, 1};
    EXPECT_THROW(mutable_stream.append(indices, in), std::invalid_argument);
    // counts of 0 are the same as no history
    const std::vector<i32> zeros = {0, 0};
    in.history.count = zeros.data();
    EXPECT_NO_THROW(stream.append(in));
}

TEST(EncoderLayers, CorruptedHistoryAndSubgoalArraysAreRejected)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);
    const Steps steps = {{-2, {on("on", true, {1, 0}), on("clear", true, {1})}}, {-1, {on("clear", false, {0})}}};
    const History history = make_history(*table, {0}, {steps});
    const Masks masks = make_masks(*table, {0}, {{{on("clear", true, {1}), on("ontable", true, {2})}}});
    const auto encoder = FlatEncoder::create(table, layer_options(1, true));
    EncodeInput in = input_of(rows);
    in.history = history.input();
    in.subgoals = layer_input(masks);
    const auto bundle = encoder->encode(in);
    const EncodingSchema& schema = encoder->schema();
    validate(*bundle, schema);
    const Graph g = decode(*bundle, schema)[0];
    ASSERT_EQ(g.history_entity.size(), 2u);
    const u64 step0 = static_cast<u64>(g.history_entity[0]);
    const u64 target0 = step0 + 2;
    const u64 subgoal0 = 3 + 1 + 2;  // objects, auxiliary, goal entities
    ASSERT_EQ(g.role[subgoal0], 3);
    ASSERT_EQ(g.role[target0], 6);
    auto rejected = [&](const rl::ArrayBundle& changed, const char* what)
    { EXPECT_THROW(validate(changed, schema), std::invalid_argument) << what; };
    rejected(test::with_change(*bundle, "history_entity", 0, 1), "a history step entity that is not the next");
    rejected(test::with_change(*bundle, "history_offsets", 1, -1), "history_offsets that miscount the steps");
    rejected(test::with_value(*bundle, "entity_ref", step0 + 1, 0), "history steps without ascending references");
    rejected(test::with_value(*bundle, "entity_role", step0, 4), "a history step with the role of an action");
    rejected(test::with_value(*bundle, "entity_role", target0, 5), "a history target with the role of a step");
    rejected(test::with_change(*bundle, "entity_ref", target0, 1), "a history target that refers to another tuple");
    rejected(test::with_change(*bundle, "entity_ref", subgoal0, 1), "a subgoal entity that refers to another tuple");
    rejected(test::with_value(*bundle, "entity_role", subgoal0, 2), "a subgoal entity with the role of a goal");
    const auto groups = ints(*bundle, "target_group");
    ASSERT_EQ(groups.size(), 2u + 2u + 3u);  // goal, subgoal, history targets
    rejected(test::with_value(*bundle, "target_group", 2, 0), "a subgoal target in the goal group");
    rejected(test::with_value(*bundle, "target_group", 4, 1), "a history target in the subgoal group");
    rejected(test::with_change(*bundle, "target_entity", 5, 1), "a history target that lists another entity");
}

TEST(EncoderLayers, TheFillRefusesLayersThatTheCountDidNotSee)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 2, 1);
    const Masks masks = make_masks(*table, {0, 0}, {{{on("clear", true, {0})}}, {{on("clear", true, {1})}}});
    const Masks other = masks;
    const auto encoder = FlatEncoder::create(table, layer_options(1, false));
    EncodeInput in = input_of(rows);
    in.subgoals = layer_input(masks);
    ThreadPool pool(1);
    EncodeScratch scratch;
    const FlatSizes sizes = encoder->count(in, pool, scratch);
    EncodeInput changed = in;
    changed.subgoals = layer_input(other);
    EXPECT_THROW(encoder->fill(changed, sizes, FlatBuffers{}, pool, scratch), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ collation

test::Rows rows_between(const test::Rows& rows, u64 start, u64 count)
{
    test::Rows out;
    out.words = rows.words;
    out.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(start * rows.words),
                    rows.data.begin() + static_cast<std::ptrdiff_t>((start + count) * rows.words));
    out.task_ids.assign(rows.task_ids.begin() + static_cast<std::ptrdiff_t>(start),
                        rows.task_ids.begin() + static_cast<std::ptrdiff_t>(start + count));
    return out;
}

void expect_same_bundles(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    EXPECT_EQ(a.scalars(), b.scalars());
    ASSERT_EQ(a.arrays().size(), b.arrays().size());
    for (const auto& info : a.arrays())
    {
        const auto* other = b.find(info.name.view());
        ASSERT_NE(other, nullptr) << info.name;
        ASSERT_EQ(info.shape, other->shape) << info.name;
        EXPECT_EQ(std::memcmp(a.data(info), b.data(*other), info.elements() * rl::dtype_bytes(info.dtype)), 0)
            << info.name;
    }
}

/// The batch with array `name` one element shorter.
std::shared_ptr<const rl::ArrayBundle> shortened(const rl::ArrayBundle& batch, std::string_view name)
{
    rl::ArrayBundle::Builder builder;
    for (const auto& [key, value] : batch.scalars())
        builder.scalar(key, value);
    for (const auto& a : batch.arrays())
    {
        const auto* p = static_cast<const std::byte*>(batch.data(a));
        u64 n = a.elements();
        rl::ArrayShape shape = a.shape;
        if (a.name.view() == name)
            shape = rl::ArrayShape{static_cast<i64>(--n)};
        builder.add_bytes(a.name, a.dtype, shape, std::vector<std::byte>(p, p + n * rl::dtype_bytes(a.dtype)), a.words);
    }
    return std::make_shared<const rl::ArrayBundle>(builder.build());
}

TEST(EncoderLayers, CollatedHistorySlicesEqualOneEncodeInBothLayoutsAndIndexTypes)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const u64 n = 60;
    const test::Rows rows = test::random_walk(*table, 0, n, 11);
    const std::vector<Steps> per_row = random_steps(*table->task(0), n, 5, 13);
    const History history = make_history(*table, std::vector<u32>(n, 0), per_row);
    for (const u32 k : {0u, 2u})
        for (const Layout layout : {Layout::Graph, Layout::Relation})
            for (const IndexType index : {IndexType::I32, IndexType::I64})
            {
                SCOPED_TRACE(testing::Message() << k << " " << static_cast<int>(layout) << " " << static_cast<int>(index));
                FlatOptions options = layer_options(0, true, k, layout);
                options.index = index;
                const auto encoder = FlatEncoder::create(table, options);
                EncodeInput in = input_of(rows);
                in.history = history.input();
                const auto whole = encoder->encode(in);
                ASSERT_GT(whole->find("history_entity")->elements(), 0u);
                std::vector<std::shared_ptr<const rl::ArrayBundle>> batches;
                u64 start = 0;
                for (const u64 count : {9u, 0u, 51u})
                {
                    const test::Rows part = rows_between(rows, start, count);
                    const std::vector<Steps> steps(per_row.begin() + static_cast<std::ptrdiff_t>(start),
                                                   per_row.begin() + static_cast<std::ptrdiff_t>(start + count));
                    const History h = make_history(*table, std::vector<u32>(count, 0), steps);
                    EncodeInput slice = input_of(part);
                    slice.history = h.input();
                    batches.push_back(encoder->encode(slice));
                    start += count;
                }
                const auto collated = collate(batches);
                validate(*collated, encoder->schema());
                expect_same_bundles(*whole, *collated);
            }
}

TEST(EncoderLayers, CollationRefusesEveryArrayShorterThanItsCountsBeforeReadingIt)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const u64 n = 5;
    const test::Rows rows = test::random_walk(*table, 0, n, 17);
    const History history = make_history(*table, std::vector<u32>(n, 0), random_steps(*table->task(0), n, 3, 19));
    const auto encoder = FlatEncoder::create(table, layer_options(0, true));
    EncodeInput in = input_of(rows);
    in.history = history.input();
    const auto batch = encoder->encode(in);
    std::set<std::string> checked;
    for (const auto& a : batch->arrays())
    {
        const std::string name = a.name.str();
        SCOPED_TRACE(name);
        ASSERT_GT(a.elements(), 0u);
        const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{batch, shortened(*batch, name)};
        try
        {
            (void)collate(batches);
            ADD_FAILURE() << "collated a batch with a short " << name;
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("too short: " + name), std::string::npos) << e.what();
        }
        checked.insert(name);
    }
    for (const char* name : {"task_ids", "num_objects", "goal_complete", "entity_offsets", "entity_role", "entity_ref",
                             "relation_offsets", "relation_arg_offsets", "relation_counts", "relation_args",
                             "target_offsets", "target_entity", "target_group", "history_offsets", "history_entity",
                             "history_dt"})
        EXPECT_TRUE(checked.count(name)) << name;
}

}  // namespace
