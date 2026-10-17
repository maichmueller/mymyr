// The goal relations of the relational encoder: the plain, satisfied and unsatisfied variants for fluent, derived and
// static predicates, the constant part, goal entities and targets, the goals of rows, and the layouts and thread counts.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
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
    std::vector<std::vector<i64>> tuple_ids;  // [relation][tuple]: the index of the tuple among the graph's tuples
                                              // in graph-major order (the cells in relation order)
    std::vector<i64> role, ref;               // per entity
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
    std::vector<Graph> out(G);
    for (u64 g = 0; g < G; ++g)
    {
        Graph& graph = out[g];
        graph.num_objects = num_objects[g];
        graph.tuples.assign(R, {});
        graph.tuple_ids.assign(R, {});
        for (i64 e = entity_offsets[g]; e < entity_offsets[g + 1]; ++e)
        {
            graph.role.push_back(role[e]);
            graph.ref.push_back(ref[e]);
        }
        i64 index = 0;
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
                graph.tuple_ids[r].push_back(index++);
            }
        }
    }
    return out;
}

using AtomKey = std::pair<std::string, Tuple>;

std::string predicate_name(const Task& task, u32 predicate)
{
    return std::string(task.data().str(task.data().predicates[predicate].name));
}

/// The true atoms of a row, by predicate name: fluent, derived and static ones, from the task's own atom records.
std::set<AtomKey> true_atoms(const rl::TaskTable& table, u32 instance, const u64* row)
{
    const Task& task = *table.task(instance);
    std::set<AtomKey> out;
    auto add = [&](u32 predicate, Tuple objects) { out.insert({predicate_name(task, predicate), std::move(objects)}); };
    auto objects_of = [&](const u32* record)
    {
        Tuple objects;
        for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
            objects.push_back(record[1 + k]);
        return objects;
    };
    for (u32 slot = 0; slot < task.atoms().fluent_slots(); ++slot)
        if ((row[slot >> 6] >> (slot & 63)) & 1)
        {
            const u32* record = task.atoms().record(AtomKind::Fluent, slot);
            add(record[0], objects_of(record));
        }
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
            {
                const u32* record = task.atoms().record(AtomKind::Derived, slot);
                add(record[0], objects_of(record));
            }
    }
    for (const formalism::GroundAtom& atom : task.data().static_init)
    {
        Tuple objects;
        for (const auto object : task.data().objects_of(atom))
            objects.push_back(object.v);
        add(atom.pred.v, objects);
    }
    return out;
}

/// A goal literal: predicate name, sign and objects.
struct Literal
{
    std::string predicate;
    bool positive;
    Tuple objects;
};

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

using Named = std::map<std::string, std::vector<Tuple>>;

/// The tuples of the goal relations (without a goal target) of a goal that holds `literals`, in the row of `atoms`.
Named goal_reference(const EncodingSchema& schema, i64 num_objects, const std::vector<Literal>& literals,
                     const std::set<AtomKey>& atoms)
{
    Named out;
    for (const Literal& literal : literals)
    {
        const bool holds = (atoms.count({literal.predicate, literal.objects}) != 0) == literal.positive;
        const std::string base = std::string(literal.positive ? "[+]" : "[-]") + literal.predicate + "[g]";
        const Tuple tuple = literal.objects.empty() ? Tuple{num_objects} : literal.objects;
        for (const std::string& name : {base, base + (holds ? "[sat]" : "[unsat]")})
            if (schema.find(name))
                out[name].push_back(tuple);
    }
    for (auto& entry : out)
    {
        std::sort(entry.second.begin(), entry.second.end());
        entry.second.erase(std::unique(entry.second.begin(), entry.second.end()), entry.second.end());
    }
    return out;
}

bool is_goal_relation(const Relation& relation) { return relation.lane == Lane::Goal || relation.lane == Lane::GoalDerivation; }

/// Checks the goal relations, the goal entities and the targets of a batch of rows against a reference goal per row.
void expect_goals(const rl::ArrayBundle& bundle, const EncodingSchema& schema, const std::vector<Graph>& graphs,
                  const std::vector<Named>& expected)
{
    const bool targets = schema.has_targets();
    const auto target_offsets = targets ? ints(bundle, "target_offsets") : std::vector<i64>{};
    const auto target_entity = targets ? ints(bundle, "target_entity") : std::vector<i64>{};
    const auto target_group = targets ? ints(bundle, "target_group") : std::vector<i64>{};
    const auto entity_offsets = ints(bundle, "entity_offsets");
    ASSERT_EQ(graphs.size(), expected.size());
    for (u64 g = 0; g < graphs.size(); ++g)
    {
        const Graph& graph = graphs[g];
        bool nullary = false;
        for (u32 r = 0; r < schema.num_relations(); ++r)
            if (schema.relation(r).logical_arity == 0 && !graph.tuples[r].empty())
                nullary = true;
        i64 running = 0;
        std::vector<i64> goal_entities;
        const i64 goal_start = graph.num_objects + (nullary ? 1 : 0);
        for (u32 r = 0; r < schema.num_relations(); ++r)
        {
            const Relation& relation = schema.relation(r);
            if (!is_goal_relation(relation))
                continue;
            const bool target = schema.slot_roles(r).front() == SlotRole::GoalTarget;
            std::vector<Tuple> got;
            for (usize t = 0; t < graph.tuples[r].size(); ++t)
            {
                Tuple tuple = graph.tuples[r][t];
                if (target)
                {
                    EXPECT_EQ(tuple.front(), goal_start + running) << "graph " << g << " " << relation.name;
                    EXPECT_EQ(graph.role[static_cast<usize>(tuple.front())], 2) << relation.name;
                    EXPECT_EQ(graph.ref[static_cast<usize>(tuple.front())], graph.tuple_ids[r][t]) << relation.name;
                    goal_entities.push_back(tuple.front() + entity_offsets[g]);
                    ++running;
                    tuple.erase(tuple.begin());
                }
                got.push_back(tuple);
            }
            const auto found = expected[g].find(relation.name);
            const std::vector<Tuple> want = found == expected[g].end() ? std::vector<Tuple>{} : found->second;
            EXPECT_EQ(got, want) << "graph " << g << " relation " << relation.name;
        }
        EXPECT_EQ(static_cast<i64>(graph.role.size()), goal_start + running) << "graph " << g;
        if (targets)
        {
            const std::vector<i64> mine(target_entity.begin() + target_offsets[g], target_entity.begin() + target_offsets[g + 1]);
            EXPECT_EQ(mine, goal_entities);
            for (i64 t = target_offsets[g]; t < target_offsets[g + 1]; ++t)
                EXPECT_EQ(target_group[static_cast<usize>(t)], 0);
        }
    }
}

FlatOptions goal_options(Layout layout = Layout::Graph, bool targets = true)
{
    FlatOptions options;
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    if (targets)
        options.targets = {TargetSource::Goal};
    options.layout = layout;
    options.equality = true;
    return options;
}

EncodeInput input_of(const test::Rows& rows)
{
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    return in;
}

struct Problem
{
    const char* domain;
    const char* problem;
};
const std::vector<Problem> k_problems = {
    {"blocks", "smedium"},           {"blocks", "probBLOCKS-8-0"},   {"gripper", "p01"},
    {"logistics00", "probLOGISTICS-6-1"}, {"philosophers", "p03-phil4"}, {"features", "p01"},
    {"goals", "conjunctive"},        {"goals", "forall"},            {"goals", "static_false"},
    {"goals", "forall_negative"},    {"goals", "or_static_true"},    {"goals", "forall_partly_expandable"},
    {"goals", "forall_equality"},    {"goals", "imply_true"},        {"goals", "static_true"},
    {"goals", "contradiction"},      {"goals", "forall_nested"}};

rl::TaskTablePtr goals_table()
{
    std::vector<TaskPtr> tasks;
    for (const Problem& p : k_problems)
        if (std::string(p.domain) == "goals")
            tasks.push_back(test::pddl_task(p.domain, p.problem));
    return rl::TaskTable::create(tasks);
}

test::Rows mixed_rows(const rl::TaskTable& table, u64 per_instance, u64 seed)
{
    std::vector<test::Rows> sources;
    for (u32 i = 0; i < table.size(); ++i)
        sources.push_back(test::random_walk(table, i, per_instance, seed + i));
    test::Rows mixed;
    mixed.words = table.words();
    const u32 row_words = table.row_words();
    for (u64 k = 0; k < per_instance; ++k)
        for (const test::Rows& source : sources)
        {
            mixed.data.insert(mixed.data.end(), source.data.begin() + static_cast<std::ptrdiff_t>(k * row_words),
                              source.data.begin() + static_cast<std::ptrdiff_t>((k + 1) * row_words));
            mixed.task_ids.push_back(source.task_ids[k]);
        }
    return mixed;
}

std::vector<Named> instance_goal_reference(const EncodingSchema& schema, const rl::TaskTable& table, const test::Rows& rows,
                                           const std::vector<Graph>& graphs)
{
    std::vector<Named> expected;
    for (u64 g = 0; g < rows.size(); ++g)
    {
        const u32 instance = static_cast<u32>(rows.task_ids[g]);
        expected.push_back(goal_reference(schema, graphs[g].num_objects, encoding_goal(*table.task(instance)),
                                          true_atoms(table, instance, rows.data.data() + g * rows.words)));
    }
    return expected;
}

/// How many goal tuples of each kind the checked batches held, so that a test can tell it compared something.
struct Tally
{
    u64 plain = 0, satisfied = 0, unsatisfied = 0, negative = 0, derived = 0;
};

/// Whether the goal of `task` stays complete under `options`: it was complete, and no literal was dropped whose truth a
/// state can change (a nullary literal over a fluent or derived predicate under nullary = false; a dropped static or
/// equality literal has a fixed truth).
bool stays_complete(const Task& task, const FlatOptions& options)
{
    if (!task.encoding_goal_complete())
        return false;
    for (const formalism::Literal& literal : task.encoding_goal())
        if (!options.nullary && task.compiled().arity[literal.pred.v] == 0 &&
            task.data().predicates[literal.pred.v].kind != formalism::PredKind::Static)
            return false;
    return true;
}

Tally expect_instance_goals(const FlatOptions& options)
{
    Tally tally;
    for (const Problem& p : k_problems)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const auto encoder = FlatEncoder::create(table, options);
        const test::Rows rows = test::random_walk(*table, 0, 96, 31);
        const auto bundle = encoder->encode(input_of(rows));
        validate(*bundle, encoder->schema());
        const auto graphs = decode(*bundle, encoder->schema());
        expect_goals(*bundle, encoder->schema(), graphs, instance_goal_reference(encoder->schema(), *table, rows, graphs));
        for (const Graph& graph : graphs)
            for (u32 r = 0; r < encoder->schema().num_relations(); ++r)
            {
                const Relation& relation = encoder->schema().relation(r);
                if (!is_goal_relation(relation))
                    continue;
                const u64 n = graph.tuples[r].size();
                (relation.derivation == GoalDerivation::Satisfied     ? tally.satisfied
                 : relation.derivation == GoalDerivation::Unsatisfied ? tally.unsatisfied
                                                                      : tally.plain) += n;
                tally.negative += relation.sign < 0 ? n : 0;
                tally.derived += table->predicate_kinds()[static_cast<u32>(relation.predicate)] == rl::k_pred_derived ? n : 0;
            }
        const auto complete = ints(*bundle, "goal_complete");
        EXPECT_EQ(complete.size(), rows.size());
        for (const i64 c : complete)
            EXPECT_EQ(c, stays_complete(*table->task(0), options) ? 1 : 0);
    }
    return tally;
}

TEST(EncoderGoals, TuplesMatchTheGoalAndTheStateGraphMajor)
{
    const Tally tally = expect_instance_goals(goal_options());
    EXPECT_GT(tally.plain, 0u);
    EXPECT_GT(tally.satisfied, 0u);
    EXPECT_GT(tally.unsatisfied, 0u);
    EXPECT_GT(tally.negative, 0u);
    EXPECT_GT(tally.derived, 0u);
}

TEST(EncoderGoals, TuplesMatchTheGoalAndTheStateRelationMajor)
{
    expect_instance_goals(goal_options(Layout::Relation));
}

TEST(EncoderGoals, WithoutTargetsAndWithInt64Indices)
{
    FlatOptions options = goal_options(Layout::Graph, false);
    expect_instance_goals(options);
    options.index = IndexType::I64;
    options.targets = {TargetSource::Goal};
    expect_instance_goals(options);
}

TEST(EncoderGoals, PlainOnlyAndDerivationsOnly)
{
    FlatOptions options = goal_options();
    options.goal_derivations = {GoalDerivation::Plain};
    expect_instance_goals(options);
    options.goal_derivations = {GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    options.targets = {};
    expect_instance_goals(options);
}

TEST(EncoderGoals, WithoutNullaryEqualityStaticOrDerivedRelations)
{
    FlatOptions options = goal_options();
    options.nullary = false;
    expect_instance_goals(options);
    options.equality = false;
    expect_instance_goals(options);
    options.include_static = false;
    options.include_derived = false;
    expect_instance_goals(options);
}

TEST(EncoderGoals, ForallGoalsAreExpandedAndTheFlagIsPerInstance)
{
    const auto table = goals_table();
    const auto encoder = FlatEncoder::create(table, goal_options());
    const test::Rows rows = mixed_rows(*table, 4, 3);
    const auto bundle = encoder->encode(input_of(rows));
    const auto complete = ints(*bundle, "goal_complete");
    for (u64 g = 0; g < rows.size(); ++g)
        EXPECT_EQ(complete[g], table->task(static_cast<u32>(rows.task_ids[g]))->encoding_goal_complete() ? 1 : 0);
    EXPECT_NE(std::count(complete.begin(), complete.end(), 0), 0);
    EXPECT_NE(std::count(complete.begin(), complete.end(), 1), 0);
    // forall (?p - place) (lit ?p): one tuple for each of the places a, b and the constant home
    const auto forall = FlatEncoder::create(rl::TaskTable::single(test::pddl_task("goals", "forall")), goal_options());
    const rl::TaskTablePtr one = rl::TaskTable::single(test::pddl_task("goals", "forall"));
    const test::Rows initial = test::random_walk(*one, 0, 1, 1);
    const auto single = forall->encode(input_of(initial));
    const auto graphs = decode(*single, forall->schema());
    EXPECT_EQ(graphs[0].tuples[*forall->schema().find("[+]lit[g]")].size(), 3u);
}

TEST(EncoderGoals, TheConstantPartIsEmittedEvenWhenItIsFalse)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "static_false"));
    const auto encoder = FlatEncoder::create(table, goal_options());
    EXPECT_TRUE(table->task(0)->encoding_goal_unsatisfiable());
    const test::Rows rows = test::random_walk(*table, 0, 8, 2);
    const auto bundle = encoder->encode(input_of(rows));
    const auto graphs = decode(*bundle, encoder->schema());
    const u32 plain = *encoder->schema().find("[+]heavy[g]");
    const u32 unsat = *encoder->schema().find("[+]heavy[g][unsat]");
    const u32 sat = *encoder->schema().find("[+]heavy[g][sat]");
    for (const Graph& graph : graphs)
    {
        EXPECT_EQ(graph.tuples[plain].size(), 1u);  // (heavy t2) is false in the instance and listed anyway
        EXPECT_EQ(graph.tuples[unsat].size(), 1u);
        EXPECT_EQ(graph.tuples[sat].size(), 0u);
    }
}

TEST(EncoderGoals, TheConstantPartIsSortedAndMergedWithTheMaskedLiterals)
{
    // (lit a) is a fluent literal; the statics of the goal are sorted by their objects
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "forall_static_guard"));
    const auto encoder = FlatEncoder::create(table, goal_options());
    const test::Rows rows = test::random_walk(*table, 0, 32, 4);
    const auto bundle = encoder->encode(input_of(rows));
    validate(*bundle, encoder->schema());  // every cell is strictly ascending
    const auto graphs = decode(*bundle, encoder->schema());
    expect_goals(*bundle, encoder->schema(), graphs, instance_goal_reference(encoder->schema(), *table, rows, graphs));
}

/// The fluent goal masks of the instance, shared by every row.
struct GoalRows
{
    std::vector<u64> pos, neg;
};

TEST(EncoderGoals, RowGoalsEqualToTheTaskGoalGiveTheDefaultOutput)
{
    for (const Problem& p : k_problems)
    {
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const auto& instance = table->instance(0);
        const Task& task = *table->task(0);
        // goals whose literals are all over fluent predicates: the masks of the table are the whole encoding goal
        bool only_fluent = !instance.goal_derived && task.encoding_goal_complete();
        for (const auto& literal : task.encoding_goal())
            if (task.data().predicates[literal.pred.v].kind != formalism::PredKind::Fluent)
                only_fluent = false;
        if (!only_fluent)
            continue;
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        for (const Layout layout : {Layout::Graph, Layout::Relation})
        {
            const auto encoder = FlatEncoder::create(table, goal_options(layout));
            const test::Rows rows = test::random_walk(*table, 0, 40, 9);
            EncodeInput given = input_of(rows);
            given.goal_pos = instance.goal_pos.data();
            given.goal_neg = instance.goal_neg.data();
            given.goal_words = static_cast<u32>(instance.goal_pos.size());
            const auto a = encoder->encode(input_of(rows));
            const auto b = encoder->encode(given);
            ASSERT_EQ(a->arrays().size(), b->arrays().size());
            for (usize i = 0; i < a->arrays().size(); ++i)
                EXPECT_EQ(std::memcmp(a->data(a->arrays()[i]), b->data(b->arrays()[i]),
                                      a->arrays()[i].elements() * rl::dtype_bytes(a->arrays()[i].dtype)),
                          0)
                    << a->arrays()[i].name;
        }
    }
}

TEST(EncoderGoals, RowGoalsReplaceTheInstanceGoalWithoutAConstantPart)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "static_false"));
    const Task& task = *table->task(0);
    const u32 words = table->words();
    const test::Rows rows = test::random_walk(*table, 0, 64, 6);
    // a seeded goal for every row: random bits of the fluent slots, positive and negative
    std::mt19937_64 rng(13);
    std::vector<u64> pos(rows.size() * words, 0), neg(rows.size() * words, 0);
    const u32 slots = task.atoms().fluent_slots();
    for (u64 g = 0; g < rows.size(); ++g)
        for (u32 s = 0; s < slots; ++s)
        {
            const u64 draw = rng() % 4;
            if (draw == 0)
                pos[g * words + (s >> 6)] |= u64{1} << (s & 63);
            else if (draw == 1)
                neg[g * words + (s >> 6)] |= u64{1} << (s & 63);
        }
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        const auto encoder = FlatEncoder::create(table, goal_options(layout));
        EncodeInput in = input_of(rows);
        in.goal_pos = pos.data();
        in.goal_neg = neg.data();
        in.goal_stride = words;
        in.goal_words = words;
        const auto bundle = encoder->encode(in);
        validate(*bundle, encoder->schema());
        const auto graphs = decode(*bundle, encoder->schema());
        std::vector<Named> expected;
        for (u64 g = 0; g < rows.size(); ++g)
        {
            std::vector<Literal> literals;
            for (u32 s = 0; s < slots; ++s)
                for (const bool positive : {true, false})
                {
                    const u64* mask = (positive ? pos : neg).data() + g * words;
                    if (!((mask[s >> 6] >> (s & 63)) & 1))
                        continue;
                    const u32* record = task.atoms().record(AtomKind::Fluent, s);
                    Literal literal{predicate_name(task, record[0]), positive, {}};
                    for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
                        literal.objects.push_back(record[1 + k]);
                    literals.push_back(std::move(literal));
                }
            expected.push_back(goal_reference(encoder->schema(), graphs[g].num_objects, literals,
                                              true_atoms(*table, 0, rows.data.data() + g * words)));
        }
        expect_goals(*bundle, encoder->schema(), graphs, expected);
        for (const i64 c : ints(*bundle, "goal_complete"))
            EXPECT_EQ(c, 1);
    }
}

TEST(EncoderGoals, RowGoalsOverDerivedSlots)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    const Task& task = *table->task(0);
    ASSERT_GT(task.atoms().derived_slots(), 0u);
    const u32 words = table->words(), derived_words = std::max(1u, table->derived_words());
    const test::Rows rows = test::random_walk(*table, 0, 32, 8);
    std::vector<u64> pos(words, 0), neg(words, 0), dpos(derived_words, 0), dneg(derived_words, 0);
    dpos[0] = 1;  // the first derived atom, positive
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        const auto encoder = FlatEncoder::create(table, goal_options(layout));
        EncodeInput in = input_of(rows);
        in.goal_pos = pos.data();
        in.goal_neg = neg.data();
        in.goal_words = words;
        in.goal_derived_pos = dpos.data();
        in.goal_derived_neg = dneg.data();
        in.goal_derived_words = derived_words;
        const auto bundle = encoder->encode(in);
        validate(*bundle, encoder->schema());
        const auto graphs = decode(*bundle, encoder->schema());
        const u32* record = task.atoms().record(AtomKind::Derived, 0);
        const std::string name = predicate_name(task, record[0]);
        std::vector<Named> expected;
        for (u64 g = 0; g < rows.size(); ++g)
            expected.push_back(goal_reference(encoder->schema(), graphs[g].num_objects, {Literal{name, true, {}}},
                                              true_atoms(*table, 0, rows.data.data() + g * words)));
        // the atom's arguments (none for a nullary derived atom) are the tuple
        for (u64 g = 0; g < rows.size(); ++g)
            ASSERT_EQ(task.compiled().arity[record[0]], 0u);
        expect_goals(*bundle, encoder->schema(), graphs, expected);
    }
}

TEST(EncoderGoals, GoalsWithBitsOfAnotherInstanceAndNarrowMasksAreRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    const auto encoder = FlatEncoder::create(table, goal_options());
    const test::Rows rows = test::random_walk(*table, 0, 4, 1);
    const u32 words = table->words();
    std::vector<u64> pos(words, 0), neg(words, 0);
    auto refused = [&](EncodeInput in, const char* what)
    {
        try
        {
            (void)encoder->encode(in);
            ADD_FAILURE() << "accepted: " << what;
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("mymyr: encoding: "), std::string::npos) << error.what();
        }
    };
    EncodeInput in = input_of(rows);
    in.goal_pos = pos.data();
    in.goal_words = words;
    refused(in, "goal_pos without goal_neg");
    in.goal_neg = neg.data();
    in.goal_words = 0;
    refused(in, "narrow masks");
    in.goal_words = words;
    in.goal_derived_pos = pos.data();
    refused(in, "a derived mask without its pair");
    in.goal_derived_pos = nullptr;
    pos.back() = u64{1} << 63;  // a slot past the instance's atoms
    refused(in, "a bit the instance does not have");
}

TEST(EncoderGoals, SingleEqualsBatchForMixedInstances)
{
    const auto table = goals_table();
    const auto encoder = FlatEncoder::create(table, goal_options());
    const test::Rows rows = mixed_rows(*table, 6, 11);
    const auto bundle = encoder->encode(input_of(rows));
    const auto graphs = decode(*bundle, encoder->schema());
    for (u64 g = 0; g < rows.size(); ++g)
    {
        test::Rows one;
        one.words = rows.words;
        one.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(g * rows.words),
                        rows.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * rows.words));
        one.task_ids = {rows.task_ids[g]};
        const auto alone = decode(*encoder->encode(input_of(one)), encoder->schema());
        ASSERT_EQ(alone.size(), 1u);
        Graph a = alone[0], b = graphs[g];
        EXPECT_EQ(a.tuples, b.tuples) << g;
        EXPECT_EQ(a.role, b.role) << g;
        EXPECT_EQ(a.ref, b.ref) << g;
    }
}

TEST(EncoderGoals, LayoutsHoldTheSameGraphs)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 6, 21);
    const auto graph_encoder = FlatEncoder::create(table, goal_options(Layout::Graph));
    const auto relation_encoder = FlatEncoder::create(table, goal_options(Layout::Relation));
    const auto a = graph_encoder->encode(input_of(rows));
    const auto b = relation_encoder->encode(input_of(rows));
    validate(*b, relation_encoder->schema());
    for (const char* name : {"task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref",
                             "relation_counts", "goal_complete", "target_offsets", "target_entity", "target_group"})
        EXPECT_EQ(ints(*a, name), ints(*b, name)) << name;
    const auto ga = decode(*a, graph_encoder->schema());
    const auto gb = decode(*b, relation_encoder->schema());
    ASSERT_EQ(ga.size(), gb.size());
    for (u64 g = 0; g < ga.size(); ++g)
    {
        EXPECT_EQ(ga[g].tuples, gb[g].tuples);
        EXPECT_EQ(ga[g].role, gb[g].role);
        EXPECT_EQ(ga[g].ref, gb[g].ref);
    }
}

TEST(EncoderGoals, EveryThreadCountGivesTheSameBytes)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 400, 5);  // more graphs than one member's share
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        const auto encoder = FlatEncoder::create(table, goal_options(layout));
        const auto reference = encoder->encode(input_of(rows));
        for (const u32 threads : {2u, 3u, 5u})
        {
            ThreadPool pool(threads);
            EncodeScratch scratch;
            const auto bundle = encoder->encode(input_of(rows), pool, scratch);
            ASSERT_EQ(bundle->bytes(), reference->bytes());
            EXPECT_EQ(std::memcmp(bundle->block(), reference->block(), bundle->bytes()), 0) << threads;
        }
    }
}

TEST(EncoderGoals, TheCapacityFormKeepsTheGoalArrays)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 3, 17);
    const auto encoder = FlatEncoder::create(table, goal_options());
    const auto exact = encoder->encode(input_of(rows));
    ThreadPool pool(1);
    EncodeScratch scratch;
    FlatSizes sizes;
    const FlatCapacities fits{ints(*exact, "entity_ref").size() + 5, ints(*exact, "relation_args").size() + 7,
                              ints(*exact, "relation_args").size() + 7, ints(*exact, "target_entity").size() + 3};
    const auto padded = encoder->encode_capacity(input_of(rows), fits, pool, scratch, sizes);
    EXPECT_FALSE(sizes.overflowed());
    EXPECT_EQ(sizes.targets, ints(*exact, "target_entity").size());
    validate(*padded, encoder->schema());
    EXPECT_EQ(ints(*padded, "target_entity").size(), sizes.targets + 3);
    EXPECT_EQ(ints(*padded, "goal_complete"), ints(*exact, "goal_complete"));
    FlatCapacities small = fits;
    small.targets = sizes.targets - 1;
    const auto over = encoder->encode_capacity(input_of(rows), small, pool, scratch, sizes);
    EXPECT_TRUE(sizes.overflow & k_overflow_capacity);
    for (const rl::ArrayInfo& info : over->arrays())
        for (const i64 value : ints(*over, std::string(info.name.str()).c_str()))
            EXPECT_EQ(value, 0) << info.name.str();
}

TEST(EncoderGoals, TheAuxiliaryEntityCountsNullaryGoalTuples)
{
    // the goal's nullary literal (done) is derived and (flag) is fluent; the graph has the auxiliary entity because of it
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    FlatOptions options = goal_options();
    const auto encoder = FlatEncoder::create(table, options);
    const test::Rows rows = test::random_walk(*table, 0, 8, 3);
    const auto bundle = encoder->encode(input_of(rows));
    const auto graphs = decode(*bundle, encoder->schema());
    const u32 done = *encoder->schema().find("[+]done[g]");
    for (const Graph& graph : graphs)
    {
        ASSERT_EQ(graph.tuples[done].size(), 1u);
        EXPECT_EQ(graph.tuples[done][0][1], graph.num_objects);  // [goal entity, auxiliary]
        EXPECT_EQ(graph.role[static_cast<usize>(graph.num_objects)], 8);
    }
    options.nullary = false;
    const auto without = FlatEncoder::create(table, options);
    EXPECT_FALSE(without->schema().find("[+]done[g]"));
    const auto plain = decode(*without->encode(input_of(rows)), without->schema());
    for (const Graph& graph : plain)
        for (const i64 role : graph.role)
            EXPECT_NE(role, 8);
}

TEST(EncoderGoals, TheFingerprintSeparatesTargets)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    const auto with = EncodingSchema::create(*table, goal_options());
    const auto without = EncodingSchema::create(*table, goal_options(Layout::Graph, false));
    EXPECT_NE(with->fingerprint(), without->fingerprint());
    EXPECT_TRUE(with->has_goal_complete());
    EXPECT_TRUE(with->has_targets());
    EXPECT_FALSE(without->has_targets());
}

TEST(EncoderGoals, CorruptedGoalEntitiesAndTargetsAreRejected)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 2, 5);
    const auto encoder = FlatEncoder::create(table, goal_options());
    const auto batch = encoder->encode(input_of(rows));
    EXPECT_NO_THROW(validate(*batch, encoder->schema()));
    const std::vector<i64> roles = ints(*batch, "entity_role");
    const auto goal = static_cast<u64>(std::find(roles.begin(), roles.end(), 2) - roles.begin());
    ASSERT_LT(goal, roles.size());
    const auto rejected = [&](const char* array, u64 index, i64 delta)
    {
        EXPECT_THROW(validate(test::with_change(*batch, array, index, delta), encoder->schema()), std::invalid_argument)
            << array;
    };
    rejected("goal_complete", 0, 2);
    rejected("entity_ref", goal, 1);
    rejected("entity_role", goal, -2);
    rejected("target_entity", 0, 1);
    rejected("target_group", 0, 1);
    rejected("target_offsets", 1, 1);
}

TEST(EncoderGoals, AGoalTupleRepeatingTheObjectsOfItsPredecessorIsRejected)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 4, 3);
    const auto encoder = FlatEncoder::create(table, goal_options());
    const EncodingSchema& schema = encoder->schema();
    const auto batch = encoder->encode(input_of(rows));
    ASSERT_NO_THROW(validate(*batch, schema));
    const u64 R = schema.num_relations();
    const auto counts = ints(*batch, "relation_counts");
    const auto arg_offsets = ints(*batch, "relation_arg_offsets");
    const auto args = ints(*batch, "relation_args");
    bool found = false;
    for (u64 g = 0; g < rows.size() && !found; ++g)
        for (u32 r = 0; r < R && !found; ++r)
        {
            const u32 arity = schema.relation(r).arity;
            if (schema.slot_roles(r).front() != SlotRole::GoalTarget || arity < 2 || counts[g * R + r] < 2)
                continue;
            found = true;
            // the second tuple keeps its goal entity and takes the objects of the first
            const u64 first = static_cast<u64>(arg_offsets[g * R + r]), second = first + arity;
            rl::ArrayBundle corrupted = test::with_value(*batch, "relation_args", second + 1, args[first + 1]);
            for (u32 k = 2; k < arity; ++k)
                corrupted = test::with_value(corrupted, "relation_args", second + k, args[first + k]);
            EXPECT_THROW(validate(corrupted, schema), std::invalid_argument) << schema.relation(r).name;
        }
    ASSERT_TRUE(found);
}

TEST(EncoderGoals, MoreGoalTuplesThanGoalEntitiesAreRejectedBeforeAReferenceIsRead)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 2, 5);
    const auto encoder = FlatEncoder::create(table, goal_options());
    const auto batch = encoder->encode(input_of(rows));
    ASSERT_NO_THROW(validate(*batch, encoder->schema()));
    // drop the batch's last entity, a goal entity, from its graph and from the targets: its tuple has no entity left
    const auto entity_offsets = ints(*batch, "entity_offsets");
    const auto target_offsets = ints(*batch, "target_offsets");
    const u64 G = rows.size(), E = static_cast<u64>(entity_offsets[G]), T = static_cast<u64>(target_offsets[G]);
    ASSERT_GT(T, 0u);
    ASSERT_EQ(ints(*batch, "entity_role")[E - 1], 2);
    rl::ArrayBundle corrupted = test::with_value(*batch, "entity_offsets", G, static_cast<i64>(E) - 1);
    corrupted = test::with_value(corrupted, "target_offsets", G, static_cast<i64>(T) - 1);
    corrupted = test::with_value(corrupted, "target_entity", T - 1, 0);
    corrupted = test::with_value(corrupted, "entity_role", E - 1, 0);
    corrupted = test::with_value(corrupted, "entity_ref", E - 1, 0);
    try
    {
        validate(corrupted, encoder->schema());
        ADD_FAILURE() << "the batch was accepted";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("has an entity outside its graph"), std::string::npos)
            << error.what();
    }
}

TEST(EncoderGoals, TheFillRefusesAnyInputThatTheCountDidNotSee)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    const auto encoder = FlatEncoder::create(table, goal_options(Layout::Graph, false));
    const test::Rows rows = test::random_walk(*table, 0, 5, 3);
    const u32 words = table->words(), derived_words = std::max(1u, table->derived_words());
    std::vector<u64> pos(2 * words, 0), neg(2 * words, 0), dpos(2 * derived_words, 0), dneg(2 * derived_words, 0);
    std::vector<u64> other_neg = neg, other_dpos = dpos, other_dneg = dneg;
    EncodeInput in = input_of(rows);
    in.goal_pos = pos.data();
    in.goal_neg = neg.data();
    in.goal_stride = 0;
    in.goal_words = words;
    in.goal_derived_pos = dpos.data();
    in.goal_derived_neg = dneg.data();
    in.goal_derived_stride = 0;
    in.goal_derived_words = derived_words;
    ThreadPool pool(1);
    EncodeScratch scratch;
    const FlatSizes sizes = encoder->count(in, pool, scratch);
    const u64 G = rows.size(), R = encoder->schema().num_relations();
    std::vector<i32> task_ids(G), num_objects(G), entity_offsets(G + 1), counts(G * R), offsets(G * R + 1),
        arg_offsets(G * R + 1), args(sizes.args), refs(sizes.entities);
    std::vector<u8> roles(sizes.entities), complete(G);
    const FlatBuffers out{task_ids.data(), num_objects.data(), entity_offsets.data(), roles.data(), refs.data(),
                          counts.data(),   offsets.data(),     arg_offsets.data(),    args.data(),  complete.data()};
    auto refused = [&](const EncodeInput& other)
    { EXPECT_THROW(encoder->fill(other, sizes, out, pool, scratch), std::invalid_argument); };
    EncodeInput changed = in;
    changed.goal_neg = other_neg.data();
    refused(changed);
    changed = in;
    changed.goal_derived_pos = other_dpos.data();
    refused(changed);
    changed = in;
    changed.goal_derived_neg = other_dneg.data();
    refused(changed);
    changed = in;
    changed.goal_stride = words;
    refused(changed);
    changed = in;
    changed.goal_derived_stride = derived_words;
    refused(changed);
    changed = in;
    changed.goal_words = words + 1;
    refused(changed);
    changed = in;
    changed.goal_derived_words = derived_words + 1;
    refused(changed);
    changed = in;
    changed.states.stride = words + 1;
    refused(changed);
    EXPECT_NO_THROW(encoder->fill(in, sizes, out, pool, scratch));
}

TEST(EncoderGoals, SlicesOfABatchKeepTheirEntityReferences)
{
    const auto table = goals_table();
    const test::Rows rows = mixed_rows(*table, 4, 17);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        const auto encoder = FlatEncoder::create(table, goal_options(layout));
        const auto batch = encoder->encode(input_of(rows));
        const auto entity_offsets = ints(*batch, "entity_offsets");
        const auto role = ints(*batch, "entity_role");
        const auto ref = ints(*batch, "entity_ref");
        const u64 from = 3, to = 11;
        test::Rows slice;
        slice.words = rows.words;
        slice.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(from * rows.words),
                          rows.data.begin() + static_cast<std::ptrdiff_t>(to * rows.words));
        slice.task_ids.assign(rows.task_ids.begin() + static_cast<std::ptrdiff_t>(from),
                              rows.task_ids.begin() + static_cast<std::ptrdiff_t>(to));
        const auto part = encoder->encode(input_of(slice));
        const auto part_role = ints(*part, "entity_role");
        const auto part_ref = ints(*part, "entity_ref");
        const auto first = static_cast<std::ptrdiff_t>(entity_offsets[from]);
        const auto last = static_cast<std::ptrdiff_t>(entity_offsets[to]);
        EXPECT_EQ(part_role, std::vector<i64>(role.begin() + first, role.begin() + last));
        EXPECT_EQ(part_ref, std::vector<i64>(ref.begin() + first, ref.begin() + last));
        EXPECT_NE(std::count(part_role.begin(), part_role.end(), 2), 0);
    }
}

TEST(EncoderGoals, TheGoalIsIncompleteWhenALiteralWithoutAFixedTruthIsDropped)
{
    // goals/conjunctive has the nullary literal (done) over a derived predicate; blocks_flags/p01 has two over fluents
    using Problem = std::pair<const char*, const char*>;
    for (const auto& [domain, problem] : {Problem{"goals", "conjunctive"}, Problem{"blocks_flags", "p01"}})
    {
        SCOPED_TRACE(std::string(domain) + "/" + problem);
        const auto table = rl::TaskTable::single(test::pddl_task(domain, problem));
        const test::Rows rows = test::random_walk(*table, 0, 3, 2);
        auto complete = [&](const FlatOptions& options)
        { return ints(*FlatEncoder::create(table, options)->encode(input_of(rows)), "goal_complete"); };
        FlatOptions options = goal_options();
        ASSERT_TRUE(table->task(0)->encoding_goal_complete());
        EXPECT_EQ(complete(options), (std::vector<i64>{1, 1, 1}));
        options.nullary = false;
        EXPECT_EQ(complete(options), (std::vector<i64>{0, 0, 0}));
        // the equality predicate is static: dropping it leaves the goal as complete as it was
        options.nullary = true;
        options.equality = false;
        EXPECT_EQ(complete(options), (std::vector<i64>{1, 1, 1}));
    }
}

// ------------------------------------------------------------------------------- relations wider than the state's

TEST(EncoderGoals, GoalRelationsWiderThanEveryStateRelationAreEncodedWhole)
{
    // wide_predicates: on/1 is the only fluent predicate; link/3 is static and trio/3 derived, with objects a b c = 0 1
    // 2
    const auto table = rl::TaskTable::single(test::pddl_task("wide_predicates", "p01"));
    const Task& task = *table->task(0);
    const u32 words = table->words(), derived_words = std::max(1u, table->derived_words());
    const u32 on_a = test::atom_slot(task, AtomKind::Fluent, "on", {0});
    const u32 on_b = test::atom_slot(task, AtomKind::Fluent, "on", {1});
    const u32 on_c = test::atom_slot(task, AtomKind::Fluent, "on", {2});
    test::Rows rows;
    rows.words = words;
    rows.data.assign(words, 0);
    rows.task_ids = {0};
    test::set_bit(rows.data, on_a);
    test::set_bit(rows.data, on_b);
    // goal: (link a b c) (not (link b b b)) (trio a b c) (not (trio b c a)) (on b); in the state {(on a) (on b)} the
    // static literals and (on b) hold, (trio a b c) does not (c is not on), and (not (trio b c a)) holds
    const Named instance_goal = {
        {"[+]link[g]", {{0, 1, 2}}},        {"[+]link[g][sat]", {{0, 1, 2}}},
        {"[-]link[g]", {{1, 1, 1}}},        {"[-]link[g][sat]", {{1, 1, 1}}},
        {"[+]trio[g]", {{0, 1, 2}}},        {"[+]trio[g][unsat]", {{0, 1, 2}}},
        {"[-]trio[g]", {{1, 2, 0}}},        {"[-]trio[g][sat]", {{1, 2, 0}}},
        {"[+]on[g]", {{1}}},                {"[+]on[g][sat]", {{1}}}};
    // goals of the row: (on a), (not (on c)), the derived (trio a b a) and (not (trio b c a)); all four hold
    std::vector<u64> pos(words, 0), neg(words, 0), dpos(derived_words, 0), dneg(derived_words, 0);
    test::set_bit(pos, on_a);
    test::set_bit(neg, on_c);
    test::set_bit(dpos, test::atom_slot(task, AtomKind::Derived, "trio", {0, 1, 0}));
    test::set_bit(dneg, test::atom_slot(task, AtomKind::Derived, "trio", {1, 2, 0}));
    const Named row_goal = {
        {"[+]on[g]", {{0}}},
        {"[+]on[g][sat]", {{0}}},
        {"[-]on[g]", {{2}}},
        {"[-]on[g][sat]", {{2}}},
        {"[+]trio[g]", {{0, 1, 0}}},
        {"[+]trio[g][sat]", {{0, 1, 0}}},
        {"[-]trio[g]", {{1, 2, 0}}},
        {"[-]trio[g][sat]", {{1, 2, 0}}}};
    const std::vector<Tuple> on_tuples = {{0}, {1}};
    const std::vector<Tuple> link_tuples = {{0, 1, 2}, {2, 1, 0}};
    const std::vector<Tuple> trio_tuples = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    for (const Layout layout : {Layout::Graph, Layout::Relation})
        for (const bool include_static : {true, false})
            for (const bool include_derived : {true, false})
            {
                SCOPED_TRACE(std::string(layout == Layout::Graph ? "graph" : "relation") +
                             (include_static ? "" : " without static") + (include_derived ? "" : " without derived"));
                FlatOptions options = goal_options(layout, false);
                options.include_static = include_static;
                options.include_derived = include_derived;
                const auto encoder = FlatEncoder::create(table, options);
                const EncodingSchema& schema = encoder->schema();
                EXPECT_EQ(encoder->context().max_arity(), 3u);
                EXPECT_EQ(schema.find("link").has_value(), include_static);
                EXPECT_EQ(schema.find("trio").has_value(), include_derived);
                for (const bool row_goals : {false, true})
                {
                    EncodeInput in = input_of(rows);
                    if (row_goals)
                    {
                        in.goal_pos = pos.data();
                        in.goal_neg = neg.data();
                        in.goal_words = words;
                        in.goal_derived_pos = dpos.data();
                        in.goal_derived_neg = dneg.data();
                        in.goal_derived_words = derived_words;
                    }
                    const auto bundle = encoder->encode(in);
                    validate(*bundle, schema);
                    const auto graphs = decode(*bundle, schema);
                    expect_goals(*bundle, schema, graphs, {row_goals ? row_goal : instance_goal});
                    EXPECT_EQ(graphs[0].tuples[*schema.find("on")], on_tuples);
                    if (include_static)
                    {
                        EXPECT_EQ(graphs[0].tuples[*schema.find("link")], link_tuples);
                    }
                    if (include_derived)
                    {
                        EXPECT_EQ(graphs[0].tuples[*schema.find("trio")], trio_tuples);
                    }
                    EXPECT_EQ(ints(*bundle, "goal_complete"), (std::vector<i64>{1}));
                }
            }
}

}  // namespace
