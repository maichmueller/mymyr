// The horizon batches of the relational encoder: graphs that are a root state and the nodes below it.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
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

// ---------------------------------------------------------------------------------------------------- the reference

struct RefAtom
{
    std::string name;
    std::vector<i32> objects;
    [[nodiscard]] friend bool operator<(const RefAtom& a, const RefAtom& b) { return std::tie(a.name, a.objects) < std::tie(b.name, b.objects); }
    [[nodiscard]] friend bool operator==(const RefAtom& a, const RefAtom& b) = default;
};
using AtomSet = std::set<RefAtom>;

struct RefLiteral
{
    RefAtom atom;
    bool positive;
};

/// The fluent atoms a row sets.
AtomSet fluent_atoms(const Task& task, const u64* row)
{
    AtomSet out;
    for (u32 slot = 0; slot < task.atoms().fluent_slots(); ++slot)
        if ((row[slot >> 6] >> (slot & 63)) & 1)
        {
            const u32* record = task.atoms().record(AtomKind::Fluent, slot);
            RefAtom atom{std::string(task.data().str(task.data().predicates[record[0]].name)), {}};
            for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
                atom.objects.push_back(static_cast<i32>(record[1 + k]));
            out.insert(std::move(atom));
        }
    return out;
}

std::vector<RefLiteral> instance_goal(const Task& task)
{
    std::vector<RefLiteral> out;
    for (const formalism::Literal& literal : task.encoding_goal())
    {
        RefAtom atom{std::string(task.data().str(task.data().predicates[literal.pred.v].name)), {}};
        for (formalism::Term term : task.data().terms_of(literal))
            atom.objects.push_back(static_cast<i32>(formalism::term_object(term).v));
        out.push_back({std::move(atom), literal.positive});
    }
    return out;
}

/// The goal literals a pair of masks over the fluent slots names.
std::vector<RefLiteral> mask_goal(const Task& task, const std::vector<u64>& pos, const std::vector<u64>& neg)
{
    std::vector<RefLiteral> out;
    for (const bool positive : {true, false})
    {
        const AtomSet atoms = fluent_atoms(task, (positive ? pos : neg).data());
        for (const RefAtom& atom : atoms)
            out.push_back({atom, positive});
    }
    return out;
}

/// One graph as the reference reads it.
struct RefGraph
{
    std::vector<AtomSet> nodes;
    std::vector<std::array<i32, 2>> edges;       // (src, dst)
    std::vector<std::pair<std::string, std::vector<i32>>> actions;  // (relation of the edge's schema, objects)
    std::vector<RefLiteral> goal;
};

std::string join(const std::vector<std::string>& parts)
{
    std::string out;
    for (const std::string& part : parts)
        out += (out.empty() ? "" : ",") + part;
    return out;
}

/// The tuples relation `id` has in the graph, as sorted lists of entity names ("o<object>", "aux", "s<node>", "v<key>").
std::vector<std::string> expected_tuples(const EncodingSchema& schema, const rl::TaskTable& table, u32 id, const RefGraph& g)
{
    const Relation& r = schema.relation(id);
    const FlatOptions& o = schema.options();
    const bool exclude = o.horizon.root == RootPolicy::Exclude;
    const bool full = o.horizon.mode == HorizonMode::Full;
    const bool split = full && exclude;
    const bool lead = schema.slot_roles(id).front() == SlotRole::State;
    const u32 nodes = static_cast<u32>(g.nodes.size());
    std::vector<std::string> out;
    auto tuple = [&](std::vector<std::string> head, const std::vector<i32>& objects, bool with_pred)
    {
        if (with_pred && o.predicate_virtual_nodes)
            head.push_back("v" + std::to_string(schema.virtual_key(id)));
        if (objects.empty() && r.logical_arity == 0 && r.lane != Lane::Action)
            head.push_back("aux");
        for (const i32 object : objects)
            head.push_back("o" + std::to_string(object));
        out.push_back(join(head));
    };
    auto state = [&](u32 node) { return lead ? std::vector<std::string>{"s" + std::to_string(node)} : std::vector<std::string>{}; };
    const std::string& predicate = r.predicate >= 0 ? table.predicate_names()[static_cast<usize>(r.predicate)] : std::string();
    switch (r.lane)
    {
        case Lane::Fluent:
        case Lane::Derived:
        case Lane::Static:
        {
            if (r.lane != Lane::Fluent)
                break;
            auto facts = [&](u32 node)
            {
                for (const RefAtom& atom : g.nodes[node])
                    if (atom.name == predicate)
                        tuple(state(node), atom.objects, true);
            };
            if (r.sign == 0)
            {
                if (!r.state_anchored)
                {
                    facts(0);
                    if (full && !split)
                        for (u32 node = 1; node < nodes; ++node)
                            facts(node);
                }
                else
                    for (u32 node = 1; node < nodes; ++node)
                        facts(node);
            }
            else
                for (u32 node = 1; node < nodes; ++node)
                {
                    const AtomSet& from = r.sign > 0 ? g.nodes[node] : g.nodes[0];
                    const AtomSet& without = r.sign > 0 ? g.nodes[0] : g.nodes[node];
                    for (const RefAtom& atom : from)
                        if (atom.name == predicate && !without.count(atom))
                            tuple(state(node), atom.objects, true);
                }
            break;
        }
        case Lane::Goal:
            if (r.level.value_or(0) == 0)
                for (const RefLiteral& literal : g.goal)
                    if (literal.atom.name == predicate && literal.positive == (r.sign > 0))
                        tuple(state(0), literal.atom.objects, true);
            break;
        case Lane::GoalSatisfaction:
        {
            if (!r.level || *r.level != 0)
                break;
            const GoalDerivation d = *r.derivation;
            for (const RefLiteral& literal : g.goal)
            {
                if (literal.atom.name != predicate || literal.positive != (r.sign > 0))
                    continue;
                if (d == GoalDerivation::Satisfied || d == GoalDerivation::Unsatisfied)
                {
                    std::vector<u32> scope;
                    if (!r.state_anchored)
                        scope.push_back(0);
                    if (r.state_anchored || (full && !split))
                        for (u32 node = 1; node < nodes; ++node)
                            scope.push_back(node);
                    for (const u32 node : scope)
                    {
                        const bool holds = (g.nodes[node].count(literal.atom) != 0) == literal.positive;
                        if (holds == (d == GoalDerivation::Satisfied))
                            tuple(state(node), literal.atom.objects, true);
                    }
                }
                else
                    for (u32 node = 1; node < nodes; ++node)
                    {
                        const bool now = g.nodes[node].count(literal.atom) != 0, before = g.nodes[0].count(literal.atom) != 0;
                        const bool added = now && !before, deleted = before && !now;
                        const bool newly_satisfied = literal.positive ? added : deleted;
                        const bool newly_falsified = literal.positive ? deleted : added;
                        if (d == GoalDerivation::AddedSatisfied ? newly_satisfied : newly_falsified)
                            tuple(state(node), literal.atom.objects, true);
                    }
            }
            break;
        }
        case Lane::Action:
            for (usize e = 0; e < g.edges.size(); ++e)
            {
                if (exclude && g.edges[e][1] == 0)
                    continue;
                if (g.actions[e].first == r.name)
                    tuple(state(static_cast<u32>(g.edges[e][1])), g.actions[e].second, false);
            }
            break;
        case Lane::Parent:
            for (const auto& edge : g.edges)
                if (!exclude || (edge[0] != 0 && edge[1] != 0))
                    out.push_back(join({"s" + std::to_string(edge[0]), "s" + std::to_string(edge[1])}));
            break;
        case Lane::Sibling:
        case Lane::Cousin:
        {
            // from the definition: siblings share a parent, cousins have parents that are siblings
            std::vector<std::set<u32>> parents(nodes);
            for (const auto& edge : g.edges)
                parents[static_cast<u32>(edge[1])].insert(static_cast<u32>(edge[0]));
            auto siblings = [&](u32 a, u32 b)
            {
                if (a == b)
                    return false;
                return std::ranges::any_of(parents[a], [&](u32 p) { return parents[b].count(p) != 0; });
            };
            for (u32 a = 0; a < nodes; ++a)
                for (u32 b = 0; b < nodes; ++b)
                {
                    if (a == b || (exclude && (a == 0 || b == 0)))
                        continue;
                    bool related = false;
                    if (r.lane == Lane::Sibling)
                        related = siblings(a, b);
                    else
                        for (const u32 p : parents[a])
                            for (const u32 q : parents[b])
                                related = related || siblings(p, q);
                    if (related)
                        out.push_back(join({"s" + std::to_string(a), "s" + std::to_string(b)}));
                }
            break;
        }
        default: break;
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// The graphs of a horizon batch as the reference reads them.
std::vector<RefGraph> reference_graphs(const rl::TaskTable& table, const EncodingSchema& schema, const test::Horizon& h,
                                       const std::vector<std::vector<RefLiteral>>* goals = nullptr)
{
    std::vector<RefGraph> out;
    for (u64 g = 0; g < h.graphs(); ++g)
    {
        const Task& task = *table.instance(static_cast<u32>(h.task_ids[g])).task;
        RefGraph graph;
        for (i32 row = h.node_offsets[g]; row < h.node_offsets[g + 1]; ++row)
            graph.nodes.push_back(fluent_atoms(task, h.nodes.data() + static_cast<u64>(row) * h.words));
        for (i32 e = h.edge_offsets[g]; e < h.edge_offsets[g + 1]; ++e)
        {
            graph.edges.push_back({h.edge_src[static_cast<usize>(e)], h.edge_dst[static_cast<usize>(e)]});
            const usize s = static_cast<usize>(h.edge_schema[static_cast<usize>(e)]);
            std::vector<i32> objects;
            for (u32 k = 0; k < table.schema_arities()[s]; ++k)
                objects.push_back(h.edge_binding[static_cast<u64>(e) * h.label_width + k]);
            const i32 relation = schema.options().include_actions ? schema.schema_relations()[s] : -1;
            graph.actions.emplace_back(relation < 0 ? std::string() : schema.relation(static_cast<u32>(relation)).name,
                                       std::move(objects));
        }
        graph.goal = goals ? (*goals)[g] : instance_goal(task);
        out.push_back(std::move(graph));
    }
    return out;
}

// ----------------------------------------------------------------------------------------------------- the decoder

/// The tuples of every cell of a batch, by graph and relation name, as lists of entity names in file order.
using Decoded = std::vector<std::map<std::string, std::vector<std::string>>>;

Decoded decode(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const u64 G = static_cast<u64>(batch.scalar("num_graphs"));
    const u64 R = schema.num_relations();
    const auto entity_offsets = ints(batch, "entity_offsets");
    const auto role = ints(batch, "entity_role");
    const auto ref = ints(batch, "entity_ref");
    const auto tuple_offsets = ints(batch, "relation_offsets");
    const auto arg_offsets = ints(batch, "relation_arg_offsets");
    const auto args = ints(batch, "relation_args");
    const bool relation_major = schema.options().layout == Layout::Relation;
    auto name = [&](i64 entity)
    {
        const std::string r = std::to_string(ref[static_cast<u64>(entity)]);
        switch (role[static_cast<u64>(entity)])
        {
            case 0: return "o" + r;
            case 1: return "v" + r;
            case 7: return "s" + r;
            case 8: return std::string("aux");
            default: return "?" + r;
        }
    };
    Decoded out(G);
    for (u64 g = 0; g < G; ++g)
        for (u64 r = 0; r < R; ++r)
        {
            const u64 cell = relation_major ? r * G + g : g * R + r;
            const u32 arity = schema.relation(static_cast<u32>(r)).arity;
            auto& tuples = out[g][schema.relation(static_cast<u32>(r)).name];
            for (i64 t = tuple_offsets[cell]; t < tuple_offsets[cell + 1]; ++t)
            {
                std::vector<std::string> names;
                for (u32 k = 0; k < arity; ++k)
                    names.push_back(name(args[static_cast<u64>(arg_offsets[cell]) + static_cast<u64>(t - tuple_offsets[cell]) * arity + k]));
                tuples.push_back(join(names));
            }
        }
    (void)entity_offsets;
    return out;
}

void expect_reference(const FlatEncoder& encoder, const test::Horizon& h, const rl::ArrayBundle& batch,
                      const std::vector<std::vector<RefLiteral>>* goals = nullptr)
{
    const EncodingSchema& schema = encoder.schema();
    EXPECT_NO_THROW(validate(batch, schema));
    const Decoded decoded = decode(batch, schema);
    const std::vector<RefGraph> graphs = reference_graphs(encoder.table(), schema, h, goals);
    for (u64 g = 0; g < graphs.size(); ++g)
        for (u32 id = 0; id < schema.num_relations(); ++id)
        {
            std::vector<std::string> actual = decoded[g].at(schema.relation(id).name);
            std::sort(actual.begin(), actual.end());
            EXPECT_EQ(actual, expected_tuples(schema, encoder.table(), id, graphs[g]))
                << "graph " << g << ", relation " << schema.relation(id).name;
        }
}

// ----------------------------------------------------------------------------------------------------------- tests

FlatOptions horizon_options(HorizonMode mode, RootPolicy root)
{
    FlatOptions options;
    options.include_static = false;
    options.horizon.mode = mode;
    options.horizon.root = root;
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    if (mode == HorizonMode::Delta)
    {
        options.goal_derivations.insert(GoalDerivation::AddedSatisfied);
        options.goal_derivations.insert(GoalDerivation::AddedUnsatisfied);
    }
    return options;
}

struct Case
{
    const char* domain;
    const char* problem;
};
const Case k_cases[] = {{"blocks", "smedium"}, {"blocks_flags", "p01"}, {"logistics00", "probLOGISTICS-6-1"}};

constexpr HorizonMode k_modes[] = {HorizonMode::Full, HorizonMode::Delta, HorizonMode::Action};
constexpr RootPolicy k_roots[] = {RootPolicy::Include, RootPolicy::EncodeOnly, RootPolicy::Exclude};

TEST(Horizon, MatchesTheReference)
{
    for (const Case& c : k_cases)
    {
        const auto table = rl::TaskTable::single(test::pddl_task(c.domain, c.problem));
        const test::Rows roots = test::random_walk(*table, 0, 3, 7);
        const test::Horizon h = test::horizon_of(*table, roots, 2, 5);
        ASSERT_GT(h.num_nodes(), roots.size());
        for (const HorizonMode mode : k_modes)
            for (const RootPolicy root : k_roots)
                for (const bool parent : {false, true})
                    for (const Layout layout : {Layout::Graph, Layout::Relation})
                    {
                        FlatOptions options = horizon_options(mode, root);
                        options.horizon.parent_relation = parent;
                        options.horizon.sibling_relation = parent;
                        options.horizon.cousin_relation = parent;
                        options.layout = layout;
                        options.index = parent ? IndexType::I64 : IndexType::I32;
                        SCOPED_TRACE(std::string(c.domain) + " " + std::string(to_string(mode)) + " " +
                                     std::string(to_string(root)) + (parent ? " parent" : "") +
                                     (layout == Layout::Relation ? " relation-major" : ""));
                        const auto encoder = FlatEncoder::create(table, options);
                        expect_reference(*encoder, h, *encoder->encode(h.input()));
                    }
    }
}

TEST(Horizon, RowGoalsAndNullaryRelations)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks_flags", "p01"));
    const Task& task = *table->instance(0).task;
    const test::Rows roots = test::random_walk(*table, 0, 2, 3);
    const test::Horizon h = test::horizon_of(*table, roots, 2, 4);
    // the goal of each graph: a positive and a negative literal
    std::vector<u64> pos(2 * table->words(), 0), neg(2 * table->words(), 0);
    std::vector<std::vector<RefLiteral>> goals;
    for (u64 g = 0; g < 2; ++g)
    {
        std::vector<u64> p(table->words(), 0), n(table->words(), 0);
        test::set_bit(p, test::atom_slot(task, AtomKind::Fluent, "moved", {}));
        test::set_bit(n, test::atom_slot(task, AtomKind::Fluent, "handempty", {}));
        if (g == 1)
            test::set_bit(p, test::atom_slot(task, AtomKind::Fluent, "stacked", {}));
        std::copy(p.begin(), p.end(), pos.begin() + static_cast<std::ptrdiff_t>(g * table->words()));
        std::copy(n.begin(), n.end(), neg.begin() + static_cast<std::ptrdiff_t>(g * table->words()));
        goals.push_back(mask_goal(task, p, n));
    }
    for (const HorizonMode mode : k_modes)
        for (const bool nullary : {true, false})
        {
            FlatOptions options = horizon_options(mode, RootPolicy::Include);
            options.nullary = nullary;
            const auto encoder = FlatEncoder::create(table, options);
            EncodeInput in = h.input();
            in.goal_pos = pos.data();
            in.goal_neg = neg.data();
            in.goal_stride = table->words();
            in.goal_words = table->words();
            const auto batch = encoder->encode(in);
            expect_reference(*encoder, h, *batch, &goals);
            // without the nullary relations the batch has no auxiliary entity, with them it has one where a nullary atom is
            const auto role = ints(*batch, "entity_role");
            EXPECT_EQ(std::count(role.begin(), role.end(), 8) != 0, nullary) << to_string(mode);
        }
}

TEST(Horizon, DeltasToTheRootNotToTheParent)
{
    // blocks smedium: the root unstacks a from b in one step, then puts it down or stacks it back
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const Task& task = *table->instance(0).task;
    const test::Rows roots = test::random_walk(*table, 0, 1, 0);
    const test::Horizon h = test::horizon_of(*table, roots, 2);
    ASSERT_EQ(h.num_nodes(), 4u);  // root, unstack, then put-down and stack
    FlatOptions options = horizon_options(HorizonMode::Delta, RootPolicy::Exclude);
    options.include_actions = false;
    const auto encoder = FlatEncoder::create(table, options);
    const Decoded decoded = decode(*encoder->encode(h.input()), encoder->schema());
    // the objects: a is the clear one at the root, b the one it is on, c the table block
    const AtomSet root = fluent_atoms(task, h.nodes.data());
    i32 a = -1, b = -1;
    for (const RefAtom& atom : root)
        if (atom.name == "clear")
            a = atom.objects[0];
    for (const RefAtom& atom : root)
        if (atom.name == "on" && atom.objects[0] == a)
            b = atom.objects[1];
    ASSERT_GE(a, 0);
    ASSERT_GE(b, 0);
    const auto o = [](i32 object) { return "o" + std::to_string(object); };
    const auto at = [&](const char* relation) { return decoded[0].at(relation); };
    auto sorted = [](std::vector<std::string> v) { std::sort(v.begin(), v.end()); return v; };
    // node 1 (unstack a b): holding a, clear b appear; clear a, handempty, on a b go
    EXPECT_EQ(sorted(at("[+]holding")), sorted({"s1," + o(a)}));
    EXPECT_EQ(sorted(at("[+]clear")), sorted({"s1," + o(b), "s2," + o(b)}));
    // node 2 (put-down a) differs from the root in on a b (gone), clear b and ontable a (new); the atoms it differs from
    // its parent in (holding a, clear a, handempty) are not part of its delta
    EXPECT_EQ(sorted(at("[+]ontable")), sorted({"s2," + o(a)}));
    EXPECT_EQ(sorted(at("[-]on")), sorted({"s1," + o(a) + "," + o(b), "s2," + o(a) + "," + o(b)}));
    EXPECT_EQ(sorted(at("[-]handempty")), sorted({"s1,aux"}));
    EXPECT_EQ(sorted(at("[-]clear")), sorted({"s1," + o(a)}));
    // node 3 (stack a b) is the root again: no node-3 tuple in any delta relation
    for (const auto& [name, tuples] : decoded[0])
    {
        if (name.rfind("[+]", 0) == 0 || name.rfind("[-]", 0) == 0)
        {
            for (const std::string& tuple : tuples)
                EXPECT_NE(tuple.rfind("s3", 0), 0u) << name;
        }
    }
    // the root's own atoms have no state slot
    EXPECT_EQ(at("ontable").size(), 1u);
    EXPECT_EQ(at("ontable")[0].rfind("s", 0), std::string::npos);
}

TEST(Horizon, SelfLoopsHaveEmptyDeltas)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 1, 0);
    test::Horizon h = test::horizon_of(*table, roots, 0);
    // a node equal to its root, reached by a loop edge, and a node equal to its parent
    const std::vector<u64> root(h.nodes.begin(), h.nodes.begin() + h.words);
    for (int copies = 0; copies < 2; ++copies)
    {
        h.nodes.insert(h.nodes.end(), root.begin(), root.end());
        h.edge_src.push_back(copies);
        h.edge_dst.push_back(copies + 1);
        h.edge_schema.push_back(0);
        h.edge_binding.insert(h.edge_binding.end(), h.label_width, 0);
    }
    h.node_offsets.back() = 3;
    h.edge_offsets.back() = 2;
    for (const HorizonMode mode : {HorizonMode::Delta, HorizonMode::Full})
    {
        FlatOptions options = horizon_options(mode, RootPolicy::Include);
        const auto encoder = FlatEncoder::create(table, options);
        const Decoded decoded = decode(*encoder->encode(h.input()), encoder->schema());
        for (const auto& [name, tuples] : decoded[0])
        {
            const bool delta = name.rfind("[+]", 0) == 0 || name.rfind("[-]", 0) == 0;
            if (mode == HorizonMode::Delta && delta && name.find("[g]") == std::string::npos)
            {
                EXPECT_TRUE(tuples.empty()) << name;
            }
            if (name.find("[sat+]") != std::string::npos || name.find("[sat-]") != std::string::npos)
            {
                EXPECT_TRUE(tuples.empty()) << name;
            }
        }
    }
}

TEST(Horizon, NodeOrderIsInputOrder)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 1, 0);
    test::Horizon h = test::horizon_of(*table, roots, 2);
    // reverse the nodes below the root (edges follow their endpoints)
    const i32 count = static_cast<i32>(h.num_nodes());
    std::vector<u64> nodes(h.nodes.begin(), h.nodes.begin() + h.words);
    for (i32 node = count - 1; node >= 1; --node)
        nodes.insert(nodes.end(), h.nodes.begin() + static_cast<std::ptrdiff_t>(node * h.words),
                     h.nodes.begin() + static_cast<std::ptrdiff_t>((node + 1) * h.words));
    h.nodes = nodes;
    for (i32& v : h.edge_src)
        v = v == 0 ? 0 : count - v;
    for (i32& v : h.edge_dst)
        v = count - v;
    for (const RootPolicy root : k_roots)
    {
        const auto encoder = FlatEncoder::create(table, horizon_options(HorizonMode::Full, root));
        const auto batch = encoder->encode(h.input());
        expect_reference(*encoder, h, *batch);
        // the state entities refer to their nodes in input order: entity_ref is the graph-local node index
        const auto role = ints(*batch, "entity_role");
        const auto ref = ints(*batch, "entity_ref");
        std::vector<i64> refs;
        for (usize e = 0; e < role.size(); ++e)
            if (role[e] == 7)
                refs.push_back(ref[e]);
        std::vector<i64> expected;
        for (i64 node = root == RootPolicy::Exclude ? 1 : 0; node < count; ++node)
            expected.push_back(node);
        EXPECT_EQ(refs, expected);
        const auto target_entity = ints(*batch, "target_entity");
        const auto target_group = ints(*batch, "target_group");
        EXPECT_EQ(target_entity.size(), static_cast<usize>(count - (root == RootPolicy::Include ? 0 : 1)));
        for (const i64 group : target_group)
            EXPECT_EQ(group, 4);
    }
}

TEST(Horizon, LayoutsAgreeAndGraphsAreIndependent)
{
    const auto small = test::pddl_task("blocks", "smedium");
    const auto large = test::pddl_task("blocks", "probBLOCKS-8-0");
    const auto table = rl::TaskTable::create({small, large});
    test::Rows roots = test::random_walk(*table, 0, 3, 1);
    const test::Rows more = test::random_walk(*table, 1, 4, 2);
    roots.data.insert(roots.data.end(), more.data.begin(), more.data.end());
    roots.task_ids.insert(roots.task_ids.end(), more.task_ids.begin(), more.task_ids.end());
    // graphs of different depth and size
    auto one_row = [&](u64 g)
    {
        test::Rows one;
        one.words = roots.words;
        one.data.assign(roots.data.begin() + static_cast<std::ptrdiff_t>(g * roots.words),
                        roots.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * roots.words));
        one.task_ids = {roots.task_ids[g]};
        return one;
    };
    test::Horizon h = test::horizon_of(*table, one_row(0), 2);
    for (u64 g = 1; g < roots.size(); ++g)
    {
        const test::Rows one = one_row(g);
        const test::Horizon next = test::horizon_of(*table, one, g % 3, 3);
        const i32 node_base = h.node_offsets.back(), edge_base = h.edge_offsets.back();
        h.nodes.insert(h.nodes.end(), next.nodes.begin(), next.nodes.end());
        h.task_ids.push_back(next.task_ids[0]);
        h.node_offsets.push_back(node_base + next.node_offsets[1]);
        h.edge_offsets.push_back(edge_base + next.edge_offsets[1]);
        h.edge_src.insert(h.edge_src.end(), next.edge_src.begin(), next.edge_src.end());
        h.edge_dst.insert(h.edge_dst.end(), next.edge_dst.begin(), next.edge_dst.end());
        h.edge_schema.insert(h.edge_schema.end(), next.edge_schema.begin(), next.edge_schema.end());
        h.edge_binding.insert(h.edge_binding.end(), next.edge_binding.begin(), next.edge_binding.end());
    }
    for (const HorizonMode mode : k_modes)
    {
        FlatOptions graph_major = horizon_options(mode, RootPolicy::EncodeOnly);
        graph_major.horizon.parent_relation = true;
        FlatOptions relation_major = graph_major;
        relation_major.layout = Layout::Relation;
        const auto a = FlatEncoder::create(table, graph_major);
        const auto b = FlatEncoder::create(table, relation_major);
        const auto batch_a = a->encode(h.input());
        const auto batch_b = b->encode(h.input());
        expect_reference(*a, h, *batch_a);
        EXPECT_EQ(decode(*batch_a, a->schema()), decode(*batch_b, b->schema())) << to_string(mode);
        // every graph alone is the graph of the batch
        const Decoded all = decode(*batch_a, a->schema());
        for (u64 g = 0; g < h.graphs(); ++g)
        {
            test::Horizon single;
            single.words = h.words;
            single.label_width = h.label_width;
            single.task_ids = {h.task_ids[g]};
            single.nodes.assign(h.nodes.begin() + static_cast<std::ptrdiff_t>(h.node_offsets[g] * h.words),
                                h.nodes.begin() + static_cast<std::ptrdiff_t>(h.node_offsets[g + 1] * h.words));
            single.node_offsets = {0, h.node_offsets[g + 1] - h.node_offsets[g]};
            single.edge_offsets = {0, h.edge_offsets[g + 1] - h.edge_offsets[g]};
            const auto lo = static_cast<std::ptrdiff_t>(h.edge_offsets[g]), hi = static_cast<std::ptrdiff_t>(h.edge_offsets[g + 1]);
            single.edge_src.assign(h.edge_src.begin() + lo, h.edge_src.begin() + hi);
            single.edge_dst.assign(h.edge_dst.begin() + lo, h.edge_dst.begin() + hi);
            single.edge_schema.assign(h.edge_schema.begin() + lo, h.edge_schema.begin() + hi);
            single.edge_binding.assign(h.edge_binding.begin() + lo * h.label_width, h.edge_binding.begin() + hi * h.label_width);
            const Decoded alone = decode(*a->encode(single.input()), a->schema());
            EXPECT_EQ(alone[0], all[g]) << "graph " << g << " " << to_string(mode);
        }
    }
}

bool same_bytes(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    if (a.arrays().size() != b.arrays().size())
        return false;
    for (usize i = 0; i < a.arrays().size(); ++i)
    {
        const rl::ArrayInfo& x = a.arrays()[i];
        const rl::ArrayInfo& y = b.arrays()[i];
        if (x.name != y.name || x.shape != y.shape || x.dtype != y.dtype ||
            std::memcmp(a.data(x), b.data(y), x.elements() * rl::dtype_bytes(x.dtype)) != 0)
            return false;
    }
    return true;
}

TEST(Horizon, ThreadsGiveIdenticalBatches)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "probBLOCKS-8-0"));
    const test::Rows roots = test::random_walk(*table, 0, 200, 5);
    test::Horizon h = test::horizon_of(*table, roots, 2, 6);
    ASSERT_GT(h.num_nodes(), 3 * k_graphs_per_member);  // several members have work
    for (const HorizonMode mode : k_modes)
        for (const Layout layout : {Layout::Graph, Layout::Relation})
            for (const IndexType index : {IndexType::I32, IndexType::I64})
            {
                FlatOptions options = horizon_options(mode, RootPolicy::Include);
                options.horizon.parent_relation = true;
                options.layout = layout;
                options.index = index;
                const auto encoder = FlatEncoder::create(table, options);
                const auto one = encoder->encode(h.input());
                for (const u32 threads : {2u, 3u, 8u})
                {
                    ThreadPool pool(threads);
                    EncodeScratch scratch;
                    const auto many = encoder->encode(h.input(), pool, scratch);
                    EXPECT_TRUE(same_bytes(*one, *many)) << threads << " threads, " << to_string(mode);
                }
            }
}

TEST(Horizon, CapacityForm)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 5, 9);
    const test::Horizon h = test::horizon_of(*table, roots, 2, 3);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions options = horizon_options(HorizonMode::Delta, RootPolicy::Include);
        options.layout = layout;
        options.horizon.parent_relation = true;
        const auto encoder = FlatEncoder::create(table, options);
        ThreadPool pool(2);
        EncodeScratch scratch;
        const auto exact = encoder->encode(h.input(), pool, scratch);
        const FlatSizes sizes = encoder->count(h.input(), pool, scratch);
        EXPECT_EQ(sizes.targets, h.num_nodes());  // every node is a candidate under `include`
        FlatCapacities capacities{sizes.entities + 8, sizes.tuples, sizes.args + 16, sizes.targets + 3, 0};
        FlatSizes reported;
        const auto padded = encoder->encode_capacity(h.input(), capacities, pool, scratch, reported);
        EXPECT_EQ(reported.overflow, 0u);
        EXPECT_EQ(reported.args, sizes.args);
        EXPECT_NO_THROW(validate(*padded, encoder->schema()));
        // a capacity that is too small reports the true sizes and writes nothing
        capacities.args = sizes.args - 1;
        const auto cut = encoder->encode_capacity(h.input(), capacities, pool, scratch, reported);
        EXPECT_EQ(reported.overflow, k_overflow_capacity);
        EXPECT_EQ(reported.args, sizes.args);
        for (const rl::ArrayInfo& info : cut->arrays())
        {
            const auto* bytes = static_cast<const u8*>(cut->data(info));
            EXPECT_TRUE(std::all_of(bytes, bytes + info.elements() * rl::dtype_bytes(info.dtype), [](u8 b) { return b == 0; }))
                << info.name;
        }
        (void)exact;
    }
}

TEST(Horizon, PredicateVirtualNodes)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 2, 4);
    const test::Horizon h = test::horizon_of(*table, roots, 2, 3);
    for (const HorizonMode mode : k_modes)
        for (const Layout layout : {Layout::Graph, Layout::Relation})
        {
            FlatOptions options = horizon_options(mode, RootPolicy::EncodeOnly);
            options.predicate_virtual_nodes = true;
            options.layout = layout;
            const auto encoder = FlatEncoder::create(table, options);
            expect_reference(*encoder, h, *encoder->encode(h.input()));
        }
}

TEST(Horizon, EmptyBatchAndRootOnlyGraphs)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto encoder = FlatEncoder::create(table, horizon_options(HorizonMode::Full, RootPolicy::Exclude));
    EncodeInput empty;
    const auto none = encoder->encode(empty);
    EXPECT_EQ(none->scalar("num_graphs"), 0);
    const test::Rows roots = test::random_walk(*table, 0, 3, 2);
    const test::Horizon h = test::horizon_of(*table, roots, 0);
    const auto batch = encoder->encode(h.input());
    EXPECT_EQ(ints(*batch, "target_entity").size(), 0u);
    expect_reference(*encoder, h, *batch);
}

TEST(Horizon, InvalidInput)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 2, 2);
    const test::Horizon base = test::horizon_of(*table, roots, 2, 3);
    FlatOptions options = horizon_options(HorizonMode::Full, RootPolicy::Include);
    options.horizon.parent_relation = true;
    const auto encoder = FlatEncoder::create(table, options);
    EXPECT_NO_THROW((void)encoder->encode(base.input()));
    auto refused = [&](const char* what, const std::string& message, auto change)
    {
        test::Horizon h = base;
        EncodeInput in;
        change(h, in);
        in.horizon = h.view();
        if (!in.task_ids)
            in.task_ids = h.task_ids.data();
        try
        {
            (void)encoder->encode(in);
            ADD_FAILURE() << what << ": accepted";
        }
        catch (const std::invalid_argument& error)
        {
            const std::string text = error.what();
            EXPECT_NE(text.find("mymyr: encoding: "), std::string::npos) << what << ": " << text;
            EXPECT_NE(text.find(message), std::string::npos) << what << ": " << text;
        }
    };
    refused("offsets", "node_offsets must start at 0", [](test::Horizon& h, EncodeInput&) { h.node_offsets[0] = 1; });
    refused("empty graph", "graph 1 of the horizon has no nodes",
            [](test::Horizon& h, EncodeInput&) { h.node_offsets[2] = h.node_offsets[1]; });
    refused("edge offsets", "the edge_offsets of graph 1", [](test::Horizon& h, EncodeInput&)
            { h.edge_offsets[1] = h.edge_offsets[2] + 1; });
    refused("too many nodes", "node batch has", [](test::Horizon& h, EncodeInput&) { h.node_offsets[2] += 100; });
    refused("endpoint", "edge 1 of graph 0 of the horizon has an endpoint outside the graph's nodes",
            [](test::Horizon& h, EncodeInput&) { h.edge_dst[1] = 99; });
    refused("negative endpoint", "an endpoint outside", [](test::Horizon& h, EncodeInput&) { h.edge_src[0] = -1; });
    refused("schema", "a schema outside the table's", [](test::Horizon& h, EncodeInput&) { h.edge_schema[0] = 99; });
    refused("object", "an object its instance does not have",
            [](test::Horizon& h, EncodeInput&) { h.edge_binding[0] = 99; });
    refused("label width", "label width", [](test::Horizon& h, EncodeInput&) { h.label_width = 0; });
    refused("states", "in.states must be empty",
            [&](test::Horizon&, EncodeInput& in) { in.states = roots.view(); });
    refused("task id", "task id 3 of graph 1 of the horizon", [](test::Horizon& h, EncodeInput&) { h.task_ids[1] = 3; });
    refused("foreign bits", "node 1 of graph 0 of the horizon sets an atom bit its instance does not have",
            [&](test::Horizon& h, EncodeInput&)
            { h.nodes[static_cast<usize>(h.words) + h.words - 1] |= u64{1} << 63; });
    // node rows narrower than the instance
    {
        EncodeInput in = base.input();
        in.horizon.nodes.words = 0;
        try
        {
            (void)encoder->encode(in);
            ADD_FAILURE() << "narrow rows were accepted";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("the nodes have 0 words"), std::string::npos) << error.what();
        }
    }
    // expansion rows are not the edges of a horizon
    {
        EncodeInput in = base.input();
        rl::Expansion x;
        in.actions = &x;
        EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    }
    // a horizon given to a flat schema
    {
        const auto flat = FlatEncoder::create(table, test::encoder_options());
        EXPECT_THROW((void)flat->encode(base.input()), std::invalid_argument);
    }
    // streams hold independent graphs
    EXPECT_THROW(FlatStream{encoder}, std::invalid_argument);
}

/// The graph used by the kinship tests: seven nodes, node 3 reached from two parents, a parallel edge, a back edge.
///   0 -> 1, 0 -> 2, 0 -> 1 (parallel), 1 -> 3, 2 -> 3, 1 -> 4, 2 -> 5, 3 -> 6, 4 -> 0
test::Horizon kinship_graph(const rl::TaskTable& table)
{
    const test::Rows roots = test::random_walk(table, 0, 1, 2);
    test::Horizon h;
    h.words = roots.words;
    h.label_width = table.label_width();
    h.task_ids = {0};
    for (int node = 0; node < 7; ++node)
        h.nodes.insert(h.nodes.end(), roots.data.begin(), roots.data.end());
    const std::vector<std::array<i32, 2>> edges = {{0, 1}, {0, 2}, {0, 1}, {1, 3}, {2, 3}, {1, 4}, {2, 5}, {3, 6}, {4, 0}};
    for (const auto& edge : edges)
    {
        h.edge_src.push_back(edge[0]);
        h.edge_dst.push_back(edge[1]);
        h.edge_schema.push_back(0);
        h.edge_binding.insert(h.edge_binding.end(), h.label_width, 0);
    }
    h.node_offsets = {0, 7};
    h.edge_offsets = {0, static_cast<i32>(edges.size())};
    return h;
}

TEST(Horizon, KinshipOfADag)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Horizon h = kinship_graph(*table);
    // by hand: siblings share a parent (node 3 has the parents 1 and 2), cousins have parents that are siblings
    const std::vector<std::string> siblings = {"s1,s2", "s2,s1", "s3,s4", "s3,s5", "s4,s3", "s5,s3"};
    const std::vector<std::string> cousins = {"s0,s6", "s3,s4", "s3,s5", "s4,s3", "s4,s5", "s5,s3", "s5,s4", "s6,s0"};
    for (const RootPolicy root : k_roots)
        for (const Layout layout : {Layout::Graph, Layout::Relation})
            for (const IndexType index : {IndexType::I32, IndexType::I64})
            {
                FlatOptions options = horizon_options(HorizonMode::Full, root);
                options.horizon.sibling_relation = true;
                options.horizon.cousin_relation = true;
                options.horizon.parent_relation = true;
                options.include_actions = false;
                options.layout = layout;
                options.index = index;
                const auto encoder = FlatEncoder::create(table, options);
                const auto batch = encoder->encode(h.input());
                EXPECT_NO_THROW(validate(*batch, encoder->schema()));
                const Decoded decoded = decode(*batch, encoder->schema());
                const bool exclude = root == RootPolicy::Exclude;
                auto without_root = [&](std::vector<std::string> pairs)
                {
                    if (exclude)
                        std::erase_if(pairs, [](const std::string& pair) { return pair.find("s0") != std::string::npos; });
                    return pairs;
                };
                EXPECT_EQ(decoded[0].at("_sibling_"), without_root(siblings));
                EXPECT_EQ(decoded[0].at("_cousin_"), without_root(cousins));
                expect_reference(*encoder, h, *batch);
            }
    // the relations are symmetric, and either alone is allowed
    FlatOptions options = horizon_options(HorizonMode::Action, RootPolicy::Include);
    options.horizon.cousin_relation = true;
    const auto encoder = FlatEncoder::create(table, options);
    EXPECT_FALSE(encoder->schema().find("_sibling_"));
    const Decoded decoded = decode(*encoder->encode(h.input()), encoder->schema());
    EXPECT_EQ(decoded[0].at("_cousin_"), cousins);
}

TEST(Horizon, KinshipOfTrees)
{
    // a tree: the children of a node are its siblings, the children of its siblings the cousins of its children
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 2, 5);
    const test::Horizon h = test::horizon_of(*table, roots, 2, 3);
    FlatOptions options = horizon_options(HorizonMode::Delta, RootPolicy::EncodeOnly);
    options.horizon.sibling_relation = true;
    options.horizon.cousin_relation = true;
    const auto encoder = FlatEncoder::create(table, options);
    const Decoded decoded = decode(*encoder->encode(h.input()), encoder->schema());
    bool any_cousins = false;
    for (u64 g = 0; g < h.graphs(); ++g)
    {
        std::map<u32, std::vector<u32>> children;
        for (i32 e = h.edge_offsets[g]; e < h.edge_offsets[g + 1]; ++e)
            children[static_cast<u32>(h.edge_src[static_cast<usize>(e)])].push_back(static_cast<u32>(h.edge_dst[static_cast<usize>(e)]));
        std::set<std::string> siblings, cousins;
        for (const auto& [parent, list] : children)
            for (const u32 a : list)
                for (const u32 b : list)
                    if (a != b)
                        siblings.insert("s" + std::to_string(a) + ",s" + std::to_string(b));
        for (const auto& [p, list_p] : children)
            for (const auto& [q, list_q] : children)
            {
                if (p == q || !std::ranges::any_of(children, [&](const auto& entry) {
                        return std::ranges::count(entry.second, p) && std::ranges::count(entry.second, q);
                    }))
                    continue;
                for (const u32 a : list_p)
                    for (const u32 b : list_q)
                        cousins.insert("s" + std::to_string(a) + ",s" + std::to_string(b));
            }
        EXPECT_EQ(std::set<std::string>(decoded[g].at("_sibling_").begin(), decoded[g].at("_sibling_").end()), siblings);
        EXPECT_EQ(std::set<std::string>(decoded[g].at("_cousin_").begin(), decoded[g].at("_cousin_").end()), cousins);
        any_cousins = any_cousins || !cousins.empty();
    }
    EXPECT_TRUE(any_cousins);
}

TEST(Horizon, CorruptBatchesAreRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 1, 2);
    const test::Horizon h = test::horizon_of(*table, roots, 2, 3);
    const auto encoder = FlatEncoder::create(table, horizon_options(HorizonMode::Full, RootPolicy::EncodeOnly));
    const auto batch = encoder->encode(h.input());
    const EncodingSchema& schema = encoder->schema();
    const auto role = ints(*batch, "entity_role");
    const auto first_state = static_cast<u64>(std::find(role.begin(), role.end(), 7) - role.begin());
    EXPECT_NO_THROW(validate(*batch, schema));
    // a state entity that does not refer to its node, one that is not a state, a target of another group
    EXPECT_THROW(validate(test::with_change(*batch, "entity_ref", first_state, 1), schema), std::invalid_argument);
    EXPECT_THROW(validate(test::with_value(*batch, "entity_role", first_state, 0), schema), std::invalid_argument);
    EXPECT_THROW(validate(test::with_value(*batch, "target_group", 0, 0), schema), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*batch, "target_entity", 0, 1), schema), std::invalid_argument);
}
}  // namespace
