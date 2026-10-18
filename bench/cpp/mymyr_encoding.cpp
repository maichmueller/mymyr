// mymyr_encoding: the relational encoder on states of a seeded random walk, one JSON line:
//
//   mymyr_encoding domain.pddl problem.pddl [N = 1024] [seed = 1] [layout = graph|relation] [threads = 1]
//   [goals = off|on|targets] [actions = off|on|targets|virtual] [singles = 256] [subgoals = 0] [history = 0]
//
// The walk starts at the initial state and restarts there at dead ends. The line reports the best of 5 timed encodes of
// the whole batch (after one untimed): microseconds per state into a new bundle (us_per_state) and into preallocated
// buffers (into_us_per_state, the capacity form; count_us_per_state and fill_us_per_state split it), and the tuples and
// tuple arguments of the batch. faults_per_call is the minor page faults of one timed encode, averaged over the timed
// encodes. goals selects the goal relations: off (none, the default), on (the plain and satisfied relations of the
// instance goal) or targets (plain, satisfied and unsatisfied, with the goal entities as targets).
// single_us_per_call is one encode per state of the first min(N, singles) states (one row per call; null for
// singles = 0). actions selects the action lane: off (no expansion is given, so the action relations are empty, the
// default), on (the expansion of the states is given), targets (and every action tuple's entity is a target) or
// virtual (and the predicates get virtual nodes); the expansion is made before the timing and its time is not part of
// the line. subgoals is the number of subgoal layers per row (0 to 3, needs goals on or targets) and history the steps
// of each row's history (the rows have 1 + i % history steps, so they differ); both give every row random literals
// drawn from the states of the walk, and with goals = targets their entities are targets too.
// horizon = off|full|delta|action turns the rows into root states of horizons (the options in argv[12], argv[13] and
// argv[14]: depth = 1, max_children = 0 for all, root = include|encode_only|exclude): every graph is a root with the nodes
// within depth actions, the line reports the time per graph and per node (the rows of the nodes make the batch), and the
// subgoal and history options do not apply. relations = none|<letters> (argv[16]: p parent, s sibling, c cousin; "psc" is
// the rooted horizon) adds the relations of the edges and the kinship, and source = tree|builder|transitions
// (argv[17]) says where the graphs come from: tree (the default) expands every node on its own and keeps equal states
// apart, builder merges equal states (encoding::build_horizon_graphs, up to depth), transitions encodes the successors
// of the rows (depth 1; expansion and encoding::transition_graphs). The line then also reports build_us_per_node, the
// time to make the graphs per node of them.

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/horizon_graphs.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <random>
#include <sys/resource.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
long minor_faults()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_minflt;
}

/// An FNV-1a hash over the arrays of a batch (the same batch always has the same hash, whatever the machine).
std::string batch_hash(const rl::ArrayBundle& bundle)
{
    u64 hash = 14695981039346656037ull;
    for (const char* name : {"task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref", "relation_counts",
                             "relation_offsets", "relation_arg_offsets", "relation_args", "goal_complete",
                             "target_offsets", "target_entity", "target_group", "history_offsets", "history_entity",
                             "history_dt"})
    {
        const rl::ArrayInfo* info = bundle.find(name);
        if (!info)
            continue;
        const auto* bytes = static_cast<const unsigned char*>(bundle.data(*info));
        const u64 size = info->elements() * rl::dtype_bytes(info->dtype);
        for (u64 i = 0; i < size; ++i)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    char text[24];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/// The action rows of a batch of states, with the arrays that the expansion points at.
struct Expanded
{
    rl::Expansion x;
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
};

std::shared_ptr<Expanded> expand_rows(const rl::TaskTable& table, const u64* rows, u64 count, u32 words)
{
    auto e = std::make_shared<Expanded>();
    const std::vector<i32> ids(count, 0);
    u64 capacity = count * 8 + 16;
    for (;;)
    {
        e->succ.assign(capacity * words, 0);
        e->parent.assign(capacity, 0);
        e->schema.assign(capacity, 0);
        e->binding.assign(capacity * table.label_width(), 0);
        e->offsets.assign(count + 1, 0);
        e->x = {};
        e->x.capacity = capacity;
        e->x.words = words;
        e->x.label_width = table.label_width();
        e->x.succ = e->succ.data();
        e->x.parent = e->parent.data();
        e->x.schema = e->schema.data();
        e->x.binding = e->binding.data();
        e->x.offsets = e->offsets.data();
        rl::expand(table, {rows, count, words, 0, 0}, ids.data(), e->x);
        if (!e->x.overflow())
            return e;
        capacity = std::max(capacity * 2, e->x.total);
        words = std::max(words, e->x.words_needed);
    }
}

/// The nodes and edges of the horizons of `roots` (a root per graph), `depth` levels deep.
struct Horizon
{
    std::vector<u64> nodes;
    std::vector<i32> node_offsets{0}, edge_offsets{0}, edge_src, edge_dst, edge_schema, edge_binding;
    u64 num_nodes(u32 words) const { return nodes.size() / words; }
};

Horizon build_horizon(const rl::TaskTable& table, const std::vector<u64>& roots, u64 count, u32 words, u32 depth,
                      u64 max_children)
{
    Horizon h;
    const u32 width = table.label_width();
    for (u64 g = 0; g < count; ++g)
    {
        const u64 first = h.num_nodes(words);
        h.nodes.insert(h.nodes.end(), roots.begin() + static_cast<std::ptrdiff_t>(g * words),
                       roots.begin() + static_cast<std::ptrdiff_t>((g + 1) * words));
        u64 begin = first, end = first + 1;
        for (u32 level = 0; level < depth; ++level)
        {
            for (u64 node = begin; node < end; ++node)
            {
                const std::vector<u64> state(h.nodes.begin() + static_cast<std::ptrdiff_t>(node * words),
                                             h.nodes.begin() + static_cast<std::ptrdiff_t>((node + 1) * words));
                const auto e = expand_rows(table, state.data(), 1, words);
                const u64 children = max_children ? std::min<u64>(e->x.total, max_children) : e->x.total;
                for (u64 c = 0; c < children; ++c)
                {
                    h.edge_src.push_back(static_cast<i32>(node - first));
                    h.edge_dst.push_back(static_cast<i32>(h.num_nodes(words) - first));
                    h.edge_schema.push_back(e->schema[c]);
                    h.edge_binding.insert(h.edge_binding.end(), e->binding.begin() + static_cast<std::ptrdiff_t>(c * width),
                                          e->binding.begin() + static_cast<std::ptrdiff_t>((c + 1) * width));
                    h.nodes.insert(h.nodes.end(), e->succ.begin() + static_cast<std::ptrdiff_t>(c * words),
                                   e->succ.begin() + static_cast<std::ptrdiff_t>((c + 1) * words));
                }
            }
            begin = end;
            end = h.num_nodes(words);
        }
        h.node_offsets.push_back(static_cast<i32>(h.num_nodes(words)));
        h.edge_offsets.push_back(static_cast<i32>(h.edge_src.size()));
    }
    return h;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: mymyr_encoding domain.pddl problem.pddl [N] [seed]\n");
        return 2;
    }
    try
    {
        const u64 n = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1024;
        const u64 seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
        const TaskPtr task = Task::create(*frontend::load_task(argv[1], argv[2]));
        const auto table = rl::TaskTable::single(task);
        const u32 words = table->words();

        std::vector<u64> rows;
        std::mt19937_64 rng(seed);
        const auto& init = table->instance(0).init;
        std::vector<u64> current(init.begin(), init.end());
        const i32 id = 0;
        std::vector<u64> succ;
        for (u64 i = 0; i < n; ++i)
        {
            rows.insert(rows.end(), current.begin(), current.end());
            rl::Expansion out;
            out.words = words;
            out.capacity = 4096;
            for (;;)
            {
                succ.assign(out.capacity * words, 0);
                out.succ = succ.data();
                rl::expand(*table, {current.data(), 1, words, 0, 0}, &id, out);
                if (!out.overflow())
                    break;
                out.capacity = out.total;
                out.words = std::max(out.words, out.words_needed);
            }
            if (out.total == 0)
                current.assign(init.begin(), init.end());
            else
            {
                const u64 pick = rng() % out.total;
                current.assign(succ.begin() + static_cast<std::ptrdiff_t>(pick * words),
                               succ.begin() + static_cast<std::ptrdiff_t>((pick + 1) * words));
            }
        }

        const std::string layout = argc > 5 ? argv[5] : "graph";
        const u32 threads = argc > 6 ? static_cast<u32>(std::strtoul(argv[6], nullptr, 10)) : 1;
        encoding::FlatOptions options;
        const std::string goals = argc > 7 ? argv[7] : "off";
        if (goals == "off")
            options.goal_derivations = {};
        else if (goals == "targets")
        {
            options.goal_derivations = {encoding::GoalDerivation::Plain, encoding::GoalDerivation::Satisfied,
                                        encoding::GoalDerivation::Unsatisfied};
            options.targets = {encoding::TargetSource::Goal};
        }
        else if (goals != "on")
            throw std::invalid_argument("goals must be off, on or targets");
        const std::string actions = argc > 8 ? argv[8] : "off";
        if (actions == "targets" || actions == "virtual")
            options.targets.insert(encoding::TargetSource::Action);
        else if (actions != "off" && actions != "on")
            throw std::invalid_argument("actions must be off, on, targets or virtual");
        options.predicate_virtual_nodes = actions == "virtual";
        const u32 levels = argc > 10 ? static_cast<u32>(std::strtoul(argv[10], nullptr, 10)) : 0;
        const u32 steps = argc > 11 ? static_cast<u32>(std::strtoul(argv[11], nullptr, 10)) : 0;
        if (levels > 3 || (levels && goals == "off"))
            throw std::invalid_argument("subgoals must be at most 3 and needs goals on or targets");
        options.max_goal_level = levels;
        options.history = steps > 0;
        if (goals == "targets" && levels)
            options.targets.insert(encoding::TargetSource::Subgoal);
        if (goals == "targets" && steps)
            options.targets.insert(encoding::TargetSource::History);
        const auto parsed = encoding::parse_layout(layout);
        if (!parsed)
            throw std::invalid_argument("layout must be graph or relation");
        options.layout = *parsed;
        const std::string horizon = argc > 12 ? argv[12] : "off";
        if (horizon != "off")
        {
            const auto mode = encoding::parse_horizon_mode(horizon);
            if (!mode || *mode == encoding::HorizonMode::None)
                throw std::invalid_argument("horizon must be off, full, delta or action");
            options.horizon.mode = *mode;
            if (*mode == encoding::HorizonMode::Delta && goals != "off")
                options.goal_derivations = {encoding::GoalDerivation::Plain, encoding::GoalDerivation::Satisfied,
                                            encoding::GoalDerivation::AddedSatisfied,
                                            encoding::GoalDerivation::AddedUnsatisfied};
            const auto root = encoding::parse_root_policy(argc > 14 ? argv[14] : "include");
            if (!root)
                throw std::invalid_argument("root must be include, encode_only or exclude");
            options.horizon.root = *root;
            options.targets = {};
            options.max_goal_level = 0;
            options.history = false;
            options.predicate_virtual_nodes = false;
            const u32 depth = argc > 13 ? static_cast<u32>(std::strtoul(argv[13], nullptr, 10)) : 1;
            const u64 max_children = argc > 15 ? std::strtoull(argv[15], nullptr, 10) : 0;
            const std::string relations = argc > 16 ? argv[16] : "none";
            if (relations != "none")
                for (const char letter : relations)
                {
                    if (letter != 'p' && letter != 's' && letter != 'c')
                        throw std::invalid_argument("relations must be none or letters of psc");
                    options.horizon.parent_relation |= letter == 'p';
                    options.horizon.sibling_relation |= letter == 's';
                    options.horizon.cousin_relation |= letter == 'c';
                }
            const std::string source = argc > 17 ? argv[17] : "tree";
            if (source != "tree" && source != "builder" && source != "transitions")
                throw std::invalid_argument("source must be tree, builder or transitions");
            const auto encoder = encoding::FlatEncoder::create(table, options);
            ThreadPool pool(threads);
            rl::ExpandScratch expand_scratch;
            struct Graphs
            {
                Horizon tree;
                encoding::HorizonGraphs built;
                encoding::HorizonInput input;
                u64 nodes = 0, edges = 0;
            };
            const auto make = [&]
            {
                Graphs g;
                if (source == "tree")
                {
                    g.tree = build_horizon(*table, rows, n, words, depth, max_children);
                    g.input = {{g.tree.nodes.data(), g.tree.num_nodes(words), words, 0, 0},
                               n,
                               g.tree.node_offsets.data(),
                               g.tree.edge_offsets.data(),
                               g.tree.edge_src.data(),
                               g.tree.edge_dst.data(),
                               g.tree.edge_schema.data(),
                               g.tree.edge_binding.data(),
                               table->label_width()};
                    g.nodes = g.tree.num_nodes(words);
                    g.edges = g.tree.edge_src.size();
                }
                else
                {
                    const rl::StateBatchView parents{rows.data(), n, words, 0, 0};
                    if (source == "builder")
                        g.built = encoding::build_horizon_graphs(*table, parents, nullptr, encoding::HorizonBuild{depth},
                                                                 pool, expand_scratch);
                    else
                    {
                        const auto e = expand_rows(*table, rows.data(), n, words);
                        g.built = encoding::transition_graphs(*table, parents, nullptr, e->x);
                    }
                    g.input = g.built.input();
                    g.nodes = g.built.num_nodes();
                    g.edges = g.built.num_edges();
                }
                return g;
            };
            Graphs graphs = make();
            double build_best = 1e300;
            for (int rep = 0; rep < 3; ++rep)
            {
                const double t0 = now_s();
                graphs = make();
                build_best = std::min(build_best, now_s() - t0);
            }
            encoding::EncodeInput in;
            in.horizon = graphs.input;
            encoding::EncodeScratch scratch;
            std::shared_ptr<const rl::ArrayBundle> bundle = encoder->encode(in, pool, scratch);
            double best = 1e300;
            const long faults0 = minor_faults();
            for (int rep = 0; rep < 5; ++rep)
            {
                const double t0 = now_s();
                bundle = encoder->encode(in, pool, scratch);
                best = std::min(best, now_s() - t0);
            }
            const long faults_per_call = (minor_faults() - faults0) / 5;
            const u64 nodes = graphs.nodes;
            const u64 cells = n * encoder->schema().num_relations();
            const auto* arg_offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_arg_offsets")));
            std::printf("{\"task\": \"%s\", \"N\": %llu, \"layout\": \"%s\", \"horizon\": \"%s\", \"root\": \"%s\", "
                        "\"depth\": %u, \"nodes\": %llu, \"edges\": %zu, \"threads\": %u, \"us_per_graph\": %.3f, "
                        "\"us_per_node\": %.3f, \"source\": \"%s\", \"relations\": \"%s\", \"build_us_per_node\": %.3f, "
                        "\"faults_per_call\": %ld, \"args\": %d, \"hash\": \"%s\"}\n",
                        std::filesystem::path(argv[2]).stem().string().c_str(), static_cast<unsigned long long>(n),
                        layout.c_str(), horizon.c_str(), std::string(encoding::to_string(options.horizon.root)).c_str(),
                        depth, static_cast<unsigned long long>(nodes), static_cast<size_t>(graphs.edges), threads,
                        best * 1e6 / static_cast<double>(n), best * 1e6 / static_cast<double>(nodes), source.c_str(),
                        relations.c_str(), build_best * 1e6 / static_cast<double>(nodes), faults_per_call,
                        arg_offsets[cells], batch_hash(*bundle).c_str());
            return 0;
        }
        const auto encoder = encoding::FlatEncoder::create(table, options);
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        std::shared_ptr<Expanded> batch_actions;
        if (actions != "off")
        {
            batch_actions = expand_rows(*table, rows.data(), n, words);
            in.actions = &batch_actions->x;
        }
        // Random layers and history steps: literals of other states of the walk
        std::vector<u64> layer_pos(n * levels * words), layer_neg(layer_pos.size()), step_pos(n * steps * words),
            step_neg(step_pos.size());
        std::vector<i32> step_dt(n * steps), step_count(n);
        auto literals = [&](u64 row, u64 other, u64 negative, u64* pos, u64* neg)
        {
            for (u32 w = 0; w < words; ++w)
            {
                pos[w] = rows[row * words + w] & rng();
                neg[w] = rows[((row + other + negative) % n) * words + w] & rng() & ~pos[w];
            }
        };
        for (u64 i = 0; i < n; ++i)
        {
            for (u32 l = 0; l < levels; ++l)
                literals(i, 7 + l, 0, &layer_pos[(i * levels + l) * words], &layer_neg[(i * levels + l) * words]);
            step_count[i] = steps ? static_cast<i32>(1 + i % steps) : 0;
            for (u32 k = 0; k < steps; ++k)
            {
                literals((i + k + 1) % n, 3 * k + 5, 0, &step_pos[(i * steps + k) * words],
                         &step_neg[(i * steps + k) * words]);
                step_dt[i * steps + k] = static_cast<i32>(k) - static_cast<i32>(steps);
            }
        }
        auto subgoal_rows = [&](u64 first)
        {
            encoding::LayerInput x;
            x.pos = layer_pos.data() + first * levels * words;
            x.neg = layer_neg.data() + first * levels * words;
            x.stride = levels * words;
            x.layers = levels;
            x.words = words;
            return x;
        };
        auto history_rows = [&](u64 first)
        {
            encoding::HistoryInput x;
            x.masks.pos = step_pos.data() + first * steps * words;
            x.masks.neg = step_neg.data() + first * steps * words;
            x.masks.stride = steps * words;
            x.masks.layers = steps;
            x.masks.words = words;
            x.dt = step_dt.data() + first * steps;
            x.count = step_count.data() + first;
            return x;
        };
        if (levels)
            in.subgoals = subgoal_rows(0);
        if (steps)
            in.history = history_rows(0);
        double best = 1e300;
        std::shared_ptr<const rl::ArrayBundle> bundle;
        ThreadPool pool(threads);
        encoding::EncodeScratch scratch;
        bundle = encoder->encode(in, pool, scratch);
        const long faults0 = minor_faults();
        for (int rep = 0; rep < 5; ++rep)
        {
            const double t0 = now_s();
            bundle = encoder->encode(in, pool, scratch);
            best = std::min(best, now_s() - t0);
        }
        const long faults_per_call = (minor_faults() - faults0) / 5;
        const u64 singles = std::min<u64>(n, argc > 9 ? std::strtoull(argv[9], nullptr, 10) : 256);
        double best_single = 1e300;
        std::shared_ptr<const rl::ArrayBundle> one_bundle;
        std::vector<std::shared_ptr<Expanded>> single_actions;
        if (actions != "off")
            for (u64 i = 0; i < singles; ++i)
                single_actions.push_back(expand_rows(*table, rows.data() + i * words, 1, words));
        for (int rep = 0; rep < (singles ? 40 : 0); ++rep)
        {
            const double t0 = now_s();
            for (u64 i = 0; i < singles; ++i)
            {
                encoding::EncodeInput one;
                one.states = {rows.data() + i * words, 1, words, 0, 0};
                if (!single_actions.empty())
                    one.actions = &single_actions[i]->x;
                if (levels)
                    one.subgoals = subgoal_rows(i);
                if (steps)
                    one.history = history_rows(i);
                one_bundle = encoder->encode(one, pool, scratch);
            }
            best_single = std::min(best_single, now_s() - t0);
        }
        const u64 R = encoder->schema().num_relations();
        std::vector<i32> task_ids(n), num_objects(n), entity_offsets(n + 1), counts(n * R), tuple_offsets(n * R + 1),
            arg_buffer(n * R + 1), entity_ref(bundle->find("entity_ref")->elements()),
            args(bundle->find("relation_args")->elements());
        std::vector<u8> entity_role(entity_ref.size());
        std::vector<u8> goal_complete(n);
        const auto* targets_info = bundle->find("target_entity");
        std::vector<i32> target_offsets(targets_info ? n + 1 : 0), target_entity(targets_info ? targets_info->elements() : 0);
        std::vector<u8> target_group(target_entity.size());
        const auto* history_info = bundle->find("history_entity");
        std::vector<i32> history_offsets(history_info ? n + 1 : 0),
            history_entity(history_info ? history_info->elements() : 0), history_dt(history_entity.size());
        encoding::FlatBuffers buffers{task_ids.data(),    num_objects.data(),  entity_offsets.data(),
                                      entity_role.data(), entity_ref.data(),   counts.data(),
                                      tuple_offsets.data(), arg_buffer.data(), args.data()};
        if (bundle->find("goal_complete"))
            buffers.goal_complete = goal_complete.data();
        if (targets_info)
        {
            buffers.target_offsets = target_offsets.data();
            buffers.target_entity = target_entity.data();
            buffers.target_group = target_group.data();
        }
        if (history_info)
        {
            buffers.history_offsets = history_offsets.data();
            buffers.history_entity = history_entity.data();
            buffers.history_dt = history_dt.data();
        }
        const encoding::FlatCapacities capacities{entity_ref.size(), args.size() + n * R, args.size(),
                                                  target_entity.size(), history_entity.size()};
        double best_into = 1e300;
        for (int rep = 0; rep < 6; ++rep)
        {
            const double t0 = now_s();
            const encoding::FlatSizes sizes = encoder->encode_into(in, capacities, buffers, pool, scratch);
            if (sizes.overflowed())
                throw std::runtime_error("the buffers are too small");
            if (rep > 0)
                best_into = std::min(best_into, now_s() - t0);
        }
        double best_count = 1e300, best_fill = 1e300;
        for (int rep = 0; rep < 6; ++rep)
        {
            const double t0 = now_s();
            const encoding::FlatSizes sizes = encoder->count(in, pool, scratch);
            const double t1 = now_s();
            encoder->fill(in, sizes, buffers, pool, scratch);
            if (rep > 0)
            {
                best_count = std::min(best_count, t1 - t0);
                best_fill = std::min(best_fill, now_s() - t1);
            }
        }
        const auto* offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_offsets")));
        const auto* arg_offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_arg_offsets")));
        const u64 cells = n * encoder->schema().num_relations();
        // the time of a call on one state is null when no single states were encoded
        char single_us[32] = "null";
        if (singles)
            std::snprintf(single_us, sizeof single_us, "%.3f", best_single * 1e6 / static_cast<double>(singles));
        std::printf("{\"task\": \"%s\", \"N\": %llu, \"layout\": \"%s\", \"goals\": \"%s\", "
                    "\"actions\": \"%s\", \"subgoals\": %u, \"history\": %u, \"threads\": %u, "
                    "\"us_per_state\": %.3f, "
                    "\"into_us_per_state\": %.3f, \"count_us_per_state\": %.3f, \"fill_us_per_state\": %.3f, "
                    "\"single_us_per_call\": %s, \"faults_per_call\": %ld, \"tuples\": %d, \"args\": %d, \"hash\": \"%s\"}\n",
                    std::filesystem::path(argv[2]).stem().string().c_str(), static_cast<unsigned long long>(n),
                    layout.c_str(), goals.c_str(), actions.c_str(), levels, steps, threads,
                    best * 1e6 / static_cast<double>(n),
                    best_into * 1e6 / static_cast<double>(n), best_count * 1e6 / static_cast<double>(n),
                    best_fill * 1e6 / static_cast<double>(n), single_us, faults_per_call, offsets[cells],
                    arg_offsets[cells], batch_hash(*bundle).c_str());
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
