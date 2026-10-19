// mymyr_encoding: the relational encoder on states of a seeded random walk, one JSON line:
//
//   mymyr_encoding domain.pddl problem.pddl [N = 1024] [seed = 1] [layout = graph|relation] [threads = 1]
//   [goals = off|on|targets] [actions = off|on|targets|virtual] [singles = 256] [statics = repeat|shared]
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
// the line. statics selects the form of the static tuples (repeated in every graph or shared once per batch). bytes is
// the size of the batch's bundle and digest a hash of its arrays and scalars (equal digests: equal batches).

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
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
        const std::string statics = argc > 10 ? argv[10] : "repeat";
        const auto parsed_statics = encoding::parse_statics(statics);
        if (!parsed_statics)
            throw std::invalid_argument("statics must be repeat or shared");
        options.statics = *parsed_statics;
        const auto parsed = encoding::parse_layout(layout);
        if (!parsed)
            throw std::invalid_argument("layout must be graph or relation");
        options.layout = *parsed;
        const auto encoder = encoding::FlatEncoder::create(table, options);
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        std::shared_ptr<Expanded> batch_actions;
        if (actions != "off")
        {
            batch_actions = expand_rows(*table, rows.data(), n, words);
            in.actions = &batch_actions->x;
        }
        double best = 1e300;
        std::shared_ptr<const rl::ArrayBundle> bundle;
        ThreadPool pool(threads);
        encoding::EncodeScratch scratch;
        // two untimed calls: the first takes a fresh block and the second a second one (the first is still held), so
        // the timed calls below recycle the two
        bundle = encoder->encode(in, pool, scratch);
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
        const encoding::BatchShape capacities{n, entity_ref.size(), args.size(),
                                                  target_entity.size()};
        const bool shared = options.statics == encoding::Statics::Shared;
        std::vector<i32> static_task_ids, static_offsets, static_arg_offsets, static_args, static_relations;
        encoding::StaticShape static_capacities;
        encoding::StaticSizes static_sizes;
        if (shared)
        {
            static_task_ids.resize(bundle->find("static_task_ids")->elements());
            static_offsets.resize(bundle->find("static_offsets")->elements());
            static_arg_offsets.resize(bundle->find("static_arg_offsets")->elements());
            static_args.resize(bundle->find("static_args")->elements());
            static_relations.resize(bundle->find("static_relations")->elements());
            buffers.static_task_ids = static_task_ids.data();
            buffers.static_offsets = static_offsets.data();
            buffers.static_arg_offsets = static_arg_offsets.data();
            buffers.static_args = static_args.data();
            buffers.static_relations = static_relations.data();
            static_capacities = {static_task_ids.size(), static_args.size()};
        }
        const encoding::StaticShape* static_caps = shared ? &static_capacities : nullptr;
        double best_into = 1e300;
        for (int rep = 0; rep < 6; ++rep)
        {
            const double t0 = now_s();
            const encoding::FlatSizes sizes = encoder->encode_into(in, capacities, buffers, pool, scratch, static_caps,
                                                                   shared ? &static_sizes : nullptr);
            if (sizes.overflowed())
                throw std::runtime_error("the buffers are too small");
            if (rep > 0)
                best_into = std::min(best_into, now_s() - t0);
        }
        double best_count = 1e300, best_fill = 1e300;
        for (int rep = 0; rep < 6; ++rep)
        {
            const double t0 = now_s();
            const encoding::FlatSizes sizes = encoder->count(in, pool, scratch, shared ? &static_sizes : nullptr);
            const double t1 = now_s();
            encoder->fill(in, sizes, buffers, pool, scratch);
            if (rep > 0)
            {
                best_count = std::min(best_count, t1 - t0);
                best_fill = std::min(best_fill, now_s() - t1);
            }
        }
        std::uint64_t digest = 14695981039346656037ull;
        const auto mix = [&digest](const void* data, std::size_t size)
        {
            for (std::size_t i = 0; i < size; ++i)
                digest = (digest ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
        };
        for (const auto& array : bundle->arrays())
            mix(bundle->data(array), array.elements() * rl::dtype_bytes(array.dtype));
        for (const auto& [name, value] : bundle->scalars())
            mix(&value, sizeof value);
        const auto* offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_offsets")));
        const auto* arg_offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_arg_offsets")));
        const u64 cells = n * encoder->schema().num_relations();
        // the time of a call on one state is null when no single states were encoded
        char single_us[32] = "null";
        if (singles)
            std::snprintf(single_us, sizeof single_us, "%.3f", best_single * 1e6 / static_cast<double>(singles));
        std::printf("{\"task\": \"%s\", \"N\": %llu, \"layout\": \"%s\", \"goals\": \"%s\", "
                    "\"actions\": \"%s\", \"threads\": %u, \"us_per_state\": %.3f, "
                    "\"into_us_per_state\": %.3f, \"count_us_per_state\": %.3f, \"fill_us_per_state\": %.3f, "
                    "\"single_us_per_call\": %s, \"faults_per_call\": %ld, \"tuples\": %d, \"args\": %d, "
                    "\"statics\": \"%s\", \"bytes\": %llu, \"digest\": \"%016llx\"}\n",
                    std::filesystem::path(argv[2]).stem().string().c_str(), static_cast<unsigned long long>(n),
                    layout.c_str(), goals.c_str(), actions.c_str(), threads, best * 1e6 / static_cast<double>(n),
                    best_into * 1e6 / static_cast<double>(n), best_count * 1e6 / static_cast<double>(n),
                    best_fill * 1e6 / static_cast<double>(n), single_us, faults_per_call, offsets[cells],
                    arg_offsets[cells], statics.c_str(), static_cast<unsigned long long>(bundle->bytes()),
                    static_cast<unsigned long long>(digest));
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
