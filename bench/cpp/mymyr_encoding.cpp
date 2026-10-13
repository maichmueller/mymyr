// mymyr_encoding: the relational encoder on states of a seeded random walk, one JSON line:
//
//   mymyr_encoding domain.pddl problem.pddl [N = 1024] [seed = 1] [layout = graph|relation] [threads = 1]
//
// The walk starts at the initial state and restarts there at dead ends. The line reports the best of 5 timed encodes of
// the whole batch (after one untimed): microseconds per state into a new bundle (us_per_state) and into preallocated
// buffers (into_us_per_state, the capacity form; count_us_per_state and fill_us_per_state split it), and the tuples and
// tuple arguments of the batch.

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
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
        options.goal_derivations = {};
        const auto parsed = encoding::parse_layout(layout);
        if (!parsed)
            throw std::invalid_argument("layout must be graph or relation");
        options.layout = *parsed;
        const auto encoder = encoding::FlatEncoder::create(table, options);
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        double best = 1e300;
        std::shared_ptr<const rl::ArrayBundle> bundle;
        ThreadPool pool(threads);
        encoding::EncodeScratch scratch;
        bundle = encoder->encode(in, pool, scratch);
        for (int rep = 0; rep < 5; ++rep)
        {
            const double t0 = now_s();
            bundle = encoder->encode(in, pool, scratch);
            best = std::min(best, now_s() - t0);
        }
        const u64 R = encoder->schema().num_relations();
        std::vector<i32> task_ids(n), num_objects(n), entity_offsets(n + 1), counts(n * R), tuple_offsets(n * R + 1),
            arg_buffer(n * R + 1), entity_ref(bundle->find("entity_ref")->elements()),
            args(bundle->find("relation_args")->elements());
        std::vector<u8> entity_role(entity_ref.size());
        encoding::FlatBuffers buffers{task_ids.data(),    num_objects.data(),  entity_offsets.data(),
                                      entity_role.data(), entity_ref.data(),   counts.data(),
                                      tuple_offsets.data(), arg_buffer.data(), args.data()};
        const encoding::FlatCapacities capacities{entity_ref.size(), args.size() + n * R, args.size()};
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
        std::printf("{\"task\": \"%s\", \"N\": %llu, \"layout\": \"%s\", \"threads\": %u, \"us_per_state\": %.3f, "
                    "\"into_us_per_state\": %.3f, \"count_us_per_state\": %.3f, \"fill_us_per_state\": %.3f, "
                    "\"tuples\": %d, \"args\": %d}\n",
                    std::filesystem::path(argv[2]).stem().string().c_str(), static_cast<unsigned long long>(n),
                    layout.c_str(), threads, best * 1e6 / static_cast<double>(n),
                    best_into * 1e6 / static_cast<double>(n), best_count * 1e6 / static_cast<double>(n),
                    best_fill * 1e6 / static_cast<double>(n), offsets[cells], arg_offsets[cells]);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
