// mymyr_encoding: the relational encoder on states of a seeded random walk, single thread, one JSON line:
//
//   mymyr_encoding domain.pddl problem.pddl [N = 1024] [seed = 1]
//
// The walk starts at the initial state and restarts there at dead ends. The line reports the best of 5 timed encodes of
// the whole batch: microseconds per state, and the tuples and tuple arguments of the batch.

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

        encoding::FlatOptions options;
        options.goal_derivations = {};
        const auto encoder = encoding::FlatEncoder::create(table, options);
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        double best = 1e300;
        std::shared_ptr<const rl::ArrayBundle> bundle;
        for (int rep = 0; rep < 5; ++rep)
        {
            const double t0 = now_s();
            bundle = encoder->encode(in);
            best = std::min(best, now_s() - t0);
        }
        const auto* offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_offsets")));
        const auto* arg_offsets = static_cast<const i32*>(bundle->data(*bundle->find("relation_arg_offsets")));
        const u64 cells = n * encoder->schema().num_relations();
        std::printf("{\"task\": \"%s\", \"N\": %llu, \"us_per_state\": %.3f, \"tuples\": %d, \"args\": %d}\n",
                    std::filesystem::path(argv[2]).stem().string().c_str(), static_cast<unsigned long long>(n),
                    best * 1e6 / static_cast<double>(n), offsets[cells], arg_offsets[cells]);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
