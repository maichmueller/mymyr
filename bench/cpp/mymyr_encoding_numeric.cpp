// mymyr_encoding_numeric: the numeric lane of the relational encoder on the states of a seeded random walk, one JSON line:
//
//   mymyr_encoding_numeric domain.pddl problem.pddl [N = 1024] [seed = 1] [layout = graph|relation] [threads = 1]
//   [dtype = float32|float64] [outputs = relations|simple|both]
//   mymyr_encoding_numeric task.txt - [N] [seed] ...
//
// The walk starts at the initial state and restarts there at dead ends. The line reports the best of 7 timed encodes of
// the whole batch into a new bundle, with the numeric relations (us_per_state) and without them (FlatOptions::
// include_numeric = false: the numeric part of the rows is ignored; us_per_state_without), the ratio of the two
// (numeric_over_plain), the numeric tuples per state, and a hash of the batch with the numeric relations.

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string batch_hash(const rl::ArrayBundle& bundle)
{
    u64 hash = 14695981039346656037ull;
    for (const rl::ArrayInfo& info : bundle.arrays())
    {
        const auto* bytes = static_cast<const unsigned char*>(bundle.data(info));
        const u64 size = info.elements() * rl::dtype_bytes(info.dtype);
        for (u64 i = 0; i < size; ++i)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    char text[24];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

/// The best time of `reps` encodes of the batch, in seconds.
double best_time(const encoding::FlatEncoder& encoder, const encoding::EncodeInput& in, ThreadPool& pool,
                 encoding::EncodeScratch& scratch, int reps)
{
    (void)encoder.encode(in, pool, scratch);
    double best = 1e300;
    for (int rep = 0; rep < reps; ++rep)
    {
        const double t0 = now_s();
        const auto batch = encoder.encode(in, pool, scratch);
        best = std::min(best, now_s() - t0);
    }
    return best;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: mymyr_encoding_numeric domain.pddl problem.pddl [N] [seed] [layout] [threads] "
                             "[dtype] [outputs]\n");
        return 2;
    }
    try
    {
        const std::string first = argv[1];
        const TaskPtr task = first.size() > 4 && first.substr(first.size() - 4) == ".txt"
                                 ? Task::from_text_file(first)
                                 : Task::create(*frontend::load_task(argv[1], argv[2]));
        const auto table = rl::TaskTable::single(task);
        const u64 n = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1024;
        const u64 seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
        const std::string layout = argc > 5 ? argv[5] : "graph";
        const u32 threads = argc > 6 ? static_cast<u32>(std::strtoul(argv[6], nullptr, 10)) : 1;
        const std::string dtype = argc > 7 ? argv[7] : "float32";
        const std::string outputs = argc > 8 ? argv[8] : "relations";
        const u32 words = table->words(), numeric_words = table->numeric_words(), width = table->row_words();

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
            out.numeric_words = numeric_words;
            out.capacity = 4096;
            for (;;)
            {
                succ.assign(out.capacity * width, 0);
                out.succ = succ.data();
                rl::expand(*table, {current.data(), 1, words, 0, numeric_words}, &id, out);
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
                current.assign(succ.begin() + static_cast<std::ptrdiff_t>(pick * width),
                               succ.begin() + static_cast<std::ptrdiff_t>((pick + 1) * width));
            }
        }

        encoding::FlatOptions options;
        options.goal_derivations = {};
        const auto parsed = encoding::parse_layout(layout);
        if (!parsed)
            throw std::invalid_argument("layout must be graph or relation");
        options.layout = *parsed;
        const auto parsed_dtype = encoding::parse_numeric_dtype(dtype);
        if (!parsed_dtype)
            throw std::invalid_argument("dtype must be float32 or float64");
        options.numeric_dtype = *parsed_dtype;
        if (outputs == "simple")
            options.outputs = {encoding::Output::Simple};
        else if (outputs == "both")
            options.outputs = {encoding::Output::Relations, encoding::Output::Simple};
        else if (outputs != "relations")
            throw std::invalid_argument("outputs must be relations, simple or both");
        encoding::FlatOptions plain = options;
        plain.include_numeric = false;

        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, numeric_words};
        const auto with = encoding::FlatEncoder::create(table, options);
        const auto without = encoding::FlatEncoder::create(table, plain);
        ThreadPool pool(threads);
        encoding::EncodeScratch scratch;
        const double t_with = best_time(*with, in, pool, scratch, 7);
        const double numeric_tuples = static_cast<double>(scratch.numeric_tuples()) / static_cast<double>(n);
        const double t_without = best_time(*without, in, pool, scratch, 7);
        const auto batch = with->encode(in, pool, scratch);
        std::printf("{\"states\":%llu,\"layout\":\"%s\",\"threads\":%u,\"dtype\":\"%s\",\"outputs\":\"%s\","
                    "\"numeric_tuples_per_state\":%.3f,\"us_per_state\":%.6f,\"us_per_state_without\":%.6f,"
                    "\"numeric_over_plain\":%.4f,\"hash\":\"%s\"}\n",
                    static_cast<unsigned long long>(n), layout.c_str(), threads, dtype.c_str(), outputs.c_str(),
                    numeric_tuples, t_with * 1e6 / static_cast<double>(n), t_without * 1e6 / static_cast<double>(n),
                    t_with / t_without, batch_hash(*batch).c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
