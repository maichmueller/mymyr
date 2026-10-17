#pragma once
// Test support of the relational encoder: PDDL tasks of tests/data/pddl, seeded random walks over a task table, and
// options without goal relations.

#include "../support/dirty_blocks.hpp"

#include "mymyr/encoding/options.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_table.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace mymyr::test
{
inline TaskPtr pddl_task(const std::string& domain, const std::string& problem)
{
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/" + domain + "/";
    return Task::create(*frontend::load_task(dir + "domain.pddl", dir + problem + ".pddl"));
}

/// The options of the encoder: no goal relations.
inline encoding::FlatOptions encoder_options()
{
    encoding::FlatOptions options;
    options.goal_derivations = {};
    return options;
}

/// Rows of a batch over a table.
struct Rows
{
    std::vector<u64> data;     // [n, row_words]
    std::vector<i32> task_ids;  // [n]
    u32 words = 0;              // atom words per row
    [[nodiscard]] u64 size() const noexcept { return task_ids.size(); }
    [[nodiscard]] rl::StateBatchView view() const noexcept { return {data.data(), size(), words, 0, 0}; }
};

/// `n` rows of one instance: the initial state, then a seeded random walk that restarts at dead ends.
inline Rows random_walk(const rl::TaskTable& table, u32 instance, u64 n, u64 seed)
{
    Rows rows;
    rows.words = table.words();
    std::mt19937_64 rng(seed);
    const auto& init = table.instance(instance).init;
    std::vector<u64> current(init.begin(), init.end());
    const i32 id = static_cast<i32>(instance);
    for (u64 i = 0; i < n; ++i)
    {
        rows.data.insert(rows.data.end(), current.begin(), current.end());
        rows.task_ids.push_back(id);
        rl::Expansion out;
        out.words = rows.words;
        out.capacity = 4096;
        std::vector<u64> succ;
        for (;;)
        {
            succ.assign(out.capacity * rows.words, 0);
            out.succ = succ.data();
            rl::expand(table, {current.data(), 1, rows.words, 0, 0}, &id, out);
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
            current.assign(succ.begin() + static_cast<std::ptrdiff_t>(pick * rows.words),
                           succ.begin() + static_cast<std::ptrdiff_t>((pick + 1) * rows.words));
        }
    }
    return rows;
}
}  // namespace mymyr::test
