// IW ladder against the fork: per-pass {arity, expanded, generated, generated_in_tree}, status and plan length equal
// the fork's golden data on the 20 suite tasks whose counts do not depend on the successor order (organic-synthesis
// and pegsol do, on a replay of the fork's exact order); the status names.

#include "../../cpp/support/suite.hpp"
#include "../../cpp/support/text.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/task/task.hpp"

#include "status_names.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
struct GoldenPass
{
    u32 arity;
    u64 expanded, generated, in_tree;
};
struct GoldenRow
{
    const char* task;
    u32 k;
    SearchStatus status;
    u32 plan_length;
    std::vector<GoldenPass> passes;
};

// tests/parity/data/expected/<task>.json "iw" (fork 0.16.3, lifted KPKC).
const std::vector<GoldenRow>& golden_rows()
{
    static const std::vector<GoldenRow> rows = {
        {"gripper__prob05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 26, 364, 25}}},
        {"gripper__prob05", 2, SearchStatus::Exhausted, 0, {{0, 26, 364, 25}, {1, 26, 364, 25}, {2, 470, 4732, 469}}},
        {"blocks__probBLOCKS-8-0", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 43, 200, 42}}},
        {"blocks__probBLOCKS-8-0", 2, SearchStatus::Exhausted, 0, {{0, 5, 22, 4}, {1, 43, 200, 42}, {2, 1526, 7758, 1525}}},
        {"logistics00__probLOGISTICS-6-1", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 10, 114, 9}}},
        {"logistics00__probLOGISTICS-6-1", 2, SearchStatus::Exhausted, 0, {{0, 10, 114, 9}, {1, 10, 114, 9}, {2, 316, 3225, 315}}},
        {"miconic__s7-4", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 21, 289, 20}}},
        {"miconic__s7-4", 2, SearchStatus::Exhausted, 0, {{0, 15, 203, 14}, {1, 21, 289, 20}, {2, 280, 3845, 279}}},
        {"visitall__visitall_x-6_y-3_r-100", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 18, 54, 17}}},
        {"visitall__visitall_x-6_y-3_r-100", 2, SearchStatus::Exhausted, 0, {{0, 4, 11, 3}, {1, 18, 54, 17}, {2, 267, 805, 266}}},
        {"sokoban-opt08-strips__p14", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 39, 99, 38}}},
        {"sokoban-opt08-strips__p14", 2, SearchStatus::Exhausted, 0, {{0, 3, 6, 2}, {1, 39, 99, 38}, {2, 2059, 5135, 2058}}},
        {"depot__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 10, 92, 9}}},
        {"depot__p02", 2, SearchStatus::Exhausted, 0, {{0, 8, 74, 7}, {1, 10, 92, 9}, {2, 241, 2255, 240}}},
        {"driverlog__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 27, 196, 26}}},
        {"driverlog__p03", 2, SearchStatus::Exhausted, 0, {{0, 8, 54, 7}, {1, 27, 196, 26}, {2, 485, 3666, 484}}},
        {"rovers__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 22, 158, 21}}},
        {"rovers__p02", 2, SearchStatus::Exhausted, 0, {{0, 8, 48, 7}, {1, 22, 158, 21}, {2, 225, 1675, 224}}},
        {"zenotravel__p05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 30, 318, 29}}},
        {"zenotravel__p05", 2, SearchStatus::Exhausted, 0, {{0, 11, 122, 10}, {1, 26, 318, 25}, {2, 587, 8003, 586}}},
        {"transport-opt08-strips__p23", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 27, 155, 26}}},
        {"transport-opt08-strips__p23", 2, SearchStatus::Exhausted, 0, {{0, 7, 39, 6}, {1, 27, 155, 26}, {2, 1137, 6981, 1136}}},
        {"freecell__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 43, 408, 42}}},
        {"freecell__p02", 2, SearchStatus::Solved, 16, {{0, 9, 79, 8}, {1, 43, 408, 42}, {2, 867, 7063, 869}}},
        {"snake-opt18-strips__p05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 86, 196, 85}}},
        {"snake-opt18-strips__p05", 2, SearchStatus::Exhausted, 0, {{0, 2, 3, 1}, {1, 86, 196, 85}, {2, 2912, 5744, 2911}}},
        {"parcprinter-opt11-strips__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 41, 148, 40}}},
        {"parcprinter-opt11-strips__p03", 2, SearchStatus::Exhausted, 0, {{0, 2, 5, 1}, {1, 41, 148, 40}, {2, 663, 2269, 662}}},
        {"miconic-simpleadl__s10-2", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 29, 580, 28}}},
        {"miconic-simpleadl__s10-2", 2, SearchStatus::Exhausted, 0, {{0, 21, 420, 20}, {1, 29, 580, 28}, {2, 514, 10280, 513}}},
        {"caldera-split-opt18-adl__p04", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 14, 79, 13}}},
        {"caldera-split-opt18-adl__p04", 2, SearchStatus::Exhausted, 0, {{0, 5, 19, 4}, {1, 14, 79, 13}, {2, 69, 330, 68}}},
        {"pathways__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 31, 383, 30}}},
        {"pathways__p02", 2, SearchStatus::Solved, 12, {{0, 13, 156, 12}, {1, 31, 383, 30}, {2, 1096, 13799, 1097}}},
        {"folding-opt23-adl__p01", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 92, 105, 91}}},
        {"folding-opt23-adl__p01", 2, SearchStatus::Exhausted, 0, {{0, 15, 28, 14}, {1, 92, 105, 91}, {2, 127, 140, 126}}},
        {"openstacks-opt08-adl__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 25, 109, 24}}},
        {"openstacks-opt08-adl__p03", 2, SearchStatus::Exhausted, 0, {{0, 2, 9, 1}, {1, 25, 109, 24}, {2, 276, 1285, 275}}},
        {"philosophers__p03-phil4", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 25, 76, 24}}},
        {"philosophers__p03-phil4", 2, SearchStatus::Exhausted, 0, {{0, 5, 20, 4}, {1, 25, 76, 24}, {2, 737, 2016, 736}}},
    };
    return rows;
}

class IwGolden : public ::testing::TestWithParam<GoldenRow>
{
};

TEST_P(IwGolden, PerPassEqualsTheFork)
{
    const GoldenRow& g = GetParam();
    for (auto atoms : {TaskOptions::Atoms::Auto, TaskOptions::Atoms::Lazy})
    {
        TaskOptions to;
        to.atoms = atoms;
        const auto task = Task::from_text_file(task_path(g.task), to);
        IwOptions o;
        o.max_arity = g.k;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, g.status);
        EXPECT_EQ(r.plan.size(), g.plan_length);
        ASSERT_EQ(r.passes.size(), g.passes.size());
        for (usize i = 0; i < g.passes.size(); ++i)
        {
            EXPECT_EQ(r.passes[i].arity, g.passes[i].arity) << "pass " << i;
            EXPECT_EQ(r.passes[i].expanded, g.passes[i].expanded) << "pass " << i;
            EXPECT_EQ(r.passes[i].generated, g.passes[i].generated) << "pass " << i;
            EXPECT_EQ(r.passes[i].generated_in_tree, g.passes[i].in_tree) << "pass " << i;
        }
        if (g.k == 1)
        {
            EXPECT_TRUE(r.passes[0].placeholder);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Fork, IwGolden, ::testing::ValuesIn(golden_rows()),
                         [](const auto& i) { return param_name(std::string(i.param.task) + "_k" + std::to_string(i.param.k)); });
}  // namespace

TEST(IwStatusNames, TheReferenceToolsNames)
{
    EXPECT_STREQ(parity::reference_status_name(SearchStatus::Unsolvable), "unsolvable");
    EXPECT_STREQ(parity::reference_status_name(SearchStatus::Exhausted), "failed");
}
