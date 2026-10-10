#pragma once
// Instance sets for the task-table tests: instances of one domain spanning at least 3x in object count and at least
// two width buckets (WB in {2, 4, 16, 64}), from the fork's data dir through the front end. Where the fork's own
// instances of a domain are all small (gripper, logistics, miconic-simpleadl), the larger problems are generated for
// the fork's domain file (tests/data/table_instances/gen.py). Shared by the host tests (tests/cpp/rl/test_table.cpp),
// the CUDA tests (tests/cuda/test_device_table.cpp) and the benches (bench/cuda/table_*.cpp).
//
// Needs frontend/golden.hpp (fork_data_dir()) and MYMYR_TEST_DATA_DIR.

#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace mymyr::test
{
std::filesystem::path fork_data_dir();

struct InstanceSet
{
    std::string name;
    std::filesystem::path domain;
    std::vector<std::filesystem::path> problems;
};

inline std::vector<InstanceSet> table_instance_sets()
{
    const std::filesystem::path d = fork_data_dir();
    const std::filesystem::path m = std::filesystem::path(MYMYR_TEST_DATA_DIR) / "table_instances";
    const std::filesystem::path bw = d / "ipc" / "blocksworld-ipc" / "train";
    const std::filesystem::path mi = d / "ipc" / "miconic-ipc" / "test";
    return {
        {"blocks", bw / "domain.pddl", {bw / "p13.pddl", bw / "p25.pddl", bw / "p37.pddl", bw / "p49.pddl", bw / "p61.pddl"}},
        {"gripper",
         d / "gripper" / "domain.pddl",
         {d / "gripper" / "test_problem4.pddl", m / "gripper" / "gripper-10.pddl", m / "gripper" / "gripper-20.pddl",
          m / "gripper" / "gripper-40.pddl", m / "gripper" / "gripper-80.pddl"}},
        {"miconic",
         mi / "domain.pddl",
         {mi / "p01-easy.pddl", mi / "p20-easy.pddl", mi / "p01-medium.pddl", mi / "p04-medium.pddl", mi / "p16-medium.pddl"}},
        {"logistics",
         d / "logistics" / "domain.pddl",
         {d / "logistics" / "test_problem.pddl", m / "logistics" / "logistics-c2-s3-p4-a1.pddl",
          m / "logistics" / "logistics-c3-s3-p8-a2.pddl", m / "logistics" / "logistics-c4-s4-p16-a3.pddl",
          m / "logistics" / "logistics-c6-s5-p30-a4.pddl"}},
        // conditional effects; the quantified goals (-q) become goal axioms (a derived predicate of the problem)
        {"miconic-simpleadl",
         d / "miconic-simpleadl" / "domain.pddl",
         {d / "miconic-simpleadl" / "test_problem.pddl", m / "miconic-simpleadl" / "simple-f6-p4-q.pddl",
          m / "miconic-simpleadl" / "simple-f12-p8-c.pddl", m / "miconic-simpleadl" / "simple-f24-p16-q.pddl",
          m / "miconic-simpleadl" / "simple-f40-p50-q.pddl", m / "miconic-simpleadl" / "simple-f60-p100-c.pddl"}},
    };
}

/// The tasks of a set (empty if a file is missing).
inline std::vector<TaskPtr> load_set(const InstanceSet& s, TaskOptions::Atoms atoms)
{
    for (const auto& p : s.problems)
        if (!std::filesystem::exists(p) || !std::filesystem::exists(s.domain))
            return {};
    TaskOptions o;
    o.atoms = atoms;
    const auto dom = frontend::Domain::from_file(s.domain);
    std::vector<TaskPtr> out;
    for (const auto& p : s.problems)
        out.push_back(Task::create(*dom->instantiate_file(p), o));
    return out;
}

/// Rows of `per` states of each instance (a deterministic walk from the initial state: successor (7k + 3) mod count at
/// step k), interleaved by instance, at the table's width; ids[r] is row r's instance.
inline void table_rows(const rl::TaskTable& table, u32 per, std::vector<u64>& rows, std::vector<i32>& ids)
{
    const u32 W = table.words(), NN = table.numeric_words(), RW = W + NN, I = table.size();
    std::vector<std::vector<std::vector<u64>>> states(I);
    for (u32 i = 0; i < I; ++i)
    {
        const Task& task = *table.task(i);
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
        std::vector<u64> cur = table.instance(i).init;
        std::vector<u64> tmp;
        for (u32 k = 0; k < per; ++k)
        {
            states[i].push_back(cur);
            succ.prepare(StateView{cur.data(), W, NN ? cur.data() + W : nullptr, task.numeric_words()});
            std::vector<std::vector<u64>> kids;
            succ.generate<false>(
                [&](u32, const ObjectId*, const Delta& d) -> bool
                {
                    const u32 n = apply_delta(cur.data(), W, d, tmp);
                    std::vector<u64> row(RW, 0);
                    std::copy_n(tmp.begin(), std::min(n, W), row.begin());
                    for (u32 q = 0; q < d.nnum; ++q)
                        row[W + q] = d.num[q];
                    kids.push_back(std::move(row));
                    return true;
                },
                false, true);
            if (kids.empty())
                cur = table.instance(i).init;
            else
                cur = kids[(7 * k + 3) % kids.size()];
        }
    }
    rows.clear();
    ids.clear();
    for (u32 k = 0; k < per; ++k)
        for (u32 j = 0; j < I; ++j)
        {
            const u32 i = (j + k) % I;  // a different instance order in every round
            rows.insert(rows.end(), states[i][k].begin(), states[i][k].end());
            ids.push_back(static_cast<i32>(i));
        }
}
}  // namespace mymyr::test
