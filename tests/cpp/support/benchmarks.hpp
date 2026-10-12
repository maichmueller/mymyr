#pragma once
// Test support: the PDDL benchmark instances vendored under tests/data (benchmarks/, ipc/, pddl/) and the task sets
// built from them.
//
//   benchmarks/small/<domain>/        small instances of many domains (domain.pddl next to the problems)
//   benchmarks/classic/<kind>/<domain>/   classical and numeric competition instances, kind = strips | adl | numeric
//   benchmarks/domains/<domain>/      further domains with one test problem each
//   ipc/<domain>-ipc/<split>/         sampled learning-track instances
//   pddl/<domain>/                    hand-written and adjusted domains

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace mymyr::test
{
inline std::filesystem::path test_data_dir() { return std::filesystem::path(MYMYR_TEST_DATA_DIR); }
inline std::filesystem::path small_dir() { return test_data_dir() / "benchmarks" / "small"; }
inline std::filesystem::path classic_dir() { return test_data_dir() / "benchmarks" / "classic"; }
inline std::filesystem::path domains_dir() { return test_data_dir() / "benchmarks" / "domains"; }
inline std::filesystem::path ipc_dir() { return test_data_dir() / "ipc"; }

/// A domain file with one of its problem files.
struct PddlTask
{
    std::string name;  // e.g. "rovers__p02" or "cs-tpp"
    std::filesystem::path domain, problem;
};

namespace benchmark_files
{
/// The domain file next to the problem: domain_<stem>.pddl, domain-<stem>.pddl or domain.pddl.
inline bool locate(const std::filesystem::path& dir, const std::string& problem, PddlTask& t)
{
    namespace fs = std::filesystem;
    const std::string stem = fs::path(problem).stem().string();
    t.name = dir.filename().string() + "__" + stem;
    for (const auto& cand : {"domain_" + stem + ".pddl", "domain-" + stem + ".pddl", std::string("domain.pddl")})
        if (fs::exists(dir / cand))
        {
            t.domain = dir / cand;
            break;
        }
    t.problem = dir / problem;
    return !t.domain.empty() && fs::exists(t.problem);
}
}  // namespace benchmark_files

/// The 22-task classical suite (21 competition instances and the philosophers domain of tests/data/pddl). Tasks whose
/// PDDL is missing are left out and reported in `missing`.
inline std::vector<PddlTask> classic_suite(std::vector<std::string>* missing = nullptr)
{
    struct Entry
    {
        const char* tag;
        const char* problem;
        bool local;  // under tests/data/pddl instead of benchmarks/classic
    };
    const Entry entries[] = {
        {"strips/gripper", "prob05.pddl", false},
        {"strips/blocks", "probBLOCKS-8-0.pddl", false},
        {"strips/logistics00", "probLOGISTICS-6-1.pddl", false},
        {"strips/miconic", "s7-4.pddl", false},
        {"strips/visitall", "visitall_x-6_y-3_r-100.pddl", false},
        {"strips/sokoban-opt08-strips", "p14.pddl", false},
        {"strips/depot", "p02.pddl", false},
        {"strips/driverlog", "p03.pddl", false},
        {"strips/rovers", "p02.pddl", false},
        {"strips/zenotravel", "p05.pddl", false},
        {"strips/transport-opt08-strips", "p23.pddl", false},
        {"strips/freecell", "p02.pddl", false},
        {"strips/snake-opt18-strips", "p05.pddl", false},
        {"strips/parcprinter-opt11-strips", "p03.pddl", false},
        {"strips/pegsol-08-strips", "p22.pddl", false},
        {"adl/miconic-simpleadl", "s10-2.pddl", false},
        {"adl/caldera-split-opt18-adl", "p04.pddl", false},
        {"adl/pathways", "p02.pddl", false},
        {"adl/folding-opt23-adl", "p01.pddl", false},
        {"adl/openstacks-opt08-adl", "p03.pddl", false},
        {"philosophers", "p03-phil4.pddl", true},
        {"strips/organic-synthesis-opt18-strips", "p20.pddl", false},
    };
    std::vector<PddlTask> out;
    for (const auto& e : entries)
    {
        PddlTask t;
        if (benchmark_files::locate((e.local ? test_data_dir() / "pddl" : classic_dir()) / e.tag, e.problem, t))
            out.push_back(std::move(t));
        else if (missing)
            missing->push_back(std::string(e.tag) + "/" + e.problem);
    }
    return out;
}

/// The 20 numeric tasks: the first problem of ten numeric domains of benchmarks/classic, the test problem of eight
/// domains of benchmarks/domains, and two variants of tests/data/numeric_tasks/pddl. Missing PDDL is reported in
/// `missing`.
inline std::vector<PddlTask> numeric_suite(std::vector<std::string>* missing = nullptr)
{
    namespace fs = std::filesystem;
    const fs::path csn = classic_dir() / "numeric";
    const fs::path md = domains_dir();
    std::vector<PddlTask> out;
    auto add = [&](std::string name, fs::path domain, fs::path problem)
    {
        if (!fs::exists(domain) || !fs::exists(problem))
        {
            if (missing)
                missing->push_back(name);
            return;
        }
        out.push_back(PddlTask{std::move(name), std::move(domain), std::move(problem)});
    };
    for (const char* d : {"block-grouping", "counters", "delivery", "drone", "expedition", "ext-plant-watering", "farmland",
                          "hydropower", "sailing", "tpp"})
        add(std::string("cs-") + d, csn / d / "domain.pddl", csn / d / "pfile1.pddl");
    for (const char* d : {"fo-counters", "refuel", "refuel-adl", "tpp/numeric", "zenotravel/numeric", "woodworking", "barman", "transport"})
    {
        std::string name = std::string("m-") + d;
        std::replace(name.begin(), name.end(), '/', '-');
        add(name, md / d / "domain.pddl", md / d / "test_problem.pddl");
    }
    const fs::path pddl = test_data_dir() / "numeric_tasks/pddl";
    add("cs-block-grouping-conj", csn / "block-grouping" / "domain.pddl", pddl / "block-grouping-conj.pddl");
    add("cs-delivery-nometric", csn / "delivery" / "domain.pddl", pddl / "delivery-nometric.pddl");
    return out;
}
}  // namespace mymyr::test
