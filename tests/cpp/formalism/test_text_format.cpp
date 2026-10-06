// Round-trip of the fork-exported task text format: read -> TaskData -> write must reproduce the fork's export
// token by token, for every exported task in the analysis probes (plain lifted format and numeric format).

#include "mymyr/formalism/text_format.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::formalism;
namespace fs = std::filesystem;

namespace
{
std::vector<std::string> tokens(std::istream& in)
{
    std::vector<std::string> out;
    std::string w;
    while (in >> w)
        out.push_back(w);
    return out;
}

std::vector<fs::path> task_files(const fs::path& dir)
{
    std::vector<fs::path> out;
    if (fs::exists(dir))
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".txt")
                out.push_back(e.path());
    std::sort(out.begin(), out.end());
    return out;
}

void round_trip(const fs::path& p)
{
    SCOPED_TRACE(p.string());
    const TaskData t = read_task_text_file(p.string());
    std::ifstream orig(p);
    std::istringstream mine(write_task_text(t));
    const auto a = tokens(orig);
    const auto b = tokens(mine);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
        {
            // numbers may be printed differently but must parse to the same value
            char* e1 = nullptr;
            char* e2 = nullptr;
            const double x = std::strtod(a[i].c_str(), &e1), y = std::strtod(b[i].c_str(), &e2);
            ASSERT_TRUE(*e1 == 0 && *e2 == 0 && x == y) << "token " << i << ": '" << a[i] << "' vs '" << b[i] << "'";
        }
}
}  // namespace

TEST(TextFormat, RoundTripsLiftedSuiteExports)
{
    const auto files = task_files(fs::path(MYMYR_TEST_DATA_DIR) / "tasks");
    ASSERT_GE(files.size(), 20u);
    for (const auto& f : files)
        round_trip(f);
}

TEST(TextFormat, RoundTripsNumericExports)
{
    const auto files = task_files(fs::path(MYMYR_TEST_DATA_DIR) / "numeric_tasks");
    ASSERT_GE(files.size(), 10u);
    for (const auto& f : files)
        round_trip(f);
}

TEST(TextFormat, ParsedStructureIsConsistent)
{
    const auto t = read_task_text_file(
        (fs::path(MYMYR_TEST_DATA_DIR) / "tasks/openstacks-opt08-adl__p03.txt").string());
    EXPECT_EQ(t.num_objects(), 22u);
    EXPECT_EQ(t.axioms.size(), 2u);
    u32 derived = 0;
    for (const auto& p : t.predicates)
        derived += p.kind == PredKind::Derived;
    EXPECT_EQ(derived, 2u);
    for (const auto& l : t.literals_of(t.goal))
        for (Term x : t.terms_of(l))
            EXPECT_TRUE(is_object(x));
}

TEST(TextFormat, RejectsMalformedInput)
{
    std::istringstream bad("O 2\nP 1\nF 1 at\nSI 0\nFI 1\n0 5\nG 0\nA 0\n");  // object 5 out of range
    EXPECT_THROW(read_task_text(bad), std::exception);
    std::istringstream truncated("O 2\nP 1\n");
    EXPECT_THROW(read_task_text(truncated), std::runtime_error);
}
