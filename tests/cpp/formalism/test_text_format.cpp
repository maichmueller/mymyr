// Round-trip of the task text format: read -> TaskData -> write must reproduce the file token by token, for every
// task of tests/data/tasks and tests/data/numeric_tasks (plain lifted format and numeric format).

#include "mymyr/formalism/text_format.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <clocale>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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

TaskData numeric_task(f64 value)
{
    TaskData t;
    for (const auto kind : {FuncKind::Static, FuncKind::Fluent, FuncKind::Auxiliary})
    {
        Function f;
        f.name = t.intern_string("value" + std::to_string(t.functions.size()));
        f.kind = kind;
        t.functions.push_back(f);
    }
    t.static_values.push_back({FunctionId{0}, {}, value});
    t.fluent_values.push_back({FunctionId{1}, {}, value});
    t.auxiliary_initial = value;
    Expr e;
    e.value = value;
    t.exprs.push_back(e);
    t.metric = Metric{true, 0};
    return t;
}

void expect_numeric_values(const TaskData& t, f64 value)
{
    ASSERT_EQ(t.static_values.size(), 1u);
    ASSERT_EQ(t.fluent_values.size(), 1u);
    ASSERT_TRUE(t.auxiliary_initial.has_value());
    ASSERT_TRUE(t.metric.has_value());
    for (const f64 actual : {t.static_values[0].value, t.fluent_values[0].value, *t.auxiliary_initial,
                             t.exprs.at(t.metric->expr).value})
        EXPECT_EQ(std::bit_cast<u64>(actual), std::bit_cast<u64>(value));
}

TaskData read_number(std::string_view number, bool expression)
{
    std::istringstream in("O 0\nP 0\nN 1\nF 0 value\nSI 0\nFI 0\nNI 1\n0 " +
                          (expression ? std::string("0") : std::string(number)) + "\nG 0\nNG 0\nM min 1 c " +
                          (expression ? std::string(number) : std::string("0")) + "\nA 0\nX 0\n");
    return read_task_text(in);
}

struct NumericLocale
{
    std::string saved = std::setlocale(LC_NUMERIC, nullptr);
    ~NumericLocale() { std::setlocale(LC_NUMERIC, saved.c_str()); }
};
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

TEST(TextFormat, NumericValuesRoundTripBitIdentically)
{
    auto check = [](f64 value)
    {
        SCOPED_TRACE(std::bit_cast<u64>(value));
        std::istringstream in(write_task_text(numeric_task(value)));
        expect_numeric_values(read_task_text(in), value);
    };
    for (u64 exponent = 0; exponent < 2047; ++exponent)
        for (u64 mantissa : {u64{0}, u64{1}, u64{1} << 51, (u64{1} << 52) - 1})
            for (u64 sign : {u64{0}, u64{1} << 63})
                check(std::bit_cast<f64>(sign | exponent << 52 | mantissa));
    std::mt19937_64 rng(43);
    for (u32 i = 0; i < 20000; ++i)
    {
        const f64 value = std::bit_cast<f64>(rng());
        if (std::isfinite(value))
            check(value);
    }
    for (const f64 value : {std::numeric_limits<f64>::infinity(), -std::numeric_limits<f64>::infinity(),
                            std::numeric_limits<f64>::quiet_NaN(), -std::numeric_limits<f64>::quiet_NaN()})
        check(value);
}

TEST(TextFormat, WritesShortestNumbers)
{
    for (const f64 value : {0.1, -0.0, std::numeric_limits<f64>::denorm_min(), std::numeric_limits<f64>::max(),
                            std::numeric_limits<f64>::infinity(), -std::numeric_limits<f64>::infinity(),
                            std::numeric_limits<f64>::quiet_NaN(), -std::numeric_limits<f64>::quiet_NaN()})
    {
        std::array<char, 64> buffer{};
        const auto r = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        ASSERT_EQ(r.ec, std::errc{});
        const std::string number = std::isnan(value) ? (std::signbit(value) ? "-nan" : "nan")
                                                   : std::string(buffer.data(), r.ptr);
        if (std::isinf(value))
        {
            EXPECT_EQ(number, std::signbit(value) ? "-inf" : "inf");
        }
        const std::string text = write_task_text(numeric_task(value));
        EXPECT_NE(text.find("NI 3\n0 " + number + "\n1 " + number + "\n2 " + number + "\n"), std::string::npos);
        EXPECT_NE(text.find("M min 1 c " + number + "\n"), std::string::npos);
    }
}

TEST(TextFormat, NumbersIgnoreNumericLocale)
{
    const NumericLocale restore;
    if (!std::setlocale(LC_NUMERIC, "de_DE.UTF-8"))
        GTEST_SKIP() << "de_DE.UTF-8 is unavailable";
    ASSERT_STREQ(std::localeconv()->decimal_point, ",");
    for (const bool expression : {false, true})
    {
        const auto t = read_number("1.5", expression);
        EXPECT_EQ(expression ? t.exprs.at(t.metric->expr).value : t.fluent_values.at(0).value, 1.5);
        EXPECT_THROW(read_number("1,5", expression), std::runtime_error);
    }
    for (const f64 value : {1.5, 0.0125, -0.0, std::numeric_limits<f64>::denorm_min(),
                            std::numeric_limits<f64>::max(), std::numeric_limits<f64>::infinity(),
                            -std::numeric_limits<f64>::quiet_NaN()})
    {
        const std::string text = write_task_text(numeric_task(value));
        EXPECT_EQ(text.find(','), std::string::npos);
        std::istringstream in(text);
        expect_numeric_values(read_task_text(in), value);
    }
}

TEST(TextFormat, RejectsMalformedNumbers)
{
    for (const std::string_view bad : {"+1", "+0", "-", ".", "1e", "1e+", "1e-", "--1", "1_000", "1,5", "1junk",
                                      "0x1", "0x1p0", "-0x1.8p+2", "0x.8p0", "1p0", "+inf", "INF", "infinity",
                                      "-infinity", "INFINITY", "infinite", "+nan", "NaN", "nanosecond", "nan()",
                                      "nan(payload)", "-nan(123)", "nan(snan)", "-nan(ind)", "1e309", "-1e309", "1e-1000",
                                      "-1e-1000", "2.4703282292062327e-324"})
        for (const bool expression : {false, true})
        {
            SCOPED_TRACE(bad);
            SCOPED_TRACE(expression);
            try
            {
                read_number(bad, expression);
                FAIL() << "expected a number error";
            }
            catch (const std::runtime_error& e)
            {
                EXPECT_EQ(e.what(), "task text: expected number, got '" + std::string(bad) + "'");
            }
        }
}
