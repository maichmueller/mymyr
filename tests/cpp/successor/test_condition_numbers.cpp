#include "mymyr/successor/conditions.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <charconv>
#include <clocale>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

using namespace mymyr;

namespace
{
TaskPtr number_task()
{
    return Task::from_text_file((std::filesystem::path(MYMYR_TEST_DATA_DIR) / "tasks/gripper__prob05.txt").string());
}

f64 parse_number(const Task& task, std::string_view text)
{
    GroundCondition c;
    c.add_constraint(task, "(= " + std::string(text) + " 0)");
    return c.exprs.at(c.constraints.at(0).lhs).value;
}

void expect_number(const Task& task, std::string_view text, f64 expected)
{
    const f64 actual = parse_number(task, text);
    EXPECT_EQ(std::bit_cast<u64>(actual), std::bit_cast<u64>(expected)) << text;
}

struct NumericEnvironment
{
    int rounding = std::fegetround();
    std::string locale = std::setlocale(LC_NUMERIC, nullptr);
    ~NumericEnvironment()
    {
        std::fesetround(rounding);
        std::setlocale(LC_NUMERIC, locale.c_str());
    }
};
}  // namespace

TEST(Conditions, ShortestNumbersRoundTrip)
{
    const auto task = number_task();
    auto check = [&](u64 bits)
    {
        const f64 value = std::bit_cast<f64>(bits);
        if (!std::isfinite(value))
            return;
        std::array<char, 64> buffer{};
        const auto r = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        ASSERT_EQ(r.ec, std::errc{});
        expect_number(*task, std::string_view(buffer.data(), r.ptr), value);
    };
    for (u64 exponent = 0; exponent < 2047; ++exponent)
        for (u64 mantissa : {u64{0}, u64{1}, u64{1} << 51, (u64{1} << 52) - 1})
            for (u64 sign : {u64{0}, u64{1} << 63})
                check(sign | exponent << 52 | mantissa);
    std::mt19937_64 rng(17);
    for (u32 i = 0; i < 20000; ++i)
        check(rng());
}

TEST(Conditions, NumberSyntaxAndRangeErrors)
{
    const auto task = number_task();
    for (const auto& [text, value] : std::array<std::pair<std::string_view, f64>, 15>{{
             {"0", 0.0}, {"-0", -0.0}, {"1.", 1.0}, {".5", 0.5}, {"-1E-2", -0.01},
             {"0e999999999999999999999", 0.0}, {"-0e-999999999999999999999", -0.0},
             {"inf", INFINITY}, {"-INF", -INFINITY}, {"INFINITY", INFINITY}, {"-infinity", -INFINITY},
             {"nan", NAN}, {"-NaN", -NAN}, {"5e-324", std::numeric_limits<f64>::denorm_min()},
             {"1.7976931348623158e308", std::numeric_limits<f64>::max()}}})
        expect_number(*task, text, value);

    GroundCondition ground;
    ConjunctiveCondition lifted;
    ground.add_constraint(*task, "(= 2 3)");
    lifted.add_constraint(*task, "(= 2 3)");
    const auto ground_before = ground;
    const auto lifted_before = lifted;
    for (const char* bad : {"", "+1", "-", ".", "1e", "1e+", "1e-", "--1", "1_000", "1,5", "1junk",
                            "0x1", "0x1p0", "0x.8p0", "1p0", "infinite", "nanosecond", "nan(payload)",
                            "1e309", "-1e309", "1e-1000", "-1e-1000", "2.4703282292062327e-324"})
    {
        const std::string text = "(= " + std::string(bad) + " 0)";
        EXPECT_THROW(ground.add_constraint(*task, text), std::invalid_argument) << bad;
        EXPECT_THROW(lifted.add_constraint(*task, text), std::invalid_argument) << bad;
        EXPECT_EQ(ground, ground_before) << bad;
        EXPECT_EQ(lifted, lifted_before) << bad;
    }
}

TEST(Conditions, NumbersIgnoreNumericLocale)
{
    const auto task = number_task();
    const NumericEnvironment restore;
    for (const char* locale : {"C", "de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8"})
    {
        if (!std::setlocale(LC_NUMERIC, locale))
            continue;
        expect_number(*task, "1.25", 1.25);
        expect_number(*task, "1.25e-2", 0.0125);
        EXPECT_THROW(parse_number(*task, "1,25"), std::invalid_argument);
    }
}

TEST(Conditions, NumbersRoundToNearestEven)
{
    const auto task = number_task();
    const NumericEnvironment restore;
    for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(std::fesetround(mode), 0);
        expect_number(*task, "0", 0.0);
        expect_number(*task, "-0", -0.0);
        expect_number(*task, "0.1", 0.1);
        expect_number(*task, "-0.1", -0.1);
        expect_number(*task, "1.00000000000000011102230246251565404236316680908203125", 1.0);
        expect_number(*task, "1.00000000000000011102230246251565404236316680908203126", std::nextafter(1.0, 2.0));
        expect_number(*task, "1.00000000000000033306690738754696212708950042724609375", 0x1.0000000000002p0);
        expect_number(*task, "2.4703282292062328e-324", std::numeric_limits<f64>::denorm_min());
    }
}

#if defined(__linux__)
TEST(Conditions, NumbersMatchFromChars)
{
    const auto task = number_task();
    const NumericEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto check = [&](const std::string& text)
    {
        f64 expected = 42;
        const auto r = std::from_chars(text.data(), text.data() + text.size(), expected);
        if (r.ec == std::errc{} && r.ptr == text.data() + text.size())
            expect_number(*task, text, expected);
        else
            EXPECT_THROW(parse_number(*task, text), std::invalid_argument) << text;
    };
    for (const char* text : {"0", "-0", "1.", ".5", "0e999999999999999999999", "-0e-999999999999999999999",
                             "inf", "-inf", "infinity", "-infinity", "nan", "-nan", "5e-324", "2.2250738585072014e-308",
                             "2.4703282292062327e-324", "2.4703282292062328e-324", "1.7976931348623158e308",
                             "1.7976931348623159e308", "+1", "-", ".", "1e", "1e+", "0x1p0", "1p0", "1,5",
                             "1e999999999999999999999", "1e-999999999999999999999"})
        check(text);
    std::mt19937_64 rng(31);
    for (u32 i = 0; i < 10000; ++i)
    {
        std::string text = i % 2 ? "-" : "";
        const usize digits = 1 + rng() % (i % 4 == 0 ? 1800 : 30);
        const usize point = rng() % (digits + 1);
        for (usize j = 0; j < digits; ++j)
        {
            if (j == point)
                text += '.';
            text += static_cast<char>('0' + rng() % 10);
        }
        text += "e" + std::to_string(static_cast<int>(rng() % 3001) - 1500);
        check(text);
    }
}
#endif
