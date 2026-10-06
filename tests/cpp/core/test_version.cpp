#include "mymyr/core/ids.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/version.hpp"

#include <gtest/gtest.h>

#include <unordered_set>

using namespace mymyr;

TEST(Version, IsSetFromCMake)
{
    EXPECT_FALSE(version().empty());
    EXPECT_NE(version(), "0.0.0");
    EXPECT_NE(build_info().find(version()), std::string_view::npos);
}

TEST(Ids, AreStrongAndDefaultInvalid)
{
    ObjectId o;
    EXPECT_FALSE(o.valid());
    ObjectId a{3}, b{3}, c{4};
    EXPECT_EQ(a, b);
    EXPECT_LT(a, c);
    std::unordered_set<ObjectId> s{a, b, c};
    EXPECT_EQ(s.size(), 2u);
    static_assert(!std::is_convertible_v<ObjectId, SlotId>);
    static_assert(!std::is_convertible_v<u32, ObjectId>);
}

TEST(Uid, IsUniqueAndMonotone)
{
    const u64 a = next_uid(), b = next_uid();
    EXPECT_LT(a, b);
}

TEST(Memory, HostResourceIsCacheLineAligned)
{
    auto* mr = host_memory();
    EXPECT_EQ(mr->kind(), MemKind::Host);
    void* p = mr->allocate(100, 8);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % k_cache_line, 0u);
    mr->deallocate(p, 100, 8);
}
