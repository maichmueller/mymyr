// State stores: Flat and Chunked (dedup, ids, widening under growing widths), the compact fingerprint set, and the
// concurrent store under contention (run under TSan).

#include "mymyr/core/team.hpp"
#include "mymyr/state/chunked_store.hpp"
#include "mymyr/state/compact_store.hpp"
#include "mymyr/state/concurrent_store.hpp"
#include "mymyr/state/flat_store.hpp"

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <set>
#include <vector>

using namespace mymyr;

namespace
{
/// Random states of 1..maxw words (trimmed), with many duplicates.
std::vector<std::vector<u64>> random_states(u32 count, u32 distinct, u32 maxw, u64 seed)
{
    std::mt19937_64 rng(seed);
    std::vector<std::vector<u64>> pool;
    for (u32 i = 0; i < distinct; ++i)
    {
        std::vector<u64> w(1 + rng() % maxw);
        for (auto& x : w)
            x = rng() & rng();
        w.back() |= 1;  // trimmed
        pool.push_back(w);
    }
    std::vector<std::vector<u64>> out;
    for (u32 i = 0; i < count; ++i)
        out.push_back(pool[rng() % distinct]);
    return out;
}

template<class Store>
void check_single_threaded(Store& store, u32 maxw)
{
    const auto states = random_states(20000, 3000, maxw, 42);
    std::map<std::vector<u64>, u32> ids;
    for (const auto& s : states)
    {
        auto [id, fresh] = store.insert(s.data(), static_cast<u32>(s.size()));
        auto it = ids.find(s);
        if (it == ids.end())
        {
            ASSERT_TRUE(fresh);
            ASSERT_EQ(id.v, ids.size());  // ids in insertion order
            ids.emplace(s, id.v);
        }
        else
        {
            ASSERT_FALSE(fresh);
            ASSERT_EQ(id.v, it->second);
        }
    }
    ASSERT_EQ(store.size(), ids.size());
    for (const auto& [s, id] : ids)
    {
        const State back = store.state(StateId{id});
        ASSERT_EQ(back, State(s.data(), static_cast<u32>(s.size())));
    }
    // padding with zero words does not create a new state
    for (const auto& [s, id] : ids)
    {
        std::vector<u64> padded = s;
        padded.resize(s.size() + 3, 0);
        auto [pid, fresh] = store.insert(padded.data(), static_cast<u32>(padded.size()));
        ASSERT_FALSE(fresh);
        ASSERT_EQ(pid.v, id);
    }
}

TEST(FlatStore, DedupIdsAndWidening)
{
    FlatStateStore store(1);  // starts narrow: widens up to 20 words
    check_single_threaded(store, 20);
    EXPECT_GE(store.stride(), 20u);
    const std::vector<u64> absent{0xdeadbeef, 7};
    EXPECT_FALSE(store.find({absent.data(), 2, nullptr, 0}).valid());
}

TEST(ChunkedStore, DedupIdsAndWidening)
{
    ChunkedStateStore store(8);
    check_single_threaded(store, 30);
    EXPECT_GE(store.words(), 30u);
    EXPECT_GT(store.num_chunks(), 0u);
}

TEST(ChunkedStore, SuccessorInsertReusesUntouchedChunks)
{
    ChunkedStateStore store(24);
    std::vector<u64> a(24, 0), b(24, 0);
    a[0] = 1;
    a[10] = 5;
    a[20] = 9;
    auto [ia, fa] = store.insert(a.data(), 24);
    ASSERT_TRUE(fa);
    const u32 chunks_before = store.num_chunks();
    b = a;
    b[10] |= 2;  // slot 10 * 64 + 1 in chunk 1
    const SlotId add{10 * 64 + 1};
    const Delta d{std::span<const SlotId>(&add, 1), {}};
    auto [ib, fb] = store.insert_successor(ia, a.data(), b.data(), d);
    ASSERT_TRUE(fb);
    EXPECT_EQ(store.num_chunks(), chunks_before + 1);  // only chunk 1 is new
    EXPECT_EQ(store.ids(ib)[0], store.ids(ia)[0]);
    EXPECT_EQ(store.ids(ib)[2], store.ids(ia)[2]);
    EXPECT_EQ(store.state(ib), State(b.data(), 24));
    EXPECT_FALSE(store.insert(b.data(), 24).second);
}

TEST(CompactSet, InsertsFingerprints)
{
    CompactStateSet set;
    std::mt19937_64 rng(1);
    std::set<std::pair<u64, u64>> ref;
    for (u32 i = 0; i < 200000; ++i)
    {
        Fingerprint128 f{rng() % 50000, rng() % 3};
        EXPECT_EQ(set.insert(f), ref.insert({f.a, f.b}).second);
    }
    EXPECT_EQ(set.size(), ref.size());
}

TEST(ConcurrentStore, ConcurrentInsertsDeduplicateAndKeepMinimalKeys)
{
    const u32 T = 8;
    ConcurrentStateStore store(T);
    Team team(T);
    const auto states = random_states(60000, 5000, 6, 7);
    std::vector<std::vector<std::pair<ConcurrentStateStore::Handle, bool>>> res(T);
    team.run(
        [&](u32 t)
        {
            for (usize i = t; i < states.size(); i += T)
            {
                res[t].push_back(store.insert(t, states[i].data(), static_cast<u32>(states[i].size()), i));
                if (store.wants_rehash())
                    break;  // not exercised here: the initial table holds 2^18 entries
            }
        });
    // every state maps to one handle; the number of fresh inserts equals the number of distinct states
    std::map<std::vector<u64>, ConcurrentStateStore::Handle> handle_of;
    std::map<std::vector<u64>, u64> min_index;
    u64 fresh = 0;
    for (u32 t = 0; t < T; ++t)
        for (usize k = 0; k < res[t].size(); ++k)
        {
            const usize i = t + k * T;
            const auto& s = states[i];
            auto [it, ins] = handle_of.emplace(s, res[t][k].first);
            if (!ins)
            {
                ASSERT_EQ(it->second, res[t][k].first);
            }
            fresh += res[t][k].second;
            auto m = min_index.emplace(s, i);
            if (!m.second)
                m.first->second = std::min<u64>(m.first->second, i);
            const StateView r = store.record(res[t][k].first);
            ASSERT_EQ(State(r), State(s.data(), static_cast<u32>(s.size())));
        }
    EXPECT_EQ(fresh, handle_of.size());
    // one layer: every record's key is the minimal discoverer key
    for (const auto& [s, h] : handle_of)
        EXPECT_EQ(store.key(h), min_index[s]);
}

TEST(ConcurrentStore, ParallelRehashKeepsEverything)
{
    const u32 T = 4;
    ConcurrentStateStore store(T);
    Team team(T);
    std::vector<std::vector<ConcurrentStateStore::Handle>> handles(T);
    const u64 per_thread = 300000;  // 1.2M states: forces rehashes of the 2^18-entry table
    std::atomic<bool> resize{false};
    std::vector<u64> next(T, 0);
    for (;;)
    {
        team.run(
            [&](u32 t)
            {
                for (u64& i = next[t]; i < per_thread; ++i)
                {
                    const u64 w[2] = {(i * T + t) | (u64{1} << 40), i % 3};
                    handles[t].push_back(store.insert(t, w, 2, 0).first);
                    if (store.wants_rehash())
                        resize.store(true);
                    if (resize.load())
                    {
                        ++i;
                        break;
                    }
                }
            });
        if (!resize.load())
            break;
        store.rehash(team);
        resize.store(false);
    }
    EXPECT_GT(store.capacity(), u64{1} << 19);
    for (u32 t = 0; t < T; ++t)
        for (u64 i = 0; i < per_thread; ++i)
        {
            const u64 w[2] = {(i * T + t) | (u64{1} << 40), i % 3};
            auto [h, fresh] = store.insert(t, w, 2, 0);
            ASSERT_FALSE(fresh);
            ASSERT_EQ(h, handles[t][i]);
        }
}
}  // namespace
