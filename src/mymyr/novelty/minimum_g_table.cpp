#include "mymyr/novelty/minimum_g_table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mymyr::novelty
{
MinimumGNoveltyTable::MinimumGNoveltyTable(u32 arity) : m_k(arity)
{
    if (arity < 1 || arity > k_max_arity)
        throw std::invalid_argument("mymyr: minimum-g novelty arity must be in 1..5");
}

usize MinimumGNoveltyTable::KeyHash::operator()(const Key& k) const noexcept
{
    u64 h = hash::combine(k.coordinate, k.rank);
    if (k.packed)
        for (u32 a : k.atoms)
            h = hash::combine(h, a);
    return static_cast<usize>(h);
}

MinimumGNoveltyTable::Key MinimumGNoveltyTable::key(std::span<const u32> tuple, u32 coordinate)
{
    Key out;
    out.coordinate = coordinate;
    // The same combinadic numbering as NoveltyTable: sum C(a_i, i + 1).
    for (u32 i = 0; i < tuple.size() && i < k_max_arity; ++i)
    {
        u64 c = 1;
        for (u32 j = 0; j <= i; ++j)
        {
            if (tuple[i] < j + 1)
            {
                c = 0;
                break;
            }
            const u64 d = std::gcd(c, u64{j} + 1);
            const u64 x = c / d, y = (u64{tuple[i]} - j) / ((u64{j} + 1) / d);
            if (y && x > std::numeric_limits<u64>::max() / y)
            {
                out.packed = true;
                break;
            }
            c = x * y;
        }
        if (out.packed || c > std::numeric_limits<u64>::max() - out.rank)
        {
            out.packed = true;
            out.rank = 0;
            std::copy(tuple.begin(), tuple.end(), out.atoms.begin());
            return out;
        }
        out.rank += c;
    }
    return out;
}

template<class Visit>
bool MinimumGNoveltyTable::tuples(StateView state, const StateView* parent, Visit&& visit) const
{
    std::vector<u32> atoms;
    bits::for_each(state.w, state.nw, [&](u64 a) { atoms.push_back(static_cast<u32>(a)); });
    if (atoms.size() + 1 < m_k)
        return false;
    std::array<u32, k_max_arity> tuple{};
    auto walk = [&](auto& self, usize from, u32 size, bool added) -> bool
    {
        if ((!parent || added) && visit(std::span<const u32>(tuple.data(), size)))
            return true;
        if (size >= m_k || size >= k_max_arity)
            return false;
        for (usize i = from; i < atoms.size(); ++i)
        {
            tuple[size] = atoms[i];
            if (self(self, i + 1, size + 1, added || (parent && !parent->contains(SlotId{atoms[i]}))))
                return true;
        }
        return false;
    };
    return walk(walk, 0, 0, false);
}

bool MinimumGNoveltyTable::update(StateView s, const StateView* p, f64 g, u32 coordinate)
{
    if (!std::isfinite(g))
        throw std::invalid_argument("mymyr: minimum-g novelty labels must be finite");
    bool improved = false;
    tuples(s, p, [&](std::span<const u32> t)
    {
        auto [it, inserted] = m_labels[t.size()].try_emplace(key(t, coordinate), g);
        if (inserted)
            improved = true;
        else if (g < it->second)
        {
            it->second = g;
            m_lowered = improved = true;
        }
        return false;  // lower every tuple, even after finding an improvement
    });
    return improved;
}

bool MinimumGNoveltyTable::test_and_update(StateView s, f64 g, u32 coordinate) { return update(s, nullptr, g, coordinate); }
bool MinimumGNoveltyTable::test_and_update(StateView p, StateView s, f64 g, u32 coordinate) { return update(s, &p, g, coordinate); }
bool MinimumGNoveltyTable::would_improve(StateView s, f64 g, u32 coordinate) const
{
    return tuples(s, nullptr, [&](std::span<const u32> t)
    {
        const auto& labels = m_labels[t.size()];
        const auto it = labels.find(key(t, coordinate));
        return it == labels.end() || g < it->second;
    });
}
bool MinimumGNoveltyTable::would_improve(StateView p, StateView s, f64 g, u32 coordinate) const
{
    return tuples(s, &p, [&](std::span<const u32> t)
    {
        const auto& labels = m_labels[t.size()];
        const auto it = labels.find(key(t, coordinate));
        return it == labels.end() || g < it->second;
    });
}
bool MinimumGNoveltyTable::test_at_g(StateView s, f64 g, u32 coordinate) const
{
    return tuples(s, nullptr, [&](std::span<const u32> t)
    {
        const auto& labels = m_labels[t.size()];
        const auto it = labels.find(key(t, coordinate));
        return it != labels.end() && g == it->second;
    });
}
u64 MinimumGNoveltyTable::bytes() const noexcept
{
    u64 bytes = 0;
    for (const Labels& l : m_labels)
        bytes += l.bucket_count() * sizeof(void*) + l.size() * (sizeof(Labels::value_type) + sizeof(void*));
    return bytes;
}
}  // namespace mymyr::novelty
