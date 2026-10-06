#pragma once
// States are values.
//   - A State owns its fluent words and its numeric words (inline for up to 4 words in total, on the heap otherwise)
//     and caches its hash. It references no task, store or search, is valid on any thread, and compares equal to every
//     state with the same atoms and the same numeric slot values.
//   - Words are little-endian u64; fluent slot i is bit i & 63 of word i >> 6. A State keeps its fluent words
//     *trimmed* (no trailing zero words), so equality is a length check plus memcmp and the hash is hash::state.
//   - Numeric words (numeric tasks only; task/numeric.hpp): the task's numeric_words() words after the bits, in
//     the task's storage (f64 bit patterns or two int32 per word), canonical bit patterns. Classical tasks have none,
//     and their states, hashes and word layouts are exactly those without numerics.
//   - StateView is a non-owning view of any word run plus its numeric words; trailing zero words and missing words of
//     the fluent part read as zero.
//   - StateBuilder is a growable scratch buffer for building successors outside the hot search loops.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"

#include <cstring>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace mymyr
{
struct StateView
{
    const u64* w = nullptr;
    u32 nw = 0;
    const u64* num = nullptr;  // numeric words
    u32 nnum = 0;

    [[nodiscard]] std::span<const u64> words() const { return {w, nw}; }
    [[nodiscard]] std::span<const u64> numeric() const { return {num, nnum}; }
    [[nodiscard]] bool contains(SlotId s) const { return bits::test(w, nw, s.v); }
    [[nodiscard]] u32 trimmed_size() const { return bits::trimmed_size(w, nw); }
    [[nodiscard]] u64 hash() const { return hash::state(w, nw, num, nnum); }
    [[nodiscard]] u64 count() const { return bits::count(w, nw); }
    friend bool operator==(StateView a, StateView b)
    {
        return bits::equal(a.w, a.nw, b.w, b.nw) && a.nnum == b.nnum &&
               (a.nnum == 0 || std::memcmp(a.num, b.num, a.nnum * sizeof(u64)) == 0);
    }
};

class State
{
public:
    static constexpr u32 k_inline_words = 4;

    State() noexcept : m_hash(hash::state_words(nullptr, 0)) {}
    /// Copies the words (any length; trailing zero words are dropped).
    State(const u64* w, u32 n) { init(w, bits::trimmed_size(w, n), nullptr, 0); }
    /// Copies the fluent words and the numeric words.
    State(const u64* w, u32 n, const u64* num, u32 nnum) { init(w, bits::trimmed_size(w, n), num, nnum); }
    explicit State(StateView v) : State(v.w, v.nw, v.num, v.nnum) {}
    explicit State(std::span<const u64> w) : State(w.data(), static_cast<u32>(w.size())) {}
    State(const State& o) { init(o.data(), o.m_n, o.data() + o.m_n, o.m_nn, o.m_hash); }
    State(State&& o) noexcept { steal(o); }
    State& operator=(const State& o)
    {
        if (this != &o)
        {
            release();
            init(o.data(), o.m_n, o.data() + o.m_n, o.m_nn, o.m_hash);
        }
        return *this;
    }
    State& operator=(State&& o) noexcept
    {
        if (this != &o)
        {
            release();
            steal(o);
        }
        return *this;
    }
    ~State() { release(); }

    /// The fluent words (trimmed).
    [[nodiscard]] std::span<const u64> words() const noexcept { return {data(), m_n}; }
    [[nodiscard]] u32 size_words() const noexcept { return m_n; }
    [[nodiscard]] const u64* data() const noexcept { return m_n + m_nn > k_inline_words ? m_heap : m_inline; }
    /// The numeric words (empty for classical tasks).
    [[nodiscard]] std::span<const u64> numeric() const noexcept { return {data() + m_n, m_nn}; }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_nn; }
    [[nodiscard]] bool contains(SlotId s) const noexcept { return bits::test(data(), m_n, s.v); }
    [[nodiscard]] u64 hash() const noexcept { return m_hash; }
    [[nodiscard]] u64 count() const noexcept { return bits::count(data(), m_n); }
    [[nodiscard]] StateView view() const noexcept { return {data(), m_n, m_nn ? data() + m_n : nullptr, m_nn}; }
    operator StateView() const noexcept { return view(); }  // NOLINT(google-explicit-constructor)

    /// Slots of the true atoms, in increasing order.
    [[nodiscard]] std::vector<SlotId> slots() const
    {
        std::vector<SlotId> out;
        bits::for_each(data(), m_n, [&](u64 b) { out.push_back(SlotId{static_cast<u32>(b)}); });
        return out;
    }

    friend bool operator==(const State& a, const State& b) noexcept
    {
        return a.m_hash == b.m_hash && a.m_n == b.m_n && a.m_nn == b.m_nn &&
               std::memcmp(a.data(), b.data(), (a.m_n + a.m_nn) * sizeof(u64)) == 0;
    }

private:
    void init(const u64* w, u32 n, const u64* num, u32 nn) { init(w, n, num, nn, hash::state(w, n, num, nn)); }
    void init(const u64* w, u32 n, const u64* num, u32 nn, u64 h)
    {
        m_n = n;
        m_nn = nn;
        m_hash = h;
        u64* dst = m_inline;
        if (n + nn > k_inline_words)
            dst = m_heap = new u64[n + nn];
        if (n)
            std::memcpy(dst, w, n * sizeof(u64));
        if (nn)
            std::memcpy(dst + n, num, nn * sizeof(u64));
    }
    void steal(State& o) noexcept
    {
        m_n = o.m_n;
        m_nn = o.m_nn;
        m_hash = o.m_hash;
        if (m_n + m_nn > k_inline_words)
            m_heap = o.m_heap;
        else
            std::memcpy(m_inline, o.m_inline, sizeof(m_inline));
        o.m_n = 0;
        o.m_nn = 0;
        o.m_hash = hash::state_words(nullptr, 0);
    }
    void release() noexcept
    {
        if (m_n + m_nn > k_inline_words)
            delete[] m_heap;
        m_n = 0;
        m_nn = 0;
    }

    u64 m_hash = 0;
    u32 m_n = 0;   // fluent words (trimmed)
    u32 m_nn = 0;  // numeric words
    union
    {
        u64 m_inline[k_inline_words] = {};
        u64* m_heap;
    };
};

class StateBuilder
{
public:
    StateBuilder() = default;
    explicit StateBuilder(StateView v) { assign(v); }

    void assign(StateView v)
    {
        m_w.assign(v.w, v.w + v.nw);
        m_num.assign(v.num, v.num + v.nnum);
    }
    void clear()
    {
        m_w.clear();
        m_num.clear();
    }
    void set(SlotId s)
    {
        if (bits::word_of(s.v) >= m_w.size())
            m_w.resize(bits::word_of(s.v) + 1, 0);
        bits::set(m_w.data(), s.v);
    }
    void reset(SlotId s)
    {
        if (bits::word_of(s.v) < m_w.size())
            bits::reset(m_w.data(), s.v);
    }
    /// PDDL semantics: delete first, then add (an atom both deleted and added ends up true).
    void apply(std::span<const SlotId> del, std::span<const SlotId> add)
    {
        for (SlotId s : del)
            reset(s);
        for (SlotId s : add)
            set(s);
    }
    [[nodiscard]] bool contains(SlotId s) const { return bits::test(m_w.data(), static_cast<u32>(m_w.size()), s.v); }
    [[nodiscard]] StateView view() const
    {
        return {m_w.data(), static_cast<u32>(m_w.size()), m_num.empty() ? nullptr : m_num.data(), static_cast<u32>(m_num.size())};
    }
    [[nodiscard]] State build() const { return State(view()); }
    [[nodiscard]] std::vector<u64>& words() { return m_w; }
    /// The numeric words (numeric tasks: numeric_words() words).
    [[nodiscard]] std::vector<u64>& numeric() { return m_num; }

private:
    std::vector<u64> m_w;
    std::vector<u64> m_num;
};
}  // namespace mymyr

template<>
struct std::hash<mymyr::State>
{
    std::size_t operator()(const mymyr::State& s) const noexcept { return static_cast<std::size_t>(s.hash()); }
};
