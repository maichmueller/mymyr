#include "mymyr/rl/expand.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::rl
{
namespace
{
constexpr u64 k_i32_max = static_cast<u64>(std::numeric_limits<i32>::max());

/// Throws unless row `i` sets only assigned fluent slots (bits at or beyond `limit` would index unassigned records).
void check_row(const u64* row, u32 nw, u32 limit, u64 i)
{
    const u32 lw = limit >> 6;
    if (nw <= lw)
        return;
    u64 bad = row[lw] & ~((u64{1} << (limit & 63)) - 1);
    for (u32 w = lw + 1; w < nw && !bad; ++w)
        bad = row[w];
    if (bad)
        throw std::invalid_argument("mymyr: expand: state row " + std::to_string(i) +
                                    " sets atom slots its instance has not assigned (a state of another task?)");
}

void check_numeric(const TaskSuite& suite, u32 numeric_words, const char* what)
{
    if (numeric_words != suite.numeric_words())
        throw std::invalid_argument(std::string("mymyr: expand: ") + what + " carry " + std::to_string(numeric_words) +
                                    " numeric words per row, the " + suite.noun() + " has " +
                                    std::to_string(suite.numeric_words()));
}

void check_batch(const TaskSuite& table, StateBatchView in, const i32* ids, const ExpandOptions& opt)
{
    if (in.rows && !in.data)
        throw std::invalid_argument("mymyr: expand: null state buffer");
    if (in.stride && in.stride < static_cast<u64>(in.words) + in.numeric_words)
        throw std::invalid_argument("mymyr: expand: row stride smaller than the row width");
    check_numeric(table, in.numeric_words, "the states");
    if (in.rows > k_i32_max)
        throw std::invalid_argument("mymyr: expand: more than 2^31 - 1 states in one batch");
    if (!ids && in.rows && table.size() > 1)
        throw std::invalid_argument(std::string("mymyr: expand: a batch over a ") + table.noun() + " of " +
                                    std::to_string(table.size()) + " instances needs task ids");
    table.check_task_ids(ids, in.rows);
    if (!opt.validate)
        return;
    for (u64 i = 0; i < in.rows; ++i)
    {
        const u32 limit = table.task_of(ids, i).atoms().fluent_slots();
        if (static_cast<u64>(in.words) * 64 > limit)
            check_row(in.row(i), in.words, limit, i);
    }
}

void check_out(const TaskSuite& table, const Expansion& out)
{
    if (out.binding && out.label_width < table.label_width())
        throw std::invalid_argument("mymyr: expand: label width " + std::to_string(out.label_width) +
                                    " is below the largest schema arity " + std::to_string(table.label_width()));
    if (out.capacity > k_i32_max)
        throw std::invalid_argument("mymyr: expand: capacity above 2^31 - 1 rows");
    if (out.succ && out.words == 0 && out.capacity)
        throw std::invalid_argument("mymyr: expand: successor rows of zero words");
    if (out.goal && !out.succ)
        throw std::invalid_argument("mymyr: expand: goal flags need the successor rows");
    if (out.succ)
        check_numeric(table, out.numeric_words, "the successor rows");
}

/// Successor generators of a suite's instances for one thread, leased on first use per instance and returned with the
/// object.
class Generators
{
public:
    explicit Generators(const TaskSuite& table) : m_table(table), m_ws(table.size()), m_succ(table.size(), nullptr) {}
    [[nodiscard]] Successors& of(u32 instance)
    {
        Successors*& s = m_succ[instance];
        if (!s)
        {
            m_ws[instance] = m_table.task(instance)->workspace();
            s = &m_ws[instance]->successors();
        }
        return *s;
    }

private:
    const TaskSuite& m_table;
    std::vector<WorkspaceLease> m_ws;
    std::vector<Successors*> m_succ;
};

inline u32 instance_of(const i32* ids, u64 row) noexcept { return ids ? static_cast<u32>(ids[row]) : 0; }

/// Exact trimmed width of the successor of cur under d (the rare path: it does not fit the output rows).
template<class Vec>
u32 exact_width(const u64* cur, u32 nw, const Delta& d, Vec& tmp)
{
    return apply_delta(cur, nw, d, tmp);
}

/// Writes the successor of cur (nw words, nw <= W) under d into dst (W atom words, then NN numeric words: the
/// instance's d.nnum, then zeros). Returns false, writing nothing, if it needs more than W words.
inline bool write_successor(u64* dst, u32 W, u32 NN, const u64* cur, u32 nw, const Delta& d)
{
    const u64 limit = static_cast<u64>(W) * 64;
    for (SlotId a : d.add)
        if (a.v >= limit)
            return false;
    u32 i = 0;
    MYMYR_NOVECTOR
    for (; i < nw; ++i)
        dst[i] = cur[i];
    MYMYR_NOVECTOR
    for (; i < W; ++i)
        dst[i] = 0;
    for (SlotId x : d.del)
        if (x.v < limit)
            bits::reset(dst, x.v);
    for (SlotId x : d.add)
        bits::set(dst, x.v);
    u32 k = 0;
    for (; k < d.nnum; ++k)
        dst[W + k] = d.num[k];
    for (; k < NN; ++k)
        dst[W + k] = 0;
    return true;
}

inline bool fits_width(u32 W, const Delta& d)
{
    const u64 limit = static_cast<u64>(W) * 64;
    for (SlotId a : d.add)
        if (a.v >= limit)
            return false;
    return true;
}

/// Expands one state; calls put(row index local to the sink, schema, binding, delta) per successor through `sink`.
/// Returns the number of successors.
template<class Sink>
u64 expand_state(Successors& succ, const u64* row, u32 nw, const u64* num, u32 nnum, const ExpandOptions& opt, Sink& sink)
{
    succ.prepare(StateView{row, nw, num, nnum});
    u64 n = 0;
    succ.generate<false>(
        [&](u32 schema, const ObjectId* binding, const Delta& d) -> bool
        {
            sink.put(schema, binding, succ.arity(schema), d);
            ++n;
            return true;
        },
        opt.witness_pruning, opt.canonical_order);
    return n;
}

/// Direct sink: writes into the caller's buffers, rows [0, capacity).
struct DirectSink
{
    Expansion& out;
    const u64* cur;
    u32 nw;       // trimmed width of cur
    i32 parent;   // current state
    u64 next;     // next flat row
    u32 L;
    std::vector<u64>& tmp;

    void put(u32 schema, const ObjectId* b, u32 arity, const Delta& d)
    {
        const u64 j = next++;
        const u32 W = out.words, RW = W + out.numeric_words;
        if (j < out.capacity)
        {
            if (out.succ)
            {
                u64* dst = out.succ + j * RW;
                if (nw > W || !write_successor(dst, W, out.numeric_words, cur, nw, d))
                {
                    out.words_needed = std::max(out.words_needed, exact_width(cur, nw, d, tmp));
                    std::memset(dst, 0, static_cast<usize>(RW) * sizeof(u64));
                }
            }
            else if (nw > W || !fits_width(W, d))
                out.words_needed = std::max(out.words_needed, exact_width(cur, nw, d, tmp));
            if (out.parent)
                out.parent[j] = parent;
            if (out.schema)
                out.schema[j] = static_cast<i32>(schema);
            if (out.binding)
            {
                i32* dst = out.binding + j * L;
                u32 k = 0;
                for (; k < arity; ++k)
                    dst[k] = static_cast<i32>(b[k].v);
                for (; k < L; ++k)
                    dst[k] = -1;
            }
        }
        else if (nw > W || !fits_width(W, d))
            out.words_needed = std::max(out.words_needed, exact_width(cur, nw, d, tmp));
    }
};

/// Scratch sink of one pool member: rows appended to growable vectors (every row is kept; the copy to the caller's
/// buffers clips to the capacity).
struct ScratchSink
{
    LineVector<u64>& succ;
    LineVector<i32>& parent;
    LineVector<i32>& schema;
    LineVector<i32>& binding;
    bool want_succ, want_parent, want_schema, want_binding;
    const u64* cur;
    u32 nw;
    i32 state;
    u32 W, NN, L;
    u32& words_needed;
    LineVector<u64>& tmp;
    u64 rows = 0;

    void put(u32 s, const ObjectId* b, u32 arity, const Delta& d)
    {
        if (want_succ)
        {
            const usize at = succ.size();
            succ.resize(at + W + NN);
            if (nw > W || !write_successor(succ.data() + at, W, NN, cur, nw, d))
            {
                words_needed = std::max(words_needed, exact_width(cur, nw, d, tmp));
                std::memset(succ.data() + at, 0, static_cast<usize>(W + NN) * sizeof(u64));
            }
        }
        else if (nw > W || !fits_width(W, d))
            words_needed = std::max(words_needed, exact_width(cur, nw, d, tmp));
        if (want_parent)
            parent.push_back(state);
        if (want_schema)
            schema.push_back(static_cast<i32>(s));
        if (want_binding)
        {
            u32 k = 0;
            for (; k < arity; ++k)
                binding.push_back(static_cast<i32>(b[k].v));
            for (; k < L; ++k)
                binding.push_back(-1);
        }
        ++rows;
    }
};

void write_offsets(i32* offsets, u64 i, u64 value)
{
    if (value > k_i32_max)
        throw std::length_error("mymyr: expand: more than 2^31 - 1 successors in one batch");
    offsets[i] = static_cast<i32>(value);
}

/// The goal flag of a successor row of instance `task` (the row carries the table's NN numeric words, the task reads
/// its own).
inline u8 goal_of(const Task& task, const u64* s, u32 W, u32 NN)
{
    const u32 nn = task.numeric_words();
    return static_cast<u8>(task.is_goal(StateView{s, W, NN ? s + W : nullptr, nn}));
}
}  // namespace

struct ExpandScratch::Impl
{
    struct Chunk
    {
        u64 first_state, end_state;
        u64 first_row, rows;  // in the member's buffers
    };
    struct alignas(k_cache_line) Member  // one per pool thread, written per state: no shared cache lines
    {
        LineVector<u64> succ;
        LineVector<i32> parent, schema, binding;
        LineVector<u8> goal;
        LineVector<u64> tmp;
        LineVector<Chunk> chunks;
        u32 words_needed = 0;
    };
    std::vector<Member> members;
    std::vector<u64> counts;  // per state
    std::atomic<u64> next_chunk{0};
};

ExpandScratch::ExpandScratch() : m_impl(std::make_unique<Impl>()) {}
ExpandScratch::~ExpandScratch() = default;

u32 max_label_width(const Task& task) noexcept
{
    u32 L = 0;
    for (const plan::Schema& s : task.compiled().schemas)
        L = std::max(L, s.arity);
    return L;
}

void expand(const TaskSuite& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt)
{
    check_batch(table, in, ids, opt);
    check_out(table, out);
    out.total = 0;
    out.words_needed = 0;
    Generators gens(table);
    std::vector<u64> tmp;
    const u32 L = out.label_width;
    u64 next = 0;
    if (out.offsets)
        out.offsets[0] = 0;
    const u32 NN = in.numeric_words, RW = out.words + out.numeric_words;
    for (u64 i = 0; i < in.rows; ++i)
    {
        const u32 inst = instance_of(ids, i);
        const Task& task = *table.task(inst);
        const u32 nn = task.numeric_words();
        const u64* row = in.row(i);
        const u32 nw = bits::trimmed_size(row, in.words);
        DirectSink sink{out, row, nw, static_cast<i32>(i), next, L, tmp};
        expand_state(gens.of(inst), row, nw, nn ? row + in.words : nullptr, nn, opt, sink);
        if (out.goal)
            for (u64 j = next; j < sink.next && j < out.capacity; ++j)
                out.goal[j] = goal_of(task, out.succ + j * RW, out.words, NN);
        next = sink.next;
        if (out.offsets)
            write_offsets(out.offsets, i + 1, next);
    }
    out.total = next;
}

void expand(const TaskSuite& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt,
            ThreadPool& pool, ExpandScratch& scratch)
{
    const u32 T = pool.size();
    if (T == 1 || in.rows < 2)
    {
        expand(table, in, ids, out, opt);
        return;
    }
    check_batch(table, in, ids, opt);
    check_out(table, out);
    out.total = 0;
    out.words_needed = 0;
    ExpandScratch::Impl& S = scratch.impl();
    if (S.members.size() < T)
        S.members.resize(T);
    S.counts.resize(in.rows);
    // Chunks small enough to balance (several per member), large enough to amortize the claim.
    const u64 chunk = std::clamp<u64>(in.rows / (static_cast<u64>(T) * 8), 1, 256);
    const u64 nchunks = (in.rows + chunk - 1) / chunk;
    S.next_chunk.store(0, std::memory_order_relaxed);
    const u32 W = out.words, L = out.label_width, NN = in.numeric_words, RW = W + out.numeric_words;
    const bool want_goal = out.goal != nullptr;

    pool.run(
        [&](u32 t)
        {
            ExpandScratch::Impl::Member& m = S.members[t];
            m.succ.clear();
            m.parent.clear();
            m.schema.clear();
            m.binding.clear();
            m.goal.clear();
            m.chunks.clear();
            m.words_needed = 0;
            Generators gens(table);
            for (;;)
            {
                const u64 c = S.next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (c >= nchunks)
                    break;
                const u64 b = c * chunk, e = std::min(in.rows, b + chunk);
                ExpandScratch::Impl::Chunk rec{b, e, 0, 0};
                rec.first_row = m.parent.size();
                u64 rows_before = 0;
                for (u64 i = b; i < e; ++i)
                {
                    const u32 inst = instance_of(ids, i);
                    const Task& task = *table.task(inst);
                    const u32 nn = task.numeric_words();
                    const u64* row = in.row(i);
                    const u32 nw = bits::trimmed_size(row, in.words);
                    ScratchSink sink{m.succ,       m.parent, m.schema, m.binding, true, true, out.schema != nullptr,
                                     out.binding != nullptr, row, nw, static_cast<i32>(i), W, out.numeric_words, L,
                                     m.words_needed, m.tmp};
                    const u64 before = rows_before;
                    expand_state(gens.of(inst), row, nw, nn ? row + in.words : nullptr, nn, opt, sink);
                    S.counts[i] = sink.rows;
                    rows_before += sink.rows;
                    if (want_goal)
                        for (u64 j = before; j < rows_before; ++j)
                            m.goal.push_back(goal_of(task, m.succ.data() + (rec.first_row + j) * RW, W, NN));
                }
                rec.rows = rows_before;
                m.chunks.push_back(rec);
            }
        });

    // offsets (exclusive prefix sums of the counts)
    u64 total = 0;
    if (out.offsets)
        out.offsets[0] = 0;
    for (u64 i = 0; i < in.rows; ++i)
    {
        const u64 c = S.counts[i];
        S.counts[i] = total;  // now: first flat row of state i
        total += c;
        if (out.offsets)
            write_offsets(out.offsets, i + 1, total);
    }
    out.total = total;
    for (u32 t = 0; t < T; ++t)
        out.words_needed = std::max(out.words_needed, S.members[t].words_needed);

    // copy each member's rows into place, clipped to the capacity
    pool.run(
        [&](u32 t)
        {
            const ExpandScratch::Impl::Member& m = S.members[t];
            for (const ExpandScratch::Impl::Chunk& c : m.chunks)
            {
                const u64 dst = S.counts[c.first_state];
                if (dst >= out.capacity || c.rows == 0)
                    continue;
                const u64 n = std::min(c.rows, out.capacity - dst);
                if (out.succ)
                    std::memcpy(out.succ + dst * RW, m.succ.data() + c.first_row * RW, n * RW * sizeof(u64));
                if (out.parent)
                    std::memcpy(out.parent + dst, m.parent.data() + c.first_row, n * sizeof(i32));
                if (out.schema)
                    std::memcpy(out.schema + dst, m.schema.data() + c.first_row, n * sizeof(i32));
                if (out.binding)
                    std::memcpy(out.binding + dst * L, m.binding.data() + c.first_row * L, n * L * sizeof(i32));
                if (out.goal)
                    std::memcpy(out.goal + dst, m.goal.data() + c.first_row, n * sizeof(u8));
            }
        });
}

void pad(const Expansion& flat, u64 rows, PaddedExpansion& out)
{
    if (!flat.offsets)
        throw std::invalid_argument("mymyr: pad: the flat expansion has no offsets");
    const u32 K = out.K, W = out.words, L = out.label_width, NN = out.numeric_words;
    const u32 RW = W + NN, FW = flat.words + flat.numeric_words;
    if (out.succ && flat.succ && W < flat.words)
        throw std::invalid_argument("mymyr: pad: padded rows narrower than the flat rows");
    if (out.succ && flat.succ && NN != flat.numeric_words)
        throw std::invalid_argument("mymyr: pad: padded and flat rows carry different numeric words");
    if (out.binding && flat.binding && L < flat.label_width)
        throw std::invalid_argument("mymyr: pad: padded label width below the flat label width");
    out.overflow = false;
    for (u64 i = 0; i < rows; ++i)
    {
        const u64 b = static_cast<u64>(flat.offsets[i]), e = static_cast<u64>(flat.offsets[i + 1]);
        const u64 c = e - b;
        if (out.count)
            out.count[i] = static_cast<i32>(c);
        if (c > K)
            out.overflow = true;
        for (u32 k = 0; k < K; ++k)
        {
            const u64 j = b + k;
            const bool valid = k < c && j < flat.capacity;
            if (k < c && j >= flat.capacity)
                out.overflow = true;
            const u64 at = i * K + k;
            if (out.index)
                out.index[at] = valid ? static_cast<i32>(j) : -1;
            if (out.mask)
                out.mask[at] = valid ? 1 : 0;
            if (out.succ)
            {
                u64* dst = out.succ + at * RW;
                const u64* src = flat.succ + j * FW;
                u32 w = 0;
                if (valid && flat.succ)
                    for (; w < flat.words; ++w)
                        dst[w] = src[w];
                for (; w < W; ++w)
                    dst[w] = 0;
                for (u32 k = 0; k < NN; ++k)
                    dst[W + k] = valid && flat.succ ? src[flat.words + k] : 0;
            }
            if (out.schema)
                out.schema[at] = valid && flat.schema ? flat.schema[j] : -1;
            if (out.binding)
            {
                i32* dst = out.binding + at * L;
                u32 w = 0;
                if (valid && flat.binding)
                    for (; w < flat.label_width; ++w)
                        dst[w] = flat.binding[j * flat.label_width + w];
                for (; w < L; ++w)
                    dst[w] = -1;
            }
            if (out.goal)
                out.goal[at] = valid && flat.goal ? flat.goal[j] : 0;
        }
    }
}

void is_goal(const TaskSuite& table, StateBatchView in, const i32* ids, u8* out)
{
    check_numeric(table, in.numeric_words, "the states");
    if (!ids && in.rows && table.size() > 1)
        throw std::invalid_argument(std::string("mymyr: is_goal: a batch over a ") + table.noun() + " of " +
                                    std::to_string(table.size()) + " instances needs task ids");
    table.check_task_ids(ids, in.rows);
    for (u64 i = 0; i < in.rows; ++i)
        out[i] = goal_of(table.task_of(ids, i), in.row(i), in.words, in.numeric_words);
}

namespace
{
template<class F>
void for_each_goal_row(StateBatchView in, StateBatchView gpos, StateBatchView gneg, F&& f)
{
    if ((gpos.rows != 1 && gpos.rows != in.rows) || (gneg.rows != 1 && gneg.rows != in.rows))
        throw std::invalid_argument("mymyr: goal masks need one row or one row per state");
    for (u64 i = 0; i < in.rows; ++i)
    {
        const u64* s = in.row(i);
        const u64* p = gpos.row(gpos.rows == 1 ? 0 : i);
        const u64* n = gneg.row(gneg.rows == 1 ? 0 : i);
        const u32 W = std::max({in.words, gpos.words, gneg.words});
        f(i, s, p, n, W);
    }
}
}  // namespace

void goal_test(StateBatchView in, StateBatchView gpos, StateBatchView gneg, u8* out)
{
    for_each_goal_row(in, gpos, gneg,
                      [&](u64 i, const u64* s, const u64* p, const u64* n, u32 W)
                      {
                          bool ok = true;
                          for (u32 w = 0; w < W && ok; ++w)
                          {
                              const u64 sw = w < in.words ? s[w] : 0;
                              const u64 pw = w < gpos.words ? p[w] : 0;
                              const u64 nw = w < gneg.words ? n[w] : 0;
                              ok = (sw & pw) == pw && (sw & nw) == 0;
                          }
                          out[i] = ok ? 1 : 0;
                      });
}

void goal_count(StateBatchView in, StateBatchView gpos, StateBatchView gneg, i32* out)
{
    for_each_goal_row(in, gpos, gneg,
                      [&](u64 i, const u64* s, const u64* p, const u64* n, u32 W)
                      {
                          i32 c = 0;
                          for (u32 w = 0; w < W; ++w)
                          {
                              const u64 sw = w < in.words ? s[w] : 0;
                              const u64 pw = w < gpos.words ? p[w] : 0;
                              const u64 nw = w < gneg.words ? n[w] : 0;
                              c += std::popcount(pw & ~sw) + std::popcount(nw & sw);
                          }
                          out[i] = c;
                      });
}

WalkStats random_walks(const Task& task, u64 steps, u64 episode, u64 seed, const ExpandOptions& opt)
{
    WalkStats st;
    const WorkspaceLease ws = task.workspace();
    Successors& succ = ws->successors();
    const State init = task.initial_state();
    State cur = init;
    LineVector<State> kids;  // per-thread hot scratch (see LineAllocator)
    LineVector<u64> tmp;
    u64 rng = seed * 0x9e3779b97f4a7c15ULL + 0x632be59bd9b4e019ULL;
    u64 depth = 0;
    while (st.steps < steps)
    {
        kids.clear();
        const u64* w = cur.data();
        const u32 nw = cur.size_words();
        succ.prepare(cur);
        succ.generate<false>(
            [&](u32, const ObjectId*, const Delta& d) -> bool
            {
                const u32 n = apply_delta(w, nw, d, tmp);
                kids.emplace_back(tmp.data(), n, d.num, d.nnum);
                return true;
            },
            opt.witness_pruning, opt.canonical_order);
        ++st.steps;
        ++depth;
        st.successors += kids.size();
        if (kids.empty())
            ++st.dead_ends;
        if (kids.empty() || (episode && depth % episode == 0))
        {
            cur = init;
            depth = 0;
            continue;
        }
        rng += 0x9e3779b97f4a7c15ULL;
        const u64 r = hash::mix64(rng);
        cur = kids[static_cast<usize>((static_cast<u128>(r) * kids.size()) >> 64)];
        if (task.is_goal(cur))
            ++st.goals;
    }
    return st;
}
}  // namespace mymyr::rl
