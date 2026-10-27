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
struct SingleTask
{
    const Task& value;
    [[nodiscard]] u32 size() const noexcept { return 1; }
    [[nodiscard]] const char* noun() const noexcept { return "task"; }
    [[nodiscard]] const Task* task(u32) const noexcept { return &value; }
    [[nodiscard]] const Task& task_of(const i32*, u64) const noexcept { return value; }
    [[nodiscard]] u32 numeric_words() const noexcept { return value.numeric_words(); }
    [[nodiscard]] u32 derived_words() const noexcept
    {
        const auto& atoms = value.atoms();
        return bits::words_for(atoms.frozen() ? atoms.derived_slots() : atoms.max_derived_slots());
    }
    [[nodiscard]] u32 label_width() const noexcept { return max_label_width(value); }
    void check_task_ids(const i32* ids, u64 rows) const
    {
        for (u64 i = 0; ids && i < rows; ++i)
            if (ids[i] != 0)
                throw std::invalid_argument("mymyr: expand: a single task requires task id zero");
    }
};

/// Frozen instances require their full width; missing words of a lazy instance are zero. With `content`, reject
/// bits at slots the instance does not have within its last word. Words past the instance's own are padding and
/// are not read, so a row costs its instance's width.
template<class Table>
void check_rows(const Table& table, StateBatchView in, const i32* ids, bool content, const char* what)
{
    for (u64 i = 0; i < in.rows; ++i)
    {
        const Task& t = table.task_of(ids, i);
        if (t.atoms().frozen() && in.words < t.words())
            throw std::invalid_argument(std::string("mymyr: ") + what + ": state rows of " + std::to_string(in.words) +
                                        " atom words; the states of instance " + std::to_string(ids ? ids[i] : 0) +
                                        " have " + std::to_string(t.words()));
        const u32 limit = t.atoms().fluent_slots();
        if (content && (limit & 63) && (limit >> 6) < in.words && (in.row(i)[limit >> 6] >> (limit & 63)))
            throw std::invalid_argument(std::string("mymyr: ") + what + ": state row " + std::to_string(i) +
                                        " sets atom slots its instance does not have (a state of another task?)");
    }
}

template<class Table>
void check_numeric(const Table& suite, u32 numeric_words, const char* what)
{
    if (numeric_words != suite.numeric_words())
        throw std::invalid_argument(std::string("mymyr: expand: ") + what + " carry " + std::to_string(numeric_words) +
                                    " numeric words per row, the " + suite.noun() + " has " +
                                    std::to_string(suite.numeric_words()));
}

template<class Table>
void check_batch(const Table& table, StateBatchView in, const i32* ids, const ExpandOptions& opt)
{
    if (in.rows && !in.data)
        throw std::invalid_argument("mymyr: expand: null state buffer");
    if (in.stride && in.stride < static_cast<u64>(in.words) + in.numeric_words)
        throw std::invalid_argument("mymyr: expand: row stride smaller than the row width");
    check_numeric(table, in.numeric_words, "the states");
    if (in.rows > k_max_rows)
        throw std::invalid_argument("mymyr: expand: more than 2^31 - 1 states in one batch");
    if (!ids && in.rows && table.size() > 1)
        throw std::invalid_argument(std::string("mymyr: expand: a batch over a ") + table.noun() + " of " +
                                    std::to_string(table.size()) + " instances needs task ids");
    table.check_task_ids(ids, in.rows);
    check_rows(table, in, ids, opt.validate, "expand");
}

template<class Table>
void check_out(const Table& table, const Expansion& out)
{
    if (out.binding && out.label_width < table.label_width())
        throw std::invalid_argument("mymyr: expand: label width " + std::to_string(out.label_width) +
                                    " is below the largest schema arity " + std::to_string(table.label_width()));
    if (out.capacity > k_max_rows)
        throw std::invalid_argument("mymyr: expand: capacity above 2^31 - 1 rows");
    if (out.succ && out.words == 0 && out.capacity)
        throw std::invalid_argument("mymyr: expand: successor rows of zero words");
    if (out.goal && !out.succ)
        throw std::invalid_argument("mymyr: expand: goal flags need the successor rows");
    if (out.succ)
        check_numeric(table, out.numeric_words, "the successor rows");
    if (out.derived && !out.succ)
        throw std::invalid_argument("mymyr: expand: derived words need the successor rows");
    if (out.derived && out.derived_words < table.derived_words())
        throw std::invalid_argument("mymyr: expand: derived rows of " + std::to_string(out.derived_words) +
                                    " words; the " + table.noun() + " has instances of " +
                                    std::to_string(table.derived_words()));
}

/// Successor generators of a suite's instances for one thread, leased on first use per instance and returned with the
/// object.
template<class Table>
class Generators
{
public:
    explicit Generators(const Table& table) : m_table(table), m_ws(table.size()), m_succ(table.size(), nullptr) {}
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
    const Table& m_table;
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

/// hash::state of `count` rows of W atom words (row stride RW, the instance's nn numeric words after the W atom
/// words), W fixed. hash::words takes a pair step at every even position below the trimmed length (a single last word
/// is a pair step with a zero partner), so the chain runs over all W words and keeps a step only below the trimmed
/// length: no data-dependent branches, which on rows of varying trimmed length would mispredict.
template<u32 W>
void hash_fixed(const u64* rows, u64 count, u32 RW, u32 nn, u64* out)
{
    for (u64 j = 0; j < count; ++j)
    {
        const u64* r = rows + j * RW;
        u32 n = 0;
        for (u32 i = 0; i < W; ++i)
            n = r[i] ? i + 1 : n;
        u64 h = (nn ? hash::words(r + W, nn) : hash::k_seed) ^ (static_cast<u64>(n) * hash::k_p1);
        for (u32 i = 0; i < W; i += 2)
        {
            const u64 step = hash::mulfold(h ^ r[i] ^ hash::k_p1, (i + 1 < W ? r[i + 1] : 0) ^ hash::k_p2);
            h = i < n ? step : h;
        }
        out[j] = hash::mix64(h);
    }
}

/// hash::state of `count` successor rows [W atom words | NN numeric words] of an instance with nn numeric words.
void hash_successors(const u64* rows, u64 count, u32 W, u32 NN, u32 nn, u64* out)
{
    const u32 RW = W + NN;
    switch (W)
    {
        case 1: hash_fixed<1>(rows, count, RW, nn, out); return;
        case 2: hash_fixed<2>(rows, count, RW, nn, out); return;
        case 3: hash_fixed<3>(rows, count, RW, nn, out); return;
        case 4: hash_fixed<4>(rows, count, RW, nn, out); return;
        case 5: hash_fixed<5>(rows, count, RW, nn, out); return;
        case 6: hash_fixed<6>(rows, count, RW, nn, out); return;
        case 7: hash_fixed<7>(rows, count, RW, nn, out); return;
        case 8: hash_fixed<8>(rows, count, RW, nn, out); return;
        case 9: hash_fixed<9>(rows, count, RW, nn, out); return;
        case 10: hash_fixed<10>(rows, count, RW, nn, out); return;
        case 11: hash_fixed<11>(rows, count, RW, nn, out); return;
        case 12: hash_fixed<12>(rows, count, RW, nn, out); return;
        case 13: hash_fixed<13>(rows, count, RW, nn, out); return;
        case 14: hash_fixed<14>(rows, count, RW, nn, out); return;
        case 15: hash_fixed<15>(rows, count, RW, nn, out); return;
        case 16: hash_fixed<16>(rows, count, RW, nn, out); return;
        default:
            for (u64 j = 0; j < count; ++j)
            {
                const u64* row = rows + j * RW;
                out[j] = hash::state(row, W, nn ? row + W : nullptr, nn);
            }
    }
}

/// Whether the successor of cur (nw words; numeric words num, nn of them) under d equals cur: every added slot is
/// already set, every deleted slot that is not re-added is already clear, and the numeric words are bitwise equal.
inline bool is_noop(const u64* cur, u32 nw, const u64* num, u32 nn, const Delta& d)
{
    auto set = [&](u32 s) { return (s >> 6) < nw && ((cur[s >> 6] >> (s & 63)) & 1); };
    for (SlotId a : d.add)
        if (!set(a.v))
            return false;
    for (SlotId x : d.del)
        if (set(x.v) && std::none_of(d.add.begin(), d.add.end(), [&](SlotId a) { return a.v == x.v; }))
            return false;
    return d.nnum == nn && (!nn || std::equal(d.num, d.num + nn, num));
}

/// Direct sink: writes into the caller's buffers, rows [0, capacity). With `flags` (drop_noops), the no-op flag of
/// every row of the parent goes there, rows past the capacity included, and the caller compacts the parent's rows.
struct DirectSink
{
    Expansion& out;
    const u64* cur;
    u32 nw;       // trimmed width of cur
    i32 parent;   // current state
    u64 next;     // next flat row
    u32 L;
    std::vector<u64>& tmp;
    const u64* num;
    u32 nn;
    u64 first;
    bool want_hash;  // hash row by row (otherwise not wanted, or hashed by the caller over the stored rows)
    std::vector<u8>* flags = nullptr;

    void put(u32 schema, const ObjectId* b, u32 arity, const Delta& d)
    {
        const u64 j = next++;
        const u32 W = out.words, RW = W + out.numeric_words;
        if (flags)
            flags->push_back(static_cast<u8>(is_noop(cur, nw, num, nn, d)));
        if (j < out.capacity)
        {
            bool fits = true;
            if (out.succ)
            {
                u64* dst = out.succ + j * RW;
                if (nw > W || !write_successor(dst, W, out.numeric_words, cur, nw, d))
                {
                    fits = false;
                    out.words_needed = std::max(out.words_needed, exact_width(cur, nw, d, tmp));
                    std::memset(dst, 0, static_cast<usize>(RW) * sizeof(u64));
                }
            }
            if (want_hash)
            {
                if (out.succ && fits)
                    out.succ_hash[j] = hash::state(out.succ + j * RW, W, d.num, d.nnum);
                else
                {
                    const u32 size = exact_width(cur, nw, d, tmp);  // grows tmp: read its data after
                    out.succ_hash[j] = hash::state(tmp.data(), size, d.num, d.nnum);
                }
            }
            if (out.noop && !flags)
                out.noop[j] = static_cast<u8>(is_noop(cur, nw, num, nn, d));
            if (out.index)
                out.index[j] = static_cast<i32>(j - first);
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
        else if (out.succ && (nw > W || !fits_width(W, d)))
            out.words_needed = std::max(out.words_needed, exact_width(cur, nw, d, tmp));
    }

    /// drop_noops over this parent's rows [first, next): moves the kept rows of every requested array (those within
    /// the capacity) to the front, writes their no-op flags and returns the kept count. A parent whose rows are all
    /// no-ops keeps every row.
    u64 compact()
    {
        const std::vector<u8>& f = *flags;
        const u64 n = f.size();
        const u64 effective = static_cast<u64>(std::count(f.begin(), f.end(), u8{0}));
        const u64 C = out.capacity;
        if (effective == 0)
        {
            if (out.noop)
                for (u64 k = 0; k < n && first + k < C; ++k)
                    out.noop[first + k] = 1;
            return n;
        }
        const u32 RW = out.words + out.numeric_words;
        u64 kept = 0;
        for (u64 k = 0; k < n; ++k)
        {
            if (f[k])
                continue;
            const u64 src = first + k, dst = first + kept++;
            if (src >= C)
                break;  // generated > capacity: the remaining kept rows were not stored
            if (out.noop)
                out.noop[dst] = 0;
            if (src == dst)
                continue;
            if (out.succ)
                std::copy_n(out.succ + src * RW, RW, out.succ + dst * RW);
            if (out.schema)
                out.schema[dst] = out.schema[src];
            if (out.binding)
                std::copy_n(out.binding + src * L, L, out.binding + dst * L);
            if (out.succ_hash && want_hash)
                out.succ_hash[dst] = out.succ_hash[src];
            if (out.index)
                out.index[dst] = out.index[src];
        }
        return effective;
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
    bool want_parent, want_schema, want_binding;
    const u64* cur;
    u32 nw;
    i32 state;
    u32 W, NN, L;
    u32& words_needed;
    LineVector<u64>& tmp;
    u64 rows = 0;
    LineVector<u64>& hashes;
    LineVector<u8>& noops;
    LineVector<i32>& indices;
    bool want_hash, want_noop, want_index;
    const u64* num;
    u32 nn;

    void put(u32 s, const ObjectId* b, u32 arity, const Delta& d)
    {
        const usize at = succ.size();
        succ.resize(at + W + NN);
        const bool fits = nw <= W && write_successor(succ.data() + at, W, NN, cur, nw, d);
        if (!fits)
        {
            words_needed = std::max(words_needed, exact_width(cur, nw, d, tmp));
            std::memset(succ.data() + at, 0, static_cast<usize>(W + NN) * sizeof(u64));
        }
        if (want_hash)
        {
            if (fits)
                hashes.push_back(hash::state(succ.data() + at, W, d.num, d.nnum));
            else
            {
                const u32 size = exact_width(cur, nw, d, tmp);  // grows tmp: read its data after
                hashes.push_back(hash::state(tmp.data(), size, d.num, d.nnum));
            }
        }
        if (want_noop)
            noops.push_back(static_cast<u8>(is_noop(cur, nw, num, nn, d)));
        if (want_index)
            indices.push_back(static_cast<i32>(rows));
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

    /// Stable compaction of this parent's suffix; an all-no-op suffix keeps every row.
    void drop_noops()
    {
        const usize first = noops.size() - rows;
        if (std::find(noops.begin() + first, noops.end(), u8{0}) == noops.end())
            return;
        u64 kept = 0;
        auto move_row = [&](auto& v, u32 width, bool wanted, u64 src)
        {
            if (!wanted || !width)
                return;
            const usize begin = v.size() - rows * width;
            for (u32 w = 0; w < width; ++w)
                v[begin + kept * width + w] = v[begin + src * width + w];
        };
        for (u64 j = 0; j < rows; ++j)
            if (!noops[first + j])
            {
                move_row(succ, W + NN, true, j);
                move_row(parent, 1, want_parent, j);
                move_row(schema, 1, want_schema, j);
                move_row(binding, L, want_binding, j);
                move_row(hashes, 1, want_hash, j);
                move_row(indices, 1, want_index, j);
                noops[first + kept] = 0;
                ++kept;
            }
        auto shrink = [&](auto& v, u32 width, bool wanted)
        {
            if (wanted)
                v.resize(v.size() - (rows - kept) * width);
        };
        shrink(succ, W + NN, true);
        shrink(parent, 1, want_parent);
        shrink(schema, 1, want_schema);
        shrink(binding, L, want_binding);
        shrink(hashes, 1, want_hash);
        shrink(indices, 1, want_index);
        noops.resize(first + kept);
        rows = kept;
    }
};

void write_offsets(i32* offsets, u64 i, u64 value)
{
    if (value > k_max_rows)
        throw std::length_error("mymyr: expand: more than 2^31 - 1 successors in one batch");
    offsets[i] = static_cast<i32>(value);
}

/// The goal flag of a row of instance `task` with W atom words (at least the task's) and the table's NN numeric words
/// (the task reads its own words of each).
inline u8 goal_of(const Task& task, const u64* s, u32 W, u32 NN)
{
    const u32 nn = task.numeric_words();
    return static_cast<u8>(task.is_goal(StateView{s, std::min(W, task.words()), NN ? s + W : nullptr, nn}));
}

/// The goal flag and/or derived words of a successor row: one axiom evaluation serves both. `derived` (dw words, zero
/// past the instance's own) is null when not wanted; `goal` likewise.
inline void row_info(const Task& task, const u64* s, u32 W, u32 NN, u8* goal, u64* derived, u32 dw)
{
    if (derived)
        std::fill_n(derived, dw, u64{0});
    if (!derived || !task.atoms().max_derived_slots())  // no derived atoms to write: the goal test alone
    {
        if (goal)
            *goal = goal_of(task, s, W, NN);
        return;
    }
    const StateView view{s, std::min(W, task.words()), NN ? s + W : nullptr, task.numeric_words()};
    const WorkspaceLease ws = task.workspace();
    Successors& succ = ws->successors();
    succ.prepare(view);
    const u32 own = bits::words_for(task.atoms().derived_slots());
    if (own > dw)
        throw std::invalid_argument("mymyr: expand: derived output narrower than the successor's derived slots");
    std::copy_n(ws->engine().derived(), own, derived);
    if (goal)
        *goal = static_cast<u8>(succ.goal_holds());
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
        LineVector<u64> derived, hashes;
        LineVector<u8> noops;
        LineVector<i32> indices;
        LineVector<u64> tmp;
        LineVector<Chunk> chunks;
        u32 words_needed = 0;
    };
    std::vector<Member> members;
    std::vector<u64> counts;  // per state
    std::vector<u64> generated;
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

template<class Table>
void expand_impl(const Table& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt)
{
    check_batch(table, in, ids, opt);
    check_out(table, out);
    if (opt.drop_noops && out.capacity && !out.index)
        throw std::invalid_argument("mymyr: expand: drop_noops requires index");
    out.total = 0;
    out.generated = 0;
    out.words_needed = 0;
    Generators gens(table);
    std::vector<u64> tmp;
    std::vector<u8> flags;
    std::vector<u64> generated;  // per state, for the offsets of an overflow with drop_noops
    if (opt.drop_noops && out.offsets)
        generated.resize(in.rows);
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
        const u32 nw = bits::trimmed_size(row, std::min(in.words, task.words()));
        // Stored rows of a fixed-slot instance always fit: hash them in one pass after the parent.
        const bool defer_hash = out.succ_hash && out.succ && task.atoms().frozen() && out.words >= task.words();
        flags.clear();
        DirectSink sink{out, row, nw, static_cast<i32>(i), next, L, tmp, nn ? row + in.words : nullptr, nn, next,
                        out.succ_hash != nullptr && !defer_hash, opt.drop_noops ? &flags : nullptr};
        const u64 n = expand_state(gens.of(inst), row, nw, nn ? row + in.words : nullptr, nn, opt, sink);
        out.generated += n;
        if (!generated.empty())
            generated[i] = n;
        if (opt.drop_noops)
            sink.next = next + sink.compact();
        const u64 stored = sink.next < out.capacity ? sink.next : std::max(out.capacity, next);
        if (defer_hash && next < stored)
            hash_successors(out.succ + next * RW, stored - next, out.words, out.numeric_words, nn, out.succ_hash + next);
        if (out.goal || out.derived)
            for (u64 j = next; j < stored; ++j)
                row_info(task, out.succ + j * RW, out.words, NN, out.goal ? out.goal + j : nullptr,
                         out.derived ? out.derived + j * out.derived_words : nullptr, out.derived_words);
        next = sink.next;
        if (out.offsets)
            write_offsets(out.offsets, i + 1, next);
    }
    out.total = next;
    if (opt.drop_noops && out.generated > out.capacity)
    {
        // An overflow with drop_noops reports the counts before dropping, as the device does (it keeps no no-op
        // flags past the capacity): offsets[N] == total == generated.
        out.total = out.generated;
        u64 at = 0;
        for (u64 i = 0; i < generated.size(); ++i)
            write_offsets(out.offsets, i + 1, at += generated[i]);
    }
}

template<class Table>
void expand_impl(const Table& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt,
            ThreadPool& pool, ExpandScratch& scratch)
{
    const u32 T = pool.size();
    if (T == 1 || in.rows < 2)
    {
        expand_impl(table, in, ids, out, opt);
        return;
    }
    check_batch(table, in, ids, opt);
    check_out(table, out);
    if (opt.drop_noops && out.capacity && !out.index)
        throw std::invalid_argument("mymyr: expand: drop_noops requires index");
    out.total = 0;
    out.generated = 0;
    out.words_needed = 0;
    ExpandScratch::Impl& S = scratch.impl();
    if (S.members.size() < T)
        S.members.resize(T);
    S.counts.resize(in.rows);
    S.generated.resize(in.rows);
    // Chunks small enough to balance (several per member), large enough to amortize the claim.
    const u64 chunk = std::clamp<u64>(in.rows / (static_cast<u64>(T) * 8), 1, 256);
    const u64 nchunks = (in.rows + chunk - 1) / chunk;
    S.next_chunk.store(0, std::memory_order_relaxed);
    const u32 W = out.words, L = out.label_width, NN = in.numeric_words, RW = W + out.numeric_words;
    const bool want_goal = out.goal != nullptr, want_derived = out.derived != nullptr;
    const u32 DW = out.derived_words;

    pool.run(
        [&](u32 t)
        {
            ExpandScratch::Impl::Member& m = S.members[t];
            m.succ.clear();
            m.parent.clear();
            m.schema.clear();
            m.binding.clear();
            m.goal.clear();
            m.derived.clear();
            m.hashes.clear();
            m.noops.clear();
            m.indices.clear();
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
                    const u32 nw = bits::trimmed_size(row, std::min(in.words, task.words()));
                    const bool defer_hash = out.succ_hash && task.atoms().frozen() && W >= task.words();
                    ScratchSink sink{m.succ, m.parent, m.schema, m.binding, true, out.schema != nullptr,
                                     out.binding != nullptr, row, nw, static_cast<i32>(i), W, out.numeric_words, L,
                                     m.words_needed, m.tmp, 0, m.hashes, m.noops, m.indices,
                                     out.succ_hash != nullptr && !defer_hash, out.noop != nullptr || opt.drop_noops,
                                     out.index != nullptr, nn ? row + in.words : nullptr, nn};
                    const u64 before = rows_before;
                    S.generated[i] = expand_state(gens.of(inst), row, nw, nn ? row + in.words : nullptr, nn, opt, sink);
                    if (opt.drop_noops)
                        sink.drop_noops();
                    if (defer_hash)  // the kept rows only
                    {
                        const usize at = m.hashes.size();
                        m.hashes.resize(at + sink.rows);
                        hash_successors(m.succ.data() + (rec.first_row + before) * RW, sink.rows,
                                        W, out.numeric_words, nn, m.hashes.data() + at);
                    }
                    S.counts[i] = sink.rows;
                    rows_before += sink.rows;
                    if (want_goal || want_derived)
                        for (u64 j = before; j < rows_before; ++j)
                        {
                            u8 g = 0;
                            u64* d = nullptr;
                            if (want_derived)
                            {
                                m.derived.resize(m.derived.size() + DW);
                                d = m.derived.data() + m.derived.size() - DW;
                            }
                            row_info(task, m.succ.data() + (rec.first_row + j) * RW, W, NN, want_goal ? &g : nullptr, d, DW);
                            if (want_goal)
                                m.goal.push_back(g);
                        }
                }
                rec.rows = rows_before;
                m.chunks.push_back(rec);
            }
        });

    // offsets (exclusive prefix sums of the counts); an overflow with drop_noops reports the counts before dropping
    for (u64 i = 0; i < in.rows; ++i)
        out.generated += S.generated[i];
    const bool pre_drop = opt.drop_noops && out.generated > out.capacity;
    u64 total = 0, generated = 0;
    if (out.offsets)
        out.offsets[0] = 0;
    for (u64 i = 0; i < in.rows; ++i)
    {
        const u64 c = S.counts[i];
        S.counts[i] = total;  // now: first flat row of state i
        total += c;
        generated += S.generated[i];
        if (out.offsets)
            write_offsets(out.offsets, i + 1, pre_drop ? generated : total);
    }
    out.total = pre_drop ? out.generated : total;
    for (u32 t = 0; out.succ && t < T; ++t)  // the width bounds stored rows only
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
                if (out.derived)
                    std::memcpy(out.derived + dst * DW, m.derived.data() + c.first_row * DW, n * DW * sizeof(u64));
                if (out.succ_hash)
                    std::copy_n(m.hashes.data() + c.first_row, n, out.succ_hash + dst);
                if (out.noop)
                    std::copy_n(m.noops.data() + c.first_row, n, out.noop + dst);
                if (out.index)
                    std::copy_n(m.indices.data() + c.first_row, n, out.index + dst);
            }
        });
}

void expand(const TaskSuite& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt)
{
    expand_impl(table, in, ids, out, opt);
}

void expand(const TaskSuite& table, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt,
            ThreadPool& pool, ExpandScratch& scratch)
{
    expand_impl(table, in, ids, out, opt, pool, scratch);
}

void expand(const Task& task, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt)
{
    expand_impl(SingleTask{task}, in, ids, out, opt);
}

void expand(const Task& task, StateBatchView in, const i32* ids, Expansion& out, const ExpandOptions& opt,
            ThreadPool& pool, ExpandScratch& scratch)
{
    expand_impl(SingleTask{task}, in, ids, out, opt, pool, scratch);
}

void pad(const Expansion& flat, u64 rows, PaddedExpansion& out)
{
    if (!flat.offsets)
        throw std::invalid_argument("mymyr: pad: the flat expansion has no offsets");
    const u32 K = out.k, W = out.words, L = out.label_width, NN = out.numeric_words;
    const u32 RW = W + NN, FW = flat.words + flat.numeric_words;
    if (out.succ && flat.succ && W < flat.words)
        throw std::invalid_argument("mymyr: pad: padded rows narrower than the flat rows");
    if (out.succ && flat.succ && NN != flat.numeric_words)
        throw std::invalid_argument("mymyr: pad: padded and flat rows carry different numeric words");
    if (out.binding && flat.binding && L < flat.label_width)
        throw std::invalid_argument("mymyr: pad: padded label width below the flat label width");
    if (out.derived && flat.derived && out.derived_words < flat.derived_words)
        throw std::invalid_argument("mymyr: pad: padded derived rows narrower than the flat rows");
    out.overflow = flat.overflow();
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
            if (out.canonical)
                out.canonical[at] = valid && flat.index ? flat.index[j] : -1;
            if (out.succ_hash)
                out.succ_hash[at] = valid && flat.succ_hash ? flat.succ_hash[j] : 0;
            if (out.noop)
                out.noop[at] = valid && flat.noop ? flat.noop[j] : 0;
            if (out.mask)
                out.mask[at] = valid ? 1 : 0;
            if (out.succ)
            {
                u64* dst = out.succ + at * RW;
                const u64* src = valid && flat.succ ? flat.succ + j * FW : nullptr;
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
            if (out.derived)
            {
                u64* dst = out.derived + at * out.derived_words;
                const u64* src = valid && flat.derived ? flat.derived + j * flat.derived_words : nullptr;
                for (u32 w = 0; w < out.derived_words; ++w)
                    dst[w] = src && w < flat.derived_words ? src[w] : 0;
            }
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
    check_rows(table, in, ids, true, "is_goal");
    for (u64 i = 0; i < in.rows; ++i)
        out[i] = goal_of(table.task_of(ids, i), in.row(i), in.words, in.numeric_words);
}

u32 max_numeric_slots(const TaskSuite& suite) noexcept
{
    u32 n = 0;
    for (u32 g = 0; g < suite.size(); ++g)
        n = std::max(n, suite.task(g)->numeric_slots());
    return n;
}

void numeric_values(const TaskSuite& suite, StateBatchView in, const i32* ids, f64* out, u32 slots)
{
    check_numeric(suite, in.numeric_words, "the states");
    if (slots < max_numeric_slots(suite))
        throw std::invalid_argument("mymyr: numeric_values: " + std::to_string(slots) + " columns; the " + suite.noun() +
                                    " has instances of " + std::to_string(max_numeric_slots(suite)) + " numeric slots");
    if (!ids && in.rows && suite.size() > 1)
        throw std::invalid_argument(std::string("mymyr: numeric_values: a batch over a ") + suite.noun() + " of " +
                                    std::to_string(suite.size()) + " instances needs task ids");
    suite.check_task_ids(ids, in.rows);
    check_rows(suite, in, ids, true, "numeric_values");
    for (u64 i = 0; i < in.rows; ++i)
    {
        const Task& task = suite.task_of(ids, i);
        const u64* row = in.row(i);
        const StateView s{row, std::min(in.words, task.words()), in.numeric_words ? row + in.words : nullptr,
                          task.numeric_words()};
        f64* o = out + i * slots;
        const u32 n = task.numeric_slots();
        for (u32 k = 0; k < n; ++k)
            o[k] = task.numeric_value(s, k);
        std::fill(o + n, o + slots, std::numeric_limits<f64>::quiet_NaN());
    }
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
