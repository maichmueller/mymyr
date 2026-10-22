#include "mymyr/encoding/stream.hpp"

#include "mymyr/core/hash.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }

/// The per-row inputs the rows of a stream carry.
struct Inputs
{
    bool derived = false;        // the derived words of each row
    bool goals = false;          // the goal masks over the fluent slots
    bool derived_goals = false;  // and over the derived slots
    bool actions = false;        // the action rows of each row's state
    [[nodiscard]] friend bool operator==(const Inputs&, const Inputs&) noexcept = default;
};

[[nodiscard]] std::string describe(const Inputs& inputs)
{
    std::string out;
    const auto add = [&out](const char* what)
    {
        out += out.empty() ? "" : ", ";
        out += what;
    };
    if (inputs.derived)
        add("derived words");
    if (inputs.goals)
        add(inputs.derived_goals ? "goals with derived masks" : "goals");
    if (inputs.actions)
        add("actions");
    return out.empty() ? "no per-row inputs" : out;
}

/// Whether the words [from, width) of a row are zero.
[[nodiscard]] bool zero_tail(const u64* row, u32 width, u32 from) noexcept
{
    for (u32 w = from; w < width; ++w)
        if (row[w] != 0)
            return false;
    return true;
}

/// Copies `source` (`from` words) into `to` words: truncated, or padded with zero words.
void copy_words(u64* to, u32 width, const u64* source, u32 from)
{
    const u32 common = std::min(width, from);
    if (common)
        std::memcpy(to, source, usize{common} * sizeof(u64));
    if (width > common)
        std::memset(to + common, 0, usize{width - common} * sizeof(u64));
}

/// The rows of a stream and their per-row inputs, in fixed-width columns indexed by slot, in the layout of EncodeInput.
/// A slot holds one row; the rows of an append-only stream occupy the slots 0 .. n - 1, a mutable stream also frees
/// slots and reuses them. Not synchronized.
class RowStore
{
public:
    explicit RowStore(const FlatEncoder& encoder)
        : m_encoder(encoder), m_words(encoder.table().words()), m_numeric_words(encoder.table().numeric_words()),
          m_row_words(m_words + m_numeric_words), m_derived_words(encoder.context().max_derived_words()),
          m_label_width(encoder.table().label_width()), m_subgoals(encoder.schema().has_subgoals()),
          m_history(encoder.schema().has_history()), m_levels(encoder.schema().options().max_goal_level)
    {
    }

    [[nodiscard]] u64 live() const noexcept { return m_live; }
    [[nodiscard]] u64 slots() const noexcept { return m_slots; }
    [[nodiscard]] const Inputs& inputs() const noexcept { return m_inputs; }

    /// Checks `in` as rows to store and returns the inputs it gives. Throws std::invalid_argument.
    [[nodiscard]] Inputs check(const EncodeInput& in) const
    {
        const rl::TaskTable& table = m_encoder.table();
        const EncodingSchema& schema = m_encoder.schema();
        const u64 n = in.states.rows;
        if (in.states.numeric_words != m_numeric_words)
            invalid("the state rows have " + std::to_string(in.states.numeric_words) + " numeric words, the table's rows have " +
                    std::to_string(m_numeric_words));
        if (n && !in.states.data)
            invalid("the state buffer is null");
        if (in.states.stride && in.states.stride < u64{in.states.words} + in.states.numeric_words)
            invalid("the state row stride is smaller than its width");
        table.check_task_ids(in.task_ids, n);
        m_encoder.check_layer_inputs(in);
        Inputs given;
        if (in.derived && m_derived_words)
        {
            if (in.derived_stride < m_derived_words)
                invalid("the derived row stride is smaller than the derived words of an instance");
            given.derived = true;
        }
        if (in.goal_pos && !schema.has_goal_complete())
            invalid("goals are given, but the schema has no goal relations");
        if ((in.goal_pos == nullptr) != (in.goal_neg == nullptr))
            invalid("the goal needs both goal_pos and goal_neg");
        if ((in.goal_derived_pos == nullptr) != (in.goal_derived_neg == nullptr))
            invalid("the derived goal needs both goal_derived_pos and goal_derived_neg");
        if (in.goal_derived_pos && !in.goal_pos)
            invalid("the derived goal masks need the goal masks over the fluent slots");
        given.goals = in.goal_pos != nullptr;
        given.derived_goals = given.goals && in.goal_derived_pos != nullptr && m_derived_words != 0;
        if (const rl::Expansion* x = in.actions)
        {
            if (!schema.options().include_actions)
                invalid("actions are given, but the schema has no action relations");
            if (!x->offsets || !x->schema || !x->binding)
                invalid("the action expansion needs its offsets, schema and binding arrays");
            if (x->overflow())
                invalid("the action expansion overflowed: it has " + std::to_string(x->total) +
                        " rows for a capacity of " + std::to_string(x->capacity) + " and rows of " +
                        std::to_string(x->words) + " words (" + std::to_string(x->words_needed) + " needed)");
            if (x->label_width < m_label_width)
                invalid("the action expansion's label width " + std::to_string(x->label_width) +
                        " is below the table's largest action arity " + std::to_string(m_label_width));
            given.actions = true;
        }
        for (u64 g = 0; g < n; ++g)
        {
            const rl::TaskTable::Instance& instance = table.instance(in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0);
            if (in.states.words < instance.words)
                invalid("state " + std::to_string(g) + " has " + std::to_string(in.states.words) +
                        " words, its instance needs " + std::to_string(instance.words));
            if (!zero_tail(in.states.row(g), in.states.words, m_words))
                invalid("state " + std::to_string(g) + " sets an atom bit its instance does not have");
            if (in.goal_pos)
            {
                if (in.goal_words < instance.words)
                    invalid("the goal of state " + std::to_string(g) + " has " + std::to_string(in.goal_words) +
                            " words, its instance needs " + std::to_string(instance.words));
                const u64 at = g * in.goal_stride;
                if (!zero_tail(in.goal_pos + at, in.goal_words, m_words) || !zero_tail(in.goal_neg + at, in.goal_words, m_words))
                    invalid("the goal of state " + std::to_string(g) + " sets an atom bit its instance does not have");
            }
            if (in.goal_derived_pos)
            {
                if (in.goal_derived_words < instance.derived_words)
                    invalid("the derived goal of state " + std::to_string(g) + " has " +
                            std::to_string(in.goal_derived_words) + " words, its instance needs " +
                            std::to_string(instance.derived_words));
                const u64 at = g * in.goal_derived_stride;
                if (!zero_tail(in.goal_derived_pos + at, in.goal_derived_words, m_derived_words) ||
                    !zero_tail(in.goal_derived_neg + at, in.goal_derived_words, m_derived_words))
                    invalid("the derived goal of state " + std::to_string(g) + " sets an atom bit its instance does not have");
            }
            if (const LayerInput& l = in.subgoals; l.layers != 0)
            {
                if (l.words < instance.words || (l.derived_pos && l.derived_words < instance.derived_words))
                    invalid("the subgoal layers of state " + std::to_string(g) +
                            " are narrower than its instance needs (" + std::to_string(instance.words) + " words, " +
                            std::to_string(instance.derived_words) + " derived words)");
                if (!zero_layer_tails(l, g, l.layers))
                    invalid("the subgoal layers of state " + std::to_string(g) +
                            " set an atom bit its instance does not have");
            }
            if (const LayerInput& l = in.history.masks; l.layers != 0 || in.history.count)
            {
                const i64 steps = in.history.count ? in.history.count[g] : static_cast<i64>(l.layers);
                if (steps < 0 || steps > static_cast<i64>(l.layers))
                    invalid("the history count of state " + std::to_string(g) + " is " + std::to_string(steps) +
                            ", outside [0, " + std::to_string(l.layers) + "]");
                if (steps != 0 &&
                    (l.words < instance.words || (l.derived_pos && l.derived_words < instance.derived_words)))
                    invalid("the history steps of state " + std::to_string(g) +
                            " are narrower than its instance needs (" + std::to_string(instance.words) + " words, " +
                            std::to_string(instance.derived_words) + " derived words)");
                if (!zero_layer_tails(l, g, static_cast<u32>(steps)))
                    invalid("the history steps of state " + std::to_string(g) +
                            " set an atom bit its instance does not have");
            }
            if (const rl::Expansion* x = in.actions)
            {
                const i32 first = x->offsets[g], last = x->offsets[g + 1];
                if (first < 0 || first > last || static_cast<u64>(last) > x->capacity)
                    invalid("the action rows of state " + std::to_string(g) + " are not a slice of the expansion");
                if (x->parent)
                    for (i32 r = first; r < last; ++r)
                        if (static_cast<u64>(x->parent[r]) != g)
                            invalid("expansion row " + std::to_string(r) + " belongs to state " +
                                    std::to_string(x->parent[r]) + ", not to state " + std::to_string(g));
            }
        }
        return given;
    }

    /// Whether the first `layers` layers of row g of `l` are zero beyond the words (derived words) of the table.
    [[nodiscard]] bool zero_layer_tails(const LayerInput& l, u64 g, u32 layers) const noexcept
    {
        for (u32 k = 0; k < layers; ++k)
        {
            const u64 at = g * l.stride + u64{k} * l.words;
            if (!zero_tail(l.pos + at, l.words, m_words) || !zero_tail(l.neg + at, l.words, m_words))
                return false;
            if (l.derived_pos)
            {
                const u64 derived_at = g * l.derived_stride + u64{k} * l.derived_words;
                if (!zero_tail(l.derived_pos + derived_at, l.derived_words, m_derived_words) ||
                    !zero_tail(l.derived_neg + derived_at, l.derived_words, m_derived_words))
                    return false;
            }
        }
        return true;
    }

    /// Throws std::invalid_argument unless rows that give `given` can join the rows held.
    void check_inputs(const Inputs& given) const
    {
        if (m_live != 0 && !(given == m_inputs))
            invalid("the stream holds rows with " + describe(m_inputs) + ", these rows have " + describe(given));
    }

    /// Makes room for `rows` more rows of `given` (checked) and `action_rows` more action rows, so that storing them
    /// does not grow the columns row by row.
    void reserve(const Inputs& given, u64 rows, u64 action_rows)
    {
        adopt(given);
        const u64 slots = m_slots + rows;
        grow(m_states, slots * m_row_words);
        grow(m_task_ids, slots);
        grow(m_spans, slots);
        if (m_inputs.derived)
            grow(m_derived, slots * m_derived_words);
        if (m_inputs.goals)
        {
            grow(m_goal_pos, slots * m_words);
            grow(m_goal_neg, slots * m_words);
        }
        if (m_inputs.derived_goals)
        {
            grow(m_goal_derived_pos, slots * m_derived_words);
            grow(m_goal_derived_neg, slots * m_derived_words);
        }
        if (m_inputs.actions)
        {
            grow(m_schema, m_schema.size() + action_rows);
            grow(m_binding, (m_schema.size() + action_rows) * m_label_width);
        }
    }

    /// A free slot, for rows of `given` (checked); the first row after the stream was empty sets its inputs.
    u32 allocate(const Inputs& given)
    {
        adopt(given);
        u32 slot;
        if (!m_free.empty())
        {
            slot = m_free.back();
            m_free.pop_back();
        }
        else
        {
            if (m_slots >= std::numeric_limits<u32>::max())
                throw std::length_error("mymyr: encoding: the stream holds too many rows");
            slot = static_cast<u32>(m_slots++);
            m_states.resize(m_slots * m_row_words);
            m_task_ids.resize(m_slots);
            m_spans.resize(m_slots);
            if (m_subgoals)
            {
                m_has_layers.resize(m_slots, 0);
                m_sub_pos.resize(m_slots * m_levels * m_words);
                m_sub_neg.resize(m_slots * m_levels * m_words);
                m_sub_derived_pos.resize(m_slots * m_levels * m_derived_words);
                m_sub_derived_neg.resize(m_slots * m_levels * m_derived_words);
            }
            if (m_history)
                m_history_spans.resize(m_slots);
            if (m_inputs.derived)
                m_derived.resize(m_slots * m_derived_words);
            if (m_inputs.goals)
            {
                m_goal_pos.resize(m_slots * m_words);
                m_goal_neg.resize(m_slots * m_words);
            }
            if (m_inputs.derived_goals)
            {
                m_goal_derived_pos.resize(m_slots * m_derived_words);
                m_goal_derived_neg.resize(m_slots * m_derived_words);
            }
        }
        ++m_live;
        return slot;
    }

    /// Writes row g of `in` (checked) into `slot`; a slot that held a row gives up its action rows.
    void write(u32 slot, const EncodeInput& in, u64 g)
    {
        u64* row = &m_states[usize{slot} * m_row_words];
        copy_words(row, m_words, in.states.row(g), in.states.words);
        if (m_numeric_words)
            std::memcpy(row + m_words, in.states.row(g) + in.states.words, usize{m_numeric_words} * sizeof(u64));
        m_task_ids[slot] = in.task_ids ? in.task_ids[g] : 0;
        if (m_inputs.derived)
            copy_words(&m_derived[usize{slot} * m_derived_words], m_derived_words, in.derived + g * in.derived_stride,
                       m_derived_words);
        if (m_inputs.goals)
        {
            const u64 at = g * in.goal_stride;
            copy_words(&m_goal_pos[usize{slot} * m_words], m_words, in.goal_pos + at, in.goal_words);
            copy_words(&m_goal_neg[usize{slot} * m_words], m_words, in.goal_neg + at, in.goal_words);
        }
        if (m_inputs.derived_goals)
        {
            const u64 at = g * in.goal_derived_stride;
            copy_words(&m_goal_derived_pos[usize{slot} * m_derived_words], m_derived_words, in.goal_derived_pos + at,
                       in.goal_derived_words);
            copy_words(&m_goal_derived_neg[usize{slot} * m_derived_words], m_derived_words, in.goal_derived_neg + at,
                       in.goal_derived_words);
        }
        if (m_subgoals)
            write_subgoals(slot, in, g);
        if (m_history)
            write_history(slot, in, g);
        if (m_inputs.actions)
        {
            const rl::Expansion& x = *in.actions;
            drop_actions(slot);
            const u64 first = static_cast<u64>(x.offsets[g]);
            const u32 count = static_cast<u32>(x.offsets[g + 1] - x.offsets[g]);
            const u64 begin = m_schema.size();
            m_spans[slot] = {begin, count};
            m_schema.insert(m_schema.end(), x.schema + first, x.schema + first + count);
            m_binding.resize((begin + count) * m_label_width);
            for (u32 r = 0; r < count && m_label_width; ++r)
                std::memcpy(&m_binding[(begin + r) * m_label_width], x.binding + (first + r) * x.label_width,
                            usize{m_label_width} * sizeof(i32));
        }
    }

    /// Frees `slot`; once no row is held the stream is empty again.
    void release(u32 slot)
    {
        drop_actions(slot);
        drop_history(slot);
        mark_layers(slot, false);
        m_free.push_back(slot);
        if (--m_live == 0)
            reset();
    }

    /// Drops every row and its storage's contents (the memory is kept for the next rows).
    void reset()
    {
        m_live = m_slots = 0;
        m_inputs = {};
        m_states.clear();
        m_task_ids.clear();
        m_derived.clear();
        m_goal_pos.clear();
        m_goal_neg.clear();
        m_goal_derived_pos.clear();
        m_goal_derived_neg.clear();
        m_spans.clear();
        m_schema.clear();
        m_binding.clear();
        m_sub_pos.clear();
        m_sub_neg.clear();
        m_sub_derived_pos.clear();
        m_sub_derived_neg.clear();
        m_has_layers.clear();
        m_layered_rows = 0;
        m_history_spans.clear();
        m_history_pos.clear();
        m_history_neg.clear();
        m_history_derived_pos.clear();
        m_history_derived_neg.clear();
        m_history_dt.clear();
        m_free.clear();
        m_garbage = 0;
        m_history_garbage = 0;
    }

    /// The input of the rows in the order `order` (null: slots 0 .. n - 1, which must be all slots, none free). It
    /// points into the store, or into the gather buffers when the order is not the storage order; both are valid until
    /// the next call that changes the store or assembles.
    [[nodiscard]] EncodeInput assemble(const u32* order, u64 n)
    {
        EncodeInput in;
        in.states = {nullptr, n, m_words, m_row_words, m_numeric_words};
        if (n == 0)
            return in;
        const bool direct = order == nullptr;
        const auto gather = [&](auto& to, const auto& from, u32 width, auto*& pointer)
        {
            if (direct)
            {
                pointer = from.data();
                return;
            }
            to.resize(n * width);
            for (u64 i = 0; i < n; ++i)
                if (width)
                    std::memcpy(&to[i * width], &from[usize{order[i]} * width], usize{width} * sizeof(from[0]));
            pointer = to.data();
        };
        const u64* states = nullptr;
        gather(m_gather_states, m_states, m_row_words, states);
        in.states.data = states;
        const i32* ids = nullptr;
        gather(m_gather_ids, m_task_ids, 1, ids);
        in.task_ids = ids;
        if (m_inputs.derived)
        {
            gather(m_gather_derived, m_derived, m_derived_words, in.derived);
            in.derived_stride = m_derived_words;
        }
        if (m_inputs.goals)
        {
            gather(m_gather_goal_pos, m_goal_pos, m_words, in.goal_pos);
            gather(m_gather_goal_neg, m_goal_neg, m_words, in.goal_neg);
            in.goal_stride = in.goal_words = m_words;
        }
        if (m_inputs.derived_goals)
        {
            gather(m_gather_goal_derived_pos, m_goal_derived_pos, m_derived_words, in.goal_derived_pos);
            gather(m_gather_goal_derived_neg, m_goal_derived_neg, m_derived_words, in.goal_derived_neg);
            in.goal_derived_stride = in.goal_derived_words = m_derived_words;
        }
        if (m_subgoals && m_layered_rows != 0)  // rows without layers need no layer input
        {
            gather(m_gather_sub_pos, m_sub_pos, m_levels * m_words, in.subgoals.pos);
            gather(m_gather_sub_neg, m_sub_neg, m_levels * m_words, in.subgoals.neg);
            in.subgoals.stride = u64{m_levels} * m_words;
            in.subgoals.layers = m_levels;
            in.subgoals.words = m_words;
            if (m_derived_words)
            {
                const u32 per_row = m_levels * m_derived_words;
                gather(m_gather_sub_derived_pos, m_sub_derived_pos, per_row, in.subgoals.derived_pos);
                gather(m_gather_sub_derived_neg, m_sub_derived_neg, per_row, in.subgoals.derived_neg);
                in.subgoals.derived_stride = u64{m_levels} * m_derived_words;
                in.subgoals.derived_words = m_derived_words;
            }
        }
        if (m_history)
            assemble_history(order, n, in);
        if (m_inputs.actions)
            assemble_actions(order, n, in);
        return in;
    }

    /// Rewrites the arenas without the rows (steps) of freed or replaced slots once they outweigh the live ones.
    void compact_if_sparse()
    {
        compact_history_if_sparse();
        if (!m_inputs.actions || m_garbage < k_compact_rows || m_garbage * 2 < m_schema.size())
            return;
        std::vector<i32> schema, binding;
        schema.reserve(m_schema.size() - m_garbage);
        binding.reserve((m_schema.size() - m_garbage) * m_label_width);
        for (Span& span : m_spans)
        {
            const u64 begin = schema.size();
            schema.insert(schema.end(), m_schema.begin() + static_cast<std::ptrdiff_t>(span.begin),
                          m_schema.begin() + static_cast<std::ptrdiff_t>(span.begin + span.count));
            binding.insert(binding.end(), m_binding.begin() + static_cast<std::ptrdiff_t>(span.begin * m_label_width),
                           m_binding.begin() + static_cast<std::ptrdiff_t>((span.begin + span.count) * m_label_width));
            span.begin = begin;
        }
        m_schema = std::move(schema);
        m_binding = std::move(binding);
        m_garbage = 0;
    }

private:
    /// Records whether `slot` holds subgoal layers, and counts the rows that do.
    void mark_layers(u32 slot, bool has)
    {
        if (!m_subgoals || m_has_layers[slot] == static_cast<u8>(has))
            return;
        m_has_layers[slot] = static_cast<u8>(has);
        if (has)
            ++m_layered_rows;
        else
            --m_layered_rows;
    }

    /// Stores the subgoal layers of row g of `in` into `slot`: its layers, then zero layers up to the schema's levels.
    void write_subgoals(u32 slot, const EncodeInput& in, u64 g)
    {
        const LayerInput& l = in.subgoals;
        const u32 given = std::min(l.layers, m_levels);
        mark_layers(slot, given != 0);
        const auto store = [&](std::vector<u64>& to, const u64* from, u32 width, u32 from_width, u64 at)
        {
            u64* dst = to.data() + usize{slot} * m_levels * width;
            for (u32 k = 0; k < m_levels; ++k)
                if (k < given && from)
                    copy_words(dst + usize{k} * width, width, from + at + u64{k} * from_width, from_width);
                else
                    std::fill_n(dst + usize{k} * width, width, u64{0});
        };
        store(m_sub_pos, l.pos, m_words, l.words, g * l.stride);
        store(m_sub_neg, l.neg, m_words, l.words, g * l.stride);
        if (m_derived_words)
        {
            store(m_sub_derived_pos, l.derived_pos, m_derived_words, l.derived_words, g * l.derived_stride);
            store(m_sub_derived_neg, l.derived_neg, m_derived_words, l.derived_words, g * l.derived_stride);
        }
    }

    /// Stores the valid history steps of row g of `in` into the history arena as the slot's span.
    void write_history(u32 slot, const EncodeInput& in, u64 g)
    {
        drop_history(slot);
        const LayerInput& l = in.history.masks;
        if (l.layers == 0)
            return;
        const u32 steps = in.history.count ? static_cast<u32>(in.history.count[g]) : l.layers;
        if (steps == 0)
            return;
        const u64 begin = m_history_dt.size();
        m_history_spans[slot] = {begin, steps};
        const auto append = [&](std::vector<u64>& to, const u64* from, u32 width, u32 from_width, u64 at)
        {
            to.resize((begin + steps) * width);
            for (u32 k = 0; k < steps; ++k)
                copy_words(to.data() + (begin + k) * width, width, from + at + u64{k} * from_width, from_width);
        };
        append(m_history_pos, l.pos, m_words, l.words, g * l.stride);
        append(m_history_neg, l.neg, m_words, l.words, g * l.stride);
        if (m_derived_words)
        {
            append(m_history_derived_pos, l.derived_pos, m_derived_words, l.derived_words, g * l.derived_stride);
            append(m_history_derived_neg, l.derived_neg, m_derived_words, l.derived_words, g * l.derived_stride);
        }
        m_history_dt.insert(m_history_dt.end(), in.history.dt + g * l.layers, in.history.dt + g * l.layers + steps);
    }

    /// Gives up the history steps of `slot`.
    void drop_history(u32 slot)
    {
        if (!m_history)
            return;
        m_history_garbage += m_history_spans[slot].count;
        m_history_spans[slot] = {};
    }

    /// The history of the rows in the order `order`, padded to the longest history: one set of masks per row, its steps
    /// oldest first and zero past its count.
    void assemble_history(const u32* order, u64 n, EncodeInput& in)
    {
        u32 longest = 0;
        for (u64 i = 0; i < n; ++i)
            longest = std::max(longest, m_history_spans[order ? order[i] : i].count);
        if (longest == 0)
            return;
        m_gather_history_count.resize(n);
        m_gather_history_dt.assign(n * longest, 0);
        const auto pad = [&](std::vector<u64>& to, const std::vector<u64>& from, u32 width)
        {
            to.assign(n * longest * width, 0);
            for (u64 i = 0; i < n; ++i)
            {
                const Span span = m_history_spans[order ? order[i] : i];
                if (span.count && width)
                    std::memcpy(&to[i * longest * width], &from[span.begin * width],
                                usize{span.count} * width * sizeof(u64));
            }
        };
        pad(m_gather_history_pos, m_history_pos, m_words);
        pad(m_gather_history_neg, m_history_neg, m_words);
        if (m_derived_words)
        {
            pad(m_gather_history_derived_pos, m_history_derived_pos, m_derived_words);
            pad(m_gather_history_derived_neg, m_history_derived_neg, m_derived_words);
        }
        for (u64 i = 0; i < n; ++i)
        {
            const Span span = m_history_spans[order ? order[i] : i];
            m_gather_history_count[i] = static_cast<i32>(span.count);
            if (span.count)
                std::copy_n(m_history_dt.begin() + static_cast<std::ptrdiff_t>(span.begin), span.count,
                            m_gather_history_dt.begin() + static_cast<std::ptrdiff_t>(i * longest));
        }
        LayerInput& l = in.history.masks;
        l.pos = m_gather_history_pos.data();
        l.neg = m_gather_history_neg.data();
        l.stride = u64{longest} * m_words;
        l.layers = longest;
        l.words = m_words;
        if (m_derived_words)
        {
            l.derived_pos = m_gather_history_derived_pos.data();
            l.derived_neg = m_gather_history_derived_neg.data();
            l.derived_stride = u64{longest} * m_derived_words;
            l.derived_words = m_derived_words;
        }
        in.history.dt = m_gather_history_dt.data();
        in.history.count = m_gather_history_count.data();
    }

    void compact_history_if_sparse()
    {
        if (!m_history || m_history_garbage < k_compact_rows || m_history_garbage * 2 < m_history_dt.size())
            return;
        const auto keep = [&](std::vector<u64>& arena, u32 width)
        {
            std::vector<u64> next;
            next.reserve((m_history_dt.size() - m_history_garbage) * width);
            for (const Span& span : m_history_spans)
                next.insert(next.end(), arena.begin() + static_cast<std::ptrdiff_t>(span.begin * width),
                            arena.begin() + static_cast<std::ptrdiff_t>((span.begin + span.count) * width));
            arena = std::move(next);
        };
        keep(m_history_pos, m_words);
        keep(m_history_neg, m_words);
        if (m_derived_words)
        {
            keep(m_history_derived_pos, m_derived_words);
            keep(m_history_derived_neg, m_derived_words);
        }
        std::vector<i32> dt;
        dt.reserve(m_history_dt.size() - m_history_garbage);
        for (Span& span : m_history_spans)
        {
            const u64 begin = dt.size();
            dt.insert(dt.end(), m_history_dt.begin() + static_cast<std::ptrdiff_t>(span.begin),
                      m_history_dt.begin() + static_cast<std::ptrdiff_t>(span.begin + span.count));
            span.begin = begin;
        }
        m_history_dt = std::move(dt);
        m_history_garbage = 0;
    }

    /// The first rows after the stream was empty set its inputs.
    void adopt(const Inputs& given)
    {
        if (m_live == 0 && !(given == m_inputs))
        {
            reset();
            m_inputs = given;
        }
    }

    /// Reserves capacity for `size` elements, at least doubling it so that repeated appends stay linear.
    template <class T>
    static void grow(std::vector<T>& column, u64 size)
    {
        if (size > column.capacity())
            column.reserve(std::max<u64>(size, 2 * column.capacity()));
    }

    /// Gives up the action rows of `slot`.
    void drop_actions(u32 slot)
    {
        if (!m_inputs.actions)
            return;
        m_garbage += m_spans[slot].count;
        m_spans[slot] = {};
    }

    void assemble_actions(const u32* order, u64 n, EncodeInput& in)
    {
        static const i32 k_empty = 0;
        // the rows are one run of the arena when each row's rows follow the previous row's
        bool contiguous = order == nullptr;
        for (u64 i = 1; contiguous && i < n; ++i)
            contiguous = m_spans[i].begin == m_spans[i - 1].begin + m_spans[i - 1].count;
        m_action_offsets.resize(n + 1);
        u64 total = 0;
        for (u64 i = 0; i < n; ++i)
        {
            m_action_offsets[i] = static_cast<i32>(total);
            total += m_spans[order ? order[i] : i].count;
            if (total > rl::k_max_rows)
                throw std::length_error("mymyr: encoding: the stream holds more action rows than an expansion can");
        }
        m_action_offsets[n] = static_cast<i32>(total);
        const i32* schema = &k_empty;
        const i32* binding = &k_empty;
        if (contiguous)
        {
            const u64 begin = m_spans[0].begin;
            if (total)
            {
                schema = m_schema.data() + begin;
                binding = m_label_width ? m_binding.data() + begin * m_label_width : &k_empty;
            }
        }
        else
        {
            m_gather_schema.resize(total);
            m_gather_binding.resize(total * m_label_width);
            for (u64 i = 0; i < n; ++i)
            {
                const Span span = m_spans[order ? order[i] : i];
                const u64 to = static_cast<u64>(m_action_offsets[i]);
                std::copy_n(m_schema.begin() + static_cast<std::ptrdiff_t>(span.begin), span.count,
                            m_gather_schema.begin() + static_cast<std::ptrdiff_t>(to));
                if (m_label_width)
                    std::copy_n(m_binding.begin() + static_cast<std::ptrdiff_t>(span.begin * m_label_width),
                                span.count * m_label_width,
                                m_gather_binding.begin() + static_cast<std::ptrdiff_t>(to * m_label_width));
            }
            if (total)
            {
                schema = m_gather_schema.data();
                binding = m_label_width ? m_gather_binding.data() : &k_empty;
            }
        }
        m_expansion = {};
        m_expansion.capacity = m_expansion.total = total;
        m_expansion.label_width = m_label_width;
        m_expansion.schema = const_cast<i32*>(schema);
        m_expansion.binding = const_cast<i32*>(binding);
        m_expansion.offsets = m_action_offsets.data();
        in.actions = &m_expansion;
    }

    struct Span
    {
        u64 begin = 0;  // the first row of the slot's action rows in the arena
        u32 count = 0;
    };

    /// Rows of garbage in the action arena below which it is not compacted.
    static constexpr u64 k_compact_rows = 4096;

    const FlatEncoder& m_encoder;
    u32 m_words, m_numeric_words, m_row_words, m_derived_words, m_label_width;
    bool m_subgoals, m_history;  // the schema has subgoal / history relations: the store holds the rows' layers / steps
    u32 m_levels;                // the subgoal layers stored per row: the schema's levels
    Inputs m_inputs;
    u64 m_live = 0, m_slots = 0;
    std::vector<u32> m_free;
    std::vector<u64> m_states, m_derived, m_goal_pos, m_goal_neg, m_goal_derived_pos, m_goal_derived_neg;
    std::vector<i32> m_task_ids;
    std::vector<Span> m_spans;
    std::vector<i32> m_schema, m_binding;  // the action arena: the schema of each row, and its binding [rows, label width]
    u64 m_garbage = 0;                     // rows of the arena that no slot owns
    std::vector<u64> m_sub_pos, m_sub_neg, m_sub_derived_pos, m_sub_derived_neg;  // [slots, levels * words]
    std::vector<u8> m_has_layers;  // [slots] whether the row holds subgoal layers
    u64 m_layered_rows = 0;        // the live rows that do
    // the history arena: the steps of each slot's history one after the other (a slot's span), oldest first
    std::vector<Span> m_history_spans;
    std::vector<u64> m_history_pos, m_history_neg, m_history_derived_pos, m_history_derived_neg;  // [steps, words]
    std::vector<i32> m_history_dt;                                                                // [steps]
    u64 m_history_garbage = 0;  // steps of the arena that no slot owns
    // the buffers of a flush
    std::vector<u64> m_gather_states, m_gather_derived, m_gather_goal_pos, m_gather_goal_neg, m_gather_goal_derived_pos,
        m_gather_goal_derived_neg;
    std::vector<u64> m_gather_sub_pos, m_gather_sub_neg, m_gather_sub_derived_pos, m_gather_sub_derived_neg;
    std::vector<u64> m_gather_history_pos, m_gather_history_neg, m_gather_history_derived_pos,
        m_gather_history_derived_neg;
    std::vector<i32> m_gather_history_dt, m_gather_history_count;
    std::vector<i32> m_gather_ids, m_gather_schema, m_gather_binding, m_action_offsets;
    rl::Expansion m_expansion;
};

/// The action rows of the states of `rows` (they have been checked).
[[nodiscard]] u64 action_rows_of(const EncodeInput& rows) noexcept
{
    return rows.actions ? static_cast<u64>(rows.actions->offsets[rows.states.rows] - rows.actions->offsets[0]) : 0;
}

[[nodiscard]] const FlatEncoder& required(const std::shared_ptr<const FlatEncoder>& encoder)
{
    if (!encoder)
        invalid("a stream needs an encoder");
    if (encoder->schema().has_horizon())
        invalid("a stream does not encode horizons: a horizon batch is built whole and encoded in one call");
    return *encoder;
}

/// The batch of `in`, and for the capacity form its sizes. The working memory is the calling thread's, kept between
/// calls, so that the first flush of a stream on a thread that has encoded before does not grow it again.
std::shared_ptr<const rl::ArrayBundle> encode_rows(const FlatEncoder& encoder, const EncodeInput& in, ThreadPool& pool,
                                                   const BatchShape* capacities, FlatSizes* sizes,
                                                   const StaticShape* static_capacities = nullptr,
                                                   ExtraSizes* extra = nullptr)
{
    EncodeScratch& scratch = EncodeScratch::of_this_thread();
    return capacities ? encoder.encode_capacity(in, *capacities, pool, scratch, *sizes, static_capacities, extra)
                      : encoder.encode(in, pool, scratch);
}
}  // namespace

// ------------------------------------------------------------------------------------------------ FlatStream

struct FlatStream::Impl
{
    explicit Impl(std::shared_ptr<const FlatEncoder> e) : encoder(std::move(e)), store(required(encoder)) {}

    std::shared_ptr<const FlatEncoder> encoder;
    mutable std::mutex mutex;
    RowStore store;
};

FlatStream::FlatStream(std::shared_ptr<const FlatEncoder> encoder) : m_impl(std::make_unique<Impl>(std::move(encoder)))
{
}
FlatStream::~FlatStream() = default;
FlatStream::FlatStream(FlatStream&&) noexcept = default;
FlatStream& FlatStream::operator=(FlatStream&&) noexcept = default;

const std::shared_ptr<const FlatEncoder>& FlatStream::encoder() const noexcept { return m_impl->encoder; }

void FlatStream::append(const EncodeInput& rows)
{
    std::lock_guard lock(m_impl->mutex);
    RowStore& store = m_impl->store;
    if (rows.states.rows == 0)
        return;
    const Inputs given = store.check(rows);
    store.check_inputs(given);
    store.reserve(given, rows.states.rows, action_rows_of(rows));
    for (u64 g = 0; g < rows.states.rows; ++g)
        store.write(store.allocate(given), rows, g);
}

u64 FlatStream::size() const
{
    std::lock_guard lock(m_impl->mutex);
    return m_impl->store.live();
}

void FlatStream::clear()
{
    std::lock_guard lock(m_impl->mutex);
    m_impl->store.reset();
}

std::shared_ptr<const rl::ArrayBundle> FlatStream::flush(ThreadPool& pool)
{
    std::lock_guard lock(m_impl->mutex);
    RowStore& store = m_impl->store;
    const EncodeInput in = store.assemble(nullptr, store.live());
    auto out = encode_rows(*m_impl->encoder, in, pool, nullptr, nullptr);
    store.reset();
    return out;
}

std::shared_ptr<const rl::ArrayBundle> FlatStream::flush(ThreadPool& pool, const BatchShape& capacities, FlatSizes& sizes,
                                                         const StaticShape* static_capacities, ExtraSizes* extra)
{
    std::lock_guard lock(m_impl->mutex);
    RowStore& store = m_impl->store;
    const EncodeInput in = store.assemble(nullptr, store.live());
    auto out = encode_rows(*m_impl->encoder, in, pool, &capacities, &sizes, static_capacities, extra);
    if (!sizes.overflowed())
        store.reset();
    return out;
}

// ------------------------------------------------------------------------------------------------ FlatMutableStream

namespace
{
/// A map from an index to a slot: open addressing over a power-of-two table (linear probing, deletion by shifting
/// the following entries back, so there are no tombstones), at most 5/8 full.
class SlotMap
{
public:
    [[nodiscard]] usize size() const noexcept { return m_size; }

    [[nodiscard]] const u32* find(i64 key) const noexcept
    {
        if (m_size == 0)
            return nullptr;
        for (usize i = home(key);; i = (i + 1) & m_mask)
        {
            const Entry& e = m_entries[i];
            if (!e.used)
                return nullptr;
            if (e.key == key)
                return &e.slot;
        }
    }
    [[nodiscard]] bool contains(i64 key) const noexcept { return find(key) != nullptr; }

    /// Makes room for `count` entries in all.
    void reserve(usize count)
    {
        usize capacity = std::max<usize>(m_entries.size(), k_min_capacity);
        while (count * 8 > capacity * 5)
            capacity *= 2;
        if (capacity > m_entries.size())
            rehash(capacity);
    }

    /// Adds `key`, which must not be in the map.
    void insert(i64 key, u32 slot)
    {
        reserve(m_size + 1);
        usize i = home(key);
        while (m_entries[i].used)
            i = (i + 1) & m_mask;
        m_entries[i] = {key, slot, true};
        ++m_size;
    }

    /// Removes `key`, which must be in the map.
    void erase(i64 key) noexcept
    {
        usize hole = home(key);
        while (!m_entries[hole].used || m_entries[hole].key != key)
            hole = (hole + 1) & m_mask;
        for (usize i = (hole + 1) & m_mask; m_entries[i].used; i = (i + 1) & m_mask)
        {
            // the entry moves into the hole unless its home lies cyclically in (hole, i]
            if (((i - home(m_entries[i].key)) & m_mask) >= ((i - hole) & m_mask))
            {
                m_entries[hole] = m_entries[i];
                hole = i;
            }
        }
        m_entries[hole].used = false;
        --m_size;
    }

    void clear() noexcept
    {
        for (Entry& e : m_entries)
            e.used = false;
        m_size = 0;
    }

    /// Calls `f(key, slot)` for every entry, in no particular order.
    template <class F>
    void for_each(F&& f) const
    {
        for (const Entry& e : m_entries)
            if (e.used)
                f(e.key, e.slot);
    }

private:
    struct Entry
    {
        i64 key = 0;
        u32 slot = 0;
        bool used = false;
    };

    static constexpr usize k_min_capacity = 16;

    /// Where the probe for `key` starts: eight consecutive keys start eight consecutive places (counters are the
    /// common indices, and a run of them then touches one or two cache lines), the runs are scattered by a mix.
    [[nodiscard]] usize home(i64 key) const noexcept
    {
        const u64 k = static_cast<u64>(key);
        return (hash::mix64(k >> 3) + k) & m_mask;
    }

    void rehash(usize capacity)
    {
        std::vector<Entry> old(capacity);
        old.swap(m_entries);
        m_mask = capacity - 1;
        for (const Entry& e : old)
        {
            if (!e.used)
                continue;
            usize i = home(e.key);
            while (m_entries[i].used)
                i = (i + 1) & m_mask;
            m_entries[i] = e;
        }
    }

    std::vector<Entry> m_entries;
    usize m_mask = 0;
    usize m_size = 0;
};
}  // namespace

struct FlatMutableStream::Impl
{
    explicit Impl(std::shared_ptr<const FlatEncoder> e) : encoder(std::move(e)), store(required(encoder)) {}

    /// The rows in ascending index order: `sorted` (index, slot) and the slots alone. An append keeps them current
    /// (the new rows are added, or merged in when they do not ascend above the largest index); a removal invalidates
    /// them and the next call rebuilds them from the map.
    void order()
    {
        if (order_valid)
            return;
        sorted.clear();
        sorted.reserve(slot_of.size());
        slot_of.for_each([this](i64 index, u32 slot) { sorted.emplace_back(index, slot); });
        std::sort(sorted.begin(), sorted.end(), by_index);
        slots_from(0);
        order_valid = true;
    }

    /// An append calls begin_append() before it stores its rows, track() for each stored row and end_append() after
    /// the last one.
    void begin_append()
    {
        merging = order_valid;
        order_valid = false;
        first_new = sorted.size();
        ascending = true;
    }
    void track(i64 index, u32 slot)
    {
        if (!merging)
            return;
        ascending = ascending && (sorted.empty() || index > sorted.back().first);
        sorted.emplace_back(index, slot);
    }
    void end_append()
    {
        if (!merging)
            return;
        if (!ascending)
        {
            const auto middle = sorted.begin() + static_cast<std::ptrdiff_t>(first_new);
            std::sort(middle, sorted.end(), by_index);
            std::inplace_merge(sorted.begin(), middle, sorted.end(), by_index);
        }
        slots_from(ascending ? first_new : 0);
        order_valid = true;
    }

    /// Whether the slots of `sorted` are 0, 1, 2, ... and fill the store, so that a flush reads its storage as it is.
    [[nodiscard]] bool direct() const noexcept { return identity && store.slots() == slots.size(); }

    void reset()
    {
        store.reset();
        slot_of.clear();
        sorted.clear();
        slots.clear();
        identity = order_valid = true;
    }

    std::shared_ptr<const FlatEncoder> encoder;
    mutable std::mutex mutex;
    RowStore store;
    SlotMap slot_of;
    std::vector<std::pair<i64, u32>> sorted;  // (index, slot) by index; valid iff order_valid
    std::vector<u32> slots;                   // the slot of each entry of `sorted`
    bool order_valid = true;
    bool identity = true;  // slots[i] == i for every entry

private:
    static bool by_index(const std::pair<i64, u32>& a, const std::pair<i64, u32>& b) noexcept { return a.first < b.first; }

    /// Brings `slots` and `identity` up to date for the entries of `sorted` from `from` on.
    void slots_from(usize from)
    {
        slots.resize(sorted.size());
        if (from == 0)
            identity = true;
        for (usize i = from; i < sorted.size(); ++i)
        {
            slots[i] = sorted[i].second;
            identity = identity && slots[i] == i;
        }
    }

    bool merging = false;   // an append is under way on a current order
    bool ascending = true;  // the rows it added ascend above the largest index
    usize first_new = 0;    // where they start in `sorted`
};

namespace
{
/// Throws std::invalid_argument if `indices` repeats an index.
void check_distinct(std::span<const i64> indices)
{
    // indices that ascend strictly (the usual append) are distinct
    if (std::adjacent_find(indices.begin(), indices.end(), std::greater_equal<>{}) == indices.end())
        return;
    std::vector<i64> sorted(indices.begin(), indices.end());
    std::sort(sorted.begin(), sorted.end());
    const auto repeated = std::adjacent_find(sorted.begin(), sorted.end());
    if (repeated != sorted.end())
        invalid("the index " + std::to_string(*repeated) + " is given twice");
}

void check_row_count(std::span<const i64> indices, const EncodeInput* rows)
{
    if (rows && indices.size() != rows->states.rows)
        invalid(std::to_string(indices.size()) + " indices for " + std::to_string(rows->states.rows) + " rows");
}

[[noreturn]] void not_held(i64 index)
{
    throw std::out_of_range("mymyr: encoding: the index " + std::to_string(index) + " is not in the stream");
}
}  // namespace

FlatMutableStream::FlatMutableStream(std::shared_ptr<const FlatEncoder> encoder)
    : m_impl(std::make_unique<Impl>(std::move(encoder)))
{
}
FlatMutableStream::~FlatMutableStream() = default;
FlatMutableStream::FlatMutableStream(FlatMutableStream&&) noexcept = default;
FlatMutableStream& FlatMutableStream::operator=(FlatMutableStream&&) noexcept = default;

const std::shared_ptr<const FlatEncoder>& FlatMutableStream::encoder() const noexcept { return m_impl->encoder; }

void FlatMutableStream::append(std::span<const i64> indices, const EncodeInput& rows)
{
    std::lock_guard lock(m_impl->mutex);
    Impl& s = *m_impl;
    check_row_count(indices, &rows);
    if (indices.empty())
        return;
    check_distinct(indices);
    for (const i64 index : indices)
        if (s.slot_of.contains(index))
            invalid("the index " + std::to_string(index) + " is in the stream already; update it instead");
    const Inputs given = s.store.check(rows);
    s.store.check_inputs(given);
    s.store.reserve(given, indices.size(), action_rows_of(rows));
    s.slot_of.reserve(s.slot_of.size() + indices.size());
    s.begin_append();
    for (u64 g = 0; g < indices.size(); ++g)
    {
        const u32 slot = s.store.allocate(given);
        s.store.write(slot, rows, g);
        s.slot_of.insert(indices[g], slot);
        s.track(indices[g], slot);
    }
    s.end_append();
}

void FlatMutableStream::update(std::span<const i64> indices, const EncodeInput& rows)
{
    std::lock_guard lock(m_impl->mutex);
    Impl& s = *m_impl;
    check_row_count(indices, &rows);
    if (indices.empty())
        return;
    for (const i64 index : indices)
        if (!s.slot_of.contains(index))
            not_held(index);
    check_distinct(indices);
    const Inputs given = s.store.check(rows);
    s.store.check_inputs(given);
    for (u64 g = 0; g < indices.size(); ++g)
        s.store.write(*s.slot_of.find(indices[g]), rows, g);
    s.store.compact_if_sparse();
}

void FlatMutableStream::remove(std::span<const i64> indices)
{
    std::lock_guard lock(m_impl->mutex);
    Impl& s = *m_impl;
    for (const i64 index : indices)
        if (!s.slot_of.contains(index))
            not_held(index);
    check_distinct(indices);
    if (indices.empty())
        return;
    s.order_valid = false;
    for (const i64 index : indices)
    {
        const u32 slot = *s.slot_of.find(index);
        s.slot_of.erase(index);
        s.store.release(slot);
    }
    if (s.slot_of.size() == 0)
        s.reset();
    else
        s.store.compact_if_sparse();
}

bool FlatMutableStream::contains(i64 index) const
{
    std::lock_guard lock(m_impl->mutex);
    return m_impl->slot_of.contains(index);
}

u64 FlatMutableStream::size() const
{
    std::lock_guard lock(m_impl->mutex);
    return m_impl->slot_of.size();
}

void FlatMutableStream::clear()
{
    std::lock_guard lock(m_impl->mutex);
    m_impl->reset();
}

std::vector<i64> FlatMutableStream::indices() const
{
    std::lock_guard lock(m_impl->mutex);
    m_impl->order();
    std::vector<i64> out(m_impl->sorted.size());
    for (usize i = 0; i < out.size(); ++i)
        out[i] = m_impl->sorted[i].first;
    return out;
}

std::shared_ptr<const rl::ArrayBundle> FlatMutableStream::flush(ThreadPool& pool)
{
    std::lock_guard lock(m_impl->mutex);
    Impl& s = *m_impl;
    s.order();
    const EncodeInput in = s.store.assemble(s.direct() ? nullptr : s.slots.data(), s.slots.size());
    return encode_rows(*s.encoder, in, pool, nullptr, nullptr);
}

std::shared_ptr<const rl::ArrayBundle> FlatMutableStream::flush(ThreadPool& pool, const BatchShape& capacities,
                                                                FlatSizes& sizes, const StaticShape* static_capacities,
                                                                ExtraSizes* extra)
{
    std::lock_guard lock(m_impl->mutex);
    Impl& s = *m_impl;
    s.order();
    const EncodeInput in = s.store.assemble(s.direct() ? nullptr : s.slots.data(), s.slots.size());
    return encode_rows(*s.encoder, in, pool, &capacities, &sizes, static_capacities, extra);
}
}  // namespace mymyr::encoding
