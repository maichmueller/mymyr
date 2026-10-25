#pragma once

#include "iw_detail.hpp"

#include "tree_rows.hpp"
#include "mymyr/task/task.hpp"

#include <limits>
#include <numeric>

namespace mymyr::search::detail
{
/// Per-pass native recorder. Goals of popped nodes are cached, so finalization only tests rows the search did not
/// test. Transition rows are interned without preparing successors inside the generator's callback.
class TreeSink
{
public:
    TreeSink(TreeRecord level, const Task& task, u32 width)
        : m_task(task), m_tree(std::make_shared<SearchTree>())
    {
        // Every generated label is drawn from these task domains; validate the signed export range once.
        index(task.num_schemas());
        index(task.data().num_objects());
        m_tree->width = width;
        m_tree->numeric_words = task.numeric_words();
        for (const auto& action : task.data().schemas)
            m_tree->label_width = std::max(m_tree->label_width, action.arity());
        if (level == TreeRecord::Transitions)
            m_rows = std::make_unique<TreeRows>(task.words(), task.numeric_words());
    }

    static i32 index(usize n)
    {
        if (n > static_cast<usize>(std::numeric_limits<i32>::max()))
            throw std::length_error("SearchTree: more than 2^31 - 1 rows or nodes");
        return static_cast<i32>(n);
    }
    void flag(u32 id, TreeNodeFlag flag)
    {
        m_tree->node_flags.resize(std::max<usize>(m_tree->node_flags.size(), usize{id} + 1), 0);
        m_tree->node_flags[id] |= static_cast<u8>(flag);
    }
    void selected(const std::vector<u32>& before, const std::vector<u32>& after)
    {
        for (u32 id : before)
            flag(id, TreeNodeFlag::Dropped);
        for (u32 id : after)
            m_tree->node_flags[id] &= ~static_cast<u8>(TreeNodeFlag::Dropped);
    }
    void goal(u32 id, bool value)
    {
        m_goals.resize(std::max<usize>(m_goals.size(), usize{id} + 1), -1);
        m_goals[id] = value ? 1 : 0;
        if (value)
            flag(id, TreeNodeFlag::Goal);
    }
    void transition(u32 parent, u32 schema, const ObjectId* binding, u32 arity, StateView state,
                    u32 child, TransitionOutcome outcome)
    {
        if (!m_rows)
            return;
        const auto [row, fresh] = m_rows->insert(state);
        (void)fresh;
        if (m_segments.empty() || m_segments.back().parent != parent)
            m_segments.push_back({parent, m_tree->t_row.size()});
        m_tree->t_schema.push_back(static_cast<i32>(schema));
        append_binding(m_tree->t_binding, binding, arity);
        m_tree->t_row.push_back(static_cast<i32>(row));
        m_tree->t_child.push_back(child == ~u32{0} ? -1 : index(child));
        m_tree->t_outcome.push_back(static_cast<u8>(outcome));
    }

    /// Move the engine's storage into the public layout. Padding is rebuilt only when lazy-slot growth changed
    /// the row stride. Binding records are packed to the task's maximum action arity.
    void nodes(std::vector<u64>&& rows, u32 stride, std::vector<i32>&& parent, std::vector<i32>&& depth,
               std::vector<i32>&& schema, const std::vector<u64>& boff, const std::vector<ObjectId>& binding)
    {
        SearchTree& t = *m_tree;
        t.parent = std::move(parent);
        t.depth = std::move(depth);
        t.schema = std::move(schema);
        t.words = t.numeric_words ? std::max<u32>(1, m_task.words()) : m_task.words();
        t.rows = std::move(rows);
        if (t.words != stride)
        {
            std::vector<u64> padded(t.num_nodes() * (t.words + t.numeric_words), 0);
            for (usize i = 0; i < t.num_nodes(); ++i)
            {
                const u64* src = t.rows.data() + i * (stride + t.numeric_words);
                u64* dst = padded.empty() ? nullptr : padded.data() + i * (t.words + t.numeric_words);
                if (t.words)
                    std::copy_n(src, std::min(stride, t.words), dst);
                if (t.numeric_words)
                    std::copy_n(src + stride, t.numeric_words, dst + t.words);
            }
            t.rows = std::move(padded);
        }
        t.binding.reserve(t.num_nodes() * t.label_width);
        for (usize i = 0; i < t.num_nodes(); ++i)
        {
            const usize end = i + 1 < boff.size() ? boff[i + 1] : binding.size();
            append_binding(t.binding, end > boff[i] ? binding.data() + boff[i] : nullptr, static_cast<u32>(end - boff[i]));
        }
    }

    std::shared_ptr<const SearchTree> finish(Successors& succ, const GoalTest& goal)
    {
        SearchTree& t = *m_tree;
        const usize N = t.num_nodes(), R = t.words + t.numeric_words;
        usize row_count = N;
        t.node_flags.resize(N, 0);
        m_goals.resize(N, -1);
        if (m_rows)
        {
            std::vector<i32> remap(m_rows->size(), -1);
            for (usize i = 0; i < N; ++i)
            {
                const u64* row = R ? t.rows.data() + i * R : nullptr;
                const auto [id, fresh] = m_rows->insert({row, t.words, t.numeric_words ? row + t.words : nullptr, t.numeric_words});
                if (fresh)
                    remap.resize(m_rows->size(), -1);
                remap[id] = index(i);
            }
            usize next_row = N;
            row_count = N + static_cast<usize>(std::count(remap.begin(), remap.end(), -1));
            t.rows.resize(row_count * R, 0);
            for (u32 i = 0; i < m_rows->size(); ++i)
                if (remap[i] < 0)
                {
                    remap[i] = index(next_row);
                    const StateView s = (*m_rows)[i];
                    const usize start = next_row++ * R;
                    if (t.words)
                        std::copy_n(s.w, std::min(t.words, s.nw), t.rows.data() + start);
                    if (t.numeric_words)
                        std::copy_n(s.num, t.numeric_words, t.rows.data() + start + t.words);
                }
            for (i32& r : t.t_row)
                r = remap[r];
            csr();
        }
        // Ground fluent goal literals are grouped by word so a row needs only the relevant bit-mask tests.
        const bool masked = goal.task_goal() && !goal.needs_view() && m_task.compiled().num.goal.empty();
        struct Mask { u32 word; u64 positive, negative; };
        std::vector<Mask> masks;
        bool possible = !goal.statically_false();
        if (masked && possible)
        {
            std::vector<u64> positive(t.words, 0), negative(t.words, 0);
            const auto& engine = succ.engine();
            for (const auto& literal : m_task.compiled().goal.lits)
            {
                if (literal.pat.kind != plan::LitKind::Fluent)
                {
                    possible = possible && engine.holds(literal);
                    continue;
                }
                const u32 slot = engine.find(literal.pat);
                if (slot / 64 >= t.words)
                    possible = possible && !literal.pos;
                else
                    (literal.pos ? positive : negative)[slot / 64] |= u64{1} << (slot % 64);
            }
            for (u32 w = 0; w < t.words; ++w)
                if (positive[w] || negative[w])
                    masks.push_back({w, positive[w], negative[w]});
        }
        t.row_goal.resize(row_count);
        for (usize i = 0; i < t.row_goal.size(); ++i)
        {
            bool value;
            if (i < N && m_goals[i] >= 0)
                value = m_goals[i] != 0;
            else
            {
                const u64* row = R ? t.rows.data() + i * R : nullptr;
                if (masked)
                {
                    value = possible;
                    for (const auto& mask : masks)
                        if ((row[mask.word] & mask.positive) != mask.positive || (row[mask.word] & mask.negative) != 0)
                        {
                            value = false;
                            break;
                        }
                }
                else
                    value = goal.test(succ, {row, t.words, t.numeric_words ? row + t.words : nullptr, t.numeric_words}, false);
            }
            t.row_goal[i] = value ? 1 : 0;
            if (i < N && value)
                t.node_flags[i] |= static_cast<u8>(TreeNodeFlag::Goal);
        }
        return std::move(m_tree);
    }

private:
    void append_binding(std::vector<i32>& dst, const ObjectId* b, u32 arity)
    {
        const usize at = dst.size();
        dst.resize(at + m_tree->label_width, -1);
        for (u32 i = 0; i < arity; ++i)
            dst[at + i] = static_cast<i32>(b[i].v);
    }
    void csr()
    {
        SearchTree& t = *m_tree;
        t.offsets.assign(t.num_nodes() + 1, 0);
        bool ordered = true;
        for (usize i = 0; i < m_segments.size(); ++i)
        {
            const usize end = i + 1 < m_segments.size() ? m_segments[i + 1].first : t.num_transitions();
            t.offsets[m_segments[i].parent + 1] += static_cast<i64>(end - m_segments[i].first);
            ordered = ordered && (i == 0 || m_segments[i - 1].parent <= m_segments[i].parent);
        }
        std::partial_sum(t.offsets.begin(), t.offsets.end(), t.offsets.begin());
        if (ordered)
            return;
        std::vector<i64> cursor = t.offsets;
        std::vector<usize> order(t.num_transitions());
        for (usize s = 0; s < m_segments.size(); ++s)
        {
            const usize end = s + 1 < m_segments.size() ? m_segments[s + 1].first : t.num_transitions();
            for (usize i = m_segments[s].first; i < end; ++i)
                order[cursor[m_segments[s].parent]++] = i;
        }
        auto reorder = [&](auto& v, usize stride)
        {
            auto old = std::move(v);
            v.resize(old.size());
            for (usize i = 0; i < order.size(); ++i)
                std::copy_n(old.begin() + order[i] * stride, stride, v.begin() + i * stride);
        };
        reorder(t.t_schema, 1);
        reorder(t.t_binding, t.label_width);
        reorder(t.t_row, 1);
        reorder(t.t_child, 1);
        reorder(t.t_outcome, 1);
    }

    const Task& m_task;
    std::shared_ptr<SearchTree> m_tree;
    std::unique_ptr<TreeRows> m_rows;
    struct Segment { u32 parent; usize first; };
    std::vector<Segment> m_segments;
    std::vector<i8> m_goals;
};

inline u64 tree_bytes(const SearchTree& t)
{
    return t.rows.capacity() * sizeof(u64) + t.offsets.capacity() * sizeof(i64) +
           (t.parent.capacity() + t.depth.capacity() + t.schema.capacity() + t.binding.capacity() +
            t.t_schema.capacity() + t.t_binding.capacity() + t.t_row.capacity() + t.t_child.capacity()) * sizeof(i32) +
           t.node_flags.capacity() + t.t_outcome.capacity() + t.row_goal.capacity();
}

inline std::shared_ptr<const SearchTree> record_root(TreeRecord level, const Task& task, StateView root,
                                                      u32 width, Successors& succ, const GoalTest& goal)
{
    if (level == TreeRecord::None)
        return {};
    TreeSink sink(level, task, width);
    const u32 W = std::max<u32>(1, root.nw);
    std::vector<u64> rows(W + task.numeric_words(), 0);
    std::copy_n(root.w, root.nw, rows.data());
    std::copy_n(root.num, task.numeric_words(), rows.data() + W);
    sink.nodes(std::move(rows), W, {-1}, {0}, {-1}, {0}, {});
    return sink.finish(succ, goal);
}
}  // namespace mymyr::search::detail
