#pragma once

#include "iw_detail.hpp"

#include "mymyr/state/flat_store.hpp"
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
        m_tree->width = width;
        m_tree->numeric_words = task.numeric_words();
        for (const auto& action : task.data().schemas)
            m_tree->label_width = std::max(m_tree->label_width, action.arity());
        if (level == TreeRecord::Transitions)
            m_rows = std::make_unique<FlatStateStore>(task.words(), 10, task.numeric_words());
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
    usize transition(u32 parent, u32 schema, const ObjectId* binding, u32 arity, StateView state,
                    u32 child, TransitionOutcome outcome)
    {
        if (!m_rows)
            return 0;
        const auto [row, fresh] = m_rows->insert(state);
        (void)fresh;
        m_parents.push_back(parent);
        m_tree->t_schema.push_back(index(schema));
        append_binding(m_tree->t_binding, binding, arity);
        m_tree->t_row.push_back(index(row.v));
        m_tree->t_child.push_back(child == ~u32{0} ? -1 : index(child));
        m_tree->t_outcome.push_back(static_cast<u8>(outcome));
        return m_tree->t_row.size() - 1;
    }

    void resolve(usize transition, u32 child, TransitionOutcome outcome)
    {
        if (!m_rows)
            return;
        m_tree->t_child[transition] = child == ~u32{0} ? -1 : index(child);
        m_tree->t_outcome[transition] = static_cast<u8>(outcome);
    }

    /// Move the engine's storage into the public layout. Padding is only rebuilt when a lazy task widened past
    /// the stored stride. Binding records are packed to the task's maximum action arity.
    void nodes(std::vector<u64>&& rows, u32 stride, std::vector<i32>&& parent, std::vector<i32>&& depth,
               std::vector<i32>&& schema, const std::vector<u64>& boff, const std::vector<ObjectId>& binding)
    {
        SearchTree& t = *m_tree;
        t.parent = std::move(parent);
        t.depth = std::move(depth);
        t.schema = std::move(schema);
        t.words = std::max(stride, m_task.words());
        t.rows = std::move(rows);
        if (t.words != stride)
        {
            std::vector<u64> padded(t.num_nodes() * (t.words + t.numeric_words), 0);
            for (usize i = 0; i < t.num_nodes(); ++i)
            {
                const u64* src = t.rows.data() + i * (stride + t.numeric_words);
                u64* dst = padded.data() + i * (t.words + t.numeric_words);
                std::copy_n(src, stride, dst);
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
        t.node_flags.resize(N, 0);
        m_goals.resize(N, -1);
        if (m_rows)
        {
            std::vector<i32> remap(m_rows->size(), -1);
            for (usize i = 0; i < N; ++i)
            {
                const u64* row = t.rows.data() + i * R;
                const auto [id, fresh] = m_rows->insert({row, t.words, row + t.words, t.numeric_words});
                if (fresh)
                    remap.resize(m_rows->size(), -1);
                remap[id.v] = index(i);
            }
            for (u32 i = 0; i < m_rows->size(); ++i)
                if (remap[i] < 0)
                {
                    remap[i] = index(t.rows.size() / R);
                    const StateView s = (*m_rows)[StateId{i}];
                    const usize start = t.rows.size();
                    t.rows.resize(start + R, 0);
                    std::copy_n(s.w, std::min(t.words, s.nw), t.rows.data() + start);
                    std::copy_n(s.num, t.numeric_words, t.rows.data() + start + t.words);
                }
            for (i32& r : t.t_row)
                r = remap[r];
            for (i32& child : t.t_child)
                if (child >= index(N))
                    child = -1;
            csr();
        }
        t.row_goal.resize(t.rows.size() / R);
        for (usize i = 0; i < t.row_goal.size(); ++i)
        {
            bool value;
            if (i < N && m_goals[i] >= 0)
                value = m_goals[i] != 0;
            else
            {
                const u64* row = t.rows.data() + i * R;
                value = goal.test(succ, {row, t.words, row + t.words, t.numeric_words}, false);
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
            dst[at + i] = index(b[i].v);
    }
    void csr()
    {
        SearchTree& t = *m_tree;
        t.offsets.assign(t.num_nodes() + 1, 0);
        for (u32 p : m_parents)
            ++t.offsets[p + 1];
        std::partial_sum(t.offsets.begin(), t.offsets.end(), t.offsets.begin());
        if (std::is_sorted(m_parents.begin(), m_parents.end()))
            return;
        std::vector<i64> cursor = t.offsets;
        std::vector<usize> order(m_parents.size());
        for (usize i = 0; i < order.size(); ++i)
            order[cursor[m_parents[i]]++] = i;
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
    std::unique_ptr<FlatStateStore> m_rows;
    std::vector<u32> m_parents;
    std::vector<i8> m_goals;
};
}  // namespace mymyr::search::detail
