#pragma once
// Test support of the relational encoder: PDDL tasks of tests/data/pddl, seeded random walks over a task table, and
// options without goal relations.

#include "../support/dirty_blocks.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/options.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_table.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mymyr::test
{
inline TaskPtr pddl_task(const std::string& domain, const std::string& problem)
{
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/" + domain + "/";
    return Task::create(*frontend::load_task(dir + "domain.pddl", dir + problem + ".pddl"));
}

/// The options of the encoder: no goal relations.
inline encoding::FlatOptions encoder_options()
{
    encoding::FlatOptions options;
    options.goal_derivations = {};
    return options;
}

/// Rows of a batch over a table.
struct Rows
{
    std::vector<u64> data;     // [n, words + numeric_words]
    std::vector<i32> task_ids;  // [n]
    u32 words = 0;              // atom words per row
    u32 numeric_words = 0;      // numeric words per row (numeric tables)
    [[nodiscard]] u64 size() const noexcept { return task_ids.size(); }
    [[nodiscard]] rl::StateBatchView view() const noexcept
    {
        return {data.data(), size(), words, 0, numeric_words};
    }
};

/// `n` rows of one instance: the initial state, then a seeded random walk that restarts at dead ends.
inline Rows random_walk(const rl::TaskTable& table, u32 instance, u64 n, u64 seed)
{
    Rows rows;
    rows.words = table.words();
    rows.numeric_words = table.numeric_words();
    const u32 width = rows.words + rows.numeric_words;
    std::mt19937_64 rng(seed);
    const auto& init = table.instance(instance).init;
    std::vector<u64> current(init.begin(), init.end());
    const i32 id = static_cast<i32>(instance);
    for (u64 i = 0; i < n; ++i)
    {
        rows.data.insert(rows.data.end(), current.begin(), current.end());
        rows.task_ids.push_back(id);
        rl::Expansion out;
        out.words = rows.words;
        out.numeric_words = rows.numeric_words;
        out.capacity = 4096;
        std::vector<u64> succ;
        for (;;)
        {
            succ.assign(out.capacity * width, 0);
            out.succ = succ.data();
            rl::expand(table, {current.data(), 1, rows.words, 0, rows.numeric_words}, &id, out);
            if (!out.overflow())
                break;
            out.capacity = out.total;
            out.words = std::max(out.words, out.words_needed);
        }
        if (out.total == 0)
            current.assign(init.begin(), init.end());
        else
        {
            const u64 pick = rng() % out.total;
            current.assign(succ.begin() + static_cast<std::ptrdiff_t>(pick * width),
                           succ.begin() + static_cast<std::ptrdiff_t>((pick + 1) * width));
        }
    }
    return rows;
}

/// A batch of horizon graphs built by expanding roots breadth first: graph g is a root and, for each of `depth` levels,
/// the successors (in canonical order, at most `max_children` of each node) of the nodes of the level before. Node
/// order is the order of discovery, so a node's index is its row in the graph, and edge e of a graph leads to its node
/// e + 1.
struct Horizon
{
    std::vector<u64> nodes;     // [M, words]
    std::vector<i32> task_ids;  // [G]
    std::vector<i32> node_offsets{0}, edge_offsets{0}, edge_src, edge_dst, edge_schema, edge_binding;
    u32 words = 0, label_width = 0;
    [[nodiscard]] u64 graphs() const noexcept { return task_ids.size(); }
    [[nodiscard]] u64 num_nodes() const noexcept { return words ? nodes.size() / words : 0; }
    [[nodiscard]] encoding::HorizonInput view() const noexcept
    {
        encoding::HorizonInput h;
        h.nodes = {nodes.data(), num_nodes(), words, 0, 0};
        h.graphs = graphs();
        h.node_offsets = node_offsets.data();
        h.edge_offsets = edge_offsets.data();
        h.edge_src = edge_src.data();
        h.edge_dst = edge_dst.data();
        h.edge_schema = edge_schema.data();
        h.edge_binding = edge_binding.data();
        h.label_width = label_width;
        return h;
    }
    [[nodiscard]] encoding::EncodeInput input() const noexcept
    {
        encoding::EncodeInput in;
        in.horizon = view();
        in.task_ids = task_ids.data();
        return in;
    }
};

inline Horizon horizon_of(const rl::TaskTable& table, const Rows& roots, u32 depth, u64 max_children = ~u64{0})
{
    Horizon h;
    h.words = roots.words;
    h.label_width = table.label_width();
    for (u64 g = 0; g < roots.size(); ++g)
    {
        const i32 id = roots.task_ids[g];
        h.task_ids.push_back(id);
        const u64 first = h.num_nodes();
        h.nodes.insert(h.nodes.end(), roots.data.begin() + static_cast<std::ptrdiff_t>(g * roots.words),
                       roots.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * roots.words));
        u64 level_begin = first, level_end = first + 1;
        for (u32 level = 0; level < depth; ++level)
        {
            for (u64 node = level_begin; node < level_end; ++node)
            {
                const std::vector<u64> state(h.nodes.begin() + static_cast<std::ptrdiff_t>(node * h.words),
                                             h.nodes.begin() + static_cast<std::ptrdiff_t>((node + 1) * h.words));
                rl::Expansion out;
                out.words = h.words;
                out.label_width = h.label_width;
                out.capacity = 256;
                std::vector<u64> succ;
                std::vector<i32> schema, binding, offsets(2);
                for (;;)
                {
                    succ.assign(out.capacity * h.words, 0);
                    schema.assign(out.capacity, 0);
                    binding.assign(out.capacity * h.label_width, 0);
                    out.succ = succ.data();
                    out.schema = schema.data();
                    out.binding = binding.data();
                    out.offsets = offsets.data();
                    rl::expand(table, {state.data(), 1, h.words, 0, 0}, &id, out);
                    if (!out.overflow())
                        break;
                    out.capacity = out.total;
                    out.words = std::max(out.words, out.words_needed);
                }
                const u64 children = std::min<u64>(out.total, max_children);
                for (u64 c = 0; c < children; ++c)
                {
                    h.edge_src.push_back(static_cast<i32>(node - first));
                    h.edge_dst.push_back(static_cast<i32>(h.num_nodes() - first));
                    h.edge_schema.push_back(schema[c]);
                    h.edge_binding.insert(h.edge_binding.end(), binding.begin() + static_cast<std::ptrdiff_t>(c * h.label_width),
                                          binding.begin() + static_cast<std::ptrdiff_t>((c + 1) * h.label_width));
                    h.nodes.insert(h.nodes.end(), succ.begin() + static_cast<std::ptrdiff_t>(c * h.words),
                                   succ.begin() + static_cast<std::ptrdiff_t>((c + 1) * h.words));
                }
            }
            level_begin = level_end;
            level_end = h.num_nodes();
        }
        h.node_offsets.push_back(static_cast<i32>(h.num_nodes()));
        h.edge_offsets.push_back(static_cast<i32>(h.edge_src.size()));
    }
    return h;
}

/// The slot of the atom `name(objects)` (object ids) among the fluent or derived atoms of `task`; a test failure if the
/// task has none.
inline u32 atom_slot(const Task& task, AtomKind kind, const std::string& name, const std::vector<i64>& objects)
{
    const u32 slots = kind == AtomKind::Fluent ? task.atoms().fluent_slots() : task.atoms().derived_slots();
    for (u32 slot = 0; slot < slots; ++slot)
    {
        const u32* record = task.atoms().record(kind, slot);
        if (task.data().str(task.data().predicates[record[0]].name) != name)
            continue;
        std::vector<i64> found;
        for (u32 k = 0; k < task.compiled().arity[record[0]]; ++k)
            found.push_back(record[1 + k]);
        if (found == objects)
            return slot;
    }
    ADD_FAILURE() << name << " has no such atom slot";
    return 0;
}

inline void set_bit(std::vector<u64>& words, u32 slot) { words[slot >> 6] |= u64{1} << (slot & 63); }

/// A copy of `bundle` whose entry `index` of array `name` has `change(old value)` in its place.
template<class Change>
rl::ArrayBundle with_edit(const rl::ArrayBundle& bundle, const char* name, u64 index, Change change)
{
    rl::ArrayBundle::Builder out;
    for (const rl::ArrayInfo& info : bundle.arrays())
    {
        const auto* first = static_cast<const std::byte*>(bundle.data(info));
        std::vector<std::byte> bytes(first, first + info.elements() * rl::dtype_bytes(info.dtype));
        if (info.name == name)
        {
            std::byte* at = bytes.data() + index * rl::dtype_bytes(info.dtype);
            switch (info.dtype)
            {
                case rl::DType::U8: at[0] = static_cast<std::byte>(change(std::to_integer<i64>(at[0]))); break;
                case rl::DType::I32:
                {
                    i32 v;
                    std::memcpy(&v, at, 4);
                    v = static_cast<i32>(change(v));
                    std::memcpy(at, &v, 4);
                    break;
                }
                case rl::DType::I64:
                {
                    i64 v;
                    std::memcpy(&v, at, 8);
                    v = change(v);
                    std::memcpy(at, &v, 8);
                    break;
                }
                default: ADD_FAILURE() << "unexpected dtype of " << name;
            }
        }
        out.add_bytes(info.name, info.dtype, info.shape, std::move(bytes), info.words);
    }
    for (const auto& [scalar, value] : bundle.scalars())
        out.scalar(scalar, value);
    return out.build();
}

/// A copy of `bundle` whose array `name` has `delta` added to the entry at `index`.
inline rl::ArrayBundle with_change(const rl::ArrayBundle& bundle, const char* name, u64 index, i64 delta)
{
    return with_edit(bundle, name, index, [delta](i64 v) { return v + delta; });
}

/// A copy of `bundle` whose entry `index` of array `name` is `value`.
inline rl::ArrayBundle with_value(const rl::ArrayBundle& bundle, const char* name, u64 index, i64 value)
{
    return with_edit(bundle, name, index, [value](i64) { return value; });
}
}  // namespace mymyr::test
