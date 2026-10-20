#include "mymyr/encoding/horizon_graphs.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
constexpr u64 k_max_index = static_cast<u64>(std::numeric_limits<i32>::max());
constexpr u32 k_empty = std::numeric_limits<u32>::max();

[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }

/// Expands chunks of states with rl::expand.
class ExpandSource
{
public:
    ExpandSource(const rl::TaskTable& table, u32 words, ThreadPool* pool, rl::ExpandScratch* scratch)
        : m_table(table), m_words(words), m_label_width(table.label_width()), m_pool(pool), m_scratch(scratch)
    {
    }

    void expand(const u64* rows, const i32* ids, u64 count, u64)
    {
        rl::Expansion out;
        out.words = m_words;
        out.label_width = m_label_width;
        out.capacity = std::max<u64>(64, count * 8);
        m_offsets.assign(count + 1, 0);
        const rl::StateBatchView in{rows, count, m_words, 0, 0};
        for (;;)
        {
            m_succ.resize(out.capacity * m_words);
            m_schema.resize(out.capacity);
            m_binding.resize(out.capacity * m_label_width);
            out.succ = m_succ.data();
            out.schema = m_schema.data();
            out.binding = m_binding.data();
            out.offsets = m_offsets.data();
            if (m_pool)
                rl::expand(m_table, in, ids, out, rl::ExpandOptions{}, *m_pool, *m_scratch);
            else
                rl::expand(m_table, in, ids, out);
            if (out.words_needed > m_words)
                invalid("a successor needs " + std::to_string(out.words_needed) + " state words, the rows have " +
                        std::to_string(m_words));
            if (out.total <= out.capacity)
                break;
            out.capacity = out.total;
        }
    }
    [[nodiscard]] u64 begin(u64 i) const { return static_cast<u64>(m_offsets[i]); }
    [[nodiscard]] u64 end(u64 i) const { return static_cast<u64>(m_offsets[i + 1]); }
    [[nodiscard]] const u64* row(u64 k) const { return m_succ.data() + k * m_words; }
    [[nodiscard]] i32 schema(u64 k) const { return m_schema[k]; }
    void binding(u64 k, i32* dst) const { std::memcpy(dst, m_binding.data() + k * m_label_width, m_label_width * sizeof(i32)); }
    void adopt(u64, u64) {}

private:
    const rl::TaskTable& m_table;
    u32 m_words, m_label_width;
    ThreadPool* m_pool;
    rl::ExpandScratch* m_scratch;
    std::vector<u64> m_succ;
    std::vector<i32> m_schema, m_binding, m_offsets;
};

/// Walks the transitions of a state space.
class SpaceSource
{
public:
    SpaceSource(const SuccessorGraph& space, std::span<const u32> roots) : m_space(space), m_state(roots.begin(), roots.end())
    {
    }

    void expand(const u64*, const i32*, u64, u64 first) { m_first = first; }
    [[nodiscard]] u64 begin(u64 i) const { return m_space.offsets[m_state[m_first + i]]; }
    [[nodiscard]] u64 end(u64 i) const { return m_space.offsets[m_state[m_first + i] + 1]; }
    [[nodiscard]] const u64* row(u64 edge) const
    {
        return m_space.state_words + u64{m_space.targets[edge]} * m_space.row_stride;
    }
    [[nodiscard]] i32 schema(u64 edge) const { return static_cast<i32>(m_space.schemas[edge]); }
    void binding(u64 edge, i32* dst) const
    {
        for (u32 k = 0; k < m_space.label_width; ++k)
        {
            const u32 object = m_space.bindings[edge * m_space.label_width + k];
            dst[k] = object == ~u32{0} ? -1 : static_cast<i32>(object);
        }
    }
    void adopt(u64 edge, u64 node)
    {
        assert(node == m_state.size());
        m_state.push_back(m_space.targets[edge]);
    }

private:
    const SuccessorGraph& m_space;
    std::vector<u32> m_state;  // the space state of every node, in node order
    u64 m_first = 0;
};

/// The nodes reached so far, in the order they were reached, with a hash set over (graph, row) when merging.
class NodeStore
{
public:
    NodeStore(u32 words, bool merge, u64 graphs) : m_words(words), m_merge(merge)
    {
        if (merge)
            m_table.assign(u64{1} << std::bit_width(std::max<u64>(16, graphs * 2)), k_empty);
    }

    [[nodiscard]] u64 size() const noexcept { return m_graph.size(); }
    [[nodiscard]] const u64* row(u64 node) const noexcept { return m_rows.data() + node * m_words; }
    [[nodiscard]] u32 graph(u64 node) const noexcept { return m_graph[node]; }
    [[nodiscard]] u32 local(u64 node) const noexcept { return m_local[node]; }

    /// The node of `graph` whose row is `row`, or k_empty (always, without merging).
    [[nodiscard]] u32 find(u32 graph, const u64* row, u64 hash) const noexcept
    {
        if (!m_merge)
            return k_empty;
        for (u64 at = hash & (m_table.size() - 1);; at = (at + 1) & (m_table.size() - 1))
        {
            const u32 node = m_table[at];
            if (node == k_empty)
                return k_empty;
            if (m_hash[node] == hash && m_graph[node] == graph && std::equal(row, row + m_words, this->row(node)))
                return node;
        }
    }
    [[nodiscard]] u64 hash_of(u32 graph, const u64* row) const noexcept
    {
        return m_merge ? hash::combine(hash::state_words(row, m_words), graph) : 0;
    }

    u32 add(u32 graph, u32 local, const u64* row, u64 hash)
    {
        if (m_graph.size() >= k_max_index)
            throw std::length_error("mymyr: encoding: the horizon graphs have too many nodes");
        const u32 node = static_cast<u32>(m_graph.size());
        m_rows.insert(m_rows.end(), row, row + m_words);
        m_graph.push_back(graph);
        m_local.push_back(local);
        if (m_merge)
        {
            m_hash.push_back(hash);
            if (m_graph.size() * 2 > m_table.size())
                grow();
            insert(node);
        }
        return node;
    }

private:
    void insert(u32 node) noexcept
    {
        u64 at = m_hash[node] & (m_table.size() - 1);
        while (m_table[at] != k_empty)
            at = (at + 1) & (m_table.size() - 1);
        m_table[at] = node;
    }
    void grow()
    {
        m_table.assign(m_table.size() * 2, k_empty);
        for (u32 node = 0; node + 1 < m_graph.size(); ++node)
            insert(node);
    }

    u32 m_words;
    bool m_merge;
    std::vector<u64> m_rows;
    std::vector<u32> m_graph, m_local;
    std::vector<u64> m_hash;
    std::vector<u32> m_table;
};

/// The edges in the order they were found: (graph, src, dst, schema, binding).
struct EdgeStore
{
    explicit EdgeStore(u32 label_width) : label_width(label_width) {}
    u32 label_width;
    std::vector<u32> graph;
    std::vector<i32> src, dst, schema, binding;

    void add(u32 g, i32 s, i32 d, i32 sch)
    {
        if (graph.size() >= k_max_index)
            throw std::length_error("mymyr: encoding: the horizon graphs have too many edges");
        graph.push_back(g);
        src.push_back(s);
        dst.push_back(d);
        schema.push_back(sch);
        binding.resize(binding.size() + label_width);
    }
    [[nodiscard]] i32* last_binding() { return binding.data() + binding.size() - label_width; }
};

template<class Source, class Root>
HorizonGraphs run(Source& source, std::vector<i32> task_ids, u64 graphs, const Root& root, u32 words, u32 label_width,
                  const HorizonBuild& options)
{
    NodeStore nodes(words, options.merge, graphs);
    EdgeStore edges(label_width);
    std::vector<u32> count(graphs, 1);
    HorizonGraphs out;
    out.words = words;
    out.label_width = label_width;
    out.truncated.assign(graphs, 0);
    for (u64 g = 0; g < graphs; ++g)
    {
        const u64* row = root(g);
        nodes.add(static_cast<u32>(g), 0, row, nodes.hash_of(static_cast<u32>(g), row));
    }
    if (options.max_nodes == 0)
        invalid("max_nodes must be at least 1: a graph holds its root");
    u64 lo = 0, hi = graphs;
    std::vector<i32> ids;
    for (u32 level = 0; level < options.depth && lo < hi; ++level)
    {
        for (u64 chunk = lo; chunk < hi; chunk += k_horizon_chunk)
        {
            const u64 n = std::min(k_horizon_chunk, hi - chunk);
            ids.resize(n);
            for (u64 i = 0; i < n; ++i)
                ids[i] = task_ids[nodes.graph(chunk + i)];
            source.expand(nodes.row(chunk), ids.data(), n, chunk);
            for (u64 i = 0; i < n; ++i)
            {
                const u32 g = nodes.graph(chunk + i);
                const u32 from = nodes.local(chunk + i);
                for (u64 k = source.begin(i); k < source.end(i); ++k)
                {
                    const u64* row = source.row(k);
                    const u64 hash = nodes.hash_of(g, row);
                    u32 node = nodes.find(g, row, hash);
                    u32 to;
                    if (node != k_empty)
                        to = nodes.local(node);
                    else
                    {
                        if (count[g] >= options.max_nodes)
                        {
                            if (!options.truncate)
                                invalid("graph " + std::to_string(g) + " of the horizon would have more than max_nodes = " +
                                        std::to_string(options.max_nodes) + " nodes");
                            out.truncated[g] = 1;
                            continue;
                        }
                        to = count[g]++;
                        node = nodes.add(g, to, row, hash);
                        source.adopt(k, node);
                    }
                    edges.add(g, static_cast<i32>(from), static_cast<i32>(to), source.schema(k));
                    source.binding(k, edges.last_binding());
                }
            }
        }
        lo = hi;
        hi = nodes.size();
    }

    // graph after graph: the nodes and the edges in the order they were found
    out.task_ids = std::move(task_ids);
    out.node_offsets.assign(graphs + 1, 0);
    out.edge_offsets.assign(graphs + 1, 0);
    for (u64 g = 0; g < graphs; ++g)
        out.node_offsets[g + 1] = out.node_offsets[g] + static_cast<i32>(count[g]);
    for (const u32 g : edges.graph)
        ++out.edge_offsets[g + 1];
    for (u64 g = 0; g < graphs; ++g)
        out.edge_offsets[g + 1] += out.edge_offsets[g];
    out.nodes.resize(nodes.size() * words);
    for (u64 node = 0; node < nodes.size(); ++node)
        std::copy_n(nodes.row(node), words,
                    out.nodes.begin() + (u64{static_cast<u32>(out.node_offsets[nodes.graph(node)])} + nodes.local(node)) * words);
    const u64 m = edges.graph.size();
    out.edge_src.resize(m);
    out.edge_dst.resize(m);
    out.edge_schema.resize(m);
    out.edge_binding.resize(m * label_width);
    std::vector<i32> next(out.edge_offsets.begin(), out.edge_offsets.end() - 1);
    for (u64 e = 0; e < m; ++e)
    {
        const u64 at = static_cast<u64>(next[edges.graph[e]]++);
        out.edge_src[at] = edges.src[e];
        out.edge_dst[at] = edges.dst[e];
        out.edge_schema[at] = edges.schema[e];
        std::copy_n(edges.binding.begin() + e * label_width, label_width, out.edge_binding.begin() + at * label_width);
    }
    return out;
}

std::vector<i32> checked_ids(const rl::TaskTable& table, const i32* task_ids, u64 graphs)
{
    std::vector<i32> ids(graphs, 0);
    if (task_ids)
        std::copy_n(task_ids, graphs, ids.begin());
    for (u64 g = 0; g < graphs; ++g)
        if (ids[g] < 0 || static_cast<u32>(ids[g]) >= table.size())
            invalid("task id " + std::to_string(ids[g]) + " of graph " + std::to_string(g) + " of the horizon is outside the table's " +
                    std::to_string(table.size()) + " instances");
    if (!task_ids && table.size() > 1 && graphs)
        invalid("task_ids are required for a table of several instances");
    return ids;
}
}  // namespace

HorizonGraphs build_horizon_graphs(const rl::TaskTable& table, rl::StateBatchView roots, const i32* task_ids,
                                   const HorizonBuild& options)
{
    if (roots.numeric_words != 0)
        invalid("horizon encodings do not encode numeric state rows");
    ExpandSource source(table, roots.words, nullptr, nullptr);
    return run(source, checked_ids(table, task_ids, roots.rows), roots.rows, [&roots](u64 g) { return roots.row(g); },
               roots.words, table.label_width(), options);
}

HorizonGraphs build_horizon_graphs(const rl::TaskTable& table, rl::StateBatchView roots, const i32* task_ids,
                                   const HorizonBuild& options, ThreadPool& pool, rl::ExpandScratch& scratch)
{
    if (roots.numeric_words != 0)
        invalid("horizon encodings do not encode numeric state rows");
    ExpandSource source(table, roots.words, &pool, &scratch);
    return run(source, checked_ids(table, task_ids, roots.rows), roots.rows, [&roots](u64 g) { return roots.row(g); },
               roots.words, table.label_width(), options);
}

HorizonGraphs transition_graphs(const rl::TaskTable& table, rl::StateBatchView parents, const i32* task_ids,
                                const rl::Expansion& expansion, const u64* parent_derived, u64 parent_derived_stride)
{
    if (parents.numeric_words != 0 || expansion.numeric_words != 0)
        invalid("horizon encodings do not encode numeric state rows");
    if (expansion.overflow())
        invalid("the expansion overflowed: its capacity is " + std::to_string(expansion.capacity) + " rows, it has " +
                std::to_string(expansion.total));
    if (!expansion.offsets || !expansion.succ || !expansion.schema || !expansion.binding)
        invalid("the expansion needs its succ, schema, binding and offsets");
    if (expansion.label_width < table.label_width())
        invalid("the expansion's label width " + std::to_string(expansion.label_width) + " is below the table's largest action arity " +
                std::to_string(table.label_width()));
    const u64 states = parents.rows;
    const u32 words = std::max(parents.words, expansion.words);
    HorizonGraphs out;
    out.words = words;
    out.label_width = expansion.label_width;
    out.task_ids = checked_ids(table, task_ids, states);
    const u64 successors = states ? static_cast<u64>(expansion.offsets[states]) : 0;
    if (expansion.offsets && expansion.offsets[0] != 0)
        invalid("the expansion's offsets must start at 0");
    if (successors > expansion.capacity)
        invalid("the expansion's offsets name more rows than it holds");
    if (states + successors > k_max_index)
        throw std::length_error("mymyr: encoding: the transition graphs have too many nodes");
    out.nodes.assign((states + successors) * words, 0);
    const bool derived = parent_derived && expansion.derived && expansion.derived_words != 0;
    out.derived_words = derived ? expansion.derived_words : 0;
    if (derived)
    {
        if (parent_derived_stride < expansion.derived_words)
            invalid("the parent derived stride is below the expansion's derived words");
        out.derived.assign((states + successors) * out.derived_words, 0);
    }
    out.node_offsets.assign(states + 1, 0);
    out.edge_offsets.assign(states + 1, 0);
    out.edge_src.assign(successors, 0);
    out.edge_dst.resize(successors);
    out.edge_schema.assign(expansion.schema, expansion.schema + successors);
    out.edge_binding.assign(expansion.binding, expansion.binding + successors * expansion.label_width);
    out.truncated.assign(states, 0);
    u64 node = 0;
    for (u64 s = 0; s < states; ++s)
    {
        const i32 first = expansion.offsets[s], last = expansion.offsets[s + 1];
        if (last < first || static_cast<u64>(last) > successors)
            invalid("the expansion's offsets decrease at state " + std::to_string(s));
        std::copy_n(parents.row(s), parents.words, out.nodes.begin() + node * words);
        if (derived)
            std::copy_n(parent_derived + s * parent_derived_stride, out.derived_words,
                        out.derived.begin() + node * out.derived_words);
        ++node;
        for (i32 k = first; k < last; ++k, ++node)
        {
            std::copy_n(expansion.succ + u64{static_cast<u32>(k)} * expansion.words, expansion.words,
                        out.nodes.begin() + node * words);
            if (derived)
                std::copy_n(expansion.derived + u64{static_cast<u32>(k)} * expansion.derived_words, out.derived_words,
                            out.derived.begin() + node * out.derived_words);
            out.edge_dst[static_cast<u64>(k)] = k - first + 1;
        }
        out.node_offsets[s + 1] = static_cast<i32>(node);
        out.edge_offsets[s + 1] = last;
    }
    return out;
}

HorizonGraphs build_horizon_graphs(const SuccessorGraph& space, std::span<const u32> roots, i32 task_id,
                                   const HorizonBuild& options)
{
    for (const u32 root : roots)
        if (root >= space.states)
            invalid("state " + std::to_string(root) + " is not a state of the space of " + std::to_string(space.states) + " states");
    SpaceSource source(space, roots);
    return run(source, std::vector<i32>(roots.size(), task_id), roots.size(),
               [&space, roots](u64 g) { return space.state_words + u64{roots[g]} * space.row_stride; }, space.words,
               space.label_width, options);
}
}  // namespace mymyr::encoding
