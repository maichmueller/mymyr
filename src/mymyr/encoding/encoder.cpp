#include "mymyr/encoding/encoder.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace
{
constexpr usize k_cache_line = 64;
// Graphs per tile of the relation-major scan: it runs over the tile's graphs one relation at a time, so that the strided
// writes (G entries apart) fill whole cache lines.
constexpr u64 k_scan_tile = 128;
constexpr u64 k_none = std::numeric_limits<u64>::max();

[[nodiscard]] u64 index_limit(IndexType index)
{
    return static_cast<u64>(index == IndexType::I64 ? std::numeric_limits<i64>::max() : std::numeric_limits<i32>::max());
}

[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }
}  // namespace

namespace detail
{
void check_index_range(u64 value, IndexType index, const char* what)
{
    const bool wide = index == IndexType::I64;
    if (value <= index_limit(index))
        return;
    throw std::overflow_error("mymyr: encoding: the batch needs " + std::to_string(value) + " " + what + ", more than " +
                              (wide ? "int64" : "int32") + " indices hold" + (wide ? "" : "; use index_dtype='int64'"));
}
}  // namespace detail

namespace
{
[[noreturn]] void unsupported(const char* option)
{
    throw std::invalid_argument(std::string("mymyr: encoding: ") + option + " is not supported by this encoder yet");
}

void refuse_unsupported(const FlatOptions& o)
{
    if (!o.goal_derivations.empty())
        unsupported("goal_derivations");
    if (o.max_goal_level > 0)
        unsupported("max_goal_level");
    if (!o.targets.empty())
        unsupported("targets");
    if (o.predicate_virtual_nodes)
        unsupported("predicate_virtual_nodes");
    if (o.horizon.mode != HorizonMode::None)
        unsupported("horizon");
}

std::vector<i64> shape1(u64 n) { return {static_cast<i64>(n)}; }
std::vector<i64> shape2(u64 n, u64 m) { return {static_cast<i64>(n), static_cast<i64>(m)}; }

/// The state of one pool member: it counts a contiguous slice of the graphs and later fills the same slice, so that the
/// per-graph data of the slice stays with the member. Aligned so that members never share a cache line.
struct alignas(k_cache_line) Member
{
    // pass one
    std::vector<detail::GraphTotals> totals;  // the slice's graphs
    std::vector<u32> cell_counts;             // the slice's cells, graph after graph, in cell order (relation-major)
    std::vector<u64> tally;                   // [R] tuples per relation in the slice (relation-major)
    u64 entities = 0, tuples = 0, args = 0;   // the slice's sizes
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none;  // the first row of each kind of invalid input
    // between the passes
    u64 entity_start = 0, tuple_start = 0, arg_start = 0;  // where the slice starts in the batch
    std::vector<u64> start_tuples, start_args;             // [R] where the slice starts in each relation (relation-major)
    // pass two
    std::vector<u64> run_tuples, run_args;  // [R] the running positions
    std::vector<u32> tile_counts;           // [R, k_scan_tile] the counts of the tile being scanned
};
}  // namespace

struct EncodeScratch::Impl
{
    // The input count() saw, so that fill() can refuse another one.
    bool counted = false;
    const u64* counted_data = nullptr;
    const i32* counted_ids = nullptr;
    const u64* counted_derived = nullptr;
    u64 counted_rows = 0;
    FlatSizes sizes;

    std::vector<u32> arity;          // [R] the arguments per tuple of each relation
    std::unique_ptr<u64[]> derived;  // [G, derived_stride] when the rows' derived words are evaluated here
    u64 derived_capacity = 0;
    u64 derived_stride = 0;
    std::vector<Member> members;
};

EncodeScratch::EncodeScratch() : m_impl(std::make_unique<Impl>()) {}
EncodeScratch::~EncodeScratch() = default;

namespace
{
using Scratch = EncodeScratch::Impl;

/// What the passes of one call share.
struct Pass
{
    const EncodeInput& in;
    Scratch& s;
    const EncodingContext& context;
    detail::ContextView view;
    u64 graphs;
    u32 members;
    u32 R;
    bool relation_major;
    bool evaluate;  // the derived words of the rows are evaluated here, into s.derived

    [[nodiscard]] u32 instance_of(u64 g) const noexcept { return in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0; }

    [[nodiscard]] const u64* derived_of(u64 g, u32 instance) const noexcept
    {
        if (!context.needs_derived(instance))
            return nullptr;
        return in.derived ? in.derived + g * s.derived_stride : s.derived.get() + g * s.derived_stride;
    }
};

/// Pass one of a member: validates its graphs, evaluates their derived words, counts them, and (relation-major) tallies
/// the tuples per relation. An invalid graph is recorded by kind and counts as empty.
void count_range(const Pass& p, u32 t)
{
    const auto [lo, hi] = ThreadPool::slice(p.graphs, t, p.members);
    const rl::TaskTable& table = p.context.table();
    const u32 instances = table.size();
    Member& m = p.s.members[t];
    m.totals.assign(hi - lo, detail::GraphTotals{});
    m.entities = m.tuples = m.args = 0;
    m.bad_id = m.narrow = m.bad_bits = k_none;
    if (p.relation_major)
    {
        m.tally.assign(p.R, 0);
        m.cell_counts.clear();
    }
    for (u64 g = lo; g < hi; ++g)
    {
        const u32 instance = p.instance_of(g);
        if (p.in.task_ids && (p.in.task_ids[g] < 0 || instance >= instances))
        {
            m.bad_id = std::min(m.bad_id, g);
            continue;
        }
        if (p.in.states.words < table.instance(instance).words)
        {
            m.narrow = std::min(m.narrow, g);
            continue;
        }
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        const u64* row = p.in.states.row(g);
        const u64* derived = p.derived_of(g, instance);
        if (derived && p.evaluate)
        {
            u64* dst = p.s.derived.get() + g * p.s.derived_stride;
            std::fill_n(dst, p.s.derived_stride, u64{0});
            if (!detail::has_foreign_atom_bits(v, row))  // the row is refused after the pass
            {
                const Task& task = *table.instance(instance).task;
                const WorkspaceLease workspace = task.workspace();
                auto& engine = workspace->engine();
                engine.set_state(row, v.words);
                engine.build_view();
                workspace->axioms().evaluate();
                std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), dst);
            }
        }
        u32* cell_counts = nullptr;
        if (p.relation_major)
        {
            const usize at = m.cell_counts.size();
            m.cell_counts.resize(at + v.num_cells);
            cell_counts = m.cell_counts.data() + at;
        }
        detail::GraphTotals& totals = m.totals[g - lo];
        totals = detail::count_graph(v, {row, derived}, cell_counts);
        if (totals.flags & detail::k_graph_bad_row)
        {
            m.bad_bits = std::min(m.bad_bits, g);
            continue;
        }
        m.entities += totals.entities;
        m.tuples += totals.tuples;
        m.args += totals.args;
        if (p.relation_major)
            for (u32 i = 0; i < v.num_cells; ++i)
                m.tally[v.cells[i].relation] += cell_counts[i];
    }
}

/// Pass two of a member: the positions (relation-major) and the contents of its graphs.
template<class I>
void fill_range(const Pass& p, const FlatBuffers& b, u32 t)
{
    const auto [lo, hi] = ThreadPool::slice(p.graphs, t, p.members);
    const detail::GraphOut<I> out{p.graphs,
                                  static_cast<I*>(b.num_objects),
                                  static_cast<I*>(b.relation_counts),
                                  static_cast<I*>(b.relation_offsets),
                                  static_cast<I*>(b.relation_arg_offsets),
                                  static_cast<I*>(b.relation_args),
                                  b.entity_role,
                                  static_cast<I*>(b.entity_ref)};
    I* entity_offsets = static_cast<I*>(b.entity_offsets);
    Member& m = p.s.members[t];
    u64 entity = m.entity_start, tuple = m.tuple_start, arg = m.arg_start;
    auto fill_one = [&](u64 g)
    {
        const u32 instance = p.instance_of(g);
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        const detail::GraphTotals& totals = m.totals[g - lo];
        const detail::RowInput row{p.in.states.row(g), p.derived_of(g, instance)};
        entity_offsets[g] = static_cast<I>(entity);
        b.task_ids[g] = p.in.task_ids ? p.in.task_ids[g] : 0;
        if (p.relation_major)
            detail::fill_graph_relation_major<I>(v, row, g, entity, totals, out);
        else
            detail::fill_graph<I>(v, row, g, tuple, arg, entity, totals, out);
        entity += totals.entities;
        tuple += totals.tuples;
        arg += totals.args;
    };
    if (!p.relation_major)
    {
        for (u64 g = lo; g < hi; ++g)
            fill_one(g);
        return;
    }
    m.run_tuples = m.start_tuples;
    m.run_args = m.start_args;
    m.tile_counts.resize(p.R * k_scan_tile);
    const u32* cell_counts = m.cell_counts.data();
    for (u64 first = lo; first < hi; first += k_scan_tile)
    {
        const u64 width = std::min(k_scan_tile, hi - first);
        // the tile's counts, relation by relation (a relation without a cell in a graph has count 0)
        std::fill(m.tile_counts.begin(), m.tile_counts.end(), 0u);
        const u32* next = cell_counts;
        for (u64 j = 0; j < width; ++j)
        {
            const detail::InstanceView v = detail::instance_view(p.view, p.instance_of(first + j));
            for (u32 i = 0; i < v.num_cells; ++i)
                m.tile_counts[v.cells[i].relation * k_scan_tile + j] = next[i];
            next += v.num_cells;
        }
        // the exclusive scan in segment order r * G + g, written in runs of `width` per relation
        for (u32 r = 0; r < p.R; ++r)
        {
            I* offsets = out.relation_offsets + r * p.graphs + first;
            I* arg_offsets = out.relation_arg_offsets + r * p.graphs + first;
            const u32* counts = m.tile_counts.data() + r * k_scan_tile;
            const u64 arity = p.s.arity[r];
            u64 tuples = m.run_tuples[r];
            u64 args = m.run_args[r];
            if (m.tally[r] == 0)  // no graph of the slice has a tuple of the relation: the offsets do not advance
            {
                std::fill_n(offsets, width, static_cast<I>(tuples));
                std::fill_n(arg_offsets, width, static_cast<I>(args));
                continue;
            }
            for (u64 j = 0; j < width; ++j)
            {
                offsets[j] = static_cast<I>(tuples);
                arg_offsets[j] = static_cast<I>(args);
                tuples += counts[j];
                args += counts[j] * arity;
            }
            m.run_tuples[r] = tuples;
            m.run_args[r] = args;
        }
        for (u64 j = 0; j < width; ++j)
            fill_one(first + j);
        cell_counts = next;
    }
}

/// Runs `f(t)` on every member of the pool without allocating a closure.
template<class F>
void run(ThreadPool& pool, const F& f)
{
    const std::function<void(u32)> job = [&f](u32 t) { f(t); };
    pool.run(job);
}
}  // namespace

std::shared_ptr<const FlatEncoder> FlatEncoder::create(rl::TaskTablePtr table, const FlatOptions& options)
{
    if (!table)
        invalid("an encoder needs a task table");
    auto encoder = std::shared_ptr<FlatEncoder>(new FlatEncoder());
    encoder->m_schema = EncodingSchema::create(*table, options);
    refuse_unsupported(options);
    encoder->m_context = EncodingContext::create(std::move(table), encoder->m_schema);
    return encoder;
}

std::shared_ptr<const FlatEncoder> FlatEncoder::create(TaskPtr task, const FlatOptions& options)
{
    return create(rl::TaskTable::single(std::move(task)), options);
}

FlatSizes FlatEncoder::count(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const
{
    const EncodingContext& context = *m_context;
    Scratch& s = scratch.impl();
    s.counted = false;
    const IndexType index = m_schema->options().index;
    const bool relation_major = m_schema->options().layout == Layout::Relation;
    const u64 graphs = in.states.rows;
    const u32 R = static_cast<u32>(m_schema->num_relations());
    const rl::StateBatchView& states = in.states;

    // 1. check what does not depend on a row
    if (states.numeric_words != 0)
        invalid("numeric state rows are not supported");
    if (graphs && !states.data)
        invalid("the state buffer is null");
    if (states.stride && states.stride < states.words)
        invalid("the state row stride is smaller than its width");
    if (in.derived && in.derived_stride < context.max_derived_words())
        invalid("the derived row stride is smaller than the derived words of an instance");
    if (R && graphs > std::numeric_limits<u64>::max() / R - 1)
        throw std::overflow_error("mymyr: encoding: the batch has too many cells");
    const bool evaluate = !in.derived && context.max_derived_words() > 0;
    s.derived_stride = in.derived ? in.derived_stride : context.max_derived_words();
    if (evaluate && graphs * s.derived_stride > s.derived_capacity)
    {
        s.derived_capacity = graphs * s.derived_stride;
        s.derived = std::make_unique_for_overwrite<u64[]>(s.derived_capacity);
    }
    s.members.resize(pool.size());

    // 2. validate, evaluate and count, in parallel over the graphs
    const Pass pass{in, s, context, context.view(), graphs, pool.size(), R, relation_major, evaluate};
    run(pool, [&pass](u32 t) { count_range(pass, t); });
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none;
    for (const Member& m : s.members)
    {
        bad_id = std::min(bad_id, m.bad_id);
        narrow = std::min(narrow, m.narrow);
        bad_bits = std::min(bad_bits, m.bad_bits);
    }
    const rl::TaskTable& table = this->table();
    if (bad_id != k_none)
        invalid("task id " + std::to_string(in.task_ids[bad_id]) + " of row " + std::to_string(bad_id) +
                " is outside the table's " + std::to_string(table.size()) + " instances");
    if (narrow != k_none)
        invalid("row " + std::to_string(narrow) + " has " + std::to_string(states.words) + " words, its instance needs " +
                std::to_string(table.instance(pass.instance_of(narrow)).words));
    if (bad_bits != k_none)
        invalid("row " + std::to_string(bad_bits) + " sets an atom bit its instance does not have");

    // 3. scan the members, in u64: where each starts in the batch and, relation-major, in every relation
    u64 entities = 0, tuples = 0, args = 0;
    for (Member& m : s.members)
    {
        m.entity_start = entities;
        m.tuple_start = tuples;
        m.arg_start = args;
        entities += m.entities;
        tuples += m.tuples;
        args += m.args;
    }
    if (relation_major)
    {
        for (Member& m : s.members)
        {
            m.start_tuples.resize(R);
            m.start_args.resize(R);
        }
        s.arity.resize(R);
        u64 tuple_position = 0, arg_position = 0;
        for (u32 r = 0; r < R; ++r)
        {
            s.arity[r] = m_schema->relation(r).arity;
            for (Member& m : s.members)
            {
                m.start_tuples[r] = tuple_position;
                m.start_args[r] = arg_position;
                tuple_position += m.tally[r];
                arg_position += m.tally[r] * s.arity[r];
            }
        }
    }

    FlatSizes sizes;
    sizes.entities = entities;
    sizes.tuples = tuples;
    sizes.args = args;
    const u64 limit = index_limit(index);
    if (entities > limit || tuples > limit || args > limit || graphs * R + 1 > limit || graphs + 1 > limit)
        sizes.overflow |= k_overflow_index;

    s.sizes = sizes;
    s.counted = true;
    s.counted_data = states.data;
    s.counted_ids = in.task_ids;
    s.counted_derived = in.derived;
    s.counted_rows = graphs;
    return sizes;
}

void FlatEncoder::fill(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
                       EncodeScratch& scratch) const
{
    Scratch& s = scratch.impl();
    if (!s.counted || s.counted_rows != in.states.rows || s.counted_data != in.states.data ||
        s.counted_ids != in.task_ids || s.counted_derived != in.derived || s.sizes.entities != sizes.entities ||
        s.sizes.tuples != sizes.tuples || s.sizes.args != sizes.args)
        invalid("fill needs the scratch and the sizes of count() on the same input");
    if (sizes.overflow != 0)
        invalid("the batch does not fit: its sizes overflow");
    if (s.members.size() != pool.size())
        invalid("fill needs the thread pool of count()");
    const EncodingSchema& schema = *m_schema;
    const FlatOptions& options = schema.options();
    const u64 graphs = in.states.rows;
    if (!out.task_ids || !out.num_objects || !out.entity_offsets || !out.relation_offsets || !out.relation_arg_offsets)
        invalid("a destination array of the batch is null");
    if ((out.relation_counts != nullptr) != options.emit_counts)
        invalid(options.emit_counts ? "relation_counts is null, the schema stores it"
                                    : "relation_counts must be null, the schema omits it");
    if ((out.entity_role != nullptr) != !schema.implicit_entity_arrays() ||
        (out.entity_ref != nullptr) != !schema.implicit_entity_arrays())
        invalid(schema.implicit_entity_arrays() ? "entity_role and entity_ref must be null, the schema omits them"
                                                : "entity_role and entity_ref are null, the schema stores them");
    const bool evaluate = !in.derived && m_context->max_derived_words() > 0;
    const Pass pass{in,
                    s,
                    *m_context,
                    m_context->view(),
                    graphs,
                    pool.size(),
                    static_cast<u32>(schema.num_relations()),
                    options.layout == Layout::Relation,
                    evaluate};
    if (options.index == IndexType::I64)
        run(pool, [&pass, &out](u32 t) { fill_range<i64>(pass, out, t); });
    else
        run(pool, [&pass, &out](u32 t) { fill_range<i32>(pass, out, t); });
    const u64 cells = graphs * pass.R;
    auto close = [&]<class I>(I)
    {
        static_cast<I*>(out.entity_offsets)[graphs] = static_cast<I>(sizes.entities);
        static_cast<I*>(out.relation_offsets)[cells] = static_cast<I>(sizes.tuples);
        static_cast<I*>(out.relation_arg_offsets)[cells] = static_cast<I>(sizes.args);
    };
    if (options.index == IndexType::I64)
        close(i64{});
    else
        close(i32{});
}

FlatSizes FlatEncoder::encode_into(const EncodeInput& in, const FlatCapacities& capacities, const FlatBuffers& out,
                                   ThreadPool& pool, EncodeScratch& scratch) const
{
    FlatSizes sizes = count(in, pool, scratch);
    if (sizes.entities > capacities.entities || sizes.tuples > capacities.tuples || sizes.args > capacities.args)
        sizes.overflow |= k_overflow_capacity;
    if (!sizes.overflowed())
        fill(in, sizes, out, pool, scratch);
    return sizes;
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in) const
{
    ThreadPool pool(1);
    EncodeScratch scratch;
    return encode(in, pool, scratch);
}

namespace
{
/// The array of a bundle under construction, null if the bundle has none of that name.
void* array_of(rl::ArrayBundle& bundle, const char* name) { return bundle.mutable_data(name); }

FlatBuffers buffers_of(rl::ArrayBundle& bundle)
{
    FlatBuffers out;
    out.task_ids = static_cast<i32*>(array_of(bundle, "task_ids"));
    out.num_objects = array_of(bundle, "num_objects");
    out.entity_offsets = array_of(bundle, "entity_offsets");
    out.entity_role = static_cast<u8*>(array_of(bundle, "entity_role"));
    out.entity_ref = array_of(bundle, "entity_ref");
    out.relation_counts = array_of(bundle, "relation_counts");
    out.relation_offsets = array_of(bundle, "relation_offsets");
    out.relation_arg_offsets = array_of(bundle, "relation_arg_offsets");
    out.relation_args = array_of(bundle, "relation_args");
    return out;
}
}  // namespace

rl::ArrayBundle FlatEncoder::allocate(u64 graphs, u64 entities, u64 args) const
{
    const IndexType index = m_schema->options().index;
    const u64 R = m_schema->num_relations();
    const rl::DType itype = index == IndexType::I64 ? rl::DType::I64 : rl::DType::I32;
    rl::ArrayBundle::Builder builder;
    builder.add_uninitialized("task_ids", rl::DType::I32, shape1(graphs));
    builder.add_uninitialized("num_objects", itype, shape1(graphs));
    builder.add_uninitialized("entity_offsets", itype, shape1(graphs + 1));
    if (!m_schema->implicit_entity_arrays())
    {
        builder.add_uninitialized("entity_role", rl::DType::U8, shape1(entities));
        builder.add_uninitialized("entity_ref", itype, shape1(entities));
    }
    if (m_schema->emits_counts())
        builder.add_uninitialized("relation_counts", itype, shape2(graphs, R));
    builder.add_uninitialized("relation_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_arg_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_args", itype, shape1(args));
    builder.scalar("version", 1);
    builder.scalar("num_graphs", static_cast<i64>(graphs));
    builder.scalar("num_relations", static_cast<i64>(R));
    builder.scalar("layout", m_schema->options().layout == Layout::Relation ? 1 : 0);
    builder.scalar("index_dtype", index == IndexType::I64 ? 1 : 0);
    builder.scalar("schema_fingerprint", std::bit_cast<i64>(m_schema->fingerprint()));
    builder.scalar("statics", 0);
    return builder.build();
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in, ThreadPool& pool,
                                                           EncodeScratch& scratch) const
{
    const FlatSizes sizes = count(in, pool, scratch);
    if (sizes.overflow & k_overflow_index)
    {
        const IndexType index = m_schema->options().index;
        const u64 graphs = in.states.rows;
        const u64 R = m_schema->num_relations();
        detail::check_index_range(sizes.entities, index, "entities");
        detail::check_index_range(sizes.tuples, index, "tuples");
        detail::check_index_range(sizes.args, index, "tuple arguments");
        detail::check_index_range(graphs * R + 1, index, "relation cells");
        detail::check_index_range(graphs + 1, index, "graphs");
    }
    auto bundle = std::make_shared<rl::ArrayBundle>(allocate(in.states.rows, sizes.entities, sizes.args));
    fill(in, sizes, buffers_of(*bundle), pool, scratch);
    return bundle;
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode_capacity(const EncodeInput& in, const FlatCapacities& capacities,
                                                                    ThreadPool& pool, EncodeScratch& scratch,
                                                                    FlatSizes& sizes) const
{
    sizes = count(in, pool, scratch);
    if (sizes.entities > capacities.entities || sizes.tuples > capacities.tuples || sizes.args > capacities.args)
        sizes.overflow |= k_overflow_capacity;
    const u64 graphs = in.states.rows;
    const bool fits = !sizes.overflowed();
    auto bundle = std::make_shared<rl::ArrayBundle>(allocate(graphs, capacities.entities, capacities.args));
    const FlatBuffers out = buffers_of(*bundle);
    const usize index_bytes = m_schema->options().index == IndexType::I64 ? 8 : 4;
    // the entries a batch does not use are zero, and an overflowing batch is all zero
    auto zero = [](void* data, u64 from, u64 to, usize width)
    { std::memset(static_cast<std::byte*>(data) + from * width, 0, (to - from) * width); };
    if (fits)
    {
        fill(in, sizes, out, pool, scratch);
        zero(out.relation_args, sizes.args, capacities.args, index_bytes);
        if (out.entity_role)
        {
            zero(out.entity_role, sizes.entities, capacities.entities, 1);
            zero(out.entity_ref, sizes.entities, capacities.entities, index_bytes);
        }
    }
    else
        for (const rl::ArrayInfo& info : bundle->arrays())
            std::memset(bundle->mutable_data(info), 0, info.elements() * rl::dtype_bytes(info.dtype));
    return bundle;
}
}  // namespace mymyr::encoding
