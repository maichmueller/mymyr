#include "mymyr/encoding/encoder.hpp"

#include "encode_pass.hpp"

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
using detail::k_none;
using detail::Member;
using detail::members_for;
using detail::run;

[[nodiscard]] u64 index_limit(IndexType index)
{
    return static_cast<u64>(index == IndexType::I64 ? std::numeric_limits<i64>::max()
                                                    : std::numeric_limits<i32>::max());
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
rl::ArrayShape shape1(u64 n) { return {static_cast<i64>(n)}; }
rl::ArrayShape shape2(u64 n, u64 m) { return {static_cast<i64>(n), static_cast<i64>(m)}; }

}  // namespace

EncodeScratch::EncodeScratch() : m_impl(std::make_unique<Impl>()) {}
EncodeScratch::~EncodeScratch() = default;


EncodeScratch& EncodeScratch::of_this_thread()
{
    thread_local EncodeScratch scratch;
    return scratch;
}

namespace
{
using Scratch = EncodeScratch::Impl;

/// Whether two inputs read the same buffers with the same shape: every pointer, stride and width equal.
bool same_input(const EncodeInput& a, const EncodeInput& b)
{
    const auto same_layers = [](const LayerInput& x, const LayerInput& y)
    {
        return x.pos == y.pos && x.neg == y.neg && x.stride == y.stride && x.layers == y.layers && x.words == y.words &&
               x.derived_pos == y.derived_pos && x.derived_neg == y.derived_neg &&
               x.derived_stride == y.derived_stride && x.derived_words == y.derived_words;
    };
    const auto& ha = a.horizon;
    const auto& hb = b.horizon;
    return ha.nodes.data == hb.nodes.data && ha.nodes.rows == hb.nodes.rows && ha.nodes.words == hb.nodes.words &&
           ha.nodes.stride == hb.nodes.stride && ha.nodes.numeric_words == hb.nodes.numeric_words &&
           ha.graphs == hb.graphs && ha.node_offsets == hb.node_offsets && ha.edge_offsets == hb.edge_offsets &&
           ha.edge_src == hb.edge_src && ha.edge_dst == hb.edge_dst && ha.edge_schema == hb.edge_schema &&
           ha.edge_binding == hb.edge_binding && ha.label_width == hb.label_width &&
           same_layers(a.subgoals, b.subgoals) && same_layers(a.history.masks, b.history.masks) &&
           a.history.dt == b.history.dt && a.history.count == b.history.count && a.states.data == b.states.data &&
           a.states.rows == b.states.rows && a.states.words == b.states.words &&
           a.states.stride == b.states.stride && a.states.numeric_words == b.states.numeric_words &&
           a.task_ids == b.task_ids && a.derived == b.derived && a.derived_stride == b.derived_stride &&
           a.goal_pos == b.goal_pos && a.goal_neg == b.goal_neg && a.goal_stride == b.goal_stride &&
           a.goal_words == b.goal_words && a.goal_derived_pos == b.goal_derived_pos &&
           a.goal_derived_neg == b.goal_derived_neg && a.goal_derived_stride == b.goal_derived_stride &&
           a.goal_derived_words == b.goal_derived_words && a.actions == b.actions;
}

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
    u64 starts_stride;  // the words of run starts of one graph: num_schemas + 1 with expansion rows, else 0
    // the subgoal layers and history of the call, as the lanes read them
    detail::LayerSource source = {in.subgoals, in.history, context.schema().options().history_max_steps.value_or(0)};
    // some row may have subgoal layers or history steps
    bool layered = source.layered();
    // the batch stores the Simple graph, which follows the fill of each graph with its cells' counts
    bool simple = context.schema().emits_simple();
    // some lane beyond the relation arrays is present: the Simple graph or numeric rows. The passes are instantiated
    // without these lanes for the batches without them, which are the hot ones.
    bool lanes = simple || in.states.numeric_words != 0;
    // the dtype of the batch's numeric values
    NumericDtype numeric_dtype = context.schema().options().numeric_dtype;
    detail::SimpleParams simple_params{R, context.schema().simple_max_arity(), context.schema().simple_constants()};

    [[nodiscard]] u32 instance_of(u64 g) const noexcept { return in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0; }

    [[nodiscard]] const u64* derived_of(u64 g, const detail::InstanceView& v) const noexcept
    {
        if (!detail::reads_derived(v, in.goal_pos != nullptr, in.goal_derived_pos != nullptr,
                                   in.subgoals.layers != 0 && in.subgoals.derived_pos != nullptr))
            return nullptr;
        return in.derived ? in.derived + g * s.derived_stride : s.derived.get() + g * s.derived_stride;
    }

    /// What the lanes read for graph g of `instance`: the state row, its derived words (null if not needed) and the
    /// goal masks of the row (null: the instance's goal).
    /// Whether graph g has expansion rows.
    [[nodiscard]] bool has_actions(u64 g) const noexcept
    {
        return in.actions && in.actions->offsets[g + 1] > in.actions->offsets[g];
    }

    /// The valid history steps of graph g, and how many of the newest of them are encoded.
    [[nodiscard]] u32 history_used(u64 g) const noexcept { return detail::history_used(source, g); }

    /// The cells of graph g of `v`'s instance.
    template<bool Layered>
    [[nodiscard]] detail::CellList cells_of(const detail::InstanceView& v, u64 g) const noexcept
    {
        if (!Layered)
            return detail::cell_list(v, in.goal_pos != nullptr, has_actions(g));
        return detail::cell_list(v, in.goal_pos != nullptr, has_actions(g), in.subgoals.layers != 0,
                                 history_used(g) != 0);
    }

    /// Gives `row` the subgoal layers and history steps of graph g, stored in `layers`, when it has any. Out of line:
    /// rows without layers are the hot ones.
    [[gnu::noinline]] void attach_layers(u64 g, detail::RowLayers& layers, detail::RowInput& row) const noexcept
    {
        if (detail::bind_layers(source, g, layers))
            row.layers = &layers;
    }

    /// Computes the subgoal levels of a row that has subgoal layers, into `storage`; the state words, and so the
    /// derived words of the row, are final. Out of line, like attach_layers.
    [[gnu::noinline]] void bind_levels(const detail::InstanceView& v, const detail::RowInput& row,
                                       detail::RowLayers& layers, std::vector<u64>& storage) const noexcept
    {
        if (layers.subgoals.layers == 0)
            return;
        storage.resize(detail::effective_layer_words(v, layers.subgoals));
        detail::bind_levels(detail::SoloGroup{}, v, row, layers, storage.data());
    }

    /// The input of graph g; `layers` is the storage of its subgoal layers and history steps (the row refers to it iff
    /// it has any). `Lanes`: the batch has numeric rows.
    template<bool Layered, bool Lanes>
    [[nodiscard]] detail::RowInput row_input(u64 g, const detail::InstanceView& v, u32* starts, u32* cursors,
                                             detail::RowLayers& layers) const noexcept
    {
        detail::RowInput row{in.states.row(g), derived_of(g, v)};
        if constexpr (Lanes)
            if (in.states.numeric_words != 0)
                row.numeric = row.atoms + in.states.words;
        if (in.goal_pos)
        {
            row.goal_pos = in.goal_pos + g * in.goal_stride;
            row.goal_neg = in.goal_neg + g * in.goal_stride;
        }
        if (in.goal_derived_pos)
        {
            row.goal_derived_pos = in.goal_derived_pos + g * in.goal_derived_stride;
            row.goal_derived_neg = in.goal_derived_neg + g * in.goal_derived_stride;
        }
        if (in.actions)
        {
            const rl::Expansion& x = *in.actions;
            const u64 first = static_cast<u64>(x.offsets[g]);
            row.actions = {x.schema + first,
                           x.binding + first * x.label_width,
                           x.parent ? x.parent + first : nullptr,
                           x.label_width,
                           static_cast<u32>(x.offsets[g + 1] - x.offsets[g]),
                           g,
                           starts,
                           cursors};
        }
        if constexpr (Layered)
            attach_layers(g, layers, row);
        return row;
    }

    /// Whether the expansion rows of graph g are a slice of the expansion: ascending offsets inside the rows written.
    [[nodiscard]] bool action_slice_valid(u64 g) const noexcept
    {
        const rl::Expansion& x = *in.actions;
        return x.offsets[g] >= 0 && x.offsets[g] <= x.offsets[g + 1] && static_cast<u64>(x.offsets[g + 1]) <= x.capacity;
    }
};

/// Whether the subgoal layers or history steps of graph g cannot be encoded for its instance (`v`): a history count
/// outside [0, H], or masks narrower than the instance. The first graph of each kind is recorded in the member.
[[gnu::noinline]] bool refuse_layers(const Pass& p, Member& m, const detail::InstanceView& v, u64 g)
{
    switch (detail::layer_validity(v, p.source, g))
    {
        case detail::LayerValidity::Valid: return false;
        case detail::LayerValidity::BadHistoryCount: m.bad_history_count = std::min(m.bad_history_count, g); break;
        case detail::LayerValidity::NarrowSubgoals: m.narrow_subgoals = std::min(m.narrow_subgoals, g); break;
        case detail::LayerValidity::NarrowHistory: m.narrow_history = std::min(m.narrow_history, g); break;
    }
    return true;
}

/// Pass one of a member: validates its graphs, evaluates their derived words, counts them, and (relation-major) tallies
/// the tuples per relation. An invalid graph is recorded by kind and counts as empty.
template<bool Layered, bool Lanes>
void count_range(const Pass& p, u32 t)
{
    const auto [lo, hi] = ThreadPool::slice(p.graphs, t, p.members);
    const rl::TaskTable& table = p.context.table();
    Member& m = p.s.members[t];
    m.totals.assign(hi - lo, detail::GraphTotals{});
    m.entities = m.tuples = m.args = m.targets = m.history = m.numeric = 0;
    m.bad_id = m.narrow = m.bad_bits = m.narrow_goal = m.bad_goal_bits = k_none;
    m.narrow_subgoals = m.bad_subgoal_bits = m.narrow_history = m.bad_history_bits = m.bad_history_count = k_none;
    m.bad_slice = m.bad_action_order = m.bad_action_label = k_none;
    m.virtual_scratch.resize(p.view.num_virtual);
    const bool keep_counts = p.relation_major || p.view.num_virtual != 0 || p.layered || p.simple;
    if (p.starts_stride)
    {
        m.action_starts.resize((hi - lo) * p.starts_stride);
        m.action_cursors.resize(p.view.num_schemas);
    }
    if (p.relation_major)
        m.tally.assign(p.R, 0);
    m.cell_counts.clear();
    for (u64 g = lo; g < hi; ++g)
    {
        const i32 id = p.in.task_ids ? p.in.task_ids[g] : 0;
        if (id < 0 || static_cast<u32>(id) >= p.view.num_instances)
        {
            m.bad_id = std::min(m.bad_id, g);
            continue;
        }
        const u32 instance = static_cast<u32>(id);
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        if (const auto validity = detail::row_validity(v, p.in.states.words, p.in.goal_pos != nullptr, p.in.goal_words,
                                                       p.in.goal_derived_pos != nullptr, p.in.goal_derived_words);
            validity != detail::RowValidity::Valid)
        {
            u64& bad = validity == detail::RowValidity::NarrowState ? m.narrow : m.narrow_goal;
            bad = std::min(bad, g);
            continue;
        }
        if (p.in.actions && !p.action_slice_valid(g))
        {
            m.bad_slice = std::min(m.bad_slice, g);
            continue;
        }
        if (Layered && refuse_layers(p, m, v, g))
            continue;
        detail::RowLayers layers;
        const detail::RowInput input = p.row_input<Layered, Lanes>(
            g, v, p.starts_stride ? m.action_starts.data() + (g - lo) * p.starts_stride : nullptr,
            m.action_cursors.data(), layers);
        const u64* row = input.atoms;
        const u64* derived = input.derived;
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
                engine.set_numeric(input.numeric);
                engine.build_view();
                workspace->axioms().evaluate();
                std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), dst);
            }
        }
        if (Layered && input.layers)
            p.bind_levels(v, input, layers, m.effective);
        const detail::CellList cells = detail::cell_list<Layered>(v, input);
        u32* cell_counts = nullptr;
        if (keep_counts)
        {
            const usize at = m.cell_counts.size();
            m.cell_counts.resize(at + cells.count);
            cell_counts = m.cell_counts.data() + at;
        }
        detail::GraphTotals& totals = m.totals[g - lo];
        totals = detail::count_graph<Layered>(v, input, cell_counts, m.virtual_scratch.data());
        if (totals.flags & (detail::k_graph_bad_row | detail::k_graph_bad_goal | detail::k_graph_bad_action_order |
                            detail::k_graph_bad_action_label | detail::k_graph_bad_layers |
                            detail::k_graph_bad_history))
        {
            if (totals.flags & detail::k_graph_bad_row)
                m.bad_bits = std::min(m.bad_bits, g);
            else if (totals.flags & detail::k_graph_bad_goal)
                m.bad_goal_bits = std::min(m.bad_goal_bits, g);
            else if (totals.flags & detail::k_graph_bad_layers)
                m.bad_subgoal_bits = std::min(m.bad_subgoal_bits, g);
            else if (totals.flags & detail::k_graph_bad_history)
                m.bad_history_bits = std::min(m.bad_history_bits, g);
            else if (totals.flags & detail::k_graph_bad_action_label)
                m.bad_action_label = std::min(m.bad_action_label, g);
            else
                m.bad_action_order = std::min(m.bad_action_order, g);
            continue;
        }
        m.entities += totals.entities;
        m.tuples += totals.tuples;
        m.args += totals.args;
        m.targets += totals.targets;
        m.history += totals.history_steps;
        if constexpr (Lanes)
            m.numeric += detail::numeric_slot_count(v);
        if (p.relation_major)
            for (u32 i = 0; i < cells.count; ++i)
                m.tally[cells.cells[i].relation] += cell_counts[i];
    }
}

/// Pass two of a member: the positions (relation-major) and the contents of its graphs.
template<class I, bool Layered, bool Lanes>
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
                                  static_cast<I*>(b.entity_ref),
                                  b.goal_complete,
                                  static_cast<I*>(b.target_offsets),
                                  static_cast<I*>(b.target_entity),
                                  b.target_group,
                                  static_cast<I*>(b.history_offsets),
                                  static_cast<I*>(b.history_entity),
                                  b.history_dt,
                                  p.numeric_dtype == NumericDtype::F32 ? static_cast<f32*>(b.numeric_values) : nullptr,
                                  p.numeric_dtype == NumericDtype::F64 ? static_cast<f64*>(b.numeric_values) : nullptr,
                                  p.s.sizes.tuples - p.s.numeric_total};
    const detail::SimpleOut<I> simple{static_cast<I*>(b.node_offsets), static_cast<I*>(b.edge_offsets), b.node_label,
                                      static_cast<I*>(b.edge_index),
                                      b.edge_index ? static_cast<I*>(b.edge_index) + b.edge_stride : nullptr,
                                      b.edge_label, b.node_value};
    I* entity_offsets = static_cast<I*>(b.entity_offsets);
    Member& m = p.s.members[t];
    u64 entity = m.entity_start, tuple = m.tuple_start, arg = m.arg_start, target = m.target_start,
        history = m.history_start, numeric = m.numeric_start;
    u32* virtual_scratch = m.virtual_scratch.empty() ? nullptr : m.virtual_scratch.data();
    // fills graph g; `graph_counts` are its cell counts (null unless the pass kept them)
    auto fill_one = [&](u64 g, const u32* graph_counts)
    {
        const u32 instance = p.instance_of(g);
        const detail::InstanceView v = detail::instance_view(p.view, instance);
        const detail::GraphTotals& totals = m.totals[g - lo];
        detail::RowLayers layers;
        const detail::RowInput row = p.row_input<Layered, Lanes>(
            g, v, p.starts_stride ? m.action_starts.data() + (g - lo) * p.starts_stride : nullptr,
            m.action_cursors.data(), layers);
        if (Layered && row.layers)
            p.bind_levels(v, row, layers, m.effective);
        entity_offsets[g] = static_cast<I>(entity);
        b.task_ids[g] = p.in.task_ids ? p.in.task_ids[g] : 0;
        if (p.relation_major)
            detail::fill_graph_relation_major<I, Layered>(v, row, g, entity, totals, out, target, virtual_scratch,
                                                          graph_counts, history);
        else
            detail::fill_graph<I, Layered>(v, row, g, tuple, arg, entity, totals, out, target, virtual_scratch,
                                           graph_counts, history, numeric);
        if constexpr (Lanes)
            if (p.simple)
                detail::simple_fill_graph<I>(p.simple_params, simple, out, p.cells_of<Layered>(v, g), graph_counts,
                                             p.relation_major, g,
                                             {entity, totals.entities, tuple, totals.tuples, arg, totals.args,
                                              row.numeric, v.numeric_storage});
        entity += totals.entities;
        tuple += totals.tuples;
        arg += totals.args;
        target += totals.targets;
        if constexpr (Lanes)
            numeric += detail::numeric_slot_count(v);
        if constexpr (Layered)
            history += totals.history_steps;
    };
    const auto cells_of = [&](u64 g)
    { return p.cells_of<Layered>(detail::instance_view(p.view, p.instance_of(g)), g); };
    detail::fill_member<I>(m, {p.graphs, p.R, p.relation_major, p.s.arity.data()}, lo, hi, out.relation_offsets,
                           out.relation_arg_offsets, fill_one, cells_of);
}

/// Throws unless the static capacities and sizes are given exactly when the schema shares its statics.
void check_static_arguments(const EncodingSchema& schema, const StaticShape* capacities, const ExtraSizes* extra)
{
    if (schema.shared_statics() && (!capacities || !extra))
        invalid("the schema shares its statics: the capacity form needs the capacities and the sizes of the static blocks");
    if (!schema.shared_statics() && capacities)
        invalid("the schema repeats its statics: the capacity form takes no capacities of static blocks");
}

/// Whether the batch of `sizes` and `extra` exceeds the capacities.
bool exceeds_capacity(const EncodingSchema& schema, const FlatSizes& sizes, const ExtraSizes& extra,
                      const BatchShape& capacities, const StaticShape* static_capacities)
{
    return sizes.entities > capacities.entities || sizes.args > capacities.args || sizes.targets > capacities.targets ||
           sizes.history > capacities.history ||
           (schema.stores_numeric_values() && extra.numeric > capacities.numeric) ||
           (static_capacities && (extra.statics.instances > static_capacities->instances ||
                                  extra.statics.args > static_capacities->args));
}

/// The distinct instances of the rows of a call, ascending, into s.present.
void collect_instances(const EncodeInput& in, u32 instances, Scratch& s)
{
    s.present.clear();
    const u64 rows = in.states.rows;
    if (rows == 0)
        return;
    if (!in.task_ids)
    {
        s.present.push_back(0);
        return;
    }
    if (s.seen.size() < instances)
        s.seen.assign(instances, 0);
    i32 previous = -1;
    for (u64 g = 0; g < rows; ++g)
    {
        const i32 id = in.task_ids[g];
        if (id == previous)
            continue;
        previous = id;
        if (!s.seen[static_cast<u32>(id)])
        {
            s.seen[static_cast<u32>(id)] = 1;
            s.present.push_back(static_cast<u32>(id));
        }
    }
    std::ranges::sort(s.present);
    for (const u32 id : s.present)
        s.seen[id] = 0;
}

/// Writes the static blocks of the instances in s.present: the arrays of `out` from `static_task_ids` to
/// `static_relations`.
template<class I>
void write_static_blocks(const EncodingSchema& schema, const detail::ContextView& view, const Scratch& s,
                         const FlatBuffers& out)
{
    const std::span<const u32> relations = schema.static_relations();
    const u64 Rs = relations.size();
    I* offsets = static_cast<I*>(out.static_offsets);
    I* arg_offsets = static_cast<I*>(out.static_arg_offsets);
    I* args = static_cast<I*>(out.static_args);
    u64 tuples = 0, positions = 0, block = 0;
    for (const u32 id : s.present)
    {
        const detail::InstanceView v = detail::instance_view(view, id);
        out.static_task_ids[block] = static_cast<i32>(id);
        detail::fill_static_block<I>(v, view.static_relations, view.num_static, block * Rs, tuples, positions, offsets,
                                     arg_offsets, args);
        const detail::StaticTotals totals = detail::static_block_totals(v);
        tuples += totals.tuples;
        positions += totals.args;
        ++block;
    }
    offsets[block * Rs] = static_cast<I>(tuples);
    arg_offsets[block * Rs] = static_cast<I>(positions);
    for (u64 k = 0; k < Rs; ++k)
        static_cast<I*>(out.static_relations)[k] = static_cast<I>(relations[k]);
}
}  // namespace

std::shared_ptr<const FlatEncoder> FlatEncoder::create(rl::TaskTablePtr table, const FlatOptions& options)
{
    if (!table)
        invalid("an encoder needs a task table");
    auto encoder = std::shared_ptr<FlatEncoder>(new FlatEncoder());
    encoder->m_schema = EncodingSchema::create(*table, options);
    encoder->m_table_fingerprint = table->fingerprint();
    encoder->m_context = EncodingContext::create(std::move(table), encoder->m_schema);
    return encoder;
}

std::shared_ptr<const FlatEncoder> FlatEncoder::create(TaskPtr task, const FlatOptions& options)
{
    return create(rl::TaskTable::single(std::move(task)), options);
}

void FlatEncoder::check_layer_inputs(const EncodeInput& in) const
{
    const auto check_layers = [](const LayerInput& l, const char* what, bool rows_share)
    {
        if (l.layers == 0)
            return;
        if (!l.pos || !l.neg)
            invalid(std::string(what) + " need both their positive and negative masks");
        if ((l.derived_pos == nullptr) != (l.derived_neg == nullptr))
            invalid(std::string(what) + " need both their positive and negative derived masks");
        if (l.stride < u64{l.layers} * l.words && !(rows_share && l.stride == 0))
            invalid(std::string("the row stride of the ") + what + " is smaller than their layers");
        if (l.derived_pos && l.derived_stride < u64{l.layers} * l.derived_words &&
            !(rows_share && l.derived_stride == 0))
            invalid(std::string("the derived row stride of the ") + what + " is smaller than their layers");
    };
    if (in.subgoals.layers != 0)
    {
        if (!m_schema->has_subgoals())
            invalid("subgoal layers are given, but the schema has no subgoal relations");
        if (in.subgoals.layers > m_schema->options().max_goal_level)
            invalid(std::to_string(in.subgoals.layers) + " subgoal layers are given, the schema has " +
                    std::to_string(m_schema->options().max_goal_level) + " levels");
        check_layers(in.subgoals, "subgoal layers", true);
    }
    if (in.history.masks.layers != 0)
    {
        if (!m_schema->has_history())
            invalid("history steps are given, but the schema has no history relations");
        if (!in.history.dt)
            invalid("the history steps need their dt");
        check_layers(in.history.masks, "history steps", false);
    }
}

u64 FlatEncoder::graphs_of(const EncodeInput& in) const noexcept
{
    return m_schema->has_horizon() ? in.horizon.graphs : in.states.rows;
}

FlatSizes FlatEncoder::count(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch, ExtraSizes* extra) const
{
    const EncodingContext& context = *m_context;
    Scratch& s = scratch.impl();
    s.counted = false;
    if (extra)
        *extra = {};
    if (m_schema->has_horizon())
        return count_horizon(in, pool, scratch);
    if (in.horizon.graphs != 0 || in.horizon.node_offsets || in.horizon.nodes.rows != 0)
        invalid("a horizon is given, but the schema has no horizon mode");
    const bool relation_major = m_schema->options().layout == Layout::Relation;
    const u64 graphs = in.states.rows;
    const u32 R = static_cast<u32>(m_schema->num_relations());
    const rl::StateBatchView& states = in.states;

    // 1. check what does not depend on a row
    if (states.numeric_words != table().numeric_words())
        invalid("the state rows have " + std::to_string(states.numeric_words) + " numeric words, the table's rows have " +
                std::to_string(table().numeric_words()));
    if (graphs && !states.data)
        invalid("the state buffer is null");
    if (states.stride && states.stride < u64{states.words} + states.numeric_words)
        invalid("the state row stride is smaller than its width");
    if (in.derived && in.derived_stride < context.max_derived_words())
        invalid("the derived row stride is smaller than the derived words of an instance");
    if (in.goal_pos && !m_schema->has_goal_complete())
        invalid("goals are given, but the schema has no goal relations");
    if ((in.goal_pos == nullptr) != (in.goal_neg == nullptr))
        invalid("the goal needs both goal_pos and goal_neg");
    if ((in.goal_derived_pos == nullptr) != (in.goal_derived_neg == nullptr))
        invalid("the derived goal needs both goal_derived_pos and goal_derived_neg");
    if (in.goal_pos && in.goal_stride && in.goal_stride < in.goal_words)
        invalid("the goal row stride is smaller than its width");
    if (in.goal_derived_pos && in.goal_derived_stride && in.goal_derived_stride < in.goal_derived_words)
        invalid("the derived goal row stride is smaller than its width");
    if (in.goal_derived_pos && !in.goal_pos)
        invalid("the derived goal masks need the goal masks over the fluent slots");
    check_layer_inputs(in);
    if (const rl::Expansion* x = in.actions)
    {
        if (!m_schema->options().include_actions)
            invalid("actions are given, but the schema has no action relations");
        if (!x->offsets || !x->schema || !x->binding)
            invalid("the action expansion needs its offsets, schema and binding arrays");
        if (x->overflow())
            invalid("the action expansion overflowed: it has " + std::to_string(x->total) + " rows for a capacity of " +
                    std::to_string(x->capacity) + " and rows of " + std::to_string(x->words) + " words (" +
                    std::to_string(x->words_needed) + " needed)");
        if (x->label_width < context.table().label_width())
            invalid("the action expansion's label width " + std::to_string(x->label_width) +
                    " is below the table's largest action arity " + std::to_string(context.table().label_width()));
    }
    if (R && graphs > std::numeric_limits<u64>::max() / R - 1)
        throw std::overflow_error("mymyr: encoding: the batch has too many cells");
    const bool evaluate = !in.derived && context.max_derived_words() > 0;
    s.derived_stride = in.derived ? in.derived_stride : context.max_derived_words();
    if (evaluate && graphs * s.derived_stride > s.derived_capacity)
    {
        s.derived_capacity = graphs * s.derived_stride;
        s.derived = std::make_unique_for_overwrite<u64[]>(s.derived_capacity);
    }
    const u32 members = members_for(graphs, pool);
    s.members.resize(members);

    // 2. validate, evaluate and count, in parallel over the graphs
    const detail::ContextView view = context.view();
    const Pass pass{in, s, context, view, graphs, members, R, relation_major, evaluate,
                    in.actions ? u64{view.num_schemas} + 1 : 0};
    run(pool, members,
        [&pass](u32 t)
        {
            if (pass.lanes)
            {
                if (pass.layered)
                    count_range<true, true>(pass, t);
                else
                    count_range<false, true>(pass, t);
            }
            else if (pass.layered)
                count_range<true, false>(pass, t);
            else
                count_range<false, false>(pass, t);
        });
    u64 bad_id = k_none, narrow = k_none, bad_bits = k_none, narrow_goal = k_none, bad_goal_bits = k_none;
    u64 bad_slice = k_none, bad_action_order = k_none, bad_action_label = k_none;
    u64 narrow_subgoals = k_none, bad_subgoal_bits = k_none, narrow_history = k_none, bad_history_bits = k_none,
        bad_history_count = k_none;
    for (const Member& m : s.members)
    {
        narrow_subgoals = std::min(narrow_subgoals, m.narrow_subgoals);
        bad_subgoal_bits = std::min(bad_subgoal_bits, m.bad_subgoal_bits);
        narrow_history = std::min(narrow_history, m.narrow_history);
        bad_history_bits = std::min(bad_history_bits, m.bad_history_bits);
        bad_history_count = std::min(bad_history_count, m.bad_history_count);
        bad_slice = std::min(bad_slice, m.bad_slice);
        bad_action_order = std::min(bad_action_order, m.bad_action_order);
        bad_action_label = std::min(bad_action_label, m.bad_action_label);
        bad_id = std::min(bad_id, m.bad_id);
        narrow = std::min(narrow, m.narrow);
        bad_bits = std::min(bad_bits, m.bad_bits);
        narrow_goal = std::min(narrow_goal, m.narrow_goal);
        bad_goal_bits = std::min(bad_goal_bits, m.bad_goal_bits);
    }
    const rl::TaskTable& table = this->table();
    if (bad_id != k_none)
        invalid("task id " + std::to_string(in.task_ids[bad_id]) + " of row " + std::to_string(bad_id) +
                " is outside the table's " + std::to_string(table.size()) + " instances");
    if (narrow != k_none)
        invalid("row " + std::to_string(narrow) + " has " + std::to_string(states.words) +
                " words, its instance needs " + std::to_string(table.instance(pass.instance_of(narrow)).words));
    if (narrow_goal != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_goal));
        invalid("the goal masks of row " + std::to_string(narrow_goal) + " are narrower than its instance needs (" +
                std::to_string(v.words) + " words, " + std::to_string(v.derived_words) + " derived words)");
    }
    if (narrow_subgoals != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_subgoals));
        invalid("the subgoal layers of row " + std::to_string(narrow_subgoals) +
                " are narrower than its instance needs (" + std::to_string(v.words) + " words, " +
                std::to_string(v.derived_words) + " derived words)");
    }
    if (narrow_history != k_none)
    {
        const detail::InstanceView v = detail::instance_view(pass.view, pass.instance_of(narrow_history));
        invalid("the history steps of row " + std::to_string(narrow_history) +
                " are narrower than its instance needs (" + std::to_string(v.words) + " words, " +
                std::to_string(v.derived_words) + " derived words)");
    }
    if (bad_history_count != k_none)
        invalid("the history count of row " + std::to_string(bad_history_count) + " is " +
                std::to_string(in.history.count[bad_history_count]) + ", outside [0, " +
                std::to_string(in.history.masks.layers) + "]");
    if (bad_bits != k_none)
        invalid("row " + std::to_string(bad_bits) + " sets an atom bit its instance does not have");
    if (bad_subgoal_bits != k_none)
        invalid("the subgoal layers of row " + std::to_string(bad_subgoal_bits) +
                " set an atom bit its instance does not have");
    if (bad_history_bits != k_none)
        invalid("the history steps of row " + std::to_string(bad_history_bits) +
                " set an atom bit its instance does not have");
    if (bad_goal_bits != k_none)
        invalid("the goal of row " + std::to_string(bad_goal_bits) + " sets an atom bit its instance does not have");
    if (bad_slice != k_none)
        invalid("the offsets of the action expansion do not give state " + std::to_string(bad_slice) +
                " a slice of its rows");
    if (bad_action_label != k_none)
        invalid("the action rows of state " + std::to_string(bad_action_label) +
                " have a schema or an object its instance does not have, or belong to another state");
    if (bad_action_order != k_none)
        invalid("the action rows of state " + std::to_string(bad_action_order) +
                " are not in canonical order: strictly ascending in (schema, binding)");

    // 3. scan the members, in u64: where each starts in the batch and, relation-major, in every relation
    FlatSizes sizes = detail::scan_members(s, *m_schema, graphs);
    // the static blocks of the instances present
    StaticSizes statics;
    if (m_schema->shared_statics())
    {
        const u64 limit = index_limit(m_schema->options().index);
        collect_instances(in, view.num_instances, s);
        statics.instances = s.present.size();
        for (const u32 id : s.present)
        {
            const detail::StaticTotals totals = detail::static_block_totals(detail::instance_view(view, id));
            statics.tuples += totals.tuples;
            statics.args += totals.args;
        }
        if (statics.tuples > limit || statics.args > limit ||
            statics.instances * m_schema->static_relations().size() + 1 > limit)
            sizes.overflow |= k_overflow_index;
    }
    s.static_sizes = statics;
    if (extra)
        *extra = {s.numeric_total, statics};

    s.sizes = sizes;
    s.counted = true;
    s.counted_input = in;
    return sizes;
}

void FlatEncoder::fill(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
                       EncodeScratch& scratch) const
{
    Scratch& s = scratch.impl();
    if (!s.counted || !same_input(s.counted_input, in) || s.sizes.entities != sizes.entities ||
        s.sizes.tuples != sizes.tuples || s.sizes.args != sizes.args || s.sizes.targets != sizes.targets ||
        s.sizes.history != sizes.history)
        invalid("fill needs the scratch and the sizes of count() on the same input");
    if (sizes.overflow != 0)
        invalid("the batch does not fit: its sizes overflow");
    const u32 members = static_cast<u32>(s.members.size());
    if (pool.size() < members)
        invalid("fill needs the thread pool of count()");
    const EncodingSchema& schema = *m_schema;
    const FlatOptions& options = schema.options();
    const u64 graphs = graphs_of(in);
    const bool relations = schema.emits_relations();
    if (!out.task_ids || !out.num_objects || !out.entity_offsets)
        invalid("a destination array of the batch is null");
    if (relations)
    {
        if (!out.relation_offsets || !out.relation_arg_offsets)
            invalid("a destination array of the batch is null");
        if (sizes.args > 0 && !out.relation_args)
            invalid("relation_args is null, the batch has tuple arguments");
        if ((out.relation_counts != nullptr) != options.emit_counts)
            invalid(options.emit_counts ? "relation_counts is null, the schema stores it"
                                        : "relation_counts must be null, the schema omits it");
    }
    else if (out.relation_offsets || out.relation_arg_offsets || out.relation_args || out.relation_counts)
        invalid("the relation arrays must be null, the schema does not store them");
    if (schema.emits_simple())
    {
        if (!out.node_offsets || !out.edge_offsets)
            invalid("node_offsets or edge_offsets is null, the schema stores the Simple graph");
        if (sizes.entities + sizes.tuples > 0 && !out.node_label)
            invalid("node_label is null, the batch has nodes");
        if (sizes.args > 0 && (!out.edge_index || !out.edge_label))
            invalid("edge_index or edge_label is null, the batch has edges");
        if (out.edge_stride < 2 * sizes.args)
            invalid("edge_stride is below the batch's edges");
    }
    else if (out.node_offsets || out.node_label || out.edge_offsets || out.edge_index || out.edge_label)
        invalid("the Simple arrays must be null, the schema does not store them");
    if ((out.entity_role != nullptr) != !schema.implicit_entity_arrays() ||
        (out.entity_ref != nullptr) != !schema.implicit_entity_arrays())
        invalid(schema.implicit_entity_arrays() ? "entity_role and entity_ref must be null, the schema omits them"
                                                : "entity_role and entity_ref are null, the schema stores them");
    if ((out.goal_complete != nullptr) != schema.has_goal_complete())
        invalid(schema.has_goal_complete() ? "goal_complete is null, the schema stores it"
                                           : "goal_complete must be null, the schema has no goal relations");
    if ((out.target_offsets != nullptr) != schema.has_targets() ||
        (!schema.has_targets() && (out.target_entity != nullptr || out.target_group != nullptr)) ||
        (sizes.targets > 0 && (!out.target_entity || !out.target_group)))
        invalid(schema.has_targets() ? "a target array is null, the schema stores them"
                                     : "the target arrays must be null, the schema has no targets");
    if ((out.history_offsets != nullptr) != schema.has_history() ||
        (!schema.has_history() && (out.history_entity != nullptr || out.history_dt != nullptr)) ||
        (sizes.history > 0 && (!out.history_entity || !out.history_dt)))
        invalid(schema.has_history() ? "a history array is null, the schema stores them"
                                     : "the history arrays must be null, the schema has no history");
    if (schema.stores_numeric_values() ? (s.numeric_total > 0 && !out.numeric_values) : out.numeric_values != nullptr)
        invalid(schema.stores_numeric_values() ? "numeric_values is null, the batch has numeric tuples"
                                               : "numeric_values must be null, the schema does not store it");
    if (schema.stores_node_value() ? (sizes.entities + sizes.tuples > 0 && !out.node_value) : out.node_value != nullptr)
        invalid(schema.stores_node_value() ? "node_value is null, the batch has nodes"
                                           : "node_value must be null, the schema does not store it");
    const bool shared = schema.shared_statics();
    if (shared)
    {
        const StaticSizes& statics = s.static_sizes;
        if (!out.static_offsets || !out.static_arg_offsets || (statics.instances > 0 && !out.static_task_ids) ||
            (statics.args > 0 && !out.static_args) || (!schema.static_relations().empty() && !out.static_relations))
            invalid("a static block array is null, the schema shares its statics");
    }
    else if (out.static_task_ids || out.static_offsets || out.static_arg_offsets || out.static_args ||
             out.static_relations)
        invalid("the static block arrays must be null, the schema repeats its statics");
    const u64 cells = graphs * schema.num_relations();
    // an encoding without the relation arrays writes its tuples into scratch, from which the Simple graph is built
    FlatBuffers work = out;
    const usize index_bytes = options.index == IndexType::I64 ? 8 : 4;
    if (!relations)
    {
        const u64 bytes = (2 * (cells + 1) + sizes.args) * index_bytes;
        if (bytes > s.relation_scratch_bytes)
        {
            s.relation_scratch = std::make_unique_for_overwrite<std::byte[]>(bytes);
            s.relation_scratch_bytes = bytes;
        }
        work.relation_offsets = s.relation_scratch.get();
        work.relation_arg_offsets = s.relation_scratch.get() + (cells + 1) * index_bytes;
        work.relation_args = s.relation_scratch.get() + 2 * (cells + 1) * index_bytes;
    }
    if (schema.has_horizon())
        fill_horizon(in, sizes, work, pool, scratch);
    else
    {
        const bool evaluate = !in.derived && m_context->max_derived_words() > 0;
        const detail::ContextView view = m_context->view();
        const Pass pass{in,
                        s,
                        *m_context,
                        view,
                        graphs,
                        members,
                        static_cast<u32>(schema.num_relations()),
                        options.layout == Layout::Relation,
                        evaluate,
                        in.actions ? u64{view.num_schemas} + 1 : 0};
        run(pool, members,
            [&pass, &work, wide = options.index == IndexType::I64](u32 t)
            {
                if (pass.lanes)
                {
                    if (pass.layered)
                        wide ? fill_range<i64, true, true>(pass, work, t) : fill_range<i32, true, true>(pass, work, t);
                    else
                        wide ? fill_range<i64, false, true>(pass, work, t)
                             : fill_range<i32, false, true>(pass, work, t);
                }
                else if (pass.layered)
                    wide ? fill_range<i64, true, false>(pass, work, t) : fill_range<i32, true, false>(pass, work, t);
                else
                    wide ? fill_range<i64, false, false>(pass, work, t) : fill_range<i32, false, false>(pass, work, t);
            });
    }
    auto close = [&]<class I>(I)
    {
        static_cast<I*>(out.entity_offsets)[graphs] = static_cast<I>(sizes.entities);
        if (out.target_offsets)
            static_cast<I*>(out.target_offsets)[graphs] = static_cast<I>(sizes.targets);
        if (out.history_offsets)
            static_cast<I*>(out.history_offsets)[graphs] = static_cast<I>(sizes.history);
        static_cast<I*>(work.relation_offsets)[cells] = static_cast<I>(sizes.tuples);
        static_cast<I*>(work.relation_arg_offsets)[cells] = static_cast<I>(sizes.args);
        if (out.node_offsets)
        {
            static_cast<I*>(out.node_offsets)[graphs] = static_cast<I>(sizes.entities + sizes.tuples);
            static_cast<I*>(out.edge_offsets)[graphs] = static_cast<I>(2 * sizes.args);
        }
    };
    if (options.index == IndexType::I64)
        close(i64{});
    else
        close(i32{});
    if (shared)
    {
        const detail::ContextView view = m_context->view();
        if (options.index == IndexType::I64)
            write_static_blocks<i64>(schema, view, s, out);
        else
            write_static_blocks<i32>(schema, view, s, out);
    }
}

FlatSizes FlatEncoder::encode_into(const EncodeInput& in, const BatchShape& capacities, const FlatBuffers& out,
                                   ThreadPool& pool, EncodeScratch& scratch, const StaticShape* static_capacities,
                                   ExtraSizes* extra) const
{
    check_static_arguments(*m_schema, static_capacities, extra);
    if (graphs_of(in) > capacities.graphs)
        invalid("the batch has more graphs than the output capacity");
    ExtraSizes true_extra;
    FlatSizes sizes = count(in, pool, scratch, &true_extra);
    if (extra)
        *extra = true_extra;
    if (exceeds_capacity(*m_schema, sizes, true_extra, capacities, static_capacities))
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

FlatEncoder::Allocation FlatEncoder::allocate(u64 graphs, u64 entities, u64 args, u64 targets, u64 history,
                                              u64 numeric, u64 nodes, u64 edges, const StaticShape& statics) const
{
    const IndexType index = m_schema->options().index;
    const u64 R = m_schema->num_relations();
    const rl::DType itype = index == IndexType::I64 ? rl::DType::I64 : rl::DType::I32;
    constexpr usize absent = std::numeric_limits<usize>::max();
    rl::ArrayBundle::Builder builder;
    usize count = 0;
    auto reserve = [&](const char* name, rl::DType dtype, const rl::ArrayShape& shape)
    {
        builder.add_uninitialized(name, dtype, shape);
        return count++;
    };
    const usize task_ids = reserve("task_ids", rl::DType::I32, shape1(graphs));
    const usize num_objects = reserve("num_objects", itype, shape1(graphs));
    const usize entity_offsets = reserve("entity_offsets", itype, shape1(graphs + 1));
    usize entity_role = absent, entity_ref = absent, relation_counts = absent, goal_complete = absent,
          target_offsets = absent, target_entity = absent, target_group = absent, history_offsets = absent,
          history_entity = absent, history_dt = absent, relation_offsets = absent, relation_arg_offsets = absent,
          relation_args = absent, node_offsets = absent, node_label = absent, edge_offsets = absent,
          edge_index = absent, edge_label = absent, numeric_values = absent, node_value = absent,
          static_task_ids = absent, static_offsets = absent, static_arg_offsets = absent, static_args = absent,
          static_relations = absent;
    const bool relations = m_schema->emits_relations(), simple = m_schema->emits_simple();
    if (!m_schema->implicit_entity_arrays())
    {
        entity_role = reserve("entity_role", rl::DType::U8, shape1(entities));
        entity_ref = reserve("entity_ref", itype, shape1(entities));
    }
    if (relations)
    {
        if (m_schema->emits_counts())
            relation_counts = reserve("relation_counts", itype, shape2(graphs, R));
        relation_offsets = reserve("relation_offsets", itype, shape1(graphs * R + 1));
        relation_arg_offsets = reserve("relation_arg_offsets", itype, shape1(graphs * R + 1));
        relation_args = reserve("relation_args", itype, shape1(args));
    }
    if (m_schema->has_goal_complete())
        goal_complete = reserve("goal_complete", rl::DType::U8, shape1(graphs));
    if (m_schema->has_targets())
    {
        target_offsets = reserve("target_offsets", itype, shape1(graphs + 1));
        target_entity = reserve("target_entity", itype, shape1(targets));
        target_group = reserve("target_group", rl::DType::U8, shape1(targets));
    }
    if (m_schema->has_history())
    {
        history_offsets = reserve("history_offsets", itype, shape1(graphs + 1));
        history_entity = reserve("history_entity", itype, shape1(history));
        history_dt = reserve("history_dt", rl::DType::I32, shape1(history));
    }
    if (simple)
    {
        node_offsets = reserve("node_offsets", itype, shape1(graphs + 1));
        node_label = reserve("node_label", rl::DType::I32, shape1(nodes));
        edge_offsets = reserve("edge_offsets", itype, shape1(graphs + 1));
        edge_index = reserve("edge_index", itype, shape2(2, edges));
        edge_label = reserve("edge_label", rl::DType::I32, shape1(edges));
    }
    const bool wide_values = m_schema->options().numeric_dtype == NumericDtype::F64;
    if (m_schema->stores_numeric_values())
        numeric_values = reserve("numeric_values", wide_values ? rl::DType::F64 : rl::DType::F32, shape1(numeric));
    if (m_schema->stores_node_value())
        node_value = reserve("node_value", rl::DType::F32, shape1(nodes));
    if (m_schema->shared_statics())
    {
        const u64 Rs = m_schema->static_relations().size();
        static_task_ids = reserve("static_task_ids", rl::DType::I32, shape1(statics.instances));
        static_offsets = reserve("static_offsets", itype, shape1(statics.instances * Rs + 1));
        static_arg_offsets = reserve("static_arg_offsets", itype, shape1(statics.instances * Rs + 1));
        static_args = reserve("static_args", itype, shape1(statics.args));
        static_relations = reserve("static_relations", itype, shape1(Rs));
    }
    builder.scalar("version", 1);
    builder.scalar("num_graphs", static_cast<i64>(graphs));
    builder.scalar("num_relations", static_cast<i64>(R));
    builder.scalar("layout", m_schema->options().layout == Layout::Relation ? 1 : 0);
    builder.scalar("index_dtype", index == IndexType::I64 ? 1 : 0);
    builder.scalar("schema_fingerprint", std::bit_cast<i64>(m_schema->fingerprint()));
    builder.scalar("table_fingerprint", std::bit_cast<i64>(m_table_fingerprint));
    builder.scalar("statics", m_schema->shared_statics() ? 1 : 0);
    if (m_schema->stores_numeric_values())
    {
        builder.scalar("numeric_dtype", wide_values ? 1 : 0);
        builder.scalar("numeric_relations", static_cast<i64>(R - m_schema->first_numeric_relation()));
    }
    if (simple)
    {
        builder.scalar("simple_node_labels", m_schema->simple_node_labels());
        builder.scalar("simple_edge_labels", m_schema->simple_edge_labels());
    }
    Allocation allocation{std::make_shared<rl::ArrayBundle>(builder.build()), {}};
    rl::ArrayBundle& bundle = *allocation.bundle;
    auto at = [&](usize i) -> void* { return i == absent ? nullptr : bundle.mutable_data(bundle.arrays()[i]); };
    FlatBuffers& out = allocation.buffers;
    out.task_ids = static_cast<i32*>(at(task_ids));
    out.num_objects = at(num_objects);
    out.entity_offsets = at(entity_offsets);
    out.entity_role = static_cast<u8*>(at(entity_role));
    out.entity_ref = at(entity_ref);
    out.relation_counts = at(relation_counts);
    out.relation_offsets = at(relation_offsets);
    out.relation_arg_offsets = at(relation_arg_offsets);
    out.relation_args = at(relation_args);
    out.goal_complete = static_cast<u8*>(at(goal_complete));
    out.target_offsets = at(target_offsets);
    out.target_entity = at(target_entity);
    out.target_group = static_cast<u8*>(at(target_group));
    out.history_offsets = at(history_offsets);
    out.history_entity = at(history_entity);
    out.history_dt = static_cast<i32*>(at(history_dt));
    out.node_offsets = at(node_offsets);
    out.node_label = static_cast<i32*>(at(node_label));
    out.edge_offsets = at(edge_offsets);
    out.edge_index = at(edge_index);
    out.edge_stride = edges;
    out.edge_label = static_cast<i32*>(at(edge_label));
    out.numeric_values = at(numeric_values);
    out.node_value = static_cast<f32*>(at(node_value));
    out.static_task_ids = static_cast<i32*>(at(static_task_ids));
    out.static_offsets = at(static_offsets);
    out.static_arg_offsets = at(static_arg_offsets);
    out.static_args = at(static_args);
    out.static_relations = at(static_relations);
    return allocation;
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in, ThreadPool& pool,
                                                           EncodeScratch& scratch) const
{
    ExtraSizes extra;
    const FlatSizes sizes = count(in, pool, scratch, &extra);
    const StaticSizes& statics = extra.statics;
    if (sizes.overflow & k_overflow_index)
    {
        const IndexType index = m_schema->options().index;
        const u64 graphs = graphs_of(in);
        const u64 R = m_schema->num_relations();
        detail::check_index_range(sizes.entities, index, "entities");
        detail::check_index_range(sizes.tuples, index, "tuples");
        detail::check_index_range(sizes.args, index, "tuple arguments");
        detail::check_index_range(sizes.targets, index, "targets");
        detail::check_index_range(sizes.history, index, "history steps");
        detail::check_index_range(graphs * R + 1, index, "relation cells");
        detail::check_index_range(graphs + 1, index, "graphs");
        if (m_schema->emits_simple())
        {
            detail::check_index_range(sizes.entities + sizes.tuples, index, "nodes");
            detail::check_index_range(2 * sizes.args, index, "edges");
        }
        detail::check_index_range(statics.tuples, index, "static tuples");
        detail::check_index_range(statics.args, index, "static tuple arguments");
        detail::check_index_range(statics.instances * m_schema->static_relations().size() + 1, index,
                                  "static block cells");
    }
    Allocation allocation = allocate(graphs_of(in), sizes.entities, sizes.args, sizes.targets, sizes.history,
                                     extra.numeric, sizes.entities + sizes.tuples, 2 * sizes.args,
                                     {statics.instances, statics.args});
    fill(in, sizes, allocation.buffers, pool, scratch);
    return std::move(allocation.bundle);
}

void FlatEncoder::clear_capacity_tails(const FlatBuffers& out, const FlatSizes& sizes, const ExtraSizes& extra,
                                       const BatchShape& capacities, const StaticShape& static_capacities) const
{
    const usize index_bytes = m_schema->options().index == IndexType::I64 ? 8 : 4;
    const StaticSizes& statics = extra.statics;
    auto zero = [](void* data, u64 from, u64 to, usize width)
    { std::memset(static_cast<std::byte*>(data) + from * width, 0, (to - from) * width); };
    if (out.relation_args)
        zero(out.relation_args, sizes.args, capacities.args, index_bytes);
    if (out.numeric_values)
        zero(out.numeric_values, extra.numeric, capacities.numeric,
             m_schema->options().numeric_dtype == NumericDtype::F64 ? 8 : 4);
    if (out.node_value)
        zero(out.node_value, sizes.entities + sizes.tuples, capacities.entities + capacities.args, 4);
    if (out.node_label)
    {
        zero(out.node_label, sizes.entities + sizes.tuples, capacities.entities + capacities.args, 4);
        zero(out.edge_label, 2 * sizes.args, 2 * capacities.args, 4);
        zero(out.edge_index, 2 * sizes.args, 2 * capacities.args, index_bytes);
        zero(static_cast<std::byte*>(out.edge_index) + out.edge_stride * index_bytes, 2 * sizes.args,
             2 * capacities.args, index_bytes);
    }
    if (out.entity_role)
    {
        zero(out.entity_role, sizes.entities, capacities.entities, 1);
        zero(out.entity_ref, sizes.entities, capacities.entities, index_bytes);
    }
    if (out.target_entity)
    {
        zero(out.target_entity, sizes.targets, capacities.targets, index_bytes);
        zero(out.target_group, sizes.targets, capacities.targets, 1);
    }
    if (out.history_entity)
    {
        zero(out.history_entity, sizes.history, capacities.history, index_bytes);
        zero(out.history_dt, sizes.history, capacities.history, 4);
    }
    if (out.static_task_ids)
    {
        zero(out.static_task_ids, statics.instances, static_capacities.instances, 4);
        zero(out.static_args, statics.args, static_capacities.args, index_bytes);
        // the blocks past the batch's are empty: their offsets repeat the last entry
        const u64 Rs = m_schema->static_relations().size();
        auto repeat_last = [&]<class I>(void* offsets, I)
        {
            I* data = static_cast<I*>(offsets);
            std::fill(data + statics.instances * Rs + 1, data + static_capacities.instances * Rs + 1,
                      data[statics.instances * Rs]);
        };
        for (void* offsets : {out.static_offsets, out.static_arg_offsets})
            if (m_schema->options().index == IndexType::I64)
                repeat_last(offsets, i64{});
            else
                repeat_last(offsets, i32{});
    }
}

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode_capacity(const EncodeInput& in,
                                                                    const BatchShape& capacities, ThreadPool& pool,
                                                                    EncodeScratch& scratch, FlatSizes& sizes,
                                                                    const StaticShape* static_capacities,
                                                                    ExtraSizes* extra) const
{
    check_static_arguments(*m_schema, static_capacities, extra);
    if (graphs_of(in) > capacities.graphs)
        invalid("the batch has more graphs than the output capacity");
    ExtraSizes true_extra;
    sizes = count(in, pool, scratch, &true_extra);
    if (extra)
        *extra = true_extra;
    if (exceeds_capacity(*m_schema, sizes, true_extra, capacities, static_capacities))
        sizes.overflow |= k_overflow_capacity;
    const u64 graphs = graphs_of(in);
    const bool fits = !sizes.overflowed();
    const StaticShape static_shape = static_capacities ? *static_capacities : StaticShape{};
    Allocation allocation = allocate(graphs, capacities.entities, capacities.args, capacities.targets,
                                     capacities.history, capacities.numeric, capacities.entities + capacities.args,
                                     2 * capacities.args, static_shape);
    const FlatBuffers out = allocation.buffers;
    const std::shared_ptr<rl::ArrayBundle>& bundle = allocation.bundle;
    if (fits)
    {
        fill(in, sizes, out, pool, scratch);
        clear_capacity_tails(out, sizes, true_extra, capacities, static_shape);
    }
    else
        for (const rl::ArrayInfo& info : bundle->arrays())
            std::memset(bundle->mutable_data(info), 0, info.elements() * rl::dtype_bytes(info.dtype));
    return bundle;
}
}  // namespace mymyr::encoding
