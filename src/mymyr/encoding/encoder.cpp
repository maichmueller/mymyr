#include "mymyr/encoding/encoder.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::encoding
{
namespace detail
{
void check_index_range(u64 value, IndexType index, const char* what)
{
    const bool wide = index == IndexType::I64;
    const u64 limit = wide ? static_cast<u64>(std::numeric_limits<i64>::max()) : static_cast<u64>(std::numeric_limits<i32>::max());
    if (value <= limit)
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
    if (o.layout != Layout::Graph)
        unsupported("layout='relation'");
}

/// Where a graph's entities, tuples and arguments start in the batch.
struct GraphBase
{
    u64 entity, tuple, arg;
};

std::vector<i64> shape1(u64 n) { return {static_cast<i64>(n)}; }
std::vector<i64> shape2(u64 n, u64 m) { return {static_cast<i64>(n), static_cast<i64>(m)}; }

template<class I>
void fill(const detail::ContextView& view, const EncodeInput& in, const std::vector<u32>& instance_of,
          const std::vector<const u64*>& derived, const std::vector<detail::GraphTotals>& totals,
          const std::vector<GraphBase>& bases, u64 graphs, rl::ArrayBundle& bundle)
{
    const detail::GraphOut<I> out{static_cast<I*>(bundle.mutable_data("num_objects")),
                                  static_cast<I*>(bundle.mutable_data("relation_counts")),
                                  static_cast<I*>(bundle.mutable_data("relation_offsets")),
                                  static_cast<I*>(bundle.mutable_data("relation_arg_offsets")),
                                  static_cast<I*>(bundle.mutable_data("relation_args")),
                                  static_cast<u8*>(bundle.mutable_data("entity_role")),
                                  static_cast<I*>(bundle.mutable_data("entity_ref"))};
    I* entity_offsets = static_cast<I*>(bundle.mutable_data("entity_offsets"));
    for (u64 g = 0; g < graphs; ++g)
    {
        const detail::InstanceView v = detail::instance_view(view, instance_of[g]);
        entity_offsets[g] = static_cast<I>(bases[g].entity);
        detail::fill_graph<I>(v, {in.states.row(g), derived[g]}, g, bases[g].tuple, bases[g].arg, bases[g].entity, totals[g], out);
    }
    const u64 R = view.num_relations;
    entity_offsets[graphs] = static_cast<I>(graphs ? bases[graphs - 1].entity + totals[graphs - 1].entities : 0);
    out.relation_offsets[graphs * R] = static_cast<I>(graphs ? bases[graphs - 1].tuple + totals[graphs - 1].tuples : 0);
    out.relation_arg_offsets[graphs * R] = static_cast<I>(graphs ? bases[graphs - 1].arg + totals[graphs - 1].args : 0);
}
}  // namespace

std::shared_ptr<const FlatEncoder> FlatEncoder::create(rl::TaskTablePtr table, const FlatOptions& options)
{
    if (!table)
        throw std::invalid_argument("mymyr: encoding: an encoder needs a task table");
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

std::shared_ptr<const rl::ArrayBundle> FlatEncoder::encode(const EncodeInput& in) const
{
    const rl::TaskTable& table = this->table();
    const EncodingContext& context = *m_context;
    const IndexType index = m_schema->options().index;
    const u64 graphs = in.states.rows;
    const u64 R = m_schema->num_relations();
    const rl::StateBatchView& states = in.states;

    // 1. check the input
    if (states.numeric_words != 0)
        throw std::invalid_argument("mymyr: encoding: numeric state rows are not supported");
    if (graphs && !states.data)
        throw std::invalid_argument("mymyr: encoding: the state buffer is null");
    if (states.stride && states.stride < states.words)
        throw std::invalid_argument("mymyr: encoding: the state row stride is smaller than its width");
    table.check_task_ids(in.task_ids, graphs);
    if (in.derived && in.derived_stride < context.max_derived_words())
        throw std::invalid_argument("mymyr: encoding: the derived row stride is smaller than the derived words of an instance");
    const detail::ContextView view = context.view();
    std::vector<u32> instance_of(graphs);
    bool evaluate = false;
    for (u64 g = 0; g < graphs; ++g)
    {
        const u32 instance = in.task_ids ? static_cast<u32>(in.task_ids[g]) : 0;
        instance_of[g] = instance;
        const u32 words = static_cast<u32>(table.instance(instance).words);
        if (states.words < words)
            throw std::invalid_argument("mymyr: encoding: row " + std::to_string(g) + " has " + std::to_string(states.words) +
                                        " words, its instance needs " + std::to_string(words));
        evaluate = evaluate || (!in.derived && context.needs_derived(instance));
    }

    // 2. the derived words of the rows whose instance reads them
    const u64 derived_stride = in.derived ? in.derived_stride : context.max_derived_words();
    std::vector<u64> scratch;
    if (evaluate)
    {
        scratch.assign(graphs * derived_stride, 0);
        for (u64 g = 0; g < graphs; ++g)
        {
            const u32 instance = instance_of[g];
            if (!context.needs_derived(instance))
                continue;
            const detail::InstanceView v = detail::instance_view(view, instance);
            if (detail::has_foreign_atom_bits(v, states.row(g)))
                throw std::invalid_argument("mymyr: encoding: row " + std::to_string(g) +
                                            " sets an atom bit its instance does not have");
            const Task& task = *table.instance(instance).task;
            const WorkspaceLease workspace = task.workspace();
            auto& engine = workspace->engine();
            engine.set_state(states.row(g), v.words);
            engine.build_view();
            workspace->axioms().evaluate();
            std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), scratch.begin() + static_cast<std::ptrdiff_t>(g * derived_stride));
        }
    }
    std::vector<const u64*> derived(graphs, nullptr);
    for (u64 g = 0; g < graphs; ++g)
        if (context.needs_derived(instance_of[g]))
            derived[g] = in.derived ? in.derived + g * derived_stride : scratch.data() + g * derived_stride;

    // 3. count
    std::vector<detail::GraphTotals> totals(graphs);
    for (u64 g = 0; g < graphs; ++g)
    {
        totals[g] = detail::count_graph(detail::instance_view(view, instance_of[g]), {states.row(g), derived[g]}, nullptr);
        if (totals[g].flags & detail::k_graph_bad_row)
            throw std::invalid_argument("mymyr: encoding: row " + std::to_string(g) + " sets an atom bit its instance does not have");
    }

    // 4. scan, in u64, and check the index range before anything is allocated
    std::vector<GraphBase> bases(graphs);
    u64 entities = 0, tuples = 0, args = 0;
    for (u64 g = 0; g < graphs; ++g)
    {
        bases[g] = {entities, tuples, args};
        entities += totals[g].entities;
        tuples += totals[g].tuples;
        args += totals[g].args;
    }
    if (R && graphs > std::numeric_limits<u64>::max() / R - 1)
        throw std::overflow_error("mymyr: encoding: the batch has too many cells");
    detail::check_index_range(entities, index, "entities");
    detail::check_index_range(tuples, index, "tuples");
    detail::check_index_range(args, index, "tuple arguments");
    detail::check_index_range(graphs * R + 1, index, "relation cells");
    detail::check_index_range(graphs + 1, index, "graphs");

    // 5. allocate
    const rl::DType itype = index == IndexType::I64 ? rl::DType::I64 : rl::DType::I32;
    rl::ArrayBundle::Builder builder;
    builder.add_uninitialized("task_ids", rl::DType::I32, shape1(graphs));
    builder.add_uninitialized("num_objects", itype, shape1(graphs));
    builder.add_uninitialized("entity_offsets", itype, shape1(graphs + 1));
    builder.add_uninitialized("entity_role", rl::DType::U8, shape1(entities));
    builder.add_uninitialized("entity_ref", itype, shape1(entities));
    builder.add_uninitialized("relation_counts", itype, shape2(graphs, R));
    builder.add_uninitialized("relation_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_arg_offsets", itype, shape1(graphs * R + 1));
    builder.add_uninitialized("relation_args", itype, shape1(args));
    builder.scalar("version", 1);
    builder.scalar("num_graphs", static_cast<i64>(graphs));
    builder.scalar("num_relations", static_cast<i64>(R));
    builder.scalar("layout", 0);
    builder.scalar("index_dtype", index == IndexType::I64 ? 1 : 0);
    builder.scalar("schema_fingerprint", std::bit_cast<i64>(m_schema->fingerprint()));
    builder.scalar("statics", 0);
    auto bundle = std::make_shared<rl::ArrayBundle>(builder.build());

    // 6. fill
    i32* task_ids = static_cast<i32*>(bundle->mutable_data("task_ids"));
    for (u64 g = 0; g < graphs; ++g)
        task_ids[g] = in.task_ids ? in.task_ids[g] : 0;
    if (index == IndexType::I64)
        fill<i64>(view, in, instance_of, derived, totals, bases, graphs, *bundle);
    else
        fill<i32>(view, in, instance_of, derived, totals, bases, graphs, *bundle);
    return bundle;
}
}  // namespace mymyr::encoding
