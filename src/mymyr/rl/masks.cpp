#include "mymyr/rl/masks.hpp"

#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mymyr::rl
{
namespace
{
void validate(const TaskSuite& suite, StateBatchView rows, const i32* ids)
{
    if (rows.rows && !rows.data)
        throw std::invalid_argument("mymyr: action masks: null state buffer");
    if (rows.stride && rows.stride < u64{rows.words} + rows.numeric_words)
        throw std::invalid_argument("mymyr: action masks: row stride smaller than the row width");
    if (rows.numeric_words != suite.numeric_words())
        throw std::invalid_argument("mymyr: action masks: states have the wrong number of numeric words");
    if (rows.rows && !ids && suite.size() > 1)
        throw std::invalid_argument("mymyr: action masks: a batch of several instances needs task ids");
    suite.check_task_ids(ids, rows.rows);
    for (u64 i = 0; i < rows.rows; ++i)
    {
        const Task& task = suite.task_of(ids, i);
        if (rows.words < task.words())
            throw std::invalid_argument("mymyr: action masks: state rows truncate an instance's atom words");
        const u32 limit = task.atoms().fluent_slots();
        if ((limit & 63) && (rows.row(i)[limit >> 6] >> (limit & 63)))
            throw std::invalid_argument("mymyr: action masks: state row sets atom slots its instance does not have");
    }
}

template<class F>
void each_row(const TaskSuite& suite, StateBatchView rows, const i32* ids, ThreadPool* pool, F&& f)
{
    validate(suite, rows, ids);
    auto run = [&](u32 member)
    {
        const auto [begin, end] = ThreadPool::slice(rows.rows, member, pool ? pool->size() : 1);
        std::vector<WorkspaceLease> leases(suite.size());
        std::vector<ObjectId> objects;
        std::vector<u64> mask(bits::words_for(suite.max_objects()));
        for (u64 i = begin; i < end; ++i)
        {
            const u32 g = ids ? static_cast<u32>(ids[i]) : 0;
            const Task& task = *suite.task(g);
            if (!leases[g])
                leases[g] = task.workspace();
            Successors& succ = leases[g]->successors();
            const StateView state{rows.row(i), task.words(), rows.numeric_words ? rows.row(i) + rows.words : nullptr,
                                  task.numeric_words()};
            f(i, succ, state, objects, mask);
        }
    };
    if (pool)
        pool->run(run);
    else
        run(0);
}
}  // namespace

void schema_masks(const TaskSuite& suite, StateBatchView rows, const i32* ids, u8* out, ThreadPool* pool)
{
    const u32 S = suite.max_schemas();
    if (rows.rows && S && !out)
        throw std::invalid_argument("mymyr: schema_masks: null output buffer");
    each_row(suite, rows, ids, pool, [&](u64 i, Successors& succ, StateView state, auto&, auto&)
    {
        succ.prepare_masks(state);
        for (u32 s = 0; s < S; ++s)
            out[i * S + s] = succ.schema_applicable(s);
    });
}

void parameter_masks(const TaskSuite& suite, StateBatchView rows, const i32* ids, const i32* schema,
                     const i32* prefix, u32 stride, u32 depth, u8* out, u32 O, ThreadPool* pool)
{
    if (O < suite.max_objects())
        throw std::invalid_argument("mymyr: parameter_masks: num_objects is below the suite's maximum");
    if (stride < depth || (rows.rows && depth && !prefix))
        throw std::invalid_argument("mymyr: parameter_masks: prefix needs at least depth columns");
    if (rows.rows && (!schema || (O && !out)))
        throw std::invalid_argument("mymyr: parameter_masks: null schema or output buffer");
    each_row(suite, rows, ids, pool, [&](u64 i, Successors& succ, StateView state, auto& objects, auto& mask)
    {
        if (O)
            std::fill_n(out + i * O, O, u8{0});
        if (schema[i] < 0 || static_cast<u32>(schema[i]) >= succ.num_schemas() ||
            depth >= succ.arity(static_cast<u32>(schema[i])))
            return;
        objects.resize(depth);
        for (u32 d = 0; d < depth; ++d)
            objects[d] = ObjectId{static_cast<u32>(prefix[i * stride + d])};
        succ.prepare_masks(state);
        succ.parameter_mask(static_cast<u32>(schema[i]), objects, depth, mask.data());
        for (u32 o = 0; o < succ.engine().compiled().num_objects; ++o)
            out[i * O + o] = bits::test(mask.data(), static_cast<u32>(mask.size()), o);
    });
}

void is_applicable(const TaskSuite& suite, StateBatchView rows, const i32* ids, const i32* schema,
                   const i32* binding, u32 width, u8* out, ThreadPool* pool)
{
    if (rows.rows && (!schema || !out || (width && !binding)))
        throw std::invalid_argument("mymyr: is_applicable: null label or output buffer");
    each_row(suite, rows, ids, pool, [&](u64 i, Successors& succ, StateView state, auto& objects, auto&)
    {
        out[i] = 0;
        if (schema[i] < 0 || static_cast<u32>(schema[i]) >= succ.num_schemas())
            return;
        const u32 s = static_cast<u32>(schema[i]), arity = succ.arity(s);
        if (width < arity)
            throw std::invalid_argument("mymyr: is_applicable: binding width must be at least the schema arity");
        objects.resize(arity);
        for (u32 d = 0; d < arity; ++d)
            objects[d] = ObjectId{static_cast<u32>(binding[i * width + d])};
        out[i] = succ.is_applicable(state, {SchemaId{s}, objects});
    });
}
}  // namespace mymyr::rl
