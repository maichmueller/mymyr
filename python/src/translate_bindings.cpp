#include "translate_bindings.hpp"

#include "arrays.hpp"
#include "device_hooks.hpp"
#include "py_pool.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mymyr::python
{
using namespace nb::literals;
namespace
{
void init_map(PySlotMap& map, nb::handle source, nb::handle target, bool table)
{
    map.source = nb::borrow(source);
    map.target = nb::borrow(target);
    if (table)
    {
        const auto a = table_of(source), b = table_of(target);
        if (a.table->size() != b.table->size())
            throw nb::value_error("mymyr: translate: tables must have the same instance count");
        map.source_words = a.table->words();
        map.target_words = b.table->words();
        map.source_numeric = a.table->numeric_words();
        map.target_numeric = b.table->numeric_words();
        for (u32 i = 0; i < a.table->size(); ++i)
        {
            map.sources.push_back(a.table->task(i));
            map.targets.push_back(b.table->task(i));
        }
    }
    else
    {
        map.sources.push_back(owner_of(source).core->task);
        map.targets.push_back(owner_of(target).core->task);
    }
    nb::gil_scoped_release release;
    for (u32 i = 0; i < map.sources.size(); ++i)
    {
        map.maps.push_back(map.sources[i]->translation_map_to(*map.targets[i]));
    }
}

bool drop_arg(std::string_view missing)
{
    if (missing != "raise" && missing != "drop")
        throw nb::value_error("mymyr: translate: missing must be 'raise' or 'drop'");
    return missing == "drop";
}

nb::object run(PySlotMap& map, nb::handle rows, nb::handle ids, std::string_view missing,
               nb::handle pool, nb::handle status, nb::handle stream, nb::handle ctx, bool table)
{
    const bool drop = drop_arg(missing);
    if (is_cuda_array(rows))
    {
#if defined(MYMYR_HAS_CUDA)
        if (!pool.is_none())
            throw nb::value_error("mymyr: translate: pool is for host rows");
        return translate_device(map, rows, ids, drop, status, stream, ctx);
#else
        throw nb::type_error("mymyr: translate: CUDA rows require a CUDA build");
#endif
    }
    if (!status.is_none() || !stream.is_none() || !ctx.is_none())
        throw nb::value_error("mymyr: translate: status, stream and ctx are for device rows");
    PyPool* workers = pool_arg(pool);
    std::vector<std::shared_ptr<const std::vector<u32>>> slots;
    for (const auto& mapping : map.maps)
        slots.push_back(mapping->snapshot());
    const u32 sw = table ? map.source_words : std::max<u32>(1, map.sources[0]->words());
    const u32 sn = table ? map.source_numeric : map.sources[0]->numeric_words();
    StateBatch input = import_rows(rows, sw, sn);
    const bool states = is_state(rows) || nb::isinstance<nb::list>(rows) || nb::isinstance<nb::tuple>(rows);
    if (map.maps.size() > 1 && ids.is_none())
        throw nb::value_error("mymyr: translate: task_ids is required for a table with multiple instances");
    std::vector<i32> id_copy;
    std::shared_ptr<void> id_keep;
    const i32* task_ids = import_task_ids(ids, input.view.rows, static_cast<u32>(map.maps.size()), id_copy, id_keep);
    if (task_ids)
        for (u64 r = 0; r < input.view.rows; ++r)
            if (task_ids[r] < 0 || static_cast<usize>(task_ids[r]) >= map.maps.size())
                throw nb::value_error(("mymyr: translate: task_ids[" + std::to_string(r) + "] is outside the source table").c_str());
    for (u64 r = 0; r < input.owners.size(); ++r)
    {
        const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
        if (input.owners[r] != map.sources[i]->uid())
            throw nb::value_error("mymyr: translate: state belongs to another source task");
    }
    const u32 tn = table ? map.target_numeric : map.targets[0]->numeric_words();
    if (!states && !table && map.sources[0]->atoms().frozen() && sn == tn &&
        input.view.words == std::max<u32>(1, map.targets[0]->words()) &&
        map.sources[0]->encoding_fingerprint() == map.targets[0]->encoding_fingerprint())
    {
        check_task_rows(input, *map.sources[0]);
        if (!drop)
            return nb::borrow(rows);
        auto mask = Block::make(input.view.rows);
        std::memset(mask->data(), 1, static_cast<usize>(input.view.rows));
        return nb::make_tuple(nb::borrow(rows), export_array({mask, mask->data(), rl::DType::Bool,
                              {static_cast<i64>(input.view.rows)}, {}, false, false}, input.fw, input.enc));
    }
    const auto view_of = [&](u64 r) {
        const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
        const u32 source_words = bits::words_for(static_cast<u32>(slots[i]->size()));
        if (input.view.words < source_words || input.view.numeric_words < map.sources[i]->numeric_words())
            throw std::invalid_argument("mymyr: translate: source row is too narrow");
        const u64* row = input.view.row(r);
        return StateView{row, std::min(input.view.words, source_words), sn ? row + input.view.words : nullptr,
                         input.view.numeric_words};
    };
    {
        // Every row is checked before any is translated, so that an invalid row is reported before missing values,
        // and a lazy target interns the atoms of all rows before the output width is fixed.
        nb::gil_scoped_release release;
        for (u64 r = 0; r < input.view.rows; ++r)
        {
            const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
            map.maps[i]->prepare_row(*slots[i], view_of(r), r);
        }
    }
    const u32 tw = table ? map.target_words : std::max<u32>(1, map.targets[0]->words());
    const u32 rw = tw + tn;
    auto output = Block::make(checked_mul({input.view.rows, rw, sizeof(u64)}, "translated rows"));
    auto mask = Block::make(input.view.rows);
    auto* data = reinterpret_cast<u64*>(output->data());
    auto* lossless = reinterpret_cast<u8*>(mask->data());
    {
        nb::gil_scoped_release release;
        const auto process = [&](u64 begin, u64 end) {
            for (u64 r = begin; r < end; ++r)
            {
                const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
                lossless[r] = map.maps[i]->translate_row(*slots[i], view_of(r), {data + r * rw, tw},
                                                        {data + r * rw + tw, tn}, drop || workers != nullptr, r);
            }
        };
        if (workers)
        {
            const u32 threads = workers->pool->size();
            workers->pool->run([&](u32 t) { const auto [a, b] = ThreadPool::slice(input.view.rows, t, threads); process(a, b); });
        }
        else
            process(0, input.view.rows);
        if (workers && !drop)
            for (u64 r = 0; r < input.view.rows; ++r)
                if (!lossless[r])
                {
                    const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
                    map.maps[i]->translate_row(*slots[i], view_of(r), {data + r * rw, tw}, {data + r * rw + tw, tn}, false, r);
                }
    }
    nb::object result;
    if (states)
    {
        nb::list list;
        for (u64 r = 0; r < input.view.rows; ++r)
        {
            const u32 i = task_ids ? static_cast<u32>(task_ids[r]) : 0;
            const Owner owner = table ? table_of(map.target).owner(i) : owner_of(map.target);
            list.append(make_state(owner, State(data + r * rw, tw, data + r * rw + tw, map.targets[i]->numeric_words())));
        }
        result = is_state(rows) ? nb::borrow<nb::object>(list[0]) : nb::object(list);
    }
    else
    {
        result = export_array({output, data, rl::DType::U64, {static_cast<i64>(input.view.rows), rw}, {}, false, true},
                              input.fw, input.enc);
    }
    if (drop)
        return nb::make_tuple(result, export_array({mask, lossless, rl::DType::Bool, {static_cast<i64>(input.view.rows)}, {}, false, false},
                                                   input.fw, input.enc));
    return result;
}
}

nb::object translate_task(nb::handle source, nb::handle states, nb::handle target, std::string_view missing,
                          nb::handle pool, nb::handle status, nb::handle stream, nb::handle ctx)
{
    PySlotMap map;
    init_map(map, source, target, false);
    return run(map, states, nb::none(), missing, pool, status, stream, ctx, false);
}

nb::object translate_table(nb::handle source, nb::handle rows, nb::handle ids, nb::handle target,
                           std::string_view missing, nb::handle pool, nb::handle status, nb::handle stream, nb::handle ctx)
{
    PySlotMap map;
    init_map(map, source, target, true);
    return run(map, rows, ids, missing, pool, status, stream, ctx, true);
}

void bind_translate(nb::module_& m)
{
    nb::class_<PySlotMap>(m, "SlotMap", "Reusable name-based state translation between two tasks or two equally sized tables.")
        .def("__init__", [](PySlotMap* self, TableArg source, TableArg target) {
            auto* map = new (self) PySlotMap();
            init_map(*map, source, target, nb::isinstance<PyTable>(source));
        }, "source"_a, "target"_a)
        .def_prop_ro("unmatched", [](const PySlotMap& map) {
            std::vector<std::string> result;
            for (const auto& m : map.maps)
                result.insert(result.end(), m->unmatched().begin(), m->unmatched().end());
            return result;
        }, "Unmatched source predicate signatures, object names and numeric values.")
        .def("translate", [](PySlotMap& map, StatesLike rows, TaskIdsArg ids, std::string_view missing,
                              Arg<PyPool> pool, Arg<ann::ArrayLike> status, StreamArg stream, Arg<ann::Any> ctx) {
            return Arg<ann::Any>(run(map, rows, ids, missing, pool, status, stream, ctx, nb::isinstance<PyTable>(map.source)));
        }, "states"_a, nb::kw_only(), "task_ids"_a = nb::none(), "missing"_a = "raise", "pool"_a = nb::none(),
           "status"_a = nb::none(), "stream"_a = nb::none(), "ctx"_a = nb::none(),
           "Translate States or word rows; missing='drop' returns (translated, lossless [N]). Device status is int32 [3].");
}
}
