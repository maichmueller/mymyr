#include "arrays.hpp"
#include "py_formula.hpp"
#include "py_table.hpp"
#include "rl_imports.hpp"

#include "mymyr/encoding/flat.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <vector>

namespace mymyr::python::ann
{
struct EncodingExpansion
{
};
}

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::EncodingExpansion>
{
    static constexpr auto Name = const_name("mymyr._core._rl.Expansion");
};
}

namespace mymyr::python
{
using namespace nb::literals;

namespace
{
using ArrayOut = Arg<ann::Any>;
using ResultDict = nb::typed<nb::dict, std::string, ann::Any>;
using ActionsArg = Arg<ann::EncodingExpansion>;
using GoalsArg = Arg<std::variant<PyGroundCondition, nb::typed<nb::sequence, std::optional<PyGroundCondition>>>>;

struct PyFlatBatch
{
    encoding::FlatBatch value;
    Framework framework = Framework::Numpy;
};

struct PyFlatEncoder
{
    TableRef reference;
    encoding::FlatRelationEncoder encoder;

    PyFlatEncoder(TableRef table, encoding::FlatOptions options)
        : reference(std::move(table)), encoder(reference.table, options)
    {
    }
};

nb::object batch_array(const PyFlatBatch& batch, const char* name)
{
    const auto& bundle = batch.value.arrays;
    const rl::ArrayInfo* info = bundle->find(name);
    ArraySpec spec;
    spec.owner = std::shared_ptr<const void>(bundle, bundle->data(*info));
    spec.data = bundle->data(*info);
    spec.dtype = info->dtype;
    spec.shape = info->shape;
    spec.readonly = true;
    return export_array(std::move(spec), batch.framework, default_words(batch.framework));
}

ResultDict schema_dict(const encoding::FlatSchema& schema)
{
    nb::dict output;
    output["version"] = schema.version;
    output["num_domain_predicates"] = schema.domain_predicates;
    output["relation_names"] = nb::cast(schema.names);
    output["relation_arities"] = nb::cast(schema.arities);
    output["relation_sources"] = nb::cast(schema.sources);
    output["relation_predicates"] = nb::cast(schema.predicates);
    output["relation_schemas"] = nb::cast(schema.schemas);
    output["relation_kinds"] = nb::cast(schema.kinds);
    output["relation_instances"] = nb::cast(schema.instances);
    output["relation_polarities"] = nb::cast(schema.polarities);
    return ResultDict(output);
}

PyFlatBatch encode(const PyFlatEncoder& self, StatesLike state_arg, TaskIdsArg task_ids, ActionsArg action_arg,
                   GoalsArg goal_arg, Arg<ann::Framework> framework)
{
    StateBatch states = import_states(state_arg, current_words(self.encoder.table()));
    std::vector<i32> ids_copy;
    std::shared_ptr<void> ids_keep;
    const auto& table = self.encoder.table();
    const i32* ids = import_task_ids(task_ids, states.view.rows, table.size(), ids_copy, ids_keep);
    if (states.view.rows && table.size() > 1 && !ids)
        throw nb::value_error("mymyr: encoding: task_ids are required for a multi-instance table");
    table.check_task_ids(ids, states.view.rows);
    for (u64 row = 0; row < states.owners.size(); ++row)
        if (states.owners[row] != table.task(ids ? static_cast<u32>(ids[row]) : 0)->uid())
            throw nb::value_error("mymyr: encoding: a State belongs to another task or task id");

    rlimp::Imports imports(-1, dl::k_stream_default, false);
    encoding::ActionBatchView actions;
    if (!action_arg.is_none())
    {
        if (!nb::isinstance(action_arg, nb::module_::import_("mymyr._core._rl").attr("Expansion")))
            throw nb::type_error("mymyr: encoding: actions must be an rl.Expansion");
        if (nb::cast<bool>(action_arg.attr("overflow")))
            throw nb::value_error("mymyr: encoding: an overflowing expansion cannot provide complete action targets");
        if (nb::cast<u64>(action_arg.attr("num_states")) != states.view.rows)
            throw nb::value_error("mymyr: encoding: expansion and states have different row counts");
        const TableRef action_table = table_of(action_arg.attr("table"));
        if (action_table.table->size() != table.size())
            throw nb::value_error("mymyr: encoding: expansion belongs to another task table");
        for (u32 instance = 0; instance < table.size(); ++instance)
            if (action_table.table->task(instance)->uid() != table.task(instance)->uid())
                throw nb::value_error("mymyr: encoding: expansion belongs to another task table");
        std::vector<i64> shape;
        actions.schema = imports.get<i32>(action_arg.attr("schema"), "action_schema", {rlimp::i32t}, {-1}, false, &shape);
        actions.size = static_cast<u64>(shape[0]);
        actions.offsets = imports.get<i32>(action_arg.attr("offsets"), "action_offsets", {rlimp::i32t},
                                           {static_cast<i64>(states.view.rows) + 1}, false);
        actions.binding = imports.get<i32>(action_arg.attr("binding"), "action_binding", {rlimp::i32t},
                                           {static_cast<i64>(actions.size), -1}, false, &shape);
        actions.label_width = static_cast<u32>(shape[1]);
    }

    std::vector<const GroundCondition*> goals;
    nb::object goals_keep = nb::borrow(goal_arg);
    if (!goal_arg.is_none())
    {
        auto take_goal = [&](nb::handle value, u64 row) -> const GroundCondition*
        {
            if (value.is_none())
                return nullptr;
            if (!nb::isinstance<PyGroundCondition>(value))
                throw nb::type_error("mymyr: encoding: goals must be native GroundConditions or None");
            const auto& goal = *nb::inst_ptr<PyGroundCondition>(value);
            const u32 instance = ids ? static_cast<u32>(ids[row]) : 0;
            if (goal.o.data.get() != self.reference.core_of(instance).data.get())
                throw nb::value_error("mymyr: encoding: a goal belongs to another task or task id");
            return goal.c.get();
        };
        if (nb::isinstance<PyGroundCondition>(goal_arg))
        {
            if (table.size() != 1)
                throw nb::value_error("mymyr: encoding: table goals require one native condition per state row");
            for (u64 row = 0; row < states.view.rows; ++row)
                goals.push_back(take_goal(goal_arg, row));
        }
        else
        {
            if (!nb::isinstance<nb::list>(goal_arg) && !nb::isinstance<nb::tuple>(goal_arg))
                throw nb::type_error("mymyr: encoding: goals must be a GroundCondition or an aligned list");
            nb::sequence sequence = nb::borrow<nb::sequence>(goal_arg);
            if (nb::len(sequence) != states.view.rows)
                throw nb::value_error("mymyr: encoding: goals need one condition per state row");
            for (u64 row = 0; row < states.view.rows; ++row)
                goals.push_back(take_goal(sequence[row], row));
        }
    }
    PyFlatBatch output;
    output.framework = parse_framework(framework, state_arg);
    {
        nb::gil_scoped_release release;
        output.value = self.encoder.encode(states.view, ids, actions, goals);
    }
    return output;
}
}

void bind_encoding(nb::module_& parent)
{
    nb::module_ module = parent.def_submodule("_encoding", "Native classical relational encodings");
    nb::class_<PyFlatBatch> batch(module, "FlatBatch");
    batch.def_prop_ro("num_graphs", [](const PyFlatBatch& value) { return value.value.graphs; })
        .def_prop_ro("version", [](const PyFlatBatch&) { return encoding::k_flat_version; })
        .def_prop_ro("schema", [](const PyFlatBatch& value) { return schema_dict(*value.value.schema); })
        .def_prop_ro("relation_names", [](const PyFlatBatch& value) { return value.value.schema->names; })
        .def_prop_ro("relation_sources", [](const PyFlatBatch& value) { return value.value.schema->sources; })
        .def_prop_ro("object_names", [](const PyFlatBatch& value) { return value.value.object_names; })
        .def_prop_ro("task_fingerprints", [](const PyFlatBatch& value) { return value.value.task_fingerprints; })
        .def("as_dict", [](const PyFlatBatch& value, Arg<ann::Framework> framework)
        {
            const Framework target = framework.is_none() ? value.framework : parse_framework(framework);
            nb::dict output = nb::cast<nb::dict>(export_bundle(value.value.arrays, target, default_words(target)));
            output["schema"] = schema_dict(*value.value.schema);
            output["object_names"] = nb::cast(value.value.object_names);
            output["task_fingerprints"] = nb::cast(value.value.task_fingerprints);
            return ResultDict(output);
        }, "framework"_a = nb::none());
    for (const char* name : {"relation_counts", "relation_offsets", "relation_tuple_offsets", "relation_args",
                             "object_offsets", "task_ids", "action_offsets", "action_schema", "action_binding",
                             "action_parent", "action_tuple", "relation_arities", "relation_predicates",
                             "relation_schemas", "relation_kinds", "relation_instances", "relation_polarities"})
        batch.def_prop_ro(name, [name](const PyFlatBatch& value) { return ArrayOut(batch_array(value, name)); });

    nb::class_<PyFlatEncoder>(module, "FlatRelationEncoder")
        .def("__init__", [](PyFlatEncoder* self, TableArg table, bool include_static, bool include_derived)
        {
            new (self) PyFlatEncoder(table_of(table), {include_static, include_derived});
        }, "table"_a, nb::kw_only(), "include_static"_a = true, "include_derived"_a = true)
        .def_prop_ro("table", [](const PyFlatEncoder& value) { return TableArg(value.reference.obj); })
        .def_prop_ro("schema", [](const PyFlatEncoder& value) { return schema_dict(*value.encoder.schema()); })
        .def_prop_ro("relation_names", [](const PyFlatEncoder& value) { return value.encoder.schema()->names; })
        .def_prop_ro("relation_arities", [](const PyFlatEncoder& value) { return value.encoder.schema()->arities; })
        .def_prop_ro("relation_sources", [](const PyFlatEncoder& value) { return value.encoder.schema()->sources; })
        .def("encode", &encode, "states"_a, nb::kw_only(), "task_ids"_a = nb::none(), "actions"_a = nb::none(),
             "goals"_a = nb::none(), "framework"_a = nb::none(),
             "Encode facts, signed goal satisfaction and optional existing expansion labels into owned arrays.");
}
}
