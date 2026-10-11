#include "arrays.hpp"
#include "py_formula.hpp"
#include "py_table.hpp"
#include "rl_imports.hpp"

#include "mymyr/encoding/flat.hpp"
#include "mymyr/encoding/schema.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::python::ann
{
struct EncodingExpansion
{
};
struct EncodingLayout
{
};
struct EncodingIndexDtype
{
};
struct EncodingHorizonMode
{
};
struct EncodingRootPolicy
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
template<>
struct type_caster<mymyr::python::ann::EncodingLayout>
{
    static constexpr auto Name = const_name("typing.Literal['graph', 'relation']");
};
template<>
struct type_caster<mymyr::python::ann::EncodingIndexDtype>
{
    static constexpr auto Name = const_name("typing.Literal['int32', 'int64']");
};
template<>
struct type_caster<mymyr::python::ann::EncodingHorizonMode>
{
    static constexpr auto Name = const_name("typing.Literal['full', 'delta', 'action'] | None");
};
template<>
struct type_caster<mymyr::python::ann::EncodingRootPolicy>
{
    static constexpr auto Name = const_name("typing.Literal['include', 'exclude', 'encode_only'] | None");
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

    PyFlatEncoder(TableRef table, encoding::FlatRelationEncoderOptions options)
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
        const auto source_ids = nb::cast<std::vector<i32>>(action_arg.attr("task_ids"));
        for (u64 row = 0; row < states.view.rows; ++row)
            if (source_ids[row] != (ids ? ids[row] : 0))
                throw nb::value_error("mymyr: encoding: expansion and states have different source task ids");
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

// ------------------------------------------------------------------------------------------------ EncodingSchema
struct PyEncodingSchema
{
    std::shared_ptr<const encoding::EncodingSchema> schema;
};

using TextSequenceArg = Arg<std::vector<std::string>>;

[[noreturn]] void option_error(const char* name, const std::string& allowed, nb::handle got)
{
    throw nb::value_error(("mymyr: encoding: " + std::string(name) + " must be one of " + allowed + ", got " +
                           nb::cast<std::string>(nb::repr(got)))
                              .c_str());
}

std::string text_of(nb::handle value, const char* name)
{
    if (!nb::isinstance<nb::str>(value))
        throw nb::type_error(("mymyr: encoding: " + std::string(name) + " must be a string, not " +
                              nb::cast<std::string>(nb::str(value.type().attr("__name__"))))
                                 .c_str());
    return nb::cast<std::string>(value);
}

/// The enumerator named by a string argument; `parse` maps text to an optional enumerator, `allowed` lists the texts.
template<class Parse>
auto choice(nb::handle value, const char* name, Parse parse, const std::string& allowed)
{
    const auto parsed = parse(text_of(value, name));
    if (!parsed)
        option_error(name, allowed, value);
    return *parsed;
}

/// The enumerators named by an iterable of strings (a single string is not a collection of options).
template<class E, class Parse>
encoding::EnumSet<E> choice_set(nb::handle value, const char* name, Parse parse, const std::string& allowed)
{
    if (nb::isinstance<nb::str>(value) || !PyObject_HasAttrString(value.ptr(), "__iter__"))
        throw nb::type_error(("mymyr: encoding: " + std::string(name) + " must be a collection of strings").c_str());
    encoding::EnumSet<E> out;
    for (nb::handle item : value)
        out.insert(choice(item, name, parse, allowed));
    return out;
}

encoding::FlatOptions schema_options(nb::handle max_goal_level, bool include_static, bool include_derived,
                                     bool nullary, bool equality, bool predicate_virtual_nodes, bool include_actions,
                                     nb::handle goal_derivations, nb::handle target_sources, nb::handle layout,
                                     nb::handle index_dtype, nb::handle horizon_mode, nb::handle root_policy,
                                     bool parent_relation)
{
    using namespace encoding;
    FlatOptions options;
    options.goal_derivations = choice_set<GoalDerivation>(
        goal_derivations, "goal_derivations", parse_goal_derivation,
        "'plain', 'satisfied', 'unsatisfied', 'added_satisfied', 'added_unsatisfied'");
    options.max_goal_level = int_arg<u32>(max_goal_level, "max_goal_level", 0, 3);
    options.include_static = include_static;
    options.include_derived = include_derived;
    options.nullary = nullary;
    options.equality = equality;
    options.predicate_virtual_nodes = predicate_virtual_nodes;
    options.include_actions = include_actions;
    options.targets = choice_set<TargetSource>(target_sources, "target_sources", parse_target_source,
                                               "'goal', 'subgoal', 'action', 'history'");
    options.layout = choice(layout, "layout", parse_layout, "'graph', 'relation'");
    options.index = choice(index_dtype, "index_dtype", parse_index_type, "'int32', 'int64'");
    if (!horizon_mode.is_none())
        options.horizon.mode =
            choice(horizon_mode, "horizon_mode", parse_horizon_mode, "'full', 'delta', 'action' or None");
    if (options.horizon.mode == HorizonMode::None && !root_policy.is_none())
        throw nb::value_error("mymyr: encoding: root_policy needs a horizon_mode");
    if (!root_policy.is_none())
        options.horizon.root =
            choice(root_policy, "root_policy", parse_root_policy, "'include', 'exclude', 'encode_only' or None");
    options.horizon.parent_relation = parent_relation;
    return options;
}

template<class E>
nb::list texts_of(const std::vector<E>& values)
{
    nb::list out;
    for (E value : values)
        out.append(std::string(encoding::to_string(value)));
    return out;
}

nb::dict options_dict(const encoding::FlatOptions& o)
{
    nb::dict out;
    out["goal_derivations"] = nb::tuple(texts_of(o.goal_derivations.members()));
    out["max_goal_level"] = o.max_goal_level;
    out["include_static"] = o.include_static;
    out["include_derived"] = o.include_derived;
    out["nullary"] = o.nullary;
    out["equality"] = o.equality;
    out["predicate_virtual_nodes"] = o.predicate_virtual_nodes;
    out["include_actions"] = o.include_actions;
    out["target_sources"] = nb::tuple(texts_of(o.targets.members()));
    out["layout"] = std::string(encoding::to_string(o.layout));
    out["index_dtype"] = std::string(encoding::to_string(o.index));
    const bool horizon = o.horizon.mode != encoding::HorizonMode::None;
    out["horizon_mode"] = horizon ? nb::object(nb::cast(std::string(encoding::to_string(o.horizon.mode)))) : nb::none();
    out["root_policy"] = horizon ? nb::object(nb::cast(std::string(encoding::to_string(o.horizon.root)))) : nb::none();
    out["parent_relation"] = o.horizon.parent_relation;
    return out;
}

template<class Field>
auto relation_column(const PyEncodingSchema& self, Field field)
{
    using Out = std::decay_t<decltype(field(std::declval<const encoding::Relation&>()))>;
    std::vector<Out> out;
    out.reserve(self.schema->num_relations());
    for (const encoding::Relation& relation : self.schema->relations())
        out.push_back(field(relation));
    return out;
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

    nb::class_<PyEncodingSchema>(
        module, "EncodingSchema",
        "The relation catalog of an encoding: the relations a batch of encoded states holds, sorted by name, with their "
        "arities and slot roles. It depends on the domain of the task or table and on the options, never on the "
        "instances, so tables over the same domain agree on the relation ids.")
        .def(
            "__init__",
            [](PyEncodingSchema* self, TableArg table, TextSequenceArg goal_derivations, IntArg max_goal_level,
               bool include_static, bool include_derived, bool nullary, bool equality, bool predicate_virtual_nodes,
               bool include_actions, TextSequenceArg target_sources, Arg<ann::EncodingLayout> layout,
               Arg<ann::EncodingIndexDtype> index_dtype, Arg<ann::EncodingHorizonMode> horizon_mode,
               Arg<ann::EncodingRootPolicy> root_policy, bool parent_relation)
            {
                const TableRef reference = table_of(table);
                const encoding::FlatOptions options = schema_options(
                    max_goal_level, include_static, include_derived, nullary, equality, predicate_virtual_nodes, include_actions,
                    goal_derivations, target_sources, layout, index_dtype, horizon_mode, root_policy, parent_relation);
                try
                {
                    new (self) PyEncodingSchema{encoding::EncodingSchema::create(*reference.table, options)};
                }
                catch (const std::invalid_argument& error)
                {
                    throw nb::value_error(error.what());
                }
            },
            "table"_a, nb::kw_only(), "goal_derivations"_a = nb::make_tuple("plain", "satisfied"),
            "max_goal_level"_a = 0, "include_static"_a = true, "include_derived"_a = true, "nullary"_a = true, "equality"_a = false,
            "predicate_virtual_nodes"_a = false, "include_actions"_a = true, "target_sources"_a = nb::tuple(),
            "layout"_a = "graph", "index_dtype"_a = "int32", "horizon_mode"_a = nb::none(),
            "root_policy"_a = nb::none(), "parent_relation"_a = false,
            "The catalog of the domain of `table` (a Task, TaskHandle or TaskTable) under the options. "
            "goal_derivations: the variants of the goal relations (plain, satisfied, unsatisfied; added_satisfied and "
            "added_unsatisfied for the delta horizon). max_goal_level: subgoal layers beyond the goal (0 to 3). "
            "include_static / include_derived: relations of the static (types included) / derived predicates. "
            "nullary: relations of the predicates without arguments (their atoms are tuples over the graph's auxiliary entity). equality: relations of the equality predicate `=`. predicate_virtual_nodes: a predicate slot in "
            "the predicate relations' tuples. include_actions: one relation per action schema. target_sources: "
            "which of goal, subgoal, action and history literals become entities that tuples refer to. layout, "
            "index_dtype: how a batch is stored. horizon_mode, root_policy, parent_relation: the horizon variants "
            "(root_policy defaults to 'include'). Raises ValueError for invalid combinations and for numeric tasks.")
        .def_prop_ro(
            "options", [](const PyEncodingSchema& self) { return options_dict(self.schema->options()); },
            "The options in normal form, as the keyword arguments that rebuild this schema.")
        .def_prop_ro("num_relations", [](const PyEncodingSchema& self) { return self.schema->num_relations(); })
        .def_prop_ro(
            "relation_names",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return r.name; }); },
            "The relation names in catalog order; a relation's id is its position.")
        .def_prop_ro(
            "relation_arities",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return r.arity; }); },
            "The encoded arity of each relation: its slots plus its arguments.")
        .def_prop_ro(
            "relation_logical_arities",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return r.logical_arity; }); },
            "The number of arguments of each relation (the arity of its predicate or action schema).")
        .def_prop_ro(
            "relation_lanes",
            [](const PyEncodingSchema& self)
            {
                return relation_column(
                    self, [](const encoding::Relation& r) { return std::string(encoding::to_string(r.lane)); });
            },
            "The lane of each relation: fluent, static, derived, goal, goal_derivation, goal_satisfaction, history, "
            "action or parent.")
        .def_prop_ro(
            "relation_signs",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return static_cast<int>(r.sign); }); },
            "1 for `[+]` relations, -1 for `[-]` relations, 0 for relations without a sign.")
        .def_prop_ro(
            "relation_levels",
            [](const PyEncodingSchema& self)
            {
                return relation_column(
                    self, [](const encoding::Relation& r) { return r.level ? static_cast<int>(*r.level) : -1; });
            },
            "The goal level of each relation (0: the goal, 1: first subgoal layer, ...), -1 for none.")
        .def_prop_ro(
            "relation_derivations",
            [](const PyEncodingSchema& self)
            {
                return relation_column(self,
                                       [](const encoding::Relation& r)
                                       {
                                           return r.derivation
                                                      ? std::optional<std::string>(std::string(encoding::to_string(*r.derivation)))
                                                      : std::nullopt;
                                       });
            },
            "The goal derivation of each relation (satisfied, unsatisfied, added_satisfied, added_unsatisfied), None "
            "for plain goal relations and relations without a goal.")
        .def_prop_ro(
            "relation_predicates",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return r.predicate; }); },
            "The domain predicate id of each relation, -1 for action and parent relations.")
        .def_prop_ro(
            "relation_schemas",
            [](const PyEncodingSchema& self)
            { return relation_column(self, [](const encoding::Relation& r) { return r.schema; }); },
            "The first action schema id of each action relation, -1 for the others.")
        .def_prop_ro(
            "slot_roles",
            [](const PyEncodingSchema& self)
            {
                std::vector<std::vector<int>> out;
                for (u32 id = 0; id < self.schema->num_relations(); ++id)
                {
                    std::vector<int> roles;
                    for (encoding::SlotRole role : self.schema->slot_roles(id))
                        roles.push_back(static_cast<int>(role));
                    out.push_back(std::move(roles));
                }
                return out;
            },
            "The role of each position of each relation's tuples, as indices into slot_role_names.")
        .def_prop_ro(
            "schema_relations",
            [](const PyEncodingSchema& self)
            {
                const auto ids = self.schema->schema_relations();
                return std::vector<i32>(ids.begin(), ids.end());
            },
            "The relation id of each action schema of the domain (empty without action relations). Action schemas "
            "of one name share a relation.")
        .def_prop_ro("slot_role_names",
                     [](const PyEncodingSchema& self)
                     {
                         const auto names = self.schema->slot_role_names();
                         return std::vector<std::string>(names.begin(), names.end());
                     })
        .def_prop_ro(
            "entity_role_names",
            [](const PyEncodingSchema& self)
            {
                const auto names = self.schema->entity_role_names();
                return std::vector<std::string>(names.begin(), names.end());
            },
            "The roles of the entities of a graph; the role of an entity is its index. The auxiliary role exists with nullary relations.")
        .def_prop_ro(
            "target_group_names",
            [](const PyEncodingSchema&)
            {
                return std::vector<std::string>(encoding::k_target_group_names.begin(),
                                                encoding::k_target_group_names.end());
            },
            "The target groups: the group code of a target is its index.")
        .def_prop_ro(
            "schema_fingerprint", [](const PyEncodingSchema& self) { return self.schema->fingerprint(); },
            "A 64-bit hash of the catalog and the options that shape the output; stable across processes and builds.")
        .def(
            "relation_id",
            [](const PyEncodingSchema& self, const std::string& name)
            {
                const auto id = self.schema->find(name);
                if (!id)
                    throw nb::key_error(("mymyr: encoding: no relation named '" + name + "'").c_str());
                return *id;
            },
            "name"_a, "The id of the relation with this name. Raises KeyError for an unknown name.");
}
}
