#include "arrays.hpp"
#include "py_table.hpp"
#include "rl_imports.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/schema.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mymyr::python::ann
{
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
struct EncodingEntityArrays
{
};
}

namespace nanobind::detail
{
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
struct type_caster<mymyr::python::ann::EncodingEntityArrays>
{
    static constexpr auto Name = const_name("typing.Literal['auto', 'always']");
};
template<>
struct type_caster<mymyr::python::ann::EncodingHorizonMode>
{
    static constexpr auto Name = const_name("typing.Literal['full', 'delta', 'action']");
};
template<>
struct type_caster<mymyr::python::ann::EncodingRootPolicy>
{
    static constexpr auto Name = const_name("typing.Literal['include', 'exclude', 'encode_only']");
};
}

namespace mymyr::python
{
using namespace nb::literals;

namespace
{
using ArrayOut = Arg<ann::Any>;
using ResultDict = nb::typed<nb::dict, std::string, ann::Any>;
using TextSequenceArg = Arg<std::vector<std::string>>;
using DerivedArg = Arg<ann::ArrayLike>;

struct PyEncodingSchema
{
    std::shared_ptr<const encoding::EncodingSchema> schema;
};

struct PyFlatEncoder
{
    TableRef reference;
    std::shared_ptr<const encoding::FlatEncoder> encoder;
    std::mutex pools_mutex;                            // guards pools
    std::map<u32, std::shared_ptr<ThreadPool>> pools;  // by thread count, created on first use

    PyFlatEncoder(TableRef reference_, std::shared_ptr<const encoding::FlatEncoder> encoder_)
        : reference(std::move(reference_)), encoder(std::move(encoder_))
    {
    }

    /// The pool of `threads` members (at least 1), shared by the calls of this encoder.
    std::shared_ptr<ThreadPool> pool(u32 threads)
    {
        std::lock_guard lock(pools_mutex);
        auto& slot = pools[threads];
        if (!slot)
            slot = std::make_shared<ThreadPool>(threads);
        return slot;
    }
};

struct PyFlatBatch
{
    std::shared_ptr<const rl::ArrayBundle> arrays;
    std::shared_ptr<const encoding::EncodingSchema> schema;
    Framework framework = Framework::Numpy;
    std::optional<encoding::FlatSizes> sizes;  // set by an encoding with capacities
};

// ------------------------------------------------------------------------------------------------ options

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
                                     nb::handle index_dtype, bool emit_counts, nb::handle entity_arrays, nb::handle horizon_mode, nb::handle root_policy,
                                     bool parent_relation)
{
    using namespace encoding;
    FlatOptions options;
    options.goal_derivations = choice_set<GoalDerivation>(
        goal_derivations, "goal_derivations", parse_goal_derivation,
        "'plain', 'satisfied', 'unsatisfied', 'added_satisfied', 'added_unsatisfied'");
    const i64 level = int_arg<i64>(max_goal_level, "max_goal_level");
    if (level < 0 || level > 3)
        throw nb::value_error(("mymyr: encoding: max_goal_level must be in [0, 3], got " + std::to_string(level)).c_str());
    options.max_goal_level = static_cast<u32>(level);
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
    options.emit_counts = emit_counts;
    options.entity_arrays = choice(entity_arrays, "entity_arrays", parse_entity_arrays, "'auto', 'always'");
    if (!horizon_mode.is_none())
    {
        options.horizon.mode =
            choice(horizon_mode, "horizon_mode", parse_horizon_mode, "'full', 'delta', 'action' or None");
        if (options.horizon.mode == HorizonMode::None)  // None is the spelling of no horizon
            option_error("horizon_mode", "'full', 'delta', 'action' or None", horizon_mode);
    }
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
    out["emit_counts"] = o.emit_counts;
    out["entity_arrays"] = std::string(encoding::to_string(o.entity_arrays));
    const bool horizon = o.horizon.mode != encoding::HorizonMode::None;
    out["horizon_mode"] = horizon ? nb::object(nb::cast(std::string(encoding::to_string(o.horizon.mode)))) : nb::none();
    out["root_policy"] = horizon ? nb::object(nb::cast(std::string(encoding::to_string(o.horizon.root)))) : nb::none();
    out["parent_relation"] = o.horizon.parent_relation;
    return out;
}

constexpr const char* k_option_doc =
    "goal_derivations: the variants of the goal relations (plain, satisfied, unsatisfied; added_satisfied and "
    "added_unsatisfied for the delta horizon). max_goal_level: subgoal layers beyond the goal (0 to 3). "
    "include_static / include_derived: the state relations of the static (types included) / derived "
    "predicates. nullary: relations of the predicates without arguments (their atoms are tuples over the "
    "graph's auxiliary entity). equality: relations of the equality predicate `=`. predicate_virtual_nodes: a predicate "
    "slot in the predicate relations' tuples. include_actions: one relation per action schema. target_sources: "
    "which of goal, subgoal, action and history literals become entities that tuples refer to (goal and subgoal "
    "targets need the plain goal derivation, action targets include_actions). layout, "
    "index_dtype: how a batch is stored; layout='relation' stores the tuples relation by relation across the graphs "
    "(cell (g, r) is segment r * G + g instead of g * R + r). emit_counts=False omits relation_counts (the tuples of "
    "a segment are the difference of two relation_offsets). entity_arrays='auto' omits entity_role and entity_ref "
    "when every entity is an object or the auxiliary entity (no target_sources, predicate_virtual_nodes, "
    "include_actions or horizon_mode). horizon_mode, root_policy, parent_relation: the horizon variants "
    "(root_policy defaults to 'include' and needs a horizon_mode for any other value). Raises ValueError for invalid "
    "combinations and for numeric tasks.";

/// The constructor of the classes that are built from a table and the encoding options (EncodingSchema,
/// FlatRelationEncoder): the keyword arguments are parsed here once. `make(self, reference, options)` constructs the
/// object.
template<class Py, class Make>
void define_init(nb::class_<Py>& cls, Make make, const char* doc)
{
    cls.def(
        "__init__",
        [make](Py* self, TableArg table, TextSequenceArg goal_derivations, IntArg max_goal_level, bool include_static,
               bool include_derived, bool nullary, bool equality, bool predicate_virtual_nodes, bool include_actions,
               TextSequenceArg target_sources, Arg<ann::EncodingLayout> layout, Arg<ann::EncodingIndexDtype> index_dtype,
               bool emit_counts, Arg<ann::EncodingEntityArrays> entity_arrays, Arg<ann::EncodingHorizonMode> horizon_mode, Arg<ann::EncodingRootPolicy> root_policy,
               bool parent_relation)
        {
            const TableRef reference = table_of(table);
            const encoding::FlatOptions options = schema_options(
                max_goal_level, include_static, include_derived, nullary, equality, predicate_virtual_nodes,
                include_actions, goal_derivations, target_sources, layout, index_dtype, emit_counts, entity_arrays,
                horizon_mode, root_policy, parent_relation);
            make(self, reference, options);
        },
        "table"_a, nb::kw_only(), "goal_derivations"_a = nb::make_tuple("plain", "satisfied"), "max_goal_level"_a = 0,
        "include_static"_a = true, "include_derived"_a = true, "nullary"_a = true, "equality"_a = false,
        "predicate_virtual_nodes"_a = false, "include_actions"_a = true, "target_sources"_a = nb::tuple(),
        "layout"_a = "graph", "index_dtype"_a = "int32", "emit_counts"_a = true, "entity_arrays"_a = "always", "horizon_mode"_a = nb::none(), "root_policy"_a = nb::none(),
        "parent_relation"_a = false, doc);
}

// ------------------------------------------------------------------------------------------------ the catalog

template<class Field>
auto relation_column(const encoding::EncodingSchema& schema, Field field)
{
    using Out = std::decay_t<decltype(field(std::declval<const encoding::Relation&>()))>;
    std::vector<Out> out;
    out.reserve(schema.num_relations());
    for (const encoding::Relation& relation : schema.relations())
        out.push_back(field(relation));
    return out;
}

/// The read-only properties that describe the catalog of an EncodingSchema, defined on every class that has one;
/// `schema_of(self)` returns it.
template<class Py, class SchemaOf>
void define_schema_properties(nb::class_<Py>& cls, SchemaOf schema_of)
{
    cls.def_prop_ro(
           "options", [schema_of](const Py& self) { return options_dict(schema_of(self).options()); },
           "The options in normal form, as the keyword arguments that rebuild this schema.")
        .def_prop_ro("num_relations", [schema_of](const Py& self) { return schema_of(self).num_relations(); })
        .def_prop_ro(
            "relation_names",
            [schema_of](const Py& self)
            { return relation_column(schema_of(self), [](const encoding::Relation& r) { return r.name; }); },
            "The relation names in catalog order; a relation's id is its position.")
        .def_prop_ro(
            "relation_arities",
            [schema_of](const Py& self)
            { return relation_column(schema_of(self), [](const encoding::Relation& r) { return r.arity; }); },
            "The encoded arity of each relation: its slots plus its arguments.")
        .def_prop_ro(
            "relation_logical_arities",
            [schema_of](const Py& self)
            { return relation_column(schema_of(self), [](const encoding::Relation& r) { return r.logical_arity; }); },
            "The number of arguments of each relation (the arity of its predicate or action schema).")
        .def_prop_ro(
            "relation_lanes",
            [schema_of](const Py& self)
            {
                return relation_column(
                    schema_of(self), [](const encoding::Relation& r) { return std::string(encoding::to_string(r.lane)); });
            },
            "The lane of each relation: fluent, static, derived, goal, goal_derivation, goal_satisfaction, history, "
            "action or parent.")
        .def_prop_ro(
            "relation_signs",
            [schema_of](const Py& self)
            {
                return relation_column(schema_of(self),
                                       [](const encoding::Relation& r) { return static_cast<int>(r.sign); });
            },
            "1 for `[+]` relations, -1 for `[-]` relations, 0 for relations without a sign.")
        .def_prop_ro(
            "relation_levels",
            [schema_of](const Py& self)
            {
                return relation_column(schema_of(self), [](const encoding::Relation& r)
                                       { return r.level ? static_cast<int>(*r.level) : -1; });
            },
            "The goal level of each relation (0: the goal, 1: first subgoal layer, ...), -1 for none.")
        .def_prop_ro(
            "relation_derivations",
            [schema_of](const Py& self)
            {
                return relation_column(
                    schema_of(self),
                    [](const encoding::Relation& r)
                    {
                        return r.derivation ? std::optional<std::string>(std::string(encoding::to_string(*r.derivation)))
                                            : std::nullopt;
                    });
            },
            "The goal derivation of each relation (satisfied, unsatisfied, added_satisfied, added_unsatisfied), None "
            "for plain goal relations and relations without a goal.")
        .def_prop_ro(
            "relation_predicates",
            [schema_of](const Py& self)
            { return relation_column(schema_of(self), [](const encoding::Relation& r) { return r.predicate; }); },
            "The domain predicate id of each relation, -1 for action and parent relations.")
        .def_prop_ro(
            "relation_schemas",
            [schema_of](const Py& self)
            { return relation_column(schema_of(self), [](const encoding::Relation& r) { return r.schema; }); },
            "The first action schema id of each action relation, -1 for the others.")
        .def_prop_ro(
            "relation_slot_roles",
            [schema_of](const Py& self)
            {
                const encoding::EncodingSchema& schema = schema_of(self);
                std::vector<std::vector<int>> out;
                for (u32 id = 0; id < schema.num_relations(); ++id)
                {
                    std::vector<int> roles;
                    for (encoding::SlotRole role : schema.slot_roles(id))
                        roles.push_back(static_cast<int>(role));
                    out.push_back(std::move(roles));
                }
                return out;
            },
            "The role of each position of each relation's tuples, as indices into slot_role_names.")
        .def_prop_ro(
            "schema_relations",
            [schema_of](const Py& self)
            {
                const auto ids = schema_of(self).schema_relations();
                return std::vector<i32>(ids.begin(), ids.end());
            },
            "The relation id of each action schema of the domain (empty without action relations). Action schemas "
            "of one name and arity share a relation; a name with several arities has one relation `name#arity` each.")
        .def_prop_ro("slot_role_names",
                     [schema_of](const Py& self)
                     {
                         const auto names = schema_of(self).slot_role_names();
                         return std::vector<std::string>(names.begin(), names.end());
                     })
        .def_prop_ro(
            "entity_role_names",
            [schema_of](const Py& self)
            {
                const auto names = schema_of(self).entity_role_names();
                return std::vector<std::string>(names.begin(), names.end());
            },
            "The roles of the entities of a graph; the role of an entity is its index. The auxiliary role is used by "
            "the nullary relations.")
        .def_prop_ro(
            "target_group_names",
            [](const Py&)
            {
                return std::vector<std::string>(encoding::k_target_group_names.begin(),
                                                encoding::k_target_group_names.end());
            },
            "The target groups: the group code of a target is its index.")
        .def_prop_ro(
            "schema_fingerprint", [schema_of](const Py& self) { return schema_of(self).fingerprint(); },
            "A 64-bit hash of the catalog and the options that shape the output; stable across processes and builds.")
        .def(
            "relation_id",
            [schema_of](const Py& self, const std::string& name)
            {
                const auto id = schema_of(self).find(name);
                if (!id)
                    throw nb::key_error(("mymyr: encoding: no relation named '" + name + "'").c_str());
                return *id;
            },
            "name"_a, "The id of the relation with this name. Raises KeyError for an unknown name.");
}

// ------------------------------------------------------------------------------------------------ encoding

nb::object batch_array(const PyFlatBatch& batch, const char* name)
{
    const rl::ArrayInfo* info = batch.arrays->find(name);
    if (!info)
        throw nb::attribute_error((std::string("mymyr: encoding: this batch has no ") + name).c_str());
    ArraySpec spec;
    spec.owner = std::shared_ptr<const void>(batch.arrays, batch.arrays->data(*info));
    spec.data = batch.arrays->data(*info);
    spec.dtype = info->dtype;
    spec.shape = info->shape;
    spec.readonly = true;
    return export_array(std::move(spec), batch.framework, default_words(batch.framework));
}

using CapacitiesArg = std::optional<nb::typed<nb::dict, std::string, std::uint64_t>>;

encoding::FlatCapacities capacities_of(const nb::dict& value)
{
    encoding::FlatCapacities out;
    bool entities = false, tuples = false, args = false;
    for (auto [key, item] : value)
    {
        const std::string name = text_of(key, "capacities key");
        if (name == "entities")
            out.entities = int_arg<u64>(item, "capacities['entities']"), entities = true;
        else if (name == "tuples")
            out.tuples = int_arg<u64>(item, "capacities['tuples']"), tuples = true;
        else if (name == "args")
            out.args = int_arg<u64>(item, "capacities['args']"), args = true;
        else
            throw nb::value_error(("mymyr: encoding: capacities has the unknown key '" + name +
                                   "'; the keys are 'entities', 'tuples' and 'args'")
                                      .c_str());
    }
    if (!(entities && tuples && args))
        throw nb::value_error("mymyr: encoding: capacities needs the keys 'entities', 'tuples' and 'args'");
    return out;
}

PyFlatBatch encode(PyFlatEncoder& self, StatesLike state_arg, TaskIdsArg task_ids, DerivedArg derived_arg,
                   Arg<ann::Framework> framework, IntArg threads_in, CapacitiesArg capacities_in)
{
    const u32 threads = threads_arg(threads_in);
    const std::optional<encoding::FlatCapacities> capacities =
        capacities_in ? std::optional(capacities_of(*capacities_in)) : std::nullopt;

    StateBatch states = import_states(state_arg, packed_words(state_arg));
    std::vector<i32> ids_copy;
    std::shared_ptr<void> ids_keep;
    const rl::TaskTable& table = self.encoder->table();
    const i32* ids = import_task_ids(task_ids, states.view.rows, table.size(), ids_copy, ids_keep);
    if (states.view.rows && table.size() > 1 && !ids)
        throw nb::value_error("mymyr: encoding: task_ids are required for a multi-instance table");
    table.check_task_ids(ids, states.view.rows);
    for (u64 row = 0; row < states.owner_tasks.size(); ++row)
    {
        const u32 instance = ids ? static_cast<u32>(ids[row]) : 0;
        const Task& expected = *table.task(instance);
        const Task& given = *states.owner_tasks[row];
        if (given.uid() == expected.uid() ||
            (given.fingerprint() == expected.fingerprint() && given.encoding_fingerprint() == expected.encoding_fingerprint()))
            continue;
        throw nb::value_error(("mymyr: encoding: state " + std::to_string(row) + " is not a state of instance " +
                               std::to_string(instance) + " of the table (its task id): its task differs in fingerprint or "
                               "atom encoding")
                                  .c_str());
    }

    rlimp::Imports imports(-1, dl::k_stream_default, false);
    encoding::EncodeInput input;
    input.states = states.view;
    input.task_ids = ids;
    std::vector<i64> rows;
    u32 derived_words = 0;
    input.derived =
        imports.words_nd(derived_arg, "derived", {static_cast<i64>(states.view.rows)}, false, rows, derived_words);
    input.derived_stride = derived_words;

    PyFlatBatch output;
    output.framework = parse_framework(framework, state_arg);
    output.schema = self.encoder->schema_ptr();
    const std::shared_ptr<ThreadPool> pool = self.pool(threads == 0 ? hardware_threads() : threads);
    thread_local encoding::EncodeScratch scratch;
    {
        nb::gil_scoped_release release;
        if (capacities)
        {
            encoding::FlatSizes sizes;
            output.arrays = self.encoder->encode_capacity(input, *capacities, *pool, scratch, sizes);
            output.sizes = sizes;
        }
        else
            output.arrays = self.encoder->encode(input, *pool, scratch);
    }
    return output;
}

constexpr std::pair<const char*, const char*> k_batch_arrays[] = {
    {"task_ids", "int32 [G]: the instance of each graph."},
    {"num_objects", "index [G]: the objects of each graph's instance (the auxiliary entity is not counted)."},
    {"entity_offsets",
     "index [G + 1]: graph g's entities are [entity_offsets[g], entity_offsets[g + 1]); entity ids are batch-global."},
    {"entity_role",
     "uint8 [E]: the role of each entity, an index into entity_role_names. Absent when the entity arrays are implicit "
     "(FlatBatch.implicit_entities)."},
    {"entity_ref", "index [E]: an object's instance-local id, -1 for the auxiliary entity. Absent when implicit."},
    {"relation_counts",
     "index [G, R]: the tuples of each (graph, relation) cell, indexed (g, r) in both layouts. Absent with "
     "emit_counts=False."},
    {"relation_offsets",
     "index [G * R + 1]: cell (g, r) is the segment s = g * R + r (layout 'graph') or r * G + g (layout 'relation') and "
     "holds the tuples [relation_offsets[s], relation_offsets[s + 1])."},
    {"relation_arg_offsets", "index [G * R + 1]: the same offsets into relation_args."},
    {"relation_args", "index [A]: the entities of the tuples, tuple after tuple, `arity` per tuple of the relation."},
};
}  // namespace

void bind_encoding(nb::module_& parent)
{
    nb::module_ module = parent.def_submodule("_encoding", "Native classical relational encodings");

    nb::class_<PyEncodingSchema> schema_class(
        module, "EncodingSchema",
        "The relation catalog of an encoding: the relations a batch of encoded states holds, sorted by name, with their "
        "arities and slot roles. It depends on the domain of the task or table and on the options, never on the "
        "instances, so tables over the same domain agree on the relation ids.");
    static const std::string schema_doc =
        std::string("The catalog of the domain of `table` (a Task, TaskHandle or TaskTable) under the options. ") +
        k_option_doc;
    define_init(
        schema_class,
        [](PyEncodingSchema* self, const TableRef& reference, const encoding::FlatOptions& options)
        { new (self) PyEncodingSchema{encoding::EncodingSchema::create(*reference.table, options)}; },
        schema_doc.c_str());
    define_schema_properties(schema_class,
                             [](const PyEncodingSchema& self) -> const encoding::EncodingSchema& { return *self.schema; });

    nb::class_<PyFlatBatch> batch(
        module, "FlatBatch",
        "The graphs of a batch of states: one graph per row, with the tuples of every relation of the schema. The arrays "
        "are read-only and keep the batch alive; every export shares their memory.");
    batch.def_prop_ro("num_graphs", [](const PyFlatBatch& self) { return self.arrays->scalar("num_graphs"); })
        .def_prop_ro("num_relations", [](const PyFlatBatch& self) { return self.arrays->scalar("num_relations"); })
        .def_prop_ro(
            "schema", [](const PyFlatBatch& self) { return PyEncodingSchema{self.schema}; },
            "The EncodingSchema the batch was encoded for.")
        .def_prop_ro(
            "implicit_entities", [](const PyFlatBatch& self) { return self.schema->implicit_entity_arrays(); },
            "True if the batch has no entity_role and entity_ref: entity j of graph g is an object with role 0 and ref j "
            "if j < num_objects[g], else the graph's auxiliary entity (role 8, ref -1).")
        .def_prop_ro(
            "sizes",
            [](const PyFlatBatch& self) -> std::optional<nb::dict>
            {
                if (!self.sizes)
                    return std::nullopt;
                nb::dict out;
                out["entities"] = self.sizes->entities;
                out["tuples"] = self.sizes->tuples;
                out["args"] = self.sizes->args;
                out["overflow"] = (self.sizes->overflow & encoding::k_overflow_capacity) != 0;
                out["index_overflow"] = (self.sizes->overflow & encoding::k_overflow_index) != 0;
                return out;
            },
            "After an encoding with capacities: the true entities, tuples and args of the batch and whether it "
            "overflowed the capacities (`overflow`) or the index type (`index_overflow`); None otherwise.")
        .def(
            "as_dict",
            [](const PyFlatBatch& self, Arg<ann::Framework> framework)
            {
                const Framework target = framework.is_none() ? self.framework : parse_framework(framework);
                return ResultDict(nb::cast<nb::dict>(export_bundle(self.arrays, target, default_words(target))));
            },
            "framework"_a = nb::none(),
            "The scalars (version, num_graphs, num_relations, layout, index_dtype, schema_fingerprint, statics) and the "
            "arrays of the batch, zero-copy, in the framework (default: the batch's).");
    for (const auto& [name, doc] : k_batch_arrays)
        batch.def_prop_ro(name, [name = name](const PyFlatBatch& self) { return ArrayOut(batch_array(self, name)); }, doc);

    nb::class_<PyFlatEncoder> encoder_class(
        module, "FlatRelationEncoder",
        "Encodes states into relational graphs: one graph per state, with the instance's objects as entities (plus one "
        "auxiliary entity when a nullary tuple exists) and, for every relation of the schema, the tuples of the state's "
        "fluent, derived and static atoms. A graph holds nothing that its state and instance do not have.");
    static const std::string encoder_doc =
        std::string("The encoder of a Task, TaskHandle or one-domain TaskTable under the options of EncodingSchema. "
                    "Options that this encoder does not encode raise ValueError (\"mymyr: encoding: ...\"); among them "
                    "are goal derivations, whose default is therefore refused: pass `goal_derivations=()`. ") +
        k_option_doc;
    define_init(
        encoder_class,
        [](PyFlatEncoder* self, const TableRef& reference, const encoding::FlatOptions& options)
        { new (self) PyFlatEncoder{reference, encoding::FlatEncoder::create(reference.table, options)}; },
        encoder_doc.c_str());
    define_schema_properties(encoder_class,
                             [](const PyFlatEncoder& self) -> const encoding::EncodingSchema& { return self.encoder->schema(); });
    encoder_class.def_prop_ro("table", [](const PyFlatEncoder& self) { return TableArg(self.reference.obj); })
        .def_prop_ro(
            "schema", [](const PyFlatEncoder& self) { return PyEncodingSchema{self.encoder->schema_ptr()}; },
            "The EncodingSchema of the encoder.")
        .def("encode", &encode, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "derived"_a = nb::none(),
             "framework"_a = nb::none(), "threads"_a = 1, "capacities"_a = nb::none(),
             "Encodes the rows `states` (an array of state words [N, W] of any DLPack framework, u64 words or u32 halves "
             "[N, 2W], or a State or list of States) into one graph each. task_ids: the instance of each row (required "
             "for a table of several instances). derived: the derived atom words of each row [N, D] (default: evaluated "
             "here). The arrays of the result are in `framework` (default: that of `states`; NumPy for States). Raises "
             "ValueError for invalid input. threads: graphs are split over this many threads (0: one per hardware thread); "
             "the result does not depend on it. capacities: a dict with the keys 'entities', 'tuples' and 'args'; the "
             "batch then has relation_args and the entity arrays of exactly the capacities' sizes (the first "
             "sizes['entities'] / sizes['args'] entries hold the encoding, the rest is zero) and `sizes` reports the true "
             "sizes. A batch that does not fit its capacities (or the index type) is all zero and has sizes['overflow'] "
             "set; nothing raises.");
}
}
