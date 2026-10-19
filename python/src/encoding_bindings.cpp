#include "arrays.hpp"
#include "device_hooks.hpp"
#include "py_expansion.hpp"
#include "py_pool.hpp"
#include "py_table.hpp"
#include "rl_imports.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/schema.hpp"
#include "mymyr/encoding/stream.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <variant>
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
struct EncodingGoals
{
};
struct EncodingActions
{
};
struct EncodingSubgoals
{
};
struct EncodingHistory
{
};
struct EncodingGraphDescription
{
};
struct EncodingHorizonView
{
};
struct EncodingBatch
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
struct type_caster<mymyr::python::ann::EncodingGoals>
{
    static constexpr auto Name = const_name(
        "tuple[mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack] | tuple[mymyr._typing.SupportsDLPack, "
        "mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack]");
};
template<>
struct type_caster<mymyr::python::ann::EncodingSubgoals>
{
    static constexpr auto Name = const_name(
        "tuple[mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack] | tuple[mymyr._typing.SupportsDLPack, "
        "mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack]");
};
template<>
struct type_caster<mymyr::python::ann::EncodingHistory>
{
    static constexpr auto Name = const_name(
        "tuple[mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack] | "
        "tuple[mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, "
        "mymyr._typing.SupportsDLPack | None] | tuple[mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack, "
        "mymyr._typing.SupportsDLPack, mymyr._typing.SupportsDLPack | None, mymyr._typing.SupportsDLPack, "
        "mymyr._typing.SupportsDLPack]");
};
template<>
struct type_caster<mymyr::python::ann::EncodingGraphDescription>
{
    static constexpr auto Name = const_name("mymyr._typing.GraphDescription");
};
template<>
struct type_caster<mymyr::python::ann::EncodingHorizonView>
{
    static constexpr auto Name = const_name("mymyr._typing.SupportsHorizonView");
};
template<>
struct type_caster<mymyr::python::ann::EncodingBatch>
{
    static constexpr auto Name = const_name("FlatBatch | mymyr._typing.DeviceFlatBatch");
};
template<>
struct type_caster<mymyr::python::ann::EncodingActions>
{
    static constexpr auto Name = const_name("mymyr._core._rl.Expansion");
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
using GoalsArg = Arg<ann::EncodingGoals>;
using ActionsArg = Arg<ann::EncodingActions>;
using SubgoalsArg = Arg<ann::EncodingSubgoals>;
using HistoryArg = Arg<ann::EncodingHistory>;

struct PyEncodingSchema
{
    std::shared_ptr<const encoding::EncodingSchema> schema;
};

struct PyFlatEncoder
{
    TableRef reference;
    std::shared_ptr<const encoding::FlatEncoder> encoder;
    std::mutex pool_mutex;  // guards the cached pool
    struct CachedPool
    {
        std::shared_ptr<ThreadPool> pool;
        u32 threads = 0;
        pid_t process = 0;
    } cached;

    PyFlatEncoder(TableRef reference_, std::shared_ptr<const encoding::FlatEncoder> encoder_)
        : reference(std::move(reference_)), encoder(std::move(encoder_))
    {
    }

    /// The pool of `threads` members (at least 1) that the calls of this encoder share: one pool, rebuilt when the
    /// count changes. A pool made by another process (before a fork) has no workers here and is dropped without being
    /// joined.
    std::shared_ptr<ThreadPool> pool(u32 threads)
    {
        if (threads == 1)
        {
            // one member never runs a job on the pool: the calling thread encodes
            static const auto single = std::make_shared<ThreadPool>(1);
            return single;
        }
        std::lock_guard lock(pool_mutex);
        const pid_t process = getpid();
        if (cached.pool && cached.process != process)
            new std::shared_ptr<ThreadPool>(std::move(cached.pool));  // intentionally leaked: its workers are gone
        if (!cached.pool || cached.threads != threads)
            cached = {std::make_shared<ThreadPool>(threads), threads, process};
        return cached.pool;
    }
};

/// The horizon encoder: the state of a FlatRelationEncoder (the two classes share it; the horizon encoder has no streams).
struct PyFlatHorizonEncoder : PyFlatEncoder
{
    using PyFlatEncoder::PyFlatEncoder;
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
                                     nb::handle index_dtype, bool emit_counts, nb::handle entity_arrays,
                                     nb::handle horizon_mode, nb::handle root_policy, bool parent_relation,
                                     bool history, nb::handle history_max_steps, bool horizon_optional)
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
        const std::string allowed = horizon_optional ? "'full', 'delta', 'action' or None" : "'full', 'delta', 'action'";
        options.horizon.mode = choice(horizon_mode, "horizon_mode", parse_horizon_mode, allowed);
        if (options.horizon.mode == HorizonMode::None)  // None is the spelling of no horizon
            option_error("horizon_mode", allowed, horizon_mode);
    }
    if (!root_policy.is_none())
        options.horizon.root =
            choice(root_policy, "root_policy", parse_root_policy, "'include', 'exclude', 'encode_only' or None");
    options.horizon.parent_relation = parent_relation;
    options.history = history;
    options.history_max_steps = opt_int_arg<u32>(history_max_steps, "history_max_steps", 1);
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
    out["history"] = o.history;
    out["history_max_steps"] = o.history_max_steps ? nb::object(nb::cast(*o.history_max_steps)) : nb::none();
    return out;
}

constexpr const char* k_option_doc =
    "goal_derivations: the variants of the goal relations (plain, satisfied, unsatisfied; added_satisfied and "
    "added_unsatisfied for the delta horizon). max_goal_level: subgoal levels beyond the goal (0 to 3): the "
    "relations `[sg]`, `[ssg]`, `[sssg]` of each level hold the literals of the `subgoals` layers of a row. history: "
    "the history relations `[+]p[hist]` / `[-]p[hist]` hold the literals of the `history` steps of a row; "
    "history_max_steps: only the newest this many steps of a row are encoded (None: all). "
    "include_static / include_derived: the state relations of the static (types included) / derived "
    "predicates. nullary: relations of the predicates without arguments (their atoms are tuples over the "
    "graph's auxiliary entity). equality: relations of the equality predicate `=`. predicate_virtual_nodes: a predicate "
    "slot in the predicate relations' tuples: a graph gets a virtual entity (role predicate_virtual) for each predicate that "
    "has a tuple in it, numbered in catalog order, and none for the others; its entity_ref is the id of the predicate's "
    "state relation. include_actions: one relation per action schema. target_sources: "
    "which of goal, subgoal, action and history literals become entities that tuples refer to (goal and subgoal "
    "targets need the plain goal derivation, subgoal targets max_goal_level > 0, action targets include_actions, "
    "history targets history). layout, "
    "index_dtype: how a batch is stored; layout='relation' stores the tuples relation by relation across the graphs "
    "(cell (g, r) is segment r * G + g instead of g * R + r). emit_counts=False omits relation_counts (the tuples of "
    "a segment are the difference of two relation_offsets). entity_arrays='auto' omits entity_role and entity_ref "
    "when every entity is an object or the auxiliary entity (no target_sources, predicate_virtual_nodes, "
    "include_actions, history or horizon_mode). Raises ValueError for invalid combinations and for numeric tasks.";

constexpr const char* k_horizon_option_doc =
    " horizon_mode, root_policy, parent_relation: the horizon variants (root_policy defaults to 'include' and needs a "
    "horizon_mode for any other value; see FlatHorizonEncoder).";

/// Whether a class that is built from a table takes the horizon options: EncodingSchema takes them as options,
/// FlatRelationEncoder does not take them, FlatHorizonEncoder requires `horizon_mode`.
enum class HorizonArgs
{
    Optional,
    Absent,
    Required,
};

// the options other than the horizon's, in the order of the signatures
#define MYMYR_ENCODING_ARGS                                                                                            \
    TableArg table, TextSequenceArg goal_derivations, IntArg max_goal_level, bool include_static, bool include_derived, \
        bool nullary, bool equality, bool predicate_virtual_nodes, bool include_actions, TextSequenceArg target_sources, \
        Arg<ann::EncodingLayout> layout, Arg<ann::EncodingIndexDtype> index_dtype, bool emit_counts,                    \
        Arg<ann::EncodingEntityArrays> entity_arrays
#define MYMYR_ENCODING_DEFAULTS                                                                                        \
    "goal_derivations"_a = nb::make_tuple("plain", "satisfied"), "max_goal_level"_a = 0, "include_static"_a = true,     \
    "include_derived"_a = true, "nullary"_a = true, "equality"_a = false, "predicate_virtual_nodes"_a = false,          \
    "include_actions"_a = true, "target_sources"_a = nb::tuple(), "layout"_a = "graph", "index_dtype"_a = "int32",      \
    "emit_counts"_a = true, "entity_arrays"_a = "always"

/// The constructor of the classes that are built from a table and the encoding options (EncodingSchema,
/// FlatRelationEncoder, FlatHorizonEncoder): the keyword arguments are parsed here once. `make(self, reference,
/// options)` constructs the object.
template<HorizonArgs H, class Py, class Make>
void define_init(nb::class_<Py>& cls, Make make, const char* doc)
{
    auto construct = [make](Py* self, MYMYR_ENCODING_ARGS, nb::handle horizon_mode, nb::handle root_policy,
                            bool parent_relation, bool history, nb::handle history_max_steps)
    {
        const TableRef reference = table_of(table);
        if (H == HorizonArgs::Required && horizon_mode.is_none())
            throw nb::value_error(
                "mymyr: encoding: FlatHorizonEncoder needs a horizon_mode ('full', 'delta' or 'action'); "
                "FlatRelationEncoder encodes states without a horizon");
        const encoding::FlatOptions options = schema_options(
            max_goal_level, include_static, include_derived, nullary, equality, predicate_virtual_nodes, include_actions,
            goal_derivations, target_sources, layout, index_dtype, emit_counts, entity_arrays, horizon_mode, root_policy,
            parent_relation, history, history_max_steps, H != HorizonArgs::Required);
        make(self, reference, options);
    };
    if constexpr (H == HorizonArgs::Optional)
    {
        cls.def(
            "__init__",
            [construct](Py* self, MYMYR_ENCODING_ARGS, Arg<ann::EncodingHorizonMode> horizon_mode,
                        Arg<ann::EncodingRootPolicy> root_policy, bool parent_relation, bool history,
                        IntArg history_max_steps)
            {
                construct(self, table, goal_derivations, max_goal_level, include_static, include_derived, nullary, equality,
                          predicate_virtual_nodes, include_actions, target_sources, layout, index_dtype, emit_counts,
                          entity_arrays, horizon_mode, root_policy, parent_relation, history, history_max_steps);
            },
            "table"_a, nb::kw_only(), MYMYR_ENCODING_DEFAULTS, "horizon_mode"_a = nb::none(),
            "root_policy"_a = nb::none(), "parent_relation"_a = false, "history"_a = false, "history_max_steps"_a.none() = nb::none(), doc);
    }
    else if constexpr (H == HorizonArgs::Absent)
    {
        cls.def(
            "__init__",
            [construct](Py* self, MYMYR_ENCODING_ARGS, bool history, IntArg history_max_steps)
            {
                construct(self, table, goal_derivations, max_goal_level, include_static, include_derived, nullary, equality,
                          predicate_virtual_nodes, include_actions, target_sources, layout, index_dtype, emit_counts,
                          entity_arrays, nb::none(), nb::none(), false, history, history_max_steps);
            },
            "table"_a, nb::kw_only(), MYMYR_ENCODING_DEFAULTS, "history"_a = false, "history_max_steps"_a.none() = nb::none(), doc);
    }
    else
    {
        cls.def(
            "__init__",
            [construct](Py* self, TableArg table, Arg<ann::EncodingHorizonMode> horizon_mode,
                        TextSequenceArg goal_derivations, IntArg max_goal_level, bool include_static, bool include_derived,
                        bool nullary, bool equality, bool predicate_virtual_nodes, bool include_actions,
                        TextSequenceArg target_sources, Arg<ann::EncodingLayout> layout,
                        Arg<ann::EncodingIndexDtype> index_dtype, bool emit_counts,
                        Arg<ann::EncodingEntityArrays> entity_arrays, Arg<ann::EncodingRootPolicy> root_policy,
                        bool parent_relation)
            {
                construct(self, table, goal_derivations, max_goal_level, include_static, include_derived, nullary, equality,
                          predicate_virtual_nodes, include_actions, target_sources, layout, index_dtype, emit_counts,
                          entity_arrays, horizon_mode, root_policy, parent_relation, false, nb::none());
            },
            "table"_a, nb::kw_only(), "horizon_mode"_a.none(), MYMYR_ENCODING_DEFAULTS, "root_policy"_a = nb::none(),
            "parent_relation"_a = false, doc);
    }
}
#undef MYMYR_ENCODING_ARGS
#undef MYMYR_ENCODING_DEFAULTS

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
        .def_prop_ro("num_relations", [schema_of](const Py& self) { return schema_of(self).num_relations(); },
                     "The number of relations in the catalog.")
        .def_prop_ro(
            "implicit_entities", [schema_of](const Py& self) { return schema_of(self).implicit_entity_arrays(); },
            "Whether the entity arrays are implicit under these options: entity_role and entity_ref are then omitted "
            "from batches (entity_arrays='auto' and every entity is an object or the auxiliary entity).")
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
                     },
                     "The names of the slot roles (relation_slot_roles holds indices into them).")
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
    spec.shape = info->shape.vec();
    spec.readonly = true;
    return export_array(std::move(spec), batch.framework, default_words(batch.framework));
}

using PoolArg = Arg<PyPool>;
using CapacitiesArg = std::optional<nb::typed<nb::dict, std::string, std::uint64_t>>;

encoding::BatchShape capacities_of(const nb::dict& value, const encoding::EncodingSchema& schema)
{
    encoding::BatchShape out;
    bool entities = false, graphs = false, args = false, targets = false, history = false;
    for (auto [key, item] : value)
    {
        const std::string name = text_of(key, "capacities key");
        if (name == "entities")
            out.entities = int_arg<u64>(item, "capacities['entities']"), entities = true;
        else if (name == "graphs")
            out.graphs = int_arg<u64>(item, "capacities['graphs']"), graphs = true;
        else if (name == "args")
            out.args = int_arg<u64>(item, "capacities['args']"), args = true;
        else if (name == "targets")
            out.targets = int_arg<u64>(item, "capacities['targets']"), targets = true;
        else if (name == "history")
            out.history = int_arg<u64>(item, "capacities['history']"), history = true;
        else
            throw nb::value_error(("mymyr: encoding: capacities has the unknown key '" + name +
                                   "'; the keys are 'graphs', 'entities', 'args', 'targets' and 'history'")
                                      .c_str());
    }
    if (!(entities && graphs && args))
        throw nb::value_error("mymyr: encoding: capacities needs the keys 'entities', 'graphs' and 'args'");
    if (!targets && schema.has_targets())
        throw nb::value_error("mymyr: encoding: the schema has targets: capacities needs the key 'targets' as well");
    if (!history && schema.has_history())
        throw nb::value_error("mymyr: encoding: the schema has history: capacities needs the key 'history' as well");
    return out;
}

/// The goal masks of the rows from `goals`, a pair (goal_pos, goal_neg) or a quadruple that adds the masks over the
/// derived slots: arrays [N, W] or [1, W] (one goal for every row) of 64-bit words or 32-bit halves.
void import_goals(rlimp::Imports& imports, nb::handle goals, u64 rows, encoding::EncodeInput& input)
{
    constexpr const char* k_shape = "mymyr: encoding: goals must be (goal_pos, goal_neg) or (goal_pos, goal_neg, "
                                    "goal_derived_pos, goal_derived_neg)";
    if (nb::isinstance<nb::str>(goals) || !PyObject_HasAttrString(goals.ptr(), "__len__"))
        throw nb::type_error(k_shape);
    const usize size = nb::len(goals);
    if (size != 2 && size != 4)
        throw nb::value_error(k_shape);
    static constexpr const char* names[] = {"goals[0] (goal_pos)", "goals[1] (goal_neg)", "goals[2] (goal_derived_pos)",
                                            "goals[3] (goal_derived_neg)"};
    const u64* masks[4] = {};
    u64 strides[2] = {};
    u32 widths[2] = {};
    i64 extents[2] = {};
    for (usize k = 0; k < size; ++k)
    {
        std::vector<i64> lead;
        u32 words = 0;
        masks[k] = imports.words_nd(goals[k], names[k], {-1}, false, lead, words);
        if (!masks[k])
            throw nb::type_error((std::string("mymyr: encoding: ") + names[k] + " must be an array, not None").c_str());
        if (lead[0] != 1 && static_cast<u64>(lead[0]) != rows)
            throw nb::value_error((std::string("mymyr: encoding: ") + names[k] + " has " + std::to_string(lead[0]) +
                                   " rows: one per state (" + std::to_string(rows) + ") or a single shared row")
                                      .c_str());
        const usize pair = k / 2;
        if (k % 2 == 1 && (widths[pair] != words || extents[pair] != lead[0]))
            throw nb::value_error(
                (std::string("mymyr: encoding: ") + names[k] + " has another shape than " + names[k - 1]).c_str());
        widths[pair] = words;
        extents[pair] = lead[0];
        strides[pair] = (lead[0] == 1 && rows != 1) ? 0 : words;
    }
    input.goal_pos = masks[0];
    input.goal_neg = masks[1];
    input.goal_words = widths[0];
    input.goal_stride = strides[0];
    if (size == 4)
    {
        input.goal_derived_pos = masks[2];
        input.goal_derived_neg = masks[3];
        input.goal_derived_words = widths[1];
        input.goal_derived_stride = strides[1];
    }
}

/// One mask array [N or 1, L, W] of the layers of a call: its extents and word width.
struct LayerArray
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 layers = 0, words = 0;
};

LayerArray import_layer_array(rlimp::Imports& imports, nb::handle value, const char* name, u64 rows, bool may_share)
{
    LayerArray out;
    std::vector<i64> lead;
    out.data = imports.words_nd(value, name, {-1, -1}, false, lead, out.words);
    if (!out.data)
        throw nb::type_error((std::string("mymyr: encoding: ") + name + " must be an array, not None").c_str());
    if (static_cast<u64>(lead[0]) != rows && !(may_share && lead[0] == 1))
        throw nb::value_error((std::string("mymyr: encoding: ") + name + " has " + std::to_string(lead[0]) +
                               " rows: one per state (" + std::to_string(rows) +
                               (may_share ? ") or a single shared row" : ")"))
                                  .c_str());
    out.rows = static_cast<u64>(lead[0]);
    out.layers = static_cast<u32>(lead[1]);
    return out;
}

/// The positive and negative masks at `at` and `at + 1` of `items` as the layers of `l`, with the same extents.
void import_layer_pair(rlimp::Imports& imports, nb::handle items, usize at, const char* const* names, u64 rows,
                       bool may_share, bool derived, encoding::LayerInput& l)
{
    const LayerArray pos = import_layer_array(imports, items[at], names[at], rows, may_share);
    const LayerArray neg = import_layer_array(imports, items[at + 1], names[at + 1], rows, may_share);
    if (pos.rows != neg.rows || pos.layers != neg.layers || pos.words != neg.words)
        throw nb::value_error(
            (std::string("mymyr: encoding: ") + names[at + 1] + " has another shape than " + names[at]).c_str());
    const u64 stride = (pos.rows == 1 && rows != 1) ? 0 : u64{pos.layers} * pos.words;
    if (!derived)
    {
        l.pos = pos.data;
        l.neg = neg.data;
        l.stride = stride;
        l.layers = pos.layers;
        l.words = pos.words;
        return;
    }
    if (pos.layers != l.layers)
        throw nb::value_error((std::string("mymyr: encoding: ") + names[at] + " has " + std::to_string(pos.layers) +
                               " layers, " + names[0] + " has " + std::to_string(l.layers))
                                  .c_str());
    l.derived_pos = pos.data;
    l.derived_neg = neg.data;
    l.derived_stride = (pos.rows == 1 && rows != 1) ? 0 : u64{pos.layers} * pos.words;
    l.derived_words = pos.words;
}

usize layer_tuple_size(nb::handle value, const char* shape_message, std::initializer_list<usize> allowed)
{
    if (nb::isinstance<nb::str>(value) || !PyObject_HasAttrString(value.ptr(), "__len__"))
        throw nb::type_error(shape_message);
    const usize size = nb::len(value);
    for (const usize a : allowed)
        if (size == a)
            return size;
    throw nb::value_error(shape_message);
}

/// The subgoal layers of the rows from `subgoals`, a pair (pos, neg) or a quadruple that adds the masks over the
/// derived slots: arrays [N, L, W] or [1, L, W] (the layers of every row) of 64-bit words or 32-bit halves.
void import_subgoals(rlimp::Imports& imports, nb::handle subgoals, u64 rows, encoding::EncodeInput& input)
{
    constexpr const char* k_shape =
        "mymyr: encoding: subgoals must be (pos, neg) or (pos, neg, derived_pos, derived_neg)";
    const usize size = layer_tuple_size(subgoals, k_shape, {2, 4});
    static constexpr const char* names[] = {"subgoals[0] (pos)", "subgoals[1] (neg)", "subgoals[2] (derived_pos)",
                                            "subgoals[3] (derived_neg)"};
    import_layer_pair(imports, subgoals, 0, names, rows, true, false, input.subgoals);
    if (size == 4)
        import_layer_pair(imports, subgoals, 2, names, rows, true, true, input.subgoals);
}

/// The history of the rows from `history`, (pos, neg, dt), (pos, neg, dt, count) or (pos, neg, dt, count, derived_pos,
/// derived_neg): masks [N, H, W], dt [N, H] and count [N] (None: every step valid).
void import_history(rlimp::Imports& imports, std::vector<i32>& dt_copy, std::vector<i32>& count_copy,
                    nb::handle history, u64 count_rows, encoding::HistoryInput& out)
{
    constexpr const char* k_shape =
        "mymyr: encoding: history must be (pos, neg, dt), (pos, neg, dt, count) or (pos, neg, dt, count, derived_pos, "
        "derived_neg)";
    const usize size = layer_tuple_size(history, k_shape, {3, 4, 6});
    static constexpr const char* names[] = {"history[0] (pos)",         "history[1] (neg)",
                                            "history[2] (dt)",          "history[3] (count)",
                                            "history[4] (derived_pos)", "history[5] (derived_neg)"};
    import_layer_pair(imports, history, 0, names, count_rows, false, false, out.masks);
    const i64 steps = out.masks.layers;
    out.dt = imports.ints32(history[2], names[2], {static_cast<i64>(count_rows), steps}, dt_copy);
    if (!out.dt && count_rows * static_cast<u64>(steps) != 0)
        throw nb::type_error("mymyr: encoding: history[2] (dt) must be an array, not None");
    if (size >= 4)
        out.count = imports.ints32(history[3], names[3], {static_cast<i64>(count_rows)}, count_copy);
    if (size == 6)
        import_layer_pair(imports, history, 4, names, count_rows, false, true, out.masks);
}

/// The rows of one call, imported from the Python objects: the arrays are read where they are, and the members keep
/// them alive.
struct ImportedRows
{
    StateBatch states;
    std::vector<i32> ids_copy, dt_copy, count_copy, edge_binding_copy;
    std::vector<std::vector<i32>> horizon_copies;  // the converted offset and edge arrays of a horizon
    std::shared_ptr<void> ids_keep;
    rlimp::Imports imports{-1, dl::k_stream_default, false};
    ExpansionBuffers actions;
    encoding::EncodeInput input;
};

void import_rows(ImportedRows& rows, const encoding::FlatEncoder& encoder, StatesLike state_arg, TaskIdsArg task_ids,
                 DerivedArg derived_arg, GoalsArg goals_arg, ActionsArg actions_arg, SubgoalsArg subgoals_arg,
                 HistoryArg history_arg)
{
    rows.states = import_states(state_arg, packed_words(state_arg));
    const StateBatch& states = rows.states;
    const rl::TaskTable& table = encoder.table();
    const i32* ids = import_task_ids(task_ids, states.view.rows, table.size(), rows.ids_copy, rows.ids_keep);
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

    encoding::EncodeInput& input = rows.input;
    input.states = states.view;
    input.task_ids = ids;
    std::vector<i64> extents;
    u32 derived_words = 0;
    input.derived = rows.imports.words_nd(derived_arg, "derived", {static_cast<i64>(states.view.rows)}, false, extents,
                                          derived_words);
    input.derived_stride = derived_words;
    if (!goals_arg.is_none())
        import_goals(rows.imports, goals_arg, states.view.rows, input);
    if (!subgoals_arg.is_none())
        import_subgoals(rows.imports, subgoals_arg, states.view.rows, input);
    if (!history_arg.is_none())
        import_history(rows.imports, rows.dt_copy, rows.count_copy, history_arg, states.view.rows, input.history);

    if (!actions_arg.is_none())
    {
        rows.actions = expansion_buffers(actions_arg);
        const rl::TaskTable& made_over = *rows.actions.table;
        if (rows.actions.states != states.view.rows)
            throw nb::value_error(("mymyr: encoding: actions expand " + std::to_string(rows.actions.states) +
                                   " states, the batch has " + std::to_string(states.view.rows) + " rows")
                                      .c_str());
        if (made_over.schema_names() != table.schema_names() || made_over.schema_arities() != table.schema_arities())
            throw nb::value_error("mymyr: encoding: actions were expanded over another domain than the encoder's");
        for (u64 row = 0; row < states.view.rows; ++row)
        {
            const u32 instance = ids ? static_cast<u32>(ids[row]) : 0;
            if (rows.actions.task_ids[row] != static_cast<i32>(instance))
                throw nb::value_error(("mymyr: encoding: the actions of state " + std::to_string(row) +
                                       " were expanded for instance " + std::to_string(rows.actions.task_ids[row]) +
                                       ", the batch has instance " + std::to_string(instance))
                                          .c_str());
            // the rows' bindings and schemas are those of the instance they were expanded for: it must be the encoder's
            if (instance >= made_over.size())
                throw nb::value_error(("mymyr: encoding: the actions were expanded over a table without instance " +
                                       std::to_string(instance))
                                          .c_str());
            const Task& expected = *table.task(instance);
            const Task& given = *made_over.task(instance);
            if (given.uid() != expected.uid() && (given.fingerprint() != expected.fingerprint() ||
                                                  given.encoding_fingerprint() != expected.encoding_fingerprint()))
                throw nb::value_error(("mymyr: encoding: the actions of state " + std::to_string(row) +
                                       " were expanded over another task than instance " + std::to_string(instance) +
                                       " of the encoder's table: it differs in fingerprint or atom encoding")
                                          .c_str());
        }
        input.actions = &rows.actions.view;
    }
}

/// Runs `work(pool)` on the pool of a call (`pool=`, else the encoder's own of `threads` members) with the GIL released.
template<class Work>
void run_on_pool(PyFlatEncoder& self, u32 threads, PoolArg pool_in, Work&& work)
{
    PyPool* held = pool_arg(pool_in);
    std::shared_ptr<ThreadPool> own;
    if (!held)
        own = self.pool(threads == 0 ? hardware_threads() : threads);
    nb::gil_scoped_release release;
    std::unique_lock<std::mutex> hold;
    if (held)
        hold = std::unique_lock(held->mutex);
    work(held ? *held->pool : *own);
}

/// Encodes the imported rows on the pool of the call.
PyFlatBatch run_encoding(PyFlatEncoder& self, ImportedRows& rows, Framework framework, u32 threads, PoolArg pool_in,
                         const std::optional<encoding::BatchShape>& capacities)
{
    PyFlatBatch output;
    output.framework = framework;
    output.schema = self.encoder->schema_ptr();
    encoding::EncodeScratch& scratch = encoding::EncodeScratch::of_this_thread();
    run_on_pool(self, threads, pool_in,
                [&](ThreadPool& pool)
                {
                    if (capacities)
                    {
                        encoding::FlatSizes sizes;
                        output.arrays = self.encoder->encode_capacity(rows.input, *capacities, pool, scratch, sizes);
                        output.sizes = sizes;
                    }
                    else
                        output.arrays = self.encoder->encode(rows.input, pool, scratch);
                });
    return output;
}

/// A NumPy array, which is never on the device: the check that keeps host encodes at their cost.
bool is_numpy_array(nb::handle value)
{
    static const nb::handle ndarray = nb::object(nb::module_::import_("numpy").attr("ndarray")).release();
    return value.type().ptr() == ndarray.ptr();
}

/// An array object that lives in CUDA memory: DLPack's device, or a __cuda_array_interface__.
bool is_device_array(nb::handle value)
{
    const PyTypeObject* type = Py_TYPE(value.ptr());
    if (value.is_none() || type == &PyList_Type || type == &PyTuple_Type || type == &PyLong_Type ||
        is_numpy_array(value) || type == reinterpret_cast<PyTypeObject*>(nb::type<PyState>().ptr()))
        return false;
    const nb::handle cls = value.type();
    if (PyObject_HasAttrString(cls.ptr(), "__dlpack_device__"))
        return is_cuda_array(value);
    return PyObject_HasAttrString(cls.ptr(), "__cuda_array_interface__") ||
           PyObject_HasAttrString(value.ptr(), "__cuda_array_interface__");
}

/// Whether an argument of `encode` is a CUDA device array; with `masks`, a tuple of masks (the goals, the subgoal layers
/// or the history) of which one is.
bool on_device(nb::handle value, bool masks = false)
{
    if (!masks)
        return is_device_array(value);
    if (nb::isinstance<nb::tuple>(value) || nb::isinstance<nb::list>(value))
        for (nb::handle item : value)
            if (is_device_array(item))
                return true;
    return false;
}

/// The rows of a call that has device arrays are encoded on the device (mymyr.encoding._device).
Arg<ann::EncodingBatch> encode(PyFlatEncoder& self, StatesLike state_arg, TaskIdsArg task_ids, DerivedArg derived_arg,
                               GoalsArg goals_arg, ActionsArg actions_arg, SubgoalsArg subgoals_arg,
                               HistoryArg history_arg, Arg<ann::Framework> framework, IntArg threads_in,
                               PoolArg pool_in, CapacitiesArg capacities_in)
{
    const u32 threads = threads_arg(threads_in);
    if (on_device(state_arg) || on_device(task_ids) || on_device(derived_arg) || on_device(goals_arg, true) ||
        on_device(subgoals_arg, true) || on_device(history_arg, true))
    {
        return Arg<ann::EncodingBatch>(
            nb::module_::import_("mymyr.encoding._device")
                .attr("encode")(nb::find(&self), state_arg, task_ids, derived_arg, goals_arg, actions_arg,
                                subgoals_arg, history_arg, framework, threads_in, pool_in,
                                capacities_in ? nb::object(*capacities_in) : nb::none()));
    }
    const std::optional<encoding::BatchShape> capacities =
        capacities_in ? std::optional(capacities_of(*capacities_in, self.encoder->schema()))
                      : std::nullopt;
    ImportedRows rows;
    import_rows(rows, *self.encoder, state_arg, task_ids, derived_arg, goals_arg, actions_arg, subgoals_arg,
                history_arg);
    return Arg<ann::EncodingBatch>(
        nb::cast(run_encoding(self, rows, parse_framework(framework, state_arg), threads, pool_in, capacities)));
}

/// The nodes and edges of a horizon batch, imported from the attributes of `view` (see HorizonView): the node rows go to
/// rows.states, the offsets and edges are read where they are or converted into the copies held by `rows`.
void import_horizon(ImportedRows& rows, const encoding::FlatEncoder& encoder, nb::handle view, DerivedArg derived_arg,
                    GoalsArg goals_arg, SubgoalsArg subgoals_arg)
{
    auto field = [&](const char* name)
    {
        if (!PyObject_HasAttrString(view.ptr(), name))
            throw nb::attribute_error(
                (std::string("mymyr: encoding: the horizon view has no attribute '") + name +
                 "' (nodes, node_offsets, edge_offsets, edge_src, edge_dst, edge_schema, edge_binding, task_ids)")
                    .c_str());
        return nb::getattr(view, name);
    };
    const nb::object nodes = field("nodes");
    rows.states = import_states(nodes, packed_words(nodes));
    const StateBatch& states = rows.states;
    const rl::TaskTable& table = encoder.table();
    encoding::HorizonInput& horizon = rows.input.horizon;
    horizon.nodes = states.view;

    rows.horizon_copies.resize(5);
    std::vector<i64> extent;
    const nb::object node_offsets = field("node_offsets");
    if (node_offsets.is_none())
        throw nb::type_error("mymyr: encoding: node_offsets must be an array, not None");
    horizon.node_offsets = rows.imports.ints32(node_offsets, "node_offsets", {-1}, rows.horizon_copies[0], &extent);
    if (extent[0] < 1)
        throw nb::value_error("mymyr: encoding: node_offsets must have G + 1 entries (G graphs)");
    horizon.graphs = static_cast<u64>(extent[0] - 1);
    const nb::object edge_offsets = field("edge_offsets");
    if (edge_offsets.is_none())
        throw nb::type_error("mymyr: encoding: edge_offsets must be an array, not None");
    horizon.edge_offsets = rows.imports.ints32(edge_offsets, "edge_offsets", {-1}, rows.horizon_copies[1], &extent);
    if (static_cast<u64>(extent[0]) != horizon.graphs + 1)
        throw nb::value_error(("mymyr: encoding: edge_offsets must have G + 1 = " + std::to_string(horizon.graphs + 1) +
                               " entries (one more than graphs), got " + std::to_string(extent[0]))
                                  .c_str());
    const nb::object edge_src = field("edge_src");
    horizon.edge_src = rows.imports.ints32(edge_src, "edge_src", {-1}, rows.horizon_copies[2]);
    horizon.edge_dst = rows.imports.ints32(field("edge_dst"), "edge_dst", {-1}, rows.horizon_copies[3]);
    // the edge arrays share one extent E
    const i64 edges = horizon.edge_offsets[horizon.graphs];
    for (const char* name : {"edge_src", "edge_dst", "edge_schema"})
    {
        const nb::object a = field(name);
        if (!a.is_none() && nb::len(a) != static_cast<usize>(edges))
            throw nb::value_error((std::string("mymyr: encoding: ") + name + " has " + std::to_string(nb::len(a)) +
                                   " entries, edge_offsets ends at " + std::to_string(edges))
                                      .c_str());
    }
    horizon.edge_schema = rows.imports.ints32(field("edge_schema"), "edge_schema", {-1}, rows.horizon_copies[4]);
    std::vector<i64> binding_extent;
    horizon.edge_binding =
        rows.imports.ints32(field("edge_binding"), "edge_binding", {-1, -1}, rows.edge_binding_copy, &binding_extent);
    if (horizon.edge_binding)
    {
        if (binding_extent[0] != edges)
            throw nb::value_error(("mymyr: encoding: edge_binding has " + std::to_string(binding_extent[0]) +
                                   " rows, edge_offsets ends at " + std::to_string(edges))
                                      .c_str());
        horizon.label_width = static_cast<u32>(binding_extent[1]);
    }

    const i32* ids = import_task_ids(field("task_ids"), horizon.graphs, table.size(), rows.ids_copy, rows.ids_keep);
    if (horizon.graphs && table.size() > 1 && !ids)
        throw nb::value_error("mymyr: encoding: task_ids are required for a multi-instance table");
    rows.input.task_ids = ids;
    // States given as objects belong to the task of their graph (the encoder reports offsets that are not valid)
    if (!states.owner_tasks.empty())
        for (u64 g = 0; g < horizon.graphs; ++g)
        {
            const i64 first = horizon.node_offsets[g], last = horizon.node_offsets[g + 1];
            if (first < 0 || last < first || static_cast<u64>(last) > states.owner_tasks.size())
                break;
            const u32 instance = ids ? static_cast<u32>(ids[g]) : 0;
            const Task& expected = *table.task(instance);
            for (i64 node = first; node < last; ++node)
            {
                const Task& given = *states.owner_tasks[static_cast<usize>(node)];
                if (given.uid() == expected.uid() ||
                    (given.fingerprint() == expected.fingerprint() &&
                     given.encoding_fingerprint() == expected.encoding_fingerprint()))
                    continue;
                throw nb::value_error(("mymyr: encoding: node " + std::to_string(node - first) + " of graph " +
                                       std::to_string(g) + " is not a state of instance " + std::to_string(instance) +
                                       " of the table (its task id): its task differs in fingerprint or atom encoding")
                                          .c_str());
            }
        }

    encoding::EncodeInput& input = rows.input;
    u32 derived_words = 0;
    std::vector<i64> lead;
    input.derived = rows.imports.words_nd(derived_arg, "derived", {static_cast<i64>(states.view.rows)}, false, lead,
                                          derived_words);
    input.derived_stride = derived_words;
    if (!goals_arg.is_none())
        import_goals(rows.imports, goals_arg, horizon.graphs, input);
    if (!subgoals_arg.is_none())
        import_subgoals(rows.imports, subgoals_arg, horizon.graphs, input);
}

PyFlatBatch encode_horizon(PyFlatHorizonEncoder& self, Arg<ann::EncodingHorizonView> view, DerivedArg derived_arg,
                           GoalsArg goals_arg, SubgoalsArg subgoals_arg, Arg<ann::Framework> framework,
                           IntArg threads_in, PoolArg pool_in, CapacitiesArg capacities_in)
{
    const u32 threads = threads_arg(threads_in);
    const std::optional<encoding::BatchShape> capacities =
        capacities_in ? std::optional(capacities_of(*capacities_in, self.encoder->schema()))
                      : std::nullopt;
    ImportedRows rows;
    import_horizon(rows, *self.encoder, view, derived_arg, goals_arg, subgoals_arg);
    return run_encoding(self, rows, parse_framework(framework, nb::getattr(view, "nodes")), threads, pool_in, capacities);
}

constexpr std::pair<const char*, const char*> k_batch_arrays[] = {
    {"task_ids", "int32 [G]: the instance of each graph."},
    {"num_objects", "index [G]: the objects of each graph's instance (the auxiliary entity is not counted)."},
    {"entity_offsets",
     "index [G + 1]: graph g's entities are [entity_offsets[g], entity_offsets[g + 1]); entity ids are batch-global."},
    {"entity_role",
     "uint8 [E]: the role of each entity, an index into entity_role_names. Absent when the entity arrays are implicit "
     "(FlatBatch.implicit_entities)."},
    {"entity_ref",
     "index [E]: an object's instance-local id; a goal, subgoal or history target entity's tuple (its index among the "
     "graph's tuples in graph-major order); an action entity's first row among its graph's rows of the expansion given "
     "as `actions` (the expansion row is actions.offsets[g] + entity_ref); a history step's index among the valid "
     "steps of its row (step entity_ref of the row's `history`); a predicate virtual node's key (the id of its "
     "predicate's state relation); -1 for the auxiliary entity. The same in both layouts. Absent when implicit."},
    {"relation_counts",
     "index [G, R]: the tuples of each (graph, relation) cell, indexed (g, r) in both layouts. Absent with "
     "emit_counts=False."},
    {"relation_offsets",
     "index [G * R + 1]: cell (g, r) is the segment s = g * R + r (layout 'graph') or r * G + g (layout "
     "'relation') and holds the tuples [relation_offsets[s], relation_offsets[s + 1])."},
    {"relation_arg_offsets", "index [G * R + 1]: the same offsets into relation_args."},
    {"relation_args", "index [A]: the entities of the tuples, tuple after tuple, `arity` per tuple of the relation."},
    {"goal_complete",
     "uint8 [G]: 1 iff the goal relations of the graph hold its whole goal: its instance's goal when every conjunct could "
     "be expanded into literals (Task.encoding_goal_complete) and no literal whose truth a state can change was left "
     "out by the options (a nullary literal with nullary=False), always for a goal given with `goals`. Absent when the "
     "schema has no goal relations."},
    {"target_offsets",
     "index [G + 1]: graph g's targets are [target_offsets[g], target_offsets[g + 1]). Absent without target_sources."},
    {"target_entity",
     "index [T]: the entity of each target (goal, subgoal, action, then history target entities), graph after graph in "
     "entity order. Absent like target_offsets."},
    {"target_group",
     "uint8 [T]: the group of each target, an index into target_group_names (0: goal, 1: subgoal, 2: action, 3: "
     "history). Absent like target_offsets."},
    {"history_offsets",
     "index [G + 1]: graph g's history steps are [history_offsets[g], history_offsets[g + 1]). Absent without "
     "history."},
    {"history_entity",
     "index [S]: the entity of each history step, graph after graph in step order (oldest first). Absent like "
     "history_offsets."},
    {"history_dt", "int32 [S]: the dt of each history step. Absent like history_offsets."},
};

// ------------------------------------------------------------------------------------------------ streams

using IndicesArg = Arg<std::variant<ann::ArrayLike, nb::typed<nb::sequence, int>, int>>;
using Int64Array = nb::ndarray<nb::numpy, i64, nb::ndim<1>>;

Int64Array int64_array(std::vector<i64> values)
{
    auto* owner = new std::vector<i64>(std::move(values));
    nb::capsule capsule(owner, [](void* p) noexcept { delete static_cast<std::vector<i64>*>(p); });
    const usize shape[1] = {owner->size()};
    return Int64Array(owner->data(), 1, shape, capsule);
}

/// The indices of a call: an int, an integer array or a sequence of ints.
std::vector<i64> index_list(nb::handle obj)
{
    if (nb::isinstance<nb::int_>(obj))
        return {int_arg<i64>(obj, "index")};
    // a contiguous int64 array is copied as it is; any other array or sequence is read item by item
    using Array = nb::ndarray<const i64, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    Array array;
    if (nb::try_cast(obj, array, false))
        return std::vector<i64>(array.data(), array.data() + array.shape(0));
    if (nb::isinstance<nb::str>(obj) || !nb::isinstance<nb::sequence>(obj))
        throw nb::type_error("mymyr: encoding: indices must be an int, a sequence of ints or an integer array");
    std::vector<i64> out;
    for (nb::handle item : obj)
        out.push_back(int_arg<i64>(item, "index"));
    return out;
}

struct PyFlatStream
{
    nb::object encoder;  // the FlatRelationEncoder
    Framework framework = Framework::Numpy;
    std::unique_ptr<encoding::FlatStream> stream;
    [[nodiscard]] PyFlatEncoder& owner() const { return *nb::inst_ptr<PyFlatEncoder>(encoder); }
};

struct PyFlatMutableStream
{
    nb::object encoder;
    Framework framework = Framework::Numpy;
    std::unique_ptr<encoding::FlatMutableStream> stream;
    [[nodiscard]] PyFlatEncoder& owner() const { return *nb::inst_ptr<PyFlatEncoder>(encoder); }
};

/// The batch of the rows a stream holds. The encoding runs on the pool of the call, as `encode` does.
template<class Py>
PyFlatBatch flush_stream(Py& self, Arg<ann::Framework> framework, IntArg threads_in, PoolArg pool_in,
                         CapacitiesArg capacities_in)
{
    PyFlatEncoder& encoder = self.owner();
    const u32 threads = threads_arg(threads_in);
    const std::optional<encoding::BatchShape> capacities =
        capacities_in ? std::optional(capacities_of(*capacities_in, encoder.encoder->schema()))
                      : std::nullopt;
    PyFlatBatch output;
    output.framework = framework.is_none() ? self.framework : parse_framework(framework);
    output.schema = encoder.encoder->schema_ptr();
    run_on_pool(encoder, threads, pool_in,
                [&](ThreadPool& pool)
                {
                    if (capacities)
                    {
                        encoding::FlatSizes sizes;
                        output.arrays = self.stream->flush(pool, *capacities, sizes);
                        output.sizes = sizes;
                    }
                    else
                        output.arrays = self.stream->flush(pool);
                });
    return output;
}

void append_to_stream(PyFlatStream& self, StatesLike states, TaskIdsArg task_ids, DerivedArg derived, GoalsArg goals,
                      ActionsArg actions, SubgoalsArg subgoals, HistoryArg history)
{
    ImportedRows rows;
    import_rows(rows, *self.owner().encoder, states, task_ids, derived, goals, actions, subgoals, history);
    nb::gil_scoped_release release;
    self.stream->append(rows.input);
}

/// append / update of a mutable stream: `change(stream, indices, input)`.
template<class Change>
void change_mutable_stream(PyFlatMutableStream& self, IndicesArg indices_arg, StatesLike states, TaskIdsArg task_ids,
                           DerivedArg derived, GoalsArg goals, ActionsArg actions, SubgoalsArg subgoals,
                           HistoryArg history, Change change)
{
    const std::vector<i64> indices = index_list(indices_arg);
    ImportedRows rows;
    import_rows(rows, *self.owner().encoder, states, task_ids, derived, goals, actions, subgoals, history);
    try
    {
        nb::gil_scoped_release release;
        change(*self.stream, std::span<const i64>(indices), rows.input);
    }
    catch (const std::out_of_range& e)
    {
        throw nb::key_error(e.what());
    }
}

constexpr const char* k_row_inputs_doc =
    "`states`, `task_ids`, `derived`, `goals` and `actions` are as in `FlatRelationEncoder.encode`, and the rows are "
    "copied. All rows of a stream give the same inputs (a stream without rows takes any set of them with its first "
    "rows): a stream with `goals` keeps a goal for every row, a stream with `actions` the action rows of every state. "
    "With `actions`, the rows of the expansion belong to the states of this call, in order (`actions.num_states` equals "
    "the number of rows): the stream keeps their schemas and bindings, graph by graph. ";
}  // namespace

/// The members that the two encoder classes share: the catalog's properties, the table, the names of a graph's entities
/// and pickling.
template<class Py>
void define_encoder_common(nb::class_<Py>& cls)
{
    define_schema_properties(cls, [](const Py& self) -> const encoding::EncodingSchema& { return self.encoder->schema(); });
    cls.def_prop_ro("table", [](const Py& self) { return TableArg(self.reference.obj); },
                    "The TaskTable (or TaskSuite) the encoder encodes states of.")
        .def_prop_ro(
            "schema", [](const Py& self) { return PyEncodingSchema{self.encoder->schema_ptr()}; },
            "The EncodingSchema of the encoder.")
        .def(
            "entity_names",
            [](nb::handle self, nb::typed<nb::handle, PyFlatBatch> batch, IntArg graph)
            {
                return nb::typed<nb::object, std::vector<std::string>>(
                    nb::module_::import_("mymyr.encoding._names").attr("entity_names")(self, batch, graph));
            },
            "batch"_a, "graph"_a,
            "The names of the entities of graph `graph` of `batch` (a batch of this encoder's schema), in entity order: "
            "an object by its name in the task, the auxiliary entity `auxiliary`, a goal entity `[+](on a b)[g]` "
            "(`[-]` for a negative literal; a nullary literal is `[+](handempty)[g]`), a subgoal entity "
            "`[+](on a b)[sg]` (`[ssg]`, `[sssg]` for the higher levels), an action entity `(unstack a b)`, a history "
            "step `history:<dt>#<i>` (i counts the graph's history steps from 0), a history target "
            "`history:<dt>#<i>:[+](on a b)`, a state entity of a horizon graph `target:<k>` (k is the index of its node "
            "in the graph; the root is `target:0`), a predicate virtual node `predicate:on`. Entities of a graph that carry the same name (an object called "
            "`auxiliary`, a goal literal that two relations list) are told apart by position only. The names are built "
            "from the batch's arrays and the task; nothing is stored in the batch.")
        .def(
            "describe",
            [](nb::handle self, nb::typed<nb::handle, PyFlatBatch> batch, IntArg graph)
            {
                return nb::typed<nb::object, ann::EncodingGraphDescription>(
                    nb::module_::import_("mymyr.encoding._names").attr("describe")(self, batch, graph));
            },
            "batch"_a, "graph"_a,
            "A readable view of graph `graph` of `batch`: a GraphDescription with its entities (role and name, in entity "
            "order) and, for each relation that has tuples, the tuples as tuples of entity names; `str(description)` "
            "prints it.")
        .def(
            "__reduce__",
            [](Py& self)
            {
                return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_flat_encoder"),
                                      nb::make_tuple(self.reference.obj, options_dict(self.encoder->schema().options()),
                                                     self.encoder->schema().fingerprint()));
            },
            "An encoder pickles as its table and options with the schema fingerprint; loading builds the encoder again "
            "and raises ValueError (\"mymyr: encoding: ...\") if its schema's fingerprint differs.");
}

std::shared_ptr<const encoding::FlatEncoder> flat_encoder_of(nb::handle obj)
{
    return nb::cast<const PyFlatEncoder&>(obj).encoder;
}

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
        k_option_doc + k_horizon_option_doc;
    define_init<HorizonArgs::Optional>(
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
    batch.def_prop_ro("num_graphs", [](const PyFlatBatch& self) { return self.arrays->scalar("num_graphs"); },
                      "The number of graphs (states) in the batch.")
        .def_prop_ro("num_relations", [](const PyFlatBatch& self) { return self.arrays->scalar("num_relations"); },
                     "The number of relations of the schema the batch was encoded with.")
        .def_prop_ro(
            "schema", [](const PyFlatBatch& self) { return PyEncodingSchema{self.schema}; },
            "The EncodingSchema the batch was encoded for.")
        .def_prop_ro(
            "implicit_entities", [](const PyFlatBatch& self) { return self.schema->implicit_entity_arrays(); },
            "True if the batch has no entity_role and entity_ref: entity j of graph g is an object with role 0 and "
            "ref j if j < num_objects[g], else the graph's auxiliary entity (role 8, ref -1).")
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
                out["targets"] = self.sizes->targets;
                if (self.schema->has_history())
                    out["history"] = self.sizes->history;
                out["overflow"] = self.sizes->overflowed();
                out["capacity_overflow"] = (self.sizes->overflow & encoding::k_overflow_capacity) != 0;
                out["index_overflow"] = (self.sizes->overflow & encoding::k_overflow_index) != 0;
                out["invalid_input"] = false;
                return out;
            },
            "After an encoding with capacities: the true entities, tuples, args and targets of the batch (and its "
            "history steps, if the schema has history) and whether it overflowed (`overflow`: the batch is all zero), "
            "because of the capacities (`capacity_overflow`) or the index type (`index_overflow`); None otherwise.")
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
        module, "FlatRelationEncoder", nb::is_weak_referenceable(),
        "Encodes states into relational graphs: one graph per state, with the instance's objects as entities (plus one "
        "auxiliary entity when a nullary tuple exists) and, for every relation of the schema, the tuples of the state's "
        "fluent, derived and static atoms. A graph holds nothing that its state and instance do not have.");
    static const std::string encoder_doc =
        std::string("The encoder of a Task, TaskHandle or one-domain TaskTable under the options of EncodingSchema, "
                    "except the horizon options (FlatHorizonEncoder encodes horizons). ") +
        k_option_doc;
    define_init<HorizonArgs::Absent>(
        encoder_class,
        [](PyFlatEncoder* self, const TableRef& reference, const encoding::FlatOptions& options)
        { new (self) PyFlatEncoder{reference, encoding::FlatEncoder::create(reference.table, options)}; },
        encoder_doc.c_str());
    define_encoder_common(encoder_class);
    encoder_class
        .def("encode", &encode, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "derived"_a = nb::none(),
             "goals"_a = nb::none(), "actions"_a = nb::none(), "subgoals"_a = nb::none(), "history"_a = nb::none(),
             "framework"_a = nb::none(), "threads"_a = 1, "pool"_a = nb::none(),
             "capacities"_a = nb::none(),
             "Encodes the rows `states` (an array of state words [N, W] of any DLPack framework, u64 words or u32 halves "
             "[N, 2W], or a State or list of States) into one graph each. task_ids: the instance of each row (required "
             "for a table of several instances). derived: the derived atom words of each row [N, D] (default: evaluated "
             "here). goals: a goal for every row instead of its instance's goal, (goal_pos, goal_neg) or "
             "(goal_pos, goal_neg, goal_derived_pos, goal_derived_neg): masks over the fluent (derived) atom slots "
             "[N, W] ([N, D]), or [1, W] for one goal shared by all rows, as `rl.goal_masks` returns them; the "
             "goal relations then hold exactly these literals (no constant part), and goal_complete is 1. "
             "`rl.goal_masks` gives the instance's goal over fluent slots only, no derived masks, so that a "
             "call with them equals a call without `goals` only for an instance whose complete goal has no literal "
             "over a static or derived predicate or over an atom without a slot. "
             "actions: the mymyr.rl.Expansion of the same states over the same table (rl.expand(table, states, "
             "task_ids)): the action relations then hold one tuple (action entity, objects...) per distinct ground "
             "action among the expansion rows of each state, in lexicographic order of the objects, and "
             "`entity_ref` of an action entity is the index of its first row among the rows of its own state "
             "(apply that action with expansion.succ[expansion.offsets[g] + entity_ref]). Without it the "
             "action relations are empty. The expansion must be over the encoder's own table or over tasks that "
             "have, per instance, the same fingerprint and atom encoding. The rows must be in canonical order "
             "(rl.expand's default); an expansion made with witness_pruning=True gives the pruned set of actions. "
             "subgoals: the subgoal layers of each row, (pos, neg) or (pos, neg, derived_pos, derived_neg): masks "
             "[N, L, W] ([N, L, D]), or [1, L, W] for layers shared by all rows, with L at most max_goal_level; layer "
             "l fills the relations of level l + 1. A literal that the goal or a lower layer has is encoded once, at "
             "its lowest level. history: the history of each row, (pos, neg, dt), (pos, neg, dt, count) or (pos, neg, "
             "dt, count, derived_pos, derived_neg): masks [N, H, W] ([N, H, D]) of its steps oldest first, dt "
             "[N, H] integers and count [N] the valid steps of each row (a prefix; None: all H); only the newest "
             "history_max_steps of them are encoded. `mymyr.encoding.subgoal_masks` and `history_masks` build both "
             "from lists of literals. "
             "The arrays of the result are in `framework` (default: that of `states`; NumPy for States). Raises "
             "ValueError for invalid input. threads: graphs are split over this many threads (0: one per "
             "hardware thread; a batch of 1024 graphs or fewer is encoded on the calling thread); the encoder keeps "
             "one pool of that size, rebuilt when the count changes. pool: a mymyr.rl.ThreadPool to use instead (it "
             "takes precedence over threads; one batch at a time runs on it). The result does not depend on either. "
             "capacities: a dict with the keys 'graphs', 'entities', 'args' (and 'targets' with target_sources, "
             "'history' with history); the batch then has relation_args, the entity arrays, the target arrays and the "
             "history arrays of exactly the capacities' sizes (the first sizes['entities'] / sizes['args'] / "
             "sizes['targets'] / sizes['history'] entries hold the encoding, the rest is "
             "zero) and `sizes` reports the true sizes. A batch that does not fit its "
             "capacities (or the index type) is all zero and has sizes['overflow'] "
             "set; nothing raises. Rows (and task_ids, derived, goals) in CUDA memory are encoded on the device into "
             "torch tensors: a DeviceFlatBatch is returned, actions, subgoals and history are refused, threads has no effect and pool is "
             "refused.")
        .def(
            "reserve_device",
            [](nb::handle self, IntArg graphs, Arg<ann::Any> device, StreamArg stream)
            {
                nb::module_::import_("mymyr.encoding._device").attr("reserve")(self, graphs, device, stream);
            },
            "graphs"_a, nb::kw_only(), "device"_a = nb::none(), "stream"_a = nb::none(),
            "Makes the device workspace of `stream` (a torch.cuda.Stream or a cudaStream_t; default: torch's current "
            "stream of `device`, default the current CUDA device) hold batches of up to `graphs` rows, uploading the "
            "encoder's context to the device if needed. Exact encodes reserve on demand. Capacity encodes use "
            "torch-owned scratch and need no reservation, including during capture. "
            "Raises ValueError for options the device does not encode.")
        .def(
            "release_device",
            [](nb::handle self, Arg<ann::Any> device, StreamArg stream)
            {
                nb::module_::import_("mymyr.encoding._device").attr("release")(self, device, stream);
            },
            nb::kw_only(), "device"_a = nb::none(), "stream"_a = nb::none(),
            "Frees the device workspace of `stream`, or of every stream of `device` (default: the current CUDA device) "
            "without one. The uploaded context stays with the encoder. Captured capacity scratch belongs to its graph.")
        .def(
            "stream",
            [](PyFlatEncoder& self, Arg<ann::Framework> framework)
            {
                return PyFlatStream{nb::borrow(nb::find(&self)), parse_framework(framework),
                                    std::make_unique<encoding::FlatStream>(self.encoder)};
            },
            nb::kw_only(), "framework"_a = nb::none(),
            "An append-only stream of rows for this encoder (FlatStream). framework: the framework of the batches its "
            "flushes return (default NumPy).")
        .def(
            "mutable_stream",
            [](PyFlatEncoder& self, Arg<ann::Framework> framework)
            {
                return PyFlatMutableStream{nb::borrow(nb::find(&self)), parse_framework(framework),
                                           std::make_unique<encoding::FlatMutableStream>(self.encoder)};
            },
            nb::kw_only(), "framework"_a = nb::none(),
            "A stream of rows under indices you choose, whose rows can be replaced and removed (FlatMutableStream). "
            "framework: the framework of the batches its flushes return (default NumPy).")
    ;

    nb::class_<PyFlatHorizonEncoder> horizon_class(
        module, "FlatHorizonEncoder",
        "Encodes horizons into relational graphs: one graph per root state, holding the root and the nodes below it (a "
        "node is a state reached from the root by a path of actions) as state entities, next to the instance's objects "
        "(and the auxiliary entity when a nullary tuple exists). The relations hold the facts of the nodes (anchored by "
        "their state entity), the goal literals of the root, the goal's satisfaction at the nodes, what a node adds to or "
        "deletes from its root (the delta mode) and the edges of the horizon: one action tuple (the action relations, "
        "with the action's objects) and, with parent_relation, one `_parent_` tuple per edge. The nodes and edges are "
        "given as a HorizonView (mymyr.encoding.HorizonView; `rl.expand` rows have this form). Streams and the device "
        "do not encode horizons.");
    static const std::string horizon_doc =
        std::string("The horizon encoder of a Task, TaskHandle or one-domain TaskTable. horizon_mode ('full', 'delta' or "
                    "'action') is required; `history` is not an option of a horizon. ") +
        k_option_doc + k_horizon_option_doc;
    define_init<HorizonArgs::Required>(
        horizon_class,
        [](PyFlatHorizonEncoder* self, const TableRef& reference, const encoding::FlatOptions& options)
        { new (self) PyFlatHorizonEncoder{reference, encoding::FlatEncoder::create(reference.table, options)}; },
        horizon_doc.c_str());
    define_encoder_common(horizon_class);
    horizon_class.def(
        "encode", &encode_horizon, "view"_a, nb::kw_only(), "derived"_a = nb::none(), "goals"_a = nb::none(),
        "subgoals"_a = nb::none(), "framework"_a = nb::none(), "threads"_a = 1, "pool"_a = nb::none(),
        "capacities"_a = nb::none(),
        "Encodes the horizons of `view` (a HorizonView: its attributes `nodes`, `node_offsets`, `edge_offsets`, "
        "`edge_src`, `edge_dst`, `edge_schema`, `edge_binding` and `task_ids`) into one graph each. nodes: the node rows "
        "[M, W] (state words of any DLPack framework, u64 words or u32 halves [M, 2W], or a list of States); graph g's "
        "nodes are the rows node_offsets[g] to node_offsets[g + 1] (node_offsets [G + 1] int32 from 0, strictly "
        "ascending), its first row is the root and node k of the graph is row node_offsets[g] + k. The edges of graph "
        "g are the entries edge_offsets[g] to edge_offsets[g + 1] of edge_src and edge_dst [E] (graph-local node "
        "indices: the edge leads from node edge_src to node edge_dst), edge_schema [E] (the action schema) and "
        "edge_binding [E, L] (the action's objects, L at least the table's label width): they give the action and "
        "parent tuples one per edge, in the order of the edges, without sorting or removing duplicates; the arrays a "
        "schema does not read may be None. task_ids: the instance of each graph [G] (required for a table of several "
        "instances). derived: the derived atom words of each node row [M, D] (default: evaluated here). goals, "
        "subgoals: as in FlatRelationEncoder.encode, but per graph [G, ...] (or one row shared by all graphs). "
        "framework, threads, pool and capacities are as in FlatRelationEncoder.encode (capacities' 'graphs' bounds the horizons, not the node rows, and it needs the key "
        "'targets': the candidate nodes). A state entity's `entity_ref` is the node's index in its graph, so the row of "
        "the node in `nodes` is node_offsets[g] + entity_ref. Raises ValueError (\"mymyr: encoding: ...\", naming the "
        "graph) for invalid input.");

    nb::class_<PyFlatStream> stream_class(
        module, "FlatStream",
        "An append-only stream of rows (`FlatRelationEncoder.stream()`). `append` copies rows with their per-row "
        "inputs; `flush` encodes everything appended since the last flush in one call, returns the FlatBatch "
        "`FlatRelationEncoder.encode` returns for those rows in the order of the appends, and empties the stream. "
        "Calls are safe from several threads (they take turns).");
    stream_class
        .def("append", &append_to_stream, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "derived"_a = nb::none(),
             "goals"_a = nb::none(), "actions"_a = nb::none(), "subgoals"_a = nb::none(), "history"_a = nb::none(),
             (std::string("Appends the rows `states`. ") + k_row_inputs_doc +
              "A call that raises appends nothing. Errors that only the encoder sees (a state bit the instance has no "
              "atom for, actions out of canonical order, labels of another instance) raise at the flush, which then keeps "
              "the rows: `clear()` drops them.")
                 .c_str())
        .def(
            "flush", &flush_stream<PyFlatStream>, nb::kw_only(), "framework"_a = nb::none(), "threads"_a = 1,
            "pool"_a = nb::none(), "capacities"_a = nb::none(),
            "Encodes the rows held, in the order they were appended, and empties the stream: the result equals "
            "`encoder.encode(rows, ...)` over the same rows bit for bit. framework, threads, pool and capacities are "
            "as in `encode`; `framework` defaults to the stream's. The graph g of the batch is the g-th appended row. "
            "With `actions`, the action rows are those appended for each state: the expansion row of an action "
            "entity of the state that was row k of an append with expansion x is `x.offsets[k] + entity_ref`. If the "
            "encoding raises (or, with `capacities`, overflows: `sizes['overflow']`), the rows are kept, so that the "
            "flush can be repeated with larger capacities. A stream without rows flushes to a batch of no graphs.")
        .def(
            "clear", [](PyFlatStream& self) { self.stream->clear(); }, "Drops every row.")
        .def("__len__", [](const PyFlatStream& self) { return self.stream->size(); }, "The rows held.")
        .def_prop_ro("encoder", [](const PyFlatStream& self) { return nb::typed<nb::object, PyFlatEncoder>(self.encoder); },
                     "The FlatRelationEncoder of the stream.");

    nb::class_<PyFlatMutableStream> mutable_class(
        module, "FlatMutableStream",
        "A stream of rows under indices you choose (`FlatRelationEncoder.mutable_stream()`): `append` stores rows "
        "under new indices, `update` replaces the rows under held indices, `remove` drops them. `flush` encodes "
        "the rows held in ascending order of their indices, so the graph j of the batch is the row of `indices()[j]`, "
        "and keeps the rows (`clear()` drops them). The batch equals `FlatRelationEncoder.encode` over the same rows "
        "in that order, bit for bit. Calls are safe from several threads (they take turns).");
    mutable_class
        .def(
            "append",
            [](PyFlatMutableStream& self, IndicesArg indices, StatesLike states, TaskIdsArg task_ids, DerivedArg derived,
               GoalsArg goals, ActionsArg actions, SubgoalsArg subgoals, HistoryArg history)
            {
                change_mutable_stream(self, indices, states, task_ids, derived, goals, actions, subgoals, history,
                                      [](encoding::FlatMutableStream& stream, std::span<const i64> at,
                                         const encoding::EncodeInput& in) { stream.append(at, in); });
            },
            "indices"_a, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "derived"_a = nb::none(),
            "goals"_a = nb::none(), "actions"_a = nb::none(), "subgoals"_a = nb::none(), "history"_a = nb::none(),
            (std::string("Stores row k of `states` under `indices[k]` (an int, an integer array or a sequence of ints; "
                         "any 64-bit integers, negative ones included). ") +
             k_row_inputs_doc +
             "Raises ValueError for an index that is held already or repeats; a call that raises changes nothing.")
                .c_str())
        .def(
            "update",
            [](PyFlatMutableStream& self, IndicesArg indices, StatesLike states, TaskIdsArg task_ids, DerivedArg derived,
               GoalsArg goals, ActionsArg actions, SubgoalsArg subgoals, HistoryArg history)
            {
                change_mutable_stream(self, indices, states, task_ids, derived, goals, actions, subgoals, history,
                                      [](encoding::FlatMutableStream& stream, std::span<const i64> at,
                                         const encoding::EncodeInput& in) { stream.update(at, in); });
            },
            "indices"_a, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "derived"_a = nb::none(),
            "goals"_a = nb::none(), "actions"_a = nb::none(), "subgoals"_a = nb::none(), "history"_a = nb::none(),
            (std::string("Replaces the rows under `indices` with the rows `states`, with all their inputs. ") +
             k_row_inputs_doc +
             "Raises KeyError for an index that is not held and ValueError for one that repeats; a call that raises "
             "changes nothing.")
                .c_str())
        .def(
            "remove",
            [](PyFlatMutableStream& self, IndicesArg indices_arg)
            {
                const std::vector<i64> indices = index_list(indices_arg);
                try
                {
                    nb::gil_scoped_release release;
                    self.stream->remove(indices);
                }
                catch (const std::out_of_range& e)
                {
                    throw nb::key_error(e.what());
                }
            },
            "indices"_a,
            "Removes the rows under `indices`. Raises KeyError for an index that is not held and ValueError for one that "
            "repeats; a call that raises changes nothing.")
        .def(
            "flush", &flush_stream<PyFlatMutableStream>, nb::kw_only(), "framework"_a = nb::none(), "threads"_a = 1,
            "pool"_a = nb::none(), "capacities"_a = nb::none(),
            "Encodes the rows held in ascending order of their indices and keeps them: the graph j of the batch is the "
            "row of `indices()[j]`. framework, threads, pool and capacities are as in `encode`; `framework` defaults "
            "to the stream's. With `actions`, the action rows are those given for each state: the expansion row of an "
            "action entity of the state that was row k of an append or update with expansion x is "
            "`x.offsets[k] + entity_ref`. A stream without rows flushes to a batch of no graphs.")
        .def(
            "clear", [](PyFlatMutableStream& self) { self.stream->clear(); }, "Drops every row.")
        .def(
            "indices", [](PyFlatMutableStream& self) { return int64_array(self.stream->indices()); },
            "The indices of the rows held, ascending: the order of the graphs of a flush.")
        .def(
            "__contains__", [](const PyFlatMutableStream& self, Arg<ann::Any> index)
            {
                // a value that is not an integer, or not a 64-bit one, is no index of a row
                i64 value = 0;
                if (!PyIndex_Check(index.ptr()))
                    return false;
                const nb::object integer = nb::steal(PyNumber_Index(index.ptr()));
                if (!integer.is_valid())
                {
                    PyErr_Clear();
                    return false;
                }
                return nb::try_cast<i64>(integer, value) && self.stream->contains(value);
            },
            "index"_a.none(), "Whether a row is held under `index`; False for a value that is not a 64-bit integer.")
        .def("__len__", [](const PyFlatMutableStream& self) { return self.stream->size(); }, "The rows held.")
        .def_prop_ro(
            "encoder", [](const PyFlatMutableStream& self) { return nb::typed<nb::object, PyFlatEncoder>(self.encoder); },
            "The FlatRelationEncoder of the stream.");
}
}
