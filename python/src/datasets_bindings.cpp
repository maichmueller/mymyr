// mymyr._core._datasets: state spaces, generalized state spaces, knowledge bases, tuple graphs, samplers, object graphs
// and certificates (mymyr.datasets; datasets/*.hpp in the C++ core).
//
// Every generation releases the thread state while it runs. The results are immutable C++ objects shared by the
// Python wrappers; their arrays are exported zero-copy (read-only views that keep the result alive) to NumPy, torch or
// JAX (`framework`, as in Task.device_arrays) through export_array (arrays.hpp).
// A sampler holds its random state behind a mutex: one sampler may be shared by threads (free-threaded CPython), but
// a sampler per thread avoids the contention. Samplers also take the device state spaces of mymyr.cuda (CUDA builds:
// their unit goal distances are downloaded, py_datasets.hpp).

#include "arrays.hpp"
#include "py_datasets.hpp"
#include "py_formula.hpp"
#include "py_table.hpp"
#include "py_task.hpp"

#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/knowledge_base.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#if defined(MYMYR_HAS_CUDA)
namespace mymyr::python::ann
{
/// A mymyr.cuda.DeviceStateSpace (bound in cuda_datasets_bindings.cpp).
struct DeviceStateSpace
{
};
}  // namespace mymyr::python::ann

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::DeviceStateSpace>
{
    static constexpr auto Name = const_name("mymyr._core._cuda.DeviceStateSpace");
};
}  // namespace nanobind::detail
#endif

namespace mymyr::python
{
namespace
{
DeviceDistancesHook g_device_distances = nullptr;
}  // namespace

void set_device_distances_hook(DeviceDistancesHook hook) noexcept { g_device_distances = hook; }

namespace
{
using namespace nb::literals;
using namespace mymyr::datasets;

using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using TasksArg = Arg<nb::typed<nb::sequence, std::variant<PyTask, PyHandle>>>;
using StateArg = Arg<PyState>;
using IntArg = Arg<u64>;
using FloatArg = Arg<double>;
using StrArg = Arg<std::string>;
using AnyArray = Arg<ann::Any>;
using ArrayDict = nb::typed<nb::dict, std::string, ann::Any>;
/// The space of a sampler: a StateSpace, or (CUDA builds) a mymyr.cuda.DeviceStateSpace.
#if defined(MYMYR_HAS_CUDA)
using SamplerSpace = Arg<std::variant<PyStateSpace, ann::DeviceStateSpace>>;
#else
using SamplerSpace = Arg<PyStateSpace>;
#endif

// ------------------------------------------------------------------------------------------------ wrappers
struct PyGenerationResult
{
    StateSpaceStatus status = StateSpaceStatus::Ok;
    std::optional<PyStateSpace> space;
    u64 states = 0;
    f64 seconds = 0;
};

struct PyGeneralizedStateSpace
{
    GeneralizedStateSpacePtr gss;
    std::vector<PyStateSpace> spaces;  // in gss->spaces() order
};

/// A sampler and the lock of its random state; shared with the views of its id lists.
struct SamplerCore
{
    SamplerCore(StateSpacePtr space, u64 seed) : sampler(std::move(space), seed) {}
    SamplerCore(std::span<const i32> unit_goal_distances, u64 seed) : sampler(unit_goal_distances, seed) {}
    std::mutex mutex;
    StateSpaceSampler sampler;
};

struct PySampler
{
    nb::object space;  // the StateSpace or DeviceStateSpace sampled
    std::shared_ptr<SamplerCore> core;
};

struct PyObjectGraph
{
    std::shared_ptr<const ObjectGraph> graph;
};

struct PyObjectGraphBuilder
{
    explicit PyObjectGraphBuilder(const Owner& o) : owner(o), builder(*o.core->task) {}
    Owner owner;
    std::mutex mutex;  // the builder holds scratch
    ObjectGraphBuilder builder;
};

/// A tuple graph and the state space it is of (shared: a graph of a knowledge base aliases the knowledge base).
struct PyTupleGraph
{
    std::shared_ptr<const TupleGraph> graph;
    PyStateSpace space;
};

/// A knowledge base, its mymyr.rl.TaskTable, its state spaces (owned by the table's Task objects) and the arguments
/// it was built with (its pickled state).
struct PyKnowledgeBase
{
    KnowledgeBasePtr kb;
    nb::object table;
    std::vector<PyStateSpace> spaces;
    std::optional<PyGeneralizedStateSpace> gss;
    nb::tuple args;
};

// ------------------------------------------------------------------------------------------------ helpers
std::string lower(std::string s)
{
    for (char& c : s)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return s;
}

std::string str_arg(nb::handle h, const char* what)
{
    if (!nb::isinstance<nb::str>(h))
        throw nb::type_error((std::string("mymyr: ") + what + " must be a string").c_str());
    return lower(nb::cast<std::string>(h));
}

template<class T>
std::optional<T> opt(nb::handle h)
{
    if (h.is_none())
        return std::nullopt;
    return nb::cast<T>(h);
}

CertificateKind parse_certificate(nb::handle h)
{
    const std::string s = str_arg(h, "certificate");
    if (s == "color_refinement" || s == "colour_refinement" || s == "1-wl" || s == "wl")
        return CertificateKind::ColorRefinement;
    if (s == "kfwl" || s == "k-fwl" || s == "fwl")
        return CertificateKind::KFwl;
    throw nb::value_error("mymyr: certificate must be 'color_refinement' or 'kfwl'");
}

StateSpaceOptions make_options(u32 threads, nb::handle max_states, nb::handle max_seconds, bool remove_if_unsolvable,
                               bool symmetry_pruning, nb::handle certificate, u32 k, bool labels)
{
    StateSpaceOptions o;
    o.threads = threads;
    if (auto v = opt<u64>(max_states))
        o.max_states = *v;
    if (auto v = opt<double>(max_seconds))
    {
        if (!(*v >= 0))
            throw nb::value_error("mymyr: max_seconds must be non-negative");
        o.max_seconds = *v;
    }
    o.remove_if_unsolvable = remove_if_unsolvable;
    o.symmetry_pruning = symmetry_pruning;
    o.certificate = parse_certificate(certificate);
    o.fwl_k = k;
    if (o.symmetry_pruning && o.certificate == CertificateKind::KFwl && (k < 2 || k > 4))
        throw nb::value_error("mymyr: k-FWL certificates support k = 2, 3 and 4");
    o.labels = labels;
    return o;
}

template<class T>
constexpr rl::DType dtype_of()
{
    if constexpr (std::is_same_v<T, u8>)
        return rl::DType::Bool;
    else if constexpr (std::is_same_v<T, i32>)
        return rl::DType::I32;
    else if constexpr (std::is_same_v<T, u32>)
        return rl::DType::U32;
    else if constexpr (std::is_same_v<T, f64>)
        return rl::DType::F64;
    else
        return rl::DType::U64;
}

/// A read-only view of `data` (shape: default 1-d), kept alive by `owner`.
template<class T>
nb::object view(std::shared_ptr<const void> owner, std::span<const T> data, Framework fw, std::vector<i64> shape = {},
                bool words = false)
{
    static const T empty{};
    ArraySpec spec;
    spec.owner = std::move(owner);
    spec.data = data.empty() ? &empty : data.data();
    spec.dtype = dtype_of<T>();
    spec.shape = shape.empty() ? std::vector<i64>{static_cast<i64>(data.size())} : std::move(shape);
    spec.readonly = true;
    spec.words = words;
    return export_array(std::move(spec), fw, default_words(fw));
}

/// A fresh u32 vector (sampler output).
nb::object fresh_u32(const std::vector<u32>& v, Framework fw)
{
    auto block = Block::make(std::max<u64>(v.size(), 1) * sizeof(u32));
    if (!v.empty())
        std::memcpy(block->data(), v.data(), v.size() * sizeof(u32));
    ArraySpec spec{block, block->data(), rl::DType::U32, {static_cast<i64>(v.size())}, {}, false, false};
    return export_array(std::move(spec), fw, default_words(fw));
}

nb::int_ certificate_int(const Certificate& c)
{
    unsigned char b[16];
    for (int i = 0; i < 8; ++i)
    {
        b[i] = static_cast<unsigned char>(c.lo >> (8 * i));
        b[8 + i] = static_cast<unsigned char>(c.hi >> (8 * i));
    }
    PyObject* r = PyLong_FromUnsignedNativeBytes(b, 16, Py_ASNATIVEBYTES_LITTLE_ENDIAN | Py_ASNATIVEBYTES_UNSIGNED_BUFFER);
    if (!r)
        throw nb::python_error();
    return nb::steal<nb::int_>(r);
}

StateSpaceResult run_generation(const TaskPtr& task, const StateSpaceOptions& o)
{
    nb::gil_scoped_release release;
    return generate_state_space(task, o);
}

PyGenerationResult wrap(const Owner& owner, StateSpaceResult&& r)
{
    PyGenerationResult g;
    g.status = r.status;
    g.states = r.states;
    g.seconds = r.seconds;
    if (r.space)
        g.space = PyStateSpace{std::move(r.space), owner};
    return g;
}

std::vector<Owner> owners_of(nb::handle tasks)
{
    if (!nb::isinstance<nb::sequence>(tasks) || nb::isinstance<nb::str>(tasks))
        throw nb::type_error("mymyr: tasks must be a sequence of Tasks or TaskHandles");
    std::vector<Owner> out;
    for (nb::handle t : nb::borrow<nb::sequence>(tasks))
    {
        if (!nb::isinstance<PyTask>(t) && !nb::isinstance<PyHandle>(t))
            throw nb::type_error("mymyr: tasks must be a sequence of Tasks or TaskHandles");
        out.push_back(owner_of(t));
    }
    return out;
}

Owner task_owner(nb::handle task)
{
    if (!nb::isinstance<PyTask>(task) && !nb::isinstance<PyHandle>(task))
        throw nb::type_error("mymyr: task must be a Task or TaskHandle");
    return owner_of(task);
}

u32 state_id(const PyStateSpace& S, i64 id)
{
    if (id < 0 || id >= static_cast<i64>(S.space->num_states()))
        throw nb::index_error("mymyr: state id out of range");
    return static_cast<u32>(id);
}

u64 edge_id(const PyStateSpace& S, i64 e)
{
    if (e < 0 || static_cast<u64>(e) >= S.space->num_transitions())
        throw nb::index_error("mymyr: transition index out of range");
    return static_cast<u64>(e);
}

Arg<PyAction> label_of(const PyStateSpace& S, u64 e)
{
    if (!S.space->has_labels())
        throw nb::value_error("mymyr: this state space was generated without labels (labels=False)");
    const Action a = S.space->label(e);
    std::vector<i32> b(a.binding.size());
    for (usize i = 0; i < b.size(); ++i)
        b[i] = static_cast<i32>(a.binding[i].v);
    return make_label(S.owner, a.schema.v, b.data(), static_cast<u32>(b.size()));
}

ArrayDict space_arrays(const PyStateSpace& P, Framework fw)
{
    const StateSpace& S = *P.space;
    const std::shared_ptr<const void> own = P.space;
    ArrayDict d{nb::dict()};
    const i64 N = S.num_states(), E = static_cast<i64>(S.num_transitions());
    d["state_words"] = view<u64>(own, S.state_words(), fw, {N, static_cast<i64>(S.row_words())}, true);
    d["forward_offsets"] = view<u64>(own, S.forward_offsets(), fw);
    d["forward_targets"] = view<u32>(own, S.forward_targets(), fw);
    if (S.has_labels())
    {
        d["label_schemas"] = view<u32>(own, S.label_schemas(), fw);
        d["label_bindings"] = view<u32>(own, S.label_bindings(), fw, {E, static_cast<i64>(S.label_width())});
    }
    d["backward_offsets"] = view<u64>(own, S.backward_offsets(), fw);
    d["backward_sources"] = view<u32>(own, S.backward_sources(), fw);
    d["backward_edges"] = view<u32>(own, S.backward_edges(), fw);
    d["unit_goal_distances"] = view<i32>(own, S.unit_goal_distances(), fw);
    d["cost_goal_distances"] = view<f64>(own, S.cost_goal_distances(), fw);
    if (!S.unit_costs())
        d["costs"] = view<f64>(own, S.costs(), fw);
    d["goal"] = view<u8>(own, S.goal_flags(), fw);
    d["unsolvable"] = view<u8>(own, S.unsolvable_flags(), fw);
    d["alive"] = view<u8>(own, S.alive_flags(), fw);
    return d;
}

u32 tuple_vertex(const TupleGraph& g, i64 v)
{
    if (v < 0 || v >= static_cast<i64>(g.num_vertices()))
        throw nb::index_error("mymyr: tuple graph vertex out of range");
    return static_cast<u32>(v);
}

u32 tuple_distance(const TupleGraph& g, i64 d)
{
    if (d < 0 || d >= static_cast<i64>(g.num_distances()))
        throw nb::index_error("mymyr: tuple graph distance out of range");
    return static_cast<u32>(d);
}

std::vector<u32> to_vector(std::span<const u32> s) { return {s.begin(), s.end()}; }

/// The keyword arguments of KnowledgeBase(), in order (its pickled state is the table, then these).
#define MYMYR_KB_ARGS                                                                                                      \
    nb::kw_only(), "threads"_a = 0, "max_states"_a = nb::none(), "max_seconds"_a = nb::none(),                           \
        "remove_if_unsolvable"_a = true, "symmetry_pruning"_a = false, "certificate"_a = "kfwl", "k"_a = 2,             \
        "labels"_a = true, "sort_by_size"_a = true, "generalized"_a = false, "width"_a = nb::none(),                    \
        "dominance_pruning"_a = true

using KbTasksArg = Arg<std::variant<PyTable, nb::typed<nb::sequence, std::variant<PyTask, PyHandle>>>>;
using KbState = nb::typed<nb::tuple, PyTable, u32, std::optional<u64>, std::optional<double>, bool, bool, std::string, u32,
                          bool, bool, bool, std::optional<u32>, bool>;

void init_knowledge_base(PyKnowledgeBase* self, nb::handle tasks, u32 threads, nb::handle max_states, nb::handle max_seconds,
                         bool remove_if_unsolvable, bool symmetry_pruning, nb::handle certificate, u32 k, bool labels,
                         bool sort_by_size, bool generalized, std::optional<u32> width, bool dominance_pruning)
{
    nb::object table = nb::isinstance<PyTable>(tasks) ? nb::borrow(tasks) : nb::type<PyTable>()(tasks);
    KnowledgeBaseOptions o;
    o.state_space = make_options(1, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate, k, labels);
    o.sort_by_size = sort_by_size;
    o.generalized = generalized;
    if (width)
        o.tuple_graphs = TupleGraphOptions{.width = *width, .dominance_pruning = dominance_pruning, .threads = threads};
    o.threads = threads;
    const PyTable& t = *nb::inst_ptr<PyTable>(table);
    KnowledgeBasePtr kb;
    {
        nb::gil_scoped_release release;  // std::invalid_argument (a width above 5) becomes a ValueError
        kb = KnowledgeBase::create(t.table, o);
    }
    auto* p = new (self) PyKnowledgeBase{};
    p->kb = kb;
    p->table = table;
    for (usize i = 0; i < kb->state_spaces().size(); ++i)
        p->spaces.push_back(PyStateSpace{kb->state_spaces()[i], owner_of(t.tasks[kb->task_indices()[i]])});
    if (kb->generalized_state_space())
        p->gss = PyGeneralizedStateSpace{kb->generalized_state_space(), p->spaces};
    p->args = nb::make_tuple(table, threads, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate, k,
                             labels, sort_by_size, generalized, width, dominance_pruning);
}

usize kb_space(const PyKnowledgeBase& kb, i64 i)
{
    if (i < 0 || i >= static_cast<i64>(kb.spaces.size()))
        throw nb::index_error("mymyr: state space index out of range");
    return static_cast<usize>(i);
}

void require_tuple_graphs(const PyKnowledgeBase& kb)
{
    if (!kb.kb->has_tuple_graphs())
        throw nb::value_error("mymyr: this knowledge base has no tuple graphs (width=None)");
}

PyTupleGraph kb_tuple_graph(const PyKnowledgeBase& kb, usize i, u32 v)
{
    return PyTupleGraph{std::shared_ptr<const TupleGraph>(kb.kb, &kb.kb->tuple_graphs()[i][v]), kb.spaces[i]};
}

const char* k_space_doc =
    "The full transition model of a task as flat arrays (datasets/state_space.hpp), with the semantics of mimir's "
    "StateSpace: every reachable state (ids in breadth-first discovery order, the same at every thread count), every "
    "applicable action as a transition (parallel edges and self-loops included), forward and reverse CSR, labels, "
    "transition costs, goal distances (V*: unit by BFS, cost by Dijkstra) and goal / unsolvable / alive flags. "
    "arrays() exports them zero-copy. The initial state is id 0.";

const char* k_options_doc =
    "Options: threads (0: all cores; 1: sequential), max_states (fail when the space has max(max_states, 2) states "
    "or more, as in mimir), max_seconds, remove_if_unsolvable (no space when the initial state cannot reach a goal), "
    "symmetry_pruning (one state per certificate class of its object graph; single-threaded), certificate "
    "('kfwl', the default, or the cheaper but weaker 'color_refinement') and k (2, 3 or 4) for symmetry pruning, labels (keep (schema, binding) per "
    "transition). State words are independent of the thread count when the task has frozen atoms (atoms='frozen').";

#define MYMYR_SS_ARGS                                                                                                      \
    nb::kw_only(), "threads"_a = 1, "max_states"_a = nb::none(), "max_seconds"_a = nb::none(),                           \
        "remove_if_unsolvable"_a = true, "symmetry_pruning"_a = false, "certificate"_a = "kfwl", "k"_a = 2, \
        "labels"_a = true
}  // namespace

void bind_datasets(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule(
        "_datasets", "State spaces, knowledge bases, tuple graphs, samplers, object graphs and certificates (mymyr.datasets)");

    nb::enum_<StateSpaceStatus>(m, "Status", "Outcome of a state space generation.")
        .value("OK", StateSpaceStatus::Ok)
        .value("OUT_OF_STATES", StateSpaceStatus::OutOfStates)
        .value("TIMEOUT", StateSpaceStatus::Timeout)
        .value("UNSOLVABLE", StateSpaceStatus::Unsolvable)
        .def("__str__", [](StateSpaceStatus s) { return std::string(to_string(s)); });

    // ---------------------------------------------------------------------------------------------- StateSpace
    nb::class_<PyStateSpace>(m, "StateSpace", k_space_doc)
        .def_prop_ro("task", [](const PyStateSpace& s) { return task_object(s.owner.obj); })
        .def_prop_ro("num_states", [](const PyStateSpace& s) { return s.space->num_states(); })
        .def_prop_ro("num_transitions", [](const PyStateSpace& s) { return s.space->num_transitions(); })
        .def_prop_ro("initial_state_id", [](const PyStateSpace& s) { return s.space->initial_state(); })
        .def_prop_ro("symmetry_reduced", [](const PyStateSpace& s) { return s.space->symmetry_reduced(); })
        .def_prop_ro("words", [](const PyStateSpace& s) { return s.space->words(); }, "Fluent words per state row.")
        .def_prop_ro("numeric_words", [](const PyStateSpace& s) { return s.space->numeric_words(); })
        .def_prop_ro("row_words", [](const PyStateSpace& s) { return s.space->row_words(); })
        .def_prop_ro("has_labels", [](const PyStateSpace& s) { return s.space->has_labels(); })
        .def_prop_ro("label_width", [](const PyStateSpace& s) { return s.space->label_width(); })
        .def_prop_ro("unit_costs", [](const PyStateSpace& s) { return s.space->unit_costs(); },
                     "True if every transition costs 1 (arrays() then has no 'costs').")
        .def_prop_ro("num_goal_states", [](const PyStateSpace& s) { return s.space->num_goal_states(); })
        .def_prop_ro("num_unsolvable_states", [](const PyStateSpace& s) { return s.space->num_unsolvable_states(); })
        .def_prop_ro("max_goal_distance", [](const PyStateSpace& s) { return s.space->max_goal_distance(); },
                     "The largest finite unit goal distance (-1 if no state reaches a goal).")
        .def_prop_ro("threads", [](const PyStateSpace& s) { return s.space->threads(); })
        .def_prop_ro("layers", [](const PyStateSpace& s) { return s.space->layers(); })
        .def_prop_ro("search_seconds", [](const PyStateSpace& s) { return s.space->search_seconds(); })
        .def_prop_ro("post_seconds", [](const PyStateSpace& s) { return s.space->post_seconds(); })
        .def_prop_ro("nbytes", [](const PyStateSpace& s) { return s.space->bytes(); })
        .def(
            "arrays", [](const PyStateSpace& s, FrameworkArg framework) { return space_arrays(s, parse_framework(framework)); },
            "framework"_a = nb::none(),
            "Zero-copy read-only views: state_words [N, W + NN] (the task's word encoding), forward_offsets [N + 1], "
            "forward_targets [E], label_schemas [E] and label_bindings [E, label_width] (unused positions 2^32 - 1; "
            "with labels), backward_offsets [N + 1], backward_sources and backward_edges [E] (per target, ascending "
            "forward edge indices), unit_goal_distances [N] (int32, -1: unsolvable), cost_goal_distances [N] (float64, "
            "inf: unsolvable), costs [E] (float64; absent for unit costs), goal, unsolvable, alive [N] (bool).")
        .def(
            "state",
            [](const PyStateSpace& s, i64 id) { return make_state(s.owner, State(s.space->state(state_id(s, id)))); },
            "id"_a, "The state with this id (a mymyr.State of the space's task).")
        .def(
            "states",
            [](const PyStateSpace& s) {
                nb::typed<nb::list, PyState> out{nb::list()};
                for (u32 i = 0; i < s.space->num_states(); ++i)
                    out.append(make_state(s.owner, State(s.space->state(i))));
                return out;
            },
            "Every state, in id order.")
        .def(
            "find",
            [](const PyStateSpace& s, StateArg state) -> i64 {
                if (!is_state(state))
                    throw nb::type_error("mymyr: state must be a mymyr.State");
                const PyState& p = state_of(state);
                if (p.core->task->uid() != s.owner.core->task->uid())
                    throw nb::value_error("mymyr: the state belongs to another task");
                nb::gil_scoped_release release;  // the first call builds the index
                return s.space->find(p.s.view());
            },
            "state"_a, "The id of a state of this space, or -1.")
        .def(
            "transitions",
            [](const PyStateSpace& s, i64 id) {
                const u32 v = state_id(s, id);
                nb::typed<nb::list, nb::typed<nb::tuple, PyAction, int>> out{nb::list()};
                const auto off = s.space->forward_offsets();
                for (u64 e = off[v]; e < off[v + 1]; ++e)
                    out.append(nb::make_tuple(label_of(s, e), s.space->forward_targets()[e]));
                return out;
            },
            "id"_a, "The transitions out of a state: (action, target id), in canonical action order.")
        .def("label", [](const PyStateSpace& s, i64 e) { return label_of(s, edge_id(s, e)); }, "edge"_a,
             "The action of a transition.")
        .def("source", [](const PyStateSpace& s, i64 e) { return s.space->source(edge_id(s, e)); }, "edge"_a)
        .def("target", [](const PyStateSpace& s, i64 e) { return s.space->forward_targets()[edge_id(s, e)]; }, "edge"_a)
        .def("cost", [](const PyStateSpace& s, i64 e) { return s.space->cost(edge_id(s, e)); }, "edge"_a)
        .def("is_goal", [](const PyStateSpace& s, i64 id) { return s.space->is_goal(state_id(s, id)); }, "id"_a)
        .def("is_unsolvable", [](const PyStateSpace& s, i64 id) { return s.space->is_unsolvable(state_id(s, id)); }, "id"_a)
        .def("is_alive", [](const PyStateSpace& s, i64 id) { return s.space->is_alive(state_id(s, id)); }, "id"_a)
        .def(
            "unit_goal_distance", [](const PyStateSpace& s, i64 id) { return s.space->unit_goal_distances()[state_id(s, id)]; },
            "id"_a, "-1 if no goal is reachable.")
        .def(
            "cost_goal_distance", [](const PyStateSpace& s, i64 id) { return s.space->cost_goal_distances()[state_id(s, id)]; },
            "id"_a, "inf if no goal is reachable.")
        .def("goal_states", [](const PyStateSpace& s) { return s.space->goal_states(); }, "Goal state ids, ascending.")
        .def("unsolvable_states", [](const PyStateSpace& s) { return s.space->unsolvable_states(); },
             "Unsolvable state ids, ascending.")
        .def("__len__", [](const PyStateSpace& s) { return s.space->num_states(); })
        .def("__repr__", [](const PyStateSpace& s) {
            return "StateSpace(states=" + std::to_string(s.space->num_states()) + ", transitions=" +
                   std::to_string(s.space->num_transitions()) + ", goal=" + std::to_string(s.space->num_goal_states()) +
                   ", unsolvable=" + std::to_string(s.space->num_unsolvable_states()) +
                   (s.space->symmetry_reduced() ? ", symmetry_reduced=True)" : ")");
        });

    nb::class_<PyGenerationResult>(m, "GenerationResult")
        .def_prop_ro("status", [](const PyGenerationResult& r) { return r.status; })
        .def_prop_ro("space", [](const PyGenerationResult& r) { return r.space; }, "The state space (None unless status is OK).")
        .def_prop_ro("states", [](const PyGenerationResult& r) { return r.states; },
                     "States stored when the generation ended (also when it failed).")
        .def_prop_ro("seconds", [](const PyGenerationResult& r) { return r.seconds; })
        .def("__bool__", [](const PyGenerationResult& r) { return r.space.has_value(); })
        .def("__repr__", [](const PyGenerationResult& r) {
            return std::string("GenerationResult(status=") + to_string(r.status) + ", states=" + std::to_string(r.states) + ")";
        });

    m.def(
        "generate",
        [](TaskArg task, u32 threads, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool symmetry_pruning, StrArg certificate, u32 k, bool labels) {
            const Owner o = task_owner(task);
            const StateSpaceOptions opts =
                make_options(threads, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate, k, labels);
            return wrap(o, run_generation(o.core->task, opts));
        },
        "task"_a, MYMYR_SS_ARGS, (std::string("Generates the state space of a task: a GenerationResult. ") + k_options_doc).c_str());

    m.def(
        "state_space",
        [](TaskArg task, u32 threads, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool symmetry_pruning, StrArg certificate, u32 k, bool labels) -> std::optional<PyStateSpace> {
            const Owner o = task_owner(task);
            const StateSpaceOptions opts =
                make_options(threads, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate, k, labels);
            return wrap(o, run_generation(o.core->task, opts)).space;
        },
        "task"_a, MYMYR_SS_ARGS,
        (std::string("The state space of a task, or None if the generation failed (as in mimir's StateSpace.create). ") +
         k_options_doc)
            .c_str());

    m.def(
        "generate_many",
        [](TasksArg tasks, u32 threads, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool symmetry_pruning, StrArg certificate, u32 k, bool labels) {
            const std::vector<Owner> owners = owners_of(tasks);
            StateSpaceOptions opts =
                make_options(1, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate, k, labels);
            std::vector<TaskPtr> ts;
            for (const Owner& o : owners)
                ts.push_back(o.core->task);
            std::vector<StateSpaceResult> rs;
            {
                nb::gil_scoped_release release;
                rs = generate_state_spaces(ts, opts, threads);
            }
            nb::typed<nb::list, PyGenerationResult> out{nb::list()};
            for (usize i = 0; i < rs.size(); ++i)
                out.append(nb::cast(wrap(owners[i], std::move(rs[i])), nb::rv_policy::move));
            return out;
        },
        "tasks"_a, nb::kw_only(), "threads"_a = 0, "max_states"_a = nb::none(), "max_seconds"_a = nb::none(),
        "remove_if_unsolvable"_a = true, "symmetry_pruning"_a = false, "certificate"_a = "kfwl", "k"_a = 2,
        "labels"_a = true,
        "The instance pool: the state spaces of many tasks, one task per worker thread (threads workers; 0: all "
        "cores), each generated single-threaded. Results in input order. Options as generate().");

    // ---------------------------------------------------------------------------------------------- generalized
    nb::class_<PyGeneralizedStateSpace>(m, "GeneralizedStateSpace",
                                        "A class graph over the state spaces of several problems of one domain (as in "
                                        "mimir's GeneralizedStateSpace; datasets/generalized_state_space.hpp). Without "
                                        "symmetry reduction the disjoint union of the problem graphs; with it (every "
                                        "space symmetry reduced) one class vertex per certificate class, problems "
                                        "isomorphic to earlier ones dropped.")
        .def(
            "__init__",
            [](PyGeneralizedStateSpace* self, nb::typed<nb::sequence, PyStateSpace> spaces) {
                std::vector<PyStateSpace> in;
                for (nb::handle h : spaces)
                {
                    if (!nb::isinstance<PyStateSpace>(h))
                        throw nb::type_error("mymyr: spaces must be StateSpaces");
                    in.push_back(nb::cast<PyStateSpace>(h));
                }
                std::vector<StateSpacePtr> sp;
                for (const PyStateSpace& s : in)
                    sp.push_back(s.space);
                GeneralizedStateSpacePtr g;
                {
                    nb::gil_scoped_release release;  // std::invalid_argument (other domains) becomes a ValueError
                    g = GeneralizedStateSpace::create(std::move(sp));
                }
                auto* p = new (self) PyGeneralizedStateSpace{g, {}};
                for (const StateSpacePtr& s : g->spaces())
                    for (const PyStateSpace& x : in)
                        if (x.space == s)
                        {
                            p->spaces.push_back(x);
                            break;
                        }
            },
            "spaces"_a, "From state spaces in their order (see sorted_by_size / generalized_state_space for mimir's order).")
        .def_prop_ro("spaces", [](const PyGeneralizedStateSpace& g) { return g.spaces; },
                     "The problems' state spaces kept, in class-graph problem order.")
        .def_prop_ro("symmetry_reduced", [](const PyGeneralizedStateSpace& g) { return g.gss->symmetry_reduced(); })
        .def_prop_ro("num_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->num_vertices(); })
        .def_prop_ro("num_edges", [](const PyGeneralizedStateSpace& g) { return g.gss->num_edges(); })
        .def(
            "arrays",
            [](const PyGeneralizedStateSpace& g, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                const std::shared_ptr<const void> own = g.gss;
                const GeneralizedStateSpace& G = *g.gss;
                ArrayDict d{nb::dict()};
                d["vertex_problems"] = view<u32>(own, G.vertex_problems(), fw);
                d["vertex_problem_vertices"] = view<u32>(own, G.vertex_problem_vertices(), fw);
                d["edge_sources"] = view<u32>(own, G.edge_sources(), fw);
                d["edge_targets"] = view<u32>(own, G.edge_targets(), fw);
                d["edge_problems"] = view<u32>(own, G.edge_problems(), fw);
                d["edge_problem_edges"] = view<u32>(own, G.edge_problem_edges(), fw);
                d["forward_offsets"] = view<u64>(own, G.forward_offsets(), fw);
                d["forward_edges"] = view<u32>(own, G.forward_edges(), fw);
                d["initial"] = view<u8>(own, G.initial_flags(), fw);
                d["goal"] = view<u8>(own, G.goal_flags(), fw);
                d["unsolvable"] = view<u8>(own, G.unsolvable_flags(), fw);
                return d;
            },
            "framework"_a = nb::none(),
            "Zero-copy views of the class graph: per class vertex its representative (vertex_problems, "
            "vertex_problem_vertices) and flags (initial, goal, unsolvable); per class edge its ends (edge_sources, "
            "edge_targets) and representative (edge_problems, edge_problem_edges); the forward CSR (forward_offsets, "
            "forward_edges).")
        .def(
            "vertex_mapping",
            [](const PyGeneralizedStateSpace& g, u32 problem, FrameworkArg framework) {
                if (problem >= g.spaces.size())
                    throw nb::index_error("mymyr: problem index out of range");
                return AnyArray(view<u32>(g.gss, g.gss->vertex_mapping(problem), parse_framework(framework)));
            },
            "problem"_a, "framework"_a = nb::none(), "The class vertex of every vertex of a problem's state space.")
        .def(
            "edge_mapping",
            [](const PyGeneralizedStateSpace& g, u32 problem, FrameworkArg framework) {
                if (problem >= g.spaces.size())
                    throw nb::index_error("mymyr: problem index out of range");
                return AnyArray(view<u32>(g.gss, g.gss->edge_mapping(problem), parse_framework(framework)));
            },
            "problem"_a, "framework"_a = nb::none(), "The class edge of every transition of a problem's state space.")
        .def("initial_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->initial_vertices(); })
        .def("goal_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->goal_vertices(); })
        .def("unsolvable_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->unsolvable_vertices(); })
        .def("__repr__", [](const PyGeneralizedStateSpace& g) {
            return "GeneralizedStateSpace(problems=" + std::to_string(g.spaces.size()) + ", vertices=" +
                   std::to_string(g.gss->num_vertices()) + ", edges=" + std::to_string(g.gss->num_edges()) +
                   (g.gss->symmetry_reduced() ? ", symmetry_reduced=True)" : ")");
        });

    m.def(
        "sorted_by_size",
        [](nb::typed<nb::sequence, std::optional<PyStateSpace>> spaces) {
            std::vector<PyStateSpace> in;
            for (nb::handle h : spaces)
                if (!h.is_none())
                    in.push_back(nb::cast<PyStateSpace>(h));
            std::stable_sort(in.begin(), in.end(), [](const PyStateSpace& a, const PyStateSpace& b) {
                return a.space->num_states() < b.space->num_states();
            });
            return in;
        },
        "spaces"_a,
        "The spaces (None entries dropped) sorted ascending by their number of states, ties in input order: "
        "mimir's order of the problems of a generalized state space.");

    // ---------------------------------------------------------------------------------------------- samplers
    nb::class_<PySampler>(m, "StateSpaceSampler",
                          "Uniform samplers over a state space (as in mimir's StateSpaceSampler): any state, states n "
                          "steps from a goal, dead ends. Deterministic for a seed on every platform (xoshiro256**, not "
                          "mimir's std::mt19937). Samples are state ids. The space may be a device state space "
                          "(mymyr.cuda.DeviceStateSpace, CUDA builds): the sampler then holds its unit goal distances "
                          "on the host (downloaded once) and gives the samples of its host copy.")
        .def(
            "__init__",
            [](PySampler* self, SamplerSpace space, u64 seed) {
                if (nb::isinstance<PyStateSpace>(space))
                {
                    new (self) PySampler{space, std::make_shared<SamplerCore>(nb::cast<const PyStateSpace&>(space).space, seed)};
                    return;
                }
                std::vector<i32> unit;
                if (!g_device_distances || !g_device_distances(space, unit))
                    throw nb::type_error("mymyr: StateSpaceSampler needs a StateSpace (or a mymyr.cuda.DeviceStateSpace)");
                new (self) PySampler{space, std::make_shared<SamplerCore>(std::span<const i32>(unit), seed)};
            },
            "space"_a, "seed"_a = 0)
        .def_prop_ro("space", [](const PySampler& s) { return SamplerSpace(s.space); })
        .def("set_seed", [](PySampler& s, u64 seed) {
            std::lock_guard lock(s.core->mutex);
            s.core->sampler.set_seed(seed);
        }, "seed"_a)
        .def("sample_state", [](PySampler& s) {
            std::lock_guard lock(s.core->mutex);
            return s.core->sampler.sample_state();
        })
        .def("sample_state_n_steps_from_goal", [](PySampler& s, i32 n) {
            std::lock_guard lock(s.core->mutex);
            try
            {
                return s.core->sampler.sample_state_n_steps_from_goal(n);
            }
            catch (const std::out_of_range& e)
            {
                throw nb::value_error(e.what());
            }
        }, "n"_a, "A state with unit goal distance n (ValueError if there is none).")
        .def("sample_dead_end_state", [](PySampler& s) {
            std::lock_guard lock(s.core->mutex);
            try
            {
                return s.core->sampler.sample_dead_end_state();
            }
            catch (const std::out_of_range& e)
            {
                throw nb::value_error(e.what());
            }
        })
        .def(
            "sample_states",
            [](PySampler& s, u64 count, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                std::vector<u32> v(count);
                {
                    std::lock_guard lock(s.core->mutex);
                    s.core->sampler.sample_states(v);
                }
                return AnyArray(fresh_u32(v, fw));
            },
            "count"_a, "framework"_a = nb::none(), "count uniform state ids (uint32).")
        .def(
            "sample_states_n_steps_from_goal",
            [](PySampler& s, i32 n, u64 count, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                std::vector<u32> v(count);
                {
                    std::lock_guard lock(s.core->mutex);
                    try
                    {
                        s.core->sampler.sample_states_n_steps_from_goal(n, v);
                    }
                    catch (const std::out_of_range& e)
                    {
                        throw nb::value_error(e.what());
                    }
                }
                return AnyArray(fresh_u32(v, fw));
            },
            "n"_a, "count"_a, "framework"_a = nb::none())
        .def(
            "sample_dead_end_states",
            [](PySampler& s, u64 count, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                std::vector<u32> v(count);
                {
                    std::lock_guard lock(s.core->mutex);
                    try
                    {
                        s.core->sampler.sample_dead_end_states(v);
                    }
                    catch (const std::out_of_range& e)
                    {
                        throw nb::value_error(e.what());
                    }
                }
                return AnyArray(fresh_u32(v, fw));
            },
            "count"_a, "framework"_a = nb::none())
        .def_prop_ro("num_states", [](const PySampler& s) { return s.core->sampler.num_states(); })
        .def_prop_ro("num_dead_end_states", [](const PySampler& s) { return s.core->sampler.num_dead_end_states(); })
        .def_prop_ro("num_alive_states", [](const PySampler& s) { return s.core->sampler.num_alive_states(); },
                     "States that are not dead ends (goal states included), as in mimir's sampler.")
        .def_prop_ro("max_steps_to_goal", [](const PySampler& s) { return s.core->sampler.max_steps_to_goal(); })
        .def(
            "states_n_steps_from_goal",
            [](const PySampler& s, i32 n, FrameworkArg framework) {
                // the id lists are the sampler's (fixed at construction): the view keeps the sampler alive
                const auto v = s.core->sampler.states_n_steps_from_goal(n);
                return AnyArray(view<u32>(s.core, v, parse_framework(framework)));
            },
            "n"_a, "framework"_a = nb::none(), "The ids with unit goal distance n, ascending (a view).")
        .def(
            "dead_end_states",
            [](const PySampler& s, FrameworkArg framework) {
                return AnyArray(view<u32>(s.core, s.core->sampler.dead_end_states(), parse_framework(framework)));
            },
            "framework"_a = nb::none());

    // ---------------------------------------------------------------------------------------------- object graphs
    nb::class_<PyObjectGraph>(m, "ObjectGraph",
                              "The vertex-coloured object graph of a state (as in mimir's create_object_graph; "
                              "datasets/object_graph.hpp): vertex i < num_objects is object i. A structure for "
                              "isomorphism tests, not an observation encoder.")
        .def_prop_ro("num_objects", [](const PyObjectGraph& g) { return g.graph->num_objects; })
        .def_prop_ro("num_vertices", [](const PyObjectGraph& g) { return g.graph->num_vertices(); })
        .def_prop_ro("num_edges", [](const PyObjectGraph& g) { return g.graph->num_edges(); }, "Undirected edges.")
        .def_prop_ro("num_colors", [](const PyObjectGraph& g) { return g.graph->num_colors(); })
        .def(
            "palette", [](const PyObjectGraph& g, u32 c) {
                if (c >= g.graph->num_colors())
                    throw nb::index_error("mymyr: colour out of range");
                const auto p = g.graph->palette(c);
                return std::vector<u32>(p.begin(), p.end());
            },
            "color"_a,
            "A colour as its integer sequence: object vertex [0, n, preds.., m, (pred, polarity)..], atom [1, pred], atom "
            "position [2, pred, pos], literal [3, pred, polarity], literal position [4, pred, pos, polarity].")
        .def(
            "arrays",
            [](const PyObjectGraph& g, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                const std::shared_ptr<const void> own = g.graph;
                const ObjectGraph& G = *g.graph;
                ArrayDict d{nb::dict()};
                d["color"] = view<u32>(own, std::span<const u32>(G.color), fw);
                d["palette_offsets"] = view<u32>(own, std::span<const u32>(G.palette_offsets), fw);
                d["palette_values"] = view<u32>(own, std::span<const u32>(G.palette_values), fw);
                d["offsets"] = view<u64>(own, std::span<const u64>(G.offsets), fw);
                d["neighbors"] = view<u32>(own, std::span<const u32>(G.neighbors), fw);
                return d;
            },
            "framework"_a = nb::none(),
            "Zero-copy views: color [V] (palette index), palette_offsets / palette_values (the colours' integer "
            "sequences, sorted), offsets [V + 1] and neighbors (undirected adjacency as CSR, both directions).")
        .def(
            "color_refinement_certificate",
            [](const PyObjectGraph& g) {
                Certificate c;
                {
                    nb::gil_scoped_release release;
                    c = color_refinement_certificate(*g.graph);
                }
                return certificate_int(c);
            },
            "A 128-bit certificate by colour refinement (1-WL): equal for isomorphic graphs.")
        .def(
            "kfwl_certificate",
            [](const PyObjectGraph& g, u32 k, std::optional<u64> max_tuples, std::optional<u64> max_round_work) {
                if (k < 2 || k > 4)
                    throw nb::value_error("mymyr: k-FWL certificates support k = 2, 3 and 4");
                KfwlLimits limits;
                if (max_tuples)
                    limits.max_tuples = *max_tuples;
                if (max_round_work)
                    limits.max_round_work = *max_round_work;
                Certificate c;
                {
                    nb::gil_scoped_release release;
                    c = kfwl_certificate(*g.graph, k, limits);
                }
                return certificate_int(c);
            },
            "k"_a = 2, "max_tuples"_a = nb::none(), "max_round_work"_a = nb::none(),
            "A 128-bit certificate by k-dimensional folklore Weisfeiler-Leman (k = 2, 3 or 4): equal for isomorphic "
            "graphs. With n vertices it holds n^k tuples of 28 bytes and hashes n^(k+1) colour k-tuples per round; "
            "max_tuples (default 2^26) and max_round_work (default 2^30) bound them, and a larger graph raises "
            "ValueError.")
        .def(
            "stable_colors",
            [](const PyObjectGraph& g) {
                std::vector<u32> colors;
                (void)color_refinement_certificate(*g.graph, &colors);
                return colors;
            },
            "The stable colour refinement colour of every vertex (canonical across isomorphic graphs).")
        .def("__repr__", [](const PyObjectGraph& g) {
            return "ObjectGraph(vertices=" + std::to_string(g.graph->num_vertices()) + ", edges=" +
                   std::to_string(g.graph->num_edges()) + ", colors=" + std::to_string(g.graph->num_colors()) + ")";
        });

    nb::class_<PyObjectGraphBuilder>(m, "ObjectGraphBuilder",
                                     "Builds object graphs of one task's states (per-task tables built once).")
        .def(
            "__init__", [](PyObjectGraphBuilder* self, TaskArg task) { new (self) PyObjectGraphBuilder(task_owner(task)); },
            "task"_a)
        .def(
            "build",
            [](PyObjectGraphBuilder& b, StateArg state) {
                if (!is_state(state))
                    throw nb::type_error("mymyr: state must be a mymyr.State");
                const PyState& p = state_of(state);
                if (p.core->task->uid() != b.owner.core->task->uid())
                    throw nb::value_error("mymyr: the state belongs to another task");
                auto g = std::make_shared<ObjectGraph>();
                {
                    nb::gil_scoped_release release;
                    std::lock_guard lock(b.mutex);
                    b.builder.build(p.s.view(), *g);
                }
                return PyObjectGraph{std::move(g)};
            },
            "state"_a);

    m.def(
        "object_graph",
        [](StateArg state) {
            if (!is_state(state))
                throw nb::type_error("mymyr: state must be a mymyr.State");
            const PyState& p = state_of(state);
            auto g = std::make_shared<ObjectGraph>();
            {
                nb::gil_scoped_release release;
                *g = datasets::object_graph(*p.core->task, p.s.view());
            }
            return PyObjectGraph{std::move(g)};
        },
        "state"_a, "The object graph of a state (ObjectGraphBuilder for many states of one task).");

    // ---------------------------------------------------------------------------------------------- tuple graphs
    nb::class_<PyTupleGraph>(
        m, "TupleGraph",
        "The tuple graph of a state-space vertex (the root), as in mimir's TupleGraph (datasets/tuple_graph.hpp; "
        "Lipovetzky and Geffner 2012): which tuples of at most `width` fluent atoms are first reached at each "
        "breadth-first distance from the root, the problem vertices (states) at that distance in which each is novel, "
        "and the edges u -> t between consecutive distances where every problem vertex of u has a successor among "
        "those of t. Width 0: the root and one vertex per successor state. Vertices are ordered by distance; "
        "dominance pruning (default) keeps the vertices with minimal problem-vertex sets. Over a symmetry-reduced space "
        "the problem vertices are class vertices. The result does not depend on the thread count.")
        .def_prop_ro("space", [](const PyTupleGraph& g) { return g.space; }, "The state space the root belongs to.")
        .def_prop_ro("root", [](const PyTupleGraph& g) { return g.graph->root(); }, "The root's state id.")
        .def_prop_ro("width", [](const PyTupleGraph& g) { return g.graph->width(); })
        .def_prop_ro("dominance_pruning", [](const PyTupleGraph& g) { return g.graph->dominance_pruning(); })
        .def_prop_ro("num_vertices", [](const PyTupleGraph& g) { return g.graph->num_vertices(); })
        .def_prop_ro("num_edges", [](const PyTupleGraph& g) { return g.graph->num_edges(); })
        .def_prop_ro("num_distances", [](const PyTupleGraph& g) { return g.graph->num_distances(); },
                     "The number of distances with vertices (the largest distance + 1).")
        .def_prop_ro("nbytes", [](const PyTupleGraph& g) { return g.graph->bytes(); })
        .def(
            "vertices_at",
            [](const PyTupleGraph& g, i64 d) {
                const u32 k = tuple_distance(*g.graph, d);
                std::vector<u32> out;
                for (const u32 v : g.graph->vertices_at(k))
                    out.push_back(v);
                return out;
            },
            "distance"_a, "The vertices at a distance (consecutive ids, ascending).")
        .def("distance", [](const PyTupleGraph& g, i64 v) { return g.graph->distance(tuple_vertex(*g.graph, v)); }, "vertex"_a)
        .def("tuple", [](const PyTupleGraph& g, i64 v) { return to_vector(g.graph->tuple(tuple_vertex(*g.graph, v))); },
             "vertex"_a, "The vertex's tuple: fluent atom slots of the space's task, ascending (empty: the empty tuple).")
        .def(
            "atoms",
            [](const PyTupleGraph& g, i64 v) {
                const Owner& o = g.space.owner;
                const Task& task = *o.core->task;
                nb::typed<nb::list, PyGroundAtom> out{nb::list()};
                for (const u32 s : g.graph->tuple(tuple_vertex(*g.graph, v)))
                {
                    const SlotId slot{s};
                    const auto args = task.atoms().arguments(slot);
                    out.append(make_ground_atom(mymyr::python::task_owner(o),
                                                GroundAtom{task.atoms().predicate(slot), std::vector<ObjectId>(args.begin(), args.end())}));
                }
                return out;
            },
            "vertex"_a, "The vertex's tuple as GroundAtoms (slot order).")
        .def("problem_vertices",
             [](const PyTupleGraph& g, i64 v) { return to_vector(g.graph->problem_vertices(tuple_vertex(*g.graph, v))); }, "vertex"_a,
             "The states at the vertex's distance in which its tuple is novel (state ids, ascending).")
        .def("successors", [](const PyTupleGraph& g, i64 v) { return to_vector(g.graph->successors(tuple_vertex(*g.graph, v))); },
             "vertex"_a, "The vertices at the next distance with an edge from this one (ascending).")
        .def("predecessors",
             [](const PyTupleGraph& g, i64 v) { return to_vector(g.graph->predecessors(tuple_vertex(*g.graph, v))); }, "vertex"_a,
             "The vertices at the previous distance with an edge to this one (ascending).")
        .def("problem_vertices_at",
             [](const PyTupleGraph& g, i64 d) { return to_vector(g.graph->problem_vertices_at(tuple_distance(*g.graph, d))); },
             "distance"_a, "The states at a breadth-first distance from the root (state ids, ascending).")
        .def(
            "arrays",
            [](const PyTupleGraph& g, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                const std::shared_ptr<const void> own = g.graph;
                const TupleGraph& G = *g.graph;
                ArrayDict d{nb::dict()};
                d["distance_offsets"] = view<u32>(own, G.distance_offsets(), fw);
                d["tuple_offsets"] = view<u32>(own, G.tuple_offsets(), fw);
                d["tuple_atoms"] = view<u32>(own, G.tuple_atoms(), fw);
                d["problem_offsets"] = view<u32>(own, G.problem_offsets(), fw);
                d["problem_vertices"] = view<u32>(own, G.problem_vertex_ids(), fw);
                d["successor_offsets"] = view<u32>(own, G.successor_offsets(), fw);
                d["successors"] = view<u32>(own, G.successor_ids(), fw);
                d["predecessor_offsets"] = view<u32>(own, G.predecessor_offsets(), fw);
                d["predecessors"] = view<u32>(own, G.predecessor_ids(), fw);
                d["layer_offsets"] = view<u32>(own, G.layer_offsets(), fw);
                d["layer_vertices"] = view<u32>(own, G.layer_vertex_ids(), fw);
                return d;
            },
            "framework"_a = nb::none(),
            "Zero-copy views (uint32) of the CSR arrays: vertex v's tuple is tuple_atoms[tuple_offsets[v] : "
            "tuple_offsets[v + 1]], and likewise its problem vertices (problem_offsets, problem_vertices), successors "
            "and predecessors; the vertices at distance d are distance_offsets[d] .. distance_offsets[d + 1] - 1 and "
            "the states at distance d are layer_vertices[layer_offsets[d] : layer_offsets[d + 1]].")
        .def("__len__", [](const PyTupleGraph& g) { return g.graph->num_vertices(); })
        .def("__eq__", [](const PyTupleGraph& a, const PyTupleGraph& b) { return *a.graph == *b.graph; }, nb::is_operator(),
             "other"_a, "The same state space, root and options, and equal vertices, tuples, problem vertices and edges.")
        .def("__repr__", [](const PyTupleGraph& g) {
            return "TupleGraph(root=" + std::to_string(g.graph->root()) + ", width=" + std::to_string(g.graph->width()) +
                   ", vertices=" + std::to_string(g.graph->num_vertices()) + ", edges=" + std::to_string(g.graph->num_edges()) +
                   ", distances=" + std::to_string(g.graph->num_distances()) + ")";
        });

    m.def(
        "tuple_graphs",
        [](const PyStateSpace& space, u32 width, bool dominance_pruning, u32 threads) {
            const TupleGraphOptions o{.width = width, .dominance_pruning = dominance_pruning, .threads = threads};
            std::vector<TupleGraph> graphs;
            {
                nb::gil_scoped_release release;  // std::invalid_argument (a width above 5) becomes a ValueError
                graphs = datasets::tuple_graphs(space.space, o);
            }
            nb::typed<nb::list, PyTupleGraph> out{nb::list()};
            for (TupleGraph& g : graphs)
                out.append(nb::cast(PyTupleGraph{std::make_shared<const TupleGraph>(std::move(g)), space}, nb::rv_policy::move));
            return out;
        },
        "space"_a, nb::kw_only(), "width"_a = 0, "dominance_pruning"_a = true, "threads"_a = 0,
        "The tuple graph of every vertex of a state space (index = state id), as mimir's TupleGraphFactory: width "
        "0..5 (0: the root and its successor states), dominance_pruning, threads (0: all cores; the result is the "
        "same at every count). Cost: a breadth-first search per vertex, enumerating the tuples of at most width atoms "
        "of every state it reaches; tuple_graph() builds one.");

    m.def(
        "tuple_graph",
        [](const PyStateSpace& space, i64 vertex, u32 width, bool dominance_pruning) {
            if (vertex < 0 || vertex >= static_cast<i64>(space.space->num_states()))
                throw nb::index_error("mymyr: state id out of range");
            const TupleGraphOptions o{.width = width, .dominance_pruning = dominance_pruning, .threads = 1};
            std::shared_ptr<const TupleGraph> g;
            {
                nb::gil_scoped_release release;
                g = std::make_shared<const TupleGraph>(datasets::tuple_graph(space.space, static_cast<u32>(vertex), o));
            }
            return PyTupleGraph{std::move(g), space};
        },
        "space"_a, "vertex"_a, nb::kw_only(), "width"_a = 0, "dominance_pruning"_a = true,
        "The tuple graph of one vertex of a state space (equal to tuple_graphs(space)[vertex]).");

    // ---------------------------------------------------------------------------------------------- knowledge bases
    nb::class_<PyKnowledgeBase>(
        m, "KnowledgeBase",
        "What is known about a set of tasks of one domain, as mimir's KnowledgeBase (datasets/knowledge_base.hpp): the "
        "tasks (a mymyr.rl.TaskTable), the state space of every task whose generation succeeded (failures skipped; "
        "max_states and max_seconds bound each one), sorted ascending by size unless sort_by_size=False (ties in task "
        "order), optionally the generalized state space over them (generalized=True; with symmetry_pruning, problems "
        "isomorphic to an earlier one are dropped from the knowledge base), and optionally the tuple graphs of every "
        "vertex of every space (width=0..5; tuple graphs of large spaces are better built per vertex with "
        "tuple_graph(space, vertex)). Every step runs on `threads` threads (0: all cores) with the same result at "
        "every count. Pickling stores the tasks and the arguments, and unpickling builds the knowledge base again.")
        .def(
            "__init__",
            [](PyKnowledgeBase* self, KbTasksArg tasks, u32 threads, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
               bool symmetry_pruning, StrArg certificate, u32 k, bool labels, bool sort_by_size, bool generalized,
               std::optional<u32> width, bool dominance_pruning) {
                init_knowledge_base(self, tasks, threads, max_states, max_seconds, remove_if_unsolvable, symmetry_pruning, certificate,
                                    k, labels, sort_by_size, generalized, width, dominance_pruning);
            },
            "tasks"_a, MYMYR_KB_ARGS,
            "The knowledge base of tasks of one domain: a TaskTable or a sequence of Tasks (made into a TaskTable). The "
            "state space options are those of generate().")
        .def_prop_ro("tasks", [](const PyKnowledgeBase& kb) { return nb::typed<nb::object, PyTable>(kb.table); },
                     "The TaskTable of the tasks.")
        .def_prop_ro("state_spaces", [](const PyKnowledgeBase& kb) { return kb.spaces; },
                     "The state spaces in knowledge-base order.")
        .def_prop_ro(
            "task_indices",
            [](const PyKnowledgeBase& kb) { return std::vector<u32>(kb.kb->task_indices().begin(), kb.kb->task_indices().end()); },
            "Per state space: the index of its task in tasks.")
        .def_prop_ro("generalized_state_space", [](const PyKnowledgeBase& kb) { return kb.gss; },
                     "The generalized state space over state_spaces (problem i = state space i), or None unless "
                     "generalized=True.")
        .def_prop_ro("has_tuple_graphs", [](const PyKnowledgeBase& kb) { return kb.kb->has_tuple_graphs(); })
        .def_prop_ro(
            "width",
            [](const PyKnowledgeBase& kb) {
                return kb.kb->has_tuple_graphs() ? std::optional<u32>(kb.kb->options().tuple_graphs->width) : std::nullopt;
            },
            "The tuple graph width, or None without tuple graphs.")
        .def(
            "tuple_graphs",
            [](const PyKnowledgeBase& kb, i64 i) {
                require_tuple_graphs(kb);
                const usize s = kb_space(kb, i);
                nb::typed<nb::list, PyTupleGraph> out{nb::list()};
                for (u32 v = 0; v < kb.kb->tuple_graphs()[s].size(); ++v)
                    out.append(nb::cast(kb_tuple_graph(kb, s, v), nb::rv_policy::move));
                return out;
            },
            "space"_a, "The tuple graphs of every vertex of state space i (index = state id).")
        .def(
            "tuple_graph",
            [](const PyKnowledgeBase& kb, i64 i, i64 vertex) {
                require_tuple_graphs(kb);
                const usize s = kb_space(kb, i);
                if (vertex < 0 || vertex >= static_cast<i64>(kb.kb->tuple_graphs()[s].size()))
                    throw nb::index_error("mymyr: state id out of range");
                return kb_tuple_graph(kb, s, static_cast<u32>(vertex));
            },
            "space"_a, "vertex"_a, "The tuple graph of a vertex of state space i.")
        .def("__len__", [](const PyKnowledgeBase& kb) { return kb.spaces.size(); }, "The number of state spaces.")
        .def("__getstate__", [](const PyKnowledgeBase& kb) { return KbState(kb.args); })
        .def("__setstate__",
             [](PyKnowledgeBase* self, KbState s) {
                 init_knowledge_base(self, s[0], nb::cast<u32>(s[1]), s[2], s[3], nb::cast<bool>(s[4]), nb::cast<bool>(s[5]), s[6],
                                     nb::cast<u32>(s[7]), nb::cast<bool>(s[8]), nb::cast<bool>(s[9]), nb::cast<bool>(s[10]),
                                     nb::cast<std::optional<u32>>(s[11]), nb::cast<bool>(s[12]));
             })
        .def("__repr__", [](const PyKnowledgeBase& kb) {
            std::string r = "KnowledgeBase(spaces=" + std::to_string(kb.spaces.size()) + " of " +
                            std::to_string(kb.kb->tasks()->size()) + " tasks";
            if (kb.gss)
                r += ", generalized";
            if (kb.kb->has_tuple_graphs())
                r += ", width=" + std::to_string(kb.kb->options().tuple_graphs->width);
            return r + ")";
        });
}
#undef MYMYR_KB_ARGS
#undef MYMYR_SS_ARGS
}  // namespace mymyr::python
