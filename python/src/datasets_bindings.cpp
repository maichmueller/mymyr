// mymyr._core._datasets: state spaces, generalized state spaces, knowledge bases, tuple graphs and samplers
// (mymyr.datasets; datasets/*.hpp in the C++ core).
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

#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/knowledge_base.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
using AnyArray = Arg<ann::Any>;
using ArrayDict = nb::typed<nb::dict, std::string, ann::Any>;
/// The space of a sampler: a StateSpace, or (CUDA builds) a mymyr.cuda.DeviceStateSpace.
#if defined(MYMYR_HAS_CUDA)
using SamplerSpace = Arg<std::variant<PyStateSpace, ann::DeviceStateSpace>>;
#else
using SamplerSpace = Arg<PyStateSpace>;
#endif

// ------------------------------------------------------------------------------------------------ wrappers
// The wrappers that refer to state spaces hold their Python objects, so that a space shared by several of them is
// one object (and pickles once).
struct PyGenerationResult
{
    StateSpaceStatus status = StateSpaceStatus::Ok;
    nb::object space;  // a StateSpace, or None
    u64 states = 0;
    f64 seconds = 0;
};

struct PyGeneralizedStateSpace
{
    GeneralizedStateSpacePtr gss;
    std::vector<nb::object> spaces;  // the StateSpaces, in gss->spaces() order
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

/// A tuple graph and the state space it is of (shared: a graph of a knowledge base aliases the knowledge base).
struct PyTupleGraph
{
    std::shared_ptr<const TupleGraph> graph;
    nb::object space;  // the StateSpace
    [[nodiscard]] const PyStateSpace& py_space() const { return *nb::inst_ptr<PyStateSpace>(space); }
};

/// A knowledge base, its mymyr.rl.TaskTable, its state spaces (owned by the table's Task objects) and the keyword
/// arguments it was built with (KbOptions).
struct PyKnowledgeBase
{
    KnowledgeBasePtr kb;
    nb::object table;
    std::vector<nb::object> spaces;  // the StateSpaces
    std::optional<PyGeneralizedStateSpace> gss;
    nb::tuple args;
};

// ------------------------------------------------------------------------------------------------ helpers
StateSpaceOptions make_options(u32 threads, nb::handle max_states, nb::handle max_seconds, bool remove_if_unsolvable,
                               bool labels)
{
    StateSpaceOptions o;
    o.threads = threads;
    if (auto v = opt_int_arg<u64>(max_states, "max_states", 0))
        o.max_states = *v;
    if (auto v = opt_float_arg(max_seconds, "max_seconds", 0))
        o.max_seconds = *v;
    o.remove_if_unsolvable = remove_if_unsolvable;
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
    g.space = r.space ? nb::cast(PyStateSpace{std::move(r.space), owner}) : nb::none();
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

/// A StateSpace argument (TypeError otherwise).
const PyStateSpace& space_arg(nb::handle space)
{
    if (!nb::isinstance<PyStateSpace>(space))
        throw nb::type_error("mymyr: space must be a mymyr.datasets.StateSpace");
    return *nb::inst_ptr<PyStateSpace>(space);
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

/// The arrays of a space (zero-copy views). state_words: the rows, for a task with frozen atom slots or `rows` (the
/// pickled and saved form: lazy slots travel with them).
ArrayDict space_arrays(const PyStateSpace& P, Framework fw, bool rows = false)
{
    const StateSpace& S = *P.space;
    const std::shared_ptr<const void> own = P.space;
    ArrayDict d{nb::dict()};
    const i64 N = S.num_states(), E = static_cast<i64>(S.num_transitions());
    if (rows || S.task()->atoms().frozen())  // lazy slots: the rows' layout is the generation's history
        d["state_words"] = view<u64>(own, S.state_words(), fw, {N, static_cast<i64>(S.row_words())}, rows ? false : true);
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

/// A generalized state space index (problem, state, transition, vertex or edge) below n, or IndexError.
u64 gss_index(i64 i, u64 n, const char* what)
{
    if (i < 0 || static_cast<u64>(i) >= n)
        throw nb::index_error((std::string("mymyr: ") + what + " out of range").c_str());
    return static_cast<u64>(i);
}

// ------------------------------------------------------------------------------------------------ arrays in
// Rebuilding objects from their arrays (from_arrays, unpickling, datasets.load): host arrays of the dtypes arrays()
// gives, copied into the C++ object, which validates them (std::invalid_argument: ValueError).

/// Item `key` of a mapping, or None if it has none.
nb::object mapping_item(nb::handle mapping, const char* key)
{
    PyObject* out = nullptr;
    const int r = PyMapping_GetOptionalItemString(mapping.ptr(), key, &out);
    if (r < 0)
        throw nb::python_error();
    return r == 0 ? nb::none() : nb::steal(out);
}

/// TypeError unless `arrays` is a mapping whose keys are among `known`.
void check_keys(nb::handle arrays, std::initializer_list<std::string_view> known, const char* what)
{
    if (!nb::isinstance<nb::dict>(arrays) && (!PyMapping_Check(arrays.ptr()) || !nb::hasattr(arrays, "keys")))
        throw nb::type_error((std::string("mymyr: ") + what + " arrays must be a mapping of names to arrays (as arrays() gives them)").c_str());
    nb::object keys = nb::steal(PyMapping_Keys(arrays.ptr()));
    if (!keys.is_valid())
        throw nb::python_error();
    for (nb::handle k : keys)
    {
        if (!nb::isinstance<nb::str>(k))
            throw nb::type_error((std::string("mymyr: ") + what + " array names must be strings").c_str());
        const std::string_view name = nb::cast<std::string_view>(k);
        if (std::ranges::find(known, name) == known.end())
            throw nb::value_error((std::string("mymyr: unknown ") + what + " array '" + std::string(name) + "'").c_str());
    }
}

template<class T>
constexpr const char* dtype_name()
{
    if constexpr (std::is_same_v<T, u8>)
        return "bool";
    else if constexpr (std::is_same_v<T, i32>)
        return "int32";
    else if constexpr (std::is_same_v<T, u32>)
        return "uint32";
    else if constexpr (std::is_same_v<T, f64>)
        return "float64";
    else
        return "uint64";
}

/// An array in host memory (NumPy, or a DLPack array on the CPU), not converted.
nb::ndarray<nb::ro> host_array(nb::handle obj, const std::string& name)
{
    nb::ndarray<nb::ro> a;
    if (!nb::try_cast(obj, a, false) || a.device_type() != nb::device::cpu::value)
        throw nb::type_error(("mymyr: " + name + " must be an array in host memory (NumPy, or a DLPack array on the CPU)").c_str());
    return a;
}

/// A copy of array `key` of `arrays` (`ndim` dimensions of T; flags: bool or uint8), and its shape. Absent: nullopt
/// unless `required` (then ValueError).
template<class T>
std::optional<std::pair<std::vector<T>, std::vector<u64>>> take(nb::handle arrays, const char* key, usize ndim, bool required,
                                                                const char* what)
{
    nb::object obj = mapping_item(arrays, key);
    const std::string name = std::string(what) + " array '" + key + "'";
    if (obj.is_none())
    {
        if (required)
            throw nb::value_error(("mymyr: the " + name + " is missing").c_str());
        return std::nullopt;
    }
    const nb::ndarray<nb::ro> a = host_array(obj, name);
    const bool dtype_ok = a.dtype() == nb::dtype<T>() || (std::is_same_v<T, u8> && a.dtype() == nb::dtype<bool>());
    if (!dtype_ok || a.ndim() != ndim)
        throw nb::type_error(("mymyr: the " + name + " must be a " + std::to_string(ndim) + "-d " + dtype_name<T>() + " array").c_str());
    std::vector<u64> shape(ndim);
    u64 n = 1;
    for (usize i = 0; i < ndim; ++i)
        n *= shape[i] = a.shape(i);
    std::vector<T> v(n);
    const auto* base = static_cast<const unsigned char*>(a.data());
    const i64 s0 = a.stride(0), s1 = ndim == 2 ? a.stride(1) : 1;
    const u64 cols = ndim == 2 ? shape[1] : 1, rows = ndim == 2 ? shape[0] : shape[0];
    if (n > 0 && (ndim == 1 ? s0 == 1 : (s1 == 1 && s0 == static_cast<i64>(cols))))
        std::memcpy(v.data(), base, n * sizeof(T));
    else
        for (u64 r = 0; r < rows; ++r)
            for (u64 c = 0; c < cols; ++c)
            {
                const i64 off = ndim == 2 ? static_cast<i64>(r) * s0 + static_cast<i64>(c) * s1 : static_cast<i64>(r) * s0;
                std::memcpy(&v[r * cols + c], base + off * static_cast<i64>(sizeof(T)), sizeof(T));
            }
    return std::pair{std::move(v), std::move(shape)};
}

template<class T>
std::vector<T> take_required(nb::handle arrays, const char* key, usize ndim, const char* what)
{
    return std::move(take<T>(arrays, key, ndim, true, what)->first);
}

/// The state space of `task` from the arrays of arrays() (state_words included). The rows: [N, W + NN] 64-bit words
/// (uint64 or int64), or 32-bit halves [N, 2 (W + NN)]. Generation statistics: `stats` (threads, layers,
/// search_seconds, post_seconds) if given, else 1, 0, 0, 0.
StateSpacePtr space_from_arrays(TaskPtr task, nb::handle arrays, nb::handle stats)
{
    const char* what = "state space";
    check_keys(arrays,
               {"state_words", "forward_offsets", "forward_targets", "label_schemas", "label_bindings", "backward_offsets",
                "backward_sources", "backward_edges", "unit_goal_distances", "cost_goal_distances", "costs", "goal", "unsolvable",
                "alive"},
               what);
    StateSpaceArrays a;
    a.forward_offsets = take_required<u64>(arrays, "forward_offsets", 1, what);
    if (a.forward_offsets.empty())
        throw nb::value_error("mymyr: the state space array 'forward_offsets' is empty (it has num_states + 1 entries)");
    const u64 N = a.forward_offsets.size() - 1;
    if (N > std::numeric_limits<u32>::max())
        throw nb::value_error("mymyr: a state space has fewer than 2^32 states");
    a.num_states = static_cast<u32>(N);
    // the rows, in any of the word encodings arrays() gives
    {
        nb::object obj = mapping_item(arrays, "state_words");
        if (obj.is_none())
            throw nb::value_error("mymyr: the state space array 'state_words' is missing");
        const nb::ndarray<nb::ro> w = host_array(obj, "the state space array 'state_words'");
        if (w.ndim() != 2)
            throw nb::type_error("mymyr: the state space array 'state_words' must be 2-d [num_states, words + numeric_words]");
        const i64 shape[2] = {static_cast<i64>(w.shape(0)), static_cast<i64>(w.shape(1))};
        const i64 strides[2] = {w.stride(0), w.stride(1)};
        const nb::dlpack::dtype dt = w.dtype();
        const WordsLayout L = words_layout(w.data(), dt.code, dt.bits, dt.lanes, 2, shape, strides);
        if (L.rows != N)
            throw nb::value_error("mymyr: the state space array 'state_words' does not have num_states rows");
        const u32 NN = task->numeric_words();
        if (L.words < NN)
            throw nb::value_error("mymyr: the state space array 'state_words' is narrower than the task's numeric words");
        a.words = L.words - NN;
        a.numeric_words = NN;
        a.state_words.resize(N * L.words);
        for (u64 r = 0; r < N; ++r)
            std::memcpy(a.state_words.data() + r * L.words, L.data + r * L.stride, u64{L.words} * sizeof(u64));
    }
    a.forward_targets = take_required<u32>(arrays, "forward_targets", 1, what);
    auto schemas = take<u32>(arrays, "label_schemas", 1, false, what);
    auto bindings = take<u32>(arrays, "label_bindings", 2, false, what);
    if (schemas.has_value() != bindings.has_value())
        throw nb::value_error("mymyr: the state space arrays have label_schemas without label_bindings or the reverse");
    if (schemas)
    {
        a.labels = true;
        a.label_schemas = std::move(schemas->first);
        a.label_width = static_cast<u32>(bindings->second[1]);
        a.label_bindings = std::move(bindings->first);
    }
    a.backward_offsets = take_required<u64>(arrays, "backward_offsets", 1, what);
    a.backward_sources = take_required<u32>(arrays, "backward_sources", 1, what);
    a.backward_edges = take_required<u32>(arrays, "backward_edges", 1, what);
    a.unit_goal_distances = take_required<i32>(arrays, "unit_goal_distances", 1, what);
    a.cost_goal_distances = take_required<f64>(arrays, "cost_goal_distances", 1, what);
    if (auto c = take<f64>(arrays, "costs", 1, false, what))
        a.costs = std::move(c->first);
    a.goal_flags = take_required<u8>(arrays, "goal", 1, what);
    a.unsolvable_flags = take_required<u8>(arrays, "unsolvable", 1, what);
    a.alive_flags = take_required<u8>(arrays, "alive", 1, what);
    if (!stats.is_none())
    {
        const nb::tuple t = nb::cast<nb::tuple>(stats);
        if (t.size() != 4)
            throw nb::value_error("mymyr: corrupt state space statistics");
        a.threads = nb::cast<u32>(t[0]);
        a.layers = nb::cast<u32>(t[1]);
        a.search_seconds = nb::cast<f64>(t[2]);
        a.post_seconds = nb::cast<f64>(t[3]);
    }
    a.task = std::move(task);
    nb::gil_scoped_release release;  // validation: linear in the size of the arrays
    return StateSpace::create(std::move(a));
}

/// The generation statistics of a space: (threads, layers, search_seconds, post_seconds).
/// (threads, layers, search seconds, post seconds) of a state space.
using SpaceStats = nb::typed<nb::tuple, u32, u32, double, double>;
/// Arrays by name, as given to from_arrays and the reconstructors.
using ArraysArg = Arg<nb::typed<nb::mapping, std::string, ann::ArrayLike>>;

SpaceStats space_stats(const StateSpace& S)
{
    return SpaceStats(nb::make_tuple(S.threads(), S.layers(), S.search_seconds(), S.post_seconds()));
}

/// A fresh NumPy u64 array (the lazy slot table).
nb::object fresh_u64(const std::vector<u64>& v)
{
    auto block = Block::make(std::max<u64>(v.size(), 1) * sizeof(u64));
    if (!v.empty())
        std::memcpy(block->data(), v.data(), v.size() * sizeof(u64));
    ArraySpec spec{block, block->data(), rl::DType::U64, {static_cast<i64>(v.size())}, {}, false, false};
    return export_array(std::move(spec), Framework::Numpy, default_words(Framework::Numpy));
}

/// Checks that `task` is the task a state space was saved from (content fingerprint; encoding fingerprint for frozen
/// atom slots), and gives a task with lazy slots the saved numbering (`slots`: lazy_slots of the saved task).
void adopt_task(const Task& T, u64 fingerprint, u64 encoding_fingerprint, nb::handle slots, const char* what)
{
    if (T.fingerprint() != fingerprint)
        throw nb::value_error((std::string("mymyr: the ") + what + " is of another task (content fingerprint mismatch)").c_str());
    if (T.atoms().frozen())
    {
        if (!slots.is_none() && nb::len(slots) != 0)
            throw nb::value_error((std::string("mymyr: the ") + what + " was saved under lazy atom slots; the task has frozen ones").c_str());
        if (T.encoding_fingerprint() != encoding_fingerprint)
            throw nb::value_error((std::string("mymyr: the task encodes states differently from the one the ") + what +
                                   " was saved with (encoding fingerprint mismatch: another atom mode, or another mymyr version)")
                                      .c_str());
        return;
    }
    if (slots.is_none())
        throw nb::value_error((std::string("mymyr: the ") + what + " was saved under frozen atom slots; the task has lazy ones").c_str());
    const nb::ndarray<nb::ro> a = host_array(slots, "the lazy atom slots");
    if (a.dtype() != nb::dtype<u64>() || a.ndim() != 1 || (a.shape(0) > 0 && a.stride(0) != 1))
        throw nb::type_error("mymyr: the lazy atom slots must be a contiguous 1-d uint64 array");
    adopt_lazy_slots(T, std::span<const u64>(static_cast<const u64*>(a.data()), a.shape(0)), what);
}

// ------------------------------------------------------------------------------------------------ tuple graphs in and out
/// The array names of a tuple graph, in TupleGraphArrays order.
constexpr std::string_view k_tuple_graph_arrays[] = {"distance_offsets", "tuple_offsets", "tuple_atoms", "problem_offsets",
                                                     "problem_vertices", "successor_offsets", "successors", "predecessor_offsets",
                                                     "predecessors", "layer_offsets", "layer_vertices"};

std::array<std::vector<u32>*, 11> fields(TupleGraphArrays& a)
{
    return {&a.distance_offsets, &a.tuple_offsets,   &a.tuple_atoms,         &a.problem_offsets,
            &a.problem_vertex_ids, &a.successor_offsets, &a.successor_ids, &a.predecessor_offsets,
            &a.predecessor_ids, &a.layer_offsets,     &a.layer_vertex_ids};
}

std::array<std::span<const u32>, 11> fields(const TupleGraph& g)
{
    return {g.distance_offsets(), g.tuple_offsets(),   g.tuple_atoms(),         g.problem_offsets(),
            g.problem_vertex_ids(), g.successor_offsets(), g.successor_ids(), g.predecessor_offsets(),
            g.predecessor_ids(), g.layer_offsets(),     g.layer_vertex_ids()};
}

/// The tuple graph of a vertex of `space` from the arrays of arrays().
TupleGraph tuple_graph_from_arrays(const StateSpacePtr& space, u32 root, u32 width, bool dominance_pruning, nb::handle arrays)
{
    const char* what = "tuple graph";
    check_keys(arrays,
               {"distance_offsets", "tuple_offsets", "tuple_atoms", "problem_offsets", "problem_vertices", "successor_offsets",
                "successors", "predecessor_offsets", "predecessors", "layer_offsets", "layer_vertices"},
               what);
    TupleGraphArrays a;
    a.space = space;
    a.root = root;
    a.width = width;
    a.dominance_pruning = dominance_pruning;
    const auto f = fields(a);
    for (usize i = 0; i < f.size(); ++i)
        *f[i] = take_required<u32>(arrays, k_tuple_graph_arrays[i].data(), 1, what);
    nb::gil_scoped_release release;
    return TupleGraph::create(std::move(a));
}

/// Tuple graphs of one state space packed into a few arrays (pickles and files of many graphs): "meta" [G, 3] (root,
/// width, dominance pruning), "sizes" [G, 11] (each graph's array lengths) and the 11 arrays of the graphs
/// concatenated in graph order.
nb::dict pack_tuple_graphs(std::span<const TupleGraph* const> graphs)
{
    const u64 G = graphs.size();
    std::vector<u32> meta(G * 3);
    std::vector<u64> sizes(G * 11), totals(11, 0);
    for (u64 i = 0; i < G; ++i)
    {
        const TupleGraph& g = *graphs[i];
        meta[i * 3] = g.root();
        meta[i * 3 + 1] = g.width();
        meta[i * 3 + 2] = g.dominance_pruning() ? 1 : 0;
        const auto f = fields(g);
        for (usize k = 0; k < 11; ++k)
            totals[k] += sizes[i * 11 + k] = f[k].size();
    }
    BlockLayout layout;
    const u64 at_meta = layout.add(meta.size(), sizeof(u32)), at_sizes = layout.add(sizes.size(), sizeof(u64));
    u64 at[11];
    for (usize k = 0; k < 11; ++k)
        at[k] = layout.add(totals[k], sizeof(u32));
    auto block = Block::make(std::max<u64>(layout.bytes, 1));
    auto* base = reinterpret_cast<unsigned char*>(block->data());
    std::memcpy(base + at_meta, meta.data(), meta.size() * sizeof(u32));
    std::memcpy(base + at_sizes, sizes.data(), sizes.size() * sizeof(u64));
    {
        nb::gil_scoped_release release;
        std::array<u64, 11> pos{};
        for (u64 i = 0; i < G; ++i)
        {
            const auto f = fields(*graphs[i]);
            for (usize k = 0; k < 11; ++k)
            {
                if (!f[k].empty())
                    std::memcpy(base + at[k] + pos[k] * sizeof(u32), f[k].data(), f[k].size_bytes());
                pos[k] += f[k].size();
            }
        }
    }
    nb::dict d;
    auto out = [&](u64 offset, rl::DType dt, std::vector<i64> shape)
    {
        ArraySpec spec{block, base + offset, dt, std::move(shape), {}, true, false};
        return export_array(std::move(spec), Framework::Numpy, default_words(Framework::Numpy));
    };
    d["meta"] = out(at_meta, rl::DType::U32, {static_cast<i64>(G), 3});
    d["sizes"] = out(at_sizes, rl::DType::U64, {static_cast<i64>(G), 11});
    for (usize k = 0; k < 11; ++k)
        d[nb::str(k_tuple_graph_arrays[k].data(), k_tuple_graph_arrays[k].size())] = out(at[k], rl::DType::U32, {static_cast<i64>(totals[k])});
    return d;
}

/// The tuple graphs of pack_tuple_graphs, of `space`.
std::vector<TupleGraph> unpack_tuple_graphs(const StateSpacePtr& space, nb::handle packed)
{
    const char* what = "packed tuple graph";
    check_keys(packed,
               {"meta", "sizes", "distance_offsets", "tuple_offsets", "tuple_atoms", "problem_offsets", "problem_vertices",
                "successor_offsets", "successors", "predecessor_offsets", "predecessors", "layer_offsets", "layer_vertices"},
               what);
    auto meta = take<u32>(packed, "meta", 2, true, what);
    auto sizes = take<u64>(packed, "sizes", 2, true, what);
    const u64 G = meta->second[0];
    if (meta->second[1] != 3 || sizes->second[0] != G || sizes->second[1] != 11)
        throw nb::value_error("mymyr: corrupt packed tuple graphs (meta [G, 3], sizes [G, 11])");
    std::array<std::vector<u32>, 11> all;
    for (usize k = 0; k < 11; ++k)
        all[k] = take_required<u32>(packed, k_tuple_graph_arrays[k].data(), 1, what);
    nb::gil_scoped_release release;
    std::vector<TupleGraph> out;
    out.reserve(G);
    std::array<u64, 11> pos{};
    for (u64 i = 0; i < G; ++i)
    {
        TupleGraphArrays a;
        a.space = space;
        a.root = meta->first[i * 3];
        a.width = meta->first[i * 3 + 1];
        a.dominance_pruning = meta->first[i * 3 + 2] != 0;
        const auto f = fields(a);
        for (usize k = 0; k < 11; ++k)
        {
            const u64 n = sizes->first[i * 11 + k];
            if (n > all[k].size() - pos[k])
                throw std::invalid_argument("mymyr: corrupt packed tuple graphs (sizes exceed the arrays)");
            f[k]->assign(all[k].begin() + static_cast<std::ptrdiff_t>(pos[k]), all[k].begin() + static_cast<std::ptrdiff_t>(pos[k] + n));
            pos[k] += n;
        }
        out.push_back(TupleGraph::create(std::move(a)));
    }
    for (usize k = 0; k < 11; ++k)
        if (pos[k] != all[k].size())
            throw std::invalid_argument("mymyr: corrupt packed tuple graphs (arrays longer than the sizes)");
    return out;
}

/// The keyword arguments of KnowledgeBase() but the tasks, as a tuple (its pickled and saved options), and back.
using KbOptions = nb::typed<nb::tuple, u32, std::optional<u64>, std::optional<double>, bool, bool, bool, bool, std::optional<u32>, bool>;

KnowledgeBaseOptions kb_options(u32 threads, nb::handle max_states, nb::handle max_seconds, bool remove_if_unsolvable, bool labels,
                                bool sort_by_size, bool generalized, std::optional<u32> width, bool dominance_pruning)
{
    KnowledgeBaseOptions o;
    o.state_space = make_options(1, max_states, max_seconds, remove_if_unsolvable, labels);
    o.sort_by_size = sort_by_size;
    o.generalized = generalized;
    if (width)
        o.tuple_graphs = TupleGraphOptions{.width = *width, .dominance_pruning = dominance_pruning, .threads = threads};
    o.threads = threads;
    return o;
}

using SpaceState = nb::typed<nb::tuple, PyTask, int, int, std::optional<ann::Any>, SpaceStats, ArrayDict>;

/// The pickled (and saved) state of a state space: (task, fingerprint, encoding fingerprint, lazy slots (None for
/// frozen slots), statistics, arrays with the rows).
SpaceState space_state(const PyStateSpace& s)
{
    const Task& T = *s.space->task();
    nb::object slots = T.atoms().frozen() ? nb::none() : fresh_u64(lazy_slots(T));
    return SpaceState(nb::make_tuple(task_object(s.owner.obj), T.fingerprint(), T.encoding_fingerprint(), slots,
                                     space_stats(*s.space), space_arrays(s, Framework::Numpy, true)));
}

PyStateSpace restore_space(nb::handle task, u64 fingerprint, u64 encoding_fingerprint, nb::handle slots, nb::handle stats,
                           nb::handle arrays)
{
    const Owner o = task_owner(task);
    adopt_task(*o.core->task, fingerprint, encoding_fingerprint, slots, "state space");
    return PyStateSpace{space_from_arrays(o.core->task, arrays, stats), o};
}

/// The keyword arguments of KnowledgeBase(), in order (KbOptions: those after the tasks).
#define MYMYR_KB_ARGS                                                                                                      \
    nb::kw_only(), "threads"_a = 0, "max_states"_a = nb::none(), "max_seconds"_a = nb::none(),                           \
        "remove_if_unsolvable"_a = true, "labels"_a = true, "sort_by_size"_a = true, "generalized"_a = false,           \
        "width"_a = nb::none(), "dominance_pruning"_a = true

using KbTasksArg = Arg<std::variant<PyTable, nb::typed<nb::sequence, std::variant<PyTask, PyHandle>>>>;

void init_knowledge_base(PyKnowledgeBase* self, nb::handle tasks, u32 threads, nb::handle max_states, nb::handle max_seconds,
                         bool remove_if_unsolvable, bool labels, bool sort_by_size, bool generalized, std::optional<u32> width,
                         bool dominance_pruning)
{
    nb::object table = nb::isinstance<PyTable>(tasks) ? nb::borrow(tasks) : nb::type<PyTable>()(tasks);
    const KnowledgeBaseOptions o =
        kb_options(threads, max_states, max_seconds, remove_if_unsolvable, labels, sort_by_size, generalized, width, dominance_pruning);
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
        p->spaces.push_back(nb::cast(PyStateSpace{kb->state_spaces()[i], owner_of(t.tasks[kb->task_indices()[i]])}));
    if (kb->generalized_state_space())
        p->gss = PyGeneralizedStateSpace{kb->generalized_state_space(), p->spaces};
    p->args = nb::make_tuple(threads, max_states, max_seconds, remove_if_unsolvable, labels, sort_by_size, generalized, width,
                             dominance_pruning);
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

using KbState = nb::typed<nb::tuple, PyTable, KbOptions, nb::typed<nb::list, PyStateSpace>, nb::typed<nb::list, int>,
                          std::optional<nb::typed<nb::list, ArrayDict>>>;

/// The pickled (and saved) state of a knowledge base: (tasks, options, state spaces, task indices, the tuple graphs of
/// each space packed (pack_tuple_graphs) or None without tuple graphs).
KbState kb_state(const PyKnowledgeBase& kb)
{
    nb::list spaces, indices;
    for (const nb::object& s : kb.spaces)
        spaces.append(s);
    for (const u32 i : kb.kb->task_indices())
        indices.append(i);
    nb::object packs = nb::none();
    if (kb.kb->has_tuple_graphs())
    {
        nb::list l;
        for (const auto& graphs : kb.kb->tuple_graphs())
        {
            std::vector<const TupleGraph*> ptrs;
            for (const TupleGraph& g : graphs)
                ptrs.push_back(&g);
            l.append(pack_tuple_graphs(ptrs));
        }
        packs = l;
    }
    return KbState(nb::make_tuple(kb.table, kb.args, spaces, indices, packs));
}

/// A knowledge base of `table` from kb_state's parts: validated (KnowledgeBase::create from parts), nothing generated.
PyKnowledgeBase restore_kb(nb::handle table, nb::handle options, nb::handle spaces, nb::handle indices, nb::handle packs)
{
    if (!nb::isinstance<PyTable>(table))
        throw nb::type_error("mymyr: a knowledge base's tasks must be a mymyr.rl.TaskTable");
    const PyTable& t = *nb::inst_ptr<PyTable>(table);
    const nb::tuple a = nb::cast<nb::tuple>(options);
    if (a.size() != 9)
        throw nb::value_error("mymyr: corrupt knowledge base options");
    const KnowledgeBaseOptions o = kb_options(nb::cast<u32>(a[0]), a[1], a[2], nb::cast<bool>(a[3]), nb::cast<bool>(a[4]),
                                              nb::cast<bool>(a[5]), nb::cast<bool>(a[6]), nb::cast<std::optional<u32>>(a[7]),
                                              nb::cast<bool>(a[8]));
    KnowledgeBaseParts parts;
    for (nb::handle s : nb::borrow<nb::sequence>(spaces))
    {
        if (!nb::isinstance<PyStateSpace>(s))
            throw nb::type_error("mymyr: a knowledge base's state spaces must be StateSpaces");
        parts.state_spaces.push_back(nb::inst_ptr<PyStateSpace>(s)->space);
    }
    for (nb::handle i : nb::borrow<nb::sequence>(indices))
        parts.task_indices.push_back(nb::cast<u32>(i));
    if (!packs.is_none())
    {
        const nb::sequence p = nb::borrow<nb::sequence>(packs);
        if (nb::len(p) != parts.state_spaces.size())
            throw nb::value_error("mymyr: corrupt knowledge base (tuple graphs are not one list per state space)");
        for (usize i = 0; i < parts.state_spaces.size(); ++i)
            parts.tuple_graphs.push_back(unpack_tuple_graphs(parts.state_spaces[i], p[i]));
    }
    KnowledgeBasePtr kb;
    {
        nb::gil_scoped_release release;
        kb = KnowledgeBase::create(t.table, o, std::move(parts));
    }
    PyKnowledgeBase out;
    out.kb = kb;
    out.table = nb::borrow(table);
    for (usize i = 0; i < kb->state_spaces().size(); ++i)
        out.spaces.push_back(nb::cast(PyStateSpace{kb->state_spaces()[i], owner_of(t.tasks[kb->task_indices()[i]])}));
    if (kb->generalized_state_space())
        out.gss = PyGeneralizedStateSpace{kb->generalized_state_space(), out.spaces};
    out.args = nb::borrow<nb::tuple>(options);
    return out;
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
    "labels (keep (schema, binding) per transition). State words are the same for every thread count (a task with lazy atom slots has none).";

#define MYMYR_SS_ARGS                                                                                                      \
    nb::kw_only(), "threads"_a = 1, "max_states"_a = nb::none(), "max_seconds"_a = nb::none(),                           \
        "remove_if_unsolvable"_a = true, "labels"_a = true
}  // namespace

void bind_datasets(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule(
        "_datasets", "State spaces, generalized state spaces, knowledge bases, tuple graphs and samplers (mymyr.datasets)");

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
            "Zero-copy read-only views: state_words [N, W + NN] (the task's word encoding; absent for a task with lazy "
            "atom slots, whose numbering depends on the generation's order), forward_offsets [N + 1], "
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
        .def_static(
            "from_arrays",
            [](TaskArg task, Arg<nb::typed<nb::mapping, std::string, ann::ArrayLike>> arrays) {
                const Owner o = task_owner(task);
                if (!o.core->task->atoms().frozen())
                    throw nb::value_error("mymyr: a state space of a task with lazy atom slots is not rebuilt from arrays "
                                          "(the slot numbering is the task object's history): pickle the StateSpace or "
                                          "use datasets.save / load");
                return PyStateSpace{space_from_arrays(o.core->task, arrays, nb::none()), o};
            },
            "task"_a, "arrays"_a,
            "The state space of `task` from its arrays: a mapping with the names, dtypes and shapes arrays() gives "
            "(NumPy, or DLPack arrays on the CPU; copied), state_words included (any framework's word layout). The "
            "arrays are validated: shapes and dtypes, unknown names, ranges, the reverse CSR against the forward one, "
            "flags against distances, and that the rows are states of the task (its row layout and atom slots, its "
            "initial state first, labels of its schemas and objects); ValueError or TypeError otherwise. A task with "
            "lazy atom slots is refused (pickle the space or use datasets.save / load). The generation statistics "
            "(threads, layers, seconds) of the result are 1, 0, 0, 0. Linear in the size of the arrays.")
        .def("__reduce__",
             [](const PyStateSpace& s) {
                 return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_state_space"), space_state(s));
             })
        .def("__repr__", [](const PyStateSpace& s) {
            return "StateSpace(states=" + std::to_string(s.space->num_states()) + ", transitions=" +
                   std::to_string(s.space->num_transitions()) + ", goal=" + std::to_string(s.space->num_goal_states()) +
                   ", unsolvable=" + std::to_string(s.space->num_unsolvable_states()) + ")";
        });

    nb::class_<PyGenerationResult>(m, "GenerationResult")
        .def_prop_ro("status", [](const PyGenerationResult& r) { return r.status; })
        .def_prop_ro("space", [](const PyGenerationResult& r) { return nb::typed<nb::object, std::optional<PyStateSpace>>(r.space); },
                     "The state space (None unless status is OK).")
        .def_prop_ro("states", [](const PyGenerationResult& r) { return r.states; },
                     "States stored when the generation ended (also when it failed).")
        .def_prop_ro("seconds", [](const PyGenerationResult& r) { return r.seconds; })
        .def("__bool__", [](const PyGenerationResult& r) { return !r.space.is_none(); })
        .def("__reduce__",
             [](const PyGenerationResult& r) {
                 return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_generation_result"),
                                       nb::make_tuple(static_cast<int>(r.status), r.space, r.states, r.seconds));
             })
        .def("__repr__", [](const PyGenerationResult& r) {
            return std::string("GenerationResult(status=") + to_string(r.status) + ", states=" + std::to_string(r.states) + ")";
        });

    m.def(
        "generate",
        [](TaskArg task, IntArg threads_in, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool labels) {
            const u32 threads = threads_arg(threads_in);
            const Owner o = task_owner(task);
            const StateSpaceOptions opts = make_options(threads, max_states, max_seconds, remove_if_unsolvable, labels);
            return wrap(o, run_generation(o.core->task, opts));
        },
        "task"_a, MYMYR_SS_ARGS, (std::string("Generates the state space of a task: a GenerationResult. ") + k_options_doc).c_str());

    m.def(
        "state_space",
        [](TaskArg task, IntArg threads_in, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool labels) -> nb::typed<nb::object, std::optional<PyStateSpace>> {
            const u32 threads = threads_arg(threads_in);
            const Owner o = task_owner(task);
            const StateSpaceOptions opts = make_options(threads, max_states, max_seconds, remove_if_unsolvable, labels);
            return nb::typed<nb::object, std::optional<PyStateSpace>>(wrap(o, run_generation(o.core->task, opts)).space);
        },
        "task"_a, MYMYR_SS_ARGS,
        (std::string("The state space of a task, or None if the generation failed (as in mimir's StateSpace.create). ") +
         k_options_doc)
            .c_str());

    m.def(
        "generate_many",
        [](TasksArg tasks, IntArg threads_in, IntArg max_states, FloatArg max_seconds, bool remove_if_unsolvable,
           bool labels) {
            const u32 threads = threads_arg(threads_in);
            const std::vector<Owner> owners = owners_of(tasks);
            const StateSpaceOptions opts = make_options(1, max_states, max_seconds, remove_if_unsolvable, labels);
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
        "remove_if_unsolvable"_a = true, "labels"_a = true,
        "The instance pool: the state spaces of many tasks, one task per worker thread (threads workers; 0: all "
        "cores), each generated single-threaded. Results in input order. Options as generate().");

    // ---------------------------------------------------------------------------------------------- generalized
    nb::class_<PyGeneralizedStateSpace>(
        m, "GeneralizedStateSpace",
        "One graph over the state spaces of several problems of one domain, as mimir's GeneralizedStateSpace "
        "(datasets/generalized_state_space.hpp): their disjoint union in input order. State s of problem p is vertex "
        "vertex_offsets[p] + s and transition e of problem p is edge edge_offsets[p] + e, so the edges out of a vertex "
        "are its state's transitions in their order. The initial, goal and unsolvable vertices are the problems' ones.")
        .def(
            "__init__",
            [](PyGeneralizedStateSpace* self, nb::typed<nb::sequence, PyStateSpace> spaces) {
                std::vector<nb::object> in;
                std::vector<StateSpacePtr> sp;
                for (nb::handle h : spaces)
                {
                    if (!nb::isinstance<PyStateSpace>(h))
                        throw nb::type_error("mymyr: spaces must be StateSpaces");
                    in.push_back(nb::borrow(h));
                    sp.push_back(nb::inst_ptr<PyStateSpace>(h)->space);
                }
                GeneralizedStateSpacePtr g;
                {
                    nb::gil_scoped_release release;  // std::invalid_argument (other domains) becomes a ValueError
                    g = GeneralizedStateSpace::create(std::move(sp));
                }
                new (self) PyGeneralizedStateSpace{g, std::move(in)};
            },
            "spaces"_a, "From state spaces in their order (see sorted_by_size / generalized_state_space for mimir's order).")
        .def_prop_ro(
            "spaces",
            [](const PyGeneralizedStateSpace& g) {
                nb::typed<nb::list, PyStateSpace> out{nb::list()};
                for (const nb::object& s : g.spaces)
                    out.append(s);
                return out;
            },
            "The problems' state spaces (problem p = spaces[p]).")
        .def_prop_ro("num_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->num_vertices(); })
        .def_prop_ro("num_edges", [](const PyGeneralizedStateSpace& g) { return g.gss->num_edges(); })
        .def(
            "arrays",
            [](const PyGeneralizedStateSpace& g, FrameworkArg framework) {
                const Framework fw = parse_framework(framework);
                const std::shared_ptr<const void> own = g.gss;
                const GeneralizedStateSpace& G = *g.gss;
                ArrayDict d{nb::dict()};
                d["vertex_offsets"] = view<u32>(own, G.vertex_offsets(), fw);
                d["edge_offsets"] = view<u64>(own, G.edge_offsets(), fw);
                d["forward_offsets"] = view<u64>(own, G.forward_offsets(), fw);
                d["forward_targets"] = view<u32>(own, G.forward_targets(), fw);
                d["initial"] = view<u8>(own, G.initial_flags(), fw);
                d["goal"] = view<u8>(own, G.goal_flags(), fw);
                d["unsolvable"] = view<u8>(own, G.unsolvable_flags(), fw);
                return d;
            },
            "framework"_a = nb::none(),
            "Zero-copy views of the graph: per problem the first vertex and edge (vertex_offsets, edge_offsets, each "
            "with a final entry: the total), the forward CSR (forward_offsets, forward_targets) and per vertex the "
            "flags (initial, goal, unsolvable).")
        .def(
            "vertex",
            [](const PyGeneralizedStateSpace& g, i64 problem, i64 state) {
                const u32 p = static_cast<u32>(gss_index(problem, g.spaces.size(), "problem index"));
                return g.gss->vertex(p, static_cast<u32>(gss_index(state, g.gss->spaces()[p]->num_states(), "state id")));
            },
            "problem"_a, "state"_a, "The vertex of a state of a problem.")
        .def(
            "edge",
            [](const PyGeneralizedStateSpace& g, i64 problem, i64 edge) {
                const u32 p = static_cast<u32>(gss_index(problem, g.spaces.size(), "problem index"));
                return g.gss->edge(p, gss_index(edge, g.gss->spaces()[p]->num_transitions(), "transition index"));
            },
            "problem"_a, "edge"_a, "The edge of a transition of a problem.")
        .def(
            "problem_of",
            [](const PyGeneralizedStateSpace& g, i64 vertex) {
                const u32 v = static_cast<u32>(gss_index(vertex, g.gss->num_vertices(), "vertex"));
                const u32 p = g.gss->problem_of(v);
                return std::pair<u32, u32>{p, v - g.gss->vertex_offsets()[p]};
            },
            "vertex"_a, "The (problem, state id) of a vertex.")
        .def(
            "source", [](const PyGeneralizedStateSpace& g, i64 e) { return g.gss->source(gss_index(e, g.gss->num_edges(), "edge")); },
            "edge"_a)
        .def(
            "target",
            [](const PyGeneralizedStateSpace& g, i64 e) { return g.gss->forward_targets()[gss_index(e, g.gss->num_edges(), "edge")]; },
            "edge"_a)
        .def("__reduce__",
             [](nb::handle self) {
                 nb::list spaces;
                 for (const nb::object& s : nb::inst_ptr<PyGeneralizedStateSpace>(self)->spaces)
                     spaces.append(s);
                 return nb::make_tuple(self.type(), nb::make_tuple(spaces));
             })
        .def("initial_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->initial_vertices(); })
        .def("goal_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->goal_vertices(); })
        .def("unsolvable_vertices", [](const PyGeneralizedStateSpace& g) { return g.gss->unsolvable_vertices(); })
        .def("__repr__", [](const PyGeneralizedStateSpace& g) {
            return "GeneralizedStateSpace(problems=" + std::to_string(g.spaces.size()) + ", vertices=" +
                   std::to_string(g.gss->num_vertices()) + ", edges=" + std::to_string(g.gss->num_edges()) + ")";
        });

    m.def(
        "sorted_by_size",
        [](nb::typed<nb::sequence, std::optional<PyStateSpace>> spaces) {
            std::vector<std::pair<u32, nb::object>> in;
            for (nb::handle h : spaces)
                if (!h.is_none())
                    in.emplace_back(space_arg(h).space->num_states(), nb::borrow(h));
            std::stable_sort(in.begin(), in.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            nb::typed<nb::list, PyStateSpace> out{nb::list()};
            for (const auto& [n, s] : in)
                out.append(s);
            return out;
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
            [](PySampler* self, SamplerSpace space, IntArg seed_in) {
                const u64 seed = int_arg<u64>(seed_in, "seed");
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
        .def("set_seed", [](PySampler& s, IntArg seed_in) {
            const u64 seed = int_arg<u64>(seed_in, "seed");
            std::lock_guard lock(s.core->mutex);
            s.core->sampler.set_seed(seed);
        }, "seed"_a)
        .def("sample_state", [](PySampler& s) {
            std::lock_guard lock(s.core->mutex);
            return s.core->sampler.sample_state();
        })
        .def("sample_state_n_steps_from_goal", [](PySampler& s, IntArg n_in) {
            const i32 n = int_arg<i32>(n_in, "n", 0);
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
            [](PySampler& s, IntArg count_in, FrameworkArg framework) {
                const u64 count = int_arg<u64>(count_in, "count");
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
            [](PySampler& s, IntArg n_in, IntArg count_in, FrameworkArg framework) {
                const i32 n = int_arg<i32>(n_in, "n", 0);
                const u64 count = int_arg<u64>(count_in, "count");
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
            [](PySampler& s, IntArg count_in, FrameworkArg framework) {
                const u64 count = int_arg<u64>(count_in, "count");
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
            [](const PySampler& s, IntArg n_in, FrameworkArg framework) {
                const i32 n = int_arg<i32>(n_in, "n", 0);
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

    // ---------------------------------------------------------------------------------------------- tuple graphs
    nb::class_<PyTupleGraph>(
        m, "TupleGraph",
        "The tuple graph of a state-space vertex (the root), as in mimir's TupleGraph (datasets/tuple_graph.hpp; "
        "Lipovetzky and Geffner 2012): which tuples of at most `width` fluent atoms are first reached at each "
        "breadth-first distance from the root, the problem vertices (states) at that distance in which each is novel, "
        "and the edges u -> t between consecutive distances where every problem vertex of u has a successor among "
        "those of t. Width 0: the root and one vertex per successor state. Vertices are ordered by distance; "
        "dominance pruning (default) keeps the vertices with minimal problem-vertex sets. The result does not depend "
        "on the thread count.")
        .def_prop_ro("space", [](const PyTupleGraph& g) { return nb::typed<nb::object, PyStateSpace>(g.space); },
                     "The state space the root belongs to.")
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
                const Owner& o = g.py_space().owner;
                const Task& task = *o.core->task;
                nb::typed<nb::list, PyGroundAtom> out{nb::list()};
                std::vector<u32> slots = to_vector(g.graph->tuple(tuple_vertex(*g.graph, v)));
                if (!task.atoms().frozen())
                    std::ranges::sort(slots, {}, [&](u32 s) { return task.atoms().canonical(mymyr::AtomKind::Fluent, s); });
                for (const u32 s : slots)
                {
                    const SlotId slot{s};
                    const auto args = task.atoms().arguments(slot);
                    out.append(make_ground_atom(mymyr::python::task_owner(o),
                                                GroundAtom{task.atoms().predicate(slot), std::vector<ObjectId>(args.begin(), args.end())}));
                }
                return out;
            },
            "vertex"_a, "The vertex's tuple as GroundAtoms, in the task's canonical atom order (the same under every atom mode).")
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
        .def_static(
            "from_arrays",
            [](Arg<PyStateSpace> space, IntArg root, Arg<nb::typed<nb::mapping, std::string, ann::ArrayLike>> arrays,
               IntArg width, bool dominance_pruning) {
                const u32 r = int_arg<u32>(root, "root"), w = int_arg<u32>(width, "width");
                return PyTupleGraph{
                    std::make_shared<const TupleGraph>(tuple_graph_from_arrays(space_arg(space).space, r, w, dominance_pruning, arrays)),
                    space};
            },
            "space"_a, "root"_a, "arrays"_a, nb::kw_only(), "width"_a, "dominance_pruning"_a = true,
            "The tuple graph of vertex `root` of `space` from its arrays (the uint32 arrays arrays() gives; copied) "
            "and the options it was built with. Validated: the CSR structure, ascending rows, tuples of at most "
            "`width` atom slots of the task, problem vertices in their breadth-first layer, layer 0 the root, "
            "successors at the next distance and predecessors their reverse; ValueError or TypeError otherwise. "
            "Equal to the graph the arrays came from if that one is of `space`.")
        .def("__reduce__",
             [](const PyTupleGraph& g) {
                 const std::shared_ptr<const void> own = g.graph;
                 nb::dict d;
                 const auto f = fields(*g.graph);
                 for (usize k = 0; k < f.size(); ++k)
                     d[nb::str(k_tuple_graph_arrays[k].data(), k_tuple_graph_arrays[k].size())] = view<u32>(own, f[k], Framework::Numpy);
                 return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_tuple_graph"),
                                       nb::make_tuple(g.space, g.graph->root(), g.graph->width(), g.graph->dominance_pruning(), d));
             })
        .def("__eq__", [](const PyTupleGraph& a, const PyTupleGraph& b) { return *a.graph == *b.graph; }, nb::is_operator(),
             "other"_a, "The same state space, root and options, and equal vertices, tuples, problem vertices and edges.")
        .def("__repr__", [](const PyTupleGraph& g) {
            return "TupleGraph(root=" + std::to_string(g.graph->root()) + ", width=" + std::to_string(g.graph->width()) +
                   ", vertices=" + std::to_string(g.graph->num_vertices()) + ", edges=" + std::to_string(g.graph->num_edges()) +
                   ", distances=" + std::to_string(g.graph->num_distances()) + ")";
        });

    m.def(
        "tuple_graphs",
        [](Arg<PyStateSpace> space_obj, IntArg width_in, bool dominance_pruning, IntArg threads_in) {
            const PyStateSpace& space = space_arg(space_obj);
            const u32 width = int_arg<u32>(width_in, "width");
            const u32 threads = threads_arg(threads_in);
            const TupleGraphOptions o{.width = width, .dominance_pruning = dominance_pruning, .threads = threads};
            std::vector<TupleGraph> graphs;
            {
                nb::gil_scoped_release release;  // std::invalid_argument (a width above 5) becomes a ValueError
                graphs = datasets::tuple_graphs(space.space, o);
            }
            nb::typed<nb::list, PyTupleGraph> out{nb::list()};
            for (TupleGraph& g : graphs)
                out.append(nb::cast(PyTupleGraph{std::make_shared<const TupleGraph>(std::move(g)), space_obj}, nb::rv_policy::move));
            return out;
        },
        "space"_a, nb::kw_only(), "width"_a = 0, "dominance_pruning"_a = true, "threads"_a = 0,
        "The tuple graph of every vertex of a state space (index = state id), as mimir's TupleGraphFactory: width "
        "0..5 (0: the root and its successor states), dominance_pruning, threads (0: all cores; the result is the "
        "same at every count). Cost: a breadth-first search per vertex, enumerating the tuples of at most width atoms "
        "of every state it reaches; tuple_graph() builds one.");

    m.def(
        "tuple_graph",
        [](Arg<PyStateSpace> space_obj, i64 vertex, IntArg width_in, bool dominance_pruning) {
            const PyStateSpace& space = space_arg(space_obj);
            const u32 width = int_arg<u32>(width_in, "width");
            if (vertex < 0 || vertex >= static_cast<i64>(space.space->num_states()))
                throw nb::index_error("mymyr: state id out of range");
            const TupleGraphOptions o{.width = width, .dominance_pruning = dominance_pruning, .threads = 1};
            std::shared_ptr<const TupleGraph> g;
            {
                nb::gil_scoped_release release;
                g = std::make_shared<const TupleGraph>(datasets::tuple_graph(space.space, static_cast<u32>(vertex), o));
            }
            return PyTupleGraph{std::move(g), space_obj};
        },
        "space"_a, "vertex"_a, nb::kw_only(), "width"_a = 0, "dominance_pruning"_a = true,
        "The tuple graph of one vertex of a state space (equal to tuple_graphs(space)[vertex]).");

    // ---------------------------------------------------------------------------------------------- knowledge bases
    nb::class_<PyKnowledgeBase>(
        m, "KnowledgeBase",
        "What is known about a set of tasks of one domain, as mimir's KnowledgeBase (datasets/knowledge_base.hpp): the "
        "tasks (a mymyr.rl.TaskTable), the state space of every task whose generation succeeded (failures skipped; "
        "max_states and max_seconds bound each one), sorted ascending by size unless sort_by_size=False (ties in task "
        "order), optionally the generalized state space over them (generalized=True), and optionally the tuple graphs "
        "of every vertex of every space (width=0..5; tuple graphs of large spaces are better built per vertex with "
        "tuple_graph(space, vertex)). Every step runs on `threads` threads (0: all cores) with the same result at "
        "every count. A knowledge base pickles as its tasks, options, state spaces and tuple graphs (unpickling "
        "validates them and rebuilds the generalized state space; nothing is generated again), and saves to a file "
        "with datasets.save.")
        .def(
            "__init__",
            [](PyKnowledgeBase* self, KbTasksArg tasks, IntArg threads_in, IntArg max_states, FloatArg max_seconds,
               bool remove_if_unsolvable, bool labels, bool sort_by_size, bool generalized, IntArg width_in,
               bool dominance_pruning) {
                const u32 threads = threads_arg(threads_in);
                const std::optional<u32> width = opt_int_arg<u32>(width_in, "width");
                init_knowledge_base(self, tasks, threads, max_states, max_seconds, remove_if_unsolvable, labels, sort_by_size,
                                    generalized, width, dominance_pruning);
            },
            "tasks"_a, MYMYR_KB_ARGS,
            "The knowledge base of tasks of one domain: a TaskTable or a sequence of Tasks (made into a TaskTable). The "
            "state space options are those of generate().")
        .def_prop_ro("tasks", [](const PyKnowledgeBase& kb) { return nb::typed<nb::object, PyTable>(kb.table); },
                     "The TaskTable of the tasks.")
        .def_prop_ro("state_spaces",
                     [](const PyKnowledgeBase& kb) {
                         nb::typed<nb::list, PyStateSpace> out{nb::list()};
                         for (const nb::object& s : kb.spaces)
                             out.append(s);
                         return out;
                     },
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
        .def("__reduce__",
             [](const PyKnowledgeBase& kb) {
                 return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_knowledge_base"), kb_state(kb));
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

    // ---------------------------------------------------------------------------------------------- persistence
    // The reconstructors of the pickles (named by mymyr._pickle) and the parts of mymyr.datasets.save / load.
    m.def("_state_space_state", &space_state, "space"_a,
          "(task, fingerprint, encoding_fingerprint, lazy slots or None, statistics, arrays with state_words): what a "
          "StateSpace pickles and saves.");
    m.def(
        "_restore_state_space",
        [](TaskArg task, u64 fingerprint, u64 encoding_fingerprint, std::optional<Arg<ann::ArrayLike>> slots, Arg<SpaceStats> stats,
           ArraysArg arrays) {
            return restore_space(task, fingerprint, encoding_fingerprint, slots ? nb::handle(*slots) : nb::none(), stats,
                                 arrays);
        },
        "task"_a, "fingerprint"_a, "encoding_fingerprint"_a, "slots"_a.none(), "stats"_a, "arrays"_a,
        "The state space of _state_space_state's parts, of `task` (checked against the fingerprints; a task with lazy "
        "atom slots takes the saved numbering).");
    m.def(
        "_restore_tuple_graph",
        [](Arg<PyStateSpace> space, u32 root, u32 width, bool dominance_pruning, ArraysArg arrays) {
            return PyTupleGraph{
                std::make_shared<const TupleGraph>(tuple_graph_from_arrays(space_arg(space).space, root, width, dominance_pruning, arrays)),
                space};
        },
        "space"_a, "root"_a, "width"_a, "dominance_pruning"_a, "arrays"_a);
    m.def(
        "_pack_tuple_graphs",
        [](nb::typed<nb::sequence, PyTupleGraph> graphs) {
            std::vector<const TupleGraph*> gs;
            const StateSpace* space = nullptr;
            for (nb::handle h : graphs)
            {
                if (!nb::isinstance<PyTupleGraph>(h))
                    throw nb::type_error("mymyr: expected TupleGraphs");
                const PyTupleGraph& g = *nb::inst_ptr<PyTupleGraph>(h);
                if (space && g.graph->space().get() != space)
                    throw nb::value_error("mymyr: the tuple graphs are of different state spaces");
                space = g.graph->space().get();
                gs.push_back(g.graph.get());
            }
            return ArrayDict(pack_tuple_graphs(gs));
        },
        "graphs"_a, "Tuple graphs of one state space packed into arrays: meta [G, 3] (root, width, dominance_pruning), "
        "sizes [G, 11] and the graphs' arrays concatenated.");
    m.def(
        "_unpack_tuple_graphs",
        [](Arg<PyStateSpace> space, ArraysArg packed) {
            std::vector<TupleGraph> gs = unpack_tuple_graphs(space_arg(space).space, packed);
            nb::typed<nb::list, PyTupleGraph> out{nb::list()};
            for (TupleGraph& g : gs)
                out.append(nb::cast(PyTupleGraph{std::make_shared<const TupleGraph>(std::move(g)), space}, nb::rv_policy::move));
            return out;
        },
        "space"_a, "packed"_a);
    m.def("_knowledge_base_state", &kb_state, "kb"_a,
          "(tasks, options, state spaces, task indices, packed tuple graphs per space or None): what a KnowledgeBase "
          "pickles and saves.");
    m.def(
        "_restore_knowledge_base",
        [](Arg<PyTable> table, Arg<KbOptions> options, Arg<nb::typed<nb::sequence, PyStateSpace>> spaces,
           Arg<nb::typed<nb::sequence, int>> indices, std::optional<Arg<nb::typed<nb::sequence, ArraysArg>>> packs) {
            return restore_kb(table, options, spaces, indices, packs ? nb::handle(*packs) : nb::none());
        },
        "tasks"_a, "options"_a, "spaces"_a, "task_indices"_a, "tuple_graphs"_a.none());
    m.def(
        "_restore_generation_result",
        [](int status, std::optional<Arg<PyStateSpace>> space_arg, u64 states, f64 seconds) {
            const nb::handle space = space_arg ? nb::handle(*space_arg) : nb::none();
            if (status < 0 || status > static_cast<int>(StateSpaceStatus::Unsolvable) ||
                space.is_none() == (status == static_cast<int>(StateSpaceStatus::Ok)) ||
                (!space.is_none() && !nb::isinstance<PyStateSpace>(space)))
                throw nb::value_error("mymyr: corrupt generation result");
            return PyGenerationResult{static_cast<StateSpaceStatus>(status), nb::borrow(space), states, seconds};
        },
        "status"_a, "space"_a.none(), "states"_a, "seconds"_a);
}
#undef MYMYR_KB_ARGS
#undef MYMYR_SS_ARGS
}  // namespace mymyr::python
