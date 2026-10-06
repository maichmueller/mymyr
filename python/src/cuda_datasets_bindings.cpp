// mymyr.cuda: device state spaces (cuda/state_space.hpp): generate_state_space / state_space for one task,
// generate_state_spaces / state_spaces for every instance of a task table at once (waves), and their result,
// DeviceStateSpace: the CPU generator's arrays (mymyr.datasets.state_space) on the device, exported zero-copy
// (arrays()), with to_host() giving the mymyr.datasets.StateSpace. Registered from module.cpp after bind_cuda
// (the submodule _cuda exists then); the default
// contexts are the tables' and tasks' of cuda_bindings.cpp (table_device_context). StateSpaceSampler takes a
// DeviceStateSpace through the hook of py_datasets.hpp.
// Exports follow the array API's stream semantics (cuda_bindings.cpp): the arrays are written on the space's stream;
// a consumer's __dlpack__(stream=s) makes s wait for that work and keeps the memory until the work enqueued on s so far
// has completed (DeviceStateSpace::use_on).

#include "arrays.hpp"
#include "dlpack.hpp"
#include "py_datasets.hpp"
#include "py_table.hpp"
#include "py_task.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/state_space.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mymyr::python::ann
{
/// A mymyr.cuda.Context (bound in cuda_bindings.cpp).
struct CudaContext
{
};
/// The output of a device generation.
struct SpaceOutput
{
};
}  // namespace mymyr::python::ann

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::CudaContext>
{
    static constexpr auto Name = const_name("mymyr._core._cuda.Context");
};
template<>
struct type_caster<mymyr::python::ann::SpaceOutput>
{
    static constexpr auto Name = const_name("typing.Literal['device', 'host', 'both']");
};
}  // namespace nanobind::detail

namespace mymyr::python
{
using namespace nb::literals;

/// The device context of an operation on a table (cuda_bindings.cpp): `ctx`, or the table's default context on `device`.
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device);

void bind_cuda_datasets(nb::module_& parent);

namespace
{
using datasets::StateSpaceStatus;

// ------------------------------------------------------------------------------------------------ argument types
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using ContextArg = Arg<ann::CudaContext>;
using OutputArg = Arg<ann::SpaceOutput>;
using IntArg = Arg<u64>;
using FloatArg = Arg<double>;
using DeviceArg = Arg<int>;
using ArrayDict = nb::typed<nb::dict, std::string, ann::Any>;
using StatsDict = nb::typed<nb::dict, std::string, std::variant<double, u64>>;

// ------------------------------------------------------------------------------------------------ wrappers
struct PyDeviceStateSpace
{
    cuda::DeviceStateSpacePtr d;
    Owner owner;  // the Task (or handle) of the instance: States and Actions are owned by it
};

struct PyDeviceGeneration
{
    StateSpaceStatus status = StateSpaceStatus::Ok;
    std::optional<PyDeviceStateSpace> space;
    std::optional<PyStateSpace> host;
    u64 states = 0;
    f64 seconds = 0;
    cuda::DeviceStateSpaceStats stats;
};

// ------------------------------------------------------------------------------------------------ exports

cudaStream_t cuda_stream(std::intptr_t v)
{
    if (v == dl::k_stream_legacy)
        return cudaStreamLegacy;
    if (v == dl::k_stream_per_thread)
        return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(v);
}

/// The handoff of a space's arrays to a consumer stream.
class SpaceSync final : public DeviceSync
{
public:
    explicit SpaceSync(cuda::DeviceStateSpacePtr d) : m_d(std::move(d)) {}
    void handoff(std::intptr_t consumer) override { m_d->use_on(cuda_stream(consumer)); }

private:
    cuda::DeviceStateSpacePtr m_d;
};

ArrayDict device_arrays(const PyDeviceStateSpace& x, nb::handle framework)
{
    const Framework fw = framework.is_none() ? Framework::DLPack : parse_framework(framework, nb::none());
    const cuda::DeviceStateSpace& D = *x.d;
    const auto sync = std::make_shared<SpaceSync>(x.d);
    const int device = D.context()->device();
    const i64 N = D.num_states(), E = static_cast<i64>(D.num_transitions());
    auto out = [&](const void* p, rl::DType dtype, std::vector<i64> shape, bool words = false)
    {
        ArraySpec s;
        if (!p)
            p = D.forward_offsets();  // an empty array (label_width 0): any device address
        s.owner = std::shared_ptr<const void>(x.d, p);
        s.data = p;
        s.dtype = dtype;
        s.shape = std::move(shape);
        s.readonly = true;
        s.words = words;
        return export_device_array(std::move(s), DeviceExport{dl::k_cuda, device, sync}, fw, default_words(fw));
    };
    ArrayDict d{nb::dict()};
    d["state_words"] = out(D.state_words(), rl::DType::U64, {N, static_cast<i64>(D.words())}, true);
    d["forward_offsets"] = out(D.forward_offsets(), rl::DType::U64, {N + 1});
    d["forward_targets"] = out(D.forward_targets(), rl::DType::U32, {E});
    if (D.has_labels())
    {
        d["label_schemas"] = out(D.label_schemas(), rl::DType::U32, {E});
        d["label_bindings"] = out(D.label_bindings(), rl::DType::U32, {E, static_cast<i64>(D.label_width())});
    }
    d["backward_offsets"] = out(D.backward_offsets(), rl::DType::U64, {N + 1});
    d["backward_sources"] = out(D.backward_sources(), rl::DType::U32, {E});
    d["backward_edges"] = out(D.backward_edges(), rl::DType::U32, {E});
    d["unit_goal_distances"] = out(D.unit_goal_distances(), rl::DType::I32, {N});
    d["cost_goal_distances"] = out(D.cost_goal_distances(), rl::DType::F64, {N});
    if (!D.unit_costs())
        d["costs"] = out(D.costs(), rl::DType::F64, {E});
    d["goal"] = out(D.goal_flags(), rl::DType::Bool, {N});
    d["unsolvable"] = out(D.unsolvable_flags(), rl::DType::Bool, {N});
    d["alive"] = out(D.alive_flags(), rl::DType::Bool, {N});
    return d;
}

/// The bytes of a space's arrays.
u64 space_bytes(const cuda::DeviceStateSpace& D)
{
    const u64 N = D.num_states(), E = D.num_transitions();
    u64 b = N * D.words() * 8 + (N + 1) * 16 + E * 12 + N * (4 + 8 + 3);
    if (D.has_labels())
        b += E * 4 * (1 + u64{D.label_width()});
    if (!D.unit_costs())
        b += E * 8;
    return b;
}

/// Copies `n` elements at device address `src` to the host after the space's work (synchronizes its stream).
template<class T>
std::vector<T> download(const cuda::DeviceStateSpace& D, const T* src, u64 n)
{
    std::vector<T> v(n);
    if (!n)
        return v;
    const cuda::DeviceGuard guard(D.context()->device());
    cuda::check(cudaMemcpyAsync(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost, D.stream()), "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(D.stream()), "cudaStreamSynchronize");
    return v;
}

bool device_distances(nb::handle obj, std::vector<i32>& out)
{
    if (!nb::isinstance<PyDeviceStateSpace>(obj))
        return false;
    const cuda::DeviceStateSpace& D = *nb::inst_ptr<PyDeviceStateSpace>(obj)->d;
    nb::gil_scoped_release release;
    out = download(D, D.unit_goal_distances(), D.num_states());
    return true;
}

u32 state_id(const PyDeviceStateSpace& x, i64 id)
{
    if (id < 0 || id >= static_cast<i64>(x.d->num_states()))
        throw nb::index_error("mymyr: state id out of range");
    return static_cast<u32>(id);
}

// ------------------------------------------------------------------------------------------------ options

cuda::StateSpaceOutput parse_output(nb::handle v)
{
    if (!nb::isinstance<nb::str>(v))
        throw nb::type_error("mymyr: output must be 'device', 'host' or 'both'");
    const std::string s = nb::cast<std::string>(v);
    if (s == "device")
        return cuda::StateSpaceOutput::Device;
    if (s == "host")
        return cuda::StateSpaceOutput::Host;
    if (s == "both")
        return cuda::StateSpaceOutput::Both;
    throw nb::value_error("mymyr: output must be 'device', 'host' or 'both'");
}

struct Common
{
    nb::handle ctx, device, stream;
    u32 threads = 0;
    nb::handle max_states, max_seconds;
    bool remove_if_unsolvable = true, labels = true;
    nb::handle chunk_states, view_bytes, expected_states;
};

cudaStream_t stream_value(nb::handle stream)
{
    if (stream.is_none())
        return nullptr;
    nb::object h = nb::getattr(stream, "cuda_stream", nb::none());
    const std::intptr_t v = nb::cast<std::intptr_t>(h.is_none() ? nb::borrow(stream) : h);
    if (v == dl::k_stream_none)
        throw nb::value_error("mymyr: stream -1 means 'no synchronization'; pass a stream to run on");
    if (v == 0)
        return cudaStreamLegacy;  // torch reports its default stream as 0
    return cuda_stream(v);
}

cuda::DeviceStateSpaceOptions options_of(const Common& a)
{
    cuda::DeviceStateSpaceOptions o;
    o.space.threads = a.threads;
    if (!a.max_states.is_none())
        o.space.max_states = nb::cast<u64>(a.max_states);
    if (!a.max_seconds.is_none())
    {
        const double s = nb::cast<double>(a.max_seconds);
        if (!(s >= 0) || std::isnan(s))
            throw nb::value_error("mymyr: max_seconds must be non-negative");
        o.space.max_seconds = s;
    }
    o.space.remove_if_unsolvable = a.remove_if_unsolvable;
    o.space.labels = a.labels;
    if (!a.chunk_states.is_none())
    {
        o.chunk_states = nb::cast<u32>(a.chunk_states);
        if (o.chunk_states == 0)
            throw nb::value_error("mymyr: chunk_states must be positive");
    }
    if (!a.view_bytes.is_none())
        o.view_bytes = nb::cast<u64>(a.view_bytes);
    if (!a.expected_states.is_none())
        o.expected_states = nb::cast<u64>(a.expected_states);
    o.stream = stream_value(a.stream);
    return o;
}

cuda::ContextPtr context_for(nb::handle table, const Common& a)
{
    const int device = a.device.is_none() ? 0 : nb::cast<int>(a.device);
    cuda::ContextPtr c = table_device_context(table, a.ctx, device);
    if (!a.device.is_none() && c->device() != device)
        throw nb::value_error(("mymyr: device " + std::to_string(device) + " differs from the context's (cuda:" +
                               std::to_string(c->device()) + ")")
                                  .c_str());
    return c;
}

StatsDict stats_dict(const cuda::DeviceStateSpaceStats& s)
{
    StatsDict d{nb::dict()};
    d["waves"] = u64{s.waves};
    d["chunks"] = s.chunks;
    d["multi"] = u64{s.multi ? 1u : 0u};
    d["states"] = s.states;
    d["transitions"] = s.transitions;
    d["generate_ms"] = s.generate_ms;
    d["post_ms"] = s.post_ms;
    d["output_ms"] = s.output_ms;
    d["host_cost_ms"] = s.host_cost_ms;
    d["rehashes"] = u64{s.rehashes};
    d["widenings"] = u64{s.widenings};
    d["device_bytes"] = s.device_bytes;
    return d;
}

PyDeviceGeneration wrap(const Owner& o, cuda::DeviceStateSpaceResult&& r, const cuda::DeviceStateSpaceStats& stats)
{
    PyDeviceGeneration g;
    g.status = r.status;
    g.states = r.states;
    g.seconds = r.seconds;
    g.stats = stats;
    if (r.space)
        g.space = PyDeviceStateSpace{std::move(r.space), o};
    if (r.host)
        g.host = PyStateSpace{std::move(r.host), o};
    return g;
}

PyDeviceGeneration generate_one(nb::handle task, const Common& a, cuda::StateSpaceOutput output)
{
    if (!nb::isinstance<PyTask>(task) && !nb::isinstance<PyHandle>(task))
        throw nb::type_error("mymyr: task must be a Task or TaskHandle (state_spaces takes tables)");
    const Owner o = owner_of(task);
    const cuda::ContextPtr c = context_for(task, a);
    cuda::DeviceStateSpaceOptions opts = options_of(a);
    opts.output = output;
    cuda::DeviceStateSpaceResult r;
    cuda::DeviceStateSpaceStats st;
    {
        nb::gil_scoped_release release;
        r = cuda::state_space(c, o.core->task, opts, &st);
    }
    return wrap(o, std::move(r), st);
}

std::vector<PyDeviceGeneration> generate_many(nb::handle table, const Common& a, cuda::StateSpaceOutput output,
                                              nb::handle wave_states, nb::handle wave_instances)
{
    const TableRef ref = table_of(table);
    const cuda::ContextPtr c = context_for(ref.obj, a);
    cuda::DeviceStateSpaceOptions opts = options_of(a);
    opts.output = output;
    if (!wave_states.is_none())
    {
        opts.wave_states = nb::cast<u64>(wave_states);
        if (opts.wave_states == 0)
            throw nb::value_error("mymyr: wave_states must be positive");
    }
    if (!wave_instances.is_none())
        opts.wave_instances = nb::cast<u32>(wave_instances);
    cuda::DeviceStateSpaces rs;
    {
        nb::gil_scoped_release release;
        rs = cuda::state_spaces(c, ref.table, opts);
    }
    std::vector<PyDeviceGeneration> out;
    out.reserve(rs.results.size());
    for (usize i = 0; i < rs.results.size(); ++i)
        out.push_back(wrap(ref.owner(static_cast<u32>(i)), std::move(rs.results[i]), rs.stats));
    return out;
}

const char* k_options_doc =
    " Options: as in the CPU generator's (mymyr.datasets.state_space): max_states (fail at max(max_states, 2) states "
    "or more), max_seconds, remove_if_unsolvable, labels; threads are host threads for the host-side work "
    "(state-dependent costs, host output; 0: all cores); symmetry pruning is not offered (the CPU generator only). "
    "chunk_states caps the parents per "
    "chunk, view_bytes the per-chunk view memory, expected_states pre-sizes the arrays; none changes the result. "
    "ctx: a mymyr.cuda.Context, or None: the task's (table's) default context on `device` (default 0). stream: the "
    "stream of the work and of the result's arrays (None: the context's). Numeric tasks and tasks the device cannot run "
    "raise ValueError; negative, NaN or undefined transition costs ValueError.";

#define MYMYR_DSS_ARGS                                                                                                  \
    nb::kw_only(), "ctx"_a = nb::none(), "device"_a = nb::none(), "stream"_a = nb::none(), "threads"_a = 0,            \
        "max_states"_a = nb::none(), "max_seconds"_a = nb::none(), "remove_if_unsolvable"_a = true, "labels"_a = true, \
        "chunk_states"_a = nb::none(), "view_bytes"_a = nb::none(), "expected_states"_a = nb::none()
}  // namespace

// ------------------------------------------------------------------------------------------------ bindings

void bind_cuda_datasets(nb::module_& parent)
{
    nb::module_ m = nb::borrow<nb::module_>(parent.attr("_cuda"));
    set_device_distances_hook(&device_distances);

    nb::class_<PyDeviceStateSpace>(
        m, "DeviceStateSpace",
        "A state space on the device: mymyr.datasets.StateSpace's content (the same ids, transitions, labels, "
        "costs, reverse CSR, goal distances and flags), with the arrays in device memory. arrays() exports them "
        "zero-copy (torch, jax or dlpack; the consumer's stream waits for the work that wrote them); to_host() is the "
        "mymyr.datasets.StateSpace; mymyr.datasets.StateSpaceSampler(space) samples it. The initial state is id 0.")
        .def_prop_ro("task", [](const PyDeviceStateSpace& x) { return task_object(x.owner.obj); })
        .def_prop_ro("num_states", [](const PyDeviceStateSpace& x) { return x.d->num_states(); })
        .def_prop_ro("num_transitions", [](const PyDeviceStateSpace& x) { return x.d->num_transitions(); })
        .def_prop_ro("initial_state_id", [](const PyDeviceStateSpace& x) { return x.d->initial_state(); })
        .def_prop_ro("words", [](const PyDeviceStateSpace& x) { return x.d->words(); }, "Fluent words per state row.")
        .def_prop_ro("has_labels", [](const PyDeviceStateSpace& x) { return x.d->has_labels(); })
        .def_prop_ro("label_width", [](const PyDeviceStateSpace& x) { return x.d->label_width(); })
        .def_prop_ro("unit_costs", [](const PyDeviceStateSpace& x) { return x.d->unit_costs(); },
                     "True if every transition costs 1 (arrays() then has no 'costs').")
        .def_prop_ro("num_goal_states", [](const PyDeviceStateSpace& x) { return x.d->num_goal_states(); })
        .def_prop_ro("num_unsolvable_states", [](const PyDeviceStateSpace& x) { return x.d->num_unsolvable_states(); })
        .def_prop_ro("max_goal_distance", [](const PyDeviceStateSpace& x) { return x.d->max_goal_distance(); },
                     "The largest finite unit goal distance (-1 if no state reaches a goal).")
        .def_prop_ro("layers", [](const PyDeviceStateSpace& x) { return x.d->layers(); })
        .def_prop_ro("nbytes", [](const PyDeviceStateSpace& x) { return space_bytes(*x.d); }, "Device bytes of the arrays.")
        .def_prop_ro("device", [](const PyDeviceStateSpace& x) { return x.d->context()->device(); })
        .def_prop_ro("stream", [](const PyDeviceStateSpace& x) { return reinterpret_cast<std::intptr_t>(x.d->stream()); },
                     "The stream the arrays were written on (a cudaStream_t).")
        .def(
            "arrays", [](const PyDeviceStateSpace& x, FrameworkArg framework) { return device_arrays(x, framework); },
            "framework"_a = nb::none(),
            "Zero-copy read-only device arrays (framework 'torch', 'jax' or 'dlpack'; None: dlpack), the keys and "
            "layouts of mymyr.datasets.StateSpace.arrays(): state_words [N, W], forward_offsets [N + 1] (uint64), "
            "forward_targets [E], label_schemas [E] and label_bindings [E, label_width] (with labels), "
            "backward_offsets [N + 1], backward_sources and backward_edges [E], unit_goal_distances [N] (int32), "
            "cost_goal_distances [N] (float64), costs [E] (float64; absent for unit costs), goal, unsolvable, alive "
            "[N] (bool). 64-bit integer arrays reach JAX as uint32 pairs; float64 needs jax_enable_x64.")
        .def(
            "to_host",
            [](const PyDeviceStateSpace& x) {
                datasets::StateSpacePtr h;
                {
                    nb::gil_scoped_release release;
                    h = x.d->to_host();
                }
                return PyStateSpace{std::move(h), x.owner};
            },
            "The space as a mymyr.datasets.StateSpace (downloads the arrays; equal to the CPU generator's for the "
            "same options).")
        .def(
            "state",
            [](const PyDeviceStateSpace& x, i64 id) {
                const u32 i = state_id(x, id);
                std::vector<u64> row;
                {
                    nb::gil_scoped_release release;
                    row = download(*x.d, x.d->state_words() + u64{i} * x.d->words(), x.d->words());
                }
                return make_state(x.owner, State(row.data(), bits::trimmed_size(row.data(), x.d->words())));
            },
            "id"_a, "The state with this id (copies its row to the host).")
        .def("__len__", [](const PyDeviceStateSpace& x) { return x.d->num_states(); })
        .def("__repr__", [](const PyDeviceStateSpace& x) {
            return "DeviceStateSpace(states=" + std::to_string(x.d->num_states()) + ", transitions=" +
                   std::to_string(x.d->num_transitions()) + ", goal=" + std::to_string(x.d->num_goal_states()) +
                   ", unsolvable=" + std::to_string(x.d->num_unsolvable_states()) + ", device=cuda:" +
                   std::to_string(x.d->context()->device()) + ")";
        });

    nb::class_<PyDeviceGeneration>(m, "DeviceGenerationResult",
                                   "The outcome of one device generation (mymyr.datasets.GenerationResult's fields): "
                                   "the device space and / or its host copy, per the output option.")
        .def_prop_ro("status", [](const PyDeviceGeneration& g) { return g.status; })
        .def_prop_ro("space", [](const PyDeviceGeneration& g) { return g.space; },
                     "The device space (status OK, output 'device' or 'both'), else None.")
        .def_prop_ro("host", [](const PyDeviceGeneration& g) { return g.host; },
                     "The mymyr.datasets.StateSpace (status OK, output 'host' or 'both'), else None.")
        .def_prop_ro("states", [](const PyDeviceGeneration& g) { return g.states; },
                     "States stored when the instance's generation ended (also when it failed).")
        .def_prop_ro("seconds", [](const PyDeviceGeneration& g) { return g.seconds; },
                     "Wall time from the instance's admission to its output.")
        .def_prop_ro("stats", [](const PyDeviceGeneration& g) { return stats_dict(g.stats); },
                     "Counters of the call that generated it (shared by a table's results): waves, chunks, multi (1: "
                     "the multi-instance kernels ran the table), states, transitions, generate_ms, post_ms, output_ms, "
                     "host_cost_ms, rehashes, widenings, device_bytes.")
        .def("__bool__", [](const PyDeviceGeneration& g) { return g.status == StateSpaceStatus::Ok; })
        .def("__repr__", [](const PyDeviceGeneration& g) {
            return std::string("DeviceGenerationResult(status=") + datasets::to_string(g.status) +
                   ", states=" + std::to_string(g.states) + ")";
        });

    m.def(
        "generate_state_space",
        [](TaskArg task, ContextArg ctx, DeviceArg device, StreamArg stream, u32 threads, IntArg max_states,
           FloatArg max_seconds, bool remove_if_unsolvable, bool labels, IntArg chunk_states, IntArg view_bytes,
           IntArg expected_states, OutputArg output) {
            const Common a{ctx, device, stream, threads, max_states, max_seconds, remove_if_unsolvable, labels,
                           chunk_states, view_bytes, expected_states};
            return generate_one(task, a, parse_output(output));
        },
        "task"_a, MYMYR_DSS_ARGS, "output"_a = "device",
        (std::string("The state space of a task on the device (a DeviceGenerationResult): output 'device' (the "
                     "DeviceStateSpace), 'host' (the mymyr.datasets.StateSpace) or 'both'.") +
         k_options_doc)
            .c_str());

    m.def(
        "state_space",
        [](TaskArg task, ContextArg ctx, DeviceArg device, StreamArg stream, u32 threads, IntArg max_states,
           FloatArg max_seconds, bool remove_if_unsolvable, bool labels, IntArg chunk_states, IntArg view_bytes,
           IntArg expected_states) {
            const Common a{ctx, device, stream, threads, max_states, max_seconds, remove_if_unsolvable, labels,
                           chunk_states, view_bytes, expected_states};
            return generate_one(task, a, cuda::StateSpaceOutput::Device).space;
        },
        "task"_a, MYMYR_DSS_ARGS,
        (std::string("The DeviceStateSpace of a task, or None if the generation failed (as "
                     "mymyr.datasets.state_space).") +
         k_options_doc)
            .c_str());

    m.def(
        "generate_state_spaces",
        [](TableArg table, ContextArg ctx, DeviceArg device, StreamArg stream, u32 threads, IntArg max_states,
           FloatArg max_seconds, bool remove_if_unsolvable, bool labels, IntArg chunk_states, IntArg view_bytes,
           IntArg expected_states, OutputArg output, IntArg wave_states, IntArg wave_instances) {
            const Common a{ctx, device, stream, threads, max_states, max_seconds, remove_if_unsolvable, labels,
                           chunk_states, view_bytes, expected_states};
            return generate_many(table, a, parse_output(output), wave_states, wave_instances);
        },
        "table"_a, MYMYR_DSS_ARGS, "output"_a = "device", "wave_states"_a = nb::none(), "wave_instances"_a = nb::none(),
        (std::string("The state spaces of every instance of a task table, all in one device pipeline (results in "
                     "table order; instance i's equal generate_state_space(table[i])). Instances are admitted in "
                     "waves: while a wave holds fewer than wave_states states (default 2^24) and wave_instances "
                     "instances (default unlimited); a wave's device memory is about states x (row bytes + 64) + "
                     "transitions x (32 + 4 x label_width) bytes. With output 'host' each wave's device memory is "
                     "freed before the next.") +
         k_options_doc)
            .c_str());

    m.def(
        "state_spaces",
        [](TableArg table, ContextArg ctx, DeviceArg device, StreamArg stream, u32 threads, IntArg max_states,
           FloatArg max_seconds, bool remove_if_unsolvable, bool labels, IntArg chunk_states, IntArg view_bytes,
           IntArg expected_states, IntArg wave_states, IntArg wave_instances) {
            const Common a{ctx, device, stream, threads, max_states, max_seconds, remove_if_unsolvable, labels,
                           chunk_states, view_bytes, expected_states};
            std::vector<std::optional<PyDeviceStateSpace>> out;
            for (PyDeviceGeneration& g : generate_many(table, a, cuda::StateSpaceOutput::Device, wave_states, wave_instances))
                out.push_back(std::move(g.space));
            return out;
        },
        "table"_a, MYMYR_DSS_ARGS, "wave_states"_a = nb::none(), "wave_instances"_a = nb::none(),
        (std::string("The DeviceStateSpaces of every instance of a task table (None where the generation failed), "
                     "in table order; generate_state_spaces with output 'device'.") +
         k_options_doc)
            .c_str());
}
#undef MYMYR_DSS_ARGS
}  // namespace mymyr::python
