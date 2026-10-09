// mymyr.cuda: the batched device heuristics, device A* and device GBFS (cuda/heuristics.hpp, cuda/astar.hpp,
// cuda/gbfs.hpp): Heuristic, astar, gbfs and their result, DeviceSearchResult. Registered by bind_cuda
// (cuda_bindings.cpp), which passes the lookup of a task's device context.

#include "arrays.hpp"
#include "device_hooks.hpp"
#include "dlpack.hpp"
#include "py_task.hpp"

#include "mymyr/cuda/astar.hpp"
#include "mymyr/cuda/gbfs.hpp"
#include "mymyr/cuda/heuristics.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::python::ann
{
/// A mymyr.cuda.Context (bound in cuda_bindings.cpp).
struct CudaContext
{
};
/// The heuristics of the device searches and of Heuristic.
struct DeviceKind
{
};
struct EvalKind
{
};
struct HeuristicCosts
{
};
struct HeuristicVariant
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
struct type_caster<mymyr::python::ann::DeviceKind>
{
    static constexpr auto Name = const_name("typing.Literal['blind', 'max', 'add', 'ff', 'h2', 'set_additive']");
};
template<>
struct type_caster<mymyr::python::ann::EvalKind>
{
    static constexpr auto Name = const_name("typing.Literal['max', 'add', 'ff', 'h2', 'set_additive']");
};
template<>
struct type_caster<mymyr::python::ann::HeuristicCosts>
{
    static constexpr auto Name = const_name("typing.Literal['unit', 'real']");
};
template<>
struct type_caster<mymyr::python::ann::HeuristicVariant>
{
    static constexpr auto Name = const_name("typing.Literal['auto', 'sweep', 'frontier']");
};
}  // namespace nanobind::detail

namespace mymyr::python
{
using namespace nb::literals;

using ContextLookup = cuda::ContextPtr (*)(PyTaskCore& core, nb::handle ctx, int device);

void bind_cuda_heuristics(nb::module_& m, ContextLookup lookup);

namespace
{
ContextLookup g_lookup = nullptr;

// ------------------------------------------------------------------------------------------------ argument types
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using ContextArg = Arg<ann::CudaContext>;
using KindArg = Arg<ann::DeviceKind>;
using EvalKindArg = Arg<ann::EvalKind>;
using BoolArg = Arg<bool>;
using CostsArg = Arg<ann::HeuristicCosts>;
using VariantArg = Arg<ann::HeuristicVariant>;
using StateArg = Arg<PyState>;
using ActionList = nb::typed<nb::list, PyAction>;
using ArrayOut = Arg<ann::Any>;
using StatsDict = nb::typed<nb::dict, std::string, std::variant<double, u64, bool, std::string>>;

std::string str_of(nb::handle v, const char* name)
{
    if (!nb::isinstance<nb::str>(v))
        throw nb::type_error((std::string("mymyr: ") + name + " must be a string").c_str());
    return nb::cast<std::string>(v);
}

/// A heuristic kind (heuristics::parse_kind's names); the device refuses kinds it does not run itself.
heuristics::Kind kind_of(nb::handle v, const char* name)
{
    try
    {
        return heuristics::parse_kind(str_of(v, name));
    }
    catch (const std::invalid_argument& e)
    {
        throw nb::value_error(e.what());
    }
}

heuristics::Costs costs_of(nb::handle v)
{
    const std::string c = str_of(v, "costs");
    if (c == "unit")
        return heuristics::Costs::Unit;
    if (c == "real")
        return heuristics::Costs::Real;
    throw nb::value_error("mymyr: costs must be 'unit' or 'real'");
}

cuda::HeuristicVariant variant_of(nb::handle v)
{
    const std::string s = str_of(v, "variant");
    if (s == "auto")
        return cuda::HeuristicVariant::Auto;
    if (s == "sweep")
        return cuda::HeuristicVariant::Sweep;
    if (s == "frontier")
        return cuda::HeuristicVariant::Frontier;
    throw nb::value_error("mymyr: variant must be 'auto', 'sweep' or 'frontier'");
}

/// A State of the task `o` (TypeError / ValueError otherwise).
const State& task_state(const Owner& o, nb::handle v, const char* name)
{
    if (!is_state(v))
        throw nb::type_error((std::string("mymyr: ") + name + " must be a mymyr.State").c_str());
    const PyState& s = state_of(v);
    if (s.core->task->uid() != o.core->task->uid())
        throw nb::value_error((std::string("mymyr: ") + name + " belongs to another task").c_str());
    return s.s;
}

cudaStream_t stream_value(nb::handle stream, cudaStream_t fallback, std::intptr_t& dl_value)
{
    if (stream.is_none())
    {
        dl_value = reinterpret_cast<std::intptr_t>(fallback);
        return fallback;
    }
    nb::object h = nb::getattr(stream, "cuda_stream", nb::none());
    const std::intptr_t v = nb::cast<std::intptr_t>(h.is_none() ? nb::borrow(stream) : h);
    if (v == dl::k_stream_none)
        throw nb::value_error("mymyr: stream -1 means 'no synchronization'; pass a stream to run on");
    if (v == 0 || v == dl::k_stream_legacy)
    {
        dl_value = dl::k_stream_legacy;
        return cudaStreamLegacy;
    }
    dl_value = v;
    return v == dl::k_stream_per_thread ? cudaStreamPerThread : reinterpret_cast<cudaStream_t>(v);
}

/// Handoff of the values written on `producer`: the consumer's stream waits for them and becomes a user of the buffer.
class ValuesSync final : public DeviceSync
{
public:
    ValuesSync(std::shared_ptr<cuda::DeviceBuffer> buffer, cudaStream_t producer) : m_buffer(std::move(buffer)), m_producer(producer) {}
    void handoff(std::intptr_t consumer) override
    {
        const cudaStream_t c = consumer == dl::k_stream_legacy       ? cudaStreamLegacy
                               : consumer == dl::k_stream_per_thread ? cudaStreamPerThread
                                                                     : reinterpret_cast<cudaStream_t>(consumer);
        cuda::DeviceGuard g(m_buffer->context()->device());
        cuda::stream_wait(c, m_producer);
        m_buffer->record_stream(c);
    }

private:
    std::shared_ptr<cuda::DeviceBuffer> m_buffer;
    cudaStream_t m_producer;
};

// ------------------------------------------------------------------------------------------------ Heuristic

struct PyDeviceHeuristic
{
    Owner o;
    std::unique_ptr<cuda::DeviceHeuristic> h;
    std::mutex m;  // one evaluation at a time (the heuristic's scratch is reused)
};

nb::object host_values(const std::vector<f64>& v)
{
    auto* data = new f64[v.size() ? v.size() : 1];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<f64*>(p); });
    const size_t shape[1] = {v.size()};
    return nb::ndarray<nb::numpy, f64, nb::ndim<1>>(data, 1, shape, owner).cast();
}

ArrayOut evaluate(PyDeviceHeuristic& self, StatesLike states, StreamArg stream)
{
    const Task& task = *self.o.core->task;
    if (!is_cuda_array(states))
    {
        // host states: uploaded, evaluated, downloaded
        const StateBatch sb = import_task_states(states, task);
        std::vector<State> v;
        v.reserve(sb.view.rows);
        for (u64 i = 0; i < sb.view.rows; ++i)
            v.emplace_back(sb.view.row(i), sb.view.words);
        std::vector<f64> out;
        {
            nb::gil_scoped_release release;
            std::lock_guard lock(self.m);
            out = self.h->evaluate(v);
        }
        return ArrayOut(host_values(out));
    }
    const cuda::ContextPtr& c = self.h->context();
    const dl::Device d = dl::dlpack_device(states);
    if (d.device_id != c->device())
        throw nb::value_error(("mymyr: 'states' lives on cuda:" + std::to_string(d.device_id) + ", the heuristic on cuda:" +
                               std::to_string(c->device()))
                                  .c_str());
    std::intptr_t dl_stream = 0;
    const cudaStream_t st = stream_value(stream, c->stream(), dl_stream);
    const dl::Imported im = dl::import_dlpack(states, dl_stream);
    WordsLayout w = words_layout(im.data, im.dtype.code, im.dtype.bits, im.dtype.lanes, im.shape.size(),
                                       im.shape.data(), im.strides.data());
    if (w.words < task.numeric_words())
        throw nb::value_error("mymyr: heuristic state rows are narrower than the numeric block");
    w.words -= task.numeric_words();
    const Framework fw = framework_of(states);
    std::shared_ptr<cuda::DeviceBuffer> buf;
    {
        nb::gil_scoped_release release;
        std::lock_guard lock(self.m);
        const cuda::DeviceGuard guard(c->device());
        buf = std::make_shared<cuda::DeviceBuffer>(c, std::max<u64>(w.rows, 1) * sizeof(u32), st);
        self.h->evaluate(w.data, w.stride, w.words, w.rows, static_cast<u32*>(buf->data()), st);
    }
    ArraySpec spec;
    spec.owner = buf;
    spec.data = buf->data();
    spec.dtype = rl::DType::U32;
    spec.shape = {static_cast<i64>(w.rows)};
    DeviceExport dev{dl::k_cuda, c->device(), std::make_shared<ValuesSync>(buf, st)};
    return ArrayOut(export_device_array(std::move(spec), std::move(dev), fw == Framework::Numpy ? Framework::DLPack : fw,
                                        default_words(fw)));
}

StatsDict heuristic_stats(const cuda::DeviceHeuristicStats& s)
{
    StatsDict d{nb::dict()};
    d["evaluations"] = s.evaluations;
    d["launches"] = s.launches;
    d["fallbacks"] = s.fallbacks;
    d["propositions"] = u64{s.propositions};
    d["operators"] = u64{s.operators};
    d["ground_actions"] = u64{s.ground_actions};
    d["grounding_seconds"] = s.grounding_seconds;
    d["variant"] = std::string(cuda::to_string(s.variant));
    d["threads"] = u64{s.threads};
    d["blocks"] = u64{s.blocks};
    d["warp_groups"] = s.warp_groups;
    d["shared"] = s.shared;
    d["group_bytes"] = s.group_bytes;
    d["supporter_levels"] = s.supporter_levels;
    return d;
}

// ------------------------------------------------------------------------------------------------ searches

struct PyDeviceSearchResult
{
    cuda::DeviceBestFirstResult r;
    Owner o;
};

ActionList plan_list(const Owner& o, const std::vector<Action>& plan)
{
    ActionList out{nb::list()};
    std::vector<i32> b;
    for (const Action& a : plan)
    {
        b.assign(a.binding.size(), 0);
        for (usize j = 0; j < a.binding.size(); ++j)
            b[j] = static_cast<i32>(a.binding[j].v);
        out.append(make_label(o, a.schema.v, b.data(), static_cast<u32>(a.binding.size())));
    }
    return out;
}

PyDeviceSearchResult run_search(bool greedy, TaskArg task, nb::handle heuristic, nb::handle costs, nb::handle ctx, nb::handle start,
                                u32 batch, bool single_bucket, bool reopen, bool witness_pruning, bool canonical_order,
                                nb::handle max_states, nb::handle max_expanded, nb::handle max_depth, nb::handle max_seconds,
                                nb::handle chunk_states, nb::handle variant)
{
    const Owner o = owner_of(task);
    const TaskPtr& t = o.core->task;
    const cuda::ContextPtr c = g_lookup(*o.core, ctx, 0);
    cuda::DeviceBestFirstOptions opts;
    search::BestFirstOptions& so = opts.search;
    so.heuristic.kind = kind_of(heuristic, "heuristic");
    so.heuristic.costs = costs_of(costs);
    if (!start.is_none())
        so.start = task_state(o, start, "start");
    so.reopen = reopen;
    so.witness_pruning = witness_pruning;
    so.canonical_order = canonical_order;
    if (!max_states.is_none())
        so.control.budget.max_states = int_arg<u64>(max_states, "max_states");
    if (!max_expanded.is_none())
        so.control.budget.max_expanded = int_arg<u64>(max_expanded, "max_expanded");
    if (!max_depth.is_none())
        so.control.budget.max_depth = int_arg<u32>(max_depth, "max_depth");
    if (!max_seconds.is_none())
    {
        const double s = nb::cast<double>(max_seconds);
        if (!(s >= 0) || std::isnan(s))
            throw nb::value_error("mymyr: max_seconds must be non-negative");
        so.control.budget.max_seconds = s;
    }
    if (batch == 0)
        throw nb::value_error("mymyr: batch must be positive");
    opts.batch = batch;
    opts.single_bucket = single_bucket;
    if (!chunk_states.is_none())
    {
        opts.chunk_states = int_arg<u64>(chunk_states, "chunk_states");
        if (opts.chunk_states == 0)
            throw nb::value_error("mymyr: chunk_states must be positive");
    }
    opts.heuristic.variant = variant_of(variant);
    PyDeviceSearchResult x;
    x.o = o;
    {
        nb::gil_scoped_release release;
        x.r = greedy ? cuda::gbfs(c, t, opts) : cuda::astar(c, t, opts);
    }
    return x;
}

StatsDict device_stats(const cuda::DeviceBestFirstStats& s)
{
    StatsDict d{nb::dict()};
    d["steps"] = s.steps;
    d["chunks"] = s.chunks;
    d["popped"] = s.popped;
    d["stale"] = s.stale;
    d["max_batch"] = s.max_batch;
    d["open_entries"] = s.open_entries;
    d["graph_steps"] = s.graph_steps;
    d["host_steps"] = s.host_steps;
    d["loops"] = s.loops;
    d["captures"] = s.captures;
    d["aborts"] = s.aborts;
    d["widenings"] = u64{s.widenings};
    d["rehashes"] = u64{s.rehashes};
    d["uploads"] = u64{s.uploads};
    d["table_slots"] = s.table_slots;
    d["device_bytes"] = s.device_bytes;
    d["host_ms"] = s.host_ms;
    d["heuristic_ms"] = s.heuristic_ms;
    return d;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ bindings

void bind_cuda_heuristics(nb::module_& m, ContextLookup lookup)
{
    g_lookup = lookup;

    nb::class_<PyDeviceHeuristic>(m, "Heuristic",
                                  "A batched grounded heuristic on the device: h_max, h_add, h_FF, h² or set-additive of many states "
                                  "per launch over the relaxed grounding, uploaded once. h_max, h_add and h² equal "
                                  "mymyr.search.Heuristic's; h_FF and set-additive break ties among equally cheap supporters by BFS level "
                                  "and operator id (reference() is the CPU implementation of that rule). Numeric values "
                                  "and constraints are ignored in relaxation, as on the CPU. Real costs must be "
                                  "state-independent, non-negative integers below 2^31; unsupported costs and groundings "
                                  "beyond the budget raise ValueError.")
        .def_ro_static("DEAD_END", &cuda::DeviceHeuristic::k_dead_end, "The uint32 value of a dead end on the device.")
        .def(
            "__init__",
            [](PyDeviceHeuristic* self, TaskArg task, EvalKindArg kind, CostsArg costs, ContextArg ctx, VariantArg variant,
               IntArg threads, BoolArg warp_groups, IntArg max_blocks, bool force_global,
               IntArg max_operators, IntArg max_scratch_bytes) {
                const Owner o = owner_of(task);
                cuda::DeviceHeuristicOptions ho;
                ho.kind = kind_of(kind, "kind");
                ho.costs = costs_of(costs);
                ho.variant = variant_of(variant);
                if (!threads.is_none())
                    ho.threads = int_arg<u32>(threads, "threads");
                if (!warp_groups.is_none())
                    ho.warp_groups = nb::cast<bool>(warp_groups) ? 1 : 0;
                if (!max_blocks.is_none())
                    ho.max_blocks = int_arg<u32>(max_blocks, "max_blocks");
                ho.force_global = force_global;
                if (!max_scratch_bytes.is_none())
                    ho.max_scratch_bytes = int_arg<u64>(max_scratch_bytes, "max_scratch_bytes");
                if (!max_operators.is_none())
                    ho.budget.max_operators = int_arg<u64>(max_operators, "max_operators");
                const cuda::ContextPtr c = g_lookup(*o.core, ctx, 0);
                std::unique_ptr<cuda::DeviceHeuristic> h;
                {
                    nb::gil_scoped_release release;
                    h = std::make_unique<cuda::DeviceHeuristic>(c, o.core->task, ho);
                }
                new (self) PyDeviceHeuristic();
                self->o = o;
                self->h = std::move(h);
            },
            "task"_a, "kind"_a = "ff", nb::kw_only(), "costs"_a = "unit", "ctx"_a = nb::none(), "variant"_a = "auto",
            "threads"_a = nb::none(), "warp_groups"_a = nb::none(), "max_blocks"_a = nb::none(), "force_global"_a = false,
            "max_operators"_a = nb::none(), "max_scratch_bytes"_a = nb::none(),
            "kind: 'max', 'add', 'ff', 'h2' or 'set_additive'; costs: 'unit' (every action 1) or 'real' (the task's action costs). The launch "
            "configuration (variant, threads, warp_groups, max_blocks, force_global) changes no value. h2 uses global "
            "scratch and sweeps, at most 8191 propositions. max_scratch_bytes limits global scratch (default 512 MiB); "
            "h2 raises ValueError when a single state needs more.")
        .def("evaluate", &evaluate, "states"_a, nb::kw_only(), "stream"_a = nb::none(),
             "h of every state. Host states (a State, a sequence of States, a host word array) give a NumPy float64 array "
             "(inf for dead ends). A CUDA word array [N, W] uint64 / [N, 2W] uint32 (torch, JAX, DLPack) is read in place "
             "after the work on `stream` (None: the context's stream) and gives a uint32 device array of N values in the "
             "input's framework (Heuristic.DEAD_END for dead ends), produced on that stream.")
        .def(
            "reference",
            [](PyDeviceHeuristic& self, StateArg state) {
                const State& s = task_state(self.o, state, "state");
                nb::gil_scoped_release release;
                std::lock_guard lock(self.m);
                return self.h->reference(s.view());
            },
            "state"_a,
            "The CPU reference of the device's value (h_max, h_add, h²: mymyr.search.Heuristic's; h_FF, set-additive: the device's "
            "supporter rule on the CPU); inf for dead ends.")
        .def_prop_ro("kind", [](const PyDeviceHeuristic& self) { return std::string(heuristics::to_string(self.h->kind())); })
        .def_prop_ro("stats", [](const PyDeviceHeuristic& self) { return heuristic_stats(self.h->stats()); },
                     "evaluations, launches, CPU fallbacks, the grounding's size and the resolved launch configuration")
        .def("__repr__", [](const PyDeviceHeuristic& self) {
            const cuda::DeviceHeuristicStats& s = self.h->stats();
            return std::string("Heuristic(kind=") + heuristics::to_string(self.h->kind()) +
                   ", propositions=" + std::to_string(s.propositions) + ", operators=" + std::to_string(s.operators) +
                   ", variant=" + cuda::to_string(s.variant) + ")";
        });

    nb::class_<PyDeviceSearchResult>(m, "DeviceSearchResult",
                                     "The result of mymyr.cuda.astar / gbfs: what mymyr.search.BestFirstResult reports, and "
                                     "the device statistics.")
        .def_prop_ro("status", [](const PyDeviceSearchResult& x) { return x.r.result.status; })
        .def_prop_ro("solved", [](const PyDeviceSearchResult& x) { return x.r.result.status == search::SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyDeviceSearchResult& x) { return plan_list(x.o, x.r.result.plan); })
        .def_prop_ro("cost", [](const PyDeviceSearchResult& x) { return x.r.result.cost; })
        .def_prop_ro("goal_state",
                     [](const PyDeviceSearchResult& x) -> Arg<std::optional<PyState>> {
                         if (!x.r.result.goal_state)
                             return Arg<std::optional<PyState>>(nb::none());
                         return Arg<std::optional<PyState>>(make_state(x.o, State(*x.r.result.goal_state)));
                     })
        .def_prop_ro("stats", [](const PyDeviceSearchResult& x) { return x.r.result.stats; })
        .def_prop_ro("evaluations", [](const PyDeviceSearchResult& x) { return x.r.result.evaluations; })
        .def_prop_ro("dead_ends", [](const PyDeviceSearchResult& x) { return x.r.result.dead_ends; })
        .def_prop_ro("reopened", [](const PyDeviceSearchResult& x) { return x.r.result.reopened; })
        .def_prop_ro("initial_h", [](const PyDeviceSearchResult& x) { return x.r.result.initial_h; })
        .def_prop_ro("heuristic", [](const PyDeviceSearchResult& x) { return x.r.result.heuristic; })
        .def_prop_ro("setup_seconds", [](const PyDeviceSearchResult& x) { return x.r.result.setup_seconds; })
        .def_prop_ro("algorithm", [](const PyDeviceSearchResult& x) { return x.r.result.algorithm; })
        .def_prop_ro("store", [](const PyDeviceSearchResult& x) { return x.r.result.store; })
        .def_prop_ro("queue", [](const PyDeviceSearchResult& x) { return x.r.result.queue; })
        .def_prop_ro("message", [](const PyDeviceSearchResult& x) { return x.r.result.message; })
        .def_prop_ro("device", [](const PyDeviceSearchResult& x) { return device_stats(x.r.device); },
                     "steps (graph / host-driven), device loops, captures, aborts, chunks, popped / stale entries, the largest batch, "
                     "device bytes, host and heuristic times")
        .def("__repr__", [](const PyDeviceSearchResult& x) {
            const search::BestFirstResult& r = x.r.result;
            return "DeviceSearchResult(algorithm=" + r.algorithm + ", status=" + search::to_string(r.status) +
                   ", plan_length=" + std::to_string(r.plan.size()) + ", cost=" + std::to_string(r.cost) +
                   ", expanded=" + std::to_string(r.stats.expanded) + ", steps=" + std::to_string(x.r.device.steps) + ")";
        });

#define MYMYR_DEVICE_SEARCH_ARGS                                                                                           \
    "costs"_a = "unit", "ctx"_a = nb::none(), "start"_a = nb::none(), "batch"_a = 10000, "single_bucket"_a = true,       \
        "reopen"_a = true, "witness_pruning"_a = false, "canonical_order"_a = true, "max_states"_a = nb::none(),         \
        "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(),                        \
        "chunk_states"_a = nb::none(), "variant"_a = "auto"

    m.def(
        "astar",
        [](TaskArg task, KindArg heuristic, CostsArg costs, ContextArg ctx, StateArg start, IntArg batch_in, bool single_bucket,
           bool reopen, bool witness_pruning, bool canonical_order, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds, IntArg chunk_states, VariantArg variant) {
            const u32 batch = int_arg<u32>(batch_in, "batch", 1);
            return run_search(false, task, heuristic, costs, ctx, start, batch, single_bucket, reopen, witness_pruning,
                              canonical_order, max_states, max_expanded, max_depth, max_seconds, chunk_states, variant);
        },
        "task"_a, nb::kw_only(), "heuristic"_a = "max", MYMYR_DEVICE_SEARCH_ARGS,
        "A* on the device with batched expansion: every step expands up to `batch` open nodes of the lowest bucket "
        "(f, h; single_bucket=False: the whole lowest f layer), the heuristic of the new states in one batch. With "
        "an admissible heuristic and non-negative additive costs the plan is optimal; at batch=1 it is mymyr.search.astar "
        "(same statistics and plan). Numeric tasks evaluate fluent costs and state metrics on the device and use a "
        "host double-priority heap, one parent per step, with CPU eager tie ordering; they do not capture graphs.");

    m.def(
        "gbfs",
        [](TaskArg task, KindArg heuristic, CostsArg costs, ContextArg ctx, StateArg start, IntArg batch_in, bool single_bucket,
           bool reopen, bool witness_pruning, bool canonical_order, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds, IntArg chunk_states, VariantArg variant) {
            const u32 batch = int_arg<u32>(batch_in, "batch", 1);
            return run_search(true, task, heuristic, costs, ctx, start, batch, single_bucket, reopen, witness_pruning,
                              canonical_order, max_states, max_expanded, max_depth, max_seconds, chunk_states, variant);
        },
        "task"_a, nb::kw_only(), "heuristic"_a = "ff", MYMYR_DEVICE_SEARCH_ARGS,
        "Greedy best-first search on the device with batched expansion: every step expands the `batch` open nodes "
        "of the smallest (h, g); at batch=1 with 'max' or 'add' it is mymyr.search.gbfs. A batch expands many "
        "more states than the sequential search would. Numeric tasks use one parent per step with a host double-priority "
        "heap and device fluent/conditional cost and state-metric programs.");
#undef MYMYR_DEVICE_SEARCH_ARGS
}
}  // namespace mymyr::python
