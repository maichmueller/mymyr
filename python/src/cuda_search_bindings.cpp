// mymyr.cuda: many IW searches at once on the device (cuda/multi_iw.hpp, cuda/rollouts.hpp): multi_iw, rollouts
// and batched_iw1, and their result, IwBatch. multi_iw and batched_iw1 also take a task table with a task id per search
// (DeviceTableIw). Registered by bind_cuda (cuda_bindings.cpp), which passes the lookup of a task's device
// context (the contexts and streams of the CUDA module live there; tables use table_device_context).

#include "arrays.hpp"
#include "device_hooks.hpp"
#include "dlpack.hpp"
#include "py_table.hpp"
#include "py_task.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/rollouts.hpp"
#include "mymyr/search/iw.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mymyr::python::ann
{
/// A mymyr.cuda.Context (bound in cuda_bindings.cpp).
struct CudaContext
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
}  // namespace nanobind::detail

namespace mymyr::python
{
using namespace nb::literals;

/// The device context of an operation on a task: `ctx` (a mymyr.cuda.Context), or the task's default context on
/// `device` (cuda_bindings.cpp: task_context).
using ContextLookup = cuda::ContextPtr (*)(PyTaskCore& core, nb::handle ctx, int device);

void bind_cuda_search(nb::module_& m, ContextLookup lookup);
/// The device context of an operation on a table (cuda_bindings.cpp): `ctx`, or the table's default context on `device`.
cuda::ContextPtr table_device_context(nb::handle table, nb::handle ctx, int device);

namespace
{
ContextLookup g_lookup = nullptr;

// ------------------------------------------------------------------------------------------------ argument types
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using ContextArg = Arg<ann::CudaContext>;
using IntArg = Arg<u64>;
using FloatArg = Arg<double>;
using StrArg = Arg<std::string>;
using StateArg = Arg<PyState>;
using Slots = nb::typed<nb::sequence, int>;
/// One goal: (positive fluent slots, negative fluent slots) (State.atom_slots(), Atom.slot).
using GoalArg = nb::typed<nb::tuple, Slots, Slots>;
using GoalsArg = Arg<nb::typed<nb::sequence, GoalArg>>;
using SeedsArg = nb::typed<nb::sequence, int>;
/// Start states: host states (a State, a sequence of States, a host word array) or, for batched_iw1, a CUDA device
/// word array [N, W] uint64 / [N, 2W] uint32 read in place.
using StartsArg = StatesLike;
using ActionList = nb::typed<nb::list, PyAction>;

struct PyIwBatch
{
    cuda::MultiIwBatch b;
    std::vector<Owner> owners;  // per instance (one for a task)
    std::vector<u32> ids;       // the instance of each search (empty: all 0)
    std::vector<State> starts;  // one per search (rollouts: one for all), for the plan costs
    bool rollouts = false;
};

/// The owner of search i's labels and states (its instance's task).
const Owner& owner_of_search(const PyIwBatch& x, u32 i) { return x.owners.at(x.ids.empty() ? 0 : x.ids.at(i)); }

u32 checked_index(const PyIwBatch& x, i64 i)
{
    if (i < 0)
        i += x.b.n;
    if (i < 0 || static_cast<u64>(i) >= x.b.n)
        throw nb::index_error("mymyr: search index out of range");
    return static_cast<u32>(i);
}

const State& start_of(const PyIwBatch& x, u32 i) { return x.starts.size() == 1 ? x.starts[0] : x.starts.at(i); }

search::WidthZero width_zero_of(nb::handle v)
{
    const std::string wz = nb::cast<std::string>(v);
    if (wz == "expand_depth_one")
        return search::WidthZero::ExpandDepthOne;
    if (wz == "root_only")
        return search::WidthZero::RootOnly;
    throw nb::value_error("mymyr: width_zero must be 'expand_depth_one' or 'root_only'");
}

/// The options shared by the three entry points.
cuda::MultiIwOptions options_of(u32 max_arity, nb::handle width_zero, bool optimize_iw1, bool witness_pruning,
                                bool canonical_order, bool exact, nb::handle max_states, nb::handle max_expanded,
                                nb::handle max_depth, nb::handle max_seconds, nb::handle max_searches, nb::handle chunk_states)
{
    cuda::MultiIwOptions o;
    o.max_arity = max_arity;
    o.width_zero = width_zero_of(width_zero);
    o.optimize_iw1 = optimize_iw1;
    o.witness_pruning = witness_pruning;
    o.canonical_order = canonical_order;
    o.exact = exact;
    if (!max_states.is_none())
        o.budget.max_states = nb::cast<u64>(max_states);
    if (!max_expanded.is_none())
        o.budget.max_expanded = nb::cast<u64>(max_expanded);
    if (!max_depth.is_none())
        o.budget.max_depth = nb::cast<u32>(max_depth);
    if (!max_seconds.is_none())
    {
        const double s = nb::cast<double>(max_seconds);
        if (!(s >= 0) || std::isnan(s))
            throw nb::value_error("mymyr: max_seconds must be non-negative");
        o.budget.max_seconds = s;
    }
    if (!max_searches.is_none())
        o.max_searches = nb::cast<u32>(max_searches);
    if (!chunk_states.is_none())
    {
        o.chunk_states = nb::cast<u32>(chunk_states);
        if (o.chunk_states == 0)
            throw nb::value_error("mymyr: chunk_states must be positive");
    }
    return o;
}

search::GoalSpec::AtomGoal goal_of(nb::handle g, const Task& task)
{
    if (!nb::isinstance<nb::tuple>(g) || nb::len(g) != 2)
        throw nb::type_error("mymyr: a goal is a tuple (positive slots, negative slots)");
    search::GoalSpec::AtomGoal out;
    const u32 slots = task.atoms().fluent_slots();
    auto fill = [&](nb::handle seq, std::vector<SlotId>& dst)
    {
        for (nb::handle v : seq)
        {
            const i64 s = nb::cast<i64>(v);
            if (s < 0 || static_cast<u64>(s) >= slots)
                throw nb::value_error(("mymyr: goal slot " + std::to_string(s) + " is not a fluent slot of the task (0.." +
                                       std::to_string(slots) + ")")
                                          .c_str());
            dst.push_back(SlotId{static_cast<u32>(s)});
        }
    };
    fill(g[0], out.positive);
    fill(g[1], out.negative);
    return out;
}

std::vector<search::GoalSpec::AtomGoal> goals_of(nb::handle goals, const Task& task, u64 n)
{
    std::vector<search::GoalSpec::AtomGoal> out;
    if (goals.is_none())
        return out;
    for (nb::handle g : goals)
        out.push_back(goal_of(g, task));
    if (out.size() != n)
        throw nb::value_error(("mymyr: " + std::to_string(out.size()) + " goals for " + std::to_string(n) +
                               " searches (pass None, or one per search)")
                                  .c_str());
    return out;
}

/// Host start states of a StatesLike argument.
std::vector<State> host_starts(nb::handle obj, const Task& task)
{
    const StateBatch sb = import_task_states(obj, task);
    std::vector<State> out;
    out.reserve(sb.view.rows);
    for (u64 i = 0; i < sb.view.rows; ++i)
        out.emplace_back(sb.view.row(i), sb.view.words, sb.view.numeric_words ? sb.view.row(i) + sb.view.words : nullptr,
                         sb.view.numeric_words);
    return out;
}

/// Host start states over a table: States, or rows of at most the table's width.
std::vector<State> table_starts(nb::handle obj, const rl::TaskTable& table)
{
    if (table.numeric())
        throw nb::value_error("mymyr: CUDA multi-instance IW does not support numeric task tables");
    const StateBatch sb = import_rows(obj, current_words(table), 0);
    std::vector<State> out;
    out.reserve(sb.view.rows);
    for (u64 i = 0; i < sb.view.rows; ++i)
        out.emplace_back(sb.view.row(i), bits::trimmed_size(sb.view.row(i), sb.view.words));
    return out;
}

/// Start States must be states of their search's instance.
void check_start_owners(nb::handle obj, const rl::TaskTable& table, std::span<const u32> ids)
{
    const bool seq = nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj);
    if (!is_state(obj) && !seq)
        return;
    u64 i = 0;
    auto check = [&](nb::handle s)
    {
        if (is_state(s) && i < ids.size() && state_of(s).core->task.get() != table.task(ids[i]).get())
            throw nb::value_error(("mymyr: start " + std::to_string(i) + " is not a state of instance " +
                                   std::to_string(ids[i]) + " (its task id)")
                                      .c_str());
        ++i;
    };
    if (seq)
        for (nb::handle s : nb::borrow<nb::sequence>(obj))
            check(s);
    else
        check(obj);
}

/// The task ids of `rows` searches over a table: host ints (import_task_ids), or an int32 CUDA array read after the
/// work on `stream` (copied to the host); None for a table of one.
std::vector<u32> search_task_ids(nb::handle obj, u64 rows, const rl::TaskTable& table, cudaStream_t stream, std::intptr_t dl_stream)
{
    if (obj.is_none())
    {
        if (table.size() != 1)
            throw nb::value_error("mymyr: a table of several instances needs task_ids (one per search)");
        return std::vector<u32>(rows, 0);
    }
    const dl::Device d = dl::dlpack_device(obj);
    std::vector<u32> out(rows);
    if (d.device_type >= 0 && !dl::host_accessible(d.device_type))
    {
        const dl::Imported im = dl::import_dlpack(obj, dl_stream);
        if (im.dtype.code != dl::k_int || im.dtype.bits != 32 || im.dtype.lanes != 1 || im.shape.size() != 1 ||
            static_cast<u64>(im.shape[0]) != rows || (rows > 1 && im.strides[0] != 1))
            throw nb::type_error(("mymyr: device 'task_ids' must be a contiguous int32 array [" + std::to_string(rows) + "]").c_str());
        {
            nb::gil_scoped_release release;
            const cuda::DeviceGuard guard(d.device_id);
            if (rows)
                cuda::check(cudaMemcpyAsync(out.data(), im.data, rows * 4, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (task_ids)");
            cuda::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        }
        for (u64 r = 0; r < rows; ++r)
            if (out[r] >= table.size())
                throw nb::value_error(("mymyr: task id " + std::to_string(static_cast<i32>(out[r])) + " of search " +
                                       std::to_string(r) + " is outside the table's " + std::to_string(table.size()) +
                                       " instances")
                                          .c_str());
        return out;
    }
    std::vector<i32> copy;
    std::shared_ptr<void> keep;
    const i32* p = import_task_ids(obj, rows, table.size(), copy, keep);
    for (u64 r = 0; r < rows; ++r)
        out[r] = static_cast<u32>(p[r]);
    return out;
}

/// The goals of a table's searches (one per search, each checked against its instance), or none.
std::vector<search::GoalSpec::AtomGoal> table_goals(nb::handle goals, const rl::TaskTable& table, std::span<const u32> ids)
{
    std::vector<search::GoalSpec::AtomGoal> out;
    if (goals.is_none())
        return out;
    u64 i = 0;
    for (nb::handle g : goals)
    {
        if (i >= ids.size())
            break;
        out.push_back(goal_of(g, *table.task(ids[i++])));
    }
    if (out.size() != ids.size() || nb::len(goals) != ids.size())
        throw nb::value_error(("mymyr: " + std::to_string(nb::len(goals)) + " goals for " + std::to_string(ids.size()) +
                               " searches (pass None, or one per search)")
                                  .c_str());
    return out;
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
}  // namespace

// ------------------------------------------------------------------------------------------------ bindings

void bind_cuda_search(nb::module_& m, ContextLookup lookup)
{
    g_lookup = lookup;

    nb::class_<PyIwBatch>(m, "IwBatch",
                          "Results of many device IW searches (mymyr.cuda.multi_iw, rollouts, batched_iw1): per search i "
                          "what mymyr.search.iw reports (status, plan, goal state, per-pass counts, cost) and, for "
                          "rollouts, the reached fluent atoms.")
        .def("__len__", [](const PyIwBatch& x) { return x.b.n; })
        .def_prop_ro("status", [](const PyIwBatch& x) { return x.b.status; }, "per search")
        .def_prop_ro("solved",
                     [](const PyIwBatch& x) {
                         std::vector<bool> out(x.b.n);
                         for (u32 i = 0; i < x.b.n; ++i)
                             out[i] = x.b.status[i] == search::SearchStatus::Solved;
                         return out;
                     },
                     "per search")
        .def_prop_ro("plan_length", [](const PyIwBatch& x) { return x.b.plan_length; }, "per search; -1 unless solved")
        .def_prop_ro("effective_width", [](const PyIwBatch& x) { return x.b.effective_width; },
                     "per search: the arity of the pass that solved it (0 unless solved)")
        .def_prop_ro("expanded",
                     [](const PyIwBatch& x) {
                         std::vector<u64> out(x.b.n);
                         for (u32 i = 0; i < x.b.n; ++i)
                             out[i] = x.b.expanded(i);
                         return out;
                     },
                     "per search, over its passes")
        .def_prop_ro("generated",
                     [](const PyIwBatch& x) {
                         std::vector<u64> out(x.b.n);
                         for (u32 i = 0; i < x.b.n; ++i)
                             out[i] = x.b.generated(i);
                         return out;
                     },
                     "per search, over its passes")
        .def(
            "plan",
            [](const PyIwBatch& x, i64 i) {
                const u32 k = checked_index(x, i);
                ActionList out{nb::list()};
                const u32 L = x.b.label_width;
                std::vector<i32> b(L);
                const Owner& o = owner_of_search(x, k);
                for (const Action& a : x.b.plan(k, *o.core->task))
                {
                    for (usize j = 0; j < a.binding.size(); ++j)
                        b[j] = static_cast<i32>(a.binding[j].v);
                    out.append(make_label(o, a.schema.v, b.data(), static_cast<u32>(a.binding.size())));
                }
                return out;
            },
            "i"_a, "The plan of search i (empty unless solved).")
        .def(
            "goal_state",
            [](const PyIwBatch& x, i64 i) -> Arg<std::optional<PyState>> {
                const u32 k = checked_index(x, i);
                if (x.b.status[k] != search::SearchStatus::Solved)
                    return Arg<std::optional<PyState>>(nb::none());
                const u64* r = x.b.goal_rows.data() + u64{k} * (x.b.words + x.b.numeric_words);
                return Arg<std::optional<PyState>>(make_state(owner_of_search(x, k),
                    State(r, bits::trimmed_size(r, x.b.words), x.b.numeric_words ? r + x.b.words : nullptr, x.b.numeric_words)));
            },
            "i"_a, "The goal state search i reached, or None.")
        .def(
            "passes",
            [](const PyIwBatch& x, i64 i) {
                const std::span<const search::IwPassStatistics> p = x.b.passes(checked_index(x, i));
                return std::vector<search::IwPassStatistics>(p.begin(), p.end());
            },
            "i"_a,
            "The IW passes of search i (mymyr.search.IwPass; as mymyr.search.IwResult.passes).")
        .def(
            "cost",
            [](const PyIwBatch& x, i64 i) {
                const u32 k = checked_index(x, i);
                if (x.b.status[k] != search::SearchStatus::Solved)
                    return 0.0;
                return x.b.result(k, *owner_of_search(x, k).core->task, start_of(x, k), true).cost;
            },
            "i"_a, "The plan cost of search i (mymyr.search.IwResult.cost; 0 unless solved).")
        .def(
            "reached_slots",
            [](const PyIwBatch& x, i64 i) {
                const u32 k = checked_index(x, i);
                std::vector<u32> out;
                if (!x.b.reached.empty())
                    bits::for_each(x.b.reached.data() + u64{k} * x.b.reached_words, x.b.reached_words,
                                   [&](u64 s) { out.push_back(static_cast<u32>(s)); });
                return out;
            },
            "i"_a, "Rollouts: the fluent slots of every state rollout i created (ascending); empty for other searches.")
        .def(
            "reached_atoms",
            [](const PyIwBatch& x, i64 i) {
                const u32 k = checked_index(x, i);
                return cuda::reached_atoms(x.b, k, *owner_of_search(x, k).core->task);
            },
            "i"_a,
            "Rollouts: the canonical ids of the reached fluent atoms of rollout i (ascending; "
            "mymyr.search rollouts' reached_fluent_atoms).")
        .def_prop_ro("task_ids",
                     [](const PyIwBatch& x) { return x.ids.empty() ? std::vector<u32>(x.b.n, 0) : x.ids; },
                     "per search: its instance of the table (0 for a task)")
        .def_prop_ro("stats",
                     [](const PyIwBatch& x) {
                         const cuda::MultiIwStats& s = x.b.stats;
                         nb::typed<nb::dict, std::string, std::variant<double, u64>> d{nb::dict()};
                         d["seconds"] = s.seconds;
                         d["host_ms"] = s.host_ms;
                         d["groups"] = s.groups;
                         d["passes"] = s.passes;
                         d["layers"] = s.layers;
                         d["chunks"] = s.chunks;
                         d["splits"] = s.splits;
                         d["redone"] = s.redone;
                         d["distinct"] = s.distinct;
                         d["replays"] = s.replays;
                         d["device_loops"] = s.device_loops;
                         d["loop_handoffs"] = s.loop_handoffs;
                         d["nodes"] = s.nodes;
                         d["candidates"] = s.candidates;
                         d["uploads"] = u64{s.uploads};
                         d["widenings"] = u64{s.widenings};
                         d["host_schemas"] = u64{s.host_schemas};
                         d["device_bytes"] = s.device_bytes;
                         return d;
                     },
                     "device statistics of the call")
        .def("__repr__", [](const PyIwBatch& x) {
            u64 solved = 0;
            for (u32 i = 0; i < x.b.n; ++i)
                solved += x.b.status[i] == search::SearchStatus::Solved;
            return std::string(x.rollouts ? "IwBatch(rollouts=" : "IwBatch(searches=") + std::to_string(x.b.n) +
                   ", solved=" + std::to_string(solved) + ")";
        });

#define MYMYR_MULTI_IW_COMMON_ARGS                                                                                            \
    "max_arity"_a = 1, "width_zero"_a = "expand_depth_one", "optimize_iw1"_a = true, "witness_pruning"_a = false,        \
        "canonical_order"_a = true, "exact"_a = true, "max_states"_a = nb::none(), "max_expanded"_a = nb::none(),       \
        "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(), "max_searches"_a = nb::none(),                        \
        "chunk_states"_a = nb::none()

    m.def(
        "multi_iw",
        [](TableArg task, StartsArg starts, ContextArg ctx, GoalsArg goals, u32 max_arity, StrArg width_zero,
           bool optimize_iw1, bool witness_pruning, bool canonical_order, bool exact, IntArg max_states,
           IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, IntArg max_searches, IntArg chunk_states,
           TaskIdsArg task_ids) {
            const cuda::MultiIwOptions opts = options_of(max_arity, width_zero, optimize_iw1, witness_pruning, canonical_order,
                                                         exact, max_states, max_expanded, max_depth, max_seconds,
                                                         max_searches, chunk_states);
            PyIwBatch x;
            if (nb::isinstance<PyTable>(task))
            {
                // over a table: search i on instance task_ids[i]
                const TableRef ref = table_of(task);
                const cuda::ContextPtr c = table_device_context(ref.obj, ctx, 0);
                for (u32 i = 0; i < ref.table->size(); ++i)
                    x.owners.push_back(ref.owner(i));
                x.starts = table_starts(starts, *ref.table);
                x.ids = search_task_ids(task_ids, x.starts.size(), *ref.table, c->stream(),
                                        reinterpret_cast<std::intptr_t>(c->stream()));
                check_start_owners(starts, *ref.table, x.ids);
                const std::vector<search::GoalSpec::AtomGoal> g = table_goals(goals, *ref.table, x.ids);
                nb::gil_scoped_release release;
                cuda::DeviceTableIw run(c, ref.table, opts);
                x.b = run.run(x.starts, x.ids, g);
                return x;
            }
            if (!task_ids.is_none())
                throw nb::value_error("mymyr: task_ids go with a TaskTable (a task runs every search itself)");
            const Owner o = owner_of(task);
            const TaskPtr& t = o.core->task;
            const cuda::ContextPtr c = g_lookup(*o.core, ctx, 0);
            x.owners.push_back(o);
            x.starts = host_starts(starts, *t);
            const std::vector<search::GoalSpec::AtomGoal> g = goals_of(goals, *t, x.starts.size());
            {
                nb::gil_scoped_release release;
                cuda::DeviceMultiIw run(c, t, opts);
                x.b = run.run(x.starts, g);
            }
            return x;
        },
        "task"_a, "starts"_a, nb::kw_only(), "ctx"_a = nb::none(), "goals"_a = nb::none(), MYMYR_MULTI_IW_COMMON_ARGS,
        "task_ids"_a = nb::none(),
        "search.iw from every start state, all searches at once on the device: search i equals "
        "mymyr.search.iw(task, start=starts[i], ...) (status, plan, goal state, per-pass counts) for every group and "
        "chunk size (exact batch novelty; exact=False is relaxed novelty: valid IW, but the kept states depend on "
        "timing). goals: None (the task's goal) or one (positive slots, negative slots) per start. Budgets apply per "
        "search and pass as in search.iw; max_seconds spans the call. Numeric tasks raise ValueError. "
        "Over a mymyr.rl.TaskTable: search i runs on instance task_ids[i] (starts: States of their "
        "instances, or rows of the table's width; goals in the instance's slots) and equals the search of that "
        "instance's task; the searches of an instance run together, the instances one after the other.");

    m.def(
        "rollouts",
        [](TaskArg task, SeedsArg seeds, ContextArg ctx, StateArg start, Arg<GoalArg> goal,
           IntArg max_next_layer_states, u32 max_arity, StrArg width_zero, bool optimize_iw1, bool witness_pruning,
           bool canonical_order, bool exact, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds, IntArg max_searches, IntArg chunk_states) {
            const Owner o = owner_of(task);
            const TaskPtr& t = o.core->task;
            const cuda::ContextPtr c = g_lookup(*o.core, ctx, 0);
            cuda::MultiIwOptions opts = options_of(max_arity, width_zero, optimize_iw1, witness_pruning, canonical_order,
                                                   exact, max_states, max_expanded, max_depth, max_seconds, max_searches,
                                                   chunk_states);
            if (!max_next_layer_states.is_none())
            {
                opts.max_next_layer_states = nb::cast<u32>(max_next_layer_states);
                if (opts.max_next_layer_states == 0)
                    throw nb::value_error("mymyr: max_next_layer_states must be positive");
            }
            std::vector<u64> s;
            for (nb::handle v : seeds)
                s.push_back(nb::cast<u64>(v));
            PyIwBatch x;
            x.owners.push_back(o);
            x.rollouts = true;
            if (!start.is_none() && !is_state(start))
                throw nb::type_error("mymyr: start must be a State or None");
            x.starts.push_back(start.is_none() ? t->initial_state() : state_of(start).s);
            std::vector<search::GoalSpec::AtomGoal> g;
            if (!goal.is_none())
                g.assign(s.size(), goal_of(goal, *t));
            {
                nb::gil_scoped_release release;
                cuda::DeviceRollouts r(c, t, opts);
                x.b = r.run(s, &x.starts[0], g);
            }
            return x;
        },
        "task"_a, "seeds"_a, nb::kw_only(), "ctx"_a = nb::none(), "start"_a = nb::none(), "goal"_a = nb::none(),
        "max_next_layer_states"_a = nb::none(), MYMYR_MULTI_IW_COMMON_ARGS,
        "Randomized IW rollouts on the device, one per seed: rollout k equals the CPU rollout of the same seed "
        "(search/parallel_rollouts.hpp: every next layer shuffled by the rollout's SplitMix64 stream, "
        "max_next_layer_states truncates it): status, plan, per-pass counts and the reached fluent atoms "
        "(IwBatch.reached_atoms). goal: None (the task's goal) or (positive slots, negative slots). Not reported: "
        "derived atoms, landing states, co-occurrence.");

    m.def(
        "batched_iw1",
        [](TableArg task, StartsArg starts, ContextArg ctx, StreamArg stream, GoalsArg goals, bool exact,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, IntArg max_searches,
           IntArg chunk_states, TaskIdsArg task_ids) {
            cuda::MultiIwOptions opts = options_of(1, nb::str("expand_depth_one"), true, false, true, exact, max_states,
                                                   max_expanded, max_depth, max_seconds, max_searches, chunk_states);
            PyIwBatch x;
            if (nb::isinstance<PyTable>(task))
            {
                const TableRef ref = table_of(task);
                for (u32 i = 0; i < ref.table->size(); ++i)
                    x.owners.push_back(ref.owner(i));
                if (!is_cuda_array(starts))
                {
                    const cuda::ContextPtr c = table_device_context(ref.obj, ctx, 0);
                    x.starts = table_starts(starts, *ref.table);
                    x.ids = search_task_ids(task_ids, x.starts.size(), *ref.table, c->stream(),
                                            reinterpret_cast<std::intptr_t>(c->stream()));
                    check_start_owners(starts, *ref.table, x.ids);
                    const std::vector<search::GoalSpec::AtomGoal> g = table_goals(goals, *ref.table, x.ids);
                    nb::gil_scoped_release release;
                    cuda::DeviceTableIw run(c, ref.table, opts);
                    x.b = run.run(x.starts, x.ids, g);
                    return x;
                }
                const dl::Device d = dl::dlpack_device(starts);
                const cuda::ContextPtr c = table_device_context(ref.obj, ctx, d.device_id);
                if (d.device_id != c->device())
                    throw nb::value_error(("mymyr: 'starts' lives on cuda:" + std::to_string(d.device_id) +
                                           ", the context on cuda:" + std::to_string(c->device()))
                                              .c_str());
                std::intptr_t dl_stream = 0;
                const cudaStream_t st = stream_value(stream, c->stream(), dl_stream);
                const dl::Imported im = dl::import_dlpack(starts, dl_stream);
                const WordsLayout w = words_layout(im.data, im.dtype.code, im.dtype.bits, im.dtype.lanes, im.shape.size(),
                                                   im.shape.data(), im.strides.data());
                if (w.rows > 0xFFFFFFFFull)
                    throw nb::value_error("mymyr: more than 2^32 start states");
                x.ids = search_task_ids(task_ids, w.rows, *ref.table, st, dl_stream);
                const std::vector<search::GoalSpec::AtomGoal> g = table_goals(goals, *ref.table, x.ids);
                {
                    nb::gil_scoped_release release;
                    cuda::DeviceTableIw run(c, ref.table, opts);
                    x.b = run.run(cuda::DeviceStarts{w.data, w.stride, w.words, static_cast<u32>(w.rows)}, x.ids, g, {}, st);
                    // the start states, for the plan costs (run() synchronized: the rows are final)
                    const cuda::DeviceGuard guard(c->device());
                    std::vector<u64> rows(w.rows * w.words);
                    if (!rows.empty())
                        cuda::check(cudaMemcpy2D(rows.data(), w.words * sizeof(u64), w.data, w.stride * sizeof(u64),
                                                 w.words * sizeof(u64), w.rows, cudaMemcpyDeviceToHost),
                                    "cudaMemcpy2D (starts)");
                    x.starts.reserve(w.rows);
                    for (u64 i = 0; i < w.rows; ++i)
                        x.starts.emplace_back(rows.data() + i * w.words, bits::trimmed_size(rows.data() + i * w.words, w.words));
                }
                return x;
            }
            if (!task_ids.is_none())
                throw nb::value_error("mymyr: task_ids go with a TaskTable (a task runs every search itself)");
            const Owner o = owner_of(task);
            const TaskPtr& t = o.core->task;
            x.owners.push_back(o);
            if (!is_cuda_array(starts))
            {
                const cuda::ContextPtr c = g_lookup(*o.core, ctx, 0);
                x.starts = host_starts(starts, *t);
                const std::vector<search::GoalSpec::AtomGoal> g = goals_of(goals, *t, x.starts.size());
                nb::gil_scoped_release release;
                cuda::DeviceMultiIw run(c, t, opts);
                x.b = run.run(x.starts, g);
                return x;
            }
            // device rows, read in place on the caller's stream
            const dl::Device d = dl::dlpack_device(starts);
            const cuda::ContextPtr c = g_lookup(*o.core, ctx, d.device_id);
            if (d.device_id != c->device())
                throw nb::value_error(("mymyr: 'starts' lives on cuda:" + std::to_string(d.device_id) + ", the context on cuda:" +
                                       std::to_string(c->device()))
                                          .c_str());
            std::intptr_t dl_stream = 0;
            const cudaStream_t st = stream_value(stream, c->stream(), dl_stream);
            const dl::Imported im = dl::import_dlpack(starts, dl_stream);
            const WordsLayout w = words_layout(im.data, im.dtype.code, im.dtype.bits, im.dtype.lanes, im.shape.size(),
                                               im.shape.data(), im.strides.data());
            const std::vector<search::GoalSpec::AtomGoal> g = goals_of(goals, *t, w.rows);
            if (w.rows > 0xFFFFFFFFull)
                throw nb::value_error("mymyr: more than 2^32 start states");
            {
                nb::gil_scoped_release release;
                cuda::DeviceMultiIw run(c, t, opts);
                x.b = run.run(cuda::DeviceStarts{w.data, w.stride, w.words, static_cast<u32>(w.rows), t->numeric_words()}, g, {}, st);
                // the start states, for the plan costs (run() synchronized: the rows are final)
                const cuda::DeviceGuard guard(c->device());
                std::vector<u64> rows(w.rows * w.words);
                if (!rows.empty())
                    cuda::check(cudaMemcpy2D(rows.data(), w.words * sizeof(u64), w.data, w.stride * sizeof(u64),
                                             w.words * sizeof(u64), w.rows, cudaMemcpyDeviceToHost),
                                "cudaMemcpy2D (starts)");
                x.starts.reserve(w.rows);
                for (u64 i = 0; i < w.rows; ++i)
                    x.starts.emplace_back(rows.data() + i * w.words, w.words - t->numeric_words(),
                                          t->numeric_words() ? rows.data() + (i + 1) * w.words - t->numeric_words() : nullptr,
                                          t->numeric_words());
            }
            return x;
        },
        "task"_a, "starts"_a, nb::kw_only(), "ctx"_a = nb::none(), "stream"_a = nb::none(), "goals"_a = nb::none(),
        "exact"_a = true, "max_states"_a = nb::none(), "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(),
        "max_seconds"_a = nb::none(), "max_searches"_a = nb::none(), "chunk_states"_a = nb::none(),
        "task_ids"_a = nb::none(),
        "Exact batched IW(1): optimized IW(1) from each start state, all at once on the "
        "device. starts: host states, or a CUDA word array [N, W] uint64 / [N, 2W] uint32 (torch, JAX, DLPack) read in "
        "place after the work on `stream` (None: the context's stream). Per search: status, plan_length, goal_state "
        "(IwBatch); equal to mymyr.search.iw(task, max_arity=1, start=...). Over a mymyr.rl.TaskTable: "
        "search i runs on instance task_ids[i] (host ints, or an int32 CUDA array read after the work on `stream`); "
        "device starts are rows of at most the table's width.");
#undef MYMYR_MULTI_IW_COMMON_ARGS
}
}  // namespace mymyr::python
