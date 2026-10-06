// mymyr._core._rl_ops: the native side of mymyr.rl's RL helpers:
//   novelty_update(seen, states, reward)                          width-1 novelty rewards (rl/novelty.hpp)
//   prefix_masks(parent, schema, binding, query, prefix, depth, mask)   factored action masks (rl/prefix.hpp)
//   schema_masks(parent, schema, mask)                            the first factored decision
//   her_relabel(states, done, goal, source, achieved, reward, ...)      hindsight relabels (rl/her.hpp)
// Destination-passing: every array is the caller's, read and written in place through DLPack. Host arrays run
// the host loops; CUDA arrays (all on one device) the device kernels of cuda/rl_ops.hpp, enqueued on `stream` (None:
// the legacy default stream; torch passes its current stream) with the array API's stream handoff and no host
// synchronization. mymyr.rl (python/mymyr/rl/_ops.py) allocates the outputs in the inputs' framework; JAX arrays take
// the jnp versions of mymyr.rl.jax instead (JAX arrays are immutable).

#include "arrays.hpp"
#include "dlpack.hpp"
#include "rl_imports.hpp"
#include "rl_typing.hpp"
#include "typing.hpp"

#include "mymyr/rl/her.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"

#if defined(MYMYR_HAS_CUDA)
#include "mymyr/cuda/rl_ops.hpp"
#include "mymyr/cuda/runtime.hpp"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <string>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

namespace
{
using namespace rlimp;
using ArrayArg = Arg<ann::ArrayLike>;

/// The device of a call (-1: the host) from its first array, and its stream as a DLPack stream value.
struct Place
{
    int device = -1;
    std::intptr_t stream = dl::k_stream_default;
};

Place place_of(nb::handle first, nb::handle stream)
{
    Place p;
    const dl::Device d = dl::dlpack_device(first);
    if (d.device_type < 0)
        throw nb::type_error("mymyr: the arrays must speak DLPack");
    if (dl::host_accessible(d.device_type))
    {
        if (!stream.is_none())
            throw nb::value_error("mymyr: stream is for CUDA arrays (these are host arrays)");
        return p;
    }
    if (d.device_type != dl::k_cuda)
        throw nb::type_error("mymyr: the arrays must be host or CUDA arrays");
#if defined(MYMYR_HAS_CUDA)
    p.device = d.device_id;
    if (stream.is_none())
        p.stream = dl::k_stream_legacy;
    else
    {
        nb::object h = nb::getattr(stream, "cuda_stream", nb::none());
        p.stream = nb::cast<std::intptr_t>(h.is_none() ? nb::borrow(stream) : h);
        if (p.stream == dl::k_stream_none)
            throw nb::value_error("mymyr: stream -1 means 'no synchronization'; pass a stream to run on");
        if (p.stream == 0)
            p.stream = dl::k_stream_legacy;  // torch reports its default stream as 0
    }
    return p;
#else
    throw nb::value_error("mymyr: CUDA arrays need a mymyr built with the CUDA backend");
#endif
}

#if defined(MYMYR_HAS_CUDA)
/// Runs a launcher on the call's device and stream (DLPack stream values 1 / 2 are cudaStreamLegacy /
/// cudaStreamPerThread as cudaStream_t).
template<class F>
void launch(const Place& p, const char* what, F&& f)
{
    nb::gil_scoped_release release;
    cuda::DeviceGuard g(p.device);
    cuda::check(f(reinterpret_cast<cudaStream_t>(p.stream)), what);
}
#endif

i64 dim(const std::vector<i64>& ext, usize k) { return ext.at(k); }

void novelty_update(ArrayArg seen, ArrayArg states, ArrayArg reward, StreamArg stream)
{
    const Place pl = place_of(seen, stream);
    Imports in(pl.device, pl.stream, false);
    std::vector<i64> lead;
    u32 W = 0, SW = 0;
    u64* s = in.words_nd(seen, "seen", {-1}, true, lead, W);
    const i64 n = lead.at(0);
    std::vector<i64> lead2;
    const u64* x = in.words_nd(states, "states", {n}, false, lead2, SW);
    if (SW < W)
        throw nb::value_error("mymyr: 'states' rows are narrower than the 'seen' rows");
    i32* r = in.get<i32>(reward, "reward", {i32t}, {n}, true);
    const u64 rows = static_cast<u64>(n);
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "novelty_update", [&](cudaStream_t cs) { return cuda::launch_novelty_update(s, x, rows, W, SW, r, cs); });
#endif
    nb::gil_scoped_release release;
    rl::novelty_update(s, x, rows, W, SW, r);
}

rl::LabelRows label_rows(Imports& in, nb::handle parent, nb::handle schema, nb::handle binding)
{
    std::vector<i64> ext;
    rl::LabelRows x;
    x.parent = in.get<const i32>(parent, "parent", {i32t}, {-1}, false, &ext);
    const i64 m = ext.at(0);
    x.size = static_cast<u64>(m);
    x.schema = in.get<const i32>(schema, "schema", {i32t}, {m}, false);
    if (!binding.is_none())
    {
        x.binding = in.get<const i32>(binding, "binding", {i32t}, {m, -1}, false, &ext);
        x.label_width = static_cast<u32>(ext.at(1));
    }
    return x;
}

void prefix_masks(ArrayArg parent, ArrayArg schema, ArrayArg binding, ArrayArg query, ArrayArg prefix, u32 depth,
                  ArrayArg mask, StreamArg stream)
{
    const Place pl = place_of(mask, stream);
    Imports in(pl.device, pl.stream, false);
    const rl::LabelRows x = label_rows(in, parent, schema, binding);
    std::vector<i64> ext;
    u8* m = in.get<u8>(mask, "mask", {boolt, u8t}, {-1, -1}, true, &ext);
    const i64 n = dim(ext, 0);
    rl::PrefixQuery q;
    q.rows = static_cast<u64>(n);
    q.num_objects = static_cast<u32>(dim(ext, 1));
    q.depth = depth;
    q.schema = in.get<const i32>(query, "query", {i32t}, {n}, false);
    if (!q.schema)
        throw nb::type_error("mymyr: 'query' (the chosen schemas) is required");
    if (depth > 0)
    {
        q.prefix = in.get<const i32>(prefix, "prefix", {i32t}, {n, -1}, false, &ext);
        if (!q.prefix || ext.at(1) < static_cast<i64>(depth))
            throw nb::value_error("mymyr: 'prefix' must be [N, >= depth] int32");
        q.prefix_stride = static_cast<u32>(ext.at(1));
    }
    if (!x.binding)
        throw nb::type_error("mymyr: 'binding' is required");
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "prefix_masks", [&](cudaStream_t cs) { return cuda::launch_prefix_masks(x, q, m, cs); });
#endif
    nb::gil_scoped_release release;
    rl::prefix_masks(x, q, m);
}

void schema_masks(ArrayArg parent, ArrayArg schema, ArrayArg mask, StreamArg stream)
{
    const Place pl = place_of(mask, stream);
    Imports in(pl.device, pl.stream, false);
    const rl::LabelRows x = label_rows(in, parent, schema, nb::none());
    std::vector<i64> ext;
    u8* m = in.get<u8>(mask, "mask", {boolt, u8t}, {-1, -1}, true, &ext);
    const u64 n = static_cast<u64>(dim(ext, 0));
    const u32 S = static_cast<u32>(dim(ext, 1));
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "schema_masks", [&](cudaStream_t cs) { return cuda::launch_schema_masks(x, n, S, m, cs); });
#endif
    nb::gil_scoped_release release;
    rl::schema_masks(x, n, S, m);
}

rl::HerStrategy her_strategy(const std::string& s)
{
    if (s == "future")
        return rl::HerStrategy::Future;
    if (s == "final")
        return rl::HerStrategy::Final;
    if (s == "episode")
        return rl::HerStrategy::Episode;
    throw nb::value_error("mymyr: strategy must be 'future', 'final' or 'episode'");
}

void her_relabel(ArrayArg states, ArrayArg done, ArrayArg goal, ArrayArg source, ArrayArg achieved, ArrayArg reward,
                 Arg<ann::HerStrategy> strategy, u32 subset, u64 seed, u64 first_env, ArrayArg env_ids, ArrayArg goal_atoms,
                 f32 step_reward, f32 goal_reward, StreamArg stream)
{
    const Place pl = place_of(states, stream);
    Imports in(pl.device, pl.stream, false);
    rl::HerConfig c;
    c.strategy = her_strategy(nb::cast<std::string>(strategy));
    c.subset = subset;
    c.seed = seed;
    c.first_env = first_env;
    c.step_reward = step_reward;
    c.goal_reward = goal_reward;
    rl::HerBatch b;
    std::vector<i64> lead;
    u32 RW = 0, W = 0;
    b.states = in.words_nd(states, "states", {-1, -1}, false, lead, RW);
    const i64 T = lead.at(0), N = lead.at(1);
    b.steps = static_cast<u64>(T);
    b.envs = static_cast<u64>(N);
    b.row_words = RW;
    b.done = in.get<const u8>(done, "done", {boolt, u8t}, {T, N}, false);
    std::vector<i64> glead;
    b.goal = in.words_nd(goal, "goal", {T, N, -1}, true, glead, W);
    if (!b.goal || !b.done)
        throw nb::type_error("mymyr: 'done' and 'goal' are required");
    if (W > RW)
        throw nb::value_error("mymyr: 'goal' rows are wider than the state rows");
    b.words = W;
    const i64 K = glead.at(2);
    if (K < 1)
        throw nb::value_error("mymyr: 'goal' must be [T, N, k, W] with k >= 1");
    c.k = static_cast<u32>(K);
    b.source = in.get<i32>(source, "source", {i32t}, {T, N, K}, true);
    b.achieved = in.get<u8>(achieved, "achieved", {boolt, u8t}, {T, N, K}, true);
    b.reward = in.get<f32>(reward, "reward", {f32t}, {T, N, K}, true);
    b.env_ids = in.get<const u32>(env_ids, "env_ids", {u32t, i32t}, {N}, false);
    if (!goal_atoms.is_none())
    {
        std::vector<i64> alead;
        u32 AW = 0;
        const nb::object shape = nb::getattr(goal_atoms, "shape", nb::none());
        const bool per_env = !shape.is_none() && nb::len(shape) == 2;
        b.goal_atoms = per_env ? in.words_nd(goal_atoms, "goal_atoms", {N}, false, alead, AW)
                               : in.words_nd(goal_atoms, "goal_atoms", {}, false, alead, AW);
        if (AW != W)
            throw nb::value_error("mymyr: 'goal_atoms' must have the goal's width");
        b.goal_atoms_stride = per_env ? W : 0;
    }
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "her_relabel", [&](cudaStream_t cs) { return cuda::launch_her_relabel(b, c, cs); });
#endif
    nb::gil_scoped_release release;
    rl::her_relabel(b, c);
}
}  // namespace

void bind_rl_ops(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_rl_ops", "Novelty rewards, prefix masks and hindsight relabels (mymyr.rl)");
    m.def("novelty_update", &novelty_update, "seen"_a, "states"_a, "reward"_a = nb::none(), nb::kw_only(),
          "stream"_a = nb::none(),
          "Width-1 novelty update in place (rl/novelty.hpp): reward[i] = popcount(states[i] & ~seen[i]) over the first S "
          "words, then seen[i] |= states[i]. seen [N, S] and states [N, >= S] are 64-bit words (or 32-bit halves); "
          "reward [N] int32 (None: not written). Host or CUDA arrays.");
    m.def("prefix_masks", &prefix_masks, "parent"_a, "schema"_a, "binding"_a, "query"_a, "prefix"_a.none(), "depth"_a, "mask"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          "Factored action masks (rl/prefix.hpp) into mask [N, num_objects] (bool), mask[i, o] set iff a successor "
          "row j (parent [M], schema [M], binding [M, L] int32; parent -1 = padding) of state i is labelled "
          "(query[i], prefix[i, :depth], o, ...). query [N] int32, prefix [N, >= depth] int32 (None for depth 0).");
    m.def("schema_masks", &schema_masks, "parent"_a, "schema"_a, "mask"_a, nb::kw_only(), "stream"_a = nb::none(),
          "The schema masks [N, num_schemas] (bool), mask[i, s] set iff a successor of state i is labelled with schema "
          "s.");
    m.def("her_relabel", &her_relabel, "states"_a, "done"_a, "goal"_a, "source"_a = nb::none(), "achieved"_a = nb::none(),
          "reward"_a = nb::none(), nb::kw_only(), "strategy"_a = "future", "subset"_a = 0, "seed"_a = 0,
          "first_env"_a = 0, "env_ids"_a = nb::none(), "goal_atoms"_a = nb::none(), "step_reward"_a = -1.0f,
          "goal_reward"_a = 0.0f, "stream"_a = nb::none(),
          "Hindsight relabels (rl/her.hpp) of states [T, N, RW] (64-bit words or 32-bit halves; the reached states) "
          "and done [T, N] into goal [T, N, k, W] (k relabels, W <= RW goal words), source [T, N, k] int32, achieved "
          "[T, N, k] bool and reward [T, N, k] float32. subset 0: every atom of the source state (restricted to "
          "goal_atoms [W], or [N, W]: env i's own, e.g. its instance's fluent atoms); env_ids [N] uint32 (None: first_env + "
          "i) name the RNG streams.");
}
}  // namespace mymyr::python
