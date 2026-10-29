// mymyr._core._rl_ops: the native side of mymyr.rl's RL helpers:
//   novelty_update(seen, states, reward)                          width-1 novelty rewards (rl/novelty.hpp)
//   prefix_masks(parent, schema, binding, query, prefix, depth, mask)   factored action masks (rl/prefix.hpp)
//   schema_masks(parent, schema, mask)                            the first factored decision
//   her_relabel(states, done, goal, source, achieved, reward, ...)      hindsight relabels (rl/her.hpp)
//   visited_mask(table, count, succ_hash, parent, rows, out)      visited-table membership (rl/visited.hpp)
//   target_visited(target, role, ref, ..., out)                   visited successors of action targets (rl/visited.hpp)
// Destination-passing: every array is the caller's, read and written in place through DLPack, or as a tensor view
// (rl_imports.hpp: the torch wrappers pass views, which skip the cost of a DLPack export). Host arrays run the host
// loops; CUDA arrays (all on one device) the device kernels of cuda/rl_ops.hpp, enqueued on `stream` (None: the legacy
// default stream; torch passes its current stream) with the array API's stream handoff and no host synchronization.
// mymyr.rl (python/mymyr/rl/_ops.py) allocates the outputs in the inputs' framework; JAX arrays take the jnp versions
// of mymyr.rl.jax instead (JAX arrays are immutable).

#include "arrays.hpp"
#include "dlpack.hpp"
#include "rl_imports.hpp"
#include "py_pool.hpp"
#include "rl_typing.hpp"
#include "typing.hpp"

#include "mymyr/encoding/schema.hpp"
#include "mymyr/rl/her.hpp"
#include "mymyr/rl/goal_templates.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"
#include "mymyr/rl/visited.hpp"

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

std::shared_ptr<const rl::GoalTemplates> goal_templates_of(nb::handle object);

namespace
{
using namespace rlimp;
using ArrayArg = Arg<ann::ArrayOrView>;

/// The device of a call (-1: the host) from its first array, and its stream as a DLPack stream value.
struct Place
{
    int device = -1;
    std::intptr_t stream = dl::k_stream_default;
};

Place place_of(nb::handle first, nb::handle stream)
{
    Place p;
    const dl::Device d = device_of(first);
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

void visited_mask(ArrayArg table, ArrayArg count, ArrayArg succ_hash, ArrayArg parent, ArrayArg rows, ArrayArg out,
                  StreamArg stream)
{
    if (table.is_none() || count.is_none() || succ_hash.is_none() || parent.is_none() || out.is_none())
        throw nb::type_error("mymyr: visited_mask requires table, count, succ_hash, parent and out arrays");
    const Place pl = place_of(table, stream);
    Imports in(pl.device, pl.stream, false);
    std::vector<i64> ext, hashes, mapping;
    const u64* tab = in.get<const u64>(table, "table", {u64t, i64t}, {-1, -1}, false, &ext);
    const i64 n = ext[0], H = ext[1];
    if (H < 2 || H > (1 << 20) || (H & (H - 1)))
        throw nb::value_error("mymyr: visited capacity must be a power of two in [2, 2^20]");
    const i32* cnt = in.get<const i32>(count, "count", {i32t}, {n}, false);
    const u64* hash = in.get<const u64>(succ_hash, "succ_hash", {u64t, i64t}, {-1}, false, &hashes);
    const i64 size = hashes[0];
    const i32* p = in.get<const i32>(parent, "parent", {i32t}, {size}, false);
    const i32* map = in.get<const i32>(rows, "rows", {i32t}, {-1}, false, &mapping);
    u8* dst = in.get<u8>(out, "out", {boolt}, {size}, true);
    const i64 parents = rows.is_none() ? n : mapping[0];
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "visited_mask", [&](cudaStream_t cs) {
            return cuda::launch_visited_mask(tab, cnt, static_cast<u64>(n), static_cast<u32>(H), hash, p, map,
                                             static_cast<u64>(parents), static_cast<u64>(size), dst, cs);
        });
#endif
    nb::gil_scoped_release release;
    for (i64 j = 0; j < size; ++j)
        if (p[j] < -1 || p[j] >= parents || (p[j] >= 0 && map && (map[p[j]] < -1 || map[p[j]] >= n)))
            throw std::invalid_argument("mymyr: visited_mask: parent or mapped row is outside the table");
    rl::visited_mask(tab, cnt, static_cast<u32>(H), hash, p, map, static_cast<u64>(size), dst);
}

/// The rest of target_visited, with the index type I of target, ref and entity_offsets (x.target is set).
template<class I>
void target_visited_as(Imports& in, const Place& pl, rl::TargetRows<I>& x, nb::handle ref, nb::handle entity_offsets,
                       i64 row_graphs, nb::handle env_rows, const u64* tab, const i32* cnt, u64 n, u32 H, u8* dst)
{
    constexpr DType idx = sizeof(I) == 4 ? i32t : i64t;
    x.ref = in.get<const I>(ref, "ref", {idx}, {static_cast<i64>(x.entities)}, false);
    std::vector<i64> ext;
    x.entity_offsets = in.get<const I>(entity_offsets, "entity_offsets", {idx}, {-1}, false, &ext);
    if (ext[0] < 1 || row_graphs < ext[0])
        throw nb::value_error("mymyr: 'entity_offsets' and 'row_offsets' need an entry per graph and one more");
    x.graphs = static_cast<u64>(ext[0] - 1);
    x.env_rows = in.get<const i32>(env_rows, "env_rows", {i32t}, {static_cast<i64>(x.graphs)}, false);
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
        return launch(pl, "target_visited",
                      [&](cudaStream_t cs) { return cuda::launch_target_visited(x, tab, cnt, n, H, dst, cs); });
#endif
    nb::gil_scoped_release release;
    rl::target_visited(x, tab, cnt, n, H, dst);
}

template<class I>
void target_visited_fill(Imports& in, rl::TargetRows<I>& x, const void* target, i64 targets, nb::handle role,
                         nb::handle live, nb::handle row_offsets, nb::handle row_index, const u64* hash, i64 rows,
                         nb::handle status)
{
    std::vector<i64> ext;
    x.target = static_cast<const I*>(target);
    x.targets = static_cast<u64>(targets);
    x.live = in.get<const i64>(live, "live", {i64t}, {1}, false);
    x.role = in.get<const u8>(role, "role", {u8t}, {-1}, false, &ext);
    x.entities = static_cast<u64>(ext[0]);
    x.action_role = static_cast<u8>(encoding::EntityRole::Action);
    x.row_offsets = in.get<const i32>(row_offsets, "row_offsets", {i32t}, {-1}, false);
    x.row_index = in.get<const i32>(row_index, "row_index", {i32t}, {rows}, false);
    x.succ_hash = hash;
    x.rows = static_cast<u64>(rows);
    x.status = in.get<const i32>(status, "status", {i32t}, {1}, false);
}

void target_visited(ArrayArg target, ArrayArg role, ArrayArg ref, ArrayArg entity_offsets, ArrayArg row_offsets,
                    ArrayArg succ_hash, ArrayArg table, ArrayArg count, ArrayArg live, ArrayArg row_index,
                    ArrayArg status, ArrayArg env_rows, ArrayArg out, StreamArg stream)
{
    if (target.is_none() || role.is_none() || ref.is_none() || entity_offsets.is_none() || row_offsets.is_none() ||
        succ_hash.is_none() || table.is_none() || count.is_none() || out.is_none())
        throw nb::type_error("mymyr: target_visited requires target, role, ref, entity_offsets, row_offsets, "
                             "succ_hash, table, count and out arrays");
    const Place pl = place_of(table, stream);
    Imports in(pl.device, pl.stream, false);
    std::vector<i64> ext, targets, rows, graphs;
    const u64* tab = in.get<const u64>(table, "table", {u64t, i64t}, {-1, -1}, false, &ext);
    const i64 n = ext[0], H = ext[1];
    if (H < 2 || H > (1 << 20) || (H & (H - 1)))
        throw nb::value_error("mymyr: visited capacity must be a power of two in [2, 2^20]");
    const i32* cnt = in.get<const i32>(count, "count", {i32t}, {n}, false);
    DType index{};
    const void* tgt = in.get<const void>(target, "target", {i32t, i64t}, {-1}, false, &targets, &index);
    const u64* hash = in.get<const u64>(succ_hash, "succ_hash", {u64t, i64t}, {-1}, false, &rows);
    (void)in.get<const i32>(row_offsets, "row_offsets", {i32t}, {-1}, false, &graphs);
    u8* dst = in.get<u8>(out, "out", {boolt}, {targets[0]}, true);
    if (index.bits == 32)
    {
        rl::TargetRows<i32> x;
        target_visited_fill(in, x, tgt, targets[0], role, live, row_offsets, row_index, hash, rows[0], status);
        return target_visited_as(in, pl, x, ref, entity_offsets, graphs[0], env_rows, tab, cnt, static_cast<u64>(n),
                                 static_cast<u32>(H), dst);
    }
    rl::TargetRows<i64> x;
    target_visited_fill(in, x, tgt, targets[0], role, live, row_offsets, row_index, hash, rows[0], status);
    target_visited_as(in, pl, x, ref, entity_offsets, graphs[0], env_rows, tab, cnt, static_cast<u64>(n),
                      static_cast<u32>(H), dst);
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

void match_first(ArrayArg states, ArrayArg task_ids, Arg<ann::GoalTemplates> templates, ArrayArg block,
                 ArrayArg found, ArrayArg binding, ArrayArg template_id, ArrayArg count, ArrayArg status,
                 bool first, ArrayArg derived, StreamArg stream, Arg<PyPool> pool, u64 max_steps)
{
    PyPool* workers = pool_arg(pool);
    const auto t = goal_templates_of(templates);
    const Place pl = place_of(states, stream);
    Imports in(pl.device, pl.stream, false);
    rl::MatchBatch b;
    std::vector<i64> lead, ext;
    b.states = in.words_nd(states, "rows", {-1}, false, lead, b.row_words);
    const i64 n = lead.at(0);
    b.rows = static_cast<u64>(n);
    b.first = first;
    b.max_steps = max_steps;
    if (b.row_words < t->suite()->words())
        throw nb::value_error("mymyr: match_first: rows are narrower than the template source");
    b.task_ids = in.get<const i32>(task_ids, "task_ids", {i32t}, {n}, false);
    if (task_ids.is_none() && t->num_instances() != 1)
        throw nb::value_error("mymyr: match_first: task_ids are required for several instances");
    b.derived = in.words_nd(derived, "derived", {n}, false, lead, b.derived_words);
    if (t->needs_derived() && (derived.is_none() || b.derived_words < t->suite()->derived_words()))
        throw nb::value_error("mymyr: match_first: derived words are required and must cover the template source");
    b.found = in.get<u8>(found, "found", {boolt, u8t}, {n, -1}, true, &ext);
    b.capacity = static_cast<u32>(ext.at(1));
    if (first && b.capacity != 1)
        throw nb::value_error("mymyr: match_first: first=True requires capacity 1");
    b.binding = in.get<i32>(binding, "binding", {i32t}, {n, b.capacity, t->max_vars()}, true);
    b.template_id = in.get<i32>(template_id, "template_id", {i32t}, {n, b.capacity}, true);
    b.count = in.get<i32>(count, "count", {i32t}, {n}, true);
    b.status = in.get<u64>(status, "status", {i64t, u64t}, {}, true);
    if (found.is_none() || binding.is_none() || template_id.is_none() || count.is_none() || status.is_none())
        throw nb::value_error("mymyr: match_first: all output buffers are required");
    [[maybe_unused]] const auto* packed = in.get<const u8>(block, "block", {u8t}, {static_cast<i64>(t->arrays().bytes())}, false);
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
    {
        if (workers)
            throw nb::value_error("mymyr: match_first: pool is only used for host rows");
        if (block.is_none())
            throw nb::value_error("mymyr: match_first: CUDA templates must be uploaded before matching");
        const auto view = t->view(reinterpret_cast<const std::byte*>(packed));
        return launch(pl, "match_first", [&](cudaStream_t cs) { return cuda::launch_match_first(view, b, cs); });
    }
#endif
    if (!block.is_none())
        throw nb::value_error("mymyr: match_first: block is only used for CUDA templates");
    nb::gil_scoped_release release;
    rl::match_first(t->view(), b, workers ? workers->pool.get() : nullptr);
}

void her_relabel_lifted(ArrayArg states, ArrayArg done, ArrayArg task_ids, Arg<ann::GoalTemplates> templates,
                        ArrayArg block, ArrayArg prev_states, nb::tuple outputs, nb::tuple scratch,
                        ArrayArg derived, ArrayArg prev_derived, u64 max_generated, u64 max_steps,
                        f32 step_reward, f32 goal_reward, StreamArg stream, Arg<PyPool> pool)
{
    const auto t = goal_templates_of(templates);
    PyPool* workers = pool_arg(pool);
    const Place pl = place_of(states, stream);
    Imports in(pl.device, pl.stream, false);
    if (outputs.size() != 14 || scratch.size() != 8)
        throw nb::value_error("mymyr: her_relabel_lifted: expected 14 outputs and 8 scratch arrays");
    for (nb::handle x : outputs)
        if (x.is_none())
            throw nb::value_error("mymyr: her_relabel_lifted: every output is required");
    for (nb::handle x : scratch)
        if (x.is_none())
            throw nb::value_error("mymyr: her_relabel_lifted: every scratch array is required");
    if (states.is_none() || prev_states.is_none() || done.is_none())
        throw nb::value_error("mymyr: her_relabel_lifted: states, prev_states and done are required");
    rl::HerLiftedBatch b;
    rl::HerLiftedConfig c;
    c.max_generated = max_generated;
    c.max_steps = max_steps;
    c.step_reward = step_reward;
    c.goal_reward = goal_reward;
    std::vector<i64> lead, ext;
    b.states = in.words_nd(states, "states", {-1, -1}, false, lead, b.row_words);
    const i64 T = lead.at(0), N = lead.at(1);
    if (T >= i64{1} << 31 || N >= i64{1} << 31)
        throw nb::value_error("mymyr: her_relabel_lifted: T and N must be below 2^31");
    b.steps = static_cast<u64>(T);
    b.envs = static_cast<u64>(N);
    u32 PW = 0, PD = 0;
    b.prev_states = in.words_nd(prev_states, "prev_states", {T, N}, false, lead, PW);
    b.done = in.get<const u8>(done, "done", {boolt, u8t}, {T, N}, false);
    if (PW != b.row_words || b.row_words < t->suite()->words() + t->suite()->numeric_words())
        throw nb::value_error("mymyr: her_relabel_lifted: prev_states must match states, covering the template source");
    if (!task_ids.is_none())
    {
        const nb::object shape = is_tensor_view(task_ids)
            ? nb::borrow<nb::object>(view_item(task_ids, k_view_shape)) : nb::getattr(task_ids, "shape");
        if (nb::len(shape) == 1)
            b.task_ids = in.get<const i32>(task_ids, "task_ids", {i32t}, {N}, false);
        else
        {
            b.task_ids = in.get<const i32>(task_ids, "task_ids", {i32t}, {T, N}, false);
            b.task_stride = static_cast<u32>(N);
        }
    }
    else if (t->num_instances() != 1)
        throw nb::value_error("mymyr: her_relabel_lifted: task_ids are required for several instances");
    b.derived = in.words_nd(derived, "derived", {T, N}, false, lead, b.derived_words);
    b.prev_derived = in.words_nd(prev_derived, "prev_derived", {T, N}, false, lead, PD);
    if ((derived.is_none() != prev_derived.is_none()) || PD != b.derived_words ||
        (t->needs_derived() && (derived.is_none() || b.derived_words < t->suite()->derived_words())))
        throw nb::value_error("mymyr: her_relabel_lifted: derived and prev_derived must have matching widths covering the source");
    b.segment = in.get<i32>(outputs[0], "segment", {i32t}, {T, N}, true);
    b.reward = in.get<f32>(outputs[1], "reward", {f32t}, {T, N}, true);
    b.achieved = in.get<u8>(outputs[2], "achieved", {boolt, u8t}, {T, N}, true);
    b.seg_start = in.get<i32>(outputs[3], "seg_start", {i32t}, {N, -1}, true, &ext);
    const i64 S = ext.at(1);
    if (S >= i64{1} << 31)
        throw nb::value_error("mymyr: her_relabel_lifted: max_segments must be below 2^31");
    c.max_segments = static_cast<u32>(S);
    b.seg_end = in.get<i32>(outputs[4], "seg_end", {i32t}, {N, S}, true);
    b.seg_valid = in.get<u8>(outputs[5], "seg_valid", {boolt, u8t}, {N, S}, true);
    b.seg_template = in.get<i32>(outputs[6], "seg_template", {i32t}, {N, S}, true);
    b.seg_binding = in.get<i32>(outputs[7], "seg_binding", {i32t}, {N, S, t->max_vars()}, true);
    b.goal_pos = in.words_nd(outputs[8], "goal_pos", {N, S}, true, lead, b.words);
    u32 W = 0, D = 0;
    b.goal_neg = in.words_nd(outputs[9], "goal_neg", {N, S}, true, lead, W);
    b.goal_pos_derived = in.words_nd(outputs[10], "goal_pos_derived", {N, S}, true, lead, D);
    b.goal_neg_derived = in.words_nd(outputs[11], "goal_neg_derived", {N, S}, true, lead, PD);
    if (W != b.words || W < t->suite()->words() || W > b.row_words || D != b.derived_words || PD != D)
        throw nb::value_error("mymyr: her_relabel_lifted: goal masks must have matching widths covering the template source");
    b.count = in.get<i32>(outputs[12], "count", {i32t}, {N}, true);
    b.status = in.get<u64>(outputs[13], "status", {i64t, u64t}, {}, true);
    const i64 rows = T * N;
    b.hashes = in.get<u64>(scratch[0], "hashes", {i64t, u64t}, {rows}, true);
    b.prev_hashes = in.get<u64>(scratch[1], "prev_hashes", {i64t, u64t}, {rows}, true);
    b.ids = in.get<i32>(scratch[2], "ids", {i32t}, {rows}, true);
    b.matches.found = in.get<u8>(scratch[3], "found", {boolt, u8t}, {rows}, true);
    b.matches.binding = in.get<i32>(scratch[4], "binding", {i32t}, {rows, t->max_vars()}, true);
    b.matches.template_id = in.get<i32>(scratch[5], "template", {i32t}, {rows}, true);
    b.matches.count = in.get<i32>(scratch[6], "match_count", {i32t}, {rows}, true);
    b.work_goals = in.get<u64>(scratch[7], "work_goals", {i64t, u64t}, {N, 2 * (W + D)}, true);
    [[maybe_unused]] const auto* packed = in.get<const u8>(block, "block", {u8t}, {static_cast<i64>(t->arrays().bytes())}, false);
#if defined(MYMYR_HAS_CUDA)
    if (pl.device >= 0)
    {
        if (workers || block.is_none())
            throw nb::value_error("mymyr: her_relabel_lifted: CUDA requires uploaded templates and no host pool");
        return launch(pl, "her_relabel_lifted", [&](cudaStream_t cs)
        {
            return cuda::launch_her_relabel_lifted(t->view(reinterpret_cast<const std::byte*>(packed)), b, c, cs);
        });
    }
#endif
    if (!block.is_none())
        throw nb::value_error("mymyr: her_relabel_lifted: CPU rows require host templates");
    nb::gil_scoped_release release;
    rl::her_relabel_lifted(t->view(), b, c, workers ? workers->pool.get() : nullptr);
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
                 ArrayArg valid, ArrayArg prev_states, Arg<ann::HerStrategy> strategy, u32 subset, u64 seed, u64 first_env, ArrayArg env_ids, ArrayArg goal_atoms,
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
    std::vector<i64> plead;
    u32 PW = 0;
    b.prev_states = in.words_nd(prev_states, "prev_states", {T, N}, false, plead, PW);
    if (b.prev_states && PW != RW)
        throw nb::value_error("mymyr: 'prev_states' must have the shape of 'states'");
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
    b.valid = in.get<u8>(valid, "valid", {boolt, u8t}, {T, N, K}, true);
    b.env_ids = in.get<const u32>(env_ids, "env_ids", {u32t, i32t}, {N}, false);
    if (!goal_atoms.is_none())
    {
        std::vector<i64> alead;
        u32 AW = 0;
        const nb::object shape = is_tensor_view(goal_atoms) ? nb::borrow(view_item(goal_atoms, k_view_shape))
                                                            : nb::getattr(goal_atoms, "shape", nb::none());
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
    nb::class_<TensorViewMarker>(m, "TensorViewMarker",
                                 "The type of TENSOR_VIEW, the first item of a tensor view (mymyr._typing.TensorView)");
    view_marker() = nb::cast(TensorViewMarker{}).release();
    m.attr("TENSOR_VIEW") = view_marker();
    m.def("her_relabel_lifted", &her_relabel_lifted, "states"_a, "done"_a, "task_ids"_a.none(), "templates"_a,
          "block"_a.none(), "prev_states"_a, "outputs"_a, "scratch"_a, nb::kw_only(),
          "derived"_a = nb::none(), "prev_derived"_a = nb::none(), "max_generated"_a, "max_steps"_a,
          "step_reward"_a = 0.0f, "goal_reward"_a = 1.0f, "stream"_a = nb::none(), "pool"_a = nb::none(),
          "Destination-passing lifted hindsight relabelling with fixed outputs and scratch. Window [T,N,RW], "
          "task_ids int32 [N] or [T,N]. Outputs follow HerLiftedBatch's field order; scratch holds hashes, "
          "previous hashes, expanded ids, match found/binding/template/count, and working goal masks. "
          "Status bits: 0 segment overflow, 1 variable limit, 2 invalid instance, 3 matcher budget exhaustion.");
    m.def("match_first", &match_first, "rows"_a, "task_ids"_a.none(), "templates"_a, "block"_a.none(),
          "found"_a, "binding"_a, "template_id"_a, "count"_a, "status"_a, nb::kw_only(), "first"_a = false,
          "derived"_a = nb::none(), "stream"_a = nb::none(), "pool"_a = nb::none(), "max_steps"_a,
          "Destination-passing lifted matching; outputs are padded [N, C], bindings [N, C, V], counts [N], status scalar. "
          "The device block is the uploaded packed template bytes. Status bits: 0 truncation, 1 variable limit, "
          "2 invalid instance, 3 search budget exhausted. "
          "max_steps bounds candidate tiles and binding trials per row/template.");
    m.def("visited_mask", &visited_mask, "table"_a, "count"_a, "succ_hash"_a, "parent"_a, nb::kw_only(),
          "rows"_a = nb::none(), "out"_a, "stream"_a = nb::none(),
          "Writes one bool per successor hash into out [M]: whether succ_hash[j] is in the visited table (table [N, H] "
          "64-bit keys, count [N] int32) of row parent[j], or of rows[parent[j]] when rows is supplied; -1 is padding. "
          "Host arrays raise ValueError for a parent or mapped row outside the table; on CUDA arrays such an entry reads "
          "as not visited.");
    m.def("target_visited", &target_visited, "target"_a, "role"_a, "ref"_a, "entity_offsets"_a, "row_offsets"_a,
          "succ_hash"_a, "table"_a, "count"_a, nb::kw_only(), "live"_a = nb::none(), "row_index"_a = nb::none(),
          "status"_a = nb::none(), "env_rows"_a = nb::none(), "out"_a, "stream"_a = nb::none(),
          "Writes one bool per target into out [T]: whether target t (target [T], int32 or int64 entity indices; those "
          "past live [1] int64 are false) is an action entity (role [E] uint8) whose successor is in its graph's "
          "visited table (table [N, H], count [N] int32). Graph g of entity e by entity_offsets [G + 1] (the targets' "
          "dtype); its expansion rows [row_offsets[g], row_offsets[g + 1]) of succ_hash [M] (row_offsets int32 "
          "[>= G + 1]); the action's row: position ref[e] (ref [E], the targets' dtype) among them, or the row whose "
          "row_index [M] int32 is ref[e]; its table row: env_rows[g] (int32 [G], -1 none; None: g). A nonzero status "
          "[1] int32 (an overflowed expansion) makes every target false. Host or CUDA arrays.");
    m.def("novelty_update", &novelty_update, "seen"_a, "states"_a, "reward"_a = nb::none(), nb::kw_only(),
          "stream"_a = nb::none(),
          "Width-1 novelty update in place: reward[i] = popcount(states[i] & ~seen[i]) over the first S "
          "words, then seen[i] |= states[i]. seen [N, S] and states [N, >= S] are 64-bit words (or 32-bit halves); "
          "reward [N] int32 (None: not written). Host or CUDA arrays.");
    m.def("prefix_masks", &prefix_masks, "parent"_a, "schema"_a, "binding"_a, "query"_a, "prefix"_a.none(), "depth"_a, "mask"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          "Factored action masks into mask [N, num_objects] (bool), mask[i, o] set iff a successor "
          "row j (parent [M], schema [M], binding [M, L] int32; parent -1 = padding) of state i is labelled "
          "(query[i], prefix[i, :depth], o, ...). query [N] int32, prefix [N, >= depth] int32 (None for depth 0).");
    m.def("schema_masks", &schema_masks, "parent"_a, "schema"_a, "mask"_a, nb::kw_only(), "stream"_a = nb::none(),
          "The schema masks [N, num_schemas] (bool), mask[i, s] set iff a successor of state i is labelled with schema "
          "s.");
    m.def("her_relabel", &her_relabel, "states"_a, "done"_a, "goal"_a, "source"_a = nb::none(), "achieved"_a = nb::none(),
          "reward"_a = nb::none(), "valid"_a = nb::none(), nb::kw_only(), "prev_states"_a = nb::none(),
          "strategy"_a = "future", "subset"_a = 0, "seed"_a = 0,
          "first_env"_a = 0, "env_ids"_a = nb::none(), "goal_atoms"_a = nb::none(), "step_reward"_a = 0.0f,
          "goal_reward"_a = 1.0f, "stream"_a = nb::none(),
          "Hindsight relabels of states [T, N, RW] (64-bit words or 32-bit halves; the reached states) "
          "and done [T, N] into goal [T, N, k, W] (k relabels, W <= RW goal words), source [T, N, k] int32, achieved "
          "[T, N, k] bool, reward [T, N, k] float32 and valid [T, N, k] bool (no earlier state of the episode holds the "
          "goal; prev_states [T, N, RW], the states the steps started from, adds each episode's first state). subset 0: every atom of the source state (restricted to "
          "goal_atoms [W], or [N, W]: env i's own, e.g. its instance's fluent atoms); env_ids [N] uint32 (None: first_env + "
          "i) name the RNG streams.");
}
}  // namespace mymyr::python
