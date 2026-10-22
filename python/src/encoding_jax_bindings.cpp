// mymyr._core._encoding_jax: the native side of mymyr.encoding.jax.
//   register(encoder) -> handle  enters a FlatRelationEncoder into the FFI handle registry (ffi_registry.hpp), where
//                                it stays until unregister(handle)
//   info(handle), scratch_bytes(handle, graphs, device, derived), prepare(handle, device), leaked_devices(),
//   calls(handle)
//   ffi_targets(), ffi_types()   the XLA FFI handlers and the state type, for jax.ffi.register_ffi_target and
//                                jax.ffi.register_ffi_type (empty without the XLA FFI headers)
//
// The target mymyr_encode_flat (typed XLA FFI; CPU: FlatEncoder::encode_into, CUDA: cuda::DeviceEncoder on XLA's
// stream with XLA's scratch). Every array is XLA's: the handlers allocate no output.
//   operands    states uint32 [N, 2 W] (the u64 words of each row as uint32 halves, low word first); task_ids int32 [N]
//               (or [0]: every row is instance 0); derived uint32 [N, 2 D] (or [0]: none); goal_pos and goal_neg
//               uint32 [M, 2 GW] and goal_derived_pos and goal_derived_neg uint32 [M, 2 GD] (or [0]: none), M = N or
//               M = 1 (one goal for every row); subgoal_pos and subgoal_neg uint32 [M, L, 2 SW] and subgoal_derived_pos
//               and subgoal_derived_neg uint32 [M, L, 2 SD] (or [0]: none), the L subgoal layers of each row (M = 1:
//               of every row); history_pos and history_neg uint32 [N, H, 2 HW], history_dt int32 [N, H],
//               history_count int32 [N] (or [0]: every step valid) and history_derived_pos and history_derived_neg
//               uint32 [N, H, 2 HD] (or [0]: none), the H history steps of each row, oldest first (the layouts and
//               rules of encoding::LayerInput and encoding::HistoryInput)
//   attributes  args, entities, targets, history: the capacities of relation_args, of the entity arrays, of the
//               target arrays and of the history step arrays; handle: the registry handle; threads: the CPU target's
//               worker threads (0: one per hardware thread; the CUDA target ignores it)
//   results     the 16 arrays of a batch in the order of mymyr.encoding.DeviceFlatBatch.array_names (an array the
//               schema does not store is [0]): the per-graph and per-cell arrays sized by N, the variable-size ones by
//               the capacities; then sizes uint32 [S, 2] (the S 64-bit words of encoding::FlatSizes as uint32 halves,
//               low half first, in the field order of the struct, the overflow flags among them); then, required on
//               CUDA and ignored on the CPU, a uint8 scratch of at least scratch_bytes(handle, N, device, derived) +
//               255 bytes
// Both targets give the capacity form's result: the arrays hold the batch and are zero past its sizes, or are all
// zero when a flag is set (1: a capacity, 2: the index type, 4: an invalid row; an invalid row is a flag on both
// targets, its sizes are those its counting reached).
//
// State and lifetime: the instantiate stage looks the handle up once and keeps the encoder's entry as the call's
// state, which XLA destroys with the executable; the registry and every executable that calls the encoder keep it
// alive, so a compiled function keeps working after unregister(). On CUDA the initialize stage (before XLA runs a call
// or traces it into a command buffer) uploads the encoder's context to the call's device once and holds an
// acquisition of it for the entry's lifetime; the execute stage then enqueues memsets and the encoding kernels on XLA's
// stream with XLA's scratch, and nothing else: no allocation, synchronization, event or stream operation, so the
// target is command-buffer compatible.

#include "ffi_registry.hpp"
#include "typing.hpp"

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/core/threads.hpp"
#include "mymyr/encoding/encoder.hpp"

#if defined(MYMYR_HAS_CUDA)
#include "mymyr/cuda/encoding.hpp"
#include "mymyr/cuda/runtime.hpp"
#endif

#if defined(MYMYR_HAS_XLA_FFI)
#include "xla/ffi/api/ffi.h"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mymyr::python
{
using namespace nb::literals;

std::shared_ptr<const encoding::FlatEncoder> flat_encoder_of(nb::handle obj);  // encoding_bindings.cpp

namespace
{
/// A registered encoder: the encoder, the CPU target's pool and scratch, and per CUDA device the uploaded context.
struct EncoderEntry : FfiEntry
{
    std::shared_ptr<const encoding::FlatEncoder> encoder;

    // The CPU target: one call at a time (the scratch), on the pool of the call's thread count. The pools stay, so
    // that calls alternating between thread counts build none after the first of each.
    std::mutex host_mutex;
    encoding::EncodeScratch scratch;
    std::map<u32, std::shared_ptr<ThreadPool>> pools;
    static constexpr usize k_max_pools = 4;

    std::atomic<u64> cpu_calls{0}, cuda_calls{0}, captured_calls{0}, pools_built{0};

    explicit EncoderEntry(std::shared_ptr<const encoding::FlatEncoder> e) : encoder(std::move(e))
    {
        if (encoder->schema().shared_statics())
            throw std::invalid_argument("mymyr: encoding: JAX FFI does not encode shared statics: "
                                        "its result signature has no static block arrays");
    }

    /// The pool of `threads` workers (the caller holds host_mutex).
    ThreadPool& pool_of(u32 threads)
    {
        auto it = pools.find(threads);
        if (it == pools.end())
        {
            if (pools.size() >= k_max_pools)
                pools.clear();
            it = pools.emplace(threads, std::make_shared<ThreadPool>(threads)).first;
            ++pools_built;
        }
        return *it->second;
    }

#if defined(MYMYR_HAS_CUDA)
    /// Detaches the calling thread from the interpreter while it waits for a device (a thread of XLA has no thread
    /// state to detach).
    struct Detached
    {
        PyThreadState* state = PyThreadState_GetUnchecked() ? PyEval_SaveThread() : nullptr;
        Detached() = default;
        Detached(const Detached&) = delete;
        Detached& operator=(const Detached&) = delete;
        ~Detached()
        {
            if (state)
                PyEval_RestoreThread(state);
        }
    };

    /// The encoder on one device: its own device context, the uploaded encoding context and an acquisition of it that
    /// every call of the device uses.
    struct Device
    {
        int ordinal = 0;
        cuda::ContextPtr ctx;
        std::unique_ptr<cuda::DeviceEncoder> encoder;
        std::optional<cuda::EncodingAcquisition> held;

        Device() = default;
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;
        ~Device()
        {
            // The calls ran on XLA's streams, which the acquisition's release does not wait for: the device's work
            // completes before the context generation can be freed. A device that cannot synchronize (CUDA refuses
            // the call while another thread captures a CUDA graph, and a failed device may still run the work) keeps
            // its generation: it is leaked rather than freed under the work that may read it.
            const Detached detached;  // the last reference may be dropped by a Python thread
            bool synchronized = false;
            try
            {
                cuda::DeviceGuard guard(ordinal);
                synchronized = cudaDeviceSynchronize() == cudaSuccess;
                if (!synchronized)
                    (void)cudaGetLastError();
            }
            catch (...)
            {
            }
            if (synchronized)
            {
                try
                {
                    cuda::DeviceGuard guard(ordinal);
                    held.reset();
                    encoder.reset();
                }
                catch (...)
                {
                }
                return;
            }
            struct Leaked
            {
                cuda::ContextPtr ctx;
                std::unique_ptr<cuda::DeviceEncoder> encoder;
                std::optional<cuda::EncodingAcquisition> held;
            };
            new Leaked{std::move(ctx), std::move(encoder), std::move(held)};
            ++leaked_devices;
        }
    };
    static inline std::atomic<u64> leaked_devices{0};
    std::mutex devices_mutex;
    std::map<int, std::shared_ptr<Device>> devices;

    /// The encoder on device `ordinal`, uploaded and acquired on first use (allocates and waits for the upload, so it
    /// runs outside a capture: in the initialize stage).
    std::shared_ptr<Device> device(int ordinal, bool create)
    {
        std::lock_guard lock(devices_mutex);
        std::shared_ptr<Device>& d = devices[ordinal];
        if (!d && create)
        {
            auto made = std::make_shared<Device>();
            made->ordinal = ordinal;
            made->ctx = cuda::DeviceContext::create(ordinal);
            made->encoder = std::make_unique<cuda::DeviceEncoder>(made->ctx, encoder);
            made->held.emplace(made->encoder->context().acquire());
            cuda::DeviceGuard guard(ordinal);
            cuda::check(cudaStreamSynchronize(made->held->stream()), "cudaStreamSynchronize (encoding context)");
            d = std::move(made);
        }
        return d;
    }
#endif
};
using EntryPtr = std::shared_ptr<EncoderEntry>;

EntryPtr entry_of(i64 handle)
{
    EntryPtr e = ffi_registry().find_as<EncoderEntry>(handle);
    if (!e)
        throw nb::value_error(("mymyr: encoding: no encoder is registered under handle " + std::to_string(handle)).c_str());
    return e;
}

// ------------------------------------------------------------------------------------------------ FFI handlers

#if defined(MYMYR_HAS_XLA_FFI)
namespace ffi = xla::ffi;

/// The state of one mymyr_encode_flat call site of an executable: the encoder's entry. Each platform's XLA runtime
/// keeps a type registry of its own (the CUDA one is a plugin library) and assigns its own type id, so each platform
/// has its own state type: 0 the CPU, 1 CUDA.
template<int Platform>
struct EncodeCall
{
    static ffi::TypeId id;
    EntryPtr entry;
};
template<int Platform>
ffi::TypeId EncodeCall<Platform>::id = {};
template<int Platform>
constexpr ffi::TypeInfo k_encode_call_info = ffi::MakeTypeInfo<EncodeCall<Platform>>();
using CpuCall = EncodeCall<0>;
using CudaCall = EncodeCall<1>;

struct Failure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& msg) { throw Failure("mymyr: encoding: " + msg); }

template<class F>
ffi::Error guarded(F&& f)
{
    try
    {
        f();
        return ffi::Error::Success();
    }
    catch (const Failure& x)
    {
        return ffi::Error::InvalidArgument(x.what());
    }
    catch (const std::invalid_argument& x)
    {
        return ffi::Error::InvalidArgument(x.what());
    }
    catch (const std::exception& x)
    {
        return ffi::Error::Internal(std::string(x.what()));
    }
}

/// The number of results before the sizes and the scratch.
constexpr usize k_arrays = 16;
constexpr const char* k_array_names[k_arrays] = {
    "task_ids",        "num_objects",      "entity_offsets",       "entity_role",     "entity_ref",
    "relation_counts", "relation_offsets", "relation_arg_offsets", "relation_args",   "goal_complete",
    "target_offsets",  "target_entity",    "target_group",         "history_offsets", "history_entity",
    "history_dt"};

/// A pair of row inputs as u64 words: rows and words per row (uint32 [rows, 2 words], or [0]: absent).
struct Words
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 words = 0;
};

Words words_of(const ffi::AnyBuffer& b, const char* name)
{
    if (b.element_type() != ffi::DataType::U32)
        fail(std::string("'") + name + "' must be uint32 (the u64 words as halves)");
    Words w;
    if (b.element_count() == 0)
        return w;
    const auto dims = b.dimensions();
    if (dims.size() != 2 || dims[1] % 2 != 0)
        fail(std::string("'") + name + "' must be [rows, 2 words] uint32");
    if (reinterpret_cast<std::uintptr_t>(b.untyped_data()) % 8 != 0)
        fail(std::string("'") + name + "' is not 8-byte aligned");
    w.data = static_cast<const u64*>(b.untyped_data());
    w.rows = static_cast<u64>(dims[0]);
    w.words = static_cast<u32>(dims[1] / 2);
    return w;
}

/// A set of mask layers per row as u64 words: rows, layers and words per layer (uint32 [rows, layers, 2 words], or
/// [0]: absent). `data` is null when the array has no entries.
struct Layers
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 layers = 0;
    u32 words = 0;
    bool given = false;
};

Layers layers_of(const ffi::AnyBuffer& b, const char* name)
{
    if (b.element_type() != ffi::DataType::U32)
        fail(std::string("'") + name + "' must be uint32 (the u64 words as halves)");
    const auto dims = b.dimensions();
    Layers l;
    if (dims.size() == 1 && dims[0] == 0)
        return l;
    if (dims.size() != 3 || dims[2] % 2 != 0)
        fail(std::string("'") + name + "' must be [rows, layers, 2 words] uint32");
    l.given = true;
    l.rows = static_cast<u64>(dims[0]);
    l.layers = static_cast<u32>(dims[1]);
    l.words = static_cast<u32>(dims[2] / 2);
    if (b.element_count() == 0)
        return l;
    if (reinterpret_cast<std::uintptr_t>(b.untyped_data()) % 8 != 0)
        fail(std::string("'") + name + "' is not 8-byte aligned");
    l.data = static_cast<const u64*>(b.untyped_data());
    return l;
}

/// The operands of mymyr_encode_flat, in order.
namespace operand
{
enum : usize
{
    states,
    task_ids,
    derived,
    goal_pos,
    goal_neg,
    goal_derived_pos,
    goal_derived_neg,
    subgoal_pos,
    subgoal_neg,
    subgoal_derived_pos,
    subgoal_derived_neg,
    history_pos,
    history_neg,
    history_dt,
    history_count,
    history_derived_pos,
    history_derived_neg,
    count
};
constexpr const char* names[count] = {
    "states",          "task_ids",          "derived",           "goal_pos",           "goal_neg",
    "goal_derived_pos", "goal_derived_neg", "subgoal_pos",       "subgoal_neg",        "subgoal_derived_pos",
    "subgoal_derived_neg", "history_pos",   "history_neg",       "history_dt",         "history_count",
    "history_derived_pos", "history_derived_neg"};
}  // namespace operand

/// One call's inputs and destinations, checked against the encoder (shapes and dtypes; the rows' contents are the
/// targets' to check).
struct Call
{
    u64 graphs = 0;
    Words states;
    const i32* task_ids = nullptr;
    Words derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg;
    encoding::LayerInput subgoals{};
    encoding::HistoryInput history{};
    encoding::BatchShape capacities;
    std::array<void*, k_arrays> arrays{};   // null: absent or without entries
    std::array<u64, k_arrays> bytes{};
    encoding::FlatSizes* sizes = nullptr;
    void* scratch = nullptr;
    u64 scratch_bytes = 0;
};

u64 index_bytes(const encoding::FlatEncoder& enc)
{
    return enc.schema().options().index == encoding::IndexType::I64 ? 8 : 4;
}

/// The masks `p` and `n` (checked to be a pair) as the fluent or the derived masks of `l`; one row for every graph
/// has a stride of 0.
void set_layers(const Call& c, const Layers& p, const Layers& n, encoding::LayerInput& l, bool derived)
{
    const u64 stride = p.rows == 1 && c.graphs != 1 ? 0 : u64{p.layers} * p.words;
    if (derived)
    {
        l.derived_pos = p.data;
        l.derived_neg = n.data;
        l.derived_stride = stride;
        l.derived_words = p.words;
    }
    else
    {
        l.pos = p.data;
        l.neg = n.data;
        l.stride = stride;
        l.layers = p.layers;
        l.words = p.words;
    }
}

Call call_of(const EncoderEntry& e, i64 args, i64 entities, i64 history, i64 targets, const ffi::RemainingArgs& ops,
             const ffi::RemainingRets& rets, bool scratch_required)
{
    const encoding::FlatEncoder& enc = *e.encoder;
    const encoding::EncodingSchema& schema = enc.schema();
    Call c;
    if (args < 0 || entities < 0 || targets < 0 || history < 0)
        fail("the capacities must be at least 0");
    if (ops.size() != operand::count)
        fail("mymyr_encode_flat has " + std::to_string(operand::count) + " operands, not " + std::to_string(ops.size()));
    std::array<ffi::AnyBuffer, operand::count> buffers = [&]<usize... K>(std::index_sequence<K...>) {
        auto one = [&](usize k) {
            auto b = ops.get<ffi::AnyBuffer>(k);
            if (!b.has_value())
                fail(std::string("the operand '") + operand::names[k] + "' must be a buffer");
            return *b;
        };
        return std::array<ffi::AnyBuffer, operand::count>{one(K)...};
    }(std::make_index_sequence<operand::count>{});
    const ffi::AnyBuffer& states = buffers[operand::states];
    const ffi::AnyBuffer& task_ids = buffers[operand::task_ids];
    if (states.element_type() != ffi::DataType::U32)
        fail("'states' must be uint32 (the u64 words as halves)");
    const auto sdims = states.dimensions();
    if (sdims.size() != 2 || sdims[1] % 2 != 0)
        fail("'states' must be [N, 2 W] uint32");
    c.graphs = static_cast<u64>(sdims[0]);
    if (c.graphs && reinterpret_cast<std::uintptr_t>(states.untyped_data()) % 8 != 0)
        fail("'states' is not 8-byte aligned");
    c.states = {c.graphs ? static_cast<const u64*>(states.untyped_data()) : nullptr, c.graphs,
                static_cast<u32>(sdims[1] / 2)};
    if (task_ids.element_type() != ffi::DataType::S32)
        fail("'task_ids' must be int32");
    if (task_ids.element_count() != 0 && task_ids.element_count() != c.graphs)
        fail("'task_ids' must have one entry per row");
    if (task_ids.element_count() == 0 && c.graphs && enc.table().size() > 1)
        fail("task_ids are required for a table of several instances");
    c.task_ids = task_ids.element_count() ? static_cast<const i32*>(task_ids.untyped_data()) : nullptr;
    c.derived = words_of(buffers[operand::derived], "derived");
    if (c.derived.data && c.derived.rows != c.graphs)
        fail("'derived' must have one row per state");
    if (c.derived.data && c.derived.words < enc.context().max_derived_words())
        fail("'derived' needs " + std::to_string(enc.context().max_derived_words()) + " words per row");
    c.goal_pos = words_of(buffers[operand::goal_pos], "goal_pos");
    c.goal_neg = words_of(buffers[operand::goal_neg], "goal_neg");
    c.goal_derived_pos = words_of(buffers[operand::goal_derived_pos], "goal_derived_pos");
    c.goal_derived_neg = words_of(buffers[operand::goal_derived_neg], "goal_derived_neg");
    auto pair = [&](const Words& p, const Words& n, const char* what) {
        if ((p.data == nullptr) != (n.data == nullptr) || p.rows != n.rows || p.words != n.words)
            fail(std::string(what) + " are two masks of the same shape");
        if (p.data && p.rows != c.graphs && p.rows != 1)
            fail(std::string(what) + " have one row per state or one row for all");
    };
    pair(c.goal_pos, c.goal_neg, "goal_pos and goal_neg");
    pair(c.goal_derived_pos, c.goal_derived_neg, "goal_derived_pos and goal_derived_neg");
    if (c.goal_derived_pos.data && !c.goal_pos.data)
        fail("the derived goal masks need the goal masks over the fluent slots");
    if (c.goal_pos.data && !schema.has_goal_complete())
        fail("goals are given, but the schema has no goal relations");

    // the subgoal layers and the history steps
    std::array<Layers, operand::count> layers{};
    for (const usize k : {operand::subgoal_pos, operand::subgoal_neg, operand::subgoal_derived_pos,
                          operand::subgoal_derived_neg, operand::history_pos, operand::history_neg,
                          operand::history_derived_pos, operand::history_derived_neg})
        layers[k] = layers_of(buffers[k], operand::names[k]);
    auto layer_pair = [&](usize at, bool may_share) {
        const Layers &p = layers[at], &n = layers[at + 1];
        const std::string what = std::string(operand::names[at]) + " and " + operand::names[at + 1];
        if (p.given != n.given || p.rows != n.rows || p.layers != n.layers || p.words != n.words)
            fail(what + " are two arrays of masks of the same shape");
        if (p.given && p.rows != c.graphs && !(may_share && p.rows == 1))
            fail(what + (may_share ? " have one row per state or one row for all" : " have one row per state"));
    };
    layer_pair(operand::subgoal_pos, true);
    layer_pair(operand::subgoal_derived_pos, true);
    layer_pair(operand::history_pos, false);
    layer_pair(operand::history_derived_pos, false);
    const Layers& sub = layers[operand::subgoal_pos];
    const Layers& steps = layers[operand::history_pos];
    if (layers[operand::subgoal_derived_pos].given &&
        (!sub.given || layers[operand::subgoal_derived_pos].layers != sub.layers))
        fail("the derived subgoal masks need the subgoal masks over the fluent slots, with as many layers");
    if (layers[operand::history_derived_pos].given &&
        (!steps.given || layers[operand::history_derived_pos].layers != steps.layers))
        fail("the derived history masks need the history masks over the fluent slots, with as many steps");
    if (sub.layers != 0 && !schema.has_subgoals())
        fail("subgoal layers are given, but the schema has no subgoal relations");
    if (sub.layers > schema.options().max_goal_level)
        fail(std::to_string(sub.layers) + " subgoal layers are given, the schema has " +
             std::to_string(schema.options().max_goal_level) + " levels");
    if (steps.layers != 0 && !schema.has_history())
        fail("history steps are given, but the schema has no history relations");
    // dt [N, H] and count [N] of the history, or [0]: absent
    auto ints = [&](usize k, std::initializer_list<u64> shape) -> const i32* {
        const ffi::AnyBuffer& b = buffers[k];
        const auto dims = b.dimensions();
        std::string want = "[";
        for (const u64 d : shape)
            want += (want.size() > 1 ? ", " : "") + std::to_string(d);
        want += "]";
        if (b.element_type() != ffi::DataType::S32)
            fail(std::string("'") + operand::names[k] + "' must be int32 " + want);
        if (dims.size() == 1 && dims[0] == 0)
            return nullptr;
        if (!steps.given)
            fail(std::string("'") + operand::names[k] + "' is given without the history masks");
        bool same = dims.size() == shape.size();
        for (usize d = 0; same && d < shape.size(); ++d)
            same = static_cast<u64>(dims[d]) == shape.begin()[d];
        if (!same)
            fail(std::string("'") + operand::names[k] + "' must be int32 " + want);
        return b.element_count() ? static_cast<const i32*>(b.untyped_data()) : nullptr;
    };
    c.history.dt = ints(operand::history_dt, {c.graphs, steps.layers});
    c.history.count = ints(operand::history_count, {c.graphs});
    if (steps.layers != 0 && c.graphs != 0 && !c.history.dt)
        fail("the history steps need their dt");
    set_layers(c, sub, layers[operand::subgoal_neg], c.subgoals, false);
    set_layers(c, layers[operand::subgoal_derived_pos], layers[operand::subgoal_derived_neg], c.subgoals, true);
    set_layers(c, steps, layers[operand::history_neg], c.history.masks, false);
    set_layers(c, layers[operand::history_derived_pos], layers[operand::history_derived_neg], c.history.masks, true);
    c.capacities = {c.graphs, static_cast<u64>(entities), static_cast<u64>(args), static_cast<u64>(targets),
                    static_cast<u64>(history)};

    const usize expected = k_arrays + 1 + (scratch_required ? 1 : 0);
    if (rets.size() != expected && !(rets.size() == k_arrays + 2 && !scratch_required))
        fail("mymyr_encode_flat has " + std::to_string(expected) + " results here, not " + std::to_string(rets.size()));
    const u64 I = index_bytes(enc);
    const u64 G = c.graphs, R = schema.num_relations();
    const bool entity_arrays = !schema.implicit_entity_arrays();
    const bool counts = schema.emits_counts();
    const bool goal = schema.has_goal_complete();
    const bool has_targets = schema.has_targets();
    const bool has_history = schema.has_history();
    // the elements and element size of every array (0 elements: absent)
    const std::array<std::pair<u64, u64>, k_arrays> want = {{
        {G, 4},
        {G, I},
        {G + 1, I},
        {entity_arrays ? c.capacities.entities : 0, 1},
        {entity_arrays ? c.capacities.entities : 0, I},
        {counts ? G * R : 0, I},
        {G * R + 1, I},
        {G * R + 1, I},
        {c.capacities.args, I},
        {goal ? G : 0, 1},
        {has_targets ? G + 1 : 0, I},
        {has_targets ? c.capacities.targets : 0, I},
        {has_targets ? c.capacities.targets : 0, 1},
        {has_history ? G + 1 : 0, I},
        {has_history ? c.capacities.history : 0, I},
        {has_history ? c.capacities.history : 0, 4},
    }};
    for (usize k = 0; k < k_arrays; ++k)
    {
        auto r = rets.get<ffi::AnyBuffer>(k);
        if (!r.has_value())
            fail(std::string("the result '") + k_array_names[k] + "' must be a buffer");
        const ffi::AnyBuffer& b = **r;
        const auto [n, size] = want[k];
        if (b.element_count() != n || (n && ffi::ByteWidth(b.element_type()) != size))
            fail(std::string("the result '") + k_array_names[k] + "' must have " + std::to_string(n) +
                 " elements of " + std::to_string(size) + " bytes");
        c.arrays[k] = n ? b.untyped_data() : nullptr;
        c.bytes[k] = n * size;
    }
    auto s = rets.get<ffi::AnyBuffer>(k_arrays);
    if (!s.has_value() || (**s).size_bytes() != sizeof(encoding::FlatSizes))
        fail("the result 'sizes' must be uint32 [" + std::to_string(sizeof(encoding::FlatSizes) / 8) + ", 2]");
    c.sizes = static_cast<encoding::FlatSizes*>((**s).untyped_data());
    if (rets.size() == k_arrays + 2)
    {
        auto x = rets.get<ffi::AnyBuffer>(k_arrays + 1);
        if (!x.has_value() || (**x).element_type() != ffi::DataType::U8)
            fail("the result 'scratch' must be uint8");
        c.scratch = (**x).untyped_data();
        c.scratch_bytes = (**x).size_bytes();
    }
    return c;
}

/// A goal mask pair as the encoders take it: a stride of 0 makes one row serve every graph.
template<class Input>
void set_goal(const Call& c, const Words& p, const Words& n, Input& in, bool derived)
{
    const u64 stride = p.rows == 1 && c.graphs != 1 ? 0 : p.words;
    if (derived)
    {
        in.goal_derived_pos = p.data;
        in.goal_derived_neg = n.data;
        in.goal_derived_stride = stride;
        in.goal_derived_words = p.words;
    }
    else
    {
        in.goal_pos = p.data;
        in.goal_neg = n.data;
        in.goal_stride = stride;
        in.goal_words = p.words;
    }
}

encoding::FlatBuffers buffers_of(const Call& c)
{
    encoding::FlatBuffers b;
    b.task_ids = static_cast<i32*>(c.arrays[0]);
    b.num_objects = c.arrays[1];
    b.entity_offsets = c.arrays[2];
    b.entity_role = static_cast<u8*>(c.arrays[3]);
    b.entity_ref = c.arrays[4];
    b.relation_counts = c.arrays[5];
    b.relation_offsets = c.arrays[6];
    b.relation_arg_offsets = c.arrays[7];
    b.relation_args = c.arrays[8];
    b.goal_complete = static_cast<u8*>(c.arrays[9]);
    b.target_offsets = c.arrays[10];
    b.target_entity = c.arrays[11];
    b.target_group = static_cast<u8*>(c.arrays[12]);
    b.history_offsets = c.arrays[13];
    b.history_entity = c.arrays[14];
    b.history_dt = static_cast<i32*>(c.arrays[15]);
    return b;
}

template<class Call>
Call* state_of(Call* call)
{
    if (!call || !call->entry)
        fail("the call has no encoder (its instantiate stage did not run)");
    return call;
}

template<class Call>
ffi::ErrorOr<std::unique_ptr<Call>> encode_instantiate(ffi::Dictionary attrs)
{
    const auto attr = attrs.get<i64>("handle");
    if (!attr.has_value())
        return ffi::Error::InvalidArgument("mymyr: encoding: a call without its 'handle' attribute");
    const i64 handle = attr.value();
    EntryPtr e = ffi_registry().find_as<EncoderEntry>(handle);
    if (!e)
        return ffi::Error::InvalidArgument("mymyr: encoding: no encoder is registered under handle " +
                                           std::to_string(handle) + " (see mymyr.encoding.jax.register)");
    auto call = std::make_unique<Call>();
    call->entry = std::move(e);
    return call;
}

// ---- CPU

ffi::Error encode_cpu(CpuCall* call, i64 args, i64 entities, i64 /*handle*/, i64 history, i64 targets, i64 threads,
                      ffi::RemainingArgs ops, ffi::RemainingRets rets)
{
    return guarded([&] {
        EncoderEntry& e = *state_of(call)->entry;
        const Call c = call_of(e, args, entities, history, targets, ops, rets, false);
        ++e.cpu_calls;
        if (c.graphs == 0)  // the empty batch: zero offsets and sizes
        {
            for (usize k = 0; k < k_arrays; ++k)
                if (c.arrays[k])
                    std::memset(c.arrays[k], 0, c.bytes[k]);
            *c.sizes = encoding::FlatSizes{};
            return;
        }
        encoding::EncodeInput in;
        in.states = rl::StateBatchView{c.states.data, c.graphs, c.states.words, c.states.words, 0};
        in.task_ids = c.task_ids;
        in.derived = c.derived.data;
        in.derived_stride = c.derived.words;
        set_goal(c, c.goal_pos, c.goal_neg, in, false);
        set_goal(c, c.goal_derived_pos, c.goal_derived_neg, in, true);
        in.subgoals = c.subgoals;
        in.history = c.history;
        const encoding::FlatBuffers out = buffers_of(c);
        encoding::FlatSizes sizes;
        {
            std::lock_guard lock(e.host_mutex);
            ThreadPool& pool = e.pool_of(resolve_threads(static_cast<u64>(std::max<i64>(threads, 0))));
            try
            {
                sizes = e.encoder->encode_into(in, c.capacities, out, pool, e.scratch);
            }
            catch (const std::invalid_argument&)
            {
                // the rows' contents (a task id outside the table, a row, goal mask, subgoal layer or history step
                // narrower than its instance, a bit its instance has no atom for, a history count outside the row's
                // steps): a flag, as on the device
                sizes = encoding::FlatSizes{};
                sizes.overflow = encoding::k_overflow_input;
            }
        }
        auto zero_from = [&](usize k, u64 from, u64 size) {
            if (c.arrays[k] && from * size < c.bytes[k])
                std::memset(static_cast<std::byte*>(c.arrays[k]) + from * size, 0, c.bytes[k] - from * size);
        };
        if (sizes.overflow)
            for (usize k = 0; k < k_arrays; ++k)
                zero_from(k, 0, 1);
        else
        {
            const u64 I = index_bytes(*e.encoder);
            zero_from(3, sizes.entities, 1);
            zero_from(4, sizes.entities, I);
            zero_from(8, sizes.args, I);
            zero_from(11, sizes.targets, I);
            zero_from(12, sizes.targets, 1);
            zero_from(14, sizes.history, I);
            zero_from(15, sizes.history, 4);
        }
        std::memcpy(c.sizes, &sizes, sizeof sizes);
    });
}

#define MYMYR_ENCODE_ATTRS                                                                                              \
    .Attr<i64>("args")                                                                                                  \
        .Attr<i64>("entities")                                                                                          \
        .Attr<i64>("handle")                                                                                            \
        .Attr<i64>("history")                                                                                           \
        .Attr<i64>("targets")                                                                                           \
        .Attr<i64>("threads")
// the operands of `operand` (checked by call_of), then the results
#define MYMYR_ENCODE_OPERANDS .RemainingArgs().RemainingRets()

XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_encode_instantiate_cpu, encode_instantiate<CpuCall>,
                              ffi::Ffi::BindInstantiate().Attrs());
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_encode_cpu, encode_cpu,
                              ffi::Ffi::Bind().Ctx<ffi::State<CpuCall>>() MYMYR_ENCODE_ATTRS MYMYR_ENCODE_OPERANDS);

// ---- CUDA

#if defined(MYMYR_HAS_CUDA)
/// Scratch addresses are aligned to this many bytes (DeviceEncoder's requirement); the scratch result has this much
/// slack.
constexpr u64 k_scratch_align = 256;

ffi::Error encode_cuda(cudaStream_t st, i32 ordinal, CudaCall* call, i64 args, i64 entities, i64 /*handle*/,
                       i64 history, i64 targets, i64 /*threads*/, ffi::RemainingArgs ops, ffi::RemainingRets rets)
{
    return guarded([&] {
        EncoderEntry& e = *state_of(call)->entry;
        const Call c = call_of(e, args, entities, history, targets, ops, rets, true);
        cuda::DeviceGuard guard(ordinal);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        cuda::check(cudaStreamIsCapturing(st, &status), "cudaStreamIsCapturing");
        const bool capturing = status != cudaStreamCaptureStatusNone;
        auto d = e.device(ordinal, !capturing);
        if (!d)
            fail("the encoder's context is not on cuda:" + std::to_string(ordinal) + " (the initialize stage uploads it)");
        ++e.cuda_calls;
        if (capturing)
            ++e.captured_calls;
        if (c.graphs == 0)  // the empty batch: zero arrays and sizes
        {
            for (usize k = 0; k < k_arrays; ++k)
                if (c.arrays[k])
                    cuda::check(cudaMemsetAsync(c.arrays[k], 0, c.bytes[k], st), "cudaMemsetAsync");
            cuda::check(cudaMemsetAsync(c.sizes, 0, sizeof(encoding::FlatSizes), st), "cudaMemsetAsync");
            return;
        }
        const encoding::BatchShape shape = c.capacities;
        const cuda::DerivedWords derived = c.derived.data ? cuda::DerivedWords::Supplied : cuda::DerivedWords::Evaluated;
        const u64 need = d->encoder->scratch_bytes(*d->held, shape, derived);
        const auto base = reinterpret_cast<std::uintptr_t>(c.scratch);
        const u64 pad = (k_scratch_align - base % k_scratch_align) % k_scratch_align;
        if (!c.scratch || c.scratch_bytes < pad || c.scratch_bytes - pad < need)
            fail("the scratch result has " + std::to_string(c.scratch_bytes) + " bytes; the batch needs " +
                 std::to_string(need + k_scratch_align - 1));
        cuda::DeviceEncodeInput in;
        in.graphs = c.graphs;
        in.states = c.states.data;
        in.state_stride = c.states.words;
        in.state_words = c.states.words;
        in.task_ids = c.task_ids;
        in.derived = c.derived.data;
        in.derived_stride = c.derived.words;
        set_goal(c, c.goal_pos, c.goal_neg, in, false);
        set_goal(c, c.goal_derived_pos, c.goal_derived_neg, in, true);
        in.subgoals = c.subgoals;
        in.history = c.history;
        cuda::DeviceOutput out;
        out.arrays = buffers_of(c);
        out.capacities = shape;
        out.sizes = c.sizes;
        out.zero_unused = true;
        d->encoder->encode(*d->held, st, reinterpret_cast<void*>(base + pad), c.scratch_bytes - pad, shape, in, out);
    });
}

/// The initialize stage: uploads the encoder's context to the call's device (once per entry and device), so that the
/// execute stage, which XLA may trace into a command buffer, allocates nothing.
ffi::Error encode_initialize_cuda(i32 ordinal, CudaCall* call, ffi::Dictionary, ffi::RemainingArgs, ffi::RemainingRets)
{
    return guarded([&] { state_of(call)->entry->device(ordinal, true); });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_encode_instantiate_cuda, encode_instantiate<CudaCall>,
                              ffi::Ffi::BindInstantiate().Attrs());
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_encode_initialize_cuda, encode_initialize_cuda,
                              ffi::Ffi::BindInitialize()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Ctx<ffi::State<CudaCall>>()
                                  .Attrs()
                                  .RemainingArgs()
                                  .RemainingRets());
// Memsets and the encoding kernels on XLA's stream, with XLA's scratch: compatible with command buffers.
XLA_FFI_DEFINE_HANDLER_SYMBOL(mymyr_ffi_encode_cuda, encode_cuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Ctx<ffi::State<CudaCall>>() MYMYR_ENCODE_ATTRS MYMYR_ENCODE_OPERANDS,
                              {ffi::Traits::kCmdBufferCompatible});
#endif  // MYMYR_HAS_CUDA
#undef MYMYR_ENCODE_ATTRS
#undef MYMYR_ENCODE_OPERANDS
#endif  // MYMYR_HAS_XLA_FFI

/// A target: its name, platform, handlers per execution stage and whether XLA may trace its calls into command buffers
/// (as rl_jax_bindings.cpp's ffi_targets()).
using Target = std::tuple<std::string, std::string, std::map<std::string, nb::capsule>, bool>;

std::vector<Target> ffi_targets()
{
    std::vector<Target> out;
#if defined(MYMYR_HAS_XLA_FFI)
    auto capsule = [](XLA_FFI_Handler* h) { return nb::capsule(reinterpret_cast<void*>(h)); };
    {
        std::map<std::string, nb::capsule> stages;
        stages.emplace("instantiate", capsule(mymyr_ffi_encode_instantiate_cpu));
        stages.emplace("execute", capsule(mymyr_ffi_encode_cpu));
        out.emplace_back("mymyr_encode_flat", "cpu", std::move(stages), false);
    }
#if defined(MYMYR_HAS_CUDA)
    {
        std::map<std::string, nb::capsule> stages;
        stages.emplace("instantiate", capsule(mymyr_ffi_encode_instantiate_cuda));
        stages.emplace("initialize", capsule(mymyr_ffi_encode_initialize_cuda));
        stages.emplace("execute", capsule(mymyr_ffi_encode_cuda));
        out.emplace_back("mymyr_encode_flat", "CUDA", std::move(stages), true);
    }
#endif
#endif
    return out;
}

/// The FFI state types: (name, platform, {"type_id": capsule, "type_info": capsule}) for jax.ffi.register_ffi_type.
using TypeRegistration = std::tuple<std::string, std::string, std::map<std::string, nb::capsule>>;
std::vector<TypeRegistration> ffi_types()
{
    std::vector<TypeRegistration> out;
#if defined(MYMYR_HAS_XLA_FFI)
    auto add = [&](const char* platform, ffi::TypeId* id, const ffi::TypeInfo* info) {
        std::map<std::string, nb::capsule> reg;
        reg.emplace("type_id", nb::capsule(static_cast<void*>(id)));
        reg.emplace("type_info", nb::capsule(const_cast<void*>(static_cast<const void*>(info))));
        out.emplace_back("mymyr_encode_flat_call", platform, std::move(reg));
    };
    add("cpu", &CpuCall::id, &k_encode_call_info<0>);
#if defined(MYMYR_HAS_CUDA)
    add("CUDA", &CudaCall::id, &k_encode_call_info<1>);
#endif
#endif
    return out;
}
}  // namespace

void bind_encoding_jax(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_encoding_jax", "The XLA FFI target of the relational encoder "
                                                          "(mymyr.encoding.jax)");
    m.def(
        "register",
        [](Arg<ann::Any> encoder) {
            auto e = std::make_shared<EncoderEntry>(flat_encoder_of(encoder));
            return ffi_registry().add(std::move(e));
        },
        "encoder"_a,
        "Enters a FlatRelationEncoder into the FFI handle registry and returns its handle (a new one per call). The "
        "registry keeps the encoder until unregister(handle).");
    m.def(
        "unregister",
        [](i64 handle) {
            nb::gil_scoped_release release;  // the last reference destroys the entry, which waits for its devices
            ffi_registry().remove(handle);
        },
        "handle"_a,
        "Drops the registry's reference: calls traced afterwards fail; executables already compiled keep the encoder.");
    m.def(
        "info",
        [](i64 handle) {
            const EntryPtr e = entry_of(handle);
            const encoding::FlatEncoder& enc = *e->encoder;
            nb::dict d;
            d["instances"] = enc.table().size();
            d["derived_words"] = enc.context().max_derived_words();
#if defined(MYMYR_HAS_CUDA)
            try
            {
                cuda::DeviceEncoder::check_encoder(enc);
                d["device_refusal"] = nb::none();
            }
            catch (const std::invalid_argument& x)
            {
                d["device_refusal"] = std::string(x.what());
            }
            nb::list needs;
            for (const bool layer_masks : {false, true})
                for (const auto& [goal_masks, derived_goal_masks] : {std::pair{false, false}, {true, false}, {true, true}})
                    needs.append(cuda::DeviceEncoder::needs_derived(enc, goal_masks, derived_goal_masks, layer_masks));
            d["needs_derived"] = nb::tuple(needs);
            d["cuda"] = true;
#else
            d["device_refusal"] = nb::none();
            d["needs_derived"] = nb::none();
            d["cuda"] = false;
#endif
            return d;
        },
        "handle"_a,
        "What the Python side checks at trace time: instances, derived_words (the widest instance's), device_refusal "
        "(why the device does not encode the schema, or None), needs_derived (whether the device needs the rows' "
        "derived words without goal masks, with fluent goal masks, with derived goal masks too, then the same three "
        "with derived subgoal masks; None without the CUDA backend) and cuda (whether the CUDA target exists).");
    m.def(
        "scratch_bytes",
        [](i64 handle, u64 graphs, int device, bool derived) -> u64 {
#if defined(MYMYR_HAS_CUDA)
            const EntryPtr e = entry_of(handle);
            nb::gil_scoped_release release;
            cuda::DeviceGuard guard(device);
            return cuda::DeviceEncoder::scratch_bytes(
                *e->encoder, encoding::BatchShape{graphs, 0, 0, 0, 0},
                derived ? cuda::DerivedWords::Supplied : cuda::DerivedWords::Evaluated);
#else
            (void)handle;
            (void)graphs;
            (void)device;
            (void)derived;
            throw nb::value_error("mymyr: encoding: this mymyr was built without the CUDA backend");
#endif
        },
        "handle"_a, "graphs"_a, "device"_a, "derived"_a,
        "The device scratch of a batch of `graphs` rows on CUDA device `device`, for a call that passes its derived words "
        "(`derived`) or lets the device evaluate them (the result 'scratch' adds 255 bytes of alignment slack).");
    m.def(
        "prepare",
        [](i64 handle, int device) {
#if defined(MYMYR_HAS_CUDA)
            const EntryPtr e = entry_of(handle);
            nb::gil_scoped_release release;
            (void)e->device(device, true);
#else
            (void)handle;
            (void)device;
            throw nb::value_error("mymyr: encoding: this mymyr was built without the CUDA backend");
#endif
        },
        "handle"_a, "device"_a,
        "Uploads the encoder's context to CUDA device `device` now, as the first call of a compiled function on it "
        "does, and holds it for the encoder's lifetime.");
    m.def(
        "leaked_devices",
#if defined(MYMYR_HAS_CUDA)
        [] { return EncoderEntry::leaked_devices.load(); },
#else
        [] { return u64{0}; },
#endif
        "How many encoder slots on a device were leaked instead of freed because the device could not synchronize when "
        "the slot was destroyed (CUDA refuses the synchronization while another thread captures a CUDA graph).");
    m.def(
        "calls",
        [](i64 handle) {
            const EntryPtr e = entry_of(handle);
            nb::dict d;
            d["cpu"] = e->cpu_calls.load();
            d["cuda"] = e->cuda_calls.load();
            d["captured"] = e->captured_calls.load();
            d["pools"] = e->pools_built.load();
            return d;
        },
        "handle"_a,
        "How often the targets ran for this encoder: 'cpu', 'cuda' and 'captured' (CUDA calls traced into a command "
        "buffer, which XLA then replays without calling the target), and the worker pools the CPU target built "
        "('pools': one per thread count it ran with).");
    m.def(
        "has_cuda",
        [] {
#if defined(MYMYR_HAS_CUDA) && defined(MYMYR_HAS_XLA_FFI)
            return true;
#else
            return false;
#endif
        },
        "Whether the target has its CUDA handler (a CUDA build with the XLA FFI headers).");
    m.def("ffi_targets", &ffi_targets,
          "The XLA FFI handlers as (target name, platform, {stage: capsule}, command-buffer compatible) for "
          "jax.ffi.register_ffi_target; empty without the XLA FFI headers.");
    m.def("ffi_types", &ffi_types,
          "The FFI state types as (name, platform, {'type_id': capsule, 'type_info': capsule}) for "
          "jax.ffi.register_ffi_type.");
}
}  // namespace mymyr::python
