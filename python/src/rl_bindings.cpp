// mymyr._core._rl: batched RL primitives.
//   TaskTable(tasks)                          many instances of one domain
//   TaskSuite(tables), TaskSuite.group(tasks) instances of several domains: every entry point takes a suite or a
//                                             table (the suite of its domain)
//   expand(table, states, task_ids)           flat CSR + optional padded view, canonical order, labels
//   expand_into(table, states, task_ids, succ=..., ...)   destination-passing into caller arrays
//   is_goal / goal_test / goal_count           bulk goal tests
//   random_walks                              the native rollout reference loop
//   ThreadPool                                a user-held pool for splitting one batch over threads
//   CpuEnvPool                                EnvPool-style asynchronous environments on the CPU
// A Task or TaskHandle is accepted wherever a TaskTable is: its table of one (cached on the task core); task_ids may
// then be None. Task ids are global over a suite (rl/task_suite.hpp). expand / expand_into take CUDA device arrays in
// CUDA builds: the device path (device_hooks.hpp).
// Every array input is taken zero-copy, and every output is a view of one mymyr-owned block per call, exported to the
// caller's framework (arrays.hpp).

#include "arrays.hpp"
#include "device_hooks.hpp"
#include "dlpack.hpp"
#include "py_table.hpp"
#include "py_task.hpp"
#include "rl_imports.hpp"
#include "rl_typing.hpp"

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/pool.hpp"
#include "mymyr/task/workspace.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>

namespace mymyr::python
{
using namespace nb::literals;

DeviceHooks& device_hooks()
{
    static DeviceHooks h;
    return h;
}

bool is_cuda_array(nb::handle obj) { return !obj.is_none() && dl::dlpack_device(obj).device_type == dl::k_cuda; }

// ------------------------------------------------------------------------------------------------ tables (py_table.hpp)

TableRef table_of(nb::handle obj)
{
    if (nb::isinstance<PyTable>(obj))
    {
        PyTable* p = nb::inst_ptr<PyTable>(obj);
        return {p->table, p, nullptr, nb::borrow(obj)};
    }
    if (nb::isinstance<PyTask>(obj) || nb::isinstance<PyHandle>(obj))
    {
        const Owner o = owner_of(obj);
        return {o.core->table(), nullptr, o.core, o.obj};
    }
    throw nb::type_error(("mymyr: expected a mymyr.rl.TaskTable, a Task or a TaskHandle, not " +
                          nb::cast<std::string>(nb::type_name(obj.type())))
                             .c_str());
}

SuiteRef suite_of(nb::handle obj)
{
    if (nb::isinstance<PySuite>(obj))
    {
        PySuite* p = nb::inst_ptr<PySuite>(obj);
        return {p->suite, p, nullptr, nullptr, nb::borrow(obj)};
    }
    if (nb::isinstance<PyTable>(obj))
    {
        PyTable* p = nb::inst_ptr<PyTable>(obj);
        return {p->suite, nullptr, p, nullptr, nb::borrow(obj)};
    }
    if (nb::isinstance<PyTask>(obj) || nb::isinstance<PyHandle>(obj))
    {
        const Owner o = owner_of(obj);
        return {o.core->suite(), nullptr, nullptr, o.core, o.obj};
    }
    throw nb::type_error(("mymyr: expected a mymyr.rl.TaskSuite, a TaskTable, a Task or a TaskHandle, not " +
                          nb::cast<std::string>(nb::type_name(obj.type())))
                             .c_str());
}

u32 current_words(const rl::TaskTable& table) noexcept
{
    u32 w = 1;
    for (const auto& inst : table.instances())
        w = std::max(w, inst.task->words());
    return w;
}

u32 current_words(const rl::TaskSuite& suite) noexcept
{
    u32 w = 1;
    for (const rl::TaskTablePtr& t : suite.tables())
        w = std::max(w, current_words(*t));
    return w;
}

const i32* import_task_ids(nb::handle obj, u64 rows, u32 instances, std::vector<i32>& copy, std::shared_ptr<void>& keep,
                           const char* name, const char* noun)
{
    if (obj.is_none())
        return nullptr;
    const std::string q = std::string("mymyr: '") + name + "'";
    auto checked = [&](i64 v, u64 row) {
        if (v < 0 || v >= static_cast<i64>(instances))
            throw nb::value_error(("mymyr: task id " + std::to_string(v) + " of row " + std::to_string(row) +
                                   " is outside the " + noun + "'s " + std::to_string(instances) + " instances")
                                      .c_str());
        return static_cast<i32>(v);
    };
    const dl::Device dev = dl::dlpack_device(obj);
    if (dev.device_type < 0)
    {
        if (!nb::isinstance<nb::sequence>(obj) || nb::isinstance<nb::str>(obj))
            throw nb::type_error((q + " must be an int32 array [N] or a sequence of ints").c_str());
        const nb::sequence seq = nb::borrow<nb::sequence>(obj);
        if (nb::len(seq) != rows)
            throw nb::value_error((q + " has " + std::to_string(nb::len(seq)) + " entries; the batch has " +
                                   std::to_string(rows) + " rows")
                                      .c_str());
        copy.resize(rows);
        for (u64 r = 0; r < rows; ++r)
            copy[r] = checked(nb::cast<i64>(seq[r]), r);
        return copy.data();
    }
    if (!dl::host_accessible(dev.device_type))
        throw nb::type_error((q + " must be a host array, as the states").c_str());
    dl::Imported im = dl::import_dlpack(obj, dl::k_stream_default);
    const bool is_int = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.lanes == 1;
    if (!is_int || (im.dtype.bits != 32 && im.dtype.bits != 64))
        throw nb::type_error((q + " must hold int32 (or int64) instance indices").c_str());
    if (im.shape.size() != 1 || static_cast<u64>(im.shape[0]) != rows)
        throw nb::value_error((q + " must have shape [" + std::to_string(rows) + "] (one instance per row)").c_str());
    const i64 stride = rows > 1 ? im.strides[0] : 1;
    if (im.dtype.bits == 32 && stride == 1 && im.dtype.code == dl::k_int)
    {
        keep = std::move(im.keep);
        return static_cast<const i32*>(im.data);
    }
    copy.resize(rows);
    for (u64 r = 0; r < rows; ++r)
    {
        const u64 at = r * static_cast<u64>(stride);
        i64 v = 0;
        if (im.dtype.bits == 32)
            v = im.dtype.code == dl::k_int ? i64{static_cast<const i32*>(im.data)[at]} : i64{static_cast<const u32*>(im.data)[at]};
        else if (im.dtype.code == dl::k_int)
            v = static_cast<const i64*>(im.data)[at];
        else
        {
            const u64 u = static_cast<const u64*>(im.data)[at];
            v = u > static_cast<u64>(std::numeric_limits<i64>::max()) ? -1 : static_cast<i64>(u);
        }
        copy[r] = checked(v, r);
    }
    return copy.data();
}

rl::DeadEnd parse_dead_end(const std::string& s)
{
    if (s == "no_successors")
        return rl::DeadEnd::NoSuccessors;
    if (s == "none")
        return rl::DeadEnd::None;
    throw nb::value_error("mymyr: dead_end must be 'no_successors' or 'none'");
}

const char* dead_end_name(rl::DeadEnd d) noexcept { return d == rl::DeadEnd::None ? "none" : "no_successors"; }

namespace
{
struct PyPool
{
    explicit PyPool(u32 threads) : pool(std::make_unique<ThreadPool>(threads)) {}
    std::unique_ptr<ThreadPool> pool;
    rl::ExpandScratch scratch;
    std::mutex mutex;  // one batch at a time (the scratch is shared by the pool's members)
};

/// The flat result of one expand call: one block holding every array.
struct FlatResult
{
    std::shared_ptr<Block> block;
    u64 rows = 0, capacity = 0, total = 0;
    u32 words = 0, label_width = 0, words_needed = 0;
    u32 numeric_words = 0;  // numeric tables: rows are [words | numeric_words]
    u64 off_succ = 0, off_parent = 0, off_schema = 0, off_binding = 0, off_goal = 0, off_offsets = 0;
    bool has_goal = false;

    [[nodiscard]] u64 valid() const noexcept { return std::min(total, capacity); }
    template<class T>
    [[nodiscard]] T* at(u64 off) const noexcept
    {
        return reinterpret_cast<T*>(block->data() + off);
    }
    [[nodiscard]] rl::Expansion view() const
    {
        rl::Expansion x;
        x.capacity = capacity;
        x.words = words;
        x.label_width = label_width;
        x.succ = at<u64>(off_succ);
        x.parent = at<i32>(off_parent);
        x.schema = at<i32>(off_schema);
        x.binding = at<i32>(off_binding);
        x.goal = has_goal ? at<u8>(off_goal) : nullptr;
        x.offsets = at<i32>(off_offsets);
        x.total = total;
        x.words_needed = words_needed;
        x.numeric_words = numeric_words;
        return x;
    }
    [[nodiscard]] i64 row_words() const noexcept { return static_cast<i64>(words) + numeric_words; }
};

FlatResult allocate_flat(u64 rows, u64 cap, u32 W, u32 NN, u32 L, bool goal)
{
    FlatResult r;
    BlockLayout lay;
    r.rows = rows;
    r.capacity = cap;
    r.words = W;
    r.numeric_words = NN;
    r.label_width = L;
    r.has_goal = goal;
    r.off_succ = lay.add(cap * (W + NN) * sizeof(u64));
    r.off_parent = lay.add(cap * sizeof(i32));
    r.off_schema = lay.add(cap * sizeof(i32));
    r.off_binding = lay.add(cap * L * sizeof(i32));
    r.off_goal = lay.add(goal ? cap : 0);
    r.off_offsets = lay.add((rows + 1) * sizeof(i32));
    r.block = Block::make(lay.bytes);
    return r;
}

struct PaddedResult
{
    std::shared_ptr<Block> block;
    u64 rows = 0;
    u32 K = 0, words = 0, label_width = 0, numeric_words = 0;
    u64 off_index = 0, off_mask = 0, off_count = 0, off_succ = 0, off_schema = 0, off_binding = 0, off_goal = 0;
    bool has_goal = false, overflow = false;
    Framework fw = Framework::Numpy;
    WordEncoding enc;
};

PaddedResult make_padded(const FlatResult& f, u32 K, Framework fw, WordEncoding enc)
{
    PaddedResult p;
    BlockLayout lay;
    p.rows = f.rows;
    p.K = K;
    p.words = f.words;
    p.numeric_words = f.numeric_words;
    p.label_width = f.label_width;
    p.has_goal = f.has_goal;
    p.fw = fw;
    p.enc = enc;
    const u64 NK = f.rows * K;
    p.off_index = lay.add(NK * sizeof(i32));
    p.off_mask = lay.add(NK);
    p.off_count = lay.add(f.rows * sizeof(i32));
    p.off_succ = lay.add(NK * (f.words + f.numeric_words) * sizeof(u64));
    p.off_schema = lay.add(NK * sizeof(i32));
    p.off_binding = lay.add(NK * f.label_width * sizeof(i32));
    p.off_goal = lay.add(f.has_goal ? NK : 0);
    p.block = Block::make(lay.bytes);
    std::byte* b = p.block->data();
    rl::PaddedExpansion out;
    out.K = K;
    out.index = reinterpret_cast<i32*>(b + p.off_index);
    out.mask = reinterpret_cast<u8*>(b + p.off_mask);
    out.count = reinterpret_cast<i32*>(b + p.off_count);
    out.words = f.words;
    out.numeric_words = f.numeric_words;
    out.succ = reinterpret_cast<u64*>(b + p.off_succ);
    out.schema = reinterpret_cast<i32*>(b + p.off_schema);
    out.label_width = f.label_width;
    out.binding = reinterpret_cast<i32*>(b + p.off_binding);
    out.goal = f.has_goal ? reinterpret_cast<u8*>(b + p.off_goal) : nullptr;
    {
        nb::gil_scoped_release release;
        rl::pad(f.view(), f.rows, out);
    }
    p.overflow = out.overflow;
    return p;
}

u32 auto_K(const FlatResult& f)
{
    const i32* off = f.at<i32>(f.off_offsets);
    i32 m = 1;
    for (u64 i = 0; i < f.rows; ++i)
        m = std::max(m, off[i + 1] - off[i]);
    return std::bit_ceil(static_cast<u32>(m));
}

/// The Python-visible expansion: arrays are exported on access (zero-copy views of the result block).
struct PyExpansion
{
    FlatResult flat;
    Framework fw = Framework::Numpy;
    WordEncoding enc;
    SuiteRef ref;               // for action() (the owner of row i's instance)
    std::vector<i32> task_ids;  // the batch's instances (tables of several instances; empty: instance 0)
    std::optional<PaddedResult> padded;
    bool single = false;

    nb::object array(u64 off, rl::DType dt, std::vector<i64> shape, bool words = false) const
    {
        ArraySpec s;
        s.owner = std::shared_ptr<const void>(flat.block, flat.block->data() + off);
        s.data = flat.block->data() + off;
        s.dtype = dt;
        s.shape = std::move(shape);
        s.words = words;
        return export_array(std::move(s), fw, enc);
    }
    [[nodiscard]] u32 instance_of_state(u64 i) const { return task_ids.empty() ? 0u : static_cast<u32>(task_ids[i]); }
};

nb::object padded_array(const PaddedResult& p, u64 off, rl::DType dt, std::vector<i64> shape, bool words = false)
{
    ArraySpec s;
    s.owner = std::shared_ptr<const void>(p.block, p.block->data() + off);
    s.data = p.block->data() + off;
    s.dtype = dt;
    s.shape = std::move(shape);
    s.words = words;
    return export_array(std::move(s), p.fw, p.enc);
}

rl::ExpandOptions expand_options(bool canonical, bool witness, bool validate)
{
    rl::ExpandOptions o;
    o.canonical_order = canonical;
    o.witness_pruning = witness;
    o.validate = validate;
    return o;
}

/// Output framework and word encoding: the input's, unless a framework is named.
std::pair<Framework, WordEncoding> output_kind(const StateBatch& in, nb::handle framework, bool from_array)
{
    if (!framework.is_none())
    {
        const Framework fw = parse_framework(framework, nb::none());
        // another framework gets its own word encoding; "dlpack" is framework-neutral and keeps the input's
        if (!from_array || (fw != in.fw && fw != Framework::DLPack))
            return {fw, default_words(fw)};
        return {fw, in.enc};
    }
    if (!from_array)
        return {Framework::Numpy, default_words(Framework::Numpy)};
    return {in.fw, in.enc};
}

void run_expand(const rl::TaskSuite& table, const rl::StateBatchView& in, const i32* ids, rl::Expansion& x,
                const rl::ExpandOptions& o, PyPool* pool, u32 threads)
{
    nb::gil_scoped_release release;
    if (pool)
    {
        std::lock_guard lock(pool->mutex);
        rl::expand(table, in, ids, x, o, *pool->pool, pool->scratch);
    }
    else if (threads > 1)
    {
        ThreadPool tmp(threads);
        rl::ExpandScratch scratch;
        rl::expand(table, in, ids, x, o, tmp, scratch);
    }
    else
        rl::expand(table, in, ids, x, o);
}

/// The rows of a batch over a suite: [N, words + NN] (States packed at the suite's current width).
StateBatch import_table_states(nb::handle states, const rl::TaskSuite& t)
{
    return import_rows(states, current_words(t), t.numeric_words());
}

/// The task ids of a batch (see import_task_ids) with what keeps them alive.
struct TaskIds
{
    const i32* ptr = nullptr;
    std::vector<i32> copy;
    std::shared_ptr<void> keep;
    TaskIds(nb::handle obj, u64 rows, const rl::TaskSuite& t)
    {
        ptr = import_task_ids(obj, rows, t.size(), copy, keep, "task_ids", t.noun());
    }
    TaskIds(const TaskIds&) = delete;
    TaskIds& operator=(const TaskIds&) = delete;
};

// ------------------------------------------------------------------------------------------------ argument types
// (typing.hpp, py_task.hpp, py_table.hpp: rendered in the stubs; the functions check their arguments themselves)
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using SizeArg = Arg<u64>;
using PoolArg = Arg<PyPool>;
using ArrayArg = Arg<ann::ArrayLike>;
/// A mymyr.cuda.Context: typing.Any, since CPU builds have no mymyr.cuda (streams: StreamArg, typing.hpp).
using ContextArg = Arg<ann::Any>;
using ArrayOut = Arg<ann::Any>;
/// Host inputs give an Expansion; device inputs (CUDA builds) a mymyr.cuda.DeviceExpansion.
using ExpansionOut = Arg<std::variant<PyExpansion, ann::Any>>;
using ResultDict = nb::typed<nb::dict, std::string, ann::Any>;
using BundleDict = nb::typed<nb::dict, std::string, ann::Any>;

PyPool* pool_arg(nb::handle pool)
{
    if (pool.is_none())
        return nullptr;
    if (!nb::isinstance<PyPool>(pool))
        throw nb::type_error("mymyr: pool must be a mymyr.rl.ThreadPool");
    return nb::inst_ptr<PyPool>(pool);
}

ExpansionOut expand(SuiteArg table, StatesLike states, TaskIdsArg task_ids, SizeArg capacity, SizeArg words, SizeArg K,
                    bool goal, bool canonical, bool witness, PoolArg pool, u32 threads, FrameworkArg framework,
                    bool validate, StreamArg stream, ContextArg ctx)
{
    if (device_hooks().expand && is_cuda_array(states))
        return device_hooks().expand(
            {table, states, task_ids, capacity, words, K, goal, canonical, witness, framework, validate, stream, ctx});
    SuiteRef ref = suite_of(table);
    const rl::TaskSuite& tt = *ref.suite;
    StateBatch in = import_table_states(states, tt);
    const bool from_array = !in.packed;
    const auto [fw, enc] = output_kind(in, framework, from_array);
    PyPool* p = pool_arg(pool);
    const rl::ExpandOptions opt = expand_options(canonical, witness, validate);
    const u64 N = in.view.rows;
    const TaskIds ids(task_ids, N, tt);
    const u32 L = tt.label_width();
    const bool fixed_cap = !capacity.is_none();
    const bool fixed_words = !words.is_none();
    std::atomic<u32>& branching = ref.branching();
    u64 cap = fixed_cap ? nb::cast<u64>(capacity) : N * branching.load(std::memory_order_relaxed) + 16;
    u32 W = fixed_words ? nb::cast<u32>(words) : std::max(in.view.words, current_words(tt));
    if (W == 0)
        W = 1;
    FlatResult f;
    for (int attempt = 0;; ++attempt)
    {
        f = allocate_flat(N, cap, W, tt.numeric_words(), L, goal);
        rl::Expansion x = f.view();
        run_expand(tt, in.view, ids.ptr, x, opt, p, threads);
        f.total = x.total;
        f.words_needed = x.words_needed;
        const bool short_rows = x.total > cap && !fixed_cap;
        const bool narrow = x.words_needed > W && !fixed_words;
        if ((!short_rows && !narrow) || attempt >= 3)
            break;
        cap = std::max(cap, x.total);
        W = std::max(W, x.words_needed);
    }
    if (N > 0 && !fixed_cap)
    {
        // Sizing estimate for the next call: 1.25 x this batch's mean branching, rounded up, plus one. It grows at
        // once (a short estimate costs a rerun) and shrinks with hysteresis after batches of 16 or more states (an
        // oversized block costs cache: 5-10% of bulk_expand at 32 threads). It is stored only when it changes, so
        // steady calls from many threads do not write the shared line; a lost race only affects sizing.
        const u32 est = static_cast<u32>(std::min<u64>((f.total * 5 + 4 * N - 1) / (4 * N) + 1, 1u << 20));
        const u32 cur = branching.load(std::memory_order_relaxed);
        if (est > cur || (N >= 16 && cur > est + 1))
            branching.store(est, std::memory_order_relaxed);
    }
    PyExpansion e;
    e.flat = std::move(f);
    e.fw = fw;
    e.enc = enc;
    e.ref = std::move(ref);
    if (ids.ptr && tt.size() > 1)
        e.task_ids.assign(ids.ptr, ids.ptr + N);
    e.single = in.single;
    if (!K.is_none())
    {
        const u32 k = nb::cast<u32>(K);
        e.padded = make_padded(e.flat, k == 0 ? auto_K(e.flat) : k, fw, enc);
    }
    return nb::cast(std::move(e), nb::rv_policy::move);
}

/// A destination array as a typed pointer, checking dtype, rank and contiguity.
template<class T>
T* dest_ptr(const Dest& d, const char* name, rl::DType want, rl::DType also, usize ndim)
{
    if (d.dtype != want && d.dtype != also)
        throw nb::type_error((std::string("mymyr: '") + name + "' has the wrong dtype").c_str());
    if (d.shape.size() != ndim)
        throw nb::type_error((std::string("mymyr: '") + name + "' has the wrong number of dimensions").c_str());
    i64 expect = 1;
    for (usize k = ndim; k-- > 0;)
    {
        if (d.shape[k] > 1 && d.strides[k] != expect)
            throw nb::type_error((std::string("mymyr: '") + name + "' must be C-contiguous").c_str());
        expect *= d.shape[k];
    }
    return static_cast<T*>(d.data);
}

ResultDict expand_into(SuiteArg table, StatesLike states, TaskIdsArg task_ids, ArrayArg succ, ArrayArg parent,
                       ArrayArg schema, ArrayArg binding, ArrayArg goal, ArrayArg offsets, bool canonical, bool witness,
                       PoolArg pool, bool validate, StreamArg stream, ContextArg ctx)
{
    if (device_hooks().expand_into && (is_cuda_array(states) || is_cuda_array(succ) || is_cuda_array(parent) ||
                                       is_cuda_array(schema) || is_cuda_array(binding) || is_cuda_array(goal) ||
                                       is_cuda_array(offsets)))
        return device_hooks().expand_into({table, states, task_ids, succ, parent, schema, binding, goal, offsets,
                                           canonical, witness, validate, stream, ctx});
    SuiteRef ref = suite_of(table);
    const rl::TaskSuite& tt = *ref.suite;
    StateBatch in = import_table_states(states, tt);
    const TaskIds ids(task_ids, in.view.rows, tt);
    const u32 NN = tt.numeric_words();
    rl::Expansion x;
    x.numeric_words = NN;
    std::optional<u64> cap;
    auto take_cap = [&](i64 n) { cap = cap ? std::min<u64>(*cap, static_cast<u64>(n)) : static_cast<u64>(n); };
    std::vector<Dest> keep;
    if (!succ.is_none())
    {
        Dest d = import_dest(succ, "succ");
        if (d.shape.size() != 2)
            throw nb::type_error("mymyr: 'succ' must be [capacity, W] (64-bit) or [capacity, 2W] (32-bit)");
        const u32 isz = rl::dtype_bytes(d.dtype);
        if (isz == 8)
            x.succ = dest_ptr<u64>(d, "succ", rl::DType::U64, rl::DType::I64, 2), x.words = static_cast<u32>(d.shape[1]);
        else if (isz == 4)
        {
            x.succ = reinterpret_cast<u64*>(dest_ptr<u32>(d, "succ", rl::DType::U32, rl::DType::I32, 2));
            if (d.shape[1] % 2 != 0 || reinterpret_cast<std::uintptr_t>(d.data) % 8 != 0)
                throw nb::type_error("mymyr: 32-bit 'succ' needs an even number of columns and 8-byte alignment");
            x.words = static_cast<u32>(d.shape[1] / 2);
        }
        else
            throw nb::type_error("mymyr: 'succ' must hold 64-bit or 32-bit integers");
        if (x.words <= NN && NN)
            throw nb::value_error("mymyr: 'succ' of a numeric table needs W + NN columns (atom words, then numeric words)");
        x.words -= NN;
        take_cap(d.shape[0]);
        keep.push_back(std::move(d));
    }
    if (!parent.is_none())
    {
        Dest d = import_dest(parent, "parent");
        x.parent = dest_ptr<i32>(d, "parent", rl::DType::I32, rl::DType::I32, 1);
        take_cap(d.shape[0]);
        keep.push_back(std::move(d));
    }
    if (!schema.is_none())
    {
        Dest d = import_dest(schema, "schema");
        x.schema = dest_ptr<i32>(d, "schema", rl::DType::I32, rl::DType::I32, 1);
        take_cap(d.shape[0]);
        keep.push_back(std::move(d));
    }
    if (!binding.is_none())
    {
        Dest d = import_dest(binding, "binding");
        x.binding = dest_ptr<i32>(d, "binding", rl::DType::I32, rl::DType::I32, 2);
        x.label_width = static_cast<u32>(d.shape[1]);
        take_cap(d.shape[0]);
        keep.push_back(std::move(d));
    }
    if (!goal.is_none())
    {
        Dest d = import_dest(goal, "goal");
        x.goal = dest_ptr<u8>(d, "goal", rl::DType::Bool, rl::DType::U8, 1);
        take_cap(d.shape[0]);
        keep.push_back(std::move(d));
    }
    if (!offsets.is_none())
    {
        Dest d = import_dest(offsets, "offsets");
        x.offsets = dest_ptr<i32>(d, "offsets", rl::DType::I32, rl::DType::I32, 1);
        if (static_cast<u64>(d.shape[0]) < in.view.rows + 1)
            throw nb::value_error("mymyr: 'offsets' needs N + 1 entries");
        keep.push_back(std::move(d));
    }
    x.capacity = cap.value_or(0);
    run_expand(tt, in.view, ids.ptr, x, expand_options(canonical, witness, validate), pool_arg(pool), 1);
    nb::dict r;
    r["total"] = x.total;
    r["words_needed"] = x.words_needed;
    r["overflow"] = x.overflow();
    r["capacity"] = x.capacity;
    return r;
}

nb::object bool_array(std::shared_ptr<Block> block, u64 n, Framework fw)
{
    ArraySpec s;
    s.owner = block;
    s.data = block->data();
    s.dtype = rl::DType::Bool;
    s.shape = {static_cast<i64>(n)};
    return export_array(std::move(s), fw, default_words(fw));
}

std::string words_repr(const PyExpansion& e)
{
    return "Expansion(states=" + std::to_string(e.flat.rows) + ", successors=" + std::to_string(e.flat.total) +
           ", words=" + std::to_string(e.flat.words) + (e.flat.total > e.flat.capacity || e.flat.words_needed > e.flat.words ? ", overflow" : "") +
           ", framework=" + framework_name(e.fw) + ")";
}

// ------------------------------------------------------------------------------------------------ TaskTable, TaskSuite

/// A hash of the instances' assigned slot counts (the metadata snapshots are rebuilt when lazy slots grew).
u64 slots_key(const rl::TaskSuite& s)
{
    u64 h = 1469598103934665603ull;
    for (u32 g = 0; g < s.size(); ++g)
    {
        const auto& a = s.task(g)->atoms();
        h = (h ^ ((static_cast<u64>(a.fluent_slots()) << 32) | a.derived_slots())) * 1099511628211ull;
    }
    return h;
}

std::shared_ptr<const rl::ArrayBundle> table_metadata(PyTable& t)
{
    std::lock_guard lock(t.meta_mutex);
    const u64 key = slots_key(*t.suite);
    if (!t.meta || t.meta_key != key)
    {
        t.meta = std::make_shared<const rl::ArrayBundle>(rl::table_atom_metadata(*t.table));
        t.meta_key = key;
    }
    return t.meta;
}

std::shared_ptr<const rl::ArrayBundle> suite_metadata(PySuite& t)
{
    std::lock_guard lock(t.meta_mutex);
    const u64 key = slots_key(*t.suite);
    if (!t.meta || t.meta_key != key)
    {
        t.meta = std::make_shared<const rl::ArrayBundle>(rl::suite_atom_metadata(*t.suite));
        t.meta_key = key;
    }
    return t.meta;
}

/// An [I, W] u64 word array of the instances' goal masks (a fresh block; `pos`: the positive literals).
nb::object goal_rows(const rl::TaskSuite& s, bool pos, Framework fw)
{
    for (u32 g = 0; g < s.size(); ++g)
        if (s.instance(g).goal_derived)
            throw nb::value_error(("mymyr: the goal of instance " + std::to_string(g) +
                                   " mentions derived predicates; a mask test cannot decide it (use rl.is_goal)")
                                      .c_str());
    const u64 I = s.size(), W = s.words();
    auto block = Block::make(std::max<u64>(I * W, 1) * sizeof(u64));
    auto* w = reinterpret_cast<u64*>(block->data());
    for (u64 g = 0; g < I; ++g)
    {
        const std::vector<u64>& r = pos ? s.instance(static_cast<u32>(g)).goal_pos : s.instance(static_cast<u32>(g)).goal_neg;
        std::fill_n(w + g * W, W, u64{0});
        std::copy_n(r.data(), std::min<u64>(r.size(), W), w + g * W);
    }
    ArraySpec a{block, w, rl::DType::U64, {static_cast<i64>(I), static_cast<i64>(W)}, {}, true, true};
    return export_array(std::move(a), fw, default_words(fw));
}

/// The instances' initial states as rows [I, row_words] (a fresh block; TaskSuite::initial_row).
nb::object initial_rows(const rl::TaskSuite& s, Framework fw)
{
    const u64 I = s.size(), RW = s.row_words();
    auto block = Block::make(std::max<u64>(I * RW, 1) * sizeof(u64));
    auto* w = reinterpret_cast<u64*>(block->data());
    for (u64 g = 0; g < I; ++g)
        s.initial_row(static_cast<u32>(g), w + g * RW, s.words(), s.numeric_words());
    ArraySpec a{block, w, rl::DType::U64, {static_cast<i64>(I), static_cast<i64>(RW)}, {}, true, true};
    return export_array(std::move(a), fw, default_words(fw));
}

nb::typed<nb::tuple, ann::Any, ann::Any> goal_masks_of(const rl::TaskSuite& s, nb::handle framework)
{
    const Framework fw = parse_framework(framework, nb::none());
    return nb::typed<nb::tuple, ann::Any, ann::Any>(nb::make_tuple(goal_rows(s, true, fw), goal_rows(s, false, fw)));
}

/// A new mymyr.rl.TaskTable over `table` (its instances' cores and Task objects in table order).
nb::object new_table(rl::TaskTablePtr table, std::vector<CorePtr> cores, std::vector<nb::object> tasks)
{
    nb::object o = nb::inst_alloc(nb::type<PyTable>());
    auto* p = new (nb::inst_ptr<PyTable>(o)) PyTable();
    p->suite = rl::TaskSuite::of(table);
    p->table = std::move(table);
    p->cores = std::move(cores);
    p->tasks = std::move(tasks);
    nb::inst_mark_ready(o);
    return o;
}

/// The cores, Task objects and C++ tasks of a sequence of Tasks / TaskHandles (`what` names the caller in messages).
struct TaskList
{
    std::vector<CorePtr> cores;
    std::vector<nb::object> objs;
    std::vector<TaskPtr> tasks;

    TaskList(nb::handle seq, const char* what)
    {
        for (nb::handle h : seq)
        {
            if (!nb::isinstance<PyTask>(h) && !nb::isinstance<PyHandle>(h))
                throw nb::type_error((std::string("mymyr: ") + what + ": expected Tasks or TaskHandles").c_str());
            CorePtr c = nb::isinstance<PyTask>(h) ? nb::inst_ptr<PyTask>(h)->core : nb::inst_ptr<PyHandle>(h)->core;
            tasks.push_back(c->task);
            cores.push_back(std::move(c));
            objs.push_back(nb::borrow(h));
        }
    }
};

/// A TaskSuite's pickled state: its tasks in global order (unpickled as TaskSuite.group of them).
using SuiteState = nb::typed<nb::tuple, nb::typed<nb::list, PyTask>>;

/// Builds a suite of these tasks into `self` (uninitialized): TaskSuite.group, unpickling.
void init_grouped(PySuite* self, nb::handle tasks)
{
    TaskList l(tasks, "TaskSuite.group");
    rl::TaskSuitePtr s;
    {
        nb::gil_scoped_release release;
        s = rl::TaskSuite::group(l.tasks);
    }
    std::vector<nb::object> tables;
    for (u32 d = 0; d < s->num_domains(); ++d)
    {
        // domain d's table holds its instances in global order
        std::vector<CorePtr> cores;
        std::vector<nb::object> objs;
        for (u32 k = 0; k < s->table(d)->size(); ++k)
        {
            const u32 g = s->global_id(d, k);
            cores.push_back(l.cores[g]);
            objs.push_back(l.objs[g]);
        }
        tables.push_back(new_table(s->table(d), std::move(cores), std::move(objs)));
    }
    auto* p = new (self) PySuite();
    p->suite = std::move(s);
    p->tables = std::move(tables);
}

/// One list entry per instance (global order).
template<class F>
std::vector<u32> per_instance(const rl::TaskSuite& s, F&& f)
{
    std::vector<u32> v(s.size());
    for (u32 g = 0; g < s.size(); ++g)
        v[g] = f(g);
    return v;
}

const char* kind_name(i32 k) { return k == rl::k_pred_static ? "static" : k == rl::k_pred_fluent ? "fluent" : "derived"; }

nb::object table_reduce(nb::handle self)
{
    const PyTable& t = *nb::inst_ptr<PyTable>(self);
    nb::list tasks;
    for (const nb::object& o : t.tasks)
        tasks.append(task_object(o));
    return nb::make_tuple(self.type(), nb::make_tuple(tasks));
}

/// The Task of global instance i (Python index rules).
Arg<PyTask> suite_task(const SuiteRef& r, i64 i)
{
    const i64 n = r.suite->size();
    if (i < 0)
        i += n;
    if (i < 0 || i >= n)
        throw nb::index_error("mymyr: instance index out of range");
    return task_object(r.owner(static_cast<u32>(i)).obj);
}

// ------------------------------------------------------------------------------------------------ CpuEnvPool

/// A 1-d host integer array (32- or 64-bit, read in place when contiguous) or a sequence of ints (copied as int64).
struct HostInts
{
    const void* data = nullptr;
    u64 n = 0;
    u8 bits = 64;
    bool is_signed = true;
    std::shared_ptr<void> keep;

    [[nodiscard]] i64 at(u64 i) const noexcept
    {
        if (bits == 32)
            return is_signed ? i64{static_cast<const i32*>(data)[i]} : i64{static_cast<const u32*>(data)[i]};
        const u64 u = static_cast<const u64*>(data)[i];
        return is_signed ? static_cast<i64>(u) : (u > static_cast<u64>(std::numeric_limits<i64>::max()) ? -1 : static_cast<i64>(u));
    }
};

HostInts host_ints(nb::handle obj, const char* name)
{
    const std::string q = std::string("mymyr: '") + name + "'";
    HostInts h;
    const dl::Device dev = dl::dlpack_device(obj);
    if (dev.device_type < 0)
    {
        if (!nb::isinstance<nb::sequence>(obj) || nb::isinstance<nb::str>(obj))
            throw nb::type_error((q + " must be an integer array [n] or a sequence of ints").c_str());
        const nb::sequence seq = nb::borrow<nb::sequence>(obj);
        auto v = std::make_shared<std::vector<i64>>(nb::len(seq));
        for (usize i = 0; i < v->size(); ++i)
            (*v)[i] = nb::cast<i64>(seq[i]);
        h.data = v->data();
        h.n = v->size();
        h.keep = std::move(v);
        return h;
    }
    if (!dl::host_accessible(dev.device_type))
        throw nb::type_error((q + " must be a host array (the pool runs on the CPU)").c_str());
    dl::Imported im = dl::import_dlpack(obj, dl::k_stream_default);
    const bool is_int = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.lanes == 1;
    if (!is_int || (im.dtype.bits != 32 && im.dtype.bits != 64))
        throw nb::type_error((q + " must hold 32- or 64-bit integers").c_str());
    if (im.shape.size() != 1)
        throw nb::value_error((q + " must be one-dimensional").c_str());
    h.n = static_cast<u64>(im.shape[0]);
    h.bits = im.dtype.bits;
    h.is_signed = im.dtype.code == dl::k_int;
    if (h.n > 1 && im.strides[0] != 1)
        throw nb::type_error((q + " must be contiguous").c_str());
    h.data = im.data;
    h.keep = std::move(im.keep);
    return h;
}

/// The host arrays of a pool call.
struct PoolArgs
{
    std::vector<u32> env_copy;
    std::vector<i32> id_copy, next_copy;
    std::vector<i64> act64;
    std::shared_ptr<void> id_keep, next_keep;
    HostInts env, act;
    u64 n = 0;

    /// env_ids [n] (None: every env, n = num_envs, a null pointer).
    const u32* env_ids(nb::handle obj, u32 num_envs)
    {
        if (obj.is_none())
        {
            n = num_envs;
            return nullptr;
        }
        env = host_ints(obj, "env_ids");
        n = env.n;
        if (env.bits == 32)
            return static_cast<const u32*>(env.data);  // negative int32 ids become large: the pool rejects them
        env_copy.resize(n);
        for (u64 i = 0; i < n; ++i)
        {
            const i64 v = env.at(i);
            if (v < 0 || v >= static_cast<i64>(num_envs))
                throw nb::value_error(("mymyr: env id " + std::to_string(v) + " is outside the pool's " +
                                       std::to_string(num_envs) + " envs")
                                          .c_str());
            env_copy[i] = static_cast<u32>(v);
        }
        return env_copy.data();
    }

    /// actions [n] int64 or int32 (None: the random policy).
    rl::Actions actions(nb::handle obj)
    {
        if (obj.is_none())
            return {};
        act = host_ints(obj, "actions");
        if (act.n != n)
            throw nb::value_error(("mymyr: 'actions' has " + std::to_string(act.n) + " entries; the call steps " +
                                   std::to_string(n) + " envs")
                                      .c_str());
        if (act.is_signed && act.bits == 64)
            return {static_cast<const i64*>(act.data), nullptr};
        if (act.is_signed && act.bits == 32)
            return {nullptr, static_cast<const i32*>(act.data)};
        act64.resize(n);
        for (u64 i = 0; i < n; ++i)
            act64[i] = act.at(i);
        return {act64.data(), nullptr};
    }
};

/// The results of a pool call: the arrays of an rl::PoolBatch, exported on access (zero-copy, read-only views).
struct PyPoolBatch
{
    std::shared_ptr<const rl::PoolBatch> b;
    Framework fw = Framework::Numpy;

    template<class T>
    nb::object array(const rl::PoolArray<T>& v, rl::DType dt, std::vector<i64> shape, bool words = false) const
    {
        alignas(64) static const u64 empty[8] = {};
        ArraySpec s;
        const void* data = v.empty() ? static_cast<const void*>(empty) : static_cast<const void*>(v.data());
        s.owner = std::shared_ptr<const void>(b, data);
        s.data = data;
        s.dtype = dt;
        s.shape = std::move(shape);
        s.readonly = true;
        s.words = words;
        return export_array(std::move(s), fw, default_words(fw));
    }
    [[nodiscard]] i64 rows() const noexcept { return static_cast<i64>(b->rows); }
    /// Whether the batch holds step results (a reset's does not).
    [[nodiscard]] bool stepped() const noexcept { return b->step; }
};

/// The pool a batch hands its arrays back to when Python drops it (null once the pool is gone).
struct PoolRecycler
{
    std::mutex mu;
    rl::CpuEnvPool* pool = nullptr;
};

class PyCpuPool
{
public:
    PyCpuPool(SuiteRef ref, const rl::EnvConfig& cfg, u32 num_envs, const rl::PoolOptions& opt, Framework fw)
        : m_ref(std::move(ref)), m_fw(fw), m_recycler(std::make_shared<PoolRecycler>())
    {
        nb::gil_scoped_release release;
        m_pool = std::make_unique<rl::CpuEnvPool>(m_ref.suite, cfg, num_envs, opt);
        m_recycler->pool = m_pool.get();
    }
    ~PyCpuPool()
    {
        nb::gil_scoped_release release;  // the workers finish their steps
        {
            std::lock_guard lock(m_recycler->mu);
            m_recycler->pool = nullptr;
        }
        m_pool.reset();
    }
    PyCpuPool(const PyCpuPool&) = delete;
    PyCpuPool& operator=(const PyCpuPool&) = delete;

    [[nodiscard]] const rl::CpuEnvPool& pool() const noexcept { return *m_pool; }
    [[nodiscard]] const SuiteRef& ref() const noexcept { return m_ref; }
    [[nodiscard]] Framework framework() const noexcept { return m_fw; }

    PyPoolBatch reset(nb::handle env_ids, nb::handle task_ids, nb::handle goal_pos, nb::handle goal_neg)
    {
        PoolArgs a;
        const u32* envs = a.env_ids(env_ids, m_pool->num_envs());
        const i32* ids = import_task_ids(task_ids, a.n, m_ref.suite->size(), a.id_copy, a.id_keep, "task_ids",
                                         m_ref.suite->noun());
        rlimp::Imports in(-1, dl::k_stream_default, false);
        const u64* gp = in.words(goal_pos, "goal_pos", static_cast<i64>(a.n), m_pool->words(), false);
        const u64* gn = in.words(goal_neg, "goal_neg", static_cast<i64>(a.n), m_pool->words(), false);
        rl::PoolBatch b;
        {
            nb::gil_scoped_release release;
            b = m_pool->reset(envs, a.n, ids, gp, gn);
        }
        return wrap(std::move(b));
    }

    u64 send(nb::handle env_ids, nb::handle actions, nb::handle next_task_ids)
    {
        PoolArgs a;
        const u32* envs = a.env_ids(env_ids, m_pool->num_envs());
        const rl::Actions act = a.actions(actions);
        const i32* next = import_task_ids(next_task_ids, a.n, m_ref.suite->size(), a.next_copy, a.next_keep,
                                          "next_task_ids", m_ref.suite->noun());
        nb::gil_scoped_release release;
        return m_pool->send(envs, a.n, act, next);
    }

    PyPoolBatch recv(u64 min_rows)
    {
        rl::PoolBatch b;
        {
            nb::gil_scoped_release release;
            b = m_pool->recv(min_rows);
        }
        return wrap(std::move(b));
    }

    PyPoolBatch recv_ticket(u64 ticket)
    {
        rl::PoolBatch b;
        {
            nb::gil_scoped_release release;
            b = m_pool->recv_ticket(ticket);
        }
        return wrap(std::move(b));
    }

    PyPoolBatch step(nb::handle env_ids, nb::handle actions, nb::handle next_task_ids)
    {
        PoolArgs a;
        const u32* envs = a.env_ids(env_ids, m_pool->num_envs());
        const rl::Actions act = a.actions(actions);
        const i32* next = import_task_ids(next_task_ids, a.n, m_ref.suite->size(), a.next_copy, a.next_keep,
                                          "next_task_ids", m_ref.suite->noun());
        rl::PoolBatch b;
        {
            nb::gil_scoped_release release;
            b = m_pool->step(envs, a.n, act, next);
        }
        return wrap(std::move(b));
    }

private:
    /// A batch whose arrays go back to the pool (rl::CpuEnvPool::recycle) once the last Python reference is dropped.
    PyPoolBatch wrap(rl::PoolBatch&& b) const
    {
        std::weak_ptr<PoolRecycler> w = m_recycler;
        std::shared_ptr<const rl::PoolBatch> p(new rl::PoolBatch(std::move(b)),
                                               [w](const rl::PoolBatch* x)
                                               {
                                                   if (const auto r = w.lock())
                                                   {
                                                       std::lock_guard lock(r->mu);
                                                       if (r->pool)
                                                           r->pool->recycle(std::move(*const_cast<rl::PoolBatch*>(x)));
                                                   }
                                                   delete x;
                                               });
        return {std::move(p), m_fw};
    }

    SuiteRef m_ref;
    Framework m_fw;
    std::shared_ptr<PoolRecycler> m_recycler;
    std::unique_ptr<rl::CpuEnvPool> m_pool;
};
}  // namespace

void bind_rl(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_rl", "Batched RL primitives (mymyr.rl)");

    nb::class_<PyPool>(m, "ThreadPool",
                       "A pool of worker threads for splitting one batch over threads (rl.expand(..., pool=)). Workers "
                       "sleep between calls; one batch runs at a time. Hold it as long as you expand.")
        .def(nb::init<u32>(), "threads"_a)
        .def_prop_ro("threads", [](const PyPool& p) { return p.pool->size(); })
        .def("__repr__", [](const PyPool& p) { return "ThreadPool(" + std::to_string(p.pool->size()) + ")"; });

    // --- TaskTable
    nb::class_<PyTable>(
        m, "TaskTable",
        "Many instances of one domain: the table the RL entry points take (rl.expand, "
        "rl.is_goal, the torch and JAX environments, CpuEnvPool; a TaskSuite mixes domains). A batch over a table says "
        "the instance of each row (task_ids, int32 [N]); a Task is accepted wherever a TaskTable is (its table of "
        "one). The instances must have "
        "the same action schemas and domain predicates (in the same order); rows are words atom words wide (the "
        "widest instance's), plus numeric_words. Immutable; picklable (it pickles its tasks).")
        .def(
            "__init__",
            [](PyTable* self, nb::typed<nb::sequence, std::variant<PyTask, PyHandle>> tasks) {
                TaskList l(tasks, "TaskTable");
                rl::TaskTablePtr t;
                {
                    nb::gil_scoped_release release;
                    t = rl::TaskTable::create(std::move(l.tasks));
                }
                auto* p = new (self) PyTable();
                p->suite = rl::TaskSuite::of(t);
                p->table = std::move(t);
                p->cores = std::move(l.cores);
                p->tasks = std::move(l.objs);
            },
            "tasks"_a,
            "A table of the tasks' instances (instance i = tasks[i]). Raises ValueError for an empty list or tasks "
            "of different domains (TaskSuite.group mixes domains).")
        .def("__len__", [](const PyTable& t) { return t.table->size(); })
        .def(
            "__getitem__",
            [](const PyTable& t, i64 i) {
                const i64 n = t.table->size();
                if (i < 0)
                    i += n;
                if (i < 0 || i >= n)
                    throw nb::index_error("mymyr: instance index out of range");
                return task_object(t.tasks[static_cast<usize>(i)]);
            },
            "i"_a, "The Task of instance i.")
        .def_prop_ro(
            "tasks",
            [](const PyTable& t) {
                nb::typed<nb::list, PyTask> out{nb::list()};
                for (const nb::object& o : t.tasks)
                    out.append(task_object(o));
                return out;
            },
            "The instances' Tasks, in table order.")
        .def_prop_ro("words", [](const PyTable& t) { return t.table->words(); },
                     "Atom words of a row: the widest instance's max_words (at least 1).")
        .def_prop_ro("numeric_words", [](const PyTable& t) { return t.table->numeric_words(); },
                     "Numeric words of a row after the atom words (the most any instance has; 0 for classical tables).")
        .def_prop_ro("row_words", [](const PyTable& t) { return t.table->row_words(); }, "words + numeric_words.")
        .def_prop_ro("label_width", [](const PyTable& t) { return t.table->label_width(); }, "The largest schema arity.")
        .def_prop_ro("num_schemas", [](const PyTable& t) { return t.table->num_schemas(); })
        .def_prop_ro("max_objects", [](const PyTable& t) { return t.table->max_objects(); },
                     "The most objects of an instance (the columns of rl.prefix_masks over the table).")
        .def_prop_ro("numeric", [](const PyTable& t) { return t.table->numeric(); },
                     "Whether an instance has numeric fluents (numeric tables run on the CPU only).")
        .def_prop_ro("fingerprint", [](const PyTable& t) { return t.table->fingerprint(); },
                     "A hash of the instances' fingerprints in table order.")
        .def_prop_ro(
            "num_objects",
            [](const PyTable& t) {
                std::vector<u32> v;
                for (const auto& inst : t.table->instances())
                    v.push_back(inst.num_objects);
                return v;
            },
            "Objects per instance [I] (rl.object_offsets globalizes label objects with them).")
        .def_prop_ro(
            "instance_words",
            [](const PyTable& t) {
                std::vector<u32> v;
                for (const auto& inst : t.table->instances())
                    v.push_back(inst.words);
                return v;
            },
            "Atom words per instance [I] (each instance's max_words; its rows leave the words past them zero).")
        .def_prop_ro("schema_names", [](const PyTable& t) { return t.table->schema_names(); })
        .def_prop_ro("schema_arities", [](const PyTable& t) { return t.table->schema_arities(); })
        .def_prop_ro("predicate_names", [](const PyTable& t) { return t.table->predicate_names(); },
                     "Table predicates: the domain's (ids [0, num_domain_predicates)), then the problem-local ones.")
        .def_prop_ro("predicate_arities", [](const PyTable& t) { return t.table->predicate_arities(); })
        .def_prop_ro(
            "predicate_kinds",
            [](const PyTable& t) {
                std::vector<std::string> v;
                for (i32 k : t.table->predicate_kinds())
                    v.emplace_back(kind_name(k));
                return v;
            },
            "'static', 'fluent' or 'derived' per table predicate.")
        .def_prop_ro("predicate_instances", [](const PyTable& t) { return t.table->predicate_instances(); },
                     "Per table predicate: -1 for a domain predicate, else the instance of a problem-local one (a "
                     "derived predicate only the problem's axioms define, e.g. goal normalization's axiom_0).")
        .def_prop_ro("num_domain_predicates", [](const PyTable& t) { return t.table->num_domain_predicates(); },
                     "The domain predicates: table predicate ids [0, num_domain_predicates), a function of the domain "
                     "alone (every table over the domain gives them the same ids).")
        .def(
            "atom_metadata",
            [](PyTable& t, FrameworkArg framework) {
                const Framework fw = parse_framework(framework, nb::none());
                std::shared_ptr<const rl::ArrayBundle> b;
                {
                    nb::gil_scoped_release release;
                    b = table_metadata(t);
                }
                return nb::borrow<BundleDict>(export_bundle(b, fw, default_words(fw)));
            },
            "framework"_a = nb::none(),
            "The instances' atom metadata concatenated (rl/task_table.hpp table_atom_metadata; zero-copy views of a "
            "cached snapshot): atom_offsets [I + 1] (fluent slot j of instance i is row atom_offsets[i] + j), atom_pred "
            "(table predicate ids), atom_args [F, A] (instance-local objects, -1 padding), atom_cid, derived_* and "
            "static_* likewise, num_objects [I], object_offsets [I + 1], num_atoms [I], pred_arity / pred_kind / "
            "pred_instance [P], and scalars (num_instances, num_predicates, num_domain_predicates, ...).")
        .def(
            "goal_masks", [](const PyTable& t, FrameworkArg framework) { return goal_masks_of(*t.suite, framework); },
            "framework"_a = nb::none(),
            "(gpos, gneg) [I, words]: each instance's positive and negative fluent goal literals as state words "
            "(read-only). rl.goal_masks(table, task_ids) gathers per-env copies.")
        .def(
            "initial_states",
            [](const PyTable& t, FrameworkArg framework) {
                return ArrayOut(initial_rows(*t.suite, parse_framework(framework, nb::none())));
            },
            "framework"_a = nb::none(), "The instances' initial states as rows [I, row_words] (read-only).")
        .def("__reduce__", &table_reduce)
        .def("__repr__", [](const PyTable& t) {
            const auto& I = t.table->instances();
            const std::string domain = I.empty() ? "" : t.cores.front()->data->domain_name;
            return "TaskTable(" + domain + ", " + std::to_string(I.size()) + " instances, words=" +
                   std::to_string(t.table->words()) + ", max_objects=" + std::to_string(t.table->max_objects()) + ")";
        });

    // --- TaskSuite
    nb::class_<PySuite>(
        m, "TaskSuite",
        "Instances of several domains: one TaskTable per domain, taken by every "
        "RL entry point a TaskTable is (rl.expand, rl.is_goal, the torch and JAX environments, CpuEnvPool, the dataset "
        "and IW entry points). Task ids are global: int32 indices into the suite, [0, len(suite)); instance i is "
        "instance local_ids[i] of domain domain_of[i]'s table. Labels keep each domain's schema ids and instance-local "
        "objects (schema_offsets flattens them), atom metadata each domain's predicate ids. Rows are words atom words "
        "wide (the widest domain's), plus numeric_words. Immutable; picklable (it pickles its tasks).")
        .def(
            "__init__",
            [](PySuite* self, nb::typed<nb::sequence, PyTable> tables) {
                std::vector<rl::TaskTablePtr> ts;
                std::vector<nb::object> objs;
                for (nb::handle h : tables)
                {
                    if (!nb::isinstance<PyTable>(h))
                        throw nb::type_error("mymyr: TaskSuite: expected TaskTables (TaskSuite.group takes tasks)");
                    ts.push_back(nb::inst_ptr<PyTable>(h)->table);
                    objs.push_back(nb::borrow(h));
                }
                rl::TaskSuitePtr s;
                {
                    nb::gil_scoped_release release;
                    s = rl::TaskSuite::create(std::move(ts));
                }
                auto* p = new (self) PySuite();
                p->suite = std::move(s);
                p->tables = std::move(objs);
            },
            "tables"_a,
            "The suite of these domains, one table each (global ids: table 0's instances, then table 1's, ...). Raises "
            "ValueError for an empty list or two tables of one domain.")
        .def_static(
            "group",
            [](nb::typed<nb::sequence, std::variant<PyTask, PyHandle>> tasks) {
                nb::object o = nb::inst_alloc(nb::type<PySuite>());
                init_grouped(nb::inst_ptr<PySuite>(o), tasks);
                nb::inst_mark_ready(o);
                return Arg<PySuite>(o);
            },
            "tasks"_a,
            "The suite of these tasks (global id i = tasks[i]), one table per domain: each task joins the table of the "
            "first domain it is an instance of, the domains in the order they are first seen.")
        .def("__len__", [](const PySuite& t) { return t.suite->size(); })
        .def(
            "__getitem__", [](nb::handle self, i64 i) { return suite_task(suite_of(self), i); }, "i"_a,
            "The Task of global instance i.")
        .def_prop_ro(
            "tasks",
            [](nb::handle self) {
                const SuiteRef r = suite_of(self);
                nb::typed<nb::list, PyTask> out{nb::list()};
                for (u32 g = 0; g < r.suite->size(); ++g)
                    out.append(task_object(r.owner(g).obj));
                return out;
            },
            "The instances' Tasks, in global order.")
        .def_prop_ro(
            "tables",
            [](const PySuite& t) {
                nb::typed<nb::list, PyTable> out{nb::list()};
                for (const nb::object& o : t.tables)
                    out.append(o);
                return out;
            },
            "The domains' TaskTables (domain d's is tables[d]).")
        .def_prop_ro("num_domains", [](const PySuite& t) { return t.suite->num_domains(); })
        .def_prop_ro(
            "domain_names",
            [](const PySuite& t) {
                std::vector<std::string> v;
                for (u32 d = 0; d < t.suite->num_domains(); ++d)
                    v.push_back(t.suite->domain_name(d));
                return v;
            },
            "Each domain's name.")
        .def_prop_ro(
            "domain_of", [](const PySuite& t) { return per_instance(*t.suite, [&](u32 g) { return t.suite->domain_of(g); }); },
            "Per global instance: its domain [I] (rl.task_domains(suite, task_ids) gathers it per row).")
        .def_prop_ro(
            "local_ids", [](const PySuite& t) { return per_instance(*t.suite, [&](u32 g) { return t.suite->local_id(g); }); },
            "Per global instance: its index in its domain's table [I].")
        .def(
            "global_id", [](const PySuite& t, u32 domain, u32 local) { return t.suite->global_id(domain, local); },
            "domain"_a, "local"_a, "The global id of instance `local` of domain `domain`'s table.")
        .def_prop_ro("words", [](const PySuite& t) { return t.suite->words(); },
                     "Atom words of a row: the widest domain's table words.")
        .def_prop_ro("numeric_words", [](const PySuite& t) { return t.suite->numeric_words(); },
                     "Numeric words of a row after the atom words (the most any domain has).")
        .def_prop_ro("row_words", [](const PySuite& t) { return t.suite->row_words(); }, "words + numeric_words.")
        .def_prop_ro("label_width", [](const PySuite& t) { return t.suite->label_width(); },
                     "The largest schema arity of any domain.")
        .def_prop_ro("num_schemas", [](const PySuite& t) { return t.suite->max_schemas(); },
                     "The most schemas of any domain: the columns of rl.schema_masks (schema ids are the domain's).")
        .def_prop_ro("schema_offsets", [](const PySuite& t) { return t.suite->schema_offsets(); },
                     "[D + 1]: domain d's schemas are flat ids schema_offsets[d] + schema.")
        .def_prop_ro("max_objects", [](const PySuite& t) { return t.suite->max_objects(); },
                     "The most objects of an instance (the columns of rl.prefix_masks over the suite).")
        .def_prop_ro("numeric", [](const PySuite& t) { return t.suite->numeric(); })
        .def_prop_ro("fingerprint", [](const PySuite& t) { return t.suite->fingerprint(); },
                     "A hash of the domains' table fingerprints and the global ids (a one-domain suite: its table's).")
        .def_prop_ro(
            "num_objects",
            [](const PySuite& t) { return per_instance(*t.suite, [&](u32 g) { return t.suite->instance(g).num_objects; }); },
            "Objects per instance [I] (rl.object_offsets globalizes label objects with them).")
        .def_prop_ro(
            "instance_words",
            [](const PySuite& t) { return per_instance(*t.suite, [&](u32 g) { return t.suite->instance(g).words; }); },
            "Atom words per instance [I].")
        .def(
            "atom_metadata",
            [](PySuite& t, FrameworkArg framework) {
                const Framework fw = parse_framework(framework, nb::none());
                std::shared_ptr<const rl::ArrayBundle> b;
                {
                    nb::gil_scoped_release release;
                    b = suite_metadata(t);
                }
                return nb::borrow<BundleDict>(export_bundle(b, fw, default_words(fw)));
            },
            "framework"_a = nb::none(),
            "The instances' atom metadata in global order (rl/task_suite.hpp suite_atom_metadata; zero-copy views of "
            "a cached snapshot): TaskTable.atom_metadata()'s arrays with each domain's predicate ids (atom_pred, "
            "derived_pred, static_pred index domain[i]'s rows pred_offsets[d]:pred_offsets[d + 1] of pred_arity / "
            "pred_kind / pred_instance), plus domain [I], local_id [I], pred_offsets [D + 1], domain_predicates [D], "
            "schema_offsets [D + 1], schema_arity and the scalar num_domains.")
        .def(
            "goal_masks", [](const PySuite& t, FrameworkArg framework) { return goal_masks_of(*t.suite, framework); },
            "framework"_a = nb::none(),
            "(gpos, gneg) [I, words]: each instance's fluent goal literals as state words (read-only).")
        .def(
            "initial_states",
            [](const PySuite& t, FrameworkArg framework) {
                return ArrayOut(initial_rows(*t.suite, parse_framework(framework, nb::none())));
            },
            "framework"_a = nb::none(), "The instances' initial states as rows [I, row_words] (read-only).")
        .def("__getstate__",
             [](nb::handle self) {
                 const SuiteRef r = suite_of(self);
                 nb::list tasks;
                 for (u32 g = 0; g < r.suite->size(); ++g)
                     tasks.append(task_object(r.owner(g).obj));
                 return SuiteState(nb::make_tuple(tasks));
             })
        .def("__setstate__",
             [](PySuite& self, SuiteState state) { init_grouped(&self, state[0]); })
        .def("__repr__", [](const PySuite& t) {
            std::string names;
            for (u32 d = 0; d < t.suite->num_domains(); ++d)
                names += (d ? ", " : "") + t.suite->domain_name(d) + " (" + std::to_string(t.suite->table(d)->size()) + ")";
            return "TaskSuite(" + names + "; words=" + std::to_string(t.suite->words()) + ", max_objects=" +
                   std::to_string(t.suite->max_objects()) + ")";
        });

    nb::class_<PaddedResult>(m, "PaddedExpansion",
                             "The padded [N, K] view of an expansion: successor k of state i, with a validity mask and "
                             "the true counts (count > K is an overflow).")
        .def_prop_ro("K", [](const PaddedResult& p) { return p.K; })
        .def_prop_ro("overflow", [](const PaddedResult& p) { return p.overflow; })
        .def_prop_ro("index", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_index, rl::DType::I32, {static_cast<i64>(p.rows), p.K}));
        }, "[N, K] int32: row of the flat arrays, -1 = none")
        .def_prop_ro("mask", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_mask, rl::DType::Bool, {static_cast<i64>(p.rows), p.K}));
        }, "[N, K] bool")
        .def_prop_ro("count", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_count, rl::DType::I32, {static_cast<i64>(p.rows)}));
        }, "[N] int32 true successor counts")
        .def_prop_ro("succ", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_succ, rl::DType::U64,
                                {static_cast<i64>(p.rows), p.K, static_cast<i64>(p.words) + p.numeric_words}, true));
        }, "[N, K, W] state words (zero padding); numeric tables: [N, K, W + NN], the numeric words last")
        .def_prop_ro("schema", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_schema, rl::DType::I32, {static_cast<i64>(p.rows), p.K}));
        }, "[N, K] int32, -1 padding")
        .def_prop_ro("binding", [](const PaddedResult& p) {
            return ArrayOut(padded_array(p, p.off_binding, rl::DType::I32, {static_cast<i64>(p.rows), p.K, p.label_width}));
        }, "[N, K, L] int32 objects (of the row's instance), -1 padding")
        .def_prop_ro("goal", [](const PaddedResult& p) -> ArrayOut {
            if (!p.has_goal)
                return nb::none();
            return padded_array(p, p.off_goal, rl::DType::Bool, {static_cast<i64>(p.rows), p.K});
        }, "[N, K] bool goal flags (if requested)")
        .def("__repr__", [](const PaddedResult& p) {
            return "PaddedExpansion(states=" + std::to_string(p.rows) + ", K=" + std::to_string(p.K) +
                   (p.overflow ? ", overflow" : "") + ")";
        });

    nb::class_<PyExpansion>(m, "Expansion",
                            "The flat CSR expansion of a batch: successors of state i are rows "
                            "offsets[i]:offsets[i+1], in canonical order (schema, then binding). Arrays are zero-copy views "
                            "in the input's framework. A successor row's instance is its parent's (task_ids[parent]); its "
                            "label's objects are that instance's.")
        .def_prop_ro("succ", [](const PyExpansion& e) {
            return ArrayOut(e.array(e.flat.off_succ, rl::DType::U64, {static_cast<i64>(e.flat.valid()), e.flat.row_words()}, true));
        }, "[M, W] successor state words; numeric tables: [M, W + NN], the numeric words last")
        .def_prop_ro("parent", [](const PyExpansion& e) {
            return ArrayOut(e.array(e.flat.off_parent, rl::DType::I32, {static_cast<i64>(e.flat.valid())}));
        }, "[M] int32 index of the expanded state")
        .def_prop_ro("schema", [](const PyExpansion& e) {
            return ArrayOut(e.array(e.flat.off_schema, rl::DType::I32, {static_cast<i64>(e.flat.valid())}));
        }, "[M] int32 schema of the label")
        .def_prop_ro("binding", [](const PyExpansion& e) {
            return ArrayOut(e.array(e.flat.off_binding, rl::DType::I32, {static_cast<i64>(e.flat.valid()), e.flat.label_width}));
        }, "[M, L] int32 objects of the label (instance-local), -1 past the schema's arity")
        .def_prop_ro("offsets", [](const PyExpansion& e) {
            return ArrayOut(e.array(e.flat.off_offsets, rl::DType::I32, {static_cast<i64>(e.flat.rows + 1)}));
        }, "[N + 1] int32 CSR offsets (true counts, even past an overflow)")
        .def_prop_ro("goal", [](const PyExpansion& e) -> ArrayOut {
            if (!e.flat.has_goal)
                return nb::none();
            return e.array(e.flat.off_goal, rl::DType::Bool, {static_cast<i64>(e.flat.valid())});
        }, "[M] bool goal flags of the successors (expand(..., goal=True))")
        .def_prop_ro("counts", [](const PyExpansion& e) {
            auto block = Block::make(e.flat.rows * sizeof(i32));
            const i32* off = e.flat.at<i32>(e.flat.off_offsets);
            auto* c = reinterpret_cast<i32*>(block->data());
            for (u64 i = 0; i < e.flat.rows; ++i)
                c[i] = off[i + 1] - off[i];
            ArraySpec s{block, c, rl::DType::I32, {static_cast<i64>(e.flat.rows)}, {}, false, false};
            return ArrayOut(export_array(std::move(s), e.fw, e.enc));
        }, "[N] int32 successors per state (a fresh array)")
        .def_prop_ro("total", [](const PyExpansion& e) { return e.flat.total; })
        .def_prop_ro("capacity", [](const PyExpansion& e) { return e.flat.capacity; })
        .def_prop_ro("num_states", [](const PyExpansion& e) { return e.flat.rows; })
        .def_prop_ro("words", [](const PyExpansion& e) { return e.flat.words; }, "Atom words per row (W).")
        .def_prop_ro("numeric_words", [](const PyExpansion& e) { return e.flat.numeric_words; },
                     "Numeric words per row after the atom words (0 for classical tables).")
        .def_prop_ro("words_needed", [](const PyExpansion& e) { return e.flat.words_needed; })
        .def_prop_ro("label_width", [](const PyExpansion& e) { return e.flat.label_width; })
        .def_prop_ro("num_objects", [](const PyExpansion& e) { return e.ref.suite->max_objects(); },
                     "The most objects of an instance of the table or suite (the columns of rl.prefix_masks).")
        .def_prop_ro("num_schemas", [](const PyExpansion& e) { return e.ref.suite->max_schemas(); },
                     "Action schemas of the table, the most of any domain of a suite (the columns of rl.schema_masks).")
        .def_prop_ro("table", [](const PyExpansion& e) { return SuiteArg(e.ref.obj); },
                     "The suite, table or Task the expansion was made over.")
        .def_prop_ro("overflow", [](const PyExpansion& e) {
            return e.flat.total > e.flat.capacity || e.flat.words_needed > e.flat.words;
        })
        .def_prop_ro("framework", [](const PyExpansion& e) { return framework_name(e.fw); })
        .def_prop_ro("padded", [](const PyExpansion& e) -> Arg<std::optional<PaddedResult>> {
            if (!e.padded)
                return nb::none();
            return nb::cast(*e.padded);
        }, "The padded view, if expand() was called with K")
        .def("pad",
             [](const PyExpansion& e, u32 K) { return make_padded(e.flat, K == 0 ? auto_K(e.flat) : K, e.fw, e.enc); },
             "K"_a = 0, "The padded [N, K] view (K = 0: the next power of two at or above the largest count).")
        .def("action",
             [](const PyExpansion& e, u64 j) {
                 if (j >= e.flat.valid())
                     throw nb::index_error("mymyr: successor row out of range");
                 const i32* b = e.flat.at<i32>(e.flat.off_binding) + j * e.flat.label_width;
                 const u32 s = static_cast<u32>(e.flat.at<i32>(e.flat.off_schema)[j]);
                 const u32 inst = e.instance_of_state(static_cast<u64>(e.flat.at<i32>(e.flat.off_parent)[j]));
                 const Owner o = e.ref.owner(inst);
                 return make_label(o, s, b, o.core->data->schemas[s].arity());
             },
             "row"_a, "The Action (label) of flat row j (of its parent's instance).")
        .def("actions",
             [](nb::pointer_and_handle<PyExpansion> self, u64 i) {
                 const PyExpansion& e = *self.p;
                 if (i >= e.flat.rows)
                     throw nb::index_error("mymyr: state index out of range");
                 const i32* off = e.flat.at<i32>(e.flat.off_offsets);
                 nb::typed<nb::list, PyAction> out{nb::list()};
                 for (i64 j = off[i]; j < off[i + 1] && static_cast<u64>(j) < e.flat.valid(); ++j)
                     out.append(self.h.attr("action")(j));
                 return out;
             },
             "state"_a, "The Actions of state i's successors, in order.")
        .def("__len__", [](const PyExpansion& e) { return e.flat.valid(); })
        .def("__repr__", &words_repr);

    m.def("expand", &expand, "table"_a, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(), "capacity"_a = nb::none(),
          "words"_a = nb::none(), "K"_a = nb::none(), "goal"_a = false, "canonical"_a = true, "witness"_a = false,
          "pool"_a = nb::none(), "threads"_a = 1, "framework"_a = nb::none(), "validate"_a = true,
          "stream"_a = nb::none(), "ctx"_a = nb::none(), "Expands a batch of states [N, W] (see mymyr.rl.expand).");
    m.def("expand_into", &expand_into, "table"_a, "states"_a, "task_ids"_a = nb::none(), nb::kw_only(),
          "succ"_a = nb::none(), "parent"_a = nb::none(), "schema"_a = nb::none(), "binding"_a = nb::none(),
          "goal"_a = nb::none(), "offsets"_a = nb::none(), "canonical"_a = true, "witness"_a = false,
          "pool"_a = nb::none(), "validate"_a = true, "stream"_a = nb::none(), "ctx"_a = nb::none(),
          "Destination-passing expand into caller arrays (see mymyr.rl.expand_into).");
    m.def(
        "is_goal",
        [](SuiteArg table, StatesLike states, TaskIdsArg task_ids, FrameworkArg framework) {
            const SuiteRef ref = suite_of(table);
            StateBatch in = import_table_states(states, *ref.suite);
            const TaskIds ids(task_ids, in.view.rows, *ref.suite);
            const auto [fw, enc] = output_kind(in, framework, !in.packed);
            auto block = Block::make(std::max<u64>(in.view.rows, 1));
            {
                nb::gil_scoped_release release;
                rl::is_goal(*ref.suite, in.view, ids.ptr, reinterpret_cast<u8*>(block->data()));
            }
            return ArrayOut(bool_array(block, in.view.rows, fw));
        },
        "table"_a, "states"_a, "task_ids"_a = nb::none(), "framework"_a = nb::none(),
        "Goal flags [N]: row i under the goal test of instance task_ids[i] (axioms evaluated where the goal needs them).");
    auto goal_fn = [](bool count) {
        return [count](StatesLike states, ArrayArg gpos, ArrayArg gneg, FrameworkArg framework) -> ArrayOut {
            StateBatch in = import_states(states, 0);
            StateBatch p = import_states(gpos, 0);
            StateBatch n = import_states(gneg, 0);
            const auto [fw, enc] = output_kind(in, framework, !in.packed);
            auto block = Block::make(std::max<u64>(in.view.rows, 1) * sizeof(i32));
            {
                nb::gil_scoped_release release;
                if (count)
                    rl::goal_count(in.view, p.view, n.view, reinterpret_cast<i32*>(block->data()));
                else
                    rl::goal_test(in.view, p.view, n.view, reinterpret_cast<u8*>(block->data()));
            }
            if (!count)
                return bool_array(block, in.view.rows, fw);
            ArraySpec s{block, block->data(), rl::DType::I32, {static_cast<i64>(in.view.rows)}, {}, false, false};
            return export_array(std::move(s), fw, enc);
        };
    };
    m.def("goal_test", goal_fn(false), "states"_a, "gpos"_a, "gneg"_a, "framework"_a = nb::none(),
          "Mask goal test [N] bool: (s & gpos) == gpos and (s & gneg) == 0; masks are one row or one row per state.");
    m.def("goal_count", goal_fn(true), "states"_a, "gpos"_a, "gneg"_a, "framework"_a = nb::none(),
          "Unsatisfied goal literals per state [N] int32.");
    m.def(
        "random_walks",
        [](TaskArg task, u64 steps, u64 episode, u64 seed, bool canonical, bool witness) {
            Owner o = owner_of(task);
            rl::WalkStats st;
            {
                nb::gil_scoped_release release;
                st = rl::random_walks(*o.core->task, steps, episode, seed, expand_options(canonical, witness, true));
            }
            nb::typed<nb::dict, std::string, int> d{nb::dict()};
            d["steps"] = st.steps;
            d["successors"] = st.successors;
            d["dead_ends"] = st.dead_ends;
            d["goals"] = st.goals;
            return d;
        },
        "task"_a, "steps"_a, "episode"_a = 50, "seed"_a = 0, nb::kw_only(), "canonical"_a = true, "witness"_a = false,
        "Native random walks (one call): expand, move to a uniformly random successor, restart after `episode` steps "
        "or at a dead end.");

    // --- CpuEnvPool
    using OptArray = Arg<std::optional<ann::Any>>;
    auto opt = [](bool present, auto&& make) -> OptArray {
        if (!present)
            return nb::none();
        return make();
    };
    nb::class_<PyPoolBatch>(m, "PoolBatch",
                            "The results of a CpuEnvPool call: arrays of B rows in the order of their env ids "
                            "(zero-copy, read-only views in the pool's framework). The step results are None after a "
                            "reset. Once the batch and every array taken from it are dropped, its memory goes back to "
                            "the pool for later results (CpuEnvPool::recycle).")
        .def_prop_ro("rows", [](const PyPoolBatch& p) { return p.b->rows; })
        .def_prop_ro("stepped", &PyPoolBatch::stepped, "Whether these are step results (False: a reset's).")
        .def_prop_ro("env_ids", [](const PyPoolBatch& p) {
            return ArrayOut(p.array(p.b->env_ids, rl::DType::I32, {p.rows()}));
        }, "[B] int32 env ids")
        .def_prop_ro("states", [](const PyPoolBatch& p) {
            return ArrayOut(p.array(p.b->states, rl::DType::U64, {p.rows(), static_cast<i64>(p.b->row_words())}, true));
        }, "[B, row_words] the states after the step (and the autoreset)")
        .def_prop_ro("task_ids", [](const PyPoolBatch& p) {
            return ArrayOut(p.array(p.b->task_ids, rl::DType::I32, {p.rows()}));
        }, "[B] int32 the envs' instances")
        .def_prop_ro("count", [](const PyPoolBatch& p) {
            return ArrayOut(p.array(p.b->count, rl::DType::I32, {p.rows()}));
        }, "[B] int32 successor counts of the states")
        .def_prop_ro("steps", [](const PyPoolBatch& p) {
            return ArrayOut(p.array(p.b->steps, rl::DType::I32, {p.rows()}));
        }, "[B] int32 the episodes' step counts")
        .def_prop_ro("goal_pos", [opt](const PyPoolBatch& p) {
            return opt(p.b->goals, [&] {
                return p.array(p.b->goal_pos, rl::DType::U64, {p.rows(), static_cast<i64>(p.b->words)}, true);
            });
        }, "[B, words] per-env goal masks (pools with goals=True), else None")
        .def_prop_ro("goal_neg", [opt](const PyPoolBatch& p) {
            return opt(p.b->goals, [&] {
                return p.array(p.b->goal_neg, rl::DType::U64, {p.rows(), static_cast<i64>(p.b->words)}, true);
            });
        }, "[B, words] per-env goal masks (pools with goals=True), else None")
        .def_prop_ro("reward", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->reward, rl::DType::F32, {p.rows()}); });
        }, "[B] float32 (None after a reset)")
        .def_prop_ro("terminated", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->terminated, rl::DType::Bool, {p.rows()}); });
        }, "[B] bool (None after a reset)")
        .def_prop_ro("truncated", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->truncated, rl::DType::Bool, {p.rows()}); });
        }, "[B] bool (None after a reset)")
        .def_prop_ro("invalid", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->invalid, rl::DType::Bool, {p.rows()}); });
        }, "[B] bool: the given action was outside [0, count) (None after a reset)")
        .def_prop_ro("goal", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->goal, rl::DType::Bool, {p.rows()}); });
        }, "[B] bool: the reached state is a goal state (None after a reset)")
        .def_prop_ro("final_states", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] {
                return p.array(p.b->final_states, rl::DType::U64, {p.rows(), static_cast<i64>(p.b->row_words())}, true);
            });
        }, "[B, row_words] the states the steps reached, before the autoreset (None after a reset)")
        .def_prop_ro("schema", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] { return p.array(p.b->schema, rl::DType::I32, {p.rows()}); });
        }, "[B] int32 the actions' schemas (-1: no move; None after a reset)")
        .def_prop_ro("binding", [opt](const PyPoolBatch& p) {
            return opt(p.stepped(), [&] {
                return p.array(p.b->binding, rl::DType::I32, {p.rows(), static_cast<i64>(p.b->label_width)});
            });
        }, "[B, label_width] int32 the actions' objects (-1 padding; None after a reset)")
        .def("__len__", [](const PyPoolBatch& p) { return p.b->rows; })
        .def("__repr__", [](const PyPoolBatch& p) {
            return "PoolBatch(rows=" + std::to_string(p.b->rows) + (p.stepped() ? ", step" : ", reset") + ")";
        });

    nb::class_<PyCpuPool>(
        m, "CpuEnvPool",
        "EnvPool-style asynchronous environments on the CPU (rl/pool.hpp): num_envs envs over a "
        "table or a suite, stepped by worker threads. send(env_ids, actions) enqueues a step of some envs and returns at once; "
        "recv() returns the oldest sends' results in send order. An env's step is the host env's (rl::HostEnv): the "
        "same semantics and RNG streams (env id = the env's index), so results do not depend on the thread count or "
        "the order of sends. Every method may be called from any number of Python threads at once (the calls release "
        "the thread state while they wait).")
        .def(
            "__init__",
            [](PyCpuPool* self, SuiteArg table, u32 num_envs, u32 threads, bool goals, u64 seed, u32 max_steps,
               f32 step_reward, f32 goal_reward, Arg<ann::DeadEnd> dead_end, f32 dead_end_reward, bool dead_end_terminal,
               bool autoreset, bool canonical, bool witness, FrameworkArg framework) {
                rl::EnvConfig c;
                c.seed = seed;
                c.max_steps = max_steps;
                c.step_reward = step_reward;
                c.goal_reward = goal_reward;
                c.dead_end = parse_dead_end(nb::cast<std::string>(dead_end));
                c.dead_end_reward = dead_end_reward;
                c.dead_end_terminal = dead_end_terminal;
                c.autoreset = autoreset;
                c.canonical_order = canonical;
                c.witness_pruning = witness;
                rl::PoolOptions o;
                o.threads = threads;
                o.goals = goals;
                const Framework fw = framework.is_none() ? Framework::Numpy : parse_framework(framework, nb::none());
                new (self) PyCpuPool(suite_of(table), c, num_envs, o, fw);
            },
            "table"_a, "num_envs"_a, nb::kw_only(), "threads"_a = 0, "goals"_a = false, "seed"_a = 0, "max_steps"_a = 0,
            "step_reward"_a = -1.0f, "goal_reward"_a = 0.0f, "dead_end"_a = "no_successors", "dead_end_reward"_a = 0.0f,
            "dead_end_terminal"_a = true, "autoreset"_a = true, "canonical"_a = true, "witness"_a = false,
            "framework"_a = nb::none(),
            "threads 0: the hardware's; goals: per-env goal masks (the step's goal test is then the mask test); the "
            "results come as `framework` arrays (None: NumPy). Every env starts at instance 0's initial state: "
            "reset(task_ids=...) places them.")
        .def_prop_ro("table", [](const PyCpuPool& p) { return SuiteArg(p.ref().obj); })
        .def_prop_ro("num_envs", [](const PyCpuPool& p) { return p.pool().num_envs(); })
        .def_prop_ro("threads", [](const PyCpuPool& p) { return p.pool().threads(); })
        .def_prop_ro("goals", [](const PyCpuPool& p) { return p.pool().goals(); })
        .def_prop_ro("words", [](const PyCpuPool& p) { return p.pool().words(); })
        .def_prop_ro("numeric_words", [](const PyCpuPool& p) { return p.pool().numeric_words(); })
        .def_prop_ro("row_words", [](const PyCpuPool& p) { return p.pool().words() + p.pool().numeric_words(); })
        .def_prop_ro("label_width", [](const PyCpuPool& p) { return p.pool().label_width(); })
        .def_prop_ro("framework", [](const PyCpuPool& p) { return framework_name(p.framework()); })
        .def_prop_ro("seed", [](const PyCpuPool& p) { return p.pool().config().seed; })
        .def_prop_ro("max_steps", [](const PyCpuPool& p) { return p.pool().config().max_steps; })
        .def_prop_ro("dead_end", [](const PyCpuPool& p) { return std::string(dead_end_name(p.pool().config().dead_end)); })
        .def_prop_ro("pending", [](const PyCpuPool& p) { return p.pool().pending(); }, "Sends not yet received.")
        .def(
            "reset",
            [](PyCpuPool& p, TaskIdsArg env_ids, TaskIdsArg task_ids, ArrayArg goal_pos, ArrayArg goal_neg) {
                return p.reset(env_ids, task_ids, goal_pos, goal_neg);
            },
            "env_ids"_a = nb::none(), "task_ids"_a = nb::none(), nb::kw_only(), "goal_pos"_a = nb::none(),
            "goal_neg"_a = nb::none(),
            "Resets envs env_ids [n] (None: every env) into instances task_ids [n] (None: their current ones), with "
            "goal masks goal_pos / goal_neg [n, words] (None: their instances' goals; goals=True pools only); returns "
            "their observations. Draw counters are kept. Raises if one of them has a step in flight.")
        .def(
            "send",
            [](PyCpuPool& p, TaskIdsArg env_ids, TaskIdsArg actions, TaskIdsArg next_task_ids) {
                return p.send(env_ids, actions, next_task_ids);
            },
            "env_ids"_a = nb::none(), "actions"_a = nb::none(), nb::kw_only(), "next_task_ids"_a = nb::none(),
            "Enqueues one step of envs env_ids [n] (None: every env) and returns its ticket at once: actions [n] "
            "(int64 or int32 indices into the canonical successor order; None: the random policy), next_task_ids [n] "
            "(the instances autoresetting envs restart in; None: their own). Raises for an unknown or repeated env id "
            "or one with a step in flight.")
        .def("recv", &PyCpuPool::recv, "min_rows"_a = 0,
             "The results of the oldest pending sends, in send order: whole sends with at least min_rows rows together "
             "(0: exactly one send). Blocks until they are done; raises if fewer rows are pending.")
        .def("recv_ticket", &PyCpuPool::recv_ticket, "ticket"_a, "The results of the send with this ticket (blocks).")
        .def(
            "step",
            [](PyCpuPool& p, TaskIdsArg env_ids, TaskIdsArg actions, TaskIdsArg next_task_ids) {
                return p.step(env_ids, actions, next_task_ids);
            },
            "env_ids"_a = nb::none(), "actions"_a = nb::none(), nb::kw_only(), "next_task_ids"_a = nb::none(),
            "send + recv_ticket: one step of envs env_ids (None: every env), its results.")
        .def("__repr__", [](const PyCpuPool& p) {
            return "CpuEnvPool(num_envs=" + std::to_string(p.pool().num_envs()) + ", threads=" +
                   std::to_string(p.pool().threads()) + ")";
        });
}
}  // namespace mymyr::python
