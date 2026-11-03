// Evaluators written in Python (py_evaluator.hpp): the StateBatch they receive, the values they return, the
// coalescing of a SharedEvaluator, and the search-side adapters.

#include "py_evaluator.hpp"

#include "arrays.hpp"
#include "py_task.hpp"
#include "typing.hpp"

#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace mymyr::python
{
using namespace nb::literals;

namespace
{
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

/// mymyr.search.StateBatch: the states of one evaluate_batch call as packed rows. Owns its arrays.
struct PyStateBatch
{
    nb::object rows, task_ids, encoded;
    std::shared_ptr<const Evaluator> evaluator;  // with an encoder: its table names the tasks
    std::vector<Owner> owners;                   // without an encoder: the tasks, in order of first appearance
    const u64* data = nullptr;
    const i32* ids = nullptr;
    u64 n = 0;
    u32 words = 0, numeric_words = 0;
    nb::object states_cache, tasks_cache;

    [[nodiscard]] Owner owner(u32 task) const
    {
        return evaluator && evaluator->has_encoder() ? evaluator->table.owner(task) : owners.at(task);
    }
};

[[noreturn]] void raise(PyObject* type, const std::string& text)
{
    PyErr_SetString(type, text.c_str());
    throw nb::python_error();
}

std::string repr_of(f64 v) { return nb::cast<std::string>(nb::repr(nb::float_(v))); }

/// `v` as a value of `kind`, or a ValueError.
f64 checked(f64 v, ValueKind kind)
{
    if (std::isnan(v))
        raise(PyExc_ValueError, kind == ValueKind::Heuristic
                                    ? "mymyr: a Python heuristic returned nan; heuristic values are >= 0 (inf: a dead end)"
                                    : "mymyr: a layer score is nan; scores are numbers (lower first)");
    if (kind == ValueKind::Heuristic && v < 0)
        raise(PyExc_ValueError,
              "mymyr: a Python heuristic returned " + repr_of(v) + "; heuristic values are >= 0 (inf: a dead end)");
    return v;
}

/// The float of a Python number (anything with __float__); TypeError otherwise.
f64 number_of(PyObject* x)
{
    const double v = PyFloat_AsDouble(x);
    if (v == -1.0 && PyErr_Occurred())
    {
        if (!PyErr_ExceptionMatches(PyExc_TypeError))
            throw nb::python_error();
        PyErr_Clear();
        raise(PyExc_TypeError, "mymyr: a Python evaluator returned a " +
                                   nb::cast<std::string>(nb::str(nb::handle(x).type().attr("__name__"))) +
                                   "; values are numbers (heuristic values >= 0, inf: a dead end)");
    }
    return v;
}

/// The values of an evaluate_batch result: a CPU array (buffer protocol or DLPack: NumPy, a CPU torch tensor, ...) of
/// shape [n] or [n, 1] with a real or integer dtype, or a sequence of n numbers.
void read_values(nb::handle result, std::span<f64> out)
{
    using nb::dlpack::dtype_code;
    nb::ndarray<nb::ro, nb::device::cpu> a;
    if (!nb::isinstance<nb::list>(result) && !nb::isinstance<nb::tuple>(result) && nb::try_cast(result, a, false))
    {
        const bool shape_ok = (a.ndim() == 1 || (a.ndim() == 2 && a.shape(1) == 1));
        if (!shape_ok || a.shape(0) != out.size())
        {
            std::string shape = "(";
            for (usize d = 0; d < a.ndim(); ++d)
                shape += (d ? ", " : "") + std::to_string(a.shape(d));
            shape += a.ndim() == 1 ? ",)" : ")";
            raise(PyExc_ValueError, "mymyr: evaluate_batch returned an array of shape " + shape + " for " +
                                        std::to_string(out.size()) + " states (expected one value per state)");
        }
        const auto dt = a.dtype();
        const i64 stride = a.stride(0);
        const auto* base = static_cast<const std::byte*>(a.data());
        auto read = [&]<class T>(T*)
        {
            for (usize i = 0; i < out.size(); ++i)
            {
                T x;
                std::memcpy(&x, base + static_cast<i64>(i) * stride * static_cast<i64>(sizeof(T)), sizeof(T));
                out[i] = static_cast<f64>(x);
            }
        };
        if (dt == nb::dtype<f64>())
            read(static_cast<f64*>(nullptr));
        else if (dt == nb::dtype<float>())
            read(static_cast<float*>(nullptr));
        else if (dt == nb::dtype<i64>())
            read(static_cast<i64*>(nullptr));
        else if (dt == nb::dtype<i32>())
            read(static_cast<i32*>(nullptr));
        else if (dt == nb::dtype<u64>())
            read(static_cast<u64*>(nullptr));
        else if (dt == nb::dtype<u32>())
            read(static_cast<u32*>(nullptr));
        else
            raise(PyExc_TypeError, "mymyr: evaluate_batch returned an array of an unsupported dtype (use a float or "
                                   "integer dtype)");
        return;
    }
    nb::object seq = nb::steal(PySequence_Fast(result.ptr(), "mymyr: evaluate_batch must return an array or a sequence "
                                                             "of numbers, one per state"));
    if (!seq.is_valid())
        throw nb::python_error();
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.ptr());
    if (static_cast<usize>(n) != out.size())
        raise(PyExc_ValueError, "mymyr: evaluate_batch returned " + std::to_string(n) + " values for " +
                                    std::to_string(out.size()) + " states");
    PyObject** items = PySequence_Fast_ITEMS(seq.ptr());
    for (Py_ssize_t i = 0; i < n; ++i)
        out[static_cast<usize>(i)] = number_of(items[i]);  // checked by kind by the caller
}

/// A read-only NumPy array over a new block of `count` elements of T, and the block's memory.
template<class T>
std::pair<nb::object, T*> new_array(std::vector<i64> shape, rl::DType dtype, bool words)
{
    u64 count = 1;
    for (const i64 s : shape)
        count *= static_cast<u64>(s);
    std::shared_ptr<Block> block = Block::make(std::max<u64>(count, 1) * sizeof(T));
    T* data = reinterpret_cast<T*>(block->data());
    ArraySpec spec;
    spec.owner = block;
    spec.data = data;
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.readonly = true;
    spec.words = words;
    return {export_array(std::move(spec), Framework::Numpy, default_words(Framework::Numpy)), data};
}
}  // namespace

// ------------------------------------------------------------------------------------------------ values

f64 value_of(PyObject* x, ValueKind kind) { return checked(number_of(x), kind); }

// ------------------------------------------------------------------------------------------------ Evaluator

Evaluator::Evaluator(nb::handle o) : obj(nb::borrow(o))
{
    if (nb::isinstance<PySharedEvaluator>(o))
        raise(PyExc_TypeError, "mymyr: a SharedEvaluator wraps an evaluator, not another SharedEvaluator");
    const bool has_one = nb::hasattr(o, "evaluate"), has_batch = nb::hasattr(o, "evaluate_batch");
    if (has_one || has_batch)
    {
        if (has_one)
            one = nb::getattr(o, "evaluate");
        if (has_batch)
            batch = nb::getattr(o, "evaluate_batch");
        if (nb::hasattr(o, "preferred_actions"))
            preferred = nb::getattr(o, "preferred_actions");
    }
    else if (PyCallable_Check(o.ptr()) && !nb::isinstance<nb::str>(o))
        one = nb::borrow(o);
    else
        raise(PyExc_TypeError, "mymyr: an evaluator is a callable state -> float or an object with evaluate(state) "
                               "or evaluate_batch(batch) (mymyr._typing.Evaluator)");
    if (nb::hasattr(o, "encoder"))
    {
        nb::object enc = nb::getattr(o, "encoder");
        if (!enc.is_none())
        {
            if (!batch.is_valid())
                raise(PyExc_TypeError, "mymyr: an evaluator with an encoder needs evaluate_batch(batch): the encoding "
                                       "is batch.encoded");
            if (!nb::hasattr(enc, "encode") || !nb::hasattr(enc, "table"))
                raise(PyExc_TypeError, "mymyr: an evaluator's encoder is a mymyr.encoding encoder (with encode(states, "
                                       "task_ids, ...) and table)");
            table = table_of(enc.attr("table"));
            encoder = std::move(enc);
        }
    }
    if (nb::hasattr(o, "encode_options"))
    {
        nb::object opts = nb::getattr(o, "encode_options");
        if (!opts.is_none())
        {
            if (!nb::isinstance<nb::dict>(opts))
                raise(PyExc_TypeError, "mymyr: an evaluator's encode_options is a dict of keyword arguments of "
                                       "encoder.encode, or None");
            encode_options = std::move(opts);
        }
    }
}

Evaluator::~Evaluator()
{
    nb::gil_scoped_acquire acquire;
    obj.reset();
    one.reset();
    batch.reset();
    preferred.reset();
    encoder.reset();
    encode_options.reset();
    table = {};
}

i32 Evaluator::instance_of(const Owner& o) const
{
    if (!has_encoder())
        return -1;
    const rl::TaskTable& t = *table.table;
    const Task& task = *o.core->task;
    for (u32 i = 0; i < t.size(); ++i)
        if (t.task(i)->uid() == task.uid())
            return static_cast<i32>(i);
    for (u32 i = 0; i < t.size(); ++i)
        if (task.atoms().frozen() && t.task(i)->fingerprint() == task.fingerprint() &&
            t.task(i)->encoding_fingerprint() == task.encoding_fingerprint())
            return static_cast<i32>(i);
    raise(PyExc_ValueError, "mymyr: the search's task is not an instance of the table of the evaluator's encoder (the "
                            "same task, or a task of the same fingerprint and atom encoding)");
}

void evaluate_parts(const std::shared_ptr<const Evaluator>& e, std::span<BatchPart> parts)
{
    u64 n = 0;
    for (const BatchPart& p : parts)
        n += p.states.size();
    if (n == 0)
        return;
    std::vector<Owner> owners;
    std::vector<i32> part_task(parts.size());
    u32 W = 1, NN = 0;
    if (e->has_encoder())
    {
        W = e->table.table->words();
        NN = e->table.table->numeric_words();
        for (usize p = 0; p < parts.size(); ++p)
            part_task[p] = parts[p].instance;
    }
    else
        for (usize p = 0; p < parts.size(); ++p)
        {
            const Owner& o = *parts[p].owner;
            usize t = 0;
            while (t < owners.size() && owners[t].core != o.core)
                ++t;
            if (t == owners.size())
                owners.push_back(o);
            part_task[p] = static_cast<i32>(t);
            W = std::max(W, o.core->task->words());
            NN = std::max(NN, o.core->task->numeric_words());
        }
    for (const BatchPart& p : parts)
        for (const StateView& s : p.states)
            W = std::max(W, s.nw);
    const u32 stride = W + NN;
    auto [rows, data] = new_array<u64>({static_cast<i64>(n), static_cast<i64>(stride)}, rl::DType::U64, true);
    auto [ids, id_data] = new_array<i32>({static_cast<i64>(n)}, rl::DType::I32, false);
    {
        u64 r = 0;
        for (usize p = 0; p < parts.size(); ++p)
            for (const StateView& s : parts[p].states)
            {
                u64* row = data + r * stride;
                std::memcpy(row, s.w, s.nw * sizeof(u64));
                std::memset(row + s.nw, 0, (stride - s.nw) * sizeof(u64));
                if (s.nnum)
                    std::memcpy(row + W, s.num, s.nnum * sizeof(u64));
                id_data[r++] = part_task[p];
            }
    }
    nb::object encoded = nb::none();
    if (e->has_encoder())
    {
        nb::object encode = e->encoder.attr("encode");
        encoded = e->encode_options.is_valid() ? encode(rows, ids, **nb::borrow<nb::dict>(e->encode_options))
                                               : encode(rows, ids);
    }
    PyStateBatch b;
    b.rows = rows;
    b.task_ids = ids;
    b.encoded = std::move(encoded);
    b.evaluator = e;
    b.owners = std::move(owners);
    b.data = data;
    b.ids = id_data;
    b.n = n;
    b.words = W;
    b.numeric_words = NN;
    nb::object batch = nb::cast(std::move(b), nb::rv_policy::move);
    nb::object result = e->batch(batch);
    std::vector<f64> values(n);
    read_values(result, values);
    u64 r = 0;
    for (BatchPart& p : parts)
        for (f64& v : p.out)
            v = checked(values[r++], p.kind);
}

// ------------------------------------------------------------------------------------------------ SharedEvaluator

PySharedEvaluator::PySharedEvaluator(nb::handle obj, u64 max_batch_, double max_wait_)
    : evaluator(std::make_shared<const Evaluator>(obj)), max_batch(max_batch_)
{
    if (!evaluator->batch.is_valid())
        raise(PyExc_TypeError, "mymyr: a SharedEvaluator calls evaluate_batch(batch); the evaluator has none");
    if (max_batch == 0)
        raise(PyExc_ValueError, "mymyr: SharedEvaluator max_batch must be positive");
    if (!(max_wait_ >= 0) || !std::isfinite(max_wait_))
        raise(PyExc_ValueError, "mymyr: SharedEvaluator max_wait must be a finite number of seconds >= 0");
    max_wait = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(max_wait_));
}

void PySharedEvaluator::enter()
{
    std::lock_guard lock(m_mutex);
    ++m_active;
}

void PySharedEvaluator::leave()
{
    {
        std::lock_guard lock(m_mutex);
        --m_active;
    }
    m_cv.notify_all();  // the other searches may all be waiting now
}

bool PySharedEvaluator::due(Clock::time_point now) const
{
    return !m_queue.empty() &&
           (m_queued_rows >= max_batch || m_waiting >= m_active || now >= m_queue.front()->since + max_wait);
}

PyObject* PySharedEvaluator::evaluate(const BatchPart& part)
{
    std::unique_lock lock(m_mutex);
    Request r{&part, Clock::now()};
    m_queue.push_back(&r);
    m_queued_rows += part.states.size();
    ++m_waiting;
    m_cv.notify_all();
    while (!r.done)
    {
        if (!m_calling && due(Clock::now()))
            lead(lock);
        else if (m_calling || m_queue.empty())
            m_cv.wait(lock);
        else
            m_cv.wait_until(lock, m_queue.front()->since + max_wait);
    }
    return r.error;
}

void PySharedEvaluator::lead(std::unique_lock<std::mutex>& lock)
{
    m_calling = true;
    std::vector<Request*> taken;
    u64 rows = 0;
    while (!m_queue.empty())
    {
        Request* q = m_queue.front();
        const u64 k = q->part->states.size();
        if (!taken.empty() && rows + k > max_batch)
            break;  // requests are never split: one larger than max_batch is called alone
        taken.push_back(q);
        rows += k;
        m_queue.pop_front();
    }
    m_queued_rows -= rows;
    lock.unlock();
    PyObject* error = nullptr;
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            std::vector<BatchPart> parts;
            parts.reserve(taken.size());
            for (const Request* q : taken)
                parts.push_back(*q->part);
            evaluate_parts(evaluator, parts);
        }
        catch (nb::python_error& e)
        {
            error = e.value().inc_ref().ptr();
        }
        catch (const std::exception& e)
        {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            error = PyErr_GetRaisedException();
        }
        for (usize i = 1; error && i < taken.size(); ++i)
            Py_INCREF(error);  // one reference per request
    }
    m_calls.fetch_add(1, std::memory_order_relaxed);
    m_rows.fetch_add(rows, std::memory_order_relaxed);
    u64 largest = m_largest.load(std::memory_order_relaxed);
    while (rows > largest && !m_largest.compare_exchange_weak(largest, rows, std::memory_order_relaxed))
    {
    }
    lock.lock();
    for (Request* q : taken)
    {
        q->done = true;
        q->error = error;
    }
    m_waiting -= static_cast<u32>(taken.size());
    m_calling = false;
    m_cv.notify_all();
}

// ------------------------------------------------------------------------------------------------ search adapters

PyEvaluatorHeuristic::PyEvaluatorHeuristic(nb::handle obj, Owner owner, CallbackErrors& errors, ValueKind kind)
    : m_owner(std::move(owner)), m_errors(errors), m_kind(kind)
{
    if (nb::isinstance<PySharedEvaluator>(obj))
    {
        PySharedEvaluator* shared = nb::inst_ptr<PySharedEvaluator>(obj);
        m_eval = shared->evaluator;
        m_instance = m_eval->instance_of(m_owner);
        m_shared_obj = nb::borrow(obj);
        m_shared = shared;
        m_batched = true;
        m_shared->enter();
        return;
    }
    m_eval = std::make_shared<const Evaluator>(obj);
    m_instance = m_eval->instance_of(m_owner);
    m_batched = m_eval->batch.is_valid();
}

PyEvaluatorHeuristic::~PyEvaluatorHeuristic()
{
    if (m_shared)
        m_shared->leave();
    nb::gil_scoped_acquire acquire;
    m_eval.reset();
    m_shared_obj.reset();
    m_owner.obj.reset();
}

void PyEvaluatorHeuristic::count(heuristics::Value v)
{
    ++m_stats.evaluations;
    if (std::isinf(v) && v > 0)
        ++m_stats.dead_ends;
}

heuristics::Value PyEvaluatorHeuristic::evaluate(StateView s)
{
    if (m_shared || !m_eval->one.is_valid())
    {
        heuristics::Value v = 0;
        evaluate_batch(std::span<const StateView>(&s, 1), std::span<heuristics::Value>(&v, 1));
        return v;
    }
    if (m_errors.failed.load(std::memory_order_relaxed))
        return m_kind == ValueKind::Score ? 0 : k_inf;
    nb::gil_scoped_acquire acquire;
    try
    {
        const heuristics::Value v = value_of(m_eval->one(make_state(m_owner, State(s))).ptr(), m_kind);
        if (m_eval->preferred.is_valid())
            load_preferred(m_eval->preferred());
        count(v);
        return v;
    }
    catch (nb::python_error& e)
    {
        m_errors.record(std::move(e));
        m_pref.clear();
        return m_kind == ValueKind::Score ? 0 : k_inf;
    }
}

void PyEvaluatorHeuristic::evaluate_batch(std::span<const StateView> states, std::span<heuristics::Value> out)
{
    const f64 failed_value = m_kind == ValueKind::Score ? 0 : k_inf;
    if (!m_batched)
    {
        for (usize i = 0; i < states.size(); ++i)
            out[i] = evaluate(states[i]);
        return;
    }
    if (m_errors.failed.load(std::memory_order_relaxed))
    {
        std::ranges::fill(out, failed_value);
        return;
    }
    if (states.empty())
        return;
    BatchPart part{states, &m_owner, m_instance, out, m_kind};
    if (m_shared)
    {
        PyObject* error = nullptr;
        if (PyGILState_Check())
        {
            nb::gil_scoped_release release;  // the coalesced call may run on another thread
            error = m_shared->evaluate(part);
        }
        else
            error = m_shared->evaluate(part);
        if (error)
        {
            nb::gil_scoped_acquire acquire;
            PyErr_SetRaisedException(error);
            m_errors.record(nb::python_error());
            std::ranges::fill(out, failed_value);
            return;
        }
        for (const heuristics::Value v : out)
            count(v);
        return;
    }
    nb::gil_scoped_acquire acquire;
    try
    {
        evaluate_parts(m_eval, std::span<BatchPart>(&part, 1));
        for (const heuristics::Value v : out)
            count(v);
    }
    catch (nb::python_error& e)
    {
        m_errors.record(std::move(e));
        std::ranges::fill(out, failed_value);
    }
}

bool PyEvaluatorHeuristic::preferred(const ActionLabel& a) const
{
    const auto less = [](const Action& x, const ActionLabel& y) {
        if (x.schema != y.schema)
            return x.schema < y.schema;
        return std::lexicographical_compare(x.binding.begin(), x.binding.end(), y.binding.begin(), y.binding.end());
    };
    const auto it = std::lower_bound(m_pref.begin(), m_pref.end(), a, less);
    return it != m_pref.end() && it->schema == a.schema && std::ranges::equal(it->binding, a.binding);
}

void PyEvaluatorHeuristic::load_preferred(nb::handle actions)
{
    m_pref.clear();
    for (nb::handle a : actions)
    {
        if (!nb::isinstance<PyAction>(a))
            raise(PyExc_TypeError, "mymyr: preferred_actions must return mymyr Actions");
        const PyAction& p = *nb::inst_ptr<PyAction>(a);
        if (p.core != m_owner.core)
            raise(PyExc_ValueError, "mymyr: preferred_actions returned an action of another task");
        std::vector<ObjectId> binding(p.binding.size());
        for (usize i = 0; i < binding.size(); ++i)
            binding[i] = ObjectId{p.binding[i]};
        m_pref.emplace_back(SchemaId{p.schema}, std::move(binding));
    }
    std::ranges::sort(m_pref);
}

std::function<void(std::span<const StateView>, std::span<f64>)> layer_scorer(nb::handle obj, const Owner& owner,
                                                                             CallbackErrors& errors)
{
    auto h = std::make_shared<PyEvaluatorHeuristic>(obj, owner, errors, ValueKind::Score);
    return [h](std::span<const StateView> states, std::span<f64> out) { h->evaluate_batch(states, out); };
}

bool is_evaluator(nb::handle obj)
{
    if (nb::isinstance<nb::str>(obj))
        return false;
    return nb::isinstance<PySharedEvaluator>(obj) || PyCallable_Check(obj.ptr()) || nb::hasattr(obj, "evaluate") ||
           nb::hasattr(obj, "evaluate_batch");
}

// ------------------------------------------------------------------------------------------------ bindings

void bind_evaluators(nb::module_& m)
{
    nb::class_<PyStateBatch>(
        m, "StateBatch",
        "The states of one evaluate_batch call of a Python evaluator (mymyr._typing.Evaluator), as arrays: `rows` "
        "holds one packed state row per state, `task_ids` the task of each row, and `encoded` their encoding when the "
        "evaluator declares an `encoder`. The batch owns its arrays: they stay valid after the call.")
        .def("__len__", [](const PyStateBatch& b) { return b.n; })
        .def_prop_ro(
            "rows", [](const PyStateBatch& b) { return b.rows; },
            "The state rows: a read-only NumPy uint64 array [N, words + numeric_words], the atom words (fluent slot i "
            "is bit i & 63 of word i >> 6; words past a state's own are zero) followed by the numeric words of a numeric "
            "task: the layout of state rows elsewhere in mymyr (encoders, datasets, rl).")
        .def_prop_ro(
            "task_ids", [](const PyStateBatch& b) { return b.task_ids; },
            "The task of each row: a read-only NumPy int32 array [N], an index into `tasks`. With an encoder it is "
            "the instance of the row in the encoder's table (the task_ids encoder.encode took).")
        .def_prop_ro(
            "tasks",
            [](PyStateBatch& b) -> nb::object
            {
                if (!b.tasks_cache.is_valid())
                {
                    const u32 count = b.evaluator && b.evaluator->has_encoder() ? b.evaluator->table.table->size()
                                                                                : static_cast<u32>(b.owners.size());
                    nb::list tasks;
                    for (u32 t = 0; t < count; ++t)
                        tasks.append(task_object(b.owner(t).obj));
                    b.tasks_cache = nb::tuple(tasks);
                }
                return b.tasks_cache;
            },
            "The tasks the rows index (`tasks[task_ids[i]]` is the Task of row i): the instances of the encoder's "
            "table with an encoder, else the tasks of the batch in order of their first row.")
        .def_prop_ro(
            "encoded", [](const PyStateBatch& b) { return b.encoded; },
            "encoder.encode(rows, task_ids, **encode_options) when the evaluator declares an encoder (a FlatBatch for "
            "a FlatRelationEncoder; its arrays export to NumPy, torch or JAX without a copy), else None.")
        .def_prop_ro("words", [](const PyStateBatch& b) { return b.words; }, "Atom words per row.")
        .def_prop_ro("numeric_words", [](const PyStateBatch& b) { return b.numeric_words; },
                     "Numeric words per row, after the atom words (0 for classical tasks).")
        .def_prop_ro(
            "states",
            [](PyStateBatch& b) -> nb::object
            {
                if (!b.states_cache.is_valid())
                {
                    nb::list states;
                    for (u64 i = 0; i < b.n; ++i)
                    {
                        const Owner o = b.owner(static_cast<u32>(b.ids[i]));
                        const u64* row = b.data + i * (b.words + b.numeric_words);
                        const u32 nn = o.core->task->numeric_words();
                        states.append(make_state(o, State(row, b.words, nn ? row + b.words : nullptr, nn)));
                    }
                    b.states_cache = std::move(states);
                }
                return b.states_cache;
            },
            "The rows as a list of States (made on first use: the arrays are the fast path).");

    nb::class_<PySharedEvaluator>(
        m, "SharedEvaluator",
        "An evaluator shared by many searches: pass it as heuristic= (or layer_order=) to searches running on any "
        "number of threads. Each search hands over its batches and waits; one waiting search at a time calls "
        "evaluator.evaluate_batch once on the requests of all of them, in arrival order and never split. A call is made "
        "when the requests hold max_batch states, when every search using the evaluator is waiting, or when the oldest "
        "request has waited max_wait seconds. Single evaluations (the start state, the lazy searches) are requests of "
        "one state. The evaluator's other methods are not used; evaluate_batch is never called by two threads at "
        "once. A search's values, and so its result, are those of the search run alone whenever the evaluator's "
        "value of a row does not depend on the other rows of the call; only the coalescing (and so the number of "
        "calls and the batches' composition) depends on the timing. An exception raised by a call fails every "
        "search with a request in it (each re-raises it).")
        .def(nb::init<nb::handle, u64, double>(), "evaluator"_a, nb::kw_only(), "max_batch"_a = 1024,
             "max_wait"_a = 0.001,
             nb::sig("def __init__(self, evaluator: mymyr._typing.Evaluator, *, max_batch: int = 1024, "
                     "max_wait: float = 0.001) -> None"),
             "evaluator: an object with evaluate_batch(batch) (mymyr._typing.Evaluator), with an optional encoder. "
             "max_batch: the states a call takes at most (requests are whole: a larger one is called alone). "
             "max_wait: seconds a request waits for others before it is called.")
        .def_prop_ro("evaluator", [](const PySharedEvaluator& s) { return s.evaluator->obj; }, "The wrapped evaluator.")
        .def_prop_ro("max_batch", [](const PySharedEvaluator& s) { return s.max_batch; })
        .def_prop_ro("max_wait",
                     [](const PySharedEvaluator& s) { return std::chrono::duration<double>(s.max_wait).count(); })
        .def_prop_ro("calls", &PySharedEvaluator::calls, "evaluate_batch calls made so far.")
        .def_prop_ro("rows", &PySharedEvaluator::rows, "States evaluated so far, over all calls.")
        .def_prop_ro("largest_batch", &PySharedEvaluator::largest, "The most states of one call so far.");
}
}  // namespace mymyr::python
