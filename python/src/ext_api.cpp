// The provider side of mymyr/ext.h: one static function table exported as the capsule mymyr._core._C_API.
// minor 0: version, hashing, state views. minor 1: tasks, successors, apply, is_goal, atom lookup, state stores.

#include "mymyr/ext.h"

#include "py_task.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"
#include "mymyr/version.hpp"

#include <nanobind/nanobind.h>

#include <new>
#include <string>
#include <vector>

namespace nb = nanobind;

struct mymyr_task
{
    // never instantiated: a mymyr_task* is a mymyr::python::PyTaskCore*
};

struct mymyr_store
{
    mymyr::TaskPtr task;
    mymyr::FlatStateStore store;
};

namespace mymyr::python
{
namespace
{
static_assert(sizeof(ObjectId) == sizeof(uint32_t));

const char* api_version() { return mymyr::version().data(); }

uint64_t api_hash_words(const uint64_t* words, uint32_t n) { return hash::state_words(words, n); }

int api_equal_words(const uint64_t* a, uint32_t na, const uint64_t* b, uint32_t nb)
{
    return bits::equal(a, na, b, nb) ? 1 : 0;
}

int api_state_from_py(PyObject* obj, mymyr_state_view* out, Py_buffer* keep)
{
    if (is_state(nb::handle(obj)))
    {
        const PyState& s = state_of(nb::handle(obj));
        keep->obj = nullptr;  // borrowed from the State (valid while it is alive); PyBuffer_Release is a no-op
        keep->buf = nullptr;
        out->words = s.s.data();
        out->num_words = s.s.size_words();
        out->num_numeric = s.s.numeric_words();  // numeric tasks: the numeric words (task/numeric.hpp encoding)
        out->numeric = s.s.numeric_words() ? s.s.numeric().data() : nullptr;
        out->task_uid = s.core->task->uid();
        return 0;
    }
    if (PyObject_GetBuffer(obj, keep, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0)
        return -1;
    const bool u64 = keep->itemsize == 8 && keep->format && (std::string(keep->format) == "Q" ||
                                                             std::string(keep->format) == "<Q" ||
                                                             std::string(keep->format) == "=Q" ||
                                                             std::string(keep->format) == "L" /* LP64 */);
    if (!u64 || keep->ndim > 1)
    {
        PyBuffer_Release(keep);
        PyErr_SetString(PyExc_TypeError, "mymyr state: expected a State or a 1-d C-contiguous uint64 buffer");
        return -1;
    }
    out->words = static_cast<const uint64_t*>(keep->buf);
    out->num_words = static_cast<uint32_t>(keep->len / 8);
    out->num_numeric = 0;
    out->numeric = nullptr;
    out->task_uid = 0;
    return 0;
}

void api_state_release(Py_buffer* keep) { PyBuffer_Release(keep); }

// ------------------------------------------------------------------------------------------------ minor 1

PyTaskCore* core_of(const mymyr_task* t) { return reinterpret_cast<PyTaskCore*>(const_cast<mymyr_task*>(t)); }

bool assigned_only(const Task& T, const uint64_t* w, uint32_t n)
{
    const u32 limit = T.atoms().fluent_slots();
    for (u32 i = limit >> 6; i < n; ++i)
        if (i == (limit >> 6) ? (w[i] & ~((u64{1} << (limit & 63)) - 1)) : w[i])
            return false;
    return true;
}

int api_task_from_py(PyObject* obj, mymyr_task_view* out)
{
    const nb::handle h(obj);
    PyTaskCore* core = nullptr;
    if (nb::isinstance<PyTask>(h))
        core = nb::inst_ptr<PyTask>(h)->core.get();
    else if (nb::isinstance<PyHandle>(h))
        core = nb::inst_ptr<PyHandle>(h)->core.get();
    else if (nb::isinstance<PyState>(h))
        core = nb::inst_ptr<PyState>(h)->core;
    else if (nb::isinstance<PyAction>(h))
        core = nb::inst_ptr<PyAction>(h)->core;
    if (!core)
    {
        PyErr_SetString(PyExc_TypeError, "mymyr task: expected a Task, TaskHandle, State or Action");
        return -1;
    }
    try
    {
        const Task& T = *core->task;
        if (T.numeric_slots() > 0)
        {
            // minor 1 passes states as atom words only: a numeric task's states also carry values
            PyErr_SetString(PyExc_ValueError, "mymyr task: tasks with numeric fluents are not available through the C API "
                                              "(its states are atom words only)");
            return -1;
        }
        out->task = reinterpret_cast<const mymyr_task*>(core);
        out->uid = T.uid();
        out->fingerprint = T.fingerprint();
        out->num_objects = T.num_objects();
        out->num_predicates = static_cast<uint32_t>(T.data().predicates.size());
        out->num_schemas = T.num_schemas();
        out->words = T.words();
        out->max_words = T.max_words();
        out->label_width = core->label_width;
        out->num_atoms = T.atoms().fluent_slots();
        out->atom_mode = T.atoms().mode() == AtomMode::Frozen ? 1 : 0;
        return 0;
    }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

int64_t api_successors(const mymyr_task* t, const uint64_t* words, uint32_t num_words, mymyr_emit_fn emit, void* ctx)
{
    try
    {
        const Task& T = *core_of(t)->task;
        if (!assigned_only(T, words, num_words))
            return -1;
        Successors& succ = T.workspace().successors();
        LineVector<u64> tmp;  // per-thread hot scratch (see LineAllocator)
        int64_t n = 0;
        succ.prepare(StateView{words, num_words, nullptr, 0});
        succ.generate<true>(
            [&](u32 schema, const ObjectId* b, const Delta& d) -> bool
            {
                const u32 k = apply_delta(words, num_words, d, tmp);
                ++n;
                return emit(ctx, schema, reinterpret_cast<const uint32_t*>(b), succ.arity(schema), tmp.data(), k) == 0;
            },
            false, true);
        return n;
    }
    catch (...)
    {
        return -1;
    }
}

int64_t api_apply(const mymyr_task* t, const uint64_t* words, uint32_t num_words, uint32_t schema, const uint32_t* binding,
                  uint32_t arity, uint64_t* out, uint32_t out_capacity)
{
    try
    {
        const Task& T = *core_of(t)->task;
        if (!assigned_only(T, words, num_words) || schema >= T.num_schemas() ||
            arity != T.data().schemas[schema].arity())
            return -1;
        Successors& succ = T.workspace().successors();
        const ActionLabel label{SchemaId{schema}, {reinterpret_cast<const ObjectId*>(binding), arity}};
        const StateView s{words, num_words, nullptr, 0};
        if (!succ.is_applicable(s, label))
            return -1;
        StateBuilder b;
        succ.apply(s, label, b);
        const u32 n = bits::trimmed_size(b.words().data(), static_cast<u32>(b.words().size()));
        if (n <= out_capacity)
            for (u32 i = 0; i < n; ++i)
                out[i] = b.words()[i];
        return n;
    }
    catch (...)
    {
        return -1;
    }
}

int api_is_goal(const mymyr_task* t, const uint64_t* words, uint32_t num_words)
{
    try
    {
        const Task& T = *core_of(t)->task;
        if (!assigned_only(T, words, num_words))
            return -1;
        return T.is_goal(StateView{words, num_words, nullptr, 0}) ? 1 : 0;
    }
    catch (...)
    {
        return -1;
    }
}

int64_t api_atom_slot(const mymyr_task* t, uint32_t predicate, const uint32_t* objects, uint32_t arity)
{
    try
    {
        const Task& T = *core_of(t)->task;
        const SlotId s = T.find_atom(PredicateId{predicate}, {reinterpret_cast<const ObjectId*>(objects), arity});
        return s.valid() ? static_cast<int64_t>(s.v) : -1;
    }
    catch (...)
    {
        return -1;
    }
}

int32_t api_atom_of_slot(const mymyr_task* t, uint32_t slot, uint32_t* predicate, uint32_t* objects, uint32_t capacity)
{
    const Task& T = *core_of(t)->task;
    if (slot >= T.atoms().fluent_slots())
        return -1;
    const SlotId s{slot};
    const auto args = T.atoms().arguments(s);
    if (predicate)
        *predicate = T.atoms().predicate(s).v;
    for (u32 i = 0; i < args.size() && i < capacity; ++i)
        objects[i] = args[i];
    return static_cast<int32_t>(args.size());
}

mymyr_store* api_store_new(const mymyr_task* t)
{
    try
    {
        const TaskPtr& task = core_of(t)->task;
        return new (std::nothrow) mymyr_store{task, FlatStateStore(std::max<u32>(1, task->words()))};
    }
    catch (...)
    {
        return nullptr;
    }
}

void api_store_free(mymyr_store* s) { delete s; }

int64_t api_store_insert(mymyr_store* s, const uint64_t* words, uint32_t num_words, int* inserted)
{
    try
    {
        const auto [id, is_new] = s->store.insert(words, num_words);
        if (inserted)
            *inserted = is_new ? 1 : 0;
        return id.v;
    }
    catch (...)
    {
        return -1;
    }
}

int64_t api_store_lookup(const mymyr_store* s, const uint64_t* words, uint32_t num_words)
{
    const StateId id = s->store.find(StateView{words, num_words, nullptr, 0});
    return id.valid() ? static_cast<int64_t>(id.v) : -1;
}

const uint64_t* api_store_state(const mymyr_store* s, uint64_t id, uint32_t* num_words)
{
    if (id >= s->store.size())
        return nullptr;
    if (num_words)
        *num_words = s->store.stride();
    return s->store.words(StateId{static_cast<u32>(id)});
}

uint64_t api_store_size(const mymyr_store* s) { return s->store.size(); }

const mymyr_api k_api = {
    MYMYR_EXT_ABI_VERSION,
    MYMYR_EXT_MINOR,
    static_cast<uint32_t>(sizeof(mymyr_api)),
    0,
    // minor 0
    &api_version,
    &api_hash_words,
    &api_equal_words,
    &api_state_from_py,
    &api_state_release,
    // minor 1
    &api_task_from_py,
    &api_successors,
    &api_apply,
    &api_is_goal,
    &api_atom_slot,
    &api_atom_of_slot,
    &api_store_new,
    &api_store_free,
    &api_store_insert,
    &api_store_lookup,
    &api_store_state,
    &api_store_size,
};
}  // namespace

void bind_ext_api(nb::module_& m)
{
    PyObject* cap = PyCapsule_New(const_cast<mymyr_api*>(&k_api), MYMYR_EXT_CAPSULE_NAME, nullptr);
    if (!cap)
        throw nb::python_error();
    m.attr("_C_API") = nb::steal(cap);
    m.attr("_C_API_VERSION") = nb::make_tuple(MYMYR_EXT_ABI_VERSION, MYMYR_EXT_MINOR);
}
}  // namespace mymyr::python
