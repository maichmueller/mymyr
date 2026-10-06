/* A downstream extension written against mymyr/ext.h only: it includes no mymyr C++ header, links no mymyr
 * library and shares no nanobind domain. tests/python/test_ext_api.py drives it.
 * minor 0: version, hashing, state views. minor 1: tasks, successors, apply, is_goal, atom lookup, stores.
 * minor 2: numeric tasks (slots, values, metric), the state-view entries and a breadth-first search over any task. */
#include "mymyr/ext.h"

#include <stdlib.h>
#include <string.h>

static const mymyr_api* g_api = NULL;

static PyObject* abi(PyObject* self, PyObject* args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("(III)", g_api->abi_version, g_api->minor, g_api->size);
}

static PyObject* version(PyObject* self, PyObject* args)
{
    (void)self;
    (void)args;
    return PyUnicode_FromString(g_api->version());
}

/* hash_state(words) -> int: hashes any State-like object through the API. */
static PyObject* hash_state(PyObject* self, PyObject* obj)
{
    (void)self;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->state_from_py(obj, &v, &keep) != 0)
        return NULL;
    uint64_t h = g_api->hash_words(v.words, v.num_words);
    g_api->state_release(&keep);
    return PyLong_FromUnsignedLongLong(h);
}

static PyObject* equal_states(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;
    mymyr_state_view va, vb;
    Py_buffer ka, kb;
    if (g_api->state_from_py(a, &va, &ka) != 0)
        return NULL;
    if (g_api->state_from_py(b, &vb, &kb) != 0)
    {
        g_api->state_release(&ka);
        return NULL;
    }
    int eq = g_api->equal_words(va.words, va.num_words, vb.words, vb.num_words);
    g_api->state_release(&ka);
    g_api->state_release(&kb);
    return PyBool_FromLong(eq);
}

/* ----------------------------------------------------------------------------------------------------- minor 1 */

static int require_minor1(void)
{
    if (!MYMYR_API_HAS(g_api, store_size))
    {
        PyErr_SetString(PyExc_RuntimeError, "the running mymyr predates C API minor 1");
        return -1;
    }
    return 0;
}

static PyObject* words_tuple(const uint64_t* w, uint32_t n)
{
    PyObject* t = PyTuple_New(n);
    if (!t)
        return NULL;
    for (uint32_t i = 0; i < n; ++i)
        PyTuple_SET_ITEM(t, i, PyLong_FromUnsignedLongLong(w[i]));
    return t;
}

static PyObject* binding_tuple(const uint32_t* b, uint32_t n)
{
    PyObject* t = PyTuple_New(n);
    if (!t)
        return NULL;
    for (uint32_t i = 0; i < n; ++i)
        PyTuple_SET_ITEM(t, i, PyLong_FromUnsignedLong(b[i]));
    return t;
}

/* task_info(obj) -> (uid, fingerprint, num_objects, num_predicates, num_schemas, words, max_words, label_width,
 *                    num_atoms, atom_mode) */
static PyObject* task_info(PyObject* self, PyObject* obj)
{
    (void)self;
    mymyr_task_view t;
    if (require_minor1() != 0 || g_api->task_from_py(obj, &t) != 0)
        return NULL;
    return Py_BuildValue("(KKIIIIIIII)", (unsigned long long)t.uid, (unsigned long long)t.fingerprint, t.num_objects,
                         t.num_predicates, t.num_schemas, t.words, t.max_words, t.label_width, t.num_atoms, t.atom_mode);
}

/* state_task_uid(state) -> the task uid state_from_py reports (0 for plain buffers) */
static PyObject* state_task_uid(PyObject* self, PyObject* obj)
{
    (void)self;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->state_from_py(obj, &v, &keep) != 0)
        return NULL;
    uint64_t uid = v.task_uid;
    g_api->state_release(&keep);
    return PyLong_FromUnsignedLongLong(uid);
}

typedef struct
{
    PyObject* list;
    int failed;
    int64_t stop_after; /* -1: never */
} collect_ctx;

static int collect(void* p, uint32_t schema, const uint32_t* binding, uint32_t arity, const uint64_t* succ, uint32_t n)
{
    collect_ctx* c = (collect_ctx*)p;
    /* This callback touches Python objects, so successors() is called with the thread attached (see below). */
    PyObject* item = Py_BuildValue("(INN)", schema, binding_tuple(binding, arity), words_tuple(succ, n));
    if (!item || PyList_Append(c->list, item) != 0)
        c->failed = 1;
    Py_XDECREF(item);
    if (c->failed)
        return 1;
    return c->stop_after >= 0 && PyList_GET_SIZE(c->list) >= c->stop_after;
}

/* successors(task, state, stop_after=-1) -> [(schema, binding, words), ...] in canonical order */
static PyObject* successors(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    long long stop_after = -1;
    if (!PyArg_ParseTuple(args, "OO|L", &task, &state, &stop_after) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    collect_ctx c = {PyList_New(0), 0, stop_after};
    int64_t n = c.list ? g_api->successors(t.task, v.words, v.num_words, collect, &c) : -1;
    g_api->state_release(&keep);
    if (!c.list || c.failed)
    {
        Py_XDECREF(c.list);
        return NULL;
    }
    if (n < 0)
    {
        Py_DECREF(c.list);
        PyErr_SetString(PyExc_ValueError, "successors() failed");
        return NULL;
    }
    return c.list;
}

static int count_only(void* p, uint32_t schema, const uint32_t* binding, uint32_t arity, const uint64_t* succ, uint32_t n)
{
    (void)schema;
    (void)binding;
    (void)arity;
    (void)succ;
    (void)n;
    ++*(int64_t*)p;
    return 0;
}

/* count_successors_nogil(task, state) -> int: the enumeration runs with the thread detached (no Python calls). */
static PyObject* count_successors_nogil(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    int64_t count = 0, n;
    Py_BEGIN_ALLOW_THREADS
    n = g_api->successors(t.task, v.words, v.num_words, count_only, &count);
    Py_END_ALLOW_THREADS
    g_api->state_release(&keep);
    if (n < 0)
    {
        PyErr_SetString(PyExc_ValueError, "successors() failed");
        return NULL;
    }
    return PyLong_FromLongLong(count);
}

static int binding_from(PyObject* seq, uint32_t* out, uint32_t cap, uint32_t* n)
{
    PyObject* fast = PySequence_Fast(seq, "binding must be a sequence of object indices");
    if (!fast)
        return -1;
    Py_ssize_t k = PySequence_Fast_GET_SIZE(fast);
    if ((size_t)k > cap)
    {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "binding too long");
        return -1;
    }
    for (Py_ssize_t i = 0; i < k; ++i)
        out[i] = (uint32_t)PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(fast, i));
    Py_DECREF(fast);
    *n = (uint32_t)k;
    return PyErr_Occurred() ? -1 : 0;
}

/* apply(task, state, schema, binding) -> words tuple, or None if not applicable */
static PyObject* apply(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state, *bind;
    unsigned int schema;
    if (!PyArg_ParseTuple(args, "OOIO", &task, &state, &schema, &bind) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    uint32_t binding[64], arity = 0;
    if (g_api->task_from_py(task, &t) != 0 || binding_from(bind, binding, 64, &arity) != 0 ||
        g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    uint64_t small[1];
    int64_t n = g_api->apply(t.task, v.words, v.num_words, schema, binding, arity, small, 0); /* ask for the width */
    PyObject* result = NULL;
    if (n < 0)
        result = Py_NewRef(Py_None);
    else
    {
        uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
        if (!out)
            result = PyErr_NoMemory();
        else
        {
            int64_t m = g_api->apply(t.task, v.words, v.num_words, schema, binding, arity, out, (uint32_t)n);
            result = m == n ? words_tuple(out, (uint32_t)n) : (PyErr_SetString(PyExc_RuntimeError, "apply() width changed"), NULL);
            free(out);
        }
    }
    g_api->state_release(&keep);
    return result;
}

static PyObject* is_goal(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    int g = g_api->is_goal(t.task, v.words, v.num_words);
    g_api->state_release(&keep);
    if (g < 0)
    {
        PyErr_SetString(PyExc_ValueError, "is_goal() failed");
        return NULL;
    }
    return PyBool_FromLong(g);
}

/* atom_slot(task, predicate, objects) -> slot or -1 */
static PyObject* atom_slot(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *objs;
    unsigned int pred;
    if (!PyArg_ParseTuple(args, "OIO", &task, &pred, &objs) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    uint32_t o[64], k = 0;
    if (g_api->task_from_py(task, &t) != 0 || binding_from(objs, o, 64, &k) != 0)
        return NULL;
    return PyLong_FromLongLong(g_api->atom_slot(t.task, pred, o, k));
}

/* atom_of_slot(task, slot) -> (predicate, objects) or None */
static PyObject* atom_of_slot(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* task;
    unsigned int slot;
    if (!PyArg_ParseTuple(args, "OI", &task, &slot) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    if (g_api->task_from_py(task, &t) != 0)
        return NULL;
    uint32_t pred = 0, o[64];
    int32_t k = g_api->atom_of_slot(t.task, slot, &pred, o, 64);
    if (k < 0)
        Py_RETURN_NONE;
    return Py_BuildValue("(IN)", pred, binding_tuple(o, (uint32_t)k));
}

/* store_roundtrip(task, [states]) -> (ids, inserted flags, lookups, size, words of id 0 or None) */
static PyObject* store_roundtrip(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *states;
    if (!PyArg_ParseTuple(args, "OO", &task, &states) || require_minor1() != 0)
        return NULL;
    mymyr_task_view t;
    if (g_api->task_from_py(task, &t) != 0)
        return NULL;
    PyObject* fast = PySequence_Fast(states, "states must be a sequence");
    if (!fast)
        return NULL;
    mymyr_store* store = g_api->store_new(t.task);
    if (!store)
    {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject *ids = PyList_New(n), *flags = PyList_New(n), *lookups = PyList_New(n);
    PyObject* result = NULL;
    int ok = ids && flags && lookups;
    for (Py_ssize_t i = 0; ok && i < n; ++i)
    {
        mymyr_state_view v;
        Py_buffer keep;
        if (g_api->state_from_py(PySequence_Fast_GET_ITEM(fast, i), &v, &keep) != 0)
        {
            ok = 0;
            break;
        }
        int inserted = -1;
        int64_t id = g_api->store_insert(store, v.words, v.num_words, &inserted);
        g_api->state_release(&keep);
        PyList_SET_ITEM(ids, i, PyLong_FromLongLong(id));
        PyList_SET_ITEM(flags, i, PyBool_FromLong(inserted));
    }
    for (Py_ssize_t i = 0; ok && i < n; ++i)
    {
        mymyr_state_view v;
        Py_buffer keep;
        if (g_api->state_from_py(PySequence_Fast_GET_ITEM(fast, i), &v, &keep) != 0)
        {
            ok = 0;
            break;
        }
        PyList_SET_ITEM(lookups, i, PyLong_FromLongLong(g_api->store_lookup(store, v.words, v.num_words)));
        g_api->state_release(&keep);
    }
    if (ok)
    {
        uint32_t nw = 0;
        const uint64_t* w0 = g_api->store_state(store, 0, &nw);
        PyObject* first = w0 ? words_tuple(w0, nw) : Py_NewRef(Py_None);
        result = Py_BuildValue("(OOOKN)", ids, flags, lookups, (unsigned long long)g_api->store_size(store), first);
    }
    g_api->store_free(store);
    Py_XDECREF(ids);
    Py_XDECREF(flags);
    Py_XDECREF(lookups);
    Py_DECREF(fast);
    return result;
}

/* ----------------------------------------------------------------------------------------------------- minor 2 */

static int g_pretend_size = 0; /* set by pretend_size(): gates minor 2 as a provider of that table size would */

static int has_minor2(void)
{
    mymyr_api view = *g_api;
    if (g_pretend_size > 0)
        view.size = (uint32_t)g_pretend_size;
    return MYMYR_API_HAS(&view, store_state_view);
}

static int require_minor2(void)
{
    if (!has_minor2())
    {
        PyErr_SetString(PyExc_RuntimeError, "the running mymyr predates C API minor 2");
        return -1;
    }
    return 0;
}

/* pretend_size(n): later minor-2 calls see a table of n bytes (0: the real one); returns whether minor 2 is present. */
static PyObject* pretend_size(PyObject* self, PyObject* arg)
{
    (void)self;
    long n = PyLong_AsLong(arg);
    if (n == -1 && PyErr_Occurred())
        return NULL;
    g_pretend_size = (int)n;
    return PyBool_FromLong(has_minor2());
}

/* table_layout() -> (offset of the last minor-1 entry, end of it, offset of the first minor-2 entry, table size):
 * entries are appended, never moved */
static PyObject* table_layout(PyObject* self, PyObject* args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("(nnnn)", (Py_ssize_t)offsetof(mymyr_api, store_size),
                         (Py_ssize_t)(offsetof(mymyr_api, store_size) + sizeof(g_api->store_size)),
                         (Py_ssize_t)offsetof(mymyr_api, task_numeric), (Py_ssize_t)sizeof(mymyr_api));
}

static PyObject* numeric_words_tuple(const mymyr_state_view* v)
{
    return words_tuple((const uint64_t*)v->numeric, v->num_numeric);
}

/* numeric_info(task) -> (slots, words, storage, metric) */
static PyObject* numeric_info(PyObject* self, PyObject* obj)
{
    (void)self;
    mymyr_task_view t;
    mymyr_numeric_view n;
    if (require_minor2() != 0 || g_api->task_from_py(obj, &t) != 0)
        return NULL;
    if (g_api->task_numeric(t.task, &n) != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "task_numeric() failed");
        return NULL;
    }
    return Py_BuildValue("(IIII)", n.slots, n.words, n.storage, n.metric);
}

/* numeric_names(task) -> [(name, slot found by that name)] for every numeric slot */
static PyObject* numeric_names(PyObject* self, PyObject* obj)
{
    (void)self;
    mymyr_task_view t;
    mymyr_numeric_view n;
    if (require_minor2() != 0 || g_api->task_from_py(obj, &t) != 0 || g_api->task_numeric(t.task, &n) != 0)
        return NULL;
    PyObject* list = PyList_New(n.slots);
    char buf[512];
    for (uint32_t i = 0; list && i < n.slots; ++i)
    {
        int32_t len = g_api->numeric_slot_name(t.task, i, buf, sizeof buf);
        if (len < 0 || (size_t)len >= sizeof buf)
        {
            Py_DECREF(list);
            PyErr_SetString(PyExc_RuntimeError, "numeric_slot_name() failed");
            return NULL;
        }
        PyList_SET_ITEM(list, i, Py_BuildValue("(sL)", buf, (long long)g_api->numeric_slot(t.task, buf)));
    }
    return list;
}

/* numeric_name_truncated(task, slot, capacity) -> (the buffer's string, the full length), or None for no such slot */
static PyObject* numeric_name_truncated(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* task;
    unsigned int slot, capacity;
    if (!PyArg_ParseTuple(args, "OII", &task, &slot, &capacity) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    if (g_api->task_from_py(task, &t) != 0)
        return NULL;
    char buf[512] = {0};
    int32_t len = g_api->numeric_slot_name(t.task, slot, buf, capacity < sizeof buf ? capacity : (unsigned)sizeof buf);
    if (len < 0)
        Py_RETURN_NONE;
    return Py_BuildValue("(si)", buf, len);
}

/* numeric_slot_of(task, name) -> slot or -1 */
static PyObject* numeric_slot_of(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* task;
    const char* name;
    if (!PyArg_ParseTuple(args, "Os", &task, &name) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    if (g_api->task_from_py(task, &t) != 0)
        return NULL;
    return PyLong_FromLongLong(g_api->numeric_slot(t.task, name));
}

/* numeric_values(task, state) -> [float] in slot order */
static PyObject* numeric_values(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_numeric_view n;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->task_numeric(t.task, &n) != 0 ||
        g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    double* out = (double*)malloc(sizeof(double) * (n.slots ? n.slots : 1));
    PyObject* list = NULL;
    if (!out)
        PyErr_NoMemory();
    else
    {
        int32_t k = g_api->numeric_values(t.task, &v, out, n.slots);
        if (k != (int32_t)n.slots)
            PyErr_SetString(PyExc_ValueError, "numeric_values() failed");
        else if ((list = PyList_New(k)))
            for (int32_t i = 0; i < k; ++i)
                PyList_SET_ITEM(list, i, PyFloat_FromDouble(out[i]));
        free(out);
    }
    g_api->state_release(&keep);
    return list;
}

/* metric_initial(task, state) -> float */
static PyObject* metric_initial(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    double g = 0;
    int rc = g_api->metric_initial(t.task, &v, &g);
    g_api->state_release(&keep);
    if (rc != 0)
    {
        PyErr_SetString(PyExc_ValueError, "metric_initial() failed");
        return NULL;
    }
    return PyFloat_FromDouble(g);
}

typedef struct
{
    PyObject* list;
    int failed;
} state_collect_ctx;

static int collect_state(void* p, uint32_t schema, const uint32_t* binding, uint32_t arity, const mymyr_state_view* succ,
                         double metric)
{
    state_collect_ctx* c = (state_collect_ctx*)p;
    PyObject* item = Py_BuildValue("(INNNd)", schema, binding_tuple(binding, arity), words_tuple(succ->words, succ->num_words),
                                   numeric_words_tuple(succ), metric);
    if (!item || PyList_Append(c->list, item) != 0)
        c->failed = 1;
    Py_XDECREF(item);
    return c->failed;
}

/* successors_state(task, state, g) -> [(schema, binding, words, numeric words, metric)] in canonical order */
static PyObject* successors_state(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    double g = 0;
    if (!PyArg_ParseTuple(args, "OOd", &task, &state, &g) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    state_collect_ctx c = {PyList_New(0), 0};
    int64_t n = c.list ? g_api->successors_state(t.task, &v, g, collect_state, &c) : -1;
    g_api->state_release(&keep);
    if (!c.list || c.failed)
    {
        Py_XDECREF(c.list);
        return NULL;
    }
    if (n < 0)
    {
        Py_DECREF(c.list);
        PyErr_SetString(PyExc_ValueError, "successors_state() failed");
        return NULL;
    }
    return c.list;
}

static int count_state(void* p, uint32_t schema, const uint32_t* binding, uint32_t arity, const mymyr_state_view* succ,
                       double metric)
{
    (void)schema;
    (void)binding;
    (void)arity;
    (void)succ;
    (void)metric;
    ++*(int64_t*)p;
    return 0;
}

/* count_successors_state_nogil(task, state) -> int, with the thread detached */
static PyObject* count_successors_state_nogil(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    int64_t count = 0, n;
    Py_BEGIN_ALLOW_THREADS
    n = g_api->successors_state(t.task, &v, 0, count_state, &count);
    Py_END_ALLOW_THREADS
    g_api->state_release(&keep);
    if (n < 0)
    {
        PyErr_SetString(PyExc_ValueError, "successors_state() failed");
        return NULL;
    }
    return PyLong_FromLongLong(count);
}

/* apply_state(task, state, g, schema, binding) -> (words, numeric words, metric), or None if not applicable */
static PyObject* apply_state(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state, *bind;
    double g;
    unsigned int schema;
    if (!PyArg_ParseTuple(args, "OOdIO", &task, &state, &g, &schema, &bind) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_numeric_view nv;
    mymyr_state_view v;
    Py_buffer keep;
    uint32_t binding[64], arity = 0;
    if (g_api->task_from_py(task, &t) != 0 || g_api->task_numeric(t.task, &nv) != 0 ||
        binding_from(bind, binding, 64, &arity) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    uint64_t* num = (uint64_t*)malloc(sizeof(uint64_t) * (nv.words ? nv.words : 1));
    double out_g = 0;
    PyObject* result = NULL;
    int64_t n = num ? g_api->apply_state(t.task, &v, g, schema, binding, arity, NULL, 0, num, &out_g) : -1; /* the width */
    if (!num)
        PyErr_NoMemory();
    else if (n < 0)
        result = Py_NewRef(Py_None);
    else
    {
        uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
        if (!out)
            PyErr_NoMemory();
        else
        {
            int64_t m = g_api->apply_state(t.task, &v, g, schema, binding, arity, out, (uint32_t)n, num, &out_g);
            if (m != n)
                PyErr_SetString(PyExc_RuntimeError, "apply_state() width changed");
            else
                result = Py_BuildValue("(NNd)", words_tuple(out, (uint32_t)n), words_tuple(num, nv.words), out_g);
            free(out);
        }
    }
    free(num);
    g_api->state_release(&keep);
    return result;
}

static PyObject* is_goal_state(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    int g = g_api->is_goal_state(t.task, &v);
    g_api->state_release(&keep);
    if (g < 0)
    {
        PyErr_SetString(PyExc_ValueError, "is_goal_state() failed");
        return NULL;
    }
    return PyBool_FromLong(g);
}

/* state_hash_equal(a, b) -> (hash(a), hash(b), a == b) through the state-view entries */
static PyObject* state_hash_equal(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b) || require_minor2() != 0)
        return NULL;
    mymyr_state_view va, vb;
    Py_buffer ka, kb;
    if (g_api->state_from_py(a, &va, &ka) != 0)
        return NULL;
    if (g_api->state_from_py(b, &vb, &kb) != 0)
    {
        g_api->state_release(&ka);
        return NULL;
    }
    PyObject* r = Py_BuildValue("(KKN)", (unsigned long long)g_api->state_hash(&va), (unsigned long long)g_api->state_hash(&vb),
                                PyBool_FromLong(g_api->state_equal(&va, &vb)));
    g_api->state_release(&ka);
    g_api->state_release(&kb);
    return r;
}

/* minor1_on_numeric(task, state) -> (successors, apply, is_goal, store_insert, store_lookup): what the atom-words
 * entries of minor 1 return on a task with numeric slots (all -1) */
static PyObject* minor1_on_numeric(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    if (!PyArg_ParseTuple(args, "OO", &task, &state) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    uint64_t out[8];
    uint32_t binding[1] = {0};
    int64_t counted = 0;
    long long r_succ = g_api->successors(t.task, v.words, v.num_words, count_only, &counted);
    long long r_apply = g_api->apply(t.task, v.words, v.num_words, 0, binding, 0, out, 8);
    long long r_goal = g_api->is_goal(t.task, v.words, v.num_words);
    mymyr_store* store = g_api->store_new(t.task);
    long long r_ins = store ? g_api->store_insert(store, v.words, v.num_words, NULL) : -2;
    long long r_look = store ? g_api->store_lookup(store, v.words, v.num_words) : -2;
    if (store)
        g_api->store_free(store);
    g_api->state_release(&keep);
    return Py_BuildValue("(LLLLL)", r_succ, r_apply, r_goal, r_ins, r_look);
}

/* store_roundtrip_state(task, [states]) -> (ids, inserted flags, lookups, size, (words, numeric words) of id 0) */
static PyObject* store_roundtrip_state(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *states;
    if (!PyArg_ParseTuple(args, "OO", &task, &states) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    if (g_api->task_from_py(task, &t) != 0)
        return NULL;
    PyObject* fast = PySequence_Fast(states, "states must be a sequence");
    if (!fast)
        return NULL;
    mymyr_store* store = g_api->store_new(t.task);
    if (!store)
    {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject *ids = PyList_New(n), *flags = PyList_New(n), *lookups = PyList_New(n);
    PyObject* result = NULL;
    int ok = ids && flags && lookups;
    for (int pass = 0; ok && pass < 2; ++pass)
        for (Py_ssize_t i = 0; ok && i < n; ++i)
        {
            mymyr_state_view v;
            Py_buffer keep;
            if (g_api->state_from_py(PySequence_Fast_GET_ITEM(fast, i), &v, &keep) != 0)
            {
                ok = 0;
                break;
            }
            if (pass == 0)
            {
                int inserted = -1;
                int64_t id = g_api->store_insert_state(store, &v, &inserted);
                PyList_SET_ITEM(ids, i, PyLong_FromLongLong(id));
                PyList_SET_ITEM(flags, i, PyBool_FromLong(inserted));
            }
            else
                PyList_SET_ITEM(lookups, i, PyLong_FromLongLong(g_api->store_lookup_state(store, &v)));
            g_api->state_release(&keep);
        }
    if (ok)
    {
        mymyr_state_view f;
        PyObject* first = g_api->store_state_view(store, 0, &f) == 0
                              ? Py_BuildValue("(NN)", words_tuple(f.words, f.num_words), numeric_words_tuple(&f))
                              : Py_NewRef(Py_None);
        result = Py_BuildValue("(OOOKN)", ids, flags, lookups, (unsigned long long)g_api->store_size(store), first);
    }
    g_api->store_free(store);
    Py_XDECREF(ids);
    Py_XDECREF(flags);
    Py_XDECREF(lookups);
    Py_DECREF(fast);
    return result;
}

/* A breadth-first search written against the C API alone: ids are the store's, in insertion order, so the queue is the
 * id range. Every node keeps its parent, the action that reached it and its metric value. */
typedef struct
{
    int64_t* parent;
    uint32_t* schema;
    uint32_t* arity;
    size_t* offset;
    double* metric;
    uint32_t* binding;
    size_t size, capacity, bindings, bindings_capacity;
} bfs_nodes;

static int bfs_reserve(void** p, size_t bytes)
{
    void* q = realloc(*p, bytes);
    if (!q)
        return -1;
    *p = q;
    return 0;
}

static int bfs_push(bfs_nodes* n, int64_t parent, uint32_t schema, const uint32_t* binding, uint32_t arity, double metric)
{
    if (n->size == n->capacity)
    {
        size_t cap = n->capacity ? n->capacity * 2 : 1024;
        if (bfs_reserve((void**)&n->parent, cap * sizeof *n->parent) || bfs_reserve((void**)&n->schema, cap * sizeof *n->schema) ||
            bfs_reserve((void**)&n->arity, cap * sizeof *n->arity) || bfs_reserve((void**)&n->offset, cap * sizeof *n->offset) ||
            bfs_reserve((void**)&n->metric, cap * sizeof *n->metric))
            return -1;
        n->capacity = cap;
    }
    if (n->bindings + arity > n->bindings_capacity)
    {
        size_t cap = n->bindings_capacity ? n->bindings_capacity * 2 : 4096;
        while (cap < n->bindings + arity)
            cap *= 2;
        if (bfs_reserve((void**)&n->binding, cap * sizeof *n->binding))
            return -1;
        n->bindings_capacity = cap;
    }
    n->parent[n->size] = parent;
    n->schema[n->size] = schema;
    n->arity[n->size] = arity;
    n->offset[n->size] = n->bindings;
    n->metric[n->size] = metric;
    for (uint32_t i = 0; i < arity; ++i)
        n->binding[n->bindings + i] = binding[i];
    n->bindings += arity;
    ++n->size;
    return 0;
}

static void bfs_free(bfs_nodes* n)
{
    free(n->parent);
    free(n->schema);
    free(n->arity);
    free(n->offset);
    free(n->metric);
    free(n->binding);
}

typedef struct
{
    mymyr_store* store;
    bfs_nodes* nodes;
    int64_t parent;
    uint64_t generated;
    int failed;
} bfs_ctx;

static int bfs_emit(void* p, uint32_t schema, const uint32_t* binding, uint32_t arity, const mymyr_state_view* succ,
                    double metric)
{
    bfs_ctx* c = (bfs_ctx*)p;
    int inserted = 0;
    ++c->generated;
    int64_t id = g_api->store_insert_state(c->store, succ, &inserted);
    if (id < 0 || (inserted && bfs_push(c->nodes, c->parent, schema, binding, arity, metric) != 0))
    {
        c->failed = 1;
        return 1;
    }
    return 0;
}

typedef struct
{
    uint64_t states, expanded, generated, goal_states;
    uint32_t layers;
    int exhausted, solved, failed;
    int64_t goal;
} bfs_result;

/* The search proper: no Python object is touched, so it runs with the thread detached. */
static void run_bfs(const mymyr_task* task, const mymyr_state_view* init, double g0, uint64_t max_states, int stop_at_goal,
                    bfs_result* r, bfs_nodes* nodes)
{
    memset(r, 0, sizeof *r);
    r->goal = -1;
    mymyr_store* store = g_api->store_new(task);
    uint64_t *words = NULL, *numeric = NULL;
    uint32_t words_capacity = 0;
    if (!store || g_api->store_insert_state(store, init, NULL) < 0 || bfs_push(nodes, -1, 0, NULL, 0, g0) != 0)
    {
        r->failed = 1;
        if (store)
            g_api->store_free(store);
        return;
    }
    bfs_ctx ctx = {store, nodes, 0, 0, 0};
    uint64_t pos = 0, layer_end = 0;
    for (; pos < g_api->store_size(store) && g_api->store_size(store) < max_states; ++pos)
    {
        if (pos == layer_end)
        {
            ++r->layers;
            layer_end = g_api->store_size(store);
        }
        mymyr_state_view cur;
        if (g_api->store_state_view(store, pos, &cur) != 0)
        {
            r->failed = 1;
            break;
        }
        /* the store's arena may move while the successors are inserted: expand a copy */
        if (cur.num_words > words_capacity)
        {
            free(words);
            words = (uint64_t*)malloc(sizeof(uint64_t) * cur.num_words);
            words_capacity = cur.num_words;
            if (!words)
            {
                r->failed = 1;
                break;
            }
        }
        memcpy(words, cur.words, sizeof(uint64_t) * cur.num_words);
        if (cur.num_numeric)
        {
            if (!numeric)
                numeric = (uint64_t*)malloc(sizeof(uint64_t) * cur.num_numeric);
            if (!numeric)
            {
                r->failed = 1;
                break;
            }
            memcpy(numeric, cur.numeric, sizeof(uint64_t) * cur.num_numeric);
        }
        cur.words = words;
        cur.numeric = numeric;
        int goal = g_api->is_goal_state(task, &cur);
        if (goal < 0)
        {
            r->failed = 1;
            break;
        }
        r->goal_states += (uint64_t)goal;
        if (goal && stop_at_goal)
        {
            r->solved = 1;
            r->goal = (int64_t)pos;
            break;
        }
        ++r->expanded;
        ctx.parent = (int64_t)pos;
        if (g_api->successors_state(task, &cur, nodes->metric[pos], bfs_emit, &ctx) < 0 || ctx.failed)
        {
            r->failed = 1;
            break;
        }
    }
    r->generated = ctx.generated;
    r->states = g_api->store_size(store);
    r->exhausted = !r->solved && !r->failed && pos == r->states;
    free(words);
    free(numeric);
    g_api->store_free(store);
}

/* brfs(task, state, g, max_states, stop_at_goal) -> (states, expanded, generated, goal_states, layers, exhausted,
 * solved, plan, cost): the whole search runs through the C API with the thread detached. plan is the list of
 * (schema, binding) to the first goal state found (stop_at_goal), cost its metric value. */
static PyObject* brfs(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *task, *state;
    double g0;
    unsigned long long max_states;
    int stop_at_goal;
    if (!PyArg_ParseTuple(args, "OOdKp", &task, &state, &g0, &max_states, &stop_at_goal) || require_minor2() != 0)
        return NULL;
    mymyr_task_view t;
    mymyr_state_view v;
    Py_buffer keep;
    if (g_api->task_from_py(task, &t) != 0 || g_api->state_from_py(state, &v, &keep) != 0)
        return NULL;
    bfs_result r;
    bfs_nodes nodes = {0};
    Py_BEGIN_ALLOW_THREADS
    run_bfs(t.task, &v, g0, max_states, stop_at_goal, &r, &nodes);
    Py_END_ALLOW_THREADS
    g_api->state_release(&keep);
    if (r.failed)
    {
        bfs_free(&nodes);
        PyErr_SetString(PyExc_RuntimeError, "the search through the C API failed");
        return NULL;
    }
    PyObject* plan = PyList_New(0);
    double cost = 0;
    if (r.goal >= 0 && plan)
    {
        cost = nodes.metric[r.goal];
        for (int64_t i = r.goal; i > 0 && plan; i = nodes.parent[i])
        {
            PyObject* step = Py_BuildValue("(IN)", nodes.schema[i], binding_tuple(nodes.binding + nodes.offset[i], nodes.arity[i]));
            if (!step || PyList_Insert(plan, 0, step) != 0)
            {
                Py_XDECREF(plan);
                plan = NULL;
            }
            Py_XDECREF(step);
        }
    }
    bfs_free(&nodes);
    if (!plan)
        return NULL;
    return Py_BuildValue("(KKKKIOONd)", (unsigned long long)r.states, (unsigned long long)r.expanded,
                         (unsigned long long)r.generated, (unsigned long long)r.goal_states, r.layers,
                         r.exhausted ? Py_True : Py_False, r.solved ? Py_True : Py_False, plan, cost);
}

static PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "(abi_version, minor, table size) of the running mymyr"},
    {"version", version, METH_NOARGS, "mymyr version through the C API"},
    {"hash_state", hash_state, METH_O, "state hash through the C API"},
    {"equal_states", equal_states, METH_VARARGS, "state equality through the C API"},
    {"task_info", task_info, METH_O, "task view through the C API (minor 1)"},
    {"state_task_uid", state_task_uid, METH_O, "task uid of a state view"},
    {"successors", successors, METH_VARARGS, "(schema, binding, words) triples through the C API"},
    {"count_successors_nogil", count_successors_nogil, METH_VARARGS, "successor count with the thread detached"},
    {"apply", apply, METH_VARARGS, "successor words through the C API, None if not applicable"},
    {"is_goal", is_goal, METH_VARARGS, "goal test through the C API"},
    {"atom_slot", atom_slot, METH_VARARGS, "slot of predicate(objects), -1 if none"},
    {"atom_of_slot", atom_of_slot, METH_VARARGS, "(predicate, objects) of a slot"},
    {"store_roundtrip", store_roundtrip, METH_VARARGS, "insert and look up states in a C API store"},
    {"pretend_size", pretend_size, METH_O, "gate minor 2 as a provider with a table of that many bytes would"},
    {"table_layout", table_layout, METH_NOARGS, "(offset and end of the last minor-1 entry, first minor-2 entry, size)"},
    {"numeric_info", numeric_info, METH_O, "(slots, words, storage, metric) of a task (minor 2)"},
    {"numeric_names", numeric_names, METH_O, "[(name, slot found by name)] of every numeric slot"},
    {"numeric_name_truncated", numeric_name_truncated, METH_VARARGS, "a slot name through a buffer of a given capacity"},
    {"numeric_slot_of", numeric_slot_of, METH_VARARGS, "slot of a numeric slot name, -1 if none"},
    {"numeric_values", numeric_values, METH_VARARGS, "the values of a state's numeric slots"},
    {"metric_initial", metric_initial, METH_VARARGS, "metric value of an initial state"},
    {"successors_state", successors_state, METH_VARARGS, "successors with numeric words and metric values"},
    {"count_successors_state_nogil", count_successors_state_nogil, METH_VARARGS, "successor count, thread detached"},
    {"apply_state", apply_state, METH_VARARGS, "successor words, numeric words and metric value, None if not applicable"},
    {"is_goal_state", is_goal_state, METH_VARARGS, "goal test through the state-view entry"},
    {"state_hash_equal", state_hash_equal, METH_VARARGS, "state hashes and equality through the state-view entries"},
    {"minor1_on_numeric", minor1_on_numeric, METH_VARARGS, "what the minor-1 entries return on a numeric task"},
    {"store_roundtrip_state", store_roundtrip_state, METH_VARARGS, "insert and look up states in a numeric-aware store"},
    {"brfs", brfs, METH_VARARGS, "breadth-first search through the C API"},
    {NULL, NULL, 0, NULL},
};

static int exec_module(PyObject* m)
{
    (void)m;
    g_api = mymyr_import_api();
    return g_api ? 0 : -1;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void*)exec_module},
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ext_consumer", NULL, 0, methods, slots, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__ext_consumer(void) { return PyModuleDef_Init(&module); }
