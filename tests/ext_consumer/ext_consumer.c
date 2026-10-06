/* A downstream extension written against mymyr/ext.h only: it includes no mymyr C++ header, links no mymyr
 * library and shares no nanobind domain. tests/python/test_ext_api.py drives it.
 * minor 0: version, hashing, state views. minor 1: tasks, successors, apply, is_goal, atom lookup, stores. */
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
