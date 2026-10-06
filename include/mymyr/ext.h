/* mymyr/ext.h: the versioned C API for downstream native extension modules.
 *
 * Downstream modules include this header (it is plain C, header-only) and fetch mymyr's function table at runtime:
 *
 *     const mymyr_api* api = mymyr_import_api();   // imports mymyr._core, returns NULL with an ImportError set
 *     if (!api) return NULL;
 *     uint64_t h = api->hash_words(words, n);
 *
 * They never link libmymyr and never share nanobind internals with mymyr, so a mymyr minor release needs no
 * downstream rebuild (mimir's six modules sharing NB_DOMAIN had to be rebuilt in lockstep).
 *
 * Minor versions: 0 = version, hashing, state views. 1 = tasks (task_from_py), successors, apply, is_goal, atom
 * lookup, state stores; state_from_py accepts mymyr.State.
 *
 * Compatibility rules:
 *   - The table is append-only. New entries go at the end and bump MYMYR_EXT_MINOR. A consumer built against a newer
 *     header checks MYMYR_API_HAS(api, field) before using an entry the running mymyr may predate.
 *   - Incompatible changes bump MYMYR_EXT_ABI_VERSION; mymyr_import_api() then fails cleanly.
 *   - Views are POD and borrowed: they are valid while the Python object they came from is alive.
 */
#ifndef MYMYR_EXT_H
#define MYMYR_EXT_H

#include <Python.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYMYR_EXT_ABI_VERSION 1
#define MYMYR_EXT_MINOR 1
#define MYMYR_EXT_CAPSULE_NAME "mymyr._core._C_API"

/* A state's words (little-endian u64, atom slot i at word i >> 6, bit i & 63; missing words read as zero). */
typedef struct mymyr_state_view
{
    const uint64_t* words;
    uint32_t num_words;
    uint32_t num_numeric; /* numeric tasks: the state's numeric u64 words (two int32 or one double per word,
                             mymyr.Task.numeric_storage); 0 for classical tasks and buffers */
    const void* numeric;  /* those words, borrowed like `words` */
    uint64_t task_uid; /* the owning task's uid, 0 if unknown */
} mymyr_state_view;

/* minor 1 ------------------------------------------------------------------------------------------------------ */

/* Opaque handles. A task pointer is borrowed from the Python object it came from (task_from_py) and valid while that
 * object is alive. A store is created by store_new, owned by the caller and freed with store_free; it keeps its task
 * alive by itself. */
typedef struct mymyr_task mymyr_task;
typedef struct mymyr_store mymyr_store;

typedef struct mymyr_task_view
{
    const mymyr_task* task;
    uint64_t uid;         /* process-unique id of the task instance (states compare by uid + words) */
    uint64_t fingerprint; /* content hash of the normalized task (equal across processes) */
    uint32_t num_objects;
    uint32_t num_predicates;
    uint32_t num_schemas;
    uint32_t words;       /* current state width in u64 words (grows only under lazy slots) */
    uint32_t max_words;   /* largest width any state can have */
    uint32_t label_width; /* largest schema arity */
    uint32_t num_atoms;   /* assigned fluent atom slots */
    uint32_t atom_mode;   /* 0 lazy slots, 1 frozen (slot = canonical id) */
} mymyr_task_view;

/* Called once per successor by successors(), in canonical order (schema, then binding). The binding lists the
 * schema's full parameter list; succ holds the successor's trimmed words. All pointers are valid during the call only.
 * Return nonzero to stop the enumeration. */
typedef int (*mymyr_emit_fn)(void* ctx, uint32_t schema, const uint32_t* binding, uint32_t arity, const uint64_t* succ,
                             uint32_t succ_words);

typedef struct mymyr_api
{
    /* header: never changes */
    uint32_t abi_version; /* MYMYR_EXT_ABI_VERSION of the provider */
    uint32_t minor;       /* MYMYR_EXT_MINOR of the provider */
    uint32_t size;        /* sizeof(mymyr_api) of the provider: entries at offsets >= size are absent */
    uint32_t reserved;

    /* minor 0 */
    const char* (*version)(void); /* mymyr's version string */
    /* State hash over the trimmed words: equal for states that compare equal (trailing zero words ignored). */
    uint64_t (*hash_words)(const uint64_t* words, uint32_t num_words);
    /* Words equality with missing words read as zero. */
    int (*equal_words)(const uint64_t* a, uint32_t na, const uint64_t* b, uint32_t nb);
    /* Fills *out from a mymyr.State (words borrowed from the State, task_uid set) or, as in minor 0, from any
     * C-contiguous 1-d uint64 buffer (task_uid 0). Returns 0, or -1 with a Python exception set. */
    int (*state_from_py)(PyObject* obj, mymyr_state_view* out, Py_buffer* keep);
    /* Releases what state_from_py acquired (always call it after a successful state_from_py). */
    void (*state_release)(Py_buffer* keep);

    /* minor 1: tasks, successors, atoms and stores (check MYMYR_API_HAS(api, task_from_py) first).
     * Every function below except task_from_py touches no Python object: call it from any thread, with or without an
     * attached thread state (release the GIL around long loops). Task functions run on the calling thread's
     * workspace, so any number of threads may use one task at once; an emit callback must not call successors,
     * apply or is_goal of the same task on its own thread. Stores are single-threaded (one per thread). */

    /* Fills *out from a mymyr.Task, TaskHandle, State or Action (its task). Returns 0, or -1 with a Python exception
     * set (ValueError for a task with numeric fluents: the functions below take atom words only). */
    int (*task_from_py)(PyObject* obj, mymyr_task_view* out);
    /* Enumerates the applicable actions of a state and their successors (witness pruning off). Returns the number of
     * successors emitted, or -1 if the state sets atom slots the task has not assigned. */
    int64_t (*successors)(const mymyr_task* task, const uint64_t* words, uint32_t num_words, mymyr_emit_fn emit, void* ctx);
    /* Writes the successor of a state under the action (schema, binding of `arity` objects) into out. Returns the
     * successor's trimmed width in words (nothing is written if it exceeds out_capacity: call again with that much
     * room), or -1 if the action is malformed or not applicable. */
    int64_t (*apply)(const mymyr_task* task, const uint64_t* words, uint32_t num_words, uint32_t schema,
                     const uint32_t* binding, uint32_t arity, uint64_t* out, uint32_t out_capacity);
    /* 1 if the state is a goal state, 0 if not (axioms are evaluated where the goal needs them). */
    int (*is_goal)(const mymyr_task* task, const uint64_t* words, uint32_t num_words);
    /* Fluent slot of predicate(objects) (indices), or -1 if it has none (not a fluent atom, unreachable, or, under
     * lazy slots, never produced yet). */
    int64_t (*atom_slot)(const mymyr_task* task, uint32_t predicate, const uint32_t* objects, uint32_t arity);
    /* Predicate and objects of a fluent slot. Writes at most `capacity` objects; returns the arity, or -1 for a slot
     * the task has not assigned. */
    int32_t (*atom_of_slot)(const mymyr_task* task, uint32_t slot, uint32_t* predicate, uint32_t* objects, uint32_t capacity);
    /* A state store with dense ids in insertion order (the flat store), or NULL on allocation failure. */
    mymyr_store* (*store_new)(const mymyr_task* task);
    void (*store_free)(mymyr_store* store);
    /* Id of the state, inserting it if new (*inserted = 1, else 0; inserted may be NULL). -1 on failure. */
    int64_t (*store_insert)(mymyr_store* store, const uint64_t* words, uint32_t num_words, int* inserted);
    /* Id of the state, or -1 if it is not in the store. */
    int64_t (*store_lookup)(const mymyr_store* store, const uint64_t* words, uint32_t num_words);
    /* Words of state `id` (*num_words set), valid until the next insert; NULL for an unknown id. */
    const uint64_t* (*store_state)(const mymyr_store* store, uint64_t id, uint32_t* num_words);
    uint64_t (*store_size)(const mymyr_store* store);
} mymyr_api;

#define MYMYR_API_HAS(api, field) (offsetof(mymyr_api, field) + sizeof(((mymyr_api*)0)->field) <= (api)->size)

/* Imports mymyr and returns its API table, or NULL with ImportError set (missing module or incompatible ABI). */
static inline const mymyr_api* mymyr_import_api(void)
{
    const mymyr_api* api = (const mymyr_api*)PyCapsule_Import(MYMYR_EXT_CAPSULE_NAME, 0);
    if (!api)
        return NULL;
    if (api->abi_version != MYMYR_EXT_ABI_VERSION)
    {
        PyErr_Format(PyExc_ImportError, "mymyr C API version %u, this module was built for %u",
                     (unsigned)api->abi_version, (unsigned)MYMYR_EXT_ABI_VERSION);
        return NULL;
    }
    return api;
}

#ifdef __cplusplus
}
#endif

#endif /* MYMYR_EXT_H */
