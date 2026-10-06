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
 * lookup, state stores; state_from_py accepts mymyr.State. 2 = numeric tasks: numeric slots and values, action costs
 * (the metric), and successors / apply / is_goal / stores over states that carry numeric values.
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
#define MYMYR_EXT_MINOR 2
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

/* minor 2 ------------------------------------------------------------------------------------------------------ */

typedef struct mymyr_numeric_view
{
    uint32_t slots;   /* numeric slots (ground fluent functions with an initial value); 0 for a classical task */
    uint32_t words;   /* numeric u64 words per state: slots (storage 0) or ceil(slots / 2) (storage 1) */
    uint32_t storage; /* 0: one IEEE double per word (its bit pattern); 1: two int32 per word, slot 2k in the low half */
    uint32_t metric;  /* 0: unit costs, 1: total-cost effects, 2: the problem's metric expression */
} mymyr_numeric_view;

/* Called once per successor by successors_state(), in canonical order: the action (schema and full binding as for
 * mymyr_emit_fn), the successor state and its metric value. Return nonzero to stop the enumeration. */
typedef int (*mymyr_emit_state_fn)(void* ctx, uint32_t schema, const uint32_t* binding, uint32_t arity,
                                   const mymyr_state_view* succ, double metric);

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
     * set. Numeric tasks are accepted, but the entries of this minor take atom words only: on a task with numeric
     * slots (task_numeric, minor 2) successors, apply, is_goal, store_insert and store_lookup return -1; use their
     * *_state counterparts. */
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

    /* minor 2: numeric tasks (check MYMYR_API_HAS(api, store_state_view) first). The thread rules of minor 1 apply.
     *
     * A numeric task's state is its atom words plus its numeric words (mymyr_state_view.numeric, task_numeric().words
     * of them, the encoding of numeric_values below): the *_state entries take and return states as
     * mymyr_state_view, and a view whose num_numeric is not the task's word count is rejected with -1. They also work
     * on classical tasks (no numeric words).
     *
     * Metric. A search's path cost is the metric value g of mimir: the initial state has metric_initial; the
     * successor of a state with value g has the value the emit callback / apply_state report. Without total-cost and
     * without a metric every action costs 1 (g' = g + 1); with total-cost g' applies the fired total-cost effects to
     * g; with a metric g' is the metric expression evaluated on the successor. The action cost is g' - g. */

    /* The numeric part of a task: always succeeds (a classical task has slots == 0). */
    int (*task_numeric)(const mymyr_task* task, mymyr_numeric_view* out);
    /* Writes the name "(function o1 ... ok)" of a numeric slot into buf as a NUL-terminated string, truncated to
     * `capacity` bytes. Returns the full length without the NUL, or -1 for a slot the task does not have. */
    int32_t (*numeric_slot_name)(const mymyr_task* task, uint32_t slot, char* buf, uint32_t capacity);
    /* Slot of the name numeric_slot_name returns, or -1 if no slot has it. */
    int64_t (*numeric_slot)(const mymyr_task* task, const char* name);
    /* Writes the value of every numeric slot of the state, in slot order, into out (at most `capacity` of them) and
     * returns the number of slots, or -1 if the state's numeric words do not fit the task. */
    int32_t (*numeric_values)(const mymyr_task* task, const mymyr_state_view* state, double* out, uint32_t capacity);
    /* Metric value g of the initial state `state` (see above). Returns 0, or -1 if the state does not fit the task. */
    int (*metric_initial)(const mymyr_task* task, const mymyr_state_view* state, double* g);
    /* successors for any task: enumerates the applicable actions of a state whose metric value is g, in canonical
     * order, with each successor (words trimmed, numeric words) and its metric value. The views passed to emit are
     * valid during the call only. Returns the number of successors emitted, or -1 as successors does. */
    int64_t (*successors_state)(const mymyr_task* task, const mymyr_state_view* state, double g, mymyr_emit_state_fn emit,
                                void* ctx);
    /* apply for any task: writes the successor's trimmed words into out_words (nothing is written, and nothing in
     * out_numeric either, if they exceed words_capacity: call again with that much room), its numeric words into
     * out_numeric (task_numeric().words of them; may be NULL for a classical task) and its metric value into
     * *out_g (may be NULL). Returns the successor's trimmed width in words, or -1 as apply does. */
    int64_t (*apply_state)(const mymyr_task* task, const mymyr_state_view* state, double g, uint32_t schema,
                           const uint32_t* binding, uint32_t arity, uint64_t* out_words, uint32_t words_capacity,
                           uint64_t* out_numeric, double* out_g);
    /* Hash of a state over its trimmed words and its numeric words (hash_words of the words alone for a classical
     * task): equal for states that state_equal says are equal. */
    uint64_t (*state_hash)(const mymyr_state_view* state);
    /* 1 if both states have equal words (missing words read as zero) and bitwise equal numeric words. */
    int (*state_equal)(const mymyr_state_view* a, const mymyr_state_view* b);
    /* is_goal for any task: the atoms and the numeric goal constraints. */
    int (*is_goal_state)(const mymyr_task* task, const mymyr_state_view* state);
    /* store_insert / store_lookup keyed on atoms and numeric values (store_new covers numeric tasks). */
    int64_t (*store_insert_state)(mymyr_store* store, const mymyr_state_view* state, int* inserted);
    int64_t (*store_lookup_state)(const mymyr_store* store, const mymyr_state_view* state);
    /* Fills *out with state `id` of the store (words padded to the store's width, numeric words), valid until the
     * next insert; task_uid is the store's task. Returns 0, or -1 for an unknown id. */
    int (*store_state_view)(const mymyr_store* store, uint64_t id, mymyr_state_view* out);
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
