// mymyr._core: the nanobind extension.
//   - version, build info, hash_rows (M0);
//   - _formalism: read-only views of the normalized task (formalism_bindings.cpp);
//   - Domain: the loki front end (frontend_bindings.cpp, when built);
//   - Task, TaskHandle, State, Action, Atom (task_bindings.cpp), DLArray (arrays.cpp);
//   - _rl: batched expand and friends (rl_bindings.cpp);
//   - _rl_torch: batched planning environments and the counter-based RNG (rl_torch_bindings.cpp; mymyr.rl.torch);
//   - _rl_jax: the XLA FFI targets of the planning environments (rl_jax_bindings.cpp; mymyr.rl.jax);
//   - _encoding_jax: the XLA FFI target of the relational encoder (encoding_jax_bindings.cpp; mymyr.encoding.jax);
//   - _rl_ops: novelty rewards, prefix masks and hindsight relabels (rl_ops_bindings.cpp; mymyr.rl);
//   - _search: the search family and heuristics (search_bindings.cpp);
//   - _datasets: state spaces, generalized state spaces, samplers, tuple graphs, knowledge bases
//     (datasets_bindings.cpp);
//   - _C_API: the capsule for downstream native modules (ext_api.cpp);
//   - _cuda: the CUDA backend (cuda_bindings.cpp, CUDA builds only; mymyr.cuda), with the device state spaces
//     (cuda_datasets_bindings.cpp).

#include "arrays.hpp"
#include "device_hooks.hpp"
#include "py_task.hpp"
#include "typing.hpp"

#include "mymyr/core/block_cache.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/version.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace mymyr::python
{
void bind_formalism(nb::module_& parent);
#if defined(MYMYR_HAS_FRONTEND)
void bind_frontend(nb::module_& m);
#endif
void bind_ext_api(nb::module_& m);
void bind_search(nb::module_& m);
void bind_datasets(nb::module_& parent);
void bind_rl_torch(nb::module_& parent);
void bind_rl_jax(nb::module_& parent);
void bind_rl_ops(nb::module_& parent);
void bind_encoding(nb::module_& parent);
void bind_encoding_jax(nb::module_& parent);
#if defined(MYMYR_HAS_CUDA)
void bind_cuda(nb::module_& parent);
void bind_cuda_datasets(nb::module_& parent);
void bind_encoding_device(nb::module_& parent);
#endif
}  // namespace mymyr::python

namespace
{
mymyr::python::Arg<mymyr::python::ann::Any> hash_rows(mymyr::python::Arg<mymyr::python::ann::ArrayLike> states, mymyr::u32 numeric_words, mymyr::python::StreamArg stream)
{
    using namespace mymyr;
    using namespace mymyr::python;
    if (device_hooks().hash_rows && is_cuda_array(states))
        return device_hooks().hash_rows(states, numeric_words, stream);
    if (!stream.is_none())
        throw nb::type_error("mymyr: hash_rows: stream requires device rows");
    const StateBatch in = import_states(states, 0);
    if (in.single)
        throw nb::type_error("mymyr: hash_rows: states must be a matrix");
    if (numeric_words > in.view.words)
        throw nb::value_error("mymyr: hash_rows: numeric_words exceeds the row width");
    const u32 W = in.view.words - numeric_words;
    auto block = Block::make(in.view.rows * sizeof(u64));
    auto* out = reinterpret_cast<u64*>(block->data());
    {
        nb::gil_scoped_release release;
        for (u64 i = 0; i < in.view.rows; ++i)
        {
            const u64* row = in.view.row(i);
            out[i] = hash::state(row, W, numeric_words ? row + W : nullptr, numeric_words);
        }
    }
    ArraySpec spec{block, out, in.fw == Framework::Torch ? rl::DType::I64 : rl::DType::U64, {static_cast<i64>(in.view.rows)}, {}, false, false};
    if (in.fw == Framework::Jax)
    {
        spec.shape.push_back(1);
        spec.words = true;
    }
    return export_array(std::move(spec), in.fw, in.enc);
}

#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E0000
/// Free-threaded CPython adjusts the shared reference count of a descriptor or function object on every attribute
/// lookup unless the object uses deferred reference counting; nanobind's function objects and the properties it
/// creates do not. That refcount is then contended across threads reading the same property, so the module's
/// functions, and the methods and properties of its classes, are switched to deferred reference counting. They live
/// as long as the module, so leaving their collection to the GC costs nothing.
void defer(nb::handle v)
{
    PyUnstable_Object_EnableDeferredRefcount(v.ptr());  // a no-op for objects outside the GC
    if (PyObject_TypeCheck(v.ptr(), &PyProperty_Type))
        for (const char* part : {"fget", "fset"})
            PyUnstable_Object_EnableDeferredRefcount(nb::getattr(v, part).ptr());
}

void defer_refcounts(nb::handle module)
{
    const std::string prefix = nb::cast<std::string>(nb::getattr(module, "__name__")) + ".";
    for (nb::handle v : nb::borrow<nb::dict>(PyModule_GetDict(module.ptr())).values())
    {
        if (nb::isinstance<nb::module_>(v))
        {
            if (nb::cast<std::string>(nb::getattr(v, "__name__")).starts_with(prefix))  // _formalism, _rl
                defer_refcounts(v);
            continue;
        }
        defer(v);
        if (PyType_Check(v.ptr()))
            for (nb::handle d : nb::getattr(v, "__dict__").attr("values")())
                defer(d);
    }
}
#else
void defer_refcounts(nb::handle) {}
#endif
}  // namespace

NB_MODULE(_core, m)
{
    m.doc() = "mymyr native core";
    m.attr("__version__") = std::string(mymyr::version());
    m.def("build_info", [] { return std::string(mymyr::build_info()); },
          "How the extension was built: 'mymyr <version> (<build type>, <compiler>)'.");
    m.def("free_threaded_build", [] {
#ifdef Py_GIL_DISABLED
        return true;
#else
        return false;
#endif
    },
          "Whether the extension was built for free-threaded CPython (Py_GIL_DISABLED).");
    m.def("trim_block_cache", [] { mymyr::trim_block_cache(); }, nb::call_guard<nb::gil_scoped_release>(),
          "Returns the memory of recycled output blocks to the operating system.\n\n"
          "Batch outputs (encodings, expansions, table batches) of more than 128 KiB are served from a cache of "
          "recycled blocks, so that repeated calls do not pay for fresh memory pages. The cache holds at most 4 blocks "
          "of up to 4 MiB per thread and, shared by all threads, at most 8 blocks and 512 MiB in all (no block above "
          "256 MiB); a shared block that nothing takes within 256 later acquires of such blocks is released by itself. "
          "This frees what the calling thread's cache and the shared cache hold; the caches of other threads are "
          "freed when those threads end, and arrays that still use a block keep it.");
    m.def("block_cache_bytes", [] { return mymyr::block_cache_bytes(); },
          "The bytes that the calling thread's block cache and the shared block cache hold (see trim_block_cache).");
    m.def("hash_rows", &hash_rows, "states"_a, nb::kw_only(), "numeric_words"_a = 0, "stream"_a = nb::none(),
          "Hash each row of a (N, W) uint64 state-word matrix (trailing zero words ignored). Numeric words follow atom words and seed the hash. Same framework output.");
    mymyr::python::bind_formalism(m);
#if defined(MYMYR_HAS_FRONTEND)
    mymyr::python::bind_frontend(m);
#endif
    mymyr::python::bind_arrays(m);
    mymyr::python::bind_task(m);
    mymyr::python::bind_rl(m);
    mymyr::python::bind_rl_torch(m);
    mymyr::python::bind_rl_jax(m);
    mymyr::python::bind_rl_ops(m);
    mymyr::python::bind_encoding(m);
    mymyr::python::bind_encoding_jax(m);
    mymyr::python::bind_search(m);
    mymyr::python::bind_datasets(m);
    mymyr::python::bind_ext_api(m);
#if defined(MYMYR_HAS_CUDA)
    mymyr::python::bind_cuda(m);
    mymyr::python::bind_cuda_datasets(m);
    mymyr::python::bind_encoding_device(m);
#endif
    defer_refcounts(m);
}
