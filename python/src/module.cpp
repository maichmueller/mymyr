// mymyr._core: the nanobind extension.
//   - version, build info, hash_rows (M0);
//   - _formalism: read-only views of the normalized task (formalism_bindings.cpp);
//   - Domain: the loki front end (frontend_bindings.cpp, when built);
//   - Task, TaskHandle, State, Action, Atom (task_bindings.cpp), DLArray (arrays.cpp);
//   - _rl: batched expand and friends (rl_bindings.cpp);
//   - _rl_torch: batched planning environments and the counter-based RNG (rl_torch_bindings.cpp; mymyr.rl.torch);
//   - _rl_jax: the XLA FFI targets of the planning environments (rl_jax_bindings.cpp; mymyr.rl.jax);
//   - _rl_ops: novelty rewards, prefix masks and hindsight relabels (rl_ops_bindings.cpp; mymyr.rl);
//   - _search: the search family and heuristics (search_bindings.cpp);
//   - _datasets: state spaces, samplers, object graphs, certificates (datasets_bindings.cpp);
//   - _C_API: the capsule for downstream native modules (ext_api.cpp);
//   - _cuda: the CUDA backend (cuda_bindings.cpp, CUDA builds only; mymyr.cuda), with the device state spaces
//     (cuda_datasets_bindings.cpp).

#include "arrays.hpp"
#include "py_task.hpp"

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
#if defined(MYMYR_HAS_CUDA)
void bind_cuda(nb::module_& parent);
void bind_cuda_datasets(nb::module_& parent);
#endif
}  // namespace mymyr::python

namespace
{
using WordMatrix = nb::ndarray<const mymyr::u64, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using HashVector = nb::ndarray<nb::numpy, mymyr::u64, nb::ndim<1>>;

HashVector hash_rows(WordMatrix states)
{
    const size_t n = states.shape(0);
    const auto w = static_cast<mymyr::u32>(states.shape(1));
    auto* out = new mymyr::u64[n];
    {
        nb::gil_scoped_release release;  // no-op on free-threaded builds, required on GIL builds
        const mymyr::u64* data = states.data();
        for (size_t i = 0; i < n; ++i)
            out[i] = mymyr::hash::state_words(data + i * w, w);
    }
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<mymyr::u64*>(p); });
    return HashVector(out, {n}, owner);
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
    m.def("build_info", [] { return std::string(mymyr::build_info()); });
    m.def("free_threaded_build", [] {
#ifdef Py_GIL_DISABLED
        return true;
#else
        return false;
#endif
    });
    m.def("hash_rows", &hash_rows, "states"_a,
          "Hash each row of a (N, W) uint64 state-word matrix (trailing zero words ignored). Zero-copy input.");
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
    mymyr::python::bind_search(m);
    mymyr::python::bind_datasets(m);
    mymyr::python::bind_ext_api(m);
#if defined(MYMYR_HAS_CUDA)
    mymyr::python::bind_cuda(m);
    mymyr::python::bind_cuda_datasets(m);
#endif
    defer_refcounts(m);
}
