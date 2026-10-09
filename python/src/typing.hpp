#pragma once
// Parameter and result types for signatures and the stubs. nb::typed<nb::object, T> accepts or holds
// any object (the bindings check their arguments themselves, with their own messages) and renders as T; nanobind
// appends " | None" to a None default. The tags in `ann` name Python types that have no C++ counterpart: only their
// names exist (they are never converted).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <cstdint>
#include <variant>

namespace mymyr::python
{
namespace nb = nanobind;

template<class T>
using Arg = nb::typed<nb::object, T>;

namespace ann
{
/// An array with __dlpack__ (mymyr._typing.SupportsDLPack): NumPy, torch and JAX arrays, mymyr.DLArray. Not
/// numpy.typing.ArrayLike: nested sequences are no arrays here, and its sequence members would hide the other members
/// of a union from a type checker's inference (a list literal of mixed atoms would be list[object]).
struct ArrayLike
{
};
/// typing.Any: arrays of the requested framework (numpy.ndarray, torch.Tensor, jax.Array or mymyr.DLArray), so that the
/// stubs need none of the frameworks, and the values of dicts that mix kinds.
struct Any
{
};
/// A framework name (parse_framework also accepts "np", "pytorch" and "array_api").
struct Framework
{
};
/// A stream object with an int `cuda_stream`, its cudaStream_t (mymyr._typing.SupportsCudaStream): torch.cuda.Stream.
struct CudaStream
{
};
}  // namespace ann

/// A CUDA stream: an int cudaStream_t (1 / 2: the legacy / per-thread default stream) or a stream object. With a None
/// default, None is the stream of the operation's context.
using StreamArg = Arg<std::variant<std::intptr_t, ann::CudaStream>>;
}  // namespace mymyr::python

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::ArrayLike>
{
    static constexpr auto Name = const_name("mymyr._typing.SupportsDLPack");
};
template<>
struct type_caster<mymyr::python::ann::Any>
{
    static constexpr auto Name = const_name("typing.Any");
};
template<>
struct type_caster<mymyr::python::ann::Framework>
{
    static constexpr auto Name = const_name("typing.Literal['numpy', 'torch', 'jax', 'dlpack']");
};
template<>
struct type_caster<mymyr::python::ann::CudaStream>
{
    static constexpr auto Name = const_name("mymyr._typing.SupportsCudaStream");
};
}  // namespace nanobind::detail
