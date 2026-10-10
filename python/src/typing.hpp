#pragma once
// Parameter and result types for signatures and the stubs. nb::typed<nb::object, T> accepts or holds
// any object (the bindings check their arguments themselves, with their own messages) and renders as T; nanobind
// appends " | None" to a None default. The tags in `ann` name Python types that have no C++ counterpart: only their
// names exist (they are never converted).

#include "mymyr/core/threads.hpp"
#include "mymyr/successor/symmetry.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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
/// A symmetry pruning mode: 'off' or 'wl1' (successor/symmetry.hpp).
struct SymmetryPruning
{
};
}  // namespace ann

/// A CUDA stream: an int cudaStream_t (1 / 2: the legacy / per-thread default stream) or a stream object. With a None
/// default, None is the stream of the operation's context.
using StreamArg = Arg<std::variant<std::intptr_t, ann::CudaStream>>;

using SymmetryArg = Arg<ann::SymmetryPruning>;

// ------------------------------------------------------------------------------------------------ numeric arguments
// Integer and float parameters are declared as Arg<T> (any object, rendered as int / float) and read with int_arg /
// float_arg, so that a value out of range raises ValueError naming the parameter and its range (not nanobind's
// overload error or a failed cast). A non-number raises TypeError.

/// An int parameter (rendered as int), read with int_arg / opt_int_arg / threads_arg.
using IntArg = Arg<std::uint64_t>;
/// A float parameter (rendered as float), read with float_arg / opt_float_arg.
using FloatArg = Arg<double>;

namespace arg_detail
{
/// The range [lo, hi] as text. Only the 64-bit limits go unnamed (a narrower type's limit is part of the range a
/// caller has to know), unless the value was beyond that limit (name_lo / name_hi).
template<class T>
std::string range_text(T lo, T hi, bool name_lo = false, bool name_hi = false)
{
    const bool lo_open = !name_lo && std::is_signed_v<T> && sizeof(T) == 8 && lo == std::numeric_limits<T>::lowest();
    const bool hi_open = !name_hi && sizeof(T) == 8 && hi == std::numeric_limits<T>::max();
    if (lo_open && hi_open)
        return "in the 64-bit range";
    if (hi_open)
        return ">= " + std::to_string(lo);
    if (lo_open)
        return "<= " + std::to_string(hi);
    return "in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
}

/// "mymyr: NAME must be [None or ]KIND RANGE, got REPR" as a ValueError.
[[noreturn]] inline void range_error(const char* name, const char* kind, const std::string& range, bool none, nb::handle got)
{
    const std::string text = nb::cast<std::string>(nb::repr(got));
    throw nb::value_error(("mymyr: " + std::string(name) + " must be " + (none ? "None or " : "") + kind + " " + range +
                           ", got " + text)
                              .c_str());
}

template<std::integral T>
T int_value(nb::handle h, const char* name, T lo, T hi, bool none)
{
    if (PyFloat_Check(h.ptr()) || !PyIndex_Check(h.ptr()))
        throw nb::type_error(("mymyr: " + std::string(name) + " must be an int, not " +
                              nb::cast<std::string>(nb::str(h.type().attr("__name__"))))
                                 .c_str());
    const nb::object i = nb::steal(PyNumber_Index(h.ptr()));
    if (!i.is_valid())
        throw nb::python_error();
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
    if (v == -1 && PyErr_Occurred())
        throw nb::python_error();
    bool ok = false;
    T out{};
    if (overflow == 0)
    {
        ok = std::in_range<T>(v) && std::cmp_greater_equal(v, lo) && std::cmp_less_equal(v, hi);
        out = static_cast<T>(v);
    }
    else if (overflow > 0 && std::is_unsigned_v<T>)
    {
        const unsigned long long u = PyLong_AsUnsignedLongLong(i.ptr());
        if (u == static_cast<unsigned long long>(-1) && PyErr_Occurred())
            PyErr_Clear();  // above 2**64 - 1: out of range
        else
        {
            ok = std::in_range<T>(u) && std::cmp_greater_equal(u, lo) && std::cmp_less_equal(u, hi);
            out = static_cast<T>(u);
        }
    }
    if (!ok)
    {
        const bool below = overflow < 0 || (overflow == 0 && std::cmp_less(v, lo));
        const bool above = overflow > 0 || (overflow == 0 && std::cmp_greater(v, hi));
        range_error(name, "an int", range_text(lo, hi, below, above), none, h);
    }
    return out;
}
}  // namespace arg_detail

/// The int argument `name` in [lo, hi].
template<std::integral T>
T int_arg(nb::handle h, const char* name, T lo = std::numeric_limits<T>::lowest(), T hi = std::numeric_limits<T>::max())
{
    return arg_detail::int_value<T>(h, name, lo, hi, false);
}

/// The int argument `name` in [lo, hi], or nullopt for None.
template<std::integral T>
std::optional<T> opt_int_arg(nb::handle h, const char* name, T lo = std::numeric_limits<T>::lowest(),
                             T hi = std::numeric_limits<T>::max())
{
    if (h.is_none())
        return std::nullopt;
    return arg_detail::int_value<T>(h, name, lo, hi, true);
}

/// The ints of the sequence argument `name`, each in [lo, hi].
template<std::integral T>
std::vector<T> int_list_arg(nb::handle h, const char* name, T lo = std::numeric_limits<T>::lowest(),
                            T hi = std::numeric_limits<T>::max())
{
    if (!PySequence_Check(h.ptr()) || nb::isinstance<nb::str>(h))
        throw nb::type_error(("mymyr: " + std::string(name) + " must be a sequence of ints").c_str());
    std::vector<T> out;
    for (nb::handle x : h)
        out.push_back(arg_detail::int_value<T>(x, name, lo, hi, false));
    return out;
}

/// A thread count `name` in [0, max_threads()] (core/threads.hpp; 0: one per hardware thread).
inline u32 threads_arg(nb::handle h, const char* name = "threads")
{
    return int_arg<u32>(h, name, 0, max_threads());
}

/// The float argument `name` (an int or a float) in [lo, hi]; NaN is out of every range.
inline double float_arg(nb::handle h, const char* name, double lo = -std::numeric_limits<double>::infinity(),
                        double hi = std::numeric_limits<double>::infinity(), bool none = false)
{
    auto not_a_number = [&] {
        throw nb::type_error(("mymyr: " + std::string(name) + " must be a number, not " +
                              nb::cast<std::string>(nb::str(h.type().attr("__name__"))))
                                 .c_str());
    };
    if (PyUnicode_Check(h.ptr()) || PyBytes_Check(h.ptr()))
        not_a_number();  // no float("1") parsing
    const double v = PyFloat_AsDouble(h.ptr());  // __float__, then __index__
    if (v == -1.0 && PyErr_Occurred())
    {
        if (!PyErr_ExceptionMatches(PyExc_TypeError))
            throw nb::python_error();
        PyErr_Clear();
        not_a_number();
    }
    if (!(v >= lo && v <= hi))
    {
        std::string range;
        if (std::isinf(lo) && std::isinf(hi))
            range = "other than NaN";
        else if (std::isinf(hi))
            range += ">= " + nb::cast<std::string>(nb::repr(nb::float_(lo)));
        else if (std::isinf(lo))
            range += "<= " + nb::cast<std::string>(nb::repr(nb::float_(hi)));
        else
            range += "in [" + nb::cast<std::string>(nb::repr(nb::float_(lo))) + ", " +
                     nb::cast<std::string>(nb::repr(nb::float_(hi))) + "]";
        arg_detail::range_error(name, "a number", range, none, h);
    }
    return v;
}

/// The float argument `name` in [lo, hi], or nullopt for None.
inline std::optional<double> opt_float_arg(nb::handle h, const char* name,
                                           double lo = -std::numeric_limits<double>::infinity(),
                                           double hi = std::numeric_limits<double>::infinity())
{
    if (h.is_none())
        return std::nullopt;
    return float_arg(h, name, lo, hi, true);
}

/// The mode of a symmetry_pruning= argument; ValueError unless it is 'off' or 'wl1'.
inline mymyr::SymmetryPruning parse_symmetry_pruning(nb::handle h)
{
    if (nb::isinstance<nb::str>(h))
        if (const auto p = mymyr::parse_symmetry_pruning(nb::cast<std::string>(h)))
            return *p;
    throw nb::value_error("mymyr: symmetry_pruning must be 'off' or 'wl1'");
}
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
template<>
struct type_caster<mymyr::python::ann::SymmetryPruning>
{
    static constexpr auto Name = const_name("typing.Literal['off', 'wl1']");
};
}  // namespace nanobind::detail
