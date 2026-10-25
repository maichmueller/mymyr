#pragma once
// Objects as mappings of names to host arrays (StateSpace and FactLandmarkGraph arrays, from_arrays, pickling,
// datasets.save / load).
//
// Out: fresh() exports a vector it takes over as a NumPy array. In: take() copies an array of a mapping (NumPy, or a
// DLPack array on the CPU) after checking its dtype and number of dimensions; the C++ object built from the copies
// validates their contents (std::invalid_argument: ValueError).

#include "arrays.hpp"

#include "mymyr/core/types.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mymyr::python
{
/// The array dtype of T (u8: bool flags).
template<class T>
constexpr rl::DType dtype_of()
{
    if constexpr (std::is_same_v<T, u8>)
        return rl::DType::Bool;
    else if constexpr (std::is_same_v<T, i32>)
        return rl::DType::I32;
    else if constexpr (std::is_same_v<T, u32>)
        return rl::DType::U32;
    else if constexpr (std::is_same_v<T, i64>)
        return rl::DType::I64;
    else if constexpr (std::is_same_v<T, f64>)
        return rl::DType::F64;
    else
    {
        static_assert(std::is_same_v<T, u64>);
        return rl::DType::U64;
    }
}

/// The NumPy name of the dtype of T.
template<class T>
constexpr const char* dtype_name()
{
    if constexpr (std::is_same_v<T, u8>)
        return "bool";
    else if constexpr (std::is_same_v<T, i32>)
        return "int32";
    else if constexpr (std::is_same_v<T, u32>)
        return "uint32";
    else if constexpr (std::is_same_v<T, i64>)
        return "int64";
    else if constexpr (std::is_same_v<T, f64>)
        return "float64";
    else
        return "uint64";
}

/// A writable NumPy array owning `v` (shape: default 1-d).
template<class T>
nb::object fresh(std::vector<T> v, std::vector<i64> shape = {})
{
    auto owner = std::make_shared<std::vector<T>>(std::move(v));
    if (owner->empty())
        owner->reserve(1);  // a valid data pointer for an empty array
    ArraySpec spec;
    spec.data = owner->data();
    spec.dtype = dtype_of<T>();
    spec.shape = shape.empty() ? std::vector<i64>{static_cast<i64>(owner->size())} : std::move(shape);
    spec.owner = std::move(owner);
    return export_array(std::move(spec), Framework::Numpy, default_words(Framework::Numpy));
}

/// Item `key` of a mapping, or None if it has none.
inline nb::object mapping_item(nb::handle mapping, const char* key)
{
    PyObject* out = nullptr;
    const int r = PyMapping_GetOptionalItemString(mapping.ptr(), key, &out);
    if (r < 0)
        throw nb::python_error();
    return r == 0 ? nb::none() : nb::steal(out);
}

/// TypeError unless `arrays` is a mapping whose keys are among `known`.
inline void check_keys(nb::handle arrays, std::initializer_list<std::string_view> known, const char* what)
{
    if (!nb::isinstance<nb::dict>(arrays) && (!PyMapping_Check(arrays.ptr()) || !nb::hasattr(arrays, "keys")))
        throw nb::type_error((std::string("mymyr: ") + what + " arrays must be a mapping of names to arrays (as arrays() gives them)").c_str());
    nb::object keys = nb::steal(PyMapping_Keys(arrays.ptr()));
    if (!keys.is_valid())
        throw nb::python_error();
    for (nb::handle k : keys)
    {
        if (!nb::isinstance<nb::str>(k))
            throw nb::type_error((std::string("mymyr: ") + what + " array names must be strings").c_str());
        const std::string_view name = nb::cast<std::string_view>(k);
        if (std::ranges::find(known, name) == known.end())
            throw nb::value_error((std::string("mymyr: unknown ") + what + " array '" + std::string(name) + "'").c_str());
    }
}

/// An array in host memory (NumPy, or a DLPack array on the CPU), not converted.
inline nb::ndarray<nb::ro> host_array(nb::handle obj, const std::string& name)
{
    nb::ndarray<nb::ro> a;
    if (!nb::try_cast(obj, a, false) || a.device_type() != nb::device::cpu::value)
        throw nb::type_error(("mymyr: " + name + " must be an array in host memory (NumPy, or a DLPack array on the CPU)").c_str());
    return a;
}

/// A copy of array `key` of `arrays` (`ndim` dimensions of T; flags: bool or uint8), and its shape. Absent: nullopt
/// unless `required` (then ValueError).
template<class T>
std::optional<std::pair<std::vector<T>, std::vector<u64>>> take(nb::handle arrays, const char* key, usize ndim, bool required,
                                                                const char* what)
{
    nb::object obj = mapping_item(arrays, key);
    const std::string name = std::string(what) + " array '" + key + "'";
    if (obj.is_none())
    {
        if (required)
            throw nb::value_error(("mymyr: the " + name + " is missing").c_str());
        return std::nullopt;
    }
    const nb::ndarray<nb::ro> a = host_array(obj, name);
    const bool dtype_ok = a.dtype() == nb::dtype<T>() || (std::is_same_v<T, u8> && a.dtype() == nb::dtype<bool>());
    if (!dtype_ok || a.ndim() != ndim)
        throw nb::type_error(("mymyr: the " + name + " must be a " + std::to_string(ndim) + "-d " + dtype_name<T>() + " array").c_str());
    std::vector<u64> shape(ndim);
    u64 n = 1;
    for (usize i = 0; i < ndim; ++i)
        n *= shape[i] = a.shape(i);
    std::vector<T> v(n);
    const auto* base = static_cast<const unsigned char*>(a.data());
    const i64 s0 = a.stride(0), s1 = ndim == 2 ? a.stride(1) : 1;
    const u64 cols = ndim == 2 ? shape[1] : 1, rows = shape[0];
    if (n > 0 && (ndim == 1 ? s0 == 1 : (s1 == 1 && s0 == static_cast<i64>(cols))))
        std::memcpy(v.data(), base, n * sizeof(T));
    else
        for (u64 r = 0; r < rows; ++r)
            for (u64 c = 0; c < cols; ++c)
            {
                const i64 off = ndim == 2 ? static_cast<i64>(r) * s0 + static_cast<i64>(c) * s1 : static_cast<i64>(r) * s0;
                std::memcpy(&v[r * cols + c], base + off * static_cast<i64>(sizeof(T)), sizeof(T));
            }
    return std::pair{std::move(v), std::move(shape)};
}

template<class T>
std::vector<T> take_required(nb::handle arrays, const char* key, usize ndim, const char* what)
{
    return std::move(take<T>(arrays, key, ndim, true, what)->first);
}
}  // namespace mymyr::python
