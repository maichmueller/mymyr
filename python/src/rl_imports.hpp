#pragma once
// Array imports of the RL bindings (rl_torch_bindings.cpp, rl_ops_bindings.cpp): arrays read and written in place
// through DLPack (destination passing), host memory or CUDA memory on one device, imported with the array API's
// stream handoff on the call's stream; or tensor views (mymyr._typing.TensorView, see tensor_view below), which the
// torch wrappers pass without the cost of a DLPack export.

#include "arrays.hpp"
#include "dlpack.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mymyr::python::rlimp
{
struct DType
{
    u8 code, bits;
};
constexpr DType i32t{dl::k_int, 32}, u32t{dl::k_uint, 32}, i64t{dl::k_int, 64}, u64t{dl::k_uint, 64}, u8t{dl::k_uint, 8},
    boolt{dl::k_bool, 8}, f32t{dl::k_float, 32};

inline std::string quoted(const char* name) { return std::string("mymyr: '") + name + "'"; }

/// The type of the first item of every tensor view (mymyr._core._rl_ops.TensorViewMarker).
struct TensorViewMarker
{
};

/// The first item of every tensor view (mymyr._core._rl_ops.TENSOR_VIEW): the one TensorViewMarker, which the module
/// creates when it is imported, so that no other tuple reads as a view. A borrowed reference: the module attribute
/// owns the marker, so that it is freed with the module at interpreter exit; it is set once, before any view exists.
[[nodiscard]] inline nb::handle& view_marker()
{
    static nb::handle marker;
    return marker;
}

/// The items of a tensor view after its marker.
enum ViewItem : Py_ssize_t
{
    k_view_data = 1,  // data pointer
    k_view_code,      // DLPack dtype code
    k_view_bits,      // DLPack dtype bits
    k_view_device,    // CUDA device, or -1 for host memory
    k_view_shape,     // shape
    k_view_strides,   // strides in elements
    k_view_size
};

/// Whether obj is a tensor view: a tuple (view_marker(), data pointer, DLPack dtype code, bits, CUDA device or -1 for
/// host memory, shape, strides in elements). The caller keeps the memory alive for the call and has ordered its
/// pending writes before the call's stream (mymyr.rl's torch wrappers pass torch's current stream).
[[nodiscard]] inline bool is_tensor_view(nb::handle obj)
{
    return PyTuple_CheckExact(obj.ptr()) && PyTuple_GET_SIZE(obj.ptr()) == k_view_size &&
        view_marker().ptr() && PyTuple_GET_ITEM(obj.ptr(), 0) == view_marker().ptr();
}

/// Item k of a tensor view (is_tensor_view).
[[nodiscard]] inline nb::handle view_item(nb::handle view, ViewItem k) { return PyTuple_GET_ITEM(view.ptr(), k); }

/// The device of a tensor view (is_tensor_view).
[[nodiscard]] inline dl::Device view_device(nb::handle view, const char* name)
{
    int device = -1;
    if (!nb::try_cast(view_item(view, k_view_device), device))
        throw nb::type_error((quoted(name) + " is a tensor view without a device").c_str());
    return device < 0 ? dl::Device{dl::k_cpu, 0} : dl::Device{dl::k_cuda, device};
}

/// The memory of a tensor view (is_tensor_view), as an import with nothing to keep: its fields typed, its extents and
/// strides nonnegative, and its pointer set unless it has no elements.
inline dl::Imported tensor_view(nb::handle obj, const char* name)
{
    const auto bad = [&](const char* what)
    { return nb::type_error((quoted(name) + " is a tensor view " + what).c_str()); };
    dl::Imported im;
    std::uintptr_t data = 0;
    u8 code = 0, bits = 0;
    if (!nb::try_cast(view_item(obj, k_view_data), data) || !nb::try_cast(view_item(obj, k_view_code), code) ||
        !nb::try_cast(view_item(obj, k_view_bits), bits) || !nb::try_cast(view_item(obj, k_view_shape), im.shape) ||
        !nb::try_cast(view_item(obj, k_view_strides), im.strides))
        throw bad("with fields of the wrong types");
    if (bits == 0 || bits % 8)
        throw bad("with a dtype of no whole bytes");
    if (im.strides.size() != im.shape.size())
        throw bad("with mismatched shape and strides");
    u64 elements = 1;
    for (usize k = 0; k < im.shape.size(); ++k)
    {
        if (im.shape[k] < 0 || im.strides[k] < 0)
            throw bad("with a negative extent or stride");
        elements *= static_cast<u64>(im.shape[k]);
    }
    if (!data && elements)
        throw bad("with elements but no memory");
    im.data = reinterpret_cast<void*>(data);
    im.dtype = dl::DataType{code, bits, 1};
    im.device = view_device(obj, name);
    return im;
}

/// The (device type, device id) of an array or tensor view; {-1, -1} for neither.
[[nodiscard]] inline dl::Device device_of(nb::handle obj, const char* name = "array")
{
    return is_tensor_view(obj) ? view_device(obj, name) : dl::dlpack_device(obj);
}

/// The arrays of one call, read and written in place: host memory for a host Env, CUDA memory on the Env's device for
/// a device Env (imported on the call's stream). `keep` holds the imports.
class Imports
{
public:
    /// device: -1 for host arrays, else the CUDA device of every array; `env`: the messages name an Env (else the
    /// call's other arrays).
    Imports(int device, std::intptr_t stream, bool env = true) : m_device(device), m_stream(stream), m_env(env) {}

    /// obj as a C-contiguous array of one of `dtypes` with `shape` (-1: any extent; the extents are stored in `ext`,
    /// its dtype in `dtype`), or null for None. Writable unless `write` is false.
    template<class T>
    T* get(nb::handle obj, const char* name, std::initializer_list<DType> dtypes, std::initializer_list<i64> shape,
           bool write, std::vector<i64>* ext = nullptr, DType* dtype = nullptr)
    {
        if (obj.is_none())
            return nullptr;
        dl::Imported im = take(obj, name, write);
        bool ok = false;
        for (const DType& d : dtypes)
            ok |= im.dtype.code == d.code && im.dtype.bits == d.bits && im.dtype.lanes == 1;
        if (!ok)
            throw nb::type_error((quoted(name) + " has the wrong dtype").c_str());
        check_shape(im, name, shape);
        if (ext)
            *ext = im.shape;
        if (dtype)
            *dtype = DType{im.dtype.code, im.dtype.bits};
        auto* p = static_cast<T*>(im.data);
        keep.push_back(std::move(im.keep));
        return p;
    }

    /// State rows [rows, row_words] of 64-bit words, or [rows, 2 row_words] of 32-bit halves. rows = -1: any
    /// number (stored in *n).
    u64* words(nb::handle obj, const char* name, i64 rows, u32 row_words, bool write, u64* n = nullptr)
    {
        if (obj.is_none())
            return nullptr;
        dl::Imported im = take(obj, name, write);
        const bool is_int = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.lanes == 1;
        if (!is_int || (im.dtype.bits != 64 && im.dtype.bits != 32))
            throw nb::type_error((quoted(name) + " must hold 64-bit state words (or 32-bit halves)").c_str());
        const i64 cols = im.dtype.bits == 64 ? row_words : 2 * static_cast<i64>(row_words);
        check_shape(im, name, {rows, cols});
        if (reinterpret_cast<std::uintptr_t>(im.data) % 8 != 0)
            throw nb::type_error((quoted(name) + " must be 8-byte aligned").c_str());
        if (n)
            *n = static_cast<u64>(im.shape[0]);
        auto* p = static_cast<u64*>(im.data);
        keep.push_back(std::move(im.keep));
        return p;
    }

    /// Word arrays of any width: `ndim` dimensions, the last one 64-bit words (or twice as many 32-bit halves). The
    /// leading extents go to `lead` (-1 in `want`: any extent), the words per row to *words.
    u64* words_nd(nb::handle obj, const char* name, std::initializer_list<i64> want, bool write, std::vector<i64>& lead,
                  u32& words)
    {
        if (obj.is_none())
            return nullptr;
        dl::Imported im = take(obj, name, write);
        const bool is_int = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.lanes == 1;
        if (!is_int || (im.dtype.bits != 64 && im.dtype.bits != 32))
            throw nb::type_error((quoted(name) + " must hold 64-bit words (or 32-bit halves)").c_str());
        std::vector<i64> shape(want);
        shape.push_back(-1);
        if (im.shape.size() != shape.size())
            throw nb::value_error((quoted(name) + " must have " + std::to_string(shape.size()) + " dimensions").c_str());
        for (usize k = 0; k + 1 < shape.size(); ++k)
            if (shape[k] >= 0 && im.shape[k] != shape[k])
                throw nb::value_error((quoted(name) + " has the wrong shape").c_str());
        const i64 cols = im.shape.back();
        if (im.dtype.bits == 32 && cols % 2 != 0)
            throw nb::value_error((quoted(name) + " holds 32-bit halves: it needs an even number of columns").c_str());
        check_contiguous(im, name);
        if (reinterpret_cast<std::uintptr_t>(im.data) % 8 != 0)
            throw nb::type_error((quoted(name) + " must be 8-byte aligned").c_str());
        lead.assign(im.shape.begin(), im.shape.end() - 1);
        words = static_cast<u32>(im.dtype.bits == 64 ? cols : cols / 2);
        auto* p = static_cast<u64*>(im.data);
        keep.push_back(std::move(im.keep));
        return p;
    }

    /// An integer array of `shape` (-1: any extent; the extents go to *ext) as int32 values: read where it is when it
    /// holds int32, else (any other integer width) converted into `copy`; a value outside int32 raises ValueError.
    /// Null for None.
    const i32* ints32(nb::handle obj, const char* name, std::initializer_list<i64> shape, std::vector<i32>& copy,
                      std::vector<i64>* ext = nullptr)
    {
        if (obj.is_none())
            return nullptr;
        dl::Imported im = take(obj, name, false);
        const bool is_int = (im.dtype.code == dl::k_int || im.dtype.code == dl::k_uint) && im.dtype.lanes == 1;
        if (!is_int || (im.dtype.bits != 32 && im.dtype.bits != 64 && im.dtype.bits != 16 && im.dtype.bits != 8))
            throw nb::type_error((quoted(name) + " must hold integers").c_str());
        check_shape(im, name, shape);
        if (ext)
            *ext = im.shape;
        i64 count = 1;
        for (const i64 e : im.shape)
            count *= e;
        if (im.dtype.bits == 32 && im.dtype.code == dl::k_int)
        {
            const auto* p = static_cast<const i32*>(im.data);
            keep.push_back(std::move(im.keep));
            return p;
        }
        copy.resize(static_cast<usize>(count));
        for (i64 k = 0; k < count; ++k)
        {
            i64 v = 0;
            const bool is_signed = im.dtype.code == dl::k_int;
            switch (im.dtype.bits)
            {
                case 8:
                    v = is_signed ? i64{static_cast<const i8*>(im.data)[k]} : i64{static_cast<const u8*>(im.data)[k]};
                    break;
                case 16:
                    v = is_signed ? i64{static_cast<const i16*>(im.data)[k]} : i64{static_cast<const u16*>(im.data)[k]};
                    break;
                case 32: v = i64{static_cast<const u32*>(im.data)[k]}; break;
                default:
                {
                    const u64 raw = static_cast<const u64*>(im.data)[k];
                    constexpr u64 largest = static_cast<u64>(std::numeric_limits<i64>::max());
                    v = is_signed ? static_cast<i64>(raw) : static_cast<i64>(std::min(raw, largest));
                }
            }
            if (v < std::numeric_limits<i32>::lowest() || v > std::numeric_limits<i32>::max())
                throw nb::value_error((quoted(name) + " has a value that int32 cannot hold").c_str());
            copy[static_cast<usize>(k)] = static_cast<i32>(v);
        }
        return copy.data();
    }

    std::vector<std::shared_ptr<void>> keep;

private:
    dl::Imported take(nb::handle obj, const char* name, bool write)
    {
        const bool view = is_tensor_view(obj);
        if (write && !view && framework_of(obj) == Framework::Jax)
            throw nb::type_error((quoted(name) + " is a JAX array; JAX arrays are immutable").c_str());
        const dl::Device d = device_of(obj, name);
        if (d.device_type < 0)
            throw nb::type_error((quoted(name) + " must be an array that speaks DLPack").c_str());
        if (m_device < 0 && !dl::host_accessible(d.device_type))
            throw nb::type_error((quoted(name) + (m_env ? " must be a host array: this Env runs on the CPU (device=None)"
                                                        : " must be a host array, as the call's other arrays"))
                                     .c_str());
        if (m_device >= 0 && (d.device_type != dl::k_cuda || d.device_id != m_device))
            throw nb::type_error((quoted(name) + " must be a CUDA array on cuda:" + std::to_string(m_device) +
                                  (m_env ? " (the Env's device)" : " (the device of the call's other arrays)"))
                                     .c_str());
        if (view)
            return tensor_view(obj, name);
        dl::Imported im = dl::import_dlpack(obj, m_device < 0 ? dl::k_stream_default : m_stream);
        if (write && im.readonly)
            throw nb::type_error((quoted(name) + " is read-only").c_str());
        return im;
    }

    static void check_shape(const dl::Imported& im, const char* name, std::initializer_list<i64> shape)
    {
        bool ok = im.shape.size() == shape.size();
        for (usize k = 0; ok && k < shape.size(); ++k)
            ok = shape.begin()[k] < 0 || im.shape[k] == shape.begin()[k];
        if (!ok)
        {
            std::string want = "[";
            for (usize k = 0; k < shape.size(); ++k)
                want += (k ? ", " : "") + (shape.begin()[k] < 0 ? std::string("*") : std::to_string(shape.begin()[k]));
            throw nb::value_error((quoted(name) + " must have shape " + want + "]").c_str());
        }
        check_contiguous(im, name);
    }

    static void check_contiguous(const dl::Imported& im, const char* name)
    {
        if (std::ranges::find(im.shape, i64{0}) != im.shape.end())
            return;  // an empty array has no layout
        i64 expect = 1;
        for (usize k = im.shape.size(); k-- > 0;)
        {
            if (im.shape[k] > 1 && im.strides[k] != expect)
                throw nb::type_error((quoted(name) + " must be C-contiguous").c_str());
            expect *= im.shape[k];
        }
    }

    int m_device;
    std::intptr_t m_stream;
    bool m_env;
};

}  // namespace mymyr::python::rlimp
