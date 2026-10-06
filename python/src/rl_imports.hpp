#pragma once
// Array imports of the RL bindings (rl_torch_bindings.cpp, rl_ops_bindings.cpp): arrays read and written in place
// through DLPack (destination passing), host memory or CUDA memory on one device, imported with the array API's
// stream handoff on the call's stream.

#include "arrays.hpp"
#include "dlpack.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <initializer_list>
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

/// The arrays of one call, read and written in place: host memory for a host Env, CUDA memory on the Env's device for
/// a device Env (imported on the call's stream). `keep` holds the imports.
class Imports
{
public:
    /// device: -1 for host arrays, else the CUDA device of every array; `env`: the messages name an Env (else the
    /// call's other arrays).
    Imports(int device, std::intptr_t stream, bool env = true) : m_device(device), m_stream(stream), m_env(env) {}

    /// obj as a C-contiguous array of one of `dtypes` with `shape` (-1: any extent; the extents are stored in `ext`),
    /// or null for None. Writable unless `write` is false.
    template<class T>
    T* get(nb::handle obj, const char* name, std::initializer_list<DType> dtypes, std::initializer_list<i64> shape,
           bool write, std::vector<i64>* ext = nullptr)
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

    std::vector<std::shared_ptr<void>> keep;

private:
    dl::Imported take(nb::handle obj, const char* name, bool write)
    {
        if (write && framework_of(obj) == Framework::Jax)
            throw nb::type_error((quoted(name) + " is a JAX array; JAX arrays are immutable").c_str());
        const dl::Device d = dl::dlpack_device(obj);
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
