#pragma once
// Zero-copy array interop for every bulk entry point.
//
// Inputs: any NumPy, torch or JAX CPU array (or anything else speaking DLPack or the buffer protocol) is taken as it
// is, without a copy. State batches may be u64/i64 [N, W] or u32/i32 [N, 2W] (the same little-endian bytes), with
// any row stride.
//
// Outputs: memory owned by mymyr (a shared, 64-byte aligned block per result) is exported as
//   - NumPy: an ndarray view (nanobind's buffer export);
//   - torch / JAX / "dlpack": our own DLArray, whose __dlpack__ / __dlpack_device__ hand out DLPack 1.x (versioned) or
//     legacy capsules with the offset folded into the data pointer (torch rejects a byte_offset) and the array-API
//     protocol that jax.dlpack.from_dlpack requires; torch.from_dlpack / jax.dlpack.from_dlpack then wrap it zero-copy.
// Word arrays take the caller's word encoding: uint64 (NumPy), int64 (torch), uint32 pairs [.., 2W] (JAX, no x64).
// Other 64-bit arrays (the u64 tables of device_arrays version 2) reach JAX as uint32 pairs too ([.., k] -> [.., 2k],
// low half first), since JAX without x64 would narrow them by a truncating copy.
//
// Device arrays: a DLArray may hold CUDA memory (device, pinned host or managed), exported by the CUDA bindings
// (cuda_bindings.cpp, CUDA builds only). Its __dlpack__(stream=s) follows the array API: the producer's pending writes
// are made visible on the consumer's stream s through an event (s = None or 1: the legacy default stream, 2: the
// per-thread default stream, -1: no synchronization, 0: rejected), and s is recorded as a user of the memory, so the
// memory is freed only after the consumer's work enqueued before the release (DeviceSync).
// Inputs of host operations (rl.expand, encode, goal tests, ...): pinned-host and managed CUDA memory is read in place
// after the producer's stream handoff (CUDA builds); device memory raises a TypeError, since these operations run on
// the CPU until the device expand.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_arrays.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <string>
#include <vector>

namespace mymyr::python
{
namespace nb = nanobind;

enum class Framework : u8
{
    Numpy,
    Torch,
    Jax,
    DLPack,  // our DLArray itself (array-API consumers)
};

/// How state words are laid out for a framework: 64-bit words (signed for torch) or 32-bit halves (JAX).
struct WordEncoding
{
    u8 bits = 64;
    bool is_signed = false;
};

[[nodiscard]] Framework framework_of(nb::handle obj);
/// `name` None: taken from `like` (None: NumPy). Otherwise "numpy", "torch", "jax" or "dlpack".
[[nodiscard]] Framework parse_framework(nb::handle name, nb::handle like = nb::none());
[[nodiscard]] WordEncoding default_words(Framework fw);
[[nodiscard]] const char* framework_name(Framework fw);

/// A shared, 64-byte aligned block of bytes. The contents start unspecified: the memory is recycled through a small
/// per-thread cache (fresh memory costs a kernel page fault per 4 KB page), so writers fill everything they export.
class Block
{
public:
    static std::shared_ptr<Block> make(u64 bytes);
    Block() = default;
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    ~Block();
    [[nodiscard]] std::byte* data() noexcept { return m_base; }
    [[nodiscard]] u64 size() const noexcept { return m_size; }

private:
    std::unique_ptr<std::byte[]> m_raw;
    u64 m_capacity = 0;  // bytes in m_raw
    std::byte* m_base = nullptr;
    u64 m_size = 0;
};

/// Byte layout of several arrays in one Block (each 64-byte aligned).
struct BlockLayout
{
    u64 bytes = 0;
    u64 add(u64 n)
    {
        const u64 at = bytes;
        bytes += (n + 63) & ~u64{63};
        return at;
    }
};

/// One array to export: memory kept alive by `owner`.
struct ArraySpec
{
    std::shared_ptr<const void> owner;
    const void* data = nullptr;
    rl::DType dtype = rl::DType::I32;
    std::vector<i64> shape;
    std::vector<i64> strides;  // in elements; empty = C-contiguous
    bool readonly = false;
    bool words = false;  // last dimension holds u64 state words: re-encoded per WordEncoding
};

/// Exports an array to a framework (zero-copy). Word arrays are re-encoded (a dtype/shape reinterpretation only).
[[nodiscard]] nb::object export_array(ArraySpec spec, Framework fw, WordEncoding enc);
[[nodiscard]] nb::object export_bundle(const std::shared_ptr<const rl::ArrayBundle>& bundle, Framework fw, WordEncoding enc);

/// The producer side of a device array's DLPack stream handoff (implemented by the CUDA bindings).
class DeviceSync
{
public:
    virtual ~DeviceSync() = default;
    /// Makes the consumer's stream (a DLPack stream value: 1 legacy, 2 per-thread, otherwise a cudaStream_t) wait for
    /// the producer's pending writes, and records it as a user of the memory. Never called with -1.
    virtual void handoff(std::intptr_t consumer_stream) = 0;
};

/// Where an exported array lives (DLPack device type and id) and how consumers synchronize with it.
struct DeviceExport
{
    int32_t device_type = 1;  // kDLCPU
    int32_t device_id = 0;
    std::shared_ptr<DeviceSync> sync;
};

/// export_array for memory on a device: framework torch, JAX or dlpack (NumPy holds host memory only; TypeError).
[[nodiscard]] nb::object export_device_array(ArraySpec spec, DeviceExport device, Framework fw, WordEncoding enc);

/// A batch of states taken from a Python object without a copy (unless it is a State or a sequence of States).
struct StateBatch
{
    rl::StateBatchView view;
    Framework fw = Framework::Numpy;
    WordEncoding enc;
    bool single = false;  // a 1-d input (one state)
    nb::object keep;      // keeps the source alive (an ndarray handle or a packed copy's owner)
    std::shared_ptr<std::vector<u64>> packed;
    std::shared_ptr<void> keep_native;
    /// States packed from State objects: the Task::uid() of each row's task (empty for arrays). The importer of a
    /// batch checks them against the task or instance a row belongs to.
    std::vector<u64> owners;
};

/// `task_words` is used to pack State sequences.
[[nodiscard]] StateBatch import_states(nb::handle obj, u32 task_words);

/// State words described by a DLPack-style array (dtype code/bits/lanes, shape, strides in elements), validated as
/// state words: [N, W] 64-bit or [N, 2W] 32-bit integers, contiguous words, 8-byte aligned. Raises TypeError.
struct WordsLayout
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 words = 0;
    u64 stride = 0;  // in words
    bool single = false;
    WordEncoding enc;
};
[[nodiscard]] WordsLayout words_layout(const void* data, u8 code, u8 bits, u16 lanes, usize ndim, const i64* shape,
                                       const i64* strides);

/// A writable destination array for expand_into: data pointer and shape in elements.
struct Dest
{
    void* data = nullptr;
    rl::DType dtype = rl::DType::I32;
    std::vector<i64> shape;
    std::vector<i64> strides;
    nb::object keep;
};
[[nodiscard]] Dest import_dest(nb::handle obj, const char* name);

void bind_arrays(nb::module_& m);
}  // namespace mymyr::python
