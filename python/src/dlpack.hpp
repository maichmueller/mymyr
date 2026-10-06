#pragma once
// The DLPack ABI (dmlc/dlpack include/dlpack/dlpack.h v1.1), declared here so no header is needed, and our own
// DLPack import with the stream handoff of the Python array API:
//   obj.__dlpack__(stream=s) asks the producer to make its pending writes visible on the consumer's stream s
//   (s = 1: the legacy default stream, 2: the per-thread default stream, -1: no synchronization, else a cudaStream_t).
// nanobind's ndarray import passes no stream, so CUDA inputs go through import_dlpack().

#include "mymyr/core/types.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mymyr::python::dl
{
namespace nb = nanobind;

struct Device
{
    int32_t device_type;
    int32_t device_id;
};
struct DataType
{
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};
struct Tensor
{
    void* data;
    Device device;
    int32_t ndim;
    DataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};
struct ManagedTensor
{
    Tensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(ManagedTensor*);
};
struct PackVersion
{
    uint32_t major;
    uint32_t minor;
};
struct ManagedTensorVersioned
{
    PackVersion version;
    void* manager_ctx;
    void (*deleter)(ManagedTensorVersioned*);
    uint64_t flags;
    Tensor dl_tensor;
};
inline constexpr uint64_t k_flag_read_only = 1;
inline constexpr uint64_t k_flag_is_copied = 2;
inline constexpr int32_t k_cpu = 1, k_cuda = 2, k_cuda_host = 3, k_cuda_managed = 13;
inline constexpr uint8_t k_int = 0, k_uint = 1, k_float = 2, k_bool = 6;
inline constexpr uint32_t k_major = 1, k_minor = 1;

/// DLPack stream values (array API): the legacy default stream, the per-thread default stream, no synchronization.
inline constexpr std::intptr_t k_stream_legacy = 1, k_stream_per_thread = 2, k_stream_none = -1;
/// Not a stream value: import_dlpack passes stream=None (the producer's default).
inline constexpr std::intptr_t k_stream_default = INTPTR_MIN;

[[nodiscard]] inline bool is_cuda_family(int32_t t) { return t == k_cuda || t == k_cuda_host || t == k_cuda_managed; }
/// Memory the host may read (CPU, pinned host, managed).
[[nodiscard]] inline bool host_accessible(int32_t t) { return t == k_cpu || t == k_cuda_host || t == k_cuda_managed; }

/// A consumed DLPack tensor: the producer's memory stays valid while `keep` lives (its deleter runs with the last
/// reference, on any thread; the DLPack deleters of torch, JAX and mymyr need no Python thread state).
struct Imported
{
    void* data = nullptr;  // byte_offset folded in
    Device device{k_cpu, 0};
    DataType dtype{k_int, 64, 1};
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // in elements, always set (C-contiguous when the producer gave none)
    bool readonly = false;
    std::shared_ptr<void> keep;
};

/// The (device type, device id) of obj.__dlpack_device__(), or {-1, -1} if obj has no such method.
[[nodiscard]] Device dlpack_device(nb::handle obj);

/// Imports obj through obj.__dlpack__(stream=stream, max_version=(1, 1)) (without max_version for producers that
/// predate it; stream=None for k_stream_default) and consumes the capsule. Raises TypeError if obj speaks no DLPack.
[[nodiscard]] Imported import_dlpack(nb::handle obj, std::intptr_t stream);
}  // namespace mymyr::python::dl
