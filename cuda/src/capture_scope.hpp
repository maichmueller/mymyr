#pragma once

#include <cuda_runtime_api.h>

namespace mymyr::cuda::detail
{
/// Allows cleanup outside an unrelated capture without changing stream dependencies.
/// Restores the calling thread's capture interaction mode on scope exit; cleanup never throws.
class RelaxedCaptureScope
{
public:
    RelaxedCaptureScope() noexcept : m_restore(cudaThreadExchangeStreamCaptureMode(&m_mode) == cudaSuccess) {}
    ~RelaxedCaptureScope()
    {
        if (m_restore)
            (void)cudaThreadExchangeStreamCaptureMode(&m_mode);
    }
    RelaxedCaptureScope(const RelaxedCaptureScope&) = delete;
    RelaxedCaptureScope& operator=(const RelaxedCaptureScope&) = delete;

private:
    cudaStreamCaptureMode m_mode = cudaStreamCaptureModeRelaxed;
    bool m_restore;
};
}  // namespace mymyr::cuda::detail
