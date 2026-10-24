#pragma once

#include "mymyr/cuda/runtime.hpp"

namespace mymyr::cuda::detail
{
/// Non-null only while destroying exclusive resources whose device work has completed. Their buffers release on
/// this stream without consulting allocation streams, which may have been destroyed or used for another capture.
extern thread_local cudaStream_t completed_release_stream;

/// Releases completed work independently of capture streams. Relaxed mode permits cleanup of unrelated resources
/// while another thread captures; no allocation or use of those resources can belong to that capture.
class CompletedRelease
{
public:
    explicit CompletedRelease(ContextPtr ctx);
    ~CompletedRelease();
    CompletedRelease(const CompletedRelease&) = delete;
    CompletedRelease& operator=(const CompletedRelease&) = delete;

private:
    ContextPtr m_ctx;
    DeviceGuard m_guard;
    cudaStreamCaptureMode m_mode = cudaStreamCaptureModeRelaxed;
    cudaStream_t m_stream = nullptr;
    cudaStream_t m_previous = nullptr;
};
}  // namespace mymyr::cuda::detail
