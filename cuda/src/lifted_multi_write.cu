// Multi-instance launches of the write kernel (include/mymyr/cuda/lifted.hpp, launch_write_multi): lifted.cu's
// launch_write with the instance resolved per thread, one launch per OW group and row-width bucket (see
// lifted_multi.cu). No conditional effects (the multi-instance paths run tables whose schemas have none).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted
{
cudaError_t launch_write_multi(const Multi& m, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                               SuccessorRows out, u32 wb, cudaStream_t s)
{
    if (!m.views || !m.inst || !m.instances || !m.fc_of || m.num_schemas != set.num_schemas || m.words_hi > wb ||
        (out.deferred && !labels.binding) || m.starts)
        return cudaErrorInvalidValue;
    GenArgs a{TaskView{}, p, v, set, offsets, nullptr, labels, out};
    a.m = m;
    switch (wb)
    {
        case 2: return dispatch_ow<true, 2, false, true>(a, s);
        case 4: return dispatch_ow<true, 4, false, true>(a, s);
        case 16: return dispatch_ow<true, 16, false, true>(a, s);
        case k_max_words: return dispatch_ow<true, k_max_words, false, true>(a, s);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t launch_put_multi(const Multi& m, Parents p, const u32* offsets, Labels labels, SuccessorRows out,
                             const PutRows& to, u64 rows_hint, cudaStream_t s)
{
    if (!m.views || !m.inst || !m.instances || !m.num_schemas || !labels.binding || (to.goal && !out.words) || m.starts)
        return cudaErrorInvalidValue;
    SchemaSet set;
    set.num_schemas = m.num_schemas;
    out.deferred = 0;
    GenArgs a{TaskView{}, p, Views{}, set, offsets, nullptr, labels, out};
    a.m = m;
    return launch_put_rows<true>(a, to, nullptr, rows_hint, s);
}
}  // namespace mymyr::cuda::lifted
