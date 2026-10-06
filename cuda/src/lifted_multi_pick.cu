// Multi-instance launches of the pick kernel (include/mymyr/cuda/lifted.hpp, launch_pick_multi): lifted.cu's
// launch_pick with the instance resolved per thread, one launch per OW group and row-width bucket (see lifted_multi.cu).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_env.cuh"

namespace mymyr::cuda::lifted
{
cudaError_t launch_pick_multi(const Multi& m, Parents p, Views v, SchemaSet set, Picks picks, Labels labels,
                              SuccessorRows out, u32 wb, cudaStream_t s)
{
    if (p.rows == 0 || set.count == 0)
        return cudaSuccess;
    if (!m.views || !m.inst || !m.instances || !m.fc_of || m.num_schemas != set.num_schemas || !picks_valid(p, set, picks, labels, true) ||
        (picks.cache.regions && picks.cache.region_stride < picks.cache.num_schemas) || m.words_hi > wb || !range_valid(m))
        return cudaErrorInvalidValue;
    GenArgs a{TaskView{}, p, v, set, nullptr, nullptr, labels, out};
    a.m = m;
    switch (wb)
    {
        case 2: return dispatch_pick<2, true>(a, picks, s);
        case 4: return dispatch_pick<4, true>(a, picks, s);
        case 16: return dispatch_pick<16, true>(a, picks, s);
        case k_max_words: return dispatch_pick<k_max_words, true>(a, picks, s);
        default: return cudaErrorInvalidValue;
    }
}
}  // namespace mymyr::cuda::lifted
