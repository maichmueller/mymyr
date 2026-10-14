// A compile check of the device-code subset of the relational encoding (encoding/detail/lanes.hpp): one kernel per index
// type and layout that counts a graph and writes it with the per-graph functions the host encoder uses. It has no host API and
// nothing launches it.

#include "mymyr/encoding/detail/lanes.hpp"

namespace mymyr::encoding::detail
{
/// One thread per graph: graph g is row g of `atoms` (`atom_stride` words apart) of instance `instance[g]`. `bases` holds
/// the entity, tuple and argument base of each graph (three u64 per graph).
template<class I>
__global__ void encode_lanes_kernel(ContextView context, const u32* instance, const u64* atoms, u64 atom_stride,
                                    const u64* derived, u64 derived_stride, const u64* bases, u64 graphs,
                                    GraphTotals* totals, GraphOut<I> out)
{
    const u64 g = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= graphs)
        return;
    const InstanceView v = instance_view(context, instance[g]);
    const RowInput in{atoms + g * atom_stride, derived ? derived + g * derived_stride : nullptr};
    totals[g] = count_graph(v, in, nullptr);
    fill_graph<I>(v, in, g, bases[3 * g + 1], bases[3 * g + 2], bases[3 * g], totals[g], out);
}

/// The same for the relation-major layout: the segment offsets of `out` were scanned beforehand, and the graph reads its
/// cells' positions from them.
template<class I>
__global__ void encode_lanes_relation_major_kernel(ContextView context, const u32* instance, const u64* atoms, u64 atom_stride,
                                                   const u64* derived, u64 derived_stride, const u64* entity_bases, u64 graphs,
                                                   GraphTotals* totals, GraphOut<I> out)
{
    const u64 g = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= graphs)
        return;
    const InstanceView v = instance_view(context, instance[g]);
    const RowInput in{atoms + g * atom_stride, derived ? derived + g * derived_stride : nullptr};
    totals[g] = count_graph(v, in, nullptr);
    fill_graph_relation_major<I>(v, in, g, entity_bases[g], totals[g], out);
}

template __global__ void encode_lanes_kernel<i32>(ContextView, const u32*, const u64*, u64, const u64*, u64, const u64*, u64,
                                                  GraphTotals*, GraphOut<i32>);
template __global__ void encode_lanes_kernel<i64>(ContextView, const u32*, const u64*, u64, const u64*, u64, const u64*, u64,
                                                  GraphTotals*, GraphOut<i64>);
template __global__ void encode_lanes_relation_major_kernel<i32>(ContextView, const u32*, const u64*, u64, const u64*, u64,
                                                                 const u64*, u64, GraphTotals*, GraphOut<i32>);
template __global__ void encode_lanes_relation_major_kernel<i64>(ContextView, const u32*, const u64*, u64, const u64*, u64,
                                                                 const u64*, u64, GraphTotals*, GraphOut<i64>);
}  // namespace mymyr::encoding::detail
