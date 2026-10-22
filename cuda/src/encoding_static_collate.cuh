#pragma once

#include "encoding_collate.hpp"

namespace mymyr::cuda::detail
{
struct StaticSource
{
    const DeviceCollationBatch* batch = nullptr;
    u64 block = 0;
};

__device__ inline StaticSource static_source(const CollationParams& p, u32 instance, u64 begin = 0)
{
    for (u64 b = begin; b < p.batches; ++b)
    {
        const auto& batch = p.slices[b].batch;
        if (batch.sizes && batch.sizes->overflow) continue;
        const auto sizes = *batch.static_sizes;
        if (sizes.instances > batch.static_capacities.instances || sizes.args > batch.static_capacities.args) continue;
        for (u64 block = 0; block < sizes.instances; ++block)
            if (batch.arrays.static_task_ids[block] == static_cast<i32>(instance))
                return {&batch, block};
    }
    return {};
}

template<class I>
__global__ void count_static_collation(CollationParams p)
{
    encoding::StaticSizes totals{};
    const u64 relations = p.static_relations;
    for (u32 instance = 0; instance < p.instances; ++instance)
    {
        const auto first = static_source(p, instance);
        if (!first.batch) continue;
        const auto& a = first.batch->arrays;
        const auto* offsets = static_cast<const I*>(a.static_offsets) + first.block * relations;
        const auto* args = static_cast<const I*>(a.static_arg_offsets) + first.block * relations;
        ++totals.instances;
        totals.tuples += offsets[relations] - offsets[0];
        totals.args += args[relations] - args[0];
        for (u64 b = 0; b < p.batches; ++b)
        {
            const auto& batch = p.slices[b].batch;
            if (batch.sizes && batch.sizes->overflow) { *p.bad = 1; continue; }
            if (batch.static_sizes->instances > batch.static_capacities.instances ||
                batch.static_sizes->args > batch.static_capacities.args) { *p.bad = 1; continue; }
            for (u64 s = 0; s < relations; ++s)
                if (static_cast<const I*>(batch.arrays.static_relations)[s] !=
                    static_cast<const I*>(a.static_relations)[s]) *p.bad = 1;
            for (u64 block = 0; block < batch.static_sizes->instances; ++block)
            {
                if (batch.arrays.static_task_ids[block] != static_cast<i32>(instance)) continue;
                const auto* other = static_cast<const I*>(batch.arrays.static_offsets) + block * relations;
                const auto* other_args = static_cast<const I*>(batch.arrays.static_arg_offsets) + block * relations;
                bool equal = true;
                for (u64 s = 0; s <= relations; ++s)
                    equal = equal && other[s] - other[0] == offsets[s] - offsets[0] &&
                            other_args[s] - other_args[0] == args[s] - args[0];
                if (equal)
                    for (I j = 0; j < args[relations] - args[0]; ++j)
                        equal = equal && static_cast<const I*>(batch.arrays.static_args)[other_args[0] + j] ==
                                         static_cast<const I*>(a.static_args)[args[0] + j];
                if (!equal) *p.bad = 1;
            }
        }
    }
    *p.static_totals = totals;
    *p.output.static_sizes = totals;
}

template<class I>
__global__ void fill_static_collation(CollationParams p)
{
    const auto& out = p.output.arrays;
    const auto sizes = *p.static_totals;
    *p.output.static_sizes = sizes;
    auto* offsets = static_cast<I*>(out.static_offsets);
    auto* arg_offsets = static_cast<I*>(out.static_arg_offsets);
    auto* args = static_cast<I*>(out.static_args);
    const u64 relations = p.static_relations;
    u64 blocks = 0, tuples = 0, arguments = 0;
    const bool overflow = p.output.sizes->overflow;
    if (!overflow)
    {
        for (u32 instance = 0; instance < p.instances; ++instance)
        {
            const auto source = static_source(p, instance);
            if (!source.batch) continue;
            const auto& a = source.batch->arrays;
            const auto* src = static_cast<const I*>(a.static_offsets) + source.block * relations;
            const auto* src_args = static_cast<const I*>(a.static_arg_offsets) + source.block * relations;
            out.static_task_ids[blocks] = static_cast<i32>(instance);
            for (u64 s = 0; s < relations; ++s)
            {
                offsets[blocks * relations + s] = static_cast<I>(tuples + src[s] - src[0]);
                arg_offsets[blocks * relations + s] = static_cast<I>(arguments + src_args[s] - src_args[0]);
                static_cast<I*>(out.static_relations)[s] = static_cast<const I*>(a.static_relations)[s];
            }
            for (I j = src_args[0]; j < src_args[relations]; ++j)
                args[arguments + j - src_args[0]] = static_cast<const I*>(a.static_args)[j];
            ++blocks;
            tuples += src[relations] - src[0];
            arguments += src_args[relations] - src_args[0];
        }
        // An empty batch still carries its static relation catalog.
        for (u64 s = 0; s < relations; ++s)
            static_cast<I*>(out.static_relations)[s] =
                static_cast<const I*>(p.slices[0].batch.arrays.static_relations)[s];
        offsets[blocks * relations] = static_cast<I>(tuples);
        arg_offsets[blocks * relations] = static_cast<I>(arguments);
    }
    if (p.output.zero_unused)
    {
        for (u64 b = blocks; b < p.output.static_capacities.instances; ++b) out.static_task_ids[b] = 0;
        for (u64 s = blocks * relations + (overflow ? 0 : 1);
             s <= p.output.static_capacities.instances * relations; ++s)
        {
            offsets[s] = static_cast<I>(tuples);
            arg_offsets[s] = static_cast<I>(arguments);
        }
        for (u64 j = arguments; j < p.output.static_capacities.args; ++j) args[j] = 0;
        if (overflow)
            for (u64 s = 0; s < relations; ++s) static_cast<I*>(out.static_relations)[s] = 0;
    }
}
}
