#pragma once

#include "mymyr/cuda/rl_ops.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/rl/expand.hpp"

namespace mymyr::cuda::detail
{
inline ExpansionRows expansion_rows(const rl::Expansion& e)
{
    return {e.capacity, e.words + e.numeric_words, e.label_width, e.derived_words,
            e.succ, e.parent, e.schema, e.binding, e.goal, e.derived, e.succ_hash, e.noop, e.index, e.offsets};
}

/// Pre-drop destinations and scan scratch, retained across launches and graph replays.
struct ExpansionDrop
{
    Scratch succ, parent, schema, binding, goal, derived, hashes, noops, offsets, counts, scan;

    rl::Expansion prepare(const rl::Expansion& out, u64 rows, const ContextPtr& ctx, cudaStream_t s)
    {
        rl::Expansion e = out;
        auto alloc = [&](Scratch& x, u64 bytes) { return x.ensure(ctx, bytes ? bytes : 1, s); };
        const u64 n = out.capacity;
        e.succ = out.succ ? static_cast<u64*>(alloc(succ, n * (out.words + out.numeric_words) * sizeof(u64))) : nullptr;
        e.parent = out.parent ? static_cast<i32*>(alloc(parent, n * sizeof(i32))) : nullptr;
        e.schema = out.schema ? static_cast<i32*>(alloc(schema, n * sizeof(i32))) : nullptr;
        e.binding = out.binding ? static_cast<i32*>(alloc(binding, n * out.label_width * sizeof(i32))) : nullptr;
        e.goal = out.goal ? static_cast<u8*>(alloc(goal, n)) : nullptr;
        e.derived = out.derived ? static_cast<u64*>(alloc(derived, n * out.derived_words * sizeof(u64))) : nullptr;
        e.succ_hash = out.succ_hash ? static_cast<u64*>(alloc(hashes, n * sizeof(u64))) : nullptr;
        e.noop = static_cast<u8*>(alloc(noops, n));
        e.index = nullptr;
        e.offsets = static_cast<i32*>(alloc(offsets, (rows + 1) * sizeof(i32)));
        return e;
    }

    void compact(const rl::Expansion& src, rl::Expansion& out, u64 rows, const ContextPtr& ctx, cudaStream_t s,
                 const u32* status = nullptr)
    {
        auto* count = static_cast<u32*>(counts.ensure(ctx, (rows + 1) * sizeof(u32), s));
        check(launch_noop_counts(expansion_rows(src), rows, count, s, status), "launch_noop_counts");
        const u64 bytes = lifted::scan_temp_bytes(rows + 1);
        check(lifted::launch_scan(count, reinterpret_cast<u32*>(out.offsets), rows + 1,
                                 scan.ensure(ctx, bytes, s), bytes, s), "launch_scan");
        check(launch_noop_scatter(expansion_rows(src), expansion_rows(out), rows, s, status), "launch_noop_scatter");
    }
};
}  // namespace mymyr::cuda::detail
