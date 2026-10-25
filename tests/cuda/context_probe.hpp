#pragma once
// A readback of an encoding context through its POD view, on the host and on the device: per instance, the scalar
// columns and a hash of the rows of every array the instance owns. The host and device results of equal views are
// equal. A test kernel for the views of cuda/encoding.hpp, not an encoder.

#include "mymyr/encoding/detail/lanes.hpp"
#include "mymyr/encoding/detail/literals.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::test
{
inline constexpr u32 k_probe_width = 13;

MYMYR_HD u64 probe_mix(u64 h, u64 v) { return (h ^ v) * 0x100000001b3ull; }

/// out[0..6]: objects, words, fluent slots, derived words, derived slots, cells, encoding fingerprint; out[7..10]:
/// hashes of the instance's cells, fluent slot rows, derived slot rows and static rows; out[11]: relations and the
/// widest arity; out[12]: a hash of the instance's literal section and of the resolution (resolve_literal) of the atom
/// of every slot and static row it lists.
MYMYR_HD void probe_instance(const encoding::detail::ContextView& c, u32 instance, u64* out)
{
    using namespace encoding::detail;
    const u64* row = c.inst + u64{instance} * k_instance_columns;
    const InstanceView v = instance_view(c, instance);
    out[0] = v.num_objects;
    out[1] = v.words;
    out[2] = v.fluent_slots;
    out[3] = v.derived_words;
    out[4] = v.derived_slots;
    out[5] = v.num_cells;
    out[6] = row[k_encoding_fingerprint];
    u64 cells = 0xcbf29ce484222325ull, static_rows = 0;
    for (u32 i = 0; i < v.num_cells; ++i)
    {
        const Cell& cell = v.cells[i];
        for (u64 x : {u64{cell.relation}, u64{cell.op}, u64{cell.a}, u64{cell.b}, u64{cell.arity}, u64{cell.logical_arity}})
            cells = probe_mix(cells, x);
        if (cell.op == static_cast<u32>(CellOp::Static) && cell.b > static_rows)
            static_rows = cell.b;
    }
    auto rows = [&](const i32* args, u64 n)
    {
        u64 h = 0xcbf29ce484222325ull;
        for (u64 i = 0; i < n * v.max_arity; ++i)
            h = probe_mix(h, static_cast<u64>(static_cast<i64>(args[i])));
        return h;
    };
    out[7] = cells;
    out[8] = rows(v.slot_args, v.fluent_slots);
    out[9] = rows(v.derived_args, v.derived_slots);
    out[10] = rows(v.static_args, static_rows);
    out[11] = u64{v.num_relations} << 32 | v.max_arity;
    u64 literals = 0xcbf29ce484222325ull;
    for (u32 p = 0; p < c.num_predicates; ++p)
    {
        const u32* r = c.literal_ranges + (u64{instance} * c.num_predicates + p) * k_literal_columns;
        for (u32 k = 0; k < k_literal_columns; ++k)
            literals = probe_mix(literals, r[k]);
        const i32* sets[3] = {v.slot_args, v.derived_args, v.static_args};
        for (u32 set = 0; set < 3; ++set)
            for (u32 at = r[2 * set]; at < r[2 * set + 1]; ++at)
            {
                const LiteralSlot s = resolve_literal(c, instance, p, sets[set] + u64{at} * v.max_arity, v.max_arity);
                literals = probe_mix(literals, u64{static_cast<u32>(s.kind)} << 32 | s.slot);
            }
    }
    out[12] = literals;
}

/// One thread per instance: probe_instance(context, i, out + i * k_probe_width). Enqueued on `stream`.
void launch_context_probe(const encoding::detail::ContextView& context, u32 instances, u64* out, cudaStream_t stream);
}  // namespace mymyr::test
