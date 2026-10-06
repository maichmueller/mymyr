#pragma once
// Task-level arrays for encoders (mifrost), environments and device kernels.
//
//   - atom_metadata(task): what an observation encoder needs, and nothing more (mymyr ships no encoders):
//       atom_pred [F], atom_args (CSR: atom_args_offsets [F+1] + atom_args), atom_args_padded [F, A] (-1 padding),
//       atom_cid [F] (the portable canonical id), predicate-grouped slots (pred_slot_offsets [P+1] + pred_slots [F]),
//       pred_arity / pred_kind [P], the derived atoms' pred / args, the static atoms, and object counts.
//   - goal_masks(task): gpos / gneg as state words. Positive and negative fluent goal literals; the goal's
//     derived literals, if any, are listed separately (a mask test cannot decide them).
//   - device_arrays(task): the flat, versioned POD export: scalars plus named arrays, a superset of the above
//     with the initial state and the typed-dense canonical layout. Version 1 holds what the host already fixes.
//     Version 2 (the default) is a strict superset: it appends section "plan" (arrays plan_*, scalars plan_*),
//     the compiled matcher, effect, axiom and goal tables with every pointer replaced by an offset, the slot table
//     snapshot and the static relations; rl/task_arrays_view.hpp documents the layout and reads it on the host and the
//     device (cuda::DeviceTask uploads the block once). Sections are versioned (scalars section_core, section_plan),
//     so later versions can append sections without moving what exists. Numeric tasks add init_numeric [NN] and
//     the scalars numeric_slots, numeric_words, numeric_storage to the core (absent for classical tasks); no device
//     kernel reads them yet, so DeviceTask rejects numeric tasks.
//
// Everything is a snapshot: immutable once built, so it may be exported zero-copy to any number of consumers (JAX
// requires that an imported buffer is never written again). Under lazy slots the snapshot covers the atom slots
// published when it was taken (F grows later); in frozen mode it covers every slot.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/task_arrays_view.hpp"
#include "mymyr/task/task.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::rl
{
enum class DType : u8
{
    Bool,
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
    F32,  // the CPU pool's rewards
    F64,  // state spaces' costs and cost goal distances
};

[[nodiscard]] constexpr u32 dtype_bytes(DType d) noexcept
{
    switch (d)
    {
        case DType::Bool:
        case DType::I8:
        case DType::U8: return 1;
        case DType::I32:
        case DType::U32:
        case DType::F32: return 4;
        case DType::I64:
        case DType::U64:
        case DType::F64: return 8;
    }
    return 1;
}

struct ArrayInfo
{
    std::string name;
    DType dtype = DType::I32;
    std::vector<i64> shape;
    u64 offset = 0;      // bytes into the bundle's block (64-byte aligned)
    bool words = false;  // the last dimension holds state words: exporters may reinterpret it per framework
    [[nodiscard]] u64 elements() const noexcept
    {
        u64 n = 1;
        for (i64 s : shape)
            n *= static_cast<u64>(s);
        return n;
    }
};

/// Named immutable arrays in one aligned block, plus named integer scalars.
class ArrayBundle
{
public:
    [[nodiscard]] const std::vector<ArrayInfo>& arrays() const noexcept { return m_arrays; }
    [[nodiscard]] const std::vector<std::pair<std::string, i64>>& scalars() const noexcept { return m_scalars; }
    [[nodiscard]] const void* data(const ArrayInfo& a) const noexcept { return m_base + a.offset; }
    [[nodiscard]] const ArrayInfo* find(std::string_view name) const noexcept;
    [[nodiscard]] i64 scalar(std::string_view name, i64 fallback = -1) const noexcept;
    [[nodiscard]] u64 bytes() const noexcept { return m_bytes; }
    /// The block holding every array (64-byte aligned, bytes() long): uploads copy it as one piece.
    [[nodiscard]] const std::byte* block() const noexcept { return m_base; }

    class Builder;

private:
    std::vector<ArrayInfo> m_arrays;
    std::vector<std::pair<std::string, i64>> m_scalars;
    std::unique_ptr<std::byte[]> m_block;
    std::byte* m_base = nullptr;  // 64-byte aligned start inside m_block
    u64 m_bytes = 0;
};

class ArrayBundle::Builder
{
public:
    /// Adds an array (copied at build()).
    template<class T>
    void add(std::string name, DType dtype, std::vector<i64> shape, const std::vector<T>& values, bool words = false)
    {
        const auto* p = reinterpret_cast<const std::byte*>(values.data());
        add_bytes(std::move(name), dtype, std::move(shape), std::vector<std::byte>(p, p + values.size() * sizeof(T)), words);
    }
    void add_bytes(std::string name, DType dtype, std::vector<i64> shape, std::vector<std::byte> bytes, bool words);
    void scalar(std::string name, i64 value) { m_scalars.emplace_back(std::move(name), value); }
    [[nodiscard]] ArrayBundle build();

private:
    std::vector<std::pair<ArrayInfo, std::vector<std::byte>>> m_items;
    std::vector<std::pair<std::string, i64>> m_scalars;
};

/// Kind codes of pred_kind.
inline constexpr i32 k_pred_static = 0, k_pred_fluent = 1, k_pred_derived = 2;

/// Snapshot of the atom metadata (see the file comment).
[[nodiscard]] ArrayBundle atom_metadata(const Task& task);

struct GoalMasks
{
    std::vector<u64> pos, neg;              // fluent goal literals as state words (trimmed to the assigned width)
    std::vector<u32> derived_pos, derived_neg;  // derived goal literals, as derived slots
    bool unsatisfiable = false;             // the goal contains a statically false literal
    [[nodiscard]] bool uses_derived() const noexcept { return !derived_pos.empty() || !derived_neg.empty(); }
};

/// The goal as masks. Assigns slots to the goal's atoms under lazy slots (the task's only mutation).
[[nodiscard]] GoalMasks goal_masks(const Task& task);

/// The latest layout version (the default); version 1 stays available and unchanged.
inline constexpr u32 k_device_arrays_version = 2;
/// Section versions of the latest layout.
inline constexpr u32 k_section_core_version = 1;  // version 1's arrays
inline constexpr u32 k_section_plan_version = 2;  // compiled plans (task_arrays_view.hpp); 2: the compact view, plan_view_map

/// The versioned POD export of the task. Throws std::invalid_argument for an unknown version, and
/// std::length_error when a version-2 table outgrows its u32 offsets.
[[nodiscard]] ArrayBundle device_arrays(const Task& task, u32 version = k_device_arrays_version);

/// The work of the device's two matchers of one plan matcher: its fixed-order search and its fail-first forward
/// checking (cuda/src/lifted_device.cuh run_fixed / run_fc), in units of a word read or written, summed over the
/// sample states of device_search_costs.
struct SearchCost
{
    u64 fixed = 0, fc = 0;
};
/// Forward checking copies and scans a stack of domains per search node where the fixed order intersects a few rows
/// (the kernels' local-memory stacks: 2.4-9.3 KB per thread against 0.3-2 KB), and the device launches the
/// forward-checking schemas apart from the others: the device prefers it where the model puts it at less than
/// 1 / k_fc_margin of the fixed order's work.
inline constexpr u64 k_fc_margin = 4;
[[nodiscard]] constexpr bool prefers_fc(SearchCost c) noexcept { return c.fc * k_fc_margin < c.fixed; }
/// Sample states of device_search_costs: the initial state and its first successors.
inline constexpr usize k_probe_states = 32;
struct DeviceSearchCosts
{
    std::vector<std::array<SearchCost, 2>> schemas;  // per schema: pre[0] (witness pruning), pre[1]; zero: not probed
    std::vector<SearchCost> axioms;                  // per axiom body, in plan_axiom order (strata, then axioms)
    u32 states = 0;                                  // sample states
};
/// The device's search costs of every schema matcher and axiom body the device kernels can run (at most
/// dev::k_deep_matcher_depth parameters, no numeric constraints), replayed on the CPU's engine over sample states: the
/// initial state and, in frozen mode, its successors in canonical order (at most k_probe_states states; under lazy
/// slots generating them would intern atoms, so only the initial state). Deterministic. device_arrays sets
/// dev::k_mc_device_fc from it; numeric tasks get zeros.
[[nodiscard]] DeviceSearchCosts device_search_costs(const Task& task);

/// A view of a version-2 export whose block starts at `base` (nullptr: the bundle's own host block; otherwise a copy
/// of block(), e.g. on the device). Throws std::invalid_argument if the bundle is not version 2 or later.
[[nodiscard]] dev::TaskView task_view(const ArrayBundle& bundle, const void* base = nullptr);
}  // namespace mymyr::rl
