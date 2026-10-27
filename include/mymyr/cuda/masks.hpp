#pragma once
// Lifted action masks and label applicability on the device: rl::schema_masks, rl::parameter_masks and
// rl::is_applicable (rl/masks.hpp) over state rows in device memory, with the same results bit for bit, and the
// existence test of the label-mode environment (cuda/env.hpp). The kernels (cuda/mask_kernels.hpp) run each schema's
// mask matchers (rl/task_arrays_view.hpp, plan_schema k_sc_masks) over per-row matcher views: parameters before the
// asked one prebound from the prefix, the asked one enumerated, a witness searched for the rest.
//
// Destination-passing: the rows, the outputs, the views scratch ([rows, view_words()] u64 words) and the status word
// are device memory the caller owns; a call runs on one stream, zeroes its outputs and status first, synchronizes
// nothing and allocates nothing, so calls can be captured into CUDA graphs and replayed. Conditions under which the
// host functions throw set bits of the status word instead (cuda/mask_kernels.hpp: maskk::k_bad_task_id, ...); the
// outputs of such rows are zero or unspecified.
//
// Tasks the kernels run: those of the device matchers (ChunkGenerator::unsupported) without numeric fluents and axioms,
// whose schemas have at most lifted::k_max_depth parameters (unsupported() says why not). The object is immutable after
// construction (the tables are uploaded and the context's stream synchronized), so any number of threads may call it
// at once on their own streams.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/task_suite.hpp"

#include <mutex>
#include <string>
#include <vector>

namespace mymyr::cuda
{
class DeviceMasks
{
public:
    /// Uploads the suite's tables. Throws std::invalid_argument when the kernels do not run some instance.
    DeviceMasks(ContextPtr ctx, rl::TaskSuitePtr suite);
    DeviceMasks(const DeviceMasks&) = delete;
    DeviceMasks& operator=(const DeviceMasks&) = delete;

    /// Empty if the kernels run every instance of the suite, else why not (for the first instance they do not run).
    [[nodiscard]] static std::string unsupported(const rl::TaskSuite& suite);

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] const rl::TaskSuitePtr& suite() const noexcept { return m_suite; }
    /// Words of the views scratch per row (the widest instance's matcher view, at least 1).
    [[nodiscard]] u64 view_words() const noexcept { return m_view_words; }
    /// The instances' device views [I] (the suite's global instance ids).
    [[nodiscard]] const rl::dev::TaskView* device_views() const noexcept;

    /// State rows in device memory: [rows, stride] u64 atom words (the suite has no numeric words), row i of instance
    /// task_ids[i] (device; null: a suite of one instance).
    struct Batch
    {
        const u64* states = nullptr;
        u64 rows = 0;
        u32 words = 0;
        u64 stride = 0;
        const i32* task_ids = nullptr;
    };
    /// A schema per row: schema[i] (device), or `value` for every row when schema is null.
    struct Schemas
    {
        const i32* schema = nullptr;
        i32 value = -1;
    };

    /// out [rows, suite.max_schemas()] u8: rl::schema_masks.
    void schemas(const Batch& b, u8* out, u64* views, u32* status, cudaStream_t s) const;
    /// out [rows, num_objects] u8 (num_objects >= suite.max_objects()): rl::parameter_masks; prefix [rows, stride]
    /// (stride >= depth; may be null at depth 0). A depth at or past a selected schema's arity sets maskk::k_bad_depth.
    void objects(const Batch& b, Schemas schema, const i32* prefix, u32 stride, u32 depth, u8* out, u32 num_objects,
                 u64* views, u32* status, cudaStream_t s) const;
    /// out [rows] u8: rl::is_applicable; binding [rows, width]. Fewer columns than a selected schema's arity set
    /// maskk::k_narrow_binding.
    void applicable(const Batch& b, Schemas schema, const i32* binding, u32 width, u8* out, u32* status,
                    cudaStream_t s) const;
    /// out[i * out_stride] = 1 if some action is applicable in row i, else 0, for the rows with skip[i] zero (skip
    /// null: every row; the others are not written). No checks of the rows.
    void any(const Batch& b, const u8* skip, u32* out, u64 out_stride, u64* views, cudaStream_t s) const;

private:
    void use(cudaStream_t s) const;
    void views_of(const Batch& b, u64* views, cudaStream_t s) const;
    void check_rows(const Batch& b, u32* status, cudaStream_t s) const;

    ContextPtr m_ctx;
    rl::TaskSuitePtr m_suite;
    std::vector<DeviceTaskTablePtr> m_tables;  // per domain
    rl::dev::TaskView m_view0{};               // instance 0's view (single-instance launches)
    DeviceBuffer m_views, m_words, m_slots;    // [I] device views, atom words, fluent slots
    std::vector<u32> m_ows;                    // the object-bitset width classes of the instances
    u64 m_view_words = 1;
    mutable std::mutex m_mutex;
    mutable std::vector<cudaStream_t> m_streams;  // the streams the buffers were used on
};
}  // namespace mymyr::cuda
