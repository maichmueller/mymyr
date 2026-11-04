#pragma once
// The launch plan of multi-instance batches over a task table (internal to cuda/src: DeviceExpander's one-pass
// mode and DeviceEnv's fast path over several instances). Host C++.
//
// The multi-instance kernels (cuda/lifted.hpp, "several instances") run one launch per object-bitset width (OW) group
// of the instances (their tables are OW-strided), and the successor kernels (write, pick) one per row-width bucket or
// one per group at its widest bucket (BucketLaunch). A batch's rows are put in a launch order grouped by instance (key
// [I]: the instance's rank by OW, bucket and id), so a warp's lanes mostly run one instance's matchers. Whether a
// schema's matcher forward checks is per instance (kinds [I, S]: 0 fixed order, 1 forward checking, 2 never
// applicable); a launch of a kind takes the union of the instances' schemas of that kind.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mymyr::cuda::detail
{
/// Empty if the multi-instance kernels run every instance of the table completely under both witness settings (every
/// schema on the device without conditional effects, frozen slots, and no axioms, or with `device_axioms` axioms the
/// device evaluates: ChunkGenerator::axioms_unsupported, for callers that close the derived atoms themselves); else
/// why not, for the first instance that does not ("instance i: ...").
[[nodiscard]] std::string multi_unsupported(const rl::TaskTable& table, bool device_axioms = false);

struct TableLaunch
{
    /// Launch-order keys [lo, hi): the instances of a launch group are the ranks of one OW and a range of buckets, so
    /// their keys are contiguous (a range launch's Multi::key_lo / key_hi).
    struct Keys
    {
        u32 lo = 0, hi = 0;
    };
    struct Launch
    {
        u32 ow = 1;
        u32 lo = 0, hi = 2;  // the rows whose instance has state words in (lo, hi]
        u32 wb = 2;          // the kernels' WB (>= hi)
        Keys keys;           // its instances' keys
    };
    std::vector<u32> ows;        // the instances' object-bitset widths, ascending
    std::vector<Keys> ow_keys;   // [ows] the keys of each OW group's instances
    std::vector<Launch> widest, per_bucket;
    std::vector<u32> host_key;  // [I] the launch-order key of each instance: its rank, in [0, I)
    DeviceBuffer key;           // [I] u32
    struct Kinds
    {
        DeviceBuffer kinds;  // [I, S] u8: the device's matcher (rl::dev::device_fc) 0 fixed, 1 forward checking, 2 none
        DeviceBuffer sets;   // [fixed union | fc union] u32
        u32 n_fixed = 0, n_fc = 0;
        bool sorts = false;  // some instance sorts segments under canonical order
    };
    Kinds w[2][2];  // [witness pruning ? 0 : 1][canonical order ? 0 : 1]
    u32 instances = 0, num_schemas = 0;
    u64 view_words = 0;  // the widest instance's view words (the row width of a batch's views)

    [[nodiscard]] const std::vector<Launch>& launches(BucketLaunch b) const
    {
        return b == BucketLaunch::Widest ? widest : per_bucket;
    }
    /// The union schema set of a kind (fc: forward checking).
    [[nodiscard]] lifted::SchemaSet set(bool witness, bool canonical, bool fc) const;
    /// The instances of a launch over rows with task ids `inst` (no order, every OW and bucket: set ow / words_lo /
    /// words_hi / order per launch).
    [[nodiscard]] lifted::Multi multi(const DeviceTaskTable& dt, bool witness, bool canonical, const i32* inst) const;
    /// Declares stream `s` a user of the plan's device buffers.
    void record_stream(cudaStream_t s);
    /// The launch-order keys (lifted::launch_order).
    [[nodiscard]] lifted::OrderKeys order_keys() const
    {
        return {static_cast<const u32*>(key.data()), instances ? instances : 1};
    }
};

/// The plan of a table the multi-instance kernels run (multi_unsupported empty); uploads on stream `s`. `num_schemas`
/// (0: the table's): the segments per state of the launches; more than the table's lays the plan out for a task suite's
/// count cache (S = the widest domain's), the schemas past the table's never applicable (kind 2, count 0).
[[nodiscard]] TableLaunch plan_launches(const ContextPtr& ctx, const DeviceTaskTable& dt, cudaStream_t s,
                                        u32 num_schemas = 0);

/// True while stream s captures a CUDA graph (cudaStreamBeginCapture, torch.cuda.graph, XLA's command buffers).
[[nodiscard]] bool capturing(cudaStream_t s);

/// Debug builds (NDEBUG not defined): throws std::logic_error unless order / pos (at `stride`) are lifted::launch_order's
/// order of the batch (lifted::launch_check_order_keys: a permutation with its inverse, stable by key; synchronizes s;
/// nothing while s captures a graph). Release builds: nothing.
void check_order(const ContextPtr& ctx, const u32* inst, u32 instances, lifted::OrderKeys keys, u64 n, const u32* order,
                 const u32* pos, u64 stride, cudaStream_t s);

/// The device's axiom evaluation over the rows of a set of instances (lifted::launch_encoding_axioms, the evaluator of
/// the device expansion and encoder): per instance its device view and the semi-naive skipping data of its axioms
/// (axiom_reads), the launch classes by object-bitset width (forward checking where a body of the class forward checks
/// on the device), and the derived words of a row (the widest instance's, at least 1). Its instances' axioms must be
/// within the device's evaluator (ChunkGenerator::axioms_unsupported empty).
struct AxiomClosure
{
    DeviceBuffer records, reads;  // [instances] lifted::EncodingAxiomInstance, and their reads
    std::vector<u32> ow;
    std::vector<u8> fc;
    u32 instances = 0;
    u32 words = 0;

    /// The instances g in [0, n): `device_view(g)` (device pointers) and `host_view(g)` (host pointers) of the arrays of
    /// task `task(g)`. Uploads on stream s.
    [[nodiscard]] static AxiomClosure create(const ContextPtr& ctx, u32 n,
                                             const std::function<rl::dev::TaskView(u32)>& device_view,
                                             const std::function<rl::dev::TaskView(u32)>& host_view,
                                             const std::function<const Task&(u32)>& task, cudaStream_t s);
    /// Rows [0, n) of `states` (rows `stride` words apart, `row_words` atom words; task ids `ids`, null: instance 0)
    /// closed under their instances' axioms: their derived words into derived [n, derived_stride] (0: `words`; at
    /// least `words`, the rest of a row zero), their matcher views (fluent and derived atoms; the instance's view words
    /// of each row) into views (rows `view_words` apart). Rows with a task id outside the instances, and rows at or past
    /// *live (device; null: none), are left as they are. `invalid` (one device word) is the evaluator's flag for
    /// instances it refuses (never set here). Stream-ordered on s, no synchronization: capturable.
    void run(const u64* states, u64 stride, u32 row_words, u64 n, const i32* ids, u64* views, u64 view_words,
             u64* derived, u32* invalid, cudaStream_t s, u32 derived_stride = 0, const u64* live = nullptr) const;
};

/// Concurrent launches over disjoint row groups (the OW groups and row-width buckets of a plan): group 0 runs on the
/// caller's stream, groups 1.. on auxiliary streams forked from it and joined back. The groups' kernels are
/// latency-bound over part of the rows, so they overlap; the results do not depend on it (disjoint rows). Under stream
/// capture the fork and join are the graph's edges (events recorded and waited on inside the capture), so a captured
/// fan-out is one graph with parallel branches; its streams must exist before the capture (reserve()).
class Fanout
{
public:
    /// The stream of group i (0: s).
    [[nodiscard]] cudaStream_t lane(cudaStream_t s, u32 i) const { return i == 0 ? s : m_aux[i - 1]->get(); }
    /// Creates the streams and events of groups [1, n) on the current device.
    void reserve(u32 n);
    /// Makes the streams of groups [1, n) wait for the work enqueued on s so far (reserve(n) if they do not exist).
    void fork(cudaStream_t s, u32 n);
    /// Makes s wait for the work enqueued on the streams of groups [1, n).
    void join(cudaStream_t s, u32 n);

private:
    std::vector<std::unique_ptr<Stream>> m_aux;
    std::vector<Event> m_done;
    Event m_forked;
};
}  // namespace mymyr::cuda::detail
