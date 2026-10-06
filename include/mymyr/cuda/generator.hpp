#pragma once
// ChunkGenerator: the host driver of the lifted successor kernels (cuda/lifted.hpp), shared by the device BrFS
// (cuda/brfs.hpp), the device expand (cuda/expand.hpp) and the multi-search IW (cuda/multi_iw.hpp). It expands a chunk
// of parent states that live on the device into label rows (schema, binding, parent) and successor rows, in canonical
// order:
//   - conditional effects: a schema's conditional effects run as sub-plans at the DFS leaf of the write kernel
//     (lifted::launch_write_ce): each condition matcher, with the schema's binding prebound, over the parent's view and
//     derived bitset;
//   - axioms: views() closes the parents' derived bitsets under the axiom strata on the device
//     (lifted::launch_axioms) and adds the derived atoms to the views, before the goal test, the counts and the rows;
//   - matchers deeper than lifted::k_max_depth or with arity above lifted::k_max_label: the deep kernels
//     (lifted::SchemaSet::deep) up to lifted::k_deep_depth parameters and label columns;
//   - the per-schema CPU fallback for what the kernels cannot run: matchers deeper than lifted::k_deep_depth (or deep
//     ones over object bitsets wider than lifted::k_deep_ow words, or with conditional effects), forward checking over
//     object bitsets wider than
//     lifted::k_max_fc_ow words, and conditional effects whose condition matcher has more than lifted::k_ce_depth
//     free parameters. The CPU engine
//     generates those schemas (Successors::set_schema_filter) and their rows are placed into their (state, schema)
//     segments on the device. Axioms fall back to the CPU evaluator as a whole when a body matcher
//     is beyond the kernels (host_axioms_reason()); the derived bitsets are then uploaded with the chunk;
//   - lazy slots: the device reads a snapshot of the slot table (the task's device export). Atoms the kernels meet
//     without a slot (successors' fluent atoms, derived atoms) are reported in a bitmap over canonical ids;
//     resolve_missing() interns them in id order (so the numbering does not depend on timing) and refreshes the upload,
//     and the caller reruns write(); views() does the same for derived atoms by itself. The host work above may intern
//     atoms too; begin() refreshes the upload whenever the slot counts grew.
// The CPU fallback needs the parents on the host (ChunkInput::host_states; needs_host()). Under frozen slots its
// work runs on the host's threads (slices of at least 64 parents, taken as threads finish, each thread with its task
// workspace; the slices' rows are concatenated in parent order, so nothing depends on the threads); lazy slots intern
// atoms as they generate, so one thread.
//
// One ChunkGenerator serves one stream and one thread at a time; its scratch is reused across chunks (grown only).

#include "mymyr/cuda/device_task.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace mymyr
{
class ThreadPool;
}

namespace mymyr::cuda
{
/// Grow-only device scratch on one stream.
class Scratch
{
public:
    /// At least `bytes` bytes on stream `s` (contents undefined after growth; a buffer replaced is freed on s).
    void* ensure(const ContextPtr& ctx, u64 bytes, cudaStream_t s);
    [[nodiscard]] void* data() const noexcept { return m_buf.data(); }
    [[nodiscard]] u64 size() const noexcept { return m_buf.size(); }
    void reset() noexcept { m_buf.reset(); }
    /// The buffer (and every later one) tracks its owner's LastUse (DeviceBuffer::track).
    void track(std::shared_ptr<const LastUse> last);

private:
    DeviceBuffer m_buf;
    std::shared_ptr<const LastUse> m_last;
};

/// Where the schemas of a task run (per witness setting: the matcher differs).
struct SchemaPlacement
{
    std::vector<u32> device;  // on the device (with or without conditional effects), by schema id; the matcher of a
                              // launch is rl::dev::device_fc of the export's flags
    std::vector<u8> host;     // per schema: 1 = CPU fallback
    u32 host_count = 0;
    std::vector<u8> ce;      // per schema: 1 = on the device with conditional effects (sub-plans at the leaf)
    u32 ce_count = 0;        // schemas with conditional effects on the device
    u32 host_ce_count = 0;   // schemas with conditional effects on the CPU fallback
    std::vector<u8> deep;  // per schema: 1 = on the device in the deep kernels (lifted::SchemaSet::deep)
    u32 deep_count = 0;
    // (schemas neither on the device nor on the host are never applicable: their counts are 0)
};

/// Counters of a ChunkGenerator: where the axioms ran and what they cost.
struct GeneratorStats
{
    double device_axiom_ms = 0;  // device time of the axiom kernels (views()), when timing is on
    double host_axiom_ms = 0;    // CPU time on axioms: the host evaluator (fallback), and the CPU engine's axioms for
                                 // the parents of CPU-fallback schemas (wall time)
    u64 axiom_chunks = 0;        // chunks whose derived atoms the device computed (reruns included)
    u64 axiom_reruns = 0;        // lazy slots: evaluations redone after derived atoms were interned
    u64 host_rows = 0;           // rows generated by the CPU fallback
};

struct ChunkInput
{
    const u64* states = nullptr;  // device rows
    u64 stride = 0;               // words between rows
    u32 words = 0;
    u32 rows = 0;
    const u64* host_states = nullptr;  // host copy (needed when needs_host())
    u64 host_stride = 0;
    u32 parent_base = 0;  // label parent = parent_base + row
    // The live rows as a device count (null: all `rows`); `rows` is then the capacity the launches are sized for,
    // and rows at or past the count are absent (lifted::Parents::rows_dev). Not with host_states (the CPU fallback
    // needs the count on the host).
    const u32* rows_dev = nullptr;
};

class ChunkGenerator
{
public:
    /// `stream` null: the context's stream. Throws std::invalid_argument for tasks the device cannot run at all
    /// (supported() explains why).
    ChunkGenerator(ContextPtr ctx, TaskPtr task, cudaStream_t stream = nullptr);
    ChunkGenerator(const ChunkGenerator&) = delete;
    ChunkGenerator& operator=(const ChunkGenerator&) = delete;
    ~ChunkGenerator();

    /// Empty if the device can run the task (with CPU fallback for some schemas), otherwise the reason: numeric
    /// fluents, object bitsets wider than lifted::k_max_ow words, or states wider than lifted::k_max_words words.
    [[nodiscard]] static std::string unsupported(const Task& task);

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] const TaskPtr& task() const noexcept { return m_task; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_s; }
    /// Moves the following work to stream `s` (null: the context's stream), after the work enqueued on the old one (after
    /// the last mark_last_use() once there was one: the old stream may be gone then).
    void set_stream(cudaStream_t s);
    /// The generator's work so far ends on its stream here (its owner calls it at the end of each call on a
    /// caller's stream): its buffers free after it on a live stream, and set_stream does not touch the old stream, so
    /// the caller may destroy its stream.
    void mark_last_use();
    [[nodiscard]] const SchemaPlacement& placement(bool witness) const noexcept { return m_place[witness ? 0 : 1]; }
    /// Where the schemas of `task` run under `witness` (without constructing a generator: no upload).
    [[nodiscard]] static SchemaPlacement place(const Task& task, bool witness);
    /// Why schema `schema` runs on the CPU fallback under `witness` (empty: on the device).
    [[nodiscard]] static std::string host_reason(const Task& task, u32 schema, bool witness);
    /// Whether chunks need the parents on the host (CPU-fallback schemas, or axioms on the CPU).
    [[nodiscard]] bool needs_host(bool witness) const noexcept
    {
        return (m_task->has_axioms() && !m_device_axioms) || placement(witness).host_count > 0;
    }
    /// Whether the axioms (if any) are evaluated on the device.
    [[nodiscard]] bool device_axioms() const noexcept { return m_device_axioms; }
    /// Why the axioms are evaluated on the CPU (empty: on the device, or no axioms).
    [[nodiscard]] const std::string& host_axioms_reason() const noexcept { return m_host_axioms_reason; }
    /// Device axioms: the strata that run flat (lifted::Strata: no body reads a head of its own stratum).
    [[nodiscard]] u32 flat_strata() const noexcept
    {
        return static_cast<u32>(std::count(m_strata_flat.begin(), m_strata_flat.end(), u8{1}));
    }
    /// Words of the derived bitsets of the current chunk (the device's; 0 without axioms).
    [[nodiscard]] u32 derived_words() const noexcept { return m_parents.derived ? m_parents.derived_words : 0; }
    [[nodiscard]] u32 num_schemas() const noexcept { return m_S; }
    /// Whether some device schema sorts its segments under canonical order (deep matchers, and matchers that do not bind
    /// in parameter order): long segments are sorted in the label rows, or in Labels::scratch when no binding rows are
    /// written.
    [[nodiscard]] bool sorts(bool witness) const noexcept { return m_sorts[witness ? 0 : 1]; }
    /// Label columns of the kernels' rows (the largest schema arity, at least 1).
    [[nodiscard]] u32 label_width() const noexcept { return m_L; }
    /// Words per state view.
    [[nodiscard]] u64 view_words() const noexcept { return m_view_words; }

    /// The current device view of the task (refreshed by begin() and resolve_missing()).
    [[nodiscard]] const rl::dev::TaskView& view() const noexcept { return m_view; }
    [[nodiscard]] const DeviceTaskPtr& device_task() const noexcept { return m_dt; }
    /// Re-exports and uploads the task if its atom slots grew since the last upload (lazy slots); true if it did.
    bool refresh();
    /// Uploads since construction (1 + the refreshes).
    [[nodiscard]] u32 uploads() const noexcept { return m_uploads; }

    // ------------------------------------------------------------------------------------------ chunk phases
    /// Host work of a chunk (CPU-fallback rows, derived bitsets when the axioms run on the CPU; may intern atoms), the
    /// refresh, and the uploads.
    void begin(const ChunkInput& in, bool witness, bool canonical);
    /// begin() and count() in two steps each, so that the host work of a chunk runs while the device
    /// computes its views and counts: begin_device() takes the chunk without the host work; views(), goal_count() and
    /// count_device() may follow; then host_work() (the calling thread and the host pool) and count_host() complete
    /// it. Only where the views do not need the host work and it interns nothing: can_defer_host() (frozen slots, no
    /// axioms on the CPU).
    [[nodiscard]] bool can_defer_host() const noexcept;
    void begin_device(const ChunkInput& in, bool witness, bool canonical);
    void host_work();
    /// The views of the chunk's parents (after begin()), with their derived atoms: on the device, the axiom strata close
    /// the derived bitsets (parents().derived) and add them to the views. Lazy slots: derived atoms without a slot are
    /// interned in canonical-id order and the views and axioms rerun (a synchronization per chunk).
    void views();
    [[nodiscard]] const lifted::Parents& parents() const noexcept { return m_parents; }
    /// The goal test of the chunk's parents into out[0] (count, added) and out[1] (first goal row, min): device u32s.
    void goal_count(u32* out);
    /// Counts and scans: seg_offsets() [rows * S + 1] holds every (state, schema) segment's first row; the last entry
    /// is the chunk's total.
    void count();
    /// count() in two steps (begin_device() above): the device schemas' counts, then the host work's counts
    /// and the scan.
    void count_device();
    void count_host();
    [[nodiscard]] const u32* seg_offsets() const noexcept { return static_cast<const u32*>(m_offsets.data()); }
    /// Writes the rows [0, labels.capacity): labels (the arrays that are set) and successor rows into `words`
    /// (out_words wide; null: none), the device schemas' by the kernels, the CPU-fallback rows placed. words_needed
    /// (device u32, may be null) gets the widest successor that does not fit out_words (such rows are zero).
    void write(const lifted::Labels& labels, u64* words, u32 out_words, u32* words_needed = nullptr);
    /// Lazy slots: after write(), whether atoms without a slot were met (synchronizes the stream). If so they are
    /// interned in canonical order and the upload is refreshed: rerun write() (the task's width may have grown).
    /// Always false in frozen mode (no synchronization).
    bool resolve_missing();
    /// Lazy slots: the device flag write() and views() set when they meet atoms without a slot (null in frozen mode):
    /// a chunk that must not synchronize checks it on the device and calls resolve_missing() after it saw it set.
    [[nodiscard]] const u32* missing_flag() const noexcept
    {
        return m_missing_words ? static_cast<const u32*>(m_missing.data()) + m_missing_words : nullptr;
    }
    /// Host-side facts of the chunk's CPU-fallback rows: how many, and the widest successor among them (exact trimmed
    /// width; rows wider than write()'s out_words were zeroed, as rl::expand does).
    [[nodiscard]] u64 host_rows() const noexcept { return m_host_segment.size(); }
    [[nodiscard]] u32 host_words_needed(u32 out_words) const noexcept;

    /// Goal flags of n state rows on the device, out[i] = 0/1 (device u8): row i is row order[i] of `rows` (row i when
    /// order is null; a zero row when order[i] >= row_count), rows of `words` words, `stride` apart. Goals with derived
    /// literals evaluate the axioms of the rows on the device (views and axioms in chunks, in scratch of their own: the
    /// chunk's views stay). Needs device_axioms() or a goal without derived literals.
    void goal_flags(const u64* rows, u64 stride, u32 words, u64 row_count, const u32* order, u64 n, u8* out);

    /// Records the device time of the axiom kernels (events per chunk; GeneratorStats::device_axiom_ms).
    void set_timing(bool on) noexcept { m_timing = on; }
    /// The deep kernels' first-pass budget in search nodes per segment (lifted::SchemaSet::budget; 0: theirs).
    /// A segment past it is searched by several warps in the second pass: tests set 1 to send every segment there.
    void set_deep_budget(u32 nodes) noexcept { m_deep_budget = nodes; }
    /// The counters so far (synchronizes the stream when timing is on).
    [[nodiscard]] GeneratorStats stats();

    // ------------------------------------------------------------------------------------------ capture
    /// Whether a chunk's launches (begin() .. write()) can be captured into a CUDA graph: no host work and no
    /// synchronization inside (the CPU fallback, host axioms and derived atoms under lazy slots have them).
    [[nodiscard]] bool capturable(bool witness) const noexcept
    {
        return !needs_host(witness) && !(m_device_axioms && m_task->has_axioms() && m_missing_words);
    }
    /// Sizes the scratch for chunks of up to `rows` parents and refreshes the upload, so that begin() .. write() of
    /// such a chunk allocate nothing (a capture must not).
    void prepare(u32 rows);
    /// A fingerprint of what the chunk launches embed (scratch addresses, the uploaded task): a captured chunk replays
    /// only while it is unchanged.
    [[nodiscard]] u64 capture_key() const noexcept;

private:
    void upload();
    /// The schema lists, sorts() and the axioms' matcher kinds from the current export.
    void build_lists();
    void host_work(const ChunkInput& in, bool witness, bool canonical);
    /// The CPU-fallback rows of write(): labels and words at their segments' offsets.
    void place_host(const lifted::Labels& labels, u64* words, u32 out_words);
    /// Views of `p` into `views` and, with device axioms, the derived bitsets into `derived` (reruns after interning
    /// derived atoms under lazy slots). Returns the derived words (0 without device axioms).
    u32 views_and_axioms(const lifted::Parents& p, u64* views, Scratch& derived, u64** derived_out);
    /// Adds the recorded axiom event pairs to m_stats and reuses them (waits for the last one).
    void fold_axiom_events();

    ContextPtr m_ctx;
    TaskPtr m_task;
    cudaStream_t m_s;
    u32 m_S = 0, m_L = 1;
    u64 m_view_words = 0;
    SchemaPlacement m_place[2];
    bool m_sorts[2] = {false, false};
    // device schema lists per witness setting and launch order (rl::dev::MatchOrder: count() takes Free or Matcher,
    // write() Witness in place of Free when it writes binding labels under witness pruning): [fixed | fixed with CEs |
    // fc | fc with CEs | deep fixed | deep fc] by the device's matcher (rl::dev::device_fc; count() launches the first
    // two and the next two together). Built from the export's flags at every upload.
    struct Lists
    {
        u64 at = 0;
        u32 fixed = 0, fixed_ce = 0, fc = 0, fc_ce = 0, deep = 0, deep_fc = 0;
    };
    Lists m_lists[2][3];
    u32* deep_work(const Lists& L);  // the deep launches' scratch (null without deep schemas in L)
    // axioms
    bool m_device_axioms = false;
    bool m_axiom_fc = false;
    std::string m_host_axioms_reason;
    DeviceBuffer m_axiom_reads;  // [axioms, 2] (lifted::Derived::reads)
    std::vector<u32> m_strata_begin, m_strata_count;  // lifted::Strata
    std::vector<u8> m_strata_flat;
    Scratch m_dev_derived, m_goal_views, m_goal_derived, m_goal_rows;
    bool m_timing = false;
    u32 m_deep_budget = 0;
    std::vector<Event> m_axiom_events;  // pairs, folded into m_stats every k_axiom_events (bounded host memory)
    usize m_axiom_event_next = 0;
    GeneratorStats m_stats;
    DeviceTaskPtr m_dt;
    rl::dev::TaskView m_view{};
    u32 m_exported_fluent = 0, m_exported_derived = 0;
    u32 m_uploads = 0;
    bool m_witness = true, m_canonical = true;
    ChunkInput m_deferred{};      // begin_device()'s chunk, until host_work()
    bool m_host_pending = false;  // begin_device() ran, host_work() not yet

    lifted::Parents m_parents{};
    Scratch m_views, m_counts, m_offsets, m_scan, m_derived, m_schema_sets;
    Scratch m_put_mask;  // [2, S] u8 per witness setting: the schema's rows are deferred (launch_put)
    Scratch m_deep_work;  // the deep launches' scratch (lifted::SchemaSet::work, deep_work_words)
    Scratch m_missing;  // bitmap over canonical ids + flag
    u64 m_missing_words = 0;
    Scratch m_h_seg, m_h_rank, m_h_bind, m_h_words, m_h_counts_seg, m_h_counts;
    PinnedLease m_flag;  // resolve_missing's read (a u32), leased from the context's cache
    std::shared_ptr<LastUse> m_last;  // mark_last_use(); every scratch tracks it

    // host rows of the current chunk
    std::vector<u64> m_host_derived;
    std::vector<u32> m_host_segment, m_host_rank, m_host_binding;
    std::vector<u32> m_host_need, m_host_exact;  // per row: words to fit (parent width, adds), exact successor width
    std::vector<u64> m_host_words;               // per row: m_host_row_words words
    u32 m_host_row_words = 0;
    std::vector<u32> m_host_count_seg, m_host_count;
    std::vector<u64> m_staging;
    // The CPU fallback's threads (frozen slots), started at the first chunk that has the work for two
    u32 m_host_threads = std::max<u32>(1, std::thread::hardware_concurrency());
    std::unique_ptr<ThreadPool> m_pool;
};
}  // namespace mymyr::cuda
