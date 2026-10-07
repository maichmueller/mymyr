#pragma once
// The environment step on the device: rl::HostEnv's step over a task table or a task suite of several domains
// (rl/env.hpp: same semantics, same RNG streams, same results bit for bit) for environment batches that live in
// device memory. Destination-passing: every array of rl::EnvBatch and rl::StepOutputs is device memory the caller owns; the work
// runs on one stream (set_stream; null: the context's stream) in stream order.
//
// Two paths, chosen per table or suite (fast_unsupported() says why the fast one does not apply):
//   - fast: no host round trip in a step. The batch carries a per-environment cache of its current states, the count
//     cache [rows, cache_schemas()] (lifted::CountCache: the successor counts per schema, a fingerprint of the state,
//     the packed canonical keys of the bindings of the schemas whose segments are sorted and, over a table of several
//     instances, two columns of the launch order that groups the rows by instance) and the matcher views [rows, V]
//     (V: the widest instance's view; EnvBatch::counts / views; reset(), refresh() and step() keep them current). A
//     step is select (the index from the action or the RNG, then its (schema, rank) from the cached counts), pick (the
//     rank-th binding of that one segment in canonical order, read from the cached keys, or found by the segment's
//     matcher where nothing is recorded; written over the parent row), views and count caches of the reached states
//     (in L2-sized chunks), and finish (goal test, rewards, termination, truncation, autoreset). No synchronization,
//     no allocation after the first step. Over several instances the multi-instance kernels run the same steps, each
//     thread resolving its row's instance; the launches go per object-bitset width group and row-width bucket
//     (set_launch; the groups concurrently, on auxiliary streams of the env forked from its stream and joined back
//     into it by events), in the cached launch order (recomputed by reset(), refresh() and steps with next_task_ids; a
//     batch rearranged since then is detected and runs in batch order: the same results, slower). A step's select also
//     finds where each instance's rows start in that order, so the step's launches take the positions of their own
//     groups' rows only (range launches, lifted::Multi::starts). Needs: every
//     instance runs completely on the fast kernels (no numeric fluents, axioms, CPU-fallback schemas or conditional effects),
//     frozen atom slots, and packed canonical-order keys within 64 bits (lifted::pick_keys_fit). Over a suite of
//     several domains every domain runs the multi-instance kernels of its table (detail::multi_unsupported
//     empty): the launch order groups the rows by domain, then as each table's; each domain's picks and counts take its
//     rows' positions only and run on a stream of their own (forked from the env's stream and joined back by events);
//     select, the views (one launch over every domain's rows) and finish run once over the batch (the count cache is
//     laid out with the suite's widest schema count, a domain's schemas past its own counting 0);
//   - general: tables with numeric fluents, CPU-fallback schemas, conditional effects, axioms or lazy slots. A
//     step expands the current states into internal flat buffers with SuiteExpander (a table's DeviceExpander, or one
//     per domain; canonical order, goal flags; it synchronizes), moves every environment to its chosen row, and counts
//     the reached states' successors with a second expansion. The cache arrays are not used.
// The results do not depend on the path, the chunk sizes, the bucket launches or the launch configuration, and an
// environment's trajectory does not depend on the batch it runs in (its RNG stream is its id, EnvBatch::first_env +
// row, or env_ids[row]).
//
// CUDA graphs: on the fast path, reset(), refresh() and step() may be captured into a CUDA graph (stream capture
// of the env's stream: cudaStreamBeginCapture, torch.cuda.graph, torch.compile's CUDA graph trees, XLA's command
// buffers) and replayed: a call launches the same kernels with the same shapes for a given batch size and table (the
// auxiliary streams of a table's launch groups fork from the env's stream and join back into it by events, so a
// capture sees one stream graph), synchronizes nothing and allocates nothing once its scratch is sized for the batch
// (reserve(), or one call of that size before the capture). Everything a step reads is device memory: the batch's
// arrays, the env's constants and scratch, and the RNG key (set_seed() writes it on the device, so replays of graphs
// captured before follow it). A replay equals the calls it captured, byte for byte, on whatever the batch's arrays
// hold then. The general path synchronizes on the successor count (capture_unsupported() says so). Scratch that grows
// after a capture keeps its old buffers alive until the env is destroyed, since graphs captured before still use them;
// a graph must not outlive its env.
//
// One DeviceEnv serves one thread at a time; its scratch is reused across calls.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <string>

namespace mymyr::cuda
{
class DeviceEnv
{
public:
    enum class Path : u8
    {
        Auto,     // fast when the table or suite allows it
        Fast,     // throws std::invalid_argument when it does not
        General,  // always the flat expand
    };

    /// Throws std::invalid_argument for suites the device cannot run (a domain's
    /// DeviceTaskTable::unsupported, rows wider than lifted::k_max_words).
    DeviceEnv(ContextPtr ctx, rl::TaskSuitePtr suite, const rl::EnvConfig& config, Path path = Path::Auto,
              cudaStream_t stream = nullptr);
    /// Over a table: the suite of its one domain (TaskSuite::of).
    DeviceEnv(ContextPtr ctx, const rl::TaskTablePtr& table, const rl::EnvConfig& config, Path path = Path::Auto,
              cudaStream_t stream = nullptr)
        : DeviceEnv(std::move(ctx), rl::TaskSuite::of(table), config, path, stream)
    {
    }
    ~DeviceEnv();
    DeviceEnv(const DeviceEnv&) = delete;
    DeviceEnv& operator=(const DeviceEnv&) = delete;

    /// Empty if the fast path runs `suite`, else the reason (for the first instance it does not run; over several
    /// domains "domain d (name): ...").
    [[nodiscard]] static std::string fast_unsupported(const rl::TaskSuite& suite, const rl::EnvConfig& config);
    [[nodiscard]] static std::string fast_unsupported(const rl::TaskTable& table, const rl::EnvConfig& config)
    {
        return fast_unsupported(*rl::TaskSuite::of(table), config);
    }

    [[nodiscard]] const ContextPtr& context() const noexcept;
    /// The suite of the env's instances (over a table: the suite of its one domain, TaskSuite::of).
    [[nodiscard]] const rl::TaskSuitePtr& suite() const noexcept;
    [[nodiscard]] const rl::EnvConfig& config() const noexcept;
    /// The RNG key of the random policy (EnvConfig::seed), on the host and on the device (stream-ordered: steps
    /// enqueued before keep the old key; graphs replayed after take the new one).
    void set_seed(u64 seed);
    [[nodiscard]] bool fast() const noexcept;
    /// Empty if reset(), refresh() and step() can be captured into a CUDA graph (the fast path), else why not.
    [[nodiscard]] std::string capture_unsupported() const;
    /// Sizes the scratch of calls over batches of up to `rows` rows (a call of more rows grows it; a captured call
    /// cannot).
    void reserve(u64 rows);
    /// The stream of the following calls (null: the context's stream). A stream that is capturing is taken without
    /// a wait on the previous stream (the capture was begun after the work it depends on).
    void set_stream(cudaStream_t s);
    [[nodiscard]] cudaStream_t stream() const noexcept;
    /// Rows per chunk of the fast path's views and counts, or per chunk of the general path's DeviceExpander (0:
    /// automatic). The results do not depend on it.
    void set_chunk_rows(u64 rows) noexcept;
    /// How the fast path's pick (and the general path's write) is launched over the row-width buckets of a table of
    /// several instances, or of each domain of a suite (defaults: the pick Widest, the general path's write the
    /// expander's PerBucket; BucketLaunch).
    /// The results do not depend on it.
    void set_launch(BucketLaunch launch) noexcept;

    /// Row width in atom words (TaskSuite::words(): the widest domain's).
    [[nodiscard]] u32 words() const noexcept;
    /// Numeric words per row, in the CPU encoding of each instance.
    [[nodiscard]] u32 numeric_words() const noexcept { return suite()->numeric_words(); }
    [[nodiscard]] u32 row_words() const noexcept { return words() + numeric_words(); }
    /// Largest schema arity (at least 1): the label width of step().
    [[nodiscard]] u32 label_width() const noexcept;
    /// Columns of the fast path's cache arrays: EnvBatch::counts [rows, cache_schemas()] (u32 words of the count cache,
    /// lifted::CountCache: the counts per schema come first, TaskSuite::max_schemas() of them; then, over several
    /// instances, the two launch-order columns) and views [rows, cache_view_words()]; 0 on the general path.
    [[nodiscard]] u32 cache_schemas() const noexcept;
    [[nodiscard]] u64 cache_view_words() const noexcept;
    /// Schemas whose segments the count cache records (lifted::CountCache regions, the most of any instance; 0: none,
    /// or the general path): sorted segments whose canonical ranks fit a region (their rank bitmap) or of deep
    /// matchers (their keys), whose picks then read the region instead of searching again.
    [[nodiscard]] u32 cache_regions() const noexcept;
    /// Successors of an instance's initial state.
    [[nodiscard]] u32 initial_count(u32 instance) const;

    /// rl::HostEnv::reset on device arrays; mask [rows] (device, null: all rows). The rows' task ids (device) are the
    /// caller's (global ids of the suite); an id outside it resets the row into instance 0 and is reported by
    /// check_errors().
    void reset(rl::EnvBatch& b, const u8* mask = nullptr, i32* count = nullptr, bool keep_goals = false);
    /// Recomputes the cache (fast path: counts, views, launch order) and the successor counts of the current states,
    /// after the caller wrote states or task ids.
    void refresh(rl::EnvBatch& b, i32* count = nullptr);
    /// rl::HostEnv::step on device arrays; action [rows] (device, none: the random policy); next_task_ids [rows]
    /// (device, optional: the instance an autoresetting row restarts in, of any domain of the suite; an id outside it
    /// keeps the row's instance and is reported by check_errors()).
    void step(rl::EnvBatch& b, const rl::StepOutputs& out, rl::Actions action = {}, const i32* next_task_ids = nullptr);
    /// The random policy's next choices without a step (TorchRL's rand_action): out[i] = the successor index step()
    /// would draw now for row i (draws, count [rows] device; the env's key, env id first_env + i), limited to
    /// max_actions (0: no limit), 0 where count[i] is 0.
    void random_actions(const u64* draws, const i32* count, u64 rows, u64 first_env, u32 max_actions, i64* out);
    /// Synchronizes the stream; throws std::logic_error if a fast-path step found a state whose cache did not match it
    /// (states written without refresh()), std::invalid_argument for task ids outside the suite (reset, next_task_ids).
    /// Not during a capture (the flags of replayed graphs accumulate until the next check).
    void check_errors();

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::cuda
