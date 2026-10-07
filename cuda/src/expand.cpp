// The device expand (include/mymyr/cuda/expand.hpp): the single-instance path (ChunkGenerator), the one-pass
// multi-instance path (the multi-instance kernels, table_launch.hpp) and the per-instance path (parts expanded apart by
// single-instance expanders and scattered back).

#include "mymyr/cuda/expand.hpp"

#include "numeric_pad.hpp"

#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/numeric_kernels.hpp"
#include "table_launch.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::cuda
{
namespace
{
constexpr u64 k_i32_max = static_cast<u64>(std::numeric_limits<i32>::max());
/// Views of one chunk at most (larger batches are expanded in chunks of parents).
constexpr u64 k_view_budget = u64{256} << 20;

// control words (device and pinned host)
constexpr u32 k_bad_row = 0, k_words_needed = 1, k_error = 2, k_overflow = 3, k_bad_id = 4, k_ctl_words = 5;

template<class T>
T* scratch_of(Scratch& sc, const ContextPtr& ctx, u64 n, cudaStream_t s)
{
    return static_cast<T*>(sc.ensure(ctx, std::max<u64>(n, 1) * sizeof(T), s));
}
}  // namespace

struct DeviceExpander::Impl
{
    ContextPtr ctx;
    rl::TaskTablePtr table;
    Mode mode = Mode::Single;
    BucketLaunch launch = BucketLaunch::PerBucket;
    cudaStream_t s = nullptr;
    u64 chunk_rows = 0;  // set_chunk_rows
    Scratch ctl, sort, numeric_in, numeric_out;
    // an expansion's rows apart from the caller's outputs: Multi's labels of deferred successors the caller does not
    // take, or of a mapped write; Single's and PerInstance's expansion before it is scattered to the caller's rows
    Scratch t_succ, t_schema, t_binding, t_parent, t_goal, t_offsets;
    PinnedBuffer host_ctl{k_ctl_words * sizeof(u32)};
    PinnedBuffer totals;  // per chunk

    struct Chunk
    {
        u64 first = 0, rows = 0, total = 0, base = 0;
        u64 seg_base = 0;  // Multi: the chunk's segment offsets in `seg`
    };
    rl::StateBatchView in{};
    rl::ExpandOptions opt{};
    const i32* ids = nullptr;
    u64 stride = 0;
    std::vector<Chunk> chunks;
    u64 total = 0;
    bool counted = false;
    bool kept = false;  // one chunk: its views and segment offsets are still there

    // Single
    std::unique_ptr<ChunkGenerator> gen;
    std::vector<u64> host_states;
    // Multi
    DeviceTaskTablePtr dt;
    detail::TableLaunch plan;
    DeviceBuffer limits;  // [I] the instances' fluent slots (frozen)
    Scratch views, counts, seg, scan, order_temp, order, pos;
    detail::Fanout fan;  // the launches over the plan's row groups, concurrently
    bool segs_kept = false;  // every chunk's launch order and segment offsets are still there (count() ran)
    i64 views_chunk = -1;    // the chunk whose views are in `views`
    // PerInstance
    struct Part
    {
        u32 instance = 0;
        u64 start = 0, rows = 0, total = 0, base = 0;
    };
    std::vector<std::unique_ptr<DeviceExpander>> subs;  // per instance, made on first use
    std::vector<Part> parts;
    std::vector<i32> host_ids;
    Scratch part_limits, index, gathered, row_counts, batch_offsets;

    [[nodiscard]] u32* dctl() { return scratch_of<u32>(ctl, ctx, k_ctl_words, s); }
    [[nodiscard]] u32* hctl() const { return static_cast<u32*>(host_ctl.data()); }

    /// Zeroes the control words; bad_row and bad_id start at "none".
    u32* reset_ctl()
    {
        u32* d = dctl();
        check(cudaMemsetAsync(d, 0, k_ctl_words * sizeof(u32), s), "cudaMemsetAsync");
        check(cudaMemsetAsync(d + k_bad_row, 0xFF, sizeof(u32), s), "cudaMemsetAsync");
        check(cudaMemsetAsync(d + k_bad_id, 0xFF, sizeof(u32), s), "cudaMemsetAsync");
        return d;
    }

    /// After the control words were copied down and the stream synchronized: the validation errors of count().
    void throw_bad_rows()
    {
        const u32 bad_id = hctl()[k_bad_id], bad_row = hctl()[k_bad_row];
        if (bad_id != 0xFFFFFFFFu)
        {
            i32 v = 0;
            check(cudaMemcpy(&v, ids + bad_id, sizeof(i32), cudaMemcpyDeviceToHost), "cudaMemcpy");
            throw std::invalid_argument("mymyr: task id " + std::to_string(v) + " of row " + std::to_string(bad_id) +
                                        " is outside the table's " + std::to_string(table->size()) + " instances");
        }
        if (bad_row != 0xFFFFFFFFu)
            throw std::invalid_argument("mymyr: expand: state row " + std::to_string(bad_row) +
                                        " sets atom slots its instance has not assigned (a state of another task?)");
    }

    /// Chunks of parents: the views (view_words each) within the budget, the (state, schema) segments within int32.
    void make_chunks(u64 view_words, u32 num_schemas)
    {
        const u64 S = std::max<u32>(num_schemas, 1);
        u64 per = std::max<u64>(1, std::min<u64>(k_view_budget / std::max<u64>(8, view_words * 8), (u64{1} << 30) / S));
        if (chunk_rows)
            per = std::min(per, chunk_rows);
        chunks.clear();
        for (u64 b = 0; b < in.rows; b += per)
            chunks.push_back({b, std::min(per, in.rows - b), 0, 0});
        if (totals.size() < std::max<usize>(chunks.size(), 1) * sizeof(u32))
            totals = PinnedBuffer(std::max<usize>(chunks.size(), 1) * sizeof(u32));
    }

    /// The chunks' totals (copied down, the stream synchronized) into chunks[].total / base and `total`.
    void sum_chunks()
    {
        const auto* tot = static_cast<const u32*>(totals.data());
        total = 0;
        for (usize k = 0; k < chunks.size(); ++k)
        {
            chunks[k].total = tot[k];
            chunks[k].base = total;
            total += tot[k];
        }
    }

    [[nodiscard]] u32 label_columns(const rl::Expansion& out) const
    {
        return out.binding ? out.label_width : std::max<u32>(1, table->label_width());
    }

    // ------------------------------------------------------------------------------------------ Single
    ChunkInput input(const Chunk& c) const
    {
        ChunkInput x;
        const u32 slots = table->task(0)->numeric_slots();
        const u32 row_words = in.words + slots;
        x.states = slots ? static_cast<const u64*>(numeric_in.data()) + c.first * row_words : in.data + c.first * stride;
        x.stride = slots ? row_words : stride;
        x.words = row_words;
        x.rows = static_cast<u32>(c.rows);
        x.host_states = host_states.empty() ? nullptr : host_states.data() + c.first * row_words;
        x.host_stride = row_words;
        x.parent_base = static_cast<u32>(c.first);
        return x;
    }

    void prepare_single(const Chunk& c)
    {
        gen->begin(input(c), opt.witness_pruning, opt.canonical_order);
        gen->views();
        gen->count();
    }

    u64 count_single()
    {
        u32* d = reset_ctl();
        const u32 slots = table->task(0)->numeric_slots(), row_words = in.words + slots;
        if (slots)
        {
            auto* rows = static_cast<u64*>(numeric_in.ensure(ctx, in.rows * row_words * sizeof(u64), s));
            check(numeric::launch_convert(gen->view(), in.data, stride, in.words, rows, row_words, in.words, in.rows, true, s),
                  "numeric input");
        }
        // the parents on the host for the CPU fallback and the axioms
        host_states.clear();
        if (gen->needs_host(opt.witness_pruning) && in.rows)
        {
            host_states.resize(in.rows * row_words);
            if (row_words)
                check(cudaMemcpy2DAsync(host_states.data(), row_words * sizeof(u64),
                                        slots ? numeric_in.data() : in.data, (slots ? row_words : stride) * sizeof(u64),
                                        row_words * sizeof(u64), in.rows, cudaMemcpyDeviceToHost, s),
                      "cudaMemcpy2DAsync (parents)");
        }
        if (ids)
            check(lifted::launch_check_ids(ids, 1, in.rows, d + k_bad_id, s), "launch_check_ids");
        // rows with bits past the assigned slots (rl::expand's validation)
        const u32 limit = table->task(0)->atoms().fluent_slots();
        if (opt.validate && in.rows && static_cast<u64>(in.words) * 64 > limit)
            check(lifted::launch_check_rows(in.data, stride, in.words, in.rows, limit, d + k_bad_row, s), "launch_check_rows");
        if (!host_states.empty())
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        make_chunks(gen->view_words(), gen->num_schemas());
        auto* tot = static_cast<u32*>(totals.data());
        for (usize k = 0; k < chunks.size(); ++k)
        {
            prepare_single(chunks[k]);
            check(cudaMemcpyAsync(tot + k, gen->seg_offsets() + chunks[k].rows * gen->num_schemas(), sizeof(u32),
                                  cudaMemcpyDeviceToHost, s),
                  "cudaMemcpyAsync");
        }
        check(cudaMemcpyAsync(hctl(), d, k_ctl_words * sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        throw_bad_rows();
        sum_chunks();
        return total;
    }

    void write_single(rl::Expansion& out)
    {
        const Task& task = *table->task(0);
        u32* d = dctl();
        check(cudaMemsetAsync(d + k_words_needed, 0, 2 * sizeof(u32), s), "cudaMemsetAsync");
        const u32 slots = task.numeric_slots(), W = out.words + slots, S = gen->num_schemas();
        u64* successor_rows = out.succ;
        if (slots && out.succ)
            successor_rows = static_cast<u64*>(numeric_out.ensure(ctx, std::min(total, out.capacity) * W * sizeof(u64), s));
        const u32 L = out.binding ? out.label_width : gen->label_width();
        // long segments that need sorting get an indexed scratch when no binding rows are written (or at the capacity)
        u64 widest = 0;
        for (const Chunk& c : chunks)
            widest = std::max(widest, c.total);
        u32* sc = nullptr;
        if (opt.canonical_order && gen->sorts(opt.witness_pruning) && widest)
            sc = static_cast<u32*>(sort.ensure(ctx, widest * L * sizeof(u32), s));
        if (out.offsets && chunks.empty())
            check(cudaMemsetAsync(out.offsets, 0, sizeof(i32), s), "cudaMemsetAsync");
        u32 host_need = 0;
        for (const Chunk& c : chunks)
        {
            if (!kept)
                prepare_single(c);
            const u64 base = c.base;
            lifted::Labels lab;
            lab.binding = out.binding ? reinterpret_cast<u32*>(out.binding) + base * L : nullptr;
            lab.schema = out.schema ? reinterpret_cast<u32*>(out.schema) + base : nullptr;
            lab.parent = out.parent ? reinterpret_cast<u32*>(out.parent) + base : nullptr;
            lab.capacity = base < out.capacity ? out.capacity - base : 0;
            lab.label_width = L;
            lab.parent_base = static_cast<u32>(c.first);
            lab.scratch = sc;
            lab.scratch_rows = sc ? c.total : 0;
            lab.scratch_indexed = 1;
            lab.error = d + k_error;
            u64* words = successor_rows && base < out.capacity ? successor_rows + base * W : nullptr;
            gen->write(lab, words, W, d + k_words_needed);
            while (gen->resolve_missing())
                gen->write(lab, words, W, d + k_words_needed);
            host_need = std::max(host_need, gen->host_words_needed(W));
            if (out.offsets)
                check(lifted::launch_state_offsets(gen->seg_offsets(), S, c.rows, base, out.offsets + c.first, s),
                      "launch_state_offsets");
        }
        // goal flags of the written successors
        const u64 n = std::min(total, out.capacity);
        if (out.goal && n)
        {
            if (slots && (!task.compiled().goal.uses_derived || gen->device_axioms()))
                gen->goal_flags(successor_rows, W, W, n, nullptr, n, out.goal);
            else if (!task.compiled().goal.uses_derived)
                check(lifted::launch_goal_rows(gen->view(), out.succ, W, n, nullptr, out.goal, s), "launch_goal_rows");
            else if (gen->device_axioms())
            {
                // goals over derived atoms: the successors' axioms on the device (lazy slots: may intern derived atoms,
                // which does not change the rows)
                gen->goal_flags(out.succ, W, W, n, nullptr, n, out.goal);
            }
            else
            {
                std::vector<u64> rows(n * W);
                std::vector<u8> flags(n);
                check(cudaMemcpyAsync(rows.data(), successor_rows, n * W * sizeof(u64), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
                check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
                for (u64 j = 0; j < n; ++j)
                {
                    const State decoded = numeric::decode(task, rows.data() + j * W, W);
                    flags[j] = static_cast<u8>(task.is_goal(decoded.view()));
                }
                check(cudaMemcpyAsync(out.goal, flags.data(), n, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
                check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // pageable source
            }
        }
        if (slots && out.succ)
            check(numeric::launch_convert(gen->view(), successor_rows, W, out.words, out.succ,
                                           out.words + out.numeric_words, out.words, n, false, s), "numeric output");
        finish_write(out, host_need);
    }

    /// The words_needed / error words of a write (synchronizes), out.total and out.words_needed.
    void finish_write(rl::Expansion& out, u32 extra_need)
    {
        u32* d = dctl();
        check(cudaMemcpyAsync(hctl() + k_words_needed, d + k_words_needed, 2 * sizeof(u32), cudaMemcpyDeviceToHost, s),
              "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        if (hctl()[k_error])
            throw std::logic_error("mymyr: device expand: a sorted segment found no scratch (internal error)");
        out.total = total;
        out.words_needed = std::max(hctl()[k_words_needed], extra_need);
        kept = chunks.size() == 1;  // one chunk: a second write() reuses its views and offsets
    }

    // ------------------------------------------------------------------------------------------ Multi
    [[nodiscard]] lifted::Multi multi_of(const Chunk& c) const
    {
        lifted::Multi m = plan.multi(*dt, opt.witness_pruning, opt.canonical_order, ids + c.first);
        m.order = static_cast<const u32*>(order.data()) + c.first;
        return m;
    }

    [[nodiscard]] lifted::Parents parents_of(const Chunk& c) const
    {
        return lifted::Parents{in.data + c.first * stride, stride, in.words, static_cast<u32>(c.rows), nullptr, 0};
    }

    /// The chunk's launch order, views, counts and segment offsets (views_only: its views, from the kept order). The
    /// orders and segment offsets of all chunks are kept (count() sizes them), so write() rebuilds only the views of
    /// the chunks whose views are gone.
    void prepare_multi(const Chunk& c, bool views_only)
    {
        const u32 S = plan.num_schemas;
        const u64 n = c.rows, nseg = n * S + 1;
        if (nseg > 0x7FFFFFFFull)
            throw std::length_error("mymyr: device expand: more than 2^31 (state, schema) segments in one chunk");
        auto* ord = scratch_of<u32>(order, ctx, in.rows, s) + c.first;
        auto* ps = scratch_of<u32>(pos, ctx, in.rows, s) + c.first;
        if (!views_only)
        {
            const u64 tb = lifted::order_temp_bytes(n);
            const auto* rid = reinterpret_cast<const u32*>(ids + c.first);
            check(lifted::launch_order(rid, plan.instances, plan.order_keys(), n, order_temp.ensure(ctx, tb, s), tb, ord, ps,
                                       1, s),
                  "launch_order");
            detail::check_order(ctx, rid, plan.instances, plan.order_keys(), n, ord, ps, 1, s);
        }
        const lifted::Parents p = parents_of(c);
        const lifted::Views v{scratch_of<u64>(views, ctx, n * plan.view_words, s), plan.view_words};
        lifted::Multi m = multi_of(c);
        check(lifted::launch_view_multi(m, p, v, s), "launch_view_multi");
        views_chunk = static_cast<i64>(&c - chunks.data());
        if (views_only)
            return;
        auto* cnt = scratch_of<u32>(counts, ctx, nseg, s);
        auto* sg = static_cast<u32*>(seg.data()) + c.seg_base;
        check(cudaMemsetAsync(cnt, 0, nseg * sizeof(u32), s), "cudaMemsetAsync");
        const lifted::SchemaSet fixed = plan.set(opt.witness_pruning, opt.canonical_order, false);
        const lifted::SchemaSet fc = plan.set(opt.witness_pruning, opt.canonical_order, true);
        // the OW groups concurrently
        const auto groups = static_cast<u32>(plan.ows.size());
        fan.fork(s, groups);
        for (u32 gi = 0; gi < groups; ++gi)
        {
            m.ow = plan.ows[gi];
            if (fixed.count)
                check(lifted::launch_count_multi(m, p, v, fixed, cnt, fan.lane(s, gi)), "launch_count_multi");
            if (fc.count && m.ow <= lifted::k_max_fc_ow)
                check(lifted::launch_count_multi(m, p, v, fc, cnt, fan.lane(s, gi)), "launch_count_multi (fc)");
        }
        fan.join(s, groups);
        const u64 stb = lifted::scan_temp_bytes(nseg);
        check(lifted::launch_scan(cnt, sg, nseg, scan.ensure(ctx, stb, s), stb, s), "launch_scan");
    }

    u64 count_multi()
    {
        u32* d = reset_ctl();
        check(lifted::launch_check_multi(static_cast<const u32*>(limits.data()), plan.instances, ids,
                                         opt.validate ? in.data : nullptr, stride, in.words, in.rows, d + k_bad_id,
                                         d + k_bad_row, s),
              "launch_check_multi");
        make_chunks(plan.view_words, plan.num_schemas);
        u64 nseg = 0;
        for (Chunk& c : chunks)
        {
            c.seg_base = nseg;
            nseg += c.rows * plan.num_schemas + 1;
        }
        (void)scratch_of<u32>(seg, ctx, nseg, s);
        auto* tot = static_cast<u32*>(totals.data());
        for (usize k = 0; k < chunks.size(); ++k)
        {
            prepare_multi(chunks[k], false);
            check(cudaMemcpyAsync(tot + k,
                                  static_cast<const u32*>(seg.data()) + chunks[k].seg_base + chunks[k].rows * plan.num_schemas,
                                  sizeof(u32), cudaMemcpyDeviceToHost, s),
                  "cudaMemcpyAsync");
        }
        segs_kept = true;
        check(cudaMemcpyAsync(hctl(), d, k_ctl_words * sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        throw_bad_rows();
        sum_chunks();
        return total;
    }

    void write_multi(rl::Expansion& out, const RowMap* map)
    {
        u32* d = dctl();
        check(cudaMemsetAsync(d + k_words_needed, 0, 2 * sizeof(u32), s), "cudaMemsetAsync");
        const u32 W = out.words, S = plan.num_schemas;
        // mapped: the labels in scratch at the table's label width (the put widens them at the batch rows)
        const u32 L = map ? std::max<u32>(1, table->label_width()) : label_columns(out);
        u64 widest = 0;
        for (const Chunk& c : chunks)
            widest = std::max(widest, c.total);
        u32* sc = nullptr;
        if (opt.canonical_order && plan.w[opt.witness_pruning ? 0 : 1][0].sorts && widest)
            sc = static_cast<u32*>(sort.ensure(ctx, widest * L * sizeof(u32), s));
        if (out.offsets && chunks.empty() && !map)
            check(cudaMemsetAsync(out.offsets, 0, sizeof(i32), s), "cudaMemsetAsync");
        const lifted::SchemaSet fixed = plan.set(opt.witness_pruning, opt.canonical_order, false);
        const lifted::SchemaSet fc = plan.set(opt.witness_pruning, opt.canonical_order, true);
        // the last chunk first: count() left its views
        for (usize ci = chunks.size(); ci-- > 0;)
        {
            const Chunk& c = chunks[ci];
            if (!segs_kept)
                prepare_multi(c, false);
            else if (views_chunk != static_cast<i64>(ci))
                prepare_multi(c, true);
            const u64 base = c.base;
            // the rows labeled here: the chunk's below the capacity at its first successor, or (mapped) all of them
            const u64 cap = map ? c.total : base < out.capacity ? std::min(out.capacity - base, c.total) : 0;
            lifted::SuccessorRows rows;
            rows.words = out.succ && cap ? (map ? out.succ : out.succ + base * W) : nullptr;
            rows.out_words = W;
            rows.words_needed = d + k_words_needed;
            // the successors after the labels, by launch_put_multi: the widest instance's long segments do not
            // bound the launch; the labels it reads are the caller's or scratch
            const bool defer = map || rows.words;
            rows.deferred = defer ? 1 : 0;
            lifted::Labels lab;
            lab.binding = !map && out.binding ? reinterpret_cast<u32*>(out.binding) + base * L
                          : defer             ? scratch_of<u32>(t_binding, ctx, cap * L, s)
                                              : nullptr;
            lab.schema = !map && out.schema ? reinterpret_cast<u32*>(out.schema) + base
                         : defer            ? scratch_of<u32>(t_schema, ctx, cap, s)
                                            : nullptr;
            lab.parent = !map && out.parent ? reinterpret_cast<u32*>(out.parent) + base
                         : defer            ? scratch_of<u32>(t_parent, ctx, cap, s)
                                            : nullptr;
            lab.capacity = cap;
            lab.label_width = L;
            lab.parent_base = static_cast<u32>(c.first);
            lab.scratch = sc;
            lab.scratch_rows = sc ? c.total : 0;
            lab.scratch_indexed = 1;
            lab.error = d + k_error;
            const lifted::Parents p = parents_of(c);
            const lifted::Views v{static_cast<u64*>(views.data()), plan.view_words};
            const auto* sg = static_cast<const u32*>(seg.data()) + c.seg_base;
            // the launches (row groups by OW and bucket) concurrently
            lifted::Multi m = multi_of(c);
            const auto& launches = plan.launches(launch);
            const auto groups = static_cast<u32>(launches.size());
            fan.fork(s, groups);
            for (u32 gi = 0; gi < groups; ++gi)
            {
                const detail::TableLaunch::Launch& l = launches[gi];
                m.ow = l.ow;
                m.words_lo = l.lo;
                m.words_hi = l.hi;
                if (fixed.count)
                    check(lifted::launch_write_multi(m, p, v, fixed, sg, lab, rows, l.wb, fan.lane(s, gi)),
                          "launch_write_multi");
                if (fc.count && l.ow <= lifted::k_max_fc_ow)
                    check(lifted::launch_write_multi(m, p, v, fc, sg, lab, rows, l.wb, fan.lane(s, gi)),
                          "launch_write_multi (fc)");
            }
            fan.join(s, groups);
            if (defer)
            {
                // every row of the chunk in one launch, with its goal flag (mapped: at its batch row, with its labels)
                lifted::PutRows to;
                if (map)
                {
                    to.rows = map->rows;
                    to.batch_offsets = map->batch_offsets;
                    to.capacity = out.capacity;
                    to.binding = reinterpret_cast<u32*>(out.binding);
                    to.label_width = out.label_width;
                    to.schema = reinterpret_cast<u32*>(out.schema);
                    to.parent = reinterpret_cast<u32*>(out.parent);
                }
                to.goal = out.goal && rows.words ? (map ? out.goal : out.goal + base) : nullptr;
                check(lifted::launch_put_multi(m, p, sg, lab, rows, to, c.total, s), "launch_put_multi");
            }
            if (out.offsets && !map)
                check(lifted::launch_state_offsets(sg, S, c.rows, base, out.offsets + c.first, s), "launch_state_offsets");
        }
        finish_write(out, 0);
    }

    /// The counted batch's CSR offsets (Multi: from the kept segment offsets).
    void offsets_multi(i32* out)
    {
        if (chunks.empty())
            check(cudaMemsetAsync(out, 0, sizeof(i32), s), "cudaMemsetAsync");
        for (const Chunk& c : chunks)
        {
            if (!segs_kept)
                prepare_multi(c, false);
            check(lifted::launch_state_offsets(static_cast<const u32*>(seg.data()) + c.seg_base, plan.num_schemas, c.rows,
                                               c.base, out + c.first, s),
                  "launch_state_offsets");
        }
        segs_kept = true;
    }

    // ------------------------------------------------------------------------------------------ PerInstance
    DeviceExpander& sub(u32 i)
    {
        if (!subs[i])
        {
            subs[i] = std::make_unique<DeviceExpander>(ctx, rl::TaskTable::single(table->task(i)), s);
            subs[i]->set_chunk_rows(chunk_rows);
        }
        return *subs[i];
    }

    u64 count_parts()
    {
        const u32 I = table->size();
        const u64 n = in.rows;
        // the task ids on the host: the rows' grouping by instance
        host_ids.resize(n);
        if (n)
            check(cudaMemcpyAsync(host_ids.data(), ids, n * sizeof(i32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        table->check_task_ids(host_ids.data(), n);
        // rows with bits past their instance's assigned slots (the slots of lazy instances as they are now)
        if (opt.validate && n)
        {
            std::vector<u32> lim(I);
            for (u32 i = 0; i < I; ++i)
                lim[i] = table->task(i)->atoms().fluent_slots();
            auto* dl = scratch_of<u32>(part_limits, ctx, I, s);
            check(cudaMemcpyAsync(dl, lim.data(), I * sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
            u32* d = reset_ctl();
            check(lifted::launch_check_multi(dl, I, ids, in.data, stride, in.words, n, d + k_bad_id, d + k_bad_row, s),
                  "launch_check_multi");
            check(cudaMemcpyAsync(hctl(), d, k_ctl_words * sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            throw_bad_rows();
        }
        // parts: the rows of each instance, in batch order, gathered into consecutive positions
        std::vector<u64> per(I, 0);
        for (u64 r = 0; r < n; ++r)
            ++per[static_cast<u32>(host_ids[r])];
        parts.clear();
        std::vector<u64> next(I, 0);
        u64 at = 0;
        for (u32 i = 0; i < I; ++i)
            if (per[i])
            {
                parts.push_back({i, at, per[i], 0, 0});
                next[i] = at;
                at += per[i];
            }
        std::vector<u32> idx(n);
        for (u64 r = 0; r < n; ++r)
            idx[next[static_cast<u32>(host_ids[r])]++] = static_cast<u32>(r);
        auto* di = scratch_of<u32>(index, ctx, n, s);
        auto* g = scratch_of<u64>(gathered, ctx, n * std::max<u32>(in.words + in.numeric_words, 1), s);
        if (n)
        {
            check(cudaMemcpyAsync(di, idx.data(), n * sizeof(u32), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync");
            check(lifted::launch_gather_rows(in.data, stride, in.words + in.numeric_words, n, di, n, g, s), "launch_gather_rows");
        }
        rl::ExpandOptions o = opt;
        o.validate = false;  // validated above, against the batch's rows
        total = 0;
        for (Part& p : parts)
        {
            p.total = sub(p.instance).count(rl::StateBatchView{g + p.start * (in.words + in.numeric_words), p.rows, in.words,
                                                            in.words + in.numeric_words, table->task(p.instance)->numeric_words()}, nullptr, o);
            p.base = total;
            total += p.total;
        }
        chunks.clear();
        return total;
    }

    /// The batch's CSR offsets [rows + 1] into `out` from the parts' (part k's at toff + start + k).
    void batch_offsets_of(const i32* toff, i32* out)
    {
        const u64 n = in.rows;
        auto* cnt = scratch_of<u32>(row_counts, ctx, n + 1, s);
        check(cudaMemsetAsync(cnt, 0, (n + 1) * sizeof(u32), s), "cudaMemsetAsync");
        const auto* di = static_cast<const u32*>(index.data());
        for (usize k = 0; k < parts.size(); ++k)
            check(lifted::launch_part_counts(di + parts[k].start, toff + parts[k].start + k, parts[k].rows, cnt, s),
                  "launch_part_counts");
        const u64 stb = lifted::scan_temp_bytes(n + 1);
        check(lifted::launch_scan(cnt, reinterpret_cast<u32*>(out), n + 1, scan.ensure(ctx, stb, s), stb, s), "launch_scan");
    }

    void offsets_parts(i32* out)
    {
        auto* toff = scratch_of<i32>(t_offsets, ctx, in.rows + parts.size(), s);
        for (usize k = 0; k < parts.size(); ++k)
            sub(parts[k].instance).offsets(toff + parts[k].start + k);
        batch_offsets_of(toff, out);
    }

    /// Each instance's part expanded into scratch, then its rows put at their batch positions (mapped: at the rows of
    /// the larger batch, through map->rows).
    void write_parts(rl::Expansion& out, const RowMap* map)
    {
        const u32 W = out.words, RW = W + out.numeric_words, L = label_columns(out);
        const u64 n = in.rows, T = total;
        auto* ts = out.succ ? scratch_of<u64>(t_succ, ctx, T * std::max<u32>(RW, 1), s) : nullptr;
        auto* tsc = out.schema ? scratch_of<i32>(t_schema, ctx, T, s) : nullptr;
        auto* tb = out.binding ? scratch_of<i32>(t_binding, ctx, T * L, s) : nullptr;
        auto* tp = scratch_of<i32>(t_parent, ctx, T, s);
        auto* tg = out.goal ? scratch_of<u8>(t_goal, ctx, T, s) : nullptr;
        auto* toff = scratch_of<i32>(t_offsets, ctx, n + parts.size(), s);
        u32 need = 0;
        for (usize k = 0; k < parts.size(); ++k)
        {
            const Part& p = parts[k];
            rl::Expansion e;
            e.capacity = p.total;
            e.words = W;
            e.numeric_words = table->task(p.instance)->numeric_words();
            e.label_width = L;
            e.succ = ts ? ts + p.base * RW : nullptr;
            e.parent = tp + p.base;
            e.schema = tsc ? tsc + p.base : nullptr;
            e.binding = tb ? tb + p.base * L : nullptr;
            e.goal = tg ? tg + p.base : nullptr;
            e.offsets = toff + p.start + k;
            sub(p.instance).write(e);
            need = std::max(need, e.words_needed);
        }
        // the batch's CSR offsets from the parts' counts (mapped: the larger batch's), then every part's rows to their
        // batch positions
        const auto* di = static_cast<const u32*>(index.data());
        const i32* go = map ? map->batch_offsets : nullptr;
        if (!map)
        {
            i32* o = out.offsets ? out.offsets : scratch_of<i32>(batch_offsets, ctx, n + 1, s);
            batch_offsets_of(toff, o);
            go = o;
        }
        for (usize k = 0; k < parts.size(); ++k)
        {
            const Part& p = parts[k];
            lifted::PartScatter x;
            x.rows = di + p.start;
            x.outer = map ? map->rows : nullptr;
            x.offsets = toff + p.start + k;
            x.batch_offsets = go;
            x.total = p.total;
            x.parent = tp + p.base;
            x.succ = ts ? ts + p.base * RW : nullptr;
            x.words = W + table->task(p.instance)->numeric_words();
            x.out_words = RW;
            x.schema = tsc ? tsc + p.base : nullptr;
            x.binding = tb ? tb + p.base * L : nullptr;
            x.label_width = L;
            x.out_label_width = L;
            x.goal = tg ? tg + p.base : nullptr;
            x.capacity = out.capacity;
            x.out_succ = out.succ;
            x.out_schema = out.schema;
            x.out_binding = out.binding;
            x.out_parent = out.parent;
            x.out_goal = out.goal;
            check(lifted::launch_part_scatter(x, s), "launch_part_scatter");
        }
        u32* d = dctl();
        check(cudaMemsetAsync(d + k_words_needed, 0, 2 * sizeof(u32), s), "cudaMemsetAsync");
        finish_write(out, need);
    }

    // ------------------------------------------------------------------------------------------ a part of a larger batch
    void offsets_single(i32* out)
    {
        if (chunks.empty())
            check(cudaMemsetAsync(out, 0, sizeof(i32), s), "cudaMemsetAsync");
        for (const Chunk& c : chunks)
        {
            if (!kept)
                prepare_single(c);
            check(lifted::launch_state_offsets(gen->seg_offsets(), gen->num_schemas(), c.rows, c.base, out + c.first, s),
                  "launch_state_offsets");
        }
    }

    /// Single's mapped write: the expansion into scratch at the table's widths (ChunkGenerator writes at its rows), then
    /// scattered to the batch rows, widened.
    void write_single_mapped(rl::Expansion& out, const RowMap& map)
    {
        const u64 T = total;
        const bool any = out.succ || out.schema || out.binding || out.parent || out.goal;
        rl::Expansion e;
        e.capacity = any ? T : 0;
        e.words = table->numeric() ? out.words : std::max<u32>(1, std::min(out.words, table->words()));
        e.numeric_words = table->numeric_words();
        e.label_width = std::max<u32>(1, std::min(out.label_width, table->label_width()));
        e.succ = out.succ ? scratch_of<u64>(t_succ, ctx, T * (e.words + e.numeric_words), s) : nullptr;
        e.parent = any ? scratch_of<i32>(t_parent, ctx, T, s) : nullptr;
        e.schema = out.schema ? scratch_of<i32>(t_schema, ctx, T, s) : nullptr;
        e.binding = out.binding ? scratch_of<i32>(t_binding, ctx, T * e.label_width, s) : nullptr;
        e.goal = out.goal ? scratch_of<u8>(t_goal, ctx, T, s) : nullptr;
        e.offsets = scratch_of<i32>(t_offsets, ctx, in.rows + 1, s);
        write_single(e);
        out.total = e.total;
        out.words_needed = e.words_needed;
        if (!any)
            return;
        lifted::PartScatter x;
        x.rows = map.rows;
        x.offsets = e.offsets;
        x.batch_offsets = map.batch_offsets;
        x.total = T;
        x.parent = e.parent;
        x.succ = e.succ;
        x.words = e.words + e.numeric_words;
        x.out_words = out.words + out.numeric_words;
        x.schema = e.schema;
        x.binding = e.binding;
        x.label_width = e.label_width;
        x.out_label_width = out.label_width;
        x.goal = e.goal;
        x.capacity = out.capacity;
        x.out_succ = out.succ;
        x.out_schema = out.schema;
        x.out_binding = out.binding;
        x.out_parent = out.parent;
        x.out_goal = out.goal;
        check(lifted::launch_part_scatter(x, s), "launch_part_scatter");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // as write(): the outputs are done
    }

    u32 max_numeric_slots() const
    {
        u32 n = 0;
        for (u32 i = 0; i < table->size(); ++i)
            n = std::max(n, table->task(i)->numeric_slots());
        return n;
    }

    /// The checks of write() on its destinations.
    void check_write(const rl::Expansion& out, bool mapped = false) const
    {
        if (!counted)
            throw std::logic_error("mymyr: DeviceExpander::write: count() first");
        if (out.binding && out.label_width < table->label_width())
            throw std::invalid_argument("mymyr: expand: label width " + std::to_string(out.label_width) +
                                        " is below the largest schema arity " + std::to_string(table->label_width()));
        if (out.capacity > k_i32_max)
            throw std::invalid_argument("mymyr: expand: capacity above 2^31 - 1 rows");
        if (out.succ && out.words == 0 && out.capacity)
            throw std::invalid_argument("mymyr: expand: successor rows of zero words");
        if (out.goal && !out.succ)
            throw std::invalid_argument("mymyr: expand: goal flags need the successor rows");
        if (out.succ && out.words + max_numeric_slots() > lifted::k_max_words)
            throw std::invalid_argument("mymyr: device expand: successor rows of " + std::to_string(out.words) +
                                        " words (the device kernels take at most " + std::to_string(lifted::k_max_words) +
                                        ")");
        if (mapped ? out.numeric_words < table->numeric_words() : out.numeric_words != table->numeric_words())
            throw std::invalid_argument("mymyr: device expand: output numeric width differs from the table");
        if (out.offsets && total > k_i32_max)
            throw std::length_error("mymyr: expand: more than 2^31 - 1 successors in one batch");
    }
};

DeviceExpander::DeviceExpander(ContextPtr ctx, rl::TaskTablePtr table, cudaStream_t stream) : m(std::make_unique<Impl>())
{
    if (!ctx || !table)
        throw std::invalid_argument("mymyr: DeviceExpander: null context or table");
    if (const std::string why = table->size() == 1 ? ChunkGenerator::unsupported(*table->task(0))
                                                  : DeviceTaskTable::unsupported(*table); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run " + why);
    Impl& I = *m;
    I.ctx = std::move(ctx);
    I.table = std::move(table);
    I.s = stream ? stream : I.ctx->stream();
    DeviceGuard g(I.ctx->device());
    if (I.table->size() == 1)
    {
        I.mode = Mode::Single;
        I.gen = std::make_unique<ChunkGenerator>(I.ctx, I.table->task(0), I.s);
    }
    else if (detail::multi_unsupported(*I.table).empty())
    {
        I.mode = Mode::Multi;
        I.dt = DeviceTaskTable::upload(I.ctx, I.table);
        I.dt->acquire(I.s);
        I.plan = detail::plan_launches(I.ctx, *I.dt, I.s);
        std::vector<u32> lim(I.table->size());
        for (u32 i = 0; i < I.table->size(); ++i)
            lim[i] = I.table->task(i)->atoms().fluent_slots();
        I.limits = DeviceBuffer(I.ctx, lim.size() * sizeof(u32), I.s);
        check(cudaMemcpyAsync(I.limits.data(), lim.data(), lim.size() * sizeof(u32), cudaMemcpyHostToDevice, I.s),
              "cudaMemcpyAsync");
        check(cudaStreamSynchronize(I.s), "cudaStreamSynchronize");  // the pageable source
    }
    else
    {
        I.mode = Mode::PerInstance;
        I.subs.resize(I.table->size());
    }
}

DeviceExpander::~DeviceExpander()
{
    if (m && m->ctx)
    {
        try
        {
            DeviceGuard g(m->ctx->device());
            check(cudaStreamSynchronize(m->s), "cudaStreamSynchronize");
        }
        catch (...)
        {
        }
    }
}

const ContextPtr& DeviceExpander::context() const noexcept { return m->ctx; }
const rl::TaskTablePtr& DeviceExpander::table() const noexcept { return m->table; }
DeviceExpander::Mode DeviceExpander::mode() const noexcept { return m->mode; }
cudaStream_t DeviceExpander::stream() const noexcept { return m->s; }
void DeviceExpander::set_launch(BucketLaunch launch) noexcept { m->launch = launch; }

void DeviceExpander::set_chunk_rows(u64 rows) noexcept
{
    m->chunk_rows = rows;
    for (auto& x : m->subs)
        if (x)
            x->set_chunk_rows(rows);
}

void DeviceExpander::set_stream(cudaStream_t s)
{
    Impl& I = *m;
    s = s ? s : I.ctx->stream();
    if (s == I.s)
        return;
    DeviceGuard g(I.ctx->device());
    if (!detail::capturing(s))  // a capturing stream cannot wait on work outside its graph
        stream_wait(s, I.s);
    if (I.gen)
        I.gen->set_stream(s);
    if (I.dt)
    {
        I.dt->acquire(s);
        I.plan.record_stream(s);
        I.limits.record_stream(s);
    }
    for (auto& x : I.subs)
        if (x)
            x->set_stream(s);
    I.s = s;
    I.kept = false;  // the scratch moves to the new stream as it is next needed
    I.segs_kept = false;
    I.views_chunk = -1;
    if (I.mode == Mode::PerInstance)
        I.counted = false;  // the gathered parts are scratch
}

u64 DeviceExpander::count(rl::StateBatchView in, const i32* task_ids, const rl::ExpandOptions& opt)
{
    Impl& I = *m;
    DeviceGuard guard(I.ctx->device());
    I.counted = false;
    I.kept = false;
    I.segs_kept = false;
    I.views_chunk = -1;
    if (in.rows && !in.data)
        throw std::invalid_argument("mymyr: expand: null state buffer");
    if (in.stride && in.stride < in.words + in.numeric_words)
        throw std::invalid_argument("mymyr: expand: row stride smaller than the row width");
    if (in.numeric_words != I.table->numeric_words())
        throw std::invalid_argument("mymyr: device expand: input numeric width differs from the table");
    if (in.rows > k_i32_max)
        throw std::invalid_argument("mymyr: expand: more than 2^31 - 1 states in one batch");
    if (in.words + I.max_numeric_slots() > lifted::k_max_words)
        throw std::invalid_argument("mymyr: device expand: state rows of " + std::to_string(in.words) +
                                    " words (the device kernels take at most " + std::to_string(lifted::k_max_words) + ")");
    if (!task_ids && in.rows && I.table->size() > 1)
        throw std::invalid_argument("mymyr: expand: a batch over a table of " + std::to_string(I.table->size()) +
                                    " instances needs task ids");
    I.in = in;
    I.opt = opt;
    I.ids = task_ids;
    I.stride = in.stride ? in.stride : in.words + in.numeric_words;
    switch (I.mode)
    {
        case Mode::Single: I.count_single(); break;
        case Mode::Multi: I.count_multi(); break;
        case Mode::PerInstance: I.count_parts(); break;
    }
    I.counted = true;
    I.kept = I.mode != Mode::PerInstance && I.chunks.size() == 1;
    return I.total;
}

void DeviceExpander::write(rl::Expansion& out)
{
    Impl& I = *m;
    I.check_write(out);
    DeviceGuard guard(I.ctx->device());
    switch (I.mode)
    {
        case Mode::Single: I.write_single(out); break;
        case Mode::Multi: I.write_multi(out, nullptr); break;
        case Mode::PerInstance: I.write_parts(out, nullptr); break;
    }
}

void DeviceExpander::offsets(i32* out)
{
    Impl& I = *m;
    if (!I.counted)
        throw std::logic_error("mymyr: DeviceExpander::offsets: count() first");
    if (!out)
        throw std::invalid_argument("mymyr: DeviceExpander::offsets: null destination");
    if (I.total > k_i32_max)
        throw std::length_error("mymyr: expand: more than 2^31 - 1 successors in one batch");
    DeviceGuard guard(I.ctx->device());
    switch (I.mode)
    {
        case Mode::Single: I.offsets_single(out); break;
        case Mode::Multi: I.offsets_multi(out); break;
        case Mode::PerInstance: I.offsets_parts(out); break;
    }
}

void DeviceExpander::write(rl::Expansion& out, const RowMap& map)
{
    Impl& I = *m;
    I.check_write(out, true);
    if (I.in.rows && (!map.rows || !map.batch_offsets))
        throw std::invalid_argument("mymyr: DeviceExpander::write: a row map without rows or batch offsets");
    DeviceGuard guard(I.ctx->device());
    switch (I.mode)
    {
        case Mode::Single: I.write_single_mapped(out, map); break;
        case Mode::Multi: I.write_multi(out, &map); break;
        case Mode::PerInstance: I.write_parts(out, &map); break;
    }
}

void DeviceExpander::expand(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, const rl::ExpandOptions& options)
{
    count(in, task_ids, options);
    write(out);
}

void DeviceExpander::pad(const rl::Expansion& flat, u64 rows, rl::PaddedExpansion& out)
{
    Impl& I = *m;
    if (!flat.offsets)
        throw std::invalid_argument("mymyr: pad: the flat expansion has no offsets");
    if (out.succ && flat.succ && out.words < flat.words)
        throw std::invalid_argument("mymyr: pad: padded rows narrower than the flat rows");
    if (out.numeric_words != flat.numeric_words)
        throw std::invalid_argument("mymyr: pad: numeric widths differ");
    if (out.binding && flat.binding && out.label_width < flat.label_width)
        throw std::invalid_argument("mymyr: pad: padded label width below the flat label width");
    DeviceGuard guard(I.ctx->device());
    const cudaStream_t s = I.s;
    u32* d = I.dctl();
    check(cudaMemsetAsync(d + k_overflow, 0, sizeof(u32), s), "cudaMemsetAsync");
    lifted::Flat f{flat.offsets, flat.capacity, flat.succ, flat.words + flat.numeric_words, flat.schema, flat.binding, flat.label_width, flat.goal};
    lifted::Padded p{out.K, out.index, out.mask, out.count, out.succ, out.words + out.numeric_words, out.schema, out.binding, out.label_width,
                     out.goal, d + k_overflow};
    if (flat.numeric_words)
        check(numeric::launch_pad(f, rows, p, flat.numeric_words, s), "numeric::launch_pad");
    else
        check(lifted::launch_pad(f, rows, p, s), "launch_pad");
    check(cudaMemcpyAsync(I.hctl() + k_overflow, d + k_overflow, sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    out.overflow = I.hctl()[k_overflow] != 0;
}
}  // namespace mymyr::cuda
