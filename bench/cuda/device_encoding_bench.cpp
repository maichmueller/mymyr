// mymyr_device_encoding_bench: the device relational encoder on the rows of one PDDL task, one JSON object on standard
// output.
//
//   mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W [--n 16384] [--rounds 20]
//                               [--replays 10] [--device 0] [--secondary-problem P.pddl]
//                               [--lanes 4|8|32] [--fill auto|thread|group|warp] [--streams] [--collate-slices K]
//                               [--ring-rows L] [--instances K]
//                               [--levels L] [--history H] [--max-steps M] [--literals K] [--targets]
//                               [--cpu-threads T] [--outputs relations|simple|both]
//
// FILE holds the state rows, little-endian u64 [rows, W]; they are tiled to --n graphs. For each of the graph-major and
// the relation-major layout the encode of the batch is captured into a CUDA graph on a stream and replayed --replays
// times between two events per round (the time of one round), and the graph-major encode is also timed as one eager call
// followed by a synchronize on the host clock. The rounds of the three timings alternate, the first rotating. The object
// also reports, for each layout, the nodes of the captured graph (kernels, memsets, anything else), the scratch of the
// reserved workspace, and whether the device's sizes equal the host encoder's.
//
// --levels L gives every row L subgoal layers and --history H a history of H valid steps (M: the newest steps encoded),
// each layer and step K random literals of the row's task (the plain and satisfied goal relations are on with levels;
// --targets makes the goal, subgoal and history entities targets). The host encoder's time on the same rows with T
// threads is reported as cpu_graph_major_us (one entry per round, the whole batch).

// With --streams, also times captured append, flush, append plus flush and four-slice collation (--collate-slices K:
// K slices of equal size). Append and flush arrays hold one replay per sample; combined and collation arrays hold
// --replays calls per sample. --ring-rows L appends and flushes L rows in a ring of --n rows (default: --n rows in a
// ring of --n + 1). Tasks with derived atoms also time supplied derived words, prepared on the host before any timing.
// --instances K encodes over a table of K instances of the problem, row g of instance g % K.

#include "mymyr/cuda/encoding.hpp"
#include "../../cuda/src/encoding_scratch.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
[[noreturn]] void usage(const char* message)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W "
                 "[--n N] [--rounds R] [--replays K] [--device I] [--lanes L] [--fill auto|thread|group|warp] [--streams] "
                 "[--levels L] [--history H] [--max-steps M] [--literals K] [--targets] [--cpu-threads T]\n",
                 message);
    std::exit(2);
}

struct Args
{
    std::string domain, problem, secondary_problem, rows, outputs = "relations";
    u32 words = 0;
    u64 n = 16384;
    u32 rounds = 20, replays = 10;
    int device = 0;
    cuda::LaunchConfig launch;
    u32 levels = 0, history = 0, max_steps = 0, literals = 6, cpu_threads = 1;
    bool targets = false;
    bool streams = false;
    u64 collate_slices = 0, ring_rows = 0;
    u32 instances = 1;
};

Args parse(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string flag = argv[i];
        auto value = [&]() -> const char*
        {
            if (i + 1 >= argc)
                usage(("missing value for " + flag).c_str());
            return argv[++i];
        };
        if (flag == "--outputs")
            a.outputs = value();
        else if (flag == "--domain")
            a.domain = value();
        else if (flag == "--problem")
            a.problem = value();
        else if (flag == "--secondary-problem")
            a.secondary_problem = value();
        else if (flag == "--rows")
            a.rows = value();
        else if (flag == "--words")
            a.words = static_cast<u32>(std::stoul(value()));
        else if (flag == "--n")
            a.n = std::stoull(value());
        else if (flag == "--rounds")
            a.rounds = static_cast<u32>(std::stoul(value()));
        else if (flag == "--replays")
            a.replays = static_cast<u32>(std::stoul(value()));
        else if (flag == "--streams")
            a.streams = true;
        else if (flag == "--collate-slices")
            a.collate_slices = std::stoull(value());
        else if (flag == "--ring-rows")
            a.ring_rows = std::stoull(value());
        else if (flag == "--instances")
            a.instances = static_cast<u32>(std::stoul(value()));
        else if (flag == "--device")
            a.device = std::stoi(value());
        else if (flag == "--lanes")
            a.launch.count_lanes = static_cast<u32>(std::stoul(value()));
        else if (flag == "--levels")
            a.levels = static_cast<u32>(std::stoul(value()));
        else if (flag == "--history")
            a.history = static_cast<u32>(std::stoul(value()));
        else if (flag == "--max-steps")
            a.max_steps = static_cast<u32>(std::stoul(value()));
        else if (flag == "--literals")
            a.literals = static_cast<u32>(std::stoul(value()));
        else if (flag == "--cpu-threads")
            a.cpu_threads = static_cast<u32>(std::stoul(value()));
        else if (flag == "--targets")
            a.targets = true;
        else if (flag == "--fill")
        {
            const std::string mode = value();
            if (mode == "auto")
                a.launch.fill = cuda::FillMode::Auto;
            else if (mode == "thread")
                a.launch.fill = cuda::FillMode::Thread;
            else if (mode == "group")
                a.launch.fill = cuda::FillMode::Group;
            else if (mode == "warp")
                a.launch.fill = cuda::FillMode::Warp;
            else
                usage("--fill takes auto, thread, group or warp");
        }
        else
            usage(("unknown flag " + flag).c_str());
    }
    if (a.domain.empty() || a.problem.empty() || a.rows.empty() || a.words == 0 || a.n == 0 || a.rounds == 0 ||
        a.replays == 0)
        usage("--domain, --problem, --rows and --words are required, the counts must be positive");
    if (a.ring_rows > a.n || a.collate_slices > a.n || a.instances == 0 ||
        (a.instances > 1 && !a.secondary_problem.empty()))
        usage("--ring-rows and --collate-slices are at most --n; --instances is positive and excludes --secondary-problem");
    return a;
}

std::vector<u64> read_rows(const std::string& path, u32 words, u64 n)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        usage(("cannot read " + path).c_str());
    const u64 bytes = static_cast<u64>(in.tellg());
    const u64 row_bytes = u64{words} * 8;
    if (bytes == 0 || bytes % row_bytes != 0)
        usage("the rows file is empty or not a whole number of rows");
    std::vector<u64> rows(bytes / 8);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(bytes));
    const u64 distinct = bytes / row_bytes;
    std::vector<u64> tiled(n * words);
    for (u64 g = 0; g < n; ++g)
        std::memcpy(tiled.data() + g * words, rows.data() + (g % distinct) * words, row_bytes);
    return tiled;
}

/// The subgoal layers and history of the rows: K random literals per layer and step, over the slots of the first task
/// (a literal is positive if the row's state has the atom, negative otherwise, one in two).
struct LayerRows
{
    u32 levels = 0, steps = 0, words = 0;
    std::vector<u64> layer_pos, layer_neg, step_pos, step_neg;
    std::vector<i32> dt;

    LayerRows(const Args& a, const std::vector<u64>& rows, u32 row_words, u32 slots) : levels(a.levels), steps(a.history), words(row_words)
    {
        const u64 n = rows.size() / row_words;
        layer_pos.assign(n * levels * words, 0);
        layer_neg = layer_pos;
        step_pos.assign(n * steps * words, 0);
        step_neg = step_pos;
        dt.assign(n * steps, 0);
        if (slots == 0)
            return;
        for (u64 g = 0; g < n; ++g)
        {
            std::mt19937_64 rng(g + 1);
            const u64* state = rows.data() + g * words;
            auto draw = [&](u64* pos, u64* neg)
            {
                for (u32 k = 0; k < a.literals; ++k)
                {
                    const u32 slot = static_cast<u32>(rng() % slots);
                    const bool held = (state[slot >> 6] >> (slot & 63)) & 1u;
                    (held == (rng() % 2 == 0) ? pos : neg)[slot >> 6] |= u64{1} << (slot & 63);
                }
            };
            for (u32 l = 0; l < levels; ++l)
                draw(layer_pos.data() + (g * levels + l) * words, layer_neg.data() + (g * levels + l) * words);
            for (u32 j = 0; j < steps; ++j)
            {
                draw(step_pos.data() + (g * steps + j) * words, step_neg.data() + (g * steps + j) * words);
                dt[g * steps + j] = static_cast<i32>(j + 1);
            }
        }
    }

    [[nodiscard]] encoding::LayerInput subgoals(const u64* pos, const u64* neg) const
    {
        encoding::LayerInput in;
        if (levels)
            in = {pos, neg, u64{levels} * words, levels, words};
        return in;
    }
    [[nodiscard]] encoding::HistoryInput history_of(const u64* pos, const u64* neg, const i32* step_dt) const
    {
        encoding::HistoryInput in;
        if (steps)
        {
            in.masks = {pos, neg, u64{steps} * words, steps, words};
            in.dt = step_dt;
        }
        return in;
    }
};

struct NodeCounts
{
    u32 kernels = 0, memsets = 0, others = 0, thread_fills = 0, group_fills = 0, warp_fills = 0;
};

NodeCounts count_nodes(cudaGraph_t graph)
{
    size_t count = 0;
    cuda::check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    cuda::check(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
    NodeCounts counts;
    for (const cudaGraphNode_t node : nodes)
    {
        cudaGraphNodeType type;
        cuda::check(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        if (type == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams params{};
            cuda::check(cudaGraphKernelNodeGetParams(node, &params), "cudaGraphKernelNodeGetParams");
            const char* symbol = nullptr;
            cuda::check(cudaFuncGetName(&symbol, params.func), "cudaFuncGetName");
            const std::string name(symbol);
            // the lanes per graph are the fill kernel's first unsigned template argument (1, 8 or 32)
            const auto fill = name.find("fill_kernel");
            const auto lanes = fill == std::string::npos ? fill : name.find("Lj", fill);
            if (lanes != std::string::npos)
                (name.compare(lanes, 4, "Lj1E") == 0   ? counts.thread_fills
                 : name.compare(lanes, 4, "Lj8E") == 0 ? counts.group_fills
                                                       : counts.warp_fills) += 1;
        }
        (type == cudaGraphNodeTypeKernel ? counts.kernels : type == cudaGraphNodeTypeMemset ? counts.memsets : counts.others) += 1;
    }
    return counts;
}

/// One layout: its encoder, the rows and the output on the device, and the captured encode.
struct Session
{
    encoding::Layout layout;
    std::shared_ptr<const encoding::FlatEncoder> host;
    std::unique_ptr<cuda::DeviceEncoder> device;
    cuda::DeviceBuffer states, derived, block, task_ids;
    std::vector<i32> ids;
    std::vector<u64> derived_rows;  // supplied derived words on the host, which host_input reads
    cuda::DeviceEncodeInput input;
    cuda::DeviceOutput output;
    cuda::OutputLayout layout_of;
    encoding::FlatSizes host_sizes;
    std::optional<cuda::EncodingWorkspace> work;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    NodeCounts nodes;
    cuda::detail::ScratchLayout scratch;
    u64 cells = 0;
    bool sizes_match = false;

    cuda::LaunchConfig launch;
    cuda::DeviceBuffer layer_pos, layer_neg, step_pos, step_neg, step_dt;
    encoding::EncodeInput host_input;
    std::vector<double> cpu_times;

    Session(const cuda::ContextPtr& ctx, cudaStream_t stream, rl::TaskTablePtr table, encoding::Layout l, const std::vector<u64>& rows,
            u32 words, u64 n, const cuda::LaunchConfig& config, const Args& args, const LayerRows& layers,
            bool supplied_derived = false)
        : layout(l), launch(config)
    {
        encoding::FlatOptions options;
        options.goal_derivations = {};
        if (args.outputs == "both")
            options.outputs = {encoding::Output::Relations, encoding::Output::Simple};
        else if (args.outputs == "simple")
            options.outputs = {encoding::Output::Simple};
        else if (args.outputs != "relations")
            usage("outputs must be relations, simple or both");
        options.layout = l;
        if (args.outputs == "simple" && l == encoding::Layout::Relation)
            options.outputs.insert(encoding::Output::Relations);
        if (args.levels)
        {
            options.max_goal_level = args.levels;
            options.goal_derivations = {encoding::GoalDerivation::Plain, encoding::GoalDerivation::Satisfied};
            if (args.targets)
                options.targets.insert(encoding::TargetSource::Goal), options.targets.insert(encoding::TargetSource::Subgoal);
        }
        if (args.history)
        {
            options.history = true;
            if (args.max_steps)
                options.history_max_steps = args.max_steps;
            if (args.targets)
                options.targets.insert(encoding::TargetSource::History);
        }
        host = encoding::FlatEncoder::create(std::move(table), options);
        // the host's sizes size the output
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        in.subgoals = layers.subgoals(layers.layer_pos.data(), layers.layer_neg.data());
        in.history = layers.history_of(layers.step_pos.data(), layers.step_neg.data(), layers.dt.data());
        if (host->table().size() > 1)
        {
            for (u64 g = 0; g < n; ++g)
                ids.push_back(static_cast<i32>(g % host->table().size()));
            in.task_ids = ids.data();
        }
        const u32 dw = host->table().derived_words();
        if (supplied_derived && dw)
        {
            derived_rows.resize(n * dw, 0);
            const auto workspace = host->table().task(0)->workspace();
            const auto& instance = host->table().instance(0);
            for (u64 g = 0; g < n; ++g)
            {
                workspace->engine().set_state(rows.data() + g * words, instance.words);
                workspace->engine().build_view();
                workspace->axioms().evaluate();
                if (instance.derived_words)
                    std::copy_n(workspace->engine().derived(), instance.derived_words, derived_rows.data() + g * dw);
            }
            in.derived = derived_rows.data();
            in.derived_stride = dw;
        }
        host_input = in;
        ThreadPool pool(1);
        encoding::EncodeScratch scratch_of_host;
        host_sizes = host->count(in, pool, scratch_of_host);
        const cuda::BatchShape shape{n, host_sizes.entities, host_sizes.args, host_sizes.targets, host_sizes.history};
        device = std::make_unique<cuda::DeviceEncoder>(ctx, host);
        device->reserve(shape, stream);
        states = cuda::DeviceBuffer(ctx, rows.size() * 8, stream);
        cuda::check(cudaMemcpyAsync(states.data(), rows.data(), rows.size() * 8, cudaMemcpyHostToDevice, stream), "upload rows");
        layout_of = device->output_layout(shape);
        block = cuda::DeviceBuffer(ctx, layout_of.bytes, stream);
        output = cuda::output_of(layout_of, shape, block.data());
        input.graphs = n;
        input.states = static_cast<const u64*>(states.data());
        input.state_stride = words;
        input.state_words = words;
        auto upload = [&](cuda::DeviceBuffer& buffer, const auto& v)
        {
            buffer = cuda::DeviceBuffer(ctx, std::max<u64>(v.size() * sizeof(v[0]), 8), stream);
            if (!v.empty())
                cuda::check(cudaMemcpyAsync(buffer.data(), v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice, stream),
                            "upload layers");
        };
        upload(layer_pos, layers.layer_pos);
        upload(layer_neg, layers.layer_neg);
        upload(step_pos, layers.step_pos);
        upload(step_neg, layers.step_neg);
        upload(step_dt, layers.dt);
        input.subgoals = layers.subgoals(static_cast<const u64*>(layer_pos.data()), static_cast<const u64*>(layer_neg.data()));
        input.history = layers.history_of(static_cast<const u64*>(step_pos.data()), static_cast<const u64*>(step_neg.data()),
                                          static_cast<const i32*>(step_dt.data()));
        if (!ids.empty())
        {
            task_ids = cuda::DeviceBuffer(ctx, ids.size() * sizeof(i32), stream);
            cuda::check(cudaMemcpyAsync(task_ids.data(), ids.data(), task_ids.size(), cudaMemcpyHostToDevice, stream), "upload task ids");
            input.task_ids = static_cast<const i32*>(task_ids.data());
        }
        if (!derived_rows.empty())
        {
            derived = cuda::DeviceBuffer(ctx, derived_rows.size() * sizeof(u64), stream);
            cuda::check(cudaMemcpyAsync(derived.data(), derived_rows.data(), derived.size(), cudaMemcpyHostToDevice, stream), "upload derived rows");
            input.derived = static_cast<const u64*>(derived.data());
            input.derived_stride = dw;
        }
        // an eager encode first, outside any capture, for the sizes
        device->encode(input, output, stream, launch);
        encoding::FlatSizes device_sizes;
        cuda::check(cudaMemcpyAsync(&device_sizes, output.sizes, sizeof device_sizes, cudaMemcpyDeviceToHost, stream), "sizes");
        cuda::check(cudaStreamSynchronize(stream), "sizes");
        sizes_match = device_sizes.entities == host_sizes.entities && device_sizes.tuples == host_sizes.tuples &&
                      device_sizes.args == host_sizes.args && device_sizes.targets == host_sizes.targets &&
                      device_sizes.history == host_sizes.history && device_sizes.overflow == 0 && host_sizes.overflow == 0;
        scratch = cuda::detail::scratch_layout(shape, device->bounds(), l, host->schema().num_relations());
        scratch.bytes = device->scratch_bytes(shape);
        cells = device->bounds().cells;
        // the captured encode
        work.emplace(device->acquire(stream));
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        std::exception_ptr error;
        try { device->encode(*work, input, output, launch); }
        catch (...) { error = std::current_exception(); }
        cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
        if (error)
            std::rethrow_exception(error);
        nodes = count_nodes(graph);
        cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int i = 0; i < 3; ++i)
            cuda::check(cudaGraphLaunch(exec, stream), "warm up");
        cuda::check(cudaStreamSynchronize(stream), "warm up");
    }

    /// The host encoder on the same rows with `threads` threads: the time of the whole batch in microseconds per round.
    void time_host(u32 threads, u32 rounds)
    {
        ThreadPool pool(threads);
        encoding::EncodeScratch scratch_of_host;
        (void)host->encode(host_input, pool, scratch_of_host);
        for (u32 r = 0; r < rounds; ++r)
        {
            const auto t0 = std::chrono::steady_clock::now();
            (void)host->encode(host_input, pool, scratch_of_host);
            cpu_times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (exec)
            (void)cudaGraphExecDestroy(exec);
        if (graph)
            (void)cudaGraphDestroy(graph);
    }
};

template<class Operation>
double replay_us(cudaStream_t stream, const Operation& s, u32 replays)
{
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda::check(cudaEventCreate(&start), "event");
    cuda::check(cudaEventCreate(&stop), "event");
    cuda::check(cudaEventRecord(start, stream), "event");
    for (u32 i = 0; i < replays; ++i)
        cuda::check(cudaGraphLaunch(s.exec, stream), "replay");
    cuda::check(cudaEventRecord(stop, stream), "event");
    cuda::check(cudaEventSynchronize(stop), "event");
    float ms = 0;
    cuda::check(cudaEventElapsedTime(&ms, start, stop), "event");
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
    return static_cast<double>(ms) * 1e3;
}

double eager_us(cudaStream_t stream, const Session& s)
{
    const auto t0 = std::chrono::steady_clock::now();
    s.device->encode(s.input, s.output, stream, s.launch);
    cuda::check(cudaStreamSynchronize(stream), "eager encode");
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

struct CapturedOperation
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    NodeCounts nodes;
    template<class F> void capture(cudaStream_t stream, F&& operation)
    {
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        std::exception_ptr error;
        try { operation(); } catch (...) { error = std::current_exception(); }
        cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
        if (error) std::rethrow_exception(error);
        nodes = count_nodes(graph);
        cuda::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    }
    ~CapturedOperation()
    {
        if (exec) (void)cudaGraphExecDestroy(exec);
        if (graph) (void)cudaGraphDestroy(graph);
    }
};

struct StreamSession
{
    cuda::DeviceRowStream ring;
    cuda::DeviceEncodeInput input;
    cuda::DeviceBuffer ring_output, collate_output, scratch;
    std::vector<cuda::DeviceBuffer> parts;
    std::unique_ptr<cuda::DeviceCollator> collator;
    CapturedOperation append, flush, append_flush, collate;
    std::vector<double> append_times, flush_times, append_flush_times, collate_times;
    u64 slices = 0;

    StreamSession(const cuda::ContextPtr& ctx, cudaStream_t stream, const Session& base, const std::vector<u64>& rows,
                  u64 ring_rows, u64 slices)
        : ring(ctx, base.host, ring_rows ? base.input.graphs : base.input.graphs + 1, stream), input(base.input)
    {
        const auto& sizes = base.host_sizes;
        const cuda::BatchShape ring_shape{ring.capacity(), sizes.entities, sizes.args, sizes.targets};
        ThreadPool pool(1);
        encoding::EncodeScratch host_scratch;
        auto host_sizes_of = [&](u64 first, u64 count)
        {
            encoding::EncodeInput host_input;
            host_input.states = {rows.data() + first * input.state_stride, count, input.state_words, input.state_stride, 0};
            if (!base.ids.empty()) host_input.task_ids = base.ids.data() + first;
            return base.host->count(host_input, pool, host_scratch);
        };
        auto appended = input;
        if (ring_rows) appended.graphs = ring_rows;
        const auto ring_expected = ring_rows ? host_sizes_of(0, ring_rows) : sizes;
        const auto ring_layout = base.device->output_layout(ring_shape);
        ring_output = cuda::DeviceBuffer(ctx, ring_layout.bytes, stream);
        const auto ring_out = cuda::output_of(ring_layout, ring_shape, ring_output.data());
        input = appended;
        append.capture(stream, [&] { ring.append(input); });
        flush.capture(stream, [&] { ring.flush(ring_out); });
        append_flush.capture(stream, [&] { ring.append(input); ring.flush(ring_out); });
        std::vector<cuda::DeviceCollationBatch> batches;
        u64 first = 0;
        const u64 n = base.input.graphs;
        std::vector<u64> counts{n / 3, u64{0}, n / 3, n - 2 * (n / 3)};
        if (slices)
        {
            counts.assign(slices, n / slices);
            counts.back() += n % slices;
        }
        for (const u64 count : counts)
        {
            const auto part_sizes = host_sizes_of(first, count);
            const cuda::BatchShape shape{count, part_sizes.entities, part_sizes.args, part_sizes.targets};
            const auto layout = base.device->output_layout(shape);
            parts.emplace_back(ctx, layout.bytes, stream);
            const auto out = cuda::output_of(layout, shape, parts.back().data());
            auto part = base.input;
            part.states += first * input.state_stride;
            if (part.task_ids) part.task_ids += first;
            if (part.derived) part.derived += first * part.derived_stride;
            part.graphs = count;
            base.device->encode(part, out, stream);
            batches.push_back(cuda::collation_batch(*base.host, out.arrays, out.sizes, count));
            first += count;
        }
        this->slices = batches.size();
        collator = std::make_unique<cuda::DeviceCollator>(ctx, std::move(batches), stream);
        scratch = cuda::DeviceBuffer(ctx, collator->scratch_bytes(), stream);
        const cuda::BatchShape shape{n, sizes.entities, sizes.args, sizes.targets};
        const auto layout = base.device->output_layout(shape);
        collate_output = cuda::DeviceBuffer(ctx, layout.bytes, stream);
        const auto out = cuda::output_of(layout, shape, collate_output.data());
        collate.capture(stream, [&] { collator->encode(scratch.data(), scratch.size(), out); });
        for (int i = 0; i < 3; ++i)
        {
            cuda::check(cudaGraphLaunch(append_flush.exec, stream), "warm stream");
            cuda::check(cudaGraphLaunch(collate.exec, stream), "warm collation");
        }
        encoding::FlatSizes ring_sizes, collated_sizes;
        cuda::check(cudaMemcpyAsync(&ring_sizes, ring_out.sizes, sizeof(ring_sizes), cudaMemcpyDeviceToHost, stream), "ring sizes");
        cuda::check(cudaMemcpyAsync(&collated_sizes, out.sizes, sizeof(collated_sizes), cudaMemcpyDeviceToHost, stream), "collated sizes");
        cuda::check(cudaStreamSynchronize(stream), "warm stream");
        if (std::memcmp(&ring_sizes, &ring_expected, sizeof(ring_sizes)) ||
            std::memcmp(&collated_sizes, &base.host_sizes, sizeof(collated_sizes)))
            throw std::runtime_error("the device stream or collation sizes differ from the host encoding");
    }
    void measure(cudaStream_t stream, u32 replays)
    {
        ring.clear();
        append_times.push_back(replay_us(stream, append, 1));
        flush_times.push_back(replay_us(stream, flush, 1));
        append_flush_times.push_back(replay_us(stream, append_flush, replays));
        collate_times.push_back(replay_us(stream, collate, replays));
    }
};

void print_times(const char* name, const std::vector<double>& times)
{
    std::printf("\"%s\":[", name);
    for (usize i = 0; i < times.size(); ++i)
        std::printf("%s%.3f", i ? "," : "", times[i]);
    std::printf("]");
}

void print_streams(const char* name, const StreamSession& s)
{
    std::printf("\"%s\":{", name);
    print_times("append_us", s.append_times); std::printf(",");
    print_times("flush_us", s.flush_times); std::printf(",");
    print_times("append_flush_us", s.append_flush_times); std::printf(",");
    print_times("collate_us", s.collate_times);
    std::printf(",\"ring_capacity\":%llu,\"ring_rows\":%llu,\"collate_slices\":%llu",
                static_cast<unsigned long long>(s.ring.capacity()), static_cast<unsigned long long>(s.input.graphs),
                static_cast<unsigned long long>(s.slices));
    std::printf(",\"append_kernels\":%u,\"flush_kernels\":%u,\"append_flush_kernels\":%u,\"collate_kernels\":%u,"
                "\"collate_scratch_bytes\":%llu}", s.append.nodes.kernels, s.flush.nodes.kernels,
                s.append_flush.nodes.kernels, s.collate.nodes.kernels,
                static_cast<unsigned long long>(s.collator->scratch_bytes()));
}

void print_layout(const char* name, const Session& s)
{
    std::printf("\"%s\":{\"kernels\":%u,\"memsets\":%u,\"other_nodes\":%u,\"scratch_bytes\":%llu,\"cub_bytes\":%llu,"
                "\"thread_fills\":%u,\"group_fills\":%u,\"warp_fills\":%u,\"cells\":%llu,\"relations\":%llu,\"wide\":%s,\"output_bytes\":%llu,\"sizes_match\":%s,"
                "\"entities\":%llu,\"tuples\":%llu,\"args\":%llu,\"targets\":%llu,\"history\":%llu,"
                "\"stores_relations\":%s,\"stores_simple\":%s}",
                name, s.nodes.kernels, s.nodes.memsets, s.nodes.others,
                static_cast<unsigned long long>(s.scratch.bytes), static_cast<unsigned long long>(s.scratch.scan_temp_bytes),
                s.nodes.thread_fills, s.nodes.group_fills, s.nodes.warp_fills, static_cast<unsigned long long>(s.cells), static_cast<unsigned long long>(s.scratch.relations),
                s.scratch.wide ? "true" : "false", static_cast<unsigned long long>(s.layout_of.sizes),
                s.sizes_match ? "true" : "false", static_cast<unsigned long long>(s.host_sizes.entities),
                static_cast<unsigned long long>(s.host_sizes.tuples), static_cast<unsigned long long>(s.host_sizes.args),
                static_cast<unsigned long long>(s.host_sizes.targets), static_cast<unsigned long long>(s.host_sizes.history),
                s.host->schema().emits_relations() ? "true" : "false", s.host->schema().emits_simple() ? "true" : "false");
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args a = parse(argc, argv);
        const TaskPtr task = Task::create(*frontend::load_task(a.domain, a.problem));
        std::vector<TaskPtr> tasks{task};
        if (!a.secondary_problem.empty())
            tasks.push_back(Task::create(*frontend::load_task(a.domain, a.secondary_problem)));
        for (u32 k = 1; k < a.instances; ++k)
            tasks.push_back(Task::create(*frontend::load_task(a.domain, a.problem)));
        const auto table = rl::TaskTable::create(tasks);
        const u32 words = std::max(a.words, table->words());
        const auto raw = read_rows(a.rows, a.words, a.n);
        std::vector<u64> rows(a.n * words, 0);
        for (u64 g = 0; g < a.n; ++g)
            std::copy_n(raw.data() + g * a.words, a.words, rows.data() + g * words);
        cuda::ContextOptions options;
        options.max_bytes = u64{3} << 30;
        const cuda::ContextPtr ctx = cuda::DeviceContext::create(a.device, options);
        cuda::Stream stream;
        const LayerRows layers(a, rows, words, table->task(0)->atoms().fluent_slots());
        Session graph_major(ctx, stream, table, encoding::Layout::Graph, rows, words, a.n, a.launch, a, layers);
        Session relation_major(ctx, stream, table, encoding::Layout::Relation, rows, words, a.n, a.launch, a, layers);
        graph_major.time_host(a.cpu_threads, a.rounds);
        std::unique_ptr<Session> graph_given, relation_given;
        if (graph_major.host->context().max_derived_words())
        {
            graph_given = std::make_unique<Session>(ctx, stream, table, encoding::Layout::Graph, rows, words, a.n, a.launch,
                                                    a, layers, true);
            relation_given = std::make_unique<Session>(ctx, stream, table, encoding::Layout::Relation, rows, words, a.n,
                                                       a.launch, a, layers, true);
        }
        std::unique_ptr<StreamSession> stream_graph, stream_relation;
        if (a.streams)
        {
            stream_graph = std::make_unique<StreamSession>(ctx, stream, graph_major, rows, a.ring_rows, a.collate_slices);
            stream_relation = std::make_unique<StreamSession>(ctx, stream, relation_major, rows, a.ring_rows, a.collate_slices);
        }
        std::vector<double> graph_times, relation_times, eager_times, graph_given_times, relation_given_times;
        for (u32 r = 0; r < a.rounds; ++r)
        {
            if (stream_graph)
            {
                (r % 2 ? stream_relation : stream_graph)->measure(stream, a.replays);
                (r % 2 ? stream_graph : stream_relation)->measure(stream, a.replays);
            }
            // The first timing rotates so each mode occupies every position across rounds.
            const u32 modes = graph_given ? 5 : 3;
            for (u32 k = 0; k < modes; ++k)
                switch ((r + k) % modes)
                {
                    case 0: graph_times.push_back(replay_us(stream, graph_major, a.replays)); break;
                    case 1: relation_times.push_back(replay_us(stream, relation_major, a.replays)); break;
                    case 2: eager_times.push_back(eager_us(stream, graph_major)); break;
                    case 3: graph_given_times.push_back(replay_us(stream, *graph_given, a.replays)); break;
                    default: relation_given_times.push_back(replay_us(stream, *relation_given, a.replays)); break;
                }
        }
        std::printf("{\"outputs\":\"%s\",\"n\":%llu,\"words\":%u,\"instances\":%u,\"replays\":%u,\"rounds\":%u,\"times_us\":{",
                    a.outputs.c_str(), static_cast<unsigned long long>(a.n), a.words, table->size(), a.replays, a.rounds);
        print_times("graph", graph_times);
        std::printf(",");
        print_times("relation", relation_times);
        std::printf(",");
        print_times("eager", eager_times);
        std::printf(",");
        print_times("cpu_graph_major_us", graph_major.cpu_times);
        if (graph_given)
        {
            std::printf(","); print_times("graph_given", graph_given_times);
            std::printf(","); print_times("relation_given", relation_given_times);
        }
        std::printf("},");
        print_layout("graph", graph_major);
        std::printf(",");
        print_layout("relation", relation_major);
        if (graph_given)
        {
            std::printf(","); print_layout("graph_given", *graph_given);
            std::printf(","); print_layout("relation_given", *relation_given);
        }
        if (stream_graph)
        {
            std::printf(",\"streams\":{");
            print_streams("graph", *stream_graph); std::printf(","); print_streams("relation", *stream_relation);
            std::printf("}");
        }
        std::printf("}\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::string message = e.what();
        for (char& c : message)
            if (c == '"' || c == '\\' || c == '\n')
                c = ' ';
        std::fprintf(stderr, "error: %s\n", message.c_str());
        std::printf("{\"error\":\"%s\"}\n", message.c_str());
        return 1;
    }
}
