// mymyr_device_encoding_bench: the device relational encoder on the rows of one PDDL task, one JSON object on standard
// output.
//
//   mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W [--n 16384] [--rounds 20]
//                               [--replays 10] [--device 0] [--secondary-problem P.pddl]
//                               [--lanes 4|8|32] [--fill auto|thread|warp]
//                               [--levels L] [--history H] [--max-steps M] [--literals K] [--targets]
//                               [--cpu-threads T]
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
// threads is reported as cpu_times_us (one entry per round, the whole batch).

#include "mymyr/cuda/encoding.hpp"
#include "../../cuda/src/encoding_scratch.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/frontend/domain.hpp"

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
                 "[--n N] [--rounds R] [--replays K] [--device I] [--lanes L] [--fill auto|thread|warp] "
                 "[--levels L] [--history H] [--max-steps M] [--literals K] [--targets] [--cpu-threads T]\n",
                 message);
    std::exit(2);
}

struct Args
{
    std::string domain, problem, secondary_problem, rows;
    u32 words = 0;
    u64 n = 16384;
    u32 rounds = 20, replays = 10;
    int device = 0;
    cuda::LaunchConfig launch;
    u32 levels = 0, history = 0, max_steps = 0, literals = 6, cpu_threads = 1;
    bool targets = false;
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
        if (flag == "--domain")
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
            else if (mode == "warp")
                a.launch.fill = cuda::FillMode::Warp;
            else
                usage("--fill takes auto, thread or warp");
        }
        else
            usage(("unknown flag " + flag).c_str());
    }
    if (a.domain.empty() || a.problem.empty() || a.rows.empty() || a.words == 0 || a.n == 0 || a.rounds == 0 ||
        a.replays == 0)
        usage("--domain, --problem, --rows and --words are required, the counts must be positive");
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
    u32 kernels = 0, memsets = 0, others = 0, thread_fills = 0, warp_fills = 0;
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
            const auto fill = name.find("fill_kernel");
            if (fill != std::string::npos)
                (name.find("Lb1E", fill) < name.find("Lb0E", fill) ? counts.warp_fills : counts.thread_fills) += 1;
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
    cuda::DeviceBuffer states, block;
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
            u32 words, u64 n, const cuda::LaunchConfig& config, const Args& args, const LayerRows& layers)
        : layout(l), launch(config)
    {
        encoding::FlatOptions options;
        options.goal_derivations = {};
        options.layout = l;
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
        if (host->context().max_derived_words() != 0)
            throw std::runtime_error("the task has derived atoms; the device encoder takes their words as input");
        // the host's sizes size the output
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        in.subgoals = layers.subgoals(layers.layer_pos.data(), layers.layer_neg.data());
        in.history = layers.history_of(layers.step_pos.data(), layers.step_neg.data(), layers.dt.data());
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
        // an eager encode first, outside any capture, for the sizes
        device->encode(input, output, stream, launch);
        encoding::FlatSizes device_sizes;
        cuda::check(cudaMemcpyAsync(&device_sizes, output.sizes, sizeof device_sizes, cudaMemcpyDeviceToHost, stream), "sizes");
        cuda::check(cudaStreamSynchronize(stream), "sizes");
        sizes_match = device_sizes.entities == host_sizes.entities && device_sizes.tuples == host_sizes.tuples &&
                      device_sizes.args == host_sizes.args && device_sizes.targets == host_sizes.targets &&
                      device_sizes.history == host_sizes.history && device_sizes.overflow == 0 && host_sizes.overflow == 0;
        scratch = cuda::detail::scratch_layout(shape, device->bounds(), l, host->schema().num_relations());
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

double replay_us(cudaStream_t stream, const Session& s, u32 replays)
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

void print_times(const char* name, const std::vector<double>& times)
{
    std::printf("\"%s\":[", name);
    for (usize i = 0; i < times.size(); ++i)
        std::printf("%s%.3f", i ? "," : "", times[i]);
    std::printf("]");
}

void print_layout(const char* name, const Session& s)
{
    std::printf("\"%s\":{\"kernels\":%u,\"memsets\":%u,\"other_nodes\":%u,\"scratch_bytes\":%llu,\"cub_bytes\":%llu,"
                "\"thread_fills\":%u,\"warp_fills\":%u,\"cells\":%llu,\"relations\":%llu,\"wide\":%s,\"output_bytes\":%llu,\"sizes_match\":%s,"
                "\"entities\":%llu,\"tuples\":%llu,\"args\":%llu,\"targets\":%llu,\"history\":%llu}",
                name, s.nodes.kernels, s.nodes.memsets, s.nodes.others,
                static_cast<unsigned long long>(s.scratch.bytes), static_cast<unsigned long long>(s.scratch.scan_temp_bytes),
                s.nodes.thread_fills, s.nodes.warp_fills, static_cast<unsigned long long>(s.cells), static_cast<unsigned long long>(s.scratch.relations),
                s.scratch.wide ? "true" : "false", static_cast<unsigned long long>(s.layout_of.sizes),
                s.sizes_match ? "true" : "false", static_cast<unsigned long long>(s.host_sizes.entities),
                static_cast<unsigned long long>(s.host_sizes.tuples), static_cast<unsigned long long>(s.host_sizes.args),
                static_cast<unsigned long long>(s.host_sizes.targets), static_cast<unsigned long long>(s.host_sizes.history));
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args a = parse(argc, argv);
        const TaskPtr task = Task::create(*frontend::load_task(a.domain, a.problem));
        const auto table = a.secondary_problem.empty() ? rl::TaskTable::single(task) :
            rl::TaskTable::create({task, Task::create(*frontend::load_task(a.domain, a.secondary_problem))});
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
        std::vector<double> graph_times, relation_times, eager_times;
        for (u32 r = 0; r < a.rounds; ++r)
        {
            // the three timings alternate, the first rotating
            for (u32 k = 0; k < 3; ++k)
                switch ((r + k) % 3)
                {
                    case 0: graph_times.push_back(replay_us(stream, graph_major, a.replays)); break;
                    case 1: relation_times.push_back(replay_us(stream, relation_major, a.replays)); break;
                    default: eager_times.push_back(eager_us(stream, graph_major)); break;
                }
        }
        std::printf("{\"n\":%llu,\"words\":%u,\"replays\":%u,\"rounds\":%u,\"times_us\":{", static_cast<unsigned long long>(a.n),
                    a.words, a.replays, a.rounds);
        print_times("graph", graph_times);
        std::printf(",");
        print_times("relation", relation_times);
        std::printf(",");
        print_times("eager", eager_times);
        std::printf(",");
        print_times("cpu_graph_major_us", graph_major.cpu_times);
        std::printf("},");
        print_layout("graph", graph_major);
        std::printf(",");
        print_layout("relation", relation_major);
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
