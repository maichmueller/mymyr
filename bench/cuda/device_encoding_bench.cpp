// mymyr_device_encoding_bench: the device relational encoder on the rows of one PDDL task, one JSON object on standard
// output.
//
//   mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W [--n 16384] [--rounds 20]
//                               [--replays 10] [--device 0] [--secondary-problem P.pddl]
//                               [--lanes 4|8|32] [--fill auto|thread|warp] [--streams]
//
// FILE holds the state rows, little-endian u64 [rows, W]; they are tiled to --n graphs. For each of the graph-major and
// the relation-major layout the encode of the batch is captured into a CUDA graph on a stream and replayed --replays
// times between two events per round (the time of one round), and the graph-major encode is also timed as one eager call
// followed by a synchronize on the host clock. The rounds of the three timings alternate, the first rotating. The object
// also reports, for each layout, the nodes of the captured graph (kernels, memsets, anything else), the scratch of the
// reserved workspace, and whether the device's sizes equal the host encoder's.

// With --streams, also times captured append, flush, append plus flush and four-slice collation. Append and flush
// arrays hold one replay per sample; combined and collation arrays hold --replays calls per sample.

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
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
[[noreturn]] void usage(const char* message)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W "
                 "[--n N] [--rounds R] [--replays K] [--device I] [--lanes L] [--fill auto|thread|warp] [--streams]\n",
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
    bool streams = false;
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
        else if (flag == "--streams")
            a.streams = true;
        else if (flag == "--device")
            a.device = std::stoi(value());
        else if (flag == "--lanes")
            a.launch.count_lanes = static_cast<u32>(std::stoul(value()));
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

    Session(const cuda::ContextPtr& ctx, cudaStream_t stream, rl::TaskTablePtr table, encoding::Layout l, const std::vector<u64>& rows,
            u32 words, u64 n, const cuda::LaunchConfig& config)
        : layout(l), launch(config)
    {
        encoding::FlatOptions options;
        options.goal_derivations = {};
        options.layout = l;
        host = encoding::FlatEncoder::create(std::move(table), options);
        // the host's sizes size the output
        encoding::EncodeInput in;
        in.states = {rows.data(), n, words, 0, 0};
        ThreadPool pool(1);
        encoding::EncodeScratch scratch_of_host;
        host_sizes = host->count(in, pool, scratch_of_host);
        const cuda::BatchShape shape{n, host_sizes.entities, host_sizes.args, host_sizes.targets};
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
        // an eager encode first, outside any capture, for the sizes
        device->encode(input, output, stream, launch);
        encoding::FlatSizes device_sizes;
        cuda::check(cudaMemcpyAsync(&device_sizes, output.sizes, sizeof device_sizes, cudaMemcpyDeviceToHost, stream), "sizes");
        cuda::check(cudaStreamSynchronize(stream), "sizes");
        sizes_match = device_sizes.entities == host_sizes.entities && device_sizes.tuples == host_sizes.tuples &&
                      device_sizes.args == host_sizes.args && device_sizes.targets == host_sizes.targets &&
                      device_sizes.overflow == 0 && host_sizes.overflow == 0;
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

    StreamSession(const cuda::ContextPtr& ctx, cudaStream_t stream, const Session& base, const std::vector<u64>& rows)
        : ring(ctx, base.host, base.input.graphs + 1, stream), input(base.input)
    {
        const auto& sizes = base.host_sizes;
        const cuda::BatchShape ring_shape{input.graphs + 1, sizes.entities, sizes.args, sizes.targets};
        const auto ring_layout = base.device->output_layout(ring_shape);
        ring_output = cuda::DeviceBuffer(ctx, ring_layout.bytes, stream);
        const auto ring_out = cuda::output_of(ring_layout, ring_shape, ring_output.data());
        append.capture(stream, [&] { ring.append(input); });
        flush.capture(stream, [&] { ring.flush(ring_out); });
        append_flush.capture(stream, [&] { ring.append(input); ring.flush(ring_out); });
        std::vector<cuda::DeviceCollationBatch> batches;
        u64 first = 0;
        ThreadPool pool(1);
        encoding::EncodeScratch host_scratch;
        for (const u64 count : {input.graphs / 3, u64{0}, input.graphs / 3, input.graphs - 2 * (input.graphs / 3)})
        {
            encoding::EncodeInput host_input;
            host_input.states = {rows.data() + first * input.state_stride, count, input.state_words, input.state_stride, 0};
            const auto part_sizes = base.host->count(host_input, pool, host_scratch);
            const cuda::BatchShape shape{count, part_sizes.entities, part_sizes.args, part_sizes.targets};
            const auto layout = base.device->output_layout(shape);
            parts.emplace_back(ctx, layout.bytes, stream);
            const auto out = cuda::output_of(layout, shape, parts.back().data());
            auto part = input;
            part.states += first * input.state_stride;
            part.graphs = count;
            base.device->encode(part, out, stream);
            cuda::DeviceCollationBatch batch;
            batch.arrays = out.arrays;
            batch.sizes = out.sizes;
            batch.graphs = count;
            batch.relations = base.host->schema().num_relations();
            batch.schema_fingerprint = base.host->schema().fingerprint();
            batch.layout = base.layout;
            batch.index = base.host->schema().options().index;
            batch.stored = cuda::output_arrays(base.host->schema());
            batches.push_back(batch);
            first += count;
        }
        collator = std::make_unique<cuda::DeviceCollator>(ctx, std::move(batches), stream);
        scratch = cuda::DeviceBuffer(ctx, collator->scratch_bytes(), stream);
        const cuda::BatchShape shape{input.graphs, sizes.entities, sizes.args, sizes.targets};
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
        if (std::memcmp(&ring_sizes, &base.host_sizes, sizeof(ring_sizes)) ||
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
    std::printf(",\"append_kernels\":%u,\"flush_kernels\":%u,\"append_flush_kernels\":%u,\"collate_kernels\":%u,"
                "\"collate_scratch_bytes\":%llu}", s.append.nodes.kernels, s.flush.nodes.kernels,
                s.append_flush.nodes.kernels, s.collate.nodes.kernels,
                static_cast<unsigned long long>(s.collator->scratch_bytes()));
}

void print_layout(const char* name, const Session& s)
{
    std::printf("\"%s\":{\"kernels\":%u,\"memsets\":%u,\"other_nodes\":%u,\"scratch_bytes\":%llu,\"cub_bytes\":%llu,"
                "\"thread_fills\":%u,\"warp_fills\":%u,\"cells\":%llu,\"relations\":%llu,\"wide\":%s,\"output_bytes\":%llu,\"sizes_match\":%s,"
                "\"entities\":%llu,\"tuples\":%llu,\"args\":%llu,\"targets\":%llu}",
                name, s.nodes.kernels, s.nodes.memsets, s.nodes.others,
                static_cast<unsigned long long>(s.scratch.bytes), static_cast<unsigned long long>(s.scratch.scan_temp_bytes),
                s.nodes.thread_fills, s.nodes.warp_fills, static_cast<unsigned long long>(s.cells), static_cast<unsigned long long>(s.scratch.relations),
                s.scratch.wide ? "true" : "false", static_cast<unsigned long long>(s.layout_of.sizes),
                s.sizes_match ? "true" : "false", static_cast<unsigned long long>(s.host_sizes.entities),
                static_cast<unsigned long long>(s.host_sizes.tuples), static_cast<unsigned long long>(s.host_sizes.args),
                static_cast<unsigned long long>(s.host_sizes.targets));
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
        Session graph_major(ctx, stream, table, encoding::Layout::Graph, rows, words, a.n, a.launch);
        Session relation_major(ctx, stream, table, encoding::Layout::Relation, rows, words, a.n, a.launch);
        std::unique_ptr<StreamSession> stream_graph, stream_relation;
        if (a.streams)
        {
            stream_graph = std::make_unique<StreamSession>(ctx, stream, graph_major, rows);
            stream_relation = std::make_unique<StreamSession>(ctx, stream, relation_major, rows);
        }
        std::vector<double> graph_times, relation_times, eager_times;
        for (u32 r = 0; r < a.rounds; ++r)
        {
            if (stream_graph)
            {
                (r % 2 ? stream_relation : stream_graph)->measure(stream, a.replays);
                (r % 2 ? stream_graph : stream_relation)->measure(stream, a.replays);
            }
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
        std::printf("},");
        print_layout("graph", graph_major);
        std::printf(",");
        print_layout("relation", relation_major);
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
