// mymyr_device_encoding_bench: the device relational encoder on the rows of one PDDL task, one JSON object on standard
// output.
//
//   mymyr_device_encoding_bench --domain D.pddl --problem P.pddl --rows FILE --words W [--n 16384] [--rounds 20]
//                               [--replays 10] [--device 0]
//
// FILE holds the state rows, little-endian u64 [rows, W]; they are tiled to --n graphs. For each of the graph-major and
// the relation-major layout the encode of the batch is captured into a CUDA graph on a stream and replayed --replays
// times between two events per round (the time of one round), and the graph-major encode is also timed as one eager call
// followed by a synchronize on the host clock. The rounds of the three timings alternate, the first rotating. The object
// also reports, for each layout, the nodes of the captured graph (kernels, memsets, anything else), the scratch of the
// reserved workspace, and whether the device's sizes equal the host encoder's.

#include "mymyr/cuda/encoding.hpp"
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
                 "[--n N] [--rounds R] [--replays K] [--device I]\n",
                 message);
    std::exit(2);
}

struct Args
{
    std::string domain, problem, rows;
    u32 words = 0;
    u64 n = 16384;
    u32 rounds = 20, replays = 10;
    int device = 0;
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
    u32 kernels = 0, memsets = 0, others = 0;
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
    cuda::ScratchLayout scratch;
    u64 cells = 0;
    bool sizes_match = false;

    Session(const cuda::ContextPtr& ctx, cudaStream_t stream, TaskPtr task, encoding::Layout l, const std::vector<u64>& rows,
            u32 words, u64 n)
        : layout(l)
    {
        encoding::FlatOptions options;
        options.goal_derivations = {};
        options.layout = l;
        host = encoding::FlatEncoder::create(rl::TaskTable::single(task), options);
        if (host->context().max_derived_words() != 0)
            throw std::runtime_error("the task has derived atoms; the device encoder takes their words as input");
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
        device->encode(input, output, stream);
        encoding::FlatSizes device_sizes;
        cuda::check(cudaMemcpyAsync(&device_sizes, output.sizes, sizeof device_sizes, cudaMemcpyDeviceToHost, stream), "sizes");
        cuda::check(cudaStreamSynchronize(stream), "sizes");
        sizes_match = device_sizes.entities == host_sizes.entities && device_sizes.tuples == host_sizes.tuples &&
                      device_sizes.args == host_sizes.args && device_sizes.targets == host_sizes.targets &&
                      device_sizes.overflow == 0 && host_sizes.overflow == 0;
        scratch = device->scratch_layout(shape);
        cells = device->bounds().cells;
        // the captured encode
        work.emplace(device->acquire(stream));
        cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        std::exception_ptr error;
        try { device->encode(*work, input, output); }
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
    s.device->encode(s.input, s.output, stream);
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
                "\"cells\":%llu,\"relations\":%llu,\"wide\":%s,\"output_bytes\":%llu,\"sizes_match\":%s,"
                "\"entities\":%llu,\"tuples\":%llu,\"args\":%llu,\"targets\":%llu}",
                name, s.nodes.kernels, s.nodes.memsets, s.nodes.others,
                static_cast<unsigned long long>(s.scratch.bytes), static_cast<unsigned long long>(s.scratch.scan_temp_bytes),
                static_cast<unsigned long long>(s.cells), static_cast<unsigned long long>(s.scratch.relations),
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
        const std::vector<u64> rows = read_rows(a.rows, a.words, a.n);
        cuda::ContextOptions options;
        options.max_bytes = u64{3} << 30;
        const cuda::ContextPtr ctx = cuda::DeviceContext::create(a.device, options);
        cuda::Stream stream;
        Session graph_major(ctx, stream, task, encoding::Layout::Graph, rows, a.words, a.n);
        Session relation_major(ctx, stream, task, encoding::Layout::Relation, rows, a.words, a.n);
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
