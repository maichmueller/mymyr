// mymyr_brfs: exhaustive or depth-capped BrFS on an exported task text file or (when built with the PDDL front end)
// a domain/problem pair, printing one JSON line (with --layers the per-layer [expanded, generated, new] triples as
// "layer_counts", matching the fork's own per-layer counts).
//
//   mymyr_brfs (task.txt | --domain D.pddl --problem P.pddl) [--atoms auto|lazy|frozen] [--match auto|fixed|fc]
//              [--fc-arity K] [--no-witness] [--canonical|--no-canonical]
//              [--store auto|flat|chunked|compact|concurrent] [--threads T] [--nondet] [--max-states N]
//              [--depth D] [--layers] [--fp] [--stop-at-goal]
//              [--numeric-storage auto|f64|i32] [--quantum Q] [--tolerant]
//              [--order goal_count|goal_count_fewer] [--beam W] [--beam-mode all_tested|survivors_only|relaxed]
//              [--tie-seed S] [--chunk C]: the layer ordering and beam (search/layer_ordering.hpp)

#include "mymyr/formalism/text_format.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <sys/resource.h>

using namespace mymyr;

namespace
{
double peak_rss_mb()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<double>(ru.ru_maxrss) / 1048576.0;
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

[[noreturn]] void usage(const char* msg)
{
    std::fprintf(stderr,
                 "error: %s\nusage: mymyr_brfs (task.txt | --domain D --problem P) [--atoms auto|lazy|frozen]\n"
                 "       [--match auto|fixed|fc] [--fc-arity K] [--no-witness] [--canonical|--no-canonical]\n"
                 "       [--store auto|flat|chunked|compact|concurrent] [--threads T] [--nondet] [--max-states N]\n"
                 "       [--depth D] [--layers] [--fp] [--stop-at-goal] [--numeric-storage auto|f64|i32] [--quantum Q]\n"
                 "       [--tolerant] [--order goal_count|goal_count_fewer] [--beam W]\n"
                 "       [--beam-mode all_tested|survivors_only|relaxed] [--tie-seed S] [--chunk C]\n",
                 msg);
    std::exit(2);
}
}  // namespace

int main(int argc, char** argv)
{
    TaskOptions to;
    BrfsOptions bo;
    bo.stop_at_goal = false;  // whole spaces unless --stop-at-goal
    std::string task_file, domain, problem;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
                usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--atoms")
        {
            const std::string v = value();
            to.atoms = v == "lazy" ? TaskOptions::Atoms::Lazy : v == "frozen" ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Auto;
        }
        else if (a == "--match")
        {
            const std::string v = value();
            to.matching = v == "fixed" ? TaskOptions::Matching::FixedOrder
                          : v == "fc"  ? TaskOptions::Matching::ForwardChecking
                                       : TaskOptions::Matching::Auto;
        }
        else if (a == "--fc-arity")
            to.fc_auto_free_params = static_cast<u32>(std::stoul(value()));
        else if (a == "--no-witness")
            bo.witness_pruning = false;
        else if (a == "--canonical")
            bo.canonical_order = true;
        else if (a == "--no-canonical")
            bo.canonical_order = false;
        else if (a == "--store")
        {
            const std::string v = value();
            bo.store = v == "flat"         ? BrfsOptions::Store::Flat
                       : v == "chunked"    ? BrfsOptions::Store::Chunked
                       : v == "compact"    ? BrfsOptions::Store::Compact
                       : v == "concurrent" ? BrfsOptions::Store::Concurrent
                                           : BrfsOptions::Store::Auto;
        }
        else if (a == "--threads")
            bo.threads = static_cast<u32>(std::stoul(value()));
        else if (a == "--nondet")
            bo.deterministic_ids = false;
        else if (a == "--max-states")
            bo.max_states = std::stoull(value());
        else if (a == "--fp")
            bo.fingerprint = true;
        else if (a == "--stop-at-goal")
            bo.stop_at_goal = true;
        else if (a == "--depth")
            bo.max_depth = static_cast<u32>(std::stoul(value()));
        else if (a == "--layers")
            bo.layer_stats = true;
        else if (a == "--domain")
            domain = value();
        else if (a == "--problem")
            problem = value();
        else if (a == "--numeric-storage")
        {
            const std::string v = value();
            to.numeric_storage = v == "f64"   ? TaskOptions::NumericStorageMode::F64
                                 : v == "i32" ? TaskOptions::NumericStorageMode::I32
                                              : TaskOptions::NumericStorageMode::Auto;
        }
        else if (a == "--quantum")
            to.numeric_quantum = std::stod(value());
        else if (a == "--tolerant")
            to.numeric_tolerant = true;
        else if (a == "--order")
        {
            const std::string v = value();
            bo.layers.kind = search::LayerOrdering::Kind::GoalCount;
            bo.layers.prefer_more_satisfied_goals = v != "goal_count_fewer";
        }
        else if (a == "--beam")
            bo.layers.beam_width = static_cast<u32>(std::stoul(value()));
        else if (a == "--beam-mode")
        {
            const std::string v = value();
            bo.layers.beam_novelty = v == "survivors_only" ? search::LayerOrdering::BeamNovelty::SurvivorsOnly
                                : v == "relaxed"      ? search::LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly
                                                      : search::LayerOrdering::BeamNovelty::AllTested;
        }
        else if (a == "--tie-seed")
        {
            bo.layers.randomize_ties = true;
            bo.layers.seed = std::stoull(value());
        }
        else if (a == "--chunk")
            bo.layers.beam_chunk = static_cast<u32>(std::stoul(value()));
        else if (!a.starts_with("--") && task_file.empty())
            task_file = a;
        else
            usage(("unknown argument " + a).c_str());
    }
    if (task_file.empty() && (domain.empty() || problem.empty()))
        usage("missing task (task.txt or --domain/--problem)");
    try
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<const Task> task;
        if (!task_file.empty())
            task = Task::from_text_file(task_file, to);
        else
        {
#if defined(MYMYR_HAS_FRONTEND)
            const auto data = frontend::load_task(domain, problem);
            task = Task::create(*data, to);
#else
            usage("built without the PDDL front end: pass an exported task.txt");
#endif
        }
        const double prep = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const BrfsResult r = brfs(*task, bo);
        std::string layers;
        for (const auto& l : r.layer_counts)
            layers += (layers.empty() ? "[" : ",[") + std::to_string(l[0]) + "," + std::to_string(l[1]) + "," + std::to_string(l[2]) + "]";
        std::printf("{\"NF\":%u,\"NW\":%u,\"numeric_storage\":\"%s\",\"depth\":%u,", task->numeric_slots(), task->numeric_words(),
                    task->numeric_storage() == NumericStorage::I32 ? "i32" : "f64", r.layers);
        if (bo.layer_stats)
            std::printf("\"layer_counts\":[%s],", layers.c_str());
        std::printf("\"objects\":%u,\"schemas\":%u,\"atoms\":\"%s\",\"dense_F\":%llu,\"F\":%u,\"W\":%u,\"store\":\"%s\","
                    "\"threads\":%u,\"witness\":%s,\"canonical\":%s,\"states\":%llu,\"expanded\":%llu,\"generated\":%llu,"
                    "\"goal_states\":%llu,\"layers\":%u,\"exhausted\":%s,\"solved\":%s,\"plan_len\":%zu,\"prep_s\":%.4f,"
                    "\"search_s\":%.4f,\"states_per_s\":%.0f,\"fp\":\"%016llx\",\"store_mb\":%.1f,\"peak_rss_mb\":%.1f}\n",
                    task->num_objects(), task->num_schemas(), task->atoms().mode() == AtomMode::Frozen ? "frozen" : "lazy",
                    static_cast<unsigned long long>(task->info().dense_fluent), r.fluent_slots, r.words, r.store.c_str(),
                    r.threads, bo.witness_pruning ? "true" : "false", bo.canonical_order ? "true" : "false",
                    static_cast<unsigned long long>(r.states), static_cast<unsigned long long>(r.expanded),
                    static_cast<unsigned long long>(r.generated), static_cast<unsigned long long>(r.goal_states), r.layers,
                    r.exhausted ? "true" : "false", r.solved ? "true" : "false", r.plan.size(), prep, r.search_s,
                    r.search_s > 0 ? static_cast<double>(r.states) / r.search_s : 0.0,
                    static_cast<unsigned long long>(r.fingerprint), static_cast<double>(r.store_bytes) / 1048576.0, peak_rss_mb());
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
