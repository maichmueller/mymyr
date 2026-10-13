// mymyr_export_task: instantiate a PDDL problem with mymyr's front end and write it in the fork exporters' text format
// (formalism/text_format.hpp), for diffing against the fork and for timing.
//
//   mymyr_export_task domain.pddl problem.pddl [out.txt] [--full-init] [--reps N] [--golden fork.txt canon-prefix]
//
// With --golden, also writes <canon-prefix>.ours and <canon-prefix>.golden: both tasks in the canonical form of the
// golden tests (golden.hpp), for diffing.
// Prints one JSON line with the timings: domain parse+normalize, and the best of N problem instantiations.

#include "golden.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::vector<std::string> pos;
    bool full = false;
    int reps = 1;
    std::string golden, canon;
    bool debug_axioms = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--golden") == 0 && i + 2 < argc)
        {
            golden = argv[++i];
            canon = argv[++i];
        }
        else if (std::strcmp(argv[i], "--axioms") == 0)
            debug_axioms = true;
        else if (std::strcmp(argv[i], "--full-init") == 0)
            full = true;
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            reps = std::max(1, std::atoi(argv[++i]));
        else
            pos.emplace_back(argv[i]);
    }
    if (pos.size() < 2)
    {
        std::fprintf(stderr, "usage: mymyr_export_task domain.pddl problem.pddl [out.txt] [--full-init] [--reps N]\n");
        return 2;
    }
    try
    {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        const auto domain = mymyr::frontend::Domain::from_file(pos[0]);
        const double domain_s = std::chrono::duration<double>(clock::now() - t0).count();
        mymyr::frontend::InstantiateOptions opt;
        opt.fast_init = !full;
        double best = 1e300;
        mymyr::frontend::TaskPtr task;
        for (int r = 0; r < reps; ++r)
        {
            const auto t1 = clock::now();
            task = domain->instantiate_file(pos[1], opt);
            best = std::min(best, std::chrono::duration<double>(clock::now() - t1).count());
        }
        if (pos.size() >= 3)
            std::ofstream(pos[2]) << mymyr::formalism::write_task_text(*task);
        if (debug_axioms)
        {
            const auto& t = *task;
            for (const auto& x : t.axioms)
            {
                std::printf("axiom %s(", std::string(t.str(t.predicates[x.head.pred.v].name)).c_str());
                for (const auto tt : t.terms_of(x.head))
                    std::printf(" %d", tt);
                std::printf(" ) params:");
                for (const auto& p : mymyr::formalism::TaskData::slice(t.params, x.params))
                    std::printf(" %s", std::string(t.str(p.name)).c_str());
                std::printf("\n");
                for (const auto& l : t.literals_of(x.body))
                {
                    std::printf("    %s%s", l.positive ? "" : "not ", std::string(t.str(t.predicates[l.pred.v].name)).c_str());
                    for (const auto tt : t.terms_of(l))
                        std::printf(" %d", tt);
                    std::printf("\n");
                }
            }
        }
        if (!golden.empty())
        {
            const auto g = mymyr::formalism::read_task_text_file(golden);
            const auto tp = mymyr::test::type_predicates(*task);
            std::ofstream(canon + ".ours") << mymyr::test::canonical_text(*task, tp);
            std::ofstream(canon + ".golden") << mymyr::test::canonical_text(g, tp);
        }
        std::printf("{\"domain_s\": %.6f, \"instantiate_s\": %.6f, \"objects\": %zu, \"static_init\": %zu, \"fluent_init\": %zu, "
                    "\"schemas\": %zu, \"axioms\": %zu, \"fast_init\": %s}\n",
                    domain_s, best, task->objects.size(), task->static_init.size(), task->fluent_init.size(), task->schemas.size(),
                    task->axioms.size(), full ? "false" : "true");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
