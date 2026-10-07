// Parses a task, runs IW on it and exits 0 when it is solved.
//   consumer <domain.pddl> <problem.pddl>   (linked against mymyr::frontend)
//   consumer <task.txt>                     (core only)
#include "mymyr/search/iw.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/novelty/minimum_g_table.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/version.hpp"
#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"
#endif

#include <cstdio>
#include <exception>
#include <memory>

int main(int argc, char** argv)
{
    try
    {
#if defined(MYMYR_HAS_FRONTEND)
        if (argc != 3)
        {
            std::fprintf(stderr, "usage: consumer <domain.pddl> <problem.pddl>\n");
            return 2;
        }
        const auto data = mymyr::frontend::load_task(argv[1], argv[2]);
        const auto task = mymyr::Task::create(*data);
#else
        if (argc != 2)
        {
            std::fprintf(stderr, "usage: consumer <task.txt>\n");
            return 2;
        }
        const auto task = mymyr::Task::from_text_file(argv[1]);
#endif
        const mymyr::search::IwResult result = mymyr::search::iw(*task);
        std::printf("mymyr %.*s: IW %s, plan length %zu\n", static_cast<int>(mymyr::version().size()), mymyr::version().data(),
                    mymyr::search::to_string(result.status), result.plan.size());
        mymyr::search::AStarIwOptions options;
        options.width = 2;
        const auto astar = mymyr::search::astar_iw(*task, options);
        mymyr::novelty::MinimumGNoveltyTable labels(1);
        labels.test_and_update(task->initial_state().view(), 0);
        if (astar.status != mymyr::search::SearchStatus::Solved)
            return 1;
        return result.status == mymyr::search::SearchStatus::Solved ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
