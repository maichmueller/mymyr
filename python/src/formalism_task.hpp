#pragma once
// Python-facing handle of a normalized task (mymyr.formalism.NormalizedTask): owns the TaskData. Shared by the
// formalism bindings, the front end (mymyr.Domain) and mymyr.Task, which compiles it.

#include "mymyr/formalism/task_data.hpp"

#include <memory>
#include <string>

namespace mymyr::python
{
/// Where a normalized task came from, so that a mymyr.Task pickles as its source.
struct TaskSource
{
    enum class Kind : unsigned char
    {
        Pddl,  // domain + problem PDDL text (rebuilt through the loki front end)
        Text,  // the normalized task in the text format (formalism/text_format.hpp)
    };
    Kind kind = Kind::Text;
    std::string domain, problem;  // Pddl: the two texts; Text: `domain` holds the task text
    std::string domain_path, problem_path;  // for messages only
};

struct FormalismTask
{
    std::shared_ptr<const formalism::TaskData> t;
    std::shared_ptr<const TaskSource> source;  // null: unknown (pickling falls back to the text format)
};
}  // namespace mymyr::python
