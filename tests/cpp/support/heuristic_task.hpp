#pragma once
// Test support: an operator-free task with one initially true fluent per object.

#include "mymyr/task/task.hpp"

namespace mymyr::test
{
inline TaskPtr initial_proposition_task(u32 n)
{
    formalism::TaskData t;
    t.domain_name = "initial-propositions";
    t.problem_name = "initial-propositions";
    t.params.push_back({t.intern_string("x"), {}});
    t.predicates.push_back({t.intern_string("p"), formalism::PredKind::Fluent, 1, {0, 1}});
    for (u32 i = 0; i < n; ++i)
    {
        t.objects.push_back({t.intern_string("o" + std::to_string(i)), {}, false});
        t.object_ids.push_back(ObjectId{i});
        t.fluent_init.push_back({PredicateId{0}, {i, 1}});
    }
    return Task::create(std::move(t));
}
}  // namespace mymyr::test
