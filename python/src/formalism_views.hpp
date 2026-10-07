#pragma once
// The views of mymyr.formalism that other binding files create (formalism_bindings.cpp registers them): an entity of a
// normalized task is (task data, index).

#include "mymyr/formalism/task_data.hpp"

#include <memory>

namespace mymyr::python
{
struct FormalismView
{
    std::shared_ptr<const formalism::TaskData> t;
    u32 i = 0;
    [[nodiscard]] const formalism::TaskData& d() const { return *t; }
};

/// mymyr.formalism.Object
struct ObjectView : FormalismView
{
};
/// mymyr.formalism.Type
struct TypeView : FormalismView
{
};
/// mymyr.formalism.Predicate
struct PredicateView : FormalismView
{
};
/// mymyr.formalism.Function
struct FunctionView : FormalismView
{
};
}  // namespace mymyr::python
