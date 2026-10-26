#pragma once

#include "py_task.hpp"

#include "mymyr/search/iw.hpp"

namespace mymyr::python
{
/// Shared ownership of flat native tree arrays and the task that interprets their state rows and action labels.
struct PySearchTree
{
    std::shared_ptr<const search::SearchTree> tree;
    Owner o;
};
struct PyIwPass
{
    search::IwPassStatistics p;
    Owner o;
};

inline nb::typed<nb::list, PyIwPass> pass_list(const Owner& o, std::span<const search::IwPassStatistics> passes)
{
    nb::typed<nb::list, PyIwPass> out{nb::list()};
    for (const auto& p : passes)
        out.append(PyIwPass{p, o});
    return out;
}
}  // namespace mymyr::python
