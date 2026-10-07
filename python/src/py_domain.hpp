#pragma once
// mymyr.Domain's C++ side, shared by the front-end bindings and mymyr.Task.from_pddl. Only built with the front end.

#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"

#include <memory>
#include <string>

namespace mymyr::python
{
/// Owns the immutable Domain and its PDDL text (the pickling source of the tasks it instantiates).
struct PyDomain
{
    std::shared_ptr<const frontend::Domain> d;
    std::shared_ptr<const std::string> text;
    std::string path;
};

/// Reads a whole file (throws std::filesystem::filesystem_error).
[[nodiscard]] std::string read_file(const std::string& path);
}  // namespace mymyr::python
#endif
