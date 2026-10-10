#pragma once
// mymyr.Domain's C++ side, shared by the front-end bindings and mymyr.Task.from_pddl. Only built with the front end.

#if defined(MYMYR_HAS_FRONTEND)
#include "mymyr/frontend/domain.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

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

/// Refuses PDDL text that defines the other kind than `kind` ("domain" or "problem"): a frontend::PddlError naming
/// `path` (empty for a string) when a problem is given where the domain belongs or the converse. Text that starts
/// with neither is left to the parser.
void expect_pddl(std::string_view text, std::string_view kind, const std::string& path);

/// Reads and parses a domain file (expect_pddl, then frontend::Domain::from_file).
[[nodiscard]] PyDomain load_domain(const std::filesystem::path& path);

/// Reads a problem file's text (expect_pddl).
[[nodiscard]] std::string read_problem(const std::filesystem::path& path);
}  // namespace mymyr::python
#endif
