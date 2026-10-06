// mymyr.Domain: the loki PDDL front end. A domain is parsed and normalized once; problems
// are instantiated from it many times, from any number of threads. Each instantiation returns the normalized task as
// mymyr.formalism.NormalizedTask, which mymyr.Task compiles. The NormalizedTask remembers its PDDL text, so a Task
// built from it pickles as its source.

#include "formalism_task.hpp"
#include "py_domain.hpp"

#include "mymyr/frontend/domain.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace mymyr::python
{
std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("mymyr: cannot open '" + path + "'");
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

namespace
{
std::shared_ptr<const TaskSource> pddl_source(const PyDomain& d, std::string problem, std::string problem_path)
{
    auto s = std::make_shared<TaskSource>();
    s->kind = TaskSource::Kind::Pddl;
    s->domain = d.text ? *d.text : std::string();
    s->domain_path = d.path;
    s->problem = std::move(problem);
    s->problem_path = std::move(problem_path);
    return s;
}
}  // namespace

void bind_frontend(nb::module_& m)
{
    nb::class_<PyDomain>(m, "Domain",
                         "A PDDL domain, parsed and normalized once by loki. Instantiating problems is thread-safe.")
        .def_static(
            "from_file",
            [](const std::filesystem::path& path) {
                nb::gil_scoped_release release;
                auto text = std::make_shared<const std::string>(read_file(path.string()));
                return PyDomain{frontend::Domain::from_file(path), std::move(text), path.string()};
            },
            "path"_a, "Parse and normalize the domain file.")
        .def_static(
            "from_string",
            [](std::string_view text) {
                auto copy = std::make_shared<const std::string>(text);
                nb::gil_scoped_release release;
                return PyDomain{frontend::Domain::from_string(*copy), copy, ""};
            },
            "text"_a, "Parse and normalize domain PDDL text.")
        .def(
            "instantiate",
            [](const PyDomain& self, const std::filesystem::path& problem, bool fast_init) {
                nb::gil_scoped_release release;
                std::string text = read_file(problem.string());
                auto data = self.d->instantiate_file(problem, {.fast_init = fast_init});
                return FormalismTask{std::move(data), pddl_source(self, std::move(text), problem.string())};
            },
            "problem"_a, nb::kw_only(), "fast_init"_a = true,
            "Instantiate a problem file. fast_init reads :init in one pass instead of through loki; the result is the "
            "same.")
        .def(
            "instantiate_string",
            [](const PyDomain& self, std::string_view text, bool fast_init) {
                std::string copy(text);
                nb::gil_scoped_release release;
                auto data = self.d->instantiate_string(copy, "", {.fast_init = fast_init});
                return FormalismTask{std::move(data), pddl_source(self, std::move(copy), "")};
            },
            "text"_a, nb::kw_only(), "fast_init"_a = true, "Instantiate problem PDDL text.")
        .def_prop_ro("name", [](const PyDomain& self) { return self.d->name(); })
        .def_prop_ro("path", [](const PyDomain& self) { return self.d->path(); })
        .def("__repr__", [](const PyDomain& self) { return "Domain(" + self.d->name() + ")"; });
}
}  // namespace mymyr::python
