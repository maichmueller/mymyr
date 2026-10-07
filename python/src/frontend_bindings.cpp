// mymyr.Domain: the loki PDDL front end. A domain is parsed and normalized once; problems
// are instantiated from it many times, from any number of threads. Each instantiation returns the normalized task as
// mymyr.formalism.NormalizedTask, which mymyr.Task compiles. The NormalizedTask remembers its PDDL text, so a Task
// built from it pickles as its source.
//
// Errors: frontend::PddlError becomes mymyr.PddlError (a ValueError with the path, line and action), and a file that
// cannot be read an OSError (FileNotFoundError when it does not exist).

#include "formalism_task.hpp"
#include "py_domain.hpp"

#include "mymyr/frontend/domain.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cerrno>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace nb = nanobind;
using namespace nb::literals;

namespace mymyr::python
{
std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::filesystem::filesystem_error("mymyr: cannot open the file", path,
                                                std::error_code(errno ? errno : ENOENT, std::generic_category()));
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

namespace
{
/// Raises the Python exception of a C++ exception the front end throws (see the top of the file).
void translate_frontend_error(const std::exception_ptr& p, void*)
{
    try
    {
        std::rethrow_exception(p);
    }
    catch (const frontend::PddlError& e)
    {
        nb::object cls = nb::module_::import_("mymyr._errors").attr("PddlError");
        auto or_none = [](const std::string& s) -> nb::object {
            if (s.empty())
                return nb::none();
            return nb::str(s.c_str(), s.size());
        };
        nb::object line = nb::none();
        if (e.line() != 0)
            line = nb::cast(e.line());
        nb::object err = cls(e.message(), or_none(e.path()), line, or_none(e.action()));
        PyErr_SetObject(cls.ptr(), err.ptr());
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        // OSError(errno, strerror, filename) is the subclass of the errno: FileNotFoundError for ENOENT
        const std::error_code& c = e.code();
        nb::object err = nb::handle(PyExc_OSError)(c.value(), c.message(), e.path1().string());
        PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(err.ptr())), err.ptr());
    }
}
}  // namespace

void bind_frontend(nb::module_& m)
{
    nb::register_exception_translator(translate_frontend_error);

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
