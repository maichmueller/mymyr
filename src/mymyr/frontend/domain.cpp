#include "mymyr/frontend/domain.hpp"

#include "diagnose.hpp"
#include "init_reader.hpp"
#include "translate.hpp"

#include <loki/details/pddl/parser.hpp>
#include <loki/details/pddl/translator.hpp>
#include <loki/details/utils/filesystem.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <new>
#include <stdexcept>

namespace mymyr::frontend
{
namespace
{
/// Runs `f`, which reads `text` (the PDDL of `path`), and turns its errors into PddlError (diagnose.hpp). Allocation
/// failures and unreadable files propagate as they are.
template<class F>
auto reading(std::string_view text, const std::filesystem::path& path, F&& f)
{
    try
    {
        return f();
    }
    catch (const std::bad_alloc&)
    {
        throw;
    }
    catch (const std::filesystem::filesystem_error&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        throw detail::diagnose(e, text, path.string());
    }
}
}  // namespace

struct Domain::Impl
{
    std::filesystem::path path;
    std::unique_ptr<loki::Parser> parser;  // parse_problem mutates it: guarded by parse_mutex
    std::unique_ptr<loki::DomainTranslationResult> translation;
    std::unique_ptr<detail::DomainState> state;
    detail::InitOrderTables init_tables;
    DomainOptions options;
    detail::PredicateOrigin origin;  // declared: the predicates of the parsed domain (the others are loki's)
    mutable std::mutex parse_mutex;

    Impl(std::string text, std::filesystem::path p, DomainOptions o) : path(std::move(p)), options(std::move(o))
    {
        reading(text, path, [&] {
            detail::reject_dropped_constructs(text, path.string());
            parser = std::make_unique<loki::Parser>(text, path, loki::ParserOptions());
            translation = std::make_unique<loki::DomainTranslationResult>(loki::translate(parser->get_domain()));
            for (const auto& pred : parser->get_domain()->get_predicates())
                origin.declared.insert(pred->get_name());
            origin.argument_order = &options.generated_argument_order;
            state = detail::translate_domain(translation->get_translated_domain(), origin);
            init_tables = detail::make_init_order_tables(*state, parser->get_domain(), translation->get_translated_domain());
        });
    }

    TaskPtr instantiate(const std::string& text, const std::filesystem::path& problem_path, const InstantiateOptions& options) const
    {
        return reading(text, problem_path, [&] {
            detail::reject_dropped_constructs(text, problem_path.string());
            return instantiate_unchecked(text, problem_path, options);
        });
    }

    TaskPtr instantiate_unchecked(const std::string& text, const std::filesystem::path& problem_path,
                                  const InstantiateOptions& options) const
    {
        // profiling aid: MYMYR_FRONTEND_TIMING=1 prints the time of each stage to stderr
        static const bool timing = std::getenv("MYMYR_FRONTEND_TIMING") != nullptr;
        auto t0 = std::chrono::steady_clock::now();
        auto lap = [&](const char* what)
        {
            if (!timing)
                return;
            const auto t1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  %-14s %8.3f ms\n", what, std::chrono::duration<double, std::milli>(t1 - t0).count());
            t0 = t1;
        };
        const detail::TextRange section = options.fast_init ? detail::find_init_section(text) : detail::TextRange{};
        lap("find section");
        loki::Problem parsed;
        {
            const std::lock_guard lock(parse_mutex);
            parsed = section.empty() ? parser->parse_problem(text, problem_path, loki::ParserOptions())
                                     : parser->parse_problem(detail::without_section(text, section), problem_path, loki::ParserOptions());
        }
        lap("loki parse");
        const loki::Problem translated = loki::translate(parsed, *translation);
        lap("loki translate");
        const detail::ObjectMap objects = detail::map_objects(*state, translated);
        const detail::GroundInit init =
            section.empty() ? detail::read_loki_init(*state, translated, objects)
                            : detail::read_fast_init(*state, init_tables, translated, objects,
                                                     std::string_view(text).substr(section.begin, section.end - section.begin),
                                                     problem_path.string(), detail::line_of(text, section.begin));
        lap("init");
        detail::PredicateOrigin problem_origin = origin;
        for (const auto& pred : parsed->get_predicates())
            problem_origin.declared.insert(pred->get_name());
        auto out = std::make_shared<const formalism::TaskData>(detail::translate_problem(*state, translated, init, objects, problem_origin));
        lap("translate");
        return out;
    }
};

Domain::Domain(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}
Domain::~Domain() = default;

std::shared_ptr<const Domain> Domain::from_file(const std::filesystem::path& path, const DomainOptions& options)
{
    return std::shared_ptr<const Domain>(new Domain(std::make_unique<Impl>(detail::read_pddl_file(path), path, options)));
}

std::shared_ptr<const Domain> Domain::from_string(std::string_view text, const std::filesystem::path& path, const DomainOptions& options)
{
    return std::shared_ptr<const Domain>(new Domain(std::make_unique<Impl>(detail::preprocess_pddl(text), path, options)));
}

TaskPtr Domain::instantiate_file(const std::filesystem::path& problem_path, const InstantiateOptions& options) const
{
    return m_impl->instantiate(detail::read_pddl_file(problem_path), problem_path, options);
}

TaskPtr Domain::instantiate_string(std::string_view text, const std::filesystem::path& path, const InstantiateOptions& options) const
{
    return m_impl->instantiate(detail::preprocess_pddl(text), path, options);
}

const formalism::TaskData& Domain::domain_data() const { return m_impl->state->data; }
const std::string& Domain::name() const { return m_impl->state->data.domain_name; }
const std::filesystem::path& Domain::path() const { return m_impl->path; }

TaskPtr load_task(const std::filesystem::path& domain_path, const std::filesystem::path& problem_path, const InstantiateOptions& options)
{
    return Domain::from_file(domain_path)->instantiate_file(problem_path, options);
}
}  // namespace mymyr::frontend
