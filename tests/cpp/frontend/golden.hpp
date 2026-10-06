#pragma once
// Golden data of the front-end tests: the suite and numeric tasks, where their PDDL lives, the fork exports they are
// compared against, and the comparison itself.
//
// Comparison. The fork's export is not deterministic in typed domains: loki walks type hierarchies through an
// std::unordered_set of type pointers, so the order of the type predicates (and with it every predicate id), of the
// type atoms of one object and of the type literals of one parameter depends on heap addresses (different between
// runs with ASLR; see domain.hpp and .work/FORK_READY). `canonical_text` removes exactly that freedom and nothing
// else: predicates are referred to by name; every maximal run of consecutive type predicates, of consecutive type
// atoms of one object and of consecutive type literals over one term is sorted by name. Everything else, including
// the position of each run, must match token for token. Two more address-dependent orders of loki are factored out:
// the numeric effects of one conditional effect are a multiset, and in the axioms of the predicates that loki
// generates for universal quantifiers ("axiom_<k>") the type literals of all parameters form one run, by parameter.
// The argument order of those predicates is not factored out here: it changes parameter encodings, and with them the
// literal order elsewhere, so the tests replay the fork's order instead (DomainOptions::generated_argument_order).

#include "mymyr/formalism/task_data.hpp"

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mymyr::test
{
struct GoldenTask
{
    std::string name;  // file stem of the golden exports, e.g. "rovers__p02" or "m-barman"
    std::filesystem::path domain, problem;
    std::vector<std::filesystem::path> goldens;  // fork exports of this task that exist
};

/// Directory holding `mimir-cs`, `mimir` (env MYMYR_WORK, else the configured MYMYR_WORK_DIR).
std::filesystem::path work_dir();
/// The mimir fork's data/ directory with ipc/ (env MYMYR_FORK_DATA, else the configured MYMYR_FORK_DATA_DIR).
std::filesystem::path fork_data_dir();
/// tests/data (MYMYR_TEST_DATA_DIR).
std::filesystem::path test_data_dir();
/// The 22-task BrFS suite and its 20 numeric variants. Tasks whose PDDL is missing are left out (the caller reports
/// them).
std::vector<GoldenTask> suite_tasks(std::vector<std::string>* missing = nullptr);
std::vector<GoldenTask> numeric_tasks(std::vector<std::string>* missing = nullptr);
/// The IPC-2023 sample exported by tests/data/make_golden.sh (tests/data/golden/ipc), PDDL under fork_data_dir().
std::vector<GoldenTask> ipc_tasks(std::vector<std::string>* missing = nullptr);

std::vector<std::string> tokens(const std::string& text);

/// Type predicates of a task, by name: static unary predicates named after a type (loki's AddTypePredicates).
std::set<std::string> type_predicates(const formalism::TaskData& t);

/// The task's text (formalism::write_task_text) in the canonical form described above.
std::string canonical_text(const formalism::TaskData& t, const std::set<std::string>& type_preds);

/// A derived predicate that loki generated for a universal quantifier ("axiom_<k>") with two or more arguments: the
/// fork's argument order depends on heap addresses (domain.hpp; tests replay it with DomainOptions).
bool is_generated_axiom_predicate(const formalism::TaskData& t, u32 p);

/// First difference between two token streams, or nullopt if equal.
std::optional<std::string> first_difference(const std::vector<std::string>& a, const std::vector<std::string>& b);
}  // namespace mymyr::test
