#pragma once
// PDDL writer: the normalized task as a PDDL domain and problem that the front end reads back into an equal task.
//
//   const mymyr::formalism::PddlFiles f = mymyr::formalism::write_pddl(task->data());
//   // f.domain, f.problems[0]
//
// The files describe the normalized task, not the PDDL it came from: every precondition, effect condition and axiom
// body is a conjunction of literals and numeric constraints, the conditions that normalization compiled into derived
// predicates (disjunctions, quantifiers, implications) are those derived predicates and their `:derived` rules, a
// parameter that normalization added for an existential precondition is written as an `exists` around the
// precondition, and the schemas normalization split out of one action (a disjunctive precondition) are one action
// again whose precondition is the disjunction of theirs. Types, constants, objects, predicates, functions, the
// initial atoms and function values, the goal and the metric are written as declared. Requirements are those of the
// task plus the ones the written constructs need.
//
// Goal axioms: normalization turns a goal that is not a conjunction of literals into derived predicates of the
// problem (formalism::Axiom::from_problem). They are written into the problem's goal as the formula they stand for (a
// disjunction over the rules of a derived predicate, each an `exists` over its variables), so that the domain holds
// only what its problems share.
//
// Names: objects, predicates, functions, actions and types keep their names; variables are written with the names
// normalization gave them. Numbers are written in their shortest exact decimal form.

#include "mymyr/formalism/task_data.hpp"

#include <span>
#include <string>
#include <vector>

namespace mymyr::formalism
{
struct PddlFiles
{
    std::string domain;
    std::vector<std::string> problems;  // one per task
};

/// The task as one domain and one problem.
[[nodiscard]] PddlFiles write_pddl(const TaskData& task);

/// Tasks of one domain as one domain and one problem each. Throws
/// std::invalid_argument when the tasks do not share their domain (the domains written for them differ), or when
/// `tasks` is empty.
[[nodiscard]] PddlFiles write_pddl(std::span<const TaskData* const> tasks);
}  // namespace mymyr::formalism
