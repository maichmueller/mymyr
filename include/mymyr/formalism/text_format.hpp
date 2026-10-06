#pragma once
// Reader and writer for mymyr's normalized task text format: a plain-text serialization of a TaskData (O P SI FI G
// A, with numeric sections N NI NG added for tasks with functions). It carries no names beyond predicates, functions
// and schemas; objects become "o<i>" and parameters "x<i>".
//
// mimir can export the same tasks in this format, so it doubles as a golden reference: a task built by mymyr's own
// front end is written in this format and compared against mimir's export (tests/data). write_task_text() writes
// it; read_task_text() and read_task_text_file() read it back.

#include "mymyr/formalism/task_data.hpp"

#include <iosfwd>
#include <string>

namespace mymyr::formalism
{
/// Reads either the plain lifted format (O P SI FI G A [X]) or the numeric one (O P N SI FI NI G NG M A X, with NC /
/// NE / NA blocks per condition and effect). Throws std::runtime_error on malformed input.
TaskData read_task_text(std::istream& in);
TaskData read_task_text_file(const std::string& path);

/// Writes the task in this text format: the numeric variant iff the task has functions or a metric.
std::string write_task_text(const TaskData& t);
}  // namespace mymyr::formalism
