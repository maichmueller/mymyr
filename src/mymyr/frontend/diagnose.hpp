#pragma once
// The front end's errors (frontend/domain.hpp, PddlError): what went wrong while loki or mymyr read a PDDL text, as
// one line that names the file, the line, the action and the construct.

#include "mymyr/core/types.hpp"
#include "mymyr/frontend/domain.hpp"

#include <exception>
#include <string>
#include <string_view>

namespace mymyr::frontend::detail
{
/// The PddlError for an exception `e` thrown while reading `text` (the PDDL of the file `path`, as loki got it).
/// loki's messages are rewritten to name the construct, and a parse error is explained by the first construct of the
/// text that mymyr does not support, if there is one.
[[nodiscard]] PddlError diagnose(const std::exception& e, std::string_view text, const std::string& path);

/// Throws PddlError if `text` has a construct that loki parses but drops, so that mymyr would silently ignore it:
/// a `(:constraints ...)` section. Linear in the text (a substring search).
void reject_dropped_constructs(std::string_view text, const std::string& path);

/// 1-based line of byte offset `pos` of `text`.
[[nodiscard]] u32 line_of(std::string_view text, size_t pos);

/// The name of the action whose definition contains byte offset `pos` of `text`; empty if none does.
[[nodiscard]] std::string action_at(std::string_view text, size_t pos);
}  // namespace mymyr::frontend::detail
