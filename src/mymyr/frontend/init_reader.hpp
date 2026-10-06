#pragma once
// The problem fast path: the `:init` section is read in one pass straight into a `GroundInit`, instead of going
// through loki (whose ten problem translation passes rebuild every initial literal).
//
// What must be reproduced is loki's *order* of the translated problem's initial literals, because mimir orders the
// initial atoms by creation index, i.e. by loki's order. That order follows from loki's repositories (every pass sorts
// the literals by index; indices continue the parent repositories' index space):
//
//   1. literals that exist in the translated domain's literal repository (ground literals of the domain: the static
//      type literals of constants, nullary literals, literals over constants), by their index there;
//   2. the type literals of the problem objects (AddTypePredicates, the last pass, creates them first), in loki's
//      order;
//   3. the literals written in `:init`: first those that exist in the *parsed* domain's repository (the parser's
//      parent), by their index there, then the others in text order;
//   4. the equality literals of InitializeEquality (constants, then objects) that `:init` does not already contain.
//
// Groups 1, 2 and 4 are exactly the initial literals of the problem that loki translates *without* its `:init`
// section, so the fast path takes them from there and only merges group 3 in. Function values keep their text order.
// Duplicates are left to translate_problem (first occurrence wins), which also puts the type atoms of each object in
// the canonical hierarchy order (domain.hpp), for this path and the full loki path alike.

#include "translate.hpp"

#include <absl/container/flat_hash_map.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace mymyr::frontend::detail
{
/// Preprocesses PDDL text the way loki::read_file does (per line: the comment from ';' on is dropped, a tab becomes
/// four spaces, ASCII letters are lower-cased; every line ends with '\n'), in one pass. Idempotent.
std::string preprocess_pddl(std::string_view raw);
/// Reads a PDDL file and preprocesses it (preprocess_pddl).
std::string read_pddl_file(const std::filesystem::path& path);

/// [begin, end) of the `(:init ...)` element of a problem text; `begin == end` if there is none.
struct TextRange
{
    size_t begin = 0;
    size_t end = 0;
    [[nodiscard]] bool empty() const { return begin == end; }
};
TextRange find_init_section(std::string_view problem_text);

/// The problem text without its `:init` element (newlines are kept, so loki's error positions stay valid).
std::string without_section(std::string_view text, TextRange r);

/// Per-Domain lookup tables of the fast path. Keys are [pred, polarity, objects...] in TaskData ids.
struct InitOrderTables
{
    absl::flat_hash_map<Key, u32> translated;  // ground literals of the translated domain -> loki index
    absl::flat_hash_map<Key, u32> parsed;      // ground literals of the parsed domain -> loki index
    u32 translated_size = 0;                   // index space of the translated domain's literal repository
};
InitOrderTables make_init_order_tables(const DomainState& ds, const loki::Domain& parsed, const loki::Domain& translated);

/// The initial literals and function values of the problem, in loki's order (see above). `problem` is the translated
/// problem without its `:init`; `section` is the text of the `(:init ...)` element.
GroundInit read_fast_init(const DomainState& ds, const InitOrderTables& tables, const loki::Problem& problem,
                          const ObjectMap& object_of, std::string_view section, std::string_view path);
}  // namespace mymyr::frontend::detail
