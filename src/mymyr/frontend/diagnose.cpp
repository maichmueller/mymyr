#include "diagnose.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <optional>
#include <utility>
#include <vector>

namespace mymyr::frontend
{
namespace
{
std::string format_error(const std::string& message, const std::string& path, u32 line, const std::string& action)
{
    std::string out = path;
    if (line != 0)
        out += (out.empty() ? "line " : ":") + std::to_string(line);
    if (!out.empty())
        out += ": ";
    out += message;
    if (!action.empty())
        out += " (in action " + action + ")";
    return out;
}
}  // namespace

PddlError::PddlError(std::string message, std::string path, u32 line, std::string action)
    : std::invalid_argument(format_error(message, path, line, action)), m_message(std::move(message)),
      m_path(std::move(path)), m_line(line), m_action(std::move(action))
{
}
}  // namespace mymyr::frontend

namespace mymyr::frontend::detail
{
namespace
{
bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

/// Case-insensitive equality with a lower-case word (texts given as strings are not lower-cased).
bool is(std::string_view word, std::string_view lower)
{
    return word.size() == lower.size() &&
           std::equal(word.begin(), word.end(), lower.begin(),
                      [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == b; });
}

std::string lower(std::string_view s)
{
    std::string out(s);
    for (char& c : out)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool is_number(std::string_view s)
{
    if (!s.empty() && (s.front() == '-' || s.front() == '+'))
        s.remove_prefix(1);
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) || c == '.'; });
}

/// "(", ")" or a word, at byte offset `pos`. Comments (from ';' to the end of the line) are skipped.
struct Token
{
    std::string_view text;
    size_t pos;
};

std::vector<Token> tokenize(std::string_view t)
{
    std::vector<Token> out;
    size_t i = 0;
    while (i < t.size())
    {
        const char c = t[i];
        if (c == ';')
        {
            while (i < t.size() && t[i] != '\n')
                ++i;
            continue;
        }
        if (is_space(c))
        {
            ++i;
            continue;
        }
        if (c == '(' || c == ')')
        {
            out.push_back({t.substr(i, 1), i});
            ++i;
            continue;
        }
        const size_t b = i;
        while (i < t.size() && !is_space(t[i]) && t[i] != '(' && t[i] != ')' && t[i] != ';')
            ++i;
        out.push_back({t.substr(b, i - b), b});
    }
    return out;
}

bool is_word(const Token& t) { return t.text != "(" && t.text != ")"; }

struct Finding
{
    std::string message;
    size_t pos = 0;
};

/// Requirement flags mymyr does not support, with what they declare.
constexpr std::array<std::pair<std::string_view, std::string_view>, 10> k_unsupported_requirements{{
    {":durative-actions", "durative actions"},
    {":duration-inequalities", "duration inequalities"},
    {":continuous-effects", "continuous effects"},
    {":time", "processes and events"},
    {":timed-initial-literals", "timed initial literals"},
    {":preferences", "preferences"},
    {":constraints", "trajectory constraints"},
    {":object-fluents", "object fluents"},
    {":non-deterministic", "non-deterministic effects"},
    {":probabilistic-effects", "probabilistic effects"},
}};

/// Requirement flags mymyr reads.
constexpr std::array<std::string_view, 14> k_supported_requirements{
    ":strips", ":typing", ":negative-preconditions", ":disjunctive-preconditions", ":equality",
    ":existential-preconditions", ":universal-preconditions", ":quantified-preconditions", ":conditional-effects",
    ":fluents", ":numeric-fluents", ":adl", ":derived-predicates", ":action-costs",
};

/// The first construct of `text` mymyr does not support (by position), else the first requirement flag it does not
/// support or know.
std::optional<Finding> unsupported_construct(std::string_view text)
{
    const std::vector<Token> tk = tokenize(text);
    std::vector<std::string_view> heads;  // the head word of every open list ("" if it has none)
    std::optional<Finding> flag;
    std::string_view function;  // the function declared last in a :functions list
    const auto word_at = [&](size_t i) -> std::string_view { return i < tk.size() && is_word(tk[i]) ? tk[i].text : ""; };
    for (size_t i = 0; i < tk.size(); ++i)
    {
        const Token& t = tk[i];
        if (t.text == ")")
        {
            if (!heads.empty())
                heads.pop_back();
            continue;
        }
        const std::string_view parent = heads.empty() ? std::string_view{} : heads.back();
        if (t.text == "(")
        {
            const std::string_view head = word_at(i + 1);
            heads.push_back(head);
            const std::string_view name = word_at(i + 2);
            if (is(head, ":durative-action"))
                return Finding{"durative actions are not supported (:durative-action " + std::string(name) + ")", t.pos};
            if (is(head, ":process"))
                return Finding{"processes are not supported (:process " + std::string(name) + ")", t.pos};
            if (is(head, ":event"))
                return Finding{"events are not supported (:event " + std::string(name) + ")", t.pos};
            if (is(head, "preference"))
                return Finding{"preferences are not supported (preference " + std::string(name) + ")", t.pos};
            if (is(head, ":constraints"))
                return Finding{"trajectory constraints are not supported (:constraints)", t.pos};
            if (is(head, "oneof"))
                return Finding{"non-deterministic effects are not supported (oneof)", t.pos};
            if (is(head, "probabilistic"))
                return Finding{"probabilistic effects are not supported (probabilistic)", t.pos};
            if (is(parent, ":init") && is(head, "at") && is_number(name))
                return Finding{"timed initial literals are not supported (at " + std::string(name) + " ...)", t.pos};
            if (is(parent, ":functions"))
                function = head;
            continue;
        }
        if (i > 0 && tk[i - 1].text == "(")
            continue;  // a list's head word
        if (is(parent, ":requirements") && !flag)
        {
            const std::string f = lower(t.text);
            const auto u = std::ranges::find(k_unsupported_requirements, std::string_view(f),
                                             &std::pair<std::string_view, std::string_view>::first);
            if (u != k_unsupported_requirements.end())
                flag = Finding{"requirement " + f + " is not supported (" + std::string(u->second) + ")", t.pos};
            else if (std::ranges::find(k_supported_requirements, std::string_view(f)) == k_supported_requirements.end())
                flag = Finding{"unknown requirement " + f, t.pos};
        }
        if (is(parent, ":functions") && t.text == "-")
        {
            const std::string_view type = word_at(i + 1);
            if (!type.empty() && !is(type, "number"))
                return Finding{"object fluents are not supported (function '" + std::string(function) + "' has type '" +
                                   std::string(type) + "')",
                               t.pos};
        }
    }
    return flag;
}

/// The parts of a loki error message: "<head>\nIn file <path>, line <n>:\n[Error! Expecting: <x> here:]\n<source
/// line>\n<marker line>" (the marker '^' or '~...~' under the offending text); "In line <n>:" for text without a file.
struct LokiMessage
{
    std::string head;
    u32 line = 0;
    bool syntax = false;  // a parse error ("Error! Expecting: ...")
    std::string expecting;
    std::string marked;  // the source text under the marker
    u32 column = 0;
};

std::optional<LokiMessage> parse_loki(std::string_view msg)
{
    size_t in_file = msg.find("In file ");
    if (in_file == std::string_view::npos)
        in_file = msg.find("In line ");
    if (in_file == std::string_view::npos)
        return std::nullopt;
    LokiMessage m;
    std::string_view head = msg.substr(0, in_file);
    while (!head.empty() && is_space(head.back()))
        head.remove_suffix(1);
    while (!head.empty() && is_space(head.front()))
        head.remove_prefix(1);
    m.head = std::string(head);
    const size_t eol = msg.find('\n', in_file);
    const std::string_view loc = msg.substr(in_file, eol == std::string_view::npos ? std::string_view::npos : eol - in_file);
    const size_t at = loc.rfind("line ");
    if (at == std::string_view::npos)
        return std::nullopt;
    for (size_t i = at + 5; i < loc.size() && std::isdigit(static_cast<unsigned char>(loc[i])); ++i)
        m.line = m.line * 10 + static_cast<u32>(loc[i] - '0');
    std::vector<std::string_view> lines;
    for (size_t p = eol == std::string_view::npos ? msg.size() : eol + 1; p < msg.size();)
    {
        const size_t e = std::min(msg.find('\n', p), msg.size());
        lines.push_back(msg.substr(p, e - p));
        p = e + 1;
    }
    std::string_view source, marker;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const std::string_view l = lines[i];
        if (l.starts_with("Error! Expecting: "))
        {
            m.syntax = true;
            std::string_view x = l.substr(18);
            if (x.ends_with(" here:"))
                x.remove_suffix(6);
            m.expecting = std::string(x);
            continue;
        }
        if (std::all_of(l.begin(), l.end(), is_space))
            continue;
        source = l;
        if (i + 1 < lines.size())
            marker = lines[i + 1];
        break;
    }
    const size_t c = marker.find_first_of("^~");
    if (c != std::string_view::npos && c < source.size())
    {
        m.column = static_cast<u32>(c);
        size_t n = 1;
        while (c + n < marker.size() && marker[c + n] == '~')
            ++n;
        m.marked = std::string(source.substr(c, n));
    }
    return m;
}

/// Byte offset of (line, column) in `text` (clamped to the text).
size_t offset_of(std::string_view text, u32 line, u32 column)
{
    size_t p = 0;
    for (u32 l = 1; l < line && p < text.size(); ++l)
    {
        const size_t e = text.find('\n', p);
        if (e == std::string_view::npos)
            return text.size();
        p = e + 1;
    }
    return std::min(text.size(), p + column);
}

/// The word that starts at byte offset `pos` (a parenthesis is a word of its own; leading space is skipped).
std::string word_near(std::string_view text, size_t pos)
{
    while (pos < text.size() && is_space(text[pos]))
        ++pos;
    if (pos < text.size() && (text[pos] == '(' || text[pos] == ')'))
        return std::string(1, text[pos]);
    size_t e = pos;
    while (e < text.size() && !is_space(text[e]) && text[e] != '(' && text[e] != ')')
        ++e;
    return std::string(text.substr(pos, e - pos));
}

/// The text between the first pair of double quotes after `from` in `s`, and the position after it.
std::optional<std::pair<std::string, size_t>> quoted(std::string_view s, size_t from = 0)
{
    const size_t a = s.find('"', from);
    if (a == std::string_view::npos)
        return std::nullopt;
    const size_t b = s.find('"', a + 1);
    if (b == std::string_view::npos)
        return std::nullopt;
    return std::pair{std::string(s.substr(a + 1, b - a - 1)), b + 1};
}

/// loki's message for a semantic error, worded to name the construct.
std::string reword(const LokiMessage& m)
{
    std::string_view h = m.head;
    while (!h.empty() && (h.back() == '.' || is_space(h.back())))
        h.remove_suffix(1);
    // "The predicate with name "x" is undefined", "The function skeleton with name "x" is not defined in the ..."
    if (h.starts_with("The ") && h.find(" with name \"") != std::string_view::npos &&
        (h.ends_with(" is undefined") || h.find(" is not defined") != std::string_view::npos))
    {
        std::string kind(h.substr(4, h.find(" with name \"") - 4));
        if (kind.ends_with(" skeleton"))
            kind.resize(kind.size() - 9);
        if (auto q = quoted(h))
            return "undefined " + kind + " '" + q->first + "'";
    }
    // "Mismatched arity 1!=2" under "(clear ?x ?y)": the declared arity, then the number of arguments given
    if (h.starts_with("Mismatched arity "))
    {
        const std::string_view n = h.substr(17);
        const size_t ne = n.find("!=");
        std::string_view marked = m.marked;
        if (marked.starts_with('('))
            marked.remove_prefix(1);
        const std::string what = word_near(marked, 0);
        const std::string name = what.empty() ? std::string("a predicate") : "'" + what + "'";
        if (ne != std::string_view::npos)
            return "wrong number of arguments for " + name + " (" + std::string(n.substr(ne + 2)) + " given, " +
                   std::string(n.substr(0, ne)) + " expected)";
    }
    // Mismatched domain names "d != other
    if (h.starts_with("Mismatched domain names"))
    {
        std::string_view names = h.substr(23);
        while (!names.empty() && (is_space(names.front()) || names.front() == '"'))
            names.remove_prefix(1);
        while (!names.empty() && names.back() == '"')
            names.remove_suffix(1);
        const size_t ne = names.find(" != ");
        if (ne != std::string_view::npos)
            return "the problem is for domain '" + std::string(names.substr(ne + 4)) + "', not for domain '" +
                   std::string(names.substr(0, ne)) + "'";
    }
    if (h.starts_with("Undefined requirement: "))
        return "requirement " + std::string(h.substr(23)) + " is used but not declared in :requirements";
    if (h.starts_with("Unsupported requirement: "))
        return "requirement " + std::string(h.substr(25)) + " is not supported";
    if (!h.empty())
        return std::string(h);
    return "invalid PDDL";
}

std::string_view strip_prefix(std::string_view s, std::string_view prefix)
{
    if (s.starts_with(prefix))
        s.remove_prefix(prefix.size());
    return s;
}
}  // namespace

void reject_dropped_constructs(std::string_view text, const std::string& path)
{
    static constexpr std::string_view k = ":constraints";
    for (size_t p = text.find(k); p != std::string_view::npos; p = text.find(k, p + 1))
    {
        size_t open = p;
        while (open > 0 && is_space(text[open - 1]))
            --open;
        const size_t end = p + k.size();
        if (open > 0 && text[open - 1] == '(' && (end == text.size() || is_space(text[end]) || text[end] == '('))
            throw PddlError("trajectory constraints are not supported (:constraints)", path, line_of(text, open - 1));
    }
}

u32 line_of(std::string_view text, size_t pos)
{
    pos = std::min(pos, text.size());
    return 1 + static_cast<u32>(std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(pos), '\n'));
}

std::string action_at(std::string_view text, size_t pos)
{
    const std::vector<Token> tk = tokenize(text);
    u32 depth = 0, action_depth = 0;
    std::string_view name;
    for (size_t i = 0; i < tk.size() && tk[i].pos <= pos; ++i)
    {
        if (tk[i].text == "(")
        {
            ++depth;
            if (action_depth == 0 && i + 2 < tk.size() && is(tk[i + 1].text, ":action") && is_word(tk[i + 2]))
            {
                action_depth = depth;
                name = tk[i + 2].text;
            }
        }
        else if (tk[i].text == ")")
        {
            if (depth == action_depth)
            {
                action_depth = 0;
                name = {};
            }
            if (depth > 0)
                --depth;
        }
    }
    return std::string(name);
}

PddlError diagnose(const std::exception& e, std::string_view text, const std::string& path)
{
    if (const auto* p = dynamic_cast<const PddlError*>(&e))
        return *p;
    const std::string_view what = e.what();
    if (const std::optional<LokiMessage> m = parse_loki(what); m && m->line != 0)
    {
        const size_t pos = offset_of(text, m->line, m->column);
        if (m->syntax || m->head.starts_with("Unsupported requirement") || m->head.starts_with("Undefined requirement"))
            if (std::optional<Finding> f = unsupported_construct(text))
                return PddlError(std::move(f->message), path, line_of(text, f->pos), action_at(text, f->pos));
        if (m->syntax)
        {
            std::string msg = "syntax error";
            // loki names what it expected by a quoted literal, or by a grammar rule (not shown)
            if (m->expecting.size() >= 3 && m->expecting.front() == '\'' && m->expecting.back() == '\'')
                msg += ", expected " + m->expecting;
            if (const std::string w = word_near(text, pos); !w.empty())
                msg += " at '" + w + "'";
            else if (pos >= text.size())
                msg += " at the end of the file";
            return PddlError(std::move(msg), path, m->line, action_at(text, pos));
        }
        return PddlError(reword(*m), path, m->line, action_at(text, pos));
    }
    // an error of the translation into mymyr's formalism: an unsupported construct loki accepted explains it best
    if (std::optional<Finding> f = unsupported_construct(text))
        return PddlError(std::move(f->message), path, line_of(text, f->pos), action_at(text, f->pos));
    std::string msg(strip_prefix(what, "mymyr frontend: "));
    u32 line = 0;
    if (msg.starts_with("negative literals in the initial state"))
    {
        // the first negative literal of :init
        const std::vector<Token> tk = tokenize(text);
        bool in_init = false;
        for (size_t i = 0; i + 1 < tk.size() && line == 0; ++i)
            if (tk[i].text == "(" && is(tk[i + 1].text, ":init"))
                in_init = true;
            else if (in_init && tk[i].text == "(" && is(tk[i + 1].text, "not"))
                line = line_of(text, tk[i].pos);
    }
    return PddlError(std::move(msg), path, line);
}
}  // namespace mymyr::frontend::detail
