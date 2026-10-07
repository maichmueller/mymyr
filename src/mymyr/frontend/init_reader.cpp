#include "init_reader.hpp"

#include <absl/container/flat_hash_set.h>
#include <absl/strings/charconv.h>
#include <boost/hana.hpp>
#include <loki/details/pddl/repositories.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace mymyr::frontend::detail
{
namespace
{
bool name_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'; }
constexpr bool is_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

bool keyword_at(std::string_view text, size_t pos, std::string_view kw)
{
    if (pos + kw.size() > text.size())
        return false;
    for (size_t i = 0; i < kw.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(text[pos + i])) != kw[i])
            return false;
    return pos + kw.size() == text.size() || !name_char(text[pos + kw.size()]);
}

/// Tokens of the `:init` section: "(", ")" or a word (name, number, keyword).
class Lexer
{
public:
    Lexer(std::string_view text, std::string_view path) : m_text(text), m_path(path) {}

    enum class Kind : u8
    {
        Open,
        Close,
        Word,
        End,
    };
    struct Token
    {
        Kind kind;
        std::string_view text;
    };

    Token next()
    {
        skip();
        if (m_pos >= m_text.size())
            return {Kind::End, {}};
        const char c = m_text[m_pos];
        if (c == '(')
            return {Kind::Open, m_text.substr(m_pos++, 1)};
        if (c == ')')
            return {Kind::Close, m_text.substr(m_pos++, 1)};
        const size_t b = m_pos;
        while (m_pos < m_text.size() && !is_space(m_text[m_pos]) && m_text[m_pos] != '('
               && m_text[m_pos] != ')' && m_text[m_pos] != ';')
            ++m_pos;
        return {Kind::Word, m_text.substr(b, m_pos - b)};
    }
    Token peek()
    {
        const size_t save = m_pos;
        Token t = next();
        m_pos = save;
        return t;
    }
    [[noreturn]] void fail(const std::string& msg) const
    {
        // line number of the current position, for the error message
        size_t line = 1;
        for (size_t i = 0; i < std::min(m_pos, m_text.size()); ++i)
            line += m_text[i] == '\n';
        throw std::runtime_error("mymyr frontend: " + std::string(m_path) + ": :init section, line " + std::to_string(line)
                                 + " of the section: " + msg);
    }
    Token expect(Kind k, const char* what)
    {
        Token t = next();
        if (t.kind != k)
            fail(std::string("expected ") + what + ", got '" + std::string(t.text) + "'");
        return t;
    }

private:
    void skip()
    {
        while (m_pos < m_text.size())
        {
            const char c = m_text[m_pos];
            if (c == ';')
                while (m_pos < m_text.size() && m_text[m_pos] != '\n')
                    ++m_pos;
            else if (is_space(c))
                ++m_pos;
            else
                break;
        }
    }

    std::string_view m_text;
    std::string_view m_path;
    size_t m_pos = 0;
};

std::optional<f64> parse_number(std::string_view s)
{
    // absl::from_chars also reads a "0x" hexadecimal float under chars_format::general; a number is decimal here
    if (s.find_first_of("xX") != std::string_view::npos)
        return std::nullopt;
    f64 v = 0;
    auto [p, ec] = absl::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size())
        return std::nullopt;
    return v;
}

u32 predicate_id(const DomainState& ds, loki::Predicate p)
{
    if (auto it = ds.pred_of.find(p); it != ds.pred_of.end())
        return it->second;
    if (auto it = ds.pred_by_name.find(p->get_name()); it != ds.pred_by_name.end())
        return it->second;
    return ~0u;
}

/// Key of a ground loki literal in TaskData ids, or nullopt if it is not ground over known entities.
std::optional<Key> literal_key(const DomainState& ds, loki::Literal l, const ObjectMap* objects)
{
    const u32 p = predicate_id(ds, l->get_atom()->get_predicate());
    if (p == ~0u)
        return std::nullopt;
    Key key{p, l->get_polarity() ? 1u : 0u};
    for (const auto& t : l->get_atom()->get_terms())
    {
        const auto* o = std::get_if<loki::Object>(&t->get_object_or_variable());
        if (!o)
            return std::nullopt;
        if (objects)
            if (auto it = objects->find(*o); it != objects->end())
            {
                key.push_back(it->second);
                continue;
            }
        if (auto it = ds.const_of.find(*o); it != ds.const_of.end())
            key.push_back(it->second);
        else if (auto jt = ds.const_by_name.find((*o)->get_name()); jt != ds.const_by_name.end())
            key.push_back(jt->second);
        else
            return std::nullopt;
    }
    return key;
}

void index_literals(const DomainState& ds, const loki::Domain& d, absl::flat_hash_map<Key, u32>& out, u32* size)
{
    const auto& repo = boost::hana::at_key(d->get_repositories().get_hana_repositories(), boost::hana::type<loki::LiteralImpl>{});
    for (size_t i = 0; i < repo.size(); ++i)
    {
        const loki::Literal l = repo[i];
        if (auto key = literal_key(ds, l, nullptr))
            out.try_emplace(std::move(*key), static_cast<u32>(l->get_index()));
    }
    if (size)
        *size = static_cast<u32>(repo.size());
}

/// Type closure per object: the types an object has, including all supertypes (loki's collect_types_from_hierarchy).
class TypeCheck
{
public:
    explicit TypeCheck(const formalism::TaskData& d) : d(d), m_words((d.types.size() + 63) / 64)
    {
        m_closure.resize(d.types.size() * m_words, 0);
        for (u32 t = 0; t < d.types.size(); ++t)
            close(t, t);
    }
    /// Is object `o` (with declared types `otypes`) compatible with a parameter of types `ptypes`?
    [[nodiscard]] bool compatible(std::span<const TypeId> otypes, std::span<const TypeId> ptypes) const
    {
        if (ptypes.empty())
            return true;
        for (TypeId ot : otypes)
            for (TypeId pt : ptypes)
                if (m_closure[ot.v * m_words + pt.v / 64] >> (pt.v % 64) & 1)
                    return true;
        return false;
    }

private:
    void close(u32 root, u32 t)
    {
        u64& w = m_closure[root * m_words + t / 64];
        if (w >> (t % 64) & 1)
            return;
        w |= u64{1} << (t % 64);
        for (TypeId b : formalism::TaskData::slice(d.type_ids, d.types[t].bases))
            close(root, b.v);
    }
    const formalism::TaskData& d;
    size_t m_words;
    std::vector<u64> m_closure;
};
}  // namespace

// ------------------------------------------------------------------------------------------------ text

std::string read_pddl_file(const std::filesystem::path& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("mymyr frontend: cannot open " + path.string());
    std::string raw;
    char buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        raw.append(buf, n);
    std::fclose(f);
    return preprocess_pddl(raw);
}

std::string preprocess_pddl(std::string_view raw)
{
    std::string out;
    out.reserve(raw.size() + raw.size() / 64 + 1);
    size_t i = 0;
    while (i < raw.size())
    {
        const size_t nl = raw.find('\n', i);
        const size_t end = nl == std::string::npos ? raw.size() : nl;
        const void* semi = std::memchr(raw.data() + i, ';', end - i);
        const size_t stop = semi ? static_cast<size_t>(static_cast<const char*>(semi) - raw.data()) : end;
        const size_t from = out.size();
        if (std::memchr(raw.data() + i, '\t', stop - i))
        {
            for (size_t k = i; k < stop; ++k)
                if (raw[k] == '\t')
                    out.append(4, ' ');
                else
                    out.push_back(raw[k]);
        }
        else
            out.append(raw.substr(i, stop - i));
        for (size_t k = from; k < out.size(); ++k)
            out[k] = static_cast<char>(out[k] + ((static_cast<unsigned char>(out[k] - 'A') < 26u) ? 'a' - 'A' : 0));
        out.push_back('\n');
        if (nl == std::string::npos)
            break;
        i = nl + 1;
    }
    return out;
}

TextRange find_init_section(std::string_view text)
{
    // the problem's elements are at depth 2 (inside "(define"); comments run to the end of the line
    const char* const b = text.data();
    const char* const e = b + text.size();
    auto skip_comment = [&](const char* p) { const void* nl = std::memchr(p, '\n', static_cast<size_t>(e - p)); return nl ? static_cast<const char*>(nl) : e; };
    int depth = 0;
    for (const char* p = b; p < e; ++p)
    {
        const char c = *p;
        if (c == ';')
            p = skip_comment(p);
        else if (c == ')')
            --depth;
        else if (c == '(' && ++depth == 2)
        {
            const char* q = p + 1;
            while (q < e && is_space(*q))
                ++q;
            if (!keyword_at(text, static_cast<size_t>(q - b), ":init"))
                continue;
            // Shortcut: nothing inside :init contains ':' (names start with a letter), so the next ':' starts the
            // next element, "( :goal" say, and the section closes at the last ')' before that '('. Only with no ';'
            // (a comment could hold anything); the result is checked, and otherwise the parentheses are matched.
            if (!std::memchr(q, ';', static_cast<size_t>(e - q)))
            {
                const char* k = q + 5;
                const void* colon = std::memchr(k, ':', static_cast<size_t>(e - k));
                const char* r = colon ? static_cast<const char*>(colon) : e;
                // back over the next element's "(" (or the define's closing ")" at the end)
                const char* c = r - 1;
                while (c > q && is_space(*c))
                    --c;
                if (colon)
                {
                    if (*c == '(')
                        --c;
                    else
                        c = q;  // not an element start: fall back
                }
                else
                {
                    if (*c == ')')
                        --c;  // the define's
                    else
                        c = q;
                }
                while (c > q && is_space(*c))
                    --c;
                if (c > q && *c == ')')
                    return {static_cast<size_t>(p - b), static_cast<size_t>(c + 1 - b)};
            }
            // matching parenthesis
            int d = 1;
            const char* k = q;
            for (; k < e && d > 0; ++k)
            {
                const char ch = *k;
                if (ch == '(')
                    ++d;
                else if (ch == ')')
                    --d;
                else if (ch == ';')
                    k = skip_comment(k);
            }
            if (d != 0)
                return {};  // unbalanced: let loki report it
            return {static_cast<size_t>(p - b), static_cast<size_t>(k - b)};
        }
    }
    return {};
}

std::string without_section(std::string_view text, TextRange r)
{
    std::string out;
    out.reserve(text.size() - (r.end - r.begin) + 64);
    out.append(text.substr(0, r.begin));
    out.append(static_cast<size_t>(std::count(text.begin() + static_cast<std::ptrdiff_t>(r.begin),
                                              text.begin() + static_cast<std::ptrdiff_t>(r.end), '\n')),
               '\n');
    out.push_back(' ');
    out.append(text.substr(r.end));
    return out;
}

// ------------------------------------------------------------------------------------------------ tables

InitOrderTables make_init_order_tables(const DomainState& ds, const loki::Domain& parsed, const loki::Domain& translated)
{
    InitOrderTables t;
    index_literals(ds, translated, t.translated, &t.translated_size);
    index_literals(ds, parsed, t.parsed, nullptr);
    return t;
}

// ------------------------------------------------------------------------------------------------ reader

GroundInit read_fast_init(const DomainState& ds, const InitOrderTables& tables, const loki::Problem& problem,
                          const ObjectMap& object_of, std::string_view section, std::string_view path)
{
    const formalism::TaskData& d = ds.data;
    const u32 num_constants = static_cast<u32>(d.objects.size());

    // ---- names of the problem
    absl::flat_hash_map<std::string_view, u32> object_by_name;
    std::vector<std::string_view> object_name;       // by object id
    std::vector<std::vector<TypeId>> object_types;   // declared types, by object id
    object_by_name.reserve(object_of.size());
    for (const auto& [o, id] : object_of)
    {
        object_by_name.try_emplace(o->get_name(), id);
        if (id >= object_name.size())
        {
            object_name.resize(id + 1);
            object_types.resize(id + 1);
        }
        object_name[id] = o->get_name();
        auto& ts = object_types[id];
        ts.clear();
        for (const auto& b : o->get_bases())
            if (auto it = ds.type_by_name.find(b->get_name()); it != ds.type_by_name.end())
                ts.push_back(TypeId{it->second});
    }
    absl::flat_hash_map<std::string_view, u32> pred_by_name, func_by_name;
    for (const auto& [name, id] : ds.pred_by_name)
        pred_by_name.emplace(name, id);
    for (const auto& [name, id] : ds.func_by_name)
        func_by_name.emplace(name, id);
    // problem predicates (derived, from `:derived-predicates` of the problem): ids continue the domain's
    std::vector<u32> problem_pred_arity;
    {
        u32 next = static_cast<u32>(d.predicates.size());
        for (const auto& p : problem->get_predicates())
            if (!ds.pred_by_name.contains(p->get_name()))
            {
                pred_by_name.emplace(p->get_name(), next++);
                problem_pred_arity.push_back(static_cast<u32>(p->get_parameters().size()));
            }
    }
    auto pred_arity = [&](u32 p) { return p < d.predicates.size() ? d.predicates[p].arity : problem_pred_arity.at(p - d.predicates.size()); };

    // ---- groups 1, 2 and 4 from loki (the problem without `:init`)
    struct Ranked
    {
        u32 rank;
        Key key;
    };
    std::vector<Ranked> members;               // group 1: (translated-domain index, key)
    std::vector<Key> type_literals, equality;  // groups 2 and 4, in loki's order
    auto unary = [](u64 pred, u64 obj) { return pred << 32 | obj; };
    absl::flat_hash_set<u64> generated_type;  // group 2 (unary, positive): (pred, object)
    for (const auto& l : problem->get_initial_literals())
    {
        auto key = literal_key(ds, l, &object_of);
        if (!key)
            throw std::logic_error("mymyr frontend: unexpected loki initial literal");
        if (l->get_index() < tables.translated_size)
            members.push_back({static_cast<u32>(l->get_index()), std::move(*key)});
        else if (l->get_atom()->get_predicate()->get_name() == "=")
            equality.push_back(std::move(*key));
        else
        {
            if (key->size() != 3 || (*key)[1] != 1)
                throw std::logic_error("mymyr frontend: unexpected loki initial literal (not a type literal)");
            generated_type.insert(unary((*key)[0], (*key)[2]));
            type_literals.push_back(std::move(*key));
        }
    }
    const u32 eq_pred = ds.pred_by_name.contains("=") ? ds.pred_by_name.at("=") : ~0u;

    // ---- group 3 from the text (duplicates are removed later, by translate_problem, keeping the first occurrence)
    TypeCheck types(d);
    std::vector<Ranked> parsed_members;  // (parsed-domain index, key)
    GroundInit text;                     // the others, text order
    absl::flat_hash_set<Key> text_equality;  // equality literals written in the text (excluded from group 4)
    GroundInit init;
    std::vector<u32> objs;

    Lexer lx(section, path);
    lx.expect(Lexer::Kind::Open, "'('");
    if (auto kw = lx.expect(Lexer::Kind::Word, "':init'"); !keyword_at(kw.text, 0, ":init"))
        lx.fail("expected ':init'");

    auto object_id = [&](std::string_view name) -> u32
    {
        auto it = object_by_name.find(name);
        if (it == object_by_name.end())
            lx.fail("undefined object '" + std::string(name) + "'");
        return it->second;
    };
    auto check_types = [&](u32 pred, std::span<const u32> args)
    {
        if (pred >= d.predicates.size())
            return;  // problem predicates are derived; loki rejects them in :init anyway (translate_problem does too)
        const formalism::Predicate& p = d.predicates[pred];
        for (u32 i = 0; i < args.size(); ++i)
        {
            const auto ptypes = formalism::TaskData::slice(d.type_ids, d.params[p.params.begin + i].types);
            if (!types.compatible(object_types[args[i]], ptypes))
                lx.fail("object '" + std::string(object_name[args[i]]) + "' has a type incompatible with parameter "
                        + std::to_string(i) + " of '" + std::string(d.str(p.name)) + "'");
        }
    };
    auto add_literal = [&](u32 pred, bool positive, std::span<const u32> args)
    {
        if (args.size() != pred_arity(pred))
            lx.fail("wrong number of arguments for '" + std::string(pred < d.predicates.size() ? d.str(d.predicates[pred].name) : "?")
                    + "'");
        check_types(pred, args);
        const bool over_constants = std::all_of(args.begin(), args.end(), [&](u32 o) { return o < num_constants; });
        if (over_constants)  // only ground literals of the domain can be members of its repositories
        {
            Key key{pred, positive ? 1u : 0u};
            key.insert(key.end(), args.begin(), args.end());
            if (auto it = tables.translated.find(key); it != tables.translated.end())
            {
                members.push_back({it->second, std::move(key)});
                return;
            }
            if (auto it = tables.parsed.find(key); it != tables.parsed.end() && !(args.size() == 1 && positive && generated_type.contains(unary(pred, args[0]))))
            {
                parsed_members.push_back({it->second, std::move(key)});
                return;
            }
        }
        if (args.size() == 1 && positive && generated_type.contains(unary(pred, args[0])))
            return;  // AddTypePredicates created it before the text literals
        if (pred == eq_pred)
        {
            Key key{pred, positive ? 1u : 0u};
            key.insert(key.end(), args.begin(), args.end());
            text_equality.insert(std::move(key));
        }
        text.add_atom(pred, positive, args);
    };
    // (pred args...) after the opening parenthesis; returns the predicate id
    auto atom = [&](std::string_view head, bool positive)
    {
        objs.clear();
        if (head == "=")
        {
            objs.push_back(object_id(lx.expect(Lexer::Kind::Word, "an object name").text));
            objs.push_back(object_id(lx.expect(Lexer::Kind::Word, "an object name").text));
            lx.expect(Lexer::Kind::Close, "')'");
        }
        else
            for (;;)
            {
                const auto t = lx.next();
                if (t.kind == Lexer::Kind::Close)
                    break;
                if (t.kind != Lexer::Kind::Word)
                    lx.fail("expected an object name or ')'");
                objs.push_back(object_id(t.text));
            }
        auto it = pred_by_name.find(head);
        if (it == pred_by_name.end())
            lx.fail("undefined predicate '" + std::string(head) + "'");
        add_literal(it->second, positive, objs);
    };

    for (;;)
    {
        auto t = lx.next();
        if (t.kind == Lexer::Kind::Close)
            break;
        if (t.kind != Lexer::Kind::Open)
            lx.fail("expected '(' or ')', got '" + std::string(t.text) + "'");
        const auto head = lx.expect(Lexer::Kind::Word, "a predicate name").text;
        if (head == "not")
        {
            lx.expect(Lexer::Kind::Open, "'('");
            atom(lx.expect(Lexer::Kind::Word, "a predicate name").text, false);
            lx.expect(Lexer::Kind::Close, "')'");
        }
        else if (head == "at" && lx.peek().kind == Lexer::Kind::Word && parse_number(lx.peek().text))
            lx.fail("timed initial literals are not supported");
        else if (head == "=")
        {
            // function value `(= (f o...) v)` / `(= f v)`, or an equality atom `(= a b)`
            auto n = lx.next();
            std::string_view fname;
            objs.clear();
            if (n.kind == Lexer::Kind::Open)
            {
                fname = lx.expect(Lexer::Kind::Word, "a function name").text;
                for (;;)
                {
                    const auto a = lx.next();
                    if (a.kind == Lexer::Kind::Close)
                        break;
                    if (a.kind != Lexer::Kind::Word)
                        lx.fail("expected an object name or ')'");
                    objs.push_back(object_id(a.text));
                }
            }
            else if (n.kind == Lexer::Kind::Word)
            {
                const auto v = lx.peek();
                if (v.kind != Lexer::Kind::Word || !parse_number(v.text))
                {
                    // equality atom: re-read "= a b)"
                    objs.clear();
                    objs.push_back(object_id(n.text));
                    objs.push_back(object_id(lx.expect(Lexer::Kind::Word, "an object name").text));
                    lx.expect(Lexer::Kind::Close, "')'");
                    auto it = pred_by_name.find("=");
                    if (it == pred_by_name.end())
                        lx.fail("equality atom without the :equality requirement");
                    add_literal(it->second, true, objs);
                    continue;
                }
                fname = n.text;
            }
            else
                lx.fail("malformed '(=' element");
            const auto num = lx.expect(Lexer::Kind::Word, "a number");
            const auto value = parse_number(num.text);
            if (!value)
                lx.fail("expected a number, got '" + std::string(num.text) + "'");
            lx.expect(Lexer::Kind::Close, "')'");
            auto it = func_by_name.find(fname);
            if (it == func_by_name.end())
                lx.fail("undefined function '" + std::string(fname) + "'");
            if (objs.size() != d.functions[it->second].arity)
                lx.fail("wrong number of arguments for function '" + std::string(fname) + "'");
            init.add_value(it->second, objs, *value);
        }
        else
            atom(head, true);
    }

    // ---- merge: group 1 by translated index, group 2, group 3 (parsed-domain members first), group 4
    std::sort(members.begin(), members.end(), [](const Ranked& a, const Ranked& b) { return a.rank < b.rank; });
    std::sort(parsed_members.begin(), parsed_members.end(), [](const Ranked& a, const Ranked& b) { return a.rank < b.rank; });
    auto emit = [&](const Key& k)
    {
        objs.assign(k.begin() + 2, k.end());
        init.add_atom(static_cast<u32>(k[0]), k[1] != 0, objs);
    };
    for (size_t i = 0; i < members.size(); ++i)
        if (i == 0 || members[i].rank != members[i - 1].rank)
            emit(members[i].key);
    for (const auto& k : type_literals)
        emit(k);
    for (const auto& r : parsed_members)
        emit(r.key);
    init.atoms.reserve(init.atoms.size() + text.atoms.size() + equality.size());
    init.objects.reserve(init.objects.size() + text.objects.size() + 2 * equality.size());
    for (const auto& a : text.atoms)
        init.add_atom(a.pred, a.positive, std::span<const u32>(text.objects.data() + a.begin, a.count));
    for (const auto& k : equality)
        if (!text_equality.contains(k))
            emit(k);
    return init;
}
}  // namespace mymyr::frontend::detail
