#pragma once
// A minimal JSON reader for test data (tests/data/expected/*.json): null, booleans, numbers, strings with the usual
// escapes (\uXXXX as UTF-8, surrogate pairs included), arrays and objects. Throws std::runtime_error on malformed input.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr::test::json
{
struct Value
{
    enum class Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };
    Type type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value, std::less<>> obj;

    [[nodiscard]] bool is_null() const { return type == Type::Null; }
    [[nodiscard]] bool has(std::string_view k) const { return type == Type::Object && obj.find(k) != obj.end(); }
    [[nodiscard]] const Value& operator[](std::string_view k) const
    {
        static const Value null;
        if (type != Type::Object)
            return null;
        const auto it = obj.find(k);
        return it == obj.end() ? null : it->second;
    }
    [[nodiscard]] const Value& operator[](std::size_t i) const { return arr.at(i); }
    [[nodiscard]] std::size_t size() const { return type == Type::Array ? arr.size() : obj.size(); }
};

class Parser
{
public:
    explicit Parser(std::string_view text) : m_s(text) {}

    Value parse()
    {
        Value v = value();
        ws();
        if (m_i != m_s.size())
            fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const
    {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(m_i));
    }
    void ws()
    {
        while (m_i < m_s.size() && (m_s[m_i] == ' ' || m_s[m_i] == '\n' || m_s[m_i] == '\r' || m_s[m_i] == '\t'))
            ++m_i;
    }
    char peek()
    {
        ws();
        if (m_i >= m_s.size())
            fail("unexpected end");
        return m_s[m_i];
    }
    void expect(char c)
    {
        if (peek() != c)
            fail("unexpected character");
        ++m_i;
    }
    bool literal(std::string_view w)
    {
        if (m_s.substr(m_i, w.size()) == w)
        {
            m_i += w.size();
            return true;
        }
        return false;
    }
    static void utf8(std::string& out, std::uint32_t cp)
    {
        if (cp < 0x80)
            out += static_cast<char>(cp);
        else if (cp < 0x800)
        {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000)
        {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else
        {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    std::uint32_t hex4()
    {
        if (m_i + 4 > m_s.size())
            fail("short \\u escape");
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k)
        {
            const char c = m_s[m_i++];
            v <<= 4;
            if (c >= '0' && c <= '9')
                v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else
                fail("bad \\u escape");
        }
        return v;
    }
    std::string string()
    {
        expect('"');
        std::string out;
        while (true)
        {
            if (m_i >= m_s.size())
                fail("unterminated string");
            const char c = m_s[m_i++];
            if (c == '"')
                return out;
            if (c != '\\')
            {
                out += c;
                continue;
            }
            if (m_i >= m_s.size())
                fail("bad escape");
            const char e = m_s[m_i++];
            switch (e)
            {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                {
                    std::uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && literal("\\u"))
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (hex4() - 0xDC00);
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Value value()
    {
        Value v;
        const char c = peek();
        if (c == '{')
        {
            v.type = Value::Type::Object;
            ++m_i;
            if (peek() == '}')
            {
                ++m_i;
                return v;
            }
            while (true)
            {
                std::string k = string();
                expect(':');
                v.obj.insert_or_assign(std::move(k), value());
                if (peek() == ',')
                {
                    ++m_i;
                    continue;
                }
                expect('}');
                return v;
            }
        }
        if (c == '[')
        {
            v.type = Value::Type::Array;
            ++m_i;
            if (peek() == ']')
            {
                ++m_i;
                return v;
            }
            while (true)
            {
                v.arr.push_back(value());
                if (peek() == ',')
                {
                    ++m_i;
                    continue;
                }
                expect(']');
                return v;
            }
        }
        if (c == '"')
        {
            v.type = Value::Type::String;
            v.str = string();
            return v;
        }
        if (literal("null"))
            return v;
        if (literal("true"))
        {
            v.type = Value::Type::Bool;
            v.b = true;
            return v;
        }
        if (literal("false"))
        {
            v.type = Value::Type::Bool;
            return v;
        }
        const std::size_t start = m_i;
        while (m_i < m_s.size() && (std::string_view("+-0123456789.eE").find(m_s[m_i]) != std::string_view::npos))
            ++m_i;
        if (m_i == start)
            fail("unexpected character");
        v.type = Value::Type::Number;
        v.num = std::strtod(std::string(m_s.substr(start, m_i - start)).c_str(), nullptr);
        return v;
    }

    std::string_view m_s;
    std::size_t m_i = 0;
};

inline Value parse(std::string_view text) { return Parser(text).parse(); }

inline Value parse_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("json: cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse(ss.str());
}
}  // namespace mymyr::test::json
