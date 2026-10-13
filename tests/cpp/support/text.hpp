#pragma once
// Test support: whitespace-separated tokens of a text, for comparisons that ignore the layout.

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace mymyr::test
{
inline std::vector<std::string> tokens(const std::string& text)
{
    std::istringstream in(text);
    std::vector<std::string> out;
    std::string w;
    while (in >> w)
        out.push_back(w);
    return out;
}

/// A name made of letters, digits and underscores, for naming parameterized tests.
inline std::string param_name(std::string n)
{
    for (char& c : n)
        if (!std::isalnum(static_cast<unsigned char>(c)))
            c = '_';
    return n;
}
}  // namespace mymyr::test
