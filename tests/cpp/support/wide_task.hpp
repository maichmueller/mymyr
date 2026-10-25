#pragma once

#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/task.hpp"

#include <sstream>

namespace mymyr::test
{
/// Four reachable states over a dense binary predicate. Either goal atom can be added first; their conjunction
/// needs width two because both singletons are seen at depth one. The changing atoms occupy the last slots.
inline TaskPtr wide_task(u32 objects, TaskOptions::Atoms atoms = TaskOptions::Atoms::Dense, u32 initial_atoms = 1)
{
    std::ostringstream text;
    text << "O " << objects << "\nP 1\nF 2 cell\nSI 0\nFI " << initial_atoms << '\n';
    for (u32 i = 0; i < initial_atoms; ++i)
        text << "0 " << i / objects << ' ' << i % objects << '\n';
    text << "G 2\n0 1 " << objects - 1 << ' ' << objects - 1 << "\n0 1 " << objects - 1 << ' ' << objects - 2 << "\nA 3\n";
    for (u32 i = 0; i < 2; ++i)
    {
        const i32 a = -static_cast<i32>(objects), b = a + static_cast<i32>(i);
        text << "set" << i << " 0\nL 1\n0 0 " << a << ' ' << b << "\nE 1\n0\nL 0\nF 1\n0 1 " << a << ' ' << b << '\n';
    }
    text << "touch 2\nL 1\n0 1 0 1\nE 1\n0\nL 0\nF 1\n0 1 0 1\n";
    std::istringstream input(text.str());
    TaskOptions options;
    options.atoms = atoms;
    return Task::create(formalism::read_task_text(input), options);
}
}  // namespace mymyr::test
