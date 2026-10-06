#pragma once
// Stub annotations of the RL bindings (rl_torch_bindings.cpp, rl_jax_bindings.cpp, rl_ops_bindings.cpp): string
// arguments with a closed set of values render as typing.Literal (typing.hpp's Arg<T>; the functions check the values).

#include <nanobind/nanobind.h>

namespace mymyr::python::ann
{
/// The path of a device Env.
struct EnvPath
{
};
/// The strategy of a hindsight relabel (rl/her.hpp).
struct HerStrategy
{
};
/// What makes a state a dead end (rl::DeadEnd, rl/env.hpp).
struct DeadEnd
{
};
}  // namespace mymyr::python::ann

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::EnvPath>
{
    static constexpr auto Name = const_name("typing.Literal['auto', 'fast', 'general']");
};
template<>
struct type_caster<mymyr::python::ann::DeadEnd>
{
    static constexpr auto Name = const_name("typing.Literal['no_successors', 'none']");
};
template<>
struct type_caster<mymyr::python::ann::HerStrategy>
{
    static constexpr auto Name = const_name("typing.Literal['future', 'final', 'episode']");
};
}  // namespace nanobind::detail
