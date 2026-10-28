#pragma once
// Stub annotations of the RL bindings (rl_bindings.cpp, rl_torch_bindings.cpp, rl_jax_bindings.cpp,
// rl_ops_bindings.cpp): string arguments with a closed set of values render as typing.Literal (typing.hpp's Arg<T>; the
// functions check the values).

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
/// The action representation of a batched environment.
struct ActionSpace
{
};
/// An index array, a (schema, binding) pair, or the random policy.
struct EnvAction
{
};
/// An instance sampler argument: "auto" or "uniform" (uniform sampling) or a mymyr.rl.InstanceSampler.
struct SamplerArg
{
};
/// A mymyr.rl.InstanceSampler, or None.
struct Sampler
{
};
/// A packed host query table.
struct GoalTemplates
{
};
/// A packed query table uploaded to a torch CUDA device.
struct DeviceGoalTemplates
{
};
/// Which template variables bind distinct objects (rl::Distinct): True, False or "equality".
struct TemplateDistinct
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
struct type_caster<mymyr::python::ann::ActionSpace>
{
    static constexpr auto Name = const_name("typing.Literal['index', 'label']");
};
template<>
struct type_caster<mymyr::python::ann::EnvAction>
{
    static constexpr auto Name = const_name("mymyr._typing.SupportsDLPack | tuple[mymyr._typing.SupportsDLPack, "
                                          "mymyr._typing.SupportsDLPack]");
};
template<>
struct type_caster<mymyr::python::ann::SamplerArg>
{
    static constexpr auto Name = const_name("typing.Literal['auto', 'uniform'] | mymyr._typing.InstanceSampler");
};
template<>
struct type_caster<mymyr::python::ann::Sampler>
{
    static constexpr auto Name = const_name("mymyr._typing.InstanceSampler | None");
};
template<>
struct type_caster<mymyr::python::ann::HerStrategy>
{
    static constexpr auto Name = const_name("typing.Literal['future', 'final', 'episode']");
};
template<>
struct type_caster<mymyr::python::ann::GoalTemplates>
{
    static constexpr auto Name = const_name("mymyr._core._rl.GoalTemplates");
};
template<>
struct type_caster<mymyr::python::ann::DeviceGoalTemplates>
{
    static constexpr auto Name = const_name("mymyr._typing.DeviceGoalTemplates");
};
template<>
struct type_caster<mymyr::python::ann::TemplateDistinct>
{
    static constexpr auto Name = const_name("bool | typing.Literal['equality']");
};
}  // namespace nanobind::detail
