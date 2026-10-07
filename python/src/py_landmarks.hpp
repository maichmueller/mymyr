#pragma once
// Python-side objects of the landmark and relaxed reachability API (landmarks_bindings.cpp, in mymyr.search as in
// mimir's pymimir.advanced.search), shared with search_bindings.cpp, whose LIW and abstracted IW take a
// FactLandmarkGraph and whose iw() takes a LandmarkTransitionOrdering. Also the conversions between Python atoms and
// actions and the core's canonical atom ids and Action labels.
//
// A FactLandmarkGraph names its atoms by canonical id (task/atom_index.hpp), which is a property of one task: every
// object keeps its task's owner (Task or TaskHandle) and checks that arguments come from the same task.

#include "py_task.hpp"

#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/landmarks/transition_ordering.hpp"
#include "mymyr/reachability/relaxed_reachability.hpp"
#include "mymyr/successor/action.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mymyr::python
{
struct PyFactLandmarkGraph
{
    std::shared_ptr<const landmarks::FactLandmarkGraph> g;
    Owner o;
};

struct PyRelaxedReachability
{
    std::shared_ptr<const reachability::RelaxedReachability> rr;
    Owner o;
};

struct PyLandmarkOrdering
{
    std::shared_ptr<const landmarks::LandmarkTransitionOrdering> t;
    PyFactLandmarkGraph graph;
};

/// Literal names for signatures (rendered in the stubs; the bindings check the values themselves).
namespace ann
{
struct LandmarkSource  // how LIW / abstracted IW compute a landmark graph when given a name
{
};
struct LayerOrder
{
};
struct BeamNovelty  // LayerOrdering::BeamNovelty
{
};
struct WidthZero  // the IW family's arity-0 pass
{
};
/// A search observer (mymyr._typing.SearchObserver): any object; the searches call whichever on_* methods it has.
struct Observer
{
};
/// A heuristic object written in Python (mymyr._typing.HeuristicObject): evaluate(state) and/or evaluate_batch(states).
struct HeuristicObject
{
};
}  // namespace ann

/// An Action (label) of `o`'s task.
[[nodiscard]] inline Arg<PyAction> action_object(const Owner& o, const Action& a)
{
    PyAction p;
    p.schema = a.schema.v;
    p.binding.resize(a.binding.size());
    for (usize i = 0; i < a.binding.size(); ++i)
        p.binding[i] = a.binding[i].v;
    p.owner = o.obj;
    p.core = o.core;
    return nb::cast(std::move(p), nb::rv_policy::move);
}

/// The Atom of a canonical id (fluent or derived) of `o`'s task. Fluent atoms carry their slot if they have one.
[[nodiscard]] inline Arg<PyAtom> atom_object(const Owner& o, CanonicalAtom c)
{
    const Task& t = *o.core->task;
    const CanonicalLayout& L = t.atoms().layout();
    PyAtom a;
    std::vector<u32> args(std::max<u32>(1, L.max_arity));
    a.pred = L.decode(c, args.data());
    args.resize(L.arity[a.pred]);
    a.args = std::move(args);
    if (c < L.fluent_count)
    {
        const u32 slot = t.atoms().find(c);
        a.slot = slot == AtomIndex::k_empty ? -1 : static_cast<i64>(slot);
    }
    a.owner = o.obj;
    a.core = o.core;
    return nb::cast(std::move(a), nb::rv_policy::move);
}

/// Canonical ids as Atoms, in the given order.
[[nodiscard]] inline nb::typed<nb::list, PyAtom> atom_list(const Owner& o, std::span<const CanonicalAtom> atoms)
{
    nb::typed<nb::list, PyAtom> out{nb::list()};
    for (CanonicalAtom c : atoms)
        out.append(atom_object(o, c));
    return out;
}

/// The canonical id of a ground atom given as anything Task.atom accepts. std::nullopt when it lies outside the task's
/// canonical space (an argument outside the reachable domain of its position: the atom can never hold). Raises
/// ValueError for static atoms, and for derived ones unless `derived` is set.
[[nodiscard]] inline std::optional<CanonicalAtom> canonical_atom(const Owner& o, nb::handle atom, bool derived)
{
    nb::object a = task_object(o.obj).attr("atom")(atom);
    const PyAtom& pa = nb::cast<const PyAtom&>(a);
    const Task& task = *o.core->task;
    const formalism::PredKind kind = task.compiled().kinds[pa.pred];
    if (kind == formalism::PredKind::Static || (kind == formalism::PredKind::Derived && !derived))
        throw nb::value_error(derived ? "mymyr: expected a fluent or derived atom (static atoms have no canonical id)"
                                      : "mymyr: expected a fluent atom (static and derived atoms are not supported)");
    const CanonicalLayout& L = task.atoms().layout();
    const CanonicalAtom c = L.encode(pa.pred, pa.args.data());
    if (c >= L.total)
        return std::nullopt;
    return c;
}

/// The label of an action given as anything Task.action accepts.
[[nodiscard]] inline Action action_of(const Owner& o, nb::handle action)
{
    nb::object a = task_object(o.obj).attr("action")(action);
    return Action(nb::cast<const PyAction&>(a).label());
}

/// Raises ValueError unless `other` belongs to the task of `o`.
inline void same_task(const Owner& o, const Owner& other, const char* what)
{
    if (other.core->task->uid() != o.core->task->uid())
        throw nb::value_error((std::string("mymyr: ") + what + " belongs to another task").c_str());
}

void bind_landmarks(nb::module_& m);
}  // namespace mymyr::python

namespace nanobind::detail
{
template<>
struct type_caster<mymyr::python::ann::LandmarkSource>
{
    static constexpr auto Name = const_name("typing.Literal['approximate', 'lifted']");
};
template<>
struct type_caster<mymyr::python::ann::LayerOrder>
{
    static constexpr auto Name = const_name("typing.Literal['queue', 'in_order', 'reverse', 'randomized', 'goal_count']");
};
template<>
struct type_caster<mymyr::python::ann::BeamNovelty>
{
    static constexpr auto Name = const_name("typing.Literal['all_tested', 'survivors_only']");
};
template<>
struct type_caster<mymyr::python::ann::WidthZero>
{
    static constexpr auto Name = const_name("typing.Literal['expand_depth_one', 'root_only']");
};
template<>
struct type_caster<mymyr::python::ann::Observer>
{
    static constexpr auto Name = const_name("mymyr._typing.SearchObserver");
};
template<>
struct type_caster<mymyr::python::ann::HeuristicObject>
{
    static constexpr auto Name = const_name("mymyr._typing.HeuristicObject");
};
}  // namespace nanobind::detail
