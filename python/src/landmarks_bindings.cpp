// Landmarks and relaxed reachability in mymyr._core._search (mymyr.search; as in mimir's
// pymimir.advanced.search): FactLandmarkGraph, approximate_fact_landmarks, lifted_fact_landmarks,
// verify_pi_plus_fact_landmarks, RelaxedReachability with its tables, witness queries and conjunctive queries, and
// LandmarkTransitionOrdering (the width-1 transition ordering that iw() takes).
//
// Atoms are GroundAtoms (as input, any ground atom Task.atom accepts), actions mymyr Actions; the core's canonical ids
// stay inside. Generators and fixpoints run with the thread state detached. Graphs, reachability engines and tables are
// immutable and safe to share between threads; a WitnessQuery serializes its calls (it memoizes).

#include "py_landmarks.hpp"

#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mymyr::python
{
namespace
{
using namespace nb::literals;
namespace lm = mymyr::landmarks;
namespace rr = mymyr::reachability;

using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using StateArg = Arg<PyState>;
using AtomsArg = Arg<nb::typed<nb::iterable, AtomLike>>;
using AtomSetsArg = Arg<nb::typed<nb::iterable, nb::typed<nb::iterable, AtomLike>>>;
using OrderingsArg = Arg<nb::typed<nb::iterable, nb::typed<nb::tuple, AtomLike, AtomLike>>>;
using AtomList = nb::typed<nb::list, PyGroundAtom>;
using ActionList = nb::typed<nb::list, PyAction>;
using ReachabilityArg = Arg<PyRelaxedReachability>;
using PredicateArg = Arg<std::variant<std::string, int, PredicateView>>;
/// A term of a conjunctive query: an int is a variable index, a str an object name.
using TermArg = std::variant<int, std::string>;
using QueryLiteralArg = std::variant<nb::typed<nb::tuple, std::string, nb::typed<nb::sequence, TermArg>>,
                                     nb::typed<nb::tuple, std::string, nb::typed<nb::sequence, TermArg>, bool>>;
using TermPairsArg = Arg<nb::typed<nb::iterable, nb::typed<nb::tuple, TermArg, TermArg>>>;

struct PyLiftedLandmark
{
    lm::LiftedLandmark l;
    Owner o;
};

struct PyReachabilityTable
{
    std::shared_ptr<const rr::Table> t;  // aliases the engine for its main table
    Owner o;
};

struct PyWitnessQuery
{
    std::shared_ptr<const rr::Table> t;
    std::unique_ptr<rr::WitnessQuery> q;
    Owner o;
    std::unique_ptr<std::mutex> m = std::make_unique<std::mutex>();
};

// ------------------------------------------------------------------------------------------------ conversions

std::string lower(std::string s)
{
    for (char& c : s)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return s;
}

/// The canonical ids of `atoms`; atoms outside the task's canonical space are dropped (they can never hold).
std::vector<CanonicalAtom> canonical_atoms(const Owner& o, nb::handle atoms, bool derived)
{
    std::vector<CanonicalAtom> out;
    if (atoms.is_none())
        return out;
    for (nb::handle a : atoms)
        if (const auto c = canonical_atom(o, a, derived))
            out.push_back(*c);
    return out;
}

const PyFactLandmarkGraph& graph_arg(nb::handle h)
{
    if (!nb::isinstance<PyFactLandmarkGraph>(h))
        throw nb::type_error("mymyr: expected a mymyr.search.FactLandmarkGraph");
    return *nb::inst_ptr<PyFactLandmarkGraph>(h);
}

const PyRelaxedReachability* engine_arg(const Owner& o, nb::handle h)
{
    if (h.is_none())
        return nullptr;
    if (!nb::isinstance<PyRelaxedReachability>(h))
        throw nb::type_error("mymyr: reachability must be a mymyr.search.RelaxedReachability");
    const PyRelaxedReachability* e = nb::inst_ptr<PyRelaxedReachability>(h);
    same_task(o, e->o, "the reachability engine");
    return e;
}

State state_arg(const Owner& o, nb::handle h, const char* what)
{
    if (!is_state(h))
        throw nb::type_error((std::string("mymyr: ") + what + " must be a mymyr.State").c_str());
    const PyState& s = state_of(h);
    if (s.core->task->uid() != o.core->task->uid())
        throw nb::value_error((std::string("mymyr: ") + what + " belongs to another task").c_str());
    return s.s;
}

u32 predicate_arg(const Owner& o, nb::handle h)
{
    const formalism::TaskData& t = o.core->task->data();
    if (nb::isinstance<nb::int_>(h))
    {
        const i64 p = nb::cast<i64>(h);
        if (p < 0 || p >= static_cast<i64>(t.predicates.size()))
            throw nb::index_error("mymyr: predicate index out of range");
        return static_cast<u32>(p);
    }
    if (nb::isinstance<PredicateView>(h))
    {
        const PredicateView& v = *nb::inst_ptr<PredicateView>(h);
        if (v.t.get() == o.core->data.get())
            return v.i;
        const auto n = v.d().str(v.d().predicates[v.i].name);
        return predicate_arg(o, nb::str(n.data(), n.size()));
    }
    if (!nb::isinstance<nb::str>(h))
        throw nb::type_error("mymyr: a predicate is a name, an index or a mymyr.formalism.Predicate");
    const auto& names = o.core->names().predicates;
    const auto it = names.find(nb::cast<std::string>(h));
    if (it == names.end())
        throw nb::key_error(("mymyr: no predicate named '" + nb::cast<std::string>(h) + "'").c_str());
    return it->second;
}

template<class E>
E enum_arg(nb::handle h, std::initializer_list<std::pair<const char*, E>> names, const char* what)
{
    if (nb::isinstance<E>(h))
        return nb::cast<E>(h);
    if (nb::isinstance<nb::str>(h))
    {
        const std::string s = lower(nb::cast<std::string>(h));
        for (const auto& [n, v] : names)
            if (s == n)
                return v;
        std::string msg = std::string("mymyr: ") + what + " must be one of";
        for (const auto& [n, v] : names)
            msg += std::string(" '") + n + "'";
        throw nb::value_error(msg.c_str());
    }
    throw nb::type_error((std::string("mymyr: ") + what + " must be an enum member or its name").c_str());
}

std::string object_name(const Task& t, u32 o) { return std::string(t.data().str(t.data().objects[o].name)); }

// ------------------------------------------------------------------------------------------------ graphs

Arg<PyFactLandmarkGraph> make_graph(const Owner& o, lm::FactLandmarkGraph&& g)
{
    return nb::cast(PyFactLandmarkGraph{std::make_shared<const lm::FactLandmarkGraph>(std::move(g)), o}, nb::rv_policy::move);
}

/// The id of an action in the graph's achiever index (ValueError without one; k_no_action if it is not in the index).
u32 achiever_id(const PyFactLandmarkGraph& g, nb::handle action)
{
    if (!g.g->has_achiever_index())
        throw nb::value_error("mymyr: this landmark graph has no achiever index (only approximate_fact_landmarks builds one)");
    const Action a = action_of(g.o, action);
    return g.g->achievers().find(a.label());
}

std::optional<CanonicalAtom> graph_atom(const PyFactLandmarkGraph& g, nb::handle atom) { return canonical_atom(g.o, atom, false); }

ActionList action_ids(const PyFactLandmarkGraph& g, std::span<const u32> ids)
{
    ActionList out{nb::list()};
    for (u32 i : ids)
        out.append(action_object(g.o, g.g->achievers().actions[i]));
    return out;
}

std::string graph_repr(const PyFactLandmarkGraph& g)
{
    return "FactLandmarkGraph(landmarks=" + std::to_string(g.g->size()) + ", disjunctive=" +
           std::to_string(g.g->disjunctive().size()) + ", orderings=" + std::to_string(g.g->orderings().size()) +
           ", lifted=" + std::to_string(g.g->lifted().size()) + ", achiever_index=" +
           (g.g->has_achiever_index() ? "True" : "False") + ")";
}

Arg<PyReachabilityTable> make_table(const Owner& o, std::shared_ptr<const rr::Table> t)
{
    return nb::cast(PyReachabilityTable{std::move(t), o}, nb::rv_policy::move);
}

/// A conjunctive query from Python terms: ints are variables, strs object names.
rr::ConjunctiveQuery make_query(const Owner& o, u32 num_variables, nb::handle literals, nb::handle equalities,
                                nb::handle disequalities)
{
    const auto& objects = o.core->names().objects;
    auto term = [&](nb::handle x) -> rr::QueryTerm {
        if (nb::isinstance<nb::int_>(x))
        {
            const i64 v = nb::cast<i64>(x);
            if (v < 0 || v >= static_cast<i64>(num_variables))
                throw nb::value_error("mymyr: a query variable index must be in [0, num_variables)");
            return rr::query_variable(static_cast<u32>(v));
        }
        if (!nb::isinstance<nb::str>(x))
            throw nb::type_error("mymyr: a query term is a variable index (int) or an object name (str)");
        const auto it = objects.find(nb::cast<std::string>(x));
        if (it == objects.end())
            throw nb::key_error(("mymyr: no object named '" + nb::cast<std::string>(x) + "'").c_str());
        return rr::query_object(ObjectId{it->second});
    };
    rr::ConjunctiveQuery q;
    q.num_variables = num_variables;
    for (nb::handle l : literals)
    {
        nb::sequence seq = nb::borrow<nb::sequence>(l);
        const usize n = nb::len(seq);
        if (n != 2 && n != 3)
            throw nb::value_error("mymyr: a query literal is (predicate, terms) or (predicate, terms, positive)");
        rr::QueryLiteral ql;
        ql.predicate = PredicateId{predicate_arg(o, seq[0])};
        for (nb::handle x : nb::borrow<nb::iterable>(seq[1]))
            ql.terms.push_back(term(x));
        if (ql.terms.size() != o.core->task->data().predicates[ql.predicate.v].arity)
            throw nb::value_error("mymyr: a query literal has the wrong number of terms for its predicate");
        ql.positive = n == 2 || nb::cast<bool>(seq[2]);
        q.literals.push_back(std::move(ql));
    }
    auto pairs = [&](nb::handle h, std::vector<std::pair<rr::QueryTerm, rr::QueryTerm>>& out) {
        if (h.is_none())
            return;
        for (nb::handle p : h)
        {
            nb::sequence s = nb::borrow<nb::sequence>(p);
            if (nb::len(s) != 2)
                throw nb::value_error("mymyr: an (in)equality is a pair of terms");
            out.emplace_back(term(s[0]), term(s[1]));
        }
    };
    pairs(equalities, q.equalities);
    pairs(disequalities, q.disequalities);
    return q;
}
}  // namespace

void bind_landmarks(nb::module_& m)
{
    // enums ----------------------------------------------------------------------------------------------------------
    nb::enum_<lm::ReachabilityDisambiguation>(m, "ReachabilityDisambiguation",
                                              "How lifted_fact_landmarks narrows a lifted landmark's free parameters "
                                              "against relaxed reachability, beyond the static filter: OFF does not; "
                                              "PER_LITERAL narrows each precondition literal's variables against the "
                                              "reachable atoms on its own; JOINT narrows against the projection of the "
                                              "whole precondition conjunction.")
        .value("OFF", lm::ReachabilityDisambiguation::Off)
        .value("PER_LITERAL", lm::ReachabilityDisambiguation::PerLiteral)
        .value("JOINT", lm::ReachabilityDisambiguation::Joint);

    nb::enum_<lm::CompleteFactLandmarks>(m, "CompleteFactLandmarks",
                                         "Which atoms lifted_fact_landmarks tests for being fact landmarks beyond its "
                                         "own discoveries: OFF tests none; MEMBERS tests every member of every partial "
                                         "record; ALL tests every relaxed-reachable atom not true initially (one "
                                         "restricted fixpoint per atom).")
        .value("OFF", lm::CompleteFactLandmarks::Off)
        .value("MEMBERS", lm::CompleteFactLandmarks::Members)
        .value("ALL", lm::CompleteFactLandmarks::All);

    nb::enum_<rr::WitnessVerdict>(m, "WitnessVerdict", "Answer of a witness query (reachability/relaxed_reachability.hpp).")
        .value("REACHABLE_WITHOUT", rr::WitnessVerdict::ReachableWithout)
        .value("UNKNOWN", rr::WitnessVerdict::Unknown);

    // lifted landmarks -----------------------------------------------------------------------------------------------
    nb::class_<PyLiftedLandmark>(m, "LiftedLandmark",
                                 "A lifted landmark: predicate(objects) with free positions "
                                 "(None); every plan makes one of its members true.")
        .def_prop_ro("predicate",
                     [](const PyLiftedLandmark& x) {
                         const formalism::TaskData& t = x.o.core->task->data();
                         return std::string(t.str(t.predicates[x.l.predicate.v].name));
                     })
        .def_prop_ro("objects",
                     [](const PyLiftedLandmark& x) {
                         std::vector<std::optional<std::string>> out;
                         for (u32 b : x.l.binding)
                             out.push_back(b == lm::k_free ? std::nullopt
                                                           : std::optional<std::string>(object_name(*x.o.core->task, b)));
                         return out;
                     })
        .def_prop_ro("members", [](const PyLiftedLandmark& x) { return atom_list(x.o, x.l.members); })
        .def_prop_ro("fact",
                     [](const PyLiftedLandmark& x) -> Arg<std::optional<PyGroundAtom>> {
                         if (!x.l.is_fact())
                             return nb::none();
                         return atom_object(x.o, x.l.fact);
                     })
        .def_prop_ro("is_fact", [](const PyLiftedLandmark& x) { return x.l.is_fact(); })
        .def_prop_ro("parents", [](const PyLiftedLandmark& x) { return x.l.parents; },
                     "Indices into FactLandmarkGraph.lifted of the landmarks this one was back-chained from.")
        .def_prop_ro("initially_true", [](const PyLiftedLandmark& x) { return x.l.initially_true; })
        .def("__str__", [](const PyLiftedLandmark& x) { return lm::format_lifted(*x.o.core->task, x.l); })
        .def("__repr__", [](const PyLiftedLandmark& x) { return "LiftedLandmark(" + lm::format_lifted(*x.o.core->task, x.l) + ")"; });

    // the graph ------------------------------------------------------------------------------------------------------
    nb::class_<PyFactLandmarkGraph>(
        m, "FactLandmarkGraph",
        "The fact landmarks of a task (landmarks/fact_landmark_graph.hpp): fact landmarks "
        "(every plan makes each true at some point) in discovery order, disjunctive landmarks (sets of which every plan "
        "makes one member true), greedy-necessary orderings, and, from approximate_fact_landmarks, the achiever index. "
        "Immutable; LIW and abstracted IW take it as landmarks=.")
        .def(
            "__init__",
            [](PyFactLandmarkGraph* self, TaskArg task, AtomsArg landmarks, AtomSetsArg disjunctive,
               OrderingsArg orderings) {
                Owner o = owner_of(task);
                std::vector<CanonicalAtom> facts = canonical_atoms(o, landmarks, false);
                std::vector<std::vector<CanonicalAtom>> sets;
                if (!disjunctive.is_none())
                    for (nb::handle s : disjunctive)
                        sets.push_back(canonical_atoms(o, s, false));
                std::vector<std::pair<CanonicalAtom, CanonicalAtom>> order;
                if (!orderings.is_none())
                    for (nb::handle p : orderings)
                    {
                        nb::sequence s = nb::borrow<nb::sequence>(p);
                        if (nb::len(s) != 2)
                            throw nb::value_error("mymyr: an ordering is a pair (before, after)");
                        const auto a = canonical_atom(o, s[0], false), b = canonical_atom(o, s[1], false);
                        if (a && b)
                            order.emplace_back(*a, *b);
                    }
                new (self) PyFactLandmarkGraph{std::make_shared<const lm::FactLandmarkGraph>(
                                                   lm::FactLandmarkGraph::create(std::move(facts), std::move(sets), std::move(order))),
                                               std::move(o)};
            },
            "task"_a, "landmarks"_a, "disjunctive"_a = nb::none(), "orderings"_a = nb::none(),
            "A graph from given fluent atoms (mimir's FactLandmarkGraph.create): fact landmarks, disjunctive sets and "
            "(before, after) orderings, normalized as mimir does (duplicates dropped, sets holding a fact landmark "
            "dropped). Atoms outside the task's reachable domains are dropped (they can never hold). No achiever index.")
        .def_prop_ro("task", [](const PyFactLandmarkGraph& g) -> Arg<std::variant<PyTask, PyHandle>> { return g.o.obj; })
        .def_prop_ro("landmarks", [](const PyFactLandmarkGraph& g) { return atom_list(g.o, g.g->landmarks()); },
                     "The fact landmarks in discovery order.")
        .def("__len__", [](const PyFactLandmarkGraph& g) { return g.g->size(); })
        .def_prop_ro("disjunctive",
                     [](const PyFactLandmarkGraph& g) {
                         nb::typed<nb::list, AtomList> out{nb::list()};
                         for (const auto& set : g.g->disjunctive())
                             out.append(atom_list(g.o, set));
                         return out;
                     },
                     "The disjunctive landmarks (each ascending by canonical id; the list sorted).")
        .def_prop_ro("disjunctive_atoms", [](const PyFactLandmarkGraph& g) { return atom_list(g.o, g.g->disjunctive_atoms()); })
        .def_prop_ro("orderings",
                     [](const PyFactLandmarkGraph& g) {
                         nb::typed<nb::list, nb::typed<nb::tuple, PyGroundAtom, PyGroundAtom>> out{nb::list()};
                         for (const auto& [a, b] : g.g->orderings())
                             out.append(nb::make_tuple(atom_object(g.o, a), atom_object(g.o, b)));
                         return out;
                     },
                     "Greedy-necessary orderings (before, after) between fact landmarks.")
        .def(
            "is_landmark",
            [](const PyFactLandmarkGraph& g, AtomLike atom) {
                const auto c = graph_atom(g, atom);
                return c && g.g->is_landmark(*c);
            },
            "atom"_a)
        .def(
            "predecessors",
            [](const PyFactLandmarkGraph& g, AtomLike atom) {
                const auto c = graph_atom(g, atom);
                return atom_list(g.o, c ? g.g->predecessors(*c) : std::vector<CanonicalAtom>{});
            },
            "atom"_a, "The landmarks ordered directly before `atom`.")
        .def(
            "successors",
            [](const PyFactLandmarkGraph& g, AtomLike atom) {
                const auto c = graph_atom(g, atom);
                return atom_list(g.o, c ? g.g->successors(*c) : std::vector<CanonicalAtom>{});
            },
            "atom"_a, "The landmarks ordered directly after `atom`.")
        .def_prop_ro("has_achiever_index", [](const PyFactLandmarkGraph& g) { return g.g->has_achiever_index(); })
        .def(
            "achievers",
            [](const PyFactLandmarkGraph& g, AtomLike atom) {
                if (!g.g->has_achiever_index())
                    throw nb::value_error("mymyr: this landmark graph has no achiever index");
                const auto c = graph_atom(g, atom);
                return action_ids(g, c ? g.g->achievers().achievers_of(*c) : std::span<const u32>{});
            },
            "atom"_a, "Every ground action adding `atom` in the grounded relaxation (achiever index only).")
        .def(
            "first_achievers",
            [](const PyFactLandmarkGraph& g, AtomLike atom) {
                if (!g.g->has_achiever_index())
                    throw nb::value_error("mymyr: this landmark graph has no achiever index");
                const auto c = graph_atom(g, atom);
                return action_ids(g, c ? g.g->achievers().first_achievers_of(*c) : std::span<const u32>{});
            },
            "atom"_a, "The ground actions adding `atom` at its h_max cost (achiever index only).")
        .def(
            "unique_achiever",
            [](const PyFactLandmarkGraph& g, AtomLike atom) -> Arg<std::optional<PyAction>> {
                if (!g.g->has_achiever_index())
                    throw nb::value_error("mymyr: this landmark graph has no achiever index");
                const auto c = graph_atom(g, atom);
                const u32 a = c ? g.g->unique_achiever(*c) : lm::k_no_action;
                if (a == lm::k_no_action)
                    return nb::none();
                return action_object(g.o, g.g->achievers().actions[a]);
            },
            "atom"_a, "The only achiever of `atom`, or None.")
        .def(
            "achieved_by",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return atom_list(g.o, g.g->achievers().achieved_by(a));
            },
            "action"_a, "The landmarks the action adds.")
        .def(
            "first_achieved_by",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return atom_list(g.o, g.g->achievers().first_achieved_by(a));
            },
            "action"_a, "The landmarks the action first-achieves.")
        .def(
            "uniquely_achieved_by",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return atom_list(g.o, g.g->achievers().uniquely_achieved_by(a));
            },
            "action"_a, "The landmarks of which the action is the only achiever.")
        .def(
            "is_landmark_achiever",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return a != lm::k_no_action && g.g->is_landmark_achiever(a);
            },
            "action"_a)
        .def(
            "is_first_landmark_achiever",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return a != lm::k_no_action && g.g->is_first_landmark_achiever(a);
            },
            "action"_a)
        .def(
            "is_unique_landmark_achiever",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return a != lm::k_no_action && g.g->is_unique_landmark_achiever(a);
            },
            "action"_a)
        .def_prop_ro("lifted",
                     [](const PyFactLandmarkGraph& g) {
                         nb::typed<nb::list, PyLiftedLandmark> out{nb::list()};
                         for (const lm::LiftedLandmark& l : g.g->lifted())
                             out.append(nb::cast(PyLiftedLandmark{l, g.o}, nb::rv_policy::move));
                         return out;
                     },
                     "The lifted landmarks in discovery order (lifted_fact_landmarks only).")
        .def(
            "achieved",
            [](const PyFactLandmarkGraph& g, StateArg state) {
                const State s = state_arg(g.o, state, "state");
                return atom_list(g.o, g.g->achieved(*g.o.core->task, s.view()));
            },
            "state"_a, "The fact landmarks true in `state` (discovery order).")
        .def(
            "unachieved",
            [](const PyFactLandmarkGraph& g, StateArg state) {
                const State s = state_arg(g.o, state, "state");
                return atom_list(g.o, g.g->unachieved(*g.o.core->task, s.view()));
            },
            "state"_a, "The fact landmarks false in `state` (discovery order).")
        .def(
            "num_achieved",
            [](const PyFactLandmarkGraph& g, StateArg state) {
                const State s = state_arg(g.o, state, "state");
                return g.g->achieved(*g.o.core->task, s.view()).size();
            },
            "state"_a)
        .def(
            "num_unachieved",
            [](const PyFactLandmarkGraph& g, StateArg state) {
                const State s = state_arg(g.o, state, "state");
                return g.g->unachieved(*g.o.core->task, s.view()).size();
            },
            "state"_a, "The number of fact landmarks false in `state` (a landmark count without orderings).")
        .def("__repr__", &graph_repr);

    // generators -----------------------------------------------------------------------------------------------------
    m.def(
        "approximate_fact_landmarks",
        [](TaskArg task, bool include_positive_goal_facts, bool compute_greedy_necessary_orderings,
           usize max_disjunctive_landmark_size, usize max_disjunctive_landmark_depth, u64 max_operators, u64 max_bindings,
           double max_seconds) {
            const Owner o = owner_of(task);
            lm::ApproximateFactLandmarkOptions opts;
            opts.include_positive_goal_facts = include_positive_goal_facts;
            opts.compute_greedy_necessary_orderings = compute_greedy_necessary_orderings;
            opts.max_disjunctive_landmark_size = max_disjunctive_landmark_size;
            opts.max_disjunctive_landmark_depth = max_disjunctive_landmark_depth;
            opts.budget.max_operators = max_operators;
            opts.budget.max_bindings = max_bindings;
            opts.budget.max_seconds = max_seconds;
            const Task& t = *o.core->task;
            std::optional<lm::FactLandmarkGraph> g;
            {
                nb::gil_scoped_release release;
                g.emplace(lm::approximate_fact_landmarks(t, opts));
            }
            return make_graph(o, std::move(*g));
        },
        "task"_a, nb::kw_only(), "include_positive_goal_facts"_a = true, "compute_greedy_necessary_orderings"_a = true,
        "max_disjunctive_landmark_size"_a = 0, "max_disjunctive_landmark_depth"_a = 0, "max_operators"_a = 5'000'000,
        "max_bindings"_a = 100'000'000, "max_seconds"_a = 60.0,
        "Fact landmarks over the grounded delete relaxation (landmarks/approximate.hpp; as in mimir's "
        "ApproximateFactLandmarkGenerator), with the achiever index. Grounds the task: max_operators, max_bindings and "
        "max_seconds bound the grounding (RuntimeError beyond). max_disjunctive_landmark_size 0: no disjunctive "
        "landmarks; max_disjunctive_landmark_depth 0: unbounded.");

    m.def(
        "lifted_fact_landmarks",
        [](TaskArg task, ReachabilityArg reachability, bool include_positive_goal_facts,
           bool compute_greedy_necessary_orderings, bool use_static_filter, usize max_occurrence_combinations,
           usize max_disjunctive_members, bool reachability_filter_members,
           Arg<std::variant<lm::ReachabilityDisambiguation, std::string>> reachability_disambiguation,
           bool first_achievers_restricted, bool verify_pi_plus,
           Arg<std::variant<lm::CompleteFactLandmarks, std::string>> complete_fact_landmarks) {
            const Owner o = owner_of(task);
            const PyRelaxedReachability* engine = engine_arg(o, reachability);
            lm::LiftedFactLandmarkOptions opts;
            opts.include_positive_goal_facts = include_positive_goal_facts;
            opts.compute_greedy_necessary_orderings = compute_greedy_necessary_orderings;
            opts.use_static_filter = use_static_filter;
            opts.max_occurrence_combinations = max_occurrence_combinations;
            opts.max_disjunctive_members = max_disjunctive_members;
            opts.reachability_filter_members = reachability_filter_members;
            opts.reachability_disambiguation = enum_arg<lm::ReachabilityDisambiguation>(
                reachability_disambiguation,
                {{"off", lm::ReachabilityDisambiguation::Off},
                 {"per_literal", lm::ReachabilityDisambiguation::PerLiteral},
                 {"joint", lm::ReachabilityDisambiguation::Joint}},
                "reachability_disambiguation");
            opts.first_achievers_restricted = first_achievers_restricted;
            opts.verify_pi_plus = verify_pi_plus;
            opts.complete_fact_landmarks = enum_arg<lm::CompleteFactLandmarks>(
                complete_fact_landmarks,
                {{"off", lm::CompleteFactLandmarks::Off},
                 {"members", lm::CompleteFactLandmarks::Members},
                 {"all", lm::CompleteFactLandmarks::All}},
                "complete_fact_landmarks");
            const Task& t = *o.core->task;
            std::shared_ptr<const rr::RelaxedReachability> e = engine ? engine->rr : nullptr;
            std::optional<lm::FactLandmarkGraph> g;
            {
                nb::gil_scoped_release release;
                g.emplace(e ? lm::lifted_fact_landmarks(*e, opts) : lm::lifted_fact_landmarks(t, opts));
            }
            return make_graph(o, std::move(*g));
        },
        "task"_a, nb::kw_only(), "reachability"_a = nb::none(), "include_positive_goal_facts"_a = true,
        "compute_greedy_necessary_orderings"_a = true, "use_static_filter"_a = true, "max_occurrence_combinations"_a = 64,
        "max_disjunctive_members"_a = 0, "reachability_filter_members"_a = true,
        "reachability_disambiguation"_a = lm::ReachabilityDisambiguation::Joint, "first_achievers_restricted"_a = true,
        "verify_pi_plus"_a = true, "complete_fact_landmarks"_a = lm::CompleteFactLandmarks::Members,
        "Lifted fact landmarks straight from the schemas (landmarks/lifted.hpp; as in mimir's LiftedFactLandmarkGenerator): "
        "no grounding and no achiever index, but lifted landmarks (graph.lifted). reachability: a RelaxedReachability "
        "of the task (default options) to reuse; the enums also take their names ('joint', 'members', ...). Raises "
        "RuntimeError when verify_pi_plus finds a fact that is not a landmark.");

    m.def(
        "verify_pi_plus_fact_landmarks",
        [](Arg<PyFactLandmarkGraph> graph, ReachabilityArg reachability) {
            const PyFactLandmarkGraph& g = graph_arg(graph);
            const PyRelaxedReachability* engine = engine_arg(g.o, reachability);
            std::shared_ptr<const rr::RelaxedReachability> e = engine ? engine->rr : nullptr;
            std::shared_ptr<const lm::FactLandmarkGraph> gg = g.g;
            const Task& t = *g.o.core->task;
            nb::gil_scoped_release release;
            if (e)
                lm::verify_pi_plus_fact_landmarks(*e, *gg);
            else
                lm::verify_pi_plus_fact_landmarks(t, *gg);
        },
        "graph"_a, nb::kw_only(), "reachability"_a = nb::none(),
        "Checks that every fact landmark (not a positive goal atom, not initially true) blocks the relaxed goal once "
        "forbidden; RuntimeError naming the first that does not.");

    // relaxed reachability -------------------------------------------------------------------------------------------
    nb::class_<rr::Statistics>(m, "ReachabilityStatistics")
        .def_ro("rules", &rr::Statistics::rules)
        .def_ro("dropped_rules", &rr::Statistics::dropped_rules)
        .def_ro("plans", &rr::Statistics::plans)
        .def_ro("rounds", &rr::Statistics::rounds)
        .def_ro("fluent_atoms", &rr::Statistics::fluent_atoms)
        .def_ro("derived_atoms", &rr::Statistics::derived_atoms)
        .def_ro("bindings", &rr::Statistics::bindings)
        .def_ro("compile_ms", &rr::Statistics::compile_ms)
        .def_ro("fixpoint_ms", &rr::Statistics::fixpoint_ms)
        .def("__repr__", [](const rr::Statistics& s) {
            return "ReachabilityStatistics(rules=" + std::to_string(s.rules) + ", rounds=" + std::to_string(s.rounds) +
                   ", fluent_atoms=" + std::to_string(s.fluent_atoms) + ", derived_atoms=" + std::to_string(s.derived_atoms) + ")";
        });

    nb::class_<PyWitnessQuery>(m, "WitnessQuery",
                               "Is an atom reachable without the forbidden atoms? From the recorded first derivations "
                               "(REACHABLE_WITHOUT is sound, UNKNOWN carries no information); memoized, calls serialized.")
        .def(
            "avoids",
            [](PyWitnessQuery& q, AtomLike atom) {
                const auto c = canonical_atom(q.o, atom, true);
                if (!c)
                    return rr::WitnessVerdict::Unknown;
                nb::gil_scoped_release release;  // never block on the mutex while attached
                std::lock_guard lock(*q.m);
                return q.q->avoids(*c);
            },
            "atom"_a)
        .def_prop_ro("memoized", [](PyWitnessQuery& q) {
            u64 n = 0;
            {
                nb::gil_scoped_release release;
                std::lock_guard lock(*q.m);
                n = q.q->memoized();
            }
            return n;
        });

    nb::class_<PyReachabilityTable>(m, "ReachabilityTable",
                                    "The atoms one delete-relaxed fixpoint reached (RelaxedReachability.table, "
                                    ".restricted(...)). Immutable and thread-safe.")
        .def(
            "is_reachable",
            [](const PyReachabilityTable& t, AtomLike atom) {
                const auto c = canonical_atom(t.o, atom, true);
                return c && t.t->is_reachable(*c);
            },
            "atom"_a, "Whether a fluent or derived ground atom is reachable.")
        .def(
            "atoms",
            [](const PyReachabilityTable& t, PredicateArg predicate) {
                return atom_list(t.o, t.t->atoms(PredicateId{predicate_arg(t.o, predicate)}));
            },
            "predicate"_a, "The reachable atoms of a fluent or derived predicate (name, index or Predicate), in derivation order.")
        .def_prop_ro("num_atoms", [](const PyReachabilityTable& t) { return t.t->num_atoms(); })
        .def_prop_ro("num_fluent_atoms", [](const PyReachabilityTable& t) { return t.t->num_fluent_atoms(); })
        .def_prop_ro("num_derived_atoms", [](const PyReachabilityTable& t) { return t.t->num_derived_atoms(); })
        .def_prop_ro("goal_reachable", [](const PyReachabilityTable& t) { return t.t->goal_reachable(); })
        .def_prop_ro("rounds", [](const PyReachabilityTable& t) { return t.t->rounds(); })
        .def_prop_ro("has_witnesses", [](const PyReachabilityTable& t) { return t.t->has_witnesses(); })
        .def(
            "witness_query",
            [](const PyReachabilityTable& t, AtomsArg forbidden) -> Arg<PyWitnessQuery> {
                if (!t.t->has_witnesses())
                    throw nb::value_error("mymyr: this table records no witnesses (record_witnesses=False, or a "
                                          "restricted table)");
                const std::vector<CanonicalAtom> f = canonical_atoms(t.o, forbidden, true);
                auto q = std::make_unique<rr::WitnessQuery>(t.t->witness_query(f));
                return nb::cast(PyWitnessQuery{t.t, std::move(q), t.o}, nb::rv_policy::move);
            },
            "forbidden"_a)
        .def(
            "project",
            [](const PyReachabilityTable& t, u32 num_variables, nb::typed<nb::iterable, QueryLiteralArg> literals,
               TermPairsArg equalities, TermPairsArg disequalities) {
                const rr::ConjunctiveQuery q = make_query(t.o, num_variables, literals, equalities, disequalities);
                std::vector<std::vector<ObjectId>> r;
                {
                    nb::gil_scoped_release release;
                    r = t.t->project(q);
                }
                nb::typed<nb::list, nb::typed<nb::list, std::string>> out{nb::list()};
                for (const auto& v : r)
                {
                    nb::typed<nb::list, std::string> row{nb::list()};
                    for (ObjectId x : v)
                        row.append(object_name(*t.o.core->task, x.v));
                    out.append(row);
                }
                return out;
            },
            "num_variables"_a, "literals"_a, nb::kw_only(), "equalities"_a = nb::none(), "disequalities"_a = nb::none(),
            "A conjunctive query over this table: per variable the objects (names, ascending by index) it takes in some "
            "solution. literals: (predicate, terms) or (predicate, terms, positive), terms being variable indices (int) or "
            "object names (str); negative literals only over static predicates; (in)equalities: pairs of terms.");

    nb::class_<PyRelaxedReachability>(
        m, "RelaxedReachability",
        "Exact delete-relaxed reachability over ground atoms without grounding (reachability/relaxed_reachability.hpp; "
        "as in mimir's RelaxedReachability): the unrestricted fixpoint at construction, restricted fixpoints and the "
        "complete landmark test on demand. Immutable and thread-safe; the fixpoints run with the thread state detached.")
        .def(
            "__init__",
            [](PyRelaxedReachability* self, TaskArg task, bool enforce_negative_static, bool record_witnesses) {
                Owner o = owner_of(task);
                rr::Options opts;
                opts.enforce_negative_static = enforce_negative_static;
                opts.record_witnesses = record_witnesses;
                const Task& t = *o.core->task;
                std::shared_ptr<const rr::RelaxedReachability> e;
                {
                    nb::gil_scoped_release release;
                    e = rr::RelaxedReachability::create(t, opts);
                }
                new (self) PyRelaxedReachability{std::move(e), std::move(o)};
            },
            "task"_a, nb::kw_only(), "enforce_negative_static"_a = true, "record_witnesses"_a = true)
        .def_prop_ro("task", [](const PyRelaxedReachability& e) -> Arg<std::variant<PyTask, PyHandle>> { return e.o.obj; })
        .def_prop_ro("table",
                     [](const PyRelaxedReachability& e) {
                         return make_table(e.o, std::shared_ptr<const rr::Table>(e.rr, &e.rr->table()));
                     })
        .def(
            "is_reachable",
            [](const PyRelaxedReachability& e, AtomLike atom) {
                const auto c = canonical_atom(e.o, atom, true);
                return c && e.rr->is_reachable(*c);
            },
            "atom"_a)
        .def_prop_ro("goal_reachable", [](const PyRelaxedReachability& e) { return e.rr->goal_reachable(); })
        .def(
            "restricted",
            [](const PyRelaxedReachability& e, AtomsArg forbidden) {
                const std::vector<CanonicalAtom> f = canonical_atoms(e.o, forbidden, true);
                std::shared_ptr<const rr::Table> t;
                {
                    nb::gil_scoped_release release;
                    t = std::make_shared<const rr::Table>(e.rr->restricted(f));
                }
                return make_table(e.o, std::move(t));
            },
            "forbidden"_a, "The fixpoint with `forbidden` removed from the initial state and never derived.")
        .def(
            "goal_reachable_without",
            [](const PyRelaxedReachability& e, AtomsArg forbidden) {
                const std::vector<CanonicalAtom> f = canonical_atoms(e.o, forbidden, true);
                nb::gil_scoped_release release;
                return e.rr->goal_reachable_without(f);
            },
            "forbidden"_a, "The complete delete-relaxation landmark test: is the goal reachable without `forbidden`?")
        .def_prop_ro("statistics", [](const PyRelaxedReachability& e) { return e.rr->statistics(); })
        .def_prop_ro("enforce_negative_static", [](const PyRelaxedReachability& e) { return e.rr->options().enforce_negative_static; })
        .def_prop_ro("record_witnesses", [](const PyRelaxedReachability& e) { return e.rr->options().record_witnesses; })
        .def("__repr__", [](const PyRelaxedReachability& e) {
            const rr::Statistics& s = e.rr->statistics();
            return "RelaxedReachability(fluent_atoms=" + std::to_string(s.fluent_atoms) + ", derived_atoms=" +
                   std::to_string(s.derived_atoms) + ", goal_reachable=" + (e.rr->goal_reachable() ? "True" : "False") + ")";
        });

    // transition ordering --------------------------------------------------------------------------------------------
    nb::class_<lm::LandmarkTransitionScore>(m, "LandmarkTransitionScore")
        .def_ro("num_new_landmarks", &lm::LandmarkTransitionScore::num_new_landmarks)
        .def_ro("num_new_unique_landmarks", &lm::LandmarkTransitionScore::num_new_unique_landmarks)
        .def_ro("unique_achiever_action", &lm::LandmarkTransitionScore::unique_achiever_action)
        .def_ro("deletes_achieved_landmark", &lm::LandmarkTransitionScore::deletes_achieved_landmark)
        .def_ro("num_deleted_achieved_landmarks", &lm::LandmarkTransitionScore::num_deleted_achieved_landmarks)
        .def("__eq__", [](const lm::LandmarkTransitionScore& a, const lm::LandmarkTransitionScore& b) { return a == b; })
        .def("__repr__", [](const lm::LandmarkTransitionScore& s) {
            return "LandmarkTransitionScore(new=" + std::to_string(s.num_new_landmarks) + ", new_unique=" +
                   std::to_string(s.num_new_unique_landmarks) + ", unique_achiever_action=" +
                   (s.unique_achiever_action ? "True" : "False") + ", deleted=" +
                   std::to_string(s.num_deleted_achieved_landmarks) + ")";
        });

    nb::class_<PyLandmarkOrdering>(
        m, "LandmarkTransitionOrdering",
        "As in mimir's LandmarkTransitionOrderingStrategy (landmarks/transition_ordering.hpp), for IW's width-1 pass: "
        "iw(task, max_arity=1, transition_ordering=...). Scores each transition against the fact landmarks (more new "
        "ones, more new unique ones, a unique achiever action first, fewer deleted ones) and stable-sorts the layer. "
        "Needs a graph with an achiever index (approximate_fact_landmarks).")
        .def(
            "__init__",
            [](PyLandmarkOrdering* self, Arg<PyFactLandmarkGraph> graph, bool prefer_new_landmarks,
               bool prefer_unique_achievers, bool prefer_landmark_actions_when_deleting,
               bool prefer_fewer_deleted_landmarks) {
                const PyFactLandmarkGraph& g = graph_arg(graph);
                lm::LandmarkTransitionOrderingOptions opts;
                opts.prefer_new_landmarks = prefer_new_landmarks;
                opts.prefer_unique_achievers = prefer_unique_achievers;
                opts.prefer_landmark_actions_when_deleting = prefer_landmark_actions_when_deleting;
                opts.prefer_fewer_deleted_landmarks = prefer_fewer_deleted_landmarks;
                auto t = std::make_shared<const lm::LandmarkTransitionOrdering>(*g.o.core->task, *g.g, opts);
                new (self) PyLandmarkOrdering{std::move(t), g};
            },
            "graph"_a, nb::kw_only(), "prefer_new_landmarks"_a = true, "prefer_unique_achievers"_a = true,
            "prefer_landmark_actions_when_deleting"_a = true, "prefer_fewer_deleted_landmarks"_a = true)
        .def_prop_ro("graph", [](const PyLandmarkOrdering& x) { return x.graph; })
        .def(
            "score",
            [](const PyLandmarkOrdering& x, StateArg parent, ActionLike action, StateArg child) {
                const State p = state_arg(x.graph.o, parent, "parent");
                const State c = state_arg(x.graph.o, child, "child");
                const Action a = action_of(x.graph.o, action);
                return x.t->score(p.view(), a.label(), c.view());
            },
            "parent"_a, "action"_a, "child"_a)
        .def(
            "prefer",
            [](const PyLandmarkOrdering& x, const lm::LandmarkTransitionScore& a, const lm::LandmarkTransitionScore& b) {
                return x.t->prefer(a, b);
            },
            "a"_a, "b"_a, "Whether a transition scored `a` comes strictly before one scored `b`.")
        .def_prop_ro("prefer_new_landmarks", [](const PyLandmarkOrdering& x) { return x.t->options().prefer_new_landmarks; })
        .def_prop_ro("prefer_unique_achievers", [](const PyLandmarkOrdering& x) { return x.t->options().prefer_unique_achievers; })
        .def_prop_ro("prefer_landmark_actions_when_deleting",
                     [](const PyLandmarkOrdering& x) { return x.t->options().prefer_landmark_actions_when_deleting; })
        .def_prop_ro("prefer_fewer_deleted_landmarks",
                     [](const PyLandmarkOrdering& x) { return x.t->options().prefer_fewer_deleted_landmarks; });
}
}  // namespace mymyr::python
