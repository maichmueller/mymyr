// Landmarks and relaxed reachability in mymyr._core._search (mymyr.search): FactLandmarkGraph, approximate_fact_landmarks, lifted_fact_landmarks,
// verify_pi_plus_fact_landmarks, RelaxedReachability with its tables, witness queries and conjunctive queries, and
// LandmarkTransitionOrdering (the width-1 transition ordering that iw() takes).
//
// Atoms are GroundAtoms (as input, any ground atom Task.atom accepts), actions mymyr Actions; the core's canonical ids
// stay inside. Generators and fixpoints run with the thread state detached. Graphs, reachability engines and tables are
// immutable and safe to share between threads; a WitnessQuery serializes its calls (it memoizes).

#include "named_arrays.hpp"
#include "py_landmarks.hpp"

#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
        const i64 p = int_arg<i64>(h, "a predicate index");
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
        throw nb::value_error("mymyr: this landmark graph has no achiever index (approximate_fact_landmarks builds one, "
                              "FactLandmarkGraph(..., achiever_index=True) rebuilds one)");
    const Action a = action_of(g.o, action);
    return g.g->achievers().find(a.label());
}

std::optional<CanonicalAtom> graph_atom(const PyFactLandmarkGraph& g, nb::handle atom) { return canonical_atom(g.o, atom, true); }

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

heuristics::GroundingBudget grounding_budget(IntArg max_operators, IntArg max_bindings, FloatArg max_seconds, IntArg max_ground_actions)
{
    heuristics::GroundingBudget budget;
    budget.max_operators = int_arg<u64>(max_operators, "max_operators");
    budget.max_bindings = int_arg<u64>(max_bindings, "max_bindings");
    budget.max_seconds = opt_float_arg(max_seconds, "max_seconds", 0).value_or(std::numeric_limits<double>::infinity());
    budget.max_ground_actions = opt_int_arg<u64>(max_ground_actions, "max_ground_actions").value_or(~u64{0});
    return budget;
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
            const i64 v = int_arg<i64>(x, "a query variable index");
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

// ------------------------------------------------------------------------------------------------ arrays

using ArrayDict = nb::typed<nb::dict, std::string, ann::Any>;
using ArraysArg = Arg<nb::typed<nb::mapping, std::string, ann::ArrayLike>>;

constexpr const char* k_graph = "landmark graph";
constexpr std::string_view k_lifted_keys[] = {"lifted_predicate",       "lifted_binding_offsets", "lifted_binding",
                                              "lifted_members_offsets", "lifted_members",         "lifted_fact",
                                              "lifted_parents_offsets", "lifted_parents",         "lifted_initially_true"};
constexpr std::string_view k_index_keys[] = {"actions_schema",
                                             "actions_binding",
                                             "atoms",
                                             "achievers_offsets",
                                             "achievers",
                                             "first_achievers_offsets",
                                             "first_achievers",
                                             "achieved_offsets",
                                             "achieved",
                                             "first_achieved_offsets",
                                             "first_achieved",
                                             "uniquely_achieved_offsets",
                                             "uniquely_achieved"};

/// The widest schema of a task (the second dimension of actions_binding).
u32 label_width(const Task& t)
{
    u32 w = 0;
    for (const formalism::Schema& s : t.data().schemas)
        w = std::max(w, s.arity());
    return w;
}

/// Lists as CSR: offsets (one per list and the total) and the concatenation.
template<class T, class Lists, class Get>
void put_csr(ArrayDict& d, const char* offsets_key, const char* values_key, const Lists& lists, Get get)
{
    std::vector<u64> offsets{0};
    std::vector<T> values;
    for (const auto& x : lists)
    {
        const auto& v = get(x);
        values.insert(values.end(), v.begin(), v.end());
        offsets.push_back(values.size());
    }
    d[offsets_key] = fresh(std::move(offsets));
    d[values_key] = fresh(std::move(values));
}

template<class T>
void put_index_csr(ArrayDict& d, const char* offsets_key, const char* values_key, const std::vector<u32>& begin,
                   const std::vector<T>& values)
{
    d[offsets_key] = fresh(std::vector<u64>(begin.begin(), begin.end()));
    d[values_key] = fresh(std::vector<T>(values));
}

ArrayDict graph_arrays(const lm::FactLandmarkGraph& g, const Task& t)
{
    ArrayDict d{nb::dict()};
    d["fingerprint"] = fresh(std::vector<u64>{t.fingerprint()});
    d["landmarks"] = fresh(std::vector<u64>(g.landmarks().begin(), g.landmarks().end()));
    put_csr<u64>(d, "disjunctive_offsets", "disjunctive_atoms", g.disjunctive(), [](const auto& s) -> const auto& { return s; });
    std::vector<u64> orderings;
    for (const auto& [a, b] : g.orderings())
    {
        orderings.push_back(a);
        orderings.push_back(b);
    }
    d["orderings"] = fresh(std::move(orderings), {static_cast<i64>(g.orderings().size()), 2});
    const auto& lifted = g.lifted();
    std::vector<u32> predicate;
    std::vector<u64> fact;
    std::vector<u8> initially_true;
    for (const lm::LiftedLandmark& l : lifted)
    {
        predicate.push_back(l.predicate.v);
        fact.push_back(l.fact);
        initially_true.push_back(l.initially_true ? 1 : 0);
    }
    d["lifted_predicate"] = fresh(std::move(predicate));
    std::vector<u64> binding_offsets{0};
    std::vector<i32> binding;
    for (const lm::LiftedLandmark& l : lifted)
    {
        for (u32 b : l.binding)
            binding.push_back(b == lm::k_free ? -1 : static_cast<i32>(b));
        binding_offsets.push_back(binding.size());
    }
    d["lifted_binding_offsets"] = fresh(std::move(binding_offsets));
    d["lifted_binding"] = fresh(std::move(binding));
    put_csr<u64>(d, "lifted_members_offsets", "lifted_members", lifted, [](const lm::LiftedLandmark& l) -> const auto& { return l.members; });
    d["lifted_fact"] = fresh(std::move(fact));
    put_csr<u32>(d, "lifted_parents_offsets", "lifted_parents", lifted, [](const lm::LiftedLandmark& l) -> const auto& { return l.parents; });
    d["lifted_initially_true"] = fresh(std::move(initially_true));
    if (!g.has_achiever_index())
        return d;
    const lm::AchieverIndex& ix = g.achievers();
    const u32 W = label_width(t);
    const usize A = ix.actions.size();
    std::vector<i32> schema(A), bind(A * W, -1);
    for (usize i = 0; i < A; ++i)
    {
        schema[i] = static_cast<i32>(ix.actions[i].schema.v);
        for (usize j = 0; j < ix.actions[i].binding.size(); ++j)
            bind[i * W + j] = static_cast<i32>(ix.actions[i].binding[j].v);
    }
    d["actions_schema"] = fresh(std::move(schema));
    d["actions_binding"] = fresh(std::move(bind), {static_cast<i64>(A), static_cast<i64>(W)});
    d["atoms"] = fresh(std::vector<u64>(ix.atoms));
    put_index_csr(d, "achievers_offsets", "achievers", ix.achievers_begin, ix.achievers);
    put_index_csr(d, "first_achievers_offsets", "first_achievers", ix.first_achievers_begin, ix.first_achievers);
    put_index_csr(d, "achieved_offsets", "achieved", ix.achieved_begin, ix.achieved);
    put_index_csr(d, "first_achieved_offsets", "first_achieved", ix.first_achieved_begin, ix.first_achieved);
    put_index_csr(d, "uniquely_achieved_offsets", "uniquely_achieved", ix.uniquely_achieved_begin, ix.uniquely_achieved);
    return d;
}

[[noreturn]] void corrupt(const std::string& what) { throw nb::value_error(("mymyr: the landmark graph arrays: " + what).c_str()); }

/// Offsets of a CSR (ValueError unless they start at 0, never decrease and end at `size`; `lists`: their number, if
/// known).
std::vector<u64> offsets_arg(nb::handle arrays, const char* key, u64 size, std::optional<u64> lists)
{
    std::vector<u64> o = take_required<u64>(arrays, key, 1, k_graph);
    if (o.empty() || o.front() != 0 || o.back() != size)
        corrupt(std::string("'") + key + "' must start at 0 and end at the length of its list");
    if (lists && o.size() != *lists + 1)
        corrupt(std::string("'") + key + "' must have " + std::to_string(*lists + 1) + " entries");
    for (usize i = 0; i + 1 < o.size(); ++i)
        if (o[i] > o[i + 1])
            corrupt(std::string("'") + key + "' decreases");
    return o;
}

/// A CSR of `arrays` as lists.
template<class T>
std::vector<std::vector<T>> lists_arg(nb::handle arrays, const char* offsets_key, const char* values_key, std::optional<u64> lists)
{
    const std::vector<T> values = take_required<T>(arrays, values_key, 1, k_graph);
    const std::vector<u64> o = offsets_arg(arrays, offsets_key, values.size(), lists);
    std::vector<std::vector<T>> out(o.size() - 1);
    for (usize i = 0; i + 1 < o.size(); ++i)
        out[i].assign(values.begin() + static_cast<std::ptrdiff_t>(o[i]), values.begin() + static_cast<std::ptrdiff_t>(o[i + 1]));
    return out;
}

/// A CSR of the achiever index: u32 offsets (`lists` + 1 of them) and the values.
template<class T>
void index_csr_arg(nb::handle arrays, const char* offsets_key, const char* values_key, u64 lists, std::vector<u32>& begin,
                   std::vector<T>& values)
{
    values = take_required<T>(arrays, values_key, 1, k_graph);
    if (values.size() > std::numeric_limits<u32>::max())
        corrupt(std::string("'") + values_key + "' is longer than the index holds (2^32 - 1 entries)");
    const std::vector<u64> o = offsets_arg(arrays, offsets_key, values.size(), lists);
    begin.assign(o.begin(), o.end());
}

/// Whether `arrays` holds some / every key of a group (ValueError for a partial group).
bool group_arg(nb::handle arrays, std::span<const std::string_view> keys, const char* what)
{
    usize n = 0;
    for (std::string_view k : keys)
        n += mapping_item(arrays, std::string(k).c_str()).is_none() ? 0 : 1;
    if (n != 0 && n != keys.size())
        corrupt(std::string("the ") + what + " arrays come all together or not at all");
    return n != 0;
}

/// The graph of `task` from the arrays of to_arrays (validated: landmarks::check).
lm::FactLandmarkGraph graph_from_arrays(const Task& t, nb::handle arrays)
{
    std::vector<std::string_view> known = {"fingerprint", "landmarks", "disjunctive_offsets", "disjunctive_atoms", "orderings"};
    known.insert(known.end(), std::begin(k_lifted_keys), std::end(k_lifted_keys));
    known.insert(known.end(), std::begin(k_index_keys), std::end(k_index_keys));
    if (!nb::isinstance<nb::dict>(arrays) && (!PyMapping_Check(arrays.ptr()) || !nb::hasattr(arrays, "keys")))
        throw nb::type_error("mymyr: landmark graph arrays must be a mapping of names to arrays (as to_arrays() gives them)");
    bool has_fingerprint = false;
    {
        nb::object keys = nb::steal(PyMapping_Keys(arrays.ptr()));
        if (!keys.is_valid())
            throw nb::python_error();
        for (nb::handle k : keys)
        {
            if (!nb::isinstance<nb::str>(k) || std::ranges::find(known, nb::cast<std::string_view>(k)) == known.end())
                throw nb::value_error(("mymyr: unknown landmark graph array " + nb::cast<std::string>(nb::repr(k))).c_str());
            has_fingerprint |= nb::cast<std::string_view>(k) == "fingerprint";
        }
    }
    if (auto fingerprint = take<u64>(arrays, "fingerprint", 1, has_fingerprint, k_graph))
    {
        if (fingerprint->first.size() != 1)
            corrupt("'fingerprint' must have shape [1]");
        if (fingerprint->first[0] != t.fingerprint())
            throw nb::value_error("mymyr: the landmark graph arrays are of another task (content fingerprint mismatch)");
    }
    std::vector<CanonicalAtom> landmarks = take_required<u64>(arrays, "landmarks", 1, k_graph);
    std::vector<std::vector<CanonicalAtom>> disjunctive;
    const bool has_disjunctive = !mapping_item(arrays, "disjunctive_atoms").is_none() || !mapping_item(arrays, "disjunctive_offsets").is_none();
    if (has_disjunctive)
        disjunctive = lists_arg<u64>(arrays, "disjunctive_offsets", "disjunctive_atoms", std::nullopt);
    std::vector<std::pair<CanonicalAtom, CanonicalAtom>> orderings;
    if (auto o = take<u64>(arrays, "orderings", 2, false, k_graph))
    {
        if (o->second[1] != 2)
            corrupt("'orderings' must have shape [E, 2]");
        for (usize i = 0; i < o->second[0]; ++i)
            orderings.emplace_back(o->first[2 * i], o->first[2 * i + 1]);
    }
    std::vector<lm::LiftedLandmark> lifted;
    if (group_arg(arrays, k_lifted_keys, "lifted landmark"))
    {
        const std::vector<u32> predicate = take_required<u32>(arrays, "lifted_predicate", 1, k_graph);
        const u64 K = predicate.size();
        const auto binding = lists_arg<i32>(arrays, "lifted_binding_offsets", "lifted_binding", K);
        auto members = lists_arg<u64>(arrays, "lifted_members_offsets", "lifted_members", K);
        auto parents = lists_arg<u32>(arrays, "lifted_parents_offsets", "lifted_parents", K);
        const std::vector<u64> fact = take_required<u64>(arrays, "lifted_fact", 1, k_graph);
        const std::vector<u8> initially_true = take_required<u8>(arrays, "lifted_initially_true", 1, k_graph);
        if (fact.size() != K || initially_true.size() != K)
            corrupt("'lifted_fact' and 'lifted_initially_true' must have one entry per lifted landmark");
        lifted.resize(K);
        for (u64 k = 0; k < K; ++k)
        {
            lm::LiftedLandmark& l = lifted[k];
            l.predicate = PredicateId{predicate[k]};
            for (i32 b : binding[k])
            {
                if (b < -1)
                    corrupt("'lifted_binding' holds an object id below -1");
                l.binding.push_back(b == -1 ? lm::k_free : static_cast<u32>(b));
            }
            l.members = std::move(members[k]);
            l.fact = fact[k];
            l.parents = std::move(parents[k]);
            l.initially_true = initially_true[k] != 0;
        }
    }
    std::optional<lm::AchieverIndex> index;
    if (group_arg(arrays, k_index_keys, "achiever index"))
    {
        lm::AchieverIndex& ix = index.emplace();
        const std::vector<i32> schema = take_required<i32>(arrays, "actions_schema", 1, k_graph);
        const auto bind = take<i32>(arrays, "actions_binding", 2, true, k_graph);
        const u64 A = schema.size(), W = bind->second[1];
        if (bind->second[0] != A)
            corrupt("'actions_binding' must have one row per entry of 'actions_schema'");
        if (A > std::numeric_limits<u32>::max())
            corrupt("the achiever index holds more than 2^32 - 1 actions");
        const auto& schemas = t.data().schemas;
        ix.actions.reserve(A);
        for (u64 i = 0; i < A; ++i)
        {
            if (schema[i] < 0 || static_cast<u64>(schema[i]) >= schemas.size())
                corrupt("'actions_schema' holds a schema id outside the task");
            const u32 arity = schemas[static_cast<u32>(schema[i])].arity();
            if (arity > W)
                corrupt("an action of 'actions_binding' is narrower than its schema's arity");
            std::vector<ObjectId> b(arity);
            for (u64 j = 0; j < W; ++j)
            {
                const i32 x = bind->first[i * W + j];
                if (j < arity ? x < 0 : x != -1)
                    corrupt("an action of 'actions_binding' does not hold its schema's arity of objects, padded with -1");
                if (j < arity)
                    b[j] = ObjectId{static_cast<u32>(x)};
            }
            ix.actions.emplace_back(SchemaId{static_cast<u32>(schema[i])}, std::move(b));
        }
        ix.atoms = take_required<u64>(arrays, "atoms", 1, k_graph);
        index_csr_arg(arrays, "achievers_offsets", "achievers", ix.atoms.size(), ix.achievers_begin, ix.achievers);
        index_csr_arg(arrays, "first_achievers_offsets", "first_achievers", ix.atoms.size(), ix.first_achievers_begin, ix.first_achievers);
        index_csr_arg(arrays, "achieved_offsets", "achieved", A, ix.achieved_begin, ix.achieved);
        index_csr_arg(arrays, "first_achieved_offsets", "first_achieved", A, ix.first_achieved_begin, ix.first_achieved);
        index_csr_arg(arrays, "uniquely_achieved_offsets", "uniquely_achieved", A, ix.uniquely_achieved_begin, ix.uniquely_achieved);
    }
    nb::gil_scoped_release release;
    lm::FactLandmarkGraph g = lm::FactLandmarkGraph::create(std::move(landmarks), std::move(disjunctive), std::move(orderings),
                                                            std::move(lifted), std::move(index));
    lm::check(t, g);
    return g;
}
}  // namespace

void bind_landmarks(nb::module_& m)
{
    // enums ----------------------------------------------------------------------------------------------------------
    nb::enum_<lm::FirstAchievers>(m, "FirstAchievers",
                                  "The first achievers of approximate_fact_landmarks: HMAX takes the "
                                  "operators reaching an atom at its h_max-style cost, which is not sound in general (an "
                                  "atom every cheapest achiever needs is a landmark although a costlier achiever does "
                                  "without it); POSSIBLY_BEFORE takes the operators applicable in the delete relaxation "
                                  "before the atom can first hold (Zhu and Givan; Richter, Helmert and Westphal), which "
                                  "makes the landmarks and orderings sound.")
        .value("HMAX", lm::FirstAchievers::HMax)
        .value("POSSIBLY_BEFORE", lm::FirstAchievers::PossiblyBefore);

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

    nb::enum_<rr::WitnessVerdict>(m, "WitnessVerdict", "Answer of a witness query.")
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
                     },
                     "The predicate name.")
        .def_prop_ro("objects",
                     [](const PyLiftedLandmark& x) {
                         std::vector<std::optional<std::string>> out;
                         for (u32 b : x.l.binding)
                             out.push_back(b == lm::k_free ? std::nullopt
                                                           : std::optional<std::string>(object_name(*x.o.core->task, b)));
                         return out;
                     },
                     "Per argument the object name, or None for a free position.")
        .def_prop_ro("members", [](const PyLiftedLandmark& x) { return atom_list(x.o, x.l.members); },
                     "The ground atoms it stands for; every plan makes one of them true.")
        .def_prop_ro("fact",
                     [](const PyLiftedLandmark& x) -> Arg<std::optional<PyGroundAtom>> {
                         if (!x.l.is_fact())
                             return nb::none();
                         return atom_object(x.o, x.l.fact);
                     },
                     "The GroundAtom when the landmark has no free position, else None.")
        .def_prop_ro("is_fact", [](const PyLiftedLandmark& x) { return x.l.is_fact(); },
                     "Whether it has no free position (a fact landmark).")
        .def_prop_ro("parents", [](const PyLiftedLandmark& x) { return x.l.parents; },
                     "Indices into FactLandmarkGraph.lifted of the landmarks this one was back-chained from.")
        .def_prop_ro("initially_true", [](const PyLiftedLandmark& x) { return x.l.initially_true; },
                     "Whether it holds in the initial state.")
        .def("__str__", [](const PyLiftedLandmark& x) { return lm::format_lifted(*x.o.core->task, x.l); })
        .def("__repr__", [](const PyLiftedLandmark& x) { return "LiftedLandmark(" + lm::format_lifted(*x.o.core->task, x.l) + ")"; });

    // the graph ------------------------------------------------------------------------------------------------------
    nb::class_<PyFactLandmarkGraph>(
        m, "FactLandmarkGraph",
        "The fact landmarks of a task: fact landmarks "
        "(every plan makes each true at some point) in discovery order, disjunctive landmarks (sets of which every plan "
        "makes one member true), greedy-necessary orderings, and, from approximate_fact_landmarks, the achiever index. "
        "Immutable; LIW and abstracted IW take it as landmarks=.")
        .def(
            "__init__",
            [](PyFactLandmarkGraph* self, TaskArg task, AtomsArg landmarks, AtomSetsArg disjunctive,
               OrderingsArg orderings, bool achiever_index, Arg<std::variant<lm::FirstAchievers, std::string>> first_achievers,
               IntArg max_operators_in, IntArg max_bindings_in, FloatArg max_seconds_in, IntArg max_ground_actions_in) {
                Owner o = owner_of(task);
                const lm::FirstAchievers rule = enum_arg<lm::FirstAchievers>(
                    first_achievers, {{"hmax", lm::FirstAchievers::HMax}, {"possibly_before", lm::FirstAchievers::PossiblyBefore}},
                    "first_achievers");
                const auto budget = grounding_budget(max_operators_in, max_bindings_in, max_seconds_in, max_ground_actions_in);
                std::vector<CanonicalAtom> facts = canonical_atoms(o, landmarks, true);
                std::vector<std::vector<CanonicalAtom>> sets;
                if (!disjunctive.is_none())
                    for (nb::handle s : disjunctive)
                        sets.push_back(canonical_atoms(o, s, true));
                std::vector<std::pair<CanonicalAtom, CanonicalAtom>> order;
                if (!orderings.is_none())
                    for (nb::handle p : orderings)
                    {
                        nb::sequence s = nb::borrow<nb::sequence>(p);
                        if (nb::len(s) != 2)
                            throw nb::value_error("mymyr: an ordering is a pair (before, after)");
                        const auto a = canonical_atom(o, s[0], true), b = canonical_atom(o, s[1], true);
                        if (a && b)
                            order.emplace_back(*a, *b);
                    }
                lm::FactLandmarkGraph g = lm::FactLandmarkGraph::create(std::move(facts), std::move(sets), std::move(order));
                if (achiever_index)
                {
                    const Task& t = *o.core->task;
                    nb::gil_scoped_release release;
                    std::vector<CanonicalAtom> lms(g.landmarks().begin(), g.landmarks().end());
                    auto sets_n = g.disjunctive();
                    std::vector<std::pair<CanonicalAtom, CanonicalAtom>> order_n(g.orderings().begin(), g.orderings().end());
                    lm::AchieverIndex ix = lm::build_achiever_index(t, lms, sets_n, rule, budget);
                    g = lm::FactLandmarkGraph::create(std::move(lms), std::move(sets_n), std::move(order_n), {}, std::move(ix));
                }
                new (self) PyFactLandmarkGraph{std::make_shared<const lm::FactLandmarkGraph>(std::move(g)), std::move(o)};
            },
            "task"_a, "landmarks"_a, "disjunctive"_a = nb::none(), "orderings"_a = nb::none(), nb::kw_only(),
            "achiever_index"_a = false, "first_achievers"_a = lm::FirstAchievers::HMax, "max_operators"_a = 5'000'000,
            "max_bindings"_a = 100'000'000, "max_seconds"_a.none() = 60.0, "max_ground_actions"_a = nb::none(),
            "A graph from given fluent or derived atoms: fact landmarks, disjunctive sets and (before, after) orderings, "
            "normalized (duplicates dropped, sets holding a fact landmark dropped). Atoms outside the task's reachable "
            "domains are dropped (they can never hold).\n\n"
            "achiever_index=True grounds the task as approximate_fact_landmarks does: max_operators, max_bindings, "
            "max_seconds (None: unlimited) and max_ground_actions (None: unlimited) bound the grounding "
            "(RuntimeError beyond). It builds the achiever index for these landmarks: the achievers of every atom, the "
            "first achievers by first_achievers (a FirstAchievers or 'hmax' / 'possibly_before'; HMAX: of every atom, "
            "POSSIBLY_BEFORE: of every fact landmark and disjunctive member) and per action the fact landmarks it "
            "achieves, first-achieves and uniquely achieves. First achievers are computed from the task's initial "
            "state, including for subgoals taken from later states. Derived atoms have no action achievers because "
            "axioms derive them; achieved / unachieved see fluent atoms only. On the landmarks of approximate_fact_landmarks with the "
            "same first_achievers the index equals the generator's, except that with POSSIBLY_BEFORE the generator "
            "lists first achievers only for the atoms it expanded, so first_achievers() of an atom that is not a fact "
            "landmark can differ. To keep a computed index, pickle the graph or use to_arrays / from_arrays. Without "
            "achiever_index the graph has no index (first_achievers and the grounding budgets are not used).")
        .def_prop_ro("task", [](const PyFactLandmarkGraph& g) -> Arg<std::variant<PyTask, PyHandle>> { return g.o.obj; },
                     "The Task (or TaskHandle) the graph was computed for.")
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
        .def_prop_ro("disjunctive_atoms", [](const PyFactLandmarkGraph& g) { return atom_list(g.o, g.g->disjunctive_atoms()); },
                     "Every atom of some disjunctive landmark, ascending and without duplicates.")
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
            "atom"_a,
            "Whether `atom` is a fact landmark.")
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
        .def_prop_ro("has_achiever_index", [](const PyFactLandmarkGraph& g) { return g.g->has_achiever_index(); },
                     "Whether the graph has an achiever index (approximate_fact_landmarks and "
                     "FactLandmarkGraph(..., achiever_index=True) build one, lifted_fact_landmarks does not); the "
                     "achiever methods need it.")
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
            "atom"_a,
            "The first achievers of `atom` by the FirstAchievers rule the index was built with (HMAX: the "
            "ground actions adding it at its h_max cost; achiever index only). Computed from the task's initial "
            "state, including for subgoals taken from later states; empty for derived atoms, which axioms derive.")
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
            "action"_a,
            "Whether `action` adds some fact landmark.")
        .def(
            "is_first_landmark_achiever",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return a != lm::k_no_action && g.g->is_first_landmark_achiever(a);
            },
            "action"_a,
            "Whether `action` is a first achiever of some fact landmark.")
        .def(
            "is_unique_landmark_achiever",
            [](const PyFactLandmarkGraph& g, ActionLike action) {
                const u32 a = achiever_id(g, action);
                return a != lm::k_no_action && g.g->is_unique_landmark_achiever(a);
            },
            "action"_a,
            "Whether `action` is the only achiever of some fact landmark.")
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
            "state"_a,
            "The number of fact landmarks true in `state`.")
        .def(
            "num_unachieved",
            [](const PyFactLandmarkGraph& g, StateArg state) {
                const State s = state_arg(g.o, state, "state");
                return g.g->unachieved(*g.o.core->task, s.view()).size();
            },
            "state"_a, "The number of fact landmarks false in `state` (a landmark count without orderings).")
        .def(
            "to_arrays", [](const PyFactLandmarkGraph& g) { return graph_arrays(*g.g, *g.o.core->task); },
            "The graph as NumPy arrays (fresh copies), the form from_arrays and pickling take. Atoms are canonical "
            "ids (uint64) of the task's canonical atom layout, the same in every process and under lazy atom slots; lists are CSR: "
            "*_offsets (uint64, one entry per list and the total) and the concatenation.\n\n"
            "- fingerprint [1] (uint64): the task's content fingerprint.\n"
            "- landmarks [L]: the fact landmarks in discovery order; disjunctive_offsets, disjunctive_atoms: the "
            "disjunctive landmarks; orderings [E, 2]: (before, after).\n"
            "- lifted landmarks (K of them; empty for other graphs): lifted_predicate [K] (uint32), lifted_binding "
            "(int32 object ids, -1 for a free position) with lifted_binding_offsets, lifted_members with "
            "lifted_members_offsets, lifted_fact [K] (2**64 - 1 for a landmark with free positions), lifted_parents "
            "(uint32 positions) with lifted_parents_offsets, lifted_initially_true [K] (bool).\n"
            "- the achiever index, only if the graph has one: actions_schema [A] (int32) and actions_binding [A, W] "
            "(int32 object ids, -1 padding; W the widest schema) list the ground actions in canonical order; atoms [N] "
            "the atoms with achievers, ascending; achievers / first_achievers (uint32 action positions) per atom with "
            "their offsets [N + 1]; achieved / first_achieved / uniquely_achieved (atoms) per action with their "
            "offsets [A + 1].\n\n"
            "The index dominates the size: one row of 4 (W + 1) bytes per ground action of the grounding, and the "
            "achiever lists.")
        .def_static(
            "from_arrays",
            [](TaskArg task, ArraysArg arrays) {
                Owner o = owner_of(task);
                return make_graph(o, graph_from_arrays(*o.core->task, arrays));
            },
            "task"_a, "arrays"_a,
            "The graph of `task` from the arrays of to_arrays (NumPy, or DLPack arrays on the CPU; copied). The "
            "lifted landmark arrays and the achiever index arrays are optional, each group all or nothing. Validated: "
            "dtypes and shapes, unknown names, CSR offsets, atoms inside the task's canonical space, lifted "
            "landmarks against their predicates and bindings, actions of the task's schemas and objects in canonical "
            "order, and the per-action lists against the achievers (ValueError or TypeError otherwise). If present, "
            "fingerprint [1] (uint64) must match the task's content fingerprint (ValueError otherwise). Without it, "
            "task identity is not checked. Runs with the thread state "
            "detached once the arrays are copied.")
        .def("__reduce__",
             [](const PyFactLandmarkGraph& g) {
                 const Task& t = *g.o.core->task;
                 return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_landmark_graph"),
                                       nb::make_tuple(g.o.obj, t.fingerprint(), graph_arrays(*g.g, t)));
             })
        .def("__repr__", &graph_repr);

    m.def(
        "_restore_landmark_graph",
        [](TaskArg task, u64 fingerprint, ArraysArg arrays) {
            Owner o = owner_of(task);
            if (o.core->task->fingerprint() != fingerprint)
                throw nb::value_error("mymyr: the landmark graph is of another task (content fingerprint mismatch)");
            return make_graph(o, graph_from_arrays(*o.core->task, arrays));
        },
        "task"_a, "fingerprint"_a, "arrays"_a,
        "The pickled landmark graph of `task` (checked against its content fingerprint) from its arrays.");

    // generators -----------------------------------------------------------------------------------------------------
    m.def(
        "approximate_fact_landmarks",
        [](TaskArg task, bool include_positive_goal_facts, bool compute_greedy_necessary_orderings,
           IntArg max_disjunctive_landmark_size_in, IntArg max_disjunctive_landmark_depth_in,
           Arg<std::variant<lm::FirstAchievers, std::string>> first_achievers, bool include_derived_atoms,
           IntArg max_operators_in, IntArg max_bindings_in, FloatArg max_seconds_in, IntArg max_ground_actions_in) {
            const u64 max_disjunctive_landmark_size = int_arg<u64>(max_disjunctive_landmark_size_in, "max_disjunctive_landmark_size");
            const u64 max_disjunctive_landmark_depth =
                opt_int_arg<u64>(max_disjunctive_landmark_depth_in, "max_disjunctive_landmark_depth").value_or(~u64{0});
            const Owner o = owner_of(task);
            lm::ApproximateFactLandmarkOptions opts;
            opts.first_achievers = enum_arg<lm::FirstAchievers>(
                first_achievers, {{"hmax", lm::FirstAchievers::HMax}, {"possibly_before", lm::FirstAchievers::PossiblyBefore}},
                "first_achievers");
            opts.include_derived_atoms = include_derived_atoms;
            opts.include_positive_goal_facts = include_positive_goal_facts;
            opts.compute_greedy_necessary_orderings = compute_greedy_necessary_orderings;
            opts.max_disjunctive_landmark_size = max_disjunctive_landmark_size;
            opts.max_disjunctive_landmark_depth = max_disjunctive_landmark_depth;
            opts.budget = grounding_budget(max_operators_in, max_bindings_in, max_seconds_in, max_ground_actions_in);
            const Task& t = *o.core->task;
            std::optional<lm::FactLandmarkGraph> g;
            {
                nb::gil_scoped_release release;
                g.emplace(lm::approximate_fact_landmarks(t, opts));
            }
            return make_graph(o, std::move(*g));
        },
        "task"_a, nb::kw_only(), "include_positive_goal_facts"_a = true, "compute_greedy_necessary_orderings"_a = true,
        "max_disjunctive_landmark_size"_a = 0, "max_disjunctive_landmark_depth"_a = nb::none(),
        "first_achievers"_a = lm::FirstAchievers::HMax, "include_derived_atoms"_a = false, "max_operators"_a = 5'000'000,
        "max_bindings"_a = 100'000'000, "max_seconds"_a.none() = 60.0, "max_ground_actions"_a = nb::none(),
        "Fact landmarks over the grounded delete relaxation, with the achiever index. Grounds the task: max_operators, "
        "max_bindings, max_seconds (None: unlimited) and max_ground_actions (the rows of the achiever index; None: "
        "unlimited) bound the grounding (RuntimeError beyond). max_disjunctive_landmark_size 0: no "
        "disjunctive landmarks; max_disjunctive_landmark_depth: the layers of expanded members (None: unbounded; 0: no "
        "disjunctive landmarks). first_achievers: a FirstAchievers or 'hmax' / "
        "'possibly_before' (sound; the landmark count's). include_derived_atoms: derived goal atoms and derived "
        "preconditions are landmarks too, achieved by the axioms (the graph's achieved / unachieved see fluent atoms "
        "only).");

    m.def(
        "lifted_fact_landmarks",
        [](TaskArg task, ReachabilityArg reachability, bool include_positive_goal_facts,
           bool compute_greedy_necessary_orderings, bool use_static_filter, IntArg max_occurrence_combinations_in,
           IntArg max_disjunctive_members_in, bool reachability_filter_members,
           Arg<std::variant<lm::ReachabilityDisambiguation, std::string>> reachability_disambiguation,
           bool first_achievers_restricted, bool verify_pi_plus,
           Arg<std::variant<lm::CompleteFactLandmarks, std::string>> complete_fact_landmarks) {
            const u64 max_occurrence_combinations = int_arg<u64>(max_occurrence_combinations_in, "max_occurrence_combinations");
            const u64 max_disjunctive_members = int_arg<u64>(max_disjunctive_members_in, "max_disjunctive_members");
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
        "Lifted fact landmarks straight from the schemas: "
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
    nb::class_<rr::Statistics>(m, "ReachabilityStatistics",
                               "The counters of a RelaxedReachability fixpoint.")
        .def_ro("rules", &rr::Statistics::rules,
                "Compiled rules: one per schema, conditional effect and positive effect literal, and one per axiom.")
        .def_ro("dropped_rules", &rr::Statistics::dropped_rules,
                "Rules dropped as statically unsatisfiable.")
        .def_ro("plans", &rr::Statistics::plans,
                "Compiled matchers (one per rule and one per relation literal).")
        .def_ro("rounds", &rr::Statistics::rounds,
                "Semi-naive rounds until the fixpoint.")
        .def_ro("fluent_atoms", &rr::Statistics::fluent_atoms,
                "Reachable fluent atoms.")
        .def_ro("derived_atoms", &rr::Statistics::derived_atoms,
                "Reachable derived atoms.")
        .def_ro("bindings", &rr::Statistics::bindings,
                "Rule bindings the fixpoint emitted.")
        .def_ro("compile_ms", &rr::Statistics::compile_ms,
                "Time to compile the rules, in milliseconds.")
        .def_ro("fixpoint_ms", &rr::Statistics::fixpoint_ms,
                "Time of the fixpoint, in milliseconds.")
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
            "atom"_a,
            "Whether `atom` is reachable without the forbidden atoms: REACHABLE_WITHOUT (sound), or UNKNOWN (no "
            "information).")
        .def_prop_ro("memoized", [](PyWitnessQuery& q) {
            u64 n = 0;
            {
                nb::gil_scoped_release release;
                std::lock_guard lock(*q.m);
                n = q.q->memoized();
            }
            return n;
        },
                     "The atoms whose verdict is memoized.");

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
        .def_prop_ro("num_atoms", [](const PyReachabilityTable& t) { return t.t->num_atoms(); },
                     "The reachable fluent and derived atoms.")
        .def_prop_ro("num_fluent_atoms", [](const PyReachabilityTable& t) { return t.t->num_fluent_atoms(); },
                     "The reachable fluent atoms.")
        .def_prop_ro("num_derived_atoms", [](const PyReachabilityTable& t) { return t.t->num_derived_atoms(); },
                     "The reachable derived atoms.")
        .def_prop_ro("goal_reachable", [](const PyReachabilityTable& t) { return t.t->goal_reachable(); },
                     "Whether the static goal holds and every positive fluent and derived goal atom is reachable.")
        .def_prop_ro("rounds", [](const PyReachabilityTable& t) { return t.t->rounds(); },
                     "Semi-naive rounds until the fixpoint.")
        .def_prop_ro("has_witnesses", [](const PyReachabilityTable& t) { return t.t->has_witnesses(); },
                     "Whether the first derivation of every atom was recorded (record_witnesses=True), which "
                     "witness_query needs.")
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
            "forbidden"_a,
            "A WitnessQuery: is an atom reachable without the `forbidden` atoms?\n\nRaises:\n    ValueError: the "
            "table has no witnesses.")
        .def(
            "project",
            [](const PyReachabilityTable& t, IntArg num_variables_in, nb::typed<nb::iterable, QueryLiteralArg> literals,
               TermPairsArg equalities, TermPairsArg disequalities) {
                const u32 num_variables = int_arg<u32>(num_variables_in, "num_variables");
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
        "Exact delete-relaxed reachability over ground atoms without grounding: the unrestricted fixpoint at construction, restricted fixpoints and the "
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
        .def_prop_ro("task", [](const PyRelaxedReachability& e) -> Arg<std::variant<PyTask, PyHandle>> { return e.o.obj; },
                     "The Task (or TaskHandle) it was computed for.")
        .def_prop_ro("table",
                     [](const PyRelaxedReachability& e) {
                         return make_table(e.o, std::shared_ptr<const rr::Table>(e.rr, &e.rr->table()));
                     },
                     "The ReachabilityTable of the reachable atoms.")
        .def(
            "is_reachable",
            [](const PyRelaxedReachability& e, AtomLike atom) {
                const auto c = canonical_atom(e.o, atom, true);
                return c && e.rr->is_reachable(*c);
            },
            "atom"_a,
            "Whether `atom` (a GroundAtom or '(predicate o1 ... ok)') is reachable in the delete relaxation.")
        .def_prop_ro("goal_reachable", [](const PyRelaxedReachability& e) { return e.rr->goal_reachable(); },
                     "Whether the goal is reachable in the delete relaxation.")
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
        .def_prop_ro("stats", [](const PyRelaxedReachability& e) { return e.rr->statistics(); },
                     "The fixpoint's ReachabilityStatistics.")
        .def_prop_ro("enforce_negative_static", [](const PyRelaxedReachability& e) { return e.rr->options().enforce_negative_static; },
                     "Whether negative static literals (including (not (= ?x ?y))) were evaluated; False drops "
                     "them, as a grounded delete relaxation does.")
        .def_prop_ro("record_witnesses", [](const PyRelaxedReachability& e) { return e.rr->options().record_witnesses; },
                     "Whether the first derivation of every atom was recorded.")
        .def("__repr__", [](const PyRelaxedReachability& e) {
            const rr::Statistics& s = e.rr->statistics();
            return "RelaxedReachability(fluent_atoms=" + std::to_string(s.fluent_atoms) + ", derived_atoms=" +
                   std::to_string(s.derived_atoms) + ", goal_reachable=" + (e.rr->goal_reachable() ? "True" : "False") + ")";
        });

    // transition ordering --------------------------------------------------------------------------------------------
    nb::class_<lm::LandmarkTransitionScore>(m, "LandmarkTransitionScore",
                                            "How a transition parent --action--> child scores against the fact "
                                            "landmarks.")
        .def_ro("num_new_landmarks", &lm::LandmarkTransitionScore::num_new_landmarks,
                "Landmarks the child holds and the parent does not.")
        .def_ro("num_new_unique_landmarks", &lm::LandmarkTransitionScore::num_new_unique_landmarks,
                "New landmarks whose only achiever is the action.")
        .def_ro("unique_achiever_action", &lm::LandmarkTransitionScore::unique_achiever_action,
                "Whether the action is the only achiever of some landmark.")
        .def_ro("deletes_achieved_landmark", &lm::LandmarkTransitionScore::deletes_achieved_landmark,
                "Whether the parent holds a landmark the child does not.")
        .def_ro("num_deleted_achieved_landmarks", &lm::LandmarkTransitionScore::num_deleted_achieved_landmarks,
                "Landmarks the parent holds and the child does not.")
        .def("__eq__", [](const lm::LandmarkTransitionScore& a, const lm::LandmarkTransitionScore& b) { return a == b; })
        .def("__repr__", [](const lm::LandmarkTransitionScore& s) {
            return "LandmarkTransitionScore(new=" + std::to_string(s.num_new_landmarks) + ", new_unique=" +
                   std::to_string(s.num_new_unique_landmarks) + ", unique_achiever_action=" +
                   (s.unique_achiever_action ? "True" : "False") + ", deleted=" +
                   std::to_string(s.num_deleted_achieved_landmarks) + ")";
        });

    nb::class_<PyLandmarkOrdering>(
        m, "LandmarkTransitionOrdering",
        "The landmark transition ordering, for IW's width-1 pass: "
        "iw(task, max_width=1, transition_ordering=...). Scores each transition against the fact landmarks (more new "
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
        .def_prop_ro("graph", [](const PyLandmarkOrdering& x) { return x.graph; },
                     "The FactLandmarkGraph it scores against.")
        .def(
            "score",
            [](const PyLandmarkOrdering& x, StateArg parent, ActionLike action, StateArg child) {
                const State p = state_arg(x.graph.o, parent, "parent");
                const State c = state_arg(x.graph.o, child, "child");
                const Action a = action_of(x.graph.o, action);
                return x.t->score(p.view(), a.label(), c.view());
            },
            "parent"_a, "action"_a, "child"_a,
            "The LandmarkTransitionScore of the transition parent --action--> child.")
        .def(
            "prefer",
            [](const PyLandmarkOrdering& x, const lm::LandmarkTransitionScore& a, const lm::LandmarkTransitionScore& b) {
                return x.t->prefer(a, b);
            },
            "a"_a, "b"_a, "Whether a transition scored `a` comes strictly before one scored `b`.")
        .def_prop_ro("prefer_new_landmarks", [](const PyLandmarkOrdering& x) { return x.t->options().prefer_new_landmarks; },
                     "Whether more new landmarks rank first.")
        .def_prop_ro("prefer_unique_achievers", [](const PyLandmarkOrdering& x) { return x.t->options().prefer_unique_achievers; },
                     "Whether more new uniquely achieved landmarks rank first.")
        .def_prop_ro("prefer_landmark_actions_when_deleting",
                     [](const PyLandmarkOrdering& x) { return x.t->options().prefer_landmark_actions_when_deleting; },
                     "Whether an action that is the only achiever of some landmark ranks first.")
        .def_prop_ro("prefer_fewer_deleted_landmarks",
                     [](const PyLandmarkOrdering& x) { return x.t->options().prefer_fewer_deleted_landmarks; },
                     "Whether fewer deleted landmarks rank first.");
}
}  // namespace mymyr::python
