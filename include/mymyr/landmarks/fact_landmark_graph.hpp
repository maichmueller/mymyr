#pragma once
// FactLandmarkGraph: an immutable value holding the fact landmarks of a task. Produced by landmarks::approximate_fact_landmarks (grounded, with an
// achiever index) and landmarks::lifted_fact_landmarks (no grounding: lifted landmarks instead of achievers); read by
// LIW (landmark novelty coordinates) and by search::LandmarkTransitionOrdering.
//
//   - Atoms are positive fluent ground atoms named by their canonical id (task/atom_index.hpp): portable across
//     processes and threads, independent of the order in which lazy slots were assigned. slots() resolves them to
//     state bit positions of a task (interning a slot on first touch, which is thread-safe).
//   - Fact landmarks: atoms every plan makes true at some point (for a sound generator: approximate_fact_landmarks with
//     FirstAchievers::PossiblyBefore; the HMax first achievers are not sound in general). Kept in discovery order,
//     without duplicates.
//   - Disjunctive landmarks: sets of which every plan makes at least one member true. Each set is ascending and
//     duplicate-free, no set contains a fact landmark (the fact subsumes it), no two sets are equal, and the list is
//     sorted. Deliberately separate from the fact landmarks: a member is not individually mandatory.
//   - Greedy-necessary orderings: edges (before, after) between fact landmarks, ascending, without duplicates.
//   - Achiever index (grounded graphs only): ground actions by label and, per atom of the grounded relaxation,
//     the achievers (every ground action adding the atom) and the first achievers (those adding it at its h_max cost
//     in the landmark generator's relaxation); per action the landmarks it achieves, first-achieves and uniquely
//     achieves. A graph without it answers achiever queries with std::logic_error ("this action achieves nothing" and
//     "this graph cannot tell" are different answers).
//   - Lifted landmarks (lifted graphs only): the intensional form, a partially ground atom with its ground members and
//     the positions of the landmarks it was back-chained from.
//
// Header-only on purpose: the IW family (LIW) consumes it without linking the landmark generators. Later changes to
// this header are appends.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::landmarks
{
/// Marks a free position in a lifted landmark's binding.
inline constexpr u32 k_free = ~u32{0};
/// "No atom" (a lifted landmark that is not fully bound, an unknown atom).
inline constexpr CanonicalAtom k_no_atom = ~CanonicalAtom{0};
/// "No action" in achiever queries.
inline constexpr u32 k_no_action = ~u32{0};

/// The intensional form of a landmark: `predicate(binding)` where a free position is
/// k_free. Every plan makes one of `members` true; a fully bound landmark is a fact landmark whose only member is its
/// own atom.
struct LiftedLandmark
{
    PredicateId predicate;
    std::vector<u32> binding;             // per position: an object id or k_free
    std::vector<CanonicalAtom> members;   // ascending
    CanonicalAtom fact = k_no_atom;       // the atom when fully bound
    std::vector<u32> parents;             // positions in FactLandmarkGraph::lifted() this one was back-chained from
    bool initially_true = false;          // some member holds in the initial state (recorded, never expanded)

    [[nodiscard]] bool is_fact() const noexcept { return fact != k_no_atom; }
};

/// The grounded achiever index. Actions are ground action labels; the u32 action ids below index `actions`.
struct AchieverIndex
{
    std::vector<Action> actions;  // ascending (canonical order)

    // per atom of the grounded relaxation: ascending canonical ids with CSR lists of action ids (ascending)
    std::vector<CanonicalAtom> atoms;
    std::vector<u32> achievers_begin{0}, achievers;              // atoms.size() + 1
    std::vector<u32> first_achievers_begin{0}, first_achievers;  // atoms.size() + 1

    // per action: the landmark atoms it achieves / first-achieves / uniquely achieves (in landmark order)
    std::vector<u32> achieved_begin{0}, first_achieved_begin{0}, uniquely_achieved_begin{0};  // actions.size() + 1
    std::vector<CanonicalAtom> achieved, first_achieved, uniquely_achieved;

    /// Id of a ground action, or k_no_action.
    [[nodiscard]] u32 find(const ActionLabel& a) const
    {
        const auto it = std::lower_bound(actions.begin(), actions.end(), a,
                                         [](const Action& x, const ActionLabel& y) { return less(x.label(), y); });
        if (it == actions.end() || less(a, it->label()))
            return k_no_action;
        return static_cast<u32>(it - actions.begin());
    }
    /// Position of an atom in `atoms`, or k_no_action.
    [[nodiscard]] u32 atom_position(CanonicalAtom c) const
    {
        const auto it = std::lower_bound(atoms.begin(), atoms.end(), c);
        return (it == atoms.end() || *it != c) ? k_no_action : static_cast<u32>(it - atoms.begin());
    }
    [[nodiscard]] std::span<const u32> achievers_of(CanonicalAtom c) const { return csr(achievers_begin, achievers, atom_position(c)); }
    [[nodiscard]] std::span<const u32> first_achievers_of(CanonicalAtom c) const
    {
        return csr(first_achievers_begin, first_achievers, atom_position(c));
    }
    [[nodiscard]] std::span<const CanonicalAtom> achieved_by(u32 action) const { return csr(achieved_begin, achieved, action); }
    [[nodiscard]] std::span<const CanonicalAtom> first_achieved_by(u32 action) const
    {
        return csr(first_achieved_begin, first_achieved, action);
    }
    [[nodiscard]] std::span<const CanonicalAtom> uniquely_achieved_by(u32 action) const
    {
        return csr(uniquely_achieved_begin, uniquely_achieved, action);
    }

    [[nodiscard]] static bool less(const ActionLabel& a, const ActionLabel& b)
    {
        if (a.schema != b.schema)
            return a.schema < b.schema;
        return std::lexicographical_compare(a.binding.begin(), a.binding.end(), b.binding.begin(), b.binding.end());
    }

private:
    template<class T>
    [[nodiscard]] static std::span<const T> csr(const std::vector<u32>& begin, const std::vector<T>& v, u32 i)
    {
        if (i == k_no_action || i + 1 >= begin.size())
            return {};
        return {v.data() + begin[i], begin[i + 1] - begin[i]};
    }
};

class FactLandmarkGraph
{
public:
    FactLandmarkGraph() = default;

    /// Normalizes what the caller hands over, fact landmarks
    /// deduplicated in order; every disjunctive set sorted and deduplicated, and empty sets, sets containing a fact
    /// landmark and duplicate sets dropped; orderings sorted and deduplicated.
    [[nodiscard]] static FactLandmarkGraph create(std::vector<CanonicalAtom> landmarks,
                                                  std::vector<std::vector<CanonicalAtom>> disjunctive = {},
                                                  std::vector<std::pair<CanonicalAtom, CanonicalAtom>> orderings = {},
                                                  std::vector<LiftedLandmark> lifted = {},
                                                  std::optional<AchieverIndex> achievers = std::nullopt)
    {
        FactLandmarkGraph g;
        for (CanonicalAtom c : landmarks)
        {
            const auto it = std::lower_bound(g.m_sorted.begin(), g.m_sorted.end(), c);
            if (it != g.m_sorted.end() && *it == c)
                continue;
            g.m_sorted.insert(it, c);
            g.m_landmarks.push_back(c);
        }
        for (auto& members : disjunctive)
        {
            std::sort(members.begin(), members.end());
            members.erase(std::unique(members.begin(), members.end()), members.end());
        }
        std::sort(disjunctive.begin(), disjunctive.end());
        disjunctive.erase(std::unique(disjunctive.begin(), disjunctive.end()), disjunctive.end());
        std::erase_if(disjunctive, [&](const std::vector<CanonicalAtom>& members)
                      { return members.empty() || std::any_of(members.begin(), members.end(), [&](CanonicalAtom m) { return g.is_landmark(m); }); });
        g.m_disjunctive = std::move(disjunctive);
        std::sort(orderings.begin(), orderings.end());
        orderings.erase(std::unique(orderings.begin(), orderings.end()), orderings.end());
        g.m_orderings = std::move(orderings);
        g.m_by_after = g.m_orderings;
        std::sort(g.m_by_after.begin(), g.m_by_after.end(),
                  [](const auto& a, const auto& b) { return a.second != b.second ? a.second < b.second : a.first < b.first; });
        g.m_lifted = std::move(lifted);
        g.m_achievers = std::move(achievers);
        return g;
    }

    // ------------------------------------------------------------------------------------ fact landmarks
    /// Fact landmarks in discovery order.
    [[nodiscard]] std::span<const CanonicalAtom> landmarks() const noexcept { return m_landmarks; }
    [[nodiscard]] usize size() const noexcept { return m_landmarks.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_landmarks.empty() && m_disjunctive.empty(); }
    [[nodiscard]] bool is_landmark(CanonicalAtom c) const { return std::binary_search(m_sorted.begin(), m_sorted.end(), c); }
    /// Fact landmarks, ascending.
    [[nodiscard]] std::span<const CanonicalAtom> sorted_landmarks() const noexcept { return m_sorted; }

    // ------------------------------------------------------------------------------------ disjunctive landmarks
    [[nodiscard]] const std::vector<std::vector<CanonicalAtom>>& disjunctive() const noexcept { return m_disjunctive; }
    /// Every atom of some disjunctive landmark, ascending and duplicate-free.
    [[nodiscard]] std::vector<CanonicalAtom> disjunctive_atoms() const
    {
        std::vector<CanonicalAtom> out;
        for (const auto& members : m_disjunctive)
            out.insert(out.end(), members.begin(), members.end());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // ------------------------------------------------------------------------------------ orderings
    /// Greedy-necessary orderings (before, after), ascending.
    [[nodiscard]] std::span<const std::pair<CanonicalAtom, CanonicalAtom>> orderings() const noexcept { return m_orderings; }
    /// Landmarks ordered directly before `c` (ascending); empty for an atom the graph does not know.
    [[nodiscard]] std::vector<CanonicalAtom> predecessors(CanonicalAtom c) const
    {
        std::vector<CanonicalAtom> out;
        auto it = std::lower_bound(m_by_after.begin(), m_by_after.end(), c, [](const auto& e, CanonicalAtom x) { return e.second < x; });
        for (; it != m_by_after.end() && it->second == c; ++it)
            out.push_back(it->first);
        return out;
    }
    /// Landmarks ordered directly after `c` (ascending).
    [[nodiscard]] std::vector<CanonicalAtom> successors(CanonicalAtom c) const
    {
        std::vector<CanonicalAtom> out;
        auto it = std::lower_bound(m_orderings.begin(), m_orderings.end(), c, [](const auto& e, CanonicalAtom x) { return e.first < x; });
        for (; it != m_orderings.end() && it->first == c; ++it)
            out.push_back(it->second);
        return out;
    }

    // ------------------------------------------------------------------------------------ achievers
    [[nodiscard]] bool has_achiever_index() const noexcept { return m_achievers.has_value(); }
    /// Throws std::logic_error for a graph built without grounding.
    [[nodiscard]] const AchieverIndex& achievers() const
    {
        if (!m_achievers)
            throw std::logic_error("mymyr: landmark graph carries no achiever index (built without grounding)");
        return *m_achievers;
    }
    /// The unique achiever (action id) of a landmark atom, or k_no_action.
    [[nodiscard]] u32 unique_achiever(CanonicalAtom c) const
    {
        const auto a = achievers().achievers_of(c);
        return a.size() == 1 ? a[0] : k_no_action;
    }
    [[nodiscard]] bool is_landmark_achiever(u32 action) const { return !achievers().achieved_by(action).empty(); }
    [[nodiscard]] bool is_first_landmark_achiever(u32 action) const { return !achievers().first_achieved_by(action).empty(); }
    [[nodiscard]] bool is_unique_landmark_achiever(u32 action) const { return !achievers().uniquely_achieved_by(action).empty(); }

    // ------------------------------------------------------------------------------------ lifted landmarks
    /// The intensional landmarks in discovery order (empty for a grounded graph).
    [[nodiscard]] const std::vector<LiftedLandmark>& lifted() const noexcept { return m_lifted; }

    // ------------------------------------------------------------------------------------ against a task's states
    /// Slots of `atoms` in `task` (assigned on first touch under lazy slots; thread-safe). An atom outside the task's
    /// canonical space, or without a slot (no state holds it), maps to SlotId::invalid().
    [[nodiscard]] static std::vector<SlotId> slots(const Task& task, std::span<const CanonicalAtom> atoms)
    {
        std::vector<SlotId> out;
        out.reserve(atoms.size());
        const AtomIndex& ix = task.atoms();
        for (CanonicalAtom c : atoms)
        {
            const u32 s = c < ix.layout().fluent_count ? ix.slot_if_any(c) : AtomIndex::k_empty;
            out.push_back(s == AtomIndex::k_empty ? SlotId::invalid() : SlotId{s});
        }
        return out;
    }
    /// Fact landmarks true in s (discovery order).
    [[nodiscard]] std::vector<CanonicalAtom> achieved(const Task& task, StateView s) const { return filter(task, s, true); }
    /// Fact landmarks false in s (discovery order).
    [[nodiscard]] std::vector<CanonicalAtom> unachieved(const Task& task, StateView s) const { return filter(task, s, false); }

private:
    [[nodiscard]] std::vector<CanonicalAtom> filter(const Task& task, StateView s, bool want) const
    {
        std::vector<CanonicalAtom> out;
        const AtomIndex& ix = task.atoms();
        for (CanonicalAtom c : m_landmarks)
        {
            const u32 slot = c < ix.layout().fluent_count ? ix.find(c) : AtomIndex::k_empty;
            const bool t = slot != AtomIndex::k_empty && s.contains(SlotId{slot});
            if (t == want)
                out.push_back(c);
        }
        return out;
    }

    std::vector<CanonicalAtom> m_landmarks, m_sorted;
    std::vector<std::vector<CanonicalAtom>> m_disjunctive;
    std::vector<std::pair<CanonicalAtom, CanonicalAtom>> m_orderings, m_by_after;
    std::vector<LiftedLandmark> m_lifted;
    std::optional<AchieverIndex> m_achievers;
};

/// `pred(a, b, ...)` of a canonical atom of `task` (fluent or derived); "?" for an id outside the canonical space.
[[nodiscard]] inline std::string format_atom(const Task& task, CanonicalAtom c)
{
    const CanonicalLayout& L = task.atoms().layout();
    if (c >= L.total)
        return "?";
    std::vector<u32> args(std::max<u32>(1, L.max_arity));
    const u32 p = L.decode(c, args.data());
    const formalism::TaskData& t = task.data();
    std::string s(t.str(t.predicates[p].name));
    s += '(';
    for (u32 i = 0; i < L.arity[p]; ++i)
    {
        if (i)
            s += ", ";
        s += t.str(t.objects[args[i]].name);
    }
    s += ')';
    return s;
}

/// `pred(a, ?, ...)` of a lifted landmark.
[[nodiscard]] inline std::string format_lifted(const Task& task, const LiftedLandmark& l)
{
    const formalism::TaskData& t = task.data();
    std::string s(t.str(t.predicates[l.predicate.v].name));
    s += '(';
    for (usize i = 0; i < l.binding.size(); ++i)
    {
        if (i)
            s += ", ";
        s += l.binding[i] == k_free ? std::string("?") : std::string(t.str(t.objects[l.binding[i]].name));
    }
    s += ')';
    return s;
}

/// Checks that `g` is a landmark graph of `task`, for graphs loaded from storage. Throws std::invalid_argument naming
/// the first violation of:
///   - every atom (fact landmarks, disjunctive members, ordering endpoints, lifted members and facts, index atoms and
///     per-action lists) lies in the task's canonical space;
///   - a lifted landmark names a fluent or derived predicate and binds each of its positions to an object or k_free;
///     its members are strictly ascending atoms of that predicate agreeing with the binding, its fact is k_no_atom or
///     the atom of a binding without free positions, and its parents are positions in lifted();
///   - the achiever index lists ground actions of the task's schemas and objects, strictly ascending; its atoms are
///     strictly ascending; its CSR offsets start at 0, never decrease and end at the list sizes; each atom's achievers
///     and first achievers are strictly ascending action ids of the index, the first achievers among the achievers;
///     and the per-action lists are those the achievers, the first achievers and the fact landmarks determine.
/// Linear in the size of the graph.
inline void check(const Task& task, const FactLandmarkGraph& g)
{
    const CanonicalLayout& L = task.atoms().layout();
    const formalism::TaskData& t = task.data();
    auto fail = [](const std::string& what) { throw std::invalid_argument("mymyr: invalid landmark graph: " + what); };
    auto atom = [&](CanonicalAtom c, const char* what)
    {
        if (c >= L.total)
            fail(std::string(what) + " " + std::to_string(c) + " lies outside the task's canonical space");
    };
    for (CanonicalAtom c : g.landmarks())
        atom(c, "fact landmark");
    for (const auto& members : g.disjunctive())
        for (CanonicalAtom c : members)
            atom(c, "disjunctive member");
    for (const auto& [a, b] : g.orderings())
    {
        atom(a, "ordering endpoint");
        atom(b, "ordering endpoint");
    }
    std::vector<u32> args(std::max<u32>(1, L.max_arity));
    for (usize k = 0; k < g.lifted().size(); ++k)
    {
        const LiftedLandmark& l = g.lifted()[k];
        const std::string at = "lifted landmark " + std::to_string(k) + ": ";
        const u32 p = l.predicate.v;
        if (p >= t.predicates.size() || !L.has_predicate(p))
            fail(at + "its predicate is not a fluent or derived predicate of the task");
        if (l.binding.size() != L.arity[p])
            fail(at + "its binding does not have the predicate's arity");
        bool bound = true;
        for (u32 b : l.binding)
        {
            if (b != k_free && b >= L.num_objects)
                fail(at + "its binding names an object outside the task");
            bound = bound && b != k_free;
        }
        for (usize i = 0; i < l.members.size(); ++i)
        {
            atom(l.members[i], "lifted member");
            if (i > 0 && l.members[i - 1] >= l.members[i])
                fail(at + "its members are not strictly ascending");
            if (L.decode(l.members[i], args.data()) != p)
                fail(at + "a member is an atom of another predicate");
            for (u32 j = 0; j < L.arity[p]; ++j)
                if (l.binding[j] != k_free && l.binding[j] != args[j])
                    fail(at + "a member disagrees with the binding");
        }
        if (l.fact != k_no_atom)
        {
            atom(l.fact, "lifted fact");
            if (!bound || L.encode(p, l.binding.data()) != l.fact)
                fail(at + "its fact is not the atom of its binding");
        }
        for (u32 x : l.parents)
            if (x >= g.lifted().size())
                fail(at + "a parent lies outside the lifted landmarks");
    }
    if (!g.has_achiever_index())
        return;
    const AchieverIndex& ix = g.achievers();
    const usize A = ix.actions.size(), N = ix.atoms.size();
    for (usize i = 0; i < A; ++i)
    {
        const Action& a = ix.actions[i];
        if (a.schema.v >= t.schemas.size() || a.binding.size() != t.schemas[a.schema.v].arity())
            fail("achiever index action " + std::to_string(i) + " is not a ground action of the task's schemas");
        for (ObjectId o : a.binding)
            if (o.v >= L.num_objects)
                fail("achiever index action " + std::to_string(i) + " names an object outside the task");
        if (i > 0 && !AchieverIndex::less(ix.actions[i - 1].label(), a.label()))
            fail("the achiever index actions are not strictly ascending");
    }
    for (usize k = 0; k < N; ++k)
    {
        atom(ix.atoms[k], "achiever index atom");
        if (k > 0 && ix.atoms[k - 1] >= ix.atoms[k])
            fail("the achiever index atoms are not strictly ascending");
    }
    auto offsets = [&](const std::vector<u32>& begin, usize n, usize size, const char* what)
    {
        if (begin.size() != n + 1 || begin.front() != 0 || begin.back() != size)
            fail(std::string("the offsets of the ") + what + " do not span their list");
        for (usize i = 0; i < n; ++i)
            if (begin[i] > begin[i + 1])
                fail(std::string("the offsets of the ") + what + " decrease");
    };
    offsets(ix.achievers_begin, N, ix.achievers.size(), "achievers");
    offsets(ix.first_achievers_begin, N, ix.first_achievers.size(), "first achievers");
    offsets(ix.achieved_begin, A, ix.achieved.size(), "achieved landmarks");
    offsets(ix.first_achieved_begin, A, ix.first_achieved.size(), "first-achieved landmarks");
    offsets(ix.uniquely_achieved_begin, A, ix.uniquely_achieved.size(), "uniquely achieved landmarks");
    auto ascending_actions = [&](std::span<const u32> v, const char* what)
    {
        for (usize i = 0; i < v.size(); ++i)
            if (v[i] >= A || (i > 0 && v[i - 1] >= v[i]))
                fail(std::string("the ") + what + " of an atom are not strictly ascending action ids of the index");
    };
    for (usize k = 0; k < N; ++k)
    {
        const auto all = ix.achievers_of(ix.atoms[k]), first = ix.first_achievers_of(ix.atoms[k]);
        ascending_actions(all, "achievers");
        ascending_actions(first, "first achievers");
        if (!std::includes(all.begin(), all.end(), first.begin(), first.end()))
            fail("a first achiever of atom " + std::to_string(ix.atoms[k]) + " is not one of its achievers");
    }
    // per action: the fact landmarks (ascending) whose list `by` names it
    auto per_action = [&](auto&& by, const std::vector<u32>& begin, const std::vector<CanonicalAtom>& flat, const char* what)
    {
        std::vector<u32> count(A + 1, 0);
        for (CanonicalAtom c : g.sorted_landmarks())
            for (u32 x : by(c))
                ++count[x + 1];
        for (usize i = 0; i < A; ++i)
            count[i + 1] += count[i];
        bool same = count == begin;
        if (same)
        {
            std::vector<CanonicalAtom> want(flat.size());
            for (CanonicalAtom c : g.sorted_landmarks())
                for (u32 x : by(c))
                    want[count[x]++] = c;
            same = want == flat;
        }
        if (!same)
            fail(std::string("the ") + what + " landmarks per action disagree with the achievers and the fact landmarks");
    };
    per_action([&](CanonicalAtom c) { return ix.achievers_of(c); }, ix.achieved_begin, ix.achieved, "achieved");
    per_action([&](CanonicalAtom c) { return ix.first_achievers_of(c); }, ix.first_achieved_begin, ix.first_achieved, "first-achieved");
    per_action(
        [&](CanonicalAtom c)
        {
            const auto a = ix.achievers_of(c);
            return a.size() == 1 ? a : std::span<const u32>{};
        },
        ix.uniquely_achieved_begin, ix.uniquely_achieved, "uniquely achieved");
}
}  // namespace mymyr::landmarks
