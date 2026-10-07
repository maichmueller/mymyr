// mymyr.Task, TaskHandle, State, Action, Atom. See py_task.hpp for the object model.
//
// Hot-path rules for free-threaded Python:
//   - per-state calls (applicable_actions, successors, apply, is_goal) run on the calling thread's workspace of the
//     task and hold no lock; bulk calls release the thread state (nb::gil_scoped_release) while native code runs;
//   - objects created through a handle own a reference to the handle, not to the shared Task object, so threads that
//     each use task.local() never touch one reference count together.

#include "arrays.hpp"
#include "formalism_task.hpp"
#include "formalism_views.hpp"
#include "py_domain.hpp"
#include "py_task.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/successor/bindings.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace mymyr::python
{
using namespace nb::literals;

// ------------------------------------------------------------------------------------------------ core

PyTaskCore::PyTaskCore(TaskPtr t, std::shared_ptr<const formalism::TaskData> d, std::shared_ptr<const TaskSource> s)
    : task(std::move(t)), data(std::move(d)), source(std::move(s))
{
    label_width = rl::max_label_width(*task);
}

const NameIndex& PyTaskCore::names()
{
    std::call_once(m_names_once,
                   [this]
                   {
                       const formalism::TaskData& D = *data;
                       for (u32 i = 0; i < D.objects.size(); ++i)
                           m_names.objects.emplace(std::string(D.str(D.objects[i].name)), i);
                       for (u32 i = 0; i < D.predicates.size(); ++i)
                           m_names.predicates.emplace(std::string(D.str(D.predicates[i].name)), i);
                       for (u32 i = 0; i < D.schemas.size(); ++i)
                           m_names.schemas.emplace(std::string(D.str(D.schemas[i].name)), i);
                   });
    return m_names;
}

namespace
{
u64 slots_key(const Task& t) { return (static_cast<u64>(t.atoms().fluent_slots()) << 32) | t.atoms().derived_slots(); }
}  // namespace

std::shared_ptr<const rl::ArrayBundle> PyTaskCore::atom_metadata()
{
    std::lock_guard lock(m_mutex);
    const u64 key = slots_key(*task);
    if (!m_meta || m_meta_key != key)
    {
        m_meta = std::make_shared<const rl::ArrayBundle>(rl::atom_metadata(*task));
        m_meta_key = key;
    }
    return m_meta;
}

std::shared_ptr<const rl::ArrayBundle> PyTaskCore::device_arrays(u32 version)
{
    std::lock_guard lock(m_mutex);
    if (version != rl::k_device_arrays_version)
        return std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*task, version));  // throws
    const u64 key = slots_key(*task);
    if (!m_device || m_device_key != key)
    {
        m_device = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*task, version));
        m_device_key = slots_key(*task);  // after the goal's atoms got their slots
    }
    return m_device;
}

std::shared_ptr<const rl::GoalMasks> PyTaskCore::goal_masks()
{
    std::lock_guard lock(m_mutex);
    if (!m_goal)
        m_goal = std::make_shared<const rl::GoalMasks>(rl::goal_masks(*task));  // slots never move: compute once
    return m_goal;
}

// ------------------------------------------------------------------------------------------------ owners

Owner owner_of(nb::handle h)
{
    if (nb::isinstance<PyTask>(h))
        return {nb::inst_ptr<PyTask>(h)->core.get(), nb::borrow(h)};
    if (nb::isinstance<PyHandle>(h))
        return {nb::inst_ptr<PyHandle>(h)->core.get(), nb::borrow(h)};
    if (nb::isinstance<PyState>(h))
    {
        const PyState* s = nb::inst_ptr<PyState>(h);
        return {s->core, s->owner};
    }
    if (nb::isinstance<PyAction>(h))
    {
        const PyAction* a = nb::inst_ptr<PyAction>(h);
        return {a->core, a->owner};
    }
    throw nb::type_error("mymyr: expected a Task, TaskHandle, State or Action");
}

Arg<PyTask> task_object(nb::handle owner)
{
    if (nb::isinstance<PyHandle>(owner))
        return nb::inst_ptr<PyHandle>(owner)->task;
    return nb::borrow(owner);
}

Arg<PyState> make_state(const Owner& o, State&& s)
{
    return nb::cast(PyState{std::move(s), o.obj, o.core}, nb::rv_policy::move);
}

Arg<PyAction> make_label(const Owner& o, u32 schema, const i32* binding, u32 arity)
{
    PyAction a;
    a.schema = schema;
    a.binding.assign(binding, binding + arity);
    a.owner = o.obj;
    a.core = o.core;
    return nb::cast(std::move(a), nb::rv_policy::move);
}

bool is_state(nb::handle h) { return nb::isinstance<PyState>(h); }
const PyState& state_of(nb::handle h) { return *nb::inst_ptr<PyState>(h); }

/// The states of a batch for `task`. Numeric tasks: rows are [bits | slots]; arrays are [N, W + NN] (the last
/// task.numeric_words() columns are the numeric words), States are packed with their values.
StateBatch import_task_states(nb::handle obj, const Task& task)
{
    return import_rows(obj, task.words(), task.numeric_words());
}

StateBatch import_rows(nb::handle obj, u32 words, u32 NN)
{
    if (NN == 0)
        return import_states(obj, words);
    const bool seq = nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj);
    if (is_state(obj) || seq)
    {
        std::vector<const State*> states;
        if (seq)
            for (nb::handle item : nb::borrow<nb::sequence>(obj))
            {
                if (!is_state(item))
                    throw nb::type_error("mymyr: expected an array of state rows, a State, or a sequence of States");
                states.push_back(&state_of(item).s);
            }
        else
            states.push_back(&state_of(obj).s);
        u32 W = std::max<u32>(1, words);
        for (const State* s : states)
            W = std::max(W, s->size_words());
        StateBatch b;
        b.packed = std::make_shared<std::vector<u64>>(states.size() * (W + NN), 0);
        for (usize i = 0; i < states.size(); ++i)
        {
            u64* row = b.packed->data() + i * (W + NN);
            std::copy(states[i]->words().begin(), states[i]->words().end(), row);
            if (states[i]->numeric_words() > NN)
                throw nb::value_error("mymyr: a state of another task (numeric words)");
            std::copy(states[i]->numeric().begin(), states[i]->numeric().end(), row + W);
        }
        b.view = {b.packed->data(), states.size(), W, W + NN, NN};
        b.single = !seq;
        b.enc = {64, false};
        return b;
    }
    StateBatch b = import_states(obj, 0);
    if (b.view.words <= NN)
        throw nb::value_error((std::string("mymyr: states of a numeric task are rows [W + NN]: the atom words, then the ") +
                              std::to_string(NN) + " numeric words")
                                  .c_str());
    b.view.words -= NN;
    b.view.numeric_words = NN;
    return b;
}

namespace
{
// ------------------------------------------------------------------------------------------------ helpers

std::string name_of(const formalism::TaskData& D, formalism::Str s) { return std::string(D.str(s)); }

const char* atoms_name(AtomMode m) { return m == AtomMode::Frozen ? "frozen" : "lazy"; }

TaskOptions make_options(std::string_view atoms, std::string_view matching, u32 fc_free_params, u32 frozen_max_words,
                         u32 pilot_expansions)
{
    TaskOptions o;
    if (atoms == "auto")
        o.atoms = TaskOptions::Atoms::Auto;
    else if (atoms == "lazy")
        o.atoms = TaskOptions::Atoms::Lazy;
    else if (atoms == "frozen")
        o.atoms = TaskOptions::Atoms::Frozen;
    else
        throw nb::value_error("atoms must be 'auto', 'lazy' or 'frozen'");
    if (matching == "auto")
        o.matching = TaskOptions::Matching::Auto;
    else if (matching == "fixed")
        o.matching = TaskOptions::Matching::FixedOrder;
    else if (matching == "fc" || matching == "forward_checking")
        o.matching = TaskOptions::Matching::ForwardChecking;
    else
        throw nb::value_error("matching must be 'auto', 'fixed' or 'fc'");
    o.fc_auto_free_params = fc_free_params;
    o.frozen_max_words = frozen_max_words;
    o.pilot_expansions = pilot_expansions;
    return o;
}

nb::tuple options_tuple(const TaskOptions& o)
{
    const char* atoms = o.atoms == TaskOptions::Atoms::Auto ? "auto" : o.atoms == TaskOptions::Atoms::Lazy ? "lazy" : "frozen";
    const char* matching = o.matching == TaskOptions::Matching::Auto        ? "auto"
                           : o.matching == TaskOptions::Matching::FixedOrder ? "fixed"
                                                                             : "fc";
    return nb::make_tuple(atoms, matching, o.fc_auto_free_params, o.frozen_max_words, o.pilot_expansions);
}

CorePtr build_core(std::shared_ptr<const formalism::TaskData> data, std::shared_ptr<const TaskSource> source,
                   const TaskOptions& options)
{
    TaskPtr task;
    {
        nb::gil_scoped_release release;  // compiling (and the Auto rule's pilot search) can take a while
        task = Task::create(formalism::TaskData(*data), options);
    }
    return std::make_shared<PyTaskCore>(std::move(task), std::move(data), std::move(source));
}

#define MYMYR_TASK_OPTION_ARGS                                                                                         \
    nb::kw_only(), "atoms"_a = "auto", "matching"_a = "auto", "fc_free_params"_a = 4, "frozen_max_words"_a = 8,         \
        "pilot_expansions"_a = 1024

constexpr const char* k_options_doc =
    "Options: atoms ('auto': frozen slots when the dense state width is small, else lazy; 'lazy'; 'frozen'), "
    "matching ('auto', 'fixed', 'fc'), fc_free_params, frozen_max_words, pilot_expansions.";

/// A state argument: a State of this task, or state words (one row).
struct StateArg
{
    StateView view;
    StateBatch batch;  // keeps an imported array alive
};

StateArg state_arg(PyTaskCore& core, nb::handle h)
{
    StateArg a;
    if (is_state(h))
    {
        const PyState& s = state_of(h);
        if (s.core->task->uid() != core.task->uid())
            throw nb::value_error("mymyr: the state belongs to another task");
        a.view = s.s.view();
        return a;
    }
    a.batch = import_task_states(h, *core.task);
    if (a.batch.view.rows != 1)
        throw nb::value_error("mymyr: expected one state");
    const u32 limit = core.task->atoms().fluent_slots();
    const u64* w = a.batch.view.data;
    const u32 nw = a.batch.view.words;
    const u32 NN = a.batch.view.numeric_words;
    for (u32 i = limit >> 6; i < nw; ++i)
    {
        const u64 bad = i == (limit >> 6) ? w[i] & ~((u64{1} << (limit & 63)) - 1) : w[i];
        if (bad)
            throw nb::value_error("mymyr: the state words set atom slots this task has not assigned");
    }
    a.view = StateView{w, nw, NN ? w + nw : nullptr, NN};
    return a;
}

/// Parses "(name a b)" into tokens.
std::vector<std::string> tokens(std::string_view s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == '(' || c == ')' || c == ' ' || c == '\t' || c == '\n' || c == ',')
        {
            if (!cur.empty())
                out.push_back(std::move(cur));
            cur.clear();
        }
        else
            cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (!cur.empty())
        out.push_back(std::move(cur));
    return out;
}

u32 object_index(PyTaskCore& core, nb::handle h)
{
    if (nb::isinstance<nb::int_>(h))
    {
        const i64 v = nb::cast<i64>(h);
        if (v < 0 || v >= static_cast<i64>(core.data->objects.size()))
            throw nb::index_error("mymyr: object index out of range");
        return static_cast<u32>(v);
    }
    const std::string name = nb::cast<std::string>(nb::str(h));
    const auto& names = core.names().objects;
    auto it = names.find(name);
    if (it == names.end())
        throw nb::key_error(("mymyr: no object named '" + name + "'").c_str());
    return it->second;
}

/// (schema, binding) from an Action, a string "(stack a b)", or (schema, objects) with names or indices.
std::pair<u32, std::vector<u32>> action_label(PyTaskCore& core, nb::handle schema, nb::handle objects)
{
    std::pair<u32, std::vector<u32>> out;
    if (objects.is_none())
    {
        if (nb::isinstance<PyAction>(schema))
        {
            const PyAction& a = *nb::inst_ptr<PyAction>(schema);
            if (a.core->task->uid() != core.task->uid() && a.core->task->fingerprint() != core.task->fingerprint())
                throw nb::value_error("mymyr: the action belongs to another task");
            return {a.schema, a.binding};
        }
        if (nb::isinstance<nb::str>(schema))
        {
            const std::vector<std::string> t = tokens(nb::cast<std::string_view>(schema));
            if (t.empty())
                throw nb::value_error("mymyr: empty action string");
            auto it = core.names().schemas.find(t[0]);
            if (it == core.names().schemas.end())
                throw nb::key_error(("mymyr: no schema named '" + t[0] + "'").c_str());
            out.first = it->second;
            for (usize i = 1; i < t.size(); ++i)
                out.second.push_back(object_index(core, nb::str(t[i].c_str())));
        }
        else
        {
            // (schema, binding) tuple
            nb::sequence seq = nb::borrow<nb::sequence>(schema);
            if (nb::len(seq) != 2)
                throw nb::type_error("mymyr: expected an Action, '(schema o1 ... on)', or (schema, objects)");
            return action_label(core, seq[0], seq[1]);
        }
    }
    else
    {
        if (nb::isinstance<nb::int_>(schema))
            out.first = nb::cast<u32>(schema);
        else
        {
            const std::string n = nb::cast<std::string>(nb::str(schema));
            auto it = core.names().schemas.find(n);
            if (it == core.names().schemas.end())
                throw nb::key_error(("mymyr: no schema named '" + n + "'").c_str());
            out.first = it->second;
        }
        for (nb::handle o : nb::borrow<nb::iterable>(objects))
            out.second.push_back(object_index(core, o));
    }
    if (out.first >= core.data->schemas.size())
        throw nb::index_error("mymyr: schema index out of range");
    if (out.second.size() != core.data->schemas[out.first].arity())
        throw nb::value_error(("mymyr: schema '" + name_of(*core.data, core.data->schemas[out.first].name) + "' takes " +
                               std::to_string(core.data->schemas[out.first].arity()) + " objects")
                                  .c_str());
    return out;
}

Arg<PyAction> make_action(const Owner& o, u32 schema, const ObjectId* b, u32 arity)
{
    PyAction a;
    a.schema = schema;
    a.binding.resize(arity);
    for (u32 i = 0; i < arity; ++i)
        a.binding[i] = b[i].v;
    a.owner = o.obj;
    a.core = o.core;
    return nb::cast(std::move(a), nb::rv_policy::move);
}

std::string action_str(const PyAction& a)
{
    const formalism::TaskData& D = *a.core->data;
    std::string s = "(" + name_of(D, D.schemas[a.schema].name);
    for (u32 o : a.binding)
        s += " " + name_of(D, D.objects[o].name);
    return s + ")";
}

std::string atom_str(const formalism::TaskData& D, u32 pred, const std::vector<u32>& args)
{
    std::string s = "(" + name_of(D, D.predicates[pred].name);
    for (u32 o : args)
        s += " " + name_of(D, D.objects[o].name);
    return s + ")";
}

Arg<PyAtom> make_atom(const Owner& o, u32 pred, std::vector<u32> args, i64 slot)
{
    return nb::cast(PyAtom{pred, std::move(args), slot, o.obj, o.core}, nb::rv_policy::move);
}

/// (pred, args) of an Atom, "(on a b)", ("on", "a", "b"), ("on", ["a", "b"]), or a fluent slot (int).
std::pair<u32, std::vector<u32>> atom_key(PyTaskCore& core, nb::handle x)
{
    if (nb::isinstance<PyAtom>(x))
    {
        const PyAtom& a = *nb::inst_ptr<PyAtom>(x);
        return {a.pred, a.args};
    }
    if (nb::isinstance<nb::int_>(x))
    {
        const i64 s = nb::cast<i64>(x);
        if (s < 0 || s >= core.task->atoms().fluent_slots())
            throw nb::index_error("mymyr: atom slot out of range");
        const auto args = core.task->atoms().arguments(SlotId{static_cast<u32>(s)});
        return {core.task->atoms().predicate(SlotId{static_cast<u32>(s)}).v, std::vector<u32>(args.begin(), args.end())};
    }
    std::vector<std::string> t;
    std::vector<nb::handle> objs;
    std::pair<u32, std::vector<u32>> out;
    std::string pred;
    if (nb::isinstance<nb::str>(x))
    {
        t = tokens(nb::cast<std::string_view>(x));
        if (t.empty())
            throw nb::value_error("mymyr: empty atom string");
        pred = t[0];
        for (usize i = 1; i < t.size(); ++i)
            out.second.push_back(object_index(core, nb::str(t[i].c_str())));
    }
    else
    {
        nb::sequence seq = nb::borrow<nb::sequence>(x);
        const usize n = nb::len(seq);
        if (n == 0)
            throw nb::value_error("mymyr: empty atom");
        pred = nb::cast<std::string>(nb::str(seq[0]));
        if (n == 2 && !nb::isinstance<nb::str>(seq[1]) && !nb::isinstance<nb::int_>(seq[1]))
            for (nb::handle o : nb::borrow<nb::iterable>(seq[1]))
                out.second.push_back(object_index(core, o));
        else
            for (usize i = 1; i < n; ++i)
                out.second.push_back(object_index(core, seq[i]));
    }
    auto it = core.names().predicates.find(pred);
    if (it == core.names().predicates.end())
        throw nb::key_error(("mymyr: no predicate named '" + pred + "'").c_str());
    out.first = it->second;
    if (out.second.size() != core.data->predicates[out.first].arity)
        throw nb::value_error(("mymyr: predicate '" + pred + "' takes " + std::to_string(core.data->predicates[out.first].arity) +
                               " objects")
                                  .c_str());
    return out;
}

/// Truth of pred(args) in a state: fluent atoms from the words, static ones from the static relation, derived ones by
/// evaluating the axioms.
bool holds(PyTaskCore& core, StateView s, u32 pred, const std::vector<u32>& args)
{
    const Task& T = *core.task;
    const plan::Compiled& C = T.compiled();
    switch (C.kinds[pred])
    {
        case formalism::PredKind::Fluent:
        {
            std::vector<ObjectId> a(args.size());
            for (usize i = 0; i < args.size(); ++i)
                a[i] = ObjectId{args[i]};
            const SlotId slot = T.find_atom(PredicateId{pred}, a);
            return slot.valid() && s.contains(slot);
        }
        case formalism::PredKind::Static:
        {
            const plan::StaticRelation& R = C.statics[pred];
            u64 key = 0;
            for (u32 i = 0; i < args.size(); ++i)
                key += R.position_table(i, C.num_objects)[args[i]];
            return R.contains(key);
        }
        case formalism::PredKind::Derived:
        {
            const u64 cid = C.layout.encode(pred, args.data());
            const u32 slot = T.atoms().find(cid);
            if (slot == AtomIndex::k_empty)
                return false;
            Successors& succ = T.workspace().successors();
            succ.prepare(s);
            const detail::Engine& e = succ.engine();
            return bits::test(e.derived(), e.derived_words(), slot);
        }
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ per-state API

using ActionList = nb::typed<nb::list, PyAction>;
using StateList = nb::typed<nb::list, PyState>;
using SuccessorList = nb::typed<nb::list, nb::typed<nb::tuple, PyAction, PyState>>;
using AtomList = nb::typed<nb::list, PyAtom>;
/// A schema: a name or an index, or the whole action as action() takes it without objects.
using SchemaLike = Arg<std::variant<PyAction, std::string, int, nb::typed<nb::tuple, ObjectKey, ObjectKeys>>>;
using ObjectsArg = Arg<ObjectKeys>;
/// What Task.state takes: fluent atoms (a list, tuple, set or frozenset), a State, or state words.
using AtomsOrWords = Arg<std::variant<nb::typed<nb::iterable, AtomLike>, PyState, ann::ArrayLike>>;
using WordsArg = Arg<ann::ArrayLike>;
/// Numeric values: a sequence in slot order or a dict from numeric_names to values; None for a classical task.
using ValuesArg = Arg<std::variant<nb::typed<nb::sequence, float>, nb::typed<nb::dict, std::string, float>>>;

/// The values of every numeric slot from Task.state's `values`.
std::vector<f64> numeric_values_arg(PyTaskCore& core, nb::handle values)
{
    const Task& t = *core.task;
    const u32 n = t.numeric_slots();
    if (values.is_none())
    {
        if (n > 0)
            throw nb::value_error("mymyr: a state of a numeric task needs the values of its numeric slots (values=)");
        return {};
    }
    std::vector<f64> out;
    if (nb::isinstance<nb::dict>(values))
    {
        std::unordered_map<std::string, u32> slot_of;
        for (u32 i = 0; i < n; ++i)
            slot_of.emplace(t.numeric_name(i), i);
        out.assign(n, 0.0);
        std::vector<char> given(n, 0);
        for (auto [k, v] : nb::borrow<nb::dict>(values))
        {
            const std::string name = nb::cast<std::string>(k);
            const auto it = slot_of.find(name);
            if (it == slot_of.end())
                throw nb::value_error(("mymyr: no numeric slot named '" + name + "'").c_str());
            out[it->second] = nb::cast<f64>(v);
            given[it->second] = 1;
        }
        for (u32 i = 0; i < n; ++i)
            if (!given[i])
                throw nb::value_error(("mymyr: no value for the numeric slot " + t.numeric_name(i)).c_str());
        return out;
    }
    for (nb::handle v : nb::borrow<nb::iterable>(values))
        out.push_back(nb::cast<f64>(v));
    return out;
}
/// An exported array bundle (export_bundle): scalars and arrays by name.
using BundleDict = nb::typed<nb::dict, std::string, ann::Any>;

ActionList applicable_actions(const Owner& o, StateView s)
{
    Successors& succ = o.core->task->workspace().successors();
    std::vector<u32> schemas;
    std::vector<ObjectId> bindings;
    succ.prepare(s);
    succ.generate<false>(
        [&](u32 schema, const ObjectId* b, const Delta&)
        {
            schemas.push_back(schema);
            bindings.insert(bindings.end(), b, b + succ.arity(schema));
            return true;
        },
        false, true);
    nb::list out;
    usize at = 0;
    for (u32 schema : schemas)
    {
        const u32 k = succ.arity(schema);
        out.append(make_action(o, schema, bindings.data() + at, k));
        at += k;
    }
    return out;
}

/// The lazy applicable actions of a state (Task.iter_applicable_actions): the actions of one schema at a time, in
/// canonical order. Each refill prepares the state in the calling thread's workspace, so the iterator may be advanced
/// from any thread, and other calls in between do not disturb it.
struct PyApplicableIter
{
    State s;
    Owner o;
    u32 next_schema = 0;
    std::vector<u32> schemas;  // the buffered actions of the current schema
    std::vector<ObjectId> bindings;
    usize pos = 0, at = 0;
    nb::ft_mutex m;
};

Arg<PyAction> applicable_next(PyApplicableIter& it)
{
    nb::ft_lock_guard lock(it.m);
    Successors& succ = it.o.core->task->workspace().successors();
    if (it.pos == it.schemas.size())
    {
        it.schemas.clear();
        it.bindings.clear();
        it.pos = it.at = 0;
        const u32 n = succ.num_schemas();
        if (it.next_schema < n)
        {
            succ.prepare(it.s.view());
            while (it.next_schema < n && it.schemas.empty())
            {
                const u32 k = it.next_schema++;
                succ.generate<false>(
                    [&](u32 schema, const ObjectId* b, const Delta&) {
                        it.schemas.push_back(schema);
                        it.bindings.insert(it.bindings.end(), b, b + succ.arity(schema));
                        return true;
                    },
                    false, true, k, k + 1);
            }
        }
        if (it.schemas.empty())
            throw nb::stop_iteration();
    }
    const u32 schema = it.schemas[it.pos++];
    const u32 k = succ.arity(schema);
    Arg<PyAction> a = make_action(it.o, schema, it.bindings.data() + it.at, k);
    it.at += k;
    return a;
}

Arg<PyApplicableIter> iter_applicable(const Owner& o, StateView s)
{
    auto it = std::make_unique<PyApplicableIter>();
    it->s = State(s);
    it->o = o;
    return nb::cast(it.release(), nb::rv_policy::take_ownership);
}

/// The derived atoms of a state: the closure of its fluent atoms under the axioms.
AtomList derived_atoms(const Owner& o, StateView s)
{
    const Task& t = *o.core->task;
    AtomList out{nb::list()};
    if (!t.has_axioms())
        return out;
    std::vector<u32> slots;
    {
        Successors& succ = t.workspace().successors();
        succ.prepare(s);
        const detail::Engine& e = succ.engine();
        bits::for_each(e.derived(), e.derived_words(), [&](u64 b) { slots.push_back(static_cast<u32>(b)); });
    }
    const CanonicalLayout& L = t.compiled().layout;
    for (u32 slot : slots)
    {
        const u32* r = t.atoms().record(AtomKind::Derived, slot);
        out.append(make_atom(o, r[0], std::vector<u32>(r + 1, r + 1 + L.arity[r[0]]), -1));
    }
    return out;
}

/// Successor states in canonical order; with labels, (Action, State) pairs.
nb::list successors(const Owner& o, StateView s, bool labels)
{
    Successors& succ = o.core->task->workspace().successors();
    std::vector<State> states;
    std::vector<u32> schemas;
    std::vector<ObjectId> bindings;
    LineVector<u64> tmp;  // per-thread hot scratch (see LineAllocator)
    succ.prepare(s);
    succ.generate<false>(
        [&](u32 schema, const ObjectId* b, const Delta& d)
        {
            const u32 n = apply_delta(s.w, s.nw, d, tmp);
            states.emplace_back(tmp.data(), n, d.num, d.nnum);
            if (labels)
            {
                schemas.push_back(schema);
                bindings.insert(bindings.end(), b, b + succ.arity(schema));
            }
            return true;
        },
        false, true);
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(states.size()));
    if (!list)
        throw nb::python_error();
    nb::list out = nb::steal<nb::list>(list);
    usize at = 0;
    for (usize i = 0; i < states.size(); ++i)
    {
        nb::object st = make_state(o, std::move(states[i]));
        if (labels)
        {
            const u32 k = succ.arity(schemas[i]);
            nb::object a = make_action(o, schemas[i], bindings.data() + at, k);
            at += k;
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), nb::make_tuple(a, st).release().ptr());
        }
        else
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), st.release().ptr());
    }
    return out;
}

Arg<PyState> apply(const Owner& o, StateView s, nb::handle action)
{
    const auto [schema, binding] = action_label(*o.core, action, nb::none());
    const ActionLabel label{SchemaId{schema}, {reinterpret_cast<const ObjectId*>(binding.data()), binding.size()}};
    StateBuilder b;
    o.core->task->workspace().successors().apply(s, label, b);  // std::invalid_argument -> ValueError
    return make_state(o, b.build());
}

// ------------------------------------------------------------------------------------------------ binding generators

i64 py_hash(u64 h);

/// A lifted conjunctive condition of a task (successor/bindings.hpp): a schema's precondition or the goal.
struct PyCondition
{
    std::shared_ptr<const ConjunctiveCondition> c;
    nb::object owner;  // a Task or TaskHandle object: keeps `core` alive
    PyTaskCore* core = nullptr;
};

/// A ground literal of a binding (Task.ground_conjunctions).
struct PyGroundLiteral
{
    u32 pred = 0;
    bool positive = true;
    std::vector<u32> args;
    nb::object owner;
    PyTaskCore* core = nullptr;
};

/// A schema name or index.
using SchemaArg = Arg<std::variant<std::string, int>>;
/// What Task.bindings enumerates: a schema (name or index) or a ConjunctiveCondition.
using TargetArg = Arg<std::variant<std::string, int, PyCondition>>;
/// An object: a mymyr.formalism.Object, a name, or an index.
using ObjectLike = std::variant<ObjectView, std::string, int>;
/// A partial binding: {variable index or name: object}, or one entry per variable with None for the free ones.
using PartialArg = Arg<std::variant<nb::typed<nb::dict, std::variant<int, std::string>, std::optional<ObjectLike>>,
                                     nb::typed<nb::sequence, std::optional<ObjectLike>>>>;
/// A limit (None: no limit).
using LimitArg = Arg<int>;
using ObjectTuple = nb::typed<nb::tuple, ObjectView, nb::ellipsis>;
using BindingItem = Arg<std::variant<PyAction, ObjectTuple>>;
using GroundLiteralList = nb::typed<nb::list, PyGroundLiteral>;
using ConjunctionItem = Arg<nb::typed<nb::tuple, std::variant<PyAction, ObjectTuple>, GroundLiteralList, GroundLiteralList,
                                      GroundLiteralList>>;

u32 schema_index(PyTaskCore& core, nb::handle h)
{
    if (nb::isinstance<nb::int_>(h))
    {
        const i64 v = nb::cast<i64>(h);
        if (v < 0 || v >= static_cast<i64>(core.data->schemas.size()))
            throw nb::index_error("mymyr: schema index out of range");
        return static_cast<u32>(v);
    }
    const std::string n = nb::cast<std::string>(nb::str(h));
    const auto it = core.names().schemas.find(n);
    if (it == core.names().schemas.end())
        throw nb::key_error(("mymyr: no schema named '" + n + "'").c_str());
    return it->second;
}

/// A binding target: a schema, or a condition (cond set).
struct Target
{
    u32 schema = 0;
    std::shared_ptr<const ConjunctiveCondition> cond;
    u32 arity = 0;
};

Target target_arg(PyTaskCore& core, nb::handle h)
{
    if (nb::isinstance<PyCondition>(h))
    {
        const PyCondition& c = *nb::inst_ptr<PyCondition>(h);
        if (c.core->task->uid() != core.task->uid())
            throw nb::value_error("mymyr: the condition belongs to another task");
        return {0, c.c, c.c->arity()};
    }
    const u32 s = schema_index(core, h);
    return {s, nullptr, core.data->schemas[s].arity()};
}

/// The variable names of a target, without '?'.
std::vector<std::string> variable_names(PyTaskCore& core, const Target& t)
{
    std::vector<std::string> out;
    if (t.cond)
        for (const auto& v : t.cond->variables)
            out.push_back(v.name);
    else
        for (const formalism::Parameter& p : formalism::TaskData::slice(core.data->params, core.data->schemas[t.schema].params))
            out.push_back(name_of(*core.data, p.name));
    return out;
}

/// An object argument: an Object of this task's problem (by name if it comes from another parse), a name, or an index.
u32 object_value(PyTaskCore& core, nb::handle h)
{
    if (nb::isinstance<ObjectView>(h))
    {
        const ObjectView& v = *nb::inst_ptr<ObjectView>(h);
        if (v.t.get() == core.data.get())
            return v.i;
        return object_index(core, nb::str(v.d().str(v.d().objects[v.i].name).data(), v.d().str(v.d().objects[v.i].name).size()));
    }
    return object_index(core, h);
}

std::vector<std::optional<ObjectId>> partial_arg(PyTaskCore& core, const Target& t, nb::handle h)
{
    std::vector<std::optional<ObjectId>> out;
    if (h.is_none())
        return out;
    out.resize(t.arity);
    if (nb::isinstance<nb::dict>(h))
    {
        std::vector<std::string> names;
        for (auto [k, v] : nb::borrow<nb::dict>(h))
        {
            u32 var = 0;
            if (nb::isinstance<nb::int_>(k))
            {
                const i64 i = nb::cast<i64>(k);
                if (i < 0 || i >= static_cast<i64>(t.arity))
                    throw nb::index_error(("mymyr: partial: variable index " + std::to_string(i) + " out of range (arity " +
                                           std::to_string(t.arity) + ")")
                                              .c_str());
                var = static_cast<u32>(i);
            }
            else if (nb::isinstance<nb::str>(k))
            {
                if (names.empty())
                    names = variable_names(core, t);
                std::string n = nb::cast<std::string>(k);
                if (!n.empty() && n[0] == '?')
                    n.erase(0, 1);
                u32 found = 0, hits = 0;
                for (u32 i = 0; i < names.size(); ++i)
                    if (names[i] == n)
                        found = i, ++hits;
                if (hits != 1)
                    throw nb::key_error(("mymyr: partial: " + std::string(hits ? "more than one variable" : "no variable") +
                                         " named '?" + n + "'")
                                            .c_str());
                var = found;
            }
            else
                throw nb::type_error("mymyr: partial: the keys are variable indices or names");
            if (!v.is_none())
                out[var] = ObjectId{object_value(core, v)};
        }
        return out;
    }
    if (nb::isinstance<nb::str>(h) || !nb::isinstance<nb::sequence>(h))
        throw nb::type_error("mymyr: partial: expected a dict {variable: object} or a sequence with None for the free variables");
    nb::sequence seq = nb::borrow<nb::sequence>(h);
    if (nb::len(seq) != t.arity)
        throw nb::value_error(("mymyr: partial: expected " + std::to_string(t.arity) + " entries (one per variable), got " +
                               std::to_string(nb::len(seq)))
                                  .c_str());
    for (u32 i = 0; i < t.arity; ++i)
    {
        nb::object v = seq[i];
        if (!v.is_none())
            out[i] = ObjectId{object_value(core, v)};
    }
    return out;
}

u64 limit_arg(nb::handle h)
{
    if (h.is_none())
        return ~u64{0};
    const i64 v = nb::cast<i64>(h);
    if (v < 0)
        throw nb::value_error("mymyr: limit must be >= 0");
    return static_cast<u64>(v);
}

ObjectTuple object_tuple(PyTaskCore& core, const ObjectId* b, u32 n)
{
    PyObject* t = PyTuple_New(static_cast<Py_ssize_t>(n));
    if (!t)
        throw nb::python_error();
    nb::tuple out = nb::steal<nb::tuple>(t);
    for (u32 i = 0; i < n; ++i)
        PyTuple_SET_ITEM(t, static_cast<Py_ssize_t>(i), nb::cast(ObjectView{{core.data, b[i].v}}).release().ptr());
    return ObjectTuple(std::move(out));
}

/// The lazy bindings of a target in a state (Task.bindings, Task.ground_conjunctions). Chunks of bindings are
/// enumerated with the thread state released, holding no Python objects, on the calling thread's workspace; the
/// next chunk resumes after the last binding of the previous one (BindingOptions::resume_after).
struct PyBindingsIter
{
    Owner o;
    State s;
    Target t;
    std::vector<std::optional<ObjectId>> partial;
    bool ground = false;
    u64 remaining = 0;  // the limit left
    u64 chunk = 16;     // grows to k_max_chunk
    bool done = false;  // no binding after the buffered ones
    bool busy = false;  // a chunk is being enumerated (the thread state is released)
    std::vector<ObjectId> buf;   // the bindings of the chunk
    std::vector<u32> lits;       // ground: per binding, per kind (static, fluent, derived) a count, then the literals
    std::vector<usize> lit_at;   // ground: where each binding's literals start in `lits`
    std::vector<ObjectId> last;  // the last binding enumerated
    usize pos = 0, count = 0;
    nb::ft_mutex m;

    static constexpr u64 k_max_chunk = 4096;
};

struct PyGroundConjunctionsIter : PyBindingsIter
{
};

void refill(PyBindingsIter& it)
{
    it.buf.clear();
    it.lits.clear();
    it.lit_at.clear();
    it.pos = it.count = 0;
    const u64 want = std::min(it.chunk, it.remaining);
    if (want == 0)
    {
        it.done = true;
        return;
    }
    const Task& task = *it.o.core->task;
    it.busy = true;
    struct Busy
    {
        bool& b;
        ~Busy() { b = false; }
    } busy{it.busy};
    u64 n = 0;
    {
        nb::gil_scoped_release release;
        BindingOptions opt;
        opt.limit = want;
        opt.resume_after = it.last;
        const PartialBinding partial(it.partial);
        Workspace& ws = task.workspace();
        if (it.ground)
        {
            auto keep = [&](const GroundConjunction& g)
            {
                it.buf.insert(it.buf.end(), g.binding.begin(), g.binding.end());
                it.lit_at.push_back(it.lits.size());
                for (std::span<const GroundLiteralView> part : {g.static_literals, g.fluent_literals, g.derived_literals})
                {
                    it.lits.push_back(static_cast<u32>(part.size()));
                    for (const GroundLiteralView& l : part)
                    {
                        it.lits.push_back(l.predicate.v);
                        it.lits.push_back(l.positive ? 1 : 0);
                        for (ObjectId o : l.objects)
                            it.lits.push_back(o.v);
                    }
                }
            };
            n = it.t.cond ? for_each_ground_conjunction(task, ws, *it.t.cond, it.s.view(), partial, keep, opt)
                          : for_each_ground_conjunction(task, ws, SchemaId{it.t.schema}, it.s.view(), partial, keep, opt);
        }
        else
        {
            auto keep = [&](std::span<const ObjectId> b) { it.buf.insert(it.buf.end(), b.begin(), b.end()); };
            n = it.t.cond ? for_each_binding(task, ws, *it.t.cond, it.s.view(), partial, keep, opt)
                          : for_each_binding(task, ws, SchemaId{it.t.schema}, it.s.view(), partial, keep, opt);
        }
    }
    it.count = static_cast<usize>(n);
    it.remaining -= n;
    if (n < want || it.t.arity == 0)
        it.done = true;  // a condition without variables has at most one binding
    if (n > 0)
        it.last.assign(it.buf.end() - it.t.arity, it.buf.end());
    it.chunk = std::min(it.chunk * 2, PyBindingsIter::k_max_chunk);
}

GroundLiteralList ground_literals(PyBindingsIter& it, const u32*& p)
{
    const formalism::TaskData& D = *it.o.core->data;
    GroundLiteralList out{nb::list()};
    const u32 n = *p++;
    for (u32 j = 0; j < n; ++j)
    {
        PyGroundLiteral l;
        l.pred = *p++;
        l.positive = *p++ != 0;
        const u32 ar = D.predicates[l.pred].arity;
        l.args.assign(p, p + ar);
        p += ar;
        l.owner = it.o.obj;
        l.core = it.o.core;
        out.append(nb::cast(std::move(l), nb::rv_policy::move));
    }
    return out;
}

/// The next binding: an Action for a schema, a tuple of Objects for a condition; with ground literals, the tuple
/// (binding, static, fluent, derived).
nb::object bindings_next(PyBindingsIter& it)
{
    nb::ft_lock_guard lock(it.m);
    if (it.busy)
        throw std::runtime_error("mymyr: the binding iterator is being advanced by another thread");
    if (it.pos == it.count)
    {
        if (it.done)
            throw nb::stop_iteration();
        refill(it);
        if (it.count == 0)
            throw nb::stop_iteration();
    }
    const usize i = it.pos++;
    const ObjectId* b = it.buf.data() + i * it.t.arity;
    nb::object item = it.t.cond ? nb::object(object_tuple(*it.o.core, b, it.t.arity))
                                : nb::object(make_action(it.o, it.t.schema, b, it.t.arity));
    if (!it.ground)
        return item;
    const u32* p = it.lits.data() + it.lit_at[i];
    GroundLiteralList st = ground_literals(it, p);
    GroundLiteralList fl = ground_literals(it, p);
    GroundLiteralList de = ground_literals(it, p);
    return nb::make_tuple(item, st, fl, de);
}

template<class Iter>
nb::object make_bindings_iter(const Owner& o, StateView s, nb::handle target, nb::handle partial, nb::handle limit, bool ground)
{
    auto it = std::make_unique<Iter>();
    it->o = o;
    it->s = State(s);
    it->t = target_arg(*o.core, target);
    it->partial = partial_arg(*o.core, it->t, partial);
    it->remaining = limit_arg(limit);
    it->ground = ground;
    return nb::cast(it.release(), nb::rv_policy::take_ownership);
}

/// The applicable actions of one schema with some parameters fixed, in canonical order (by binding).
ActionList schema_actions(const Owner& o, StateView s, u32 schema, const std::vector<std::optional<ObjectId>>& partial)
{
    const Task& task = *o.core->task;
    std::vector<std::vector<ObjectId>> found = mymyr::bindings(task, task.workspace(), SchemaId{schema}, s, partial);
    std::ranges::sort(found, [](const std::vector<ObjectId>& a, const std::vector<ObjectId>& b)
                      { return std::ranges::lexicographical_compare(a, b, {}, &ObjectId::v, &ObjectId::v); });
    ActionList out{nb::list()};
    for (const auto& b : found)
        out.append(make_action(o, schema, b.data(), static_cast<u32>(b.size())));
    return out;
}

Arg<PyCondition> make_condition(const Owner& o, ConjunctiveCondition&& c)
{
    return nb::cast(PyCondition{std::make_shared<const ConjunctiveCondition>(std::move(c)), o.obj, o.core}, nb::rv_policy::move);
}

std::string ground_literal_str(const PyGroundLiteral& l)
{
    const std::string a = atom_str(*l.core->data, l.pred, l.args);
    return l.positive ? a : "(not " + a + ")";
}

/// Methods shared by Task and TaskHandle (the owner of what they return is `self`).
template<class C>
void bind_task_api(nb::class_<C>& cls)
{
    using Self = nb::pointer_and_handle<C>;
    auto owner = [](const Self& self) { return Owner{self.p->core.get(), nb::borrow(self.h)}; };
    cls.def_prop_ro(
           "initial_state", [owner](Self self) { return make_state(owner(self), State(self.p->core->task->initial_state())); },
           "The initial state.")
        .def(
            "applicable_actions",
            [owner](Self self, StateLike state, SchemaArg schema, PartialArg partial) {
                StateArg s = state_arg(*self.p->core, state);
                if (schema.is_none())
                {
                    if (!partial.is_none())
                        throw nb::value_error("mymyr: partial= needs schema=");
                    return applicable_actions(owner(self), s.view);
                }
                const Target t = target_arg(*self.p->core, schema);
                return schema_actions(owner(self), s.view, t.schema, partial_arg(*self.p->core, t, partial));
            },
            "state"_a, nb::kw_only(), "schema"_a = nb::none(), "partial"_a = nb::none(),
            "The applicable ground actions of a state in canonical order (schema, then binding), witness pruning off. "
            "With schema= (a name or index), the actions of that schema only; partial= then fixes some of its "
            "parameters, as in bindings().")
        .def(
            "bindings",
            [owner](Self self, TargetArg target, StateLike state, PartialArg partial, LimitArg limit) {
                StateArg s = state_arg(*self.p->core, state);
                return Arg<PyBindingsIter>(make_bindings_iter<PyBindingsIter>(owner(self), s.view, target, partial, limit, false));
            },
            "target"_a, "state"_a, "partial"_a = nb::none(), "limit"_a = nb::none(),
            "The bindings of a schema or a ConjunctiveCondition in a state, as a lazy iterator (Bindings).\n\n"
            "A schema (name or index) yields its applicable Actions (every parameter enumerated, the numeric "
            "applicability rules included); a condition (precondition(), goal_condition) yields tuples of "
            "mymyr.formalism.Object, one per variable, such that every literal, equality and numeric constraint "
            "holds in the state (derived atoms by the axioms).\n\n"
            "partial fixes variables in advance: a dict {index or name ('?x' or 'x'): object} or a sequence with one "
            "entry per variable and None for the free ones; an object is a mymyr.formalism.Object, a name or an "
            "index. A fixed object that violates the variable's type or the condition gives no binding. limit caps "
            "the number of bindings.\n\n"
            "Order: deterministic, lexicographic by object index over the free variables in an order chosen per "
            "(target, fixed variables), the same in every state and thread (applicable_actions(state, schema=...) "
            "sorts canonically instead). The iterator enumerates chunks of bindings natively with the thread state "
            "released and continues each chunk after the previous one, so stopping early saves the rest.")
        .def(
            "ground_conjunctions",
            [owner](Self self, TargetArg target, StateLike state, PartialArg partial, LimitArg limit) {
                StateArg s = state_arg(*self.p->core, state);
                return Arg<PyGroundConjunctionsIter>(
                    make_bindings_iter<PyGroundConjunctionsIter>(owner(self), s.view, target, partial, limit, true));
            },
            "target"_a, "state"_a, "partial"_a = nb::none(), "limit"_a = nb::none(),
            "The bindings of bindings() with the condition's literals grounded under each: an iterator "
            "(GroundConjunctions) of (binding, static, fluent, derived), the GroundLiterals split by predicate kind in "
            "the condition's literal order. Equalities and numeric constraints are not literals and do not appear.")
        .def(
            "precondition",
            [owner](Self self, SchemaArg schema) {
                const u32 k = schema_index(*self.p->core, schema);
                return make_condition(owner(self), ConjunctiveCondition::precondition(*self.p->core->task, SchemaId{k}));
            },
            "schema"_a,
            "The precondition of a schema (name or index) as a ConjunctiveCondition over its parameters. Its "
            "bindings are those of the precondition alone; bindings(schema) also applies the numeric effect rules.")
        .def_prop_ro(
            "goal_condition",
            [owner](Self self) { return make_condition(owner(self), ConjunctiveCondition::goal(*self.p->core->task)); },
            "The goal as a ConjunctiveCondition without variables: one (empty) binding in a goal state, none otherwise.")
        .def(
            "iter_applicable_actions",
            [owner](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return iter_applicable(owner(self), s.view);
            },
            "state"_a,
            "The applicable actions of a state as a lazy iterator (ApplicableActions): the actions of a schema are "
            "enumerated only when the iterator reaches it, so stopping early saves the remaining schemas. Same actions "
            "and order as applicable_actions.")
        .def(
            "any_applicable",
            [](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return self.p->core->task->workspace().successors().any_applicable(s.view);
            },
            "state"_a, "Whether some action is applicable in the state (stops at the first one).")
        .def(
            "is_dead_end",
            [](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return !self.p->core->task->workspace().successors().any_applicable(s.view);
            },
            "state"_a, "Whether no action is applicable in the state (not any_applicable).")
        .def(
            "derived_atoms",
            [owner](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return derived_atoms(owner(self), s.view);
            },
            "state"_a,
            "The derived atoms of a state: its fluent atoms closed under the axioms (including the derived predicates "
            "that normalization introduced). Empty for a task without axioms.")
        .def(
            "successors",
            [owner](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return SuccessorList(successors(owner(self), s.view, true));
            },
            "state"_a, "(Action, State) pairs in canonical order.")
        .def(
            "successor_states",
            [owner](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return StateList(successors(owner(self), s.view, false));
            },
            "state"_a, "Successor states in canonical order (the labels are those of applicable_actions).")
        .def(
            "apply",
            [owner](Self self, StateLike state, ActionLike action) {
                StateArg s = state_arg(*self.p->core, state);
                return apply(owner(self), s.view, action);
            },
            "state"_a, "action"_a,
            "The successor under an action: an Action, '(schema o1 ... on)', or (schema, objects). Raises ValueError if it "
            "is not applicable.")
        .def(
            "is_goal",
            [](Self self, StateLike state) {
                StateArg s = state_arg(*self.p->core, state);
                return self.p->core->task->is_goal(s.view);
            },
            "state"_a)
        .def(
            "is_applicable",
            [](Self self, StateLike state, ActionLike action) {
                StateArg s = state_arg(*self.p->core, state);
                const auto [schema, binding] = action_label(*self.p->core, action, nb::none());
                const ActionLabel label{SchemaId{schema}, {reinterpret_cast<const ObjectId*>(binding.data()), binding.size()}};
                return self.p->core->task->workspace().successors().is_applicable(s.view, label);
            },
            "state"_a, "action"_a)
        .def(
            "action",
            [owner](Self self, SchemaLike schema, ObjectsArg objects) {
                const auto [s, b] = action_label(*self.p->core, schema, objects);
                return make_action(owner(self), s, reinterpret_cast<const ObjectId*>(b.data()), static_cast<u32>(b.size()));
            },
            "schema"_a, "objects"_a = nb::none(),
            "An action label: action('stack', ['a', 'b']), action(2, [0, 1]), action('(stack a b)'). Not checked for "
            "applicability.")
        .def(
            "atom",
            [owner](Self self, AtomLike atom) {
                const auto [p, a] = atom_key(*self.p->core, atom);
                i64 slot = -1;
                if (self.p->core->task->compiled().kinds[p] == formalism::PredKind::Fluent)
                {
                    std::vector<ObjectId> args(a.size());
                    for (usize i = 0; i < a.size(); ++i)
                        args[i] = ObjectId{a[i]};
                    const SlotId s = self.p->core->task->find_atom(PredicateId{p}, args);
                    slot = s.valid() ? static_cast<i64>(s.v) : -1;
                }
                return make_atom(owner(self), p, a, slot);
            },
            "atom"_a, "A ground atom: atom('(on a b)'), atom(('on', 'a', 'b')), or atom(slot).")
        .def(
            "state",
            [owner](Self self, AtomsOrWords x, ValuesArg values) {
                PyTaskCore& core = *self.p->core;
                if (nb::isinstance<nb::list>(x) || nb::isinstance<nb::tuple>(x) || nb::isinstance<nb::set>(x) ||
                    nb::isinstance<nb::frozenset>(x))
                {
                    std::vector<u32> preds;
                    std::vector<std::vector<ObjectId>> objects;
                    for (nb::handle item : nb::borrow<nb::iterable>(x))
                    {
                        const auto [p, a] = atom_key(core, item);
                        preds.push_back(p);
                        objects.emplace_back(a.size());
                        for (usize i = 0; i < a.size(); ++i)
                            objects.back()[i] = ObjectId{a[i]};
                    }
                    std::vector<AtomArgs> atoms(preds.size());
                    for (usize i = 0; i < atoms.size(); ++i)
                        atoms[i] = AtomArgs{PredicateId{preds[i]}, objects[i]};
                    const std::vector<f64> v = numeric_values_arg(core, values);
                    return make_state(owner(self), core.task->make_state(atoms, v));  // invalid_argument -> ValueError
                }
                if (!values.is_none())
                    throw nb::value_error("mymyr: values go with atoms; state words carry their numeric words");
                StateArg s = state_arg(core, x);
                return make_state(owner(self), State(s.view));
            },
            "atoms_or_words"_a, "values"_a = nb::none(),
            "A state from its fluent atoms (strings, tuples or Atoms) and, for a numeric task, the values of its numeric "
            "slots (a sequence in the order of numeric_names, or a dict from those names to values, every slot given); or "
            "a copy of a State; or a state from state words (one [W + NN] row of an array). Derived atoms follow from the "
            "axioms (State.derived_atoms, State.holds). Raises ValueError for static or derived atoms, atoms outside the "
            "reachable domains, and missing or NaN values; OverflowError for a value an int32 slot cannot hold.")
        .def(
            "encode",
            [](Self self, StatesLike states, FrameworkArg framework) {
                StateBatch b = import_task_states(states, *self.p->core->task);
                const Framework fw = parse_framework(framework, nb::none());
                const u32 RW = b.view.words + b.view.numeric_words;  // numeric tasks: [W | NN] rows
                auto block = Block::make(b.view.rows * RW * sizeof(u64));
                auto* out = reinterpret_cast<u64*>(block->data());
                for (u64 i = 0; i < b.view.rows; ++i)
                    std::memcpy(out + i * RW, b.view.row(i), RW * sizeof(u64));
                ArraySpec spec{block, out, rl::DType::U64, {static_cast<i64>(b.view.rows), static_cast<i64>(RW)}, {}, false, true};
                return Arg<ann::Any>(export_array(std::move(spec), fw, default_words(fw)));
            },
            "states"_a, "framework"_a = nb::none(),
            "States as a fresh [N, W] word array (NumPy uint64, torch int64, JAX uint32 [N, 2W]); numeric tasks: "
            "[N, W + NN], the numeric words last.")
        .def(
            "decode",
            [owner](Self self, WordsArg words) {
                PyTaskCore& core = *self.p->core;
                StateBatch b = import_task_states(words, *core.task);
                const u32 limit = core.task->atoms().fluent_slots();
                const u32 NN = b.view.numeric_words;
                StateList out{nb::list()};
                for (u64 i = 0; i < b.view.rows; ++i)
                {
                    const u64* w = b.view.row(i);
                    for (u32 k = limit >> 6; k < b.view.words; ++k)
                        if (k == (limit >> 6) ? (w[k] & ~((u64{1} << (limit & 63)) - 1)) : w[k])
                            throw nb::value_error("mymyr: a row sets atom slots this task has not assigned");
                    out.append(make_state(owner(self), State(w, b.view.words, NN ? w + b.view.words : nullptr, NN)));
                }
                return out;
            },
            "words"_a, "States from a word array [N, W] (any framework); numeric tasks: [N, W + NN] rows.");
}

nb::object restore_task(std::string_view kind, std::string a, std::string b, std::string apath, std::string bpath,
                        nb::tuple opts, u64 fingerprint);

nb::object task_reduce(nb::handle self)
{
    const PyTaskCore& core = *nb::inst_ptr<PyTask>(self)->core;
    nb::object restore = nb::module_::import_("mymyr._pickle").attr("_restore_task");
    const nb::tuple opts = options_tuple(core.task->options());
    const u64 fp = core.task->fingerprint();
    if (core.source && core.source->kind == TaskSource::Kind::Pddl)
        return nb::make_tuple(restore, nb::make_tuple("pddl", core.source->domain, core.source->problem, core.source->domain_path,
                                                      core.source->problem_path, opts, fp));
    return nb::make_tuple(restore, nb::make_tuple("text", formalism::write_task_text(*core.data), "", "", "", opts, fp));
}

nb::object restore_task(std::string_view kind, std::string a, std::string b, std::string apath, std::string bpath,
                        nb::tuple opts, u64 fingerprint)
{
    const TaskOptions options = make_options(nb::cast<std::string_view>(opts[0]), nb::cast<std::string_view>(opts[1]),
                                             nb::cast<u32>(opts[2]), nb::cast<u32>(opts[3]), nb::cast<u32>(opts[4]));
    std::shared_ptr<const formalism::TaskData> data;
    auto source = std::make_shared<TaskSource>();
    if (kind == "pddl")
    {
#if defined(MYMYR_HAS_FRONTEND)
        nb::gil_scoped_release release;
        auto dom = frontend::Domain::from_string(a, apath);
        data = dom->instantiate_string(b, bpath);
        source->kind = TaskSource::Kind::Pddl;
        source->domain = std::move(a);
        source->problem = std::move(b);
        source->domain_path = std::move(apath);
        source->problem_path = std::move(bpath);
#else
        throw std::runtime_error("mymyr: this build has no PDDL front end; cannot unpickle a task from PDDL");
#endif
    }
    else if (kind == "text")
    {
        std::istringstream in(a);
        data = std::make_shared<const formalism::TaskData>(formalism::read_task_text(in));
        source = nullptr;
    }
    else
        throw nb::value_error("mymyr: unknown task pickle kind");
    CorePtr core = build_core(std::move(data), std::move(source), options);
    if (core->task->fingerprint() != fingerprint)
        throw nb::value_error("mymyr: the unpickled task differs from the pickled one (content fingerprint mismatch): "
                              "the PDDL or mymyr's normalization changed");
    return nb::cast(PyTask{std::move(core)}, nb::rv_policy::move);
}

nb::bytes to_bytes(const void* p, usize n) { return nb::bytes(static_cast<const char*>(p), n); }

nb::object state_reduce(const PyState& s)
{
    const Task& T = *s.core->task;
    nb::object restore = nb::module_::import_("mymyr._pickle").attr("_restore_state");
    nb::object task = task_object(s.owner);
    std::vector<u64> payload;
    std::string kind;
    if (T.atoms().mode() == AtomMode::Frozen)
    {
        kind = "w";
        payload.assign(s.s.data(), s.s.data() + s.s.size_words());
    }
    else
    {
        // lazy slots depend on the order of first touch: states cross processes as canonical ids
        kind = "c";
        bits::for_each(s.s.data(), s.s.size_words(),
                       [&](u64 slot) { payload.push_back(T.atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot))); });
    }
    if (s.s.numeric_words())
    {
        // numeric tasks: kind "wn" / "cn", payload = [count, atoms..., numeric words...] (slots in canonical order)
        kind += "n";
        payload.insert(payload.begin(), payload.size());
        payload.insert(payload.end(), s.s.numeric().begin(), s.s.numeric().end());
    }
    return nb::make_tuple(restore, nb::make_tuple(task, T.fingerprint(), kind, to_bytes(payload.data(), payload.size() * sizeof(u64))));
}

nb::object restore_state(nb::handle task, u64 fingerprint, std::string_view kind, nb::bytes payload)
{
    if (!nb::isinstance<PyTask>(task))
        throw nb::type_error("mymyr: _restore_state needs a Task");
    Owner o{nb::inst_ptr<PyTask>(task)->core.get(), nb::borrow(task)};
    const Task& T = *o.core->task;
    if (T.fingerprint() != fingerprint)
        throw nb::value_error("mymyr: the state was pickled from a different task (content fingerprint mismatch)");
    usize n = payload.size() / sizeof(u64);
    std::vector<u64> raw(n);
    std::memcpy(raw.data(), payload.c_str(), n * sizeof(u64));
    const u64 F = T.compiled().layout.fluent_count;
    std::vector<u64> num;
    if (kind == "wn" || kind == "cn")
    {
        if (n == 0 || raw[0] > n - 1 || n - 1 - raw[0] != T.numeric_words())
            throw nb::value_error("mymyr: corrupt state pickle (numeric words)");
        num.assign(raw.begin() + 1 + static_cast<std::ptrdiff_t>(raw[0]), raw.end());
        raw = std::vector<u64>(raw.begin() + 1, raw.begin() + 1 + static_cast<std::ptrdiff_t>(raw[0]));
        n = raw.size();
        kind = kind.substr(0, 1);
    }
    else if (T.numeric_words())
        throw nb::value_error("mymyr: corrupt state pickle (no numeric words for a numeric task)");
    if (kind == "w" && T.atoms().mode() == AtomMode::Frozen)
    {
        bool ok = true;  // frozen: slot = canonical id, every bit below F
        bits::for_each(raw.data(), static_cast<u32>(n), [&](u64 b) { ok = ok && b < F; });
        if (!ok)
            throw nb::value_error("mymyr: corrupt state pickle (atom id out of range)");
        return make_state(o, State(raw.data(), static_cast<u32>(n), num.data(), static_cast<u32>(num.size())));
    }
    std::vector<u64> cids;
    if (kind == "w")
        bits::for_each(raw.data(), static_cast<u32>(n), [&](u64 b) { cids.push_back(b); });  // frozen: slot = cid
    else if (kind == "c")
        cids = std::move(raw);
    else
        throw nb::value_error("mymyr: unknown state pickle kind");
    StateBuilder b;
    for (u64 c : cids)
    {
        if (c >= F)
            throw nb::value_error("mymyr: corrupt state pickle (atom id out of range)");
        b.set(SlotId{T.atoms().intern(c)});
    }
    b.numeric() = std::move(num);
    return make_state(o, b.build());
}

std::string state_str(const PyState& s, usize limit)
{
    const std::vector<std::string> atoms = s.core->task->format_atoms(s.s.view());
    std::string out;
    for (usize i = 0; i < atoms.size() && i < limit; ++i)
        out += (i ? ", " : "") + atoms[i];
    if (atoms.size() > limit)
        out += ", ... (" + std::to_string(atoms.size() - limit) + " more)";
    // numeric tasks: "(function objects)=value" per slot
    const Task& T = *s.core->task;
    for (u32 i = 0; i < T.numeric_slots() && i < limit; ++i)
    {
        std::ostringstream v;
        v << T.numeric_value(s.s.view(), i);
        out += (out.empty() ? "" : ", ") + T.numeric_name(i) + "=" + v.str();
    }
    if (T.numeric_slots() > limit)
        out += ", ... (" + std::to_string(T.numeric_slots() - limit) + " more values)";
    return out;
}

i64 py_hash(u64 h)
{
    const i64 v = static_cast<i64>(h >> 1);
    return v == -1 ? -2 : v;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ bindings

void bind_task(nb::module_& m)
{
    nb::class_<PyApplicableIter>(m, "ApplicableActions",
                                 "An iterator over the applicable actions of a state (iter_applicable_actions): lazy per "
                                 "schema (the actions of a schema are enumerated when the iterator reaches it), in "
                                 "canonical order. It may be advanced from any thread.")
        .def("__iter__", [](nb::handle self) { return Arg<PyApplicableIter>(nb::borrow(self)); })
        .def("__next__", &applicable_next);

    nb::class_<PyCondition>(m, "ConjunctiveCondition",
                            "A lifted conjunctive condition of a task: variables, literals over static, fluent and derived "
                            "predicates, equalities and numeric constraints (Task.precondition, Task.goal_condition). "
                            "Task.bindings enumerates its bindings in a state.")
        .def_prop_ro("arity", [](const PyCondition& c) { return c.c->arity(); }, "The number of variables.")
        .def_prop_ro("variables",
                     [](const PyCondition& c) {
                         std::vector<std::string> v;
                         for (u32 i = 0; i < c.c->arity(); ++i)
                             v.push_back("?" + (c.c->variables[i].name.empty() ? "x" + std::to_string(i) : c.c->variables[i].name));
                         return v;
                     },
                     "The variable names ('?x'; an unnamed variable is '?x<index>').")
        .def("__eq__",
             [](const PyCondition& a, nb::handle b) -> nb::object {
                 if (!nb::isinstance<PyCondition>(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyCondition& o = *nb::inst_ptr<PyCondition>(b);
                 return nb::bool_(a.core->task->uid() == o.core->task->uid() && *a.c == *o.c);
             })
        .def("__hash__",
             [](const PyCondition& c) {
                 return py_hash(hash::combine(c.core->task->uid(), std::hash<std::string>{}(c.c->str(*c.core->task))));
             })
        .def("__str__", [](const PyCondition& c) { return c.c->str(*c.core->task); })
        .def("__repr__", [](const PyCondition& c) { return "ConjunctiveCondition" + c.c->str(*c.core->task); });

    nb::class_<PyGroundLiteral>(m, "GroundLiteral", "A ground literal: an atom and its polarity (Task.ground_conjunctions).")
        .def_prop_ro("atom",
                     [](const PyGroundLiteral& l) {
                         i64 slot = -1;
                         if (l.core->task->compiled().kinds[l.pred] == formalism::PredKind::Fluent)
                         {
                             std::vector<ObjectId> args(l.args.size());
                             for (usize i = 0; i < args.size(); ++i)
                                 args[i] = ObjectId{l.args[i]};
                             const SlotId s = l.core->task->find_atom(PredicateId{l.pred}, args);
                             slot = s.valid() ? static_cast<i64>(s.v) : -1;
                         }
                         return make_atom(Owner{l.core, l.owner}, l.pred, l.args, slot);
                     })
        .def_prop_ro("positive", [](const PyGroundLiteral& l) { return l.positive; })
        .def("__eq__",
             [](const PyGroundLiteral& a, nb::handle b) -> nb::object {
                 if (!nb::isinstance<PyGroundLiteral>(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyGroundLiteral& o = *nb::inst_ptr<PyGroundLiteral>(b);
                 return nb::bool_(a.core->task->uid() == o.core->task->uid() && a.pred == o.pred &&
                                  a.positive == o.positive && a.args == o.args);
             })
        .def("__hash__",
             [](const PyGroundLiteral& l) {
                 u64 h = hash::combine(l.core->task->uid(), l.pred * 2ULL + (l.positive ? 1 : 0));
                 for (u32 o : l.args)
                     h = hash::combine(h, o);
                 return py_hash(h);
             })
        .def("__str__", [](const PyGroundLiteral& l) { return ground_literal_str(l); })
        .def("__repr__", [](const PyGroundLiteral& l) { return "GroundLiteral" + ground_literal_str(l); });

    nb::class_<PyBindingsIter>(m, "Bindings",
                               "An iterator over the bindings of a schema (Actions) or a condition (tuples of Objects) "
                               "in a state (Task.bindings). It may be advanced from any thread, one at a time.")
        .def("__iter__", [](nb::handle self) { return Arg<PyBindingsIter>(nb::borrow(self)); })
        .def("__next__", [](PyBindingsIter& it) { return BindingItem(bindings_next(it)); });
    nb::class_<PyGroundConjunctionsIter>(m, "GroundConjunctions",
                                         "An iterator over (binding, static, fluent, derived) ground literals "
                                         "(Task.ground_conjunctions). It may be advanced from any thread, one at a time.")
        .def("__iter__", [](nb::handle self) { return Arg<PyGroundConjunctionsIter>(nb::borrow(self)); })
        .def("__next__", [](PyGroundConjunctionsIter& it) { return ConjunctionItem(bindings_next(it)); });

    // --- Task
    nb::class_<PyTask> task(m, "Task",
                            "An immutable compiled planning task, shared by any number of threads. Per-state methods "
                            "run on the calling thread's workspace; use task.local() for a per-thread handle in hot "
                            "loops.");
    task.def(
            "__init__",
            [](PyTask* self, const FormalismTask& normalized, std::string_view atoms, std::string_view matching,
               u32 fc_free_params, u32 frozen_max_words, u32 pilot_expansions) {
                const TaskOptions o = make_options(atoms, matching, fc_free_params, frozen_max_words, pilot_expansions);
                new (self) PyTask{build_core(normalized.t, normalized.source, o)};
            },
            "normalized"_a, MYMYR_TASK_OPTION_ARGS, k_options_doc)
        .def_static(
            "from_text",
            [](const std::filesystem::path& path, std::string_view atoms, std::string_view matching, u32 fc, u32 fmw, u32 pilot) {
                const TaskOptions o = make_options(atoms, matching, fc, fmw, pilot);
                std::shared_ptr<const formalism::TaskData> data;
                {
                    nb::gil_scoped_release release;
                    data = std::make_shared<const formalism::TaskData>(formalism::read_task_text_file(path.string()));
                }
                return PyTask{build_core(std::move(data), nullptr, o)};
            },
            "path"_a, MYMYR_TASK_OPTION_ARGS,
            "A task in mymyr's normalized text format (formalism::write_task_text writes it).")
#if defined(MYMYR_HAS_FRONTEND)
        .def_static(
            "from_pddl",
            [](Arg<std::variant<PyDomain, std::filesystem::path>> domain, const std::filesystem::path& problem,
               std::string_view atoms, std::string_view matching, u32 fc, u32 fmw, u32 pilot) {
                const TaskOptions o = make_options(atoms, matching, fc, fmw, pilot);
                PyDomain dom;
                if (nb::isinstance<PyDomain>(domain))
                    dom = *nb::inst_ptr<PyDomain>(domain);
                else
                {
                    const std::filesystem::path path = nb::cast<std::filesystem::path>(domain);
                    nb::gil_scoped_release release;
                    dom = PyDomain{frontend::Domain::from_file(path), std::make_shared<const std::string>(read_file(path.string())),
                                   path.string()};
                }
                std::shared_ptr<const formalism::TaskData> data;
                auto source = std::make_shared<TaskSource>();
                {
                    nb::gil_scoped_release release;
                    source->kind = TaskSource::Kind::Pddl;
                    source->domain = dom.text ? *dom.text : std::string();
                    source->domain_path = dom.path;
                    source->problem = read_file(problem.string());
                    source->problem_path = problem.string();
                    data = dom.d->instantiate_file(problem);
                }
                return PyTask{build_core(std::move(data), std::move(source), o)};
            },
            "domain"_a, "problem"_a, MYMYR_TASK_OPTION_ARGS,
            "Parse (domain: a path or a mymyr.Domain) and instantiate the problem file.")
#endif
        .def("local", [](nb::handle self) { return PyHandle{nb::inst_ptr<PyTask>(self)->core, nb::borrow(self)}; },
             "A per-thread handle: same methods, but the states and actions it creates reference the handle instead of "
             "this shared object (no reference-count contention between threads).")
        .def_prop_ro("uid", [](const PyTask& t) { return t.core->task->uid(); }, "Process-unique id of this instance.")
        .def_prop_ro("fingerprint", [](const PyTask& t) { return t.core->task->fingerprint(); },
                     "Content hash of the normalized task (equal across processes).")
        .def_prop_ro("formalism", [](const PyTask& t) { return FormalismTask{t.core->data, t.core->source}; },
                     "The normalized task (mymyr.formalism.NormalizedTask).")
        .def_prop_ro("domain_name", [](const PyTask& t) { return t.core->data->domain_name; })
        .def_prop_ro("problem_name", [](const PyTask& t) { return t.core->data->problem_name; })
        .def_prop_ro("num_objects", [](const PyTask& t) { return t.core->task->num_objects(); })
        .def_prop_ro("num_schemas", [](const PyTask& t) { return t.core->task->num_schemas(); })
        .def_prop_ro("num_predicates", [](const PyTask& t) { return t.core->data->predicates.size(); })
        .def_prop_ro("objects", [](const PyTask& t) {
            std::vector<std::string> v;
            for (const auto& o : t.core->data->objects)
                v.push_back(name_of(*t.core->data, o.name));
            return v;
        }, "Object names by index.")
        .def_prop_ro("schemas", [](const PyTask& t) {
            std::vector<std::string> v;
            for (const auto& s : t.core->data->schemas)
                v.push_back(name_of(*t.core->data, s.name));
            return v;
        }, "Schema names by index.")
        .def_prop_ro("predicates", [](const PyTask& t) {
            std::vector<std::string> v;
            for (const auto& p : t.core->data->predicates)
                v.push_back(name_of(*t.core->data, p.name));
            return v;
        }, "Predicate names by index.")
        .def_prop_ro("words", [](const PyTask& t) { return t.core->task->words(); },
                     "Current state width W in u64 words (grows only under lazy slots).")
        .def_prop_ro("max_words", [](const PyTask& t) { return t.core->task->max_words(); })
        .def_prop_ro("num_atoms", [](const PyTask& t) { return t.core->task->atoms().fluent_slots(); },
                     "Assigned fluent atom slots.")
        .def_prop_ro("atom_mode", [](const PyTask& t) { return atoms_name(t.core->task->atoms().mode()); })
        .def_prop_ro("label_width", [](const PyTask& t) { return t.core->label_width; }, "Largest schema arity.")
        .def_prop_ro("numeric_slots", [](const PyTask& t) { return t.core->task->numeric_slots(); },
                     "Numeric fluent slots (ground fluent functions with a value; 0 for classical tasks).")
        .def_prop_ro("numeric_words", [](const PyTask& t) { return t.core->task->numeric_words(); },
                     "u64 words per state holding the numeric values (after the atom words in rl rows).")
        .def_prop_ro("numeric_storage", [](const PyTask& t) -> Arg<std::optional<std::string>> {
            if (t.core->task->numeric_slots() == 0)
                return nb::none();
            return nb::str(t.core->task->numeric_storage() == NumericStorage::I32 ? "i32" : "f64");
        }, "'i32' (two int32 per word) or 'f64' (one double per word); None without numeric slots.")
        .def_prop_ro("numeric_names", [](const PyTask& t) {
            std::vector<std::string> v;
            for (u32 i = 0; i < t.core->task->numeric_slots(); ++i)
                v.push_back(t.core->task->numeric_name(i));
            return v;
        }, "The ground function of each numeric slot, e.g. '(fuel truck1)'.")
        .def_prop_ro("has_axioms", [](const PyTask& t) { return t.core->task->has_axioms(); })
        .def_prop_ro("options", [](const PyTask& t) {
            const nb::tuple o = options_tuple(t.core->task->options());
            nb::typed<nb::dict, std::string, ann::Any> d{nb::dict()};
            d["atoms"] = o[0];
            d["matching"] = o[1];
            d["fc_free_params"] = o[2];
            d["frozen_max_words"] = o[3];
            d["pilot_expansions"] = o[4];
            return d;
        })
        .def_prop_ro("info", [](const PyTask& t) {
            const TaskInfo& i = t.core->task->info();
            nb::typed<nb::dict, std::string, ann::Any> d{nb::dict()};
            d["atom_mode"] = atoms_name(i.atom_mode);
            d["dense_fluent"] = i.dense_fluent;
            d["dense_derived"] = i.dense_derived;
            d["pilot_words"] = i.pilot_words;
            d["build_seconds"] = i.build_s;
            return d;
        })
        .def(
            "atom_metadata",
            [](const PyTask& t, FrameworkArg framework) {
                const Framework fw = parse_framework(framework, nb::none());
                std::shared_ptr<const rl::ArrayBundle> b;
                {
                    nb::gil_scoped_release release;
                    b = t.core->atom_metadata();
                }
                return nb::borrow<BundleDict>(export_bundle(b, fw, default_words(fw)));
            },
            "framework"_a = nb::none(),
            "Atom metadata for encoders (mifrost), zero-copy views of a cached snapshot: atom_pred, atom_args (CSR with "
            "atom_args_offsets), atom_args_padded, atom_cid, pred_slots / pred_slot_offsets, pred_arity, pred_kind, "
            "derived_* and static_* atoms, and scalars (num_atoms, num_objects, ...). Under lazy slots the snapshot "
            "covers the slots assigned so far.")
        .def(
            "device_arrays",
            [](const PyTask& t, u32 version, FrameworkArg framework) {
                const Framework fw = parse_framework(framework, nb::none());
                std::shared_ptr<const rl::ArrayBundle> b;
                {
                    nb::gil_scoped_release release;
                    b = t.core->device_arrays(version);
                }
                return nb::borrow<BundleDict>(export_bundle(b, fw, default_words(fw)));
            },
            "version"_a = rl::k_device_arrays_version, "framework"_a = nb::none(),
            "The flat, versioned POD export of the task: scalars plus arrays (a dict: a JAX pytree or torch "
            "tensors). Version 1 = atom metadata + init + goal masks + canonical layout + schema table; version 2 (the "
            "default) adds section 'plan': the matcher tables of the lifted successor generator as offsets (plan_*, "
            "section_core, section_plan), which mymyr.cuda.DeviceTask uploads.")
        .def(
            "goal_masks",
            [](const PyTask& t, FrameworkArg framework) {
                const Framework fw = parse_framework(framework, nb::none());
                auto g = t.core->goal_masks();
                if (g->uses_derived())
                    throw nb::value_error("mymyr: the goal mentions derived predicates; a mask test cannot decide it "
                                          "(use rl.is_goal)");
                auto one = [&](const std::vector<u64>& v) {
                    ArraySpec s{std::shared_ptr<const void>(g, v.data()), v.data(), rl::DType::U64,
                                {static_cast<i64>(v.size())}, {}, true, true};
                    return export_array(std::move(s), fw, default_words(fw));
                };
                return nb::typed<nb::tuple, ann::Any, ann::Any>(nb::make_tuple(one(g->pos), one(g->neg)));
            },
            "framework"_a = nb::none(),
            "(gpos, gneg): the goal's positive and negative fluent atoms as state words (zero-copy, read-only).")
        .def("__reduce__", &task_reduce)
        .def("__repr__", [](const PyTask& t) {
            return "Task(" + t.core->data->domain_name + "/" + t.core->data->problem_name + ", " +
                   std::to_string(t.core->task->num_objects()) + " objects, " + std::to_string(t.core->task->num_schemas()) +
                   " schemas, " + atoms_name(t.core->task->atoms().mode()) + " slots)";
        });
    bind_task_api(task);

    // --- TaskHandle
    nb::class_<PyHandle> handle(m, "TaskHandle",
                                "A per-thread handle of a Task (task.local()): the Task's per-state methods, without "
                                "sharing a Python object between threads.");
    handle.def_prop_ro("task", [](const PyHandle& h) { return Arg<PyTask>(h.task); })
        .def("__reduce__", [](const PyHandle& h) {
            return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_handle"), nb::make_tuple(h.task));
        })
        .def("__repr__", [](const PyHandle& h) { return "TaskHandle(" + h.core->data->problem_name + ")"; });
    bind_task_api(handle);

    // --- State
    nb::class_<PyState>(m, "State",
                        "A state: an immutable value (its fluent atoms as u64 words) plus the task it belongs to. "
                        "Hashable; equal to the states of the same task instance with the same atoms; picklable.")
        .def("__eq__",
             [](const PyState& a, nb::handle b) -> nb::object {
                 if (!is_state(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyState& o = state_of(b);
                 return nb::bool_(a.core->task->uid() == o.core->task->uid() && a.s == o.s);
             })
        .def("__ne__",
             [](const PyState& a, nb::handle b) -> nb::object {
                 if (!is_state(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyState& o = state_of(b);
                 return nb::bool_(!(a.core->task->uid() == o.core->task->uid() && a.s == o.s));
             })
        .def("__hash__", [](const PyState& s) { return py_hash(hash::combine(s.core->task->uid(), s.s.hash())); })
        .def("__len__", [](const PyState& s) { return s.s.count(); }, "Number of true fluent atoms.")
        .def_prop_ro("words",
                     [](nb::handle self) {
                         const PyState& s = state_of(self);
                         const size_t shape[1] = {s.s.size_words()};
                         return nb::ndarray<nb::numpy, const u64, nb::ndim<1>>(s.s.data(), 1, shape, self);
                     },
                     "The trimmed state words as a read-only NumPy uint64 view.")
        .def_prop_ro("num_words", [](const PyState& s) { return s.s.size_words(); })
        .def("numeric_values",
             [](const PyState& s) {
                 const std::vector<f64> v = s.core->task->numeric_values(s.s.view());
                 auto* data = new f64[std::max<usize>(1, v.size())];
                 std::copy(v.begin(), v.end(), data);
                 nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<f64*>(p); });
                 const size_t shape[1] = {v.size()};
                 return nb::ndarray<nb::numpy, f64, nb::ndim<1>>(data, 1, shape, owner);
             },
             "The values of the task's numeric slots (task.numeric_names) as a new float64 array (empty for classical "
             "tasks).")
        .def_prop_ro("numeric_words",
                     [](nb::handle self) {
                         const PyState& s = state_of(self);
                         const size_t shape[1] = {s.s.numeric_words()};
                         return nb::ndarray<nb::numpy, const u64, nb::ndim<1>>(s.s.numeric().data(), 1, shape, self);
                     },
                     "The numeric words as a read-only NumPy uint64 view (task.numeric_storage encoding; empty for "
                     "classical tasks).")
        .def_prop_ro("task", [](const PyState& s) { return task_object(s.owner); })
        .def_prop_ro("owner", [](const PyState& s) { return Arg<std::variant<PyTask, PyHandle>>(s.owner); },
                     "The Task or TaskHandle that created it.")
        .def("canonical_hash", [](const PyState& s) { return s.core->task->canonical_hash(s.s.view()); },
             "Hash over the canonical ids of the true atoms: equal for equal states of tasks with equal content, under "
             "any slot numbering.")
        .def("atom_slots",
             [](const PyState& s) {
                 std::vector<u32> slots;
                 bits::for_each(s.s.data(), s.s.size_words(), [&](u64 b) { slots.push_back(static_cast<u32>(b)); });
                 return slots;
             },
             "Slots of the true fluent atoms (ascending).")
        .def("atoms",
             [](const PyState& s) {
                 const Owner o{s.core, s.owner};
                 AtomList out{nb::list()};
                 bits::for_each(s.s.data(), s.s.size_words(),
                                [&](u64 b) {
                                    const SlotId slot{static_cast<u32>(b)};
                                    const auto args = s.core->task->atoms().arguments(slot);
                                    out.append(make_atom(o, s.core->task->atoms().predicate(slot).v,
                                                         std::vector<u32>(args.begin(), args.end()), static_cast<i64>(b)));
                                });
                 return out;
             },
             "The true fluent atoms (slot order).")
        .def("holds",
             [](const PyState& s, AtomLike atom) {
                 const auto [p, a] = atom_key(*s.core, atom);
                 return holds(*s.core, s.s.view(), p, a);
             },
             "atom"_a,
             "Truth of a ground atom (an Atom, '(on a b)', ('on', 'a', 'b') or a slot): fluent atoms from the state, "
             "static ones from the task, derived ones by evaluating the axioms.")
        .def("is_goal", [](const PyState& s) { return s.core->task->is_goal(s.s.view()); })
        .def("applicable_actions", [](const PyState& s) { return applicable_actions(Owner{s.core, s.owner}, s.s.view()); })
        .def("iter_applicable_actions",
             [](const PyState& s) { return iter_applicable(Owner{s.core, s.owner}, s.s.view()); },
             "The applicable actions as a lazy iterator (Task.iter_applicable_actions).")
        .def("any_applicable",
             [](const PyState& s) { return s.core->task->workspace().successors().any_applicable(s.s.view()); },
             "Whether some action is applicable (stops at the first one).")
        .def("is_dead_end",
             [](const PyState& s) { return !s.core->task->workspace().successors().any_applicable(s.s.view()); },
             "Whether no action is applicable (not any_applicable).")
        .def("derived_atoms", [](const PyState& s) { return derived_atoms(Owner{s.core, s.owner}, s.s.view()); },
             "The derived atoms: the fluent atoms closed under the axioms (Task.derived_atoms).")
        .def("successors",
             [](const PyState& s) { return SuccessorList(successors(Owner{s.core, s.owner}, s.s.view(), true)); })
        .def("successor_states",
             [](const PyState& s) { return StateList(successors(Owner{s.core, s.owner}, s.s.view(), false)); })
        .def("apply",
             [](const PyState& s, ActionLike action) { return apply(Owner{s.core, s.owner}, s.s.view(), action); },
             "action"_a)
        .def("__reduce__", &state_reduce)
        .def("__str__", [](const PyState& s) { return "{" + state_str(s, ~usize{0}) + "}"; })
        .def("__repr__", [](const PyState& s) { return "State({" + state_str(s, 8) + "})"; });

    // --- Action
    nb::class_<PyAction>(m, "Action",
                         "An action label (schema, binding): the schema's full parameter list bound to objects. "
                         "Equal to the labels of the same task with the same schema and binding.")
        .def_prop_ro("schema", [](const PyAction& a) { return a.schema; })
        .def_prop_ro("binding", [](const PyAction& a) {
            nb::list l;
            for (u32 o : a.binding)
                l.append(o);
            return nb::typed<nb::tuple, int, nb::ellipsis>(nb::tuple(l));
        })
        .def_prop_ro("label", [](nb::handle self) {
            nb::object b = self.attr("binding");
            return nb::typed<nb::tuple, int, nb::typed<nb::tuple, int, nb::ellipsis>>(
                nb::make_tuple(nb::inst_ptr<PyAction>(self)->schema, b));
        }, "(schema, binding)")
        .def_prop_ro("name", [](const PyAction& a) { return name_of(*a.core->data, a.core->data->schemas[a.schema].name); },
                     "The schema name.")
        .def_prop_ro("objects", [](const PyAction& a) {
            std::vector<std::string> v;
            for (u32 o : a.binding)
                v.push_back(name_of(*a.core->data, a.core->data->objects[o].name));
            return v;
        }, "Object names of the binding.")
        .def_prop_ro("arity", [](const PyAction& a) { return a.binding.size(); })
        .def_prop_ro("original_arity", [](const PyAction& a) { return a.core->data->schemas[a.schema].original_arity; },
                     "Parameters written in the PDDL (the rest were introduced by normalization).")
        .def_prop_ro("task", [](const PyAction& a) { return task_object(a.owner); })
        .def("apply",
             [](const PyAction& a, StateLike state) {
                 const Owner o{a.core, a.owner};
                 StateArg s = state_arg(*a.core, state);
                 StateBuilder b;
                 a.core->task->workspace().successors().apply(s.view, a.label(), b);
                 return make_state(o, b.build());
             },
             "state"_a)
        .def("is_applicable",
             [](const PyAction& a, StateLike state) {
                 StateArg s = state_arg(*a.core, state);
                 return a.core->task->workspace().successors().is_applicable(s.view, a.label());
             },
             "state"_a)
        .def("__eq__",
             [](const PyAction& a, nb::handle b) -> nb::object {
                 if (!nb::isinstance<PyAction>(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyAction& o = *nb::inst_ptr<PyAction>(b);
                 return nb::bool_(a.core->task->uid() == o.core->task->uid() && a.schema == o.schema && a.binding == o.binding);
             })
        .def("__ne__",
             [](const PyAction& a, nb::handle b) -> nb::object {
                 if (!nb::isinstance<PyAction>(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyAction& o = *nb::inst_ptr<PyAction>(b);
                 return nb::bool_(!(a.core->task->uid() == o.core->task->uid() && a.schema == o.schema && a.binding == o.binding));
             })
        .def("__hash__", [](const PyAction& a) {
            u64 h = hash::combine(a.core->task->uid(), a.schema);
            for (u32 o : a.binding)
                h = hash::combine(h, o);
            return py_hash(h);
        })
        .def("__reduce__", [](const PyAction& a) {
            nb::list b;
            for (u32 o : a.binding)
                b.append(o);
            return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_action"),
                                  nb::make_tuple(task_object(a.owner), a.core->task->fingerprint(), a.schema, nb::tuple(b)));
        })
        .def("__str__", [](const PyAction& a) { return action_str(a); })
        .def("__repr__", [](const PyAction& a) { return "Action" + action_str(a); });

    // --- Atom
    nb::class_<PyAtom>(m, "Atom", "A ground atom (predicate, objects).")
        .def_prop_ro("predicate", [](const PyAtom& a) { return name_of(*a.core->data, a.core->data->predicates[a.pred].name); })
        .def_prop_ro("predicate_index", [](const PyAtom& a) { return a.pred; })
        .def_prop_ro("objects", [](const PyAtom& a) {
            std::vector<std::string> v;
            for (u32 o : a.args)
                v.push_back(name_of(*a.core->data, a.core->data->objects[o].name));
            return v;
        })
        .def_prop_ro("object_indices", [](const PyAtom& a) { return a.args; })
        .def_prop_ro("slot", [](const PyAtom& a) -> Arg<std::optional<i64>> {
            if (a.slot < 0)
                return nb::none();
            return nb::int_(a.slot);
        },
                     "The fluent slot (the state bit), or None.")
        .def_prop_ro("kind", [](const PyAtom& a) {
            const auto k = a.core->data->predicates[a.pred].kind;
            return k == formalism::PredKind::Static ? "static" : k == formalism::PredKind::Fluent ? "fluent" : "derived";
        })
        .def("__eq__",
             [](const PyAtom& a, nb::handle b) -> nb::object {
                 if (!nb::isinstance<PyAtom>(b))
                     return nb::borrow(Py_NotImplemented);
                 const PyAtom& o = *nb::inst_ptr<PyAtom>(b);
                 return nb::bool_(a.core->task->uid() == o.core->task->uid() && a.pred == o.pred && a.args == o.args);
             })
        .def("__hash__", [](const PyAtom& a) {
            u64 h = hash::combine(a.core->task->uid(), a.pred + 0x9e37ULL);
            for (u32 o : a.args)
                h = hash::combine(h, o);
            return py_hash(h);
        })
        .def("__str__", [](const PyAtom& a) { return atom_str(*a.core->data, a.pred, a.args); })
        .def("__repr__", [](const PyAtom& a) { return "Atom" + atom_str(*a.core->data, a.pred, a.args); });

    // --- pickling helpers (module-level, so pickle can find them by name)
    m.def("_restore_task", &restore_task);
    m.def("_restore_state", &restore_state);
    m.def("_restore_action", [](nb::handle task, u64 fingerprint, u32 schema, nb::tuple binding) {
        if (!nb::isinstance<PyTask>(task))
            throw nb::type_error("mymyr: _restore_action needs a Task");
        PyTaskCore& core = *nb::inst_ptr<PyTask>(task)->core;
        if (core.task->fingerprint() != fingerprint)
            throw nb::value_error("mymyr: the action was pickled from a different task (content fingerprint mismatch)");
        const auto [s, b] = action_label(core, nb::int_(schema), binding);
        return make_action(Owner{&core, nb::borrow(task)}, s, reinterpret_cast<const ObjectId*>(b.data()), static_cast<u32>(b.size()));
    });
    m.def("_restore_handle", [](nb::handle task) {
        if (!nb::isinstance<PyTask>(task))
            throw nb::type_error("mymyr: _restore_handle needs a Task");
        return PyHandle{nb::inst_ptr<PyTask>(task)->core, nb::borrow(task)};
    });
}
}  // namespace mymyr::python
