#include "mymyr/task/task.hpp"

#include "compile.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/formalism/fingerprint.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/workspace.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <unordered_set>

namespace mymyr
{
std::shared_ptr<const Task> Task::create(formalism::TaskData data, const TaskOptions& options)
{
    return std::make_shared<const Task>(Private{}, std::move(data), options);
}

std::shared_ptr<const Task> Task::from_text_file(const std::string& path, const TaskOptions& options)
{
    return create(formalism::read_task_text_file(path), options);
}

Task::Task(Private, formalism::TaskData data, const TaskOptions& options) : m_data(std::move(data)), m_options(options)
{
    const auto t0 = std::chrono::steady_clock::now();
    formalism::validate(m_data);
    detail::check_supported(m_data);
    detail::compile_task(m_data, m_options, m_compiled);
    m_info.dense_fluent = m_compiled.layout.fluent_count;
    m_info.dense_derived = m_compiled.layout.total - m_compiled.layout.fluent_count;
    choose_atom_mode();
    m_info.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Task::~Task() = default;

u64 Task::fingerprint() const
{
    m_fingerprint_once.call([this] { m_fingerprint = formalism::fingerprint(m_data); });
    return m_fingerprint;
}

namespace
{
/// Bytes of the per-atom records a frozen index allocates up front.
u64 frozen_bytes(const CanonicalLayout& L)
{
    const u64 per = (1 + std::max<u64>(1, L.max_arity)) * sizeof(u32) + sizeof(ViewOp) + sizeof(CanonicalAtom);
    return L.total * per + (L.total + 1) * sizeof(u32);
}
constexpr u64 k_frozen_max_bytes = u64{4} << 30;
}  // namespace

void Task::choose_atom_mode()
{
    const CanonicalLayout& L = m_compiled.layout;
    AtomMode mode = AtomMode::Lazy;
    switch (m_options.atoms)
    {
        case TaskOptions::Atoms::Lazy: mode = AtomMode::Lazy; break;
        case TaskOptions::Atoms::Frozen:
            if (frozen_bytes(L) > k_frozen_max_bytes)
                throw std::length_error("mymyr: frozen atom slots need " + std::to_string(frozen_bytes(L) >> 20) +
                                        " MiB of records (typed-dense space of " + std::to_string(L.total) + " atoms)");
            mode = AtomMode::Frozen;
            break;
        case TaskOptions::Atoms::Auto:
        {
            // frozen when W_dense <= frozen_max_words, or W_dense <= 2 * W_lazy (estimated by a bounded pilot).
            const u32 wd = bits::words_for(L.fluent_count);
            if (L.total > CanonicalLayout::k_max_total || frozen_bytes(L) > k_frozen_max_bytes)
                mode = AtomMode::Lazy;
            else if (wd <= m_options.frozen_max_words)
                mode = AtomMode::Frozen;
            else if (m_options.pilot_expansions > 0)
            {
                m_atoms.init(&L, &m_compiled.view, AtomMode::Lazy);
                build_initial_state();
                const u32 wl = pilot_words();
                m_info.pilot_words = wl;
                mode = wd <= 2 * wl ? AtomMode::Frozen : AtomMode::Lazy;
            }
            break;
        }
    }
    m_atoms.init(&L, &m_compiled.view, mode);  // also discards the pilot's lazy slots
    m_info.atom_mode = mode;
    build_initial_state();
}

u32 Task::pilot_words()
{
    // A bounded lazy BrFS with its own (temporary) engine: the index is reset afterwards.
    detail::Engine engine(*this);
    AxiomEvaluator axioms(engine);
    Successors succ(engine, axioms);
    succ.set_canonical_order(false);
    std::unordered_set<State> seen;
    std::deque<State> queue;
    seen.insert(m_initial);
    queue.push_back(m_initial);
    std::vector<u64> next;
    u32 expansions = 0;
    while (!queue.empty() && expansions < m_options.pilot_expansions)
    {
        const State cur = std::move(queue.front());
        queue.pop_front();
        ++expansions;
        succ.prepare(cur);
        succ.generate<false>(
            [&](u32, const ObjectId*, const Delta& d)
            {
                const u32 n = apply_delta(cur.data(), cur.size_words(), d, next);
                State s(next.data(), n, d.num, d.nnum);
                if (seen.insert(s).second)
                    queue.push_back(std::move(s));
                return true;
            });
    }
    return bits::words_for(m_atoms.fluent_slots());
}

void Task::build_initial_state()
{
    const CanonicalLayout& L = m_compiled.layout;
    std::vector<u64> w;
    for (const auto& rec : m_compiled.initial)
    {
        const u32 s = m_atoms.intern(L.encode(rec[0], rec.data() + 1));
        if (bits::word_of(s) >= w.size())
            w.resize(bits::word_of(s) + 1, 0);
        bits::set(w.data(), s);
    }
    const plan::Numeric& N = m_compiled.num;
    std::vector<u64> num(N.words, 0);
    for (u32 i = 0; i < N.slots; ++i)
        plan::store(N, num.data(), i, N.initial[i]);
    m_initial = State(w.data(), static_cast<u32>(w.size()), num.data(), N.words);
}

WorkspaceLease Task::workspace() const
{
    detail::WorkspacePool& pool = m_workspaces.local([] { return std::make_unique<detail::WorkspacePool>(); });
    std::unique_ptr<Workspace> ws;
    if (!pool.idle.empty())
    {
        ws = std::move(pool.idle.back());
        pool.idle.pop_back();
    }
    else if (m_spare_count.load(std::memory_order_relaxed) != 0)
    {
        std::lock_guard lock(m_spare_mutex);
        if (!m_spare.empty())
        {
            ws = std::move(m_spare.back());
            m_spare.pop_back();
            m_spare_count.store(m_spare.size(), std::memory_order_relaxed);
        }
    }
    if (!ws)
    {
        ws = std::make_unique<Workspace>(*this);
        m_workspaces_made.fetch_add(1, std::memory_order_relaxed);
    }
    Successors& succ = ws->successors();
    succ.set_witness_pruning(true);
    succ.set_canonical_order(true);
    succ.set_schema_filter(nullptr);
    return WorkspaceLease(*this, &pool, std::move(ws));
}

void Task::give_back(detail::WorkspacePool* home, std::unique_ptr<Workspace> ws) const noexcept
{
    try
    {
        if (m_workspaces.find() == home)  // the pool of the thread that took it, or of a thread that inherited it
            home->idle.push_back(std::move(ws));
        else
        {
            std::lock_guard lock(m_spare_mutex);
            m_spare.push_back(std::move(ws));
            m_spare_count.store(m_spare.size(), std::memory_order_relaxed);
        }
    }
    catch (...)  // out of memory for one pointer: free the workspace instead of keeping it
    {
    }
}

WorkspaceLease::WorkspaceLease(const Task& task, detail::WorkspacePool* home, std::unique_ptr<Workspace> ws) noexcept
    : m_task(&task), m_home(home), m_ws(std::move(ws))
{
}

WorkspaceLease::WorkspaceLease(WorkspaceLease&& other) noexcept
    : m_task(other.m_task), m_home(other.m_home), m_ws(std::move(other.m_ws))
{
}

WorkspaceLease& WorkspaceLease::operator=(WorkspaceLease&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_task = other.m_task;
        m_home = other.m_home;
        m_ws = std::move(other.m_ws);
    }
    return *this;
}

WorkspaceLease::~WorkspaceLease() { release(); }

void WorkspaceLease::release() noexcept
{
    if (m_ws)
        m_task->give_back(m_home, std::move(m_ws));
}

State Task::make_state(std::span<const AtomArgs> atoms, std::span<const f64> values) const
{
    const CanonicalLayout& L = m_compiled.layout;
    const auto atom_name = [this](const AtomArgs& a) {
        std::string s = "(" + std::string(m_data.str(m_data.predicates[a.predicate.v].name));
        for (ObjectId o : a.objects)
            s += " " + (o.v < m_data.objects.size() ? std::string(m_data.str(m_data.objects[o.v].name)) : "?");
        return s + ")";
    };
    std::vector<u64> w;
    std::vector<u32> args;
    for (const AtomArgs& a : atoms)
    {
        if (a.predicate.v >= m_data.predicates.size())
            throw std::invalid_argument("mymyr make_state: predicate index out of range");
        if (m_compiled.kinds[a.predicate.v] != formalism::PredKind::Fluent)
            throw std::invalid_argument("mymyr make_state: " + atom_name(a) +
                                        " is not a fluent atom (static and derived atoms are not part of a state)");
        if (a.objects.size() != L.arity[a.predicate.v])
            throw std::invalid_argument("mymyr make_state: " + atom_name(a) + " has the wrong number of objects");
        args.resize(a.objects.size());
        for (usize i = 0; i < args.size(); ++i)
        {
            if (a.objects[i].v >= num_objects())
                throw std::invalid_argument("mymyr make_state: object index out of range");
            args[i] = a.objects[i].v;
        }
        const CanonicalAtom c = L.encode(a.predicate.v, args.data());
        if (c >= L.fluent_count)
            throw std::invalid_argument("mymyr make_state: " + atom_name(a) +
                                        " lies outside the reachable domains of its predicate");
        const u32 slot = m_atoms.intern(c);
        if (bits::word_of(slot) >= w.size())
            w.resize(bits::word_of(slot) + 1, 0);
        bits::set(w.data(), slot);
    }
    const plan::Numeric& N = m_compiled.num;
    if (values.size() != N.slots)
        throw std::invalid_argument("mymyr make_state: " + std::to_string(values.size()) + " numeric values for " +
                                    std::to_string(N.slots) + " numeric slots");
    std::vector<u64> num(N.words, 0);
    for (u32 i = 0; i < N.slots; ++i)
    {
        if (std::isnan(values[i]))
            throw std::invalid_argument("mymyr make_state: numeric value " + numeric_name(i) + " is NaN");
        plan::store(N, num.data(), i, values[i]);
    }
    return State(w.data(), static_cast<u32>(w.size()), num.data(), N.words);
}

bool Task::is_goal(StateView s) const
{
    const plan::Goal& g = m_compiled.goal;
    if (g.unsatisfiable)
        return false;
    if (g.uses_derived || !m_compiled.num.goal.empty())
    {
        const WorkspaceLease ws = workspace();
        return ws->successors().is_goal(s);
    }
    for (const plan::Check& c : g.lits)
    {
        const u32 slot = m_atoms.find(c.pat.base);  // ground pattern: the key is the base
        if (bits::test(s.w, s.nw, slot) != c.pos)
            return false;
    }
    return true;
}

u64 Task::canonical_hash(StateView s) const
{
    u64 x = 0;
    bits::for_each(s.w, s.nw,
                   [&](u64 slot) { x ^= hash::mix64(m_atoms.canonical(AtomKind::Fluent, static_cast<u32>(slot)) + 0x1234567ULL); });
    if (s.nnum)
    {
        // numeric slots are in canonical (function, arguments) order; the values are hashed as canonical doubles, so the
        // hash does not depend on the slot type (I32 or F64)
        const plan::Numeric& N = m_compiled.num;
        u64 h = 0x6a09e667f3bcc909ULL;
        for (u32 i = 0; i < N.slots; ++i)
            h = hash::mix64(h ^ std::bit_cast<u64>(plan::canonical(N, plan::load(N, s.num, i))) ^ (u64{i} << 1));
        x ^= h;
    }
    return x;
}

f64 Task::numeric_value(StateView s, u32 slot) const
{
    if (slot >= m_compiled.num.slots)
        throw std::out_of_range("mymyr: numeric slot out of range");
    if (s.nnum < m_compiled.num.words)
        throw std::invalid_argument("mymyr: the state has no numeric values of this task");
    return plan::load(m_compiled.num, s.num, slot);
}

std::vector<f64> Task::numeric_values(StateView s) const
{
    std::vector<f64> out(m_compiled.num.slots);
    for (u32 i = 0; i < m_compiled.num.slots; ++i)
        out[i] = numeric_value(s, i);
    return out;
}

std::string Task::numeric_name(u32 slot) const
{
    const plan::Numeric& N = m_compiled.num;
    if (slot >= N.slots)
        throw std::out_of_range("mymyr: numeric slot out of range");
    std::string out = "(" + std::string(m_data.str(m_data.functions.at(N.slot_function[slot]).name));
    for (u32 i = N.slot_args_begin[slot]; i < N.slot_args_begin[slot + 1]; ++i)
        out += " " + object_name(N.slot_args[i]);
    return out + ")";
}

std::string Task::schema_name(SchemaId s) const { return std::string(m_data.str(m_data.schemas.at(s.v).name)); }

std::string Task::object_name(ObjectId o) const { return std::string(m_data.str(m_data.objects.at(o.v).name)); }

std::string Task::format(const ActionLabel& a) const
{
    std::string out = "(" + schema_name(a.schema);
    for (ObjectId o : a.binding)
        out += " " + object_name(o);
    return out + ")";
}

std::string Task::format(SlotId fluent_slot) const
{
    const u32* rec = m_atoms.record(AtomKind::Fluent, fluent_slot.v);
    std::string out = "(" + std::string(m_data.str(m_data.predicates.at(rec[0]).name));
    for (u32 i = 0; i < m_compiled.arity[rec[0]]; ++i)
        out += " " + object_name(ObjectId{rec[1 + i]});
    return out + ")";
}

std::vector<std::string> Task::format_atoms(StateView s) const
{
    std::vector<std::string> out;
    bits::for_each(s.w, s.nw, [&](u64 slot) { out.push_back(format(SlotId{static_cast<u32>(slot)})); });
    return out;
}

SlotId Task::find_atom(PredicateId pred, std::span<const ObjectId> args) const
{
    if (pred.v >= m_compiled.kinds.size() || m_compiled.kinds[pred.v] != formalism::PredKind::Fluent ||
        args.size() != m_compiled.arity[pred.v])
        return SlotId{};
    std::vector<u32> a(args.size());
    for (usize i = 0; i < args.size(); ++i)
    {
        if (args[i].v >= m_compiled.num_objects)
            return SlotId{};
        a[i] = args[i].v;
    }
    const u32 s = m_atoms.find(m_compiled.layout.encode(pred.v, a.data()));
    return s == AtomIndex::k_empty ? SlotId{} : SlotId{s};
}
}  // namespace mymyr
