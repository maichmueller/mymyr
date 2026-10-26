// Batched planning environments on the host (include/mymyr/rl/env.hpp).

#include "mymyr/rl/env.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/rl/rng.hpp"
#include "mymyr/rl/visited.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::rl
{
namespace
{
enum Status : u8
{
    k_ok = 0,
    k_stuck = 1,    // no successors: no move
    k_invalid = 2,  // the given action is out of range: no move
    k_goal = 3,
};

/// Writes the successor of cur (nw <= W words) under d into dst: W atom words (the row is wide enough for every
/// state of the instance), then the instance's numeric words, then zeros up to NN.
inline void write_row(u64* dst, u32 W, u32 NN, const u64* cur, u32 nw, const Delta& d)
{
    u32 i = 0;
    MYMYR_NOVECTOR
    for (; i < nw; ++i)
        dst[i] = cur[i];
    MYMYR_NOVECTOR
    for (; i < W; ++i)
        dst[i] = 0;
    for (SlotId x : d.del)
        bits::reset(dst, x.v);
    for (SlotId x : d.add)
        bits::set(dst, x.v);
    u32 k = 0;
    for (; k < d.nnum; ++k)
        dst[W + k] = d.num[k];
    for (; k < NN; ++k)
        dst[W + k] = 0;
}

/// The mask goal test over W words.
inline bool mask_goal(const u64* s, const u64* gpos, const u64* gneg, u32 W)
{
    for (u32 w = 0; w < W; ++w)
        if ((s[w] & gpos[w]) != gpos[w] || (s[w] & gneg[w]) != 0)
            return false;
    return true;
}

/// Successors of the prepared state of `succ`.
inline u32 count_prepared(Successors& succ, const EnvConfig& c)
{
    if (c.actions == ActionSpace::Label)
        return succ.any() ? 1 : 0;
    u32 n = 0;
    succ.generate<false>(
        [&](u32, const ObjectId*, const Delta&) -> bool
        {
            ++n;
            return true;
        },
        c.witness_pruning, c.canonical_order);
    return n;
}

StateView view_of(const u64* row, u32 W, u32 nn) { return StateView{row, bits::trimmed_size(row, W), nn ? row + W : nullptr, nn}; }

/// Row i of b as a state of `task`: the task's own atom words (the words past them are padding) and its numeric words.
StateView row_view(const EnvBatch& b, u64 i, const Task& task)
{
    const u64* row = b.states + i * b.row_words();
    const u32 nn = task.numeric_words();
    return StateView{row, bits::trimmed_size(row, std::min(b.words, task.words())), nn ? row + b.words : nullptr, nn};
}
}  // namespace

void check_env(const TaskSuite& table, const EnvBatch& b, const StepOutputs* out)
{
    if (b.visited_capacity && (b.visited_capacity < 2 || b.visited_capacity > (1u << 20) ||
                               (b.visited_capacity & (b.visited_capacity - 1))))
        throw std::invalid_argument("mymyr: env: visited capacity must be a power of two in [2, 2^20]");
    if (b.visited_capacity && b.rows && (!b.visited || !b.visited_count || !b.visited_status))
        throw std::invalid_argument("mymyr: env: visited tracking needs table, count and status arrays");
    if (b.rows && !b.states)
        throw std::invalid_argument("mymyr: env: null state array");
    if (b.rows > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::invalid_argument("mymyr: env: more than 2^31 - 1 environments in one batch");
    if (b.words < table.words())
        throw std::invalid_argument("mymyr: env: state rows of " + std::to_string(b.words) + " words; the " +
                                    table.noun() + " needs " + std::to_string(table.words()) +
                                    " (its widest instance's words)");
    if (b.numeric_words != table.numeric_words())
        throw std::invalid_argument("mymyr: env: rows carry " + std::to_string(b.numeric_words) + " numeric words, the " +
                                    table.noun() + " has " + std::to_string(table.numeric_words()));
    if ((b.goal_pos == nullptr) != (b.goal_neg == nullptr))
        throw std::invalid_argument("mymyr: env: per-env goals need both goal_pos and goal_neg");
    if (b.goals())
        for (u32 i = 0; i < table.size(); ++i)
            if (table.instance(i).goal_derived || table.instance(i).goal_unsatisfiable)
                throw std::invalid_argument(
                    "mymyr: env: per-env goal masks cannot express the goal of instance " + std::to_string(i) +
                    (table.instance(i).goal_derived ? " (it has derived literals)" : " (it is statically unsatisfiable)") +
                    "; use the instances' goal tests (no goal masks)");
    if (!b.task_ids && b.rows && table.size() > 1)
        throw std::invalid_argument(std::string("mymyr: env: a batch over a ") + table.noun() + " of " +
                                    std::to_string(table.size()) + " instances needs task ids");
    if (out && out->binding && out->label_width < table.label_width())
        throw std::invalid_argument("mymyr: env: label width " + std::to_string(out->label_width) +
                                    " is below the largest schema arity " + std::to_string(table.label_width()));
}

HostEnv::HostEnv(TaskSuitePtr suite, const EnvConfig& config) : m_suite(std::move(suite)), m_config(config)
{
    if (!m_suite)
        throw std::invalid_argument("mymyr: HostEnv: null task suite");
    m_init_count.resize(m_suite->size());
    m_init_goal.resize(m_suite->size());
    m_init_hash.resize(m_suite->size());
    m_tasks.resize(m_suite->size());
    m_label_width = std::max<u32>(1, m_suite->label_width());
    for (u32 i = 0; i < m_suite->size(); ++i)
    {
        const Task& task = *m_suite->task(i);
        m_tasks[i] = &task;
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
        succ.prepare(view_of(m_suite->instance(i).init.data(), m_suite->table_of(i).words(), task.numeric_words()));
        m_init_hash[i] = view_of(m_suite->instance(i).init.data(), m_suite->table_of(i).words(), task.numeric_words()).hash();
        m_init_count[i] = count_prepared(succ, m_config);
        m_init_goal[i] = succ.goal_holds() ? 1 : 0;
    }
}

HostEnv::~HostEnv() = default;

u32 HostEnv::reset_row(EnvBatch& b, u64 i, u32 instance, bool keep_goals, u32 dirty) const
{
    m_suite->initial_row(instance, b.states + i * b.row_words(), b.words, b.numeric_words, dirty);
    if (b.task_ids)
        b.task_ids[i] = static_cast<i32>(instance);
    if (b.steps)
        b.steps[i] = 0;
    if (b.goals() && !keep_goals)
    {
        const TaskTable& t = m_suite->table_of(instance);
        const TaskTable::Instance& in = t.instance(m_suite->local_id(instance));
        const u32 w = in.words, z = std::min(b.words, std::max(w, dirty));
        u64* gp = b.goal_pos + i * b.words;
        u64* gn = b.goal_neg + i * b.words;
        std::copy_n(in.goal_pos.begin(), w, gp);
        std::fill(gp + w, gp + z, u64{0});
        std::copy_n(in.goal_neg.begin(), w, gn);
        std::fill(gn + w, gn + z, u64{0});
    }
    if (b.visited_capacity)
    {
        u64* table = b.visited + i * b.visited_capacity;
        visited_clear(table, b.visited_capacity, b.visited_count[i]);
        visited_insert(table, b.visited_capacity, b.visited_count[i], m_init_hash[instance]);
    }
    return m_init_count[instance];
}

void HostEnv::clear_visited(EnvBatch& b, const u8* mask) const
{
    check_env(*m_suite, b, nullptr);
    m_suite->check_task_ids(b.task_ids, b.rows);
    if (!b.visited_capacity)
        return;
    for (u64 i = 0; i < b.rows; ++i)
    {
        if (mask && !mask[i])
            continue;
        u64* table = b.visited + i * b.visited_capacity;
        visited_clear(table, b.visited_capacity, b.visited_count[i]);
        visited_insert(table, b.visited_capacity, b.visited_count[i], row_view(b, i, *m_tasks[b.instance(i)]).hash());
    }
}

void HostEnv::reset(EnvBatch& b, const u8* mask, i32* count, bool keep_goals) const
{
    check_env(*m_suite, b, nullptr);
    m_suite->check_task_ids(b.task_ids, b.rows);
    for (u64 i = 0; i < b.rows; ++i)
    {
        if (mask && !mask[i])
            continue;
        const u32 c = reset_row(b, i, b.instance(i), keep_goals, b.words);  // what the row held: unknown
        if (count)
            count[i] = static_cast<i32>(c);
        if (b.at_goal)
            b.at_goal[i] = reset_goal(b, i, b.instance(i)) ? 1 : 0;
    }
}

bool HostEnv::goal_row(const EnvBatch& b, u64 i) const
{
    const Task& task = *m_tasks[b.instance(i)];
    if (b.goals())
        return mask_goal(b.states + i * b.row_words(), b.goal_pos + i * b.words, b.goal_neg + i * b.words,
                         std::min(b.words, task.words()));
    return task.is_goal(row_view(b, i, task));
}

bool HostEnv::reset_goal(const EnvBatch& b, u64 i, u32 instance) const
{
    return b.goals() ? goal_row(b, i) : m_init_goal[instance] != 0;
}

void HostEnv::goal(const EnvBatch& b, u8* goal) const
{
    if (b.rows && !goal)
        throw std::invalid_argument("mymyr: env: goal query needs an output array");
    check_env(*m_suite, b, nullptr);
    m_suite->check_task_ids(b.task_ids, b.rows);
    for (u64 row = 0; row < b.rows; ++row)
        goal[row] = goal_row(b, row) ? 1 : 0;
}

void HostEnv::refresh(EnvBatch& b, i32* count, ThreadPool* pool)
{
    if (count)
        this->count(b, count, pool);
    if (b.at_goal)
        goal(b, b.at_goal);
}

u32 HostEnv::count_row(const EnvBatch& b, u64 i) const
{
    const Task& task = *m_tasks[b.instance(i)];
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    succ.prepare(row_view(b, i, task));
    return count_prepared(succ, m_config);
}

void HostEnv::count(const EnvBatch& b, i32* count, ThreadPool* pool)
{
    check_env(*m_suite, b, nullptr);
    m_suite->check_task_ids(b.task_ids, b.rows);
    auto rows = [&](u64 lo, u64 hi)
    {
        for (u64 i = lo; i < hi; ++i)
            count[i] = static_cast<i32>(count_row(b, i));
    };
    if (!pool || pool->size() == 1)
        rows(0, b.rows);
    else
        pool->run(
            [&](u32 t)
            {
                const auto [lo, hi] = ThreadPool::slice(b.rows, t, pool->size());
                rows(lo, hi);
            });
}

void HostEnv::check_step(const EnvBatch& b, const StepOutputs& out, Actions action, const i32* next_task_ids) const
{
    check_env(*m_suite, b, &out);
    m_suite->check_task_ids(b.task_ids, b.rows);
    if (action.v64 && action.v32)
        throw std::invalid_argument("mymyr: env: actions given as int64 and as int32");
    if (m_config.actions == ActionSpace::Index && !action.given() && b.rows && !b.draws)
        throw std::invalid_argument("mymyr: env: the random policy needs the draw counters");
    if (m_config.max_steps && b.rows && !b.steps)
        throw std::invalid_argument("mymyr: env: truncation (max_steps) needs the step counters");
    m_suite->check_task_ids(next_task_ids, b.rows);
    if (out.final_states && out.words && out.words < m_suite->words())
        for (u64 i = 0; i < b.rows; ++i)
            if (out.words < m_suite->instance(b.instance(i)).words)
                throw std::invalid_argument("mymyr: env: final state rows of " + std::to_string(out.words) +
                                            " words; instance " + std::to_string(b.instance(i)) + " has " +
                                            std::to_string(m_suite->instance(b.instance(i)).words));
}

void HostEnv::step_row(EnvBatch& b, u64 i, const StepOutputs& out, u64 o, i64 action, bool random,
                       i32 next_instance, Scratch& S) const
{
    if (m_config.actions != ActionSpace::Index)
        throw std::invalid_argument("mymyr: env: label mode requires explicit labels");
    step_row_impl(b, i, out, o, action, random, next_instance, S, nullptr);
}

void HostEnv::step_row_impl(EnvBatch& b, u64 i, const StepOutputs& out, u64 o, i64 action, bool random,
                            i32 next_instance, Scratch& S, const Labels* labels) const
{
    const u32 inst = b.instance(i);
    const Task& task = *m_tasks[inst];
    // w: the instance's own words, all a step reads or writes of the row (the rest is padding); successors are kept
    // in scratch rows of w + NN words
    const u32 W = b.words, NN = b.numeric_words, RW = W + NN, nn = task.numeric_words();
    const u32 w = std::min(W, task.words()), SW = w + NN;
    const u32 FW = out.words ? out.words : W;
    const u32 L = m_label_width;
    u64* row = b.states + i * RW;
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();

    // 1. the chosen label's successor, or the current state's indexed successors
    const u32 nw = bits::trimmed_size(row, w);
    succ.prepare(StateView{row, nw, nn ? row + W : nullptr, nn});
    const bool already_goal = b.at_goal ? b.at_goal[i] != 0
                              : b.goals() ? mask_goal(row, b.goal_pos + i * W, b.goal_neg + i * W, w)
                                          : succ.goal_holds();
    S.succ.clear();
    S.schema.clear();
    S.binding.clear();
    u32 c = 0;
    if (labels)
    {
        c = count_prepared(succ, m_config);
        const i32 schema = labels->schema[i];
        if (!already_goal && c && schema >= 0 && static_cast<u32>(schema) < succ.num_schemas())
        {
            const u32 arity = succ.arity(static_cast<u32>(schema));
            S.label.resize(arity);
            for (u32 k = 0; k < arity; ++k)
                S.label[k] = ObjectId{static_cast<u32>(labels->binding[i * labels->stride + k])};
            const ActionLabel label{SchemaId{static_cast<u32>(schema)}, S.label};
            const StateView current{row, nw, nn ? row + W : nullptr, nn};
            if (succ.is_applicable(current, label))
            {
                const Delta d = succ.apply_with_delta(current, label, S.next);
                S.succ.resize(SW);
                write_row(S.succ.data(), w, NN, row, nw, d);
                S.schema.push_back(schema);
                for (u32 k = 0; k < L; ++k)
                    S.binding.push_back(k < arity ? static_cast<i32>(S.label[k].v) : -1);
            }
        }
    }
    else if (!already_goal)
    {
        succ.generate<false>(
            [&](u32 schema, const ObjectId* binding, const Delta& d) -> bool
            {
                const usize at = S.succ.size();
                S.succ.resize(at + SW);
                write_row(S.succ.data() + at, w, NN, row, nw, d);
                S.schema.push_back(static_cast<i32>(schema));
                const u32 arity = succ.arity(schema);
                for (u32 k = 0; k < L; ++k)
                    S.binding.push_back(k < arity ? static_cast<i32>(binding[k].v) : -1);
                ++c;
                return true;
            },
            m_config.witness_pruning, m_config.canonical_order);
    }

    // 2. the choice
    u8 st = already_goal ? k_goal : k_ok;
    u64 a = 0;
    if (already_goal && !labels)
        c = count_prepared(succ, m_config);
    else if (!already_goal && c == 0)
        st = k_stuck;
    else if (!already_goal && labels)
    {
        if (S.schema.empty())
            st = k_invalid;
    }
    else if (!already_goal && !random)
    {
        if (action < 0 || static_cast<u64>(action) >= c)
            st = k_invalid;
        else
            a = static_cast<u64>(action);
    }
    else if (!already_goal)
        a = rng::successor_index(b.seeds ? b.seeds[i] : m_config.seed, b.env_ids ? u64{b.env_ids[i]} : b.first_env + i,
                                 b.draws[i], c);
    if (b.draws)
        ++b.draws[i];
    if (out.schema)
        out.schema[o] = st == k_ok ? S.schema[a] : -1;
    if (out.binding)
    {
        i32* dst = out.binding + o * out.label_width;
        for (u32 k = 0; k < out.label_width; ++k)
            dst[k] = st == k_ok && k < L ? S.binding[a * L + k] : -1;
    }
    if (out.invalid)
        out.invalid[o] = st == k_invalid ? 1 : 0;

    // 3. goal and successor count of the reached state (s' = s without a move): its w atom words at s2, its numeric
    // words at s2n
    const u64* s2 = st == k_ok ? S.succ.data() + a * SW : row;
    const u64* s2n = st == k_ok ? s2 + w : row + W;
    bool g = already_goal;
    u32 c2 = c;
    if (st == k_ok)
    {
        succ.prepare(StateView{s2, bits::trimmed_size(s2, w), nn ? s2n : nullptr, nn});
        g = b.goals() ? mask_goal(s2, b.goal_pos + i * W, b.goal_neg + i * W, w) : succ.goal_holds();
        c2 = count_prepared(succ, m_config);
    }

    // 4. rewards, termination, truncation
    const bool dead = m_config.dead_end == DeadEnd::NoSuccessors && !g && ((st == k_ok && c2 == 0) || st == k_stuck);
    const bool term = g || (dead && m_config.dead_end_terminal);
    const bool absorbed = already_goal || (st == k_stuck && dead && m_config.dead_end_terminal);
    const i32 t2 = b.steps ? b.steps[i] + (absorbed ? 0 : 1) : 0;
    const bool trunc = !term && m_config.max_steps && t2 >= static_cast<i32>(m_config.max_steps);
    if (out.reward)
        out.reward[o] = absorbed ? 0.0f : m_config.step_reward + (g ? m_config.goal_reward : 0.0f) + (dead ? m_config.dead_end_reward : 0.0f);
    if (out.terminated)
        out.terminated[o] = term ? 1 : 0;
    if (out.truncated)
        out.truncated[o] = trunc ? 1 : 0;
    if (out.goal)
        out.goal[o] = g ? 1 : 0;
    if (out.final_states)
    {
        u64* f = out.final_states + o * (FW + NN);
        std::copy_n(s2, w, f);
        std::fill(f + w, f + FW, u64{0});
        std::copy_n(s2n, NN, f + FW);
    }

    bool revisited = false;
    const bool episode_over = !m_config.autoreset &&
        (absorbed || (m_config.max_steps && b.steps && b.steps[i] >= static_cast<i32>(m_config.max_steps)));
    if (b.visited_capacity && !episode_over)
    {
        const u32 H = b.visited_capacity;
        revisited = visited_insert(b.visited + i * H, H, b.visited_count[i],
                                   StateView{s2, bits::trimmed_size(s2, w), nn ? s2n : nullptr, nn}.hash());
        if (b.visited_count[i] == static_cast<i32>(H))
            std::atomic_ref<i32>(*b.visited_status).fetch_or(1, std::memory_order_relaxed);
    }
    if (out.revisited)
        out.revisited[o] = revisited ? 1 : 0;

    // 5. the move, or the autoreset
    if (m_config.autoreset && (term || trunc))
    {
        const u32 next = next_instance >= 0 ? static_cast<u32>(next_instance) : inst;
        const u32 n0 = reset_row(b, i, next, false, w);
        if (out.count)
            out.count[o] = static_cast<i32>(n0);
        if (b.at_goal)
            b.at_goal[i] = reset_goal(b, i, next) ? 1 : 0;
        return;
    }
    if (b.at_goal)
        b.at_goal[i] = g ? 1 : 0;
    if (st == k_ok)
    {
        std::copy_n(s2, w, row);
        std::copy_n(s2n, NN, row + W);
    }
    if (b.steps)
        b.steps[i] = t2;
    if (out.count)
        out.count[o] = static_cast<i32>(c2);
}

void HostEnv::step(EnvBatch& b, const StepOutputs& out, Actions action, const i32* next_task_ids, ThreadPool* pool)
{
    if (m_config.actions != ActionSpace::Index)
        throw std::invalid_argument("mymyr: env: label mode requires explicit labels; the random policy is unavailable");
    check_step(b, out, action, next_task_ids);
    step_impl(b, out, action, nullptr, next_task_ids, pool);
}

void HostEnv::step(EnvBatch& b, const StepOutputs& out, Labels labels, const i32* next_task_ids, ThreadPool* pool)
{
    if (m_config.actions != ActionSpace::Label)
        throw std::invalid_argument("mymyr: env: explicit labels require actions=Label");
    check_step(b, out, {}, next_task_ids);
    if (b.rows && (!labels.schema || (m_suite->label_width() && !labels.binding)))
        throw std::invalid_argument("mymyr: env: label mode needs schema and binding arrays");
    if (labels.stride < m_suite->label_width())
        throw std::invalid_argument("mymyr: env: label binding stride is below the suite's largest schema arity");
    step_impl(b, out, {}, &labels, next_task_ids, pool);
}

void HostEnv::step_impl(EnvBatch& b, const StepOutputs& out, Actions action, const Labels* labels,
                        const i32* next_task_ids, ThreadPool* pool)
{
    const bool random = !labels && !action.given();
    const u32 T = pool ? pool->size() : 1;
    if (m_scratch.size() < T)
        m_scratch.resize(T);
    auto rows = [&](u64 lo, u64 hi, Scratch& S)
    {
        for (u64 i = lo; i < hi; ++i)
            step_row_impl(b, i, out, i, labels || random ? 0 : action.at(i), random,
                          next_task_ids ? next_task_ids[i] : -1, S, labels);
    };
    if (T == 1 || b.rows < 2)
        rows(0, b.rows, m_scratch[0]);
    else
        pool->run(
            [&](u32 t)
            {
                const auto [lo, hi] = ThreadPool::slice(b.rows, t, T);
                rows(lo, hi, m_scratch[t]);
            });
}
}  // namespace mymyr::rl
