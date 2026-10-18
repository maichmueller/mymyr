#pragma once
// Shared rows and configurations for host and device relational encoding comparisons.
#include "support.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/task/workspace.hpp"
#include <array>
#include <random>

namespace mymyr::test::encoding_matrix
{
rl::TaskTablePtr table_of(const std::string& domain, const std::vector<std::string>& problems)
{
    std::vector<TaskPtr> tasks;
    for (const std::string& problem : problems)
        tasks.push_back(test::pddl_task(domain, problem));
    return rl::TaskTable::create(tasks);
}

/// Instances of 1, 42 and 184 words: `link` over n objects has n * n slots, `flag` n more, and `armed` one.
rl::TaskTablePtr wide_table()
{
    const auto domain = frontend::Domain::from_string(
        "(define (domain wide) (:requirements :strips)"
        " (:predicates (link ?x ?y) (flag ?x) (armed) (fixed ?x ?y))"
        " (:action connect :parameters (?x ?y) :precondition (and) :effect (link ?x ?y))"
        " (:action mark :parameters (?x) :precondition (and) :effect (flag ?x))"
        " (:action arm :parameters () :precondition (and) :effect (armed)))");
    std::vector<TaskPtr> tasks;
    for (const u32 n : {5u, 51u, 108u})
    {
        std::string objects, init = "(fixed o0 o1) (fixed o1 o2) (armed)";
        for (u32 i = 0; i < n; ++i)
            objects += " o" + std::to_string(i);
        const std::string problem = "(define (problem p" + std::to_string(n) + ") (:domain wide) (:objects" + objects +
                                    ") (:init " + init + ") (:goal (and (link o0 o1) (flag o2) (armed))))";
        tasks.push_back(Task::create(*domain->instantiate_string(problem)));
    }
    return rl::TaskTable::create(tasks);
}

// ------------------------------------------------------------------------------------------------ rows

/// The rows of a batch over a table with everything an encode call can take besides the options.
struct Batch
{
    rl::TaskTablePtr table;
    u64 count = 0;
    u32 words = 0, derived_words = 0;
    std::vector<u64> states;  // [count, words]
    std::vector<i32> task_ids;
    std::vector<u64> derived;  // [count, derived_words], the axiom evaluation of each row; empty without derived atoms
    std::vector<u64> goal_pos, goal_neg;                  // [count, words] masks over a subset of the row's fluent slots
    std::vector<u64> goal_derived_pos, goal_derived_neg;  // [count, derived_words]
};

void set_bit(u64* row, u32 slot) { row[slot >> 6] |= u64{1} << (slot & 63); }
bool test_bit(const u64* row, u32 slot) { return (row[slot >> 6] >> (slot & 63)) & 1u; }

/// All zeros, all ones within the instance's slots, single bits at 63, 64 and 65 and at the last slot, and the last
/// partial word set.
std::vector<std::vector<u64>> adversarial_rows(const rl::TaskTable& table, u32 instance)
{
    const u32 slots = table.task(instance)->atoms().fluent_slots();
    std::vector<std::vector<u64>> rows;
    auto blank = [&] { return std::vector<u64>(table.words(), 0); };
    rows.push_back(blank());
    {
        auto row = blank();
        for (u32 s = 0; s < slots; ++s)
            set_bit(row.data(), s);
        rows.push_back(row);
    }
    for (const u32 bit : {63u, 64u, 65u, slots ? slots - 1 : 0u})
        if (bit < slots)
        {
            auto row = blank();
            set_bit(row.data(), bit);
            rows.push_back(row);
        }
    if (slots)
    {
        auto row = blank();
        for (u32 s = ((slots - 1) / 64) * 64; s < slots; ++s)
            set_bit(row.data(), s);
        rows.push_back(row);
    }
    return rows;
}

/// The derived words of a state row by a fresh axiom evaluation, as the host encoder does it.
std::vector<u64> derived_words_of(const rl::TaskTable& table, u32 instance, const u64* row, u32 width)
{
    const Task& task = *table.task(instance);
    std::vector<u64> out(width, 0);
    const WorkspaceLease workspace = task.workspace();
    auto& engine = workspace->engine();
    engine.set_state(row, table.instance(instance).words);
    engine.build_view();
    workspace->axioms().evaluate();
    std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), out.begin());
    return out;
}

/// `count` rows: the adversarial rows of every instance first, then rows interleaved by instance: random walks from
/// the initial states (`walk`) or random bit patterns of mixed density with garbage in the padding words past a narrower
/// instance's words.
Batch make_batch(const rl::TaskTablePtr& table, u64 count, bool walk, u64 seed)
{
    Batch b;
    b.table = table;
    b.words = table->words();
    b.derived_words = table->derived_words();
    const u32 instances = table->size();
    std::mt19937_64 rng(seed);
    std::vector<test::Rows> walks;
    if (walk)
        for (u32 i = 0; i < instances; ++i)
            walks.push_back(test::random_walk(*table, i, count / instances + 1, seed + i));
    std::vector<std::vector<u64>> special;
    std::vector<i32> special_ids;
    for (u32 i = 0; i < instances; ++i)
        for (auto& row : adversarial_rows(*table, i))
        {
            special.push_back(std::move(row));
            special_ids.push_back(static_cast<i32>(i));
        }
    for (u64 k = 0; k < count; ++k)
    {
        std::vector<u64> row(b.words, 0);
        i32 id;
        if (k < special.size())
        {
            row = special[k];
            id = special_ids[k];
        }
        else
        {
            const u64 j = k - special.size();
            id = static_cast<i32>(j % instances);
            const u32 slots = table->task(static_cast<u32>(id))->atoms().fluent_slots();
            if (walk)
            {
                const u64* from = walks[static_cast<u32>(id)].data.data() + (j / instances) * b.words;
                std::copy(from, from + b.words, row.begin());
            }
            else
            {
                const double density = std::array{0.002, 0.01, 0.05}[rng() % 3];
                for (u32 s = 0; s < slots; ++s)
                    if (std::uniform_real_distribution<double>(0, 1)(rng) < density)
                        set_bit(row.data(), s);
                if (rng() % 2)
                    for (u32 w = table->instance(static_cast<u32>(id)).words; w < b.words; ++w)
                        row[w] = rng();
            }
        }
        b.states.insert(b.states.end(), row.begin(), row.end());
        b.task_ids.push_back(id);
    }
    b.count = count;
    // derived words and goals of the rows
    std::vector<u64> goal_pos(count * b.words, 0), goal_neg(count * b.words, 0);
    std::vector<u64> goal_derived_pos(count * b.derived_words, 0), goal_derived_neg(count * b.derived_words, 0);
    b.derived.assign(count * b.derived_words, 0);
    for (u64 g = 0; g < count; ++g)
    {
        const u32 instance = static_cast<u32>(b.task_ids[g]);
        const Task& task = *table->task(instance);
        const u64* row = b.states.data() + g * b.words;
        for (u32 s = 0; s < task.atoms().fluent_slots(); ++s)
        {
            if (rng() % 5)
                continue;
            set_bit(test_bit(row, s) ? goal_pos.data() + g * b.words : goal_neg.data() + g * b.words, s);
        }
        if (b.derived_words)
        {
            const auto derived = derived_words_of(*table, instance, row, b.derived_words);
            std::copy(derived.begin(), derived.end(), b.derived.begin() + static_cast<std::ptrdiff_t>(g * b.derived_words));
            for (u32 s = 0; s < task.atoms().derived_slots(); ++s)
            {
                if (rng() % 3)
                    continue;
                set_bit(test_bit(derived.data(), s) ? goal_derived_pos.data() + g * b.derived_words
                                                    : goal_derived_neg.data() + g * b.derived_words,
                        s);
            }
        }
    }
    b.goal_pos = std::move(goal_pos);
    b.goal_neg = std::move(goal_neg);
    b.goal_derived_pos = std::move(goal_derived_pos);
    b.goal_derived_neg = std::move(goal_derived_neg);
    return b;
}

/// Which optional inputs a call has.
struct Inputs
{
    bool derived = true;
    bool row_goals = false;
    bool derived_goals = false;
};

encoding::EncodeInput host_input(const Batch& b, u64 n, const Inputs& inputs)
{
    encoding::EncodeInput in;
    in.states = {b.states.data(), n, b.words, 0, 0};
    in.task_ids = b.task_ids.data();
    if (inputs.derived && b.derived_words)
    {
        in.derived = b.derived.data();
        in.derived_stride = b.derived_words;
    }
    if (inputs.row_goals)
    {
        in.goal_pos = b.goal_pos.data();
        in.goal_neg = b.goal_neg.data();
        in.goal_stride = b.words;
        in.goal_words = b.words;
        if (inputs.derived_goals && b.derived_words)
        {
            in.goal_derived_pos = b.goal_derived_pos.data();
            in.goal_derived_neg = b.goal_derived_neg.data();
            in.goal_derived_stride = b.derived_words;
            in.goal_derived_words = b.derived_words;
        }
    }
    return in;
}

struct Config
{
    const char* name;
    encoding::FlatOptions options;
    Inputs inputs;
};

using encoding::GoalDerivation;
using encoding::TargetSource;

/// The configurations the device encodes. Rows for the first one carry derived words whenever the table has derived atoms.
std::vector<Config> configs()
{
    const encoding::GoalDerivations all = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    std::vector<Config> out;
    auto add = [&](const char* name, auto&& change, Inputs inputs = {})
    {
        encoding::FlatOptions o = test::encoder_options();
        change(o);
        out.push_back({name, o, inputs});
    };
    add("state", [](auto&) {});
    add("goal_plain_satisfied", [](auto& o) { o.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied}; });
    add("goal_all_targets",
        [&](auto& o)
        {
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
        });
    add("virtual_nodes", [](auto& o) { o.predicate_virtual_nodes = true; });
    add("virtual_goal_targets",
        [&](auto& o)
        {
            o.predicate_virtual_nodes = true;
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
        });
    add("no_nullary_equality",
        [](auto& o)
        {
            o.nullary = false;
            o.equality = true;
        });
    add("no_static_no_derived",
        [](auto& o)
        {
            o.include_static = false;
            o.include_derived = false;
        });
    add("implicit_entities_no_counts",
        [](auto& o)
        {
            o.emit_counts = false;
            o.entity_arrays = encoding::EntityArrays::Auto;
        });
    add("row_goals", [&](auto& o) { o.goal_derivations = all; }, Inputs{true, true, false});
    add("row_goals_derived_targets_virtual",
        [&](auto& o)
        {
            o.goal_derivations = all;
            o.targets = {TargetSource::Goal};
            o.predicate_virtual_nodes = true;
        },
        Inputs{true, true, true});
    return out;
}

struct Corpus
{
    const char* name;
    rl::TaskTablePtr table;
    bool walk;
};

std::vector<Corpus> corpus()
{
    std::vector<Corpus> out;
    out.push_back({"blocks", table_of("blocks", {"smedium", "probBLOCKS-8-0"}), true});
    out.push_back({"features", table_of("features", {"p01", "p02", "p03", "p04"}), true});
    out.push_back({"goals", table_of("goals", {"conjunctive", "forall", "or", "exists", "not_and", "static_true"}), true});
    out.push_back({"philosophers", table_of("philosophers", {"p03-phil4", "p03-phil4-one-blocked"}), true});
    out.push_back({"gripper", table_of("gripper", {"p01", "p02"}), true});
    out.push_back({"wide", wide_table(), false});
    return out;
}

encoding::FlatOptions with(encoding::FlatOptions o, encoding::Layout layout, encoding::IndexType index)
{
    o.layout = layout;
    o.index = index;
    return o;
}


}
