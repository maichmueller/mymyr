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
inline rl::TaskTablePtr table_of(const std::string& domain, const std::vector<std::string>& problems)
{
    std::vector<TaskPtr> tasks;
    for (const std::string& problem : problems)
        tasks.push_back(test::pddl_task(domain, problem));
    return rl::TaskTable::create(tasks);
}

/// Instances of 1, 42 and 184 words: `link` over n objects has n * n slots, `flag` n more, and `armed` one.
inline rl::TaskTablePtr wide_table()
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
    // Subgoal layers: [count, layers, words] masks over each row's slots (derived: [count, layers, derived_words]);
    // the layers a row does not use are empty. Empty vectors: the batch has none.
    u32 layers = 0;
    std::vector<u64> layer_pos, layer_neg, layer_derived_pos, layer_derived_neg;
    // History: [count, steps, words] masks, the dt of each step and the number of valid steps of each row.
    u32 steps = 0;
    std::vector<u64> step_pos, step_neg, step_derived_pos, step_derived_neg;
    std::vector<i32> step_dt;     // [count, steps]
    std::vector<i32> step_count;  // [count]
};

inline void set_bit(u64* row, u32 slot) { row[slot >> 6] |= u64{1} << (slot & 63); }
inline bool test_bit(const u64* row, u32 slot) { return (row[slot >> 6] >> (slot & 63)) & 1u; }

/// All zeros, all ones within the instance's slots, single bits at 63, 64 and 65 and at the last slot, and the last
/// partial word set.
inline std::vector<std::vector<u64>> adversarial_rows(const rl::TaskTable& table, u32 instance)
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
inline std::vector<u64> derived_words_of(const rl::TaskTable& table, u32 instance, const u64* row, u32 width)
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
inline Batch make_batch(const rl::TaskTablePtr& table, u64 count, bool walk, u64 seed)
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

/// Adds `layers` subgoal layers and up to `steps` history steps to the rows: each row uses a random number of layers
/// (zero to all) and has a random number of valid steps (zero to all), and the masks are random literals over the
/// row's slots, some of them the row's goal literals and the literals of the layer below (so that the levels differ
/// from the layers). The words past a narrower instance's words are garbage in half of the rows.
inline void add_layered_rows(Batch& b, u32 layers, u32 steps, u64 seed)
{
    std::mt19937_64 rng(seed);
    const u64 n = b.count;
    b.layers = layers;
    b.steps = steps;
    const u32 W = b.words, D = b.derived_words;
    auto zeros = [](u64 size) { return std::vector<u64>(size, 0); };
    b.layer_pos = zeros(n * layers * W);
    b.layer_neg = zeros(n * layers * W);
    b.layer_derived_pos = zeros(n * layers * D);
    b.layer_derived_neg = zeros(n * layers * D);
    b.step_pos = zeros(n * steps * W);
    b.step_neg = zeros(n * steps * W);
    b.step_derived_pos = zeros(n * steps * D);
    b.step_derived_neg = zeros(n * steps * D);
    b.step_dt.assign(n * steps, 0);
    b.step_count.assign(n, 0);
    auto density = [&] { return std::array{0.03, 0.12, 0.4}[rng() % 3]; };
    auto draw = [&](u64* pos, u64* neg, u32 slots, const u64* below_pos, const u64* below_neg, const u64* goal_pos,
                    const u64* goal_neg, double d)
    {
        for (u32 slot = 0; slot < slots; ++slot)
        {
            const bool copied = (below_pos && test_bit(below_pos, slot) && rng() % 3 == 0) ||
                                (goal_pos && test_bit(goal_pos, slot) && rng() % 3 == 0);
            const bool copied_neg = (below_neg && test_bit(below_neg, slot) && rng() % 3 == 0) ||
                                    (goal_neg && test_bit(goal_neg, slot) && rng() % 3 == 0);
            if (copied)
                set_bit(pos, slot);
            else if (copied_neg)
                set_bit(neg, slot);
            else if (std::uniform_real_distribution<double>(0, 1)(rng) < d)
                set_bit(rng() % 2 ? pos : neg, slot);
        }
    };
    for (u64 g = 0; g < n; ++g)
    {
        const u32 instance = static_cast<u32>(b.task_ids[g]);
        const Task& task = *b.table->task(instance);
        const u32 slots = task.atoms().fluent_slots(), derived_slots = task.atoms().derived_slots();
        const u32 used = static_cast<u32>(rng() % (layers + 1));
        const u32 valid = static_cast<u32>(rng() % (steps + 1));
        b.step_count[g] = static_cast<i32>(valid);
        const u64* goal_pos = b.goal_pos.data() + g * W;
        const u64* goal_neg = b.goal_neg.data() + g * W;
        const u64* goal_derived_pos = D ? b.goal_derived_pos.data() + g * D : nullptr;
        const u64* goal_derived_neg = D ? b.goal_derived_neg.data() + g * D : nullptr;
        for (u32 l = 0; l < used; ++l)
        {
            const u64 at = (g * layers + l) * W, derived_at = (g * layers + l) * D;
            const double d = density();
            draw(b.layer_pos.data() + at, b.layer_neg.data() + at, slots,
                 l ? b.layer_pos.data() + at - W : nullptr, l ? b.layer_neg.data() + at - W : nullptr, goal_pos, goal_neg, d);
            if (D)
                draw(b.layer_derived_pos.data() + derived_at, b.layer_derived_neg.data() + derived_at, derived_slots,
                     l ? b.layer_derived_pos.data() + derived_at - D : nullptr,
                     l ? b.layer_derived_neg.data() + derived_at - D : nullptr, goal_derived_pos, goal_derived_neg, d);
        }
        for (u32 j = 0; j < steps; ++j)
        {
            const u64 at = (g * steps + j) * W, derived_at = (g * steps + j) * D;
            const double d = density();
            draw(b.step_pos.data() + at, b.step_neg.data() + at, slots, nullptr, nullptr, nullptr, nullptr, d);
            if (D)
                draw(b.step_derived_pos.data() + derived_at, b.step_derived_neg.data() + derived_at, derived_slots,
                     nullptr, nullptr, nullptr, nullptr, d);
            b.step_dt[g * steps + j] = static_cast<i32>(rng() % 61) - 20;
        }
        if (rng() % 2)
        {
            // garbage past the instance's words
            const u32 words = b.table->instance(instance).words;
            for (u32 l = 0; l < layers; ++l)
                for (u32 w = words; w < W; ++w)
                    b.layer_pos[(g * layers + l) * W + w] = b.layer_neg[(g * layers + l) * W + w] = rng();
            for (u32 j = 0; j < steps; ++j)
                for (u32 w = words; w < W; ++w)
                    b.step_pos[(g * steps + j) * W + w] = b.step_neg[(g * steps + j) * W + w] = rng();
        }
    }
}

/// Which optional inputs a call has.
struct Inputs
{
    bool derived = true;
    bool row_goals = false;
    bool derived_goals = false;
    u32 layers = 0;              // the subgoal layers given (at most the batch's)
    bool history = false;        // the history steps given
    bool history_count = false;  // with the valid steps of each row (else every step is valid)
};

/// The first `layers` subgoal layers of a batch over the arrays `pos`, `neg` (and the derived ones, which the batch may
/// not have) in the batch's layout; host or device pointers.
inline encoding::LayerInput layer_view(const Batch& b, u32 layers, const u64* pos, const u64* neg, const u64* derived_pos,
                                const u64* derived_neg)
{
    encoding::LayerInput in;
    in.pos = pos;
    in.neg = neg;
    in.layers = layers;
    in.words = b.words;
    in.stride = u64{b.layers} * b.words;
    if (b.derived_words)
    {
        in.derived_pos = derived_pos;
        in.derived_neg = derived_neg;
        in.derived_words = b.derived_words;
        in.derived_stride = u64{b.layers} * b.derived_words;
    }
    return in;
}

/// The history of a batch.
inline encoding::HistoryInput history_view(const Batch& b, const Inputs& inputs, const u64* pos, const u64* neg,
                                    const u64* derived_pos, const u64* derived_neg, const i32* dt, const i32* count)
{
    encoding::HistoryInput in;
    in.masks.pos = pos;
    in.masks.neg = neg;
    in.masks.layers = b.steps;
    in.masks.words = b.words;
    in.masks.stride = u64{b.steps} * b.words;
    if (b.derived_words)
    {
        in.masks.derived_pos = derived_pos;
        in.masks.derived_neg = derived_neg;
        in.masks.derived_words = b.derived_words;
        in.masks.derived_stride = u64{b.steps} * b.derived_words;
    }
    in.dt = dt;
    in.count = inputs.history_count ? count : nullptr;
    return in;
}

inline encoding::EncodeInput host_input(const Batch& b, u64 n, const Inputs& inputs)
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
    if (inputs.layers)
        in.subgoals = layer_view(b, inputs.layers, b.layer_pos.data(), b.layer_neg.data(), b.layer_derived_pos.data(),
                                 b.layer_derived_neg.data());
    if (inputs.history)
        in.history = history_view(b, inputs, b.step_pos.data(), b.step_neg.data(), b.step_derived_pos.data(),
                                  b.step_derived_neg.data(), b.step_dt.data(), b.step_count.data());
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
inline std::vector<Config> configs()
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

/// The configurations with subgoal levels and history steps, on the batches `add_layered_rows(batch, 3, 4, ...)` makes.
inline std::vector<Config> layered_configs()
{
    const encoding::GoalDerivations all = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    std::vector<Config> out;
    auto add = [&](const char* name, auto&& change, Inputs inputs)
    {
        encoding::FlatOptions o = test::encoder_options();
        o.goal_derivations = all;
        change(o);
        out.push_back({name, o, inputs});
    };
    Inputs layers;
    layers.layers = 1;
    add("one_level", [](auto& o) { o.max_goal_level = 1; }, layers);
    layers.layers = 3;
    add("three_levels_targets",
        [](auto& o)
        {
            o.max_goal_level = 3;
            o.targets = {TargetSource::Goal, TargetSource::Subgoal};
        },
        layers);
    // the layers given are fewer than the levels the schema has
    add("two_layers_of_three_levels_row_goals",
        [](auto& o)
        {
            o.max_goal_level = 3;
            o.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied};
            o.predicate_virtual_nodes = true;
        },
        Inputs{true, true, false, 2});
    add("levels_row_goals_derived_targets_virtual",
        [](auto& o)
        {
            o.max_goal_level = 2;
            o.targets = {TargetSource::Goal, TargetSource::Subgoal};
            o.predicate_virtual_nodes = true;
        },
        Inputs{true, true, true, 2});
    Inputs history;
    history.history = true;
    history.history_count = true;
    add("history", [](auto& o) { o.history = true; }, history);
    add("history_every_step_valid", [](auto& o) { o.history = true; }, Inputs{true, false, false, 0, true, false});
    add("history_newest_two_targets",
        [](auto& o)
        {
            o.history = true;
            o.history_max_steps = 2;
            o.targets = {TargetSource::History};
        },
        history);
    add("history_targets_with_goal_targets_virtual",
        [](auto& o)
        {
            o.history = true;
            o.history_max_steps = 3;
            o.targets = {TargetSource::Goal, TargetSource::History};
            o.predicate_virtual_nodes = true;
        },
        history);
    add("history_no_static_no_nullary",
        [](auto& o)
        {
            o.history = true;
            o.include_static = false;
            o.nullary = false;
            o.emit_counts = false;
            o.entity_arrays = encoding::EntityArrays::Auto;
        },
        history);
    add("levels_and_history",
        [](auto& o)
        {
            o.max_goal_level = 3;
            o.history = true;
            o.history_max_steps = 2;
            o.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::History};
            o.predicate_virtual_nodes = true;
        },
        Inputs{true, true, true, 3, true, true});
    add("levels_and_history_without_row_goals",
        [](auto& o)
        {
            o.max_goal_level = 2;
            o.history = true;
            o.targets = {TargetSource::Subgoal, TargetSource::History, TargetSource::Goal};
        },
        Inputs{true, false, false, 2, true, true});
    return out;
}

struct Corpus
{
    const char* name;
    rl::TaskTablePtr table;
    bool walk;
};

inline std::vector<Corpus> corpus()
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

/// Tables with numeric functions: two instances of different sizes with i32 storage, three with f64 storage, and one
/// instance whose functions have several arities.
inline std::vector<Corpus> numeric_corpus()
{
    using Mode = TaskOptions::NumericStorageMode;
    auto counters = [](const char* problem, Mode mode)
    {
        const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
        TaskOptions options;
        options.numeric_storage = mode;
        return Task::create(*frontend::load_task(dir + "domain.pddl", dir + problem + ".pddl"), options);
    };
    const std::string drone = std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-drone.txt";
    std::vector<Corpus> out;
    out.push_back({"counters_i32", rl::TaskTable::create({counters("p01", Mode::Auto), counters("p02", Mode::Auto)}), true});
    out.push_back({"counters_f64",
                   rl::TaskTable::create({counters("p01", Mode::F64), counters("p03-half", Mode::F64),
                                          counters("p02", Mode::F64)}),
                   true});
    out.push_back({"drone", rl::TaskTable::single(Task::from_text_file(drone)), true});
    return out;
}

/// `per_instance` walk rows of each instance, interleaved, with their numeric words.
inline test::Rows walk_rows(const rl::TaskTable& table, u64 per_instance, u64 seed)
{
    std::vector<test::Rows> sources;
    for (u32 i = 0; i < table.size(); ++i)
        sources.push_back(test::random_walk(table, i, per_instance, seed + i));
    test::Rows out;
    out.words = table.words();
    out.numeric_words = table.numeric_words();
    const u32 width = table.row_words();
    for (u64 k = 0; k < per_instance; ++k)
        for (const auto& source : sources)
        {
            out.data.insert(out.data.end(), source.data.begin() + static_cast<std::ptrdiff_t>(k * width),
                            source.data.begin() + static_cast<std::ptrdiff_t>((k + 1) * width));
            out.task_ids.push_back(source.task_ids[k]);
        }
    return out;
}

inline encoding::EncodeInput input_of(const test::Rows& rows)
{
    encoding::EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    return in;
}

inline encoding::FlatOptions with(encoding::FlatOptions o, encoding::Layout layout, encoding::IndexType index)
{
    o.layout = layout;
    o.index = index;
    return o;
}


}
