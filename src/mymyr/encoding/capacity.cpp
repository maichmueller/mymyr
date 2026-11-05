#include "mymyr/encoding/capacity.hpp"

#include "mymyr/encoding/detail/lanes.hpp"
#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void invalid(const std::string& what) { throw std::invalid_argument("mymyr: encoding: " + what); }

// saturating at k_unbounded, which stays unbounded
u64 sat_add(u64 a, u64 b) { return a > k_unbounded - b ? k_unbounded : a + b; }
u64 sat_mul(u64 a, u64 b)
{
    if (a == 0 || b == 0)
        return 0;
    return a > k_unbounded / b ? k_unbounded : a * b;
}

/// Per action schema of `task`, its ground actions that the delete relaxation reaches and whose static preconditions
/// hold; nullopt (and `reason`) when the grounding exceeds its budget.
std::optional<std::vector<u64>> reachable_actions(const Task& task, std::string& reason)
{
    heuristics::GroundingStats stats;
    const auto relaxed = heuristics::RelaxedTask::build(task, {}, &stats);
    if (!relaxed)
    {
        reason = stats.reason.empty() ? std::string("the grounding exceeds its budget") : stats.reason;
        return std::nullopt;
    }
    std::vector<u64> out(task.data().schemas.size(), 0);
    for (u32 ga = 0; ga < relaxed->num_ground_actions(); ++ga)
        ++out[relaxed->ground_action_label(ga).schema.v];
    return out;
}

u64 history_steps(const EncodingSchema& schema, u64 given)
{
    if (!schema.has_history())
        return 0;
    if (const std::optional<u32> limit = schema.options().history_max_steps)
        return given ? std::min<u64>(given, *limit) : *limit;
    if (given == 0)
        invalid("the schema encodes every history step: give the most steps one row's history has");
    return given;
}
}  // namespace

CapacityBounds capacity_bounds(const EncodingContext& context, const CapacityBoundsOptions& options)
{
    namespace ed = detail;
    const EncodingSchema& schema = context.schema();
    if (schema.has_horizon())
        invalid("capacity bounds need a schema without a horizon: a horizon's graphs grow with its nodes");
    if (options.graphs == 0)
        invalid("capacity bounds need at least one graph");
    const u64 steps = history_steps(schema, options.history_steps);
    const bool actions = options.actions && schema.options().include_actions;
    const bool levels = schema.has_subgoals(), history = schema.has_history();
    const ed::ContextView view = context.view();
    const rl::TaskTable& table = context.table();
    const u32 relations = schema.num_relations();

    // the bounds of one graph, the largest over the instances and the cell lists
    std::vector<u64> tuples(relations, 0);
    std::vector<u8> inexact(relations, 0);
    u64 entities = 0, args = 0, targets = 0, numeric = 0, successors = 0;
    std::vector<u64> blocks;  // the static block arguments of each instance (shared statics)
    CapacityBounds out;
    for (u32 i = 0; i < table.size(); ++i)
    {
        const ed::InstanceView inst = ed::instance_view(view, i);
        std::optional<std::vector<u64>> ground;
        if (actions)
        {
            std::string reason;
            ground = reachable_actions(*table.instance(i).task, reason);
            if (!ground && out.unbounded_reason.empty())
                out.unbounded_reason = "instance " + std::to_string(i) + ": " + reason;
            u64 rows = ground ? 0 : k_unbounded;
            if (ground)
                for (u64 n : *ground)
                    rows = sat_add(rows, n);
            successors = std::max(successors, rows);
        }
        if (schema.stores_numeric_values())
            numeric = std::max<u64>(numeric, ed::numeric_slot_count(inst));
        if (schema.shared_statics())
            blocks.push_back(ed::static_block_totals(inst).args);
        for (const bool row_goals : {false, true})
        {
            const ed::CellList list = ed::cell_list(inst, row_goals, actions, levels, history);
            u64 graph_args = 0, graph_targets = 0, extra_entities = 0;
            bool nullary = false;
            for (u32 c = 0; c < list.count; ++c)
            {
                const ed::Cell& cell = list.cells[c];
                u64 n = 0;
                if (ed::is_action_op(cell.op))
                {
                    // one tuple, one action entity and, with action targets, one target per distinct ground action
                    for (u32 k = cell.a; k < cell.b; ++k)
                        n = sat_add(n, ground ? (*ground)[inst.action_schemas[k]] : k_unbounded);
                    extra_entities = sat_add(extra_entities, n);
                    if (cell.op == static_cast<u32>(ed::CellOp::ActionTarget))
                        graph_targets = sat_add(graph_targets, n);
                }
                else if (ed::is_history_op(cell.op))
                {
                    // every step may hold a literal in each slot of the cell
                    n = sat_mul(steps, cell.b - cell.a);
                    if (ed::history_op_target(cell.op))
                    {
                        graph_targets = sat_add(graph_targets, n);
                        extra_entities = sat_add(extra_entities, n);
                    }
                    nullary = nullary || (n != 0 && cell.logical_arity == 0);
                }
                else
                {
                    // a goal cell of the instance's goal also lists the goal's literals of fixed truth
                    const bool instance_goal = ed::is_goal_op(cell.op) && !row_goals && ed::goal_op_level(cell.op) == 0;
                    n = u64{cell.b - cell.a} + (instance_goal ? cell.d - cell.c : 0u);
                    if (ed::is_goal_op(cell.op) && ed::goal_op_mode(cell.op) == ed::GoalMode::PlainTarget)
                    {
                        graph_targets = sat_add(graph_targets, n);
                        extra_entities = sat_add(extra_entities, n);
                    }
                    nullary = nullary || (ed::cell_present(cell, static_cast<u32>(n)) && cell.logical_arity == 0);
                }
                const bool fixed = cell.op == static_cast<u32>(ed::CellOp::Static) || ed::is_numeric_op(cell.op);
                inexact[cell.relation] = inexact[cell.relation] || !fixed;
                tuples[cell.relation] = std::max(tuples[cell.relation], n);
                graph_args = sat_add(graph_args, sat_mul(n, cell.arity));
            }
            // objects, the auxiliary entity, the predicate virtual nodes, one entity per history step, and the
            // target and action entities
            u64 graph_entities = u64{inst.num_objects} + inst.num_virtual + steps;
            graph_entities += nullary && inst.nullary_auxiliary ? 1 : 0;
            graph_entities = sat_add(graph_entities, extra_entities);
            entities = std::max(entities, graph_entities);
            args = std::max(args, graph_args);
            targets = std::max(targets, graph_targets);
        }
    }

    const u64 graphs = options.graphs;
    out.graphs = graphs;
    out.entities = sat_mul(graphs, entities);
    out.args = sat_mul(graphs, args);
    out.targets = sat_mul(graphs, targets);
    out.history = sat_mul(graphs, steps);
    out.numeric = sat_mul(graphs, numeric);
    out.successors = actions ? sat_mul(graphs, successors) : 0;
    if (schema.shared_statics())
    {
        // the largest blocks, one per instance among the rows
        std::ranges::sort(blocks, std::greater<>());
        out.static_instances = std::min<u64>(graphs, blocks.size());
        for (u64 k = 0; k < out.static_instances; ++k)
            out.static_args = sat_add(out.static_args, blocks[k]);
    }
    out.relation_tuples.resize(relations);
    out.relation_exact.resize(relations);
    for (u32 r = 0; r < relations; ++r)
    {
        out.relation_tuples[r] = sat_mul(graphs, tuples[r]);
        out.relation_exact[r] = inexact[r] ? 0 : 1;
    }
    return out;
}
}  // namespace mymyr::encoding
