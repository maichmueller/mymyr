#include "mymyr/encoding/context.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/encoding/detail/horizon.hpp"
#include "mymyr/encoding/detail/literals.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <string>

namespace mymyr::encoding
{
namespace
{
using detail::Cell;
using detail::CellOp;

bool is_state_relation(const Relation& relation)
{
    const bool state_lane = relation.lane == Lane::Fluent || relation.lane == Lane::Static || relation.lane == Lane::Derived;
    return state_lane && relation.sign == 0 && !relation.level && !relation.derivation && !relation.state_anchored &&
           relation.predicate >= 0;
}

/// Whether the first slot of relation `id` has role `role` (a relation of arity 0 has no slot).
bool leads_with(const EncodingSchema& schema, u32 id, SlotRole role)
{
    const auto roles = schema.slot_roles(id);
    return !roles.empty() && roles.front() == role;
}

std::span<const i32> ints(const rl::ArrayBundle& bundle, const char* name)
{
    const rl::ArrayInfo* info = bundle.find(name);
    return {static_cast<const i32*>(bundle.data(*info)), info->elements()};
}

/// The goal relations of the schema: for every domain predicate, the relation of its goal literals of each sign
/// (positive, negative), as the plain, satisfied and unsatisfied variants (-1: the schema has none).
struct GoalRelations
{
    std::vector<i32> id;     // [predicate, sign, variant]
    std::vector<u8> target;  // [relation]: the plain relation carries a goal target slot

    [[nodiscard]] i32 at(u32 predicate, u32 sign, u32 variant) const { return id[(u64{predicate} * 2 + sign) * 3 + variant]; }
};

/// The subgoal and history relations of the schema, by predicate: for each of the levels 1 to 3 the relations of the
/// goal variants of each sign (plain, satisfied, unsatisfied), and the history relation of each sign (-1: the schema
/// has none).
struct LayerRelations
{
    std::vector<i32> subgoal;  // [level - 1, predicate, sign, variant]
    std::vector<i32> history;  // [predicate, sign]
    std::vector<u8> target;    // [relation]: the plain subgoal / history relation carries a target slot

    [[nodiscard]] i32 subgoal_at(u32 level, u32 predicate, u32 sign, u32 variant) const
    {
        const u64 per_level = subgoal.size() / 3;
        return subgoal[u64{level - 1} * per_level + (u64{predicate} * 2 + sign) * 3 + variant];
    }
    [[nodiscard]] i32 history_at(u32 predicate, u32 sign) const { return history[u64{predicate} * 2 + sign]; }
};

/// What one relation of a horizon schema lists, for the relations of a predicate.
struct HorizonRelation
{
    u32 relation;
    detail::HorizonKind kind;
    detail::HorizonScope scope;
    bool lead;      // each tuple starts with the entity of its state node
    bool minus;     // Delta: the atoms the node deletes; Added: the literals it falsifies
    bool negative;  // Goal, Sat, Added: the negative literals
    u32 level;      // Goal, Sat, Added: the goal level
    Lane lane;
    detail::GoalMode mode;  // Goal, Sat, Added: the goal op's mode
};

/// The horizon relations of a schema's predicates (by domain predicate id), and the cells of its edge relations.
struct HorizonRelations
{
    std::vector<std::vector<HorizonRelation>> by_predicate;
    std::vector<u8> goal;  // [predicate, sign]: some relation lists literals of the goal (level 0) for them
    std::vector<Cell> edge_cells;
};

/// The relation `id` of a horizon schema as its cells are made; none for the edge relations.
std::optional<HorizonRelation> describe_horizon(const EncodingSchema& schema, u32 id)
{
    using detail::HorizonKind;
    using detail::HorizonScope;
    const Relation& r = schema.relation(id);
    const FlatOptions& o = schema.options();
    if (r.predicate < 0)
        return std::nullopt;
    const bool split = o.horizon.mode == HorizonMode::Full && o.horizon.root == RootPolicy::Exclude;
    const bool nodes_too = o.horizon.mode == HorizonMode::Full && !split;
    const auto roles = schema.slot_roles(id);
    HorizonRelation d{id, HorizonKind::Facts, HorizonScope::Root, !roles.empty() && roles.front() == SlotRole::State,
                      false, r.sign < 0, r.level.value_or(0), r.lane, detail::GoalMode::Plain};
    switch (r.lane)
    {
    case Lane::Fluent:
    case Lane::Static:
    case Lane::Derived:
        if (r.sign != 0)
        {
            d.kind = HorizonKind::Delta;
            d.scope = HorizonScope::Nodes;
            d.minus = r.sign < 0;
        }
        else if (r.state_anchored)
            d.scope = HorizonScope::Nodes;
        else if (nodes_too && r.lane != Lane::Static)
            d.scope = HorizonScope::All;
        return d;
    case Lane::Goal:
        d.kind = HorizonKind::Goal;
        return d;
    case Lane::GoalSatisfaction:
        if (*r.derivation == GoalDerivation::Satisfied || *r.derivation == GoalDerivation::Unsatisfied)
        {
            d.kind = HorizonKind::Sat;
            d.mode = *r.derivation == GoalDerivation::Satisfied ? detail::GoalMode::Satisfied
                                                                  : detail::GoalMode::Unsatisfied;
            d.scope = r.state_anchored ? HorizonScope::Nodes : nodes_too ? HorizonScope::All : HorizonScope::Root;
        }
        else
        {
            d.kind = HorizonKind::Added;
            d.scope = HorizonScope::Nodes;
            d.minus = *r.derivation == GoalDerivation::AddedUnsatisfied;
        }
        return d;
    default:
        return std::nullopt;
    }
}

/// Bits of the per-instance flags that tell whether the derived atoms of a row are needed.
constexpr u8 k_needs_state = detail::k_needs_state_derived;
constexpr u8 k_needs_instance_goal = detail::k_needs_goal_derived;
constexpr u8 k_needs_row_goal = detail::k_needs_row_goal_derived;

/// What one instance contributes to the arrays.
struct InstanceRows
{
    std::vector<Cell> cells;      // with the instance's goal: the goal cells that can hold a tuple, narrowed
    std::vector<Cell> row_cells;  // with the goals of a row: the goal cells of every predicate that has slots
    std::vector<Cell> plain_cells, plain_row_cells;  // the same two lists without the action cells
    std::array<std::vector<Cell>, 12> extended;      // the four lists with the cells of subgoal layers, of history
                                                     // steps, and of both: index (e - 1) * 4 + k for e (1: layers,
                                                     // 2: history, 3: both) and the list k of {cells, row_cells,
                                                     // plain_cells, plain_row_cells}
    std::array<std::vector<Cell>, 16> fixed;         // the 16 lists for rows with constant literals: index e * 4 + k,
                                                     // the list k (e 0) or the extended list, with the goal cells of
                                                     // the predicates without slots (detail::cell_list)
    std::vector<i32> slot_args, derived_args, static_args;  // rows of max_arity objects, -1 padded
    std::vector<i32> numeric_args;                          // the same for the numeric slots
    std::vector<Cell> numeric_cells;                        // one per numeric function that has slots
    u64 numeric_tuples = 0;
    std::vector<u64> goal_words;                            // goal masks: pos, neg (words each), derived pos, neg
    std::vector<i32> goal_constants;  // the constant goal literals, rows of detail::k_constant_objects + max_arity,
                                      // then the rows of the goal cells (InstanceView::goal_constants)
    u64 goal_constant_count = 0;      // the constant goal literals
    std::vector<u32> literal_ranges;                        // [domain predicates, detail::k_literal_columns]
    u64 num_objects = 0, words = 0, fluent_slots = 0, derived_words = 0, derived_slots = 0;
    u64 fingerprint = 0;
    u64 goal_complete = 1;
    u8 needs = 0;
};

/// A goal literal whose truth no state changes: over a static predicate, or over an atom that has no slot.
struct ConstantLiteral
{
    std::vector<i32> objects;
    bool holds;
};

bool before(const std::vector<i32>& a, const std::vector<i32>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

/// Narrows the slot range [begin, end) to the slots from the first to the last set bit of `mask` in it ([0, 0) if there
/// is none).
void narrow_to_set_bits(const u64* mask, u32& begin, u32& end)
{
    u32 first = end, last = begin;
    for (u32 w = begin >> 6; begin < end && w <= ((end - 1) >> 6); ++w)
    {
        const u64 bits = detail::range_word(mask, w, begin, end);
        if (!bits)
            continue;
        if (first == end)
            first = w * 64u + detail::lowest_bit64(bits);
        last = w * 64u + static_cast<u32>(std::bit_width(bits));
    }
    if (first == end)
        begin = end = 0;
    else
    {
        begin = first;
        end = last;
    }
}

/// The objects of every slot of `pred` / `padded`, as rows of max_arity entries.
void append_slot_rows(std::vector<i32>& to, std::span<const i32> pred, std::span<const i32> padded, u32 width,
                      std::span<const i32> arity, u32 max_arity)
{
    for (u64 slot = 0; slot < pred.size(); ++slot)
    {
        const u32 logical = static_cast<u32>(arity[static_cast<u32>(pred[slot])]);
        for (u32 k = 0; k < max_arity; ++k)
            to.push_back(k < logical ? padded[slot * width + k] : -1);
    }
}

/// The slot of the atom `objects` among the slots [begin, end) of `padded` (rows of `width` objects, ascending), if any.
std::optional<u32> find_slot(std::span<const i32> padded, u32 width, u32 begin, u32 end, const std::vector<i32>& objects)
{
    const u32 arity = static_cast<u32>(objects.size());
    u32 lo = begin, hi = end;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2;
        const i32* row = padded.data() + static_cast<u64>(mid) * width;
        if (std::lexicographical_compare(row, row + arity, objects.begin(), objects.end()))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < end && std::equal(objects.begin(), objects.end(), padded.data() + static_cast<u64>(lo) * width))
        return lo;
    return std::nullopt;
}

/// The cells of an instance's numeric slots, one per numeric relation of the schema whose function has slots in the
/// instance, and the objects of every slot as rows of `max_arity` entries into `numeric_args`. The slots of a function
/// are consecutive and ascend in their objects.
void build_numeric_cells(const Task& task, const EncodingSchema& schema, u32 max_arity, std::vector<Cell>& cells,
                         std::vector<i32>& numeric_args)
{
    const plan::Numeric& numeric = task.numeric();
    if (!schema.has_numeric() || numeric.slots == 0)
        return;
    auto broken = [&](const std::string& what)
    { throw std::logic_error("mymyr: encoding: the numeric slots of the task are not in function order: " + what); };
    for (u32 slot = 0; slot < numeric.slots; ++slot)
    {
        const u32 begin = numeric.slot_args_begin[slot], end = numeric.slot_args_begin[slot + 1];
        for (u32 k = 0; k < max_arity; ++k)
            numeric_args.push_back(k < end - begin ? static_cast<i32>(numeric.slot_args[begin + k].v) : -1);
    }
    for (u32 id = schema.first_numeric_relation(); id < schema.num_relations(); ++id)
    {
        const Relation& relation = schema.relation(id);
        const u32 function = static_cast<u32>(relation.function);
        u32 first = numeric.slots, last = numeric.slots;
        for (u32 slot = 0; slot < numeric.slots; ++slot)
            if (numeric.slot_function[slot] == function)
            {
                if (first == numeric.slots)
                    first = slot;
                else if (last != slot)
                    broken(relation.name);
                last = slot + 1;
            }
        if (first == numeric.slots)
            continue;
        for (u32 slot = first + 1; slot < last; ++slot)
        {
            const i32* a = numeric_args.data() + u64{slot - 1} * max_arity;
            const i32* b = numeric_args.data() + u64{slot} * max_arity;
            if (!std::lexicographical_compare(a, a + relation.logical_arity, b, b + relation.logical_arity))
                broken(relation.name);
        }
        cells.push_back({id, static_cast<u32>(CellOp::Numeric), first, last, relation.arity, relation.logical_arity});
    }
}

InstanceRows build_instance(const rl::TaskTable& table, u32 index, const std::vector<i32>& state_relation,
                            const GoalRelations& goal_relations, const LayerRelations& layer_relations,
                            const std::vector<Cell>& action_cells, const EncodingSchema& schema, u32 max_arity,
                            const HorizonRelations* horizon)
{
    const rl::TaskTable::Instance& instance = table.instance(index);
    const Task& task = *instance.task;
    const rl::ArrayBundle meta = rl::atom_metadata(task);
    const auto atom_pred = ints(meta, "atom_pred");
    const auto atom_padded = ints(meta, "atom_args_padded");
    const auto derived_pred = ints(meta, "derived_pred");
    const auto derived_padded = ints(meta, "derived_args_padded");
    const auto fluent_offsets = ints(meta, "pred_slot_offsets");
    const auto derived_offsets = ints(meta, "derived_pred_slot_offsets");
    const auto static_pred = ints(meta, "static_pred");
    const auto static_padded = ints(meta, "static_args_padded");
    const auto arity = ints(meta, "pred_arity");
    const auto kind = ints(meta, "pred_kind");
    const u32 predicates = static_cast<u32>(arity.size());
    auto width_of = [](std::span<const i32> pred, std::span<const i32> padded)
    { return static_cast<u32>(pred.empty() ? 0 : padded.size() / pred.size()); };
    const u32 fluent_width = width_of(atom_pred, atom_padded);
    const u32 derived_width = width_of(derived_pred, derived_padded);
    const u32 static_width = width_of(static_pred, static_padded);

    InstanceRows out;
    out.num_objects = instance.num_objects;
    out.words = instance.words;
    out.fluent_slots = task.atoms().fluent_slots();
    out.derived_slots = task.atoms().derived_slots();
    out.derived_words = bits::words_for(out.derived_slots);
    out.fingerprint = task.encoding_fingerprint();
    u32 static_rows = 0;
    // the sorted, deduplicated static tuples of predicate `ip`, appended to the instance's static rows: [first, last)
    auto append_static_rows = [&](u32 ip, u32 logical)
    {
        std::vector<std::vector<i32>> rows;
        for (u64 s = 0; s < static_pred.size(); ++s)
            if (static_cast<u32>(static_pred[s]) == ip)
                rows.emplace_back(static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width),
                                  static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width + logical));
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        const u32 first = static_rows;
        for (const std::vector<i32>& row : rows)
        {
            for (u32 k = 0; k < max_arity; ++k)
                out.static_args.push_back(k < logical ? row[k] : -1);
            ++static_rows;
        }
        return std::pair{first, static_rows};
    };
    // the static rows of each predicate, appended once: the cells and the literal section share them
    std::vector<std::optional<std::pair<u32, u32>>> static_ranges(predicates);
    auto static_rows_of = [&](u32 ip, u32 logical)
    {
        if (!static_ranges[ip])
            static_ranges[ip] = append_static_rows(ip, logical);
        return *static_ranges[ip];
    };
    // The literal section: where the atoms of every domain predicate are, whatever the options leave out of the cells.
    auto fill_literal_ranges = [&]
    { out.literal_ranges = detail::build_literal_ranges(table, index, meta, static_rows_of); };
    for (u32 ip = 0; ip < predicates && !horizon; ++ip)
    {
        const u32 tp = instance.pred_map[ip];
        if (tp >= table.num_domain_predicates() || state_relation[tp] < 0)
            continue;
        const u32 relation_id = static_cast<u32>(state_relation[tp]);
        const Relation& relation = schema.relation(relation_id);
        const u32 logical = static_cast<u32>(arity[ip]);
        const Lane lane = kind[ip] == rl::k_pred_static ? Lane::Static : kind[ip] == rl::k_pred_fluent ? Lane::Fluent : Lane::Derived;
        const u32 arguments = logical != 0 || schema.options().nullary == Nullary::Auxiliary ? std::max(logical, 1u) : 0u;
        if (relation.lane != lane || relation.logical_arity != logical ||
            relation.arity != arguments + (schema.options().predicate_virtual_nodes ? 1u : 0u))
            throw std::logic_error("mymyr: encoding: the relation " + relation.name + " does not match its predicate");
        Cell cell{relation_id, 0, 0, 0, relation.arity, logical};
        if (lane == Lane::Fluent)
        {
            cell.op = static_cast<u32>(CellOp::Fluent);
            cell.a = static_cast<u32>(fluent_offsets[ip]);
            cell.b = static_cast<u32>(fluent_offsets[ip + 1]);
            detail::check_slot_range(atom_pred, atom_padded, fluent_width, cell.a, cell.b, ip, logical, relation.name);
        }
        else if (lane == Lane::Derived)
        {
            cell.op = static_cast<u32>(CellOp::Derived);
            cell.a = static_cast<u32>(derived_offsets[ip]);
            cell.b = static_cast<u32>(derived_offsets[ip + 1]);
            detail::check_slot_range(derived_pred, derived_padded, derived_width, cell.a, cell.b, ip, logical, relation.name);
        }
        else
        {
            cell.op = static_cast<u32>(CellOp::Static);
            std::tie(cell.a, cell.b) = static_rows_of(ip, logical);
            if (schema.shared_statics())
            {
                // the tuples go to the static block of the batch: the cell has no row in a graph
                cell.c = cell.a;
                cell.d = cell.b;
                cell.a = cell.b = 0;
            }
        }
        if (cell.a != cell.b || cell.c != cell.d)
            out.cells.push_back(cell);
    }
    for (const Cell& cell : out.cells)
        if (cell.op == static_cast<u32>(CellOp::Derived))
            out.needs |= k_needs_state;

    // The goal: masks over the slots of the goal's atoms, and the literals whose truth is fixed as constant tuples.
    out.goal_complete = task.encoding_goal_complete() ? 1 : 0;
    const u32 W = static_cast<u32>(out.words), D = static_cast<u32>(out.derived_words);
    out.goal_words.assign(u64{2} * W + u64{2} * D, 0);
    u64* masks[2][2] = {{out.goal_words.data(), out.goal_words.data() + W},
                        {out.goal_words.data() + 2 * u64{W}, out.goal_words.data() + 2 * u64{W} + D}};  // [derived][sign]
    std::map<std::pair<u32, u32>, std::vector<ConstantLiteral>> constants;  // (predicate, sign)
    std::vector<std::vector<std::vector<i32>>> static_atoms;
    auto static_holds = [&](u32 ip, const std::vector<i32>& objects)
    {
        if (static_atoms.empty())
        {
            static_atoms.resize(predicates);
            for (u64 s = 0; s < static_pred.size(); ++s)
            {
                const u32 p = static_cast<u32>(static_pred[s]);
                static_atoms[p].emplace_back(static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width),
                                            static_padded.begin() + static_cast<std::ptrdiff_t>(s * static_width + arity[p]));
            }
            for (auto& rows : static_atoms)
            {
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            }
        }
        return std::binary_search(static_atoms[ip].begin(), static_atoms[ip].end(), objects);
    };
    const formalism::TaskData& data = task.data();
    for (const formalism::Literal& literal : task.encoding_goal())
    {
        const u32 ip = literal.pred.v;
        // A literal the encoding leaves out (a predicate without a goal relation: nullary and equality predicates can
        // be left out) makes the goal incomplete, unless its truth is fixed (a static predicate).
        if (ip >= predicates || instance.pred_map[ip] >= table.num_domain_predicates())
        {
            out.goal_complete = 0;
            continue;
        }
        const u32 tp = instance.pred_map[ip];
        const u32 sign = literal.positive ? 0 : 1;
        const bool has_goal_relation = horizon ? horizon->goal[u64{tp} * 2 + sign] != 0
                                               : goal_relations.at(tp, sign, 0) >= 0 ||
                                                     goal_relations.at(tp, sign, 1) >= 0 ||
                                                     goal_relations.at(tp, sign, 2) >= 0;
        if (!has_goal_relation)
        {
            if (kind[ip] != rl::k_pred_static)
                out.goal_complete = 0;
            continue;
        }
        std::vector<i32> objects;
        for (formalism::Term term : data.terms_of(literal))
            objects.push_back(static_cast<i32>(formalism::term_object(term).v));
        const bool derived = kind[ip] == rl::k_pred_derived;
        bool holds = false, slotted = false;
        if (kind[ip] == rl::k_pred_static)
            holds = static_holds(ip, objects) == literal.positive;
        else
        {
            const auto slot = derived ? find_slot(derived_padded, derived_width, static_cast<u32>(derived_offsets[ip]),
                                                  static_cast<u32>(derived_offsets[ip + 1]), objects)
                                      : find_slot(atom_padded, fluent_width, static_cast<u32>(fluent_offsets[ip]),
                                                  static_cast<u32>(fluent_offsets[ip + 1]), objects);
            if (slot)
            {
                masks[derived ? 1 : 0][sign][*slot >> 6] |= u64{1} << (*slot & 63u);
                slotted = true;
                if (!horizon && derived && (goal_relations.at(tp, sign, 1) >= 0 || goal_relations.at(tp, sign, 2) >= 0))
                    out.needs |= k_needs_instance_goal;
            }
            else
                holds = !literal.positive;  // an atom no state has is false
        }
        if (!slotted)
            constants[{ip, sign}].push_back({std::move(objects), holds});
    }
    for (auto& entry : constants)
    {
        auto& rows = entry.second;
        std::sort(rows.begin(), rows.end(),
                  [](const ConstantLiteral& x, const ConstantLiteral& y) { return before(x.objects, y.objects); });
        rows.erase(std::unique(rows.begin(), rows.end(),
                               [](const ConstantLiteral& x, const ConstantLiteral& y) { return x.objects == y.objects; }),
                   rows.end());
    }
    {
        // the constant part as constant literals of layer 0, sorted by table predicate, sign and objects
        std::vector<std::vector<i32>> literals;
        for (const auto& [key, rows] : constants)
            for (const ConstantLiteral& row : rows)
            {
                std::vector<i32> literal{0, static_cast<i32>(instance.pred_map[key.first]), static_cast<i32>(key.second)};
                for (u32 k = 0; k < max_arity; ++k)
                    literal.push_back(k < row.objects.size() ? row.objects[k] : -1);
                literals.push_back(std::move(literal));
            }
        std::sort(literals.begin(), literals.end());
        for (const std::vector<i32>& literal : literals)
            out.goal_constants.insert(out.goal_constants.end(), literal.begin(), literal.end());
        out.goal_constant_count = literals.size();
    }
    // Gives the goal cell `cell` of level 0 of the instance's lists the rows [c, d) of the constant literals of
    // (ip, sign) of the instance's goal that its variant (0 plain, 1 satisfied, 2 unsatisfied) lists, appended to the
    // instance's goal_constants; returns whether it lists one.
    const u32 constant_width = detail::k_constant_objects + max_arity;
    auto list_constants = [&](Cell& cell, u32 ip, u32 sign, u32 variant)
    {
        cell.c = cell.d = static_cast<u32>(out.goal_constants.size() / constant_width);
        const auto found = constants.find({ip, sign});
        if (found == constants.end())
            return false;
        for (const ConstantLiteral& row : found->second)
        {
            if (variant != 0 && (variant == 1) != row.holds)
                continue;
            out.goal_constants.insert(out.goal_constants.end(),
                                      {0, static_cast<i32>(instance.pred_map[ip]), static_cast<i32>(sign)});
            for (u32 k = 0; k < max_arity; ++k)
                out.goal_constants.push_back(k < row.objects.size() ? row.objects[k] : -1);
            ++cell.d;
        }
        return cell.c != cell.d;
    };
    if (horizon)
    {
        // The cells of a horizon graph. The instance's own goal narrows the slot range of a goal cell of level 0 to the
        // literals of its mask and adds the cell's constant tuples; the cells of row goals and of the subgoal levels
        // span the predicate's slots.
        // the goal and satisfaction cells of the predicates without slots: of level 0 (row goals) and of the subgoal
        // levels, for the fixed lists
        std::vector<Cell> fixed_goal, fixed_levels;
        for (u32 ip = 0; ip < predicates; ++ip)
        {
            const u32 tp = instance.pred_map[ip];
            if (tp >= table.num_domain_predicates())
                continue;
            const bool derived = kind[ip] == rl::k_pred_derived, fixed = kind[ip] == rl::k_pred_static;
            const u32 logical = static_cast<u32>(arity[ip]);
            const Lane lane = fixed ? Lane::Static : derived ? Lane::Derived : Lane::Fluent;
            const u32 slot_begin = static_cast<u32>(derived ? derived_offsets[ip] : fluent_offsets[ip]);
            const u32 slot_end = static_cast<u32>(derived ? derived_offsets[ip + 1] : fluent_offsets[ip + 1]);
            for (const HorizonRelation& d : horizon->by_predicate[tp])
            {
                const Relation& relation = schema.relation(d.relation);
                if ((d.kind == detail::HorizonKind::Facts || d.kind == detail::HorizonKind::Delta) && d.lane != lane)
                    throw std::logic_error("mymyr: encoding: the relation " + relation.name + " does not match its predicate");
                if (relation.logical_arity != logical)
                    throw std::logic_error("mymyr: encoding: the relation " + relation.name + " does not match its predicate");
                Cell cell{d.relation, 0, 0, 0, relation.arity, logical};
                const u32 sign = d.negative ? 1 : 0;
                switch (d.kind)
                {
                case detail::HorizonKind::Facts:
                    if (fixed)
                    {
                        std::tie(cell.a, cell.b) = static_rows_of(ip, logical);
                        cell.op = detail::horizon_op(d.kind, d.scope, d.lead, false, static_cast<u32>(CellOp::Static));
                    }
                    else
                    {
                        cell.a = slot_begin;
                        cell.b = slot_end;
                        cell.op = detail::horizon_op(d.kind, d.scope, d.lead, false,
                                                     static_cast<u32>(derived ? CellOp::Derived : CellOp::Fluent));
                        if (derived)
                            out.needs |= k_needs_state;
                    }
                    break;
                case detail::HorizonKind::Delta:
                    cell.a = slot_begin;
                    cell.b = slot_end;
                    cell.op = detail::horizon_op(d.kind, d.scope, d.lead, d.minus,
                                                 static_cast<u32>(derived ? CellOp::Derived : CellOp::Fluent));
                    if (derived && cell.a != cell.b)
                        out.needs |= k_needs_state;
                    break;
                default:
                {
                    // Goal, Sat, Added
                    if (!fixed)
                    {
                        cell.a = slot_begin;
                        cell.b = slot_end;
                    }
                    cell.c = tp;
                    cell.op = detail::horizon_op(d.kind, d.scope, d.lead, d.minus,
                                                 detail::goal_op(derived, d.negative, d.mode, d.level));
                    if (derived && cell.a != cell.b && d.kind != detail::HorizonKind::Goal)
                        out.needs |= k_needs_state;
                    break;
                }
                }
                const bool goal_cell = d.kind == detail::HorizonKind::Goal || d.kind == detail::HorizonKind::Sat ||
                                       d.kind == detail::HorizonKind::Added;
                if (!goal_cell)
                {
                    if (cell.a != cell.b)
                    {
                        out.cells.push_back(cell);
                        out.row_cells.push_back(cell);
                    }
                    continue;
                }
                // a constant literal holds or fails at every node: an added-satisfaction cell never lists one
                const bool constant_part = d.kind != detail::HorizonKind::Added;
                const u32 variant = d.mode == detail::GoalMode::Satisfied ? 1 : d.mode == detail::GoalMode::Unsatisfied ? 2 : 0;
                if (cell.a != cell.b)
                {
                    out.row_cells.push_back(cell);
                    if (d.level == 0)
                        narrow_to_set_bits(masks[derived ? 1 : 0][sign], cell.a, cell.b);
                }
                else if (constant_part)
                    (d.level == 0 ? fixed_goal : fixed_levels).push_back(cell);
                bool listed = false;
                if (d.level == 0)
                {
                    listed = constant_part && list_constants(cell, ip, sign, variant);
                    if (!constant_part)
                        cell.c = cell.d = 0;
                }
                if (cell.a == cell.b && !listed)
                    continue;  // the instance's goal has no tuple here
                out.cells.push_back(cell);
            }
        }
        for (std::vector<Cell>* list : {&out.cells, &out.row_cells})
        {
            list->insert(list->end(), horizon->edge_cells.begin(), horizon->edge_cells.end());
            std::sort(list->begin(), list->end(), [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
        }
        // the rows of a horizon graph are all of one kind: the other lists are the two lists again
        out.plain_cells = out.cells;
        out.plain_row_cells = out.row_cells;
        {
            const std::array<const std::vector<Cell>*, 4> bases = {&out.cells, &out.row_cells, &out.plain_cells,
                                                                   &out.plain_row_cells};
            for (usize k = 0; k < out.extended.size(); ++k)
                out.extended[k] = *bases[k % 4];
            for (usize k = 0; k < out.fixed.size(); ++k)
            {
                std::vector<Cell>& list = out.fixed[k];
                list = *bases[k % 4];
                if (k % 2 == 1)
                    list.insert(list.end(), fixed_goal.begin(), fixed_goal.end());
                list.insert(list.end(), fixed_levels.begin(), fixed_levels.end());
                std::sort(list.begin(), list.end(), [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
            }
        }
        append_slot_rows(out.slot_args, atom_pred, atom_padded, fluent_width, arity, max_arity);
        append_slot_rows(out.derived_args, derived_pred, derived_padded, derived_width, arity, max_arity);
        fill_literal_ranges();
        return out;
    }
    out.row_cells = out.cells;
    std::vector<Cell> fixed_goal;  // the goal cells of the predicates without slots, for the fixed lists of row goals
    for (u32 ip = 0; ip < predicates; ++ip)
    {
        const u32 tp = instance.pred_map[ip];
        if (tp >= table.num_domain_predicates())
            continue;
        const bool derived = kind[ip] == rl::k_pred_derived, fixed = kind[ip] == rl::k_pred_static;
        const u32 logical = static_cast<u32>(arity[ip]);
        for (u32 sign = 0; sign < 2; ++sign)
        {
            for (u32 variant = 0; variant < 3; ++variant)
            {
                const i32 id = goal_relations.at(tp, sign, variant);
                if (id < 0)
                    continue;
                const Relation& relation = schema.relation(static_cast<u32>(id));
                Cell cell{static_cast<u32>(id), 0, 0, 0, relation.arity, logical};
                if (!fixed)
                {
                    cell.a = static_cast<u32>(derived ? derived_offsets[ip] : fluent_offsets[ip]);
                    cell.b = static_cast<u32>(derived ? derived_offsets[ip + 1] : fluent_offsets[ip + 1]);
                }
                cell.c = tp;
                const detail::GoalMode mode =
                    variant == 0 ? (goal_relations.target[static_cast<u32>(id)] ? detail::GoalMode::PlainTarget
                                                                                : detail::GoalMode::Plain)
                    : variant == 1 ? detail::GoalMode::Satisfied
                                   : detail::GoalMode::Unsatisfied;
                cell.op = detail::goal_op(derived, sign == 1, mode);
                if (cell.a != cell.b)
                {
                    // the goals of a row have masks over the predicate's whole slot range
                    out.row_cells.push_back(cell);
                    if (derived && variant > 0)
                        out.needs |= k_needs_row_goal;
                    narrow_to_set_bits(masks[derived ? 1 : 0][sign], cell.a, cell.b);
                }
                else
                    fixed_goal.push_back(cell);
                if (!list_constants(cell, ip, sign, variant) && cell.a == cell.b)
                    continue;  // the instance's goal has no tuple here
                out.cells.push_back(cell);
            }
        }
    }
    build_numeric_cells(task, schema, max_arity, out.numeric_cells, out.numeric_args);
    out.numeric_tuples = out.numeric_args.size() / max_arity;
    for (std::vector<Cell>* list : {&out.cells, &out.row_cells})
    {
        list->insert(list->end(), action_cells.begin(), action_cells.end());
        list->insert(list->end(), out.numeric_cells.begin(), out.numeric_cells.end());
        std::sort(list->begin(), list->end(), [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
    }
    for (const auto& [list, plain] :
         {std::pair{&out.cells, &out.plain_cells}, std::pair{&out.row_cells, &out.plain_row_cells}})
        std::ranges::copy_if(*list, std::back_inserter(*plain),
                             [](const Cell& c) { return !detail::is_action_op(c.op); });
    // The cells of subgoal layers and history steps: one per relation of a predicate that has slots, over all of its
    // slots; they come from the rows, so every graph that has such rows uses them whichever goal it has.
    // The fixed lists also have the cells of the subgoal levels of the predicates without slots (fixed_levels).
    std::vector<Cell> layer_cells, history_cells, fixed_levels;
    for (u32 ip = 0; ip < predicates; ++ip)
    {
        const u32 tp = instance.pred_map[ip];
        if (tp >= table.num_domain_predicates())
            continue;
        const bool derived = kind[ip] == rl::k_pred_derived;
        const bool slotted = kind[ip] != rl::k_pred_static;
        const u32 a = slotted ? static_cast<u32>(derived ? derived_offsets[ip] : fluent_offsets[ip]) : 0;
        const u32 b = slotted ? static_cast<u32>(derived ? derived_offsets[ip + 1] : fluent_offsets[ip + 1]) : 0;
        const u32 logical = static_cast<u32>(arity[ip]);
        if (a == b)
        {
            for (u32 level = 1; level <= schema.options().max_goal_level; ++level)
                for (u32 sign = 0; sign < 2; ++sign)
                    for (u32 variant = 0; variant < 3; ++variant)
                    {
                        const i32 id = layer_relations.subgoal_at(level, tp, sign, variant);
                        if (id < 0)
                            continue;
                        const detail::GoalMode mode =
                            variant == 0 ? (layer_relations.target[static_cast<u32>(id)] ? detail::GoalMode::PlainTarget
                                                                                         : detail::GoalMode::Plain)
                            : variant == 1 ? detail::GoalMode::Satisfied
                                           : detail::GoalMode::Unsatisfied;
                        fixed_levels.push_back({static_cast<u32>(id), detail::goal_op(derived, sign == 1, mode, level),
                                                0, 0, schema.relation(static_cast<u32>(id)).arity, logical, tp});
                    }
            continue;
        }
        for (u32 level = 1; level <= schema.options().max_goal_level; ++level)
            for (u32 sign = 0; sign < 2; ++sign)
                for (u32 variant = 0; variant < 3; ++variant)
                {
                    const i32 id = layer_relations.subgoal_at(level, tp, sign, variant);
                    if (id < 0)
                        continue;
                    const detail::GoalMode mode =
                        variant == 0 ? (layer_relations.target[static_cast<u32>(id)] ? detail::GoalMode::PlainTarget
                                                                                     : detail::GoalMode::Plain)
                        : variant == 1 ? detail::GoalMode::Satisfied
                                       : detail::GoalMode::Unsatisfied;
                    layer_cells.push_back({static_cast<u32>(id), detail::goal_op(derived, sign == 1, mode, level), a, b,
                                           schema.relation(static_cast<u32>(id)).arity, logical, tp});
                    if (derived && variant > 0)
                        out.needs |= k_needs_row_goal;
                }
        if (schema.options().history)
            for (u32 sign = 0; sign < 2; ++sign)
            {
                const i32 id = layer_relations.history_at(tp, sign);
                if (id >= 0)
                    history_cells.push_back(
                        {static_cast<u32>(id),
                         detail::history_op(derived, sign == 1, layer_relations.target[static_cast<u32>(id)] != 0),
                         a,
                         b,
                         schema.relation(static_cast<u32>(id)).arity,
                         logical});
            }
    }
    {
        const std::array<const std::vector<Cell>*, 4> bases = {&out.cells, &out.row_cells, &out.plain_cells,
                                                               &out.plain_row_cells};
        for (u32 extension = 1; extension <= 3; ++extension)
            for (u32 k = 0; k < 4; ++k)
            {
                std::vector<Cell>& list = out.extended[(extension - 1) * 4 + k];
                list = *bases[k];
                if (extension & 1u)
                    list.insert(list.end(), layer_cells.begin(), layer_cells.end());
                if (extension & 2u)
                    list.insert(list.end(), history_cells.begin(), history_cells.end());
                std::sort(list.begin(), list.end(),
                          [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
            }
        for (u32 extension = 0; extension <= 3; ++extension)
            for (u32 k = 0; k < 4; ++k)
            {
                std::vector<Cell>& list = out.fixed[extension * 4 + k];
                list = extension == 0 ? *bases[k] : out.extended[(extension - 1) * 4 + k];
                if (k % 2 == 1)
                    list.insert(list.end(), fixed_goal.begin(), fixed_goal.end());
                if (extension & 1u)
                    list.insert(list.end(), fixed_levels.begin(), fixed_levels.end());
                std::sort(list.begin(), list.end(),
                          [](const Cell& x, const Cell& y) { return x.relation < y.relation; });
            }
    }
    append_slot_rows(out.slot_args, atom_pred, atom_padded, fluent_width, arity, max_arity);
    append_slot_rows(out.derived_args, derived_pred, derived_padded, derived_width, arity, max_arity);
    fill_literal_ranges();
    return out;
}
}  // namespace

std::shared_ptr<const EncodingContext> EncodingContext::create(rl::TaskTablePtr table,
                                                              std::shared_ptr<const EncodingSchema> schema)
{
    if (!table || !schema)
        throw std::invalid_argument("mymyr: encoding: a context needs a table and a schema");
    // The schema of the table's own domain. Relation ids follow the sorted names, so they agree with the passed schema
    // when the fingerprints are equal, but a relation's predicate id follows the table's declaration order, which the
    // fingerprint ignores: predicates are mapped to relations through this schema only.
    const std::shared_ptr<const EncodingSchema> own = EncodingSchema::create(*table, schema->options());
    if (own->fingerprint() != schema->fingerprint())
        throw std::invalid_argument("mymyr: encoding: the schema is not the schema of the table's domain");
    // The relation of each domain predicate's state atoms.
    std::vector<i32> state_relation(table->num_domain_predicates(), -1);
    // The rows of the object arrays are as wide as the widest domain predicate (or numeric function): a state or goal
    // cell reads a row of its predicate's arity, and the literal section resolves the atoms of every domain predicate,
    // whether the schema has relations of it or not. (An action relation reads the expansion's bindings and no row.)
    u32 max_arity = 1;
    for (u32 tp = 0; tp < table->num_domain_predicates(); ++tp)
        max_arity = std::max(max_arity, table->predicate_arities()[tp]);
    for (u32 id = 0; id < own->num_relations(); ++id)
    {
        const Relation& relation = own->relation(id);
        if (relation.lane == Lane::Numeric)
            max_arity = std::max(max_arity, relation.logical_arity);
        if (!is_state_relation(relation))
            continue;
        state_relation.at(static_cast<u32>(relation.predicate)) = static_cast<i32>(id);
        max_arity = std::max(max_arity, relation.logical_arity);
    }
    // The subgoal and history relations: cells over a predicate's slots read rows of the predicate's arity too.
    LayerRelations layer_relations;
    layer_relations.subgoal.assign(u64{3} * table->num_domain_predicates() * 6, -1);
    layer_relations.history.assign(u64{table->num_domain_predicates()} * 2, -1);
    layer_relations.target.assign(own->num_relations(), 0);
    for (u32 id = 0; id < own->num_relations(); ++id)
    {
        const Relation& relation = own->relation(id);
        const u32 sign = relation.sign > 0 ? 0 : 1;
        if (relation.lane == Lane::History)
        {
            if (!own->options().history)
                continue;
            max_arity = std::max(max_arity, relation.logical_arity);
            layer_relations.history[u64{static_cast<u32>(relation.predicate)} * 2 + sign] = static_cast<i32>(id);
            if (leads_with(*own, id, SlotRole::HistoryTarget))
                layer_relations.target[id] = 1;
            continue;
        }
        const bool plain = relation.lane == Lane::Goal && !relation.derivation;
        const bool derivation =
            relation.lane == Lane::GoalDerivation &&
            (relation.derivation == GoalDerivation::Satisfied || relation.derivation == GoalDerivation::Unsatisfied);
        if (relation.level.value_or(0) == 0 || !(plain || derivation))
            continue;
        max_arity = std::max(max_arity, relation.logical_arity);
        const u32 variant = plain ? 0 : relation.derivation == GoalDerivation::Satisfied ? 1 : 2;
        const u32 level = *relation.level;
        const u64 per_level = u64{table->num_domain_predicates()} * 6;
        const u64 predicate = static_cast<u32>(relation.predicate);
        layer_relations.subgoal[(level - 1) * per_level + (predicate * 2 + sign) * 3 + variant] = static_cast<i32>(id);
        if (plain && leads_with(*own, id, SlotRole::SubgoalTarget))
            layer_relations.target[id] = 1;
    }
    GoalRelations goal_relations;
    goal_relations.id.assign(u64{table->num_domain_predicates()} * 6, -1);
    goal_relations.target.assign(own->num_relations(), 0);
    for (u32 id = 0; id < own->num_relations(); ++id)
    {
        const Relation& relation = own->relation(id);
        const bool plain = relation.lane == Lane::Goal && !relation.derivation;
        const bool derivation =
            relation.lane == Lane::GoalDerivation &&
            (relation.derivation == GoalDerivation::Satisfied || relation.derivation == GoalDerivation::Unsatisfied);
        if (relation.level.value_or(1) != 0 || !(plain || derivation))
            continue;
        max_arity = std::max(max_arity, relation.logical_arity);
        const u32 variant = plain ? 0 : relation.derivation == GoalDerivation::Satisfied ? 1 : 2;
        goal_relations.id[(u64{static_cast<u32>(relation.predicate)} * 2 + (relation.sign > 0 ? 0 : 1)) * 3 + variant] =
            static_cast<i32>(id);
        if (plain && leads_with(*own, id, SlotRole::GoalTarget))
            goal_relations.target[id] = 1;
    }
    // The relations of a horizon: what each lists, and the cells of the edge relations (every instance has them).
    const bool horizon = own->options().horizon.mode != HorizonMode::None;
    HorizonRelations horizon_relations;
    if (horizon)
    {
        horizon_relations.by_predicate.resize(table->num_domain_predicates());
        horizon_relations.goal.assign(u64{table->num_domain_predicates()} * 2, 0);
        for (u32 id = 0; id < own->num_relations(); ++id)
        {
            const Relation& relation = own->relation(id);
            if (relation.lane == Lane::Action)
                horizon_relations.edge_cells.push_back(
                    {id, detail::horizon_op(detail::HorizonKind::Action, detail::HorizonScope::All, true, false, 0), 0,
                     0, relation.arity, relation.logical_arity});
            else if (relation.lane == Lane::Parent || relation.lane == Lane::Sibling || relation.lane == Lane::Cousin)
            {
                const detail::HorizonKind kind = relation.lane == Lane::Parent    ? detail::HorizonKind::Parent
                                                 : relation.lane == Lane::Sibling ? detail::HorizonKind::Sibling
                                                                                  : detail::HorizonKind::Cousin;
                horizon_relations.edge_cells.push_back(
                    {id, detail::horizon_op(kind, detail::HorizonScope::All, true, false, 0), 0, 0, relation.arity, 0});
            }
            if (relation.predicate >= 0)
                max_arity = std::max(max_arity, relation.logical_arity);
            const auto described = describe_horizon(*own, id);
            if (!described)
                continue;
            const u32 predicate = static_cast<u32>(relation.predicate);
            horizon_relations.by_predicate[predicate].push_back(*described);
            if (described->level == 0 && (described->kind == detail::HorizonKind::Goal ||
                                          described->kind == detail::HorizonKind::Sat ||
                                          described->kind == detail::HorizonKind::Added))
                horizon_relations.goal[u64{predicate} * 2 + (described->negative ? 1 : 0)] = 1;
        }
    }
    // The action relations: the schemas of each, and one cell per relation in every instance (the domain's schemas are
    // the instances'). A horizon's action relations list its edges instead.
    std::vector<u32> action_schemas, schema_arity(table->schema_arities().begin(), table->schema_arities().end());
    std::vector<Cell> action_cells;
    for (u32 id = 0; id < own->num_relations() && own->options().include_actions && !horizon; ++id)
    {
        const Relation& relation = own->relation(id);
        if (relation.lane != Lane::Action)
            continue;
        const u32 begin = static_cast<u32>(action_schemas.size());
        for (u32 s = 0; s < own->schema_relations().size(); ++s)
            if (own->schema_relations()[s] == static_cast<i32>(id))
            {
                if (schema_arity[s] != relation.logical_arity)
                    throw std::logic_error("mymyr: encoding: the action relation " + relation.name + " does not match its schema");
                action_schemas.push_back(s);
            }
        const u32 end = static_cast<u32>(action_schemas.size());
        const CellOp op = own->options().targets.contains(TargetSource::Action) ? CellOp::ActionTarget : CellOp::Action;
        action_cells.push_back({id, static_cast<u32>(op), begin, end, relation.arity, relation.logical_arity});
    }
    // The predicate virtual nodes: the virtual index of every relation's predicate (its rank by predicate name) and
    // the key relation of every virtual index.
    std::vector<u32> relation_virtual, virtual_relation;
    if (own->options().predicate_virtual_nodes)
    {
        virtual_relation.assign(own->num_virtual_predicates(), 0);
        for (u32 id = 0; id < own->num_relations(); ++id)
        {
            const i32 rank = own->virtual_rank(id);
            relation_virtual.push_back(rank < 0 ? detail::k_no_virtual : static_cast<u32>(rank));
            if (rank >= 0)
                virtual_relation[static_cast<u32>(rank)] = static_cast<u32>(own->virtual_key(id));
        }
    }
    std::vector<InstanceRows> instances;
    instances.reserve(table->size());
    for (u32 i = 0; i < table->size(); ++i)
        instances.push_back(build_instance(*table, i, state_relation, goal_relations, layer_relations, action_cells,
                                           *own, max_arity, horizon ? &horizon_relations : nullptr));

    auto context = std::shared_ptr<EncodingContext>(new EncodingContext());
    context->m_max_arity = max_arity;
    std::vector<u64> inst;
    std::vector<u32> cells;
    std::vector<i32> slot_args, derived_args, static_args, goal_constants, numeric_args;
    std::vector<u32> literal_ranges;
    std::vector<u64> goal_words;
    for (u32 i = 0; i < instances.size(); ++i)
    {
        const InstanceRows& rows = instances[i];
        u64 columns[detail::k_instance_columns] = {};
        columns[detail::k_num_objects] = rows.num_objects;
        columns[detail::k_words] = rows.words;
        columns[detail::k_fluent_slots] = rows.fluent_slots;
        columns[detail::k_derived_words] = rows.derived_words;
        columns[detail::k_derived_slots] = rows.derived_slots;
        // each list of cells is stored once: a list equal to an earlier one of the instance shares its rows
        const auto same_cell = [](const Cell& x, const Cell& y) { return std::memcmp(&x, &y, sizeof(Cell)) == 0; };
        std::array<const std::vector<Cell>*, 32> lists = {&rows.cells, &rows.row_cells, &rows.plain_cells,
                                                          &rows.plain_row_cells};
        for (usize k = 0; k < rows.extended.size(); ++k)
            lists[4 + k] = &rows.extended[k];
        for (usize k = 0; k < rows.fixed.size(); ++k)
            lists[16 + k] = &rows.fixed[k];
        constexpr std::array<std::pair<detail::InstanceColumn, detail::InstanceColumn>, 4> list_columns = {
            std::pair{detail::k_num_cells, detail::k_cell_offset},
            std::pair{detail::k_num_row_cells, detail::k_row_cell_offset},
            std::pair{detail::k_num_plain_cells, detail::k_plain_cell_offset},
            std::pair{detail::k_num_plain_row_cells, detail::k_plain_row_cell_offset}};
        std::array<u64, 32> list_offset{};
        u64 next_offset = cells.size() / 8;
        for (usize k = 0; k < lists.size(); ++k)
        {
            usize same = 0;
            while (same < k && !std::equal(lists[same]->begin(), lists[same]->end(), lists[k]->begin(), lists[k]->end(),
                                           same_cell))
                ++same;
            if (same < k)
                list_offset[k] = list_offset[same];
            else
            {
                list_offset[k] = next_offset;
                next_offset += lists[k]->size();
                for (const Cell& cell : *lists[k])
                    for (u32 value :
                         {cell.relation, cell.op, cell.a, cell.b, cell.arity, cell.logical_arity, cell.c, cell.d})
                        cells.push_back(value);
            }
            if (k < 4)
            {
                columns[list_columns[k].first] = lists[k]->size();
                columns[list_columns[k].second] = list_offset[k];
            }
            else if (k < 16)
            {
                columns[detail::k_ext_lists + 2 * (k - 4)] = lists[k]->size();
                columns[detail::k_ext_lists + 2 * (k - 4) + 1] = list_offset[k];
            }
            else
            {
                columns[detail::k_fixed_lists + 2 * (k - 16)] = lists[k]->size();
                columns[detail::k_fixed_lists + 2 * (k - 16) + 1] = list_offset[k];
            }
        }
        columns[detail::k_slot_args_offset] = slot_args.size() / max_arity;
        columns[detail::k_derived_args_offset] = derived_args.size() / max_arity;
        columns[detail::k_static_args_offset] = static_args.size() / max_arity;
        columns[detail::k_numeric_args_offset] = numeric_args.size() / max_arity;
        columns[detail::k_numeric_tuples] = rows.numeric_tuples;
        columns[detail::k_encoding_fingerprint] = rows.fingerprint;
        columns[detail::k_goal_offset] = goal_words.size();
        columns[detail::k_goal_constants_offset] = goal_constants.size() / (detail::k_constant_objects + max_arity);
        columns[detail::k_goal_constant_count] = rows.goal_constant_count;
        columns[detail::k_goal_complete] = rows.goal_complete;
        columns[detail::k_needs_derived] = rows.needs;
        inst.insert(inst.end(), std::begin(columns), std::end(columns));
        if (rows.needs != 0)
            context->m_max_derived_words = std::max(context->m_max_derived_words, static_cast<u32>(rows.derived_words));
        slot_args.insert(slot_args.end(), rows.slot_args.begin(), rows.slot_args.end());
        derived_args.insert(derived_args.end(), rows.derived_args.begin(), rows.derived_args.end());
        static_args.insert(static_args.end(), rows.static_args.begin(), rows.static_args.end());
        numeric_args.insert(numeric_args.end(), rows.numeric_args.begin(), rows.numeric_args.end());
        goal_words.insert(goal_words.end(), rows.goal_words.begin(), rows.goal_words.end());
        goal_constants.insert(goal_constants.end(), rows.goal_constants.begin(), rows.goal_constants.end());
        literal_ranges.insert(literal_ranges.end(), rows.literal_ranges.begin(), rows.literal_ranges.end());
    }
    auto shape = [&](const std::vector<i32>& rows) { return std::vector<i64>{static_cast<i64>(rows.size() / max_arity), max_arity}; };
    rl::ArrayBundle::Builder builder;
    builder.add("inst", rl::DType::U64, {static_cast<i64>(instances.size()), detail::k_instance_columns}, inst);
    builder.add("cell", rl::DType::U32, {static_cast<i64>(cells.size() / 8), 8}, cells);
    builder.add("slot_args", rl::DType::I32, shape(slot_args), slot_args);
    builder.add("derived_args", rl::DType::I32, shape(derived_args), derived_args);
    builder.add("static_args", rl::DType::I32, shape(static_args), static_args);
    builder.add("goal_words", rl::DType::U64, {static_cast<i64>(goal_words.size())}, goal_words);
    builder.add("goal_constants", rl::DType::I32,
                {static_cast<i64>(goal_constants.size() / (detail::k_constant_objects + max_arity)),
                 static_cast<i64>(detail::k_constant_objects + max_arity)},
                goal_constants);
    builder.add("action_schemas", rl::DType::U32, {static_cast<i64>(action_schemas.size())}, action_schemas);
    builder.add("schema_arity", rl::DType::U32, {static_cast<i64>(schema_arity.size())}, schema_arity);
    builder.add("relation_virtual", rl::DType::U32, {static_cast<i64>(relation_virtual.size())}, relation_virtual);
    builder.add("virtual_relation", rl::DType::U32, {static_cast<i64>(virtual_relation.size())}, virtual_relation);
    std::vector<u32> relation_arity;
    for (u32 id = 0; id < own->num_relations(); ++id)
        relation_arity.push_back(own->relation(id).arity);
    builder.add("relation_arity", rl::DType::U32, {static_cast<i64>(relation_arity.size())}, relation_arity);
    builder.add("static_relations", rl::DType::U32, {static_cast<i64>(own->static_relations().size())},
                std::vector<u32>(own->static_relations().begin(), own->static_relations().end()));
    builder.add("numeric_args", rl::DType::I32, shape(numeric_args), numeric_args);
    builder.add("literal_ranges", rl::DType::U32,
                {static_cast<i64>(instances.size()), static_cast<i64>(table->num_domain_predicates()),
                 detail::k_literal_columns},
                literal_ranges);
    context->m_num_virtual = static_cast<u32>(virtual_relation.size());
    context->m_arrays = builder.build();
    context->m_table = std::move(table);
    context->m_schema = std::move(schema);
    return context;
}

detail::ContextView EncodingContext::view(const std::byte* block) const noexcept
{
    const std::byte* base = block ? block : m_arrays.block();
    // the arrays are in the order create() adds them: inst, cell, slot_args, derived_args, static_args, goal_words,
    // goal_constants, action_schemas, schema_arity, relation_virtual, virtual_relation, relation_arity, static_relations,
    // numeric_args, literal_ranges
    const auto& arrays = m_arrays.arrays();
    auto at = [&](usize k) { return base + arrays[k].offset; };
    detail::ContextView v;
    v.inst = reinterpret_cast<const u64*>(at(0));
    v.num_instances = m_table->size();
    v.cells = reinterpret_cast<const Cell*>(at(1));
    v.slot_args = reinterpret_cast<const i32*>(at(2));
    v.derived_args = reinterpret_cast<const i32*>(at(3));
    v.static_args = reinterpret_cast<const i32*>(at(4));
    v.goal_words = reinterpret_cast<const u64*>(at(5));
    v.goal_constants = reinterpret_cast<const i32*>(at(6));
    v.action_schemas = reinterpret_cast<const u32*>(at(7));
    v.schema_arity = reinterpret_cast<const u32*>(at(8));
    v.num_schemas = m_table->num_schemas();
    if (m_schema->options().predicate_virtual_nodes)
    {
        v.relation_virtual = reinterpret_cast<const u32*>(at(9));
        v.virtual_relation = reinterpret_cast<const u32*>(at(10));
        v.num_virtual = m_num_virtual;
    }
    v.relation_arity = reinterpret_cast<const u32*>(at(11));
    v.static_relations = reinterpret_cast<const u32*>(at(12));
    v.num_static = static_cast<u32>(m_schema->static_relations().size());
    v.numeric_args = reinterpret_cast<const i32*>(at(13));
    v.numeric_storage = m_table->numeric() && m_table->numeric_storage() == NumericStorage::I32
                            ? detail::k_numeric_storage_i32
                            : detail::k_numeric_storage_f64;
    v.literal_ranges = reinterpret_cast<const u32*>(at(14));
    v.num_predicates = m_table->num_domain_predicates();
    v.max_arity = m_max_arity;
    v.num_relations = m_schema->num_relations();
    v.repeated_literals = m_schema->options().repeated_literals;
    v.nullary_auxiliary = m_schema->options().nullary != Nullary::Arity0;
    return v;
}
}
