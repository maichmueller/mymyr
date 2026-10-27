#pragma once
// The literal ranges shared by encoders and lifted queries. Static tuples are supplied by the caller so that an
// encoder's cells and its literal ranges refer to the same sorted, deduplicated rows.
#include "mymyr/encoding/detail/literals.hpp"
#include "mymyr/rl/task_table.hpp"
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace mymyr::encoding::detail
{
void check_slot_range(std::span<const i32> pred, std::span<const i32> padded, u32 width, u32 begin, u32 end,
                      u32 predicate, u32 arity, const std::string& name);

inline std::span<const i32> literal_ints(const rl::ArrayBundle& b, const char* name)
{
    const auto& a = *b.find(name);
    return {static_cast<const i32*>(b.data(a)), static_cast<usize>(a.elements())};
}

template<class StaticRows>
std::vector<u32> build_literal_ranges(const rl::TaskTable& table, u32 index, const rl::ArrayBundle& meta,
                                      StaticRows&& static_rows_of)
{
    const auto atom_pred = literal_ints(meta, "atom_pred"), atom_padded = literal_ints(meta, "atom_args_padded");
    const auto derived_pred = literal_ints(meta, "derived_pred"), derived_padded = literal_ints(meta, "derived_args_padded");
    const auto fluent_offsets = literal_ints(meta, "pred_slot_offsets");
    const auto derived_offsets = literal_ints(meta, "derived_pred_slot_offsets");
    const auto arity = literal_ints(meta, "pred_arity"), kind = literal_ints(meta, "pred_kind");
    const u32 predicates = static_cast<u32>(arity.size());
    const u32 fluent_width = atom_pred.empty() ? 0 : static_cast<u32>(atom_padded.size() / atom_pred.size());
    const u32 derived_width = derived_pred.empty() ? 0 : static_cast<u32>(derived_padded.size() / derived_pred.size());
    std::vector<u32> ranges;
        const u32 domain_predicates = table.num_domain_predicates();
        ranges.assign(u64{domain_predicates} * detail::k_literal_columns, 0);
        for (u32 tp = 0; tp < domain_predicates; ++tp)
        {
            u32* r = ranges.data() + u64{tp} * detail::k_literal_columns;
            r[detail::k_literal_arity] = table.predicate_arities()[tp];
            r[detail::k_literal_predicate_kind] = detail::k_literal_absent_predicate;
        }
        for (u32 ip = 0; ip < predicates; ++ip)
        {
            const u32 tp = table.instance(index).pred_map[ip];
            if (tp >= domain_predicates)
                continue;
            u32* r = ranges.data() + u64{tp} * detail::k_literal_columns;
            const u32 logical = static_cast<u32>(arity[ip]);
            const std::string& name = table.predicate_names()[tp];
            r[detail::k_literal_arity] = logical;
            if (kind[ip] == rl::k_pred_static)
            {
                r[detail::k_literal_predicate_kind] = detail::k_literal_static_predicate;
                std::tie(r[detail::k_literal_static_begin], r[detail::k_literal_static_end]) =
                    static_rows_of(ip, logical);
            }
            else if (kind[ip] == rl::k_pred_derived)
            {
                r[detail::k_literal_predicate_kind] = detail::k_literal_derived_predicate;
                r[detail::k_literal_derived_begin] = static_cast<u32>(derived_offsets[ip]);
                r[detail::k_literal_derived_end] = static_cast<u32>(derived_offsets[ip + 1]);
                detail::check_slot_range(derived_pred, derived_padded, derived_width, r[detail::k_literal_derived_begin],
                                         r[detail::k_literal_derived_end], ip, logical, name);
            }
            else
            {
                r[detail::k_literal_predicate_kind] = detail::k_literal_fluent_predicate;
                r[detail::k_literal_fluent_begin] = static_cast<u32>(fluent_offsets[ip]);
                r[detail::k_literal_fluent_end] = static_cast<u32>(fluent_offsets[ip + 1]);
                detail::check_slot_range(atom_pred, atom_padded, fluent_width, r[detail::k_literal_fluent_begin],
                                         r[detail::k_literal_fluent_end], ip, logical, name);
            }
        }
    return ranges;
}
}  // namespace mymyr::encoding::detail
