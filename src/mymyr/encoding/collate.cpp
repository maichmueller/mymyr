#include "mymyr/encoding/collate.hpp"

#include "mymyr/encoding/encoder.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mymyr::encoding
{
namespace
{
[[noreturn]] void invalid(const std::string& why) { throw std::invalid_argument("mymyr: encoding: " + why); }
u64 add(u64 a, u64 b)
{
    if (b > std::numeric_limits<u64>::max() - a)
        throw std::overflow_error("mymyr: encoding: collation total exceeds uint64");
    return a + b;
}
u64 mul(u64 a, u64 b)
{
    if (b && a > std::numeric_limits<u64>::max() / b)
        throw std::overflow_error("mymyr: encoding: collation size exceeds uint64");
    return a * b;
}
const rl::ArrayInfo& array(const rl::ArrayBundle& b, const char* name)
{
    const auto* a = b.find(name);
    if (!a)
        invalid(std::string("a collated batch is missing ") + name);
    return *a;
}
template<class I> const I* data(const rl::ArrayBundle& b, const char* name)
{
    return static_cast<const I*>(b.data(array(b, name)));
}
template<class I> I* data(rl::ArrayBundle& b, const char* name)
{
    return static_cast<I*>(b.mutable_data(name));
}
template<class I> u64 last(const rl::ArrayBundle& b, const char* name)
{
    const auto& a = array(b, name);
    if (!a.elements())
        invalid(std::string("a collated offset array is empty: ") + name);
    const auto value = static_cast<const I*>(b.data(a))[a.elements() - 1];
    if (value < 0)
        invalid("a collated offset is negative");
    return static_cast<u64>(value);
}

struct Slice
{
    const rl::ArrayBundle* batch;
    u64 graph, graphs, entity, entities, args, targets, history;
};

template<class I> std::shared_ptr<const rl::ArrayBundle> concatenate(
    std::span<const std::shared_ptr<const rl::ArrayBundle>> batches)
{
    const auto& first = *batches.front();
    const u64 relations = static_cast<u64>(first.scalar("num_relations"));
    const bool relation_major = first.scalar("layout") == 1;
    const bool targets = first.find("target_offsets"), history = first.find("history_offsets");
    const IndexType index = sizeof(I) == 8 ? IndexType::I64 : IndexType::I32;
    u64 G = 0, E = 0, A = 0, T = 0, H = 0;
    std::vector<Slice> slices;
    slices.reserve(batches.size());
    for (const auto& b : batches)
    {
        const auto graphs = b->scalar("num_graphs");
        if (graphs < 0)
            invalid("a collated batch has no graph count");
        const u64 entities = last<I>(*b, "entity_offsets"), args = last<I>(*b, "relation_arg_offsets");
        const u64 t = targets ? last<I>(*b, "target_offsets") : 0;
        const u64 h = history ? last<I>(*b, "history_offsets") : 0;
        slices.push_back({b.get(), G, static_cast<u64>(graphs), E, entities, args, t, h});
        G = add(G, static_cast<u64>(graphs)); E = add(E, entities); A = add(A, args); T = add(T, t); H = add(H, h);
    }
    const u64 cells = mul(G, relations);
    for (const u64 value : {add(G, 1), add(cells, 1), E, A, T, H})
        detail::check_index_range(value, index, "collated total");

    // Count cells in destination order. All sums are wide until every total has passed the index check.
    std::vector<u64> tuples(add(cells, 1)), arguments(add(cells, 1));
    for (const auto& s : slices)
    {
        const I* ro = data<I>(*s.batch, "relation_offsets");
        const I* ao = data<I>(*s.batch, "relation_arg_offsets");
        for (u64 g = 0; g < s.graphs; ++g)
            for (u64 r = 0; r < relations; ++r)
            {
                const u64 src = relation_major ? r * s.graphs + g : g * relations + r;
                const u64 dst = relation_major ? r * G + s.graph + g : (s.graph + g) * relations + r;
                if (ro[src] < 0 || ro[src + 1] < ro[src] || ao[src] < 0 || ao[src + 1] < ao[src])
                    invalid("collated offsets must be nonnegative and monotone");
                tuples[dst + 1] = static_cast<u64>(ro[src + 1] - ro[src]);
                arguments[dst + 1] = static_cast<u64>(ao[src + 1] - ao[src]);
            }
    }
    for (u64 c = 0; c < cells; ++c)
    {
        tuples[c + 1] = add(tuples[c], tuples[c + 1]);
        arguments[c + 1] = add(arguments[c], arguments[c + 1]);
    }
    detail::check_index_range(tuples.back(), index, "collated tuples");
    if (arguments.back() != A)
        invalid("collated argument offsets have inconsistent totals");

    rl::ArrayBundle::Builder builder;
    for (const auto& [key, value] : first.scalars())
        builder.scalar(key, key == "num_graphs" ? static_cast<i64>(G) : value);
    for (const auto& a : first.arrays())
    {
        const auto name = a.name.view();
        u64 n = G;
        if (name == "entity_role" || name == "entity_ref") n = E;
        else if (name == "entity_offsets" || name == "target_offsets" || name == "history_offsets") n = G + 1;
        else if (name == "relation_offsets" || name == "relation_arg_offsets") n = cells + 1;
        else if (name == "relation_args") n = A;
        else if (name == "target_entity" || name == "target_group") n = T;
        else if (name == "history_entity" || name == "history_dt") n = H;
        else if (name == "relation_counts") n = cells;
        else if (name != "task_ids" && name != "num_objects" && name != "goal_complete")
            invalid("collation does not support array " + a.name.str());
        const rl::ArrayShape shape = name == "relation_counts" ? rl::ArrayShape{static_cast<i64>(G), static_cast<i64>(relations)}
                                                               : rl::ArrayShape{static_cast<i64>(n)};
        builder.add_uninitialized(a.name, a.dtype, shape);
    }
    auto result = std::make_shared<rl::ArrayBundle>(builder.build());
    for (u64 c = 0; c <= cells; ++c)
    {
        data<I>(*result, "relation_offsets")[c] = static_cast<I>(tuples[c]);
        data<I>(*result, "relation_arg_offsets")[c] = static_cast<I>(arguments[c]);
    }
    u64 target_base = 0, history_base = 0;
    for (const auto& s : slices)
    {
        for (const auto& a : first.arrays())
        {
            const auto name = a.name.view();
            const auto& source = array(*s.batch, a.name.str().c_str());
            u64 start = s.graph, count = s.graphs;
            if (name == "entity_role" || name == "entity_ref") { start = s.entity; count = s.entities; }
            else if (name == "target_group") { start = target_base; count = s.targets; }
            else if (name == "history_dt") { start = history_base; count = s.history; }
            else if (name == "relation_counts") { start = mul(s.graph, relations); count = mul(s.graphs, relations); }
            else if (name != "task_ids" && name != "num_objects" && name != "goal_complete") continue;
            const u64 bytes = mul(count, rl::dtype_bytes(a.dtype));
            if (source.elements() < count)
                invalid("a collated array is too short: " + a.name.str());
            if (bytes)
                std::memcpy(static_cast<std::byte*>(result->mutable_data(a.name.view())) + mul(start, rl::dtype_bytes(a.dtype)),
                            s.batch->data(source), bytes);
        }
        for (const auto& [offset_name, entity_name, base, count] : {
                 std::tuple{"entity_offsets", "", s.entity, s.entities},
                 std::tuple{"target_offsets", "target_entity", target_base, s.targets},
                 std::tuple{"history_offsets", "history_entity", history_base, s.history}})
        {
            if (!first.find(offset_name)) continue;
            const I* src = data<I>(*s.batch, offset_name);
            I* dst = data<I>(*result, offset_name);
            for (u64 g = 0; g <= s.graphs; ++g)
                dst[s.graph + g] = static_cast<I>(base + static_cast<u64>(src[g]));
            if (*entity_name)
                for (u64 j = 0; j < count; ++j)
                    data<I>(*result, entity_name)[base + j] = static_cast<I>(s.entity + data<I>(*s.batch, entity_name)[j]);
        }
        const I* source_args = data<I>(*s.batch, "relation_args");
        const I* source_offsets = data<I>(*s.batch, "relation_arg_offsets");
        I* dest_args = data<I>(*result, "relation_args");
        for (u64 g = 0; g < s.graphs; ++g)
            for (u64 r = 0; r < relations; ++r)
            {
                const u64 src = relation_major ? r * s.graphs + g : g * relations + r;
                const u64 dst = relation_major ? r * G + s.graph + g : (s.graph + g) * relations + r;
                for (u64 j = 0; j < arguments[dst + 1] - arguments[dst]; ++j)
                    dest_args[arguments[dst] + j] = static_cast<I>(s.entity + source_args[source_offsets[src] + j]);
            }
        target_base += s.targets; history_base += s.history;
    }
    return result;
}
}

std::shared_ptr<const rl::ArrayBundle> collate(std::span<const std::shared_ptr<const rl::ArrayBundle>> batches)
{
    if (batches.empty()) invalid("collate needs at least one batch");
    for (const auto& b : batches)
        if (!b) invalid("collate received a null batch");
    const auto& first = *batches.front();
    for (const auto& b : batches)
    {
        for (const char* field : {"version", "schema_fingerprint", "num_relations", "layout", "index_dtype", "statics"})
            if (b->scalar(field) != first.scalar(field))
                invalid(std::string("collate received mixed ") + field);
        if (b->arrays().size() != first.arrays().size()) invalid("collate received different stored arrays");
        for (const auto& a : first.arrays())
            if (!b->find(a.name.view()) || b->find(a.name.view())->dtype != a.dtype)
                invalid("collate received different stored arrays or dtypes");
    }
    if (first.scalar("statics") != 0) invalid("collate requires repeated statics");
    if (first.scalar("num_relations") < 0) invalid("collate received no relation count");
    if (first.scalar("index_dtype") == 0) return concatenate<i32>(batches);
    if (first.scalar("index_dtype") == 1) return concatenate<i64>(batches);
    invalid("collate received an invalid index type");
}
}
