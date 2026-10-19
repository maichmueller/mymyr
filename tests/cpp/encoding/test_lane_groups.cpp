#include "matrix.hpp"

#include <map>
#include <tuple>

using namespace mymyr;
namespace ed = mymyr::encoding::detail;
using namespace mymyr::test::encoding_matrix;

namespace
{
struct Destination
{
    std::map<std::string, std::vector<std::byte>> arrays;
    explicit Destination(const rl::ArrayBundle& reference)
    {
        for (const auto& a : reference.arrays())
        {
            const auto* first = static_cast<const std::byte*>(reference.data(a));
            arrays[a.name.str()] = {first, first + a.elements() * rl::dtype_bytes(a.dtype)};
        }
    }
    template<class T> T* at(const char* name)
    {
        const auto it = arrays.find(name);
        return it == arrays.end() ? nullptr : reinterpret_cast<T*>(it->second.data());
    }
    void dirty(const char* name)
    {
        const auto it = arrays.find(name);
        if (it != arrays.end())
            std::fill(it->second.begin(), it->second.end(), std::byte{0xa5});
    }
    template<class I> ed::GraphOut<I> out(u64 graphs)
    {
        return {graphs, at<I>("num_objects"), at<I>("relation_counts"), at<I>("relation_offsets"),
                at<I>("relation_arg_offsets"), at<I>("relation_args"), at<u8>("entity_role"), at<I>("entity_ref"),
                at<u8>("goal_complete"), at<I>("target_offsets"), at<I>("target_entity"), at<u8>("target_group"),
                at<I>("history_offsets"), at<I>("history_entity"), at<i32>("history_dt")};
    }
};

auto totals(const ed::GraphTotals& t)
{
    return std::tuple{t.entities, t.tuples, t.args,  t.targets,        t.goal_entities, t.actions,
                      t.virtuals, t.flags,  t.subgoal_entities, t.history_steps, t.history_targets};
}

template<class I>
void compare(const Batch& batch, const Config& config, encoding::Layout layout)
{
    const auto encoder = encoding::FlatEncoder::create(batch.table, with(config.options, layout,
        sizeof(I) == 8 ? encoding::IndexType::I64 : encoding::IndexType::I32));
    const auto in = host_input(batch, batch.count, config.inputs);
    const auto reference = encoder->encode(in);
    const u64 G = batch.count, R = encoder->schema().num_relations();
    Destination expected(*reference), plain(*reference), grouped(*reference);
    for (Destination* dst : {&plain, &grouped})
        for (const char* name : {"num_objects", "relation_counts", "relation_args", "entity_role", "entity_ref",
                                 "goal_complete", "target_entity", "target_group", "history_entity", "history_dt"})
            dst->dirty(name);
    for (Destination* dst : {&plain, &grouped})
    {
        if (layout == encoding::Layout::Graph)
            for (const char* name : {"relation_offsets", "relation_arg_offsets"})
            {
                dst->dirty(name);
                dst->at<I>(name)[G * R] = expected.at<I>(name)[G * R];
            }
        for (const char* name : {"target_offsets", "history_offsets"})
            if (dst->at<I>(name))
            {
                dst->dirty(name);
                dst->at<I>(name)[G] = expected.at<I>(name)[G];
            }
    }
    struct Positions
    {
        const I* offsets;
        const I* arguments;
        u32 count(u64 s) const { return static_cast<u32>(offsets[s + 1] - offsets[s]); }
        u64 args(u64 s, u32) const { return static_cast<u64>(arguments[s]); }
    } positions{expected.at<I>("relation_offsets"), expected.at<I>("relation_arg_offsets")};
    for (u64 g = 0; g < G; ++g)
    {
        const auto v = ed::instance_view(encoder->context().view(), static_cast<u32>(batch.task_ids[g]));
        ed::RowInput row{batch.states.data() + g * batch.words,
                        batch.derived_words ? batch.derived.data() + g * batch.derived_words : nullptr};
        if (config.inputs.row_goals)
        {
            row.goal_pos = batch.goal_pos.data() + g * batch.words;
            row.goal_neg = batch.goal_neg.data() + g * batch.words;
            if (config.inputs.derived_goals && batch.derived_words)
            {
                row.goal_derived_pos = batch.goal_derived_pos.data() + g * batch.derived_words;
                row.goal_derived_neg = batch.goal_derived_neg.data() + g * batch.derived_words;
            }
        }
        // the rows' subgoal layers and history steps, and the levels of the layers
        const ed::LayerSource source{in.subgoals, in.history, encoder->schema().options().history_max_steps.value_or(0)};
        ed::RowLayers layers;
        std::vector<u64> storage;
        if (source.layered())
        {
            ASSERT_EQ(ed::layer_validity(v, source, g), ed::LayerValidity::Valid);
            if (ed::bind_layers(source, g, layers))
            {
                row.layers = &layers;
                storage.assign(ed::effective_layer_words(v, layers.subgoals), 0);
                ed::bind_levels(ed::SoloGroup{}, v, row, layers, storage.data());
            }
        }
        const u64 history = expected.at<I>("history_offsets") ? static_cast<u64>(expected.at<I>("history_offsets")[g]) : 0;
        const u32 cells = ed::cell_list(v, row).count;
        std::vector<u32> a(cells), b(cells), va(v.num_virtual), vb(v.num_virtual);
        const auto ta = ed::count_graph(v, row, a.data(), va.data());
        const auto tb = ed::count_graph_group(ed::SoloGroup{}, v, row, b.data(), vb.data());
        ASSERT_EQ(totals(ta), totals(tb));
        ASSERT_EQ(a, b);
        const u64 entity = static_cast<u64>(expected.at<I>("entity_offsets")[g]);
        const u64 target = expected.at<I>("target_offsets") ? static_cast<u64>(expected.at<I>("target_offsets")[g]) : 0;
        if (layout == encoding::Layout::Graph)
        {
            const u64 tuple = static_cast<u64>(expected.at<I>("relation_offsets")[g * R]);
            const u64 arg = static_cast<u64>(expected.at<I>("relation_arg_offsets")[g * R]);
            ed::fill_graph<I>(v, row, g, tuple, arg, entity, ta, plain.out<I>(G), target, va.data(), a.data(), history);
            ed::fill_graph_group<I>(ed::SoloGroup{}, v, row, g, tuple, arg, entity, tb, grouped.out<I>(G), target,
                                    vb.data(), b.data(), history);
        }
        else
        {
            ed::fill_graph_positioned<I>(v, row, g, entity, ta, plain.out<I>(G), positions, target, va.data(), a.data(),
                                         history);
            ed::fill_graph_positioned_group<I>(ed::SoloGroup{}, v, row, g, entity, tb, grouped.out<I>(G), positions,
                                               target, vb.data(), b.data(), history);
        }
    }
    EXPECT_EQ(plain.arrays, expected.arrays);
    EXPECT_EQ(grouped.arrays, expected.arrays);
}

class EncodingLaneCorpus : public ::testing::TestWithParam<u32> {};

TEST_P(EncodingLaneCorpus, EverySupportedOptionEqualsThePlainForms)
{
    auto cases = corpus();
    for (const auto& [domain, problem] : std::vector<std::pair<std::string, std::string>>{
             {"logistics00", "probLOGISTICS-6-1"}, {"visitall", "test_problem"}, {"delivery", "test_problem"},
             {"blocks_flags", "p01"}, {"relay", "p01"}})
        cases.push_back({"additional", table_of(domain, {problem}), true});
    const auto& c = cases[GetParam()];
    const auto batch = make_batch(c.table, 33, c.walk, 13);
    for (u32 flags = 0; flags < 256; ++flags)
        for (u32 goals = 0; goals < 8; ++goals)
            for (u32 targets = 0; targets < 4; ++targets)
            {
                Config config{"options", {}, {}};
                auto& o = config.options;
                o.include_static = (flags & 1) != 0;
                o.include_derived = (flags & 2) != 0;
                o.nullary = (flags & 4) != 0;
                o.equality = (flags & 8) != 0;
                o.predicate_virtual_nodes = (flags & 16) != 0;
                o.include_actions = (flags & 32) != 0;
                o.emit_counts = (flags & 64) != 0;
                o.entity_arrays = (flags & 128) ? encoding::EntityArrays::Always : encoding::EntityArrays::Auto;
                o.goal_derivations = {};
                for (u32 bit = 0; bit < 3; ++bit)
                    if (goals & (1u << bit))
                        o.goal_derivations.insert(static_cast<encoding::GoalDerivation>(bit));
                if (targets & 1) o.targets.insert(encoding::TargetSource::Goal);
                if (targets & 2) o.targets.insert(encoding::TargetSource::Action);
                if (((targets & 1) && !(goals & 1)) || ((targets & 2) && !o.include_actions))
                    continue;
                SCOPED_TRACE(::testing::Message() << c.name << " flags=" << flags << " goals=" << goals << " targets=" << targets);
                for (u32 input = 0; input < (goals ? (batch.derived_words ? 3u : 2u) : 1u); ++input)
                {
                    config.inputs = Inputs{true, input > 0, input > 1};
                    for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                    {
                        compare<i32>(batch, config, layout);
                        compare<i64>(batch, config, layout);
                    }
                }
            }
}
INSTANTIATE_TEST_SUITE_P(Corpus, EncodingLaneCorpus, ::testing::Range(u32{0}, u32{11}));

TEST(EncodingLaneGroups, SubgoalLevelsAndHistoryEqualThePlainForms)
{
    for (const auto& c : corpus())
    {
        SCOPED_TRACE(c.name);
        auto batch = make_batch(c.table, 33, c.walk, 13);
        add_layered_rows(batch, 3, 4, 14);
        for (const Config& config : layered_configs())
        {
            SCOPED_TRACE(config.name);
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            {
                compare<i32>(batch, config, layout);
                compare<i64>(batch, config, layout);
            }
            if (HasFatalFailure())
                return;
        }
    }
}
}

namespace
{
TEST(EncodingLaneGroups, GoalStridesRejectOverlapAndAllowBroadcast)
{
    const auto table = table_of("philosophers", {"p03-phil4"});
    const auto batch = make_batch(table, 33, true, 101);
    const auto config = configs().back();
    const auto encoder = encoding::FlatEncoder::create(table, config.options);
    auto in = host_input(batch, 33, config.inputs);
    ASSERT_GT(batch.words, 0u);
    ASSERT_GT(batch.derived_words, 0u);
    ++in.goal_words;
    EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    --in.goal_words;
    ++in.goal_derived_words;
    EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    --in.goal_derived_words;
    in.goal_stride = in.goal_derived_stride = 0;
    EXPECT_NO_THROW((void)encoder->encode(in));
    ThreadPool pool(1);
    encoding::EncodeScratch scratch;
    encoding::FlatSizes sizes;
    EXPECT_THROW((void)encoder->encode_capacity(in, {32, 1000, 1000, 1000}, pool, scratch, sizes), std::invalid_argument);
}
}
