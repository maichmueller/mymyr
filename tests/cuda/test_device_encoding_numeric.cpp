#include "../cpp/encoding/support.hpp"
#include "mymyr/cuda/encoding.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
std::vector<rl::TaskTablePtr> numeric_tables()
{
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR);
    std::vector<rl::TaskTablePtr> tables;
    for (const auto storage : {TaskOptions::NumericStorageMode::Auto, TaskOptions::NumericStorageMode::F64})
    {
        TaskOptions options;
        options.numeric_storage = storage;
        std::vector<TaskPtr> tasks;
        for (const char* problem : {"p01", "p02"})
            tasks.push_back(Task::create(*frontend::load_task(dir + "/pddl/counters/domain.pddl",
                                                             dir + "/pddl/counters/" + problem + ".pddl"), options));
        tables.push_back(rl::TaskTable::create(tasks));
    }
    for (const char* name : {"cs-drone", "m-refuel", "cs-expedition"})
        tables.push_back(rl::TaskTable::single(Task::from_text_file(dir + "/numeric_tasks/" + name + ".txt")));
    return tables;
}

TEST(DeviceEncodingNumeric, AllArraysMatchHostForBothStoragesLayoutsAndWidths)
{
    if (!cuda::device_count()) GTEST_SKIP() << "no CUDA device";
    cuda::ContextOptions context_options;
    context_options.max_bytes = u64{512} << 20;
    const auto context = cuda::DeviceContext::create(0, context_options);
    const auto stream = context->stream();
    for (const auto& table : numeric_tables())
    {
        test::Rows rows;
        rows.words = table->words();
        rows.numeric_words = table->numeric_words();
        for (u32 i = 0; i < table->size(); ++i)
        {
            const auto part = test::random_walk(*table, i, 17, 71 + i);
            rows.data.insert(rows.data.end(), part.data.begin(), part.data.end());
            rows.task_ids.insert(rows.task_ids.end(), part.task_ids.begin(), part.task_ids.end());
        }
        cuda::DeviceBuffer states(context, rows.data.size() * sizeof(u64), stream);
        cuda::DeviceBuffer ids(context, rows.task_ids.size() * sizeof(i32), stream);
        cuda::check(cudaMemcpyAsync(states.data(), rows.data.data(), states.size(), cudaMemcpyHostToDevice, stream), "rows");
        cuda::check(cudaMemcpyAsync(ids.data(), rows.task_ids.data(), ids.size(), cudaMemcpyHostToDevice, stream), "ids");
        cuda::DeviceEncodeInput input;
        input.graphs = rows.size();
        input.states = static_cast<const u64*>(states.data());
        input.state_words = rows.words;
        input.numeric_words = rows.numeric_words;
        input.state_stride = table->row_words();
        input.task_ids = static_cast<const i32*>(ids.data());
        for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
            for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                for (const auto dtype : {encoding::NumericDtype::F32, encoding::NumericDtype::F64})
                {
                    auto options = test::encoder_options();
                    options.layout = layout;
                    options.index = index;
                    options.numeric_dtype = dtype;
                    const auto host = encoding::FlatEncoder::create(table, options);
                    encoding::EncodeInput host_input;
                    host_input.states = rows.view();
                    host_input.task_ids = rows.task_ids.data();
                    const auto expected = host->encode(host_input);
                    cuda::DeviceEncoder encoder(context, host);
                    const auto shape = encoder.worst_case(rows.size());
                    const auto offsets = encoder.output_layout(shape);
                    cuda::DeviceBuffer block(context, offsets.bytes, stream);
                    auto output = cuda::output_of(offsets, shape, block.data());
                    output.zero_unused = true;
                    const std::map<std::string, u64> fields = {
                        {"task_ids", offsets.task_ids}, {"num_objects", offsets.num_objects},
                        {"entity_offsets", offsets.entity_offsets}, {"entity_role", offsets.entity_role},
                        {"entity_ref", offsets.entity_ref}, {"relation_counts", offsets.relation_counts},
                        {"relation_offsets", offsets.relation_offsets}, {"relation_arg_offsets", offsets.relation_arg_offsets},
                        {"relation_args", offsets.relation_args}, {"numeric_values", offsets.numeric_values}};
                    for (const auto fill : {cuda::FillMode::Thread, cuda::FillMode::Warp})
                    {
                        cuda::LaunchConfig launch;
                        launch.fill = fill;
                        launch.count_lanes = fill == cuda::FillMode::Thread ? 4 : 32;
                        encoder.encode(input, output, stream, launch);
                        std::vector<std::byte> result(block.size());
                        cuda::check(cudaMemcpyAsync(result.data(), block.data(), result.size(), cudaMemcpyDeviceToHost, stream), "batch");
                        cuda::check(cudaStreamSynchronize(stream), "batch");
                        encoding::FlatSizes sizes;
                        std::memcpy(&sizes, result.data() + offsets.sizes, sizeof(sizes));
                        ASSERT_EQ(sizes.overflow, 0u);
                        u64 numeric = 0;
                        std::memcpy(&numeric, result.data() + offsets.numeric_size, sizeof(numeric));
                        EXPECT_EQ(numeric, expected->find("numeric_values")->elements());
                        for (const auto& info : expected->arrays())
                        {
                            const auto found = fields.find(info.name.str());
                            ASSERT_NE(found, fields.end()) << info.name.str();
                            EXPECT_EQ(std::memcmp(result.data() + found->second, expected->data(info),
                                                  info.elements() * rl::dtype_bytes(info.dtype)), 0) << info.name.str();
                        }
                    }
                }
    }
}

TEST(DeviceEncodingNumeric, SimpleWithNumericRelationsIsRefusedBeforeUpload)
{
    auto options = test::encoder_options();
    options.outputs = {encoding::Output::Relations, encoding::Output::Simple};
    const auto schema = encoding::EncodingSchema::create(*numeric_tables().front(), options);
    EXPECT_THROW(cuda::DeviceEncoder::check_schema(*schema), std::invalid_argument);
}
}
