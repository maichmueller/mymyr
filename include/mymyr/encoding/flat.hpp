#pragma once

#include "mymyr/rl/task_table.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/successor/conditions.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr::encoding
{
inline constexpr u32 k_flat_version = 1;

struct FlatOptions
{
    bool include_static = true;
    bool include_derived = true;
};

struct FlatSchema
{
    u32 version = k_flat_version;
    u32 domain_predicates = 0;
    std::vector<std::string> names, sources;
    std::vector<i32> arities, predicates, schemas, kinds, instances, polarities;
};

struct ActionBatchView
{
    const i32* offsets = nullptr;
    const i32* schema = nullptr;
    const i32* binding = nullptr;
    u64 size = 0;
    u32 label_width = 0;
};

struct FlatBatch
{
    std::shared_ptr<const FlatSchema> schema;
    std::shared_ptr<const rl::ArrayBundle> arrays;
    std::vector<std::string> object_names;
    std::vector<u64> task_fingerprints;
    u32 graphs = 0;
};

class FlatRelationEncoder
{
public:
    explicit FlatRelationEncoder(rl::TaskTablePtr table, FlatOptions options = {});
    explicit FlatRelationEncoder(TaskPtr task, FlatOptions options = {});

    [[nodiscard]] const rl::TaskTable& table() const noexcept { return *m_table; }
    [[nodiscard]] const std::shared_ptr<const FlatSchema>& schema() const noexcept { return m_schema; }
    [[nodiscard]] FlatOptions options() const noexcept { return m_options; }
    [[nodiscard]] FlatBatch encode(rl::StateBatchView states, const i32* task_ids = nullptr,
                                  ActionBatchView actions = {},
                                  std::span<const GroundCondition* const> goals = {}) const;

private:
    rl::TaskTablePtr m_table;
    FlatOptions m_options;
    std::shared_ptr<const FlatSchema> m_schema;
    std::vector<GroundCondition> m_goals;
};
}
