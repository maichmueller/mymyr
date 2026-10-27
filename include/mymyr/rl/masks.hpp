#pragma once
// Action masks and label applicability over state rows, without enumerating ground actions.

#include "mymyr/rl/expand.hpp"

namespace mymyr::rl
{
/// Applicable schemas [N, suite.max_schemas()]. Schemas are local to each row's domain; padding is zero.
void schema_masks(const TaskSuite& suite, StateBatchView rows, const i32* task_ids, u8* out,
                  ThreadPool* pool = nullptr);

/// Objects extending each prefix to an applicable full binding [N, num_objects]. Prefix rows have stride entries;
/// stride >= depth, num_objects >= suite.max_objects(). Invalid schemas, prefix objects and depths yield zero rows.
void parameter_masks(const TaskSuite& suite, StateBatchView rows, const i32* task_ids, const i32* schema,
                     const i32* prefix, u32 stride, u32 depth, u8* out, u32 num_objects, ThreadPool* pool = nullptr);

/// Full-label applicability [N]. Binding rows have label_width entries, at least each selected schema's arity;
/// narrower rows throw std::invalid_argument. Invalid schemas and objects yield false. Entries past the arity
/// are padding and ignored.
void is_applicable(const TaskSuite& suite, StateBatchView rows, const i32* task_ids, const i32* schema,
                   const i32* binding, u32 label_width, u8* out, ThreadPool* pool = nullptr);
}  // namespace mymyr::rl
